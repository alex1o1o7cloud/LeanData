import Mathlib

namespace alcohol_dilution_l466_46619

theorem alcohol_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 15 → 
  initial_concentration = 0.6 → 
  final_concentration = 0.4 → 
  water_added = 7.5 → 
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

#check alcohol_dilution

end alcohol_dilution_l466_46619


namespace normal_symmetric_probability_l466_46699

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- The cumulative distribution function for a normal random variable -/
noncomputable def normalCDF (ξ : NormalRV) (x : ℝ) : ℝ := sorry

theorem normal_symmetric_probability (ξ : NormalRV) (h : ξ.μ = 2) 
  (h_cdf : normalCDF ξ 4 = 0.8) : 
  normalCDF ξ 2 - normalCDF ξ 0 = 0.3 := by sorry

end normal_symmetric_probability_l466_46699


namespace positive_rationals_characterization_l466_46625

theorem positive_rationals_characterization (M : Set ℚ) (h_nonempty : Set.Nonempty M) :
  (∀ a b : ℚ, a ∈ M → b ∈ M → (a + b) ∈ M ∧ (a * b) ∈ M) →
  (∀ r : ℚ, (r ∈ M ∧ -r ∉ M ∧ r ≠ 0) ∨ (-r ∈ M ∧ r ∉ M ∧ r ≠ 0) ∨ (r ∉ M ∧ -r ∉ M ∧ r = 0)) →
  M = {x : ℚ | x > 0} :=
by sorry

end positive_rationals_characterization_l466_46625


namespace motorcycle_distance_l466_46670

theorem motorcycle_distance (bus_speed : ℝ) (motorcycle_speed_ratio : ℝ) (time : ℝ) :
  bus_speed = 90 →
  motorcycle_speed_ratio = 2 / 3 →
  time = 1 / 2 →
  motorcycle_speed_ratio * bus_speed * time = 30 := by
  sorry

end motorcycle_distance_l466_46670


namespace sum_inequality_l466_46664

theorem sum_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end sum_inequality_l466_46664


namespace raft_travel_time_l466_46653

/-- Given a distance between two docks, prove that if a motor ship travels this distance downstream
    in 5 hours and upstream in 6 hours, then a raft traveling at the speed of the current will take
    60 hours to cover the same distance downstream. -/
theorem raft_travel_time (s : ℝ) (h_s : s > 0) : 
  (∃ (v_s v_c : ℝ), v_s > v_c ∧ v_c > 0 ∧ s / (v_s + v_c) = 5 ∧ s / (v_s - v_c) = 6) →
  s / v_c = 60 :=
by sorry

end raft_travel_time_l466_46653


namespace tan_sum_l466_46682

theorem tan_sum (x y : Real) 
  (h1 : Real.sin x + Real.sin y = 15/13)
  (h2 : Real.cos x + Real.cos y = 5/13)
  (h3 : Real.cos (x - y) = 4/5) :
  Real.tan x + Real.tan y = -3/5 := by sorry

end tan_sum_l466_46682


namespace water_added_for_nine_percent_solution_l466_46651

/-- Represents a solution of alcohol and water -/
structure Solution where
  volume : ℝ
  alcohol_percentage : ℝ

/-- Calculates the amount of alcohol in a solution -/
def alcohol_amount (s : Solution) : ℝ :=
  s.volume * s.alcohol_percentage

/-- The initial solution -/
def initial_solution : Solution :=
  { volume := 40, alcohol_percentage := 0.05 }

/-- The amount of alcohol added -/
def added_alcohol : ℝ := 2.5

/-- Theorem stating the condition for the final solution to be 9% alcohol -/
theorem water_added_for_nine_percent_solution (x : ℝ) : 
  let final_solution : Solution := 
    { volume := initial_solution.volume + added_alcohol + x,
      alcohol_percentage := 0.09 }
  alcohol_amount final_solution = alcohol_amount initial_solution + added_alcohol ↔ x = 7.5 :=
sorry

end water_added_for_nine_percent_solution_l466_46651


namespace mia_egg_decoration_rate_l466_46622

/-- Mia's egg decoration problem -/
theorem mia_egg_decoration_rate
  (billy_rate : ℕ)
  (total_eggs : ℕ)
  (total_time : ℕ)
  (h1 : billy_rate = 10)
  (h2 : total_eggs = 170)
  (h3 : total_time = 5)
  : ∃ (mia_rate : ℕ), mia_rate = 24 ∧ mia_rate + billy_rate = total_eggs / total_time :=
by sorry

end mia_egg_decoration_rate_l466_46622


namespace average_buying_cost_theorem_ritas_average_buying_cost_l466_46649

/-- Represents the cost and quantity of an item --/
structure Item where
  quantity : ℕ
  totalCost : ℚ

/-- Calculates the average buying cost per unit across all items --/
def averageBuyingCost (items : List Item) : ℚ :=
  let totalCost := items.map (λ i => i.totalCost) |>.sum
  let totalQuantity := items.map (λ i => i.quantity) |>.sum
  totalCost / totalQuantity

/-- The main theorem stating that the average buying cost is equal to the total cost divided by total quantity --/
theorem average_buying_cost_theorem (itemA itemB itemC : Item) :
  let items := [itemA, itemB, itemC]
  averageBuyingCost items = (itemA.totalCost + itemB.totalCost + itemC.totalCost) / (itemA.quantity + itemB.quantity + itemC.quantity) := by
  sorry

/-- Application of the theorem to Rita's specific case --/
theorem ritas_average_buying_cost :
  let itemA : Item := { quantity := 20, totalCost := 500 }
  let itemB : Item := { quantity := 15, totalCost := 700 }
  let itemC : Item := { quantity := 10, totalCost := 400 }
  let items := [itemA, itemB, itemC]
  averageBuyingCost items = 1600 / 45 := by
  sorry

end average_buying_cost_theorem_ritas_average_buying_cost_l466_46649


namespace intersection_complement_equality_l466_46608

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | |x| < 2}
def N : Set ℝ := {y | ∃ x, y = 2^x - 1}

theorem intersection_complement_equality :
  M ∩ (U \ N) = Ioc (-2) (-1) :=
sorry

end intersection_complement_equality_l466_46608


namespace f_negative_a_eq_zero_l466_46617

/-- Given a function f(x) = x³ + sin(x) + 1 where x ∈ ℝ, 
    if f(a) = 2 for some a ∈ ℝ, then f(-a) = 0 -/
theorem f_negative_a_eq_zero (a : ℝ) : 
  (fun x : ℝ => x^3 + Real.sin x + 1) a = 2 → 
  (fun x : ℝ => x^3 + Real.sin x + 1) (-a) = 0 := by sorry

end f_negative_a_eq_zero_l466_46617


namespace sqrt_two_not_periodic_l466_46691

-- Define a property for numbers with periodic decimal expansions
def has_periodic_decimal_expansion (x : ℝ) : Prop := sorry

-- State that numbers with periodic decimal expansions are rational
axiom periodic_is_rational : ∀ x : ℝ, has_periodic_decimal_expansion x → ∃ q : ℚ, x = q

-- State that √2 is irrational
axiom sqrt_two_irrational : ∀ q : ℚ, q * q ≠ 2

-- Theorem: √2 does not have a periodic decimal expansion
theorem sqrt_two_not_periodic : ¬ has_periodic_decimal_expansion (Real.sqrt 2) := by
  sorry

end sqrt_two_not_periodic_l466_46691


namespace sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l466_46636

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (6 + Real.sqrt (36 - 32)) / 2
  let r₂ := (6 - Real.sqrt (36 - 32)) / 2
  r₁ + r₂ = 6 := by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l466_46636


namespace difference_value_l466_46690

theorem difference_value (x : ℝ) (h : x = -10) : 2 * x - (-8) = -12 := by
  sorry

end difference_value_l466_46690


namespace total_average_marks_l466_46610

theorem total_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 ∧ n2 > 0 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = 
    ((n1 : ℚ) + (n2 : ℚ)) * ((n1 : ℚ) * avg1 + (n2 : ℚ) * avg2) / ((n1 : ℚ) + (n2 : ℚ)) :=
by
  sorry

#eval ((45 : ℚ) * 39 + (70 : ℚ) * 35) / ((39 : ℚ) + (35 : ℚ))

end total_average_marks_l466_46610


namespace plot_width_calculation_l466_46678

/-- Calculates the width of a rectangular plot given tilling parameters -/
theorem plot_width_calculation (plot_length : ℝ) (tilling_time : ℝ) (tiller_width : ℝ) (tilling_rate : ℝ) : 
  plot_length = 120 →
  tilling_time = 220 →
  tiller_width = 2 →
  tilling_rate = 1 / 2 →
  (tilling_time * 60 * tilling_rate * tiller_width) / plot_length = 110 := by
  sorry

end plot_width_calculation_l466_46678


namespace new_quadratic_from_original_l466_46674

theorem new_quadratic_from_original (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 9 * x₁ + 8 = 0) ∧ (2 * x₂^2 - 9 * x₂ + 8 = 0) →
  ∃ y₁ y₂ : ℝ, 
    (36 * y₁^2 - 161 * y₁ + 34 = 0) ∧
    (36 * y₂^2 - 161 * y₂ + 34 = 0) ∧
    (y₁ = 1 / (x₁ + x₂) ∨ y₁ = (x₁ - x₂)^2) ∧
    (y₂ = 1 / (x₁ + x₂) ∨ y₂ = (x₁ - x₂)^2) ∧
    y₁ ≠ y₂ :=
by sorry

end new_quadratic_from_original_l466_46674


namespace polygon_with_120_degree_interior_angles_has_6_sides_l466_46659

theorem polygon_with_120_degree_interior_angles_has_6_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    interior_angle = 120 →
    (n : ℝ) * (180 - interior_angle) = 360 →
    n = 6 := by
  sorry

end polygon_with_120_degree_interior_angles_has_6_sides_l466_46659


namespace kendall_correlation_coefficient_l466_46612

def scores_A : List ℝ := [95, 90, 86, 84, 75, 70, 62, 60, 57, 50]
def scores_B : List ℝ := [92, 93, 83, 80, 55, 60, 45, 72, 62, 70]

def kendall_tau_b (x y : List ℝ) : ℝ := sorry

theorem kendall_correlation_coefficient :
  kendall_tau_b scores_A scores_B = 0.51 := by sorry

end kendall_correlation_coefficient_l466_46612


namespace circle_properties_l466_46695

theorem circle_properties (k : ℚ) : 
  let circle_eq (x y : ℚ) := x^2 + 2*x + y^2 = 1992
  ∃ (x y : ℚ), 
    circle_eq 42 12 ∧ 
    circle_eq x y ∧ 
    y - 12 = k * (x - 42) := by
  sorry

end circle_properties_l466_46695


namespace p_sufficient_not_necessary_for_q_l466_46623

theorem p_sufficient_not_necessary_for_q :
  ∃ (x y : ℝ), 
    (((x - 2) * (y - 5) ≠ 0) → (x ≠ 2 ∨ y ≠ 5)) ∧
    ¬(((x ≠ 2 ∨ y ≠ 5) → ((x - 2) * (y - 5) ≠ 0))) :=
by sorry

end p_sufficient_not_necessary_for_q_l466_46623


namespace fraction_denominator_l466_46698

theorem fraction_denominator (x : ℚ) : 
  (525 : ℚ) / x = (21 : ℚ) / 40 →
  (∃ n : ℕ, n ≥ 81 ∧ (525 : ℚ) / x - ((525 : ℚ) / x).floor = 5 / (10 ^ n)) →
  x = 40 := by
sorry

end fraction_denominator_l466_46698


namespace eleventh_term_is_192_l466_46614

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The 5th term of the sequence is 3 -/
def FifthTerm (a : ℕ → ℝ) : Prop := a 5 = 3

/-- The 8th term of the sequence is 24 -/
def EighthTerm (a : ℕ → ℝ) : Prop := a 8 = 24

theorem eleventh_term_is_192 (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_fifth : FifthTerm a) 
  (h_eighth : EighthTerm a) : 
  a 11 = 192 := by
  sorry

end eleventh_term_is_192_l466_46614


namespace ellipse_equation_l466_46645

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let C : Set (ℝ × ℝ) := {(x, y) | x^2/a^2 + y^2/b^2 = 1}
  let e := (Real.sqrt (a^2 - b^2)) / a
  (0, 4) ∈ C ∧ e = 3/5 → C = {(x, y) | x^2/25 + y^2/16 = 1} := by
  sorry

end ellipse_equation_l466_46645


namespace S_intersect_T_eq_T_l466_46627

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end S_intersect_T_eq_T_l466_46627


namespace mary_donated_books_l466_46624

def books_donated (initial_books : ℕ) (monthly_books : ℕ) (bookstore_books : ℕ) 
  (yard_sale_books : ℕ) (daughter_books : ℕ) (mother_books : ℕ) 
  (sold_books : ℕ) (final_books : ℕ) : ℕ :=
  initial_books + (monthly_books * 12) + bookstore_books + yard_sale_books + 
  daughter_books + mother_books - sold_books - final_books

theorem mary_donated_books : 
  books_donated 72 1 5 2 1 4 3 81 = 12 := by
  sorry

end mary_donated_books_l466_46624


namespace franks_age_l466_46638

theorem franks_age (frank_age : ℕ) (gabriel_age : ℕ) : 
  gabriel_age = frank_age - 3 →
  frank_age + gabriel_age = 17 →
  frank_age = 10 :=
by sorry

end franks_age_l466_46638


namespace sqrt_eight_minus_sqrt_two_equals_sqrt_two_l466_46663

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two :
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt_eight_minus_sqrt_two_equals_sqrt_two_l466_46663


namespace min_sum_of_reciprocal_sum_l466_46648

theorem min_sum_of_reciprocal_sum (x y : ℝ) : 
  x > 0 → y > 0 → (1 / (x + 2) + 1 / (y + 2) = 1 / 6) → 
  ∀ a b : ℝ, a > 0 → b > 0 → (1 / (a + 2) + 1 / (b + 2) = 1 / 6) → 
  x + y ≤ a + b ∧ x + y ≥ 20 := by
sorry

end min_sum_of_reciprocal_sum_l466_46648


namespace cuboid_diagonal_l466_46600

/-- Given a cuboid with dimensions a, b, and c (in cm), prove that if its
    total surface area is 20 cm² and the sum of all edge lengths is 24 cm,
    then the length of its diagonal is 4 cm. -/
theorem cuboid_diagonal (a b c : ℝ) : 
  (2 * (a * b + b * c + a * c) = 20) →
  (4 * (a + b + c) = 24) →
  Real.sqrt (a^2 + b^2 + c^2) = 4 := by
  sorry

end cuboid_diagonal_l466_46600


namespace sqrt_two_three_power_l466_46689

/-- Given that (√2 + √3)^(2n-1) = aₙ√2 + bₙ√3 and aₙ₊₁ = paₙ + qbₙ for n ∈ ℕ₊,
    prove that p + q = 11 and 2aₙ² - 3bₙ² = -1 -/
theorem sqrt_two_three_power (n : ℕ) (hn : n > 0) 
  (a b : ℕ → ℝ) (p q : ℝ)
  (h1 : ∀ n, (Real.sqrt 2 + Real.sqrt 3) ^ (2 * n - 1) = a n * Real.sqrt 2 + b n * Real.sqrt 3)
  (h2 : ∀ n, a (n + 1) = p * a n + q * b n) :
  p + q = 11 ∧ ∀ n, 2 * (a n)^2 - 3 * (b n)^2 = -1 := by
  sorry

end sqrt_two_three_power_l466_46689


namespace flyers_made_total_l466_46644

/-- The total number of flyers made by Jack and Rose for their dog-walking business -/
def total_flyers (jack_handed_out : ℕ) (rose_handed_out : ℕ) (flyers_left : ℕ) : ℕ :=
  jack_handed_out + rose_handed_out + flyers_left

/-- Theorem stating the total number of flyers made by Jack and Rose -/
theorem flyers_made_total :
  total_flyers 120 320 796 = 1236 := by
  sorry

end flyers_made_total_l466_46644


namespace inequality_one_integer_solution_l466_46629

theorem inequality_one_integer_solution (a : ℝ) :
  (∃! (x : ℤ), 2 * a * (x : ℝ)^2 - 4 * (x : ℝ) < a * (x : ℝ) - 2) ↔ 1 ≤ a ∧ a < 2 := by
  sorry

end inequality_one_integer_solution_l466_46629


namespace expression_equality_l466_46669

theorem expression_equality : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end expression_equality_l466_46669


namespace problem_statement_l466_46657

theorem problem_statement (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end problem_statement_l466_46657


namespace gravitational_force_at_satellite_orbit_l466_46601

/-- Gravitational force calculation -/
theorem gravitational_force_at_satellite_orbit 
  (surface_distance : ℝ) 
  (surface_force : ℝ) 
  (satellite_distance : ℝ) 
  (h1 : surface_distance = 6400)
  (h2 : surface_force = 800)
  (h3 : satellite_distance = 384000)
  (h4 : ∀ (d f : ℝ), f * d^2 = surface_force * surface_distance^2) :
  ∃ (satellite_force : ℝ), 
    satellite_force * satellite_distance^2 = surface_force * surface_distance^2 ∧ 
    satellite_force = 2/9 := by
  sorry


end gravitational_force_at_satellite_orbit_l466_46601


namespace shaded_area_square_with_circles_l466_46696

/-- The area of the shaded region in a square with circles at its vertices -/
theorem shaded_area_square_with_circles (s : ℝ) (r : ℝ) (h_s : s = 8) (h_r : r = 3) :
  s^2 - 4 * Real.pi * r^2 = 64 - 36 * Real.pi := by
  sorry

end shaded_area_square_with_circles_l466_46696


namespace distance_on_foot_for_given_journey_l466_46605

/-- A journey with two modes of transportation -/
structure Journey where
  total_distance : ℝ
  total_time : ℝ
  foot_speed : ℝ
  bicycle_speed : ℝ

/-- The distance traveled on foot for a given journey -/
def distance_on_foot (j : Journey) : ℝ :=
  -- Define this as a real number, but don't provide the actual calculation
  sorry

/-- Theorem stating the distance traveled on foot for the specific journey -/
theorem distance_on_foot_for_given_journey :
  let j : Journey := {
    total_distance := 80,
    total_time := 7,
    foot_speed := 8,
    bicycle_speed := 16
  }
  distance_on_foot j = 32 := by
  sorry

end distance_on_foot_for_given_journey_l466_46605


namespace tree_leaf_drop_l466_46679

theorem tree_leaf_drop (initial_leaves : ℕ) (final_drop : ℕ) : 
  initial_leaves = 340 → 
  final_drop = 204 → 
  ∃ (n : ℕ), n = 4 ∧ 
    initial_leaves * (9/10)^n = final_drop ∧ 
    ∀ (k : ℕ), k < n → initial_leaves * (9/10)^k > final_drop :=
by sorry

end tree_leaf_drop_l466_46679


namespace sum_C_D_equals_28_l466_46672

theorem sum_C_D_equals_28 (C D : ℝ) :
  (∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-4 * x^2 + 18 * x + 32) / (x - 3)) →
  C + D = 28 := by
sorry

end sum_C_D_equals_28_l466_46672


namespace holly_fence_length_l466_46692

/-- The length of Holly's fence in yards -/
def fence_length_yards : ℚ := 25

/-- The cost of trees to cover the fence -/
def total_cost : ℚ := 400

/-- The cost of each tree -/
def tree_cost : ℚ := 8

/-- The width of each tree in feet -/
def tree_width_feet : ℚ := 1.5

/-- The number of feet in a yard -/
def feet_per_yard : ℚ := 3

theorem holly_fence_length :
  fence_length_yards * feet_per_yard = (total_cost / tree_cost) * tree_width_feet :=
by sorry

end holly_fence_length_l466_46692


namespace range_of_b_l466_46603

theorem range_of_b (y b : ℝ) (h1 : b > 1) (h2 : |y - 2| + |y - 5| < b) : b > 3 := by
  sorry

end range_of_b_l466_46603


namespace luke_carries_four_trays_l466_46632

/-- The number of trays Luke can carry at a time -/
def trays_per_trip (trays_table1 trays_table2 total_trips : ℕ) : ℕ :=
  (trays_table1 + trays_table2) / total_trips

/-- Theorem stating that Luke can carry 4 trays at a time -/
theorem luke_carries_four_trays :
  trays_per_trip 20 16 9 = 4 := by
  sorry

end luke_carries_four_trays_l466_46632


namespace octagon_diagonal_relation_l466_46628

/-- In a regular octagon, a is the side length, b is the diagonal spanning two sides, and d is the diagonal spanning three sides. -/
structure RegularOctagon where
  a : ℝ
  b : ℝ
  d : ℝ
  a_pos : 0 < a

/-- The relation between side length and diagonals in a regular octagon -/
theorem octagon_diagonal_relation (oct : RegularOctagon) : oct.d^2 = oct.a^2 + oct.b^2 := by
  sorry

end octagon_diagonal_relation_l466_46628


namespace total_birds_count_l466_46621

/-- The number of blackbirds in each tree -/
def blackbirds_per_tree : ℕ := 3

/-- The number of trees in the park -/
def number_of_trees : ℕ := 7

/-- The number of magpies in the park -/
def number_of_magpies : ℕ := 13

/-- The total number of birds in the park -/
def total_birds : ℕ := blackbirds_per_tree * number_of_trees + number_of_magpies

theorem total_birds_count : total_birds = 34 := by
  sorry

end total_birds_count_l466_46621


namespace valid_parameterization_l466_46693

/-- A vector parameterization of a line --/
structure VectorParam where
  x0 : ℝ
  y0 : ℝ
  dx : ℝ
  dy : ℝ

/-- The line y = 2x + 4 --/
def line (x : ℝ) : ℝ := 2 * x + 4

/-- Checks if a vector is a scalar multiple of (2, 1) --/
def isValidDirection (dx dy : ℝ) : Prop := ∃ (k : ℝ), dx = 2 * k ∧ dy = k

/-- Checks if a point (x0, y0) lies on the line y = 2x + 4 --/
def isOnLine (x0 y0 : ℝ) : Prop := y0 = line x0

/-- Theorem: A vector parameterization is valid iff its direction is a scalar multiple of (2, 1) and its initial point lies on the line --/
theorem valid_parameterization (p : VectorParam) : 
  (isValidDirection p.dx p.dy ∧ isOnLine p.x0 p.y0) ↔ 
  (∀ t : ℝ, line (p.x0 + t * p.dx) = p.y0 + t * p.dy) :=
sorry

end valid_parameterization_l466_46693


namespace alexs_class_size_l466_46630

theorem alexs_class_size :
  ∃! b : ℕ, 100 < b ∧ b < 150 ∧
  (∃ k : ℕ, b = 4 * k - 2) ∧
  (∃ m : ℕ, b = 5 * m - 3) ∧
  (∃ n : ℕ, b = 6 * n - 4) := by
sorry

end alexs_class_size_l466_46630


namespace abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three_l466_46677

theorem abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three :
  (∀ x : ℝ, |x - 1| < 2 → x < 3) ∧
  ¬(∀ x : ℝ, x < 3 → |x - 1| < 2) := by
sorry

end abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three_l466_46677


namespace max_volume_box_l466_46688

/-- The volume of a lidless box formed from a rectangular sheet with corners cut out -/
def boxVolume (sheetLength sheetWidth squareSide : ℝ) : ℝ :=
  (sheetLength - 2 * squareSide) * (sheetWidth - 2 * squareSide) * squareSide

/-- The theorem stating the maximum volume of the box and the optimal side length of cut squares -/
theorem max_volume_box (sheetLength sheetWidth : ℝ) 
  (hL : sheetLength = 8) (hW : sheetWidth = 5) :
  ∃ (optimalSide maxVolume : ℝ),
    optimalSide = 1 ∧ 
    maxVolume = 18 ∧
    ∀ x, 0 < x → x < sheetWidth / 2 → 
      boxVolume sheetLength sheetWidth x ≤ maxVolume := by
  sorry

end max_volume_box_l466_46688


namespace polynomial_coefficient_theorem_l466_46666

theorem polynomial_coefficient_theorem (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, x^6 = a₀ + a₁*(2+x) + a₂*(2+x)^2 + a₃*(2+x)^3 + a₄*(2+x)^4 + a₅*(2+x)^5 + a₆*(2+x)^6) →
  (a₃ = -160 ∧ a₁ + a₃ + a₅ = -364) :=
by sorry

end polynomial_coefficient_theorem_l466_46666


namespace arithmetic_sequence_sum_l466_46665

def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmeticSequence a →
  (a 1 + a 2 = 3) →
  (a 3 + a 4 = 5) →
  (a 7 + a 8 = 9) := by
sorry

end arithmetic_sequence_sum_l466_46665


namespace combined_distance_of_trains_l466_46633

-- Define the speeds of the trains in km/h
def train_A_speed : ℝ := 120
def train_B_speed : ℝ := 160

-- Define the time in hours (45 minutes = 0.75 hours)
def time : ℝ := 0.75

-- Theorem statement
theorem combined_distance_of_trains :
  train_A_speed * time + train_B_speed * time = 210 := by
  sorry

end combined_distance_of_trains_l466_46633


namespace quadratic_function_property_l466_46604

/-- Given a quadratic function f(x) = x^2 - x + a, if f(-t) < 0, then f(t+1) < 0 -/
theorem quadratic_function_property (a t : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - x + a
  f (-t) < 0 → f (t + 1) < 0 := by
  sorry

end quadratic_function_property_l466_46604


namespace slope_of_line_l466_46640

theorem slope_of_line (x y : ℝ) :
  y = (Real.sqrt 3 / 3) * x - (Real.sqrt 7 / 3) →
  (y - (-(Real.sqrt 7 / 3))) / x = Real.sqrt 3 / 3 :=
by sorry

end slope_of_line_l466_46640


namespace cost_price_per_metre_l466_46656

/-- Calculates the cost price per metre of cloth given the total metres sold,
    total selling price, and loss per metre. -/
theorem cost_price_per_metre
  (total_metres : ℕ)
  (total_selling_price : ℕ)
  (loss_per_metre : ℕ)
  (h1 : total_metres = 600)
  (h2 : total_selling_price = 36000)
  (h3 : loss_per_metre = 10) :
  (total_selling_price + total_metres * loss_per_metre) / total_metres = 70 := by
  sorry

#check cost_price_per_metre

end cost_price_per_metre_l466_46656


namespace james_monthly_earnings_l466_46637

/-- Calculates the monthly earnings of a Twitch streamer based on their subscribers and earnings per subscriber. -/
def monthly_earnings (initial_subscribers : ℕ) (gifted_subscribers : ℕ) (earnings_per_subscriber : ℕ) : ℕ :=
  (initial_subscribers + gifted_subscribers) * earnings_per_subscriber

/-- Theorem stating that James' monthly earnings from Twitch are $1800 -/
theorem james_monthly_earnings :
  monthly_earnings 150 50 9 = 1800 := by
  sorry

end james_monthly_earnings_l466_46637


namespace sqrt_equals_self_l466_46611

theorem sqrt_equals_self (x : ℝ) : Real.sqrt x = x ↔ x = 0 ∨ x = 1 := by
  sorry

end sqrt_equals_self_l466_46611


namespace frog_eats_two_flies_per_day_l466_46647

/-- The number of flies Betty's frog eats per day -/
def frog_daily_flies : ℕ :=
  let current_flies : ℕ := 5 + 6 - 1
  let total_flies : ℕ := current_flies + 4
  total_flies / 7

theorem frog_eats_two_flies_per_day : frog_daily_flies = 2 := by
  sorry

end frog_eats_two_flies_per_day_l466_46647


namespace remainder_of_sum_product_l466_46671

theorem remainder_of_sum_product (p q r s : ℕ) : 
  p < 12 → q < 12 → r < 12 → s < 12 →
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s →
  Nat.gcd p 12 = 1 → Nat.gcd q 12 = 1 → Nat.gcd r 12 = 1 → Nat.gcd s 12 = 1 →
  (p * q + q * r + r * s + s * p) % 12 = 8 := by
sorry

end remainder_of_sum_product_l466_46671


namespace fifteenth_prime_l466_46626

def is_prime (n : ℕ) : Prop := sorry

def nth_prime (n : ℕ) : ℕ := sorry

theorem fifteenth_prime :
  (nth_prime 8 = 19) → (nth_prime 15 = 47) := by
  sorry

end fifteenth_prime_l466_46626


namespace scientific_notation_equivalence_l466_46635

theorem scientific_notation_equivalence : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 3080000 = a * (10 : ℝ) ^ n :=
by
  sorry

end scientific_notation_equivalence_l466_46635


namespace triangle_angles_theorem_l466_46680

/-- A triangle with vertices A, B, and C -/
structure Triangle (α : Type*) :=
  (A B C : α)

/-- The altitudes of a triangle -/
structure Altitudes (α : Type*) :=
  (AA₁ BB₁ CC₁ : α × α)

/-- Similarity of triangles -/
def similar {α : Type*} (t1 t2 : Triangle α) : Prop := sorry

/-- The angles of a triangle -/
def angles {α : Type*} (t : Triangle α) : ℝ × ℝ × ℝ := sorry

theorem triangle_angles_theorem {α : Type*} (ABC : Triangle α) (A₁B₁C₁ : Triangle α) (alt : Altitudes α) :
  (alt.AA₁.1 = ABC.A ∧ alt.BB₁.1 = ABC.B ∧ alt.CC₁.1 = ABC.C) →
  similar ABC A₁B₁C₁ →
  (angles ABC = (60, 60, 60) ∨ angles ABC = (720/7, 360/7, 180/7)) := by
  sorry

end triangle_angles_theorem_l466_46680


namespace goods_train_speed_l466_46616

theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (passing_time : ℝ) 
  (goods_train_length : ℝ) :
  man_train_speed = 100 →
  passing_time = 9 →
  goods_train_length = 280 →
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 12 ∧
    goods_train_length = (man_train_speed + goods_train_speed) * (5/18) * passing_time :=
by sorry

end goods_train_speed_l466_46616


namespace rick_card_distribution_l466_46634

theorem rick_card_distribution (total_cards : ℕ) (kept_cards : ℕ) (miguel_cards : ℕ) 
  (num_friends : ℕ) (num_sisters : ℕ) (sister_cards : ℕ) 
  (h1 : total_cards = 130) 
  (h2 : kept_cards = 15)
  (h3 : miguel_cards = 13)
  (h4 : num_friends = 8)
  (h5 : num_sisters = 2)
  (h6 : sister_cards = 3) :
  (total_cards - kept_cards - miguel_cards - num_sisters * sister_cards) / num_friends = 12 := by
  sorry

end rick_card_distribution_l466_46634


namespace area_code_letters_l466_46662

/-- The number of letters in each area code. -/
def n : ℕ := 2

/-- The total number of signs available. -/
def total_signs : ℕ := 224

/-- The number of signs fully used. -/
def used_signs : ℕ := 222

/-- The number of unused signs. -/
def unused_signs : ℕ := 2

/-- The number of additional area codes created by using all signs. -/
def additional_codes : ℕ := 888

theorem area_code_letters :
  n = 2 ∧
  total_signs = 224 ∧
  used_signs = 222 ∧
  unused_signs = 2 ∧
  additional_codes = 888 ∧
  total_signs ^ n - used_signs ^ n - n * unused_signs = additional_codes :=
sorry

end area_code_letters_l466_46662


namespace stratified_sample_problem_l466_46652

theorem stratified_sample_problem (m : ℕ) : 
  let total_male : ℕ := 56
  let sample_size : ℕ := 28
  let male_in_sample : ℕ := (sample_size + 4) / 2
  let female_in_sample : ℕ := (sample_size - 4) / 2
  (male_in_sample : ℚ) / female_in_sample = (total_male : ℚ) / m →
  m = 42 := by
sorry

end stratified_sample_problem_l466_46652


namespace largest_single_digit_divisor_of_9984_l466_46661

theorem largest_single_digit_divisor_of_9984 : ∃ (d : ℕ), d < 10 ∧ d ∣ 9984 ∧ ∀ (n : ℕ), n < 10 ∧ n ∣ 9984 → n ≤ d :=
by sorry

end largest_single_digit_divisor_of_9984_l466_46661


namespace probability_between_lines_l466_46650

/-- Line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The first quadrant -/
def firstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- Region below a line in the first quadrant -/
def regionBelowLine (l : Line) : Set (ℝ × ℝ) :=
  {p ∈ firstQuadrant | p.2 ≤ l.slope * p.1 + l.yIntercept}

/-- Region between two lines in the first quadrant -/
def regionBetweenLines (l1 l2 : Line) : Set (ℝ × ℝ) :=
  {p ∈ firstQuadrant | l2.slope * p.1 + l2.yIntercept ≤ p.2 ∧ p.2 ≤ l1.slope * p.1 + l1.yIntercept}

/-- Area of a region in the first quadrant -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Probability of selecting a point in a subregion of a given region -/
noncomputable def probability (subregion region : Set (ℝ × ℝ)) : ℝ :=
  area subregion / area region

theorem probability_between_lines :
  let k : Line := ⟨-3, 9⟩
  let n : Line := ⟨-6, 9⟩
  probability (regionBetweenLines k n) (regionBelowLine k) = 0.5 := by
  sorry

end probability_between_lines_l466_46650


namespace condition_relationship_l466_46609

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a / b ≥ 1 → b * (b - a) ≤ 0) ∧
  (∃ a b, b * (b - a) ≤ 0 ∧ ¬(a / b ≥ 1)) :=
sorry

end condition_relationship_l466_46609


namespace fraction_subtraction_simplification_l466_46676

theorem fraction_subtraction_simplification :
  8 / 21 - 3 / 63 = 1 / 3 := by
  sorry

end fraction_subtraction_simplification_l466_46676


namespace star_operation_two_neg_three_l466_46646

def star_operation (a b : ℤ) : ℤ := a * b - (b - 1) * b

theorem star_operation_two_neg_three :
  star_operation 2 (-3) = -18 := by sorry

end star_operation_two_neg_three_l466_46646


namespace base_eight_distinct_digits_l466_46606

/-- The number of four-digit numbers with distinct digits in base b -/
def distinctDigitCount (b : ℕ) : ℕ := (b - 1) * (b - 2) * (b - 3)

/-- Theorem stating that there are exactly 168 four-digit numbers with distinct digits in base 8 -/
theorem base_eight_distinct_digits :
  ∃ (b : ℕ), b > 4 ∧ distinctDigitCount b = 168 ↔ distinctDigitCount 8 = 168 :=
sorry

end base_eight_distinct_digits_l466_46606


namespace complex_power_210_deg_30_l466_46615

theorem complex_power_210_deg_30 :
  (Complex.exp (210 * Real.pi / 180 * Complex.I)) ^ 30 = -1 := by
  sorry

end complex_power_210_deg_30_l466_46615


namespace triangle_angle_measure_l466_46685

theorem triangle_angle_measure (A B C : Real) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h1 : Real.sin B ^ 2 - Real.sin C ^ 2 - Real.sin A ^ 2 = Real.sqrt 3 * Real.sin A * Real.sin C) : 
  B = 5 * π / 6 := by
sorry

end triangle_angle_measure_l466_46685


namespace product_from_hcf_lcm_l466_46655

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 9) (h2 : Nat.lcm a b = 200) :
  a * b = 1800 := by
  sorry

end product_from_hcf_lcm_l466_46655


namespace bicycle_price_reduction_l466_46607

theorem bicycle_price_reduction (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 200 → 
  discount1 = 0.4 → 
  discount2 = 0.25 → 
  (original_price * (1 - discount1) * (1 - discount2)) = 90 := by
sorry

end bicycle_price_reduction_l466_46607


namespace sum_of_120_terms_l466_46631

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  first_term : ℚ
  common_difference : ℚ

/-- Sum of the first n terms of an arithmetic progression. -/
def sum_of_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.first_term + (n - 1 : ℚ) * ap.common_difference)

/-- Theorem stating the sum of the first 120 terms of a specific arithmetic progression. -/
theorem sum_of_120_terms (ap : ArithmeticProgression) 
  (h1 : sum_of_terms ap 15 = 150)
  (h2 : sum_of_terms ap 115 = 5) :
  sum_of_terms ap 120 = -2620 / 77 := by
  sorry

end sum_of_120_terms_l466_46631


namespace minimum_value_of_sum_of_reciprocals_l466_46694

theorem minimum_value_of_sum_of_reciprocals (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
sorry

end minimum_value_of_sum_of_reciprocals_l466_46694


namespace product_of_sums_l466_46639

theorem product_of_sums (x y : ℝ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h1 : x * y = 1 / 9)
  (h2 : x * (y + 1) = 7 / 9)
  (h3 : y * (x + 1) = 5 / 18) :
  (x + 1) * (y + 1) = 35 / 18 := by
sorry

end product_of_sums_l466_46639


namespace perfect_square_trinomial_l466_46641

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 2*(m-1)*x + 16 = (x + a)^2) →
  (m = 5 ∨ m = -3) :=
by sorry

end perfect_square_trinomial_l466_46641


namespace sphere_segment_heights_l466_46675

/-- Given a sphere of radius r intersected by three parallel planes, if the heights
    of the resulting segments form a geometric progression with common ratio q,
    then the height of the first segment (m₁) is equal to 2r(q-1)/(q⁴-1). -/
theorem sphere_segment_heights (r q : ℝ) (h_r : r > 0) (h_q : q > 1) :
  let m₁ := 2 * r * (q - 1) / (q^4 - 1)
  let m₂ := m₁ * q
  let m₃ := m₁ * q^2
  let m₄ := m₁ * q^3
  (m₁ + m₂ + m₃ + m₄ = 2 * r) ∧
  (m₂ / m₁ = q) ∧ (m₃ / m₂ = q) ∧ (m₄ / m₃ = q) :=
by sorry

end sphere_segment_heights_l466_46675


namespace fraction_equality_l466_46643

theorem fraction_equality : 48 / (7 + 3/4) = 192/31 := by
  sorry

end fraction_equality_l466_46643


namespace trapezoid_area_l466_46683

/-- A trapezoid with given base and diagonal lengths has area 80 -/
theorem trapezoid_area (a b d₁ d₂ : ℝ) (ha : a = 5) (hb : b = 15) (hd₁ : d₁ = 12) (hd₂ : d₂ = 16) :
  ∃ h : ℝ, (a + b) * h / 2 = 80 :=
by sorry

end trapezoid_area_l466_46683


namespace negation_implication_geometric_sequence_squared_increasing_l466_46658

-- Proposition 3
theorem negation_implication (P Q : Prop) :
  (¬(P → Q)) ↔ (P ∧ ¬Q) :=
sorry

-- Proposition 4
theorem geometric_sequence_squared_increasing
  (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q) (h2 : q > 1) :
  ∀ n, a (n + 1)^2 > a n^2 :=
sorry

end negation_implication_geometric_sequence_squared_increasing_l466_46658


namespace one_more_stork_than_birds_l466_46668

/-- Given the initial conditions of birds and storks on a fence, prove that there is one more stork than birds. -/
theorem one_more_stork_than_birds : 
  let initial_birds : ℕ := 3
  let additional_birds : ℕ := 2
  let storks : ℕ := 6
  let total_birds : ℕ := initial_birds + additional_birds
  storks - total_birds = 1 := by sorry

end one_more_stork_than_birds_l466_46668


namespace ellipse_properties_l466_46673

/-- Given an ellipse C with specific properties, we prove its equation,
    the range of dot product of vectors OA and OB, and a fixed intersection point. -/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C : Set (ℝ × ℝ) := {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}
  let eccentricity : ℝ := Real.sqrt (1 - b^2/a^2)
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = b^2}
  let tangent_line : Set (ℝ × ℝ) := {p | p.1 - p.2 + Real.sqrt 6 = 0}
  ∃ (A B : ℝ × ℝ),
    (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2/4 + y^2/3 = 1) ∧
    (eccentricity = 1/2) ∧
    (∃ (p : ℝ × ℝ), p ∈ circle ∩ tangent_line) ∧
    (A ∈ C ∧ B ∈ C) ∧
    (∃ (k : ℝ), k ≠ 0 ∧ A.2 - 0 = k * (A.1 - 4) ∧ B.2 - 0 = k * (B.1 - 4)) ∧
    (-4 ≤ (A.1 * B.1 + A.2 * B.2) ∧ (A.1 * B.1 + A.2 * B.2) < 13/4) ∧
    (∃ (E : ℝ × ℝ), E.1 = B.1 ∧ E.2 = -B.2 ∧
      ∃ (t : ℝ), t * A.1 + (1 - t) * E.1 = 1 ∧ t * A.2 + (1 - t) * E.2 = 0) :=
by
  sorry

end ellipse_properties_l466_46673


namespace pens_paid_equals_pens_bought_l466_46681

/-- Represents a retail transaction -/
structure RetailTransaction where
  pens_bought : ℕ
  discount_percent : ℝ
  profit_percent : ℝ

/-- Theorem: The number of pens paid for at market price equals the number of pens bought -/
theorem pens_paid_equals_pens_bought (transaction : RetailTransaction) :
  transaction.pens_bought = transaction.pens_bought := by
  sorry

/-- Example transaction matching the problem -/
def example_transaction : RetailTransaction :=
  { pens_bought := 40
  , discount_percent := 1
  , profit_percent := 9.999999999999996 }

#check pens_paid_equals_pens_bought example_transaction

end pens_paid_equals_pens_bought_l466_46681


namespace symmetry_implies_coordinates_l466_46660

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

theorem symmetry_implies_coordinates : 
  ∀ (a b : ℝ), 
  symmetric_wrt_origin (a, 1) (5, b) → 
  a = -5 ∧ b = -1 := by
sorry

end symmetry_implies_coordinates_l466_46660


namespace parabola_coefficient_l466_46620

/-- Given a parabola y = ax^2 + bx + c with vertex (p, -p) and y-intercept (0, p), where p ≠ 0, 
    the value of b is -4. -/
theorem parabola_coefficient (a b c p : ℝ) : p ≠ 0 → 
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ (x - p)^2 = (y + p) / a) → 
  c = p → 
  b = -4 := by
sorry

end parabola_coefficient_l466_46620


namespace second_term_of_geometric_series_l466_46618

theorem second_term_of_geometric_series
  (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = -1/4) (h2 : S = 40) (h3 : S = a / (1 - r)) :
  a * r = -12.5 :=
by sorry

end second_term_of_geometric_series_l466_46618


namespace investment_profit_ratio_l466_46613

/-- Represents a partner's investment details -/
structure Partner where
  investment : ℕ
  time : ℕ

/-- Calculates the profit ratio of two partners given their investment details -/
def profitRatio (p q : Partner) : Rat :=
  (p.investment * p.time : ℚ) / (q.investment * q.time)

theorem investment_profit_ratio :
  let p : Partner := ⟨7, 5⟩
  let q : Partner := ⟨5, 13⟩
  profitRatio p q = 7 / 13 := by
  sorry

end investment_profit_ratio_l466_46613


namespace specific_pyramid_volume_l466_46654

/-- A pyramid with a square base and known face areas -/
structure Pyramid where
  base_area : ℝ
  face_area1 : ℝ
  face_area2 : ℝ

/-- The volume of the pyramid -/
def pyramid_volume (p : Pyramid) : ℝ := sorry

/-- Theorem: The volume of the specific pyramid is 784 -/
theorem specific_pyramid_volume :
  let p : Pyramid := { base_area := 196, face_area1 := 105, face_area2 := 91 }
  pyramid_volume p = 784 := by sorry

end specific_pyramid_volume_l466_46654


namespace pure_imaginary_condition_l466_46667

theorem pure_imaginary_condition (a : ℝ) : 
  (a^2 - 2*a = 0 ∧ a^2 - a - 2 ≠ 0) → a = 0 :=
by sorry

end pure_imaginary_condition_l466_46667


namespace linear_function_order_l466_46684

/-- For a linear function f(x) = -3x + 5, prove that the y-coordinates
    of the points (1, f(1)), (-1, f(-1)), and (-2, f(-2)) are in ascending order. -/
theorem linear_function_order :
  let f : ℝ → ℝ := λ x ↦ -3 * x + 5
  let x₁ : ℝ := 1
  let x₂ : ℝ := -1
  let x₃ : ℝ := -2
  f x₁ < f x₂ ∧ f x₂ < f x₃ := by sorry

end linear_function_order_l466_46684


namespace total_paths_A_to_D_l466_46602

/-- Number of paths between two points -/
def num_paths (start finish : ℕ) : ℕ := sorry

/-- The problem setup -/
axiom paths_A_to_B : num_paths 0 1 = 2
axiom paths_B_to_C : num_paths 1 2 = 2
axiom paths_A_to_C_direct : num_paths 0 2 = 1
axiom paths_C_to_D : num_paths 2 3 = 2

/-- The theorem to prove -/
theorem total_paths_A_to_D : num_paths 0 3 = 10 := by
  sorry

end total_paths_A_to_D_l466_46602


namespace unique_solution_trig_equation_l466_46642

theorem unique_solution_trig_equation :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ 180 ∧
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 60 := by
  sorry

end unique_solution_trig_equation_l466_46642


namespace divisors_of_900_l466_46687

theorem divisors_of_900 : Finset.card (Nat.divisors 900) = 27 := by
  sorry

end divisors_of_900_l466_46687


namespace range_of_f_l466_46697

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f :
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end range_of_f_l466_46697


namespace isosceles_triangle_perimeter_l466_46686

/-- An isosceles triangle with side lengths 6 and 9 has a perimeter of either 21 or 24 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  (a = 6 ∨ a = 9) →
  (b = 6 ∨ b = 9) →
  (a = b) →
  (c = 6 ∨ c = 9) →
  (a + b + c = 21 ∨ a + b + c = 24) := by
sorry

end isosceles_triangle_perimeter_l466_46686
