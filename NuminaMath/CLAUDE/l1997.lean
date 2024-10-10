import Mathlib

namespace boys_to_girls_ratio_l1997_199795

theorem boys_to_girls_ratio : 
  ∀ (boys girls : ℕ), 
    boys = 80 →
    girls = boys + 128 →
    ∃ (a b : ℕ), a = 5 ∧ b = 13 ∧ a * girls = b * boys :=
by
  sorry

end boys_to_girls_ratio_l1997_199795


namespace cube_properties_l1997_199728

/-- A cube is a convex polyhedron with specific properties -/
structure Cube where
  vertices : ℕ
  faces : ℕ
  edges : ℕ

/-- Euler's formula for convex polyhedrons -/
def euler_formula (c : Cube) : Prop :=
  c.vertices - c.edges + c.faces = 2

/-- Theorem stating the properties of a cube -/
theorem cube_properties : ∃ (c : Cube), c.vertices = 8 ∧ c.faces = 6 ∧ c.edges = 12 ∧ euler_formula c := by
  sorry

end cube_properties_l1997_199728


namespace simplify_expression_l1997_199705

theorem simplify_expression (x : ℝ) : (3*x + 25) + (150*x - 5) + x^2 = x^2 + 153*x + 20 := by
  sorry

end simplify_expression_l1997_199705


namespace price_change_l1997_199710

theorem price_change (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * 1.3
  let final_price := increased_price * 0.75
  final_price = original_price * 0.975 :=
by sorry

end price_change_l1997_199710


namespace field_trip_lunch_cost_l1997_199742

/-- Calculates the total cost of lunches for a field trip. -/
def total_lunch_cost (num_children : ℕ) (num_chaperones : ℕ) (num_teachers : ℕ) (num_extra : ℕ) (cost_per_lunch : ℕ) : ℕ :=
  (num_children + num_chaperones + num_teachers + num_extra) * cost_per_lunch

/-- Proves that the total cost of lunches for the given field trip is $308. -/
theorem field_trip_lunch_cost :
  total_lunch_cost 35 5 1 3 7 = 308 := by
  sorry

end field_trip_lunch_cost_l1997_199742


namespace tan_product_eighths_pi_l1997_199769

theorem tan_product_eighths_pi : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = 1 := by
  sorry

end tan_product_eighths_pi_l1997_199769


namespace soap_discount_theorem_l1997_199719

/-- The original price of a bar of soap in yuan -/
def original_price : ℝ := 2

/-- The discount rate for the first method (applied to all bars except the first) -/
def discount_rate1 : ℝ := 0.3

/-- The discount rate for the second method (applied to all bars) -/
def discount_rate2 : ℝ := 0.2

/-- The cost of n bars using the first discount method -/
def cost1 (n : ℕ) : ℝ := original_price + (n - 1) * original_price * (1 - discount_rate1)

/-- The cost of n bars using the second discount method -/
def cost2 (n : ℕ) : ℝ := n * original_price * (1 - discount_rate2)

/-- The minimum number of bars needed for the first method to provide more discount -/
def min_bars : ℕ := 4

theorem soap_discount_theorem :
  ∀ n : ℕ, n ≥ min_bars → cost1 n < cost2 n ∧
  ∀ m : ℕ, m < min_bars → cost1 m ≥ cost2 m :=
sorry

end soap_discount_theorem_l1997_199719


namespace polynomial_integer_roots_l1997_199745

theorem polynomial_integer_roots (p : ℤ → ℤ) 
  (h1 : ∃ a : ℤ, p a = 1) 
  (h3 : ∃ b : ℤ, p b = 3) : 
  ¬(∃ y1 y2 : ℤ, y1 ≠ y2 ∧ p y1 = 2 ∧ p y2 = 2) :=
by sorry

end polynomial_integer_roots_l1997_199745


namespace at_op_four_nine_l1997_199732

-- Define the operation @
def at_op (a b : ℝ) : ℝ := a * b ^ (1 / 2)

-- Theorem statement
theorem at_op_four_nine : at_op 4 9 = 12 := by
  sorry

end at_op_four_nine_l1997_199732


namespace yellas_computer_usage_l1997_199734

theorem yellas_computer_usage (last_week_hours : ℕ) (reduction : ℕ) : 
  last_week_hours = 91 → 
  reduction = 35 → 
  (last_week_hours - reduction) / 7 = 8 := by
sorry

end yellas_computer_usage_l1997_199734


namespace point_in_first_quadrant_l1997_199787

theorem point_in_first_quadrant (x y : ℝ) : 
  (|3*x - 2*y - 1| + Real.sqrt (x + y - 2) = 0) → (x > 0 ∧ y > 0) := by
  sorry

end point_in_first_quadrant_l1997_199787


namespace janice_stair_climb_l1997_199767

/-- The number of times Janice goes up the stairs in a day. -/
def times_up : ℕ := 5

/-- The number of flights of stairs for each trip up. -/
def flights_per_trip : ℕ := 3

/-- The number of times Janice goes down the stairs in a day. -/
def times_down : ℕ := 3

/-- The total number of flights walked (up and down) in a day. -/
def total_flights : ℕ := 24

theorem janice_stair_climb :
  times_up * flights_per_trip + times_down * flights_per_trip = total_flights :=
by sorry

end janice_stair_climb_l1997_199767


namespace exam_failure_count_l1997_199799

theorem exam_failure_count (total : ℕ) (pass_percentage : ℚ) (fail_count : ℕ) : 
  total = 400 → pass_percentage = 35 / 100 → fail_count = total - (pass_percentage * total).floor → fail_count = 260 := by
  sorry

end exam_failure_count_l1997_199799


namespace solution_set_range_no_k_exists_positive_roots_k_range_l1997_199708

/-- The quadratic function y(x) = kx² - 2kx + 2k - 1 -/
def y (k x : ℝ) : ℝ := k * x^2 - 2 * k * x + 2 * k - 1

/-- The solution set of y ≥ 4k - 2 is all real numbers iff k ∈ [0, 1/3] -/
theorem solution_set_range (k : ℝ) :
  (∀ x, y k x ≥ 4 * k - 2) ↔ k ∈ Set.Icc 0 (1/3) := by sorry

/-- No k ∈ (0, 1) satisfies x₁² + x₂² = 3x₁x₂ - 4 for roots of y(x) = 0 -/
theorem no_k_exists (k : ℝ) (hk : k ∈ Set.Ioo 0 1) :
  ¬∃ x₁ x₂ : ℝ, y k x₁ = 0 ∧ y k x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + x₂^2 = 3*x₁*x₂ - 4 := by sorry

/-- If roots of y(x) = 0 are positive, then k ∈ (1/2, 1) -/
theorem positive_roots_k_range (k : ℝ) :
  (∃ x₁ x₂ : ℝ, y k x₁ = 0 ∧ y k x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0) →
  k ∈ Set.Ioo (1/2) 1 := by sorry

end solution_set_range_no_k_exists_positive_roots_k_range_l1997_199708


namespace number_equation_solution_l1997_199757

theorem number_equation_solution : ∃ x : ℝ, 
  x^(5/4) * 12^(1/4) * 60^(3/4) = 300 ∧ 
  ∀ ε > 0, |x - 6| < ε :=
by sorry

end number_equation_solution_l1997_199757


namespace decreasing_function_implies_a_range_l1997_199743

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(1-a)*x + 2

-- State the theorem
theorem decreasing_function_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Iic 4, ∀ y ∈ Set.Iic 4, x < y → f a x > f a y) →
  a ∈ Set.Ici 5 :=
by sorry

end decreasing_function_implies_a_range_l1997_199743


namespace mod_equivalence_l1997_199738

theorem mod_equivalence (m : ℕ) : 
  198 * 935 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 30 := by
  sorry

end mod_equivalence_l1997_199738


namespace new_persons_joined_l1997_199755

/-- Proves that 20 new persons joined the group given the initial conditions and final average age -/
theorem new_persons_joined (initial_avg : ℝ) (new_avg : ℝ) (final_avg : ℝ) (initial_count : ℕ) : 
  initial_avg = 16 → new_avg = 15 → final_avg = 15.5 → initial_count = 20 → 
  ∃ (new_count : ℕ), 
    (initial_count * initial_avg + new_count * new_avg) / (initial_count + new_count) = final_avg ∧
    new_count = 20 := by
  sorry

end new_persons_joined_l1997_199755


namespace arithmetic_sequence_properties_l1997_199735

/-- An arithmetic sequence with sum of first n terms S_n = 2n^2 - 25n -/
def S (n : ℕ) : ℤ := 2 * n^2 - 25 * n

/-- The nth term of the arithmetic sequence -/
def a (n : ℕ) : ℤ := 4 * n - 27

theorem arithmetic_sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n = 4 * n - 27) ∧
  (∀ n : ℕ, n ≠ 6 → S n > S 6) ∧
  S 6 = -78 := by sorry

end arithmetic_sequence_properties_l1997_199735


namespace intersection_point_unique_l1997_199717

/-- Two lines in a 2D plane --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in a 2D plane --/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def lies_on (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The intersection point of two lines --/
def intersection_point (l1 l2 : Line2D) : Point2D :=
  { x := 1, y := 0 }

theorem intersection_point_unique (l1 l2 : Line2D) :
  l1 = Line2D.mk 1 (-4) (-1) →
  l2 = Line2D.mk 2 1 (-2) →
  let p := intersection_point l1 l2
  lies_on p l1 ∧ lies_on p l2 ∧
  ∀ q : Point2D, lies_on q l1 → lies_on q l2 → q = p :=
by sorry

end intersection_point_unique_l1997_199717


namespace expression_value_l1997_199733

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (sum_zero : x + y + z = 0) (sum_prod_nonzero : x*y + x*z + y*z ≠ 0) :
  (x^6 + y^6 + z^6) / (x*y*z * (x*y + x*z + y*z)) = -6 := by
  sorry

end expression_value_l1997_199733


namespace total_turtles_is_100_l1997_199784

/-- The number of turtles Martha received -/
def martha_turtles : ℕ := 40

/-- The difference between Marion's and Martha's turtles -/
def difference : ℕ := 20

/-- The number of turtles Marion received -/
def marion_turtles : ℕ := martha_turtles + difference

/-- The total number of turtles received by Martha and Marion -/
def total_turtles : ℕ := martha_turtles + marion_turtles

theorem total_turtles_is_100 : total_turtles = 100 := by
  sorry

end total_turtles_is_100_l1997_199784


namespace strawberry_harvest_l1997_199788

/-- Calculates the total number of strawberries harvested from a square garden -/
theorem strawberry_harvest (garden_side : ℝ) (plants_per_sqft : ℝ) (strawberries_per_plant : ℝ) :
  garden_side = 10 →
  plants_per_sqft = 5 →
  strawberries_per_plant = 12 →
  garden_side * garden_side * plants_per_sqft * strawberries_per_plant = 6000 := by
  sorry

#check strawberry_harvest

end strawberry_harvest_l1997_199788


namespace obrien_hats_count_l1997_199753

/-- The number of hats Fire chief Simpson has -/
def simpson_hats : ℕ := 15

/-- The initial number of hats Policeman O'Brien had -/
def obrien_initial_hats : ℕ := 2 * simpson_hats + 5

/-- The number of hats Policeman O'Brien lost -/
def obrien_lost_hats : ℕ := 1

/-- The current number of hats Policeman O'Brien has -/
def obrien_current_hats : ℕ := obrien_initial_hats - obrien_lost_hats

theorem obrien_hats_count : obrien_current_hats = 34 := by
  sorry

end obrien_hats_count_l1997_199753


namespace panda_bamboo_transport_l1997_199736

/-- Represents the maximum number of bamboo sticks that can be transported -/
def max_bamboo_transported (initial_bamboo : ℕ) (capacity : ℕ) (consumption : ℕ) : ℕ :=
  initial_bamboo - consumption * (2 * (initial_bamboo / capacity) - 1)

/-- Theorem stating that the maximum number of bamboo sticks transported is 165 -/
theorem panda_bamboo_transport :
  max_bamboo_transported 200 50 5 = 165 := by
  sorry

end panda_bamboo_transport_l1997_199736


namespace special_operation_l1997_199785

theorem special_operation (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum : a + b = 12) (product : a * b = 35) : 
  (1 : ℚ) / a + (1 : ℚ) / b = 12 / 35 := by
  sorry

end special_operation_l1997_199785


namespace soccer_season_games_l1997_199737

/-- Represents a soccer team's season performance -/
structure SoccerSeason where
  totalGames : ℕ
  firstGames : ℕ
  firstWins : ℕ
  remainingWins : ℕ

/-- Conditions for the soccer season -/
def validSeason (s : SoccerSeason) : Prop :=
  s.totalGames % 2 = 0 ∧ 
  s.firstGames = 36 ∧
  s.firstWins = 16 ∧
  s.remainingWins ≥ (s.totalGames - s.firstGames) * 3 / 4 ∧
  (s.firstWins + s.remainingWins) * 100 = s.totalGames * 62

theorem soccer_season_games (s : SoccerSeason) (h : validSeason s) : s.totalGames = 84 :=
sorry

end soccer_season_games_l1997_199737


namespace bat_pattern_area_l1997_199774

/-- A bat pattern is composed of squares and triangles -/
structure BatPattern where
  large_squares : Nat
  medium_squares : Nat
  triangles : Nat
  large_square_area : ℝ
  medium_square_area : ℝ
  triangle_area : ℝ

/-- The total area of a bat pattern -/
def total_area (b : BatPattern) : ℝ :=
  b.large_squares * b.large_square_area +
  b.medium_squares * b.medium_square_area +
  b.triangles * b.triangle_area

/-- Theorem: The area of the specific bat pattern is 27 -/
theorem bat_pattern_area :
  ∃ (b : BatPattern),
    b.large_squares = 2 ∧
    b.medium_squares = 2 ∧
    b.triangles = 3 ∧
    b.large_square_area = 8 ∧
    b.medium_square_area = 4 ∧
    b.triangle_area = 1 ∧
    total_area b = 27 := by
  sorry

end bat_pattern_area_l1997_199774


namespace cistern_problem_solution_l1997_199729

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width water_breadth : ℝ) : ℝ :=
  let bottom_area := length * width
  let longer_side_area := 2 * length * water_breadth
  let shorter_side_area := 2 * width * water_breadth
  bottom_area + longer_side_area + shorter_side_area

/-- Theorem stating that the wet surface area of the given cistern is 121.5 m² -/
theorem cistern_problem_solution :
  cistern_wet_surface_area 9 6 2.25 = 121.5 := by
  sorry

#eval cistern_wet_surface_area 9 6 2.25

end cistern_problem_solution_l1997_199729


namespace largest_domain_is_plus_minus_one_l1997_199777

def is_valid_domain (S : Set ℝ) : Prop :=
  (∀ x ∈ S, x ≠ 0) ∧ 
  (∀ x ∈ S, (1 / x) ∈ S) ∧
  (∃ g : ℝ → ℝ, ∀ x ∈ S, g x + g (1 / x) = 2 * x)

theorem largest_domain_is_plus_minus_one :
  ∀ S : Set ℝ, is_valid_domain S → S ⊆ {-1, 1} := by sorry

end largest_domain_is_plus_minus_one_l1997_199777


namespace angle_measure_problem_l1997_199775

theorem angle_measure_problem (C D : ℝ) : 
  C + D = 180 →  -- angles are supplementary
  C = 12 * D →   -- C is 12 times D
  C = 2160 / 13  -- measure of angle C
  := by sorry

end angle_measure_problem_l1997_199775


namespace grandmas_will_l1997_199754

theorem grandmas_will (total : ℕ) (shelby_share : ℕ) (other_grandchildren : ℕ) (one_share : ℕ) :
  total = 124600 ∧
  shelby_share = total / 2 ∧
  other_grandchildren = 10 ∧
  one_share = 6230 ∧
  (total - shelby_share) / other_grandchildren = one_share →
  total = 124600 :=
by sorry

end grandmas_will_l1997_199754


namespace math_test_score_distribution_l1997_199759

theorem math_test_score_distribution (total_students : ℕ) (percentile_80_score : ℕ) :
  total_students = 1200 →
  percentile_80_score = 103 →
  (∃ (students_above_threshold : ℕ),
    students_above_threshold ≥ 240 ∧
    students_above_threshold = total_students - (total_students * 80 / 100)) := by
  sorry

end math_test_score_distribution_l1997_199759


namespace line_l_satisfies_conditions_l1997_199702

-- Define the points
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (5, 4)
def C : ℝ × ℝ := (3, 7)
def D : ℝ × ℝ := (7, 1)
def E : ℝ × ℝ := (10, 2)
def F : ℝ × ℝ := (8, 6)

-- Define the line l
def l (x y : ℝ) : Prop := 10 * x - 2 * y - 55 = 0

-- Define the line DF
def DF (x y : ℝ) : Prop := y = 5 * x - 34

-- Define the triangles ABC and DEF
def triangle_ABC : Set (ℝ × ℝ) :=
  {p | ∃ (t₁ t₂ t₃ : ℝ), t₁ ≥ 0 ∧ t₂ ≥ 0 ∧ t₃ ≥ 0 ∧ t₁ + t₂ + t₃ = 1 ∧
    p = (t₁ * A.1 + t₂ * B.1 + t₃ * C.1, t₁ * A.2 + t₂ * B.2 + t₃ * C.2)}

def triangle_DEF : Set (ℝ × ℝ) :=
  {p | ∃ (t₁ t₂ t₃ : ℝ), t₁ ≥ 0 ∧ t₂ ≥ 0 ∧ t₃ ≥ 0 ∧ t₁ + t₂ + t₃ = 1 ∧
    p = (t₁ * D.1 + t₂ * E.1 + t₃ * F.1, t₁ * D.2 + t₂ * E.2 + t₃ * F.2)}

-- Define the distance function
def distance (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  sorry -- Implementation of distance function

theorem line_l_satisfies_conditions :
  (∀ x y, l x y → DF x y) ∧ -- l is parallel to DF
  (∃ d : ℝ, 
    (∀ p ∈ triangle_ABC, distance p l ≥ d) ∧
    (∃ p₁ ∈ triangle_ABC, distance p₁ l = d) ∧
    (∀ p ∈ triangle_DEF, distance p l ≥ d) ∧
    (∃ p₂ ∈ triangle_DEF, distance p₂ l = d)) :=
by sorry

end line_l_satisfies_conditions_l1997_199702


namespace sine_cosine_inequality_l1997_199766

theorem sine_cosine_inequality (a : ℝ) : 
  (∀ x : ℝ, Real.sin x ^ 6 + Real.cos x ^ 6 + 2 * a * Real.sin x * Real.cos x ≥ 0) ↔ 
  |a| ≤ (1/4 : ℝ) := by
  sorry

end sine_cosine_inequality_l1997_199766


namespace equation_solution_l1997_199791

theorem equation_solution : 
  ∃ y : ℝ, (1/8: ℝ)^(3*y+12) = (64 : ℝ)^(y+4) ∧ y = -4 := by
  sorry

end equation_solution_l1997_199791


namespace jessie_muffin_division_l1997_199726

/-- The number of muffins each person receives when 35 muffins are divided equally among Jessie and her friends -/
def muffins_per_person (total_muffins : ℕ) (num_friends : ℕ) : ℕ :=
  total_muffins / (num_friends + 1)

/-- Theorem stating that when 35 muffins are divided equally among Jessie and her 6 friends, each person will receive 5 muffins -/
theorem jessie_muffin_division :
  muffins_per_person 35 6 = 5 := by
  sorry

end jessie_muffin_division_l1997_199726


namespace sawyer_octopus_count_l1997_199716

-- Define the number of legs Sawyer saw
def total_legs : ℕ := 40

-- Define the number of legs each octopus has
def legs_per_octopus : ℕ := 8

-- Theorem statement
theorem sawyer_octopus_count :
  total_legs / legs_per_octopus = 5 := by
  sorry

end sawyer_octopus_count_l1997_199716


namespace roller_coaster_tickets_l1997_199781

/-- Calculates the total number of tickets needed for a group of friends riding roller coasters -/
theorem roller_coaster_tickets (
  first_coaster_cost : ℕ)
  (discount_rate : ℚ)
  (discount_threshold : ℕ)
  (new_coaster_cost : ℕ)
  (num_friends : ℕ)
  (first_coaster_rides : ℕ)
  (new_coaster_rides : ℕ)
  (h1 : first_coaster_cost = 6)
  (h2 : discount_rate = 15 / 100)
  (h3 : discount_threshold = 10)
  (h4 : new_coaster_cost = 8)
  (h5 : num_friends = 8)
  (h6 : first_coaster_rides = 2)
  (h7 : new_coaster_rides = 1)
  : ℕ :=
  160

#check roller_coaster_tickets

end roller_coaster_tickets_l1997_199781


namespace min_value_expression_l1997_199758

theorem min_value_expression (x : ℝ) (hx : x > 0) : 
  (x + 5) / Real.sqrt (x + 1) ≥ 4 ∧ 
  ∃ y : ℝ, y > 0 ∧ (y + 5) / Real.sqrt (y + 1) = 4 := by
sorry

end min_value_expression_l1997_199758


namespace total_rock_is_16_l1997_199741

/-- The amount of rock costing $30 per ton -/
def rock_30 : ℕ := 8

/-- The amount of rock costing $40 per ton -/
def rock_40 : ℕ := 8

/-- The total amount of rock needed -/
def total_rock : ℕ := rock_30 + rock_40

theorem total_rock_is_16 : total_rock = 16 := by
  sorry

end total_rock_is_16_l1997_199741


namespace warehouse_capacity_is_510_l1997_199703

/-- The total capacity of a grain-storage warehouse --/
def warehouse_capacity (total_bins : ℕ) (large_bins : ℕ) (large_capacity : ℕ) (small_capacity : ℕ) : ℕ :=
  large_bins * large_capacity + (total_bins - large_bins) * small_capacity

/-- Theorem: The warehouse capacity is 510 tons --/
theorem warehouse_capacity_is_510 :
  warehouse_capacity 30 12 20 15 = 510 :=
by sorry

end warehouse_capacity_is_510_l1997_199703


namespace binary_sum_theorem_l1997_199756

/-- Converts a binary number (represented as a list of 0s and 1s) to decimal -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

theorem binary_sum_theorem :
  let binary1 := [1, 0, 1, 1, 0, 0, 1]
  let binary2 := [0, 0, 0, 1, 1, 1]
  let binary3 := [0, 1, 0, 1]
  (binary_to_decimal binary1) + (binary_to_decimal binary2) + (binary_to_decimal binary3) = 143 := by
  sorry

end binary_sum_theorem_l1997_199756


namespace school_supplies_cost_l1997_199763

/-- Calculate the total amount spent on school supplies --/
theorem school_supplies_cost 
  (original_backpack_price : ℕ) 
  (original_binder_price : ℕ) 
  (backpack_price_increase : ℕ) 
  (binder_price_decrease : ℕ) 
  (num_binders : ℕ) 
  (h1 : original_backpack_price = 50)
  (h2 : original_binder_price = 20)
  (h3 : backpack_price_increase = 5)
  (h4 : binder_price_decrease = 2)
  (h5 : num_binders = 3) :
  (original_backpack_price + backpack_price_increase) + 
  num_binders * (original_binder_price - binder_price_decrease) = 109 :=
by sorry

end school_supplies_cost_l1997_199763


namespace largest_prime_factor_of_8250_l1997_199776

theorem largest_prime_factor_of_8250 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 8250 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 8250 → q ≤ p :=
by sorry

end largest_prime_factor_of_8250_l1997_199776


namespace solution_set_equality_l1997_199744

-- Define the solution set of |8x+9| < 7
def solution_set : Set ℝ := {x : ℝ | |8*x + 9| < 7}

-- Define the inequality ax^2 + bx > 2
def inequality (a b : ℝ) (x : ℝ) : Prop := a*x^2 + b*x > 2

-- State the theorem
theorem solution_set_equality (a b : ℝ) : 
  (∀ x : ℝ, x ∈ solution_set ↔ inequality a b x) → a + b = -13 := by
  sorry

end solution_set_equality_l1997_199744


namespace inequality_and_range_l1997_199790

-- Define the function f
def f (x : ℝ) : ℝ := |3 * x + 2|

-- Define the theorem
theorem inequality_and_range :
  -- Part I: Solution set of f(x) < 4 - |x-1|
  (∀ x : ℝ, f x < 4 - |x - 1| ↔ x > -5/4 ∧ x < 1/2) ∧
  -- Part II: Range of a
  (∀ m n a : ℝ, m > 0 → n > 0 → m + n = 1 → a > 0 →
    (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) →
    a ≤ 10/3) :=
by sorry

end inequality_and_range_l1997_199790


namespace square_area_from_adjacent_points_l1997_199789

/-- Given two adjacent points of a square at (1,2) and (5,5), the area of the square is 25. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (5, 5)
  let distance := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := distance^2
  area = 25 := by sorry

end square_area_from_adjacent_points_l1997_199789


namespace power_of_power_l1997_199773

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end power_of_power_l1997_199773


namespace purely_imaginary_complex_number_l1997_199778

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, (Complex.mk 2 a) / (Complex.mk 2 (-1)) = Complex.I * b) → a = -4 := by
  sorry

end purely_imaginary_complex_number_l1997_199778


namespace unattainable_y_value_l1997_199727

theorem unattainable_y_value (x y : ℝ) :
  x ≠ -5/4 →
  y = (2 - 3*x) / (4*x + 5) →
  y ≠ -3/4 :=
by sorry

end unattainable_y_value_l1997_199727


namespace no_real_solutions_l1997_199771

theorem no_real_solutions : ¬∃ (x y : ℝ), x^2 + 3*y^2 - 4*x - 6*y + 10 = 0 := by
  sorry

end no_real_solutions_l1997_199771


namespace fraction_value_l1997_199731

theorem fraction_value (a b : ℝ) (h : 1/a - 1/b = 4) :
  (a - 2*a*b - b) / (2*a - 2*b + 7*a*b) = 6 := by
sorry

end fraction_value_l1997_199731


namespace unique_solution_iff_a_eq_half_l1997_199739

/-- The equation has a unique solution if and only if a = 1/2 -/
theorem unique_solution_iff_a_eq_half (a : ℝ) (ha : a > 0) :
  (∃! x : ℝ, 2 * a * x = x^2 - 2 * a * Real.log x) ↔ a = 1 / 2 := by
  sorry

end unique_solution_iff_a_eq_half_l1997_199739


namespace year_2078_is_wu_xu_l1997_199721

/-- Represents the Heavenly Stems in the Chinese calendar system -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches in the Chinese calendar system -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Chinese calendar system -/
structure ChineseYear where
  stem : HeavenlyStem
  branch : EarthlyBranch

/-- The number of Heavenly Stems -/
def numHeavenlyStems : Nat := 10

/-- The number of Earthly Branches -/
def numEarthlyBranches : Nat := 12

/-- The starting year of the reform and opening up period -/
def reformStartYear : Nat := 1978

/-- Function to get the next Heavenly Stem in the cycle -/
def nextHeavenlyStem (s : HeavenlyStem) : HeavenlyStem := sorry

/-- Function to get the next Earthly Branch in the cycle -/
def nextEarthlyBranch (b : EarthlyBranch) : EarthlyBranch := sorry

/-- Function to get the Chinese Year representation for a given year -/
def getChineseYear (year : Nat) : ChineseYear := sorry

/-- Theorem stating that the year 2078 is represented as "Wu Xu" -/
theorem year_2078_is_wu_xu :
  let year2016 := ChineseYear.mk HeavenlyStem.Bing EarthlyBranch.Shen
  let year2078 := getChineseYear 2078
  year2078 = ChineseYear.mk HeavenlyStem.Wu EarthlyBranch.Xu := by
  sorry

end year_2078_is_wu_xu_l1997_199721


namespace collinear_points_x_value_l1997_199783

/-- Three points in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point2D) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

theorem collinear_points_x_value :
  let p : Point2D := ⟨1, 1⟩
  let a : Point2D := ⟨2, -4⟩
  let b : Point2D := ⟨x, -9⟩
  collinear p a b → x = 3 := by
  sorry

end collinear_points_x_value_l1997_199783


namespace f_derivative_at_zero_l1997_199782

theorem f_derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x*(deriv f 1)) :
  deriv f 0 = -4 := by sorry

end f_derivative_at_zero_l1997_199782


namespace f_equals_g_l1997_199715

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem f_equals_g : f = g := by sorry

end f_equals_g_l1997_199715


namespace perpendicular_line_through_point_l1997_199797

/-- Given line l1 with equation 4x + 5y - 8 = 0 -/
def l1 : ℝ → ℝ → Prop :=
  λ x y => 4*x + 5*y - 8 = 0

/-- Point A with coordinates (3, 2) -/
def A : ℝ × ℝ := (3, 2)

/-- The perpendicular line l2 passing through A -/
def l2 : ℝ → ℝ → Prop :=
  λ x y => 5*x - 4*y - 7 = 0

theorem perpendicular_line_through_point :
  (∀ x y, l2 x y ↔ 5*x - 4*y - 7 = 0) ∧
  l2 A.1 A.2 ∧
  (∀ x1 y1 x2 y2, l1 x1 y1 → l1 x2 y2 → l2 x1 y1 → l2 x2 y2 →
    (x2 - x1) * (4) + (y2 - y1) * (5) = 0) :=
by sorry

end perpendicular_line_through_point_l1997_199797


namespace total_items_is_110_l1997_199720

/-- The number of croissants each person eats per day -/
def croissants_per_person : ℕ := 7

/-- The number of cakes each person eats per day -/
def cakes_per_person : ℕ := 18

/-- The number of pizzas each person eats per day -/
def pizzas_per_person : ℕ := 30

/-- The number of people eating -/
def number_of_people : ℕ := 2

/-- The total number of items consumed by both people in a day -/
def total_items : ℕ := 
  (croissants_per_person + cakes_per_person + pizzas_per_person) * number_of_people

theorem total_items_is_110 : total_items = 110 := by
  sorry

end total_items_is_110_l1997_199720


namespace conor_carrot_count_l1997_199700

/-- Represents the number of vegetables Conor can chop in a day -/
structure DailyVegetables where
  eggplants : ℕ
  carrots : ℕ
  potatoes : ℕ

/-- Represents Conor's weekly vegetable chopping -/
def WeeklyVegetables (d : DailyVegetables) (workDays : ℕ) : ℕ :=
  workDays * (d.eggplants + d.carrots + d.potatoes)

/-- Theorem stating the number of carrots Conor can chop in a day -/
theorem conor_carrot_count :
  ∀ (d : DailyVegetables),
    d.eggplants = 12 →
    d.potatoes = 8 →
    WeeklyVegetables d 4 = 116 →
    d.carrots = 9 := by
  sorry


end conor_carrot_count_l1997_199700


namespace min_sum_squares_l1997_199750

/-- B-neighborhood of A is defined as the solution set of |x - A| < B where A ∈ ℝ and B > 0 -/
def neighborhood (A : ℝ) (B : ℝ) : Set ℝ :=
  {x : ℝ | |x - A| < B}

theorem min_sum_squares (a b : ℝ) :
  neighborhood (a + b - 3) (a + b) = Set.Ioo (-3 : ℝ) 3 →
  ∃ (m : ℝ), m = 9/2 ∧ ∀ x y : ℝ, x^2 + y^2 ≥ m := by
  sorry

end min_sum_squares_l1997_199750


namespace ramanujan_identity_a_l1997_199712

theorem ramanujan_identity_a : 
  (((2 : ℝ) ^ (1/3) - 1) ^ (1/3) = (1/9 : ℝ) ^ (1/3) - (2/9 : ℝ) ^ (1/3) + (4/9 : ℝ) ^ (1/3)) := by
  sorry

end ramanujan_identity_a_l1997_199712


namespace julia_drove_214_miles_l1997_199762

/-- Calculates the number of miles driven given the total cost, daily rental rate, and per-mile rate -/
def miles_driven (total_cost daily_rate mile_rate : ℚ) : ℚ :=
  (total_cost - daily_rate) / mile_rate

/-- Proves that Julia drove 214 miles given the rental conditions -/
theorem julia_drove_214_miles :
  let total_cost : ℚ := 46.12
  let daily_rate : ℚ := 29
  let mile_rate : ℚ := 0.08
  miles_driven total_cost daily_rate mile_rate = 214 := by
    sorry

#eval miles_driven 46.12 29 0.08

end julia_drove_214_miles_l1997_199762


namespace quadratic_roots_sum_of_squares_l1997_199701

theorem quadratic_roots_sum_of_squares (a b c : ℝ) : 
  (∀ x, x^2 - 7*x + c = 0 ↔ x = a ∨ x = b) →
  a^2 + b^2 = 17 →
  c = 16 := by
sorry

end quadratic_roots_sum_of_squares_l1997_199701


namespace fraction_simplification_l1997_199796

theorem fraction_simplification : 
  (1 / (1 + 1 / (3 + 1 / 4))) = 13 / 17 := by
  sorry

end fraction_simplification_l1997_199796


namespace best_approximation_log5_10_l1997_199706

/-- Approximation of log₁₀2 -/
def log10_2 : ℝ := 0.301

/-- Approximation of log₁₀3 -/
def log10_3 : ℝ := 0.477

/-- The set of possible fractions for approximating log₅10 -/
def fraction_options : List ℚ := [8/7, 9/7, 10/7, 11/7, 12/7]

/-- Statement: The fraction 10/7 is the closest approximation to log₅10 among the given options -/
theorem best_approximation_log5_10 : 
  ∃ (x : ℚ), x ∈ fraction_options ∧ 
  ∀ (y : ℚ), y ∈ fraction_options → |x - (1 / (1 - log10_2))| ≤ |y - (1 / (1 - log10_2))| ∧
  x = 10/7 := by
  sorry

end best_approximation_log5_10_l1997_199706


namespace library_reorganization_l1997_199740

theorem library_reorganization (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) : 
  initial_boxes = 2025 →
  books_per_initial_box = 25 →
  books_per_new_box = 28 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 21 := by
sorry

end library_reorganization_l1997_199740


namespace center_after_transformations_l1997_199722

-- Define the initial center coordinates
def initial_center : ℝ × ℝ := (3, -4)

-- Define the reflection across x-axis function
def reflect_x (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Define the translation function
def translate_right (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1 + units, point.2)

-- Theorem statement
theorem center_after_transformations :
  let reflected := reflect_x initial_center
  let final := translate_right reflected 5
  final = (8, 4) := by sorry

end center_after_transformations_l1997_199722


namespace sum_of_prime_factors_3150_l1997_199751

theorem sum_of_prime_factors_3150 : (Finset.sum (Finset.filter Nat.Prime (Finset.range (3150 + 1))) id) = 17 :=
by sorry

end sum_of_prime_factors_3150_l1997_199751


namespace base4_addition_subtraction_l1997_199764

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (a b c : ℕ) : ℕ := a * 4^2 + b * 4^1 + c * 4^0

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let d := n / 64
  let r := n % 64
  let c := r / 16
  let r' := r % 16
  let b := r' / 4
  let a := r' % 4
  (d, c, b, a)

theorem base4_addition_subtraction :
  let x := base4ToBase10 2 0 3
  let y := base4ToBase10 3 2 1
  let z := base4ToBase10 1 1 2
  base10ToBase4 (x + y - z) = (1, 0, 1, 2) := by sorry

end base4_addition_subtraction_l1997_199764


namespace total_pencils_l1997_199709

theorem total_pencils (drawer : ℕ) (desk : ℕ) (added : ℕ) 
  (h1 : drawer = 43)
  (h2 : desk = 19)
  (h3 : added = 16) :
  drawer + desk + added = 78 := by
  sorry

end total_pencils_l1997_199709


namespace larger_number_proof_larger_number_is_1891_l1997_199730

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def has_at_most_three_decimal_places (n : ℕ) : Prop := n < 1000

theorem larger_number_proof (small : ℕ) (large : ℕ) : Prop :=
  large - small = 1355 ∧
  large / small = 6 ∧
  large % small = 15 ∧
  is_prime (sum_of_digits large) ∧
  has_at_most_three_decimal_places small ∧
  has_at_most_three_decimal_places large ∧
  large = 1891

theorem larger_number_is_1891 : ∃ (small : ℕ) (large : ℕ), larger_number_proof small large := by
  sorry

end larger_number_proof_larger_number_is_1891_l1997_199730


namespace sqrt_sum_equals_two_l1997_199794

theorem sqrt_sum_equals_two (x y θ : ℝ) 
  (h1 : x + y = 3 - Real.cos (4 * θ)) 
  (h2 : x - y = 4 * Real.sin (2 * θ)) : 
  Real.sqrt x + Real.sqrt y = 2 := by
sorry

end sqrt_sum_equals_two_l1997_199794


namespace inscribed_triangle_regular_polygon_sides_l1997_199786

/-- Represents a triangle inscribed in a circle -/
structure InscribedTriangle :=
  (A B C : ℝ × ℝ)  -- Vertices of the triangle
  (center : ℝ × ℝ)  -- Center of the circle
  (radius : ℝ)  -- Radius of the circle

/-- Calculates the angle at a vertex of a triangle -/
def angle (t : InscribedTriangle) (v : Fin 3) : ℝ :=
  sorry  -- Definition of angle calculation

/-- Represents a regular polygon inscribed in a circle -/
structure RegularPolygon :=
  (n : ℕ)  -- Number of sides
  (center : ℝ × ℝ)  -- Center of the circle
  (radius : ℝ)  -- Radius of the circle

/-- Checks if two points are adjacent vertices of a regular polygon -/
def areAdjacentVertices (p : RegularPolygon) (v1 v2 : ℝ × ℝ) : Prop :=
  sorry  -- Definition of adjacency check

theorem inscribed_triangle_regular_polygon_sides 
  (t : InscribedTriangle) 
  (p : RegularPolygon) 
  (h1 : angle t 1 = angle t 2)  -- ∠B = ∠C
  (h2 : angle t 1 = 3 * angle t 0)  -- ∠B = 3∠A
  (h3 : t.center = p.center ∧ t.radius = p.radius)  -- Same circle
  (h4 : areAdjacentVertices p t.B t.C)  -- B and C are adjacent vertices
  : p.n = 2 :=
sorry

end inscribed_triangle_regular_polygon_sides_l1997_199786


namespace cube_sum_from_sum_and_square_sum_l1997_199711

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 4) (h2 : x^2 + y^2 = 8) : x^3 + y^3 = 16 := by
  sorry

end cube_sum_from_sum_and_square_sum_l1997_199711


namespace part_one_part_two_l1997_199704

-- Define the propositions r(x) and s(x)
def r (m : ℝ) (x : ℝ) : Prop := Real.sin x + Real.cos x > m
def s (m : ℝ) (x : ℝ) : Prop := x^2 + m*x + 1 > 0

-- Part 1
theorem part_one (m : ℝ) : 
  (∀ x ∈ Set.Ioo (1/2 : ℝ) 2, s m x) → m > -2 :=
sorry

-- Part 2
theorem part_two (m : ℝ) :
  (∀ x : ℝ, (r m x ∧ ¬s m x) ∨ (¬r m x ∧ s m x)) →
  m ∈ Set.Iic (-2) ∪ Set.Ioc (-Real.sqrt 2) 2 :=
sorry

end part_one_part_two_l1997_199704


namespace equation_system_solution_l1997_199749

theorem equation_system_solution :
  ∀ x y : ℝ,
  x * y * (x + y) = 30 ∧
  x^3 + y^3 = 35 →
  ((x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = 3)) :=
by
  sorry

end equation_system_solution_l1997_199749


namespace johnny_fish_count_l1997_199792

theorem johnny_fish_count (total : ℕ) (sony_multiplier : ℕ) (johnny_count : ℕ) : 
  total = 120 →
  sony_multiplier = 7 →
  total = johnny_count + sony_multiplier * johnny_count →
  johnny_count = 15 := by
sorry

end johnny_fish_count_l1997_199792


namespace triangle_abc_properties_l1997_199768

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given conditions
  c = Real.sqrt 2 →
  Real.cos C = 3/4 →
  2 * c * Real.sin A = b * Real.sin C →
  -- Conclusions
  b = 2 ∧
  Real.sin A = Real.sqrt 14 / 8 ∧
  Real.sin (2 * A + π/6) = (5 * Real.sqrt 21 + 9) / 32 :=
by sorry

end triangle_abc_properties_l1997_199768


namespace complex_number_problem_l1997_199760

theorem complex_number_problem (z z₁ z₂ : ℂ) : 
  z₁ = 5 + 10 * Complex.I ∧ 
  z₂ = 3 - 4 * Complex.I ∧ 
  1 / z = 1 / z₁ + 1 / z₂ → 
  z = 5 - (5 / 2) * Complex.I := by
sorry

end complex_number_problem_l1997_199760


namespace silverware_probability_l1997_199714

/-- The number of each type and color of silverware in the drawer -/
def num_each : ℕ := 8

/-- The total number of pieces of silverware in the drawer -/
def total_pieces : ℕ := 6 * num_each

/-- The number of ways to choose any 3 items from the drawer -/
def total_ways : ℕ := Nat.choose total_pieces 3

/-- The number of ways to choose one fork, one spoon, and one knife of different colors -/
def favorable_ways : ℕ := 2 * (num_each * num_each * num_each)

/-- The probability of selecting one fork, one spoon, and one knife of different colors -/
def probability : ℚ := favorable_ways / total_ways

theorem silverware_probability :
  probability = 32 / 541 := by sorry

end silverware_probability_l1997_199714


namespace max_value_of_function_l1997_199723

theorem max_value_of_function (x : ℝ) (h : x^2 + x + 1 ≠ 0) :
  ∃ (M : ℝ), M = 13/3 ∧ ∀ (y : ℝ), (3*x^2 + 3*x + 4) / (x^2 + x + 1) ≤ M :=
by sorry

end max_value_of_function_l1997_199723


namespace modulus_of_z_l1997_199725

theorem modulus_of_z (z : ℂ) (r θ : ℝ) (h1 : z + 1/z = r) (h2 : r = 2 * Real.sin θ) (h3 : |r| < 3) : Complex.abs z = 1 := by
  sorry

end modulus_of_z_l1997_199725


namespace photos_per_album_l1997_199770

/-- Given 180 total photos divided equally among 9 albums, prove that each album contains 20 photos. -/
theorem photos_per_album (total_photos : ℕ) (num_albums : ℕ) (photos_per_album : ℕ) : 
  total_photos = 180 → num_albums = 9 → total_photos = num_albums * photos_per_album → photos_per_album = 20 := by
  sorry

end photos_per_album_l1997_199770


namespace squares_in_figure_100_l1997_199746

-- Define the sequence function
def f (n : ℕ) : ℕ := 2 * n^3 + 2 * n^2 + 4 * n + 1

-- State the theorem
theorem squares_in_figure_100 :
  f 0 = 1 ∧ f 1 = 9 ∧ f 2 = 29 ∧ f 3 = 65 → f 100 = 2020401 :=
by
  sorry


end squares_in_figure_100_l1997_199746


namespace unique_point_equal_angles_l1997_199747

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the focus
def F : ℝ × ℝ := (1, 0)

-- Define the point P
def P : ℝ × ℝ := (2, 0)

-- Define a chord passing through F
def is_chord_through_F (A B : ℝ × ℝ) : Prop :=
  is_on_ellipse A.1 A.2 ∧ is_on_ellipse B.1 B.2 ∧ 
  ∃ t : ℝ, (1 - t) • A + t • B = F

-- Define equality of angles APF and BPF
def angles_equal (A B : ℝ × ℝ) : Prop :=
  (A.2 / (A.1 - P.1)) + (B.2 / (B.1 - P.1)) = 0

-- Theorem statement
theorem unique_point_equal_angles :
  ∀ A B : ℝ × ℝ, is_chord_through_F A B → angles_equal A B ∧
  ∀ p : ℝ, p > 0 ∧ p ≠ 2 → ∃ A' B' : ℝ × ℝ, is_chord_through_F A' B' ∧ ¬angles_equal A' B' :=
sorry

end unique_point_equal_angles_l1997_199747


namespace cos_2018pi_over_3_l1997_199772

theorem cos_2018pi_over_3 : Real.cos (2018 * Real.pi / 3) = -(1 / 2) := by sorry

end cos_2018pi_over_3_l1997_199772


namespace square_root_of_64_l1997_199752

theorem square_root_of_64 : ∃ x : ℝ, x^2 = 64 ∧ (x = 8 ∨ x = -8) :=
  sorry

end square_root_of_64_l1997_199752


namespace no_point_satisfies_both_systems_l1997_199713

/-- A point in the 2D plane satisfies System I if it meets all these conditions -/
def satisfies_system_I (x y : ℝ) : Prop :=
  y < 3 ∧ x - y < 3 ∧ x + y < 4

/-- A point in the 2D plane satisfies System II if it meets all these conditions -/
def satisfies_system_II (x y : ℝ) : Prop :=
  (y - 3) * (x - y - 3) ≥ 0 ∧
  (y - 3) * (x + y - 4) ≤ 0 ∧
  (x - y - 3) * (x + y - 4) ≤ 0

/-- There is no point that satisfies both System I and System II -/
theorem no_point_satisfies_both_systems :
  ¬ ∃ (x y : ℝ), satisfies_system_I x y ∧ satisfies_system_II x y :=
by sorry

end no_point_satisfies_both_systems_l1997_199713


namespace quadratic_solutions_solution_at_minus_four_solution_at_minus_three_no_solution_at_minus_five_l1997_199707

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - (a - 12) * x + 36 - 5 * a

-- Define the condition for x
def x_condition (x : ℝ) : Prop := -6 < x ∧ x ≤ -2 ∧ x ≠ -5 ∧ x ≠ -4 ∧ x ≠ -3

-- Define the range for a
def a_range (a : ℝ) : Prop := (4 < a ∧ a < 4.5) ∨ (4.5 < a ∧ a ≤ 16/3)

-- Main theorem
theorem quadratic_solutions (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x_condition x₁ ∧ x_condition x₂ ∧ 
   quadratic a x₁ = 0 ∧ quadratic a x₂ = 0) ↔ a_range a :=
sorry

-- Theorems for specific points
theorem solution_at_minus_four :
  quadratic 4 (-4) = 0 :=
sorry

theorem solution_at_minus_three :
  quadratic 4.5 (-3) = 0 :=
sorry

theorem no_solution_at_minus_five (a : ℝ) :
  quadratic a (-5) ≠ 0 :=
sorry

end quadratic_solutions_solution_at_minus_four_solution_at_minus_three_no_solution_at_minus_five_l1997_199707


namespace initial_books_correct_l1997_199780

/-- The number of books in the special collection at the beginning of the month. -/
def initial_books : ℕ := 75

/-- The number of books loaned out during the month. -/
def loaned_books : ℕ := 60

/-- The percentage of loaned books that are returned by the end of the month. -/
def return_rate : ℚ := 70 / 100

/-- The number of books in the special collection at the end of the month. -/
def final_books : ℕ := 57

/-- Theorem stating that the initial number of books is correct given the conditions. -/
theorem initial_books_correct : 
  initial_books = final_books + (loaned_books - (return_rate * loaned_books).floor) :=
sorry

end initial_books_correct_l1997_199780


namespace tangent_line_at_origin_l1997_199765

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem: The equation of the tangent line to y = x^3 - 3x^2 + 1 at (0, 1) is y = 1
theorem tangent_line_at_origin (x : ℝ) : 
  (f' 0) * x + f 0 = 1 := by
  sorry

end tangent_line_at_origin_l1997_199765


namespace triple_overlap_area_is_six_l1997_199748

/-- Represents a rectangular carpet with width and length -/
structure Carpet where
  width : ℝ
  length : ℝ

/-- Represents the auditorium floor -/
structure Auditorium where
  width : ℝ
  length : ℝ

/-- Calculates the area of triple overlap given three carpets and an auditorium -/
def tripleOverlapArea (c1 c2 c3 : Carpet) (a : Auditorium) : ℝ :=
  sorry

/-- Theorem stating that the area of triple overlap is 6 square meters -/
theorem triple_overlap_area_is_six 
  (c1 : Carpet) 
  (c2 : Carpet) 
  (c3 : Carpet) 
  (a : Auditorium) 
  (h1 : c1.width = 6 ∧ c1.length = 8)
  (h2 : c2.width = 6 ∧ c2.length = 6)
  (h3 : c3.width = 5 ∧ c3.length = 7)
  (h4 : a.width = 10 ∧ a.length = 10) :
  tripleOverlapArea c1 c2 c3 a = 6 := by
  sorry

end triple_overlap_area_is_six_l1997_199748


namespace sqrt_product_equality_l1997_199798

theorem sqrt_product_equality : Real.sqrt 12 * Real.sqrt 8 = 4 * Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l1997_199798


namespace factor_expression_l1997_199724

theorem factor_expression (b : ℝ) : 63 * b^2 + 189 * b = 63 * b * (b + 3) := by
  sorry

end factor_expression_l1997_199724


namespace rice_mixture_price_l1997_199793

/-- Given two types of rice mixed together, prove the price of the first type --/
theorem rice_mixture_price (price2 : ℚ) (weight1 weight2 : ℚ) (mixture_price : ℚ) 
  (h1 : price2 = 960 / 100)  -- Rs. 9.60 converted to a rational number
  (h2 : weight1 = 49)
  (h3 : weight2 = 56)
  (h4 : mixture_price = 820 / 100)  -- Rs. 8.20 converted to a rational number
  : ∃ (price1 : ℚ), price1 = 660 / 100 ∧  -- Rs. 6.60 converted to a rational number
    (weight1 * price1 + weight2 * price2) / (weight1 + weight2) = mixture_price :=
by sorry

end rice_mixture_price_l1997_199793


namespace greatest_integer_quadratic_inequality_l1997_199718

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 9*n + 20 ≤ 0 ∧ n = 5 ∧ ∀ (m : ℤ), m^2 - 9*m + 20 ≤ 0 → m ≤ 5 := by
  sorry

end greatest_integer_quadratic_inequality_l1997_199718


namespace square_sum_equals_six_l1997_199761

theorem square_sum_equals_six (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) :
  x^2 + y^2 = 6 := by
sorry

end square_sum_equals_six_l1997_199761


namespace sum_abc_l1997_199779

theorem sum_abc (a b c : ℚ) 
  (eq1 : 2 * a + 3 * b + c = 27) 
  (eq2 : 4 * a + 6 * b + 5 * c = 71) : 
  a + b + c = 115 / 9 := by
  sorry

end sum_abc_l1997_199779
