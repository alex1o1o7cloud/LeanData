import Mathlib

namespace store_products_l3448_344889

theorem store_products (big_box_capacity small_box_capacity total_products : ℕ) 
  (h1 : big_box_capacity = 50)
  (h2 : small_box_capacity = 40)
  (h3 : total_products = 212) :
  ∃ (big_boxes small_boxes : ℕ), 
    big_boxes * big_box_capacity + small_boxes * small_box_capacity = total_products :=
by sorry

end store_products_l3448_344889


namespace history_students_count_l3448_344852

def total_students : ℕ := 86
def math_students : ℕ := 17
def english_students : ℕ := 36
def all_three_classes : ℕ := 3
def exactly_two_classes : ℕ := 3

theorem history_students_count : 
  ∃ (history_students : ℕ), 
    history_students = total_students - math_students - english_students + all_three_classes := by
  sorry

end history_students_count_l3448_344852


namespace sine_phase_shift_specific_sine_phase_shift_l3448_344835

/-- The phase shift of a sine function y = a * sin(bx - c) is c/b to the right when c is positive. -/
theorem sine_phase_shift (a b c : ℝ) (h : c > 0) :
  let f := fun x => a * Real.sin (b * x - c)
  let phase_shift := c / b
  (∀ x, f (x + phase_shift) = a * Real.sin (b * x)) :=
sorry

/-- The phase shift of y = 3 * sin(3x - π/4) is π/12 to the right. -/
theorem specific_sine_phase_shift :
  let f := fun x => 3 * Real.sin (3 * x - π/4)
  let phase_shift := π/12
  (∀ x, f (x + phase_shift) = 3 * Real.sin (3 * x)) :=
sorry

end sine_phase_shift_specific_sine_phase_shift_l3448_344835


namespace base_9_to_10_3562_l3448_344873

def base_9_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

theorem base_9_to_10_3562 :
  base_9_to_10 [2, 6, 5, 3] = 2648 := by
  sorry

end base_9_to_10_3562_l3448_344873


namespace jill_minus_jake_equals_one_l3448_344850

def peach_problem (jake steven jill : ℕ) : Prop :=
  (jake + 16 = steven) ∧ 
  (steven = jill + 15) ∧ 
  (jill = 12)

theorem jill_minus_jake_equals_one :
  ∀ jake steven jill : ℕ, peach_problem jake steven jill → jill - jake = 1 :=
by
  sorry

end jill_minus_jake_equals_one_l3448_344850


namespace probability_of_all_even_sums_l3448_344822

/-- Represents a tile with a number from 1 to 10 -/
def Tile := Fin 10

/-- Represents a player's selection of 3 tiles -/
def PlayerSelection := Fin 3 → Tile

/-- The set of all possible distributions of tiles to three players -/
def AllDistributions := Fin 3 → PlayerSelection

/-- Checks if a player's selection sum is even -/
def isEvenSum (selection : PlayerSelection) : Prop :=
  (selection 0).val + (selection 1).val + (selection 2).val % 2 = 0

/-- Checks if all players have even sums in a distribution -/
def allEvenSums (distribution : AllDistributions) : Prop :=
  ∀ i : Fin 3, isEvenSum (distribution i)

/-- The number of distributions where all players have even sums -/
def favorableDistributions : ℕ := sorry

/-- The total number of possible distributions -/
def totalDistributions : ℕ := sorry

/-- The main theorem stating the probability -/
theorem probability_of_all_even_sums :
  (favorableDistributions : ℚ) / totalDistributions = 1 / 28 := sorry

end probability_of_all_even_sums_l3448_344822


namespace function_range_condition_l3448_344888

theorem function_range_condition (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, |2 * x₁ - a| + |2 * x₁ + 3| = |x₂ - 1| + 2) →
  (a ≥ -1 ∨ a ≤ -5) := by
sorry

end function_range_condition_l3448_344888


namespace min_value_of_reciprocal_sum_l3448_344837

theorem min_value_of_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (2*a*(-1) - b*2 + 2 = 0) → (1/a + 1/b ≥ 4) ∧ ∃ a b, (1/a + 1/b = 4) := by
  sorry

end min_value_of_reciprocal_sum_l3448_344837


namespace stephens_number_l3448_344866

theorem stephens_number : ∃! n : ℕ, 
  9000 ≤ n ∧ n ≤ 15000 ∧ 
  n % 216 = 0 ∧ 
  n % 55 = 0 ∧ 
  n = 11880 := by sorry

end stephens_number_l3448_344866


namespace min_value_quadratic_l3448_344853

theorem min_value_quadratic (x : ℝ) :
  ∃ (m : ℝ), m = 1438 ∧ ∀ x, 3 * x^2 - 12 * x + 1450 ≥ m := by
  sorry

end min_value_quadratic_l3448_344853


namespace square_area_in_circle_l3448_344851

/-- Given a circle with radius 1 and a square with two vertices on the circle
    and one edge passing through the center, prove the area of the square is 4/5 -/
theorem square_area_in_circle (circle_radius : ℝ) (square_side : ℝ) : 
  circle_radius = 1 →
  ∃ (x : ℝ), square_side = 2 * x ∧ 
  x ^ 2 + (2 * x) ^ 2 = circle_radius ^ 2 →
  square_side ^ 2 = 4 / 5 := by
  sorry

#check square_area_in_circle

end square_area_in_circle_l3448_344851


namespace log_comparison_l3448_344846

theorem log_comparison : Real.log 675 / Real.log 135 > Real.log 75 / Real.log 45 := by
  sorry

end log_comparison_l3448_344846


namespace quadratic_equation_single_solution_l3448_344857

theorem quadratic_equation_single_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, b * x^2 + 15 * x + 6 = 0) :
  ∃ x, b * x^2 + 15 * x + 6 = 0 ∧ x = -4/5 := by
  sorry

end quadratic_equation_single_solution_l3448_344857


namespace railway_graph_theorem_l3448_344883

/-- A graph representing the railway network --/
structure RailwayGraph where
  V : Finset Nat
  E : Finset (Nat × Nat)
  edge_in_V : ∀ (e : Nat × Nat), e ∈ E → e.1 ∈ V ∧ e.2 ∈ V

/-- The degree of a vertex in the graph --/
def degree (G : RailwayGraph) (v : Nat) : Nat :=
  (G.E.filter (fun e => e.1 = v ∨ e.2 = v)).card

/-- The theorem statement --/
theorem railway_graph_theorem (G : RailwayGraph) 
  (hV : G.V.card = 9)
  (hM : degree G 1 = 7)
  (hSP : degree G 2 = 5)
  (hT : degree G 3 = 4)
  (hY : degree G 4 = 2)
  (hB : degree G 5 = 2)
  (hS : degree G 6 = 2)
  (hZ : degree G 7 = 1)
  (hEven : Even (G.E.card * 2))
  (hVV : G.V.card = 9 → ∃ v ∈ G.V, v ≠ 1 ∧ v ≠ 2 ∧ v ≠ 3 ∧ v ≠ 4 ∧ v ≠ 5 ∧ v ≠ 6 ∧ v ≠ 7 ∧ v ≠ 8) :
  ∃ v ∈ G.V, v ≠ 1 ∧ v ≠ 2 ∧ v ≠ 3 ∧ v ≠ 4 ∧ v ≠ 5 ∧ v ≠ 6 ∧ v ≠ 7 ∧ v ≠ 8 ∧ 
    (degree G v = 2 ∨ degree G v = 3 ∨ degree G v = 4 ∨ degree G v = 5) :=
by sorry

end railway_graph_theorem_l3448_344883


namespace c_range_l3448_344817

def p (c : ℝ) : Prop := c^2 < c

def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 + 4*c*x + 1 > 0

def range_of_c (c : ℝ) : Prop :=
  (c > -1/2 ∧ c ≤ 0) ∨ (c ≥ 1/2 ∧ c < 1)

theorem c_range (c : ℝ) :
  (p c ∨ q c) ∧ ¬(p c ∧ q c) → range_of_c c :=
sorry

end c_range_l3448_344817


namespace square_rectangle_area_relation_l3448_344820

theorem square_rectangle_area_relation : 
  ∀ x : ℝ,
  let square_side := x - 4
  let rect_length := x - 2
  let rect_width := x + 6
  let square_area := square_side * square_side
  let rect_area := rect_length * rect_width
  rect_area = 3 * square_area →
  (∃ x₁ x₂ : ℝ, 
    (square_side = x₁ - 4 ∧ rect_length = x₁ - 2 ∧ rect_width = x₁ + 6 ∧
     square_side = x₂ - 4 ∧ rect_length = x₂ - 2 ∧ rect_width = x₂ + 6) ∧
    x₁ + x₂ = 13) :=
by
  sorry

end square_rectangle_area_relation_l3448_344820


namespace function_domain_implies_m_range_l3448_344882

/-- The function f(x) = √(mx² - (1-m)x + m) has domain R if and only if m ≥ 1/3 -/
theorem function_domain_implies_m_range (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (m * x^2 - (1 - m) * x + m)) ↔ m ≥ 1/3 :=
by sorry

end function_domain_implies_m_range_l3448_344882


namespace octagon_arc_length_l3448_344800

/-- The length of the arc intercepted by one side of a regular octagon inscribed in a circle -/
theorem octagon_arc_length (r : ℝ) (h : r = 4) : 
  (2 * π * r) / 8 = π := by sorry

end octagon_arc_length_l3448_344800


namespace ellipse_foci_distance_l3448_344842

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 144 is 8√2 -/
theorem ellipse_foci_distance : 
  ∀ (x y : ℝ), 9 * x^2 + y^2 = 144 → 
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 128 := by
  sorry

end ellipse_foci_distance_l3448_344842


namespace approximation_equality_l3448_344865

/-- For any function f, f(69.28 × 0.004) / 0.03 = f(9.237333...) -/
theorem approximation_equality (f : ℝ → ℝ) : f (69.28 * 0.004) / 0.03 = f 9.237333333333333 := by
  sorry

end approximation_equality_l3448_344865


namespace village_population_problem_l3448_344895

theorem village_population_problem (P : ℝ) : 
  (P * 1.3 * 0.7 = 13650) → P = 15000 := by
  sorry

end village_population_problem_l3448_344895


namespace at_least_one_not_less_than_two_l3448_344816

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 := by
  sorry

end at_least_one_not_less_than_two_l3448_344816


namespace shopkeeper_theft_loss_l3448_344896

theorem shopkeeper_theft_loss (profit_percent : ℝ) (loss_percent : ℝ) : 
  profit_percent = 10 → loss_percent = 45 → 
  (loss_percent / 100) * (1 + profit_percent / 100) * 100 = 49.5 := by
  sorry

end shopkeeper_theft_loss_l3448_344896


namespace expected_votes_for_a_l3448_344808

-- Define the total number of voters (for simplicity, we'll use 100 as in the solution)
def total_voters : ℝ := 100

-- Define the percentage of Democratic voters
def dem_percentage : ℝ := 0.7

-- Define the percentage of Republican voters
def rep_percentage : ℝ := 1 - dem_percentage

-- Define the percentage of Democratic voters voting for candidate A
def dem_vote_for_a : ℝ := 0.8

-- Define the percentage of Republican voters voting for candidate A
def rep_vote_for_a : ℝ := 0.3

-- Theorem to prove
theorem expected_votes_for_a :
  (dem_percentage * dem_vote_for_a + rep_percentage * rep_vote_for_a) * 100 = 65 := by
  sorry


end expected_votes_for_a_l3448_344808


namespace m_range_l3448_344867

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- Theorem statement
theorem m_range (m : ℝ) : p m ∧ q m → m ∈ Set.Ioo (-2 : ℝ) 0 := by
  sorry

end m_range_l3448_344867


namespace quadratic_equation_roots_l3448_344858

theorem quadratic_equation_roots (k : ℝ) (h : k > 1) :
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
  (2 * x₁^2 - (4*k + 1) * x₁ + 2*k^2 - 1 = 0) ∧
  (2 * x₂^2 - (4*k + 1) * x₂ + 2*k^2 - 1 = 0) :=
by sorry

end quadratic_equation_roots_l3448_344858


namespace fraction_simplification_l3448_344859

theorem fraction_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  (1 - 2 / (x + 1)) / (x / (x + 1)) = (x - 1) / x := by
  sorry

end fraction_simplification_l3448_344859


namespace octagon_area_theorem_l3448_344827

/-- The area of an octagon formed by the intersection of two unit squares with the same center -/
def octagon_area (side_length : ℚ) : ℚ :=
  8 * (side_length * (1 / 2) * (1 / 2))

/-- The theorem stating the area of the octagon given the side length -/
theorem octagon_area_theorem (h : octagon_area (43 / 99) = 86 / 99) : True := by
  sorry

#eval octagon_area (43 / 99)

end octagon_area_theorem_l3448_344827


namespace work_completion_time_l3448_344864

/-- Given workers A and B, where A can complete a job in 15 days and B in 9 days,
    if A works for 5 days and then leaves, B will complete the remaining work in 6 days. -/
theorem work_completion_time (a_total_days b_total_days a_worked_days : ℕ) 
    (ha : a_total_days = 15)
    (hb : b_total_days = 9)
    (hw : a_worked_days = 5) : 
    (b_total_days : ℚ) * (1 - (a_worked_days : ℚ) / (a_total_days : ℚ)) = 6 := by
  sorry

end work_completion_time_l3448_344864


namespace jaysons_mom_age_at_birth_l3448_344812

/-- Proves that Jayson's mom was 28 when he was born, given the conditions -/
theorem jaysons_mom_age_at_birth (jayson_age : ℕ) (dad_age : ℕ) (mom_age : ℕ) 
  (h1 : jayson_age = 10)
  (h2 : dad_age = 4 * jayson_age)
  (h3 : mom_age = dad_age - 2) :
  mom_age - jayson_age = 28 := by
  sorry

end jaysons_mom_age_at_birth_l3448_344812


namespace new_year_firework_boxes_l3448_344893

/-- Calculates the number of firework boxes used in a New Year's Eve display. -/
def firework_boxes_used (total_fireworks : ℕ) (fireworks_per_digit : ℕ) (fireworks_per_letter : ℕ) (fireworks_per_box : ℕ) (year_digits : ℕ) (phrase_letters : ℕ) : ℕ :=
  let year_fireworks := fireworks_per_digit * year_digits
  let phrase_fireworks := fireworks_per_letter * phrase_letters
  let remaining_fireworks := total_fireworks - (year_fireworks + phrase_fireworks)
  remaining_fireworks / fireworks_per_box

/-- The number of firework boxes used in the New Year's Eve display is 50. -/
theorem new_year_firework_boxes :
  firework_boxes_used 484 6 5 8 4 12 = 50 := by
  sorry

end new_year_firework_boxes_l3448_344893


namespace rationalize_sqrt_5_12_l3448_344825

theorem rationalize_sqrt_5_12 : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end rationalize_sqrt_5_12_l3448_344825


namespace fraction_to_decimal_l3448_344840

theorem fraction_to_decimal : (5 : ℚ) / 16 = (3125 : ℚ) / 10000 := by sorry

end fraction_to_decimal_l3448_344840


namespace triangle_height_theorem_l3448_344811

-- Define the triangle's properties
def triangle_area : ℝ := 48  -- in square decimeters
def triangle_base : ℝ := 6   -- in meters

-- Convert base from meters to decimeters
def triangle_base_dm : ℝ := triangle_base * 10

-- Define the theorem
theorem triangle_height_theorem :
  ∃ (height : ℝ), 
    (triangle_base_dm * height / 2 = triangle_area) ∧ 
    (height = 1.6) := by
  sorry

end triangle_height_theorem_l3448_344811


namespace arctan_sum_of_cubic_roots_l3448_344847

theorem arctan_sum_of_cubic_roots (u v w : ℝ) : 
  u^3 - 10*u + 11 = 0 → 
  v^3 - 10*v + 11 = 0 → 
  w^3 - 10*w + 11 = 0 → 
  u + v + w = 0 →
  u*v + v*w + w*u = -10 →
  u*v*w = -11 →
  Real.arctan u + Real.arctan v + Real.arctan w = π/4 := by sorry

end arctan_sum_of_cubic_roots_l3448_344847


namespace no_extremum_range_l3448_344828

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- Theorem stating the range of a for which f(x) has no extremum -/
theorem no_extremum_range (a : ℝ) : 
  (∀ x : ℝ, f_derivative a x ≥ 0) ↔ a ∈ Set.Icc (-1 : ℝ) 2 := by sorry

end no_extremum_range_l3448_344828


namespace exists_m_for_all_x_m_range_when_exists_x_l3448_344862

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Theorem 1: Existence of m such that m + f(x) > 0 for all x
theorem exists_m_for_all_x (m : ℝ) : 
  (∀ x, m + f x > 0) ↔ m > -2 := by sorry

-- Theorem 2: Range of m when there exists x such that m - f(x) > 0
theorem m_range_when_exists_x (m : ℝ) :
  (∃ x, m - f x > 0) → m > 2 := by sorry

end exists_m_for_all_x_m_range_when_exists_x_l3448_344862


namespace inequality_proof_l3448_344899

theorem inequality_proof (a b c : ℝ) (h : a^6 + b^6 + c^6 = 3) :
  a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := by sorry

end inequality_proof_l3448_344899


namespace school_boys_count_l3448_344897

/-- Represents the number of boys in the school -/
def num_boys : ℕ := 410

/-- Represents the initial number of girls in the school -/
def initial_girls : ℕ := 632

/-- Represents the number of additional girls that joined the school -/
def additional_girls : ℕ := 465

/-- Represents the difference between girls and boys after the addition -/
def girl_boy_difference : ℕ := 687

/-- Proves that the number of boys in the school is 410 -/
theorem school_boys_count :
  initial_girls + additional_girls = num_boys + girl_boy_difference := by
  sorry

#check school_boys_count

end school_boys_count_l3448_344897


namespace johns_days_off_l3448_344836

/-- Calculates the number of days John takes off per week given his streaming schedule and earnings. -/
theorem johns_days_off (hours_per_session : ℕ) (hourly_rate : ℕ) (weekly_earnings : ℕ) (days_per_week : ℕ)
  (h1 : hours_per_session = 4)
  (h2 : hourly_rate = 10)
  (h3 : weekly_earnings = 160)
  (h4 : days_per_week = 7) :
  days_per_week - (weekly_earnings / hourly_rate / hours_per_session) = 3 :=
by sorry

end johns_days_off_l3448_344836


namespace triangle_perimeter_l3448_344860

theorem triangle_perimeter (a b c : ℕ) : 
  a = 2 → b = 7 → 
  c % 2 = 0 →
  c > (b - a) →
  c < (b + a) →
  a + b + c = 15 :=
by
  sorry

end triangle_perimeter_l3448_344860


namespace circle_radius_sqrt_61_l3448_344891

/-- Given a circle with center on the x-axis passing through points (2,5) and (3,2),
    its radius is √61. -/
theorem circle_radius_sqrt_61 :
  ∀ x : ℝ,
  (∃ r : ℝ, r > 0 ∧
    r^2 = (x - 2)^2 + 5^2 ∧
    r^2 = (x - 3)^2 + 2^2) →
  ∃ r : ℝ, r > 0 ∧ r^2 = 61 :=
by sorry


end circle_radius_sqrt_61_l3448_344891


namespace sara_remaining_money_l3448_344881

/-- Calculates the remaining money after a two-week pay period and a purchase -/
def remaining_money (hours_per_week : ℕ) (hourly_rate : ℚ) (purchase_cost : ℚ) : ℚ :=
  2 * (hours_per_week : ℚ) * hourly_rate - purchase_cost

/-- Proves that given the specified work conditions and purchase, the remaining money is $510 -/
theorem sara_remaining_money :
  remaining_money 40 (11.5) 410 = 510 := by
  sorry

end sara_remaining_money_l3448_344881


namespace smallest_integer_with_remainders_l3448_344841

theorem smallest_integer_with_remainders :
  ∃ (x : ℕ), x > 0 ∧
  x % 6 = 5 ∧
  x % 7 = 6 ∧
  x % 8 = 7 ∧
  ∀ (y : ℕ), y > 0 →
    (y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7) → x ≤ y :=
by sorry

end smallest_integer_with_remainders_l3448_344841


namespace arcsin_one_half_l3448_344869

theorem arcsin_one_half : Real.arcsin (1/2) = π/6 := by
  sorry

end arcsin_one_half_l3448_344869


namespace frisbee_game_probability_l3448_344870

/-- The probability that Alice has the frisbee after three turns in the frisbee game. -/
theorem frisbee_game_probability : 
  let alice_toss_prob : ℚ := 2/3
  let alice_keep_prob : ℚ := 1/3
  let bob_toss_prob : ℚ := 1/4
  let bob_keep_prob : ℚ := 3/4
  let alice_has_frisbee_after_three_turns : ℚ := 
    alice_toss_prob * bob_keep_prob * bob_keep_prob +
    alice_keep_prob * alice_keep_prob
  alice_has_frisbee_after_three_turns = 35/72 :=
by sorry

end frisbee_game_probability_l3448_344870


namespace teds_age_l3448_344854

theorem teds_age (s t j : ℕ) : 
  t = 2 * s - 20 →
  j = s + 6 →
  t + s + j = 90 →
  t = 32 := by
  sorry

end teds_age_l3448_344854


namespace tangent_y_intercept_l3448_344821

/-- The function representing the curve y = x³ + 11 -/
def f (x : ℝ) : ℝ := x^3 + 11

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3 * x^2

/-- The point of tangency -/
def point_of_tangency : ℝ × ℝ := (1, 12)

/-- The slope of the tangent line at the point of tangency -/
def tangent_slope : ℝ := f' point_of_tangency.1

/-- The y-intercept of the tangent line -/
def y_intercept : ℝ := point_of_tangency.2 - tangent_slope * point_of_tangency.1

theorem tangent_y_intercept :
  y_intercept = 9 :=
sorry

end tangent_y_intercept_l3448_344821


namespace hyperbola_eccentricity_l3448_344831

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The left focus of a hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Points where a perpendicular from the right focus intersects the hyperbola -/
def intersection_points (h : Hyperbola a b) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- The inscribed circle of a triangle -/
def inscribed_circle (A B C : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The eccentricity of the hyperbola is (1 + √5) / 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) :
  let (A, B) := intersection_points h
  let F₁ := left_focus h
  let (_, _, r) := inscribed_circle A B F₁
  r = a →
  eccentricity h = (1 + Real.sqrt 5) / 2 := by sorry

end hyperbola_eccentricity_l3448_344831


namespace smallest_fraction_between_l3448_344868

theorem smallest_fraction_between (p q : ℕ+) : 
  (5 : ℚ) / 9 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (3 : ℚ) / 5 ∧ 
  (∀ (p' q' : ℕ+), (5 : ℚ) / 9 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (3 : ℚ) / 5 → q ≤ q') →
  q - p = 3 := by
sorry

end smallest_fraction_between_l3448_344868


namespace triangle_side_b_l3448_344805

theorem triangle_side_b (a : ℝ) (A B : ℝ) (h1 : a = 5) (h2 : A = π/6) (h3 : Real.tan B = 3/4) :
  ∃ (b : ℝ), b = 6 ∧ (b / Real.sin B = a / Real.sin A) := by
  sorry

end triangle_side_b_l3448_344805


namespace probability_at_least_one_multiple_of_four_l3448_344819

theorem probability_at_least_one_multiple_of_four :
  let total_numbers : ℕ := 100
  let multiples_of_four : ℕ := 25
  let non_multiples_of_four : ℕ := total_numbers - multiples_of_four
  let prob_non_multiple : ℚ := non_multiples_of_four / total_numbers
  let prob_both_non_multiples : ℚ := prob_non_multiple * prob_non_multiple
  let prob_at_least_one_multiple : ℚ := 1 - prob_both_non_multiples
  prob_at_least_one_multiple = 7 / 16 := by
sorry

end probability_at_least_one_multiple_of_four_l3448_344819


namespace arithmetic_sum_example_l3448_344875

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sum_example : arithmetic_sum 2 20 2 = 110 := by
  sorry

end arithmetic_sum_example_l3448_344875


namespace pyramid_frustum_volume_l3448_344818

/-- Calculate the volume of a pyramid frustum given the dimensions of the original and smaller pyramids --/
theorem pyramid_frustum_volume
  (base_edge_original : ℝ)
  (altitude_original : ℝ)
  (base_edge_smaller : ℝ)
  (altitude_smaller : ℝ)
  (h_base_edge_original : base_edge_original = 18)
  (h_altitude_original : altitude_original = 12)
  (h_base_edge_smaller : base_edge_smaller = 12)
  (h_altitude_smaller : altitude_smaller = 8) :
  (1/3 * base_edge_original^2 * altitude_original) - (1/3 * base_edge_smaller^2 * altitude_smaller) = 912 := by
  sorry

#check pyramid_frustum_volume

end pyramid_frustum_volume_l3448_344818


namespace two_digit_triple_reverse_difference_l3448_344838

theorem two_digit_triple_reverse_difference (A B : ℕ) : 
  A ≠ 0 → 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  2 ∣ ((30 * B + A) - (10 * B + A)) := by
sorry

end two_digit_triple_reverse_difference_l3448_344838


namespace kendra_age_l3448_344802

/-- Proves that Kendra's age is 18 given the conditions in the problem -/
theorem kendra_age :
  ∀ (k s t : ℕ), -- k: Kendra's age, s: Sam's age, t: Sue's age
  s = 2 * t →    -- Sam is twice as old as Sue
  k = 3 * s →    -- Kendra is 3 times as old as Sam
  (k + 3) + (s + 3) + (t + 3) = 36 → -- Their total age in 3 years will be 36
  k = 18 := by
    sorry -- Proof omitted

end kendra_age_l3448_344802


namespace exactly_one_true_l3448_344830

def proposition1 : Prop := ∀ x : ℝ, x^4 > x^2

def proposition2 : Prop := ∀ p q : Prop, (¬(p ∧ q)) → (¬p ∧ ¬q)

def proposition3 : Prop := (¬∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0)

theorem exactly_one_true : 
  (proposition1 ∧ ¬proposition2 ∧ ¬proposition3) ∨
  (¬proposition1 ∧ proposition2 ∧ ¬proposition3) ∨
  (¬proposition1 ∧ ¬proposition2 ∧ proposition3) :=
sorry

end exactly_one_true_l3448_344830


namespace max_value_of_a_l3448_344872

theorem max_value_of_a (a b c d : ℤ) 
  (h1 : a < 2*b) 
  (h2 : b < 3*c) 
  (h3 : c < 4*d) 
  (h4 : d < 100) : 
  a ≤ 2367 ∧ ∃ (a₀ b₀ c₀ d₀ : ℤ), 
    a₀ < 2*b₀ ∧ 
    b₀ < 3*c₀ ∧ 
    c₀ < 4*d₀ ∧ 
    d₀ < 100 ∧ 
    a₀ = 2367 :=
sorry

end max_value_of_a_l3448_344872


namespace max_value_implies_a_equals_5_l3448_344885

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + a

-- State the theorem
theorem max_value_implies_a_equals_5 :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 2, f a x ≤ 5) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 5) → a = 5 :=
by sorry

end max_value_implies_a_equals_5_l3448_344885


namespace dodecahedron_path_count_l3448_344807

/-- Represents a face of the dodecahedron --/
inductive Face
  | Top
  | Bottom
  | UpperRing (n : Fin 5)
  | LowerRing (n : Fin 5)

/-- Represents a valid path on the dodecahedron --/
def ValidPath : List Face → Prop :=
  sorry

/-- The number of valid paths from top to bottom face --/
def numValidPaths : Nat :=
  sorry

/-- Theorem stating that the number of valid paths is 810 --/
theorem dodecahedron_path_count :
  numValidPaths = 810 :=
sorry

end dodecahedron_path_count_l3448_344807


namespace special_function_properties_l3448_344803

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) - f y = (x + 2*y + 2) * x) ∧ (f 2 = 12)

theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  (f 0 = 4) ∧
  (Set.Icc (-1 : ℝ) 5 = {a | ∃ x₀ ∈ Set.Ioo 1 4, f x₀ - 8 = a * x₀}) :=
sorry

end special_function_properties_l3448_344803


namespace fourteenth_root_of_unity_l3448_344832

theorem fourteenth_root_of_unity : 
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 13 ∧ 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (2 * n * π * Complex.I / 14) :=
by sorry

end fourteenth_root_of_unity_l3448_344832


namespace derivative_at_one_l3448_344890

def f (x : ℝ) (k : ℝ) : ℝ := x^3 - 2*k*x + 1

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, deriv f x = f' x) →
  (∃ k, ∀ x, f x = x^3 - 2*k*x + 1) →
  f' 1 = 1 := by
sorry

end derivative_at_one_l3448_344890


namespace square_root_equality_l3448_344806

theorem square_root_equality (x : ℝ) (a : ℝ) 
  (h_pos : x > 0) 
  (h1 : Real.sqrt x = 2 * a - 3) 
  (h2 : Real.sqrt x = 5 - a) : 
  a = -2 := by
sorry

end square_root_equality_l3448_344806


namespace cos_sum_sevenths_pi_l3448_344815

theorem cos_sum_sevenths_pi : 
  Real.cos (π / 7) + Real.cos (2 * π / 7) + Real.cos (3 * π / 7) + 
  Real.cos (4 * π / 7) + Real.cos (5 * π / 7) + Real.cos (6 * π / 7) = 0 := by
  sorry

end cos_sum_sevenths_pi_l3448_344815


namespace ten_person_meeting_handshakes_l3448_344876

/-- The number of handshakes in a meeting of n people where each person
    shakes hands exactly once with every other person. -/
def handshakes (n : ℕ) : ℕ := Nat.choose n 2

/-- Theorem stating that in a meeting of 10 people, where each person
    shakes hands exactly once with every other person, the total number
    of handshakes is 45. -/
theorem ten_person_meeting_handshakes :
  handshakes 10 = 45 := by sorry

end ten_person_meeting_handshakes_l3448_344876


namespace radical_simplification_l3448_344839

theorem radical_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (20 * p) * Real.sqrt (10 * p^3) * Real.sqrt (6 * p^4) * Real.sqrt (15 * p^5) = 20 * p^6 * Real.sqrt (15 * p) :=
by sorry

end radical_simplification_l3448_344839


namespace line_equation_perpendicular_line_equation_opposite_intercepts_l3448_344849

-- Define the line l
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define the perpendicular line
def perpendicularLine : Line := { a := 2, b := 1, c := 3 }

-- Define the condition for a line to pass through a point
def passesThrough (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

-- Define the condition for two lines to be perpendicular
def isPerpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define the condition for a line to have intercepts with opposite signs
def hasOppositeIntercepts (l : Line) : Prop :=
  (l.a * l.c < 0 ∧ l.b * l.c < 0) ∨ (l.a = 0 ∧ l.b ≠ 0) ∨ (l.a ≠ 0 ∧ l.b = 0)

theorem line_equation_perpendicular (l : Line) :
  passesThrough l P ∧ isPerpendicular l perpendicularLine →
  l = { a := 1, b := -2, c := -4 } :=
sorry

theorem line_equation_opposite_intercepts (l : Line) :
  passesThrough l P ∧ hasOppositeIntercepts l →
  (l = { a := 1, b := 2, c := 0 } ∨ l = { a := 1, b := -1, c := -3 }) :=
sorry

end line_equation_perpendicular_line_equation_opposite_intercepts_l3448_344849


namespace percentage_change_relation_l3448_344861

theorem percentage_change_relation (n c : ℝ) (hn : n > 0) (hc : c > 0) :
  (∀ x : ℝ, x > 0 → x * (1 + n / 100) * (1 - c / 100) = x) →
  n^2 / c^2 = (100 + n) / (100 - c) := by
  sorry

end percentage_change_relation_l3448_344861


namespace intersection_of_A_and_B_l3448_344834

def A : Set ℤ := {x : ℤ | x^2 - 4*x ≤ 0}
def B : Set ℤ := {x : ℤ | -1 ≤ x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := by sorry

end intersection_of_A_and_B_l3448_344834


namespace expression_value_at_three_l3448_344898

theorem expression_value_at_three : 
  let x : ℝ := 3
  x + x * (x^3 - x) = 75 := by sorry

end expression_value_at_three_l3448_344898


namespace revenue_maximization_l3448_344880

/-- Revenue function for a scenic area with three ticket options -/
def revenue (x : ℝ) : ℝ := -0.1 * x^2 + 1.8 * x + 180

/-- Initial number of people choosing option A -/
def initial_A : ℝ := 20000

/-- Initial number of people choosing option B -/
def initial_B : ℝ := 10000

/-- Initial number of people choosing combined option -/
def initial_combined : ℝ := 10000

/-- Number of people switching from A to combined per 1 yuan decrease -/
def switch_rate_A : ℝ := 400

/-- Number of people switching from B to combined per 1 yuan decrease -/
def switch_rate_B : ℝ := 600

/-- Price of ticket A -/
def price_A : ℝ := 30

/-- Price of ticket B -/
def price_B : ℝ := 50

/-- Initial price of combined ticket -/
def initial_price_combined : ℝ := 70

theorem revenue_maximization :
  ∃ (x : ℝ), x = 9 ∧ 
  revenue x = 188.1 ∧ 
  ∀ y, revenue y ≤ revenue x :=
sorry

#check revenue_maximization

end revenue_maximization_l3448_344880


namespace matrix_power_in_M_l3448_344886

/-- The set M of 2x2 complex matrices where ab = cd -/
def M : Set (Matrix (Fin 2) (Fin 2) ℂ) :=
  {A | A 0 0 * A 0 1 = A 1 0 * A 1 1}

/-- Theorem statement -/
theorem matrix_power_in_M
  (A : Matrix (Fin 2) (Fin 2) ℂ)
  (k : ℕ)
  (hk : k ≥ 1)
  (hA : A ∈ M)
  (hAk : A ^ k ∈ M)
  (hAk1 : A ^ (k + 1) ∈ M)
  (hAk2 : A ^ (k + 2) ∈ M) :
  ∀ n : ℕ, n ≥ 1 → A ^ n ∈ M :=
sorry

end matrix_power_in_M_l3448_344886


namespace factor_expression_l3448_344814

theorem factor_expression (a b : ℝ) : 2*a^2*b - 4*a*b^2 + 2*b^3 = 2*b*(a-b)^2 := by
  sorry

end factor_expression_l3448_344814


namespace students_liking_both_sports_l3448_344884

/-- The number of students who like both basketball and cricket -/
def students_liking_both (b c t : ℕ) : ℕ := b + c - t

/-- Theorem: Given the conditions, prove that 3 students like both basketball and cricket -/
theorem students_liking_both_sports :
  let b := 7  -- number of students who like basketball
  let c := 5  -- number of students who like cricket
  let t := 9  -- total number of students who like basketball or cricket or both
  students_liking_both b c t = 3 := by
sorry

end students_liking_both_sports_l3448_344884


namespace f_properties_l3448_344877

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 1 / x

-- Theorem statement
theorem f_properties (a : ℝ) :
  (∀ x > 0, (deriv (f a)) x = 0 → x = 1) →
  (a = 0) ∧
  (∀ x > 0, f 0 x ≤ x * Real.exp x - x + 1 / x - 1) := by
  sorry

end f_properties_l3448_344877


namespace f_two_equals_two_l3448_344826

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f (f x) = (x^2 - x)/2 * f x + 2 - x

-- Theorem statement
theorem f_two_equals_two (h : has_property f) : f 2 = 2 := by
  sorry

end f_two_equals_two_l3448_344826


namespace units_digit_17_pow_2007_l3448_344804

theorem units_digit_17_pow_2007 : (17^2007 : ℕ) % 10 = 3 := by
  sorry

end units_digit_17_pow_2007_l3448_344804


namespace fifteen_by_fifteen_grid_toothpicks_l3448_344879

/-- Calculates the number of toothpicks in a square grid with a missing corner --/
def toothpicks_in_grid (height : ℕ) (width : ℕ) : ℕ :=
  (height + 1) * width + (width + 1) * height - 1

/-- Theorem: A 15x15 square grid with a missing corner uses 479 toothpicks --/
theorem fifteen_by_fifteen_grid_toothpicks :
  toothpicks_in_grid 15 15 = 479 := by
  sorry

end fifteen_by_fifteen_grid_toothpicks_l3448_344879


namespace isosceles_right_pyramid_leg_length_l3448_344894

/-- Represents a pyramid with an isosceles right triangle base -/
structure IsoscelesRightPyramid where
  height : ℝ
  volume : ℝ
  leg : ℝ

/-- The volume of a pyramid is one-third the product of its base area and height -/
axiom pyramid_volume (p : IsoscelesRightPyramid) : p.volume = (1/3) * (1/2 * p.leg^2) * p.height

/-- Theorem: If a pyramid with an isosceles right triangle base has height 4 and volume 6,
    then the length of the leg of the base triangle is 3 -/
theorem isosceles_right_pyramid_leg_length :
  ∀ (p : IsoscelesRightPyramid), p.height = 4 → p.volume = 6 → p.leg = 3 :=
by
  sorry


end isosceles_right_pyramid_leg_length_l3448_344894


namespace min_value_a_l3448_344856

theorem min_value_a (a : ℝ) : 
  (∀ x > a, 2 * x + 2 / (x - 1) ≥ 7) → a ≥ 3 :=
by sorry

end min_value_a_l3448_344856


namespace linear_equation_result_l3448_344833

theorem linear_equation_result (x m : ℝ) : 
  (∃ a b : ℝ, x^(2*m-3) + 6 = a*x + b) → (x + 3)^2010 = 1 := by
sorry

end linear_equation_result_l3448_344833


namespace line_not_intersecting_segment_l3448_344874

/-- Given points P and Q, and a line l that does not intersect line segment PQ,
    prove that the parameter m in the line equation satisfies m < -2/3 or m > 1/2 -/
theorem line_not_intersecting_segment (m : ℝ) :
  let P : ℝ × ℝ := (-1, 1)
  let Q : ℝ × ℝ := (2, 2)
  let l := {(x, y) : ℝ × ℝ | x + m * y + m = 0}
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (1 - t) • P + t • Q ∉ l) →
  m < -2/3 ∨ m > 1/2 := by
  sorry

end line_not_intersecting_segment_l3448_344874


namespace laura_has_435_l3448_344829

/-- Calculates Laura's money given Darwin's money -/
def lauras_money (darwins_money : ℕ) : ℕ :=
  let mias_money := 2 * darwins_money + 20
  let combined_money := mias_money + darwins_money
  3 * combined_money - 30

/-- Proves that Laura has $435 given the conditions -/
theorem laura_has_435 : lauras_money 45 = 435 := by
  sorry

end laura_has_435_l3448_344829


namespace bells_synchronization_l3448_344823

def church_interval : ℕ := 18
def school_interval : ℕ := 24
def daycare_interval : ℕ := 30
def library_interval : ℕ := 35

def noon_hour : ℕ := 12

theorem bells_synchronization :
  let intervals := [church_interval, school_interval, daycare_interval, library_interval]
  let lcm_minutes := Nat.lcm (Nat.lcm (Nat.lcm church_interval school_interval) daycare_interval) library_interval
  let hours_after_noon := lcm_minutes / 60
  let next_sync_hour := (noon_hour + hours_after_noon) % 24
  next_sync_hour = 6 ∧ hours_after_noon = 42 := by sorry

end bells_synchronization_l3448_344823


namespace equation_solutions_l3448_344887

theorem equation_solutions : 
  (∃ (s₁ : Set ℝ), s₁ = {x : ℝ | x^2 - 4*x = 0} ∧ s₁ = {0, 4}) ∧
  (∃ (s₂ : Set ℝ), s₂ = {x : ℝ | x^2 = -2*x + 3} ∧ s₂ = {-3, 1}) :=
by sorry

end equation_solutions_l3448_344887


namespace condition_relationship_l3448_344848

open Set

def condition_p (x : ℝ) : Prop := |x - 1| < 2
def condition_q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

def set_p : Set ℝ := {x | -1 < x ∧ x < 3}
def set_q : Set ℝ := {x | -1 < x ∧ x < 6}

theorem condition_relationship :
  (∀ x, condition_p x → x ∈ set_p) ∧
  (∀ x, condition_q x → x ∈ set_q) ∧
  set_p ⊂ set_q :=
sorry

end condition_relationship_l3448_344848


namespace system_solution_difference_l3448_344844

theorem system_solution_difference (x y : ℝ) : 
  1012 * x + 1016 * y = 1020 →
  1014 * x + 1018 * y = 1022 →
  x - y = 1.09 := by
sorry

end system_solution_difference_l3448_344844


namespace min_value_ab_l3448_344871

/-- Given b > 0 and two perpendicular lines, prove the minimum value of ab is 2 -/
theorem min_value_ab (b : ℝ) (a : ℝ) (h1 : b > 0) 
  (h2 : ∀ x y : ℝ, (b^2 + 1) * x + a * y + 2 = 0 ↔ x - b^2 * y - 1 = 0) : 
  (∀ a' b' : ℝ, b' > 0 ∧ (∀ x y : ℝ, (b'^2 + 1) * x + a' * y + 2 = 0 ↔ x - b'^2 * y - 1 = 0) → a' * b' ≥ 2) ∧ 
  (∃ a₀ b₀ : ℝ, b₀ > 0 ∧ (∀ x y : ℝ, (b₀^2 + 1) * x + a₀ * y + 2 = 0 ↔ x - b₀^2 * y - 1 = 0) ∧ a₀ * b₀ = 2) :=
sorry

end min_value_ab_l3448_344871


namespace optimal_production_plan_l3448_344892

/-- Represents the production plan for the factory -/
structure ProductionPlan where
  hoursA : ℝ  -- Hours to produce Product A
  hoursB : ℝ  -- Hours to produce Product B

/-- Calculates the total profit for a given production plan -/
def totalProfit (plan : ProductionPlan) : ℝ :=
  30 * plan.hoursA + 40 * plan.hoursB

/-- Checks if a production plan is feasible given the material constraints -/
def isFeasible (plan : ProductionPlan) : Prop :=
  3 * plan.hoursA + 2 * plan.hoursB ≤ 1200 ∧
  plan.hoursA + 2 * plan.hoursB ≤ 800 ∧
  plan.hoursA ≥ 0 ∧ plan.hoursB ≥ 0

/-- The optimal production plan -/
def optimalPlan : ProductionPlan :=
  { hoursA := 200, hoursB := 300 }

theorem optimal_production_plan :
  isFeasible optimalPlan ∧
  ∀ plan : ProductionPlan, isFeasible plan →
    totalProfit plan ≤ totalProfit optimalPlan ∧
  totalProfit optimalPlan = 18000 := by
  sorry


end optimal_production_plan_l3448_344892


namespace cube_plane_intersection_l3448_344863

-- Define a cube
def Cube : Type := Unit

-- Define a plane
def Plane : Type := Unit

-- Define the set of faces of a cube
def faces (Q : Cube) : Set Unit := sorry

-- Define the union of faces
def S (Q : Cube) : Set Unit := faces Q

-- Define the set of planes intersecting the cube
def intersecting_planes (Q : Cube) (k : ℕ) : Set Plane := sorry

-- Define the union of intersecting planes
def P (Q : Cube) (k : ℕ) : Set Unit := sorry

-- Define the set of one-third points on the edges of a cube face
def one_third_points (face : Unit) : Set Unit := sorry

-- Define the set of segments joining one-third points on the same face
def one_third_segments (Q : Cube) : Set Unit := sorry

-- State the theorem
theorem cube_plane_intersection (Q : Cube) :
  ∃ k : ℕ, 
    (∀ k' : ℕ, k' ≥ k → 
      (P Q k' ∩ S Q = one_third_segments Q) → 
      k' = k) ∧
    (∀ k' : ℕ, k' ≤ k → 
      (P Q k' ∩ S Q = one_third_segments Q) → 
      k' = k) :=
sorry

end cube_plane_intersection_l3448_344863


namespace hyperbola_eccentricity_l3448_344878

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and an asymptote x/3 + y = 0 is √10/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b/a = 1/3) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 10 / 3 := by sorry

end hyperbola_eccentricity_l3448_344878


namespace employee_pay_percentage_l3448_344810

/-- Given two employees A and B with a total weekly pay of 550 and B's pay of 220,
    prove that A's pay is 150% of B's pay. -/
theorem employee_pay_percentage (total_pay : ℝ) (b_pay : ℝ) (a_pay : ℝ)
  (h1 : total_pay = 550)
  (h2 : b_pay = 220)
  (h3 : a_pay + b_pay = total_pay) :
  a_pay / b_pay * 100 = 150 := by
sorry

end employee_pay_percentage_l3448_344810


namespace fraction_equality_l3448_344813

theorem fraction_equality (x y : ℝ) (h : (1 / x) - (1 / y) = 2) :
  (x + x*y - y) / (x - x*y - y) = 1/3 := by
  sorry

end fraction_equality_l3448_344813


namespace shaded_semicircle_perimeter_l3448_344843

/-- The perimeter of a shaded region in a semicircle -/
theorem shaded_semicircle_perimeter (r : ℝ) (h : r = 2) :
  let arc_length := π * r / 2
  let radii_length := 2 * r
  arc_length + radii_length = π + 4 := by
  sorry


end shaded_semicircle_perimeter_l3448_344843


namespace polynomial_division_quotient_l3448_344801

theorem polynomial_division_quotient (x : ℝ) :
  (x^2 + 7*x + 17) * (x - 2) + 43 = x^3 + 5*x^2 + 3*x + 9 := by
  sorry

end polynomial_division_quotient_l3448_344801


namespace faulty_passed_ratio_is_one_to_eight_l3448_344845

/-- Represents the ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the circuit board inspection results -/
structure CircuitBoardInspection where
  total : ℕ
  failed : ℕ
  faulty : ℕ

def faultyPassedRatio (inspection : CircuitBoardInspection) : Ratio :=
  { numerator := inspection.faulty - inspection.failed,
    denominator := inspection.total - inspection.failed }

theorem faulty_passed_ratio_is_one_to_eight 
  (inspection : CircuitBoardInspection) 
  (h1 : inspection.total = 3200)
  (h2 : inspection.failed = 64)
  (h3 : inspection.faulty = 456) : 
  faultyPassedRatio inspection = { numerator := 1, denominator := 8 } := by
  sorry

#check faulty_passed_ratio_is_one_to_eight

end faulty_passed_ratio_is_one_to_eight_l3448_344845


namespace negation_of_universal_proposition_l3448_344824

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℕ, n^2 < 2^n) ↔ (∃ n₀ : ℕ, n₀^2 ≥ 2^n₀) :=
sorry

end negation_of_universal_proposition_l3448_344824


namespace angle_sum_is_pi_over_two_l3448_344809

theorem angle_sum_is_pi_over_two (α β : Real)
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h4 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π/2 := by
sorry

end angle_sum_is_pi_over_two_l3448_344809


namespace ratio_problem_l3448_344855

theorem ratio_problem (a b x y : ℕ) : 
  a > b → 
  a - b = 5 → 
  a * 5 = b * 6 → 
  (a - x) * 4 = (b - x) * 5 → 
  (a + y) * 6 = (b + y) * 7 → 
  x = 5 ∧ y = 5 := by
sorry

end ratio_problem_l3448_344855
