import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l2306_230651

theorem expression_evaluation (x y z w : ℝ) :
  (x - (y - 3 * z + w)) - ((x - y + w) - 3 * z) = 6 * z - 2 * w := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2306_230651


namespace NUMINAMATH_CALUDE_tan_sum_l2306_230681

theorem tan_sum (x y : Real) 
  (h1 : Real.sin x + Real.sin y = 15/13)
  (h2 : Real.cos x + Real.cos y = 5/13)
  (h3 : Real.cos (x - y) = 4/5) :
  Real.tan x + Real.tan y = -3/5 := by sorry

end NUMINAMATH_CALUDE_tan_sum_l2306_230681


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l2306_230659

theorem restaurant_bill_calculation (total_people adults kids : ℕ) (adult_meal_cost : ℚ) :
  total_people = adults + kids →
  total_people = 12 →
  kids = 7 →
  adult_meal_cost = 3 →
  adults * adult_meal_cost = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l2306_230659


namespace NUMINAMATH_CALUDE_trees_planted_l2306_230688

def road_length : ℕ := 2575
def tree_interval : ℕ := 25

theorem trees_planted (n : ℕ) : 
  n = road_length / tree_interval + 1 → n = 104 := by
  sorry

end NUMINAMATH_CALUDE_trees_planted_l2306_230688


namespace NUMINAMATH_CALUDE_circle_center_and_sum_l2306_230669

/-- Given a circle described by the equation x^2 + y^2 = 4x - 2y + 10,
    prove that its center is at (2, -1) and the sum of the center's coordinates is 1. -/
theorem circle_center_and_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 2*y + 10) → 
  (∃ (center_x center_y : ℝ), 
    center_x = 2 ∧ 
    center_y = -1 ∧ 
    (x - center_x)^2 + (y - center_y)^2 = 15 ∧
    center_x + center_y = 1) := by
  sorry


end NUMINAMATH_CALUDE_circle_center_and_sum_l2306_230669


namespace NUMINAMATH_CALUDE_problem_statement_l2306_230653

theorem problem_statement (x y : ℝ) (h : 2 * x - y = 8) : 6 - 2 * x + y = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2306_230653


namespace NUMINAMATH_CALUDE_opposite_reciprocal_theorem_l2306_230620

theorem opposite_reciprocal_theorem (a b c d m : ℝ) 
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |m| = 2) :
  (a + b) / (4 * m) + m^2 - 3 * c * d = 1 := by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_theorem_l2306_230620


namespace NUMINAMATH_CALUDE_pool_capacity_l2306_230649

theorem pool_capacity (C : ℝ) 
  (h1 : C / 4 - C / 6 = C / 12)  -- Net rate of water level change
  (h2 : C - 3 * (C / 12) = 90)   -- Remaining water after 3 hours
  : C = 120 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l2306_230649


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_specific_case_l2306_230633

theorem least_subtraction_for_divisibility (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ (k : ℕ), k ≤ p - 1 ∧ (n - k) % p = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % p ≠ 0 :=
by
  sorry

theorem specific_case : 
  ∃ (k : ℕ), k ≤ 16 ∧ (165826 - k) % 17 = 0 ∧ ∀ (m : ℕ), m < k → (165826 - m) % 17 ≠ 0 :=
by
  sorry

#eval (165826 - 12) % 17  -- Should output 0

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_specific_case_l2306_230633


namespace NUMINAMATH_CALUDE_min_buses_needed_l2306_230643

/-- The capacity of each bus -/
def bus_capacity : ℕ := 45

/-- The total number of students to be transported -/
def total_students : ℕ := 540

/-- The minimum number of buses needed -/
def min_buses : ℕ := 12

/-- Theorem: The minimum number of buses needed to transport all students is 12 -/
theorem min_buses_needed : 
  (∀ n : ℕ, n * bus_capacity ≥ total_students → n ≥ min_buses) ∧ 
  (min_buses * bus_capacity ≥ total_students) :=
sorry

end NUMINAMATH_CALUDE_min_buses_needed_l2306_230643


namespace NUMINAMATH_CALUDE_sallys_nickels_l2306_230611

theorem sallys_nickels (initial_nickels : ℕ) (dad_nickels : ℕ) (total_nickels : ℕ) 
  (h1 : initial_nickels = 7)
  (h2 : dad_nickels = 9)
  (h3 : total_nickels = 18) :
  total_nickels - (initial_nickels + dad_nickels) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sallys_nickels_l2306_230611


namespace NUMINAMATH_CALUDE_inequality_solution_l2306_230634

theorem inequality_solution (x : ℝ) : 
  x / ((x + 3) * (x + 1)) > 0 ↔ x < -3 ∨ x > -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2306_230634


namespace NUMINAMATH_CALUDE_seven_reverse_sum_squares_l2306_230676

/-- A function that reverses a two-digit number -/
def reverse_digits (n : Nat) : Nat :=
  (n % 10) * 10 + (n / 10)

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

/-- The main theorem stating that there are exactly 7 two-digit numbers
    (including 38) where the sum of the number and its reverse is a perfect square -/
theorem seven_reverse_sum_squares :
  ∃! (list : List Nat),
    list.length = 7 ∧
    (∀ n ∈ list, 10 ≤ n ∧ n < 100) ∧
    (∀ n ∈ list, is_perfect_square (n + reverse_digits n)) ∧
    (38 ∈ list) ∧
    (∀ m, 10 ≤ m ∧ m < 100 →
      is_perfect_square (m + reverse_digits m) →
      m ∈ list) :=
  sorry

end NUMINAMATH_CALUDE_seven_reverse_sum_squares_l2306_230676


namespace NUMINAMATH_CALUDE_alice_unanswered_questions_l2306_230647

/-- Represents a scoring system for a test --/
structure ScoringSystem where
  startPoints : ℤ
  correctPoints : ℤ
  wrongPoints : ℤ
  unansweredPoints : ℤ

/-- Calculates the score based on a scoring system and the number of correct, wrong, and unanswered questions --/
def calculateScore (system : ScoringSystem) (correct wrong unanswered : ℤ) : ℤ :=
  system.startPoints + system.correctPoints * correct + system.wrongPoints * wrong + system.unansweredPoints * unanswered

theorem alice_unanswered_questions : ∃ (correct wrong unanswered : ℤ),
  let newSystem : ScoringSystem := ⟨0, 6, 0, 3⟩
  let oldSystem : ScoringSystem := ⟨50, 5, -2, 0⟩
  let hypotheticalSystem : ScoringSystem := ⟨40, 7, -1, -1⟩
  correct + wrong + unanswered = 25 ∧
  calculateScore newSystem correct wrong unanswered = 130 ∧
  calculateScore oldSystem correct wrong unanswered = 100 ∧
  calculateScore hypotheticalSystem correct wrong unanswered = 120 ∧
  unanswered = 20 := by
  sorry

#check alice_unanswered_questions

end NUMINAMATH_CALUDE_alice_unanswered_questions_l2306_230647


namespace NUMINAMATH_CALUDE_number_of_men_in_first_scenario_l2306_230677

/-- Represents the number of men working in the first scenario -/
def M : ℕ := 15

/-- Represents the number of hours worked per day in the first scenario -/
def hours_per_day_1 : ℕ := 9

/-- Represents the number of days worked in the first scenario -/
def days_1 : ℕ := 16

/-- Represents the number of men working in the second scenario -/
def men_2 : ℕ := 18

/-- Represents the number of hours worked per day in the second scenario -/
def hours_per_day_2 : ℕ := 8

/-- Represents the number of days worked in the second scenario -/
def days_2 : ℕ := 15

/-- Theorem stating that the number of men in the first scenario is 15 -/
theorem number_of_men_in_first_scenario :
  M * hours_per_day_1 * days_1 = men_2 * hours_per_day_2 * days_2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_in_first_scenario_l2306_230677


namespace NUMINAMATH_CALUDE_calculate_monthly_income_l2306_230628

/-- Calculates the total monthly income given the specified distributions and remaining amount. -/
theorem calculate_monthly_income (children_percentage : Real) (investment_percentage : Real)
  (tax_percentage : Real) (fixed_expenses : Real) (donation_percentage : Real)
  (remaining_amount : Real) :
  let total_income := (remaining_amount + fixed_expenses) /
    (1 - 3 * children_percentage - investment_percentage - tax_percentage -
     donation_percentage * (1 - 3 * children_percentage - investment_percentage - tax_percentage))
  (total_income - 3 * (children_percentage * total_income) -
   investment_percentage * total_income - tax_percentage * total_income - fixed_expenses -
   donation_percentage * (total_income - 3 * (children_percentage * total_income) -
   investment_percentage * total_income - tax_percentage * total_income - fixed_expenses)) =
  remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_calculate_monthly_income_l2306_230628


namespace NUMINAMATH_CALUDE_min_editors_at_conference_l2306_230615

theorem min_editors_at_conference (total : Nat) (writers : Nat) (x : Nat) :
  total = 100 →
  writers = 45 →
  x ≤ 18 →
  total = writers + (55 + x) - x + 2 * x →
  55 + x ≥ 73 :=
by
  sorry

end NUMINAMATH_CALUDE_min_editors_at_conference_l2306_230615


namespace NUMINAMATH_CALUDE_polynomial_value_l2306_230606

theorem polynomial_value (a b : ℝ) (h1 : a * b = 7) (h2 : a + b = 2) :
  a^2 * b + a * b^2 - 20 = -6 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_l2306_230606


namespace NUMINAMATH_CALUDE_linear_function_order_l2306_230683

/-- For a linear function f(x) = -3x + 5, prove that the y-coordinates
    of the points (1, f(1)), (-1, f(-1)), and (-2, f(-2)) are in ascending order. -/
theorem linear_function_order :
  let f : ℝ → ℝ := λ x ↦ -3 * x + 5
  let x₁ : ℝ := 1
  let x₂ : ℝ := -1
  let x₃ : ℝ := -2
  f x₁ < f x₂ ∧ f x₂ < f x₃ := by sorry

end NUMINAMATH_CALUDE_linear_function_order_l2306_230683


namespace NUMINAMATH_CALUDE_motorcycle_distance_l2306_230642

theorem motorcycle_distance (bus_speed : ℝ) (motorcycle_speed_ratio : ℝ) (time : ℝ) :
  bus_speed = 90 →
  motorcycle_speed_ratio = 2 / 3 →
  time = 1 / 2 →
  motorcycle_speed_ratio * bus_speed * time = 30 := by
  sorry

end NUMINAMATH_CALUDE_motorcycle_distance_l2306_230642


namespace NUMINAMATH_CALUDE_solution_difference_l2306_230614

theorem solution_difference (r s : ℝ) : 
  (∀ x, x ≠ 3 → (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) →
  r ≠ s →
  r > s →
  r - s = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l2306_230614


namespace NUMINAMATH_CALUDE_count_integers_eq_880_l2306_230640

/-- Fibonacci sequence with F₁ = 2 and F₂ = 3 -/
def F : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => F (n + 1) + F n

/-- The number of 10-digit integers with digits 1 or 2 and two consecutive 1's -/
def count_integers : ℕ := 2^10 - F 9

theorem count_integers_eq_880 : count_integers = 880 := by
  sorry

#eval count_integers  -- Should output 880

end NUMINAMATH_CALUDE_count_integers_eq_880_l2306_230640


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l2306_230684

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l2306_230684


namespace NUMINAMATH_CALUDE_apple_pie_apples_apple_pie_theorem_l2306_230664

theorem apple_pie_apples (total_greg_sarah : ℕ) (susan_multiplier : ℕ) (mark_difference : ℕ) (mom_leftover : ℕ) : ℕ :=
  let greg_apples := total_greg_sarah / 2
  let susan_apples := greg_apples * susan_multiplier
  let mark_apples := susan_apples - mark_difference
  let pie_apples := susan_apples - mom_leftover
  pie_apples

theorem apple_pie_theorem :
  apple_pie_apples 18 2 5 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_apples_apple_pie_theorem_l2306_230664


namespace NUMINAMATH_CALUDE_divisibility_by_42p_l2306_230638

theorem divisibility_by_42p (p : Nat) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ k : ℤ, (3^p - 2^p - 1 : ℤ) = 42 * p * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_42p_l2306_230638


namespace NUMINAMATH_CALUDE_power_negative_one_equals_half_l2306_230658

-- Define the theorem
theorem power_negative_one_equals_half : 2^(-1 : ℤ) = (1/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_power_negative_one_equals_half_l2306_230658


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2306_230607

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 166) (h2 : divisor = 20) (h3 : quotient = 8) :
  dividend % divisor = 6 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2306_230607


namespace NUMINAMATH_CALUDE_paint_area_is_134_l2306_230657

/-- The area to be painted on a wall with a window and door -/
def areaToPaint (wallHeight wallLength windowSide doorWidth doorHeight : ℝ) : ℝ :=
  wallHeight * wallLength - windowSide * windowSide - doorWidth * doorHeight

/-- Theorem: The area to be painted is 134 square feet -/
theorem paint_area_is_134 :
  areaToPaint 10 15 3 1 7 = 134 := by
  sorry

end NUMINAMATH_CALUDE_paint_area_is_134_l2306_230657


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2306_230652

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {y | ∃ x ∈ A, y = x + 1}

theorem intersection_of_A_and_B : A ∩ B = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2306_230652


namespace NUMINAMATH_CALUDE_glue_drops_in_cube_l2306_230673

/-- 
For an n × n × n cube built from n³ unit cubes, where one drop of glue is used for each pair 
of touching faces between two cubes, the total number of glue drops used is 3n²(n-1).
-/
theorem glue_drops_in_cube (n : ℕ) : 
  n > 0 → 3 * n^2 * (n - 1) = 
    (n - 1) * n * n  -- drops for vertical contacts
    + (n - 1) * n * n  -- drops for horizontal contacts
    + (n - 1) * n * n  -- drops for depth contacts
  := by sorry

end NUMINAMATH_CALUDE_glue_drops_in_cube_l2306_230673


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2306_230604

theorem fraction_equals_zero (x : ℝ) :
  (2*x - 4) / (x + 1) = 0 ∧ x + 1 ≠ 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2306_230604


namespace NUMINAMATH_CALUDE_talia_total_distance_l2306_230698

/-- Represents the total distance Talia drives in a day -/
def total_distance (home_to_park park_to_grocery grocery_to_friend friend_to_home : ℕ) : ℕ :=
  home_to_park + park_to_grocery + grocery_to_friend + friend_to_home

/-- Theorem stating that Talia drives 18 miles in total -/
theorem talia_total_distance :
  ∃ (home_to_park park_to_grocery grocery_to_friend friend_to_home : ℕ),
    home_to_park = 5 ∧
    park_to_grocery = 3 ∧
    grocery_to_friend = 6 ∧
    friend_to_home = 4 ∧
    total_distance home_to_park park_to_grocery grocery_to_friend friend_to_home = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_talia_total_distance_l2306_230698


namespace NUMINAMATH_CALUDE_head_probability_l2306_230699

/-- Represents the possible outcomes of a coin toss -/
inductive CoinOutcome
  | Head
  | Tail

/-- A fair coin toss -/
def FairCoin : Type := CoinOutcome

/-- The probability of an outcome in a fair coin toss -/
def prob (outcome : CoinOutcome) : ℚ :=
  1 / 2

theorem head_probability (c : FairCoin) :
  prob CoinOutcome.Head = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_head_probability_l2306_230699


namespace NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_l2306_230602

theorem smallest_sum_arithmetic_geometric (A B C D : ℤ) : 
  (∃ d : ℤ, C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (C * C = B * D) →           -- B, C, D form a geometric sequence
  (C = (4 * B) / 3) →         -- C/B = 4/3
  (∀ A' B' C' D' : ℤ, 
    (∃ d' : ℤ, C' - B' = B' - A') → 
    (C' * C' = B' * D') → 
    (C' = (4 * B') / 3) → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 43 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_l2306_230602


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2306_230626

theorem inequality_solution_range (m : ℝ) : 
  (∀ x : ℕ+, (x = 1 ∨ x = 2 ∨ x = 3) ↔ 3 * (x : ℝ) - m ≤ 0) → 
  (9 ≤ m ∧ m < 12) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2306_230626


namespace NUMINAMATH_CALUDE_circle_tangent_k_range_l2306_230637

/-- The range of k for a circle with two tangents from P(2,2) -/
theorem circle_tangent_k_range :
  ∀ k : ℝ,
  (∃ x y : ℝ, x^2 + y^2 - 2*k*x - 2*y + k^2 - k = 0) →
  (∃ t₁ t₂ : ℝ × ℝ, 
    (t₁.1 - 2)^2 + (t₁.2 - 2)^2 = ((2 - k)^2 + 1) ∧
    (t₂.1 - 2)^2 + (t₂.2 - 2)^2 = ((2 - k)^2 + 1) ∧
    t₁ ≠ t₂) →
  (k ∈ Set.Ioo (-1) 1 ∪ Set.Ioi 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_k_range_l2306_230637


namespace NUMINAMATH_CALUDE_distance_A_to_C_distance_A_to_C_is_300_l2306_230631

/-- The distance between city A and city C given the travel times and speeds of Eddy and Freddy -/
theorem distance_A_to_C : ℝ :=
  let eddy_time : ℝ := 3
  let freddy_time : ℝ := 4
  let distance_A_to_B : ℝ := 570
  let speed_ratio : ℝ := 2.533333333333333

  let eddy_speed : ℝ := distance_A_to_B / eddy_time
  let freddy_speed : ℝ := eddy_speed / speed_ratio
  
  freddy_speed * freddy_time

/-- The distance between city A and city C is 300 km -/
theorem distance_A_to_C_is_300 : distance_A_to_C = 300 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_to_C_distance_A_to_C_is_300_l2306_230631


namespace NUMINAMATH_CALUDE_vector_sum_coordinates_l2306_230601

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, -2)

theorem vector_sum_coordinates : 2 • a + b = (-3, 4) := by sorry

end NUMINAMATH_CALUDE_vector_sum_coordinates_l2306_230601


namespace NUMINAMATH_CALUDE_hadley_walk_l2306_230663

/-- Hadley's walk problem -/
theorem hadley_walk (x : ℝ) :
  (x ≥ 0) →
  (x - 1 ≥ 0) →
  (x + (x - 1) + 3 = 6) →
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_hadley_walk_l2306_230663


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l2306_230696

-- Define the types for lines and planes
variable (L : Type) [LinearOrderedField L]
variable (P : Type) [AddCommGroup P] [Module L P]

-- Define the perpendicular and parallel relations
variable (perpLine : L → P → Prop)  -- Line perpendicular to plane
variable (perpPlane : P → P → Prop)  -- Plane perpendicular to plane
variable (parallel : P → P → Prop)  -- Plane parallel to plane

-- State the theorem
theorem line_perp_parallel_planes 
  (l : L) (α β : P) 
  (h1 : perpLine l β) 
  (h2 : parallel α β) : 
  perpLine l α :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l2306_230696


namespace NUMINAMATH_CALUDE_sample_capacity_proof_l2306_230667

/-- Given a sample divided into groups, prove that if one group has a frequency of 30
    and a frequency rate of 0.25, then the sample capacity is 120. -/
theorem sample_capacity_proof (n : ℕ) (frequency : ℕ) (frequency_rate : ℚ) 
    (h1 : frequency = 30)
    (h2 : frequency_rate = 1/4)
    (h3 : frequency_rate = frequency / n) : n = 120 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_proof_l2306_230667


namespace NUMINAMATH_CALUDE_log_expression_equals_half_l2306_230670

theorem log_expression_equals_half :
  (1/2) * (Real.log 12 / Real.log 6) - (Real.log (Real.sqrt 2) / Real.log 6) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_half_l2306_230670


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2306_230645

/-- 
Given a quadratic function y = 3x^2 + px + q, 
prove that the value of q that makes the minimum value of y equal to 1 is 1 + p^2/18
-/
theorem quadratic_minimum (p : ℝ) : 
  ∃ (q : ℝ), (∀ (x : ℝ), 3 * x^2 + p * x + q ≥ 1) ∧ 
  (∃ (x : ℝ), 3 * x^2 + p * x + q = 1) → 
  q = 1 + p^2 / 18 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2306_230645


namespace NUMINAMATH_CALUDE_dinner_lunch_ratio_is_two_l2306_230674

/-- Represents the daily calorie intake of John -/
structure DailyCalories where
  breakfast : ℕ
  lunch : ℕ
  dinner : ℕ
  shakes : ℕ
  total : ℕ

/-- The ratio of dinner calories to lunch calories -/
def dinner_lunch_ratio (dc : DailyCalories) : ℚ :=
  dc.dinner / dc.lunch

/-- John's daily calorie intake satisfies the given conditions -/
def johns_calories : DailyCalories :=
  { breakfast := 500,
    lunch := 500 + (500 * 25 / 100),
    dinner := 3275 - (500 + (500 + (500 * 25 / 100)) + (3 * 300)),
    shakes := 3 * 300,
    total := 3275 }

theorem dinner_lunch_ratio_is_two : dinner_lunch_ratio johns_calories = 2 := by
  sorry

end NUMINAMATH_CALUDE_dinner_lunch_ratio_is_two_l2306_230674


namespace NUMINAMATH_CALUDE_tony_water_consumption_l2306_230616

theorem tony_water_consumption (yesterday : ℝ) (two_days_ago : ℝ) 
  (h1 : yesterday = 48)
  (h2 : yesterday = two_days_ago - 0.04 * two_days_ago) :
  two_days_ago = 50 := by
  sorry

end NUMINAMATH_CALUDE_tony_water_consumption_l2306_230616


namespace NUMINAMATH_CALUDE_johns_donation_size_l2306_230609

/-- Represents the donation problem with given conditions -/
structure DonationProblem where
  num_previous_donations : ℕ
  new_average : ℚ
  increase_percentage : ℚ

/-- Calculates John's donation size based on the given conditions -/
def calculate_donation_size (problem : DonationProblem) : ℚ :=
  let previous_average := problem.new_average / (1 + problem.increase_percentage)
  let total_before := previous_average * problem.num_previous_donations
  let total_after := problem.new_average * (problem.num_previous_donations + 1)
  total_after - total_before

/-- Theorem stating that John's donation size is $225 given the problem conditions -/
theorem johns_donation_size (problem : DonationProblem) 
  (h1 : problem.num_previous_donations = 6)
  (h2 : problem.new_average = 75)
  (h3 : problem.increase_percentage = 1/2) :
  calculate_donation_size problem = 225 := by
  sorry

#eval calculate_donation_size { num_previous_donations := 6, new_average := 75, increase_percentage := 1/2 }

end NUMINAMATH_CALUDE_johns_donation_size_l2306_230609


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2306_230624

/-- The line equation passes through a fixed point for all values of m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (m - 1) * (-2) - 1 + (2 * m - 1) = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2306_230624


namespace NUMINAMATH_CALUDE_number_of_hens_l2306_230617

/-- Represents the number of hens and cows a man has. -/
structure Animals where
  hens : ℕ
  cows : ℕ

/-- The total number of heads for the given animals. -/
def totalHeads (a : Animals) : ℕ := a.hens + a.cows

/-- The total number of feet for the given animals. -/
def totalFeet (a : Animals) : ℕ := 2 * a.hens + 4 * a.cows

/-- Theorem stating that given the conditions, the number of hens is 24. -/
theorem number_of_hens : 
  ∃ (a : Animals), totalHeads a = 48 ∧ totalFeet a = 144 ∧ a.hens = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_of_hens_l2306_230617


namespace NUMINAMATH_CALUDE_no_real_roots_l2306_230650

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 9) + Real.sqrt (x - 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l2306_230650


namespace NUMINAMATH_CALUDE_charles_earnings_l2306_230661

/-- Charles' earnings problem -/
theorem charles_earnings (housesitting_rate : ℝ) (housesitting_hours : ℝ) (dogs_walked : ℝ) (total_earnings : ℝ) :
  housesitting_rate = 15 →
  housesitting_hours = 10 →
  dogs_walked = 3 →
  total_earnings = 216 →
  (total_earnings - housesitting_rate * housesitting_hours) / dogs_walked = 22 := by
  sorry

end NUMINAMATH_CALUDE_charles_earnings_l2306_230661


namespace NUMINAMATH_CALUDE_only_one_student_passes_l2306_230693

theorem only_one_student_passes (prob_A prob_B prob_C : ℚ)
  (hA : prob_A = 4/5)
  (hB : prob_B = 3/5)
  (hC : prob_C = 7/10) :
  (prob_A * (1 - prob_B) * (1 - prob_C)) +
  ((1 - prob_A) * prob_B * (1 - prob_C)) +
  ((1 - prob_A) * (1 - prob_B) * prob_C) = 47/250 := by
  sorry

end NUMINAMATH_CALUDE_only_one_student_passes_l2306_230693


namespace NUMINAMATH_CALUDE_sum_first_eight_primes_mod_ninth_prime_l2306_230692

def first_nine_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23]

theorem sum_first_eight_primes_mod_ninth_prime : 
  (List.sum (List.take 8 first_nine_primes)) % (List.get! first_nine_primes 8) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_eight_primes_mod_ninth_prime_l2306_230692


namespace NUMINAMATH_CALUDE_greatest_m_value_l2306_230613

def reverse_number (n : ℕ) : ℕ := sorry

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_m_value (m : ℕ) 
  (h1 : is_four_digit m)
  (h2 : is_four_digit (reverse_number m))
  (h3 : m % 63 = 0)
  (h4 : (reverse_number m) % 63 = 0)
  (h5 : m % 11 = 0) :
  m ≤ 9811 ∧ ∃ (m : ℕ), m = 9811 ∧ 
    is_four_digit m ∧
    is_four_digit (reverse_number m) ∧
    m % 63 = 0 ∧
    (reverse_number m) % 63 = 0 ∧
    m % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_greatest_m_value_l2306_230613


namespace NUMINAMATH_CALUDE_pizzas_served_today_l2306_230625

/-- The number of pizzas served during lunch -/
def lunch_pizzas : ℕ := 9

/-- The number of pizzas served during dinner -/
def dinner_pizzas : ℕ := 6

/-- The total number of pizzas served today -/
def total_pizzas : ℕ := lunch_pizzas + dinner_pizzas

theorem pizzas_served_today : total_pizzas = 15 := by
  sorry

end NUMINAMATH_CALUDE_pizzas_served_today_l2306_230625


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2306_230660

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 17 = 0 → n ≤ 986 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2306_230660


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_ge_sum_l2306_230690

theorem gcd_lcm_sum_ge_sum (a b : ℕ+) : Nat.gcd a b + Nat.lcm a b ≥ a + b := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_ge_sum_l2306_230690


namespace NUMINAMATH_CALUDE_work_done_by_resultant_force_l2306_230687

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Calculates the dot product of two 2D vectors -/
def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Adds two 2D vectors -/
def add_vectors (v1 v2 : Vector2D) : Vector2D :=
  ⟨v1.x + v2.x, v1.y + v2.y⟩

theorem work_done_by_resultant_force : 
  let f1 : Vector2D := ⟨3, -4⟩
  let f2 : Vector2D := ⟨2, -5⟩
  let f3 : Vector2D := ⟨3, 1⟩
  let a : Vector2D := ⟨1, 1⟩
  let b : Vector2D := ⟨0, 5⟩
  let resultant_force := add_vectors (add_vectors f1 f2) f3
  let displacement := ⟨b.x - a.x, b.y - a.y⟩
  dot_product resultant_force displacement = -40 := by
  sorry

end NUMINAMATH_CALUDE_work_done_by_resultant_force_l2306_230687


namespace NUMINAMATH_CALUDE_negative_a_fourth_div_negative_a_l2306_230680

theorem negative_a_fourth_div_negative_a (a : ℝ) : (-a)^4 / (-a) = -a^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_fourth_div_negative_a_l2306_230680


namespace NUMINAMATH_CALUDE_line_passes_through_P_triangle_perimeter_l2306_230679

/-- The equation of line l is (a+1)x + y - 5 - 2a = 0, where a ∈ ℝ -/
def line_equation (a x y : ℝ) : Prop := (a + 1) * x + y - 5 - 2 * a = 0

/-- Point P that the line passes through -/
def point_P : ℝ × ℝ := (2, 3)

/-- The area of triangle AOB -/
def triangle_area : ℝ := 12

theorem line_passes_through_P (a : ℝ) : line_equation a (point_P.1) (point_P.2) := by sorry

theorem triangle_perimeter : 
  ∃ (a x_A y_B : ℝ), 
    line_equation a x_A 0 ∧ 
    line_equation a 0 y_B ∧ 
    x_A * y_B / 2 = triangle_area ∧ 
    x_A + y_B + Real.sqrt (x_A^2 + y_B^2) = 10 + 2 * Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_P_triangle_perimeter_l2306_230679


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l2306_230648

/-- The time taken for a train to pass a jogger under specific conditions -/
theorem train_passing_jogger_time (jogger_speed train_speed : ℝ) 
  (initial_distance train_length : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 240 →
  train_length = 120 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l2306_230648


namespace NUMINAMATH_CALUDE_equation_solution_l2306_230668

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (4 * (x₁ - 1)^2 = 36 ∧ 4 * (x₂ - 1)^2 = 36) ∧ 
  x₁ = 4 ∧ x₂ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2306_230668


namespace NUMINAMATH_CALUDE_flag_rectangle_ratio_l2306_230641

/-- Given a rectangle with side lengths in ratio 3:5, divided into four equal area rectangles,
    the ratio of the shorter side to the longer side of one of these rectangles is 4:15 -/
theorem flag_rectangle_ratio :
  ∀ (k : ℝ), k > 0 →
  let flag_width := 5 * k
  let flag_height := 3 * k
  let small_rect_area := (flag_width * flag_height) / 4
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    x * y = small_rect_area ∧
    3 * y = k ∧
    5 * y = x ∧
    y / x = 4 / 15 :=
by sorry

end NUMINAMATH_CALUDE_flag_rectangle_ratio_l2306_230641


namespace NUMINAMATH_CALUDE_correct_quotient_problem_l2306_230612

theorem correct_quotient_problem (D : ℕ) : 
  D % 21 = 0 →  -- The remainder is 0 when divided by 21
  D / 12 = 56 →  -- Dividing by 12 (incorrectly) gives a quotient of 56
  D / 21 = 32  -- The correct quotient when dividing by 21 is 32
:= by sorry

end NUMINAMATH_CALUDE_correct_quotient_problem_l2306_230612


namespace NUMINAMATH_CALUDE_median_salary_is_25000_l2306_230622

/-- Represents a position in the company with its title, number of employees, and salary. -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- Calculates the median salary given a list of positions and the total number of employees. -/
def medianSalary (positions : List Position) (totalEmployees : Nat) : Nat :=
  sorry

/-- The list of positions in the company. -/
def companyPositions : List Position := [
  { title := "President", count := 1, salary := 140000 },
  { title := "Vice-President", count := 4, salary := 95000 },
  { title := "Director", count := 11, salary := 78000 },
  { title := "Associate Director", count := 8, salary := 55000 },
  { title := "Administrative Specialist", count := 39, salary := 25000 }
]

/-- The total number of employees in the company. -/
def totalEmployees : Nat := 63

/-- Theorem stating that the median salary of the company is $25,000. -/
theorem median_salary_is_25000 : 
  medianSalary companyPositions totalEmployees = 25000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_25000_l2306_230622


namespace NUMINAMATH_CALUDE_equation_solution_l2306_230655

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -4 ∧ x₂ = -2) ∧ 
  (∀ x : ℝ, (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2306_230655


namespace NUMINAMATH_CALUDE_total_amount_l2306_230623

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℚ
  y : ℚ
  z : ℚ

/-- The problem setup -/
def problem_setup (s : ShareDistribution) : Prop :=
  s.y = 18 ∧ s.y = 0.45 * s.x ∧ s.z = 0.3 * s.x

/-- The theorem statement -/
theorem total_amount (s : ShareDistribution) : 
  problem_setup s → s.x + s.y + s.z = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_l2306_230623


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2306_230686

def A : Set ℕ := {0,1,2,3,4,6,7}
def B : Set ℕ := {1,2,4,8,0}

theorem intersection_of_A_and_B : A ∩ B = {1,2,4,0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2306_230686


namespace NUMINAMATH_CALUDE_union_of_sets_l2306_230666

theorem union_of_sets : 
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | 0 < x ∧ x < 2}
  A ∪ B = {x : ℝ | -2 < x ∧ x < 2} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2306_230666


namespace NUMINAMATH_CALUDE_max_value_of_function_max_value_attained_l2306_230694

theorem max_value_of_function (x : ℝ) : 
  (3 * Real.sin x + 2 * Real.sqrt (2 + 2 * Real.cos (2 * x))) ≤ 5 := by
  sorry

theorem max_value_attained (x : ℝ) : 
  ∃ x, 3 * Real.sin x + 2 * Real.sqrt (2 + 2 * Real.cos (2 * x)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_max_value_attained_l2306_230694


namespace NUMINAMATH_CALUDE_multiple_solutions_exist_l2306_230685

def f (x : ℝ) : ℝ := x^2 - 2*x

theorem multiple_solutions_exist :
  ∃ (c₁ c₂ : ℝ), c₁ ≠ c₂ ∧ f (f (f (f c₁))) = 2 ∧ f (f (f (f c₂))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_multiple_solutions_exist_l2306_230685


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2306_230608

/-- The function f(x) = a^(x+2) - 3 passes through the point (-2, -2) for a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2) - 3
  f (-2) = -2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2306_230608


namespace NUMINAMATH_CALUDE_third_stick_length_l2306_230662

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem third_stick_length (x : ℕ) 
  (h1 : is_even x)
  (h2 : x + 10 > 2)
  (h3 : x + 2 > 10)
  (h4 : 10 + 2 > x) : 
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_third_stick_length_l2306_230662


namespace NUMINAMATH_CALUDE_solution_set_f_max_value_g_l2306_230656

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Define the function g(x)
def g (x : ℝ) : ℝ := f x - x^2 + x

-- Theorem 1: Solution set of f(x) ≥ 1
theorem solution_set_f (x : ℝ) : f x ≥ 1 ↔ x ≥ 1 := by sorry

-- Theorem 2: Maximum value of g(x)
theorem max_value_g : ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 5/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_max_value_g_l2306_230656


namespace NUMINAMATH_CALUDE_parabola_focus_l2306_230697

/-- A parabola is defined by the equation y = x^2 -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- The focus of a parabola is a point with specific properties -/
def IsFocus (f : ℝ × ℝ) (p : Set (ℝ × ℝ)) : Prop :=
  ∃ (a : ℝ), p = {point : ℝ × ℝ | point.2 = point.1^2} ∧ f = (0, 1/(4*a))

/-- The theorem states that the focus of the parabola y = x^2 is at (0, 1/4) -/
theorem parabola_focus : IsFocus (0, 1/4) Parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l2306_230697


namespace NUMINAMATH_CALUDE_three_digit_sum_property_divisibility_condition_l2306_230619

def three_digit_num (x y z : ℕ) : ℕ := 100 * x + 10 * y + z

theorem three_digit_sum_property (a b c : ℕ) 
  (ha : a ≥ 1 ∧ a ≤ 9) (hb : b ≥ 1 ∧ b ≤ 9) (hc : c ≥ 1 ∧ c ≤ 9) :
  three_digit_num a b c + three_digit_num b c a + three_digit_num c a b = 111 * (a + b + c) :=
sorry

theorem divisibility_condition (a b c : ℕ) 
  (ha : a ≥ 1 ∧ a ≤ 9) (hb : b ≥ 1 ∧ b ≤ 9) (hc : c ≥ 1 ∧ c ≤ 9) :
  (∃ k : ℕ, three_digit_num a b c + three_digit_num b c a + three_digit_num c a b = 7 * k) →
  (a + b + c = 7 ∨ a + b + c = 14 ∨ a + b + c = 21) :=
sorry

end NUMINAMATH_CALUDE_three_digit_sum_property_divisibility_condition_l2306_230619


namespace NUMINAMATH_CALUDE_percentage_problem_l2306_230618

theorem percentage_problem (P : ℝ) : P * 300 - 70 = 20 → P = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2306_230618


namespace NUMINAMATH_CALUDE_expenditure_recording_l2306_230639

-- Define a type for financial transactions
inductive Transaction
| Income (amount : ℤ)
| Expenditure (amount : ℤ)

-- Define a function to record transactions
def record_transaction (t : Transaction) : ℤ :=
  match t with
  | Transaction.Income a => a
  | Transaction.Expenditure a => -a

-- Theorem statement
theorem expenditure_recording (income_amount expenditure_amount : ℤ) 
  (h1 : income_amount > 0) (h2 : expenditure_amount > 0) :
  record_transaction (Transaction.Income income_amount) = income_amount ∧
  record_transaction (Transaction.Expenditure expenditure_amount) = -expenditure_amount :=
by sorry

end NUMINAMATH_CALUDE_expenditure_recording_l2306_230639


namespace NUMINAMATH_CALUDE_min_b_over_a_l2306_230665

theorem min_b_over_a (a b : ℝ) (h : ∀ x > -1, Real.log (x + 1) - 1 ≤ a * x + b) : 
  (∀ c : ℝ, (∀ x > -1, Real.log (x + 1) - 1 ≤ a * x + c) → b / a ≤ c / a) → b / a = 1 - Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_b_over_a_l2306_230665


namespace NUMINAMATH_CALUDE_max_value_theorem_l2306_230600

theorem max_value_theorem (x y : ℝ) (h : 2 * x^2 + y^2 = 6 * x) :
  ∃ (max_val : ℝ), max_val = 15 ∧ ∀ (a b : ℝ), 2 * a^2 + b^2 = 6 * a → a^2 + b^2 + 2 * a ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2306_230600


namespace NUMINAMATH_CALUDE_half_inequality_l2306_230605

theorem half_inequality (a b : ℝ) (h : a > b) : a / 2 > b / 2 := by
  sorry

end NUMINAMATH_CALUDE_half_inequality_l2306_230605


namespace NUMINAMATH_CALUDE_clock_cost_price_l2306_230671

/-- The cost price of each clock satisfies the given conditions -/
theorem clock_cost_price (total_clocks : ℕ) (sold_at_10_percent : ℕ) (sold_at_20_percent : ℕ) 
  (uniform_profit_percentage : ℚ) (price_difference : ℚ) :
  total_clocks = 90 →
  sold_at_10_percent = 40 →
  sold_at_20_percent = 50 →
  uniform_profit_percentage = 15 / 100 →
  price_difference = 40 →
  ∃ (cost_price : ℚ),
    cost_price = 80 ∧
    cost_price * (sold_at_10_percent * (1 + 10 / 100) + sold_at_20_percent * (1 + 20 / 100)) =
    cost_price * total_clocks * (1 + uniform_profit_percentage) + price_difference :=
by sorry

end NUMINAMATH_CALUDE_clock_cost_price_l2306_230671


namespace NUMINAMATH_CALUDE_combined_selling_price_l2306_230603

/-- Calculate the combined selling price of two articles given their costs, desired profits, tax rate, and packaging fees. -/
theorem combined_selling_price
  (cost_A cost_B : ℚ)
  (profit_rate_A profit_rate_B : ℚ)
  (tax_rate : ℚ)
  (packaging_fee : ℚ) :
  cost_A = 500 →
  cost_B = 800 →
  profit_rate_A = 1/10 →
  profit_rate_B = 3/20 →
  tax_rate = 1/20 →
  packaging_fee = 50 →
  ∃ (selling_price : ℚ),
    selling_price = 
      (cost_A + cost_A * profit_rate_A) * (1 + tax_rate) + packaging_fee +
      (cost_B + cost_B * profit_rate_B) * (1 + tax_rate) + packaging_fee ∧
    selling_price = 1643.5 := by
  sorry

#check combined_selling_price

end NUMINAMATH_CALUDE_combined_selling_price_l2306_230603


namespace NUMINAMATH_CALUDE_difference_of_squares_l2306_230610

theorem difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2306_230610


namespace NUMINAMATH_CALUDE_n_value_l2306_230621

-- Define the cubic polynomial
def cubic_poly (x m : ℝ) : ℝ := x^3 - 3*x^2 + m*x + 24

-- Define the quadratic polynomial
def quad_poly (x n : ℝ) : ℝ := x^2 + n*x - 6

theorem n_value (a b c m n : ℝ) : 
  (cubic_poly a m = 0) ∧ 
  (cubic_poly b m = 0) ∧ 
  (cubic_poly c m = 0) ∧
  (quad_poly (-a) n = 0) ∧ 
  (quad_poly (-b) n = 0) →
  n = -1 := by
sorry

end NUMINAMATH_CALUDE_n_value_l2306_230621


namespace NUMINAMATH_CALUDE_composition_of_odd_functions_l2306_230630

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem composition_of_odd_functions (f : ℝ → ℝ) (h : IsOdd f) :
  IsOdd (fun x ↦ f (f (f (f x)))) := by sorry

end NUMINAMATH_CALUDE_composition_of_odd_functions_l2306_230630


namespace NUMINAMATH_CALUDE_nested_fraction_equals_27_over_73_l2306_230675

theorem nested_fraction_equals_27_over_73 :
  1 / (3 - 1 / (3 + 1 / (3 - 1 / 3))) = 27 / 73 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equals_27_over_73_l2306_230675


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2306_230636

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∃ (z : ℂ), z = (a^2 + a - 2 : ℝ) + (a^2 - 3*a + 2 : ℝ)*I ∧ z.re = 0 ∧ z.im ≠ 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2306_230636


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2306_230678

theorem negation_of_proposition (p : Prop) : 
  (¬(∀ (x : ℝ), x ≥ 0 → Real.log (x^2 + 1) ≥ 0)) ↔ 
  (∃ (x : ℝ), x ≥ 0 ∧ Real.log (x^2 + 1) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2306_230678


namespace NUMINAMATH_CALUDE_cheese_distribution_l2306_230627

theorem cheese_distribution (M : ℝ) (x y : ℝ) : 
  -- Total cheese weight
  M > 0 →
  -- White's slice is exactly one-quarter of the total
  y = M / 4 →
  -- Thin's slice weighs x
  -- Fat's slice weighs x + 20
  -- White's slice weighs y
  -- Gray's slice weighs y + 8
  x + (x + 20) + y + (y + 8) = M →
  -- Gray cuts 8 grams, Fat cuts 20 grams
  -- To achieve equal distribution, Fat and Thin should each get 14 grams
  14 = (28 : ℝ) / 2 ∧
  x + 14 = y ∧
  (x + 20) - 20 + 14 = y ∧
  (y + 8) - 8 = y :=
by
  sorry

end NUMINAMATH_CALUDE_cheese_distribution_l2306_230627


namespace NUMINAMATH_CALUDE_derivative_at_one_l2306_230691

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) * (x - 1)

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2306_230691


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l2306_230672

/-- 
Given three points (x₁, y₁), (x₂, y₂), and (x₃, y₃) in ℝ², 
this function returns true if they are collinear (lie on the same line).
-/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- 
Theorem: If the points (7, 10), (1, k), and (-8, 5) are collinear, then k = 40.
-/
theorem collinear_points_k_value : 
  collinear 7 10 1 k (-8) 5 → k = 40 := by
  sorry


end NUMINAMATH_CALUDE_collinear_points_k_value_l2306_230672


namespace NUMINAMATH_CALUDE_ice_cream_sandwiches_l2306_230695

theorem ice_cream_sandwiches (nieces : ℕ) (sandwiches_per_niece : ℕ) 
  (h1 : nieces = 11) (h2 : sandwiches_per_niece = 13) : 
  nieces * sandwiches_per_niece = 143 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sandwiches_l2306_230695


namespace NUMINAMATH_CALUDE_trapezoid_area_l2306_230682

/-- A trapezoid with given base and diagonal lengths has area 80 -/
theorem trapezoid_area (a b d₁ d₂ : ℝ) (ha : a = 5) (hb : b = 15) (hd₁ : d₁ = 12) (hd₂ : d₂ = 16) :
  ∃ h : ℝ, (a + b) * h / 2 = 80 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2306_230682


namespace NUMINAMATH_CALUDE_largest_and_smallest_A_l2306_230632

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def move_last_digit_to_front (n : ℕ) : ℕ :=
  let d := n % 10
  let r := n / 10
  d * 10^7 + r

def satisfies_conditions (b : ℕ) : Prop :=
  b > 44444444 ∧ is_coprime b 12

theorem largest_and_smallest_A :
  ∃ (a_max a_min : ℕ),
    (∀ a b : ℕ, 
      a = move_last_digit_to_front b ∧ 
      satisfies_conditions b →
      a ≤ a_max ∧ a ≥ a_min) ∧
    a_max = 99999998 ∧
    a_min = 14444446 :=
sorry

end NUMINAMATH_CALUDE_largest_and_smallest_A_l2306_230632


namespace NUMINAMATH_CALUDE_least_multiple_of_15_greater_than_500_l2306_230644

theorem least_multiple_of_15_greater_than_500 : 
  (∃ (n : ℕ), 15 * n = 510 ∧ 
   510 > 500 ∧ 
   ∀ (m : ℕ), 15 * m > 500 → 15 * m ≥ 510) := by
sorry

end NUMINAMATH_CALUDE_least_multiple_of_15_greater_than_500_l2306_230644


namespace NUMINAMATH_CALUDE_emily_spent_12_dollars_l2306_230654

def flower_cost : ℕ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2

theorem emily_spent_12_dollars :
  flower_cost * (roses_bought + daisies_bought) = 12 := by
  sorry

end NUMINAMATH_CALUDE_emily_spent_12_dollars_l2306_230654


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l2306_230629

/-- A regular polygon with interior angles measuring 150 degrees has 12 sides. -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ θ : ℝ, θ = 150 → (n : ℝ) * θ = 180 * ((n : ℝ) - 2)) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l2306_230629


namespace NUMINAMATH_CALUDE_factor_sum_l2306_230635

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 50 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l2306_230635


namespace NUMINAMATH_CALUDE_inequality_solution_l2306_230646

theorem inequality_solution (x : ℝ) :
  (x - 1) * (x - 4) * (x - 5) * (x - 7) / ((x - 3) * (x - 6) * (x - 8) * (x - 9)) > 0 →
  |x - 2| ≥ 1 →
  x ∈ Set.Ioo 3 4 ∪ Set.Ioo 6 7 ∪ Set.Ioo 8 9 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2306_230646


namespace NUMINAMATH_CALUDE_root_transformation_l2306_230689

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - r₁^2 + 3*r₁ - 7 = 0) ∧ 
  (r₂^3 - r₂^2 + 3*r₂ - 7 = 0) ∧ 
  (r₃^3 - r₃^2 + 3*r₃ - 7 = 0) →
  ((3*r₁)^3 - 3*(3*r₁)^2 + 27*(3*r₁) - 189 = 0) ∧ 
  ((3*r₂)^3 - 3*(3*r₂)^2 + 27*(3*r₂) - 189 = 0) ∧ 
  ((3*r₃)^3 - 3*(3*r₃)^2 + 27*(3*r₃) - 189 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l2306_230689
