import Mathlib

namespace NUMINAMATH_CALUDE_largest_reciprocal_l1771_177167

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/4 → b = 3/8 → c = 0 → d = -2 → e = 4 → 
  (1/a > 1/b ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

#check largest_reciprocal

end NUMINAMATH_CALUDE_largest_reciprocal_l1771_177167


namespace NUMINAMATH_CALUDE_number_of_boys_l1771_177195

/-- Proves that the number of boys in a group is 5 given specific conditions about weights --/
theorem number_of_boys (num_girls : ℕ) (num_total : ℕ) (avg_girls : ℚ) (avg_boys : ℚ) (avg_total : ℚ) :
  num_girls = 5 →
  num_total = 10 →
  avg_girls = 45 →
  avg_boys = 55 →
  avg_total = 50 →
  ∃ (num_boys : ℕ), num_boys = 5 ∧ num_girls + num_boys = num_total ∧
    (num_girls : ℚ) * avg_girls + (num_boys : ℚ) * avg_boys = (num_total : ℚ) * avg_total :=
by
  sorry


end NUMINAMATH_CALUDE_number_of_boys_l1771_177195


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l1771_177114

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A heptagon has 7 sides -/
def heptagon_sides : ℕ := 7

theorem heptagon_diagonals : diagonals heptagon_sides = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l1771_177114


namespace NUMINAMATH_CALUDE_smallest_n_guarantee_same_length_l1771_177163

/-- The number of vertices in the regular polygon -/
def n : ℕ := 2017

/-- The number of distinct diagonal lengths from a single vertex -/
def distinct_lengths : ℕ := (n - 3) / 2

/-- The smallest number of diagonals to guarantee two of the same length -/
def smallest_n : ℕ := distinct_lengths + 1

theorem smallest_n_guarantee_same_length :
  smallest_n = 1008 := by sorry

end NUMINAMATH_CALUDE_smallest_n_guarantee_same_length_l1771_177163


namespace NUMINAMATH_CALUDE_parabola_axis_equation_l1771_177107

/-- Given a parabola with equation x = (1/4)y^2, its axis equation is x = -1 -/
theorem parabola_axis_equation (y : ℝ) :
  let x := (1/4) * y^2
  (∃ p : ℝ, p/2 = 1) → (x = -1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_axis_equation_l1771_177107


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1771_177120

theorem complex_equation_solution (z₁ z₂ : ℂ) : 
  z₁ = 1 - I ∧ z₁ * z₂ = 1 + I → z₂ = I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1771_177120


namespace NUMINAMATH_CALUDE_intersection_M_N_l1771_177181

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | ∃ y, y = Real.log (x - x^2)}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1771_177181


namespace NUMINAMATH_CALUDE_subset_of_A_l1771_177118

def A : Set ℝ := {x | x > -1}

theorem subset_of_A : {0} ⊆ A := by sorry

end NUMINAMATH_CALUDE_subset_of_A_l1771_177118


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1771_177151

theorem partial_fraction_decomposition :
  ∃ (P Q : ℝ), P = 5.5 ∧ Q = 1.5 ∧
  ∀ x : ℝ, x ≠ 12 → x ≠ -4 →
    (7 * x + 4) / (x^2 - 8*x - 48) = P / (x - 12) + Q / (x + 4) :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1771_177151


namespace NUMINAMATH_CALUDE_geometric_sum_remainder_l1771_177139

theorem geometric_sum_remainder (n : ℕ) (a r : ℤ) (m : ℕ) (h : m > 0) :
  (a * (r^(n+1) - 1)) / (r - 1) % m = 91 :=
by
  sorry

#check geometric_sum_remainder 2002 1 9 500

end NUMINAMATH_CALUDE_geometric_sum_remainder_l1771_177139


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l1771_177103

theorem smallest_gcd_bc (a b c : ℕ+) (hab : Nat.gcd a b = 168) (hac : Nat.gcd a c = 693) :
  ∃ (d : ℕ), d = Nat.gcd b c ∧ d ≥ 21 ∧ ∀ (e : ℕ), e = Nat.gcd b c → e ≥ d :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l1771_177103


namespace NUMINAMATH_CALUDE_chocolate_difference_l1771_177197

/-- The number of chocolates Nick has -/
def nick_chocolates : ℕ := 10

/-- The factor by which Alix's chocolates exceed Nick's -/
def alix_factor : ℕ := 3

/-- The number of chocolates mom took from Alix -/
def mom_took : ℕ := 5

/-- The number of chocolates Alix has after mom took some -/
def alix_chocolates : ℕ := alix_factor * nick_chocolates - mom_took

theorem chocolate_difference : alix_chocolates - nick_chocolates = 15 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_difference_l1771_177197


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1771_177142

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = Complex.I → z = (1/2 : ℂ) + (1/2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1771_177142


namespace NUMINAMATH_CALUDE_fourth_root_equality_l1771_177106

theorem fourth_root_equality (M : ℝ) (h : M > 1) :
  (M^2 * (M * M^(1/4))^(1/3))^(1/4) = M^(29/48) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equality_l1771_177106


namespace NUMINAMATH_CALUDE_intersection_equals_closed_open_interval_l1771_177168

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x : ℝ | x ≥ 3}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_closed_open_interval :
  A_intersect_B = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_equals_closed_open_interval_l1771_177168


namespace NUMINAMATH_CALUDE_distance_to_incenter_in_isosceles_right_triangle_l1771_177150

/-- An isosceles right triangle with hypotenuse length 6 -/
structure IsoscelesRightTriangle where
  -- A is the right angle
  AB : ℝ
  BC : ℝ
  AC : ℝ
  isIsosceles : AB = BC
  isRight : AC = 6

/-- The incenter of a triangle -/
def incenter (t : IsoscelesRightTriangle) : ℝ × ℝ := sorry

/-- The distance from a vertex to the incenter -/
def distanceToIncenter (t : IsoscelesRightTriangle) : ℝ := sorry

theorem distance_to_incenter_in_isosceles_right_triangle (t : IsoscelesRightTriangle) :
  distanceToIncenter t = 6 * Real.sqrt 2 - 6 := by sorry

end NUMINAMATH_CALUDE_distance_to_incenter_in_isosceles_right_triangle_l1771_177150


namespace NUMINAMATH_CALUDE_triangle_problem_l1771_177119

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a = 2 * Real.sqrt 3 →
  C = π / 3 →
  Real.tan A = 3 / 4 →
  (Real.sin A = 3 / 5 ∧ b = 4 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1771_177119


namespace NUMINAMATH_CALUDE_equation_solvable_by_factoring_l1771_177155

/-- The equation to be solved -/
def equation (x : ℝ) : Prop := (5*x - 1)^2 = 3*(5*x - 1)

/-- Factoring method can be applied if the equation can be written as a product of factors equal to zero -/
def factoring_method_applicable (f : ℝ → Prop) : Prop :=
  ∃ g h : ℝ → ℝ, ∀ x, f x ↔ g x * h x = 0

/-- The theorem stating that the given equation can be solved using the factoring method -/
theorem equation_solvable_by_factoring : factoring_method_applicable equation := by
  sorry

end NUMINAMATH_CALUDE_equation_solvable_by_factoring_l1771_177155


namespace NUMINAMATH_CALUDE_equation_solution_l1771_177186

theorem equation_solution :
  let f (x : ℝ) := (x - 4)^4 + (x - 6)^4
  ∃ x₁ x₂ : ℝ, 
    (f x₁ = 240 ∧ f x₂ = 240) ∧
    x₁ = 5 + Real.sqrt (5 * Real.sqrt 2 - 3) ∧
    x₂ = 5 - Real.sqrt (5 * Real.sqrt 2 - 3) ∧
    ∀ x : ℝ, f x = 240 → (x = x₁ ∨ x = x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1771_177186


namespace NUMINAMATH_CALUDE_equation_equivalence_l1771_177102

theorem equation_equivalence (x y z : ℝ) :
  (x - z)^2 - 4*(x - y)*(y - z) = 0 → z + x - 2*y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1771_177102


namespace NUMINAMATH_CALUDE_ned_second_table_trays_l1771_177101

/-- The number of trays Ned can carry at a time -/
def trays_per_trip : ℕ := 8

/-- The number of trips Ned made -/
def total_trips : ℕ := 4

/-- The number of trays Ned picked up from the first table -/
def trays_first_table : ℕ := 27

/-- The number of trays Ned picked up from the second table -/
def trays_second_table : ℕ := total_trips * trays_per_trip - trays_first_table

theorem ned_second_table_trays : trays_second_table = 5 := by
  sorry

end NUMINAMATH_CALUDE_ned_second_table_trays_l1771_177101


namespace NUMINAMATH_CALUDE_log_expression_equals_21_l1771_177183

theorem log_expression_equals_21 :
  2 * Real.log 25 / Real.log 5 + 3 * Real.log 64 / Real.log 2 - Real.log (Real.log (3^10) / Real.log 3) = 21 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_21_l1771_177183


namespace NUMINAMATH_CALUDE_badminton_probabilities_l1771_177188

/-- Represents the state of a badminton game -/
structure BadmintonGame where
  score_a : Nat
  score_b : Nat
  a_serving : Bool

/-- Rules for winning a badminton game -/
def game_won (game : BadmintonGame) : Bool :=
  (game.score_a = 21 && game.score_b < 20) ||
  (game.score_b = 21 && game.score_a < 20) ||
  (game.score_a ≥ 20 && game.score_b ≥ 20 && 
   ((game.score_a = 30) || (game.score_b = 30) || 
    (game.score_a ≥ 22 && game.score_a - game.score_b = 2) ||
    (game.score_b ≥ 22 && game.score_b - game.score_a = 2)))

/-- Probability of player A winning a rally when serving -/
def p_a_serving : ℝ := 0.4

/-- Probability of player A winning a rally when not serving -/
def p_a_not_serving : ℝ := 0.5

/-- The initial game state at 28:28 with A serving -/
def initial_state : BadmintonGame := ⟨28, 28, true⟩

theorem badminton_probabilities :
  let p_game_ends_in_two : ℝ := 0.46
  let p_a_wins : ℝ := 0.4
  (∃ (p_game_ends_in_two' p_a_wins' : ℝ),
    p_game_ends_in_two' = p_game_ends_in_two ∧
    p_a_wins' = p_a_wins ∧
    p_game_ends_in_two' = p_a_serving * p_a_serving + (1 - p_a_serving) * (1 - p_a_not_serving) ∧
    p_a_wins' = p_a_serving * p_a_serving + 
                p_a_serving * (1 - p_a_serving) * p_a_not_serving +
                (1 - p_a_serving) * p_a_not_serving * p_a_serving) :=
by sorry

end NUMINAMATH_CALUDE_badminton_probabilities_l1771_177188


namespace NUMINAMATH_CALUDE_school_seminar_cost_l1771_177135

/-- Calculates the total amount spent by a school for a seminar with discounts and food allowance -/
theorem school_seminar_cost
  (regular_fee : ℝ)
  (discount_percent : ℝ)
  (num_teachers : ℕ)
  (food_allowance : ℝ)
  (h1 : regular_fee = 150)
  (h2 : discount_percent = 5)
  (h3 : num_teachers = 10)
  (h4 : food_allowance = 10)
  : (1 - discount_percent / 100) * regular_fee * num_teachers + food_allowance * num_teachers = 1525 := by
  sorry

#check school_seminar_cost

end NUMINAMATH_CALUDE_school_seminar_cost_l1771_177135


namespace NUMINAMATH_CALUDE_quadratic_shift_l1771_177104

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 1

/-- Shift a function to the left -/
def shiftLeft (g : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := λ x ↦ g (x + d)

/-- Shift a function down -/
def shiftDown (g : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := λ x ↦ g x - d

/-- The resulting function after shifts -/
def g (x : ℝ) : ℝ := (x + 1)^2 - 2

theorem quadratic_shift :
  shiftDown (shiftLeft f 2) 3 = g := by sorry

end NUMINAMATH_CALUDE_quadratic_shift_l1771_177104


namespace NUMINAMATH_CALUDE_original_price_calculation_l1771_177154

/-- Proves that given an article sold at a 30% profit with a selling price of 715, 
    the original price (cost price) of the article is 550. -/
theorem original_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
    (h1 : selling_price = 715)
    (h2 : profit_percentage = 30) : 
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 + profit_percentage / 100) ∧ 
    original_price = 550 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1771_177154


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1771_177138

theorem fraction_equivalence :
  (14 / 12 : ℚ) = 7 / 6 ∧
  (1 + 1 / 6 : ℚ) = 7 / 6 ∧
  (1 + 5 / 30 : ℚ) = 7 / 6 ∧
  (1 + 2 / 6 : ℚ) ≠ 7 / 6 ∧
  (1 + 14 / 42 : ℚ) = 7 / 6 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1771_177138


namespace NUMINAMATH_CALUDE_fermat_like_congruence_l1771_177128

theorem fermat_like_congruence (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) : 
  let n : ℕ := (2^(2*p) - 1) / 3
  2^n - 2 ≡ 0 [MOD n] := by
  sorry

end NUMINAMATH_CALUDE_fermat_like_congruence_l1771_177128


namespace NUMINAMATH_CALUDE_point_location_l1771_177113

theorem point_location (x y : ℝ) : 
  (4 * x + 7 * y = 28) →  -- Line equation
  (abs x = abs y) →       -- Equidistant from axes
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=  -- In quadrant I or II
by sorry

end NUMINAMATH_CALUDE_point_location_l1771_177113


namespace NUMINAMATH_CALUDE_resort_worker_tips_l1771_177146

theorem resort_worker_tips (total_months : ℕ) (specific_month_multiplier : ℕ) :
  total_months = 7 ∧ specific_month_multiplier = 10 →
  (specific_month_multiplier : ℚ) / ((total_months - 1 : ℕ) + specific_month_multiplier : ℚ) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_resort_worker_tips_l1771_177146


namespace NUMINAMATH_CALUDE_larger_number_problem_l1771_177145

theorem larger_number_problem (x y : ℤ) 
  (h1 : y = x + 10) 
  (h2 : x + y = 34) : 
  y = 22 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1771_177145


namespace NUMINAMATH_CALUDE_banana_bread_theorem_l1771_177180

/-- The number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

/-- The number of loaves of banana bread made on Monday -/
def monday_loaves : ℕ := 3

/-- The number of loaves of banana bread made on Tuesday -/
def tuesday_loaves : ℕ := 2 * monday_loaves

/-- The total number of bananas used for banana bread on both days -/
def total_bananas : ℕ := bananas_per_loaf * (monday_loaves + tuesday_loaves)

theorem banana_bread_theorem : total_bananas = 36 := by
  sorry

end NUMINAMATH_CALUDE_banana_bread_theorem_l1771_177180


namespace NUMINAMATH_CALUDE_quadratic_root_arithmetic_sequence_l1771_177115

/-- Given real numbers a, b, c forming an arithmetic sequence with a ≥ b ≥ c ≥ 0,
    if the quadratic ax^2 + bx + c has exactly one root, then this root is -2 + √3 -/
theorem quadratic_root_arithmetic_sequence (a b c : ℝ) : 
  (∃ d : ℝ, b = a - d ∧ c = a - 2*d) →  -- arithmetic sequence
  a ≥ b ∧ b ≥ c ∧ c ≥ 0 →  -- ordering condition
  (∃! x : ℝ, a*x^2 + b*x + c = 0) →  -- exactly one root
  (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ x = -2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_arithmetic_sequence_l1771_177115


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l1771_177124

theorem twenty_five_percent_less_than_80 (x : ℝ) : x + (1/3) * x = 60 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l1771_177124


namespace NUMINAMATH_CALUDE_jack_cell_phone_cost_l1771_177122

/- Define the cell phone plan parameters -/
def base_cost : ℝ := 25
def text_cost : ℝ := 0.08
def extra_minute_cost : ℝ := 0.10
def free_hours : ℝ := 25

/- Define Jack's usage -/
def texts_sent : ℕ := 150
def hours_talked : ℝ := 26

/- Calculate the total cost -/
def total_cost : ℝ :=
  base_cost +
  (↑texts_sent * text_cost) +
  ((hours_talked - free_hours) * 60 * extra_minute_cost)

/- Theorem to prove -/
theorem jack_cell_phone_cost : total_cost = 43 := by
  sorry

end NUMINAMATH_CALUDE_jack_cell_phone_cost_l1771_177122


namespace NUMINAMATH_CALUDE_derivative_of_even_is_odd_l1771_177192

/-- If a real-valued function is even, then its derivative is odd. -/
theorem derivative_of_even_is_odd (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_even : ∀ x, f (-x) = f x) :
  ∀ x, deriv f (-x) = -deriv f x := by sorry

end NUMINAMATH_CALUDE_derivative_of_even_is_odd_l1771_177192


namespace NUMINAMATH_CALUDE_value_added_to_numbers_l1771_177125

theorem value_added_to_numbers (n : ℕ) (initial_avg final_avg x : ℚ) : 
  n = 15 → initial_avg = 40 → final_avg = 52 → 
  n * final_avg = n * initial_avg + n * x → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_numbers_l1771_177125


namespace NUMINAMATH_CALUDE_quadratic_form_h_l1771_177131

theorem quadratic_form_h (a k : ℝ) (h : ℝ) :
  (∀ x, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) →
  h = -3/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_h_l1771_177131


namespace NUMINAMATH_CALUDE_range_of_3a_minus_2b_l1771_177164

theorem range_of_3a_minus_2b (a b : ℝ) 
  (h1 : 1 ≤ a - b) (h2 : a - b ≤ 2) 
  (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) : 
  ∃ (x : ℝ), (7/2 ≤ x ∧ x ≤ 7) ∧ (∃ (a' b' : ℝ), 
    (1 ≤ a' - b' ∧ a' - b' ≤ 2) ∧ 
    (2 ≤ a' + b' ∧ a' + b' ≤ 4) ∧ 
    (3 * a' - 2 * b' = x)) ∧
  (∀ (y : ℝ), (∃ (a'' b'' : ℝ), 
    (1 ≤ a'' - b'' ∧ a'' - b'' ≤ 2) ∧ 
    (2 ≤ a'' + b'' ∧ a'' + b'' ≤ 4) ∧ 
    (3 * a'' - 2 * b'' = y)) → 
    (7/2 ≤ y ∧ y ≤ 7)) := by
  sorry


end NUMINAMATH_CALUDE_range_of_3a_minus_2b_l1771_177164


namespace NUMINAMATH_CALUDE_no_solutions_for_prime_power_equation_l1771_177132

theorem no_solutions_for_prime_power_equation (p m n k : ℕ) : 
  Nat.Prime p → 
  p % 2 = 1 → 
  0 < n → 
  n ≤ m → 
  m ≤ 3 * n → 
  p^m + p^n + 1 = k^2 → 
  False :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_for_prime_power_equation_l1771_177132


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1771_177117

def sandwich_cost : ℝ := 4
def soda_cost : ℝ := 3
def tax_rate : ℝ := 0.1
def num_sandwiches : ℕ := 4
def num_sodas : ℕ := 6

theorem total_cost_calculation :
  let subtotal := sandwich_cost * num_sandwiches + soda_cost * num_sodas
  let tax := subtotal * tax_rate
  let total_cost := subtotal + tax
  total_cost = 37.4 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1771_177117


namespace NUMINAMATH_CALUDE_dans_remaining_money_l1771_177134

/-- 
Given an initial amount of money and the cost of a candy bar, 
calculate the remaining amount after purchasing the candy bar.
-/
def remaining_money (initial_amount : ℝ) (candy_cost : ℝ) : ℝ :=
  initial_amount - candy_cost

/-- 
Theorem: Given an initial amount of $4 and a candy bar cost of $1, 
the remaining amount after purchasing the candy bar is $3.
-/
theorem dans_remaining_money : 
  remaining_money 4 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l1771_177134


namespace NUMINAMATH_CALUDE_lcm_9_12_15_l1771_177121

theorem lcm_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_9_12_15_l1771_177121


namespace NUMINAMATH_CALUDE_sons_age_l1771_177129

theorem sons_age (mother_age son_age : ℕ) : 
  mother_age = 4 * son_age →
  mother_age + son_age = 49 + 6 →
  son_age = 11 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1771_177129


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1771_177190

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (m-2)*x - m
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (∀ (y : ℝ), f y = 0 → y = x₁ ∨ y = x₂) ∧
  x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ - x₂ = 5/2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1771_177190


namespace NUMINAMATH_CALUDE_cyclists_average_speed_cyclists_average_speed_is_22_point_5_l1771_177179

/-- Cyclist's average speed problem -/
theorem cyclists_average_speed (total_distance : ℝ) (initial_speed : ℝ) 
  (speed_increase : ℝ) (distance_fraction : ℝ) : ℝ :=
  let new_speed := initial_speed * (1 + speed_increase)
  let time_first_part := (distance_fraction * total_distance) / initial_speed
  let time_second_part := ((1 - distance_fraction) * total_distance) / new_speed
  let total_time := time_first_part + time_second_part
  total_distance / total_time

/-- Proof of the cyclist's average speed -/
theorem cyclists_average_speed_is_22_point_5 :
  cyclists_average_speed 1 20 0.2 (1/3) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_average_speed_cyclists_average_speed_is_22_point_5_l1771_177179


namespace NUMINAMATH_CALUDE_square_perimeter_l1771_177178

/-- The perimeter of a square is 160 cm, given that its area is five times
    the area of a rectangle with dimensions 32 cm * 10 cm. -/
theorem square_perimeter (square_area rectangle_area : ℝ) : 
  square_area = 5 * rectangle_area →
  rectangle_area = 32 * 10 →
  4 * Real.sqrt square_area = 160 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1771_177178


namespace NUMINAMATH_CALUDE_equation_solution_l1771_177112

theorem equation_solution : ∃! x : ℝ, (x^2 - 6*x + 8)/(x^2 - 7*x + 12) = (x^2 - 3*x - 10)/(x^2 + x - 12) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1771_177112


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1771_177176

theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![x, 1]
  let b : Fin 2 → ℝ := ![1, -1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1771_177176


namespace NUMINAMATH_CALUDE_graph_composition_l1771_177109

-- Define the equation
def equation (x y : ℝ) : Prop := x^2 * (x + y + 1) = y^3 * (x + y + 1)

-- Define the components of the graph
def parabola_component (x y : ℝ) : Prop := x^2 = y^3 ∧ x + y + 1 ≠ 0
def line_component (x y : ℝ) : Prop := y = -x - 1

-- Theorem stating that the graph consists of a parabola and a line
theorem graph_composition :
  ∀ x y : ℝ, equation x y ↔ parabola_component x y ∨ line_component x y :=
sorry

end NUMINAMATH_CALUDE_graph_composition_l1771_177109


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1771_177187

/-- An isosceles triangle with side lengths 5 and 2 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  isIsosceles : a = b
  sideLength1 : a = 5
  sideLength2 : b = 5
  base : ℝ
  baseLength : base = 2

/-- The perimeter of the isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.a + t.b + t.base

theorem isosceles_triangle_perimeter :
  ∀ (t : IsoscelesTriangle), perimeter t = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1771_177187


namespace NUMINAMATH_CALUDE_alternating_sequence_property_l1771_177133

def alternatingSequence (n : ℕ) : ℤ := (-1) ^ (n + 1)

theorem alternating_sequence_property : ∀ n : ℕ, 
  (alternatingSequence n = 1 ∧ alternatingSequence (n + 1) = -1) ∨
  (alternatingSequence n = -1 ∧ alternatingSequence (n + 1) = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_alternating_sequence_property_l1771_177133


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1771_177143

def i : ℂ := Complex.I

theorem imaginary_part_of_z (z : ℂ) : z = (2 + i) / i → z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1771_177143


namespace NUMINAMATH_CALUDE_travel_distance_ratio_l1771_177194

theorem travel_distance_ratio :
  ∀ (total_distance plane_distance bus_distance train_distance : ℕ),
    total_distance = 900 →
    plane_distance = total_distance / 3 →
    bus_distance = 360 →
    train_distance = total_distance - (plane_distance + bus_distance) →
    (train_distance : ℚ) / bus_distance = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_travel_distance_ratio_l1771_177194


namespace NUMINAMATH_CALUDE_even_decreasing_inequality_l1771_177140

-- Define an even function that is decreasing on (0,+∞)
def is_even_and_decreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x ∧ x < y → f y < f x)

-- State the theorem
theorem even_decreasing_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h : is_even_and_decreasing f) : 
  f (-3/4) ≥ f (a^2 - a + 1) := by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_inequality_l1771_177140


namespace NUMINAMATH_CALUDE_plane_equation_proof_l1771_177184

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the vector between two points -/
def vectorBetweenPoints (p1 p2 : Point3D) : Vector3D :=
  { x := p2.x - p1.x
    y := p2.y - p1.y
    z := p2.z - p1.z }

/-- Checks if a vector is perpendicular to a plane -/
def isPerpendicularToPlane (v : Vector3D) (a b c : ℝ) : Prop :=
  a * v.x + b * v.y + c * v.z = 0

/-- Checks if a point lies on a plane -/
def isPointOnPlane (p : Point3D) (a b c d : ℝ) : Prop :=
  a * p.x + b * p.y + c * p.z + d = 0

theorem plane_equation_proof (A B C : Point3D) 
    (h1 : A.x = -4 ∧ A.y = -2 ∧ A.z = 5)
    (h2 : B.x = 3 ∧ B.y = -3 ∧ B.z = -7)
    (h3 : C.x = 9 ∧ C.y = 3 ∧ C.z = -7) :
    let BC := vectorBetweenPoints B C
    isPerpendicularToPlane BC 1 1 0 ∧ isPointOnPlane A 1 1 0 (-6) := by
  sorry

#check plane_equation_proof

end NUMINAMATH_CALUDE_plane_equation_proof_l1771_177184


namespace NUMINAMATH_CALUDE_binomial_p_value_l1771_177193

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  ξ : ℝ

/-- The expected value of a binomial random variable -/
def expectedValue (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: If a binomial random variable has E[ξ] = 8 and D[ξ] = 1.6, then p = 0.8 -/
theorem binomial_p_value (X : BinomialRV) 
  (h1 : expectedValue X = 8) 
  (h2 : variance X = 1.6) : 
  X.p = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_p_value_l1771_177193


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l1771_177149

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l1771_177149


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1771_177111

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { a := 1, b := 0, c := 0 }

/-- Shift a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c }

/-- Shift a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

/-- The resulting parabola after shifts -/
def resulting_parabola : Parabola :=
  shift_vertical (shift_horizontal original_parabola 3) 4

theorem parabola_shift_theorem :
  resulting_parabola = { a := 1, b := -6, c := 13 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1771_177111


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l1771_177161

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for part 1
theorem solution_part1 :
  {x : ℝ | f x (-1) ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem for part 2
theorem solution_part2 :
  {a : ℝ | ∀ x, f x a ≥ 2} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l1771_177161


namespace NUMINAMATH_CALUDE_problem_statement_l1771_177105

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3) :
  (∀ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → y/x + 3/y ≥ 4) ∧
  (∀ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → x*y ≤ 9/8) ∧
  (∀ ε > 0, ∃ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 3 ∧ Real.sqrt a + Real.sqrt (2*b) > 2 - ε) ∧
  (∀ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → a^2 + 4*b^2 ≥ 9/2) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1771_177105


namespace NUMINAMATH_CALUDE_hyperbola_other_asymptote_l1771_177170

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One of the asymptotes of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- The x-coordinate of the foci -/
  foci_x : ℝ

/-- The other asymptote of the hyperbola -/
def otherAsymptote (h : Hyperbola) : ℝ → ℝ := 
  fun x => 2 * x + 16

theorem hyperbola_other_asymptote (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x => -2 * x) 
  (h2 : h.foci_x = -4) : 
  otherAsymptote h = fun x => 2 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_other_asymptote_l1771_177170


namespace NUMINAMATH_CALUDE_university_groups_l1771_177108

theorem university_groups (total_students : ℕ) (group_reduction : ℕ) 
  (h1 : total_students = 2808)
  (h2 : group_reduction = 4)
  (h3 : ∃ (n : ℕ), n > 0 ∧ total_students % n = 0 ∧ total_students % (n + group_reduction) = 0)
  (h4 : ∀ (n : ℕ), n > 0 → total_students % n = 0 → (total_students / n < 30)) :
  ∃ (new_groups : ℕ), new_groups = 104 ∧ 
    total_students % new_groups = 0 ∧
    total_students % (new_groups + group_reduction) = 0 ∧
    total_students / new_groups < 30 :=
by sorry

end NUMINAMATH_CALUDE_university_groups_l1771_177108


namespace NUMINAMATH_CALUDE_fuel_station_problem_l1771_177162

/-- Fuel station problem -/
theorem fuel_station_problem 
  (service_cost : ℝ) 
  (fuel_cost_per_liter : ℝ) 
  (total_cost : ℝ) 
  (num_minivans : ℕ) 
  (minivan_tank : ℝ) 
  (truck_tank : ℝ) 
  (h1 : service_cost = 2.20)
  (h2 : fuel_cost_per_liter = 0.70)
  (h3 : total_cost = 395.4)
  (h4 : num_minivans = 4)
  (h5 : minivan_tank = 65)
  (h6 : truck_tank = minivan_tank * 2.2)
  : ∃ (num_trucks : ℕ), num_trucks = 2 ∧ 
    total_cost = (num_minivans * (service_cost + fuel_cost_per_liter * minivan_tank)) + 
                 (num_trucks : ℝ) * (service_cost + fuel_cost_per_liter * truck_tank) :=
by sorry


end NUMINAMATH_CALUDE_fuel_station_problem_l1771_177162


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1771_177191

/-- A parabola with equation y = 2x^2 + 8x + m -/
structure Parabola where
  m : ℝ

/-- Predicate to check if a parabola has only two common points with the coordinate axes -/
def has_two_axis_intersections (p : Parabola) : Prop :=
  -- This is a placeholder for the actual condition
  True

/-- Theorem stating that if a parabola y = 2x^2 + 8x + m has only two common points
    with the coordinate axes, then m = 0 or m = 8 -/
theorem parabola_intersection_theorem (p : Parabola) :
  has_two_axis_intersections p → p.m = 0 ∨ p.m = 8 := by
  sorry


end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1771_177191


namespace NUMINAMATH_CALUDE_vector_simplification_l1771_177169

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points in the vector space
variable (A B O M : V)

-- Define the theorem
theorem vector_simplification (A B O M : V) :
  (B - A) + (O - B) + (M - O) + (B - M) = B - A :=
by sorry

end NUMINAMATH_CALUDE_vector_simplification_l1771_177169


namespace NUMINAMATH_CALUDE_max_absolute_value_of_Z_l1771_177110

theorem max_absolute_value_of_Z (Z : ℂ) (h : Complex.abs (Z - (3 + 4*I)) = 1) :
  ∃ (M : ℝ), M = 6 ∧ ∀ (W : ℂ), Complex.abs (W - (3 + 4*I)) = 1 → Complex.abs W ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_absolute_value_of_Z_l1771_177110


namespace NUMINAMATH_CALUDE_david_recreation_spending_l1771_177185

-- Define the wages from last week as a parameter
def last_week_wages : ℝ := sorry

-- Define the percentage spent on recreation last week
def last_week_recreation_percent : ℝ := 0.40

-- Define the wage reduction percentage
def wage_reduction_percent : ℝ := 0.05

-- Define the increase in recreation spending
def recreation_increase_percent : ℝ := 1.1875

-- Calculate this week's wages
def this_week_wages : ℝ := last_week_wages * (1 - wage_reduction_percent)

-- Calculate the amount spent on recreation last week
def last_week_recreation_amount : ℝ := last_week_wages * last_week_recreation_percent

-- Calculate the amount spent on recreation this week
def this_week_recreation_amount : ℝ := last_week_recreation_amount * recreation_increase_percent

-- Define the theorem
theorem david_recreation_spending :
  this_week_recreation_amount / this_week_wages = 0.5 := by sorry

end NUMINAMATH_CALUDE_david_recreation_spending_l1771_177185


namespace NUMINAMATH_CALUDE_feb_first_is_wednesday_l1771_177166

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the previous day
def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

-- Define a function to go back n days
def goBackDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => prevDay (goBackDays d m)

-- Theorem statement
theorem feb_first_is_wednesday (h : DayOfWeek) :
  h = DayOfWeek.Tuesday → goBackDays h 27 = DayOfWeek.Wednesday :=
by
  sorry

end NUMINAMATH_CALUDE_feb_first_is_wednesday_l1771_177166


namespace NUMINAMATH_CALUDE_sequence_2023_l1771_177160

theorem sequence_2023 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, a n > 0) →
  (∀ n, 2 * S n = a n * (a n + 1)) →
  a 2023 = 2023 := by
sorry

end NUMINAMATH_CALUDE_sequence_2023_l1771_177160


namespace NUMINAMATH_CALUDE_min_value_expression_l1771_177159

theorem min_value_expression (x : ℝ) : (x^2 + 7) / Real.sqrt (x^2 + 3) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1771_177159


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1771_177127

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  Complex.im (2 * i / (1 - i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1771_177127


namespace NUMINAMATH_CALUDE_parametric_to_general_form_l1771_177157

/-- Given parametric equations of a line, prove its general form -/
theorem parametric_to_general_form (t : ℝ) (x y : ℝ) :
  x = 2 - 3 * t ∧ y = 1 + 2 * t → 2 * x + 3 * y - 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_general_form_l1771_177157


namespace NUMINAMATH_CALUDE_intersection_and_parallel_line_l1771_177199

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define point P as the intersection of l₁ and l₂
def P : ℝ × ℝ := (-2, 2)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 3 * x - 2 * y + 10 = 0

-- Define the two possible perpendicular lines
def perp_line₁ (x y : ℝ) : Prop := 4 * x - 3 * y = 0
def perp_line₂ (x y : ℝ) : Prop := x - 2 * y = 0

theorem intersection_and_parallel_line :
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) ∧
  parallel_line P.1 P.2 ∧
  (∃ (x y : ℝ), parallel_line x y ∧ 3 * x - 2 * y + 4 = 0) ∧
  (perp_line₁ 0 0 ∧ (∃ (x y : ℝ), perp_line₁ x y ∧ l₁ x y ∧ l₂ x y) ∨
   perp_line₂ 0 0 ∧ (∃ (x y : ℝ), perp_line₂ x y ∧ l₁ x y ∧ l₂ x y)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_line_l1771_177199


namespace NUMINAMATH_CALUDE_simplify_expression_l1771_177173

theorem simplify_expression (y : ℝ) : y - 3*(2+y) + 4*(2-y) - 5*(2+3*y) = -21*y - 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1771_177173


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l1771_177126

theorem sum_of_distinct_prime_factors : ∃ (s : Finset Nat), 
  (∀ p ∈ s, Nat.Prime p ∧ p ∣ (25^3 - 27^2)) ∧ 
  (∀ p, Nat.Prime p → p ∣ (25^3 - 27^2) → p ∈ s) ∧
  (s.sum id = 28) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l1771_177126


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1771_177158

/-- Given that i is the imaginary unit, prove that (3 + i) / (1 + 2*i) = 1 - i -/
theorem complex_fraction_simplification :
  (3 + I : ℂ) / (1 + 2*I) = 1 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1771_177158


namespace NUMINAMATH_CALUDE_preimage_of_2_neg2_l1771_177100

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x^2 - y)

-- Define the theorem
theorem preimage_of_2_neg2 :
  ∃ (x y : ℝ), x ≥ 0 ∧ f x y = (2, -2) ∧ (x, y) = (0, 2) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_2_neg2_l1771_177100


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1771_177123

def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1771_177123


namespace NUMINAMATH_CALUDE_equality_and_inequality_proof_l1771_177182

theorem equality_and_inequality_proof (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_eq : (3 : ℝ)^x = (4 : ℝ)^y ∧ (4 : ℝ)^y = (6 : ℝ)^z) : 
  (1 / z - 1 / x = 1 / (2 * y)) ∧ (3 * x < 4 * y ∧ 4 * y < 6 * z) := by
sorry

end NUMINAMATH_CALUDE_equality_and_inequality_proof_l1771_177182


namespace NUMINAMATH_CALUDE_perfect_square_addition_subtraction_l1771_177144

theorem perfect_square_addition_subtraction : ∃! n : ℤ, 
  (∃ u : ℤ, n + 5 = u^2) ∧ (∃ v : ℤ, n - 11 = v^2) :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_square_addition_subtraction_l1771_177144


namespace NUMINAMATH_CALUDE_three_W_seven_equals_thirteen_l1771_177198

/-- Definition of operation W -/
def W (x y : ℤ) : ℤ := y + 5*x - x^2

/-- Theorem: 3W7 equals 13 -/
theorem three_W_seven_equals_thirteen : W 3 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_three_W_seven_equals_thirteen_l1771_177198


namespace NUMINAMATH_CALUDE_number_calculation_l1771_177153

theorem number_calculation (N : ℝ) : 
  0.2 * (|(-0.05)|^3 * 0.35 * 0.7 * N) = 182.7 → N = 20880000 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l1771_177153


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l1771_177165

def g (x : ℝ) : ℝ := |x - 5| + |x - 7| - |2*x - 12| + |3*x - 21|

theorem sum_of_max_min_g :
  ∃ (max min : ℝ),
    (∀ x : ℝ, 5 ≤ x → x ≤ 10 → g x ≤ max) ∧
    (∃ x : ℝ, 5 ≤ x ∧ x ≤ 10 ∧ g x = max) ∧
    (∀ x : ℝ, 5 ≤ x → x ≤ 10 → min ≤ g x) ∧
    (∃ x : ℝ, 5 ≤ x ∧ x ≤ 10 ∧ g x = min) ∧
    max + min = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l1771_177165


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1771_177130

-- Define the coin values
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def half_dollar_value : ℕ := 50

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^5

-- Define the function to calculate successful outcomes
def successful_outcomes : ℕ := 18

-- Define the target value
def target_value : ℕ := 40

-- Theorem statement
theorem coin_flip_probability :
  (successful_outcomes : ℚ) / total_outcomes = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1771_177130


namespace NUMINAMATH_CALUDE_multiple_compounds_same_weight_l1771_177196

/-- Represents a chemical compound -/
structure Compound where
  molecular_weight : ℕ

/-- Represents the set of all possible compounds -/
def AllCompounds : Set Compound := sorry

/-- The given molecular weight -/
def given_weight : ℕ := 391

/-- Compounds with the given molecular weight -/
def compounds_with_given_weight : Set Compound :=
  {c ∈ AllCompounds | c.molecular_weight = given_weight}

/-- Theorem stating that multiple compounds can have the same molecular weight -/
theorem multiple_compounds_same_weight :
  ∃ (c1 c2 : Compound), c1 ≠ c2 ∧ c1 ∈ compounds_with_given_weight ∧ c2 ∈ compounds_with_given_weight :=
sorry

end NUMINAMATH_CALUDE_multiple_compounds_same_weight_l1771_177196


namespace NUMINAMATH_CALUDE_math_interest_group_size_l1771_177147

theorem math_interest_group_size (total_cards : ℕ) : 
  (total_cards = 182) → 
  (∃ n : ℕ, n * (n - 1) = total_cards ∧ n > 0) → 
  (∃ n : ℕ, n * (n - 1) = total_cards ∧ n = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_math_interest_group_size_l1771_177147


namespace NUMINAMATH_CALUDE_business_trip_bus_distance_l1771_177137

theorem business_trip_bus_distance (total_distance : ℝ) 
  (h_total : total_distance = 1800) 
  (h_plane : total_distance / 4 = 450) 
  (h_train : total_distance / 6 = 300) 
  (h_taxi : total_distance / 8 = 225) 
  (h_bus_rental : ∃ (bus rental : ℝ), 
    bus + rental = total_distance - (450 + 300 + 225) ∧ 
    bus = rental / 2) : 
  ∃ (bus : ℝ), bus = 275 := by
  sorry

end NUMINAMATH_CALUDE_business_trip_bus_distance_l1771_177137


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1771_177136

theorem triangle_angle_measure (D E F : ℝ) : 
  0 < D ∧ 0 < E ∧ 0 < F →  -- Angles are positive
  D + E + F = 180 →        -- Sum of angles in a triangle
  E = 3 * F →              -- Angle E is three times angle F
  F = 18 →                 -- Angle F is 18 degrees
  D = 108 :=               -- Conclusion: Angle D is 108 degrees
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1771_177136


namespace NUMINAMATH_CALUDE_negative_two_triangle_five_l1771_177189

/-- Definition of the triangle operation for rational numbers -/
def triangle (a b : ℚ) : ℚ := a * b + b - a

/-- Theorem stating that (-2) triangle 5 equals -3 -/
theorem negative_two_triangle_five : triangle (-2) 5 = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_triangle_five_l1771_177189


namespace NUMINAMATH_CALUDE_num_assignments_is_15000_l1771_177171

/-- Represents a valid assignment of students to events -/
structure Assignment where
  /-- The mapping of students to events -/
  student_to_event : Fin 7 → Fin 5
  /-- Ensures that students 0 and 1 (representing A and B) are not in the same event -/
  students_separated : student_to_event 0 ≠ student_to_event 1
  /-- Ensures that each event has at least one participant -/
  events_nonempty : ∀ e : Fin 5, ∃ s : Fin 7, student_to_event s = e

/-- The number of valid assignments -/
def num_valid_assignments : ℕ := sorry

/-- The main theorem stating that the number of valid assignments is 15000 -/
theorem num_assignments_is_15000 : num_valid_assignments = 15000 := by sorry

end NUMINAMATH_CALUDE_num_assignments_is_15000_l1771_177171


namespace NUMINAMATH_CALUDE_kongming_total_score_l1771_177141

/-- Represents a recruitment exam with a written test and an interview -/
structure RecruitmentExam where
  writtenTestWeight : Real
  interviewWeight : Real
  writtenTestScore : Real
  interviewScore : Real

/-- Calculates the total score for a recruitment exam -/
def totalScore (exam : RecruitmentExam) : Real :=
  exam.writtenTestScore * exam.writtenTestWeight + exam.interviewScore * exam.interviewWeight

theorem kongming_total_score :
  let exam : RecruitmentExam := {
    writtenTestWeight := 0.6,
    interviewWeight := 0.4,
    writtenTestScore := 90,
    interviewScore := 85
  }
  totalScore exam = 88 := by sorry

end NUMINAMATH_CALUDE_kongming_total_score_l1771_177141


namespace NUMINAMATH_CALUDE_polynomial_divisibility_theorem_l1771_177148

theorem polynomial_divisibility_theorem : 
  ∃ (r : ℝ), 
    (∀ (x : ℝ), ∃ (q : ℝ), 8*x^3 - 4*x^2 - 42*x + 45 = (x - r)^2 * q) ∧ 
    (abs (r - 1.5) < 0.1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_theorem_l1771_177148


namespace NUMINAMATH_CALUDE_lillian_mushroom_foraging_l1771_177156

/-- Calculates the number of uncertain mushrooms given the total, safe, and poisonous counts. -/
def uncertain_mushrooms (total safe : ℕ) : ℕ :=
  total - (safe + 2 * safe)

/-- Proves that the number of uncertain mushrooms is 5 given the problem conditions. -/
theorem lillian_mushroom_foraging :
  uncertain_mushrooms 32 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_lillian_mushroom_foraging_l1771_177156


namespace NUMINAMATH_CALUDE_events_B_C_complementary_l1771_177175

-- Define the sample space (cube faces)
def Ω : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define events A, B, and C
def A : Set ℕ := {n ∈ Ω | n % 2 = 1}
def B : Set ℕ := {n ∈ Ω | n ≤ 3}
def C : Set ℕ := {n ∈ Ω | n ≥ 4}

-- Theorem to prove
theorem events_B_C_complementary : B ∪ C = Ω ∧ B ∩ C = ∅ := by
  sorry

end NUMINAMATH_CALUDE_events_B_C_complementary_l1771_177175


namespace NUMINAMATH_CALUDE_square_area_error_l1771_177177

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := 1.05 * s
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 10.25 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l1771_177177


namespace NUMINAMATH_CALUDE_solution_exists_l1771_177174

theorem solution_exists : ∃ c : ℝ, 
  (∃ x : ℤ, (x = ⌊c⌋ ∧ 3 * (x : ℝ)^2 - 9 * (x : ℝ) - 30 = 0)) ∧
  (∃ y : ℝ, (y = c - ⌊c⌋ ∧ 4 * y^2 - 8 * y + 1 = 0)) ∧
  (c = -1 - Real.sqrt 3 / 2 ∨ c = 6 - Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_exists_l1771_177174


namespace NUMINAMATH_CALUDE_boys_without_calculators_l1771_177172

theorem boys_without_calculators (total_boys : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ) 
  (h1 : total_boys = 20)
  (h2 : students_with_calculators = 25)
  (h3 : girls_with_calculators = 15) :
  total_boys - (students_with_calculators - girls_with_calculators) = 10 :=
by sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l1771_177172


namespace NUMINAMATH_CALUDE_sum_of_cubic_equations_l1771_177152

theorem sum_of_cubic_equations (x y : ℝ) 
  (hx : (x - 1)^3 + 1997*(x - 1) = -1)
  (hy : (y - 1)^3 + 1997*(y - 1) = 1) :
  x + y = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubic_equations_l1771_177152


namespace NUMINAMATH_CALUDE_ribbon_lengths_after_cutting_l1771_177116

def initial_lengths : List ℝ := [15, 20, 24, 26, 30]

def median (l : List ℝ) : ℝ := sorry
def range (l : List ℝ) : ℝ := sorry
def average (l : List ℝ) : ℝ := sorry

theorem ribbon_lengths_after_cutting (new_lengths : List ℝ) :
  (average new_lengths = average initial_lengths - 5) →
  (median new_lengths = median initial_lengths) →
  (range new_lengths = range initial_lengths) →
  new_lengths.length = initial_lengths.length →
  (∀ x ∈ new_lengths, x > 0) →
  new_lengths = [9, 9, 24, 24, 24] :=
by sorry

end NUMINAMATH_CALUDE_ribbon_lengths_after_cutting_l1771_177116
