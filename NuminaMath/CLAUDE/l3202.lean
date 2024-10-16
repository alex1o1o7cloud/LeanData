import Mathlib

namespace NUMINAMATH_CALUDE_least_subtraction_l3202_320298

theorem least_subtraction (n : ℕ) : ∃! x : ℕ, 
  (∀ d ∈ ({9, 11, 17} : Set ℕ), (3381 - x) % d = 8) ∧ 
  (∀ y : ℕ, y < x → ∃ d ∈ ({9, 11, 17} : Set ℕ), (3381 - y) % d ≠ 8) :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_l3202_320298


namespace NUMINAMATH_CALUDE_Q_roots_nature_l3202_320204

/-- The polynomial Q(x) = x^6 - 4x^5 + 3x^4 - 7x^3 - x^2 + x + 10 -/
def Q (x : ℝ) : ℝ := x^6 - 4*x^5 + 3*x^4 - 7*x^3 - x^2 + x + 10

/-- Theorem stating that Q(x) has at least one negative root and at least two positive roots -/
theorem Q_roots_nature :
  (∃ x : ℝ, x < 0 ∧ Q x = 0) ∧
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x ≠ y ∧ Q x = 0 ∧ Q y = 0) :=
sorry

end NUMINAMATH_CALUDE_Q_roots_nature_l3202_320204


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l3202_320254

theorem simplify_sqrt_sum (h : π / 2 < 2 ∧ 2 < 3 * π / 4) :
  Real.sqrt (1 + Real.sin 4) + Real.sqrt (1 - Real.sin 4) = 2 * Real.sin 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l3202_320254


namespace NUMINAMATH_CALUDE_smallest_factorial_divisor_l3202_320286

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_factorial_divisor (n : ℕ) (h1 : n > 1) :
  (∀ k : ℕ, k > 1 ∧ k < 7 → ¬(factorial k % n = 0)) ∧ (factorial 7 % n = 0) →
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factorial_divisor_l3202_320286


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_l3202_320294

/-- Proves that given the ratio of sheep to horses is 4:7, there are 32 sheep on the farm,
    and the farm needs a total of 12,880 ounces of horse food per day,
    each horse needs 230 ounces of horse food per day. -/
theorem stewart_farm_horse_food (sheep : ℕ) (horses : ℕ) (total_food : ℕ) :
  sheep = 32 →
  4 * horses = 7 * sheep →
  total_food = 12880 →
  total_food / horses = 230 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_horse_food_l3202_320294


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l3202_320281

open Real

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (log x) / x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := (1 - log x) / (x^2)

theorem tangent_parallel_to_x_axis :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ f_derivative x₀ = 0 → f x₀ = 1/Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l3202_320281


namespace NUMINAMATH_CALUDE_dress_cost_equals_total_savings_l3202_320288

/-- Calculates the cost of the dress based on initial savings, weekly allowance, weekly spending, and waiting period. -/
def dress_cost (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ) (waiting_weeks : ℕ) : ℕ :=
  initial_savings + (weekly_allowance - weekly_spending) * waiting_weeks

/-- Proves that the dress cost is equal to Vanessa's total savings after the waiting period. -/
theorem dress_cost_equals_total_savings :
  dress_cost 20 30 10 3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_dress_cost_equals_total_savings_l3202_320288


namespace NUMINAMATH_CALUDE_diamond_two_seven_l3202_320293

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a^2 * b - a * b^2

-- Theorem statement
theorem diamond_two_seven : diamond 2 7 = -70 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_seven_l3202_320293


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3202_320251

/-- An arithmetic sequence with a_5 = 3 and a_6 = -2 has common difference -5 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a5 : a 5 = 3) 
  (h_a6 : a 6 = -2) : 
  a 6 - a 5 = -5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3202_320251


namespace NUMINAMATH_CALUDE_correct_password_contains_one_and_seven_l3202_320263

/-- Represents a four-digit password -/
def Password := Fin 4 → Fin 10

/-- Checks if two passwords have exactly two matching digits in different positions -/
def hasTwoMatchingDigits (p1 p2 : Password) : Prop :=
  (∃ i j : Fin 4, i ≠ j ∧ p1 i = p2 i ∧ p1 j = p2 j) ∧
  (∀ i j k : Fin 4, i ≠ j → j ≠ k → k ≠ i → ¬(p1 i = p2 i ∧ p1 j = p2 j ∧ p1 k = p2 k))

/-- The first four incorrect attempts -/
def attempts : Fin 4 → Password
| 0 => λ i => [3, 4, 0, 6].get i
| 1 => λ i => [1, 6, 3, 0].get i
| 2 => λ i => [7, 3, 6, 4].get i
| 3 => λ i => [6, 1, 7, 3].get i

/-- The theorem stating that the correct password must contain 1 and 7 -/
theorem correct_password_contains_one_and_seven 
  (correct : Password)
  (h1 : ∀ i : Fin 4, hasTwoMatchingDigits (attempts i) correct)
  (h2 : correct ≠ attempts 0 ∧ correct ≠ attempts 1 ∧ correct ≠ attempts 2 ∧ correct ≠ attempts 3) :
  (∃ i j : Fin 4, i ≠ j ∧ correct i = 1 ∧ correct j = 7) :=
sorry

end NUMINAMATH_CALUDE_correct_password_contains_one_and_seven_l3202_320263


namespace NUMINAMATH_CALUDE_cubic_km_to_cubic_m_l3202_320289

/-- Proves that 1 cubic kilometer equals 1,000,000,000 cubic meters -/
theorem cubic_km_to_cubic_m : 
  (∀ (km m : ℝ), km = 1 ∧ m = 1000 ∧ km * 1000 = m) → 
  (1 : ℝ)^3 * 1000^3 = 1000000000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_km_to_cubic_m_l3202_320289


namespace NUMINAMATH_CALUDE_train_speed_l3202_320270

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : Real) (time : Real) :
  length = 125.01 →
  time = 5 →
  let speed := (length / 1000) / (time / 3600)
  ∃ ε > 0, abs (speed - 90.0072) < ε := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3202_320270


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l3202_320273

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a = 6 → b = 8 →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c = 10 ∨ c = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l3202_320273


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_two_l3202_320246

theorem opposite_of_sqrt_two : -(Real.sqrt 2) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_two_l3202_320246


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3202_320265

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n, S n = n * (a 0 + a (n-1)) / 2

/-- Theorem: If S_6/S_3 = 4 for an arithmetic sequence, then S_5/S_6 = 25/36 -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.S 6 / seq.S 3 = 4) : 
  seq.S 5 / seq.S 6 = 25/36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3202_320265


namespace NUMINAMATH_CALUDE_milk_mixture_theorem_l3202_320256

/-- Proves that adding 24 gallons of 10% butterfat milk to 8 gallons of 50% butterfat milk 
    results in a mixture with 20% butterfat. -/
theorem milk_mixture_theorem :
  let initial_volume : ℝ := 8
  let initial_butterfat_percent : ℝ := 50
  let added_volume : ℝ := 24
  let added_butterfat_percent : ℝ := 10
  let desired_butterfat_percent : ℝ := 20
  let total_volume := initial_volume + added_volume
  let total_butterfat := (initial_volume * initial_butterfat_percent / 100) + 
                         (added_volume * added_butterfat_percent / 100)
  (total_butterfat / total_volume) * 100 = desired_butterfat_percent :=
by sorry

end NUMINAMATH_CALUDE_milk_mixture_theorem_l3202_320256


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3202_320239

theorem perfect_square_trinomial (m : ℚ) : 
  m > 0 → 
  (∃ a : ℚ, ∀ x : ℚ, x^2 - 2*m*x + 36 = (x - a)^2) → 
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3202_320239


namespace NUMINAMATH_CALUDE_ordered_pairs_sum_30_l3202_320258

theorem ordered_pairs_sum_30 :
  (Finset.filter (fun p : ℕ × ℕ => p.1 + p.2 = 30 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 31) (Finset.range 31))).card = 29 := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_sum_30_l3202_320258


namespace NUMINAMATH_CALUDE_oranges_left_l3202_320274

/-- Proves that the number of oranges Joan is left with is equal to the number she picked minus the number Sara sold. -/
theorem oranges_left (joan_picked : ℕ) (sara_sold : ℕ) (joan_left : ℕ)
  (h1 : joan_picked = 37)
  (h2 : sara_sold = 10)
  (h3 : joan_left = 27) :
  joan_left = joan_picked - sara_sold :=
by sorry

end NUMINAMATH_CALUDE_oranges_left_l3202_320274


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_a_range_for_negative_f_l3202_320266

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| - |2 * x - a|

-- Theorem for part (1)
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x > 0} = {x : ℝ | 1 < x ∧ x < 5/3} := by sorry

-- Theorem for part (2)
theorem a_range_for_negative_f :
  (∀ x < 2, f a x < 0) ↔ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_a_range_for_negative_f_l3202_320266


namespace NUMINAMATH_CALUDE_min_value_zero_at_one_sixth_l3202_320226

/-- The quadratic expression as a function of x, y, and c -/
def f (x y c : ℝ) : ℝ :=
  2 * x^2 - 4 * c * x * y + (2 * c^2 + 1) * y^2 - 2 * x - 6 * y + 9

/-- Theorem stating that 1/6 is the value of c that makes the minimum of f zero -/
theorem min_value_zero_at_one_sixth :
  ∃ (x y : ℝ), f x y (1/6) = 0 ∧ ∀ (x' y' : ℝ), f x' y' (1/6) ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_zero_at_one_sixth_l3202_320226


namespace NUMINAMATH_CALUDE_unique_determination_l3202_320222

-- Define the triangle types
inductive TriangleType
  | Isosceles
  | Equilateral
  | Right
  | Scalene

-- Define the given parts
inductive GivenParts
  | BaseAngleVertexAngle
  | VertexAngleBase
  | CircumscribedRadius
  | ArmInscribedRadius
  | TwoAnglesOneSide

-- Function to check if a combination uniquely determines a triangle
def uniquelyDetermines (t : TriangleType) (p : GivenParts) : Prop :=
  match t, p with
  | TriangleType.Isosceles, GivenParts.BaseAngleVertexAngle => False
  | TriangleType.Isosceles, GivenParts.VertexAngleBase => True
  | TriangleType.Equilateral, GivenParts.CircumscribedRadius => True
  | TriangleType.Right, GivenParts.ArmInscribedRadius => False
  | TriangleType.Scalene, GivenParts.TwoAnglesOneSide => True
  | _, _ => False

theorem unique_determination :
  ∀ (t : TriangleType) (p : GivenParts),
    (t = TriangleType.Isosceles ∧ p = GivenParts.BaseAngleVertexAngle) ↔ ¬(uniquelyDetermines t p) :=
sorry

end NUMINAMATH_CALUDE_unique_determination_l3202_320222


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l3202_320232

theorem polynomial_root_sum (p q : ℝ) : 
  (∃ (a b c d : ℕ+), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x : ℝ, x^4 - 10*x^3 + p*x^2 - q*x + 24 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d))) →
  p + q = 85 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l3202_320232


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_and_parallel_l3202_320245

/-- A line with equation x + y + b = 0 is tangent to the circle x^2 + y^2 = 2
    and parallel to x + y - 1 = 0 if and only if b = 2 or b = -2 -/
theorem line_tangent_to_circle_and_parallel (b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 2 → x + y + b ≠ 0) ∧ 
  (∃ x y : ℝ, x^2 + y^2 = 2 ∧ x + y + b = 0) ∧
  (∀ x y : ℝ, x + y + b = 0 → x + y - 1 ≠ 0) ↔ 
  b = 2 ∨ b = -2 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_and_parallel_l3202_320245


namespace NUMINAMATH_CALUDE_piggy_bank_problem_l3202_320210

theorem piggy_bank_problem (initial_amount : ℝ) : 
  initial_amount = 204 → 
  (initial_amount * (1 - 0.6) * (1 - 0.5) * (1 - 0.35)) = 26.52 :=
by sorry

end NUMINAMATH_CALUDE_piggy_bank_problem_l3202_320210


namespace NUMINAMATH_CALUDE_square_equality_necessary_not_sufficient_l3202_320208

theorem square_equality_necessary_not_sufficient :
  (∀ x y : ℝ, x = y → x^2 = y^2) ∧
  ¬(∀ x y : ℝ, x^2 = y^2 → x = y) :=
by sorry

end NUMINAMATH_CALUDE_square_equality_necessary_not_sufficient_l3202_320208


namespace NUMINAMATH_CALUDE_parabola_one_x_intercept_l3202_320292

/-- A parabola defined by x = -3y^2 + 2y + 2 has exactly one x-intercept. -/
theorem parabola_one_x_intercept :
  ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 2 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_one_x_intercept_l3202_320292


namespace NUMINAMATH_CALUDE_shoe_multiple_l3202_320249

theorem shoe_multiple (jacob edward brian : ℕ) : 
  jacob = edward / 2 →
  brian = 22 →
  jacob + edward + brian = 121 →
  edward / brian = 3 :=
by sorry

end NUMINAMATH_CALUDE_shoe_multiple_l3202_320249


namespace NUMINAMATH_CALUDE_range_of_a_l3202_320271

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2) - Real.log (abs x)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f (-a*x + Real.log x + 1) + f (a*x - Real.log x - 1) ≥ 2 * f 1) →
  a ∈ Set.Icc (1 / Real.exp 1) ((2 + Real.log 3) / 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3202_320271


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3202_320225

/-- Represents a hyperbola -/
structure Hyperbola where
  center : ℝ × ℝ
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  point : ℝ × ℝ

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : (ℝ → ℝ → Prop) :=
  fun x y => y^2 / 20 - x^2 / 16 = 1

/-- Theorem: Given a hyperbola with center at (0, 0), foci at (0, -6) and (0, 6),
    and passing through the point (2, -5), its standard equation is y^2/20 - x^2/16 = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_center : h.center = (0, 0))
    (h_foci : h.foci = ((0, -6), (0, 6)))
    (h_point : h.point = (2, -5)) :
    standard_equation h = fun x y => y^2 / 20 - x^2 / 16 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3202_320225


namespace NUMINAMATH_CALUDE_num_boys_in_class_l3202_320282

-- Define the number of girls in the class
def num_girls : ℕ := 10

-- Define the number of ways to select 1 girl and 2 boys
def num_selections : ℕ := 1050

-- Define the function to calculate the number of ways to select 1 girl and 2 boys
def selection_ways (n : ℕ) : ℕ := num_girls * (n * (n - 1) / 2)

-- Theorem statement
theorem num_boys_in_class : ∃ (n : ℕ), n > 0 ∧ selection_ways n = num_selections :=
sorry

end NUMINAMATH_CALUDE_num_boys_in_class_l3202_320282


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l3202_320261

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n.val = 3 ∨ n.val = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l3202_320261


namespace NUMINAMATH_CALUDE_inequality_proof_l3202_320275

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3202_320275


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3202_320252

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x y : ℝ), (x - 2)^2 + y^2 = 3 ∧ 
    (∃ (k : ℝ), b * x + a * y = k ∨ b * x - a * y = k)) →
  (a^2 = 1 ∧ b^2 = 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3202_320252


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l3202_320200

theorem quadratic_solution_property (a b : ℝ) :
  (a * 1^2 + b * 1 - 1 = 0) → (2023 - a - b = 2022) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l3202_320200


namespace NUMINAMATH_CALUDE_base_conversion_digit_sum_l3202_320285

theorem base_conversion_digit_sum : 
  (∃ (d_min d_max : ℕ), 
    (∀ n : ℕ, 
      (9^3 ≤ n ∧ n < 9^4) → 
      (d_min ≤ Nat.log2 (n + 1) ∧ Nat.log2 (n + 1) ≤ d_max)) ∧
    (d_max - d_min = 2) ∧
    (d_min + (d_min + 1) + d_max = 33)) := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_digit_sum_l3202_320285


namespace NUMINAMATH_CALUDE_expression_equals_sum_l3202_320217

theorem expression_equals_sum (a b c : ℝ) (ha : a = 13) (hb : b = 19) (hc : c = 23) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_sum_l3202_320217


namespace NUMINAMATH_CALUDE_units_digit_problem_l3202_320206

theorem units_digit_problem : (8 * 18 * 1988 - 8^3) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l3202_320206


namespace NUMINAMATH_CALUDE_lions_count_l3202_320218

theorem lions_count (lions tigers cougars : ℕ) : 
  tigers = 14 →
  cougars = (lions + tigers) / 2 →
  lions + tigers + cougars = 39 →
  lions = 12 := by
sorry

end NUMINAMATH_CALUDE_lions_count_l3202_320218


namespace NUMINAMATH_CALUDE_square_side_increase_l3202_320283

theorem square_side_increase (p : ℝ) : 
  (1 + p / 100)^2 = 1.1025 → p = 5 := by
sorry

end NUMINAMATH_CALUDE_square_side_increase_l3202_320283


namespace NUMINAMATH_CALUDE_set_equality_l3202_320296

-- Define the universal set I
def I : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {3, 4, 5}

-- Define set N
def N : Set Nat := {1, 3, 6}

-- Define the set we want to prove equal to (C_I M) ∩ (C_I N)
def target_set : Set Nat := {2, 7}

-- Theorem statement
theorem set_equality : 
  target_set = (I \ M) ∩ (I \ N) := by sorry

end NUMINAMATH_CALUDE_set_equality_l3202_320296


namespace NUMINAMATH_CALUDE_crude_oil_mixture_l3202_320230

/-- Given two sources of crude oil, prove that the second source contains 75% hydrocarbons -/
theorem crude_oil_mixture (
  source1_percent : ℝ)
  (source2_percent : ℝ)
  (final_volume : ℝ)
  (final_percent : ℝ)
  (source2_volume : ℝ) :
  source1_percent = 25 →
  final_volume = 50 →
  final_percent = 55 →
  source2_volume = 30 →
  source2_percent = 75 :=
by
  sorry

#check crude_oil_mixture

end NUMINAMATH_CALUDE_crude_oil_mixture_l3202_320230


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l3202_320278

/-- The value of k that makes the line 4x + 6y + k = 0 tangent to the parabola y^2 = 32x -/
def tangent_k : ℝ := 72

/-- The line equation -/
def line (x y k : ℝ) : Prop := 4 * x + 6 * y + k = 0

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 32 * x

/-- The tangency condition -/
def is_tangent (k : ℝ) : Prop :=
  ∃! (x y : ℝ), line x y k ∧ parabola x y

theorem tangent_line_to_parabola :
  is_tangent tangent_k :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l3202_320278


namespace NUMINAMATH_CALUDE_exactly_four_pairs_l3202_320276

/-- The number of ordered pairs (m,n) of positive integers satisfying the given conditions -/
def count_pairs : ℕ := 4

/-- Predicate to check if a pair (m,n) satisfies the required conditions -/
def is_valid_pair (m n : ℕ) : Prop :=
  m ≥ n ∧ m * m - n * n = 144

/-- The theorem stating that there are exactly 4 valid pairs -/
theorem exactly_four_pairs :
  (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = count_pairs ∧
    ∀ (p : ℕ × ℕ), p ∈ pairs ↔ is_valid_pair p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_exactly_four_pairs_l3202_320276


namespace NUMINAMATH_CALUDE_no_solution_l3202_320205

def equation1 (x₁ x₂ x₃ : ℝ) : Prop := 2 * x₁ + 5 * x₂ - 4 * x₃ = 8
def equation2 (x₁ x₂ x₃ : ℝ) : Prop := 3 * x₁ + 15 * x₂ - 9 * x₃ = 5
def equation3 (x₁ x₂ x₃ : ℝ) : Prop := 5 * x₁ + 5 * x₂ - 7 * x₃ = 1

theorem no_solution : ¬∃ x₁ x₂ x₃ : ℝ, equation1 x₁ x₂ x₃ ∧ equation2 x₁ x₂ x₃ ∧ equation3 x₁ x₂ x₃ := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l3202_320205


namespace NUMINAMATH_CALUDE_quadratic_inequality_relation_l3202_320248

theorem quadratic_inequality_relation :
  (∀ x : ℝ, x > 3 → x^2 - 2*x - 3 > 0) ∧
  (∃ x : ℝ, x^2 - 2*x - 3 > 0 ∧ ¬(x > 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_relation_l3202_320248


namespace NUMINAMATH_CALUDE_gcd_problems_l3202_320238

theorem gcd_problems : 
  (Nat.gcd 91 49 = 7) ∧ (Nat.gcd (Nat.gcd 319 377) 116 = 29) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l3202_320238


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3202_320202

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_inequality (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ x < 0 ∨ x > 2) →
  (a = -2 ∧ b = 0) ∧
  (∀ m : ℝ,
    (m = 0 → ∀ x, ¬(f a b x < m^2 - 1)) ∧
    (m > 0 → ∀ x, f a b x < m^2 - 1 ↔ 1 - m < x ∧ x < 1 + m) ∧
    (m < 0 → ∀ x, f a b x < m^2 - 1 ↔ 1 + m < x ∧ x < 1 - m)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3202_320202


namespace NUMINAMATH_CALUDE_average_marks_chemistry_math_l3202_320260

theorem average_marks_chemistry_math (P C M : ℕ) : 
  P + C + M = P + 110 → (C + M) / 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_math_l3202_320260


namespace NUMINAMATH_CALUDE_dog_reachable_area_theorem_l3202_320295

/-- The area outside a regular hexagonal doghouse that a dog can reach when tethered to a vertex -/
def dogReachableArea (side_length : Real) (rope_length : Real) : Real :=
  -- Definition to be filled
  sorry

/-- Theorem stating the area the dog can reach outside the doghouse -/
theorem dog_reachable_area_theorem :
  dogReachableArea 1 4 = (82 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_dog_reachable_area_theorem_l3202_320295


namespace NUMINAMATH_CALUDE_factorial_fraction_is_integer_l3202_320212

/-- Given that m and n are non-negative integers and 0! = 1, 
    prove that (2m)!(2n)! / (m!n!(m+n)!) is an integer. -/
theorem factorial_fraction_is_integer (m n : ℕ) : 
  ∃ k : ℤ, (↑((2*m).factorial * (2*n).factorial) : ℚ) / 
    (↑(m.factorial * n.factorial * (m+n).factorial)) = ↑k :=
sorry

end NUMINAMATH_CALUDE_factorial_fraction_is_integer_l3202_320212


namespace NUMINAMATH_CALUDE_three_integers_product_2008th_power_l3202_320220

theorem three_integers_product_2008th_power :
  ∃ (x y z : ℕ), 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧  -- distinct
    x > 0 ∧ y > 0 ∧ z > 0 ∧  -- positive
    y = (x + z) / 2 ∧        -- one is average of other two
    ∃ (k : ℕ), x * y * z = k^2008 := by
  sorry

end NUMINAMATH_CALUDE_three_integers_product_2008th_power_l3202_320220


namespace NUMINAMATH_CALUDE_sum_of_angles_two_triangles_l3202_320257

/-- The sum of interior angles of a triangle in degrees -/
def triangle_angle_sum : ℝ := 180

/-- The number of triangles in the diagram -/
def number_of_triangles : ℕ := 2

/-- Theorem: The sum of all interior angles in two triangles is 360° -/
theorem sum_of_angles_two_triangles : 
  (↑number_of_triangles : ℝ) * triangle_angle_sum = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_two_triangles_l3202_320257


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l3202_320224

theorem triangle_side_and_area 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : a = 1) 
  (h2 : b = 2) 
  (h3 : C = 60 * π / 180) : 
  c = Real.sqrt 3 ∧ (1/2 * a * b * Real.sin C) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_and_area_l3202_320224


namespace NUMINAMATH_CALUDE_coefficient_a7_equals_negative_eight_l3202_320290

/-- Given x^8 = a₀ + a₁(x+1) + a₂(x+1)² + ... + a₈(x+1)⁸, prove that a₇ = -8 -/
theorem coefficient_a7_equals_negative_eight (x : ℝ) (a : Fin 9 → ℝ) :
  x^8 = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + a 3 * (x + 1)^3 + a 4 * (x + 1)^4 + 
        a 5 * (x + 1)^5 + a 6 * (x + 1)^6 + a 7 * (x + 1)^7 + a 8 * (x + 1)^8 →
  a 7 = -8 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a7_equals_negative_eight_l3202_320290


namespace NUMINAMATH_CALUDE_uncle_ben_farm_l3202_320207

def farm_problem (total_chickens : ℕ) (non_laying_hens : ℕ) (eggs_per_hen : ℕ) (total_eggs : ℕ) : Prop :=
  ∃ (roosters hens : ℕ),
    roosters + hens = total_chickens ∧
    3 * (hens - non_laying_hens) = total_eggs ∧
    roosters = 39

theorem uncle_ben_farm :
  farm_problem 440 15 3 1158 :=
sorry

end NUMINAMATH_CALUDE_uncle_ben_farm_l3202_320207


namespace NUMINAMATH_CALUDE_total_hamburgers_calculation_l3202_320229

/-- Calculates the total number of hamburgers bought given the total amount spent,
    costs of single and double burgers, and the number of double burgers bought. -/
theorem total_hamburgers_calculation 
  (total_spent : ℚ)
  (single_burger_cost : ℚ)
  (double_burger_cost : ℚ)
  (double_burgers_bought : ℕ)
  (h1 : total_spent = 66.5)
  (h2 : single_burger_cost = 1)
  (h3 : double_burger_cost = 1.5)
  (h4 : double_burgers_bought = 33) :
  ∃ (single_burgers_bought : ℕ),
    single_burgers_bought + double_burgers_bought = 50 ∧
    total_spent = single_burger_cost * single_burgers_bought + double_burger_cost * double_burgers_bought :=
by sorry


end NUMINAMATH_CALUDE_total_hamburgers_calculation_l3202_320229


namespace NUMINAMATH_CALUDE_regular_polygon_radius_inequality_l3202_320253

/-- For a regular polygon with n sides, n ≥ 3, the circumradius R is at most twice the inradius r. -/
theorem regular_polygon_radius_inequality (n : ℕ) (r R : ℝ) 
  (h_n : n ≥ 3) 
  (h_r : r > 0) 
  (h_R : R > 0) 
  (h_relation : r / R = Real.cos (π / n)) : 
  R ≤ 2 * r := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_radius_inequality_l3202_320253


namespace NUMINAMATH_CALUDE_milburg_population_l3202_320211

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := 5256

/-- The number of children in Milburg -/
def children : ℕ := 2987

/-- The total population of Milburg -/
def total_population : ℕ := grown_ups + children

/-- Theorem stating that the total population of Milburg is 8243 -/
theorem milburg_population : total_population = 8243 := by
  sorry

end NUMINAMATH_CALUDE_milburg_population_l3202_320211


namespace NUMINAMATH_CALUDE_prob_red_white_red_is_7_66_l3202_320280

-- Define the number of red and white marbles
def red_marbles : ℕ := 5
def white_marbles : ℕ := 7

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + white_marbles

-- Define the probability of drawing red, white, and red marbles in order
def prob_red_white_red : ℚ := (red_marbles : ℚ) / total_marbles *
                              (white_marbles : ℚ) / (total_marbles - 1) *
                              (red_marbles - 1 : ℚ) / (total_marbles - 2)

-- Theorem statement
theorem prob_red_white_red_is_7_66 :
  prob_red_white_red = 7 / 66 := by sorry

end NUMINAMATH_CALUDE_prob_red_white_red_is_7_66_l3202_320280


namespace NUMINAMATH_CALUDE_equation_solution_l3202_320240

theorem equation_solution : 
  ∃ x : ℚ, (1 / 4 : ℚ) + 5 / x = 12 / x + (1 / 15 : ℚ) → x = 420 / 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3202_320240


namespace NUMINAMATH_CALUDE_sphere_division_theorem_l3202_320259

/-- The maximum number of regions into which a sphere can be divided by n great circles -/
def sphere_regions (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem: The maximum number of regions into which a sphere can be divided by n great circles is n^2 - n + 2 -/
theorem sphere_division_theorem (n : ℕ) : 
  sphere_regions n = n^2 - n + 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_division_theorem_l3202_320259


namespace NUMINAMATH_CALUDE_min_value_is_four_l3202_320216

/-- The line passing through points A(3, 0) and B(1, 1) -/
def line_AB (x y : ℝ) : Prop := y = (x - 3) / (-2)

/-- The objective function to be minimized -/
def objective_function (x y : ℝ) : ℝ := 2 * x + 4 * y

/-- Theorem stating that the minimum value of the objective function is 4 -/
theorem min_value_is_four :
  ∀ x y : ℝ, line_AB x y → objective_function x y ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_is_four_l3202_320216


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l3202_320236

theorem theater_ticket_difference :
  ∀ (orchestra_price balcony_price : ℕ) 
    (total_tickets total_cost : ℕ) 
    (orchestra_tickets balcony_tickets : ℕ),
  orchestra_price = 18 →
  balcony_price = 12 →
  total_tickets = 450 →
  total_cost = 6300 →
  orchestra_tickets + balcony_tickets = total_tickets →
  orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_cost →
  balcony_tickets - orchestra_tickets = 150 := by
sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l3202_320236


namespace NUMINAMATH_CALUDE_x_value_l3202_320267

theorem x_value : ∃ x : ℝ, (x = 88 * (1 + 0.20)) ∧ (x = 105.6) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3202_320267


namespace NUMINAMATH_CALUDE_line_outside_plane_iff_at_most_one_point_l3202_320272

-- Define the basic types
variable (L : Type*) -- Type for lines
variable (P : Type*) -- Type for planes

-- Define the relationships between lines and planes
variable (parallel : L → P → Prop)
variable (intersects : L → P → Prop)
variable (within : L → P → Prop)
variable (outside : L → P → Prop)

-- Define the number of common points
variable (common_points : L → P → ℕ)

-- Theorem statement
theorem line_outside_plane_iff_at_most_one_point 
  (l : L) (p : P) : 
  outside l p ↔ common_points l p ≤ 1 := by sorry

end NUMINAMATH_CALUDE_line_outside_plane_iff_at_most_one_point_l3202_320272


namespace NUMINAMATH_CALUDE_poverty_definition_l3202_320215

-- Define poverty as a string
def poverty : String := "poverty"

-- State the theorem
theorem poverty_definition : poverty = "poverty" := by
  sorry

end NUMINAMATH_CALUDE_poverty_definition_l3202_320215


namespace NUMINAMATH_CALUDE_garden_perimeter_is_72_l3202_320255

/-- A rectangular garden with specific properties -/
structure Garden where
  /-- The shorter side of the garden -/
  short_side : ℝ
  /-- The longer side of the garden -/
  long_side : ℝ
  /-- The diagonal of the garden is 34 meters -/
  diagonal_eq : short_side ^ 2 + long_side ^ 2 = 34 ^ 2
  /-- The area of the garden is 240 square meters -/
  area_eq : short_side * long_side = 240
  /-- The longer side is three times the shorter side -/
  side_ratio : long_side = 3 * short_side

/-- The perimeter of a rectangular garden -/
def perimeter (g : Garden) : ℝ :=
  2 * (g.short_side + g.long_side)

/-- Theorem stating that the perimeter of the garden is 72 meters -/
theorem garden_perimeter_is_72 (g : Garden) : perimeter g = 72 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_is_72_l3202_320255


namespace NUMINAMATH_CALUDE_gems_calculation_l3202_320268

/-- Calculates the total number of gems received given an initial spend, gem rate, and bonus percentage. -/
def total_gems (spend : ℕ) (rate : ℕ) (bonus_percent : ℕ) : ℕ :=
  let initial_gems := spend * rate
  let bonus_gems := initial_gems * bonus_percent / 100
  initial_gems + bonus_gems

/-- Proves that given the specified conditions, the total number of gems received is 30000. -/
theorem gems_calculation :
  let spend := 250
  let rate := 100
  let bonus_percent := 20
  total_gems spend rate bonus_percent = 30000 := by
  sorry

end NUMINAMATH_CALUDE_gems_calculation_l3202_320268


namespace NUMINAMATH_CALUDE_solve_inequality_find_m_range_l3202_320247

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| + 2
def g (m : ℝ) (x : ℝ) : ℝ := m * |x|

-- Theorem for part (1)
theorem solve_inequality (x : ℝ) : f x > 5 ↔ x < -1 ∨ x > 5 := by sorry

-- Theorem for part (2)
theorem find_m_range (m : ℝ) : (∀ x : ℝ, f x ≥ g m x) ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_find_m_range_l3202_320247


namespace NUMINAMATH_CALUDE_andrews_age_l3202_320235

/-- Andrew's age problem -/
theorem andrews_age :
  ∀ (a g : ℚ),
  g = 10 * a →
  g - a = 60 →
  a = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_andrews_age_l3202_320235


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l3202_320209

def X : ℕ := 4444^4444

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def A : ℕ := sum_of_digits X

def B : ℕ := sum_of_digits A

theorem sum_of_digits_of_B_is_seven :
  sum_of_digits B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l3202_320209


namespace NUMINAMATH_CALUDE_sum_of_square_and_cube_minority_l3202_320279

theorem sum_of_square_and_cube_minority :
  let max_num := 1000000
  let max_square := (max_num.sqrt : ℕ)
  let max_cube := (max_num ^ (1/3) : ℕ)
  let possible_sums := (max_square + 1) * (max_cube + 1)
  possible_sums ≤ 101101 ∧ 101101 < max_num / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_and_cube_minority_l3202_320279


namespace NUMINAMATH_CALUDE_polynomial_equality_l3202_320291

theorem polynomial_equality : 
  103^5 - 5 * 103^4 + 10 * 103^3 - 10 * 103^2 + 5 * 103 - 1 = 102^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3202_320291


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_bounds_l3202_320214

theorem quadratic_roots_sum_squares_bounds (k : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 - (k - 2) * x₁ + (k^2 + 3 * k + 5) = 0 →
  x₂^2 - (k - 2) * x₂ + (k^2 + 3 * k + 5) = 0 →
  x₁ ≠ x₂ →
  ∃ (y : ℝ), y = x₁^2 + x₂^2 ∧ y ≤ 18 ∧ y ≥ 50/9 ∧
  (∃ (k₁ : ℝ), x₁^2 - (k₁ - 2) * x₁ + (k₁^2 + 3 * k₁ + 5) = 0 ∧
               x₂^2 - (k₁ - 2) * x₂ + (k₁^2 + 3 * k₁ + 5) = 0 ∧
               x₁^2 + x₂^2 = 18) ∧
  (∃ (k₂ : ℝ), x₁^2 - (k₂ - 2) * x₁ + (k₂^2 + 3 * k₂ + 5) = 0 ∧
               x₂^2 - (k₂ - 2) * x₂ + (k₂^2 + 3 * k₂ + 5) = 0 ∧
               x₁^2 + x₂^2 = 50/9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_bounds_l3202_320214


namespace NUMINAMATH_CALUDE_jenna_profit_l3202_320241

/-- Calculates the total profit for Jenna's wholesale business --/
def calculate_profit (
  widget_cost : ℝ)
  (widget_price : ℝ)
  (rent : ℝ)
  (tax_rate : ℝ)
  (worker_salary : ℝ)
  (num_workers : ℕ)
  (widgets_sold : ℕ) : ℝ :=
  let total_sales := widget_price * widgets_sold
  let total_cost := widget_cost * widgets_sold
  let salaries := worker_salary * num_workers
  let profit_before_tax := total_sales - total_cost - rent - salaries
  let tax := tax_rate * profit_before_tax
  profit_before_tax - tax

/-- Theorem stating that Jenna's profit is $4000 given the problem conditions --/
theorem jenna_profit :
  calculate_profit 3 8 10000 0.2 2500 4 5000 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_jenna_profit_l3202_320241


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3202_320227

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 = 2 →
  a 6 = 8 →
  a 3 * a 4 * a 5 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3202_320227


namespace NUMINAMATH_CALUDE_binomial_coefficient_21_13_l3202_320242

theorem binomial_coefficient_21_13 :
  (Nat.choose 20 13 = 77520) →
  (Nat.choose 20 12 = 125970) →
  (Nat.choose 21 13 = 203490) := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_21_13_l3202_320242


namespace NUMINAMATH_CALUDE_cubic_polynomial_problem_l3202_320269

variable (a b c : ℝ)
variable (P : ℝ → ℝ)

theorem cubic_polynomial_problem :
  (∀ x, x^3 - 2*x^2 - 4*x - 1 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∃ p q r s : ℝ, ∀ x, P x = p*x^3 + q*x^2 + r*x + s) →
  P a = b + 2*c →
  P b = 2*a + c →
  P c = a + 2*b →
  P (a + b + c) = -20 →
  ∀ x, P x = 4*x^3 - 6*x^2 - 12*x := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_problem_l3202_320269


namespace NUMINAMATH_CALUDE_profit_percentage_l3202_320244

theorem profit_percentage (cp sp : ℝ) (h : cp / sp = 4 / 5) :
  (sp - cp) / cp * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l3202_320244


namespace NUMINAMATH_CALUDE_perfect_square_quotient_l3202_320287

theorem perfect_square_quotient (a b : ℕ+) (h : (a * b + 1) ∣ (a ^ 2 + b ^ 2)) :
  ∃ k : ℕ, (a ^ 2 + b ^ 2) / (a * b + 1) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quotient_l3202_320287


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_sufficient_not_necessary_l3202_320262

theorem ac_squared_gt_bc_squared_sufficient_not_necessary (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2) :=
by sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_sufficient_not_necessary_l3202_320262


namespace NUMINAMATH_CALUDE_oxygen_atom_diameter_scientific_notation_l3202_320250

theorem oxygen_atom_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000000148 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = -10 := by
  sorry

end NUMINAMATH_CALUDE_oxygen_atom_diameter_scientific_notation_l3202_320250


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3202_320297

/-- Represents a date with year, month, and day components. -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Calculates the ending date given the start date and time period. -/
def calculateEndDate (startDate : Date) (timePeriod : Rat) : Date :=
  sorry

/-- Calculates the time period in years given principal, rate, and interest. -/
def calculateTimePeriod (principal : Rat) (rate : Rat) (interest : Rat) : Rat :=
  sorry

theorem simple_interest_problem (principal : Rat) (rate : Rat) (startDate : Date) (interest : Rat) :
  principal = 2000 →
  rate = 25 / (4 * 100) →
  startDate = ⟨2005, 2, 4⟩ →
  interest = 25 →
  let timePeriod := calculateTimePeriod principal rate interest
  let endDate := calculateEndDate startDate timePeriod
  endDate = ⟨2005, 4, 16⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3202_320297


namespace NUMINAMATH_CALUDE_actual_toddler_count_l3202_320299

theorem actual_toddler_count (bill_count : ℕ) (double_counted : ℕ) (missed : ℕ) 
  (h1 : bill_count = 26) 
  (h2 : double_counted = 8) 
  (h3 : missed = 3) : 
  bill_count - double_counted + missed = 21 := by
  sorry

end NUMINAMATH_CALUDE_actual_toddler_count_l3202_320299


namespace NUMINAMATH_CALUDE_books_movies_difference_l3202_320284

/-- The number of books in the "crazy silly school" series -/
def num_books : ℕ := 36

/-- The number of movies in the "crazy silly school" series -/
def num_movies : ℕ := 25

/-- The number of books read -/
def books_read : ℕ := 17

/-- The number of movies watched -/
def movies_watched : ℕ := 13

/-- Theorem stating the difference between the number of books and movies -/
theorem books_movies_difference : num_books - num_movies = 11 := by
  sorry

end NUMINAMATH_CALUDE_books_movies_difference_l3202_320284


namespace NUMINAMATH_CALUDE_average_problem_l3202_320264

theorem average_problem (t b c d e : ℝ) : 
  (t + b + c + 14 + 15) / 5 = 12 →
  t = 2 * b →
  (t + b + c + d + e + 14 + 15) / 7 = 12 →
  (t + b + c + 29) / 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_average_problem_l3202_320264


namespace NUMINAMATH_CALUDE_smallest_n_value_l3202_320213

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 2020 →
  c > a → c > b →
  c > a + 100 →
  a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  ¬(10 ∣ m) →
  (∀ k, k < n → ∃ l, a.factorial * b.factorial * c.factorial = l * (10 ^ k) ∧ 10 ∣ l) →
  n = 499 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_value_l3202_320213


namespace NUMINAMATH_CALUDE_candy_difference_l3202_320223

theorem candy_difference (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : red = 40)
  (h2 : yellow < 3 * red)
  (h3 : blue = yellow / 2)
  (h4 : red + blue = 90) :
  3 * red - yellow = 20 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_l3202_320223


namespace NUMINAMATH_CALUDE_product_of_sums_l3202_320201

theorem product_of_sums (x y : ℝ) (h1 : x + y = -3) (h2 : x * y = 1) :
  (x + 5) * (y + 5) = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_l3202_320201


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l3202_320228

/-- The jumping distances of animals in a contest -/
structure JumpingContest where
  mouse_jump : ℕ
  frog_jump : ℕ
  grasshopper_jump : ℕ
  mouse_frog_diff : frog_jump = mouse_jump + 12
  grasshopper_frog_diff : grasshopper_jump = frog_jump + 19

/-- Theorem: In a jumping contest where the mouse jumped 8 inches, 
    the mouse jumped 12 inches less than the frog, 
    and the grasshopper jumped 19 inches farther than the frog, 
    the grasshopper jumped 39 inches. -/
theorem grasshopper_jump_distance (contest : JumpingContest) 
  (h_mouse_jump : contest.mouse_jump = 8) : 
  contest.grasshopper_jump = 39 := by
  sorry


end NUMINAMATH_CALUDE_grasshopper_jump_distance_l3202_320228


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3202_320237

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  a 4 = 1 ∧ (a 1 + a 2 + a 3 + a 4 + a 5 = 31/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3202_320237


namespace NUMINAMATH_CALUDE_number_minus_division_equals_l3202_320277

theorem number_minus_division_equals (x : ℝ) : x - (104 / 20.8) = 545 ↔ x = 550 := by
  sorry

end NUMINAMATH_CALUDE_number_minus_division_equals_l3202_320277


namespace NUMINAMATH_CALUDE_expression_is_perfect_square_l3202_320233

/-- The expression is a perfect square when x = 0.04 -/
theorem expression_is_perfect_square : 
  ∃ y : ℝ, (11.98 * 11.98 + 11.98 * 0.04 + 0.02 * 0.02) = y * y := by
  sorry

end NUMINAMATH_CALUDE_expression_is_perfect_square_l3202_320233


namespace NUMINAMATH_CALUDE_vector_subtraction_magnitude_l3202_320234

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_subtraction_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((Prod.fst b)^2 + (Prod.snd b)^2) = 1) :
  Real.sqrt ((Prod.fst (a - 2 • b))^2 + (Prod.snd (a - 2 • b))^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_magnitude_l3202_320234


namespace NUMINAMATH_CALUDE_quadratic_equation_transform_l3202_320221

/-- Given a quadratic equation 4x^2 + 16x - 400 = 0, prove that when transformed
    into the form (x + k)^2 = t, the value of t is 104. -/
theorem quadratic_equation_transform (x k t : ℝ) : 
  (4 * x^2 + 16 * x - 400 = 0) → 
  (∃ k, ∀ x, 4 * x^2 + 16 * x - 400 = 0 ↔ (x + k)^2 = t) →
  t = 104 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_transform_l3202_320221


namespace NUMINAMATH_CALUDE_trig_values_special_angles_l3202_320219

theorem trig_values_special_angles :
  (Real.sin (π/6) = 1/2) ∧
  (Real.cos (π/6) = Real.sqrt 3 / 2) ∧
  (Real.tan (π/6) = Real.sqrt 3 / 3) ∧
  (Real.sin (π/4) = Real.sqrt 2 / 2) ∧
  (Real.cos (π/4) = Real.sqrt 2 / 2) ∧
  (Real.tan (π/4) = 1) ∧
  (Real.sin (π/3) = Real.sqrt 3 / 2) ∧
  (Real.cos (π/3) = 1/2) ∧
  (Real.tan (π/3) = Real.sqrt 3) ∧
  (Real.sin (π/2) = 1) ∧
  (Real.cos (π/2) = 0) := by
  sorry

-- Note: tan(π/2) is undefined, so it's not included in the theorem statement

end NUMINAMATH_CALUDE_trig_values_special_angles_l3202_320219


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3202_320203

theorem min_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 6 ∧ ∀ (x y : ℝ), 0 < x ∧ 0 < y ∧ 1/x + 1/y = 1 → 1/(x-1) + 9/(y-1) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3202_320203


namespace NUMINAMATH_CALUDE_arctg_sum_pi_half_l3202_320231

theorem arctg_sum_pi_half : Real.arctan 1 + Real.arctan (1/2) + Real.arctan (1/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctg_sum_pi_half_l3202_320231


namespace NUMINAMATH_CALUDE_comparison_of_A_and_B_l3202_320243

theorem comparison_of_A_and_B (a b c : ℝ) : a^2 + b^2 + c^2 + 14 ≥ 2*a + 4*b + 6*c := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_A_and_B_l3202_320243
