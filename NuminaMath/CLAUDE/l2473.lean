import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2473_247392

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 5 * x^2 + 4 * x + 20 = 0 ↔ x = a + b * I ∨ x = a - b * I) → 
  a + b^2 = 86/25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2473_247392


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l2473_247397

theorem distinct_prime_factors_of_30_factorial (n : ℕ) :
  n = 30 →
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card = 
  (Finset.filter (λ p => p.Prime ∧ p ∣ n.factorial) (Finset.range (n + 1))).card :=
by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l2473_247397


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l2473_247385

/-- Given a triangle ABC with area 144√3 and satisfying the relation 
    (sin A * sin B * sin C) / (sin A + sin B + sin C) = 1/4, 
    prove that the smallest possible perimeter is achieved when the triangle is equilateral 
    with side length 24. -/
theorem min_perimeter_triangle (A B C : ℝ) (area : ℝ) (h_area : area = 144 * Real.sqrt 3) 
    (h_relation : (Real.sin A * Real.sin B * Real.sin C) / (Real.sin A + Real.sin B + Real.sin C) = 1/4) :
  ∃ (s : ℝ), s = 24 ∧ 
    ∀ (a b c : ℝ), 
      (a * b * Real.sin C / 2 = area) → 
      ((Real.sin A * Real.sin B * Real.sin C) / (Real.sin A + Real.sin B + Real.sin C) = 1/4) → 
      (a + b + c ≥ 3 * s) :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l2473_247385


namespace NUMINAMATH_CALUDE_abc_congruence_l2473_247345

theorem abc_congruence (a b c : ℕ) : 
  a < 7 → b < 7 → c < 7 →
  (2 * a + 3 * b + c) % 7 = 1 →
  (3 * a + b + 2 * c) % 7 = 2 →
  (a + b + c) % 7 = 3 →
  (2 * a * b * c) % 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_congruence_l2473_247345


namespace NUMINAMATH_CALUDE_min_value_theorem_l2473_247390

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  ∃ (m : ℝ), m = 7 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + 3 * b = 5 * a * b → 3 * a + 4 * b ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2473_247390


namespace NUMINAMATH_CALUDE_meghan_money_l2473_247322

/-- The total amount of money Meghan has, given the number of bills of each denomination -/
def total_money (hundred_bills : ℕ) (fifty_bills : ℕ) (ten_bills : ℕ) : ℕ :=
  100 * hundred_bills + 50 * fifty_bills + 10 * ten_bills

/-- Theorem stating that Meghan's total money is $550 -/
theorem meghan_money : total_money 2 5 10 = 550 := by
  sorry

end NUMINAMATH_CALUDE_meghan_money_l2473_247322


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2473_247329

theorem complex_expression_equality :
  (7 - 3*Complex.I) - 3*(2 - 5*Complex.I) + (1 + 2*Complex.I) = 2 + 14*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2473_247329


namespace NUMINAMATH_CALUDE_dividend_proof_l2473_247310

theorem dividend_proof : 
  let dividend : ℕ := 11889708
  let divisor : ℕ := 12
  let quotient : ℕ := 990809
  dividend = divisor * quotient := by sorry

end NUMINAMATH_CALUDE_dividend_proof_l2473_247310


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l2473_247364

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (|6 * x₁| + 5 = 47 ∧ |6 * x₂| + 5 = 47 ∧ x₁ ≠ x₂) ∧ x₁ * x₂ = -49) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l2473_247364


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2473_247384

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 2) % 12 = 0 ∧
  (n + 2) % 30 = 0 ∧
  (n + 2) % 48 = 0 ∧
  (n + 2) % 74 = 0 ∧
  (n + 2) % 100 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 44398 ∧
  ∀ m : ℕ, m < 44398 → ¬(is_divisible_by_all m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2473_247384


namespace NUMINAMATH_CALUDE_unique_square_property_l2473_247302

theorem unique_square_property (A : ℕ+) 
  (h1 : 100 ≤ A.val^2 ∧ A.val^2 < 1000)
  (h2 : ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ A.val^2 = 100*x + 10*y + z)
  (h3 : ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ x*100 + y*10 + z = A.val - 1) :
  A.val = 19 ∧ A.val^2 = 361 := by
sorry

end NUMINAMATH_CALUDE_unique_square_property_l2473_247302


namespace NUMINAMATH_CALUDE_fraction_equality_l2473_247328

theorem fraction_equality (m n p r : ℚ) 
  (h1 : m / n = 21)
  (h2 : p / n = 7)
  (h3 : p / r = 1 / 7) :
  m / r = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2473_247328


namespace NUMINAMATH_CALUDE_base7_to_base10_5213_l2473_247327

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

/-- The theorem states that 5213 in base 7 is equal to 1823 in base 10 -/
theorem base7_to_base10_5213 :
  base7ToBase10 5 2 1 3 = 1823 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_5213_l2473_247327


namespace NUMINAMATH_CALUDE_abs_neg_five_l2473_247373

theorem abs_neg_five : |(-5 : ℝ)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_l2473_247373


namespace NUMINAMATH_CALUDE_otimes_inequality_range_l2473_247320

/-- Custom binary operation ⊗ -/
def otimes (a b : ℝ) : ℝ := a - 2 * b

/-- Theorem stating the range of a given the conditions -/
theorem otimes_inequality_range (a : ℝ) :
  (∀ x : ℝ, x > 6 ↔ (otimes x 3 > 0 ∧ otimes x a > a)) →
  a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_otimes_inequality_range_l2473_247320


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l2473_247346

theorem power_tower_mod_500 : 7^(7^(7^7)) ≡ 343 [ZMOD 500] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l2473_247346


namespace NUMINAMATH_CALUDE_sports_club_overlap_l2473_247355

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) : 
  total = 27 → badminton = 17 → tennis = 19 → neither = 2 → 
  badminton + tennis - (total - neither) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l2473_247355


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l2473_247315

theorem basketball_lineup_combinations (total_players : ℕ) (quadruplets : ℕ) (lineup_size : ℕ) (quadruplets_in_lineup : ℕ) : 
  total_players = 16 → 
  quadruplets = 4 → 
  lineup_size = 7 → 
  quadruplets_in_lineup = 2 → 
  (Nat.choose quadruplets quadruplets_in_lineup) * 
  (Nat.choose (total_players - quadruplets + quadruplets_in_lineup) (lineup_size - quadruplets_in_lineup)) = 12012 := by
  sorry

#check basketball_lineup_combinations

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l2473_247315


namespace NUMINAMATH_CALUDE_mean_of_data_l2473_247339

def data : List ℕ := [7, 5, 3, 5, 10]

theorem mean_of_data : (data.sum : ℚ) / data.length = 6 := by sorry

end NUMINAMATH_CALUDE_mean_of_data_l2473_247339


namespace NUMINAMATH_CALUDE_f_at_pi_third_l2473_247350

noncomputable def f (θ : Real) : Real :=
  (2 * Real.cos θ ^ 2 + Real.sin (2 * Real.pi - θ) ^ 2 + Real.sin (Real.pi / 2 + θ) - 3) /
  (2 + 2 * Real.cos (Real.pi + θ) ^ 2 + Real.cos (-θ))

theorem f_at_pi_third : f (Real.pi / 3) = -5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_f_at_pi_third_l2473_247350


namespace NUMINAMATH_CALUDE_problem_1_l2473_247307

theorem problem_1 : -3 + 8 - 15 - 6 = -16 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2473_247307


namespace NUMINAMATH_CALUDE_a_value_proof_l2473_247396

/-- The function f(x) = ax³ + 3x² + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem a_value_proof (a : ℝ) : f_derivative a (-1) = -12 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_a_value_proof_l2473_247396


namespace NUMINAMATH_CALUDE_k_inv_h_10_l2473_247316

-- Define the functions h and k
variable (h k : ℝ → ℝ)

-- Define the condition
axiom h_k_relation : ∀ x, h⁻¹ (k x) = 4 * x - 5

-- State the theorem
theorem k_inv_h_10 : k⁻¹ (h 10) = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_k_inv_h_10_l2473_247316


namespace NUMINAMATH_CALUDE_matrix_commutation_l2473_247324

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]

def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]

theorem matrix_commutation (x y z w : ℝ) :
  A * B x y z w = B x y z w * A →
  4 * z ≠ y →
  (x - w) / (y - 4 * z) = -3/13 := by sorry

end NUMINAMATH_CALUDE_matrix_commutation_l2473_247324


namespace NUMINAMATH_CALUDE_scalar_product_formula_l2473_247356

def vector_2d (x y : ℝ) : ℝ × ℝ := (x, y)

def scalar_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem scalar_product_formula (x₁ y₁ x₂ y₂ : ℝ) :
  scalar_product (vector_2d x₁ y₁) (vector_2d x₂ y₂) = x₁ * x₂ + y₁ * y₂ := by
  sorry

end NUMINAMATH_CALUDE_scalar_product_formula_l2473_247356


namespace NUMINAMATH_CALUDE_inequality_expression_l2473_247333

theorem inequality_expression (x : ℝ) : (x + 4 ≥ -1) ↔ (x + 4 ≥ -1) := by sorry

end NUMINAMATH_CALUDE_inequality_expression_l2473_247333


namespace NUMINAMATH_CALUDE_point_positions_l2473_247398

def line_equation (x y : ℝ) : Prop := 3 * x - 5 * y + 8 = 0

def point_A : ℝ × ℝ := (2, 5)
def point_B : ℝ × ℝ := (1, 2.2)

theorem point_positions :
  (¬ line_equation point_A.1 point_A.2) ∧
  (line_equation point_B.1 point_B.2) :=
by sorry

end NUMINAMATH_CALUDE_point_positions_l2473_247398


namespace NUMINAMATH_CALUDE_count_nonzero_monomials_l2473_247386

/-- The number of monomials with nonzero coefficients in the expansion of (x+y+z)^2030 + (x-y-z)^2030 -/
def nonzero_monomials_count : ℕ := 1032256

/-- The exponent used in the expression -/
def exponent : ℕ := 2030

theorem count_nonzero_monomials :
  (∃ (x y z : ℝ), (x + y + z)^exponent + (x - y - z)^exponent ≠ 0) →
  nonzero_monomials_count = (exponent / 2 + 1)^2 :=
sorry

end NUMINAMATH_CALUDE_count_nonzero_monomials_l2473_247386


namespace NUMINAMATH_CALUDE_fabric_cost_and_length_l2473_247309

/-- Given two identical pieces of fabric with the following properties:
    1. The total cost of the first piece is 126 rubles more than the second piece
    2. The cost of 4 meters from the first piece exceeds the cost of 3 meters from the second piece by 135 rubles
    3. 3 meters from the first piece and 4 meters from the second piece cost 382.50 rubles in total

    This theorem proves that:
    1. The length of each piece is 5.6 meters
    2. The cost per meter of the first piece is 67.5 rubles
    3. The cost per meter of the second piece is 45 rubles
-/
theorem fabric_cost_and_length 
  (cost_second : ℝ) -- Total cost of the second piece
  (length : ℝ) -- Length of each piece
  (h1 : cost_second + 126 = (cost_second / length + 126 / length) * length) -- First piece costs 126 more
  (h2 : 4 * (cost_second / length + 126 / length) - 3 * (cost_second / length) = 135) -- 4m of first vs 3m of second
  (h3 : 3 * (cost_second / length + 126 / length) + 4 * (cost_second / length) = 382.5) -- Total cost of 3m+4m
  : length = 5.6 ∧ 
    cost_second / length + 126 / length = 67.5 ∧ 
    cost_second / length = 45 := by
  sorry

end NUMINAMATH_CALUDE_fabric_cost_and_length_l2473_247309


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_coefficient_l2473_247358

theorem quadratic_equal_roots_coefficient (k : ℝ) (h : k = 1.7777777777777777) : 
  let eq := fun x : ℝ => 2 * k * x^2 + 3 * k * x + 2
  let discriminant := (3 * k)^2 - 4 * (2 * k) * 2
  discriminant = 0 → 3 * k = 5.333333333333333 :=
by
  sorry

#eval (3 : Float) * 1.7777777777777777

end NUMINAMATH_CALUDE_quadratic_equal_roots_coefficient_l2473_247358


namespace NUMINAMATH_CALUDE_total_worksheets_is_nine_l2473_247363

/-- Represents the grading problem for a teacher -/
structure GradingProblem where
  problems_per_worksheet : ℕ
  graded_worksheets : ℕ
  remaining_problems : ℕ

/-- Calculates the total number of worksheets to grade -/
def total_worksheets (gp : GradingProblem) : ℕ :=
  gp.graded_worksheets + (gp.remaining_problems / gp.problems_per_worksheet)

/-- Theorem stating that the total number of worksheets to grade is 9 -/
theorem total_worksheets_is_nine :
  ∀ (gp : GradingProblem),
    gp.problems_per_worksheet = 4 →
    gp.graded_worksheets = 5 →
    gp.remaining_problems = 16 →
    total_worksheets gp = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_total_worksheets_is_nine_l2473_247363


namespace NUMINAMATH_CALUDE_industrial_lubricants_percentage_l2473_247344

theorem industrial_lubricants_percentage (microphotonics : ℝ) (home_electronics : ℝ) 
  (food_additives : ℝ) (genetically_modified_microorganisms : ℝ) 
  (basic_astrophysics_degrees : ℝ) :
  microphotonics = 10 →
  home_electronics = 24 →
  food_additives = 15 →
  genetically_modified_microorganisms = 29 →
  basic_astrophysics_degrees = 50.4 →
  ∃ (industrial_lubricants : ℝ),
    industrial_lubricants = 8 ∧
    microphotonics + home_electronics + food_additives + 
    genetically_modified_microorganisms + industrial_lubricants + 
    (basic_astrophysics_degrees / 360 * 100) = 100 := by
  sorry

end NUMINAMATH_CALUDE_industrial_lubricants_percentage_l2473_247344


namespace NUMINAMATH_CALUDE_linear_function_property_l2473_247370

/-- A linear function f(x) = ax + b satisfying f(1) = 2 and f'(1) = 2 -/
def f (x : ℝ) : ℝ := 2 * x

theorem linear_function_property : f 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l2473_247370


namespace NUMINAMATH_CALUDE_percentage_increase_l2473_247303

theorem percentage_increase (x : ℝ) (h : x = 99.9) :
  (x - 90) / 90 * 100 = 11 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2473_247303


namespace NUMINAMATH_CALUDE_problem_solution_l2473_247377

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((4*x + a) * Real.log x) / (3*x + 1)

def tangent_perpendicular (a : ℝ) : Prop :=
  let f' := deriv (f a) 1
  f' * (-1) = 1

def inequality_holds (m : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 1 → (f 0 x) ≤ m * (x - 1)

theorem problem_solution :
  (∃ a : ℝ, tangent_perpendicular a ∧ a = 0) ∧
  (∃ m : ℝ, inequality_holds m ∧ ∀ m' : ℝ, m' ≥ m → inequality_holds m') :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2473_247377


namespace NUMINAMATH_CALUDE_average_daily_net_income_is_366_l2473_247382

/-- Represents the financial data for a single day -/
structure DailyData where
  income : ℕ
  tips : ℕ
  expenses : ℕ

/-- Calculates the net income for a single day -/
def netIncome (data : DailyData) : ℕ :=
  data.income + data.tips - data.expenses

/-- The financial data for 5 days -/
def fiveDaysData : Vector DailyData 5 :=
  ⟨[
    { income := 300, tips := 50, expenses := 80 },
    { income := 150, tips := 20, expenses := 40 },
    { income := 750, tips := 100, expenses := 150 },
    { income := 200, tips := 30, expenses := 50 },
    { income := 600, tips := 70, expenses := 120 }
  ], rfl⟩

/-- Calculates the average daily net income -/
def averageDailyNetIncome (data : Vector DailyData 5) : ℚ :=
  (data.toList.map netIncome).sum / 5

/-- Theorem stating that the average daily net income is $366 -/
theorem average_daily_net_income_is_366 :
  averageDailyNetIncome fiveDaysData = 366 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_net_income_is_366_l2473_247382


namespace NUMINAMATH_CALUDE_existence_of_n_div_prime_count_l2473_247335

/-- π(x) denotes the number of prime numbers less than or equal to x -/
def prime_counting_function (x : ℕ) : ℕ := sorry

/-- For any integer m > 1, there exists an integer n > 1 such that n/π(n) = m -/
theorem existence_of_n_div_prime_count (m : ℕ) (h : m > 1) : 
  ∃ n : ℕ, n > 1 ∧ n = m * prime_counting_function n :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_div_prime_count_l2473_247335


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_l2473_247301

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 44 ↔ n * (n + 1) ≤ 2000 := by sorry

#check max_consecutive_integers_sum

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_l2473_247301


namespace NUMINAMATH_CALUDE_min_value_x_l2473_247304

theorem min_value_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ 2 * Real.log 3 + (1/3) * Real.log x + 1) :
  x ≥ 27 * Real.exp (3/2) := by
sorry

end NUMINAMATH_CALUDE_min_value_x_l2473_247304


namespace NUMINAMATH_CALUDE_lake_circumference_difference_is_680_l2473_247366

/-- The difference between the circumferences of two lakes -/
def lake_circumference_difference : ℕ :=
  let eastern_trees : ℕ := 96
  let eastern_interval : ℕ := 10
  let western_trees : ℕ := 82
  let western_interval : ℕ := 20
  let eastern_circumference := eastern_trees * eastern_interval
  let western_circumference := western_trees * western_interval
  western_circumference - eastern_circumference

/-- Theorem stating that the difference between the circumferences of the two lakes is 680 meters -/
theorem lake_circumference_difference_is_680 : lake_circumference_difference = 680 := by
  sorry

end NUMINAMATH_CALUDE_lake_circumference_difference_is_680_l2473_247366


namespace NUMINAMATH_CALUDE_parents_gift_cost_l2473_247349

def total_budget : ℕ := 100
def num_friends : ℕ := 8
def friend_gift_cost : ℕ := 9
def num_parents : ℕ := 2

theorem parents_gift_cost (parent_gift_cost : ℕ) : 
  parent_gift_cost * num_parents + num_friends * friend_gift_cost = total_budget →
  parent_gift_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_parents_gift_cost_l2473_247349


namespace NUMINAMATH_CALUDE_expression_evaluation_l2473_247376

/-- Evaluates the expression 2x^y + 5y^x - z^2 for given x, y, and z values -/
def evaluate (x y z : ℕ) : ℕ :=
  2 * (x ^ y) + 5 * (y ^ x) - (z ^ 2)

/-- Theorem stating that the expression evaluates to 42 for x=3, y=2, and z=4 -/
theorem expression_evaluation :
  evaluate 3 2 4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2473_247376


namespace NUMINAMATH_CALUDE_product_expansion_l2473_247399

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * ((8 / x) + 12 * x^3 - (2 / x^2)) = (6 / x) + 9 * x^3 - (3 / (2 * x^2)) := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2473_247399


namespace NUMINAMATH_CALUDE_cycle_alignment_l2473_247372

def letter_cycle_length : ℕ := 5
def digit_cycle_length : ℕ := 4

theorem cycle_alignment :
  Nat.lcm letter_cycle_length digit_cycle_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_cycle_alignment_l2473_247372


namespace NUMINAMATH_CALUDE_smallest_k_sum_squares_divisible_250_l2473_247338

/-- Sum of squares formula -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Predicate to check if a number is divisible by 250 -/
def divisible_by_250 (n : ℕ) : Prop := ∃ m : ℕ, n = 250 * m

theorem smallest_k_sum_squares_divisible_250 :
  (∀ k < 375, ¬(divisible_by_250 (sum_of_squares k))) ∧
  (divisible_by_250 (sum_of_squares 375)) := by sorry

end NUMINAMATH_CALUDE_smallest_k_sum_squares_divisible_250_l2473_247338


namespace NUMINAMATH_CALUDE_problem_solution_l2473_247326

theorem problem_solution (x : ℚ) : 
  4 * x - 8 = 13 * x + 3 → 5 * (x - 2) = -145 / 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2473_247326


namespace NUMINAMATH_CALUDE_triangle_longest_side_range_l2473_247347

/-- Given a rope of length l that can exactly enclose two congruent triangles,
    prove that the longest side x of one of the triangles satisfies l/6 ≤ x < l/4 -/
theorem triangle_longest_side_range (l : ℝ) (x y z : ℝ) :
  l > 0 →
  x > 0 ∧ y > 0 ∧ z > 0 →
  x + y + z = l / 2 →
  x ≥ y ∧ x ≥ z →
  l / 6 ≤ x ∧ x < l / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_longest_side_range_l2473_247347


namespace NUMINAMATH_CALUDE_product_evaluation_l2473_247314

theorem product_evaluation (n : ℤ) (h : n = 3) : 
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) + 10 = 130 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2473_247314


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2473_247334

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y a : ℝ) : Prop := (x+3)^2 + (y-a)^2 = 16

-- Define the tangency condition
def tangent (a : ℝ) : Prop := ∃ x y, circle1 x y ∧ circle2 x y a

-- Define the cube and sphere relationship
def cube_on_sphere (a : ℝ) : Prop := 
  ∃ r, r = a * Real.sqrt 3 / 2

-- Main theorem
theorem sphere_surface_area (a : ℝ) :
  a > 0 → tangent a → cube_on_sphere a → 4 * Real.pi * (a * Real.sqrt 3)^2 = 48 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2473_247334


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2473_247330

theorem complex_number_quadrant (z : ℂ) : iz = -1 + I → z.re > 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2473_247330


namespace NUMINAMATH_CALUDE_triangle_similarity_criterion_l2473_247313

theorem triangle_similarity_criterion (a b c a₁ b₁ c₁ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  (∃ k : ℝ, k > 0 ∧ a₁ = k * a ∧ b₁ = k * b ∧ c₁ = k * c) ↔
  Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) =
    Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_criterion_l2473_247313


namespace NUMINAMATH_CALUDE_right_triangle_area_l2473_247357

/-- The area of a right triangle with vertices at (-3,0), (0,2), and (0,0) is 3 square units. -/
theorem right_triangle_area : 
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (0, 2)
  let C : ℝ × ℝ := (0, 0)
  -- Assume the triangle is right-angled
  (B.1 - C.1) * (A.2 - C.2) = (A.1 - C.1) * (B.2 - C.2) →
  -- The area of the triangle
  1/2 * |A.1 - C.1| * |B.2 - C.2| = 3 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_area_l2473_247357


namespace NUMINAMATH_CALUDE_transport_cost_bounds_l2473_247308

/-- Represents the transportation problem with cities A, B, C, D, and E. -/
structure TransportProblem where
  trucksA : ℕ := 10
  trucksB : ℕ := 10
  trucksC : ℕ := 8
  trucksToD : ℕ := 18
  trucksToE : ℕ := 10
  costAD : ℕ := 200
  costAE : ℕ := 800
  costBD : ℕ := 300
  costBE : ℕ := 700
  costCD : ℕ := 400
  costCE : ℕ := 500

/-- Calculates the total transportation cost given the number of trucks from A and B to D. -/
def totalCost (p : TransportProblem) (x : ℕ) : ℕ :=
  p.costAD * x + p.costBD * x + p.costCD * (p.trucksToD - 2*x) +
  p.costAE * (p.trucksA - x) + p.costBE * (p.trucksB - x) + p.costCE * (x + x - p.trucksToE)

/-- Theorem stating the minimum and maximum transportation costs. -/
theorem transport_cost_bounds (p : TransportProblem) :
  ∃ (xMin xMax : ℕ), 
    (∀ x, 5 ≤ x ∧ x ≤ 9 → totalCost p x ≥ totalCost p xMin) ∧
    (∀ x, 5 ≤ x ∧ x ≤ 9 → totalCost p x ≤ totalCost p xMax) ∧
    totalCost p xMin = 10000 ∧
    totalCost p xMax = 13200 :=
  sorry

end NUMINAMATH_CALUDE_transport_cost_bounds_l2473_247308


namespace NUMINAMATH_CALUDE_odd_function_value_at_negative_one_l2473_247331

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value_at_negative_one
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_nonneg : ∀ x ≥ 0, f x = 2^x + 2*x + f 0)
  : f (-1) = -3 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_value_at_negative_one_l2473_247331


namespace NUMINAMATH_CALUDE_two_line_relationships_l2473_247342

-- Define a type for lines in a plane
def Line : Type := sorry

-- Define a plane
def Plane : Type := sorry

-- Define what it means for two lines to be in the same plane
def inSamePlane (l1 l2 : Line) (p : Plane) : Prop := sorry

-- Define what it means for two lines to be non-overlapping
def nonOverlapping (l1 l2 : Line) : Prop := sorry

-- Define what it means for two lines to intersect
def intersecting (l1 l2 : Line) : Prop := sorry

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- The theorem to be proved
theorem two_line_relationships (l1 l2 : Line) (p : Plane) :
  inSamePlane l1 l2 p → nonOverlapping l1 l2 → intersecting l1 l2 ∨ parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_two_line_relationships_l2473_247342


namespace NUMINAMATH_CALUDE_expression_perfect_square_iff_l2473_247340

def factorial (n : ℕ) : ℕ := Nat.factorial n

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def expression (n : ℕ) : ℕ := 
  (List.range (2*n + 1)).foldl (λ acc i => acc * factorial i) 1 / factorial (n + 1)

theorem expression_perfect_square_iff (n : ℕ) : 
  is_perfect_square (expression n) ↔ 
  (∃ k : ℕ, n = 4 * k * (k + 1)) ∨ (∃ k : ℕ, n = 2 * k * k - 1) :=
sorry

end NUMINAMATH_CALUDE_expression_perfect_square_iff_l2473_247340


namespace NUMINAMATH_CALUDE_vector_basis_l2473_247343

def e₁ : ℝ × ℝ := (-1, 3)
def e₂ : ℝ × ℝ := (5, -2)

theorem vector_basis : LinearIndependent ℝ ![e₁, e₂] ∧ Submodule.span ℝ {e₁, e₂} = ⊤ := by
  sorry

end NUMINAMATH_CALUDE_vector_basis_l2473_247343


namespace NUMINAMATH_CALUDE_platform_length_l2473_247371

/-- Given a train of length 300 meters, which takes 39 seconds to cross a platform
    and 16 seconds to cross a signal pole, prove that the length of the platform
    is 431.25 meters. -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 16) :
  let speed := train_length / time_cross_pole
  let platform_length := speed * time_cross_platform - train_length
  platform_length = 431.25 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l2473_247371


namespace NUMINAMATH_CALUDE_only_four_sevenths_greater_than_half_l2473_247323

theorem only_four_sevenths_greater_than_half : 
  (2 : ℚ) / 5 ≤ 1 / 2 ∧ 
  (3 : ℚ) / 7 ≤ 1 / 2 ∧ 
  (4 : ℚ) / 7 > 1 / 2 ∧ 
  (3 : ℚ) / 8 ≤ 1 / 2 ∧ 
  (4 : ℚ) / 9 ≤ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_only_four_sevenths_greater_than_half_l2473_247323


namespace NUMINAMATH_CALUDE_correct_packs_for_spoons_l2473_247351

/-- Calculates the number of packs needed to buy a specific number of spoons -/
def packs_needed (total_utensils_per_pack : ℕ) (spoons_wanted : ℕ) : ℕ :=
  let spoons_per_pack := total_utensils_per_pack / 3
  (spoons_wanted + spoons_per_pack - 1) / spoons_per_pack

theorem correct_packs_for_spoons :
  packs_needed 30 50 = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_packs_for_spoons_l2473_247351


namespace NUMINAMATH_CALUDE_bankers_discount_l2473_247336

/-- Banker's discount calculation -/
theorem bankers_discount 
  (PV : ℝ) -- Present Value
  (BG : ℝ) -- Banker's Gain
  (n : ℕ) -- Total number of years
  (r1 : ℝ) -- Interest rate for first half of the period
  (r2 : ℝ) -- Interest rate for second half of the period
  (h : n = 8) -- The sum is due 8 years hence
  (h1 : r1 = 0.10) -- Interest rate is 10% for the first 4 years
  (h2 : r2 = 0.12) -- Interest rate is 12% for the remaining 4 years
  (h3 : BG = 900) -- The banker's gain is Rs. 900
  : ∃ (BD : ℝ), BD = BG + ((PV * (1 + r1) ^ (n / 2)) * (1 + r2) ^ (n / 2) - PV) :=
by sorry

end NUMINAMATH_CALUDE_bankers_discount_l2473_247336


namespace NUMINAMATH_CALUDE_solution_x_is_three_fourths_l2473_247312

-- Define the * operation
def star (a b : ℝ) : ℝ := 4 * a - 2 * b

-- State the theorem
theorem solution_x_is_three_fourths :
  ∃ x : ℝ, star 7 (star 3 (x - 1)) = 3 ∧ x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_is_three_fourths_l2473_247312


namespace NUMINAMATH_CALUDE_product_equals_square_l2473_247367

theorem product_equals_square : 50 * 39.96 * 3.996 * 500 = 3996^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l2473_247367


namespace NUMINAMATH_CALUDE_expression_evaluation_l2473_247348

theorem expression_evaluation :
  let y : ℚ := 1/2
  (y + 1) * (y - 1) + (2*y - 1)^2 - 2*y*(2*y - 1) = -3/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2473_247348


namespace NUMINAMATH_CALUDE_distinct_necklaces_count_l2473_247325

/-- Represents a necklace made of white and black beads -/
structure Necklace :=
  (white_beads : ℕ)
  (black_beads : ℕ)

/-- Determines if two necklaces are equivalent under rotation and flipping -/
def necklace_equivalent (n1 n2 : Necklace) : Prop :=
  (n1.white_beads = n2.white_beads) ∧ (n1.black_beads = n2.black_beads)

/-- Counts the number of distinct necklaces with given white and black beads -/
def count_distinct_necklaces (white black : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of distinct necklaces with 5 white and 2 black beads is 3 -/
theorem distinct_necklaces_count :
  count_distinct_necklaces 5 2 = 3 :=
sorry

end NUMINAMATH_CALUDE_distinct_necklaces_count_l2473_247325


namespace NUMINAMATH_CALUDE_jennis_age_l2473_247321

theorem jennis_age (sum diff : ℕ) (h_sum : sum = 70) (h_diff : diff = 32) :
  ∃ (age_jenni age_bai : ℕ), age_jenni + age_bai = sum ∧ age_bai - age_jenni = diff ∧ age_jenni = 19 :=
by sorry

end NUMINAMATH_CALUDE_jennis_age_l2473_247321


namespace NUMINAMATH_CALUDE_point_B_coordinates_l2473_247387

-- Define the vector a
def a : ℝ × ℝ := (1, 1)

-- Define point A
def A : ℝ × ℝ := (-3, -1)

-- Define the line y = 2x
def line (x : ℝ) : ℝ × ℝ := (x, 2 * x)

-- Define vector parallelism
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem point_B_coordinates :
  ∃ (x : ℝ), 
    let B := line x
    parallel (a) (B.1 - A.1, B.2 - A.2) →
    B = (2, 4) := by sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l2473_247387


namespace NUMINAMATH_CALUDE_total_books_on_shelf_l2473_247317

/-- Given a shelf with history books, geography books, and math books,
    prove that the total number of books is 100. -/
theorem total_books_on_shelf (history_books : ℕ) (geography_books : ℕ) (math_books : ℕ)
    (h1 : history_books = 32)
    (h2 : geography_books = 25)
    (h3 : math_books = 43) :
    history_books + geography_books + math_books = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_books_on_shelf_l2473_247317


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_ratio_l2473_247394

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Length of the smaller base -/
  ab : ℝ
  /-- Length of the larger base -/
  cd : ℝ
  /-- Length of the diagonal AC -/
  ac : ℝ
  /-- Height of the trapezoid (altitude from D to AB) -/
  h : ℝ
  /-- The smaller base is less than the larger base -/
  ab_lt_cd : ab < cd
  /-- The diagonal AC is twice the length of the larger base CD -/
  ac_eq_2cd : ac = 2 * cd
  /-- The smaller base AB equals the height of the trapezoid -/
  ab_eq_h : ab = h

/-- The ratio of the smaller base to the larger base in the specific isosceles trapezoid is 3:1 -/
theorem isosceles_trapezoid_ratio (t : IsoscelesTrapezoid) : t.ab / t.cd = 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_ratio_l2473_247394


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2473_247374

/-- Given that a² and √b vary inversely, if a = 3 when b = 64, then b = 18 when ab = 72 -/
theorem inverse_variation_problem (a b : ℝ) : 
  (∃ k : ℝ, ∀ a b : ℝ, a^2 * Real.sqrt b = k) →  -- a² and √b vary inversely
  (3^2 * Real.sqrt 64 = 3 * 64) →                -- a = 3 when b = 64
  (a * b = 72) →                                 -- ab = 72
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2473_247374


namespace NUMINAMATH_CALUDE_ben_and_brothers_pizza_order_l2473_247379

/-- The number of small pizzas ordered for Ben and his brothers -/
def small_pizzas_ordered (num_people : ℕ) (slices_per_person : ℕ) (large_pizza_slices : ℕ) (small_pizza_slices : ℕ) (large_pizzas_ordered : ℕ) : ℕ :=
  let total_slices_needed := num_people * slices_per_person
  let slices_from_large := large_pizzas_ordered * large_pizza_slices
  let remaining_slices := total_slices_needed - slices_from_large
  (remaining_slices + small_pizza_slices - 1) / small_pizza_slices

theorem ben_and_brothers_pizza_order :
  small_pizzas_ordered 3 12 14 8 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ben_and_brothers_pizza_order_l2473_247379


namespace NUMINAMATH_CALUDE_pencil_difference_proof_l2473_247359

def pencil_distribution (total : ℕ) (kept : ℕ) (given_to_manny : ℕ) : Prop :=
  let given_away := total - kept
  let given_to_nilo := given_away - given_to_manny
  given_to_nilo - given_to_manny = 10

theorem pencil_difference_proof :
  pencil_distribution 50 20 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_difference_proof_l2473_247359


namespace NUMINAMATH_CALUDE_five_to_five_sum_equals_five_to_six_l2473_247332

theorem five_to_five_sum_equals_five_to_six : 
  5^5 + 5^5 + 5^5 + 5^5 + 5^5 = 5^6 := by
  sorry

end NUMINAMATH_CALUDE_five_to_five_sum_equals_five_to_six_l2473_247332


namespace NUMINAMATH_CALUDE_no_natural_solutions_l2473_247306

theorem no_natural_solutions : ∀ x y z : ℕ, x^2 + y^2 + z^2 ≠ 2*x*y*z := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l2473_247306


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2473_247319

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + 2*x*y = 3) :
  x + 2*y ≥ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ + 2*x₀*y₀ = 3 ∧ x₀ + 2*y₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2473_247319


namespace NUMINAMATH_CALUDE_variance_and_shifted_average_l2473_247311

theorem variance_and_shifted_average
  (x₁ x₂ x₃ x₄ : ℝ)
  (pos_x : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0)
  (variance : (1/4) * (x₁^2 + x₂^2 + x₃^2 + x₄^2 - 16) = (1/4) * ((x₁ - (x₁ + x₂ + x₃ + x₄)/4)^2 +
                                                                  (x₂ - (x₁ + x₂ + x₃ + x₄)/4)^2 +
                                                                  (x₃ - (x₁ + x₂ + x₃ + x₄)/4)^2 +
                                                                  (x₄ - (x₁ + x₂ + x₃ + x₄)/4)^2)) :
  ((x₁ + 3) + (x₂ + 3) + (x₃ + 3) + (x₄ + 3)) / 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_variance_and_shifted_average_l2473_247311


namespace NUMINAMATH_CALUDE_odd_function_sum_l2473_247362

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f is periodic with period p if f(x + p) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_sum (f : ℝ → ℝ) (h_odd : IsOdd f) (h_period : ∀ x, f (x + 2) = -f x) (h_f1 : f 1 = 8) :
  f 2012 + f 2013 + f 2014 = 8 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l2473_247362


namespace NUMINAMATH_CALUDE_set_operation_result_l2473_247365

def set_operation (M N : Set Int) : Set Int :=
  {x | ∃ y z, y ∈ N ∧ z ∈ M ∧ x = y - z}

theorem set_operation_result :
  let M : Set Int := {0, 1, 2}
  let N : Set Int := {-2, -3}
  set_operation M N = {-2, -3, -4, -5} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_result_l2473_247365


namespace NUMINAMATH_CALUDE_four_sets_gemstones_l2473_247368

/-- Calculates the number of gemstones needed for a given number of earring sets -/
def gemstones_needed (num_sets : ℕ) : ℕ :=
  let earrings_per_set := 2
  let magnets_per_earring := 2
  let buttons_per_earring := magnets_per_earring / 2
  let gemstones_per_earring := buttons_per_earring * 3
  num_sets * earrings_per_set * gemstones_per_earring

/-- Theorem stating that 4 sets of earrings require 24 gemstones -/
theorem four_sets_gemstones : gemstones_needed 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_sets_gemstones_l2473_247368


namespace NUMINAMATH_CALUDE_first_player_wins_l2473_247389

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents a rectangle piece -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents the game board -/
structure Board where
  rows : Nat
  cols : Nat

/-- Represents a game state -/
structure GameState where
  board : Board
  currentPlayer : Player

/-- Defines a valid move in the game -/
def ValidMove (rect : Rectangle) (state : GameState) : Prop :=
  match state.currentPlayer with
  | Player.First => rect.width = 1 ∧ rect.height = 2
  | Player.Second => rect.width = 2 ∧ rect.height = 1

/-- Defines the winning condition -/
def HasWinningStrategy (player : Player) (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → Rectangle),
    ∀ (opponent_move : Rectangle),
      ValidMove opponent_move initialState →
      ∃ (final_state : GameState),
        (final_state.currentPlayer = player) ∧
        (¬∃ (move : Rectangle), ValidMove move final_state)

/-- The main theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  let initial_board : Board := { rows := 3, cols := 1000 }
  let initial_state : GameState := { board := initial_board, currentPlayer := Player.First }
  HasWinningStrategy Player.First initial_state :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l2473_247389


namespace NUMINAMATH_CALUDE_equation_solution_l2473_247361

theorem equation_solution (x y : ℝ) : 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ (y = -x - 2 ∨ y = -2 * x + 1) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2473_247361


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2473_247381

theorem arithmetic_equality : (3652 * 2487) + (979 - 45 * 13) = 9085008 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2473_247381


namespace NUMINAMATH_CALUDE_plane_Q_satisfies_conditions_l2473_247383

/-- Plane represented by its normal vector and constant term -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Line represented by a point and a direction vector -/
structure Line where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

def plane_intersection (p1 p2 : Plane) : Line := sorry

def distance_point_to_plane (point : ℝ × ℝ × ℝ) (plane : Plane) : ℝ := sorry

def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem plane_Q_satisfies_conditions : 
  let π₁ : Plane := ⟨2, -3, 4, -5⟩
  let π₂ : Plane := ⟨3, 1, -2, -1⟩
  let Q : Plane := ⟨6, -1, 10, -11⟩
  let intersection := plane_intersection π₁ π₂
  let point := (1, 2, 3)
  line_in_plane intersection Q ∧ 
  distance_point_to_plane point Q = 3 / Real.sqrt 5 ∧
  Q ≠ π₁ ∧ 
  Q ≠ π₂ := by
  sorry


end NUMINAMATH_CALUDE_plane_Q_satisfies_conditions_l2473_247383


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2473_247318

/-- 
Given a quadratic equation x^2 - 2x + k = 0, 
if it has two equal real roots, then k = 1.
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + k = 0 → y = x) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2473_247318


namespace NUMINAMATH_CALUDE_problem_solution_l2473_247393

theorem problem_solution (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 72) :
  let y := ((x - 3)^2 * (x + 4)) / (3*x - 4)
  y = 2 ∨ y = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2473_247393


namespace NUMINAMATH_CALUDE_circle_center_line_max_ab_l2473_247360

theorem circle_center_line_max_ab (a b : ℝ) :
  let circle := (fun (x y : ℝ) => x^2 + y^2 + 2*x - 4*y + 1 = 0)
  let center_line := (fun (x y : ℝ) => a*x - b*y + 1 = 0)
  let center := (-1, 2)
  (∀ x y, circle x y ↔ (x + 1)^2 + (y - 2)^2 = 4) →
  center_line (-1) 2 →
  (∀ k, k * a * b ≤ 1/8) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_line_max_ab_l2473_247360


namespace NUMINAMATH_CALUDE_alcohol_mixture_problem_l2473_247388

-- Define the initial volume of the solution
def initial_volume : ℝ := 6

-- Define the volume of pure alcohol added
def added_alcohol : ℝ := 1.8

-- Define the final alcohol percentage
def final_percentage : ℝ := 0.5

-- Define the initial alcohol percentage (to be proven)
def initial_percentage : ℝ := 0.35

-- Theorem statement
theorem alcohol_mixture_problem :
  initial_volume * initial_percentage + added_alcohol =
  (initial_volume + added_alcohol) * final_percentage := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_problem_l2473_247388


namespace NUMINAMATH_CALUDE_equation_solution_l2473_247305

theorem equation_solution : 
  ∀ x : ℝ, (40 / 60 : ℝ) = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2473_247305


namespace NUMINAMATH_CALUDE_luncheon_cost_theorem_l2473_247352

/-- Represents the cost of items in a luncheon -/
structure LuncheonCost where
  sandwich : ℚ
  coffee : ℚ
  pie : ℚ

/-- The conditions of the problem -/
axiom luncheon_condition_1 : ∀ (c : LuncheonCost), 
  5 * c.sandwich + 8 * c.coffee + c.pie = 5.25

axiom luncheon_condition_2 : ∀ (c : LuncheonCost), 
  7 * c.sandwich + 12 * c.coffee + c.pie = 7.35

/-- The theorem to be proved -/
theorem luncheon_cost_theorem (c : LuncheonCost) : 
  c.sandwich + c.coffee + c.pie = 1.05 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_cost_theorem_l2473_247352


namespace NUMINAMATH_CALUDE_nine_qualified_possible_l2473_247378

/-- Represents the probability of a product passing inspection -/
def pass_rate : ℝ := 0.9

/-- The number of products drawn for inspection -/
def sample_size : ℕ := 10

/-- Represents whether it's possible to have exactly 9 qualified products in a sample of 10 -/
def possible_nine_qualified : Prop :=
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧ p ≠ 0 ∧ p ≠ 1

theorem nine_qualified_possible (h : pass_rate = 0.9) : possible_nine_qualified := by
  sorry

#check nine_qualified_possible

end NUMINAMATH_CALUDE_nine_qualified_possible_l2473_247378


namespace NUMINAMATH_CALUDE_max_term_a_l2473_247354

def a (n : ℕ+) : ℚ := n / (n^2 + 2020)

theorem max_term_a :
  ∃ (k : ℕ+), k = 45 ∧ 
  (∀ (n : ℕ+), a n ≤ a k) ∧
  a k = 45 / 4045 := by sorry

end NUMINAMATH_CALUDE_max_term_a_l2473_247354


namespace NUMINAMATH_CALUDE_z_local_minimum_l2473_247337

-- Define the function
def z (x y : ℝ) : ℝ := x^3 + y^3 - 3*x*y

-- State the theorem
theorem z_local_minimum :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x y : ℝ),
    (x - 1)^2 + (y - 1)^2 < ε^2 → z x y ≥ z 1 1 ∧ z 1 1 = -1 :=
sorry

end NUMINAMATH_CALUDE_z_local_minimum_l2473_247337


namespace NUMINAMATH_CALUDE_even_heads_probability_l2473_247395

def probability_even_heads (p1 p2 : ℚ) (n1 n2 : ℕ) : ℚ :=
  let P1 := (1 + ((1 - 2*p1) / (1 - p1))^n1) / 2
  let P2 := (1 + ((1 - 2*p2) / (1 - p2))^n2) / 2
  P1 * P2 + (1 - P1) * (1 - P2)

theorem even_heads_probability :
  probability_even_heads (3/4) (1/2) 40 10 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_even_heads_probability_l2473_247395


namespace NUMINAMATH_CALUDE_minimum_speed_to_clear_building_l2473_247300

/-- The minimum speed required for a stone to clear a building -/
theorem minimum_speed_to_clear_building 
  (g H l : ℝ) (α : ℝ) (h_g : g > 0) (h_H : H > 0) (h_l : l > 0) 
  (h_α : 0 < α ∧ α < π / 2) : 
  ∃ (v₀ : ℝ), v₀ = Real.sqrt (g * (2 * H + l * (1 - Real.sin α) / Real.cos α)) ∧ 
  (∀ (v : ℝ), v > v₀ → 
    ∃ (trajectory : ℝ → ℝ), 
      (∀ x, trajectory x ≤ H + Real.tan α * (l - x)) ∧
      (∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ l ∧ 
        trajectory x₁ = H ∧ trajectory x₂ = H + Real.tan α * (l - x₂))) :=
sorry

end NUMINAMATH_CALUDE_minimum_speed_to_clear_building_l2473_247300


namespace NUMINAMATH_CALUDE_meeting_distance_l2473_247341

-- Define the speeds and distance
def xiaoBinSpeed : ℝ := 15
def xiaoMingSpeed : ℝ := 5
def distanceToSchool : ℝ := 30

-- Define the theorem
theorem meeting_distance :
  let totalDistance : ℝ := 2 * distanceToSchool
  let meetingTime : ℝ := totalDistance / (xiaoBinSpeed + xiaoMingSpeed)
  let xiaoMingDistance : ℝ := meetingTime * xiaoMingSpeed
  xiaoMingDistance = 15 := by sorry

end NUMINAMATH_CALUDE_meeting_distance_l2473_247341


namespace NUMINAMATH_CALUDE_function_inequality_l2473_247391

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2473_247391


namespace NUMINAMATH_CALUDE_combinable_with_sqrt_three_l2473_247375

theorem combinable_with_sqrt_three : ∃! x : ℝ, x > 0 ∧ 
  (x = Real.sqrt (3^2) ∨ x = Real.sqrt 27 ∨ x = Real.sqrt 30 ∨ x = Real.sqrt (2/3)) ∧
  ∃ (r : ℚ), x = r * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_combinable_with_sqrt_three_l2473_247375


namespace NUMINAMATH_CALUDE_expression_evaluation_l2473_247380

theorem expression_evaluation :
  let x : ℝ := 2
  (x^2 * (x - 1) - x * (x^2 + x - 1)) = -6 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2473_247380


namespace NUMINAMATH_CALUDE_total_chocolates_l2473_247353

/-- The number of large boxes in the warehouse -/
def large_boxes : ℕ := 150

/-- The number of small boxes in each large box -/
def small_boxes_per_large : ℕ := 45

/-- The number of chocolate bars in each small box -/
def chocolates_per_small : ℕ := 35

/-- Theorem stating the total number of chocolate bars in the warehouse -/
theorem total_chocolates : 
  large_boxes * small_boxes_per_large * chocolates_per_small = 236250 := by
  sorry

end NUMINAMATH_CALUDE_total_chocolates_l2473_247353


namespace NUMINAMATH_CALUDE_concert_cost_l2473_247369

theorem concert_cost (ticket_price : ℚ) (processing_fee_rate : ℚ) 
  (parking_fee : ℚ) (entrance_fee : ℚ) (num_people : ℕ) :
  ticket_price = 50 ∧ 
  processing_fee_rate = 0.15 ∧ 
  parking_fee = 10 ∧ 
  entrance_fee = 5 ∧ 
  num_people = 2 → 
  (ticket_price + ticket_price * processing_fee_rate) * num_people + 
  parking_fee + entrance_fee * num_people = 135 :=
by sorry

end NUMINAMATH_CALUDE_concert_cost_l2473_247369
