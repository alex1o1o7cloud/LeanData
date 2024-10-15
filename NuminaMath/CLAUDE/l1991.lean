import Mathlib

namespace NUMINAMATH_CALUDE_sugar_recipe_reduction_sugar_mixed_number_l1991_199174

theorem sugar_recipe_reduction : 
  let original_sugar : ℚ := 31/4
  let reduced_sugar : ℚ := (1/3) * original_sugar
  reduced_sugar = 31/12 := by sorry

theorem sugar_mixed_number :
  let reduced_sugar : ℚ := 31/12
  ∃ (whole : ℕ) (numerator : ℕ) (denominator : ℕ),
    reduced_sugar = whole + (numerator : ℚ) / denominator ∧
    whole = 2 ∧ numerator = 7 ∧ denominator = 12 := by sorry

end NUMINAMATH_CALUDE_sugar_recipe_reduction_sugar_mixed_number_l1991_199174


namespace NUMINAMATH_CALUDE_solve_pickle_problem_l1991_199179

def pickle_problem (total_jars total_cucumbers initial_vinegar pickles_per_jar vinegar_per_jar remaining_vinegar : ℕ) : Prop :=
  let used_vinegar := initial_vinegar - remaining_vinegar
  let filled_jars := used_vinegar / vinegar_per_jar
  let total_pickles := filled_jars * pickles_per_jar
  let pickles_per_cucumber := total_pickles / total_cucumbers
  pickles_per_cucumber = 4 ∧
  total_jars = 4 ∧
  total_cucumbers = 10 ∧
  initial_vinegar = 100 ∧
  pickles_per_jar = 12 ∧
  vinegar_per_jar = 10 ∧
  remaining_vinegar = 60

theorem solve_pickle_problem :
  ∃ (total_jars total_cucumbers initial_vinegar pickles_per_jar vinegar_per_jar remaining_vinegar : ℕ),
  pickle_problem total_jars total_cucumbers initial_vinegar pickles_per_jar vinegar_per_jar remaining_vinegar :=
by
  sorry

end NUMINAMATH_CALUDE_solve_pickle_problem_l1991_199179


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1991_199153

theorem greatest_divisor_with_remainders : Nat.gcd (1657 - 6) (2037 - 5) = 127 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1991_199153


namespace NUMINAMATH_CALUDE_nonzero_digits_count_l1991_199194

-- Define the fraction
def f : ℚ := 84 / (2^5 * 5^9)

-- Define a function to count non-zero digits after the decimal point
noncomputable def count_nonzero_digits_after_decimal (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem nonzero_digits_count :
  count_nonzero_digits_after_decimal f = 2 := by sorry

end NUMINAMATH_CALUDE_nonzero_digits_count_l1991_199194


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1991_199172

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 1/3 ∧ 
  (∀ x : ℝ, 3*x^2 - 4*x + 1 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1991_199172


namespace NUMINAMATH_CALUDE_integer_less_than_sqrt5_l1991_199177

theorem integer_less_than_sqrt5 : ∃ z : ℤ, |z| < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_integer_less_than_sqrt5_l1991_199177


namespace NUMINAMATH_CALUDE_max_value_of_a_l1991_199104

-- Define the condition function
def condition (x : ℝ) : Prop := x^2 - 2*x - 3 > 0

-- Define the theorem
theorem max_value_of_a :
  (∃ a : ℝ, ∀ x : ℝ, x < a → condition x) ∧
  (∀ a : ℝ, ∃ x : ℝ, condition x ∧ x ≥ a) →
  (∀ a : ℝ, (∀ x : ℝ, x < a → condition x) → a ≤ -1) ∧
  (∀ x : ℝ, x < -1 → condition x) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1991_199104


namespace NUMINAMATH_CALUDE_jessica_roses_thrown_away_l1991_199123

/-- The number of roses Jessica threw away -/
def roses_thrown_away (initial : ℕ) (added : ℕ) (final : ℕ) : ℕ :=
  initial + added - final

/-- Proof that Jessica threw away 4 roses -/
theorem jessica_roses_thrown_away :
  roses_thrown_away 2 25 23 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jessica_roses_thrown_away_l1991_199123


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1991_199188

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I + 1) / (Complex.I - 1) = Complex.mk a b → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1991_199188


namespace NUMINAMATH_CALUDE_hamburgers_left_over_l1991_199142

theorem hamburgers_left_over (total : ℕ) (served : ℕ) (left_over : ℕ) : 
  total = 9 → served = 3 → left_over = total - served → left_over = 6 := by
sorry

end NUMINAMATH_CALUDE_hamburgers_left_over_l1991_199142


namespace NUMINAMATH_CALUDE_total_stamps_l1991_199147

theorem total_stamps (x y : ℕ) (hx : x = 34) (hy : y = x + 44) : x + y = 112 := by
  sorry

end NUMINAMATH_CALUDE_total_stamps_l1991_199147


namespace NUMINAMATH_CALUDE_janice_age_problem_l1991_199139

theorem janice_age_problem :
  ∀ x : ℕ,
  (x + 12 = 8 * (x - 2)) → x = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_janice_age_problem_l1991_199139


namespace NUMINAMATH_CALUDE_square_diff_cubed_l1991_199159

theorem square_diff_cubed : (5^2 - 4^2)^3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_cubed_l1991_199159


namespace NUMINAMATH_CALUDE_horse_tile_problem_representation_l1991_199167

/-- Represents the equation for the horse and tile problem -/
def horse_tile_equation (x : ℝ) : Prop :=
  3 * x + (1/3) * (100 - x) = 100

/-- The total number of horses -/
def total_horses : ℝ := 100

/-- The total number of tiles -/
def total_tiles : ℝ := 100

/-- The number of tiles a big horse can pull -/
def big_horse_capacity : ℝ := 3

/-- The number of small horses needed to pull one tile -/
def small_horses_per_tile : ℝ := 3

/-- Theorem stating that the equation correctly represents the problem -/
theorem horse_tile_problem_representation :
  ∀ x, x ≥ 0 ∧ x ≤ total_horses →
  horse_tile_equation x ↔
    (x * big_horse_capacity + (total_horses - x) / small_horses_per_tile = total_tiles) :=
by sorry

end NUMINAMATH_CALUDE_horse_tile_problem_representation_l1991_199167


namespace NUMINAMATH_CALUDE_exists_counterfeit_finding_algorithm_l1991_199185

/-- Represents a coin, which can be either genuine or counterfeit -/
inductive Coin
| genuine : Coin
| counterfeit : Coin

/-- Represents the result of a weighing operation -/
inductive WeighResult
| balanced : WeighResult
| leftLighter : WeighResult
| rightLighter : WeighResult

/-- A function that simulates weighing two sets of coins -/
def weigh (left : List Coin) (right : List Coin) : WeighResult :=
  sorry

/-- The type of an algorithm to find the counterfeit coin -/
def FindCounterfeitAlgorithm := List Coin → Coin

/-- Theorem stating that there exists an algorithm to find the counterfeit coin -/
theorem exists_counterfeit_finding_algorithm :
  ∃ (algo : FindCounterfeitAlgorithm),
    ∀ (coins : List Coin),
      coins.length = 9 →
      (∃! (c : Coin), c ∈ coins ∧ c = Coin.counterfeit) →
      algo coins = Coin.counterfeit :=
sorry

end NUMINAMATH_CALUDE_exists_counterfeit_finding_algorithm_l1991_199185


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l1991_199162

theorem range_of_a_minus_b (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 2) :
  ∀ x, -3 < x ∧ x < 0 ↔ ∃ a b, -1 < a ∧ a < b ∧ b < 2 ∧ x = a - b :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l1991_199162


namespace NUMINAMATH_CALUDE_book_pages_difference_l1991_199173

theorem book_pages_difference : 
  let purple_books : ℕ := 8
  let orange_books : ℕ := 7
  let blue_books : ℕ := 5
  let purple_pages_per_book : ℕ := 320
  let orange_pages_per_book : ℕ := 640
  let blue_pages_per_book : ℕ := 450
  let total_purple_pages := purple_books * purple_pages_per_book
  let total_orange_pages := orange_books * orange_pages_per_book
  let total_blue_pages := blue_books * blue_pages_per_book
  let total_orange_blue_pages := total_orange_pages + total_blue_pages
  total_orange_blue_pages - total_purple_pages = 4170 := by
sorry

end NUMINAMATH_CALUDE_book_pages_difference_l1991_199173


namespace NUMINAMATH_CALUDE_min_sum_of_product_2310_l1991_199131

theorem min_sum_of_product_2310 (a b c : ℕ+) : 
  a * b * c = 2310 → (∀ x y z : ℕ+, x * y * z = 2310 → a + b + c ≤ x + y + z) → a + b + c = 40 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2310_l1991_199131


namespace NUMINAMATH_CALUDE_function_inequality_l1991_199181

noncomputable def f (x : ℝ) := x^2 - Real.pi * x

theorem function_inequality (α β γ : ℝ) 
  (h_α : α ∈ Set.Ioo 0 Real.pi) 
  (h_β : β ∈ Set.Ioo 0 Real.pi) 
  (h_γ : γ ∈ Set.Ioo 0 Real.pi)
  (h_sin_α : Real.sin α = 1/3)
  (h_tan_β : Real.tan β = 5/4)
  (h_cos_γ : Real.cos γ = -1/3) :
  f α > f β ∧ f β > f γ := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1991_199181


namespace NUMINAMATH_CALUDE_student_marks_proof_l1991_199110

/-- Given a student's marks in mathematics, physics, and chemistry,
    prove that the total marks in mathematics and physics is 50. -/
theorem student_marks_proof (M P C : ℕ) : 
  C = P + 20 →
  (M + C) / 2 = 35 →
  M + P = 50 := by
sorry

end NUMINAMATH_CALUDE_student_marks_proof_l1991_199110


namespace NUMINAMATH_CALUDE_parabola_transformation_l1991_199180

-- Define the original function
def f (x : ℝ) : ℝ := (x - 3)^2 - 4

-- Define the transformation
def transform (g : ℝ → ℝ) : ℝ → ℝ := λ x => g (x - 1) + 2

-- Define the expected result function
def expected_result (x : ℝ) : ℝ := (x - 4)^2 - 2

-- Theorem statement
theorem parabola_transformation :
  ∀ x, transform f x = expected_result x :=
sorry

end NUMINAMATH_CALUDE_parabola_transformation_l1991_199180


namespace NUMINAMATH_CALUDE_lcm_factor_14_l1991_199160

theorem lcm_factor_14 (A B : ℕ+) (h1 : Nat.gcd A B = 16) (h2 : A = 224) :
  ∃ (X Y : ℕ+), Nat.lcm A B = 16 * X * Y ∧ (X = 14 ∨ Y = 14) := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_14_l1991_199160


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l1991_199141

theorem factorization_difference_of_squares (m x y : ℝ) : m * x^2 - m * y^2 = m * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l1991_199141


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1991_199143

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 6 + Real.sqrt 5) =
  6 * Real.sqrt 2 - 2 * Real.sqrt 15 + Real.sqrt 30 - 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1991_199143


namespace NUMINAMATH_CALUDE_northton_capsule_depth_l1991_199136

theorem northton_capsule_depth (southton_depth : ℝ) (northton_offset : ℝ) : 
  southton_depth = 15 →
  northton_offset = 12 →
  (4 * southton_depth + northton_offset) = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_northton_capsule_depth_l1991_199136


namespace NUMINAMATH_CALUDE_zoo_bus_distribution_l1991_199138

theorem zoo_bus_distribution (total_people : ℕ) (num_buses : ℕ) 
  (h1 : total_people = 219) (h2 : num_buses = 3) :
  total_people / num_buses = 73 := by
  sorry

end NUMINAMATH_CALUDE_zoo_bus_distribution_l1991_199138


namespace NUMINAMATH_CALUDE_min_workers_for_profit_l1991_199114

/-- Represents the company's financial model -/
structure CompanyModel where
  maintenance_fee : ℕ  -- Daily maintenance fee in dollars
  hourly_wage : ℕ      -- Hourly wage per worker in dollars
  widgets_per_hour : ℕ -- Widgets produced per worker per hour
  widget_price : ℚ     -- Selling price per widget in dollars
  work_hours : ℕ       -- Work hours per day

/-- Calculates the daily cost for a given number of workers -/
def daily_cost (model : CompanyModel) (workers : ℕ) : ℕ :=
  model.maintenance_fee + model.hourly_wage * workers * model.work_hours

/-- Calculates the daily revenue for a given number of workers -/
def daily_revenue (model : CompanyModel) (workers : ℕ) : ℚ :=
  (model.widgets_per_hour : ℚ) * model.widget_price * (workers : ℚ) * (model.work_hours : ℚ)

/-- Theorem stating the minimum number of workers needed for profit -/
theorem min_workers_for_profit (model : CompanyModel) 
  (h_maintenance : model.maintenance_fee = 600)
  (h_wage : model.hourly_wage = 20)
  (h_widgets : model.widgets_per_hour = 6)
  (h_price : model.widget_price = 7/2)
  (h_hours : model.work_hours = 7) :
  ∃ n : ℕ, (∀ m : ℕ, m ≥ n → daily_revenue model m > daily_cost model m) ∧
           (∀ m : ℕ, m < n → daily_revenue model m ≤ daily_cost model m) ∧
           n = 86 :=
sorry

end NUMINAMATH_CALUDE_min_workers_for_profit_l1991_199114


namespace NUMINAMATH_CALUDE_power_difference_squared_l1991_199105

theorem power_difference_squared (n : ℕ) :
  (5^(1001 : ℕ) + 6^(1002 : ℕ))^2 - (5^(1001 : ℕ) - 6^(1002 : ℕ))^2 = 24 * 30^(1001 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_power_difference_squared_l1991_199105


namespace NUMINAMATH_CALUDE_division_problem_l1991_199134

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 222 →
  quotient = 17 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 13 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1991_199134


namespace NUMINAMATH_CALUDE_count_integer_lengths_for_specific_triangle_l1991_199124

-- Define the right triangle DEF
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  right_angle : DE > 0 ∧ EF > 0

-- Define the function to count integer lengths
def count_integer_lengths (t : RightTriangle) : ℕ :=
  sorry

-- Theorem statement
theorem count_integer_lengths_for_specific_triangle :
  ∃ (t : RightTriangle), t.DE = 24 ∧ t.EF = 25 ∧ count_integer_lengths t = 14 :=
sorry

end NUMINAMATH_CALUDE_count_integer_lengths_for_specific_triangle_l1991_199124


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l1991_199150

theorem quadratic_form_minimum : 
  ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 9 ≥ -10 ∧ 
  ∃ x₀ y₀ : ℝ, 3 * x₀^2 + 4 * x₀ * y₀ + 2 * y₀^2 - 6 * x₀ + 8 * y₀ + 9 = -10 := by
  sorry

#check quadratic_form_minimum

end NUMINAMATH_CALUDE_quadratic_form_minimum_l1991_199150


namespace NUMINAMATH_CALUDE_inequality_proof_l1991_199186

theorem inequality_proof (m : ℕ+) (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (x^(m : ℕ) / ((1 + y) * (1 + z))) + 
  (y^(m : ℕ) / ((1 + x) * (1 + z))) + 
  (z^(m : ℕ) / ((1 + x) * (1 + y))) ≥ 3/4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1991_199186


namespace NUMINAMATH_CALUDE_overall_average_speed_l1991_199112

theorem overall_average_speed
  (car_time : Real) (car_speed : Real) (horse_time : Real) (horse_speed : Real)
  (h1 : car_time = 45 / 60)
  (h2 : car_speed = 20)
  (h3 : horse_time = 30 / 60)
  (h4 : horse_speed = 6)
  : (car_speed * car_time + horse_speed * horse_time) / (car_time + horse_time) = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_overall_average_speed_l1991_199112


namespace NUMINAMATH_CALUDE_largest_and_smallest_numbers_l1991_199165

-- Define the numbers in their respective bases
def num1 : ℕ := 63  -- 111111₂ in decimal
def num2 : ℕ := 78  -- 210₆ in decimal
def num3 : ℕ := 64  -- 1000₄ in decimal
def num4 : ℕ := 65  -- 81₈ in decimal

-- Theorem statement
theorem largest_and_smallest_numbers :
  (num2 = max num1 (max num2 (max num3 num4))) ∧
  (num1 = min num1 (min num2 (min num3 num4))) := by
  sorry

end NUMINAMATH_CALUDE_largest_and_smallest_numbers_l1991_199165


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l1991_199113

theorem nested_sqrt_value :
  ∀ y : ℝ, y = Real.sqrt (4 + y) → y = (1 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l1991_199113


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1991_199120

def p (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 3

theorem polynomial_remainder : p 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1991_199120


namespace NUMINAMATH_CALUDE_real_part_of_z_l1991_199197

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l1991_199197


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1991_199190

theorem algebraic_expression_equality (x y : ℝ) : 
  x + 2 * y + 1 = 3 → 2 * x + 4 * y + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1991_199190


namespace NUMINAMATH_CALUDE_intersection_A_B_l1991_199175

open Set

def A : Set ℝ := {x | (x - 2) / x ≤ 0 ∧ x ≠ 0}
def B : Set ℝ := Icc (-1 : ℝ) 1

theorem intersection_A_B : A ∩ B = Ioc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1991_199175


namespace NUMINAMATH_CALUDE_sum_of_valid_starting_numbers_l1991_199156

def machine_rule (n : ℕ) : ℕ :=
  if n % 2 = 0 then n + 5 else 2 * n

def iterate_machine (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => machine_rule (iterate_machine n k)

def valid_starting_numbers : List ℕ :=
  (List.range 55).filter (λ n => iterate_machine n 4 = 54)

theorem sum_of_valid_starting_numbers :
  valid_starting_numbers.sum = 39 :=
sorry

end NUMINAMATH_CALUDE_sum_of_valid_starting_numbers_l1991_199156


namespace NUMINAMATH_CALUDE_cupcake_difference_l1991_199129

theorem cupcake_difference (morning_cupcakes afternoon_cupcakes total_cupcakes : ℕ) : 
  morning_cupcakes = 20 →
  total_cupcakes = 55 →
  afternoon_cupcakes = total_cupcakes - morning_cupcakes →
  afternoon_cupcakes - morning_cupcakes = 15 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_difference_l1991_199129


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l1991_199102

/-- An arithmetic progression with sum of first n terms S_n -/
structure ArithmeticProgression where
  S : ℕ → ℚ
  sum_formula : ∀ n : ℕ, S n = n / 2 * (2 * a + (n - 1) * d)
  a : ℚ
  d : ℚ

/-- Theorem stating that if S_3 = 2 and S_6 = 6, then S_24 = 510 for an arithmetic progression -/
theorem arithmetic_progression_sum (ap : ArithmeticProgression) 
  (h1 : ap.S 3 = 2) (h2 : ap.S 6 = 6) : ap.S 24 = 510 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l1991_199102


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1991_199161

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 1 → x^2 + 2*x > 0) ∧
  (∃ x, x^2 + 2*x > 0 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1991_199161


namespace NUMINAMATH_CALUDE_f_minimum_f_has_root_l1991_199103

noncomputable section

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (x - m) - x

-- Statement for the extremum of f(x)
theorem f_minimum (m : ℝ) : 
  (∀ x : ℝ, f m x ≥ f m m) ∧ f m m = 1 - m :=
sorry

-- Statement for the existence of a root in (m, 2m) when m > 1
theorem f_has_root (m : ℝ) (h : m > 1) : 
  ∃ x : ℝ, m < x ∧ x < 2*m ∧ f m x = 0 :=
sorry

end

end NUMINAMATH_CALUDE_f_minimum_f_has_root_l1991_199103


namespace NUMINAMATH_CALUDE_inequality_proof_l1991_199170

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / c + c / b ≥ 4 * a / (a + b) ∧
  (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1991_199170


namespace NUMINAMATH_CALUDE_negation_of_universal_nonnegative_l1991_199155

theorem negation_of_universal_nonnegative :
  (¬ ∀ x : ℝ, x ≥ 0) ↔ (∃ x : ℝ, x < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_nonnegative_l1991_199155


namespace NUMINAMATH_CALUDE_box_counting_l1991_199192

theorem box_counting (initial_boxes : ℕ) (boxes_per_operation : ℕ) (final_nonempty_boxes : ℕ) :
  initial_boxes = 2013 →
  boxes_per_operation = 13 →
  final_nonempty_boxes = 2013 →
  initial_boxes + boxes_per_operation * final_nonempty_boxes = 28182 := by
  sorry

#check box_counting

end NUMINAMATH_CALUDE_box_counting_l1991_199192


namespace NUMINAMATH_CALUDE_sphere_center_sum_l1991_199193

-- Define the points and constants
variable (a b c p q r α β γ : ℝ)

-- Define the conditions
variable (h1 : p^3 = α)
variable (h2 : q^3 = β)
variable (h3 : r^3 = γ)

-- Define the plane equation
variable (h4 : a/α + b/β + c/γ = 1)

-- Define that (p,q,r) is the center of the sphere passing through O, A, B, C
variable (h5 : p^2 + q^2 + r^2 = (p - α)^2 + q^2 + r^2)
variable (h6 : p^2 + q^2 + r^2 = p^2 + (q - β)^2 + r^2)
variable (h7 : p^2 + q^2 + r^2 = p^2 + q^2 + (r - γ)^2)

-- Theorem statement
theorem sphere_center_sum :
  a/p^3 + b/q^3 + c/r^3 = 1 :=
sorry

end NUMINAMATH_CALUDE_sphere_center_sum_l1991_199193


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l1991_199149

theorem initial_milk_water_ratio 
  (M W : ℝ) 
  (h1 : M + W = 45) 
  (h2 : M / (W + 18) = 4/3) : 
  M / W = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l1991_199149


namespace NUMINAMATH_CALUDE_parallel_vectors_m_l1991_199158

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given two vectors a and b, where a = (m, 4) and b = (3, -2),
    if a is parallel to b, then m = -6 -/
theorem parallel_vectors_m (m : ℝ) :
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  parallel a b → m = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_l1991_199158


namespace NUMINAMATH_CALUDE_four_roots_condition_l1991_199166

/-- If the equation x^2 - 4|x| + 5 = m has four distinct real roots, then 1 < m < 5 -/
theorem four_roots_condition (m : ℝ) : 
  (∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x : ℝ, x^2 - 4*|x| + 5 = m ↔ (x = a ∨ x = b ∨ x = c ∨ x = d))) →
  1 < m ∧ m < 5 := by
sorry


end NUMINAMATH_CALUDE_four_roots_condition_l1991_199166


namespace NUMINAMATH_CALUDE_least_n_modulo_121_l1991_199176

theorem least_n_modulo_121 : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → m < n → ¬(25^m + 16^m) % 121 = 1) ∧ (25^n + 16^n) % 121 = 1 :=
by
  use 32
  sorry

end NUMINAMATH_CALUDE_least_n_modulo_121_l1991_199176


namespace NUMINAMATH_CALUDE_original_number_proof_l1991_199111

theorem original_number_proof (x : ℝ) 
  (h1 : x * 74 = 19832) 
  (h2 : x / 100 * 0.74 = 1.9832) : 
  x = 268 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l1991_199111


namespace NUMINAMATH_CALUDE_chord_intersection_length_l1991_199130

/-- In a circle with radius R, chord AB of length a, diameter AC, and chord PQ perpendicular to AC
    intersecting AB at M with PM : MQ = 3 : 1, prove that AM = (4R²a) / (16R² - 3a²) -/
theorem chord_intersection_length (R a : ℝ) (h1 : R > 0) (h2 : a > 0) (h3 : a < 2*R) :
  ∃ (AM : ℝ), AM = (4 * R^2 * a) / (16 * R^2 - 3 * a^2) :=
sorry

end NUMINAMATH_CALUDE_chord_intersection_length_l1991_199130


namespace NUMINAMATH_CALUDE_min_value_of_function_l1991_199140

theorem min_value_of_function (x : ℝ) (h : x > 2) : 
  4 / (x - 2) + x ≥ 6 ∧ ∃ y > 2, 4 / (y - 2) + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1991_199140


namespace NUMINAMATH_CALUDE_f_properties_l1991_199107

noncomputable section

variables {f : ℝ → ℝ} {a : ℝ}

-- f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- f is symmetric about x = 1
def symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f (2 - x) = f x

-- f satisfies the multiplicative property for x₁, x₂ ∈ [0, 1/2]
def multiplicative_property (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ Set.Icc 0 (1/2) → x₂ ∈ Set.Icc 0 (1/2) → f (x₁ + x₂) = f x₁ * f x₂

theorem f_properties (heven : even_function f) (hsym : symmetric_about_one f)
    (hmult : multiplicative_property f) (hf1 : f 1 = a) (ha : a > 0) :
    f (1/2) = Real.sqrt a ∧ f (1/4) = Real.sqrt (Real.sqrt a) ∧ ∀ x, f (x + 2) = f x := by
  sorry

end

end NUMINAMATH_CALUDE_f_properties_l1991_199107


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1991_199121

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1991_199121


namespace NUMINAMATH_CALUDE_video_vote_ratio_l1991_199133

theorem video_vote_ratio : 
  let up_votes : ℕ := 18
  let down_votes : ℕ := 4
  let ratio : ℚ := up_votes / down_votes
  ratio = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_video_vote_ratio_l1991_199133


namespace NUMINAMATH_CALUDE_problem_statement_l1991_199101

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1991_199101


namespace NUMINAMATH_CALUDE_sum_evaluation_l1991_199164

theorem sum_evaluation : 
  4/3 + 8/9 + 16/27 + 32/81 + 64/243 + 128/729 - 8 = -1/729 := by sorry

end NUMINAMATH_CALUDE_sum_evaluation_l1991_199164


namespace NUMINAMATH_CALUDE_decimal_521_equals_octal_1011_l1991_199171

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Checks if a list of digits represents a valid octal number -/
def is_valid_octal (l : List ℕ) : Prop :=
  l.all (λ d => d < 8)

theorem decimal_521_equals_octal_1011 :
  decimal_to_octal 521 = [1, 0, 1, 1] ∧ is_valid_octal [1, 0, 1, 1] :=
by sorry

end NUMINAMATH_CALUDE_decimal_521_equals_octal_1011_l1991_199171


namespace NUMINAMATH_CALUDE_equation_solution_l1991_199128

theorem equation_solution :
  ∃ y : ℚ, (2 * y + 3 * y = 500 - (4 * y + 5 * y)) ∧ (y = 250 / 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1991_199128


namespace NUMINAMATH_CALUDE_product_prices_and_min_units_l1991_199199

/-- Represents the unit price of product A in yuan -/
def price_A : ℝ := sorry

/-- Represents the unit price of product B in yuan -/
def price_B : ℝ := sorry

/-- Represents the total number of units produced (in thousands) -/
def total_units : ℕ := 80

/-- Represents the relationship between the sales revenue of A and B -/
axiom revenue_relation : 2 * price_A = 3 * price_B

/-- Represents the difference in sales revenue between A and B -/
axiom revenue_difference : 3 * price_A - 2 * price_B = 1500

/-- Represents the minimum number of units of A to be sold (in thousands) -/
def min_units_A : ℕ := sorry

/-- Theorem stating the unit prices of A and B, and the minimum units of A to be sold -/
theorem product_prices_and_min_units : 
  price_A = 900 ∧ price_B = 600 ∧ 
  (∀ m : ℕ, m ≥ min_units_A → 
    900 * m + 600 * (total_units - m) ≥ 54000) ∧
  min_units_A = 2 := by sorry

end NUMINAMATH_CALUDE_product_prices_and_min_units_l1991_199199


namespace NUMINAMATH_CALUDE_correct_departure_time_l1991_199125

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Calculates the difference between two times in minutes -/
def timeDiffMinutes (t1 t2 : Time) : Int :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

/-- The performance start time -/
def performanceTime : Time := { hours := 8, minutes := 30 }

/-- The travel time in minutes -/
def travelTime : Nat := 20

/-- The latest departure time -/
def latestDepartureTime : Time := { hours := 8, minutes := 10 }

theorem correct_departure_time :
  timeDiffMinutes performanceTime latestDepartureTime = travelTime := by
  sorry

end NUMINAMATH_CALUDE_correct_departure_time_l1991_199125


namespace NUMINAMATH_CALUDE_inequality_range_l1991_199157

theorem inequality_range (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, x^2 + a*x > 4*x + a - 3 ↔ x < -1 ∨ x > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1991_199157


namespace NUMINAMATH_CALUDE_product_of_roots_l1991_199198

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 20 → ∃ y : ℝ, (x + 3) * (x - 4) = 20 ∧ (y + 3) * (y - 4) = 20 ∧ x * y = -32 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1991_199198


namespace NUMINAMATH_CALUDE_total_cost_theorem_l1991_199122

/-- Represents the cost of utensils in Moneda -/
structure UtensilCost where
  teaspoon : ℕ
  tablespoon : ℕ
  dessertSpoon : ℕ

/-- Represents the number of utensils Clara has -/
structure UtensilCount where
  teaspoon : ℕ
  tablespoon : ℕ
  dessertSpoon : ℕ

/-- Calculates the total cost of exchanged utensils and souvenirs in euros -/
def totalCostInEuros (costs : UtensilCost) (counts : UtensilCount) 
  (monedaToEuro : ℚ) (souvenirCostDollars : ℕ) (euroToDollar : ℚ) : ℚ :=
  sorry

/-- Theorem stating the total cost in euros -/
theorem total_cost_theorem (costs : UtensilCost) (counts : UtensilCount) 
  (monedaToEuro : ℚ) (souvenirCostDollars : ℕ) (euroToDollar : ℚ) :
  costs.teaspoon = 9 ∧ costs.tablespoon = 12 ∧ costs.dessertSpoon = 18 ∧
  counts.teaspoon = 7 ∧ counts.tablespoon = 10 ∧ counts.dessertSpoon = 12 ∧
  monedaToEuro = 0.04 ∧ souvenirCostDollars = 40 ∧ euroToDollar = 1.15 →
  totalCostInEuros costs counts monedaToEuro souvenirCostDollars euroToDollar = 50.74 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l1991_199122


namespace NUMINAMATH_CALUDE_more_silver_than_gold_fish_l1991_199148

theorem more_silver_than_gold_fish (x g s r : ℕ) : 
  x = g + s + r →
  x - g = (2 * x) / 3 - 1 →
  x - r = (2 * x) / 3 + 4 →
  s = g + 2 := by
sorry

end NUMINAMATH_CALUDE_more_silver_than_gold_fish_l1991_199148


namespace NUMINAMATH_CALUDE_root_product_expression_l1991_199168

theorem root_product_expression (p q r s : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + r = 0) → 
  (β^2 + p*β + r = 0) → 
  (γ^2 + q*γ + s = 0) → 
  (δ^2 + q*δ + s = 0) → 
  (α - γ)*(β - γ)*(α - δ)*(β - δ) = (p-q)^4 * s^2 + 2*(p-q)^3 * s * (r-s) + (p-q)^2 * (r-s)^2 := by
sorry

end NUMINAMATH_CALUDE_root_product_expression_l1991_199168


namespace NUMINAMATH_CALUDE_triangle_theorem_l1991_199195

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π

theorem triangle_theorem (t : Triangle) 
  (h : t.b * Real.cos t.A + Real.sqrt 3 * t.b * Real.sin t.A - t.c - t.a = 0) :
  t.B = π / 3 ∧ 
  (t.b = Real.sqrt 3 → ∀ (a c : ℝ), a + c ≤ 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1991_199195


namespace NUMINAMATH_CALUDE_total_vacations_and_classes_l1991_199109

/-- Represents the number of classes Kelvin has -/
def kelvin_classes : ℕ := 90

/-- Represents the cost of each of Kelvin's classes in dollars -/
def kelvin_class_cost : ℕ := 75

/-- Represents Grant's maximum budget for vacations in dollars -/
def grant_max_budget : ℕ := 100000

/-- Theorem stating that the sum of Grant's vacations and Kelvin's classes is 450 -/
theorem total_vacations_and_classes : 
  ∃ (grant_vacations : ℕ),
    grant_vacations = 4 * kelvin_classes ∧ 
    grant_vacations * (2 * kelvin_class_cost) ≤ grant_max_budget ∧
    grant_vacations + kelvin_classes = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_vacations_and_classes_l1991_199109


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1991_199152

theorem simplify_trig_expression (α : ℝ) :
  2 * Real.sin α * Real.cos α * (Real.cos α ^ 2 - Real.sin α ^ 2) = (1/2) * Real.sin (4 * α) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1991_199152


namespace NUMINAMATH_CALUDE_original_number_is_ten_l1991_199178

theorem original_number_is_ten : ∃ x : ℝ, 3 * (2 * x + 8) = 84 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_ten_l1991_199178


namespace NUMINAMATH_CALUDE_percent_problem_l1991_199116

theorem percent_problem (x : ℝ) : (0.0001 * x = 1.2356) → x = 12356 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l1991_199116


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1991_199135

-- Define the sets A and B
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 3 ∨ x = 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1991_199135


namespace NUMINAMATH_CALUDE_product_of_roots_l1991_199118

theorem product_of_roots (x : ℝ) : 
  (∃ α β : ℝ, α * β = -21 ∧ -α^2 + 4*α = -21 ∧ -β^2 + 4*β = -21) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1991_199118


namespace NUMINAMATH_CALUDE_n_is_even_l1991_199196

-- Define a type for points in space
def Point : Type := ℝ × ℝ × ℝ

-- Define a function to check if four points are coplanar
def are_coplanar (p q r s : Point) : Prop := sorry

-- Define a function to check if a point is inside a tetrahedron
def is_interior_point (p q r s t : Point) : Prop := sorry

-- Define the main theorem
theorem n_is_even (n : ℕ) (P : Fin n → Point) (Q : Point) :
  (∀ (i j k l : Fin n), i ≠ j → j ≠ k → k ≠ l → i ≠ k → j ≠ l → i ≠ l → 
    ¬ are_coplanar (P i) (P j) (P k) (P l)) →
  (∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    ∃ (l : Fin n), l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ 
      is_interior_point Q (P i) (P j) (P k) (P l)) →
  Even n := by
  sorry

end NUMINAMATH_CALUDE_n_is_even_l1991_199196


namespace NUMINAMATH_CALUDE_composite_expression_prime_case_n_one_l1991_199137

theorem composite_expression (n : ℕ) :
  n > 1 → ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 3^(2*n+1) - 2^(2*n+1) - 6^n = a * b :=
sorry

theorem prime_case_n_one :
  3^3 - 2^3 - 6 = 13 :=
sorry

end NUMINAMATH_CALUDE_composite_expression_prime_case_n_one_l1991_199137


namespace NUMINAMATH_CALUDE_point_on_graph_l1991_199183

/-- The function f(x) = -3x + 3 -/
def f (x : ℝ) : ℝ := -3 * x + 3

/-- The point p = (-2, 9) -/
def p : ℝ × ℝ := (-2, 9)

/-- Theorem: The point p lies on the graph of f -/
theorem point_on_graph : f p.1 = p.2 := by sorry

end NUMINAMATH_CALUDE_point_on_graph_l1991_199183


namespace NUMINAMATH_CALUDE_arctan_sum_of_cubic_roots_l1991_199119

theorem arctan_sum_of_cubic_roots (x₁ x₂ x₃ : ℝ) : 
  x₁^3 - 10*x₁ + 11 = 0 →
  x₂^3 - 10*x₂ + 11 = 0 →
  x₃^3 - 10*x₃ + 11 = 0 →
  -5 < x₁ ∧ x₁ < 5 →
  -5 < x₂ ∧ x₂ < 5 →
  -5 < x₃ ∧ x₃ < 5 →
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = π/4 := by
sorry

end NUMINAMATH_CALUDE_arctan_sum_of_cubic_roots_l1991_199119


namespace NUMINAMATH_CALUDE_sqrt_x_minus_6_meaningful_l1991_199184

theorem sqrt_x_minus_6_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 6) ↔ x ≥ 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_6_meaningful_l1991_199184


namespace NUMINAMATH_CALUDE_remainder_sum_divided_by_11_l1991_199145

theorem remainder_sum_divided_by_11 : 
  (2 * (8735 + 8736 + 8737 + 8738 + 8739)) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_divided_by_11_l1991_199145


namespace NUMINAMATH_CALUDE_sin_power_five_expansion_l1991_199163

theorem sin_power_five_expansion (b₁ b₂ b₃ b₄ b₅ : ℝ) : 
  (∀ θ : ℝ, Real.sin θ ^ 5 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + 
    b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ) + b₅ * Real.sin (5 * θ)) → 
  b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 + b₄ ^ 2 + b₅ ^ 2 = 63 / 128 := by
  sorry

end NUMINAMATH_CALUDE_sin_power_five_expansion_l1991_199163


namespace NUMINAMATH_CALUDE_length_of_AB_l1991_199106

-- Define the line l: kx + y - 2 = 0
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x + y - 2 = 0

-- Define the circle C: x^2 + y^2 - 6x + 2y + 9 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 2 * y + 9 = 0

-- Define the point A
def point_A (k : ℝ) : ℝ × ℝ := (0, k)

-- Define the condition that line l is the axis of symmetry for circle C
def is_axis_of_symmetry (k : ℝ) : Prop :=
  ∃ (center_x center_y : ℝ), line_l k center_x center_y ∧
    ∀ (x y : ℝ), circle_C x y ↔ circle_C (2 * center_x - x) (2 * center_y - y)

-- Define the tangency condition
def is_tangent (k : ℝ) (B : ℝ × ℝ) : Prop :=
  circle_C B.1 B.2 ∧
  ∃ (t : ℝ), B = (t, k * t + 2) ∧
    ∀ (x y : ℝ), line_l k x y → (circle_C x y → x = B.1 ∧ y = B.2)

-- State the theorem
theorem length_of_AB (k : ℝ) (B : ℝ × ℝ) :
  is_axis_of_symmetry k →
  is_tangent k B →
  Real.sqrt ((B.1 - 0)^2 + (B.2 - k)^2) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_l1991_199106


namespace NUMINAMATH_CALUDE_jovana_shells_total_l1991_199115

/-- The total amount of shells in Jovana's bucket -/
def total_shells (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that given the initial and additional amounts of shells,
    the total amount in Jovana's bucket is 17 pounds -/
theorem jovana_shells_total :
  total_shells 5 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_jovana_shells_total_l1991_199115


namespace NUMINAMATH_CALUDE_profit_calculation_l1991_199189

-- Define the buy rate
def buy_rate : ℚ := 15 / 4

-- Define the sell rate
def sell_rate : ℚ := 30 / 6

-- Define the target profit
def target_profit : ℚ := 200

-- Define the number of oranges to be sold
def oranges_to_sell : ℕ := 160

-- Theorem statement
theorem profit_calculation :
  (oranges_to_sell : ℚ) * (sell_rate - buy_rate) = target_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_calculation_l1991_199189


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1991_199100

theorem min_value_quadratic :
  ∃ (min_y : ℝ), min_y = -44 ∧ ∀ (x y : ℝ), y = x^2 + 16*x + 20 → y ≥ min_y :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1991_199100


namespace NUMINAMATH_CALUDE_marks_towers_count_l1991_199126

/-- The number of sandcastles on Mark's beach -/
def marks_castles : ℕ := 20

/-- The number of sandcastles on Jeff's beach -/
def jeffs_castles : ℕ := 3 * marks_castles

/-- The number of towers on each of Jeff's sandcastles -/
def jeffs_towers_per_castle : ℕ := 5

/-- The total number of sandcastles and towers on both beaches -/
def total_count : ℕ := 580

/-- The number of towers on each of Mark's sandcastles -/
def marks_towers_per_castle : ℕ := 10

theorem marks_towers_count : 
  marks_castles + (marks_castles * marks_towers_per_castle) + 
  jeffs_castles + (jeffs_castles * jeffs_towers_per_castle) = total_count := by
  sorry

end NUMINAMATH_CALUDE_marks_towers_count_l1991_199126


namespace NUMINAMATH_CALUDE_max_product_of_sums_l1991_199146

theorem max_product_of_sums (a b c d e f : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  a + b + c + d + e + f = 45 →
  (a + b + c) * (d + e + f) ≤ 550 := by
sorry

end NUMINAMATH_CALUDE_max_product_of_sums_l1991_199146


namespace NUMINAMATH_CALUDE_hyperbola_solution_is_three_halves_l1991_199127

/-- The set of all real numbers m that satisfy the conditions of the hyperbola problem -/
def hyperbola_solution : Set ℝ :=
  {m : ℝ | m > 0 ∧ 2 * m^2 + 3 * m = 9}

/-- The theorem stating that the solution set contains only 3/2 -/
theorem hyperbola_solution_is_three_halves : hyperbola_solution = {3/2} := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_solution_is_three_halves_l1991_199127


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1991_199117

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(- Real.sqrt b < a ∧ a < Real.sqrt b) → ¬(a^2 < b)) ↔
  ((a ≥ Real.sqrt b ∨ a ≤ - Real.sqrt b) → a^2 ≥ b) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1991_199117


namespace NUMINAMATH_CALUDE_a_gt_b_neither_necessary_nor_sufficient_for_a_sq_gt_b_sq_l1991_199144

theorem a_gt_b_neither_necessary_nor_sufficient_for_a_sq_gt_b_sq :
  ¬(∀ a b : ℝ, a > b → a^2 > b^2) ∧ 
  ¬(∀ a b : ℝ, a^2 > b^2 → a > b) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_neither_necessary_nor_sufficient_for_a_sq_gt_b_sq_l1991_199144


namespace NUMINAMATH_CALUDE_average_of_abcd_l1991_199108

theorem average_of_abcd (a b c d : ℝ) : 
  (4 + 6 + 9 + a + b + c + d) / 7 = 20 → (a + b + c + d) / 4 = 30.25 := by
  sorry

end NUMINAMATH_CALUDE_average_of_abcd_l1991_199108


namespace NUMINAMATH_CALUDE_division_problem_l1991_199132

theorem division_problem (x : ℝ) (h : x = 1) : 4 / (1 + 3/x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1991_199132


namespace NUMINAMATH_CALUDE_max_k_value_l1991_199182

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, (1/m + 2/(1 - 2*m) ≥ k) → k ≤ 8) ∧ 
  (∃ k : ℝ, k = 8 ∧ 1/m + 2/(1 - 2*m) ≥ k) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l1991_199182


namespace NUMINAMATH_CALUDE_division_of_decimals_l1991_199191

theorem division_of_decimals : (0.45 : ℚ) / (0.005 : ℚ) = 90 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l1991_199191


namespace NUMINAMATH_CALUDE_product_difference_l1991_199154

theorem product_difference (a b : ℕ+) : 
  a * b = 323 → a = 17 → b - a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_product_difference_l1991_199154


namespace NUMINAMATH_CALUDE_percentage_of_36_l1991_199169

theorem percentage_of_36 : (33 + 1 / 3 : ℚ) / 100 * 36 = 12 := by sorry

end NUMINAMATH_CALUDE_percentage_of_36_l1991_199169


namespace NUMINAMATH_CALUDE_all_triangles_in_S_are_similar_l1991_199187

-- Define a structure for triangles in set S
structure TriangleS where
  A : Real
  B : Real
  C : Real
  tan_A_pos_int : ℕ+
  tan_B_pos_int : ℕ+
  tan_C_pos_int : ℕ+
  angle_sum : A + B + C = Real.pi
  tan_sum_identity : Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C

-- Define similarity for triangles in S
def similar (t1 t2 : TriangleS) : Prop :=
  ∃ (k : Real), k > 0 ∧
    t1.A = t2.A ∧ t1.B = t2.B ∧ t1.C = t2.C

-- State the theorem
theorem all_triangles_in_S_are_similar (t1 t2 : TriangleS) :
  similar t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_all_triangles_in_S_are_similar_l1991_199187


namespace NUMINAMATH_CALUDE_xy_equals_two_l1991_199151

theorem xy_equals_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
  (h : x^2 + 2/x = y + 2/y) : x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_two_l1991_199151
