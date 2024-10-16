import Mathlib

namespace NUMINAMATH_CALUDE_matrix_determinant_equals_four_l3301_330197

def A (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![3*y, 2], ![3, y]]

theorem matrix_determinant_equals_four (y : ℝ) :
  Matrix.det (A y) = 4 ↔ y = Real.sqrt (10/3) ∨ y = -Real.sqrt (10/3) := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_equals_four_l3301_330197


namespace NUMINAMATH_CALUDE_keno_probability_value_l3301_330181

/-- The set of integers from 1 to 80 -/
def keno_numbers : Finset Nat := Finset.range 80

/-- The set of numbers from 1 to 80 that contain the digit 8 -/
def numbers_with_eight : Finset Nat := {8, 18, 28, 38, 48, 58, 68, 78}

/-- The set of numbers from 1 to 80 that do not contain the digit 8 -/
def numbers_without_eight : Finset Nat := keno_numbers \ numbers_with_eight

/-- The number of numbers to be drawn in a KENO game -/
def draw_count : Nat := 20

/-- The probability of drawing 20 numbers from 1 to 80 such that none contain the digit 8 -/
def keno_probability : ℚ := (Nat.choose numbers_without_eight.card draw_count : ℚ) / (Nat.choose keno_numbers.card draw_count)

theorem keno_probability_value : keno_probability = 27249 / 4267580 := by
  sorry

end NUMINAMATH_CALUDE_keno_probability_value_l3301_330181


namespace NUMINAMATH_CALUDE_typists_calculation_l3301_330148

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 10

/-- The number of letters typed by the initial group in 20 minutes -/
def initial_letters : ℕ := 20

/-- The time taken by the initial group to type the initial letters (in minutes) -/
def initial_time : ℕ := 20

/-- The number of typists in the second group -/
def second_typists : ℕ := 40

/-- The number of letters typed by the second group in 1 hour -/
def second_letters : ℕ := 240

/-- The time taken by the second group to type the second letters (in minutes) -/
def second_time : ℕ := 60

theorem typists_calculation :
  initial_typists * second_typists * second_time * initial_letters =
  initial_time * second_typists * second_letters * initial_typists :=
by sorry

end NUMINAMATH_CALUDE_typists_calculation_l3301_330148


namespace NUMINAMATH_CALUDE_max_daily_sales_revenue_l3301_330105

-- Define the domain of t
def T : Set ℕ := {t : ℕ | 1 ≤ t ∧ t ≤ 20}

-- Define the daily sales volume function
def f (t : ℕ) : ℝ := -t + 30

-- Define the daily sales price function
def g (t : ℕ) : ℝ :=
  if t ≤ 10 then 2 * t + 40 else 15

-- Define the daily sales revenue function
def S (t : ℕ) : ℝ := f t * g t

-- Theorem stating the maximum daily sales revenue
theorem max_daily_sales_revenue :
  ∃ (t_max : ℕ), t_max ∈ T ∧
    (∀ (t : ℕ), t ∈ T → S t ≤ S t_max) ∧
    t_max = 5 ∧ S t_max = 1250 := by
  sorry

end NUMINAMATH_CALUDE_max_daily_sales_revenue_l3301_330105


namespace NUMINAMATH_CALUDE_grocery_theorem_l3301_330190

def grocery_problem (initial_budget : ℚ) (bread_cost : ℚ) (candy_cost : ℚ) (final_remaining : ℚ) : Prop :=
  let remaining_after_bread_candy := initial_budget - (bread_cost + candy_cost)
  let spent_on_turkey := remaining_after_bread_candy - final_remaining
  spent_on_turkey / remaining_after_bread_candy = 1 / 3

theorem grocery_theorem :
  grocery_problem 32 3 2 18 := by
  sorry

end NUMINAMATH_CALUDE_grocery_theorem_l3301_330190


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3301_330155

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 2 → f x < f y :=
sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3301_330155


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3301_330198

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) (h2 : x * y = 16) : x^2 + y^2 = 356 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3301_330198


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3301_330191

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ complement_B = Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3301_330191


namespace NUMINAMATH_CALUDE_melissa_bananas_l3301_330175

/-- Calculates the remaining bananas after sharing -/
def remaining_bananas (initial : ℕ) (shared : ℕ) : ℕ :=
  initial - shared

theorem melissa_bananas : remaining_bananas 88 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_melissa_bananas_l3301_330175


namespace NUMINAMATH_CALUDE_expected_rolls_in_non_leap_year_l3301_330139

/-- Represents the outcomes of rolling an eight-sided die -/
inductive DieOutcome
| Composite
| Prime
| RollAgain

/-- The probability of each outcome when rolling the die -/
def outcomeProbability (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Composite => 2/8
  | DieOutcome.Prime => 4/8
  | DieOutcome.RollAgain => 2/8

/-- The expected number of rolls on a single day -/
def expectedRollsPerDay : ℚ :=
  4/3

/-- The number of days in a non-leap year -/
def daysInNonLeapYear : ℕ := 365

/-- The expected number of rolls in a non-leap year -/
def expectedRollsInYear : ℚ :=
  expectedRollsPerDay * daysInNonLeapYear

theorem expected_rolls_in_non_leap_year :
  expectedRollsInYear = 486 + 2/3 :=
sorry

end NUMINAMATH_CALUDE_expected_rolls_in_non_leap_year_l3301_330139


namespace NUMINAMATH_CALUDE_integral_equation_solution_l3301_330141

theorem integral_equation_solution (k : ℝ) : 
  (∫ (x : ℝ), 2*x - 3*x^2) = 0 → k = 0 ∨ k = 1 :=
by sorry

end NUMINAMATH_CALUDE_integral_equation_solution_l3301_330141


namespace NUMINAMATH_CALUDE_monochromatic_equilateral_triangle_l3301_330122

-- Define a type for colors
inductive Color
| White
| Black

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point
  eq_sides : distance a b = distance b c ∧ distance b c = distance c a

-- Theorem statement
theorem monochromatic_equilateral_triangle :
  ∃ (t : EquilateralTriangle),
    (distance t.a t.b = 1 ∨ distance t.a t.b = Real.sqrt 3) ∧
    (coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c) := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_equilateral_triangle_l3301_330122


namespace NUMINAMATH_CALUDE_island_age_conversion_l3301_330152

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- The age of the island in base 7 and base 10 --/
theorem island_age_conversion :
  base7ToBase10 3 4 6 = 181 := by
  sorry

end NUMINAMATH_CALUDE_island_age_conversion_l3301_330152


namespace NUMINAMATH_CALUDE_identical_pairs_x_value_l3301_330104

def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := λ (a, b) (c, d) ↦ (a - c, b + d)

theorem identical_pairs_x_value :
  ∀ x y : ℤ, star (2, 2) (4, 1) = star (x, y) (1, 4) → x = -1 :=
by sorry

end NUMINAMATH_CALUDE_identical_pairs_x_value_l3301_330104


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3301_330193

theorem arithmetic_calculations : 
  (-6 * (-2) + (-5) * 16 = -68) ∧ 
  ((-1)^4 + (1/4) * (2 * (-6) - (-4)^2) = -8) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3301_330193


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3301_330176

theorem sin_2alpha_value (α : Real) 
  (h1 : α > -π/2 ∧ α < 0) 
  (h2 : Real.tan (π/4 - α) = 3 * Real.cos (2 * α)) : 
  Real.sin (2 * α) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3301_330176


namespace NUMINAMATH_CALUDE_sarah_finished_problems_l3301_330114

/-- Calculates the number of problems Sarah finished given the initial number of problems,
    remaining pages, and problems per page. -/
def problems_finished (initial_problems : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  initial_problems - (remaining_pages * problems_per_page)

/-- Proves that Sarah finished 20 problems given the initial conditions. -/
theorem sarah_finished_problems :
  problems_finished 60 5 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sarah_finished_problems_l3301_330114


namespace NUMINAMATH_CALUDE_solution_pairs_l3301_330188

theorem solution_pairs (x y : ℝ) (hxy : x ≠ y) 
  (eq1 : x^100 - y^100 = 2^99 * (x - y))
  (eq2 : x^200 - y^200 = 2^199 * (x - y)) :
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_solution_pairs_l3301_330188


namespace NUMINAMATH_CALUDE_parallel_line_slope_l3301_330192

/-- Given a line with equation 3x + 6y = -12, this theorem states that
    the slope of any line parallel to it is -1/2. -/
theorem parallel_line_slope (x y : ℝ) :
  (3 : ℝ) * x + 6 * y = -12 →
  ∃ (m b : ℝ), y = m * x + b ∧ m = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l3301_330192


namespace NUMINAMATH_CALUDE_same_price_at_12_sheets_unique_equal_price_at_12_sheets_l3301_330112

/-- Represents the pricing structure of a photo company -/
structure PhotoCompany where
  perSheetCost : ℚ
  sittingFee : ℚ

/-- Calculates the total cost for a given number of sheets -/
def totalCost (company : PhotoCompany) (sheets : ℚ) : ℚ :=
  company.perSheetCost * sheets + company.sittingFee

/-- John's Photo World pricing -/
def johnsPhotoWorld : PhotoCompany :=
  { perSheetCost := 2.75, sittingFee := 125 }

/-- Sam's Picture Emporium pricing -/
def samsPictureEmporium : PhotoCompany :=
  { perSheetCost := 1.50, sittingFee := 140 }

/-- Theorem stating that the companies charge the same for 12 sheets -/
theorem same_price_at_12_sheets :
  totalCost johnsPhotoWorld 12 = totalCost samsPictureEmporium 12 := by
  sorry

/-- Theorem stating that 12 is the unique number of sheets where prices are equal -/
theorem unique_equal_price_at_12_sheets :
  ∀ x : ℚ, totalCost johnsPhotoWorld x = totalCost samsPictureEmporium x ↔ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_same_price_at_12_sheets_unique_equal_price_at_12_sheets_l3301_330112


namespace NUMINAMATH_CALUDE_finite_nonempty_set_is_good_l3301_330164

/-- An expression using real numbers, ±, +, ×, and parentheses -/
inductive Expression : Type
| Const : ℝ → Expression
| PlusMinus : Expression → Expression → Expression
| Plus : Expression → Expression → Expression
| Times : Expression → Expression → Expression

/-- The range of an expression -/
def range (e : Expression) : Set ℝ :=
  sorry

/-- A set is good if it's the range of some expression -/
def is_good (S : Set ℝ) : Prop :=
  ∃ e : Expression, range e = S

theorem finite_nonempty_set_is_good (S : Set ℝ) (h₁ : S.Finite) (h₂ : S.Nonempty) :
  is_good S :=
sorry

end NUMINAMATH_CALUDE_finite_nonempty_set_is_good_l3301_330164


namespace NUMINAMATH_CALUDE_simplify_expression_l3301_330142

theorem simplify_expression : ((5 * 10^7) / (2 * 10^2)) + (4 * 10^5) = 650000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3301_330142


namespace NUMINAMATH_CALUDE_constant_c_value_l3301_330140

theorem constant_c_value : ∃ c : ℚ, 
  (∀ x : ℚ, (3 * x^3 - 5 * x^2 + 6 * x - 4) * (2 * x^2 + c * x + 8) = 
   6 * x^5 - 19 * x^4 + 40 * x^3 + c * x^2 - 32 * x + 32) ∧ 
  c = 48 / 5 := by
sorry

end NUMINAMATH_CALUDE_constant_c_value_l3301_330140


namespace NUMINAMATH_CALUDE_box_volume_l3301_330131

theorem box_volume (side1 side2 upper : ℝ) 
  (h1 : side1 = 120)
  (h2 : side2 = 72)
  (h3 : upper = 60) :
  ∃ (l w h : ℝ), 
    l * w = side1 ∧ 
    w * h = side2 ∧ 
    l * h = upper ∧ 
    l * w * h = 720 :=
sorry

end NUMINAMATH_CALUDE_box_volume_l3301_330131


namespace NUMINAMATH_CALUDE_salary_calculation_l3301_330126

def monthly_salary : ℝ → Prop := λ s => 
  let original_savings := 0.2 * s
  let original_expenses := 0.8 * s
  let increased_expenses := 1.2 * original_expenses
  s - increased_expenses = 250

theorem salary_calculation : ∃ s : ℝ, monthly_salary s ∧ s = 6250 := by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l3301_330126


namespace NUMINAMATH_CALUDE_lcm_gcd_275_570_l3301_330118

theorem lcm_gcd_275_570 : 
  (Nat.lcm 275 570 = 31350) ∧ (Nat.gcd 275 570 = 5) := by sorry

end NUMINAMATH_CALUDE_lcm_gcd_275_570_l3301_330118


namespace NUMINAMATH_CALUDE_problem_statement_l3301_330187

theorem problem_statement (x : ℝ) (h : x - 1/x = Real.sqrt 3) :
  x^243 - 1/x^243 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3301_330187


namespace NUMINAMATH_CALUDE_star_replacement_impossibility_l3301_330143

theorem star_replacement_impossibility : ∀ (f : Fin 9 → Bool),
  ∃ (result : ℤ), result ≠ 0 ∧
  result = (if f 0 then 1 else -1) +
           (if f 1 then 2 else -2) +
           (if f 2 then 3 else -3) +
           (if f 3 then 4 else -4) +
           (if f 4 then 5 else -5) +
           (if f 5 then 6 else -6) +
           (if f 6 then 7 else -7) +
           (if f 7 then 8 else -8) +
           (if f 8 then 9 else -9) +
           10 :=
by
  sorry

end NUMINAMATH_CALUDE_star_replacement_impossibility_l3301_330143


namespace NUMINAMATH_CALUDE_tangent_plane_equation_l3301_330156

-- Define the function f(x, y)
def f (x y : ℝ) : ℝ := x^2 + y^2 + 2*x + 1

-- Define the point A
def A : ℝ × ℝ := (2, 3)

-- Theorem statement
theorem tangent_plane_equation :
  let (x₀, y₀) := A
  let z₀ := f x₀ y₀
  let fx := (2 * x₀ + 2 : ℝ)  -- Partial derivative with respect to x
  let fy := (2 * y₀ : ℝ)      -- Partial derivative with respect to y
  ∀ x y z, z - z₀ = fx * (x - x₀) + fy * (y - y₀) ↔ 6*x + 6*y - z - 12 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_plane_equation_l3301_330156


namespace NUMINAMATH_CALUDE_series_sum_equals_three_halves_l3301_330100

/-- The sum of the infinite series ∑(n=1 to ∞) (4n-3)/(3^n) is equal to 3/2 -/
theorem series_sum_equals_three_halves :
  ∑' n, (4 * n - 3 : ℝ) / (3 : ℝ) ^ n = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_three_halves_l3301_330100


namespace NUMINAMATH_CALUDE_rabbit_storage_l3301_330184

/-- Represents the number of items stored per hole for each animal -/
structure StorageRate where
  rabbit : ℕ
  deer : ℕ
  fox : ℕ

/-- Represents the number of holes dug by each animal -/
structure Holes where
  rabbit : ℕ
  deer : ℕ
  fox : ℕ

/-- The main theorem stating that given the conditions, the rabbit stored 60 items -/
theorem rabbit_storage (rate : StorageRate) (holes : Holes) : 
  rate.rabbit = 4 →
  rate.deer = 5 →
  rate.fox = 7 →
  rate.rabbit * holes.rabbit = rate.deer * holes.deer →
  rate.rabbit * holes.rabbit = rate.fox * holes.fox →
  holes.deer = holes.rabbit - 3 →
  holes.fox = holes.deer + 2 →
  rate.rabbit * holes.rabbit = 60 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_storage_l3301_330184


namespace NUMINAMATH_CALUDE_quadratic_positivity_quadratic_positivity_range_l3301_330150

/-- Given a quadratic function f(x) = x^2 + 2x + a, if f(x) > 0 for all x ≥ 1,
    then a > -3. -/
theorem quadratic_positivity (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → x^2 + 2*x + a > 0) → a > -3 := by
  sorry

/-- The range of a for which f(x) = x^2 + 2x + a is positive for all x ≥ 1
    is the open interval (-3, +∞). -/
theorem quadratic_positivity_range :
  {a : ℝ | ∀ x : ℝ, x ≥ 1 → x^2 + 2*x + a > 0} = Set.Ioi (-3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_positivity_quadratic_positivity_range_l3301_330150


namespace NUMINAMATH_CALUDE_a_4_equals_4_l3301_330130

def sequence_term (n : ℕ) : ℤ := (-1)^n * n

theorem a_4_equals_4 : sequence_term 4 = 4 := by sorry

end NUMINAMATH_CALUDE_a_4_equals_4_l3301_330130


namespace NUMINAMATH_CALUDE_percentage_increase_l3301_330127

theorem percentage_increase (x : ℝ) (h : x = 123.2) : 
  (x - 88) / 88 * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3301_330127


namespace NUMINAMATH_CALUDE_inequality_solution_l3301_330157

theorem inequality_solution (x : ℤ) : 
  Real.sqrt (3 * x - 7) - Real.sqrt (3 * x^2 - 13 * x + 13) ≥ 3 * x^2 - 16 * x + 20 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3301_330157


namespace NUMINAMATH_CALUDE_johns_number_l3301_330146

theorem johns_number : ∃! n : ℕ, 
  200 ∣ n ∧ 
  18 ∣ n ∧ 
  1000 < n ∧ 
  n < 2500 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_johns_number_l3301_330146


namespace NUMINAMATH_CALUDE_james_water_storage_l3301_330160

def cask_capacity : ℕ := 20

def barrel_capacity (cask_cap : ℕ) : ℕ := 2 * cask_cap + 3

def total_storage (cask_cap barrel_cap num_barrels : ℕ) : ℕ :=
  cask_cap + num_barrels * barrel_cap

theorem james_water_storage :
  total_storage cask_capacity (barrel_capacity cask_capacity) 4 = 192 := by
  sorry

end NUMINAMATH_CALUDE_james_water_storage_l3301_330160


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l3301_330182

/-- Given a class of students and their dessert preferences, calculate the number of students who like both desserts. -/
theorem students_liking_both_desserts
  (total : ℕ)
  (like_apple : ℕ)
  (like_chocolate : ℕ)
  (like_neither : ℕ)
  (h1 : total = 35)
  (h2 : like_apple = 20)
  (h3 : like_chocolate = 17)
  (h4 : like_neither = 8) :
  like_apple + like_chocolate - (total - like_neither) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l3301_330182


namespace NUMINAMATH_CALUDE_product_of_numbers_l3301_330136

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 16) (h2 : x^2 + y^2 = 200) : x * y = 28 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3301_330136


namespace NUMINAMATH_CALUDE_ratio_problem_l3301_330119

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 2) (h2 : c/b = 3) : (a+b)/(b+c) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3301_330119


namespace NUMINAMATH_CALUDE_readers_intersection_l3301_330170

theorem readers_intersection (total : ℕ) (sci_fi : ℕ) (literary : ℕ) 
  (h1 : total = 250) (h2 : sci_fi = 180) (h3 : literary = 88) :
  sci_fi + literary - total = 18 := by
  sorry

end NUMINAMATH_CALUDE_readers_intersection_l3301_330170


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l3301_330128

theorem mod_equivalence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 500000 [ZMOD 9] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l3301_330128


namespace NUMINAMATH_CALUDE_g_zero_at_negative_one_l3301_330116

def g (x s : ℝ) : ℝ := 3 * x^4 + x^3 - 2 * x^2 - 4 * x + s

theorem g_zero_at_negative_one (s : ℝ) : g (-1) s = 0 ↔ s = -4 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_at_negative_one_l3301_330116


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3301_330132

/-- A polynomial p(x) = x^2 + mx + 9 is a perfect square trinomial if and only if
    there exists a real number a such that p(x) = (x + a)^2 for all x. -/
def IsPerfectSquareTrinomial (m : ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x^2 + m*x + 9 = (x + a)^2

/-- If x^2 + mx + 9 is a perfect square trinomial, then m = 6 or m = -6. -/
theorem perfect_square_trinomial_condition (m : ℝ) :
  IsPerfectSquareTrinomial m → m = 6 ∨ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3301_330132


namespace NUMINAMATH_CALUDE_salt_production_theorem_l3301_330169

/-- Calculates the average daily salt production for a year given the initial production and monthly increase. -/
def averageDailyProduction (initialProduction : ℕ) (monthlyIncrease : ℕ) : ℚ :=
  let totalProduction := initialProduction + (initialProduction + monthlyIncrease + initialProduction + monthlyIncrease * 11) * 11 / 2
  totalProduction / 365

/-- Theorem stating that the average daily production is approximately 83.84 tonnes. -/
theorem salt_production_theorem (initialProduction monthlyIncrease : ℕ) 
  (h1 : initialProduction = 2000)
  (h2 : monthlyIncrease = 100) :
  ∃ ε > 0, |averageDailyProduction initialProduction monthlyIncrease - 83.84| < ε :=
sorry

end NUMINAMATH_CALUDE_salt_production_theorem_l3301_330169


namespace NUMINAMATH_CALUDE_intersection_M_N_l3301_330180

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3301_330180


namespace NUMINAMATH_CALUDE_real_part_of_z_l3301_330102

theorem real_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.abs (3 - 4 * Complex.I)) : 
  z.re = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3301_330102


namespace NUMINAMATH_CALUDE_piano_lesson_cost_l3301_330161

/-- Calculate the total cost of piano lessons -/
theorem piano_lesson_cost (lesson_cost : ℝ) (lesson_duration : ℝ) (total_hours : ℝ) : 
  lesson_cost = 30 ∧ lesson_duration = 1.5 ∧ total_hours = 18 →
  (total_hours / lesson_duration) * lesson_cost = 360 := by
  sorry

end NUMINAMATH_CALUDE_piano_lesson_cost_l3301_330161


namespace NUMINAMATH_CALUDE_complex_power_four_l3301_330154

theorem complex_power_four (i : ℂ) : i^2 = -1 → (1 - i)^4 = -4 := by sorry

end NUMINAMATH_CALUDE_complex_power_four_l3301_330154


namespace NUMINAMATH_CALUDE_chocolate_bars_distribution_l3301_330162

theorem chocolate_bars_distribution (total_bars : ℕ) (num_people : ℕ) (num_combined : ℕ) :
  total_bars = 12 →
  num_people = 3 →
  num_combined = 2 →
  (total_bars / num_people) * num_combined = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_distribution_l3301_330162


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l3301_330151

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 11

-- Define the point of tangency
def P : ℝ × ℝ := (1, 12)

-- Theorem statement
theorem tangent_line_y_intercept :
  let slope := (3 : ℝ) -- Derivative of f at x = 1
  let tangent_line (x : ℝ) := slope * (x - P.1) + P.2
  (tangent_line 0) = 9 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l3301_330151


namespace NUMINAMATH_CALUDE_quadrilateral_is_parallelogram_l3301_330109

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

-- Define an acute angle
structure AcuteAngle where
  vertex : Point2D
  side1 : Line2D
  side2 : Line2D

-- Function to calculate the distance from a point to a line
def distancePointToLine (p : Point2D) (l : Line2D) : ℝ :=
  sorry

-- Function to check if a quadrilateral is convex
def isConvex (q : Quadrilateral) : Prop :=
  sorry

-- Function to check if a quadrilateral is within an angle
def isWithinAngle (q : Quadrilateral) (angle : AcuteAngle) : Prop :=
  sorry

-- Function to check if a quadrilateral is a parallelogram
def isParallelogram (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem quadrilateral_is_parallelogram
  (q : Quadrilateral)
  (angle : AcuteAngle)
  (h_convex : isConvex q)
  (h_within : isWithinAngle q angle)
  (h_distance1 : distancePointToLine q.A angle.side1 + distancePointToLine q.C angle.side1 =
                 distancePointToLine q.B angle.side1 + distancePointToLine q.D angle.side1)
  (h_distance2 : distancePointToLine q.A angle.side2 + distancePointToLine q.C angle.side2 =
                 distancePointToLine q.B angle.side2 + distancePointToLine q.D angle.side2) :
  isParallelogram q :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_is_parallelogram_l3301_330109


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3301_330186

/-- The number of games in a chess tournament where each player plays twice with every other player. -/
def num_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 12 players, where each player plays twice with every other player, the total number of games played is 264. -/
theorem chess_tournament_games :
  num_games 12 * 2 = 264 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3301_330186


namespace NUMINAMATH_CALUDE_multiply_to_all_ones_l3301_330123

theorem multiply_to_all_ones : 
  ∃ (A : ℕ) (n : ℕ), (10^9 - 1) * A = (10^n - 1) / 9 :=
sorry

end NUMINAMATH_CALUDE_multiply_to_all_ones_l3301_330123


namespace NUMINAMATH_CALUDE_square_sum_inequality_l3301_330135

theorem square_sum_inequality (a b u : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hu : 0 < u) (hab : a + b = 1) : 
  (∀ a b, a^2 + b^2 ≥ u) ↔ u ≤ 1/2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l3301_330135


namespace NUMINAMATH_CALUDE_p_true_q_false_l3301_330137

-- Define the quadratic equation
def hasRealRoots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + x - m = 0

-- Proposition p
theorem p_true : ∀ m : ℝ, m > 0 → hasRealRoots m :=
sorry

-- Converse of p (proposition q) is false
theorem q_false : ∃ m : ℝ, m ≥ -1 ∧ m ≤ 0 ∧ hasRealRoots m :=
sorry

end NUMINAMATH_CALUDE_p_true_q_false_l3301_330137


namespace NUMINAMATH_CALUDE_jumping_contest_l3301_330124

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump squirrel_jump : ℕ)
  (grasshopper_obstacle frog_obstacle mouse_obstacle squirrel_obstacle : ℕ)
  (h1 : grasshopper_jump = 19)
  (h2 : grasshopper_obstacle = 3)
  (h3 : frog_jump = grasshopper_jump + 10)
  (h4 : frog_obstacle = 0)
  (h5 : mouse_jump = frog_jump + 20)
  (h6 : mouse_obstacle = 5)
  (h7 : squirrel_jump = mouse_jump - 7)
  (h8 : squirrel_obstacle = 2) :
  (mouse_jump - mouse_obstacle) - (grasshopper_jump - grasshopper_obstacle) = 28 := by
  sorry

#check jumping_contest

end NUMINAMATH_CALUDE_jumping_contest_l3301_330124


namespace NUMINAMATH_CALUDE_milk_fraction_in_cup1_l3301_330106

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  milk : ℚ

/-- Represents the state of both cups --/
structure TwoCups where
  cup1 : CupContents
  cup2 : CupContents

def initial_state : TwoCups :=
  { cup1 := { coffee := 3, milk := 0 },
    cup2 := { coffee := 0, milk := 7 } }

def transfer_coffee (state : TwoCups) : TwoCups :=
  { cup1 := { coffee := state.cup1.coffee * 2/3, milk := state.cup1.milk },
    cup2 := { coffee := state.cup2.coffee + state.cup1.coffee * 1/3, milk := state.cup2.milk } }

def transfer_mixture (state : TwoCups) : TwoCups :=
  let total_cup2 := state.cup2.coffee + state.cup2.milk
  let transfer_amount := total_cup2 * 1/4
  let coffee_ratio := state.cup2.coffee / total_cup2
  let milk_ratio := state.cup2.milk / total_cup2
  { cup1 := { coffee := state.cup1.coffee + transfer_amount * coffee_ratio,
              milk := state.cup1.milk + transfer_amount * milk_ratio },
    cup2 := { coffee := state.cup2.coffee - transfer_amount * coffee_ratio,
              milk := state.cup2.milk - transfer_amount * milk_ratio } }

def final_state : TwoCups :=
  transfer_mixture (transfer_coffee initial_state)

theorem milk_fraction_in_cup1 :
  let total_liquid := final_state.cup1.coffee + final_state.cup1.milk
  final_state.cup1.milk / total_liquid = 7/16 := by sorry

end NUMINAMATH_CALUDE_milk_fraction_in_cup1_l3301_330106


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3301_330145

theorem largest_prime_factor_of_expression :
  (Nat.factors (18^3 + 15^4 - 10^5)).maximum? = some 98359 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3301_330145


namespace NUMINAMATH_CALUDE_sandwich_cost_is_four_l3301_330168

/-- The cost of Karen's fast food order --/
def fast_food_order (burger_cost smoothie_cost sandwich_cost : ℚ) : Prop :=
  burger_cost = 5 ∧
  smoothie_cost = 4 ∧
  burger_cost + 2 * smoothie_cost + sandwich_cost = 17

theorem sandwich_cost_is_four :
  ∀ (burger_cost smoothie_cost sandwich_cost : ℚ),
    fast_food_order burger_cost smoothie_cost sandwich_cost →
    sandwich_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_is_four_l3301_330168


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3301_330172

theorem absolute_value_inequality (x : ℝ) : 
  |((5 - x) / 3)| < 2 ↔ -1 < x ∧ x < 11 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3301_330172


namespace NUMINAMATH_CALUDE_regular_150_sided_polygon_diagonals_l3301_330195

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 150 sides has 11025 diagonals -/
theorem regular_150_sided_polygon_diagonals :
  num_diagonals 150 = 11025 := by sorry

end NUMINAMATH_CALUDE_regular_150_sided_polygon_diagonals_l3301_330195


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3301_330165

theorem negation_of_existence_proposition :
  (¬∃ (c : ℝ), c > 0 ∧ ∃ (x : ℝ), x^2 - x + c = 0) ↔
  (∀ (c : ℝ), c > 0 → ∀ (x : ℝ), x^2 - x + c ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3301_330165


namespace NUMINAMATH_CALUDE_inequality_property_l3301_330199

theorem inequality_property (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l3301_330199


namespace NUMINAMATH_CALUDE_log_equation_implies_y_equals_nine_l3301_330173

theorem log_equation_implies_y_equals_nine 
  (x y : ℝ) 
  (h : x > 0) 
  (h2x : 2*x > 0) 
  (hy : y > 0) : 
  (Real.log x / Real.log 3) * (Real.log (2*x) / Real.log x) * (Real.log y / Real.log (2*x)) = 2 → 
  y = 9 := by
sorry

end NUMINAMATH_CALUDE_log_equation_implies_y_equals_nine_l3301_330173


namespace NUMINAMATH_CALUDE_fraction_less_than_one_l3301_330183

theorem fraction_less_than_one (a b : ℝ) (h1 : a < b) (h2 : b < 0) : b / a < 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_one_l3301_330183


namespace NUMINAMATH_CALUDE_angle_bisector_ratio_bound_angle_bisector_ratio_bound_tight_l3301_330120

/-- A triangle with sides a and b, and corresponding angle bisectors t_a and t_b -/
structure Triangle where
  a : ℝ
  b : ℝ
  t_a : ℝ
  t_b : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < t_a ∧ 0 < t_b
  h_triangle : a < b + t_b ∧ b < a + t_a ∧ t_b < a + b
  h_bisector_a : t_a < (2 * b * (a + b)) / (a + 2 * b)
  h_bisector_b : t_b < (2 * a * (a + b)) / (2 * a + b)

/-- The upper bound for the ratio of sum of angle bisectors to sum of sides is 4/3 -/
theorem angle_bisector_ratio_bound (T : Triangle) :
  (T.t_a + T.t_b) / (T.a + T.b) < 4/3 :=
sorry

/-- The upper bound 4/3 is the least possible -/
theorem angle_bisector_ratio_bound_tight :
  ∀ ε > 0, ∃ T : Triangle, (4/3 - ε) < (T.t_a + T.t_b) / (T.a + T.b) :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_ratio_bound_angle_bisector_ratio_bound_tight_l3301_330120


namespace NUMINAMATH_CALUDE_raul_shopping_spree_l3301_330189

def initial_amount : ℚ := 87
def comic_price : ℚ := 4
def comic_quantity : ℕ := 8
def novel_price : ℚ := 7
def novel_quantity : ℕ := 3
def magazine_price : ℚ := 5.5
def magazine_quantity : ℕ := 2

def total_spent : ℚ :=
  comic_price * comic_quantity +
  novel_price * novel_quantity +
  magazine_price * magazine_quantity

def remaining_amount : ℚ := initial_amount - total_spent

theorem raul_shopping_spree :
  remaining_amount = 23 := by sorry

end NUMINAMATH_CALUDE_raul_shopping_spree_l3301_330189


namespace NUMINAMATH_CALUDE_ai_chip_pass_rate_below_threshold_l3301_330185

-- Define the probabilities for intelligent testing indicators
def p_safety : ℚ := 49/50
def p_energy : ℚ := 48/49
def p_performance : ℚ := 47/48

-- Define the probability of passing manual testing
def p_manual : ℚ := 49/50

-- Define the number of chips selected for manual testing
def n_chips : ℕ := 50

-- Theorem statement
theorem ai_chip_pass_rate_below_threshold :
  let p_intelligent := p_safety * p_energy * p_performance
  let p_overall := p_intelligent * p_manual
  p_overall < 93/100 := by
  sorry

end NUMINAMATH_CALUDE_ai_chip_pass_rate_below_threshold_l3301_330185


namespace NUMINAMATH_CALUDE_books_on_cart_l3301_330111

def top_section : ℕ := 12 + 8 + 4

def bottom_section_non_mystery : ℕ := 5 + 6

def bottom_section : ℕ := 2 * bottom_section_non_mystery

def total_books : ℕ := top_section + bottom_section

theorem books_on_cart : total_books = 46 := by
  sorry

end NUMINAMATH_CALUDE_books_on_cart_l3301_330111


namespace NUMINAMATH_CALUDE_solution_exists_l3301_330133

theorem solution_exists : ∃ (x y z : ℝ), 
  (15 + (1/4) * x = 27) ∧ 
  ((1/2) * x - y^2 = 37) ∧ 
  (y^3 + z = 50) ∧ 
  (x = 48) ∧ 
  ((y = Real.sqrt 13 ∧ z = 50 - 13 * Real.sqrt 13) ∨ 
   (y = -Real.sqrt 13 ∧ z = 50 + 13 * Real.sqrt 13)) := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l3301_330133


namespace NUMINAMATH_CALUDE_math_competition_score_l3301_330149

theorem math_competition_score 
  (a₁ a₂ a₃ a₄ a₅ : ℕ) 
  (h_distinct : a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅)
  (h_first_two : a₁ + a₂ = 10)
  (h_last_two : a₄ + a₅ = 18) :
  a₁ + a₂ + a₃ + a₄ + a₅ = 35 :=
by sorry

end NUMINAMATH_CALUDE_math_competition_score_l3301_330149


namespace NUMINAMATH_CALUDE_divisible_by_ten_l3301_330107

theorem divisible_by_ten (S : Finset ℤ) : 
  (Finset.card S = 5) →
  (∀ (a b c : ℤ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → (10 ∣ a * b * c)) →
  (∃ x ∈ S, 10 ∣ x) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_ten_l3301_330107


namespace NUMINAMATH_CALUDE_simplify_power_expression_l3301_330196

theorem simplify_power_expression (x y : ℝ) : (3 * x^2 * y)^4 = 81 * x^8 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l3301_330196


namespace NUMINAMATH_CALUDE_folding_coincidence_implies_rhombus_l3301_330115

/-- A quadrilateral on a piece of paper. -/
structure PaperQuadrilateral where
  /-- The four vertices of the quadrilateral -/
  vertices : Fin 4 → ℝ × ℝ

/-- Represents the result of folding a paper quadrilateral along a diagonal -/
def foldAlongDiagonal (q : PaperQuadrilateral) (d : Fin 2) : Prop :=
  -- This is a placeholder for the actual folding operation
  sorry

/-- A quadrilateral is a rhombus if it satisfies certain properties -/
def isRhombus (q : PaperQuadrilateral) : Prop :=
  -- This is a placeholder for the actual definition of a rhombus
  sorry

/-- 
If folding a quadrilateral along both diagonals results in coinciding parts each time, 
then the quadrilateral is a rhombus.
-/
theorem folding_coincidence_implies_rhombus (q : PaperQuadrilateral) :
  (∀ d : Fin 2, foldAlongDiagonal q d) → isRhombus q :=
by
  sorry

end NUMINAMATH_CALUDE_folding_coincidence_implies_rhombus_l3301_330115


namespace NUMINAMATH_CALUDE_symmetry_wrt_y_axis_l3301_330171

/-- Given a point P with coordinates (x, y), the point symmetrical to P
    with respect to the y-axis has coordinates (-x, y) -/
def symmetrical_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (-(P.1), P.2)

theorem symmetry_wrt_y_axis :
  let P : ℝ × ℝ := (3, -5)
  symmetrical_point P = (-3, -5) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_wrt_y_axis_l3301_330171


namespace NUMINAMATH_CALUDE_unique_prime_power_equation_l3301_330194

theorem unique_prime_power_equation :
  ∃! (p n : ℕ), Prime p ∧ n > 0 ∧ (1 + p)^n = 1 + p*n + n^p := by sorry

end NUMINAMATH_CALUDE_unique_prime_power_equation_l3301_330194


namespace NUMINAMATH_CALUDE_simplify_expression_calculate_expression_calculate_profit_l3301_330166

-- Problem 1
theorem simplify_expression (a b : ℝ) :
  -3 * (a + b)^2 - 6 * (a + b)^2 + 8 * (a + b)^2 = -(a + b)^2 := by sorry

-- Problem 2
theorem calculate_expression (a b c d : ℝ) 
  (h1 : a - 2*b = 5) 
  (h2 : 2*b - c = -7) 
  (h3 : c - d = 12) :
  4*(a - c) + 4*(2*b - d) - 4*(2*b - c) = 40 := by sorry

-- Problem 3
def standard_price : ℝ := 56
def initial_cost : ℝ := 400
def sales_records : List ℝ := [-3, 7, -8, 9, -2, 0, -1, -6]

theorem calculate_profit :
  (List.sum sales_records + standard_price * sales_records.length) - initial_cost = 44 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_calculate_expression_calculate_profit_l3301_330166


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l3301_330178

/-- Calculates the gain percent given the number of chocolates at cost price and selling price that are equal in value. -/
def gain_percent (cost_count : ℕ) (sell_count : ℕ) : ℚ :=
  ((cost_count - sell_count) / sell_count) * 100

/-- Theorem stating that when the cost price of 65 chocolates equals the selling price of 50 chocolates, the gain percent is 30%. -/
theorem chocolate_gain_percent :
  gain_percent 65 50 = 30 := by
  sorry

#eval gain_percent 65 50

end NUMINAMATH_CALUDE_chocolate_gain_percent_l3301_330178


namespace NUMINAMATH_CALUDE_sin_sum_less_than_sum_of_sins_l3301_330177

theorem sin_sum_less_than_sum_of_sins (x y : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  y ∈ Set.Icc 0 (Real.pi / 2) →
  Real.sin (x + y) < Real.sin x + Real.sin y :=
by sorry

end NUMINAMATH_CALUDE_sin_sum_less_than_sum_of_sins_l3301_330177


namespace NUMINAMATH_CALUDE_even_painted_faces_count_l3301_330144

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with an even number of painted faces -/
def countEvenPaintedFaces (b : Block) : ℕ :=
  -- We don't implement the actual counting logic here
  sorry

/-- Theorem stating that a 3x4x2 block has 12 cubes with even number of painted faces -/
theorem even_painted_faces_count (b : Block) 
  (h1 : b.length = 3) 
  (h2 : b.width = 4) 
  (h3 : b.height = 2) : 
  countEvenPaintedFaces b = 12 := by
  sorry

end NUMINAMATH_CALUDE_even_painted_faces_count_l3301_330144


namespace NUMINAMATH_CALUDE_min_value_a_l3301_330129

theorem min_value_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x : ℝ, x ≥ 1 → a^x ≥ a*x) :
  ∀ b : ℝ, (b > 0 ∧ b ≠ 1 ∧ (∀ x : ℝ, x ≥ 1 → b^x ≥ b*x)) → a ≤ b → a = Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l3301_330129


namespace NUMINAMATH_CALUDE_product_mod_600_l3301_330110

theorem product_mod_600 : (1497 * 2003) % 600 = 291 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_600_l3301_330110


namespace NUMINAMATH_CALUDE_coordinates_unique_location_l3301_330134

-- Define the types of location descriptions
inductive LocationDescription
  | CinemaRow (row : ℕ)
  | StreetName (street : String) (city : String)
  | Direction (angle : ℝ) (direction : String)
  | Coordinates (longitude : ℝ) (latitude : ℝ)

-- Define a function to check if a location description is unique
def isUniqueLocation (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.CinemaRow _ => False
  | LocationDescription.StreetName _ _ => False
  | LocationDescription.Direction _ _ => False
  | LocationDescription.Coordinates _ _ => True

-- Theorem statement
theorem coordinates_unique_location :
  ∀ (desc : LocationDescription),
    isUniqueLocation desc ↔ ∃ (long lat : ℝ), desc = LocationDescription.Coordinates long lat :=
  sorry

end NUMINAMATH_CALUDE_coordinates_unique_location_l3301_330134


namespace NUMINAMATH_CALUDE_pyramid_with_10_edges_has_6_vertices_l3301_330179

-- Define a pyramid structure
structure Pyramid where
  base_sides : ℕ
  edges : ℕ
  vertices : ℕ

-- Theorem statement
theorem pyramid_with_10_edges_has_6_vertices :
  ∀ p : Pyramid, p.edges = 10 → p.vertices = 6 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_with_10_edges_has_6_vertices_l3301_330179


namespace NUMINAMATH_CALUDE_minimize_quadratic_expression_l3301_330117

theorem minimize_quadratic_expression (b : ℝ) :
  let f : ℝ → ℝ := λ x => (1/3) * x^2 + 7*x - 6
  ∀ x, f b ≤ f x ↔ b = -21/2 :=
by sorry

end NUMINAMATH_CALUDE_minimize_quadratic_expression_l3301_330117


namespace NUMINAMATH_CALUDE_inheritance_problem_l3301_330167

theorem inheritance_problem (S₁ S₂ S₃ S₄ D N : ℕ) :
  S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0 ∧ D > 0 ∧ N > 0 →
  Nat.sqrt S₁ = S₂ / 2 →
  Nat.sqrt S₁ = S₃ - 2 →
  Nat.sqrt S₁ = S₄ + 2 →
  Nat.sqrt S₁ = 2 * D →
  Nat.sqrt S₁ = N * N →
  S₁ + S₂ + S₃ + S₄ + D + N < 1500 →
  S₁ + S₂ + S₃ + S₄ + D + N = 1464 :=
by sorry

#eval Nat.sqrt 1296  -- Should output 36
#eval 72 / 2         -- Should output 36
#eval 38 - 2         -- Should output 36
#eval 34 + 2         -- Should output 36
#eval 2 * 18         -- Should output 36
#eval 6 * 6          -- Should output 36
#eval 1296 + 72 + 38 + 34 + 18 + 6  -- Should output 1464

end NUMINAMATH_CALUDE_inheritance_problem_l3301_330167


namespace NUMINAMATH_CALUDE_anitas_class_size_l3301_330163

/-- The number of students in Anita's class -/
def num_students : ℕ := 360 / 6

/-- Theorem: The number of students in Anita's class is 60 -/
theorem anitas_class_size :
  num_students = 60 :=
by sorry

end NUMINAMATH_CALUDE_anitas_class_size_l3301_330163


namespace NUMINAMATH_CALUDE_subset_relation_l3301_330121

theorem subset_relation (x : ℝ) : x^2 - x < 0 → x < 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_relation_l3301_330121


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_f_always_greater_equal_g_iff_l3301_330153

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |2*x + 1|
def g (x : ℝ) : ℝ := x + 2

-- Statement for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ g x} = {x : ℝ | 0 ≤ x ∧ x ≤ 2/3} := by sorry

-- Statement for part (2)
theorem f_always_greater_equal_g_iff (a : ℝ) :
  (∀ x : ℝ, f a x ≥ g x) ↔ a ≥ 2 := by sorry

-- Condition that a > 0
axiom a_positive : ∀ a : ℝ, a > 0

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_f_always_greater_equal_g_iff_l3301_330153


namespace NUMINAMATH_CALUDE_power_equality_l3301_330159

theorem power_equality : (3 : ℕ) ^ 20 = 243 ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3301_330159


namespace NUMINAMATH_CALUDE_coffee_mix_theorem_l3301_330108

/-- Calculates the price per pound of a coffee mix given the prices and quantities of two types of coffee. -/
def coffee_mix_price (price1 price2 : ℚ) (quantity1 quantity2 : ℚ) : ℚ :=
  (price1 * quantity1 + price2 * quantity2) / (quantity1 + quantity2)

/-- Theorem stating that mixing equal quantities of two types of coffee priced at $2.15 and $2.45 per pound
    results in a mix priced at $2.30 per pound. -/
theorem coffee_mix_theorem :
  let price1 : ℚ := 215 / 100
  let price2 : ℚ := 245 / 100
  let quantity1 : ℚ := 9
  let quantity2 : ℚ := 9
  coffee_mix_price price1 price2 quantity1 quantity2 = 230 / 100 := by
  sorry

#eval coffee_mix_price (215/100) (245/100) 9 9

end NUMINAMATH_CALUDE_coffee_mix_theorem_l3301_330108


namespace NUMINAMATH_CALUDE_smallest_number_less_than_negative_one_l3301_330101

theorem smallest_number_less_than_negative_one :
  let numbers : List ℝ := [-1/2, 0, |(-2)|, -3]
  ∀ x ∈ numbers, x < -1 ↔ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_less_than_negative_one_l3301_330101


namespace NUMINAMATH_CALUDE_chocolate_bars_unsold_l3301_330113

theorem chocolate_bars_unsold (total_bars : ℕ) (price_per_bar : ℕ) (revenue : ℕ) : 
  total_bars = 11 → price_per_bar = 4 → revenue = 16 → total_bars - (revenue / price_per_bar) = 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_unsold_l3301_330113


namespace NUMINAMATH_CALUDE_other_x_axis_point_on_circle_l3301_330125

def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

theorem other_x_axis_point_on_circle :
  let C : Set (ℝ × ℝ) := Circle (0, 0) 16
  (16, 0) ∈ C →
  (-16, 0) ∈ C ∧ (-16, 0).2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_x_axis_point_on_circle_l3301_330125


namespace NUMINAMATH_CALUDE_skew_symmetric_determinant_nonnegative_l3301_330158

theorem skew_symmetric_determinant_nonnegative 
  (a b c d e f : ℝ) : 
  (a * f - b * e + c * d)^2 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_skew_symmetric_determinant_nonnegative_l3301_330158


namespace NUMINAMATH_CALUDE_CD_possible_values_l3301_330138

-- Define the points on the number line
def A : ℝ := -3
def B : ℝ := 6

-- Define the distances
def AC : ℝ := 8
def BD : ℝ := 2

-- Define the possible positions for C and D
def C1 : ℝ := A + AC
def C2 : ℝ := A - AC
def D1 : ℝ := B + BD
def D2 : ℝ := B - BD

-- Define the set of possible CD values
def CD_values : Set ℝ := {|C1 - D1|, |C1 - D2|, |C2 - D1|, |C2 - D2|}

-- Theorem statement
theorem CD_possible_values : CD_values = {3, 1, 19, 15} := by sorry

end NUMINAMATH_CALUDE_CD_possible_values_l3301_330138


namespace NUMINAMATH_CALUDE_sheena_sewing_hours_per_week_l3301_330103

/-- Proves that Sheena sews 4 hours per week given the problem conditions -/
theorem sheena_sewing_hours_per_week 
  (time_per_dress : ℕ) 
  (num_dresses : ℕ) 
  (total_weeks : ℕ) 
  (h1 : time_per_dress = 12)
  (h2 : num_dresses = 5)
  (h3 : total_weeks = 15) :
  (time_per_dress * num_dresses) / total_weeks = 4 :=
by sorry

end NUMINAMATH_CALUDE_sheena_sewing_hours_per_week_l3301_330103


namespace NUMINAMATH_CALUDE_square_rectangle_area_relationship_l3301_330174

theorem square_rectangle_area_relationship : 
  ∃ (x₁ x₂ : ℝ), 
    (∀ x : ℝ, 3 * (x - 2)^2 = (x - 3) * (x + 4) → x = x₁ ∨ x = x₂) ∧
    x₁ + x₂ = 19/2 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relationship_l3301_330174


namespace NUMINAMATH_CALUDE_inscribed_rectangle_max_area_l3301_330147

theorem inscribed_rectangle_max_area :
  ∀ (x : ℝ) (r l b : ℝ),
  x > 0 ∧
  x^2 - 25*x + 144 = 0 ∧
  r^2 = x ∧
  l = (2/5) * r ∧
  ∃ (ratio : ℝ), ratio^2 - 3*ratio - 10 = 0 ∧ ratio > 0 ∧ l / b = ratio →
  l * b ≤ 0.512 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_max_area_l3301_330147
