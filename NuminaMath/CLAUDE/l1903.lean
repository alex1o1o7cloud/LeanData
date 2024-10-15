import Mathlib

namespace NUMINAMATH_CALUDE_angles_in_range_l1903_190343

-- Define the set S
def S : Set ℝ := {x | ∃ k : ℤ, x = k * 360 + 370 + 23 / 60}

-- Define the range of angles
def inRange (x : ℝ) : Prop := -720 ≤ x ∧ x < 360

-- State the theorem
theorem angles_in_range :
  ∃! (a b c : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    inRange a ∧ inRange b ∧ inRange c ∧
    a = -709 - 37 / 60 ∧
    b = -349 - 37 / 60 ∧
    c = 10 + 23 / 60 :=
  sorry

end NUMINAMATH_CALUDE_angles_in_range_l1903_190343


namespace NUMINAMATH_CALUDE_triangle_area_comparison_l1903_190389

/-- A triangle with side lengths and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

/-- Predicate to check if a triangle is acute -/
def is_acute (t : Triangle) : Prop := sorry

theorem triangle_area_comparison (t₁ t₂ : Triangle) 
  (h_acute : is_acute t₂)
  (h_a : t₁.a ≤ t₂.a)
  (h_b : t₁.b ≤ t₂.b)
  (h_c : t₁.c ≤ t₂.c) :
  t₁.area ≤ t₂.area :=
sorry

end NUMINAMATH_CALUDE_triangle_area_comparison_l1903_190389


namespace NUMINAMATH_CALUDE_condition_A_neither_necessary_nor_sufficient_l1903_190375

/-- Condition A: √(1 + sin θ) = a -/
def condition_A (θ : Real) (a : Real) : Prop :=
  Real.sqrt (1 + Real.sin θ) = a

/-- Condition B: sin(θ/2) + cos(θ/2) = a -/
def condition_B (θ : Real) (a : Real) : Prop :=
  Real.sin (θ / 2) + Real.cos (θ / 2) = a

/-- Theorem stating that condition A is neither necessary nor sufficient for condition B -/
theorem condition_A_neither_necessary_nor_sufficient :
  ¬(∀ θ a, condition_A θ a ↔ condition_B θ a) ∧
  ¬(∀ θ a, condition_A θ a → condition_B θ a) ∧
  ¬(∀ θ a, condition_B θ a → condition_A θ a) := by
  sorry

end NUMINAMATH_CALUDE_condition_A_neither_necessary_nor_sufficient_l1903_190375


namespace NUMINAMATH_CALUDE_pencil_sharpening_hours_l1903_190332

/-- The number of times Jenine can sharpen a pencil before it runs out -/
def sharpen_times : ℕ := 5

/-- The number of pencils Jenine already has -/
def initial_pencils : ℕ := 10

/-- The total number of hours Jenine needs to write -/
def total_writing_hours : ℕ := 105

/-- The cost of a new pencil in dollars -/
def pencil_cost : ℕ := 2

/-- The amount Jenine needs to spend on more pencils in dollars -/
def additional_pencil_cost : ℕ := 8

/-- The number of hours of use Jenine gets from sharpening a pencil once -/
def hours_per_sharpen : ℚ := 1.5

theorem pencil_sharpening_hours :
  let total_pencils := initial_pencils + additional_pencil_cost / pencil_cost
  total_pencils * sharpen_times * hours_per_sharpen = total_writing_hours :=
by sorry

end NUMINAMATH_CALUDE_pencil_sharpening_hours_l1903_190332


namespace NUMINAMATH_CALUDE_justin_sabrina_pencils_l1903_190397

/-- Given that Justin and Sabrina have 50 pencils combined, Justin has 8 more pencils than m times 
    Sabrina's pencils, and Sabrina has 14 pencils, prove that m = 2. -/
theorem justin_sabrina_pencils (total : ℕ) (justin_extra : ℕ) (sabrina_pencils : ℕ) (m : ℕ) 
  (h1 : total = 50)
  (h2 : justin_extra = 8)
  (h3 : sabrina_pencils = 14)
  (h4 : total = (m * sabrina_pencils + justin_extra) + sabrina_pencils) :
  m = 2 := by sorry

end NUMINAMATH_CALUDE_justin_sabrina_pencils_l1903_190397


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1903_190323

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1903_190323


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1903_190329

theorem inscribed_square_area (XY ZC : ℝ) (h1 : XY = 40) (h2 : ZC = 70) :
  let s := Real.sqrt (XY * ZC)
  s * s = 2800 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1903_190329


namespace NUMINAMATH_CALUDE_multiplication_problems_l1903_190320

theorem multiplication_problems :
  (25 * 5 * 2 * 4 = 1000) ∧ (1111 * 9999 = 11108889) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problems_l1903_190320


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1903_190363

theorem arithmetic_expression_equality : (5 * 4)^2 + (10 * 2) - 36 / 3 = 408 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1903_190363


namespace NUMINAMATH_CALUDE_compatible_pairs_theorem_l1903_190347

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def product_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * product_of_digits (n / 10)

def is_compatible (a b : ℕ) : Prop :=
  (a = sum_of_digits b ∧ b = product_of_digits a) ∨
  (b = sum_of_digits a ∧ a = product_of_digits b)

def compatible_pairs_within (n : ℕ) : Set (ℕ × ℕ) :=
  {p | p.1 ≤ n ∧ p.2 ≤ n ∧ is_compatible p.1 p.2}

def compatible_pairs_within_one_greater (n m : ℕ) : Set (ℕ × ℕ) :=
  {p | p.1 ≤ n ∧ p.2 ≤ n ∧ (p.1 > m ∨ p.2 > m) ∧ is_compatible p.1 p.2}

theorem compatible_pairs_theorem :
  compatible_pairs_within 100 = {(9, 11), (12, 36)} ∧
  compatible_pairs_within_one_greater 1000 99 = {(135, 19), (144, 19)} := by sorry

end NUMINAMATH_CALUDE_compatible_pairs_theorem_l1903_190347


namespace NUMINAMATH_CALUDE_figure_area_l1903_190356

/-- The total area of a figure composed of four rectangles with given dimensions --/
def total_area (r1_height r1_width r2_height r2_width r3_height r3_width r4_height r4_width : ℕ) : ℕ :=
  r1_height * r1_width + r2_height * r2_width + r3_height * r3_width + r4_height * r4_width

/-- Theorem stating that the total area of the given figure is 89 square units --/
theorem figure_area : total_area 7 6 2 6 5 4 3 5 = 89 := by
  sorry

end NUMINAMATH_CALUDE_figure_area_l1903_190356


namespace NUMINAMATH_CALUDE_cube_edge_length_proof_l1903_190388

/-- The edge length of a cube that, when fully immersed in a rectangular vessel
    with base dimensions 20 cm × 15 cm, causes a water level rise of 5.76 cm. -/
def cube_edge_length : ℝ := 12

/-- The base area of the rectangular vessel in square centimeters. -/
def vessel_base_area : ℝ := 20 * 15

/-- The rise in water level in centimeters when the cube is fully immersed. -/
def water_level_rise : ℝ := 5.76

/-- The volume of water displaced by the cube in cubic centimeters. -/
def displaced_volume : ℝ := vessel_base_area * water_level_rise

theorem cube_edge_length_proof :
  cube_edge_length ^ 3 = displaced_volume :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_length_proof_l1903_190388


namespace NUMINAMATH_CALUDE_general_equation_l1903_190362

theorem general_equation (n : ℝ) : n ≠ 4 ∧ 8 - n ≠ 4 → 
  (n / (n - 4)) + ((8 - n) / ((8 - n) - 4)) = 2 := by sorry

end NUMINAMATH_CALUDE_general_equation_l1903_190362


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l1903_190374

theorem last_digit_of_one_over_three_to_fifteen (n : ℕ) : 
  n = 15 → (1 : ℚ) / (3 ^ n) * 10^n % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l1903_190374


namespace NUMINAMATH_CALUDE_consecutive_integer_product_divisibility_l1903_190361

theorem consecutive_integer_product_divisibility (k : ℤ) : 
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 5 * m) →
  (∃ m : ℤ, n = 10 * m) ∧
  (∃ m : ℤ, n = 15 * m) ∧
  (∃ m : ℤ, n = 30 * m) ∧
  (∃ m : ℤ, n = 60 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 20 * m) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integer_product_divisibility_l1903_190361


namespace NUMINAMATH_CALUDE_difference_of_squares_l1903_190393

theorem difference_of_squares (m n : ℝ) : (m + n) * (-m + n) = -m^2 + n^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1903_190393


namespace NUMINAMATH_CALUDE_magazine_selling_price_l1903_190301

/-- Given the cost price, number of magazines, and total gain, 
    calculate the selling price per magazine. -/
theorem magazine_selling_price 
  (cost_price : ℝ) 
  (num_magazines : ℕ) 
  (total_gain : ℝ) 
  (h1 : cost_price = 3)
  (h2 : num_magazines = 10)
  (h3 : total_gain = 5) :
  (cost_price * num_magazines + total_gain) / num_magazines = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_magazine_selling_price_l1903_190301


namespace NUMINAMATH_CALUDE_original_price_calculation_l1903_190304

theorem original_price_calculation (original_price new_price : ℝ) : 
  new_price = 0.8 * original_price ∧ new_price = 80 → original_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1903_190304


namespace NUMINAMATH_CALUDE_complex_modulus_l1903_190360

theorem complex_modulus (z : ℂ) (h : z * (1 + Complex.I) = Complex.I ^ 2016) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1903_190360


namespace NUMINAMATH_CALUDE_inequality_proof_l1903_190302

theorem inequality_proof (a b c α : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c * (a^α + b^α + c^α) ≥ a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ∧
  (a * b * c * (a^α + b^α + c^α) = a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1903_190302


namespace NUMINAMATH_CALUDE_shot_put_distance_l1903_190322

/-- The horizontal distance at which a shot put hits the ground, given its trajectory. -/
theorem shot_put_distance : ∃ x : ℝ, x > 0 ∧ 
  (-1/12 * x^2 + 2/3 * x + 5/3 = 0) ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_shot_put_distance_l1903_190322


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l1903_190386

theorem product_of_sum_and_difference (a b : ℝ) 
  (sum_eq : a + b = 7)
  (diff_eq : a - b = 2) :
  a * b = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l1903_190386


namespace NUMINAMATH_CALUDE_sqrt_seven_less_than_three_l1903_190349

theorem sqrt_seven_less_than_three : Real.sqrt 7 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_less_than_three_l1903_190349


namespace NUMINAMATH_CALUDE_can_obtain_any_number_l1903_190336

/-- Represents the allowed operations on natural numbers -/
inductive Operation
  | append4 : Operation
  | append0 : Operation
  | divideBy2 : Operation

/-- Applies an operation to a natural number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.append4 => n * 10 + 4
  | Operation.append0 => n * 10
  | Operation.divideBy2 => if n % 2 = 0 then n / 2 else n

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a natural number -/
def applySequence (n : ℕ) (seq : OperationSequence) : ℕ :=
  seq.foldl applyOperation n

/-- Theorem: Any natural number can be obtained from 4 using the allowed operations -/
theorem can_obtain_any_number : ∀ (n : ℕ), ∃ (seq : OperationSequence), applySequence 4 seq = n := by
  sorry

end NUMINAMATH_CALUDE_can_obtain_any_number_l1903_190336


namespace NUMINAMATH_CALUDE_eel_count_l1903_190398

theorem eel_count (electric moray freshwater : ℕ) 
  (h1 : moray + freshwater = 12)
  (h2 : electric + freshwater = 14)
  (h3 : electric + moray = 16) :
  electric + moray + freshwater = 21 := by
sorry

end NUMINAMATH_CALUDE_eel_count_l1903_190398


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_side_length_l1903_190313

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the circle
def Circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }

-- Define the theorem
theorem cyclic_quadrilateral_side_length 
  (ABCD : Quadrilateral) 
  (inscribed : ABCD.A ∈ Circle ∧ ABCD.B ∈ Circle ∧ ABCD.C ∈ Circle ∧ ABCD.D ∈ Circle) 
  (perp_diagonals : (ABCD.A.1 - ABCD.C.1) * (ABCD.B.1 - ABCD.D.1) + 
                    (ABCD.A.2 - ABCD.C.2) * (ABCD.B.2 - ABCD.D.2) = 0)
  (AB_length : (ABCD.A.1 - ABCD.B.1)^2 + (ABCD.A.2 - ABCD.B.2)^2 = 9) :
  (ABCD.C.1 - ABCD.D.1)^2 + (ABCD.C.2 - ABCD.D.2)^2 = 7 :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_side_length_l1903_190313


namespace NUMINAMATH_CALUDE_min_value_expression_l1903_190394

theorem min_value_expression (x y : ℝ) : (x * y + 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1903_190394


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1903_190391

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (1 - Complex.I) * (-2 + a * Complex.I)
  (z.re = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1903_190391


namespace NUMINAMATH_CALUDE_conditional_equivalence_l1903_190390

theorem conditional_equivalence (P Q : Prop) :
  (P → ¬Q) ↔ (Q → ¬P) := by sorry

end NUMINAMATH_CALUDE_conditional_equivalence_l1903_190390


namespace NUMINAMATH_CALUDE_iphone_cost_l1903_190396

/-- The cost of the new iPhone given trade-in value, weekly earnings, and work duration -/
theorem iphone_cost (trade_in_value : ℕ) (weekly_earnings : ℕ) (work_weeks : ℕ) : 
  trade_in_value = 240 → weekly_earnings = 80 → work_weeks = 7 →
  trade_in_value + weekly_earnings * work_weeks = 800 := by
  sorry

end NUMINAMATH_CALUDE_iphone_cost_l1903_190396


namespace NUMINAMATH_CALUDE_lindas_furniture_fraction_l1903_190379

theorem lindas_furniture_fraction (savings : ℚ) (tv_cost : ℚ) 
  (h1 : savings = 920)
  (h2 : tv_cost = 230) :
  (savings - tv_cost) / savings = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_lindas_furniture_fraction_l1903_190379


namespace NUMINAMATH_CALUDE_average_first_17_even_numbers_l1903_190344

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * (i + 1))

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem average_first_17_even_numbers : 
  average (first_n_even_numbers 17) = 20 := by
sorry

end NUMINAMATH_CALUDE_average_first_17_even_numbers_l1903_190344


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1903_190381

theorem cubic_root_sum (p q r : ℂ) : 
  (p^3 - 2*p - 2 = 0) → 
  (q^3 - 2*q - 2 = 0) → 
  (r^3 - 2*r - 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1903_190381


namespace NUMINAMATH_CALUDE_pizza_area_increase_l1903_190385

/-- The percentage increase in area from a circle with radius 5 to a circle with radius 7 -/
theorem pizza_area_increase : ∀ (π : ℝ), π > 0 →
  (π * 7^2 - π * 5^2) / (π * 5^2) * 100 = 96 := by
  sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l1903_190385


namespace NUMINAMATH_CALUDE_even_n_with_specific_digit_sums_l1903_190395

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem even_n_with_specific_digit_sums 
  (n : ℕ) 
  (n_positive : 0 < n) 
  (sum_n : sum_of_digits n = 2014) 
  (sum_5n : sum_of_digits (5 * n) = 1007) : 
  Even n := by sorry

end NUMINAMATH_CALUDE_even_n_with_specific_digit_sums_l1903_190395


namespace NUMINAMATH_CALUDE_least_common_multiple_7_6_4_l1903_190300

theorem least_common_multiple_7_6_4 : ∃ (n : ℕ), n > 0 ∧ 7 ∣ n ∧ 6 ∣ n ∧ 4 ∣ n ∧ ∀ (m : ℕ), m > 0 → 7 ∣ m → 6 ∣ m → 4 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_7_6_4_l1903_190300


namespace NUMINAMATH_CALUDE_completing_square_l1903_190316

theorem completing_square (x : ℝ) : x^2 - 2*x - 2 = 0 ↔ (x - 1)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l1903_190316


namespace NUMINAMATH_CALUDE_smallest_d_divisible_by_11_l1903_190352

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def number_with_d (d : ℕ) : ℕ :=
  457000 + d * 100 + 1

theorem smallest_d_divisible_by_11 :
  ∀ d : ℕ, d < 10 →
    (is_divisible_by_11 (number_with_d d) → d ≥ 5) ∧
    (is_divisible_by_11 (number_with_d 5)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_divisible_by_11_l1903_190352


namespace NUMINAMATH_CALUDE_paint_remaining_paint_problem_l1903_190372

theorem paint_remaining (initial_paint : ℝ) (first_day_usage : ℝ) (second_day_usage : ℝ) (spill_loss : ℝ) : ℝ :=
  let remaining_after_first_day := initial_paint - first_day_usage
  let remaining_after_second_day := remaining_after_first_day - second_day_usage
  let remaining_after_spill := remaining_after_second_day - spill_loss
  remaining_after_spill

theorem paint_problem : paint_remaining 1 (1/2) ((1/2)/2) ((1/4)/4) = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_paint_problem_l1903_190372


namespace NUMINAMATH_CALUDE_grace_age_calculation_l1903_190367

/-- Grace's age in years -/
def grace_age : ℕ := sorry

/-- Grace's mother's age in years -/
def mother_age : ℕ := 80

/-- Grace's grandmother's age in years -/
def grandmother_age : ℕ := sorry

/-- Theorem stating Grace's age based on the given conditions -/
theorem grace_age_calculation :
  (grace_age = 3 * grandmother_age / 8) ∧
  (grandmother_age = 2 * mother_age) ∧
  (mother_age = 80) →
  grace_age = 60 := by
    sorry

end NUMINAMATH_CALUDE_grace_age_calculation_l1903_190367


namespace NUMINAMATH_CALUDE_real_roots_range_not_p_and_q_implies_range_l1903_190342

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0
def q (m : ℝ) : Prop := m ∈ Set.Icc (-1 : ℝ) 5

-- Theorem 1
theorem real_roots_range (m : ℝ) : p m → m ∈ Set.Iic 1 := by sorry

-- Theorem 2
theorem not_p_and_q_implies_range (m : ℝ) : 
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) → m ∈ Set.Iio (-1) ∪ Set.Ioc 1 5 := by sorry

end NUMINAMATH_CALUDE_real_roots_range_not_p_and_q_implies_range_l1903_190342


namespace NUMINAMATH_CALUDE_new_stationary_points_order_l1903_190399

-- Define the "new stationary point" for each function
def alpha : ℝ := 1

-- β is implicitly defined by the equation ln(β+1) = 1/(β+1)
def beta_equation (x : ℝ) : Prop := Real.log (x + 1) = 1 / (x + 1) ∧ x > 0

-- γ is implicitly defined by the equation γ³ - 1 = 3γ²
def gamma_equation (x : ℝ) : Prop := x^3 - 1 = 3 * x^2 ∧ x > 0

-- State the theorem
theorem new_stationary_points_order 
  (beta : ℝ) (h_beta : beta_equation beta)
  (gamma : ℝ) (h_gamma : gamma_equation gamma) :
  gamma > alpha ∧ alpha > beta := by
  sorry

end NUMINAMATH_CALUDE_new_stationary_points_order_l1903_190399


namespace NUMINAMATH_CALUDE_erased_number_proof_l1903_190306

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  n > 1 →
  (n : ℝ) * ((n : ℝ) + 21) / 2 - x = 23 * ((n : ℝ) - 1) →
  x = 36 := by
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l1903_190306


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1903_190312

noncomputable def nondecreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem functional_equation_solution
  (f : ℝ → ℝ)
  (h_nondecreasing : nondecreasing_function f)
  (h_f_0 : f 0 = 0)
  (h_f_1 : f 1 = 1)
  (h_equation : ∀ a b, a < 1 ∧ 1 < b → f a + f b = f a * f b + f (a + b - a * b)) :
  ∃ c k, c > 0 ∧ k ≥ 0 ∧
    (∀ x, x > 1 → f x = c * (x - 1) ^ k) ∧
    (∀ x, x < 1 → f x = 1 - (1 - x) ^ k) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1903_190312


namespace NUMINAMATH_CALUDE_holiday_duty_arrangements_l1903_190376

def staff_count : ℕ := 6
def days_count : ℕ := 3
def staff_per_day : ℕ := 2

def arrangement_count (n m k : ℕ) (restricted_days : ℕ) : ℕ :=
  (Nat.choose n k * Nat.choose (n - k) k) -
  (restricted_days * Nat.choose (n - 1) k * Nat.choose (n - k - 1) k) +
  (Nat.choose (n - 2) k * Nat.choose (n - k - 2) k)

theorem holiday_duty_arrangements :
  arrangement_count staff_count days_count staff_per_day 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_holiday_duty_arrangements_l1903_190376


namespace NUMINAMATH_CALUDE_no_definitive_inference_l1903_190369

-- Define the sets
variable (Mem Ens Vee : Set α)

-- Define the conditions
variable (h1 : ∃ x, x ∈ Mem ∧ x ∉ Ens)
variable (h2 : Ens ∩ Vee = ∅)

-- Define the potential inferences
def inference_A := ∃ x, x ∈ Mem ∧ x ∉ Vee
def inference_B := ∃ x, x ∈ Vee ∧ x ∉ Mem
def inference_C := Mem ∩ Vee = ∅
def inference_D := ∃ x, x ∈ Mem ∧ x ∈ Vee

-- The theorem to prove
theorem no_definitive_inference :
  ¬(inference_A Mem Vee) ∧
  ¬(inference_B Mem Vee) ∧
  ¬(inference_C Mem Vee) ∧
  ¬(inference_D Mem Vee) :=
sorry

end NUMINAMATH_CALUDE_no_definitive_inference_l1903_190369


namespace NUMINAMATH_CALUDE_h_j_h_3_equals_277_l1903_190354

def h (x : ℝ) : ℝ := 5 * x + 2

def j (x : ℝ) : ℝ := 3 * x + 4

theorem h_j_h_3_equals_277 : h (j (h 3)) = 277 := by
  sorry

end NUMINAMATH_CALUDE_h_j_h_3_equals_277_l1903_190354


namespace NUMINAMATH_CALUDE_interest_rate_is_zero_l1903_190368

/-- The interest rate for a TV purchase with installment payments -/
def interest_rate_tv_purchase (tv_price : ℕ) (num_installments : ℕ) (installment_amount : ℕ) (last_installment : ℕ) : ℚ :=
  if tv_price = 60000 ∧ 
     num_installments = 20 ∧ 
     installment_amount = 1000 ∧ 
     last_installment = 59000 ∧
     tv_price - installment_amount = last_installment
  then 0
  else 1 -- arbitrary non-zero value for other cases

/-- Theorem stating that the interest rate is 0% for the given TV purchase conditions -/
theorem interest_rate_is_zero :
  interest_rate_tv_purchase 60000 20 1000 59000 = 0 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_zero_l1903_190368


namespace NUMINAMATH_CALUDE_scott_sales_theorem_scott_total_sales_l1903_190303

/-- Calculates the total money made from selling items at given prices and quantities -/
def total_money_made (smoothie_price cake_price : ℕ) (smoothie_qty cake_qty : ℕ) : ℕ :=
  smoothie_price * smoothie_qty + cake_price * cake_qty

/-- Theorem stating that the total money made is equal to the sum of products of prices and quantities -/
theorem scott_sales_theorem (smoothie_price cake_price : ℕ) (smoothie_qty cake_qty : ℕ) :
  total_money_made smoothie_price cake_price smoothie_qty cake_qty =
  smoothie_price * smoothie_qty + cake_price * cake_qty :=
by
  sorry

/-- Verifies that Scott's total sales match the calculated amount -/
theorem scott_total_sales :
  total_money_made 3 2 40 18 = 156 :=
by
  sorry

end NUMINAMATH_CALUDE_scott_sales_theorem_scott_total_sales_l1903_190303


namespace NUMINAMATH_CALUDE_stock_percentage_sold_l1903_190341

/-- 
Given:
- cash_realized: The cash realized on selling the stock
- brokerage_rate: The brokerage rate as a percentage
- total_amount: The total amount including brokerage

Prove that the percentage of stock sold is equal to 
(cash_realized / (total_amount - total_amount * brokerage_rate / 100)) * 100
-/
theorem stock_percentage_sold 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (total_amount : ℝ) 
  (h1 : cash_realized = 104.25)
  (h2 : brokerage_rate = 0.25)
  (h3 : total_amount = 104) :
  (cash_realized / (total_amount - total_amount * brokerage_rate / 100)) * 100 = 
    (104.25 / (104 - 104 * 0.25 / 100)) * 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_sold_l1903_190341


namespace NUMINAMATH_CALUDE_midpoint_sum_l1903_190350

/-- Given points A (a, 6), B (-2, b), and P (2, 3) where P bisects AB, prove a + b = 6 -/
theorem midpoint_sum (a b : ℝ) : 
  (2 : ℝ) = (a + (-2)) / 2 → 
  (3 : ℝ) = (6 + b) / 2 → 
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_l1903_190350


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l1903_190346

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw_x : w / x = 5 / 4)
  (hy_z : y / z = 3 / 2)
  (hz_x : z / x = 1 / 4)
  (hsum : w + x + y + z = 60) :
  w / y = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l1903_190346


namespace NUMINAMATH_CALUDE_rectangle_formation_count_l1903_190333

/-- The number of ways to choose 2 items from a set of 5 items -/
def choose_two_from_five : ℕ := 10

/-- The number of horizontal lines -/
def num_horizontal_lines : ℕ := 5

/-- The number of vertical lines -/
def num_vertical_lines : ℕ := 5

/-- The number of ways to choose 4 lines to form a rectangle -/
def ways_to_form_rectangle : ℕ := choose_two_from_five * choose_two_from_five

theorem rectangle_formation_count :
  ways_to_form_rectangle = 100 :=
sorry

end NUMINAMATH_CALUDE_rectangle_formation_count_l1903_190333


namespace NUMINAMATH_CALUDE_value_of_a_l1903_190307

theorem value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 15 - 6 * a) : a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1903_190307


namespace NUMINAMATH_CALUDE_expression_simplification_l1903_190370

theorem expression_simplification :
  (4^2 * 7) / (8 * 9^2) * (8 * 9 * 11^2) / (4 * 7 * 11) = 44 / 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1903_190370


namespace NUMINAMATH_CALUDE_boys_without_glasses_in_class_l1903_190325

/-- The number of boys who do not wear glasses in Mrs. Lee's class -/
def boys_without_glasses (total_boys : ℕ) (total_with_glasses : ℕ) (girls_with_glasses : ℕ) : ℕ :=
  total_boys - (total_with_glasses - girls_with_glasses)

/-- Theorem stating the number of boys without glasses in Mrs. Lee's class -/
theorem boys_without_glasses_in_class : boys_without_glasses 30 36 21 = 15 := by
  sorry

end NUMINAMATH_CALUDE_boys_without_glasses_in_class_l1903_190325


namespace NUMINAMATH_CALUDE_scientific_notation_of_34_million_l1903_190311

theorem scientific_notation_of_34_million :
  ∃ (a : ℝ) (n : ℤ), 34000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.4 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_34_million_l1903_190311


namespace NUMINAMATH_CALUDE_candy_bar_cost_l1903_190340

theorem candy_bar_cost (initial_amount : ℕ) (change : ℕ) (cost : ℕ) : 
  initial_amount = 50 →
  change = 5 →
  cost = initial_amount - change →
  cost = 45 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l1903_190340


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l1903_190331

theorem book_sale_loss_percentage (selling_price_loss : ℝ) (selling_price_gain : ℝ) 
  (gain_percentage : ℝ) (loss_percentage : ℝ) :
  selling_price_loss = 450 →
  selling_price_gain = 550 →
  gain_percentage = 10 →
  (selling_price_gain = (100 + gain_percentage) / 100 * (100 / (100 + gain_percentage) * selling_price_gain)) →
  (loss_percentage = (((100 / (100 + gain_percentage) * selling_price_gain) - selling_price_loss) / 
    (100 / (100 + gain_percentage) * selling_price_gain)) * 100) →
  loss_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l1903_190331


namespace NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l1903_190384

theorem tan_fifteen_pi_fourths : Real.tan (15 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l1903_190384


namespace NUMINAMATH_CALUDE_hadley_total_distance_l1903_190359

-- Define the distances
def distance_to_grocery : ℕ := 2
def distance_to_pet_store : ℕ := 2 - 1
def distance_to_home : ℕ := 4 - 1

-- State the theorem
theorem hadley_total_distance :
  distance_to_grocery + distance_to_pet_store + distance_to_home = 6 := by
  sorry

end NUMINAMATH_CALUDE_hadley_total_distance_l1903_190359


namespace NUMINAMATH_CALUDE_intersection_point_y_value_l1903_190314

def f (x : ℝ) := 2 * x^2 - 3 * x + 10

theorem intersection_point_y_value :
  ∀ c : ℝ, f 7 = c → c = 87 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_y_value_l1903_190314


namespace NUMINAMATH_CALUDE_racketCostProof_l1903_190305

/-- Calculates the total cost of two rackets under a specific promotion --/
def totalCostOfRackets (fullPrice : ℚ) : ℚ :=
  fullPrice + (fullPrice / 2)

/-- Proves that the total cost of two rackets is $90 under the given conditions --/
theorem racketCostProof : totalCostOfRackets 60 = 90 := by
  sorry

end NUMINAMATH_CALUDE_racketCostProof_l1903_190305


namespace NUMINAMATH_CALUDE_matrix_product_equal_l1903_190364

/-- A 3x3 matrix of natural numbers -/
def Matrix3x3 := Fin 3 → Fin 3 → ℕ

/-- Check if all numbers in the matrix are distinct and not exceeding 40 -/
def valid_matrix (m : Matrix3x3) : Prop :=
  (∀ i j, m i j ≤ 40) ∧
  (∀ i j i' j', (i, j) ≠ (i', j') → m i j ≠ m i' j')

/-- Calculate the product of a row -/
def row_product (m : Matrix3x3) (i : Fin 3) : ℕ :=
  (m i 0) * (m i 1) * (m i 2)

/-- Calculate the product of a column -/
def col_product (m : Matrix3x3) (j : Fin 3) : ℕ :=
  (m 0 j) * (m 1 j) * (m 2 j)

/-- Calculate the product of the main diagonal -/
def diag1_product (m : Matrix3x3) : ℕ :=
  (m 0 0) * (m 1 1) * (m 2 2)

/-- Calculate the product of the other diagonal -/
def diag2_product (m : Matrix3x3) : ℕ :=
  (m 0 2) * (m 1 1) * (m 2 0)

/-- The given matrix -/
def given_matrix : Matrix3x3
| 0, 0 => 12
| 0, 1 => 9
| 0, 2 => 2
| 1, 0 => 1
| 1, 1 => 6
| 1, 2 => 36
| 2, 0 => 18
| 2, 1 => 4
| 2, 2 => 3

theorem matrix_product_equal :
  valid_matrix given_matrix ∧
  ∃ p : ℕ, (∀ i : Fin 3, row_product given_matrix i = p) ∧
           (∀ j : Fin 3, col_product given_matrix j = p) ∧
           (diag1_product given_matrix = p) ∧
           (diag2_product given_matrix = p) :=
by sorry

end NUMINAMATH_CALUDE_matrix_product_equal_l1903_190364


namespace NUMINAMATH_CALUDE_stock_price_increase_l1903_190357

theorem stock_price_increase (opening_price closing_price : ℝ) 
  (h1 : opening_price = 25)
  (h2 : closing_price = 28) :
  (closing_price - opening_price) / opening_price * 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l1903_190357


namespace NUMINAMATH_CALUDE_train_distance_problem_l1903_190358

/-- The distance between two cities given train travel conditions -/
theorem train_distance_problem : ∃ (dist : ℝ) (speed_A speed_B : ℝ),
  -- Two trains meet after 3.3 hours
  dist = 3.3 * (speed_A + speed_B) ∧
  -- Train A departing 24 minutes earlier condition
  0.4 * speed_A + 3 * (speed_A + speed_B) + 14 = 3.3 * (speed_A + speed_B) ∧
  -- Train B departing 36 minutes earlier condition
  0.6 * speed_B + 3 * (speed_A + speed_B) + 9 = 3.3 * (speed_A + speed_B) ∧
  -- The distance between the two cities is 660 km
  dist = 660 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_problem_l1903_190358


namespace NUMINAMATH_CALUDE_parabola_shift_equation_l1903_190365

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (k : ℝ) : Parabola :=
  { f := λ x => p.f (x + k) }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (m : ℝ) : Parabola :=
  { f := λ x => p.f x - m }

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { f := λ x => x^2 }

/-- The resulting parabola after shifting -/
def shifted_parabola : Parabola :=
  shift_vertical (shift_horizontal original_parabola 3) 4

theorem parabola_shift_equation :
  ∀ x, shifted_parabola.f x = (x + 3)^2 - 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_shift_equation_l1903_190365


namespace NUMINAMATH_CALUDE_monotone_increasing_iff_t_geq_five_l1903_190310

def a (x : ℝ) : ℝ × ℝ := (x^2, x + 1)
def b (x t : ℝ) : ℝ × ℝ := (1 - x, t)

def f (x t : ℝ) : ℝ := (a x).1 * (b x t).1 + (a x).2 * (b x t).2

theorem monotone_increasing_iff_t_geq_five :
  ∀ t : ℝ, (∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f x t < f y t) ↔ t ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_monotone_increasing_iff_t_geq_five_l1903_190310


namespace NUMINAMATH_CALUDE_gcd_of_multiple_4500_l1903_190373

theorem gcd_of_multiple_4500 (k : ℤ) : 
  let b : ℤ := 4500 * k
  Int.gcd (b^2 + 11*b + 40) (b + 8) = 3 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_multiple_4500_l1903_190373


namespace NUMINAMATH_CALUDE_minimum_cost_green_plants_l1903_190309

/-- Represents the number of pots of green lily -/
def green_lily_pots : ℕ → Prop :=
  λ x => x ≥ 31 ∧ x ≤ 46

/-- Represents the number of pots of spider plant -/
def spider_plant_pots : ℕ → Prop :=
  λ y => y ≥ 0 ∧ y ≤ 15

/-- The total cost of purchasing the plants -/
def total_cost (x y : ℕ) : ℕ :=
  9 * x + 6 * y

theorem minimum_cost_green_plants :
  ∀ x y : ℕ,
    green_lily_pots x →
    spider_plant_pots y →
    x + y = 46 →
    x ≥ 2 * y →
    total_cost x y ≥ 369 :=
by
  sorry

#check minimum_cost_green_plants

end NUMINAMATH_CALUDE_minimum_cost_green_plants_l1903_190309


namespace NUMINAMATH_CALUDE_stamp_sum_l1903_190321

theorem stamp_sum : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n < 100 ∧ n % 6 = 4 ∧ n % 8 = 2) ∧ 
  (∀ n < 100, n % 6 = 4 ∧ n % 8 = 2 → n ∈ S) ∧
  S.sum id = 68 := by
sorry

end NUMINAMATH_CALUDE_stamp_sum_l1903_190321


namespace NUMINAMATH_CALUDE_special_function_property_l1903_190324

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((x + y)^2) = f x^2 + 2*x*(f y) + y^2

/-- The number of possible values of f(1) -/
def m (f : ℝ → ℝ) : ℕ := sorry

/-- The sum of all possible values of f(1) -/
def t (f : ℝ → ℝ) : ℝ := sorry

/-- The main theorem -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) : 
  (m f : ℝ) * t f = 1 := by sorry

end NUMINAMATH_CALUDE_special_function_property_l1903_190324


namespace NUMINAMATH_CALUDE_michael_birdhouse_earnings_l1903_190371

/-- The amount of money Michael made from selling birdhouses -/
def michael_earnings (extra_large_price large_price medium_price small_price extra_small_price : ℕ)
  (extra_large_sold large_sold medium_sold small_sold extra_small_sold : ℕ) : ℕ :=
  extra_large_price * extra_large_sold +
  large_price * large_sold +
  medium_price * medium_sold +
  small_price * small_sold +
  extra_small_price * extra_small_sold

/-- Theorem stating that Michael's earnings from selling birdhouses is $487 -/
theorem michael_birdhouse_earnings :
  michael_earnings 45 22 16 10 5 3 5 7 8 10 = 487 := by sorry

end NUMINAMATH_CALUDE_michael_birdhouse_earnings_l1903_190371


namespace NUMINAMATH_CALUDE_remainder_problem_l1903_190328

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k % 5 = 2) 
  (h3 : k < 41) 
  (h4 : k % 7 = 3) : 
  k % 6 = 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1903_190328


namespace NUMINAMATH_CALUDE_triangle_side_length_l1903_190308

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths

-- State the theorem
theorem triangle_side_length 
  (t : Triangle) 
  (h1 : t.c = 1)           -- AC = 1
  (h2 : t.b = 3)           -- BC = 3
  (h3 : t.A + t.B = π / 3) -- A + B = 60° (in radians)
  : t.a = 2 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l1903_190308


namespace NUMINAMATH_CALUDE_binomial_probability_theorem_l1903_190319

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (X : BinomialRV) : ℝ := X.n * X.p

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Probability mass function for a binomial random variable -/
def pmf (X : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose X.n k : ℝ) * X.p ^ k * (1 - X.p) ^ (X.n - k)

theorem binomial_probability_theorem (X : BinomialRV) 
  (h_exp : expected_value X = 2)
  (h_var : variance X = 4/3) :
  pmf X 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_theorem_l1903_190319


namespace NUMINAMATH_CALUDE_min_green_chips_l1903_190366

/-- Given a basket of chips with three colors: green, yellow, and violet.
    This theorem proves that the minimum number of green chips is 120,
    given the conditions stated in the problem. -/
theorem min_green_chips (y v g : ℕ) : 
  v ≥ (2 : ℕ) * y / 3 →  -- violet chips are at least two-thirds of yellow chips
  v ≤ g / 4 →            -- violet chips are at most one-fourth of green chips
  y + v ≥ 75 →           -- sum of yellow and violet chips is at least 75
  g ≥ 120 :=             -- prove that the minimum number of green chips is 120
by sorry

end NUMINAMATH_CALUDE_min_green_chips_l1903_190366


namespace NUMINAMATH_CALUDE_room_area_in_sq_meters_l1903_190387

/-- The conversion factor from square feet to square meters -/
def sq_ft_to_sq_m : ℝ := 0.092903

/-- The length of the room in feet -/
def room_length : ℝ := 15

/-- The width of the room in feet -/
def room_width : ℝ := 8

/-- Theorem stating that the area of the room in square meters is approximately 11.14836 -/
theorem room_area_in_sq_meters :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ 
  |room_length * room_width * sq_ft_to_sq_m - 11.14836| < ε :=
sorry

end NUMINAMATH_CALUDE_room_area_in_sq_meters_l1903_190387


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1903_190378

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1903_190378


namespace NUMINAMATH_CALUDE_parabola_c_value_l1903_190338

/-- A parabola passing through three given points has a specific c value -/
theorem parabola_c_value (b c : ℝ) :
  (1^2 + b*1 + c = 2) ∧ 
  (4^2 + b*4 + c = 5) ∧ 
  (7^2 + b*7 + c = 2) →
  c = 9 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1903_190338


namespace NUMINAMATH_CALUDE_most_likely_outcome_is_equal_distribution_l1903_190337

def probability_of_outcome (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

theorem most_likely_outcome_is_equal_distribution :
  ∀ k : ℕ, k ≤ 8 →
    probability_of_outcome 8 4 ≥ probability_of_outcome 8 k :=
sorry

end NUMINAMATH_CALUDE_most_likely_outcome_is_equal_distribution_l1903_190337


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1903_190326

theorem quadratic_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1903_190326


namespace NUMINAMATH_CALUDE_quadratic_polynomial_value_at_14_l1903_190317

/-- A quadratic polynomial -/
def QuadraticPolynomial (α : Type*) [Field α] := α → α

/-- Divisibility condition for the polynomial -/
def DivisibilityCondition (q : QuadraticPolynomial ℝ) : Prop :=
  ∃ p : ℝ → ℝ, ∀ x, (q x)^3 - x = p x * (x - 2) * (x + 2) * (x - 9)

theorem quadratic_polynomial_value_at_14 
  (q : QuadraticPolynomial ℝ) 
  (h : DivisibilityCondition q) : 
  q 14 = -82 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_value_at_14_l1903_190317


namespace NUMINAMATH_CALUDE_head_start_problem_l1903_190339

/-- The head start problem -/
theorem head_start_problem (cristina_speed nicky_speed : ℝ) (catch_up_time : ℝ) 
  (h1 : cristina_speed = 5)
  (h2 : nicky_speed = 3)
  (h3 : catch_up_time = 24) :
  cristina_speed * catch_up_time - nicky_speed * catch_up_time = 48 := by
  sorry

#check head_start_problem

end NUMINAMATH_CALUDE_head_start_problem_l1903_190339


namespace NUMINAMATH_CALUDE_polynomial_zero_l1903_190334

-- Define the polynomial
def P (x : ℂ) (p q α β : ℤ) : ℂ := 
  (x - p) * (x - q) * (x^2 + α*x + β)

-- State the theorem
theorem polynomial_zero (p q : ℤ) : 
  ∃ (α β : ℤ), P ((3 + Complex.I * Real.sqrt 15) / 2) p q α β = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_zero_l1903_190334


namespace NUMINAMATH_CALUDE_cupcakes_eaten_later_is_22_l1903_190348

/-- Represents the cupcake business scenario --/
structure CupcakeBusiness where
  cost_per_cupcake : ℚ
  burnt_cupcakes : ℕ
  perfect_cupcakes : ℕ
  eaten_immediately : ℕ
  made_later : ℕ
  selling_price : ℚ
  net_profit : ℚ

/-- Calculates the number of cupcakes eaten later --/
def cupcakes_eaten_later (business : CupcakeBusiness) : ℚ :=
  let total_made := business.perfect_cupcakes + business.made_later
  let total_cost := (business.burnt_cupcakes + total_made) * business.cost_per_cupcake
  let available_for_sale := total_made - business.eaten_immediately
  ((available_for_sale * business.selling_price - total_cost - business.net_profit) / business.selling_price)

/-- Theorem stating the number of cupcakes eaten later --/
theorem cupcakes_eaten_later_is_22 (business : CupcakeBusiness)
  (h1 : business.cost_per_cupcake = 3/4)
  (h2 : business.burnt_cupcakes = 24)
  (h3 : business.perfect_cupcakes = 24)
  (h4 : business.eaten_immediately = 5)
  (h5 : business.made_later = 24)
  (h6 : business.selling_price = 2)
  (h7 : business.net_profit = 24) :
  cupcakes_eaten_later business = 22 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_eaten_later_is_22_l1903_190348


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1903_190345

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals_nonagon : ℕ :=
  let n : ℕ := 9  -- number of sides in a nonagon
  (n * (n - 3)) / 2

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonals :
  num_diagonals_nonagon = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1903_190345


namespace NUMINAMATH_CALUDE_grain_output_scientific_notation_l1903_190355

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem grain_output_scientific_notation :
  toScientificNotation 736000000 = ScientificNotation.mk 7.36 8 (by norm_num) := by
  sorry

end NUMINAMATH_CALUDE_grain_output_scientific_notation_l1903_190355


namespace NUMINAMATH_CALUDE_circle_area_difference_l1903_190335

theorem circle_area_difference : 
  let r1 : ℝ := 25
  let d2 : ℝ := 15
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1 ^ 2
  let area2 : ℝ := π * r2 ^ 2
  area1 - area2 = 568.75 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1903_190335


namespace NUMINAMATH_CALUDE_fraction_addition_l1903_190330

theorem fraction_addition : (1 : ℚ) / 4 + (3 : ℚ) / 5 = (17 : ℚ) / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1903_190330


namespace NUMINAMATH_CALUDE_range_of_a_when_one_in_B_range_of_a_when_B_subset_A_l1903_190353

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - a - 1) < 0}

-- Part 1: Range of a when 1 ∈ B
theorem range_of_a_when_one_in_B :
  ∀ a : ℝ, 1 ∈ B a ↔ 0 < a ∧ a < 1 := by sorry

-- Part 2: Range of a when B is a proper subset of A
theorem range_of_a_when_B_subset_A :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ B a → x ∈ A) ∧ (∃ y : ℝ, y ∈ A ∧ y ∉ B a) ↔ -1 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_when_one_in_B_range_of_a_when_B_subset_A_l1903_190353


namespace NUMINAMATH_CALUDE_f_max_value_l1903_190380

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := (S n : ℚ) / ((n + 32 : ℚ) * (S (n + 1) : ℚ))

theorem f_max_value :
  (∀ n : ℕ, f n ≤ 1/50) ∧ (∃ n : ℕ, f n = 1/50) := by sorry

end NUMINAMATH_CALUDE_f_max_value_l1903_190380


namespace NUMINAMATH_CALUDE_pencil_box_sequence_l1903_190392

theorem pencil_box_sequence (a : ℕ → ℕ) (h1 : a 1 = 78) (h2 : a 2 = 87) (h3 : a 3 = 96) (h5 : a 5 = 114)
  (h_arithmetic : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 9) : a 4 = 105 := by
  sorry

end NUMINAMATH_CALUDE_pencil_box_sequence_l1903_190392


namespace NUMINAMATH_CALUDE_cereal_box_total_price_l1903_190383

/-- Calculates the total price paid for discounted cereal boxes -/
theorem cereal_box_total_price 
  (initial_price : ℕ) 
  (price_reduction : ℕ) 
  (num_boxes : ℕ) 
  (h1 : initial_price = 104)
  (h2 : price_reduction = 24)
  (h3 : num_boxes = 20) : 
  (initial_price - price_reduction) * num_boxes = 1600 := by
  sorry

#check cereal_box_total_price

end NUMINAMATH_CALUDE_cereal_box_total_price_l1903_190383


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1903_190382

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) : 
  a + b = 5 → a^3 + b^3 = 125 → a * b = 0 := by sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1903_190382


namespace NUMINAMATH_CALUDE_circus_crowns_l1903_190377

theorem circus_crowns (total_feathers : ℕ) (feathers_per_crown : ℕ) (h1 : total_feathers = 6538) (h2 : feathers_per_crown = 7) :
  total_feathers / feathers_per_crown = 934 := by
sorry

end NUMINAMATH_CALUDE_circus_crowns_l1903_190377


namespace NUMINAMATH_CALUDE_power_comparison_a_l1903_190318

theorem power_comparison_a : 3^200 > 2^300 := by sorry

end NUMINAMATH_CALUDE_power_comparison_a_l1903_190318


namespace NUMINAMATH_CALUDE_triangle_problem_l1903_190351

theorem triangle_problem (A B C a b c : ℝ) : 
  a ≠ b →
  c = Real.sqrt 7 →
  b * Real.sin B - a * Real.sin A = Real.sqrt 3 * a * Real.cos A - Real.sqrt 3 * b * Real.cos B →
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 →
  C = π / 3 ∧ ((a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1903_190351


namespace NUMINAMATH_CALUDE_davids_english_marks_l1903_190327

/-- Given David's marks in four subjects and his average across five subjects,
    prove that his marks in English must be 74. -/
theorem davids_english_marks
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (h1 : math_marks = 65)
  (h2 : physics_marks = 82)
  (h3 : chemistry_marks = 67)
  (h4 : biology_marks = 90)
  (h5 : average_marks = 75.6)
  (h6 : (math_marks + physics_marks + chemistry_marks + biology_marks + english_marks : ℚ) / 5 = average_marks)
  : english_marks = 74 := by
  sorry

#check davids_english_marks

end NUMINAMATH_CALUDE_davids_english_marks_l1903_190327


namespace NUMINAMATH_CALUDE_first_number_equation_l1903_190315

theorem first_number_equation (x : ℝ) : (x + 32 + 53) / 3 = (21 + 47 + 22) / 3 + 3 ↔ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_first_number_equation_l1903_190315
