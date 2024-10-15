import Mathlib

namespace NUMINAMATH_CALUDE_min_modulus_complex_l566_56618

theorem min_modulus_complex (z : ℂ) : 
  (∃ x : ℝ, x^2 - 2*z*x + (3/4 : ℂ) + Complex.I = 0) → Complex.abs z ≥ 1 ∧ ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧ 
  (∃ x : ℝ, x^2 - 2*z₀*x + (3/4 : ℂ) + Complex.I = 0) := by
sorry

end NUMINAMATH_CALUDE_min_modulus_complex_l566_56618


namespace NUMINAMATH_CALUDE_add_zero_or_nine_divisible_by_nine_l566_56674

/-- Represents a ten-digit number with different digits -/
def TenDigitNumber := {n : Fin 10 → Fin 10 // Function.Injective n}

/-- The sum of digits in a ten-digit number -/
def digitSum (n : TenDigitNumber) : ℕ :=
  (Finset.univ.sum fun i => (n.val i).val)

/-- The theorem stating that adding 0 or 9 to a ten-digit number with different digits 
    results in a number divisible by 9 -/
theorem add_zero_or_nine_divisible_by_nine (n : TenDigitNumber) :
  (∃ x : Fin 10, x = 0 ∨ x = 9) ∧ 
  (∃ m : ℕ, (digitSum n + x) = 9 * m) := by
  sorry


end NUMINAMATH_CALUDE_add_zero_or_nine_divisible_by_nine_l566_56674


namespace NUMINAMATH_CALUDE_gunther_tractor_payment_l566_56690

/-- Calculates the monthly payment for a loan given the total amount and loan term in years -/
def monthly_payment (total_amount : ℕ) (years : ℕ) : ℚ :=
  (total_amount : ℚ) / (years * 12 : ℚ)

/-- Proves that for a $9000 loan over 5 years, the monthly payment is $150 -/
theorem gunther_tractor_payment :
  monthly_payment 9000 5 = 150 := by
  sorry

end NUMINAMATH_CALUDE_gunther_tractor_payment_l566_56690


namespace NUMINAMATH_CALUDE_no_x_axis_intersection_l566_56662

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x + 2)^2 - 1

-- Theorem stating that the function does not intersect the x-axis
theorem no_x_axis_intersection :
  ∀ x : ℝ, f x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_x_axis_intersection_l566_56662


namespace NUMINAMATH_CALUDE_triangle_angle_sum_and_type_l566_56624

/-- A triangle with angles a, b, and c is right if its largest angle is 90 degrees --/
def is_right_triangle (a b c : ℝ) : Prop :=
  max a (max b c) = 90

theorem triangle_angle_sum_and_type 
  (a b : ℝ) 
  (ha : a = 56)
  (hb : b = 34) :
  let c := 180 - a - b
  ∃ (x : ℝ), x = c ∧ x = 90 ∧ is_right_triangle a b c :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_and_type_l566_56624


namespace NUMINAMATH_CALUDE_theater_empty_showtime_l566_56612

/-- Represents a theater --/
structure Theater :=
  (id : Nat)

/-- Represents a student --/
structure Student :=
  (id : Nat)

/-- Represents a showtime --/
structure Showtime :=
  (id : Nat)

/-- Represents the attendance of students at a theater for a specific showtime --/
def Attendance := Theater → Showtime → Finset Student

theorem theater_empty_showtime 
  (students : Finset Student) 
  (theaters : Finset Theater) 
  (showtimes : Finset Showtime) 
  (attendance : Attendance) :
  (students.card = 7) →
  (theaters.card = 7) →
  (showtimes.card = 8) →
  (∀ s : Showtime, ∃! t : Theater, (attendance t s).card = 6) →
  (∀ s : Showtime, ∃! t : Theater, (attendance t s).card = 1) →
  (∀ stud : Student, ∀ t : Theater, ∃ s : Showtime, stud ∈ attendance t s) →
  (∀ t : Theater, ∃ s : Showtime, (attendance t s).card = 0) :=
by sorry

end NUMINAMATH_CALUDE_theater_empty_showtime_l566_56612


namespace NUMINAMATH_CALUDE_xy_minus_ten_squared_ge_64_l566_56617

theorem xy_minus_ten_squared_ge_64 (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
  (x * y - 10)^2 ≥ 64 := by
  sorry

end NUMINAMATH_CALUDE_xy_minus_ten_squared_ge_64_l566_56617


namespace NUMINAMATH_CALUDE_betty_oranges_l566_56658

theorem betty_oranges (boxes : ℝ) (oranges_per_box : ℕ) :
  boxes = 3.0 → oranges_per_box = 24 → boxes * oranges_per_box = 72 := by
  sorry

end NUMINAMATH_CALUDE_betty_oranges_l566_56658


namespace NUMINAMATH_CALUDE_square_root_of_16_l566_56623

theorem square_root_of_16 : {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_16_l566_56623


namespace NUMINAMATH_CALUDE_cube_root_simplification_l566_56675

theorem cube_root_simplification (N : ℝ) (h : N > 1) :
  (N^2 * (N^3 * N^(2/3))^(1/3))^(1/3) = N^(29/27) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l566_56675


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l566_56683

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 85 →
  a * d + b * c = 187 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 1486 := by
  sorry


end NUMINAMATH_CALUDE_max_sum_of_squares_l566_56683


namespace NUMINAMATH_CALUDE_remaining_two_average_l566_56664

theorem remaining_two_average (n₁ n₂ n₃ n₄ n₅ n₆ : ℝ) : 
  (n₁ + n₂ + n₃ + n₄ + n₅ + n₆) / 6 = 3.95 →
  (n₁ + n₂) / 2 = 3.4 →
  (n₃ + n₄) / 2 = 3.85 →
  (n₅ + n₆) / 2 = 4.6 := by
sorry

end NUMINAMATH_CALUDE_remaining_two_average_l566_56664


namespace NUMINAMATH_CALUDE_absolute_value_problem_l566_56625

theorem absolute_value_problem (x y : ℝ) 
  (h1 : x^2 + y^2 = 1) 
  (h2 : 20 * x^3 - 15 * x = 3) : 
  |20 * y^3 - 15 * y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l566_56625


namespace NUMINAMATH_CALUDE_binomial_expansion_cube_l566_56645

theorem binomial_expansion_cube (x y : ℝ) : 
  (x + y)^3 = x^3 + 3*x^2*y + 3*x*y^2 + y^3 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_cube_l566_56645


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l566_56613

/-- Represents the number of papers drawn from a school --/
structure SchoolSample where
  total : ℕ
  drawn : ℕ

/-- Represents the sampling data for all schools --/
structure SamplingData where
  schoolA : SchoolSample
  schoolB : SchoolSample
  schoolC : SchoolSample

/-- Calculates the total number of papers drawn using stratified sampling --/
def totalDrawn (data : SamplingData) : ℕ :=
  let ratio := data.schoolC.drawn / data.schoolC.total
  (data.schoolA.total + data.schoolB.total + data.schoolC.total) * ratio

theorem stratified_sampling_theorem (data : SamplingData) 
  (h1 : data.schoolA.total = 1260)
  (h2 : data.schoolB.total = 720)
  (h3 : data.schoolC.total = 900)
  (h4 : data.schoolC.drawn = 50) :
  totalDrawn data = 160 := by
  sorry

#eval totalDrawn { 
  schoolA := { total := 1260, drawn := 0 },
  schoolB := { total := 720, drawn := 0 },
  schoolC := { total := 900, drawn := 50 }
}

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l566_56613


namespace NUMINAMATH_CALUDE_unit_square_quadrilateral_bounds_l566_56671

/-- A quadrilateral formed by selecting one point on each side of a unit square -/
structure UnitSquareQuadrilateral where
  a : ℝ  -- Length of side a
  b : ℝ  -- Length of side b
  c : ℝ  -- Length of side c
  d : ℝ  -- Length of side d
  ha : 0 ≤ a ∧ a ≤ 1  -- a is between 0 and 1
  hb : 0 ≤ b ∧ b ≤ 1  -- b is between 0 and 1
  hc : 0 ≤ c ∧ c ≤ 1  -- c is between 0 and 1
  hd : 0 ≤ d ∧ d ≤ 1  -- d is between 0 and 1

theorem unit_square_quadrilateral_bounds (q : UnitSquareQuadrilateral) :
  2 ≤ q.a^2 + q.b^2 + q.c^2 + q.d^2 ∧ q.a^2 + q.b^2 + q.c^2 + q.d^2 ≤ 4 ∧
  2 * Real.sqrt 2 ≤ q.a + q.b + q.c + q.d ∧ q.a + q.b + q.c + q.d ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_unit_square_quadrilateral_bounds_l566_56671


namespace NUMINAMATH_CALUDE_owen_profit_l566_56673

/-- Calculates the profit from selling face masks given the following conditions:
  * Number of boxes bought
  * Cost per box
  * Number of masks per box
  * Number of boxes repacked
  * Number of large packets sold
  * Price of large packets
  * Number of masks in large packets
  * Price of small baggies
  * Number of masks in small baggies
-/
def calculate_profit (
  boxes_bought : ℕ
  ) (cost_per_box : ℚ
  ) (masks_per_box : ℕ
  ) (boxes_repacked : ℕ
  ) (large_packets_sold : ℕ
  ) (large_packet_price : ℚ
  ) (masks_per_large_packet : ℕ
  ) (small_baggie_price : ℚ
  ) (masks_per_small_baggie : ℕ
  ) : ℚ :=
  let total_cost := boxes_bought * cost_per_box
  let total_masks := boxes_bought * masks_per_box
  let repacked_masks := boxes_repacked * masks_per_box
  let large_packet_revenue := large_packets_sold * large_packet_price
  let remaining_masks := total_masks - (large_packets_sold * masks_per_large_packet)
  let small_baggies := remaining_masks / masks_per_small_baggie
  let small_baggie_revenue := small_baggies * small_baggie_price
  let total_revenue := large_packet_revenue + small_baggie_revenue
  total_revenue - total_cost

theorem owen_profit :
  calculate_profit 12 9 50 6 3 12 100 3 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_owen_profit_l566_56673


namespace NUMINAMATH_CALUDE_right_triangle_squares_problem_l566_56693

theorem right_triangle_squares_problem (x : ℝ) : 
  (3 * x)^2 + (6 * x)^2 + (1/2 * 3 * x * 6 * x) = 1200 → x = (10 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_squares_problem_l566_56693


namespace NUMINAMATH_CALUDE_unique_g_two_l566_56672

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → g x * g y = g (x * y) + 2010 * (1 / x^2 + 1 / y^2 + 2009)

theorem unique_g_two (g : ℝ → ℝ) (h : FunctionalEquation g) :
    ∃! v, g 2 = v ∧ v = 8041 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_g_two_l566_56672


namespace NUMINAMATH_CALUDE_new_average_weight_l566_56643

theorem new_average_weight (initial_students : Nat) (initial_avg_weight : ℝ) (new_student_weight : ℝ) :
  initial_students = 19 →
  initial_avg_weight = 15 →
  new_student_weight = 7 →
  let total_weight := initial_students * initial_avg_weight
  let new_total_weight := total_weight + new_student_weight
  let new_avg_weight := new_total_weight / (initial_students + 1)
  new_avg_weight = 14.6 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l566_56643


namespace NUMINAMATH_CALUDE_expression_equals_one_l566_56631

theorem expression_equals_one (x : ℝ) : 
  ((((x + 1)^2 * (x^2 - x + 1)^2) / (x^3 + 1)^2)^2) * 
  ((((x - 1)^2 * (x^2 + x + 1)^2) / (x^3 - 1)^2)^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_one_l566_56631


namespace NUMINAMATH_CALUDE_g_is_even_symmetry_axes_increasing_function_l566_56629

variable (f : ℝ → ℝ)

-- f is not constant
axiom not_constant : ∃ x y, f x ≠ f y

-- Definition of g
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem 1: g is even
theorem g_is_even : ∀ x, g f x = g f (-x) := by sorry

-- Definition of odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem 2: If f is odd and f(x) + f(2 + x) = 0, then f has axes of symmetry at x = 2n + 1
theorem symmetry_axes (h_odd : is_odd f) (h_sum : ∀ x, f x + f (2 + x) = 0) :
  ∀ n : ℤ, ∀ x : ℝ, f (2 * n + 1 + x) = -f (2 * n + 1 - x) := by sorry

-- Theorem 3: If (f(x₁) - f(x₂))/(x₁ - x₂) > 0 for x₁ ≠ x₂, then f is increasing
theorem increasing_function (h : ∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0) :
  ∀ x y, x < y → f x < f y := by sorry

end NUMINAMATH_CALUDE_g_is_even_symmetry_axes_increasing_function_l566_56629


namespace NUMINAMATH_CALUDE_total_cost_calculation_l566_56681

def tshirt_price : ℝ := 8
def sweater_price : ℝ := 18
def jacket_price : ℝ := 80
def jeans_price : ℝ := 35
def shoe_price : ℝ := 60

def jacket_discount : ℝ := 0.1
def shoe_discount : ℝ := 0.15

def clothing_tax_rate : ℝ := 0.05
def shoe_tax_rate : ℝ := 0.08

def tshirt_quantity : ℕ := 6
def sweater_quantity : ℕ := 4
def jacket_quantity : ℕ := 5
def jeans_quantity : ℕ := 3
def shoe_quantity : ℕ := 2

theorem total_cost_calculation :
  let tshirt_cost := tshirt_price * tshirt_quantity
  let sweater_cost := sweater_price * sweater_quantity
  let jacket_cost := jacket_price * jacket_quantity * (1 - jacket_discount)
  let jeans_cost := jeans_price * jeans_quantity
  let shoe_cost := shoe_price * shoe_quantity * (1 - shoe_discount)
  
  let clothing_subtotal := tshirt_cost + sweater_cost + jacket_cost + jeans_cost
  let shoe_subtotal := shoe_cost
  
  let clothing_tax := clothing_subtotal * clothing_tax_rate
  let shoe_tax := shoe_subtotal * shoe_tax_rate
  
  let total_cost := clothing_subtotal + shoe_subtotal + clothing_tax + shoe_tax
  
  total_cost = 724.41 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l566_56681


namespace NUMINAMATH_CALUDE_b_age_l566_56602

def problem (a b c d : ℕ) : Prop :=
  (a = b + 2) ∧ 
  (b = 2 * c) ∧ 
  (d = b - 3) ∧ 
  (a + b + c + d = 60)

theorem b_age (a b c d : ℕ) (h : problem a b c d) : b = 17 := by
  sorry

end NUMINAMATH_CALUDE_b_age_l566_56602


namespace NUMINAMATH_CALUDE_long_jump_records_correct_l566_56665

/-- Represents a long jump record -/
structure LongJumpRecord where
  height : Real
  record : Real

/-- Checks if a long jump record is correctly calculated and recorded -/
def is_correct_record (standard : Real) (jump : LongJumpRecord) : Prop :=
  jump.record = jump.height - standard

/-- The problem statement -/
theorem long_jump_records_correct (standard : Real) (xiao_ming : LongJumpRecord) (xiao_liang : LongJumpRecord)
  (h1 : standard = 1.5)
  (h2 : xiao_ming.height = 1.95)
  (h3 : xiao_ming.record = 0.45)
  (h4 : xiao_liang.height = 1.23)
  (h5 : xiao_liang.record = -0.23) :
  ¬(is_correct_record standard xiao_ming ∧ is_correct_record standard xiao_liang) :=
sorry

end NUMINAMATH_CALUDE_long_jump_records_correct_l566_56665


namespace NUMINAMATH_CALUDE_eel_length_ratio_l566_56647

theorem eel_length_ratio (total_length : ℝ) (jenna_length : ℝ) :
  total_length = 64 →
  jenna_length = 16 →
  (jenna_length / (total_length - jenna_length) = 1 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_eel_length_ratio_l566_56647


namespace NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_sixth_l566_56694

theorem arcsin_one_half_equals_pi_sixth : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_sixth_l566_56694


namespace NUMINAMATH_CALUDE_stating_weaver_production_increase_l566_56648

/-- Represents the daily increase in fabric production -/
def daily_increase : ℚ := 16 / 29

/-- Represents the initial daily production -/
def initial_production : ℕ := 5

/-- Represents the number of days -/
def days : ℕ := 30

/-- Represents the total production over the given period -/
def total_production : ℕ := 390

/-- 
Theorem stating that given the initial production and total production over a period,
the daily increase in production is as calculated.
-/
theorem weaver_production_increase : 
  initial_production * days + (days * (days - 1) / 2) * daily_increase = total_production := by
  sorry


end NUMINAMATH_CALUDE_stating_weaver_production_increase_l566_56648


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l566_56622

/-- Given a real number a, prove that the function f(x) = a^(x-1) + 3 passes through the point (1, 4) -/
theorem fixed_point_of_exponential_function (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 3
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l566_56622


namespace NUMINAMATH_CALUDE_problem_statement_l566_56668

theorem problem_statement (x y : ℝ) (h : |x - 5| + (x - y - 1)^2 = 0) : 
  (x - y)^2023 = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l566_56668


namespace NUMINAMATH_CALUDE_probability_of_red_is_half_l566_56688

/-- A cube with a specific color distribution -/
structure ColoredCube where
  total_faces : ℕ
  red_faces : ℕ
  yellow_faces : ℕ
  green_faces : ℕ
  tricolor_faces : ℕ

/-- The probability of a specific color facing up when throwing the cube -/
def probability_of_color (cube : ColoredCube) (color_faces : ℕ) : ℚ :=
  color_faces / cube.total_faces

/-- Our specific cube with the given color distribution -/
def our_cube : ColoredCube :=
  { total_faces := 6
  , red_faces := 2
  , yellow_faces := 2
  , green_faces := 1
  , tricolor_faces := 1 }

/-- Theorem stating that the probability of red facing up is 1/2 -/
theorem probability_of_red_is_half :
  probability_of_color our_cube (our_cube.red_faces + our_cube.tricolor_faces) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_is_half_l566_56688


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l566_56608

/-- Given three lines that intersect at the same point, prove the value of k. -/
theorem intersection_of_three_lines (x y : ℝ) (k : ℝ) : 
  y = -4 * x + 2 ∧ 
  y = 3 * x - 18 ∧ 
  y = 7 * x + k 
  → k = -206 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l566_56608


namespace NUMINAMATH_CALUDE_angle_terminal_side_ratio_l566_56684

theorem angle_terminal_side_ratio (a : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos a = 1 ∧ r * Real.sin a = -2) →
  (2 * Real.sin a) / Real.cos a = 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_ratio_l566_56684


namespace NUMINAMATH_CALUDE_fifth_term_value_l566_56699

/-- Given a sequence {aₙ} with sum of first n terms Sₙ = 2n(n+1), prove a₅ = 20 -/
theorem fifth_term_value (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : ∀ n : ℕ, S n = 2 * n * (n + 1)) : 
  a 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l566_56699


namespace NUMINAMATH_CALUDE_maplewood_elementary_difference_l566_56669

theorem maplewood_elementary_difference (students_per_class : ℕ) (guinea_pigs_per_class : ℕ) (num_classes : ℕ) :
  students_per_class = 20 →
  guinea_pigs_per_class = 3 →
  num_classes = 4 →
  students_per_class * num_classes - guinea_pigs_per_class * num_classes = 68 :=
by
  sorry

end NUMINAMATH_CALUDE_maplewood_elementary_difference_l566_56669


namespace NUMINAMATH_CALUDE_symmetry_axis_implies_equal_coefficients_l566_56611

/-- Given a function f(x) = a*sin(2x) + b*cos(2x) where ab ≠ 0,
    if f has a symmetry axis at x = π/8, then a = b -/
theorem symmetry_axis_implies_equal_coefficients
  (a b : ℝ) (hab : a * b ≠ 0)
  (h_symmetry : ∀ x : ℝ, a * Real.sin (2 * (π/8 + x)) + b * Real.cos (2 * (π/8 + x)) =
                         a * Real.sin (2 * (π/8 - x)) + b * Real.cos (2 * (π/8 - x))) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_symmetry_axis_implies_equal_coefficients_l566_56611


namespace NUMINAMATH_CALUDE_combined_savings_equal_individual_savings_l566_56652

/-- Represents the number of windows in a bundle -/
def bundle_size : ℕ := 7

/-- Represents the number of windows paid for in a bundle -/
def paid_windows_per_bundle : ℕ := 5

/-- Represents the cost of a single window -/
def window_cost : ℕ := 100

/-- Calculates the number of bundles needed for a given number of windows -/
def bundles_needed (windows : ℕ) : ℕ :=
  (windows + bundle_size - 1) / bundle_size

/-- Calculates the cost of windows with the promotion -/
def promotional_cost (windows : ℕ) : ℕ :=
  bundles_needed windows * paid_windows_per_bundle * window_cost

/-- Calculates the savings for a given number of windows -/
def savings (windows : ℕ) : ℕ :=
  windows * window_cost - promotional_cost windows

/-- Dave's required number of windows -/
def dave_windows : ℕ := 12

/-- Doug's required number of windows -/
def doug_windows : ℕ := 10

theorem combined_savings_equal_individual_savings :
  savings (dave_windows + doug_windows) = savings dave_windows + savings doug_windows :=
by sorry

end NUMINAMATH_CALUDE_combined_savings_equal_individual_savings_l566_56652


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l566_56606

/-- Proves the number of bottle caps Danny found at the park -/
theorem danny_bottle_caps 
  (thrown_away : ℕ) 
  (current_total : ℕ) 
  (found_more_than_thrown : ℕ) : 
  thrown_away = 35 → 
  current_total = 22 → 
  found_more_than_thrown = 1 → 
  ∃ (previous_total : ℕ) (found : ℕ), 
    found = thrown_away + found_more_than_thrown ∧ 
    current_total = previous_total - thrown_away + found ∧
    found = 36 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l566_56606


namespace NUMINAMATH_CALUDE_imaginary_part_proof_l566_56667

def i : ℂ := Complex.I

def z : ℂ := 1 - i

theorem imaginary_part_proof : Complex.im ((2 / z) + i ^ 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_proof_l566_56667


namespace NUMINAMATH_CALUDE_cos_alpha_value_l566_56651

theorem cos_alpha_value (α : Real) : 
  (∃ (x y : Real), x = 2 * Real.sin (π / 6) ∧ y = -2 * Real.cos (π / 6) ∧ 
   x = 2 * Real.sin α ∧ y = -2 * Real.cos α) → 
  Real.cos α = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l566_56651


namespace NUMINAMATH_CALUDE_remaining_crayons_l566_56634

def initial_crayons : ℕ := 440
def crayons_given_away : ℕ := 111
def crayons_lost : ℕ := 106

theorem remaining_crayons :
  initial_crayons - crayons_given_away - crayons_lost = 223 := by
  sorry

end NUMINAMATH_CALUDE_remaining_crayons_l566_56634


namespace NUMINAMATH_CALUDE_basketball_team_starters_l566_56601

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players -/
def total_players : ℕ := 18

/-- The number of quadruplets -/
def quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def starters : ℕ := 7

/-- The number of non-quadruplet players -/
def non_quadruplets : ℕ := total_players - quadruplets

theorem basketball_team_starters :
  choose total_players starters - choose non_quadruplets (starters - quadruplets) = 31460 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_starters_l566_56601


namespace NUMINAMATH_CALUDE_sin_90_degrees_l566_56637

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l566_56637


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l566_56653

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (7 - 5*x > 22) → x ≤ -4 ∧ (7 - 5*(-4) > 22) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l566_56653


namespace NUMINAMATH_CALUDE_red_shirt_pairs_l566_56661

theorem red_shirt_pairs (total_students : ℕ) (blue_students : ℕ) (red_students : ℕ)
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  total_students = 144 →
  blue_students = 63 →
  red_students = 81 →
  total_pairs = 72 →
  blue_blue_pairs = 21 →
  total_students = blue_students + red_students →
  ∃ (red_red_pairs : ℕ), red_red_pairs = 30 ∧
    red_red_pairs + blue_blue_pairs + (blue_students - 2 * blue_blue_pairs) = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_red_shirt_pairs_l566_56661


namespace NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l566_56650

theorem sin_product_equals_one_eighth :
  Real.sin (12 * π / 180) * Real.sin (36 * π / 180) * Real.sin (54 * π / 180) * Real.sin (84 * π / 180) = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l566_56650


namespace NUMINAMATH_CALUDE_fib_linear_combination_fib_quadratic_combination_l566_56616

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Part (a)
theorem fib_linear_combination (a b : ℝ) :
  (∀ n : ℕ, ∃ k : ℕ, a * fib n + b * fib (n + 1) = fib k) ↔
  ∃ k : ℕ, a = fib (k - 1) ∧ b = fib k :=
sorry

-- Part (b)
theorem fib_quadratic_combination (u v : ℝ) :
  (u > 0 ∧ v > 0 ∧ ∀ n : ℕ, ∃ k : ℕ, u * (fib n)^2 + v * (fib (n + 1))^2 = fib k) ↔
  u = 1 ∧ v = 1 :=
sorry

end NUMINAMATH_CALUDE_fib_linear_combination_fib_quadratic_combination_l566_56616


namespace NUMINAMATH_CALUDE_max_children_in_class_l566_56642

theorem max_children_in_class (x : ℕ) : 
  (∃ (chocolates_per_box : ℕ),
    -- Original plan with 6 boxes
    6 * chocolates_per_box = 10 * x + 40 ∧
    -- New plan with 4 boxes
    4 * chocolates_per_box ≥ 8 * (x - 1) + 4 ∧
    4 * chocolates_per_box < 8 * (x - 1) + 8) →
  x ≤ 23 :=
sorry

end NUMINAMATH_CALUDE_max_children_in_class_l566_56642


namespace NUMINAMATH_CALUDE_lucas_initial_beds_l566_56685

/-- The number of pet beds Lucas can add to his room -/
def additional_beds : ℕ := 8

/-- The number of beds required per pet -/
def beds_per_pet : ℕ := 2

/-- The total number of pets Lucas's room can accommodate -/
def total_pets : ℕ := 10

/-- The initial number of pet beds in Lucas's room -/
def initial_beds : ℕ := total_pets * beds_per_pet - additional_beds

theorem lucas_initial_beds :
  initial_beds = 12 :=
by sorry

end NUMINAMATH_CALUDE_lucas_initial_beds_l566_56685


namespace NUMINAMATH_CALUDE_polynomial_root_product_l566_56614

theorem polynomial_root_product (y₁ y₂ y₃ : ℂ) : 
  (y₁^3 - 3*y₁ + 1 = 0) → 
  (y₂^3 - 3*y₂ + 1 = 0) → 
  (y₃^3 - 3*y₃ + 1 = 0) → 
  (y₁^3 + 2) * (y₂^3 + 2) * (y₃^3 + 2) = -26 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_product_l566_56614


namespace NUMINAMATH_CALUDE_chess_tournament_games_l566_56692

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournamentGames (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 8 players, where each player plays twice with every other player, the total number of games played is 112 -/
theorem chess_tournament_games :
  tournamentGames 8 * 2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l566_56692


namespace NUMINAMATH_CALUDE_hyperbola_equation_l566_56635

/-- Given a hyperbola with a = 5 and c = 7, prove its standard equation. -/
theorem hyperbola_equation (a c : ℝ) (ha : a = 5) (hc : c = 7) :
  ∃ (x y : ℝ → ℝ), 
    (∀ t, x t ^ 2 / 25 - y t ^ 2 / 24 = 1) ∨
    (∀ t, y t ^ 2 / 25 - x t ^ 2 / 24 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l566_56635


namespace NUMINAMATH_CALUDE_two_zeros_iff_m_in_range_l566_56615

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * (2 * log x - x) + 1 / x^2 - 1 / x

theorem two_zeros_iff_m_in_range (m : ℝ) :
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x ≠ y ∧ f m x = 0 ∧ f m y = 0 ∧
    ∀ z : ℝ, 0 < z → f m z = 0 → (z = x ∨ z = y)) ↔
  m ∈ Set.Ioo (1 / (8 * (log 2 - 1))) 0 :=
sorry

end NUMINAMATH_CALUDE_two_zeros_iff_m_in_range_l566_56615


namespace NUMINAMATH_CALUDE_tangent_sum_identity_l566_56696

theorem tangent_sum_identity (α β γ : Real) (h : α + β + γ = Real.pi / 2) :
  Real.tan α * Real.tan β + Real.tan β * Real.tan γ + Real.tan γ * Real.tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_identity_l566_56696


namespace NUMINAMATH_CALUDE_book_donation_equation_l566_56659

/-- Proves that the equation for book donations over three years is correct -/
theorem book_donation_equation (x : ℝ) : 
  (400 : ℝ) + 400 * (1 + x) + 400 * (1 + x)^2 = 1525 → 
  (∃ (y : ℝ), y > 0 ∧ 400 * (1 + y) + 400 * (1 + y)^2 = 1125) :=
by
  sorry


end NUMINAMATH_CALUDE_book_donation_equation_l566_56659


namespace NUMINAMATH_CALUDE_f_monotone_increasing_iff_a_range_l566_56680

/-- A function f(x) = 2x^2 - ax + 5 that is monotonically increasing on [1, +∞) -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - a * x + 5

/-- The property of f being monotonically increasing on [1, +∞) -/
def monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y → f a x < f a y

/-- The theorem stating the range of a for which f is monotonically increasing on [1, +∞) -/
theorem f_monotone_increasing_iff_a_range :
  ∀ a : ℝ, monotone_increasing a ↔ a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_iff_a_range_l566_56680


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l566_56678

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of nitrogen atoms in the compound -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in the compound -/
def num_O : ℕ := 5

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := num_N * atomic_weight_N + num_O * atomic_weight_O

theorem compound_molecular_weight :
  molecular_weight = 108.02 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l566_56678


namespace NUMINAMATH_CALUDE_multiple_subtracted_l566_56698

theorem multiple_subtracted (a b : ℝ) (h1 : a / b = 4 / 1) 
  (h2 : ∃ x : ℝ, (a - x * b) / (2 * a - b) = 0.14285714285714285) : 
  ∃ x : ℝ, (a - x * b) / (2 * a - b) = 0.14285714285714285 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiple_subtracted_l566_56698


namespace NUMINAMATH_CALUDE_larger_number_problem_l566_56641

theorem larger_number_problem (smaller larger : ℚ) : 
  smaller = 48 → 
  larger - smaller = (1 : ℚ) / 3 * larger →
  larger = 72 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l566_56641


namespace NUMINAMATH_CALUDE_expression_evaluation_l566_56649

theorem expression_evaluation : 
  let x : ℤ := -2
  let expr := (x^2 - 4*x + 4) / (x^2 - 1) / ((x^2 - 2*x) / (x + 1)) + 1 / (x - 1)
  expr = -1 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l566_56649


namespace NUMINAMATH_CALUDE_rachel_albums_count_l566_56691

/-- The number of songs per album -/
def songs_per_album : ℕ := 2

/-- The total number of songs Rachel bought -/
def total_songs : ℕ := 16

/-- The number of albums Rachel bought -/
def albums_bought : ℕ := total_songs / songs_per_album

theorem rachel_albums_count : albums_bought = 8 := by
  sorry

end NUMINAMATH_CALUDE_rachel_albums_count_l566_56691


namespace NUMINAMATH_CALUDE_swimmer_speed_proof_l566_56610

def swimmer_problem (distance : ℝ) (current_speed : ℝ) (time : ℝ) : Prop :=
  let still_water_speed := (distance / time) + current_speed
  still_water_speed = 3

theorem swimmer_speed_proof :
  swimmer_problem 8 1.4 5 :=
sorry

end NUMINAMATH_CALUDE_swimmer_speed_proof_l566_56610


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l566_56695

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 30 ∧ x - y = 6 → x * y = 216 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l566_56695


namespace NUMINAMATH_CALUDE_car_speed_comparison_l566_56687

theorem car_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) :
  3 / (1/u + 1/v + 1/w) ≤ (u + v + w) / 3 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l566_56687


namespace NUMINAMATH_CALUDE_boys_who_left_l566_56620

theorem boys_who_left (initial_boys : ℕ) (initial_girls : ℕ) (additional_girls : ℕ) (final_total : ℕ) : 
  initial_boys = 5 →
  initial_girls = 4 →
  additional_girls = 2 →
  final_total = 8 →
  initial_boys - (final_total - (initial_girls + additional_girls)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_boys_who_left_l566_56620


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_l566_56679

theorem polygon_interior_angle_sum (n : ℕ) (interior_angle : ℝ) : 
  n ≥ 3 → 
  interior_angle = 144 → 
  (n - 2) * 180 = n * interior_angle :=
by
  sorry

#check polygon_interior_angle_sum

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_l566_56679


namespace NUMINAMATH_CALUDE_sum_in_base_5_l566_56604

/-- Converts a natural number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- Converts a list representing a number in a given base to base 10 -/
def fromBase (digits : List ℕ) (base : ℕ) : ℕ :=
  sorry

/-- Adds two numbers in a given base -/
def addInBase (a b : List ℕ) (base : ℕ) : List ℕ :=
  sorry

theorem sum_in_base_5 :
  let n1 := 29
  let n2 := 45
  let base4 := toBase n1 4
  let base5 := toBase n2 5
  let sum := addInBase base4 base5 5
  sum = [2, 4, 4] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base_5_l566_56604


namespace NUMINAMATH_CALUDE_cable_on_hand_theorem_l566_56676

/-- Given a total length of cable and a section length, calculates the number of sections. -/
def calculateSections (totalLength sectionLength : ℕ) : ℕ := totalLength / sectionLength

/-- Calculates the number of sections given away. -/
def sectionsGivenAway (totalSections : ℕ) : ℕ := totalSections / 4

/-- Calculates the number of sections remaining after giving some away. -/
def remainingSections (totalSections givenAway : ℕ) : ℕ := totalSections - givenAway

/-- Calculates the number of sections put in storage. -/
def sectionsInStorage (remainingSections : ℕ) : ℕ := remainingSections / 2

/-- Calculates the number of sections kept on hand. -/
def sectionsOnHand (remainingSections inStorage : ℕ) : ℕ := remainingSections - inStorage

/-- Calculates the total length of cable kept on hand. -/
def cableOnHand (sectionsOnHand sectionLength : ℕ) : ℕ := sectionsOnHand * sectionLength

theorem cable_on_hand_theorem (totalLength sectionLength : ℕ) 
    (h1 : totalLength = 1000)
    (h2 : sectionLength = 25) : 
  cableOnHand 
    (sectionsOnHand 
      (remainingSections 
        (calculateSections totalLength sectionLength) 
        (sectionsGivenAway (calculateSections totalLength sectionLength)))
      (sectionsInStorage 
        (remainingSections 
          (calculateSections totalLength sectionLength) 
          (sectionsGivenAway (calculateSections totalLength sectionLength)))))
    sectionLength = 375 := by
  sorry

end NUMINAMATH_CALUDE_cable_on_hand_theorem_l566_56676


namespace NUMINAMATH_CALUDE_apple_tree_problem_l566_56607

/-- The number of apples Rachel picked from the tree -/
def apples_picked : ℝ := 7.5

/-- The number of new apples that grew on the tree after Rachel picked -/
def new_apples : ℝ := 2.3

/-- The number of apples currently on the tree -/
def current_apples : ℝ := 6.2

/-- The original number of apples on the tree -/
def original_apples : ℝ := apples_picked + current_apples - new_apples

theorem apple_tree_problem :
  original_apples = 11.4 := by sorry

end NUMINAMATH_CALUDE_apple_tree_problem_l566_56607


namespace NUMINAMATH_CALUDE_weight_measurement_l566_56677

theorem weight_measurement (n : ℕ) (h : 1 ≤ n ∧ n ≤ 63) :
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ : Bool),
    n = (if a₀ then 1 else 0) +
        (if a₁ then 2 else 0) +
        (if a₂ then 4 else 0) +
        (if a₃ then 8 else 0) +
        (if a₄ then 16 else 0) +
        (if a₅ then 32 else 0) :=
by sorry

end NUMINAMATH_CALUDE_weight_measurement_l566_56677


namespace NUMINAMATH_CALUDE_time_for_b_alone_l566_56638

/-- The time it takes for person B to complete the work alone, given the conditions of the problem. -/
theorem time_for_b_alone (a b c : ℝ) : 
  a = 1/3 →  -- A can do the work in 3 hours
  b + c = 1/3 →  -- B and C together can do it in 3 hours
  a + c = 1/2 →  -- A and C together can do it in 2 hours
  1/b = 6 :=  -- B alone takes 6 hours
by sorry


end NUMINAMATH_CALUDE_time_for_b_alone_l566_56638


namespace NUMINAMATH_CALUDE_total_campers_rowing_l566_56627

theorem total_campers_rowing (morning_campers afternoon_campers : ℕ) 
  (h1 : morning_campers = 53)
  (h2 : afternoon_campers = 7) :
  morning_campers + afternoon_campers = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_rowing_l566_56627


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l566_56689

theorem last_two_digits_sum (n : ℕ) : n = 30 → (7^n + 13^n) % 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l566_56689


namespace NUMINAMATH_CALUDE_mod_seven_equivalence_l566_56605

theorem mod_seven_equivalence : 47^1357 - 23^1357 ≡ 3 [ZMOD 7] := by sorry

end NUMINAMATH_CALUDE_mod_seven_equivalence_l566_56605


namespace NUMINAMATH_CALUDE_antiderivative_of_f_l566_56630

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x

-- Define the antiderivative F
def F (x : ℝ) : ℝ := x^2 + 2

-- Theorem statement
theorem antiderivative_of_f (x : ℝ) : 
  (deriv F x = f x) ∧ (F 1 = 3) := by sorry

end NUMINAMATH_CALUDE_antiderivative_of_f_l566_56630


namespace NUMINAMATH_CALUDE_product_103_97_l566_56609

theorem product_103_97 : 103 * 97 = 9991 := by
  sorry

end NUMINAMATH_CALUDE_product_103_97_l566_56609


namespace NUMINAMATH_CALUDE_nested_bracket_equals_two_l566_56628

def bracket (x y z : ℚ) : ℚ := (x + y) / z

theorem nested_bracket_equals_two :
  bracket (bracket 45 15 60) (bracket 3 3 6) (bracket 24 6 30) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_bracket_equals_two_l566_56628


namespace NUMINAMATH_CALUDE_alloy_mixture_specific_alloy_mixture_l566_56670

/-- Given two alloys with different chromium percentages, prove the amount of the second alloy
    needed to create a new alloy with a specific chromium percentage. -/
theorem alloy_mixture (first_alloy_chromium_percent : ℝ) 
                      (second_alloy_chromium_percent : ℝ)
                      (new_alloy_chromium_percent : ℝ)
                      (first_alloy_amount : ℝ) : ℝ :=
  let second_alloy_amount := 
    (new_alloy_chromium_percent * first_alloy_amount - first_alloy_chromium_percent * first_alloy_amount) /
    (second_alloy_chromium_percent - new_alloy_chromium_percent)
  second_alloy_amount

/-- Prove that 35 kg of the second alloy is needed to create the new alloy with 8.6% chromium. -/
theorem specific_alloy_mixture : 
  alloy_mixture 0.10 0.08 0.086 15 = 35 := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixture_specific_alloy_mixture_l566_56670


namespace NUMINAMATH_CALUDE_altitude_and_angle_bisector_equations_l566_56636

/-- Triangle ABC with vertices A(1,-1), B(-1,3), C(3,0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The given triangle -/
def ABC : Triangle :=
  { A := (1, -1),
    B := (-1, 3),
    C := (3, 0) }

/-- Altitude from A to BC -/
def altitude (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => 4 * p.1 - 3 * p.2 - 7 = 0

/-- Angle bisector of ∠BAC -/
def angle_bisector (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => p.1 - p.2 - 4 = 0

/-- Main theorem -/
theorem altitude_and_angle_bisector_equations :
  (∀ p, altitude ABC p ↔ 4 * p.1 - 3 * p.2 - 7 = 0) ∧
  (∀ p, angle_bisector ABC p ↔ p.1 - p.2 - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_altitude_and_angle_bisector_equations_l566_56636


namespace NUMINAMATH_CALUDE_max_discount_percentage_l566_56633

/-- The maximum discount percentage that can be applied to a product while maintaining a minimum profit margin. -/
theorem max_discount_percentage
  (cost : ℝ)              -- Cost price in yuan
  (price : ℝ)             -- Selling price in yuan
  (min_margin : ℝ)        -- Minimum profit margin as a decimal
  (h_cost : cost = 100)   -- Cost is 100 yuan
  (h_price : price = 150) -- Price is 150 yuan
  (h_margin : min_margin = 0.2) -- Minimum margin is 20%
  : ∃ (max_discount : ℝ),
    max_discount = 20 ∧
    ∀ (discount : ℝ),
      0 ≤ discount ∧ discount ≤ max_discount →
      (price * (1 - discount / 100) - cost) / cost ≥ min_margin :=
by sorry

end NUMINAMATH_CALUDE_max_discount_percentage_l566_56633


namespace NUMINAMATH_CALUDE_quiche_volume_l566_56666

/-- Calculate the total volume of a vegetable quiche --/
theorem quiche_volume 
  (spinach_initial : ℝ) 
  (mushrooms_initial : ℝ) 
  (onions_initial : ℝ)
  (spinach_reduction : ℝ) 
  (mushrooms_reduction : ℝ) 
  (onions_reduction : ℝ)
  (cream_cheese : ℝ)
  (eggs : ℝ)
  (h1 : spinach_initial = 40)
  (h2 : mushrooms_initial = 25)
  (h3 : onions_initial = 15)
  (h4 : spinach_reduction = 0.20)
  (h5 : mushrooms_reduction = 0.65)
  (h6 : onions_reduction = 0.50)
  (h7 : cream_cheese = 6)
  (h8 : eggs = 4) :
  spinach_initial * spinach_reduction + 
  mushrooms_initial * mushrooms_reduction + 
  onions_initial * onions_reduction + 
  cream_cheese + eggs = 41.75 := by
sorry

end NUMINAMATH_CALUDE_quiche_volume_l566_56666


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l566_56626

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (4, -2)
  let b : ℝ × ℝ := (x, 5)
  parallel a b → x = -10 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l566_56626


namespace NUMINAMATH_CALUDE_ellipse_condition_l566_56655

/-- An ellipse equation with parameter k -/
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (9 - k) + y^2 / (k - 4) = 1

/-- The condition 4 < k < 9 -/
def condition (k : ℝ) : Prop := 4 < k ∧ k < 9

/-- The statement to be proven -/
theorem ellipse_condition :
  (∀ k, is_ellipse k → condition k) ∧
  ¬(∀ k, condition k → is_ellipse k) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l566_56655


namespace NUMINAMATH_CALUDE_abc_inequality_l566_56686

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a + b + c + a * b + b * c + c * a ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l566_56686


namespace NUMINAMATH_CALUDE_cost_of_apples_and_oranges_l566_56657

/-- The cost of apples and oranges given the initial amount and remaining amount -/
def cost_of_fruits (initial_amount remaining_amount : ℚ) : ℚ :=
  initial_amount - remaining_amount

/-- Theorem: The cost of apples and oranges is $15.00 -/
theorem cost_of_apples_and_oranges :
  cost_of_fruits 100 85 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_apples_and_oranges_l566_56657


namespace NUMINAMATH_CALUDE_circle_radius_problem_l566_56697

theorem circle_radius_problem (r : ℝ) (h : r > 0) :
  3 * (2 * 2 * Real.pi * r) = 3 * (Real.pi * r ^ 2) → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l566_56697


namespace NUMINAMATH_CALUDE_x_lt_5_necessary_not_sufficient_for_x_lt_2_l566_56632

theorem x_lt_5_necessary_not_sufficient_for_x_lt_2 :
  (∀ x : ℝ, x < 2 → x < 5) ∧ (∃ x : ℝ, x < 5 ∧ ¬(x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_x_lt_5_necessary_not_sufficient_for_x_lt_2_l566_56632


namespace NUMINAMATH_CALUDE_rhombus_area_l566_56640

/-- The area of a rhombus with vertices at (0, 3.5), (12, 0), (0, -3.5), and (-12, 0) is 84 square units. -/
theorem rhombus_area : 
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (12, 0), (0, -3.5), (-12, 0)]
  let diagonal1 : ℝ := |3.5 - (-3.5)|
  let diagonal2 : ℝ := |12 - (-12)|
  let area : ℝ := (diagonal1 * diagonal2) / 2
  area = 84 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l566_56640


namespace NUMINAMATH_CALUDE_substance_mass_l566_56619

/-- Given a substance where 1 gram occupies 5 cubic centimeters, 
    the mass of 1 cubic meter of this substance is 200 kilograms. -/
theorem substance_mass (substance_density : ℝ) : 
  substance_density = 1 / 5 → -- 1 gram occupies 5 cubic centimeters
  (1 : ℝ) * substance_density * 1000000 / 1000 = 200 := by
  sorry

end NUMINAMATH_CALUDE_substance_mass_l566_56619


namespace NUMINAMATH_CALUDE_disjoint_equal_sum_subsets_l566_56663

theorem disjoint_equal_sum_subsets (S : Finset ℕ) 
  (h1 : S ⊆ Finset.range 2018)
  (h2 : S.card = 68) :
  ∃ (A B C : Finset ℕ), 
    A ⊆ S ∧ B ⊆ S ∧ C ⊆ S ∧
    A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
    A.card = B.card ∧ B.card = C.card ∧
    A.sum id = B.sum id ∧ B.sum id = C.sum id :=
by sorry

end NUMINAMATH_CALUDE_disjoint_equal_sum_subsets_l566_56663


namespace NUMINAMATH_CALUDE_solution_set_f_neg_x_l566_56660

/-- Given a function f(x) = (ax-1)(x-b) where the solution set of f(x) > 0 is (-1,3),
    prove that the solution set of f(-x) < 0 is (-∞,-3)∪(1,+∞) -/
theorem solution_set_f_neg_x (a b : ℝ) : 
  (∀ x, (a * x - 1) * (x - b) > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x, (a * (-x) - 1) * (-x - b) < 0 ↔ x < -3 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_f_neg_x_l566_56660


namespace NUMINAMATH_CALUDE_angle_symmetry_l566_56603

theorem angle_symmetry (α : Real) : 
  (∃ k : ℤ, α = π/3 + 2*k*π) →  -- Condition 1 (symmetry implies α = π/3 + 2kπ)
  α ∈ Set.Ioo (-4*π) (-2*π) →   -- Condition 2
  (α = -11*π/3 ∨ α = -5*π/3) :=
by sorry

end NUMINAMATH_CALUDE_angle_symmetry_l566_56603


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_sum_and_11_l566_56654

/-- Represents a three-digit integer -/
structure ThreeDigitInteger where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value ≤ 999

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem largest_three_digit_divisible_by_sum_and_11 :
  ∃ (n : ThreeDigitInteger),
    (n.value % sum_of_digits n.value = 0) ∧
    (sum_of_digits n.value % 11 = 0) ∧
    (∀ (m : ThreeDigitInteger),
      (m.value % sum_of_digits m.value = 0) ∧
      (sum_of_digits m.value % 11 = 0) →
      m.value ≤ n.value) ∧
    n.value = 990 :=
  sorry


end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_sum_and_11_l566_56654


namespace NUMINAMATH_CALUDE_sequence_general_term_l566_56646

theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, S n = 2^n + 3) →
  (a 1 = 5 ∧ ∀ n : ℕ, n ≥ 2 → a n = 2^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l566_56646


namespace NUMINAMATH_CALUDE_factory_earnings_l566_56682

-- Define the parameters
def hours_machines_123 : ℕ := 23
def hours_machine_4 : ℕ := 12
def production_rate_12 : ℕ := 2
def production_rate_34 : ℕ := 3
def price_13 : ℕ := 50
def price_24 : ℕ := 60

-- Define the earnings calculation function
def calculate_earnings (hours : ℕ) (rate : ℕ) (price : ℕ) : ℕ :=
  hours * rate * price

-- Theorem statement
theorem factory_earnings :
  calculate_earnings hours_machines_123 production_rate_12 price_13 +
  calculate_earnings hours_machines_123 production_rate_12 price_24 +
  calculate_earnings hours_machines_123 production_rate_34 price_13 +
  calculate_earnings hours_machine_4 production_rate_34 price_24 = 10670 := by
  sorry


end NUMINAMATH_CALUDE_factory_earnings_l566_56682


namespace NUMINAMATH_CALUDE_three_primes_sum_47_product_1705_l566_56644

theorem three_primes_sum_47_product_1705 : ∃ p q r : ℕ, 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p + q + r = 47 ∧ 
  p * q * r = 1705 := by
sorry

end NUMINAMATH_CALUDE_three_primes_sum_47_product_1705_l566_56644


namespace NUMINAMATH_CALUDE_four_digit_number_satisfies_condition_l566_56600

/-- Represents a four-digit number -/
def FourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- Splits a four-digit number into two two-digit numbers -/
def SplitNumber (n : ℕ) : ℕ × ℕ :=
  (n / 100, n % 100)

/-- Checks if a number satisfies the given condition -/
def SatisfiesCondition (n : ℕ) : Prop :=
  let (a, b) := SplitNumber n
  (10 * a + b / 10) * (b % 10 + 10 * (b / 10)) + 10 * a = n

theorem four_digit_number_satisfies_condition :
  FourDigitNumber 1995 ∧
  (SplitNumber 1995).2 % 10 = 5 ∧
  SatisfiesCondition 1995 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_satisfies_condition_l566_56600


namespace NUMINAMATH_CALUDE_smallest_leading_coeff_quadratic_roots_existence_quadratic_roots_five_smallest_leading_coeff_is_five_l566_56621

theorem smallest_leading_coeff_quadratic_roots (a : ℕ) : 
  (∃ (b c : ℤ) (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    0 < x₁ ∧ x₁ < 1 ∧ 
    0 < x₂ ∧ x₂ < 1 ∧ 
    ∀ (x : ℝ), (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = x₁ ∨ x = x₂) →
  a ≥ 5 :=
by sorry

theorem existence_quadratic_roots_five :
  ∃ (b c : ℤ) (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    0 < x₁ ∧ x₁ < 1 ∧ 
    0 < x₂ ∧ x₂ < 1 ∧ 
    ∀ (x : ℝ), (5 : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = x₁ ∨ x = x₂ :=
by sorry

theorem smallest_leading_coeff_is_five : 
  ∀ (a : ℕ), 
    (∃ (b c : ℤ) (x₁ x₂ : ℝ), 
      x₁ ≠ x₂ ∧ 
      0 < x₁ ∧ x₁ < 1 ∧ 
      0 < x₂ ∧ x₂ < 1 ∧ 
      ∀ (x : ℝ), (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = x₁ ∨ x = x₂) →
    a ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_leading_coeff_quadratic_roots_existence_quadratic_roots_five_smallest_leading_coeff_is_five_l566_56621


namespace NUMINAMATH_CALUDE_max_answered_A_l566_56639

/-- Represents the number of people who answered each combination of questions correctly. -/
structure Answers :=
  (a : ℕ)  -- Only A
  (b : ℕ)  -- Only B
  (c : ℕ)  -- Only C
  (ab : ℕ) -- A and B
  (ac : ℕ) -- A and C
  (bc : ℕ) -- B and C
  (abc : ℕ) -- All three

/-- The conditions of the math competition problem. -/
def ValidAnswers (ans : Answers) : Prop :=
  -- Total participants
  ans.a + ans.b + ans.c + ans.ab + ans.ac + ans.bc + ans.abc = 39 ∧
  -- Condition about A answers
  ans.a = ans.ab + ans.ac + ans.abc + 5 ∧
  -- Condition about B and C answers (not A)
  ans.b + ans.bc = 2 * (ans.c + ans.bc) ∧
  -- Condition about only A, B, and C answers
  ans.a = ans.b + ans.c

/-- The number of people who answered A correctly. -/
def AnsweredA (ans : Answers) : ℕ :=
  ans.a + ans.ab + ans.ac + ans.abc

/-- The theorem stating the maximum number of people who answered A correctly. -/
theorem max_answered_A :
  ∀ ans : Answers, ValidAnswers ans → AnsweredA ans ≤ 23 :=
sorry

end NUMINAMATH_CALUDE_max_answered_A_l566_56639


namespace NUMINAMATH_CALUDE_impossible_to_empty_heap_l566_56656

/-- Represents the state of the three heaps of stones -/
structure HeapState :=
  (heap1 : Nat) (heap2 : Nat) (heap3 : Nat)

/-- Defines the allowed operations on the heaps -/
inductive Operation
  | Add (target : Nat) (source1 : Nat) (source2 : Nat)
  | Remove (target : Nat) (source1 : Nat) (source2 : Nat)

/-- Applies an operation to a heap state -/
def applyOperation (state : HeapState) (op : Operation) : HeapState :=
  match op with
  | Operation.Add 0 1 2 => HeapState.mk (state.heap1 + state.heap2 + state.heap3) state.heap2 state.heap3
  | Operation.Add 1 0 2 => HeapState.mk state.heap1 (state.heap2 + state.heap1 + state.heap3) state.heap3
  | Operation.Add 2 0 1 => HeapState.mk state.heap1 state.heap2 (state.heap3 + state.heap1 + state.heap2)
  | Operation.Remove 0 1 2 => HeapState.mk (state.heap1 - state.heap2 - state.heap3) state.heap2 state.heap3
  | Operation.Remove 1 0 2 => HeapState.mk state.heap1 (state.heap2 - state.heap1 - state.heap3) state.heap3
  | Operation.Remove 2 0 1 => HeapState.mk state.heap1 state.heap2 (state.heap3 - state.heap1 - state.heap2)
  | _ => state  -- Invalid operations return the original state

/-- Defines the initial state of the heaps -/
def initialState : HeapState := HeapState.mk 1993 199 19

/-- Theorem stating that it's impossible to make a heap empty -/
theorem impossible_to_empty_heap :
  ∀ (operations : List Operation),
    let finalState := operations.foldl applyOperation initialState
    ¬(finalState.heap1 = 0 ∨ finalState.heap2 = 0 ∨ finalState.heap3 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_impossible_to_empty_heap_l566_56656
