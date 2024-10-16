import Mathlib

namespace NUMINAMATH_CALUDE_bicycle_license_combinations_l2262_226294

def license_letter : Nat := 2  -- B or C
def license_digits : Nat := 6
def free_digit_positions : Nat := license_digits - 1  -- All but the last digit
def digits_per_position : Nat := 10  -- 0 to 9
def last_digit : Nat := 1  -- Only 5 is allowed

theorem bicycle_license_combinations :
  license_letter * digits_per_position ^ free_digit_positions * last_digit = 200000 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_license_combinations_l2262_226294


namespace NUMINAMATH_CALUDE_smallest_whole_number_greater_than_50_with_odd_factors_l2262_226285

def has_odd_number_of_factors (n : ℕ) : Prop :=
  Odd (Finset.card (Finset.filter (· ∣ n) (Finset.range (n + 1))))

theorem smallest_whole_number_greater_than_50_with_odd_factors : 
  ∀ n : ℕ, n > 50 → has_odd_number_of_factors n → n ≥ 64 :=
by
  sorry

#check smallest_whole_number_greater_than_50_with_odd_factors

end NUMINAMATH_CALUDE_smallest_whole_number_greater_than_50_with_odd_factors_l2262_226285


namespace NUMINAMATH_CALUDE_martine_has_sixteen_peaches_l2262_226281

/-- Given the number of peaches Gabrielle has -/
def gabrielle_peaches : ℕ := 15

/-- Benjy's peaches in terms of Gabrielle's -/
def benjy_peaches : ℕ := gabrielle_peaches / 3

/-- Martine's peaches in terms of Benjy's -/
def martine_peaches : ℕ := 2 * benjy_peaches + 6

/-- Theorem: Martine has 16 peaches -/
theorem martine_has_sixteen_peaches : martine_peaches = 16 := by
  sorry

end NUMINAMATH_CALUDE_martine_has_sixteen_peaches_l2262_226281


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l2262_226241

theorem cubic_expression_evaluation (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 9 = 73/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l2262_226241


namespace NUMINAMATH_CALUDE_pumpkin_multiple_l2262_226212

theorem pumpkin_multiple (moonglow sunshine : ℕ) (h1 : moonglow = 14) (h2 : sunshine = 54) :
  ∃ x : ℕ, x * moonglow + 12 = sunshine ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_multiple_l2262_226212


namespace NUMINAMATH_CALUDE_max_intersections_circle_ellipse_triangle_l2262_226227

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- An ellipse in a 2D plane -/
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ -- semi-major axis
  b : ℝ -- semi-minor axis

/-- A triangle in a 2D plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The maximum number of intersection points between a circle and a triangle -/
def max_intersections_circle_triangle : ℕ := 6

/-- The maximum number of intersection points between an ellipse and a triangle -/
def max_intersections_ellipse_triangle : ℕ := 6

/-- The maximum number of intersection points between a circle and an ellipse -/
def max_intersections_circle_ellipse : ℕ := 4

/-- Theorem: The maximum number of intersection points among a circle, an ellipse, and a triangle is 16 -/
theorem max_intersections_circle_ellipse_triangle :
  ∀ (c : Circle) (e : Ellipse) (t : Triangle),
  max_intersections_circle_triangle +
  max_intersections_ellipse_triangle +
  max_intersections_circle_ellipse = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_max_intersections_circle_ellipse_triangle_l2262_226227


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l2262_226268

/-- Represents a trapezoid ABCD with sides AB and CD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ

/-- The theorem statement -/
theorem trapezoid_segment_length (t : Trapezoid) :
  (t.AB / t.CD = 3) →  -- Area ratio implies base ratio
  (t.AB + t.CD = 320) →
  t.AB = 240 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l2262_226268


namespace NUMINAMATH_CALUDE_transformations_correct_l2262_226224

theorem transformations_correct (a b c : ℝ) (h1 : a = b) (h2 : c ≠ 0) (h3 : a / c = b / c) (h4 : -2 * a = -2 * b) : 
  (a + 6 = b + 6) ∧ 
  (a / 9 = b / 9) ∧ 
  (a = b) ∧ 
  (a = b) := by
  sorry

end NUMINAMATH_CALUDE_transformations_correct_l2262_226224


namespace NUMINAMATH_CALUDE_largest_divisor_is_60_l2262_226275

def is_largest_divisor (n : ℕ) : Prop :=
  n ∣ 540 ∧ n < 80 ∧ n ∣ 180 ∧
  ∀ m : ℕ, m ∣ 540 → m < 80 → m ∣ 180 → m ≤ n

theorem largest_divisor_is_60 : is_largest_divisor 60 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_is_60_l2262_226275


namespace NUMINAMATH_CALUDE_three_digit_square_last_three_l2262_226282

theorem three_digit_square_last_three (n : ℕ) : 
  (100 ≤ n ∧ n < 1000) → (n = n^2 % 1000 ↔ n = 376 ∨ n = 625) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_square_last_three_l2262_226282


namespace NUMINAMATH_CALUDE_sine_inequality_in_acute_triangle_l2262_226288

theorem sine_inequality_in_acute_triangle (A B C : Real) 
  (triangle_condition : A ≤ B ∧ B ≤ C ∧ C < Real.pi / 2) : 
  Real.sin (2 * A) ≥ Real.sin (2 * B) ∧ Real.sin (2 * B) ≥ Real.sin (2 * C) := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_in_acute_triangle_l2262_226288


namespace NUMINAMATH_CALUDE_stratified_sampling_first_year_l2262_226233

theorem stratified_sampling_first_year
  (total_sample : ℕ)
  (first_year_ratio second_year_ratio third_year_ratio : ℕ)
  (h_total_sample : total_sample = 56)
  (h_ratios : first_year_ratio = 7 ∧ second_year_ratio = 3 ∧ third_year_ratio = 4) :
  (total_sample * first_year_ratio) / (first_year_ratio + second_year_ratio + third_year_ratio) = 28 := by
  sorry

#check stratified_sampling_first_year

end NUMINAMATH_CALUDE_stratified_sampling_first_year_l2262_226233


namespace NUMINAMATH_CALUDE_line_increase_percentage_l2262_226270

/-- Given that increasing the number of lines by 60 results in 240 lines,
    prove that the percentage increase is 100/3%. -/
theorem line_increase_percentage : ℝ → Prop :=
  fun original_lines =>
    (original_lines + 60 = 240) →
    ((60 / original_lines) * 100 = 100 / 3)

/-- Proof of the theorem -/
lemma prove_line_increase_percentage : ∃ x : ℝ, line_increase_percentage x := by
  sorry

end NUMINAMATH_CALUDE_line_increase_percentage_l2262_226270


namespace NUMINAMATH_CALUDE_sector_central_angle_l2262_226251

/-- Given a circular sector with circumference 10 and area 4, 
    prove that its central angle in radians is 1/2 -/
theorem sector_central_angle (r l : ℝ) : 
  (2 * r + l = 10) →  -- circumference condition
  ((1 / 2) * l * r = 4) →  -- area condition
  (l / r = 1 / 2) :=  -- central angle in radians
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2262_226251


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2262_226219

theorem rectangle_perimeter (a b : ℕ) : 
  a ≠ b → -- non-square condition
  a * b = 3 * (2 * a + 2 * b) → -- area equals 3 times perimeter
  2 * a + 2 * b = 36 ∨ 2 * a + 2 * b = 28 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2262_226219


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l2262_226264

theorem more_girls_than_boys (total_students : ℕ) (boys_ratio girls_ratio : ℕ) : 
  total_students = 42 →
  boys_ratio = 3 →
  girls_ratio = 4 →
  (girls_ratio - boys_ratio) * (total_students / (boys_ratio + girls_ratio)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l2262_226264


namespace NUMINAMATH_CALUDE_hostel_mess_expenditure_l2262_226254

/-- The original expenditure of a hostel mess given specific conditions -/
theorem hostel_mess_expenditure :
  ∀ (initial_students : ℕ) 
    (student_increase : ℕ) 
    (expense_increase : ℕ) 
    (avg_expenditure_decrease : ℕ),
  initial_students = 35 →
  student_increase = 7 →
  expense_increase = 42 →
  avg_expenditure_decrease = 1 →
  ∃ (original_expenditure : ℕ),
    original_expenditure = 420 ∧
    (initial_students + student_increase) * 
      ((original_expenditure / initial_students) - avg_expenditure_decrease) =
    original_expenditure + expense_increase :=
by sorry

end NUMINAMATH_CALUDE_hostel_mess_expenditure_l2262_226254


namespace NUMINAMATH_CALUDE_journey_speed_proof_l2262_226274

/-- Proves that given a journey of 300 km completed in 11 hours, where the second half of the distance 
    is traveled at 25 kmph, the speed for the first half of the journey is 30 kmph. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) 
  (h1 : total_distance = 300)
  (h2 : total_time = 11)
  (h3 : second_half_speed = 25) : 
  (total_distance / 2) / (total_time - (total_distance / 2) / second_half_speed) = 30 :=
by
  sorry

#check journey_speed_proof

end NUMINAMATH_CALUDE_journey_speed_proof_l2262_226274


namespace NUMINAMATH_CALUDE_phase_shift_sin_5x_minus_pi_half_l2262_226261

/-- The phase shift of the function y = sin(5x - π/2) is π/10 to the right or -π/10 to the left -/
theorem phase_shift_sin_5x_minus_pi_half :
  let f : ℝ → ℝ := λ x => Real.sin (5 * x - π / 2)
  ∃ φ : ℝ, (φ = π / 10 ∨ φ = -π / 10) ∧
    ∀ x : ℝ, f x = Real.sin (5 * (x - φ)) :=
by sorry

end NUMINAMATH_CALUDE_phase_shift_sin_5x_minus_pi_half_l2262_226261


namespace NUMINAMATH_CALUDE_medical_team_probability_l2262_226260

theorem medical_team_probability (male_doctors female_doctors team_size : ℕ) 
  (h1 : male_doctors = 6)
  (h2 : female_doctors = 3)
  (h3 : team_size = 5) : 
  (1 - (Nat.choose male_doctors team_size : ℚ) / (Nat.choose (male_doctors + female_doctors) team_size)) = 60/63 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_probability_l2262_226260


namespace NUMINAMATH_CALUDE_f_has_six_zeros_l2262_226208

noncomputable def f (x : ℝ) : ℝ :=
  (1 + x - x^2/2 + x^3/3 - x^4/4 - x^2018/2018 + x^2019/2019) * Real.cos (2*x)

theorem f_has_six_zeros :
  ∃ (S : Finset ℝ), S.card = 6 ∧ 
  (∀ x ∈ S, x ∈ Set.Icc (-3) 4 ∧ f x = 0) ∧
  (∀ x ∈ Set.Icc (-3) 4, f x = 0 → x ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_f_has_six_zeros_l2262_226208


namespace NUMINAMATH_CALUDE_orange_juice_count_l2262_226248

/-- The number of bottles of orange juice bought -/
def orange_juice_bottles : ℕ := sorry

/-- The number of bottles of apple juice bought -/
def apple_juice_bottles : ℕ := sorry

/-- The cost of a bottle of orange juice in cents -/
def orange_juice_cost : ℕ := 70

/-- The cost of a bottle of apple juice in cents -/
def apple_juice_cost : ℕ := 60

/-- The total number of bottles bought -/
def total_bottles : ℕ := 70

/-- The total cost of all bottles in cents -/
def total_cost : ℕ := 4620

theorem orange_juice_count : 
  orange_juice_bottles = 42 ∧
  orange_juice_bottles + apple_juice_bottles = total_bottles ∧
  orange_juice_bottles * orange_juice_cost + apple_juice_bottles * apple_juice_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_orange_juice_count_l2262_226248


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l2262_226249

/-- For a hyperbola with equation x^2/a^2 - y^2/b^2 = 1, where a > b and the angle between
    the asymptotes is 30°, the ratio a/b = 2 - √3. -/
theorem hyperbola_asymptote_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.pi / 6 = Real.arctan ((2 * b / a) / (1 - (b / a)^2))) →
  a / b = 2 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l2262_226249


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l2262_226229

theorem only_one_divides_power_minus_one : 
  ∀ n : ℕ, n > 0 → n ∣ (2^n - 1) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l2262_226229


namespace NUMINAMATH_CALUDE_flea_can_reach_all_points_l2262_226236

/-- The length of the k-th jump for the flea -/
def jumpLength (k : ℕ) : ℕ := 2^k + 1

/-- A jump is represented by its length and direction -/
structure Jump where
  length : ℕ
  direction : Bool  -- true for right, false for left

/-- The final position after a sequence of jumps -/
def finalPosition (jumps : List Jump) : ℤ :=
  jumps.foldl (fun pos jump => 
    if jump.direction then pos + jump.length else pos - jump.length) 0

/-- Theorem: For any natural number n, there exists a sequence of jumps
    that allows the flea to move from point 0 to point n -/
theorem flea_can_reach_all_points (n : ℕ) : 
  ∃ (jumps : List Jump), finalPosition jumps = n := by
  sorry

end NUMINAMATH_CALUDE_flea_can_reach_all_points_l2262_226236


namespace NUMINAMATH_CALUDE_wall_washing_problem_l2262_226287

theorem wall_washing_problem (boys_5 boys_7 : ℕ) (wall_5 wall_7 : ℝ) (days : ℕ) :
  boys_5 = 5 →
  boys_7 = 7 →
  wall_5 = 25 →
  days = 4 →
  (boys_5 : ℝ) * wall_5 * (boys_7 : ℝ) = boys_7 * wall_7 * (boys_5 : ℝ) →
  wall_7 = 35 := by
sorry

end NUMINAMATH_CALUDE_wall_washing_problem_l2262_226287


namespace NUMINAMATH_CALUDE_original_number_reciprocal_l2262_226272

theorem original_number_reciprocal (x : ℝ) : 1 / x - 3 = 5 / 2 → x = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_original_number_reciprocal_l2262_226272


namespace NUMINAMATH_CALUDE_triangle_trig_inequality_triangle_trig_equality_l2262_226271

/-- For any triangle ABC, sin A + sin B sin C + cos B cos C ≤ 2 -/
theorem triangle_trig_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A + Real.sin B * Real.sin C + Real.cos B * Real.cos C ≤ 2 :=
sorry

/-- The equality holds when A = π/2 and B = C = π/4 -/
theorem triangle_trig_equality :
  Real.sin (Real.pi/2) + Real.sin (Real.pi/4) * Real.sin (Real.pi/4) + 
  Real.cos (Real.pi/4) * Real.cos (Real.pi/4) = 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_trig_inequality_triangle_trig_equality_l2262_226271


namespace NUMINAMATH_CALUDE_pages_in_harrys_book_l2262_226210

/-- Given that Selena's book has x pages and Harry's book has y fewer pages than half the number
    of pages of Selena's book, prove that the number of pages in Harry's book is equal to (x/2) - y. -/
theorem pages_in_harrys_book (x y : ℕ) : ℕ :=
  x / 2 - y

#check pages_in_harrys_book

end NUMINAMATH_CALUDE_pages_in_harrys_book_l2262_226210


namespace NUMINAMATH_CALUDE_quadratic_equation_linear_coefficient_l2262_226237

/-- The exponent of x in the first term of the equation -/
def exponent (m : ℝ) : ℝ := m^2 - 2*m - 1

/-- The equation is quadratic when the exponent equals 2 -/
def is_quadratic (m : ℝ) : Prop := exponent m = 2

/-- The coefficient of x in the equation -/
def linear_coefficient (m : ℝ) : ℝ := -m

theorem quadratic_equation_linear_coefficient :
  ∀ m : ℝ, (m ≠ 3) → is_quadratic m → linear_coefficient m = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_linear_coefficient_l2262_226237


namespace NUMINAMATH_CALUDE_expression_simplification_l2262_226235

theorem expression_simplification 
  (p q : ℝ) (x : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) :
  (x^(3/p) - x^(3/q)) / ((x^(1/p) + x^(1/q))^2 - 2*x^(1/q)*(x^(1/q) + x^(1/p))) + 
  x^(1/p) / (x^((q-p)/(p*q)) + 1) = x^(1/p) + x^(1/q) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2262_226235


namespace NUMINAMATH_CALUDE_cos_sin_shift_l2262_226225

theorem cos_sin_shift (x : ℝ) : 
  Real.cos (x + 2 * Real.pi / 3) = Real.sin (Real.pi / 3 - (x + Real.pi / 2)) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_shift_l2262_226225


namespace NUMINAMATH_CALUDE_ratio_problem_l2262_226286

theorem ratio_problem (x y : ℤ) : 
  (y = 3 * x) → -- The two integers are in the ratio of 1 to 3
  (x + 10 = y) → -- Adding 10 to the smaller number makes them equal
  y = 15 := by -- The larger integer is 15
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2262_226286


namespace NUMINAMATH_CALUDE_max_value_P_l2262_226278

open Real

/-- Given positive real numbers a, b, and c satisfying abc + a + c = b,
    the maximum value of P = 2/(a² + 1) - 2/(b² + 1) + 3/(c² + 1) is 1. -/
theorem max_value_P (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_eq : a * b * c + a + c = b) :
  ∃ (M : ℝ), M = 1 ∧ ∀ x y z, 0 < x → 0 < y → 0 < z → x * y * z + x + z = y →
    2 / (x^2 + 1) - 2 / (y^2 + 1) + 3 / (z^2 + 1) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_P_l2262_226278


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l2262_226215

theorem sqrt_50_between_consecutive_integers_product (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ b = a + 1 ∧ (a : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (b : ℝ) → a * b = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l2262_226215


namespace NUMINAMATH_CALUDE_square_of_two_power_minus_twice_l2262_226259

theorem square_of_two_power_minus_twice (N : ℕ+) :
  (∃ k : ℕ, 2^N.val - 2 * N.val = k^2) ↔ N = 1 ∨ N = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_two_power_minus_twice_l2262_226259


namespace NUMINAMATH_CALUDE_min_workers_for_profit_l2262_226293

/-- Represents the company's daily operations and profit calculation -/
structure Company where
  maintenance_fee : ℕ := 600
  hourly_wage : ℕ := 20
  widgets_per_hour : ℕ := 6
  widget_price : ℚ := 7/2
  work_hours : ℕ := 8

/-- Calculates whether the company is profitable given a number of workers -/
def is_profitable (c : Company) (workers : ℕ) : Prop :=
  (c.widgets_per_hour * c.work_hours * c.widget_price : ℚ) * workers >
  (c.maintenance_fee : ℚ) + (c.hourly_wage * c.work_hours : ℚ) * workers

/-- Theorem stating the minimum number of workers needed for profitability -/
theorem min_workers_for_profit (c : Company) :
  ∀ n : ℕ, is_profitable c n ↔ n ≥ 76 :=
sorry

end NUMINAMATH_CALUDE_min_workers_for_profit_l2262_226293


namespace NUMINAMATH_CALUDE_simplify_rational_expression_l2262_226267

theorem simplify_rational_expression (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 4) * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_rational_expression_l2262_226267


namespace NUMINAMATH_CALUDE_one_time_cost_correct_l2262_226202

/-- The one-time product cost for editing and printing --/
def one_time_cost : ℝ := 56430

/-- The variable cost per book --/
def variable_cost : ℝ := 8.25

/-- The selling price per book --/
def selling_price : ℝ := 21.75

/-- The number of books at the break-even point --/
def break_even_books : ℕ := 4180

/-- Theorem stating that the one-time cost is correct given the conditions --/
theorem one_time_cost_correct :
  one_time_cost = (selling_price - variable_cost) * break_even_books :=
by sorry

end NUMINAMATH_CALUDE_one_time_cost_correct_l2262_226202


namespace NUMINAMATH_CALUDE_a_gt_b_iff_a_plus_ln_a_gt_b_plus_ln_b_l2262_226217

theorem a_gt_b_iff_a_plus_ln_a_gt_b_plus_ln_b 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a > b) ↔ (a + Real.log a > b + Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_iff_a_plus_ln_a_gt_b_plus_ln_b_l2262_226217


namespace NUMINAMATH_CALUDE_gross_revenue_increase_l2262_226298

theorem gross_revenue_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_reduction_rate : ℝ)
  (quantity_increase_rate : ℝ)
  (h1 : price_reduction_rate = 0.2)
  (h2 : quantity_increase_rate = 0.6)
  : (((1 - price_reduction_rate) * (1 + quantity_increase_rate) - 1) * 100 = 28) :=
by sorry

end NUMINAMATH_CALUDE_gross_revenue_increase_l2262_226298


namespace NUMINAMATH_CALUDE_old_manufacturing_cost_l2262_226258

/-- Proves that the old manufacturing cost was $65 given the conditions of the problem -/
theorem old_manufacturing_cost (selling_price : ℝ) (new_manufacturing_cost : ℝ) : 
  selling_price = 100 →
  new_manufacturing_cost = 50 →
  (selling_price - new_manufacturing_cost) / selling_price = 0.5 →
  (selling_price - 0.65 * selling_price) = 65 :=
by sorry

end NUMINAMATH_CALUDE_old_manufacturing_cost_l2262_226258


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2262_226201

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

-- Define the distance between vertices
def vertex_distance : ℝ := 8

-- Theorem statement
theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), hyperbola_equation x y → vertex_distance = 8 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2262_226201


namespace NUMINAMATH_CALUDE_sin_330_degrees_l2262_226245

theorem sin_330_degrees :
  Real.sin (330 * π / 180) = -(1 / 2) := by sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l2262_226245


namespace NUMINAMATH_CALUDE_expression_evaluation_l2262_226234

theorem expression_evaluation (x y : ℝ) 
  (h : (x + 2)^2 + |y - 2/3| = 0) : 
  1/2 * x - 2 * (x - 1/3 * y^2) + (-3/2 * x + 1/3 * y^2) = 6 + 4/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2262_226234


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2262_226257

theorem min_value_of_sum_of_squares (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → 
  (∀ c d : ℝ, (∃ x : ℝ, x^4 + c*x^3 + d*x^2 + c*x + 1 = 0) → a^2 + b^2 ≤ c^2 + d^2) →
  a^2 + b^2 = 4/5 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2262_226257


namespace NUMINAMATH_CALUDE_remainder_difference_l2262_226228

theorem remainder_difference (d : ℕ) (r : ℕ) (h1 : d > 1) : 
  (1059 % d = r ∧ 1417 % d = r ∧ 2312 % d = r) → d - r = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_difference_l2262_226228


namespace NUMINAMATH_CALUDE_circle_coordinates_l2262_226213

theorem circle_coordinates (π : ℝ) (h : π > 0) :
  let radii : List ℝ := [2, 4, 6, 8, 10]
  let circumference (r : ℝ) : ℝ := 2 * π * r
  let area (r : ℝ) : ℝ := π * r^2
  let coordinates := radii.map (λ r => (circumference r, area r))
  coordinates = [(4*π, 4*π), (8*π, 16*π), (12*π, 36*π), (16*π, 64*π), (20*π, 100*π)] :=
by sorry

end NUMINAMATH_CALUDE_circle_coordinates_l2262_226213


namespace NUMINAMATH_CALUDE_auction_price_increase_l2262_226246

/-- Represents an auction with a starting price, ending price, number of bidders, and bids per bidder -/
structure Auction where
  start_price : ℕ
  end_price : ℕ
  num_bidders : ℕ
  bids_per_bidder : ℕ

/-- Calculates the price increase per bid in an auction -/
def price_increase_per_bid (a : Auction) : ℚ :=
  (a.end_price - a.start_price : ℚ) / (a.num_bidders * a.bids_per_bidder : ℚ)

/-- Theorem stating that for the given auction conditions, the price increase per bid is $5 -/
theorem auction_price_increase (a : Auction)
  (h1 : a.start_price = 15)
  (h2 : a.end_price = 65)
  (h3 : a.num_bidders = 2)
  (h4 : a.bids_per_bidder = 5) :
  price_increase_per_bid a = 5 := by
  sorry

end NUMINAMATH_CALUDE_auction_price_increase_l2262_226246


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l2262_226279

theorem sin_cos_sum_equals_sqrt3_over_2 :
  Real.sin (43 * π / 180) * Real.cos (17 * π / 180) + 
  Real.cos (43 * π / 180) * Real.sin (17 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l2262_226279


namespace NUMINAMATH_CALUDE_unseen_faces_sum_21_l2262_226253

/-- A standard six-sided die -/
structure Die :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)
  (distinct : ∀ i j : Fin 6, i ≠ j → faces i ≠ faces j)

/-- The sum of numbers on three faces of a die -/
def sum_three_faces (d : Die) (f1 f2 f3 : Fin 6) : ℕ :=
  d.faces f1 + d.faces f2 + d.faces f3

/-- The sum of numbers on the opposite faces of three given faces -/
def sum_opposite_faces (d : Die) (f1 f2 f3 : Fin 6) : ℕ :=
  d.faces (5 - f1) + d.faces (5 - f2) + d.faces (5 - f3)

theorem unseen_faces_sum_21 (d1 d2 : Die) 
  (h1 : sum_three_faces d1 0 1 2 = 11) 
  (h2 : sum_three_faces d2 0 3 4 = 10) : 
  sum_opposite_faces d1 0 1 2 + sum_opposite_faces d2 0 3 4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_unseen_faces_sum_21_l2262_226253


namespace NUMINAMATH_CALUDE_max_gcd_of_three_digit_numbers_l2262_226204

theorem max_gcd_of_three_digit_numbers :
  ∀ a b : ℕ,
  a ≠ b →
  a < 10 →
  b < 10 →
  (∃ (x y : ℕ), x = 100 * a + 11 * b ∧ y = 101 * b + 10 * a ∧ Nat.gcd x y ≤ 45) ∧
  (∃ (a' b' : ℕ), a' ≠ b' ∧ a' < 10 ∧ b' < 10 ∧
    Nat.gcd (100 * a' + 11 * b') (101 * b' + 10 * a') = 45) :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_three_digit_numbers_l2262_226204


namespace NUMINAMATH_CALUDE_triangle_area_l2262_226276

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (2, 3)

theorem triangle_area : 
  let doubled_a : ℝ × ℝ := (2 * a.1, 2 * a.2)
  (1/2) * |doubled_a.1 * b.2 - doubled_a.2 * b.1| = 14 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2262_226276


namespace NUMINAMATH_CALUDE_count_quadruples_product_98_l2262_226252

theorem count_quadruples_product_98 :
  (Finset.filter (fun q : Nat × Nat × Nat × Nat => q.1 * q.2.1 * q.2.2.1 * q.2.2.2 = 98)
    (Finset.product (Finset.range 99) (Finset.product (Finset.range 99) (Finset.product (Finset.range 99) (Finset.range 99))))).card = 28 :=
by sorry

end NUMINAMATH_CALUDE_count_quadruples_product_98_l2262_226252


namespace NUMINAMATH_CALUDE_lamps_turned_on_l2262_226211

theorem lamps_turned_on (total_lamps : ℕ) (statement1 statement2 statement3 statement4 : Prop) :
  total_lamps = 10 →
  (statement1 ↔ (∃ x : ℕ, x = 5 ∧ x = total_lamps - (total_lamps - x))) →
  (statement2 ↔ ¬statement1) →
  (statement3 ↔ (∃ y : ℕ, y = 3 ∧ y = total_lamps - (total_lamps - y))) →
  (statement4 ↔ ∃ z : ℕ, z = total_lamps - (total_lamps - z) ∧ 2 ∣ z) →
  (statement1 ∨ statement2 ∨ statement3 ∨ statement4) →
  (statement1 → ¬statement2 ∧ ¬statement3 ∧ ¬statement4) →
  (statement2 → ¬statement1 ∧ ¬statement3 ∧ ¬statement4) →
  (statement3 → ¬statement1 ∧ ¬statement2 ∧ ¬statement4) →
  (statement4 → ¬statement1 ∧ ¬statement2 ∧ ¬statement3) →
  ∃ (lamps_on : ℕ), lamps_on = 9 ∧ lamps_on = total_lamps - (total_lamps - lamps_on) :=
by sorry

end NUMINAMATH_CALUDE_lamps_turned_on_l2262_226211


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2262_226280

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x - 4*a ≥ 0) ↔ -16 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2262_226280


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2262_226273

-- Define the right triangle ABC
def Triangle (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = 180

-- Define the right angle at C
def RightAngleAtC (C : ℝ) : Prop := C = 90

-- Define the sine of angle A
def SineA (sinA : ℝ) : Prop := sinA = Real.sqrt 5 / 3

-- Define the length of side BC
def LengthBC (BC : ℝ) : Prop := BC = 2 * Real.sqrt 5

-- Theorem statement
theorem right_triangle_side_length 
  (A B C AC BC : ℝ) 
  (h_triangle : Triangle A B C) 
  (h_right_angle : RightAngleAtC C) 
  (h_sine_A : SineA (Real.sin (A * π / 180))) 
  (h_BC : LengthBC BC) : 
  AC = 4 := by sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2262_226273


namespace NUMINAMATH_CALUDE_divisibility_of_square_l2262_226230

theorem divisibility_of_square (n : ℕ) (h1 : n > 0) (h2 : ∀ d : ℕ, d > 0 → d ∣ n → d ≤ 30) :
  900 ∣ n^2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_square_l2262_226230


namespace NUMINAMATH_CALUDE_sum_equals_thirteen_thousand_two_hundred_l2262_226239

theorem sum_equals_thirteen_thousand_two_hundred : 9773 + 3427 = 13200 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_thirteen_thousand_two_hundred_l2262_226239


namespace NUMINAMATH_CALUDE_circle_line_disjoint_radius_l2262_226207

-- Define the circle and line
def Circle (O : ℝ × ℝ) (r : ℝ) := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}
def Line (a b c : ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define the distance function
def distance (O : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem circle_line_disjoint_radius (O : ℝ × ℝ) (l : Set (ℝ × ℝ)) (r : ℝ) :
  (∃ (a b c : ℝ), l = Line a b c) →
  (distance O l)^2 - (distance O l) - 20 = 0 →
  (distance O l > 0) →
  (∀ p ∈ Circle O r, p ∉ l) →
  r = 4 := by sorry

end NUMINAMATH_CALUDE_circle_line_disjoint_radius_l2262_226207


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2262_226255

/-- A geometric sequence with first term 1 and common ratio q ≠ -1 -/
def geometric_sequence (q : ℝ) (n : ℕ) : ℝ :=
  q^(n-1)

theorem geometric_sequence_fifth_term
  (q : ℝ)
  (h1 : q ≠ -1)
  (h2 : geometric_sequence q 5 + geometric_sequence q 4 = 3 * (geometric_sequence q 3 + geometric_sequence q 2)) :
  geometric_sequence q 5 = 9 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2262_226255


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l2262_226205

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A * B * C = 1764 →
  A + B + C ≤ 33 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l2262_226205


namespace NUMINAMATH_CALUDE_genesis_work_hours_l2262_226263

/-- The number of hours Genesis worked per day on the new project -/
def hoursPerDayNewProject : ℕ := 6

/-- The number of weeks Genesis worked on the new project -/
def weeksNewProject : ℕ := 3

/-- The number of hours Genesis worked per day on the additional task -/
def hoursPerDayAdditionalTask : ℕ := 3

/-- The number of weeks Genesis worked on the additional task -/
def weeksAdditionalTask : ℕ := 2

/-- The number of days in a week -/
def daysPerWeek : ℕ := 7

/-- The total number of hours Genesis worked during the entire period -/
def totalHoursWorked : ℕ :=
  hoursPerDayNewProject * weeksNewProject * daysPerWeek +
  hoursPerDayAdditionalTask * weeksAdditionalTask * daysPerWeek

theorem genesis_work_hours : totalHoursWorked = 168 := by
  sorry

end NUMINAMATH_CALUDE_genesis_work_hours_l2262_226263


namespace NUMINAMATH_CALUDE_only_five_regular_polyhedra_five_platonic_solids_l2262_226283

/-- A regular polyhedron with n-gon faces and m faces meeting at each vertex -/
structure RegularPolyhedron where
  n : ℕ  -- number of sides of each face
  m : ℕ  -- number of faces meeting at each vertex
  n_ge_3 : n ≥ 3
  m_ge_3 : m ≥ 3

/-- The set of all possible (m, n) pairs for regular polyhedra -/
def valid_regular_polyhedra : Set (ℕ × ℕ) :=
  {(3, 3), (3, 4), (4, 3), (3, 5), (5, 3)}

/-- Theorem stating that only five regular polyhedra exist -/
theorem only_five_regular_polyhedra :
  ∀ p : RegularPolyhedron, (p.m, p.n) ∈ valid_regular_polyhedra := by
  sorry

/-- Corollary: There are exactly five types of regular polyhedra -/
theorem five_platonic_solids :
  ∃! (s : Set (ℕ × ℕ)), s = valid_regular_polyhedra ∧ (∀ p : RegularPolyhedron, (p.m, p.n) ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_only_five_regular_polyhedra_five_platonic_solids_l2262_226283


namespace NUMINAMATH_CALUDE_three_digit_cube_sum_l2262_226243

theorem three_digit_cube_sum : ∃ (n : ℕ), 
  100 ≤ n ∧ n < 1000 ∧ 
  (n = (n / 100)^3 + ((n / 10) % 10)^3 + (n % 10)^3) ∧
  n = 153 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_cube_sum_l2262_226243


namespace NUMINAMATH_CALUDE_area_bounded_by_curves_l2262_226284

/-- The area between the parabola y = x^2 - x and the line y = mx from x = 0 to their intersection point. -/
def area_under_curve (m : ℤ) : ℚ :=
  (m + 1)^3 / 6

/-- The theorem statement -/
theorem area_bounded_by_curves (m n : ℤ) (h1 : m > n) (h2 : n > 0) :
  area_under_curve m - area_under_curve n = 37 / 6 → m = 3 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_area_bounded_by_curves_l2262_226284


namespace NUMINAMATH_CALUDE_any_nonzero_to_zero_power_is_one_l2262_226295

theorem any_nonzero_to_zero_power_is_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_any_nonzero_to_zero_power_is_one_l2262_226295


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2262_226244

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → Real.log (x + 1) > 0) ↔ 
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.log (x₀ + 1) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2262_226244


namespace NUMINAMATH_CALUDE_movie_of_the_year_requirement_l2262_226250

theorem movie_of_the_year_requirement (total_members : ℕ) (fraction : ℚ) : total_members = 775 → fraction = 1/4 → ↑(⌈total_members * fraction⌉) = 194 := by
  sorry

end NUMINAMATH_CALUDE_movie_of_the_year_requirement_l2262_226250


namespace NUMINAMATH_CALUDE_multiply_by_special_number_l2262_226232

theorem multiply_by_special_number : ∃ x : ℝ, x * (1/1000) = 0.735 ∧ 10 * x = 7350 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_special_number_l2262_226232


namespace NUMINAMATH_CALUDE_unbroken_seashells_l2262_226206

theorem unbroken_seashells (total : ℕ) (broken : ℕ) (h1 : total = 7) (h2 : broken = 4) :
  total - broken = 3 := by
  sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l2262_226206


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2262_226216

def tangent_of_angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors
  (a b : ℝ × ℝ)
  (h1 : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 5)
  (h2 : Real.sqrt (a.1^2 + a.2^2) = 2)
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 1) :
  tangent_of_angle_between_vectors a b = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2262_226216


namespace NUMINAMATH_CALUDE_competition_result_l2262_226203

structure Athlete where
  longJump : ℝ
  tripleJump : ℝ
  highJump : ℝ

def totalDistance (a : Athlete) : ℝ :=
  a.longJump + a.tripleJump + a.highJump

def isWinner (a : Athlete) : Prop :=
  totalDistance a = 22 * 3

theorem competition_result (x : ℝ) :
  let athlete1 := Athlete.mk x 30 7
  let athlete2 := Athlete.mk 24 34 8
  isWinner athlete2 ∧ ¬∃y, y = x ∧ isWinner (Athlete.mk y 30 7) := by
  sorry

end NUMINAMATH_CALUDE_competition_result_l2262_226203


namespace NUMINAMATH_CALUDE_fraction_simplification_l2262_226240

theorem fraction_simplification : (3 * 4) / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2262_226240


namespace NUMINAMATH_CALUDE_comic_book_arrangement_count_comic_book_arrangement_count_is_correct_l2262_226209

/-- The number of ways to arrange comic books from different publishers in a stack -/
theorem comic_book_arrangement_count : Nat :=
  let marvel_books : Nat := 8
  let dc_books : Nat := 6
  let image_books : Nat := 5
  let publisher_groups : Nat := 3

  let marvel_arrangements := Nat.factorial marvel_books
  let dc_arrangements := Nat.factorial dc_books
  let image_arrangements := Nat.factorial image_books
  let group_arrangements := Nat.factorial publisher_groups

  marvel_arrangements * dc_arrangements * image_arrangements * group_arrangements

/-- Proof that the number of arrangements is 20,901,888,000 -/
theorem comic_book_arrangement_count_is_correct : 
  comic_book_arrangement_count = 20901888000 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_arrangement_count_comic_book_arrangement_count_is_correct_l2262_226209


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l2262_226290

/-- Given that the solution set of x²-px-q<0 is {x | 2<x<3}, 
    prove the values of p and q and the solution set of qx²-px-1>0 -/
theorem quadratic_inequality_problem 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 - p*x - q < 0}) : 
  (p = 5 ∧ q = -6) ∧ 
  {x : ℝ | q*x^2 - p*x - 1 > 0} = Set.Ioo (-1/2) (-1/3) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_problem_l2262_226290


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2262_226242

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | x^2 ≥ 4}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2262_226242


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2262_226222

open Real

theorem trigonometric_identities (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : sin α = sqrt 5 / 5) : 
  sin (α + π/4) = 3 * sqrt 10 / 10 ∧ tan (2 * α) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2262_226222


namespace NUMINAMATH_CALUDE_larger_group_size_l2262_226231

/-- Given that 36 men can complete a piece of work in 18 days, and a larger group
    of men can complete the same work in 6 days, prove that the larger group
    consists of 108 men. -/
theorem larger_group_size (work : ℕ) (small_group : ℕ) (large_group : ℕ)
    (small_days : ℕ) (large_days : ℕ)
    (h1 : small_group = 36)
    (h2 : small_days = 18)
    (h3 : large_days = 6)
    (h4 : small_group * small_days = work)
    (h5 : large_group * large_days = work) :
    large_group = 108 := by
  sorry

#check larger_group_size

end NUMINAMATH_CALUDE_larger_group_size_l2262_226231


namespace NUMINAMATH_CALUDE_benjamin_presents_l2262_226292

theorem benjamin_presents (ethan_presents : ℝ) (alissa_more_than_ethan : ℝ) (benjamin_less_than_alissa : ℝ) 
  (h1 : ethan_presents = 31.5)
  (h2 : alissa_more_than_ethan = 22)
  (h3 : benjamin_less_than_alissa = 8.5) :
  ethan_presents + alissa_more_than_ethan - benjamin_less_than_alissa = 45 :=
by sorry

end NUMINAMATH_CALUDE_benjamin_presents_l2262_226292


namespace NUMINAMATH_CALUDE_shopkeeper_change_l2262_226218

/-- Represents the change given by the shopkeeper -/
structure Change where
  total_bills : ℕ
  bill_value_1 : ℕ
  bill_value_2 : ℕ
  noodles_value : ℕ

/-- The problem statement -/
theorem shopkeeper_change (c : Change) (h1 : c.total_bills = 16)
    (h2 : c.bill_value_1 = 10) (h3 : c.bill_value_2 = 5) (h4 : c.noodles_value = 5)
    (h5 : 100 = c.noodles_value + c.bill_value_1 * x + c.bill_value_2 * (c.total_bills - x)) :
    x = 3 :=
  sorry

end NUMINAMATH_CALUDE_shopkeeper_change_l2262_226218


namespace NUMINAMATH_CALUDE_edward_received_amount_l2262_226297

def edward_problem (initial_amount spent_amount final_amount received_amount : ℝ) : Prop :=
  initial_amount = 14 ∧
  spent_amount = 17 ∧
  final_amount = 7 ∧
  initial_amount - spent_amount + received_amount = final_amount

theorem edward_received_amount :
  ∃ (received_amount : ℝ), edward_problem 14 17 7 received_amount ∧ received_amount = 10 := by
  sorry

end NUMINAMATH_CALUDE_edward_received_amount_l2262_226297


namespace NUMINAMATH_CALUDE_min_cyclic_fraction_sum_l2262_226226

theorem min_cyclic_fraction_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b + b / c + c / d + d / a) ≥ 4 ∧ 
  ((a / b + b / c + c / d + d / a) = 4 ↔ a = b ∧ b = c ∧ c = d) := by
  sorry

end NUMINAMATH_CALUDE_min_cyclic_fraction_sum_l2262_226226


namespace NUMINAMATH_CALUDE_square_root_of_four_l2262_226214

theorem square_root_of_four (x : ℝ) : x^2 = 4 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_four_l2262_226214


namespace NUMINAMATH_CALUDE_inequality_proof_l2262_226299

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  c^2 < c*d :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2262_226299


namespace NUMINAMATH_CALUDE_computer_factory_month_days_l2262_226277

/-- Proves the number of days in a month given computer production rates --/
theorem computer_factory_month_days
  (monthly_production : ℕ)
  (half_hour_production : ℚ)
  (h1 : monthly_production = 3024)
  (h2 : half_hour_production = 225 / 100) :
  (monthly_production : ℚ) / ((half_hour_production * 2 * 24) : ℚ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_computer_factory_month_days_l2262_226277


namespace NUMINAMATH_CALUDE_equal_distribution_probability_l2262_226262

/-- Represents a player in the game -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Charlie : Player
| Dana : Player

/-- The state of the game is represented by the money each player has -/
def GameState := Player → ℕ

/-- The initial state of the game where each player has 1 dollar -/
def initialState : GameState := fun _ => 1

/-- A single turn of the game where a player gives 1 dollar to another randomly chosen player -/
def turn (state : GameState) : GameState := sorry

/-- The probability that after 40 turns, each player has 1 dollar -/
def probabilityEqualDistribution (n : ℕ) : ℝ :=
  sorry

/-- The main theorem stating that the probability of equal distribution after 40 turns is 1/9 -/
theorem equal_distribution_probability :
  probabilityEqualDistribution 40 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_probability_l2262_226262


namespace NUMINAMATH_CALUDE_no_regular_polygon_inscription_l2262_226269

-- Define an ellipse with unequal axes
structure Ellipse where
  majorAxis : ℝ
  minorAxis : ℝ
  axesUnequal : majorAxis ≠ minorAxis

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  moreThanFourSides : sides > 4

-- Define the concept of inscribing a polygon in an ellipse
def isInscribed (p : RegularPolygon) (e : Ellipse) : Prop :=
  sorry -- Definition of inscription

-- Theorem statement
theorem no_regular_polygon_inscription 
  (e : Ellipse) (p : RegularPolygon) : ¬ isInscribed p e := by
  sorry

#check no_regular_polygon_inscription

end NUMINAMATH_CALUDE_no_regular_polygon_inscription_l2262_226269


namespace NUMINAMATH_CALUDE_alberts_to_marys_age_ratio_l2262_226247

theorem alberts_to_marys_age_ratio (albert_age mary_age betty_age : ℕ) : 
  betty_age = 4 → 
  albert_age = 4 * betty_age → 
  mary_age = albert_age - 8 → 
  (albert_age : ℚ) / mary_age = 2 := by
sorry

end NUMINAMATH_CALUDE_alberts_to_marys_age_ratio_l2262_226247


namespace NUMINAMATH_CALUDE_calculate_expression_l2262_226296

theorem calculate_expression : 
  (Real.pi - 3.14) ^ 0 + |-Real.sqrt 3| - (1/2)⁻¹ - Real.sin (60 * π / 180) = -1 + Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2262_226296


namespace NUMINAMATH_CALUDE_x_squared_ge_one_necessary_not_sufficient_l2262_226223

theorem x_squared_ge_one_necessary_not_sufficient :
  (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) ∧
  (∃ x : ℝ, x^2 ≥ 1 ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_ge_one_necessary_not_sufficient_l2262_226223


namespace NUMINAMATH_CALUDE_mold_radius_l2262_226265

/-- The radius of a circular mold with diameter 4 inches is 2 inches -/
theorem mold_radius (d : ℝ) (h : d = 4) : d / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mold_radius_l2262_226265


namespace NUMINAMATH_CALUDE_base8_digit_product_l2262_226266

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

theorem base8_digit_product :
  productOfList (toBase8 8679) = 392 := by
  sorry

end NUMINAMATH_CALUDE_base8_digit_product_l2262_226266


namespace NUMINAMATH_CALUDE_fraction_simplest_form_l2262_226289

/-- A fraction is in simplest form if its numerator and denominator have no common factors other than 1. -/
def IsSimplestForm (n d : ℤ) : Prop :=
  ∀ k : ℤ, k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1

/-- Given x and y are integers and x ≠ 0, prove that (x+y)/(2x) is in simplest form. -/
theorem fraction_simplest_form (x y : ℤ) (hx : x ≠ 0) : 
  IsSimplestForm (x + y) (2 * x) :=
sorry

end NUMINAMATH_CALUDE_fraction_simplest_form_l2262_226289


namespace NUMINAMATH_CALUDE_olivias_phone_pictures_l2262_226291

theorem olivias_phone_pictures :
  ∀ (phone_pics camera_pics total_albums pics_per_album : ℕ),
    camera_pics = 35 →
    total_albums = 8 →
    pics_per_album = 5 →
    phone_pics + camera_pics = total_albums * pics_per_album →
    phone_pics = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_olivias_phone_pictures_l2262_226291


namespace NUMINAMATH_CALUDE_divisibility_by_three_l2262_226221

theorem divisibility_by_three (B : Nat) : 
  B < 10 → (514 * 10 + B) % 3 = 0 ↔ B = 2 ∨ B = 5 ∨ B = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l2262_226221


namespace NUMINAMATH_CALUDE_total_tiles_from_black_tiles_total_tiles_is_2601_l2262_226256

/-- Represents a square floor covered with tiles -/
structure TiledFloor where
  size : ℕ
  blackTilesCount : ℕ

/-- Theorem stating the relationship between the number of black tiles and total tiles -/
theorem total_tiles_from_black_tiles (floor : TiledFloor) 
  (h1 : floor.blackTilesCount = 101) : 
  floor.size * floor.size = 2601 := by
  sorry

/-- Main theorem proving the total number of tiles -/
theorem total_tiles_is_2601 (floor : TiledFloor) 
  (h1 : floor.blackTilesCount = 101) 
  (h2 : floor.blackTilesCount = 2 * floor.size - 1) : 
  floor.size * floor.size = 2601 := by
  sorry

end NUMINAMATH_CALUDE_total_tiles_from_black_tiles_total_tiles_is_2601_l2262_226256


namespace NUMINAMATH_CALUDE_max_correct_answers_l2262_226238

theorem max_correct_answers (total_questions : Nat) (correct_points : Int) (incorrect_points : Int) (total_score : Int) :
  total_questions = 30 →
  correct_points = 4 →
  incorrect_points = -3 →
  total_score = 72 →
  ∃ (correct incorrect unanswered : Nat),
    correct + incorrect + unanswered = total_questions ∧
    correct * correct_points + incorrect * incorrect_points = total_score ∧
    correct ≤ 21 ∧
    ∀ (c i u : Nat),
      c + i + u = total_questions →
      c * correct_points + i * incorrect_points = total_score →
      c ≤ 21 :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l2262_226238


namespace NUMINAMATH_CALUDE_smallest_four_digit_sum_15_l2262_226220

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem smallest_four_digit_sum_15 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 15 → n ≥ 1009 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_sum_15_l2262_226220


namespace NUMINAMATH_CALUDE_rabbit_can_escape_l2262_226200

/-- Represents a point in 2D space -/
structure Point where
  x : Real
  y : Real

/-- Represents a square with side length 1 -/
structure Square where
  center : Point
  side_length : Real := 1

/-- Represents an entity (rabbit or wolf) with a position and speed -/
structure Entity where
  position : Point
  speed : Real

/-- Theorem stating that the rabbit can escape the square -/
theorem rabbit_can_escape (s : Square) (rabbit : Entity) (wolves : Finset Entity) :
  rabbit.position = s.center →
  wolves.card = 4 →
  (∀ w ∈ wolves, w.speed = 1.4 * rabbit.speed) →
  (∀ w ∈ wolves, w.position.x = 0 ∨ w.position.x = 1) →
  (∀ w ∈ wolves, w.position.y = 0 ∨ w.position.y = 1) →
  ∃ (escape_path : Real → Point),
    (escape_path 0 = rabbit.position) ∧
    (∃ t : Real, t > 0 ∧ (escape_path t).x = 0 ∨ (escape_path t).x = 1 ∨ (escape_path t).y = 0 ∨ (escape_path t).y = 1) ∧
    (∀ w ∈ wolves, ∀ t : Real, t ≥ 0 → 
      (escape_path t).x ≠ w.position.x ∨ (escape_path t).y ≠ w.position.y) :=
sorry

end NUMINAMATH_CALUDE_rabbit_can_escape_l2262_226200
