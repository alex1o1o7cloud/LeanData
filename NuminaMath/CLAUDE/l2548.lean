import Mathlib

namespace NUMINAMATH_CALUDE_original_price_calculation_l2548_254839

/-- Given a discount percentage and a discounted price, calculates the original price. -/
def calculateOriginalPrice (discountPercentage : ℚ) (discountedPrice : ℚ) : ℚ :=
  discountedPrice / (1 - discountPercentage)

/-- Theorem: Given a 30% discount and a discounted price of 560, the original price is 800. -/
theorem original_price_calculation :
  calculateOriginalPrice (30 / 100) 560 = 800 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2548_254839


namespace NUMINAMATH_CALUDE_zeros_after_one_in_100_pow_250_l2548_254899

/-- The number of zeros following the digit '1' in the expanded form of 100^250 -/
def zeros_after_one : ℕ := 500

/-- The exponent in the expression 100^250 -/
def exponent : ℕ := 250

theorem zeros_after_one_in_100_pow_250 : zeros_after_one = 2 * exponent := by
  sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_100_pow_250_l2548_254899


namespace NUMINAMATH_CALUDE_soap_box_width_theorem_l2548_254823

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The partial dimensions of the soap box (width is unknown) -/
def soapBoxPartialDimensions (w : ℝ) : BoxDimensions :=
  { length := 7, width := w, height := 5 }

/-- The maximum number of soap boxes that can fit in the carton -/
def maxSoapBoxes : ℕ := 300

/-- Theorem stating the width of the soap box that allows exactly 300 to fit in the carton -/
theorem soap_box_width_theorem (w : ℝ) :
  (boxVolume cartonDimensions = maxSoapBoxes * boxVolume (soapBoxPartialDimensions w)) ↔ w = 6 := by
  sorry

end NUMINAMATH_CALUDE_soap_box_width_theorem_l2548_254823


namespace NUMINAMATH_CALUDE_scale_division_l2548_254868

/-- The length of the scale in inches -/
def scale_length : ℕ := 80

/-- The length of each part in inches -/
def part_length : ℕ := 20

/-- The number of equal parts the scale is divided into -/
def num_parts : ℕ := scale_length / part_length

theorem scale_division :
  scale_length % part_length = 0 ∧ num_parts = 4 :=
by sorry

end NUMINAMATH_CALUDE_scale_division_l2548_254868


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2548_254856

theorem complex_equation_solution (z z₁ z₂ : ℂ) : 
  z₁ = 5 + 10*I ∧ z₂ = 3 - 4*I ∧ (1 : ℂ)/z = 1/z₁ + 1/z₂ → z = 5 - (5/2)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2548_254856


namespace NUMINAMATH_CALUDE_archibald_tennis_game_l2548_254840

/-- Archibald's Tennis Game Problem -/
theorem archibald_tennis_game (games_won_by_archibald : ℕ) (percentage_won_by_archibald : ℚ) :
  games_won_by_archibald = 12 →
  percentage_won_by_archibald = 2/5 →
  ∃ (total_games : ℕ) (games_won_by_brother : ℕ),
    total_games = games_won_by_archibald + games_won_by_brother ∧
    (games_won_by_archibald : ℚ) / total_games = percentage_won_by_archibald ∧
    games_won_by_brother = 18 :=
by
  sorry

#check archibald_tennis_game

end NUMINAMATH_CALUDE_archibald_tennis_game_l2548_254840


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l2548_254835

/-- Two points are symmetric with respect to the x-axis if they have the same x-coordinate
    and their y-coordinates are negatives of each other -/
def symmetric_wrt_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- The theorem states that if point A(m,n) is symmetric to point B(1,-2) with respect to the x-axis,
    then m = 1 and n = 2 -/
theorem symmetry_coordinates :
  ∀ m n : ℝ, symmetric_wrt_x_axis (m, n) (1, -2) → m = 1 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l2548_254835


namespace NUMINAMATH_CALUDE_red_subsequence_2009_l2548_254870

/-- Represents the coloring rule for the red subsequence -/
def red_subsequence : ℕ → ℕ → ℕ → ℕ
| 0, _, _ => 1
| (n+1), count, last =>
  if n % 2 = 0 then
    if count < n + 1 then red_subsequence n (count + 1) (last + 2)
    else red_subsequence n 0 (last + 1)
  else
    if count < n + 2 then red_subsequence n (count + 1) (last + 2)
    else red_subsequence (n + 1) 0 last

/-- The 2009th number in the red subsequence is 3953 -/
theorem red_subsequence_2009 :
  (red_subsequence 1000 0 1) = 3953 := by sorry

end NUMINAMATH_CALUDE_red_subsequence_2009_l2548_254870


namespace NUMINAMATH_CALUDE_correct_operation_l2548_254853

theorem correct_operation (a b : ℝ) : 2 * a^2 * b - a^2 * b = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2548_254853


namespace NUMINAMATH_CALUDE_square_sum_value_l2548_254825

theorem square_sum_value (x y : ℝ) (h : 5 * x^2 + y^2 - 4*x*y + 24 ≤ 10*x - 1) : x^2 + y^2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l2548_254825


namespace NUMINAMATH_CALUDE_average_difference_with_data_error_l2548_254882

theorem average_difference_with_data_error (data : List ℝ) (wrong_value correct_value : ℝ) : 
  data.length = 30 →
  wrong_value = 15 →
  correct_value = 105 →
  wrong_value ∈ data →
  (data.sum / data.length) - ((data.sum - wrong_value + correct_value) / data.length) = -3 := by
sorry

end NUMINAMATH_CALUDE_average_difference_with_data_error_l2548_254882


namespace NUMINAMATH_CALUDE_inequality_proof_l2548_254809

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b * c + 1) + b / (a * c + 1) + c / (a * b + 1) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2548_254809


namespace NUMINAMATH_CALUDE_lara_future_age_l2548_254896

def lara_age_7_years_ago : ℕ := 9

def lara_current_age : ℕ := lara_age_7_years_ago + 7

def lara_age_10_years_from_now : ℕ := lara_current_age + 10

theorem lara_future_age : lara_age_10_years_from_now = 26 := by
  sorry

end NUMINAMATH_CALUDE_lara_future_age_l2548_254896


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l2548_254863

-- Define the dimensions of the rectangles
def rect1_width : ℕ := 4
def rect1_height : ℕ := 5
def rect2_width : ℕ := 3
def rect2_height : ℕ := 6

-- Define the area calculation function
def area (width height : ℕ) : ℕ := width * height

-- Theorem statement
theorem rectangle_area_difference :
  area rect1_width rect1_height - area rect2_width rect2_height = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l2548_254863


namespace NUMINAMATH_CALUDE_triangle_side_length_l2548_254824

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  c = Real.sqrt 2 →
  b = Real.sqrt 6 →
  B = 2 * π / 3 →  -- 120° in radians
  a^2 + a * Real.sqrt 2 - 4 = 0 →
  a = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2548_254824


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l2548_254875

theorem inequality_system_integer_solutions :
  let S : Set ℤ := {x | (5 * x - 2 > 3 * (x + 1)) ∧ (x / 3 ≤ (5 - x) / 2)}
  S = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l2548_254875


namespace NUMINAMATH_CALUDE_ali_ate_four_times_more_l2548_254861

def total_apples : ℕ := 80
def sara_apples : ℕ := 16

def ali_apples : ℕ := total_apples - sara_apples

theorem ali_ate_four_times_more :
  ali_apples / sara_apples = 4 :=
by sorry

end NUMINAMATH_CALUDE_ali_ate_four_times_more_l2548_254861


namespace NUMINAMATH_CALUDE_binary_1010001011_equals_base7_1620_l2548_254842

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem binary_1010001011_equals_base7_1620 :
  decimal_to_base7 (binary_to_decimal [true, true, false, true, false, false, false, false, true, false]) = [1, 6, 2, 0] := by
  sorry

end NUMINAMATH_CALUDE_binary_1010001011_equals_base7_1620_l2548_254842


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l2548_254808

theorem final_sum_after_operations (x y S : ℝ) (h : x + y = S) :
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l2548_254808


namespace NUMINAMATH_CALUDE_cookie_value_approx_l2548_254866

/-- Calculates the value of cookies left after a series of operations -/
def cookie_value (initial : ℝ) (eaten1 : ℝ) (received1 : ℝ) (eaten2 : ℝ) (received2 : ℝ) (share_fraction : ℝ) (cost_per_cookie : ℝ) : ℝ :=
  let remaining1 := initial - eaten1
  let after_first_gift := remaining1 + received1
  let after_second_eating := after_first_gift - eaten2
  let total_before_sharing := after_second_eating + received2
  let shared := total_before_sharing * share_fraction
  let final_count := total_before_sharing - shared
  final_count * cost_per_cookie

/-- The value of cookies left is approximately $1.73 -/
theorem cookie_value_approx :
  ∃ ε > 0, |cookie_value 7 2.5 4.2 1.3 3 (1/3) 0.25 - 1.73| < ε :=
sorry

end NUMINAMATH_CALUDE_cookie_value_approx_l2548_254866


namespace NUMINAMATH_CALUDE_percentage_problem_l2548_254874

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : x / 100 * 20 = 8) : x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2548_254874


namespace NUMINAMATH_CALUDE_x_equation_implies_polynomial_value_l2548_254895

theorem x_equation_implies_polynomial_value (x : ℝ) (h : x + 1/x = Real.sqrt 3) :
  x^7 - 5*x^5 + x^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_implies_polynomial_value_l2548_254895


namespace NUMINAMATH_CALUDE_polynomial_coefficient_a1_l2548_254833

theorem polynomial_coefficient_a1 (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^10 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
            a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9 + a₁₀*(x-1)^10) →
  a₁ = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_a1_l2548_254833


namespace NUMINAMATH_CALUDE_more_women_than_men_l2548_254821

/-- Proves that in a group of 15 people where the ratio of men to women is 0.5, there are 5 more women than men. -/
theorem more_women_than_men (total : ℕ) (ratio : ℚ) (men : ℕ) (women : ℕ) : 
  total = 15 → 
  ratio = 1/2 → 
  men + women = total → 
  (men : ℚ) / (women : ℚ) = ratio → 
  women - men = 5 := by
sorry

end NUMINAMATH_CALUDE_more_women_than_men_l2548_254821


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l2548_254826

theorem different_color_chips_probability :
  let total_chips : ℕ := 7 + 6 + 5
  let purple_chips : ℕ := 7
  let green_chips : ℕ := 6
  let orange_chips : ℕ := 5
  let prob_purple : ℚ := purple_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  let prob_orange : ℚ := orange_chips / total_chips
  let prob_not_purple : ℚ := (green_chips + orange_chips) / total_chips
  let prob_not_green : ℚ := (purple_chips + orange_chips) / total_chips
  let prob_not_orange : ℚ := (purple_chips + green_chips) / total_chips
  (prob_purple * prob_not_purple + prob_green * prob_not_green + prob_orange * prob_not_orange) = 107 / 162 :=
by sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l2548_254826


namespace NUMINAMATH_CALUDE_problem_solution_inequality_proof_l2548_254838

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := |x - m|
def g (m : ℝ) (x : ℝ) : ℝ := 2 * f m x - f m (x + m)

-- Theorem statement
theorem problem_solution (m : ℝ) (h_m : m > 0) :
  (∃ (x : ℝ), g m x = -1 ∧ ∀ (y : ℝ), g m y ≥ -1) ↔ m = 1 :=
sorry

theorem inequality_proof (m : ℝ) (h_m : m > 0) 
  (a b : ℝ) (h_a : |a| < m) (h_b : |b| < m) (h_a_neq_0 : a ≠ 0) :
  |a * b - m| > |a| * |b / a - m| :=
sorry

end NUMINAMATH_CALUDE_problem_solution_inequality_proof_l2548_254838


namespace NUMINAMATH_CALUDE_number_square_problem_l2548_254834

theorem number_square_problem : ∃! x : ℝ, x^2 + 95 = (x - 19)^2 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_square_problem_l2548_254834


namespace NUMINAMATH_CALUDE_triangle_side_length_l2548_254848

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  a = 2 →
  b = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2548_254848


namespace NUMINAMATH_CALUDE_equal_chord_circle_equation_l2548_254836

/-- A circle passing through two given points with equal chord lengths on coordinate axes -/
structure EqualChordCircle where
  -- Center of the circle
  center : ℝ × ℝ
  -- Radius of the circle
  radius : ℝ
  -- The circle passes through P(1, 2)
  passes_through_P : (center.1 - 1)^2 + (center.2 - 2)^2 = radius^2
  -- The circle passes through Q(-2, 3)
  passes_through_Q : (center.1 + 2)^2 + (center.2 - 3)^2 = radius^2
  -- Equal chord lengths on coordinate axes
  equal_chords : (center.1)^2 + (radius^2 - center.1^2) = (center.2)^2 + (radius^2 - center.2^2)

/-- The theorem stating that the circle has one of the two specific equations -/
theorem equal_chord_circle_equation (c : EqualChordCircle) :
  ((c.center.1 = -2 ∧ c.center.2 = -2 ∧ c.radius^2 = 25) ∨
   (c.center.1 = -1 ∧ c.center.2 = 1 ∧ c.radius^2 = 5)) :=
sorry

end NUMINAMATH_CALUDE_equal_chord_circle_equation_l2548_254836


namespace NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_is_15_l2548_254817

/-- Given a parallelogram with vertices (4,4), (7,4), (5,9), and (8,9) in a rectangular coordinate system,
    prove that its area is 15 square units. -/
theorem parallelogram_area : ℝ → Prop :=
  fun area =>
    let x₁ : ℝ := 4
    let y₁ : ℝ := 4
    let x₂ : ℝ := 7
    let y₂ : ℝ := 4
    let x₃ : ℝ := 5
    let y₃ : ℝ := 9
    let x₄ : ℝ := 8
    let y₄ : ℝ := 9
    let base : ℝ := x₂ - x₁
    let height : ℝ := y₃ - y₁
    area = base * height

/-- Proof of the parallelogram area theorem -/
theorem parallelogram_area_is_15 : parallelogram_area 15 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_is_15_l2548_254817


namespace NUMINAMATH_CALUDE_vector_sum_norm_l2548_254893

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_sum_norm (a b : E) 
  (ha : ‖a‖ = 1) 
  (hb : ‖b‖ = 1) 
  (hab : ‖a - b‖ = 1) : 
  ‖a + b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_norm_l2548_254893


namespace NUMINAMATH_CALUDE_h_neither_even_nor_odd_l2548_254888

-- Define an even function
def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- Define the function h
def h (g : ℝ → ℝ) (x : ℝ) : ℝ := g (g x + x)

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem h_neither_even_nor_odd (g : ℝ → ℝ) (hg : even_function g) :
  ¬(is_even (h g)) ∧ ¬(is_odd (h g)) :=
sorry

end NUMINAMATH_CALUDE_h_neither_even_nor_odd_l2548_254888


namespace NUMINAMATH_CALUDE_gmat_scores_l2548_254884

theorem gmat_scores (u v : ℝ) (h1 : u > v) (h2 : u - v = (u + v) / 2) : v / u = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gmat_scores_l2548_254884


namespace NUMINAMATH_CALUDE_polycarp_multiplication_l2548_254829

theorem polycarp_multiplication (a b : ℕ) : 
  100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 →
  1000 * a + b = 7 * a * b →
  a = 143 ∧ b = 143 := by
sorry

end NUMINAMATH_CALUDE_polycarp_multiplication_l2548_254829


namespace NUMINAMATH_CALUDE_four_is_integer_l2548_254818

-- Define the set of natural numbers
def NaturalNumber : Type := ℕ

-- Define the set of integers
def Integer : Type := ℤ

-- Define the property that all natural numbers are integers
axiom natural_are_integers : ∀ (n : NaturalNumber), Integer

-- Define 4 as a natural number
axiom four_is_natural : NaturalNumber

-- Theorem to prove
theorem four_is_integer : Integer :=
  sorry

end NUMINAMATH_CALUDE_four_is_integer_l2548_254818


namespace NUMINAMATH_CALUDE_best_approximation_l2548_254827

/-- The function representing the sum of squared differences between measurements and x -/
def y (x : ℝ) : ℝ := (x - 6.5)^2 + (x - 5.9)^2 + (x - 6.0)^2 + (x - 6.7)^2 + (x - 4.5)^2

/-- The theorem stating that 5.92 minimizes the function y -/
theorem best_approximation :
  ∀ x : ℝ, y 5.92 ≤ y x :=
sorry

end NUMINAMATH_CALUDE_best_approximation_l2548_254827


namespace NUMINAMATH_CALUDE_point_on_line_l2548_254849

/-- The value of m that makes the point (3, -2) lie on the line 2m - my = 3x + 1 -/
theorem point_on_line (m : ℚ) : m = 5/2 ↔ 2*m - m*(-2) = 3*3 + 1 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l2548_254849


namespace NUMINAMATH_CALUDE_lakers_win_series_in_five_l2548_254847

def probability_win_series_in_five (p : ℚ) : ℚ :=
  let q := 1 - p
  6 * q^2 * p^2 * q

theorem lakers_win_series_in_five :
  probability_win_series_in_five (3/4) = 27/512 := by
  sorry

end NUMINAMATH_CALUDE_lakers_win_series_in_five_l2548_254847


namespace NUMINAMATH_CALUDE_parabola_directrix_l2548_254813

/-- Given a parabola with equation x = 4y², prove that its directrix has equation x = -1/16 -/
theorem parabola_directrix (y : ℝ) : 
  (∃ x, x = 4 * y^2) → 
  (∃ x, x = -1/16 ∧ ∀ y, x = -1/16 → (x + 1/16)^2 = y^2/4) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2548_254813


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2548_254871

theorem sum_of_solutions (x : ℝ) : 
  (9 * x / 45 = 6 / x) → (x = 0 ∨ x = 6 / 5) ∧ (0 + 6 / 5 = 6 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2548_254871


namespace NUMINAMATH_CALUDE_evaluate_expression_l2548_254831

theorem evaluate_expression : 5 - 7 * (8 - 3^2) * 4 = 33 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2548_254831


namespace NUMINAMATH_CALUDE_intersection_points_differ_from_roots_l2548_254894

-- Define the original quadratic equation
def original_equation (x : ℝ) : Prop := x^2 - 4*x + 3 = 0

-- Define the roots of the original equation
def roots : Set ℝ := {1, 3}

-- Define the intersection points of y = x and y = x^2 - 4x + 4
def intersection_points : Set ℝ := {x : ℝ | x = x^2 - 4*x + 4}

-- Theorem statement
theorem intersection_points_differ_from_roots : intersection_points ≠ roots := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_differ_from_roots_l2548_254894


namespace NUMINAMATH_CALUDE_steak_knife_cost_l2548_254851

/-- The cost of each single steak knife, given the number of sets, knives per set, and cost per set. -/
theorem steak_knife_cost (num_sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℝ) :
  num_sets = 2 → knives_per_set = 4 → cost_per_set = 80 →
  (num_sets * cost_per_set) / (num_sets * knives_per_set) = 20 := by
  sorry

end NUMINAMATH_CALUDE_steak_knife_cost_l2548_254851


namespace NUMINAMATH_CALUDE_x_minus_q_equals_three_minus_two_q_l2548_254816

theorem x_minus_q_equals_three_minus_two_q (x q : ℝ) 
  (h1 : |x - 3| = q) 
  (h2 : x < 3) : 
  x - q = 3 - 2*q := by
sorry

end NUMINAMATH_CALUDE_x_minus_q_equals_three_minus_two_q_l2548_254816


namespace NUMINAMATH_CALUDE_job_application_age_range_l2548_254879

theorem job_application_age_range 
  (average_age : ℝ) 
  (standard_deviation : ℝ) 
  (max_different_ages : ℕ) 
  (h1 : average_age = 31) 
  (h2 : standard_deviation = 9) 
  (h3 : max_different_ages = 19) :
  (max_different_ages : ℝ) / standard_deviation = 19 / 18 := by
sorry

end NUMINAMATH_CALUDE_job_application_age_range_l2548_254879


namespace NUMINAMATH_CALUDE_overall_profit_l2548_254897

def grinder_cost : ℚ := 15000
def mobile_cost : ℚ := 8000
def grinder_loss_percent : ℚ := 5
def mobile_profit_percent : ℚ := 10

def grinder_selling_price : ℚ := grinder_cost * (1 - grinder_loss_percent / 100)
def mobile_selling_price : ℚ := mobile_cost * (1 + mobile_profit_percent / 100)

def total_cost : ℚ := grinder_cost + mobile_cost
def total_selling_price : ℚ := grinder_selling_price + mobile_selling_price

theorem overall_profit : total_selling_price - total_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_overall_profit_l2548_254897


namespace NUMINAMATH_CALUDE_scientific_notation_of_2720000_l2548_254852

theorem scientific_notation_of_2720000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 2720000 = a * (10 : ℝ) ^ n ∧ a = 2.72 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2720000_l2548_254852


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l2548_254810

noncomputable def f (x : ℝ) : ℝ := -Real.cos x + Real.log x

theorem f_derivative_at_one : 
  deriv f 1 = 1 + Real.sin 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l2548_254810


namespace NUMINAMATH_CALUDE_sin_neg_ten_pi_thirds_l2548_254862

theorem sin_neg_ten_pi_thirds : Real.sin (-10 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_ten_pi_thirds_l2548_254862


namespace NUMINAMATH_CALUDE_common_roots_product_square_l2548_254887

-- Define the two cubic equations
def cubic1 (x A : ℝ) : ℝ := x^3 + A*x + 20
def cubic2 (x B : ℝ) : ℝ := x^3 + B*x^2 + 80

-- Define the property of having two common roots
def has_two_common_roots (A B : ℝ) : Prop :=
  ∃ (p q : ℝ), p ≠ q ∧ 
    cubic1 p A = 0 ∧ cubic1 q A = 0 ∧
    cubic2 p B = 0 ∧ cubic2 q B = 0

-- Theorem statement
theorem common_roots_product_square (A B : ℝ) 
  (h : has_two_common_roots A B) :
  ∃ (p q : ℝ), p ≠ q ∧ 
    cubic1 p A = 0 ∧ cubic1 q A = 0 ∧
    cubic2 p B = 0 ∧ cubic2 q B = 0 ∧
    (p*q)^2 = 16 * Real.sqrt 100 :=
sorry

end NUMINAMATH_CALUDE_common_roots_product_square_l2548_254887


namespace NUMINAMATH_CALUDE_domain_of_f_sin_l2548_254883

def is_in_domain (f : ℝ → ℝ) (x : ℝ) : Prop :=
  0 < x ∧ x ≤ 1

theorem domain_of_f_sin (f : ℝ → ℝ) :
  (∀ x, is_in_domain f x ↔ 0 < x ∧ x ≤ 1) →
  ∀ x, is_in_domain f (Real.sin x) ↔ ∃ k : ℤ, 2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_sin_l2548_254883


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2548_254811

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a1 : a 1 = 2) 
  (h_S3 : a 1 + a 2 + a 3 = 26) 
  : q = 3 ∨ q = -4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2548_254811


namespace NUMINAMATH_CALUDE_radical_axes_intersect_l2548_254819

/-- A hexagon with vertices in a 2D plane -/
structure Hexagon :=
  (vertices : Fin 6 → ℝ × ℝ)

/-- A circle in a 2D plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- The radical axis of two circles -/
def radical_axis (c1 c2 : Circle) : Set (ℝ × ℝ) := sorry

/-- A point lies on a circle -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

/-- The diagonals of a hexagon -/
def diagonals (h : Hexagon) : List (Set (ℝ × ℝ)) := sorry

/-- The intersection point of a list of sets -/
def intersection_point (sets : List (Set (ℝ × ℝ))) : Option (ℝ × ℝ) := sorry

/-- Main theorem -/
theorem radical_axes_intersect (h : Hexagon) : 
  (∀ (i j k l : Fin 6), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
    ¬∃ (c : Circle), on_circle (h.vertices i) c ∧ on_circle (h.vertices j) c ∧ 
                     on_circle (h.vertices k) c ∧ on_circle (h.vertices l) c) →
  (∃! p, ∀ d ∈ diagonals h, p ∈ d) →
  ∃! p, ∀ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ k ≠ i → 
    p ∈ radical_axis 
      (Circle.mk (h.vertices i) (dist (h.vertices i) (h.vertices j))) 
      (Circle.mk (h.vertices j) (dist (h.vertices j) (h.vertices k))) :=
by sorry

end NUMINAMATH_CALUDE_radical_axes_intersect_l2548_254819


namespace NUMINAMATH_CALUDE_min_sum_squares_groups_l2548_254873

def S : Finset Int := {-9, -8, -4, -1, 1, 5, 7, 10}

theorem min_sum_squares_groups (p q r s t u v w : Int)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  ∀ (x y : Int), x = (p + q + r + s)^2 + (t + u + v + w)^2 → x ≥ 1 ∧ (x = 1 → y = 1) :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_groups_l2548_254873


namespace NUMINAMATH_CALUDE_abe_age_problem_l2548_254867

/-- The number of years ago that satisfies the equation for Abe's ages -/
def years_ago : ℕ := 21

/-- Abe's present age -/
def present_age : ℕ := 25

/-- The sum of Abe's present age and his age a certain number of years ago -/
def age_sum : ℕ := 29

theorem abe_age_problem :
  present_age + (present_age - years_ago) = age_sum :=
by sorry

end NUMINAMATH_CALUDE_abe_age_problem_l2548_254867


namespace NUMINAMATH_CALUDE_intersection_distance_squared_specific_circles_l2548_254805

/-- Two circles in a 2D plane -/
structure TwoCircles where
  center1 : ℝ × ℝ
  radius1 : ℝ
  center2 : ℝ × ℝ
  radius2 : ℝ

/-- The square of the distance between intersection points of two circles -/
def intersectionDistanceSquared (circles : TwoCircles) : ℝ := sorry

theorem intersection_distance_squared_specific_circles :
  let circles : TwoCircles := {
    center1 := (3, -2),
    radius1 := 5,
    center2 := (3, 4),
    radius2 := 3
  }
  intersectionDistanceSquared circles = 224 / 9 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_specific_circles_l2548_254805


namespace NUMINAMATH_CALUDE_tuesday_earnings_l2548_254865

/-- Lauren's earnings from social media on Tuesday -/
def laurens_earnings (commercial_rate : ℚ) (subscription_rate : ℚ) 
  (commercial_views : ℕ) (subscriptions : ℕ) : ℚ :=
  commercial_rate * commercial_views + subscription_rate * subscriptions

/-- Theorem: Lauren's earnings on Tuesday -/
theorem tuesday_earnings : 
  laurens_earnings (1/2) 1 100 27 = 77 := by sorry

end NUMINAMATH_CALUDE_tuesday_earnings_l2548_254865


namespace NUMINAMATH_CALUDE_triangle_side_length_l2548_254846

/-- Given a triangle ABC with the following properties:
  * The product of sides a and b is 60√3
  * The sine of angle B equals the sine of angle C
  * The area of the triangle is 15√3
  This theorem states that the length of side b is 2√15 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  a * b = 60 * Real.sqrt 3 →
  Real.sin B = Real.sin C →
  (1/2) * a * b * Real.sin C = 15 * Real.sqrt 3 →
  b = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2548_254846


namespace NUMINAMATH_CALUDE_divisor_problem_l2548_254892

theorem divisor_problem (x : ℝ) (h : x = 1280) : ∃ d : ℝ, (x + 720) / d = 7392 / 462 ∧ d = 125 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2548_254892


namespace NUMINAMATH_CALUDE_square_side_length_l2548_254830

theorem square_side_length (area : ℚ) (h : area = 9 / 16) :
  ∃ side : ℚ, side * side = area ∧ side = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2548_254830


namespace NUMINAMATH_CALUDE_max_product_of_digits_optimal_solution_l2548_254872

-- Define a structure for a four-digit number of the form (abab)
structure FourDigitNumber (a b : Nat) : Type :=
  (a_valid : a > 0 ∧ a < 10)
  (b_valid : b ≥ 0 ∧ b < 10)

def value (n : FourDigitNumber a b) : Nat :=
  1000 * a + 100 * b + 10 * a + b

-- Define the problem
theorem max_product_of_digits (m : FourDigitNumber a b) (n : FourDigitNumber c d) :
  (∃ (t : Nat), value m + value n = t * t) →
  a * b * c * d ≤ 600 := by
  sorry

-- Define the optimality
theorem optimal_solution :
  ∃ (m : FourDigitNumber a b) (n : FourDigitNumber c d),
    (∃ (t : Nat), value m + value n = t * t) ∧
    a * b * c * d = 600 := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_digits_optimal_solution_l2548_254872


namespace NUMINAMATH_CALUDE_clothing_cost_price_l2548_254804

theorem clothing_cost_price 
  (marked_price : ℝ)
  (discount_rate : ℝ)
  (profit_rate : ℝ)
  (h1 : marked_price = 132)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.1)
  : ∃ (cost_price : ℝ), 
    cost_price = 108 ∧ 
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_clothing_cost_price_l2548_254804


namespace NUMINAMATH_CALUDE_butterfly_equal_roots_l2548_254841

/-- A quadratic equation ax^2 + bx + c = 0 is a "butterfly" equation if a - b + c = 0 -/
def is_butterfly_equation (a b c : ℝ) : Prop := a - b + c = 0

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem butterfly_equal_roots (a b c : ℝ) (ha : a ≠ 0) 
  (h_butterfly : is_butterfly_equation a b c) 
  (h_equal_roots : discriminant a b c = 0) : 
  a = c := by sorry

end NUMINAMATH_CALUDE_butterfly_equal_roots_l2548_254841


namespace NUMINAMATH_CALUDE_dot_product_specific_value_l2548_254886

/-- Dot product of two 3D vectors -/
def dot_product (a b c p q r : ℝ) : ℝ := a * p + b * q + c * r

theorem dot_product_specific_value :
  let y : ℝ := 12.5
  let n : ℝ := dot_product 3 4 5 y (-2) 1
  n = 34.5 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_specific_value_l2548_254886


namespace NUMINAMATH_CALUDE_divisibility_transitivity_l2548_254802

theorem divisibility_transitivity (m n k : ℕ+) 
  (h1 : m ^ n.val ∣ n ^ m.val) 
  (h2 : n ^ k.val ∣ k ^ n.val) : 
  m ^ k.val ∣ k ^ m.val := by
  sorry

end NUMINAMATH_CALUDE_divisibility_transitivity_l2548_254802


namespace NUMINAMATH_CALUDE_two_digit_number_rounded_to_3_8_l2548_254890

/-- A number that rounds to 3.8 when rounded to one decimal place -/
def RoundsTo3_8 (n : ℝ) : Prop := 3.75 ≤ n ∧ n < 3.85

/-- The set of two-digit numbers -/
def TwoDigitNumber (n : ℝ) : Prop := 10 ≤ n ∧ n < 100

theorem two_digit_number_rounded_to_3_8 :
  ∃ (max min : ℝ), 
    (∀ n : ℝ, TwoDigitNumber n → RoundsTo3_8 n → n ≤ max) ∧
    (∀ n : ℝ, TwoDigitNumber n → RoundsTo3_8 n → min ≤ n) ∧
    max = 3.84 ∧ min = 3.75 :=
sorry

end NUMINAMATH_CALUDE_two_digit_number_rounded_to_3_8_l2548_254890


namespace NUMINAMATH_CALUDE_max_edge_product_sum_is_9420_l2548_254812

/-- Represents a cube with labeled vertices -/
structure LabeledCube where
  vertices : Fin 8 → ℕ
  is_square_label : ∀ i, ∃ j, vertices i = j^2
  is_permutation : Function.Bijective vertices

/-- Calculates the sum of products of numbers at the ends of each edge -/
def edge_product_sum (cube : LabeledCube) : ℕ := sorry

/-- The maximum possible sum of edge products -/
def max_edge_product_sum : ℕ := 9420

/-- Theorem stating that the maximum sum of edge products is 9420 -/
theorem max_edge_product_sum_is_9420 :
  ∀ cube : LabeledCube, edge_product_sum cube ≤ max_edge_product_sum :=
sorry

end NUMINAMATH_CALUDE_max_edge_product_sum_is_9420_l2548_254812


namespace NUMINAMATH_CALUDE_set_intersection_range_l2548_254845

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}
def B : Set ℝ := {x | x^2 + 4*x = 0}

-- Define the theorem
theorem set_intersection_range (a : ℝ) :
  A a ∩ B = A a → (a ≤ -1 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_set_intersection_range_l2548_254845


namespace NUMINAMATH_CALUDE_square_64_implies_product_63_l2548_254850

theorem square_64_implies_product_63 (m : ℝ) : (m + 2)^2 = 64 → (m + 1) * (m + 3) = 63 := by
  sorry

end NUMINAMATH_CALUDE_square_64_implies_product_63_l2548_254850


namespace NUMINAMATH_CALUDE_negation_of_implication_for_all_negation_of_real_number_implication_l2548_254878

theorem negation_of_implication_for_all (P : ℝ → ℝ → Prop) :
  (¬ ∀ a b : ℝ, P a b) ↔ ∃ a b : ℝ, ¬(P a b) :=
sorry

theorem negation_of_real_number_implication :
  (¬ ∀ a b : ℝ, a = 0 → a * b = 0) ↔ ∃ a b : ℝ, a = 0 ∧ a * b ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_negation_of_implication_for_all_negation_of_real_number_implication_l2548_254878


namespace NUMINAMATH_CALUDE_no_real_solutions_l2548_254855

theorem no_real_solutions :
  ∀ x y : ℝ, x^2 + 3*y^2 - 4*x - 12*y + 28 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2548_254855


namespace NUMINAMATH_CALUDE_green_block_weight_l2548_254877

/-- The weight of the yellow block in pounds -/
def yellow_weight : ℝ := 0.6

/-- The difference in weight between the yellow and green blocks in pounds -/
def weight_difference : ℝ := 0.2

/-- The weight of the green block in pounds -/
def green_weight : ℝ := yellow_weight - weight_difference

theorem green_block_weight :
  green_weight = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_green_block_weight_l2548_254877


namespace NUMINAMATH_CALUDE_one_correct_statement_l2548_254844

theorem one_correct_statement : 
  (∃! n : ℕ, n = 1 ∧ 
    (∀ x : ℤ, x < 0 → x ≤ -1) ∧ 
    (∃ y : ℝ, -(y) ≤ 0) ∧
    (∃ z : ℚ, z = 0) ∧
    (∃ a : ℝ, -a > 0) ∧
    (∃ b₁ b₂ : ℚ, b₁ < 0 ∧ b₂ < 0 ∧ b₁ * b₂ > 0)) :=
by sorry

end NUMINAMATH_CALUDE_one_correct_statement_l2548_254844


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_six_l2548_254898

theorem sum_of_x_and_y_equals_six (x y : ℝ) (h : x^2 + y^2 = 8*x + 4*y - 20) : x + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_six_l2548_254898


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l2548_254869

theorem largest_divisor_of_expression (x : ℤ) (h : Even x) :
  (∃ (k : ℤ), (10*x + 4) * (10*x + 8) * (5*x + 2) = 32 * k) ∧
  (∀ (m : ℤ), m > 32 → ∃ (y : ℤ), Even y ∧ ¬(∃ (l : ℤ), (10*y + 4) * (10*y + 8) * (5*y + 2) = m * l)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l2548_254869


namespace NUMINAMATH_CALUDE_function_satisfies_equation_value_at_sqrt_2014_l2548_254806

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1

/-- The theorem stating that the function f(x) = 1 - x²/2 satisfies the functional equation -/
theorem function_satisfies_equation :
    ∃ f : ℝ → ℝ, FunctionalEquation f ∧ ∀ x : ℝ, f x = 1 - x^2 / 2 := by
  sorry

/-- The value of f(√2014) -/
theorem value_at_sqrt_2014 :
    ∀ f : ℝ → ℝ, FunctionalEquation f → f (Real.sqrt 2014) = -1006 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_value_at_sqrt_2014_l2548_254806


namespace NUMINAMATH_CALUDE_area_third_polygon_l2548_254807

/-- Given three regular polygons inscribed in a circle, where each subsequent polygon
    has twice as many sides as the previous one, the area of the third polygon can be
    expressed in terms of the areas of the first two polygons. -/
theorem area_third_polygon (S₁ S₂ : ℝ) (h₁ : S₁ > 0) (h₂ : S₂ > 0) :
  ∃ (S : ℝ), S = Real.sqrt (2 * S₂^3 / (S₁ + S₂)) :=
sorry

end NUMINAMATH_CALUDE_area_third_polygon_l2548_254807


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2548_254881

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / (d : ℚ) = 3 + 17 / 99 ∧
  n + d = 413 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2548_254881


namespace NUMINAMATH_CALUDE_range_of_a_range_of_x_min_value_ratio_l2548_254891

-- Define the quadratic function
def quadratic (a b x : ℝ) := a * x^2 + b * x + 2

-- Part 1
theorem range_of_a (a b : ℝ) :
  (∀ x, quadratic a b x > 0) →
  quadratic a b 1 = 1 →
  a ∈ Set.Ioo (3 - 2 * Real.sqrt 2) (3 + 2 * Real.sqrt 2) :=
sorry

-- Part 2
theorem range_of_x (a b : ℝ) :
  (∀ a ∈ Set.Icc (-2) (-1), ∀ x, quadratic a b x > 0) →
  quadratic a b 1 = 1 →
  ∃ x ∈ Set.Ioo ((1 - Real.sqrt 17) / 4) ((1 + Real.sqrt 17) / 4),
    quadratic a b x = 0 :=
sorry

-- Part 3
theorem min_value_ratio (a b : ℝ) :
  (∀ x, quadratic a b x ≥ 0) →
  b > 0 →
  (a + 2) / b ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_x_min_value_ratio_l2548_254891


namespace NUMINAMATH_CALUDE_monotonic_function_value_l2548_254880

/-- A monotonically increasing function f: ℝ → ℝ satisfying f(f(x) - 2^x) = 3 for all x ∈ ℝ -/
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (f x - 2^x) = 3)

/-- Theorem: For a monotonically increasing function f satisfying the given condition, f(3) = 9 -/
theorem monotonic_function_value (f : ℝ → ℝ) (h : MonotonicFunction f) : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_value_l2548_254880


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2548_254800

theorem quadratic_equation_equivalence (x : ℝ) : 
  (x + 1)^2 + (x - 2) * (x + 2) = 1 ↔ 2 * x^2 + 2 * x - 4 = 0 :=
by sorry

-- Definitions for the components of the quadratic equation
def quadratic_term (x : ℝ) : ℝ := 2 * x^2
def quadratic_coefficient : ℝ := 2
def linear_term (x : ℝ) : ℝ := 2 * x
def linear_coefficient : ℝ := 2
def constant_term : ℝ := -4

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2548_254800


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2548_254876

theorem cubic_equation_solution (w : ℝ) :
  (w + 5)^3 = (w + 2) * (3 * w^2 + 13 * w + 14) →
  w^3 = -2 * w^2 + (35 / 2) * w + 97 / 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2548_254876


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l2548_254864

/-- Emma's current age -/
def emma_age : ℕ := sorry

/-- Sarah's current age -/
def sarah_age : ℕ := sorry

/-- The number of years until the age ratio becomes 3:2 -/
def years_until_ratio : ℕ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem age_ratio_theorem :
  (emma_age - 3 = 2 * (sarah_age - 3)) →  -- Three years ago, Emma was twice as old as Sarah
  (emma_age - 5 = 3 * (sarah_age - 5)) →  -- Five years ago, Emma was three times as old as Sarah
  (emma_age + years_until_ratio) * 2 = 3 * (sarah_age + years_until_ratio) →  -- Future ratio condition
  years_until_ratio = 1 := by  -- The result we want to prove
  sorry


end NUMINAMATH_CALUDE_age_ratio_theorem_l2548_254864


namespace NUMINAMATH_CALUDE_solution_difference_l2548_254889

theorem solution_difference (m n : ℝ) : 
  (m - 4) * (m + 4) = 24 * m - 96 →
  (n - 4) * (n + 4) = 24 * n - 96 →
  m ≠ n →
  m > n →
  m - n = 16 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l2548_254889


namespace NUMINAMATH_CALUDE_function_root_implies_parameter_range_l2548_254832

theorem function_root_implies_parameter_range :
  ∀ a : ℝ,
  (∃ c ∈ Set.Icc (-2 : ℝ) 1, 2 * c - a + 1 = 0) →
  a ∈ Set.Icc (-3 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_function_root_implies_parameter_range_l2548_254832


namespace NUMINAMATH_CALUDE_quadradois_theorem_l2548_254854

/-- A number is quadradois if a square can be divided into that many squares of at most two different sizes. -/
def IsQuadradois (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a * a + b * b = n ∧ (a = 0 ∨ b = 0 ∨ a ≠ b)

theorem quadradois_theorem :
  IsQuadradois 6 ∧ 
  IsQuadradois 2015 ∧ 
  ∀ n : ℕ, n > 5 → IsQuadradois n :=
by sorry

end NUMINAMATH_CALUDE_quadradois_theorem_l2548_254854


namespace NUMINAMATH_CALUDE_point_returns_after_seven_steps_l2548_254814

/-- Represents a point in a triangle -/
structure TrianglePoint where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : TrianglePoint
  B : TrianglePoint
  C : TrianglePoint

/-- Represents the movement of a point parallel to a side of the triangle -/
def moveParallel (start : TrianglePoint) (side : Triangle → TrianglePoint × TrianglePoint) : TrianglePoint := sorry

/-- Represents the sequence of movements in the triangle -/
def moveSequence (start : TrianglePoint) (triangle : Triangle) : TrianglePoint := 
  let step1 := moveParallel start (λ t => (t.B, t.C))
  let step2 := moveParallel step1 (λ t => (t.A, t.B))
  let step3 := moveParallel step2 (λ t => (t.C, t.A))
  let step4 := moveParallel step3 (λ t => (t.B, t.C))
  let step5 := moveParallel step4 (λ t => (t.A, t.B))
  let step6 := moveParallel step5 (λ t => (t.C, t.A))
  moveParallel step6 (λ t => (t.B, t.C))

/-- The main theorem stating that the point returns to its original position after 7 steps -/
theorem point_returns_after_seven_steps (triangle : Triangle) (start : TrianglePoint) :
  moveSequence start triangle = start := sorry

end NUMINAMATH_CALUDE_point_returns_after_seven_steps_l2548_254814


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l2548_254837

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The symmetric point about the x-axis -/
def symmetricAboutXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem symmetric_point_x_axis :
  let P : Point := { x := -1, y := 2 }
  symmetricAboutXAxis P = { x := -1, y := -2 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l2548_254837


namespace NUMINAMATH_CALUDE_min_cookie_count_l2548_254822

def is_valid_cookie_count (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + 21 * b ∧ n % 13 = 0

theorem min_cookie_count : 
  (∀ m : ℕ, m > 0 ∧ m < 52 → ¬(is_valid_cookie_count m)) ∧
  is_valid_cookie_count 52 :=
sorry

end NUMINAMATH_CALUDE_min_cookie_count_l2548_254822


namespace NUMINAMATH_CALUDE_unique_number_power_l2548_254843

/-- A function that checks if a number is a four-digit number with 3 as the first digit and 5 as the last digit -/
def isFourDigitThreeFive (n : ℕ) : Prop :=
  n ≥ 3000 ∧ n < 4000 ∧ n % 10 = 5

/-- The theorem stating that 55 is the only number that, when raised to some even power, 
    results in a four-digit number with 3 as the first digit and 5 as the last digit -/
theorem unique_number_power : 
  ∃! (x : ℕ), ∃ (k : ℕ), k > 0 ∧ isFourDigitThreeFive (x^(2*k)) ∧ x = 55 :=
sorry

end NUMINAMATH_CALUDE_unique_number_power_l2548_254843


namespace NUMINAMATH_CALUDE_division_problem_l2548_254801

theorem division_problem (y : ℚ) : y ≠ 0 → (6 / y) * 12 = 12 → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2548_254801


namespace NUMINAMATH_CALUDE_park_distance_l2548_254828

theorem park_distance (d : ℝ) 
  (alice_false : ¬(d ≥ 8))
  (bob_false : ¬(d ≤ 7))
  (charlie_false : ¬(d ≤ 6)) :
  7 < d ∧ d < 8 := by
sorry

end NUMINAMATH_CALUDE_park_distance_l2548_254828


namespace NUMINAMATH_CALUDE_equal_digit_probability_l2548_254815

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The number of one-digit numbers on each die -/
def one_digit_count : ℕ := 9

/-- The number of two-digit numbers on each die -/
def two_digit_count : ℕ := 11

/-- The probability of rolling an equal number of one-digit and two-digit numbers -/
def equal_digit_prob : ℚ := 539055 / 1600000

theorem equal_digit_probability :
  let p_one_digit : ℚ := one_digit_count / num_sides
  let p_two_digit : ℚ := two_digit_count / num_sides
  (num_dice.choose (num_dice / 2)) * p_one_digit ^ (num_dice / 2) * p_two_digit ^ (num_dice / 2) = equal_digit_prob := by
  sorry

end NUMINAMATH_CALUDE_equal_digit_probability_l2548_254815


namespace NUMINAMATH_CALUDE_investment_sum_l2548_254803

/-- 
Given a sum P invested at simple interest for two years, 
if the difference in interest between 18% p.a. and 12% p.a. is Rs. 840, 
then P = 7000.
-/
theorem investment_sum (P : ℝ) : 
  (P * (18 / 100) * 2 - P * (12 / 100) * 2 = 840) → P = 7000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l2548_254803


namespace NUMINAMATH_CALUDE_equation_solutions_l2548_254885

def solution_set : Set (ℤ × ℤ) :=
  {(-15, -3), (-1, -1), (2, 14), (3, -21), (5, -7), (6, -6), (20, -4)}

def satisfies_equation (pair : ℤ × ℤ) : Prop :=
  let (x, y) := pair
  x ≠ 0 ∧ y ≠ 0 ∧ (5 : ℚ) / x - (7 : ℚ) / y = 2

theorem equation_solutions :
  ∀ (x y : ℤ), satisfies_equation (x, y) ↔ (x, y) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2548_254885


namespace NUMINAMATH_CALUDE_problem_solution_l2548_254820

theorem problem_solution (m : ℝ) (h : m + 1/m = 5) : m^2 + 1/m^2 + 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2548_254820


namespace NUMINAMATH_CALUDE_milburg_grownups_l2548_254857

/-- The number of grown-ups in Milburg -/
def num_grownups (total_population children : ℕ) : ℕ :=
  total_population - children

/-- Theorem: The number of grown-ups in Milburg is 5256 -/
theorem milburg_grownups :
  num_grownups 8243 2987 = 5256 := by
  sorry

end NUMINAMATH_CALUDE_milburg_grownups_l2548_254857


namespace NUMINAMATH_CALUDE_tuesday_grading_percentage_l2548_254858

theorem tuesday_grading_percentage 
  (total_exams : ℕ) 
  (monday_percentage : ℚ) 
  (wednesday_exams : ℕ) : 
  total_exams = 120 → 
  monday_percentage = 60 / 100 → 
  wednesday_exams = 12 → 
  (((total_exams : ℚ) - (monday_percentage * total_exams) - wednesday_exams) / 
   ((total_exams : ℚ) - (monday_percentage * total_exams))) = 75 / 100 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_grading_percentage_l2548_254858


namespace NUMINAMATH_CALUDE_angle_AMD_is_45_degrees_l2548_254860

/-- A rectangle with sides 8 and 4 -/
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (is_rectangle : sorry)
  (AB_length : dist A B = 8)
  (BC_length : dist B C = 4)

/-- A point on side AB of the rectangle -/
def point_on_AB (rect : Rectangle) (M : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • rect.A + t • rect.B

/-- The angle AMD -/
def angle_AMD (rect : Rectangle) (M : ℝ × ℝ) : ℝ := sorry

/-- The angle CMD -/
def angle_CMD (rect : Rectangle) (M : ℝ × ℝ) : ℝ := sorry

/-- The theorem to be proved -/
theorem angle_AMD_is_45_degrees (rect : Rectangle) (M : ℝ × ℝ) 
  (h1 : point_on_AB rect M) 
  (h2 : angle_AMD rect M = angle_CMD rect M) : 
  angle_AMD rect M = 45 := by sorry

end NUMINAMATH_CALUDE_angle_AMD_is_45_degrees_l2548_254860


namespace NUMINAMATH_CALUDE_sodium_in_salt_calculation_l2548_254859

/-- The amount of sodium (in mg) per teaspoon of salt -/
def sodium_per_tsp_salt : ℝ := 50

/-- The amount of sodium (in mg) per oz of parmesan cheese -/
def sodium_per_oz_parmesan : ℝ := 25

/-- The number of teaspoons of salt in the recipe -/
def tsp_salt_in_recipe : ℝ := 2

/-- The number of oz of parmesan cheese in the original recipe -/
def oz_parmesan_in_recipe : ℝ := 8

/-- The reduction in oz of parmesan cheese to achieve 1/3 sodium reduction -/
def oz_parmesan_reduction : ℝ := 4

theorem sodium_in_salt_calculation : 
  sodium_per_tsp_salt = 50 := by sorry

end NUMINAMATH_CALUDE_sodium_in_salt_calculation_l2548_254859
