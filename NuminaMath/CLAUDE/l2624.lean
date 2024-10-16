import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2624_262454

theorem equation_solution :
  ∃! x : ℚ, x + 5/6 = 11/18 - 2/9 ∧ x = -4/9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2624_262454


namespace NUMINAMATH_CALUDE_complement_of_16_51_l2624_262404

/-- Represents an angle in degrees and minutes -/
structure DegreeMinute where
  degrees : ℕ
  minutes : ℕ

/-- Calculates the complement of an angle given in degrees and minutes -/
def complement (angle : DegreeMinute) : DegreeMinute :=
  let totalMinutes := 90 * 60 - (angle.degrees * 60 + angle.minutes)
  { degrees := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem complement_of_16_51 :
  complement { degrees := 16, minutes := 51 } = { degrees := 73, minutes := 9 } := by
  sorry

end NUMINAMATH_CALUDE_complement_of_16_51_l2624_262404


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2624_262402

/-- Definition of the hyperbola with given foci and passing point -/
def Hyperbola (f : ℝ × ℝ) (p : ℝ × ℝ) :=
  {(x, y) : ℝ × ℝ | 
    |((x - f.1)^2 + (y - f.2)^2).sqrt - ((x - f.1)^2 + (y + f.2)^2).sqrt| = 
    |((p.1 - f.1)^2 + (p.2 - f.2)^2).sqrt - ((p.1 - f.1)^2 + (p.2 + f.2)^2).sqrt|}

/-- The theorem stating the equation of the hyperbola -/
theorem hyperbola_equation :
  let f : ℝ × ℝ := (0, 3)
  let p : ℝ × ℝ := (Real.sqrt 15, 4)
  ∀ (x y : ℝ), (x, y) ∈ Hyperbola f p ↔ y^2 / 4 - x^2 / 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2624_262402


namespace NUMINAMATH_CALUDE_ones_digit_of_13_power_l2624_262481

-- Define a function to get the ones digit of a natural number
def ones_digit (n : ℕ) : ℕ := n % 10

-- Define the exponent
def exponent : ℕ := 13 * (7^7)

-- Theorem statement
theorem ones_digit_of_13_power : ones_digit (13^exponent) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_13_power_l2624_262481


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2624_262464

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 2 - 3 * Complex.I) * 
               (Real.sqrt 5 + 5 * Complex.I) * 
               (2 - 2 * Complex.I)) = 18 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2624_262464


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l2624_262484

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, y^5 + 2*x*y = x^2 + 2*y^4 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l2624_262484


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_minus_one_l2624_262486

theorem imaginary_part_of_i_minus_one (i : ℂ) (h : i * i = -1) :
  Complex.im (i - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_minus_one_l2624_262486


namespace NUMINAMATH_CALUDE_N_is_composite_l2624_262437

/-- The number formed by k+1 ones and k zeros in between -/
def N (k : ℕ) : ℕ := 10^(k+1) + 1

/-- Theorem stating that N(k) is composite for k > 1 -/
theorem N_is_composite (k : ℕ) (h : k > 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N k = a * b :=
sorry

end NUMINAMATH_CALUDE_N_is_composite_l2624_262437


namespace NUMINAMATH_CALUDE_susan_money_problem_l2624_262478

theorem susan_money_problem (S : ℚ) :
  S - (S / 6 + S / 8 + S * (30 / 100) + 100) = 480 →
  S = 1420 := by
sorry

end NUMINAMATH_CALUDE_susan_money_problem_l2624_262478


namespace NUMINAMATH_CALUDE_cube_face_perimeter_l2624_262421

theorem cube_face_perimeter (volume : ℝ) (perimeter : ℝ) : 
  volume = 125 → perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_perimeter_l2624_262421


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2624_262441

-- Define the hyperbola
structure Hyperbola where
  asymptote_slope : ℝ

-- Define eccentricity
def eccentricity (h : Hyperbola) : Set ℝ :=
  {e : ℝ | e = 2 ∨ e = (2 * Real.sqrt 3) / 3}

-- Theorem statement
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_asymptote : h.asymptote_slope = Real.sqrt 3) : 
  ∃ e : ℝ, e ∈ eccentricity h := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2624_262441


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2624_262477

theorem polynomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2624_262477


namespace NUMINAMATH_CALUDE_parity_equality_of_extrema_l2624_262495

/-- A set of elements related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The maximum element of A_P -/
def max_element (A : Set ℤ) : ℤ := sorry

/-- The minimum element of A_P -/
def min_element (A : Set ℤ) : ℤ := sorry

/-- Parity of an integer -/
def parity (n : ℤ) : Bool := n % 2 = 0

/-- Theorem: The parity of the smallest and largest elements of A_P is the same -/
theorem parity_equality_of_extrema :
  parity (min_element A_P) = parity (max_element A_P) := by
  sorry

end NUMINAMATH_CALUDE_parity_equality_of_extrema_l2624_262495


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2624_262427

theorem smallest_integer_with_remainders : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 6 = 5) ∧ 
  (a % 8 = 7) ∧ 
  (∀ b : ℕ, b > 0 → b % 6 = 5 → b % 8 = 7 → a ≤ b) ∧
  (a = 23) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2624_262427


namespace NUMINAMATH_CALUDE_pyramid_height_proof_l2624_262460

/-- The height of a square-based pyramid with base edge length 10 units, 
    which has the same volume as a cube with edge length 5 units. -/
def pyramid_height : ℝ := 3.75

theorem pyramid_height_proof :
  let cube_edge : ℝ := 5
  let pyramid_base : ℝ := 10
  let cube_volume : ℝ := cube_edge ^ 3
  let pyramid_volume (h : ℝ) : ℝ := (1/3) * pyramid_base ^ 2 * h
  pyramid_volume pyramid_height = cube_volume :=
by sorry

end NUMINAMATH_CALUDE_pyramid_height_proof_l2624_262460


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l2624_262461

theorem quadratic_equation_transformation (p q : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -3 ∧ ∀ x : ℝ, x^2 + p*x + q = 0 ↔ (x = x₁ ∨ x = x₂)) →
  ∀ x : ℝ, x^2 + p*x + q = 0 ↔ (x - 2)*(x + 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l2624_262461


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2624_262446

theorem pure_imaginary_complex_number (m : ℝ) : 
  let i : ℂ := Complex.I
  let Z : ℂ := (1 + i) / (1 - i) + m * (1 - i)
  (Z.re = 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2624_262446


namespace NUMINAMATH_CALUDE_roots_equation_s_value_l2624_262409

theorem roots_equation_s_value (n r : ℝ) (c d : ℝ) : 
  (c^2 - n*c + 6 = 0) → 
  (d^2 - n*d + 6 = 0) → 
  ((c + 2/d)^2 - r*(c + 2/d) + s = 0) → 
  ((d + 2/c)^2 - r*(d + 2/c) + s = 0) → 
  (s = 32/3) := by
sorry

end NUMINAMATH_CALUDE_roots_equation_s_value_l2624_262409


namespace NUMINAMATH_CALUDE_factorable_p_values_l2624_262456

def is_factorable (p : ℤ) : Prop :=
  ∃ (a b : ℤ), ∀ x, x^2 + p*x + 12 = (x + a) * (x + b)

theorem factorable_p_values (p : ℤ) :
  is_factorable p ↔ p ∈ ({-13, -8, -7, 7, 8, 13} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_factorable_p_values_l2624_262456


namespace NUMINAMATH_CALUDE_vector_operation_l2624_262407

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![2, -2]

theorem vector_operation : 2 • a - b = ![2, 4] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l2624_262407


namespace NUMINAMATH_CALUDE_judy_shopping_cost_l2624_262480

-- Define the prices and quantities
def carrot_price : ℝ := 1.50
def carrot_quantity : ℕ := 6
def milk_price : ℝ := 3.50
def milk_quantity : ℕ := 4
def pineapple_price : ℝ := 5.00
def pineapple_quantity : ℕ := 3
def pineapple_discount : ℝ := 0.25
def flour_price : ℝ := 6.00
def flour_quantity : ℕ := 3
def flour_discount : ℝ := 0.10
def ice_cream_price : ℝ := 8.00
def coupon_value : ℝ := 10.00
def coupon_threshold : ℝ := 50.00

-- Define the theorem
theorem judy_shopping_cost :
  let carrot_total := carrot_price * carrot_quantity
  let milk_total := milk_price * milk_quantity
  let pineapple_total := pineapple_price * (1 - pineapple_discount) * pineapple_quantity
  let flour_total := flour_price * (1 - flour_discount) * flour_quantity
  let subtotal := carrot_total + milk_total + pineapple_total + flour_total + ice_cream_price
  let final_total := if subtotal ≥ coupon_threshold then subtotal - coupon_value else subtotal
  final_total = 48.45 := by sorry

end NUMINAMATH_CALUDE_judy_shopping_cost_l2624_262480


namespace NUMINAMATH_CALUDE_vieta_relation_l2624_262444

/-- The quadratic equation x^2 - x - 1 = 0 --/
def quadratic_equation (x : ℝ) : Prop := x^2 - x - 1 = 0

/-- Definition of S_n --/
def S (n : ℕ) (M N : ℝ) : ℝ := M^n + N^n

/-- Theorem: Relationship between S_n, S_{n-1}, and S_{n-2} --/
theorem vieta_relation (M N : ℝ) (h : quadratic_equation M ∧ quadratic_equation N) :
  ∀ n ≥ 3, S n M N = S (n-1) M N + S (n-2) M N :=
sorry

end NUMINAMATH_CALUDE_vieta_relation_l2624_262444


namespace NUMINAMATH_CALUDE_expression_simplification_l2624_262410

theorem expression_simplification (m : ℝ) (hm : m = 2) : 
  (((m + 1) / (m - 1) + 1) / ((m + m^2) / (m^2 - 2*m + 1))) - ((2 - 2*m) / (m^2 - 1)) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2624_262410


namespace NUMINAMATH_CALUDE_overlap_area_theorem_l2624_262417

-- Define the square ABCD
def square_side : ℝ := 8

-- Define the rectangle WXYZ
def rect_length : ℝ := 12
def rect_width : ℝ := 8

-- Define the theorem
theorem overlap_area_theorem (shaded_area : ℝ) (AP : ℝ) :
  -- Conditions
  shaded_area = (rect_length * rect_width) / 2 →
  shaded_area = (square_side - AP) * square_side →
  -- Conclusion
  AP = 2 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_theorem_l2624_262417


namespace NUMINAMATH_CALUDE_range_of_m_l2624_262440

theorem range_of_m (x m : ℝ) : 
  (∀ x, (|x - 3| ≤ 2 → (x - m + 1) * (x - m - 1) ≤ 0) ∧
        ((x < 1 ∨ x > 5) → (x < m - 1 ∨ x > m + 1)) ∧
        (∃ x, (x < m - 1 ∨ x > m + 1) ∧ ¬(x < 1 ∨ x > 5)))
  → 2 ≤ m ∧ m ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2624_262440


namespace NUMINAMATH_CALUDE_log_equation_solution_l2624_262426

theorem log_equation_solution (x : ℝ) :
  x > 0 → ((Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 4 ↔ x ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2624_262426


namespace NUMINAMATH_CALUDE_optimal_rectangle_dimensions_l2624_262498

/-- The total length of the fence in meters -/
def fence_length : ℝ := 60

/-- The area of the rectangle as a function of its width -/
def area (x : ℝ) : ℝ := x * (fence_length - 2 * x)

/-- The width that maximizes the area -/
def optimal_width : ℝ := 15

/-- The length that maximizes the area -/
def optimal_length : ℝ := 30

theorem optimal_rectangle_dimensions :
  (∀ x : ℝ, 0 < x → x < fence_length / 2 → area x ≤ area optimal_width) ∧
  optimal_length = fence_length - 2 * optimal_width :=
sorry

end NUMINAMATH_CALUDE_optimal_rectangle_dimensions_l2624_262498


namespace NUMINAMATH_CALUDE_m_range_theorem_l2624_262416

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem m_range_theorem (f : ℝ → ℝ) (m : ℝ) 
  (h_odd : is_odd_function f)
  (h_prop : ∀ x, f (3/4 * x) = f (3/4 * x))
  (h_lower_bound : ∀ x, f x > -2)
  (h_f_1 : f 1 = -3/m) :
  (0 < m ∧ m < 3) ∨ m < -1 := by
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l2624_262416


namespace NUMINAMATH_CALUDE_one_fifth_equals_point_two_l2624_262479

theorem one_fifth_equals_point_two : (1 : ℚ) / 5 = (2 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_equals_point_two_l2624_262479


namespace NUMINAMATH_CALUDE_four_numbers_product_2002_sum_less_40_l2624_262458

theorem four_numbers_product_2002_sum_less_40 (a b c d : ℕ+) :
  a * b * c * d = 2002 ∧ a + b + c + d < 40 →
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
  (a = 2 ∧ b = 7 ∧ c = 13 ∧ d = 11) ∨ (a = 1 ∧ b = 14 ∧ c = 13 ∧ d = 11) ∨
  (a = 2 ∧ b = 11 ∧ c = 7 ∧ d = 13) ∨ (a = 1 ∧ b = 11 ∧ c = 14 ∧ d = 13) ∨
  (a = 2 ∧ b = 11 ∧ c = 13 ∧ d = 7) ∨ (a = 1 ∧ b = 11 ∧ c = 13 ∧ d = 14) ∨
  (a = 2 ∧ b = 13 ∧ c = 7 ∧ d = 11) ∨ (a = 1 ∧ b = 13 ∧ c = 14 ∧ d = 11) ∨
  (a = 2 ∧ b = 13 ∧ c = 11 ∧ d = 7) ∨ (a = 1 ∧ b = 13 ∧ c = 11 ∧ d = 14) :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_product_2002_sum_less_40_l2624_262458


namespace NUMINAMATH_CALUDE_function_increasing_decreasing_implies_m_range_l2624_262442

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

-- State the theorem
theorem function_increasing_decreasing_implies_m_range :
  ∀ m : ℝ, 
  (∀ x ≥ 2, ∀ y ≥ 2, x < y → f m x < f m y) ∧ 
  (∀ x ≤ 1, ∀ y ≤ 1, x < y → f m x > f m y) →
  8 ≤ m ∧ m ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_function_increasing_decreasing_implies_m_range_l2624_262442


namespace NUMINAMATH_CALUDE_cube_root_of_product_rewrite_cube_root_l2624_262443

theorem cube_root_of_product (a b c : ℕ) : 
  (a^9 * b^3 * c^3 : ℝ)^(1/3) = (a^3 * b * c : ℝ) :=
by sorry

theorem rewrite_cube_root : (2^9 * 5^3 * 7^3 : ℝ)^(1/3) = 280 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_product_rewrite_cube_root_l2624_262443


namespace NUMINAMATH_CALUDE_jill_phone_time_l2624_262445

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem jill_phone_time : geometric_sum 5 2 5 = 155 := by
  sorry

end NUMINAMATH_CALUDE_jill_phone_time_l2624_262445


namespace NUMINAMATH_CALUDE_digit_configuration_impossible_l2624_262406

/-- Represents a configuration of digits on a shape with 6 segments -/
structure DigitConfiguration :=
  (digits : Finset ℕ)
  (segments : Finset (Finset ℕ))

/-- The property that all segments have the same sum -/
def has_equal_segment_sums (config : DigitConfiguration) : Prop :=
  ∃ (sum : ℕ), ∀ segment ∈ config.segments, (segment.sum id = sum)

/-- The main theorem stating the impossibility of the configuration -/
theorem digit_configuration_impossible : 
  ¬ ∃ (config : DigitConfiguration), 
    (config.digits = Finset.range 10) ∧ 
    (config.segments.card = 6) ∧
    (∀ segment ∈ config.segments, segment.card = 3) ∧
    (has_equal_segment_sums config) :=
sorry

end NUMINAMATH_CALUDE_digit_configuration_impossible_l2624_262406


namespace NUMINAMATH_CALUDE_calculate_fifth_subject_score_l2624_262411

/-- Given a student's scores in 4 subjects and the average of all 5 subjects,
    calculate the score in the 5th subject. -/
theorem calculate_fifth_subject_score
  (math_score science_score english_score biology_score : ℕ)
  (average_score : ℚ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 62)
  (h4 : biology_score = 85)
  (h5 : average_score = 74)
  : ∃ (social_studies_score : ℕ),
    (math_score + science_score + english_score + biology_score + social_studies_score : ℚ) / 5 = average_score ∧
    social_studies_score = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_calculate_fifth_subject_score_l2624_262411


namespace NUMINAMATH_CALUDE_equation_to_lines_l2624_262433

theorem equation_to_lines (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) :=
sorry

end NUMINAMATH_CALUDE_equation_to_lines_l2624_262433


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l2624_262435

def total_balls : ℕ := 15
def red_balls : ℕ := 7
def blue_balls : ℕ := 8

theorem probability_two_red_balls :
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l2624_262435


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2624_262449

theorem sphere_surface_area (d : ℝ) (h : d = 4) : 
  4 * Real.pi * (d / 2)^2 = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2624_262449


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2624_262487

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x > 0, x^2 - 5*x + 6 = 0) ↔ (∀ x > 0, x^2 - 5*x + 6 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2624_262487


namespace NUMINAMATH_CALUDE_xy_max_value_l2624_262474

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + y = 2) :
  xy ≤ (1 : ℝ) / 2 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 2 ∧ x₀ * y₀ = (1 : ℝ) / 2 :=
sorry

end NUMINAMATH_CALUDE_xy_max_value_l2624_262474


namespace NUMINAMATH_CALUDE_log_difference_equals_four_l2624_262492

theorem log_difference_equals_four (a : ℝ) (h : a > 0) :
  Real.log (100 * a) / Real.log 10 - Real.log (a / 100) / Real.log 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_four_l2624_262492


namespace NUMINAMATH_CALUDE_dance_club_average_age_l2624_262448

theorem dance_club_average_age 
  (num_females : Nat) 
  (num_males : Nat) 
  (avg_age_females : ℝ) 
  (avg_age_males : ℝ) 
  (h1 : num_females = 12)
  (h2 : num_males = 18)
  (h3 : avg_age_females = 25)
  (h4 : avg_age_males = 40) :
  let total_people := num_females + num_males
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  total_age / total_people = 34 := by
  sorry

end NUMINAMATH_CALUDE_dance_club_average_age_l2624_262448


namespace NUMINAMATH_CALUDE_average_visitors_is_276_l2624_262405

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitors (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let numSundays := 4
  let numOtherDays := 26
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / 30

/-- Theorem stating that the average number of visitors is 276 given the specified conditions -/
theorem average_visitors_is_276 :
  averageVisitors 510 240 = 276 := by
  sorry

#eval averageVisitors 510 240

end NUMINAMATH_CALUDE_average_visitors_is_276_l2624_262405


namespace NUMINAMATH_CALUDE_min_value_and_inequality_range_l2624_262428

def f (x : ℝ) : ℝ := |2*x - 1| + |x + 2|

theorem min_value_and_inequality_range :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 5/2) ∧
  (∀ (a b x : ℝ), a ≠ 0 → |2*b - a| + |b + 2*a| ≥ |a| * (|x + 1| + |x - 1|) → -5/4 ≤ x ∧ x ≤ 5/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_range_l2624_262428


namespace NUMINAMATH_CALUDE_al_sewing_time_l2624_262468

theorem al_sewing_time (allison_time : ℝ) (together_time : ℝ) (allison_remaining_time : ℝ) :
  allison_time = 9 →
  together_time = 3 →
  allison_remaining_time = 3.75 →
  ∃ al_time : ℝ,
    al_time = 12 ∧
    together_time * (1 / allison_time + 1 / al_time) + allison_remaining_time * (1 / allison_time) = 1 :=
by sorry

end NUMINAMATH_CALUDE_al_sewing_time_l2624_262468


namespace NUMINAMATH_CALUDE_calculation_proof_l2624_262465

theorem calculation_proof : 
  (((15^15 / 15^10)^3 * 5^6) / 25^2) = 3^15 * 5^17 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2624_262465


namespace NUMINAMATH_CALUDE_toothpick_pattern_l2624_262414

/-- Given an arithmetic sequence with first term 4 and common difference 4,
    the 150th term is equal to 600. -/
theorem toothpick_pattern (a : ℕ) (d : ℕ) (n : ℕ) :
  a = 4 → d = 4 → n = 150 → a + (n - 1) * d = 600 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_pattern_l2624_262414


namespace NUMINAMATH_CALUDE_extreme_point_of_g_l2624_262457

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x - 1

noncomputable def g (x : ℝ) : ℝ := x * (f 1 x) + (1/2) * x^2 + 2 * x

def has_unique_extreme_point (h : ℝ → ℝ) (m : ℤ) : Prop :=
  ∃! (x : ℝ), m < x ∧ x < m + 1 ∧ 
  (∀ y ∈ Set.Ioo m (m + 1), h y ≤ h x) ∨ 
  (∀ y ∈ Set.Ioo m (m + 1), h y ≥ h x)

theorem extreme_point_of_g :
  ∃ m : ℤ, has_unique_extreme_point g m → m = 0 ∨ m = 3 :=
sorry

end NUMINAMATH_CALUDE_extreme_point_of_g_l2624_262457


namespace NUMINAMATH_CALUDE_prob_at_least_four_girls_l2624_262490

-- Define the number of children
def num_children : ℕ := 6

-- Define the probability of a child being a girl
def prob_girl : ℚ := 1/2

-- Define the function to calculate the probability of at least k girls out of n children
def prob_at_least_k_girls (n k : ℕ) : ℚ :=
  sorry

theorem prob_at_least_four_girls :
  prob_at_least_k_girls num_children 4 = 11/32 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_four_girls_l2624_262490


namespace NUMINAMATH_CALUDE_octagon_area_in_circle_l2624_262472

theorem octagon_area_in_circle (R : ℝ) : 
  R > 0 → 
  (4 * (1/2 * R^2 * Real.sin (π/4)) + 4 * (1/2 * R^2 * Real.sin (π/2))) = R^2 * (Real.sqrt 2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_in_circle_l2624_262472


namespace NUMINAMATH_CALUDE_min_cut_length_for_non_triangle_l2624_262491

/-- Given three sticks of lengths 9, 18, and 21 inches, this theorem proves that
    the minimum integral length that can be cut from each stick to prevent
    the remaining pieces from forming a triangle is 6 inches. -/
theorem min_cut_length_for_non_triangle : ∃ (x : ℕ),
  (∀ y : ℕ, y < x → (9 - y) + (18 - y) > 21 - y) ∧
  (9 - x) + (18 - x) ≤ 21 - x ∧
  x = 6 :=
sorry

end NUMINAMATH_CALUDE_min_cut_length_for_non_triangle_l2624_262491


namespace NUMINAMATH_CALUDE_angle_trigonometry_l2624_262420

open Real

theorem angle_trigonometry (x : ℝ) (h1 : π/2 < x) (h2 : x < π) 
  (h3 : cos x = tan x) (h4 : sin x ≠ cos x) : 
  sin x = (-1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_trigonometry_l2624_262420


namespace NUMINAMATH_CALUDE_f_negative_two_lt_f_one_l2624_262494

/-- A function f : ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The derivative of f for positive x -/
def DerivativePositive (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, deriv f x = (x - 1) * (x - 2)

theorem f_negative_two_lt_f_one
  (f : ℝ → ℝ)
  (heven : IsEven f)
  (hderiv : DerivativePositive f) :
  f (-2) < f 1 :=
sorry

end NUMINAMATH_CALUDE_f_negative_two_lt_f_one_l2624_262494


namespace NUMINAMATH_CALUDE_seating_theorem_l2624_262400

/-- The number of ways to seat 9 children (5 sons and 4 daughters) in a row
    such that at least two girls are next to each other. -/
def seating_arrangements (total : ℕ) (sons : ℕ) (daughters : ℕ) : ℕ :=
  Nat.factorial total - (Nat.factorial sons * Nat.factorial daughters)

theorem seating_theorem :
  seating_arrangements 9 5 4 = 359400 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l2624_262400


namespace NUMINAMATH_CALUDE_negation_equivalence_l2624_262422

theorem negation_equivalence (x y : ℝ) : 
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2624_262422


namespace NUMINAMATH_CALUDE_cone_height_from_cube_l2624_262485

/-- The height of a cone formed by melting a cube -/
theorem cone_height_from_cube (cube_edge : ℝ) (cone_base_area : ℝ) (cone_height : ℝ) : 
  cube_edge = 6 →
  cone_base_area = 54 →
  (cube_edge ^ 3) = (1 / 3) * cone_base_area * cone_height →
  cone_height = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_cone_height_from_cube_l2624_262485


namespace NUMINAMATH_CALUDE_irina_square_area_l2624_262499

/-- Given a square with side length 12 cm, if another square has a perimeter 8 cm larger,
    then the area of the second square is 196 cm². -/
theorem irina_square_area (original_side : ℝ) (irina_side : ℝ) : 
  original_side = 12 →
  4 * irina_side = 4 * original_side + 8 →
  irina_side * irina_side = 196 :=
by
  sorry

#check irina_square_area

end NUMINAMATH_CALUDE_irina_square_area_l2624_262499


namespace NUMINAMATH_CALUDE_middle_zero_between_zero_and_one_l2624_262463

/-- The cubic function f(x) = x^3 - 4x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 4*x + a

/-- Theorem: For 0 < a < 2, if f(x) has three zeros x₁ < x₂ < x₃, then 0 < x₂ < 1 -/
theorem middle_zero_between_zero_and_one (a : ℝ) (x₁ x₂ x₃ : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hzeros : f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0)
  (horder : x₁ < x₂ ∧ x₂ < x₃) :
  0 < x₂ ∧ x₂ < 1 := by
  sorry


end NUMINAMATH_CALUDE_middle_zero_between_zero_and_one_l2624_262463


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_seven_sixths_l2624_262412

theorem negative_sixty_four_to_seven_sixths (x : ℝ) : x = (-64)^(7/6) → x = -16384 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_seven_sixths_l2624_262412


namespace NUMINAMATH_CALUDE_sqrt_sum_ge_product_sum_l2624_262467

theorem sqrt_sum_ge_product_sum {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 3) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ a * b + b * c + c * a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_ge_product_sum_l2624_262467


namespace NUMINAMATH_CALUDE_not_divide_power_plus_one_l2624_262430

theorem not_divide_power_plus_one (p q m : ℕ) : 
  Nat.Prime p → Nat.Prime q → Odd p → Odd q → p > q → m > 0 → ¬(p * q ∣ m^(p - q) + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divide_power_plus_one_l2624_262430


namespace NUMINAMATH_CALUDE_inequality_proof_l2624_262470

theorem inequality_proof (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a^2 + b^2 = 4) :
  (a * b) / (a + b + 2) ≤ Real.sqrt 2 - 1 ∧
  ((a * b) / (a + b + 2) = Real.sqrt 2 - 1 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2) :=
by sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l2624_262470


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2624_262447

theorem polynomial_division_theorem (a b : ℝ) : 
  (∃ (P : ℝ → ℝ), (fun X => a * X^4 + b * X^3 + 1) = fun X => (X - 1)^2 * P X) → 
  a = 3 ∧ b = -4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2624_262447


namespace NUMINAMATH_CALUDE_circle_triangle_angle_measure_l2624_262455

-- Define the circle and triangle
def Circle : Type := Unit
def Point : Type := Unit
def Triangle : Type := Unit

-- Define the center of the circle
def center (c : Circle) : Point := sorry

-- Define the vertices of the triangle
def X (t : Triangle) : Point := sorry
def Y (t : Triangle) : Point := sorry
def Z (t : Triangle) : Point := sorry

-- Define the property of being circumscribed
def is_circumscribed (c : Circle) (t : Triangle) : Prop := sorry

-- Define the measure of an angle
def angle_measure (p q r : Point) : ℝ := sorry

-- State the theorem
theorem circle_triangle_angle_measure 
  (c : Circle) (t : Triangle) (h_circumscribed : is_circumscribed c t) 
  (h_XOY : angle_measure (X t) (center c) (Y t) = 120)
  (h_YOZ : angle_measure (Y t) (center c) (Z t) = 140) :
  angle_measure (X t) (Y t) (Z t) = 60 := by sorry

end NUMINAMATH_CALUDE_circle_triangle_angle_measure_l2624_262455


namespace NUMINAMATH_CALUDE_pizza_geometric_sum_l2624_262497

/-- The sum of a geometric series with first term 1/4, common ratio 1/2, and 6 terms -/
def geometricSum : ℚ :=
  let a : ℚ := 1/4
  let r : ℚ := 1/2
  let n : ℕ := 6
  a * (1 - r^n) / (1 - r)

/-- The fraction of pizza eaten after 6 trips -/
def pizzaEaten : ℚ := 63/128

theorem pizza_geometric_sum :
  geometricSum = pizzaEaten := by sorry

end NUMINAMATH_CALUDE_pizza_geometric_sum_l2624_262497


namespace NUMINAMATH_CALUDE_max_new_sum_is_406_l2624_262482

/-- Represents a sum of two-digit numbers -/
structure TwoDigitSum where
  D : ℕ  -- Sum of tens digits
  U : ℕ  -- Sum of units digits

/-- The property that a TwoDigitSum represents 100 -/
def represents_100 (s : TwoDigitSum) : Prop :=
  10 * s.D + s.U = 100

/-- The new sum after swapping digits -/
def new_sum (s : TwoDigitSum) : ℕ :=
  100 + 9 * (s.U - s.D)

/-- The theorem stating that the maximum new sum is 406 -/
theorem max_new_sum_is_406 :
  ∀ s : TwoDigitSum, represents_100 s → new_sum s ≤ 406 :=
by sorry

end NUMINAMATH_CALUDE_max_new_sum_is_406_l2624_262482


namespace NUMINAMATH_CALUDE_binomial_18_10_l2624_262488

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 43758 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l2624_262488


namespace NUMINAMATH_CALUDE_common_tangent_bisection_l2624_262425

-- Define the basic geometric objects
variable (Circle₁ Circle₂ : Type) [MetricSpace Circle₁] [MetricSpace Circle₂]
variable (A B : ℝ × ℝ)  -- Intersection points of the circles
variable (M N : ℝ × ℝ)  -- Points of tangency on the common tangent

-- Define the property of being a point on a circle
def OnCircle (p : ℝ × ℝ) (circle : Type) [MetricSpace circle] : Prop := sorry

-- Define the property of being a tangent line to a circle
def IsTangent (p q : ℝ × ℝ) (circle : Type) [MetricSpace circle] : Prop := sorry

-- Define the property of a line bisecting another line segment
def Bisects (p q r s : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem common_tangent_bisection 
  (hA₁ : OnCircle A Circle₁) (hA₂ : OnCircle A Circle₂)
  (hB₁ : OnCircle B Circle₁) (hB₂ : OnCircle B Circle₂)
  (hM₁ : OnCircle M Circle₁) (hN₂ : OnCircle N Circle₂)
  (hMN₁ : IsTangent M N Circle₁) (hMN₂ : IsTangent M N Circle₂) :
  Bisects A B M N := by sorry

end NUMINAMATH_CALUDE_common_tangent_bisection_l2624_262425


namespace NUMINAMATH_CALUDE_smallest_m_plus_n_l2624_262483

def is_multiple_of_100 (x : ℕ) : Prop := ∃ k : ℕ, x = 100 * k

theorem smallest_m_plus_n : 
  ∀ m n : ℕ, m > n → is_multiple_of_100 (4^m + 4^n) → 
  ∀ p q : ℕ, p > q → is_multiple_of_100 (4^p + 4^q) → m + n ≤ p + q → m + n = 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_plus_n_l2624_262483


namespace NUMINAMATH_CALUDE_letter_distribution_l2624_262434

/-- The number of ways to distribute n distinct objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- There are 5 distinct letters -/
def num_letters : ℕ := 5

/-- There are 3 distinct mailboxes -/
def num_mailboxes : ℕ := 3

/-- The number of ways to distribute 5 letters into 3 mailboxes is 3^5 -/
theorem letter_distribution : distribute num_letters num_mailboxes = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_letter_distribution_l2624_262434


namespace NUMINAMATH_CALUDE_average_of_numbers_l2624_262419

def numbers : List ℝ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : (numbers.sum / numbers.length : ℝ) = 114391.81818181818 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l2624_262419


namespace NUMINAMATH_CALUDE_cube_surface_area_l2624_262413

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 1) : 
  6 * edge_length^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2624_262413


namespace NUMINAMATH_CALUDE_unknown_number_proof_l2624_262489

theorem unknown_number_proof (x : ℝ) : 
  (x + 30 + 50) / 3 = (20 + 40 + 6) / 3 + 8 → x = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l2624_262489


namespace NUMINAMATH_CALUDE_marys_final_book_count_marys_library_end_year_l2624_262418

/-- Calculates the final number of books in Mary's mystery book library after a year of changes. -/
theorem marys_final_book_count (initial : ℕ) (book_club : ℕ) (bookstore : ℕ) (yard_sales : ℕ) 
  (daughter : ℕ) (mother : ℕ) (donated : ℕ) (sold : ℕ) : ℕ :=
  initial + book_club + bookstore + yard_sales + daughter + mother - donated - sold

/-- Proves that Mary has 81 books at the end of the year given the specific conditions. -/
theorem marys_library_end_year : 
  marys_final_book_count 72 12 5 2 1 4 12 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_marys_final_book_count_marys_library_end_year_l2624_262418


namespace NUMINAMATH_CALUDE_rectangle_area_constraint_l2624_262452

theorem rectangle_area_constraint (m : ℝ) : 
  (3 * m + 8) * (m - 3) = 70 → m = (1 + Real.sqrt 1129) / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_constraint_l2624_262452


namespace NUMINAMATH_CALUDE_no_negative_roots_l2624_262459

theorem no_negative_roots :
  ∀ x : ℝ, x < 0 → x^4 - 3*x^3 - 2*x^2 - 4*x + 1 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_negative_roots_l2624_262459


namespace NUMINAMATH_CALUDE_square_plot_area_l2624_262432

theorem square_plot_area (perimeter : ℝ) (h1 : perimeter * 55 = 3740) : 
  (perimeter / 4) ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_square_plot_area_l2624_262432


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2624_262439

-- Define the quadratic polynomial
def q (x : ℚ) : ℚ := (29 * x^2 - 44 * x + 135) / 15

-- State the theorem
theorem quadratic_polynomial_satisfies_conditions :
  q (-1) = 6 ∧ q 2 = 1 ∧ q 4 = 17 := by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2624_262439


namespace NUMINAMATH_CALUDE_car_length_is_113_steps_l2624_262408

/-- Represents the scenario of a person jogging alongside a moving car --/
structure CarJoggingScenario where
  /-- The length of the car in terms of the jogger's steps --/
  car_length : ℝ
  /-- The distance the car moves during one step of the jogger --/
  car_step : ℝ
  /-- The number of steps counted when jogging from rear to front --/
  steps_rear_to_front : ℕ
  /-- The number of steps counted when jogging from front to rear --/
  steps_front_to_rear : ℕ
  /-- The car is moving faster than the jogger --/
  car_faster : car_step > 0
  /-- The car has a positive length --/
  car_positive : car_length > 0
  /-- Equation for jogging from rear to front --/
  eq_rear_to_front : (steps_rear_to_front : ℝ) = car_length / car_step + steps_rear_to_front
  /-- Equation for jogging from front to rear --/
  eq_front_to_rear : (steps_front_to_rear : ℝ) = car_length / car_step - steps_front_to_rear

/-- The length of the car is 113 steps when jogging 150 steps rear to front and 30 steps front to rear --/
theorem car_length_is_113_steps (scenario : CarJoggingScenario) 
  (h1 : scenario.steps_rear_to_front = 150) 
  (h2 : scenario.steps_front_to_rear = 30) : 
  scenario.car_length = 113 := by
  sorry

end NUMINAMATH_CALUDE_car_length_is_113_steps_l2624_262408


namespace NUMINAMATH_CALUDE_plot_length_l2624_262424

theorem plot_length (breadth : ℝ) (perimeter : ℝ) :
  (breadth + 22 = breadth + 22) →  -- Length is 22 more than breadth
  (perimeter = 4 * breadth + 44) →  -- Perimeter formula
  (26.50 * perimeter = 5300) →  -- Fencing cost
  (breadth + 22 = 61) :=  -- Length is 61 meters
by sorry

end NUMINAMATH_CALUDE_plot_length_l2624_262424


namespace NUMINAMATH_CALUDE_problem_solution_l2624_262466

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x^a

-- Define an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the specific function f(x) = lg(x + √(x^2 + 1))
noncomputable def f (x : ℝ) : ℝ := lg (x + Real.sqrt (x^2 + 1))

theorem problem_solution :
  (isPowerFunction (λ _ : ℝ => 1)) ∧
  (∀ g : ℝ → ℝ, isOddFunction g → g 0 = 0) ∧
  (isOddFunction f) ∧
  (∃ a : ℝ, a < 0 ∧ (a^2)^(3/2) ≠ a^3) ∧
  (∃! x : ℝ, (λ _ : ℝ => 1) x = 0 → False) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2624_262466


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2624_262496

theorem polynomial_division_remainder : ∃ q r : Polynomial ℝ, 
  (X^3 - 2 : Polynomial ℝ) = (X^2 - 2) * q + r ∧ 
  r.degree < (X^2 - 2).degree ∧ 
  r = 2*X - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2624_262496


namespace NUMINAMATH_CALUDE_mall_computer_sales_l2624_262451

theorem mall_computer_sales (planned_sales : ℕ) (golden_week_avg : ℕ) (increase_percent : ℕ) (remaining_days : ℕ) :
  planned_sales = 900 →
  golden_week_avg = 54 →
  increase_percent = 30 →
  remaining_days = 24 →
  (∃ x : ℕ, x ≥ 33 ∧ golden_week_avg * 7 + x * remaining_days ≥ planned_sales + planned_sales * increase_percent / 100) :=
by
  sorry

#check mall_computer_sales

end NUMINAMATH_CALUDE_mall_computer_sales_l2624_262451


namespace NUMINAMATH_CALUDE_third_participant_score_l2624_262401

/-- Represents the score of a participant -/
structure ParticipantScore where
  score : ℕ

/-- Represents the total number of competitions -/
def totalCompetitions : ℕ := 10

/-- Represents the total points awarded in each competition -/
def pointsPerCompetition : ℕ := 4

/-- Theorem: Given the conditions of the competition and two participants' scores,
    the third participant's score is determined -/
theorem third_participant_score 
  (dima misha : ParticipantScore)
  (h1 : dima.score = 22)
  (h2 : misha.score = 8) :
  ∃ yura : ParticipantScore, yura.score = 10 := by
  sorry

end NUMINAMATH_CALUDE_third_participant_score_l2624_262401


namespace NUMINAMATH_CALUDE_f_minimum_value_l2624_262469

noncomputable def f (x : ℝ) : ℝ := x + 2/x + 1/(x + 2/x)

theorem f_minimum_value (x : ℝ) (h : x > 0) : f x ≥ 5 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_value_l2624_262469


namespace NUMINAMATH_CALUDE_cardinality_of_star_product_l2624_262475

def P : Finset ℕ := {3, 4, 5}
def Q : Finset ℕ := {4, 5, 6, 7}

def star_product (P Q : Finset ℕ) : Finset (ℕ × ℕ) :=
  Finset.product P Q

theorem cardinality_of_star_product :
  Finset.card (star_product P Q) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cardinality_of_star_product_l2624_262475


namespace NUMINAMATH_CALUDE_cube_fraction_equals_150_l2624_262476

theorem cube_fraction_equals_150 :
  (68^3 - 65^3) * (32^3 + 18^3) / ((32^2 - 32 * 18 + 18^2) * (68^2 + 68 * 65 + 65^2)) = 150 := by
  sorry

end NUMINAMATH_CALUDE_cube_fraction_equals_150_l2624_262476


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l2624_262471

def a (n : ℕ) : ℚ := (1 - 2 * n^2) / (2 + 4 * n^2)

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-1/2)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l2624_262471


namespace NUMINAMATH_CALUDE_lcm_812_smallest_lcm_812_24_smallest_lcm_812_672_l2624_262473

theorem lcm_812_smallest (n : ℕ) : n > 0 ∧ Nat.lcm 812 n = 672 → n ≥ 24 := by
  sorry

theorem lcm_812_24 : Nat.lcm 812 24 = 672 := by
  sorry

theorem smallest_lcm_812_672 : ∃ (n : ℕ), n > 0 ∧ Nat.lcm 812 n = 672 ∧ ∀ (m : ℕ), m > 0 → Nat.lcm 812 m = 672 → m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_lcm_812_smallest_lcm_812_24_smallest_lcm_812_672_l2624_262473


namespace NUMINAMATH_CALUDE_lemonade_recipe_correct_l2624_262462

/-- Represents the ratio of ingredients in the lemonade mixture -/
structure LemonadeRatio where
  water : ℕ
  lemon_juice : ℕ

/-- Converts gallons to quarts -/
def gallons_to_quarts (gallons : ℕ) : ℕ := 4 * gallons

/-- Calculates the amount of each ingredient needed for a given total volume -/
def ingredient_amount (ratio : LemonadeRatio) (total_volume : ℕ) (ingredient : ℕ) : ℕ :=
  (ingredient * total_volume) / (ratio.water + ratio.lemon_juice)

theorem lemonade_recipe_correct (ratio : LemonadeRatio) (total_gallons : ℕ) :
  ratio.water = 5 →
  ratio.lemon_juice = 3 →
  total_gallons = 2 →
  let total_quarts := gallons_to_quarts total_gallons
  ingredient_amount ratio total_quarts ratio.water = 5 ∧
  ingredient_amount ratio total_quarts ratio.lemon_juice = 3 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_recipe_correct_l2624_262462


namespace NUMINAMATH_CALUDE_circle_center_l2624_262403

/-- The equation of a circle C in the form x^2 + y^2 + ax + by + c = 0 -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The center of a circle -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a circle C with equation x^2 + y^2 - 2x + y + 1/4 = 0,
    its center is the point (1, -1/2) -/
theorem circle_center (C : Circle) 
  (h : C = { a := -2, b := 1, c := 1/4 }) : 
  ∃ center : Point, center = { x := 1, y := -1/2 } :=
sorry

end NUMINAMATH_CALUDE_circle_center_l2624_262403


namespace NUMINAMATH_CALUDE_triangle_properties_l2624_262438

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ) -- angles
  (a b c : ℝ) -- opposite sides

-- Define the conditions and theorems
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = Real.sqrt 7)
  (h2 : Real.sin t.A = Real.sqrt 3 * Real.sin t.C) :
  (t.B = π / 6 → Real.sin t.B = Real.sin t.C) ∧ 
  (t.B > π / 2 ∧ Real.cos (2 * t.B) = 1 / 2 → 
    t.a * Real.sin t.C = Real.sqrt 21 / 14) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l2624_262438


namespace NUMINAMATH_CALUDE_fire_truck_ladder_height_l2624_262431

theorem fire_truck_ladder_height (distance_to_building : ℝ) (ladder_length : ℝ) :
  distance_to_building = 5 →
  ladder_length = 13 →
  ∃ (height : ℝ), height^2 + distance_to_building^2 = ladder_length^2 ∧ height = 12 :=
by sorry

end NUMINAMATH_CALUDE_fire_truck_ladder_height_l2624_262431


namespace NUMINAMATH_CALUDE_travel_time_percentage_l2624_262423

/-- Given the total travel time and the time from Ngapara to Zipra,
    prove that the time from Ningi to Zipra is 80% of the time from Ngapara to Zipra. -/
theorem travel_time_percentage (total_time ngapara_zipra_time : ℝ)
  (h1 : total_time = 108)
  (h2 : ngapara_zipra_time = 60) :
  (total_time - ngapara_zipra_time) / ngapara_zipra_time = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_percentage_l2624_262423


namespace NUMINAMATH_CALUDE_tub_drain_time_l2624_262493

/-- Given a tub that drains at a constant rate, this function calculates the additional time
    needed to empty the tub completely after a certain fraction has been drained. -/
def additional_drain_time (initial_fraction : ℚ) (initial_time : ℚ) : ℚ :=
  let remaining_fraction := 1 - initial_fraction
  let drain_rate := initial_fraction / initial_time
  remaining_fraction / drain_rate

/-- Theorem stating that for a tub draining 5/7 of its content in 4 minutes,
    it will take an additional 11.2 minutes to empty completely. -/
theorem tub_drain_time : additional_drain_time (5/7) 4 = 11.2 := by
  sorry

end NUMINAMATH_CALUDE_tub_drain_time_l2624_262493


namespace NUMINAMATH_CALUDE_common_chord_equation_l2624_262436

/-- The equation of the line where the common chord of two circles lies -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 - 2*x - 8 = 0) ∧ (x^2 + y^2 + 2*x - 4*y - 4 = 0) →
  (x - y + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2624_262436


namespace NUMINAMATH_CALUDE_cannot_retile_after_replacement_l2624_262453

-- Define a type for tiles
inductive Tile
| OneByFour : Tile
| TwoByTwo : Tile

-- Define a type for a tiling of a rectangle
structure Tiling :=
  (width : ℕ)
  (height : ℕ)
  (tiles : List Tile)

-- Define a function to check if a tiling is valid
def isValidTiling (t : Tiling) : Prop :=
  -- Add conditions for a valid tiling
  sorry

-- Define a function to replace one 2×2 tile with a 1×4 tile
def replaceTile (t : Tiling) : Tiling :=
  -- Implement the replacement logic
  sorry

-- Theorem statement
theorem cannot_retile_after_replacement (t : Tiling) :
  isValidTiling t → ¬(isValidTiling (replaceTile t)) :=
by
  sorry

end NUMINAMATH_CALUDE_cannot_retile_after_replacement_l2624_262453


namespace NUMINAMATH_CALUDE_min_abs_a_plus_b_l2624_262415

theorem min_abs_a_plus_b (a b : ℤ) (h1 : |a| < |b|) (h2 : |b| ≤ 4) :
  ∃ (m : ℤ), (∀ (x y : ℤ), |x| < |y| → |y| ≤ 4 → m ≤ |x| + y) ∧ m = -4 :=
sorry

end NUMINAMATH_CALUDE_min_abs_a_plus_b_l2624_262415


namespace NUMINAMATH_CALUDE_correct_calculation_l2624_262429

theorem correct_calculation (x : ℤ) : x + 19 = 50 → 16 * x = 496 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2624_262429


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_theorem_l2624_262450

-- Define a monic cubic polynomial with real coefficients
def monicCubicPolynomial (a b c : ℝ) : ℝ → ℂ :=
  fun x => (x : ℂ)^3 + a * (x : ℂ)^2 + b * (x : ℂ) + c

-- State the theorem
theorem monic_cubic_polynomial_theorem (a b c : ℝ) :
  let q := monicCubicPolynomial a b c
  (q (3 - 2*I) = 0 ∧ q 0 = -108) →
  a = -(186/13) ∧ b = 1836/13 ∧ c = -108 :=
by sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_theorem_l2624_262450
