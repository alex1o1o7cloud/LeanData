import Mathlib

namespace NUMINAMATH_CALUDE_new_average_rent_l3706_370605

/-- Calculates the new average rent per person after one person's rent is increased -/
theorem new_average_rent (num_friends : ℕ) (initial_average : ℚ) (increased_rent : ℚ) (increase_percentage : ℚ) : 
  num_friends = 4 →
  initial_average = 800 →
  increased_rent = 1250 →
  increase_percentage = 16 / 100 →
  (num_friends * initial_average - increased_rent + increased_rent * (1 + increase_percentage)) / num_friends = 850 :=
by sorry

end NUMINAMATH_CALUDE_new_average_rent_l3706_370605


namespace NUMINAMATH_CALUDE_smallest_among_given_numbers_l3706_370694

theorem smallest_among_given_numbers :
  let a := 1
  let b := Real.sqrt 2 / 2
  let c := Real.sqrt 3 / 3
  let d := Real.sqrt 5 / 5
  d < c ∧ d < b ∧ d < a := by
  sorry

end NUMINAMATH_CALUDE_smallest_among_given_numbers_l3706_370694


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l3706_370627

/-- Given 6 people in an elevator with an average weight of 156 lbs,
    if a 7th person enters and the new average weight becomes 151 lbs,
    then the weight of the 7th person is 121 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight new_avg_weight : ℝ) :
  initial_people = 6 →
  initial_avg_weight = 156 →
  new_avg_weight = 151 →
  (initial_people * initial_avg_weight + (initial_people + 1) * new_avg_weight - initial_people * new_avg_weight) = 121 :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l3706_370627


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_linear_l3706_370636

/-- The system of equations y = x^2 and y = 2x + k has exactly one solution if and only if k = -1 -/
theorem unique_solution_quadratic_linear (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = p.1^2 ∧ p.2 = 2 * p.1 + k) ↔ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_linear_l3706_370636


namespace NUMINAMATH_CALUDE_non_coincident_terminal_sides_l3706_370666

def has_coincident_terminal_sides (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

theorem non_coincident_terminal_sides :
  ¬ has_coincident_terminal_sides 1050 (-300) :=
by sorry

end NUMINAMATH_CALUDE_non_coincident_terminal_sides_l3706_370666


namespace NUMINAMATH_CALUDE_fraction_simplification_l3706_370678

theorem fraction_simplification : (2 + 4) / (1 + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3706_370678


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3706_370658

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 + (5 + 1/2)x - 1/2 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/2
def c : ℚ := -1/2

theorem quadratic_discriminant : discriminant a b c = 161/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3706_370658


namespace NUMINAMATH_CALUDE_exponent_equality_comparison_l3706_370653

theorem exponent_equality_comparison : 
  (4^3 ≠ 3^4) ∧ 
  (-5^3 = (-5)^3) ∧ 
  ((-6)^2 ≠ -6^2) ∧ 
  (((-5/2)^2 : ℚ) ≠ ((-2/5)^2 : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_exponent_equality_comparison_l3706_370653


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_chord_l3706_370683

-- Define the given line
def givenLine (x y : ℝ) : Prop := 2 * x + 3 * y + 1 = 0

-- Define the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 4 * y = 0

-- Define the perpendicular bisector
def perpendicularBisector (x y : ℝ) : Prop := 3 * x - 2 * y - 7 = 0

-- Theorem statement
theorem perpendicular_bisector_of_chord (A B : ℝ × ℝ) :
  givenLine A.1 A.2 ∧ givenLine B.1 B.2 ∧
  givenCircle A.1 A.2 ∧ givenCircle B.1 B.2 →
  ∃ (M : ℝ × ℝ), perpendicularBisector M.1 M.2 ∧
    (M.1 - A.1)^2 + (M.2 - A.2)^2 = (M.1 - B.1)^2 + (M.2 - B.2)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_chord_l3706_370683


namespace NUMINAMATH_CALUDE_sin_45_degrees_l3706_370692

theorem sin_45_degrees : Real.sin (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l3706_370692


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3706_370644

theorem polynomial_simplification (x : ℝ) :
  (5 * x^10 + 8 * x^9 + 3 * x^8) + (2 * x^12 + 3 * x^10 + x^9 + 4 * x^8 + 6 * x^4 + 7 * x^2 + 9) =
  2 * x^12 + 8 * x^10 + 9 * x^9 + 7 * x^8 + 6 * x^4 + 7 * x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3706_370644


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_longest_obtuse_triangle_longest_side_l3706_370616

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_angles : 0 < α ∧ 0 < β ∧ 0 < γ
  h_sum_angles : α + β + γ = π

-- Theorem for right-angled triangle
theorem right_triangle_hypotenuse_longest (t : Triangle) (h_right : t.γ = π/2) :
  t.c ≥ t.a ∧ t.c ≥ t.b := by sorry

-- Theorem for obtuse-angled triangle
theorem obtuse_triangle_longest_side (t : Triangle) (h_obtuse : t.γ > π/2) :
  t.c > t.a ∧ t.c > t.b := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_longest_obtuse_triangle_longest_side_l3706_370616


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3706_370687

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b : ℝ, b ≥ 0 → (a + 1)^2 + b ≥ 0) ∧ 
  (∃ a b : ℝ, (a + 1)^2 + b ≥ 0 ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3706_370687


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3706_370691

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  n : ℕ  -- number of terms
  first_sum : ℕ  -- sum of first 4 terms
  last_sum : ℕ  -- sum of last 4 terms
  total_sum : ℕ  -- sum of all terms

/-- The theorem stating the properties of the specific arithmetic sequence -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.first_sum = 26)
  (h2 : seq.last_sum = 110)
  (h3 : seq.total_sum = 187) :
  seq.n = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3706_370691


namespace NUMINAMATH_CALUDE_range_of_m_when_g_has_three_zeros_l3706_370650

/-- The quadratic function f(x) -/
def f (x : ℝ) : ℝ := x^2 + x - 2

/-- The function g(x) -/
def g (m : ℝ) (x : ℝ) : ℝ := |f x| - f x - 2*m*x - 2*m^2

/-- The theorem stating the range of m when g(x) has three distinct zeros -/
theorem range_of_m_when_g_has_three_zeros :
  ∀ m : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g m x = 0 ∧ g m y = 0 ∧ g m z = 0) →
  m ∈ Set.Ioo ((1 - 2*Real.sqrt 7)/3) (-1) ∪ Set.Ioo 2 ((1 + 2*Real.sqrt 7)/3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_when_g_has_three_zeros_l3706_370650


namespace NUMINAMATH_CALUDE_robyn_cookie_sales_l3706_370610

theorem robyn_cookie_sales (total_sales lucy_sales : ℕ) 
  (h1 : total_sales = 76) 
  (h2 : lucy_sales = 29) : 
  total_sales - lucy_sales = 47 := by
  sorry

end NUMINAMATH_CALUDE_robyn_cookie_sales_l3706_370610


namespace NUMINAMATH_CALUDE_simplify_sqrt_neg_seven_squared_l3706_370663

theorem simplify_sqrt_neg_seven_squared : Real.sqrt ((-7)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_neg_seven_squared_l3706_370663


namespace NUMINAMATH_CALUDE_find_r_l3706_370620

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_r_l3706_370620


namespace NUMINAMATH_CALUDE_dining_bill_share_l3706_370699

theorem dining_bill_share (people : ℕ) (bill tip_percent tax_percent : ℚ) 
  (h_people : people = 15)
  (h_bill : bill = 350)
  (h_tip_percent : tip_percent = 18 / 100)
  (h_tax_percent : tax_percent = 5 / 100) :
  (bill + bill * tip_percent + bill * tax_percent) / people = 287 / 10 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_share_l3706_370699


namespace NUMINAMATH_CALUDE_average_book_width_l3706_370621

def book_widths : List ℝ := [3, 7.5, 1.25, 0.75, 4, 12]

theorem average_book_width : 
  (book_widths.sum / book_widths.length : ℝ) = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_average_book_width_l3706_370621


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3706_370682

theorem partial_fraction_decomposition (M₁ M₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 2 → (45*x - 31) / (x^2 - 3*x + 2) = M₁ / (x - 1) + M₂ / (x - 2)) →
  M₁ * M₂ = -826 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3706_370682


namespace NUMINAMATH_CALUDE_greatest_4digit_base9_divisible_by_7_l3706_370606

/-- Converts a base 9 number to decimal --/
def base9ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 9 --/
def decimalToBase9 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 4-digit base 9 number --/
def is4DigitBase9 (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 8888

theorem greatest_4digit_base9_divisible_by_7 :
  ∃ n : ℕ, is4DigitBase9 n ∧
           base9ToDecimal n % 7 = 0 ∧
           ∀ m : ℕ, is4DigitBase9 m ∧ base9ToDecimal m % 7 = 0 → m ≤ n ∧
           n = 9000 :=
sorry

end NUMINAMATH_CALUDE_greatest_4digit_base9_divisible_by_7_l3706_370606


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3706_370690

/-- Given a rectangle with width s and height h, where increasing the width by 3 and
    decreasing the height by 3 doesn't change the area, prove that decreasing the width
    by 4 and increasing the height by 4 results in an area decrease of 28 square units. -/
theorem rectangle_area_change (s h : ℝ) (h_area : (s + 3) * (h - 3) = s * h) :
  s * h - (s - 4) * (h + 4) = 28 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3706_370690


namespace NUMINAMATH_CALUDE_difference_of_fractions_difference_for_7000_l3706_370686

theorem difference_of_fractions (n : ℝ) : n * (1 / 10) - n * (1 / 1000) = n * (99 / 1000) :=
by sorry

theorem difference_for_7000 : 7000 * (1 / 10) - 7000 * (1 / 1000) = 693 :=
by sorry

end NUMINAMATH_CALUDE_difference_of_fractions_difference_for_7000_l3706_370686


namespace NUMINAMATH_CALUDE_rectangular_strip_dimensions_l3706_370677

theorem rectangular_strip_dimensions (a b c : ℕ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43 →
  (a = 1 ∧ b + c = 22) ∨ (a = 22 ∧ b + c = 1) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_strip_dimensions_l3706_370677


namespace NUMINAMATH_CALUDE_g_expression_l3706_370679

-- Define polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom poly_sum : ∀ x, f x + g x = 2 * x^2 + 3 * x + 4
axiom f_def : ∀ x, f x = 2 * x^3 - x^2 - 4 * x + 5

-- State the theorem
theorem g_expression : ∀ x, g x = -2 * x^3 + 3 * x^2 + 7 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l3706_370679


namespace NUMINAMATH_CALUDE_car_speed_proof_l3706_370649

/-- Proves that a car's speed is 36 km/h given the conditions of the problem -/
theorem car_speed_proof (v : ℝ) : v > 0 →
  (1 / v) * 3600 = (1 / 40) * 3600 + 10 → v = 36 :=
by
  sorry

#check car_speed_proof

end NUMINAMATH_CALUDE_car_speed_proof_l3706_370649


namespace NUMINAMATH_CALUDE_distance_between_points_l3706_370631

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, -2)
  let p2 : ℝ × ℝ := (8, 8)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 136 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l3706_370631


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l3706_370669

theorem rectangle_shorter_side
  (area : ℝ) (perimeter : ℝ)
  (h_area : area = 117)
  (h_perimeter : perimeter = 44)
  : ∃ (length width : ℝ),
    length * width = area ∧
    2 * (length + width) = perimeter ∧
    min length width = 9 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l3706_370669


namespace NUMINAMATH_CALUDE_jenny_peanut_butter_cookies_l3706_370673

theorem jenny_peanut_butter_cookies :
  ∀ (jenny_pb : ℕ) (jenny_cc marcus_pb marcus_lemon : ℕ),
    jenny_cc = 50 →
    marcus_pb = 30 →
    marcus_lemon = 20 →
    jenny_pb + marcus_pb = jenny_cc + marcus_lemon →
    jenny_pb = 40 := by
  sorry

end NUMINAMATH_CALUDE_jenny_peanut_butter_cookies_l3706_370673


namespace NUMINAMATH_CALUDE_equation_solution_l3706_370638

theorem equation_solution : 
  ∃ (x₁ x₂ : ℚ), x₁ = 7/15 ∧ x₂ = 4/5 ∧ 
  (∀ x : ℚ, ⌊(5 + 6*x)/8⌋ = (15*x - 7)/5 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3706_370638


namespace NUMINAMATH_CALUDE_larger_number_l3706_370604

theorem larger_number (x y : ℕ) (h1 : y = x + 10) (h2 : x + y = 34) : y = 22 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l3706_370604


namespace NUMINAMATH_CALUDE_square_sum_theorem_l3706_370635

theorem square_sum_theorem (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 8) :
  x^2 + y^2 = 33 := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l3706_370635


namespace NUMINAMATH_CALUDE_simplify_expression_l3706_370633

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3706_370633


namespace NUMINAMATH_CALUDE_mans_rate_l3706_370660

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 26)
  (h2 : speed_against_stream = 12) :
  (speed_with_stream + speed_against_stream) / 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_l3706_370660


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3706_370609

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let a : ℝ := -24
  let b : ℝ := 72
  let c : ℝ := -120
  let sum_of_roots := -b / a
  (a * x^2 + b * x + c = 0) → sum_of_roots = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3706_370609


namespace NUMINAMATH_CALUDE_unique_row_with_53_in_pascal_triangle_l3706_370681

theorem unique_row_with_53_in_pascal_triangle :
  ∃! n : ℕ, ∃ k : ℕ, k ≤ n ∧ Nat.choose n k = 53 :=
by sorry

end NUMINAMATH_CALUDE_unique_row_with_53_in_pascal_triangle_l3706_370681


namespace NUMINAMATH_CALUDE_downstream_distance_is_36_l3706_370632

/-- Represents the swimming scenario --/
structure SwimmingScenario where
  still_water_speed : ℝ
  upstream_distance : ℝ
  swim_time : ℝ

/-- Calculates the downstream distance given a swimming scenario --/
def downstream_distance (s : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating that the downstream distance is 36 km for the given scenario --/
theorem downstream_distance_is_36 (s : SwimmingScenario) 
  (h1 : s.still_water_speed = 9)
  (h2 : s.upstream_distance = 18)
  (h3 : s.swim_time = 3) : 
  downstream_distance s = 36 :=
sorry

end NUMINAMATH_CALUDE_downstream_distance_is_36_l3706_370632


namespace NUMINAMATH_CALUDE_triangle_ratio_equality_l3706_370602

/-- Given a triangle with side length a, height h_a corresponding to side a,
    inradius r, and semiperimeter p, prove that (2p / a) = (h_a / r) -/
theorem triangle_ratio_equality (a h_a r p : ℝ) (h_positive : a > 0 ∧ h_a > 0 ∧ r > 0 ∧ p > 0)
  (h_area_inradius : p * r = (1/2) * a * h_a) : 
  (2 * p / a) = (h_a / r) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_equality_l3706_370602


namespace NUMINAMATH_CALUDE_newspaper_printing_time_l3706_370698

/-- Represents the time taken to print newspapers -/
def print_time (presses : ℕ) (newspapers : ℕ) : ℚ :=
  6 * (4 : ℚ) * newspapers / (8000 * presses)

theorem newspaper_printing_time :
  print_time 2 6000 = 9 :=
by sorry

end NUMINAMATH_CALUDE_newspaper_printing_time_l3706_370698


namespace NUMINAMATH_CALUDE_cosine_axis_of_symmetry_l3706_370656

/-- The axis of symmetry for a cosine function translated to the left by π/6 units -/
theorem cosine_axis_of_symmetry (k : ℤ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.cos (2 * (x + π / 6))
  ∃ (x : ℝ), x = -π / 6 + k * π / 2 ∧ 
    (∀ (y : ℝ), f (x - y) = f (x + y)) :=
sorry

end NUMINAMATH_CALUDE_cosine_axis_of_symmetry_l3706_370656


namespace NUMINAMATH_CALUDE_circle_radii_sum_l3706_370670

theorem circle_radii_sum : ∀ r : ℝ, 
  r > 0 →
  (r - 4)^2 + r^2 = (r + 2)^2 →
  ∃ r' : ℝ, r' > 0 ∧ (r' - 4)^2 + r'^2 = (r' + 2)^2 ∧ r + r' = 12 :=
by sorry

end NUMINAMATH_CALUDE_circle_radii_sum_l3706_370670


namespace NUMINAMATH_CALUDE_square_difference_equality_l3706_370607

theorem square_difference_equality : 30^2 - 2*(30*5) + 5^2 = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3706_370607


namespace NUMINAMATH_CALUDE_fraction_equality_l3706_370662

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 2) 
  (h2 : s / u = 11 / 7) : 
  (5 * p * s - 3 * q * u) / (7 * q * u - 2 * p * s) = -233 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3706_370662


namespace NUMINAMATH_CALUDE_circle_center_l3706_370671

/-- The center of a circle given by the equation 3x^2 - 6x + 3y^2 + 12y - 75 = 0 is (1, -2) -/
theorem circle_center (x y : ℝ) : 
  (3 * x^2 - 6 * x + 3 * y^2 + 12 * y - 75 = 0) → 
  (∃ r : ℝ, (x - 1)^2 + (y - (-2))^2 = r^2) := by
sorry

end NUMINAMATH_CALUDE_circle_center_l3706_370671


namespace NUMINAMATH_CALUDE_paintable_area_theorem_l3706_370623

/-- Calculates the total paintable area of rooms with given dimensions and unpaintable area -/
def totalPaintableArea (numRooms length width height unpaintableArea : ℕ) : ℕ :=
  let wallArea := 2 * (length * height + width * height)
  let paintableAreaPerRoom := wallArea - unpaintableArea
  numRooms * paintableAreaPerRoom

/-- Theorem stating that the total paintable area of 4 rooms with given dimensions is 1644 sq ft -/
theorem paintable_area_theorem :
  totalPaintableArea 4 15 12 9 75 = 1644 := by
  sorry

end NUMINAMATH_CALUDE_paintable_area_theorem_l3706_370623


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3706_370672

theorem sum_of_two_numbers (x y : ℝ) : 
  y = 2 * x + 3 ∧ y = 19 → x + y = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3706_370672


namespace NUMINAMATH_CALUDE_downstream_distance_l3706_370685

/-- Calculates the distance traveled downstream by a boat -/
theorem downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 30) 
  (h2 : stream_speed = 5) 
  (h3 : time = 2) : 
  boat_speed + stream_speed * time = 70 := by
  sorry

#check downstream_distance

end NUMINAMATH_CALUDE_downstream_distance_l3706_370685


namespace NUMINAMATH_CALUDE_rhombus_diagonals_perpendicular_rectangle_not_l3706_370612

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  let (x₁, y₁) := q.A
  let (x₂, y₂) := q.B
  let (x₃, y₃) := q.C
  let (x₄, y₄) := q.D
  -- All sides are equal
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₂ - x₃)^2 + (y₂ - y₃)^2 ∧
  (x₂ - x₃)^2 + (y₂ - y₃)^2 = (x₃ - x₄)^2 + (y₃ - y₄)^2 ∧
  (x₃ - x₄)^2 + (y₃ - y₄)^2 = (x₄ - x₁)^2 + (y₄ - y₁)^2

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  let (x₁, y₁) := q.A
  let (x₂, y₂) := q.B
  let (x₃, y₃) := q.C
  let (x₄, y₄) := q.D
  -- Opposite sides are equal and all angles are right angles
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₃ - x₄)^2 + (y₃ - y₄)^2 ∧
  (x₂ - x₃)^2 + (y₂ - y₃)^2 = (x₄ - x₁)^2 + (y₄ - y₁)^2 ∧
  (x₂ - x₁) * (x₃ - x₂) + (y₂ - y₁) * (y₃ - y₂) = 0 ∧
  (x₃ - x₂) * (x₄ - x₃) + (y₃ - y₂) * (y₄ - y₃) = 0 ∧
  (x₄ - x₃) * (x₁ - x₄) + (y₄ - y₃) * (y₁ - y₄) = 0 ∧
  (x₁ - x₄) * (x₂ - x₁) + (y₁ - y₄) * (y₂ - y₁) = 0

-- Define perpendicular diagonals
def perpendicular_diagonals (q : Quadrilateral) : Prop :=
  let (x₁, y₁) := q.A
  let (x₃, y₃) := q.C
  let (x₂, y₂) := q.B
  let (x₄, y₄) := q.D
  (x₃ - x₁) * (x₄ - x₂) + (y₃ - y₁) * (y₄ - y₂) = 0

-- Theorem statement
theorem rhombus_diagonals_perpendicular_rectangle_not :
  (∀ q : Quadrilateral, is_rhombus q → perpendicular_diagonals q) ∧
  ¬(∀ q : Quadrilateral, is_rectangle q → perpendicular_diagonals q) :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_perpendicular_rectangle_not_l3706_370612


namespace NUMINAMATH_CALUDE_prob_green_is_five_sevenths_l3706_370626

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from the given containers -/
def prob_green (containers : List Container) : ℚ :=
  let total_prob := containers.map (λ c => (c.green : ℚ) / (c.red + c.green))
  (total_prob.sum) / containers.length

/-- Theorem: The probability of selecting a green ball is 5/7 -/
theorem prob_green_is_five_sevenths : 
  let containers := [
    Container.mk 8 4,  -- Container I
    Container.mk 3 4,  -- Container II
    Container.mk 3 4   -- Container III
  ]
  prob_green containers = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_is_five_sevenths_l3706_370626


namespace NUMINAMATH_CALUDE_regular_polygon_lattice_points_regular_polyhedra_lattice_points_l3706_370622

-- Define a 3D lattice point
def LatticePoint := ℤ × ℤ × ℤ

-- Define a regular polygon
structure RegularPolygon (n : ℕ) where
  vertices : List LatticePoint
  is_regular : Bool  -- This should be a proof in a real implementation
  vertex_count : vertices.length = n

-- Define a regular polyhedron
structure RegularPolyhedron where
  vertices : List LatticePoint
  is_regular : Bool  -- This should be a proof in a real implementation

-- Theorem for regular polygons
theorem regular_polygon_lattice_points :
  ∀ n : ℕ, (∃ p : RegularPolygon n, True) ↔ n = 3 ∨ n = 4 ∨ n = 6 :=
sorry

-- Define the Platonic solids
inductive PlatonicSolid
  | Tetrahedron
  | Cube
  | Octahedron
  | Dodecahedron
  | Icosahedron

-- Function to check if a Platonic solid can have lattice point vertices
def has_lattice_vertices : PlatonicSolid → Prop
  | PlatonicSolid.Tetrahedron => True
  | PlatonicSolid.Cube => True
  | PlatonicSolid.Octahedron => True
  | PlatonicSolid.Dodecahedron => False
  | PlatonicSolid.Icosahedron => False

-- Theorem for regular polyhedra
theorem regular_polyhedra_lattice_points :
  ∀ s : PlatonicSolid, has_lattice_vertices s ↔
    (s = PlatonicSolid.Tetrahedron ∨ s = PlatonicSolid.Cube ∨ s = PlatonicSolid.Octahedron) :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_lattice_points_regular_polyhedra_lattice_points_l3706_370622


namespace NUMINAMATH_CALUDE_system_solvability_l3706_370664

/-- The system of equations and inequalities -/
def system (a b x y : ℝ) : Prop :=
  x * Real.cos a - y * Real.sin a - 3 ≤ 0 ∧
  x^2 + y^2 - 8*x + 2*y - b^2 - 6*b + 8 = 0

/-- The set of valid b values -/
def valid_b_set : Set ℝ :=
  {b | b ≤ -Real.sqrt 17 ∨ b ≥ Real.sqrt 17 - 6}

/-- Theorem stating the equivalence between the system having a solution
    for any a and b being in the valid set -/
theorem system_solvability (b : ℝ) :
  (∀ a, ∃ x y, system a b x y) ↔ b ∈ valid_b_set :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l3706_370664


namespace NUMINAMATH_CALUDE_percentage_comparison_l3706_370695

theorem percentage_comparison (x : ℝ) : 0.9 * x > 0.8 * 30 + 12 → x > 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_comparison_l3706_370695


namespace NUMINAMATH_CALUDE_ross_breaths_per_minute_l3706_370640

/-- Calculates the number of breaths per minute given the air inhaled per breath and total air inhaled in 24 hours. -/
def breaths_per_minute (air_per_breath : ℚ) (total_air_24h : ℚ) : ℚ :=
  (total_air_24h / air_per_breath) / (24 * 60)

/-- Theorem stating that Ross takes 17 breaths per minute. -/
theorem ross_breaths_per_minute :
  breaths_per_minute (5/9) 13600 = 17 := by
  sorry

#eval breaths_per_minute (5/9) 13600

end NUMINAMATH_CALUDE_ross_breaths_per_minute_l3706_370640


namespace NUMINAMATH_CALUDE_complex_conjugate_sum_l3706_370652

theorem complex_conjugate_sum (α β : ℝ) :
  2 * Complex.exp (Complex.I * α) + 2 * Complex.exp (Complex.I * β) = -1/2 + 4/5 * Complex.I →
  2 * Complex.exp (-Complex.I * α) + 2 * Complex.exp (-Complex.I * β) = -1/2 - 4/5 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_conjugate_sum_l3706_370652


namespace NUMINAMATH_CALUDE_not_ellipse_for_certain_m_l3706_370614

/-- The equation of the curve -/
def curve_equation (m x y : ℝ) : Prop :=
  (m - 1) * x^2 + (3 - m) * y^2 = (m - 1) * (3 - m)

/-- Definition of an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  1 < m ∧ m < 3 ∧ m ≠ 2

/-- The theorem to be proved -/
theorem not_ellipse_for_certain_m :
  ∀ m : ℝ, (m ≤ 1 ∨ m = 2 ∨ m ≥ 3) →
    ¬(is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_not_ellipse_for_certain_m_l3706_370614


namespace NUMINAMATH_CALUDE_midpoint_minus_eighth_l3706_370619

theorem midpoint_minus_eighth (a b c : ℚ) : 
  a = 1/4 → b = 1/2 → c = 1/8 → 
  ((a + b) / 2) - c = 1/4 := by sorry

end NUMINAMATH_CALUDE_midpoint_minus_eighth_l3706_370619


namespace NUMINAMATH_CALUDE_money_distribution_l3706_370629

theorem money_distribution (a b c d total : ℕ) : 
  a + b + c + d = total →
  2 * a = b →
  5 * a = 2 * c →
  a = d →
  a + b = 1800 →
  total = 4500 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l3706_370629


namespace NUMINAMATH_CALUDE_john_vowel_learning_days_l3706_370693

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of days John takes to learn one alphabet -/
def days_per_alphabet : ℕ := 3

/-- The total number of days John needs to finish learning all vowels -/
def total_days : ℕ := num_vowels * days_per_alphabet

theorem john_vowel_learning_days : total_days = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_vowel_learning_days_l3706_370693


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l3706_370661

theorem consecutive_even_integers_sum (a : ℤ) : 
  (a + (a + 4) = 144) → 
  (a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 370) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l3706_370661


namespace NUMINAMATH_CALUDE_smallest_c_value_l3706_370674

/-- The smallest possible value of c in a sequence satisfying specific conditions -/
theorem smallest_c_value : ∃ (a b c : ℤ),
  (a < b ∧ b < c) ∧                    -- a < b < c are integers
  (2 * b = a + c) ∧                    -- arithmetic progression
  (a * a = c * b) ∧                    -- geometric progression
  (∃ (m n p : ℤ), a = 5 * m ∧ b = 5 * n ∧ c = 5 * p) ∧  -- multiples of 5
  (0 < a ∧ 0 < b ∧ 0 < c) ∧            -- all numbers are positive
  (c = 20) ∧                           -- c equals 20
  (∀ (a' b' c' : ℤ),                   -- for any other triple satisfying the conditions
    (a' < b' ∧ b' < c') →
    (2 * b' = a' + c') →
    (a' * a' = c' * b') →
    (∃ (m' n' p' : ℤ), a' = 5 * m' ∧ b' = 5 * n' ∧ c' = 5 * p') →
    (0 < a' ∧ 0 < b' ∧ 0 < c') →
    (c ≤ c')) :=                       -- c is the smallest possible value
by sorry


end NUMINAMATH_CALUDE_smallest_c_value_l3706_370674


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l3706_370697

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17) 
  (h2 : c + a = 20) 
  (h3 : a + b = 23) : 
  Real.sqrt (a * b * c * (a + b + c)) = 10 * Real.sqrt 273 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l3706_370697


namespace NUMINAMATH_CALUDE_expression_value_l3706_370648

theorem expression_value (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : 
  -2*a - b^2 + 2*a*b = -41 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3706_370648


namespace NUMINAMATH_CALUDE_cube_root_of_four_condition_l3706_370641

theorem cube_root_of_four_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a ^ b = b ^ a) (h2 : b = 4 * a) : a = (4 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_four_condition_l3706_370641


namespace NUMINAMATH_CALUDE_liam_money_left_l3706_370615

/-- Calculates the amount of money Liam has left after paying his bills -/
def money_left_after_bills (monthly_savings : ℕ) (savings_duration_months : ℕ) (bills : ℕ) : ℕ :=
  monthly_savings * savings_duration_months - bills

/-- Theorem stating that Liam will have $8,500 left after paying his bills -/
theorem liam_money_left :
  money_left_after_bills 500 24 3500 = 8500 := by
  sorry

#eval money_left_after_bills 500 24 3500

end NUMINAMATH_CALUDE_liam_money_left_l3706_370615


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l3706_370611

theorem sin_cos_sum_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (47 * π / 180) * Real.cos (103 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l3706_370611


namespace NUMINAMATH_CALUDE_triangle_formation_l3706_370655

/-- Triangle inequality theorem check for three sides -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Check if a set of three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 5 10 13 ∧
  ¬can_form_triangle 1 2 3 ∧
  ¬can_form_triangle 4 5 10 ∧
  ∀ a : ℝ, a > 0 → ¬can_form_triangle (2*a) (3*a) (6*a) :=
by sorry


end NUMINAMATH_CALUDE_triangle_formation_l3706_370655


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l3706_370624

theorem floor_ceiling_sum : ⌊(-2.54 : ℝ)⌋ + ⌈(25.4 : ℝ)⌉ = 23 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l3706_370624


namespace NUMINAMATH_CALUDE_test_number_satisfies_conditions_l3706_370675

/-- The test number that satisfies the given conditions -/
def test_number : ℕ := 5

/-- The average score before the current test -/
def previous_average : ℚ := 85

/-- The desired new average score -/
def new_average : ℚ := 88

/-- The score needed on the current test -/
def current_test_score : ℕ := 100

theorem test_number_satisfies_conditions :
  (new_average * test_number : ℚ) - (previous_average * (test_number - 1) : ℚ) = current_test_score := by
  sorry

end NUMINAMATH_CALUDE_test_number_satisfies_conditions_l3706_370675


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l3706_370665

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 55) (h2 : Nat.lcm a b = 1500) :
  a * b = 82500 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l3706_370665


namespace NUMINAMATH_CALUDE_difference_of_difference_eq_intersection_l3706_370643

-- Define the difference of two sets
def set_difference (A B : Set α) : Set α := {x | x ∈ A ∧ x ∉ B}

-- Theorem statement
theorem difference_of_difference_eq_intersection
  {α : Type*} (A B : Set α) (hA : A.Nonempty) (hB : B.Nonempty) :
  A \ (A \ B) = A ∩ B :=
sorry

end NUMINAMATH_CALUDE_difference_of_difference_eq_intersection_l3706_370643


namespace NUMINAMATH_CALUDE_min_value_of_f_l3706_370603

def f (x : ℝ) (m : ℝ) := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3706_370603


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l3706_370654

/-- A quadratic function with specific properties -/
def q (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, q a b c x = q a b c (15 - x)) →
  q a b c 0 = -3 →
  q a b c 15 = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l3706_370654


namespace NUMINAMATH_CALUDE_upper_pyramid_volume_l3706_370608

/-- The volume of the upper smaller pyramid formed by cutting a right square pyramid -/
theorem upper_pyramid_volume 
  (base_edge : ℝ) 
  (slant_edge : ℝ) 
  (cut_height : ℝ) 
  (h : base_edge = 12 * Real.sqrt 2) 
  (s : slant_edge = 15) 
  (c : cut_height = 5) : 
  ∃ (volume : ℝ), 
    volume = (1/6) * ((12 * Real.sqrt 2 * (Real.sqrt 153 - 5)) / Real.sqrt 153)^2 * (Real.sqrt 153 - 5) :=
by sorry

end NUMINAMATH_CALUDE_upper_pyramid_volume_l3706_370608


namespace NUMINAMATH_CALUDE_prob_at_least_one_female_l3706_370657

/-- The probability of selecting at least one female student when choosing 3 students from a group of 3 male and 2 female students. -/
theorem prob_at_least_one_female (n_male : ℕ) (n_female : ℕ) (n_select : ℕ) : 
  n_male = 3 → n_female = 2 → n_select = 3 →
  (1 : ℚ) - (Nat.choose n_male n_select : ℚ) / (Nat.choose (n_male + n_female) n_select : ℚ) = 9/10 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_female_l3706_370657


namespace NUMINAMATH_CALUDE_intersection_property_l3706_370628

-- Define the circle
def Circle (a : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = a}

-- Define the line
def Line := {p : ℝ × ℝ | p.1 + p.2 = 1}

-- Define the origin
def O : ℝ × ℝ := (0, 0)

theorem intersection_property (a : ℝ) 
  (A B : ℝ × ℝ) 
  (hA : A ∈ Circle a ∩ Line) 
  (hB : B ∈ Circle a ∩ Line) 
  (C : ℝ × ℝ) 
  (hC : C ∈ Circle a) 
  (h_vec : (A.1 - O.1, A.2 - O.2) + (B.1 - O.1, B.2 - O.2) = (C.1 - O.1, C.2 - O.2)) :
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_property_l3706_370628


namespace NUMINAMATH_CALUDE_walk_distance_proof_l3706_370625

/-- Given a total distance walked and a distance walked before rest,
    calculate the distance walked after rest. -/
def distance_after_rest (total : Real) (before_rest : Real) : Real :=
  total - before_rest

/-- Theorem: Given a total distance of 1 mile and a distance of 0.75 mile
    walked before rest, the distance walked after rest is 0.25 mile. -/
theorem walk_distance_proof :
  let total_distance : Real := 1
  let before_rest : Real := 0.75
  distance_after_rest total_distance before_rest = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_walk_distance_proof_l3706_370625


namespace NUMINAMATH_CALUDE_max_A_value_l3706_370696

def digits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_valid_arrangement (a b c d e f g h i : Nat) : Prop :=
  {a, b, c, d, e, f, g, h, i} = digits

def A (a b c d e f g h i : Nat) : Nat :=
  (100*a + 10*b + c) + (100*b + 10*c + d) + (100*c + 10*d + e) +
  (100*d + 10*e + f) + (100*e + 10*f + g) + (100*f + 10*g + h) +
  (100*g + 10*h + i)

theorem max_A_value :
  ∃ (a b c d e f g h i : Nat),
    is_valid_arrangement a b c d e f g h i ∧
    A a b c d e f g h i = 4648 ∧
    ∀ (a' b' c' d' e' f' g' h' i' : Nat),
      is_valid_arrangement a' b' c' d' e' f' g' h' i' →
      A a' b' c' d' e' f' g' h' i' ≤ 4648 :=
by sorry

end NUMINAMATH_CALUDE_max_A_value_l3706_370696


namespace NUMINAMATH_CALUDE_triangle_side_length_l3706_370680

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 60 * π / 180 →  -- Convert 60° to radians
  c = Real.sqrt 3 →
  b = 2 * Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3706_370680


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l3706_370651

theorem mistaken_multiplication (x : ℤ) : 139 * 43 - 139 * x = 1251 → x = 34 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l3706_370651


namespace NUMINAMATH_CALUDE_sum_of_v_at_specific_points_l3706_370688

-- Define the function v
def v (x : ℝ) : ℝ := x^3 - 3*x + 1

-- State the theorem
theorem sum_of_v_at_specific_points : 
  v 2 + v (-2) + v 1 + v (-1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_v_at_specific_points_l3706_370688


namespace NUMINAMATH_CALUDE_xiaoSiScore_l3706_370613

/-- Represents the correctness of an answer -/
inductive Correctness
| Correct
| Incorrect

/-- Represents a single question in the test -/
structure Question where
  number : Nat
  points : Nat
  correctness : Correctness

/-- Calculates the score for a single question -/
def scoreQuestion (q : Question) : Nat :=
  match q.correctness with
  | Correctness.Correct => q.points
  | Correctness.Incorrect => 0

/-- Xiao Si's test answers -/
def xiaoSiAnswers : List Question :=
  [
    { number := 1, points := 20, correctness := Correctness.Correct },
    { number := 2, points := 20, correctness := Correctness.Incorrect },
    { number := 3, points := 20, correctness := Correctness.Incorrect },
    { number := 4, points := 20, correctness := Correctness.Incorrect },
    { number := 5, points := 20, correctness := Correctness.Incorrect }
  ]

/-- Calculates the total score for the test -/
def calculateTotalScore (answers : List Question) : Nat :=
  answers.foldl (fun acc q => acc + scoreQuestion q) 0

/-- Theorem stating that Xiao Si's total score is 20 points -/
theorem xiaoSiScore : calculateTotalScore xiaoSiAnswers = 20 := by
  sorry


end NUMINAMATH_CALUDE_xiaoSiScore_l3706_370613


namespace NUMINAMATH_CALUDE_policeman_catches_thief_l3706_370642

/-- Represents a point on the square --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a pathway on the square --/
inductive Pathway
  | Edge : Pathway
  | Diagonal : Pathway

/-- Represents the square with its pathways --/
structure Square :=
  (side_length : ℝ)
  (pathways : List Pathway)

/-- Represents the positions and speeds of the policeman and thief --/
structure ChaseState :=
  (policeman_pos : Point)
  (thief_pos : Point)
  (policeman_speed : ℝ)
  (thief_speed : ℝ)

/-- Defines the chase dynamics --/
def chase (square : Square) (initial_state : ChaseState) : Prop :=
  sorry

theorem policeman_catches_thief 
  (square : Square) 
  (initial_state : ChaseState) 
  (h1 : square.side_length > 0)
  (h2 : square.pathways.length = 6)
  (h3 : initial_state.policeman_speed > 2.1 * initial_state.thief_speed)
  (h4 : initial_state.thief_speed > 0) :
  chase square initial_state :=
by sorry

end NUMINAMATH_CALUDE_policeman_catches_thief_l3706_370642


namespace NUMINAMATH_CALUDE_u_converges_to_L_least_k_is_zero_l3706_370637

def u : ℕ → ℚ
  | 0 => 1/3
  | n+1 => 3 * u n - 3 * (u n)^2

def L : ℚ := 1/3

theorem u_converges_to_L (n : ℕ) : |u n - L| ≤ 1 / 2^100 := by
  sorry

theorem least_k_is_zero : ∀ k : ℕ, (∀ n : ℕ, n < k → |u n - L| > 1 / 2^100) → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_u_converges_to_L_least_k_is_zero_l3706_370637


namespace NUMINAMATH_CALUDE_trajectory_of_A_l3706_370659

-- Define the points B and C
def B : ℝ × ℝ := (-3, 0)
def C : ℝ × ℝ := (3, 0)

-- Define the perimeter of triangle ABC
def perimeter : ℝ := 16

-- Define the trajectory of point A
def trajectory (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 16 = 1 ∧ y ≠ 0

-- Theorem statement
theorem trajectory_of_A (A : ℝ × ℝ) :
  (dist A B + dist A C + dist B C = perimeter) →
  trajectory A.1 A.2 :=
by sorry


end NUMINAMATH_CALUDE_trajectory_of_A_l3706_370659


namespace NUMINAMATH_CALUDE_binomial_16_13_l3706_370668

theorem binomial_16_13 : Nat.choose 16 13 = 560 := by
  sorry

end NUMINAMATH_CALUDE_binomial_16_13_l3706_370668


namespace NUMINAMATH_CALUDE_square_root_equality_l3706_370667

theorem square_root_equality (x : ℝ) : 
  (Real.sqrt (x + 3) = 3) → ((x + 3)^2 = 81) := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_l3706_370667


namespace NUMINAMATH_CALUDE_paco_ate_18_cookies_l3706_370646

/-- The number of cookies Paco ate -/
def cookies_eaten (initial : ℕ) (given : ℕ) : ℕ := given + given

/-- Proof that Paco ate 18 cookies -/
theorem paco_ate_18_cookies (initial : ℕ) (given : ℕ) 
  (h1 : initial = 41)
  (h2 : given = 9) :
  cookies_eaten initial given = 18 := by
  sorry

end NUMINAMATH_CALUDE_paco_ate_18_cookies_l3706_370646


namespace NUMINAMATH_CALUDE_original_profit_margin_exists_l3706_370676

/-- Given a reduction in purchase price and an increase in profit margin,
    there exists a unique original profit margin. -/
theorem original_profit_margin_exists :
  ∃! x : ℝ, 
    0 ≤ x ∧ x ≤ 100 ∧
    (1 + (x + 8) / 100) * (1 - 0.064) = 1 + x / 100 :=
by sorry

end NUMINAMATH_CALUDE_original_profit_margin_exists_l3706_370676


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l3706_370600

/-- Given a rhombus with area 150 cm² and one diagonal of length 10 cm, 
    prove that the length of the other diagonal is 30 cm. -/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_area : area = 150)
  (h_d1 : d1 = 10)
  (h_rhombus_area : area = (d1 * d2) / 2) :
  d2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l3706_370600


namespace NUMINAMATH_CALUDE_calculation_proof_l3706_370601

theorem calculation_proof : 20062006 * 2007 + 20072007 * 2008 - 2006 * 20072007 - 2007 * 20082008 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3706_370601


namespace NUMINAMATH_CALUDE_exists_special_function_l3706_370618

theorem exists_special_function : 
  ∃ f : ℕ+ → ℕ+, f 1 = 2 ∧ ∀ n : ℕ+, f (f n) = f n + n :=
by sorry

end NUMINAMATH_CALUDE_exists_special_function_l3706_370618


namespace NUMINAMATH_CALUDE_floor_nested_equation_l3706_370630

theorem floor_nested_equation (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := by
  sorry

end NUMINAMATH_CALUDE_floor_nested_equation_l3706_370630


namespace NUMINAMATH_CALUDE_cereal_eating_time_l3706_370634

/-- The time it takes for two people to eat a certain amount of cereal together -/
def eating_time (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- The proposition that Mr. Fat and Mr. Thin can eat 4 pounds of cereal in 37.5 minutes -/
theorem cereal_eating_time :
  let fat_rate : ℚ := 1 / 15  -- Mr. Fat's eating rate in pounds per minute
  let thin_rate : ℚ := 1 / 25 -- Mr. Thin's eating rate in pounds per minute
  let total_amount : ℚ := 4   -- Total amount of cereal in pounds
  eating_time fat_rate thin_rate total_amount = 75 / 2 := by
  sorry

#eval eating_time (1/15 : ℚ) (1/25 : ℚ) 4

end NUMINAMATH_CALUDE_cereal_eating_time_l3706_370634


namespace NUMINAMATH_CALUDE_broomsticks_count_l3706_370617

/-- Represents the Halloween decorations problem --/
def halloween_decorations (skulls spiderwebs pumpkins cauldrons budget_left to_put_up total broomsticks : ℕ) : Prop :=
  skulls = 12 ∧
  spiderwebs = 12 ∧
  pumpkins = 2 * spiderwebs ∧
  cauldrons = 1 ∧
  budget_left = 20 ∧
  to_put_up = 10 ∧
  total = 83 ∧
  total = skulls + spiderwebs + pumpkins + cauldrons + budget_left + to_put_up + broomsticks

/-- Theorem stating that the number of broomsticks is 4 --/
theorem broomsticks_count :
  ∀ skulls spiderwebs pumpkins cauldrons budget_left to_put_up total broomsticks,
  halloween_decorations skulls spiderwebs pumpkins cauldrons budget_left to_put_up total broomsticks →
  broomsticks = 4 := by
  sorry

end NUMINAMATH_CALUDE_broomsticks_count_l3706_370617


namespace NUMINAMATH_CALUDE_dvd_rental_cost_l3706_370645

/-- The cost per DVD given the total cost and number of DVDs rented --/
def cost_per_dvd (total_cost : ℚ) (num_dvds : ℕ) : ℚ :=
  total_cost / num_dvds

/-- Theorem stating that the cost per DVD is $1.20 given the problem conditions --/
theorem dvd_rental_cost : 
  let total_cost : ℚ := 48/10
  let num_dvds : ℕ := 4
  cost_per_dvd total_cost num_dvds = 12/10 := by
  sorry

end NUMINAMATH_CALUDE_dvd_rental_cost_l3706_370645


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3706_370689

theorem complex_fraction_simplification :
  let z : ℂ := (1 - Complex.I * Real.sqrt 3) / (Complex.I + Real.sqrt 3) ^ 2
  z = -1/4 - (Complex.I * Real.sqrt 3) / 4 := by
    sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3706_370689


namespace NUMINAMATH_CALUDE_expenditure_ratio_l3706_370639

theorem expenditure_ratio (income : ℝ) (h : income > 0) :
  let savings_rate := 0.35
  let income_increase := 0.35
  let savings_increase := 1.0

  let savings_year1 := savings_rate * income
  let expenditure_year1 := income - savings_year1

  let income_year2 := income * (1 + income_increase)
  let savings_year2 := savings_year1 * (1 + savings_increase)
  let expenditure_year2 := income_year2 - savings_year2

  let total_expenditure := expenditure_year1 + expenditure_year2

  (total_expenditure / expenditure_year1) = 2
  := by sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l3706_370639


namespace NUMINAMATH_CALUDE_least_n_factorial_divisible_by_32_l3706_370684

theorem least_n_factorial_divisible_by_32 :
  ∀ n : ℕ, n > 0 → (n.factorial % 32 = 0) → n ≥ 8 :=
by
  sorry

end NUMINAMATH_CALUDE_least_n_factorial_divisible_by_32_l3706_370684


namespace NUMINAMATH_CALUDE_car_cost_difference_l3706_370647

/-- The cost difference between buying and renting a car for a year -/
theorem car_cost_difference (rental_cost : ℕ) (purchase_cost : ℕ) : 
  rental_cost = 20 → purchase_cost = 30 → purchase_cost * 12 - rental_cost * 12 = 120 := by
  sorry

end NUMINAMATH_CALUDE_car_cost_difference_l3706_370647
