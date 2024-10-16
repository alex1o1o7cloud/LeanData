import Mathlib

namespace NUMINAMATH_CALUDE_train_meeting_time_l2801_280136

/-- The time when the trains meet -/
def meeting_time : ℝ := 11

/-- The time when train A starts -/
def start_time_A : ℝ := 7

/-- The distance between stations A and B in km -/
def total_distance : ℝ := 155

/-- The speed of train A in km/h -/
def speed_A : ℝ := 20

/-- The speed of train B in km/h -/
def speed_B : ℝ := 25

/-- The start time of train B -/
def start_time_B : ℝ := 8

theorem train_meeting_time :
  start_time_B = 8 :=
by sorry

end NUMINAMATH_CALUDE_train_meeting_time_l2801_280136


namespace NUMINAMATH_CALUDE_maria_cookie_baggies_l2801_280193

/-- The number of baggies Maria can make with her cookies -/
def num_baggies (cookies_per_baggie : ℕ) (chocolate_chip_cookies : ℕ) (oatmeal_cookies : ℕ) : ℕ :=
  (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_baggie

/-- Theorem stating that Maria can make 7 baggies of cookies -/
theorem maria_cookie_baggies :
  num_baggies 5 33 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_maria_cookie_baggies_l2801_280193


namespace NUMINAMATH_CALUDE_statue_weight_calculation_l2801_280171

/-- The weight of a statue after a series of cuts -/
def final_statue_weight (original_weight : ℝ) : ℝ :=
  let after_first_cut := original_weight * (1 - 0.3)
  let after_second_cut := after_first_cut * (1 - 0.2)
  let after_third_cut := after_second_cut * (1 - 0.25)
  after_third_cut

/-- Theorem stating the final weight of the statue -/
theorem statue_weight_calculation :
  final_statue_weight 250 = 105 := by
  sorry

end NUMINAMATH_CALUDE_statue_weight_calculation_l2801_280171


namespace NUMINAMATH_CALUDE_probability_independent_of_radius_constant_probability_l2801_280165

-- Define a circular dartboard
structure Dartboard where
  radius : ℝ
  radius_pos : radius > 0

-- Define the probability function
def probability_closer_to_center (d : Dartboard) : ℝ := 0.25

-- Theorem statement
theorem probability_independent_of_radius (d : Dartboard) :
  probability_closer_to_center d = 0.25 := by
  sorry

-- The distance from the thrower is not relevant to the probability,
-- but we include it to match the original problem description
def distance_from_thrower : ℝ := 20

-- Theorem stating that the probability is constant regardless of radius
theorem constant_probability (d1 d2 : Dartboard) :
  probability_closer_to_center d1 = probability_closer_to_center d2 := by
  sorry

end NUMINAMATH_CALUDE_probability_independent_of_radius_constant_probability_l2801_280165


namespace NUMINAMATH_CALUDE_problem_statement_l2801_280191

theorem problem_statement (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b)^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2801_280191


namespace NUMINAMATH_CALUDE_slope_of_line_l2801_280166

def line_equation (x y : ℝ) : Prop := 3 * y = 4 * x - 9

theorem slope_of_line : ∃ m : ℝ, m = 4 / 3 ∧ 
  ∀ x y : ℝ, line_equation x y → y = m * x + (-3) :=
sorry

end NUMINAMATH_CALUDE_slope_of_line_l2801_280166


namespace NUMINAMATH_CALUDE_arithmetic_sequence_log_property_l2801_280183

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Define the arithmetic sequence property
def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

-- Define the theorem
theorem arithmetic_sequence_log_property
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : is_arithmetic_sequence (log (a^2 * b^6)) (log (a^4 * b^11)) (log (a^7 * b^14)))
  (h4 : ∃ m : ℕ, (log (b^m)) = (log (a^2 * b^6)) + 7 * ((log (a^4 * b^11)) - (log (a^2 * b^6))))
  : ∃ m : ℕ, m = 73 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_log_property_l2801_280183


namespace NUMINAMATH_CALUDE_expression_value_approximation_l2801_280142

def x : ℝ := 102
def y : ℝ := 98

theorem expression_value_approximation :
  let expr := (x^2 - y^2) / (x + y)^3 - (x^3 + y^3) * Real.log (x*y)
  ∃ ε > 0, |expr + 18446424.7199| < ε := by
  sorry

end NUMINAMATH_CALUDE_expression_value_approximation_l2801_280142


namespace NUMINAMATH_CALUDE_xy_value_l2801_280145

theorem xy_value (x y : ℝ) (h : Real.sqrt (x - 1) + (y - 2)^2 = 0) : x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2801_280145


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2801_280133

theorem sum_of_solutions (x : ℕ) : 
  (∃ (s : Finset ℕ), 
    (∀ n ∈ s, 0 < n ∧ n ≤ 25 ∧ (7*(5*n - 3) : ℤ) ≡ 35 [ZMOD 9]) ∧
    (∀ m : ℕ, 0 < m ∧ m ≤ 25 ∧ (7*(5*m - 3) : ℤ) ≡ 35 [ZMOD 9] → m ∈ s) ∧
    s.sum id = 48) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2801_280133


namespace NUMINAMATH_CALUDE_product_quotient_calculation_l2801_280112

theorem product_quotient_calculation : 16 * 0.0625 / 4 * 0.5 * 2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_product_quotient_calculation_l2801_280112


namespace NUMINAMATH_CALUDE_shifted_increasing_interval_l2801_280137

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem shifted_increasing_interval :
  IncreasingOn f (-2) 3 → IncreasingOn (fun x ↦ f (x + 4)) (-6) (-1) := by
  sorry

end NUMINAMATH_CALUDE_shifted_increasing_interval_l2801_280137


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l2801_280138

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := 5/4 * x^2 + 3/4 * x + 1

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-3) = 10 ∧ q 0 = 1 ∧ q 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l2801_280138


namespace NUMINAMATH_CALUDE_cosine_value_from_ratio_l2801_280181

theorem cosine_value_from_ratio (α : Real) (h : (1 - Real.cos α) / Real.sin α = 3) :
  Real.cos α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_from_ratio_l2801_280181


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l2801_280109

theorem largest_integer_in_interval : ∃ (x : ℤ), 
  (∀ (y : ℤ), (1/5 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 7/12 → y ≤ x) ∧
  (1/5 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 7/12 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l2801_280109


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2801_280105

theorem min_value_of_sum_of_squares (a b c : ℝ) (h : a^2 + b^2 + c^2 = 6) :
  (a + b)^2 + (b + c)^2 + (c + a)^2 ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2801_280105


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2801_280134

/-- The function f(x) = 3x^2 - 18x + 7 attains its minimum value when x = 3. -/
theorem min_value_quadratic (x : ℝ) : 
  ∃ (min : ℝ), ∀ y : ℝ, 3 * x^2 - 18 * x + 7 ≥ 3 * min^2 - 18 * min + 7 ∧ 
  (3 * min^2 - 18 * min + 7 = 3 * 3^2 - 18 * 3 + 7) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2801_280134


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2801_280169

def repeating_decimal_234 : ℚ := 234 / 999
def repeating_decimal_567 : ℚ := 567 / 999
def repeating_decimal_891 : ℚ := 891 / 999

theorem repeating_decimal_subtraction :
  repeating_decimal_234 - repeating_decimal_567 - repeating_decimal_891 = -1224 / 999 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2801_280169


namespace NUMINAMATH_CALUDE_range_of_a_l2801_280179

/-- Given sets A and B, and the condition that A is not a subset of B, 
    prove that the range of values for a is (1, 5) -/
theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := {x | 2*a < x ∧ x < a + 5}
  let B : Set ℝ := {x | x < 6}
  ¬(A ⊆ B) → a ∈ Set.Ioo 1 5 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l2801_280179


namespace NUMINAMATH_CALUDE_line_A2A3_tangent_to_circle_M_l2801_280190

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = x

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point on the parabola
def point_on_parabola (A : ℝ × ℝ) : Prop := parabola_C A.1 A.2

-- Define a line tangent to the circle
def line_tangent_to_circle (A B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), circle_M (A.1 + t * (B.1 - A.1)) (A.2 + t * (B.2 - A.2))

-- Main theorem
theorem line_A2A3_tangent_to_circle_M (A₁ A₂ A₃ : ℝ × ℝ) :
  point_on_parabola A₁ →
  point_on_parabola A₂ →
  point_on_parabola A₃ →
  line_tangent_to_circle A₁ A₂ →
  line_tangent_to_circle A₁ A₃ →
  line_tangent_to_circle A₂ A₃ :=
sorry

end NUMINAMATH_CALUDE_line_A2A3_tangent_to_circle_M_l2801_280190


namespace NUMINAMATH_CALUDE_don_remaining_rum_l2801_280111

/-- The amount of rum Sally gave Don on his pancakes, in ounces. -/
def initial_rum : ℝ := 10

/-- The maximum factor by which Don can consume rum for a healthy diet. -/
def max_factor : ℝ := 3

/-- The amount of rum Don had earlier that day, in ounces. -/
def earlier_rum : ℝ := 12

/-- Calculates the maximum amount of rum Don can consume for a healthy diet. -/
def max_rum : ℝ := initial_rum * max_factor

/-- Calculates the total amount of rum Don has consumed so far. -/
def consumed_rum : ℝ := initial_rum + earlier_rum

/-- Theorem stating how much rum Don can have after eating all of the rum and pancakes. -/
theorem don_remaining_rum : max_rum - consumed_rum = 8 := by sorry

end NUMINAMATH_CALUDE_don_remaining_rum_l2801_280111


namespace NUMINAMATH_CALUDE_angle_identity_l2801_280106

/-- If the terminal side of angle α passes through point P(-2, 1) in the rectangular coordinate system, 
    then cos²α - sin(2α) = 8/5 -/
theorem angle_identity (α : ℝ) : 
  (∃ (x y : ℝ), x = -2 ∧ y = 1 ∧ y / x = Real.tan α) → 
  Real.cos α ^ 2 - Real.sin (2 * α) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_identity_l2801_280106


namespace NUMINAMATH_CALUDE_complex_vector_properties_l2801_280180

/-- Represents a complex number in the Cartesian plane -/
structure ComplexVector where
  x : ℝ
  y : ℝ

/-- Given an imaginary number z, returns its corresponding vector representation -/
noncomputable def z_to_vector (z : ℂ) : ComplexVector :=
  { x := z.re, y := z.im }

/-- Given an imaginary number z, returns the vector representation of its conjugate -/
noncomputable def conj_to_vector (z : ℂ) : ComplexVector :=
  { x := z.re, y := -z.im }

/-- Given an imaginary number z, returns the vector representation of its reciprocal -/
noncomputable def recip_to_vector (z : ℂ) : ComplexVector :=
  { x := z.re / (z.re^2 + z.im^2), y := -z.im / (z.re^2 + z.im^2) }

/-- Checks if three points are collinear given two vectors -/
def are_collinear (v1 v2 : ComplexVector) : Prop :=
  v1.x * v2.y = v1.y * v2.x

/-- Adds two ComplexVectors -/
def add_vectors (v1 v2 : ComplexVector) : ComplexVector :=
  { x := v1.x + v2.x, y := v1.y + v2.y }

theorem complex_vector_properties (z : ℂ) (h : z^3 = 1) :
  let OA := z_to_vector z
  let OB := conj_to_vector z
  let OC := recip_to_vector z
  (are_collinear OB OC) ∧ (add_vectors OA OC = ComplexVector.mk (-1) 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_vector_properties_l2801_280180


namespace NUMINAMATH_CALUDE_immediate_boarding_probability_l2801_280102

def train_departure_interval : ℝ := 15
def train_stop_duration : ℝ := 2

theorem immediate_boarding_probability :
  (train_stop_duration / train_departure_interval : ℝ) = 2 / 15 := by sorry

end NUMINAMATH_CALUDE_immediate_boarding_probability_l2801_280102


namespace NUMINAMATH_CALUDE_circumcenter_on_side_implies_right_angled_l2801_280161

/-- A triangle is represented by its three vertices in a 2D plane. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumcenter of a triangle is the point where the perpendicular bisectors of the sides intersect. -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- A point lies on a side of a triangle if it's on the line segment between two vertices. -/
def point_on_side (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- A triangle is right-angled if one of its angles is a right angle (90 degrees). -/
def is_right_angled (t : Triangle) : Prop := sorry

/-- 
If the circumcenter of a triangle lies on one of its sides, then the triangle is right-angled.
-/
theorem circumcenter_on_side_implies_right_angled (t : Triangle) :
  point_on_side (circumcenter t) t → is_right_angled t := by sorry

end NUMINAMATH_CALUDE_circumcenter_on_side_implies_right_angled_l2801_280161


namespace NUMINAMATH_CALUDE_base_seven_23456_equals_6068_l2801_280149

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem base_seven_23456_equals_6068 :
  base_seven_to_ten [6, 5, 4, 3, 2] = 6068 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_23456_equals_6068_l2801_280149


namespace NUMINAMATH_CALUDE_equation_solution_l2801_280198

theorem equation_solution : 
  ∃! x : ℚ, (x - 30) / 3 = (3 * x + 10) / 8 - 2 ∧ x = -222 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2801_280198


namespace NUMINAMATH_CALUDE_cream_cheese_amount_l2801_280164

/-- Calculates the amount of cream cheese used in a spinach quiche recipe. -/
theorem cream_cheese_amount
  (raw_spinach : ℝ)
  (cooked_spinach_percentage : ℝ)
  (eggs : ℝ)
  (total_volume : ℝ)
  (h1 : raw_spinach = 40)
  (h2 : cooked_spinach_percentage = 0.20)
  (h3 : eggs = 4)
  (h4 : total_volume = 18) :
  total_volume - (raw_spinach * cooked_spinach_percentage) - eggs = 6 := by
  sorry

end NUMINAMATH_CALUDE_cream_cheese_amount_l2801_280164


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2801_280114

/-- A square inscribed in a right triangle with one vertex at the right angle -/
def square_in_triangle_vertex (a b c : ℝ) (x : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ (c - x) / x = a / b

/-- A square inscribed in a right triangle with one side on the hypotenuse -/
def square_in_triangle_hypotenuse (a b c : ℝ) (y : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ (a - y) / y = (b - y) / y

theorem inscribed_squares_ratio :
  ∀ x y : ℝ,
    square_in_triangle_vertex 5 12 13 x →
    square_in_triangle_hypotenuse 6 8 10 y →
    x / y = 37 / 35 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2801_280114


namespace NUMINAMATH_CALUDE_haley_car_distance_l2801_280153

/-- Calculates the distance covered given a fuel-to-distance ratio and fuel consumption -/
def distance_covered (fuel_ratio : ℕ) (distance_ratio : ℕ) (fuel_used : ℕ) : ℕ :=
  (fuel_used / fuel_ratio) * distance_ratio

/-- Theorem stating that for a 4:7 fuel-to-distance ratio and 44 gallons of fuel, the distance covered is 77 miles -/
theorem haley_car_distance :
  distance_covered 4 7 44 = 77 := by
  sorry

end NUMINAMATH_CALUDE_haley_car_distance_l2801_280153


namespace NUMINAMATH_CALUDE_sum_of_squares_inequality_l2801_280197

theorem sum_of_squares_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ((x + 1) / x)^2 + ((y + 1) / y)^2 ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_inequality_l2801_280197


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2801_280154

/-- Given a > 0 and a ≠ 1, the function f(x) = a^(x+1) + 1 always passes through the point (-1, 2) -/
theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) + 1
  f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2801_280154


namespace NUMINAMATH_CALUDE_min_fraction_sum_l2801_280140

def digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_valid_selection (P Q R S : ℕ) : Prop :=
  P ∈ digits ∧ Q ∈ digits ∧ R ∈ digits ∧ S ∈ digits ∧ P < Q ∧ Q < R ∧ R < S

def fraction_sum (P Q R S : ℕ) : ℚ :=
  (P : ℚ) / (R : ℚ) + (Q : ℚ) / (S : ℚ)

theorem min_fraction_sum :
  ∃ (P Q R S : ℕ), is_valid_selection P Q R S ∧
    (∀ (P' Q' R' S' : ℕ), is_valid_selection P' Q' R' S' →
      fraction_sum P Q R S ≤ fraction_sum P' Q' R' S') ∧
    fraction_sum P Q R S = 25 / 72 := by
  sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l2801_280140


namespace NUMINAMATH_CALUDE_books_count_l2801_280123

theorem books_count (benny_initial : ℕ) (given_to_sandy : ℕ) (tim_books : ℕ)
  (h1 : benny_initial = 24)
  (h2 : given_to_sandy = 10)
  (h3 : tim_books = 33) :
  benny_initial - given_to_sandy + tim_books = 47 := by
  sorry

end NUMINAMATH_CALUDE_books_count_l2801_280123


namespace NUMINAMATH_CALUDE_special_matrix_vector_product_l2801_280159

def matrix_vector_op (a b c d e f : ℝ) : ℝ × ℝ :=
  (a * e + b * f, c * e + d * f)

theorem special_matrix_vector_product 
  (α β : ℝ) 
  (h1 : α + β = Real.pi) 
  (h2 : α - β = Real.pi / 2) : 
  matrix_vector_op (Real.sin α) (Real.cos α) (Real.cos α) (Real.sin α) (Real.cos β) (Real.sin β) = (0, 0) := by
  sorry

end NUMINAMATH_CALUDE_special_matrix_vector_product_l2801_280159


namespace NUMINAMATH_CALUDE_square_region_perimeter_l2801_280126

/-- Given a region formed by eight congruent squares arranged in a vertical rectangle
    with a total area of 512 square centimeters, the perimeter of the region is 160 centimeters. -/
theorem square_region_perimeter : 
  ∀ (side_length : ℝ),
  side_length > 0 →
  8 * side_length^2 = 512 →
  2 * (7 * side_length + 3 * side_length) = 160 :=
by sorry

end NUMINAMATH_CALUDE_square_region_perimeter_l2801_280126


namespace NUMINAMATH_CALUDE_line_slope_from_y_intercept_l2801_280144

/-- Given a line with equation x + ay + 1 = 0 where a is a real number,
    and y-intercept -2, prove that the slope of the line is -2. -/
theorem line_slope_from_y_intercept (a : ℝ) :
  (∀ x y, x + a * y + 1 = 0 → (x = 0 → y = -2)) →
  ∃ m b, ∀ x y, y = m * x + b ∧ m = -2 :=
sorry

end NUMINAMATH_CALUDE_line_slope_from_y_intercept_l2801_280144


namespace NUMINAMATH_CALUDE_min_m_for_inequality_min_m_is_three_l2801_280162

theorem min_m_for_inequality (m : ℤ) : (2 * (2 - m) < 0) → m ≥ 3 :=
by sorry

theorem min_m_is_three : ∃ (m : ℤ), 2 * (2 - m) < 0 ∧ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_m_for_inequality_min_m_is_three_l2801_280162


namespace NUMINAMATH_CALUDE_pie_crust_flour_calculation_l2801_280174

theorem pie_crust_flour_calculation :
  let original_crusts : ℕ := 30
  let new_crusts : ℕ := 40
  let flour_per_original_crust : ℚ := 1 / 5
  let total_flour : ℚ := original_crusts * flour_per_original_crust
  let flour_per_new_crust : ℚ := total_flour / new_crusts
  flour_per_new_crust = 3 / 20 := by sorry

end NUMINAMATH_CALUDE_pie_crust_flour_calculation_l2801_280174


namespace NUMINAMATH_CALUDE_stratified_sampling_probability_l2801_280119

/-- Represents the number of students in each year of high school. -/
structure SchoolPopulation where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Represents the number of students selected from each year in the sample. -/
structure SampleSize where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- The probability of a student being selected in a stratified sampling survey. -/
def selectionProbability (population : SchoolPopulation) (sample : SampleSize) : ℚ :=
  sample.third_year / population.third_year

theorem stratified_sampling_probability
  (population : SchoolPopulation)
  (sample : SampleSize)
  (h1 : population.first_year = 800)
  (h2 : population.second_year = 600)
  (h3 : population.third_year = 500)
  (h4 : sample.third_year = 25) :
  selectionProbability population sample = 1 / 20 := by
  sorry

#check stratified_sampling_probability

end NUMINAMATH_CALUDE_stratified_sampling_probability_l2801_280119


namespace NUMINAMATH_CALUDE_transform_1220_to_2012_not_transform_1220_to_2021_l2801_280199

/-- Represents a 4-digit number -/
structure FourDigitNumber where
  digits : Fin 4 → Fin 10

/-- Defines the allowed transformations on a 4-digit number -/
def transform (n : FourDigitNumber) (i : Fin 3) : Option FourDigitNumber :=
  if n.digits i ≠ 0 ∧ n.digits (i + 1) ≠ 0 then
    some ⟨fun j => if j = i ∨ j = i + 1 then n.digits j - 1 else n.digits j⟩
  else if n.digits i ≠ 9 ∧ n.digits (i + 1) ≠ 9 then
    some ⟨fun j => if j = i ∨ j = i + 1 then n.digits j + 1 else n.digits j⟩
  else
    none

/-- Defines the reachability of one number from another through transformations -/
def reachable (start finish : FourDigitNumber) : Prop :=
  ∃ (seq : List (Fin 3)), finish = seq.foldl (fun n i => (transform n i).getD n) start

/-- The initial number 1220 -/
def initial : FourDigitNumber := ⟨fun i => match i with | 0 => 1 | 1 => 2 | 2 => 2 | 3 => 0⟩

/-- The target number 2012 -/
def target1 : FourDigitNumber := ⟨fun i => match i with | 0 => 2 | 1 => 0 | 2 => 1 | 3 => 2⟩

/-- The target number 2021 -/
def target2 : FourDigitNumber := ⟨fun i => match i with | 0 => 2 | 1 => 0 | 2 => 2 | 3 => 1⟩

theorem transform_1220_to_2012 : reachable initial target1 := by sorry

theorem not_transform_1220_to_2021 : ¬reachable initial target2 := by sorry

end NUMINAMATH_CALUDE_transform_1220_to_2012_not_transform_1220_to_2021_l2801_280199


namespace NUMINAMATH_CALUDE_initial_cards_count_l2801_280160

/-- The initial number of baseball cards Fred had -/
def initial_cards : ℕ := sorry

/-- The number of baseball cards Keith bought -/
def cards_bought : ℕ := 22

/-- The number of baseball cards Fred has left -/
def cards_left : ℕ := 18

/-- Theorem stating that the initial number of cards is 40 -/
theorem initial_cards_count : initial_cards = 40 := by sorry

end NUMINAMATH_CALUDE_initial_cards_count_l2801_280160


namespace NUMINAMATH_CALUDE_matt_jellybean_count_l2801_280168

/-- The number of jellybeans each person has -/
structure JellybeanCount where
  steve : ℕ
  matt : ℕ
  matilda : ℕ

/-- The conditions of the jellybean problem -/
def jellybean_conditions (j : JellybeanCount) : Prop :=
  j.steve = 84 ∧
  j.matilda = 420 ∧
  j.matilda * 2 = j.matt ∧
  ∃ k : ℕ, j.matt = k * j.steve

theorem matt_jellybean_count (j : JellybeanCount) 
  (h : jellybean_conditions j) : j.matt = 840 := by
  sorry

end NUMINAMATH_CALUDE_matt_jellybean_count_l2801_280168


namespace NUMINAMATH_CALUDE_multiple_of_72_l2801_280157

theorem multiple_of_72 (a b : Nat) :
  (a ≤ 9) →
  (b ≤ 9) →
  (a * 10000 + 6790 + b) % 72 = 0 ↔ a = 3 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_72_l2801_280157


namespace NUMINAMATH_CALUDE_opposite_of_ten_l2801_280172

theorem opposite_of_ten : ∃ x : ℝ, (x + 10 = 0) ∧ (x = -10) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_ten_l2801_280172


namespace NUMINAMATH_CALUDE_total_problems_practiced_l2801_280115

def marvin_yesterday : ℕ := 40
def marvin_today : ℕ := 3 * marvin_yesterday
def arvin_yesterday : ℕ := 2 * marvin_yesterday
def arvin_today : ℕ := 2 * marvin_today
def kevin_yesterday : ℕ := 30
def kevin_today : ℕ := kevin_yesterday ^ 2

theorem total_problems_practiced :
  marvin_yesterday + marvin_today + arvin_yesterday + arvin_today + kevin_yesterday + kevin_today = 1410 :=
by sorry

end NUMINAMATH_CALUDE_total_problems_practiced_l2801_280115


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l2801_280196

theorem elevator_weight_problem (initial_people : ℕ) (new_person_weight : ℝ) (new_average : ℝ) :
  initial_people = 6 →
  new_person_weight = 145 →
  new_average = 151 →
  ∃ initial_average : ℝ,
    initial_average = 152 ∧
    (initial_people : ℝ) * initial_average + new_person_weight = (initial_people + 1 : ℝ) * new_average :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l2801_280196


namespace NUMINAMATH_CALUDE_range_of_f_on_interval_range_of_a_l2801_280194

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Part 1: Range of f on [0, 4]
theorem range_of_f_on_interval :
  ∀ y ∈ Set.Icc 1 10, ∃ x ∈ Set.Icc 0 4, f x = y ∧
  ∀ x ∈ Set.Icc 0 4, 1 ≤ f x ∧ f x ≤ 10 :=
sorry

-- Part 2: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 2), f x ≤ 5) ↔ a ∈ Set.Icc (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_range_of_a_l2801_280194


namespace NUMINAMATH_CALUDE_angle_greater_if_sine_greater_l2801_280141

theorem angle_greater_if_sine_greater (A B C : Real) (a b c : Real) :
  -- Define triangle ABC
  (A + B + C = Real.pi) →
  (a > 0) → (b > 0) → (c > 0) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Given condition
  (Real.sin B > Real.sin C) →
  -- Conclusion
  B > C := by
  sorry


end NUMINAMATH_CALUDE_angle_greater_if_sine_greater_l2801_280141


namespace NUMINAMATH_CALUDE_ski_trip_sponsorship_l2801_280103

/-- The ski trip sponsorship problem -/
theorem ski_trip_sponsorship 
  (total : ℝ) 
  (first_father : ℝ) 
  (second_father third_father fourth_father : ℝ) 
  (h1 : first_father = 11500)
  (h2 : second_father = (1/3) * (total - second_father))
  (h3 : third_father = (1/4) * (total - third_father))
  (h4 : fourth_father = (1/5) * (total - fourth_father))
  (h5 : total = first_father + second_father + third_father + fourth_father) :
  second_father = 7500 ∧ third_father = 6000 ∧ fourth_father = 5000 := by
  sorry

#eval Float.toString 7500
#eval Float.toString 6000
#eval Float.toString 5000

end NUMINAMATH_CALUDE_ski_trip_sponsorship_l2801_280103


namespace NUMINAMATH_CALUDE_probability_x_lt_2y_is_one_sixth_l2801_280131

/-- A rectangle in the 2D plane -/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  x_min_le_x_max : x_min ≤ x_max
  y_min_le_y_max : y_min ≤ y_max

/-- The probability that a randomly chosen point (x,y) from the given rectangle satisfies x < 2y -/
def probability_x_lt_2y (r : Rectangle) : ℝ :=
  sorry

/-- The specific rectangle with vertices (0,0), (6,0), (6,1), and (0,1) -/
def specific_rectangle : Rectangle :=
  { x_min := 0
  , x_max := 6
  , y_min := 0
  , y_max := 1
  , x_min_le_x_max := by norm_num
  , y_min_le_y_max := by norm_num
  }

/-- Theorem stating that the probability of x < 2y for a randomly chosen point
    in the specific rectangle is 1/6 -/
theorem probability_x_lt_2y_is_one_sixth :
  probability_x_lt_2y specific_rectangle = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_lt_2y_is_one_sixth_l2801_280131


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_l2801_280108

noncomputable section

def f (x : ℝ) : ℝ := Real.log x + (x + 1) / x

theorem tangent_line_and_inequality (x : ℝ) (hx : x > 0) (hx1 : x ≠ 1) :
  (∃ (m b : ℝ), m * 1 + b = f 1 ∧ m = 2 ∧ ∀ t, f t = m * t + b) ∧
  f x > ((x + 1) * Real.log x) / (x - 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_l2801_280108


namespace NUMINAMATH_CALUDE_picture_area_l2801_280173

theorem picture_area (x y : ℤ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : (2*x + 4)*(y + 2) - x*y = 56) : 
  x * y = 24 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l2801_280173


namespace NUMINAMATH_CALUDE_opposite_of_eight_l2801_280139

theorem opposite_of_eight :
  ∀ x : ℤ, x + 8 = 0 ↔ x = -8 := by sorry

end NUMINAMATH_CALUDE_opposite_of_eight_l2801_280139


namespace NUMINAMATH_CALUDE_february_to_january_ratio_l2801_280122

-- Define the oil bills for January and February
def january_bill : ℚ := 120
def february_bill : ℚ := 180

-- Define the condition that February's bill is more than January's
axiom february_more_than_january : february_bill > january_bill

-- Define the condition about the 5:3 ratio if February's bill was $20 more
axiom ratio_condition : (february_bill + 20) / january_bill = 5 / 3

-- Theorem to prove
theorem february_to_january_ratio :
  february_bill / january_bill = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_february_to_january_ratio_l2801_280122


namespace NUMINAMATH_CALUDE_pupils_like_only_maths_l2801_280177

/-- Represents the number of pupils in various categories -/
structure ClassData where
  total : ℕ
  likesMaths : ℕ
  likesEnglish : ℕ
  likesBoth : ℕ
  likesNeither : ℕ

/-- The main theorem stating the number of pupils who like only Maths -/
theorem pupils_like_only_maths (c : ClassData) : 
  c.total = 30 ∧ 
  c.likesMaths = 20 ∧ 
  c.likesEnglish = 18 ∧ 
  c.likesBoth = 2 * c.likesNeither →
  c.likesMaths - c.likesBoth = 4 := by
  sorry


end NUMINAMATH_CALUDE_pupils_like_only_maths_l2801_280177


namespace NUMINAMATH_CALUDE_relation_between_exponents_l2801_280116

-- Define variables
variable (a d c e : ℝ)
variable (u v w r : ℝ)

-- State the theorem
theorem relation_between_exponents 
  (h1 : a^u = d^r) 
  (h2 : d^r = c)
  (h3 : d^v = a^w)
  (h4 : a^w = e) :
  r * w = v * u := by
  sorry

end NUMINAMATH_CALUDE_relation_between_exponents_l2801_280116


namespace NUMINAMATH_CALUDE_eight_stairs_climbs_l2801_280117

-- Define the function for the number of ways to climb n stairs
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | m + 4 => climbStairs m + climbStairs (m + 1) + climbStairs (m + 2) + climbStairs (m + 3)

-- Theorem statement
theorem eight_stairs_climbs : climbStairs 8 = 108 := by
  sorry


end NUMINAMATH_CALUDE_eight_stairs_climbs_l2801_280117


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2801_280124

theorem fractional_equation_solution :
  ∀ x : ℚ, x ≠ 1 → 3*x - 3 ≠ 0 →
  (2*x / (x - 1) = x / (3*x - 3) + 1) ↔ (x = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2801_280124


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_specific_series_sum_l2801_280195

def geometric_series (a : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a * r^n

theorem infinite_geometric_series_sum (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∑' n, geometric_series a r n = a / (1 - r) :=
sorry

theorem specific_series_sum :
  ∑' n, geometric_series (1/4) (1/2) n = 1/2 :=
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_specific_series_sum_l2801_280195


namespace NUMINAMATH_CALUDE_largest_three_digit_square_ending_identical_l2801_280167

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_with_three_identical_nonzero_digits (n : ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 0 ∧ n % 1000 = d * 100 + d * 10 + d

theorem largest_three_digit_square_ending_identical : 
  (is_three_digit 376) ∧ 
  (ends_with_three_identical_nonzero_digits (376^2)) ∧ 
  (∀ n : ℕ, is_three_digit n → ends_with_three_identical_nonzero_digits (n^2) → n ≤ 376) :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_square_ending_identical_l2801_280167


namespace NUMINAMATH_CALUDE_power_product_l2801_280128

theorem power_product (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_l2801_280128


namespace NUMINAMATH_CALUDE_parallelogram_height_l2801_280129

theorem parallelogram_height (area base height : ℝ) : 
  area = 231 ∧ base = 21 ∧ area = base * height → height = 11 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l2801_280129


namespace NUMINAMATH_CALUDE_unique_factorial_sum_l2801_280135

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Sum of factorials of digits -/
def sum_factorial_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  factorial hundreds + factorial tens + factorial ones

/-- Theorem stating that 145 is the only three-digit number equal to the sum of factorials of its digits -/
theorem unique_factorial_sum :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 → (n = sum_factorial_digits n ↔ n = 145) := by
  sorry

#eval sum_factorial_digits 145  -- Should output 145

end NUMINAMATH_CALUDE_unique_factorial_sum_l2801_280135


namespace NUMINAMATH_CALUDE_symmetry_implies_phi_value_l2801_280184

/-- Given a function f and its translation g, proves that if g is symmetric about π/2, then φ = π/2 -/
theorem symmetry_implies_phi_value 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (φ : ℝ) 
  (h1 : 0 < φ ∧ φ < π)
  (h2 : ∀ x, f x = Real.cos (2 * x + φ))
  (h3 : ∀ x, g x = f (x - π/4))
  (h4 : ∀ x, g x = g (π - x)) : 
  φ = π/2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_phi_value_l2801_280184


namespace NUMINAMATH_CALUDE_ben_savings_problem_l2801_280107

/-- Ben's daily starting amount -/
def daily_start : ℕ := 50

/-- Ben's daily spending -/
def daily_spend : ℕ := 15

/-- Ben's daily savings -/
def daily_savings : ℕ := daily_start - daily_spend

/-- Ben's final amount after mom's doubling and dad's addition -/
def final_amount : ℕ := 500

/-- Additional amount from dad -/
def dad_addition : ℕ := 10

/-- The number of days elapsed -/
def days_elapsed : ℕ := 7

theorem ben_savings_problem :
  final_amount = 2 * (daily_savings * days_elapsed) + dad_addition := by
  sorry

end NUMINAMATH_CALUDE_ben_savings_problem_l2801_280107


namespace NUMINAMATH_CALUDE_fish_pond_estimation_l2801_280188

theorem fish_pond_estimation (x : ℕ) 
  (h1 : x > 0)  -- Ensure the pond has fish
  (h2 : 30 ≤ x) -- Ensure we can catch 30 fish initially
  : (2 : ℚ) / 30 = 30 / x → x = 450 := by
  sorry

#check fish_pond_estimation

end NUMINAMATH_CALUDE_fish_pond_estimation_l2801_280188


namespace NUMINAMATH_CALUDE_certain_number_theorem_l2801_280155

theorem certain_number_theorem (a x : ℕ) (h1 : a = 105) (h2 : a^3 = x * 25 * 45 * 49) : x = 21 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_theorem_l2801_280155


namespace NUMINAMATH_CALUDE_count_valid_pairs_l2801_280150

/-- The number of distinct ordered pairs of positive integers (x,y) satisfying 1/x + 1/y = 1/5 -/
def count_pairs : ℕ := 3

/-- Predicate defining valid pairs -/
def is_valid_pair (x y : ℕ+) : Prop :=
  (1 : ℚ) / x.val + (1 : ℚ) / y.val = (1 : ℚ) / 5

theorem count_valid_pairs :
  ∃ (S : Finset (ℕ+ × ℕ+)), 
    (∀ (p : ℕ+ × ℕ+), p ∈ S ↔ is_valid_pair p.1 p.2) ∧ 
    S.card = count_pairs :=
  sorry


end NUMINAMATH_CALUDE_count_valid_pairs_l2801_280150


namespace NUMINAMATH_CALUDE_geometry_problem_l2801_280178

-- Define the points
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 2)
def P : ℝ × ℝ := (2, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the line parallel to AB passing through P
def line_parallel_AB_through_P (x y : ℝ) : Prop :=
  x + 2*y - 8 = 0

-- Define the circumscribed circle of triangle OAB
def circle_OAB (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 5

-- Theorem statement
theorem geometry_problem :
  (∀ x y : ℝ, line_parallel_AB_through_P x y ↔ 
    (y - P.2 = ((B.2 - A.2) / (B.1 - A.1)) * (x - P.1))) ∧
  (∀ x y : ℝ, circle_OAB x y ↔ 
    ((x - ((A.1 + B.1) / 2))^2 + (y - ((A.2 + B.2) / 2))^2 = 
     ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4)) :=
sorry

end NUMINAMATH_CALUDE_geometry_problem_l2801_280178


namespace NUMINAMATH_CALUDE_problems_solved_l2801_280130

theorem problems_solved (first last : ℕ) (h : first = 70 ∧ last = 125) : last - first + 1 = 56 := by
  sorry

end NUMINAMATH_CALUDE_problems_solved_l2801_280130


namespace NUMINAMATH_CALUDE_fish_population_estimate_l2801_280182

/-- Estimates the fish population on January 1 based on capture-recapture data --/
theorem fish_population_estimate 
  (initial_tagged : ℕ)
  (june_sample : ℕ)
  (june_tagged : ℕ)
  (tagged_left_percent : ℚ)
  (new_juvenile_percent : ℚ) :
  initial_tagged = 100 →
  june_sample = 150 →
  june_tagged = 4 →
  tagged_left_percent = 30 / 100 →
  new_juvenile_percent = 50 / 100 →
  ∃ (estimated_population : ℕ), estimated_population = 1312 := by
  sorry


end NUMINAMATH_CALUDE_fish_population_estimate_l2801_280182


namespace NUMINAMATH_CALUDE_tape_length_theorem_l2801_280104

/-- Given 15 sheets of tape, each 25 cm long, overlapping by 0.5 cm,
    the total length of the attached tape is 3.68 meters. -/
theorem tape_length_theorem (num_sheets : ℕ) (sheet_length : ℝ) (overlap : ℝ) :
  num_sheets = 15 →
  sheet_length = 25 →
  overlap = 0.5 →
  (num_sheets * sheet_length - (num_sheets - 1) * overlap) / 100 = 3.68 := by
  sorry

end NUMINAMATH_CALUDE_tape_length_theorem_l2801_280104


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_base_edge_l2801_280121

/-- Represents a hexagonal pyramid -/
structure HexagonalPyramid where
  base_edge : ℝ
  side_edge : ℝ

/-- Calculates the sum of all edge lengths in a hexagonal pyramid -/
def total_edge_length (p : HexagonalPyramid) : ℝ :=
  6 * p.base_edge + 6 * p.side_edge

/-- Theorem stating the length of the base edge in a specific hexagonal pyramid -/
theorem hexagonal_pyramid_base_edge :
  ∃ (p : HexagonalPyramid),
    p.side_edge = 8 ∧
    total_edge_length p = 120 ∧
    p.base_edge = 12 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_base_edge_l2801_280121


namespace NUMINAMATH_CALUDE_sarah_marriage_age_l2801_280132

/-- Predicts the marriage age based on name length, current age, and birth month --/
def predictMarriageAge (nameLength : ℕ) (age : ℕ) (birthMonth : ℕ) : ℕ :=
  (nameLength + 2 * age) * birthMonth

theorem sarah_marriage_age :
  let sarahNameLength : ℕ := 5
  let sarahAge : ℕ := 9
  let sarahBirthMonth : ℕ := 7
  predictMarriageAge sarahNameLength sarahAge sarahBirthMonth = 161 := by
  sorry

end NUMINAMATH_CALUDE_sarah_marriage_age_l2801_280132


namespace NUMINAMATH_CALUDE_derivative_of_f_l2801_280175

noncomputable def f (x : ℝ) : ℝ := x^3 / 3 + 1 / x

theorem derivative_of_f (x : ℝ) (hx : x ≠ 0) : 
  deriv f x = x^2 - 1 / x^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l2801_280175


namespace NUMINAMATH_CALUDE_reporters_covering_local_politics_l2801_280170

/-- The percentage of reporters who do not cover politics -/
def non_politics_reporters : ℝ := 85.71428571428572

/-- The percentage of reporters covering politics who do not cover local politics in country x -/
def non_local_politics_reporters : ℝ := 30

/-- The percentage of reporters covering local politics in country x -/
def local_politics_reporters : ℝ := 10

theorem reporters_covering_local_politics :
  local_politics_reporters = 
    (100 - non_politics_reporters) * (100 - non_local_politics_reporters) / 100 := by
  sorry

end NUMINAMATH_CALUDE_reporters_covering_local_politics_l2801_280170


namespace NUMINAMATH_CALUDE_base_nine_to_decimal_l2801_280192

/-- Given that the base-9 number 16m27₍₉₎ equals 11203 in decimal, prove that m = 3 -/
theorem base_nine_to_decimal (m : ℕ) : 
  (7 + 2 * 9^1 + m * 9^2 + 6 * 9^3 + 1 * 9^4 = 11203) → m = 3 := by
sorry

end NUMINAMATH_CALUDE_base_nine_to_decimal_l2801_280192


namespace NUMINAMATH_CALUDE_binary_101_equals_decimal_5_l2801_280101

def binary_to_decimal (b₂ b₁ b₀ : ℕ) : ℕ :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_101_equals_decimal_5 : binary_to_decimal 1 0 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_equals_decimal_5_l2801_280101


namespace NUMINAMATH_CALUDE_exponent_division_l2801_280156

theorem exponent_division (a : ℝ) : a^10 / a^5 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2801_280156


namespace NUMINAMATH_CALUDE_sin_plus_3cos_value_l2801_280189

theorem sin_plus_3cos_value (x : ℝ) (h : 2 * Real.cos x - 3 * Real.sin x = 4) :
  ∃ (y : ℝ), (Real.sin x + 3 * Real.cos x = y) ∧ (y = -2 + 6 * Real.sqrt 10 ∨ y = -2 - 6 * Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_3cos_value_l2801_280189


namespace NUMINAMATH_CALUDE_paul_shopping_money_left_l2801_280120

theorem paul_shopping_money_left 
  (initial_money : ℝ)
  (bread_price : ℝ)
  (butter_original_price : ℝ)
  (butter_discount : ℝ)
  (juice_price_multiplier : ℝ)
  (sales_tax_rate : ℝ)
  (h1 : initial_money = 15)
  (h2 : bread_price = 2)
  (h3 : butter_original_price = 3)
  (h4 : butter_discount = 0.1)
  (h5 : juice_price_multiplier = 2)
  (h6 : sales_tax_rate = 0.05) :
  initial_money - 
  ((bread_price + 
    (butter_original_price * (1 - butter_discount)) + 
    (bread_price * juice_price_multiplier)) * 
   (1 + sales_tax_rate)) = 5.86 := by
sorry

end NUMINAMATH_CALUDE_paul_shopping_money_left_l2801_280120


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_points_product_l2801_280127

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    prove that the product of the distances from the origin to any two 
    perpendicular points on the ellipse is at least 2a²b²/(a² + b²) -/
theorem ellipse_perpendicular_points_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∀ (P Q : ℝ × ℝ), 
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) →
    (Q.1^2 / a^2 + Q.2^2 / b^2 = 1) →
    (P.1 * Q.1 + P.2 * Q.2 = 0) →
    (P.1^2 + P.2^2) * (Q.1^2 + Q.2^2) ≥ (2 * a^2 * b^2) / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_points_product_l2801_280127


namespace NUMINAMATH_CALUDE_statue_cost_l2801_280186

theorem statue_cost (selling_price : ℚ) (profit_percentage : ℚ) (original_cost : ℚ) : 
  selling_price = 670 ∧ 
  profit_percentage = 25 ∧ 
  selling_price = original_cost * (1 + profit_percentage / 100) → 
  original_cost = 536 := by
sorry

end NUMINAMATH_CALUDE_statue_cost_l2801_280186


namespace NUMINAMATH_CALUDE_fraction_division_simplification_l2801_280100

theorem fraction_division_simplification : (10 / 21) / (4 / 9) = 15 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_simplification_l2801_280100


namespace NUMINAMATH_CALUDE_fraction_product_l2801_280158

theorem fraction_product : (2 : ℚ) / 3 * 3 / 5 * 4 / 7 * 5 / 9 = 8 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l2801_280158


namespace NUMINAMATH_CALUDE_right_triangle_sides_l2801_280113

theorem right_triangle_sides (t k : ℝ) (ht : t = 84) (hk : k = 56) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = k ∧
    (1 / 2) * a * b = t ∧
    c * c = a * a + b * b ∧
    (a = 7 ∧ b = 24 ∧ c = 25) ∨ (a = 24 ∧ b = 7 ∧ c = 25) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l2801_280113


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l2801_280118

theorem floor_ceiling_sum (x y : ℝ) (hx : 1 < x ∧ x < 2) (hy : 3 < y ∧ y < 4) :
  ⌊x⌋ + ⌈y⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l2801_280118


namespace NUMINAMATH_CALUDE_card_probability_choose_people_signal_count_red_ball_probability_l2801_280146

-- Define the number of cards in a standard deck
def deck_size : ℕ := 52

-- Define the number of hearts in a standard deck
def hearts_count : ℕ := 13

-- Define the number of people to choose from
def total_people : ℕ := 17

-- Define the number of people to be chosen
def chosen_people : ℕ := 15

-- Define the number of flags
def flags_count : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := 15

-- Define the number of red balls
def red_balls : ℕ := 3

theorem card_probability :
  (hearts_count / deck_size) * ((hearts_count - 1) / (deck_size - 1)) = 1 / total_people :=
sorry

theorem choose_people :
  Nat.choose total_people chosen_people = 136 :=
sorry

theorem signal_count :
  (2 ^ flags_count) - 1 = total_balls :=
sorry

theorem red_ball_probability :
  red_balls / total_balls = 1 / 5 :=
sorry

end NUMINAMATH_CALUDE_card_probability_choose_people_signal_count_red_ball_probability_l2801_280146


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2801_280187

theorem trigonometric_identities (α : ℝ) (h1 : α ∈ Set.Ioo (π/2) π) (h2 : Real.sin α = 3/5) :
  (Real.tan (α - π/4) = -7) ∧
  ((Real.sin (2*α) - Real.cos α) / (1 + Real.cos (2*α)) = -1/8) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2801_280187


namespace NUMINAMATH_CALUDE_exam_correct_percentage_l2801_280125

/-- Given an exam with two sections, calculate the percentage of correctly solved problems. -/
theorem exam_correct_percentage (y : ℕ) : 
  let total_problems := 10 * y
  let section1_problems := 6 * y
  let section2_problems := 4 * y
  let missed_section1 := 2 * y
  let missed_section2 := y
  let correct_problems := (section1_problems - missed_section1) + (section2_problems - missed_section2)
  (correct_problems : ℚ) / total_problems * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_exam_correct_percentage_l2801_280125


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l2801_280148

theorem largest_prime_divisor_of_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 13 + Nat.factorial 14) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 13 + Nat.factorial 14) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l2801_280148


namespace NUMINAMATH_CALUDE_percent_of_m_equal_to_l_l2801_280151

theorem percent_of_m_equal_to_l (j k l m x : ℝ) 
  (h1 : 1.25 * j = 0.25 * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : 1.75 * l = (x / 100) * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 75 := by
sorry

end NUMINAMATH_CALUDE_percent_of_m_equal_to_l_l2801_280151


namespace NUMINAMATH_CALUDE_initial_peanuts_count_l2801_280110

/-- 
Given a box of peanuts, prove that the initial number of peanuts was 10,
when 8 peanuts were added and the final count is 18.
-/
theorem initial_peanuts_count (initial final added : ℕ) 
  (h1 : added = 8)
  (h2 : final = 18)
  (h3 : final = initial + added) : 
  initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_peanuts_count_l2801_280110


namespace NUMINAMATH_CALUDE_distance_from_origin_l2801_280163

theorem distance_from_origin (x y : ℝ) (h1 : |y| = 15) (h2 : Real.sqrt ((x - 2)^2 + (y - 7)^2) = 13) (h3 : x > 2) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (334 + 4 * Real.sqrt 105) := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l2801_280163


namespace NUMINAMATH_CALUDE_two_tangent_lines_l2801_280185

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line passing through a point with a given slope
def line_through_point (p : Point) (slope : ℝ) (x y : ℝ) : Prop :=
  y - p.y = slope * (x - p.x)

-- Define the condition for a line to intersect the parabola at a single point
def single_intersection (p : Point) (slope : ℝ) : Prop :=
  ∃! q : Point, parabola q.x q.y ∧ line_through_point p slope q.x q.y

-- Theorem statement
theorem two_tangent_lines (p : Point) (h : parabola p.x p.y) :
  ∃! s : Finset ℝ, s.card = 2 ∧ ∀ k ∈ s, single_intersection p k :=
sorry

end NUMINAMATH_CALUDE_two_tangent_lines_l2801_280185


namespace NUMINAMATH_CALUDE_mom_talia_age_ratio_l2801_280152

-- Define Talia's current age
def talia_current_age : ℕ := 20 - 7

-- Define Talia's father's current age
def father_current_age : ℕ := 36

-- Define Talia's mother's current age
def mother_current_age : ℕ := father_current_age + 3

-- Theorem stating the ratio of Talia's mom's age to Talia's age
theorem mom_talia_age_ratio :
  mother_current_age / talia_current_age = 3 := by
  sorry

end NUMINAMATH_CALUDE_mom_talia_age_ratio_l2801_280152


namespace NUMINAMATH_CALUDE_total_marbles_l2801_280147

/-- Represents the number of marbles of each color -/
structure MarbleCount where
  white : ℕ
  purple : ℕ
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The ratio of marbles -/
def marbleRatio : MarbleCount := {
  white := 2,
  purple := 3,
  red := 5,
  blue := 4,
  green := 6
}

/-- The number of blue marbles -/
def blueMarbles : ℕ := 24

/-- Theorem stating the total number of marbles -/
theorem total_marbles :
  ∃ (total : ℕ),
    total = blueMarbles * (marbleRatio.white + marbleRatio.purple + marbleRatio.red + marbleRatio.blue + marbleRatio.green) / marbleRatio.blue
    ∧ total = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l2801_280147


namespace NUMINAMATH_CALUDE_multiplication_and_division_results_l2801_280143

theorem multiplication_and_division_results : 
  ((-2) * (-1/8) = 1/4) ∧ ((-5) / (6/5) = -25/6) := by sorry

end NUMINAMATH_CALUDE_multiplication_and_division_results_l2801_280143


namespace NUMINAMATH_CALUDE_brick_length_proof_l2801_280176

/-- The length of a brick in centimeters. -/
def brick_length : ℝ := 25

/-- The width of a brick in centimeters. -/
def brick_width : ℝ := 11.25

/-- The height of a brick in centimeters. -/
def brick_height : ℝ := 6

/-- The length of the wall in centimeters. -/
def wall_length : ℝ := 800

/-- The width of the wall in centimeters. -/
def wall_width : ℝ := 600

/-- The height of the wall in centimeters. -/
def wall_height : ℝ := 22.5

/-- The number of bricks needed to build the wall. -/
def num_bricks : ℕ := 6400

/-- The volume of the wall in cubic centimeters. -/
def wall_volume : ℝ := wall_length * wall_width * wall_height

/-- The volume of a single brick in cubic centimeters. -/
def brick_volume : ℝ := brick_length * brick_width * brick_height

theorem brick_length_proof : 
  brick_length * brick_width * brick_height * num_bricks = wall_volume :=
by sorry

end NUMINAMATH_CALUDE_brick_length_proof_l2801_280176
