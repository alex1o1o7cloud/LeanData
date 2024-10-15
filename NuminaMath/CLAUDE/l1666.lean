import Mathlib

namespace NUMINAMATH_CALUDE_initial_workers_count_l1666_166690

/-- The initial number of workers that can complete a job in 25 days, 
    where adding 10 more workers allows the job to be completed in 15 days -/
def initial_workers : ℕ :=
  sorry

/-- The total amount of work to be done -/
def total_work : ℝ :=
  sorry

theorem initial_workers_count : initial_workers = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_workers_count_l1666_166690


namespace NUMINAMATH_CALUDE_scientific_notation_exponent_is_integer_l1666_166685

theorem scientific_notation_exponent_is_integer (x : ℝ) (A : ℝ) (N : ℝ) :
  x > 10 →
  x = A * 10^N →
  1 ≤ A →
  A < 10 →
  ∃ n : ℤ, N = n := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_exponent_is_integer_l1666_166685


namespace NUMINAMATH_CALUDE_every_real_has_cube_root_real_number_line_bijection_correct_statements_l1666_166686

-- Statement 1: Every real number has a cube root
theorem every_real_has_cube_root : ∀ x : ℝ, ∃ y : ℝ, y^3 = x := by sorry

-- Statement 2: Bijection between real numbers and points on a number line
theorem real_number_line_bijection : ∃ f : ℝ → ℝ, Function.Bijective f := by sorry

-- Main theorem combining both statements
theorem correct_statements :
  (∀ x : ℝ, ∃ y : ℝ, y^3 = x) ∧ (∃ f : ℝ → ℝ, Function.Bijective f) := by sorry

end NUMINAMATH_CALUDE_every_real_has_cube_root_real_number_line_bijection_correct_statements_l1666_166686


namespace NUMINAMATH_CALUDE_penultimate_digit_of_power_of_three_is_even_l1666_166665

/-- The second to last digit of a natural number -/
def penultimate_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- Predicate for even numbers -/
def is_even (n : ℕ) : Prop :=
  ∃ k, n = 2 * k

theorem penultimate_digit_of_power_of_three_is_even (n : ℕ) (h : n ≥ 3) :
  is_even (penultimate_digit (3^n)) :=
sorry

end NUMINAMATH_CALUDE_penultimate_digit_of_power_of_three_is_even_l1666_166665


namespace NUMINAMATH_CALUDE_field_fencing_l1666_166691

/-- A rectangular field with one side of 20 feet and an area of 600 square feet
    requires 80 feet of fencing for the other three sides. -/
theorem field_fencing (length width : ℝ) : 
  length = 20 →
  length * width = 600 →
  length + 2 * width = 80 :=
by sorry

end NUMINAMATH_CALUDE_field_fencing_l1666_166691


namespace NUMINAMATH_CALUDE_university_students_count_l1666_166601

/-- Proves that given the initial ratio of male to female students and the ratio after increasing female students, the total number of students after the increase is 4800 -/
theorem university_students_count (x y : ℕ) : 
  x = 3 * y ∧ 
  9 * (y + 400) = 4 * x → 
  x + y + 400 = 4800 :=
by sorry

end NUMINAMATH_CALUDE_university_students_count_l1666_166601


namespace NUMINAMATH_CALUDE_simplify_expression_l1666_166672

theorem simplify_expression (x : ℝ) : (3*x - 6)*(x + 8) - (x + 6)*(3*x + 2) = -2*x - 60 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1666_166672


namespace NUMINAMATH_CALUDE_matrix_power_negative_identity_l1666_166611

open Matrix

theorem matrix_power_negative_identity
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (h : ∃ (n : ℕ), n ≠ 0 ∧ A ^ n = -1 • 1) :
  A ^ 2 = -1 • 1 ∨ A ^ 3 = -1 • 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_negative_identity_l1666_166611


namespace NUMINAMATH_CALUDE_total_time_is_383_l1666_166660

def total_time (mac_download : ℕ) (ny_audio_glitch : ℕ) (ny_video_glitch : ℕ) 
  (berlin_audio_glitch : ℕ) (berlin_video_glitch : ℕ) (tokyo_audio_glitch : ℕ) 
  (tokyo_video_glitch : ℕ) (sydney_audio_glitch : ℕ) : ℕ :=
  let windows_download := 3 * mac_download
  let ny_glitch_time := 2 * ny_audio_glitch + ny_video_glitch
  let ny_total := ny_glitch_time + 3 * ny_glitch_time
  let berlin_glitch_time := 3 * berlin_audio_glitch + 2 * berlin_video_glitch
  let berlin_total := berlin_glitch_time + 2 * berlin_glitch_time
  let tokyo_glitch_time := tokyo_audio_glitch + 2 * tokyo_video_glitch
  let tokyo_total := tokyo_glitch_time + 4 * tokyo_glitch_time
  let sydney_glitch_time := 2 * sydney_audio_glitch
  let sydney_total := sydney_glitch_time + 5 * sydney_glitch_time
  mac_download + windows_download + ny_total + berlin_total + tokyo_total + sydney_total

theorem total_time_is_383 : 
  total_time 10 6 8 4 5 7 9 6 = 383 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_383_l1666_166660


namespace NUMINAMATH_CALUDE_overlapped_area_of_reflected_triangle_l1666_166608

/-- Given three points O, A, and B on a coordinate plane, and the reflection of triangle OAB
    along the line y = 6 creating triangle PQR, prove that the overlapped area is 8 square units. -/
theorem overlapped_area_of_reflected_triangle (O A B : ℝ × ℝ) : 
  O = (0, 0) →
  A = (12, 2) →
  B = (0, 8) →
  let P : ℝ × ℝ := (O.1, 12 - O.2)
  let Q : ℝ × ℝ := (A.1, 12 - A.2)
  let R : ℝ × ℝ := (B.1, 12 - B.2)
  let M : ℝ × ℝ := (4, 6)
  8 = (1/2) * |M.1 - B.1| * |B.2 - R.2| := by sorry

end NUMINAMATH_CALUDE_overlapped_area_of_reflected_triangle_l1666_166608


namespace NUMINAMATH_CALUDE_track_length_proof_l1666_166637

/-- The length of the circular track -/
def track_length : ℝ := 330

/-- The distance Pamela runs before the first meeting -/
def pamela_first_meeting : ℝ := 120

/-- The additional distance Jane runs between the first and second meeting -/
def jane_additional : ℝ := 210

/-- Proves that the track length is correct given the meeting conditions -/
theorem track_length_proof :
  ∃ (pamela_speed jane_speed : ℝ),
    pamela_speed > 0 ∧ jane_speed > 0 ∧
    pamela_first_meeting / (track_length - pamela_first_meeting) = pamela_speed / jane_speed ∧
    (track_length - pamela_first_meeting + jane_additional) / (pamela_first_meeting + track_length - jane_additional) = jane_speed / pamela_speed :=
by sorry


end NUMINAMATH_CALUDE_track_length_proof_l1666_166637


namespace NUMINAMATH_CALUDE_sin_two_theta_plus_pi_sixth_l1666_166650

theorem sin_two_theta_plus_pi_sixth (θ : Real) 
  (h : 7 * Real.sqrt 3 * Real.sin θ = 1 + 7 * Real.cos θ) : 
  Real.sin (2 * θ + π / 6) = 97 / 98 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_theta_plus_pi_sixth_l1666_166650


namespace NUMINAMATH_CALUDE_spinner_final_direction_l1666_166622

-- Define the directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define the rotation function
def rotate (initial : Direction) (clockwise : Rat) (counterclockwise : Rat) : Direction :=
  sorry

-- Theorem statement
theorem spinner_final_direction :
  rotate Direction.South (7/2 : Rat) (7/4 : Rat) = Direction.East :=
sorry

end NUMINAMATH_CALUDE_spinner_final_direction_l1666_166622


namespace NUMINAMATH_CALUDE_max_product_constrained_l1666_166699

theorem max_product_constrained (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  a * b ≤ 1 / 24 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 1 ∧ a₀ * b₀ = 1 / 24 :=
sorry

end NUMINAMATH_CALUDE_max_product_constrained_l1666_166699


namespace NUMINAMATH_CALUDE_smallest_angle_sum_l1666_166669

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  sum_angles : angle_A + angle_B + angle_C = 180
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Define the problem conditions
def problem_triangle (t : Triangle) : Prop :=
  (t.angle_A = 45 ∨ t.angle_B = 45 ∨ t.angle_C = 45) ∧
  (180 - t.angle_A = 135 ∨ 180 - t.angle_B = 135 ∨ 180 - t.angle_C = 135)

-- Theorem statement
theorem smallest_angle_sum (t : Triangle) (h : problem_triangle t) :
  ∃ x y, x ≤ t.angle_A ∧ x ≤ t.angle_B ∧ x ≤ t.angle_C ∧
         y ≤ t.angle_A ∧ y ≤ t.angle_B ∧ y ≤ t.angle_C ∧
         x + y = 90 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_sum_l1666_166669


namespace NUMINAMATH_CALUDE_point_line_range_l1666_166600

theorem point_line_range (a : ℝ) : 
  (∀ x y : ℝ, (x = -3 ∧ y = -1) ∨ (x = 4 ∧ y = -6) → 
    (3*(-3) - 2*(-1) - a) * (3*4 - 2*(-6) - a) < 0) ↔ 
  -7 < a ∧ a < 24 :=
sorry

end NUMINAMATH_CALUDE_point_line_range_l1666_166600


namespace NUMINAMATH_CALUDE_difference_girls_boys_l1666_166678

/-- The number of male students in village A -/
def male_A : ℕ := 204

/-- The number of female students in village A -/
def female_A : ℕ := 468

/-- The number of male students in village B -/
def male_B : ℕ := 334

/-- The number of female students in village B -/
def female_B : ℕ := 516

/-- The number of male students in village C -/
def male_C : ℕ := 427

/-- The number of female students in village C -/
def female_C : ℕ := 458

/-- The number of male students in village D -/
def male_D : ℕ := 549

/-- The number of female students in village D -/
def female_D : ℕ := 239

/-- The total number of male students in all villages -/
def total_males : ℕ := male_A + male_B + male_C + male_D

/-- The total number of female students in all villages -/
def total_females : ℕ := female_A + female_B + female_C + female_D

/-- Theorem: The difference between the total number of girls and boys in the town is 167 -/
theorem difference_girls_boys : total_females - total_males = 167 := by
  sorry

end NUMINAMATH_CALUDE_difference_girls_boys_l1666_166678


namespace NUMINAMATH_CALUDE_f_inequality_l1666_166658

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- Define the set M
def M : Set ℝ := {x | x < -1 ∨ x > 1}

-- State the theorem
theorem f_inequality (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : f (a * b) > f a - f (-b) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1666_166658


namespace NUMINAMATH_CALUDE_part_one_part_two_l1666_166646

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 ≥ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

-- Define the complement of B in ℝ
def CompB (a : ℝ) : Set ℝ := {x | x ≤ a - 1 ∨ x ≥ a + 1}

-- Part I
theorem part_one : A ∩ (CompB 2) = {x | x ≥ 3} := by sorry

-- Part II
theorem part_two : ∀ a : ℝ, B a ⊆ A → a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1666_166646


namespace NUMINAMATH_CALUDE_subtract_repeating_third_from_four_l1666_166659

/-- The repeating decimal 0.3̅ -/
def repeating_third : ℚ := 1/3

/-- Proof that 4 - 0.3̅ = 11/3 -/
theorem subtract_repeating_third_from_four :
  4 - repeating_third = 11/3 := by sorry

end NUMINAMATH_CALUDE_subtract_repeating_third_from_four_l1666_166659


namespace NUMINAMATH_CALUDE_piecewise_representation_of_f_l1666_166614

def f (x : ℝ) := |x - 1| + 1

theorem piecewise_representation_of_f :
  ∀ x : ℝ, f x = if x ≥ 1 then x else 2 - x := by
  sorry

end NUMINAMATH_CALUDE_piecewise_representation_of_f_l1666_166614


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1666_166640

/-- An isosceles triangle with sides a, b, and c, where two sides are equal. -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

/-- The perimeter of a triangle. -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem: The perimeter of an isosceles triangle with one side equal to 4
    and another side equal to 6 is either 14 or 16. -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, ((t.a = 4 ∨ t.b = 4 ∨ t.c = 4) ∧ (t.a = 6 ∨ t.b = 6 ∨ t.c = 6)) →
  (perimeter t = 14 ∨ perimeter t = 16) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1666_166640


namespace NUMINAMATH_CALUDE_third_side_length_l1666_166689

/-- A triangle with two known sides and a known perimeter -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  perimeter : ℝ

/-- The length of the third side of a triangle -/
def third_side (t : Triangle) : ℝ := t.perimeter - t.side1 - t.side2

/-- Theorem: In a triangle with sides 5 cm and 20 cm, and perimeter 55 cm, the third side is 30 cm -/
theorem third_side_length (t : Triangle) 
  (h1 : t.side1 = 5) 
  (h2 : t.side2 = 20) 
  (h3 : t.perimeter = 55) : 
  third_side t = 30 := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_l1666_166689


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1666_166634

theorem quadratic_equation_solution :
  let f (x : ℝ) := x^2 - (2/3)*x - 1
  ∃ (x₁ x₂ : ℝ), 
    f x₁ = 0 ∧ 
    f x₂ = 0 ∧ 
    x₁ = (Real.sqrt 10)/3 + 1/3 ∧ 
    x₂ = -(Real.sqrt 10)/3 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1666_166634


namespace NUMINAMATH_CALUDE_find_original_number_l1666_166673

theorem find_original_number : 
  ∃ x : ℝ, 3 * (2 * x + 5) = 135 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_original_number_l1666_166673


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1666_166609

theorem trigonometric_identity (α : Real) :
  (2 * (Real.cos (2 * α))^2 - 1) / 
  (2 * Real.tan (π/4 - 2*α) * (Real.sin (3*π/4 - 2*α))^2) - 
  Real.tan (2*α) + Real.cos (2*α) - Real.sin (2*α) = 
  (2 * Real.sqrt 2 * Real.sin (π/4 - 2*α) * (Real.cos α)^2) / 
  Real.cos (2*α) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1666_166609


namespace NUMINAMATH_CALUDE_frequency_of_defectives_example_l1666_166656

/-- Given a sample of parts, calculate the frequency of defective parts -/
def frequency_of_defectives (total : ℕ) (defective : ℕ) : ℚ :=
  defective / total

/-- Theorem stating that for a sample of 500 parts with 8 defective parts, 
    the frequency of defective parts is 0.016 -/
theorem frequency_of_defectives_example : 
  frequency_of_defectives 500 8 = 16 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_frequency_of_defectives_example_l1666_166656


namespace NUMINAMATH_CALUDE_rosy_fish_count_l1666_166618

def lilly_fish : ℕ := 10
def total_fish : ℕ := 24

theorem rosy_fish_count : ∃ (rosy_fish : ℕ), rosy_fish = total_fish - lilly_fish ∧ rosy_fish = 14 := by
  sorry

end NUMINAMATH_CALUDE_rosy_fish_count_l1666_166618


namespace NUMINAMATH_CALUDE_equation_solutions_l1666_166667

theorem equation_solutions :
  (∃ x : ℝ, 9.9 + x = -18 ∧ x = -27.9) ∧
  (∃ x : ℝ, x - 8.8 = -8.8 ∧ x = 0) ∧
  (∃ x : ℚ, -3/4 + x = -1/4 ∧ x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1666_166667


namespace NUMINAMATH_CALUDE_log_five_twelve_l1666_166684

theorem log_five_twelve (a b : ℝ) (h1 : Real.log 2 = a * Real.log 10) (h2 : Real.log 3 = b * Real.log 10) :
  Real.log 12 / Real.log 5 = (2 * a + b) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_log_five_twelve_l1666_166684


namespace NUMINAMATH_CALUDE_missing_sale_proof_l1666_166649

/-- Calculates the missing sale amount to achieve a desired average -/
def calculate_missing_sale (sales : List ℕ) (required_sale : ℕ) (desired_average : ℕ) : ℕ :=
  let total_sales := desired_average * (sales.length + 2)
  let known_sales := sales.sum + required_sale
  total_sales - known_sales

theorem missing_sale_proof (sales : List ℕ) (required_sale : ℕ) (desired_average : ℕ) 
  (h1 : sales = [6335, 6927, 6855, 7230])
  (h2 : required_sale = 5091)
  (h3 : desired_average = 6500) :
  calculate_missing_sale sales required_sale desired_average = 6562 := by
  sorry

#eval calculate_missing_sale [6335, 6927, 6855, 7230] 5091 6500

end NUMINAMATH_CALUDE_missing_sale_proof_l1666_166649


namespace NUMINAMATH_CALUDE_weight_measurement_l1666_166641

def weights : List ℕ := [1, 3, 9, 27]

/-- The maximum weight that can be measured using the given weights -/
def max_weight : ℕ := 40

/-- The number of distinct weights that can be measured using the given weights -/
def distinct_weights : ℕ := 40

/-- Theorem stating the maximum weight and number of distinct weights that can be measured -/
theorem weight_measurement :
  (List.sum weights = max_weight) ∧
  (∀ w : ℕ, w ≤ max_weight → ∃ subset : List ℕ, subset.Sublist weights ∧ List.sum subset = w) ∧
  (distinct_weights = max_weight) := by
  sorry

end NUMINAMATH_CALUDE_weight_measurement_l1666_166641


namespace NUMINAMATH_CALUDE_circle_is_point_l1666_166613

/-- The equation of the supposed circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 2*y + 5 = 0

/-- The center of the supposed circle -/
def center : ℝ × ℝ := (-2, 1)

theorem circle_is_point :
  ∀ (x y : ℝ), circle_equation x y ↔ (x, y) = center :=
sorry

end NUMINAMATH_CALUDE_circle_is_point_l1666_166613


namespace NUMINAMATH_CALUDE_M_properties_l1666_166602

def M (n : ℕ) : ℤ := (-2) ^ n

theorem M_properties :
  (M 5 + M 6 = 32) ∧
  (2 * M 2015 + M 2016 = 0) ∧
  (∀ n : ℕ, 2 * M n + M (n + 1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_M_properties_l1666_166602


namespace NUMINAMATH_CALUDE_mike_debt_proof_l1666_166696

/-- Calculates the final amount owed after compound interest is applied -/
def final_amount (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that the final amount owed is approximately $530.604 -/
theorem mike_debt_proof (ε : ℝ) (h_ε : ε > 0) :
  ∃ (result : ℝ), 
    final_amount 500 0.02 3 = result ∧ 
    abs (result - 530.604) < ε :=
by
  sorry

#eval final_amount 500 0.02 3

end NUMINAMATH_CALUDE_mike_debt_proof_l1666_166696


namespace NUMINAMATH_CALUDE_y_derivative_is_zero_l1666_166648

noncomputable def y (x : ℝ) : ℝ :=
  5 * x - Real.log (1 + Real.sqrt (1 - Real.exp (10 * x))) - Real.exp (-5 * x) * Real.arcsin (Real.exp (5 * x))

theorem y_derivative_is_zero :
  ∀ x : ℝ, deriv y x = 0 :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_is_zero_l1666_166648


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l1666_166644

theorem min_value_sum_squares (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h : (a - 1)^3 + (b - 1)^3 ≥ 3*(2 - a - b)) : 
  a^2 + b^2 ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l1666_166644


namespace NUMINAMATH_CALUDE_a_2009_equals_7_l1666_166693

/-- Defines the array structure as described in the problem -/
def array_element (n i : ℕ) : ℚ :=
  if i ≤ n then i / (n + 1 - i) else 0

/-- Defines the index of the last element in the nth array -/
def last_index (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The main theorem stating that the 2009th element of the array is 7 -/
theorem a_2009_equals_7 : array_element 63 56 = 7 := by sorry

end NUMINAMATH_CALUDE_a_2009_equals_7_l1666_166693


namespace NUMINAMATH_CALUDE_towel_area_decrease_l1666_166621

theorem towel_area_decrease (L B : ℝ) (h_positive : L > 0 ∧ B > 0) : 
  let original_area := L * B
  let new_length := 0.8 * L
  let new_breadth := 0.8 * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.36 := by
sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l1666_166621


namespace NUMINAMATH_CALUDE_geometric_sequence_a2_a4_condition_l1666_166692

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_a2_a4_condition (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (is_monotonically_increasing a → a 2 < a 4) ∧
  ¬(a 2 < a 4 → is_monotonically_increasing a) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a2_a4_condition_l1666_166692


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l1666_166630

theorem add_preserves_inequality (a b : ℝ) (h : a > b) : 2 + a > 2 + b := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l1666_166630


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1666_166694

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Sulphur in g/mol -/
def atomic_weight_S : ℝ := 32.07

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Barium atoms in the compound -/
def num_Ba : ℕ := 1

/-- The number of Sulphur atoms in the compound -/
def num_S : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 4

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  num_Ba * atomic_weight_Ba + num_S * atomic_weight_S + num_O * atomic_weight_O

theorem compound_molecular_weight : 
  molecular_weight = 233.40 :=
by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1666_166694


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1666_166628

/-- The maximum marks for an exam -/
def maximum_marks : ℕ := 467

/-- The passing percentage as a rational number -/
def passing_percentage : ℚ := 60 / 100

/-- The marks scored by the student -/
def student_marks : ℕ := 200

/-- The marks by which the student failed -/
def failing_margin : ℕ := 80

theorem exam_maximum_marks :
  (↑maximum_marks * passing_percentage : ℚ).ceil = student_marks + failing_margin ∧
  (↑maximum_marks * passing_percentage : ℚ).ceil < ↑maximum_marks ∧
  (↑(maximum_marks - 1) * passing_percentage : ℚ).ceil < student_marks + failing_margin :=
sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1666_166628


namespace NUMINAMATH_CALUDE_gum_cost_l1666_166617

/-- Given that P packs of gum can be purchased for C coins,
    and 1 pack of gum costs 3 coins, prove that X packs of gum cost 3X coins. -/
theorem gum_cost (P C X : ℕ) (h1 : C = 3 * P) (h2 : X > 0) : 3 * X = C * X / P :=
sorry

end NUMINAMATH_CALUDE_gum_cost_l1666_166617


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1666_166624

def solution_set : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem inequality_solution_set :
  ∀ x : ℝ, (x - 2) / (x + 1) ≤ 0 ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1666_166624


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1666_166653

-- Define the hyperbola C
def hyperbola_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  y = (Real.sqrt 5 / 2) * x

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∀ x y a b : ℝ,
  hyperbola_C x y a b →
  (∃ x₀ y₀, asymptote x₀ y₀) →
  (∃ x₁ y₁, ellipse x₁ y₁ ∧ 
    (x₁ - x)^2 + (y₁ - y)^2 = (x₁ + x)^2 + (y₁ + y)^2) →
  x^2 / 4 - y^2 / 5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1666_166653


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1666_166632

theorem count_integers_satisfying_inequality :
  (Finset.filter (fun n : ℤ => (n - 3) * (n + 5) * (n + 9) < 0)
    (Finset.Icc (-13 : ℤ) 13)).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1666_166632


namespace NUMINAMATH_CALUDE_sum_of_ages_l1666_166682

/-- Given Bob's and Carol's ages, prove that their sum is 66 years. -/
theorem sum_of_ages (bob_age carol_age : ℕ) : 
  carol_age = 3 * bob_age + 2 →
  carol_age = 50 →
  bob_age = 16 →
  bob_age + carol_age = 66 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1666_166682


namespace NUMINAMATH_CALUDE_strawberry_calculation_l1666_166681

/-- Converts kilograms and grams to total grams -/
def to_grams (kg : ℕ) (g : ℕ) : ℕ := kg * 1000 + g

/-- Calculates remaining strawberries in grams -/
def remaining_strawberries (total_kg : ℕ) (total_g : ℕ) (given_kg : ℕ) (given_g : ℕ) : ℕ :=
  to_grams total_kg total_g - to_grams given_kg given_g

theorem strawberry_calculation :
  remaining_strawberries 3 300 1 900 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_calculation_l1666_166681


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l1666_166620

/-- Given a quadratic equation with complex coefficients that has real roots, 
    prove that the coefficient 'a' must be equal to -1. -/
theorem quadratic_equation_real_roots (a : ℝ) : 
  (∃ x : ℝ, (a * (1 + Complex.I)) * x^2 + (1 + a^2 * Complex.I) * x + (a^2 + Complex.I) = 0) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l1666_166620


namespace NUMINAMATH_CALUDE_inverse_inequality_for_negative_numbers_l1666_166655

theorem inverse_inequality_for_negative_numbers (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(1 / a < 1 / b) :=
by sorry

end NUMINAMATH_CALUDE_inverse_inequality_for_negative_numbers_l1666_166655


namespace NUMINAMATH_CALUDE_f_monotone_increasing_when_a_eq_1_f_range_of_a_l1666_166645

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 3 * |a - 1| * x^2 + 2 * a * x - a

-- Theorem 1: Monotonicity when a = 1
theorem f_monotone_increasing_when_a_eq_1 :
  ∀ x y : ℝ, x < y → (f 1 x) < (f 1 y) :=
sorry

-- Theorem 2: Range of a
theorem f_range_of_a :
  {a : ℝ | ∀ x ∈ Set.Icc 0 1, |f a x| ≤ f a 1} = Set.Ici (-3/4) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_when_a_eq_1_f_range_of_a_l1666_166645


namespace NUMINAMATH_CALUDE_age_puzzle_l1666_166670

theorem age_puzzle (A : ℕ) (x : ℕ) (h1 : A = 24) (h2 : x = 3) :
  4 * (A + x) - 4 * (A - 3) = A := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l1666_166670


namespace NUMINAMATH_CALUDE_add_squares_l1666_166631

theorem add_squares (a : ℝ) : 2 * a^2 + a^2 = 3 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_add_squares_l1666_166631


namespace NUMINAMATH_CALUDE_adam_final_score_l1666_166654

/-- Calculates the final score in a trivia game based on the given conditions --/
def calculate_final_score (
  science_correct : ℕ)
  (history_correct : ℕ)
  (sports_correct : ℕ)
  (literature_correct : ℕ)
  (science_points : ℕ)
  (history_points : ℕ)
  (sports_points : ℕ)
  (literature_points : ℕ)
  (history_multiplier : ℕ)
  (literature_penalty : ℕ) : ℕ :=
  science_correct * science_points +
  history_correct * history_points * history_multiplier +
  sports_correct * sports_points +
  literature_correct * (literature_points - literature_penalty)

/-- Theorem stating that Adam's final score is 99 points --/
theorem adam_final_score :
  calculate_final_score 5 3 1 1 10 5 15 7 2 3 = 99 := by
  sorry

#eval calculate_final_score 5 3 1 1 10 5 15 7 2 3

end NUMINAMATH_CALUDE_adam_final_score_l1666_166654


namespace NUMINAMATH_CALUDE_box_surface_area_is_744_l1666_166627

/-- The surface area of an open box formed by removing square corners from a rectangular sheet --/
def boxSurfaceArea (length width cornerSize : ℕ) : ℕ :=
  length * width - 4 * (cornerSize * cornerSize)

/-- Theorem stating that the surface area of the specified box is 744 square units --/
theorem box_surface_area_is_744 :
  boxSurfaceArea 40 25 8 = 744 := by
  sorry

end NUMINAMATH_CALUDE_box_surface_area_is_744_l1666_166627


namespace NUMINAMATH_CALUDE_max_a_value_l1666_166657

/-- The quadratic polynomial p(x) -/
def p (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 - (a - 1) * x + 2022

/-- The theorem stating the maximum value of a -/
theorem max_a_value : 
  (∀ x ∈ Set.Icc 0 1, -2022 ≤ p a x ∧ p a x ≤ 2022) → 
  a ≤ 16177 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l1666_166657


namespace NUMINAMATH_CALUDE_expression_evaluation_l1666_166683

theorem expression_evaluation : ((-2)^2)^(1^(0^2)) + 3^(0^(1^2)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1666_166683


namespace NUMINAMATH_CALUDE_f_formula_f_min_f_max_l1666_166666

/-- A quadratic function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The conditions for the quadratic function -/
axiom f_quad : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c ∧ a ≠ 0
axiom f_zero : f 0 = 1
axiom f_diff (x : ℝ) : f (x + 1) - f x = 2 * x

/-- Theorem: The quadratic function f(x) is x^2 - x + 1 -/
theorem f_formula : ∀ x, f x = x^2 - x + 1 := sorry

/-- Theorem: The minimum value of f(x) on [-1, 1] is 3/4 -/
theorem f_min : Set.Icc (-1 : ℝ) 1 ⊆ f ⁻¹' (Set.Icc (3/4 : ℝ) (f (-1))) := sorry

/-- Theorem: The maximum value of f(x) on [-1, 1] is 3 -/
theorem f_max : ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 3 := sorry

end NUMINAMATH_CALUDE_f_formula_f_min_f_max_l1666_166666


namespace NUMINAMATH_CALUDE_green_pill_cost_proof_l1666_166663

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℚ := 43 / 3

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℚ := green_pill_cost - 1

/-- The cost of a blue pill in dollars -/
def blue_pill_cost : ℚ := pink_pill_cost - 2

/-- The number of days in the treatment -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℚ := 819

theorem green_pill_cost_proof :
  (green_pill_cost + pink_pill_cost + blue_pill_cost) * treatment_days = total_cost ∧
  green_pill_cost = 43 / 3 := by
  sorry

#eval green_pill_cost -- To check the value

end NUMINAMATH_CALUDE_green_pill_cost_proof_l1666_166663


namespace NUMINAMATH_CALUDE_min_one_by_one_required_l1666_166652

/-- Represents a square on the grid -/
inductive Square
  | one : Square  -- 1x1 square
  | two : Square  -- 2x2 square
  | three : Square -- 3x3 square

/-- The size of the grid -/
def gridSize : Nat := 23

/-- Represents a cell on the grid -/
structure Cell where
  row : Fin gridSize
  col : Fin gridSize

/-- A covering of the grid -/
def Covering := List (Square × Cell)

/-- Checks if a covering is valid (covers all cells except one) -/
def isValidCovering (c : Covering) : Prop := sorry

/-- Checks if a covering uses only 2x2 and 3x3 squares -/
def usesOnlyTwoAndThree (c : Covering) : Prop := sorry

theorem min_one_by_one_required :
  ¬∃ (c : Covering), isValidCovering c ∧ usesOnlyTwoAndThree c :=
sorry

end NUMINAMATH_CALUDE_min_one_by_one_required_l1666_166652


namespace NUMINAMATH_CALUDE_expression_evaluation_l1666_166647

theorem expression_evaluation (a b c : ℝ) (ha : a = 12) (hb : b = 14) (hc : c = 18) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1666_166647


namespace NUMINAMATH_CALUDE_smallest_angle_tan_equation_l1666_166616

theorem smallest_angle_tan_equation (x : Real) : 
  (x > 0) →
  (x < 9 * Real.pi / 180) →
  (Real.tan (4 * x) ≠ (Real.cos x - Real.sin x) / (Real.cos x + Real.sin x)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_tan_equation_l1666_166616


namespace NUMINAMATH_CALUDE_no_ten_consecutive_power_of_two_values_l1666_166629

theorem no_ten_consecutive_power_of_two_values (a b : ℝ) : 
  ¬ ∃ (k : ℤ → ℕ) (x₀ : ℤ), ∀ x : ℤ, x₀ ≤ x ∧ x < x₀ + 10 → 
    x^2 + a*x + b = 2^(k x) :=
sorry

end NUMINAMATH_CALUDE_no_ten_consecutive_power_of_two_values_l1666_166629


namespace NUMINAMATH_CALUDE_courtyard_stones_l1666_166606

theorem courtyard_stones (stones : ℕ) (trees : ℕ) (birds : ℕ) : 
  trees = 3 * stones →
  birds = 2 * (trees + stones) →
  birds = 400 →
  stones = 40 := by
sorry

end NUMINAMATH_CALUDE_courtyard_stones_l1666_166606


namespace NUMINAMATH_CALUDE_system_unique_solution_l1666_166687

theorem system_unique_solution (a b c : ℝ) : 
  (a^2 + 3*a + 1 = (b + c) / 2) ∧ 
  (b^2 + 3*b + 1 = (a + c) / 2) ∧ 
  (c^2 + 3*c + 1 = (a + b) / 2) → 
  (a = -1 ∧ b = -1 ∧ c = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_unique_solution_l1666_166687


namespace NUMINAMATH_CALUDE_perpendicular_lines_product_sum_zero_l1666_166636

/-- Two lines in the plane -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Perpendicularity of two lines -/
def perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.A * l₂.A + l₁.B * l₂.B = 0

/-- Theorem: If two lines are perpendicular, then the sum of the products of their coefficients is zero -/
theorem perpendicular_lines_product_sum_zero (l₁ l₂ : Line) :
  perpendicular l₁ l₂ → l₁.A * l₂.A + l₁.B * l₂.B = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_product_sum_zero_l1666_166636


namespace NUMINAMATH_CALUDE_experimental_fertilizer_height_is_135_l1666_166625

-- Define the heights of plants with different fertilizers
def control_height : ℝ := 36

def bone_meal_height : ℝ := 1.25 * control_height

def cow_manure_height : ℝ := 2 * bone_meal_height

def experimental_fertilizer_height : ℝ := 1.5 * cow_manure_height

-- Theorem to prove
theorem experimental_fertilizer_height_is_135 :
  experimental_fertilizer_height = 135 := by
  sorry

end NUMINAMATH_CALUDE_experimental_fertilizer_height_is_135_l1666_166625


namespace NUMINAMATH_CALUDE_family_buffet_employees_l1666_166626

theorem family_buffet_employees (total : ℕ) (dining : ℕ) (snack : ℕ) (two_restaurants : ℕ) (all_restaurants : ℕ) : 
  total = 39 →
  dining = 18 →
  snack = 12 →
  two_restaurants = 4 →
  all_restaurants = 3 →
  ∃ family : ℕ, family = 20 ∧ 
    family + dining + snack - two_restaurants - 2 * all_restaurants + all_restaurants = total :=
by sorry

end NUMINAMATH_CALUDE_family_buffet_employees_l1666_166626


namespace NUMINAMATH_CALUDE_xyz_sum_product_l1666_166604

theorem xyz_sum_product (x y z : ℝ) 
  (h1 : 3 * (x + y + z) = x^2 + y^2 + z^2) 
  (h2 : x + y + z = 3) : 
  x * y + x * z + y * z = 0 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_product_l1666_166604


namespace NUMINAMATH_CALUDE_percentage_difference_l1666_166642

theorem percentage_difference (w q y z : ℝ) 
  (hw : w = 0.6 * q) 
  (hq : q = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  (z - w) / w = 0.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l1666_166642


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_squared_l1666_166635

theorem arithmetic_sequence_sum_squared (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  let seq := List.range n |>.map (fun i => a₁ + i * d)
  3 * (seq.sum)^2 = 1520832 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_squared_l1666_166635


namespace NUMINAMATH_CALUDE_instrument_probability_l1666_166639

theorem instrument_probability (total : ℕ) (at_least_one : ℕ) (two_or_more : ℕ) :
  total = 800 →
  at_least_one = total / 5 →
  two_or_more = 32 →
  (at_least_one - two_or_more : ℚ) / total = 1 / 6.25 := by
  sorry

end NUMINAMATH_CALUDE_instrument_probability_l1666_166639


namespace NUMINAMATH_CALUDE_investment_proof_l1666_166605

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof of the investment problem -/
theorem investment_proof :
  let principal : ℝ := 315.84
  let rate : ℝ := 0.12
  let time : ℕ := 6
  let final_value : ℝ := 635.48
  abs (compound_interest principal rate time - final_value) < 0.01 := by
sorry


end NUMINAMATH_CALUDE_investment_proof_l1666_166605


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1666_166638

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := a * x - y - a + 3 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 4 = 0

-- Theorem stating that the line intersects the circle for any real a
theorem line_intersects_circle (a : ℝ) : 
  ∃ x y : ℝ, line_equation a x y ∧ circle_equation x y := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1666_166638


namespace NUMINAMATH_CALUDE_num_planes_determined_by_skew_lines_l1666_166603

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line in 3D space
  -- (We don't need to fully implement this for the statement)

/-- A point in 3D space -/
structure Point3D where
  -- Define properties of a point in 3D space
  -- (We don't need to fully implement this for the statement)

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane in 3D space
  -- (We don't need to fully implement this for the statement)

/-- Two lines are skew if they are not parallel and do not intersect -/
def areSkewLines (l1 l2 : Line3D) : Prop :=
  sorry

/-- A point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  sorry

/-- A plane is determined by a line and a point not on that line -/
def planeDeterminedByLineAndPoint (l : Line3D) (p : Point3D) : Plane3D :=
  sorry

/-- The number of unique planes determined by two skew lines and points on them -/
def numUniquePlanes (a b : Line3D) (pointsOnA pointsOnB : Finset Point3D) : ℕ :=
  sorry

theorem num_planes_determined_by_skew_lines 
  (a b : Line3D) 
  (pointsOnA pointsOnB : Finset Point3D) 
  (h_skew : areSkewLines a b)
  (h_pointsA : ∀ p ∈ pointsOnA, pointOnLine p a)
  (h_pointsB : ∀ p ∈ pointsOnB, pointOnLine p b)
  (h_countA : pointsOnA.card = 5)
  (h_countB : pointsOnB.card = 4) :
  numUniquePlanes a b pointsOnA pointsOnB = 5 := by
  sorry

end NUMINAMATH_CALUDE_num_planes_determined_by_skew_lines_l1666_166603


namespace NUMINAMATH_CALUDE_sin_10_over_1_minus_sqrt3_tan_10_l1666_166623

theorem sin_10_over_1_minus_sqrt3_tan_10 :
  (Real.sin (10 * π / 180)) / (1 - Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_10_over_1_minus_sqrt3_tan_10_l1666_166623


namespace NUMINAMATH_CALUDE_rectangular_box_width_l1666_166633

/-- The width of a rectangular box that fits in a wooden box -/
theorem rectangular_box_width :
  let wooden_box_length : ℝ := 8 -- in meters
  let wooden_box_width : ℝ := 7 -- in meters
  let wooden_box_height : ℝ := 6 -- in meters
  let rect_box_length : ℝ := 4 / 100 -- in meters
  let rect_box_height : ℝ := 6 / 100 -- in meters
  let max_boxes : ℕ := 2000000
  ∃ (w : ℝ),
    w > 0 ∧
    (wooden_box_length * wooden_box_width * wooden_box_height) / 
    (rect_box_length * w * rect_box_height) = max_boxes ∧
    w = 7 / 100 -- width in meters
  := by sorry


end NUMINAMATH_CALUDE_rectangular_box_width_l1666_166633


namespace NUMINAMATH_CALUDE_distribute_5_4_l1666_166695

/-- The number of ways to distribute n distinct objects into k indistinguishable containers,
    allowing empty containers. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects into 4 indistinguishable containers,
    allowing empty containers, is 51. -/
theorem distribute_5_4 : distribute 5 4 = 51 := by sorry

end NUMINAMATH_CALUDE_distribute_5_4_l1666_166695


namespace NUMINAMATH_CALUDE_largest_divisor_of_f_l1666_166651

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem largest_divisor_of_f :
  ∀ m : ℕ, (∀ n : ℕ, m ∣ f n) → m ≤ 36 ∧
  ∀ n : ℕ, 36 ∣ f n :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_f_l1666_166651


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1666_166607

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → 
  x₂^2 - 3*x₂ - 1 = 0 → 
  x₁^2 + x₂^2 = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1666_166607


namespace NUMINAMATH_CALUDE_speed_of_sound_346_l1666_166671

/-- The speed of sound as a function of temperature -/
def speed_of_sound (t : ℝ) : ℝ := 0.6 * t + 331

/-- Theorem: When the speed of sound is 346 m/s, the temperature is 25°C -/
theorem speed_of_sound_346 :
  ∃ (t : ℝ), speed_of_sound t = 346 ∧ t = 25 := by
  sorry

end NUMINAMATH_CALUDE_speed_of_sound_346_l1666_166671


namespace NUMINAMATH_CALUDE_desired_average_sale_l1666_166664

def sales_first_five_months : List ℝ := [6435, 6927, 6855, 7230, 6562]
def sale_sixth_month : ℝ := 7991
def number_of_months : ℕ := 6

theorem desired_average_sale (sales : List ℝ) (sixth_sale : ℝ) (num_months : ℕ) :
  sales = sales_first_five_months →
  sixth_sale = sale_sixth_month →
  num_months = number_of_months →
  (sales.sum + sixth_sale) / num_months = 7000 := by
  sorry

end NUMINAMATH_CALUDE_desired_average_sale_l1666_166664


namespace NUMINAMATH_CALUDE_log_has_zero_in_open_interval_l1666_166661

theorem log_has_zero_in_open_interval :
  ∃ x, 0 < x ∧ x < 2 ∧ Real.log x = 0 := by sorry

end NUMINAMATH_CALUDE_log_has_zero_in_open_interval_l1666_166661


namespace NUMINAMATH_CALUDE_quadratic_solution_system_solution_l1666_166643

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 - 50 = 0

-- Define the system of equations
def system_eq (x y : ℝ) : Prop := x = 2*y + 7 ∧ 2*x + 5*y = -4

-- Theorem for the quadratic equation
theorem quadratic_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 5 ∧ 
  (∀ x : ℝ, quadratic_eq x ↔ (x = x₁ ∨ x = x₂)) :=
sorry

-- Theorem for the system of equations
theorem system_solution : 
  ∃! x y : ℝ, system_eq x y ∧ x = 3 ∧ y = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_system_solution_l1666_166643


namespace NUMINAMATH_CALUDE_marathon_runners_finished_l1666_166675

theorem marathon_runners_finished (total : ℕ) (difference : ℕ) (finished : ℕ) : 
  total = 1250 → 
  difference = 124 → 
  total = finished + (finished + difference) → 
  finished = 563 := by
sorry

end NUMINAMATH_CALUDE_marathon_runners_finished_l1666_166675


namespace NUMINAMATH_CALUDE_two_red_balls_probability_l1666_166676

/-- Represents a bag of colored balls -/
structure Bag where
  red : Nat
  white : Nat

/-- Calculates the probability of drawing a red ball from a given bag -/
def redProbability (bag : Bag) : Rat :=
  bag.red / (bag.red + bag.white)

/-- Theorem: The probability of drawing two red balls, one from each bag, is 1/9 -/
theorem two_red_balls_probability
  (bagA : Bag)
  (bagB : Bag)
  (hA : bagA = { red := 4, white := 2 })
  (hB : bagB = { red := 1, white := 5 }) :
  redProbability bagA * redProbability bagB = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_two_red_balls_probability_l1666_166676


namespace NUMINAMATH_CALUDE_part_one_part_two_l1666_166662

-- Define the propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Theorem for part (I)
theorem part_one (a x : ℝ) (h1 : a > 0) (h2 : a = 1) (h3 : p a x ∧ q x) : 
  2 < x ∧ x < 3 := by sorry

-- Theorem for part (II)
theorem part_two (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, ¬(p a x) → ¬(q x)) 
  (h3 : ∃ x, ¬(p a x) ∧ q x) : 
  1 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1666_166662


namespace NUMINAMATH_CALUDE_eight_divided_by_repeating_decimal_l1666_166612

-- Define the repeating decimal 0.888...
def repeating_decimal : ℚ := 8 / 9

-- State the theorem
theorem eight_divided_by_repeating_decimal : 8 / repeating_decimal = 9 := by
  sorry

end NUMINAMATH_CALUDE_eight_divided_by_repeating_decimal_l1666_166612


namespace NUMINAMATH_CALUDE_wednesdays_in_jan_feb_2012_l1666_166610

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in the year 2012 -/
structure Date2012 where
  month : Nat
  day : Nat

/-- Returns the day of the week for a given date in 2012 -/
def dayOfWeek (d : Date2012) : DayOfWeek :=
  sorry

/-- Returns the number of days in a given month of 2012 -/
def daysInMonth (m : Nat) : Nat :=
  sorry

/-- Counts the number of Wednesdays in a given month of 2012 -/
def countWednesdays (month : Nat) : Nat :=
  sorry

theorem wednesdays_in_jan_feb_2012 :
  (dayOfWeek ⟨1, 1⟩ = DayOfWeek.Sunday) →
  (countWednesdays 1 = 4 ∧ countWednesdays 2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_wednesdays_in_jan_feb_2012_l1666_166610


namespace NUMINAMATH_CALUDE_trig_simplification_l1666_166619

theorem trig_simplification :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) =
  Real.tan (45 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trig_simplification_l1666_166619


namespace NUMINAMATH_CALUDE_lawns_mowed_count_l1666_166674

def shoe_cost : ℕ := 95
def saving_months : ℕ := 3
def monthly_allowance : ℕ := 5
def lawn_mowing_charge : ℕ := 15
def driveway_shoveling_charge : ℕ := 7
def change_after_purchase : ℕ := 15
def driveways_shoveled : ℕ := 5

def total_money : ℕ := shoe_cost + change_after_purchase
def allowance_savings : ℕ := saving_months * monthly_allowance
def shoveling_earnings : ℕ := driveways_shoveled * driveway_shoveling_charge
def mowing_earnings : ℕ := total_money - allowance_savings - shoveling_earnings

theorem lawns_mowed_count : mowing_earnings / lawn_mowing_charge = 4 := by
  sorry

end NUMINAMATH_CALUDE_lawns_mowed_count_l1666_166674


namespace NUMINAMATH_CALUDE_pet_store_dogs_l1666_166697

theorem pet_store_dogs (dogs : ℕ) : 
  dogs + (dogs / 2) + (2 * dogs) + (3 * dogs) = 39 → dogs = 6 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l1666_166697


namespace NUMINAMATH_CALUDE_non_intersecting_paths_l1666_166615

/-- The number of non-intersecting pairs of paths on a grid -/
theorem non_intersecting_paths 
  (m n p q : ℕ+) 
  (h1 : p < m) 
  (h2 : q < n) : 
  ∃ S : ℕ, S = Nat.choose (m + n) m * Nat.choose (m + q - p) q - 
              Nat.choose (m + q) m * Nat.choose (m + n - p) n :=
by sorry

end NUMINAMATH_CALUDE_non_intersecting_paths_l1666_166615


namespace NUMINAMATH_CALUDE_lemoine_point_minimizes_distance_sum_l1666_166680

/-- Given a triangle ABC, this theorem proves that the sum of squares of distances
    from any point to the sides of the triangle is minimized when the distances
    are proportional to the sides, and this point is the Lemoine point. -/
theorem lemoine_point_minimizes_distance_sum (a b c : ℝ) (S_ABC : ℝ) :
  let f (x y z : ℝ) := x^2 + y^2 + z^2
  ∀ (x y z : ℝ), a * x + b * y + c * z = 2 * S_ABC →
  f x y z ≥ f ((2 * S_ABC * a) / (a^2 + b^2 + c^2))
              ((2 * S_ABC * b) / (a^2 + b^2 + c^2))
              ((2 * S_ABC * c) / (a^2 + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_lemoine_point_minimizes_distance_sum_l1666_166680


namespace NUMINAMATH_CALUDE_rectangle_area_l1666_166698

theorem rectangle_area (a b : ℝ) (h1 : (a + b)^2 = 16) (h2 : (a - b)^2 = 4) : a * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1666_166698


namespace NUMINAMATH_CALUDE_inequality_solution_l1666_166679

def solution_set (a : ℝ) : Set ℝ :=
  {x | x^2 + (1 - a) * x - a < 0}

theorem inequality_solution (a : ℝ) :
  solution_set a = 
    if a > -1 then
      {x | -1 < x ∧ x < a}
    else if a < -1 then
      {x | a < x ∧ x < -1}
    else
      ∅ :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1666_166679


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_l1666_166668

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the property of an isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  dist t.A t.B = dist t.A t.C

-- Define the angle measure in degrees
def angleMeasure (t : Triangle) (vertex : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem isosceles_triangle_angle (t : Triangle) :
  isIsosceles t →
  angleMeasure t t.B = 55 →
  angleMeasure t t.A = 70 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_angle_l1666_166668


namespace NUMINAMATH_CALUDE_wendy_furniture_time_l1666_166688

/-- Given the number of chairs, tables, and total time spent, 
    calculate the time spent on each piece of furniture. -/
def time_per_piece (chairs : ℕ) (tables : ℕ) (total_time : ℕ) : ℚ :=
  total_time / (chairs + tables)

/-- Theorem: For Wendy's furniture assembly, 
    the time spent on each piece is 6 minutes. -/
theorem wendy_furniture_time :
  let chairs := 4
  let tables := 4
  let total_time := 48
  time_per_piece chairs tables total_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_wendy_furniture_time_l1666_166688


namespace NUMINAMATH_CALUDE_popsicle_stick_sum_l1666_166677

/-- The sum of popsicle sticks owned by two people -/
theorem popsicle_stick_sum (gino_sticks : ℕ) (your_sticks : ℕ) 
  (h1 : gino_sticks = 63) (h2 : your_sticks = 50) : 
  gino_sticks + your_sticks = 113 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_stick_sum_l1666_166677
