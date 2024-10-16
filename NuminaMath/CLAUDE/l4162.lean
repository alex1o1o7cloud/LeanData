import Mathlib

namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l4162_416222

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*y - 27 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  C₁ A.1 A.2 ∧ C₂ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ B.1 B.2 ∧ A ≠ B

-- Define the perpendicular bisector
def perpendicular_bisector (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersecting_circles 
  (A B : ℝ × ℝ) (h : intersection_points A B) :
  ∀ x y, perpendicular_bisector x y ↔ 
    (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1) ∧
    2*x = A.1 + B.1 ∧ 2*y = A.2 + B.2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l4162_416222


namespace NUMINAMATH_CALUDE_teacher_grading_problem_l4162_416245

def remaining_problems (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) : ℕ :=
  (total_worksheets - graded_worksheets) * problems_per_worksheet

theorem teacher_grading_problem :
  remaining_problems 14 2 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_teacher_grading_problem_l4162_416245


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l4162_416256

theorem subtraction_of_decimals : 3.75 - 1.46 = 2.29 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l4162_416256


namespace NUMINAMATH_CALUDE_unique_solution_l4162_416226

theorem unique_solution : ∃! (x : ℝ), x > 0 ∧ (180 / x) + ((5 * 12) / x) + 80 = 81 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l4162_416226


namespace NUMINAMATH_CALUDE_investment_problem_l4162_416214

theorem investment_problem (x : ℝ) : 
  x > 0 ∧ 
  0.07 * x + 0.27 * 1500 = 0.22 * (x + 1500) → 
  x = 500 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l4162_416214


namespace NUMINAMATH_CALUDE_root_product_cubic_l4162_416236

theorem root_product_cubic (a b c : ℂ) : 
  (3 * a^3 - 4 * a^2 + a - 2 = 0) →
  (3 * b^3 - 4 * b^2 + b - 2 = 0) →
  (3 * c^3 - 4 * c^2 + c - 2 = 0) →
  a * b * c = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_root_product_cubic_l4162_416236


namespace NUMINAMATH_CALUDE_abfcde_perimeter_l4162_416230

/-- Represents a square with side length and perimeter -/
structure Square where
  side_length : ℝ
  perimeter : ℝ
  perimeter_eq : perimeter = 4 * side_length

/-- Represents the figure ABFCDE -/
structure ABFCDE where
  square : Square
  perimeter : ℝ

/-- The perimeter of ABFCDE is 80 inches, given a square with perimeter 64 inches -/
theorem abfcde_perimeter (s : Square) (fig : ABFCDE) 
  (h1 : s.perimeter = 64) 
  (h2 : fig.square = s) : 
  fig.perimeter = 80 :=
sorry

end NUMINAMATH_CALUDE_abfcde_perimeter_l4162_416230


namespace NUMINAMATH_CALUDE_stating_number_of_passed_candidates_l4162_416253

/-- Represents the number of candidates who passed the examination. -/
def passed_candidates : ℕ := 346

/-- Represents the total number of candidates. -/
def total_candidates : ℕ := 500

/-- Represents the average marks of all candidates. -/
def average_marks : ℚ := 60

/-- Represents the average marks of passed candidates. -/
def average_marks_passed : ℚ := 80

/-- Represents the average marks of failed candidates. -/
def average_marks_failed : ℚ := 15

/-- 
Theorem stating that the number of candidates who passed the examination is 346,
given the total number of candidates, average marks of all candidates,
average marks of passed candidates, and average marks of failed candidates.
-/
theorem number_of_passed_candidates : 
  passed_candidates = 346 ∧
  passed_candidates + (total_candidates - passed_candidates) = total_candidates ∧
  (passed_candidates * average_marks_passed + 
   (total_candidates - passed_candidates) * average_marks_failed) / total_candidates = average_marks :=
by sorry

end NUMINAMATH_CALUDE_stating_number_of_passed_candidates_l4162_416253


namespace NUMINAMATH_CALUDE_square_partition_impossibility_l4162_416286

theorem square_partition_impossibility :
  ¬ ∃ (partition : List (ℕ × ℕ)),
    (∀ (rect : ℕ × ℕ), rect ∈ partition →
      (2 * (rect.1 + rect.2) = 18 ∨ 2 * (rect.1 + rect.2) = 22 ∨ 2 * (rect.1 + rect.2) = 26)) ∧
    (List.sum (partition.map (λ rect => rect.1 * rect.2)) = 35 * 35) :=
by
  sorry


end NUMINAMATH_CALUDE_square_partition_impossibility_l4162_416286


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_k_equals_one_l4162_416223

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is parallel to c, then k = 1 -/
theorem parallel_vectors_imply_k_equals_one (a b c : ℝ × ℝ) (k : ℝ) :
  a = (Real.sqrt 3, 1) →
  b = (0, 1) →
  c = (k, Real.sqrt 3) →
  ∃ (t : ℝ), t • (a + 2 • b) = c →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_k_equals_one_l4162_416223


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4162_416259

theorem inequality_solution_set (x : ℝ) : 
  |5 - 2*x| - 1 > 0 ↔ x < 2 ∨ x > 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4162_416259


namespace NUMINAMATH_CALUDE_root_expression_value_l4162_416201

theorem root_expression_value (a : ℝ) : 
  (2 * a^2 - 7 * a - 1 = 0) → (a * (2 * a - 7) + 5 = 6) := by
  sorry

end NUMINAMATH_CALUDE_root_expression_value_l4162_416201


namespace NUMINAMATH_CALUDE_mandy_toys_count_mandy_toys_count_proof_l4162_416266

theorem mandy_toys_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun mandy anna amanda peter =>
    anna = 3 * mandy ∧
    amanda = anna + 2 ∧
    peter = 2 * anna ∧
    mandy + anna + amanda + peter = 278 →
    mandy = 21

-- The proof is omitted
theorem mandy_toys_count_proof : mandy_toys_count 21 63 65 126 := by
  sorry

end NUMINAMATH_CALUDE_mandy_toys_count_mandy_toys_count_proof_l4162_416266


namespace NUMINAMATH_CALUDE_expand_product_l4162_416284

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l4162_416284


namespace NUMINAMATH_CALUDE_find_n_l4162_416273

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem find_n : ∃ n : ℕ, n * factorial (n + 1) + factorial (n + 1) = 5040 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l4162_416273


namespace NUMINAMATH_CALUDE_sheila_picnic_probability_l4162_416269

-- Define the probabilities
def rain_probability : ℝ := 0.5
def picnic_if_rain : ℝ := 0.3
def picnic_if_sunny : ℝ := 0.7

-- Theorem statement
theorem sheila_picnic_probability :
  rain_probability * picnic_if_rain + (1 - rain_probability) * picnic_if_sunny = 0.5 := by
  sorry

#eval rain_probability * picnic_if_rain + (1 - rain_probability) * picnic_if_sunny

end NUMINAMATH_CALUDE_sheila_picnic_probability_l4162_416269


namespace NUMINAMATH_CALUDE_range_reduction_after_five_trials_l4162_416235

/-- The reduction factor for each trial using the 0.618 method -/
def reduction_factor : ℝ := 0.618

/-- The number of trials -/
def num_trials : ℕ := 5

/-- The range reduction after a given number of trials -/
def range_reduction (n : ℕ) : ℝ := reduction_factor ^ n

theorem range_reduction_after_five_trials :
  range_reduction (num_trials - 1) = reduction_factor ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_reduction_after_five_trials_l4162_416235


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l4162_416225

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number (1-m²) + (1+m)i where m is a real number -/
def complexNumber (m : ℝ) : ℂ :=
  ⟨1 - m^2, 1 + m⟩

theorem necessary_but_not_sufficient :
  (∀ m : ℝ, IsPurelyImaginary (complexNumber m) → m = 1 ∨ m = -1) ∧
  (∃ m : ℝ, (m = 1 ∨ m = -1) ∧ ¬IsPurelyImaginary (complexNumber m)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l4162_416225


namespace NUMINAMATH_CALUDE_center_square_side_length_l4162_416285

theorem center_square_side_length 
  (main_square_side : ℝ) 
  (l_shape_area_fraction : ℝ) 
  (num_l_shapes : ℕ) :
  main_square_side = 120 →
  l_shape_area_fraction = 1 / 5 →
  num_l_shapes = 4 →
  let total_area := main_square_side ^ 2
  let l_shapes_area := num_l_shapes * l_shape_area_fraction * total_area
  let center_square_area := total_area - l_shapes_area
  Real.sqrt center_square_area = 60 := by sorry

end NUMINAMATH_CALUDE_center_square_side_length_l4162_416285


namespace NUMINAMATH_CALUDE_boy_running_speed_l4162_416210

/-- Calculates the speed of a boy running around a square field -/
theorem boy_running_speed (side_length : ℝ) (time : ℝ) : 
  side_length = 35 → time = 56 → 
  ∃ (speed : ℝ), abs (speed - 9) < 0.1 ∧ 
  speed = (4 * side_length * 3600) / (time * 1000) := by
  sorry

end NUMINAMATH_CALUDE_boy_running_speed_l4162_416210


namespace NUMINAMATH_CALUDE_situps_total_l4162_416283

/-- The number of sit-ups Barney can do in one minute -/
def barney_situps : ℕ := 45

/-- The number of sit-ups Carrie can do in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can do in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The number of minutes Barney does sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie does sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie does sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := 
  barney_situps * barney_minutes + 
  carrie_situps * carrie_minutes + 
  jerrie_situps * jerrie_minutes

theorem situps_total : total_situps = 510 := by
  sorry

end NUMINAMATH_CALUDE_situps_total_l4162_416283


namespace NUMINAMATH_CALUDE_population_in_scientific_notation_l4162_416249

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a number to scientific notation -/
def toScientificNotation (n : ℝ) : ScientificNotation :=
  sorry

theorem population_in_scientific_notation :
  let population_millions : ℝ := 141178
  let population : ℝ := population_millions * 1000000
  let scientific_form := toScientificNotation population
  scientific_form.coefficient = 1.41178 ∧ scientific_form.exponent = 9 :=
sorry

end NUMINAMATH_CALUDE_population_in_scientific_notation_l4162_416249


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l4162_416202

/-- A hyperbola with equation mx^2 - y^2 = 1 and asymptotes y = ±3x has m = 9 -/
theorem hyperbola_asymptotes (m : ℝ) : 
  (∀ x y : ℝ, m * x^2 - y^2 = 1) → 
  (∀ x : ℝ, (∃ y : ℝ, y = 3 * x ∨ y = -3 * x) → m * x^2 - y^2 = 0) → 
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l4162_416202


namespace NUMINAMATH_CALUDE_smartphone_transactions_l4162_416261

def initial_price : ℝ := 300
def selling_price : ℝ := 255
def repurchase_price : ℝ := 275

theorem smartphone_transactions :
  (((initial_price - selling_price) / initial_price) * 100 = 15) ∧
  (((initial_price - repurchase_price) / repurchase_price) * 100 = 9.09) := by
sorry

end NUMINAMATH_CALUDE_smartphone_transactions_l4162_416261


namespace NUMINAMATH_CALUDE_sam_and_tina_distances_l4162_416239

/-- Calculates the distance traveled given speed and time -/
def distance (speed time : ℝ) : ℝ := speed * time

theorem sam_and_tina_distances 
  (marguerite_distance marguerite_time sam_time tina_time : ℝ) 
  (marguerite_distance_positive : marguerite_distance > 0)
  (marguerite_time_positive : marguerite_time > 0)
  (sam_time_positive : sam_time > 0)
  (tina_time_positive : tina_time > 0)
  (h_marguerite_distance : marguerite_distance = 150)
  (h_marguerite_time : marguerite_time = 3)
  (h_sam_time : sam_time = 4)
  (h_tina_time : tina_time = 2) :
  let marguerite_speed := marguerite_distance / marguerite_time
  (distance marguerite_speed sam_time = 200) ∧ 
  (distance marguerite_speed tina_time = 100) := by
  sorry

end NUMINAMATH_CALUDE_sam_and_tina_distances_l4162_416239


namespace NUMINAMATH_CALUDE_sarah_skateboard_speed_l4162_416238

/-- Given the following conditions:
1. Pete walks backwards three times faster than Susan walks forwards.
2. Tracy does one-handed cartwheels twice as fast as Susan walks forwards.
3. Mike swims eight times faster than Tracy does cartwheels.
4. Pete can walk on his hands at only one quarter of the speed that Tracy can do cartwheels.
5. Pete can ride his bike five times faster than Mike swims.
6. Pete walks on his hands at 2 miles per hour.
7. Patty can row three times faster than Pete walks backwards.
8. Sarah can skateboard six times faster than Patty rows.

Prove that Sarah can skateboard at 216 miles per hour. -/
theorem sarah_skateboard_speed :
  ∀ (pete_backward_speed pete_hand_speed pete_bike_speed susan_speed tracy_speed
     mike_speed patty_speed sarah_speed : ℝ),
  pete_backward_speed = 3 * susan_speed →
  tracy_speed = 2 * susan_speed →
  mike_speed = 8 * tracy_speed →
  pete_hand_speed = 1/4 * tracy_speed →
  pete_bike_speed = 5 * mike_speed →
  pete_hand_speed = 2 →
  patty_speed = 3 * pete_backward_speed →
  sarah_speed = 6 * patty_speed →
  sarah_speed = 216 := by
  sorry

end NUMINAMATH_CALUDE_sarah_skateboard_speed_l4162_416238


namespace NUMINAMATH_CALUDE_apple_discrepancy_l4162_416265

/-- Theorem: The number of apples used or given away exceeds the initial number of apples picked. -/
theorem apple_discrepancy (initial_apples : ℕ) (num_children : ℕ) (apples_to_friends : ℕ) 
  (apples_to_teachers : ℕ) (apples_to_staff : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) 
  (apples_for_salad : ℕ) (apples_to_sister : ℕ) : 
  initial_apples = 100 →
  num_children = 4 →
  apples_to_friends = 4 →
  apples_to_teachers = 6 →
  apples_to_staff = 2 →
  num_pies = 3 →
  apples_per_pie = 12 →
  apples_for_salad = 15 →
  apples_to_sister = 5 →
  (num_children * (apples_to_friends + apples_to_teachers + apples_to_staff) + 
   num_pies * apples_per_pie + apples_for_salad + apples_to_sister) > initial_apples :=
by sorry

end NUMINAMATH_CALUDE_apple_discrepancy_l4162_416265


namespace NUMINAMATH_CALUDE_fraction_equality_l4162_416203

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (2 * x - 3 * y) / (x + 2 * y) = 3) : 
  (x - 2 * y) / (2 * x + 3 * y) = 11 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4162_416203


namespace NUMINAMATH_CALUDE_deposit_calculation_l4162_416275

theorem deposit_calculation (total_price : ℝ) (deposit_percentage : ℝ) (remaining_amount : ℝ) :
  deposit_percentage = 0.1 →
  remaining_amount = 1170 →
  (1 - deposit_percentage) * total_price = remaining_amount →
  deposit_percentage * total_price = 130 := by
  sorry

end NUMINAMATH_CALUDE_deposit_calculation_l4162_416275


namespace NUMINAMATH_CALUDE_regression_line_properties_l4162_416220

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearRegression where
  b : ℝ  -- slope
  a : ℝ  -- intercept

/-- Represents a dataset -/
structure Dataset where
  points : List Point
  centroid : Point

/-- Checks if a point lies on the regression line -/
def pointOnLine (model : LinearRegression) (p : Point) : Prop :=
  p.y = model.b * p.x + model.a

/-- The main theorem stating that the regression line passes through the centroid
    but not necessarily through all data points -/
theorem regression_line_properties (data : Dataset) (model : LinearRegression) :
  (pointOnLine model data.centroid) ∧
  (∃ p ∈ data.points, ¬pointOnLine model p) := by
  sorry


end NUMINAMATH_CALUDE_regression_line_properties_l4162_416220


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l4162_416209

/-- Represents the distance traveled by a boat in one hour -/
def boat_distance (still_speed : ℝ) (stream_speed : ℝ) : ℝ :=
  still_speed - stream_speed

/-- The boat's speed in still water (km/hr) -/
def still_speed : ℝ := 7

/-- The distance the boat travels along the stream in one hour (km) -/
def downstream_distance : ℝ := 9

/-- The stream speed (km/hr) -/
def stream_speed : ℝ := downstream_distance - still_speed

theorem boat_upstream_distance :
  boat_distance still_speed stream_speed = 5 := by
  sorry


end NUMINAMATH_CALUDE_boat_upstream_distance_l4162_416209


namespace NUMINAMATH_CALUDE_min_people_for_valid_arrangement_l4162_416282

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any additional
    person must sit next to someone already seated. -/
def valid_arrangement (table : CircularTable) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ table.total_chairs →
    ∃ i j, i ≠ j ∧
           i ≤ table.seated_people ∧
           j ≤ table.seated_people ∧
           (k = i ∨ k = j ∨ (i < k ∧ k < j) ∨ (j < k ∧ k < i) ∨ (k < i ∧ j < k) ∨ (k < j ∧ i < k))

/-- The main theorem stating that 20 is the minimum number of people required
    for a valid arrangement on a table with 80 chairs. -/
theorem min_people_for_valid_arrangement :
  ∀ n : ℕ, n < 20 →
    ¬(valid_arrangement { total_chairs := 80, seated_people := n }) ∧
    (valid_arrangement { total_chairs := 80, seated_people := 20 }) := by
  sorry

end NUMINAMATH_CALUDE_min_people_for_valid_arrangement_l4162_416282


namespace NUMINAMATH_CALUDE_algebraic_simplification_l4162_416298

theorem algebraic_simplification (x : ℝ) : (3*x - 4)*(x + 8) - (x + 6)*(3*x + 2) = -44 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l4162_416298


namespace NUMINAMATH_CALUDE_section_b_average_weight_l4162_416216

/-- Proves that the average weight of section B is 35 kg given the conditions of the problem. -/
theorem section_b_average_weight
  (students_a : ℕ)
  (students_b : ℕ)
  (avg_weight_a : ℚ)
  (avg_weight_class : ℚ)
  (h1 : students_a = 24)
  (h2 : students_b = 16)
  (h3 : avg_weight_a = 40)
  (h4 : avg_weight_class = 38)
  : (students_a * avg_weight_a + students_b * ((students_a + students_b) * avg_weight_class - students_a * avg_weight_a) / students_b) / (students_a + students_b) = 38 ∧
    ((students_a + students_b) * avg_weight_class - students_a * avg_weight_a) / students_b = 35 :=
by sorry

end NUMINAMATH_CALUDE_section_b_average_weight_l4162_416216


namespace NUMINAMATH_CALUDE_sam_goal_impossible_l4162_416271

theorem sam_goal_impossible (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (a_grades : ℕ) :
  total_quizzes = 60 →
  goal_percentage = 85 / 100 →
  completed_quizzes = 40 →
  a_grades = 26 →
  ¬∃ (remaining_non_a : ℕ), 
    (a_grades + (total_quizzes - completed_quizzes - remaining_non_a) : ℚ) / total_quizzes ≥ goal_percentage :=
by sorry

end NUMINAMATH_CALUDE_sam_goal_impossible_l4162_416271


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l4162_416260

/-- Circle equation: x^2 + y^2 - 4x - 6y - 3 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y - 3 = 0

/-- Point M(-2, 0) -/
def point_M : ℝ × ℝ := (-2, 0)

/-- First tangent line equation: x + 2 = 0 -/
def tangent_line1 (x y : ℝ) : Prop :=
  x + 2 = 0

/-- Second tangent line equation: 7x + 24y + 14 = 0 -/
def tangent_line2 (x y : ℝ) : Prop :=
  7*x + 24*y + 14 = 0

/-- Theorem stating that the given lines are tangent to the circle through point M -/
theorem tangent_lines_to_circle :
  (∀ x y, tangent_line1 x y → circle_equation x y → x = point_M.1 ∧ y = point_M.2) ∧
  (∀ x y, tangent_line2 x y → circle_equation x y → x = point_M.1 ∧ y = point_M.2) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l4162_416260


namespace NUMINAMATH_CALUDE_pet_store_bird_count_l4162_416288

theorem pet_store_bird_count :
  ∀ (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ),
    num_cages = 8 →
    parrots_per_cage = 2 →
    parakeets_per_cage = 7 →
    num_cages * (parrots_per_cage + parakeets_per_cage) = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_bird_count_l4162_416288


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l4162_416287

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a corner cube to be removed -/
structure CornerCubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of the modified cube -/
def modifiedCubeSurfaceArea (originalCube : CubeDimensions) (cornerCube : CornerCubeDimensions) : ℝ :=
  sorry

/-- Theorem stating that the surface area of the modified cube is 96 sq.cm -/
theorem modified_cube_surface_area :
  let originalCube : CubeDimensions := ⟨4, 4, 4⟩
  let cornerCube : CornerCubeDimensions := ⟨1, 1, 1⟩
  modifiedCubeSurfaceArea originalCube cornerCube = 96 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l4162_416287


namespace NUMINAMATH_CALUDE_probability_two_one_is_four_fifths_l4162_416218

def total_balls : ℕ := 15
def black_balls : ℕ := 8
def white_balls : ℕ := 7
def drawn_balls : ℕ := 3

def probability_two_one : ℚ :=
  let total_ways := Nat.choose total_balls drawn_balls
  let two_black_one_white := Nat.choose black_balls 2 * Nat.choose white_balls 1
  let one_black_two_white := Nat.choose black_balls 1 * Nat.choose white_balls 2
  let favorable_ways := two_black_one_white + one_black_two_white
  ↑favorable_ways / ↑total_ways

theorem probability_two_one_is_four_fifths :
  probability_two_one = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_one_is_four_fifths_l4162_416218


namespace NUMINAMATH_CALUDE_dress_discount_percentage_l4162_416206

theorem dress_discount_percentage (d : ℝ) (x : ℝ) (h : d > 0) :
  d * ((100 - x) / 100) * 0.5 = 0.225 * d → x = 55 := by
sorry

end NUMINAMATH_CALUDE_dress_discount_percentage_l4162_416206


namespace NUMINAMATH_CALUDE_pairwise_sum_product_inequality_l4162_416293

theorem pairwise_sum_product_inequality 
  (x : Fin 64 → ℝ) 
  (h_pos : ∀ i, x i > 0) 
  (h_strict_mono : StrictMono x) : 
  (x 63 * x 64) / (x 0 * x 1) > (x 63 + x 64) / (x 0 + x 1) := by
  sorry

end NUMINAMATH_CALUDE_pairwise_sum_product_inequality_l4162_416293


namespace NUMINAMATH_CALUDE_cube_center_pyramids_l4162_416211

/-- Given a cube with edge length a, prove the volume and surface area of the pyramids formed by connecting the center to all vertices. -/
theorem cube_center_pyramids (a : ℝ) (h : a > 0) :
  ∃ (volume surface_area : ℝ),
    volume = a^3 / 6 ∧
    surface_area = a^2 * (1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_cube_center_pyramids_l4162_416211


namespace NUMINAMATH_CALUDE_some_students_not_club_members_l4162_416240

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (Punctual : U → Prop)
variable (ClubMember : U → Prop)
variable (FraternityMember : U → Prop)

-- Define the conditions
variable (h1 : ∃ x, Student x ∧ ¬Punctual x)
variable (h2 : ∀ x, ClubMember x → Punctual x)
variable (h3 : ∀ x, FraternityMember x → ¬ClubMember x)

-- State the theorem
theorem some_students_not_club_members :
  ∃ x, Student x ∧ ¬ClubMember x :=
sorry

end NUMINAMATH_CALUDE_some_students_not_club_members_l4162_416240


namespace NUMINAMATH_CALUDE_board_length_proof_l4162_416224

theorem board_length_proof :
  ∀ (short_piece long_piece total_length : ℝ),
  short_piece > 0 →
  long_piece = 2 * short_piece →
  long_piece = 46 →
  total_length = short_piece + long_piece →
  total_length = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_board_length_proof_l4162_416224


namespace NUMINAMATH_CALUDE_riku_stickers_comparison_l4162_416272

/-- The number of stickers Kristoff has -/
def kristoff_stickers : ℕ := 85

/-- The number of stickers Riku has -/
def riku_stickers : ℕ := 2210

/-- The number of times Riku has more stickers than Kristoff -/
def times_more_stickers : ℚ := riku_stickers / kristoff_stickers

/-- Theorem stating that Riku has 26 times more stickers than Kristoff -/
theorem riku_stickers_comparison : times_more_stickers = 26 := by
  sorry

end NUMINAMATH_CALUDE_riku_stickers_comparison_l4162_416272


namespace NUMINAMATH_CALUDE_worker_selection_probability_l4162_416207

theorem worker_selection_probability 
  (total_workers : ℕ) 
  (eliminated_workers : ℕ) 
  (remaining_workers : ℕ) 
  (representatives : ℕ) 
  (h1 : total_workers = 2009)
  (h2 : eliminated_workers = 9)
  (h3 : remaining_workers = 2000)
  (h4 : representatives = 100)
  (h5 : remaining_workers = total_workers - eliminated_workers) :
  (representatives : ℚ) / (total_workers : ℚ) = 100 / 2009 :=
by sorry

end NUMINAMATH_CALUDE_worker_selection_probability_l4162_416207


namespace NUMINAMATH_CALUDE_existence_of_special_number_l4162_416276

/-- Given a positive integer, returns the sum of its digits. -/
def sum_of_digits (m : ℕ+) : ℕ := sorry

/-- Given a positive integer, returns the number of its digits. -/
def num_digits (m : ℕ+) : ℕ := sorry

/-- Checks if all digits of a positive integer are non-zero. -/
def all_digits_nonzero (m : ℕ+) : Prop := sorry

theorem existence_of_special_number :
  ∀ n : ℕ+, ∃ m : ℕ+,
    (num_digits m = n) ∧
    (all_digits_nonzero m) ∧
    (m.val % sum_of_digits m = 0) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l4162_416276


namespace NUMINAMATH_CALUDE_house_market_value_l4162_416297

/-- Proves that the market value of a house is $500,000 given the specified conditions --/
theorem house_market_value : 
  ∀ (market_value selling_price revenue_per_person : ℝ),
  selling_price = market_value * 1.2 →
  selling_price = 4 * revenue_per_person →
  revenue_per_person * 0.9 = 135000 →
  market_value = 500000 := by
  sorry

end NUMINAMATH_CALUDE_house_market_value_l4162_416297


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4162_416227

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 5 + a 6 + a 7 = 15) : 
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4162_416227


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_l4162_416250

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the directrices of C₁
def l₁ : ℝ := -4
def l₂ : ℝ := 4

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = -16 * x

-- Define the intersection points A and B
def A : ℝ × ℝ := (-4, 8)
def B : ℝ × ℝ := (-4, -8)

-- Theorem statement
theorem ellipse_parabola_intersection :
  (∀ x y, C₁ x y → (x = l₁ ∨ x = l₂)) ∧
  (∀ x y, C₂ x y → (x = 0 ∨ x = l₂)) ∧
  (C₂ A.1 A.2 ∧ C₂ B.1 B.2) ∧
  (A.1 = l₁ ∧ B.1 = l₁) →
  (∀ x y, C₂ x y ↔ y^2 = -16 * x) ∧
  (A.2 - B.2 = 16) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_l4162_416250


namespace NUMINAMATH_CALUDE_surface_area_unchanged_after_cube_removal_l4162_416228

theorem surface_area_unchanged_after_cube_removal 
  (l w h : ℝ) (cube_side : ℝ) 
  (hl : l = 10) (hw : w = 5) (hh : h = 3) (hc : cube_side = 2) : 
  2 * (l * w + l * h + w * h) = 
  2 * (l * w + l * h + w * h) - 3 * cube_side^2 + 3 * cube_side^2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_after_cube_removal_l4162_416228


namespace NUMINAMATH_CALUDE_function_properties_l4162_416280

def f (x a : ℝ) : ℝ := (4*a + 2)*x^2 + (9 - 6*a)*x - 4*a + 4

theorem function_properties :
  (∀ a : ℝ, ∃ x : ℝ, f x a = 0) ∧
  (∃ a : ℤ, ∃ x : ℤ, f (x : ℝ) (a : ℝ) = 0) ∧
  ({a : ℤ | ∃ x : ℤ, f (x : ℝ) (a : ℝ) = 0} = {-2, -1, 0, 1}) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4162_416280


namespace NUMINAMATH_CALUDE_edwards_purchases_cost_edwards_total_cost_l4162_416291

/-- Calculates the total cost of Edward's purchases after applying a discount -/
theorem edwards_purchases_cost (board_game_cost : ℝ) (action_figure_cost : ℝ) 
  (action_figure_count : ℕ) (puzzle_cost : ℝ) (card_deck_cost : ℝ) 
  (discount_percentage : ℝ) : ℝ :=
  let total_action_figures_cost := action_figure_cost * action_figure_count
  let discount_amount := total_action_figures_cost * (discount_percentage / 100)
  let discounted_action_figures_cost := total_action_figures_cost - discount_amount
  board_game_cost + discounted_action_figures_cost + puzzle_cost + card_deck_cost

/-- Proves that Edward's total purchase cost is $36.70 -/
theorem edwards_total_cost : 
  edwards_purchases_cost 2 7 4 6 3.5 10 = 36.7 := by
  sorry

end NUMINAMATH_CALUDE_edwards_purchases_cost_edwards_total_cost_l4162_416291


namespace NUMINAMATH_CALUDE_variance_of_letters_l4162_416233

def letters : List ℕ := [10, 6, 8, 5, 6]

def mean (xs : List ℕ) : ℚ :=
  (xs.sum : ℚ) / xs.length

def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (fun x => ((x : ℚ) - m) ^ 2)).sum / xs.length

theorem variance_of_letters : variance letters = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_letters_l4162_416233


namespace NUMINAMATH_CALUDE_diamond_two_seven_l4162_416208

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 3 * x + 5 * y

-- State the theorem
theorem diamond_two_seven : diamond 2 7 = 41 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_seven_l4162_416208


namespace NUMINAMATH_CALUDE_complement_union_M_N_l4162_416247

-- Define the universe U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) ≠ 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ≠ p.1 + 1}

-- Theorem statement
theorem complement_union_M_N : 
  (U \ (M ∪ N)) = {(2, 3)} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l4162_416247


namespace NUMINAMATH_CALUDE_expression_equals_one_l4162_416290

theorem expression_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a - b + c = 0) :
  (a^2 * b^2) / ((a^2 + b*c) * (b^2 + a*c)) +
  (a^2 * c^2) / ((a^2 + b*c) * (c^2 + a*b)) +
  (b^2 * c^2) / ((b^2 + a*c) * (c^2 + a*b)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l4162_416290


namespace NUMINAMATH_CALUDE_fifteen_team_league_games_l4162_416294

/-- The number of games played in a league where each team plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league with 15 teams, where each team plays every other team once,
    the total number of games played is 105 -/
theorem fifteen_team_league_games :
  games_played 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_team_league_games_l4162_416294


namespace NUMINAMATH_CALUDE_no_integer_solutions_binomial_power_l4162_416295

theorem no_integer_solutions_binomial_power (n k m t : ℕ) (l : ℕ) (h1 : l ≥ 2) (h2 : 4 ≤ k) (h3 : k ≤ n - 4) :
  Nat.choose n k ≠ m ^ t := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_binomial_power_l4162_416295


namespace NUMINAMATH_CALUDE_difference_of_squares_l4162_416289

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4162_416289


namespace NUMINAMATH_CALUDE_non_sunday_average_is_240_l4162_416213

/-- Represents the average number of visitors to a library on different days. -/
structure LibraryVisitors where
  sunday : ℕ
  otherDays : ℕ
  monthlyAverage : ℕ

/-- Calculates the average number of visitors on non-Sunday days given the conditions. -/
def calculateNonSundayAverage (v : LibraryVisitors) : ℕ :=
  ((v.monthlyAverage * 30) - (v.sunday * 5)) / 25

/-- Theorem stating that under the given conditions, the average number of visitors
    on non-Sunday days is 240. -/
theorem non_sunday_average_is_240 (v : LibraryVisitors)
  (h1 : v.sunday = 600)
  (h2 : v.monthlyAverage = 300) :
  calculateNonSundayAverage v = 240 := by
  sorry

#eval calculateNonSundayAverage ⟨600, 0, 300⟩

end NUMINAMATH_CALUDE_non_sunday_average_is_240_l4162_416213


namespace NUMINAMATH_CALUDE_waiter_customer_count_l4162_416237

theorem waiter_customer_count :
  ∀ (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ),
    num_tables = 7 →
    women_per_table = 7 →
    men_per_table = 2 →
    num_tables * (women_per_table + men_per_table) = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_waiter_customer_count_l4162_416237


namespace NUMINAMATH_CALUDE_linear_function_unique_solution_l4162_416200

/-- Given a linear function f(x) = ax + 19 where f(3) = 7, 
    prove that if f(t) = 15, then t = 1 -/
theorem linear_function_unique_solution 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = a * x + 19) 
  (h2 : f 3 = 7) 
  (t : ℝ) 
  (h3 : f t = 15) : 
  t = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_function_unique_solution_l4162_416200


namespace NUMINAMATH_CALUDE_interest_rate_increase_l4162_416205

/-- Proves that if an interest rate increases by 10 percent to become 11 percent,
    the original interest rate was 10 percent. -/
theorem interest_rate_increase (original_rate : ℝ) : 
  (original_rate * 1.1 = 0.11) → (original_rate = 0.1) := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_increase_l4162_416205


namespace NUMINAMATH_CALUDE_poster_cost_l4162_416278

theorem poster_cost (initial_amount : ℕ) (notebook_cost : ℕ) (bookmark_cost : ℕ)
  (poster_count : ℕ) (leftover : ℕ) :
  initial_amount = 40 →
  notebook_cost = 12 →
  bookmark_cost = 4 →
  poster_count = 2 →
  leftover = 14 →
  (initial_amount - notebook_cost - bookmark_cost - leftover) / poster_count = 13 := by
  sorry

end NUMINAMATH_CALUDE_poster_cost_l4162_416278


namespace NUMINAMATH_CALUDE_series_equation_solutions_l4162_416262

def series_sum (x : ℝ) : ℝ := 1 + 3*x + 7*x^2 + 11*x^3 + 15*x^4 + 19*x^5 + 23*x^6 + 27*x^7 + 31*x^8 + 35*x^9 + 39*x^10

theorem series_equation_solutions :
  ∃ (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧
    -1 < x₁ ∧ x₁ < 1 ∧
    -1 < x₂ ∧ x₂ < 1 ∧
    series_sum x₁ = 50 ∧
    series_sum x₂ = 50 ∧
    abs (x₁ - 0.959) < 0.001 ∧
    abs (x₂ - 0.021) < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_series_equation_solutions_l4162_416262


namespace NUMINAMATH_CALUDE_quadratic_factorization_l4162_416242

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l4162_416242


namespace NUMINAMATH_CALUDE_number_count_l4162_416257

theorem number_count (average_all : ℝ) (average_group1 : ℝ) (average_group2 : ℝ) (average_group3 : ℝ) 
  (h1 : average_all = 3.9)
  (h2 : average_group1 = 3.4)
  (h3 : average_group2 = 3.85)
  (h4 : average_group3 = 4.45) :
  ∃ (n : ℕ), n = 6 ∧ (n : ℝ) * average_all = 2 * (average_group1 + average_group2 + average_group3) := by
  sorry

end NUMINAMATH_CALUDE_number_count_l4162_416257


namespace NUMINAMATH_CALUDE_percentage_of_2_to_50_l4162_416241

theorem percentage_of_2_to_50 : (2 : ℝ) / 50 * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_2_to_50_l4162_416241


namespace NUMINAMATH_CALUDE_fabric_cutting_l4162_416254

theorem fabric_cutting (fabric_length fabric_width dress_length dress_width : ℕ) 
  (h1 : fabric_length = 140)
  (h2 : fabric_width = 75)
  (h3 : dress_length = 45)
  (h4 : dress_width = 26)
  : ∃ (n : ℕ), n ≥ 8 ∧ n * dress_length * dress_width ≤ fabric_length * fabric_width := by
  sorry

end NUMINAMATH_CALUDE_fabric_cutting_l4162_416254


namespace NUMINAMATH_CALUDE_heartsuit_three_five_l4162_416232

-- Define the heartsuit operation
def heartsuit (x y : ℤ) : ℤ := 4*x + 6*y

-- Theorem statement
theorem heartsuit_three_five : heartsuit 3 5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_five_l4162_416232


namespace NUMINAMATH_CALUDE_cody_purchase_tax_rate_l4162_416246

/-- Proves that the tax rate is 5% given the conditions of Cody's purchase --/
theorem cody_purchase_tax_rate 
  (initial_purchase : ℝ)
  (post_tax_discount : ℝ)
  (cody_payment : ℝ)
  (h1 : initial_purchase = 40)
  (h2 : post_tax_discount = 8)
  (h3 : cody_payment = 17)
  : ∃ (tax_rate : ℝ), 
    tax_rate = 0.05 ∧ 
    (initial_purchase + initial_purchase * tax_rate - post_tax_discount) / 2 = cody_payment :=
by sorry

end NUMINAMATH_CALUDE_cody_purchase_tax_rate_l4162_416246


namespace NUMINAMATH_CALUDE_total_tickets_sold_l4162_416231

def total_revenue : ℕ := 1933
def student_ticket_price : ℕ := 2
def nonstudent_ticket_price : ℕ := 3
def student_tickets_sold : ℕ := 530

theorem total_tickets_sold :
  ∃ (nonstudent_tickets : ℕ),
    student_tickets_sold * student_ticket_price +
    nonstudent_tickets * nonstudent_ticket_price = total_revenue ∧
    student_tickets_sold + nonstudent_tickets = 821 :=
by sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l4162_416231


namespace NUMINAMATH_CALUDE_midpoint_quadrilateral_area_in_regular_hexagon_l4162_416258

/-- Represents a regular hexagon -/
structure RegularHexagon :=
  (side_length : ℝ)

/-- Represents the quadrilateral formed by joining midpoints of non-adjacent sides -/
structure MidpointQuadrilateral :=
  (hexagon : RegularHexagon)

/-- The area of the midpoint quadrilateral in a regular hexagon -/
def midpoint_quadrilateral_area (q : MidpointQuadrilateral) : ℝ :=
  q.hexagon.side_length * q.hexagon.side_length

theorem midpoint_quadrilateral_area_in_regular_hexagon 
  (h : RegularHexagon) 
  (hside : h.side_length = 12) :
  midpoint_quadrilateral_area ⟨h⟩ = 144 := by
  sorry

#check midpoint_quadrilateral_area_in_regular_hexagon

end NUMINAMATH_CALUDE_midpoint_quadrilateral_area_in_regular_hexagon_l4162_416258


namespace NUMINAMATH_CALUDE_pump_rate_calculation_l4162_416229

/-- Given two pumps operating for a total of 6 hours, with one pump rated at 250 gallons per hour
    and used for 3.5 hours, and a total volume pumped of 1325 gallons, the rate of the other pump
    is 180 gallons per hour. -/
theorem pump_rate_calculation (total_time : ℝ) (total_volume : ℝ) (pump2_rate : ℝ) (pump2_time : ℝ)
    (h1 : total_time = 6)
    (h2 : total_volume = 1325)
    (h3 : pump2_rate = 250)
    (h4 : pump2_time = 3.5) :
    (total_volume - pump2_rate * pump2_time) / (total_time - pump2_time) = 180 :=
by sorry

end NUMINAMATH_CALUDE_pump_rate_calculation_l4162_416229


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l4162_416268

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 105)
  (h_sum2 : a 2 + a 4 + a 6 = 99) :
  a 20 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l4162_416268


namespace NUMINAMATH_CALUDE_triangle_problem_l4162_416212

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  a = Real.sqrt 7 →
  b = 2 →
  A = 60 * π / 180 →  -- Convert 60° to radians
  -- Triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Positive side lengths
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Sum of angles in a triangle is π
  A + B + C = π →
  -- Sine law
  a / Real.sin A = b / Real.sin B →
  -- Cosine law
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →
  -- Conclusions
  Real.sin B = Real.sqrt 21 / 7 ∧ c = 3 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l4162_416212


namespace NUMINAMATH_CALUDE_distance_sum_bounds_l4162_416234

/-- Given three mutually perpendicular segments with lengths a, b, and c,
    this theorem proves the bounds for the sum of distances from the endpoints
    to any line passing through the origin. -/
theorem distance_sum_bounds
  (a b c : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_order : a ≤ b ∧ b ≤ c) :
  ∀ (α β γ : ℝ),
    (α^2 + β^2 + γ^2 = 1) →
    (a * α + b * β + c * γ ≥ a + b) ∧
    (a * α + b * β + c * γ ≤ c + Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_distance_sum_bounds_l4162_416234


namespace NUMINAMATH_CALUDE_smith_inheritance_l4162_416217

theorem smith_inheritance (federal_tax_rate state_tax_rate total_taxes : ℚ) 
  (h1 : federal_tax_rate = 25/100)
  (h2 : state_tax_rate = 12/100)
  (h3 : total_taxes = 15600) :
  ∃ inheritance : ℚ, 
    inheritance * federal_tax_rate + 
    (inheritance - inheritance * federal_tax_rate) * state_tax_rate = total_taxes ∧
    inheritance = 45882 := by
  sorry

end NUMINAMATH_CALUDE_smith_inheritance_l4162_416217


namespace NUMINAMATH_CALUDE_checker_moves_10_l4162_416292

/-- Represents the number of ways a checker can move n cells -/
def checkerMoves : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => checkerMoves (n + 1) + checkerMoves n

/-- Theorem stating that the number of ways a checker can move 10 cells is 89 -/
theorem checker_moves_10 : checkerMoves 10 = 89 := by
  sorry

#eval checkerMoves 10

end NUMINAMATH_CALUDE_checker_moves_10_l4162_416292


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4162_416274

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying the condition
    2(a_1 + a_3 + a_5) + 3(a_8 + a_10) = 36, prove that a_6 = 3. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_condition : 2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36) :
  a 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4162_416274


namespace NUMINAMATH_CALUDE_largest_divisor_of_1615_l4162_416279

theorem largest_divisor_of_1615 (n : ℕ) : n ≤ 5 ↔ n * 1615 ≤ 8640 ∧ n * 1615 ≥ 1000 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_1615_l4162_416279


namespace NUMINAMATH_CALUDE_square_roots_problem_l4162_416299

theorem square_roots_problem (n : ℝ) (h : n > 0) :
  (∃ a : ℝ, (a + 2)^2 = n ∧ (2*a - 11)^2 = n) → n = 225 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l4162_416299


namespace NUMINAMATH_CALUDE_arthurs_purchases_l4162_416215

/-- The cost of Arthur's purchases on two days -/
theorem arthurs_purchases (hamburger_price : ℚ) :
  (3 * hamburger_price + 4 * 1 = 10) →
  (2 * hamburger_price + 3 * 1 = 7) :=
by sorry

end NUMINAMATH_CALUDE_arthurs_purchases_l4162_416215


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l4162_416270

-- Define the opposite of a rational number
def opposite (x : ℚ) : ℚ := -x

-- Theorem statement
theorem opposite_of_negative_two_thirds : 
  opposite (-2/3) = 2/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l4162_416270


namespace NUMINAMATH_CALUDE_only_solutions_are_72_and_88_l4162_416243

/-- The product of digits of a positive integer -/
def product_of_digits (k : ℕ+) : ℕ :=
  sorry

/-- The main theorem stating that 72 and 88 are the only solutions -/
theorem only_solutions_are_72_and_88 :
  ∀ k : ℕ+, (product_of_digits k = (25 * k : ℚ) / 8 - 211) ↔ (k = 72 ∨ k = 88) :=
by sorry

end NUMINAMATH_CALUDE_only_solutions_are_72_and_88_l4162_416243


namespace NUMINAMATH_CALUDE_total_distance_is_62_l4162_416255

/-- Calculates the total distance walked over three days given specific conditions --/
def total_distance_walked (day1_distance : ℕ) (day1_speed : ℕ) : ℕ :=
  let day1_hours := day1_distance / day1_speed
  let day2_hours := day1_hours - 1
  let day2_speed := day1_speed + 1
  let day2_distance := day2_hours * day2_speed
  let day3_hours := day1_hours
  let day3_speed := day2_speed
  let day3_distance := day3_hours * day3_speed
  day1_distance + day2_distance + day3_distance

/-- Theorem stating that the total distance walked is 62 miles --/
theorem total_distance_is_62 : total_distance_walked 18 3 = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_62_l4162_416255


namespace NUMINAMATH_CALUDE_flammable_ice_scientific_notation_l4162_416251

theorem flammable_ice_scientific_notation :
  (800 * 10^9 : ℝ) = 8 * 10^11 := by sorry

end NUMINAMATH_CALUDE_flammable_ice_scientific_notation_l4162_416251


namespace NUMINAMATH_CALUDE_circle_relationship_l4162_416252

-- Define the circles and point
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle_O2 (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 1
def point_on_O1 (x y : ℝ) : Prop := circle_O1 x y

-- Define the condition for P and Q
def condition_PQ (x₁ y₁ a b : ℝ) : Prop := (a - x₁)^2 + (b - y₁)^2 = 1

-- Define the possible relationships
inductive CircleRelationship
  | ExternallyTangent
  | Intersecting
  | InternallyTangent

-- Theorem statement
theorem circle_relationship 
  (x₁ y₁ a b : ℝ) 
  (h1 : point_on_O1 x₁ y₁) 
  (h2 : condition_PQ x₁ y₁ a b) : 
  ∃ r : CircleRelationship, r = CircleRelationship.ExternallyTangent ∨ 
                            r = CircleRelationship.Intersecting ∨ 
                            r = CircleRelationship.InternallyTangent :=
sorry

end NUMINAMATH_CALUDE_circle_relationship_l4162_416252


namespace NUMINAMATH_CALUDE_store_shelves_proof_l4162_416267

/-- Calculates the number of shelves needed to store coloring books -/
def shelves_needed (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : ℕ :=
  (initial_stock - books_sold) / books_per_shelf

/-- Proves that the number of shelves needed is 7 given the problem conditions -/
theorem store_shelves_proof :
  shelves_needed 86 37 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_store_shelves_proof_l4162_416267


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4162_416244

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₁₃ + a₅ = 32,
    prove that a₉ = 16 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 13 + a 5 = 32) : 
  a 9 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4162_416244


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l4162_416281

theorem photo_arrangement_count :
  let total_people : ℕ := 7
  let adjacent_pair : ℕ := 1  -- A and B treated as one unit
  let separated_pair : ℕ := 2  -- C and D
  let other_people : ℕ := total_people - adjacent_pair - separated_pair
  
  let total_elements : ℕ := adjacent_pair + other_people + 1
  let adjacent_pair_arrangements : ℕ := 2  -- A and B can switch
  let spaces_for_separated : ℕ := total_elements + 1

  (total_elements.factorial * adjacent_pair_arrangements * 
   (spaces_for_separated * (spaces_for_separated - 1))) = 960 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l4162_416281


namespace NUMINAMATH_CALUDE_season_games_l4162_416264

/-- Represents the basketball season statistics --/
structure SeasonStats where
  total_points : ℕ
  avg_free_throws : ℕ
  avg_two_pointers : ℕ
  avg_three_pointers : ℕ

/-- Calculates the number of games in the season --/
def calculate_games (stats : SeasonStats) : ℕ :=
  stats.total_points / (stats.avg_free_throws + 2 * stats.avg_two_pointers + 3 * stats.avg_three_pointers)

/-- Theorem stating that the number of games in the season is 15 --/
theorem season_games (stats : SeasonStats) 
  (h1 : stats.total_points = 345)
  (h2 : stats.avg_free_throws = 4)
  (h3 : stats.avg_two_pointers = 5)
  (h4 : stats.avg_three_pointers = 3) :
  calculate_games stats = 15 := by
  sorry

end NUMINAMATH_CALUDE_season_games_l4162_416264


namespace NUMINAMATH_CALUDE_f_is_quadratic_l4162_416277

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 5

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

#check f_is_quadratic

end NUMINAMATH_CALUDE_f_is_quadratic_l4162_416277


namespace NUMINAMATH_CALUDE_brian_read_75_chapters_l4162_416221

/-- The total number of chapters Brian read -/
def total_chapters : ℕ :=
  let book1 : ℕ := 20
  let book2 : ℕ := 15
  let book3 : ℕ := 15
  let first_three : ℕ := book1 + book2 + book3
  let book4 : ℕ := first_three / 2
  book1 + book2 + book3 + book4

/-- Proof that Brian read 75 chapters in total -/
theorem brian_read_75_chapters : total_chapters = 75 := by
  sorry

end NUMINAMATH_CALUDE_brian_read_75_chapters_l4162_416221


namespace NUMINAMATH_CALUDE_library_charge_calculation_l4162_416296

/-- Calculates the total amount paid for borrowed books --/
def total_amount_paid (daily_rate : ℚ) (book1_days : ℕ) (book2_days : ℕ) (num_books2 : ℕ) : ℚ :=
  daily_rate * book1_days + daily_rate * book2_days * num_books2

theorem library_charge_calculation :
  let daily_rate : ℚ := 50 / 100  -- 50 cents in dollars
  let book1_days : ℕ := 20
  let book2_days : ℕ := 31
  let num_books2 : ℕ := 2
  total_amount_paid daily_rate book1_days book2_days num_books2 = 41 := by
sorry

#eval total_amount_paid (50 / 100) 20 31 2

end NUMINAMATH_CALUDE_library_charge_calculation_l4162_416296


namespace NUMINAMATH_CALUDE_min_sum_reciprocal_one_l4162_416204

theorem min_sum_reciprocal_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  x + y ≥ 4 ∧ (x + y = 4 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocal_one_l4162_416204


namespace NUMINAMATH_CALUDE_lindsey_bands_count_l4162_416219

/-- The number of exercise bands Lindsey bought -/
def num_bands : ℕ := 2

/-- The resistance added by each band in pounds -/
def resistance_per_band : ℕ := 5

/-- The weight of the dumbbell in pounds -/
def dumbbell_weight : ℕ := 10

/-- The total weight Lindsey squats in pounds -/
def total_squat_weight : ℕ := 30

theorem lindsey_bands_count :
  (2 * num_bands * resistance_per_band + dumbbell_weight = total_squat_weight) :=
by sorry

end NUMINAMATH_CALUDE_lindsey_bands_count_l4162_416219


namespace NUMINAMATH_CALUDE_watch_cost_price_l4162_416263

theorem watch_cost_price (loss_percentage : ℝ) (gain_percentage : ℝ) (price_difference : ℝ) : 
  loss_percentage = 0.15 →
  gain_percentage = 0.10 →
  price_difference = 450 →
  ∃ (cost_price : ℝ),
    cost_price * (1 - loss_percentage) + price_difference = cost_price * (1 + gain_percentage) ∧
    cost_price = 1800 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l4162_416263


namespace NUMINAMATH_CALUDE_x_range_theorem_l4162_416248

theorem x_range_theorem (x : ℝ) :
  (∀ a b c : ℝ, a^2 + b^2 + c^2 = 1 → a + b + Real.sqrt 2 * c ≤ |x^2 - 1|) →
  x ≤ -Real.sqrt 3 ∨ x ≥ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_x_range_theorem_l4162_416248
