import Mathlib

namespace NUMINAMATH_CALUDE_cube_volume_problem_l1365_136560

theorem cube_volume_problem (a : ℝ) (h : a > 0) :
  (a - 2) * a * (a + 2) = a^3 - 12 → a^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l1365_136560


namespace NUMINAMATH_CALUDE_cos_330_degrees_l1365_136532

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l1365_136532


namespace NUMINAMATH_CALUDE_equation_describes_ellipse_and_hyperbola_l1365_136572

/-- A conic section type -/
inductive ConicSection
  | Ellipse
  | Hyperbola
  | Parabola
  | Circle
  | Point
  | Line
  | CrossedLines

/-- Represents the equation y^4 - 8x^4 = 8y^2 - 4 -/
def equation (x y : ℝ) : Prop :=
  y^4 - 8*x^4 = 8*y^2 - 4

/-- The set of conic sections described by the equation -/
def describedConicSections : Set ConicSection :=
  {ConicSection.Ellipse, ConicSection.Hyperbola}

/-- Theorem stating that the equation describes the union of an ellipse and a hyperbola -/
theorem equation_describes_ellipse_and_hyperbola :
  ∀ x y : ℝ, equation x y → 
  ∃ (c : ConicSection), c ∈ describedConicSections :=
sorry

end NUMINAMATH_CALUDE_equation_describes_ellipse_and_hyperbola_l1365_136572


namespace NUMINAMATH_CALUDE_salary_increase_after_employee_reduction_l1365_136515

theorem salary_increase_after_employee_reduction (E : ℝ) (S : ℝ) (h1 : E > 0) (h2 : S > 0) :
  let new_E := 0.9 * E
  let new_S := (E * S) / new_E
  (new_S - S) / S = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_salary_increase_after_employee_reduction_l1365_136515


namespace NUMINAMATH_CALUDE_warm_production_time_l1365_136587

/-- Represents the production time for flower pots -/
structure PotProduction where
  cold_time : ℕ  -- Time to produce a pot when machine is cold (in minutes)
  warm_time : ℕ  -- Time to produce a pot when machine is warm (in minutes)
  hour_length : ℕ  -- Length of a production hour (in minutes)
  extra_pots : ℕ  -- Additional pots produced in the last hour compared to the first

/-- Theorem stating the warm production time given the conditions -/
theorem warm_production_time (p : PotProduction) 
  (h1 : p.cold_time = 6)
  (h2 : p.hour_length = 60)
  (h3 : p.extra_pots = 2)
  (h4 : p.hour_length / p.cold_time + p.extra_pots = p.hour_length / p.warm_time) :
  p.warm_time = 5 := by
  sorry

#check warm_production_time

end NUMINAMATH_CALUDE_warm_production_time_l1365_136587


namespace NUMINAMATH_CALUDE_max_value_and_inequality_l1365_136508

def f (x m : ℝ) : ℝ := |x - m| - |x + 2*m|

theorem max_value_and_inequality (m : ℝ) (hm : m > 0) 
  (hmax : ∀ x, f x m ≤ 3) :
  m = 1 ∧ 
  ∀ a b : ℝ, a * b > 0 → a^2 + b^2 = m^2 → a^3 / b + b^3 / a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_max_value_and_inequality_l1365_136508


namespace NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l1365_136592

theorem ten_thousandths_place_of_5_32 : ∃ (n : ℕ), (5 : ℚ) / 32 = (n * 10000 + 5) / 100000 :=
by sorry

end NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l1365_136592


namespace NUMINAMATH_CALUDE_problem_solution_l1365_136520

-- Define what it means for a number to be a factor of another
def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

-- Define what it means for a number to be a divisor of another
def is_divisor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem problem_solution :
  (is_factor 5 25) ∧
  (is_divisor 19 209 ∧ ¬ is_divisor 19 63) ∧
  (is_factor 9 180) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l1365_136520


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_condition_l1365_136585

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x + 1

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x, f a x > 0

-- Define the proposition q
def q (x : ℝ) : Prop := x^2 - 2*x - 8 > 0

theorem quadratic_inequality_and_condition :
  (∃ a, a ∈ Set.Icc 0 4 ∧ p a) ∧
  (∀ x, q x → x > 5) ∧
  (∃ x, q x ∧ x ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_condition_l1365_136585


namespace NUMINAMATH_CALUDE_tower_surface_area_is_1207_l1365_136534

def cube_volumes : List ℕ := [1, 27, 64, 125, 216, 343, 512, 729]

def cube_side_lengths : List ℕ := [1, 3, 4, 5, 6, 7, 8, 9]

def visible_faces : List ℕ := [6, 4, 4, 4, 4, 4, 4, 5]

def tower_surface_area (volumes : List ℕ) (side_lengths : List ℕ) (faces : List ℕ) : ℕ :=
  (List.zip (List.zip side_lengths faces) volumes).foldr
    (fun ((s, f), v) acc => acc + f * s * s)
    0

theorem tower_surface_area_is_1207 :
  tower_surface_area cube_volumes cube_side_lengths visible_faces = 1207 := by
  sorry

end NUMINAMATH_CALUDE_tower_surface_area_is_1207_l1365_136534


namespace NUMINAMATH_CALUDE_employed_males_percentage_l1365_136536

/-- Given a population where 60% are employed and 20% of the employed are females,
    prove that 48% of the population are employed males. -/
theorem employed_males_percentage
  (total_population : ℕ) 
  (employed_percentage : ℚ) 
  (employed_females_percentage : ℚ) 
  (h1 : employed_percentage = 60 / 100)
  (h2 : employed_females_percentage = 20 / 100) :
  (employed_percentage - employed_percentage * employed_females_percentage) * 100 = 48 := by
  sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l1365_136536


namespace NUMINAMATH_CALUDE_complement_A_inter_B_l1365_136591

def U : Set Int := {-3, -2, -1, 0, 1, 2, 3}
def A : Set Int := {-3, -2, 2, 3}
def B : Set Int := {-3, 0, 1, 2}

theorem complement_A_inter_B :
  (U \ A) ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_l1365_136591


namespace NUMINAMATH_CALUDE_sqrt_19_between_4_and_5_l1365_136505

theorem sqrt_19_between_4_and_5 : 4 < Real.sqrt 19 ∧ Real.sqrt 19 < 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_19_between_4_and_5_l1365_136505


namespace NUMINAMATH_CALUDE_factorial_properties_l1365_136576

-- Define ord_p
def ord_p (p : ℕ) (n : ℕ) : ℕ := sorry

-- Define S_p
def S_p (p : ℕ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem factorial_properties (n : ℕ) (p : ℕ) (m : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_div : p ^ m ∣ n ∧ ¬(p ^ (m + 1) ∣ n)) 
  (h_ord : ord_p p n = m) : 
  (ord_p p (n.factorial) = (n - S_p p n) / (p - 1)) ∧ 
  (∃ k : ℕ, (2 * n).factorial = k * n.factorial * (n + 1).factorial) ∧
  (Nat.Coprime m (n + 1) → 
    ∃ k : ℕ, (m * n + n).factorial = k * (m * n).factorial * (n + 1).factorial) := by
  sorry

end NUMINAMATH_CALUDE_factorial_properties_l1365_136576


namespace NUMINAMATH_CALUDE_interval_intersection_l1365_136530

theorem interval_intersection (x : ℝ) : 
  (2 < 3*x ∧ 3*x < 5 ∧ 2 < 4*x ∧ 4*x < 5) ↔ (2/3 < x ∧ x < 5/4) :=
by sorry

end NUMINAMATH_CALUDE_interval_intersection_l1365_136530


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l1365_136581

theorem triangle_angle_inequality (a : ℝ) : 
  (∃ (α β : ℝ), 0 < α ∧ 0 < β ∧ α + β < π ∧ 
    Real.cos (Real.sqrt α) + Real.cos (Real.sqrt β) > a + Real.cos (Real.sqrt (α * β))) 
  → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l1365_136581


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_constraint_l1365_136558

/-- Given vectors a and b, if the magnitude of their sum does not exceed 5,
    then the second component of b is in the range [-6, 2]. -/
theorem vector_sum_magnitude_constraint (a b : ℝ × ℝ) (h : ‖a + b‖ ≤ 5) :
  a = (-2, 2) → b.1 = 5 → -6 ≤ b.2 ∧ b.2 ≤ 2 := by
  sorry

#check vector_sum_magnitude_constraint

end NUMINAMATH_CALUDE_vector_sum_magnitude_constraint_l1365_136558


namespace NUMINAMATH_CALUDE_two_thousand_eight_times_two_thousand_six_l1365_136539

theorem two_thousand_eight_times_two_thousand_six (n : ℕ) :
  (2 * 2006 = 1) →
  (∀ n : ℕ, (2*n + 2) * 2006 = 3 * ((2*n) * 2006)) →
  2008 * 2006 = 3^1003 := by
sorry

end NUMINAMATH_CALUDE_two_thousand_eight_times_two_thousand_six_l1365_136539


namespace NUMINAMATH_CALUDE_min_set_size_l1365_136501

theorem min_set_size (n : ℕ) 
  (h1 : ∃ (s : Finset ℝ), s.card = 2*n + 1)
  (h2 : ∃ (s1 s2 : Finset ℝ), s1.card = n + 1 ∧ s2.card = n ∧ 
        (∀ x ∈ s1, x ≥ 10) ∧ (∀ x ∈ s2, x ≥ 1))
  (h3 : ∃ (s : Finset ℝ), s.card = 2*n + 1 ∧ 
        (Finset.sum s id) / (2*n + 1 : ℝ) = 6) :
  n ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_set_size_l1365_136501


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1365_136528

theorem trigonometric_simplification (α : ℝ) :
  2 * Real.sin (2 * α) ^ 2 + Real.sqrt 3 * Real.sin (4 * α) -
  (4 * Real.tan (2 * α) * (1 - Real.tan (2 * α) ^ 2)) /
  (Real.sin (8 * α) * (1 + Real.tan (2 * α) ^ 2) ^ 2) =
  2 * Real.sin (4 * α - π / 6) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1365_136528


namespace NUMINAMATH_CALUDE_rectangleChoicesIs54_l1365_136543

/-- The number of ways to choose 4 lines (2 horizontal and 2 vertical) from 5 horizontal
    and 4 vertical lines to form a rectangle, without selecting the first and fifth
    horizontal lines together. -/
def rectangleChoices : ℕ := by
  -- Define the number of horizontal and vertical lines
  let horizontalLines : ℕ := 5
  let verticalLines : ℕ := 4

  -- Calculate the total number of ways to choose 2 horizontal lines
  let totalHorizontalChoices : ℕ := Nat.choose horizontalLines 2

  -- Calculate the number of choices that include both first and fifth horizontal lines
  let invalidHorizontalChoices : ℕ := 1

  -- Calculate the number of valid horizontal line choices
  let validHorizontalChoices : ℕ := totalHorizontalChoices - invalidHorizontalChoices

  -- Calculate the number of ways to choose 2 vertical lines
  let verticalChoices : ℕ := Nat.choose verticalLines 2

  -- Calculate the total number of valid choices
  exact validHorizontalChoices * verticalChoices

/-- Theorem stating that the number of valid rectangle choices is 54 -/
theorem rectangleChoicesIs54 : rectangleChoices = 54 := by sorry

end NUMINAMATH_CALUDE_rectangleChoicesIs54_l1365_136543


namespace NUMINAMATH_CALUDE_karen_is_ten_l1365_136506

def sisters : Finset ℕ := {2, 4, 6, 8, 10, 12, 14}

def park_pair (a b : ℕ) : Prop :=
  a ∈ sisters ∧ b ∈ sisters ∧ a ≠ b ∧ a + b = 20

def pool_pair (a b : ℕ) : Prop :=
  a ∈ sisters ∧ b ∈ sisters ∧ a ≠ b ∧ 3 < a ∧ a < 9 ∧ 3 < b ∧ b < 9

def karen_age (k : ℕ) : Prop :=
  k ∈ sisters ∧ k ≠ 4 ∧
  ∃ (p1 p2 s1 s2 : ℕ),
    park_pair p1 p2 ∧
    pool_pair s1 s2 ∧
    p1 ≠ s1 ∧ p1 ≠ s2 ∧ p2 ≠ s1 ∧ p2 ≠ s2 ∧
    k ≠ p1 ∧ k ≠ p2 ∧ k ≠ s1 ∧ k ≠ s2

theorem karen_is_ten : ∃! k, karen_age k ∧ k = 10 := by
  sorry

end NUMINAMATH_CALUDE_karen_is_ten_l1365_136506


namespace NUMINAMATH_CALUDE_basketball_game_theorem_l1365_136516

/-- Represents the scores of a team in a basketball game -/
structure TeamScores :=
  (q1 : ℝ)
  (q2 : ℝ)
  (q3 : ℝ)
  (q4 : ℝ)

/-- The game result -/
def GameResult := TeamScores → TeamScores → Prop

/-- Checks if the scores form a decreasing geometric sequence -/
def is_decreasing_geometric (s : TeamScores) : Prop :=
  ∃ r : ℝ, r > 1 ∧ s.q2 = s.q1 / r ∧ s.q3 = s.q2 / r ∧ s.q4 = s.q3 / r

/-- Checks if the scores form a decreasing arithmetic sequence -/
def is_decreasing_arithmetic (s : TeamScores) : Prop :=
  ∃ d : ℝ, d > 0 ∧ s.q2 = s.q1 - d ∧ s.q3 = s.q2 - d ∧ s.q4 = s.q3 - d

/-- Calculates the total score of a team -/
def total_score (s : TeamScores) : ℝ := s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the score in the second half -/
def second_half_score (s : TeamScores) : ℝ := s.q3 + s.q4

/-- The main theorem to prove -/
theorem basketball_game_theorem (falcons eagles : TeamScores) : 
  (is_decreasing_geometric falcons) →
  (is_decreasing_arithmetic eagles) →
  (falcons.q1 + falcons.q2 = eagles.q1 + eagles.q2) →
  (total_score eagles = total_score falcons + 2) →
  (total_score falcons ≤ 100 ∧ total_score eagles ≤ 100) →
  (second_half_score falcons + second_half_score eagles = 27) := by
  sorry


end NUMINAMATH_CALUDE_basketball_game_theorem_l1365_136516


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l1365_136595

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 21 ∧ 
  correct_value = 48 ∧ 
  corrected_mean = 36.54 →
  ∃ initial_mean : ℝ, 
    initial_mean * n + (correct_value - wrong_value) = corrected_mean * n ∧ 
    initial_mean = 36 :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l1365_136595


namespace NUMINAMATH_CALUDE_minimum_bottles_l1365_136580

def bottle_capacity : ℕ := 15
def minimum_volume : ℕ := 150

theorem minimum_bottles : 
  ∀ n : ℕ, (n * bottle_capacity ≥ minimum_volume ∧ 
  ∀ m : ℕ, m < n → m * bottle_capacity < minimum_volume) → n = 10 :=
by sorry

end NUMINAMATH_CALUDE_minimum_bottles_l1365_136580


namespace NUMINAMATH_CALUDE_second_smallest_divisible_by_all_less_than_8_sum_of_digits_l1365_136531

def is_divisible_by_all_less_than_8 (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k ∧ k < 8 → k ∣ n

def second_smallest (P : ℕ → Prop) (n : ℕ) : Prop :=
  P n ∧ ∃ m : ℕ, P m ∧ m < n ∧ ∀ k : ℕ, P k → k = m ∨ n ≤ k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem second_smallest_divisible_by_all_less_than_8_sum_of_digits :
  ∃ M : ℕ, second_smallest is_divisible_by_all_less_than_8 M ∧ sum_of_digits M = 12 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_divisible_by_all_less_than_8_sum_of_digits_l1365_136531


namespace NUMINAMATH_CALUDE_x_plus_ten_equals_forty_l1365_136575

theorem x_plus_ten_equals_forty (x y : ℝ) (h1 : x / y = 6 / 3) (h2 : y = 15) : x + 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_ten_equals_forty_l1365_136575


namespace NUMINAMATH_CALUDE_division_problem_l1365_136538

theorem division_problem (n : ℕ) : 
  n / 20 = 10 ∧ n % 20 = 10 → n = 210 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1365_136538


namespace NUMINAMATH_CALUDE_manolo_face_masks_l1365_136500

/-- Calculates the number of face-masks Manolo makes in a four-hour shift -/
def face_masks_in_shift (first_hour_rate : ℕ) (other_hours_rate : ℕ) (shift_duration : ℕ) : ℕ :=
  let first_hour_masks := 60 / first_hour_rate
  let other_hours_masks := (shift_duration - 1) * 60 / other_hours_rate
  first_hour_masks + other_hours_masks

/-- Theorem stating that Manolo makes 45 face-masks in a four-hour shift -/
theorem manolo_face_masks :
  face_masks_in_shift 4 6 4 = 45 :=
by sorry

end NUMINAMATH_CALUDE_manolo_face_masks_l1365_136500


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l1365_136589

theorem complex_sum_theorem :
  let A : ℂ := 3 + 2*I
  let B : ℂ := -3 + I
  let C : ℂ := 1 - 2*I
  let D : ℂ := 4 + 3*I
  A + B + C + D = 5 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l1365_136589


namespace NUMINAMATH_CALUDE_original_model_cost_l1365_136533

/-- The original cost of a model before the price increase -/
def original_cost : ℝ := sorry

/-- The amount Kirsty saved -/
def saved_amount : ℝ := 30 * original_cost

/-- The new cost of a model after the price increase -/
def new_cost : ℝ := original_cost + 0.50

theorem original_model_cost :
  (saved_amount = 27 * new_cost) → original_cost = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_original_model_cost_l1365_136533


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1365_136570

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_seq (x y z : ℝ) : Prop :=
  y / x = z / y ∧ y ≠ 0

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_seq a →
  geometric_seq (a 1) (a 3) (a 4) →
  a 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1365_136570


namespace NUMINAMATH_CALUDE_second_point_y_coordinate_l1365_136565

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x = 2 * y + 5

-- Define the two points
def point1 (m n : ℝ) : ℝ × ℝ := (m, n)
def point2 (m n k : ℝ) : ℝ × ℝ := (m + 1, n + k)

-- Theorem statement
theorem second_point_y_coordinate 
  (m n : ℝ) 
  (h1 : line_equation m n) 
  (h2 : line_equation (m + 1) (n + 0.5)) : 
  (point2 m n 0.5).2 = n + 0.5 := by
  sorry

end NUMINAMATH_CALUDE_second_point_y_coordinate_l1365_136565


namespace NUMINAMATH_CALUDE_wrong_number_correction_l1365_136535

theorem wrong_number_correction (n : ℕ) (initial_avg correct_avg : ℚ) 
  (first_error second_correct : ℤ) : 
  n = 10 → 
  initial_avg = 40.2 → 
  correct_avg = 40.3 → 
  first_error = 19 → 
  second_correct = 31 → 
  ∃ (second_error : ℤ), 
    (n : ℚ) * initial_avg - first_error - second_error + second_correct = (n : ℚ) * correct_avg ∧ 
    second_error = 11 := by
  sorry

end NUMINAMATH_CALUDE_wrong_number_correction_l1365_136535


namespace NUMINAMATH_CALUDE_correlation_coefficient_comparison_l1365_136573

def X : List ℝ := [16, 18, 20, 22]
def Y : List ℝ := [15.10, 12.81, 9.72, 3.21]
def U : List ℝ := [10, 20, 30]
def V : List ℝ := [7.5, 9.5, 16.6]

def r₁ : ℝ := sorry
def r₂ : ℝ := sorry

theorem correlation_coefficient_comparison : r₁ < 0 ∧ 0 < r₂ := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_comparison_l1365_136573


namespace NUMINAMATH_CALUDE_initial_cost_calculation_l1365_136548

/-- Represents the car rental cost structure and usage --/
structure CarRental where
  initialCost : ℝ
  costPerMile : ℝ
  milesDriven : ℝ
  totalCost : ℝ

/-- Theorem stating the initial cost of the car rental --/
theorem initial_cost_calculation (rental : CarRental) 
    (h1 : rental.costPerMile = 0.50)
    (h2 : rental.milesDriven = 1364)
    (h3 : rental.totalCost = 832) :
    rental.initialCost = 150 := by
  sorry


end NUMINAMATH_CALUDE_initial_cost_calculation_l1365_136548


namespace NUMINAMATH_CALUDE_expression_simplification_l1365_136590

theorem expression_simplification (m n : ℚ) (hm : m = 1) (hn : n = -3) :
  2/3 * (6*m - 9*m*n) - (n^2 - 6*m*n) = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1365_136590


namespace NUMINAMATH_CALUDE_runners_speed_ratio_l1365_136517

theorem runners_speed_ratio :
  ∀ (C : ℝ) (v_V v_P : ℝ),
  C > 0 → v_V > 0 → v_P > 0 →
  (∃ (t_1 : ℝ), t_1 > 0 ∧ v_V * t_1 + v_P * t_1 = C) →
  (∃ (t_2 : ℝ), t_2 > 0 ∧ v_V * t_2 = C + v_V * (C / (v_V + v_P)) ∧ v_P * t_2 = C + v_P * (C / (v_V + v_P))) →
  v_V / v_P = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_runners_speed_ratio_l1365_136517


namespace NUMINAMATH_CALUDE_sweater_markup_l1365_136502

theorem sweater_markup (wholesale_cost : ℝ) (retail_price : ℝ) :
  retail_price > 0 →
  wholesale_cost > 0 →
  (retail_price * 0.4 = wholesale_cost * 1.35) →
  ((retail_price - wholesale_cost) / wholesale_cost) * 100 = 237.5 := by
sorry

end NUMINAMATH_CALUDE_sweater_markup_l1365_136502


namespace NUMINAMATH_CALUDE_tenth_vertex_label_l1365_136512

/-- Regular 2012-gon with vertices labeled according to specific conditions -/
structure Polygon2012 where
  /-- The labeling function for vertices -/
  label : Fin 2012 → Fin 2012
  /-- The first vertex is labeled A₁ -/
  first_vertex : label 0 = 0
  /-- The second vertex is labeled A₄ -/
  second_vertex : label 1 = 3
  /-- If k+ℓ and m+n have the same remainder mod 2012, then AₖAₗ and AₘAₙ don't intersect -/
  non_intersecting_chords : ∀ k ℓ m n : Fin 2012, 
    (k + ℓ) % 2012 = (m + n) % 2012 → 
    (label k + label ℓ) % 2012 ≠ (label m + label n) % 2012

/-- The label of the tenth vertex in a Polygon2012 is A₂₈ -/
theorem tenth_vertex_label (p : Polygon2012) : p.label 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_tenth_vertex_label_l1365_136512


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1365_136552

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic_sequence a →
  a 0 = 3 →
  a 1 = 9 →
  a 2 = 15 →
  (∃ n : ℕ, a (n + 2) = 33 ∧ a (n + 1) = y ∧ a n = x) →
  x + y = 48 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1365_136552


namespace NUMINAMATH_CALUDE_cannot_compare_full_mark_students_l1365_136504

/-- Represents a school with a total number of students and full-mark scorers -/
structure School where
  total_students : ℕ
  full_mark_students : ℕ
  h_full_mark_valid : full_mark_students ≤ total_students

/-- The percentage of full-mark scorers in a school -/
def full_mark_percentage (s : School) : ℚ :=
  (s.full_mark_students : ℚ) / (s.total_students : ℚ) * 100

theorem cannot_compare_full_mark_students
  (school_A school_B : School)
  (h_A : full_mark_percentage school_A = 1)
  (h_B : full_mark_percentage school_B = 2) :
  ¬ (∀ (s₁ s₂ : School),
    full_mark_percentage s₁ = 1 →
    full_mark_percentage s₂ = 2 →
    (s₁.full_mark_students < s₂.full_mark_students ∨
     s₁.full_mark_students > s₂.full_mark_students ∨
     s₁.full_mark_students = s₂.full_mark_students)) :=
by
  sorry

end NUMINAMATH_CALUDE_cannot_compare_full_mark_students_l1365_136504


namespace NUMINAMATH_CALUDE_last_three_digits_of_2_to_15000_l1365_136519

theorem last_three_digits_of_2_to_15000 (h : 2^500 ≡ 1 [ZMOD 1250]) :
  2^15000 ≡ 1 [ZMOD 1000] := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_2_to_15000_l1365_136519


namespace NUMINAMATH_CALUDE_sandwich_jam_cost_l1365_136584

theorem sandwich_jam_cost (N B J : ℕ) : 
  N > 1 → 
  B > 0 → 
  J > 0 → 
  N * (4 * B + 6 * J) = 351 → 
  N * J * 6 = 162 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_jam_cost_l1365_136584


namespace NUMINAMATH_CALUDE_prob_two_queens_or_at_least_one_king_is_34_221_l1365_136509

/-- Represents a standard deck of cards -/
structure StandardDeck :=
  (total_cards : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)
  (h_total : total_cards = 52)
  (h_kings : num_kings = 4)
  (h_queens : num_queens = 4)

/-- Calculates the probability of drawing either two queens or at least 1 king -/
def prob_two_queens_or_at_least_one_king (deck : StandardDeck) : ℚ :=
  34 / 221

/-- Theorem stating that the probability of drawing either two queens or at least 1 king is 34/221 -/
theorem prob_two_queens_or_at_least_one_king_is_34_221 (deck : StandardDeck) :
  prob_two_queens_or_at_least_one_king deck = 34 / 221 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_queens_or_at_least_one_king_is_34_221_l1365_136509


namespace NUMINAMATH_CALUDE_eight_digit_number_divisibility_l1365_136596

/-- Represents an eight-digit number in the form 757AB384 -/
def EightDigitNumber (A B : ℕ) : ℕ := 757000000 + A * 10000 + B * 1000 + 384

/-- The number is divisible by 357 -/
def IsDivisibleBy357 (n : ℕ) : Prop := ∃ k : ℕ, n = 357 * k

theorem eight_digit_number_divisibility :
  ∀ A : ℕ, (A < 10) →
    (IsDivisibleBy357 (EightDigitNumber A 5) ∧
     ∀ B : ℕ, B < 10 → B ≠ 5 → ¬IsDivisibleBy357 (EightDigitNumber A B)) :=
by sorry

end NUMINAMATH_CALUDE_eight_digit_number_divisibility_l1365_136596


namespace NUMINAMATH_CALUDE_ellipse_product_l1365_136549

/-- Represents an ellipse with center O, major axis AB, minor axis CD, and focus F. -/
structure Ellipse where
  a : ℝ  -- Length of semi-major axis
  b : ℝ  -- Length of semi-minor axis
  f : ℝ  -- Distance from center to focus

/-- Conditions for the ellipse problem -/
def EllipseProblem (e : Ellipse) : Prop :=
  e.f = 6 ∧ e.a - e.b = 4 ∧ e.a^2 - e.b^2 = e.f^2

theorem ellipse_product (e : Ellipse) (h : EllipseProblem e) : (2 * e.a) * (2 * e.b) = 65 := by
  sorry

#check ellipse_product

end NUMINAMATH_CALUDE_ellipse_product_l1365_136549


namespace NUMINAMATH_CALUDE_celsius_to_fahrenheit_l1365_136541

theorem celsius_to_fahrenheit (C F : ℝ) : 
  C = (4/7) * (F - 40) → C = 35 → F = 101.25 := by
sorry

end NUMINAMATH_CALUDE_celsius_to_fahrenheit_l1365_136541


namespace NUMINAMATH_CALUDE_original_number_is_sixty_l1365_136507

theorem original_number_is_sixty : 
  ∀ x : ℝ, (0.5 * x = 30) → x = 60 := by
sorry

end NUMINAMATH_CALUDE_original_number_is_sixty_l1365_136507


namespace NUMINAMATH_CALUDE_roots_expression_value_l1365_136562

theorem roots_expression_value (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 3 * x₁ + 1 = 0) → 
  (2 * x₂^2 - 3 * x₂ + 1 = 0) → 
  ((x₁ + x₂) / (1 + x₁ * x₂) = 1) :=
by sorry

end NUMINAMATH_CALUDE_roots_expression_value_l1365_136562


namespace NUMINAMATH_CALUDE_maria_cookies_left_maria_cookies_problem_l1365_136554

theorem maria_cookies_left (initial_cookies : ℕ) (friend_cookies : ℕ) (eat_cookies : ℕ) : ℕ :=
  let remaining_after_friend := initial_cookies - friend_cookies
  let family_cookies := remaining_after_friend / 2
  let remaining_after_family := remaining_after_friend - family_cookies
  let final_cookies := remaining_after_family - eat_cookies
  final_cookies

theorem maria_cookies_problem :
  maria_cookies_left 19 5 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_maria_cookies_left_maria_cookies_problem_l1365_136554


namespace NUMINAMATH_CALUDE_kho_kho_players_l1365_136599

theorem kho_kho_players (total : ℕ) (kabadi : ℕ) (both : ℕ) (kho_kho : ℕ) : 
  total = 25 → kabadi = 10 → both = 5 → kho_kho = total - kabadi + both := by
  sorry

end NUMINAMATH_CALUDE_kho_kho_players_l1365_136599


namespace NUMINAMATH_CALUDE_max_value_theorem_l1365_136544

theorem max_value_theorem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 10) :
  ∃ (M : ℝ), M = 100 ∧ ∀ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 = 10 → x^4 + y^2 + z^2 + w^2 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1365_136544


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1365_136529

theorem quadratic_factorization (h b c d : ℤ) : 
  (∀ x : ℝ, 6 * x^2 + x - 12 = (h * x + b) * (c * x + d)) → 
  |h| + |b| + |c| + |d| = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1365_136529


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_min_value_achievable_l1365_136583

theorem min_value_of_sum_of_roots (x : ℝ) :
  Real.sqrt (x^2 + (x - 2)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem min_value_achievable :
  ∃ x : ℝ, Real.sqrt (x^2 + (x - 2)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_min_value_achievable_l1365_136583


namespace NUMINAMATH_CALUDE_least_candies_to_remove_l1365_136540

theorem least_candies_to_remove (total : Nat) (sisters : Nat) (to_remove : Nat) : 
  total = 24 → 
  sisters = 5 → 
  (total - to_remove) % sisters = 0 → 
  ∀ x : Nat, x < to_remove → (total - x) % sisters ≠ 0 →
  to_remove = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_candies_to_remove_l1365_136540


namespace NUMINAMATH_CALUDE_smallest_rectangle_cover_l1365_136521

/-- A point in the unit square -/
def Point := Fin 2 → Real

/-- The unit square -/
def UnitSquare : Set Point :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 1}

/-- The interior of the unit square -/
def InteriorUnitSquare : Set Point :=
  {p | ∀ i, 0 < p i ∧ p i < 1}

/-- A rectangle with sides parallel to the unit square -/
structure Rectangle where
  x1 : Real
  x2 : Real
  y1 : Real
  y2 : Real
  h1 : x1 < x2
  h2 : y1 < y2

/-- A point is in the interior of a rectangle -/
def isInterior (p : Point) (r : Rectangle) : Prop :=
  r.x1 < p 0 ∧ p 0 < r.x2 ∧ r.y1 < p 1 ∧ p 1 < r.y2

/-- The theorem statement -/
theorem smallest_rectangle_cover (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, k = 2 * n + 2 ∧
  (∀ S : Set Point, Finite S → S.ncard = n → S ⊆ InteriorUnitSquare →
    ∃ R : Set Rectangle, R.ncard = k ∧
      (∀ p ∈ S, ∀ r ∈ R, ¬isInterior p r) ∧
      (∀ p ∈ InteriorUnitSquare \ S, ∃ r ∈ R, isInterior p r)) ∧
  (∀ k' : ℕ, k' < k →
    ∃ S : Set Point, Finite S ∧ S.ncard = n ∧ S ⊆ InteriorUnitSquare ∧
      ∀ R : Set Rectangle, R.ncard = k' →
        (∃ p ∈ S, ∃ r ∈ R, isInterior p r) ∨
        (∃ p ∈ InteriorUnitSquare \ S, ∀ r ∈ R, ¬isInterior p r)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_rectangle_cover_l1365_136521


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l1365_136537

theorem shaded_region_perimeter (r : ℝ) : 
  r > 0 →
  2 * Real.pi * r = 24 →
  (3 : ℝ) * (1 / 6 : ℝ) * (2 * Real.pi * r) = 12 :=
by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l1365_136537


namespace NUMINAMATH_CALUDE_paul_frosting_needs_l1365_136569

/-- Represents the amount of frosting needed for different baked goods -/
structure FrostingNeeds where
  layer_cake : ℚ
  single_cake : ℚ
  brownies : ℚ
  cupcakes : ℚ

/-- Calculates the total cans of frosting needed -/
def total_frosting_needed (f : FrostingNeeds) (layer_cakes single_cakes brownies cupcake_dozens : ℕ) : ℚ :=
  f.layer_cake * layer_cakes + 
  f.single_cake * single_cakes + 
  f.brownies * brownies + 
  f.cupcakes * cupcake_dozens

/-- Theorem stating that Paul needs 21 cans of frosting -/
theorem paul_frosting_needs : 
  let f : FrostingNeeds := { 
    layer_cake := 1,
    single_cake := 1/2,
    brownies := 1/2,
    cupcakes := 1/2
  }
  total_frosting_needed f 3 12 18 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_paul_frosting_needs_l1365_136569


namespace NUMINAMATH_CALUDE_cubic_equation_root_l1365_136578

theorem cubic_equation_root (a b : ℚ) :
  ((-2 : ℝ) - 5 * Real.sqrt 3) ^ 3 + a * ((-2 : ℝ) - 5 * Real.sqrt 3) ^ 2 + 
  b * ((-2 : ℝ) - 5 * Real.sqrt 3) + 49 = 0 →
  a = 235 / 71 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l1365_136578


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l1365_136598

theorem polynomial_expansion_equality (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₀ + a₂ + a₄ + a₆)^2 - (a₁ + a₃ + a₅)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l1365_136598


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1365_136547

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → (7 ^ m.val) % 3 ≠ (m.val ^ 4) % 3) ∧ 
  (7 ^ n.val) % 3 = (n.val ^ 4) % 3 ∧
  n = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1365_136547


namespace NUMINAMATH_CALUDE_arrangement_count_correct_l1365_136542

/-- The number of ways to arrange 7 people in a row with 2 people between A and B -/
def arrangement_count : ℕ := 960

/-- The total number of people -/
def total_people : ℕ := 7

/-- The number of people between A and B -/
def people_between : ℕ := 2

/-- Theorem stating that the arrangement count is correct -/
theorem arrangement_count_correct : 
  arrangement_count = 
    (Nat.factorial 2) *  -- Ways to arrange A and B
    (Nat.choose (total_people - 2) people_between) *  -- Ways to choose people between A and B
    (Nat.factorial people_between) *  -- Ways to arrange people between A and B
    (Nat.factorial (total_people - people_between - 2))  -- Ways to arrange remaining people
  := by sorry

end NUMINAMATH_CALUDE_arrangement_count_correct_l1365_136542


namespace NUMINAMATH_CALUDE_apple_arrangements_l1365_136564

def word : String := "apple"

/-- The number of distinct letters in the word -/
def distinctLetters : Nat := 4

/-- The total number of letters in the word -/
def totalLetters : Nat := 5

/-- The frequency of the letter 'p' in the word -/
def frequencyP : Nat := 2

/-- The frequency of the letter 'a' in the word -/
def frequencyA : Nat := 1

/-- The frequency of the letter 'l' in the word -/
def frequencyL : Nat := 1

/-- The frequency of the letter 'e' in the word -/
def frequencyE : Nat := 1

/-- The number of distinct arrangements of the letters in the word -/
def distinctArrangements : Nat := 60

theorem apple_arrangements :
  distinctArrangements = Nat.factorial totalLetters / 
    (Nat.factorial frequencyP * Nat.factorial frequencyA * 
     Nat.factorial frequencyL * Nat.factorial frequencyE) := by
  sorry

end NUMINAMATH_CALUDE_apple_arrangements_l1365_136564


namespace NUMINAMATH_CALUDE_set_inclusion_implies_m_values_l1365_136524

def A (m : ℝ) : Set ℝ := {1, 3, 2*m+3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem set_inclusion_implies_m_values (m : ℝ) : B m ⊆ A m → m = 1 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_m_values_l1365_136524


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1365_136518

/-- If the quadratic equation mx² + 2x + 1 = 0 has two equal real roots, then m = 1 -/
theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2*x + 1 = 0 ∧ 
   (∀ y : ℝ, m * y^2 + 2*y + 1 = 0 → y = x)) → 
  m = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1365_136518


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1365_136561

/-- An isosceles triangle with congruent sides of length 7 cm and perimeter 23 cm has a base of length 9 cm. -/
theorem isosceles_triangle_base_length : 
  ∀ (base congruent_side perimeter : ℝ),
  congruent_side = 7 →
  perimeter = 23 →
  perimeter = 2 * congruent_side + base →
  base = 9 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1365_136561


namespace NUMINAMATH_CALUDE_jones_elementary_population_l1365_136556

/-- The total number of students at Jones Elementary School -/
def total_students : ℕ := 150

/-- The number of boys at Jones Elementary School -/
def num_boys : ℕ := (60 * total_students) / 100

/-- Theorem stating that 90 students represent some percentage of the boys,
    and boys make up 60% of the total school population of 150 students -/
theorem jones_elementary_population :
  90 * total_students = 60 * num_boys :=
sorry

end NUMINAMATH_CALUDE_jones_elementary_population_l1365_136556


namespace NUMINAMATH_CALUDE_xy_square_plus_2xy_plus_1_l1365_136593

theorem xy_square_plus_2xy_plus_1 (x y : ℝ) 
  (h : x^2 - 2*x + y^2 - 6*y + 10 = 0) : 
  x^2 * y^2 + 2*x*y + 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_xy_square_plus_2xy_plus_1_l1365_136593


namespace NUMINAMATH_CALUDE_no_solution_to_inequalities_l1365_136525

theorem no_solution_to_inequalities : ¬∃ x : ℝ, (6*x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequalities_l1365_136525


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1365_136586

/-- The remainder when x^4 + 2x^3 is divided by x^2 + 3x + 2 is x^2 + 2x -/
theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 + 2*x^3 = (x^2 + 3*x + 2) * q + (x^2 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1365_136586


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l1365_136571

def z₁ : ℂ := -3 + Complex.I
def z₂ : ℂ := 1 - Complex.I
def z : ℂ := z₁ - z₂

theorem z_in_second_quadrant : 
  z.re < 0 ∧ z.im > 0 := by sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l1365_136571


namespace NUMINAMATH_CALUDE_range_of_m_range_of_x_l1365_136550

-- Part 1
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - 2 * m * x - 1 < 0) → m ∈ Set.Ioc (-1) 0 :=
sorry

-- Part 2
theorem range_of_x (x : ℝ) :
  (∀ m : ℝ, |m| ≤ 1 → m * x^2 - 2 * m * x - 1 < 0) → 
  x ∈ Set.Ioo (1 - Real.sqrt 2) 1 ∪ Set.Ioo 1 (1 + Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_x_l1365_136550


namespace NUMINAMATH_CALUDE_point_locations_l1365_136594

def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_locations (x y : ℝ) (h : |3*x + 2| + |2*y - 1| = 0) :
  is_in_second_quadrant x y ∧ is_in_fourth_quadrant (x + 1) (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_point_locations_l1365_136594


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1365_136522

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (4 * a^3 + 2023 * a + 4012 = 0) ∧ 
  (4 * b^3 + 2023 * b + 4012 = 0) ∧ 
  (4 * c^3 + 2023 * c + 4012 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 3009 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1365_136522


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1365_136527

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 28 x = Nat.choose 28 (2*x - 1)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1365_136527


namespace NUMINAMATH_CALUDE_quadratic_inequality_minimum_l1365_136546

theorem quadratic_inequality_minimum (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → x^2 + 2*x + a ≥ 0) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_minimum_l1365_136546


namespace NUMINAMATH_CALUDE_church_member_percentage_l1365_136551

theorem church_member_percentage (total_members : ℕ) (adult_members : ℕ) (child_members : ℕ) : 
  total_members = 120 →
  child_members = adult_members + 24 →
  total_members = adult_members + child_members →
  (adult_members : ℚ) / (total_members : ℚ) = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_church_member_percentage_l1365_136551


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1365_136545

theorem right_triangle_hypotenuse (PQ PR : ℝ) (h1 : PQ = 15) (h2 : PR = 20) :
  Real.sqrt (PQ^2 + PR^2) = 25 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1365_136545


namespace NUMINAMATH_CALUDE_divides_implies_equal_l1365_136553

theorem divides_implies_equal (a b : ℕ+) : 
  (4 * a * b - 1) ∣ ((4 * a^2 - 1)^2) → a = b := by
  sorry

end NUMINAMATH_CALUDE_divides_implies_equal_l1365_136553


namespace NUMINAMATH_CALUDE_gcd_equality_exists_l1365_136582

theorem gcd_equality_exists : ∃ k : ℕ+, 
  Nat.gcd 2012 2020 = Nat.gcd (2012 + k) 2020 ∧
  Nat.gcd 2012 2020 = Nat.gcd 2012 (2020 + k) ∧
  Nat.gcd 2012 2020 = Nat.gcd (2012 + k) (2020 + k) := by
  sorry

end NUMINAMATH_CALUDE_gcd_equality_exists_l1365_136582


namespace NUMINAMATH_CALUDE_correct_quantities_correct_min_discount_l1365_136567

-- Define the problem parameters
def total_cost : ℝ := 25000
def total_profit : ℝ := 11700
def ornament_cost : ℝ := 40
def ornament_price : ℝ := 58
def pendant_cost : ℝ := 30
def pendant_price : ℝ := 45
def second_profit_goal : ℝ := 10800

-- Define the quantities of ornaments and pendants
def ornaments : ℕ := 400
def pendants : ℕ := 300

-- Define the theorem for part 1
theorem correct_quantities :
  ornament_cost * ornaments + pendant_cost * pendants = total_cost ∧
  (ornament_price - ornament_cost) * ornaments + (pendant_price - pendant_cost) * pendants = total_profit :=
sorry

-- Define the minimum discount percentage
def min_discount_percentage : ℝ := 20

-- Define the theorem for part 2
theorem correct_min_discount :
  let new_pendant_price := pendant_price * (1 - min_discount_percentage / 100)
  (ornament_price - ornament_cost) * ornaments + (new_pendant_price - pendant_cost) * (2 * pendants) ≥ second_profit_goal ∧
  ∀ d : ℝ, d < min_discount_percentage →
    let price := pendant_price * (1 - d / 100)
    (ornament_price - ornament_cost) * ornaments + (price - pendant_cost) * (2 * pendants) < second_profit_goal :=
sorry

end NUMINAMATH_CALUDE_correct_quantities_correct_min_discount_l1365_136567


namespace NUMINAMATH_CALUDE_pencils_given_l1365_136588

theorem pencils_given (initial : ℕ) (total : ℕ) (given : ℕ) : 
  initial = 9 → total = 65 → given = total - initial :=
by
  sorry

end NUMINAMATH_CALUDE_pencils_given_l1365_136588


namespace NUMINAMATH_CALUDE_cube_sum_problem_l1365_136514

theorem cube_sum_problem (a b c : ℝ) 
  (sum_eq : a + b + c = 5)
  (sum_prod_eq : b * c + c * a + a * b = 7)
  (prod_eq : a * b * c = 2) :
  a^3 + b^3 + c^3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l1365_136514


namespace NUMINAMATH_CALUDE_some_number_value_l1365_136523

theorem some_number_value (X : ℝ) :
  2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * X)) = 1600.0000000000002 →
  X = 1.25 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l1365_136523


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1365_136568

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x ≤ 0) ↔ (∀ x : ℝ, f x > 0) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1365_136568


namespace NUMINAMATH_CALUDE_negative_square_power_equality_l1365_136513

theorem negative_square_power_equality (a b : ℝ) : (-a^2 * b^3)^2 = a^4 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_power_equality_l1365_136513


namespace NUMINAMATH_CALUDE_white_balls_count_l1365_136526

/-- Proves that in a bag with 3 red balls and x white balls, 
    if the probability of drawing a red ball is 1/4, then x = 9. -/
theorem white_balls_count (x : ℕ) : 
  (3 : ℚ) / (3 + x) = 1 / 4 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l1365_136526


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1365_136566

theorem quadratic_root_difference (C : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₁ - x₂ = 5.5 ∧ 2 * x₁^2 + 5 * x₁ = C ∧ 2 * x₂^2 + 5 * x₂ = C) → 
  C = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1365_136566


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l1365_136511

theorem quadratic_roots_theorem (r₁ r₂ : ℝ) (p q : ℝ) : 
  (r₁^2 - 5*r₁ + 6 = 0) →
  (r₂^2 - 5*r₂ + 6 = 0) →
  (r₁^2 + p*r₁^2 + q = 0) →
  (r₂^2 + p*r₂^2 + q = 0) →
  p = -13 ∧ q = 36 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l1365_136511


namespace NUMINAMATH_CALUDE_xiaoming_characters_proof_l1365_136579

theorem xiaoming_characters_proof : 
  ∀ (N : ℕ),
  (N / 2 - 50 : ℕ) + -- Day 1
  ((N / 2 + 50) / 2 - 20 : ℕ) + -- Day 2
  (((N / 4 + 45 : ℕ) / 2 + 10) : ℕ) + -- Day 3
  60 + -- Day 4
  40 = N → -- Remaining characters
  N = 700 := by
sorry

end NUMINAMATH_CALUDE_xiaoming_characters_proof_l1365_136579


namespace NUMINAMATH_CALUDE_john_metal_purchase_cost_l1365_136597

/-- Calculates the total cost of John's metal purchases in USD -/
def total_cost (silver_oz gold_oz platinum_oz palladium_oz : ℝ)
               (silver_price_usd gold_price_multiplier : ℝ)
               (platinum_price_gbp palladium_price_eur : ℝ)
               (usd_gbp_rate1 usd_gbp_rate2 : ℝ)
               (usd_eur_rate1 usd_eur_rate2 : ℝ)
               (silver_gold_discount platinum_tax : ℝ) : ℝ :=
  sorry

theorem john_metal_purchase_cost :
  total_cost 2.5 3.5 4.5 5.5 25 60 80 100 1.3 1.4 1.15 1.2 0.05 0.08 = 6184.815 := by
  sorry

end NUMINAMATH_CALUDE_john_metal_purchase_cost_l1365_136597


namespace NUMINAMATH_CALUDE_snow_clearing_time_l1365_136557

/-- Calculates the number of hours required to clear snow given the total volume,
    initial shoveling capacity, and hourly decrease in capacity. -/
def snow_clearing_hours (total_volume : ℕ) (initial_capacity : ℕ) (hourly_decrease : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

theorem snow_clearing_time :
  snow_clearing_hours 216 25 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_snow_clearing_time_l1365_136557


namespace NUMINAMATH_CALUDE_right_triangle_legs_l1365_136577

theorem right_triangle_legs (a b : ℝ) : 
  a > 0 → b > 0 →
  (a^2 + b^2 = 100) →  -- Pythagorean theorem (hypotenuse = 10)
  (a + b = 14) →       -- Derived from inradius and semiperimeter
  (a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l1365_136577


namespace NUMINAMATH_CALUDE_quadratic_opens_upwards_l1365_136574

/-- A quadratic function f(x) = ax² + bx + c -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The quadratic function opens upwards if a > 0 -/
def opens_upwards (a b c : ℝ) : Prop := a > 0

theorem quadratic_opens_upwards (a b c : ℝ) 
  (h1 : f a b c (-1) = 10)
  (h2 : f a b c 0 = 5)
  (h3 : f a b c 1 = 2)
  (h4 : f a b c 2 = 1)
  (h5 : f a b c 3 = 2) :
  opens_upwards a b c := by sorry

end NUMINAMATH_CALUDE_quadratic_opens_upwards_l1365_136574


namespace NUMINAMATH_CALUDE_residue_of_11_pow_2016_mod_19_l1365_136510

theorem residue_of_11_pow_2016_mod_19 : 11^2016 % 19 = 17 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_11_pow_2016_mod_19_l1365_136510


namespace NUMINAMATH_CALUDE_pie_distribution_l1365_136555

theorem pie_distribution (total_slices : ℕ) (carl_portion : ℚ) (nancy_slices : ℕ) :
  total_slices = 8 →
  carl_portion = 1 / 4 →
  nancy_slices = 2 →
  (total_slices - carl_portion * total_slices - nancy_slices : ℚ) / total_slices = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_pie_distribution_l1365_136555


namespace NUMINAMATH_CALUDE_protest_jail_time_ratio_l1365_136563

theorem protest_jail_time_ratio : 
  let days_of_protest : ℕ := 30
  let num_cities : ℕ := 21
  let arrests_per_day : ℕ := 10
  let days_before_trial : ℕ := 4
  let sentence_weeks : ℕ := 2
  let total_jail_weeks : ℕ := 9900
  let total_arrests : ℕ := days_of_protest * num_cities * arrests_per_day
  let weeks_before_trial : ℕ := total_arrests * days_before_trial / 7
  let weeks_after_trial : ℕ := total_jail_weeks - weeks_before_trial
  let total_possible_weeks : ℕ := total_arrests * sentence_weeks
  (weeks_after_trial : ℚ) / total_possible_weeks = 1 / 2 := by
  sorry

#check protest_jail_time_ratio

end NUMINAMATH_CALUDE_protest_jail_time_ratio_l1365_136563


namespace NUMINAMATH_CALUDE_triangle_problem_l1365_136503

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π →
  B > 0 → B < π →
  C > 0 → C < π →
  (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * (b^2 + c^2 - a^2) →
  (1/2) * b * c * Real.sin A = 3/2 →
  (A = π/3) ∧
  ((b*c - 4*Real.sqrt 3) * Real.cos A + a*c * Real.cos B) / (a^2 - b^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1365_136503


namespace NUMINAMATH_CALUDE_complex_power_2017_l1365_136559

theorem complex_power_2017 : 
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I)
  z^2017 = -Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_power_2017_l1365_136559
