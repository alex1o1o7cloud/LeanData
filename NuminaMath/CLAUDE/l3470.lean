import Mathlib

namespace NUMINAMATH_CALUDE_solutions_satisfy_system_system_implies_solutions_l3470_347038

/-- The system of equations we want to solve -/
def system (x y z : ℝ) : Prop :=
  x + y + z = 2 ∧ x^2 + y^2 + z^2 = 26 ∧ x^3 + y^3 + z^3 = 38

/-- The set of solutions to our system -/
def solutions : Set (ℝ × ℝ × ℝ) :=
  {(1, 4, -3), (1, -3, 4), (4, 1, -3), (4, -3, 1), (-3, 1, 4), (-3, 4, 1)}

/-- Theorem stating that the solutions satisfy the system -/
theorem solutions_satisfy_system : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → system x y z := by
  sorry

/-- Theorem stating that any solution to the system is in our set of solutions -/
theorem system_implies_solutions : ∀ (x y z : ℝ), system x y z → (x, y, z) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_solutions_satisfy_system_system_implies_solutions_l3470_347038


namespace NUMINAMATH_CALUDE_containers_used_is_three_l3470_347036

/-- The number of posters that can be printed with the initial amount of ink -/
def initial_posters : ℕ := 60

/-- The number of posters that can be printed after losing one container of ink -/
def remaining_posters : ℕ := 45

/-- The number of posters that can be printed with one container of ink -/
def posters_per_container : ℕ := initial_posters - remaining_posters

/-- The number of containers used to print the remaining posters -/
def containers_used : ℕ := remaining_posters / posters_per_container

theorem containers_used_is_three :
  containers_used = 3 := by sorry

end NUMINAMATH_CALUDE_containers_used_is_three_l3470_347036


namespace NUMINAMATH_CALUDE_tan_equality_proof_l3470_347001

theorem tan_equality_proof (n : Int) :
  -180 < n ∧ n < 180 → Real.tan (n * π / 180) = Real.tan (210 * π / 180) → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_proof_l3470_347001


namespace NUMINAMATH_CALUDE_number_divided_by_eight_l3470_347010

theorem number_divided_by_eight : ∀ x : ℝ, x / 8 = 4 → x = 32 := by sorry

end NUMINAMATH_CALUDE_number_divided_by_eight_l3470_347010


namespace NUMINAMATH_CALUDE_fencing_cost_is_1634_l3470_347040

/-- Represents the cost of fencing for a single side -/
structure SideCost where
  length : ℝ
  costPerFoot : ℝ

/-- Calculates the total cost of fencing for an irregular four-sided plot -/
def totalFencingCost (sideA sideB sideC sideD : SideCost) : ℝ :=
  sideA.length * sideA.costPerFoot +
  sideB.length * sideB.costPerFoot +
  sideC.length * sideC.costPerFoot +
  sideD.length * sideD.costPerFoot

/-- Theorem stating that the total fencing cost for the given plot is 1634 -/
theorem fencing_cost_is_1634 :
  let sideA : SideCost := { length := 8, costPerFoot := 58 }
  let sideB : SideCost := { length := 5, costPerFoot := 62 }
  let sideC : SideCost := { length := 6, costPerFoot := 64 }
  let sideD : SideCost := { length := 7, costPerFoot := 68 }
  totalFencingCost sideA sideB sideC sideD = 1634 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_is_1634_l3470_347040


namespace NUMINAMATH_CALUDE_tv_horizontal_length_l3470_347031

/-- Calculates the horizontal length of a TV given its aspect ratio and diagonal length -/
theorem tv_horizontal_length 
  (aspect_width : ℝ) 
  (aspect_height : ℝ) 
  (diagonal_length : ℝ) 
  (aspect_width_pos : 0 < aspect_width)
  (aspect_height_pos : 0 < aspect_height)
  (diagonal_length_pos : 0 < diagonal_length) :
  let horizontal_length := aspect_width * diagonal_length / Real.sqrt (aspect_width^2 + aspect_height^2)
  horizontal_length = 16 * diagonal_length / Real.sqrt 337 :=
by sorry

end NUMINAMATH_CALUDE_tv_horizontal_length_l3470_347031


namespace NUMINAMATH_CALUDE_percentage_decrease_in_hours_l3470_347026

/-- Represents Jane's toy bear production --/
structure BearProduction where
  bears_without_assistant : ℝ
  hours_without_assistant : ℝ
  bears_with_assistant : ℝ
  hours_with_assistant : ℝ

/-- The conditions of Jane's toy bear production --/
def production_conditions (p : BearProduction) : Prop :=
  p.bears_with_assistant = 1.8 * p.bears_without_assistant ∧
  (p.bears_with_assistant / p.hours_with_assistant) = 2 * (p.bears_without_assistant / p.hours_without_assistant)

/-- The theorem stating the percentage decrease in hours worked --/
theorem percentage_decrease_in_hours (p : BearProduction) 
  (h : production_conditions p) : 
  (p.hours_without_assistant - p.hours_with_assistant) / p.hours_without_assistant * 100 = 10 := by
  sorry


end NUMINAMATH_CALUDE_percentage_decrease_in_hours_l3470_347026


namespace NUMINAMATH_CALUDE_arrangements_count_l3470_347083

-- Define the number of people and exits
def num_people : ℕ := 5
def num_exits : ℕ := 4

-- Define the function to calculate the number of arrangements
def num_arrangements : ℕ := sorry

-- Theorem statement
theorem arrangements_count : num_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l3470_347083


namespace NUMINAMATH_CALUDE_fourth_group_number_l3470_347000

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (secondNumber : ℕ) : ℕ → ℕ :=
  fun groupIndex => secondNumber + (groupIndex - 2) * (totalStudents / sampleSize)

theorem fourth_group_number
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (secondNumber : ℕ)
  (h1 : totalStudents = 60)
  (h2 : sampleSize = 5)
  (h3 : secondNumber = 16) :
  systematicSample totalStudents sampleSize secondNumber 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_fourth_group_number_l3470_347000


namespace NUMINAMATH_CALUDE_specific_system_is_linear_l3470_347051

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop
  eq_def : ∀ x y, eq x y ↔ a * x + b * y = c

/-- A system of two equations -/
structure EquationSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The specific system of equations we want to prove is linear -/
def specificSystem : EquationSystem where
  eq1 := {
    a := 1
    b := 1
    c := 1
    eq := λ x y => x + y = 1
    eq_def := by sorry
  }
  eq2 := {
    a := 1
    b := -1
    c := 2
    eq := λ x y => x - y = 2
    eq_def := by sorry
  }

/-- Definition of a system of two linear equations -/
def isSystemOfTwoLinearEquations (system : EquationSystem) : Prop :=
  ∃ (x y : ℝ), 
    system.eq1.eq x y ∧ 
    system.eq2.eq x y ∧
    (∀ z, system.eq1.eq x z ↔ system.eq1.a * x + system.eq1.b * z = system.eq1.c) ∧
    (∀ z, system.eq2.eq x z ↔ system.eq2.a * x + system.eq2.b * z = system.eq2.c)

theorem specific_system_is_linear : isSystemOfTwoLinearEquations specificSystem := by
  sorry

end NUMINAMATH_CALUDE_specific_system_is_linear_l3470_347051


namespace NUMINAMATH_CALUDE_no_even_increasing_function_l3470_347052

open Function

-- Define what it means for a function to be even
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define what it means for a function to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Theorem stating that no function can be both even and increasing
theorem no_even_increasing_function : ¬ ∃ f : ℝ → ℝ, IsEven f ∧ IsIncreasing f := by
  sorry

end NUMINAMATH_CALUDE_no_even_increasing_function_l3470_347052


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_31_17_l3470_347095

theorem smallest_fraction_greater_than_31_17 :
  ∀ a b : ℤ, b < 17 → (a : ℚ) / b > 31 / 17 → 11 / 6 ≤ (a : ℚ) / b :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_31_17_l3470_347095


namespace NUMINAMATH_CALUDE_bacon_suggestion_count_bacon_suggestion_proof_l3470_347005

theorem bacon_suggestion_count : ℕ → ℕ → ℕ → Prop :=
  fun mashed_potatoes_count difference bacon_count =>
    (mashed_potatoes_count = 457) →
    (mashed_potatoes_count = bacon_count + difference) →
    (difference = 63) →
    (bacon_count = 394)

-- The proof is omitted
theorem bacon_suggestion_proof : bacon_suggestion_count 457 63 394 := by
  sorry

end NUMINAMATH_CALUDE_bacon_suggestion_count_bacon_suggestion_proof_l3470_347005


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3470_347014

theorem right_triangle_hypotenuse (QR RS QS : ℝ) (cos_R : ℝ) : 
  cos_R = 3/5 →  -- Given condition
  RS = 10 →     -- Given condition
  QR = RS * cos_R →  -- Definition of cosine in right triangle
  QS^2 = RS^2 - QR^2 →  -- Pythagorean theorem
  QS = 8 :=  -- Conclusion to prove
by sorry  -- Proof omitted

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3470_347014


namespace NUMINAMATH_CALUDE_temperature_difference_is_8_l3470_347037

-- Define the temperatures
def temp_top : ℝ := -9
def temp_foot : ℝ := -1

-- Define the temperature difference
def temp_difference : ℝ := temp_foot - temp_top

-- Theorem statement
theorem temperature_difference_is_8 : temp_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_is_8_l3470_347037


namespace NUMINAMATH_CALUDE_system_solution_l3470_347071

theorem system_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (eq1 : x + y^2 + z^3 = 3)
  (eq2 : y + z^2 + x^3 = 3)
  (eq3 : z + x^2 + y^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3470_347071


namespace NUMINAMATH_CALUDE_subset_implies_m_eq_two_l3470_347034

/-- The set A of solutions to the quadratic equation x^2 + 3x + 2 = 0 -/
def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}

/-- The set B of solutions to the quadratic equation x^2 + (m+1)x + m = 0 -/
def B (m : ℝ) : Set ℝ := {x | x^2 + (m+1)*x + m = 0}

/-- Theorem stating that if A is a subset of B, then m must equal 2 -/
theorem subset_implies_m_eq_two : A ⊆ B 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_eq_two_l3470_347034


namespace NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l3470_347027

theorem smallest_solution_for_floor_equation :
  let x : ℝ := 131 / 11
  ∀ y : ℝ, y > 0 → (⌊y^2⌋ : ℝ) - y * ⌊y⌋ = 10 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l3470_347027


namespace NUMINAMATH_CALUDE_notebook_cost_l3470_347053

theorem notebook_cost (total_students : ℕ) (total_cost : ℕ) 
  (h_total_students : total_students = 36)
  (h_total_cost : total_cost = 2376)
  (s : ℕ) (n : ℕ) (c : ℕ)
  (h_majority : s > total_students / 2)
  (h_same_number : ∀ i j, i ≠ j → i < s → j < s → n = n)
  (h_at_least_two : n ≥ 2)
  (h_cost_greater : c > n)
  (h_total_equation : s * c * n = total_cost) :
  c = 11 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l3470_347053


namespace NUMINAMATH_CALUDE_phase_shift_sine_function_l3470_347002

/-- The phase shift of the function y = 4 sin(3x - π/4) is π/12 to the right -/
theorem phase_shift_sine_function :
  let f : ℝ → ℝ := λ x => 4 * Real.sin (3 * x - π / 4)
  ∃ (shift : ℝ), shift = π / 12 ∧ 
    ∀ x, f (x + shift) = 4 * Real.sin (3 * x) :=
by sorry

end NUMINAMATH_CALUDE_phase_shift_sine_function_l3470_347002


namespace NUMINAMATH_CALUDE_last_ten_digits_periodicity_l3470_347092

theorem last_ten_digits_periodicity (n : ℕ) (h : n ≥ 10) :
  2^n % 10^10 = 2^(n + 4 * 10^9) % 10^10 := by
  sorry

end NUMINAMATH_CALUDE_last_ten_digits_periodicity_l3470_347092


namespace NUMINAMATH_CALUDE_manfred_average_paycheck_l3470_347019

/-- Calculates the average paycheck amount for Manfred's year, rounded to the nearest dollar. -/
def average_paycheck (total_paychecks : ℕ) (initial_paychecks : ℕ) (initial_amount : ℚ) (increase : ℚ) : ℕ :=
  let remaining_paychecks := total_paychecks - initial_paychecks
  let total_amount := initial_paychecks * initial_amount + remaining_paychecks * (initial_amount + increase)
  let average := total_amount / total_paychecks
  (average + 1/2).floor.toNat

/-- Proves that Manfred's average paycheck for the year, rounded to the nearest dollar, is $765. -/
theorem manfred_average_paycheck :
  average_paycheck 26 6 750 20 = 765 := by
  sorry

end NUMINAMATH_CALUDE_manfred_average_paycheck_l3470_347019


namespace NUMINAMATH_CALUDE_find_number_l3470_347074

theorem find_number : ∃ x : ℝ, (0.4 * x = 0.75 * 100 + 50) ∧ (x = 312.5) := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3470_347074


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3470_347081

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (sum_eq : a + b = 5)
  (sum_of_cubes_eq : a^3 + b^3 = 125) : 
  a * b = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3470_347081


namespace NUMINAMATH_CALUDE_perpendicular_skew_lines_iff_plane_exists_l3470_347015

/-- Two lines in 3D space are skew if they are not parallel and do not intersect. -/
def are_skew_lines (a b : Line3D) : Prop := sorry

/-- A line is perpendicular to another line if their direction vectors are orthogonal. -/
def line_perpendicular (a b : Line3D) : Prop := sorry

/-- A plane passes through a line if the line is contained in the plane. -/
def plane_passes_through_line (p : Plane3D) (l : Line3D) : Prop := sorry

/-- A plane is perpendicular to a line if the normal vector of the plane is parallel to the direction vector of the line. -/
def plane_perpendicular_to_line (p : Plane3D) (l : Line3D) : Prop := sorry

/-- Main theorem: For two skew lines, one line is perpendicular to the other if and only if
    there exists a plane passing through the first line and perpendicular to the second line. -/
theorem perpendicular_skew_lines_iff_plane_exists (a b : Line3D) 
  (h : are_skew_lines a b) : 
  line_perpendicular a b ↔ 
  ∃ (p : Plane3D), plane_passes_through_line p a ∧ plane_perpendicular_to_line p b := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_skew_lines_iff_plane_exists_l3470_347015


namespace NUMINAMATH_CALUDE_simple_interest_rate_l3470_347042

/-- Given a principal amount and a simple interest rate, if the amount after 12 years
    is 9/6 of the principal, then the rate is 100/24 -/
theorem simple_interest_rate (P R : ℝ) (P_pos : P > 0) : 
  P * (1 + R * 12 / 100) = P * (9 / 6) → R = 100 / 24 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l3470_347042


namespace NUMINAMATH_CALUDE_consecutive_integer_product_divisibility_l3470_347044

theorem consecutive_integer_product_divisibility (k : ℤ) :
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 11 * m) →
  (∃ m : ℤ, n = 10 * m) ∧
  (∃ m : ℤ, n = 22 * m) ∧
  (∃ m : ℤ, n = 33 * m) ∧
  (∃ m : ℤ, n = 66 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 44 * m) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integer_product_divisibility_l3470_347044


namespace NUMINAMATH_CALUDE_blocks_left_l3470_347009

theorem blocks_left (initial_blocks : ℕ) (used_blocks : ℕ) : 
  initial_blocks = 78 → used_blocks = 19 → initial_blocks - used_blocks = 59 := by
sorry

end NUMINAMATH_CALUDE_blocks_left_l3470_347009


namespace NUMINAMATH_CALUDE_f_g_one_eq_one_solution_set_eq_two_l3470_347007

-- Define the domain of x
inductive X : Type
| one : X
| two : X
| three : X

-- Define functions f and g
def f : X → ℕ
| X.one => 1
| X.two => 3
| X.three => 1

def g : X → ℕ
| X.one => 3
| X.two => 2
| X.three => 1

-- Define composition of f and g
def f_comp_g (x : X) : ℕ := f (match g x with
  | 1 => X.one
  | 2 => X.two
  | 3 => X.three
  | _ => X.one)

def g_comp_f (x : X) : ℕ := g (match f x with
  | 1 => X.one
  | 2 => X.two
  | 3 => X.three
  | _ => X.one)

theorem f_g_one_eq_one : f_comp_g X.one = 1 := by sorry

theorem solution_set_eq_two :
  (∀ x : X, f_comp_g x > g_comp_f x ↔ x = X.two) := by sorry

end NUMINAMATH_CALUDE_f_g_one_eq_one_solution_set_eq_two_l3470_347007


namespace NUMINAMATH_CALUDE_sin_pi_half_minus_x_is_even_l3470_347013

/-- The function f(x) = sin(π/2 - x) is even, implying symmetry about the y-axis -/
theorem sin_pi_half_minus_x_is_even :
  ∀ x : ℝ, Real.sin (π / 2 - x) = Real.sin (π / 2 - (-x)) :=
by sorry

end NUMINAMATH_CALUDE_sin_pi_half_minus_x_is_even_l3470_347013


namespace NUMINAMATH_CALUDE_probability_ratio_l3470_347035

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of different numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The number of cards with each number -/
def cards_per_number : ℕ := 5

/-- The probability of drawing five cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The probability of drawing four cards with one number and one card with a different number -/
def q : ℚ := (2250 : ℚ) / (Nat.choose total_cards cards_drawn)

theorem probability_ratio :
  q / p = 225 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l3470_347035


namespace NUMINAMATH_CALUDE_field_length_calculation_l3470_347057

theorem field_length_calculation (width : ℝ) (pond_area : ℝ) (pond_percentage : ℝ) : 
  pond_area = 150 →
  pond_percentage = 0.4 →
  let length := 3 * width
  let field_area := length * width
  pond_area = pond_percentage * field_area →
  length = 15 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_field_length_calculation_l3470_347057


namespace NUMINAMATH_CALUDE_red_balls_count_l3470_347018

/-- Represents a box containing white and red balls -/
structure BallBox where
  white_balls : ℕ
  red_balls : ℕ

/-- The probability of picking a red ball from the box -/
def red_probability (box : BallBox) : ℚ :=
  box.red_balls / (box.white_balls + box.red_balls)

/-- Theorem: If there are 12 white balls and the probability of picking a red ball is 1/4,
    then the number of red balls is 4 -/
theorem red_balls_count (box : BallBox) 
    (h1 : box.white_balls = 12)
    (h2 : red_probability box = 1/4) : 
    box.red_balls = 4 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l3470_347018


namespace NUMINAMATH_CALUDE_line_chart_drawing_method_l3470_347021

/-- Represents a point on a grid -/
structure GridPoint where
  x : ℝ
  y : ℝ

/-- Represents a line chart -/
structure LineChart where
  points : List GridPoint
  unit_length : ℝ
  quantity : ℝ → ℝ

/-- The method of drawing a line chart -/
def draw_line_chart (chart : LineChart) : Prop :=
  ∃ (plotted_points : List GridPoint) (connected_points : List GridPoint),
    plotted_points = chart.points ∧
    connected_points = chart.points ∧
    (∀ p ∈ chart.points, p.y = chart.quantity (p.x * chart.unit_length))

theorem line_chart_drawing_method (chart : LineChart) :
  draw_line_chart chart ↔
  (∃ (plotted_points : List GridPoint) (connected_points : List GridPoint),
    plotted_points = chart.points ∧
    connected_points = chart.points) :=
sorry

end NUMINAMATH_CALUDE_line_chart_drawing_method_l3470_347021


namespace NUMINAMATH_CALUDE_attendance_difference_l3470_347008

def football_game_attendance (saturday_attendance : ℕ) : Prop :=
  let monday_attendance : ℕ := saturday_attendance - saturday_attendance / 4
  let wednesday_attendance : ℕ := monday_attendance + monday_attendance / 2
  let friday_attendance : ℕ := saturday_attendance + monday_attendance
  let thursday_attendance : ℕ := 45
  let sunday_attendance : ℕ := saturday_attendance - saturday_attendance * 15 / 100
  let total_attendance : ℕ := saturday_attendance + monday_attendance + wednesday_attendance + 
                               thursday_attendance + friday_attendance + sunday_attendance
  let expected_attendance : ℕ := 350
  total_attendance - expected_attendance = 133

theorem attendance_difference : 
  football_game_attendance 80 :=
sorry

end NUMINAMATH_CALUDE_attendance_difference_l3470_347008


namespace NUMINAMATH_CALUDE_cricketer_matches_l3470_347082

theorem cricketer_matches (total_average : ℝ) (first_8_average : ℝ) (last_4_average : ℝ)
  (h1 : total_average = 48)
  (h2 : first_8_average = 40)
  (h3 : last_4_average = 64) :
  ∃ (n : ℕ), n * total_average = 8 * first_8_average + 4 * last_4_average ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_matches_l3470_347082


namespace NUMINAMATH_CALUDE_original_triangle_area_l3470_347011

theorem original_triangle_area
  (original_area : ℝ)
  (new_area : ℝ)
  (h1 : new_area = 32)
  (h2 : new_area = 4 * original_area) :
  original_area = 8 :=
by sorry

end NUMINAMATH_CALUDE_original_triangle_area_l3470_347011


namespace NUMINAMATH_CALUDE_original_fraction_l3470_347017

theorem original_fraction (x y : ℚ) 
  (h1 : x / (y + 1) = 1 / 2) 
  (h2 : (x + 1) / y = 1) : 
  x / y = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_l3470_347017


namespace NUMINAMATH_CALUDE_solve_for_time_l3470_347004

-- Define the exponential growth formula
def exponential_growth (P₀ A r t : ℝ) : Prop :=
  A = P₀ * Real.exp (r * t)

-- Theorem statement
theorem solve_for_time (P₀ A r t : ℝ) (h_pos : P₀ > 0) (h_r_nonzero : r ≠ 0) :
  exponential_growth P₀ A r t ↔ t = Real.log (A / P₀) / r :=
sorry

end NUMINAMATH_CALUDE_solve_for_time_l3470_347004


namespace NUMINAMATH_CALUDE_wine_glass_ball_radius_l3470_347084

theorem wine_glass_ball_radius 
  (parabola : ℝ → ℝ → Prop) 
  (h_parabola : ∀ x y, parabola x y ↔ x^2 = 2*y) 
  (h_y_range : ∀ y, parabola x y → 0 ≤ y ∧ y ≤ 20) 
  (ball_touches_bottom : ∃ r, r > 0 ∧ ∀ x y, parabola x y → x^2 + y^2 ≥ r^2) :
  ∃ r, r > 0 ∧ r ≤ 1 ∧ 
    (∀ x y, parabola x y → x^2 + y^2 ≥ r^2) ∧
    (∀ r', r' > 0 ∧ r' ≤ 1 → 
      (∀ x y, parabola x y → x^2 + y^2 ≥ r'^2) → 
      r' ≤ r) :=
by sorry

end NUMINAMATH_CALUDE_wine_glass_ball_radius_l3470_347084


namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_150_l3470_347087

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_odd_number_of_factors (n : ℕ) : Prop := is_perfect_square n

theorem greatest_number_with_odd_factors_under_150 :
  ∃ n : ℕ, n < 150 ∧ has_odd_number_of_factors n ∧
  ∀ m : ℕ, m < 150 → has_odd_number_of_factors m → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_150_l3470_347087


namespace NUMINAMATH_CALUDE_fourth_grade_students_l3470_347075

/-- The number of students in fourth grade at the end of the year -/
def final_students (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem: Given the initial number of students, the number of students who left,
    and the number of new students, prove that the final number of students is 47 -/
theorem fourth_grade_students :
  final_students 11 6 42 = 47 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l3470_347075


namespace NUMINAMATH_CALUDE_wendy_glasses_difference_l3470_347029

/-- The number of glasses polished by Wendy -/
def total_glasses : ℕ := 110

/-- The number of small glasses polished by Wendy -/
def small_glasses : ℕ := 50

/-- The number of large glasses polished by Wendy -/
def large_glasses : ℕ := total_glasses - small_glasses

theorem wendy_glasses_difference :
  large_glasses > small_glasses ∧ large_glasses - small_glasses = 10 := by
  sorry

end NUMINAMATH_CALUDE_wendy_glasses_difference_l3470_347029


namespace NUMINAMATH_CALUDE_max_cookies_eaten_24_l3470_347046

/-- Given two siblings sharing cookies, where one eats a positive multiple
    of the other's cookies, this function calculates the maximum number
    of cookies the first sibling could have eaten. -/
def max_cookies_eaten (total_cookies : ℕ) : ℕ :=
  total_cookies / 2

/-- Theorem stating that given 24 cookies shared between two siblings,
    where one sibling eats a positive multiple of the other's cookies,
    the maximum number of cookies the first sibling could have eaten is 12. -/
theorem max_cookies_eaten_24 :
  max_cookies_eaten 24 = 12 := by
  sorry

#eval max_cookies_eaten 24

end NUMINAMATH_CALUDE_max_cookies_eaten_24_l3470_347046


namespace NUMINAMATH_CALUDE_girls_picked_more_l3470_347080

-- Define the number of mushrooms picked by each person
variable (N I A V : ℕ)

-- Define the conditions
def natasha_most := N > I ∧ N > A ∧ N > V
def ira_not_least := I ≤ N ∧ I ≥ A ∧ I ≥ V
def alexey_more_than_vitya := A > V

-- Theorem to prove
theorem girls_picked_more (h1 : natasha_most N I A V) 
                          (h2 : ira_not_least N I A V) 
                          (h3 : alexey_more_than_vitya A V) : 
  N + I > A + V := by
  sorry

end NUMINAMATH_CALUDE_girls_picked_more_l3470_347080


namespace NUMINAMATH_CALUDE_truck_speed_theorem_l3470_347062

/-- The average speed of Truck X in miles per hour -/
def truck_x_speed : ℝ := 47

/-- The average speed of Truck Y in miles per hour -/
def truck_y_speed : ℝ := 53

/-- The initial distance between Truck X and Truck Y in miles -/
def initial_distance : ℝ := 13

/-- The time it takes for Truck Y to overtake and get ahead of Truck X in hours -/
def overtake_time : ℝ := 3

/-- The distance Truck Y is ahead of Truck X after overtaking in miles -/
def final_distance : ℝ := 5

theorem truck_speed_theorem :
  truck_x_speed * overtake_time + initial_distance + final_distance = truck_y_speed * overtake_time :=
by sorry

end NUMINAMATH_CALUDE_truck_speed_theorem_l3470_347062


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3470_347041

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal [false, true, true, true, false, true, true, false, true]) = [1, 1, 2, 3, 2] := by
  sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3470_347041


namespace NUMINAMATH_CALUDE_rectangle_height_decrease_l3470_347047

theorem rectangle_height_decrease (b h : ℝ) (h_pos : 0 < b) (h_pos' : 0 < h) :
  let new_base := 1.1 * b
  let new_height := h * (1 - 9 / 11 / 100)
  b * h = new_base * new_height := by
  sorry

end NUMINAMATH_CALUDE_rectangle_height_decrease_l3470_347047


namespace NUMINAMATH_CALUDE_largest_divisor_l3470_347060

theorem largest_divisor (A B : ℕ) (h1 : 13 = 4 * A + B) (h2 : B < A) : A ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_l3470_347060


namespace NUMINAMATH_CALUDE_apple_distribution_l3470_347025

theorem apple_distribution (x y : ℕ) : 
  y = 5 * x + 12 ∧ 0 < 8 * x - y ∧ 8 * x - y < 8 → 
  (x = 5 ∧ y = 37) ∨ (x = 6 ∧ y = 42) := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l3470_347025


namespace NUMINAMATH_CALUDE_hcf_problem_l3470_347059

theorem hcf_problem (x y : ℕ+) 
  (h1 : Nat.lcm x y = 560) 
  (h2 : x * y = 42000) : 
  Nat.gcd x y = 75 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l3470_347059


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l3470_347055

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def is_nonagon (n : ℕ) : Prop := n = 9

theorem nonagon_diagonals :
  ∀ n : ℕ, is_nonagon n → num_diagonals n = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l3470_347055


namespace NUMINAMATH_CALUDE_cut_rectangle_pentagon_area_cut_rectangle_pentagon_area_proof_l3470_347072

/-- Pentagon formed by removing a triangle from a rectangle --/
structure CutRectanglePentagon where
  sides : Finset ℕ
  side_count : sides.card = 5
  side_values : sides = {14, 21, 22, 28, 35}

/-- Theorem stating the area of the specific pentagon --/
theorem cut_rectangle_pentagon_area (p : CutRectanglePentagon) : ℕ :=
  1176

#check cut_rectangle_pentagon_area

/-- Proof of the theorem --/
theorem cut_rectangle_pentagon_area_proof (p : CutRectanglePentagon) :
  cut_rectangle_pentagon_area p = 1176 := by
  sorry

end NUMINAMATH_CALUDE_cut_rectangle_pentagon_area_cut_rectangle_pentagon_area_proof_l3470_347072


namespace NUMINAMATH_CALUDE_painting_price_increase_percentage_l3470_347049

/-- Proves that the percentage increase in the cost of each painting is 20% --/
theorem painting_price_increase_percentage :
  let original_jewelry_price : ℚ := 30
  let original_painting_price : ℚ := 100
  let jewelry_price_increase : ℚ := 10
  let jewelry_quantity : ℕ := 2
  let painting_quantity : ℕ := 5
  let total_cost : ℚ := 680
  let new_jewelry_price : ℚ := original_jewelry_price + jewelry_price_increase
  let painting_price_increase_percentage : ℚ := 20

  (jewelry_quantity : ℚ) * new_jewelry_price + 
  (painting_quantity : ℚ) * original_painting_price * (1 + painting_price_increase_percentage / 100) = 
  total_cost :=
by sorry

end NUMINAMATH_CALUDE_painting_price_increase_percentage_l3470_347049


namespace NUMINAMATH_CALUDE_line_equation_slope_intercept_l3470_347061

/-- The equation of a line with slope -1 and y-intercept -1 is x + y + 1 = 0 -/
theorem line_equation_slope_intercept (x y : ℝ) : 
  (∀ x y, y = -x - 1) ↔ (∀ x y, x + y + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_equation_slope_intercept_l3470_347061


namespace NUMINAMATH_CALUDE_sandy_initial_money_l3470_347050

/-- Given that Sandy spent $6 on a pie and has $57 left, prove that she initially had $63. -/
theorem sandy_initial_money :
  ∀ (initial_money spent_on_pie money_left : ℕ),
    spent_on_pie = 6 →
    money_left = 57 →
    initial_money = spent_on_pie + money_left →
    initial_money = 63 := by
  sorry

end NUMINAMATH_CALUDE_sandy_initial_money_l3470_347050


namespace NUMINAMATH_CALUDE_horner_v2_equals_16_l3470_347054

/-- Horner's method for evaluating a polynomial -/
def horner_v2 (x : ℝ) : ℝ :=
  let v0 : ℝ := x
  let v1 : ℝ := 3 * v0 + 1
  v1 * v0 + 2

/-- The polynomial f(x) = 3x^4 + x^3 + 2x^2 + x + 4 -/
def f (x : ℝ) : ℝ := 3*x^4 + x^3 + 2*x^2 + x + 4

theorem horner_v2_equals_16 : horner_v2 2 = 16 := by
  sorry


end NUMINAMATH_CALUDE_horner_v2_equals_16_l3470_347054


namespace NUMINAMATH_CALUDE_parallelogram_height_l3470_347088

/-- Given a parallelogram with area 180 square centimeters and base 18 cm, its height is 10 cm. -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 180 → base = 18 → area = base * height → height = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3470_347088


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l3470_347016

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l3470_347016


namespace NUMINAMATH_CALUDE_min_marked_elements_eq_666_l3470_347006

/-- The minimum number of marked elements in {1, ..., 2000} such that
    for every pair (k, 2k) where 1 ≤ k ≤ 1000, at least one of k or 2k is marked. -/
def min_marked_elements : ℕ :=
  let S := Finset.range 2000
  Finset.filter (fun n => ∃ k ∈ Finset.range 1000, n = k ∨ n = 2 * k) S |>.card

/-- The theorem stating that the minimum number of marked elements is 666. -/
theorem min_marked_elements_eq_666 : min_marked_elements = 666 := by
  sorry

end NUMINAMATH_CALUDE_min_marked_elements_eq_666_l3470_347006


namespace NUMINAMATH_CALUDE_not_perfect_square_l3470_347068

theorem not_perfect_square : 
  (∃ a : ℕ, 1^2016 = a^2) ∧ 
  (∀ b : ℕ, 2^2017 ≠ b^2) ∧ 
  (∃ c : ℕ, 3^2018 = c^2) ∧ 
  (∃ d : ℕ, 4^2019 = d^2) ∧ 
  (∃ e : ℕ, 5^2020 = e^2) := by
sorry

end NUMINAMATH_CALUDE_not_perfect_square_l3470_347068


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3470_347028

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_standard_equation :
  ∀ (a b m : ℝ) (P : ℝ × ℝ),
    b > a ∧ a > 0 ∧ m > 0 →
    P.1 = Real.sqrt 5 ∧ P.2 = m →
    P.1^2 / a^2 - P.2^2 / b^2 = 1 →
    P.1 = Real.sqrt (a^2 + b^2) →
    (∃ (A B : ℝ × ℝ),
      (A.2 - P.2) / (A.1 - P.1) = b / a ∧
      (B.2 - P.2) / (B.1 - P.1) = -b / a ∧
      (A.1 - P.1) * (B.2 - P.2) - (A.2 - P.2) * (B.1 - P.1) = 2) →
    ∀ (x y : ℝ), x^2 - y^2 / 4 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3470_347028


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3470_347030

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Part 1
theorem solution_set_part1 (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 - a*x + 1 > 0 ↔ x < 1/3 ∨ x > 1/2) :=
sorry

-- Part 2
theorem range_of_a_part2 (a : ℝ) :
  (∀ x, x ∈ Set.Ioc (-1) 0 → f a (3-a) x ≥ 0) →
  a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3470_347030


namespace NUMINAMATH_CALUDE_digit_move_correctness_l3470_347056

theorem digit_move_correctness : 
  let original_number := 102
  let moved_digit := 2
  let base := 10
  let new_left_term := original_number - moved_digit
  let new_right_term := base ^ moved_digit
  (new_left_term - new_right_term = 1) = True
  := by sorry

end NUMINAMATH_CALUDE_digit_move_correctness_l3470_347056


namespace NUMINAMATH_CALUDE_xy_minus_10_squared_l3470_347020

theorem xy_minus_10_squared (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : 
  (x * y - 10)^2 ≥ 64 ∧ 
  ((x * y - 10)^2 = 64 ↔ (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)) := by
  sorry

end NUMINAMATH_CALUDE_xy_minus_10_squared_l3470_347020


namespace NUMINAMATH_CALUDE_bch_unique_product_l3470_347012

/-- Represents a letter of the alphabet -/
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J | K | L | M
| N | O | P | Q | R | S | T | U | V | W | X | Y | Z

/-- Assigns a numerical value to each letter -/
def letterValue : Letter → Nat
| Letter.A => 1
| Letter.B => 2
| Letter.C => 3
| Letter.D => 4
| Letter.E => 5
| Letter.F => 6
| Letter.G => 7
| Letter.H => 8
| Letter.I => 9
| Letter.J => 10
| Letter.K => 11
| Letter.L => 12
| Letter.M => 13
| Letter.N => 14
| Letter.O => 15
| Letter.P => 16
| Letter.Q => 17
| Letter.R => 18
| Letter.S => 19
| Letter.T => 20
| Letter.U => 21
| Letter.V => 22
| Letter.W => 23
| Letter.X => 24
| Letter.Y => 25
| Letter.Z => 26

/-- Calculates the product of a three-letter list -/
def productOfList (a b c : Letter) : Nat :=
  letterValue a * letterValue b * letterValue c

/-- Checks if three letters are in alphabetical order -/
def isAlphabeticalOrder (a b c : Letter) : Prop :=
  letterValue a ≤ letterValue b ∧ letterValue b ≤ letterValue c

/-- Main theorem: BCH is the only other three-letter list with product equal to BDF -/
theorem bch_unique_product :
  ∀ (x y z : Letter),
    x ≠ y ∧ y ≠ z ∧ x ≠ z →
    isAlphabeticalOrder x y z →
    productOfList x y z = productOfList Letter.B Letter.D Letter.F →
    x = Letter.B ∧ y = Letter.C ∧ z = Letter.H :=
by sorry


end NUMINAMATH_CALUDE_bch_unique_product_l3470_347012


namespace NUMINAMATH_CALUDE_S_equality_l3470_347097

/-- S_k(n) function (not defined, assumed to exist) -/
noncomputable def S_k (k n : ℕ) : ℕ := sorry

/-- The sum S as defined in the problem -/
noncomputable def S (n k : ℕ) : ℚ :=
  (Finset.range ((k + 1) / 2)).sum (λ i =>
    Nat.choose (k + 1) (2 * i + 1) * S_k (k - 2 * i) n)

/-- Theorem stating the equality to be proved -/
theorem S_equality (n k : ℕ) :
  S n k = ((n + 1)^(k + 1) + n^(k + 1) - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_S_equality_l3470_347097


namespace NUMINAMATH_CALUDE_rhombus_area_l3470_347089

/-- Theorem: Area of a rhombus with given side and diagonal lengths -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (area : ℝ) : 
  side = 26 → diagonal1 = 20 → area = 480 → 
  ∃ (diagonal2 : ℝ), 
    diagonal2 ^ 2 = 4 * (side ^ 2 - (diagonal1 / 2) ^ 2) ∧ 
    area = (diagonal1 * diagonal2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_area_l3470_347089


namespace NUMINAMATH_CALUDE_quadratic_expression_values_l3470_347067

theorem quadratic_expression_values (a b : ℝ) 
  (ha : a^2 = 16)
  (hb : abs b = 3)
  (hab : a * b < 0) :
  (a - b)^2 + a * b^2 = 85 ∨ (a - b)^2 + a * b^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_values_l3470_347067


namespace NUMINAMATH_CALUDE_lemonade_percentage_l3470_347070

/-- Proves that the percentage of lemonade in the second solution is 45% -/
theorem lemonade_percentage
  (first_solution_carbonated : ℝ)
  (second_solution_carbonated : ℝ)
  (mixture_ratio : ℝ)
  (mixture_carbonated : ℝ)
  (h1 : first_solution_carbonated = 0.8)
  (h2 : second_solution_carbonated = 0.55)
  (h3 : mixture_ratio = 0.5)
  (h4 : mixture_carbonated = 0.675)
  (h5 : mixture_ratio * first_solution_carbonated + (1 - mixture_ratio) * second_solution_carbonated = mixture_carbonated) :
  1 - second_solution_carbonated = 0.45 :=
by sorry

end NUMINAMATH_CALUDE_lemonade_percentage_l3470_347070


namespace NUMINAMATH_CALUDE_inequality_proof_l3470_347003

theorem inequality_proof (x : ℝ) (hx : x > 0) : x^2 + 1/(4*x) ≥ 3/4 ∧ Real.sqrt 3 - 1 < 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3470_347003


namespace NUMINAMATH_CALUDE_age_difference_l3470_347086

theorem age_difference (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : 3 * b = 2 * a) (h4 : a + b = 60) : a - b = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3470_347086


namespace NUMINAMATH_CALUDE_three_hundredth_term_of_specific_sequence_l3470_347077

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem three_hundredth_term_of_specific_sequence :
  let a₁ := 8
  let a₂ := -8
  let r := a₂ / a₁
  geometric_sequence a₁ r 300 = -8 := by
sorry

end NUMINAMATH_CALUDE_three_hundredth_term_of_specific_sequence_l3470_347077


namespace NUMINAMATH_CALUDE_president_and_vp_from_seven_l3470_347048

/-- The number of ways to choose a President and a Vice-President from a group of n people,
    where the two positions must be held by different people. -/
def choose_president_and_vp (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 42 ways to choose a President and a Vice-President from a group of 7 people,
    where the two positions must be held by different people. -/
theorem president_and_vp_from_seven : choose_president_and_vp 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_president_and_vp_from_seven_l3470_347048


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l3470_347024

/-- Given a line L1: 2x + 3y = 12, and a perpendicular line L2 with y-intercept -1,
    the x-intercept of L2 is 2/3. -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 2 * x + 3 * y = 12
  let m1 : ℝ := -2 / 3  -- slope of L1
  let m2 : ℝ := 3 / 2   -- slope of L2 (perpendicular to L1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = m2 * x - 1  -- equation of L2
  (∀ x y, L2 x y → (x = 0 → y = -1)) →  -- y-intercept of L2 is -1
  (∀ x, L2 x 0 → x = 2/3) :=  -- x-intercept of L2 is 2/3
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l3470_347024


namespace NUMINAMATH_CALUDE_odd_function_half_period_zero_l3470_347093

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the smallest positive period T
variable (T : ℝ)

-- Define the oddness property of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the periodicity property of f with period T
def has_period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

-- Define that T is the smallest positive period
def is_smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ has_period f T ∧ ∀ S, 0 < S ∧ S < T → ¬(has_period f S)

-- State the theorem
theorem odd_function_half_period_zero
  (h_odd : is_odd f)
  (h_period : is_smallest_positive_period f T) :
  f (-T/2) = 0 :=
sorry

end NUMINAMATH_CALUDE_odd_function_half_period_zero_l3470_347093


namespace NUMINAMATH_CALUDE_x₄_x₁_diff_l3470_347096

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the relationship between f and g
axiom g_def : ∀ x, g x = -f (200 - x)

-- Define the x-intercepts
variable (x₁ x₂ x₃ x₄ : ℝ)

-- The x-intercepts are in increasing order
axiom x_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄

-- The difference between x₃ and x₂
axiom x₃_x₂_diff : x₃ - x₂ = 200

-- The vertex of g is on the graph of f
axiom vertex_on_f : ∃ x, g x = f x ∧ ∀ y, g y ≤ g x

-- Theorem to prove
theorem x₄_x₁_diff : x₄ - x₁ = 1000 + 800 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_x₄_x₁_diff_l3470_347096


namespace NUMINAMATH_CALUDE_binomial_60_3_l3470_347033

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end NUMINAMATH_CALUDE_binomial_60_3_l3470_347033


namespace NUMINAMATH_CALUDE_rooks_arrangement_count_l3470_347098

/-- The number of squares on a chessboard -/
def chessboardSquares : ℕ := 64

/-- The number of squares threatened by a rook (excluding its own square) -/
def squaresThreatened : ℕ := 14

/-- The number of ways to arrange two rooks on a chessboard such that they cannot capture each other -/
def rooksArrangements : ℕ := chessboardSquares * (chessboardSquares - squaresThreatened - 1)

theorem rooks_arrangement_count :
  rooksArrangements = 3136 := by sorry

end NUMINAMATH_CALUDE_rooks_arrangement_count_l3470_347098


namespace NUMINAMATH_CALUDE_dodecagon_min_rotation_l3470_347066

/-- The minimum rotation angle for a regular dodecagon to coincide with itself -/
def min_rotation_angle_dodecagon : ℝ := 30

/-- Theorem: The minimum rotation angle for a regular dodecagon to coincide with itself is 30° -/
theorem dodecagon_min_rotation :
  min_rotation_angle_dodecagon = 30 := by sorry

end NUMINAMATH_CALUDE_dodecagon_min_rotation_l3470_347066


namespace NUMINAMATH_CALUDE_graph_x_squared_minus_y_squared_l3470_347022

/-- The graph of the equation x^2 - y^2 = 0 represents two intersecting lines -/
theorem graph_x_squared_minus_y_squared : 
  ∃ (L₁ L₂ : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ L₁ ∪ L₂ ↔ x^2 - y^2 = 0) ∧
    (L₁ ≠ L₂) ∧
    (∃ (p : ℝ × ℝ), p ∈ L₁ ∧ p ∈ L₂) :=
sorry

end NUMINAMATH_CALUDE_graph_x_squared_minus_y_squared_l3470_347022


namespace NUMINAMATH_CALUDE_exists_partition_count_2007_l3470_347023

/-- 
Given positive integers N and k, count_partitions N k returns the number of ways 
to write N as a sum of three integers a + b + c, where 1 ≤ a, b, c ≤ k.
-/
def count_partitions (N k : ℕ+) : ℕ := sorry

/-- 
There exist positive integers N and k such that the number of ways to write N 
as a sum of three integers a + b + c, where 1 ≤ a, b, c ≤ k, is equal to 2007.
-/
theorem exists_partition_count_2007 : 
  ∃ (N k : ℕ+), count_partitions N k = 2007 := by sorry

end NUMINAMATH_CALUDE_exists_partition_count_2007_l3470_347023


namespace NUMINAMATH_CALUDE_max_visible_cubes_l3470_347065

/-- The size of the cube's edge -/
def n : ℕ := 12

/-- The number of unit cubes on one face of the large cube -/
def face_cubes : ℕ := n^2

/-- The number of unit cubes along one edge of the large cube -/
def edge_cubes : ℕ := n

/-- The number of visible faces from a corner -/
def visible_faces : ℕ := 3

/-- The number of visible edges from a corner -/
def visible_edges : ℕ := 3

/-- The number of visible corners from a corner -/
def visible_corners : ℕ := 1

theorem max_visible_cubes :
  visible_faces * face_cubes - (visible_edges * edge_cubes - visible_corners) = 398 := by
  sorry

end NUMINAMATH_CALUDE_max_visible_cubes_l3470_347065


namespace NUMINAMATH_CALUDE_number_multiple_problem_l3470_347039

theorem number_multiple_problem (A B k : ℕ) 
  (sum_cond : A + B = 77)
  (bigger_cond : A = 42)
  (multiple_cond : 6 * B = k * A) :
  k = 5 := by sorry

end NUMINAMATH_CALUDE_number_multiple_problem_l3470_347039


namespace NUMINAMATH_CALUDE_no_intersection_line_circle_l3470_347091

/-- The line 3x + 4y = 12 and the circle x^2 + y^2 = 4 have no intersection points. -/
theorem no_intersection_line_circle : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → False :=
by sorry

end NUMINAMATH_CALUDE_no_intersection_line_circle_l3470_347091


namespace NUMINAMATH_CALUDE_problem_solution_l3470_347085

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (2 * x) + 2, Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem problem_solution (A : ℝ) (b c : ℝ) (h1 : 0 ≤ A ∧ A ≤ π/4) 
  (h2 : f A = 4) (h3 : b = 1) (h4 : 1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) :
  (∀ x ∈ Set.Icc 0 (π/4), f x ≤ 5 ∧ 4 ≤ f x) ∧ 
  c^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3470_347085


namespace NUMINAMATH_CALUDE_range_of_m_l3470_347076

-- Define the solution set A
def A (m : ℝ) : Set ℝ := {x : ℝ | |x^2 - 4*x + m| ≤ x + 4}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (0 ∈ A m) ∧ (2 ∉ A m) ↔ m ∈ Set.Icc (-4 : ℝ) (-2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_m_l3470_347076


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l3470_347073

theorem real_roots_of_polynomial (x : ℝ) :
  x^4 - 2*x^3 - x + 2 = 0 ↔ x = 1 ∨ x = 2 :=
sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l3470_347073


namespace NUMINAMATH_CALUDE_simplify_fraction_l3470_347099

theorem simplify_fraction : (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3470_347099


namespace NUMINAMATH_CALUDE_unused_cubes_for_5x5x5_with_9_tunnels_l3470_347069

/-- Represents a large cube made of small cubes with tunnels --/
structure LargeCube where
  size : Nat
  numTunnels : Nat

/-- Calculates the number of unused small cubes in a large cube with tunnels --/
def unusedCubes (c : LargeCube) : Nat :=
  c.size^3 - (c.numTunnels * c.size - 6)

/-- Theorem stating that for a 5x5x5 cube with 9 tunnels, 39 small cubes are unused --/
theorem unused_cubes_for_5x5x5_with_9_tunnels :
  let c : LargeCube := { size := 5, numTunnels := 9 }
  unusedCubes c = 39 := by
  sorry

#eval unusedCubes { size := 5, numTunnels := 9 }

end NUMINAMATH_CALUDE_unused_cubes_for_5x5x5_with_9_tunnels_l3470_347069


namespace NUMINAMATH_CALUDE_tan_eq_sin_cos_unique_solution_l3470_347090

open Real

theorem tan_eq_sin_cos_unique_solution :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ arccos 0.1 ∧ tan x = sin (cos x) := by
  sorry

end NUMINAMATH_CALUDE_tan_eq_sin_cos_unique_solution_l3470_347090


namespace NUMINAMATH_CALUDE_function_composition_equals_log_l3470_347032

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 1/2 * x - 1/2 else Real.log x

theorem function_composition_equals_log (a : ℝ) :
  (f (f a) = Real.log (f a)) ↔ a ∈ Set.Ici (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_function_composition_equals_log_l3470_347032


namespace NUMINAMATH_CALUDE_product_mod_500_l3470_347094

theorem product_mod_500 : (1493 * 1998) % 500 = 14 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_500_l3470_347094


namespace NUMINAMATH_CALUDE_trig_product_equals_one_l3470_347058

theorem trig_product_equals_one :
  let x : Real := 30 * π / 180  -- 30 degrees in radians
  let y : Real := 60 * π / 180  -- 60 degrees in radians
  (1 - 1 / Real.cos x) * (1 + 1 / Real.sin y) * (1 - 1 / Real.sin x) * (1 + 1 / Real.cos y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equals_one_l3470_347058


namespace NUMINAMATH_CALUDE_system_solution_l3470_347064

theorem system_solution : 
  ∃! (x y : ℚ), 3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_l3470_347064


namespace NUMINAMATH_CALUDE_banner_nail_distance_l3470_347063

theorem banner_nail_distance (banner_length : ℝ) (num_nails : ℕ) (end_distance : ℝ) :
  banner_length = 20 →
  num_nails = 7 →
  end_distance = 1 →
  (banner_length - 2 * end_distance) / (num_nails - 1 : ℝ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_banner_nail_distance_l3470_347063


namespace NUMINAMATH_CALUDE_fourth_number_proof_l3470_347078

theorem fourth_number_proof (x : ℝ) : 
  3 + 33 + 333 + x = 369.63 → x = 0.63 := by sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l3470_347078


namespace NUMINAMATH_CALUDE_product_evaluation_l3470_347045

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3470_347045


namespace NUMINAMATH_CALUDE_cheryl_leftover_material_l3470_347079

theorem cheryl_leftover_material :
  let material_type1 : ℚ := 2/9
  let material_type2 : ℚ := 1/8
  let total_bought : ℚ := material_type1 + material_type2
  let material_used : ℚ := 0.125
  let material_leftover : ℚ := total_bought - material_used
  material_leftover = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_leftover_material_l3470_347079


namespace NUMINAMATH_CALUDE_consecutive_products_not_end_2019_l3470_347043

theorem consecutive_products_not_end_2019 (n : ℤ) : 
  ∃ k : ℕ, ((n - 1) * (n + 1) + n * (n - 1) + n * (n + 1)) % 10000 ≠ 2019 + 10000 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_products_not_end_2019_l3470_347043
