import Mathlib

namespace linear_function_fixed_point_l2608_260872

theorem linear_function_fixed_point (k : ℝ) : (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end linear_function_fixed_point_l2608_260872


namespace final_state_is_blue_l2608_260804

/-- Represents the color of a sheep -/
inductive SheepColor
  | Red
  | Green
  | Blue

/-- Represents the state of the sheep population -/
structure SheepState where
  red : Nat
  green : Nat
  blue : Nat

/-- The color-changing rule for sheep meetings -/
def changeColor (c1 c2 : SheepColor) : SheepColor :=
  match c1, c2 with
  | SheepColor.Red, SheepColor.Green => SheepColor.Blue
  | SheepColor.Red, SheepColor.Blue => SheepColor.Green
  | SheepColor.Green, SheepColor.Blue => SheepColor.Red
  | SheepColor.Green, SheepColor.Red => SheepColor.Blue
  | SheepColor.Blue, SheepColor.Red => SheepColor.Green
  | SheepColor.Blue, SheepColor.Green => SheepColor.Red
  | _, _ => c1  -- If same color, no change

/-- The invariant property of the sheep population -/
def invariant (state : SheepState) : Bool :=
  (state.red - state.green) % 3 = 0 ∧
  (state.green - state.blue) % 3 = 2 ∧
  (state.blue - state.red) % 3 = 1

/-- The initial state of the sheep population -/
def initialState : SheepState :=
  { red := 18, green := 15, blue := 22 }

/-- Theorem: The only possible final state is all sheep being blue -/
theorem final_state_is_blue (state : SheepState) :
  invariant initialState →
  invariant state →
  (state.red + state.green + state.blue = initialState.red + initialState.green + initialState.blue) →
  (state.red = 0 ∧ state.green = 0 ∧ state.blue = 55) :=
sorry

end final_state_is_blue_l2608_260804


namespace parallelogram_area_10_20_l2608_260826

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 10 cm and height 20 cm is 200 square centimeters -/
theorem parallelogram_area_10_20 :
  parallelogram_area 10 20 = 200 := by
  sorry

end parallelogram_area_10_20_l2608_260826


namespace min_sum_of_reciprocals_l2608_260885

theorem min_sum_of_reciprocals (x y z : ℕ+) (h : (1 : ℚ) / x + 4 / y + 9 / z = 1) :
  36 ≤ (x : ℚ) + y + z :=
sorry

end min_sum_of_reciprocals_l2608_260885


namespace angle_measure_in_special_triangle_l2608_260888

/-- Given a triangle PQR where ∠P is thrice ∠R and ∠Q is equal to ∠R, 
    the measure of ∠Q is 36°. -/
theorem angle_measure_in_special_triangle (P Q R : ℝ) : 
  P + Q + R = 180 →  -- sum of angles in a triangle
  P = 3 * R →        -- ∠P is thrice ∠R
  Q = R →            -- ∠Q is equal to ∠R
  Q = 36 :=          -- measure of ∠Q is 36°
by sorry

end angle_measure_in_special_triangle_l2608_260888


namespace weight_sum_proof_l2608_260898

/-- Given the weights of four people in pairs, prove that the sum of two specific people's weights can be determined. -/
theorem weight_sum_proof (e f g h : ℝ) 
  (ef_sum : e + f = 280)
  (fg_sum : f + g = 230)
  (gh_sum : g + h = 260) :
  e + h = 310 := by sorry

end weight_sum_proof_l2608_260898


namespace subset_implies_a_zero_l2608_260894

theorem subset_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {1, 2, a}
  let B : Set ℝ := {2, a^2 + 1}
  B ⊆ A → a = 0 := by
sorry

end subset_implies_a_zero_l2608_260894


namespace hoseok_marbles_l2608_260891

theorem hoseok_marbles : ∃ x : ℕ+, x * 80 + 260 = x * 100 ∧ x = 13 := by
  sorry

end hoseok_marbles_l2608_260891


namespace problem_solution_l2608_260853

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / (x^2 + 3)

theorem problem_solution (a : ℝ) :
  (∀ x, deriv (f a) x = (a * (x^2 + 3) - a * x * (2 * x)) / (x^2 + 3)^2) →
  deriv (f a) 1 = 1/2 →
  a = 4 := by sorry

end problem_solution_l2608_260853


namespace shaded_area_9x7_grid_l2608_260800

/-- Represents a grid with 2x2 squares, where alternate squares are split and shaded -/
structure ShadedGrid :=
  (width : ℕ)
  (height : ℕ)
  (square_size : ℕ)

/-- Calculates the area of the shaded region in the grid -/
def shaded_area (grid : ShadedGrid) : ℕ :=
  let horizontal_squares := grid.width / grid.square_size
  let vertical_squares := grid.height / grid.square_size
  let total_squares := horizontal_squares * vertical_squares
  let shaded_triangle_area := (grid.square_size * grid.square_size) / 2
  total_squares * shaded_triangle_area

/-- Theorem: The shaded area in a 9x7 grid with 2x2 squares is 24 square units -/
theorem shaded_area_9x7_grid :
  let grid : ShadedGrid := ⟨9, 7, 2⟩
  shaded_area grid = 24 := by
  sorry

end shaded_area_9x7_grid_l2608_260800


namespace work_completion_time_l2608_260817

theorem work_completion_time (x y : ℕ) (h1 : x = 14) 
  (h2 : (5 : ℝ) * ((1 : ℝ) / x + (1 : ℝ) / y) = 0.6071428571428572) : y = 20 := by
  sorry

end work_completion_time_l2608_260817


namespace inverse_g_at_negative_43_l2608_260833

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x^3 - 3

-- State the theorem
theorem inverse_g_at_negative_43 : g⁻¹ (-43) = -2 := by sorry

end inverse_g_at_negative_43_l2608_260833


namespace stating_sum_of_nth_group_is_cube_l2608_260863

/-- 
Given a grouping of consecutive odd numbers as follows:
1; (3,5); (7,9,11); (13, 15, 17, 19); ...
This function represents the sum of the numbers in the n-th group.
-/
def sumOfNthGroup (n : ℕ) : ℕ :=
  n^3

/-- 
Theorem stating that the sum of the numbers in the n-th group
of the described sequence is equal to n^3.
-/
theorem sum_of_nth_group_is_cube (n : ℕ) :
  sumOfNthGroup n = n^3 := by
  sorry

end stating_sum_of_nth_group_is_cube_l2608_260863


namespace arithmetic_sequence_product_l2608_260858

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n, b (n + 1) > b n) →  -- increasing sequence
  (∃ d : ℤ, ∀ n, b (n + 1) - b n = d) →  -- arithmetic sequence
  b 4 * b 5 = 18 →
  b 3 * b 6 = -80 := by
sorry

end arithmetic_sequence_product_l2608_260858


namespace empty_proper_subset_implies_nonempty_l2608_260882

theorem empty_proper_subset_implies_nonempty (A : Set α) :
  ∅ ⊂ A → A ≠ ∅ := by
  sorry

end empty_proper_subset_implies_nonempty_l2608_260882


namespace schedule_arrangements_l2608_260896

/-- Represents the number of subjects to be scheduled -/
def num_subjects : ℕ := 6

/-- Represents the number of periods in a day -/
def num_periods : ℕ := 6

/-- Calculates the number of arrangements for scheduling subjects with given constraints -/
def num_arrangements : ℕ :=
  5 * 4 * (Finset.range 4).prod (λ i => i + 1)

/-- Theorem stating the number of different arrangements -/
theorem schedule_arrangements :
  num_arrangements = 480 := by sorry

end schedule_arrangements_l2608_260896


namespace absolute_value_and_quadratic_equivalence_l2608_260801

theorem absolute_value_and_quadratic_equivalence :
  ∀ b c : ℝ,
  (∀ x : ℝ, |x - 3| = 5 ↔ x^2 + b*x + c = 0) →
  b = -6 ∧ c = -16 := by
sorry

end absolute_value_and_quadratic_equivalence_l2608_260801


namespace sin_pi_sufficient_not_necessary_l2608_260854

open Real

theorem sin_pi_sufficient_not_necessary :
  (∀ x : ℝ, x = π → sin x = 0) ∧
  (∃ x : ℝ, x ≠ π ∧ sin x = 0) := by
  sorry

end sin_pi_sufficient_not_necessary_l2608_260854


namespace tangent_line_y_intercept_l2608_260844

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Predicate to check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- The main theorem -/
theorem tangent_line_y_intercept (c1 c2 : Circle) (l : Line) :
  c1.center = (3, 0) →
  c1.radius = 3 →
  c2.center = (8, 0) →
  c2.radius = 2 →
  is_tangent l c1 →
  is_tangent l c2 →
  l.y_intercept = 15 * Real.sqrt 26 / 26 :=
sorry

end tangent_line_y_intercept_l2608_260844


namespace circle_and_tangents_l2608_260812

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line y = 2x
def Line (m : ℝ) := {p : ℝ × ℝ | p.2 = m * p.1}

-- Define the point P
def P : ℝ × ℝ := (-2, 2)

-- Theorem statement
theorem circle_and_tangents 
  (C : ℝ × ℝ) -- Center of the circle
  (h1 : C ∈ Line 2) -- Center lies on y = 2x
  (h2 : (0, 0) ∈ Circle C (Real.sqrt 5)) -- Circle passes through (0,0)
  (h3 : (2, 0) ∈ Circle C (Real.sqrt 5)) -- Circle passes through (2,0)
  : 
  -- 1. The circle equation
  Circle C (Real.sqrt 5) = {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 5} ∧
  -- 2. The tangent line equations
  ∃ (k₁ k₂ : ℝ), 
    k₁ = Real.sqrt 5 / 2 ∧ 
    k₂ = -Real.sqrt 5 / 2 ∧
    (∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.2 - 2 = k₁ * (p.1 + 2)} → 
      ((x, y) ∈ Circle C (Real.sqrt 5) → (x, y) = P)) ∧
    (∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.2 - 2 = k₂ * (p.1 + 2)} → 
      ((x, y) ∈ Circle C (Real.sqrt 5) → (x, y) = P)) :=
by sorry

end circle_and_tangents_l2608_260812


namespace binary_to_base4_conversion_l2608_260821

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List (Fin 4) :=
      if m = 0 then [] else (m % 4) :: aux (m / 4)
    aux n |>.reverse

def binary : List Bool := [true, true, false, true, false, false, true, false]

theorem binary_to_base4_conversion :
  decimal_to_base4 (binary_to_decimal binary) = [3, 1, 0, 2] := by
  sorry

end binary_to_base4_conversion_l2608_260821


namespace square_and_rectangle_area_sum_l2608_260870

/-- Given a square and a rectangle satisfying certain conditions, prove that the sum of their areas is approximately 118 square units. -/
theorem square_and_rectangle_area_sum :
  ∀ (s w : ℝ),
    s > 0 →
    w > 0 →
    s^2 + 2*w^2 = 130 →
    4*s - 2*(w + 2*w) = 20 →
    abs (s^2 + 2*w^2 - 118) < 1 :=
by
  sorry

#check square_and_rectangle_area_sum

end square_and_rectangle_area_sum_l2608_260870


namespace smallest_integer_with_remainder_one_l2608_260859

theorem smallest_integer_with_remainder_one (k : ℕ) : k = 400 ↔ 
  (k > 1) ∧ 
  (k % 19 = 1) ∧ 
  (k % 7 = 1) ∧ 
  (k % 3 = 1) ∧ 
  (∀ m : ℕ, m > 1 → m % 19 = 1 → m % 7 = 1 → m % 3 = 1 → k ≤ m) := by
sorry

end smallest_integer_with_remainder_one_l2608_260859


namespace least_valid_number_l2608_260852

def is_valid (n : ℕ) : Prop :=
  n > 1 ∧
  n % 4 = 3 ∧
  n % 5 = 3 ∧
  n % 7 = 3 ∧
  n % 10 = 3 ∧
  n % 11 = 3

theorem least_valid_number : 
  is_valid 1543 ∧ ∀ m : ℕ, m < 1543 → ¬(is_valid m) :=
sorry

end least_valid_number_l2608_260852


namespace apple_purchase_cost_l2608_260847

/-- The cost of apples in dollars per 7 pounds -/
def apple_cost : ℚ := 5

/-- The rate of apples in pounds per cost unit -/
def apple_rate : ℚ := 7

/-- The amount of apples we want to buy in pounds -/
def apple_amount : ℚ := 21

/-- Theorem: The cost of 21 pounds of apples is $15 -/
theorem apple_purchase_cost : (apple_amount / apple_rate) * apple_cost = 15 := by
  sorry

end apple_purchase_cost_l2608_260847


namespace largest_inscribed_square_side_length_l2608_260841

/-- The side length of the largest inscribed square in a specific configuration -/
theorem largest_inscribed_square_side_length :
  ∃ (large_square_side : ℝ) (triangle_side : ℝ) (inscribed_square_side : ℝ),
    large_square_side = 12 ∧
    triangle_side = 4 * Real.sqrt 6 ∧
    inscribed_square_side = 6 - Real.sqrt 6 ∧
    2 * inscribed_square_side * Real.sqrt 2 + triangle_side = large_square_side * Real.sqrt 2 :=
by sorry

end largest_inscribed_square_side_length_l2608_260841


namespace complement_of_A_l2608_260856

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x + 2 > 4}

theorem complement_of_A : Set.compl A = {x : ℝ | x ≤ 2} := by sorry

end complement_of_A_l2608_260856


namespace wrong_divisor_problem_l2608_260820

theorem wrong_divisor_problem (correct_divisor correct_answer student_answer : ℕ) 
  (h1 : correct_divisor = 36)
  (h2 : correct_answer = 58)
  (h3 : student_answer = 24) :
  ∃ (wrong_divisor : ℕ), 
    (correct_divisor * correct_answer) / wrong_divisor = student_answer ∧ 
    wrong_divisor = 87 := by
  sorry

end wrong_divisor_problem_l2608_260820


namespace aaron_scarves_count_l2608_260846

/-- The number of scarves Aaron made -/
def aaronScarves : ℕ := 10

/-- The number of sweaters Aaron made -/
def aaronSweaters : ℕ := 5

/-- The number of sweaters Enid made -/
def enidSweaters : ℕ := 8

/-- The number of balls of wool used for one scarf -/
def woolPerScarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def woolPerSweater : ℕ := 4

/-- The total number of balls of wool used -/
def totalWool : ℕ := 82

theorem aaron_scarves_count : 
  woolPerScarf * aaronScarves + 
  woolPerSweater * (aaronSweaters + enidSweaters) = 
  totalWool := by sorry

end aaron_scarves_count_l2608_260846


namespace m_range_l2608_260879

def y₁ (m x : ℝ) : ℝ := m * (x - 2 * m) * (x + m + 2)
def y₂ (x : ℝ) : ℝ := x - 1

theorem m_range :
  (∀ m : ℝ,
    (∀ x : ℝ, y₁ m x < 0 ∨ y₂ x < 0) ∧
    (∃ x : ℝ, x < -3 ∧ y₁ m x * y₂ x < 0)) ↔
  (∀ m : ℝ, -4 < m ∧ m < -3/2) :=
sorry

end m_range_l2608_260879


namespace car_cost_share_l2608_260860

/-- Given a car that costs $2,100 and is used for 7 days a week, with one person using it for 4 days,
    prove that the other person's share of the cost is $900. -/
theorem car_cost_share (total_cost : ℕ) (total_days : ℕ) (days_used_by_first : ℕ) :
  total_cost = 2100 →
  total_days = 7 →
  days_used_by_first = 4 →
  (total_cost * (total_days - days_used_by_first) / total_days : ℚ) = 900 := by
  sorry

#check car_cost_share

end car_cost_share_l2608_260860


namespace min_value_x_plus_y_l2608_260855

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y = 2 * x * y) :
  x + y ≥ 9 / 2 :=
sorry

end min_value_x_plus_y_l2608_260855


namespace largest_product_of_digits_l2608_260842

/-- A function that returns the product of digits of a natural number -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is equal to the product of its digits -/
def isProductOfDigits (n : ℕ) : Prop :=
  n = productOfDigits n

/-- Theorem stating that 9 is the largest natural number equal to the product of its digits -/
theorem largest_product_of_digits : 
  ∀ n : ℕ, isProductOfDigits n → n ≤ 9 := by sorry

end largest_product_of_digits_l2608_260842


namespace fraction_is_standard_notation_l2608_260803

-- Define what it means for an expression to be in standard algebraic notation
def is_standard_algebraic_notation (expr : ℚ) : Prop :=
  ∃ (n m : ℤ), m ≠ 0 ∧ expr = n / m

-- Define our fraction
def our_fraction (n m : ℤ) : ℚ := n / m

-- Theorem statement
theorem fraction_is_standard_notation (n m : ℤ) (h : m ≠ 0) :
  is_standard_algebraic_notation (our_fraction n m) :=
sorry

end fraction_is_standard_notation_l2608_260803


namespace triangular_cross_section_solids_l2608_260887

/-- Enumeration of geometric solids -/
inductive GeometricSolid
  | Cube
  | Cylinder
  | Cone
  | RegularTriangularPrism

/-- Predicate to determine if a geometric solid can have a triangular cross-section -/
def has_triangular_cross_section (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cube => True
  | GeometricSolid.Cylinder => False
  | GeometricSolid.Cone => True
  | GeometricSolid.RegularTriangularPrism => True

/-- Theorem stating which geometric solids can have a triangular cross-section -/
theorem triangular_cross_section_solids :
  ∀ (solid : GeometricSolid),
    has_triangular_cross_section solid ↔
      (solid = GeometricSolid.Cube ∨
       solid = GeometricSolid.Cone ∨
       solid = GeometricSolid.RegularTriangularPrism) :=
by sorry

end triangular_cross_section_solids_l2608_260887


namespace power_of_product_equals_power_l2608_260815

theorem power_of_product_equals_power (n : ℕ) : 3^12 * 3^18 = 243^6 := by sorry

end power_of_product_equals_power_l2608_260815


namespace solution_set_inequality_l2608_260836

theorem solution_set_inequality (x : ℝ) : -x^2 + 2*x > 0 ↔ 0 < x ∧ x < 2 := by
  sorry

end solution_set_inequality_l2608_260836


namespace solution_set_inequalities_l2608_260827

theorem solution_set_inequalities (a b : ℝ) 
  (h : ∃ x, x > a ∧ x < b) : 
  {x : ℝ | x < 1 - a ∧ x < 1 - b} = {x : ℝ | x < 1 - b} :=
by sorry

end solution_set_inequalities_l2608_260827


namespace fraction_multiplication_addition_l2608_260866

theorem fraction_multiplication_addition : 
  (1 / 3 : ℚ) * (1 / 4 : ℚ) * (1 / 5 : ℚ) + (1 / 2 : ℚ) = 31 / 60 := by
  sorry

end fraction_multiplication_addition_l2608_260866


namespace circle_center_distance_l2608_260811

theorem circle_center_distance (x y : ℝ) : 
  x^2 + y^2 = 6*x + 8*y + 3 →
  Real.sqrt ((10 - x)^2 + (5 - y)^2) = 5 * Real.sqrt 2 := by
  sorry

end circle_center_distance_l2608_260811


namespace meat_cost_per_pound_l2608_260822

/-- The cost of meat per pound given the total cost, rice quantity, rice price, and meat quantity -/
theorem meat_cost_per_pound 
  (total_cost : ℝ)
  (rice_quantity : ℝ)
  (rice_price_per_kg : ℝ)
  (meat_quantity : ℝ)
  (h1 : total_cost = 25)
  (h2 : rice_quantity = 5)
  (h3 : rice_price_per_kg = 2)
  (h4 : meat_quantity = 3)
  : (total_cost - rice_quantity * rice_price_per_kg) / meat_quantity = 5 := by
  sorry

end meat_cost_per_pound_l2608_260822


namespace total_selected_in_survey_l2608_260877

/-- The number of residents aged 21 to 35 -/
def residents_21_35 : ℕ := 840

/-- The number of residents aged 36 to 50 -/
def residents_36_50 : ℕ := 700

/-- The number of residents aged 51 to 65 -/
def residents_51_65 : ℕ := 560

/-- The number of people selected from the 36 to 50 age group -/
def selected_36_50 : ℕ := 100

/-- The total number of residents -/
def total_residents : ℕ := residents_21_35 + residents_36_50 + residents_51_65

/-- The theorem stating the total number of people selected in the survey -/
theorem total_selected_in_survey : 
  (selected_36_50 : ℚ) * (total_residents : ℚ) / (residents_36_50 : ℚ) = 300 := by
  sorry

end total_selected_in_survey_l2608_260877


namespace stratified_sampling_result_count_l2608_260816

def junior_population : ℕ := 400
def senior_population : ℕ := 200
def total_sample_size : ℕ := 60

def stratified_proportional_sample_count (n1 n2 k : ℕ) : ℕ :=
  Nat.choose n1 ((k * n1) / (n1 + n2)) * Nat.choose n2 ((k * n2) / (n1 + n2))

theorem stratified_sampling_result_count :
  stratified_proportional_sample_count junior_population senior_population total_sample_size =
  Nat.choose junior_population 40 * Nat.choose senior_population 20 := by
  sorry

end stratified_sampling_result_count_l2608_260816


namespace max_sphere_radius_squared_l2608_260864

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones and a sphere -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ
  sphereRadius : ℝ

/-- The specific configuration described in the problem -/
def problemConfig : ConeConfiguration :=
  { cone1 := { baseRadius := 4, height := 10 }
  , cone2 := { baseRadius := 4, height := 10 }
  , intersectionDistance := 4
  , sphereRadius := 0  -- To be determined
  }

/-- The theorem statement -/
theorem max_sphere_radius_squared (config : ConeConfiguration) :
  config = problemConfig →
  ∃ (r : ℝ), r > 0 ∧ 
    (∀ (s : ℝ), s > 0 → 
      (∃ (c : ConeConfiguration), c.cone1 = config.cone1 ∧ 
                                  c.cone2 = config.cone2 ∧ 
                                  c.intersectionDistance = config.intersectionDistance ∧
                                  c.sphereRadius = s) →
      s^2 ≤ r^2) ∧
    r^2 = 8704 / 29 :=
by sorry

end max_sphere_radius_squared_l2608_260864


namespace sunday_production_l2608_260840

/-- The number of toys produced on a given day of the week -/
def toysProduced (day : Nat) : Nat :=
  2500 + 25 * day

/-- The number of days in a week -/
def daysInWeek : Nat := 7

/-- Theorem stating that the number of toys produced on Sunday (day 6) is 2650 -/
theorem sunday_production :
  toysProduced (daysInWeek - 1) = 2650 := by
  sorry


end sunday_production_l2608_260840


namespace distance_to_center_of_gravity_l2608_260818

/-- Regular hexagon with side length a -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Square cut out from the hexagon -/
structure CutOutSquare (hex : RegularHexagon) where
  diagonal : ℝ
  diagonal_eq_side : diagonal = hex.side_length

/-- Remaining plate after cutting out the square -/
structure RemainingPlate (hex : RegularHexagon) (square : CutOutSquare hex) where

/-- Center of gravity of the remaining plate -/
noncomputable def center_of_gravity (plate : RemainingPlate hex square) : ℝ × ℝ := sorry

/-- Distance from the hexagon center to the center of gravity -/
noncomputable def distance_to_center (plate : RemainingPlate hex square) : ℝ :=
  let cog := center_of_gravity plate
  Real.sqrt ((cog.1 ^ 2) + (cog.2 ^ 2))

/-- Main theorem: The distance from the hexagon center to the center of gravity of the remaining plate -/
theorem distance_to_center_of_gravity 
  (hex : RegularHexagon) 
  (square : CutOutSquare hex) 
  (plate : RemainingPlate hex square) : 
  distance_to_center plate = (3 * Real.sqrt 3 + 1) / 52 * hex.side_length := by
  sorry

end distance_to_center_of_gravity_l2608_260818


namespace simplify_expressions_l2608_260883

theorem simplify_expressions :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (Real.sqrt (1 / 3) + Real.sqrt 27 * Real.sqrt 9 = 28 * Real.sqrt 3 / 3) ∧
    (Real.sqrt 32 - 3 * Real.sqrt (1 / 2) + Real.sqrt (1 / 8) = 11 * Real.sqrt 2 / 4)) :=
by sorry

end simplify_expressions_l2608_260883


namespace pat_candy_count_l2608_260830

/-- The number of cookies Pat has -/
def num_cookies : ℕ := 42

/-- The number of brownies Pat has -/
def num_brownies : ℕ := 21

/-- The number of people in Pat's family -/
def num_people : ℕ := 7

/-- The number of dessert pieces each person gets -/
def dessert_per_person : ℕ := 18

/-- The number of candy pieces Pat has -/
def num_candy : ℕ := num_people * dessert_per_person - (num_cookies + num_brownies)

theorem pat_candy_count : num_candy = 63 := by
  sorry

end pat_candy_count_l2608_260830


namespace interior_angle_non_integer_count_l2608_260810

theorem interior_angle_non_integer_count :
  ∃! (n : ℕ), 3 ≤ n ∧ n ≤ 10 ∧ ¬(∃ (k : ℕ), (180 * (n - 2)) / n = k) :=
by sorry

end interior_angle_non_integer_count_l2608_260810


namespace soccer_substitution_ratio_l2608_260849

/-- Soccer team substitution ratio theorem -/
theorem soccer_substitution_ratio 
  (total_players : ℕ) 
  (starters : ℕ) 
  (first_half_subs : ℕ) 
  (non_players : ℕ) 
  (h1 : total_players = 24) 
  (h2 : starters = 11) 
  (h3 : first_half_subs = 2) 
  (h4 : non_players = 7) : 
  (total_players - non_players - (starters + first_half_subs)) / first_half_subs = 2 := by
sorry

end soccer_substitution_ratio_l2608_260849


namespace probability_not_raining_l2608_260884

theorem probability_not_raining (p : ℚ) (h : p = 4/9) : 1 - p = 5/9 := by
  sorry

end probability_not_raining_l2608_260884


namespace spring_sales_five_million_l2608_260893

/-- Represents the annual pizza sales of a restaurant in millions -/
def annual_sales : ℝ := 20

/-- Represents the winter pizza sales of the restaurant in millions -/
def winter_sales : ℝ := 4

/-- Represents the percentage of annual sales that occur in winter -/
def winter_percentage : ℝ := 0.20

/-- Represents the percentage of annual sales that occur in summer -/
def summer_percentage : ℝ := 0.30

/-- Represents the percentage of annual sales that occur in fall -/
def fall_percentage : ℝ := 0.25

/-- Theorem stating that spring sales are 5 million pizzas -/
theorem spring_sales_five_million :
  winter_sales = winter_percentage * annual_sales →
  ∃ (spring_percentage : ℝ),
    spring_percentage + winter_percentage + summer_percentage + fall_percentage = 1 ∧
    spring_percentage * annual_sales = 5 := by
  sorry

end spring_sales_five_million_l2608_260893


namespace garrison_provision_problem_l2608_260837

/-- Calculates the initial number of days provisions would last for a garrison --/
def initial_provision_days (initial_men : ℕ) (reinforcement_men : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) : ℕ :=
  (initial_men * days_before_reinforcement + (initial_men + reinforcement_men) * days_after_reinforcement) / initial_men

theorem garrison_provision_problem :
  initial_provision_days 1850 1110 12 10 = 28 := by
  sorry

end garrison_provision_problem_l2608_260837


namespace business_value_l2608_260881

/-- Given a man who owns 2/3 of a business and sells 3/4 of his shares for 45,000 Rs,
    prove that the value of the entire business is 90,000 Rs. -/
theorem business_value (man_share : ℚ) (sold_portion : ℚ) (sold_value : ℕ) :
  man_share = 2/3 →
  sold_portion = 3/4 →
  sold_value = 45000 →
  ∃ (total_value : ℕ), total_value = 90000 ∧
    (total_value : ℚ) = sold_value / (man_share * sold_portion) :=
by sorry

end business_value_l2608_260881


namespace crypto_encoding_l2608_260843

/-- Represents the encoding of digits in the cryptographic system -/
inductive Digit
| A
| B
| C
| D

/-- Converts a Digit to its corresponding base-4 value -/
def digit_to_base4 : Digit → Nat
| Digit.A => 3
| Digit.B => 1
| Digit.C => 0
| Digit.D => 2

/-- Converts a three-digit code to its base-10 value -/
def code_to_base10 (d₁ d₂ d₃ : Digit) : Nat :=
  16 * (digit_to_base4 d₁) + 4 * (digit_to_base4 d₂) + (digit_to_base4 d₃)

/-- The main theorem stating the result of the cryptographic encoding -/
theorem crypto_encoding :
  code_to_base10 Digit.B Digit.C Digit.D + 1 = code_to_base10 Digit.B Digit.D Digit.A ∧
  code_to_base10 Digit.B Digit.D Digit.A + 1 = code_to_base10 Digit.B Digit.C Digit.A →
  code_to_base10 Digit.D Digit.A Digit.C = 44 :=
by sorry

end crypto_encoding_l2608_260843


namespace inscribed_circle_theorem_l2608_260845

/-- Given a right-angled triangle with catheti of lengths a and b, and a circle
    with radius r inscribed such that it touches both catheti and has its center
    on the hypotenuse, prove that 1/a + 1/b = 1/r. -/
theorem inscribed_circle_theorem (a b r : ℝ) 
    (ha : a > 0) (hb : b > 0) (hr : r > 0)
    (h_right_triangle : ∃ c, a^2 + b^2 = c^2)
    (h_circle_inscribed : ∃ x y, x^2 + y^2 = r^2 ∧ x + y = r ∧ x < a ∧ y < b) :
    1/a + 1/b = 1/r := by
  sorry

end inscribed_circle_theorem_l2608_260845


namespace fraction_ratio_equality_l2608_260814

theorem fraction_ratio_equality : ∃ (X Y : ℚ), (X / Y) / (2 / 6) = (1 / 2) / (1 / 2) → X / Y = 1 / 3 := by
  sorry

end fraction_ratio_equality_l2608_260814


namespace dry_grapes_weight_l2608_260835

-- Define the parameters
def fresh_water_content : Real := 0.90
def dried_water_content : Real := 0.20
def fresh_grapes_weight : Real := 5

-- Define the theorem
theorem dry_grapes_weight :
  let non_water_content := (1 - fresh_water_content) * fresh_grapes_weight
  let dry_grapes_weight := non_water_content / (1 - dried_water_content)
  dry_grapes_weight = 0.625 := by
  sorry

end dry_grapes_weight_l2608_260835


namespace lakeisha_lawn_size_l2608_260823

/-- The size of each lawn LaKeisha has already mowed -/
def lawn_size : ℝ := sorry

/-- LaKeisha's charge per square foot -/
def charge_per_sqft : ℝ := 0.10

/-- Cost of the book set -/
def book_cost : ℝ := 150

/-- Number of lawns already mowed -/
def lawns_mowed : ℕ := 3

/-- Additional square feet to mow -/
def additional_sqft : ℝ := 600

theorem lakeisha_lawn_size :
  lawn_size = 300 ∧
  charge_per_sqft * (lawns_mowed * lawn_size + additional_sqft) = book_cost :=
sorry

end lakeisha_lawn_size_l2608_260823


namespace last_three_digits_of_7_to_80_l2608_260834

theorem last_three_digits_of_7_to_80 : 7^80 ≡ 961 [MOD 1000] := by
  sorry

end last_three_digits_of_7_to_80_l2608_260834


namespace quadratic_inequality_solution_l2608_260829

theorem quadratic_inequality_solution (x : ℝ) : 3 * x^2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := by
  sorry

end quadratic_inequality_solution_l2608_260829


namespace tan_sum_from_sin_cos_sum_l2608_260848

theorem tan_sum_from_sin_cos_sum (α β : Real) 
  (h1 : Real.sin α + Real.sin β = (4/5) * Real.sqrt 2)
  (h2 : Real.cos α + Real.cos β = (4/5) * Real.sqrt 3) :
  Real.tan α + Real.tan β = 2 * Real.sqrt 6 := by
  sorry

end tan_sum_from_sin_cos_sum_l2608_260848


namespace problem_statement_l2608_260839

theorem problem_statement (n : ℤ) (a : ℝ) : 
  (6 * 11 * n > 0) → (a^(2*n) = 5) → (2 * a^(6*n) - 4 = 246) := by
  sorry

end problem_statement_l2608_260839


namespace closest_integer_to_sqrt_29_l2608_260838

theorem closest_integer_to_sqrt_29 :
  ∃ (n : ℤ), ∀ (m : ℤ), |n - Real.sqrt 29| ≤ |m - Real.sqrt 29| ∧ n = 5 :=
by
  sorry

end closest_integer_to_sqrt_29_l2608_260838


namespace daily_increase_calculation_l2608_260875

def squats_sequence (initial : ℕ) (increase : ℕ) (day : ℕ) : ℕ :=
  initial + (day - 1) * increase

theorem daily_increase_calculation (initial : ℕ) (increase : ℕ) :
  initial = 30 →
  squats_sequence initial increase 4 = 45 →
  increase = 5 := by
  sorry

end daily_increase_calculation_l2608_260875


namespace football_match_problem_l2608_260807

/-- Represents a football team's match statistics -/
structure TeamStats :=
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)

/-- Calculate the total matches played by a team -/
def total_matches (team : TeamStats) : ℕ :=
  team.wins + team.draws + team.losses

/-- The football match problem -/
theorem football_match_problem 
  (home : TeamStats)
  (rival : TeamStats)
  (h1 : home.wins = 3)
  (h2 : home.draws = 4)
  (h3 : home.losses = 0)
  (h4 : rival.wins = 2 * home.wins)
  (h5 : rival.draws = 4)
  (h6 : rival.losses = 0) :
  total_matches home + total_matches rival = 17 :=
sorry

end football_match_problem_l2608_260807


namespace probability_one_boy_one_girl_l2608_260824

/-- The probability of selecting exactly one boy and one girl when randomly choosing 2 people from 2 boys and 2 girls -/
theorem probability_one_boy_one_girl (num_boys num_girls : ℕ) (h1 : num_boys = 2) (h2 : num_girls = 2) :
  let total_combinations := num_boys * num_girls + (num_boys.choose 2) + (num_girls.choose 2)
  let favorable_combinations := num_boys * num_girls
  (favorable_combinations : ℚ) / total_combinations = 2/3 := by
sorry

end probability_one_boy_one_girl_l2608_260824


namespace pauls_remaining_books_l2608_260890

/-- Calculates the number of books remaining after a sale -/
def books_remaining (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

/-- Theorem: Paul's remaining books after the sale -/
theorem pauls_remaining_books :
  let initial_books : ℕ := 115
  let books_sold : ℕ := 78
  books_remaining initial_books books_sold = 37 := by
  sorry

end pauls_remaining_books_l2608_260890


namespace estimate_overweight_students_l2608_260876

def sample_size : ℕ := 100
def total_population : ℕ := 2000
def frequencies : List ℝ := [0.04, 0.035, 0.015]

theorem estimate_overweight_students :
  let total_frequency := (List.sum frequencies) * (total_population / sample_size)
  let estimated_students := total_population * total_frequency
  estimated_students = 360 := by sorry

end estimate_overweight_students_l2608_260876


namespace angle_of_inclination_l2608_260813

/-- The angle of inclination of the line x + √3 y - 5 = 0 is 150° -/
theorem angle_of_inclination (x y : ℝ) : 
  x + Real.sqrt 3 * y - 5 = 0 → 
  ∃ θ : ℝ, θ = 150 * π / 180 ∧ 
    Real.tan θ = -(1 / Real.sqrt 3) ∧
    0 ≤ θ ∧ θ < π := by
  sorry

end angle_of_inclination_l2608_260813


namespace divisible_by_120_l2608_260828

theorem divisible_by_120 (n : ℤ) : ∃ k : ℤ, n^6 + 2*n^5 - n^2 - 2*n = 120*k := by
  sorry

end divisible_by_120_l2608_260828


namespace greatest_distance_between_circle_centers_l2608_260869

theorem greatest_distance_between_circle_centers 
  (circle_diameter : ℝ) 
  (rectangle_length : ℝ) 
  (rectangle_width : ℝ) 
  (h_diameter : circle_diameter = 8)
  (h_length : rectangle_length = 20)
  (h_width : rectangle_width = 16)
  (h_tangent : circle_diameter ≤ rectangle_width) :
  let circle_radius := circle_diameter / 2
  let horizontal_distance := 2 * circle_radius
  let vertical_distance := rectangle_width
  ∃ (max_distance : ℝ), 
    max_distance = (horizontal_distance^2 + vertical_distance^2).sqrt ∧
    max_distance = 8 * Real.sqrt 5 :=
by sorry

end greatest_distance_between_circle_centers_l2608_260869


namespace money_division_l2608_260805

theorem money_division (p q r : ℕ) (total : ℝ) : 
  p + q + r = 22 →  -- Ratio sum: 3 + 7 + 12 = 22
  (7 / 22 * total - 3 / 22 * total = 4000) →
  (12 / 22 * total - 7 / 22 * total = 5000) :=
by
  sorry

end money_division_l2608_260805


namespace equation_solutions_l2608_260868

theorem equation_solutions :
  let f : ℝ → ℝ := λ x => x * (5 * x + 2) - 6 * (5 * x + 2)
  (f 6 = 0 ∧ f (-2/5) = 0) ∧ 
  ∀ x : ℝ, f x = 0 → x = 6 ∨ x = -2/5 := by sorry

end equation_solutions_l2608_260868


namespace negation_of_r_not_p_is_true_p_or_r_is_false_p_and_q_is_false_l2608_260806

-- Define the propositions
def p : Prop := ∃ x₀ : ℝ, x₀ > -2 ∧ 6 + |x₀| = 5

def q : Prop := ∀ x : ℝ, x < 0 → x^2 + 4/x^2 ≥ 4

def r : Prop := ∀ x y : ℝ, |x| + |y| ≤ 1 → |y| / (|x| + 2) ≤ 1/2

-- Theorem statements
theorem negation_of_r : 
  (¬r) ↔ (∃ x y : ℝ, |x| + |y| > 1 ∧ |y| / (|x| + 2) > 1/2) :=
sorry

theorem not_p_is_true : ¬p :=
sorry

theorem p_or_r_is_false : ¬(p ∨ r) :=
sorry

theorem p_and_q_is_false : ¬(p ∧ q) :=
sorry

end negation_of_r_not_p_is_true_p_or_r_is_false_p_and_q_is_false_l2608_260806


namespace max_value_of_f_l2608_260862

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 1/2 * x^2 + 2*a*x

theorem max_value_of_f (a : ℝ) (h1 : 0 < a) (h2 : a < 2) 
  (h3 : ∃ x ∈ Set.Icc 1 4, ∀ y ∈ Set.Icc 1 4, f a x ≤ f a y) 
  (h4 : ∃ x ∈ Set.Icc 1 4, f a x = -16/3) :
  ∃ x ∈ Set.Icc 1 4, ∀ y ∈ Set.Icc 1 4, f a y ≤ f a x ∧ f a x = 10/3 := by
  sorry

end max_value_of_f_l2608_260862


namespace sqrt_product_sqrt_three_times_sqrt_five_equals_sqrt_fifteen_l2608_260857

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b :=
by sorry

theorem sqrt_three_times_sqrt_five_equals_sqrt_fifteen :
  Real.sqrt 3 * Real.sqrt 5 = Real.sqrt 15 :=
by sorry

end sqrt_product_sqrt_three_times_sqrt_five_equals_sqrt_fifteen_l2608_260857


namespace profit_percentage_problem_l2608_260886

/-- Given that the cost price of 20 articles equals the selling price of x articles,
    and the profit percentage is 25%, prove that x equals 16. -/
theorem profit_percentage_problem (x : ℝ) 
  (h1 : 20 * cost_price = x * selling_price)
  (h2 : selling_price = 1.25 * cost_price) : 
  x = 16 := by
  sorry

end profit_percentage_problem_l2608_260886


namespace number_problem_l2608_260809

theorem number_problem : ∃! x : ℝ, (x / 3) + 12 = 20 ∧ x = 24 := by
  sorry

end number_problem_l2608_260809


namespace hyperbola_eccentricity_l2608_260892

/-- Hyperbola eccentricity theorem -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let c := 3 * b
  let e := c / a
  (c + b/2) / (c - b/2) = 7/5 → e = 3 * Real.sqrt 2 / 4 := by
  sorry

end hyperbola_eccentricity_l2608_260892


namespace exam_maximum_marks_l2608_260878

/-- The maximum marks of an exam, given the conditions from the problem -/
def maximum_marks : ℕ :=
  let required_percentage : ℚ := 80 / 100
  let marks_obtained : ℕ := 200
  let marks_short : ℕ := 200
  500

/-- Theorem stating that the maximum marks of the exam is 500 -/
theorem exam_maximum_marks :
  let required_percentage : ℚ := 80 / 100
  let marks_obtained : ℕ := 200
  let marks_short : ℕ := 200
  maximum_marks = 500 := by
  sorry

#check exam_maximum_marks

end exam_maximum_marks_l2608_260878


namespace wage_difference_l2608_260880

/-- The total pay for the research project -/
def total_pay : ℝ := 360

/-- Candidate P's hourly wage -/
def wage_p : ℝ := 18

/-- Candidate Q's hourly wage -/
def wage_q : ℝ := 12

/-- The number of hours candidate P needs to complete the job -/
def hours_p : ℝ := 20

/-- The number of hours candidate Q needs to complete the job -/
def hours_q : ℝ := 30

theorem wage_difference : 
  (wage_p = 1.5 * wage_q) ∧ 
  (hours_q = hours_p + 10) ∧ 
  (wage_p * hours_p = total_pay) ∧ 
  (wage_q * hours_q = total_pay) → 
  wage_p - wage_q = 6 := by
  sorry

end wage_difference_l2608_260880


namespace journey_portions_l2608_260867

/-- Proves that the journey is divided into 5 portions given the conditions -/
theorem journey_portions (total_distance : ℝ) (speed : ℝ) (time : ℝ) (portions_covered : ℕ) :
  total_distance = 35 →
  speed = 40 →
  time = 0.7 →
  portions_covered = 4 →
  (speed * time) / portions_covered = total_distance / 5 :=
by sorry

end journey_portions_l2608_260867


namespace smallest_winning_number_l2608_260819

theorem smallest_winning_number : ∃ N : ℕ, N ≤ 999 ∧ 
  (∀ m : ℕ, m < N → (16 * m + 980 > 1200 ∨ 16 * m + 1050 ≤ 1200)) ∧
  16 * N + 980 ≤ 1200 ∧ 
  16 * N + 1050 > 1200 ∧
  N = 10 := by
  sorry

end smallest_winning_number_l2608_260819


namespace nikolai_faster_l2608_260895

/-- Represents a mountain goat with a specific jump distance -/
structure Goat where
  name : String
  jump_distance : ℕ

/-- The race parameters -/
def turning_point : ℕ := 2000

/-- Gennady's characteristics -/
def gennady : Goat := ⟨"Gennady", 6⟩

/-- Nikolai's characteristics -/
def nikolai : Goat := ⟨"Nikolai", 4⟩

/-- Calculates the number of jumps needed to reach the turning point -/
def jumps_to_turning_point (g : Goat) : ℕ :=
  (turning_point + g.jump_distance - 1) / g.jump_distance

/-- Calculates the total distance traveled to the turning point -/
def distance_to_turning_point (g : Goat) : ℕ :=
  (jumps_to_turning_point g) * g.jump_distance

/-- Theorem stating that Nikolai completes the journey faster -/
theorem nikolai_faster : 
  distance_to_turning_point nikolai < distance_to_turning_point gennady :=
sorry

end nikolai_faster_l2608_260895


namespace parallelogram_inscribed_in_circle_is_rectangle_l2608_260874

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

-- Define a parallelogram
def isParallelogram (q : Quadrilateral) : Prop := sorry

-- Define an inscribed quadrilateral
def isInscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Define a rectangle
def isRectangle (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem parallelogram_inscribed_in_circle_is_rectangle 
  (q : Quadrilateral) (c : Circle) : 
  isParallelogram q → isInscribed q c → isRectangle q := by sorry

end parallelogram_inscribed_in_circle_is_rectangle_l2608_260874


namespace quentavious_gum_pieces_l2608_260861

/-- Represents the types of coins --/
inductive Coin
  | Nickel
  | Dime
  | Quarter

/-- Calculates the number of gum pieces for a given coin type --/
def gumPieces (c : Coin) : ℕ :=
  match c with
  | Coin.Nickel => 2
  | Coin.Dime => 3
  | Coin.Quarter => 5

/-- Represents the initial state of coins --/
structure InitialCoins where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Represents the final state of coins --/
structure FinalCoins where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the number of gum pieces received --/
def gumReceived (initial : InitialCoins) (final : FinalCoins) : ℕ :=
  let exchanged_nickels := initial.nickels - final.nickels
  let exchanged_dimes := initial.dimes - final.dimes
  let exchanged_quarters := initial.quarters - final.quarters
  if exchanged_nickels > 0 && exchanged_dimes > 0 && exchanged_quarters > 0 then
    15
  else
    exchanged_nickels * gumPieces Coin.Nickel +
    exchanged_dimes * gumPieces Coin.Dime +
    exchanged_quarters * gumPieces Coin.Quarter

theorem quentavious_gum_pieces :
  let initial := InitialCoins.mk 5 6 4
  let final := FinalCoins.mk 2 1 0
  gumReceived initial final = 15 := by
  sorry

end quentavious_gum_pieces_l2608_260861


namespace self_employed_tax_calculation_l2608_260873

/-- Calculates the tax amount for a self-employed citizen --/
def calculate_tax_amount (income : ℝ) (tax_rate : ℝ) : ℝ :=
  income * tax_rate

/-- The problem statement --/
theorem self_employed_tax_calculation :
  let income : ℝ := 350000
  let tax_rate : ℝ := 0.06
  calculate_tax_amount income tax_rate = 21000 := by
  sorry

end self_employed_tax_calculation_l2608_260873


namespace second_car_speed_l2608_260865

/-- Given two cars starting from opposite ends of a 60-mile highway at the same time,
    with one car traveling at 13 mph and both cars meeting after 2 hours,
    prove that the speed of the second car is 17 mph. -/
theorem second_car_speed (highway_length : ℝ) (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  highway_length = 60 →
  time = 2 →
  speed1 = 13 →
  speed1 * time + speed2 * time = highway_length →
  speed2 = 17 := by
  sorry

end second_car_speed_l2608_260865


namespace pauls_toy_boxes_l2608_260897

theorem pauls_toy_boxes (toys_per_box : ℕ) (total_toys : ℕ) (h1 : toys_per_box = 8) (h2 : total_toys = 32) :
  total_toys / toys_per_box = 4 := by
sorry

end pauls_toy_boxes_l2608_260897


namespace remainder_425421_div_12_l2608_260850

theorem remainder_425421_div_12 : 425421 % 12 = 9 := by
  sorry

end remainder_425421_div_12_l2608_260850


namespace probability_three_two_color_l2608_260851

/-- The probability of drawing 3 balls of one color and 2 of the other from a bin with 10 black and 10 white balls -/
theorem probability_three_two_color (total_balls : ℕ) (black_balls white_balls : ℕ) (drawn_balls : ℕ) : 
  total_balls = black_balls + white_balls →
  black_balls = 10 →
  white_balls = 10 →
  drawn_balls = 5 →
  (Nat.choose total_balls drawn_balls : ℚ) * (30 : ℚ) / (43 : ℚ) = 
    (Nat.choose black_balls 3 * Nat.choose white_balls 2 + 
     Nat.choose black_balls 2 * Nat.choose white_balls 3 : ℚ) :=
by sorry

#check probability_three_two_color

end probability_three_two_color_l2608_260851


namespace fraction_simplification_l2608_260808

theorem fraction_simplification : (1952^2 - 1940^2) / (1959^2 - 1933^2) = 6/13 := by
  sorry

end fraction_simplification_l2608_260808


namespace two_black_balls_probability_l2608_260802

/-- The probability of drawing two black balls without replacement from a box containing 8 white balls and 7 black balls is 1/5. -/
theorem two_black_balls_probability :
  let total_balls : ℕ := 8 + 7
  let black_balls : ℕ := 7
  let prob_first_black : ℚ := black_balls / total_balls
  let prob_second_black : ℚ := (black_balls - 1) / (total_balls - 1)
  prob_first_black * prob_second_black = 1 / 5 := by
sorry


end two_black_balls_probability_l2608_260802


namespace log_power_sum_l2608_260832

theorem log_power_sum (a b : ℝ) (h1 : a = Real.log 64) (h2 : b = Real.log 25) :
  (8 : ℝ) ^ (a / b) + (5 : ℝ) ^ (b / a) = 89 := by
  sorry

end log_power_sum_l2608_260832


namespace cakes_destroyed_or_stolen_proof_l2608_260899

def total_cakes : ℕ := 36
def num_stacks : ℕ := 2

def cakes_per_stack : ℕ := total_cakes / num_stacks

def crow_knocked_percentage : ℚ := 60 / 100
def mischievous_squirrel_stole_fraction : ℚ := 1 / 3
def red_squirrel_took_percentage : ℚ := 25 / 100
def red_squirrel_dropped_fraction : ℚ := 1 / 2
def dog_ate : ℕ := 4

def cakes_destroyed_or_stolen : ℕ := 19

theorem cakes_destroyed_or_stolen_proof :
  let crow_knocked := (crow_knocked_percentage * cakes_per_stack).floor
  let mischievous_squirrel_stole := (mischievous_squirrel_stole_fraction * crow_knocked).floor
  let red_squirrel_took := (red_squirrel_took_percentage * cakes_per_stack).floor
  let red_squirrel_destroyed := (red_squirrel_dropped_fraction * red_squirrel_took).floor
  crow_knocked + mischievous_squirrel_stole + red_squirrel_destroyed + dog_ate = cakes_destroyed_or_stolen :=
by sorry

end cakes_destroyed_or_stolen_proof_l2608_260899


namespace arithmetic_sequence_problem_l2608_260871

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ) : 
  (∀ n, a (n + 1) - a n = 1) →  -- arithmetic sequence with common difference 1
  (∀ n, S n = n * a 1 + n * (n - 1) / 2) →  -- sum formula for arithmetic sequence
  S 8 = 4 * S 4 →  -- given condition
  a 10 = 19 / 2 := by
sorry

end arithmetic_sequence_problem_l2608_260871


namespace power_of_power_l2608_260831

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l2608_260831


namespace triangle_construction_solutions_l2608_260825

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Checks if a point is the foot of an altitude in a triangle -/
def isAltitudeFoot (P : Point2D) (T : Triangle) : Prop := sorry

/-- Checks if a point is the midpoint of a side in a triangle -/
def isMidpoint (P : Point2D) (A B : Point2D) : Prop := sorry

/-- Checks if a point is the midpoint of an altitude in a triangle -/
def isAltitudeMidpoint (P : Point2D) (T : Triangle) : Prop := sorry

/-- The main theorem statement -/
theorem triangle_construction_solutions 
  (A₀ B₁ C₂ : Point2D) : 
  ∃ (T₁ T₂ : Triangle), 
    T₁ ≠ T₂ ∧ 
    isAltitudeFoot A₀ T₁ ∧
    isAltitudeFoot A₀ T₂ ∧
    isMidpoint B₁ T₁.A T₁.C ∧
    isMidpoint B₁ T₂.A T₂.C ∧
    isAltitudeMidpoint C₂ T₁ ∧
    isAltitudeMidpoint C₂ T₂ :=
  sorry

end triangle_construction_solutions_l2608_260825


namespace polynomial_equality_l2608_260889

theorem polynomial_equality (a b c d e : ℝ) :
  (∀ x : ℝ, (x - 3)^4 = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
  b + c + d + e = 15 := by
sorry

end polynomial_equality_l2608_260889
