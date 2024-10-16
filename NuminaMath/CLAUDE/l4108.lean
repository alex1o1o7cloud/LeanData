import Mathlib

namespace NUMINAMATH_CALUDE_max_value_complex_expression_l4108_410885

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = Real.sqrt 3) :
  Complex.abs ((z - 1) * (z + 1)^2) ≤ 3 * Real.sqrt 3 ∧
  ∃ z₀ : ℂ, Complex.abs z₀ = Real.sqrt 3 ∧ Complex.abs ((z₀ - 1) * (z₀ + 1)^2) = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l4108_410885


namespace NUMINAMATH_CALUDE_simplify_xy_expression_l4108_410894

theorem simplify_xy_expression (x y : ℝ) : 4 * x * y - 2 * x * y = 2 * x * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_xy_expression_l4108_410894


namespace NUMINAMATH_CALUDE_third_circle_radius_l4108_410876

/-- Two externally tangent circles with a third circle tangent to both and their common external tangent -/
structure TangentCircles where
  /-- Center of the first circle -/
  A : ℝ × ℝ
  /-- Center of the second circle -/
  B : ℝ × ℝ
  /-- Radius of the first circle -/
  r1 : ℝ
  /-- Radius of the second circle -/
  r2 : ℝ
  /-- Radius of the third circle -/
  r3 : ℝ
  /-- The first two circles are externally tangent -/
  externally_tangent : dist A B = r1 + r2
  /-- The third circle is tangent to the first circle -/
  tangent_to_first : ∃ P : ℝ × ℝ, dist P A = r1 + r3 ∧ dist P B = r2 + r3
  /-- The third circle is tangent to the second circle -/
  tangent_to_second : ∃ Q : ℝ × ℝ, dist Q A = r1 + r3 ∧ dist Q B = r2 + r3
  /-- The third circle is tangent to the common external tangent of the first two circles -/
  tangent_to_external : ∃ T : ℝ × ℝ, dist T A = r1 ∧ dist T B = r2 ∧ 
    ∃ C : ℝ × ℝ, dist C A = r1 + r3 ∧ dist C B = r2 + r3 ∧ dist C T = r3

/-- The radius of the third circle is 1 -/
theorem third_circle_radius (tc : TangentCircles) (h1 : tc.r1 = 2) (h2 : tc.r2 = 5) : tc.r3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_third_circle_radius_l4108_410876


namespace NUMINAMATH_CALUDE_integer_roots_imply_n_values_l4108_410875

theorem integer_roots_imply_n_values (n : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 - 6*x - 4*n^2 - 32*n = 0 ∧ y^2 - 6*y - 4*n^2 - 32*n = 0) →
  (n = 10 ∨ n = 0 ∨ n = -8 ∨ n = -18) :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_imply_n_values_l4108_410875


namespace NUMINAMATH_CALUDE_peetu_speed_difference_l4108_410839

/-- Peetu's cycling speed as a boy in miles per minute -/
def boy_speed : ℚ := 18 / (2 * 60 + 15)

/-- Peetu's walking speed as an older adult in miles per minute -/
def adult_speed : ℚ := 6 / (3 * 60 - 30)

/-- The difference in minutes per mile between adult walking and boyhood cycling -/
def speed_difference : ℚ := (1 / adult_speed) - (1 / boy_speed)

theorem peetu_speed_difference : speed_difference = 35/2 := by sorry

end NUMINAMATH_CALUDE_peetu_speed_difference_l4108_410839


namespace NUMINAMATH_CALUDE_work_completion_time_l4108_410823

theorem work_completion_time (b_completion_time b_work_days a_remaining_time : ℕ) 
  (hb : b_completion_time = 15)
  (hbw : b_work_days = 10)
  (ha : a_remaining_time = 7) : 
  ∃ (a_completion_time : ℕ), a_completion_time = 21 ∧
  (a_completion_time : ℚ)⁻¹ * a_remaining_time = 1 - (b_work_days : ℚ) / b_completion_time :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l4108_410823


namespace NUMINAMATH_CALUDE_parabola_sum_l4108_410803

/-- Parabola in the first quadrant -/
def parabola (x y : ℝ) : Prop := x^2 = (1/2) * y ∧ x > 0 ∧ y > 0

/-- Point on the parabola -/
def point_on_parabola (a : ℕ → ℝ) (i : ℕ) : Prop :=
  parabola (a i) (2 * (a i)^2)

/-- Tangent line intersection property -/
def tangent_intersection (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, i > 0 → point_on_parabola a i →
    ∃ m b : ℝ, (m * (a (i+1)) + b = 0) ∧
              (∀ x y : ℝ, y - 2*(a i)^2 = m*(x - a i) → parabola x y)

/-- The main theorem -/
theorem parabola_sum (a : ℕ → ℝ) :
  (∀ i : ℕ, i > 0 → point_on_parabola a i) →
  tangent_intersection a →
  a 2 = 32 →
  a 2 + a 4 + a 6 = 42 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_l4108_410803


namespace NUMINAMATH_CALUDE_watermelon_weights_sum_l4108_410882

/-- Watermelon weights problem -/
theorem watermelon_weights_sum : 
  -- Given conditions
  let michael_largest : ℝ := 12
  let clay_first : ℝ := 1.5 * michael_largest
  let john_first : ℝ := clay_first / 2
  let emily : ℝ := 0.75 * john_first
  let sophie_first : ℝ := emily + 3
  let michael_smallest : ℝ := michael_largest * 0.7
  let clay_second : ℝ := clay_first * 1.2
  let john_second : ℝ := (john_first + emily) / 2
  let sophie_second : ℝ := 3 * (clay_second - clay_first)
  -- Theorem statement
  michael_largest + michael_smallest + clay_first + clay_second + 
  john_first + john_second + emily + sophie_first + sophie_second = 104.175 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_weights_sum_l4108_410882


namespace NUMINAMATH_CALUDE_power_of_128_l4108_410825

theorem power_of_128 : (128 : ℝ) ^ (7/3) = 65536 * (2 : ℝ) ^ (1/3) := by sorry

end NUMINAMATH_CALUDE_power_of_128_l4108_410825


namespace NUMINAMATH_CALUDE_billy_ice_trays_l4108_410874

theorem billy_ice_trays (ice_cubes_per_tray : ℕ) (total_ice_cubes : ℕ) 
  (h1 : ice_cubes_per_tray = 9)
  (h2 : total_ice_cubes = 72) :
  total_ice_cubes / ice_cubes_per_tray = 8 := by
  sorry

end NUMINAMATH_CALUDE_billy_ice_trays_l4108_410874


namespace NUMINAMATH_CALUDE_actual_average_height_l4108_410845

/-- Represents the average height calculation problem in a class --/
structure HeightProblem where
  totalStudents : ℕ
  initialAverage : ℚ
  incorrectHeights : List ℚ
  actualHeights : List ℚ

/-- Calculates the actual average height given the problem data --/
def calculateActualAverage (problem : HeightProblem) : ℚ :=
  let initialTotal := problem.initialAverage * problem.totalStudents
  let heightDifference := (problem.incorrectHeights.sum - problem.actualHeights.sum)
  let correctedTotal := initialTotal - heightDifference
  correctedTotal / problem.totalStudents

/-- The theorem stating that the actual average height is 164.5 cm --/
theorem actual_average_height
  (problem : HeightProblem)
  (h1 : problem.totalStudents = 50)
  (h2 : problem.initialAverage = 165)
  (h3 : problem.incorrectHeights = [150, 175, 190])
  (h4 : problem.actualHeights = [135, 170, 185]) :
  calculateActualAverage problem = 164.5 := by
  sorry


end NUMINAMATH_CALUDE_actual_average_height_l4108_410845


namespace NUMINAMATH_CALUDE_circle_intersects_y_axis_l4108_410830

theorem circle_intersects_y_axis (D E F : ℝ) :
  (∃ y₁ y₂ : ℝ, y₁ < 0 ∧ y₂ > 0 ∧ 
    y₁^2 + E*y₁ + F = 0 ∧ 
    y₂^2 + E*y₂ + F = 0) →
  F < 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersects_y_axis_l4108_410830


namespace NUMINAMATH_CALUDE_cosine_sum_17th_roots_l4108_410819

theorem cosine_sum_17th_roots : 
  Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_17th_roots_l4108_410819


namespace NUMINAMATH_CALUDE_sum_in_base5_l4108_410888

-- Define a function to convert from base 10 to base 5
def toBase5 (n : ℕ) : List ℕ := sorry

-- Define a function to interpret a list of digits as a number in base 5
def fromBase5 (digits : List ℕ) : ℕ := sorry

theorem sum_in_base5 : 
  toBase5 (12 + 47) = [2, 1, 4] := by sorry

end NUMINAMATH_CALUDE_sum_in_base5_l4108_410888


namespace NUMINAMATH_CALUDE_tangent_angle_at_x_1_l4108_410895

/-- The angle of inclination of the tangent to the curve y = x³ - 2x + m at x = 1 is 45° -/
theorem tangent_angle_at_x_1 (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 2*x + m
  let f' : ℝ → ℝ := λ x => 3*x^2 - 2
  let slope : ℝ := f' 1
  Real.arctan slope = π/4 := by sorry

end NUMINAMATH_CALUDE_tangent_angle_at_x_1_l4108_410895


namespace NUMINAMATH_CALUDE_valid_rearrangements_count_valid_rearrangements_count_is_360_l4108_410809

/-- Represents the word to be rearranged -/
def word : String := "REPRESENT"

/-- Counts the occurrences of a character in a string -/
def count_char (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

/-- The number of vowels in the word -/
def num_vowels : Nat :=
  count_char word 'E'

/-- The number of consonants in the word -/
def num_consonants : Nat :=
  word.length - num_vowels

/-- The number of unique consonants in the word -/
def num_unique_consonants : Nat :=
  (word.toList.filter (λ c => c ≠ 'E') |>.eraseDups).length

/-- The main theorem stating the number of valid rearrangements -/
theorem valid_rearrangements_count : Nat :=
  (Nat.factorial num_consonants) / (Nat.factorial (num_consonants - num_unique_consonants + 1))

/-- The proof of the main theorem -/
theorem valid_rearrangements_count_is_360 : valid_rearrangements_count = 360 := by
  sorry

end NUMINAMATH_CALUDE_valid_rearrangements_count_valid_rearrangements_count_is_360_l4108_410809


namespace NUMINAMATH_CALUDE_production_days_l4108_410880

/-- Given the average daily production for n days and the effect of adding one more day's production,
    prove the value of n. -/
theorem production_days (n : ℕ) : 
  (∀ (P : ℕ), P / n = 60 → (P + 90) / (n + 1) = 65) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l4108_410880


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l4108_410857

/-- The perimeter of a region formed by four 90° arcs of circles with circumference 48 -/
theorem shaded_region_perimeter (c : ℝ) (h : c = 48) : 
  4 * (90 / 360 * c) = 48 := by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l4108_410857


namespace NUMINAMATH_CALUDE_perimeterDifference_l4108_410828

/-- Calculates the perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Calculates the perimeter of an L-shaped formation (2x2 square missing 1x1 square) -/
def lShapePerimeter : ℕ := 5

/-- Calculates the perimeter of Figure 1 (composite of 3x1 rectangle and L-shape) -/
def figure1Perimeter : ℕ :=
  rectanglePerimeter 3 1 + lShapePerimeter

/-- Calculates the perimeter of Figure 2 (6x2 rectangle) -/
def figure2Perimeter : ℕ :=
  rectanglePerimeter 6 2

/-- The main theorem stating the positive difference in perimeters -/
theorem perimeterDifference :
  (max figure1Perimeter figure2Perimeter) - (min figure1Perimeter figure2Perimeter) = 3 := by
  sorry

end NUMINAMATH_CALUDE_perimeterDifference_l4108_410828


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_ten_thirds_l4108_410826

theorem sum_abcd_equals_negative_ten_thirds
  (a b c d : ℚ)
  (h : a + 1 = b + 2 ∧ b + 2 = c + 3 ∧ c + 3 = d + 4 ∧ d + 4 = a + b + c + d + 5) :
  a + b + c + d = -10/3 :=
by sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_ten_thirds_l4108_410826


namespace NUMINAMATH_CALUDE_optimal_selling_price_l4108_410843

/-- Represents the profit function for a smartphone accessory -/
def profit_function (a : ℝ) (x : ℝ) : ℝ := 5 * a * (1 + 4*x - x^2 - 4*x^3)

/-- Theorem stating the optimal selling price for maximum profit -/
theorem optimal_selling_price (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧
  (∀ (y : ℝ), 0 < y ∧ y < 1 → profit_function a x ≥ profit_function a y) ∧
  20 * (1 + x) = 30 :=
sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l4108_410843


namespace NUMINAMATH_CALUDE_sheep_cant_reach_midpoint_l4108_410844

/-- Represents the setup of two posts and a sheep -/
structure MeadowSetup where
  postDistance : ℝ
  ropeLength1 : ℝ
  ropeLength2 : ℝ

/-- Defines when a sheep can reach a point given the setup -/
def canReachPoint (setup : MeadowSetup) (point : ℝ) : Prop :=
  point ≤ setup.ropeLength1 ∨ (setup.postDistance - point) ≤ setup.ropeLength2

/-- Theorem stating that to prevent the sheep from reaching the midpoint,
    at least one rope must be shorter than half the distance between posts -/
theorem sheep_cant_reach_midpoint (setup : MeadowSetup) 
  (h1 : setup.postDistance = 20)
  (h2 : ¬(canReachPoint setup (setup.postDistance / 2))) :
  setup.ropeLength1 < 10 ∨ setup.ropeLength2 < 10 := by
  sorry

end NUMINAMATH_CALUDE_sheep_cant_reach_midpoint_l4108_410844


namespace NUMINAMATH_CALUDE_video_difference_l4108_410869

/-- The number of videos watched by three friends -/
def total_videos : ℕ := 411

/-- The number of videos watched by Kelsey -/
def kelsey_videos : ℕ := 160

/-- The number of videos watched by Ekon -/
def ekon_videos : ℕ := kelsey_videos - 43

/-- The number of videos watched by Uma -/
def uma_videos : ℕ := total_videos - kelsey_videos - ekon_videos

/-- Ekon watched fewer videos than Uma -/
axiom ekon_less_than_uma : ekon_videos < uma_videos

theorem video_difference : uma_videos - ekon_videos = 17 := by
  sorry

end NUMINAMATH_CALUDE_video_difference_l4108_410869


namespace NUMINAMATH_CALUDE_rectangle_area_l4108_410824

/-- Given a rectangle with length 15 cm and perimeter-to-width ratio of 5:1, its area is 150 cm² -/
theorem rectangle_area (w : ℝ) (h1 : (2 * 15 + 2 * w) / w = 5) : w * 15 = 150 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l4108_410824


namespace NUMINAMATH_CALUDE_number_puzzle_l4108_410813

theorem number_puzzle : ∃ x : ℝ, (x / 5) + 10 = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l4108_410813


namespace NUMINAMATH_CALUDE_professor_count_l4108_410860

theorem professor_count (p : ℕ) 
  (h1 : 6480 % p = 0)  -- 6480 is divisible by p
  (h2 : 11200 % (p + 3) = 0)  -- 11200 is divisible by (p + 3)
  (h3 : (6480 : ℚ) / p < (11200 : ℚ) / (p + 3))  -- grades per professor increased
  : p = 5 := by
  sorry

end NUMINAMATH_CALUDE_professor_count_l4108_410860


namespace NUMINAMATH_CALUDE_isosceles_triangle_most_stable_l4108_410833

-- Define the shapes
inductive Shape
  | RegularPentagon
  | Square
  | Trapezoid
  | IsoscelesTriangle

-- Define the stability property
def is_stable (s : Shape) : Prop :=
  match s with
  | Shape.RegularPentagon => false
  | Shape.Square => false
  | Shape.Trapezoid => false
  | Shape.IsoscelesTriangle => true

-- Theorem statement
theorem isosceles_triangle_most_stable :
  ∀ s : Shape, is_stable s → s = Shape.IsoscelesTriangle :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_most_stable_l4108_410833


namespace NUMINAMATH_CALUDE_high_school_student_distribution_l4108_410849

theorem high_school_student_distribution :
  ∀ (freshmen sophomores juniors seniors : ℕ),
    freshmen + sophomores + juniors + seniors = 800 →
    juniors = 216 →
    sophomores = 200 →
    seniors = 160 →
    freshmen - sophomores = 24 := by
  sorry

end NUMINAMATH_CALUDE_high_school_student_distribution_l4108_410849


namespace NUMINAMATH_CALUDE_mikes_pears_l4108_410815

/-- Given that Jason picked 7 pears and the total number of pears picked was 15,
    prove that Mike picked 8 pears. -/
theorem mikes_pears (jason_pears total_pears : ℕ) 
    (h1 : jason_pears = 7)
    (h2 : total_pears = 15) :
    total_pears - jason_pears = 8 := by
  sorry

end NUMINAMATH_CALUDE_mikes_pears_l4108_410815


namespace NUMINAMATH_CALUDE_subtraction_value_l4108_410855

theorem subtraction_value (x y : ℝ) : 
  (x - 5) / 7 = 7 → (x - y) / 8 = 6 → y = 6 := by
sorry

end NUMINAMATH_CALUDE_subtraction_value_l4108_410855


namespace NUMINAMATH_CALUDE_temperature_conversion_l4108_410866

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 75 → k = 167 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l4108_410866


namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l4108_410870

/-- Proves that the initial number of bananas per child is 2 -/
theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) 
  (extra_bananas : ℕ) (h1 : total_children = 610) (h2 : absent_children = 305) 
  (h3 : extra_bananas = 2) : 
  (total_children : ℚ) * (total_children - absent_children) = 
  (total_children - absent_children) * ((total_children - absent_children) + extra_bananas) :=
by sorry

#check initial_bananas_per_child

end NUMINAMATH_CALUDE_initial_bananas_per_child_l4108_410870


namespace NUMINAMATH_CALUDE_shaded_area_proof_l4108_410853

theorem shaded_area_proof (side_length : ℝ) (circle_radius1 circle_radius2 circle_radius3 : ℝ) :
  side_length = 30 ∧ 
  circle_radius1 = 5 ∧ 
  circle_radius2 = 4 ∧ 
  circle_radius3 = 3 →
  (side_length^2 / 9) * 5 = 500 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l4108_410853


namespace NUMINAMATH_CALUDE_unique_square_board_state_l4108_410890

/-- Represents the state of numbers on the board -/
def BoardState := List Nat

/-- The process of replacing a number with its proper divisors -/
def replace_with_divisors (a : Nat) : BoardState :=
  sorry

/-- The full process of repeatedly replacing numbers until no more replacements are possible -/
def process (initial : BoardState) : BoardState :=
  sorry

/-- Theorem: The only natural number N for which the described process
    can result in exactly N^2 numbers on the board is 1 -/
theorem unique_square_board_state (N : Nat) :
  (∃ (final : BoardState), process [N] = final ∧ final.length = N^2) ↔ N = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_square_board_state_l4108_410890


namespace NUMINAMATH_CALUDE_banana_permutations_l4108_410865

-- Define the word and its properties
def banana_length : ℕ := 6
def banana_a_count : ℕ := 3
def banana_n_count : ℕ := 2
def banana_b_count : ℕ := 1

-- Define the function to calculate permutations with repetition
def permutations_with_repetition (n : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial n / (repetitions.map Nat.factorial).prod

-- Theorem statement
theorem banana_permutations :
  permutations_with_repetition banana_length [banana_a_count, banana_n_count, banana_b_count] = 60 := by
  sorry


end NUMINAMATH_CALUDE_banana_permutations_l4108_410865


namespace NUMINAMATH_CALUDE_point_inside_given_circle_l4108_410811

def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 18

def point_inside_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 3)^2 < 18

theorem point_inside_given_circle :
  point_inside_circle 1 1 := by sorry

end NUMINAMATH_CALUDE_point_inside_given_circle_l4108_410811


namespace NUMINAMATH_CALUDE_quadratic_solution_l4108_410801

theorem quadratic_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : ∀ x : ℝ, x^2 + 2*a*x + b = 0 ↔ x = a ∨ x = b) :
  a = 1 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l4108_410801


namespace NUMINAMATH_CALUDE_smallest_multiple_congruence_l4108_410842

theorem smallest_multiple_congruence : ∃! n : ℕ, 
  n > 0 ∧ 
  31 ∣ n ∧ 
  n % 103 = 7 ∧
  ∀ m : ℕ, m > 0 → 31 ∣ m → m % 103 = 7 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_congruence_l4108_410842


namespace NUMINAMATH_CALUDE_total_money_sally_condition_jolly_condition_molly_condition_l4108_410878

/-- The amount of money Sally has -/
def sally_money : ℕ := 100

/-- The amount of money Jolly has -/
def jolly_money : ℕ := 50

/-- The amount of money Molly has -/
def molly_money : ℕ := 70

/-- The theorem stating the total amount of money -/
theorem total_money : sally_money + jolly_money + molly_money = 220 := by
  sorry

/-- Sally would have $80 if she had $20 less -/
theorem sally_condition : sally_money - 20 = 80 := by
  sorry

/-- Jolly would have $70 if she had $20 more -/
theorem jolly_condition : jolly_money + 20 = 70 := by
  sorry

/-- Molly would have $100 if she had $30 more -/
theorem molly_condition : molly_money + 30 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_money_sally_condition_jolly_condition_molly_condition_l4108_410878


namespace NUMINAMATH_CALUDE_farm_animal_ratio_l4108_410858

theorem farm_animal_ratio : 
  let goats : ℕ := 66
  let chickens : ℕ := 2 * goats
  let total_goats_chickens : ℕ := goats + chickens
  let ducks : ℕ := 99  -- We define this to match the problem constraints
  let pigs : ℕ := ducks / 3
  goats = pigs + 33 →
  (ducks : ℚ) / total_goats_chickens = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_farm_animal_ratio_l4108_410858


namespace NUMINAMATH_CALUDE_fair_spending_remainder_l4108_410827

/-- Calculates the remaining amount after spending on snacks and games at a fair. -/
theorem fair_spending_remainder (initial_amount snack_cost : ℕ) : 
  initial_amount = 80 →
  snack_cost = 18 →
  initial_amount - (snack_cost + 3 * snack_cost) = 8 := by
  sorry

#check fair_spending_remainder

end NUMINAMATH_CALUDE_fair_spending_remainder_l4108_410827


namespace NUMINAMATH_CALUDE_tomatoes_calculation_l4108_410814

/-- The number of tomato plants -/
def num_plants : ℕ := 50

/-- The number of tomatoes produced by each plant -/
def tomatoes_per_plant : ℕ := 15

/-- The fraction of tomatoes that are dried -/
def dried_fraction : ℚ := 2 / 3

/-- The fraction of remaining tomatoes used for marinara sauce -/
def marinara_fraction : ℚ := 1 / 2

/-- The number of tomatoes left after drying and making marinara sauce -/
def tomatoes_left : ℕ := 125

theorem tomatoes_calculation :
  (num_plants * tomatoes_per_plant : ℚ) * (1 - dried_fraction) * (1 - marinara_fraction) = tomatoes_left := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_calculation_l4108_410814


namespace NUMINAMATH_CALUDE_three_card_selection_l4108_410806

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (face_cards_per_suit : Nat)

/-- Calculates the number of ways to choose 3 cards from a deck
    such that all three cards are of different suits and one is a face card -/
def choose_three_cards (d : Deck) : Nat :=
  d.suits * d.face_cards_per_suit * (d.suits - 1).choose 2 * (d.cards_per_suit ^ 2)

/-- Theorem stating the number of ways to choose 3 cards from a standard deck
    with the given conditions -/
theorem three_card_selection (d : Deck) 
  (h1 : d.cards = 52)
  (h2 : d.suits = 4)
  (h3 : d.cards_per_suit = 13)
  (h4 : d.face_cards_per_suit = 3) :
  choose_three_cards d = 6084 := by
  sorry

#eval choose_three_cards { cards := 52, suits := 4, cards_per_suit := 13, face_cards_per_suit := 3 }

end NUMINAMATH_CALUDE_three_card_selection_l4108_410806


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l4108_410896

theorem complex_number_quadrant : ∃ (a b : ℝ), a < 0 ∧ b < 0 ∧ (1 - I) / (1 + 2*I) = a + b*I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l4108_410896


namespace NUMINAMATH_CALUDE_best_fitting_model_has_highest_r_squared_model1_has_best_fitting_effect_l4108_410899

/-- Represents a regression model with its R² value -/
structure RegressionModel where
  name : String
  r_squared : ℝ
  r_squared_nonneg : 0 ≤ r_squared
  r_squared_le_one : r_squared ≤ 1

/-- Determines if a model has the best fitting effect among a list of models -/
def has_best_fitting_effect (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, m.r_squared ≤ model.r_squared

/-- The theorem stating that the model with the highest R² value has the best fitting effect -/
theorem best_fitting_model_has_highest_r_squared 
  (models : List RegressionModel) (model : RegressionModel) 
  (h_model_in_list : model ∈ models) 
  (h_nonempty : models ≠ []) :
  has_best_fitting_effect model models ↔ 
  ∀ m ∈ models, m.r_squared ≤ model.r_squared :=
sorry

/-- The specific problem instance -/
def model1 : RegressionModel := ⟨"Model 1", 0.98, by norm_num, by norm_num⟩
def model2 : RegressionModel := ⟨"Model 2", 0.80, by norm_num, by norm_num⟩
def model3 : RegressionModel := ⟨"Model 3", 0.54, by norm_num, by norm_num⟩
def model4 : RegressionModel := ⟨"Model 4", 0.35, by norm_num, by norm_num⟩

def problem_models : List RegressionModel := [model1, model2, model3, model4]

theorem model1_has_best_fitting_effect : 
  has_best_fitting_effect model1 problem_models :=
sorry

end NUMINAMATH_CALUDE_best_fitting_model_has_highest_r_squared_model1_has_best_fitting_effect_l4108_410899


namespace NUMINAMATH_CALUDE_anne_had_fifteen_sweettarts_l4108_410886

/-- The number of Sweettarts Anne had initially -/
def annes_initial_sweettarts (num_friends : ℕ) (sweettarts_per_friend : ℕ) : ℕ :=
  num_friends * sweettarts_per_friend

/-- Theorem stating that Anne had 15 Sweettarts initially -/
theorem anne_had_fifteen_sweettarts :
  annes_initial_sweettarts 3 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_anne_had_fifteen_sweettarts_l4108_410886


namespace NUMINAMATH_CALUDE_f_shifted_l4108_410841

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 + 4*(x + 1) - 5

-- State the theorem
theorem f_shifted (x : ℝ) : f (x + 2) = x^2 + 8*x + 7 := by
  sorry

end NUMINAMATH_CALUDE_f_shifted_l4108_410841


namespace NUMINAMATH_CALUDE_one_third_blue_faces_iff_three_l4108_410834

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  sideLength : n > 0

/-- The total number of faces of all unit cubes when a cube of side length n is cut into n^3 unit cubes -/
def totalFaces (c : Cube n) : ℕ := 6 * n^3

/-- The number of blue faces when a cube of side length n is painted on all sides and cut into n^3 unit cubes -/
def blueFaces (c : Cube n) : ℕ := 6 * n^2

/-- The theorem stating that exactly one-third of the faces are blue if and only if n = 3 -/
theorem one_third_blue_faces_iff_three (c : Cube n) :
  3 * blueFaces c = totalFaces c ↔ n = 3 :=
sorry

end NUMINAMATH_CALUDE_one_third_blue_faces_iff_three_l4108_410834


namespace NUMINAMATH_CALUDE_points_on_line_l4108_410835

-- Define the line y = -3x + b
def line (x : ℝ) (b : ℝ) : ℝ := -3 * x + b

-- Define the points
def point1 (y₁ : ℝ) (b : ℝ) : Prop := y₁ = line (-2) b
def point2 (y₂ : ℝ) (b : ℝ) : Prop := y₂ = line (-1) b
def point3 (y₃ : ℝ) (b : ℝ) : Prop := y₃ = line 1 b

-- Theorem statement
theorem points_on_line (y₁ y₂ y₃ b : ℝ) 
  (h1 : point1 y₁ b) (h2 : point2 y₂ b) (h3 : point3 y₃ b) :
  y₁ > y₂ ∧ y₂ > y₃ := by sorry

end NUMINAMATH_CALUDE_points_on_line_l4108_410835


namespace NUMINAMATH_CALUDE_xingyou_age_l4108_410872

theorem xingyou_age : ℕ :=
  let current_age : ℕ := sorry
  let current_height : ℕ := sorry
  have h1 : current_age = current_height := by sorry
  have h2 : current_age + 3 = 2 * current_height := by sorry
  have h3 : current_age = 3 := by sorry
  3

#check xingyou_age

end NUMINAMATH_CALUDE_xingyou_age_l4108_410872


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l4108_410807

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = q * a n ∧ a n > 0

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_first : a 1 = 1) 
  (h_second : ∃ x : ℝ, a 2 = x + 1) 
  (h_third : ∃ x : ℝ, a 3 = 2 * x + 5) : 
  ∀ n : ℕ, a n = 3^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l4108_410807


namespace NUMINAMATH_CALUDE_system_solution_l4108_410812

theorem system_solution (x y z : ℚ) 
  (eq1 : y + z = 15 - 2*x)
  (eq2 : x + z = -10 - 2*y)
  (eq3 : x + y = 4 - 2*z) :
  2*x + 2*y + 2*z = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4108_410812


namespace NUMINAMATH_CALUDE_fixed_points_of_f_composition_l4108_410893

def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 2*x

theorem fixed_points_of_f_composition :
  ∀ x : ℝ, f (f x) = f x ↔ x = 0 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_composition_l4108_410893


namespace NUMINAMATH_CALUDE_kai_born_in_1995_l4108_410891

/-- Kai's birth year, given his 25th birthday is in March 2020 -/
def kais_birth_year : ℕ := sorry

/-- The year of Kai's 25th birthday -/
def birthday_year : ℕ := 2020

/-- Kai's age at his birthday in 2020 -/
def kais_age : ℕ := 25

theorem kai_born_in_1995 : kais_birth_year = 1995 := by
  sorry

end NUMINAMATH_CALUDE_kai_born_in_1995_l4108_410891


namespace NUMINAMATH_CALUDE_mobile_payment_probability_l4108_410892

def group_size : ℕ := 10

def mobile_payment_prob (p : ℝ) : ℝ := p

def is_independent (p : ℝ) : Prop := true

def num_mobile_users (X : ℕ) : ℕ := X

def variance (X : ℕ) (p : ℝ) : ℝ := group_size * p * (1 - p)

def prob_X_eq (k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose group_size k : ℝ) * p^k * (1 - p)^(group_size - k)

theorem mobile_payment_probability :
  ∀ p : ℝ,
    0 ≤ p ∧ p ≤ 1 →
    is_independent p →
    variance (num_mobile_users X) p = 2.4 →
    prob_X_eq 4 p < prob_X_eq 6 p →
    p = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_mobile_payment_probability_l4108_410892


namespace NUMINAMATH_CALUDE_fraction_value_l4108_410821

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l4108_410821


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l4108_410805

/-- The hyperbola C: x² - y² = a² intersects with the directrix of the parabola y² = 16x 
    at two points with distance 4√3 between them. 
    This theorem states that the length of the real axis of hyperbola C is 4. -/
theorem hyperbola_real_axis_length (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 = -4 ∧ A.1^2 - A.2^2 = a^2) ∧ 
    (B.1 = -4 ∧ B.1^2 - B.2^2 = a^2) ∧ 
    (A.2 - B.2)^2 = 48) →
  2 * a = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l4108_410805


namespace NUMINAMATH_CALUDE_factor_tree_root_value_l4108_410820

/-- Represents a node in the factor tree -/
inductive FactorNode
  | Prime (n : Nat)
  | Composite (left right : FactorNode)

/-- Computes the value of a FactorNode -/
def nodeValue : FactorNode → Nat
  | FactorNode.Prime n => n
  | FactorNode.Composite left right => nodeValue left * nodeValue right

/-- The factor tree structure as given in the problem -/
def factorTree : FactorNode :=
  FactorNode.Composite
    (FactorNode.Composite
      (FactorNode.Prime 7)
      (FactorNode.Composite (FactorNode.Prime 7) (FactorNode.Prime 3)))
    (FactorNode.Composite
      (FactorNode.Prime 11)
      (FactorNode.Composite (FactorNode.Prime 11) (FactorNode.Prime 3)))

theorem factor_tree_root_value :
  nodeValue factorTree = 53361 := by
  sorry


end NUMINAMATH_CALUDE_factor_tree_root_value_l4108_410820


namespace NUMINAMATH_CALUDE_revolution_volume_formula_l4108_410873

/-- Region P in the coordinate plane -/
def P : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |6 - p.1| + p.2 ≤ 8 ∧ 4 * p.2 - p.1 ≥ 20}

/-- The line around which P is revolved -/
def revolveLine (x y : ℝ) : Prop := 4 * y - x = 20

/-- The volume of the solid formed by revolving P around the line -/
noncomputable def revolutionVolume : ℝ := sorry

theorem revolution_volume_formula :
  revolutionVolume = 24 * Real.pi / (85 * Real.sqrt 3741) := by sorry

end NUMINAMATH_CALUDE_revolution_volume_formula_l4108_410873


namespace NUMINAMATH_CALUDE_train_speed_second_part_l4108_410810

/-- Proves that the speed of a train during the second part of a journey is 20 kmph,
    given specific conditions about the journey. -/
theorem train_speed_second_part 
  (x : ℝ) 
  (h_positive : x > 0) 
  (speed_first : ℝ) 
  (h_speed_first : speed_first = 40) 
  (distance_first : ℝ) 
  (h_distance_first : distance_first = x) 
  (distance_second : ℝ) 
  (h_distance_second : distance_second = 2 * x) 
  (distance_total : ℝ) 
  (h_distance_total : distance_total = 6 * x) 
  (speed_average : ℝ) 
  (h_speed_average : speed_average = 48) : 
  ∃ (speed_second : ℝ), speed_second = 20 := by
sorry


end NUMINAMATH_CALUDE_train_speed_second_part_l4108_410810


namespace NUMINAMATH_CALUDE_quadratic_root_value_l4108_410836

theorem quadratic_root_value (m : ℝ) : 
  ((m - 2) * 1^2 + 4 * 1 - m^2 = 0) ∧ (m ≠ 2) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l4108_410836


namespace NUMINAMATH_CALUDE_total_pictures_calculation_l4108_410847

/-- The number of pictures that can be contained in one album -/
def pictures_per_album : ℕ := 20

/-- The number of albums needed -/
def albums_needed : ℕ := 24

/-- The total number of pictures -/
def total_pictures : ℕ := pictures_per_album * albums_needed

theorem total_pictures_calculation :
  total_pictures = 480 :=
by sorry

end NUMINAMATH_CALUDE_total_pictures_calculation_l4108_410847


namespace NUMINAMATH_CALUDE_max_value_part_i_one_root_condition_part_ii_inequality_condition_part_iii_l4108_410852

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1
def g (x : ℝ) : ℝ := Real.exp x

-- Part I
theorem max_value_part_i :
  ∃ (M : ℝ), M = 1 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 0, (f 1 x) * (g x) ≤ M :=
sorry

-- Part II
theorem one_root_condition_part_ii :
  ∀ k : ℝ, (∃! x : ℝ, f (-1) x = k * g x) ↔ 
  (k > 0 ∧ k < Real.exp (-1)) ∨ (k > 3 * Real.exp (-2)) :=
sorry

-- Part III
theorem inequality_condition_part_iii :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → x₁ ≠ x₂ → 
    |f a x₁ - f a x₂| < |g x₁ - g x₂|) ↔ 
  (a ≥ -1 ∧ a ≤ 2 - 2 * Real.log 2) :=
sorry

end

end NUMINAMATH_CALUDE_max_value_part_i_one_root_condition_part_ii_inequality_condition_part_iii_l4108_410852


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l4108_410859

theorem function_satisfies_equation :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (x^2 + 2*x) = |x + 1| :=
by
  -- Define f(x) = √(x + 1)
  let f := λ x : ℝ ↦ Real.sqrt (x + 1)
  
  -- Prove that this f satisfies the equation
  -- for all x ∈ ℝ
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l4108_410859


namespace NUMINAMATH_CALUDE_quadrilateral_angle_measure_l4108_410802

theorem quadrilateral_angle_measure :
  ∀ (a b c d : ℝ),
  a = 50 →
  b = 180 - 30 →
  d = 180 - 40 →
  a + b + c + d = 360 →
  c = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_measure_l4108_410802


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l4108_410832

theorem integer_pairs_satisfying_equation :
  ∀ (x y : ℤ), x^2 = y^2 + 2*y + 13 ↔ (x = 4 ∧ y = 1) ∨ (x = -4 ∧ y = -5) := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l4108_410832


namespace NUMINAMATH_CALUDE_fraction_equality_l4108_410856

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 6) 
  (h2 : s / u = 7 / 18) : 
  (5 * p * s - 6 * q * u) / (7 * q * u - 10 * p * s) = -473 / 406 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4108_410856


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4108_410838

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = -(1/2 : ℂ) * (1 + Complex.I)) :
  Complex.im z = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4108_410838


namespace NUMINAMATH_CALUDE_ratio_problem_l4108_410887

theorem ratio_problem (q r s t u : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 3 / 4)
  (h4 : u / q = 1 / 2) :
  t / u = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l4108_410887


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l4108_410817

theorem correct_quadratic_equation 
  (b c : ℝ) 
  (h1 : 5 + 1 = -b) 
  (h2 : (-6) * (-4) = c) : 
  b = -10 ∧ c = 6 := by
sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l4108_410817


namespace NUMINAMATH_CALUDE_max_correct_answers_l4108_410850

/-- Represents an exam with a specific scoring system and result. -/
structure Exam where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Represents a possible breakdown of answers in an exam. -/
structure ExamResult where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ

/-- Checks if an ExamResult is valid for a given Exam. -/
def is_valid_result (e : Exam) (r : ExamResult) : Prop :=
  r.correct + r.incorrect + r.unanswered = e.total_questions ∧
  r.correct * e.correct_points + r.incorrect * e.incorrect_points = e.total_score

/-- Theorem: The maximum number of correct answers for the given exam is 33. -/
theorem max_correct_answers (e : Exam) :
  e.total_questions = 60 ∧ e.correct_points = 5 ∧ e.incorrect_points = -1 ∧ e.total_score = 140 →
  (∃ (r : ExamResult), is_valid_result e r ∧
    ∀ (r' : ExamResult), is_valid_result e r' → r'.correct ≤ r.correct) ∧
  (∃ (r : ExamResult), is_valid_result e r ∧ r.correct = 33) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l4108_410850


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l4108_410831

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 221 → ∀ a b : ℕ, a^2 - b^2 = 221 → x^2 + y^2 ≤ a^2 + b^2 → x^2 + y^2 = 229 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l4108_410831


namespace NUMINAMATH_CALUDE_complex_power_8_l4108_410840

theorem complex_power_8 :
  (3 * (Complex.cos (π / 6) + Complex.I * Complex.sin (π / 6)))^8 =
  -3280.5 - 3280.5 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_8_l4108_410840


namespace NUMINAMATH_CALUDE_grandchildren_gender_probability_l4108_410818

theorem grandchildren_gender_probability :
  let n : ℕ := 12  -- total number of grandchildren
  let p : ℚ := 1/2  -- probability of a grandchild being male (or female)
  let equal_prob := (n.choose (n/2)) / 2^n  -- probability of equal number of grandsons and granddaughters
  1 - equal_prob = 793/1024 := by
  sorry

end NUMINAMATH_CALUDE_grandchildren_gender_probability_l4108_410818


namespace NUMINAMATH_CALUDE_larger_number_problem_l4108_410804

theorem larger_number_problem (x y : ℕ) 
  (h1 : x * y = 40)
  (h2 : x + y = 13)
  (h3 : Even x ∨ Even y) :
  max x y = 8 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l4108_410804


namespace NUMINAMATH_CALUDE_area_not_unique_l4108_410898

/-- A land plot with a side of length 10 units -/
structure LandPlot where
  side : ℝ
  side_positive : side > 0

/-- The area of a land plot -/
noncomputable def area (plot : LandPlot) : ℝ := sorry

/-- Theorem: The area of a land plot cannot be uniquely determined given only the length of one side -/
theorem area_not_unique (plot1 plot2 : LandPlot) 
  (h : plot1.side = plot2.side) (h_side : plot1.side = 10) : 
  ¬ (∀ (p1 p2 : LandPlot), p1.side = p2.side → area p1 = area p2) := by
  sorry

end NUMINAMATH_CALUDE_area_not_unique_l4108_410898


namespace NUMINAMATH_CALUDE_min_value_expression_l4108_410808

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^4 + b^4 + 16 / (a^2 + b^2)^2 ≥ 4 ∧
  (a^4 + b^4 + 16 / (a^2 + b^2)^2 = 4 ↔ a = b ∧ a = 2^(1/4)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l4108_410808


namespace NUMINAMATH_CALUDE_heroes_on_front_l4108_410871

theorem heroes_on_front (total : ℕ) (back : ℕ) (front : ℕ) : 
  total = 9 → back = 7 → total = front + back → front = 2 := by
  sorry

end NUMINAMATH_CALUDE_heroes_on_front_l4108_410871


namespace NUMINAMATH_CALUDE_shells_per_friend_l4108_410822

/-- Given the number of shells collected by Jillian, Savannah, and Clayton,
    and the number of friends to distribute the shells to,
    prove that each friend receives 27 shells. -/
theorem shells_per_friend
  (jillian_shells : ℕ)
  (savannah_shells : ℕ)
  (clayton_shells : ℕ)
  (num_friends : ℕ)
  (h1 : jillian_shells = 29)
  (h2 : savannah_shells = 17)
  (h3 : clayton_shells = 8)
  (h4 : num_friends = 2) :
  (jillian_shells + savannah_shells + clayton_shells) / num_friends = 27 :=
by
  sorry

#check shells_per_friend

end NUMINAMATH_CALUDE_shells_per_friend_l4108_410822


namespace NUMINAMATH_CALUDE_harry_apples_l4108_410877

theorem harry_apples (martha_apples : ℕ) (tim_less : ℕ) (harry_ratio : ℕ) :
  martha_apples = 68 →
  tim_less = 30 →
  harry_ratio = 2 →
  (martha_apples - tim_less) / harry_ratio = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_harry_apples_l4108_410877


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l4108_410868

/-- The perimeter of a region formed by three identical touching circles -/
theorem shaded_region_perimeter (c : ℝ) (θ : ℝ) : 
  c > 0 → θ > 0 → θ < 2 * Real.pi →
  let r := c / (2 * Real.pi)
  let arc_length := θ / (2 * Real.pi) * c
  3 * arc_length = c →
  c = 48 → θ = 2 * Real.pi / 3 →
  3 * arc_length = 48 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l4108_410868


namespace NUMINAMATH_CALUDE_factory_production_difference_l4108_410854

/-- Represents the production rate and total products for a machine type -/
structure MachineType where
  rate : ℕ  -- products per minute
  total : ℕ -- total products made

/-- Calculates the difference in products between two machine types -/
def productDifference (a b : MachineType) : ℕ :=
  b.total - a.total

theorem factory_production_difference :
  let machineA : MachineType := { rate := 5, total := 25 }
  let machineB : MachineType := { rate := 8, total := 40 }
  productDifference machineA machineB = 15 := by
  sorry

#eval productDifference { rate := 5, total := 25 } { rate := 8, total := 40 }

end NUMINAMATH_CALUDE_factory_production_difference_l4108_410854


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_l4108_410867

def n : ℕ := 1936000000

def is_fifth_largest_divisor (d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (a b c e : ℕ), a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ e ∣ n ∧
    a > b ∧ b > c ∧ c > e ∧ e > d ∧
    ∀ (x : ℕ), x ∣ n → x ≤ d ∨ x = e ∨ x = c ∨ x = b ∨ x = a ∨ x = n)

theorem fifth_largest_divisor :
  is_fifth_largest_divisor 121000000 := by sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_l4108_410867


namespace NUMINAMATH_CALUDE_function_property_l4108_410883

theorem function_property (f : ℝ → ℝ) (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x + f y) 
  (h2 : f 8 = -3) : ∃ a : ℝ, a > 0 ∧ f a = 1/2 ∧ a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l4108_410883


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4108_410863

-- Define an isosceles triangle with sides a, b, and c
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)
  validTriangle : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter of a triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, 
  ((t.a = 8 ∧ t.b = 8 ∧ t.c = 4) ∨ (t.a = 8 ∧ t.b = 4 ∧ t.c = 8) ∨ (t.a = 4 ∧ t.b = 8 ∧ t.c = 8)) →
  perimeter t = 20 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4108_410863


namespace NUMINAMATH_CALUDE_graces_age_l4108_410884

/-- Grace's age problem -/
theorem graces_age (mother_age : ℕ) (grandmother_age : ℕ) (grace_age : ℕ) : 
  mother_age = 80 →
  grandmother_age = 2 * mother_age →
  grace_age = (3 * grandmother_age) / 8 →
  grace_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_graces_age_l4108_410884


namespace NUMINAMATH_CALUDE_product_of_polynomials_l4108_410864

theorem product_of_polynomials (g h : ℚ) : 
  (∀ d : ℚ, (7 * d^2 - 4 * d + g) * (3 * d^2 + h * d - 9) = 
    21 * d^4 - 49 * d^3 - 44 * d^2 + 17 * d - 24) → 
  g + h = -107/24 := by sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l4108_410864


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l4108_410816

theorem baseball_card_value_decrease : 
  ∀ (initial_value : ℝ), initial_value > 0 →
  let first_year_value := initial_value * (1 - 0.5)
  let second_year_value := first_year_value * (1 - 0.1)
  let total_decrease := (initial_value - second_year_value) / initial_value
  total_decrease = 0.55 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l4108_410816


namespace NUMINAMATH_CALUDE_attendance_rate_proof_l4108_410851

theorem attendance_rate_proof (total_students : ℕ) (absent_students : ℕ) :
  total_students = 50 →
  absent_students = 2 →
  (((total_students - absent_students) : ℚ) / total_students) * 100 = 96 := by
  sorry

end NUMINAMATH_CALUDE_attendance_rate_proof_l4108_410851


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l4108_410897

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 96 → volume = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l4108_410897


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l4108_410848

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l4108_410848


namespace NUMINAMATH_CALUDE_subtraction_problem_sum_l4108_410889

theorem subtraction_problem_sum (K L M N : ℕ) : 
  K < 10 → L < 10 → M < 10 → N < 10 →
  6000 + 100 * K + L - (900 + N) = 2011 →
  K + L + M + N = 17 := by
sorry

end NUMINAMATH_CALUDE_subtraction_problem_sum_l4108_410889


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l4108_410881

def n : ℕ := 97 * 101 * 104 * 107 * 109

theorem distinct_prime_factors_count : Nat.card (Nat.factors n).toFinset = 6 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l4108_410881


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_5pi_12_l4108_410829

theorem cos_2alpha_plus_5pi_12 (α : Real) (h1 : π < α ∧ α < 2*π) 
  (h2 : Real.sin (α + π/3) = -4/5) : 
  Real.cos (2*α + 5*π/12) = 17*Real.sqrt 2/50 := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_5pi_12_l4108_410829


namespace NUMINAMATH_CALUDE_constant_term_expansion_l4108_410846

/-- The constant term in the expansion of (x^2 + 1/x^3)^5 -/
def constant_term : ℕ := 10

/-- The binomial coefficient C(5,2) -/
def C_5_2 : ℕ := Nat.choose 5 2

theorem constant_term_expansion :
  constant_term = C_5_2 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l4108_410846


namespace NUMINAMATH_CALUDE_divisibility_by_13_l4108_410800

theorem divisibility_by_13 (N : ℕ) (x : ℕ) : 
  (N = 2 * 10^2022 + x * 10^2000 + 23) →
  (N % 13 = 0) →
  (x = 3) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_13_l4108_410800


namespace NUMINAMATH_CALUDE_locus_of_intersection_point_l4108_410837

/-- The locus of the intersection point of two rotating lines in a triangle --/
theorem locus_of_intersection_point (d e : ℝ) (h1 : d ≠ 0) (h2 : e ≠ 0) :
  ∃ (f : ℝ → ℝ × ℝ),
    (∀ t, ∃ (m : ℝ),
      (f t).1 = -2 * e / m ∧
      (f t).2 = -m * d) ∧
    (∀ x y, (x, y) ∈ Set.range f ↔ x * y = d * e) :=
sorry

end NUMINAMATH_CALUDE_locus_of_intersection_point_l4108_410837


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4108_410862

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, 0 < x ∧ x < 1 → 0 < x^2 ∧ x^2 < 1) ∧
  (∃ x, 0 < x^2 ∧ x^2 < 1 ∧ ¬(0 < x ∧ x < 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4108_410862


namespace NUMINAMATH_CALUDE_unique_total_prices_l4108_410879

def gift_prices : List ℕ := [2, 5, 8, 11, 14]
def box_prices : List ℕ := [3, 5, 7, 9, 11]

def total_prices : List ℕ :=
  List.eraseDups (List.map (λ (p : ℕ × ℕ) => p.1 + p.2) (List.product gift_prices box_prices))

theorem unique_total_prices :
  total_prices.length = 19 := by sorry

end NUMINAMATH_CALUDE_unique_total_prices_l4108_410879


namespace NUMINAMATH_CALUDE_rectangle_perimeter_equals_20_l4108_410861

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rectangle with width w and length l -/
structure Rectangle where
  w : ℝ
  l : ℝ

/-- Area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := 
  sorry

/-- Area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ :=
  r.w * r.l

/-- Perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.w + r.l)

theorem rectangle_perimeter_equals_20 (t : Triangle) (r : Rectangle) :
  t.a = 6 ∧ t.b = 8 ∧ t.c = 10 ∧ r.w = 4 ∧ Triangle.area t = Rectangle.area r →
  Rectangle.perimeter r = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_equals_20_l4108_410861
