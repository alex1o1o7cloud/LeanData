import Mathlib

namespace NUMINAMATH_CALUDE_angle_B_obtuse_l3939_393925

theorem angle_B_obtuse (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- Triangle inequality
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  -- Given conditions
  (c / b < Real.cos A) ∧ (0 < A) ∧ (A < Real.pi) →
  -- Conclusion: B is obtuse
  Real.pi / 2 < B :=
by
  sorry

end NUMINAMATH_CALUDE_angle_B_obtuse_l3939_393925


namespace NUMINAMATH_CALUDE_circle_and_quadratic_inequality_l3939_393946

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 - 2*a*x + y^2 + 2*a^2 - 5*a + 4 = 0

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + (a-1)*x + 1 > 0

-- Theorem statement
theorem circle_and_quadratic_inequality (a : ℝ) :
  p a ∧ q a → 1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_and_quadratic_inequality_l3939_393946


namespace NUMINAMATH_CALUDE_ages_sum_l3939_393983

theorem ages_sum (a b c : ℕ+) : 
  b = c →                 -- twins have the same age
  b > a →                 -- twins are older than Kiana
  a * b * c = 144 →       -- product of ages is 144
  a + b + c = 16 :=       -- sum of ages is 16
by sorry

end NUMINAMATH_CALUDE_ages_sum_l3939_393983


namespace NUMINAMATH_CALUDE_fan_ratio_theorem_l3939_393986

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The ratio of NY Yankees fans to NY Mets fans is 3:2 -/
def yankees_mets_ratio (fc : FanCounts) : Prop :=
  fc.yankees * 2 = fc.mets * 3

/-- The total number of fans is 390 -/
def total_fans (fc : FanCounts) : Prop :=
  fc.yankees + fc.mets + fc.red_sox = 390

/-- There are 104 NY Mets fans -/
def mets_fans_count (fc : FanCounts) : Prop :=
  fc.mets = 104

/-- The ratio of NY Mets fans to Boston Red Sox fans is 4:5 -/
def mets_red_sox_ratio (fc : FanCounts) : Prop :=
  fc.mets * 5 = fc.red_sox * 4

theorem fan_ratio_theorem (fc : FanCounts) :
  yankees_mets_ratio fc → total_fans fc → mets_fans_count fc → mets_red_sox_ratio fc := by
  sorry

end NUMINAMATH_CALUDE_fan_ratio_theorem_l3939_393986


namespace NUMINAMATH_CALUDE_range_of_a_l3939_393914

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3939_393914


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_geq_2_l3939_393908

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 3

-- Define the property of f being decreasing on (-∞, 2)
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → x < 2 → y < 2 → f a x > f a y

-- Theorem statement
theorem f_decreasing_implies_a_geq_2 :
  ∀ a : ℝ, is_decreasing_on_interval a → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_geq_2_l3939_393908


namespace NUMINAMATH_CALUDE_snake_sale_amount_l3939_393954

/-- Given Gary's initial and final amounts, calculate the amount he received from selling his pet snake. -/
theorem snake_sale_amount (initial_amount final_amount : ℝ) 
  (h1 : initial_amount = 73.0)
  (h2 : final_amount = 128) :
  final_amount - initial_amount = 55 := by
  sorry

end NUMINAMATH_CALUDE_snake_sale_amount_l3939_393954


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l3939_393943

theorem geometric_arithmetic_sequence_problem 
  (b q a d : ℝ) 
  (h1 : b = a + d)
  (h2 : b * q = a + 3 * d)
  (h3 : b * q^2 = a + 6 * d)
  (h4 : b * (b * q) * (b * q^2) = 64) :
  b = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l3939_393943


namespace NUMINAMATH_CALUDE_max_leap_years_in_period_l3939_393924

/-- The number of years in the period -/
def period : ℕ := 125

/-- The interval between leap years -/
def leap_year_interval : ℕ := 5

/-- The maximum number of leap years in the given period -/
def max_leap_years : ℕ := period / leap_year_interval

theorem max_leap_years_in_period :
  max_leap_years = 25 := by sorry

end NUMINAMATH_CALUDE_max_leap_years_in_period_l3939_393924


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3939_393973

/-- Given two arithmetic sequences {a_n} and {b_n} with S_n and T_n as the sum of their first n terms respectively -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2 ∧ T n = (n * (b 1 + b n)) / 2

theorem arithmetic_sequence_ratio
  (a b : ℕ → ℚ) (S T : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequences a b S T)
  (h_ratio : ∀ n, S n / T n = (7 * n + 1) / (n + 3)) :
  (a 2 + a 5 + a 17 + a 22) / (b 8 + b 10 + b 12 + b 16) = 31 / 5 ∧
  a 5 / b 5 = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3939_393973


namespace NUMINAMATH_CALUDE_f_max_value_l3939_393951

/-- The quadratic function f(y) = -3y^2 + 18y - 7 -/
def f (y : ℝ) : ℝ := -3 * y^2 + 18 * y - 7

/-- The maximum value of f(y) is 20 -/
theorem f_max_value : ∃ (M : ℝ), M = 20 ∧ ∀ (y : ℝ), f y ≤ M := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l3939_393951


namespace NUMINAMATH_CALUDE_sqrt_twelve_div_sqrt_two_eq_sqrt_six_l3939_393901

theorem sqrt_twelve_div_sqrt_two_eq_sqrt_six :
  Real.sqrt 12 / Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_div_sqrt_two_eq_sqrt_six_l3939_393901


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l3939_393993

theorem smallest_positive_solution :
  ∃! (x y : ℝ), x > 0 ∧ y > 0 ∧
  x^2 - 3*x + 2.5 = Real.sin y - 0.75 ∧
  (∀ (x' y' : ℝ), x' > 0 → y' > 0 → x'^2 - 3*x' + 2.5 = Real.sin y' - 0.75 → x ≤ x' ∧ y ≤ y') ∧
  x = 3/2 ∧ y = Real.pi/2 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l3939_393993


namespace NUMINAMATH_CALUDE_point_on_line_l3939_393930

/-- Given a line passing through points (3, 6) and (-4, 0), 
    prove that if (x, 10) lies on this line, then x = 23/3 -/
theorem point_on_line (x : ℚ) : 
  (∀ (t : ℚ), (3 + t * (-4 - 3), 6 + t * (0 - 6)) = (x, 10)) → x = 23 / 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l3939_393930


namespace NUMINAMATH_CALUDE_calculation_proof_l3939_393916

theorem calculation_proof : (-8 - 1/3) - 12 - (-70) - (-8 - 1/3) = 58 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3939_393916


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l3939_393944

theorem complex_pure_imaginary (m : ℝ) : 
  (m^2 - 4 + (m + 2)*Complex.I = 0) → m = 2 :=
sorry


end NUMINAMATH_CALUDE_complex_pure_imaginary_l3939_393944


namespace NUMINAMATH_CALUDE_circle_area_equal_square_perimeter_l3939_393906

theorem circle_area_equal_square_perimeter (square_area : ℝ) (h : square_area = 121) :
  let square_side : ℝ := Real.sqrt square_area
  let square_perimeter : ℝ := 4 * square_side
  let circle_radius : ℝ := square_perimeter / (2 * Real.pi)
  let circle_area : ℝ := Real.pi * circle_radius ^ 2
  circle_area = 484 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_area_equal_square_perimeter_l3939_393906


namespace NUMINAMATH_CALUDE_delta_value_l3939_393963

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ - 3 → Δ = -9 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l3939_393963


namespace NUMINAMATH_CALUDE_third_runner_time_l3939_393974

/-- A relay race with 4 runners -/
structure RelayRace where
  mary : ℝ
  susan : ℝ
  third : ℝ
  tiffany : ℝ

/-- The conditions of the relay race -/
def validRelayRace (race : RelayRace) : Prop :=
  race.mary = 2 * race.susan ∧
  race.susan = race.third + 10 ∧
  race.tiffany = race.mary - 7 ∧
  race.mary + race.susan + race.third + race.tiffany = 223

/-- The theorem stating that the third runner's time is 30 seconds -/
theorem third_runner_time (race : RelayRace) 
  (h : validRelayRace race) : race.third = 30 := by
  sorry

end NUMINAMATH_CALUDE_third_runner_time_l3939_393974


namespace NUMINAMATH_CALUDE_cost_of_six_pens_l3939_393936

/-- Given that 3 pens cost 7.5 yuan, prove that 6 pens cost 15 yuan. -/
theorem cost_of_six_pens (cost_three_pens : ℝ) (h : cost_three_pens = 7.5) :
  let cost_one_pen := cost_three_pens / 3
  cost_one_pen * 6 = 15 := by sorry

end NUMINAMATH_CALUDE_cost_of_six_pens_l3939_393936


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3939_393952

theorem partial_fraction_decomposition 
  (a b c d : ℤ) (h : a * d ≠ b * c) :
  ∃ (r s : ℝ), ∀ (x : ℝ), 
    1 / ((a * x + b) * (c * x + d)) = 
    r / (a * x + b) + s / (c * x + d) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3939_393952


namespace NUMINAMATH_CALUDE_mary_fruits_left_l3939_393979

/-- Calculates the total number of fruits left after eating some. -/
def fruits_left (initial_apples initial_oranges initial_blueberries eaten : ℕ) : ℕ :=
  (initial_apples - eaten) + (initial_oranges - eaten) + (initial_blueberries - eaten)

/-- Proves that Mary has 26 fruits left after eating one of each. -/
theorem mary_fruits_left : fruits_left 14 9 6 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruits_left_l3939_393979


namespace NUMINAMATH_CALUDE_remaining_etching_price_l3939_393915

def total_etchings : ℕ := 16
def total_revenue : ℕ := 630
def first_batch_count : ℕ := 9
def first_batch_price : ℕ := 35

theorem remaining_etching_price :
  (total_revenue - first_batch_count * first_batch_price) / (total_etchings - first_batch_count) = 45 := by
  sorry

end NUMINAMATH_CALUDE_remaining_etching_price_l3939_393915


namespace NUMINAMATH_CALUDE_merchant_discount_l3939_393987

theorem merchant_discount (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * 1.2
  let final_price := increased_price * 0.8
  let actual_discount := (original_price - final_price) / original_price
  actual_discount = 0.04 := by
sorry

end NUMINAMATH_CALUDE_merchant_discount_l3939_393987


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3939_393902

theorem smallest_part_of_proportional_division (total : ℝ) (p1 p2 p3 : ℝ) :
  total = 105 →
  p1 + p2 + p3 = total →
  p1 / 2 = p2 / (1/2) →
  p1 / 2 = p3 / (1/4) →
  min p1 (min p2 p3) = 10.5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3939_393902


namespace NUMINAMATH_CALUDE_new_students_admitted_l3939_393984

theorem new_students_admitted (initial_students_per_section : ℕ) 
                               (new_sections : ℕ)
                               (final_total_sections : ℕ)
                               (final_students_per_section : ℕ) :
  initial_students_per_section = 23 →
  new_sections = 5 →
  final_total_sections = 20 →
  final_students_per_section = 19 →
  (final_total_sections * final_students_per_section) - 
  ((final_total_sections - new_sections) * initial_students_per_section) = 35 :=
by sorry

end NUMINAMATH_CALUDE_new_students_admitted_l3939_393984


namespace NUMINAMATH_CALUDE_train_speed_fraction_l3939_393968

theorem train_speed_fraction (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 50.000000000000014 →
  delay = 10 →
  (usual_time / (usual_time + delay)) = (5 : ℝ) / 6 := by
sorry

end NUMINAMATH_CALUDE_train_speed_fraction_l3939_393968


namespace NUMINAMATH_CALUDE_minutes_worked_yesterday_l3939_393911

/-- The number of shirts made by the machine yesterday -/
def shirts_made_yesterday : ℕ := 9

/-- The number of shirts the machine can make per minute -/
def shirts_per_minute : ℕ := 3

/-- Theorem: The number of minutes the machine worked yesterday is 3 -/
theorem minutes_worked_yesterday : 
  shirts_made_yesterday / shirts_per_minute = 3 := by
  sorry

end NUMINAMATH_CALUDE_minutes_worked_yesterday_l3939_393911


namespace NUMINAMATH_CALUDE_hikers_room_arrangements_l3939_393999

theorem hikers_room_arrangements (n : ℕ) (k : ℕ) : n = 5 → k = 3 → Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_hikers_room_arrangements_l3939_393999


namespace NUMINAMATH_CALUDE_classroom_population_classroom_population_is_8_l3939_393961

theorem classroom_population : ℕ :=
  let student_count : ℕ := sorry
  let student_avg_age : ℚ := 8
  let total_avg_age : ℚ := 11
  let teacher_age : ℕ := 32

  have h1 : (student_count * student_avg_age + teacher_age) / (student_count + 1) = total_avg_age := by sorry

  student_count + 1

theorem classroom_population_is_8 : classroom_population = 8 := by sorry

end NUMINAMATH_CALUDE_classroom_population_classroom_population_is_8_l3939_393961


namespace NUMINAMATH_CALUDE_cube_third_yellow_faces_l3939_393932

/-- Represents a cube with side length n -/
structure Cube where
  n : ℕ

/-- Total number of faces of unit cubes after division -/
def totalFaces (c : Cube) : ℕ := 6 * c.n^3

/-- Number of yellow faces after division -/
def yellowFaces (c : Cube) : ℕ := 6 * c.n^2

/-- Condition for exactly one-third of faces being yellow -/
def oneThirdYellow (c : Cube) : Prop :=
  3 * yellowFaces c = totalFaces c

/-- Theorem stating that n = 3 satisfies the condition -/
theorem cube_third_yellow_faces :
  ∃ (c : Cube), c.n = 3 ∧ oneThirdYellow c :=
sorry

end NUMINAMATH_CALUDE_cube_third_yellow_faces_l3939_393932


namespace NUMINAMATH_CALUDE_dice_product_nonzero_probability_l3939_393976

/-- The probability of getting a specific outcome when rolling a standard die -/
def roll_probability : ℚ := 1 / 6

/-- The number of faces on a standard die -/
def die_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The probability that a single die roll is not 1 -/
def prob_not_one : ℚ := (die_faces - 1) / die_faces

theorem dice_product_nonzero_probability :
  (prob_not_one ^ num_dice : ℚ) = 625 / 1296 := by sorry

end NUMINAMATH_CALUDE_dice_product_nonzero_probability_l3939_393976


namespace NUMINAMATH_CALUDE_f_zero_gt_f_one_l3939_393910

/-- A quadratic function f(x) = x^2 - 4x + m, where m is a real constant. -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + m

/-- Theorem stating that f(0) > f(1) for any real m. -/
theorem f_zero_gt_f_one (m : ℝ) : f m 0 > f m 1 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_gt_f_one_l3939_393910


namespace NUMINAMATH_CALUDE_three_digit_reversal_difference_l3939_393958

theorem three_digit_reversal_difference (A B C : ℕ) 
  (h1 : A ≠ C) 
  (h2 : A ≥ 1 ∧ A ≤ 9) 
  (h3 : B ≥ 0 ∧ B ≤ 9) 
  (h4 : C ≥ 0 ∧ C ≤ 9) : 
  ∃ k : ℤ, (100 * A + 10 * B + C) - (100 * C + 10 * B + A) = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_three_digit_reversal_difference_l3939_393958


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l3939_393949

/-- Given 5 consecutive integers whose sum is 120, their product is 7893600 -/
theorem consecutive_integers_product (x : ℤ) 
  (h1 : (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 120) : 
  (x - 2) * (x - 1) * x * (x + 1) * (x + 2) = 7893600 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l3939_393949


namespace NUMINAMATH_CALUDE_jerry_average_study_time_difference_l3939_393903

def daily_differences : List Int := [15, -5, 25, 0, -15, 10]

def extra_study_time : Int := 20

def adjust_difference (diff : Int) : Int :=
  if diff > 0 then diff + extra_study_time else diff

theorem jerry_average_study_time_difference :
  let adjusted_differences := daily_differences.map adjust_difference
  let total_difference := adjusted_differences.sum
  let num_days := daily_differences.length
  total_difference / num_days = -15 := by sorry

end NUMINAMATH_CALUDE_jerry_average_study_time_difference_l3939_393903


namespace NUMINAMATH_CALUDE_total_distance_traveled_l3939_393972

/-- Calculates the total distance traveled by a man rowing upstream and downstream in a river. -/
theorem total_distance_traveled
  (man_speed : ℝ)
  (river_speed : ℝ)
  (total_time : ℝ)
  (h1 : man_speed = 6)
  (h2 : river_speed = 1.2)
  (h3 : total_time = 1)
  : ∃ (distance : ℝ), distance = 5.76 ∧ 
    (distance / (man_speed - river_speed) + distance / (man_speed + river_speed) = total_time) :=
by sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l3939_393972


namespace NUMINAMATH_CALUDE_new_person_weight_l3939_393923

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℚ) (replaced_weight : ℚ) : 
  initial_count = 8 →
  weight_increase = 7/2 →
  replaced_weight = 62 →
  (initial_count : ℚ) * weight_increase + replaced_weight = 90 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3939_393923


namespace NUMINAMATH_CALUDE_rectangular_field_with_pond_l3939_393921

theorem rectangular_field_with_pond (l w : ℝ) : 
  l = 2 * w →                        -- length is double the width
  l * w = 2 * (8 * 8) →              -- pond area is half of field area
  l = 16 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_with_pond_l3939_393921


namespace NUMINAMATH_CALUDE_fold_symmetry_l3939_393913

-- Define the fold line
def fold_line (x y : ℝ) : Prop := y = 2 * x

-- Define the symmetric point
def symmetric_point (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  fold_line ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) ∧
  (x₂ - x₁) = (y₂ - y₁) / 2

-- Define the perpendicular bisector property
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  fold_line ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)

-- Theorem statement
theorem fold_symmetry :
  perpendicular_bisector 10 0 (-6) 8 →
  symmetric_point (-4) 2 4 (-2) :=
sorry

end NUMINAMATH_CALUDE_fold_symmetry_l3939_393913


namespace NUMINAMATH_CALUDE_crate_height_difference_is_zero_l3939_393934

/-- Represents a cylindrical pipe -/
structure Pipe where
  diameter : ℝ

/-- Represents a crate filled with pipes -/
structure Crate where
  pipes : List Pipe
  stackingPattern : String

/-- Calculate the height of a crate -/
def calculateCrateHeight (c : Crate) : ℝ := sorry

/-- The main theorem statement -/
theorem crate_height_difference_is_zero 
  (pipeA pipeB : Pipe)
  (crateA crateB : Crate)
  (h1 : pipeA.diameter = 15)
  (h2 : pipeB.diameter = 15)
  (h3 : crateA.pipes.length = 150)
  (h4 : crateB.pipes.length = 150)
  (h5 : crateA.stackingPattern = "triangular")
  (h6 : crateB.stackingPattern = "inverted triangular")
  (h7 : ∀ p ∈ crateA.pipes, p = pipeA)
  (h8 : ∀ p ∈ crateB.pipes, p = pipeB) :
  |calculateCrateHeight crateA - calculateCrateHeight crateB| = 0 := by
  sorry

end NUMINAMATH_CALUDE_crate_height_difference_is_zero_l3939_393934


namespace NUMINAMATH_CALUDE_lunks_for_apples_l3939_393975

/-- Exchange rate between lunks and kunks -/
def lunk_to_kunk_rate : ℚ := 6 / 4

/-- Exchange rate between kunks and apples -/
def kunk_to_apple_rate : ℚ := 5 / 3

/-- Number of apples to purchase -/
def apples_to_buy : ℕ := 20

/-- Calculate the number of lunks needed to buy a given number of apples -/
def lunks_needed (apples : ℕ) : ℚ :=
  (apples : ℚ) / kunk_to_apple_rate / lunk_to_kunk_rate

theorem lunks_for_apples :
  lunks_needed apples_to_buy = 8 := by
  sorry

end NUMINAMATH_CALUDE_lunks_for_apples_l3939_393975


namespace NUMINAMATH_CALUDE_probability_not_blue_l3939_393956

def odds_blue : ℚ := 5 / 6

theorem probability_not_blue (odds : ℚ) (h : odds = odds_blue) :
  1 - odds / (1 + odds) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_blue_l3939_393956


namespace NUMINAMATH_CALUDE_max_distance_for_given_tires_l3939_393933

/-- Represents the maximum distance a car can travel with tire swapping -/
def maxDistanceWithSwap (frontTireLife : ℕ) (rearTireLife : ℕ) : ℕ :=
  frontTireLife + (rearTireLife - frontTireLife) / 2

/-- Theorem stating the maximum distance for the given tire lives -/
theorem max_distance_for_given_tires :
  maxDistanceWithSwap 42000 56000 = 48000 := by
  sorry

#eval maxDistanceWithSwap 42000 56000

end NUMINAMATH_CALUDE_max_distance_for_given_tires_l3939_393933


namespace NUMINAMATH_CALUDE_cone_surface_area_l3939_393931

/-- The surface area of a cone with base radius 1 and height √3 is 3π. -/
theorem cone_surface_area :
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let surface_area : ℝ := π * r^2 + π * r * l
  surface_area = 3 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l3939_393931


namespace NUMINAMATH_CALUDE_trapezium_area_with_triangle_removed_l3939_393905

/-- The area of a trapezium with a right triangle removed -/
theorem trapezium_area_with_triangle_removed
  (e f g h : ℝ)
  (e_pos : 0 < e)
  (f_pos : 0 < f)
  (g_pos : 0 < g)
  (h_pos : 0 < h) :
  let trapezium_area := (e + f) * (g + h)
  let triangle_area := h^2 / 2
  trapezium_area - triangle_area = (e + f) * (g + h) - h^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_area_with_triangle_removed_l3939_393905


namespace NUMINAMATH_CALUDE_die_roll_probabilities_l3939_393990

def DieFaces := Finset.range 6

def roll_twice : Finset (ℕ × ℕ) :=
  DieFaces.product DieFaces

theorem die_roll_probabilities :
  let total_outcomes := (roll_twice.card : ℚ)
  let sum_at_least_nine := (roll_twice.filter (fun (a, b) => a + b ≥ 9)).card
  let tangent_to_circle := (roll_twice.filter (fun (a, b) => a^2 + b^2 = 25)).card
  let isosceles_triangle := (roll_twice.filter (fun (a, b) => 
    a = b ∨ a = 5 ∨ b = 5)).card
  (sum_at_least_nine : ℚ) / total_outcomes = 5 / 18 ∧
  (tangent_to_circle : ℚ) / total_outcomes = 1 / 18 ∧
  (isosceles_triangle : ℚ) / total_outcomes = 7 / 18 :=
by sorry

end NUMINAMATH_CALUDE_die_roll_probabilities_l3939_393990


namespace NUMINAMATH_CALUDE_christopher_alexander_difference_l3939_393971

/-- Represents the number of joggers bought by each person -/
structure JoggerPurchases where
  christopher : Nat
  tyson : Nat
  alexander : Nat

/-- The conditions of the jogger purchase problem -/
def jogger_problem (purchases : JoggerPurchases) : Prop :=
  purchases.christopher = 80 ∧
  purchases.christopher = 20 * purchases.tyson ∧
  purchases.alexander = purchases.tyson + 22

/-- The theorem to be proved -/
theorem christopher_alexander_difference 
  (purchases : JoggerPurchases) 
  (h : jogger_problem purchases) : 
  purchases.christopher - purchases.alexander = 54 := by
  sorry

end NUMINAMATH_CALUDE_christopher_alexander_difference_l3939_393971


namespace NUMINAMATH_CALUDE_train_length_calculation_l3939_393939

/-- Given two trains of equal length running on parallel lines in the same direction,
    this theorem proves the length of each train given their speeds and passing time. -/
theorem train_length_calculation (v_fast v_slow : ℝ) (t : ℝ) (L : ℝ) :
  v_fast = 50 →
  v_slow = 36 →
  t = 36 / 3600 →
  (v_fast - v_slow) * t = 2 * L →
  L = 0.07 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3939_393939


namespace NUMINAMATH_CALUDE_cake_sugar_calculation_l3939_393955

theorem cake_sugar_calculation (frosting_sugar cake_sugar : ℝ) 
  (h1 : frosting_sugar = 0.6) 
  (h2 : cake_sugar = 0.2) : 
  frosting_sugar + cake_sugar = 0.8 := by
sorry

end NUMINAMATH_CALUDE_cake_sugar_calculation_l3939_393955


namespace NUMINAMATH_CALUDE_sum_of_products_l3939_393970

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 390) 
  (h2 : a + b + c = 20) : 
  a*b + b*c + c*a = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l3939_393970


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_20_l3939_393938

theorem smallest_n_divisible_by_20 :
  ∃ (n : ℕ), n ≥ 4 ∧ n = 9 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < 9 →
    ∃ (S : Finset ℤ), S.card = m ∧
    ∀ (a b c d : ℤ), a ∈ S → b ∈ S → c ∈ S → d ∈ S →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    ¬(20 ∣ (a + b - c - d))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_20_l3939_393938


namespace NUMINAMATH_CALUDE_employee_salary_problem_l3939_393982

/-- Proves that given 20 employees, if adding a manager's salary of 3400
    increases the average salary by 100, then the initial average salary
    of the employees is 1300. -/
theorem employee_salary_problem (n : ℕ) (manager_salary : ℕ) (salary_increase : ℕ) 
    (h1 : n = 20)
    (h2 : manager_salary = 3400)
    (h3 : salary_increase = 100) :
    ∃ (initial_avg : ℕ),
      initial_avg * n + manager_salary = (initial_avg + salary_increase) * (n + 1) ∧
      initial_avg = 1300 := by
  sorry

end NUMINAMATH_CALUDE_employee_salary_problem_l3939_393982


namespace NUMINAMATH_CALUDE_program_result_l3939_393962

/-- The program's operation on input n -/
def program (n : ℝ) : ℝ := n^2 + 3*n - (2*n^2 - n)

/-- Theorem stating that the program's result equals -n^2 + 4n for any real n -/
theorem program_result (n : ℝ) : program n = -n^2 + 4*n := by
  sorry

end NUMINAMATH_CALUDE_program_result_l3939_393962


namespace NUMINAMATH_CALUDE_wire_length_theorem_l3939_393942

/-- Represents the wire and pole configuration -/
structure WireConfig where
  initial_poles : ℕ
  initial_distance : ℝ
  new_distance_increase : ℝ
  total_length : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem wire_length_theorem (config : WireConfig) 
  (h1 : config.initial_poles = 26)
  (h2 : config.new_distance_increase = 5/3)
  (h3 : (config.initial_poles - 1) * (config.initial_distance + config.new_distance_increase) = config.initial_poles * config.initial_distance - config.initial_distance) :
  config.total_length = 1000 := by
  sorry


end NUMINAMATH_CALUDE_wire_length_theorem_l3939_393942


namespace NUMINAMATH_CALUDE_train_length_l3939_393948

/-- Given a train that crosses a bridge and passes a lamp post, calculate its length -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ)
  (h1 : bridge_length = 800)
  (h2 : bridge_time = 45)
  (h3 : post_time = 15) :
  ∃ train_length : ℝ, train_length = 400 ∧ 
  train_length / post_time = (train_length + bridge_length) / bridge_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3939_393948


namespace NUMINAMATH_CALUDE_f_min_max_l3939_393977

def f (x : ℝ) : ℝ := 2 * x^2 - 6 * x + 1

theorem f_min_max :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≥ -3) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = -3) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 9) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 9) :=
by sorry

end NUMINAMATH_CALUDE_f_min_max_l3939_393977


namespace NUMINAMATH_CALUDE_modified_system_solution_l3939_393967

theorem modified_system_solution
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ)
  (h₁ : a₁ * 4 + b₁ * 6 = c₁)
  (h₂ : a₂ * 4 + b₂ * 6 = c₂) :
  ∃ (x y : ℝ), x = 5 ∧ y = 10 ∧ 4 * a₁ * x + 3 * b₁ * y = 5 * c₁ ∧ 4 * a₂ * x + 3 * b₂ * y = 5 * c₂ :=
by sorry

end NUMINAMATH_CALUDE_modified_system_solution_l3939_393967


namespace NUMINAMATH_CALUDE_floor_sum_difference_l3939_393997

theorem floor_sum_difference (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ⌊a + b⌋ - (⌊a⌋ + ⌊b⌋) = 0 ∨ ⌊a + b⌋ - (⌊a⌋ + ⌊b⌋) = 1 :=
by sorry

end NUMINAMATH_CALUDE_floor_sum_difference_l3939_393997


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l3939_393996

/-- Given two circles in the plane, this theorem states that the line passing through their
    intersection points has a specific equation. -/
theorem intersection_line_of_circles
  (circle1 : Set (ℝ × ℝ))
  (circle2 : Set (ℝ × ℝ))
  (h1 : circle1 = {(x, y) | x^2 + y^2 - 4*x + 6*y = 0})
  (h2 : circle2 = {(x, y) | x^2 + y^2 - 6*x = 0})
  (h3 : (circle1 ∩ circle2).Nonempty) :
  ∃ (A B : ℝ × ℝ),
    A ∈ circle1 ∧ A ∈ circle2 ∧
    B ∈ circle1 ∧ B ∈ circle2 ∧
    A ≠ B ∧
    (∀ (x y : ℝ), (x, y) ∈ Set.Icc A B → x + 3*y = 0) :=
sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l3939_393996


namespace NUMINAMATH_CALUDE_fraction_equality_l3939_393912

theorem fraction_equality (a b : ℚ) (h1 : 2 * a = 3 * b) (h2 : b ≠ 0) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3939_393912


namespace NUMINAMATH_CALUDE_pet_store_combinations_l3939_393969

theorem pet_store_combinations (puppies kittens hamsters : ℕ) 
  (h1 : puppies = 20) (h2 : kittens = 9) (h3 : hamsters = 12) :
  (puppies * kittens * hamsters) * 6 = 12960 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l3939_393969


namespace NUMINAMATH_CALUDE_equation_solution_l3939_393960

theorem equation_solution : ∃! x : ℝ, (x^2 + x)^2 + Real.sqrt (x^2 - 1) = 0 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3939_393960


namespace NUMINAMATH_CALUDE_C_is_circle_when_k_is_2_C_hyperbola_y_axis_implies_k_less_than_neg_one_l3939_393957

-- Define the curve C
def C (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 1) + y^2 / (5 - k) = 1

-- Theorem 1: When k=2, curve C is a circle
theorem C_is_circle_when_k_is_2 :
  ∃ (r : ℝ), ∀ (x y : ℝ), C 2 x y ↔ x^2 + y^2 = r^2 :=
sorry

-- Theorem 2: If curve C is a hyperbola with foci on the y-axis, then k < -1
theorem C_hyperbola_y_axis_implies_k_less_than_neg_one :
  (∃ (a b : ℝ), ∀ (x y : ℝ), C k x y ↔ y^2 / a^2 - x^2 / b^2 = 1) → k < -1 :=
sorry

end NUMINAMATH_CALUDE_C_is_circle_when_k_is_2_C_hyperbola_y_axis_implies_k_less_than_neg_one_l3939_393957


namespace NUMINAMATH_CALUDE_function_value_range_l3939_393988

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * a + 1

-- State the theorem
theorem function_value_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ 1 ∧ -1 ≤ x₂ ∧ x₂ ≤ 1 ∧ f a x₁ < 0 ∧ 0 < f a x₂) →
  -1 < a ∧ a < -1/3 := by
  sorry


end NUMINAMATH_CALUDE_function_value_range_l3939_393988


namespace NUMINAMATH_CALUDE_polygon_diagonals_l3939_393941

theorem polygon_diagonals (n : ℕ) (h : n > 2) :
  (360 / (360 / n) : ℚ) - 3 = 9 :=
sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l3939_393941


namespace NUMINAMATH_CALUDE_f_extrema_and_roots_l3939_393926

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - (x + 1)^2

theorem f_extrema_and_roots :
  (∃ x₀ ∈ Set.Icc (-1 : ℝ) 2, ∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≥ f x₀ ∧ f x₀ = -(Real.log 2)^2 - 1) ∧
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≤ f x₁ ∧ f x₁ = 2 * Real.exp 2 - 9) ∧
  (∀ a < -1, (∃! x, f x = a * x - 1)) ∧
  (∀ a > -1, (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = a * x₁ - 1 ∧ f x₂ = a * x₂ - 1 ∧ f x₃ = a * x₃ - 1)) :=
by sorry


end NUMINAMATH_CALUDE_f_extrema_and_roots_l3939_393926


namespace NUMINAMATH_CALUDE_max_abs_z_l3939_393909

theorem max_abs_z (z : ℂ) (h : Complex.abs (z - (0 : ℂ) + 2 * Complex.I) = 1) :
  ∃ (M : ℝ), M = 3 ∧ ∀ w : ℂ, Complex.abs (w - (0 : ℂ) + 2 * Complex.I) = 1 → Complex.abs w ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_l3939_393909


namespace NUMINAMATH_CALUDE_january_savings_l3939_393935

def savings_sequence (initial : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => 2 * savings_sequence initial n

theorem january_savings (initial : ℕ) :
  savings_sequence initial 4 = 160 → initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_january_savings_l3939_393935


namespace NUMINAMATH_CALUDE_expression_factorization_l3939_393994

theorem expression_factorization (x : ℝ) : 
  (16 * x^4 + 36 * x^2 - 9) - (4 * x^4 - 6 * x^2 - 9) = 6 * x^2 * (2 * x^2 + 7) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l3939_393994


namespace NUMINAMATH_CALUDE_race_length_is_1000_l3939_393904

/-- Represents the length of the race in meters -/
def race_length : ℝ := 1000

/-- Represents the time difference between runners A and B in seconds -/
def time_difference : ℝ := 20

/-- Represents the distance difference between runners A and B in meters -/
def distance_difference : ℝ := 50

/-- Represents the time taken by runner A to complete the race in seconds -/
def time_A : ℝ := 380

/-- Theorem stating that the race length is 1000 meters given the conditions -/
theorem race_length_is_1000 :
  (distance_difference / time_difference) * (time_A + time_difference) = race_length := by
  sorry


end NUMINAMATH_CALUDE_race_length_is_1000_l3939_393904


namespace NUMINAMATH_CALUDE_min_balls_same_color_l3939_393992

/-- Represents the number of different colors of balls in the bag -/
def num_colors : ℕ := 2

/-- Represents the minimum number of balls to draw -/
def min_balls : ℕ := 3

/-- Theorem stating that given a bag with balls of two colors, 
    the minimum number of balls that must be drawn to ensure 
    at least two balls of the same color is 3 -/
theorem min_balls_same_color :
  ∀ (n : ℕ), n ≥ min_balls → 
  ∃ (color : Fin num_colors), (n.choose 2) > 0 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_same_color_l3939_393992


namespace NUMINAMATH_CALUDE_intersection_M_N_l3939_393919

def M : Set ℝ := {x : ℝ | |x - 1| ≤ 1}
def N : Set ℝ := {x : ℝ | Real.log x > 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3939_393919


namespace NUMINAMATH_CALUDE_apples_to_eat_raw_l3939_393959

theorem apples_to_eat_raw (total : ℕ) (wormy : ℕ) (bruised : ℕ) : 
  total = 85 →
  wormy = total / 5 →
  bruised = wormy + 9 →
  total - (wormy + bruised) = 42 := by
sorry

end NUMINAMATH_CALUDE_apples_to_eat_raw_l3939_393959


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l3939_393922

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℤ) + (⌈x⌉ : ℤ) = 7 ↔ 3 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l3939_393922


namespace NUMINAMATH_CALUDE_equal_spacing_theorem_l3939_393978

/-- The width of the wall in millimeters -/
def wall_width : ℕ := 4800

/-- The width of each picture in millimeters -/
def picture_width : ℕ := 420

/-- The number of pictures -/
def num_pictures : ℕ := 4

/-- The distance from the center of each middle picture to the center of the wall -/
def middle_picture_distance : ℕ := 730

/-- Theorem stating that the distance from the center of each middle picture
    to the center of the wall is 730 mm when all pictures are equally spaced -/
theorem equal_spacing_theorem :
  let total_space := wall_width - picture_width
  let spacing := total_space / (num_pictures - 1)
  spacing / 2 = middle_picture_distance := by sorry

end NUMINAMATH_CALUDE_equal_spacing_theorem_l3939_393978


namespace NUMINAMATH_CALUDE_greatest_three_digit_sum_with_reversal_l3939_393920

/-- Reverses a three-digit number -/
def reverse (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- Checks if a number is three-digit -/
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem greatest_three_digit_sum_with_reversal :
  ∀ n : ℕ, isThreeDigit n → n + reverse n ≤ 1211 → n ≤ 952 := by
  sorry

#check greatest_three_digit_sum_with_reversal

end NUMINAMATH_CALUDE_greatest_three_digit_sum_with_reversal_l3939_393920


namespace NUMINAMATH_CALUDE_sum_faces_edges_vertices_num_diagonals_square_base_l3939_393907

/-- A square pyramid is a polyhedron with a square base and triangular sides. -/
structure SquarePyramid where
  /-- The number of faces in a square pyramid -/
  faces : Nat
  /-- The number of edges in a square pyramid -/
  edges : Nat
  /-- The number of vertices in a square pyramid -/
  vertices : Nat
  /-- The number of sides in the base of a square pyramid -/
  base_sides : Nat

/-- Properties of a square pyramid -/
def square_pyramid : SquarePyramid :=
  { faces := 5
  , edges := 8
  , vertices := 5
  , base_sides := 4 }

/-- The sum of faces, edges, and vertices in a square pyramid is 18 -/
theorem sum_faces_edges_vertices (sp : SquarePyramid) : 
  sp.faces + sp.edges + sp.vertices = 18 := by
  sorry

/-- The number of diagonals in a polygon -/
def num_diagonals (n : Nat) : Nat :=
  n * (n - 3) / 2

/-- The number of diagonals in the square base of a square pyramid is 2 -/
theorem num_diagonals_square_base (sp : SquarePyramid) : 
  num_diagonals sp.base_sides = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_faces_edges_vertices_num_diagonals_square_base_l3939_393907


namespace NUMINAMATH_CALUDE_inequalities_given_negative_order_l3939_393953

theorem inequalities_given_negative_order (a b : ℝ) (h : b < a ∧ a < 0) :
  a^2 < b^2 ∧ 
  a * b > b^2 ∧ 
  (1/2 : ℝ)^b > (1/2 : ℝ)^a ∧ 
  a / b + b / a > 2 := by
sorry

end NUMINAMATH_CALUDE_inequalities_given_negative_order_l3939_393953


namespace NUMINAMATH_CALUDE_inequality_count_l3939_393928

theorem inequality_count (x y a b : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ a > 0 ∧ b > 0)
  (h_x_lt_1 : x < 1)
  (h_y_lt_1 : y < 1)
  (h_x_lt_a : x < a)
  (h_y_lt_b : y < b)
  (h_sum : x + y = a - b) :
  (x + y < a + b) ∧ 
  (x - y < a - b) ∧ 
  (x * y < a * b) ∧ 
  ¬(∀ (x y a b : ℝ), x / y < a / b) := by
sorry

end NUMINAMATH_CALUDE_inequality_count_l3939_393928


namespace NUMINAMATH_CALUDE_trumpet_cost_l3939_393940

/-- The cost of a trumpet, given the total amount spent and the cost of a song book. -/
theorem trumpet_cost (total_spent song_book_cost : ℚ) 
  (h1 : total_spent = 151)
  (h2 : song_book_cost = 5.84) :
  total_spent - song_book_cost = 145.16 := by
  sorry

end NUMINAMATH_CALUDE_trumpet_cost_l3939_393940


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3939_393900

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) (m : ℕ) :
  d ≠ 0 →
  arithmetic_sequence a d →
  a 3 + a 6 + a 10 + a 13 = 32 →
  a m = 8 →
  m = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3939_393900


namespace NUMINAMATH_CALUDE_units_digit_problem_l3939_393966

theorem units_digit_problem : (25^3 + 17^3) * 12^2 % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l3939_393966


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3939_393985

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by
sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3939_393985


namespace NUMINAMATH_CALUDE_sum_even_implies_one_even_l3939_393998

theorem sum_even_implies_one_even (a b c : ℕ) :
  Even (a + b + c) → ¬(Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end NUMINAMATH_CALUDE_sum_even_implies_one_even_l3939_393998


namespace NUMINAMATH_CALUDE_remainder_problem_l3939_393929

theorem remainder_problem (x : ℤ) : 
  (∃ k : ℤ, x = 142 * k + 110) → 
  (∃ m : ℤ, x = 14 * m + 12) :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l3939_393929


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3939_393981

/-- The distance between the foci of a hyperbola with equation y²/25 - x²/16 = 1 is 2√41 -/
theorem hyperbola_foci_distance : 
  let a : ℝ := 5
  let b : ℝ := 4
  let c : ℝ := Real.sqrt (a^2 + b^2)
  2 * c = 2 * Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3939_393981


namespace NUMINAMATH_CALUDE_units_digit_of_F_F8_l3939_393965

def modifiedFibonacci : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | (n + 2) => modifiedFibonacci (n + 1) + modifiedFibonacci n

theorem units_digit_of_F_F8 : 
  (modifiedFibonacci (modifiedFibonacci 8)) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_F_F8_l3939_393965


namespace NUMINAMATH_CALUDE_soccer_games_played_l3939_393989

theorem soccer_games_played (wins losses ties total : ℕ) : 
  wins + losses + ties = total →
  4 * ties = wins →
  3 * ties = losses →
  losses = 9 →
  total = 24 := by
sorry

end NUMINAMATH_CALUDE_soccer_games_played_l3939_393989


namespace NUMINAMATH_CALUDE_fair_coin_toss_probability_sum_l3939_393991

/-- Represents a fair coin --/
structure FairCoin where
  prob_heads : ℚ
  fair : prob_heads = 1/2

/-- Calculates the probability of getting exactly k heads in n tosses --/
def binomial_probability (c : FairCoin) (n k : ℕ) : ℚ :=
  (n.choose k) * c.prob_heads^k * (1 - c.prob_heads)^(n-k)

/-- The main theorem --/
theorem fair_coin_toss_probability_sum :
  ∀ (c : FairCoin),
  (binomial_probability c 5 1 = binomial_probability c 5 2) →
  ∃ (i j : ℕ),
    (binomial_probability c 5 3 = i / j) ∧
    (∀ (a b : ℕ), (a / b = i / j) → (a ≤ i ∧ b ≤ j)) ∧
    i + j = 283 :=
sorry

end NUMINAMATH_CALUDE_fair_coin_toss_probability_sum_l3939_393991


namespace NUMINAMATH_CALUDE_kaleb_initial_books_l3939_393927

/-- Represents the number of books Kaleb had initially. -/
def initial_books : ℕ := 34

/-- Represents the number of books Kaleb sold. -/
def books_sold : ℕ := 17

/-- Represents the number of new books Kaleb bought. -/
def new_books : ℕ := 7

/-- Represents the number of books Kaleb has now. -/
def current_books : ℕ := 24

/-- Proves that given the conditions, Kaleb must have had 34 books initially. -/
theorem kaleb_initial_books :
  initial_books - books_sold + new_books = current_books :=
by sorry

end NUMINAMATH_CALUDE_kaleb_initial_books_l3939_393927


namespace NUMINAMATH_CALUDE_sheep_barn_problem_l3939_393950

/-- Given a number of sheep between 2000 and 2100, if the probability of selecting
    two different sheep from different barns is exactly 1/2, then the number of
    sheep must be 2025. -/
theorem sheep_barn_problem (n : ℕ) (h1 : 2000 ≤ n) (h2 : n ≤ 2100) :
  (∃ k : ℕ, k < n ∧ 2 * k * (n - k) = n * (n - 1)) → n = 2025 :=
by sorry

end NUMINAMATH_CALUDE_sheep_barn_problem_l3939_393950


namespace NUMINAMATH_CALUDE_shaded_area_of_tiled_floor_l3939_393947

theorem shaded_area_of_tiled_floor :
  let floor_length : ℝ := 12
  let floor_width : ℝ := 10
  let tile_length : ℝ := 2
  let tile_width : ℝ := 1
  let circle_radius : ℝ := 1/2
  let triangle_base : ℝ := 1/2
  let triangle_height : ℝ := 1/2
  let num_tiles : ℝ := (floor_length / tile_width) * (floor_width / tile_length)
  let tile_area : ℝ := tile_length * tile_width
  let white_circle_area : ℝ := π * circle_radius^2
  let white_triangle_area : ℝ := 1/2 * triangle_base * triangle_height
  let shaded_area_per_tile : ℝ := tile_area - white_circle_area - white_triangle_area
  let total_shaded_area : ℝ := num_tiles * shaded_area_per_tile
  total_shaded_area = 112.5 - 15 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_of_tiled_floor_l3939_393947


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_nine_l3939_393917

theorem product_of_repeating_decimal_and_nine : ∃ (s : ℚ),
  (∀ (n : ℕ), s * 10^(3*n) - s * 10^(3*n-3) = 123 * 10^(3*n-3)) ∧
  s * 9 = 41 / 37 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_nine_l3939_393917


namespace NUMINAMATH_CALUDE_parabola_transformation_l3939_393937

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * h + p.b, c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

theorem parabola_transformation (x : ℝ) :
  let p₀ : Parabola := { a := 1, b := 0, c := 0 }  -- y = x²
  let p₁ := shift_horizontal p₀ 3                  -- shift 3 units right
  let p₂ := shift_vertical p₁ 4                    -- shift 4 units up
  p₂.a * x^2 + p₂.b * x + p₂.c = (x - 3)^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l3939_393937


namespace NUMINAMATH_CALUDE_decimal_equality_and_unit_l3939_393980

/-- Represents a number with its counting unit -/
structure NumberWithUnit where
  value : ℝ
  unit : ℝ

/-- The statement we want to prove false -/
def statement (a b : NumberWithUnit) : Prop :=
  a.value = b.value ∧ a.unit = b.unit

/-- The theorem to prove -/
theorem decimal_equality_and_unit (a b : NumberWithUnit) 
  (h1 : a.value = b.value)
  (h2 : a.unit = 1)
  (h3 : b.unit = 0.1) : 
  ¬(statement a b) := by
  sorry

#check decimal_equality_and_unit

end NUMINAMATH_CALUDE_decimal_equality_and_unit_l3939_393980


namespace NUMINAMATH_CALUDE_eighty_seventh_odd_integer_l3939_393964

theorem eighty_seventh_odd_integer : ∀ n : ℕ, n > 0 → (2 * n - 1) = 173 ↔ n = 87 := by
  sorry

end NUMINAMATH_CALUDE_eighty_seventh_odd_integer_l3939_393964


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3939_393918

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 + m - 1001 = 0) → 
  (n^2 + n - 1001 = 0) → 
  m^2 + 2*m + n = 1000 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3939_393918


namespace NUMINAMATH_CALUDE_percentage_in_accounting_l3939_393995

def accountant_years : ℕ := 25
def manager_years : ℕ := 15
def total_lifespan : ℕ := 80

def accounting_years : ℕ := accountant_years + manager_years

theorem percentage_in_accounting : 
  (accounting_years : ℚ) / total_lifespan * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_in_accounting_l3939_393995


namespace NUMINAMATH_CALUDE_pipe_filling_time_l3939_393945

theorem pipe_filling_time (p q r : ℝ) (hp : p = 3) (hr : r = 18) (hall : 1/p + 1/q + 1/r = 1/2) :
  q = 9 := by
sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l3939_393945
