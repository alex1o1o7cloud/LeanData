import Mathlib

namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l3049_304994

theorem symmetric_points_sum_power (m n : ℤ) : 
  (m = -6 ∧ n = 5) → (m + n)^2012 = 1 := by sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l3049_304994


namespace NUMINAMATH_CALUDE_total_volume_of_cubes_combined_cube_volume_l3049_304987

theorem total_volume_of_cubes : ℕ → ℕ → ℕ → ℕ → ℕ
  | carl_count, carl_side, kate_count, kate_side =>
    (carl_count * carl_side^3) + (kate_count * kate_side^3)

theorem combined_cube_volume : total_volume_of_cubes 8 2 3 3 = 145 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_cubes_combined_cube_volume_l3049_304987


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l3049_304942

/-- Given the conditions of a company's financial performance over two years,
    prove that the profit percentage in the previous year was 10%. -/
theorem profit_percentage_previous_year
  (R : ℝ) -- Revenues in the previous year
  (P : ℝ) -- Profits in the previous year
  (h1 : R > 0) -- Assume positive revenue
  (h2 : P > 0) -- Assume positive profit
  (h3 : 0.8 * R * 0.12 = 0.96 * P) -- Condition from the problem
  : P / R = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l3049_304942


namespace NUMINAMATH_CALUDE_three_numbers_problem_l3049_304948

theorem three_numbers_problem (x y z : ℤ) : 
  x - y = 12 ∧ 
  (x + y) / 4 = 7 ∧ 
  z = 2 * y ∧ 
  x + z = 24 → 
  x = 20 ∧ y = 8 ∧ z = 16 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l3049_304948


namespace NUMINAMATH_CALUDE_fruit_box_theorem_l3049_304907

theorem fruit_box_theorem (total_fruits : ℕ) 
  (h_total : total_fruits = 56)
  (h_oranges : total_fruits / 4 = total_fruits / 4)  -- One-fourth are oranges
  (h_peaches : total_fruits / 8 = total_fruits / 8)  -- Half as many peaches as oranges
  (h_apples : 5 * (total_fruits / 8) = 5 * (total_fruits / 8))  -- Five times as many apples as peaches
  (h_mixed : total_fruits / 4 = total_fruits / 4)  -- Twice as many mixed fruits as peaches
  : (5 * (total_fruits / 8) = 35) ∧ 
    (total_fruits / 4 : ℚ) / total_fruits = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fruit_box_theorem_l3049_304907


namespace NUMINAMATH_CALUDE_triangle_formation_l3049_304941

/-- Triangle inequality check for three sides -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 3 3 6 ∧
  ¬ can_form_triangle 2 5 7 ∧
  can_form_triangle 4 5 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l3049_304941


namespace NUMINAMATH_CALUDE_addition_subtraction_problem_l3049_304953

theorem addition_subtraction_problem : (0.45 + 52.7) - 0.25 = 52.9 := by
  sorry

end NUMINAMATH_CALUDE_addition_subtraction_problem_l3049_304953


namespace NUMINAMATH_CALUDE_range_of_m_l3049_304988

/-- Given conditions for the problem -/
structure ProblemConditions (m : ℝ) :=
  (h1 : ∃ x : ℝ, (x^2 + 1) * (x^2 - 8*x - 20) ≤ 0)
  (h2 : ∃ x : ℝ, x^2 - 2*x + 1 - m^2 ≤ 0)
  (h3 : m > 0)
  (h4 : ∀ x : ℝ, (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m))
  (h5 : ∃ x : ℝ, (x < -2 ∨ x > 10) ∧ ¬(x < 1 - m ∨ x > 1 + m))

/-- The main theorem stating the range of m -/
theorem range_of_m (m : ℝ) (h : ProblemConditions m) : m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3049_304988


namespace NUMINAMATH_CALUDE_rohan_join_time_is_seven_l3049_304990

/-- Represents the investment scenario and profit distribution --/
structure InvestmentScenario where
  suresh_investment : ℕ
  rohan_investment : ℕ
  sudhir_investment : ℕ
  total_profit : ℕ
  rohan_sudhir_diff : ℕ
  total_months : ℕ
  sudhir_join_time : ℕ

/-- Calculates the number of months after which Rohan joined the business --/
def calculate_rohan_join_time (scenario : InvestmentScenario) : ℕ :=
  sorry

/-- Theorem stating that Rohan joined after 7 months --/
theorem rohan_join_time_is_seven (scenario : InvestmentScenario) 
  (h1 : scenario.suresh_investment = 18000)
  (h2 : scenario.rohan_investment = 12000)
  (h3 : scenario.sudhir_investment = 9000)
  (h4 : scenario.total_profit = 3795)
  (h5 : scenario.rohan_sudhir_diff = 345)
  (h6 : scenario.total_months = 12)
  (h7 : scenario.sudhir_join_time = 8) : 
  calculate_rohan_join_time scenario = 7 :=
sorry

end NUMINAMATH_CALUDE_rohan_join_time_is_seven_l3049_304990


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3049_304949

theorem solution_set_inequality (x : ℝ) : (2*x + 1) / (x + 1) < 1 ↔ -1 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3049_304949


namespace NUMINAMATH_CALUDE_nth_inequality_l3049_304971

theorem nth_inequality (x : ℝ) (n : ℕ) (h : x > 0) : 
  x + (n^n : ℝ) / x^n ≥ n + 1 := by
sorry

end NUMINAMATH_CALUDE_nth_inequality_l3049_304971


namespace NUMINAMATH_CALUDE_youngest_child_age_l3049_304954

/-- Given 5 children born at intervals of 2 years each, 
    if the sum of their ages is 55 years, 
    then the age of the youngest child is 7 years. -/
theorem youngest_child_age 
  (n : ℕ) 
  (h1 : n = 5) 
  (interval : ℕ) 
  (h2 : interval = 2) 
  (total_age : ℕ) 
  (h3 : total_age = 55) 
  (youngest_age : ℕ) 
  (h4 : youngest_age * n + (n * (n - 1) / 2) * interval = total_age) : 
  youngest_age = 7 := by
sorry

end NUMINAMATH_CALUDE_youngest_child_age_l3049_304954


namespace NUMINAMATH_CALUDE_fathers_age_fathers_age_is_52_l3049_304984

theorem fathers_age (sons_age_5_years_ago : ℕ) (years_passed : ℕ) : ℕ :=
  let sons_current_age := sons_age_5_years_ago + years_passed
  2 * sons_current_age

theorem fathers_age_is_52 : fathers_age 21 5 = 52 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_fathers_age_is_52_l3049_304984


namespace NUMINAMATH_CALUDE_track_length_is_630_l3049_304918

/-- The length of the circular track in meters -/
def track_length : ℝ := 630

/-- The angle between the starting positions of the two runners in degrees -/
def start_angle : ℝ := 120

/-- The distance run by the first runner (Tom) before the first meeting in meters -/
def first_meeting_distance : ℝ := 120

/-- The additional distance run by the second runner (Jerry) between the first and second meeting in meters -/
def second_meeting_distance : ℝ := 180

/-- Theorem stating that the given conditions imply the track length is 630 meters -/
theorem track_length_is_630 : 
  ∃ (speed_tom speed_jerry : ℝ), 
    speed_tom > 0 ∧ speed_jerry > 0 ∧
    first_meeting_distance / speed_tom = (track_length * start_angle / 360 - first_meeting_distance) / speed_jerry ∧
    (track_length - (track_length * start_angle / 360 - first_meeting_distance) - second_meeting_distance) / speed_tom = 
      (track_length * start_angle / 360 - first_meeting_distance + second_meeting_distance) / speed_jerry :=
by
  sorry


end NUMINAMATH_CALUDE_track_length_is_630_l3049_304918


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_l3049_304963

theorem sqrt_sum_difference : Real.sqrt 50 + Real.sqrt 32 - Real.sqrt 2 = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_l3049_304963


namespace NUMINAMATH_CALUDE_x_intercept_of_specific_line_l3049_304996

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ := 
  sorry

/-- The specific line passing through (-1, 1) and (0, 3) -/
def specific_line : Line := { x₁ := -1, y₁ := 1, x₂ := 0, y₂ := 3 }

theorem x_intercept_of_specific_line : 
  x_intercept specific_line = -3/2 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_specific_line_l3049_304996


namespace NUMINAMATH_CALUDE_AMC9_paths_count_l3049_304999

/-- Represents the layout of the AMC9 puzzle --/
structure AMC9Layout where
  start_A : Nat
  adjacent_Ms : Nat
  adjacent_Cs : Nat
  Cs_with_two_9s : Nat
  Cs_with_one_9 : Nat

/-- Calculates the number of paths in the AMC9 puzzle --/
def count_AMC9_paths (layout : AMC9Layout) : Nat :=
  layout.adjacent_Ms * 
  (layout.Cs_with_two_9s * 2 + layout.Cs_with_one_9 * 1)

/-- Theorem stating that the number of paths in the AMC9 puzzle is 20 --/
theorem AMC9_paths_count :
  ∀ (layout : AMC9Layout),
  layout.start_A = 1 →
  layout.adjacent_Ms = 4 →
  layout.adjacent_Cs = 3 →
  layout.Cs_with_two_9s = 2 →
  layout.Cs_with_one_9 = 1 →
  count_AMC9_paths layout = 20 := by
  sorry

end NUMINAMATH_CALUDE_AMC9_paths_count_l3049_304999


namespace NUMINAMATH_CALUDE_increasing_magnitude_l3049_304916

-- Define the variables and conditions
theorem increasing_magnitude (a : ℝ) 
  (h1 : 0.8 < a) (h2 : a < 0.9)
  (y : ℝ) (hy : y = a^a)
  (z : ℝ) (hz : z = a^(a^a))
  (w : ℝ) (hw : w = a^(Real.log a)) :
  a < z ∧ z < y ∧ y < w := by sorry

end NUMINAMATH_CALUDE_increasing_magnitude_l3049_304916


namespace NUMINAMATH_CALUDE_circle_rolling_in_triangle_l3049_304943

theorem circle_rolling_in_triangle (a b c : ℝ) (r : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) (h4 : r = 2) :
  let k := (a + b + c - 6 * r) / (a + b + c)
  (k * a + k * b + k * c) = 220 / 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_rolling_in_triangle_l3049_304943


namespace NUMINAMATH_CALUDE_inequality_proof_l3049_304975

theorem inequality_proof (a b c d p q : ℝ) 
  (h1 : a * b + c * d = 2 * p * q)
  (h2 : a * c ≥ p ^ 2)
  (h3 : p > 0) : 
  b * d ≤ q ^ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3049_304975


namespace NUMINAMATH_CALUDE_rice_weight_per_container_l3049_304951

/-- 
Given a bag of rice weighing sqrt(50) pounds divided equally into 7 containers,
prove that the weight of rice in each container, in ounces, is (80 * sqrt(2)) / 7,
assuming 1 pound = 16 ounces.
-/
theorem rice_weight_per_container 
  (total_weight : ℝ) 
  (num_containers : ℕ) 
  (pounds_to_ounces : ℝ) 
  (h1 : total_weight = Real.sqrt 50)
  (h2 : num_containers = 7)
  (h3 : pounds_to_ounces = 16) :
  (total_weight / num_containers) * pounds_to_ounces = (80 * Real.sqrt 2) / 7 := by
  sorry

end NUMINAMATH_CALUDE_rice_weight_per_container_l3049_304951


namespace NUMINAMATH_CALUDE_students_without_A_l3049_304940

theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (both : ℕ)
  (h_total : total = 40)
  (h_history : history = 10)
  (h_math : math = 18)
  (h_both : both = 6) :
  total - (history + math - both) = 18 :=
by sorry

end NUMINAMATH_CALUDE_students_without_A_l3049_304940


namespace NUMINAMATH_CALUDE_school_distance_is_150km_l3049_304904

/-- The distance from Xiaoming's home to school in kilometers. -/
def school_distance : ℝ := 150

/-- Xiaoming's walking speed in km/h. -/
def walking_speed : ℝ := 5

/-- The car speed in km/h. -/
def car_speed : ℝ := 15

/-- The time difference between going to school and returning home in hours. -/
def time_difference : ℝ := 2

/-- Theorem stating the distance from Xiaoming's home to school is 150 km. -/
theorem school_distance_is_150km :
  let d := school_distance
  let v_walk := walking_speed
  let v_car := car_speed
  let t_diff := time_difference
  (d / (2 * v_walk) + d / (2 * v_car) = d / (3 * v_car) + 2 * d / (3 * v_walk) + t_diff) →
  d = 150 := by
  sorry


end NUMINAMATH_CALUDE_school_distance_is_150km_l3049_304904


namespace NUMINAMATH_CALUDE_square_sum_equals_sixteen_l3049_304977

theorem square_sum_equals_sixteen (x : ℝ) : 
  (x - 1)^2 + 2*(x - 1)*(5 - x) + (5 - x)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_sixteen_l3049_304977


namespace NUMINAMATH_CALUDE_joan_seashells_l3049_304960

theorem joan_seashells (jessica_shells : ℕ) (total_shells : ℕ) (h1 : jessica_shells = 8) (h2 : total_shells = 14) :
  total_shells - jessica_shells = 6 :=
sorry

end NUMINAMATH_CALUDE_joan_seashells_l3049_304960


namespace NUMINAMATH_CALUDE_square_area_6cm_l3049_304939

theorem square_area_6cm (side_length : ℝ) (h : side_length = 6) :
  side_length * side_length = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_6cm_l3049_304939


namespace NUMINAMATH_CALUDE_z_squared_minus_one_equals_two_plus_four_i_l3049_304958

def z : ℂ := 2 + Complex.I

theorem z_squared_minus_one_equals_two_plus_four_i :
  z^2 - 1 = 2 + 4*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_z_squared_minus_one_equals_two_plus_four_i_l3049_304958


namespace NUMINAMATH_CALUDE_function_inequality_l3049_304968

theorem function_inequality (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : ∀ x, f x = |Real.log x|)
  (h4 : f a > f c) (h5 : f c > f b) : 
  (a - 1) * (c - 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3049_304968


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l3049_304973

-- Define the function f
def f (x : ℝ) : ℝ := (2*x + 1)^3

-- State the theorem
theorem derivative_f_at_zero : 
  deriv f 0 = 6 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l3049_304973


namespace NUMINAMATH_CALUDE_project_work_difference_l3049_304995

/-- Represents the work times of four people on a project -/
structure ProjectWork where
  time1 : ℕ
  time2 : ℕ
  time3 : ℕ
  time4 : ℕ

/-- The total work time of the project -/
def totalTime (pw : ProjectWork) : ℕ :=
  pw.time1 + pw.time2 + pw.time3 + pw.time4

/-- The work times are in the ratio 1:2:3:4 -/
def validRatio (pw : ProjectWork) : Prop :=
  2 * pw.time1 = pw.time2 ∧
  3 * pw.time1 = pw.time3 ∧
  4 * pw.time1 = pw.time4

theorem project_work_difference (pw : ProjectWork) 
  (h1 : totalTime pw = 240)
  (h2 : validRatio pw) :
  pw.time4 - pw.time1 = 72 := by
  sorry

end NUMINAMATH_CALUDE_project_work_difference_l3049_304995


namespace NUMINAMATH_CALUDE_even_increasing_function_inequality_l3049_304923

/-- An even function that is monotonically increasing on the non-negative reals -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)

theorem even_increasing_function_inequality 
  (f : ℝ → ℝ) 
  (h_even_increasing : EvenIncreasingFunction f) 
  (h_f_1 : f 1 = 0) :
  {x : ℝ | f (x - 2) ≥ 0} = {x : ℝ | x ≥ 3 ∨ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_function_inequality_l3049_304923


namespace NUMINAMATH_CALUDE_darnell_initial_fabric_l3049_304937

/-- Calculates the initial amount of fabric Darnell had --/
def initial_fabric (square_side : ℕ) (wide_length wide_width : ℕ) (tall_length tall_width : ℕ)
  (num_square num_wide num_tall : ℕ) (fabric_left : ℕ) : ℕ :=
  let square_area := square_side * square_side
  let wide_area := wide_length * wide_width
  let tall_area := tall_length * tall_width
  let total_used := square_area * num_square + wide_area * num_wide + tall_area * num_tall
  total_used + fabric_left

/-- Theorem stating that Darnell initially had 1000 square feet of fabric --/
theorem darnell_initial_fabric :
  initial_fabric 4 5 3 3 5 16 20 10 294 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_darnell_initial_fabric_l3049_304937


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l3049_304946

theorem shopkeeper_profit (cost_price : ℝ) : cost_price > 0 →
  let marked_price := cost_price * 1.2
  let selling_price := marked_price * 0.85
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 2 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l3049_304946


namespace NUMINAMATH_CALUDE_composite_expression_l3049_304914

theorem composite_expression (n : ℕ) (h : n ≥ 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 3^(2*n+1) - 2^(2*n+1) - 6^n = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_expression_l3049_304914


namespace NUMINAMATH_CALUDE_square_of_negative_two_m_squared_l3049_304913

theorem square_of_negative_two_m_squared (m : ℝ) : (-2 * m^2)^2 = 4 * m^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_m_squared_l3049_304913


namespace NUMINAMATH_CALUDE_correct_quotient_l3049_304986

theorem correct_quotient (D : ℕ) : 
  D % 21 = 0 →  -- The remainder is 0 when divided by 21
  D / 12 = 35 →  -- Dividing by 12 yields a quotient of 35
  D / 21 = 20  -- The correct quotient when dividing by 21 is 20
:= by sorry

end NUMINAMATH_CALUDE_correct_quotient_l3049_304986


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l3049_304982

theorem function_inequality_implies_m_bound (f g : ℝ → ℝ) (m : ℝ) 
  (hf : ∀ x, f x = x^2)
  (hg : ∀ x, g x = (1/2)^x - m)
  (h : ∀ x₁ ∈ Set.Icc 0 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g x₂) :
  m ≥ 1/4 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l3049_304982


namespace NUMINAMATH_CALUDE_point_on_segment_coordinates_l3049_304998

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line segment between two other points -/
def lies_on_segment (p q r : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    p.x = q.x + t * (r.x - q.x) ∧
    p.y = q.y + t * (r.y - q.y)

theorem point_on_segment_coordinates :
  let K : Point := ⟨4, 2⟩
  let M : Point := ⟨10, 11⟩
  let L : Point := ⟨6, w⟩
  lies_on_segment L K M → w = 5 := by
sorry

end NUMINAMATH_CALUDE_point_on_segment_coordinates_l3049_304998


namespace NUMINAMATH_CALUDE_sum_ratio_equals_four_sevenths_l3049_304945

theorem sum_ratio_equals_four_sevenths 
  (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 16)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 49)
  (sum_products : a*x + b*y + c*z = 28) :
  (a + b + c) / (x + y + z) = 4/7 := by
sorry

end NUMINAMATH_CALUDE_sum_ratio_equals_four_sevenths_l3049_304945


namespace NUMINAMATH_CALUDE_x_squared_mod_20_l3049_304925

theorem x_squared_mod_20 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 20]) (h2 : 2 * x ≡ 8 [ZMOD 20]) :
  x^2 ≡ 16 [ZMOD 20] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_20_l3049_304925


namespace NUMINAMATH_CALUDE_mean_median_difference_l3049_304955

/-- Represents the frequency distribution of days missed --/
structure FrequencyDistribution :=
  (zero_days : Nat)
  (one_day : Nat)
  (two_days : Nat)
  (three_days : Nat)
  (four_days : Nat)
  (five_days : Nat)

/-- Calculates the median of the dataset --/
def median (fd : FrequencyDistribution) : Rat :=
  2

/-- Calculates the mean of the dataset --/
def mean (fd : FrequencyDistribution) : Rat :=
  (0 * fd.zero_days + 1 * fd.one_day + 2 * fd.two_days + 
   3 * fd.three_days + 4 * fd.four_days + 5 * fd.five_days) / 20

/-- The main theorem to prove --/
theorem mean_median_difference 
  (fd : FrequencyDistribution) 
  (h1 : fd.zero_days = 4)
  (h2 : fd.one_day = 2)
  (h3 : fd.two_days = 5)
  (h4 : fd.three_days = 3)
  (h5 : fd.four_days = 2)
  (h6 : fd.five_days = 4)
  (h7 : fd.zero_days + fd.one_day + fd.two_days + fd.three_days + fd.four_days + fd.five_days = 20) :
  mean fd - median fd = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l3049_304955


namespace NUMINAMATH_CALUDE_angle_alpha_trig_l3049_304917

theorem angle_alpha_trig (α : Real) (m : Real) :
  m ≠ 0 →
  (∃ (x y : Real), x = -Real.sqrt 3 ∧ y = m ∧ x^2 + y^2 = (Real.cos α)^2 + (Real.sin α)^2) →
  Real.sin α = (Real.sqrt 2 / 4) * m →
  (m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
  Real.cos α = -Real.sqrt 6 / 4 ∧
  ((m > 0 → Real.tan α = -Real.sqrt 15 / 3) ∧
   (m < 0 → Real.tan α = Real.sqrt 15 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_angle_alpha_trig_l3049_304917


namespace NUMINAMATH_CALUDE_sector_area_l3049_304929

/-- Given a circular sector with circumference 8 and central angle 2 radians, its area is 4. -/
theorem sector_area (c : ℝ) (θ : ℝ) (h1 : c = 8) (h2 : θ = 2) :
  let r := c / (2 + 2 * Real.pi)
  (1/2) * r^2 * θ = 4 := by sorry

end NUMINAMATH_CALUDE_sector_area_l3049_304929


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l3049_304935

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_rate_increase : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 16)
  (h2 : regular_hours = 40)
  (h3 : overtime_rate_increase = 0.75)
  (h4 : total_hours = 52) :
  let overtime_hours := total_hours - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  regular_pay + overtime_pay = 976 := by
sorry


end NUMINAMATH_CALUDE_bus_driver_compensation_l3049_304935


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3049_304989

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, 4 * x^2 - k * x + 6 = 0 ∧ x = 2) → 
  (k = 11 ∧ ∃ y : ℝ, 4 * y^2 - k * y + 6 = 0 ∧ y = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3049_304989


namespace NUMINAMATH_CALUDE_script_writing_problem_l3049_304957

/-- Represents the number of lines for each character in the script -/
structure ScriptLines where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the script writing problem -/
def script_conditions (s : ScriptLines) : Prop :=
  s.first = s.second + 8 ∧
  s.third = 2 ∧
  s.second = 3 * s.third + 6 ∧
  s.first = 20

/-- The theorem stating the solution to the script writing problem -/
theorem script_writing_problem (s : ScriptLines) 
  (h : script_conditions s) : ∃ m : ℕ, s.second = m * s.third + 6 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_script_writing_problem_l3049_304957


namespace NUMINAMATH_CALUDE_min_green_tiles_l3049_304944

/-- Represents the colors of tiles --/
inductive Color
  | Red
  | Orange
  | Yellow
  | Green
  | Blue
  | Indigo

/-- Represents the number of tiles for each color --/
structure TileCount where
  red : ℕ
  orange : ℕ
  yellow : ℕ
  green : ℕ
  blue : ℕ
  indigo : ℕ

/-- The total number of tiles --/
def total_tiles : ℕ := 100

/-- Checks if the tile count satisfies all constraints --/
def satisfies_constraints (tc : TileCount) : Prop :=
  tc.red + tc.orange + tc.yellow + tc.green + tc.blue + tc.indigo = total_tiles ∧
  tc.indigo ≥ tc.red + tc.orange + tc.yellow + tc.green + tc.blue ∧
  tc.blue ≥ tc.red + tc.orange + tc.yellow + tc.green ∧
  tc.green ≥ tc.red + tc.orange + tc.yellow ∧
  tc.yellow ≥ tc.red + tc.orange ∧
  tc.orange ≥ tc.red

/-- Checks if one tile count is preferred over another according to the client's preferences --/
def is_preferred (tc1 tc2 : TileCount) : Prop :=
  tc1.red > tc2.red ∨
  (tc1.red = tc2.red ∧ tc1.orange > tc2.orange) ∨
  (tc1.red = tc2.red ∧ tc1.orange = tc2.orange ∧ tc1.yellow > tc2.yellow) ∨
  (tc1.red = tc2.red ∧ tc1.orange = tc2.orange ∧ tc1.yellow = tc2.yellow ∧ tc1.green > tc2.green) ∨
  (tc1.red = tc2.red ∧ tc1.orange = tc2.orange ∧ tc1.yellow = tc2.yellow ∧ tc1.green = tc2.green ∧ tc1.blue > tc2.blue) ∨
  (tc1.red = tc2.red ∧ tc1.orange = tc2.orange ∧ tc1.yellow = tc2.yellow ∧ tc1.green = tc2.green ∧ tc1.blue = tc2.blue ∧ tc1.indigo > tc2.indigo)

/-- The theorem to be proved --/
theorem min_green_tiles :
  ∃ (optimal : TileCount),
    satisfies_constraints optimal ∧
    optimal.green = 13 ∧
    ∀ (tc : TileCount), satisfies_constraints tc → ¬is_preferred tc optimal :=
by sorry

end NUMINAMATH_CALUDE_min_green_tiles_l3049_304944


namespace NUMINAMATH_CALUDE_points_collinear_l3049_304910

/-- Three points A, B, and C in the plane are collinear if there exists a real number k such that 
    vector AC = k * vector AB. -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (C.1 - A.1, C.2 - A.2) = (k * (B.1 - A.1), k * (B.2 - A.2))

/-- The points A(-1, -2), B(2, -1), and C(8, 1) are collinear. -/
theorem points_collinear : collinear (-1, -2) (2, -1) (8, 1) := by
  sorry


end NUMINAMATH_CALUDE_points_collinear_l3049_304910


namespace NUMINAMATH_CALUDE_rock_age_count_l3049_304938

/-- The set of digits used to form the rock's age -/
def rock_age_digits : Finset Nat := {2, 3, 7, 9}

/-- The number of occurrences of each digit in the rock's age -/
def digit_occurrences : Nat → Nat
  | 2 => 3
  | 3 => 1
  | 7 => 1
  | 9 => 1
  | _ => 0

/-- The set of odd digits that can start the rock's age -/
def odd_start_digits : Finset Nat := {3, 7, 9}

/-- The length of the rock's age in digits -/
def age_length : Nat := 6

/-- The number of possibilities for the rock's age -/
def rock_age_possibilities : Nat := 60

theorem rock_age_count :
  (Finset.card odd_start_digits) *
  (Nat.factorial (age_length - 1)) /
  (Nat.factorial (digit_occurrences 2)) =
  rock_age_possibilities := by sorry

end NUMINAMATH_CALUDE_rock_age_count_l3049_304938


namespace NUMINAMATH_CALUDE_ratio_calculation_l3049_304901

theorem ratio_calculation (A B C : ℚ) (h : A/B = 3/2 ∧ B/C = 2/5) :
  (4*A + 3*B) / (5*C - 2*B) = 15/23 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l3049_304901


namespace NUMINAMATH_CALUDE_max_gold_coins_max_gold_coins_proof_l3049_304911

/-- The largest number of gold coins that can be distributed among 15 friends
    with 4 coins left over and a total less than 150. -/
theorem max_gold_coins : ℕ :=
  let num_friends : ℕ := 15
  let extra_coins : ℕ := 4
  let max_total : ℕ := 149  -- less than 150
  
  have h1 : ∃ (k : ℕ), num_friends * k + extra_coins ≤ max_total :=
    sorry
  
  have h2 : ∀ (n : ℕ), num_friends * n + extra_coins > max_total → n > 9 :=
    sorry
  
  139

theorem max_gold_coins_proof (n : ℕ) :
  n ≤ max_gold_coins ∧
  (∃ (k : ℕ), n = 15 * k + 4) ∧
  n < 150 :=
by sorry

end NUMINAMATH_CALUDE_max_gold_coins_max_gold_coins_proof_l3049_304911


namespace NUMINAMATH_CALUDE_friend_meeting_probability_l3049_304969

/-- The probability that two friends meet given specific conditions -/
theorem friend_meeting_probability : 
  ∀ (wait_time : ℝ) (window : ℝ),
  wait_time > 0 → 
  window > wait_time →
  (∃ (prob : ℝ), 
    prob = (window^2 - 2 * (window - wait_time)^2 / 2) / window^2 ∧ 
    prob = 8/9) := by
  sorry

end NUMINAMATH_CALUDE_friend_meeting_probability_l3049_304969


namespace NUMINAMATH_CALUDE_emmalyn_earnings_l3049_304920

/-- Calculates the total amount earned from painting fences. -/
def total_amount_earned (price_per_meter : ℚ) (num_fences : ℕ) (fence_length : ℕ) : ℚ :=
  price_per_meter * (num_fences : ℚ) * (fence_length : ℚ)

/-- Proves that Emmalyn earned $5,000 from painting fences. -/
theorem emmalyn_earnings : 
  total_amount_earned (20 / 100) 50 500 = 5000 := by
  sorry

#eval total_amount_earned (20 / 100) 50 500

end NUMINAMATH_CALUDE_emmalyn_earnings_l3049_304920


namespace NUMINAMATH_CALUDE_equation_solutions_l3049_304926

theorem equation_solutions (b c : ℝ) : 
  (∀ x : ℝ, (|x - 4| = 3) ↔ (x^2 + b*x + c = 0)) → 
  (b = -8 ∧ c = 7) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3049_304926


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3049_304927

def A : Set Char := {'a', 'b', 'c', 'd', 'e'}
def B : Set Char := {'d', 'f', 'g'}

theorem intersection_of_A_and_B :
  A ∩ B = {'d'} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3049_304927


namespace NUMINAMATH_CALUDE_four_digit_sum_4360_l3049_304932

def is_valid_insertion (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧ 
    ((a = 2 ∧ b = 1 ∧ d = 5) ∨ (a = 2 ∧ c = 1 ∧ d = 5))

theorem four_digit_sum_4360 :
  ∀ (n₁ n₂ : ℕ), is_valid_insertion n₁ → is_valid_insertion n₂ → n₁ + n₂ = 4360 →
    ((n₁ = 2195 ∧ n₂ = 2165) ∨ (n₁ = 2185 ∧ n₂ = 2175) ∨ (n₁ = 2215 ∧ n₂ = 2145)) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_4360_l3049_304932


namespace NUMINAMATH_CALUDE_solve_equation_l3049_304959

-- Define the function F
def F (a b c d : ℕ) : ℕ := a^b + c * d

-- State the theorem
theorem solve_equation : ∃ x : ℕ, F 2 x 4 11 = 300 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3049_304959


namespace NUMINAMATH_CALUDE_line_slope_l3049_304985

/-- Given a line with equation x/4 + y/3 = 2, its slope is -3/4 -/
theorem line_slope (x y : ℝ) : (x / 4 + y / 3 = 2) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l3049_304985


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l3049_304902

theorem consecutive_odd_numbers_sum (k : ℤ) : 
  (2*k - 1) + (2*k + 1) + (2*k + 3) = (2*k - 1) + 128 → 2*k - 1 = 61 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l3049_304902


namespace NUMINAMATH_CALUDE_percentage_difference_l3049_304991

theorem percentage_difference : 
  (68.5 / 100 * 825) - (34.25 / 100 * 1620) = 10.275 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3049_304991


namespace NUMINAMATH_CALUDE_geckos_sold_last_year_l3049_304950

theorem geckos_sold_last_year (x : ℕ) : 
  x + 2 * x = 258 → x = 86 := by
  sorry

end NUMINAMATH_CALUDE_geckos_sold_last_year_l3049_304950


namespace NUMINAMATH_CALUDE_relationship_abc_l3049_304952

theorem relationship_abc : 
  let a := Real.log 2
  let b := 5^(-1/2 : ℝ)
  let c := (1/4 : ℝ) * ∫ x in (0 : ℝ)..(π : ℝ), Real.sin x
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3049_304952


namespace NUMINAMATH_CALUDE_unique_solution_l3049_304979

def system_solution (x y : ℝ) : Prop :=
  x + y = 1 ∧ x - y = -1

theorem unique_solution : 
  ∃! p : ℝ × ℝ, system_solution p.1 p.2 ∧ p = (0, 1) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3049_304979


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3049_304906

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |x - 3| = |x + 2| :=
by
  -- The unique solution is x = 1/2
  use 1/2
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3049_304906


namespace NUMINAMATH_CALUDE_sculpture_surface_area_l3049_304924

/-- Represents a cube sculpture with three layers -/
structure CubeSculpture where
  top_layer : Nat
  middle_layer : Nat
  bottom_layer : Nat
  cube_edge_length : Real

/-- Calculates the exposed surface area of a cube sculpture -/
def exposed_surface_area (sculpture : CubeSculpture) : Real :=
  let top_area := sculpture.top_layer * (5 * sculpture.cube_edge_length ^ 2)
  let middle_area := 4 * sculpture.middle_layer * sculpture.cube_edge_length ^ 2
  let bottom_area := sculpture.bottom_layer * sculpture.cube_edge_length ^ 2
  top_area + middle_area + bottom_area

/-- The main theorem stating that the exposed surface area of the specific sculpture is 35 square meters -/
theorem sculpture_surface_area :
  let sculpture : CubeSculpture := {
    top_layer := 1,
    middle_layer := 6,
    bottom_layer := 12,
    cube_edge_length := 1
  }
  exposed_surface_area sculpture = 35 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_surface_area_l3049_304924


namespace NUMINAMATH_CALUDE_slag_transport_allocation_l3049_304909

/-- Represents the daily rental income for slag transport vehicles --/
def daily_rental_income (x : ℕ) : ℕ := 80000 - 200 * x

/-- Theorem stating the properties of the slag transport vehicle allocation problem --/
theorem slag_transport_allocation :
  (∀ x : ℕ, x ≤ 20 → daily_rental_income x = 80000 - 200 * x) ∧
  (∀ x : ℕ, x ≤ 20 → (daily_rental_income x ≥ 79600 ↔ x ≤ 2)) ∧
  (∀ x : ℕ, x ≤ 20 → daily_rental_income x ≤ 80000) ∧
  (daily_rental_income 0 = 80000) := by
  sorry

#check slag_transport_allocation

end NUMINAMATH_CALUDE_slag_transport_allocation_l3049_304909


namespace NUMINAMATH_CALUDE_chess_square_exists_l3049_304976

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 100x100 table of colored cells -/
def Table := Fin 100 → Fin 100 → Color

/-- Checks if a cell is on the border of the table -/
def isBorder (i j : Fin 100) : Prop :=
  i = 0 || i = 99 || j = 0 || j = 99

/-- Checks if a 2x2 square starting at (i,j) contains cells of two colors -/
def hasTwoColors (t : Table) (i j : Fin 100) : Prop :=
  ∃ (c₁ c₂ : Color), c₁ ≠ c₂ ∧
    ((t i j = c₁ ∧ t (i+1) j = c₂) ∨
     (t i j = c₁ ∧ t i (j+1) = c₂) ∨
     (t i j = c₁ ∧ t (i+1) (j+1) = c₂) ∨
     (t (i+1) j = c₁ ∧ t i (j+1) = c₂) ∨
     (t (i+1) j = c₁ ∧ t (i+1) (j+1) = c₂) ∨
     (t i (j+1) = c₁ ∧ t (i+1) (j+1) = c₂))

/-- Checks if a 2x2 square starting at (i,j) is colored in chess order -/
def isChessOrder (t : Table) (i j : Fin 100) : Prop :=
  (t i j = Color.Black ∧ t (i+1) j = Color.White ∧ t i (j+1) = Color.White ∧ t (i+1) (j+1) = Color.Black) ∨
  (t i j = Color.White ∧ t (i+1) j = Color.Black ∧ t i (j+1) = Color.Black ∧ t (i+1) (j+1) = Color.White)

theorem chess_square_exists (t : Table) 
  (border_black : ∀ i j, isBorder i j → t i j = Color.Black)
  (two_colors : ∀ i j, hasTwoColors t i j) :
  ∃ i j, isChessOrder t i j := by
  sorry

end NUMINAMATH_CALUDE_chess_square_exists_l3049_304976


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l3049_304933

theorem factorial_difference_quotient : (Nat.factorial 13 - Nat.factorial 12) / Nat.factorial 10 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l3049_304933


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l3049_304934

/-- Given a line segment PQ with midpoint M(0,4) and P moving along x + y - 2 = 0,
    prove that the trajectory of Q is x + y - 6 = 0 -/
theorem trajectory_of_Q (P Q : ℝ × ℝ) (t : ℝ) : 
  let M := (0, 4)
  let P := (t, 2 - t)  -- parametric form of x + y - 2 = 0
  let Q := (2 * M.1 - P.1, 2 * M.2 - P.2)  -- Q is symmetric to P with respect to M
  Q.1 + Q.2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l3049_304934


namespace NUMINAMATH_CALUDE_oliver_baseball_cards_l3049_304931

theorem oliver_baseball_cards (cards_per_page new_cards old_cards : ℕ) 
  (h1 : cards_per_page = 3)
  (h2 : new_cards = 2)
  (h3 : old_cards = 10) :
  (new_cards + old_cards) / cards_per_page = 4 := by
  sorry

end NUMINAMATH_CALUDE_oliver_baseball_cards_l3049_304931


namespace NUMINAMATH_CALUDE_prism_18_edges_has_8_faces_l3049_304967

/-- A prism is a polyhedron with two congruent parallel faces (bases) and lateral faces that are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given the number of edges. -/
def num_faces (p : Prism) : ℕ :=
  2 + (p.edges / 3)  -- 2 bases + lateral faces

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_has_8_faces (p : Prism) (h : p.edges = 18) : num_faces p = 8 := by
  sorry


end NUMINAMATH_CALUDE_prism_18_edges_has_8_faces_l3049_304967


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3049_304915

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  a 1 + a 2 = 4 →
  d = 2 →
  a 7 + a 8 = 28 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3049_304915


namespace NUMINAMATH_CALUDE_calculation_proofs_l3049_304964

theorem calculation_proofs :
  (1) -2^2 * (1/4) + 4 / (4/9) + (-1)^2023 = 7 ∧
  (2) -1^4 + |2 - (-3)^2| + (1/2) / (-3/2) = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l3049_304964


namespace NUMINAMATH_CALUDE_domain_of_h_l3049_304981

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-12) 6

-- Define the function h in terms of f
def h (x : ℝ) : ℝ := f (-3 * x)

-- State the theorem about the domain of h
theorem domain_of_h :
  {x : ℝ | h x ∈ Set.range f} = Set.Icc (-2) 4 := by sorry

end NUMINAMATH_CALUDE_domain_of_h_l3049_304981


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3049_304928

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 = (X^2 + 3*X + 2) * q + (-15*X - 14) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3049_304928


namespace NUMINAMATH_CALUDE_factorization_theorem_l3049_304966

theorem factorization_theorem (a : ℝ) : (a^2 + 4)^2 - 16*a^2 = (a + 2)^2 * (a - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l3049_304966


namespace NUMINAMATH_CALUDE_solve_chocolates_problem_l3049_304908

def chocolates_problem (nick_chocolates : ℕ) (alix_multiplier : ℕ) (difference_after : ℕ) : Prop :=
  let initial_alix_chocolates := nick_chocolates * alix_multiplier
  let alix_chocolates_after := nick_chocolates + difference_after
  let chocolates_taken := initial_alix_chocolates - alix_chocolates_after
  chocolates_taken = 5

theorem solve_chocolates_problem :
  chocolates_problem 10 3 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_chocolates_problem_l3049_304908


namespace NUMINAMATH_CALUDE_factorization_problem1_l3049_304993

theorem factorization_problem1 (a b : ℝ) : -3 * a^2 + 6 * a * b - 3 * b^2 = -3 * (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem1_l3049_304993


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l3049_304956

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem ninth_term_of_arithmetic_sequence 
  (a₁ a₁₇ : ℚ) 
  (h₁ : a₁ = 3/4) 
  (h₁₇ : a₁₇ = 6/7) 
  (h_seq : ∃ d, ∀ n, arithmetic_sequence a₁ d n = a₁ + (n - 1) * d) :
  ∃ d, arithmetic_sequence a₁ d 9 = 45/56 :=
sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l3049_304956


namespace NUMINAMATH_CALUDE_water_speed_calculation_l3049_304947

/-- Proves that the speed of the water is 8 km/h given the swimming conditions -/
theorem water_speed_calculation (swimming_speed : ℝ) (distance : ℝ) (time : ℝ) :
  swimming_speed = 16 →
  distance = 12 →
  time = 1.5 →
  swimming_speed - (distance / time) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l3049_304947


namespace NUMINAMATH_CALUDE_first_bank_interest_rate_l3049_304919

/-- Proves that the interest rate of the first bank is 4% given the investment conditions --/
theorem first_bank_interest_rate 
  (total_investment : ℝ)
  (first_bank_investment : ℝ)
  (second_bank_rate : ℝ)
  (total_interest : ℝ)
  (h1 : total_investment = 5000)
  (h2 : first_bank_investment = 1700)
  (h3 : second_bank_rate = 0.065)
  (h4 : total_interest = 282.50)
  : ∃ (first_bank_rate : ℝ), 
    first_bank_rate = 0.04 ∧ 
    first_bank_investment * first_bank_rate + 
    (total_investment - first_bank_investment) * second_bank_rate = 
    total_interest := by
  sorry

end NUMINAMATH_CALUDE_first_bank_interest_rate_l3049_304919


namespace NUMINAMATH_CALUDE_min_value_abs_function_l3049_304997

theorem min_value_abs_function (x : ℝ) :
  ∀ x, |x - 1| + |x - 2| - |x - 3| ≥ -1 ∧ ∃ x, |x - 1| + |x - 2| - |x - 3| = -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abs_function_l3049_304997


namespace NUMINAMATH_CALUDE_combined_tax_rate_l3049_304970

/-- Calculate the combined tax rate for three individuals given their tax rates and income ratios -/
theorem combined_tax_rate
  (mork_rate : ℝ)
  (mindy_rate : ℝ)
  (orson_rate : ℝ)
  (mindy_income_ratio : ℝ)
  (orson_income_ratio : ℝ)
  (h1 : mork_rate = 0.45)
  (h2 : mindy_rate = 0.15)
  (h3 : orson_rate = 0.25)
  (h4 : mindy_income_ratio = 4)
  (h5 : orson_income_ratio = 2) :
  let total_tax := mork_rate + mindy_rate * mindy_income_ratio + orson_rate * orson_income_ratio
  let total_income := 1 + mindy_income_ratio + orson_income_ratio
  (total_tax / total_income) * 100 = 22.14 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l3049_304970


namespace NUMINAMATH_CALUDE_chase_cardinals_l3049_304972

/-- The number of birds Gabrielle saw -/
def gabrielle_birds : ℕ := 5 + 4 + 3

/-- The number of robins and blue jays Chase saw -/
def chase_known_birds : ℕ := 2 + 3

/-- The ratio of birds Gabrielle saw compared to Chase -/
def ratio : ℚ := 1.2

theorem chase_cardinals :
  ∃ (chase_total : ℕ) (chase_cardinals : ℕ),
    chase_total = (gabrielle_birds : ℚ) / ratio ∧
    chase_total = chase_known_birds + chase_cardinals ∧
    chase_cardinals = 5 := by
  sorry

end NUMINAMATH_CALUDE_chase_cardinals_l3049_304972


namespace NUMINAMATH_CALUDE_rectangle_division_distinctness_l3049_304974

theorem rectangle_division_distinctness (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  a * c = b * d →
  a + c = b + d →
  ¬(a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_division_distinctness_l3049_304974


namespace NUMINAMATH_CALUDE_simplify_fraction_1_l3049_304962

theorem simplify_fraction_1 (a b : ℝ) (h : a ≠ b) :
  (a^4 - b^4) / (a^2 - b^2) = a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_l3049_304962


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l3049_304965

theorem two_digit_number_puzzle : ∃ (n : ℕ) (x y : ℕ),
  0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
  n = 10 * x + y ∧
  x^2 + y^2 = 10 * x + y + 11 ∧
  2 * x * y = 10 * x + y - 5 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l3049_304965


namespace NUMINAMATH_CALUDE_marks_animals_legs_l3049_304912

def total_legs (num_kangaroos : ℕ) (num_goats : ℕ) : ℕ :=
  2 * num_kangaroos + 4 * num_goats

theorem marks_animals_legs : 
  let num_kangaroos : ℕ := 23
  let num_goats : ℕ := 3 * num_kangaroos
  total_legs num_kangaroos num_goats = 322 := by
sorry

end NUMINAMATH_CALUDE_marks_animals_legs_l3049_304912


namespace NUMINAMATH_CALUDE_expression_simplification_l3049_304903

theorem expression_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 + b^2 ≠ 0) :
  (1 / (a - b) - (2 * a * b) / (a^3 - a^2 * b + a * b^2 - b^3)) /
  ((a^2 + a * b) / (a^3 + a^2 * b + a * b^2 + b^3) + b / (a^2 + b^2)) =
  (a - b) / (a + b) :=
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3049_304903


namespace NUMINAMATH_CALUDE_unique_two_digit_number_divisible_by_eight_l3049_304905

theorem unique_two_digit_number_divisible_by_eight :
  ∃! n : ℕ, 70 < n ∧ n < 80 ∧ n % 8 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_divisible_by_eight_l3049_304905


namespace NUMINAMATH_CALUDE_books_borrowed_by_lunchtime_correct_books_borrowed_l3049_304983

theorem books_borrowed_by_lunchtime 
  (initial_books : ℕ) 
  (books_added : ℕ) 
  (books_borrowed_evening : ℕ) 
  (books_remaining : ℕ) : ℕ :=
  let books_borrowed_lunchtime := 
    initial_books + books_added - books_borrowed_evening - books_remaining
  books_borrowed_lunchtime

#check @books_borrowed_by_lunchtime

theorem correct_books_borrowed (
  initial_books : ℕ) 
  (books_added : ℕ) 
  (books_borrowed_evening : ℕ) 
  (books_remaining : ℕ) 
  (h1 : initial_books = 100) 
  (h2 : books_added = 40) 
  (h3 : books_borrowed_evening = 30) 
  (h4 : books_remaining = 60) :
  books_borrowed_by_lunchtime initial_books books_added books_borrowed_evening books_remaining = 50 := by
  sorry

end NUMINAMATH_CALUDE_books_borrowed_by_lunchtime_correct_books_borrowed_l3049_304983


namespace NUMINAMATH_CALUDE_triangle_area_is_16_l3049_304980

/-- The area of the triangle formed by the intersection of three lines -/
def triangleArea (line1 line2 line3 : ℝ → ℝ) : ℝ := sorry

/-- Line 1: y = 6 -/
def line1 : ℝ → ℝ := fun x ↦ 6

/-- Line 2: y = 2 + x -/
def line2 : ℝ → ℝ := fun x ↦ 2 + x

/-- Line 3: y = 2 - x -/
def line3 : ℝ → ℝ := fun x ↦ 2 - x

theorem triangle_area_is_16 : triangleArea line1 line2 line3 = 16 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_16_l3049_304980


namespace NUMINAMATH_CALUDE_common_value_theorem_l3049_304961

theorem common_value_theorem (a b : ℝ) 
  (h1 : a * (a - 4) = b * (b - 4))
  (h2 : a ≠ b)
  (h3 : a + b = 4) :
  a * (a - 4) = -3 := by
sorry

end NUMINAMATH_CALUDE_common_value_theorem_l3049_304961


namespace NUMINAMATH_CALUDE_sakshi_work_duration_l3049_304900

-- Define the efficiency ratio between Tanya and Sakshi
def efficiency_ratio : ℝ := 1.25

-- Define Tanya's work duration in days
def tanya_days : ℝ := 4

-- Theorem stating that Sakshi takes 5 days to complete the work
theorem sakshi_work_duration :
  efficiency_ratio * tanya_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_sakshi_work_duration_l3049_304900


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3049_304978

/-- Given a boat that travels 13 km/hr along a stream and 5 km/hr against the same stream,
    its speed in still water is 9 km/hr. -/
theorem boat_speed_in_still_water
  (speed_along_stream : ℝ)
  (speed_against_stream : ℝ)
  (h_along : speed_along_stream = 13)
  (h_against : speed_against_stream = 5) :
  (speed_along_stream + speed_against_stream) / 2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3049_304978


namespace NUMINAMATH_CALUDE_distance_between_hyperbola_and_ellipse_l3049_304992

theorem distance_between_hyperbola_and_ellipse 
  (x y z w : ℝ) 
  (h1 : x * y = 4) 
  (h2 : z^2 + 4 * w^2 = 4) : 
  (x - z)^2 + (y - w)^2 ≥ 1.6 := by
sorry

end NUMINAMATH_CALUDE_distance_between_hyperbola_and_ellipse_l3049_304992


namespace NUMINAMATH_CALUDE_initial_horses_l3049_304921

theorem initial_horses (sheep : ℕ) (chickens : ℕ) (goats : ℕ) (male_animals : ℕ) : 
  sheep = 29 → 
  chickens = 9 → 
  goats = 37 → 
  male_animals = 53 → 
  ∃ (horses : ℕ), 
    horses = 100 ∧ 
    (horses + sheep + chickens) / 2 + goats = male_animals * 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_horses_l3049_304921


namespace NUMINAMATH_CALUDE_part_one_part_two_l3049_304922

-- Part 1
theorem part_one (x : ℝ) (h1 : x^2 - 4*x + 3 < 0) (h2 : |x - 3| < 1) :
  2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h_pos : a > 0) 
  (h_subset : {x : ℝ | x^2 - 4*a*x + 3*a^2 ≥ 0} ⊂ {x : ℝ | |x - 3| ≥ 1}) :
  4/3 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3049_304922


namespace NUMINAMATH_CALUDE_replaced_student_weight_l3049_304930

theorem replaced_student_weight
  (n : ℕ)
  (new_weight : ℝ)
  (avg_decrease : ℝ)
  (h1 : n = 6)
  (h2 : new_weight = 62)
  (h3 : avg_decrease = 3)
  : ∃ (old_weight : ℝ), old_weight = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_replaced_student_weight_l3049_304930


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3049_304936

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_function_properties (a b : ℝ) :
  f a b 0 = 6 ∧ f a b 1 = 5 →
  (∀ x, f a b x = x^2 - 2*x + 6) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a b x ≥ 5) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a b x ≤ 14) ∧
  (∃ x ∈ Set.Icc (-2) 2, f a b x = 5) ∧
  (∃ x ∈ Set.Icc (-2) 2, f a b x = 14) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3049_304936
