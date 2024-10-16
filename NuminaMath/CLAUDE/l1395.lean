import Mathlib

namespace NUMINAMATH_CALUDE_box_side_face_area_l1395_139532

theorem box_side_face_area (L W H : ℝ) 
  (h1 : W * H = (1/2) * (L * W))
  (h2 : L * W = 1.5 * (H * L))
  (h3 : L * W * H = 648) :
  H * L = 72 := by
  sorry

end NUMINAMATH_CALUDE_box_side_face_area_l1395_139532


namespace NUMINAMATH_CALUDE_travel_speed_problem_l1395_139591

/-- Proves that given the conditions of the problem, the speeds of person A and person B are 4.5 km/h and 6 km/h respectively. -/
theorem travel_speed_problem (distance_A distance_B : ℝ) (speed_ratio : ℚ) (time_difference : ℝ) :
  distance_A = 6 →
  distance_B = 10 →
  speed_ratio = 3/4 →
  time_difference = 1/3 →
  ∃ (speed_A speed_B : ℝ),
    speed_A = 4.5 ∧
    speed_B = 6 ∧
    speed_A / speed_B = speed_ratio ∧
    distance_B / speed_B - distance_A / speed_A = time_difference :=
by sorry

end NUMINAMATH_CALUDE_travel_speed_problem_l1395_139591


namespace NUMINAMATH_CALUDE_circle_ratio_l1395_139510

theorem circle_ratio (a b : ℝ) (h : a > 0) (h' : b > 0) 
  (h_area : π * b^2 - π * a^2 = 4 * (π * a^2)) : 
  a / b = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l1395_139510


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l1395_139531

theorem greatest_four_digit_multiple_of_17 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 17 ∣ n → n ≤ 9996 ∧ 17 ∣ 9996 := by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l1395_139531


namespace NUMINAMATH_CALUDE_volleyball_match_probability_l1395_139563

/-- The probability of winning a single set for class 6 of senior year two -/
def win_prob : ℚ := 2/3

/-- The number of sets needed to win the match -/
def sets_to_win : ℕ := 3

/-- The probability of class 6 of senior year two winning by 3:0 -/
def prob_win_3_0 : ℚ := win_prob^sets_to_win

theorem volleyball_match_probability :
  prob_win_3_0 = 8/27 :=
sorry

end NUMINAMATH_CALUDE_volleyball_match_probability_l1395_139563


namespace NUMINAMATH_CALUDE_cube_max_volume_l1395_139541

/-- A cuboid with side lengths a, b, and c. -/
structure Cuboid where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The surface area of a cuboid. -/
def surfaceArea (x : Cuboid) : ℝ :=
  2 * (x.a * x.b + x.b * x.c + x.a * x.c)

/-- The volume of a cuboid. -/
def volume (x : Cuboid) : ℝ :=
  x.a * x.b * x.c

/-- Given a fixed surface area S, the cube maximizes the volume among all cuboids. -/
theorem cube_max_volume (S : ℝ) (h : 0 < S) :
  ∀ x : Cuboid, surfaceArea x = S →
    ∃ y : Cuboid, surfaceArea y = S ∧ y.a = y.b ∧ y.b = y.c ∧
      ∀ z : Cuboid, surfaceArea z = S → volume z ≤ volume y :=
by sorry

end NUMINAMATH_CALUDE_cube_max_volume_l1395_139541


namespace NUMINAMATH_CALUDE_unique_solution_implies_m_equals_3_l1395_139569

/-- For a quadratic equation ax^2 + bx + c = 0 to have exactly one solution,
    its discriminant (b^2 - 4ac) must be zero. -/
def has_exactly_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- The quadratic equation 3x^2 - 6x + m = 0 has exactly one solution
    if and only if m = 3. -/
theorem unique_solution_implies_m_equals_3 :
  ∀ m : ℝ, has_exactly_one_solution 3 (-6) m ↔ m = 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_m_equals_3_l1395_139569


namespace NUMINAMATH_CALUDE_second_train_speed_l1395_139596

/-- Given two trains leaving a station simultaneously, prove that if one train
    travels 200 miles at 50 MPH and the other travels 240 miles, and their
    average travel time is 4 hours, then the speed of the second train is 60 MPH. -/
theorem second_train_speed (distance1 distance2 speed1 avg_time : ℝ) :
  distance1 = 200 →
  distance2 = 240 →
  speed1 = 50 →
  avg_time = 4 →
  (distance1 / speed1 + distance2 / (distance2 / avg_time)) / 2 = avg_time →
  distance2 / avg_time = 60 := by
  sorry

#check second_train_speed

end NUMINAMATH_CALUDE_second_train_speed_l1395_139596


namespace NUMINAMATH_CALUDE_coefficient_x_fourth_power_l1395_139556

theorem coefficient_x_fourth_power (n : ℕ) (k : ℕ) : 
  n = 6 → k = 4 → (Nat.choose n k) * (2^k) = 240 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_fourth_power_l1395_139556


namespace NUMINAMATH_CALUDE_shaded_region_correct_l1395_139560

def shaded_region : Set ℂ := {z : ℂ | Complex.abs z ≤ 1 ∧ Complex.im z ≥ (1/2 : ℝ)}

theorem shaded_region_correct :
  ∀ z : ℂ, z ∈ shaded_region ↔ Complex.abs z ≤ 1 ∧ Complex.im z ≥ (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_shaded_region_correct_l1395_139560


namespace NUMINAMATH_CALUDE_correct_average_after_misreading_l1395_139565

theorem correct_average_after_misreading (n : ℕ) (incorrect_avg : ℚ) 
  (misread_numbers : List (ℚ × ℚ)) :
  n = 20 ∧ 
  incorrect_avg = 85 ∧ 
  misread_numbers = [(90, 30), (120, 60), (75, 25), (150, 50), (45, 15)] →
  (n : ℚ) * incorrect_avg + (misread_numbers.map (λ p => p.1 - p.2)).sum = n * 100 := by
  sorry

#check correct_average_after_misreading

end NUMINAMATH_CALUDE_correct_average_after_misreading_l1395_139565


namespace NUMINAMATH_CALUDE_x_value_in_terms_of_acd_l1395_139578

theorem x_value_in_terms_of_acd (x y z a b c d : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (ha : a ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : x * y / (x + y) = a)
  (h2 : x * z / (x + z) = b)
  (h3 : y * z / (y + z) = c)
  (h4 : y * z / (y - z) = d) :
  x = 2 * a * c / (a - c - d) := by
sorry

end NUMINAMATH_CALUDE_x_value_in_terms_of_acd_l1395_139578


namespace NUMINAMATH_CALUDE_dealer_truck_sales_l1395_139573

theorem dealer_truck_sales (total : ℕ) (car_truck_diff : ℕ) (trucks : ℕ) : 
  total = 69 → car_truck_diff = 27 → trucks + (trucks + car_truck_diff) = total → trucks = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_dealer_truck_sales_l1395_139573


namespace NUMINAMATH_CALUDE_c_monthly_income_l1395_139548

/-- Proves that C's monthly income is 17000, given the conditions from the problem -/
theorem c_monthly_income (a_annual_income : ℕ) (a_b_ratio : ℚ) (b_c_percentage : ℚ) :
  a_annual_income = 571200 →
  a_b_ratio = 5 / 2 →
  b_c_percentage = 112 / 100 →
  (a_annual_income / 12 : ℚ) * (2 / 5) / b_c_percentage = 17000 :=
by sorry

end NUMINAMATH_CALUDE_c_monthly_income_l1395_139548


namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l1395_139505

theorem square_difference_divided_by_nine : (104^2 - 95^2) / 9 = 199 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l1395_139505


namespace NUMINAMATH_CALUDE_shaded_area_circular_pattern_l1395_139543

/-- The area of the shaded region in a circular arc pattern -/
theorem shaded_area_circular_pattern (r : ℝ) (l : ℝ) : 
  r = 3 → l = 24 → (2 * l / (2 * r)) * (π * r^2 / 2) = 18 * π :=
by
  sorry

end NUMINAMATH_CALUDE_shaded_area_circular_pattern_l1395_139543


namespace NUMINAMATH_CALUDE_cone_surface_area_minimization_l1395_139594

/-- 
Given a right circular cone with fixed volume V, base radius R, and height H,
prove that H/R = 3 when the total surface area is minimized.
-/
theorem cone_surface_area_minimization (V : ℝ) (V_pos : V > 0) :
  ∃ (R H : ℝ), R > 0 ∧ H > 0 ∧
  (∀ (r h : ℝ), r > 0 → h > 0 → (1/3) * Real.pi * r^2 * h = V →
    R^2 * (Real.pi * R + Real.pi * Real.sqrt (R^2 + H^2)) ≤ 
    r^2 * (Real.pi * r + Real.pi * Real.sqrt (r^2 + h^2))) ∧
  H / R = 3 := by
  sorry


end NUMINAMATH_CALUDE_cone_surface_area_minimization_l1395_139594


namespace NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_200_l1395_139509

theorem largest_whole_number_nine_times_less_than_200 :
  ∀ x : ℕ, x ≤ 22 ↔ 9 * x < 200 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_200_l1395_139509


namespace NUMINAMATH_CALUDE_equation_solution_l1395_139568

theorem equation_solution : ∃ x : ℝ, 24 * 2 - 6 = 3 * x + 6 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1395_139568


namespace NUMINAMATH_CALUDE_range_of_m_l1395_139554

theorem range_of_m (x y m : ℝ) (h1 : 2/x + 1/y = 1) (h2 : x + y = 2 + 2*m) :
  -4 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1395_139554


namespace NUMINAMATH_CALUDE_count_special_numbers_is_360_l1395_139575

/-- A function that counts 4-digit numbers beginning with 1 and having exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits := Finset.range 10  -- digits from 0 to 9
  let non_one_digits := digits.erase 1  -- digits excluding 1
  let case1 := 3 * non_one_digits.card * (non_one_digits.card - 1)  -- case where one of the identical digits is 1
  let case2 := 2 * non_one_digits.card * digits.card  -- case where the identical digits are not 1
  case1 + case2

/-- Theorem stating that the count of special numbers is 360 -/
theorem count_special_numbers_is_360 : count_special_numbers = 360 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_is_360_l1395_139575


namespace NUMINAMATH_CALUDE_unique_function_property_l1395_139524

theorem unique_function_property (f : ℚ → ℚ) :
  (f 1 = 2) →
  (∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) →
  (∀ x : ℚ, f x = x + 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_property_l1395_139524


namespace NUMINAMATH_CALUDE_shared_side_angle_measure_l1395_139520

-- Define the properties of the figure
def regular_pentagon (P : Set Point) : Prop := sorry

def equilateral_triangle (T : Set Point) : Prop := sorry

def share_side (P T : Set Point) : Prop := sorry

-- Define the angle we're interested in
def angle_at_vertex (T : Set Point) (v : Point) : ℝ := sorry

-- Theorem statement
theorem shared_side_angle_measure 
  (P T : Set Point) (v : Point) :
  regular_pentagon P → 
  equilateral_triangle T → 
  share_side P T → 
  angle_at_vertex T v = 6 := by sorry

end NUMINAMATH_CALUDE_shared_side_angle_measure_l1395_139520


namespace NUMINAMATH_CALUDE_dallas_age_l1395_139502

theorem dallas_age (dexter_age : ℕ) (darcy_age : ℕ) (dallas_age_last_year : ℕ) :
  dexter_age = 8 →
  darcy_age = 2 * dexter_age →
  dallas_age_last_year = 3 * (darcy_age - 1) →
  dallas_age_last_year + 1 = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_dallas_age_l1395_139502


namespace NUMINAMATH_CALUDE_function_range_l1395_139507

theorem function_range (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = a + b * Real.cos x + c * Real.sin x) →
  f 0 = 1 →
  f (-π/4) = a →
  (∀ x ∈ Set.Icc 0 (π/2), |f x| ≤ Real.sqrt 2) →
  a ∈ Set.Icc 0 (4 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_function_range_l1395_139507


namespace NUMINAMATH_CALUDE_units_digit_of_power_l1395_139550

theorem units_digit_of_power (n : ℕ) : n % 10 = 7 → (n^1997 % 10)^2999 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_l1395_139550


namespace NUMINAMATH_CALUDE_sequence_fifth_term_l1395_139526

/-- Given a positive sequence {a_n}, prove that a_5 = 3 -/
theorem sequence_fifth_term (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_1 : a 1 = 1)
  (h_2 : a 2 = Real.sqrt 3)
  (h_rec : ∀ n ≥ 2, 2 * (a n)^2 = (a (n+1))^2 + (a (n-1))^2) :
  a 5 = 3 := by
sorry

end NUMINAMATH_CALUDE_sequence_fifth_term_l1395_139526


namespace NUMINAMATH_CALUDE_team_selection_ways_l1395_139549

def number_of_combinations (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem team_selection_ways (total_boys total_girls team_boys team_girls : ℕ) 
  (h1 : total_boys = 7)
  (h2 : total_girls = 9)
  (h3 : team_boys = 3)
  (h4 : team_girls = 3) :
  (number_of_combinations total_boys team_boys) * (number_of_combinations total_girls team_girls) = 2940 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_ways_l1395_139549


namespace NUMINAMATH_CALUDE_boxes_filled_in_five_minutes_l1395_139592

/-- A machine that fills boxes at a constant rate -/
structure BoxFillingMachine where
  boxes_per_hour : ℚ

/-- Given a machine that fills 24 boxes in 60 minutes, prove it fills 2 boxes in 5 minutes -/
theorem boxes_filled_in_five_minutes 
  (machine : BoxFillingMachine) 
  (h : machine.boxes_per_hour = 24 / 1) : 
  (machine.boxes_per_hour * 5 / 60 : ℚ) = 2 := by
  sorry


end NUMINAMATH_CALUDE_boxes_filled_in_five_minutes_l1395_139592


namespace NUMINAMATH_CALUDE_min_value_fraction_l1395_139519

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hsum : b + 2*a = 8) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 8 → 2/(x*y) ≥ 2/(a*b)) ∧ 2/(a*b) = 1/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1395_139519


namespace NUMINAMATH_CALUDE_function_value_at_two_l1395_139535

theorem function_value_at_two (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 2*x) : f 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l1395_139535


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1395_139572

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 5 + a 7 + a 9 + a 11 = 100) :
  3 * a 9 - a 13 = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1395_139572


namespace NUMINAMATH_CALUDE_cookies_difference_l1395_139590

def cookies_bought : ℝ := 125.75
def cookies_eaten : ℝ := 8.5

theorem cookies_difference : cookies_bought - cookies_eaten = 117.25 := by
  sorry

end NUMINAMATH_CALUDE_cookies_difference_l1395_139590


namespace NUMINAMATH_CALUDE_systematic_sampling_41st_number_l1395_139559

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  first_selected : ℕ

/-- The nth number in a systematic sample -/
def nth_number (s : SystematicSampling) (n : ℕ) : ℕ :=
  let part_size := s.total_students / s.sample_size
  (n - 1) * part_size + s.first_selected

theorem systematic_sampling_41st_number 
  (s : SystematicSampling) 
  (h1 : s.total_students = 1000) 
  (h2 : s.sample_size = 50) 
  (h3 : s.first_selected = 10) : 
  nth_number s 41 = 810 := by
  sorry

#eval nth_number { total_students := 1000, sample_size := 50, first_selected := 10 } 41

end NUMINAMATH_CALUDE_systematic_sampling_41st_number_l1395_139559


namespace NUMINAMATH_CALUDE_triangle_perimeter_example_l1395_139597

/-- A triangle with sides a, b, and c is valid if the sum of any two sides is greater than the third side. -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of a triangle is the sum of its sides. -/
def triangle_perimeter (a b c : ℝ) : ℝ :=
  a + b + c

/-- Theorem: The perimeter of a triangle with sides 12, 15, and 9 is 36. -/
theorem triangle_perimeter_example : 
  is_valid_triangle 12 15 9 → triangle_perimeter 12 15 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_example_l1395_139597


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l1395_139513

/-- Represents a circular table with chairs and people seated. -/
structure CircularTable where
  totalChairs : ℕ
  seatedPeople : ℕ

/-- Checks if the seating arrangement is valid. -/
def isValidSeating (table : CircularTable) : Prop :=
  table.seatedPeople > 0 ∧ 
  table.seatedPeople ≤ table.totalChairs ∧
  ∀ n : ℕ, n ≤ table.seatedPeople → ∃ m : ℕ, m < n ∧ (n - m = 1 ∨ m - n = 1 ∨ n = 1)

/-- The theorem to be proved. -/
theorem smallest_valid_seating (table : CircularTable) :
  table.totalChairs = 75 →
  (isValidSeating table ∧ ∀ t : CircularTable, t.totalChairs = 75 → isValidSeating t → t.seatedPeople ≥ table.seatedPeople) →
  table.seatedPeople = 25 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_l1395_139513


namespace NUMINAMATH_CALUDE_equation_solution_l1395_139529

theorem equation_solution : ∃! x : ℤ, 45 - (28 - (x - (15 - 19))) = 58 ∧ x = 37 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1395_139529


namespace NUMINAMATH_CALUDE_weekly_distance_calculation_l1395_139539

/-- Calculates the weekly running distance given the number of days, hours per day, and speed. -/
def weekly_running_distance (days_per_week : ℕ) (hours_per_day : ℝ) (speed_mph : ℝ) : ℝ :=
  days_per_week * hours_per_day * speed_mph

/-- Proves that running 5 days a week, 1.5 hours each day, at 8 mph results in 60 miles per week. -/
theorem weekly_distance_calculation :
  weekly_running_distance 5 1.5 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_weekly_distance_calculation_l1395_139539


namespace NUMINAMATH_CALUDE_third_speed_calculation_l1395_139518

/-- Prove that given the conditions, the third speed is 3 km/hr -/
theorem third_speed_calculation (total_time : ℝ) (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 11 →
  total_distance = 900 →
  speed1 = 6 →
  speed2 = 9 →
  ∃ (speed3 : ℝ), speed3 = 3 ∧
    total_time = (total_distance / 3) / (speed1 * 1000 / 60) +
                 (total_distance / 3) / (speed2 * 1000 / 60) +
                 (total_distance / 3) / (speed3 * 1000 / 60) :=
by sorry


end NUMINAMATH_CALUDE_third_speed_calculation_l1395_139518


namespace NUMINAMATH_CALUDE_square_of_binomial_l1395_139501

theorem square_of_binomial (m n : ℝ) : (3*m - n)^2 = 9*m^2 - 6*m*n + n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l1395_139501


namespace NUMINAMATH_CALUDE_clock_angle_theorem_l1395_139580

/-- The angle (in degrees) the minute hand moves per minute -/
def minute_hand_speed : ℝ := 6

/-- The angle (in degrees) the hour hand moves per minute -/
def hour_hand_speed : ℝ := 0.5

/-- The current time in minutes past 3:00 -/
def t : ℝ := 23

/-- The position of the minute hand 8 minutes from now -/
def minute_hand_pos : ℝ := minute_hand_speed * (t + 8)

/-- The position of the hour hand 4 minutes ago -/
def hour_hand_pos : ℝ := 90 + hour_hand_speed * (t - 4)

/-- The theorem stating that the time is approximately 23 minutes past 3:00 -/
theorem clock_angle_theorem : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  (|minute_hand_pos - hour_hand_pos| = 90 ∨ 
   |minute_hand_pos - hour_hand_pos| = 270) ∧
  t ≥ 0 ∧ t < 60 ∧ 
  |t - 23| < ε :=
sorry

end NUMINAMATH_CALUDE_clock_angle_theorem_l1395_139580


namespace NUMINAMATH_CALUDE_parabola_properties_l1395_139530

/-- Represents a parabola of the form y = ax^2 + bx - 4 -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x - 4

theorem parabola_properties (p : Parabola) 
  (h1 : p.contains (-2) 0)
  (h2 : p.contains (-1) (-4))
  (h3 : p.contains 0 (-4))
  (h4 : p.contains 1 0)
  (h5 : p.contains 2 8) :
  (p.contains 0 (-4)) ∧ 
  (p.a = 2 ∧ p.b = 2) ∧ 
  (p.contains (-3) 8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1395_139530


namespace NUMINAMATH_CALUDE_triangle_side_length_l1395_139547

theorem triangle_side_length (a b : ℝ) (A B : ℝ) :
  b = 4 * Real.sqrt 6 →
  B = π / 3 →
  A = π / 4 →
  a = (4 * Real.sqrt 6) * (Real.sin (π / 4)) / (Real.sin (π / 3)) →
  a = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1395_139547


namespace NUMINAMATH_CALUDE_circle_centers_distance_l1395_139551

/-- Given a right triangle XYZ with side lengths XY = 7, XZ = 24, and YZ = 25,
    and two circles: one centered at O tangent to XZ at Z and passing through Y,
    and another centered at P tangent to XY at Y and passing through Z,
    prove that the length of OP is 25. -/
theorem circle_centers_distance (X Y Z O P : ℝ × ℝ) : 
  -- Right triangle XYZ with given side lengths
  (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = 7^2 →
  (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = 24^2 →
  (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 = 25^2 →
  -- Circle O is tangent to XZ at Z and passes through Y
  ((O.1 - Z.1)^2 + (O.2 - Z.2)^2 = (O.1 - Y.1)^2 + (O.2 - Y.2)^2) →
  ((O.1 - Z.1) * (Z.1 - X.1) + (O.2 - Z.2) * (Z.2 - X.2) = 0) →
  -- Circle P is tangent to XY at Y and passes through Z
  ((P.1 - Y.1)^2 + (P.2 - Y.2)^2 = (P.1 - Z.1)^2 + (P.2 - Z.2)^2) →
  ((P.1 - Y.1) * (Y.1 - X.1) + (P.2 - Y.2) * (Y.2 - X.2) = 0) →
  -- The distance between O and P is 25
  (O.1 - P.1)^2 + (O.2 - P.2)^2 = 25^2 := by
sorry


end NUMINAMATH_CALUDE_circle_centers_distance_l1395_139551


namespace NUMINAMATH_CALUDE_max_cross_section_area_l1395_139540

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  sideLength : ℝ
  sideLength_pos : 0 < sideLength

/-- Represents a plane that cuts the tetrahedron parallel to two opposite edges -/
structure CuttingPlane where
  tetrahedron : RegularTetrahedron
  distanceFromEdge : ℝ
  distance_nonneg : 0 ≤ distanceFromEdge
  distance_bound : distanceFromEdge ≤ tetrahedron.sideLength

/-- The area of the cross-section formed by the cutting plane -/
def crossSectionArea (plane : CuttingPlane) : ℝ :=
  plane.distanceFromEdge * (plane.tetrahedron.sideLength - plane.distanceFromEdge)

/-- The theorem stating that the maximum cross-section area is a²/4 -/
theorem max_cross_section_area (t : RegularTetrahedron) :
  ∃ (plane : CuttingPlane), plane.tetrahedron = t ∧
  ∀ (p : CuttingPlane), p.tetrahedron = t →
  crossSectionArea p ≤ crossSectionArea plane ∧
  crossSectionArea plane = t.sideLength^2 / 4 :=
sorry

end NUMINAMATH_CALUDE_max_cross_section_area_l1395_139540


namespace NUMINAMATH_CALUDE_parallel_to_y_axis_fourth_quadrant_integer_a_l1395_139562

-- Define point A
def A (a : ℝ) : ℝ × ℝ := (3*a - 9, 2*a - 10)

-- Define point B
def B : ℝ × ℝ := (4, 5)

-- Theorem 1
theorem parallel_to_y_axis (a : ℝ) : 
  (A a).1 = B.1 → a = 13/3 := by sorry

-- Theorem 2
theorem fourth_quadrant_integer_a : 
  ∃ (a : ℤ), (A a).1 > 0 ∧ (A a).2 < 0 → A a = (3, -2) := by sorry

end NUMINAMATH_CALUDE_parallel_to_y_axis_fourth_quadrant_integer_a_l1395_139562


namespace NUMINAMATH_CALUDE_family_game_score_l1395_139599

theorem family_game_score : 
  let dad_score : ℕ := 7
  let olaf_score : ℕ := 3 * dad_score
  let sister_score : ℕ := dad_score + 4
  let mom_score : ℕ := 2 * sister_score
  dad_score + olaf_score + sister_score + mom_score = 61 :=
by sorry

end NUMINAMATH_CALUDE_family_game_score_l1395_139599


namespace NUMINAMATH_CALUDE_x_less_than_negative_one_sufficient_not_necessary_l1395_139586

theorem x_less_than_negative_one_sufficient_not_necessary :
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_negative_one_sufficient_not_necessary_l1395_139586


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l1395_139593

-- Define the ellipse (C)
def ellipse (x y : ℝ) : Prop := x^2/12 + y^2/3 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - 8*y^2 = 8

-- Define the line (l)
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x+3)

-- Define the circle with PQ as diameter passing through origin
def circle_PQ_through_origin (P Q : ℝ × ℝ) : Prop :=
  (P.1 * Q.1 + P.2 * Q.2 = 0)

theorem ellipse_and_line_theorem :
  -- The ellipse passes through (-2,√2)
  ellipse (-2) (Real.sqrt 2) →
  -- The ellipse and hyperbola share the same foci
  (∀ x y, hyperbola x y ↔ x^2/8 - y^2 = 1) →
  -- For any k, if the line intersects the ellipse at P and Q
  -- and the circle with PQ as diameter passes through origin
  (∀ k P Q, 
    line k P.1 P.2 → 
    line k Q.1 Q.2 → 
    ellipse P.1 P.2 → 
    ellipse Q.1 Q.2 → 
    circle_PQ_through_origin P Q →
    -- Then k must be ±(2√11/11)
    (k = 2 * Real.sqrt 11 / 11 ∨ k = -2 * Real.sqrt 11 / 11)) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l1395_139593


namespace NUMINAMATH_CALUDE_expression_simplification_l1395_139516

/-- For x in the open interval (0, 1], the given expression simplifies to ∛((1-x)/(3x)) -/
theorem expression_simplification (x : ℝ) (h : 0 < x ∧ x ≤ 1) :
  1.37 * Real.rpow ((2 * x^2) / (9 + 18*x + 9*x^2)) (1/3) *
  Real.sqrt (((1 + x) * Real.rpow (1 - x) (1/3)) / x) *
  Real.rpow ((3 * Real.sqrt (1 - x^2)) / (2 * x * Real.sqrt x)) (1/3) =
  Real.rpow ((1 - x) / (3 * x)) (1/3) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l1395_139516


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l1395_139558

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (1/2) * a * b = 10 →
  a^2 + b^2 = 100 →
  (1/2) * c * d = 250 →
  c/a = d/b →
  c + d = 30 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l1395_139558


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1395_139585

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 2) :
  (2 * sin α - cos α) / (sin α + 2 * cos α) = 3/4 ∧
  2 * sin α^2 - sin α * cos α + cos α^2 = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1395_139585


namespace NUMINAMATH_CALUDE_optimal_discount_order_l1395_139587

/-- Proves that the optimal order of applying discounts results in an additional savings of 125 cents --/
theorem optimal_discount_order (initial_price : ℝ) (flat_discount : ℝ) (percent_discount : ℝ) :
  initial_price = 30 →
  flat_discount = 5 →
  percent_discount = 0.25 →
  ((initial_price - flat_discount) * (1 - percent_discount) - 
   (initial_price * (1 - percent_discount) - flat_discount)) * 100 = 125 := by
  sorry

end NUMINAMATH_CALUDE_optimal_discount_order_l1395_139587


namespace NUMINAMATH_CALUDE_slope_equals_twelve_implies_m_equals_negative_two_l1395_139598

/-- Given two points A(-m, 6) and B(1, 3m), prove that m = -2 when the slope of the line passing through these points is 12. -/
theorem slope_equals_twelve_implies_m_equals_negative_two (m : ℝ) : 
  (let A : ℝ × ℝ := (-m, 6)
   let B : ℝ × ℝ := (1, 3*m)
   (3*m - 6) / (1 - (-m)) = 12) → m = -2 := by
sorry

end NUMINAMATH_CALUDE_slope_equals_twelve_implies_m_equals_negative_two_l1395_139598


namespace NUMINAMATH_CALUDE_pascal_triangle_32nd_row_31st_element_l1395_139571

theorem pascal_triangle_32nd_row_31st_element : Nat.choose 32 30 = 496 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_32nd_row_31st_element_l1395_139571


namespace NUMINAMATH_CALUDE_kris_bullying_instances_l1395_139536

/-- The number of days Kris is suspended for each bullying instance -/
def suspension_days_per_instance : ℕ := 3

/-- The total number of fingers and toes a typical person has -/
def typical_person_digits : ℕ := 20

/-- The total number of days Kris has been suspended -/
def total_suspension_days : ℕ := 3 * typical_person_digits

/-- The number of bullying instances Kris is responsible for -/
def bullying_instances : ℕ := total_suspension_days / suspension_days_per_instance

theorem kris_bullying_instances : bullying_instances = 20 := by
  sorry

end NUMINAMATH_CALUDE_kris_bullying_instances_l1395_139536


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l1395_139588

/-- Given two perpendicular lines and the foot of the perpendicular, prove that a + b + c = -4 -/
theorem perpendicular_lines_sum (a b c : ℝ) : 
  (∀ x y, a * x + 4 * y - 2 = 0 ↔ 2 * x - 5 * y + b = 0) →  -- lines are perpendicular
  (a + 4 * c - 2 = 0) →  -- foot of perpendicular satisfies first line equation
  (2 - 5 * c + b = 0) →  -- foot of perpendicular satisfies second line equation
  (a * 2 + 4 * 5 = 0) →  -- perpendicularity condition
  a + b + c = -4 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l1395_139588


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1395_139504

/-- Represents a repeating decimal in the form 0.nnn... where n is a single digit -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

theorem repeating_decimal_sum :
  RepeatingDecimal 6 - RepeatingDecimal 4 + RepeatingDecimal 8 = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1395_139504


namespace NUMINAMATH_CALUDE_percent_of_x_is_y_l1395_139582

theorem percent_of_x_is_y (x y : ℝ) (h : 0.6 * (x - y) = 0.2 * (x + y)) : y = 0.5 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_is_y_l1395_139582


namespace NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l1395_139544

/-- The value of m for which the parabola y = x^2 + 5 and the hyperbola y^2 - mx^2 = 1 are tangent -/
theorem parabola_hyperbola_tangency (m : ℝ) : 
  (∃ x y : ℝ, y = x^2 + 5 ∧ y^2 - m*x^2 = 1 ∧ 
    (∀ x' y' : ℝ, y' = x'^2 + 5 → y'^2 - m*x'^2 = 1 → (x', y') = (x, y) ∨ (x', y') ≠ (x, y))) →
  m = 10 + 4*Real.sqrt 6 ∨ m = 10 - 4*Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l1395_139544


namespace NUMINAMATH_CALUDE_wheel_rotations_per_block_l1395_139517

theorem wheel_rotations_per_block 
  (total_blocks : ℕ) 
  (initial_rotations : ℕ) 
  (additional_rotations : ℕ) : 
  total_blocks = 8 → 
  initial_rotations = 600 → 
  additional_rotations = 1000 → 
  (initial_rotations + additional_rotations) / total_blocks = 200 := by
sorry

end NUMINAMATH_CALUDE_wheel_rotations_per_block_l1395_139517


namespace NUMINAMATH_CALUDE_mika_initial_stickers_l1395_139567

/-- The number of stickers Mika had initially -/
def initial_stickers : ℕ := 26

/-- The number of stickers Mika bought -/
def bought_stickers : ℕ := 26

/-- The number of stickers Mika got for her birthday -/
def birthday_stickers : ℕ := 20

/-- The number of stickers Mika gave to her sister -/
def given_stickers : ℕ := 6

/-- The number of stickers Mika used for the greeting card -/
def used_stickers : ℕ := 58

/-- The number of stickers Mika is left with -/
def remaining_stickers : ℕ := 2

theorem mika_initial_stickers :
  initial_stickers + bought_stickers + birthday_stickers - given_stickers - used_stickers = remaining_stickers :=
by
  sorry

#check mika_initial_stickers

end NUMINAMATH_CALUDE_mika_initial_stickers_l1395_139567


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1395_139525

/-- Given a circle with equation x^2 + y^2 - 2x + 6y + 9 = 0, prove that its center is at (1, -3) and its radius is 1 -/
theorem circle_center_and_radius :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 2*x + 6*y + 9 = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -3) ∧ radius = 1 ∧
    ∀ (x y : ℝ), circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1395_139525


namespace NUMINAMATH_CALUDE_additional_plates_count_l1395_139534

/-- The number of choices for the first letter in the original configuration -/
def original_first : Nat := 5

/-- The number of choices for the second letter in the original configuration -/
def original_second : Nat := 3

/-- The number of choices for the third letter in both original and new configurations -/
def third : Nat := 4

/-- The number of choices for the first letter in the new configuration -/
def new_first : Nat := 6

/-- The number of choices for the second letter in the new configuration -/
def new_second : Nat := 4

/-- The number of additional license plates that can be made -/
def additional_plates : Nat := new_first * new_second * third - original_first * original_second * third

theorem additional_plates_count : additional_plates = 36 := by
  sorry

end NUMINAMATH_CALUDE_additional_plates_count_l1395_139534


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l1395_139577

theorem same_remainder_divisor : ∃ (N : ℕ), N > 1 ∧ 
  N = 23 ∧ 
  (∀ (k : ℕ), k > N → ¬(1743 % k = 2019 % k ∧ 2019 % k = 3008 % k)) ∧
  (1743 % N = 2019 % N ∧ 2019 % N = 3008 % N) :=
by sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l1395_139577


namespace NUMINAMATH_CALUDE_bagel_cost_is_1_50_l1395_139521

/-- The cost of a cup of coffee -/
def coffee_cost : ℝ := sorry

/-- The cost of a bagel -/
def bagel_cost : ℝ := sorry

/-- Condition 1: 3 cups of coffee and 2 bagels cost $12.75 -/
axiom condition1 : 3 * coffee_cost + 2 * bagel_cost = 12.75

/-- Condition 2: 2 cups of coffee and 5 bagels cost $14.00 -/
axiom condition2 : 2 * coffee_cost + 5 * bagel_cost = 14.00

/-- Theorem: The cost of one bagel is $1.50 -/
theorem bagel_cost_is_1_50 : bagel_cost = 1.50 := by sorry

end NUMINAMATH_CALUDE_bagel_cost_is_1_50_l1395_139521


namespace NUMINAMATH_CALUDE_mangoes_rate_per_kg_l1395_139584

/-- Given Bruce's purchase of grapes and mangoes, prove the rate per kg for mangoes. -/
theorem mangoes_rate_per_kg 
  (grapes_quantity : ℕ) 
  (grapes_rate : ℕ) 
  (mangoes_quantity : ℕ) 
  (total_paid : ℕ) 
  (h1 : grapes_quantity = 8)
  (h2 : grapes_rate = 70)
  (h3 : mangoes_quantity = 11)
  (h4 : total_paid = 1165)
  (h5 : total_paid = grapes_quantity * grapes_rate + mangoes_quantity * (total_paid - grapes_quantity * grapes_rate) / mangoes_quantity) : 
  (total_paid - grapes_quantity * grapes_rate) / mangoes_quantity = 55 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_rate_per_kg_l1395_139584


namespace NUMINAMATH_CALUDE_andrews_age_l1395_139522

theorem andrews_age (a : ℕ) (g : ℕ) : 
  g = 12 * a →  -- Andrew's grandfather's age is twelve times Andrew's age
  g - a = 55 →  -- Andrew's grandfather was 55 years old when Andrew was born
  a = 5 :=       -- Andrew's age is 5 years
by sorry

end NUMINAMATH_CALUDE_andrews_age_l1395_139522


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l1395_139576

theorem half_abs_diff_squares_20_15 : (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l1395_139576


namespace NUMINAMATH_CALUDE_prime_divisibility_pairs_l1395_139515

theorem prime_divisibility_pairs (n p : ℕ) : 
  p.Prime → 
  n ≤ 2 * p → 
  (p - 1)^n + 1 ∣ n^(p - 1) → 
  ((n = 1 ∧ p.Prime) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_pairs_l1395_139515


namespace NUMINAMATH_CALUDE_convex_curve_triangle_inequalities_l1395_139570

/-- A convex curve in a metric space -/
class ConvexCurve (α : Type*) [MetricSpace α]

/-- The distance between two convex curves -/
def curve_distance {α : Type*} [MetricSpace α] (A B : ConvexCurve α) : ℝ := sorry

/-- Triangle inequalities for distances between convex curves -/
theorem convex_curve_triangle_inequalities
  {α : Type*} [MetricSpace α]
  (A B C : ConvexCurve α) :
  let AB := curve_distance A B
  let BC := curve_distance B C
  let AC := curve_distance A C
  (AB + BC ≥ AC) ∧ (AC + BC ≥ AB) ∧ (AB + AC ≥ BC) :=
by sorry

end NUMINAMATH_CALUDE_convex_curve_triangle_inequalities_l1395_139570


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_is_1620_l1395_139579

def arithmeticSeriesSum (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_is_1620 :
  arithmeticSeriesSum 10 30 (1/4) = 1620 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_is_1620_l1395_139579


namespace NUMINAMATH_CALUDE_exists_multiple_indecomposable_factorizations_l1395_139508

/-- The set V_n for a given positive integer n -/
def V_n (n : ℕ) : Set ℕ := {m : ℕ | ∃ k : ℕ+, m = 1 + k * n}

/-- A number is indecomposable in V_n if it cannot be expressed as a product of two numbers in V_n -/
def Indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → p * q ≠ m

/-- Main theorem: There exists a number in V_n that can be expressed as a product of
    indecomposable numbers in V_n in more than one way -/
theorem exists_multiple_indecomposable_factorizations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ (a b c d : ℕ) (ha : Indecomposable n a) (hb : Indecomposable n b)
      (hc : Indecomposable n c) (hd : Indecomposable n d),
    r = a * b ∧ r = c * d ∧ (a ≠ c ∨ b ≠ d) :=
  sorry

end NUMINAMATH_CALUDE_exists_multiple_indecomposable_factorizations_l1395_139508


namespace NUMINAMATH_CALUDE_problem_solution_l1395_139514

theorem problem_solution : (2023^2 - 2023 - 4^2) / 2023 = 2022 - 16/2023 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1395_139514


namespace NUMINAMATH_CALUDE_x_is_negative_l1395_139581

theorem x_is_negative (x y : ℝ) (h1 : y ≠ 0) (h2 : y > 0) (h3 : x / y < -3) : x < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_is_negative_l1395_139581


namespace NUMINAMATH_CALUDE_second_train_length_l1395_139555

/-- Calculates the length of the second train given the parameters of two trains approaching each other -/
theorem second_train_length 
  (length_train1 : ℝ) 
  (speed_train1 : ℝ) 
  (speed_train2 : ℝ) 
  (clear_time : ℝ) 
  (h1 : length_train1 = 100) 
  (h2 : speed_train1 = 42) 
  (h3 : speed_train2 = 30) 
  (h4 : clear_time = 12.998960083193344) :
  ∃ length_train2 : ℝ, abs (length_train2 - 159.98) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_second_train_length_l1395_139555


namespace NUMINAMATH_CALUDE_zeros_after_decimal_of_fraction_l1395_139537

/-- The number of zeros after the decimal point in the decimal representation of 1/(100^15) -/
def zeros_after_decimal : ℕ := 30

/-- The fraction we're considering -/
def fraction : ℚ := 1 / (100 ^ 15)

theorem zeros_after_decimal_of_fraction :
  (∃ (x : ℚ), x * 10^zeros_after_decimal = fraction ∧ 
   x ≥ 1/10 ∧ x < 1) ∧
  (∀ (n : ℕ), n < zeros_after_decimal → 
   ∃ (y : ℚ), y * 10^n = fraction ∧ y < 1/10) :=
sorry

end NUMINAMATH_CALUDE_zeros_after_decimal_of_fraction_l1395_139537


namespace NUMINAMATH_CALUDE_tan_increasing_on_interval_l1395_139574

open Real

theorem tan_increasing_on_interval :
  StrictMonoOn tan (Set.Ioo (π / 2) π) := by
  sorry

end NUMINAMATH_CALUDE_tan_increasing_on_interval_l1395_139574


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_l1395_139583

theorem quadratic_always_nonnegative (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + 1 ≥ 0) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_l1395_139583


namespace NUMINAMATH_CALUDE_fred_initial_money_l1395_139546

/-- Fred's money situation --/
def fred_money_problem (initial_money current_money weekend_earnings : ℕ) : Prop :=
  initial_money + weekend_earnings = current_money

theorem fred_initial_money : 
  ∃ (initial_money : ℕ), fred_money_problem initial_money 86 63 ∧ initial_money = 23 :=
sorry

end NUMINAMATH_CALUDE_fred_initial_money_l1395_139546


namespace NUMINAMATH_CALUDE_square_sum_theorem_l1395_139564

theorem square_sum_theorem (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 8) : x^2 + y^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l1395_139564


namespace NUMINAMATH_CALUDE_point_coordinates_l1395_139566

theorem point_coordinates (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : y = 2) (h4 : x = 4) :
  (x, y) = (4, 2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1395_139566


namespace NUMINAMATH_CALUDE_fifth_root_unity_sum_l1395_139553

theorem fifth_root_unity_sum (x : ℂ) : x^5 = 1 → 1 + x^4 + x^8 + x^12 + x^16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_unity_sum_l1395_139553


namespace NUMINAMATH_CALUDE_binary_to_decimal_101001_l1395_139512

/-- Converts a list of binary digits to its decimal representation -/
def binaryToDecimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number we want to convert -/
def binaryNumber : List Nat := [1, 0, 1, 0, 0, 1]

/-- Theorem stating that the binary number 101001 is equal to the decimal number 41 -/
theorem binary_to_decimal_101001 :
  binaryToDecimal binaryNumber = 41 := by
  sorry

#eval binaryToDecimal binaryNumber

end NUMINAMATH_CALUDE_binary_to_decimal_101001_l1395_139512


namespace NUMINAMATH_CALUDE_decagon_equilateral_triangles_l1395_139595

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Count of distinct equilateral triangles in a regular polygon -/
def countDistinctEquilateralTriangles (n : ℕ) (p : RegularPolygon n) : ℕ :=
  sorry

/-- Theorem: In a ten-sided regular polygon, there are 82 distinct equilateral triangles
    with at least two vertices from the set of polygon vertices -/
theorem decagon_equilateral_triangles :
  ∀ (p : RegularPolygon 10), countDistinctEquilateralTriangles 10 p = 82 :=
by sorry

end NUMINAMATH_CALUDE_decagon_equilateral_triangles_l1395_139595


namespace NUMINAMATH_CALUDE_relay_race_probability_l1395_139528

/-- The number of short-distance runners --/
def total_runners : ℕ := 6

/-- The number of runners needed for the relay race --/
def team_size : ℕ := 4

/-- The probability that athlete A is not running the first leg
    and athlete B is not running the last leg in a 4x100 meter relay race --/
theorem relay_race_probability : 
  (total_runners.factorial / (total_runners - team_size).factorial - 
   (total_runners - 1).factorial / (total_runners - team_size).factorial - 
   (total_runners - 1).factorial / (total_runners - team_size).factorial + 
   (total_runners - 2).factorial / (total_runners - team_size + 1).factorial) /
  (total_runners.factorial / (total_runners - team_size).factorial) = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_relay_race_probability_l1395_139528


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1395_139506

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℚ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1395_139506


namespace NUMINAMATH_CALUDE_tomato_cucumber_price_difference_l1395_139523

theorem tomato_cucumber_price_difference :
  ∀ (tomato_price cucumber_price : ℝ),
  tomato_price < cucumber_price →
  cucumber_price = 5 →
  2 * tomato_price + 3 * cucumber_price = 23 →
  (cucumber_price - tomato_price) / cucumber_price = 0.2 :=
by
  sorry

end NUMINAMATH_CALUDE_tomato_cucumber_price_difference_l1395_139523


namespace NUMINAMATH_CALUDE_harry_pumpkin_packets_l1395_139527

/-- The number of pumpkin seed packets Harry bought -/
def pumpkin_packets : ℕ := 3

/-- The cost of one packet of pumpkin seeds in dollars -/
def pumpkin_cost : ℚ := 2.5

/-- The cost of one packet of tomato seeds in dollars -/
def tomato_cost : ℚ := 1.5

/-- The cost of one packet of chili pepper seeds in dollars -/
def chili_cost : ℚ := 0.9

/-- The number of tomato seed packets Harry bought -/
def tomato_packets : ℕ := 4

/-- The number of chili pepper seed packets Harry bought -/
def chili_packets : ℕ := 5

/-- The total amount Harry spent in dollars -/
def total_spent : ℚ := 18

theorem harry_pumpkin_packets :
  pumpkin_packets * pumpkin_cost + 
  tomato_packets * tomato_cost + 
  chili_packets * chili_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_harry_pumpkin_packets_l1395_139527


namespace NUMINAMATH_CALUDE_apple_distribution_l1395_139538

theorem apple_distribution (martha_initial : ℕ) (jane_apples : ℕ) (martha_final : ℕ) (martha_remaining : ℕ) :
  martha_initial = 20 →
  jane_apples = 5 →
  martha_remaining = 4 →
  martha_final = martha_remaining + 4 →
  martha_initial - jane_apples - martha_final = jane_apples + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1395_139538


namespace NUMINAMATH_CALUDE_unique_solution_equals_three_l1395_139589

theorem unique_solution_equals_three :
  ∃! (x : ℝ), (x^2 - t*x + 36 = 0) ∧ (x^2 - 8*x + t = 0) ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equals_three_l1395_139589


namespace NUMINAMATH_CALUDE_balls_after_2010_actions_l1395_139552

/-- Represents the state of boxes with balls -/
def BoxState := List Nat

/-- Adds a ball to the first available box and empties boxes to its left -/
def addBall (state : BoxState) : BoxState :=
  match state with
  | [] => [1]
  | (h::t) => if h < 6 then (h+1)::t else 0::addBall t

/-- Performs the ball-adding process n times -/
def performActions (n : Nat) : BoxState :=
  match n with
  | 0 => []
  | n+1 => addBall (performActions n)

/-- Calculates the sum of balls in all boxes -/
def totalBalls (state : BoxState) : Nat :=
  state.sum

/-- The main theorem to prove -/
theorem balls_after_2010_actions :
  totalBalls (performActions 2010) = 16 := by
  sorry

end NUMINAMATH_CALUDE_balls_after_2010_actions_l1395_139552


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1395_139533

theorem exponent_multiplication (a : ℝ) : a^3 * a = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1395_139533


namespace NUMINAMATH_CALUDE_exist_expression_24_set1_exist_expression_24_set2_l1395_139542

-- Define a type for arithmetic operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a type for arithmetic expressions
inductive Expr
  | Num (n : ℕ)
  | BinOp (op : Operation) (e1 e2 : Expr)

-- Define a function to evaluate expressions
def eval : Expr → ℚ
  | Expr.Num n => n
  | Expr.BinOp Operation.Add e1 e2 => eval e1 + eval e2
  | Expr.BinOp Operation.Sub e1 e2 => eval e1 - eval e2
  | Expr.BinOp Operation.Mul e1 e2 => eval e1 * eval e2
  | Expr.BinOp Operation.Div e1 e2 => eval e1 / eval e2

-- Define a function to check if an expression uses all given numbers exactly once
def usesAllNumbers (e : Expr) (nums : List ℕ) : Prop := sorry

-- Theorem for the first set of numbers
theorem exist_expression_24_set1 :
  ∃ (e : Expr), usesAllNumbers e [7, 12, 9, 12] ∧ eval e = 24 := by sorry

-- Theorem for the second set of numbers
theorem exist_expression_24_set2 :
  ∃ (e : Expr), usesAllNumbers e [3, 9, 5, 9] ∧ eval e = 24 := by sorry

end NUMINAMATH_CALUDE_exist_expression_24_set1_exist_expression_24_set2_l1395_139542


namespace NUMINAMATH_CALUDE_min_value_expression_l1395_139557

theorem min_value_expression (x : ℝ) : 
  (8 - x) * (6 - x) * (8 + x) * (6 + x) ≥ -196 ∧ 
  ∃ y : ℝ, (8 - y) * (6 - y) * (8 + y) * (6 + y) = -196 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1395_139557


namespace NUMINAMATH_CALUDE_triangle_formation_l1395_139545

/-- Function to check if three numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating which set of numbers can form a triangle -/
theorem triangle_formation :
  can_form_triangle 13 12 20 ∧
  ¬ can_form_triangle 3 4 8 ∧
  ¬ can_form_triangle 8 7 15 ∧
  ¬ can_form_triangle 5 5 11 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l1395_139545


namespace NUMINAMATH_CALUDE_dorothy_age_problem_l1395_139500

theorem dorothy_age_problem :
  let dorothy_age : ℕ := 15
  let sister_age : ℕ := dorothy_age / 3
  let years_later : ℕ := 5
  (dorothy_age + years_later) = 2 * (sister_age + years_later) :=
by sorry

end NUMINAMATH_CALUDE_dorothy_age_problem_l1395_139500


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1395_139561

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1395_139561


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1395_139511

theorem tangent_line_to_circle (p : ℝ) : 
  (∀ x y : ℝ, x = -p/2 → x^2 + y^2 + 6*x + 8 = 0 → 
    ∃! y : ℝ, x^2 + y^2 + 6*x + 8 = 0) → 
  p = 4 ∨ p = 8 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1395_139511


namespace NUMINAMATH_CALUDE_polynomial_coefficient_product_l1395_139503

theorem polynomial_coefficient_product (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a * (a₁ + a₃) = 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_product_l1395_139503
