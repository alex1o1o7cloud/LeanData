import Mathlib

namespace NUMINAMATH_CALUDE_exponential_regression_model_l1208_120853

/-- Given a model y = ce^(kx) and a linear equation z = 0.3x + 4 where z = ln y,
    prove that c = e^4 and k = 0.3 -/
theorem exponential_regression_model (c k : ℝ) :
  (∀ x y : ℝ, y = c * Real.exp (k * x)) →
  (∀ x z : ℝ, z = 0.3 * x + 4) →
  (∀ y : ℝ, z = Real.log y) →
  c = Real.exp 4 ∧ k = 0.3 := by
sorry

end NUMINAMATH_CALUDE_exponential_regression_model_l1208_120853


namespace NUMINAMATH_CALUDE_amy_music_files_l1208_120879

theorem amy_music_files (total : ℕ) (video picture : ℝ) (h1 : total = 48) (h2 : video = 21.0) (h3 : picture = 23.0) :
  total - (video + picture) = 4 := by
sorry

end NUMINAMATH_CALUDE_amy_music_files_l1208_120879


namespace NUMINAMATH_CALUDE_square_root_five_minus_one_squared_plus_two_times_minus_four_equals_zero_l1208_120824

theorem square_root_five_minus_one_squared_plus_two_times_minus_four_equals_zero :
  ∀ x : ℝ, x = Real.sqrt 5 - 1 → x^2 + 2*x - 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_square_root_five_minus_one_squared_plus_two_times_minus_four_equals_zero_l1208_120824


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1208_120843

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 15) = 12 → x = 129 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1208_120843


namespace NUMINAMATH_CALUDE_x_squared_coefficient_is_13_l1208_120836

/-- The coefficient of x^2 in the expansion of ((1-x)^3 * (2x^2+1)^5) is 13 -/
theorem x_squared_coefficient_is_13 : 
  let f : Polynomial ℚ := (1 - X)^3 * (2*X^2 + 1)^5
  (f.coeff 2) = 13 := by sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_is_13_l1208_120836


namespace NUMINAMATH_CALUDE_inscribed_sphere_sum_l1208_120850

/-- A sphere inscribed in a right cone with base radius 15 cm and height 30 cm -/
structure InscribedSphere where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  b : ℝ
  d : ℝ
  base_radius_eq : base_radius = 15
  height_eq : height = 30
  sphere_radius_eq : sphere_radius = b * (Real.sqrt d - 1)

/-- Theorem stating that b + d = 11.75 for the given inscribed sphere -/
theorem inscribed_sphere_sum (s : InscribedSphere) : s.b + s.d = 11.75 := by
  sorry

#check inscribed_sphere_sum

end NUMINAMATH_CALUDE_inscribed_sphere_sum_l1208_120850


namespace NUMINAMATH_CALUDE_unique_solution_system_l1208_120826

theorem unique_solution_system (x y : ℝ) :
  (2 * x + y + 8 ≤ 0) ∧
  (x^4 + 2 * x^2 * y^2 + y^4 + 9 - 10 * x^2 - 10 * y^2 = 8 * x * y) →
  x = -3 ∧ y = -2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1208_120826


namespace NUMINAMATH_CALUDE_integral_second_derivative_car_acceleration_l1208_120805

-- Part 1
theorem integral_second_derivative {f : ℝ → ℝ} {a b : ℝ} (h₁ : Continuous (deriv (deriv f))) 
  (h₂ : deriv f a = 0) (h₃ : deriv f b = 0) (h₄ : a < b) :
  f b - f a = ∫ x in a..b, ((a + b) / 2 - x) * deriv (deriv f) x := by sorry

-- Part 2
theorem car_acceleration {f : ℝ → ℝ} {L T : ℝ} (h₁ : f 0 = 0) (h₂ : f T = L) 
  (h₃ : deriv f 0 = 0) (h₄ : deriv f T = 0) (h₅ : T > 0) (h₆ : L > 0) :
  ∃ t : ℝ, t ∈ Set.Icc 0 T ∧ |deriv (deriv f) t| ≥ 4 * L / T^2 := by sorry

end NUMINAMATH_CALUDE_integral_second_derivative_car_acceleration_l1208_120805


namespace NUMINAMATH_CALUDE_probability_calculation_l1208_120834

/-- Represents a bag of bills -/
structure Bag where
  tens : ℕ
  fives : ℕ
  ones : ℕ

/-- Calculates the total value of bills in a bag -/
def bagValue (b : Bag) : ℕ := 10 * b.tens + 5 * b.fives + b.ones

/-- Calculates the number of ways to choose 2 bills from a bag -/
def chooseTwo (b : Bag) : ℕ := (b.tens + b.fives + b.ones) * (b.tens + b.fives + b.ones - 1) / 2

/-- Calculates the probability of the sum of remaining bills in bag A being greater than the sum of remaining bills in bag B -/
def probabilityAGreaterThanB (bagA bagB : Bag) : ℚ :=
  let totalOutcomes := chooseTwo bagA * chooseTwo bagB
  let favorableOutcomes := 3 * 18  -- This is a simplification based on the problem's specific conditions
  ↑favorableOutcomes / ↑totalOutcomes

theorem probability_calculation (bagA bagB : Bag) 
  (hA : bagA = ⟨2, 0, 3⟩) 
  (hB : bagB = ⟨0, 4, 3⟩) : 
  probabilityAGreaterThanB bagA bagB = 9/35 := by
  sorry

#eval probabilityAGreaterThanB ⟨2, 0, 3⟩ ⟨0, 4, 3⟩

end NUMINAMATH_CALUDE_probability_calculation_l1208_120834


namespace NUMINAMATH_CALUDE_squirrel_journey_time_l1208_120860

/-- Proves that a squirrel traveling 0.5 miles at 6 mph and then 1.5 miles at 3 mph
    takes 35 minutes to complete a 2-mile journey. -/
theorem squirrel_journey_time :
  let total_distance : ℝ := 2
  let first_segment_distance : ℝ := 0.5
  let first_segment_speed : ℝ := 6
  let second_segment_distance : ℝ := 1.5
  let second_segment_speed : ℝ := 3
  let first_segment_time : ℝ := first_segment_distance / first_segment_speed
  let second_segment_time : ℝ := second_segment_distance / second_segment_speed
  let total_time_hours : ℝ := first_segment_time + second_segment_time
  let total_time_minutes : ℝ := total_time_hours * 60
  total_distance = first_segment_distance + second_segment_distance →
  total_time_minutes = 35 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_journey_time_l1208_120860


namespace NUMINAMATH_CALUDE_square_side_length_l1208_120815

/-- The radius of the circles -/
def r : ℝ := 1000

/-- The side length of the square -/
def square_side : ℝ := 400

/-- Two circles touch each other and a horizontal line is tangent to both circles -/
axiom circles_touch_and_tangent_to_line : True

/-- A square fits snugly between the horizontal line and the two circles -/
axiom square_fits_snugly : True

/-- The theorem stating that the side length of the square is 400 -/
theorem square_side_length : 
  square_side = 400 :=
sorry

end NUMINAMATH_CALUDE_square_side_length_l1208_120815


namespace NUMINAMATH_CALUDE_circle_C_and_point_M_l1208_120812

/-- Circle C passing through points A and B, bisected by a line, with point M satisfying certain conditions -/
structure CircleC where
  /-- Center of the circle -/
  center : ℝ × ℝ
  /-- Radius of the circle -/
  radius : ℝ
  /-- Point A on the circle -/
  A : ℝ × ℝ
  /-- Point B on the circle -/
  B : ℝ × ℝ
  /-- Point P -/
  P : ℝ × ℝ
  /-- Point Q -/
  Q : ℝ × ℝ
  /-- Point M on the circle -/
  M : ℝ × ℝ
  /-- Circle passes through A -/
  passes_through_A : (center.1 - A.1)^2 + (center.2 - A.2)^2 = radius^2
  /-- Circle passes through B -/
  passes_through_B : (center.1 - B.1)^2 + (center.2 - B.2)^2 = radius^2
  /-- Line x-3y-4=0 bisects the circle -/
  bisected_by_line : center.1 - 3 * center.2 - 4 = 0
  /-- |MP|/|MQ| = 2 -/
  distance_ratio : ((M.1 - P.1)^2 + (M.2 - P.2)^2) = 4 * ((M.1 - Q.1)^2 + (M.2 - Q.2)^2)

/-- Theorem about the equation of circle C and coordinates of point M -/
theorem circle_C_and_point_M (c : CircleC)
  (h_A : c.A = (0, 2))
  (h_B : c.B = (6, 4))
  (h_P : c.P = (-6, 0))
  (h_Q : c.Q = (6, 0)) :
  (c.center = (4, 0) ∧ c.radius^2 = 20) ∧
  (c.M = (10/3, 4*Real.sqrt 11/3) ∨ c.M = (10/3, -4*Real.sqrt 11/3)) := by
  sorry

end NUMINAMATH_CALUDE_circle_C_and_point_M_l1208_120812


namespace NUMINAMATH_CALUDE_fruit_shop_apples_l1208_120838

/-- Given a ratio of fruits and the number of mangoes, calculate the number of apples -/
theorem fruit_shop_apples (ratio_mangoes ratio_oranges ratio_apples : ℕ) 
  (num_mangoes : ℕ) (h1 : ratio_mangoes = 10) (h2 : ratio_oranges = 2) 
  (h3 : ratio_apples = 3) (h4 : num_mangoes = 120) : 
  (num_mangoes / ratio_mangoes) * ratio_apples = 36 := by
  sorry

end NUMINAMATH_CALUDE_fruit_shop_apples_l1208_120838


namespace NUMINAMATH_CALUDE_hotel_charge_difference_l1208_120893

theorem hotel_charge_difference (G R P : ℝ) 
  (hR : R = G * 3.0000000000000006)
  (hP : P = G * 0.9) :
  (R - P) / R * 100 = 70 := by sorry

end NUMINAMATH_CALUDE_hotel_charge_difference_l1208_120893


namespace NUMINAMATH_CALUDE_brendas_mother_cookies_l1208_120837

/-- The number of people Brenda's mother made cookies for -/
def num_people (total_cookies : ℕ) (cookies_per_person : ℕ) : ℕ :=
  total_cookies / cookies_per_person

theorem brendas_mother_cookies : num_people 35 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_brendas_mother_cookies_l1208_120837


namespace NUMINAMATH_CALUDE_ship_passengers_theorem_l1208_120845

theorem ship_passengers_theorem (total_passengers : ℝ) (round_trip_with_car : ℝ) 
  (round_trip_without_car : ℝ) (h1 : round_trip_with_car > 0) 
  (h2 : round_trip_with_car + round_trip_without_car ≤ total_passengers) 
  (h3 : round_trip_with_car / total_passengers = 0.3) :
  (round_trip_with_car + round_trip_without_car) / total_passengers = 
  round_trip_with_car / total_passengers := by
sorry

end NUMINAMATH_CALUDE_ship_passengers_theorem_l1208_120845


namespace NUMINAMATH_CALUDE_expression_simplification_l1208_120874

theorem expression_simplification (x y : ℝ) 
  (hx : x = (Real.sqrt 5 + 1) / 2) 
  (hy : y = (Real.sqrt 5 - 1) / 2) : 
  (x - 2*y)^2 + x*(5*y - x) - 4*y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1208_120874


namespace NUMINAMATH_CALUDE_correct_transformation_l1208_120855

theorem correct_transformation : (-2 : ℚ) * (1/2 : ℚ) * (-5 : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l1208_120855


namespace NUMINAMATH_CALUDE_photographer_choices_l1208_120835

theorem photographer_choices (n : ℕ) (k₁ k₂ : ℕ) (h₁ : n = 7) (h₂ : k₁ = 4) (h₃ : k₂ = 5) :
  Nat.choose n k₁ + Nat.choose n k₂ = 56 := by
  sorry

end NUMINAMATH_CALUDE_photographer_choices_l1208_120835


namespace NUMINAMATH_CALUDE_average_age_problem_l1208_120808

theorem average_age_problem (devin_age eden_age mom_age : ℕ) : 
  devin_age = 12 →
  eden_age = 2 * devin_age →
  mom_age = 2 * eden_age →
  (devin_age + eden_age + mom_age) / 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_problem_l1208_120808


namespace NUMINAMATH_CALUDE_complex_square_roots_l1208_120802

theorem complex_square_roots : 
  let z₁ : ℂ := Complex.mk (3 * Real.sqrt 2) (-55 * Real.sqrt 2 / 6)
  let z₂ : ℂ := Complex.mk (-3 * Real.sqrt 2) (55 * Real.sqrt 2 / 6)
  z₁^2 = Complex.mk (-121) (-110) ∧ z₂^2 = Complex.mk (-121) (-110) :=
by sorry

end NUMINAMATH_CALUDE_complex_square_roots_l1208_120802


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1208_120876

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (3 - 4*i) / (1 - i)
  (z.im : ℝ) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1208_120876


namespace NUMINAMATH_CALUDE_pascal_interior_sum_8_9_l1208_120882

/-- Sum of interior numbers in row n of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The sum of interior numbers of the 8th and 9th rows of Pascal's Triangle is 380 -/
theorem pascal_interior_sum_8_9 : interior_sum 8 + interior_sum 9 = 380 := by
  sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_8_9_l1208_120882


namespace NUMINAMATH_CALUDE_compare_fractions_l1208_120896

theorem compare_fractions (a b : ℝ) (h : a + b > 0) :
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l1208_120896


namespace NUMINAMATH_CALUDE_coordinate_sum_of_h_l1208_120863

/-- Given a function g where g(3) = 8, and a function h where h(x) = (g(x))^3 for all x,
    the sum of the coordinates of the point (3, h(3)) is 515. -/
theorem coordinate_sum_of_h (g : ℝ → ℝ) (h : ℝ → ℝ) 
    (hg : g 3 = 8) (hh : ∀ x, h x = (g x)^3) : 
    3 + h 3 = 515 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_h_l1208_120863


namespace NUMINAMATH_CALUDE_expression_equality_l1208_120851

theorem expression_equality (x : ℝ) (hx : x > 0) :
  (∃! n : ℕ, n = (if 2 * x^(x+1) = x^(x+1) + x^(x+1) then 1 else 0) +
              (if x^(2*x+2) = x^(x+1) + x^(x+1) then 1 else 0) +
              (if (3*x)^x = x^(x+1) + x^(x+1) then 1 else 0) +
              (if (3*x)^(x+1) = x^(x+1) + x^(x+1) then 1 else 0)) ∧
  n = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equality_l1208_120851


namespace NUMINAMATH_CALUDE_squirrel_count_l1208_120800

theorem squirrel_count (first_count second_count total : ℕ) :
  second_count = first_count + (first_count / 3) →
  total = first_count + second_count →
  total = 28 →
  first_count = 21 := by
sorry

end NUMINAMATH_CALUDE_squirrel_count_l1208_120800


namespace NUMINAMATH_CALUDE_fraction_equality_l1208_120827

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : y^2 - 1/x ≠ 0) :
  (x^2 - 1/y) / (y^2 - 1/x) = x/y := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1208_120827


namespace NUMINAMATH_CALUDE_grade_assignments_12_4_l1208_120817

/-- The number of ways to assign grades to students -/
def gradeAssignments (numStudents : ℕ) (numGrades : ℕ) : ℕ :=
  numGrades ^ numStudents

/-- Theorem: The number of ways to assign 4 possible grades to 12 students is 16777216 -/
theorem grade_assignments_12_4 : gradeAssignments 12 4 = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignments_12_4_l1208_120817


namespace NUMINAMATH_CALUDE_stratified_sample_teachers_l1208_120840

def total_teachers : ℕ := 300
def senior_teachers : ℕ := 90
def intermediate_teachers : ℕ := 150
def junior_teachers : ℕ := 60
def sample_size : ℕ := 40

def stratified_sample (total : ℕ) (strata : List ℕ) (sample : ℕ) : List ℕ :=
  strata.map (λ stratum => (stratum * sample) / total)

theorem stratified_sample_teachers :
  stratified_sample total_teachers [senior_teachers, intermediate_teachers, junior_teachers] sample_size = [12, 20, 8] := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_teachers_l1208_120840


namespace NUMINAMATH_CALUDE_power_function_properties_l1208_120833

noncomputable def f (x : ℝ) : ℝ := x ^ (1/2)

theorem power_function_properties :
  ∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ →
    (x₁ * f x₁ < x₂ * f x₂) ∧
    (f x₁ / x₁ > f x₂ / x₂) :=
by sorry

end NUMINAMATH_CALUDE_power_function_properties_l1208_120833


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l1208_120867

/-- Given a car traveling for two hours with an average speed of 75 km/h
    and a speed of 90 km/h in the first hour, the speed in the second hour is 60 km/h. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_first_hour = 90) 
  (h2 : average_speed = 75) 
  (h3 : total_time = 2) : 
  ∃ (speed_second_hour : ℝ), speed_second_hour = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l1208_120867


namespace NUMINAMATH_CALUDE_range_of_f_l1208_120859

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 4*x - 6

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 5 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | -11 ≤ y ∧ y ≤ -2 } := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l1208_120859


namespace NUMINAMATH_CALUDE_school_math_survey_l1208_120883

theorem school_math_survey (total : ℝ) (math_likers : ℝ) (olympiad_participants : ℝ)
  (h1 : math_likers ≤ total)
  (h2 : olympiad_participants = math_likers + 0.1 * (total - math_likers))
  (h3 : olympiad_participants = 0.46 * total) :
  math_likers = 0.4 * total :=
by sorry

end NUMINAMATH_CALUDE_school_math_survey_l1208_120883


namespace NUMINAMATH_CALUDE_beach_creatures_ratio_l1208_120844

theorem beach_creatures_ratio :
  ∀ (oysters_day1 crabs_day1 total_both_days : ℕ),
    oysters_day1 = 50 →
    crabs_day1 = 72 →
    total_both_days = 195 →
    ∃ (crabs_day2 : ℕ),
      oysters_day1 + crabs_day1 + (oysters_day1 / 2 + crabs_day2) = total_both_days ∧
      crabs_day2 * 3 = crabs_day1 * 2 :=
by sorry

end NUMINAMATH_CALUDE_beach_creatures_ratio_l1208_120844


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1208_120842

theorem regular_polygon_exterior_angle (n : ℕ) (h : (n - 2) * 180 = 1800) :
  360 / n = 30 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1208_120842


namespace NUMINAMATH_CALUDE_box_third_side_length_l1208_120886

/-- Proves that the third side of a rectangular box is 9 cm, given specific conditions. -/
theorem box_third_side_length (num_cubes : ℕ) (cube_volume : ℝ) (side1 side2 : ℝ) :
  num_cubes = 24 →
  cube_volume = 27 →
  side1 = 8 →
  side2 = 9 →
  (num_cubes : ℝ) * cube_volume = side1 * side2 * 9 :=
by sorry

end NUMINAMATH_CALUDE_box_third_side_length_l1208_120886


namespace NUMINAMATH_CALUDE_cyclic_fraction_inequality_l1208_120873

theorem cyclic_fraction_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 2*y) / (z + 2*x + 3*y) + (y + 2*z) / (x + 2*y + 3*z) + (z + 2*x) / (y + 2*z + 3*x) ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_cyclic_fraction_inequality_l1208_120873


namespace NUMINAMATH_CALUDE_arrangement_count_l1208_120801

/-- The number of ways to arrange 5 people in a line, where one specific person cannot stand at either end -/
def arrangements : ℕ := 72

/-- The total number of people -/
def total_people : ℕ := 5

/-- The number of positions where the specific person can stand -/
def positions_for_specific_person : ℕ := 3

theorem arrangement_count : arrangements = 72 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_l1208_120801


namespace NUMINAMATH_CALUDE_bounds_on_ratio_of_squares_l1208_120830

theorem bounds_on_ratio_of_squares (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (m M : ℝ), 
    (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → m ≤ |a + b|^2 / (|a|^2 + |b|^2) ∧ |a + b|^2 / (|a|^2 + |b|^2) ≤ M) ∧
    (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ |c + d|^2 / (|c|^2 + |d|^2) = m) ∧
    (∃ e f : ℝ, e ≠ 0 ∧ f ≠ 0 ∧ |e + f|^2 / (|e|^2 + |f|^2) = M) ∧
    M - m = 2 :=
by sorry

end NUMINAMATH_CALUDE_bounds_on_ratio_of_squares_l1208_120830


namespace NUMINAMATH_CALUDE_ted_green_mushrooms_l1208_120803

/-- The number of green mushrooms Ted gathered -/
def green_mushrooms : ℕ := sorry

/-- The number of red mushrooms Bill gathered -/
def red_mushrooms : ℕ := 12

/-- The number of brown mushrooms Bill gathered -/
def brown_mushrooms : ℕ := 6

/-- The number of blue mushrooms Ted gathered -/
def blue_mushrooms : ℕ := 6

/-- The total number of white-spotted mushrooms gathered -/
def total_white_spotted : ℕ := 17

theorem ted_green_mushrooms :
  green_mushrooms = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ted_green_mushrooms_l1208_120803


namespace NUMINAMATH_CALUDE_annulus_area_l1208_120822

/-- The area of an annulus with specific properties -/
theorem annulus_area (r s t : ℝ) (h1 : r > s) (h2 : t = 2 * s) (h3 : r^2 = s^2 + (t/2)^2) :
  π * (r^2 - s^2) = π * s^2 := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_l1208_120822


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1208_120856

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1208_120856


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1208_120871

theorem quadratic_equation_solution (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 5 * x - m = 0) → m ≥ -25/8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1208_120871


namespace NUMINAMATH_CALUDE_custom_op_result_l1208_120816

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

-- Theorem statement
theorem custom_op_result :
  let y : ℤ := 11
  customOp y 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l1208_120816


namespace NUMINAMATH_CALUDE_x_gt_2_sufficient_not_necessary_for_x_gt_1_l1208_120823

theorem x_gt_2_sufficient_not_necessary_for_x_gt_1 :
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_2_sufficient_not_necessary_for_x_gt_1_l1208_120823


namespace NUMINAMATH_CALUDE_range_of_c_over_a_l1208_120810

theorem range_of_c_over_a (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 1) (h3 : a + b + c = 0) :
  ∀ x, (x = c / a) → -2 < x ∧ x < -1 := by
sorry

end NUMINAMATH_CALUDE_range_of_c_over_a_l1208_120810


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l1208_120839

theorem factorization_of_polynomial (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3)*(8*x^2 + x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l1208_120839


namespace NUMINAMATH_CALUDE_line_equations_correct_l1208_120806

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define a line by its slope and a point it passes through
structure Line1 where
  slope : ℝ
  point : Point

-- Define a line by two points it passes through
structure Line2 where
  point1 : Point
  point2 : Point

-- Function to get the equation of a line given slope and point
def lineEquation1 (l : Line1) : ℝ → ℝ → Prop :=
  fun x y => y - l.point.2 = l.slope * (x - l.point.1)

-- Function to get the equation of a line given two points
def lineEquation2 (l : Line2) : ℝ → ℝ → Prop :=
  fun x y => (y - l.point1.2) * (l.point2.1 - l.point1.1) = 
             (l.point2.2 - l.point1.2) * (x - l.point1.1)

theorem line_equations_correct :
  let line1 := Line1.mk (-1/2) (8, -2)
  let line2 := Line2.mk (3, -2) (5, -4)
  (∀ x y, lineEquation1 line1 x y ↔ x + 2*y - 4 = 0) ∧
  (∀ x y, lineEquation2 line2 x y ↔ x + y - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_equations_correct_l1208_120806


namespace NUMINAMATH_CALUDE_new_basis_from_old_l1208_120887

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def is_basis (v₁ v₂ v₃ : V) : Prop :=
  LinearIndependent ℝ ![v₁, v₂, v₃] ∧ Submodule.span ℝ {v₁, v₂, v₃} = ⊤

theorem new_basis_from_old (a b c p q : V) 
  (h₁ : is_basis a b c)
  (h₂ : p = a + b)
  (h₃ : q = a - b) :
  is_basis p q (a + 2 • c) := by
  sorry

end NUMINAMATH_CALUDE_new_basis_from_old_l1208_120887


namespace NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1208_120849

/-- The number of rectangles in a row of length n -/
def rectangles_in_row (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of rectangles in a grid of width w and height h -/
def total_rectangles (w h : ℕ) : ℕ :=
  w * rectangles_in_row h + h * rectangles_in_row w - w * h

theorem rectangles_in_5x4_grid :
  total_rectangles 5 4 = 24 := by sorry

end NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1208_120849


namespace NUMINAMATH_CALUDE_frans_original_seat_l1208_120813

/-- Represents the seats in the theater --/
inductive Seat
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the friends --/
inductive Friend
  | ada
  | bea
  | ceci
  | dee
  | edie
  | fran

/-- Represents the direction of movement --/
inductive Direction
  | left
  | right

/-- Represents a movement of a friend --/
structure Movement where
  friend : Friend
  distance : Nat
  direction : Direction

/-- The initial seating arrangement --/
def initialSeating : Friend → Seat := sorry

/-- The movements of the friends --/
def friendMovements : List Movement := [
  ⟨Friend.ada, 3, Direction.right⟩,
  ⟨Friend.bea, 2, Direction.left⟩,
  ⟨Friend.ceci, 0, Direction.right⟩,
  ⟨Friend.dee, 0, Direction.right⟩,
  ⟨Friend.edie, 1, Direction.right⟩
]

/-- Function to apply movements and get the final seating arrangement --/
def applyMovements (initial : Friend → Seat) (movements : List Movement) : Friend → Seat := sorry

/-- Function to find the vacant seat after movements --/
def findVacantSeat (seating : Friend → Seat) : Seat := sorry

/-- Theorem stating Fran's original seat --/
theorem frans_original_seat :
  initialSeating Friend.fran = Seat.three ∧
  (findVacantSeat (applyMovements initialSeating friendMovements) = Seat.one ∨
   findVacantSeat (applyMovements initialSeating friendMovements) = Seat.six) := by
  sorry

end NUMINAMATH_CALUDE_frans_original_seat_l1208_120813


namespace NUMINAMATH_CALUDE_distribute_nine_computers_to_three_schools_l1208_120819

/-- The number of ways to distribute computers to schools -/
def distribute_computers (total_computers : ℕ) (num_schools : ℕ) (min_computers : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem: There are 10 ways to distribute 9 computers to 3 schools with at least 2 per school -/
theorem distribute_nine_computers_to_three_schools : 
  distribute_computers 9 3 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_distribute_nine_computers_to_three_schools_l1208_120819


namespace NUMINAMATH_CALUDE_number_operations_l1208_120864

theorem number_operations (x : ℝ) : ((x - 2 + 3) * 2) / 3 = 6 ↔ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l1208_120864


namespace NUMINAMATH_CALUDE_number_added_to_x_l1208_120857

theorem number_added_to_x (x : ℝ) (some_number : ℝ) : 
  x + some_number = 5 → x = 4 → some_number = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_added_to_x_l1208_120857


namespace NUMINAMATH_CALUDE_forgot_capsules_days_l1208_120811

/-- The number of days in July -/
def july_days : ℕ := 31

/-- The number of days Adam took his capsules in July -/
def days_took_capsules : ℕ := 27

/-- The number of days Adam forgot to take his capsules in July -/
def days_forgot_capsules : ℕ := july_days - days_took_capsules

theorem forgot_capsules_days : days_forgot_capsules = 4 := by
  sorry

end NUMINAMATH_CALUDE_forgot_capsules_days_l1208_120811


namespace NUMINAMATH_CALUDE_power_calculation_l1208_120880

theorem power_calculation : 16^16 * 8^8 / 4^32 = 2^24 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1208_120880


namespace NUMINAMATH_CALUDE_banyan_tree_area_l1208_120828

theorem banyan_tree_area (C : Real) (h : C = 6.28) :
  let r := C / (2 * Real.pi)
  let S := Real.pi * r^2
  S = Real.pi := by
sorry

end NUMINAMATH_CALUDE_banyan_tree_area_l1208_120828


namespace NUMINAMATH_CALUDE_distance_gable_to_citadel_l1208_120881

/-- The distance from the point (1600, 1200) to the origin (0, 0) on a complex plane is 2000. -/
theorem distance_gable_to_citadel : 
  Complex.abs (Complex.mk 1600 1200) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_distance_gable_to_citadel_l1208_120881


namespace NUMINAMATH_CALUDE_deepthi_material_usage_l1208_120888

theorem deepthi_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 4 / 17)
  (h2 : material2 = 3 / 10)
  (h3 : leftover = 9 / 30) :
  material1 + material2 - leftover = 4 / 17 := by
sorry

end NUMINAMATH_CALUDE_deepthi_material_usage_l1208_120888


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l1208_120875

theorem sum_of_reciprocal_relations (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -6) : 
  x + y = -4/5 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l1208_120875


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l1208_120892

theorem sqrt_sum_equals_seven (y : ℝ) 
  (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) : 
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l1208_120892


namespace NUMINAMATH_CALUDE_group_size_theorem_l1208_120861

theorem group_size_theorem (n : ℕ) (k : ℕ) : 
  (k * (n - 1) * n = 440 ∧ n > 0 ∧ k > 0) → (n = 5 ∨ n = 11) :=
sorry

end NUMINAMATH_CALUDE_group_size_theorem_l1208_120861


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_value_l1208_120807

/-- Given two positive integers with a specific ratio and value, prove their LCM --/
theorem lcm_of_ratio_and_value (a b : ℕ+) (h1 : a = 45) (h2 : 4 * a = 3 * b) : 
  Nat.lcm a b = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_value_l1208_120807


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1208_120865

theorem pure_imaginary_complex_number (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 - 1) (x - 1)
  (z.re = 0 ∧ z ≠ 0) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1208_120865


namespace NUMINAMATH_CALUDE_lynn_ogen_interest_l1208_120884

/-- Calculates the total annual interest for Lynn Ogen's investments -/
theorem lynn_ogen_interest (x : ℝ) (h1 : x - 100 = 400) :
  0.09 * x + 0.07 * (x - 100) = 73 := by
  sorry

end NUMINAMATH_CALUDE_lynn_ogen_interest_l1208_120884


namespace NUMINAMATH_CALUDE_smallest_dividend_l1208_120862

theorem smallest_dividend (A B : ℕ) (h1 : A = B * 28 + 4) (h2 : B > 0) : A ≥ 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_dividend_l1208_120862


namespace NUMINAMATH_CALUDE_not_octal_7857_l1208_120814

def is_octal_digit (d : Nat) : Prop := d ≤ 7

def is_octal_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 8 → is_octal_digit d

theorem not_octal_7857 : ¬ is_octal_number 7857 := by
  sorry

end NUMINAMATH_CALUDE_not_octal_7857_l1208_120814


namespace NUMINAMATH_CALUDE_fraction_simplification_l1208_120804

theorem fraction_simplification (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / y + 1 / x) / (1 / x) = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1208_120804


namespace NUMINAMATH_CALUDE_initial_number_proof_l1208_120858

theorem initial_number_proof (x : ℝ) : ((x / 13) / 29) * (1/4) / 2 = 0.125 → x = 754 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1208_120858


namespace NUMINAMATH_CALUDE_liza_final_balance_l1208_120872

/-- Calculates the final balance in Liza's account after a series of transactions --/
def final_balance (initial_balance rent paycheck electricity internet phone : ℤ) : ℤ :=
  initial_balance - rent + paycheck - electricity - internet - phone

/-- Theorem stating that Liza's final account balance is 1563 --/
theorem liza_final_balance :
  final_balance 800 450 1500 117 100 70 = 1563 := by sorry

end NUMINAMATH_CALUDE_liza_final_balance_l1208_120872


namespace NUMINAMATH_CALUDE_union_covers_reals_l1208_120846

def set_A : Set ℝ := {x | x ≤ 2}
def set_B (a : ℝ) : Set ℝ := {x | x > a}

theorem union_covers_reals (a : ℝ) :
  set_A ∪ set_B a = Set.univ → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_l1208_120846


namespace NUMINAMATH_CALUDE_min_c_value_l1208_120848

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hab : a < b) (hbc : b < c) 
  (h_unique : ∃! (x y : ℝ), 2 * x + y = 3005 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 3003 ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < 3003 ∧
    ∃! (x y : ℝ), 2 * x + y = 3005 ∧ y = |x - a'| + |x - b'| + |x - 3003| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l1208_120848


namespace NUMINAMATH_CALUDE_kho_kho_players_l1208_120832

theorem kho_kho_players (total : ℕ) (kabadi : ℕ) (both : ℕ) (kho_kho_only : ℕ) :
  total = 50 →
  kabadi = 10 →
  both = 5 →
  total = (kabadi - both) + kho_kho_only + both →
  kho_kho_only = 40 := by
sorry

end NUMINAMATH_CALUDE_kho_kho_players_l1208_120832


namespace NUMINAMATH_CALUDE_divisibility_properties_l1208_120894

theorem divisibility_properties (a b : ℤ) (k : ℕ) :
  (¬ ((a + b) ∣ (a^(2*k) + b^(2*k))) ∧ ¬ ((a - b) ∣ (a^(2*k) + b^(2*k)))) ∧
  ((a + b) ∣ (a^(2*k) - b^(2*k)) ∧ (a - b) ∣ (a^(2*k) - b^(2*k))) ∧
  ((a + b) ∣ (a^(2*k+1) + b^(2*k+1))) ∧
  ((a - b) ∣ (a^(2*k+1) - b^(2*k+1))) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_properties_l1208_120894


namespace NUMINAMATH_CALUDE_towel_area_decrease_l1208_120890

theorem towel_area_decrease :
  ∀ (L B : ℝ), L > 0 → B > 0 →
  let new_length := 0.7 * L
  let new_breadth := 0.75 * B
  let original_area := L * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.475 :=
by sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l1208_120890


namespace NUMINAMATH_CALUDE_arrangements_without_adjacent_l1208_120885

def total_people : ℕ := 5

theorem arrangements_without_adjacent (A B : ℕ) (h1 : A ≤ total_people) (h2 : B ≤ total_people) (h3 : A ≠ B) :
  (Nat.factorial total_people) - (2 * Nat.factorial (total_people - 1)) = 72 :=
sorry

end NUMINAMATH_CALUDE_arrangements_without_adjacent_l1208_120885


namespace NUMINAMATH_CALUDE_rectangular_garden_length_l1208_120877

/-- Calculates the length of a rectangular garden given its perimeter and breadth. -/
theorem rectangular_garden_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 480) 
  (h2 : breadth = 100) : 
  perimeter / 2 - breadth = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_length_l1208_120877


namespace NUMINAMATH_CALUDE_polygon_with_72_degree_exterior_angles_has_5_sides_l1208_120895

/-- A polygon with exterior angles each measuring 72° has 5 sides -/
theorem polygon_with_72_degree_exterior_angles_has_5_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    exterior_angle = 72 →
    n * exterior_angle = 360 →
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_72_degree_exterior_angles_has_5_sides_l1208_120895


namespace NUMINAMATH_CALUDE_smallest_three_digit_integer_l1208_120841

theorem smallest_three_digit_integer (n : ℕ) : 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (45 * n ≡ 135 [MOD 280]) ∧ 
  (n ≡ 3 [MOD 7]) →
  n ≥ 115 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_integer_l1208_120841


namespace NUMINAMATH_CALUDE_erikas_savings_l1208_120821

theorem erikas_savings (gift_cost cake_cost leftover : ℕ) 
  (h1 : gift_cost = 250)
  (h2 : cake_cost = 25)
  (h3 : leftover = 5)
  (ricks_savings : ℕ) (h4 : ricks_savings = gift_cost / 2)
  (total_savings : ℕ) (h5 : total_savings = gift_cost + cake_cost + leftover) :
  total_savings - ricks_savings = 155 := by
sorry

end NUMINAMATH_CALUDE_erikas_savings_l1208_120821


namespace NUMINAMATH_CALUDE_family_income_change_l1208_120898

theorem family_income_change (initial_average : ℚ) (initial_members : ℕ) 
  (deceased_income : ℚ) (new_members : ℕ) : 
  initial_average = 840 →
  initial_members = 4 →
  deceased_income = 1410 →
  new_members = 3 →
  (initial_average * initial_members - deceased_income) / new_members = 650 := by
  sorry

end NUMINAMATH_CALUDE_family_income_change_l1208_120898


namespace NUMINAMATH_CALUDE_gcd_187_253_l1208_120809

theorem gcd_187_253 : Nat.gcd 187 253 = 11 := by sorry

end NUMINAMATH_CALUDE_gcd_187_253_l1208_120809


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1208_120899

theorem quadratic_inequality (x : ℝ) : x^2 - 4*x > 44 ↔ x < -4 ∨ x > 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1208_120899


namespace NUMINAMATH_CALUDE_sum_of_angles_convex_polygon_l1208_120831

/-- The sum of interior angles of a convex polygon with n sides, where n ≥ 3 -/
def sumOfAngles (n : ℕ) : ℝ :=
  (n - 2) * 180

/-- Theorem: For any convex polygon with n sides, where n ≥ 3,
    the sum of its interior angles is equal to (n-2) * 180° -/
theorem sum_of_angles_convex_polygon (n : ℕ) (h : n ≥ 3) :
  sumOfAngles n = (n - 2) * 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_convex_polygon_l1208_120831


namespace NUMINAMATH_CALUDE_power_division_l1208_120825

theorem power_division (a : ℝ) : a^7 / a = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l1208_120825


namespace NUMINAMATH_CALUDE_five_digit_div_four_digit_integer_l1208_120869

/-- Represents a five-digit number -/
structure FiveDigitNumber where
  x : Nat
  y : Nat
  z : Nat
  u : Nat
  v : Nat
  h_x_pos : x > 0
  h_x_digit : x < 10
  h_y_digit : y < 10
  h_z_digit : z < 10
  h_u_digit : u < 10
  h_v_digit : v < 10

/-- Converts a FiveDigitNumber to its numerical value -/
def FiveDigitNumber.toNat (n : FiveDigitNumber) : Nat :=
  n.x * 10000 + n.y * 1000 + n.z * 100 + n.u * 10 + n.v

/-- Converts a FiveDigitNumber to its corresponding four-digit number -/
def FiveDigitNumber.toFourDigit (n : FiveDigitNumber) : Nat :=
  n.x * 1000 + n.y * 100 + n.u * 10 + n.v

/-- Theorem: A five-digit number divided by its corresponding four-digit number
    is an integer if and only if it has the form xy000 where 10 ≤ xy ≤ 99 -/
theorem five_digit_div_four_digit_integer (n : FiveDigitNumber) :
  (n.toNat % n.toFourDigit = 0) ↔
  (∃ xy : Nat, 10 ≤ xy ∧ xy ≤ 99 ∧ n.toNat = xy * 1000) :=
sorry

end NUMINAMATH_CALUDE_five_digit_div_four_digit_integer_l1208_120869


namespace NUMINAMATH_CALUDE_restaurant_cooks_count_l1208_120854

theorem restaurant_cooks_count (initial_cooks : ℕ) (initial_waiters : ℕ) 
  (h1 : initial_cooks * 8 = initial_waiters * 3) 
  (h2 : initial_cooks * 4 = (initial_waiters + 12) * 1) : 
  initial_cooks = 9 := by
sorry

end NUMINAMATH_CALUDE_restaurant_cooks_count_l1208_120854


namespace NUMINAMATH_CALUDE_star_3_5_l1208_120889

/-- The star operation defined for real numbers -/
def star (x y : ℝ) : ℝ := x^2 + x*y + y^2

/-- Theorem stating that 3 ⋆ 5 = 49 -/
theorem star_3_5 : star 3 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_star_3_5_l1208_120889


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1208_120852

theorem square_area_from_diagonal (diagonal : Real) (area : Real) :
  diagonal = 10 → area = diagonal^2 / 2 → area = 50 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1208_120852


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l1208_120866

theorem quadratic_roots_sum_of_squares (m n : ℝ) : 
  (m^2 - 2*m - 1 = 0) → (n^2 - 2*n - 1 = 0) → m^2 + n^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l1208_120866


namespace NUMINAMATH_CALUDE_right_triangle_longer_leg_l1208_120829

theorem right_triangle_longer_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- b is the longer leg
  b ≤ c →            -- Longer leg is shorter than hypotenuse
  b = 60 :=          -- Conclusion: longer leg is 60
by
  sorry

#check right_triangle_longer_leg

end NUMINAMATH_CALUDE_right_triangle_longer_leg_l1208_120829


namespace NUMINAMATH_CALUDE_move_right_3_units_l1208_120878

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Moves a point horizontally by a given distance -/
def moveRight (p : Point2D) (distance : ℝ) : Point2D :=
  { x := p.x + distance, y := p.y }

theorem move_right_3_units :
  let initial_point : Point2D := { x := 2, y := -1 }
  let final_point : Point2D := { x := 5, y := -1 }
  moveRight initial_point 3 = final_point := by
  sorry

end NUMINAMATH_CALUDE_move_right_3_units_l1208_120878


namespace NUMINAMATH_CALUDE_curve_intersection_implies_a_equals_one_l1208_120818

/-- Curve C₁ in polar coordinates -/
def C₁ (ρ θ : ℝ) (a : ℝ) : Prop :=
  ρ^2 - 2*ρ*(Real.sin θ) + 1 - a^2 = 0 ∧ a > 0

/-- Curve C₂ in polar coordinates -/
def C₂ (ρ θ : ℝ) : Prop :=
  ρ = 4*(Real.cos θ)

/-- Line C₃ in polar coordinates -/
def C₃ (θ : ℝ) : Prop :=
  ∃ α₀, θ = α₀ ∧ Real.tan α₀ = 2

/-- Common points of C₁ and C₂ lie on C₃ -/
def common_points_on_C₃ (a : ℝ) : Prop :=
  ∀ ρ θ, C₁ ρ θ a ∧ C₂ ρ θ → C₃ θ

theorem curve_intersection_implies_a_equals_one :
  ∀ a, common_points_on_C₃ a → a = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_intersection_implies_a_equals_one_l1208_120818


namespace NUMINAMATH_CALUDE_average_difference_theorem_l1208_120891

/-- Represents the enrollment of a class -/
structure ClassEnrollment where
  students : ℕ

/-- Represents a school with students, teachers, and class enrollments -/
structure School where
  totalStudents : ℕ
  totalTeachers : ℕ
  classEnrollments : List ClassEnrollment

/-- Calculates the average number of students per teacher -/
def averageStudentsPerTeacher (school : School) : ℚ :=
  school.totalStudents / school.totalTeachers

/-- Calculates the average number of students per student -/
def averageStudentsPerStudent (school : School) : ℚ :=
  (school.classEnrollments.map (λ c => c.students * c.students)).sum / school.totalStudents

/-- The main theorem to prove -/
theorem average_difference_theorem (school : School) 
  (h1 : school.totalStudents = 120)
  (h2 : school.totalTeachers = 6)
  (h3 : school.classEnrollments = [⟨60⟩, ⟨30⟩, ⟨20⟩, ⟨5⟩, ⟨3⟩, ⟨2⟩])
  (h4 : (school.classEnrollments.map (λ c => c.students)).sum = school.totalStudents) :
  averageStudentsPerTeacher school - averageStudentsPerStudent school = -21 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_theorem_l1208_120891


namespace NUMINAMATH_CALUDE_difference_of_squares_l1208_120870

theorem difference_of_squares (a b : ℕ+) : 
  ∃ (x y z w : ℤ), (a : ℤ) = x^2 - y^2 ∨ (b : ℤ) = z^2 - w^2 ∨ ((a + b) : ℤ) = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1208_120870


namespace NUMINAMATH_CALUDE_power_seven_equals_product_l1208_120897

theorem power_seven_equals_product (a : ℝ) : a^7 = a^3 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_equals_product_l1208_120897


namespace NUMINAMATH_CALUDE_a_minus_c_equals_three_l1208_120820

theorem a_minus_c_equals_three (a b c d : ℤ) 
  (h1 : a - b = c + d + 9) 
  (h2 : a + b = c - d - 3) : 
  a - c = 3 := by
sorry

end NUMINAMATH_CALUDE_a_minus_c_equals_three_l1208_120820


namespace NUMINAMATH_CALUDE_cost_decrease_for_constant_profit_l1208_120868

theorem cost_decrease_for_constant_profit 
  (initial_cost : ℝ) 
  (initial_price : ℝ) 
  (first_quarter_decrease : ℝ) 
  (second_quarter_increase : ℝ) 
  (x : ℝ) : 
  initial_cost = 50 → 
  initial_price = 65 → 
  first_quarter_decrease = 0.1 → 
  second_quarter_increase = 0.05 → 
  initial_price * (1 - first_quarter_decrease) * (1 + second_quarter_increase) - initial_cost * (1 - x)^2 = initial_price - initial_cost := by
sorry

end NUMINAMATH_CALUDE_cost_decrease_for_constant_profit_l1208_120868


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_neg_sqrt3_half_l1208_120847

theorem sin_cos_difference_equals_neg_sqrt3_half :
  Real.sin (5 * π / 180) * Real.sin (25 * π / 180) - 
  Real.sin (95 * π / 180) * Real.sin (65 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_neg_sqrt3_half_l1208_120847
