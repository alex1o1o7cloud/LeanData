import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3719_371924

theorem quadratic_no_real_roots
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : c > 0)
  (h3 : |a - b| < c) :
  ∀ x : ℝ, a^2 * x^2 + (b^2 + a^2 - c^2) * x + b^2 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3719_371924


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3719_371918

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h1 : i * i = -1) (h2 : z * (1 + i) = Complex.abs (2 * i)) : z = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3719_371918


namespace NUMINAMATH_CALUDE_base7_multiplication_l3719_371903

/-- Converts a number from base 7 to base 10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Represents a number in base 7 --/
structure Base7 where
  value : ℕ

theorem base7_multiplication :
  let a : Base7 := ⟨345⟩
  let b : Base7 := ⟨3⟩
  let result : Base7 := ⟨1401⟩
  toBase7 (toBase10 a.value * toBase10 b.value) = result.value := by sorry

end NUMINAMATH_CALUDE_base7_multiplication_l3719_371903


namespace NUMINAMATH_CALUDE_min_value_problem_l3719_371911

theorem min_value_problem (x y : ℝ) 
  (h1 : x - 1 ≥ 0)
  (h2 : x - y + 1 ≤ 0)
  (h3 : x + y - 4 ≤ 0) :
  ∃ (m : ℝ), m = 1/4 ∧ ∀ (a b : ℝ), 
    a - 1 ≥ 0 → a - b + 1 ≤ 0 → a + b - 4 ≤ 0 → 
    a / (b + 1) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l3719_371911


namespace NUMINAMATH_CALUDE_inequality_solution_set_abs_b_greater_than_two_l3719_371945

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- Theorem for part I
theorem inequality_solution_set (x : ℝ) :
  f x + f (x + 1) ≥ 5 ↔ x ≥ 4 ∨ x ≤ -1 :=
sorry

-- Theorem for part II
theorem abs_b_greater_than_two (a b : ℝ) :
  |a| > 1 → f (a * b) > |a| * f (b / a) → |b| > 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_abs_b_greater_than_two_l3719_371945


namespace NUMINAMATH_CALUDE_friend_savings_rate_l3719_371913

/-- Proves that given the initial amounts and saving rates, the friend's weekly savings
    that result in equal total savings after 25 weeks is 5 dollars. -/
theorem friend_savings_rate (your_initial : ℕ) (your_weekly : ℕ) (friend_initial : ℕ) (weeks : ℕ) :
  your_initial = 160 →
  your_weekly = 7 →
  friend_initial = 210 →
  weeks = 25 →
  ∃ (friend_weekly : ℕ),
    your_initial + your_weekly * weeks = friend_initial + friend_weekly * weeks ∧
    friend_weekly = 5 :=
by sorry

end NUMINAMATH_CALUDE_friend_savings_rate_l3719_371913


namespace NUMINAMATH_CALUDE_valleyball_league_members_l3719_371914

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 5

/-- The total cost for all members in dollars -/
def total_cost : ℕ := 3300

/-- The number of pairs of socks each member needs -/
def socks_per_member : ℕ := 2

/-- The number of T-shirts each member needs -/
def tshirts_per_member : ℕ := 2

/-- The number of members in the Valleyball Soccer League -/
def number_of_members : ℕ := 97

theorem valleyball_league_members :
  let tshirt_cost := sock_cost + tshirt_additional_cost
  let member_cost := socks_per_member * sock_cost + tshirts_per_member * tshirt_cost
  number_of_members * member_cost = total_cost := by
  sorry


end NUMINAMATH_CALUDE_valleyball_league_members_l3719_371914


namespace NUMINAMATH_CALUDE_film_review_analysis_l3719_371962

structure FilmReviewData where
  total_sample : ℕ
  male_count : ℕ
  female_count : ℕ
  male_negative : ℕ
  female_positive : ℕ
  significance_level : ℝ
  chi_square_critical : ℝ
  stratified_sample_size : ℕ
  coupon_recipients : ℕ

def chi_square_statistic (data : FilmReviewData) : ℝ := sorry

def is_associated (data : FilmReviewData) : Prop :=
  chi_square_statistic data > data.chi_square_critical

def probability_distribution (x : ℕ) : ℝ := sorry

def expected_value : ℝ := sorry

theorem film_review_analysis (data : FilmReviewData) 
  (h1 : data.total_sample = 220)
  (h2 : data.male_count = 110)
  (h3 : data.female_count = 110)
  (h4 : data.male_negative = 70)
  (h5 : data.female_positive = 60)
  (h6 : data.significance_level = 0.010)
  (h7 : data.chi_square_critical = 6.635)
  (h8 : data.stratified_sample_size = 10)
  (h9 : data.coupon_recipients = 3) :
  is_associated data ∧ 
  probability_distribution 0 = 1/30 ∧
  probability_distribution 1 = 3/10 ∧
  probability_distribution 2 = 1/2 ∧
  probability_distribution 3 = 1/6 ∧
  expected_value = 9/5 := by sorry

end NUMINAMATH_CALUDE_film_review_analysis_l3719_371962


namespace NUMINAMATH_CALUDE_salary_fraction_on_food_l3719_371975

theorem salary_fraction_on_food 
  (salary : ℝ) 
  (rent_fraction : ℝ) 
  (clothes_fraction : ℝ) 
  (amount_left : ℝ) 
  (h1 : salary = 160000)
  (h2 : rent_fraction = 1/10)
  (h3 : clothes_fraction = 3/5)
  (h4 : amount_left = 16000)
  (h5 : salary * rent_fraction + salary * clothes_fraction + amount_left + salary * food_fraction = salary) :
  food_fraction = 1/5 := by
sorry

end NUMINAMATH_CALUDE_salary_fraction_on_food_l3719_371975


namespace NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l3719_371906

theorem range_of_a_for_quadratic_inequality :
  (∀ x : ℝ, ∀ a : ℝ, a * x^2 - a * x - 1 ≤ 0) →
  (∀ a : ℝ, (a ∈ Set.Icc (-4 : ℝ) 0) ↔ (∀ x : ℝ, a * x^2 - a * x - 1 ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l3719_371906


namespace NUMINAMATH_CALUDE_power_of_8_mod_100_l3719_371926

theorem power_of_8_mod_100 : 8^2023 % 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_8_mod_100_l3719_371926


namespace NUMINAMATH_CALUDE_initial_men_count_l3719_371935

/-- Represents the initial number of men working on the project -/
def initial_men : ℕ := sorry

/-- Represents the number of days to complete the work with the initial group -/
def initial_days : ℕ := 40

/-- Represents the number of men who leave the project -/
def men_who_leave : ℕ := 20

/-- Represents the number of days worked before some men leave -/
def days_before_leaving : ℕ := 10

/-- Represents the number of days to complete the remaining work after some men leave -/
def remaining_days : ℕ := 40

/-- Work rate of one man per day -/
def work_rate : ℚ := 1 / (initial_men * initial_days)

/-- Fraction of work completed before some men leave -/
def work_completed_before_leaving : ℚ := work_rate * initial_men * days_before_leaving

/-- Fraction of work remaining after some men leave -/
def remaining_work : ℚ := 1 - work_completed_before_leaving

/-- The theorem states that given the conditions, the initial number of men is 80 -/
theorem initial_men_count : initial_men = 80 := by sorry

end NUMINAMATH_CALUDE_initial_men_count_l3719_371935


namespace NUMINAMATH_CALUDE_sum_of_powers_eq_7290_l3719_371952

/-- The power of a triple of positive integers -/
def power (x y z : ℕ) : ℕ := max x (max y z) + min x (min y z)

/-- The sum of powers of all triples (x,y,z) where x,y,z ≤ 9 -/
def sum_of_powers : ℕ :=
  (Finset.range 9).sum (fun x =>
    (Finset.range 9).sum (fun y =>
      (Finset.range 9).sum (fun z =>
        power (x + 1) (y + 1) (z + 1))))

theorem sum_of_powers_eq_7290 : sum_of_powers = 7290 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_eq_7290_l3719_371952


namespace NUMINAMATH_CALUDE_calculate_expression_l3719_371978

theorem calculate_expression : -1^4 + 16 / (-2)^3 * |(-3) - 1| = -9 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3719_371978


namespace NUMINAMATH_CALUDE_cube_equation_solution_l3719_371927

theorem cube_equation_solution :
  ∃ x : ℝ, (x - 5)^3 = (1/27)⁻¹ ∧ x = 8 :=
by sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l3719_371927


namespace NUMINAMATH_CALUDE_xyz_value_l3719_371974

theorem xyz_value (x y z : ℝ) (h1 : x^2 * y * z^3 = 7^3) (h2 : x * y^2 = 7^9) : 
  x * y * z = 7^4 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3719_371974


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l3719_371919

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the path length of vertex C when rotating a triangle along a rectangle -/
def pathLengthC (t : Triangle) (r : Rectangle) : ℝ :=
  sorry

theorem triangle_rotation_path_length :
  let t : Triangle := { a := 2, b := 3, c := 4 }
  let r : Rectangle := { width := 8, height := 6 }
  pathLengthC t r = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l3719_371919


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_17_mod_26_l3719_371933

theorem largest_five_digit_congruent_17_mod_26 : ∃ (n : ℕ), n = 99997 ∧ 
  n < 100000 ∧ 
  n % 26 = 17 ∧ 
  ∀ (m : ℕ), m < 100000 → m % 26 = 17 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_17_mod_26_l3719_371933


namespace NUMINAMATH_CALUDE_staircase_perimeter_l3719_371990

/-- A staircase-shaped region with specific properties -/
structure StaircaseRegion where
  congruentSides : ℕ
  sideLength : ℝ
  area : ℝ

/-- Calculate the perimeter of the staircase region -/
def perimeter (s : StaircaseRegion) : ℝ :=
  7 + 11 + 3 + 7 + s.congruentSides * s.sideLength

/-- Theorem: The perimeter of the specific staircase region is 39 feet -/
theorem staircase_perimeter :
  ∀ s : StaircaseRegion,
    s.congruentSides = 10 ∧
    s.sideLength = 1 ∧
    s.area = 74 →
    perimeter s = 39 := by
  sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l3719_371990


namespace NUMINAMATH_CALUDE_fraction_equality_l3719_371901

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3719_371901


namespace NUMINAMATH_CALUDE_count_positive_numbers_l3719_371932

theorem count_positive_numbers : 
  let numbers : List ℚ := [-3, -1, 1/3, 0, -3/7, 2017]
  (numbers.filter (λ x => x > 0)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_positive_numbers_l3719_371932


namespace NUMINAMATH_CALUDE_original_number_exists_l3719_371987

theorem original_number_exists : ∃ x : ℝ, 4 * ((x^3 / 5)^2 + 15) = 224 := by
  sorry

end NUMINAMATH_CALUDE_original_number_exists_l3719_371987


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3719_371972

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ -3 → x ≠ 3 →
  (1 - 1 / (x + 3)) / ((x^2 - 9) / (x^2 + 6*x + 9)) = (x + 2) / (x - 3) ∧
  (2 + 2) / (2 - 3) = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3719_371972


namespace NUMINAMATH_CALUDE_candy_sampling_percentage_l3719_371999

theorem candy_sampling_percentage : 
  ∀ (total_customers : ℝ) (caught_percent : ℝ) (not_caught_percent : ℝ),
  caught_percent = 22 →
  not_caught_percent = 12 →
  ∃ (total_sampling_percent : ℝ),
    total_sampling_percent = caught_percent + (not_caught_percent / 100) * total_sampling_percent ∧
    total_sampling_percent = 25 := by
  sorry

end NUMINAMATH_CALUDE_candy_sampling_percentage_l3719_371999


namespace NUMINAMATH_CALUDE_total_weight_of_balls_l3719_371916

def blue_ball_weight : ℝ := 6
def brown_ball_weight : ℝ := 3.12

theorem total_weight_of_balls :
  blue_ball_weight + brown_ball_weight = 9.12 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_balls_l3719_371916


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_square_sum_l3719_371968

theorem arithmetic_geometric_mean_square_sum (a b : ℝ) :
  (a + b) / 2 = 20 → Real.sqrt (a * b) = Real.sqrt 135 → a^2 + b^2 = 1330 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_square_sum_l3719_371968


namespace NUMINAMATH_CALUDE_test_average_l3719_371996

theorem test_average (male_count : ℕ) (female_count : ℕ) (male_avg : ℝ) (female_avg : ℝ)
  (h1 : male_count = 8)
  (h2 : female_count = 24)
  (h3 : male_avg = 84)
  (h4 : female_avg = 92) :
  (male_count * male_avg + female_count * female_avg) / (male_count + female_count) = 90 := by
  sorry

end NUMINAMATH_CALUDE_test_average_l3719_371996


namespace NUMINAMATH_CALUDE_beautiful_equations_proof_l3719_371970

/-- Two linear equations are "beautiful equations" if the sum of their solutions is 1 -/
def beautiful_equations (eq1 eq2 : ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), eq1 x ∧ eq2 y ∧ x + y = 1

/-- The first pair of equations -/
def eq1 (x : ℝ) : Prop := 4 * x - (x + 5) = 1

/-- The second pair of equations -/
def eq2 (y : ℝ) : Prop := -2 * y - y = 3

/-- The third pair of equations -/
def eq3 (m : ℝ) (x : ℝ) : Prop := x / 2 + m = 0

/-- The fourth pair of equations -/
def eq4 (x : ℝ) : Prop := 3 * x = x + 4

theorem beautiful_equations_proof :
  (beautiful_equations eq1 eq2) ∧
  (∀ m : ℝ, beautiful_equations (eq3 m) eq4 → m = 1/2) := by sorry

end NUMINAMATH_CALUDE_beautiful_equations_proof_l3719_371970


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l3719_371939

def restaurant_bill (num_people_1 num_people_2 : ℕ) (cost_1 cost_2 service_charge : ℚ) 
  (discount_rate tip_rate : ℚ) : ℚ :=
  let meal_cost := num_people_1 * cost_1 + num_people_2 * cost_2
  let total_before_discount := meal_cost + service_charge
  let discount := discount_rate * meal_cost
  let total_after_discount := total_before_discount - discount
  let tip := tip_rate * total_before_discount
  total_after_discount + tip

theorem restaurant_bill_calculation :
  restaurant_bill 10 5 18 25 50 (5/100) (10/100) = 375.25 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l3719_371939


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3719_371989

/-- The equation of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/8 = 1

/-- A point is on the square if its coordinates are equal in absolute value -/
def on_square (x y : ℝ) : Prop := |x| = |y|

/-- The square is inscribed in the ellipse -/
def inscribed_square (t : ℝ) : Prop := 
  ellipse t t ∧ on_square t t ∧ t > 0

/-- The area of the inscribed square -/
def square_area (t : ℝ) : ℝ := (2*t)^2

theorem inscribed_square_area : 
  ∃ t : ℝ, inscribed_square t ∧ square_area t = 32/3 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3719_371989


namespace NUMINAMATH_CALUDE_candice_arrival_time_l3719_371923

/-- Represents the driving scenario of Candice --/
structure DrivingScenario where
  initial_speed : ℕ
  final_speed : ℕ
  total_distance : ℚ
  drive_time : ℕ

/-- The conditions of Candice's drive --/
def candice_drive : DrivingScenario :=
  { initial_speed := 10,
    final_speed := 6,
    total_distance := 2/3,
    drive_time := 5 }

/-- Theorem stating that Candice arrives home at 5:05 PM --/
theorem candice_arrival_time (d : DrivingScenario) 
  (h1 : d.initial_speed > d.final_speed)
  (h2 : d.drive_time > 0)
  (h3 : d.total_distance = (d.initial_speed + d.final_speed + 1) * (d.initial_speed - d.final_speed) / 120) :
  d.drive_time = 5 ∧ d = candice_drive :=
sorry

end NUMINAMATH_CALUDE_candice_arrival_time_l3719_371923


namespace NUMINAMATH_CALUDE_max_projection_area_is_one_l3719_371912

/-- A tetrahedron with specific properties -/
structure SpecialTetrahedron where
  /-- Two adjacent faces are isosceles right triangles -/
  isosceles_right_faces : Bool
  /-- The hypotenuse of the isosceles right triangles is 2 -/
  hypotenuse : ℝ
  /-- The dihedral angle between the two adjacent faces is 60 degrees -/
  dihedral_angle : ℝ
  /-- The tetrahedron rotates around the common edge of the two faces -/
  rotates_around_common_edge : Bool

/-- The maximum projection area of the rotating tetrahedron -/
def max_projection_area (t : SpecialTetrahedron) : ℝ := sorry

/-- Theorem stating that the maximum projection area is 1 -/
theorem max_projection_area_is_one (t : SpecialTetrahedron) 
  (h1 : t.isosceles_right_faces = true)
  (h2 : t.hypotenuse = 2)
  (h3 : t.dihedral_angle = Real.pi / 3)  -- 60 degrees in radians
  (h4 : t.rotates_around_common_edge = true) :
  max_projection_area t = 1 := by sorry

end NUMINAMATH_CALUDE_max_projection_area_is_one_l3719_371912


namespace NUMINAMATH_CALUDE_alcohol_mixture_problem_l3719_371991

theorem alcohol_mixture_problem (x : ℝ) 
  (h1 : x > 0)  -- Ensure x is positive
  (h2 : 0.9 * x = 0.54 * (x + 16)) : x = 24 := by
  sorry

#check alcohol_mixture_problem

end NUMINAMATH_CALUDE_alcohol_mixture_problem_l3719_371991


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3719_371946

theorem fraction_to_decimal : (7 : ℚ) / 16 = (4375 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3719_371946


namespace NUMINAMATH_CALUDE_wire_service_reporters_l3719_371910

theorem wire_service_reporters (x y both_local non_local_politics international_only : ℝ) 
  (hx : x = 35)
  (hy : y = 25)
  (hboth : both_local = 20)
  (hnon_local : non_local_politics = 30)
  (hinter : international_only = 15) :
  100 - ((x + y - both_local) + non_local_politics + international_only) = 75 := by
  sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l3719_371910


namespace NUMINAMATH_CALUDE_cases_in_2005_l3719_371951

/-- Calculates the number of cases in a given year assuming a linear decrease --/
def caseCount (initialYear initialCases finalYear finalCases targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let yearlyDecrease := totalDecrease / totalYears
  let targetYearsSinceInitial := targetYear - initialYear
  initialCases - (yearlyDecrease * targetYearsSinceInitial)

/-- Theorem stating that the number of cases in 2005 is 134,000 --/
theorem cases_in_2005 :
  caseCount 1980 800000 2010 800 2005 = 134000 := by
  sorry

end NUMINAMATH_CALUDE_cases_in_2005_l3719_371951


namespace NUMINAMATH_CALUDE_direction_vector_c_value_l3719_371948

-- Define the two points on the line
def point1 : ℝ × ℝ := (-7, 3)
def point2 : ℝ × ℝ := (-3, -1)

-- Define the direction vector
def direction_vector (c : ℝ) : ℝ × ℝ := (4, c)

-- Theorem statement
theorem direction_vector_c_value :
  ∃ (c : ℝ), direction_vector c = (point2.1 - point1.1, point2.2 - point1.2) :=
by sorry

end NUMINAMATH_CALUDE_direction_vector_c_value_l3719_371948


namespace NUMINAMATH_CALUDE_circle_properties_l3719_371979

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 4

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop := x + 2 * y = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define the theorem
theorem circle_properties :
  -- Part 1: Equation of circle O
  (∀ x y : ℝ, circle_O x y ↔ x^2 + y^2 = 4) ∧
  -- Part 2: Equation of line MN
  (∀ m n : ℝ × ℝ,
    circle_O m.1 m.2 ∧ circle_O n.1 n.2 ∧
    symmetry_line ((m.1 + n.1) / 2) ((m.2 + n.2) / 2) ∧
    (m.1 - n.1)^2 + (m.2 - n.2)^2 = 12 →
    ∃ b : ℝ, (2 * m.1 - m.2 + b = 0 ∧ 2 * n.1 - n.2 + b = 0) ∧ b^2 = 5) ∧
  -- Part 3: Range of PA · PB
  (∀ p : ℝ × ℝ,
    circle_O p.1 p.2 ∧
    ((p.1 + 2)^2 + p.2^2) * ((p.1 - 2)^2 + p.2^2) = (p.1^2 + p.2^2)^2 →
    -2 ≤ ((p.1 + 2) * (p.1 - 2) + p.2^2) ∧ ((p.1 + 2) * (p.1 - 2) + p.2^2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3719_371979


namespace NUMINAMATH_CALUDE_min_max_abs_expressions_l3719_371964

theorem min_max_abs_expressions (x y : ℝ) :
  ∃ (x₀ y₀ : ℝ), max (|2 * x₀ + y₀|) (max (|x₀ - y₀|) (|1 + y₀|)) = (1/2 : ℝ) ∧
  ∀ (x y : ℝ), (1/2 : ℝ) ≤ max (|2 * x + y|) (max (|x - y|) (|1 + y|)) :=
by sorry

end NUMINAMATH_CALUDE_min_max_abs_expressions_l3719_371964


namespace NUMINAMATH_CALUDE_number_of_cows_l3719_371966

-- Define the types for animals
inductive Animal : Type
| Cow : Animal
| Chicken : Animal
| Pig : Animal

-- Define the farm
def Farm : Type := Animal → ℕ

-- Define the number of legs for each animal
def legs : Animal → ℕ
| Animal.Cow => 4
| Animal.Chicken => 2
| Animal.Pig => 4

-- Define the total number of animals
def total_animals (farm : Farm) : ℕ :=
  farm Animal.Cow + farm Animal.Chicken + farm Animal.Pig

-- Define the total number of legs
def total_legs (farm : Farm) : ℕ :=
  farm Animal.Cow * legs Animal.Cow +
  farm Animal.Chicken * legs Animal.Chicken +
  farm Animal.Pig * legs Animal.Pig

-- State the theorem
theorem number_of_cows (farm : Farm) : 
  farm Animal.Chicken = 6 ∧ 
  total_legs farm = 20 + 2 * total_animals farm → 
  farm Animal.Cow = 6 :=
sorry

end NUMINAMATH_CALUDE_number_of_cows_l3719_371966


namespace NUMINAMATH_CALUDE_range_of_m_l3719_371961

theorem range_of_m (f : ℝ → ℝ) (h : ∀ x ∈ Set.Icc 0 1, f x ≥ m) :
  Set.Iic (-3 : ℝ) = {m : ℝ | ∀ x ∈ Set.Icc 0 1, f x ≥ m} :=
by sorry

#check range_of_m (fun x ↦ x^2 - 4*x)

end NUMINAMATH_CALUDE_range_of_m_l3719_371961


namespace NUMINAMATH_CALUDE_triangle_count_theorem_l3719_371905

/-- The number of trees planted in a triangular shape -/
def num_trees : ℕ := 21

/-- The number of ways to choose 3 trees from num_trees -/
def total_choices : ℕ := Nat.choose num_trees 3

/-- The number of ways to choose 3 collinear trees -/
def collinear_choices : ℕ := 114

/-- The number of ways to choose 3 trees to form a non-degenerate triangle -/
def non_degenerate_triangles : ℕ := total_choices - collinear_choices

theorem triangle_count_theorem : non_degenerate_triangles = 1216 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_theorem_l3719_371905


namespace NUMINAMATH_CALUDE_gcf_of_60_and_90_l3719_371967

theorem gcf_of_60_and_90 : Nat.gcd 60 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_60_and_90_l3719_371967


namespace NUMINAMATH_CALUDE_white_beans_count_l3719_371915

/-- The number of white jelly beans in one bag -/
def white_beans_in_bag : ℕ := sorry

/-- The number of bags needed to fill the fishbowl -/
def bags_in_fishbowl : ℕ := 3

/-- The number of red jelly beans in one bag -/
def red_beans_in_bag : ℕ := 24

/-- The total number of red and white jelly beans in the fishbowl -/
def total_red_white_in_fishbowl : ℕ := 126

theorem white_beans_count : white_beans_in_bag = 18 := by
  sorry

end NUMINAMATH_CALUDE_white_beans_count_l3719_371915


namespace NUMINAMATH_CALUDE_line_equation_l3719_371998

/-- A line in 2D space defined by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The problem statement -/
theorem line_equation : 
  ∃ (l : Line), 
    l.contains ⟨-1, 2⟩ ∧ 
    l.perpendicular ⟨2, -3, 0⟩ ∧
    l = ⟨3, 2, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l3719_371998


namespace NUMINAMATH_CALUDE_shape_to_square_possible_l3719_371992

/-- Represents a shape on a graph paper -/
structure Shape where
  -- Add necessary fields to represent the shape
  -- For example, you might use a list of coordinates

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields to represent a triangle
  -- For example, you might use three points

/-- Represents a square -/
structure Square where
  -- Add necessary fields to represent a square
  -- For example, you might use side length and position

/-- Function to check if a list of triangles can form a square -/
def can_form_square (triangles : List Triangle) : Prop :=
  -- Define the logic to check if triangles can form a square
  sorry

/-- The main theorem stating that the shape can be divided into 5 triangles
    that can be rearranged to form a square -/
theorem shape_to_square_possible (s : Shape) : 
  ∃ (t1 t2 t3 t4 t5 : Triangle), 
    (can_form_square [t1, t2, t3, t4, t5]) ∧ 
    (-- Add condition that t1, t2, t3, t4, t5 are a valid division of s
     sorry) := by
  sorry

end NUMINAMATH_CALUDE_shape_to_square_possible_l3719_371992


namespace NUMINAMATH_CALUDE_equal_charges_at_60_minutes_l3719_371958

/-- United Telephone's base rate in dollars -/
def united_base : ℝ := 9

/-- United Telephone's per-minute rate in dollars -/
def united_per_minute : ℝ := 0.25

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℝ := 12

/-- Atlantic Call's per-minute rate in dollars -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which the charges are equal -/
def equal_minutes : ℝ := 60

theorem equal_charges_at_60_minutes :
  united_base + united_per_minute * equal_minutes =
  atlantic_base + atlantic_per_minute * equal_minutes :=
by sorry

end NUMINAMATH_CALUDE_equal_charges_at_60_minutes_l3719_371958


namespace NUMINAMATH_CALUDE_g_inverse_exists_g_inverse_composition_g_inverse_triple_composition_l3719_371917

def g : Fin 5 → Fin 5
  | 1 => 4
  | 2 => 3
  | 3 => 1
  | 4 => 5
  | 5 => 2

theorem g_inverse_exists : Function.Bijective g := sorry

theorem g_inverse_composition (x : Fin 5) : Function.invFun g (g x) = x := sorry

theorem g_inverse_triple_composition :
  Function.invFun g (Function.invFun g (Function.invFun g 3)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_g_inverse_exists_g_inverse_composition_g_inverse_triple_composition_l3719_371917


namespace NUMINAMATH_CALUDE_cycle_price_problem_l3719_371943

theorem cycle_price_problem (selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : selling_price = 1170)
  (h2 : gain_percent = 30) :
  let original_price := selling_price / (1 + gain_percent / 100)
  original_price = 900 := by
sorry

end NUMINAMATH_CALUDE_cycle_price_problem_l3719_371943


namespace NUMINAMATH_CALUDE_xiaohuo_has_448_books_l3719_371938

/-- The number of books Xiaohuo, Xiaoyan, and Xiaoyi have collectively -/
def total_books : ℕ := 1248

/-- The number of books Xiaohuo has -/
def xiaohuo_books : ℕ := sorry

/-- The number of books Xiaoyan has -/
def xiaoyan_books : ℕ := sorry

/-- The number of books Xiaoyi has -/
def xiaoyi_books : ℕ := sorry

/-- Xiaohuo has 64 more books than Xiaoyan -/
axiom xiaohuo_more_than_xiaoyan : xiaohuo_books = xiaoyan_books + 64

/-- Xiaoyan has 32 fewer books than Xiaoyi -/
axiom xiaoyan_fewer_than_xiaoyi : xiaoyan_books = xiaoyi_books - 32

/-- The total number of books is the sum of books owned by each person -/
axiom total_books_sum : total_books = xiaohuo_books + xiaoyan_books + xiaoyi_books

/-- Theorem: Xiaohuo has 448 books -/
theorem xiaohuo_has_448_books : xiaohuo_books = 448 := by sorry

end NUMINAMATH_CALUDE_xiaohuo_has_448_books_l3719_371938


namespace NUMINAMATH_CALUDE_basketball_wins_needed_l3719_371936

/-- Calculates the number of additional games a basketball team needs to win to achieve a target win percentage -/
theorem basketball_wins_needed
  (games_played : ℕ)
  (games_won : ℕ)
  (games_remaining : ℕ)
  (target_percentage : ℚ)
  (h1 : games_played = 50)
  (h2 : games_won = 35)
  (h3 : games_remaining = 25)
  (h4 : target_percentage = 64 / 100) :
  ⌈(target_percentage * ↑(games_played + games_remaining) - ↑games_won)⌉ = 13 :=
by sorry

end NUMINAMATH_CALUDE_basketball_wins_needed_l3719_371936


namespace NUMINAMATH_CALUDE_sheridan_cats_l3719_371928

/-- The number of cats Mrs. Sheridan has after giving some away -/
def remaining_cats (initial : ℝ) (given_away : ℝ) : ℝ :=
  initial - given_away

/-- Proof that Mrs. Sheridan has 3.0 cats after giving away 14.0 cats -/
theorem sheridan_cats : remaining_cats 17.0 14.0 = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_cats_l3719_371928


namespace NUMINAMATH_CALUDE_birthday_pigeonhole_l3719_371988

theorem birthday_pigeonhole (n m : ℕ) (h1 : n = 39) (h2 : m = 12) :
  ∃ k : ℕ, k ≤ m ∧ 4 ≤ (n / m + (if n % m = 0 then 0 else 1)) := by
  sorry

end NUMINAMATH_CALUDE_birthday_pigeonhole_l3719_371988


namespace NUMINAMATH_CALUDE_calculator_game_sum_l3719_371942

/-- Represents the state of the calculators -/
structure CalculatorState :=
  (calc1 : ℤ)
  (calc2 : ℤ)
  (calc3 : ℤ)

/-- The operation performed on the calculators in each turn -/
def squareOperation (state : CalculatorState) : CalculatorState :=
  { calc1 := state.calc1 ^ 2,
    calc2 := state.calc2 ^ 2,
    calc3 := state.calc3 ^ 2 }

/-- The initial state of the calculators -/
def initialState : CalculatorState :=
  { calc1 := 2,
    calc2 := -2,
    calc3 := 0 }

/-- The theorem to be proved -/
theorem calculator_game_sum (n : ℕ) (h : n ≥ 1) :
  (squareOperation^[n] initialState).calc1 +
  (squareOperation^[n] initialState).calc2 +
  (squareOperation^[n] initialState).calc3 = 8 :=
sorry

end NUMINAMATH_CALUDE_calculator_game_sum_l3719_371942


namespace NUMINAMATH_CALUDE_stratified_sampling_group_c_l3719_371920

/-- Represents the number of cities selected from a group in a stratified sampling. -/
def citiesSelected (totalCities : ℕ) (sampleSize : ℕ) (groupSize : ℕ) : ℕ :=
  (sampleSize * groupSize) / totalCities

/-- Proves that in a stratified sampling of 6 cities from 24 total cities, 
    where 8 cities belong to group C, the number of cities selected from group C is 2. -/
theorem stratified_sampling_group_c : 
  citiesSelected 24 6 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_c_l3719_371920


namespace NUMINAMATH_CALUDE_k_range_theorem_l3719_371959

theorem k_range_theorem (k : ℝ) : 
  (∀ m : ℝ, 0 < m ∧ m < 3/2 → (2/m) + (1/(3-2*m)) ≥ k^2 + 2*k) → 
  -3 ≤ k ∧ k ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_k_range_theorem_l3719_371959


namespace NUMINAMATH_CALUDE_budget_allocation_l3719_371957

theorem budget_allocation (transportation research_development utilities supplies salaries equipment : ℝ) :
  transportation = 20 →
  research_development = 9 →
  utilities = 5 →
  supplies = 2 →
  salaries = 216 / 360 * 100 →
  transportation + research_development + utilities + supplies + salaries + equipment = 100 →
  equipment = 4 :=
by sorry

end NUMINAMATH_CALUDE_budget_allocation_l3719_371957


namespace NUMINAMATH_CALUDE_decagon_triangles_l3719_371931

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def num_triangles_in_decagon : ℕ := 120

/-- Theorem: The number of triangles that can be formed using the vertices of a regular decagon is 120 -/
theorem decagon_triangles : num_triangles_in_decagon = 120 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangles_l3719_371931


namespace NUMINAMATH_CALUDE_water_amount_in_sport_formulation_l3719_371941

/-- Represents the ratio of ingredients in a flavored drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

/-- The sport formulation of the drink -/
def sport_ratio (r : DrinkRatio) : DrinkRatio :=
  { flavoring := r.flavoring,
    corn_syrup := r.corn_syrup / 3,
    water := r.water * 2 }

theorem water_amount_in_sport_formulation :
  let sport := sport_ratio standard_ratio
  ∀ corn_syrup_oz : ℚ,
    corn_syrup_oz = 1 →
    (sport.water / sport.corn_syrup) * corn_syrup_oz = 15 := by
  sorry

end NUMINAMATH_CALUDE_water_amount_in_sport_formulation_l3719_371941


namespace NUMINAMATH_CALUDE_bubble_theorem_l3719_371904

/-- The number of bubbles appearing each minute -/
def k : ℕ := 36

/-- The number of minutes after which bubbles start bursting -/
def m : ℕ := 80

/-- The maximum number of bubbles on the screen -/
def max_bubbles : ℕ := k * (k + 21) / 2

theorem bubble_theorem :
  (∀ n : ℕ, n ≤ 10 + m → n * k = n * k) ∧  -- Bubbles appear every minute
  ((10 + m) * k = m * (m + 1) / 2) ∧  -- All bubbles eventually burst
  (∀ n : ℕ, n ≤ m → n * (n + 1) / 2 ≤ (10 + n) * k) ∧  -- Bursting pattern
  (k * (k + 21) / 2 = 1026) →  -- Definition of max_bubbles
  max_bubbles = 1026 := by sorry

#eval max_bubbles  -- Should output 1026

end NUMINAMATH_CALUDE_bubble_theorem_l3719_371904


namespace NUMINAMATH_CALUDE_dante_recipe_eggs_l3719_371947

theorem dante_recipe_eggs :
  ∀ (eggs flour : ℕ),
  flour = eggs / 2 →
  flour + eggs = 90 →
  eggs = 60 := by
sorry

end NUMINAMATH_CALUDE_dante_recipe_eggs_l3719_371947


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l3719_371921

theorem parabola_point_coordinates (x y : ℝ) :
  y^2 = 4*x →                             -- P is on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 100 →                 -- Distance from P to focus (1, 0) is 10
  (x = 9 ∧ (y = 6 ∨ y = -6)) :=           -- Coordinates of P are (9, ±6)
by
  sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l3719_371921


namespace NUMINAMATH_CALUDE_tower_combinations_l3719_371965

def red_cubes : ℕ := 2
def blue_cubes : ℕ := 4
def green_cubes : ℕ := 5
def tower_height : ℕ := 7

/-- The number of different towers with a height of 7 cubes that can be built
    with 2 red cubes, 4 blue cubes, and 5 green cubes. -/
def number_of_towers : ℕ := 420

theorem tower_combinations :
  (red_cubes + blue_cubes + green_cubes - tower_height = 2) →
  number_of_towers = 420 :=
by sorry

end NUMINAMATH_CALUDE_tower_combinations_l3719_371965


namespace NUMINAMATH_CALUDE_horner_first_step_value_v₁_equals_30_l3719_371981

/-- Horner's Rule first step for polynomial evaluation -/
def horner_first_step (a₄ a₃ : ℝ) (x : ℝ) : ℝ :=
  a₄ * x + a₃

/-- The polynomial f(x) = 3x⁴ + 2x² + x + 4 -/
def f (x : ℝ) : ℝ :=
  3 * x^4 + 2 * x^2 + x + 4

theorem horner_first_step_value :
  horner_first_step 3 0 10 = 30 :=
by sorry

theorem v₁_equals_30 :
  horner_first_step 3 0 10 = 30 :=
by sorry

end NUMINAMATH_CALUDE_horner_first_step_value_v₁_equals_30_l3719_371981


namespace NUMINAMATH_CALUDE_test_probabilities_l3719_371971

/-- The probability that exactly two out of three students pass their tests. -/
def prob_two_pass (pA pB pC : ℚ) : ℚ :=
  pA * pB * (1 - pC) + pA * (1 - pB) * pC + (1 - pA) * pB * pC

/-- The probability that at least one out of three students fails their test. -/
def prob_at_least_one_fail (pA pB pC : ℚ) : ℚ :=
  1 - pA * pB * pC

theorem test_probabilities (pA pB pC : ℚ) 
  (hA : pA = 4/5) (hB : pB = 3/5) (hC : pC = 7/10) : 
  prob_two_pass pA pB pC = 113/250 ∧ 
  prob_at_least_one_fail pA pB pC = 83/125 := by
  sorry

#eval prob_two_pass (4/5) (3/5) (7/10)
#eval prob_at_least_one_fail (4/5) (3/5) (7/10)

end NUMINAMATH_CALUDE_test_probabilities_l3719_371971


namespace NUMINAMATH_CALUDE_third_circle_radius_l3719_371953

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (c1.radius + c2.radius) ^ 2

/-- Checks if a circle is tangent to the x-axis -/
def is_tangent_to_x_axis (c : Circle) : Prop :=
  c.center.2 = c.radius

/-- The main theorem -/
theorem third_circle_radius 
  (circle_A circle_B circle_C : Circle)
  (h1 : circle_A.radius = 2)
  (h2 : circle_B.radius = 3)
  (h3 : are_externally_tangent circle_A circle_B)
  (h4 : circle_A.center.1 + 6 = circle_B.center.1)
  (h5 : circle_A.center.2 = circle_B.center.2)
  (h6 : are_externally_tangent circle_A circle_C)
  (h7 : are_externally_tangent circle_B circle_C)
  (h8 : is_tangent_to_x_axis circle_C) :
  circle_C.radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_circle_radius_l3719_371953


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3719_371994

-- Define the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ := 3 - 2 * n

-- Define the sum of the first k terms
def sum_of_terms (k : ℕ) : ℤ := k * (arithmetic_sequence 1 + arithmetic_sequence k) / 2

-- Theorem statement
theorem arithmetic_sequence_problem :
  (arithmetic_sequence 1 = 1) ∧
  (arithmetic_sequence 3 = -3) ∧
  (∃ k : ℕ, sum_of_terms k = -35 ∧ k = 7) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3719_371994


namespace NUMINAMATH_CALUDE_existence_of_divisibility_l3719_371977

/-- The largest proper divisor of a positive integer -/
def largest_proper_divisor (n : ℕ) : ℕ := sorry

/-- The sequence u_n as defined in the problem -/
def u : ℕ → ℕ
  | 0 => sorry  -- This value is not specified in the problem
  | 1 => sorry  -- We only know u_1 > 0, but not its exact value
  | (n + 2) => u (n + 1) + largest_proper_divisor (u (n + 1))

theorem existence_of_divisibility :
  ∃ N : ℕ, ∀ n : ℕ, n > N → (3^2019 : ℕ) ∣ u n :=
sorry

end NUMINAMATH_CALUDE_existence_of_divisibility_l3719_371977


namespace NUMINAMATH_CALUDE_circle_transformation_l3719_371956

/-- Coordinate transformation φ -/
def φ (x y : ℝ) : ℝ × ℝ :=
  (4 * x, 2 * y)

/-- Original circle equation -/
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Transformed equation -/
def transformed_equation (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 4) = 1

theorem circle_transformation (x y : ℝ) :
  original_circle x y ↔ transformed_equation (φ x y).1 (φ x y).2 :=
by sorry

end NUMINAMATH_CALUDE_circle_transformation_l3719_371956


namespace NUMINAMATH_CALUDE_sin_cos_inequality_l3719_371983

theorem sin_cos_inequality (t : ℝ) (h : t > 0) : 3 * Real.sin t < 2 * t + t * Real.cos t := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_inequality_l3719_371983


namespace NUMINAMATH_CALUDE_inequality_proof_l3719_371930

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x + y ≤ (y^2 / x) + (x^2 / y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3719_371930


namespace NUMINAMATH_CALUDE_cameron_typing_speed_l3719_371973

/-- The number of words Cameron could type per minute before breaking his arm -/
def words_before : ℕ := 10

/-- The difference in words typed in 5 minutes before and after breaking his arm -/
def word_difference : ℕ := 10

/-- The number of words Cameron could type per minute after breaking his arm -/
def words_after : ℕ := 8

/-- Proof that Cameron could type 8 words per minute after breaking his arm -/
theorem cameron_typing_speed :
  words_after = 8 ∧
  words_before * 5 - words_after * 5 = word_difference :=
by sorry

end NUMINAMATH_CALUDE_cameron_typing_speed_l3719_371973


namespace NUMINAMATH_CALUDE_kelly_games_theorem_l3719_371902

/-- The number of games Kelly needs to give away to reach her desired number of games -/
def games_to_give_away (initial_games desired_games : ℕ) : ℕ :=
  initial_games - desired_games

theorem kelly_games_theorem (initial_games desired_games : ℕ) 
  (h1 : initial_games = 120) (h2 : desired_games = 20) : 
  games_to_give_away initial_games desired_games = 100 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_theorem_l3719_371902


namespace NUMINAMATH_CALUDE_original_sales_tax_percentage_l3719_371997

/-- Proves that the original sales tax percentage was 0.5%, given the conditions of the problem -/
theorem original_sales_tax_percentage
  (new_tax_rate : ℚ)
  (market_price : ℚ)
  (tax_difference : ℚ)
  (h1 : new_tax_rate = 10 / 3)
  (h2 : market_price = 9000)
  (h3 : tax_difference = 15) :
  ∃ (original_tax_rate : ℚ),
    original_tax_rate = 1 / 2 ∧
    original_tax_rate * market_price - new_tax_rate * market_price = tax_difference :=
by
  sorry

end NUMINAMATH_CALUDE_original_sales_tax_percentage_l3719_371997


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3719_371907

theorem right_triangle_perimeter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_area : (1/2) * a * b = 150) (h_leg : a = 30) : 
  a + b + c = 40 + 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3719_371907


namespace NUMINAMATH_CALUDE_inequality_preservation_l3719_371925

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3719_371925


namespace NUMINAMATH_CALUDE_classroom_seats_count_l3719_371963

/-- Represents a rectangular classroom with seats arranged in rows and columns. -/
structure Classroom where
  seats_left : ℕ  -- Number of seats to the left of the chosen seat
  seats_right : ℕ  -- Number of seats to the right of the chosen seat
  rows_front : ℕ  -- Number of rows in front of the chosen seat
  rows_back : ℕ  -- Number of rows behind the chosen seat

/-- Calculates the total number of seats in the classroom. -/
def total_seats (c : Classroom) : ℕ :=
  (c.seats_left + c.seats_right + 1) * (c.rows_front + c.rows_back + 1)

/-- Theorem stating that a classroom with the given properties has 399 seats. -/
theorem classroom_seats_count :
  ∀ (c : Classroom),
    c.seats_left = 6 →
    c.seats_right = 12 →
    c.rows_front = 7 →
    c.rows_back = 13 →
    total_seats c = 399 := by
  sorry

end NUMINAMATH_CALUDE_classroom_seats_count_l3719_371963


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3719_371969

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set of the first inequality
def S₁ : Set ℝ := {x | x < -2 ∨ x > -1/2}

-- State the theorem
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : ∀ x, f a b c x < 0 ↔ x ∈ S₁) :
  ∀ x, f a (-b) c x > 0 ↔ 1/2 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3719_371969


namespace NUMINAMATH_CALUDE_ellipse_max_min_sum_l3719_371984

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the function we want to maximize/minimize
def f (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem ellipse_max_min_sum :
  (∃ x y : ℝ, ellipse x y ∧ f x y = Real.sqrt 5) ∧
  (∃ x y : ℝ, ellipse x y ∧ f x y = -Real.sqrt 5) ∧
  (∀ x y : ℝ, ellipse x y → f x y ≤ Real.sqrt 5) ∧
  (∀ x y : ℝ, ellipse x y → f x y ≥ -Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_max_min_sum_l3719_371984


namespace NUMINAMATH_CALUDE_unique_number_l3719_371937

theorem unique_number : ∃! n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧  -- three-digit number
  (n / 100 = 4) ∧  -- starts with 4
  ((n % 100) * 10 + 4 = (3 * n) / 4)  -- moving 4 to end results in 0.75 times original
  := by sorry

end NUMINAMATH_CALUDE_unique_number_l3719_371937


namespace NUMINAMATH_CALUDE_extra_lives_in_first_level_l3719_371960

theorem extra_lives_in_first_level :
  let initial_lives : ℕ := 2
  let lives_gained_second_level : ℕ := 11
  let total_lives_after_second_level : ℕ := 19
  let extra_lives_first_level : ℕ := total_lives_after_second_level - lives_gained_second_level - initial_lives
  extra_lives_first_level = 6 :=
by sorry

end NUMINAMATH_CALUDE_extra_lives_in_first_level_l3719_371960


namespace NUMINAMATH_CALUDE_ellen_calorie_instruction_l3719_371909

/-- The total number of calories Ellen was instructed to eat in a day -/
def total_calories : ℕ := 2200

/-- The number of calories Ellen ate for breakfast -/
def breakfast_calories : ℕ := 353

/-- The number of calories Ellen had for lunch -/
def lunch_calories : ℕ := 885

/-- The number of calories Ellen had for afternoon snack -/
def snack_calories : ℕ := 130

/-- The number of calories Ellen has left for dinner -/
def dinner_calories : ℕ := 832

/-- Theorem stating that the total calories Ellen was instructed to eat
    is equal to the sum of all meals and snacks -/
theorem ellen_calorie_instruction :
  total_calories = breakfast_calories + lunch_calories + snack_calories + dinner_calories :=
by sorry

end NUMINAMATH_CALUDE_ellen_calorie_instruction_l3719_371909


namespace NUMINAMATH_CALUDE_sphere_in_cone_angle_l3719_371982

/-- Given a sphere inscribed in a cone, if the circle of tangency divides the surface
    of the sphere in the ratio of 1:4, then the angle between the generatrix of the cone
    and its base plane is arccos(3/5). -/
theorem sphere_in_cone_angle (R : ℝ) (α : ℝ) :
  R > 0 →  -- Radius is positive
  (2 * π * R^2 * (1 - Real.cos α)) / (4 * π * R^2) = 1/5 →  -- Surface area ratio condition
  α = Real.arccos (3/5) :=
by sorry

end NUMINAMATH_CALUDE_sphere_in_cone_angle_l3719_371982


namespace NUMINAMATH_CALUDE_min_rooks_to_attack_all_white_cells_l3719_371954

/-- Represents a cell on the chessboard -/
structure Cell :=
  (row : Fin 9)
  (col : Fin 9)

/-- Determines if a cell is white based on its position -/
def isWhite (c : Cell) : Bool :=
  (c.row.val + c.col.val) % 2 = 0

/-- Represents a rook's position on the board -/
structure Rook :=
  (position : Cell)

/-- Determines if a cell is under attack by a rook -/
def isUnderAttack (c : Cell) (r : Rook) : Bool :=
  c.row = r.position.row ∨ c.col = r.position.col

/-- The main theorem stating the minimum number of rooks required -/
theorem min_rooks_to_attack_all_white_cells :
  ∃ (rooks : List Rook),
    rooks.length = 5 ∧
    (∀ c : Cell, isWhite c → ∃ r ∈ rooks, isUnderAttack c r) ∧
    (∀ (rooks' : List Rook),
      rooks'.length < 5 →
      ¬(∀ c : Cell, isWhite c → ∃ r ∈ rooks', isUnderAttack c r)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_rooks_to_attack_all_white_cells_l3719_371954


namespace NUMINAMATH_CALUDE_trajectory_is_square_l3719_371949

-- Define the set of points (x, y) satisfying |x| + |y| = 1
def trajectory : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1| + |p.2| = 1}

-- Define a square in the plane
def isSquare (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), S = {p : ℝ × ℝ | max (|p.1 - a|) (|p.2 - b|) = 1/2}

-- Theorem statement
theorem trajectory_is_square : isSquare trajectory := by sorry

end NUMINAMATH_CALUDE_trajectory_is_square_l3719_371949


namespace NUMINAMATH_CALUDE_expression_evaluation_l3719_371993

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := 1
  (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3719_371993


namespace NUMINAMATH_CALUDE_ahmed_age_l3719_371934

theorem ahmed_age (fouad_age : ℕ) (ahmed_age : ℕ) (h : fouad_age = 26) :
  (fouad_age + 4 = 2 * ahmed_age) → ahmed_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_ahmed_age_l3719_371934


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3719_371985

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x, (x ≥ b - 1 ∧ x < a / 2) ↔ (-3 ≤ x ∧ x < 3 / 2)) → 
  a * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3719_371985


namespace NUMINAMATH_CALUDE_edge_tangent_sphere_radius_l3719_371929

/-- Represents a regular tetrahedron with associated spheres -/
structure RegularTetrahedron where
  r : ℝ  -- radius of inscribed sphere
  R : ℝ  -- radius of circumscribed sphere
  ρ : ℝ  -- radius of edge-tangent sphere
  h_r_pos : 0 < r
  h_R_pos : 0 < R
  h_ρ_pos : 0 < ρ
  h_R_r : R = 3 * r

/-- 
The radius of the sphere tangent to all edges of a regular tetrahedron 
is the geometric mean of the radii of its inscribed and circumscribed spheres 
-/
theorem edge_tangent_sphere_radius (t : RegularTetrahedron) : t.ρ^2 = t.R * t.r := by
  sorry

end NUMINAMATH_CALUDE_edge_tangent_sphere_radius_l3719_371929


namespace NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l3719_371980

theorem cylinder_height_in_hemisphere (r c h : ℝ) : 
  r > 0 → c > 0 → h > 0 →
  r = 8 → c = 3 →
  h^2 + c^2 = r^2 →
  h = Real.sqrt 55 := by
sorry

end NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l3719_371980


namespace NUMINAMATH_CALUDE_units_digit_of_M_M12_is_1_l3719_371950

/-- Modified Lucas sequence -/
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | (n + 2) => M (n + 1) + M n

/-- The 12th term of the Modified Lucas sequence -/
def M12 : ℕ := M 12

/-- Theorem stating that the units digit of M_{M₁₂} is 1 -/
theorem units_digit_of_M_M12_is_1 : M M12 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_M_M12_is_1_l3719_371950


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3719_371944

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) → x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3719_371944


namespace NUMINAMATH_CALUDE_work_completion_days_l3719_371986

/-- The number of days B takes to complete the work alone -/
def B : ℝ := 12

/-- The number of days A and B work together -/
def together_days : ℝ := 3

/-- The number of days B works alone after A leaves -/
def B_alone_days : ℝ := 3

/-- The number of days A takes to complete the work alone -/
def A : ℝ := 6

theorem work_completion_days : 
  together_days * (1 / A + 1 / B) + B_alone_days * (1 / B) = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_days_l3719_371986


namespace NUMINAMATH_CALUDE_distance_to_rock_mist_mountains_value_l3719_371940

/-- The distance from the city to Rock Mist Mountains, including detours -/
def distance_to_rock_mist_mountains : ℝ :=
  let sky_falls_distance : ℝ := 8
  let rock_mist_multiplier : ℝ := 50
  let break_point_percentage : ℝ := 0.3
  let cloudy_heights_percentage : ℝ := 0.6
  let thunder_pass_detour : ℝ := 25
  
  sky_falls_distance * rock_mist_multiplier + thunder_pass_detour

/-- Theorem stating the distance to Rock Mist Mountains -/
theorem distance_to_rock_mist_mountains_value :
  distance_to_rock_mist_mountains = 425 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_rock_mist_mountains_value_l3719_371940


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3719_371922

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 2) :
  (1/x + 1/y + 1/z) ≥ 4.5 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 ∧ 1/x + 1/y + 1/z = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3719_371922


namespace NUMINAMATH_CALUDE_rectangle_horizontal_length_l3719_371900

/-- Proves that a rectangle with perimeter 54 cm and horizontal length 3 cm longer than vertical length has a horizontal length of 15 cm -/
theorem rectangle_horizontal_length : 
  ∀ (v h : ℝ), 
  (2 * v + 2 * h = 54) →  -- perimeter is 54 cm
  (h = v + 3) →           -- horizontal length is 3 cm longer than vertical length
  h = 15 := by            -- horizontal length is 15 cm
sorry

end NUMINAMATH_CALUDE_rectangle_horizontal_length_l3719_371900


namespace NUMINAMATH_CALUDE_sqrt_expression_l3719_371908

theorem sqrt_expression : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_l3719_371908


namespace NUMINAMATH_CALUDE_crow_tree_problem_l3719_371995

theorem crow_tree_problem (x y : ℕ) : 
  (3 * y + 5 = x) → (5 * (y - 1) = x) → (x = 20 ∧ y = 5) := by
  sorry

end NUMINAMATH_CALUDE_crow_tree_problem_l3719_371995


namespace NUMINAMATH_CALUDE_solution_set_implies_b_range_l3719_371955

/-- The solution set of the inequality |3x-b| < 4 -/
def SolutionSet (b : ℝ) : Set ℝ :=
  {x : ℝ | |3*x - b| < 4}

/-- The set of integers 1, 2, and 3 -/
def IntegerSet : Set ℝ := {1, 2, 3}

/-- Theorem stating that if the solution set of |3x-b| < 4 is exactly {1, 2, 3}, then 5 < b < 7 -/
theorem solution_set_implies_b_range :
  ∀ b : ℝ, SolutionSet b = IntegerSet → 5 < b ∧ b < 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_b_range_l3719_371955


namespace NUMINAMATH_CALUDE_daily_lottery_expected_profit_l3719_371976

/-- The expected profit from purchasing one "Daily Lottery" ticket -/
def expected_profit : ℝ := -0.9

/-- The price of one lottery ticket -/
def ticket_price : ℝ := 2

/-- The probability of winning the first prize -/
def first_prize_prob : ℝ := 0.001

/-- The probability of winning the second prize -/
def second_prize_prob : ℝ := 0.1

/-- The amount of the first prize -/
def first_prize_amount : ℝ := 100

/-- The amount of the second prize -/
def second_prize_amount : ℝ := 10

theorem daily_lottery_expected_profit :
  expected_profit = 
    first_prize_prob * first_prize_amount + 
    second_prize_prob * second_prize_amount - 
    ticket_price := by
  sorry

end NUMINAMATH_CALUDE_daily_lottery_expected_profit_l3719_371976
