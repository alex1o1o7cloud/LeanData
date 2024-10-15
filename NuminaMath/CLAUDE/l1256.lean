import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l1256_125668

/-- The range of k for which the quadratic equation kx^2 - 4x - 2 = 0 has two distinct real roots -/
theorem quadratic_distinct_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 4 * x₁ - 2 = 0 ∧ k * x₂^2 - 4 * x₂ - 2 = 0) ↔ 
  (k > -2 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l1256_125668


namespace NUMINAMATH_CALUDE_brody_battery_usage_l1256_125665

/-- Represents the battery life of Brody's calculator -/
def BatteryLife : Type := ℚ

/-- The total battery life of the calculator when fully charged (in hours) -/
def full_battery : ℚ := 60

/-- The duration of Brody's exam (in hours) -/
def exam_duration : ℚ := 2

/-- The remaining battery life after the exam (in hours) -/
def remaining_battery : ℚ := 13

/-- The fraction of battery Brody has used up -/
def battery_used_fraction : ℚ := 3/4

/-- Theorem stating that the fraction of battery Brody has used up is 3/4 -/
theorem brody_battery_usage :
  (full_battery - (remaining_battery + exam_duration)) / full_battery = battery_used_fraction := by
  sorry

end NUMINAMATH_CALUDE_brody_battery_usage_l1256_125665


namespace NUMINAMATH_CALUDE_polynomial_property_l1256_125676

def Q (x d e f : ℝ) : ℝ := 3 * x^4 + d * x^3 + e * x^2 + f * x - 27

theorem polynomial_property (d e f : ℝ) :
  (∀ x₁ x₂ x₃ x₄ : ℝ, Q x₁ d e f = 0 ∧ Q x₂ d e f = 0 ∧ Q x₃ d e f = 0 ∧ Q x₄ d e f = 0 →
    x₁ + x₂ + x₃ + x₄ = x₁*x₂ + x₁*x₃ + x₁*x₄ + x₂*x₃ + x₂*x₄ + x₃*x₄) ∧
  (x₁ + x₂ + x₃ + x₄ = 3 + d + e + f - 27) ∧
  e = 0 →
  f = -12 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l1256_125676


namespace NUMINAMATH_CALUDE_line_equation_l1256_125682

-- Define the circle F
def circle_F (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line passing through a point
def line_through_point (k m : ℝ) (x y : ℝ) : Prop := y = k*(x - m)

-- Define the arithmetic sequence condition
def arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

-- Main theorem
theorem line_equation (P M N Q : ℝ × ℝ) :
  let (xp, yp) := P
  let (xm, ym) := M
  let (xn, yn) := N
  let (xq, yq) := Q
  (∃ k : ℝ, 
    (∀ x y, line_through_point k 1 x y → (circle_F x y ∨ parabola_C x y)) ∧
    line_through_point k 1 xp yp ∧
    line_through_point k 1 xm ym ∧
    line_through_point k 1 xn yn ∧
    line_through_point k 1 xq yq ∧
    arithmetic_sequence (Real.sqrt ((xp-1)^2 + yp^2)) (Real.sqrt ((xm-1)^2 + ym^2)) (Real.sqrt ((xn-1)^2 + yn^2)) ∧
    arithmetic_sequence (Real.sqrt ((xm-1)^2 + ym^2)) (Real.sqrt ((xn-1)^2 + yn^2)) (Real.sqrt ((xq-1)^2 + yq^2))) →
  (k = Real.sqrt 2 ∨ k = -Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1256_125682


namespace NUMINAMATH_CALUDE_gcd_10_factorial_12_factorial_l1256_125684

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_10_factorial_12_factorial : Nat.gcd (factorial 10) (factorial 12) = factorial 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10_factorial_12_factorial_l1256_125684


namespace NUMINAMATH_CALUDE_probability_of_one_each_l1256_125646

def shirts : ℕ := 6
def shorts : ℕ := 8
def socks : ℕ := 7
def hats : ℕ := 3

def total_items : ℕ := shirts + shorts + socks + hats

theorem probability_of_one_each : 
  (shirts.choose 1 * shorts.choose 1 * socks.choose 1 * hats.choose 1) / total_items.choose 4 = 72 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_each_l1256_125646


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1256_125674

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  -3 * x^2 + 6 * x * y - 3 * y^2 = -3 * (x - y)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (m n : ℝ) :
  8 * m^2 * (m + n) - 2 * (m + n) = 2 * (m + n) * (2 * m + 1) * (2 * m - 1) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1256_125674


namespace NUMINAMATH_CALUDE_intersection_circles_properties_l1256_125642

/-- Given two circles O₁ and O₂ with equations x² + y² - 2x = 0 and x² + y² + 2x - 4y = 0 respectively,
    prove that their intersection points A and B satisfy:
    1. The line AB has equation x - y = 0
    2. The perpendicular bisector of AB has equation x + y - 1 = 0 -/
theorem intersection_circles_properties (x y : ℝ) :
  let O₁ := {(x, y) | x^2 + y^2 - 2*x = 0}
  let O₂ := {(x, y) | x^2 + y^2 + 2*x - 4*y = 0}
  let A := (x₀, y₀)
  let B := (x₁, y₁)
  ∀ x₀ y₀ x₁ y₁,
    (x₀, y₀) ∈ O₁ ∧ (x₀, y₀) ∈ O₂ ∧
    (x₁, y₁) ∈ O₁ ∧ (x₁, y₁) ∈ O₂ ∧
    (x₀, y₀) ≠ (x₁, y₁) →
    (x - y = 0 ↔ ∃ t, x = (1-t)*x₀ + t*x₁ ∧ y = (1-t)*y₀ + t*y₁) ∧
    (x + y - 1 = 0 ↔ (x - 1)^2 + y^2 = (x + 1)^2 + (y - 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_circles_properties_l1256_125642


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l1256_125625

-- Define the condition for the expression to be meaningful
def is_meaningful (x : ℝ) : Prop := x > 3

-- Theorem statement
theorem meaningful_expression_range :
  ∀ x : ℝ, is_meaningful x ↔ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l1256_125625


namespace NUMINAMATH_CALUDE_intersection_point_unique_l1256_125696

/-- The line equation -/
def line_equation (x y z : ℝ) : Prop :=
  (x - 3) / 2 = (y + 1) / 3 ∧ (x - 3) / 2 = (z + 3) / 2

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  3 * x + 4 * y + 7 * z - 16 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (5, 2, -1)

theorem intersection_point_unique :
  ∃! p : ℝ × ℝ × ℝ, 
    line_equation p.1 p.2.1 p.2.2 ∧ 
    plane_equation p.1 p.2.1 p.2.2 ∧
    p = intersection_point :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l1256_125696


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l1256_125685

theorem smallest_sum_B_plus_c : ∃ (B c : ℕ), 
  (0 ≤ B ∧ B ≤ 4) ∧ 
  (c > 6) ∧
  (31 * B = 4 * c + 4) ∧
  (∀ (B' c' : ℕ), (0 ≤ B' ∧ B' ≤ 4) ∧ (c' > 6) ∧ (31 * B' = 4 * c' + 4) → B + c ≤ B' + c') ∧
  B + c = 34 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l1256_125685


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1256_125689

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 8) * (x - 6) = -62 + k * x) ↔ 
  (k = -10 + 12 * Real.sqrt 1.5 ∨ k = -10 - 12 * Real.sqrt 1.5) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1256_125689


namespace NUMINAMATH_CALUDE_stratified_sampling_female_students_l1256_125633

/-- Calculates the number of female students selected in a stratified sampling -/
def female_students_selected (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ) : ℕ :=
  (sample_size * female_students) / total_students

/-- Theorem: In a school with 2000 total students and 800 female students,
    a stratified sampling of 50 students will select 20 female students -/
theorem stratified_sampling_female_students :
  female_students_selected 2000 800 50 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_students_l1256_125633


namespace NUMINAMATH_CALUDE_student_b_visited_a_l1256_125679

structure Student :=
  (visitedA : Bool)
  (visitedB : Bool)
  (visitedC : Bool)

def citiesVisited (s : Student) : Nat :=
  (if s.visitedA then 1 else 0) +
  (if s.visitedB then 1 else 0) +
  (if s.visitedC then 1 else 0)

theorem student_b_visited_a (studentA studentB studentC : Student) :
  citiesVisited studentA > citiesVisited studentB →
  studentA.visitedB = false →
  studentB.visitedC = false →
  (studentA.visitedA = true ∧ studentB.visitedA = true ∧ studentC.visitedA = true) ∨
  (studentA.visitedB = true ∧ studentB.visitedB = true ∧ studentC.visitedB = true) ∨
  (studentA.visitedC = true ∧ studentB.visitedC = true ∧ studentC.visitedC = true) →
  studentB.visitedA = true :=
by
  sorry

end NUMINAMATH_CALUDE_student_b_visited_a_l1256_125679


namespace NUMINAMATH_CALUDE_cars_meet_time_l1256_125635

-- Define the highway length
def highway_length : ℝ := 175

-- Define the speeds of the two cars
def speed_car1 : ℝ := 25
def speed_car2 : ℝ := 45

-- Define the meeting time
def meeting_time : ℝ := 2.5

-- Theorem statement
theorem cars_meet_time :
  speed_car1 * meeting_time + speed_car2 * meeting_time = highway_length :=
by sorry


end NUMINAMATH_CALUDE_cars_meet_time_l1256_125635


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1256_125662

theorem quadratic_root_relation (a b : ℝ) : 
  (∃ r s : ℝ, (2 * r^2 - 3 * r - 8 = 0) ∧ 
              (2 * s^2 - 3 * s - 8 = 0) ∧ 
              ((r + 3)^2 + a * (r + 3) + b = 0) ∧ 
              ((s + 3)^2 + a * (s + 3) + b = 0)) →
  b = 9.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1256_125662


namespace NUMINAMATH_CALUDE_alyssa_puppies_l1256_125611

/-- The number of puppies Alyssa has after breeding and giving some away -/
def remaining_puppies (initial : ℕ) (puppies_per_puppy : ℕ) (given_away : ℕ) : ℕ :=
  initial + initial * puppies_per_puppy - given_away

/-- Theorem stating that Alyssa has 20 puppies left -/
theorem alyssa_puppies : remaining_puppies 7 4 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_puppies_l1256_125611


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l1256_125661

theorem largest_solution_of_equation :
  let f (x : ℝ) := x / 7 + 3 / (7 * x)
  ∃ (max_x : ℝ), f max_x = 1 ∧ max_x = (7 + Real.sqrt 37) / 2 ∧
    ∀ (y : ℝ), f y = 1 → y ≤ max_x :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l1256_125661


namespace NUMINAMATH_CALUDE_orange_count_orange_count_problem_l1256_125650

theorem orange_count (initial_apples : ℕ) (removed_oranges : ℕ) 
  (apple_percentage : ℚ) (initial_oranges : ℕ) : Prop :=
  initial_apples = 14 →
  removed_oranges = 20 →
  apple_percentage = 7/10 →
  initial_apples / (initial_apples + initial_oranges - removed_oranges) = apple_percentage →
  initial_oranges = 26

/-- The theorem states that given the conditions from the problem,
    the initial number of oranges in the box is 26. -/
theorem orange_count_problem : 
  ∃ (initial_oranges : ℕ), orange_count 14 20 (7/10) initial_oranges :=
sorry

end NUMINAMATH_CALUDE_orange_count_orange_count_problem_l1256_125650


namespace NUMINAMATH_CALUDE_min_distance_circle_line_l1256_125603

/-- The minimum distance between a circle and a line --/
theorem min_distance_circle_line :
  let circle := {p : ℝ × ℝ | (p.1 + 2)^2 + p.2^2 = 4}
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  (∀ p ∈ circle, ∀ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ 3 * Real.sqrt 2 - 2) ∧
  (∃ p ∈ circle, ∃ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 3 * Real.sqrt 2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_line_l1256_125603


namespace NUMINAMATH_CALUDE_equal_distribution_of_treats_l1256_125688

def cookies : ℕ := 20
def cupcakes : ℕ := 25
def brownies : ℕ := 35
def students : ℕ := 20

theorem equal_distribution_of_treats :
  (cookies + cupcakes + brownies) / students = 4 :=
by sorry

end NUMINAMATH_CALUDE_equal_distribution_of_treats_l1256_125688


namespace NUMINAMATH_CALUDE_constant_b_value_l1256_125660

theorem constant_b_value (x y : ℝ) (b : ℝ) 
  (h1 : (7 * x + b * y) / (x - 2 * y) = 29)
  (h2 : x / (2 * y) = 3 / 2) : 
  b = 8 := by
  sorry

end NUMINAMATH_CALUDE_constant_b_value_l1256_125660


namespace NUMINAMATH_CALUDE_carol_cupcakes_l1256_125687

/-- The number of cupcakes Carol has after initially making some, selling some, and making more. -/
def total_cupcakes (initial : ℕ) (sold : ℕ) (made_more : ℕ) : ℕ :=
  initial - sold + made_more

/-- Theorem stating that Carol has 49 cupcakes in total -/
theorem carol_cupcakes : total_cupcakes 30 9 28 = 49 := by
  sorry

end NUMINAMATH_CALUDE_carol_cupcakes_l1256_125687


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1256_125638

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 2 → a^2 > 2*a) ∧ 
  (∃ a, a ≤ 2 ∧ a^2 > 2*a) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1256_125638


namespace NUMINAMATH_CALUDE_sqrt_a_minus_one_is_rational_square_l1256_125623

theorem sqrt_a_minus_one_is_rational_square (a b : ℚ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + 4*a^2*b = 4*a^2 + b^4) :
  ∃ q : ℚ, Real.sqrt a - 1 = q^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_minus_one_is_rational_square_l1256_125623


namespace NUMINAMATH_CALUDE_number_problem_l1256_125631

theorem number_problem (n : ℝ) : (40 / 100) * (3 / 5) * n = 36 → n = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1256_125631


namespace NUMINAMATH_CALUDE_first_group_size_correct_l1256_125683

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 39

/-- The number of days the first group works -/
def first_group_days : ℕ := 24

/-- The number of hours per day the first group works -/
def first_group_hours_per_day : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group works -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours_per_day : ℕ := 6

/-- Theorem stating that the first group size is correct given the conditions -/
theorem first_group_size_correct :
  first_group_size * first_group_days * first_group_hours_per_day =
  second_group_size * second_group_days * second_group_hours_per_day :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_correct_l1256_125683


namespace NUMINAMATH_CALUDE_adult_dog_cost_l1256_125602

/-- The cost to prepare animals for adoption -/
structure AdoptionCost where
  cat : ℕ → ℕ     -- Cost for cats
  dog : ℕ → ℕ     -- Cost for adult dogs
  puppy : ℕ → ℕ   -- Cost for puppies

/-- The theorem stating the cost for each adult dog -/
theorem adult_dog_cost (c : AdoptionCost) 
  (h1 : c.cat 1 = 50)
  (h2 : c.puppy 1 = 150)
  (h3 : c.cat 2 + c.dog 3 + c.puppy 2 = 700) :
  c.dog 1 = 100 := by
  sorry


end NUMINAMATH_CALUDE_adult_dog_cost_l1256_125602


namespace NUMINAMATH_CALUDE_knights_in_company_l1256_125615

def is_knight (person : Nat) : Prop := sorry

def statement (n : Nat) : Prop :=
  ∃ k, k ∣ n ∧ (∀ p, is_knight p ↔ p ≤ k)

theorem knights_in_company :
  ∀ k : Nat, k ≤ 39 →
  (∀ n : Nat, n ≤ 39 → (∃ p, is_knight p ↔ statement n)) →
  (k = 0 ∨ k = 6) :=
sorry

end NUMINAMATH_CALUDE_knights_in_company_l1256_125615


namespace NUMINAMATH_CALUDE_specific_journey_distance_l1256_125644

/-- A journey with two parts at different speeds -/
structure Journey where
  total_time : ℝ
  first_part_time : ℝ
  first_part_speed : ℝ
  second_part_speed : ℝ
  (first_part_time_valid : first_part_time > 0 ∧ first_part_time < total_time)
  (speeds_positive : first_part_speed > 0 ∧ second_part_speed > 0)

/-- Calculate the total distance of a journey -/
def total_distance (j : Journey) : ℝ :=
  j.first_part_speed * j.first_part_time + 
  j.second_part_speed * (j.total_time - j.first_part_time)

/-- The specific journey described in the problem -/
def specific_journey : Journey where
  total_time := 8
  first_part_time := 4
  first_part_speed := 4
  second_part_speed := 2
  first_part_time_valid := by sorry
  speeds_positive := by sorry

/-- Theorem stating that the total distance of the specific journey is 24 km -/
theorem specific_journey_distance : 
  total_distance specific_journey = 24 := by sorry

end NUMINAMATH_CALUDE_specific_journey_distance_l1256_125644


namespace NUMINAMATH_CALUDE_m_range_l1256_125681

/-- The function f(x) defined as x^2 + mx - 1 --/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

/-- The theorem stating the range of m given the conditions --/
theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 1), f m x < 0) → 
  m ∈ Set.Ioo (-Real.sqrt 2 / 2) 0 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1256_125681


namespace NUMINAMATH_CALUDE_carl_kevin_historical_difference_l1256_125641

/-- A stamp collector's collection --/
structure StampCollection where
  total : ℕ
  international : ℕ
  historical : ℕ
  animal : ℕ

/-- Carl's stamp collection --/
def carl : StampCollection :=
  { total := 125
  , international := 45
  , historical := 60
  , animal := 20 }

/-- Kevin's stamp collection --/
def kevin : StampCollection :=
  { total := 95
  , international := 30
  , historical := 50
  , animal := 15 }

/-- The difference in historical stamps between two collections --/
def historicalStampDifference (c1 c2 : StampCollection) : ℕ :=
  c1.historical - c2.historical

/-- Theorem stating the difference in historical stamps between Carl and Kevin --/
theorem carl_kevin_historical_difference :
  historicalStampDifference carl kevin = 10 := by
  sorry

end NUMINAMATH_CALUDE_carl_kevin_historical_difference_l1256_125641


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1256_125669

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 986 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  17 ∣ n ∧
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1256_125669


namespace NUMINAMATH_CALUDE_right_triangle_sin_A_l1256_125639

theorem right_triangle_sin_A (A B C : Real) (AB BC : Real) :
  -- Right triangle ABC with ∠BAC = 90°
  A + B + C = 180 →
  A = 90 →
  -- Side lengths
  AB = 15 →
  BC = 20 →
  -- Definition of sin A in a right triangle
  Real.sin A = Real.sqrt 7 / 4 := by sorry

end NUMINAMATH_CALUDE_right_triangle_sin_A_l1256_125639


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1256_125698

/-- Given a hyperbola with equation x²/64 - y²/36 = 1, 
    its asymptotes have equations y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), 
  (x^2 / 64 - y^2 / 36 = 1) →
  (∃ (k : ℝ), k = 3/4 ∧ (y = k*x ∨ y = -k*x)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1256_125698


namespace NUMINAMATH_CALUDE_square_root_of_square_l1256_125628

theorem square_root_of_square (x : ℝ) (h : x = 25) : Real.sqrt (x ^ 2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_square_l1256_125628


namespace NUMINAMATH_CALUDE_tan_product_undefined_l1256_125695

theorem tan_product_undefined : 
  ¬∃ (x : ℝ), Real.tan (π / 6) * Real.tan (π / 3) * Real.tan (π / 2) = x :=
by sorry

end NUMINAMATH_CALUDE_tan_product_undefined_l1256_125695


namespace NUMINAMATH_CALUDE_faucet_fill_time_l1256_125697

-- Define the constants from the problem
def tub_size_1 : ℝ := 200  -- Size of the first tub in gallons
def tub_size_2 : ℝ := 50   -- Size of the second tub in gallons
def faucets_1 : ℝ := 4     -- Number of faucets for the first tub
def faucets_2 : ℝ := 8     -- Number of faucets for the second tub
def time_1 : ℝ := 12       -- Time to fill the first tub in minutes

-- Define the theorem
theorem faucet_fill_time :
  ∃ (rate : ℝ),
    (rate * faucets_1 * time_1 = tub_size_1) ∧
    (rate * faucets_2 * (90 / 60) = tub_size_2) :=
by sorry

end NUMINAMATH_CALUDE_faucet_fill_time_l1256_125697


namespace NUMINAMATH_CALUDE_f_eq_g_l1256_125637

-- Define the two functions
def f (x : ℝ) : ℝ := x^2 - 4*x + 6
def g (x : ℝ) : ℝ := (x - 2)^2 + 2

-- Theorem stating that f and g are equal for all real x
theorem f_eq_g : ∀ x : ℝ, f x = g x := by
  sorry

end NUMINAMATH_CALUDE_f_eq_g_l1256_125637


namespace NUMINAMATH_CALUDE_diagonal_grid_4x3_triangles_l1256_125675

/-- Represents a rectangular grid with diagonals -/
structure DiagonalGrid :=
  (columns : Nat)
  (rows : Nat)

/-- Calculates the number of triangles in a diagonal grid -/
def count_triangles (grid : DiagonalGrid) : Nat :=
  let small_triangles := 2 * grid.columns * grid.rows
  let larger_rectangles := (grid.columns - 1) * (grid.rows - 1)
  let larger_triangles := 8 * larger_rectangles
  let additional_triangles := 6  -- Simplified count for larger configurations
  small_triangles + larger_triangles + additional_triangles

/-- Theorem stating that a 4x3 diagonal grid contains 78 triangles -/
theorem diagonal_grid_4x3_triangles :
  ∃ (grid : DiagonalGrid), grid.columns = 4 ∧ grid.rows = 3 ∧ count_triangles grid = 78 :=
by
  sorry

#eval count_triangles ⟨4, 3⟩

end NUMINAMATH_CALUDE_diagonal_grid_4x3_triangles_l1256_125675


namespace NUMINAMATH_CALUDE_inscribed_polygon_perimeter_l1256_125609

/-- 
Given two regular polygons with perimeters a and b circumscribed around a circle, 
and a third regular polygon inscribed in the same circle, where the second and 
third polygons have twice as many sides as the first, the perimeter of the third 
polygon is equal to b√(a / (2a - b)).
-/
theorem inscribed_polygon_perimeter 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_circumscribed : ∃ (r : ℝ) (n : ℕ), a = 2 * n * r * Real.tan (2 * π / n) ∧ 
                                        b = 4 * n * r * Real.tan (π / n))
  (h_inscribed : ∃ (x : ℝ), x = b * Real.cos (π / n)) :
  ∃ (x : ℝ), x = b * Real.sqrt (a / (2 * a - b)) :=
sorry

end NUMINAMATH_CALUDE_inscribed_polygon_perimeter_l1256_125609


namespace NUMINAMATH_CALUDE_intersection_complement_eq_l1256_125634

def M : Set ℝ := {-1, 0, 1, 3}
def N : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement_eq : M ∩ (Set.univ \ N) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_eq_l1256_125634


namespace NUMINAMATH_CALUDE_smallest_bound_for_cubic_inequality_l1256_125614

theorem smallest_bound_for_cubic_inequality :
  ∃ (M : ℝ), (∀ (a b c : ℝ),
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2) ∧
  (∀ (M' : ℝ), (∀ (a b c : ℝ),
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M' * (a^2 + b^2 + c^2)^2) →
    M ≤ M') ∧
  M = (9 * Real.sqrt 2) / 32 :=
by sorry

end NUMINAMATH_CALUDE_smallest_bound_for_cubic_inequality_l1256_125614


namespace NUMINAMATH_CALUDE_power_equality_l1256_125629

theorem power_equality (x y : ℕ) :
  2 * (3^8)^2 * (2^3)^2 * 3 = 2^x * 3^y → x = 7 ∧ y = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1256_125629


namespace NUMINAMATH_CALUDE_inequality_proof_l1256_125690

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x + y - 1)^2 / z + (y + z - 1)^2 / x + (z + x - 1)^2 / y ≥ x + y + z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1256_125690


namespace NUMINAMATH_CALUDE_zeros_product_greater_than_e_squared_l1256_125643

theorem zeros_product_greater_than_e_squared (k : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
  (Real.log x₁ = k * x₁) → (Real.log x₂ = k * x₂) →
  x₁ * x₂ > Real.exp 2 := by
sorry

end NUMINAMATH_CALUDE_zeros_product_greater_than_e_squared_l1256_125643


namespace NUMINAMATH_CALUDE_complex_modulus_product_l1256_125678

/-- Given a complex number with modulus √2018, prove that its product with its conjugate is 2018 -/
theorem complex_modulus_product (a b : ℝ) : 
  (Complex.abs (Complex.mk a b))^2 = 2018 → (a + Complex.I * b) * (a - Complex.I * b) = 2018 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l1256_125678


namespace NUMINAMATH_CALUDE_school_population_l1256_125647

theorem school_population (b g t : ℕ) : 
  b = 3 * g → g = 9 * t → b + g + t = (37 * b) / 27 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l1256_125647


namespace NUMINAMATH_CALUDE_gcd_problem_l1256_125667

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 1116 * k) :
  Int.gcd (b^2 + 11*b + 36) (b + 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1256_125667


namespace NUMINAMATH_CALUDE_solve_for_q_l1256_125618

theorem solve_for_q (x y q : ℚ) 
  (h1 : (7 : ℚ) / 8 = x / 96)
  (h2 : (7 : ℚ) / 8 = (x + y) / 104)
  (h3 : (7 : ℚ) / 8 = (q - y) / 144) :
  q = 133 := by sorry

end NUMINAMATH_CALUDE_solve_for_q_l1256_125618


namespace NUMINAMATH_CALUDE_journey_time_ratio_l1256_125664

/-- Proves the ratio of new time to original time for a given journey -/
theorem journey_time_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 288 ∧ original_time = 6 ∧ new_speed = 32 →
  (distance / new_speed) / original_time = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_ratio_l1256_125664


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1256_125666

theorem complex_equation_solution (z : ℂ) :
  (2 - Complex.I) * z = Complex.I ^ 2022 →
  z = -2/5 - (1/5) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1256_125666


namespace NUMINAMATH_CALUDE_log2_7_value_l1256_125651

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Given conditions
variable (m n : ℝ)
variable (h1 : lg 5 = m)
variable (h2 : lg 7 = n)

-- Theorem to prove
theorem log2_7_value : Real.log 7 / Real.log 2 = n / (1 - m) := by
  sorry

end NUMINAMATH_CALUDE_log2_7_value_l1256_125651


namespace NUMINAMATH_CALUDE_candy_spent_is_10_l1256_125608

-- Define the total amount spent
def total_spent : ℝ := 150

-- Define the fractions spent on each category
def fruits_veg_fraction : ℚ := 1/2
def meat_fraction : ℚ := 1/3
def bakery_fraction : ℚ := 1/10

-- Define the theorem
theorem candy_spent_is_10 :
  let remaining_fraction : ℚ := 1 - (fruits_veg_fraction + meat_fraction + bakery_fraction)
  (remaining_fraction : ℝ) * total_spent = 10 := by sorry

end NUMINAMATH_CALUDE_candy_spent_is_10_l1256_125608


namespace NUMINAMATH_CALUDE_eight_person_lineup_l1256_125636

theorem eight_person_lineup : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_eight_person_lineup_l1256_125636


namespace NUMINAMATH_CALUDE_function_value_at_five_l1256_125617

theorem function_value_at_five (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 2 * x^2 - 1) : 
  f 5 = 13/3 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_five_l1256_125617


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1256_125606

theorem absolute_value_inequality (x : ℝ) :
  abs (x - 2) + abs (x + 3) < 8 ↔ -9/2 < x ∧ x < 7/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1256_125606


namespace NUMINAMATH_CALUDE_number_of_boys_l1256_125653

theorem number_of_boys (girls : ℕ) (groups : ℕ) (members_per_group : ℕ) 
  (h1 : girls = 12)
  (h2 : groups = 7)
  (h3 : members_per_group = 3) :
  groups * members_per_group - girls = 9 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l1256_125653


namespace NUMINAMATH_CALUDE_part1_part2_l1256_125673

-- Define the operation F
def F (a b x y : ℝ) : ℝ := a * x + b * y

-- Theorem for part 1
theorem part1 (a b : ℝ) : 
  F a b (-1) 3 = 2 ∧ F a b 1 (-2) = 8 → a = 28 ∧ b = 10 := by sorry

-- Theorem for part 2
theorem part2 (a b : ℝ) :
  b ≥ 0 ∧ F a b 2 1 = 5 → a ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1256_125673


namespace NUMINAMATH_CALUDE_smallest_cube_ending_392_l1256_125630

theorem smallest_cube_ending_392 :
  ∃ (n : ℕ), n > 0 ∧ n^3 ≡ 392 [ZMOD 1000] ∧ ∀ (m : ℕ), m > 0 ∧ m^3 ≡ 392 [ZMOD 1000] → n ≤ m :=
by
  use 48
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_392_l1256_125630


namespace NUMINAMATH_CALUDE_a_left_after_ten_days_l1256_125620

/-- The number of days it takes A to complete the work -/
def days_A : ℝ := 30

/-- The number of days it takes B to complete the work -/
def days_B : ℝ := 30

/-- The number of days B worked after A left -/
def days_B_worked : ℝ := 10

/-- The number of days C worked to finish the work -/
def days_C_worked : ℝ := 10

/-- The number of days it takes C to complete the whole work -/
def days_C : ℝ := 29.999999999999996

/-- The theorem stating that A left the work after 10 days -/
theorem a_left_after_ten_days :
  ∃ (x : ℝ),
    x > 0 ∧
    x / days_A + days_B_worked / days_B + days_C_worked / days_C = 1 ∧
    x = 10 := by
  sorry

end NUMINAMATH_CALUDE_a_left_after_ten_days_l1256_125620


namespace NUMINAMATH_CALUDE_twelve_boys_handshakes_l1256_125622

/-- The number of handshakes when n boys each shake hands exactly once with every other boy -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 12 boys, where each boy shakes hands exactly once with each of the others, 
    the total number of handshakes is 66 -/
theorem twelve_boys_handshakes : handshakes 12 = 66 := by
  sorry

end NUMINAMATH_CALUDE_twelve_boys_handshakes_l1256_125622


namespace NUMINAMATH_CALUDE_term_2023_equals_2023_term_2023_in_group_64_term_2023_is_7th_in_group_l1256_125686

/-- Definition of the sequence term -/
def sequenceTerm (n : ℕ) : ℕ :=
  let start := 1 + n * (n - 1) / 2
  (n * (2 * start + (n - 1))) / 2

/-- Theorem stating that the 2023rd term of the sequence is 2023 -/
theorem term_2023_equals_2023 : sequenceTerm 64 = 2023 := by
  sorry

/-- Theorem stating that the 2023rd term is in the 64th group -/
theorem term_2023_in_group_64 :
  (63 * 64) / 2 < 2023 ∧ 2023 ≤ (64 * 65) / 2 := by
  sorry

/-- Theorem stating that the 2023rd term is the 7th term in its group -/
theorem term_2023_is_7th_in_group :
  2023 - (63 * 64) / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_term_2023_equals_2023_term_2023_in_group_64_term_2023_is_7th_in_group_l1256_125686


namespace NUMINAMATH_CALUDE_one_third_of_270_l1256_125652

theorem one_third_of_270 : (1 / 3 : ℚ) * 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_270_l1256_125652


namespace NUMINAMATH_CALUDE_largest_geometric_three_digit_number_l1256_125677

/-- Checks if three digits form a geometric sequence -/
def is_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

/-- Checks if three digits are distinct -/
def are_distinct (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- Represents a three-digit number -/
def three_digit_number (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

theorem largest_geometric_three_digit_number :
  ∀ n : ℕ,
  (∃ a b c : ℕ, n = three_digit_number a b c ∧
                a ≤ 8 ∧
                a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧
                a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
                is_geometric_sequence a b c ∧
                are_distinct a b c) →
  n ≤ 842 :=
sorry

end NUMINAMATH_CALUDE_largest_geometric_three_digit_number_l1256_125677


namespace NUMINAMATH_CALUDE_line_contains_point_l1256_125694

theorem line_contains_point (k : ℝ) : 
  (1 - 3 * k * (1/3) = -2 * 4) ↔ (k = 9) := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l1256_125694


namespace NUMINAMATH_CALUDE_goldfish_count_correct_l1256_125607

/-- The number of Goldfish Layla has -/
def num_goldfish : ℕ := 2

/-- The total amount of food Layla gives to all her fish -/
def total_food : ℕ := 12

/-- The number of Swordtails Layla has -/
def num_swordtails : ℕ := 3

/-- The amount of food each Swordtail gets -/
def food_per_swordtail : ℕ := 2

/-- The number of Guppies Layla has -/
def num_guppies : ℕ := 8

/-- The amount of food each Guppy gets -/
def food_per_guppy : ℚ := 1/2

/-- The amount of food each Goldfish gets -/
def food_per_goldfish : ℕ := 1

/-- Theorem stating that the number of Goldfish is correct given the conditions -/
theorem goldfish_count_correct : 
  total_food = num_swordtails * food_per_swordtail + 
               num_guppies * food_per_guppy + 
               num_goldfish * food_per_goldfish :=
by sorry

end NUMINAMATH_CALUDE_goldfish_count_correct_l1256_125607


namespace NUMINAMATH_CALUDE_not_always_parallel_to_plane_l1256_125621

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem not_always_parallel_to_plane
  (a b : Line) (α : Plane)
  (diff : a ≠ b)
  (h1 : subset b α)
  (h2 : parallel_lines a b) :
  ¬(∀ a b α, subset b α → parallel_lines a b → parallel_line_plane a α) :=
sorry

end NUMINAMATH_CALUDE_not_always_parallel_to_plane_l1256_125621


namespace NUMINAMATH_CALUDE_letters_with_both_in_given_alphabet_l1256_125604

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet where
  total : ℕ
  line_no_dot : ℕ
  dot_no_line : ℕ
  all_have_dot_or_line : Bool

/-- The number of letters containing both a dot and a straight line -/
def letters_with_both (a : Alphabet) : ℕ :=
  a.total - a.line_no_dot - a.dot_no_line

/-- Theorem stating the number of letters with both dot and line in the given alphabet -/
theorem letters_with_both_in_given_alphabet :
  ∀ (a : Alphabet),
    a.total = 60 ∧
    a.line_no_dot = 36 ∧
    a.dot_no_line = 4 ∧
    a.all_have_dot_or_line = true →
    letters_with_both a = 20 := by
  sorry


end NUMINAMATH_CALUDE_letters_with_both_in_given_alphabet_l1256_125604


namespace NUMINAMATH_CALUDE_ratio_problem_l1256_125612

theorem ratio_problem (second_part : ℝ) (percent : ℝ) (first_part : ℝ) :
  second_part = 20 →
  percent = 25 →
  first_part / second_part = percent / 100 →
  first_part = 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1256_125612


namespace NUMINAMATH_CALUDE_brand_a_soap_users_l1256_125680

theorem brand_a_soap_users (total : ℕ) (neither : ℕ) (both : ℕ) :
  total = 260 →
  neither = 80 →
  both = 30 →
  (total - neither) = (3 * both) + both + (total - neither - 3 * both - both) →
  (total - neither - 3 * both - both) = 60 :=
by sorry

end NUMINAMATH_CALUDE_brand_a_soap_users_l1256_125680


namespace NUMINAMATH_CALUDE_jills_first_bus_ride_time_l1256_125655

/-- Jill's journey to the library -/
def jills_journey (first_bus_wait : ℕ) (second_bus_ride : ℕ) (first_bus_ride : ℕ) : Prop :=
  second_bus_ride = (first_bus_wait + first_bus_ride) / 2

theorem jills_first_bus_ride_time :
  ∃ (first_bus_ride : ℕ),
    jills_journey 12 21 first_bus_ride ∧
    first_bus_ride = 30 := by
  sorry

end NUMINAMATH_CALUDE_jills_first_bus_ride_time_l1256_125655


namespace NUMINAMATH_CALUDE_pond_to_field_area_ratio_l1256_125663

theorem pond_to_field_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    field_length = 2 * field_width →
    field_length = 28 →
    pond_side = 7 →
    (pond_side^2) / (field_length * field_width) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_pond_to_field_area_ratio_l1256_125663


namespace NUMINAMATH_CALUDE_tournament_handshakes_l1256_125692

theorem tournament_handshakes (n : ℕ) (m : ℕ) (h : n = 4 ∧ m = 2) :
  let total_players := n * m
  let handshakes_per_player := total_players - m
  (total_players * handshakes_per_player) / 2 = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_tournament_handshakes_l1256_125692


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_12_l1256_125645

theorem unique_square_divisible_by_12 :
  ∃! x : ℕ, (∃ y : ℕ, x = y^2) ∧ 
            (12 ∣ x) ∧ 
            100 ≤ x ∧ x ≤ 200 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_12_l1256_125645


namespace NUMINAMATH_CALUDE_min_sum_equal_last_three_digits_l1256_125659

theorem min_sum_equal_last_three_digits (m n : ℕ) : 
  m ≥ 1 → n > m → 
  (1978^n - 1978^m) % 1000 = 0 → 
  (∀ k l : ℕ, k ≥ 1 → l > k → (1978^l - 1978^k) % 1000 = 0 → m + n ≤ k + l) → 
  m + n = 106 := by
sorry

end NUMINAMATH_CALUDE_min_sum_equal_last_three_digits_l1256_125659


namespace NUMINAMATH_CALUDE_solve_equation_l1256_125670

theorem solve_equation (x : ℝ) : (3 * x + 36 = 48) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1256_125670


namespace NUMINAMATH_CALUDE_enclosure_blocks_count_l1256_125654

/-- Calculates the number of cubical blocks used to create a cuboidal enclosure --/
def cubicalBlocksCount (length width height thickness : ℕ) : ℕ :=
  length * width * height - (length - 2 * thickness) * (width - 2 * thickness) * (height - thickness)

/-- Theorem stating that the number of cubical blocks for the given dimensions is 372 --/
theorem enclosure_blocks_count :
  cubicalBlocksCount 15 8 7 1 = 372 := by
  sorry

end NUMINAMATH_CALUDE_enclosure_blocks_count_l1256_125654


namespace NUMINAMATH_CALUDE_oliver_final_amount_l1256_125626

def oliver_money_left (initial_amount savings frisbee_cost puzzle_cost birthday_gift : ℕ) : ℕ :=
  initial_amount + savings - frisbee_cost - puzzle_cost + birthday_gift

theorem oliver_final_amount :
  oliver_money_left 9 5 4 3 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_oliver_final_amount_l1256_125626


namespace NUMINAMATH_CALUDE_g_value_at_4_l1256_125624

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 - 2*x + 3

-- Define the properties of g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d) ∧  -- g is a cubic polynomial
  (g 0 = 3) ∧  -- g(0) = 3
  (∀ r : ℝ, f r = 0 → ∃ s : ℝ, g s = 0 ∧ s = r^2)  -- roots of g are squares of roots of f

-- State the theorem
theorem g_value_at_4 (g : ℝ → ℝ) (h : g_properties g) : g 4 = -75 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_4_l1256_125624


namespace NUMINAMATH_CALUDE_function_simplification_l1256_125627

theorem function_simplification (x : ℝ) : 
  Real.sqrt (4 * Real.sin x ^ 4 - 2 * Real.cos (2 * x) + 3) + 
  Real.sqrt (4 * Real.cos x ^ 4 + 2 * Real.cos (2 * x) + 3) = 4 := by
sorry

end NUMINAMATH_CALUDE_function_simplification_l1256_125627


namespace NUMINAMATH_CALUDE_charity_savings_interest_l1256_125693

/-- Represents the initial savings amount in dollars -/
def P : ℝ := 2181

/-- Represents the first interest rate (8% per annum) -/
def r1 : ℝ := 0.08

/-- Represents the second interest rate (4% per annum) -/
def r2 : ℝ := 0.04

/-- Represents the time period for each interest rate (3 months = 0.25 years) -/
def t : ℝ := 0.25

/-- Represents the final amount after applying both interest rates -/
def A : ℝ := 2247.50

/-- Theorem stating that the initial amount P results in the final amount A 
    after applying the given interest rates for the specified time periods -/
theorem charity_savings_interest : 
  P * (1 + r1 * t) * (1 + r2 * t) = A := by sorry

end NUMINAMATH_CALUDE_charity_savings_interest_l1256_125693


namespace NUMINAMATH_CALUDE_range_of_t_range_of_t_lower_bound_l1256_125658

def A : Set ℝ := {x | -3 < x ∧ x < 7}
def B (t : ℝ) : Set ℝ := {x | t + 1 < x ∧ x < 2*t - 1}

theorem range_of_t (t : ℝ) : B t ⊆ A → t ≤ 4 :=
  sorry

theorem range_of_t_lower_bound : ∀ ε > 0, ∃ t : ℝ, t ≤ 4 - ε ∧ B t ⊆ A :=
  sorry

end NUMINAMATH_CALUDE_range_of_t_range_of_t_lower_bound_l1256_125658


namespace NUMINAMATH_CALUDE_no_integer_solution_l1256_125648

/-- The equation x^3 - 3xy^2 + y^3 = 2891 has no integer solutions -/
theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = 2891 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1256_125648


namespace NUMINAMATH_CALUDE_lexie_age_l1256_125610

/-- Represents the ages of Lexie, her brother, and her sister -/
structure Family where
  lexie : ℕ
  brother : ℕ
  sister : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (f : Family) : Prop :=
  f.lexie = f.brother + 6 ∧
  f.sister = 2 * f.lexie ∧
  f.sister - f.brother = 14

/-- Theorem stating that if a family satisfies the given conditions, Lexie's age is 8 -/
theorem lexie_age (f : Family) (h : satisfiesConditions f) : f.lexie = 8 := by
  sorry

#check lexie_age

end NUMINAMATH_CALUDE_lexie_age_l1256_125610


namespace NUMINAMATH_CALUDE_trip_length_proof_l1256_125699

/-- Represents the total length of the trip in miles -/
def total_distance : ℝ := 95

/-- Represents the distance traveled on battery -/
def battery_distance : ℝ := 30

/-- Represents the distance traveled on first gasoline mode -/
def first_gas_distance : ℝ := 70

/-- Represents the rate of gasoline consumption in the first gasoline mode -/
def first_gas_rate : ℝ := 0.03

/-- Represents the rate of gasoline consumption in the second gasoline mode -/
def second_gas_rate : ℝ := 0.04

/-- Represents the overall average miles per gallon -/
def average_mpg : ℝ := 50

theorem trip_length_proof :
  total_distance = battery_distance + first_gas_distance +
    (total_distance - battery_distance - first_gas_distance) ∧
  (first_gas_rate * first_gas_distance +
   second_gas_rate * (total_distance - battery_distance - first_gas_distance)) *
    average_mpg = total_distance :=
by sorry

end NUMINAMATH_CALUDE_trip_length_proof_l1256_125699


namespace NUMINAMATH_CALUDE_expression_simplification_l1256_125672

theorem expression_simplification (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (a - 3 * a / (a + 1)) / ((a^2 - 4*a + 4) / (a + 1)) = a / (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1256_125672


namespace NUMINAMATH_CALUDE_remainder_problem_l1256_125691

theorem remainder_problem (p : Nat) (h : Prime p) (h1 : p = 13) :
  (7 * 12^24 + 2^24) % p = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1256_125691


namespace NUMINAMATH_CALUDE_complex_point_in_third_quadrant_l1256_125640

/-- Given a complex number z = 1 + i, prove that the real and imaginary parts of (5/z^2) - z are both negative -/
theorem complex_point_in_third_quadrant (z : ℂ) (h : z = 1 + Complex.I) :
  (Complex.re ((5 / z^2) - z) < 0) ∧ (Complex.im ((5 / z^2) - z) < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_point_in_third_quadrant_l1256_125640


namespace NUMINAMATH_CALUDE_entrepreneur_raised_12000_l1256_125656

/-- Represents the crowdfunding levels and backers for an entrepreneur's business effort -/
structure CrowdfundingCampaign where
  highest_level : ℕ
  second_level : ℕ
  lowest_level : ℕ
  highest_backers : ℕ
  second_backers : ℕ
  lowest_backers : ℕ

/-- Calculates the total amount raised in a crowdfunding campaign -/
def total_raised (campaign : CrowdfundingCampaign) : ℕ :=
  campaign.highest_level * campaign.highest_backers +
  campaign.second_level * campaign.second_backers +
  campaign.lowest_level * campaign.lowest_backers

/-- Theorem stating that the entrepreneur raised $12000 -/
theorem entrepreneur_raised_12000 :
  ∀ (campaign : CrowdfundingCampaign),
  campaign.highest_level = 5000 ∧
  campaign.second_level = campaign.highest_level / 10 ∧
  campaign.lowest_level = campaign.second_level / 10 ∧
  campaign.highest_backers = 2 ∧
  campaign.second_backers = 3 ∧
  campaign.lowest_backers = 10 →
  total_raised campaign = 12000 :=
sorry

end NUMINAMATH_CALUDE_entrepreneur_raised_12000_l1256_125656


namespace NUMINAMATH_CALUDE_inequality_proof_l1256_125601

theorem inequality_proof (a b c : ℝ) : 
  (a + b) * (a + b - 2 * c) + (b + c) * (b + c - 2 * a) + (c + a) * (c + a - 2 * b) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1256_125601


namespace NUMINAMATH_CALUDE_rachel_total_books_l1256_125613

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves with mystery books -/
def mystery_shelves : ℕ := 6

/-- The number of shelves with picture books -/
def picture_shelves : ℕ := 2

/-- The total number of books Rachel has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem rachel_total_books : total_books = 72 := by
  sorry

end NUMINAMATH_CALUDE_rachel_total_books_l1256_125613


namespace NUMINAMATH_CALUDE_number_of_cows_farm_cows_l1256_125632

/-- The number of cows in a farm given their husk consumption -/
theorem number_of_cows (total_bags : ℕ) (total_days : ℕ) (cow_days : ℕ) : ℕ :=
  total_bags * cow_days / total_days

/-- Proof that there are 26 cows in the farm -/
theorem farm_cows : number_of_cows 26 26 26 = 26 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cows_farm_cows_l1256_125632


namespace NUMINAMATH_CALUDE_sine_curve_intersection_l1256_125600

theorem sine_curve_intersection (A a : ℝ) (h1 : A > 0) (h2 : a > 0) :
  (∃ x1 x2 x3 x4 : ℝ, 
    0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 ≤ 2 * π ∧
    A * Real.sin x1 + a = 2 ∧
    A * Real.sin x2 + a = -1 ∧
    A * Real.sin x3 + a = -1 ∧
    A * Real.sin x4 + a = 2 ∧
    (x2 - x1) = (x4 - x3) ∧
    x2 ≠ x1) →
  a = 1/2 ∧ A > 3/2 := by
sorry

end NUMINAMATH_CALUDE_sine_curve_intersection_l1256_125600


namespace NUMINAMATH_CALUDE_power_of_negative_cube_l1256_125657

theorem power_of_negative_cube (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_cube_l1256_125657


namespace NUMINAMATH_CALUDE_min_sum_abc_l1256_125671

theorem min_sum_abc (a b c : ℕ+) (h : a.val * b.val * c.val + b.val * c.val + c.val = 2014) :
  ∃ (a' b' c' : ℕ+), 
    a'.val * b'.val * c'.val + b'.val * c'.val + c'.val = 2014 ∧
    a'.val + b'.val + c'.val = 40 ∧
    ∀ (x y z : ℕ+), x.val * y.val * z.val + y.val * z.val + z.val = 2014 → 
      x.val + y.val + z.val ≥ 40 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_abc_l1256_125671


namespace NUMINAMATH_CALUDE_v_2003_equals_5_l1256_125619

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 2
| 4 => 1
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 5
| n + 1 => g (v n)

-- Theorem to prove
theorem v_2003_equals_5 : v 2003 = 5 := by
  sorry


end NUMINAMATH_CALUDE_v_2003_equals_5_l1256_125619


namespace NUMINAMATH_CALUDE_f2_form_l1256_125605

/-- A quadratic function with coefficients a, b, and c. -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The reflection of a function about the y-axis. -/
def reflect_y (g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ g (-x)

/-- The reflection of a function about the line y = 1. -/
def reflect_y_eq_1 (g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ 2 - g x

/-- Theorem stating the form of f2 after two reflections of f. -/
theorem f2_form (a b c : ℝ) (ha : a ≠ 0) :
  let f1 := reflect_y (f a b c)
  let f2 := reflect_y_eq_1 f1
  ∀ x, f2 x = -a * x^2 + b * x + (2 - c) :=
sorry

end NUMINAMATH_CALUDE_f2_form_l1256_125605


namespace NUMINAMATH_CALUDE_f_difference_l1256_125649

/-- Given a function f defined as f(n) = 1/3 * n * (n+1) * (n+2),
    prove that f(r) - f(r-1) = r * (r+1) for any real number r. -/
theorem f_difference (r : ℝ) : 
  let f (n : ℝ) := (1/3) * n * (n+1) * (n+2)
  f r - f (r-1) = r * (r+1) := by
sorry

end NUMINAMATH_CALUDE_f_difference_l1256_125649


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l1256_125616

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: In a geometric sequence where a_2 = 4 and a_6 = 16, a_4 = 8 -/
theorem geometric_sequence_a4 (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a) 
    (h_a2 : a 2 = 4) 
    (h_a6 : a 6 = 16) : 
  a 4 = 8 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_a4_l1256_125616
