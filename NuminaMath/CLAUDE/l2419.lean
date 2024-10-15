import Mathlib

namespace NUMINAMATH_CALUDE_range_of_sum_and_abs_l2419_241928

theorem range_of_sum_and_abs (a b : ℝ) 
  (ha : -1 ≤ a ∧ a ≤ 3) 
  (hb : -5 < b ∧ b < 3) : 
  ∀ x, x ∈ Set.Icc (-1 : ℝ) 8 ↔ ∃ (a' b' : ℝ), 
    -1 ≤ a' ∧ a' ≤ 3 ∧ 
    -5 < b' ∧ b' < 3 ∧ 
    x = a' + |b'| :=
by sorry

end NUMINAMATH_CALUDE_range_of_sum_and_abs_l2419_241928


namespace NUMINAMATH_CALUDE_green_socks_count_l2419_241932

theorem green_socks_count (total : ℕ) (white : ℕ) (blue : ℕ) (red : ℕ) (green : ℕ) :
  total = 900 ∧
  white = total / 3 ∧
  blue = total / 4 ∧
  red = total / 5 ∧
  green = total - (white + blue + red) →
  green = 195 := by
sorry

end NUMINAMATH_CALUDE_green_socks_count_l2419_241932


namespace NUMINAMATH_CALUDE_seaweed_distribution_l2419_241966

theorem seaweed_distribution (total_seaweed : ℝ) (fire_percentage : ℝ) (human_percentage : ℝ) :
  total_seaweed = 400 ∧ 
  fire_percentage = 0.5 ∧ 
  human_percentage = 0.25 →
  (total_seaweed * (1 - fire_percentage) * (1 - human_percentage)) = 150 := by
  sorry

end NUMINAMATH_CALUDE_seaweed_distribution_l2419_241966


namespace NUMINAMATH_CALUDE_compound_proposition_truth_l2419_241945

theorem compound_proposition_truth (p q : Prop) 
  (hp : p) (hq : ¬q) : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_compound_proposition_truth_l2419_241945


namespace NUMINAMATH_CALUDE_fathers_age_l2419_241942

/-- Proves that the father's age is 64 years given the problem conditions -/
theorem fathers_age (son_age : ℕ) : 
  (4 * son_age = 4 * son_age) →  -- Father is four times as old as his son
  ((son_age - 10) + (4 * son_age - 10) = 60) →  -- Sum of ages 10 years ago was 60
  (4 * son_age = 64) :=  -- Father's present age is 64
by
  sorry

#check fathers_age

end NUMINAMATH_CALUDE_fathers_age_l2419_241942


namespace NUMINAMATH_CALUDE_min_m_and_x_range_l2419_241967

theorem min_m_and_x_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + 3*b^2 = 3) :
  (∃ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 3*b^2 = 3 → Real.sqrt 5 * a + b ≤ m) ∧
            (∀ m' : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 3*b^2 = 3 → Real.sqrt 5 * a + b ≤ m') → m ≤ m') ∧
            m = 4) ∧
  (∀ x : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 3*b^2 = 3 → 2 * |x - 1| + |x| ≥ Real.sqrt 5 * a + b) →
            (x ≤ -2/3 ∨ x ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_min_m_and_x_range_l2419_241967


namespace NUMINAMATH_CALUDE_sum_reciprocals_of_factors_12_l2419_241978

/-- The set of natural-number factors of 12 -/
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}

/-- The sum of the reciprocals of the natural-number factors of 12 -/
def sum_reciprocals : ℚ := (factors_of_12.sum fun n => (1 : ℚ) / n)

/-- Theorem: The sum of the reciprocals of the natural-number factors of 12 is equal to 7/3 -/
theorem sum_reciprocals_of_factors_12 : sum_reciprocals = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_of_factors_12_l2419_241978


namespace NUMINAMATH_CALUDE_yoz_perpendicular_x_xoz_perpendicular_y_xoy_perpendicular_z_l2419_241937

-- Define the three-dimensional Cartesian coordinate system
structure CartesianCoordinate3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the coordinate planes
def yoz_plane : Set CartesianCoordinate3D := {p | p.x = 0}
def xoz_plane : Set CartesianCoordinate3D := {p | p.y = 0}
def xoy_plane : Set CartesianCoordinate3D := {p | p.z = 0}

-- Define the axes
def x_axis : Set CartesianCoordinate3D := {p | p.y = 0 ∧ p.z = 0}
def y_axis : Set CartesianCoordinate3D := {p | p.x = 0 ∧ p.z = 0}
def z_axis : Set CartesianCoordinate3D := {p | p.x = 0 ∧ p.y = 0}

-- Define perpendicularity between a plane and an axis
def perpendicular (plane : Set CartesianCoordinate3D) (axis : Set CartesianCoordinate3D) : Prop :=
  ∀ p ∈ plane, ∀ q ∈ axis, (p.x - q.x) * q.x + (p.y - q.y) * q.y + (p.z - q.z) * q.z = 0

-- Theorem statements
theorem yoz_perpendicular_x : perpendicular yoz_plane x_axis := sorry

theorem xoz_perpendicular_y : perpendicular xoz_plane y_axis := sorry

theorem xoy_perpendicular_z : perpendicular xoy_plane z_axis := sorry

end NUMINAMATH_CALUDE_yoz_perpendicular_x_xoz_perpendicular_y_xoy_perpendicular_z_l2419_241937


namespace NUMINAMATH_CALUDE_correct_arrival_times_l2419_241948

/-- Represents the train journey with given parameters -/
structure TrainJourney where
  totalDistance : ℝ
  uphillDistance1 : ℝ
  flatDistance : ℝ
  uphillDistance2 : ℝ
  speedDifference : ℝ
  stationDistances : List ℝ
  stopTime : ℝ
  departureTime : ℝ
  arrivalTime : ℝ

/-- Calculate arrival times at intermediate stations -/
def calculateArrivalTimes (journey : TrainJourney) : List ℝ :=
  sorry

/-- Main theorem: Arrival times at stations are correct -/
theorem correct_arrival_times (journey : TrainJourney)
  (h1 : journey.totalDistance = 185)
  (h2 : journey.uphillDistance1 = 40)
  (h3 : journey.flatDistance = 105)
  (h4 : journey.uphillDistance2 = 40)
  (h5 : journey.speedDifference = 10)
  (h6 : journey.stationDistances = [20, 70, 100, 161])
  (h7 : journey.stopTime = 3/60)
  (h8 : journey.departureTime = 8)
  (h9 : journey.arrivalTime = 10 + 22/60) :
  calculateArrivalTimes journey = [8 + 15/60, 8 + 53/60, 9 + 21/60, 10 + 34/60] :=
sorry

end NUMINAMATH_CALUDE_correct_arrival_times_l2419_241948


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_open_interval_l2419_241979

-- Define the sets M and N
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, x > 0 ∧ y = 2^x}
def N : Set ℝ := {x : ℝ | 2*x - x^2 > 0}

-- State the theorem
theorem M_intersect_N_eq_open_interval : M ∩ N = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_open_interval_l2419_241979


namespace NUMINAMATH_CALUDE_factorization_equality_l2419_241927

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2419_241927


namespace NUMINAMATH_CALUDE_sin_alpha_value_l2419_241958

theorem sin_alpha_value (α : Real) (h : Real.sin (Real.pi - α) = -1/3) :
  Real.sin α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l2419_241958


namespace NUMINAMATH_CALUDE_range_of_t_l2419_241941

theorem range_of_t (a b t : ℝ) (h1 : a^2 + a*b + b^2 = 1) (h2 : t = a*b - a^2 - b^2) :
  -3 ≤ t ∧ t ≤ -1/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_l2419_241941


namespace NUMINAMATH_CALUDE_least_seven_binary_digits_l2419_241962

/-- The number of binary digits required to represent a positive integer -/
def binary_digits (n : ℕ+) : ℕ :=
  (Nat.log 2 n.val).succ

/-- Predicate to check if a number has exactly 7 binary digits -/
def has_seven_binary_digits (n : ℕ+) : Prop :=
  binary_digits n = 7

theorem least_seven_binary_digits :
  ∃ (n : ℕ+), has_seven_binary_digits n ∧
    ∀ (m : ℕ+), has_seven_binary_digits m → n ≤ m ∧
    n = 64 := by sorry

end NUMINAMATH_CALUDE_least_seven_binary_digits_l2419_241962


namespace NUMINAMATH_CALUDE_second_number_calculation_l2419_241929

theorem second_number_calculation (first_number : ℝ) (second_number : ℝ) : 
  first_number = 640 → 
  (0.5 * first_number) = (0.2 * second_number + 190) → 
  second_number = 650 := by
sorry

end NUMINAMATH_CALUDE_second_number_calculation_l2419_241929


namespace NUMINAMATH_CALUDE_nine_digit_divisible_by_11_l2419_241917

def is_divisible_by_11 (n : ℕ) : Prop :=
  ∃ k : ℤ, n = 11 * k

def sum_odd_positions (m : ℕ) : ℕ :=
  8 + 4 + m + 6 + 8

def sum_even_positions : ℕ :=
  5 + 2 + 7 + 1

def number (m : ℕ) : ℕ :=
  8542000000 + m * 10000 + 7618

theorem nine_digit_divisible_by_11 (m : ℕ) (h : m < 10) :
  is_divisible_by_11 (number m) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_divisible_by_11_l2419_241917


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2419_241920

/-- Given vectors a and b, if 2a + b is parallel to ma - b, then m = -2 -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
    (ha : a = (1, -2))
    (hb : b = (3, 0))
    (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (2 • a + b) = k • (m • a - b)) :
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2419_241920


namespace NUMINAMATH_CALUDE_simplify_fraction_l2419_241921

theorem simplify_fraction : 25 * (9 / 14) * (2 / 27) = 25 / 21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2419_241921


namespace NUMINAMATH_CALUDE_reflection_line_property_l2419_241935

/-- A line that reflects a point (x₁, y₁) to (x₂, y₂) -/
structure ReflectionLine where
  m : ℝ
  b : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  reflects : ((x₂ - x₁) * m + y₁ + y₂) / 2 = m * ((x₂ + x₁) / 2) + b

/-- The theorem stating that for a line y = mx + b that reflects (1, -4) to (7, 2), 3m + 2b = 3 -/
theorem reflection_line_property (line : ReflectionLine) 
    (h1 : line.x₁ = 1) (h2 : line.y₁ = -4) (h3 : line.x₂ = 7) (h4 : line.y₂ = 2) : 
    3 * line.m + 2 * line.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_property_l2419_241935


namespace NUMINAMATH_CALUDE_four_intersection_points_l2419_241954

/-- The polynomial function representing the curve -/
def f (c : ℝ) (x : ℝ) : ℝ := x^4 + 9*x^3 + c*x^2 + 9*x + 4

/-- Theorem stating the condition for the existence of a line intersecting the curve in four distinct points -/
theorem four_intersection_points (c : ℝ) :
  (∃ (m n : ℝ), ∀ (x : ℝ), (f c x = m*x + n) → (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f c x₁ = m*x₁ + n ∧ f c x₂ = m*x₂ + n ∧ f c x₃ = m*x₃ + n ∧ f c x₄ = m*x₄ + n)) ↔
  c ≤ 243/8 :=
sorry

end NUMINAMATH_CALUDE_four_intersection_points_l2419_241954


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l2419_241951

theorem modular_congruence_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -5203 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l2419_241951


namespace NUMINAMATH_CALUDE_inequality_range_l2419_241974

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → Real.log (x + 2) + a * (x^2 + x) ≥ 0) ↔ 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2419_241974


namespace NUMINAMATH_CALUDE_circle_sector_radius_l2419_241934

theorem circle_sector_radius 
  (area : ℝ) 
  (arc_length : ℝ) 
  (h1 : area = 8.75) 
  (h2 : arc_length = 3.5) : 
  ∃ (radius : ℝ), radius = 5 ∧ area = (1/2) * radius * arc_length :=
by
  sorry

end NUMINAMATH_CALUDE_circle_sector_radius_l2419_241934


namespace NUMINAMATH_CALUDE_equation_solution_l2419_241998

theorem equation_solution : ∃! (x : ℝ), x ≠ 0 ∧ (5*x)^20 = (20*x)^10 ∧ x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2419_241998


namespace NUMINAMATH_CALUDE_work_completion_time_l2419_241940

/-- Given:
  * Mahesh can complete the entire work in 45 days
  * Mahesh works for 20 days
  * Rajesh finishes the remaining work in 30 days
  Prove that Y will take 54 days to complete the work -/
theorem work_completion_time (mahesh_full_time rajesh_completion_time mahesh_work_time : ℕ)
  (h1 : mahesh_full_time = 45)
  (h2 : mahesh_work_time = 20)
  (h3 : rajesh_completion_time = 30) :
  54 = (mahesh_full_time * rajesh_completion_time) / (rajesh_completion_time - mahesh_work_time) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2419_241940


namespace NUMINAMATH_CALUDE_train_speed_l2419_241906

/-- Given a train of length 160 meters that crosses a stationary point in 18 seconds, 
    its speed is 32 km/h. -/
theorem train_speed (length : Real) (time : Real) (speed : Real) : 
  length = 160 ∧ time = 18 → speed = (length / time) * 3.6 → speed = 32 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2419_241906


namespace NUMINAMATH_CALUDE_round_to_nearest_integer_l2419_241924

def number : ℝ := 7293847.2635142

theorem round_to_nearest_integer : 
  Int.floor (number + 0.5) = 7293847 := by sorry

end NUMINAMATH_CALUDE_round_to_nearest_integer_l2419_241924


namespace NUMINAMATH_CALUDE_arcsin_negative_one_l2419_241946

theorem arcsin_negative_one : Real.arcsin (-1) = -π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_negative_one_l2419_241946


namespace NUMINAMATH_CALUDE_jane_exercise_goal_l2419_241976

/-- Jane's exercise routine -/
structure ExerciseRoutine where
  daily_hours : ℕ
  days_per_week : ℕ
  total_hours : ℕ

/-- Calculate the number of weeks Jane hit her goal -/
def weeks_goal_met (routine : ExerciseRoutine) : ℕ :=
  routine.total_hours / (routine.daily_hours * routine.days_per_week)

/-- Theorem: Jane hit her goal for 8 weeks -/
theorem jane_exercise_goal (routine : ExerciseRoutine) 
  (h1 : routine.daily_hours = 1)
  (h2 : routine.days_per_week = 5)
  (h3 : routine.total_hours = 40) : 
  weeks_goal_met routine = 8 := by
  sorry

end NUMINAMATH_CALUDE_jane_exercise_goal_l2419_241976


namespace NUMINAMATH_CALUDE_gabriel_capsule_days_l2419_241926

/-- The number of days in July -/
def days_in_july : ℕ := 31

/-- The number of days Gabriel forgot to take his capsules -/
def days_forgot : ℕ := 3

/-- The number of days Gabriel took his capsules in July -/
def days_took_capsules : ℕ := days_in_july - days_forgot

theorem gabriel_capsule_days : days_took_capsules = 28 := by
  sorry

end NUMINAMATH_CALUDE_gabriel_capsule_days_l2419_241926


namespace NUMINAMATH_CALUDE_triangle_area_arithmetic_progression_l2419_241938

/-- The area of a triangle with base 2a - d and height 2a + d is 2a^2 - d^2/2 -/
theorem triangle_area_arithmetic_progression (a d : ℝ) (h_a : a > 0) :
  let base := 2 * a - d
  let height := 2 * a + d
  (1 / 2 : ℝ) * base * height = 2 * a^2 - d^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_arithmetic_progression_l2419_241938


namespace NUMINAMATH_CALUDE_angle_equality_l2419_241908

theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 2 * Real.sin (20 * π / 180) = Real.cos θ - Real.sin θ) : 
  θ = 25 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l2419_241908


namespace NUMINAMATH_CALUDE_organization_size_after_five_years_l2419_241983

def organization_size (n : ℕ) : ℕ :=
  match n with
  | 0 => 20
  | k + 1 => 4 * organization_size k - 21

theorem organization_size_after_five_years :
  organization_size 5 = 13343 := by
  sorry

end NUMINAMATH_CALUDE_organization_size_after_five_years_l2419_241983


namespace NUMINAMATH_CALUDE_sarah_pencil_multiple_l2419_241910

/-- The number of pencils Sarah bought on Monday -/
def monday_pencils : ℕ := 20

/-- The number of pencils Sarah bought on Tuesday -/
def tuesday_pencils : ℕ := 18

/-- The total number of pencils Sarah has -/
def total_pencils : ℕ := 92

/-- The multiple of pencils bought on Wednesday compared to Tuesday -/
def wednesday_multiple : ℕ := (total_pencils - monday_pencils - tuesday_pencils) / tuesday_pencils

theorem sarah_pencil_multiple : wednesday_multiple = 3 := by
  sorry

end NUMINAMATH_CALUDE_sarah_pencil_multiple_l2419_241910


namespace NUMINAMATH_CALUDE_solve_problem_l2419_241909

def problem (x : ℝ) : Prop :=
  let k_speed := x
  let m_speed := x - 0.5
  let k_time := 40 / k_speed
  let m_time := 40 / m_speed
  (m_time - k_time = 1/3) ∧ (k_time = 5)

theorem solve_problem :
  ∃ x : ℝ, problem x :=
sorry

end NUMINAMATH_CALUDE_solve_problem_l2419_241909


namespace NUMINAMATH_CALUDE_tomato_plants_count_l2419_241950

def strawberry_plants : ℕ := 5
def strawberries_per_plant : ℕ := 14
def tomatoes_per_plant : ℕ := 16
def fruits_per_basket : ℕ := 7
def strawberry_basket_price : ℕ := 9
def tomato_basket_price : ℕ := 6
def total_revenue : ℕ := 186

theorem tomato_plants_count (tomato_plants : ℕ) : 
  strawberry_plants * strawberries_per_plant / fruits_per_basket * strawberry_basket_price + 
  tomato_plants * tomatoes_per_plant / fruits_per_basket * tomato_basket_price = total_revenue → 
  tomato_plants = 7 := by
  sorry

end NUMINAMATH_CALUDE_tomato_plants_count_l2419_241950


namespace NUMINAMATH_CALUDE_melissa_games_l2419_241939

/-- The number of games Melissa played -/
def number_of_games (total_points : ℕ) (points_per_game : ℕ) : ℕ :=
  total_points / points_per_game

/-- Proof that Melissa played 3 games -/
theorem melissa_games : number_of_games 21 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_melissa_games_l2419_241939


namespace NUMINAMATH_CALUDE_max_value_of_S_fourth_power_l2419_241981

theorem max_value_of_S_fourth_power :
  let S (x : ℝ) := |Real.sqrt (x^2 + 4*x + 5) - Real.sqrt (x^2 + 2*x + 5)|
  ∀ x : ℝ, (S x)^4 ≤ 4 ∧ ∃ y : ℝ, (S y)^4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_S_fourth_power_l2419_241981


namespace NUMINAMATH_CALUDE_card_z_value_l2419_241953

/-- Given four cards W, X, Y, Z with specific tagging rules, prove that Z is tagged with 400. -/
theorem card_z_value : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun w x y z =>
    w = 200 ∧
    x = w / 2 ∧
    y = w + x ∧
    w + x + y + z = 1000 →
    z = 400

/-- Proof of the card_z_value theorem -/
lemma prove_card_z_value : card_z_value 200 100 300 400 := by
  sorry

end NUMINAMATH_CALUDE_card_z_value_l2419_241953


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2419_241922

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 1) : 
  a + b + c + d - 1 ≥ 16*a*b*c*d ∧ 
  (a + b + c + d - 1 = 16*a*b*c*d ↔ a = 1/2 ∧ b = 1/2 ∧ c = 1/2 ∧ d = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2419_241922


namespace NUMINAMATH_CALUDE_distance_between_points_l2419_241918

/-- The distance between two points with the same x-coordinate in a Cartesian coordinate system. -/
def distance_same_x (y₁ y₂ : ℝ) : ℝ := |y₂ - y₁|

/-- Theorem stating that the distance between (3,-2) and (3,1) is 3. -/
theorem distance_between_points : distance_same_x (-2) 1 = 3 := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l2419_241918


namespace NUMINAMATH_CALUDE_problem_solution_l2419_241931

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp (-x)) / a + a / (Real.exp (-x))

theorem problem_solution (a : ℝ) (h_a : a > 0) 
  (h_even : ∀ x, f a x = f a (-x)) :
  (a = 1) ∧
  (∀ x y, x ≥ 0 → y ≥ 0 → x < y → f a x < f a y) ∧
  (∀ m, (∀ x, f 1 x - m^2 + m ≥ 0) ↔ -1 ≤ m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2419_241931


namespace NUMINAMATH_CALUDE_go_stones_perimeter_l2419_241984

/-- Calculates the number of stones on the perimeter of a rectangle made of Go stones -/
def perimeter_stones (width : ℕ) (height : ℕ) : ℕ :=
  2 * (width + height) - 4

theorem go_stones_perimeter :
  let width : ℕ := 4
  let height : ℕ := 8
  perimeter_stones width height = 20 := by sorry

end NUMINAMATH_CALUDE_go_stones_perimeter_l2419_241984


namespace NUMINAMATH_CALUDE_florist_roses_sold_l2419_241992

/-- Represents the number of roses sold by a florist -/
def roses_sold (initial : ℕ) (picked : ℕ) (final : ℕ) : ℕ :=
  initial + picked - final

/-- Theorem stating that the florist sold 16 roses -/
theorem florist_roses_sold :
  roses_sold 37 19 40 = 16 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_sold_l2419_241992


namespace NUMINAMATH_CALUDE_integral_sin_plus_sqrt_one_minus_x_squared_l2419_241936

open Real MeasureTheory

theorem integral_sin_plus_sqrt_one_minus_x_squared (f g : ℝ → ℝ) :
  (∫ x in (-1)..1, f x) = 0 →
  (∫ x in (-1)..1, g x) = π / 2 →
  (∫ x in (-1)..1, f x + g x) = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_integral_sin_plus_sqrt_one_minus_x_squared_l2419_241936


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2419_241944

/-- A line passing through the origin -/
structure OriginLine where
  slope : ℝ

/-- The intersection point of two lines -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the problem setup -/
def TriangleSetup (l : OriginLine) : Prop :=
  ∃ (p q : Point),
    -- The line intersects x = 1
    p.x = 1 ∧ p.y = -l.slope
    -- The line intersects y = 1 + (√2/2)x
    ∧ q.x = 1 ∧ q.y = 1 + (Real.sqrt 2 / 2)
    -- The three lines form an equilateral triangle
    ∧ (p.x - 0)^2 + (p.y - 0)^2 = (q.x - p.x)^2 + (q.y - p.y)^2
    ∧ (q.x - 0)^2 + (q.y - 0)^2 = (q.x - p.x)^2 + (q.y - p.y)^2

/-- The main theorem -/
theorem triangle_perimeter (l : OriginLine) :
  TriangleSetup l → (3 : ℝ) + 3 * Real.sqrt 2 = 
    let p := Point.mk 1 (-l.slope)
    let q := Point.mk 1 (1 + Real.sqrt 2 / 2)
    3 * Real.sqrt ((q.x - p.x)^2 + (q.y - p.y)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_perimeter_l2419_241944


namespace NUMINAMATH_CALUDE_smallest_number_l2419_241971

theorem smallest_number (a b c d : ℝ) (h1 : a = 0) (h2 : b = -Real.rpow 8 (1/3)) (h3 : c = 2) (h4 : d = -1.7) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
sorry

end NUMINAMATH_CALUDE_smallest_number_l2419_241971


namespace NUMINAMATH_CALUDE_second_sum_calculation_l2419_241923

/-- Given a total sum of 2665 Rs divided into two parts, where the interest on the first part
    for 5 years at 3% per annum equals the interest on the second part for 3 years at 5% per annum,
    prove that the second part is equal to 1332.5 Rs. -/
theorem second_sum_calculation (total : ℝ) (first_part : ℝ) (second_part : ℝ) :
  total = 2665 →
  first_part + second_part = total →
  (first_part * 3 * 5) / 100 = (second_part * 5 * 3) / 100 →
  second_part = 1332.5 := by
  sorry

end NUMINAMATH_CALUDE_second_sum_calculation_l2419_241923


namespace NUMINAMATH_CALUDE_only_five_students_l2419_241985

/-- Represents the number of students -/
def n : ℕ := sorry

/-- Represents the total number of problems solved -/
def S : ℕ := sorry

/-- Represents the number of problems solved by one student -/
def a : ℕ := sorry

/-- The condition that each student solved more than one-fifth of the problems solved by others -/
axiom condition1 : a > (S - a) / 5

/-- The condition that each student solved less than one-third of the problems solved by others -/
axiom condition2 : a < (S - a) / 3

/-- The total number of problems is the sum of problems solved by all students -/
axiom total_problems : S = n * a

/-- The theorem stating that the only possible number of students is 5 -/
theorem only_five_students : n = 5 := by sorry

end NUMINAMATH_CALUDE_only_five_students_l2419_241985


namespace NUMINAMATH_CALUDE_sqrt_288_simplification_l2419_241991

theorem sqrt_288_simplification : Real.sqrt 288 = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_288_simplification_l2419_241991


namespace NUMINAMATH_CALUDE_crosswalk_parallelogram_l2419_241959

/-- A parallelogram with given dimensions -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  height1 : ℝ
  height2 : ℝ

/-- The theorem about the parallelogram representing the crosswalk -/
theorem crosswalk_parallelogram (p : Parallelogram) 
  (h1 : p.side1 = 18)
  (h2 : p.height1 = 60)
  (h3 : p.side2 = 60) :
  p.height2 = 18 := by
  sorry

#check crosswalk_parallelogram

end NUMINAMATH_CALUDE_crosswalk_parallelogram_l2419_241959


namespace NUMINAMATH_CALUDE_ratio_c_to_a_is_sqrt2_l2419_241911

/-- A configuration of four points on a plane -/
structure PointConfiguration where
  /-- The length of four segments -/
  a : ℝ
  /-- The length of the longest segment -/
  longest : ℝ
  /-- The length of the remaining segment -/
  c : ℝ
  /-- The longest segment is twice the length of a -/
  longest_eq_2a : longest = 2 * a
  /-- The configuration contains a 45-45-90 triangle -/
  has_45_45_90_triangle : True
  /-- The hypotenuse of the 45-45-90 triangle is the longest segment -/
  hypotenuse_is_longest : True
  /-- All points are distinct -/
  points_distinct : True

/-- The ratio of c to a in the given point configuration is √2 -/
theorem ratio_c_to_a_is_sqrt2 (config : PointConfiguration) : 
  config.c / config.a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_c_to_a_is_sqrt2_l2419_241911


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l2419_241990

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - x)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 512 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l2419_241990


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2419_241963

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 6}

theorem union_of_A_and_B :
  A ∪ B = {1, 2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2419_241963


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2419_241997

theorem pure_imaginary_complex_number (x : ℝ) :
  (Complex.I * (x + 1) = (x^2 - 1) + Complex.I * (x + 1)) → (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2419_241997


namespace NUMINAMATH_CALUDE_find_largest_base_l2419_241996

theorem find_largest_base (x : ℤ) (base : ℕ) :
  (x ≤ 3) →
  (2.134 * (base : ℝ) ^ (x : ℝ) < 21000) →
  (∀ y : ℤ, y ≤ 3 → 2.134 * (base : ℝ) ^ (y : ℝ) < 21000) →
  base ≤ 21 :=
sorry

end NUMINAMATH_CALUDE_find_largest_base_l2419_241996


namespace NUMINAMATH_CALUDE_triangle_angle_property_l2419_241913

theorem triangle_angle_property (α : Real) :
  (0 < α) ∧ (α < π) →  -- α is an interior angle of a triangle
  (1 / Real.sin α + 1 / Real.cos α = 2) →
  α = π + (1 / 2) * Real.arcsin ((1 - Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_property_l2419_241913


namespace NUMINAMATH_CALUDE_quadrilateral_sum_l2419_241905

/-- A quadrilateral ABCD with specific side lengths and angles -/
structure Quadrilateral :=
  (BC : ℝ)
  (CD : ℝ)
  (AD : ℝ)
  (angleA : ℝ)
  (angleB : ℝ)
  (p : ℕ)
  (q : ℕ)
  (h_BC : BC = 10)
  (h_CD : CD = 15)
  (h_AD : AD = 12)
  (h_angleA : angleA = 60)
  (h_angleB : angleB = 120)
  (h_AB : p + Real.sqrt q = AD + BC)

/-- The sum of p and q in the quadrilateral ABCD is 17 -/
theorem quadrilateral_sum (ABCD : Quadrilateral) : ABCD.p + ABCD.q = 17 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_sum_l2419_241905


namespace NUMINAMATH_CALUDE_largest_number_with_conditions_l2419_241955

def is_valid_digit (d : Nat) : Prop := d = 1 ∨ d = 2 ∨ d = 3

def digits_sum_to_13 (n : Nat) : Prop :=
  (Nat.digits 10 n).sum = 13

def all_digits_valid (n : Nat) : Prop :=
  ∀ d ∈ Nat.digits 10 n, is_valid_digit d

def is_largest_with_conditions (n : Nat) : Prop :=
  digits_sum_to_13 n ∧ all_digits_valid n ∧
  ∀ m : Nat, digits_sum_to_13 m → all_digits_valid m → m ≤ n

theorem largest_number_with_conditions :
  is_largest_with_conditions 322222 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_conditions_l2419_241955


namespace NUMINAMATH_CALUDE_nap_time_calculation_l2419_241995

/-- Calculates the remaining time for a nap given flight duration and time spent on activities --/
def time_for_nap (flight_duration : ℕ) (reading : ℕ) (movies : ℕ) (dinner : ℕ) (radio : ℕ) (games : ℕ) : ℕ :=
  flight_duration - (reading + movies + dinner + radio + games)

/-- Converts hours and minutes to minutes --/
def to_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * 60 + minutes

theorem nap_time_calculation :
  let flight_duration := to_minutes 11 20
  let reading := to_minutes 2 0
  let movies := to_minutes 4 0
  let dinner := 30
  let radio := 40
  let games := to_minutes 1 10
  let nap_time := time_for_nap flight_duration reading movies dinner radio games
  nap_time = to_minutes 3 0 := by sorry

end NUMINAMATH_CALUDE_nap_time_calculation_l2419_241995


namespace NUMINAMATH_CALUDE_min_value_of_f_for_shangmei_numbers_l2419_241977

/-- Definition of a Shangmei number -/
def isShangmeiNumber (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a + c = 11 ∧ b + d = 11

/-- Definition of function f -/
def f (n : ℕ) : ℚ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (b - d : ℚ) / (a - c)

/-- Definition of function G -/
def G (n : ℕ) : ℤ :=
  let ab := n / 100
  let cd := n % 100
  (ab : ℤ) - cd

/-- Main theorem -/
theorem min_value_of_f_for_shangmei_numbers :
  ∀ M : ℕ,
    isShangmeiNumber M →
    (M / 1000 < (M / 100) % 10) →
    (G M) % 7 = 0 →
    f M ≥ -3 ∧ ∃ M₀, isShangmeiNumber M₀ ∧ (M₀ / 1000 < (M₀ / 100) % 10) ∧ (G M₀) % 7 = 0 ∧ f M₀ = -3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_for_shangmei_numbers_l2419_241977


namespace NUMINAMATH_CALUDE_new_distance_between_cars_l2419_241933

/-- Calculates the new distance between cars in a convoy after speed reduction -/
theorem new_distance_between_cars 
  (initial_speed : ℝ) 
  (initial_distance : ℝ) 
  (reduced_speed : ℝ) 
  (h1 : initial_speed = 80) 
  (h2 : initial_distance = 10) 
  (h3 : reduced_speed = 60) : 
  (reduced_speed * (initial_distance / 1000) / initial_speed) * 1000 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_new_distance_between_cars_l2419_241933


namespace NUMINAMATH_CALUDE_button_difference_l2419_241943

theorem button_difference (sue_buttons kendra_buttons mari_buttons : ℕ) : 
  sue_buttons = 6 →
  sue_buttons = kendra_buttons / 2 →
  mari_buttons = 64 →
  mari_buttons - 5 * kendra_buttons = 4 := by
  sorry

end NUMINAMATH_CALUDE_button_difference_l2419_241943


namespace NUMINAMATH_CALUDE_expected_games_specific_scenario_l2419_241973

/-- Represents a table tennis game between two players -/
structure TableTennisGame where
  probAWins : ℝ
  aheadBy : ℕ

/-- Calculates the expected number of games in a table tennis match -/
def expectedGames (game : TableTennisGame) : ℝ :=
  sorry

/-- Theorem stating that the expected number of games in the specific scenario is 18/5 -/
theorem expected_games_specific_scenario :
  let game : TableTennisGame := ⟨2/3, 2⟩
  expectedGames game = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_expected_games_specific_scenario_l2419_241973


namespace NUMINAMATH_CALUDE_bingo_prize_distribution_l2419_241914

theorem bingo_prize_distribution (total_prize : ℝ) (first_winner_share : ℝ) (remaining_winners : ℕ) : 
  total_prize = 2400 →
  first_winner_share = total_prize / 3 →
  remaining_winners = 10 →
  (total_prize - first_winner_share) / remaining_winners = 160 := by
  sorry

end NUMINAMATH_CALUDE_bingo_prize_distribution_l2419_241914


namespace NUMINAMATH_CALUDE_min_value_expression_l2419_241919

theorem min_value_expression (a : ℝ) (ha : a > 0) : 
  ((a - 1) * (4 * a - 1)) / a ≥ -1 ∧ 
  ∃ (a₀ : ℝ), a₀ > 0 ∧ ((a₀ - 1) * (4 * a₀ - 1)) / a₀ = -1 := by
  sorry

#check min_value_expression

end NUMINAMATH_CALUDE_min_value_expression_l2419_241919


namespace NUMINAMATH_CALUDE_larger_number_proof_l2419_241900

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2419_241900


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2419_241972

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + k = 0 → y = x) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2419_241972


namespace NUMINAMATH_CALUDE_quadratic_trinomial_form_is_quadratic_trinomial_l2419_241982

-- Define variables
variable (x y : ℝ)

-- Define A, B, and C
def A : ℝ := x^2 * y + 2
def B : ℝ := 3 * x^2 * y + x
def C : ℝ := 4 * x^2 * y - x * y

-- Theorem statement
theorem quadratic_trinomial_form :
  A x y + B x y - C x y = 2 + x + x * y :=
by sorry

-- Theorem to classify the result as a quadratic trinomial
theorem is_quadratic_trinomial :
  ∃ (a b c : ℝ), A x y + B x y - C x y = a + b * x + c * x * y :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_form_is_quadratic_trinomial_l2419_241982


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2419_241915

theorem polynomial_division_remainder (p q : ℝ) : 
  (∀ x, (x^3 - 3*x^2 + 9*x - 7) = (x - p) * (ax^2 + bx + c) + (2*x + q) → p = 1 ∧ q = -2) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2419_241915


namespace NUMINAMATH_CALUDE_obtuse_angle_measure_l2419_241993

theorem obtuse_angle_measure (α β : Real) (p : Real) :
  (∃ (x y : Real), x^2 + p*(x+1) + 1 = 0 ∧ y^2 + p*(y+1) + 1 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  (α > 0 ∧ β > 0 ∧ α + β < Real.pi) →
  ∃ (γ : Real), γ = Real.pi - α - β ∧ γ = 3*Real.pi/4 :=
by sorry

end NUMINAMATH_CALUDE_obtuse_angle_measure_l2419_241993


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2419_241988

theorem quadratic_always_positive (m : ℝ) (h : m > 3) :
  ∀ x : ℝ, m * x^2 - (m + 3) * x + m > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2419_241988


namespace NUMINAMATH_CALUDE_gcd_9155_4892_l2419_241994

theorem gcd_9155_4892 : Nat.gcd 9155 4892 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9155_4892_l2419_241994


namespace NUMINAMATH_CALUDE_minimum_occupied_seats_theorem_l2419_241912

/-- Represents a row of seats -/
structure SeatRow where
  total_seats : ℕ
  occupied_seats : ℕ

/-- Checks if the next person must sit next to someone already seated -/
def next_person_sits_next (row : SeatRow) : Prop :=
  row.occupied_seats * 2 ≥ row.total_seats

/-- The theorem to be proved -/
theorem minimum_occupied_seats_theorem (row : SeatRow) 
  (h1 : row.total_seats = 180) 
  (h2 : row.occupied_seats = 90) :
  (∀ n : ℕ, n < 90 → ¬(next_person_sits_next ⟨180, n⟩)) ∧ 
  next_person_sits_next row :=
sorry

end NUMINAMATH_CALUDE_minimum_occupied_seats_theorem_l2419_241912


namespace NUMINAMATH_CALUDE_binary_op_property_l2419_241989

-- Define a binary operation on a type S
def binary_op (S : Type) := S → S → S

-- State the theorem
theorem binary_op_property {S : Type} (op : binary_op S) 
  (h : ∀ (a b : S), op (op a b) a = b) :
  ∀ (a b : S), op a (op b a) = b := by
  sorry

end NUMINAMATH_CALUDE_binary_op_property_l2419_241989


namespace NUMINAMATH_CALUDE_combined_average_score_l2419_241961

theorem combined_average_score (score_u score_b score_c : ℝ)
  (ratio_u ratio_b ratio_c : ℕ) :
  score_u = 65 →
  score_b = 80 →
  score_c = 77 →
  ratio_u = 4 →
  ratio_b = 6 →
  ratio_c = 5 →
  (score_u * ratio_u + score_b * ratio_b + score_c * ratio_c) / (ratio_u + ratio_b + ratio_c) = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_average_score_l2419_241961


namespace NUMINAMATH_CALUDE_odd_function_property_l2419_241907

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The main theorem -/
theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : IsOdd f)
  (h_even : IsEven (fun x ↦ f (x + 1)))
  (h_def : ∀ x ∈ Set.Icc 0 1, f x = x * (3 - 2 * x)) :
  f (31/2) = -1 := by
  sorry


end NUMINAMATH_CALUDE_odd_function_property_l2419_241907


namespace NUMINAMATH_CALUDE_symmetric_points_count_l2419_241930

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^2 + 4 * x + 1 else 2 / Real.exp x

-- Define symmetry about the origin
def symmetric_about_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

-- State the theorem
theorem symmetric_points_count :
  ∃ (p₁ q₁ p₂ q₂ : ℝ × ℝ),
    p₁ ≠ q₁ ∧ p₂ ≠ q₂ ∧ p₁ ≠ p₂ ∧
    symmetric_about_origin p₁ q₁ ∧
    symmetric_about_origin p₂ q₂ ∧
    (∀ x, f x = p₁.2 ↔ x = p₁.1) ∧
    (∀ x, f x = q₁.2 ↔ x = q₁.1) ∧
    (∀ x, f x = p₂.2 ↔ x = p₂.1) ∧
    (∀ x, f x = q₂.2 ↔ x = q₂.1) ∧
    (∀ p q : ℝ × ℝ, 
      p ≠ p₁ ∧ p ≠ q₁ ∧ p ≠ p₂ ∧ p ≠ q₂ ∧
      q ≠ p₁ ∧ q ≠ q₁ ∧ q ≠ p₂ ∧ q ≠ q₂ ∧
      symmetric_about_origin p q ∧
      (∀ x, f x = p.2 ↔ x = p.1) ∧
      (∀ x, f x = q.2 ↔ x = q.1) →
      False) :=
sorry

end NUMINAMATH_CALUDE_symmetric_points_count_l2419_241930


namespace NUMINAMATH_CALUDE_expression_value_l2419_241902

theorem expression_value (a b c : ℚ) (ha : a = 5) (hb : b = -3) (hc : c = 2) :
  (3 * c) / (a + b) + c = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2419_241902


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l2419_241980

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 20 → p.Prime → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 529) ∧
  (has_no_small_prime_factors 529) ∧
  (∀ m : ℕ, m < 529 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l2419_241980


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l2419_241999

theorem sqrt_x_minus_one_real (x : ℝ) (h : x = 2) : ∃ y : ℝ, y ^ 2 = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l2419_241999


namespace NUMINAMATH_CALUDE_power_division_rule_l2419_241949

theorem power_division_rule (a : ℝ) : a^3 / a^2 = a := by sorry

end NUMINAMATH_CALUDE_power_division_rule_l2419_241949


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l2419_241925

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 4)
  (h_a4 : a 4 = 2) :
  a 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l2419_241925


namespace NUMINAMATH_CALUDE_original_selling_price_l2419_241916

theorem original_selling_price (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  cost_price = 12500 ∧ discount_rate = 0.1 ∧ profit_rate = 0.08 →
  ∃ (selling_price : ℝ), selling_price = 15000 ∧
    (1 - discount_rate) * selling_price = (1 + profit_rate) * cost_price :=
by sorry

end NUMINAMATH_CALUDE_original_selling_price_l2419_241916


namespace NUMINAMATH_CALUDE_equal_radii_radii_formula_l2419_241952

/-- Two non-intersecting circles with their tangent circles -/
structure TwoCirclesWithTangents where
  /-- Center of the first circle -/
  O₁ : ℝ × ℝ
  /-- Center of the second circle -/
  O₂ : ℝ × ℝ
  /-- Radius of the first circle -/
  R₁ : ℝ
  /-- Radius of the second circle -/
  R₂ : ℝ
  /-- Distance between centers of the two circles -/
  d : ℝ
  /-- The circles are non-intersecting -/
  non_intersecting : d > R₁ + R₂
  /-- Radius of circle K₁ -/
  r₁ : ℝ
  /-- Radius of circle K₂ -/
  r₂ : ℝ
  /-- K₁ is tangent to the first circle and two rays from A₁ -/
  K₁_tangent : r₁ = (2 * R₁ * R₂) / d
  /-- K₂ is tangent to the second circle and two rays from A₂ -/
  K₂_tangent : r₂ = (2 * R₁ * R₂) / d

/-- Theorem: The radii of K₁ and K₂ are equal -/
theorem equal_radii (c : TwoCirclesWithTangents) : c.r₁ = c.r₂ := by
  sorry

/-- Theorem: The radii of K₁ and K₂ can be expressed as (2 * R₁ * R₂) / d -/
theorem radii_formula (c : TwoCirclesWithTangents) : c.r₁ = (2 * c.R₁ * c.R₂) / c.d ∧ c.r₂ = (2 * c.R₁ * c.R₂) / c.d := by
  sorry

end NUMINAMATH_CALUDE_equal_radii_radii_formula_l2419_241952


namespace NUMINAMATH_CALUDE_goose_eggs_count_l2419_241975

theorem goose_eggs_count (total_eggs : ℕ) : 
  (2 : ℚ) / 3 * (3 : ℚ) / 4 * (2 : ℚ) / 5 * total_eggs = 180 →
  total_eggs = 2700 := by
sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l2419_241975


namespace NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l2419_241965

theorem product_from_lcm_and_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 60) 
  (h2 : Nat.gcd a b = 12) : 
  a * b = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l2419_241965


namespace NUMINAMATH_CALUDE_suit_price_increase_l2419_241957

/-- Proves that the percentage increase in the price of a suit was 25% --/
theorem suit_price_increase (original_price : ℝ) (final_price : ℝ) : 
  original_price = 200 →
  final_price = 187.5 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 25 ∧
    final_price = (original_price + original_price * (increase_percentage / 100)) * 0.75 :=
by sorry

end NUMINAMATH_CALUDE_suit_price_increase_l2419_241957


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l2419_241964

theorem sqrt_expression_equality : 
  (Real.sqrt 3 + 2)^2 - (2 * Real.sqrt 3 + 3 * Real.sqrt 2) * (3 * Real.sqrt 2 - 2 * Real.sqrt 3) = 1 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l2419_241964


namespace NUMINAMATH_CALUDE_largest_solution_quadratic_equation_l2419_241987

theorem largest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => 5 * (9 * x^2 + 9 * x + 11) - x * (10 * x - 50)
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x
  ↔ 
  ∃ x : ℝ, x = (-19 + Real.sqrt 53) / 14 ∧ f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_quadratic_equation_l2419_241987


namespace NUMINAMATH_CALUDE_tree_growth_rate_consistency_l2419_241970

theorem tree_growth_rate_consistency :
  ∃ (a b : ℝ), 
    (a + b) / 2 = 0.15 ∧ 
    (1 + a) * (1 + b) = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_rate_consistency_l2419_241970


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l2419_241947

theorem divisibility_implies_equality (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l2419_241947


namespace NUMINAMATH_CALUDE_pies_sold_is_fifteen_l2419_241968

/-- Represents the number of slices in an apple pie -/
def apple_slices : ℕ := 8

/-- Represents the number of slices in a peach pie -/
def peach_slices : ℕ := 6

/-- Represents the number of apple pie slices ordered -/
def apple_orders : ℕ := 56

/-- Represents the number of peach pie slices ordered -/
def peach_orders : ℕ := 48

/-- Calculates the total number of pies sold based on the given conditions -/
def total_pies_sold : ℕ := apple_orders / apple_slices + peach_orders / peach_slices

/-- Theorem stating that the total number of pies sold is 15 -/
theorem pies_sold_is_fifteen : total_pies_sold = 15 := by
  sorry

end NUMINAMATH_CALUDE_pies_sold_is_fifteen_l2419_241968


namespace NUMINAMATH_CALUDE_complex_root_product_simplification_l2419_241956

theorem complex_root_product_simplification :
  2 * Real.sqrt 3 * 6 * (12 ^ (1/6 : ℝ)) * 3 * ((3/2) ^ (1/3 : ℝ)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_product_simplification_l2419_241956


namespace NUMINAMATH_CALUDE_no_function_satisfies_equation_l2419_241969

open Real

theorem no_function_satisfies_equation :
  ¬∃ f : ℝ → ℝ, (∀ x : ℝ, x > 0 → f x > 0) ∧
    (∀ x y : ℝ, x > 0 → y > 0 → f (x + y) = f x + f y + 1 / 2012) := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_equation_l2419_241969


namespace NUMINAMATH_CALUDE_volleyball_lineup_theorem_l2419_241901

def volleyball_lineup_count (n : ℕ) (k : ℕ) (mvp_count : ℕ) (trio_count : ℕ) : ℕ :=
  Nat.choose (n - mvp_count - trio_count) (k - mvp_count - 1) * trio_count +
  Nat.choose (n - mvp_count - trio_count) (k - mvp_count - 2) * Nat.choose trio_count 2 +
  Nat.choose (n - mvp_count - trio_count) (k - mvp_count - 3) * Nat.choose trio_count 3

theorem volleyball_lineup_theorem :
  volleyball_lineup_count 15 7 2 3 = 1035 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_theorem_l2419_241901


namespace NUMINAMATH_CALUDE_quadratic_condition_l2419_241904

/-- The equation (m+1)x^2 - mx + 1 = 0 is quadratic if and only if m ≠ -1 -/
theorem quadratic_condition (m : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, (m + 1) * x^2 - m * x + 1 = a * x^2 + b * x + c) ↔ m ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_l2419_241904


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2419_241986

theorem arithmetic_mean_of_fractions : 
  (3/4 : ℚ) + (5/8 : ℚ) / 2 = 11/16 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2419_241986


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2419_241960

theorem geometric_sequence_sum (a : ℝ) : 
  (a + 2*a + 4*a + 8*a = 1) →  -- Sum of first 4 terms equals 1
  (a + 2*a + 4*a + 8*a + 16*a + 32*a + 64*a + 128*a = 17) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2419_241960


namespace NUMINAMATH_CALUDE_find_x_l2419_241903

theorem find_x : ∃ X : ℝ, (X + 20 / 90) * 90 = 9020 ∧ X = 9000 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2419_241903
