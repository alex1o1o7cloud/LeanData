import Mathlib

namespace NUMINAMATH_CALUDE_jimmys_coffee_bean_weight_l3506_350694

/-- Proves the weight of Jimmy's coffee bean bags given the problem conditions -/
theorem jimmys_coffee_bean_weight 
  (suki_bags : ℝ) 
  (suki_weight_per_bag : ℝ) 
  (jimmy_bags : ℝ) 
  (container_weight : ℝ) 
  (num_containers : ℕ) 
  (h1 : suki_bags = 6.5)
  (h2 : suki_weight_per_bag = 22)
  (h3 : jimmy_bags = 4.5)
  (h4 : container_weight = 8)
  (h5 : num_containers = 28) :
  (↑num_containers * container_weight - suki_bags * suki_weight_per_bag) / jimmy_bags = 18 := by
  sorry

#check jimmys_coffee_bean_weight

end NUMINAMATH_CALUDE_jimmys_coffee_bean_weight_l3506_350694


namespace NUMINAMATH_CALUDE_triangle_rectangle_ratio_l3506_350634

/-- Given an equilateral triangle and a rectangle with the same perimeter,
    where the rectangle's length is twice its width, the ratio of the
    triangle's side length to the rectangle's width is 2. -/
theorem triangle_rectangle_ratio (t w : ℝ) : 
  t > 0 → w > 0 → 
  3 * t = 24 → 
  6 * w = 24 → 
  t / w = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_rectangle_ratio_l3506_350634


namespace NUMINAMATH_CALUDE_special_sequence_existence_l3506_350602

theorem special_sequence_existence : ∃ (a : ℕ → ℕ),
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, ∃ y, a n + 1 = y^2) ∧
  (∀ n, ∃ x, 3 * a n + 1 = x^2) ∧
  (∀ n, ∃ z, a n * a (n + 1) = z^2) :=
sorry

end NUMINAMATH_CALUDE_special_sequence_existence_l3506_350602


namespace NUMINAMATH_CALUDE_trajectory_of_P_l3506_350672

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 15

-- Define point N
def point_N : ℝ × ℝ := (1, 0)

-- Define the property of point M being on circle C
def point_M_on_C (M : ℝ × ℝ) : Prop := circle_C M.1 M.2

-- Define point P as the intersection of perpendicular bisector of MN and CM
def point_P (M : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  point_M_on_C M ∧ 
  -- Additional conditions for P would be defined here, but we omit the detailed geometric conditions
  True

-- State the theorem
theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, (∃ M : ℝ × ℝ, point_P M P) →
  (P.1^2 / 4 + P.2^2 / 3 = 1) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l3506_350672


namespace NUMINAMATH_CALUDE_bob_total_earnings_l3506_350669

-- Define constants
def regular_rate : ℚ := 5
def overtime_rate : ℚ := 6
def regular_hours : ℕ := 40
def first_week_hours : ℕ := 44
def second_week_hours : ℕ := 48

-- Define function to calculate weekly earnings
def weekly_earnings (hours_worked : ℕ) : ℚ :=
  let regular := min hours_worked regular_hours
  let overtime := max (hours_worked - regular_hours) 0
  regular * regular_rate + overtime * overtime_rate

-- Theorem statement
theorem bob_total_earnings :
  weekly_earnings first_week_hours + weekly_earnings second_week_hours = 472 :=
by sorry

end NUMINAMATH_CALUDE_bob_total_earnings_l3506_350669


namespace NUMINAMATH_CALUDE_unique_coefficients_sum_l3506_350612

theorem unique_coefficients_sum : 
  let y : ℝ := Real.sqrt ((Real.sqrt 75 / 3) - 5/2)
  ∃! (a b c : ℕ+), 
    y^100 = 3*y^98 + 15*y^96 + 12*y^94 - 2*y^50 + (a : ℝ)*y^46 + (b : ℝ)*y^44 + (c : ℝ)*y^40 ∧
    a + b + c = 66 := by sorry

end NUMINAMATH_CALUDE_unique_coefficients_sum_l3506_350612


namespace NUMINAMATH_CALUDE_square_root_problem_l3506_350600

theorem square_root_problem (x y z a : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (a - 3)^2 = x ∧ (2*a + 15)^2 = x) →
  y = (-3)^3 →
  z = Int.floor (Real.sqrt 13) →
  Real.sqrt (x + y - 2*z) = 4 ∨ Real.sqrt (x + y - 2*z) = -4 := by
  sorry

#check square_root_problem

end NUMINAMATH_CALUDE_square_root_problem_l3506_350600


namespace NUMINAMATH_CALUDE_work_completion_time_l3506_350609

/-- The number of days it takes for worker a to complete the work alone -/
def days_a : ℝ := 4

/-- The number of days it takes for worker b to complete the work alone -/
def days_b : ℝ := 9

/-- The number of days it takes for workers a, b, and c to complete the work together -/
def days_together : ℝ := 2

/-- The number of days it takes for worker c to complete the work alone -/
def days_c : ℝ := 7.2

theorem work_completion_time :
  (1 / days_a) + (1 / days_b) + (1 / days_c) = (1 / days_together) :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3506_350609


namespace NUMINAMATH_CALUDE_simplify_expression_l3506_350601

theorem simplify_expression (w x : ℝ) :
  3*w + 6*w + 9*w + 12*w + 15*w + 20*x + 24 = 45*w + 20*x + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3506_350601


namespace NUMINAMATH_CALUDE_boatman_downstream_distance_l3506_350673

/-- Represents the speed of a boat in various conditions -/
structure BoatSpeed where
  stationary : ℝ
  upstream : ℝ
  current : ℝ
  downstream : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating the distance traveled by the boatman along the current -/
theorem boatman_downstream_distance 
  (speed : BoatSpeed)
  (h1 : distance speed.upstream 3 = 3) -- 3 km against current in 3 hours
  (h2 : distance speed.stationary 2 = 3) -- 3 km in stationary water in 2 hours
  (h3 : speed.current = speed.stationary - speed.upstream)
  (h4 : speed.downstream = speed.stationary + speed.current) :
  distance speed.downstream 0.5 = 1 := by
  sorry

#check boatman_downstream_distance

end NUMINAMATH_CALUDE_boatman_downstream_distance_l3506_350673


namespace NUMINAMATH_CALUDE_sqrt_two_minus_two_sqrt_two_l3506_350697

theorem sqrt_two_minus_two_sqrt_two : Real.sqrt 2 - 2 * Real.sqrt 2 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_two_sqrt_two_l3506_350697


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_l3506_350649

/-- The Stewart farm problem -/
theorem stewart_farm_horse_food (sheep_count : ℕ) (total_horse_food : ℕ) 
  (sheep_to_horse_ratio : ℚ) : 
  sheep_count = 8 →
  total_horse_food = 12880 →
  sheep_to_horse_ratio = 1 / 7 →
  (total_horse_food : ℚ) / ((sheep_count : ℚ) / sheep_to_horse_ratio) = 230 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_horse_food_l3506_350649


namespace NUMINAMATH_CALUDE_power_function_alpha_l3506_350659

/-- Given a power function y = mx^α where m and α are real numbers,
    if the graph passes through the point (8, 1/4), then α equals -2/3. -/
theorem power_function_alpha (m α : ℝ) :
  (∃ (x y : ℝ), x = 8 ∧ y = 1/4 ∧ y = m * x^α) → α = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_alpha_l3506_350659


namespace NUMINAMATH_CALUDE_wire_length_for_square_field_l3506_350656

-- Define the area of the square field
def field_area : ℝ := 53824

-- Define the number of times the wire goes around the field
def num_rounds : ℕ := 10

-- Theorem statement
theorem wire_length_for_square_field :
  ∃ (side_length : ℝ),
    side_length * side_length = field_area ∧
    (4 * side_length * num_rounds : ℝ) = 9280 :=
by sorry

end NUMINAMATH_CALUDE_wire_length_for_square_field_l3506_350656


namespace NUMINAMATH_CALUDE_total_spent_is_correct_l3506_350653

def clothes_price : ℝ := 250
def clothes_discount : ℝ := 0.15
def movie_ticket_price : ℝ := 24
def movie_tickets : ℕ := 3
def movie_discount : ℝ := 0.10
def beans_price : ℝ := 1.25
def beans_quantity : ℕ := 20
def cucumber_price : ℝ := 2.50
def cucumber_quantity : ℕ := 5
def tomato_price : ℝ := 5.00
def tomato_quantity : ℕ := 3
def pineapple_price : ℝ := 6.50
def pineapple_quantity : ℕ := 2

def total_spent : ℝ := 
  clothes_price * (1 - clothes_discount) +
  (movie_ticket_price * movie_tickets) * (1 - movie_discount) +
  (beans_price * beans_quantity) +
  (cucumber_price * cucumber_quantity) +
  (tomato_price * tomato_quantity) +
  (pineapple_price * pineapple_quantity)

theorem total_spent_is_correct : total_spent = 342.80 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_correct_l3506_350653


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3506_350664

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) → 
  (3 * q^2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = -22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3506_350664


namespace NUMINAMATH_CALUDE_platform_length_l3506_350676

/-- Calculates the length of a platform given train specifications -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 700)
  (h2 : time_cross_platform = 45)
  (h3 : time_cross_pole = 15) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 1400 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l3506_350676


namespace NUMINAMATH_CALUDE_whale_sixth_hour_consumption_l3506_350655

/-- Represents the whale's feeding pattern over 9 hours -/
def WhaleFeedingPattern (x : ℕ) : List ℕ :=
  List.range 9 |>.map (fun i => x + 3 * i)

/-- The total amount of plankton consumed by the whale -/
def TotalConsumption (x : ℕ) : ℕ :=
  (WhaleFeedingPattern x).sum

theorem whale_sixth_hour_consumption :
  ∃ x : ℕ, 
    TotalConsumption x = 450 ∧ 
    (WhaleFeedingPattern x).get! 5 = 53 := by
  sorry

end NUMINAMATH_CALUDE_whale_sixth_hour_consumption_l3506_350655


namespace NUMINAMATH_CALUDE_angle_inequality_l3506_350691

theorem angle_inequality (θ : Real) : 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.sin θ - x * (1 - x) + (1 - x)^2 * Real.cos θ > 0) ↔ 
  (π / 12 < θ ∧ θ < 5 * π / 12) := by
sorry

end NUMINAMATH_CALUDE_angle_inequality_l3506_350691


namespace NUMINAMATH_CALUDE_at_most_one_greater_than_one_l3506_350637

theorem at_most_one_greater_than_one (x y : ℝ) (h : x + y < 2) :
  ¬(x > 1 ∧ y > 1) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_greater_than_one_l3506_350637


namespace NUMINAMATH_CALUDE_cookie_radius_l3506_350650

-- Define the cookie equation
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 36 = 6*x + 24*y

-- Theorem statement
theorem cookie_radius :
  ∃ (h k r : ℝ), r = Real.sqrt 117 ∧
  ∀ (x y : ℝ), cookie_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_cookie_radius_l3506_350650


namespace NUMINAMATH_CALUDE_product_195_205_l3506_350630

theorem product_195_205 : 195 * 205 = 39975 := by
  sorry

end NUMINAMATH_CALUDE_product_195_205_l3506_350630


namespace NUMINAMATH_CALUDE_complex_multiplication_sum_l3506_350674

theorem complex_multiplication_sum (a b : ℝ) (i : ℂ) : 
  (1 + i) * (2 + i) = a + b * i → i * i = -1 → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_sum_l3506_350674


namespace NUMINAMATH_CALUDE_min_workers_team_a_l3506_350690

theorem min_workers_team_a (a b : ℕ) : 
  (∃ c : ℕ, c > 0 ∧ 2 * (a - 90) = b + 90 ∧ a + c = 6 * (b - c)) →
  a ≥ 153 :=
by sorry

end NUMINAMATH_CALUDE_min_workers_team_a_l3506_350690


namespace NUMINAMATH_CALUDE_food_drive_mark_cans_l3506_350651

/-- Represents the number of cans brought by each person -/
structure Cans where
  mark : ℕ
  jaydon : ℕ
  sophie : ℕ
  rachel : ℕ

/-- Represents the conditions of the food drive -/
def FoodDrive (c : Cans) : Prop :=
  c.mark = 4 * c.jaydon ∧
  c.jaydon = 2 * c.rachel + 5 ∧
  c.mark + c.jaydon + c.sophie = 225 ∧
  4 * c.jaydon = 3 * c.mark ∧
  3 * c.sophie = 2 * c.mark

theorem food_drive_mark_cans :
  ∀ c : Cans, FoodDrive c → c.mark = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_food_drive_mark_cans_l3506_350651


namespace NUMINAMATH_CALUDE_floor_tiles_theorem_l3506_350661

/-- A rectangular floor covered with congruent square tiles. -/
structure TiledFloor where
  width : ℕ
  length : ℕ
  perimeterTiles : ℕ
  lengthTwiceWidth : length = 2 * width
  tilesAlongPerimeter : perimeterTiles = 2 * (width + length)

/-- The total number of tiles covering the floor. -/
def totalTiles (floor : TiledFloor) : ℕ :=
  floor.width * floor.length

/-- Theorem stating that a rectangular floor with 88 tiles along the perimeter
    and length twice the width has 430 tiles in total. -/
theorem floor_tiles_theorem (floor : TiledFloor) 
    (h : floor.perimeterTiles = 88) : totalTiles floor = 430 := by
  sorry

end NUMINAMATH_CALUDE_floor_tiles_theorem_l3506_350661


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l3506_350610

def numbers : List ℕ := [1877, 1999, 2039, 2045, 2119, 2131]

theorem mean_of_remaining_numbers :
  ∀ (subset : List ℕ),
    subset.length = 4 ∧
    subset ⊆ numbers ∧
    (subset.sum : ℚ) / 4 = 2015 →
    let remaining := numbers.filter (λ x => x ∉ subset)
    (remaining.sum : ℚ) / 2 = 2075 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l3506_350610


namespace NUMINAMATH_CALUDE_sprint_team_total_miles_l3506_350698

/-- Calculates the total miles run by a sprint team -/
def total_miles_run (num_people : ℝ) (miles_per_person : ℝ) : ℝ :=
  num_people * miles_per_person

/-- Proves that a sprint team of 150.0 people, each running 5.0 miles, runs a total of 750.0 miles -/
theorem sprint_team_total_miles :
  let num_people : ℝ := 150.0
  let miles_per_person : ℝ := 5.0
  total_miles_run num_people miles_per_person = 750.0 := by
  sorry

end NUMINAMATH_CALUDE_sprint_team_total_miles_l3506_350698


namespace NUMINAMATH_CALUDE_ages_product_l3506_350617

/-- Represents the ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.roy = ages.julia + 8 ∧
  ages.roy = ages.kelly + (ages.roy - ages.julia) / 2 ∧
  ages.roy + 2 = 3 * (ages.julia + 2)

/-- The theorem to be proved -/
theorem ages_product (ages : Ages) :
  satisfiesConditions ages →
  (ages.roy + 2) * (ages.kelly + 2) = 96 := by
  sorry

end NUMINAMATH_CALUDE_ages_product_l3506_350617


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3506_350668

theorem interest_rate_calculation (P : ℝ) (t : ℝ) (diff : ℝ) (r : ℝ) : 
  P = 3600 → 
  t = 2 → 
  P * ((1 + r)^t - 1) - P * r * t = diff → 
  diff = 36 → 
  r = 0.1 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3506_350668


namespace NUMINAMATH_CALUDE_probability_of_y_selection_l3506_350641

theorem probability_of_y_selection 
  (prob_x : ℝ) 
  (prob_both : ℝ) 
  (h1 : prob_x = 1/5)
  (h2 : prob_both = 0.05714285714285714) :
  prob_both / prob_x = 0.2857142857142857 := by
sorry

end NUMINAMATH_CALUDE_probability_of_y_selection_l3506_350641


namespace NUMINAMATH_CALUDE_triangle_properties_l3506_350677

/-- Given a triangle ABC with the following properties:
    1. f(x) = sin(2x + B) + √3 cos(2x + B) is an even function
    2. b = f(π/12)
    3. a = 3
    Prove that b = √3 and the area S of triangle ABC is either (3√3)/2 or (3√3)/4 -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (∀ x, Real.sin (2 * x + B) + Real.sqrt 3 * Real.cos (2 * x + B) =
        Real.sin (2 * -x + B) + Real.sqrt 3 * Real.cos (2 * -x + B)) →
  b = Real.sin (2 * (π / 12) + B) + Real.sqrt 3 * Real.cos (2 * (π / 12) + B) →
  a = 3 →
  b = Real.sqrt 3 ∧ (
    (1/2 * a * b = (3 * Real.sqrt 3) / 2) ∨
    (1/2 * a * b = (3 * Real.sqrt 3) / 4)
  ) := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3506_350677


namespace NUMINAMATH_CALUDE_antons_winning_numbers_infinite_l3506_350633

theorem antons_winning_numbers_infinite :
  ∃ f : ℕ → ℕ, Function.Injective f ∧
  ∀ k : ℕ,
    let n := f k
    ¬ ∃ m : ℕ, n = m ^ 2 ∧
    ¬ ∃ m : ℕ, (n + (n + 1)) = m ^ 2 ∧
    ∃ m : ℕ, ((n + (n + 1)) + (n + 2)) = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_antons_winning_numbers_infinite_l3506_350633


namespace NUMINAMATH_CALUDE_tangent_line_slope_l3506_350689

/-- The curve function f(x) = x³ + x + 16 -/
def f (x : ℝ) : ℝ := x^3 + x + 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_line_slope :
  ∃ a : ℝ,
    (f a = (f' a) * a) ∧  -- Point (a, f(a)) lies on the tangent line
    (f' a = 13) -- The slope of the tangent line is 13
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l3506_350689


namespace NUMINAMATH_CALUDE_max_value_of_g_l3506_350619

def g (x : ℝ) := 4 * x - x^4

theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3506_350619


namespace NUMINAMATH_CALUDE_adams_father_deposit_l3506_350680

/-- Calculates the total amount after a given number of years, given an initial deposit,
    annual interest rate, and immediate withdrawal of interest. -/
def totalAmount (initialDeposit : ℝ) (interestRate : ℝ) (years : ℝ) : ℝ :=
  initialDeposit + (initialDeposit * interestRate * years)

/-- Proves that given an initial deposit of $2000 with an 8% annual interest rate,
    where interest is withdrawn immediately upon receipt, the total amount after 2.5 years
    will be $2400. -/
theorem adams_father_deposit : totalAmount 2000 0.08 2.5 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_adams_father_deposit_l3506_350680


namespace NUMINAMATH_CALUDE_signup_theorem_l3506_350688

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of competitions --/
def num_competitions : ℕ := 3

/-- The total number of ways to sign up --/
def total_ways : ℕ := num_competitions ^ num_students

/-- The number of ways to sign up if each event has participants --/
def ways_with_all_events : ℕ := 
  (Nat.choose num_students (num_students - num_competitions)) * (Nat.factorial num_competitions)

theorem signup_theorem : 
  total_ways = 81 ∧ ways_with_all_events = 36 := by
  sorry

end NUMINAMATH_CALUDE_signup_theorem_l3506_350688


namespace NUMINAMATH_CALUDE_min_a_for_quadratic_inequality_l3506_350685

theorem min_a_for_quadratic_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 ∧ x ≤ 1/2 → x^2 + a*x + 1 ≥ 0) ∧
  (∀ (b : ℝ), (∀ (x : ℝ), x > 0 ∧ x ≤ 1/2 → x^2 + b*x + 1 ≥ 0) → b ≥ -5/2) :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_quadratic_inequality_l3506_350685


namespace NUMINAMATH_CALUDE_marble_distribution_marble_distribution_proof_l3506_350615

theorem marble_distribution (total_marbles : ℕ) (initial_group : ℕ) (joined : ℕ) : Prop :=
  total_marbles = 180 →
  initial_group = 18 →
  (total_marbles / initial_group : ℚ) - (total_marbles / (initial_group + joined) : ℚ) = 1 →
  joined = 2

-- The proof would go here, but we'll use sorry as requested
theorem marble_distribution_proof : marble_distribution 180 18 2 := by sorry

end NUMINAMATH_CALUDE_marble_distribution_marble_distribution_proof_l3506_350615


namespace NUMINAMATH_CALUDE_unique_root_of_sum_with_shift_l3506_350658

/-- Given a monic quadratic polynomial with two distinct roots, 
    prove that f(x) + f(x - √D) = 0 has exactly one root. -/
theorem unique_root_of_sum_with_shift 
  (b c : ℝ) 
  (h_distinct : ∃ (x y : ℝ), x ≠ y ∧ x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0) :
  ∃! x : ℝ, (x^2 + b*x + c) + ((x - Real.sqrt (b^2 - 4*c))^2 + b*(x - Real.sqrt (b^2 - 4*c)) + c) = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_root_of_sum_with_shift_l3506_350658


namespace NUMINAMATH_CALUDE_factors_of_81_l3506_350631

theorem factors_of_81 : Finset.card (Nat.divisors 81) = 5 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_81_l3506_350631


namespace NUMINAMATH_CALUDE_max_value_cubic_function_l3506_350622

theorem max_value_cubic_function :
  let f : ℝ → ℝ := λ x ↦ x^3 - 3*x + 1
  ∃ M : ℝ, M = 3 ∧ (∀ x ∈ Set.Icc (-3) 0, f x ≤ M) ∧ (∃ x ∈ Set.Icc (-3) 0, f x = M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_cubic_function_l3506_350622


namespace NUMINAMATH_CALUDE_total_highlighters_l3506_350684

theorem total_highlighters (yellow : ℕ) (pink : ℕ) (blue : ℕ) 
  (h1 : yellow = 7)
  (h2 : pink = yellow + 7)
  (h3 : blue = pink + 5) :
  yellow + pink + blue = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l3506_350684


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l3506_350682

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the duration of color changes -/
def changeDuration (cycle : TrafficLightCycle) : ℕ :=
  3 * 5 -- 5 seconds at the end of each color

/-- Theorem: Probability of observing a color change -/
theorem traffic_light_change_probability (cycle : TrafficLightCycle)
    (h1 : cycle.green = 45)
    (h2 : cycle.yellow = 5)
    (h3 : cycle.red = 50)
    (h4 : cycleDuration cycle = 100) :
    (changeDuration cycle : ℚ) / (cycleDuration cycle : ℚ) = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_change_probability_l3506_350682


namespace NUMINAMATH_CALUDE_egg_acceptance_ratio_l3506_350679

/-- Represents the egg processing plant scenario -/
structure EggPlant where
  total_eggs : ℕ  -- Total number of eggs processed per day
  normal_accepted : ℕ  -- Number of eggs normally accepted in a batch
  normal_rejected : ℕ  -- Number of eggs normally rejected in a batch
  additional_accepted : ℕ  -- Additional eggs accepted on the particular day

/-- Defines the conditions of the egg processing plant -/
def egg_plant_conditions (plant : EggPlant) : Prop :=
  plant.total_eggs = 400 ∧
  plant.normal_accepted = 96 ∧
  plant.normal_rejected = 4 ∧
  plant.additional_accepted = 12

/-- Calculates the ratio of accepted to rejected eggs on the particular day -/
def acceptance_ratio (plant : EggPlant) : ℚ :=
  let normal_batches := plant.total_eggs / (plant.normal_accepted + plant.normal_rejected)
  let accepted := normal_batches * plant.normal_accepted + plant.additional_accepted
  let rejected := plant.total_eggs - accepted
  accepted / rejected

/-- Theorem stating that under the given conditions, the acceptance ratio is 99:1 -/
theorem egg_acceptance_ratio (plant : EggPlant) 
  (h : egg_plant_conditions plant) : acceptance_ratio plant = 99 / 1 := by
  sorry


end NUMINAMATH_CALUDE_egg_acceptance_ratio_l3506_350679


namespace NUMINAMATH_CALUDE_cookie_batch_size_l3506_350665

theorem cookie_batch_size (batch_count : ℕ) (oatmeal_count : ℕ) (total_count : ℕ) : 
  batch_count = 2 → oatmeal_count = 4 → total_count = 10 → 
  ∃ (cookies_per_batch : ℕ), cookies_per_batch = 3 ∧ batch_count * cookies_per_batch + oatmeal_count = total_count :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_batch_size_l3506_350665


namespace NUMINAMATH_CALUDE_vowel_word_count_l3506_350618

/-- The number of times vowels A and E appear -/
def vowel_count_ae : ℕ := 6

/-- The number of times vowels I, O, and U appear -/
def vowel_count_iou : ℕ := 5

/-- The length of the words to be formed -/
def word_length : ℕ := 6

/-- The total number of vowel choices for each position -/
def total_choices : ℕ := 2 * vowel_count_ae + 3 * vowel_count_iou

/-- Theorem stating the number of possible six-letter words -/
theorem vowel_word_count : (total_choices ^ word_length : ℕ) = 531441 := by
  sorry

end NUMINAMATH_CALUDE_vowel_word_count_l3506_350618


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3506_350662

/-- The intersection of {x | x ≥ -1} and {x | -2 < x < 2} is [-1, 2) -/
theorem intersection_of_sets : 
  let M : Set ℝ := {x | x ≥ -1}
  let N : Set ℝ := {x | -2 < x ∧ x < 2}
  M ∩ N = Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3506_350662


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l3506_350695

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a + 1}

-- State the theorem
theorem set_operations_and_subset :
  (∃ a : ℝ, C a ⊆ B) →
  (Set.compl (A ∩ B) = {x : ℝ | x < 3 ∨ 6 ≤ x}) ∧
  (Set.compl B ∪ A = {x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ 9 ≤ x}) ∧
  (Set.Icc 2 8 = {a : ℝ | C a ⊆ B}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l3506_350695


namespace NUMINAMATH_CALUDE_symmetry_implies_a_eq_neg_one_l3506_350699

/-- A function f is symmetric about the line x = c if f(c + x) = f(c - x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- The main theorem -/
theorem symmetry_implies_a_eq_neg_one :
  let f := fun (x : ℝ) => Real.sin (2 * x) + a * Real.cos (2 * x)
  SymmetricAbout f (-π/8) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_a_eq_neg_one_l3506_350699


namespace NUMINAMATH_CALUDE_twentieth_term_of_arithmetic_sequence_l3506_350663

/-- Given an arithmetic sequence with first term 2 and common difference 3,
    prove that the 20th term is 59. -/
theorem twentieth_term_of_arithmetic_sequence :
  let a : ℕ → ℤ := λ n => 2 + 3 * (n - 1)
  a 20 = 59 := by sorry

end NUMINAMATH_CALUDE_twentieth_term_of_arithmetic_sequence_l3506_350663


namespace NUMINAMATH_CALUDE_yunas_grandfather_age_l3506_350613

/-- Calculates the age of Yuna's grandfather given the ages and age differences of family members. -/
def grandfatherAge (yunaAge : ℕ) (fatherAgeDiff : ℕ) (grandfatherAgeDiff : ℕ) : ℕ :=
  yunaAge + fatherAgeDiff + grandfatherAgeDiff

/-- Proves that Yuna's grandfather is 59 years old given the provided conditions. -/
theorem yunas_grandfather_age :
  grandfatherAge 9 27 23 = 59 := by
  sorry

#eval grandfatherAge 9 27 23

end NUMINAMATH_CALUDE_yunas_grandfather_age_l3506_350613


namespace NUMINAMATH_CALUDE_range_of_a_l3506_350696

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -|p.1| - 2}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = a^2}

theorem range_of_a (a : ℝ) :
  (set_A ∩ set_B a = ∅) ↔ (-2*Real.sqrt 2 - 2 < a ∧ a < 2*Real.sqrt 2 + 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3506_350696


namespace NUMINAMATH_CALUDE_brick_wall_theorem_l3506_350606

/-- Represents a brick wall with a given number of rows, total bricks, and bricks in the bottom row. -/
structure BrickWall where
  rows : ℕ
  totalBricks : ℕ
  bottomRowBricks : ℕ

/-- Calculates the number of bricks in a given row of the wall. -/
def bricksInRow (wall : BrickWall) (rowNumber : ℕ) : ℕ :=
  wall.bottomRowBricks - (rowNumber - 1)

theorem brick_wall_theorem (wall : BrickWall) 
    (h1 : wall.rows = 5)
    (h2 : wall.totalBricks = 100)
    (h3 : wall.bottomRowBricks = 18) :
    ∀ (r : ℕ), 1 < r ∧ r ≤ wall.rows → 
    bricksInRow wall r = bricksInRow wall (r - 1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_brick_wall_theorem_l3506_350606


namespace NUMINAMATH_CALUDE_average_study_time_difference_l3506_350666

def daily_differences : List Int := [10, -10, 20, 30, -20]

theorem average_study_time_difference : 
  (daily_differences.sum : ℚ) / daily_differences.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l3506_350666


namespace NUMINAMATH_CALUDE_circle_intersection_properties_l3506_350647

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 + 8 = 0}

-- Define point P
def P : ℝ × ℝ := (0, 1)

-- Define point Q
def Q : ℝ × ℝ := (6, 4)

-- Define the center of the circle
def center : ℝ × ℝ := (3, 0)

-- Define a line passing through P with slope k
def line_through_P (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 + 1}

-- Define the function for the equation of line PC
def line_PC (p : ℝ × ℝ) : ℝ := p.1 + 3 * p.2 - 3

-- State the theorem
theorem circle_intersection_properties :
  -- 1) The equation of line PC
  (∀ p, p ∈ C → line_PC p = 0) ∧
  -- 2) The range of slope k
  (∀ k, (∃ A B, A ≠ B ∧ A ∈ C ∧ B ∈ C ∧ A ∈ line_through_P k ∧ B ∈ line_through_P k) ↔ -3/4 < k ∧ k < 0) ∧
  -- 3) Non-existence of perpendicular bisector through Q
  (¬∃ k₁, ∃ A B, A ≠ B ∧ A ∈ C ∧ B ∈ C ∧
    (∃ k, A ∈ line_through_P k ∧ B ∈ line_through_P k) ∧
    Q ∈ {p | p.2 - 4 = k₁ * (p.1 - 6)} ∧
    (A.1 + B.1) / 2 = (Q.1 + center.1) / 2 ∧
    (A.2 + B.2) / 2 = (Q.2 + center.2) / 2 ∧
    k₁ * k = -1) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_properties_l3506_350647


namespace NUMINAMATH_CALUDE_profit_calculation_l3506_350667

/-- The profit calculation for a product with given purchase price, markup percentage, and discount. -/
theorem profit_calculation (purchase_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) :
  purchase_price = 200 →
  markup_percent = 1.25 →
  discount_percent = 0.9 →
  purchase_price * markup_percent * discount_percent - purchase_price = 25 := by
  sorry

#check profit_calculation

end NUMINAMATH_CALUDE_profit_calculation_l3506_350667


namespace NUMINAMATH_CALUDE_quadratic_equation_rewrite_l3506_350616

theorem quadratic_equation_rewrite (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → 
  b + c = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_rewrite_l3506_350616


namespace NUMINAMATH_CALUDE_work_completion_time_l3506_350642

/-- The number of days it takes to complete the remaining work after additional persons join -/
def remaining_days (initial_persons : ℕ) (total_days : ℕ) (days_worked : ℕ) (additional_persons : ℕ) : ℚ :=
  let initial_work_rate := 1 / (initial_persons * total_days : ℚ)
  let work_done := initial_persons * days_worked * initial_work_rate
  let remaining_work := 1 - work_done
  let new_work_rate := (initial_persons + additional_persons : ℚ) * initial_work_rate
  remaining_work / new_work_rate

theorem work_completion_time :
  remaining_days 12 18 6 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3506_350642


namespace NUMINAMATH_CALUDE_evaluate_expression_l3506_350626

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 1) :
  y^2 * (y - 4*x) = -7 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3506_350626


namespace NUMINAMATH_CALUDE_f_4_has_eight_zeros_l3506_350632

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Define the recursive function f_n
def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => λ x => f (f_n n x)

-- State the theorem
theorem f_4_has_eight_zeros :
  ∃! (zeros : Finset ℝ), zeros.card = 8 ∧ ∀ x ∈ zeros, f_n 4 x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_4_has_eight_zeros_l3506_350632


namespace NUMINAMATH_CALUDE_chemists_self_receipts_l3506_350604

/-- Represents a chemist in the laboratory -/
structure Chemist where
  id : Nat
  reagents : Finset Nat

/-- Represents the state of the laboratory -/
structure Laboratory where
  chemists : Finset Chemist
  num_chemists : Nat

/-- Checks if a chemist has received all reagents -/
def has_all_reagents (c : Chemist) (lab : Laboratory) : Prop :=
  c.reagents.card = lab.num_chemists

/-- Checks if no chemist has received any reagent more than once -/
def no_double_receipts (lab : Laboratory) : Prop :=
  ∀ c ∈ lab.chemists, ∀ r ∈ c.reagents, (c.reagents.filter (λ x => x = r)).card ≤ 1

/-- Counts the number of chemists who received their own reagent -/
def count_self_receipts (lab : Laboratory) : Nat :=
  (lab.chemists.filter (λ c => c.id ∈ c.reagents)).card

/-- The main theorem to be proved -/
theorem chemists_self_receipts (lab : Laboratory) 
  (h1 : ∀ c ∈ lab.chemists, has_all_reagents c lab)
  (h2 : no_double_receipts lab) :
  count_self_receipts lab ≥ lab.num_chemists - 1 :=
sorry

end NUMINAMATH_CALUDE_chemists_self_receipts_l3506_350604


namespace NUMINAMATH_CALUDE_units_digit_G_100_l3506_350611

/-- The sequence G_n defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem stating that the units digit of G_100 is 4 -/
theorem units_digit_G_100 : unitsDigit (G 100) = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_G_100_l3506_350611


namespace NUMINAMATH_CALUDE_solution_set_f_leq_3x_plus_4_range_of_m_for_f_geq_m_all_reals_l3506_350621

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- Theorem for the solution set of f(x) ≤ 3x + 4
theorem solution_set_f_leq_3x_plus_4 :
  {x : ℝ | f x ≤ 3 * x + 4} = {x : ℝ | x ≥ 0} :=
sorry

-- Theorem for the range of m
theorem range_of_m_for_f_geq_m_all_reals (m : ℝ) :
  ({x : ℝ | f x ≥ m} = Set.univ) ↔ m ∈ Set.Iic 4 :=
sorry

#check solution_set_f_leq_3x_plus_4
#check range_of_m_for_f_geq_m_all_reals

end NUMINAMATH_CALUDE_solution_set_f_leq_3x_plus_4_range_of_m_for_f_geq_m_all_reals_l3506_350621


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3506_350681

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3506_350681


namespace NUMINAMATH_CALUDE_white_squares_20th_row_l3506_350624

/-- Represents the number of squares in a row of the stair-step figure -/
def squares_in_row (n : ℕ) : ℕ := 2 * n + 1

/-- Represents the number of white squares in a row of the stair-step figure -/
def white_squares_in_row (n : ℕ) : ℕ := (squares_in_row n - 1) / 2

theorem white_squares_20th_row :
  white_squares_in_row 20 = 20 := by
  sorry

#eval white_squares_in_row 20

end NUMINAMATH_CALUDE_white_squares_20th_row_l3506_350624


namespace NUMINAMATH_CALUDE_total_inches_paved_before_today_l3506_350620

/-- Represents a road section with its length and completion percentage -/
structure RoadSection where
  length : ℝ
  percentComplete : ℝ

/-- Calculates the total inches repaved before today given three road sections and additional inches repaved today -/
def totalInchesPavedBeforeToday (sectionA sectionB sectionC : RoadSection) (additionalInches : ℝ) : ℝ :=
  sectionA.length * sectionA.percentComplete +
  sectionB.length * sectionB.percentComplete +
  sectionC.length * sectionC.percentComplete

/-- Theorem stating that the total inches repaved before today is 6900 -/
theorem total_inches_paved_before_today :
  let sectionA : RoadSection := { length := 4000, percentComplete := 0.7 }
  let sectionB : RoadSection := { length := 3500, percentComplete := 0.6 }
  let sectionC : RoadSection := { length := 2500, percentComplete := 0.8 }
  let additionalInches : ℝ := 950
  totalInchesPavedBeforeToday sectionA sectionB sectionC additionalInches = 6900 := by
  sorry

end NUMINAMATH_CALUDE_total_inches_paved_before_today_l3506_350620


namespace NUMINAMATH_CALUDE_days_2000_to_2005_l3506_350652

/-- The number of days in a given range of years -/
def totalDays (totalYears : ℕ) (leapYears : ℕ) (nonLeapDays : ℕ) (leapDays : ℕ) : ℕ :=
  (totalYears - leapYears) * nonLeapDays + leapYears * leapDays

/-- Theorem stating that the total number of days from 2000 to 2005 (inclusive) is 2192 -/
theorem days_2000_to_2005 : totalDays 6 2 365 366 = 2192 := by
  sorry

end NUMINAMATH_CALUDE_days_2000_to_2005_l3506_350652


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_8_mod_17_l3506_350654

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∀ n : ℕ, n ≥ 10000 ∧ n ≡ 8 [MOD 17] → n ≥ 10004 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_8_mod_17_l3506_350654


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3506_350648

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ x : ℝ, x > 1 → x > a) ∧ (∃ x : ℝ, x > a ∧ x ≤ 1) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3506_350648


namespace NUMINAMATH_CALUDE_income_calculation_l3506_350640

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 4 = expenditure * 5 →
  income - expenditure = savings →
  savings = 3400 →
  income = 17000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l3506_350640


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3506_350692

-- Define arithmetic sequence
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

-- Define geometric sequence
def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem arithmetic_geometric_sequence_ratio :
  ∀ (x y a b c : ℝ),
  is_arithmetic_sequence 1 x y 4 →
  is_geometric_sequence (-2) a b c (-8) →
  (y - x) / b = -1/4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3506_350692


namespace NUMINAMATH_CALUDE_bounded_expression_l3506_350645

theorem bounded_expression (x y : ℝ) :
  -1/2 ≤ ((x + y) * (1 - x * y)) / ((1 + x^2) * (1 + y^2)) ∧
  ((x + y) * (1 - x * y)) / ((1 + x^2) * (1 + y^2)) ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_bounded_expression_l3506_350645


namespace NUMINAMATH_CALUDE_derivative_at_two_l3506_350693

-- Define f as a real-valued function
variable (f : ℝ → ℝ)

-- Define the conditions
def tangent_coincide (f : ℝ → ℝ) : Prop :=
  ∃ (m : ℝ), (∀ x, x ≠ 0 → (f x) / x - 1 = m * (x - 2)) ∧
             (∀ x, f x = m * x)

-- Theorem statement
theorem derivative_at_two (f : ℝ → ℝ) 
  (h1 : tangent_coincide f) 
  (h2 : f 0 = 0) :
  deriv f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_two_l3506_350693


namespace NUMINAMATH_CALUDE_nancy_spent_95_40_l3506_350627

/-- The total amount Nancy spends on beads -/
def total_spent (crystal_price metal_price : ℚ) (crystal_sets metal_sets : ℕ) 
  (crystal_discount metal_tax : ℚ) : ℚ :=
  let crystal_cost := crystal_price * crystal_sets
  let metal_cost := metal_price * metal_sets
  let discounted_crystal := crystal_cost * (1 - crystal_discount)
  let taxed_metal := metal_cost * (1 + metal_tax)
  discounted_crystal + taxed_metal

/-- Theorem: Nancy spends $95.40 on beads -/
theorem nancy_spent_95_40 : 
  total_spent 12 15 3 4 (1/10) (1/20) = 95.4 := by
  sorry

end NUMINAMATH_CALUDE_nancy_spent_95_40_l3506_350627


namespace NUMINAMATH_CALUDE_inverse_proportion_m_value_l3506_350608

-- Define the function y as an inverse proportion function
def is_inverse_proportion (m : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → (m - 2) * x^(m^2 - 5) = k / x

-- State the theorem
theorem inverse_proportion_m_value :
  ∀ m : ℝ, is_inverse_proportion m → m - 2 ≠ 0 → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_m_value_l3506_350608


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3506_350643

theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 - 6*m*x + 2*m = 0) → m = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3506_350643


namespace NUMINAMATH_CALUDE_largest_circle_equation_l3506_350670

/-- The standard equation of the circle with the largest area, centered at (2, -3) and tangent to the line 2mx-y-2m-1=0 (m ∈ ℝ) -/
theorem largest_circle_equation (m : ℝ) : 
  ∃ (x y : ℝ), (x - 2)^2 + (y + 3)^2 = 5 ∧ 
  ∀ (x' y' r : ℝ), 
    ((x' - 2)^2 + (y' + 3)^2 = r^2) → 
    (2*m*x' - y' - 2*m - 1 = 0) → 
    r^2 ≤ 5 := by
  sorry

#check largest_circle_equation

end NUMINAMATH_CALUDE_largest_circle_equation_l3506_350670


namespace NUMINAMATH_CALUDE_dance_class_permutations_l3506_350657

theorem dance_class_permutations :
  Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_dance_class_permutations_l3506_350657


namespace NUMINAMATH_CALUDE_nth_row_equation_l3506_350605

theorem nth_row_equation (n : ℕ) : 2 * n + 1 = (n + 1)^2 - n^2 := by
  sorry

end NUMINAMATH_CALUDE_nth_row_equation_l3506_350605


namespace NUMINAMATH_CALUDE_band_formation_proof_l3506_350628

/-- Represents the number of columns in the rectangular formation -/
def n : ℕ := 14

/-- The total number of band members -/
def total_members : ℕ := n * (n + 7)

/-- The side length of the square formation -/
def square_side : ℕ := 17

theorem band_formation_proof :
  -- Square formation condition
  total_members = square_side ^ 2 + 5 ∧
  -- Rectangular formation condition
  total_members = n * (n + 7) ∧
  -- Maximum number of members
  total_members = 294 ∧
  -- No larger n satisfies the conditions
  ∀ m : ℕ, m > n → ¬(∃ k : ℕ, m * (m + 7) = k ^ 2 + 5) :=
by sorry

end NUMINAMATH_CALUDE_band_formation_proof_l3506_350628


namespace NUMINAMATH_CALUDE_factorial_four_div_one_l3506_350644

/-- Definition of factorial for natural numbers -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem stating that 4! / (4 - 3)! = 24 -/
theorem factorial_four_div_one : factorial 4 / factorial (4 - 3) = 24 := by
  sorry

end NUMINAMATH_CALUDE_factorial_four_div_one_l3506_350644


namespace NUMINAMATH_CALUDE_trilandia_sentinel_sites_l3506_350603

/-- Represents a triangular city with streets and sentinel sites. -/
structure TriangularCity where
  side_length : ℕ
  num_streets : ℕ

/-- Calculates the minimum number of sentinel sites required for a given triangular city. -/
def min_sentinel_sites (city : TriangularCity) : ℕ :=
  3 * (city.side_length / 2) - 1

/-- Theorem stating the minimum number of sentinel sites for Trilandia. -/
theorem trilandia_sentinel_sites :
  let trilandia : TriangularCity := ⟨2012, 6036⟩
  min_sentinel_sites trilandia = 3017 := by
  sorry

#eval min_sentinel_sites ⟨2012, 6036⟩

end NUMINAMATH_CALUDE_trilandia_sentinel_sites_l3506_350603


namespace NUMINAMATH_CALUDE_triangle_inequality_l3506_350614

theorem triangle_inequality (a b c : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : a + b + c = 1) (h8 : n ≥ 2) :
  (a^n + b^n)^(1/n) + (b^n + c^n)^(1/n) + (c^n + a^n)^(1/n) < 1 + (2^(1/n))/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3506_350614


namespace NUMINAMATH_CALUDE_cube_side_area_l3506_350687

theorem cube_side_area (V : ℝ) (s : ℝ) (h : V = 125) :
  V = s^3 → s^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_area_l3506_350687


namespace NUMINAMATH_CALUDE_yang_hui_problem_l3506_350686

theorem yang_hui_problem : ∃ (x : ℕ), 
  (x % 2 = 1) ∧ 
  (x % 5 = 2) ∧ 
  (x % 7 = 3) ∧ 
  (x % 9 = 4) ∧ 
  (∀ y : ℕ, y < x → ¬((y % 2 = 1) ∧ (y % 5 = 2) ∧ (y % 7 = 3) ∧ (y % 9 = 4))) ∧
  x = 157 :=
by sorry

end NUMINAMATH_CALUDE_yang_hui_problem_l3506_350686


namespace NUMINAMATH_CALUDE_percentage_problem_l3506_350629

theorem percentage_problem (x : ℝ) : 
  (40 * x / 100) + (25 / 100 * 60) = 23 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3506_350629


namespace NUMINAMATH_CALUDE_matrix_and_transformation_problem_l3506_350646

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; -2, -3]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, -2]
def C (x y : ℝ) : Prop := x^2 - 4*x*y + y^2 = 1

theorem matrix_and_transformation_problem :
  ∃ (A_inv X : Matrix (Fin 2) (Fin 2) ℝ) (C' : ℝ → ℝ → Prop),
    (A_inv = !![(-3), (-2); 2, 1]) ∧
    (A * A_inv = 1) ∧ (A_inv * A = 1) ∧
    (X = !![(-2), 1; 1, 0]) ∧
    (A * X = B) ∧
    (∀ x y x' y', C x y ∧ x' = y ∧ y' = x - 2*y → C' x' y') ∧
    (∀ x' y', C' x' y' ↔ 3*x'^2 - y'^2 = -1) := by
  sorry

end NUMINAMATH_CALUDE_matrix_and_transformation_problem_l3506_350646


namespace NUMINAMATH_CALUDE_arithmetic_sum_l3506_350683

theorem arithmetic_sum : 5 * 12 + 7 * 9 + 8 * 4 + 6 * 7 + 2 * 13 = 223 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_l3506_350683


namespace NUMINAMATH_CALUDE_new_bill_is_35_l3506_350635

/-- Calculates the new total bill after substitutions and delivery/tip --/
def calculate_new_bill (original_order : ℝ) 
                       (tomato_old tomato_new : ℝ) 
                       (lettuce_old lettuce_new : ℝ) 
                       (celery_old celery_new : ℝ) 
                       (delivery_tip : ℝ) : ℝ :=
  original_order + 
  (tomato_new - tomato_old) + 
  (lettuce_new - lettuce_old) + 
  (celery_new - celery_old) + 
  delivery_tip

/-- Theorem stating that the new bill is $35.00 --/
theorem new_bill_is_35 : 
  calculate_new_bill 25 0.99 2.20 1.00 1.75 1.96 2.00 8.00 = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_new_bill_is_35_l3506_350635


namespace NUMINAMATH_CALUDE_modulus_of_z_l3506_350638

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3506_350638


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l3506_350675

theorem shaded_area_theorem (square_side : ℝ) (h : square_side = 12) :
  let triangle_base : ℝ := square_side * 3 / 4
  let triangle_height : ℝ := square_side / 4
  triangle_base * triangle_height / 2 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l3506_350675


namespace NUMINAMATH_CALUDE_perimeter_quadrilateral_l3506_350607

/-- The perimeter of a quadrilateral PQRS with given coordinates can be expressed as x√3 + y√10, where x + y = 12 -/
theorem perimeter_quadrilateral (P Q R S : ℝ × ℝ) : 
  P = (1, 2) → Q = (3, 6) → R = (6, 3) → S = (8, 1) →
  ∃ (x y : ℤ), 
    (Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) +
     Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) +
     Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) +
     Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) =
     x * Real.sqrt 3 + y * Real.sqrt 10) ∧
    x + y = 12 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_quadrilateral_l3506_350607


namespace NUMINAMATH_CALUDE_units_produced_l3506_350671

def fixed_costs : ℕ := 15000
def variable_cost_per_unit : ℕ := 300
def total_cost : ℕ := 27500

def total_cost_function (n : ℕ) : ℕ :=
  fixed_costs + n * variable_cost_per_unit

theorem units_produced : ∃ (n : ℕ), n > 0 ∧ n ≤ 50 ∧ total_cost_function n = total_cost :=
sorry

end NUMINAMATH_CALUDE_units_produced_l3506_350671


namespace NUMINAMATH_CALUDE_pyramid_volume_l3506_350660

/-- The volume of a pyramid with specific properties -/
theorem pyramid_volume (base_angle : Real) (lateral_edge : Real) (inclination : Real) : 
  base_angle = π/8 →
  lateral_edge = Real.sqrt 6 →
  inclination = 5*π/13 →
  ∃ (volume : Real), 
    volume = Real.sqrt 3 * Real.sin (10*π/13) * Real.cos (5*π/13) ∧
    volume = (1/3) * 
             ((lateral_edge * Real.cos inclination)^2 * Real.sin (2*base_angle)) * 
             (lateral_edge * Real.sin inclination) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3506_350660


namespace NUMINAMATH_CALUDE_vector_basis_range_l3506_350639

/-- Two vectors form a basis of a 2D plane if they are linearly independent -/
def is_basis (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 ≠ a.2 * b.1

/-- The range of m for which (1,2) and (m,3m-2) form a basis -/
theorem vector_basis_range :
  ∀ m : ℝ, is_basis (1, 2) (m, 3*m-2) ↔ m ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_vector_basis_range_l3506_350639


namespace NUMINAMATH_CALUDE_tom_payment_l3506_350678

/-- The amount Tom paid to the shopkeeper -/
def total_amount (apple_quantity apple_rate mango_quantity mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Proof that Tom paid 1055 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 55 = 1055 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_l3506_350678


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3506_350636

theorem necessary_but_not_sufficient (a b : ℝ) :
  (∀ x y : ℝ, x * y ≠ 0 → x ≠ 0) ∧
  ¬(∀ x y : ℝ, x ≠ 0 → x * y ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3506_350636


namespace NUMINAMATH_CALUDE_sony_games_to_give_away_l3506_350625

theorem sony_games_to_give_away (current_sony_games : ℕ) (target_sony_games : ℕ) :
  current_sony_games = 132 →
  target_sony_games = 31 →
  current_sony_games - target_sony_games = 101 :=
by
  sorry

#check sony_games_to_give_away

end NUMINAMATH_CALUDE_sony_games_to_give_away_l3506_350625


namespace NUMINAMATH_CALUDE_cubic_sum_ratio_l3506_350623

theorem cubic_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : x + y + z = 30) 
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) : 
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_ratio_l3506_350623
