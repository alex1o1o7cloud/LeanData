import Mathlib

namespace NUMINAMATH_CALUDE_min_values_xy_l223_22370

/-- Given positive real numbers x and y satisfying lg x + lg y = lg(x + y + 3),
    prove that the minimum value of xy is 9 and the minimum value of x + y is 6. -/
theorem min_values_xy (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log x + Real.log y = Real.log (x + y + 3)) :
  (∀ a b, a > 0 → b > 0 → Real.log a + Real.log b = Real.log (a + b + 3) → x * y ≤ a * b) ∧
  (∀ a b, a > 0 → b > 0 → Real.log a + Real.log b = Real.log (a + b + 3) → x + y ≤ a + b) ∧
  x * y = 9 ∧ x + y = 6 :=
sorry

end NUMINAMATH_CALUDE_min_values_xy_l223_22370


namespace NUMINAMATH_CALUDE_product_remainder_remainder_proof_l223_22360

theorem product_remainder (a b m : ℕ) (h : m > 0) : (a * b) % m = ((a % m) * (b % m)) % m := by sorry

theorem remainder_proof : (1023 * 999999) % 139 = 32 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_product_remainder_remainder_proof_l223_22360


namespace NUMINAMATH_CALUDE_solutions_count_no_solutions_when_a_less_than_neg_one_one_solution_when_a_eq_neg_one_two_solutions_when_a_between_neg_one_and_zero_one_solution_when_a_greater_than_zero_l223_22381

/-- The number of real solutions to the equation √(3a - 2x) + x = a depends on the value of a -/
theorem solutions_count (a : ℝ) :
  (∀ x, ¬ (Real.sqrt (3 * a - 2 * x) + x = a)) ∨
  (∃! x, Real.sqrt (3 * a - 2 * x) + x = a) ∨
  (∃ x y, x ≠ y ∧ Real.sqrt (3 * a - 2 * x) + x = a ∧ Real.sqrt (3 * a - 2 * y) + y = a) :=
by
  sorry

/-- For a < -1, there are no real solutions -/
theorem no_solutions_when_a_less_than_neg_one (a : ℝ) (h : a < -1) :
  ∀ x, ¬ (Real.sqrt (3 * a - 2 * x) + x = a) :=
by
  sorry

/-- For a = -1, there is exactly one real solution -/
theorem one_solution_when_a_eq_neg_one (a : ℝ) (h : a = -1) :
  ∃! x, Real.sqrt (3 * a - 2 * x) + x = a :=
by
  sorry

/-- For -1 < a ≤ 0, there are exactly two real solutions -/
theorem two_solutions_when_a_between_neg_one_and_zero (a : ℝ) (h1 : -1 < a) (h2 : a ≤ 0) :
  ∃ x y, x ≠ y ∧ Real.sqrt (3 * a - 2 * x) + x = a ∧ Real.sqrt (3 * a - 2 * y) + y = a :=
by
  sorry

/-- For a > 0, there is exactly one real solution -/
theorem one_solution_when_a_greater_than_zero (a : ℝ) (h : a > 0) :
  ∃! x, Real.sqrt (3 * a - 2 * x) + x = a :=
by
  sorry

end NUMINAMATH_CALUDE_solutions_count_no_solutions_when_a_less_than_neg_one_one_solution_when_a_eq_neg_one_two_solutions_when_a_between_neg_one_and_zero_one_solution_when_a_greater_than_zero_l223_22381


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l223_22384

/-- A box containing colored balls -/
structure ColoredBallBox where
  redBalls : ℕ
  yellowBalls : ℕ

/-- The probability of drawing a yellow ball from a box -/
def probabilityYellowBall (box : ColoredBallBox) : ℚ :=
  box.yellowBalls / (box.redBalls + box.yellowBalls)

/-- Theorem: The probability of drawing a yellow ball from a box with 3 red and 2 yellow balls is 2/5 -/
theorem yellow_ball_probability :
  let box : ColoredBallBox := ⟨3, 2⟩
  probabilityYellowBall box = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_yellow_ball_probability_l223_22384


namespace NUMINAMATH_CALUDE_correct_seasons_before_announcement_l223_22318

/-- The number of seasons before the announcement of a TV show. -/
def seasons_before_announcement : ℕ := 9

/-- The number of episodes in a regular season. -/
def regular_season_episodes : ℕ := 22

/-- The number of episodes in the last season. -/
def last_season_episodes : ℕ := 26

/-- The duration of each episode in hours. -/
def episode_duration : ℚ := 1/2

/-- The total watch time for all episodes in hours. -/
def total_watch_time : ℕ := 112

theorem correct_seasons_before_announcement :
  seasons_before_announcement * regular_season_episodes + last_season_episodes =
  total_watch_time / (episode_duration : ℚ) := by sorry

end NUMINAMATH_CALUDE_correct_seasons_before_announcement_l223_22318


namespace NUMINAMATH_CALUDE_hawks_score_l223_22348

/-- Represents a basketball game between two teams -/
structure BasketballGame where
  total_score : ℕ
  winning_margin : ℕ

/-- Calculates the score of the losing team in a basketball game -/
def losing_team_score (game : BasketballGame) : ℕ :=
  (game.total_score - game.winning_margin) / 2

theorem hawks_score (game : BasketballGame) 
  (h1 : game.total_score = 82) 
  (h2 : game.winning_margin = 6) : 
  losing_team_score game = 38 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l223_22348


namespace NUMINAMATH_CALUDE_tan_neg_seven_pi_sixths_l223_22304

theorem tan_neg_seven_pi_sixths : 
  Real.tan (-7 * π / 6) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_seven_pi_sixths_l223_22304


namespace NUMINAMATH_CALUDE_history_or_geography_not_both_count_l223_22301

/-- The number of students taking both history and geography -/
def both : ℕ := 15

/-- The number of students taking history -/
def history : ℕ := 30

/-- The number of students taking geography only -/
def geography_only : ℕ := 12

/-- The number of students taking history or geography but not both -/
def history_or_geography_not_both : ℕ := (history - both) + geography_only

theorem history_or_geography_not_both_count : history_or_geography_not_both = 27 := by
  sorry

end NUMINAMATH_CALUDE_history_or_geography_not_both_count_l223_22301


namespace NUMINAMATH_CALUDE_intersection_circle_passes_through_zero_one_l223_22395

/-- A parabola that intersects the coordinate axes at three distinct points -/
structure TripleIntersectingParabola where
  a : ℝ
  b : ℝ
  distinct_intersections : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁^2 + a*x₁ + b = 0 ∧ x₂^2 + a*x₂ + b = 0 ∧ b ≠ 0

/-- The circle passing through the three intersection points of the parabola with the coordinate axes -/
def intersection_circle (p : TripleIntersectingParabola) : Set (ℝ × ℝ) :=
  { point | ∃ (center : ℝ × ℝ) (radius : ℝ),
    (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2 ∧
    (0 - center.1)^2 + (p.b - center.2)^2 = radius^2 ∧
    ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
    x₁^2 + p.a*x₁ + p.b = 0 ∧ x₂^2 + p.a*x₂ + p.b = 0 ∧
    (x₁ - center.1)^2 + (0 - center.2)^2 = radius^2 ∧
    (x₂ - center.1)^2 + (0 - center.2)^2 = radius^2 }

/-- The main theorem stating that the intersection circle passes through (0,1) -/
theorem intersection_circle_passes_through_zero_one (p : TripleIntersectingParabola) :
  (0, 1) ∈ intersection_circle p := by
  sorry

end NUMINAMATH_CALUDE_intersection_circle_passes_through_zero_one_l223_22395


namespace NUMINAMATH_CALUDE_wheel_distance_l223_22336

/-- Proves that a wheel rotating 10 times per minute and moving 20 cm per rotation will move 12000 cm in 1 hour -/
theorem wheel_distance (rotations_per_minute : ℕ) (cm_per_rotation : ℕ) (minutes_per_hour : ℕ) :
  rotations_per_minute = 10 →
  cm_per_rotation = 20 →
  minutes_per_hour = 60 →
  rotations_per_minute * minutes_per_hour * cm_per_rotation = 12000 := by
  sorry

#check wheel_distance

end NUMINAMATH_CALUDE_wheel_distance_l223_22336


namespace NUMINAMATH_CALUDE_picnic_attendance_l223_22377

/-- Represents the number of people at a picnic -/
structure PicnicAttendance where
  men : ℕ
  women : ℕ
  adults : ℕ
  children : ℕ

/-- Theorem: Given the conditions of the picnic, the total number of attendees is 240 -/
theorem picnic_attendance (p : PicnicAttendance) 
  (h1 : p.men = p.women + 40)
  (h2 : p.adults = p.children + 40)
  (h3 : p.men = 90)
  : p.men + p.women + p.children = 240 := by
  sorry

#check picnic_attendance

end NUMINAMATH_CALUDE_picnic_attendance_l223_22377


namespace NUMINAMATH_CALUDE_sum_100_from_neg_49_l223_22351

/-- Sum of consecutive integers -/
def sum_consecutive_integers (start : Int) (count : Nat) : Int :=
  count * (2 * start + count.pred) / 2

/-- Theorem: Sum of 100 consecutive integers from -49 is 50 -/
theorem sum_100_from_neg_49 : sum_consecutive_integers (-49) 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_100_from_neg_49_l223_22351


namespace NUMINAMATH_CALUDE_area_of_sine_curve_l223_22376

theorem area_of_sine_curve (f : ℝ → ℝ) (a b : ℝ) : 
  (f = λ x => Real.sin x) →
  (a = -π/2) →
  (b = 5*π/4) →
  (∫ x in a..b, |f x| ) = 4 - Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_area_of_sine_curve_l223_22376


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l223_22382

noncomputable def f (x : ℝ) := 2 * x^3 - 2 * x^2 + 3

theorem f_derivative_at_2 : 
  (deriv f) 2 = 16 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l223_22382


namespace NUMINAMATH_CALUDE_reciprocal_absolute_value_l223_22329

theorem reciprocal_absolute_value (x : ℝ) : 
  (1 / |x|) = -4 → x = 1/4 ∨ x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_absolute_value_l223_22329


namespace NUMINAMATH_CALUDE_steve_socks_l223_22314

theorem steve_socks (total_socks : ℕ) (matching_pairs : ℕ) (mismatching_socks : ℕ) : 
  total_socks = 25 → matching_pairs = 4 → mismatching_socks = total_socks - 2 * matching_pairs →
  mismatching_socks = 17 := by
sorry

end NUMINAMATH_CALUDE_steve_socks_l223_22314


namespace NUMINAMATH_CALUDE_hash_of_hash_of_hash_4_l223_22389

def hash (N : ℝ) : ℝ := 0.5 * N^2 + 1

theorem hash_of_hash_of_hash_4 : hash (hash (hash 4)) = 862.125 := by
  sorry

end NUMINAMATH_CALUDE_hash_of_hash_of_hash_4_l223_22389


namespace NUMINAMATH_CALUDE_milk_mixture_price_l223_22380

/-- Calculate the selling price of a milk-water mixture per litre -/
theorem milk_mixture_price (pure_milk_cost : ℝ) (pure_milk_volume : ℝ) (water_volume : ℝ) :
  pure_milk_cost = 3.60 →
  pure_milk_volume = 25 →
  water_volume = 5 →
  (pure_milk_cost * pure_milk_volume) / (pure_milk_volume + water_volume) = 3 := by
sorry


end NUMINAMATH_CALUDE_milk_mixture_price_l223_22380


namespace NUMINAMATH_CALUDE_choose_five_three_l223_22390

theorem choose_five_three (n : ℕ) (k : ℕ) : n = 5 ∧ k = 3 → Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_choose_five_three_l223_22390


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l223_22365

theorem circle_area_with_diameter_10 :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l223_22365


namespace NUMINAMATH_CALUDE_closest_perfect_square_to_1042_l223_22326

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

theorem closest_perfect_square_to_1042 :
  ∃ (n : ℤ), is_perfect_square n ∧
    ∀ (m : ℤ), is_perfect_square m → |n - 1042| ≤ |m - 1042| ∧
    n = 1024 :=
by sorry

end NUMINAMATH_CALUDE_closest_perfect_square_to_1042_l223_22326


namespace NUMINAMATH_CALUDE_optimal_solution_l223_22358

/-- Represents a vehicle type with its capacity, quantity, and fuel efficiency. -/
structure VehicleType where
  capacity : Nat
  quantity : Nat
  fuelEfficiency : Nat

/-- Represents the problem setup for the field trip. -/
structure FieldTripProblem where
  cars : VehicleType
  minivans : VehicleType
  buses : VehicleType
  totalPeople : Nat
  tripDistance : Nat

/-- Represents a solution to the field trip problem. -/
structure FieldTripSolution where
  numCars : Nat
  numMinivans : Nat
  numBuses : Nat

def fuelUsage (problem : FieldTripProblem) (solution : FieldTripSolution) : Rat :=
  (problem.tripDistance * solution.numCars / problem.cars.fuelEfficiency : Rat) +
  (problem.tripDistance * solution.numMinivans / problem.minivans.fuelEfficiency : Rat) +
  (problem.tripDistance * solution.numBuses / problem.buses.fuelEfficiency : Rat)

def totalCapacity (problem : FieldTripProblem) (solution : FieldTripSolution) : Nat :=
  solution.numCars * problem.cars.capacity +
  solution.numMinivans * problem.minivans.capacity +
  solution.numBuses * problem.buses.capacity

def isValidSolution (problem : FieldTripProblem) (solution : FieldTripSolution) : Prop :=
  solution.numCars ≤ problem.cars.quantity ∧
  solution.numMinivans ≤ problem.minivans.quantity ∧
  solution.numBuses ≤ problem.buses.quantity ∧
  totalCapacity problem solution ≥ problem.totalPeople

theorem optimal_solution (problem : FieldTripProblem) (solution : FieldTripSolution) :
  problem.cars = { capacity := 4, quantity := 3, fuelEfficiency := 30 } ∧
  problem.minivans = { capacity := 6, quantity := 2, fuelEfficiency := 20 } ∧
  problem.buses = { capacity := 20, quantity := 1, fuelEfficiency := 10 } ∧
  problem.totalPeople = 33 ∧
  problem.tripDistance = 50 ∧
  solution = { numCars := 1, numMinivans := 1, numBuses := 1 } ∧
  isValidSolution problem solution →
  ∀ (altSolution : FieldTripSolution),
    isValidSolution problem altSolution →
    fuelUsage problem solution ≤ fuelUsage problem altSolution :=
by sorry

end NUMINAMATH_CALUDE_optimal_solution_l223_22358


namespace NUMINAMATH_CALUDE_induction_sum_terms_l223_22373

theorem induction_sum_terms (k : ℕ) (h : k > 1) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k := by
  sorry

end NUMINAMATH_CALUDE_induction_sum_terms_l223_22373


namespace NUMINAMATH_CALUDE_system_solution_approximation_l223_22362

/-- The system of equations has a unique solution close to (0.4571, 0.1048) -/
theorem system_solution_approximation : ∃! (x y : ℝ), 
  (4 * x - 6 * y = -2) ∧ 
  (5 * x + 3 * y = 2.6) ∧ 
  (abs (x - 0.4571) < 0.0001) ∧ 
  (abs (y - 0.1048) < 0.0001) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_approximation_l223_22362


namespace NUMINAMATH_CALUDE_school_survey_is_stratified_sampling_l223_22340

/-- Represents a sampling method -/
inductive SamplingMethod
  | DrawingLots
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a population divided into groups -/
structure Population where
  totalSize : ℕ
  groups : List (ℕ × ℕ)  -- (group size, sample size) pairs

/-- Checks if a sampling method is stratified -/
def isStratifiedSampling (pop : Population) (method : SamplingMethod) : Prop :=
  method = SamplingMethod.Stratified ∧
  pop.groups.length ≥ 2 ∧
  (∀ (g₁ g₂ : ℕ × ℕ), g₁ ∈ pop.groups → g₂ ∈ pop.groups →
    (g₁.1 : ℚ) / (g₂.1 : ℚ) = (g₁.2 : ℚ) / (g₂.2 : ℚ))

/-- The main theorem to prove -/
theorem school_survey_is_stratified_sampling
  (totalStudents : ℕ)
  (maleStudents femaleStudents : ℕ)
  (maleSample femaleSample : ℕ)
  (h_total : totalStudents = maleStudents + femaleStudents)
  (h_male_ratio : (maleStudents : ℚ) / (totalStudents : ℚ) = 2 / 5)
  (h_female_ratio : (femaleStudents : ℚ) / (totalStudents : ℚ) = 3 / 5)
  (h_sample_ratio : (maleSample : ℚ) / (femaleSample : ℚ) = 2 / 3)
  : isStratifiedSampling
      { totalSize := totalStudents,
        groups := [(maleStudents, maleSample), (femaleStudents, femaleSample)] }
      SamplingMethod.Stratified :=
by sorry

end NUMINAMATH_CALUDE_school_survey_is_stratified_sampling_l223_22340


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l223_22330

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_sequence_product (a b c : ℝ) :
  is_geometric_sequence (-1) a b c (-2) →
  a * b * c = -2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l223_22330


namespace NUMINAMATH_CALUDE_sixth_quiz_score_l223_22315

def quiz_scores : List ℕ := [86, 91, 88, 84, 97]
def desired_average : ℕ := 95
def num_quizzes : ℕ := 6

theorem sixth_quiz_score :
  ∃ (score : ℕ),
    (quiz_scores.sum + score) / num_quizzes = desired_average ∧
    score = num_quizzes * desired_average - quiz_scores.sum :=
by sorry

end NUMINAMATH_CALUDE_sixth_quiz_score_l223_22315


namespace NUMINAMATH_CALUDE_single_weighing_correctness_check_l223_22354

/-- Represents a weight with its mass and marking -/
structure Weight where
  mass : ℝ
  marking : ℝ

/-- Represents the position of a weight on the scale -/
structure Position where
  weight : Weight
  distance : ℝ

/-- Calculates the moment of a weight at a given position -/
def moment (p : Position) : ℝ := p.weight.mass * p.distance

/-- Theorem: It's always possible to check if all markings are correct in a single weighing -/
theorem single_weighing_correctness_check 
  (weights : Finset Weight) 
  (hweights : weights.Nonempty) 
  (hmasses : ∀ w ∈ weights, ∃ w' ∈ weights, w.mass = w'.marking) 
  (hmarkings : ∀ w ∈ weights, ∃ w' ∈ weights, w.marking = w'.mass) :
  ∃ (left right : Finset Position),
    (∀ p ∈ left, p.weight ∈ weights) ∧
    (∀ p ∈ right, p.weight ∈ weights) ∧
    (left.sum moment = right.sum moment ↔ 
      ∀ w ∈ weights, w.mass = w.marking) :=
sorry

end NUMINAMATH_CALUDE_single_weighing_correctness_check_l223_22354


namespace NUMINAMATH_CALUDE_squares_in_35x2_grid_l223_22359

/-- The number of squares in a rectangular grid --/
def count_squares (length width : ℕ) : ℕ :=
  -- Count 1x1 squares
  length * width +
  -- Count 2x2 squares
  (length - 1) * (width - 1)

/-- Theorem: The number of squares in a 35x2 grid is 104 --/
theorem squares_in_35x2_grid :
  count_squares 35 2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_35x2_grid_l223_22359


namespace NUMINAMATH_CALUDE_division_problem_l223_22371

theorem division_problem : (150 : ℚ) / ((6 : ℚ) / 3) = 75 := by sorry

end NUMINAMATH_CALUDE_division_problem_l223_22371


namespace NUMINAMATH_CALUDE_water_truck_capacity_l223_22320

/-- The maximum capacity of the water truck in tons -/
def truck_capacity : ℝ := 12

/-- The amount of water (in tons) injected by pipe A when used with pipe C -/
def water_A_with_C : ℝ := 4

/-- The amount of water (in tons) injected by pipe B when used with pipe C -/
def water_B_with_C : ℝ := 6

/-- The ratio of pipe B's injection rate to pipe A's injection rate -/
def rate_ratio_B_to_A : ℝ := 2

theorem water_truck_capacity :
  truck_capacity = water_A_with_C * rate_ratio_B_to_A ∧
  truck_capacity = water_B_with_C + water_A_with_C :=
by sorry

end NUMINAMATH_CALUDE_water_truck_capacity_l223_22320


namespace NUMINAMATH_CALUDE_exam_scores_theorem_l223_22396

/-- A type representing a student's scores in three tasks -/
structure StudentScores :=
  (task1 : Nat)
  (task2 : Nat)
  (task3 : Nat)

/-- A predicate that checks if all scores are between 0 and 7 -/
def validScores (s : StudentScores) : Prop :=
  0 ≤ s.task1 ∧ s.task1 ≤ 7 ∧
  0 ≤ s.task2 ∧ s.task2 ≤ 7 ∧
  0 ≤ s.task3 ∧ s.task3 ≤ 7

/-- A predicate that checks if one student's scores are greater than or equal to another's -/
def scoresGreaterOrEqual (s1 s2 : StudentScores) : Prop :=
  s1.task1 ≥ s2.task1 ∧ s1.task2 ≥ s2.task2 ∧ s1.task3 ≥ s2.task3

/-- The main theorem to be proved -/
theorem exam_scores_theorem (students : Finset StudentScores) 
    (h : students.card = 49)
    (h_valid : ∀ s ∈ students, validScores s) :
  ∃ s1 s2 : StudentScores, s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧ scoresGreaterOrEqual s1 s2 :=
sorry

end NUMINAMATH_CALUDE_exam_scores_theorem_l223_22396


namespace NUMINAMATH_CALUDE_movie_watching_time_l223_22322

/-- The duration of Bret's train ride to Boston -/
def total_duration : ℕ := 9

/-- The time Bret spends reading a book -/
def reading_time : ℕ := 2

/-- The time Bret spends eating dinner -/
def eating_time : ℕ := 1

/-- The time Bret has left for a nap -/
def nap_time : ℕ := 3

/-- Theorem stating that the time spent watching movies is 3 hours -/
theorem movie_watching_time :
  total_duration - (reading_time + eating_time + nap_time) = 3 := by
  sorry

end NUMINAMATH_CALUDE_movie_watching_time_l223_22322


namespace NUMINAMATH_CALUDE_certain_to_draw_black_ball_l223_22306

/-- Represents the number of black balls in the bag -/
def black_balls : ℕ := 6

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 3

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := black_balls + white_balls

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 4

/-- Theorem stating that drawing at least one black ball is certain -/
theorem certain_to_draw_black_ball : 
  drawn_balls > white_balls → drawn_balls ≤ total_balls → true := by sorry

end NUMINAMATH_CALUDE_certain_to_draw_black_ball_l223_22306


namespace NUMINAMATH_CALUDE_circle_symmetry_l223_22375

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 5 = 0

-- Define circle D
def circle_D (x y : ℝ) : Prop := (x + 2)^2 + (y - 6)^2 = 1

-- Define circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 1

-- Define symmetry with respect to a line
def symmetric_wrt_line (c₁ c₂ : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (p : ℝ × ℝ), l p.1 p.2 ∧ 
    (c₁.1 + c₂.1) / 2 = p.1 ∧ 
    (c₁.2 + c₂.2) / 2 = p.2 ∧
    (c₂.1 - c₁.1) * (p.2 - c₁.2) = (c₂.2 - c₁.2) * (p.1 - c₁.1)

theorem circle_symmetry :
  symmetric_wrt_line (-2, 6) (1, 3) line_l →
  (∀ x y : ℝ, circle_D x y ↔ circle_C (x + 3) (y - 3)) :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l223_22375


namespace NUMINAMATH_CALUDE_condition1_arrangements_condition2_arrangements_condition3_arrangements_l223_22344

def num_boys : ℕ := 5
def num_girls : ℕ := 3
def num_subjects : ℕ := 5

def arrangements_condition1 : ℕ := 5520
def arrangements_condition2 : ℕ := 3360
def arrangements_condition3 : ℕ := 360

/-- The number of ways to select representatives under condition 1 -/
theorem condition1_arrangements :
  (Nat.choose num_boys num_boys +
   Nat.choose num_boys (num_boys - 1) * Nat.choose num_girls 1 +
   Nat.choose num_boys (num_boys - 2) * Nat.choose num_girls 2) *
  Nat.factorial num_subjects = arrangements_condition1 := by sorry

/-- The number of ways to select representatives under condition 2 -/
theorem condition2_arrangements :
  Nat.choose (num_boys + num_girls - 1) (num_subjects - 1) *
  (num_subjects - 1) * Nat.factorial (num_subjects - 1) = arrangements_condition2 := by sorry

/-- The number of ways to select representatives under condition 3 -/
theorem condition3_arrangements :
  Nat.choose (num_boys + num_girls - 2) (num_subjects - 2) *
  (num_subjects - 2) * Nat.factorial (num_subjects - 2) = arrangements_condition3 := by sorry

end NUMINAMATH_CALUDE_condition1_arrangements_condition2_arrangements_condition3_arrangements_l223_22344


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l223_22317

/-- Given a line L1 with equation 3x + 6y = 12 and a point P (2, -1),
    prove that the line L2 with equation y = -1/2x is parallel to L1 and passes through P. -/
theorem parallel_line_through_point (x y : ℝ) : 
  (∃ (m b : ℝ), 3*x + 6*y = 12 ↔ y = m*x + b) → -- L1 exists
  (y = -1/2 * x) →                              -- L2 equation
  (∃ (m : ℝ), 3*x + 6*y = 12 ↔ y = m*x + 2) →   -- L1 in slope-intercept form
  (-1 = -1/2 * 2 + 0) →                         -- L2 passes through (2, -1)
  (∃ (k : ℝ), y = -1/2 * x + k ∧ -1 = -1/2 * 2 + k) -- L2 in point-slope form
  :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l223_22317


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_line_equal_intercepts_lines_l223_22387

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 4 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def line3 (x y : ℝ) : Prop := 3 * x + 4 * y - 15 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (3, 2)

-- Define the perpendicular line l1
def l1 (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0

-- Define the lines l2 with equal intercepts
def l2_1 (x y : ℝ) : Prop := 2 * x - 3 * y = 0
def l2_2 (x y : ℝ) : Prop := x + y - 5 = 0

theorem intersection_and_perpendicular_line :
  (∀ x y, line1 x y ∧ line2 x y → (x, y) = P) ∧
  (∀ x y, l1 x y → (4 : ℝ) * 3 + 3 * 4 = 0) ∧
  l1 P.1 P.2 :=
sorry

theorem equal_intercepts_lines :
  (∀ x y, line1 x y ∧ line2 x y → (x, y) = P) ∧
  (l2_1 P.1 P.2 ∨ l2_2 P.1 P.2) ∧
  (∃ a ≠ 0, ∀ x y, l2_1 x y → x / a + y / a = 1) ∧
  (∃ a ≠ 0, ∀ x y, l2_2 x y → x / a + y / a = 1) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_line_equal_intercepts_lines_l223_22387


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_l223_22363

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of a point on the ellipse -/
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  ellipse_C P.1 P.2

/-- Definition of the right focus -/
def right_focus : ℝ × ℝ := (1, 0)

/-- Definition of a line passing through the right focus -/
def line_through_focus (A B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), A = right_focus + t • (B - right_focus) ∨
             B = right_focus + t • (A - right_focus)

/-- Definition of the dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The main theorem to be proved -/
theorem ellipse_fixed_point :
  ∃ (M : ℝ × ℝ), M.1 = 5/4 ∧ M.2 = 0 ∧
  ∀ (A B : ℝ × ℝ), point_on_ellipse A → point_on_ellipse B →
  line_through_focus A B →
  dot_product (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2) = -7/16 :=
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_l223_22363


namespace NUMINAMATH_CALUDE_range_of_m_when_p_true_range_of_m_when_p_or_q_true_and_p_and_q_false_l223_22367

-- Define proposition p
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 ≥ m

-- Define proposition q
def q (m : ℝ) : Prop := ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (m - 2) * (m + 2) < 0 ∧
  ∀ x y : ℝ, x^2 / (m - 2) + y^2 / (m + 2) = 1 ↔ (x / a)^2 - (y / b)^2 = 1

-- Theorem 1
theorem range_of_m_when_p_true (m : ℝ) : p m → m ≤ 1 := by sorry

-- Theorem 2
theorem range_of_m_when_p_or_q_true_and_p_and_q_false (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ≤ -2 ∨ (1 < m ∧ m < 2) := by sorry

end NUMINAMATH_CALUDE_range_of_m_when_p_true_range_of_m_when_p_or_q_true_and_p_and_q_false_l223_22367


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_third_l223_22346

theorem opposite_of_negative_one_third :
  -(-(1/3 : ℚ)) = 1/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_third_l223_22346


namespace NUMINAMATH_CALUDE_office_employees_l223_22361

/-- Proves that the total number of employees in an office is 2200 given specific conditions -/
theorem office_employees (total : ℕ) (male_ratio : ℚ) (old_male_ratio : ℚ) (young_males : ℕ) 
  (h1 : male_ratio = 2/5)
  (h2 : old_male_ratio = 3/10)
  (h3 : young_males = 616)
  (h4 : ↑young_males = (1 - old_male_ratio) * (male_ratio * ↑total)) : 
  total = 2200 := by
sorry

end NUMINAMATH_CALUDE_office_employees_l223_22361


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l223_22391

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 5) ↔ (x ∈ Set.Icc (-2) 1 ∪ Set.Icc 5 8) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l223_22391


namespace NUMINAMATH_CALUDE_jellybean_count_l223_22379

/-- The number of jellybeans needed to fill a large drinking glass -/
def large_glass : ℕ := sorry

/-- The number of jellybeans needed to fill a small drinking glass -/
def small_glass : ℕ := sorry

/-- The total number of large glasses -/
def num_large_glasses : ℕ := 5

/-- The total number of small glasses -/
def num_small_glasses : ℕ := 3

/-- The total number of jellybeans needed to fill all glasses -/
def total_jellybeans : ℕ := 325

theorem jellybean_count :
  (small_glass = large_glass / 2) →
  (num_large_glasses * large_glass + num_small_glasses * small_glass = total_jellybeans) →
  large_glass = 50 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l223_22379


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l223_22305

/-- Calculate the cost of plastering a tank's walls and bottom -/
theorem tank_plastering_cost 
  (length : ℝ) 
  (width : ℝ) 
  (depth : ℝ) 
  (cost_per_sq_m : ℝ) : 
  length = 25 → 
  width = 12 → 
  depth = 6 → 
  cost_per_sq_m = 0.75 → 
  2 * (length * depth + width * depth) + length * width = 744 ∧ 
  (2 * (length * depth + width * depth) + length * width) * cost_per_sq_m = 558 := by
  sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l223_22305


namespace NUMINAMATH_CALUDE_kelly_wendy_ratio_l223_22342

def scholarship_problem (kelly wendy nina : ℕ) : Prop :=
  let total := 92000
  wendy = 20000 ∧
  ∃ n : ℕ, kelly = n * wendy ∧
  nina = kelly - 8000 ∧
  kelly + nina + wendy = total

theorem kelly_wendy_ratio :
  ∀ kelly wendy nina : ℕ,
  scholarship_problem kelly wendy nina →
  kelly / wendy = 2 :=
sorry

end NUMINAMATH_CALUDE_kelly_wendy_ratio_l223_22342


namespace NUMINAMATH_CALUDE_two_numbers_problem_l223_22313

theorem two_numbers_problem (x y z : ℝ) 
  (h1 : x > y) 
  (h2 : x + y = 90) 
  (h3 : x - y = 15) 
  (h4 : z = x^2 - y^2) : 
  z = 1350 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l223_22313


namespace NUMINAMATH_CALUDE_total_age_is_42_l223_22316

/-- Given three people a, b, and c, where a is two years older than b, 
    b is twice as old as c, and b is 16 years old, 
    prove that the total of their ages is 42 years. -/
theorem total_age_is_42 (a b c : ℕ) : 
  a = b + 2 → b = 2 * c → b = 16 → a + b + c = 42 :=
by sorry

end NUMINAMATH_CALUDE_total_age_is_42_l223_22316


namespace NUMINAMATH_CALUDE_charley_beads_problem_l223_22372

theorem charley_beads_problem (white_beads black_beads : ℕ) 
  (black_fraction : ℚ) (total_pulled : ℕ) :
  white_beads = 51 →
  black_beads = 90 →
  black_fraction = 1 / 6 →
  total_pulled = 32 →
  ∃ (white_fraction : ℚ),
    white_fraction * white_beads + black_fraction * black_beads = total_pulled ∧
    white_fraction = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_charley_beads_problem_l223_22372


namespace NUMINAMATH_CALUDE_intersection_M_N_l223_22310

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l223_22310


namespace NUMINAMATH_CALUDE_rationalize_denominator_l223_22394

theorem rationalize_denominator :
  (3 : ℝ) / (Real.sqrt 50 + 2) = (15 * Real.sqrt 2 - 6) / 46 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l223_22394


namespace NUMINAMATH_CALUDE_number_value_relationship_l223_22308

theorem number_value_relationship (n v : ℝ) : 
  n > 0 → n = 7 → n - 4 = 21 * v → v = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_number_value_relationship_l223_22308


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l223_22341

theorem quadratic_no_real_roots (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - k ≠ 0) → k < -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l223_22341


namespace NUMINAMATH_CALUDE_max_dip_amount_l223_22366

/-- Given the following conditions:
  * total_money: The total amount of money available to spend on artichokes
  * cost_per_artichoke: The cost of each artichoke
  * artichokes_per_batch: The number of artichokes needed to make one batch of dip
  * ounces_per_batch: The number of ounces of dip produced from one batch

  Prove that the maximum amount of dip that can be made is 20 ounces.
-/
theorem max_dip_amount (total_money : ℚ) (cost_per_artichoke : ℚ) 
  (artichokes_per_batch : ℕ) (ounces_per_batch : ℚ) 
  (h1 : total_money = 15)
  (h2 : cost_per_artichoke = 5/4)
  (h3 : artichokes_per_batch = 3)
  (h4 : ounces_per_batch = 5) :
  (total_money / cost_per_artichoke) * (ounces_per_batch / artichokes_per_batch) = 20 :=
by sorry

end NUMINAMATH_CALUDE_max_dip_amount_l223_22366


namespace NUMINAMATH_CALUDE_product_cost_l223_22364

/-- The cost of a product given its selling price and profit margin -/
theorem product_cost (x a : ℝ) (h : a > 0) :
  let selling_price := x
  let profit_margin := a / 100
  selling_price = (1 + profit_margin) * (selling_price / (1 + profit_margin)) :=
by sorry

end NUMINAMATH_CALUDE_product_cost_l223_22364


namespace NUMINAMATH_CALUDE_total_worksheets_l223_22312

/-- Given a teacher grading worksheets, this theorem proves the total number of worksheets. -/
theorem total_worksheets (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ) : 
  problems_per_worksheet = 4 →
  graded_worksheets = 5 →
  remaining_problems = 16 →
  graded_worksheets + (remaining_problems / problems_per_worksheet) = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_worksheets_l223_22312


namespace NUMINAMATH_CALUDE_slope_intercept_form_parallel_lines_a_value_l223_22343

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The slope-intercept form of a line ax + by + c = 0 is y = (-a/b)x - (c/b) when b ≠ 0 -/
theorem slope_intercept_form {a b c : ℝ} (hb : b ≠ 0) :
  (∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = (-a/b) * x - (c/b)) :=
sorry

theorem parallel_lines_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, a * x - y + a = 0 ↔ (2*a-3) * x + a * y - a = 0) → a = -3 :=
sorry

end NUMINAMATH_CALUDE_slope_intercept_form_parallel_lines_a_value_l223_22343


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l223_22353

open Real

/-- The function f(x) = √3 sin x - cos x is strictly increasing in the intervals [-π/3 + 2kπ, 2π/3 + 2kπ], where k ∈ ℤ -/
theorem f_strictly_increasing (x : ℝ) :
  ∃ (k : ℤ), x ∈ Set.Icc (-π/3 + 2*π*k) (2*π/3 + 2*π*k) →
  StrictMonoOn (λ x => Real.sqrt 3 * sin x - cos x) (Set.Icc (-π/3 + 2*π*k) (2*π/3 + 2*π*k)) :=
by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l223_22353


namespace NUMINAMATH_CALUDE_cone_base_radius_l223_22345

/-- Represents a cone with given properties -/
structure Cone where
  surface_area : ℝ
  lateral_unfolds_semicircle : Prop

/-- Theorem: For a cone with surface area 12π and lateral surface that unfolds into a semicircle, 
    the radius of the base is 2 -/
theorem cone_base_radius 
  (cone : Cone) 
  (h1 : cone.surface_area = 12 * Real.pi) 
  (h2 : cone.lateral_unfolds_semicircle) : 
  ∃ (r : ℝ), r = 2 ∧ r > 0 ∧ 
  cone.surface_area = Real.pi * r^2 + Real.pi * r * (2 * r) := by
  sorry


end NUMINAMATH_CALUDE_cone_base_radius_l223_22345


namespace NUMINAMATH_CALUDE_second_player_eats_53_seeds_l223_22350

/-- The number of seeds eaten by the first player -/
def first_player_seeds : ℕ := 78

/-- The number of seeds eaten by the second player -/
def second_player_seeds : ℕ := 53

/-- The number of seeds eaten by the third player -/
def third_player_seeds : ℕ := second_player_seeds + 30

/-- The total number of seeds eaten by all players -/
def total_seeds : ℕ := 214

/-- Theorem stating that the given conditions result in the second player eating 53 seeds -/
theorem second_player_eats_53_seeds :
  first_player_seeds + second_player_seeds + third_player_seeds = total_seeds :=
by sorry

end NUMINAMATH_CALUDE_second_player_eats_53_seeds_l223_22350


namespace NUMINAMATH_CALUDE_teacher_age_l223_22386

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 50 →
  student_avg_age = 14 →
  new_avg_age = 15 →
  (num_students * student_avg_age + (65 : ℝ)) / (num_students + 1) = new_avg_age :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l223_22386


namespace NUMINAMATH_CALUDE_existence_of_four_integers_l223_22324

theorem existence_of_four_integers : ∃ (a b c d : ℤ),
  (abs a > 1000000) ∧
  (abs b > 1000000) ∧
  (abs c > 1000000) ∧
  (abs d > 1000000) ∧
  (1 / a + 1 / b + 1 / c + 1 / d : ℚ) = 1 / (a * b * c * d) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_four_integers_l223_22324


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l223_22383

theorem triangle_angle_problem (A B C : ℝ) (a b c : ℝ) : 
  A + B + C = π → 
  C = π / 5 → 
  a * Real.cos B - b * Real.cos A = c → 
  B = 3 * π / 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l223_22383


namespace NUMINAMATH_CALUDE_modulus_of_two_over_one_plus_i_l223_22392

open Complex

theorem modulus_of_two_over_one_plus_i :
  let z : ℂ := 2 / (1 + I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_two_over_one_plus_i_l223_22392


namespace NUMINAMATH_CALUDE_product_of_fractions_is_zero_l223_22300

def fraction (n : ℕ) : ℚ := (n^3 - 1) / (n^3 + 1)

theorem product_of_fractions_is_zero :
  (fraction 1) * (fraction 2) * (fraction 3) * (fraction 4) = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_fractions_is_zero_l223_22300


namespace NUMINAMATH_CALUDE_set_forms_triangle_l223_22388

/-- Triangle Inequality Theorem: A set of three positive real numbers a, b, c can form a triangle
    if and only if the sum of any two is greater than the third. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The set (7, 15, 10) can form a triangle. -/
theorem set_forms_triangle : can_form_triangle 7 15 10 := by
  sorry


end NUMINAMATH_CALUDE_set_forms_triangle_l223_22388


namespace NUMINAMATH_CALUDE_circumcenter_on_side_implies_right_angled_l223_22339

/-- A triangle is represented by its three vertices in a 2D plane. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumcenter of a triangle is the point where the perpendicular bisectors of the sides intersect. -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- A predicate to check if a point lies on a side of a triangle. -/
def point_on_side (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- A predicate to check if a triangle is right-angled. -/
def is_right_angled (t : Triangle) : Prop := sorry

/-- Theorem: If the circumcenter of a triangle lies on one of its sides, then the triangle is right-angled. -/
theorem circumcenter_on_side_implies_right_angled (t : Triangle) :
  point_on_side (circumcenter t) t → is_right_angled t := by
  sorry

end NUMINAMATH_CALUDE_circumcenter_on_side_implies_right_angled_l223_22339


namespace NUMINAMATH_CALUDE_max_common_tangents_shared_focus_l223_22309

/-- Represents an ellipse in 2D space -/
structure Ellipse where
  foci : Fin 2 → ℝ × ℝ
  majorAxis : ℝ

/-- Represents a tangent line to an ellipse -/
structure Tangent where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Returns the number of common tangents between two ellipses -/
def commonTangents (e1 e2 : Ellipse) : ℕ := sorry

/-- Theorem: The maximum number of common tangents for two ellipses sharing one focus is 2 -/
theorem max_common_tangents_shared_focus (e1 e2 : Ellipse) 
  (h : e1.foci 1 = e2.foci 1) : 
  commonTangents e1 e2 ≤ 2 := by sorry

end NUMINAMATH_CALUDE_max_common_tangents_shared_focus_l223_22309


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l223_22393

theorem sum_of_roots_quadratic (x : ℝ) : (x + 3) * (x - 5) = 20 → ∃ y z : ℝ, x^2 - 2*x - 35 = 0 ∧ y + z = 2 ∧ (x = y ∨ x = z) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l223_22393


namespace NUMINAMATH_CALUDE_marble_bag_problem_l223_22335

theorem marble_bag_problem (T : ℕ) (h1 : T > 12) : 
  (((T - 12 : ℚ) / T) * ((T - 12 : ℚ) / T) = 36 / 49) → T = 84 := by
  sorry

end NUMINAMATH_CALUDE_marble_bag_problem_l223_22335


namespace NUMINAMATH_CALUDE_parentheses_placement_l223_22321

theorem parentheses_placement :
  (7 * (9 + 12 / 3) = 91) ∧
  ((7 * 9 + 12) / 3 = 25) ∧
  (7 * (9 + 12) / 3 = 49) ∧
  ((48 * 6) / (48 * 6) = 1) := by
  sorry

end NUMINAMATH_CALUDE_parentheses_placement_l223_22321


namespace NUMINAMATH_CALUDE_pet_store_theorem_l223_22333

/-- The number of ways to choose and assign different pets to four people -/
def pet_store_combinations : ℕ :=
  let puppies : ℕ := 12
  let kittens : ℕ := 10
  let hamsters : ℕ := 9
  let parrots : ℕ := 7
  let people : ℕ := 4
  puppies * kittens * hamsters * parrots * Nat.factorial people

/-- Theorem stating the number of combinations for the pet store problem -/
theorem pet_store_theorem : pet_store_combinations = 181440 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_theorem_l223_22333


namespace NUMINAMATH_CALUDE_chef_manager_wage_difference_l223_22337

/-- Represents the hourly wages at Joe's Steakhouse -/
structure SteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ

/-- The conditions for wages at Joe's Steakhouse -/
def wage_conditions (w : SteakhouseWages) : Prop :=
  w.manager = 8.50 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.dishwasher * 1.22

theorem chef_manager_wage_difference (w : SteakhouseWages) 
  (h : wage_conditions w) : w.manager - w.chef = 3.315 := by
  sorry

#check chef_manager_wage_difference

end NUMINAMATH_CALUDE_chef_manager_wage_difference_l223_22337


namespace NUMINAMATH_CALUDE_smallest_marble_count_l223_22303

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles in the urn -/
def totalMarbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green + mc.yellow

/-- Checks if the probabilities of the five specified events are equal -/
def equalProbabilities (mc : MarbleCount) : Prop :=
  let r := mc.red
  let w := mc.white
  let b := mc.blue
  let g := mc.green
  let y := mc.yellow
  Nat.choose r 5 = w * Nat.choose r 4 ∧
  Nat.choose r 5 = w * b * Nat.choose r 3 ∧
  Nat.choose r 5 = w * b * g * Nat.choose r 2 ∧
  Nat.choose r 5 = w * b * g * y * r

/-- Theorem stating that the smallest number of marbles satisfying the conditions is 13 -/
theorem smallest_marble_count :
  ∃ (mc : MarbleCount), totalMarbles mc = 13 ∧ equalProbabilities mc ∧
  (∀ (mc' : MarbleCount), equalProbabilities mc' → totalMarbles mc' ≥ 13) := by
  sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l223_22303


namespace NUMINAMATH_CALUDE_ricks_ironing_total_l223_22368

/-- Rick's ironing problem -/
theorem ricks_ironing_total (shirts_per_hour pants_per_hour shirt_hours pant_hours : ℕ) 
  (h1 : shirts_per_hour = 4)
  (h2 : pants_per_hour = 3)
  (h3 : shirt_hours = 3)
  (h4 : pant_hours = 5) :
  shirts_per_hour * shirt_hours + pants_per_hour * pant_hours = 27 := by
  sorry

#check ricks_ironing_total

end NUMINAMATH_CALUDE_ricks_ironing_total_l223_22368


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l223_22397

theorem arctan_equation_solution :
  ∀ y : ℝ, 2 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4 → y = 1210 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l223_22397


namespace NUMINAMATH_CALUDE_jackson_money_l223_22374

/-- The amount of money each person has -/
structure Money where
  williams : ℝ
  jackson : ℝ
  lucy : ℝ
  ethan : ℝ

/-- The conditions of the problem -/
def problem_conditions (m : Money) : Prop :=
  m.jackson = 7 * m.williams ∧
  m.lucy = 3 * m.williams ∧
  m.ethan = m.lucy + 20 ∧
  m.williams + m.jackson + m.lucy + m.ethan = 600

/-- The theorem stating Jackson's money amount -/
theorem jackson_money (m : Money) (h : problem_conditions m) : 
  m.jackson = 7 * (600 - 20) / 14 := by
  sorry

end NUMINAMATH_CALUDE_jackson_money_l223_22374


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_range_l223_22398

theorem rectangular_solid_volume_range (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  2 * (a * b + b * c + a * c) = 48 →
  4 * (a + b + c) = 36 →
  16 ≤ a * b * c ∧ a * b * c ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_range_l223_22398


namespace NUMINAMATH_CALUDE_cubic_root_cube_relation_l223_22355

/-- Given a cubic polynomial f(x) = x^3 - 2x^2 + 5x - 3 with three distinct roots,
    and another cubic polynomial g(x) = x^3 + bx^2 + cx + d whose roots are
    the cubes of the roots of f(x), prove that b = -2, c = -5, and d = 3. -/
theorem cubic_root_cube_relation :
  let f (x : ℝ) := x^3 - 2*x^2 + 5*x - 3
  let g (x : ℝ) := x^3 + b*x^2 + c*x + d
  ∀ (b c d : ℝ),
  (∀ r : ℝ, f r = 0 → g (r^3) = 0) →
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  b = -2 ∧ c = -5 ∧ d = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_cube_relation_l223_22355


namespace NUMINAMATH_CALUDE_total_cantaloupes_l223_22338

def fred_cantaloupes : ℕ := 38
def tim_cantaloupes : ℕ := 44

theorem total_cantaloupes : fred_cantaloupes + tim_cantaloupes = 82 := by
  sorry

end NUMINAMATH_CALUDE_total_cantaloupes_l223_22338


namespace NUMINAMATH_CALUDE_subsets_containing_five_and_six_l223_22334

def S : Finset Nat := {1, 2, 3, 4, 5, 6}

theorem subsets_containing_five_and_six :
  (Finset.filter (λ s : Finset Nat => 5 ∈ s ∧ 6 ∈ s) (Finset.powerset S)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_subsets_containing_five_and_six_l223_22334


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l223_22378

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 1 where f(2012) = 3,
    prove that f(-2012) = -1 -/
theorem polynomial_symmetry (a b c : ℝ) :
  let f := fun x => a * x^5 + b * x^3 + c * x + 1
  (f 2012 = 3) → (f (-2012) = -1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l223_22378


namespace NUMINAMATH_CALUDE_range_of_f_l223_22399

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Define the range
def range : Set ℝ := { y | ∃ x ∈ domain, f x = y }

-- Theorem statement
theorem range_of_f : range = { y | -3 ≤ y ∧ y ≤ 5 } := by sorry

end NUMINAMATH_CALUDE_range_of_f_l223_22399


namespace NUMINAMATH_CALUDE_stratified_sample_size_l223_22349

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  stratum_size : ℕ
  stratum_sample : ℕ
  total_sample : ℕ

/-- Checks if the sampling is proportionally correct -/
def is_proportional_sampling (s : StratifiedSample) : Prop :=
  s.stratum_sample * s.total_population = s.total_sample * s.stratum_size

theorem stratified_sample_size 
  (s : StratifiedSample) 
  (h1 : s.total_population = 4320)
  (h2 : s.stratum_size = 1800)
  (h3 : s.stratum_sample = 45)
  (h4 : is_proportional_sampling s) :
  s.total_sample = 108 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l223_22349


namespace NUMINAMATH_CALUDE_infinite_sum_evaluation_l223_22369

theorem infinite_sum_evaluation : 
  (∑' n : ℕ, (n : ℝ) / ((n : ℝ)^4 + 4)) = 3/8 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_evaluation_l223_22369


namespace NUMINAMATH_CALUDE_total_leaves_on_farm_l223_22325

/-- Calculate the total number of leaves on all trees on a farm --/
theorem total_leaves_on_farm (
  num_trees : ℕ)
  (branches_per_tree : ℕ)
  (sub_branches_per_branch : ℕ)
  (leaves_per_sub_branch : ℕ)
  (h1 : num_trees = 4)
  (h2 : branches_per_tree = 10)
  (h3 : sub_branches_per_branch = 40)
  (h4 : leaves_per_sub_branch = 60)
  : num_trees * branches_per_tree * sub_branches_per_branch * leaves_per_sub_branch = 96000 := by
  sorry

#check total_leaves_on_farm

end NUMINAMATH_CALUDE_total_leaves_on_farm_l223_22325


namespace NUMINAMATH_CALUDE_finite_decimal_consecutive_denominators_l223_22319

def is_finite_decimal (q : ℚ) : Prop :=
  ∃ (a b : ℤ) (k : ℕ), q = a / (b * 10^k) ∧ b ≠ 0

theorem finite_decimal_consecutive_denominators :
  ∀ n : ℕ, (is_finite_decimal (1 / n) ∧ is_finite_decimal (1 / (n + 1))) ↔ (n = 1 ∨ n = 4) :=
sorry

end NUMINAMATH_CALUDE_finite_decimal_consecutive_denominators_l223_22319


namespace NUMINAMATH_CALUDE_circle_equation_l223_22302

/-- Given a circle C with radius 3 and center symmetric to (1,0) about y=x,
    prove that its standard equation is x^2 + (y-1)^2 = 9 -/
theorem circle_equation (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (x - center.1)^2 + (y - center.2)^2 = 3^2) →
  (center.1, center.2) = (0, 1) →
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + (y - 1)^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l223_22302


namespace NUMINAMATH_CALUDE_min_sum_unpainted_cells_l223_22352

/-- Represents a 10x10 table of natural numbers -/
def Table := Fin 10 → Fin 10 → ℕ

/-- Checks if a cell is a corner cell -/
def is_corner (i j : Fin 10) : Prop :=
  (i = 0 ∨ i = 9) ∧ (j = 0 ∨ j = 9)

/-- Checks if two cells are neighbors -/
def are_neighbors (i1 j1 i2 j2 : Fin 10) : Prop :=
  (i1 = i2 ∧ (j1 = j2 + 1 ∨ j1 + 1 = j2)) ∨
  (j1 = j2 ∧ (i1 = i2 + 1 ∨ i1 + 1 = i2))

/-- Checks if a cell should be painted based on its neighbors -/
def should_be_painted (t : Table) (i j : Fin 10) : Prop :=
  ∃ (i1 j1 i2 j2 : Fin 10), 
    are_neighbors i j i1 j1 ∧ 
    are_neighbors i j i2 j2 ∧ 
    t i j < t i1 j1 ∧ 
    t i j > t i2 j2

/-- The main theorem -/
theorem min_sum_unpainted_cells (t : Table) :
  (∃! (i1 j1 i2 j2 : Fin 10), 
    ¬is_corner i1 j1 ∧ 
    ¬is_corner i2 j2 ∧ 
    ¬should_be_painted t i1 j1 ∧ 
    ¬should_be_painted t i2 j2 ∧ 
    (∀ (i j : Fin 10), (i ≠ i1 ∨ j ≠ j1) ∧ (i ≠ i2 ∨ j ≠ j2) → should_be_painted t i j)) →
  (∃ (i1 j1 i2 j2 : Fin 10), 
    ¬is_corner i1 j1 ∧ 
    ¬is_corner i2 j2 ∧ 
    ¬should_be_painted t i1 j1 ∧ 
    ¬should_be_painted t i2 j2 ∧ 
    t i1 j1 + t i2 j2 = 3 ∧
    (∀ (k1 l1 k2 l2 : Fin 10), 
      ¬is_corner k1 l1 ∧ 
      ¬is_corner k2 l2 ∧ 
      ¬should_be_painted t k1 l1 ∧ 
      ¬should_be_painted t k2 l2 → 
      t k1 l1 + t k2 l2 ≥ 3)) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_unpainted_cells_l223_22352


namespace NUMINAMATH_CALUDE_palindrome_probability_l223_22311

/-- A function that checks if a number is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- A function that generates all 5-digit palindromes -/
def fiveDigitPalindromes : Finset ℕ := sorry

/-- The total number of 5-digit palindromes -/
def totalPalindromes : ℕ := Finset.card fiveDigitPalindromes

/-- A function that checks if a number is divisible by another number -/
def isDivisibleBy (n m : ℕ) : Prop := n % m = 0

/-- The set of 5-digit palindromes m where m/7 is a palindrome and divisible by 11 -/
def validPalindromes : Finset ℕ := sorry

/-- The number of valid palindromes -/
def validCount : ℕ := Finset.card validPalindromes

theorem palindrome_probability :
  (validCount : ℚ) / totalPalindromes = 1 / 30 := by sorry

end NUMINAMATH_CALUDE_palindrome_probability_l223_22311


namespace NUMINAMATH_CALUDE_total_soda_bottles_l223_22307

/-- The number of regular soda bottles -/
def regular_soda : ℕ := 49

/-- The number of diet soda bottles -/
def diet_soda : ℕ := 40

/-- Theorem: The total number of regular and diet soda bottles is 89 -/
theorem total_soda_bottles : regular_soda + diet_soda = 89 := by
  sorry

end NUMINAMATH_CALUDE_total_soda_bottles_l223_22307


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_solution_is_correct_l223_22357

theorem largest_multiple_of_seven (n : ℤ) : 
  (n % 7 = 0 ∧ -n > -150) → n ≤ 147 :=
by sorry

theorem solution_is_correct : 
  147 % 7 = 0 ∧ -147 > -150 ∧ 
  ∀ m : ℤ, (m % 7 = 0 ∧ -m > -150) → m ≤ 147 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_solution_is_correct_l223_22357


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l223_22385

theorem tripled_base_and_exponent (a : ℝ) (b : ℤ) (x : ℝ) :
  b ≠ 0 →
  (3 * a) ^ (3 * b) = a ^ b * x ^ (3 * b) →
  x = 3 * a ^ (2/3) :=
by sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l223_22385


namespace NUMINAMATH_CALUDE_A_divisible_by_8_l223_22328

def A (n : ℕ) : ℕ := 5^n + 2 * 3^(n-1) + 1

theorem A_divisible_by_8 (n : ℕ) (h : n > 0) : 8 ∣ A n := by
  sorry

end NUMINAMATH_CALUDE_A_divisible_by_8_l223_22328


namespace NUMINAMATH_CALUDE_initial_segment_theorem_l223_22331

theorem initial_segment_theorem (m : ℕ) : ∃ (n k : ℕ), (10^k * m : ℕ) ≤ 2^n ∧ 2^n < 10^k * (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_initial_segment_theorem_l223_22331


namespace NUMINAMATH_CALUDE_two_power_and_factorial_l223_22323

theorem two_power_and_factorial (n : ℕ) :
  (¬ (2^n ∣ n!)) ∧ (∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, 2^(n-1) ∣ n!) := by
  sorry

end NUMINAMATH_CALUDE_two_power_and_factorial_l223_22323


namespace NUMINAMATH_CALUDE_complex_square_root_l223_22347

theorem complex_square_root (z : ℂ) (h : z ^ 2 = 3 + 4 * I) :
  (z.im = 1 ∨ z.im = -1) ∧ Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_square_root_l223_22347


namespace NUMINAMATH_CALUDE_base_seventeen_distinct_digits_l223_22332

/-- The number of three-digit numbers with distinct digits in base b -/
def distinctThreeDigitNumbers (b : ℕ) : ℕ := (b - 1) * (b - 1) * (b - 2)

/-- Theorem stating that there are exactly 256 three-digit numbers with distinct digits in base 17 -/
theorem base_seventeen_distinct_digits : 
  ∃ (b : ℕ), b > 2 ∧ distinctThreeDigitNumbers b = 256 ↔ b = 17 := by sorry

end NUMINAMATH_CALUDE_base_seventeen_distinct_digits_l223_22332


namespace NUMINAMATH_CALUDE_total_age_is_47_l223_22356

/-- Given three people A, B, and C, where A is two years older than B, B is twice as old as C, 
    and B is 18 years old, prove that the total of their ages is 47 years. -/
theorem total_age_is_47 (A B C : ℕ) : 
  B = 18 → A = B + 2 → B = 2 * C → A + B + C = 47 := by sorry

end NUMINAMATH_CALUDE_total_age_is_47_l223_22356


namespace NUMINAMATH_CALUDE_nineteen_eleven_div_eight_l223_22327

theorem nineteen_eleven_div_eight (x : ℕ) : 19^11 / 19^8 = 6859 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_eleven_div_eight_l223_22327
