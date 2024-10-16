import Mathlib

namespace NUMINAMATH_CALUDE_evaluate_expression_l124_12410

theorem evaluate_expression : (16 ^ 24) / (64 ^ 8) = 16 ^ 12 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l124_12410


namespace NUMINAMATH_CALUDE_max_value_rational_function_l124_12415

theorem max_value_rational_function (x : ℝ) :
  x^4 / (x^8 + 2*x^6 + 4*x^4 + 8*x^2 + 16) ≤ 1/20 ∧
  ∃ y : ℝ, y^4 / (y^8 + 2*y^6 + 4*y^4 + 8*y^2 + 16) = 1/20 :=
by sorry

end NUMINAMATH_CALUDE_max_value_rational_function_l124_12415


namespace NUMINAMATH_CALUDE_multiply_by_17_equals_493_l124_12488

theorem multiply_by_17_equals_493 : ∃ x : ℤ, x * 17 = 493 ∧ x = 29 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_17_equals_493_l124_12488


namespace NUMINAMATH_CALUDE_sqrt_18_div_3_minus_sqrt_2_times_sqrt_half_between_0_and_1_l124_12475

theorem sqrt_18_div_3_minus_sqrt_2_times_sqrt_half_between_0_and_1 :
  0 < Real.sqrt 18 / 3 - Real.sqrt 2 * Real.sqrt (1/2) ∧
  Real.sqrt 18 / 3 - Real.sqrt 2 * Real.sqrt (1/2) < 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_div_3_minus_sqrt_2_times_sqrt_half_between_0_and_1_l124_12475


namespace NUMINAMATH_CALUDE_exists_integer_term_l124_12459

def sequence_rule (x : ℚ) : ℚ := x + 1 / (Int.floor x)

def is_valid_sequence (x : ℕ → ℚ) : Prop :=
  x 1 > 1 ∧ ∀ n : ℕ, x (n + 1) = sequence_rule (x n)

theorem exists_integer_term (x : ℕ → ℚ) (h : is_valid_sequence x) :
  ∃ k : ℕ, ∃ m : ℤ, x k = m :=
sorry

end NUMINAMATH_CALUDE_exists_integer_term_l124_12459


namespace NUMINAMATH_CALUDE_school_population_relation_l124_12478

theorem school_population_relation 
  (X : ℝ) -- Total number of students
  (p : ℝ) -- Percentage of boys that 90 students represent
  (h1 : X > 0) -- Assumption that the school has a positive number of students
  (h2 : 0 < p ∧ p < 100) -- Assumption that p is a valid percentage
  : 90 = p / 100 * 0.5 * X := by
  sorry

end NUMINAMATH_CALUDE_school_population_relation_l124_12478


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l124_12442

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular polygon satisfying the given condition has 3 sides -/
theorem regular_polygon_sides : ∃ (n : ℕ), n ≥ 3 ∧ n - num_diagonals n = 3 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l124_12442


namespace NUMINAMATH_CALUDE_distance_to_big_rock_l124_12413

/-- The distance to Big Rock given the rower's speed, river's speed, and round trip time -/
theorem distance_to_big_rock 
  (rower_speed : ℝ) 
  (river_speed : ℝ) 
  (round_trip_time : ℝ) 
  (h1 : rower_speed = 6)
  (h2 : river_speed = 2)
  (h3 : round_trip_time = 1) :
  (rower_speed + river_speed) * (rower_speed - river_speed) * round_trip_time / 
  (rower_speed + river_speed + rower_speed - river_speed) = 8/3 := by
sorry

end NUMINAMATH_CALUDE_distance_to_big_rock_l124_12413


namespace NUMINAMATH_CALUDE_unpainted_area_45_degree_cross_l124_12435

/-- The area of the unpainted region when two boards cross at 45 degrees -/
theorem unpainted_area_45_degree_cross (board_width : ℝ) (cross_angle : ℝ) : 
  board_width = 5 → cross_angle = 45 → 
  (board_width * (board_width * Real.sqrt 2)) = 25 * Real.sqrt 2 := by
  sorry

#check unpainted_area_45_degree_cross

end NUMINAMATH_CALUDE_unpainted_area_45_degree_cross_l124_12435


namespace NUMINAMATH_CALUDE_james_basketball_score_l124_12453

/-- Calculates the total points scored by James in a basketball game --/
def jamesScore (threePointers twoPointers freeThrows missedFreeThrows : ℕ) : ℤ :=
  3 * threePointers + 2 * twoPointers + freeThrows - missedFreeThrows

theorem james_basketball_score :
  jamesScore 13 20 5 2 = 82 := by
  sorry

end NUMINAMATH_CALUDE_james_basketball_score_l124_12453


namespace NUMINAMATH_CALUDE_equal_chords_length_squared_l124_12446

/-- Two circles with radii 10 and 8, centers 15 units apart -/
structure CircleConfiguration where
  center_distance : ℝ
  radius1 : ℝ
  radius2 : ℝ
  center_distance_eq : center_distance = 15
  radius1_eq : radius1 = 10
  radius2_eq : radius2 = 8

/-- Point of intersection of the two circles -/
def IntersectionPoint (config : CircleConfiguration) : Type :=
  { p : ℝ × ℝ // 
    (p.1 - 0)^2 + p.2^2 = config.radius1^2 ∧ 
    (p.1 - config.center_distance)^2 + p.2^2 = config.radius2^2 }

/-- Line through intersection point creating equal chords -/
structure EqualChordsLine (config : CircleConfiguration) where
  p : IntersectionPoint config
  q : ℝ × ℝ
  r : ℝ × ℝ
  on_circle1 : (q.1 - 0)^2 + q.2^2 = config.radius1^2
  on_circle2 : (r.1 - config.center_distance)^2 + r.2^2 = config.radius2^2
  equal_chords : (q.1 - p.val.1)^2 + (q.2 - p.val.2)^2 = (r.1 - p.val.1)^2 + (r.2 - p.val.2)^2

/-- Theorem: The square of the length of QP is 164 -/
theorem equal_chords_length_squared 
  (config : CircleConfiguration) 
  (line : EqualChordsLine config) : 
  (line.q.1 - line.p.val.1)^2 + (line.q.2 - line.p.val.2)^2 = 164 := by
  sorry

end NUMINAMATH_CALUDE_equal_chords_length_squared_l124_12446


namespace NUMINAMATH_CALUDE_matrix_commute_l124_12498

theorem matrix_commute (C D : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : C + D = C * D) 
  (h2 : C * D = !![5, 1; -2, 4]) : 
  D * C = !![5, 1; -2, 4] := by sorry

end NUMINAMATH_CALUDE_matrix_commute_l124_12498


namespace NUMINAMATH_CALUDE_no_snow_probability_l124_12436

theorem no_snow_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l124_12436


namespace NUMINAMATH_CALUDE_bridge_length_l124_12428

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 265 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l124_12428


namespace NUMINAMATH_CALUDE_red_candy_count_l124_12430

theorem red_candy_count (total : ℕ) (blue : ℕ) (h1 : total = 3409) (h2 : blue = 3264) :
  total - blue = 145 := by
  sorry

end NUMINAMATH_CALUDE_red_candy_count_l124_12430


namespace NUMINAMATH_CALUDE_limits_involving_x_and_n_l124_12486

open Real

/-- For x > 0, prove the limits of two expressions involving n and x as n approaches infinity. -/
theorem limits_involving_x_and_n (x : ℝ) (h : x > 0) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |n * log (1 + x / n) - x| < ε) ∧
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |(1 + x / n)^n - exp x| < ε) :=
sorry

end NUMINAMATH_CALUDE_limits_involving_x_and_n_l124_12486


namespace NUMINAMATH_CALUDE_min_additional_marbles_correct_l124_12464

/-- The number of friends Lisa has -/
def num_friends : ℕ := 10

/-- The initial number of marbles Lisa has -/
def initial_marbles : ℕ := 34

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The minimum number of additional marbles needed -/
def min_additional_marbles : ℕ := sum_first_n num_friends - initial_marbles

theorem min_additional_marbles_correct :
  min_additional_marbles = 21 ∧
  sum_first_n num_friends ≥ initial_marbles + min_additional_marbles ∧
  ∀ k : ℕ, k < min_additional_marbles →
    sum_first_n num_friends > initial_marbles + k :=
by sorry

end NUMINAMATH_CALUDE_min_additional_marbles_correct_l124_12464


namespace NUMINAMATH_CALUDE_language_letters_l124_12471

theorem language_letters (n : ℕ) : 
  (n + n^2) - ((n - 1) + (n - 1)^2) = 129 → n = 65 := by
  sorry

end NUMINAMATH_CALUDE_language_letters_l124_12471


namespace NUMINAMATH_CALUDE_train_or_plane_prob_not_ship_prob_prob_half_combinations_prob_sum_one_l124_12470

-- Define the probabilities of each transportation mode
def train_prob : ℝ := 0.3
def ship_prob : ℝ := 0.2
def car_prob : ℝ := 0.1
def plane_prob : ℝ := 0.4

-- Define the sum of all probabilities
def total_prob : ℝ := train_prob + ship_prob + car_prob + plane_prob

-- Theorem for the probability of taking either a train or a plane
theorem train_or_plane_prob : train_prob + plane_prob = 0.7 := by sorry

-- Theorem for the probability of not taking a ship
theorem not_ship_prob : 1 - ship_prob = 0.8 := by sorry

-- Theorem for the combinations with probability 0.5
theorem prob_half_combinations :
  (train_prob + ship_prob = 0.5 ∧ car_prob + plane_prob = 0.5) := by sorry

-- Ensure that the probabilities sum to 1
theorem prob_sum_one : total_prob = 1 := by sorry

end NUMINAMATH_CALUDE_train_or_plane_prob_not_ship_prob_prob_half_combinations_prob_sum_one_l124_12470


namespace NUMINAMATH_CALUDE_toy_selling_price_l124_12420

/-- Calculates the total selling price of toys given the number of toys sold,
    the cost price per toy, and the number of toys whose cost price equals the total gain. -/
def total_selling_price (num_toys : ℕ) (cost_price : ℕ) (gain_toys : ℕ) : ℕ :=
  num_toys * cost_price + gain_toys * cost_price

/-- Proves that the total selling price of 18 toys is 23100,
    given a cost price of 1100 per toy and a gain equal to the cost of 3 toys. -/
theorem toy_selling_price :
  total_selling_price 18 1100 3 = 23100 := by
  sorry

end NUMINAMATH_CALUDE_toy_selling_price_l124_12420


namespace NUMINAMATH_CALUDE_tangent_line_at_0_1_l124_12407

/-- A line that is tangent to the unit circle at (0, 1) has the equation y = 1 -/
theorem tangent_line_at_0_1 (l : Set (ℝ × ℝ)) :
  (∀ p ∈ l, p.1^2 + p.2^2 = 1 → p = (0, 1)) →  -- l is tangent to the circle
  (0, 1) ∈ l →                                 -- l passes through (0, 1)
  l = {p : ℝ × ℝ | p.2 = 1} :=                 -- l has the equation y = 1
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_0_1_l124_12407


namespace NUMINAMATH_CALUDE_greene_nursery_yellow_carnations_l124_12457

theorem greene_nursery_yellow_carnations :
  let total_flowers : ℕ := 6284
  let red_roses : ℕ := 1491
  let white_roses : ℕ := 1768
  let yellow_carnations : ℕ := total_flowers - (red_roses + white_roses)
  yellow_carnations = 3025 := by
  sorry

end NUMINAMATH_CALUDE_greene_nursery_yellow_carnations_l124_12457


namespace NUMINAMATH_CALUDE_janet_dress_pockets_janet_dress_problem_l124_12404

theorem janet_dress_pockets (total_dresses : ℕ) (dresses_with_pockets : ℕ) 
  (dresses_unknown_pockets : ℕ) (known_pockets : ℕ) (total_pockets : ℕ) : ℕ :=
  let dresses_known_pockets := dresses_with_pockets - dresses_unknown_pockets
  let unknown_pockets := (total_pockets - dresses_known_pockets * known_pockets) / dresses_unknown_pockets
  unknown_pockets

theorem janet_dress_problem : janet_dress_pockets 24 12 4 3 32 = 2 := by
  sorry

end NUMINAMATH_CALUDE_janet_dress_pockets_janet_dress_problem_l124_12404


namespace NUMINAMATH_CALUDE_troys_home_distance_l124_12491

/-- The distance between Troy's home and school -/
def troys_distance : ℝ := 75

/-- The distance between Emily's home and school -/
def emilys_distance : ℝ := 98

/-- The additional distance Emily walks compared to Troy in five days -/
def additional_distance : ℝ := 230

/-- The number of days -/
def days : ℕ := 5

theorem troys_home_distance :
  troys_distance = 75 ∧
  emilys_distance = 98 ∧
  additional_distance = 230 ∧
  days = 5 →
  days * (2 * emilys_distance) - days * (2 * troys_distance) = additional_distance :=
by sorry

end NUMINAMATH_CALUDE_troys_home_distance_l124_12491


namespace NUMINAMATH_CALUDE_circle_radius_square_tangents_l124_12461

theorem circle_radius_square_tangents (side_length : ℝ) (angle : ℝ) (sin_half_angle : ℝ) :
  side_length = Real.sqrt (2 + Real.sqrt 2) →
  angle = π / 4 →
  sin_half_angle = (Real.sqrt (2 - Real.sqrt 2)) / 2 →
  ∃ (radius : ℝ), radius = Real.sqrt 2 + Real.sqrt (2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_square_tangents_l124_12461


namespace NUMINAMATH_CALUDE_value_of_a_l124_12458

theorem value_of_a (a b c : ℤ) 
  (h1 : a + b = c) 
  (h2 : b + c = 7) 
  (h3 : c = 5) : 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l124_12458


namespace NUMINAMATH_CALUDE_probability_same_team_l124_12499

/-- The probability of two volunteers joining the same team out of three teams -/
theorem probability_same_team (num_teams : ℕ) (num_volunteers : ℕ) : 
  num_teams = 3 → num_volunteers = 2 → 
  (num_teams.choose num_volunteers : ℚ) / (num_teams ^ num_volunteers : ℚ) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_probability_same_team_l124_12499


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l124_12444

theorem min_value_of_sum_of_roots (x y : ℝ) :
  let z := Real.sqrt (x^2 + y^2 - 2*x - 2*y + 2) + Real.sqrt (x^2 + y^2 - 4*y + 4)
  z ≥ Real.sqrt 2 ∧
  (z = Real.sqrt 2 ↔ y = 2 - x ∧ 1 ≤ x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l124_12444


namespace NUMINAMATH_CALUDE_worker_c_completion_time_l124_12402

/-- The time it takes for worker c to complete the work alone, given the following conditions:
  * a and b together can complete the work in 2 days
  * b and c together can complete the work in 3 days
  * c and a together can complete the work in 4 days
-/
theorem worker_c_completion_time 
  (a b c : ℝ) -- work rates of workers a, b, and c in units of work per day
  (h1 : a + b = 1/2) -- a and b together complete the work in 2 days
  (h2 : b + c = 1/3) -- b and c together complete the work in 3 days
  (h3 : c + a = 1/4) -- c and a together complete the work in 4 days
  : 1/c = 24 := by
  sorry

end NUMINAMATH_CALUDE_worker_c_completion_time_l124_12402


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l124_12422

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) :
  A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l124_12422


namespace NUMINAMATH_CALUDE_wang_liang_set_exists_l124_12490

theorem wang_liang_set_exists : ∃ (a b : ℕ), 
  1 ≤ a ∧ a ≤ 13 ∧ 
  1 ≤ b ∧ b ≤ 13 ∧ 
  (a - a / b) * b = 24 ∧ 
  (a ≠ 4 ∨ b ≠ 7) := by
  sorry

end NUMINAMATH_CALUDE_wang_liang_set_exists_l124_12490


namespace NUMINAMATH_CALUDE_divisible_by_27_l124_12483

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, (10 ^ n : ℤ) + 18 * n - 1 = 27 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_27_l124_12483


namespace NUMINAMATH_CALUDE_multiply_by_112_equals_70000_l124_12426

theorem multiply_by_112_equals_70000 (x : ℝ) : 112 * x = 70000 → x = 625 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_112_equals_70000_l124_12426


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l124_12408

/-- Given a pentagon ABCDE with the following properties:
  - Angle A measures 80°
  - Angle B measures 95°
  - Angles C and D are equal
  - Angle E is 10° less than three times angle C
  Prove that the largest angle in the pentagon measures 221° -/
theorem largest_angle_in_pentagon (A B C D E : ℝ) : 
  A = 80 ∧ 
  B = 95 ∧ 
  C = D ∧ 
  E = 3 * C - 10 ∧ 
  A + B + C + D + E = 540 →
  max A (max B (max C (max D E))) = 221 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l124_12408


namespace NUMINAMATH_CALUDE_function_describes_relationship_l124_12480

-- Define the set of x values
def X : Set ℕ := {1, 2, 3, 4, 5}

-- Define the function f
def f (x : ℕ) : ℕ := x^2

-- Define the set of points (x, y)
def points : Set (ℕ × ℕ) := {(1, 1), (2, 4), (3, 9), (4, 16), (5, 25)}

-- Theorem statement
theorem function_describes_relationship :
  ∀ (x : ℕ), x ∈ X → (x, f x) ∈ points := by
  sorry

end NUMINAMATH_CALUDE_function_describes_relationship_l124_12480


namespace NUMINAMATH_CALUDE_xy_squared_minus_y_squared_x_equals_zero_l124_12414

theorem xy_squared_minus_y_squared_x_equals_zero (x y : ℝ) : x * y^2 - y^2 * x = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_minus_y_squared_x_equals_zero_l124_12414


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_fraction_multiplication_l124_12419

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem fraction_multiplication :
  (1 / 5 : ℚ) * (1 / 3 : ℚ) * (1 / 4 : ℚ) * 120 = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_fraction_multiplication_l124_12419


namespace NUMINAMATH_CALUDE_james_distance_l124_12481

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: James rode 80 miles -/
theorem james_distance : distance 16 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_james_distance_l124_12481


namespace NUMINAMATH_CALUDE_student_teacher_ratio_l124_12465

/-- Proves that the current ratio of students to teachers is 50:1 given the problem conditions -/
theorem student_teacher_ratio 
  (current_teachers : ℕ) 
  (current_students : ℕ) 
  (h1 : current_teachers = 3)
  (h2 : (current_students + 50) / (current_teachers + 5) = 25) :
  current_students / current_teachers = 50 := by
sorry

end NUMINAMATH_CALUDE_student_teacher_ratio_l124_12465


namespace NUMINAMATH_CALUDE_initial_jar_state_l124_12450

/-- Represents the initial state of the jar of balls -/
structure JarState where
  totalBalls : ℕ
  blueBalls : ℕ
  nonBlueBalls : ℕ
  hTotalSum : totalBalls = blueBalls + nonBlueBalls

/-- Represents the state of the jar after removing some blue balls -/
structure UpdatedJarState where
  initialState : JarState
  removedBlueBalls : ℕ
  newBlueBalls : ℕ
  hNewBlue : newBlueBalls = initialState.blueBalls - removedBlueBalls
  newProbability : ℚ
  hProbability : newProbability = newBlueBalls / (initialState.totalBalls - removedBlueBalls)

/-- The main theorem stating the initial number of balls in the jar -/
theorem initial_jar_state 
  (updatedState : UpdatedJarState)
  (hInitialBlue : updatedState.initialState.blueBalls = 9)
  (hRemoved : updatedState.removedBlueBalls = 5)
  (hNewProb : updatedState.newProbability = 1/5) :
  updatedState.initialState.totalBalls = 25 := by
  sorry

end NUMINAMATH_CALUDE_initial_jar_state_l124_12450


namespace NUMINAMATH_CALUDE_f_no_real_roots_l124_12445

/-- Defines the polynomial f(x) for a given positive integer n -/
def f (n : ℕ+) (x : ℝ) : ℝ :=
  (2 * n.val + 1) * x^(2 * n.val) - 2 * n.val * x^(2 * n.val - 1) + 
  (2 * n.val - 1) * x^(2 * n.val - 2) - 3 * x^2 + 2 * x - 1

/-- Theorem stating that f(x) has no real roots for any positive integer n -/
theorem f_no_real_roots (n : ℕ+) : ∀ x : ℝ, f n x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_f_no_real_roots_l124_12445


namespace NUMINAMATH_CALUDE_smallest_number_after_removal_largest_number_after_removal_l124_12469

-- Define the original number as a string
def originalNumber : String := "123456789101112...5657585960"

-- Define the number of digits to remove
def digitsToRemove : Nat := 100

-- Define the function to remove digits and get the smallest number
def smallestAfterRemoval (s : String) (n : Nat) : Nat :=
  sorry

-- Define the function to remove digits and get the largest number
def largestAfterRemoval (s : String) (n : Nat) : Nat :=
  sorry

-- Theorem for the smallest number
theorem smallest_number_after_removal :
  smallestAfterRemoval originalNumber digitsToRemove = 123450 :=
by sorry

-- Theorem for the largest number
theorem largest_number_after_removal :
  largestAfterRemoval originalNumber digitsToRemove = 56758596049 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_after_removal_largest_number_after_removal_l124_12469


namespace NUMINAMATH_CALUDE_olympic_high_school_quiz_l124_12476

theorem olympic_high_school_quiz (f s : ℚ) 
  (h1 : f > 0) 
  (h2 : s > 0) 
  (h3 : (3/7) * f = (5/7) * s) : 
  f = (5/3) * s :=
sorry

end NUMINAMATH_CALUDE_olympic_high_school_quiz_l124_12476


namespace NUMINAMATH_CALUDE_talia_father_current_age_talia_future_age_talia_mom_current_age_talia_father_future_age_l124_12460

-- Define the current year as a reference point
def current_year : ℕ := 0

-- Define Talia's age
def talia_age : ℕ → ℕ
  | year => 13 + year

-- Define Talia's mom's age
def talia_mom_age : ℕ → ℕ
  | year => 3 * talia_age current_year + year

-- Define Talia's father's age
def talia_father_age : ℕ → ℕ
  | year => talia_mom_age current_year + (year - 3)

-- State the theorem
theorem talia_father_current_age :
  talia_father_age current_year = 36 :=
by
  sorry

-- Conditions as separate theorems
theorem talia_future_age :
  talia_age 7 = 20 :=
by
  sorry

theorem talia_mom_current_age :
  talia_mom_age current_year = 3 * talia_age current_year :=
by
  sorry

theorem talia_father_future_age :
  talia_father_age 3 = talia_mom_age current_year :=
by
  sorry

end NUMINAMATH_CALUDE_talia_father_current_age_talia_future_age_talia_mom_current_age_talia_father_future_age_l124_12460


namespace NUMINAMATH_CALUDE_cube_surface_area_l124_12448

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) :
  volume = 8 →
  volume = side^3 →
  surface_area = 6 * side^2 →
  surface_area = 24 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l124_12448


namespace NUMINAMATH_CALUDE_mike_total_spent_l124_12466

def marbles_cost : ℝ := 9.05
def football_cost : ℝ := 4.95
def baseball_cost : ℝ := 6.52
def toy_car_cost : ℝ := 3.75
def puzzle_cost : ℝ := 8.99
def stickers_cost : ℝ := 1.25
def puzzle_discount : ℝ := 0.15
def toy_car_discount : ℝ := 0.10
def coupon_value : ℝ := 5.00

def discounted_puzzle_cost : ℝ := puzzle_cost * (1 - puzzle_discount)
def discounted_toy_car_cost : ℝ := toy_car_cost * (1 - toy_car_discount)

def total_cost : ℝ := marbles_cost + football_cost + baseball_cost + 
                       discounted_toy_car_cost + discounted_puzzle_cost + 
                       stickers_cost - coupon_value

theorem mike_total_spent :
  total_cost = 27.7865 :=
by sorry

end NUMINAMATH_CALUDE_mike_total_spent_l124_12466


namespace NUMINAMATH_CALUDE_number_exceeds_16_percent_l124_12409

theorem number_exceeds_16_percent : ∃ x : ℝ, x = 100 ∧ x = 0.16 * x + 84 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeds_16_percent_l124_12409


namespace NUMINAMATH_CALUDE_division_theorem_l124_12421

/-- The dividend polynomial -/
def p (x : ℝ) : ℝ := 3 * x^5 - 2 * x^3 + 5 * x - 8

/-- The divisor polynomial -/
def d (x : ℝ) : ℝ := x^2 - 2 * x + 1

/-- The proposed remainder -/
def r (x : ℝ) : ℝ := 14 * x - 14

/-- Theorem stating that r is the remainder when p is divided by d -/
theorem division_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, p x = d x * q x + r x :=
sorry

end NUMINAMATH_CALUDE_division_theorem_l124_12421


namespace NUMINAMATH_CALUDE_square_sum_17_5_l124_12432

theorem square_sum_17_5 : 17^2 + 2*(17*5) + 5^2 = 484 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_17_5_l124_12432


namespace NUMINAMATH_CALUDE_total_pieces_l124_12441

/-- Represents the number of small pieces in Figure n of Nair's puzzle -/
def small_pieces (n : ℕ) : ℕ := 4 * n

/-- Represents the number of large pieces in Figure n of Nair's puzzle -/
def large_pieces (n : ℕ) : ℕ := n^2 - n

/-- Theorem stating that the total number of pieces in Figure n is n^2 + 3n -/
theorem total_pieces (n : ℕ) : small_pieces n + large_pieces n = n^2 + 3*n := by
  sorry

#eval small_pieces 20 + large_pieces 20  -- Should output 460

end NUMINAMATH_CALUDE_total_pieces_l124_12441


namespace NUMINAMATH_CALUDE_pet_shop_total_l124_12487

/-- Given a pet shop with dogs, cats, and bunnies in stock, prove that the total number of dogs and bunnies is 375. -/
theorem pet_shop_total (dogs cats bunnies : ℕ) : 
  dogs = 75 →
  dogs / 3 = cats / 7 →
  dogs / 3 = bunnies / 12 →
  dogs + bunnies = 375 := by
sorry


end NUMINAMATH_CALUDE_pet_shop_total_l124_12487


namespace NUMINAMATH_CALUDE_third_candidate_votes_l124_12456

theorem third_candidate_votes (total_votes : ℕ) 
  (h1 : total_votes = 52500)
  (h2 : ∃ (c1 c2 c3 : ℕ), c1 + c2 + c3 = total_votes ∧ c1 = 2500 ∧ c2 = 15000)
  (h3 : ∃ (winner : ℕ), winner = (2 : ℚ) / 3 * total_votes) :
  ∃ (third : ℕ), third = 35000 := by
sorry

end NUMINAMATH_CALUDE_third_candidate_votes_l124_12456


namespace NUMINAMATH_CALUDE_product_of_special_numbers_l124_12406

theorem product_of_special_numbers (a b : ℝ) 
  (ha : a = Real.exp (2 - a)) 
  (hb : 1 + Real.log b = Real.exp (2 - (1 + Real.log b))) : 
  a * b = Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_product_of_special_numbers_l124_12406


namespace NUMINAMATH_CALUDE_octal_number_check_l124_12477

def is_octal_digit (d : Nat) : Prop := d < 8

def is_octal_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 8 → is_octal_digit d

theorem octal_number_check :
  ¬ is_octal_number 8102 ∧
  ¬ is_octal_number 793 ∧
  is_octal_number 214 ∧
  ¬ is_octal_number 998 := by sorry

end NUMINAMATH_CALUDE_octal_number_check_l124_12477


namespace NUMINAMATH_CALUDE_find_N_l124_12489

theorem find_N (a b c N : ℚ) : 
  a + b + c = 120 ∧
  a - 10 = N ∧
  b + 10 = N ∧
  7 * c = N →
  N = 56 := by
sorry

end NUMINAMATH_CALUDE_find_N_l124_12489


namespace NUMINAMATH_CALUDE_simplified_fraction_ratio_l124_12440

theorem simplified_fraction_ratio (k c d : ℤ) : 
  (5 * k + 15) / 5 = c * k + d → c / d = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_ratio_l124_12440


namespace NUMINAMATH_CALUDE_angle_C_measure_l124_12494

-- Define the triangle and its angles
structure Triangle :=
  (A B C : ℝ)

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.B = t.A + 20 ∧ t.C = t.A + 40 ∧ t.A + t.B + t.C = 180

-- Theorem statement
theorem angle_C_measure (t : Triangle) :
  satisfies_conditions t → t.C = 80 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l124_12494


namespace NUMINAMATH_CALUDE_triangle_problem_l124_12401

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π / 2 →  -- Acute angle A
  0 < B ∧ B < π / 2 →  -- Acute angle B
  0 < C ∧ C < π / 2 →  -- Acute angle C
  A + B + C = π →      -- Sum of angles in a triangle
  b + c = 10 →         -- Given condition
  a = Real.sqrt 10 →   -- Given condition
  5 * b * Real.sin A * Real.cos C + 5 * c * Real.sin A * Real.cos B = 3 * Real.sqrt 10 → -- Given condition
  Real.cos A = 4 / 5 ∧ b = 5 ∧ c = 5 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l124_12401


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l124_12479

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) → (a < -2 ∨ a > 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l124_12479


namespace NUMINAMATH_CALUDE_fred_movie_change_l124_12417

theorem fred_movie_change 
  (ticket_price : ℚ)
  (num_tickets : ℕ)
  (borrowed_movie_price : ℚ)
  (paid_amount : ℚ)
  (h1 : ticket_price = 592/100)
  (h2 : num_tickets = 2)
  (h3 : borrowed_movie_price = 679/100)
  (h4 : paid_amount = 20) :
  paid_amount - (ticket_price * num_tickets + borrowed_movie_price) = 137/100 := by
  sorry

end NUMINAMATH_CALUDE_fred_movie_change_l124_12417


namespace NUMINAMATH_CALUDE_sin_pi_half_plus_alpha_l124_12416

/-- Given a point P(-4, 3) on the terminal side of angle α, prove that sin(π/2 + α) = -4/5 -/
theorem sin_pi_half_plus_alpha (α : ℝ) : 
  let P : ℝ × ℝ := (-4, 3)
  Real.sin (π / 2 + α) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_half_plus_alpha_l124_12416


namespace NUMINAMATH_CALUDE_train_crossing_time_l124_12429

/-- Proves that a train 40 meters long, traveling at 144 km/hr, will take 1 second to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 40 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 1 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l124_12429


namespace NUMINAMATH_CALUDE_isabel_math_pages_l124_12497

/-- The number of pages of math homework Isabel had -/
def math_pages : ℕ := sorry

/-- The number of pages of reading homework Isabel had -/
def reading_pages : ℕ := 4

/-- The number of problems per page -/
def problems_per_page : ℕ := 5

/-- The total number of problems Isabel had to complete -/
def total_problems : ℕ := 30

/-- Theorem stating that Isabel had 2 pages of math homework -/
theorem isabel_math_pages : math_pages = 2 := by
  sorry

end NUMINAMATH_CALUDE_isabel_math_pages_l124_12497


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l124_12447

theorem sum_of_quadratic_solutions : 
  let f : ℝ → ℝ := λ x => x^2 - 6*x + 5 - (2*x - 8)
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l124_12447


namespace NUMINAMATH_CALUDE_laura_has_435_l124_12467

/-- Calculates Laura's money given Darwin's money -/
def lauras_money (darwins_money : ℕ) : ℕ :=
  let mias_money := 2 * darwins_money + 20
  let combined_money := mias_money + darwins_money
  3 * combined_money - 30

/-- Proves that Laura has $435 given the conditions -/
theorem laura_has_435 : lauras_money 45 = 435 := by
  sorry

end NUMINAMATH_CALUDE_laura_has_435_l124_12467


namespace NUMINAMATH_CALUDE_teddy_cats_count_l124_12496

/-- Prove that Teddy has 8 cats given the conditions of the problem -/
theorem teddy_cats_count :
  -- Teddy's dogs
  let teddy_dogs : ℕ := 7
  -- Ben's dogs relative to Teddy's
  let ben_dogs : ℕ := teddy_dogs + 9
  -- Dave's dogs relative to Teddy's
  let dave_dogs : ℕ := teddy_dogs - 5
  -- Dave's cats relative to Teddy's
  let dave_cats (teddy_cats : ℕ) : ℕ := teddy_cats + 13
  -- Total pets
  let total_pets : ℕ := 54
  -- The number of Teddy's cats that satisfies all conditions
  ∃ (teddy_cats : ℕ),
    teddy_dogs + ben_dogs + dave_dogs + teddy_cats + dave_cats teddy_cats = total_pets ∧
    teddy_cats = 8 := by
  sorry

end NUMINAMATH_CALUDE_teddy_cats_count_l124_12496


namespace NUMINAMATH_CALUDE_cosine_expression_simplification_fraction_simplification_l124_12474

-- Part 1
theorem cosine_expression_simplification :
  2 * Real.cos (45 * π / 180) - (-2 * Real.sqrt 3) ^ 0 + 1 / (Real.sqrt 2 + 1) - Real.sqrt 8 = -2 := by
  sorry

-- Part 2
theorem fraction_simplification (x : ℝ) (h : x = -Real.sqrt 2) :
  (3 / (x - 1) - x - 1) / ((x - 2) / (x^2 - 2*x + 1)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_expression_simplification_fraction_simplification_l124_12474


namespace NUMINAMATH_CALUDE_mack_journal_pages_l124_12424

/-- Calculates the number of pages written given time and rate -/
def pages_written (time minutes_per_page : ℕ) : ℕ :=
  time / minutes_per_page

/-- Represents Mack's journal writing over four days -/
structure JournalWriting where
  monday_time : ℕ
  monday_rate : ℕ
  tuesday_time : ℕ
  tuesday_rate : ℕ
  wednesday_pages : ℕ
  thursday_time1 : ℕ
  thursday_rate1 : ℕ
  thursday_time2 : ℕ
  thursday_rate2 : ℕ

/-- Calculates the total pages written over four days -/
def total_pages (j : JournalWriting) : ℕ :=
  pages_written j.monday_time j.monday_rate +
  pages_written j.tuesday_time j.tuesday_rate +
  j.wednesday_pages +
  pages_written j.thursday_time1 j.thursday_rate1 +
  pages_written j.thursday_time2 j.thursday_rate2

/-- Theorem stating the total pages written is 16 -/
theorem mack_journal_pages :
  ∀ j : JournalWriting,
    j.monday_time = 60 ∧
    j.monday_rate = 30 ∧
    j.tuesday_time = 45 ∧
    j.tuesday_rate = 15 ∧
    j.wednesday_pages = 5 ∧
    j.thursday_time1 = 30 ∧
    j.thursday_rate1 = 10 ∧
    j.thursday_time2 = 60 ∧
    j.thursday_rate2 = 20 →
    total_pages j = 16 := by
  sorry

end NUMINAMATH_CALUDE_mack_journal_pages_l124_12424


namespace NUMINAMATH_CALUDE_linear_equation_equivalence_l124_12438

theorem linear_equation_equivalence (x y : ℝ) :
  (2 * x - y = 3) ↔ (y = 2 * x - 3) := by sorry

end NUMINAMATH_CALUDE_linear_equation_equivalence_l124_12438


namespace NUMINAMATH_CALUDE_alok_order_cost_l124_12484

def chapati_quantity : ℕ := 16
def chapati_price : ℕ := 6
def rice_quantity : ℕ := 5
def rice_price : ℕ := 45
def vegetable_quantity : ℕ := 7
def vegetable_price : ℕ := 70

def total_cost : ℕ := chapati_quantity * chapati_price + 
                      rice_quantity * rice_price + 
                      vegetable_quantity * vegetable_price

theorem alok_order_cost : total_cost = 811 := by
  sorry

end NUMINAMATH_CALUDE_alok_order_cost_l124_12484


namespace NUMINAMATH_CALUDE_exactly_one_true_l124_12468

def proposition1 : Prop := ∀ x : ℝ, x^4 > x^2

def proposition2 : Prop := ∀ p q : Prop, (¬(p ∧ q)) → (¬p ∧ ¬q)

def proposition3 : Prop := (¬∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0)

theorem exactly_one_true : 
  (proposition1 ∧ ¬proposition2 ∧ ¬proposition3) ∨
  (¬proposition1 ∧ proposition2 ∧ ¬proposition3) ∨
  (¬proposition1 ∧ ¬proposition2 ∧ proposition3) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_true_l124_12468


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l124_12400

/-- Calculates the principal amount of a loan given the interest rate, time, and total interest. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  interest / (rate * time)

/-- Theorem stating that for a loan with 12% annual simple interest rate,
    where the interest after 3 years is $3600, the principal amount is $10,000. -/
theorem loan_principal_calculation :
  let rate : ℚ := 12 / 100
  let time : ℕ := 3
  let interest : ℚ := 3600
  calculate_principal rate time interest = 10000 := by
  sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l124_12400


namespace NUMINAMATH_CALUDE_expression_evaluation_l124_12463

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  ((x - 2*y)*x - (x - 2*y)*(x + 2*y)) / y = -8 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l124_12463


namespace NUMINAMATH_CALUDE_min_value_of_expression_l124_12454

theorem min_value_of_expression (a b c : ℝ) 
  (h : ∀ (x y : ℝ), 3*x + 4*y - 5 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ 3*x + 4*y + 5) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (v : ℝ), (v = a + b - c → v ≥ m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l124_12454


namespace NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l124_12418

/-- Given a hyperbola with equation x^2/a^2 - y^2/b^2 = 1 and eccentricity 2,
    if the product of the distances from a point on the hyperbola to its two asymptotes is 3/4,
    then the length of the conjugate axis is 2√3. -/
theorem hyperbola_conjugate_axis_length 
  (a b : ℝ) 
  (h1 : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → 
    (|b*x - a*y| * |b*x + a*y|) / (a^2 + b^2) = 3/4)
  (h2 : a^2 + b^2 = 5*a^2) :
  2*b = 2*Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l124_12418


namespace NUMINAMATH_CALUDE_mikes_ride_distance_l124_12433

theorem mikes_ride_distance (mike_start_fee annie_start_fee : ℚ)
  (annie_bridge_toll : ℚ) (cost_per_mile : ℚ) (annie_distance : ℚ)
  (h1 : mike_start_fee = 2.5)
  (h2 : annie_start_fee = 2.5)
  (h3 : annie_bridge_toll = 5)
  (h4 : cost_per_mile = 0.25)
  (h5 : annie_distance = 22)
  (h6 : ∃ (mike_distance : ℚ),
    mike_start_fee + cost_per_mile * mike_distance =
    annie_start_fee + annie_bridge_toll + cost_per_mile * annie_distance) :
  ∃ (mike_distance : ℚ), mike_distance = 32 :=
by sorry

end NUMINAMATH_CALUDE_mikes_ride_distance_l124_12433


namespace NUMINAMATH_CALUDE_maria_chairs_l124_12472

/-- The number of chairs Maria bought -/
def num_chairs : ℕ := 2

/-- The number of tables Maria bought -/
def num_tables : ℕ := 2

/-- The time spent on each piece of furniture (in minutes) -/
def time_per_furniture : ℕ := 8

/-- The total time spent (in minutes) -/
def total_time : ℕ := 32

theorem maria_chairs :
  num_chairs * time_per_furniture + num_tables * time_per_furniture = total_time :=
by sorry

end NUMINAMATH_CALUDE_maria_chairs_l124_12472


namespace NUMINAMATH_CALUDE_divisibility_of_polynomials_l124_12443

theorem divisibility_of_polynomials : ∃ q : Polynomial ℤ, 
  X^55 + X^44 + X^33 + X^22 + X^11 + 1 = q * (X^5 + X^4 + X^3 + X^2 + X + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_polynomials_l124_12443


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l124_12495

theorem imaginary_part_of_z : Complex.im ((-3 + Complex.I) / Complex.I^3) = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l124_12495


namespace NUMINAMATH_CALUDE_g_ge_f_implies_t_range_l124_12431

noncomputable def g (x : ℝ) : ℝ := Real.log x + 3 / (4 * x) - (1 / 4) * x - 1

def f (t x : ℝ) : ℝ := x^2 - 2 * t * x + 4

theorem g_ge_f_implies_t_range (t : ℝ) :
  (∀ x1 ∈ Set.Ioo 0 2, ∃ x2 ∈ Set.Icc 1 2, g x1 ≥ f t x2) →
  t ≥ 17/8 :=
by sorry

end NUMINAMATH_CALUDE_g_ge_f_implies_t_range_l124_12431


namespace NUMINAMATH_CALUDE_sum_base4_equals_1332_l124_12455

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (a b c : ℕ) : ℕ := a * 4^2 + b * 4 + c

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let d := n / (4^3)
  let r := n % (4^3)
  let c := r / (4^2)
  let r' := r % (4^2)
  let b := r' / 4
  let a := r' % 4
  (d, c, b, a)

theorem sum_base4_equals_1332 :
  let sum := base4ToBase10 2 1 3 + base4ToBase10 1 3 2 + base4ToBase10 3 2 1
  base10ToBase4 sum = (1, 3, 3, 2) := by sorry

end NUMINAMATH_CALUDE_sum_base4_equals_1332_l124_12455


namespace NUMINAMATH_CALUDE_min_ABFG_value_l124_12423

/-- Represents a seven-digit number ABCDEFG -/
structure SevenDigitNumber where
  digits : Fin 7 → Nat
  is_valid : ∀ i, digits i < 10

/-- Extracts a five-digit number from a seven-digit number -/
def extract_five_digits (n : SevenDigitNumber) (start : Fin 3) : Nat :=
  (n.digits start) * 10000 + (n.digits (start + 1)) * 1000 + 
  (n.digits (start + 2)) * 100 + (n.digits (start + 3)) * 10 + 
  (n.digits (start + 4))

/-- Extracts a four-digit number ABFG from a seven-digit number ABCDEFG -/
def extract_ABFG (n : SevenDigitNumber) : Nat :=
  (n.digits 0) * 1000 + (n.digits 1) * 100 + (n.digits 5) * 10 + (n.digits 6)

/-- The main theorem -/
theorem min_ABFG_value (n : SevenDigitNumber) 
  (h1 : extract_five_digits n 1 % 2013 = 0)
  (h2 : extract_five_digits n 3 % 1221 = 0) :
  3036 ≤ extract_ABFG n :=
sorry

end NUMINAMATH_CALUDE_min_ABFG_value_l124_12423


namespace NUMINAMATH_CALUDE_abs_leq_two_necessary_not_sufficient_l124_12473

theorem abs_leq_two_necessary_not_sufficient :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x| ≤ 2) ∧
  (∃ x : ℝ, |x| ≤ 2 ∧ ¬(0 ≤ x ∧ x ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_abs_leq_two_necessary_not_sufficient_l124_12473


namespace NUMINAMATH_CALUDE_largest_percent_error_circle_area_l124_12425

/-- The largest possible percent error in the computed area of a circle, given a measurement error in its diameter --/
theorem largest_percent_error_circle_area (actual_diameter : ℝ) (max_error_percent : ℝ) :
  actual_diameter = 20 →
  max_error_percent = 20 →
  let max_measured_diameter := actual_diameter * (1 + max_error_percent / 100)
  let actual_area := Real.pi * (actual_diameter / 2) ^ 2
  let max_computed_area := Real.pi * (max_measured_diameter / 2) ^ 2
  let max_percent_error := (max_computed_area - actual_area) / actual_area * 100
  max_percent_error = 44 := by sorry

end NUMINAMATH_CALUDE_largest_percent_error_circle_area_l124_12425


namespace NUMINAMATH_CALUDE_min_games_is_eight_l124_12403

/-- Represents a Go tournament between China and Japan -/
structure GoTournament where
  max_games : ℕ
  players_per_side : ℕ

/-- The maximum number of games in the tournament does not exceed 15 -/
def max_games_condition (t : GoTournament) : Prop :=
  t.max_games ≤ 15

/-- The minimum number of games is equal to the number of players per side -/
def min_games (t : GoTournament) : ℕ :=
  t.players_per_side

/-- Theorem stating that the minimum number of games in the tournament is 8 -/
theorem min_games_is_eight (t : GoTournament) 
  (h : max_games_condition t) : min_games t = 8 := by
  sorry

#check min_games_is_eight

end NUMINAMATH_CALUDE_min_games_is_eight_l124_12403


namespace NUMINAMATH_CALUDE_point_translation_l124_12449

/-- Given a point M(-2, 3) in the Cartesian coordinate system,
    prove that after translating it 3 units downwards and then 1 unit to the right,
    the resulting point has coordinates (-1, 0). -/
theorem point_translation (M : ℝ × ℝ) :
  M = (-2, 3) →
  let M' := (M.1, M.2 - 3)  -- Translate 3 units downwards
  let M'' := (M'.1 + 1, M'.2)  -- Translate 1 unit to the right
  M'' = (-1, 0) := by
sorry

end NUMINAMATH_CALUDE_point_translation_l124_12449


namespace NUMINAMATH_CALUDE_alpha_range_l124_12405

theorem alpha_range (α : Real) 
  (h1 : 0 ≤ α ∧ α < 2 * Real.pi) 
  (h2 : Real.sin α > Real.sqrt 3 * Real.cos α) : 
  Real.pi / 3 < α ∧ α < 4 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_alpha_range_l124_12405


namespace NUMINAMATH_CALUDE_ascending_order_fractions_l124_12412

theorem ascending_order_fractions (x y : ℝ) (hx : x < 0) (hy : 0 < y ∧ y < 1) :
  (1 / x) < (y / x) ∧ (y / x) < (y^2 / x) := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_fractions_l124_12412


namespace NUMINAMATH_CALUDE_unique_white_bucket_count_l124_12437

/-- Represents a bucket with its color and water content -/
structure Bucket :=
  (color : Bool)  -- true for red, false for white
  (water : ℕ)

/-- Represents a move where water is added to a pair of buckets -/
structure Move :=
  (red_bucket : ℕ)
  (white_bucket : ℕ)
  (water_added : ℕ)

/-- The main theorem statement -/
theorem unique_white_bucket_count
  (red_count : ℕ)
  (white_count : ℕ)
  (moves : List Move)
  (h_red_count : red_count = 100)
  (h_all_non_empty : ∀ b : Bucket, b.water > 0)
  (h_equal_water : ∀ m : Move, ∃ b1 b2 : Bucket,
    b1.color = true ∧ b2.color = false ∧
    b1.water = b2.water) :
  white_count = 100 := by
  sorry

end NUMINAMATH_CALUDE_unique_white_bucket_count_l124_12437


namespace NUMINAMATH_CALUDE_arithmetic_equation_l124_12482

theorem arithmetic_equation : 8 / 4 - 3^2 + 4 * 5 = 13 := by sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l124_12482


namespace NUMINAMATH_CALUDE_cubic_linear_inequality_l124_12493

theorem cubic_linear_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^3 + b^3 + a + b ≥ 4 * a * b := by
  sorry

end NUMINAMATH_CALUDE_cubic_linear_inequality_l124_12493


namespace NUMINAMATH_CALUDE_squareable_numbers_l124_12427

def isSquareable (n : ℕ) : Prop :=
  ∃ (p : Fin n → Fin n), Function.Bijective p ∧
    ∀ i : Fin n, ∃ k : ℕ, (p i).val.succ + (i.val.succ) = k * k

theorem squareable_numbers : 
  isSquareable 9 ∧ 
  isSquareable 15 ∧ 
  ¬isSquareable 7 ∧ 
  ¬isSquareable 11 :=
sorry

end NUMINAMATH_CALUDE_squareable_numbers_l124_12427


namespace NUMINAMATH_CALUDE_parabola_shift_l124_12439

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shift amount
def shift : ℝ := 1

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x + shift)^2

-- Theorem statement
theorem parabola_shift :
  ∀ x y : ℝ, y = original_parabola (x + shift) ↔ y = shifted_parabola x :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l124_12439


namespace NUMINAMATH_CALUDE_tom_initial_investment_l124_12434

/-- Represents the business partnership between Tom and Jose -/
structure Partnership where
  tom_investment : ℕ
  jose_investment : ℕ
  tom_join_time : ℕ
  jose_join_time : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Tom's initial investment given the partnership details -/
def calculate_tom_investment (p : Partnership) : ℕ :=
  sorry

/-- Theorem stating that Tom's initial investment is 3000 given the problem conditions -/
theorem tom_initial_investment :
  let p : Partnership := {
    tom_investment := 0,  -- We don't know this value yet
    jose_investment := 45000,
    tom_join_time := 0,  -- Tom joined at the start
    jose_join_time := 2,
    total_profit := 54000,
    jose_profit := 30000
  }
  calculate_tom_investment p = 3000 := by
  sorry

end NUMINAMATH_CALUDE_tom_initial_investment_l124_12434


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l124_12485

/-- Represents the number of chocolate bars in the gigantic box -/
def chocolate_bars_in_gigantic_box : ℕ :=
  let small_box_bars : ℕ := 45
  let medium_box_small_boxes : ℕ := 10
  let large_box_medium_boxes : ℕ := 25
  let gigantic_box_large_boxes : ℕ := 50
  gigantic_box_large_boxes * large_box_medium_boxes * medium_box_small_boxes * small_box_bars

/-- Theorem stating that the number of chocolate bars in the gigantic box is 562,500 -/
theorem chocolate_bars_count : chocolate_bars_in_gigantic_box = 562500 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l124_12485


namespace NUMINAMATH_CALUDE_unique_n_divisibility_l124_12452

theorem unique_n_divisibility : ∃! n : ℕ, 0 < n ∧ n < 11 ∧ (18888 - n) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_divisibility_l124_12452


namespace NUMINAMATH_CALUDE_equal_sequences_l124_12411

theorem equal_sequences (n : ℕ) (a b : Fin n → ℕ) 
  (h_gcd : Nat.gcd n 6 = 1)
  (h_a_pos : ∀ i, a i > 0)
  (h_b_pos : ∀ i, b i > 0)
  (h_a_inc : ∀ i j, i < j → a i < a j)
  (h_b_inc : ∀ i j, i < j → b i < b j)
  (h_sum_eq : ∀ j k l, j < k → k < l → a j + a k + a l = b j + b k + b l) :
  ∀ i, a i = b i :=
sorry

end NUMINAMATH_CALUDE_equal_sequences_l124_12411


namespace NUMINAMATH_CALUDE_inequality_proof_l124_12451

theorem inequality_proof (a A b B : ℝ) 
  (h1 : |A - 3*a| ≤ 1 - a)
  (h2 : |B - 3*b| ≤ 1 - b)
  (ha : a > 0)
  (hb : b > 0) :
  |A*B/3 - 3*a*b| - 3*a*b ≤ 1 - a*b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l124_12451


namespace NUMINAMATH_CALUDE_cube_sum_inverse_l124_12492

theorem cube_sum_inverse (x R S : ℝ) (hx : x ≠ 0) : 
  x + 1 / x = R → x^3 + 1 / x^3 = S → S = R^3 - 3 * R :=
by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inverse_l124_12492


namespace NUMINAMATH_CALUDE_right_triangle_check_l124_12462

theorem right_triangle_check (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_check_l124_12462
