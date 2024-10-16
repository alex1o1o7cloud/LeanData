import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_when_prime_exists_counterexample_for_composite_l1749_174904

-- Define the divisibility relation
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

-- Statement for the case when n is prime
theorem divisibility_when_prime (m n : ℕ) (h1 : n > 1) (h2 : ∀ k, 1 < k → k < n → ¬ divides k n) 
  (h3 : divides (m + n) (m * n)) : divides n m := by sorry

-- Statement for the case when n is a product of two distinct primes
theorem exists_counterexample_for_composite : 
  ∃ m n p q : ℕ, p ≠ q ∧ p.Prime ∧ q.Prime ∧ n = p * q ∧ 
  divides (m + n) (m * n) ∧ ¬ divides n m := by sorry

end NUMINAMATH_CALUDE_divisibility_when_prime_exists_counterexample_for_composite_l1749_174904


namespace NUMINAMATH_CALUDE_seven_people_circular_permutations_l1749_174980

/-- The number of distinct seating arrangements for n people around a round table,
    where rotations are considered the same. -/
def circularPermutations (n : ℕ) : ℕ := (n - 1).factorial

/-- Theorem: The number of distinct seating arrangements for 7 people around a round table,
    where rotations are considered the same, is equal to 720. -/
theorem seven_people_circular_permutations :
  circularPermutations 7 = 720 := by
  sorry

end NUMINAMATH_CALUDE_seven_people_circular_permutations_l1749_174980


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1749_174915

theorem isosceles_triangle_base_length 
  (perimeter : ℝ) 
  (one_side : ℝ) 
  (h_perimeter : perimeter = 15) 
  (h_one_side : one_side = 3) 
  (h_isosceles : ∃ (leg : ℝ), 2 * leg + one_side = perimeter) :
  one_side = 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1749_174915


namespace NUMINAMATH_CALUDE_intersection_M_N_l1749_174934

def M : Set ℝ := {x | 2 * x - x^2 ≥ 0}

def N : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (1 - x^2)}

theorem intersection_M_N : M ∩ N = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1749_174934


namespace NUMINAMATH_CALUDE_relationship_abc_l1749_174909

theorem relationship_abc : ∀ (a b c : ℝ), 
  a = Real.sqrt 0.5 → 
  b = 2^(0.5 : ℝ) → 
  c = 0.5^(0.2 : ℝ) → 
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1749_174909


namespace NUMINAMATH_CALUDE_negative_rational_identification_l1749_174986

theorem negative_rational_identification :
  let a := -(-2010)
  let b := -|-2010|
  let c := (-2011)^2010
  let d := -2010 / -2011
  (¬ (a < 0 ∧ ∃ (p q : ℤ), a = p / q ∧ q ≠ 0)) ∧
  (b < 0 ∧ ∃ (p q : ℤ), b = p / q ∧ q ≠ 0) ∧
  (¬ (c < 0 ∧ ∃ (p q : ℤ), c = p / q ∧ q ≠ 0)) ∧
  (¬ (d < 0 ∧ ∃ (p q : ℤ), d = p / q ∧ q ≠ 0)) :=
by sorry


end NUMINAMATH_CALUDE_negative_rational_identification_l1749_174986


namespace NUMINAMATH_CALUDE_race_result_l1749_174930

-- Define the participants
inductive Participant
| Hare
| Fox
| Moose

-- Define the possible positions
inductive Position
| First
| Second

-- Define the statements made by the squirrels
def squirrel1_statement (winner : Participant) (second : Participant) : Prop :=
  winner = Participant.Hare ∧ second = Participant.Fox

def squirrel2_statement (winner : Participant) (second : Participant) : Prop :=
  winner = Participant.Moose ∧ second = Participant.Hare

-- Define the owl's statement
def owl_statement (s1 : Prop) (s2 : Prop) : Prop :=
  (s1 ∧ ¬s2) ∨ (¬s1 ∧ s2)

-- The main theorem
theorem race_result :
  ∃ (winner second : Participant),
    owl_statement (squirrel1_statement winner second) (squirrel2_statement winner second) →
    winner = Participant.Moose ∧ second = Participant.Fox :=
by sorry

end NUMINAMATH_CALUDE_race_result_l1749_174930


namespace NUMINAMATH_CALUDE_ages_ratio_years_ago_sum_of_ages_correct_years_ago_l1749_174951

/-- The number of years ago when the ages of A, B, and C were in the ratio 1 : 2 : 3 -/
def years_ago : ℕ := 3

/-- The present age of A -/
def A_age : ℕ := 11

/-- The present age of B -/
def B_age : ℕ := 22

/-- The present age of C -/
def C_age : ℕ := 24

/-- The theorem stating that the ages were in ratio 1:2:3 some years ago -/
theorem ages_ratio_years_ago : 
  (A_age - years_ago) * 2 = B_age - years_ago ∧
  (A_age - years_ago) * 3 = C_age - years_ago :=
sorry

/-- The theorem stating that the sum of present ages is 57 -/
theorem sum_of_ages : A_age + B_age + C_age = 57 :=
sorry

/-- The main theorem proving that 'years_ago' is correct -/
theorem correct_years_ago : 
  ∃ (y : ℕ), y = years_ago ∧
  (A_age - y) * 2 = B_age - y ∧
  (A_age - y) * 3 = C_age - y ∧
  A_age + B_age + C_age = 57 ∧
  A_age = 11 :=
sorry

end NUMINAMATH_CALUDE_ages_ratio_years_ago_sum_of_ages_correct_years_ago_l1749_174951


namespace NUMINAMATH_CALUDE_comic_book_problem_l1749_174903

theorem comic_book_problem (initial_books : ℕ) : 
  (initial_books / 3 + 15 = 45) → initial_books = 90 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_problem_l1749_174903


namespace NUMINAMATH_CALUDE_extreme_points_when_a_neg_one_max_value_on_interval_l1749_174968

/-- The function f(x) = x³ + 3ax² -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2

/-- Theorem for extreme points and values when a = -1 -/
theorem extreme_points_when_a_neg_one :
  let f_neg_one := f (-1)
  ∃ (local_max local_min : ℝ),
    (local_max = 0 ∧ f_neg_one local_max = 0) ∧
    (local_min = 2 ∧ f_neg_one local_min = -4) ∧
    ∀ x, f_neg_one x ≤ f_neg_one local_max ∨ f_neg_one x ≥ f_neg_one local_min :=
sorry

/-- Theorem for maximum value on [0,2] -/
theorem max_value_on_interval (a : ℝ) :
  let max_value := if a ≥ 0 then f a 2
                   else if a > -1 then max (f a 0) (f a 2)
                   else f a 0
  ∀ x ∈ Set.Icc 0 2, f a x ≤ max_value :=
sorry

end NUMINAMATH_CALUDE_extreme_points_when_a_neg_one_max_value_on_interval_l1749_174968


namespace NUMINAMATH_CALUDE_parallel_transitivity_l1749_174972

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary properties for a line in 3D space
  -- This is a simplified representation
  mk :: 

-- Define parallelism for lines in 3D space
def parallel (l1 l2 : Line3D) : Prop :=
  -- Define what it means for two lines to be parallel
  sorry

-- State the theorem
theorem parallel_transitivity (a b c : Line3D) :
  parallel a c → parallel b c → parallel a b := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l1749_174972


namespace NUMINAMATH_CALUDE_chess_tournament_score_difference_l1749_174989

-- Define the number of players
def num_players : ℕ := 12

-- Define the scoring system
def win_points : ℚ := 1
def draw_points : ℚ := 1/2
def loss_points : ℚ := 0

-- Define the total number of games
def total_games : ℕ := num_players * (num_players - 1) / 2

-- Define Vasya's score (minimum possible given the conditions)
def vasya_score : ℚ := loss_points + (num_players - 2) * draw_points

-- Define the minimum score for other players to be higher than Vasya
def min_other_score : ℚ := vasya_score + 1/2

-- Define Petya's score (maximum possible)
def petya_score : ℚ := (num_players - 1) * win_points

-- Theorem statement
theorem chess_tournament_score_difference :
  petya_score - vasya_score = 1 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_score_difference_l1749_174989


namespace NUMINAMATH_CALUDE_largest_base_for_12_4th_power_l1749_174916

/-- Given a natural number n and a base b, returns the sum of digits of n in base b -/
def sumOfDigits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Returns true if b is a valid base (greater than 1) -/
def isValidBase (b : ℕ) : Prop := b > 1

theorem largest_base_for_12_4th_power :
  ∀ b : ℕ, isValidBase b →
    (b ≤ 7 ↔ sumOfDigits (12^4) b ≠ 2^5) ∧
    (b > 7 → sumOfDigits (12^4) b = 2^5) :=
sorry

end NUMINAMATH_CALUDE_largest_base_for_12_4th_power_l1749_174916


namespace NUMINAMATH_CALUDE_segments_can_be_commensurable_l1749_174950

/-- Represents a geometric segment -/
structure Segment where
  length : ℝ
  pos : length > 0

/-- Two segments are commensurable if their ratio is rational -/
def commensurable (a b : Segment) : Prop :=
  ∃ (q : ℚ), a.length = q * b.length

/-- Segment m fits into a an integer number of times -/
def fits_integer_times (m a : Segment) : Prop :=
  ∃ (k : ℤ), a.length = k * m.length

/-- No segment m/(10^n) fits into b an integer number of times -/
def no_submultiple_fits (m b : Segment) : Prop :=
  ∀ (n : ℕ), ¬∃ (j : ℤ), b.length = j * (m.length / (10^n : ℝ))

theorem segments_can_be_commensurable
  (a b m : Segment)
  (h1 : fits_integer_times m a)
  (h2 : no_submultiple_fits m b) :
  commensurable a b :=
sorry

end NUMINAMATH_CALUDE_segments_can_be_commensurable_l1749_174950


namespace NUMINAMATH_CALUDE_andrews_age_l1749_174987

/-- Proves that Andrew's current age is 30, given the donation information -/
theorem andrews_age (donation_start_age : ℕ) (annual_donation : ℕ) (total_donation : ℕ) :
  donation_start_age = 11 →
  annual_donation = 7 →
  total_donation = 133 →
  donation_start_age + (total_donation / annual_donation) = 30 :=
by sorry

end NUMINAMATH_CALUDE_andrews_age_l1749_174987


namespace NUMINAMATH_CALUDE_prob_two_red_scheme1_correct_scheme2_more_advantageous_l1749_174913

-- Define the number of red and yellow balls
def red_balls : ℕ := 2
def yellow_balls : ℕ := 3
def total_balls : ℕ := red_balls + yellow_balls

-- Define the reward amounts
def reward (red_count : ℕ) : ℝ :=
  match red_count with
  | 0 => 5
  | 1 => 10
  | 2 => 20
  | _ => 0

-- Define the probability of drawing two red balls in Scheme 1
def prob_two_red_scheme1 : ℚ := 1 / 10

-- Define the average earnings for each scheme
def avg_earnings_scheme1 : ℝ := 8.5
def avg_earnings_scheme2 : ℝ := 9.2

-- Theorem 1: Probability of drawing two red balls in Scheme 1
theorem prob_two_red_scheme1_correct :
  prob_two_red_scheme1 = 1 / 10 := by sorry

-- Theorem 2: Scheme 2 is more advantageous
theorem scheme2_more_advantageous :
  avg_earnings_scheme2 > avg_earnings_scheme1 := by sorry

end NUMINAMATH_CALUDE_prob_two_red_scheme1_correct_scheme2_more_advantageous_l1749_174913


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1749_174925

theorem rectangle_diagonal (a b d : ℝ) : 
  a = 6 → a * b = 48 → d^2 = a^2 + b^2 → d = 10 := by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1749_174925


namespace NUMINAMATH_CALUDE_jordan_weight_change_l1749_174922

def weight_change (initial_weight : ℕ) (loss_first_4_weeks : ℕ) (loss_week_5 : ℕ) 
  (loss_next_7_weeks : ℕ) (gain_week_13 : ℕ) : ℕ :=
  initial_weight - (4 * loss_first_4_weeks + loss_week_5 + 7 * loss_next_7_weeks - gain_week_13)

theorem jordan_weight_change :
  weight_change 250 3 5 2 2 = 221 :=
by sorry

end NUMINAMATH_CALUDE_jordan_weight_change_l1749_174922


namespace NUMINAMATH_CALUDE_factor_expression_l1749_174948

theorem factor_expression (a : ℝ) : 189 * a^2 + 27 * a - 54 = 9 * (7 * a - 3) * (3 * a + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1749_174948


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l1749_174937

theorem triangle_max_perimeter :
  ∀ x : ℕ,
    x > 0 →
    x < 5 →
    x + 5*x > 20 →
    5*x + 20 > x →
    x + 20 > 5*x →
    (∀ y : ℕ,
      y > 0 →
      y < 5 →
      y + 5*y > 20 →
      5*y + 20 > y →
      y + 20 > 5*y →
      x + 5*x + 20 ≥ y + 5*y + 20) →
    x + 5*x + 20 = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l1749_174937


namespace NUMINAMATH_CALUDE_pie_cost_is_six_l1749_174997

/-- The cost of a pie given initial and remaining amounts -/
def pieCost (initialAmount remainingAmount : ℕ) : ℕ :=
  initialAmount - remainingAmount

/-- Theorem: The cost of the pie is $6 -/
theorem pie_cost_is_six :
  pieCost 63 57 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pie_cost_is_six_l1749_174997


namespace NUMINAMATH_CALUDE_cool_double_l1749_174935

def is_cool (n : ℕ) : Prop := ∃ a b : ℕ, n = a^2 + b^2

theorem cool_double {k : ℕ} (h : is_cool k) : is_cool (2 * k) := by
  sorry

end NUMINAMATH_CALUDE_cool_double_l1749_174935


namespace NUMINAMATH_CALUDE_count_ways_1800_l1749_174971

/-- The number of ways to express 1800 as a sum of 4s and 5s (ignoring order) -/
def ways_to_sum_1800 : ℕ :=
  (Finset.range 201).card

/-- Theorem stating that there are 201 ways to express 1800 as a sum of 4s and 5s -/
theorem count_ways_1800 : ways_to_sum_1800 = 201 := by
  sorry

#check count_ways_1800

end NUMINAMATH_CALUDE_count_ways_1800_l1749_174971


namespace NUMINAMATH_CALUDE_convergence_of_derived_series_l1749_174979

theorem convergence_of_derived_series (a : ℕ → ℝ) 
  (h_monotonic : Monotone a) 
  (h_convergent : Summable a) :
  Summable (fun n => n * (a n - a (n + 1))) :=
sorry

end NUMINAMATH_CALUDE_convergence_of_derived_series_l1749_174979


namespace NUMINAMATH_CALUDE_range_of_x_l1749_174900

theorem range_of_x (x : ℝ) : 2 * x + 1 ≤ 0 → x ≤ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1749_174900


namespace NUMINAMATH_CALUDE_max_current_speed_is_26_l1749_174957

/-- The speed of Mumbo running -/
def mumbo_speed : ℝ := 11

/-- The speed of Yumbo walking -/
def yumbo_speed : ℝ := 6

/-- Predicate to check if a given speed is a valid river current speed -/
def is_valid_current_speed (v : ℝ) : Prop :=
  v ≥ 6 ∧ ∃ (n : ℕ), v = n

/-- Predicate to check if Yumbo arrives before Mumbo given distances and current speed -/
def yumbo_arrives_first (x y v : ℝ) : Prop :=
  y / yumbo_speed < x / mumbo_speed + (x + y) / v

/-- The maximum possible river current speed -/
def max_current_speed : ℕ := 26

/-- The main theorem stating that 26 km/h is the maximum possible river current speed -/
theorem max_current_speed_is_26 :
  ∀ (v : ℝ), is_valid_current_speed v →
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x < y ∧ yumbo_arrives_first x y v) →
  v ≤ max_current_speed :=
sorry

end NUMINAMATH_CALUDE_max_current_speed_is_26_l1749_174957


namespace NUMINAMATH_CALUDE_corn_acreage_l1749_174931

theorem corn_acreage (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : beans_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) :
  (total_land * corn_ratio) / (beans_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acreage_l1749_174931


namespace NUMINAMATH_CALUDE_apples_per_box_l1749_174991

theorem apples_per_box 
  (apples_per_crate : ℕ) 
  (crates_delivered : ℕ) 
  (rotten_apples : ℕ) 
  (boxes_used : ℕ) 
  (h1 : apples_per_crate = 42)
  (h2 : crates_delivered = 12)
  (h3 : rotten_apples = 4)
  (h4 : boxes_used = 50)
  : (apples_per_crate * crates_delivered - rotten_apples) / boxes_used = 10 :=
by
  sorry

#check apples_per_box

end NUMINAMATH_CALUDE_apples_per_box_l1749_174991


namespace NUMINAMATH_CALUDE_range_of_a_lower_bound_of_f_l1749_174983

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a - 1| + |x - 2*a|

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : f 1 a < 3 → -2/3 < a ∧ a < 4/3 := by sorry

-- Theorem for the lower bound of f(x)
theorem lower_bound_of_f (a x : ℝ) : a ≥ 1 → f x a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_lower_bound_of_f_l1749_174983


namespace NUMINAMATH_CALUDE_only_B_is_random_l1749_174944

-- Define the type for events
inductive Event
| A  -- A coin thrown from the ground will fall down
| B  -- A shooter hits the target with 10 points in one shot
| C  -- The sun rises from the east
| D  -- A horse runs at a speed of 70 meters per second

-- Define what it means for an event to be random
def is_random (e : Event) : Prop :=
  match e with
  | Event.A => false
  | Event.B => true
  | Event.C => false
  | Event.D => false

-- Theorem stating that only event B is random
theorem only_B_is_random :
  ∀ e : Event, is_random e ↔ e = Event.B :=
by
  sorry


end NUMINAMATH_CALUDE_only_B_is_random_l1749_174944


namespace NUMINAMATH_CALUDE_min_balls_for_three_colors_l1749_174919

/-- Represents the number of balls of a specific color in the box -/
def BallCount := ℕ

/-- Represents the total number of balls in the box -/
def TotalBalls : ℕ := 111

/-- Represents the number of different colors of balls in the box -/
def NumColors : ℕ := 4

/-- Represents the number of balls that guarantees at least four different colors when drawn -/
def GuaranteeFourColors : ℕ := 100

/-- Represents a function that returns the minimum number of balls to draw to ensure at least three different colors -/
def minBallsForThreeColors (total : ℕ) (numColors : ℕ) (guaranteeFour : ℕ) : ℕ := 
  total - guaranteeFour + 1

/-- Theorem stating that the minimum number of balls to draw to ensure at least three different colors is 88 -/
theorem min_balls_for_three_colors : 
  minBallsForThreeColors TotalBalls NumColors GuaranteeFourColors = 88 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_for_three_colors_l1749_174919


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1749_174926

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  /-- The sequence terms -/
  a : ℕ → ℝ
  /-- The number of terms -/
  n : ℕ
  /-- Sum of first 3 terms is 20 -/
  first_three_sum : a 1 + a 2 + a 3 = 20
  /-- Sum of last 3 terms is 130 -/
  last_three_sum : a (n - 2) + a (n - 1) + a n = 130
  /-- Sum of all terms is 200 -/
  total_sum : (Finset.range n).sum a = 200

/-- The number of terms in the arithmetic sequence is 8 -/
theorem arithmetic_sequence_length (seq : ArithmeticSequence) : seq.n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1749_174926


namespace NUMINAMATH_CALUDE_gumball_difference_l1749_174942

theorem gumball_difference (x : ℤ) : 
  (19 * 3 ≤ 16 + 12 + x ∧ 16 + 12 + x ≤ 25 * 3) →
  (∃ (max min : ℤ), 
    (∀ y : ℤ, 19 * 3 ≤ 16 + 12 + y ∧ 16 + 12 + y ≤ 25 * 3 → y ≤ max) ∧
    (∀ y : ℤ, 19 * 3 ≤ 16 + 12 + y ∧ 16 + 12 + y ≤ 25 * 3 → min ≤ y) ∧
    max - min = 18) :=
by sorry

end NUMINAMATH_CALUDE_gumball_difference_l1749_174942


namespace NUMINAMATH_CALUDE_sample_size_is_80_l1749_174953

/-- Represents the ratio of products A, B, and C in production -/
def productionRatio : Fin 3 → ℕ
| 0 => 2  -- Product A
| 1 => 3  -- Product B
| 2 => 5  -- Product C

/-- The number of products of type B selected in the sample -/
def selectedB : ℕ := 24

/-- The total sample size -/
def n : ℕ := 80

/-- Theorem stating that the given conditions lead to a sample size of 80 -/
theorem sample_size_is_80 : 
  (productionRatio 1 : ℚ) / (productionRatio 0 + productionRatio 1 + productionRatio 2) = selectedB / n :=
sorry

end NUMINAMATH_CALUDE_sample_size_is_80_l1749_174953


namespace NUMINAMATH_CALUDE_class_size_calculation_l1749_174990

theorem class_size_calculation (incorrect_mark : ℕ) (correct_mark : ℕ) (average_increase : ℚ) : 
  incorrect_mark = 67 → 
  correct_mark = 45 → 
  average_increase = 1/2 →
  (incorrect_mark - correct_mark : ℚ) / (2 * average_increase) = 44 :=
by sorry

end NUMINAMATH_CALUDE_class_size_calculation_l1749_174990


namespace NUMINAMATH_CALUDE_max_overtakes_l1749_174959

/-- Represents a team in the relay race -/
structure Team :=
  (members : Nat)
  (segments : Nat)

/-- Represents the relay race setup -/
structure RelayRace :=
  (team1 : Team)
  (team2 : Team)
  (simultaneous_start : Bool)
  (instantaneous_exchange : Bool)

/-- Defines what constitutes an overtake in the race -/
def is_valid_overtake (race : RelayRace) (position : Nat) : Prop :=
  position > 0 ∧ position < race.team1.segments ∧ position < race.team2.segments

/-- The main theorem stating the maximum number of overtakes -/
theorem max_overtakes (race : RelayRace) : 
  race.team1.members = 20 →
  race.team2.members = 20 →
  race.team1.segments = 20 →
  race.team2.segments = 20 →
  race.simultaneous_start = true →
  race.instantaneous_exchange = true →
  ∃ (n : Nat), n = 38 ∧ 
    (∀ (m : Nat), (∃ (valid_overtakes : List Nat), 
      (∀ o ∈ valid_overtakes, is_valid_overtake race o) ∧ 
      valid_overtakes.length = m) → m ≤ n) :=
sorry


end NUMINAMATH_CALUDE_max_overtakes_l1749_174959


namespace NUMINAMATH_CALUDE_num_paths_is_126_l1749_174975

/-- The number of paths from A to C passing through B on a grid -/
def num_paths_through_B : ℕ :=
  let a_to_b_right := 5
  let a_to_b_down := 2
  let b_to_c_right := 2
  let b_to_c_down := 2
  let paths_a_to_b := Nat.choose (a_to_b_right + a_to_b_down) a_to_b_right
  let paths_b_to_c := Nat.choose (b_to_c_right + b_to_c_down) b_to_c_right
  paths_a_to_b * paths_b_to_c

/-- Theorem stating the number of paths from A to C passing through B is 126 -/
theorem num_paths_is_126 : num_paths_through_B = 126 := by
  sorry

end NUMINAMATH_CALUDE_num_paths_is_126_l1749_174975


namespace NUMINAMATH_CALUDE_haunted_castle_windows_l1749_174921

theorem haunted_castle_windows (n : ℕ) (h : n = 10) : 
  n * (n - 1) * (n - 2) * (n - 3) = 5040 :=
sorry

end NUMINAMATH_CALUDE_haunted_castle_windows_l1749_174921


namespace NUMINAMATH_CALUDE_kylie_daisies_l1749_174958

theorem kylie_daisies (initial : ℕ) (final : ℕ) (sister_gave : ℕ) : 
  initial = 5 →
  final = 7 →
  (initial + sister_gave) / 2 = final →
  sister_gave = 9 :=
by sorry

end NUMINAMATH_CALUDE_kylie_daisies_l1749_174958


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_minimum_l1749_174995

theorem quadratic_roots_sum_squares_minimum (m : ℝ) :
  let a : ℝ := 6
  let b : ℝ := -8
  let c : ℝ := m
  let discriminant := b^2 - 4*a*c
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  let sum_of_squares := sum_of_roots^2 - 2*product_of_roots
  discriminant > 0 →
  (∀ m' : ℝ, discriminant > 0 → sum_of_squares ≤ ((-b/a)^2 - 2*(m'/a))) →
  m = 8/3 ∧ sum_of_squares = 8/9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_minimum_l1749_174995


namespace NUMINAMATH_CALUDE_set_equality_implies_coefficients_l1749_174955

def A : Set ℝ := {-1, 3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

theorem set_equality_implies_coefficients (a b : ℝ) : 
  A = B a b → a = -2 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_coefficients_l1749_174955


namespace NUMINAMATH_CALUDE_doubling_function_m_range_l1749_174992

/-- A function f is a "doubling function" if there exists an interval [a,b] in its domain
    such that the range of f on [a,b] is [2a,2b] -/
def DoublingFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧ (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (2*a) (2*b)) ∧
    (∀ y ∈ Set.Icc (2*a) (2*b), ∃ x ∈ Set.Icc a b, f x = y)

/-- The main theorem stating that for f(x) = ln(e^x + m) to be a doubling function,
    m must be in the range (-1/4, 0) -/
theorem doubling_function_m_range :
  ∀ m : ℝ, (DoublingFunction (fun x ↦ Real.log (Real.exp x + m))) ↔ -1/4 < m ∧ m < 0 := by
  sorry


end NUMINAMATH_CALUDE_doubling_function_m_range_l1749_174992


namespace NUMINAMATH_CALUDE_profit_increase_l1749_174967

theorem profit_increase (profit_1995 : ℝ) : 
  let profit_1996 := profit_1995 * 1.1
  let profit_1997 := profit_1995 * 1.3200000000000001
  (profit_1997 / profit_1996 - 1) * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_profit_increase_l1749_174967


namespace NUMINAMATH_CALUDE_water_mixture_adjustment_l1749_174905

theorem water_mixture_adjustment (initial_volume : ℝ) (initial_water_percentage : ℝ) 
  (initial_acid_percentage : ℝ) (water_to_add : ℝ) (final_water_percentage : ℝ) 
  (final_acid_percentage : ℝ) : 
  initial_volume = 300 →
  initial_water_percentage = 0.60 →
  initial_acid_percentage = 0.40 →
  water_to_add = 100 →
  final_water_percentage = 0.70 →
  final_acid_percentage = 0.30 →
  (initial_volume * initial_water_percentage + water_to_add) / (initial_volume + water_to_add) = final_water_percentage ∧
  (initial_volume * initial_acid_percentage) / (initial_volume + water_to_add) = final_acid_percentage :=
by sorry

end NUMINAMATH_CALUDE_water_mixture_adjustment_l1749_174905


namespace NUMINAMATH_CALUDE_find_number_l1749_174960

theorem find_number (x : ℝ) : (0.05 * x = 0.2 * 650 + 190) → x = 6400 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1749_174960


namespace NUMINAMATH_CALUDE_largest_solution_l1749_174994

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ := x - floor x

/-- The equation from the problem -/
def equation (x : ℝ) : Prop := floor x = 6 + 50 * frac x

/-- The theorem stating the largest solution -/
theorem largest_solution :
  ∃ (x : ℝ), equation x ∧ ∀ (y : ℝ), equation y → y ≤ x :=
sorry

end NUMINAMATH_CALUDE_largest_solution_l1749_174994


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l1749_174977

theorem square_circle_area_ratio (r : ℝ) (h : r > 0) :
  let s := 2 * r
  (s^2) / (π * r^2) = 4 / π := by
sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l1749_174977


namespace NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l1749_174998

/-- The ratio of the combined areas of two semicircles with radius r/2 inscribed in a circle with radius r to the area of the circle is 1/4. -/
theorem semicircles_to_circle_area_ratio (r : ℝ) (h : r > 0) : 
  (2 * (π * (r/2)^2 / 2)) / (π * r^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l1749_174998


namespace NUMINAMATH_CALUDE_shellys_total_money_l1749_174956

/-- Calculates the total amount of money Shelly has given her bill and coin counts. -/
def shellys_money (ten_dollar_bills : ℕ) : ℕ :=
  let five_dollar_bills := ten_dollar_bills - 12
  let twenty_dollar_bills := ten_dollar_bills / 2
  let one_dollar_coins := five_dollar_bills * 2
  10 * ten_dollar_bills + 5 * five_dollar_bills + 20 * twenty_dollar_bills + one_dollar_coins

/-- Proves that Shelly has $726 given the conditions in the problem. -/
theorem shellys_total_money : shellys_money 30 = 726 := by
  sorry

end NUMINAMATH_CALUDE_shellys_total_money_l1749_174956


namespace NUMINAMATH_CALUDE_basketball_probability_l1749_174933

theorem basketball_probability (jack_prob jill_prob sandy_prob : ℚ)
  (h1 : jack_prob = 1/6)
  (h2 : jill_prob = 1/7)
  (h3 : sandy_prob = 1/8) :
  (1 - jack_prob) * jill_prob * sandy_prob = 5/336 := by
sorry

end NUMINAMATH_CALUDE_basketball_probability_l1749_174933


namespace NUMINAMATH_CALUDE_ceiling_product_equation_l1749_174964

theorem ceiling_product_equation :
  ∃ x : ℝ, ⌈x⌉ * x = 210 ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_l1749_174964


namespace NUMINAMATH_CALUDE_power_equation_solution_l1749_174928

theorem power_equation_solution (x : ℝ) : (1 / 8 : ℝ) * 2^36 = 4^x → x = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1749_174928


namespace NUMINAMATH_CALUDE_road_repair_hours_l1749_174927

theorem road_repair_hours (people1 people2 days1 days2 hours2 : ℕ) 
  (h1 : people1 = 39)
  (h2 : days1 = 12)
  (h3 : people2 = 30)
  (h4 : days2 = 26)
  (h5 : hours2 = 3)
  (h6 : people1 * days1 * (people1 * days1 * hours2 / (people2 * days2)) = people2 * days2 * hours2) :
  people1 * days1 * hours2 / (people2 * days2) = 5 := by
sorry

end NUMINAMATH_CALUDE_road_repair_hours_l1749_174927


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1749_174911

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {x | x^2 + x = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1749_174911


namespace NUMINAMATH_CALUDE_boys_in_class_l1749_174970

theorem boys_in_class (initial_girls : ℕ) (initial_boys : ℕ) (final_girls : ℕ) :
  (initial_girls : ℚ) / initial_boys = 5 / 6 →
  (final_girls : ℚ) / initial_boys = 2 / 3 →
  initial_girls - final_girls = 20 →
  initial_boys = 120 := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l1749_174970


namespace NUMINAMATH_CALUDE_square_plus_one_nonnegative_l1749_174939

theorem square_plus_one_nonnegative (m : ℝ) : m^2 + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_nonnegative_l1749_174939


namespace NUMINAMATH_CALUDE_two_year_interest_calculation_l1749_174936

def compound_interest (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  initial_amount * (1 + rate1) * (1 + rate2)

theorem two_year_interest_calculation :
  let initial_amount : ℝ := 7500
  let rate1 : ℝ := 0.20
  let rate2 : ℝ := 0.25
  compound_interest initial_amount rate1 rate2 = 11250 := by
  sorry

end NUMINAMATH_CALUDE_two_year_interest_calculation_l1749_174936


namespace NUMINAMATH_CALUDE_equation_solution_l1749_174996

theorem equation_solution : ∃ x : ℚ, (5/100 * x + 12/100 * (30 + x) = 144/10) ∧ x = 108/17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1749_174996


namespace NUMINAMATH_CALUDE_simplify_fraction_l1749_174908

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1749_174908


namespace NUMINAMATH_CALUDE_thomas_work_hours_l1749_174985

theorem thomas_work_hours 
  (total_hours : ℕ)
  (rebecca_hours : ℕ)
  (h1 : total_hours = 157)
  (h2 : rebecca_hours = 56) :
  ∃ (thomas_hours : ℕ),
    thomas_hours = 37 ∧
    ∃ (toby_hours : ℕ),
      toby_hours = 2 * thomas_hours - 10 ∧
      rebecca_hours = toby_hours - 8 ∧
      total_hours = thomas_hours + toby_hours + rebecca_hours :=
by sorry

end NUMINAMATH_CALUDE_thomas_work_hours_l1749_174985


namespace NUMINAMATH_CALUDE_rhea_count_l1749_174912

theorem rhea_count (num_wombats : ℕ) (wombat_claws : ℕ) (rhea_claws : ℕ) (total_claws : ℕ) : 
  num_wombats = 9 →
  wombat_claws = 4 →
  rhea_claws = 1 →
  total_claws = 39 →
  total_claws = num_wombats * wombat_claws + (total_claws - num_wombats * wombat_claws) →
  (total_claws - num_wombats * wombat_claws) / rhea_claws = 3 := by
sorry

end NUMINAMATH_CALUDE_rhea_count_l1749_174912


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1749_174914

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 7*p^2 + 11*p = 14 →
  q^3 - 7*q^2 + 11*q = 14 →
  r^3 - 7*r^2 + 11*r = 14 →
  p + q + r = 7 →
  p*q + q*r + r*p = 11 →
  p*q*r = 14 →
  p*q/r + q*r/p + r*p/q = -75/14 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1749_174914


namespace NUMINAMATH_CALUDE_min_distance_complex_main_theorem_l1749_174973

-- Define inductive reasoning
def inductiveReasoning : String := "reasoning from specific to general"

-- Define deductive reasoning
def deductiveReasoning : String := "reasoning from general to specific"

-- Theorem for the complex number part
theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ (w : ℂ), Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
sorry

-- Main theorem combining all parts
theorem main_theorem :
  inductiveReasoning = "reasoning from specific to general" ∧
  deductiveReasoning = "reasoning from general to specific" ∧
  ∀ (z : ℂ), Complex.abs (z + 2 - 2*I) = 1 →
    ∃ (min_val : ℝ), min_val = 3 ∧ ∀ (w : ℂ), Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_main_theorem_l1749_174973


namespace NUMINAMATH_CALUDE_sequence_seventh_term_l1749_174952

theorem sequence_seventh_term : 
  ∀ (a : ℕ → ℕ), 
    a 1 = 1 ∧ 
    (∀ n : ℕ, a (n + 1) = 2 * a n + 2) → 
    a 7 = 190 := by
  sorry

end NUMINAMATH_CALUDE_sequence_seventh_term_l1749_174952


namespace NUMINAMATH_CALUDE_pet_shop_total_cost_l1749_174969

/-- The cost of purchasing all pets in a pet shop given specific conditions. -/
theorem pet_shop_total_cost :
  let num_puppies : ℕ := 2
  let num_kittens : ℕ := 2
  let num_parakeets : ℕ := 3
  let parakeet_cost : ℕ := 10
  let puppy_cost : ℕ := 3 * parakeet_cost
  let kitten_cost : ℕ := 2 * parakeet_cost
  num_puppies * puppy_cost + num_kittens * kitten_cost + num_parakeets * parakeet_cost = 130 := by
sorry

end NUMINAMATH_CALUDE_pet_shop_total_cost_l1749_174969


namespace NUMINAMATH_CALUDE_sally_quarters_l1749_174907

/-- The number of quarters Sally has after her purchases -/
def remaining_quarters (initial : ℕ) (purchase1 : ℕ) (purchase2 : ℕ) : ℕ :=
  initial - purchase1 - purchase2

/-- Theorem stating that Sally has 150 quarters left after her purchases -/
theorem sally_quarters : remaining_quarters 760 418 192 = 150 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_l1749_174907


namespace NUMINAMATH_CALUDE_total_money_value_l1749_174917

def us_100_bills : ℕ := 2
def us_50_bills : ℕ := 5
def us_10_bills : ℕ := 5
def canadian_20_bills : ℕ := 15
def euro_10_notes : ℕ := 20
def us_quarters : ℕ := 50
def us_dimes : ℕ := 120

def cad_to_usd_rate : ℚ := 0.80
def eur_to_usd_rate : ℚ := 1.10

def total_us_currency : ℚ := 
  us_100_bills * 100 + 
  us_50_bills * 50 + 
  us_10_bills * 10 + 
  us_quarters * 0.25 + 
  us_dimes * 0.10

def total_cad_in_usd : ℚ := canadian_20_bills * 20 * cad_to_usd_rate
def total_eur_in_usd : ℚ := euro_10_notes * 10 * eur_to_usd_rate

theorem total_money_value : 
  total_us_currency + total_cad_in_usd + total_eur_in_usd = 984.50 := by
  sorry

end NUMINAMATH_CALUDE_total_money_value_l1749_174917


namespace NUMINAMATH_CALUDE_children_left_l1749_174932

theorem children_left (total_guests : ℕ) (men : ℕ) (stayed : ℕ) :
  total_guests = 50 ∧ 
  men = 15 ∧ 
  stayed = 43 →
  (total_guests / 2 : ℕ) + men + ((total_guests - (total_guests / 2 + men)) - 
    (total_guests - stayed - men / 5)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_children_left_l1749_174932


namespace NUMINAMATH_CALUDE_nested_fraction_square_l1749_174910

theorem nested_fraction_square (x : ℚ) (h : x = 1/3) :
  let f := (x + 2) / (x - 2)
  ((f + 2) / (f - 2))^2 = 961/1369 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_square_l1749_174910


namespace NUMINAMATH_CALUDE_pastry_combinations_l1749_174945

/-- The number of ways to select pastries -/
def select_pastries (total : ℕ) (types : ℕ) : ℕ :=
  if types > total then 0
  else
    let remaining := total - types
    -- Ways to distribute remaining pastries among types
    (types^remaining + types * (types - 1) * remaining + Nat.choose types remaining) / Nat.factorial remaining

/-- Theorem: Selecting 8 pastries from 5 types, with at least one of each type, results in 25 combinations -/
theorem pastry_combinations : select_pastries 8 5 = 25 := by
  sorry


end NUMINAMATH_CALUDE_pastry_combinations_l1749_174945


namespace NUMINAMATH_CALUDE_real_part_divisible_by_p_l1749_174902

/-- A Gaussian integer is a complex number with integer real and imaginary parts. -/
structure GaussianInteger where
  re : ℤ
  im : ℤ

/-- The real part of a complex number z^p - z is divisible by p for any Gaussian integer z and odd prime p. -/
theorem real_part_divisible_by_p (z : GaussianInteger) (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (k : ℤ), (z.re^p - z.re : ℤ) = p * k := by
  sorry

end NUMINAMATH_CALUDE_real_part_divisible_by_p_l1749_174902


namespace NUMINAMATH_CALUDE_actual_average_height_l1749_174947

/-- The actual average height of boys in a class with measurement errors -/
theorem actual_average_height (n : ℕ) (initial_avg : ℝ) 
  (error1 : ℝ) (error2 : ℝ) : 
  n = 40 → 
  initial_avg = 184 → 
  error1 = 166 - 106 → 
  error2 = 190 - 180 → 
  (n * initial_avg - (error1 + error2)) / n = 182.25 := by
  sorry

end NUMINAMATH_CALUDE_actual_average_height_l1749_174947


namespace NUMINAMATH_CALUDE_number_division_problem_l1749_174946

theorem number_division_problem (x : ℝ) : x / 0.3 = 7.3500000000000005 → x = 2.205 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1749_174946


namespace NUMINAMATH_CALUDE_triangle_angle_b_l1749_174923

theorem triangle_angle_b (a b c : ℝ) (A B C : ℝ) :
  c = 2 * b * Real.cos B →
  C = 2 * Real.pi / 3 →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  B = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_b_l1749_174923


namespace NUMINAMATH_CALUDE_seventh_term_is_2187_l1749_174941

/-- A geometric sequence of positive integers -/
structure GeometricSequence where
  a : ℕ → ℕ  -- The sequence
  r : ℕ      -- The common ratio
  first_term : a 1 = 3
  ratio_def : ∀ n : ℕ, a (n + 1) = a n * r

theorem seventh_term_is_2187 (seq : GeometricSequence) (h : seq.a 6 = 972) :
  seq.a 7 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_2187_l1749_174941


namespace NUMINAMATH_CALUDE_janes_leave_days_l1749_174924

theorem janes_leave_days (jane_rate ashley_rate total_days extra_days : ℝ) 
  (h1 : jane_rate = 1 / 10)
  (h2 : ashley_rate = 1 / 40)
  (h3 : total_days = 15.2)
  (h4 : extra_days = 4) : 
  ∃ leave_days : ℝ, 
    (jane_rate + ashley_rate) * (total_days - leave_days) + 
    ashley_rate * leave_days + 
    jane_rate * extra_days = 1 ∧ 
    leave_days = 13 := by
sorry

end NUMINAMATH_CALUDE_janes_leave_days_l1749_174924


namespace NUMINAMATH_CALUDE_chocolate_pieces_per_box_l1749_174988

theorem chocolate_pieces_per_box 
  (total_boxes : ℕ) 
  (given_boxes : ℕ) 
  (remaining_pieces : ℕ) 
  (h1 : total_boxes = 14)
  (h2 : given_boxes = 8)
  (h3 : remaining_pieces = 18)
  (h4 : total_boxes > given_boxes) :
  (remaining_pieces / (total_boxes - given_boxes) : ℕ) = 3 := by
sorry

end NUMINAMATH_CALUDE_chocolate_pieces_per_box_l1749_174988


namespace NUMINAMATH_CALUDE_cos_two_thirds_pi_minus_two_alpha_l1749_174949

theorem cos_two_thirds_pi_minus_two_alpha (α : ℝ) 
  (h : Real.sin (α + π / 6) = Real.sqrt 6 / 3) : 
  Real.cos (2 * π / 3 - 2 * α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_thirds_pi_minus_two_alpha_l1749_174949


namespace NUMINAMATH_CALUDE_adam_figurines_l1749_174943

/-- The number of figurines that can be made from one block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be made from one block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be made from one block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The number of blocks of basswood Adam owns -/
def basswood_blocks : ℕ := 15

/-- The number of blocks of butternut wood Adam owns -/
def butternut_blocks : ℕ := 20

/-- The number of blocks of Aspen wood Adam owns -/
def aspen_blocks : ℕ := 20

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 
  basswood_blocks * basswood_figurines + 
  butternut_blocks * butternut_figurines + 
  aspen_blocks * aspen_figurines

theorem adam_figurines : total_figurines = 245 := by
  sorry

end NUMINAMATH_CALUDE_adam_figurines_l1749_174943


namespace NUMINAMATH_CALUDE_digit_sum_subtraction_l1749_174965

theorem digit_sum_subtraction (n : ℕ) : 
  2010 ≤ n ∧ n ≤ 2019 → n - (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) = 2007 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_subtraction_l1749_174965


namespace NUMINAMATH_CALUDE_shaded_area_circles_l1749_174961

/-- Given a larger circle of radius 8 and two smaller circles touching the larger circle
    and each other at the center of the larger circle, the area of the shaded region
    (the area of the larger circle minus the areas of the two smaller circles) is 32π. -/
theorem shaded_area_circles (r : ℝ) (h : r = 8) : 
  r^2 * π - 2 * (r/2)^2 * π = 32 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_circles_l1749_174961


namespace NUMINAMATH_CALUDE_division_problem_l1749_174962

theorem division_problem (a b q : ℕ) 
  (h1 : a - b = 1345)
  (h2 : a = 1596)
  (h3 : a = b * q + 15) :
  q = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1749_174962


namespace NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_inequality_l1749_174901

theorem arithmetic_geometric_harmonic_inequality {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a + b) / 2 > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > 2 * a * b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_inequality_l1749_174901


namespace NUMINAMATH_CALUDE_cube_difference_l1749_174966

theorem cube_difference (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 - n^2 = 43) :
  m^3 - n^3 = 1387 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l1749_174966


namespace NUMINAMATH_CALUDE_curve_C_properties_l1749_174993

/-- Definition of the curve C -/
def curve_C (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (25 - k) + p.2^2 / (k - 9) = 1}

/-- Definition of an ellipse -/
def is_ellipse (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ p ∈ S, p.1^2 / a^2 + p.2^2 / b^2 = 1

/-- Definition of a hyperbola with foci on the x-axis -/
def is_hyperbola_x_axis (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ p ∈ S, p.1^2 / a^2 - p.2^2 / b^2 = 1

theorem curve_C_properties (k : ℝ) :
  (9 < k ∧ k < 25 → is_ellipse (curve_C k)) ∧
  (is_hyperbola_x_axis (curve_C k) → k < 9) :=
sorry

end NUMINAMATH_CALUDE_curve_C_properties_l1749_174993


namespace NUMINAMATH_CALUDE_max_value_expression_l1749_174954

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (⨆ x : ℝ, 3 * (a - x) * (x + Real.sqrt (x^2 + b^2 + c))) = 3/2 * (b^2 + c) + 3 * a^2 := by
sorry

end NUMINAMATH_CALUDE_max_value_expression_l1749_174954


namespace NUMINAMATH_CALUDE_a_in_range_l1749_174982

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem a_in_range (a : ℝ) (h : a ∈ A) : -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_a_in_range_l1749_174982


namespace NUMINAMATH_CALUDE_problem_1_l1749_174963

theorem problem_1 : (-3) + (-9) - 10 - (-18) = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1749_174963


namespace NUMINAMATH_CALUDE_erased_number_proof_l1749_174906

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  n > 2 →
  (↑n * (↑n + 1) / 2 - 3) - x = (454 / 9 : ℚ) * (↑n - 1) →
  x = 107 :=
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l1749_174906


namespace NUMINAMATH_CALUDE_carmela_difference_l1749_174938

def cecil_money : ℕ := 600
def catherine_money : ℕ := 2 * cecil_money - 250
def total_money : ℕ := 2800

theorem carmela_difference : ℕ := by
  have h1 : cecil_money + catherine_money + (2 * cecil_money + (total_money - (cecil_money + catherine_money))) = total_money := by sorry
  have h2 : total_money - (cecil_money + catherine_money) = 50 := by sorry
  exact 50

#check carmela_difference

end NUMINAMATH_CALUDE_carmela_difference_l1749_174938


namespace NUMINAMATH_CALUDE_right_triangle_area_perimeter_l1749_174999

theorem right_triangle_area_perimeter 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 13) 
  (h_leg : a = 5) : 
  (1/2 * a * b = 30) ∧ (a + b + c = 30) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_perimeter_l1749_174999


namespace NUMINAMATH_CALUDE_central_number_is_ten_l1749_174974

/-- A triangular grid with 10 integers -/
structure TriangularGrid :=
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 x : ℤ)

/-- The sum of all ten numbers is 43 -/
def total_sum (g : TriangularGrid) : Prop :=
  g.a1 + g.a2 + g.a3 + g.b1 + g.b2 + g.b3 + g.c1 + g.c2 + g.c3 + g.x = 43

/-- The sum of any three numbers such that any two of them are close is 11 -/
def close_sum (g : TriangularGrid) : Prop :=
  g.a1 + g.a2 + g.a3 = 11 ∧
  g.b1 + g.b2 + g.b3 = 11 ∧
  g.c1 + g.c2 + g.c3 = 11

/-- Theorem: The central number is 10 -/
theorem central_number_is_ten (g : TriangularGrid) 
  (h1 : total_sum g) (h2 : close_sum g) : g.x = 10 := by
  sorry

end NUMINAMATH_CALUDE_central_number_is_ten_l1749_174974


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l1749_174920

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 6) ((Nat.factorial 9) / (Nat.factorial 4)) = 720 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l1749_174920


namespace NUMINAMATH_CALUDE_lcm_of_12_25_45_60_l1749_174929

theorem lcm_of_12_25_45_60 : Nat.lcm 12 (Nat.lcm 25 (Nat.lcm 45 60)) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_25_45_60_l1749_174929


namespace NUMINAMATH_CALUDE_trail_mix_pouches_per_pack_l1749_174976

theorem trail_mix_pouches_per_pack 
  (team_members : ℕ) 
  (coaches : ℕ) 
  (helpers : ℕ) 
  (total_packs : ℕ) 
  (h1 : team_members = 13)
  (h2 : coaches = 3)
  (h3 : helpers = 2)
  (h4 : total_packs = 3)
  : (team_members + coaches + helpers) / total_packs = 6 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_pouches_per_pack_l1749_174976


namespace NUMINAMATH_CALUDE_max_n_for_consecutive_product_l1749_174918

theorem max_n_for_consecutive_product : ∃ (n_max : ℕ), ∀ (n : ℕ), 
  (∃ (k : ℕ), 9*n^2 + 5*n + 26 = k * (k+1)) → n ≤ n_max :=
sorry

end NUMINAMATH_CALUDE_max_n_for_consecutive_product_l1749_174918


namespace NUMINAMATH_CALUDE_high_precision_census_suitability_l1749_174984

/-- Represents different types of surveys --/
inductive SurveyType
  | DestructiveTesting
  | WideScopePopulation
  | HighPrecisionRequired
  | LargeAudienceSampling

/-- Represents different survey methods --/
inductive SurveyMethod
  | Census
  | Sampling

/-- Defines the characteristics of a survey --/
structure Survey where
  type : SurveyType
  method : SurveyMethod

/-- Defines the suitability of a survey method for a given survey type --/
def is_suitable (s : Survey) : Prop :=
  match s.type, s.method with
  | SurveyType.HighPrecisionRequired, SurveyMethod.Census => true
  | SurveyType.DestructiveTesting, SurveyMethod.Sampling => true
  | SurveyType.WideScopePopulation, SurveyMethod.Sampling => true
  | SurveyType.LargeAudienceSampling, SurveyMethod.Sampling => true
  | _, _ => false

/-- Theorem: A survey requiring high precision is most suitable for a census method --/
theorem high_precision_census_suitability :
  ∀ (s : Survey), s.type = SurveyType.HighPrecisionRequired → 
  is_suitable { type := s.type, method := SurveyMethod.Census } = true :=
by
  sorry


end NUMINAMATH_CALUDE_high_precision_census_suitability_l1749_174984


namespace NUMINAMATH_CALUDE_complex_addition_point_l1749_174978

/-- A complex number corresponding to a point in the complex plane -/
def complex_point (x y : ℝ) : ℂ := x + y * Complex.I

/-- The theorem stating that if z corresponds to (2,5), then 1+z corresponds to (3,5) -/
theorem complex_addition_point (z : ℂ) (h : z = complex_point 2 5) :
  1 + z = complex_point 3 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_addition_point_l1749_174978


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1749_174981

theorem polynomial_factorization (x y : ℝ) :
  -2 * x^2 * y + 8 * x * y - 6 * y = -2 * y * (x - 1) * (x - 3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1749_174981


namespace NUMINAMATH_CALUDE_continued_fraction_evaluation_l1749_174940

theorem continued_fraction_evaluation :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_evaluation_l1749_174940
