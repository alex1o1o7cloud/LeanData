import Mathlib

namespace apples_per_box_is_10_l194_19478

/-- The number of apples in each box. -/
def apples_per_box : ℕ := 10

/-- The number of boxes Merry had on Saturday. -/
def saturday_boxes : ℕ := 50

/-- The number of boxes Merry had on Sunday. -/
def sunday_boxes : ℕ := 25

/-- The total number of apples Merry sold. -/
def sold_apples : ℕ := 720

/-- The number of boxes Merry has left. -/
def remaining_boxes : ℕ := 3

/-- Theorem stating that the number of apples in each box is 10. -/
theorem apples_per_box_is_10 :
  apples_per_box * (saturday_boxes + sunday_boxes) - sold_apples = apples_per_box * remaining_boxes :=
by sorry

end apples_per_box_is_10_l194_19478


namespace solution_set_equality_l194_19447

open Set

/-- The solution set of the inequality |x-5|+|x+3|≥10 -/
def SolutionSet : Set ℝ := {x : ℝ | |x - 5| + |x + 3| ≥ 10}

/-- The expected result set (-∞，-4]∪[6，+∞) -/
def ExpectedSet : Set ℝ := Iic (-4) ∪ Ici 6

theorem solution_set_equality : SolutionSet = ExpectedSet := by
  sorry

end solution_set_equality_l194_19447


namespace number_set_properties_l194_19475

/-- A set of natural numbers excluding 1 -/
def NumberSet : Set ℕ :=
  {n : ℕ | n > 1}

/-- Predicate for a number being prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Predicate for a number being composite -/
def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

theorem number_set_properties (S : Set ℕ) (h : S = NumberSet) :
  (¬∀ n ∈ S, isComposite n) →
  (∃ n ∈ S, isPrime n) ∧
  (∀ n ∈ S, ¬isComposite n) ∧
  (∀ n ∈ S, isPrime n) ∧
  (∃ n ∈ S, isComposite n ∧ ∃ m ∈ S, isPrime m) ∧
  (∃ n ∈ S, isPrime n ∧ ∃ m ∈ S, isComposite m) :=
by sorry

end number_set_properties_l194_19475


namespace triangle_operation_result_l194_19491

-- Define the triangle operation
def triangle (P Q : ℚ) : ℚ := (P + Q) / 3

-- State the theorem
theorem triangle_operation_result :
  triangle 3 (triangle 6 9) = 8 / 3 := by sorry

end triangle_operation_result_l194_19491


namespace rotation_of_point_l194_19480

def rotate90ClockwiseAboutOrigin (x y : ℝ) : ℝ × ℝ := (y, -x)

theorem rotation_of_point :
  let D : ℝ × ℝ := (-3, 2)
  rotate90ClockwiseAboutOrigin D.1 D.2 = (2, 3) := by
  sorry

end rotation_of_point_l194_19480


namespace parrot_phrases_l194_19429

def phrases_learned (days : ℕ) (phrases_per_week : ℕ) (initial_phrases : ℕ) : ℕ :=
  initial_phrases + (days / 7) * phrases_per_week

theorem parrot_phrases :
  phrases_learned 49 2 3 = 17 := by
  sorry

end parrot_phrases_l194_19429


namespace angle_between_vectors_l194_19451

/-- The angle between two planar vectors satisfying given conditions -/
theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2 = 7)
  (h2 : Real.sqrt (a.1^2 + a.2^2) = Real.sqrt 3)
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 2) :
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 6 := by
  sorry

end angle_between_vectors_l194_19451


namespace brendan_remaining_money_l194_19442

/-- Brendan's remaining money calculation -/
theorem brendan_remaining_money 
  (june_earnings : ℕ) 
  (car_cost : ℕ) 
  (h1 : june_earnings = 5000)
  (h2 : car_cost = 1500) :
  (june_earnings / 2) - car_cost = 1000 :=
by sorry

end brendan_remaining_money_l194_19442


namespace lineup_combinations_l194_19468

/-- The number of ways to choose a starting lineup -/
def choose_lineup (total_players : ℕ) (offensive_linemen : ℕ) (kickers : ℕ) : ℕ :=
  offensive_linemen * kickers * (total_players - 2) * (total_players - 3) * (total_players - 4)

/-- Theorem stating the number of ways to choose the lineup -/
theorem lineup_combinations :
  choose_lineup 12 4 2 = 5760 := by
  sorry

end lineup_combinations_l194_19468


namespace platform_length_calculation_l194_19400

/-- Calculates the length of a platform given train specifications and crossing time -/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmph = 72 →
  crossing_time = 25 →
  (train_speed_kmph * 1000 / 3600 * crossing_time) - train_length = 380 :=
by sorry

end platform_length_calculation_l194_19400


namespace ratio_problem_l194_19497

theorem ratio_problem (x y : ℕ) (h1 : x + y = 420) (h2 : x = 180) :
  ∃ (a b : ℕ), a = 3 ∧ b = 4 ∧ x * b = y * a :=
sorry

end ratio_problem_l194_19497


namespace virginia_started_with_96_eggs_l194_19436

/-- The number of eggs Virginia started with -/
def initial_eggs : ℕ := sorry

/-- The number of eggs Amy took away -/
def eggs_taken : ℕ := 3

/-- The number of eggs Virginia ended up with -/
def final_eggs : ℕ := 93

/-- Theorem stating that Virginia started with 96 eggs -/
theorem virginia_started_with_96_eggs : initial_eggs = 96 :=
by sorry

end virginia_started_with_96_eggs_l194_19436


namespace polynomial_identity_l194_19466

theorem polynomial_identity (a a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x, (1 - 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) : 
  (a + a₂ + a₄)^2 - (a₁ + a₃)^2 = 81 := by
  sorry

end polynomial_identity_l194_19466


namespace r_fourth_plus_reciprocal_l194_19473

theorem r_fourth_plus_reciprocal (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end r_fourth_plus_reciprocal_l194_19473


namespace arithmetic_sequence_sum_l194_19431

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 1 + a 9 = 180) := by
  sorry

end arithmetic_sequence_sum_l194_19431


namespace impossible_time_reduction_l194_19479

/-- Proves that it's impossible to reduce the time taken to travel 1 kilometer by 1 minute when starting from a speed of 60 km/h. -/
theorem impossible_time_reduction (initial_speed : ℝ) (distance : ℝ) (time_reduction : ℝ) : 
  initial_speed = 60 → distance = 1 → time_reduction = 1 → 
  ¬ ∃ (new_speed : ℝ), new_speed > 0 ∧ distance / new_speed = distance / initial_speed - time_reduction :=
by sorry

end impossible_time_reduction_l194_19479


namespace batsman_average_increase_l194_19450

def average_increase (total_innings : ℕ) (final_average : ℚ) (last_score : ℕ) : ℚ :=
  final_average - (total_innings * final_average - last_score) / (total_innings - 1)

theorem batsman_average_increase :
  average_increase 17 39 87 = 3 := by sorry

end batsman_average_increase_l194_19450


namespace bob_weight_l194_19419

theorem bob_weight (j b : ℝ) 
  (h1 : j + b = 210)
  (h2 : b - j = b / 3)
  : b = 126 := by
  sorry

end bob_weight_l194_19419


namespace quadrilateral_area_l194_19469

/-- The area of a quadrilateral with given diagonal and offsets -/
theorem quadrilateral_area (diagonal : ℝ) (offset1 offset2 : ℝ) :
  diagonal = 10 → offset1 = 7 → offset2 = 3 →
  (diagonal * offset1 / 2) + (diagonal * offset2 / 2) = 50 := by
  sorry

end quadrilateral_area_l194_19469


namespace simple_interest_rate_percent_l194_19422

/-- Given a simple interest scenario, prove that the rate percent is 10% -/
theorem simple_interest_rate_percent (P A T : ℝ) (h1 : P = 750) (h2 : A = 1125) (h3 : T = 5) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = 10 := by sorry

end simple_interest_rate_percent_l194_19422


namespace negation_of_universal_proposition_l194_19482

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 5 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 5 > 0) :=
by sorry

end negation_of_universal_proposition_l194_19482


namespace manhattan_to_bronx_travel_time_l194_19445

/-- The total travel time from Manhattan to the Bronx -/
def total_travel_time (subway_time train_time bike_time : ℕ) : ℕ :=
  subway_time + train_time + bike_time

/-- Theorem stating that the total travel time is 38 hours -/
theorem manhattan_to_bronx_travel_time :
  ∃ (subway_time train_time bike_time : ℕ),
    subway_time = 10 ∧
    train_time = 2 * subway_time ∧
    bike_time = 8 ∧
    total_travel_time subway_time train_time bike_time = 38 :=
by
  sorry

end manhattan_to_bronx_travel_time_l194_19445


namespace intersection_dot_product_l194_19443

/-- Given a line and a parabola that intersect at points A and B, and a point M,
    prove that if the dot product of MA and MB is zero, then the y-coordinate of M is √2/2. -/
theorem intersection_dot_product (A B M : ℝ × ℝ) : 
  (∃ x y, y = 2 * Real.sqrt 2 * (x - 1) ∧ y^2 = 4 * x ∧ A = (x, y)) →  -- Line and parabola intersection for A
  (∃ x y, y = 2 * Real.sqrt 2 * (x - 1) ∧ y^2 = 4 * x ∧ B = (x, y)) →  -- Line and parabola intersection for B
  M.1 = -1 →  -- x-coordinate of M is -1
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 0 →  -- Dot product of MA and MB is zero
  M.2 = Real.sqrt 2 / 2 := by  -- y-coordinate of M is √2/2
sorry

end intersection_dot_product_l194_19443


namespace marbles_given_to_brother_l194_19457

theorem marbles_given_to_brother 
  (total_marbles : ℕ) 
  (mario_ratio : ℕ) 
  (manny_ratio : ℕ) 
  (manny_current : ℕ) 
  (h1 : total_marbles = 36)
  (h2 : mario_ratio = 4)
  (h3 : manny_ratio = 5)
  (h4 : manny_current = 18) :
  (manny_ratio * total_marbles) / (mario_ratio + manny_ratio) - manny_current = 2 :=
sorry

end marbles_given_to_brother_l194_19457


namespace henry_walking_distance_l194_19485

/-- Given a constant walking rate and duration, calculate the distance walked. -/
def distance_walked (rate : ℝ) (time : ℝ) : ℝ :=
  rate * time

/-- Theorem: Henry walks 8 miles in 2 hours at a rate of 4 miles per hour. -/
theorem henry_walking_distance :
  let rate : ℝ := 4  -- miles per hour
  let time : ℝ := 2  -- hours
  distance_walked rate time = 8 := by
  sorry

end henry_walking_distance_l194_19485


namespace reading_plan_theorem_l194_19458

/-- Represents a book with a given number of pages --/
structure Book where
  pages : ℕ

/-- Represents Mrs. Hilt's reading plan --/
structure ReadingPlan where
  book1 : Book
  book2 : Book
  book3 : Book
  firstTwoDaysBook1Percent : ℝ
  firstTwoDaysBook2Percent : ℝ
  day3And4Book1Fraction : ℝ
  day3And4Book2Fraction : ℝ
  day3And4Book3Percent : ℝ
  readingRate : ℕ  -- pages per hour

def calculateRemainingPages (plan : ReadingPlan) : ℕ :=
  sorry

def calculateAverageSpeedFirstFourDays (plan : ReadingPlan) : ℝ :=
  sorry

def calculateTotalReadingHours (plan : ReadingPlan) : ℕ :=
  sorry

theorem reading_plan_theorem (plan : ReadingPlan) 
  (h1 : plan.book1.pages = 457)
  (h2 : plan.book2.pages = 336)
  (h3 : plan.book3.pages = 520)
  (h4 : plan.firstTwoDaysBook1Percent = 0.35)
  (h5 : plan.firstTwoDaysBook2Percent = 0.25)
  (h6 : plan.day3And4Book1Fraction = 1/3)
  (h7 : plan.day3And4Book2Fraction = 1/2)
  (h8 : plan.day3And4Book3Percent = 0.10)
  (h9 : plan.readingRate = 50) :
  calculateRemainingPages plan = 792 ∧
  calculateAverageSpeedFirstFourDays plan = 130.25 ∧
  calculateTotalReadingHours plan = 27 :=
sorry

end reading_plan_theorem_l194_19458


namespace average_speed_round_trip_budapest_debrecen_average_speed_l194_19416

/-- The average speed of a round trip between two cities, given the speeds for each direction. -/
theorem average_speed_round_trip (s : ℝ) (v1 v2 : ℝ) (h1 : v1 > 0) (h2 : v2 > 0) :
  let t1 := s / v1
  let t2 := s / v2
  let total_time := t1 + t2
  let total_distance := 2 * s
  total_distance / total_time = 2 * v1 * v2 / (v1 + v2) :=
by sorry

/-- The average speed of a car traveling between Budapest and Debrecen. -/
theorem budapest_debrecen_average_speed :
  let v1 := 56 -- km/h
  let v2 := 72 -- km/h
  let avg_speed := 2 * v1 * v2 / (v1 + v2)
  avg_speed = 63 :=
by sorry

end average_speed_round_trip_budapest_debrecen_average_speed_l194_19416


namespace diophantine_equation_solutions_l194_19405

theorem diophantine_equation_solutions :
  ∃! (solutions : Set (ℤ × ℤ)),
    solutions = {(4, 9), (4, -9), (-4, 9), (-4, -9)} ∧
    ∀ (x y : ℤ), (x, y) ∈ solutions ↔ 3 * x^2 + 5 * y^2 = 453 :=
by sorry

end diophantine_equation_solutions_l194_19405


namespace fraction_simplification_l194_19426

theorem fraction_simplification (a b m n : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hm : m ≠ 0) (hn : n ≠ 0) :
  (a^2 * b) / (m * n^2) / ((a * b) / (3 * m * n)) = 3 * a / n :=
by sorry

end fraction_simplification_l194_19426


namespace expression_value_l194_19498

theorem expression_value :
  let a : ℤ := 10
  let b : ℤ := 15
  let c : ℤ := 3
  let d : ℤ := 2
  (a * (b - c)) - ((a - b) * c) + d = 137 := by
  sorry

end expression_value_l194_19498


namespace locus_and_angle_property_l194_19476

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define the locus of points Q
def locus_Q (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line y = k(x-1)
def line_k (k x y : ℝ) : Prop := y = k * (x - 1)

-- Define the angle equality condition
def angle_equality (T R S : ℝ × ℝ) : Prop :=
  let (tx, _) := T
  let (rx, ry) := R
  let (sx, sy) := S
  (ry / (rx - tx)) + (sy / (sx - tx)) = 0

-- State the theorem
theorem locus_and_angle_property :
  -- Part 1: The locus of Q forms the given ellipse
  (∀ x y : ℝ, (∃ px py : ℝ, circle_E px py ∧ 
    (x - px)^2 + (y - py)^2 = ((x - 1) - px)^2 + (y - py)^2) 
    ↔ locus_Q x y) ∧
  -- Part 2: There exists a point T satisfying the angle property
  (∃ t : ℝ, t = 4 ∧ 
    ∀ k r s : ℝ, 
      locus_Q r s ∧ line_k k r s → 
      angle_equality (t, 0) (r, s) (s, k*(s-1))) :=
sorry

end locus_and_angle_property_l194_19476


namespace arithmetic_calculations_l194_19470

theorem arithmetic_calculations : 
  (26 - 7 + (-6) + 17 = 30) ∧ 
  (-81 / (9/4) * (-4/9) / (-16) = -1) ∧ 
  ((2/3 - 3/4 + 1/6) * (-36) = -3) ∧ 
  (-1^4 + 12 / (-2)^2 + 1/4 * (-8) = 0) := by
sorry

end arithmetic_calculations_l194_19470


namespace sum_equals_twelve_l194_19413

theorem sum_equals_twelve 
  (a b c : ℕ) 
  (h : 28 * a + 30 * b + 31 * c = 365) : 
  a + b + c = 12 := by
  sorry

end sum_equals_twelve_l194_19413


namespace function_relationship_l194_19487

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x ≥ -4 → y ≥ -4 → x < y → f x < f y)
variable (h2 : ∀ x, f (x - 4) = f (-x - 4))

-- State the theorem
theorem function_relationship :
  f (-4) < f (-6) ∧ f (-6) < f 0 :=
sorry

end function_relationship_l194_19487


namespace harold_adrienne_speed_difference_l194_19481

/-- Prove that Harold walks 1 mile per hour faster than Adrienne --/
theorem harold_adrienne_speed_difference :
  ∀ (total_distance : ℝ) (adrienne_speed : ℝ) (harold_catch_up_distance : ℝ),
    total_distance = 60 →
    adrienne_speed = 3 →
    harold_catch_up_distance = 12 →
    ∃ (harold_speed : ℝ),
      harold_speed > adrienne_speed ∧
      harold_speed - adrienne_speed = 1 := by
  sorry

end harold_adrienne_speed_difference_l194_19481


namespace edda_magni_winning_strategy_l194_19488

/-- Represents the hexagonal board game with n tiles on each side. -/
structure HexGame where
  n : ℕ
  n_gt_two : n > 2

/-- Represents a winning strategy for Edda and Magni. -/
def winning_strategy (game : HexGame) : Prop :=
  ∃ k : ℕ, k > 0 ∧ game.n = 3 * k + 1

/-- Theorem stating the condition for Edda and Magni to have a winning strategy. -/
theorem edda_magni_winning_strategy (game : HexGame) :
  winning_strategy game ↔ ∃ k : ℕ, k > 0 ∧ game.n = 3 * k + 1 :=
by sorry


end edda_magni_winning_strategy_l194_19488


namespace inequality_proof_l194_19418

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end inequality_proof_l194_19418


namespace unique_pair_for_n_l194_19448

theorem unique_pair_for_n (n : ℕ+) :
  ∃! (a b : ℕ+), n = (1/2) * ((a + b - 1) * (a + b - 2) : ℕ) + a := by
  sorry

end unique_pair_for_n_l194_19448


namespace fib_gcd_consecutive_fib_gcd_identity_fib_sum_identity_l194_19474

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fib_gcd_consecutive (n : ℕ) : Nat.gcd (fib n) (fib (n + 1)) = 1 := by sorry

theorem fib_gcd_identity (m n : ℕ) : 
  fib (Nat.gcd m n) = Nat.gcd (fib m) (fib n) := by sorry

theorem fib_sum_identity (m n : ℕ) :
  fib (n + m) = fib m * fib (n + 1) + fib (m - 1) * fib n := by sorry

end fib_gcd_consecutive_fib_gcd_identity_fib_sum_identity_l194_19474


namespace largest_divisor_of_n_l194_19493

theorem largest_divisor_of_n (n : ℕ) (hn : n > 0) (h_divisible : 37800 ∣ n^3) : 
  ∃ q : ℕ, q > 0 ∧ q ∣ n ∧ ∀ m : ℕ, m > 0 → m ∣ n → m ≤ q ∧ q = 6 :=
sorry

end largest_divisor_of_n_l194_19493


namespace tank_capacity_proof_l194_19489

/-- The capacity of a water tank in gallons. -/
def tank_capacity : ℝ := 72

/-- The difference in gallons between 40% full and 10% empty. -/
def difference : ℝ := 36

/-- Proves that the tank capacity is correct given the condition. -/
theorem tank_capacity_proof : 
  tank_capacity * 0.4 = tank_capacity * 0.9 - difference :=
by sorry

end tank_capacity_proof_l194_19489


namespace division_multiplication_chain_l194_19471

theorem division_multiplication_chain : (180 / 6) * 3 / 2 = 45 := by
  sorry

end division_multiplication_chain_l194_19471


namespace square_diagonals_equal_l194_19417

-- Define a structure for shapes with diagonals
structure ShapeWithDiagonals :=
  (diagonal1 : ℝ)
  (diagonal2 : ℝ)

-- Define rectangle and square
class Rectangle extends ShapeWithDiagonals

class Square extends Rectangle

-- State the theorem about rectangle diagonals
axiom rectangle_diagonals_equal (r : Rectangle) : r.diagonal1 = r.diagonal2

-- State that a square is a rectangle
axiom square_is_rectangle (s : Square) : Rectangle

-- Theorem to prove
theorem square_diagonals_equal (s : Square) : 
  (square_is_rectangle s).diagonal1 = (square_is_rectangle s).diagonal2 :=
by sorry

end square_diagonals_equal_l194_19417


namespace sum_of_x_solutions_l194_19499

theorem sum_of_x_solutions (x₁ x₂ : ℝ) (y : ℝ) (h1 : y = 5) (h2 : x₁^2 + y^2 = 169) (h3 : x₂^2 + y^2 = 169) (h4 : x₁ ≠ x₂) : x₁ + x₂ = 0 := by
  sorry

end sum_of_x_solutions_l194_19499


namespace max_rearrangeable_guests_correct_l194_19464

/-- Represents a hotel with rooms numbered from 101 to 200 --/
structure Hotel :=
  (rooms : Finset Nat)
  (room_capacity : Nat → Nat)
  (room_range : ∀ r ∈ rooms, 101 ≤ r ∧ r ≤ 200)
  (capacity_matches_number : ∀ r ∈ rooms, room_capacity r = r)

/-- The maximum number of guests that can always be rearranged --/
def max_rearrangeable_guests (h : Hotel) : Nat :=
  8824

/-- Theorem stating that max_rearrangeable_guests is correct --/
theorem max_rearrangeable_guests_correct (h : Hotel) :
  ∀ n : Nat, n ≤ max_rearrangeable_guests h →
  (∀ vacated : h.rooms, ∃ destination : h.rooms,
    vacated ≠ destination ∧
    h.room_capacity destination ≥ h.room_capacity vacated) :=
sorry

#check max_rearrangeable_guests_correct

end max_rearrangeable_guests_correct_l194_19464


namespace medal_winners_combinations_l194_19412

theorem medal_winners_combinations (semifinalists : ℕ) (advance : ℕ) (finalists : ℕ) (medals : ℕ) :
  semifinalists = 8 →
  advance = semifinalists - 2 →
  finalists = advance →
  medals = 3 →
  Nat.choose finalists medals = 20 :=
by
  sorry

end medal_winners_combinations_l194_19412


namespace distribute_7_4_l194_19432

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 104 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_7_4 : distribute 7 4 = 104 := by sorry

end distribute_7_4_l194_19432


namespace expression_evaluation_l194_19494

theorem expression_evaluation (x y : ℝ) (hx : x = 1) (hy : y = -2) :
  ((x - y)^2 - x*(3*x - 2*y) + (x + y)*(x - y)) / (2*x) = -1/2 := by
  sorry

end expression_evaluation_l194_19494


namespace value_of_B_l194_19423

theorem value_of_B : ∃ B : ℝ, (3 * B + 2 = 20) ∧ (B = 6) := by
  sorry

end value_of_B_l194_19423


namespace geometric_progression_and_sum_l194_19452

theorem geometric_progression_and_sum : ∃ x : ℝ,
  let a₁ := 10 + x
  let a₂ := 30 + x
  let a₃ := 60 + x
  (a₂ / a₁ = a₃ / a₂) ∧ (a₁ + a₂ + a₃ = 190) ∧ x = 30 := by
  sorry

end geometric_progression_and_sum_l194_19452


namespace johns_money_left_l194_19495

/-- Calculates the money John has left after walking his neighbor's dog, buying books, and giving money to his sister. -/
theorem johns_money_left (days_in_april : ℕ) (sundays_in_april : ℕ) (daily_pay : ℕ) (book_cost : ℕ) (sister_money : ℕ) : 
  days_in_april = 30 →
  sundays_in_april = 4 →
  daily_pay = 10 →
  book_cost = 50 →
  sister_money = 50 →
  (days_in_april - sundays_in_april) * daily_pay - (book_cost + sister_money) = 160 :=
by sorry

end johns_money_left_l194_19495


namespace inequality_proof_l194_19496

theorem inequality_proof (k m a b : ℝ) 
  (h : ∀ x ∈ Set.Icc a b, |x^2 - k*x - m| ≤ 1) : 
  b - a ≤ 2 * Real.sqrt 2 := by
  sorry

end inequality_proof_l194_19496


namespace cosine_symmetry_and_monotonicity_l194_19401

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

theorem cosine_symmetry_and_monotonicity (ω : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω x = f ω (3 * Real.pi / 2 - x)) →
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → f ω x ≥ f ω y) →
  ω = 2/3 ∨ ω = 2 := by sorry

end cosine_symmetry_and_monotonicity_l194_19401


namespace cinnamon_blend_probability_l194_19408

/-- The probability of exactly k successes in n independent trials with probability p of success in each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p ^ k * (1 - p) ^ (n - k)

/-- The probability of exactly 5 successes in 7 trials with 3/4 probability of success in each trial is 5103/16384. -/
theorem cinnamon_blend_probability : 
  binomial_probability 7 5 (3/4) = 5103/16384 := by
  sorry

end cinnamon_blend_probability_l194_19408


namespace replaced_man_weight_l194_19463

theorem replaced_man_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (weight_increase : ℝ) 
  (new_man_weight : ℝ) 
  (h1 : n = 10)
  (h2 : weight_increase = 2.5)
  (h3 : new_man_weight = 93) :
  new_man_weight - n * weight_increase = 68 := by
  sorry

end replaced_man_weight_l194_19463


namespace f_equals_three_implies_x_is_sqrt_three_l194_19441

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

theorem f_equals_three_implies_x_is_sqrt_three :
  ∀ x : ℝ, f x = 3 → x = Real.sqrt 3 := by sorry

end f_equals_three_implies_x_is_sqrt_three_l194_19441


namespace maurice_current_age_l194_19446

/-- Given Ron's current age and the relation between Ron and Maurice's ages after 5 years,
    prove Maurice's current age. -/
theorem maurice_current_age :
  ∀ (ron_current_age : ℕ) (maurice_current_age : ℕ),
    ron_current_age = 43 →
    ron_current_age + 5 = 4 * (maurice_current_age + 5) →
    maurice_current_age = 7 := by
  sorry

end maurice_current_age_l194_19446


namespace total_flowers_planted_l194_19402

theorem total_flowers_planted (num_people : ℕ) (num_days : ℕ) (flowers_per_day : ℕ) : 
  num_people = 5 → num_days = 2 → flowers_per_day = 20 → 
  num_people * num_days * flowers_per_day = 200 := by
  sorry

end total_flowers_planted_l194_19402


namespace cuboctahedron_volume_side_length_one_l194_19404

/-- A cuboctahedron is a polyhedron with 8 triangular faces and 6 square faces. -/
structure Cuboctahedron where
  side_length : ℝ

/-- The volume of a cuboctahedron. -/
noncomputable def volume (c : Cuboctahedron) : ℝ :=
  (5 * Real.sqrt 2) / 3

/-- Theorem: The volume of a cuboctahedron with side length 1 is (5 * √2) / 3. -/
theorem cuboctahedron_volume_side_length_one :
  volume { side_length := 1 } = (5 * Real.sqrt 2) / 3 := by
  sorry

end cuboctahedron_volume_side_length_one_l194_19404


namespace rectangle_area_theorem_l194_19425

theorem rectangle_area_theorem (m : ℕ) (hm : m > 12) :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧
  (x * y > m) ∧
  ((x - 1) * y < m) ∧
  (x * (y - 1) < m) ∧
  (∀ (a b : ℕ), a > 0 → b > 0 → a * b ≠ m) :=
by sorry

end rectangle_area_theorem_l194_19425


namespace spencer_jump_rope_l194_19462

def initial_speed : ℕ := 4
def practice_days : List ℕ := [1, 2, 4, 5, 6]
def first_session_duration : ℕ := 10
def second_session_initial : ℕ := 10
def second_session_increase : ℕ := 5

def speed_on_day (day : ℕ) : ℕ :=
  initial_speed * (2^(day - 1))

def second_session_duration (day : ℕ) : ℕ :=
  second_session_initial + (day - 1) * second_session_increase

def jumps_on_day (day : ℕ) : ℕ :=
  speed_on_day day * (first_session_duration + second_session_duration day)

def total_jumps : ℕ :=
  practice_days.map jumps_on_day |>.sum

theorem spencer_jump_rope : total_jumps = 8600 := by
  sorry

end spencer_jump_rope_l194_19462


namespace angle_subtraction_theorem_l194_19459

-- Define a custom type for angle measurements in degrees, minutes, and seconds
structure AngleDMS where
  degrees : Int
  minutes : Int
  seconds : Int

-- Define the subtraction operation for AngleDMS
def AngleDMS.sub (a b : AngleDMS) : AngleDMS :=
  sorry

theorem angle_subtraction_theorem :
  let a := AngleDMS.mk 108 18 25
  let b := AngleDMS.mk 56 23 32
  let result := AngleDMS.mk 51 54 53
  a.sub b = result := by sorry

end angle_subtraction_theorem_l194_19459


namespace expression_factorization_l194_19456

theorem expression_factorization (a b c : ℝ) (h : c ≠ 0) :
  3 * a^3 * (b^2 - c^2) - 2 * b^3 * (c^2 - a^2) + c^3 * (a^2 - b^2) =
  (a - b) * (b - c) * (c - a) * (3 * a^2 - 2 * b^2 - 3 * a^3 / c + c) := by
  sorry

end expression_factorization_l194_19456


namespace arithmetic_sequence_problem_l194_19490

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ = 7 and a₇ = 3, prove that a₁₀ = 0 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = 7) 
  (h_a7 : a 7 = 3) : 
  a 10 = 0 := by
sorry


end arithmetic_sequence_problem_l194_19490


namespace min_acute_triangles_in_square_l194_19460

/-- A triangulation of a square. -/
structure SquareTriangulation where
  /-- The number of triangles in the triangulation. -/
  num_triangles : ℕ
  /-- All triangles in the triangulation are acute-angled. -/
  all_acute : Bool
  /-- The triangulation is valid (covers the entire square without overlaps). -/
  valid : Bool

/-- The minimum number of triangles in a valid acute-angled triangulation of a square. -/
def min_acute_triangulation : ℕ := 8

/-- Theorem: The minimum number of acute-angled triangles that a square can be divided into is 8. -/
theorem min_acute_triangles_in_square :
  ∀ t : SquareTriangulation, t.valid ∧ t.all_acute → t.num_triangles ≥ min_acute_triangulation :=
by sorry

end min_acute_triangles_in_square_l194_19460


namespace expression_simplification_and_evaluation_l194_19472

theorem expression_simplification_and_evaluation :
  let m : ℚ := 2
  let expr := (2 / (m - 3) + 1) / ((2 * m - 2) / (m^2 - 6 * m + 9))
  expr = -1/2 := by sorry

end expression_simplification_and_evaluation_l194_19472


namespace min_units_B_required_twenty_units_B_not_sufficient_l194_19410

/-- Profit from selling one unit of model A (in thousand yuan) -/
def profit_A : ℝ := 3

/-- Profit from selling one unit of model B (in thousand yuan) -/
def profit_B : ℝ := 5

/-- Total number of units to be purchased -/
def total_units : ℕ := 30

/-- Minimum desired profit (in thousand yuan) -/
def min_profit : ℝ := 131

/-- Function to calculate the profit based on the number of model B units -/
def calculate_profit (units_B : ℕ) : ℝ :=
  profit_B * units_B + profit_A * (total_units - units_B)

/-- Theorem stating the minimum number of model B units required -/
theorem min_units_B_required :
  ∀ k : ℕ, k ≥ 21 → calculate_profit k ≥ min_profit :=
by sorry

/-- Theorem stating that 20 units of model B is not sufficient -/
theorem twenty_units_B_not_sufficient :
  calculate_profit 20 < min_profit :=
by sorry

end min_units_B_required_twenty_units_B_not_sufficient_l194_19410


namespace star_operation_simplification_l194_19461

/-- The star operation defined as x ★ y = 2x^2 - y -/
def star (x y : ℝ) : ℝ := 2 * x^2 - y

/-- Theorem stating that k ★ (k ★ k) = k -/
theorem star_operation_simplification (k : ℝ) : star k (star k k) = k := by
  sorry

end star_operation_simplification_l194_19461


namespace sum_product_inequality_l194_19477

theorem sum_product_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 := by
  sorry

end sum_product_inequality_l194_19477


namespace checkerboard_probability_l194_19449

/-- The size of one side of the checkerboard -/
def board_size : ℕ := 8

/-- The total number of squares on the checkerboard -/
def total_squares : ℕ := board_size * board_size

/-- The number of squares on the perimeter of the checkerboard -/
def perimeter_squares : ℕ := 4 * (board_size - 1)

/-- The number of squares not touching the outer edge -/
def inner_squares : ℕ := total_squares - perimeter_squares

/-- The probability of choosing a square not touching the outer edge -/
def inner_square_probability : ℚ := inner_squares / total_squares

theorem checkerboard_probability :
  inner_square_probability = 9 / 16 := by sorry

end checkerboard_probability_l194_19449


namespace no_valid_x_for_mean_12_l194_19465

theorem no_valid_x_for_mean_12 : 
  ¬ ∃ x : ℝ, (3 + 117 + 915 + 138 + 1917 + 2114 + x) / 7 = 12 := by
  sorry

end no_valid_x_for_mean_12_l194_19465


namespace monotonicity_when_a_is_neg_one_monotonicity_condition_on_interval_l194_19492

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Statement for part 1
theorem monotonicity_when_a_is_neg_one :
  let f₁ := f (-1)
  ∀ x y, x < y →
    (x ≤ 1/2 → y ≤ 1/2 → f₁ y ≤ f₁ x) ∧
    (1/2 ≤ x → 1/2 ≤ y → f₁ x ≤ f₁ y) :=
sorry

-- Statement for part 2
theorem monotonicity_condition_on_interval :
  ∀ a : ℝ, (∀ x y, -5 ≤ x → x < y → y ≤ 5 → 
    (f a x < f a y ∨ f a y < f a x)) ↔ 
    (a < -10 ∨ a > 10) :=
sorry

end monotonicity_when_a_is_neg_one_monotonicity_condition_on_interval_l194_19492


namespace complex_circle_equation_l194_19438

/-- The set of complex numbers z satisfying |z-i| = |3-4i| forms a circle in the complex plane -/
theorem complex_circle_equation : 
  ∃ (center : ℂ) (radius : ℝ), 
    {z : ℂ | Complex.abs (z - Complex.I) = Complex.abs (3 - 4 * Complex.I)} = 
    {z : ℂ | Complex.abs (z - center) = radius} :=
sorry

end complex_circle_equation_l194_19438


namespace regular_soda_count_l194_19455

/-- The number of regular soda bottles in a grocery store -/
def regular_soda : ℕ := sorry

/-- The number of diet soda bottles in a grocery store -/
def diet_soda : ℕ := 40

/-- The total number of regular and diet soda bottles in a grocery store -/
def total_regular_and_diet : ℕ := 89

/-- Theorem stating that the number of regular soda bottles is 49 -/
theorem regular_soda_count : regular_soda = 49 := by
  sorry

end regular_soda_count_l194_19455


namespace total_black_dots_l194_19434

/-- The number of butterflies -/
def num_butterflies : ℕ := 397

/-- The number of black dots per butterfly -/
def black_dots_per_butterfly : ℕ := 12

/-- Theorem: The total number of black dots is 4764 -/
theorem total_black_dots : num_butterflies * black_dots_per_butterfly = 4764 := by
  sorry

end total_black_dots_l194_19434


namespace integral_proof_l194_19424

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * log (abs (x - 2)) - 1 / (2 * (x - 1)^2)

theorem integral_proof (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  deriv f x = (2 * x^3 - 6 * x^2 + 7 * x - 4) / ((x - 2) * (x - 1)^3) :=
by sorry

end integral_proof_l194_19424


namespace tangent_line_and_inequality_l194_19486

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_line_and_inequality (h : Real.exp 4 > 54) :
  (∃ m : ℝ, ∀ x : ℝ, x > 0 → (2 * x + m = f x → m = -Real.exp 1)) ∧
  (∀ x : ℝ, x > 0 → -1 / Real.exp 1 ≤ f x ∧ f x < Real.exp x / (2 * x)) := by
  sorry

end tangent_line_and_inequality_l194_19486


namespace inequality_range_l194_19415

theorem inequality_range (a : ℝ) : 
  (∀ x y : ℝ, x^2 + 2*x + a ≥ -y^2 - 2*y) → a ≥ 2 := by
  sorry

end inequality_range_l194_19415


namespace inequality_system_solution_l194_19484

theorem inequality_system_solution :
  let S := {x : ℝ | (2*x - 6 < 3*x) ∧ (x - 2 + (x-1)/3 ≤ 1)}
  S = {x : ℝ | -6 < x ∧ x ≤ 5/2} := by
  sorry

end inequality_system_solution_l194_19484


namespace equal_roots_quadratic_l194_19444

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 0 → y = x) → 
  m = 4 := by
sorry

end equal_roots_quadratic_l194_19444


namespace coin_count_theorem_l194_19467

theorem coin_count_theorem (quarters_piles : Nat) (quarters_per_pile : Nat)
                           (dimes_piles : Nat) (dimes_per_pile : Nat)
                           (nickels_piles : Nat) (nickels_per_pile : Nat)
                           (pennies_piles : Nat) (pennies_per_pile : Nat) :
  quarters_piles = 7 →
  quarters_per_pile = 4 →
  dimes_piles = 4 →
  dimes_per_pile = 2 →
  nickels_piles = 6 →
  nickels_per_pile = 5 →
  pennies_piles = 3 →
  pennies_per_pile = 8 →
  quarters_piles * quarters_per_pile +
  dimes_piles * dimes_per_pile +
  nickels_piles * nickels_per_pile +
  pennies_piles * pennies_per_pile = 90 := by
  sorry

end coin_count_theorem_l194_19467


namespace marathon_end_time_l194_19403

-- Define the start time of the marathon
def start_time : Nat := 15  -- 3:00 p.m. in 24-hour format

-- Define the duration of the marathon in minutes
def duration : Nat := 780

-- Define a function to calculate the end time
def calculate_end_time (start : Nat) (duration_minutes : Nat) : Nat :=
  (start + duration_minutes / 60) % 24

-- Theorem to prove
theorem marathon_end_time :
  calculate_end_time start_time duration = 4 := by
  sorry


end marathon_end_time_l194_19403


namespace function_increasing_in_interval_l194_19453

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * sin (ω * x) + cos (ω * x)

theorem function_increasing_in_interval 
  (h_symmetry : ∀ (x : ℝ), f ω (π/6 - x) = f ω (π/6 + x))
  (h_smallest_ω : ∀ (ω' : ℝ), ω' > 0 → ω' ≥ ω)
  : StrictMonoOn f (Set.Ioo 0 (π/6)) := by sorry

end function_increasing_in_interval_l194_19453


namespace greatest_k_value_l194_19440

theorem greatest_k_value (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + k*x₁ + 7 = 0 ∧ 
    x₂^2 + k*x₂ + 7 = 0 ∧ 
    |x₁ - x₂| = Real.sqrt 85) →
  k ≤ Real.sqrt 113 :=
sorry

end greatest_k_value_l194_19440


namespace quadratic_inequality_solution_set_l194_19406

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 1 > 0} = {x : ℝ | x < -1/2 ∨ x > 1} :=
by sorry

end quadratic_inequality_solution_set_l194_19406


namespace min_value_quadratic_form_l194_19414

theorem min_value_quadratic_form :
  ∀ x y : ℝ, x^2 + x*y + y^2 ≥ 0 ∧ (x^2 + x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) :=
by sorry

end min_value_quadratic_form_l194_19414


namespace parallelogram_base_length_l194_19427

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude_base_relation : ℝ → ℝ → Prop) :
  area = 162 →
  (∀ base height, altitude_base_relation base height → height = 2 * base) →
  ∃ base : ℝ, altitude_base_relation base (2 * base) ∧ 
    area = base * (2 * base) ∧ 
    base = 9 := by
  sorry

end parallelogram_base_length_l194_19427


namespace biology_class_size_l194_19430

theorem biology_class_size :
  ∀ (S : ℕ), 
    (S : ℝ) * 0.8 * 0.25 = 8 →
    S = 40 :=
by
  sorry

end biology_class_size_l194_19430


namespace city_population_problem_l194_19420

theorem city_population_problem (p : ℝ) : 
  (0.85 * (p + 1500) = p - 45) → p = 8800 := by
  sorry

end city_population_problem_l194_19420


namespace isosceles_triangle_perimeter_l194_19411

/-- An isosceles triangle with sides of length 3 and 6 has a perimeter of 15. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 6 → b = 6 → c = 3 →
  (a = b ∨ a = c ∨ b = c) →  -- isosceles condition
  a + b > c ∧ b + c > a ∧ c + a > b →  -- triangle inequality
  a + b + c = 15 := by
  sorry

end isosceles_triangle_perimeter_l194_19411


namespace median_on_hypotenuse_l194_19454

/-- Represents a right triangle with legs a and b, and median m on the hypotenuse -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  m : ℝ

/-- The median on the hypotenuse of a right triangle with legs 6 and 8 is 5 -/
theorem median_on_hypotenuse (t : RightTriangle) (h1 : t.a = 6) (h2 : t.b = 8) : t.m = 5 := by
  sorry

end median_on_hypotenuse_l194_19454


namespace owen_turtle_ratio_l194_19433

def turtle_problem (owen_initial johanna_initial owen_after_month owen_final : ℕ) : Prop :=
  -- Owen initially has 21 turtles
  owen_initial = 21 ∧
  -- Johanna initially has 5 fewer turtles than Owen
  johanna_initial = owen_initial - 5 ∧
  -- After 1 month, Owen has a certain multiple of his initial number of turtles
  ∃ k : ℕ, owen_after_month = k * owen_initial ∧
  -- After 1 month, Johanna loses half of her turtles and donates the rest to Owen
  owen_final = owen_after_month + (johanna_initial / 2) ∧
  -- After all these events, Owen has 50 turtles
  owen_final = 50

theorem owen_turtle_ratio (owen_initial johanna_initial owen_after_month owen_final : ℕ)
  (h : turtle_problem owen_initial johanna_initial owen_after_month owen_final) :
  owen_after_month = 2 * owen_initial :=
by sorry

end owen_turtle_ratio_l194_19433


namespace log_equality_l194_19483

theorem log_equality (a b : ℝ) (ha : a = Real.log 625 / Real.log 16) (hb : b = Real.log 25 / Real.log 4) :
  a = b := by
  sorry

end log_equality_l194_19483


namespace cyclic_quadrilateral_angle_l194_19409

/-- In a cyclic quadrilateral ABCD, if angle BAC = d°, angle BCD = 43°, angle ACD = 59°, and angle BAD = 36°, then d = 42°. -/
theorem cyclic_quadrilateral_angle (d : ℝ) : 
  d + 43 + 59 + 36 = 180 → d = 42 := by sorry

end cyclic_quadrilateral_angle_l194_19409


namespace smallest_solution_of_equation_l194_19439

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 1 / (x - 3) + 1 / (x - 5) - 3 / (x - 4)
  ∃ (s : ℝ), s = 4 - Real.sqrt 3 ∧
    f s = 0 ∧
    ∀ (x : ℝ), f x = 0 → x ≥ s :=
by sorry

end smallest_solution_of_equation_l194_19439


namespace minuend_value_l194_19421

theorem minuend_value (minuend subtrahend difference : ℕ) 
  (h : minuend + subtrahend + difference = 600) : minuend = 300 := by
  sorry

end minuend_value_l194_19421


namespace not_all_odd_divisible_by_3_l194_19428

theorem not_all_odd_divisible_by_3 : ¬ (∀ n : ℕ, Odd n → 3 ∣ n) := by
  sorry

end not_all_odd_divisible_by_3_l194_19428


namespace tan_condition_l194_19435

open Real

theorem tan_condition (k : ℤ) (x : ℝ) : 
  (∃ k, x = 2 * k * π + π/4) → tan x = 1 ∧ 
  ∃ x, tan x = 1 ∧ ∀ k, x ≠ 2 * k * π + π/4 :=
by sorry

end tan_condition_l194_19435


namespace complement_intersection_MN_l194_19407

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3}

theorem complement_intersection_MN :
  (M ∩ N)ᶜ = {1, 4} :=
sorry

end complement_intersection_MN_l194_19407


namespace player_B_winning_condition_l194_19437

/-- Represents the game state -/
structure GameState where
  stones : ℕ

/-- Represents a player's move -/
structure Move where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ

/-- Checks if a move is valid according to the game rules -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  move.pile1 > 0 ∧ move.pile2 > 0 ∧ move.pile3 > 0 ∧
  move.pile1 + move.pile2 + move.pile3 = state.stones ∧
  (move.pile1 > move.pile2 ∧ move.pile1 > move.pile3) ∨
  (move.pile2 > move.pile1 ∧ move.pile2 > move.pile3) ∨
  (move.pile3 > move.pile1 ∧ move.pile3 > move.pile2)

/-- Defines a winning strategy for Player B -/
def player_B_has_winning_strategy (n : ℕ) : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (initial_move : Move),
      is_valid_move { stones := n } initial_move →
        ∃ (game_sequence : ℕ → GameState),
          game_sequence 0 = { stones := n } ∧
          (∀ i : ℕ, is_valid_move (game_sequence i) (strategy (game_sequence i))) ∧
          ∃ (end_state : ℕ), ¬is_valid_move (game_sequence end_state) (strategy (game_sequence end_state))

/-- The main theorem stating the condition for Player B's winning strategy -/
theorem player_B_winning_condition {a b : ℕ} (ha : a > 1) (hb : b > 1) :
  player_B_has_winning_strategy (a^b) ↔ ∃ k : ℕ, k > 1 ∧ (a^b = 3^k ∨ a^b = 3^k - 1) :=
sorry

end player_B_winning_condition_l194_19437
