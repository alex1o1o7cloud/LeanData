import Mathlib

namespace NUMINAMATH_CALUDE_apartment_count_l759_75996

theorem apartment_count : 
  ∀ (total : ℕ) 
    (at_least_one : ℕ) 
    (at_least_two : ℕ) 
    (only_one : ℕ),
  at_least_one = (85 * total) / 100 →
  at_least_two = (60 * total) / 100 →
  only_one = 30 →
  only_one = at_least_one - at_least_two →
  total = 75 := by
sorry

end NUMINAMATH_CALUDE_apartment_count_l759_75996


namespace NUMINAMATH_CALUDE_cruise_liner_travelers_l759_75999

theorem cruise_liner_travelers :
  ∃ a : ℕ,
    250 ≤ a ∧ a ≤ 400 ∧
    a % 15 = 8 ∧
    a % 25 = 17 ∧
    (a = 292 ∨ a = 367) :=
by sorry

end NUMINAMATH_CALUDE_cruise_liner_travelers_l759_75999


namespace NUMINAMATH_CALUDE_shifted_quadratic_coefficient_sum_l759_75964

/-- 
Given a quadratic function f(x) = 3x^2 + 2x + 5, when shifted 7 units to the right,
it results in a new quadratic function g(x) = ax^2 + bx + c.
This theorem proves that the sum of the coefficients a + b + c equals 101.
-/
theorem shifted_quadratic_coefficient_sum :
  ∀ (a b c : ℝ),
  (∀ x, (3 * (x - 7)^2 + 2 * (x - 7) + 5) = (a * x^2 + b * x + c)) →
  a + b + c = 101 := by
sorry

end NUMINAMATH_CALUDE_shifted_quadratic_coefficient_sum_l759_75964


namespace NUMINAMATH_CALUDE_total_book_pairs_l759_75966

/-- Represents the number of books in each genre -/
def books_per_genre : ℕ := 4

/-- Represents the number of genres -/
def num_genres : ℕ := 3

/-- Calculates the number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The main theorem stating the total number of possible book pairs -/
theorem total_book_pairs : 
  (choose_two num_genres * books_per_genre * books_per_genre) + 
  (choose_two books_per_genre) = 54 := by sorry

end NUMINAMATH_CALUDE_total_book_pairs_l759_75966


namespace NUMINAMATH_CALUDE_periodic_odd_function_value_l759_75982

def periodic_odd_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + 4) = f x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x)

theorem periodic_odd_function_value (f : ℝ → ℝ) 
  (h : periodic_odd_function f) : f 7.5 = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_periodic_odd_function_value_l759_75982


namespace NUMINAMATH_CALUDE_unique_coprime_pair_l759_75920

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem unique_coprime_pair :
  ∀ a b : ℕ,
    a > 0 ∧ b > 0 →
    a < b →
    (∀ n : ℕ, n > 0 → divides b ((n+2)*a^(n+1002) - (n+1)*a^(n+1001) - n*a^(n+1000))) →
    (∀ d : ℕ, d > 1 → (divides d a ∧ divides d b) → d = 1) →
    a = 3 ∧ b = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_coprime_pair_l759_75920


namespace NUMINAMATH_CALUDE_luncheon_tables_l759_75937

def tables_needed (invited : ℕ) (no_show : ℕ) (seats_per_table : ℕ) : ℕ :=
  ((invited - no_show) + seats_per_table - 1) / seats_per_table

theorem luncheon_tables :
  tables_needed 47 7 5 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_luncheon_tables_l759_75937


namespace NUMINAMATH_CALUDE_triangle_problem_l759_75922

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 7 →
  C = π / 3 →
  2 * Real.sin A = 3 * Real.sin B →
  Real.cos B = 3 * Real.sqrt 10 / 10 →
  a = 3 ∧ b = 2 ∧ Real.sin (2 * A) = (3 - 4 * Real.sqrt 3) / 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l759_75922


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l759_75991

theorem circle_ratio_after_increase (r : ℝ) (h : r > 0) : 
  (2 * π * (r + 2)) / (2 * (r + 2)) = π := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l759_75991


namespace NUMINAMATH_CALUDE_find_A_l759_75972

theorem find_A (A B : ℕ) (h : 15 = 3 * A ∧ 15 = 5 * B) : A = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l759_75972


namespace NUMINAMATH_CALUDE_sunflower_majority_on_tuesday_l759_75961

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  sunflower_seeds : Real
  other_seeds : Real

/-- Calculates the next day's feeder state -/
def next_day_state (state : FeederState) : FeederState :=
  { day := state.day + 1,
    sunflower_seeds := state.sunflower_seeds * 0.7 + 0.2,
    other_seeds := state.other_seeds * 0.4 + 0.3 }

/-- Initial state of the feeder on Sunday -/
def initial_state : FeederState :=
  { day := 1,
    sunflower_seeds := 0.4,
    other_seeds := 0.6 }

/-- Theorem stating that on Day 3 (Tuesday), sunflower seeds make up more than half of the total seeds -/
theorem sunflower_majority_on_tuesday :
  let state₃ := next_day_state (next_day_state initial_state)
  state₃.sunflower_seeds > (state₃.sunflower_seeds + state₃.other_seeds) / 2 := by
  sorry


end NUMINAMATH_CALUDE_sunflower_majority_on_tuesday_l759_75961


namespace NUMINAMATH_CALUDE_equal_share_amount_l759_75943

def emani_money : ℕ := 150
def howard_money : ℕ := emani_money - 30

def total_money : ℕ := emani_money + howard_money
def shared_amount : ℕ := total_money / 2

theorem equal_share_amount :
  shared_amount = 135 := by sorry

end NUMINAMATH_CALUDE_equal_share_amount_l759_75943


namespace NUMINAMATH_CALUDE_rectangle_count_l759_75998

/-- The number of rows and columns in the square grid -/
def gridSize : ℕ := 5

/-- The number of ways to choose 2 items from n items -/
def choose2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of different rectangles in a gridSize x gridSize square array of dots -/
def numRectangles : ℕ := (choose2 gridSize) * (choose2 gridSize)

theorem rectangle_count : numRectangles = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_count_l759_75998


namespace NUMINAMATH_CALUDE_continuity_at_6_l759_75963

def f (x : ℝ) := 5 * x^2 - 1

theorem continuity_at_6 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 6| < δ → |f x - f 6| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_6_l759_75963


namespace NUMINAMATH_CALUDE_prob_A_more_points_theorem_l759_75934

/-- Represents a soccer tournament with given conditions -/
structure SoccerTournament where
  num_teams : Nat
  num_games_per_team : Nat
  prob_A_wins_B : ℝ
  prob_win_other_games : ℝ

/-- Calculates the probability that Team A ends up with more points than Team B -/
def prob_A_more_points_than_B (tournament : SoccerTournament) : ℝ :=
  sorry

/-- The main theorem stating the probability for Team A to end up with more points -/
theorem prob_A_more_points_theorem (tournament : SoccerTournament) :
  tournament.num_teams = 7 ∧
  tournament.num_games_per_team = 6 ∧
  tournament.prob_A_wins_B = 0.6 ∧
  tournament.prob_win_other_games = 0.5 →
  prob_A_more_points_than_B tournament = 779 / 1024 :=
  sorry

end NUMINAMATH_CALUDE_prob_A_more_points_theorem_l759_75934


namespace NUMINAMATH_CALUDE_square_of_binomial_l759_75956

theorem square_of_binomial (d : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 60*x + d = (a*x + b)^2) → d = 900 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l759_75956


namespace NUMINAMATH_CALUDE_computer_operations_l759_75997

/-- Represents the performance of a computer --/
structure ComputerPerformance where
  additions_per_second : ℕ
  multiplications_per_second : ℕ
  hours : ℕ

/-- Calculates the total number of operations a computer can perform --/
def total_operations (cp : ComputerPerformance) : ℕ :=
  (cp.additions_per_second + cp.multiplications_per_second) * (cp.hours * 3600)

/-- Theorem: A computer with given specifications performs 388,800,000 operations in 3 hours --/
theorem computer_operations :
  ∃ (cp : ComputerPerformance),
    cp.additions_per_second = 12000 ∧
    cp.multiplications_per_second = 2 * cp.additions_per_second ∧
    cp.hours = 3 ∧
    total_operations cp = 388800000 := by
  sorry


end NUMINAMATH_CALUDE_computer_operations_l759_75997


namespace NUMINAMATH_CALUDE_system_solution_l759_75928

theorem system_solution (a : ℝ) :
  ∃ (x y z : ℝ),
    x^2 + y^2 - 2*z^2 = 2*a^2 ∧
    x + y + 2*z = 4*(a^2 + 1) ∧
    z^2 - x*y = a^2 ∧
    ((x = a^2 + a + 1 ∧ y = a^2 - a + 1) ∨ (x = a^2 - a + 1 ∧ y = a^2 + a + 1)) ∧
    z = a^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l759_75928


namespace NUMINAMATH_CALUDE_train_speed_conversion_l759_75940

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

/-- Conversion factor from hours to seconds -/
def h_to_s : ℝ := 3600

/-- Speed of the train in km/h -/
def train_speed_kmh : ℝ := 162

/-- Theorem stating that 162 km/h is equal to 45 m/s -/
theorem train_speed_conversion :
  (train_speed_kmh * km_to_m) / h_to_s = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l759_75940


namespace NUMINAMATH_CALUDE_homework_problem_count_l759_75921

theorem homework_problem_count (p t : ℕ) (hp : p > 10) (ht : t > 2) : 
  p * t = (2 * p - 6) * (t - 2) → p * t = 96 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_count_l759_75921


namespace NUMINAMATH_CALUDE_solution_satisfies_relationship_l759_75941

theorem solution_satisfies_relationship (x y : ℝ) : 
  (2 * x + y = 7) → (x - y = 5) → (x + 2 * y = 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_relationship_l759_75941


namespace NUMINAMATH_CALUDE_problem_solution_l759_75970

theorem problem_solution (p q r s : ℝ) 
  (h : p^2 + q^2 + r^2 + 4 = s + Real.sqrt (p + q + r - s)) : 
  s = 5/4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l759_75970


namespace NUMINAMATH_CALUDE_triangle_inequality_l759_75936

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 * (a + b + c) * (a * b + b * c + c * a) ≤ (a + b + c) * (a^2 + b^2 + c^2) + 9 * a * b * c :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l759_75936


namespace NUMINAMATH_CALUDE_a_values_l759_75958

def A (a : ℝ) : Set ℝ := {1, 2, a^2 - 3*a - 1}
def B : Set ℝ := {1, 3}

theorem a_values (a : ℝ) : (A a ∩ B = {1, 3}) → (a = -1 ∨ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_a_values_l759_75958


namespace NUMINAMATH_CALUDE_range_of_a_l759_75984

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - x - 2 ≤ 0 → x^2 - a*x - a - 2 ≤ 0) ↔ a ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l759_75984


namespace NUMINAMATH_CALUDE_remainder_divisibility_l759_75979

theorem remainder_divisibility (N : ℕ) (h : N > 0) (h1 : N % 60 = 49) : N % 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l759_75979


namespace NUMINAMATH_CALUDE_no_solutions_to_equation_l759_75930

theorem no_solutions_to_equation :
  ∀ x : ℝ, x ≠ 0 → x ≠ 5 → (2 * x^2 - 10 * x) / (x^2 - 5 * x) ≠ x - 3 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_to_equation_l759_75930


namespace NUMINAMATH_CALUDE_tom_crab_price_l759_75980

/-- A crab seller's weekly income and catch details -/
structure CrabSeller where
  buckets : ℕ
  crabs_per_bucket : ℕ
  days_per_week : ℕ
  weekly_income : ℕ

/-- Calculate the price per crab for a crab seller -/
def price_per_crab (seller : CrabSeller) : ℚ :=
  seller.weekly_income / (seller.buckets * seller.crabs_per_bucket * seller.days_per_week)

/-- Tom's crab selling business -/
def tom : CrabSeller :=
  { buckets := 8
    crabs_per_bucket := 12
    days_per_week := 7
    weekly_income := 3360 }

/-- Theorem stating that Tom sells each crab for $5 -/
theorem tom_crab_price : price_per_crab tom = 5 := by
  sorry


end NUMINAMATH_CALUDE_tom_crab_price_l759_75980


namespace NUMINAMATH_CALUDE_wide_tall_difference_l759_75946

/-- Represents a cupboard for storing glasses --/
structure Cupboard where
  capacity : ℕ

/-- Represents the collection of cupboards --/
structure CupboardCollection where
  tall : Cupboard
  wide : Cupboard
  narrow : Cupboard

/-- The problem setup --/
def setup : CupboardCollection where
  tall := { capacity := 20 }
  wide := { capacity := 0 }  -- We don't know the capacity, so we set it to 0
  narrow := { capacity := 10 }  -- After breaking one shelf

/-- The theorem to prove --/
theorem wide_tall_difference (w : ℕ) : 
  w = setup.wide.capacity → w - setup.tall.capacity = w - 20 := by
  sorry

/-- The main result --/
def result : ℕ → ℕ
  | w => w - 20

#check result

end NUMINAMATH_CALUDE_wide_tall_difference_l759_75946


namespace NUMINAMATH_CALUDE_complex_equation_solution_l759_75927

theorem complex_equation_solution (a b : ℝ) (z : ℂ) : 
  z = Complex.mk a b → z + Complex.I = (2 - Complex.I) / (1 + 2 * Complex.I) → b = -2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l759_75927


namespace NUMINAMATH_CALUDE_point_not_in_third_quadrant_l759_75912

-- Define the linear function
def f (x : ℝ) : ℝ := -x + 8

-- Define what it means for a point to be in the third quadrant
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem statement
theorem point_not_in_third_quadrant (x y : ℝ) (h : y = f x) : ¬ in_third_quadrant x y := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_third_quadrant_l759_75912


namespace NUMINAMATH_CALUDE_divisor_product_1024_implies_16_l759_75954

/-- Given a positive integer n, returns the product of all its positive integer divisors. -/
def divisorProduct (n : ℕ+) : ℕ+ :=
  sorry

/-- Theorem: If the product of the positive integer divisors of n is 1024, then n = 16. -/
theorem divisor_product_1024_implies_16 (n : ℕ+) :
  divisorProduct n = 1024 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_divisor_product_1024_implies_16_l759_75954


namespace NUMINAMATH_CALUDE_solution_set_characterization_l759_75988

theorem solution_set_characterization (k : ℝ) :
  (∀ x : ℝ, (|x - 2007| + |x + 2007| = k) ↔ (x < -2007 ∨ x > 2007)) ↔ k > 4014 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l759_75988


namespace NUMINAMATH_CALUDE_min_difference_is_one_l759_75993

/-- Triangle with integer side lengths and specific properties -/
structure IntegerTriangle where
  DE : ℕ
  EF : ℕ
  FD : ℕ
  perimeter_eq : DE + EF + FD = 398
  side_order : DE < EF ∧ EF ≤ FD

/-- The minimum difference between EF and DE in an IntegerTriangle is 1 -/
theorem min_difference_is_one :
  ∀ t : IntegerTriangle, (∀ s : IntegerTriangle, t.EF - t.DE ≤ s.EF - s.DE) → t.EF - t.DE = 1 := by
  sorry

#check min_difference_is_one

end NUMINAMATH_CALUDE_min_difference_is_one_l759_75993


namespace NUMINAMATH_CALUDE_power_sum_of_i_l759_75978

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^58 = -1 - i := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l759_75978


namespace NUMINAMATH_CALUDE_curve_is_rhombus_not_square_l759_75908

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  (|x + y| / 2) + |x - y| = 1

-- Define a rhombus
def is_rhombus (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  S = {(x, y) | |x| / a + |y| / b = 1}

-- Define the set of points satisfying the curve equation
def curve_set : Set (ℝ × ℝ) :=
  {(x, y) | curve_equation x y}

-- Theorem statement
theorem curve_is_rhombus_not_square :
  is_rhombus curve_set ∧ ¬(∃ (a : ℝ), curve_set = {(x, y) | |x| / a + |y| / a = 1}) :=
sorry

end NUMINAMATH_CALUDE_curve_is_rhombus_not_square_l759_75908


namespace NUMINAMATH_CALUDE_bug_meeting_point_l759_75945

/-- Triangle PQR with side lengths PQ = 7, QR = 8, PR = 9 -/
structure Triangle :=
  (PQ : ℝ) (QR : ℝ) (PR : ℝ)
  (PQ_eq : PQ = 7)
  (QR_eq : QR = 8)
  (PR_eq : PR = 9)

/-- Point S where bugs meet -/
def S (t : Triangle) : ℝ := sorry

/-- QS is the distance from Q to S -/
def QS (t : Triangle) : ℝ := sorry

/-- Theorem stating that QS = 5 -/
theorem bug_meeting_point (t : Triangle) : QS t = 5 := by sorry

end NUMINAMATH_CALUDE_bug_meeting_point_l759_75945


namespace NUMINAMATH_CALUDE_unique_pell_solution_l759_75935

def isPellSolution (x y : ℕ+) : Prop :=
  (x : ℤ)^2 - 2003 * (y : ℤ)^2 = 1

def isFundamentalSolution (x₀ y₀ : ℕ+) : Prop :=
  isPellSolution x₀ y₀ ∧ ∀ x y : ℕ+, isPellSolution x y → x₀ ≤ x ∧ y₀ ≤ y

def allPrimeFactorsDivide (x x₀ : ℕ+) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ x → p ∣ x₀

theorem unique_pell_solution (x₀ y₀ x y : ℕ+) :
  isFundamentalSolution x₀ y₀ →
  isPellSolution x y →
  allPrimeFactorsDivide x x₀ →
  x = x₀ ∧ y = y₀ := by
  sorry

end NUMINAMATH_CALUDE_unique_pell_solution_l759_75935


namespace NUMINAMATH_CALUDE_waxing_time_is_36_minutes_l759_75975

/-- Represents the time spent on different parts of car washing -/
structure CarWashTime where
  windows : ℕ
  body : ℕ
  tires : ℕ

/-- Calculates the total waxing time for all cars -/
def calculate_waxing_time (normal_car_time : CarWashTime) (normal_car_count : ℕ) (suv_count : ℕ) (total_time : ℕ) : ℕ :=
  let normal_car_wash_time := normal_car_time.windows + normal_car_time.body + normal_car_time.tires
  let total_wash_time_without_waxing := normal_car_wash_time * normal_car_count + (normal_car_wash_time * 2 * suv_count)
  total_time - total_wash_time_without_waxing

/-- Theorem stating that the waxing time is 36 minutes given the problem conditions -/
theorem waxing_time_is_36_minutes :
  let normal_car_time : CarWashTime := ⟨4, 7, 4⟩
  let normal_car_count : ℕ := 2
  let suv_count : ℕ := 1
  let total_time : ℕ := 96
  calculate_waxing_time normal_car_time normal_car_count suv_count total_time = 36 := by
  sorry

end NUMINAMATH_CALUDE_waxing_time_is_36_minutes_l759_75975


namespace NUMINAMATH_CALUDE_visible_friends_count_l759_75948

theorem visible_friends_count : 
  (Finset.sum (Finset.range 10) (λ i => 
    (Finset.filter (λ j => Nat.gcd (i + 1) j = 1) (Finset.range 6)).card
  )) + 10 = 36 := by
  sorry

end NUMINAMATH_CALUDE_visible_friends_count_l759_75948


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l759_75914

-- Problem 1
theorem problem_1 (a : ℝ) : (-a^2)^3 + (-2*a^3)^2 - a^3 * a^2 = 3*a^6 - a^5 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : ((x + 2*y) * (x - 2*y) + 4*(x - y)^2) + 6*x = 5*x^2 - 8*x*y + 6*x := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l759_75914


namespace NUMINAMATH_CALUDE_tan_alpha_value_l759_75967

theorem tan_alpha_value (α : Real) 
  (h1 : π/2 < α) (h2 : α < π) 
  (h3 : Real.sin α + Real.cos α = Real.sqrt 10 / 5) : 
  Real.tan α = -3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l759_75967


namespace NUMINAMATH_CALUDE_kaleb_cherries_left_l759_75992

/-- Calculates the number of cherries Kaleb has left after eating some. -/
def cherries_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem: Given Kaleb had 67 cherries initially and ate 25 cherries,
    the number of cherries he had left is equal to 42. -/
theorem kaleb_cherries_left : cherries_left 67 25 = 42 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_cherries_left_l759_75992


namespace NUMINAMATH_CALUDE_masha_sasha_numbers_l759_75995

theorem masha_sasha_numbers : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a > 11 ∧ 
  b > 11 ∧ 
  (∀ (x y : ℕ), x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y < a + b → 
    ∃! (p q : ℕ), p ≠ q ∧ p > 11 ∧ q > 11 ∧ p + q = x + y) ∧
  (Even a ∨ Even b) ∧
  (∀ (x y : ℕ), x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y = a + b → (x = 12 ∧ y = 16) ∨ (x = 16 ∧ y = 12)) :=
by
  sorry

end NUMINAMATH_CALUDE_masha_sasha_numbers_l759_75995


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l759_75924

def point_P : ℝ × ℝ := (5, -12)

theorem distance_to_x_axis :
  ‖point_P.2‖ = 12 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l759_75924


namespace NUMINAMATH_CALUDE_can_obtain_11_from_1_l759_75901

/-- Represents the allowed operations on the calculator -/
inductive Operation
  | MultiplyBy3
  | Add3
  | DivideBy3

/-- Applies a single operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.MultiplyBy3 => n * 3
  | Operation.Add3 => n + 3
  | Operation.DivideBy3 => if n % 3 = 0 then n / 3 else n

/-- Applies a sequence of operations to a number -/
def applyOperations (n : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation n

/-- Theorem stating that 11 can be obtained from 1 using the allowed operations -/
theorem can_obtain_11_from_1 : ∃ (ops : List Operation), applyOperations 1 ops = 11 :=
  sorry

end NUMINAMATH_CALUDE_can_obtain_11_from_1_l759_75901


namespace NUMINAMATH_CALUDE_best_approximation_l759_75944

def f (x : ℝ) := x^2 + 2*x

def table_values : List ℝ := [1.63, 1.64, 1.65, 1.66]

def target_value : ℝ := 6

theorem best_approximation :
  ∀ x ∈ table_values, 
    abs (f 1.65 - target_value) ≤ abs (f x - target_value) ∧
    (∀ y ∈ table_values, abs (f y - target_value) < abs (f 1.65 - target_value) → y = 1.65) :=
by sorry

end NUMINAMATH_CALUDE_best_approximation_l759_75944


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l759_75913

theorem complex_magnitude_example : Complex.abs (-3 - (5/4)*Complex.I) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l759_75913


namespace NUMINAMATH_CALUDE_coalition_percentage_is_79_percent_l759_75952

/-- Represents the election results and voter information -/
structure ElectionData where
  total_votes : ℕ
  invalid_vote_percentage : ℚ
  registered_voters : ℕ
  candidate_x_valid_percentage : ℚ
  candidate_y_valid_percentage : ℚ
  candidate_z_valid_percentage : ℚ

/-- Calculates the percentage of valid votes received by a coalition of two candidates -/
def coalition_percentage (data : ElectionData) : ℚ :=
  data.candidate_x_valid_percentage + data.candidate_y_valid_percentage

/-- Theorem stating that the coalition of candidates X and Y received 79% of the valid votes -/
theorem coalition_percentage_is_79_percent (data : ElectionData)
  (h1 : data.total_votes = 750000)
  (h2 : data.invalid_vote_percentage = 18 / 100)
  (h3 : data.registered_voters = 900000)
  (h4 : data.candidate_x_valid_percentage = 47 / 100)
  (h5 : data.candidate_y_valid_percentage = 32 / 100)
  (h6 : data.candidate_z_valid_percentage = 21 / 100) :
  coalition_percentage data = 79 / 100 := by
  sorry


end NUMINAMATH_CALUDE_coalition_percentage_is_79_percent_l759_75952


namespace NUMINAMATH_CALUDE_milan_phone_bill_l759_75939

/-- Calculates the number of minutes billed given the total bill, monthly fee, and cost per minute. -/
def minutes_billed (total_bill monthly_fee cost_per_minute : ℚ) : ℚ :=
  (total_bill - monthly_fee) / cost_per_minute

/-- Proves that given the specified conditions, the number of minutes billed is 178. -/
theorem milan_phone_bill : minutes_billed 23.36 2 0.12 = 178 := by
  sorry

end NUMINAMATH_CALUDE_milan_phone_bill_l759_75939


namespace NUMINAMATH_CALUDE_radical_simplification_l759_75906

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (8 * q^3) = 6 * q^3 * Real.sqrt (10 * q) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l759_75906


namespace NUMINAMATH_CALUDE_quadratic_properties_l759_75950

-- Define the quadratic function
def y (m x : ℝ) : ℝ := 2*m*x^2 + (1-m)*x - 1 - m

-- Theorem statement
theorem quadratic_properties :
  -- 1. When m = -1, the vertex of the graph is at (1/2, 1/2)
  (y (-1) (1/2) = 1/2) ∧
  -- 2. When m > 0, the length of the segment intercepted by the graph on the x-axis is greater than 3/2
  (∀ m > 0, ∃ x₁ x₂, y m x₁ = 0 ∧ y m x₂ = 0 ∧ |x₁ - x₂| > 3/2) ∧
  -- 3. When m ≠ 0, the graph always passes through the fixed points (1, 0) and (-1/2, -3/2)
  (∀ m ≠ 0, y m 1 = 0 ∧ y m (-1/2) = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l759_75950


namespace NUMINAMATH_CALUDE_quadratic_max_value_l759_75973

theorem quadratic_max_value (m : ℝ) : 
  (∃ (y : ℝ → ℝ), 
    (∀ x, y x = -(x - m)^2 + m^2 + 1) ∧ 
    (∀ x, -2 ≤ x ∧ x ≤ 1 → y x ≤ 4) ∧
    (∃ x, -2 ≤ x ∧ x ≤ 1 ∧ y x = 4)) →
  m = 2 ∨ m = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l759_75973


namespace NUMINAMATH_CALUDE_mikes_score_l759_75962

def passing_threshold : ℝ := 0.30
def max_score : ℕ := 770
def shortfall : ℕ := 19

theorem mikes_score : 
  ⌊(passing_threshold * max_score : ℝ)⌋ - shortfall = 212 := by
  sorry

end NUMINAMATH_CALUDE_mikes_score_l759_75962


namespace NUMINAMATH_CALUDE_smallest_t_for_70_degrees_l759_75989

-- Define the temperature function
def T (t : ℝ) : ℝ := -t^2 + 10*t + 60

-- Define the atmospheric pressure function (not used in the proof, but included for completeness)
def P (t : ℝ) : ℝ := 800 - 2*t

-- Theorem statement
theorem smallest_t_for_70_degrees :
  ∃ (t : ℝ), t > 0 ∧ T t = 70 ∧ ∀ (s : ℝ), s > 0 ∧ T s = 70 → t ≤ s :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_t_for_70_degrees_l759_75989


namespace NUMINAMATH_CALUDE_simplify_expression_l759_75953

theorem simplify_expression (a b : ℝ) (h1 : a + b = 0) (h2 : a ≠ b) :
  (1 - a) + (1 - b) = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l759_75953


namespace NUMINAMATH_CALUDE_sin_greater_cos_range_l759_75900

theorem sin_greater_cos_range (x : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) → 
  (Real.sin x > Real.cos x ↔ x ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4)) := by
  sorry

end NUMINAMATH_CALUDE_sin_greater_cos_range_l759_75900


namespace NUMINAMATH_CALUDE_go_game_draw_probability_l759_75960

theorem go_game_draw_probability 
  (p_not_lose : ℝ) 
  (p_win : ℝ) 
  (h1 : p_not_lose = 0.6) 
  (h2 : p_win = 0.5) : 
  p_not_lose - p_win = 0.1 := by
sorry

end NUMINAMATH_CALUDE_go_game_draw_probability_l759_75960


namespace NUMINAMATH_CALUDE_words_with_consonants_count_l759_75933

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'I'}
def vowels : Finset Char := {'A', 'E', 'I'}
def consonants : Finset Char := {'B', 'C', 'D'}

def word_length : Nat := 5

theorem words_with_consonants_count :
  (alphabet.card ^ word_length) - (vowels.card ^ word_length) = 7533 := by
  sorry

end NUMINAMATH_CALUDE_words_with_consonants_count_l759_75933


namespace NUMINAMATH_CALUDE_function_value_at_three_l759_75911

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

theorem function_value_at_three
    (f : ℝ → ℝ)
    (h1 : FunctionalEquation f)
    (h2 : f 1 = 2) :
    f 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_three_l759_75911


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l759_75903

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) : 
  π * r^2 = 100 * π → 2 * π * r^2 + π * r^2 = 300 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l759_75903


namespace NUMINAMATH_CALUDE_milo_run_distance_milo_two_hour_run_l759_75929

/-- Milo's running speed in miles per hour -/
def milo_run_speed : ℝ := 3

/-- Milo's skateboard speed in miles per hour -/
def milo_skateboard_speed : ℝ := 2 * milo_run_speed

/-- Cory's wheelchair speed in miles per hour -/
def cory_wheelchair_speed : ℝ := 12

theorem milo_run_distance : ℝ → ℝ
  | hours => milo_run_speed * hours

theorem milo_two_hour_run : milo_run_distance 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_milo_run_distance_milo_two_hour_run_l759_75929


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4704_l759_75951

theorem largest_prime_factor_of_4704 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4704 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 4704 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4704_l759_75951


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l759_75987

/-- Given a rectangular parallelepiped with face areas p, q, and r, its volume is √(pqr) -/
theorem parallelepiped_volume (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  ∃ (V : ℝ), V > 0 ∧ V * V = p * q * r :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l759_75987


namespace NUMINAMATH_CALUDE_smallest_positive_integer_to_multiple_of_five_l759_75947

theorem smallest_positive_integer_to_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → (725 + m) % 5 = 0 → m ≥ n) ∧ (725 + n) % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_to_multiple_of_five_l759_75947


namespace NUMINAMATH_CALUDE_max_d_value_l759_75917

def is_multiple_of_66 (n : ℕ) : Prop := n % 66 = 0

def has_form_4d645e (n : ℕ) (d e : ℕ) : Prop :=
  n = 400000 + 10000 * d + 6000 + 400 + 50 + e ∧ d < 10 ∧ e < 10

theorem max_d_value (n : ℕ) (d e : ℕ) :
  is_multiple_of_66 n → has_form_4d645e n d e → d ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l759_75917


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l759_75974

theorem quadratic_inequality_solution_set (a b : ℝ) : 
  (∀ x, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a = -12 ∧ b = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l759_75974


namespace NUMINAMATH_CALUDE_cell_population_growth_l759_75915

/-- Represents the number of cells in the population after n hours -/
def cell_count (n : ℕ) : ℕ :=
  2^(n-1) + 4

/-- The rule for cell population growth -/
def cell_growth_rule (prev : ℕ) : ℕ :=
  2 * (prev - 2)

theorem cell_population_growth (n : ℕ) :
  n > 0 →
  cell_count 1 = 5 →
  (∀ k, k ≥ 1 → cell_count (k + 1) = cell_growth_rule (cell_count k)) →
  cell_count n = 2^(n-1) + 4 :=
by
  sorry

#check cell_population_growth

end NUMINAMATH_CALUDE_cell_population_growth_l759_75915


namespace NUMINAMATH_CALUDE_student_calculation_error_l759_75931

/-- Represents a repeating decimal of the form 1.̅cd̅ where c and d are single digits -/
def repeating_decimal (c d : ℕ) : ℚ :=
  1 + (10 * c + d : ℚ) / 99

/-- The difference between the correct calculation and the student's miscalculation -/
def calculation_difference (c d : ℕ) : ℚ :=
  84 * (repeating_decimal c d - (1 + (c : ℚ) / 10 + (d : ℚ) / 100))

theorem student_calculation_error :
  ∃ (c d : ℕ), c < 10 ∧ d < 10 ∧ calculation_difference c d = 0.6 ∧ c * 10 + d = 71 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_error_l759_75931


namespace NUMINAMATH_CALUDE_power_relationship_l759_75957

theorem power_relationship (y : ℝ) (h : (10 : ℝ) ^ (4 * y) = 49) : (10 : ℝ) ^ (-2 * y) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_power_relationship_l759_75957


namespace NUMINAMATH_CALUDE_lines_1_and_4_are_perpendicular_l759_75968

-- Define the slopes of the lines
def slope1 : ℚ := 3 / 4
def slope4 : ℚ := -4 / 3

-- Define the condition for perpendicularity
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem lines_1_and_4_are_perpendicular : 
  are_perpendicular slope1 slope4 := by sorry

end NUMINAMATH_CALUDE_lines_1_and_4_are_perpendicular_l759_75968


namespace NUMINAMATH_CALUDE_rectangular_program_box_indicates_input_output_l759_75910

/-- Represents the function of a program box in an algorithm -/
inductive ProgramBoxFunction
  | StartEnd
  | InputOutput
  | AssignmentCalculation
  | ConnectBoxes

/-- The function of a rectangular program box in an algorithm -/
def rectangularProgramBoxFunction : ProgramBoxFunction := ProgramBoxFunction.InputOutput

/-- Theorem stating that a rectangular program box indicates input and output information -/
theorem rectangular_program_box_indicates_input_output :
  rectangularProgramBoxFunction = ProgramBoxFunction.InputOutput := by
  sorry

end NUMINAMATH_CALUDE_rectangular_program_box_indicates_input_output_l759_75910


namespace NUMINAMATH_CALUDE_number_of_factors_of_M_l759_75907

def M : ℕ := 57^5 + 5*57^4 + 10*57^3 + 10*57^2 + 5*57 + 1

theorem number_of_factors_of_M : 
  (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 36 :=
by sorry

end NUMINAMATH_CALUDE_number_of_factors_of_M_l759_75907


namespace NUMINAMATH_CALUDE_function_inequality_solution_set_l759_75990

open Set
open Function

theorem function_inequality_solution_set 
  (f : ℝ → ℝ) 
  (h1 : f 1 = 1) 
  (h2 : ∀ x, deriv f x < (1/2)) :
  {x | f x < x/2 + 1/2} = {x | x > 1} := by
sorry

end NUMINAMATH_CALUDE_function_inequality_solution_set_l759_75990


namespace NUMINAMATH_CALUDE_polyhedron_volume_l759_75965

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the polyhedron formed by cutting a regular quadrangular prism -/
structure Polyhedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  C1 : Point3D
  D1 : Point3D
  O : Point3D  -- Center of the base

/-- The volume of the polyhedron -/
def volume (p : Polyhedron) : ℝ := sorry

/-- The dihedral angle between two planes -/
def dihedralAngle (plane1 plane2 : Set Point3D) : ℝ := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Main theorem stating the volume of the polyhedron -/
theorem polyhedron_volume (p : Polyhedron) :
  (distance p.A p.B = 1) →  -- AB = 1
  (distance p.A p.A1 = distance p.O p.C1) →  -- AA₁ = OC₁
  (dihedralAngle {p.A, p.B, p.C, p.D} {p.A1, p.B, p.C1, p.D1} = π/4) →  -- 45° dihedral angle
  (volume p = Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l759_75965


namespace NUMINAMATH_CALUDE_number_of_tests_l759_75919

theorem number_of_tests (n : ℕ) (S : ℝ) : 
  (S + 97) / n = 90 → 
  (S + 73) / n = 87 → 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_number_of_tests_l759_75919


namespace NUMINAMATH_CALUDE_divides_two_pow_minus_one_l759_75923

theorem divides_two_pow_minus_one (n : ℕ) : n > 0 → (n ∣ 2^n - 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divides_two_pow_minus_one_l759_75923


namespace NUMINAMATH_CALUDE_trim_100_edge_polyhedron_l759_75932

/-- Represents a polyhedron before and after vertex trimming --/
structure TrimmedPolyhedron where
  initial_edges : ℕ
  is_convex : Bool
  trimmed_vertices : ℕ
  trimmed_edges : ℕ

/-- Represents the process of trimming vertices of a polyhedron --/
def trim_vertices (p : TrimmedPolyhedron) : TrimmedPolyhedron :=
  { p with
    trimmed_vertices := 2 * p.initial_edges,
    trimmed_edges := 3 * p.initial_edges
  }

/-- Theorem stating the result of trimming vertices of a specific polyhedron --/
theorem trim_100_edge_polyhedron :
  ∀ p : TrimmedPolyhedron,
    p.initial_edges = 100 →
    p.is_convex = true →
    (trim_vertices p).trimmed_vertices = 200 ∧
    (trim_vertices p).trimmed_edges = 300 := by
  sorry


end NUMINAMATH_CALUDE_trim_100_edge_polyhedron_l759_75932


namespace NUMINAMATH_CALUDE_horner_method_v3_l759_75926

def f (x : ℝ) : ℝ := 3*x^5 - 2*x^4 + 2*x^3 - 4*x^2 - 7

def horner_v3 (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (e : ℝ) (f : ℝ) (x : ℝ) : ℝ :=
  ((((a * x + b) * x + c) * x + d) * x + e) * x + f

theorem horner_method_v3 : 
  horner_v3 3 (-2) 2 (-4) 0 (-7) 2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l759_75926


namespace NUMINAMATH_CALUDE_shelter_cat_dog_difference_l759_75918

/-- Given an animal shelter with a total of 60 animals and 40 cats,
    prove that the number of cats exceeds the number of dogs by 20. -/
theorem shelter_cat_dog_difference :
  let total_animals : ℕ := 60
  let num_cats : ℕ := 40
  let num_dogs : ℕ := total_animals - num_cats
  num_cats - num_dogs = 20 := by
  sorry

end NUMINAMATH_CALUDE_shelter_cat_dog_difference_l759_75918


namespace NUMINAMATH_CALUDE_dante_coconuts_left_l759_75938

theorem dante_coconuts_left (paolo_coconuts : ℕ) (dante_coconuts : ℕ) (sold_coconuts : ℕ) : 
  paolo_coconuts = 14 →
  dante_coconuts = 3 * paolo_coconuts →
  sold_coconuts = 10 →
  dante_coconuts - sold_coconuts = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_dante_coconuts_left_l759_75938


namespace NUMINAMATH_CALUDE_qt_length_in_specific_quadrilateral_l759_75942

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Theorem: Length of QT in a specific quadrilateral -/
theorem qt_length_in_specific_quadrilateral 
  (PQRS : Quadrilateral) 
  (T : Point) 
  (h1 : distance PQRS.P PQRS.Q = 15)
  (h2 : distance PQRS.R PQRS.S = 20)
  (h3 : distance PQRS.P PQRS.R = 22)
  (h4 : triangleArea PQRS.P T PQRS.R = triangleArea PQRS.Q T PQRS.S) :
  distance PQRS.Q T = 66 / 7 := by
  sorry

end NUMINAMATH_CALUDE_qt_length_in_specific_quadrilateral_l759_75942


namespace NUMINAMATH_CALUDE_intersection_of_lines_l759_75916

theorem intersection_of_lines (k : ℝ) : 
  (∃ x y : ℝ, y = -2 * x + 3 ∧ y = k * x + 4 ∧ x = 1 ∧ y = 1) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l759_75916


namespace NUMINAMATH_CALUDE_paving_problem_l759_75949

/-- Represents a worker paving paths in a park -/
structure Worker where
  speed : ℝ
  path_length : ℝ

/-- Represents the paving scenario in the park -/
structure PavingScenario where
  worker1 : Worker
  worker2 : Worker
  total_time : ℝ

/-- The theorem statement for the paving problem -/
theorem paving_problem (scenario : PavingScenario) :
  scenario.worker1.speed > 0 ∧
  scenario.worker2.speed = 1.2 * scenario.worker1.speed ∧
  scenario.total_time = 9 ∧
  scenario.worker1.path_length * scenario.worker1.speed = scenario.worker2.path_length * scenario.worker2.speed ∧
  scenario.worker2.path_length = scenario.worker1.path_length + 2 * (scenario.worker2.path_length / 12) →
  (scenario.worker2.path_length / 12) / scenario.worker2.speed * 60 = 45 := by
  sorry

#check paving_problem

end NUMINAMATH_CALUDE_paving_problem_l759_75949


namespace NUMINAMATH_CALUDE_different_pairs_eq_48_l759_75904

/-- The number of distinct mystery novels -/
def mystery_novels : ℕ := 4

/-- The number of distinct fantasy novels -/
def fantasy_novels : ℕ := 4

/-- The number of distinct biographies -/
def biographies : ℕ := 4

/-- The number of genres -/
def num_genres : ℕ := 3

/-- The number of different pairs of books that can be chosen -/
def different_pairs : ℕ := num_genres * mystery_novels * fantasy_novels

theorem different_pairs_eq_48 : different_pairs = 48 := by
  sorry

end NUMINAMATH_CALUDE_different_pairs_eq_48_l759_75904


namespace NUMINAMATH_CALUDE_f_min_value_l759_75977

-- Define the function
def f (x : ℝ) : ℝ := |x - 2| + |3 - x|

-- State the theorem
theorem f_min_value :
  (∀ x : ℝ, f x ≥ 1) ∧ (∃ x : ℝ, f x = 1) :=
sorry

end NUMINAMATH_CALUDE_f_min_value_l759_75977


namespace NUMINAMATH_CALUDE_probability_at_least_one_girl_l759_75971

def committee_size : ℕ := 7
def num_boys : ℕ := 4
def num_girls : ℕ := 3
def num_selected : ℕ := 2

theorem probability_at_least_one_girl :
  let total_combinations := Nat.choose committee_size num_selected
  let combinations_with_no_girls := Nat.choose num_boys num_selected
  let favorable_combinations := total_combinations - combinations_with_no_girls
  (favorable_combinations : ℚ) / total_combinations = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_girl_l759_75971


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l759_75902

theorem algebraic_expression_value (x : ℝ) : 
  2 * x^2 + 3 * x + 7 = 8 → 2 * x^2 + 3 * x - 7 = -6 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l759_75902


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l759_75985

theorem imaginary_part_of_z (z : ℂ) : z = (2 + Complex.I) / Complex.I → Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l759_75985


namespace NUMINAMATH_CALUDE_maria_towel_problem_l759_75955

/-- Represents the number of towels Maria has -/
structure TowelCount where
  green : ℕ
  white : ℕ
  blue : ℕ

/-- Calculates the total number of towels -/
def TowelCount.total (t : TowelCount) : ℕ :=
  t.green + t.white + t.blue

/-- Represents the number of towels given away each day -/
structure DailyGiveaway where
  green : ℕ
  white : ℕ
  blue : ℕ

/-- Calculates the remaining towels after giving away for a number of days -/
def remainingTowels (initial : TowelCount) (daily : DailyGiveaway) (days : ℕ) : TowelCount :=
  { green := initial.green - daily.green * days,
    white := initial.white - daily.white * days,
    blue := initial.blue - daily.blue * days }

theorem maria_towel_problem :
  let initial := TowelCount.mk 35 21 15
  let daily := DailyGiveaway.mk 3 1 1
  let days := 7
  let remaining := remainingTowels initial daily days
  remaining.total = 36 := by sorry

end NUMINAMATH_CALUDE_maria_towel_problem_l759_75955


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l759_75905

/-- The length of a train given its speed, a man's speed in the opposite direction, and the time it takes to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed + man_speed
  let relative_speed_ms := relative_speed * (5 / 18)
  relative_speed_ms * passing_time

/-- Proof that a train with speed 60 km/hr passing a man running at 6 km/hr in the opposite direction
    in approximately 29.997600191984645 seconds has a length of approximately 550 meters. -/
theorem train_length_proof : 
  ∃ ε > 0, |train_length 60 6 29.997600191984645 - 550| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l759_75905


namespace NUMINAMATH_CALUDE_cookie_sheet_perimeter_is_24_l759_75925

/-- The perimeter of a rectangular cookie sheet -/
def cookie_sheet_perimeter (width length : ℝ) : ℝ :=
  2 * width + 2 * length

/-- Theorem: The perimeter of a rectangular cookie sheet with width 10 inches and length 2 inches is 24 inches -/
theorem cookie_sheet_perimeter_is_24 :
  cookie_sheet_perimeter 10 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sheet_perimeter_is_24_l759_75925


namespace NUMINAMATH_CALUDE_square_of_99_l759_75959

theorem square_of_99 : 99 ^ 2 = 9801 := by
  sorry

end NUMINAMATH_CALUDE_square_of_99_l759_75959


namespace NUMINAMATH_CALUDE_johns_children_l759_75986

theorem johns_children (john_notebooks : ℕ → ℕ) (wife_notebooks : ℕ → ℕ) (total_notebooks : ℕ) :
  (∀ c : ℕ, john_notebooks c = 2 * c) →
  (∀ c : ℕ, wife_notebooks c = 5 * c) →
  (∃ c : ℕ, john_notebooks c + wife_notebooks c = total_notebooks) →
  total_notebooks = 21 →
  ∃ c : ℕ, c = 3 ∧ john_notebooks c + wife_notebooks c = total_notebooks :=
by sorry

end NUMINAMATH_CALUDE_johns_children_l759_75986


namespace NUMINAMATH_CALUDE_log_sqrt2_and_inequality_l759_75976

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sqrt2_and_inequality :
  (log 4 (Real.sqrt 2) = 1/4) ∧
  (∀ x : ℝ, log x (Real.sqrt 2) > 1 ↔ 1 < x ∧ x < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_log_sqrt2_and_inequality_l759_75976


namespace NUMINAMATH_CALUDE_freds_carrots_l759_75981

/-- Given that Sally grew 6 carrots and the total number of carrots is 10,
    prove that Fred grew 4 carrots. -/
theorem freds_carrots (sally_carrots : ℕ) (total_carrots : ℕ) (h1 : sally_carrots = 6) (h2 : total_carrots = 10) :
  total_carrots - sally_carrots = 4 := by
  sorry

end NUMINAMATH_CALUDE_freds_carrots_l759_75981


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_5_4_l759_75994

/-- Represents the financial state of a person --/
structure FinancialState where
  income : ℕ
  savings : ℕ

/-- Calculates the expenditure given income and savings --/
def expenditure (fs : FinancialState) : ℕ :=
  fs.income - fs.savings

/-- Represents a ratio as a pair of natural numbers --/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Simplifies a ratio by dividing both parts by their GCD --/
def simplifyRatio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.numerator r.denominator
  { numerator := r.numerator / gcd,
    denominator := r.denominator / gcd }

/-- Calculates the ratio of income to expenditure --/
def incomeToExpenditureRatio (fs : FinancialState) : Ratio :=
  simplifyRatio { numerator := fs.income, denominator := expenditure fs }

theorem income_expenditure_ratio_5_4 (fs : FinancialState) 
  (h1 : fs.income = 15000) (h2 : fs.savings = 3000) : 
  incomeToExpenditureRatio fs = { numerator := 5, denominator := 4 } := by
  sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_5_4_l759_75994


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l759_75969

/-- The radius of the inscribed circle of a triangle with sides 6, 8, and 10 is 2 -/
theorem inscribed_circle_radius (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l759_75969


namespace NUMINAMATH_CALUDE_reading_difference_l759_75983

/-- Calculates the total pages read given a list of (rate, days) pairs -/
def totalPagesRead (readingPlan : List (Nat × Nat)) : Nat :=
  readingPlan.map (fun (rate, days) => rate * days) |>.sum

theorem reading_difference : 
  let gregPages := totalPagesRead [(18, 7), (22, 14)]
  let bradPages := totalPagesRead [(26, 5), (20, 12)]
  let emilyPages := totalPagesRead [(15, 3), (24, 7), (18, 7)]
  gregPages + bradPages - emilyPages = 465 := by
  sorry

#eval totalPagesRead [(18, 7), (22, 14)] -- Greg's pages
#eval totalPagesRead [(26, 5), (20, 12)] -- Brad's pages
#eval totalPagesRead [(15, 3), (24, 7), (18, 7)] -- Emily's pages

end NUMINAMATH_CALUDE_reading_difference_l759_75983


namespace NUMINAMATH_CALUDE_exist_three_distinct_digits_forming_squares_l759_75909

/-- A function that constructs a three-digit number from three digits -/
def threeDigitNumber (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

/-- Theorem stating the existence of three distinct digits forming squares -/
theorem exist_three_distinct_digits_forming_squares :
  ∃ (A B C : Nat),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    ∃ (x y z : Nat),
      threeDigitNumber A B C = x^2 ∧
      threeDigitNumber C B A = y^2 ∧
      threeDigitNumber C A B = z^2 :=
by
  sorry

#eval threeDigitNumber 9 6 1
#eval threeDigitNumber 1 6 9
#eval threeDigitNumber 1 9 6

end NUMINAMATH_CALUDE_exist_three_distinct_digits_forming_squares_l759_75909
