import Mathlib

namespace NUMINAMATH_CALUDE_intersection_probability_theorem_l3551_355169

/-- The probability that two randomly chosen diagonals intersect in a convex polygon with 2n + 1 vertices. -/
def intersection_probability (n : ℕ) : ℚ :=
  if n > 0 then
    (n * (2 * n - 1)) / (3 * (2 * n^2 - n - 2))
  else
    0

/-- Theorem: In a convex polygon with 2n + 1 vertices (n > 0), the probability that two randomly
    chosen diagonals intersect is n(2n - 1) / (3(2n^2 - n - 2)). -/
theorem intersection_probability_theorem (n : ℕ) (h : n > 0) :
  intersection_probability n = (n * (2 * n - 1)) / (3 * (2 * n^2 - n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_probability_theorem_l3551_355169


namespace NUMINAMATH_CALUDE_complex_roots_on_circle_l3551_355188

theorem complex_roots_on_circle : ∀ z : ℂ, 
  (z + 2)^6 = 64 * z^6 → Complex.abs (z - (2/3 : ℂ)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_on_circle_l3551_355188


namespace NUMINAMATH_CALUDE_noah_small_paintings_l3551_355145

/-- Represents the number of small paintings Noah sold last month -/
def small_paintings : ℕ := sorry

/-- Price of a large painting in dollars -/
def large_painting_price : ℕ := 60

/-- Price of a small painting in dollars -/
def small_painting_price : ℕ := 30

/-- Number of large paintings sold last month -/
def large_paintings_last_month : ℕ := 8

/-- This month's sales in dollars -/
def this_month_sales : ℕ := 1200

theorem noah_small_paintings : 
  2 * (large_painting_price * large_paintings_last_month + small_painting_price * small_paintings) = this_month_sales ∧ 
  small_paintings = 4 := by sorry

end NUMINAMATH_CALUDE_noah_small_paintings_l3551_355145


namespace NUMINAMATH_CALUDE_passing_marks_calculation_l3551_355136

theorem passing_marks_calculation (T : ℝ) (P : ℝ) : 
  (0.20 * T = P - 40) → 
  (0.30 * T = P + 20) → 
  P = 160 := by
sorry

end NUMINAMATH_CALUDE_passing_marks_calculation_l3551_355136


namespace NUMINAMATH_CALUDE_add_three_preserves_inequality_l3551_355107

theorem add_three_preserves_inequality (a b : ℝ) (h : a > b) : a + 3 > b + 3 := by
  sorry

end NUMINAMATH_CALUDE_add_three_preserves_inequality_l3551_355107


namespace NUMINAMATH_CALUDE_solve_system_l3551_355195

theorem solve_system (a b : ℚ) 
  (h1 : -3 / (a - 3) = 3 / (a + 2))
  (h2 : (a^2 - b^2)/(a - b) = 7) :
  a = 1/2 ∧ b = 13/2 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l3551_355195


namespace NUMINAMATH_CALUDE_bell_interval_problem_l3551_355158

theorem bell_interval_problem (x : ℕ) :
  (∃ n : ℕ, n > 0 ∧ n * 5 = 1320) ∧
  (∃ n : ℕ, n > 0 ∧ n * 8 = 1320) ∧
  (∃ n : ℕ, n > 0 ∧ n * x = 1320) ∧
  (∃ n : ℕ, n > 0 ∧ n * 15 = 1320) →
  x = 11 := by
sorry

end NUMINAMATH_CALUDE_bell_interval_problem_l3551_355158


namespace NUMINAMATH_CALUDE_odd_number_factorial_not_divisible_by_square_l3551_355114

/-- A function that checks if a natural number is odd -/
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

/-- A function that checks if a natural number is prime -/
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem odd_number_factorial_not_divisible_by_square (n : ℕ) :
  is_odd n → (factorial (n - 1) % (n^2) ≠ 0 ↔ is_prime n ∨ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_odd_number_factorial_not_divisible_by_square_l3551_355114


namespace NUMINAMATH_CALUDE_sin_right_angle_l3551_355105

theorem sin_right_angle (D E F : ℝ) (h1 : D = 90) (h2 : DE = 12) (h3 : EF = 35) : Real.sin D = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_right_angle_l3551_355105


namespace NUMINAMATH_CALUDE_even_perfect_square_factors_count_l3551_355176

def num_factors (n : ℕ) : ℕ := sorry

def is_even_perfect_square (n : ℕ) : Prop := sorry

theorem even_perfect_square_factors_count : 
  ∃ (f : ℕ → ℕ), 
    (∀ x, is_even_perfect_square (f x)) ∧ 
    (∀ x, f x ∣ (2^6 * 5^3 * 7^8)) ∧ 
    (num_factors (2^6 * 5^3 * 7^8) = 30) := by
  sorry

end NUMINAMATH_CALUDE_even_perfect_square_factors_count_l3551_355176


namespace NUMINAMATH_CALUDE_speed_difference_l3551_355192

/-- Proves that the difference in average speed between two people traveling the same distance,
    where one travels at 12 miles per hour and the other completes the journey in 10 minutes,
    is 24 miles per hour. -/
theorem speed_difference (distance : ℝ) (speed_maya : ℝ) (time_naomi : ℝ) : 
  distance > 0 ∧ speed_maya = 12 ∧ time_naomi = 1/6 →
  (distance / time_naomi) - speed_maya = 24 :=
by sorry

end NUMINAMATH_CALUDE_speed_difference_l3551_355192


namespace NUMINAMATH_CALUDE_sqrt_nat_or_irrational_l3551_355103

theorem sqrt_nat_or_irrational (n : ℕ) : 
  (∃ m : ℕ, m * m = n) ∨ (∀ p q : ℕ, q > 0 → p * p ≠ n * q * q) :=
sorry

end NUMINAMATH_CALUDE_sqrt_nat_or_irrational_l3551_355103


namespace NUMINAMATH_CALUDE_circles_are_externally_tangent_l3551_355115

/-- Circle represented by its equation in the form (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  r : ℝ  -- radius
  r_pos : r > 0

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.h - c2.h)^2 + (c1.k - c2.k)^2 = (c1.r + c2.r)^2

theorem circles_are_externally_tangent :
  let c1 : Circle := { h := 0, k := 0, r := 1, r_pos := by norm_num }
  let c2 : Circle := { h := 0, k := 3, r := 2, r_pos := by norm_num }
  are_externally_tangent c1 c2 := by sorry

end NUMINAMATH_CALUDE_circles_are_externally_tangent_l3551_355115


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3551_355130

theorem ratio_x_to_y (x y : ℚ) (h : (10 * x - 3 * y) / (13 * x - 2 * y) = 3 / 5) :
  x / y = 9 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3551_355130


namespace NUMINAMATH_CALUDE_exists_multiple_sum_of_digits_divides_l3551_355139

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For all positive integers n, there exists a multiple k of n such that 
    the sum of digits of k divides k. -/
theorem exists_multiple_sum_of_digits_divides (n : ℕ+) : 
  ∃ k : ℕ+, n ∣ k ∧ sum_of_digits k ∣ k := by
  sorry

end NUMINAMATH_CALUDE_exists_multiple_sum_of_digits_divides_l3551_355139


namespace NUMINAMATH_CALUDE_power_expression_l3551_355191

theorem power_expression (x y : ℝ) (a b : ℝ) (h1 : 10^x = a) (h2 : 10^y = b) :
  10^(3*x + 2*y) = a^3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_l3551_355191


namespace NUMINAMATH_CALUDE_black_tiles_to_total_tiles_l3551_355140

/-- Represents a square room tiled with congruent square tiles -/
structure TiledRoom where
  side_length : ℕ

/-- Counts the number of black tiles in the room -/
def count_black_tiles (room : TiledRoom) : ℕ :=
  4 * room.side_length - 3

/-- Counts the total number of tiles in the room -/
def count_total_tiles (room : TiledRoom) : ℕ :=
  room.side_length * room.side_length

/-- Theorem stating the relationship between black tiles and total tiles -/
theorem black_tiles_to_total_tiles :
  ∃ (room : TiledRoom), count_black_tiles room = 201 ∧ count_total_tiles room = 2601 :=
sorry

end NUMINAMATH_CALUDE_black_tiles_to_total_tiles_l3551_355140


namespace NUMINAMATH_CALUDE_expression_evaluation_l3551_355194

theorem expression_evaluation :
  let x : ℝ := -2
  let y : ℝ := 1
  ((x + 2*y) * (x - 2*y) + 4 * (x - y)^2) / (-x) = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3551_355194


namespace NUMINAMATH_CALUDE_chess_game_probabilities_l3551_355129

-- Define the probabilities
def prob_draw : ℚ := 1/2
def prob_B_win : ℚ := 1/3

-- Define the statements to be proven
def prob_A_win : ℚ := 1 - prob_draw - prob_B_win
def prob_A_not_lose : ℚ := prob_draw + prob_A_win
def prob_B_lose : ℚ := prob_A_win
def prob_B_not_lose : ℚ := prob_draw + prob_B_win

-- Theorem to prove the statements
theorem chess_game_probabilities :
  (prob_A_win = 1/6) ∧
  (prob_A_not_lose = 2/3) ∧
  (prob_B_lose = 1/6) ∧
  (prob_B_not_lose = 5/6) :=
by sorry

end NUMINAMATH_CALUDE_chess_game_probabilities_l3551_355129


namespace NUMINAMATH_CALUDE_circle_radius_l3551_355164

/-- Given a circle with center (0,k) where k > 5, which is tangent to the lines y=2x, y=-2x, and y=5,
    the radius of the circle is (k-5)/√5. -/
theorem circle_radius (k : ℝ) (h : k > 5) : ∃ r : ℝ,
  r > 0 ∧
  r = (k - 5) / Real.sqrt 5 ∧
  (∀ x y : ℝ, (x = 0 ∧ y = k) → (x^2 + (y - k)^2 = r^2)) ∧
  (∃ x y : ℝ, y = 2*x ∧ x^2 + (y - k)^2 = r^2) ∧
  (∃ x y : ℝ, y = -2*x ∧ x^2 + (y - k)^2 = r^2) ∧
  (∃ x : ℝ, x^2 + (5 - k)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l3551_355164


namespace NUMINAMATH_CALUDE_peyton_juice_boxes_l3551_355183

/-- Calculate the total number of juice boxes needed for Peyton's children for the school year -/
def total_juice_boxes (num_children : ℕ) (school_days_per_week : ℕ) (weeks_in_school_year : ℕ) : ℕ :=
  num_children * school_days_per_week * weeks_in_school_year

/-- Proof that Peyton needs 375 juice boxes for the entire school year for all of her children -/
theorem peyton_juice_boxes :
  total_juice_boxes 3 5 25 = 375 := by
  sorry

end NUMINAMATH_CALUDE_peyton_juice_boxes_l3551_355183


namespace NUMINAMATH_CALUDE_exists_x_y_inequality_l3551_355143

theorem exists_x_y_inequality (f : ℝ → ℝ) : ∃ x y : ℝ, f (x - f y) > y * f x + x := by
  sorry

end NUMINAMATH_CALUDE_exists_x_y_inequality_l3551_355143


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3551_355185

-- Define the function type
def FunctionType := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (c : ℝ) (h_c : c > 1) (f : FunctionType) 
  (h_f : ∀ x y : ℝ, f (x + y) = f x * f y - c * Real.sin x * Real.sin y) :
  (∀ t : ℝ, f t = Real.sqrt (c - 1) * Real.sin t + Real.cos t) ∨ 
  (∀ t : ℝ, f t = -Real.sqrt (c - 1) * Real.sin t + Real.cos t) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3551_355185


namespace NUMINAMATH_CALUDE_correct_order_count_l3551_355132

/-- Represents the number of letters in the original stack -/
def n : ℕ := 10

/-- Represents the position of the letter known to be typed -/
def k : ℕ := 9

/-- Calculates the number of possible typing orders for the remaining letters -/
def possibleOrders : ℕ := 
  (List.range (k - 1)).foldl (fun acc i => acc + (Nat.choose (k - 1) i) * (i + 2)) 0

/-- Theorem stating the correct number of possible typing orders -/
theorem correct_order_count : possibleOrders = 1536 := by
  sorry

end NUMINAMATH_CALUDE_correct_order_count_l3551_355132


namespace NUMINAMATH_CALUDE_k_range_when_f_less_than_bound_l3551_355153

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + (1-k)*x - k * Real.log x

theorem k_range_when_f_less_than_bound (k : ℝ) (h_k_pos : k > 0) :
  (∃ x₀ : ℝ, f k x₀ < 3/2 - k^2) → 0 < k ∧ k < 1 := by sorry

end NUMINAMATH_CALUDE_k_range_when_f_less_than_bound_l3551_355153


namespace NUMINAMATH_CALUDE_sum_of_altitudes_equals_2432_div_17_l3551_355184

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 15 * x + 8 * y = 120

-- Define the triangle formed by the line and coordinate axes
def triangle : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ line_equation p.1 p.2}

-- Define the function to calculate the sum of altitudes
def sum_of_altitudes (t : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem sum_of_altitudes_equals_2432_div_17 :
  sum_of_altitudes triangle = 2432 / 17 := by sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_equals_2432_div_17_l3551_355184


namespace NUMINAMATH_CALUDE_shelves_used_l3551_355147

def initial_stock : ℕ := 40
def books_sold : ℕ := 20
def books_per_shelf : ℕ := 4

theorem shelves_used : (initial_stock - books_sold) / books_per_shelf = 5 :=
by sorry

end NUMINAMATH_CALUDE_shelves_used_l3551_355147


namespace NUMINAMATH_CALUDE_population_increase_rate_is_two_l3551_355170

/-- The rate of population increase in persons per minute, given that one person is added every 30 seconds. -/
def population_increase_rate (seconds_per_person : ℕ) : ℚ :=
  60 / seconds_per_person

/-- Theorem stating that if the population increases by one person every 30 seconds, 
    then the rate of population increase is 2 persons per minute. -/
theorem population_increase_rate_is_two :
  population_increase_rate 30 = 2 := by sorry

end NUMINAMATH_CALUDE_population_increase_rate_is_two_l3551_355170


namespace NUMINAMATH_CALUDE_largest_valid_number_l3551_355165

def is_valid (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p ∣ n → (p^2 - 1) ∣ n

theorem largest_valid_number : 
  (1944 < 2012) ∧ 
  is_valid 1944 ∧ 
  (∀ m : ℕ, 1944 < m → m < 2012 → ¬ is_valid m) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3551_355165


namespace NUMINAMATH_CALUDE_probability_two_black_balls_l3551_355190

/-- Probability of drawing two black balls without replacement -/
theorem probability_two_black_balls 
  (white : ℕ) 
  (black : ℕ) 
  (h1 : white = 7) 
  (h2 : black = 8) : 
  (black * (black - 1)) / ((white + black) * (white + black - 1)) = 4 / 15 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_black_balls_l3551_355190


namespace NUMINAMATH_CALUDE_range_of_a_l3551_355122

open Set Real

def A : Set ℝ := {x | 1 ≤ x ∧ x < 3}

def B (a : ℝ) : Set ℝ := {x | x^2 - a*x ≤ x - a}

theorem range_of_a :
  ∀ a : ℝ, (B a ⊆ A) ↔ (1 ≤ a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3551_355122


namespace NUMINAMATH_CALUDE_sequence_general_term_l3551_355157

theorem sequence_general_term (a : ℕ → ℕ) :
  a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) = a n + 2 * n + 1) →
  ∀ n : ℕ, n ≥ 1 → a n = n^2 := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3551_355157


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l3551_355196

/-- Represents different sampling methods --/
inductive SamplingMethod
  | Lottery
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a school population --/
structure SchoolPopulation where
  male_students : Nat
  female_students : Nat

/-- Represents a survey plan --/
structure SurveyPlan where
  population : SchoolPopulation
  sample_size : Nat
  goal : String

/-- Determines the most appropriate sampling method for a given survey plan --/
def most_appropriate_sampling_method (plan : SurveyPlan) : SamplingMethod :=
  sorry

/-- The theorem stating that stratified sampling is most appropriate for the given scenario --/
theorem stratified_sampling_most_appropriate (plan : SurveyPlan) :
  plan.population.male_students = 500 →
  plan.population.female_students = 500 →
  plan.sample_size = 100 →
  plan.goal = "investigate differences in study interests and hobbies between male and female students" →
  most_appropriate_sampling_method plan = SamplingMethod.Stratified :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l3551_355196


namespace NUMINAMATH_CALUDE_rectangle_area_l3551_355108

/-- The area of a rectangle with sides 1.5 meters and 0.75 meters is 1.125 square meters. -/
theorem rectangle_area : 
  let length : ℝ := 1.5
  let width : ℝ := 0.75
  length * width = 1.125 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3551_355108


namespace NUMINAMATH_CALUDE_car_race_distance_l3551_355134

theorem car_race_distance (karen_speed tom_speed : ℝ) (karen_delay : ℝ) (win_margin : ℝ) : 
  karen_speed = 75 →
  tom_speed = 50 →
  karen_delay = 7 / 60 →
  win_margin = 5 →
  (karen_speed * (tom_speed * win_margin / (karen_speed - tom_speed) + karen_delay) - 
   tom_speed * (tom_speed * win_margin / (karen_speed - tom_speed) + karen_delay)) = win_margin →
  tom_speed * (tom_speed * win_margin / (karen_speed - tom_speed) + karen_delay) = 27.5 :=
by sorry

end NUMINAMATH_CALUDE_car_race_distance_l3551_355134


namespace NUMINAMATH_CALUDE_condition_implies_linear_l3551_355127

/-- A function satisfying the given inequality condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ (a b p : ℝ), f (p * a + (1 - p) * b) ≤ p * f a + (1 - p) * f b

/-- A linear function -/
def IsLinear (f : ℝ → ℝ) : Prop :=
  ∃ (A B : ℝ), ∀ x, f x = A * x + B

/-- Theorem: If a function satisfies the condition, then it is linear -/
theorem condition_implies_linear (f : ℝ → ℝ) :
  SatisfiesCondition f → IsLinear f := by
  sorry

end NUMINAMATH_CALUDE_condition_implies_linear_l3551_355127


namespace NUMINAMATH_CALUDE_sequence_length_l3551_355155

/-- Given a sequence of real numbers satisfying specific conditions, prove that the length of the sequence is 455. -/
theorem sequence_length : ∃ (n : ℕ) (b : ℕ → ℝ), 
  n > 0 ∧ 
  b 0 = 28 ∧ 
  b 1 = 81 ∧ 
  b n = 0 ∧ 
  (∀ j ∈ Finset.range (n - 1), b (j + 2) = b j - 5 / b (j + 1)) ∧
  (∀ m : ℕ, m < n → 
    m > 0 → 
    b m ≠ 0 → 
    ¬(b 0 = 28 ∧ 
      b 1 = 81 ∧ 
      b m = 0 ∧ 
      (∀ j ∈ Finset.range (m - 1), b (j + 2) = b j - 5 / b (j + 1)))) ∧
  n = 455 :=
sorry

end NUMINAMATH_CALUDE_sequence_length_l3551_355155


namespace NUMINAMATH_CALUDE_min_value_theorem_l3551_355125

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b)/c + (2*a + c)/b + (2*b + c)/a ≥ 6 ∧
  ((2*a + b)/c + (2*a + c)/b + (2*b + c)/a = 6 ↔ 2*a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3551_355125


namespace NUMINAMATH_CALUDE_profit_threshold_l3551_355189

/-- Represents the minimum number of workers needed for profit -/
def min_workers_for_profit (
  daily_maintenance : ℕ)
  (hourly_wage : ℕ)
  (gadgets_per_hour : ℕ)
  (gadget_price : ℕ)
  (workday_hours : ℕ) : ℕ :=
  16

theorem profit_threshold (
  daily_maintenance : ℕ)
  (hourly_wage : ℕ)
  (gadgets_per_hour : ℕ)
  (gadget_price : ℕ)
  (workday_hours : ℕ)
  (h1 : daily_maintenance = 600)
  (h2 : hourly_wage = 20)
  (h3 : gadgets_per_hour = 6)
  (h4 : gadget_price = 4)
  (h5 : workday_hours = 10) :
  ∀ n : ℕ, n ≥ min_workers_for_profit daily_maintenance hourly_wage gadgets_per_hour gadget_price workday_hours →
    n * workday_hours * gadgets_per_hour * gadget_price > daily_maintenance + n * workday_hours * hourly_wage :=
by sorry

#check profit_threshold

end NUMINAMATH_CALUDE_profit_threshold_l3551_355189


namespace NUMINAMATH_CALUDE_number_problem_l3551_355133

theorem number_problem (x : ℝ) : 0.2 * x = 0.3 * 120 + 80 → x = 580 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3551_355133


namespace NUMINAMATH_CALUDE_fable_village_impossible_total_l3551_355171

theorem fable_village_impossible_total (p h s c d k : ℕ) : 
  p = 4 * h ∧ 
  s = 5 * c ∧ 
  d = 2 * p ∧ 
  k = 2 * d → 
  p + h + s + c + d + k ≠ 90 :=
by sorry

end NUMINAMATH_CALUDE_fable_village_impossible_total_l3551_355171


namespace NUMINAMATH_CALUDE_smallest_n_value_l3551_355178

theorem smallest_n_value (o y v : ℝ) (ho : o > 0) (hy : y > 0) (hv : v > 0) :
  let n := Nat.lcm (Nat.lcm 10 16) 18 / 24
  ∀ m : ℕ, m > 0 → (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 10 * a = 16 * b ∧ 16 * b = 18 * c ∧ 18 * c = 24 * m) →
  m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l3551_355178


namespace NUMINAMATH_CALUDE_divisibility_property_l3551_355120

theorem divisibility_property (a b n : ℕ) (h : a^n ∣ b) : a^(n+1) ∣ (a+1)^b - 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3551_355120


namespace NUMINAMATH_CALUDE_baker_bread_rolls_l3551_355112

theorem baker_bread_rolls (regular_rolls : ℕ) (regular_flour : ℚ) 
  (new_rolls : ℕ) (new_flour : ℚ) :
  regular_rolls = 40 →
  regular_flour = 1 / 8 →
  new_rolls = 25 →
  regular_rolls * regular_flour = new_rolls * new_flour →
  new_flour = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_baker_bread_rolls_l3551_355112


namespace NUMINAMATH_CALUDE_largest_square_area_l3551_355159

theorem largest_square_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2)
  (h_sum_areas : a^2 + b^2 + c^2 = 450) : c^2 = 225 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l3551_355159


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3551_355116

theorem trigonometric_product_equals_one :
  let α : Real := 15 * π / 180  -- 15 degrees in radians
  (1 - 1 / Real.cos α) * (1 + 1 / Real.sin (π/2 - α)) *
  (1 - 1 / Real.sin α) * (1 + 1 / Real.cos (π/2 - α)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3551_355116


namespace NUMINAMATH_CALUDE_equation_is_hyperbola_l3551_355131

/-- Represents a conic section --/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- Determines the type of conic section for the given equation --/
def determineConicSection (a b c d e f : ℝ) : ConicSection :=
  sorry

/-- The equation x^2 - 25y^2 - 10x + 50 = 0 represents a hyperbola --/
theorem equation_is_hyperbola :
  determineConicSection 1 (-25) 0 (-10) 0 50 = ConicSection.Hyperbola :=
sorry

end NUMINAMATH_CALUDE_equation_is_hyperbola_l3551_355131


namespace NUMINAMATH_CALUDE_fold_crease_forms_ellipse_l3551_355181

/-- Given a circle with radius R centered at the origin and an internal point A at (a, 0),
    the set of all points P(x, y) that are equidistant from A and any point on the circle's circumference
    forms an ellipse. -/
theorem fold_crease_forms_ellipse (R a : ℝ) (h : 0 < a ∧ a < R) :
  ∀ x y : ℝ,
    (∃ α : ℝ, (x - R * Real.cos α)^2 + (y - R * Real.sin α)^2 = (x - a)^2 + y^2) ↔
    (2*x - a)^2 / R^2 + 4*y^2 / (R^2 - a^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fold_crease_forms_ellipse_l3551_355181


namespace NUMINAMATH_CALUDE_complex_division_equality_l3551_355102

theorem complex_division_equality : (2 - I) / (2 + I) = 3/5 - 4/5 * I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l3551_355102


namespace NUMINAMATH_CALUDE_number_problem_l3551_355168

theorem number_problem (x : ℚ) (h : x - (3/5) * x = 62) : x = 155 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3551_355168


namespace NUMINAMATH_CALUDE_rectangular_formation_perimeter_l3551_355152

theorem rectangular_formation_perimeter (area : ℝ) (num_squares : ℕ) :
  area = 512 →
  num_squares = 8 →
  let square_side : ℝ := Real.sqrt (area / num_squares)
  let perimeter : ℝ := 2 * (4 * square_side + 3 * square_side)
  perimeter = 152 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_formation_perimeter_l3551_355152


namespace NUMINAMATH_CALUDE_figure_102_squares_l3551_355144

/-- A function representing the number of non-overlapping unit squares in the nth figure -/
def g (n : ℕ) : ℕ := 2 * n^2 - 2 * n + 1

/-- Theorem stating that the 102nd figure contains 20605 non-overlapping unit squares -/
theorem figure_102_squares : g 102 = 20605 := by
  sorry

/-- Lemma verifying the given initial conditions -/
lemma initial_conditions :
  g 1 = 1 ∧ g 2 = 5 ∧ g 3 = 13 ∧ g 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_figure_102_squares_l3551_355144


namespace NUMINAMATH_CALUDE_Mp_not_perfect_square_l3551_355106

/-- A prime number p congruent to 3 modulo 4 -/
def p : ℕ := sorry

/-- Assumption that p is prime -/
axiom p_prime : Nat.Prime p

/-- Assumption that p is congruent to 3 modulo 4 -/
axiom p_mod_4 : p % 4 = 3

/-- Definition of a balanced sequence -/
def BalancedSequence (seq : List ℤ) : Prop :=
  (∀ x ∈ seq, ∃ y ∈ seq, x = -y) ∧
  (∀ x ∈ seq, |x| ≤ (p - 1) / 2) ∧
  (seq.length ≤ p - 1)

/-- The number of balanced sequences for prime p -/
def Mp : ℕ := sorry

/-- Theorem: Mp is not a perfect square -/
theorem Mp_not_perfect_square : ¬ ∃ (n : ℕ), Mp = n ^ 2 := by sorry

end NUMINAMATH_CALUDE_Mp_not_perfect_square_l3551_355106


namespace NUMINAMATH_CALUDE_converse_proposition_l3551_355123

theorem converse_proposition : 
  (∀ x : ℝ, x > 0 → x^2 - 1 > 0) ↔ 
  (∀ x : ℝ, x^2 - 1 > 0 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_converse_proposition_l3551_355123


namespace NUMINAMATH_CALUDE_expected_balls_in_original_position_l3551_355172

/-- The number of balls arranged in a circle -/
def n : ℕ := 7

/-- The probability of a ball being swapped twice -/
def p_twice : ℚ := 2 / (n * n)

/-- The probability of a ball never being swapped -/
def p_never : ℚ := (n - 2)^2 / (n * n)

/-- The probability of a ball being in its original position after two transpositions -/
def p_original : ℚ := p_twice + p_never

/-- The expected number of balls in their original positions after two transpositions -/
def expected_original : ℚ := n * p_original

theorem expected_balls_in_original_position :
  expected_original = 189 / 49 := by sorry

end NUMINAMATH_CALUDE_expected_balls_in_original_position_l3551_355172


namespace NUMINAMATH_CALUDE_problem_triangle_count_l3551_355174

/-- Represents a rectangle subdivided into sections with diagonal lines -/
structure SubdividedRectangle where
  vertical_sections : Nat
  horizontal_sections : Nat
  has_diagonals : Bool

/-- Counts the number of triangles in a subdivided rectangle -/
def count_triangles (rect : SubdividedRectangle) : Nat :=
  sorry

/-- The specific rectangle from the problem -/
def problem_rectangle : SubdividedRectangle :=
  { vertical_sections := 4
  , horizontal_sections := 2
  , has_diagonals := true }

/-- Theorem stating that the number of triangles in the problem rectangle is 42 -/
theorem problem_triangle_count : count_triangles problem_rectangle = 42 := by
  sorry

end NUMINAMATH_CALUDE_problem_triangle_count_l3551_355174


namespace NUMINAMATH_CALUDE_melody_reading_pages_l3551_355186

def english_pages : ℕ := 20
def science_pages : ℕ := 16
def civics_pages : ℕ := 8
def total_pages_tomorrow : ℕ := 14

def chinese_pages : ℕ := 12

theorem melody_reading_pages : 
  (english_pages / 4 + science_pages / 4 + civics_pages / 4 + chinese_pages / 4 = total_pages_tomorrow) ∧
  (chinese_pages ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_melody_reading_pages_l3551_355186


namespace NUMINAMATH_CALUDE_six_digit_number_remainder_l3551_355160

/-- Represents a 6-digit number in the form 6x62y4 -/
def SixDigitNumber (x y : Nat) : Nat :=
  600000 + 10000 * x + 6200 + 10 * y + 4

theorem six_digit_number_remainder (x y : Nat) :
  x < 10 → y < 10 →
  (SixDigitNumber x y) % 11 = 0 →
  (SixDigitNumber x y) % 9 = 6 →
  (SixDigitNumber x y) % 13 = 6 := by
sorry

end NUMINAMATH_CALUDE_six_digit_number_remainder_l3551_355160


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3551_355167

/-- Represents a quadratic equation of the form x^2 - (m-3)x - m = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 - (m-3)*x - m = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (m-3)^2 - 4*(-m)

/-- Represents the condition on the roots of the quadratic equation -/
def root_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^2 + x₂^2 - x₁*x₂ = 13

theorem quadratic_equation_properties (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂) ∧
  (∀ x₁ x₂ : ℝ, quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ root_condition x₁ x₂ →
    m = 4 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3551_355167


namespace NUMINAMATH_CALUDE_parabola_equation_l3551_355138

-- Define the parabola structure
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the focus of a parabola
def Focus := ℝ × ℝ

-- Define the line x - y + 4 = 0
def LineEquation (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the condition that the focus is on the line
def FocusOnLine (f : Focus) : Prop := LineEquation f.1 f.2

-- Define the condition that the vertex is at the origin
def VertexAtOrigin (p : Parabola) : Prop := p.equation 0 0

-- Define the condition that the axis of symmetry is one of the coordinate axes
def AxisIsCoordinateAxis (p : Parabola) : Prop :=
  (∀ x y : ℝ, p.equation x y ↔ p.equation x (-y)) ∨
  (∀ x y : ℝ, p.equation x y ↔ p.equation (-x) y)

-- Theorem statement
theorem parabola_equation (p : Parabola) (f : Focus) :
  VertexAtOrigin p →
  AxisIsCoordinateAxis p →
  FocusOnLine f →
  (∀ x y : ℝ, p.equation x y ↔ y^2 = -16*x) ∨
  (∀ x y : ℝ, p.equation x y ↔ x^2 = 16*y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3551_355138


namespace NUMINAMATH_CALUDE_c_neq_zero_necessary_not_sufficient_l3551_355128

/-- Represents a conic section defined by the equation ax² + y² = c -/
structure ConicSection where
  a : ℝ
  c : ℝ

/-- Predicate to determine if a conic section is an ellipse or hyperbola -/
def is_ellipse_or_hyperbola (conic : ConicSection) : Prop :=
  sorry

/-- Theorem stating that c ≠ 0 is necessary but not sufficient for
    ax² + y² = c to represent an ellipse or hyperbola -/
theorem c_neq_zero_necessary_not_sufficient :
  (∀ conic : ConicSection, is_ellipse_or_hyperbola conic → conic.c ≠ 0) ∧
  (∃ conic : ConicSection, conic.c ≠ 0 ∧ ¬is_ellipse_or_hyperbola conic) :=
sorry

end NUMINAMATH_CALUDE_c_neq_zero_necessary_not_sufficient_l3551_355128


namespace NUMINAMATH_CALUDE_expression_arrangements_l3551_355101

/-- Given three distinct real numbers, there are 96 possible ways to arrange
    the eight expressions ±x ±y ±z in increasing order. -/
theorem expression_arrangements (x y z : ℝ) (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) :
  (Set.ncard {l : List ℝ | 
    l.length = 8 ∧ 
    l.Nodup ∧
    (∀ a ∈ l, ∃ (s₁ s₂ s₃ : Bool), a = (if s₁ then x else -x) + (if s₂ then y else -y) + (if s₃ then z else -z)) ∧
    l.Sorted (· < ·)}) = 96 :=
by sorry

end NUMINAMATH_CALUDE_expression_arrangements_l3551_355101


namespace NUMINAMATH_CALUDE_congruence_solution_l3551_355126

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ -250 ≡ n [ZMOD 31] ∧ n = 29 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l3551_355126


namespace NUMINAMATH_CALUDE_volume_cylindrical_wedge_with_cap_l3551_355199

/-- The volume of a solid composed of a cylindrical wedge and a conical cap -/
theorem volume_cylindrical_wedge_with_cap (d : ℝ) (h : d = 16) :
  let r := d / 2
  let wedge_volume := (π * r^2 * d) / 2
  let cone_volume := (1/3) * π * r^2 * d
  wedge_volume + cone_volume = (2560/3) * π := by
  sorry

end NUMINAMATH_CALUDE_volume_cylindrical_wedge_with_cap_l3551_355199


namespace NUMINAMATH_CALUDE_min_value_fraction_l3551_355154

theorem min_value_fraction (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (m : ℝ), m = 1 - Real.sqrt 2 ∧ ∀ (z : ℝ), z = x*y/(x+y-2) → m ≤ z :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3551_355154


namespace NUMINAMATH_CALUDE_quadratic_equation_iff_m_eq_neg_one_l3551_355182

/-- The equation is quadratic if and only if m = -1 -/
theorem quadratic_equation_iff_m_eq_neg_one (m : ℝ) : 
  (∀ x, (m - 1) * x^(m^2 + 1) - x - 2 = 0 ↔ ∃ a b c, a ≠ 0 ∧ a * x^2 + b * x + c = 0) ↔ 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_iff_m_eq_neg_one_l3551_355182


namespace NUMINAMATH_CALUDE_fraction_sum_difference_equals_half_l3551_355110

theorem fraction_sum_difference_equals_half : 
  (3 : ℚ) / 9 + 5 / 12 - 1 / 4 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_equals_half_l3551_355110


namespace NUMINAMATH_CALUDE_binary_1101_equals_base5_23_l3551_355141

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert decimal to base-5
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

-- Theorem statement
theorem binary_1101_equals_base5_23 :
  decimal_to_base5 (binary_to_decimal [true, false, true, true]) = [2, 3] := by
  sorry

#eval binary_to_decimal [true, false, true, true]
#eval decimal_to_base5 13

end NUMINAMATH_CALUDE_binary_1101_equals_base5_23_l3551_355141


namespace NUMINAMATH_CALUDE_car_hire_payment_l3551_355117

/-- Represents the car hiring scenario -/
structure CarHire where
  hours_a : ℕ
  hours_b : ℕ
  hours_c : ℕ
  payment_b : ℚ

/-- Calculates the total amount paid for hiring the car -/
def total_payment (hire : CarHire) : ℚ :=
  let rate := hire.payment_b / hire.hours_b
  rate * (hire.hours_a + hire.hours_b + hire.hours_c)

/-- Theorem stating the total payment for the given scenario -/
theorem car_hire_payment :
  ∀ (hire : CarHire),
    hire.hours_a = 9 ∧
    hire.hours_b = 10 ∧
    hire.hours_c = 13 ∧
    hire.payment_b = 225 →
    total_payment hire = 720 := by
  sorry


end NUMINAMATH_CALUDE_car_hire_payment_l3551_355117


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3551_355111

def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {2, 3, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3551_355111


namespace NUMINAMATH_CALUDE_total_points_is_1390_l3551_355198

-- Define the points scored in each try
def first_try : ℕ := 400
def second_try : ℕ := first_try - 70
def third_try : ℕ := 2 * second_try

-- Define the total points
def total_points : ℕ := first_try + second_try + third_try

-- Theorem statement
theorem total_points_is_1390 : total_points = 1390 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_1390_l3551_355198


namespace NUMINAMATH_CALUDE_max_value_of_f_l3551_355121

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 2 + 2 * x) - 5 * Real.sin x

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3551_355121


namespace NUMINAMATH_CALUDE_divisibility_by_primes_less_than_1966_l3551_355180

theorem divisibility_by_primes_less_than_1966 (n : ℕ) (p : ℕ) (hp : Prime p) (hp_bound : p < 1966) :
  p ∣ (List.range 1966).foldl (λ acc i => acc * ((i + 1) * n + 1)) n :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_primes_less_than_1966_l3551_355180


namespace NUMINAMATH_CALUDE_remainder_of_product_mod_12_l3551_355187

theorem remainder_of_product_mod_12 : (1425 * 1427 * 1429) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_mod_12_l3551_355187


namespace NUMINAMATH_CALUDE_other_diagonal_length_l3551_355163

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  area : ℝ

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.diagonal1 * r.diagonal2) / 2

theorem other_diagonal_length :
  ∀ r : Rhombus, r.diagonal1 = 12 ∧ r.area = 60 → r.diagonal2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l3551_355163


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3551_355104

/-- The functional equation satisfied by f -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x ≠ 0 → y ≠ 0 → z ≠ 0 → x * y * z = 1 →
    f x ^ 2 - f y * f z = x * (x + y + z) * (f x + f y + f z)

/-- The theorem stating the possible forms of f -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f x = f x) →
    satisfies_equation f →
    (∀ x : ℝ, x ≠ 0 → f x = x^2 - 1/x) ∨ (∀ x : ℝ, x ≠ 0 → f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3551_355104


namespace NUMINAMATH_CALUDE_square_sum_equals_three_l3551_355100

theorem square_sum_equals_three (x y z : ℝ) 
  (h1 : x - y - z = 3) 
  (h2 : y * z - x * y - x * z = 3) : 
  x^2 + y^2 + z^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_three_l3551_355100


namespace NUMINAMATH_CALUDE_unique_base_for_1024_l3551_355135

theorem unique_base_for_1024 : ∃! b : ℕ, 4 ≤ b ∧ b ≤ 12 ∧ 1024 % b = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_for_1024_l3551_355135


namespace NUMINAMATH_CALUDE_three_numbers_solution_l3551_355118

theorem three_numbers_solution :
  ∃ (x y z : ℤ),
    (x + y) * z = 35 ∧
    (x + z) * y = -27 ∧
    (y + z) * x = -32 ∧
    x = 4 ∧ y = -3 ∧ z = 5 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_solution_l3551_355118


namespace NUMINAMATH_CALUDE_spider_eyes_l3551_355137

theorem spider_eyes (spider_count : ℕ) (ant_count : ℕ) (ant_eyes : ℕ) (total_eyes : ℕ) :
  spider_count = 3 →
  ant_count = 50 →
  ant_eyes = 2 →
  total_eyes = 124 →
  total_eyes = spider_count * (total_eyes - ant_count * ant_eyes) / spider_count →
  (total_eyes - ant_count * ant_eyes) / spider_count = 8 :=
by sorry

end NUMINAMATH_CALUDE_spider_eyes_l3551_355137


namespace NUMINAMATH_CALUDE_two_recess_breaks_l3551_355166

/-- Calculates the number of 15-minute recess breaks given the total time outside class,
    lunch duration, and additional recess duration. -/
def numberOfRecessBreaks (totalTimeOutside lunchDuration additionalRecessDuration : ℕ) : ℕ :=
  ((totalTimeOutside - lunchDuration - additionalRecessDuration) / 15)

/-- Proves that given the specified conditions, students get 2 fifteen-minute recess breaks. -/
theorem two_recess_breaks :
  let totalTimeOutside : ℕ := 80
  let lunchDuration : ℕ := 30
  let additionalRecessDuration : ℕ := 20
  numberOfRecessBreaks totalTimeOutside lunchDuration additionalRecessDuration = 2 := by
sorry


end NUMINAMATH_CALUDE_two_recess_breaks_l3551_355166


namespace NUMINAMATH_CALUDE_dihedral_angle_range_l3551_355197

/-- The dihedral angle between two adjacent faces in a regular n-sided polyhedron -/
def dihedralAngle (n : ℕ) (θ : ℝ) : Prop :=
  n ≥ 3 ∧ ((n - 2 : ℝ) / n) * Real.pi < θ ∧ θ < Real.pi

/-- Theorem stating the range of the dihedral angle in a regular n-sided polyhedron -/
theorem dihedral_angle_range (n : ℕ) :
  ∃ θ : ℝ, dihedralAngle n θ :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_range_l3551_355197


namespace NUMINAMATH_CALUDE_f_less_than_g_iff_m_in_range_l3551_355109

-- Define the functions f and g
def f (x m : ℝ) : ℝ := |x - 1| + |x + m|
def g (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem
theorem f_less_than_g_iff_m_in_range :
  ∀ m : ℝ, (∀ x ∈ Set.Icc (-m) 1, f x m < g x) ↔ -1 < m ∧ m < -2/3 := by sorry

end NUMINAMATH_CALUDE_f_less_than_g_iff_m_in_range_l3551_355109


namespace NUMINAMATH_CALUDE_percent_of_percent_equality_l3551_355161

theorem percent_of_percent_equality (y : ℝ) : (0.3 * (0.6 * y)) = (0.18 * y) := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_equality_l3551_355161


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l3551_355179

theorem min_value_expression (x y : ℝ) : 
  x^2 + 4*x*Real.sin y - 4*(Real.cos y)^2 ≥ -4 :=
by sorry

theorem min_value_attainable : 
  ∃ (x y : ℝ), x^2 + 4*x*Real.sin y - 4*(Real.cos y)^2 = -4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l3551_355179


namespace NUMINAMATH_CALUDE_closest_whole_number_to_ratio_l3551_355146

theorem closest_whole_number_to_ratio : ∃ n : ℕ, 
  n = 9 ∧ 
  ∀ m : ℕ, 
    |((10^3000 : ℝ) + 10^3003) / ((10^3001 : ℝ) + 10^3002) - (n : ℝ)| ≤ 
    |((10^3000 : ℝ) + 10^3003) / ((10^3001 : ℝ) + 10^3002) - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_closest_whole_number_to_ratio_l3551_355146


namespace NUMINAMATH_CALUDE_sqrt_36_div_6_l3551_355156

theorem sqrt_36_div_6 : Real.sqrt 36 / 6 = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_36_div_6_l3551_355156


namespace NUMINAMATH_CALUDE_necessary_condition_for_positive_linear_function_l3551_355193

theorem necessary_condition_for_positive_linear_function
  (a b : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = a * x + b)
  (h_positive : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x > 0) :
  a + 2 * b > 0 :=
sorry

end NUMINAMATH_CALUDE_necessary_condition_for_positive_linear_function_l3551_355193


namespace NUMINAMATH_CALUDE_trailer_homes_problem_l3551_355162

theorem trailer_homes_problem (initial_homes : ℕ) (initial_avg_age : ℕ) 
  (current_avg_age : ℕ) (years_passed : ℕ) :
  initial_homes = 20 →
  initial_avg_age = 18 →
  current_avg_age = 14 →
  years_passed = 2 →
  ∃ (new_homes : ℕ),
    (initial_homes * (initial_avg_age + years_passed) + new_homes * years_passed) / 
    (initial_homes + new_homes) = current_avg_age ∧
    new_homes = 10 := by
  sorry

end NUMINAMATH_CALUDE_trailer_homes_problem_l3551_355162


namespace NUMINAMATH_CALUDE_pi_irrational_l3551_355142

def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem pi_irrational :
  is_rational (-4/7) →
  is_rational 3.333333 →
  is_rational 1.010010001 →
  ¬ is_rational Real.pi :=
by sorry

end NUMINAMATH_CALUDE_pi_irrational_l3551_355142


namespace NUMINAMATH_CALUDE_binomial_variance_problem_l3551_355173

-- Define the binomial distribution
def binomial_distribution (n : ℕ) (p : ℝ) : ℕ → ℝ := sorry

-- Define the probability mass function for ξ = 1
def prob_xi_equals_one (n : ℕ) : ℝ := binomial_distribution n (1/2) 1

-- Define the variance of the binomial distribution
def variance_binomial (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_variance_problem (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 8) 
  (h3 : prob_xi_equals_one n = 3/32) :
  variance_binomial n (1/2) = 3/2 := by sorry

end NUMINAMATH_CALUDE_binomial_variance_problem_l3551_355173


namespace NUMINAMATH_CALUDE_circle_line_distance_l3551_355175

theorem circle_line_distance (x y : ℝ) (a : ℝ) :
  (x^2 + y^2 - 2*x - 4*y = 0) →
  ((1 - y + a) / Real.sqrt 2 = Real.sqrt 2 / 2 ∨
   (-1 + y - a) / Real.sqrt 2 = Real.sqrt 2 / 2) →
  (a = 0 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_circle_line_distance_l3551_355175


namespace NUMINAMATH_CALUDE_value_of_a_l3551_355177

/-- Proves that if 0.5% of a equals 70 paise, then a equals 140 rupees. -/
theorem value_of_a (a : ℝ) : (0.5 / 100) * a = 70 / 100 → a = 140 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3551_355177


namespace NUMINAMATH_CALUDE_expression_simplification_l3551_355149

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3 * x^2 - 4*x + 1) / ((x - 1) * (x + 3)) - (6*x - 5) / ((x - 1) * (x + 3)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3551_355149


namespace NUMINAMATH_CALUDE_pipe_fill_time_l3551_355119

/-- Time to fill tank with leak (in hours) -/
def time_with_leak : ℝ := 15

/-- Time for leak to empty full tank (in hours) -/
def time_leak_empty : ℝ := 30

/-- Time to fill tank without leak (in hours) -/
def time_without_leak : ℝ := 10

theorem pipe_fill_time :
  (1 / time_without_leak) - (1 / time_leak_empty) = (1 / time_with_leak) :=
sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l3551_355119


namespace NUMINAMATH_CALUDE_average_weight_problem_l3551_355151

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 43 →
  (b + c) / 2 = 42 →
  b = 51 →
  (a + b) / 2 = 48 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l3551_355151


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3551_355150

/-- Given a triangle DEF where ∠E is congruent to ∠F, the measure of ∠F is three times 
    the measure of ∠D, and ∠D is one-third the measure of ∠E, 
    prove that the measure of ∠E is 540/7 degrees. -/
theorem triangle_angle_measure (D E F : ℝ) : 
  D > 0 → E > 0 → F > 0 →  -- Angles are positive
  D + E + F = 180 →  -- Sum of angles in a triangle
  E = F →  -- ∠E is congruent to ∠F
  F = 3 * D →  -- Measure of ∠F is three times the measure of ∠D
  D = E / 3 →  -- ∠D is one-third the measure of ∠E
  E = 540 / 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3551_355150


namespace NUMINAMATH_CALUDE_parabola_with_conditions_l3551_355124

/-- A parabola passing through specific points with a specific tangent line -/
theorem parabola_with_conditions (a b c : ℝ) :
  (1 : ℝ)^2 * a + b * 1 + c = 1 →  -- Parabola passes through (1, 1)
  (2 : ℝ)^2 * a + b * 2 + c = -1 →  -- Parabola passes through (2, -1)
  2 * a * 2 + b = 1 →  -- Tangent line at (2, -1) is parallel to y = x - 3
  a = 3 ∧ b = -11 ∧ c = 9 := by
sorry

end NUMINAMATH_CALUDE_parabola_with_conditions_l3551_355124


namespace NUMINAMATH_CALUDE_sticker_count_l3551_355113

def ryan_stickers : ℕ := 30

def steven_stickers (ryan : ℕ) : ℕ := 3 * ryan

def terry_stickers (steven : ℕ) : ℕ := steven + 20

def total_stickers (ryan steven terry : ℕ) : ℕ := ryan + steven + terry

theorem sticker_count :
  total_stickers ryan_stickers (steven_stickers ryan_stickers) (terry_stickers (steven_stickers ryan_stickers)) = 230 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_l3551_355113


namespace NUMINAMATH_CALUDE_second_butcher_delivery_l3551_355148

/-- Represents the number of packages delivered by each butcher -/
structure ButcherDelivery where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the weight of each package and the total weight delivered -/
structure DeliveryInfo where
  package_weight : ℕ
  total_weight : ℕ

/-- Given the delivery information and the number of packages from the first and third butchers,
    proves that the second butcher delivered 7 packages -/
theorem second_butcher_delivery 
  (delivery : ButcherDelivery)
  (info : DeliveryInfo)
  (h1 : delivery.first = 10)
  (h2 : delivery.third = 8)
  (h3 : info.package_weight = 4)
  (h4 : info.total_weight = 100)
  (h5 : info.total_weight = 
    (delivery.first + delivery.second + delivery.third) * info.package_weight) :
  delivery.second = 7 := by
  sorry


end NUMINAMATH_CALUDE_second_butcher_delivery_l3551_355148
