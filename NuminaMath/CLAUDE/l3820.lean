import Mathlib

namespace NUMINAMATH_CALUDE_service_period_problem_l3820_382096

/-- Represents the problem of determining the agreed-upon period of service --/
theorem service_period_problem (total_pay : ℕ) (uniform_price : ℕ) (partial_service : ℕ) (partial_pay : ℕ) :
  let full_compensation := total_pay + uniform_price
  let partial_compensation := partial_pay + uniform_price
  (partial_service : ℚ) / (12 : ℚ) = partial_compensation / full_compensation →
  12 = (partial_service * full_compensation) / partial_compensation :=
by
  sorry

#check service_period_problem 900 100 9 650

end NUMINAMATH_CALUDE_service_period_problem_l3820_382096


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_opposite_l3820_382086

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Black : Card
| Blue : Card
| White : Card

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "Person A gets the red card"
def event_A_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Person B gets the red card"
def event_B_red (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem events_mutually_exclusive_not_opposite :
  ∃ (d : Distribution),
    (∀ (p : Person), ∃! (c : Card), d p = c) ∧  -- Each person gets exactly one card
    (∀ (c : Card), ∃! (p : Person), d p = c) ∧  -- Each card is given to exactly one person
    (¬(event_A_red d ∧ event_B_red d)) ∧        -- Events are mutually exclusive
    ¬(event_A_red d ↔ ¬event_B_red d)           -- Events are not opposite
  := by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_opposite_l3820_382086


namespace NUMINAMATH_CALUDE_system_solution_existence_l3820_382059

theorem system_solution_existence (b : ℝ) : 
  (∃ a : ℝ, ∃ x y : ℝ, x = |y - b| + 3/b ∧ x^2 + y^2 + 32 = a*(2*y - a) + 12*x) ↔ 
  (b < 0 ∨ b ≥ 3/8) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l3820_382059


namespace NUMINAMATH_CALUDE_smallest_real_number_l3820_382054

theorem smallest_real_number (S : Set ℝ) (h : S = {-2, 2, 0, 1}) : 
  ∃ m ∈ S, ∀ x ∈ S, m ≤ x ∧ m = -2 := by sorry

end NUMINAMATH_CALUDE_smallest_real_number_l3820_382054


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_a_geq_neg_eight_l3820_382042

-- Define set A
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*x + a ≥ 0}

-- Theorem statement
theorem intersection_nonempty_iff_a_geq_neg_eight :
  ∀ a : ℝ, (A ∩ B a).Nonempty ↔ a ≥ -8 := by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_a_geq_neg_eight_l3820_382042


namespace NUMINAMATH_CALUDE_chess_tournament_players_l3820_382018

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- number of players not in the lowest 8
  /-- Total number of players is n + 8 -/
  total_players : ℕ := n + 8
  /-- Each player played exactly one game against each other player -/
  total_games : ℕ := (total_players * (total_players - 1)) / 2
  /-- Points earned by n players against each other -/
  n_vs_n_points : ℕ := (n * (n - 1)) / 2
  /-- Points earned by n players against 8 lowest players -/
  n_vs_8_points : ℕ := n_vs_n_points
  /-- Points earned by 8 lowest players among themselves -/
  lowest_8_points : ℕ := 28
  /-- Total points in the tournament -/
  total_points : ℕ := 2 * n_vs_n_points + 2 * lowest_8_points

/-- The theorem stating that the total number of players is 22 -/
theorem chess_tournament_players : ∀ t : ChessTournament, t.total_players = 22 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l3820_382018


namespace NUMINAMATH_CALUDE_b_months_is_five_l3820_382028

/-- Represents the grazing arrangement for a pasture -/
structure GrazingArrangement where
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  c_oxen : ℕ
  c_months : ℕ
  total_rent : ℚ
  c_share : ℚ

/-- Calculates the number of months B's oxen grazed given a GrazingArrangement -/
def calculate_b_months (g : GrazingArrangement) : ℚ :=
  ((g.total_rent - g.c_share) * g.c_oxen * g.c_months - g.a_oxen * g.a_months * g.c_share) /
  (g.b_oxen * g.c_share)

/-- Theorem stating that B's oxen grazed for 5 months under the given conditions -/
theorem b_months_is_five (g : GrazingArrangement) 
    (h1 : g.a_oxen = 10) 
    (h2 : g.a_months = 7) 
    (h3 : g.b_oxen = 12) 
    (h4 : g.c_oxen = 15) 
    (h5 : g.c_months = 3) 
    (h6 : g.total_rent = 175) 
    (h7 : g.c_share = 45) : 
  calculate_b_months g = 5 := by
  sorry

end NUMINAMATH_CALUDE_b_months_is_five_l3820_382028


namespace NUMINAMATH_CALUDE_oil_price_reduction_l3820_382030

/-- Represents the price reduction problem for oil -/
def OilPriceReduction (original_price reduced_price : ℝ) : Prop :=
  reduced_price = 0.8 * original_price

/-- Represents the relationship between price and quantity before and after reduction -/
def QuantityIncrease (original_price reduced_price : ℝ) : Prop :=
  ∃ (original_quantity : ℝ),
    800 = original_quantity * original_price ∧
    800 = (original_quantity + 5) * reduced_price

theorem oil_price_reduction (original_price reduced_price : ℝ) 
  (h1 : OilPriceReduction original_price reduced_price)
  (h2 : QuantityIncrease original_price reduced_price) :
  reduced_price = 32 := by
  sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l3820_382030


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3820_382076

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given an arithmetic sequence with S_3 = 3 and S_6 = 7, prove S_9 = 12 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h3 : a.S 3 = 3) 
  (h6 : a.S 6 = 7) : 
  a.S 9 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3820_382076


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3820_382070

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 3 - y^2 / 4 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop := y = 2 * Real.sqrt 3 / 3 * x ∨ y = -2 * Real.sqrt 3 / 3 * x

/-- Theorem stating that the given asymptote equations are correct for the given hyperbola -/
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3820_382070


namespace NUMINAMATH_CALUDE_rhombus_in_rectangle_perimeter_l3820_382089

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Checks if a quadrilateral is a rhombus -/
def is_rhombus (q : Quadrilateral) : Prop := sorry

/-- Checks if a quadrilateral is a rectangle -/
def is_rectangle (q : Quadrilateral) : Prop := sorry

/-- Checks if a point is on a line segment -/
def is_on_segment (p : Point) (a : Point) (b : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 : Point) (p2 : Point) : ℝ := sorry

/-- Calculates the perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℝ := sorry

theorem rhombus_in_rectangle_perimeter 
  (W X Y Z : Point) 
  (A B C D : Point) :
  let rect := Quadrilateral.mk W X Y Z
  let rhom := Quadrilateral.mk A B C D
  is_rectangle rect →
  is_rhombus rhom →
  is_on_segment A W X →
  is_on_segment B X Y →
  is_on_segment C Y Z →
  is_on_segment D Z W →
  distance W A = 12 →
  distance X B = 9 →
  distance B D = 15 →
  distance A C = distance X Y →
  perimeter rect = 66 := by sorry

end NUMINAMATH_CALUDE_rhombus_in_rectangle_perimeter_l3820_382089


namespace NUMINAMATH_CALUDE_rabbit_walk_distance_l3820_382095

/-- The perimeter of a square park -/
def park_perimeter (side_length : ℝ) : ℝ := 4 * side_length

/-- The theorem stating that a rabbit walking along the perimeter of a square park
    with a side length of 13 meters walks 52 meters -/
theorem rabbit_walk_distance : park_perimeter 13 = 52 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_walk_distance_l3820_382095


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l3820_382066

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  area : ℝ
  base1 : ℝ
  base2 : ℝ
  side : ℝ

/-- The theorem stating the side length of the specific isosceles trapezoid -/
theorem isosceles_trapezoid_side_length 
  (t : IsoscelesTrapezoid) 
  (h_area : t.area = 44)
  (h_base1 : t.base1 = 8)
  (h_base2 : t.base2 = 14) :
  t.side = 5 := by
  sorry

#check isosceles_trapezoid_side_length

end NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l3820_382066


namespace NUMINAMATH_CALUDE_pancake_accumulation_time_l3820_382010

/-- Represents the pancake baking and eating scenario -/
structure PancakeScenario where
  mom_rate : ℚ     -- Mom's baking rate in pancakes per minute
  anya_rate : ℚ    -- Anya's baking rate in pancakes per minute
  andrey_rate : ℚ  -- Andrey's eating rate in pancakes per minute

/-- Calculates the time needed to accumulate a given number of pancakes -/
def time_to_accumulate (scenario : PancakeScenario) (target : ℚ) : ℚ :=
  target / (scenario.mom_rate + scenario.anya_rate - scenario.andrey_rate)

/-- Theorem stating the time to accumulate 100 pancakes is 24 minutes -/
theorem pancake_accumulation_time :
  let scenario : PancakeScenario := {
    mom_rate := 100 / 30,    -- 100 pancakes in 30 minutes
    anya_rate := 100 / 40,   -- 100 pancakes in 40 minutes
    andrey_rate := 100 / 60  -- 100 pancakes in 60 minutes
  }
  time_to_accumulate scenario 100 = 24 := by sorry

end NUMINAMATH_CALUDE_pancake_accumulation_time_l3820_382010


namespace NUMINAMATH_CALUDE_swimming_problem_solution_l3820_382009

/-- Represents the amount paid by each person -/
structure Payment where
  adam : ℕ
  bill : ℕ
  chris : ℕ

/-- The problem setup -/
def swimming_problem : Prop :=
  ∃ (cost_per_session : ℕ) (final_payment : Payment),
    -- Total number of sessions
    15 * cost_per_session = final_payment.adam + final_payment.bill + final_payment.chris
    -- Adam paid 8 times
    ∧ 8 * cost_per_session = final_payment.adam + 18
    -- Bill paid 7 times
    ∧ 7 * cost_per_session = final_payment.bill + 12
    -- Chris owes £30
    ∧ final_payment.chris = 30
    -- All have paid the same amount after Chris's payment
    ∧ final_payment.adam = final_payment.bill
    ∧ final_payment.bill = final_payment.chris

theorem swimming_problem_solution : swimming_problem := by
  sorry

end NUMINAMATH_CALUDE_swimming_problem_solution_l3820_382009


namespace NUMINAMATH_CALUDE_father_chips_amount_l3820_382072

theorem father_chips_amount (son_chips brother_chips total_chips : ℕ) 
  (h1 : son_chips = 350)
  (h2 : brother_chips = 182)
  (h3 : total_chips = 800) :
  total_chips - (son_chips + brother_chips) = 268 := by
  sorry

end NUMINAMATH_CALUDE_father_chips_amount_l3820_382072


namespace NUMINAMATH_CALUDE_bernoulli_zero_success_l3820_382045

/-- The number of trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The probability of failure in each trial -/
def q : ℚ := 1 - p

/-- The number of successes we're interested in -/
def k : ℕ := 0

/-- 
Theorem: In a series of 7 Bernoulli trials with a success probability of 2/7, 
the probability of 0 successes is equal to (5/7)^7.
-/
theorem bernoulli_zero_success : 
  (n.choose k) * p^k * q^(n-k) = (5/7)^7 := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_zero_success_l3820_382045


namespace NUMINAMATH_CALUDE_total_questions_on_test_l3820_382008

/-- Represents a student's test results -/
structure TestResult where
  score : Int
  correct : Nat
  total : Nat

/-- Calculates the score based on correct and incorrect responses -/
def calculateScore (correct : Nat) (incorrect : Nat) : Int :=
  correct - 2 * incorrect

/-- Theorem: Given the scoring system and Student A's results, prove the total number of questions -/
theorem total_questions_on_test (result : TestResult) 
  (h1 : result.score = calculateScore result.correct (result.total - result.correct))
  (h2 : result.score = 76)
  (h3 : result.correct = 92) :
  result.total = 100 := by
  sorry

#eval calculateScore 92 8

end NUMINAMATH_CALUDE_total_questions_on_test_l3820_382008


namespace NUMINAMATH_CALUDE_infinite_geometric_series_common_ratio_l3820_382083

theorem infinite_geometric_series_common_ratio 
  (a : ℝ) 
  (S : ℝ) 
  (h1 : a = 512) 
  (h2 : S = 8000) : 
  ∃ (r : ℝ), r = 0.936 ∧ S = a / (1 - r) := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_common_ratio_l3820_382083


namespace NUMINAMATH_CALUDE_log_power_base_equality_l3820_382046

theorem log_power_base_equality (a N m n : ℝ) 
  (ha : a > 0) (hN : N > 0) (hm : m ≠ 0) :
  Real.log N^n / Real.log a^m = n / m * Real.log N / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_log_power_base_equality_l3820_382046


namespace NUMINAMATH_CALUDE_steps_ratio_second_to_first_day_l3820_382048

/-- Proves that the ratio of steps walked on the second day to the first day is 1:1 -/
theorem steps_ratio_second_to_first_day (
  first_day_steps : ℕ) 
  (second_day_steps : ℕ) 
  (third_day_steps : ℕ) 
  (h1 : first_day_steps = 200 + 300)
  (h2 : third_day_steps = second_day_steps + 100)
  (h3 : first_day_steps + second_day_steps + third_day_steps = 1600) :
  second_day_steps = first_day_steps :=
sorry

end NUMINAMATH_CALUDE_steps_ratio_second_to_first_day_l3820_382048


namespace NUMINAMATH_CALUDE_asia_highest_population_l3820_382016

-- Define the structure for continent population data
structure ContinentPopulation where
  name : String
  population1950 : ℝ
  population2000 : ℝ

-- Define Asia's population data
def asia : ContinentPopulation := {
  name := "Asia",
  population1950 := 1.402,
  population2000 := 3.683
}

-- Define a function to check if a continent has the highest population
def hasHighestPopulation (continent : ContinentPopulation) (allContinents : List ContinentPopulation) (year : Nat) : Prop :=
  match year with
  | 1950 => ∀ c ∈ allContinents, continent.population1950 ≥ c.population1950
  | 2000 => ∀ c ∈ allContinents, continent.population2000 ≥ c.population2000
  | _ => False

-- Theorem statement
theorem asia_highest_population (allContinents : List ContinentPopulation) :
  asia ∈ allContinents →
  hasHighestPopulation asia allContinents 1950 ∧ hasHighestPopulation asia allContinents 2000 := by
  sorry

end NUMINAMATH_CALUDE_asia_highest_population_l3820_382016


namespace NUMINAMATH_CALUDE_veronica_ring_removal_ways_l3820_382043

/-- Represents the number of rings on each finger --/
structure RingDistribution :=
  (little : Nat)
  (middle : Nat)
  (ring : Nat)

/-- Calculates the number of ways to remove rings given a distribution --/
def removalWays (dist : RingDistribution) (fixedOrderOnRingFinger : Bool) : Nat :=
  if fixedOrderOnRingFinger then
    dist.little * dist.middle
  else
    sorry

/-- The specific ring distribution in the problem --/
def veronicaRings : RingDistribution :=
  { little := 1, middle := 1, ring := 3 }

theorem veronica_ring_removal_ways :
  removalWays veronicaRings true = 20 := by sorry

end NUMINAMATH_CALUDE_veronica_ring_removal_ways_l3820_382043


namespace NUMINAMATH_CALUDE_snowfall_sum_l3820_382094

/-- The total snowfall recorded during a three-day snowstorm -/
def total_snowfall (wednesday thursday friday : ℝ) : ℝ :=
  wednesday + thursday + friday

/-- Proof that the total snowfall is 0.88 cm given the daily measurements -/
theorem snowfall_sum :
  total_snowfall 0.33 0.33 0.22 = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_snowfall_sum_l3820_382094


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l3820_382012

/-- Arithmetic sequence term -/
def a_n (a b n : ℕ) : ℕ := a + (n - 1) * b

/-- Geometric sequence term -/
def b_n (a b n : ℕ) : ℕ := b * a^(n - 1)

/-- C_n sequence term -/
def C_n (a b n : ℕ) : ℕ := a_n a b (n + 1) + b_n a b n

theorem arithmetic_geometric_sequence_problem 
  (a b : ℕ) 
  (h_a_pos : a > 1) 
  (h_b_pos : b > 1) 
  (h_a1_lt_b1 : a < b) 
  (h_b2_lt_a3 : b * a < a + 2 * b) 
  (h_exists_m : ∀ n : ℕ, n > 0 → ∃ m : ℕ, m > 0 ∧ a_n a b m + 3 = b_n a b n) :
  (a = 2 ∧ b = 5) ∧ 
  (b = 4 → 
    ∃ n : ℕ, C_n a b n = 18 ∧ C_n a b (n + 1) = 30 ∧ C_n a b (n + 2) = 50 ∧
    ∀ k : ℕ, k ≠ n → ¬(C_n a b k * C_n a b (k + 2) = (C_n a b (k + 1))^2)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l3820_382012


namespace NUMINAMATH_CALUDE_intersection_A_B_l3820_382036

-- Define set A
def A : Set ℝ := {x : ℝ | 3 * x + 2 > 0}

-- Define set B
def B : Set ℝ := {x : ℝ | (x + 1) * (x - 3) > 0}

-- Theorem to prove
theorem intersection_A_B : A ∩ B = {x : ℝ | x > 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3820_382036


namespace NUMINAMATH_CALUDE_insulation_cost_theorem_l3820_382056

def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

def tank_a_area : ℝ := surface_area 3 5 2
def tank_b_area : ℝ := surface_area 4 6 3
def tank_c_area : ℝ := surface_area 6 8 4

def total_area : ℝ := 5 * tank_a_area + 3 * tank_b_area + 2 * tank_c_area

def cost_per_sq_ft : ℝ := 20

def discount_rate : ℝ := 0.2

theorem insulation_cost_theorem :
  let cost_before_discount := total_area * cost_per_sq_ft
  let discount := discount_rate * cost_before_discount
  cost_before_discount - discount = 16800 := by sorry

end NUMINAMATH_CALUDE_insulation_cost_theorem_l3820_382056


namespace NUMINAMATH_CALUDE_f_above_half_g_l3820_382002

/-- The function f(x) = 2x - 1 -/
def f (x : ℝ) : ℝ := 2 * x - 1

/-- The function g(x) = -2x + m, where m is a parameter -/
def g (m : ℝ) (x : ℝ) : ℝ := -2 * x + m

/-- The only integer solution to x ≥ 1 is 2 -/
axiom integer_solution (m : ℝ) : ∀ (x : ℤ), x ≥ 1 → (g m x ≥ 1 ↔ x = 2)

/-- The main theorem: f(x) is always above (1/2)g(x) if and only if a ∈ [3, +∞) -/
theorem f_above_half_g (a m : ℝ) :
  (∀ x : ℝ, f x > (1/2) * g m x) ↔ a ∈ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_f_above_half_g_l3820_382002


namespace NUMINAMATH_CALUDE_quadratic_inequality_sum_l3820_382058

/-- Given a quadratic inequality ax^2 - 5x + b > 0 with solution set {x | -3 < x < 2}, prove that a + b = 25 -/
theorem quadratic_inequality_sum (a b : ℝ) : 
  (∀ x, ax^2 - 5*x + b > 0 ↔ -3 < x ∧ x < 2) → 
  a + b = 25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_sum_l3820_382058


namespace NUMINAMATH_CALUDE_smallest_pretty_multiple_of_401_l3820_382014

/-- A positive integer is pretty if for each of its proper divisors d,
    there exist two divisors whose difference is d. -/
def IsPretty (n : ℕ) : Prop :=
  n > 0 ∧ ∀ d : ℕ, d ∣ n → 1 < d → d < n →
    ∃ d₁ d₂ : ℕ, d₁ ∣ n ∧ d₂ ∣ n ∧ 1 ≤ d₁ ∧ d₁ ≤ n ∧ 1 ≤ d₂ ∧ d₂ ≤ n ∧ d₂ - d₁ = d

theorem smallest_pretty_multiple_of_401 :
  ∃ n : ℕ, n > 401 ∧ 401 ∣ n ∧ IsPretty n ∧
    ∀ m : ℕ, m > 401 → 401 ∣ m → IsPretty m → n ≤ m :=
by
  use 160400
  sorry

end NUMINAMATH_CALUDE_smallest_pretty_multiple_of_401_l3820_382014


namespace NUMINAMATH_CALUDE_quadratic_root_sums_l3820_382057

theorem quadratic_root_sums (a b c x₁ x₂ : ℝ) (h : a ≠ 0) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁^2 + x₂^2 = (b^2 - 2*a*c) / a^2 ∧ 
   x₁^3 + x₂^3 = (3*a*b*c - b^3) / a^3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sums_l3820_382057


namespace NUMINAMATH_CALUDE_agri_products_theorem_l3820_382073

/-- Represents the prices and quantities of agricultural products A and B --/
structure AgriProducts where
  price_A : ℝ
  price_B : ℝ
  quantity_A : ℝ
  quantity_B : ℝ

/-- Represents the problem constraints and conditions --/
def problem_constraints (p : AgriProducts) : Prop :=
  2 * p.price_A + 3 * p.price_B = 690 ∧
  p.price_A + 4 * p.price_B = 720 ∧
  p.quantity_A + p.quantity_B = 40 ∧
  p.price_A * p.quantity_A + p.price_B * p.quantity_B ≤ 5400 ∧
  p.quantity_A ≤ 3 * p.quantity_B

/-- Calculates the profit given the prices and quantities --/
def profit (p : AgriProducts) : ℝ :=
  (160 - p.price_A) * p.quantity_A + (200 - p.price_B) * p.quantity_B

/-- The main theorem to prove --/
theorem agri_products_theorem (p : AgriProducts) :
  problem_constraints p →
  p.price_A = 120 ∧ p.price_B = 150 ∧
  ∀ q : AgriProducts, problem_constraints q →
    profit q ≤ profit { price_A := 120, price_B := 150, quantity_A := 20, quantity_B := 20 } :=
by sorry

end NUMINAMATH_CALUDE_agri_products_theorem_l3820_382073


namespace NUMINAMATH_CALUDE_largest_lower_bound_l3820_382005

theorem largest_lower_bound (x y : ℝ) (h1 : x ≠ y) (h2 : x * y = 2) :
  ((x + y)^2 - 6) * ((x - y)^2 + 8) / (x - y)^2 ≥ 2 ∧
  ∀ C > 2, ∃ x y : ℝ, x ≠ y ∧ x * y = 2 ∧ ((x + y)^2 - 6) * ((x - y)^2 + 8) / (x - y)^2 < C :=
by sorry

end NUMINAMATH_CALUDE_largest_lower_bound_l3820_382005


namespace NUMINAMATH_CALUDE_trivia_team_score_l3820_382026

theorem trivia_team_score :
  let total_members : ℕ := 7
  let absent_members : ℕ := 2
  let present_members : ℕ := total_members - absent_members
  let scores : List ℕ := [5, 9, 7, 5, 3]
  scores.length = present_members →
  scores.sum = 29 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_score_l3820_382026


namespace NUMINAMATH_CALUDE_sachin_rahul_age_difference_l3820_382067

theorem sachin_rahul_age_difference :
  ∀ (sachin_age rahul_age : ℝ),
    sachin_age = 38.5 →
    sachin_age / rahul_age = 11 / 9 →
    sachin_age - rahul_age = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_sachin_rahul_age_difference_l3820_382067


namespace NUMINAMATH_CALUDE_dubblefud_red_ball_value_l3820_382033

/-- The value of a red ball in the game of Dubblefud -/
def red_ball_value : ℕ := sorry

/-- The value of a blue ball in the game of Dubblefud -/
def blue_ball_value : ℕ := 4

/-- The value of a green ball in the game of Dubblefud -/
def green_ball_value : ℕ := 5

/-- The number of red balls in the selection -/
def num_red_balls : ℕ := 4

/-- The number of blue balls in the selection -/
def num_blue_balls : ℕ := sorry

/-- The number of green balls in the selection -/
def num_green_balls : ℕ := sorry

theorem dubblefud_red_ball_value :
  (red_ball_value ^ num_red_balls) * 
  (blue_ball_value ^ num_blue_balls) * 
  (green_ball_value ^ num_green_balls) = 16000 ∧
  num_blue_balls = num_green_balls →
  red_ball_value = 1 :=
sorry

end NUMINAMATH_CALUDE_dubblefud_red_ball_value_l3820_382033


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3820_382024

/-- An isosceles triangle with two sides of length 12 and a third side of length 17 has a perimeter of 41. -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b c : ℝ),
      a = 12 ∧ b = 12 ∧ c = 17 ∧  -- Two sides are 12, third side is 17
      (a = b ∨ a = c ∨ b = c) ∧   -- Definition of isosceles triangle
      perimeter = a + b + c ∧     -- Definition of perimeter
      perimeter = 41              -- The perimeter we want to prove

/-- Proof of the theorem -/
lemma proof_isosceles_triangle_perimeter : isosceles_triangle_perimeter 41 := by
  sorry

#check proof_isosceles_triangle_perimeter

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3820_382024


namespace NUMINAMATH_CALUDE_xy_squared_sum_l3820_382055

theorem xy_squared_sum (x y : ℝ) 
  (h1 : 1/x + 1/y = 3) 
  (h2 : x*y + x + y = 4) : 
  x^2*y + x*y^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_xy_squared_sum_l3820_382055


namespace NUMINAMATH_CALUDE_parallel_planes_line_condition_l3820_382015

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (lineParallel : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_line_condition
  (α β : Plane) (m : Line)
  (h_different : α ≠ β)
  (h_subset : subset m α) :
  (∀ α β m, parallel α β → lineParallel m β) ∧
  (∃ α β m, lineParallel m β ∧ ¬parallel α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_line_condition_l3820_382015


namespace NUMINAMATH_CALUDE_age_difference_l3820_382051

theorem age_difference (X Y Z : ℕ) : X + Y = Y + Z + 12 → (X - Z : ℚ) / 10 = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3820_382051


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l3820_382027

def U : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {1,2,3}
def B : Set Nat := {2,3,4,5}

theorem complement_intersection_A_B : 
  (U \ (A ∩ B)) = {1,4,5,6,7,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l3820_382027


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l3820_382023

theorem vector_perpendicular_condition (a b : ℝ × ℝ) (m : ℝ) : 
  ‖a‖ = 3 →
  ‖b‖ = 2 →
  a • b = 3 →
  (a - m • b) • a = 0 →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l3820_382023


namespace NUMINAMATH_CALUDE_P_divisible_by_Q_l3820_382087

variable (X : ℝ)
variable (n : ℕ)

def P (n : ℕ) (X : ℝ) : ℝ := n * X^(n+2) - (n+2) * X^(n+1) + (n+2) * X - n

def Q (X : ℝ) : ℝ := (X - 1)^3

theorem P_divisible_by_Q (n : ℕ) (h : n > 0) :
  ∃ k : ℝ, P n X = k * Q X := by
  sorry

end NUMINAMATH_CALUDE_P_divisible_by_Q_l3820_382087


namespace NUMINAMATH_CALUDE_gift_contributors_l3820_382047

theorem gift_contributors (total : ℝ) (min_contribution : ℝ) (max_contribution : ℝ) :
  total = 20 →
  min_contribution = 1 →
  max_contribution = 9 →
  (∃ (n : ℕ), n ≥ 1 ∧ n * min_contribution ≤ total ∧ total ≤ n * max_contribution) →
  (∀ (m : ℕ), m ≥ 1 → m * min_contribution ≤ total → total ≤ m * max_contribution → m ≥ 12) :=
by sorry

end NUMINAMATH_CALUDE_gift_contributors_l3820_382047


namespace NUMINAMATH_CALUDE_sandbox_capacity_increase_l3820_382090

theorem sandbox_capacity_increase (l w h : ℝ) : 
  l * w * h = 10 → (2 * l) * (2 * w) * (2 * h) = 80 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_capacity_increase_l3820_382090


namespace NUMINAMATH_CALUDE_triangle_area_product_l3820_382068

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 2 * a * x + 3 * b * y = 24) →
  (1/2 * (24 / (2 * a)) * (24 / (3 * b)) = 12) →
  a * b = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_product_l3820_382068


namespace NUMINAMATH_CALUDE_jinsu_work_rate_l3820_382052

/-- Given that Jinsu completes a task in 4 hours, prove that the amount of work he can do in one hour is 1/4 of the task. -/
theorem jinsu_work_rate (total_time : ℝ) (total_work : ℝ) (h : total_time = 4) :
  total_work / total_time = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_jinsu_work_rate_l3820_382052


namespace NUMINAMATH_CALUDE_clue_distance_range_l3820_382020

-- Define the distance to the nearest clue
variable (d : ℝ)

-- Define the statements of Adam, Bianca, and Cole
def adam_statement := d ≥ 8
def bianca_statement := d ≤ 7
def cole_statement := d ≤ 9

-- Theorem to prove the correct range for d
theorem clue_distance_range 
  (h1 : ¬adam_statement d) 
  (h2 : ¬bianca_statement d) 
  (h3 : ¬cole_statement d) : 
  7 < d ∧ d < 8 := by
  sorry


end NUMINAMATH_CALUDE_clue_distance_range_l3820_382020


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_function_l3820_382088

open Real

theorem symmetry_of_shifted_function :
  ∃ α : ℝ, 0 < α ∧ α < π / 3 ∧
  ∀ x : ℝ, (sin (x + α) + Real.sqrt 3 * cos (x + α)) =
           (sin (-x + α) + Real.sqrt 3 * cos (-x + α)) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_function_l3820_382088


namespace NUMINAMATH_CALUDE_quadratic_equation_conditions_l3820_382099

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 2 = 0
def q (a : ℝ) : Prop := ∀ x : ℝ, 0 < x → x < 1 → x^2 - a < 0

-- Define the set of real numbers that satisfy the conditions for p
def S₁ : Set ℝ := {a : ℝ | a ≤ -2 * Real.sqrt 2 ∨ a ≥ 2 * Real.sqrt 2}

-- Define the set of real numbers that satisfy the conditions for exactly one of p or q
def S₂ : Set ℝ := {a : ℝ | a ≤ -2 * Real.sqrt 2 ∨ (1 ≤ a ∧ a < 2 * Real.sqrt 2)}

-- State the theorem
theorem quadratic_equation_conditions (a : ℝ) :
  (p a ↔ a ∈ S₁) ∧
  ((p a ∧ ¬q a) ∨ (¬p a ∧ q a) ↔ a ∈ S₂) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_conditions_l3820_382099


namespace NUMINAMATH_CALUDE_binomial_19_10_l3820_382061

theorem binomial_19_10 (h1 : Nat.choose 17 7 = 19448) (h2 : Nat.choose 17 9 = 24310) :
  Nat.choose 19 10 = 92378 := by
  sorry

end NUMINAMATH_CALUDE_binomial_19_10_l3820_382061


namespace NUMINAMATH_CALUDE_min_distance_for_three_coloring_l3820_382044

-- Define the set of points in and on the regular hexagon
def hexagon_points : Set (ℝ × ℝ) := sorry

-- Define the distance function between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define a valid three-coloring scheme
def valid_three_coloring (r : ℝ) : Prop := 
  ∃ (coloring : (ℝ × ℝ) → Fin 3),
    ∀ (p q : hexagon_points), 
      coloring p = coloring q → distance p q < r

-- The main theorem
theorem min_distance_for_three_coloring : 
  (∀ r < 3/2, ¬ valid_three_coloring r) ∧ 
  valid_three_coloring (3/2) := by sorry

end NUMINAMATH_CALUDE_min_distance_for_three_coloring_l3820_382044


namespace NUMINAMATH_CALUDE_cruz_marbles_l3820_382063

/-- 
Given:
- Three times the sum of marbles that Atticus, Jensen, and Cruz have is equal to 60.
- Atticus has half as many marbles as Jensen.
- Atticus has 4 marbles.
Prove that Cruz has 8 marbles.
-/
theorem cruz_marbles (atticus jensen cruz : ℕ) : 
  3 * (atticus + jensen + cruz) = 60 →
  atticus = jensen / 2 →
  atticus = 4 →
  cruz = 8 := by
sorry

end NUMINAMATH_CALUDE_cruz_marbles_l3820_382063


namespace NUMINAMATH_CALUDE_small_cuboid_height_l3820_382037

/-- Proves that the height of small cuboids is 3 meters given the dimensions of a large cuboid
    and the dimensions of small cuboids that can be formed from it. -/
theorem small_cuboid_height
  (large_length large_width large_height : ℝ)
  (small_length small_width : ℝ)
  (num_small_cuboids : ℕ)
  (h_large_length : large_length = 18)
  (h_large_width : large_width = 15)
  (h_large_height : large_height = 2)
  (h_small_length : small_length = 5)
  (h_small_width : small_width = 2)
  (h_num_small_cuboids : num_small_cuboids = 18)
  : ∃ (small_height : ℝ),
    small_height = 3 ∧
    large_length * large_width * large_height =
    num_small_cuboids * small_length * small_width * small_height :=
by sorry

end NUMINAMATH_CALUDE_small_cuboid_height_l3820_382037


namespace NUMINAMATH_CALUDE_vertex_angle_is_45_degrees_l3820_382029

/-- An isosceles triangle with specific properties -/
structure SpecialIsoscelesTriangle where
  a : ℝ  -- Length of congruent sides
  s : ℝ  -- Semi-perimeter
  h : ℝ  -- Height to the base
  b : ℝ  -- Length of the base
  a_pos : 0 < a  -- Side length is positive
  s_pos : 0 < s  -- Semi-perimeter is positive
  h_pos : 0 < h  -- Height is positive
  b_pos : 0 < b  -- Base length is positive
  isosceles : s = a + b/2  -- Definition of semi-perimeter for this triangle
  right_base_angle : h = a  -- One base angle is a right angle
  area_condition : b * (2 * h) = s^2  -- Given condition

/-- The vertex angle at the base of the special isosceles triangle is 45° -/
theorem vertex_angle_is_45_degrees (t : SpecialIsoscelesTriangle) : 
  Real.arccos ((t.b / 2) / t.a) * (180 / Real.pi) = 45 := by
  sorry

end NUMINAMATH_CALUDE_vertex_angle_is_45_degrees_l3820_382029


namespace NUMINAMATH_CALUDE_older_friend_age_l3820_382097

theorem older_friend_age (A B C : ℝ) 
  (h1 : A - B = 2.5)
  (h2 : A - C = 3.75)
  (h3 : A + B + C = 110.5)
  (h4 : B = 2 * C) :
  A = 104.25 := by
  sorry

end NUMINAMATH_CALUDE_older_friend_age_l3820_382097


namespace NUMINAMATH_CALUDE_martaFamily_childless_count_marta_childless_count_l3820_382050

/-- Represents a woman in Marta's family tree -/
structure Woman where
  daughters : Nat

/-- Marta's family tree -/
structure MartaFamily where
  marta : Woman
  daughters : Finset Woman

theorem martaFamily_childless_count (f : MartaFamily) : Nat :=
  let total_women := f.daughters.card + (f.daughters.sum fun d => d.daughters)
  let daughters_with_children := f.daughters.filter fun d => d.daughters > 0
  let childless_count := f.daughters.card + (f.daughters.sum fun d => d.daughters) - daughters_with_children.card
  childless_count

/-- The number of Marta's daughters and granddaughters without daughters is 37 -/
theorem marta_childless_count : ∃ (f : MartaFamily),
  f.marta.daughters = 0 ∧
  f.daughters.card = 7 ∧
  (f.daughters.card + (f.daughters.sum fun d => d.daughters) = 42) ∧
  (∀ d ∈ f.daughters, d.daughters = 0 ∨ d.daughters = 6) ∧
  (∀ d ∈ f.daughters, ∀ g ∈ f.daughters, g.daughters = 0) →
  martaFamily_childless_count f = 37 := by
  sorry

end NUMINAMATH_CALUDE_martaFamily_childless_count_marta_childless_count_l3820_382050


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l3820_382060

theorem max_value_trig_expression (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l3820_382060


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l3820_382001

/-- Given a quadratic function y = 2x² - 8x - 1, prove it's equivalent to y = 2(x-2)² - 9 -/
theorem quadratic_equivalence :
  ∀ x y : ℝ, y = 2*x^2 - 8*x - 1 ↔ y = 2*(x-2)^2 - 9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l3820_382001


namespace NUMINAMATH_CALUDE_fraction_sum_and_simplification_l3820_382007

theorem fraction_sum_and_simplification : 
  (11 : ℚ) / 12 + 7 / 8 + 3 / 4 = 61 / 24 ∧ 
  ∀ (a b : ℤ), (b ≠ 0 ∧ (61 : ℚ) / 24 = a / b) → (|a| = 61 ∧ |b| = 24) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_and_simplification_l3820_382007


namespace NUMINAMATH_CALUDE_distance_between_A_and_B_l3820_382085

def A : ℝ × ℝ := (-3, 1)
def B : ℝ × ℝ := (6, -4)

theorem distance_between_A_and_B : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 106 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_A_and_B_l3820_382085


namespace NUMINAMATH_CALUDE_part_I_part_II_l3820_382074

-- Define set A
def A : Set ℝ := {x | 6 / (x + 1) ≥ 1}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem for part (I)
theorem part_I : A = {x | -1 < x ∧ x ≤ 5} := by sorry

-- Theorem for part (II)
theorem part_II : A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l3820_382074


namespace NUMINAMATH_CALUDE_race_catchup_time_l3820_382075

theorem race_catchup_time 
  (cristina_speed : ℝ) 
  (nicky_speed : ℝ) 
  (head_start : ℝ) 
  (h1 : cristina_speed = 6)
  (h2 : nicky_speed = 3)
  (h3 : head_start = 36) :
  ∃ t : ℝ, t = 12 ∧ cristina_speed * t = head_start + nicky_speed * t := by
sorry

end NUMINAMATH_CALUDE_race_catchup_time_l3820_382075


namespace NUMINAMATH_CALUDE_total_earnings_is_5800_l3820_382079

/-- Represents the investment and return information for three investors -/
structure InvestmentInfo where
  investment_ratio : Fin 3 → ℕ
  return_ratio : Fin 3 → ℕ
  earnings_difference : ℕ

/-- Calculates the total earnings based on the given investment information -/
def calculate_total_earnings (info : InvestmentInfo) : ℕ :=
  sorry

/-- Theorem stating that the total earnings are 5800 given the specified conditions -/
theorem total_earnings_is_5800 (info : InvestmentInfo) 
  (h1 : info.investment_ratio = ![3, 4, 5])
  (h2 : info.return_ratio = ![6, 5, 4])
  (h3 : info.earnings_difference = 200) :
  calculate_total_earnings info = 5800 :=
sorry

end NUMINAMATH_CALUDE_total_earnings_is_5800_l3820_382079


namespace NUMINAMATH_CALUDE_newspaper_spending_difference_l3820_382003

/-- Calculates the difference in yearly newspaper spending between Juanita and Grant -/
theorem newspaper_spending_difference : 
  let grant_yearly_spending : ℚ := 200
  let juanita_daily_spending : ℚ := 0.5
  let juanita_sunday_spending : ℚ := 2
  let days_per_week : ℕ := 7
  let weekdays : ℕ := 6
  let weeks_per_year : ℕ := 52
  
  let juanita_weekly_spending : ℚ := juanita_daily_spending * weekdays + juanita_sunday_spending
  let juanita_yearly_spending : ℚ := juanita_weekly_spending * weeks_per_year
  
  juanita_yearly_spending - grant_yearly_spending = 60
  := by sorry

end NUMINAMATH_CALUDE_newspaper_spending_difference_l3820_382003


namespace NUMINAMATH_CALUDE_books_in_wrong_place_l3820_382000

theorem books_in_wrong_place
  (initial_books : ℕ)
  (books_left : ℕ)
  (history_books : ℕ)
  (fiction_books : ℕ)
  (children_books : ℕ)
  (h1 : initial_books = 51)
  (h2 : books_left = 16)
  (h3 : history_books = 12)
  (h4 : fiction_books = 19)
  (h5 : children_books = 8) :
  history_books + fiction_books + children_books - (initial_books - books_left) = 4 :=
by sorry

end NUMINAMATH_CALUDE_books_in_wrong_place_l3820_382000


namespace NUMINAMATH_CALUDE_combined_perimeters_of_squares_l3820_382006

theorem combined_perimeters_of_squares (square1 square2 : Real) 
  (h1 : square1 * square1 = 36) 
  (h2 : square2 * square2 = 49) : 
  4 * square1 + 4 * square2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_combined_perimeters_of_squares_l3820_382006


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_three_l3820_382025

def is_multiple_of_three (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

theorem five_digit_multiple_of_three :
  ∀ d : ℕ, d < 10 →
    (is_multiple_of_three (56780 + d) ↔ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_three_l3820_382025


namespace NUMINAMATH_CALUDE_base_10_255_equals_base_4_3333_l3820_382069

/-- Converts a list of digits in base 4 to a natural number in base 10 -/
def base4ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- Theorem: 255 in base 10 is equal to 3333 in base 4 -/
theorem base_10_255_equals_base_4_3333 :
  255 = base4ToNat [3, 3, 3, 3] := by
  sorry

end NUMINAMATH_CALUDE_base_10_255_equals_base_4_3333_l3820_382069


namespace NUMINAMATH_CALUDE_egyptian_fraction_decomposition_l3820_382078

theorem egyptian_fraction_decomposition (n : ℕ) (h : n ≥ 5 ∧ Odd n) :
  (2 : ℚ) / 11 = 1 / 6 + 1 / 66 ∧
  (2 : ℚ) / n = 1 / ((n + 1) / 2) + 1 / (n * (n + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_egyptian_fraction_decomposition_l3820_382078


namespace NUMINAMATH_CALUDE_rehabilitation_centers_fraction_l3820_382062

theorem rehabilitation_centers_fraction (L J H Ja : ℕ) (f : ℚ) : 
  L = 6 →
  J = L - f * L →
  H = 2 * J - 2 →
  Ja = 2 * H + 6 →
  L + J + H + Ja = 27 →
  f = 1/2 := by sorry

end NUMINAMATH_CALUDE_rehabilitation_centers_fraction_l3820_382062


namespace NUMINAMATH_CALUDE_acquaintance_theorem_l3820_382041

/-- A graph with 9 vertices where every subset of 3 vertices contains at least 2 connected vertices -/
def AcquaintanceGraph : Type :=
  { g : SimpleGraph (Fin 9) // ∀ (s : Finset (Fin 9)), s.card = 3 →
    ∃ (v w : Fin 9), v ∈ s ∧ w ∈ s ∧ v ≠ w ∧ g.Adj v w }

/-- The existence of a complete subgraph of 4 vertices in the AcquaintanceGraph -/
theorem acquaintance_theorem (g : AcquaintanceGraph) :
  ∃ (s : Finset (Fin 9)), s.card = 4 ∧ ∀ (v w : Fin 9), v ∈ s → w ∈ s → v ≠ w → g.val.Adj v w :=
sorry

end NUMINAMATH_CALUDE_acquaintance_theorem_l3820_382041


namespace NUMINAMATH_CALUDE_number_relationship_l3820_382065

theorem number_relationship (A B C : ℝ) 
  (h1 : B = 10)
  (h2 : A * B = 85)
  (h3 : B * C = 115)
  (h4 : B - A = C - B) :
  B - A = 1.5 ∧ C - B = 1.5 := by
sorry

end NUMINAMATH_CALUDE_number_relationship_l3820_382065


namespace NUMINAMATH_CALUDE_min_value_abc_min_value_abc_achieved_l3820_382098

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 9) :
  a^3 * b^2 * c ≥ 64/729 := by
  sorry

theorem min_value_abc_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  1/a + 1/b + 1/c = 9 ∧
  a^3 * b^2 * c < 64/729 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_abc_min_value_abc_achieved_l3820_382098


namespace NUMINAMATH_CALUDE_tiles_needed_is_100_l3820_382082

/-- Calculates the number of tiles needed to cover a rectangular room with a central pillar -/
def calculate_tiles (room_length room_width pillar_side border_tile_side central_tile_side : ℕ) : ℕ :=
  let border_tiles := 2 * room_width
  let central_area := room_length * (room_width - 2) - pillar_side^2
  let central_tiles := (central_area + central_tile_side^2 - 1) / central_tile_side^2
  border_tiles + central_tiles

/-- The total number of tiles needed for the specific room configuration is 100 -/
theorem tiles_needed_is_100 : calculate_tiles 30 20 2 1 3 = 100 := by sorry

end NUMINAMATH_CALUDE_tiles_needed_is_100_l3820_382082


namespace NUMINAMATH_CALUDE_connect_four_shapes_l3820_382080

/-- Represents a Connect Four board configuration --/
def ConnectFourBoard := Fin 7 → Fin 9

/-- The number of unique shapes in a Connect Four board --/
def num_unique_shapes : ℕ :=
  let symmetric_shapes := 9^4
  let total_shapes := 9^7
  symmetric_shapes + (total_shapes - symmetric_shapes) / 2

/-- The sum of the first n natural numbers --/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that the number of unique shapes in a Connect Four board
    is equal to 9 times the sum of the first 729 natural numbers --/
theorem connect_four_shapes :
  num_unique_shapes = 9 * sum_first_n 729 := by
  sorry


end NUMINAMATH_CALUDE_connect_four_shapes_l3820_382080


namespace NUMINAMATH_CALUDE_small_circle_radius_l3820_382032

/-- Given a configuration of circles where:
    - There is a large circle with radius 10 meters
    - Six congruent smaller circles are arranged around it
    - Each smaller circle touches two others and the larger circle
    This theorem proves that the radius of each smaller circle is 5√3 meters -/
theorem small_circle_radius (R : ℝ) (r : ℝ) : 
  R = 10 → -- The radius of the larger circle is 10 meters
  R = (2 * r) / Real.sqrt 3 → -- Relationship between radii based on hexagon geometry
  r = 5 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l3820_382032


namespace NUMINAMATH_CALUDE_min_sum_m_n_in_arithmetic_sequence_l3820_382071

theorem min_sum_m_n_in_arithmetic_sequence (a : ℕ → ℕ) (d m n : ℕ) :
  (∀ k, a k > 0) →
  (∀ k, a (k + 1) = a k + d) →
  a 1 = 1919 →
  a m = 1949 →
  a n = 2019 →
  m > 0 →
  n > 0 →
  ∃ (m' n' : ℕ), m' > 0 ∧ n' > 0 ∧ a m' = 1949 ∧ a n' = 2019 ∧ m' + n' = 15 ∧
    ∀ (p q : ℕ), p > 0 → q > 0 → a p = 1949 → a q = 2019 → m' + n' ≤ p + q :=
by sorry

end NUMINAMATH_CALUDE_min_sum_m_n_in_arithmetic_sequence_l3820_382071


namespace NUMINAMATH_CALUDE_tv_watching_days_l3820_382004

/-- The number of days per week children are allowed to watch TV -/
def days_per_week : ℕ := sorry

/-- The number of minutes children watch TV each day they are allowed -/
def minutes_per_day : ℕ := 45

/-- The total number of hours children watch TV in 2 weeks -/
def total_hours_in_two_weeks : ℕ := 6

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem tv_watching_days : 
  days_per_week * minutes_per_day * 2 = total_hours_in_two_weeks * minutes_per_hour :=
sorry

end NUMINAMATH_CALUDE_tv_watching_days_l3820_382004


namespace NUMINAMATH_CALUDE_log_value_proof_l3820_382077

theorem log_value_proof (a : ℝ) (h1 : a > 0) (h2 : a^(1/2) = 4/9) :
  Real.log a / Real.log (2/3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_value_proof_l3820_382077


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_cubed_l3820_382011

theorem imaginary_part_of_one_minus_i_cubed (i : ℂ) : 
  Complex.im ((1 - i)^3) = -2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_cubed_l3820_382011


namespace NUMINAMATH_CALUDE_problem_solution_l3820_382022

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.sin y = 2008)
  (h2 : x + 2008 * Real.cos y = 2007)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3820_382022


namespace NUMINAMATH_CALUDE_klinker_double_age_l3820_382019

/-- The age difference between Mr. Klinker and his daughter -/
def age_difference : ℕ := 35 - 10

/-- The current age of Mr. Klinker -/
def klinker_age : ℕ := 35

/-- The current age of Mr. Klinker's daughter -/
def daughter_age : ℕ := 10

/-- The number of years until Mr. Klinker is twice as old as his daughter -/
def years_until_double : ℕ := 15

theorem klinker_double_age :
  klinker_age + years_until_double = 2 * (daughter_age + years_until_double) :=
sorry

end NUMINAMATH_CALUDE_klinker_double_age_l3820_382019


namespace NUMINAMATH_CALUDE_vance_family_stamp_cost_difference_l3820_382017

theorem vance_family_stamp_cost_difference :
  let mr_rooster_count : ℕ := 3
  let mr_rooster_price : ℚ := 3/2
  let mr_daffodil_count : ℕ := 5
  let mr_daffodil_price : ℚ := 3/4
  let mrs_rooster_count : ℕ := 2
  let mrs_rooster_price : ℚ := 5/4
  let mrs_daffodil_count : ℕ := 7
  let mrs_daffodil_price : ℚ := 4/5
  let john_rooster_count : ℕ := 4
  let john_rooster_price : ℚ := 7/5
  let john_daffodil_count : ℕ := 3
  let john_daffodil_price : ℚ := 7/10

  let total_rooster_cost : ℚ := 
    mr_rooster_count * mr_rooster_price + 
    mrs_rooster_count * mrs_rooster_price + 
    john_rooster_count * john_rooster_price

  let total_daffodil_cost : ℚ := 
    mr_daffodil_count * mr_daffodil_price + 
    mrs_daffodil_count * mrs_daffodil_price + 
    john_daffodil_count * john_daffodil_price

  total_rooster_cost - total_daffodil_cost = 23/20
  := by sorry

end NUMINAMATH_CALUDE_vance_family_stamp_cost_difference_l3820_382017


namespace NUMINAMATH_CALUDE_loan_problem_l3820_382084

/-- Proves that given the conditions of the loan problem, the second part is lent for 3 years -/
theorem loan_problem (total : ℝ) (second_part : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time1 : ℝ) (n : ℝ) : 
  total = 2717 →
  second_part = 1672 →
  rate1 = 0.03 →
  rate2 = 0.05 →
  time1 = 8 →
  (total - second_part) * rate1 * time1 = second_part * rate2 * n →
  n = 3 := by
sorry


end NUMINAMATH_CALUDE_loan_problem_l3820_382084


namespace NUMINAMATH_CALUDE_line_segment_lattice_points_l3820_382038

/-- The number of lattice points on a line segment --/
def latticePointCount (x1 y1 x2 y2 : Int) : Nat :=
  sorry

/-- Theorem: The line segment from (5, 5) to (65, 290) contains 16 lattice points --/
theorem line_segment_lattice_points :
  latticePointCount 5 5 65 290 = 16 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_lattice_points_l3820_382038


namespace NUMINAMATH_CALUDE_coefficients_of_2x2_plus_x_minus_5_l3820_382021

/-- Given a quadratic equation ax² + bx + c = 0, returns a tuple (a, b, c) of its coefficients -/
def quadratic_coefficients (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

theorem coefficients_of_2x2_plus_x_minus_5 :
  quadratic_coefficients 2 1 (-5) = (2, 1, -5) := by sorry

end NUMINAMATH_CALUDE_coefficients_of_2x2_plus_x_minus_5_l3820_382021


namespace NUMINAMATH_CALUDE_virus_spread_l3820_382034

/-- Given that one infected computer leads to 121 infected computers after two rounds of infection,
    prove that the average number of computers infected by one computer in each round is 10. -/
theorem virus_spread (x : ℝ) : (1 + x + x * x = 121) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_virus_spread_l3820_382034


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l3820_382081

/-- Given a triangle with sides 9, 12, and 15, the shortest altitude has length 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 → b = 12 → c = 15 →
  (a^2 + b^2 = c^2) →
  (h * c = 2 * (a * b / 2)) →
  h = 7.2 := by sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l3820_382081


namespace NUMINAMATH_CALUDE_sum_difference_remainder_mod_two_l3820_382092

theorem sum_difference_remainder_mod_two : 
  let n := 100
  let sum_remainder_one := (Finset.range n).sum (fun i => if i % 2 = 1 then i + 1 else 0)
  let sum_remainder_zero := (Finset.range n).sum (fun i => if i % 2 = 0 then i + 1 else 0)
  sum_remainder_zero - sum_remainder_one = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_difference_remainder_mod_two_l3820_382092


namespace NUMINAMATH_CALUDE_circle_properties_l3820_382049

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Theorem statement
theorem circle_properties :
  -- The center is on the y-axis
  ∃ (y₀ : ℝ), circle_equation 0 y₀
  -- The radius is 1
  ∧ ∀ (x y : ℝ), circle_equation x y → (x^2 + (y - 2)^2 = 1)
  -- The circle passes through (1,2)
  ∧ circle_equation 1 2 := by
sorry

end NUMINAMATH_CALUDE_circle_properties_l3820_382049


namespace NUMINAMATH_CALUDE_atMostOneHead_exactlyTwoHeads_mutually_exclusive_l3820_382013

/-- Represents the outcome of a single coin toss -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of tossing two coins simultaneously -/
def TwoCoinsOutcome := (CoinOutcome × CoinOutcome)

/-- The event of getting at most one head when tossing two coins -/
def atMostOneHead (outcome : TwoCoinsOutcome) : Prop :=
  match outcome with
  | (CoinOutcome.Tails, CoinOutcome.Tails) => True
  | (CoinOutcome.Heads, CoinOutcome.Tails) => True
  | (CoinOutcome.Tails, CoinOutcome.Heads) => True
  | (CoinOutcome.Heads, CoinOutcome.Heads) => False

/-- The event of getting exactly two heads when tossing two coins -/
def exactlyTwoHeads (outcome : TwoCoinsOutcome) : Prop :=
  match outcome with
  | (CoinOutcome.Heads, CoinOutcome.Heads) => True
  | _ => False

/-- Theorem stating that "at most one head" and "exactly two heads" are mutually exclusive -/
theorem atMostOneHead_exactlyTwoHeads_mutually_exclusive :
  ∀ (outcome : TwoCoinsOutcome), ¬(atMostOneHead outcome ∧ exactlyTwoHeads outcome) :=
by
  sorry


end NUMINAMATH_CALUDE_atMostOneHead_exactlyTwoHeads_mutually_exclusive_l3820_382013


namespace NUMINAMATH_CALUDE_fraction_equality_implies_x_value_l3820_382035

theorem fraction_equality_implies_x_value :
  ∀ x : ℚ, (x + 6) / (x - 4) = (x - 7) / (x + 2) → x = 16 / 19 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_x_value_l3820_382035


namespace NUMINAMATH_CALUDE_harris_dog_vegetable_cost_l3820_382040

/-- Represents the cost and quantity of a vegetable in a 1-pound bag -/
structure VegetableInfo where
  quantity : ℕ
  cost : ℚ

/-- Calculates the annual cost of vegetables for Harris's dog -/
def annual_vegetable_cost (carrot_info celery_info pepper_info : VegetableInfo) 
  (daily_carrot daily_celery daily_pepper : ℕ) : ℚ :=
  let daily_cost := 
    daily_carrot * (carrot_info.cost / carrot_info.quantity) +
    daily_celery * (celery_info.cost / celery_info.quantity) +
    daily_pepper * (pepper_info.cost / pepper_info.quantity)
  daily_cost * 365

/-- Theorem stating the annual cost of vegetables for Harris's dog -/
theorem harris_dog_vegetable_cost :
  let carrot_info : VegetableInfo := ⟨5, 2⟩
  let celery_info : VegetableInfo := ⟨10, 3/2⟩
  let pepper_info : VegetableInfo := ⟨3, 5/2⟩
  annual_vegetable_cost carrot_info celery_info pepper_info 1 2 1 = 11169/20 := by
  sorry

end NUMINAMATH_CALUDE_harris_dog_vegetable_cost_l3820_382040


namespace NUMINAMATH_CALUDE_inequality_proof_l3820_382093

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / a + 1 / b + 1 / c ≥ 9 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3820_382093


namespace NUMINAMATH_CALUDE_mikes_remaining_books_l3820_382091

theorem mikes_remaining_books (initial_books sold_books : ℕ) :
  initial_books = 51 →
  sold_books = 45 →
  initial_books - sold_books = 6 :=
by sorry

end NUMINAMATH_CALUDE_mikes_remaining_books_l3820_382091


namespace NUMINAMATH_CALUDE_fraction_equals_26_11_l3820_382039

def numerator : ℕ → ℚ
| 0 => 15
| n + 1 => (numerator n) * (1 + 14 / (n + 2))

def denominator : ℕ → ℚ
| 0 => 13
| n + 1 => (denominator n) * (1 + 12 / (n + 2))

def fraction : ℚ := (numerator 11) / (denominator 9)

theorem fraction_equals_26_11 : fraction = 26 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_26_11_l3820_382039


namespace NUMINAMATH_CALUDE_product_of_consecutive_integers_l3820_382031

theorem product_of_consecutive_integers : ∃ (a b c d e : ℤ),
  b = a + 1 ∧
  d = c + 1 ∧
  e = d + 1 ∧
  a * b = 300 ∧
  c * d * e = 300 ∧
  a + b + c + d + e = 49 :=
by sorry

end NUMINAMATH_CALUDE_product_of_consecutive_integers_l3820_382031


namespace NUMINAMATH_CALUDE_paul_age_in_12_years_l3820_382053

/-- Represents the ages of people in the problem -/
structure Ages where
  brian : ℝ
  christian : ℝ
  margaret : ℝ
  paul : ℝ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.christian = 3.5 * ages.brian ∧
  ages.brian + 12 = 45 ∧
  ages.margaret = 2 * ages.brian ∧
  ages.christian = ages.margaret + 15 ∧
  ages.paul = (ages.margaret + ages.christian) / 2

/-- The theorem to be proved -/
theorem paul_age_in_12_years (ages : Ages) :
  problem_conditions ages → ages.paul + 12 = 102.75 := by
  sorry

end NUMINAMATH_CALUDE_paul_age_in_12_years_l3820_382053


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l3820_382064

theorem cos_double_angle_special_case (θ : Real) 
  (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : 
  Real.cos (2 * θ) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l3820_382064
