import Mathlib

namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2751_275112

theorem absolute_value_inequality (x : ℝ) :
  |2 * x + 1| < 3 ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2751_275112


namespace NUMINAMATH_CALUDE_nearest_gardeners_to_flower_l2751_275199

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Represents a gardener -/
structure Gardener where
  position : Point

/-- Represents a flower -/
structure Flower where
  position : Point

/-- Theorem: The three nearest gardeners to a flower in the top-left quarter
    of a 2x2 grid are those at the top-left, top-right, and bottom-left corners -/
theorem nearest_gardeners_to_flower 
  (gardenerA : Gardener) 
  (gardenerB : Gardener)
  (gardenerC : Gardener)
  (gardenerD : Gardener)
  (flower : Flower)
  (h1 : gardenerA.position = ⟨0, 2⟩)
  (h2 : gardenerB.position = ⟨2, 2⟩)
  (h3 : gardenerC.position = ⟨0, 0⟩)
  (h4 : gardenerD.position = ⟨2, 0⟩)
  (h5 : 0 < flower.position.x ∧ flower.position.x < 1)
  (h6 : 1 < flower.position.y ∧ flower.position.y < 2) :
  squaredDistance flower.position gardenerA.position < squaredDistance flower.position gardenerD.position ∧
  squaredDistance flower.position gardenerB.position < squaredDistance flower.position gardenerD.position ∧
  squaredDistance flower.position gardenerC.position < squaredDistance flower.position gardenerD.position :=
by sorry

end NUMINAMATH_CALUDE_nearest_gardeners_to_flower_l2751_275199


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l2751_275157

-- Define the triangle PQR
def Triangle (P Q R : ℝ) : Prop := 
  0 < P ∧ 0 < Q ∧ 0 < R ∧ P + Q > R ∧ P + R > Q ∧ Q + R > P

-- Define the sides of the triangle
def PQ : ℝ := 8
def PR : ℝ := 7
def QR : ℝ := 5

-- State the theorem
theorem triangle_ratio_theorem (P Q R : ℝ) 
  (h : Triangle P Q R) 
  (h_pq : PQ = 8) 
  (h_pr : PR = 7) 
  (h_qr : QR = 5) : 
  (Real.cos ((P - Q) / 2) / Real.sin (R / 2)) - 
  (Real.sin ((P - Q) / 2) / Real.cos (R / 2)) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l2751_275157


namespace NUMINAMATH_CALUDE_tan_65_degrees_l2751_275152

/-- If tan 110° = α, then tan 65° = (α - 1) / (1 + α) -/
theorem tan_65_degrees (α : ℝ) (h : Real.tan (110 * π / 180) = α) :
  Real.tan (65 * π / 180) = (α - 1) / (α + 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_65_degrees_l2751_275152


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2751_275126

theorem quadratic_root_range (t : ℝ) :
  (∃ α β : ℝ, (3*t*α^2 + (3-7*t)*α + 2 = 0) ∧
              (3*t*β^2 + (3-7*t)*β + 2 = 0) ∧
              (0 < α) ∧ (α < 1) ∧ (1 < β) ∧ (β < 2)) →
  (5/4 < t) ∧ (t < 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2751_275126


namespace NUMINAMATH_CALUDE_arcade_spend_example_l2751_275125

/-- The amount of money spent at an arcade given the play time and cost per interval -/
def arcade_spend (play_time_hours : ℕ) (cost_per_interval : ℚ) (interval_minutes : ℕ) : ℚ :=
  (play_time_hours * 60 / interval_minutes) * cost_per_interval

/-- Theorem: Given 3 hours of play time and $0.50 per 6 minutes, the total spend is $15 -/
theorem arcade_spend_example : arcade_spend 3 (1/2) 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arcade_spend_example_l2751_275125


namespace NUMINAMATH_CALUDE_perpendicular_line_slope_OA_longer_than_OB_l2751_275179

/-- The ellipse C with equation x² + y²/4 = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2/4 = 1}

/-- The line y = kx + 1 for a given k -/
def Line (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + 1}

/-- A and B are the intersection points of C and the line -/
def A (k : ℝ) : ℝ × ℝ := sorry
def B (k : ℝ) : ℝ × ℝ := sorry

/-- Condition that A is in the first quadrant -/
def A_in_first_quadrant (k : ℝ) : Prop := (A k).1 > 0 ∧ (A k).2 > 0

theorem perpendicular_line_slope (k : ℝ) :
  (A k).1 * (B k).1 + (A k).2 * (B k).2 = 0 → k = 1/2 ∨ k = -1/2 := sorry

theorem OA_longer_than_OB (k : ℝ) :
  k > 0 → A_in_first_quadrant k →
  (A k).1^2 + (A k).2^2 > (B k).1^2 + (B k).2^2 := sorry

end NUMINAMATH_CALUDE_perpendicular_line_slope_OA_longer_than_OB_l2751_275179


namespace NUMINAMATH_CALUDE_managers_salary_l2751_275149

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (avg_increase : ℝ) :
  num_employees = 20 ∧ 
  avg_salary = 1500 ∧ 
  avg_increase = 100 →
  (num_employees + 1) * (avg_salary + avg_increase) - num_employees * avg_salary = 3600 :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l2751_275149


namespace NUMINAMATH_CALUDE_investment_amount_l2751_275193

/-- Proves that given a monthly interest payment of $240 and a simple annual interest rate of 9%,
    the principal amount of the investment is $32,000. -/
theorem investment_amount (monthly_interest : ℝ) (annual_rate : ℝ) (principal : ℝ) :
  monthly_interest = 240 →
  annual_rate = 0.09 →
  principal = monthly_interest / (annual_rate / 12) →
  principal = 32000 := by
  sorry

end NUMINAMATH_CALUDE_investment_amount_l2751_275193


namespace NUMINAMATH_CALUDE_floor_of_7_9_l2751_275170

theorem floor_of_7_9 : ⌊(7.9 : ℝ)⌋ = 7 := by sorry

end NUMINAMATH_CALUDE_floor_of_7_9_l2751_275170


namespace NUMINAMATH_CALUDE_unique_prime_sum_30_l2751_275197

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem unique_prime_sum_30 :
  ∃! (A B C : ℕ), 
    isPrime A ∧ isPrime B ∧ isPrime C ∧
    A < 20 ∧ B < 20 ∧ C < 20 ∧
    A + B + C = 30 ∧
    A = 2 ∧ B = 11 ∧ C = 17 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_sum_30_l2751_275197


namespace NUMINAMATH_CALUDE_broomstick_race_orderings_l2751_275178

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of competitors in the race -/
def num_competitors : ℕ := 4

theorem broomstick_race_orderings : 
  permutations num_competitors = 24 := by
  sorry

end NUMINAMATH_CALUDE_broomstick_race_orderings_l2751_275178


namespace NUMINAMATH_CALUDE_sin_80_cos_20_minus_cos_80_sin_20_l2751_275160

theorem sin_80_cos_20_minus_cos_80_sin_20 : 
  Real.sin (80 * π / 180) * Real.cos (20 * π / 180) - 
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_80_cos_20_minus_cos_80_sin_20_l2751_275160


namespace NUMINAMATH_CALUDE_zero_success_probability_l2751_275141

/-- Probability of success in a single trial -/
def p : ℚ := 2 / 7

/-- Number of trials -/
def n : ℕ := 7

/-- Probability of exactly k successes in n Bernoulli trials with success probability p -/
def binomialProbability (k : ℕ) : ℚ :=
  (n.choose k) * p ^ k * (1 - p) ^ (n - k)

/-- Theorem: The probability of 0 successes in 7 Bernoulli trials 
    with success probability 2/7 is equal to (5/7)^7 -/
theorem zero_success_probability : 
  binomialProbability 0 = (5 / 7) ^ 7 := by sorry

end NUMINAMATH_CALUDE_zero_success_probability_l2751_275141


namespace NUMINAMATH_CALUDE_floor_ceil_sqrt_50_sum_squares_l2751_275130

theorem floor_ceil_sqrt_50_sum_squares : ⌊Real.sqrt 50⌋^2 + ⌈Real.sqrt 50⌉^2 = 113 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sqrt_50_sum_squares_l2751_275130


namespace NUMINAMATH_CALUDE_smallest_period_of_given_functions_l2751_275123

open Real

noncomputable def f1 (x : ℝ) := -cos x
noncomputable def f2 (x : ℝ) := abs (sin x)
noncomputable def f3 (x : ℝ) := cos (2 * x)
noncomputable def f4 (x : ℝ) := tan (2 * x - π / 4)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ p > 0 ∧ ∀ q, is_periodic f q ∧ q > 0 → p ≤ q

theorem smallest_period_of_given_functions :
  smallest_positive_period f2 π ∧
  smallest_positive_period f3 π ∧
  (∀ p, smallest_positive_period f1 p → p > π) ∧
  (∀ p, smallest_positive_period f4 p → p > π) :=
sorry

end NUMINAMATH_CALUDE_smallest_period_of_given_functions_l2751_275123


namespace NUMINAMATH_CALUDE_max_boxes_is_240_l2751_275183

/-- Represents the weight of a box in pounds -/
inductive BoxWeight
  | light : BoxWeight  -- 10-pound box
  | heavy : BoxWeight  -- 40-pound box

/-- Calculates the total weight of a pair of boxes (one light, one heavy) -/
def pairWeight : ℕ := 50

/-- Represents the maximum weight capacity of a truck in pounds -/
def truckCapacity : ℕ := 2000

/-- Represents the number of trucks available for delivery -/
def numTrucks : ℕ := 3

/-- Calculates the maximum number of boxes that can be shipped in each delivery -/
def maxBoxesPerDelivery : ℕ := 
  (truckCapacity / pairWeight) * 2 * numTrucks

/-- Theorem stating that the maximum number of boxes that can be shipped in each delivery is 240 -/
theorem max_boxes_is_240 : maxBoxesPerDelivery = 240 := by
  sorry

end NUMINAMATH_CALUDE_max_boxes_is_240_l2751_275183


namespace NUMINAMATH_CALUDE_pizza_piece_cost_l2751_275136

/-- Represents the cost of pizzas and their division into pieces -/
structure PizzaPurchase where
  totalCost : ℕ        -- Total cost in dollars
  numPizzas : ℕ        -- Number of pizzas
  piecesPerPizza : ℕ   -- Number of pieces each pizza is cut into

/-- Calculates the cost per piece of pizza -/
def costPerPiece (purchase : PizzaPurchase) : ℚ :=
  (purchase.totalCost : ℚ) / (purchase.numPizzas * purchase.piecesPerPizza)

/-- Theorem: Given 4 pizzas cost $80 and each pizza is cut into 5 pieces, 
    the cost per piece is $4 -/
theorem pizza_piece_cost : 
  let purchase := PizzaPurchase.mk 80 4 5
  costPerPiece purchase = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_piece_cost_l2751_275136


namespace NUMINAMATH_CALUDE_joans_missed_games_l2751_275196

/-- Given that Joan's high school played 864 baseball games and she attended 395 games,
    prove that she missed 469 games. -/
theorem joans_missed_games (total_games : ℕ) (attended_games : ℕ)
  (h1 : total_games = 864)
  (h2 : attended_games = 395) :
  total_games - attended_games = 469 := by
sorry

end NUMINAMATH_CALUDE_joans_missed_games_l2751_275196


namespace NUMINAMATH_CALUDE_hall_people_count_l2751_275117

theorem hall_people_count (total_desks : ℕ) (occupied_desks : ℕ) (people : ℕ) : 
  total_desks = 72 →
  occupied_desks = 60 →
  people * 4 = occupied_desks * 5 →
  total_desks - occupied_desks = 12 →
  people = 75 := by
sorry

end NUMINAMATH_CALUDE_hall_people_count_l2751_275117


namespace NUMINAMATH_CALUDE_problem_solution_l2751_275163

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2751_275163


namespace NUMINAMATH_CALUDE_dihydrogen_monoxide_weight_is_18_016_l2751_275181

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of hydrogen atoms in a water molecule -/
def hydrogen_count : ℕ := 2

/-- The number of oxygen atoms in a water molecule -/
def oxygen_count : ℕ := 1

/-- The molecular weight of dihydrogen monoxide (H2O) in g/mol -/
def dihydrogen_monoxide_weight : ℝ := 
  hydrogen_count * hydrogen_weight + oxygen_count * oxygen_weight

/-- Theorem: The molecular weight of dihydrogen monoxide (H2O) is 18.016 g/mol -/
theorem dihydrogen_monoxide_weight_is_18_016 : 
  dihydrogen_monoxide_weight = 18.016 := by
  sorry

end NUMINAMATH_CALUDE_dihydrogen_monoxide_weight_is_18_016_l2751_275181


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2751_275172

-- Define sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | x ≤ 5}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2751_275172


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l2751_275150

/-- Given plane vectors a and b, where a is parallel to b, prove that 2a + 3b = (-4, -8) -/
theorem parallel_vectors_sum (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →  -- a is parallel to b
  (2 • a + 3 • b) = ![(-4 : ℝ), -8] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l2751_275150


namespace NUMINAMATH_CALUDE_mod_equivalence_2023_l2751_275195

theorem mod_equivalence_2023 : ∃! n : ℕ, n ≤ 11 ∧ n ≡ -2023 [ZMOD 12] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_2023_l2751_275195


namespace NUMINAMATH_CALUDE_endpoint_from_midpoint_and_one_endpoint_l2751_275134

/-- Given a line segment with midpoint (3, 4) and one endpoint at (0, -1), 
    the other endpoint is at (6, 9). -/
theorem endpoint_from_midpoint_and_one_endpoint :
  let midpoint : ℝ × ℝ := (3, 4)
  let endpoint1 : ℝ × ℝ := (0, -1)
  let endpoint2 : ℝ × ℝ := (6, 9)
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_endpoint_from_midpoint_and_one_endpoint_l2751_275134


namespace NUMINAMATH_CALUDE_chad_savings_theorem_l2751_275185

def calculate_savings (mowing_yards : ℝ) (birthday_holidays : ℝ) (video_games : ℝ) (odd_jobs : ℝ) : ℝ :=
  let total_earnings := mowing_yards + birthday_holidays + video_games + odd_jobs
  let tax_rate := 0.1
  let taxes := tax_rate * total_earnings
  let after_tax := total_earnings - taxes
  let mowing_savings := 0.5 * mowing_yards
  let birthday_savings := 0.3 * birthday_holidays
  let video_games_savings := 0.4 * video_games
  let odd_jobs_savings := 0.2 * odd_jobs
  mowing_savings + birthday_savings + video_games_savings + odd_jobs_savings

theorem chad_savings_theorem :
  calculate_savings 600 250 150 150 = 465 := by
  sorry

end NUMINAMATH_CALUDE_chad_savings_theorem_l2751_275185


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l2751_275168

/-- A geometric sequence with a₂ = 4 and a₄ = 8 has a₆ = 16 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_a2 : a 2 = 4) (h_a4 : a 4 = 8) : a 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l2751_275168


namespace NUMINAMATH_CALUDE_rowing_distance_with_tide_l2751_275120

/-- Represents the problem of a man rowing with and against the tide. -/
structure RowingProblem where
  /-- The speed of the man rowing in still water (km/h) -/
  manSpeed : ℝ
  /-- The speed of the tide (km/h) -/
  tideSpeed : ℝ
  /-- The distance traveled against the tide (km) -/
  distanceAgainstTide : ℝ
  /-- The time taken to travel against the tide (h) -/
  timeAgainstTide : ℝ
  /-- The time that would have been saved if the tide hadn't changed (h) -/
  timeSaved : ℝ

/-- Theorem stating that given the conditions of the rowing problem, 
    the distance the man can row with the help of the tide in 60 minutes is 5 km. -/
theorem rowing_distance_with_tide (p : RowingProblem) 
  (h1 : p.manSpeed - p.tideSpeed = p.distanceAgainstTide / p.timeAgainstTide)
  (h2 : p.manSpeed + p.tideSpeed = p.distanceAgainstTide / (p.timeAgainstTide - p.timeSaved))
  (h3 : p.distanceAgainstTide = 40)
  (h4 : p.timeAgainstTide = 10)
  (h5 : p.timeSaved = 2) :
  (p.manSpeed + p.tideSpeed) * 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_rowing_distance_with_tide_l2751_275120


namespace NUMINAMATH_CALUDE_m_range_l2751_275192

def f (x : ℝ) := x^3 + x

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → f (m * Real.sin θ) + f (1 - m) > 0) →
  m < 1 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2751_275192


namespace NUMINAMATH_CALUDE_candy_distribution_l2751_275161

theorem candy_distribution (adam james rubert : ℕ) 
  (h1 : rubert = 4 * james) 
  (h2 : james = 3 * adam) 
  (h3 : adam + james + rubert = 96) : 
  adam = 6 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l2751_275161


namespace NUMINAMATH_CALUDE_min_boxes_to_eliminate_l2751_275144

/-- The total number of boxes -/
def total_boxes : ℕ := 30

/-- The number of boxes containing at least $200,000 -/
def high_value_boxes : ℕ := 6

/-- The minimum number of boxes that must be eliminated -/
def boxes_to_eliminate : ℕ := 18

/-- Theorem stating that eliminating 18 boxes is the minimum required for a 50% chance of a high-value box -/
theorem min_boxes_to_eliminate :
  boxes_to_eliminate = total_boxes - 2 * high_value_boxes :=
sorry

end NUMINAMATH_CALUDE_min_boxes_to_eliminate_l2751_275144


namespace NUMINAMATH_CALUDE_average_of_series_l2751_275154

/-- The average value of the series 0², (2z)², (4z)², (8z)² is 21z² -/
theorem average_of_series (z : ℝ) : 
  (0^2 + (2*z)^2 + (4*z)^2 + (8*z)^2) / 4 = 21 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_series_l2751_275154


namespace NUMINAMATH_CALUDE_base_number_proof_l2751_275166

theorem base_number_proof (x : ℝ) : x^8 = 4^16 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l2751_275166


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2751_275171

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∀ x, x^2 + m*x + n = 0 ↔ ∃ y, y^2 + p*y + m = 0 ∧ x = 3*y) →
  n / p = 27 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2751_275171


namespace NUMINAMATH_CALUDE_dot_only_count_l2751_275175

/-- Represents an alphabet with dots and straight lines -/
structure Alphabet where
  total : ℕ
  both : ℕ
  line_only : ℕ
  has_dot_or_line : total = both + line_only + (total - (both + line_only))

/-- The number of letters with a dot but no straight line in the given alphabet -/
def letters_with_dot_only (α : Alphabet) : ℕ :=
  α.total - (α.both + α.line_only)

/-- Theorem stating the number of letters with a dot but no straight line -/
theorem dot_only_count (α : Alphabet) 
  (h1 : α.total = 80)
  (h2 : α.both = 28)
  (h3 : α.line_only = 47) :
  letters_with_dot_only α = 5 := by
  sorry

#check dot_only_count

end NUMINAMATH_CALUDE_dot_only_count_l2751_275175


namespace NUMINAMATH_CALUDE_room_width_is_four_meters_l2751_275129

/-- Proves that the width of a rectangular room is 4 meters given the specified conditions -/
theorem room_width_is_four_meters 
  (length : ℝ) 
  (cost_per_sqm : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : cost_per_sqm = 700)
  (h3 : total_cost = 15400) :
  ∃ (width : ℝ), width = 4 ∧ length * width * cost_per_sqm = total_cost :=
by sorry

end NUMINAMATH_CALUDE_room_width_is_four_meters_l2751_275129


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2751_275102

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a2 : a 2 = 3) 
  (h_sum : a 3 + a 4 = 9) : 
  a 1 * a 6 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2751_275102


namespace NUMINAMATH_CALUDE_replacement_cost_20_gyms_l2751_275116

/-- The cost to replace all cardio machines in multiple gyms -/
def total_replacement_cost (num_gyms : ℕ) (bike_cost : ℕ) : ℕ :=
  let treadmill_cost : ℕ := (3 * bike_cost) / 2
  let elliptical_cost : ℕ := 2 * treadmill_cost
  let gym_cost : ℕ := 10 * bike_cost + 5 * treadmill_cost + 5 * elliptical_cost
  num_gyms * gym_cost

/-- Theorem stating the total replacement cost for 20 gyms -/
theorem replacement_cost_20_gyms :
  total_replacement_cost 20 700 = 455000 := by
  sorry

end NUMINAMATH_CALUDE_replacement_cost_20_gyms_l2751_275116


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2751_275158

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (75 - b) + c / (55 - c) = 8) :
  6 / (30 - a) + 15 / (75 - b) + 11 / (55 - c) = 187 / 30 :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2751_275158


namespace NUMINAMATH_CALUDE_log_ratio_squared_equals_one_l2751_275153

theorem log_ratio_squared_equals_one (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (hx_neq_one : x ≠ 1) (hy_neq_one : y ≠ 1)
  (h_log : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
  (h_sum : x + y = 36) : 
  (Real.log (x / y) / Real.log 3)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_squared_equals_one_l2751_275153


namespace NUMINAMATH_CALUDE_factor_expression_l2751_275101

theorem factor_expression (y : ℝ) : 6*y*(y+2) + 15*(y+2) + 12 = 3*(2*y+5)*(y+2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2751_275101


namespace NUMINAMATH_CALUDE_weight_estimation_l2751_275122

-- Define the variables and constants
variable (x y : ℝ)
variable (x_sum y_sum : ℝ)
variable (b_hat : ℝ)
variable (n : ℕ)

-- Define the conditions
def conditions (x_sum y_sum b_hat : ℝ) (n : ℕ) : Prop :=
  x_sum = 1600 ∧ y_sum = 460 ∧ b_hat = 0.85 ∧ n = 10

-- Define the regression line equation
def regression_line (x b_hat a_hat : ℝ) : ℝ :=
  b_hat * x + a_hat

-- Theorem statement
theorem weight_estimation 
  (x_sum y_sum b_hat : ℝ) (n : ℕ) 
  (h : conditions x_sum y_sum b_hat n) : 
  ∃ a_hat : ℝ, regression_line 170 b_hat a_hat = 54.5 :=
sorry

end NUMINAMATH_CALUDE_weight_estimation_l2751_275122


namespace NUMINAMATH_CALUDE_car_travel_time_l2751_275156

-- Define the car's properties
def speed : Real := 50 -- miles per hour
def fuel_efficiency : Real := 30 -- miles per gallon
def tank_capacity : Real := 20 -- gallons
def fuel_used_fraction : Real := 0.4166666666666667 -- fraction of full tank used

-- Theorem statement
theorem car_travel_time :
  let fuel_used : Real := fuel_used_fraction * tank_capacity
  let distance_traveled : Real := fuel_used * fuel_efficiency
  let travel_time : Real := distance_traveled / speed
  travel_time = 5 := by sorry

end NUMINAMATH_CALUDE_car_travel_time_l2751_275156


namespace NUMINAMATH_CALUDE_nell_initial_cards_l2751_275184

/-- Nell's initial number of baseball cards -/
def initial_cards : ℕ := sorry

/-- Number of cards Nell gave to Jeff -/
def cards_given : ℕ := 28

/-- Number of cards Nell has left -/
def cards_left : ℕ := 276

/-- Theorem stating that Nell's initial number of cards was 304 -/
theorem nell_initial_cards : initial_cards = 304 := by
  sorry

end NUMINAMATH_CALUDE_nell_initial_cards_l2751_275184


namespace NUMINAMATH_CALUDE_journey_distance_l2751_275106

theorem journey_distance (speed : ℝ) (time : ℝ) 
  (h1 : (speed + 1/2) * (3/4 * time) = speed * time)
  (h2 : (speed - 1/2) * (time + 3) = speed * time)
  : speed * time = 9 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l2751_275106


namespace NUMINAMATH_CALUDE_sum_reciprocals_zero_implies_sum_diff_zero_l2751_275174

theorem sum_reciprocals_zero_implies_sum_diff_zero 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h : 1 / (a + 1) + 1 / (a - 1) + 1 / (b + 1) + 1 / (b - 1) = 0) : 
  a - 1 / a + b - 1 / b = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_zero_implies_sum_diff_zero_l2751_275174


namespace NUMINAMATH_CALUDE_max_rectangles_in_5x5_grid_seven_rectangles_fit_5x5_grid_l2751_275142

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the square grid -/
def Grid := ℕ × ℕ

/-- Check if a list of rectangles fits in the grid without overlap and covers it completely -/
def fits_grid (grid : Grid) (rectangles : List Rectangle) : Prop :=
  sorry

/-- The theorem stating that 7 is the maximum number of rectangles that can fit in a 5x5 grid -/
theorem max_rectangles_in_5x5_grid :
  ∀ (rectangles : List Rectangle),
    (∀ r ∈ rectangles, (r.width = 1 ∧ r.height = 4) ∨ (r.width = 1 ∧ r.height = 3)) →
    fits_grid (5, 5) rectangles →
    rectangles.length ≤ 7 :=
  sorry

/-- The theorem stating that 7 rectangles can indeed fit in a 5x5 grid -/
theorem seven_rectangles_fit_5x5_grid :
  ∃ (rectangles : List Rectangle),
    (∀ r ∈ rectangles, (r.width = 1 ∧ r.height = 4) ∨ (r.width = 1 ∧ r.height = 3)) ∧
    fits_grid (5, 5) rectangles ∧
    rectangles.length = 7 :=
  sorry

end NUMINAMATH_CALUDE_max_rectangles_in_5x5_grid_seven_rectangles_fit_5x5_grid_l2751_275142


namespace NUMINAMATH_CALUDE_madison_distance_l2751_275173

/-- Represents the distance between two locations on a map --/
structure MapDistance where
  inches : ℝ

/-- Represents a travel duration --/
structure TravelTime where
  hours : ℝ

/-- Represents a speed --/
structure Speed where
  mph : ℝ

/-- Represents a map scale --/
structure MapScale where
  inches_per_mile : ℝ

/-- Calculates the actual distance traveled given speed and time --/
def calculate_distance (speed : Speed) (time : TravelTime) : ℝ :=
  speed.mph * time.hours

/-- Calculates the map distance given actual distance and map scale --/
def calculate_map_distance (actual_distance : ℝ) (scale : MapScale) : MapDistance :=
  { inches := actual_distance * scale.inches_per_mile }

/-- The main theorem --/
theorem madison_distance (travel_time : TravelTime) (speed : Speed) (scale : MapScale) :
  travel_time.hours = 3.5 →
  speed.mph = 60 →
  scale.inches_per_mile = 0.023809523809523808 →
  (calculate_map_distance (calculate_distance speed travel_time) scale).inches = 5 := by
  sorry

end NUMINAMATH_CALUDE_madison_distance_l2751_275173


namespace NUMINAMATH_CALUDE_probability_gpa_at_least_3_6_l2751_275115

/-- Grade points for each letter grade -/
def gradePoints (grade : Char) : ℕ :=
  match grade with
  | 'A' => 4
  | 'B' => 3
  | 'C' => 2
  | 'D' => 1
  | _ => 0

/-- Calculate GPA given a list of grades -/
def calculateGPA (grades : List Char) : ℚ :=
  (grades.map gradePoints).sum / 5

/-- Probability of getting an A in English -/
def pEnglishA : ℚ := 1/4

/-- Probability of getting a B in English -/
def pEnglishB : ℚ := 1/2

/-- Probability of getting an A in History -/
def pHistoryA : ℚ := 2/5

/-- Probability of getting a B in History -/
def pHistoryB : ℚ := 1/2

/-- Theorem stating the probability of achieving a GPA of at least 3.6 -/
theorem probability_gpa_at_least_3_6 :
  let p := pEnglishA * pHistoryA + pEnglishA * pHistoryB + pEnglishB * pHistoryA
  p = 17/40 := by sorry

end NUMINAMATH_CALUDE_probability_gpa_at_least_3_6_l2751_275115


namespace NUMINAMATH_CALUDE_train_length_l2751_275124

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length. -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 72 * (5 / 18) → 
  crossing_time = 12.598992080633549 →
  bridge_length = 142 →
  train_speed * crossing_time - bridge_length = 110 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l2751_275124


namespace NUMINAMATH_CALUDE_revenue_decrease_percent_l2751_275186

theorem revenue_decrease_percent (T C : ℝ) (h1 : T > 0) (h2 : C > 0) : 
  let R := T * C
  let T_new := T * (1 - 0.20)
  let C_new := C * (1 + 0.15)
  let R_new := T_new * C_new
  (R - R_new) / R * 100 = 8 := by
sorry

end NUMINAMATH_CALUDE_revenue_decrease_percent_l2751_275186


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2751_275137

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2751_275137


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l2751_275135

theorem divisibility_equivalence (n : ℤ) : 
  let A := n % 1000
  let B := n / 1000
  let k := A - B
  (7 ∣ n ∨ 11 ∣ n ∨ 13 ∣ n) ↔ (7 ∣ k ∨ 11 ∣ k ∨ 13 ∣ k) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l2751_275135


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_squares_of_first_1013_odd_integers_l2751_275100

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def sum_of_squares (list : List ℕ) : ℕ :=
  list.map (fun x => x * x) |> List.sum

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_sum_of_squares_of_first_1013_odd_integers :
  units_digit (sum_of_squares (first_n_odd_integers 1013)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_squares_of_first_1013_odd_integers_l2751_275100


namespace NUMINAMATH_CALUDE_triangle_with_given_altitudes_exists_l2751_275121

theorem triangle_with_given_altitudes_exists (m_a m_b : ℝ) 
  (h1 : 0 < m_a) (h2 : 0 < m_b) (h3 : m_a ≤ m_b) :
  (∃ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    m_a = (2 * (a * b * c / (a + b + c))) / a ∧
    m_b = (2 * (a * b * c / (a + b + c))) / b ∧
    m_a + m_b = (2 * (a * b * c / (a + b + c))) / c) ↔
  (m_a / m_b)^2 + (m_a / m_b) > 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_given_altitudes_exists_l2751_275121


namespace NUMINAMATH_CALUDE_fraction_comparison_l2751_275169

theorem fraction_comparison (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2751_275169


namespace NUMINAMATH_CALUDE_zero_not_equivalent_to_intersection_l2751_275191

-- Define a real-valued function
def RealFunction := ℝ → ℝ

-- Define the zero of a function
def is_zero_of_function (f : RealFunction) (x : ℝ) : Prop := f x = 0

-- Define the intersection point of a function's graph and the x-axis
def is_intersection_with_x_axis (f : RealFunction) (x : ℝ) : Prop :=
  f x = 0 ∧ ∀ y : ℝ, y ≠ 0 → f x ≠ y

-- Theorem stating that these concepts are not equivalent
theorem zero_not_equivalent_to_intersection :
  ¬ (∀ (f : RealFunction) (x : ℝ), is_zero_of_function f x ↔ is_intersection_with_x_axis f x) :=
sorry

end NUMINAMATH_CALUDE_zero_not_equivalent_to_intersection_l2751_275191


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l2751_275133

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r ^ (n - 1)

theorem tenth_term_of_sequence : 
  let a : ℚ := 5
  let r : ℚ := 3/4
  geometric_sequence a r 10 = 98415/262144 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l2751_275133


namespace NUMINAMATH_CALUDE_remainder_of_m_l2751_275103

theorem remainder_of_m (m : ℕ) (h1 : m^2 % 7 = 1) (h2 : m^3 % 7 = 6) : m % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_m_l2751_275103


namespace NUMINAMATH_CALUDE_bookstore_purchasing_plans_l2751_275159

theorem bookstore_purchasing_plans :
  let n : ℕ := 3 -- number of books
  let select_at_least_one (k : ℕ) : ℕ := 
    Finset.card (Finset.powerset (Finset.range k) \ {∅})
  select_at_least_one n = 7 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_purchasing_plans_l2751_275159


namespace NUMINAMATH_CALUDE_circle_properties_l2751_275148

/-- The circle C is defined by the equation (x+1)^2 + (y-2)^2 = 4 -/
def C : Set (ℝ × ℝ) := {p | (p.1 + 1)^2 + (p.2 - 2)^2 = 4}

/-- The center of circle C -/
def center : ℝ × ℝ := (-1, 2)

/-- The radius of circle C -/
def radius : ℝ := 2

theorem circle_properties :
  ∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l2751_275148


namespace NUMINAMATH_CALUDE_max_value_theorem_l2751_275132

theorem max_value_theorem (x y z : ℝ) (h : x^2 + y^2 + z^2 = 5) :
  ∃ (max : ℝ), (∀ (a b c : ℝ), a^2 + b^2 + c^2 = 5 → a + 2*b + 3*c ≤ max) ∧
  (∃ (x₀ y₀ z₀ : ℝ), x₀^2 + y₀^2 + z₀^2 = 5 ∧ x₀ + 2*y₀ + 3*z₀ = max) ∧
  max = Real.sqrt 70 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2751_275132


namespace NUMINAMATH_CALUDE_simon_is_10_years_old_l2751_275155

/-- Simon's age given Alvin's age and their relationship -/
def simon_age (alvin_age : ℕ) (age_difference : ℕ) : ℕ :=
  alvin_age / 2 - age_difference

theorem simon_is_10_years_old :
  simon_age 30 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_simon_is_10_years_old_l2751_275155


namespace NUMINAMATH_CALUDE_alpha_value_l2751_275162

theorem alpha_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1)
  (h_min : ∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 9/y ≥ 1/m + 9/n)
  (α : ℝ) (h_curve : m^α = 2/3 * n) : α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l2751_275162


namespace NUMINAMATH_CALUDE_additive_implies_linear_l2751_275113

/-- A function satisfying the given additive property -/
def AdditiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

/-- A linear function with zero intercept -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x, f x = k * x

/-- If a function satisfies the additive property, then it is a linear function with zero intercept -/
theorem additive_implies_linear (f : ℝ → ℝ) (h : AdditiveFunction f) : LinearFunction f := by
  sorry

end NUMINAMATH_CALUDE_additive_implies_linear_l2751_275113


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_512_l2751_275114

theorem sqrt_expression_equals_512 : 
  Real.sqrt ((16^12 + 2^24) / (16^5 + 2^30)) = 512 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_512_l2751_275114


namespace NUMINAMATH_CALUDE_reflection_direction_vector_l2751_275105

/-- Given a particle moving in a plane along direction u = (1,2) and reflecting off a line l
    to move in direction v = (-2,1) according to the optical principle,
    one possible direction vector of line l is ω = (1,-3). -/
theorem reflection_direction_vector :
  let u : ℝ × ℝ := (1, 2)
  let v : ℝ × ℝ := (-2, 1)
  ∃ ω : ℝ × ℝ, ω = (1, -3) ∧
    (∀ k : ℝ, (k - 2) / (1 + 2*k) = (-1/2 - k) / (1 - 1/2*k) → k = -3) ∧
    (∀ θ₁ θ₂ : ℝ, θ₁ = θ₂ → 
      (u.2 / u.1 - ω.2 / ω.1) / (1 + (u.2 / u.1) * (ω.2 / ω.1)) =
      (v.2 / v.1 - ω.2 / ω.1) / (1 + (v.2 / v.1) * (ω.2 / ω.1))) :=
by sorry

end NUMINAMATH_CALUDE_reflection_direction_vector_l2751_275105


namespace NUMINAMATH_CALUDE_f1_properties_f2_properties_f3_properties_f4_properties_l2751_275165

-- Function 1: y = 4 - x^2 for |x| ≤ 2
def f1 (x : ℝ) := 4 - x^2

-- Function 2: y = 0.5(x^2 + x|x| + 4)
def f2 (x : ℝ) := 0.5 * (x^2 + x * |x| + 4)

-- Function 3: y = (x^3 - x) / |x|
noncomputable def f3 (x : ℝ) := (x^3 - x) / |x|

-- Function 4: y = (x - 2)|x|
def f4 (x : ℝ) := (x - 2) * |x|

-- Theorem for function 1
theorem f1_properties (x : ℝ) (h : |x| ≤ 2) :
  f1 x ≤ 4 ∧ f1 0 = 4 ∧ f1 2 = f1 (-2) := by sorry

-- Theorem for function 2
theorem f2_properties (x : ℝ) :
  (x ≥ 0 → f2 x = x^2 + 2) ∧ (x < 0 → f2 x = 2) := by sorry

-- Theorem for function 3
theorem f3_properties (x : ℝ) (h : x ≠ 0) :
  (x > 0 → f3 x = x^2 - 1) ∧ (x < 0 → f3 x = -x^2 + 1) := by sorry

-- Theorem for function 4
theorem f4_properties (x : ℝ) :
  (x ≥ 0 → f4 x = x^2 - 2*x) ∧ (x < 0 → f4 x = -x^2 + 2*x) := by sorry

end NUMINAMATH_CALUDE_f1_properties_f2_properties_f3_properties_f4_properties_l2751_275165


namespace NUMINAMATH_CALUDE_fixed_point_on_graph_l2751_275194

theorem fixed_point_on_graph (k : ℝ) : 
  let f := fun (x : ℝ) => 5 * x^2 + k * x - 3 * k
  f 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_graph_l2751_275194


namespace NUMINAMATH_CALUDE_edward_candy_purchase_l2751_275167

/-- The number of candy pieces Edward can buy given his tickets and the candy cost --/
theorem edward_candy_purchase (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) : 
  whack_a_mole_tickets = 3 →
  skee_ball_tickets = 5 →
  candy_cost = 4 →
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_edward_candy_purchase_l2751_275167


namespace NUMINAMATH_CALUDE_subset_condition_1_subset_condition_2_l2751_275127

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

-- Theorem for part 1
theorem subset_condition_1 : A ⊆ B a → a = -2 := by sorry

-- Theorem for part 2
theorem subset_condition_2 : B a ⊆ A → a ≥ 4 ∨ a < -4 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_subset_condition_1_subset_condition_2_l2751_275127


namespace NUMINAMATH_CALUDE_expand_product_l2751_275107

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2751_275107


namespace NUMINAMATH_CALUDE_log_equation_solution_l2751_275111

theorem log_equation_solution (x : ℝ) (hx : x > 0) :
  Real.log x / Real.log 3 + Real.log 3 / Real.log x - 2 * (Real.log x / Real.log 3) * (Real.log 3 / Real.log x) = 1/2 ↔ 
  x = Real.sqrt 3 ∨ x = 9 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2751_275111


namespace NUMINAMATH_CALUDE_money_distribution_l2751_275118

/-- Given three people A, B, and C with some money, prove that A and C together have 300 Rs. -/
theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 400)
  (bc_sum : B + C = 150)
  (c_amount : C = 50) : 
  A + C = 300 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l2751_275118


namespace NUMINAMATH_CALUDE_f_2011_equals_sin_l2751_275180

noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => Real.cos
  | n + 1 => deriv (f n)

theorem f_2011_equals_sin : f 2011 = Real.sin := by sorry

end NUMINAMATH_CALUDE_f_2011_equals_sin_l2751_275180


namespace NUMINAMATH_CALUDE_divisibility_of_polynomial_l2751_275187

theorem divisibility_of_polynomial (n : ℤ) : 
  (120 : ℤ) ∣ (n^5 - 5*n^3 + 4*n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_polynomial_l2751_275187


namespace NUMINAMATH_CALUDE_inverse_sum_mod_17_l2751_275147

theorem inverse_sum_mod_17 : 
  (∃ x y : ℤ, (7 * x) % 17 = 1 ∧ (7 * y) % 17 = x % 17 ∧ (x + y) % 17 = 13) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_17_l2751_275147


namespace NUMINAMATH_CALUDE_min_abs_z_on_locus_l2751_275104

theorem min_abs_z_on_locus (z : ℂ) (h : Complex.abs (z - (0 : ℂ) + 4*I) + Complex.abs (z - 5) = 7) : 
  ∃ (w : ℂ), Complex.abs (w - (0 : ℂ) + 4*I) + Complex.abs (w - 5) = 7 ∧ 
  (∀ (v : ℂ), Complex.abs (v - (0 : ℂ) + 4*I) + Complex.abs (v - 5) = 7 → Complex.abs w ≤ Complex.abs v) ∧
  Complex.abs w = 20 / 7 :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_on_locus_l2751_275104


namespace NUMINAMATH_CALUDE_find_certain_number_l2751_275164

theorem find_certain_number : ∃ x : ℤ, x - 5 = 4 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_find_certain_number_l2751_275164


namespace NUMINAMATH_CALUDE_double_real_value_interest_rate_l2751_275108

/-- Proves that the given formula for the annual compound interest rate 
    results in doubling the real value of an initial sum after 22 years, 
    considering inflation and taxes. -/
theorem double_real_value_interest_rate 
  (X : ℝ) -- Annual inflation rate
  (Y : ℝ) -- Annual tax rate on earned interest
  (r : ℝ) -- Annual compound interest rate
  (h_X : X > 0)
  (h_Y : 0 ≤ Y ∧ Y < 1)
  (h_r : r = ((2 * (1 + X)) ^ (1 / 22) - 1) / (1 - Y)) :
  (∀ P : ℝ, P > 0 → 
    P * (1 + r * (1 - Y))^22 / (1 + X)^22 = 2 * P) :=
sorry

end NUMINAMATH_CALUDE_double_real_value_interest_rate_l2751_275108


namespace NUMINAMATH_CALUDE_min_obtuse_angles_convex_octagon_min_obtuse_angles_convex_octagon_proof_l2751_275128

/-- The minimum number of obtuse interior angles in a convex octagon -/
theorem min_obtuse_angles_convex_octagon : ℕ :=
  let exterior_angles : ℕ := 8
  let sum_exterior_angles : ℕ := 360
  5

/-- Proof of the minimum number of obtuse interior angles in a convex octagon -/
theorem min_obtuse_angles_convex_octagon_proof :
  min_obtuse_angles_convex_octagon = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_obtuse_angles_convex_octagon_min_obtuse_angles_convex_octagon_proof_l2751_275128


namespace NUMINAMATH_CALUDE_total_points_after_perfect_games_l2751_275146

/-- The number of points in a perfect score -/
def perfect_score : ℕ := 21

/-- The number of perfect games played -/
def games_played : ℕ := 11

/-- Theorem: The total points scored after playing 11 perfect games,
    where a perfect score is 21 points, is equal to 231 points. -/
theorem total_points_after_perfect_games :
  perfect_score * games_played = 231 := by
  sorry

end NUMINAMATH_CALUDE_total_points_after_perfect_games_l2751_275146


namespace NUMINAMATH_CALUDE_line_plane_relationship_l2751_275182

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (α : Plane) 
  (h1 : parallel_lines a b) 
  (h2 : parallel_line_plane a α) : 
  parallel_line_plane b α ∨ line_in_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l2751_275182


namespace NUMINAMATH_CALUDE_system_solution_l2751_275189

theorem system_solution : 
  ∃ (x y : ℚ), (4 * x - 3 * y = -2) ∧ (9 * x + 5 * y = 9) ∧ (x = 17/47) ∧ (y = 54/47) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2751_275189


namespace NUMINAMATH_CALUDE_stratified_sampling_is_most_suitable_l2751_275131

structure Population where
  male : ℕ
  female : ℕ

structure Sample where
  male : ℕ
  female : ℕ

def isStratifiedSampling (pop : Population) (samp : Sample) : Prop :=
  (pop.male : ℚ) / (pop.female : ℚ) = (samp.male : ℚ) / (samp.female : ℚ)

def isMostSuitableMethod (method : String) (pop : Population) (samp : Sample) : Prop :=
  method = "Stratified sampling" ∧ isStratifiedSampling pop samp

theorem stratified_sampling_is_most_suitable :
  let pop : Population := { male := 500, female := 400 }
  let samp : Sample := { male := 25, female := 20 }
  isMostSuitableMethod "Stratified sampling" pop samp :=
by
  sorry

#check stratified_sampling_is_most_suitable

end NUMINAMATH_CALUDE_stratified_sampling_is_most_suitable_l2751_275131


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2751_275151

theorem imaginary_part_of_complex_fraction : Complex.im ((3 * Complex.I + 4) / (1 + 2 * Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2751_275151


namespace NUMINAMATH_CALUDE_negative_polynomial_count_l2751_275177

theorem negative_polynomial_count : 
  ∃ (S : Finset ℤ), (∀ x ∈ S, x^5 - 51*x^3 + 50*x < 0) ∧ 
                    (∀ x : ℤ, x^5 - 51*x^3 + 50*x < 0 → x ∈ S) ∧ 
                    Finset.card S = 12 :=
by sorry

end NUMINAMATH_CALUDE_negative_polynomial_count_l2751_275177


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2751_275110

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 7*x - 5 = 0 ↔ (x + 7/2)^2 = 69/4 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2751_275110


namespace NUMINAMATH_CALUDE_reading_growth_rate_l2751_275143

theorem reading_growth_rate (initial_amount final_amount : ℝ) (growth_period : ℕ) (x : ℝ) :
  initial_amount = 1 →
  final_amount = 1.21 →
  growth_period = 2 →
  final_amount = initial_amount * (1 + x)^growth_period →
  100 * (1 + x)^2 = 121 :=
by sorry

end NUMINAMATH_CALUDE_reading_growth_rate_l2751_275143


namespace NUMINAMATH_CALUDE_car_sale_profit_l2751_275198

theorem car_sale_profit (P : ℝ) (h : P > 0) :
  let discount_rate := 0.2
  let profit_rate := 0.28000000000000004
  let purchase_price := P * (1 - discount_rate)
  let selling_price := P * (1 + profit_rate)
  let increase_rate := (selling_price - purchase_price) / purchase_price
  increase_rate = 0.6 := by sorry

end NUMINAMATH_CALUDE_car_sale_profit_l2751_275198


namespace NUMINAMATH_CALUDE_coin_order_correct_l2751_275138

/-- Represents the set of coins --/
inductive Coin : Type
  | A | B | C | D | E | F

/-- Defines the covering relation between coins --/
def covers (x y : Coin) : Prop := sorry

/-- The correct order of coins from top to bottom --/
def correct_order : List Coin := [Coin.F, Coin.D, Coin.A, Coin.E, Coin.B, Coin.C]

/-- Theorem stating that the given order is correct based on the covering relations --/
theorem coin_order_correct :
  (∀ x : Coin, ¬covers x Coin.F) ∧
  (covers Coin.F Coin.D) ∧
  (covers Coin.D Coin.B) ∧ (covers Coin.D Coin.C) ∧ (covers Coin.D Coin.E) ∧
  (covers Coin.D Coin.A) ∧ (covers Coin.A Coin.B) ∧ (covers Coin.A Coin.C) ∧
  (covers Coin.D Coin.E) ∧ (covers Coin.E Coin.C) ∧
  (covers Coin.D Coin.B) ∧ (covers Coin.A Coin.B) ∧ (covers Coin.E Coin.B) ∧ (covers Coin.B Coin.C) ∧
  (∀ x : Coin, x ≠ Coin.C → covers x Coin.C) →
  correct_order = [Coin.F, Coin.D, Coin.A, Coin.E, Coin.B, Coin.C] :=
by sorry

end NUMINAMATH_CALUDE_coin_order_correct_l2751_275138


namespace NUMINAMATH_CALUDE_periodic_function_from_T_property_l2751_275139

-- Define the "T property" for a function
def has_T_property (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T ≠ 0 ∧ ∀ x : ℝ, (deriv f) (x + T) = (deriv f) x

-- Main theorem
theorem periodic_function_from_T_property (f : ℝ → ℝ) (T M : ℝ) 
  (hf : Continuous f) 
  (hT : has_T_property f T) 
  (hM : ∀ x : ℝ, |f x| < M) :
  ∀ x : ℝ, f (x + T) = f x :=
sorry

end NUMINAMATH_CALUDE_periodic_function_from_T_property_l2751_275139


namespace NUMINAMATH_CALUDE_increasing_function_property_l2751_275119

theorem increasing_function_property (f : ℝ → ℝ) (a b : ℝ)
  (h_increasing : ∀ x y : ℝ, x ≤ y → f x ≤ f y) :
  (a + b ≥ 0 ↔ f a + f b ≥ f (-a) + f (-b)) := by
sorry

end NUMINAMATH_CALUDE_increasing_function_property_l2751_275119


namespace NUMINAMATH_CALUDE_difference_C_D_l2751_275190

def C : ℤ := (Finset.range 20).sum (fun i => (2*i + 2) * (2*i + 3)) + 42

def D : ℤ := 2 + (Finset.range 20).sum (fun i => (2*i + 3) * (2*i + 4))

theorem difference_C_D : |C - D| = 400 := by sorry

end NUMINAMATH_CALUDE_difference_C_D_l2751_275190


namespace NUMINAMATH_CALUDE_solve_equation_l2751_275188

theorem solve_equation (A : ℝ) : 3 + A = 4 → A = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2751_275188


namespace NUMINAMATH_CALUDE_bake_sale_total_l2751_275176

theorem bake_sale_total (cookies : ℕ) (brownies : ℕ) : 
  cookies = 48 → 
  brownies * 6 = cookies * 7 →
  cookies + brownies = 104 := by
sorry

end NUMINAMATH_CALUDE_bake_sale_total_l2751_275176


namespace NUMINAMATH_CALUDE_solution_set_g_range_of_m_l2751_275145

-- Define the functions f and g
def f (x : ℝ) := x^2 - 2*x - 8
def g (x : ℝ) := 2*x^2 - 4*x - 16

-- Theorem for the solution set of g(x) < 0
theorem solution_set_g (x : ℝ) : g x < 0 ↔ -2 < x ∧ x < 4 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∀ x > 1, f x ≥ (m + 2)*x - m - 15) → m ≤ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_g_range_of_m_l2751_275145


namespace NUMINAMATH_CALUDE_horner_method_operations_l2751_275140

-- Define the polynomial
def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

-- Define Horner's method for this specific polynomial
def horner_method (x : ℝ) : ℝ := ((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1

-- Theorem statement
theorem horner_method_operations :
  ∃ (mult_ops add_ops : ℕ),
    (∀ x : ℝ, f x = horner_method x) ∧
    mult_ops = 5 ∧
    add_ops = 5 :=
sorry

end NUMINAMATH_CALUDE_horner_method_operations_l2751_275140


namespace NUMINAMATH_CALUDE_water_speed_swimming_problem_l2751_275109

/-- Proves that the speed of water is 2 km/h given the conditions of the swimming problem. -/
theorem water_speed_swimming_problem (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) (water_speed : ℝ) :
  still_water_speed = 4 →
  distance = 12 →
  time = 6 →
  distance = (still_water_speed - water_speed) * time →
  water_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_speed_swimming_problem_l2751_275109
