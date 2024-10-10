import Mathlib

namespace S_is_finite_l132_13234

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Number of positive divisors function -/
def tau (n : ℕ) : ℕ := sorry

/-- The set of positive integers satisfying the inequality -/
def S : Set ℕ := {n : ℕ | n > 0 ∧ phi n * tau n ≥ Real.sqrt (n^3 / 3)}

/-- Theorem stating that S is finite -/
theorem S_is_finite : Set.Finite S := by sorry

end S_is_finite_l132_13234


namespace candy_distribution_theorem_l132_13254

/-- Represents the number of candy bars of each type --/
structure CandyBars where
  chocolate : ℕ
  caramel : ℕ
  nougat : ℕ

/-- Represents the ratio of candy bars in a bag --/
structure BagRatio where
  chocolate : ℕ
  caramel : ℕ
  nougat : ℕ

/-- Checks if the given ratio is valid for the total number of candy bars --/
def isValidRatio (total : CandyBars) (ratio : BagRatio) (bags : ℕ) : Prop :=
  total.chocolate = ratio.chocolate * bags ∧
  total.caramel = ratio.caramel * bags ∧
  total.nougat = ratio.nougat * bags

/-- The main theorem to be proved --/
theorem candy_distribution_theorem (total : CandyBars) 
  (h1 : total.chocolate = 12) 
  (h2 : total.caramel = 18) 
  (h3 : total.nougat = 15) :
  ∃ (ratio : BagRatio) (bags : ℕ), 
    bags = 5 ∧ 
    ratio.chocolate = 2 ∧ 
    ratio.caramel = 3 ∧ 
    ratio.nougat = 3 ∧
    isValidRatio total ratio bags ∧
    ∀ (other_ratio : BagRatio) (other_bags : ℕ), 
      isValidRatio total other_ratio other_bags → other_bags ≤ bags :=
by
  sorry

end candy_distribution_theorem_l132_13254


namespace leahs_coins_value_l132_13260

theorem leahs_coins_value :
  ∀ (p n : ℕ),
  p + n = 18 →
  n + 2 = p →
  5 * n + p = 50 :=
by
  sorry

end leahs_coins_value_l132_13260


namespace cos_2α_value_l132_13296

theorem cos_2α_value (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α - π/4) = 1/4) : 
  Real.cos (2*α) = -(Real.sqrt 15)/8 := by
sorry

end cos_2α_value_l132_13296


namespace positive_integer_power_equality_l132_13235

theorem positive_integer_power_equality (a b : ℕ+) :
  a ^ b.val = b ^ (a.val ^ 2) ↔ (a, b) = (1, 1) ∨ (a, b) = (2, 16) ∨ (a, b) = (3, 27) := by
  sorry

end positive_integer_power_equality_l132_13235


namespace solve_equation_l132_13258

theorem solve_equation (n m x : ℚ) 
  (h1 : (7 : ℚ) / 8 = n / 96)
  (h2 : (7 : ℚ) / 8 = (m + n) / 112)
  (h3 : (7 : ℚ) / 8 = (x - m) / 144) : 
  x = 140 := by sorry

end solve_equation_l132_13258


namespace dance_steps_time_l132_13272

def time_step1 : ℕ := 30

def time_step2 (t1 : ℕ) : ℕ := t1 / 2

def time_step3 (t1 t2 : ℕ) : ℕ := t1 + t2

def total_time (t1 t2 t3 : ℕ) : ℕ := t1 + t2 + t3

theorem dance_steps_time :
  total_time time_step1 (time_step2 time_step1) (time_step3 time_step1 (time_step2 time_step1)) = 90 :=
by sorry

end dance_steps_time_l132_13272


namespace aluminum_decoration_problem_l132_13233

def available_lengths : List ℕ := [3, 6, 9, 12, 15, 19, 21, 30]

def is_valid_combination (combination : List ℕ) : Prop :=
  combination.all (· ∈ available_lengths) ∧ combination.sum = 50

theorem aluminum_decoration_problem :
  ∀ combination : List ℕ,
    is_valid_combination combination ↔
      combination = [19, 19, 12] ∨ combination = [19, 19] :=
by sorry

end aluminum_decoration_problem_l132_13233


namespace expression_evaluation_l132_13277

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(2*x)) / (y^(2*y) * x^(2*x)) = (x/y)^(2*(y-x)) := by
  sorry

end expression_evaluation_l132_13277


namespace cross_product_of_a_and_b_l132_13248

def a : ℝ × ℝ × ℝ := (3, 4, -5)
def b : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (u₁, u₂, u₃) := u
  let (v₁, v₂, v₃) := v
  (u₂ * v₃ - u₃ * v₂, u₃ * v₁ - u₁ * v₃, u₁ * v₂ - u₂ * v₁)

theorem cross_product_of_a_and_b :
  cross_product a b = (11, -22, -11) := by
  sorry

end cross_product_of_a_and_b_l132_13248


namespace cyclist_speed_problem_l132_13204

/-- The speed of cyclist C in miles per hour -/
def speed_C : ℝ := 10

/-- The speed of cyclist D in miles per hour -/
def speed_D : ℝ := speed_C + 5

/-- The distance to the town in miles -/
def distance_to_town : ℝ := 90

/-- The distance from the meeting point to the town in miles -/
def distance_meeting_to_town : ℝ := 18

theorem cyclist_speed_problem :
  /- Given conditions -/
  (speed_D = speed_C + 5) →
  (distance_to_town = 90) →
  (distance_meeting_to_town = 18) →
  /- The time taken by C to reach the meeting point equals
     the time taken by D to reach the town and return to the meeting point -/
  ((distance_to_town - distance_meeting_to_town) / speed_C =
   (distance_to_town + distance_meeting_to_town) / speed_D) →
  /- Conclusion: The speed of cyclist C is 10 mph -/
  speed_C = 10 := by
  sorry

end cyclist_speed_problem_l132_13204


namespace ball_ratio_proof_l132_13251

/-- Proves that the ratio of blue balls to red balls is 16:5 given the initial conditions --/
theorem ball_ratio_proof (initial_red : ℕ) (lost_red : ℕ) (yellow : ℕ) (total : ℕ) :
  initial_red = 16 →
  lost_red = 6 →
  yellow = 32 →
  total = 74 →
  ∃ (blue : ℕ), blue * 5 = (initial_red - lost_red) * 16 ∧ 
                blue + (initial_red - lost_red) + yellow = total :=
by sorry

end ball_ratio_proof_l132_13251


namespace max_min_value_l132_13230

def f (x y : ℝ) : ℝ := x^3 + (y-4)*x^2 + (y^2-4*y+4)*x + (y^3-4*y^2+4*y)

theorem max_min_value (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) 
  (h1 : f a b = f b c) (h2 : f b c = f c a) : 
  ∃ (m : ℝ), m = 1 ∧ 
  ∀ (x y z : ℝ), x ≠ y → y ≠ z → z ≠ x → f x y = f y z → f y z = f z x →
  min (x^4 - 4*x^3 + 4*x^2) (min (y^4 - 4*y^3 + 4*y^2) (z^4 - 4*z^3 + 4*z^2)) ≤ m :=
by sorry

end max_min_value_l132_13230


namespace polynomial_equality_l132_13295

/-- Given a polynomial q(x) satisfying the equation
    q(x) + (2x^6 + 4x^4 + 5x^3 + 11x) = (10x^4 + 30x^3 + 40x^2 + 8x + 3),
    prove that q(x) = -2x^6 + 6x^4 + 25x^3 + 40x^2 - 3x + 3 -/
theorem polynomial_equality (q : ℝ → ℝ) :
  (∀ x, q x + (2 * x^6 + 4 * x^4 + 5 * x^3 + 11 * x) = 
       (10 * x^4 + 30 * x^3 + 40 * x^2 + 8 * x + 3)) →
  (∀ x, q x = -2 * x^6 + 6 * x^4 + 25 * x^3 + 40 * x^2 - 3 * x + 3) := by
sorry

end polynomial_equality_l132_13295


namespace sin_pi_minus_alpha_l132_13270

theorem sin_pi_minus_alpha (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (α + π / 3) = 3 / 5) : 
  Real.sin (π - α) = (3 + 4 * Real.sqrt 3) / 10 := by
  sorry

end sin_pi_minus_alpha_l132_13270


namespace boat_trip_distance_l132_13202

/-- Proves that given a boat with speed 9 kmph in standing water, a stream with speed 1.5 kmph,
    and a round trip time of 48 hours, the distance to the destination is 210 km. -/
theorem boat_trip_distance (boat_speed : ℝ) (stream_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  boat_speed = 9 →
  stream_speed = 1.5 →
  total_time = 48 →
  distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = total_time →
  distance = 210 := by
  sorry

end boat_trip_distance_l132_13202


namespace no_solution_to_system_l132_13288

theorem no_solution_to_system :
  ¬∃ (x y z : ℝ), (x^2 - 2*y + 2 = 0) ∧ (y^2 - 4*z + 3 = 0) ∧ (z^2 + 4*x + 4 = 0) := by
sorry

end no_solution_to_system_l132_13288


namespace sufficient_necessary_but_not_sufficient_l132_13224

-- Define propositions p and q
variable (p q : Prop)

-- Define what it means for p to be a sufficient condition for q
def sufficient (p q : Prop) : Prop := p → q

-- Define what it means for p to be a necessary and sufficient condition for q
def necessary_and_sufficient (p q : Prop) : Prop := p ↔ q

-- Theorem stating that "p is a sufficient condition for q" is a necessary but not sufficient condition for "p is a necessary and sufficient condition for q"
theorem sufficient_necessary_but_not_sufficient :
  (∀ p q, necessary_and_sufficient p q → sufficient p q) ∧
  ¬(∀ p q, sufficient p q → necessary_and_sufficient p q) :=
sorry

end sufficient_necessary_but_not_sufficient_l132_13224


namespace same_color_probability_l132_13220

/-- The probability of drawing two balls of the same color from a bag with 2 red and 2 white balls, with replacement -/
theorem same_color_probability (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 2 →
  white_balls = 2 →
  (red_balls : ℚ) / total_balls * (red_balls : ℚ) / total_balls +
  (white_balls : ℚ) / total_balls * (white_balls : ℚ) / total_balls = 1 / 2 :=
by sorry

end same_color_probability_l132_13220


namespace hyperbola_iff_m_negative_l132_13255

/-- A conic section represented by the equation x^2 + my^2 = m -/
structure ConicSection (m : ℝ) where
  x : ℝ
  y : ℝ
  eq : x^2 + m * y^2 = m

/-- Predicate to determine if a conic section is a hyperbola -/
def IsHyperbola (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 + m * y^2 = m

/-- Theorem stating that the equation x^2 + my^2 = m represents a hyperbola if and only if m < 0 -/
theorem hyperbola_iff_m_negative (m : ℝ) : IsHyperbola m ↔ m < 0 := by
  sorry

end hyperbola_iff_m_negative_l132_13255


namespace football_tournament_max_points_l132_13205

theorem football_tournament_max_points (num_teams : ℕ) (points_win : ℕ) (points_draw : ℕ) (points_loss : ℕ) :
  num_teams = 15 →
  points_win = 3 →
  points_draw = 1 →
  points_loss = 0 →
  ∃ (N : ℕ), N = 34 ∧ 
    (∀ (M : ℕ), (∃ (teams : Finset (Fin num_teams)), teams.card ≥ 6 ∧ 
      (∀ t ∈ teams, ∃ (score : ℕ), score ≥ M)) → M ≤ N) :=
by sorry

end football_tournament_max_points_l132_13205


namespace roots_expression_l132_13291

theorem roots_expression (a b : ℝ) (α β γ δ : ℝ) 
  (hα : α^2 - a*α - 1 = 0)
  (hβ : β^2 - a*β - 1 = 0)
  (hγ : γ^2 - b*γ - 1 = 0)
  (hδ : δ^2 - b*δ - 1 = 0) :
  (α - γ)^2 * (β - γ)^2 * (α + δ)^2 * (β + δ)^2 = (b^2 - a^2)^2 := by
  sorry

end roots_expression_l132_13291


namespace complex_number_with_conditions_l132_13218

theorem complex_number_with_conditions (z : ℂ) :
  (((1 : ℂ) + 2 * Complex.I) * z).im = 0 →
  Complex.abs z = Real.sqrt 5 →
  z = 1 - 2 * Complex.I ∨ z = -1 + 2 * Complex.I :=
by sorry

end complex_number_with_conditions_l132_13218


namespace notebook_purchase_solution_l132_13227

/-- Notebook types -/
inductive NotebookType
| A
| B
| C

/-- Represents the notebook purchase problem -/
structure NotebookPurchase where
  totalNotebooks : ℕ
  priceA : ℕ
  priceB : ℕ
  priceC : ℕ
  totalCostI : ℕ
  totalCostII : ℕ

/-- Represents the solution for part I -/
structure SolutionI where
  numA : ℕ
  numB : ℕ

/-- Represents the solution for part II -/
structure SolutionII where
  numA : ℕ

/-- The given notebook purchase problem -/
def problem : NotebookPurchase :=
  { totalNotebooks := 30
  , priceA := 11
  , priceB := 9
  , priceC := 6
  , totalCostI := 288
  , totalCostII := 188
  }

/-- Checks if the solution for part I is correct -/
def checkSolutionI (p : NotebookPurchase) (s : SolutionI) : Prop :=
  s.numA + s.numB = p.totalNotebooks ∧
  s.numA * p.priceA + s.numB * p.priceB = p.totalCostI

/-- Checks if the solution for part II is correct -/
def checkSolutionII (p : NotebookPurchase) (s : SolutionII) : Prop :=
  ∃ (numB numC : ℕ), 
    s.numA + numB + numC = p.totalNotebooks ∧
    s.numA * p.priceA + numB * p.priceB + numC * p.priceC = p.totalCostII

/-- The main theorem to prove -/
theorem notebook_purchase_solution :
  checkSolutionI problem { numA := 9, numB := 21 } ∧
  checkSolutionII problem { numA := 1 } :=
sorry


end notebook_purchase_solution_l132_13227


namespace sheepdog_roundup_percentage_l132_13207

theorem sheepdog_roundup_percentage 
  (total_sheep : ℕ) 
  (wandered_off : ℕ) 
  (in_pen : ℕ) 
  (h1 : wandered_off = total_sheep / 10)
  (h2 : wandered_off = 9)
  (h3 : in_pen = 81)
  (h4 : total_sheep = in_pen + wandered_off) :
  (in_pen : ℚ) / total_sheep * 100 = 90 := by
sorry

end sheepdog_roundup_percentage_l132_13207


namespace semicircle_radius_l132_13215

theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 162) :
  ∃ (radius : ℝ), perimeter = radius * (Real.pi + 2) ∧ radius = 162 / (Real.pi + 2) := by
  sorry

end semicircle_radius_l132_13215


namespace total_wrappers_collected_l132_13267

/-- The total number of wrappers collected by four friends is the sum of their individual collections. -/
theorem total_wrappers_collected
  (andy_wrappers : ℕ)
  (max_wrappers : ℕ)
  (zoe_wrappers : ℕ)
  (mia_wrappers : ℕ)
  (h1 : andy_wrappers = 34)
  (h2 : max_wrappers = 15)
  (h3 : zoe_wrappers = 25)
  (h4 : mia_wrappers = 19) :
  andy_wrappers + max_wrappers + zoe_wrappers + mia_wrappers = 93 := by
  sorry

end total_wrappers_collected_l132_13267


namespace product_of_repeating_decimal_and_eight_l132_13225

theorem product_of_repeating_decimal_and_eight :
  let s : ℚ := 456 / 999
  8 * s = 1216 / 333 := by sorry

end product_of_repeating_decimal_and_eight_l132_13225


namespace max_twin_prime_sum_200_l132_13268

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_twin_prime (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ q - p = 2

def max_twin_prime_sum : ℕ := 396

theorem max_twin_prime_sum_200 :
  ∀ p q : ℕ,
  p ≤ 200 → q ≤ 200 →
  is_twin_prime p q →
  p + q ≤ max_twin_prime_sum :=
sorry

end max_twin_prime_sum_200_l132_13268


namespace mickey_horses_per_week_l132_13252

/-- The number of horses Minnie mounts per day -/
def minnie_horses_per_day : ℕ := 7 + 3

/-- The number of horses Mickey mounts per day -/
def mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Mickey mounts 98 horses per week -/
theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end mickey_horses_per_week_l132_13252


namespace number_of_observations_l132_13236

theorem number_of_observations (original_mean corrected_mean wrong_value correct_value : ℝ) 
  (h1 : original_mean = 36)
  (h2 : wrong_value = 23)
  (h3 : correct_value = 48)
  (h4 : corrected_mean = 36.5) :
  ∃ n : ℕ, (n : ℝ) * original_mean + (correct_value - wrong_value) = n * corrected_mean ∧ n = 50 := by
  sorry

end number_of_observations_l132_13236


namespace equal_area_rectangles_width_l132_13257

/-- Given two rectangles of equal area, where one rectangle measures 15 inches by 24 inches
    and the other has a length of 8 inches, prove that the width of the second rectangle
    is 45 inches. -/
theorem equal_area_rectangles_width (area carol_length carol_width jordan_length : ℕ)
    (h1 : area = carol_length * carol_width)
    (h2 : carol_length = 15)
    (h3 : carol_width = 24)
    (h4 : jordan_length = 8) :
    area / jordan_length = 45 := by
  sorry

end equal_area_rectangles_width_l132_13257


namespace shortest_tangent_is_30_l132_13223

/-- Two circles in a 2D plane --/
structure TwoCircles where
  c1 : (ℝ × ℝ) → Prop
  c2 : (ℝ × ℝ) → Prop

/-- The given circles from the problem --/
def problem_circles : TwoCircles :=
  { c1 := λ (x, y) => (x - 12)^2 + y^2 = 25,
    c2 := λ (x, y) => (x + 18)^2 + y^2 = 64 }

/-- The length of the shortest line segment tangent to both circles --/
def shortest_tangent_length (circles : TwoCircles) : ℝ :=
  sorry

/-- Theorem stating that the shortest tangent length for the given circles is 30 --/
theorem shortest_tangent_is_30 :
  shortest_tangent_length problem_circles = 30 :=
sorry

end shortest_tangent_is_30_l132_13223


namespace jan_extra_miles_l132_13244

theorem jan_extra_miles (t s : ℝ) 
  (ian_distance : ℝ → ℝ → ℝ)
  (han_distance : ℝ → ℝ → ℝ)
  (jan_distance : ℝ → ℝ → ℝ)
  (h1 : ian_distance t s = s * t)
  (h2 : han_distance t s = (s + 10) * (t + 2))
  (h3 : han_distance t s = ian_distance t s + 100)
  (h4 : jan_distance t s = (s + 15) * (t + 3)) :
  jan_distance t s - ian_distance t s = 165 := by
sorry

end jan_extra_miles_l132_13244


namespace two_white_balls_probability_l132_13293

/-- The probability of drawing two white balls sequentially without replacement
    from a box containing 7 white balls and 8 black balls is 1/5. -/
theorem two_white_balls_probability
  (white_balls : Nat)
  (black_balls : Nat)
  (h_white : white_balls = 7)
  (h_black : black_balls = 8) :
  (white_balls / (white_balls + black_balls)) *
  ((white_balls - 1) / (white_balls + black_balls - 1)) =
  1 / 5 := by
  sorry

end two_white_balls_probability_l132_13293


namespace white_ring_weight_l132_13284

/-- Given the weights of three plastic rings (orange, purple, and white) and their total weight,
    this theorem proves that the weight of the white ring is 0.42 ounces. -/
theorem white_ring_weight
  (orange_weight : ℝ)
  (purple_weight : ℝ)
  (total_weight : ℝ)
  (h1 : orange_weight = 0.08)
  (h2 : purple_weight = 0.33)
  (h3 : total_weight = 0.83)
  : total_weight - (orange_weight + purple_weight) = 0.42 := by
  sorry

#eval (0.83 : ℝ) - ((0.08 : ℝ) + (0.33 : ℝ))

end white_ring_weight_l132_13284


namespace path_cost_calculation_l132_13253

/-- Calculates the total cost of constructing a path around a rectangular field -/
def path_construction_cost (field_length field_width path_width cost_per_sqm : ℝ) : ℝ :=
  let outer_length := field_length + 2 * path_width
  let outer_width := field_width + 2 * path_width
  let total_area := outer_length * outer_width
  let field_area := field_length * field_width
  let path_area := total_area - field_area
  path_area * cost_per_sqm

/-- Theorem stating the total cost of constructing the path -/
theorem path_cost_calculation :
  path_construction_cost 75 55 2.5 10 = 6750 := by
  sorry

end path_cost_calculation_l132_13253


namespace min_value_theorem_l132_13297

theorem min_value_theorem (a b : ℝ) (h1 : a * b > 0) (h2 : 2 * a + b = 5) :
  (∀ x y : ℝ, x * y > 0 ∧ 2 * x + y = 5 → 
    2 / (a + 1) + 1 / (b + 1) ≤ 2 / (x + 1) + 1 / (y + 1)) ∧
  2 / (a + 1) + 1 / (b + 1) = 9 / 8 := by
sorry

end min_value_theorem_l132_13297


namespace sequence_equals_primes_l132_13216

theorem sequence_equals_primes (a p : ℕ → ℕ) :
  (∀ n, 0 < a n) →
  (∀ n k, n < k → a n < a k) →
  (∀ n, Nat.Prime (p n)) →
  (∀ n, p n ∣ a n) →
  (∀ n k, a n - a k = p n - p k) →
  ∀ n, a n = p n :=
by sorry

end sequence_equals_primes_l132_13216


namespace gcd_equation_solutions_l132_13265

theorem gcd_equation_solutions (x y d : ℕ) :
  d = Nat.gcd x y →
  d + x * y / d = x + y →
  (∃ k : ℕ, (x = d ∧ y = d * k) ∨ (x = d * k ∧ y = d)) := by
  sorry

end gcd_equation_solutions_l132_13265


namespace seventeen_sided_polygon_diagonals_l132_13238

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 17 sides has 119 diagonals -/
theorem seventeen_sided_polygon_diagonals :
  num_diagonals 17 = 119 := by sorry

end seventeen_sided_polygon_diagonals_l132_13238


namespace shelbys_rainy_drive_time_l132_13259

theorem shelbys_rainy_drive_time 
  (speed_no_rain : ℝ) 
  (speed_rain : ℝ) 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_no_rain = 30) 
  (h2 : speed_rain = 20) 
  (h3 : total_distance = 24) 
  (h4 : total_time = 50) : 
  ∃ (rain_time : ℝ), 
    rain_time = 3 ∧ 
    (speed_no_rain / 60) * (total_time - rain_time) + (speed_rain / 60) * rain_time = total_distance :=
by
  sorry


end shelbys_rainy_drive_time_l132_13259


namespace smallest_n_remainder_l132_13275

theorem smallest_n_remainder (n : ℕ) : 
  (n > 0 ∧ ∀ m : ℕ, m > 0 → m < n → (3 * m + 45) % 1060 ≠ 16) →
  (3 * n + 45) % 1060 = 16 →
  (18 * n + 17) % 1920 = 1043 := by
sorry

end smallest_n_remainder_l132_13275


namespace pizza_toppings_combinations_l132_13247

-- Define the number of available toppings
def n : ℕ := 8

-- Define the number of toppings to choose
def k : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem pizza_toppings_combinations : combination n k = 56 := by
  sorry

end pizza_toppings_combinations_l132_13247


namespace prob_eight_odd_rolls_l132_13273

/-- A fair twelve-sided die -/
def TwelveSidedDie : Finset ℕ := Finset.range 12

/-- The set of odd numbers on a twelve-sided die -/
def OddNumbers : Finset ℕ := TwelveSidedDie.filter (λ x => x % 2 = 1)

/-- The probability of rolling an odd number with a twelve-sided die -/
def ProbOdd : ℚ := (OddNumbers.card : ℚ) / (TwelveSidedDie.card : ℚ)

/-- The number of consecutive rolls -/
def NumRolls : ℕ := 8

theorem prob_eight_odd_rolls :
  ProbOdd ^ NumRolls = 1 / 256 := by sorry

end prob_eight_odd_rolls_l132_13273


namespace b_job_fraction_l132_13250

/-- The fraction of the job that B completes when A and B work together to finish a job -/
theorem b_job_fraction (a_time b_time : ℝ) (a_solo_time : ℝ) : 
  a_time = 6 →
  b_time = 3 →
  a_solo_time = 1 →
  (25 : ℝ) / 54 = 
    ((1 - a_solo_time / a_time) * (1 / b_time) * 
     (1 - a_solo_time / a_time) / ((1 / a_time) + (1 / b_time))) :=
by sorry

end b_job_fraction_l132_13250


namespace system_one_solution_system_two_solution_l132_13269

-- System (1)
theorem system_one_solution (x y : ℚ) : 
  (3 * x - 6 * y = 4 ∧ x + 5 * y = 6) ↔ (x = 8/3 ∧ y = 2/3) := by sorry

-- System (2)
theorem system_two_solution (x y : ℚ) :
  (x/4 + y/3 = 3 ∧ 3*(x-4) - 2*(y-1) = -1) ↔ (x = 6 ∧ y = 9/2) := by sorry

end system_one_solution_system_two_solution_l132_13269


namespace composite_function_solution_l132_13217

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := x^2 + 5
def g (x : ℝ) : ℝ := x^2 - 3
def h (x : ℝ) : ℝ := 2*x + 1

-- State the theorem
theorem composite_function_solution (a : ℝ) (ha : a > 0) 
  (h_eq : f (g (h a)) = 17) : 
  a = (-1 + Real.sqrt (3 + 2 * Real.sqrt 3)) / 2 := by
  sorry

end composite_function_solution_l132_13217


namespace percentage_calculation_l132_13226

theorem percentage_calculation (initial_amount : ℝ) : 
  initial_amount = 1200 →
  (((initial_amount * 0.60) * 0.30) * 2) / 3 = 144 := by
sorry

end percentage_calculation_l132_13226


namespace water_to_dean_height_ratio_l132_13228

-- Define the heights and water depth
def ron_height : ℝ := 14
def height_difference : ℝ := 8
def water_depth : ℝ := 12

-- Define Dean's height
def dean_height : ℝ := ron_height - height_difference

-- Theorem statement
theorem water_to_dean_height_ratio :
  (water_depth / dean_height) = 2 := by
  sorry

end water_to_dean_height_ratio_l132_13228


namespace l_shape_area_l132_13243

theorem l_shape_area (a : ℝ) (h1 : a > 0) (h2 : 5 * a^2 = 4 * ((a + 3)^2 - a^2)) :
  (a + 3)^2 - a^2 = 45 := by
  sorry

end l_shape_area_l132_13243


namespace giraffe_count_prove_giraffe_count_l132_13240

/-- The number of giraffes at a zoo, given certain conditions. -/
theorem giraffe_count : ℕ → ℕ → Prop :=
  fun (giraffes : ℕ) (other_animals : ℕ) =>
    giraffes = 3 * other_animals ∧
    giraffes = other_animals + 290 →
    giraffes = 435

/-- Proof of the giraffe count theorem. -/
theorem prove_giraffe_count : ∃ (g o : ℕ), giraffe_count g o :=
  sorry

end giraffe_count_prove_giraffe_count_l132_13240


namespace max_cone_bound_for_f_l132_13290

/-- A function f: ℝ → ℝ is cone-bottomed if there exists a constant M > 0
    such that |f(x)| ≥ M|x| for all x ∈ ℝ -/
def ConeBounded (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, |f x| ≥ M * |x|

/-- The function f(x) = x^2 + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

theorem max_cone_bound_for_f :
  (ConeBounded f) ∧ (∀ M : ℝ, (∀ x : ℝ, |f x| ≥ M * |x|) → M ≤ 2) ∧
  (∃ x : ℝ, |f x| = 2 * |x|) := by
  sorry


end max_cone_bound_for_f_l132_13290


namespace no_tangent_lines_l132_13271

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of two circles --/
structure TwoCircles where
  circle1 : Circle
  circle2 : Circle
  center_distance : ℝ

/-- Counts the number of tangent lines between two circles --/
def count_tangent_lines (tc : TwoCircles) : ℕ := sorry

/-- The specific configuration given in the problem --/
def problem_config : TwoCircles :=
  { circle1 := { center := (0, 0), radius := 4 }
  , circle2 := { center := (3, 0), radius := 6 }
  , center_distance := 3 }

/-- Theorem stating that the number of tangent lines is zero for the given configuration --/
theorem no_tangent_lines : count_tangent_lines problem_config = 0 := by sorry

end no_tangent_lines_l132_13271


namespace complex_subtraction_zero_implies_equality_l132_13289

theorem complex_subtraction_zero_implies_equality (a b : ℂ) : a - b = 0 → a = b := by
  sorry

end complex_subtraction_zero_implies_equality_l132_13289


namespace ice_cream_consumption_l132_13232

theorem ice_cream_consumption (friday_consumption : Real) (total_consumption : Real)
  (h1 : friday_consumption = 3.25)
  (h2 : total_consumption = 3.5) :
  total_consumption - friday_consumption = 0.25 := by
sorry

end ice_cream_consumption_l132_13232


namespace factor_expression_l132_13283

theorem factor_expression (a : ℝ) : 58 * a^2 + 174 * a = 58 * a * (a + 3) := by
  sorry

end factor_expression_l132_13283


namespace expression_value_l132_13210

theorem expression_value (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = 0) 
  (h2 : a * b * c < 0) : 
  a / |a| + b / |b| + c / |c| = 1 := by
  sorry

end expression_value_l132_13210


namespace even_increasing_nonpositive_property_l132_13208

-- Define an even function that is increasing on (-∞, 0]
def is_even_and_increasing_nonpositive (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y)

-- State the theorem
theorem even_increasing_nonpositive_property 
  (f : ℝ → ℝ) (h : is_even_and_increasing_nonpositive f) :
  ∀ a : ℝ, f (a^2) > f (a^2 + 1) :=
sorry

end even_increasing_nonpositive_property_l132_13208


namespace divide_by_fraction_main_proof_l132_13212

theorem divide_by_fraction (a b c : ℝ) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b :=
by sorry

theorem main_proof : (5 : ℝ) / ((7 : ℝ) / 3) = 15 / 7 :=
by sorry

end divide_by_fraction_main_proof_l132_13212


namespace triangles_count_is_120_l132_13266

/-- Represents the configuration of points on two line segments -/
structure PointConfiguration :=
  (total_points : ℕ)
  (points_on_segment1 : ℕ)
  (points_on_segment2 : ℕ)
  (end_point : ℕ)

/-- Calculates the number of triangles that can be formed with the given point configuration -/
def count_triangles (config : PointConfiguration) : ℕ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the specific configuration results in 120 triangles -/
theorem triangles_count_is_120 :
  let config := PointConfiguration.mk 11 6 4 1
  count_triangles config = 120 :=
by sorry

end triangles_count_is_120_l132_13266


namespace empty_solution_set_l132_13279

def f (x : ℝ) : ℝ := x^2 + x

theorem empty_solution_set :
  {x : ℝ | f (x - 2) + f x < 0} = ∅ := by sorry

end empty_solution_set_l132_13279


namespace square_difference_1002_1000_l132_13285

theorem square_difference_1002_1000 : 1002^2 - 1000^2 = 4004 := by
  sorry

end square_difference_1002_1000_l132_13285


namespace distance_point_to_line_l132_13221

/-- The distance from a point to a line in 3D space --/
def distancePointToLine (p : ℝ × ℝ × ℝ) (l1 l2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_point_to_line :
  let p := (2, -2, 3)
  let l1 := (1, 3, -1)
  let l2 := (0, 0, 2)
  distancePointToLine p l1 l2 = Real.sqrt 2750 / 19 := by
  sorry

end distance_point_to_line_l132_13221


namespace expression_simplification_l132_13287

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3 + 1) :
  (1 - 1 / m) / ((m^2 - 2*m + 1) / m) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l132_13287


namespace divisors_of_2121_with_units_digit_1_l132_13256

/-- The number of positive integer divisors of 2121 with a units digit of 1 is 4. -/
theorem divisors_of_2121_with_units_digit_1 : 
  (Finset.filter (fun d => d % 10 = 1) (Nat.divisors 2121)).card = 4 := by
  sorry

end divisors_of_2121_with_units_digit_1_l132_13256


namespace metallic_sheet_dimension_l132_13246

/-- Given a rectangular metallic sheet with one dimension of 36 meters,
    where a square of 8 meters is cut from each corner to form an open box,
    if the volume of the resulting box is 5760 cubic meters,
    then the length of the other dimension of the metallic sheet is 52 meters. -/
theorem metallic_sheet_dimension (sheet_width : ℝ) (cut_size : ℝ) (box_volume : ℝ) :
  sheet_width = 36 →
  cut_size = 8 →
  box_volume = 5760 →
  (sheet_width - 2 * cut_size) * (52 - 2 * cut_size) * cut_size = box_volume :=
by sorry

end metallic_sheet_dimension_l132_13246


namespace power_of_two_equation_l132_13261

theorem power_of_two_equation (m : ℕ) : 
  2^2002 - 2^2000 - 2^1999 + 2^1998 = m * 2^1998 → m = 11 := by
  sorry

end power_of_two_equation_l132_13261


namespace park_area_l132_13231

/-- The area of a rectangular park with sides in ratio 3:2 and fencing cost $150 at 60 ps per meter --/
theorem park_area (length width : ℝ) (area perimeter cost_per_meter total_cost : ℝ) : 
  length / width = 3 / 2 →
  area = length * width →
  perimeter = 2 * (length + width) →
  cost_per_meter = 0.60 →
  total_cost = 150 →
  total_cost = perimeter * cost_per_meter →
  area = 3750 :=
by sorry

end park_area_l132_13231


namespace sin_five_half_pi_plus_alpha_l132_13211

theorem sin_five_half_pi_plus_alpha (α : ℝ) : 
  Real.sin ((5 / 2) * Real.pi + α) = Real.cos α := by sorry

end sin_five_half_pi_plus_alpha_l132_13211


namespace smallest_c_for_inverse_l132_13229

def g (x : ℝ) : ℝ := (x - 3)^2 - 7

theorem smallest_c_for_inverse : 
  ∀ c : ℝ, (∀ x y, x ≥ c → y ≥ c → g x = g y → x = y) ↔ c ≥ 3 :=
sorry

end smallest_c_for_inverse_l132_13229


namespace triangular_front_view_solids_l132_13263

-- Define the types of solids
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

-- Define a predicate for solids that can have a triangular front view
def hasTriangularFrontView (s : Solid) : Prop :=
  match s with
  | Solid.TriangularPyramid => True
  | Solid.SquarePyramid => True
  | Solid.TriangularPrism => True
  | Solid.Cone => True
  | _ => False

-- Theorem stating which solids can have a triangular front view
theorem triangular_front_view_solids :
  ∀ s : Solid, hasTriangularFrontView s ↔
    (s = Solid.TriangularPyramid ∨
     s = Solid.SquarePyramid ∨
     s = Solid.TriangularPrism ∨
     s = Solid.Cone) :=
by sorry

end triangular_front_view_solids_l132_13263


namespace trigonometric_inequality_l132_13209

theorem trigonometric_inequality (φ : Real) (h : 0 < φ ∧ φ < Real.pi / 2) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end trigonometric_inequality_l132_13209


namespace quadratic_factorization_l132_13286

theorem quadratic_factorization (b k : ℝ) : 
  (∀ x, x^2 + b*x + 5 = (x - 2)^2 + k) → (b = -4 ∧ k = 1) := by
  sorry

end quadratic_factorization_l132_13286


namespace range_of_f_when_k_4_range_of_k_for_monotone_f_l132_13203

/-- The function f(x) = (k-2)x^2 + 2kx - 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x^2 + 2 * k * x - 3

/-- The range of f(x) when k = 4 in the interval (-4, 1) -/
theorem range_of_f_when_k_4 :
  Set.Icc (-11 : ℝ) 7 = Set.image (f 4) (Set.Ioo (-4 : ℝ) 1) := by sorry

/-- The range of k for which f(x) is monotonically increasing in [1, 2] -/
theorem range_of_k_for_monotone_f :
  ∀ k : ℝ, (∀ x y : ℝ, x ∈ Set.Icc (1 : ℝ) 2 → y ∈ Set.Icc (1 : ℝ) 2 → x ≤ y → f k x ≤ f k y) ↔
  k ∈ Set.Ici (4/3 : ℝ) := by sorry

end range_of_f_when_k_4_range_of_k_for_monotone_f_l132_13203


namespace draw_probability_modified_deck_l132_13294

/-- A modified deck of cards -/
structure ModifiedDeck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (cards_per_rank : ℕ)

/-- The probability of drawing a specific sequence of cards -/
def draw_probability (deck : ModifiedDeck) (heart_cards : ℕ) (spade_cards : ℕ) (king_cards : ℕ) : ℚ :=
  (heart_cards * spade_cards * king_cards : ℚ) / 
  (deck.total_cards * (deck.total_cards - 1) * (deck.total_cards - 2))

/-- The main theorem -/
theorem draw_probability_modified_deck :
  let deck := ModifiedDeck.mk 104 26 4 26 8
  draw_probability deck 26 26 8 = 169 / 34102 := by sorry

end draw_probability_modified_deck_l132_13294


namespace smallest_product_is_zero_l132_13276

def S : Set ℤ := {-8, -4, 0, 2, 5}

theorem smallest_product_is_zero :
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a * b = 0 ∧ 
  ∀ (x y : ℤ), x ∈ S → y ∈ S → a * b ≤ x * y :=
by sorry

end smallest_product_is_zero_l132_13276


namespace subtraction_base_8_to_10_l132_13278

def base_8_to_10 (n : ℕ) : ℕ :=
  let digits := n.digits 8
  (List.range digits.length).foldl (λ acc i => acc + digits[i]! * (8 ^ i)) 0

theorem subtraction_base_8_to_10 :
  base_8_to_10 (4725 - 2367) = 1246 :=
sorry

end subtraction_base_8_to_10_l132_13278


namespace jill_study_time_l132_13200

/-- Calculates the total minutes studied over 3 days given a specific study pattern -/
def totalMinutesStudied (day1Hours : ℕ) : ℕ :=
  let day2Hours := 2 * day1Hours
  let day3Hours := day2Hours - 1
  (day1Hours + day2Hours + day3Hours) * 60

/-- Proves that given Jill's study pattern, she studies for 540 minutes over 3 days -/
theorem jill_study_time : totalMinutesStudied 2 = 540 := by
  sorry

end jill_study_time_l132_13200


namespace lcm_1230_924_l132_13292

theorem lcm_1230_924 : Nat.lcm 1230 924 = 189420 := by
  sorry

end lcm_1230_924_l132_13292


namespace bus_travel_fraction_l132_13242

theorem bus_travel_fraction (total_distance : ℝ) 
  (h1 : total_distance = 105.00000000000003)
  (h2 : (1 : ℝ) / 5 * total_distance + 14 + (2 : ℝ) / 3 * total_distance = total_distance) :
  (total_distance - ((1 : ℝ) / 5 * total_distance + 14)) / total_distance = 2 / 3 := by
  sorry

end bus_travel_fraction_l132_13242


namespace complement_of_M_l132_13206

def M : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 5}

theorem complement_of_M :
  (Set.univ : Set ℝ) \ M = {x : ℝ | x < -3 ∨ x ≥ 5} := by sorry

end complement_of_M_l132_13206


namespace orange_juice_fraction_is_467_2400_l132_13282

/-- Represents a pitcher with a specific volume and content --/
structure Pitcher :=
  (volume : ℚ)
  (content : ℚ)

/-- Calculates the fraction of orange juice in the final mixture --/
def orangeJuiceFraction (p1 p2 p3 : Pitcher) : ℚ :=
  let totalVolume := p1.volume + p2.volume + p3.volume
  let orangeJuiceVolume := p1.content + p2.content
  orangeJuiceVolume / totalVolume

/-- Theorem stating the fraction of orange juice in the final mixture --/
theorem orange_juice_fraction_is_467_2400 :
  let p1 : Pitcher := ⟨800, 800 * (1/4)⟩
  let p2 : Pitcher := ⟨800, 800 * (1/3)⟩
  let p3 : Pitcher := ⟨800, 0⟩  -- Third pitcher doesn't contribute to orange juice
  orangeJuiceFraction p1 p2 p3 = 467 / 2400 := by
  sorry

#eval orangeJuiceFraction ⟨800, 800 * (1/4)⟩ ⟨800, 800 * (1/3)⟩ ⟨800, 0⟩

end orange_juice_fraction_is_467_2400_l132_13282


namespace ellipse_equation_parabola_equation_l132_13222

-- Problem 1
theorem ellipse_equation (focal_distance : ℝ) (point : ℝ × ℝ) : 
  focal_distance = 4 ∧ point = (3, -2 * Real.sqrt 6) →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧
    (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/36 + y^2/32 = 1) :=
sorry

-- Problem 2
theorem parabola_equation (hyperbola : ℝ → ℝ → Prop) (directrix : ℝ) :
  (∀ x y : ℝ, hyperbola x y ↔ x^2 - y^2/3 = 1) ∧
  directrix = -1/2 →
  ∀ x y : ℝ, y^2 = 2*x ↔ y^2 = 2*x ∧ x ≥ 0 :=
sorry

end ellipse_equation_parabola_equation_l132_13222


namespace subtracting_and_dividing_l132_13299

theorem subtracting_and_dividing (x : ℝ) : x = 32 → (x - 6) / 13 = 2 := by
  sorry

end subtracting_and_dividing_l132_13299


namespace complex_modulus_theorem_l132_13213

theorem complex_modulus_theorem (ω : ℂ) (h : ω = 8 + I) : 
  Complex.abs (ω^2 - 4*ω + 13) = 4 * Real.sqrt 130 := by
sorry

end complex_modulus_theorem_l132_13213


namespace min_distance_sum_l132_13201

/-- A rectangle with sides 20 cm and 10 cm -/
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (is_rectangle : sorry)
  (AB_length : dist A B = 20)
  (BC_length : dist B C = 10)

/-- The sum of distances BM + MN -/
def distance_sum (rect : Rectangle) (M : ℝ × ℝ) (N : ℝ × ℝ) : ℝ :=
  dist rect.B M + dist M N

/-- M is on AC -/
def M_on_AC (rect : Rectangle) (M : ℝ × ℝ) : Prop :=
  sorry

/-- N is on AB -/
def N_on_AB (rect : Rectangle) (N : ℝ × ℝ) : Prop :=
  sorry

theorem min_distance_sum (rect : Rectangle) :
  ∃ (M N : ℝ × ℝ), M_on_AC rect M ∧ N_on_AB rect N ∧
    (∀ (M' N' : ℝ × ℝ), M_on_AC rect M' → N_on_AB rect N' →
      distance_sum rect M N ≤ distance_sum rect M' N') ∧
    distance_sum rect M N = 16 :=
  sorry

end min_distance_sum_l132_13201


namespace other_sales_percentage_l132_13298

/-- The percentage of sales for notebooks -/
def notebook_sales : ℝ := 42

/-- The percentage of sales for markers -/
def marker_sales : ℝ := 26

/-- The total percentage of sales -/
def total_sales : ℝ := 100

/-- The percentage of sales that were not notebooks or markers -/
def other_sales : ℝ := total_sales - (notebook_sales + marker_sales)

theorem other_sales_percentage : other_sales = 32 := by
  sorry

end other_sales_percentage_l132_13298


namespace cubic_equation_value_l132_13214

theorem cubic_equation_value (x : ℝ) (h : x^2 + x - 2 = 0) :
  x^3 + 2*x^2 - x + 2021 = 2023 := by
  sorry

end cubic_equation_value_l132_13214


namespace teacher_assignment_count_l132_13264

/-- The number of ways to assign four teachers to three classes --/
def total_assignments : ℕ := 36

/-- The number of ways to assign teachers A and B to the same class --/
def ab_same_class : ℕ := 6

/-- The number of ways to assign four teachers to three classes with A and B in different classes --/
def valid_assignments : ℕ := total_assignments - ab_same_class

theorem teacher_assignment_count :
  valid_assignments = 30 :=
sorry

end teacher_assignment_count_l132_13264


namespace collinear_implies_relation_vector_relation_implies_coordinates_l132_13239

-- Define points A, B, and C
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (3, -1)
def C : ℝ → ℝ → ℝ × ℝ := λ a b => (a, b)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r.1 - p.1 = t * (q.1 - p.1) ∧ r.2 - p.2 = t * (q.2 - p.2)

-- Define vector multiplication
def vec_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Part 1: Collinearity implies a = 2 - b
theorem collinear_implies_relation (a b : ℝ) :
  collinear A B (C a b) → a = 2 - b := by sorry

-- Part 2: AC = 2AB implies C = (5, -3)
theorem vector_relation_implies_coordinates (a b : ℝ) :
  C a b - A = vec_mult 2 (B - A) → C a b = (5, -3) := by sorry

end collinear_implies_relation_vector_relation_implies_coordinates_l132_13239


namespace range_of_m_l132_13262

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1}
def B (m : ℝ) : Set ℝ := {x | x ≤ m}

-- State the theorem
theorem range_of_m (m : ℝ) : B m ⊆ A → m ≤ 1 := by
  sorry

end range_of_m_l132_13262


namespace quadratic_root_implies_m_l132_13219

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - x + m^2 - 4 = 0 ∧ x = 1) → (m = 2 ∨ m = -2) := by
  sorry

end quadratic_root_implies_m_l132_13219


namespace tv_series_seasons_l132_13245

theorem tv_series_seasons (episodes_per_season : ℕ) (episodes_per_day : ℕ) (days_to_finish : ℕ) : 
  episodes_per_season = 20 →
  episodes_per_day = 2 →
  days_to_finish = 30 →
  (episodes_per_day * days_to_finish) / episodes_per_season = 3 := by
sorry

end tv_series_seasons_l132_13245


namespace total_ad_cost_is_66000_l132_13237

/-- Represents an advertisement with its duration and cost per minute -/
structure Advertisement where
  duration : ℕ
  costPerMinute : ℕ

/-- Calculates the total cost of an advertisement -/
def adCost (ad : Advertisement) : ℕ := ad.duration * ad.costPerMinute

/-- The list of advertisements shown during the race -/
def raceAds : List Advertisement := [
  ⟨2, 3500⟩,
  ⟨3, 4500⟩,
  ⟨3, 3000⟩,
  ⟨2, 4000⟩,
  ⟨5, 5500⟩
]

/-- The theorem stating that the total cost of advertisements is $66000 -/
theorem total_ad_cost_is_66000 :
  (raceAds.map adCost).sum = 66000 := by sorry

end total_ad_cost_is_66000_l132_13237


namespace expression_simplification_l132_13281

theorem expression_simplification (x : ℝ) : (2*x + 1)*(2*x - 1) - x*(4*x - 1) = x - 1 := by
  sorry

end expression_simplification_l132_13281


namespace largest_angle_is_120_degrees_l132_13249

-- Define the sequence a_n
def a (n : ℕ) : ℕ := n^2 - (n-1)^2

-- Define the triangle sides
def side_a : ℕ := a 2
def side_b : ℕ := a 3
def side_c : ℕ := a 4

-- State the theorem
theorem largest_angle_is_120_degrees :
  let angle := Real.arccos ((side_a^2 + side_b^2 - side_c^2) / (2 * side_a * side_b))
  angle = 2 * Real.pi / 3 := by sorry

end largest_angle_is_120_degrees_l132_13249


namespace sum_of_special_numbers_l132_13241

/-- A function that counts the number of divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a number ends with 5 zeros -/
def ends_with_five_zeros (n : ℕ) : Prop := sorry

/-- The set of natural numbers that end with 5 zeros and have 42 divisors -/
def special_numbers : Set ℕ :=
  {n : ℕ | ends_with_five_zeros n ∧ count_divisors n = 42}

theorem sum_of_special_numbers :
  ∃ (a b : ℕ), a ∈ special_numbers ∧ b ∈ special_numbers ∧ a ≠ b ∧ a + b = 700000 := by
  sorry

end sum_of_special_numbers_l132_13241


namespace equation_solutions_l132_13280

theorem equation_solutions :
  (∀ x : ℝ, 6*x - 7 = 4*x - 5 ↔ x = 1) ∧
  (∀ x : ℝ, 5*(x + 8) - 5 = 6*(2*x - 7) ↔ x = 11) ∧
  (∀ x : ℝ, x - (x - 1)/2 = 2 - (x + 2)/5 ↔ x = 11/7) ∧
  (∀ x : ℝ, x^2 - 64 = 0 ↔ x = 8 ∨ x = -8) :=
by sorry

end equation_solutions_l132_13280


namespace total_pictures_l132_13274

/-- The number of pictures drawn by each person and their total -/
def picture_problem (randy peter quincy susan thomas : ℕ) : Prop :=
  randy = 5 ∧
  peter = randy + 3 ∧
  quincy = peter + 20 ∧
  susan = 2 * quincy - 7 ∧
  thomas = randy ^ 3 ∧
  randy + peter + quincy + susan + thomas = 215

/-- Proof that the total number of pictures drawn is 215 -/
theorem total_pictures : ∃ randy peter quincy susan thomas : ℕ, 
  picture_problem randy peter quincy susan thomas := by
  sorry

end total_pictures_l132_13274
