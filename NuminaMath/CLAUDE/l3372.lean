import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l3372_337238

theorem equation_solution :
  ∃ (x : ℚ), x ≠ -3 ∧ (x^2 + 3*x + 4) / (x + 3) = x + 6 ↔ x = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3372_337238


namespace NUMINAMATH_CALUDE_b_subscription_difference_l3372_337202

/-- Represents the subscription amounts and profit distribution for a business venture --/
structure BusinessVenture where
  total_subscription : ℕ
  total_profit : ℕ
  a_subscription : ℕ
  b_subscription : ℕ
  c_subscription : ℕ
  a_profit : ℕ

/-- The conditions of the business venture as described in the problem --/
def venture_conditions (v : BusinessVenture) : Prop :=
  v.total_subscription = 50000 ∧
  v.total_profit = 70000 ∧
  v.a_profit = 29400 ∧
  v.a_subscription = v.b_subscription + 4000 ∧
  v.b_subscription > v.c_subscription ∧
  v.a_subscription + v.b_subscription + v.c_subscription = v.total_subscription ∧
  v.a_profit * v.total_subscription = v.a_subscription * v.total_profit

/-- The theorem stating that B subscribed 5000 more than C --/
theorem b_subscription_difference (v : BusinessVenture) 
  (h : venture_conditions v) : v.b_subscription - v.c_subscription = 5000 := by
  sorry


end NUMINAMATH_CALUDE_b_subscription_difference_l3372_337202


namespace NUMINAMATH_CALUDE_circle_radius_through_triangle_vertices_l3372_337274

theorem circle_radius_through_triangle_vertices (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  let r := (max a (max b c)) / 2
  r = 17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_through_triangle_vertices_l3372_337274


namespace NUMINAMATH_CALUDE_equation_solutions_no_solutions_l3372_337271

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem equation_solutions (n k : ℕ) :
  (∃ A : ℕ, A = 7 ∧ factorial n + A * n = n^k) ↔ (n = 2 ∧ k = 4) ∨ (n = 3 ∧ k = 3) :=
sorry

theorem no_solutions (n k : ℕ) :
  ¬(∃ A : ℕ, A = 2012 ∧ factorial n + A * n = n^k) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_no_solutions_l3372_337271


namespace NUMINAMATH_CALUDE_twenty_percent_less_than_sixty_l3372_337212

theorem twenty_percent_less_than_sixty (x : ℝ) : x + (1/3) * x = 60 - 0.2 * 60 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_less_than_sixty_l3372_337212


namespace NUMINAMATH_CALUDE_afternoon_campers_l3372_337204

theorem afternoon_campers (evening_campers : ℕ) (afternoon_evening_difference : ℕ) 
  (h1 : evening_campers = 10)
  (h2 : afternoon_evening_difference = 24) :
  evening_campers + afternoon_evening_difference = 34 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_campers_l3372_337204


namespace NUMINAMATH_CALUDE_cost_of_horse_cost_of_horse_proof_l3372_337256

/-- The cost of a horse given the conditions of Albert's purchase and sale -/
theorem cost_of_horse : ℝ :=
  let total_cost : ℝ := 13400
  let total_profit : ℝ := 1880
  let num_horses : ℕ := 4
  let num_cows : ℕ := 9
  let horse_profit_rate : ℝ := 0.1
  let cow_profit_rate : ℝ := 0.2

  2000

theorem cost_of_horse_proof (total_cost : ℝ) (total_profit : ℝ) 
  (num_horses num_cows : ℕ) (horse_profit_rate cow_profit_rate : ℝ) :
  total_cost = 13400 →
  total_profit = 1880 →
  num_horses = 4 →
  num_cows = 9 →
  horse_profit_rate = 0.1 →
  cow_profit_rate = 0.2 →
  ∃ (horse_cost cow_cost : ℝ),
    num_horses * horse_cost + num_cows * cow_cost = total_cost ∧
    num_horses * horse_cost * horse_profit_rate + num_cows * cow_cost * cow_profit_rate = total_profit ∧
    horse_cost = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_horse_cost_of_horse_proof_l3372_337256


namespace NUMINAMATH_CALUDE_leadership_configurations_count_l3372_337220

-- Define the number of members in the society
def society_size : ℕ := 12

-- Define the number of positions to be filled
def chief_count : ℕ := 1
def supporting_chief_count : ℕ := 2
def inferior_officers_A_count : ℕ := 3
def inferior_officers_B_count : ℕ := 2

-- Define the function to calculate the number of ways to establish the leadership configuration
def leadership_configurations : ℕ := 
  society_size * (society_size - 1) * (society_size - 2) * 
  (Nat.choose (society_size - 3) inferior_officers_A_count) * 
  (Nat.choose (society_size - 3 - inferior_officers_A_count) inferior_officers_B_count)

-- Theorem statement
theorem leadership_configurations_count : leadership_configurations = 1663200 := by
  sorry

end NUMINAMATH_CALUDE_leadership_configurations_count_l3372_337220


namespace NUMINAMATH_CALUDE_intersection_M_N_l3372_337252

def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem intersection_M_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3372_337252


namespace NUMINAMATH_CALUDE_interest_rate_problem_l3372_337298

/-- The interest rate problem --/
theorem interest_rate_problem
  (principal : ℝ)
  (rate_a : ℝ)
  (time : ℝ)
  (gain_b : ℝ)
  (h1 : principal = 3500)
  (h2 : rate_a = 10)
  (h3 : time = 3)
  (h4 : gain_b = 157.5)
  : ∃ (rate_c : ℝ), rate_c = 11.5 ∧
    gain_b = (principal * rate_c / 100 * time) - (principal * rate_a / 100 * time) :=
by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l3372_337298


namespace NUMINAMATH_CALUDE_product_mod_thirteen_l3372_337263

theorem product_mod_thirteen : (1501 * 1502 * 1503 * 1504 * 1505) % 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_thirteen_l3372_337263


namespace NUMINAMATH_CALUDE_greatest_difference_of_arithmetic_progression_l3372_337224

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Checks if a quadratic equation has two distinct real roots -/
def hasTwoRoots (eq : QuadraticEquation) : Prop :=
  eq.b * eq.b - 4 * eq.a * eq.c > 0

/-- Generates all six quadratic equations with coefficients a, 2b, 4c in any order -/
def generateEquations (a b c : ℤ) : List QuadraticEquation :=
  [
    ⟨a, 2*b, 4*c⟩,
    ⟨a, 4*c, 2*b⟩,
    ⟨2*b, a, 4*c⟩,
    ⟨2*b, 4*c, a⟩,
    ⟨4*c, a, 2*b⟩,
    ⟨4*c, 2*b, a⟩
  ]

/-- The main theorem to be proved -/
theorem greatest_difference_of_arithmetic_progression
  (a b c : ℤ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h_decreasing : a > b ∧ b > c)
  (h_arithmetic : ∃ d : ℤ, b = a + d ∧ c = a + 2*d)
  (h_two_roots : ∀ eq ∈ generateEquations a b c, hasTwoRoots eq) :
  ∃ (d : ℤ), d = -3 ∧ a = 4 ∧ b = 1 ∧ c = -2 ∧
  ∀ (d' : ℤ) (a' b' c' : ℤ),
    a' ≠ 0 → b' ≠ 0 → c' ≠ 0 →
    a' > b' → b' > c' →
    b' = a' + d' → c' = a' + 2*d' →
    (∀ eq ∈ generateEquations a' b' c', hasTwoRoots eq) →
    d' ≥ d :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_difference_of_arithmetic_progression_l3372_337224


namespace NUMINAMATH_CALUDE_car_trading_profit_l3372_337219

theorem car_trading_profit (P : ℝ) (h : P > 0) : 
  let discount_rate := 0.20
  let increase_rate := 0.55
  let buying_price := P * (1 - discount_rate)
  let selling_price := buying_price * (1 + increase_rate)
  let profit := selling_price - P
  let profit_percentage := (profit / P) * 100
  profit_percentage = 24 := by sorry

end NUMINAMATH_CALUDE_car_trading_profit_l3372_337219


namespace NUMINAMATH_CALUDE_unique_three_digit_number_divisibility_l3372_337227

theorem unique_three_digit_number_divisibility : ∃! a : ℕ, 
  100 ≤ a ∧ a < 1000 ∧ 
  (∃ k : ℕ, 504000 + a = 693 * k) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_divisibility_l3372_337227


namespace NUMINAMATH_CALUDE_nested_sqrt_equality_l3372_337248

theorem nested_sqrt_equality : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * (3 ^ (1/4)) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_equality_l3372_337248


namespace NUMINAMATH_CALUDE_max_goats_from_coconuts_l3372_337281

/-- Represents the trading rates and initial coconut count --/
structure TradingRates :=
  (coconuts_per_crab : ℝ)
  (crabs_per_fish : ℝ)
  (fish_per_goat : ℝ)
  (initial_coconuts : ℕ)

/-- Calculates the maximum number of whole goats obtainable --/
def max_goats (rates : TradingRates) : ℕ :=
  sorry

/-- The theorem stating that given the specific trading rates and 1000 coconuts, 
    Max can obtain 33 goats --/
theorem max_goats_from_coconuts :
  let rates := TradingRates.mk 3.5 (6.25 / 5.5) 7.5 1000
  max_goats rates = 33 := by sorry

end NUMINAMATH_CALUDE_max_goats_from_coconuts_l3372_337281


namespace NUMINAMATH_CALUDE_cubic_equation_natural_roots_l3372_337229

theorem cubic_equation_natural_roots :
  ∃! P : ℝ, ∀ x : ℕ,
    (5 * x^3 - 5 * (P + 1) * x^2 + (71 * P - 1) * x + 1 = 66 * P) →
    (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      (5 * a^3 - 5 * (P + 1) * a^2 + (71 * P - 1) * a + 1 = 66 * P) ∧
      (5 * b^3 - 5 * (P + 1) * b^2 + (71 * P - 1) * b + 1 = 66 * P) ∧
      (5 * c^3 - 5 * (P + 1) * c^2 + (71 * P - 1) * c + 1 = 66 * P)) →
    P = 76 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_natural_roots_l3372_337229


namespace NUMINAMATH_CALUDE_man_rowing_speed_l3372_337241

/-- Calculates the speed of a man rowing in still water given his downstream speed and current speed -/
theorem man_rowing_speed (current_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  current_speed = 6 →
  distance = 100 →
  time = 14.998800095992323 →
  (distance / time - current_speed * 1000 / 3600) * 3.6 = 18 := by
sorry

end NUMINAMATH_CALUDE_man_rowing_speed_l3372_337241


namespace NUMINAMATH_CALUDE_hope_star_voting_l3372_337203

/-- The Hope Star finals voting problem -/
theorem hope_star_voting
  (total_votes : ℕ)
  (huanhuan_votes lele_votes yangyang_votes : ℕ)
  (h_total : total_votes = 200)
  (h_ratio1 : 3 * lele_votes = 2 * huanhuan_votes)
  (h_ratio2 : 6 * yangyang_votes = 5 * lele_votes)
  (h_sum : huanhuan_votes + lele_votes + yangyang_votes = total_votes) :
  huanhuan_votes = 90 ∧ lele_votes = 60 ∧ yangyang_votes = 50 := by
  sorry

#check hope_star_voting

end NUMINAMATH_CALUDE_hope_star_voting_l3372_337203


namespace NUMINAMATH_CALUDE_final_concentration_l3372_337210

-- Define the volumes and concentrations
def volume1 : ℝ := 2
def concentration1 : ℝ := 0.4
def volume2 : ℝ := 3
def concentration2 : ℝ := 0.6

-- Define the total volume
def totalVolume : ℝ := volume1 + volume2

-- Define the total amount of acid
def totalAcid : ℝ := volume1 * concentration1 + volume2 * concentration2

-- Theorem: The final concentration is 52%
theorem final_concentration :
  totalAcid / totalVolume = 0.52 := by sorry

end NUMINAMATH_CALUDE_final_concentration_l3372_337210


namespace NUMINAMATH_CALUDE_right_triangle_legs_sum_l3372_337221

theorem right_triangle_legs_sum : ∀ a b : ℕ,
  (a + 1 = b) →                 -- legs are consecutive whole numbers
  (a ^ 2 + b ^ 2 = 41 ^ 2) →    -- Pythagorean theorem with hypotenuse 41
  (a + b = 57) :=               -- sum of legs is 57
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_sum_l3372_337221


namespace NUMINAMATH_CALUDE_at_op_difference_l3372_337299

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 3 * x

-- Theorem statement
theorem at_op_difference : at_op 9 6 - at_op 6 9 = -9 := by
  sorry

end NUMINAMATH_CALUDE_at_op_difference_l3372_337299


namespace NUMINAMATH_CALUDE_paint_can_display_space_l3372_337231

/-- Calculates the total number of cans in a triangular arrangement -/
def totalCans (n : ℕ) : ℕ := n * (n + 1) * 3 / 2

/-- Calculates the total space required for the cans -/
def totalSpace (n : ℕ) (spacePerCan : ℕ) : ℕ := 
  (n * (n + 1) * 3 / 2) * spacePerCan

theorem paint_can_display_space : 
  ∃ n : ℕ, totalCans n = 242 ∧ totalSpace n 50 = 3900 := by
  sorry

end NUMINAMATH_CALUDE_paint_can_display_space_l3372_337231


namespace NUMINAMATH_CALUDE_strawberry_vs_cabbage_l3372_337294

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Represents the result of cutting an isosceles right triangle -/
structure CutTriangle where
  original : Triangle
  cut1 : ℝ  -- Position of first cut (0 ≤ cut1 ≤ 1)
  cut2 : ℝ  -- Position of second cut (0 ≤ cut2 ≤ 1)

/-- Calculates the area of the rectangle formed by the cuts -/
def rectangleArea (ct : CutTriangle) : ℝ := sorry

/-- Calculates the sum of areas of the two smaller triangles formed by the cuts -/
def smallTrianglesArea (ct : CutTriangle) : ℝ := sorry

/-- Theorem: The area of the rectangle is always less than or equal to 
    the sum of the areas of the two smaller triangles -/
theorem strawberry_vs_cabbage (ct : CutTriangle) : 
  rectangleArea ct ≤ smallTrianglesArea ct := by
  sorry

end NUMINAMATH_CALUDE_strawberry_vs_cabbage_l3372_337294


namespace NUMINAMATH_CALUDE_jack_needs_additional_money_l3372_337280

def socks_price : ℝ := 12.75
def shoes_price : ℝ := 145
def ball_price : ℝ := 38
def bag_price : ℝ := 47
def shoes_discount : ℝ := 0.05
def bag_discount : ℝ := 0.10
def jack_money : ℝ := 25

def total_cost : ℝ := 
  2 * socks_price + 
  shoes_price * (1 - shoes_discount) + 
  ball_price + 
  bag_price * (1 - bag_discount)

theorem jack_needs_additional_money : 
  total_cost - jack_money = 218.55 := by sorry

end NUMINAMATH_CALUDE_jack_needs_additional_money_l3372_337280


namespace NUMINAMATH_CALUDE_complement_M_wrt_U_l3372_337250

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define the set M
def M : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_M_wrt_U : 
  {x ∈ U | x ∉ M} = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_M_wrt_U_l3372_337250


namespace NUMINAMATH_CALUDE_square_expression_equals_289_l3372_337267

theorem square_expression_equals_289 (x : ℝ) (h : x = 5) : 
  (2 * x + 5 + 2)^2 = 289 := by sorry

end NUMINAMATH_CALUDE_square_expression_equals_289_l3372_337267


namespace NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l3372_337287

theorem no_solution_quadratic_inequality :
  ∀ x : ℝ, ¬(3 * x^2 + 9 * x ≤ -12) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l3372_337287


namespace NUMINAMATH_CALUDE_stratified_sampling_most_reasonable_l3372_337234

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Systematic
  | Stratified
  | SimpleRandom

/-- Represents a population with plots -/
structure Population where
  totalPlots : ℕ
  sampleSize : ℕ
  highVariability : Bool

/-- Determines if a sampling method is reasonable given a population -/
def isReasonableSamplingMethod (p : Population) (m : SamplingMethod) : Prop :=
  p.highVariability → m = SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the most reasonable method
    for a population with high variability -/
theorem stratified_sampling_most_reasonable (p : Population) 
    (h1 : p.totalPlots = 200)
    (h2 : p.sampleSize = 20)
    (h3 : p.highVariability = true) :
    isReasonableSamplingMethod p SamplingMethod.Stratified :=
  sorry

#check stratified_sampling_most_reasonable

end NUMINAMATH_CALUDE_stratified_sampling_most_reasonable_l3372_337234


namespace NUMINAMATH_CALUDE_franks_remaining_money_l3372_337217

/-- Calculates the remaining money after Frank buys the most expensive lamp -/
def remaining_money (cheapest_lamp_cost : ℝ) (expensive_lamp_multiplier : ℝ) 
  (discount_rate : ℝ) (sales_tax_rate : ℝ) (initial_money : ℝ) : ℝ :=
  let expensive_lamp_cost := cheapest_lamp_cost * expensive_lamp_multiplier
  let discounted_price := expensive_lamp_cost * (1 - discount_rate)
  let final_price := discounted_price * (1 + sales_tax_rate)
  initial_money - final_price

/-- Theorem stating that Frank's remaining money is $31.68 -/
theorem franks_remaining_money :
  remaining_money 20 3 0.1 0.08 90 = 31.68 := by
  sorry

end NUMINAMATH_CALUDE_franks_remaining_money_l3372_337217


namespace NUMINAMATH_CALUDE_total_spent_is_89_10_l3372_337218

/-- The total amount spent by Edward and his friend after the discount -/
def total_spent_after_discount (
  trick_deck_price : ℝ) 
  (edward_decks : ℕ) 
  (edward_hat_price : ℝ)
  (friend_decks : ℕ)
  (friend_wand_price : ℝ)
  (discount_rate : ℝ) : ℝ :=
  let total_before_discount := 
    trick_deck_price * (edward_decks + friend_decks) + edward_hat_price + friend_wand_price
  total_before_discount * (1 - discount_rate)

/-- Theorem stating that the total amount spent after the discount is $89.10 -/
theorem total_spent_is_89_10 :
  total_spent_after_discount 9 4 12 4 15 0.1 = 89.10 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_89_10_l3372_337218


namespace NUMINAMATH_CALUDE_x_minus_y_value_l3372_337246

theorem x_minus_y_value (x y : ℝ) (h1 : |x| = 2) (h2 : y^2 = 9) (h3 : x + y < 0) :
  x - y = 1 ∨ x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l3372_337246


namespace NUMINAMATH_CALUDE_find_y_value_l3372_337253

theorem find_y_value (x y : ℚ) (h1 : x = 51) (h2 : x^3*y - 2*x^2*y + x*y = 63000) : y = 8/17 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l3372_337253


namespace NUMINAMATH_CALUDE_part_a_part_b_l3372_337297

-- Define the main equation
def main_equation (x p : ℝ) : Prop := x^2 + p = -x/4

-- Define the condition for part a
def condition_a (x₁ x₂ : ℝ) : Prop := x₁/x₂ + x₂/x₁ = -9/4

-- Define the condition for part b
def condition_b (x₁ x₂ : ℝ) : Prop := x₂ = x₁^2 - 1

-- Theorem for part a
theorem part_a (x₁ x₂ p : ℝ) :
  main_equation x₁ p ∧ main_equation x₂ p ∧ condition_a x₁ x₂ → p = -1/23 := by
  sorry

-- Theorem for part b
theorem part_b (x₁ x₂ p : ℝ) :
  main_equation x₁ p ∧ main_equation x₂ p ∧ condition_b x₁ x₂ →
  p = -3/8 ∨ p = -15/8 := by
  sorry

end NUMINAMATH_CALUDE_part_a_part_b_l3372_337297


namespace NUMINAMATH_CALUDE_sixth_angle_measure_l3372_337213

/-- The sum of interior angles in a hexagon -/
def hexagon_angle_sum : ℝ := 720

/-- The measures of the five known angles in the hexagon -/
def known_angles : List ℝ := [130, 95, 115, 120, 110]

/-- The theorem stating that the sixth angle in the hexagon measures 150° -/
theorem sixth_angle_measure :
  hexagon_angle_sum - (known_angles.sum) = 150 := by sorry

end NUMINAMATH_CALUDE_sixth_angle_measure_l3372_337213


namespace NUMINAMATH_CALUDE_a_5_value_l3372_337254

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

/-- The conditions of the problem -/
def problem_conditions (a : ℕ → ℚ) : Prop :=
  arithmetic_sequence a ∧ a 1 + a 5 - a 8 = 1 ∧ a 9 - a 2 = 5

theorem a_5_value (a : ℕ → ℚ) (h : problem_conditions a) : a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_a_5_value_l3372_337254


namespace NUMINAMATH_CALUDE_solution_set_g_range_of_a_l3372_337208

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := |x| + 2 * |x + 2 - a|

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := g a (x - 2)

-- Part 1: Solution set for g(x) ≤ 4 when a = 3
theorem solution_set_g (x : ℝ) :
  g 3 x ≤ 4 ↔ -2/3 ≤ x ∧ x ≤ 2 := by sorry

-- Part 2: Range of a such that f(x) ≥ 1 for all x ∈ ℝ
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 1) ↔ a ≤ 1 ∨ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_g_range_of_a_l3372_337208


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_six_l3372_337289

theorem sqrt_two_times_sqrt_six : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_six_l3372_337289


namespace NUMINAMATH_CALUDE_first_number_proof_l3372_337295

theorem first_number_proof (x : ℕ) : 
  (∃ k : ℕ, x = 144 * k + 23) ∧ 
  (∃ m : ℕ, 7373 = 144 * m + 29) ∧
  (∀ d : ℕ, d > 144 → ¬(∃ r₁ r₂ : ℕ, x = d * k + r₁ ∧ 7373 = d * m + r₂)) →
  x = 7361 :=
by sorry

end NUMINAMATH_CALUDE_first_number_proof_l3372_337295


namespace NUMINAMATH_CALUDE_best_of_three_max_value_l3372_337262

/-- The maximum value of 8q - 9p in a best-of-three table tennis match -/
theorem best_of_three_max_value (p : ℝ) (q : ℝ) 
  (h1 : 0 < p) (h2 : p < 1) (h3 : q = 3 * p^2 - 2 * p^3) : 
  ∃ (max_val : ℝ), ∀ (p' : ℝ) (q' : ℝ), 
    0 < p' → p' < 1 → q' = 3 * p'^2 - 2 * p'^3 → 
    8 * q' - 9 * p' ≤ max_val ∧ max_val = 0 := by
  sorry

end NUMINAMATH_CALUDE_best_of_three_max_value_l3372_337262


namespace NUMINAMATH_CALUDE_reservoir_drainage_l3372_337296

/- Given conditions -/
def initial_drainage_rate : ℝ := 8
def initial_drain_time : ℝ := 6
def max_drainage_capacity : ℝ := 12

/- Theorem statement -/
theorem reservoir_drainage :
  let reservoir_volume : ℝ := initial_drainage_rate * initial_drain_time
  let drainage_relation (Q t : ℝ) : Prop := Q = reservoir_volume / t
  let min_drainage_5hours : ℝ := reservoir_volume / 5
  let min_time_max_capacity : ℝ := reservoir_volume / max_drainage_capacity
  
  (reservoir_volume = 48) ∧
  (∀ Q t, drainage_relation Q t ↔ Q = 48 / t) ∧
  (min_drainage_5hours = 9.6) ∧
  (min_time_max_capacity = 4) :=
by sorry

end NUMINAMATH_CALUDE_reservoir_drainage_l3372_337296


namespace NUMINAMATH_CALUDE_unique_center_symmetric_not_axis_symmetric_l3372_337242

-- Define the shapes
inductive Shape
  | Square
  | EquilateralTriangle
  | Circle
  | Parallelogram

-- Define the symmetry properties
def is_center_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Square => true
  | Shape.EquilateralTriangle => false
  | Shape.Circle => true
  | Shape.Parallelogram => true

def is_axis_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Square => true
  | Shape.EquilateralTriangle => true
  | Shape.Circle => true
  | Shape.Parallelogram => false

-- Theorem statement
theorem unique_center_symmetric_not_axis_symmetric :
  ∀ s : Shape, (is_center_symmetric s ∧ ¬is_axis_symmetric s) ↔ s = Shape.Parallelogram :=
sorry

end NUMINAMATH_CALUDE_unique_center_symmetric_not_axis_symmetric_l3372_337242


namespace NUMINAMATH_CALUDE_union_when_m_is_3_union_equals_A_iff_l3372_337232

def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem union_when_m_is_3 : A ∪ B 3 = {x | -3 ≤ x ∧ x ≤ 5} := by sorry

theorem union_equals_A_iff (m : ℝ) : A ∪ B m = A ↔ m ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_union_when_m_is_3_union_equals_A_iff_l3372_337232


namespace NUMINAMATH_CALUDE_sine_cosine_function_minimum_l3372_337222

theorem sine_cosine_function_minimum (a ω : ℝ) : 
  a > 0 → ω > 0 → 
  (∃ (f : ℝ → ℝ), 
    (∀ x, f x = Real.sin (ω * x) + a * Real.cos (ω * x)) ∧ 
    f (π / 6) = -2 ∧ 
    (∀ x, f x ≥ -2)) → 
  ω = 7 := by sorry

end NUMINAMATH_CALUDE_sine_cosine_function_minimum_l3372_337222


namespace NUMINAMATH_CALUDE_vasya_promotion_higher_revenue_l3372_337243

/-- Represents the revenue from candy box sales under different promotions -/
def candy_revenue (normal_revenue : ℝ) : Prop :=
  let vasya_revenue := normal_revenue * 2 * 0.8
  let kolya_revenue := normal_revenue * (8/3)
  (vasya_revenue = 16000) ∧ 
  (kolya_revenue = 13333.33333333333) ∧ 
  (vasya_revenue - normal_revenue = 6000)

/-- Theorem stating that Vasya's promotion leads to higher revenue -/
theorem vasya_promotion_higher_revenue :
  candy_revenue 10000 :=
sorry

end NUMINAMATH_CALUDE_vasya_promotion_higher_revenue_l3372_337243


namespace NUMINAMATH_CALUDE_tomato_land_theorem_l3372_337279

def farmer_problem (total_land : Real) (cleared_percentage : Real) 
                   (grapes_percentage : Real) (potato_percentage : Real) : Real :=
  let cleared_land := total_land * cleared_percentage
  let grapes_land := cleared_land * grapes_percentage
  let potato_land := cleared_land * potato_percentage
  cleared_land - (grapes_land + potato_land)

theorem tomato_land_theorem :
  farmer_problem 3999.9999999999995 0.90 0.60 0.30 = 360 := by
  sorry

end NUMINAMATH_CALUDE_tomato_land_theorem_l3372_337279


namespace NUMINAMATH_CALUDE_find_divisor_l3372_337230

theorem find_divisor : ∃ (d : ℕ), d > 1 ∧ 
  (1054 + 4 = 1058) ∧ 
  (1058 % d = 0) ∧
  (∀ k : ℕ, k < 4 → (1054 + k) % d ≠ 0) →
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3372_337230


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l3372_337292

theorem perfect_square_binomial :
  ∃ a : ℝ, ∀ x : ℝ, x^2 + 120*x + 3600 = (x + a)^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l3372_337292


namespace NUMINAMATH_CALUDE_root_sum_cubes_l3372_337247

theorem root_sum_cubes (a b c d : ℝ) : 
  (3 * a^4 + 6 * a^3 + 1002 * a^2 + 2005 * a + 4010 = 0) →
  (3 * b^4 + 6 * b^3 + 1002 * b^2 + 2005 * b + 4010 = 0) →
  (3 * c^4 + 6 * c^3 + 1002 * c^2 + 2005 * c + 4010 = 0) →
  (3 * d^4 + 6 * d^3 + 1002 * d^2 + 2005 * d + 4010 = 0) →
  (a + b)^3 + (b + c)^3 + (c + d)^3 + (d + a)^3 = 9362 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_cubes_l3372_337247


namespace NUMINAMATH_CALUDE_range_of_a_l3372_337257

def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

theorem range_of_a (a : ℝ) : B a ⊆ (A ∩ B a) → a ≤ -1 ∧ a ∈ Set.Iic (-1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3372_337257


namespace NUMINAMATH_CALUDE_probability_of_at_least_three_successes_l3372_337259

def probability_of_success : ℚ := 4/5

def number_of_trials : ℕ := 4

def at_least_successes : ℕ := 3

theorem probability_of_at_least_three_successes :
  (Finset.sum (Finset.range (number_of_trials - at_least_successes + 1))
    (fun k => Nat.choose number_of_trials (number_of_trials - k) *
      probability_of_success ^ (number_of_trials - k) *
      (1 - probability_of_success) ^ k)) = 512/625 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_at_least_three_successes_l3372_337259


namespace NUMINAMATH_CALUDE_inequality_holds_l3372_337239

theorem inequality_holds (x : ℝ) : 
  (4 * x^2) / (1 - Real.sqrt (1 + 2*x))^2 < 2*x + 9 ↔ 
  (x ≥ -1/2 ∧ x < 0) ∨ (x > 0 ∧ x < 45/8) := by sorry

end NUMINAMATH_CALUDE_inequality_holds_l3372_337239


namespace NUMINAMATH_CALUDE_strawberry_picking_problem_l3372_337236

/-- Calculates the number of pounds of strawberries picked given the total paid, 
    number of pickers, entrance fee per person, and price per pound of strawberries -/
def strawberries_picked (total_paid : ℚ) (num_pickers : ℕ) (entrance_fee : ℚ) (price_per_pound : ℚ) : ℚ :=
  (total_paid + num_pickers * entrance_fee) / price_per_pound

/-- Theorem stating that under the given conditions, the number of pounds of strawberries picked is 7 -/
theorem strawberry_picking_problem :
  let total_paid : ℚ := 128
  let num_pickers : ℕ := 3
  let entrance_fee : ℚ := 4
  let price_per_pound : ℚ := 20
  strawberries_picked total_paid num_pickers entrance_fee price_per_pound = 7 := by
  sorry


end NUMINAMATH_CALUDE_strawberry_picking_problem_l3372_337236


namespace NUMINAMATH_CALUDE_decaf_coffee_percentage_l3372_337215

/-- Proves that the percentage of decaffeinated coffee in the second batch is 70% --/
theorem decaf_coffee_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (second_batch : ℝ)
  (final_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : second_batch = 100)
  (h4 : final_decaf_percent = 30)
  (h5 : initial_stock * initial_decaf_percent / 100 + second_batch * x / 100 = 
        (initial_stock + second_batch) * final_decaf_percent / 100) :
  x = 70 := by
  sorry

#check decaf_coffee_percentage

end NUMINAMATH_CALUDE_decaf_coffee_percentage_l3372_337215


namespace NUMINAMATH_CALUDE_carnival_tickets_l3372_337272

theorem carnival_tickets (num_friends : ℕ) (total_tickets : ℕ) (h1 : num_friends = 6) (h2 : total_tickets = 234) :
  (total_tickets / num_friends : ℕ) = 39 := by
  sorry

end NUMINAMATH_CALUDE_carnival_tickets_l3372_337272


namespace NUMINAMATH_CALUDE_two_players_game_count_l3372_337285

/-- Represents the number of players in the league -/
def totalPlayers : ℕ := 12

/-- Represents the number of players in each game -/
def playersPerGame : ℕ := 4

/-- Calculates the number of games two specific players play together -/
def gamesPlayedTogether : ℕ :=
  (totalPlayers.choose playersPerGame) / (totalPlayers - 1) * (playersPerGame - 1) / playersPerGame

/-- Theorem stating that two specific players play together in 45 games -/
theorem two_players_game_count :
  gamesPlayedTogether = 45 := by sorry

end NUMINAMATH_CALUDE_two_players_game_count_l3372_337285


namespace NUMINAMATH_CALUDE_kennedy_school_distance_l3372_337251

/-- Represents the fuel efficiency of Kennedy's car in miles per gallon -/
def fuel_efficiency : ℝ := 19

/-- Represents the initial amount of gas in Kennedy's car in gallons -/
def initial_gas : ℝ := 2

/-- Represents the distance to the softball park in miles -/
def distance_softball : ℝ := 6

/-- Represents the distance to the burger restaurant in miles -/
def distance_burger : ℝ := 2

/-- Represents the distance to her friend's house in miles -/
def distance_friend : ℝ := 4

/-- Represents the distance home in miles -/
def distance_home : ℝ := 11

/-- Theorem stating that Kennedy drove 15 miles to school -/
theorem kennedy_school_distance : 
  ∃ (distance_school : ℝ), 
    distance_school = fuel_efficiency * initial_gas - 
      (distance_softball + distance_burger + distance_friend + distance_home) ∧ 
    distance_school = 15 := by
  sorry

end NUMINAMATH_CALUDE_kennedy_school_distance_l3372_337251


namespace NUMINAMATH_CALUDE_allison_marbles_count_l3372_337209

theorem allison_marbles_count (albert angela allison : ℕ) 
  (h1 : albert = 3 * angela)
  (h2 : angela = allison + 8)
  (h3 : albert + allison = 136) :
  allison = 28 := by
sorry

end NUMINAMATH_CALUDE_allison_marbles_count_l3372_337209


namespace NUMINAMATH_CALUDE_solve_abc_l3372_337200

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + a*x - 12 = 0}
def B (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c = 0}

-- State the theorem
theorem solve_abc (a b c : ℝ) : 
  A a ≠ B b c ∧ 
  A a ∪ B b c = {-3, 4} ∧
  A a ∩ B b c = {-3} →
  a = -1 ∧ b = 6 ∧ c = 9 := by
sorry

end NUMINAMATH_CALUDE_solve_abc_l3372_337200


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3372_337290

theorem nested_fraction_evaluation :
  2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3372_337290


namespace NUMINAMATH_CALUDE_angle_through_point_l3372_337235

theorem angle_through_point (θ : Real) :
  (∃ (k : ℤ), θ = 2 * k * Real.pi + 5 * Real.pi / 6) ↔
  (∃ (t : Real), t > 0 ∧ t * Real.cos θ = -Real.sqrt 3 / 2 ∧ t * Real.sin θ = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_through_point_l3372_337235


namespace NUMINAMATH_CALUDE_specific_pairings_probability_eva_tom_june_leo_probability_l3372_337206

/-- The probability of two specific pairings in a class -/
theorem specific_pairings_probability (n : ℕ) (h : n ≥ 28) :
  (1 : ℚ) / (n - 1) * (1 : ℚ) / (n - 2) = 1 / ((n - 1) * (n - 2)) :=
sorry

/-- The probability of Eva being paired with Tom and June being paired with Leo -/
theorem eva_tom_june_leo_probability :
  (1 : ℚ) / 27 * (1 : ℚ) / 26 = 1 / 702 :=
sorry

end NUMINAMATH_CALUDE_specific_pairings_probability_eva_tom_june_leo_probability_l3372_337206


namespace NUMINAMATH_CALUDE_rainy_days_probability_exists_l3372_337273

theorem rainy_days_probability_exists :
  ∃ (n : ℕ), n > 0 ∧ 
    (Nat.choose n 3 : ℝ) * (1/2)^3 * (1/2)^(n-3) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_rainy_days_probability_exists_l3372_337273


namespace NUMINAMATH_CALUDE_prob_comparison_l3372_337275

/-- The probability of drawing two balls of the same color from two bags -/
def prob_same_color (m n : ℕ) : ℚ :=
  2 * m * n / ((m + n) * (m + n))

/-- The probability of drawing two balls of different colors from two bags -/
def prob_diff_color (m n : ℕ) : ℚ :=
  (m * m + n * n) / ((m + n) * (m + n))

theorem prob_comparison (m n : ℕ) :
  prob_same_color m n ≤ prob_diff_color m n ∧
  (prob_same_color m n = prob_diff_color m n ↔ m = n) :=
sorry

end NUMINAMATH_CALUDE_prob_comparison_l3372_337275


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3372_337270

theorem inequality_and_equality_condition (x : ℝ) (hx : x ≥ 0) :
  1 + x^2 + x^6 + x^8 ≥ 4 * x^4 ∧
  (1 + x^2 + x^6 + x^8 = 4 * x^4 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3372_337270


namespace NUMINAMATH_CALUDE_savings_calculation_l3372_337225

def total_expenses : ℚ := 30150
def savings_rate : ℚ := 1/5

theorem savings_calculation (salary : ℚ) (h1 : salary * savings_rate + total_expenses = salary) :
  salary * savings_rate = 7537.5 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l3372_337225


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l3372_337291

/-- Given three square regions I, II, and III, where the perimeter of region I is 12 units,
    the perimeter of region II is 24 units, and the side length of region III is the sum of
    the side lengths of regions I and II, prove that the ratio of the area of region I to
    the area of region III is 1/9. -/
theorem area_ratio_of_squares (side_length_I side_length_II side_length_III : ℝ) :
  side_length_I * 4 = 12 →
  side_length_II * 4 = 24 →
  side_length_III = side_length_I + side_length_II →
  (side_length_I ^ 2) / (side_length_III ^ 2) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l3372_337291


namespace NUMINAMATH_CALUDE_only_drug_effectiveness_suitable_l3372_337240

/-- Represents the suitability of an option for a sampling survey. -/
inductive Suitability
  | Suitable
  | NotSuitable

/-- Represents the different survey options. -/
inductive SurveyOption
  | DrugEffectiveness
  | ClassVision
  | EmployeeExamination
  | SatelliteInspection

/-- Determines the suitability of a survey option for sampling. -/
def suitabilityForSampling (option : SurveyOption) : Suitability :=
  match option with
  | SurveyOption.DrugEffectiveness => Suitability.Suitable
  | _ => Suitability.NotSuitable

/-- Theorem stating that only the drug effectiveness option is suitable for sampling. -/
theorem only_drug_effectiveness_suitable :
  ∀ (option : SurveyOption),
    suitabilityForSampling option = Suitability.Suitable ↔
    option = SurveyOption.DrugEffectiveness :=
by
  sorry

#check only_drug_effectiveness_suitable

end NUMINAMATH_CALUDE_only_drug_effectiveness_suitable_l3372_337240


namespace NUMINAMATH_CALUDE_train_speed_l3372_337286

-- Define the length of the train in meters
def train_length : ℝ := 90

-- Define the time taken to cross the pole in seconds
def crossing_time : ℝ := 9

-- Define the conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed : 
  (train_length / crossing_time) * conversion_factor = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3372_337286


namespace NUMINAMATH_CALUDE_calculation_proof_l3372_337228

theorem calculation_proof : (-1)^3 - 8 / (-2) + 4 * |(-5)| = 23 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3372_337228


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3372_337293

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (1 - m) * x₁^2 + 2 * x₁ - 2 = 0 ∧ 
   (1 - m) * x₂^2 + 2 * x₂ - 2 = 0) ↔ 
  (m < 3/2 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3372_337293


namespace NUMINAMATH_CALUDE_andy_max_cookies_l3372_337216

/-- The maximum number of cookies Andy can eat given the conditions -/
def max_cookies_andy (total : ℕ) (bella_ratio : ℕ) : ℕ :=
  total / (bella_ratio + 1)

/-- Proof that Andy's maximum cookie consumption is correct -/
theorem andy_max_cookies :
  let total := 36
  let bella_ratio := 2
  let andy_cookies := max_cookies_andy total bella_ratio
  andy_cookies = 12 ∧
  andy_cookies + bella_ratio * andy_cookies = total ∧
  ∀ x : ℕ, x > andy_cookies → x + bella_ratio * x > total :=
by sorry

#eval max_cookies_andy 36 2  -- Should output 12

end NUMINAMATH_CALUDE_andy_max_cookies_l3372_337216


namespace NUMINAMATH_CALUDE_not_always_equal_l3372_337223

-- Define the binary operation
def binary_op {S : Type} (op : S → S → S) : Prop :=
  ∀ (a b : S), ∃ (c : S), op a b = c

-- Define the property of the operation
def special_property {S : Type} (op : S → S → S) : Prop :=
  ∀ (a b : S), op a (op b a) = b

theorem not_always_equal {S : Type} [Inhabited S] (op : S → S → S) 
  (h1 : binary_op op) (h2 : special_property op) (h3 : ∃ (x y : S), x ≠ y) :
  ∃ (a b : S), op (op a b) a ≠ a := by
  sorry

end NUMINAMATH_CALUDE_not_always_equal_l3372_337223


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l3372_337205

def expression (x : ℝ) : ℝ :=
  4 * (x^2 - 2*x^3 + 2*x) + 2 * (x + 3*x^3 - 2*x^2 + 4*x^5 - x^3) - 6 * (2 + x - 5*x^3 - 2*x^2)

theorem coefficient_of_x_cubed : 
  (deriv (deriv (deriv expression))) 0 / 6 = 26 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l3372_337205


namespace NUMINAMATH_CALUDE_jinas_mascots_l3372_337269

/-- The number of mascots Jina has -/
def total_mascots (teddies bunnies koalas additional_teddies : ℕ) : ℕ :=
  teddies + bunnies + koalas + additional_teddies

/-- Theorem stating the total number of Jina's mascots -/
theorem jinas_mascots :
  let teddies : ℕ := 5
  let bunnies : ℕ := 3 * teddies
  let koalas : ℕ := 1
  let additional_teddies : ℕ := 2 * bunnies
  total_mascots teddies bunnies koalas additional_teddies = 51 := by
  sorry

end NUMINAMATH_CALUDE_jinas_mascots_l3372_337269


namespace NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l3372_337244

theorem parabola_tangent_hyperbola (m : ℝ) : 
  (∃ x y : ℝ, y = x^2 + 5 ∧ y^2 - m*x^2 = 4 ∧ 
   ∀ x' y' : ℝ, y' = x'^2 + 5 → y'^2 - m*x'^2 ≥ 4) →
  (m = 10 + 2*Real.sqrt 21 ∨ m = 10 - 2*Real.sqrt 21) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l3372_337244


namespace NUMINAMATH_CALUDE_prob_event_l3372_337277

/-- Represents a standard deck of cards -/
structure Deck :=
  (total : Nat)
  (queens : Nat)
  (jacks : Nat)
  (red : Nat)

/-- Calculates the probability of drawing two queens -/
def prob_two_queens (d : Deck) : Rat :=
  (d.queens * (d.queens - 1)) / (d.total * (d.total - 1))

/-- Calculates the probability of drawing at least one jack -/
def prob_at_least_one_jack (d : Deck) : Rat :=
  1 - (d.total - d.jacks) * (d.total - d.jacks - 1) / (d.total * (d.total - 1))

/-- Calculates the probability of drawing two red cards -/
def prob_two_red (d : Deck) : Rat :=
  (d.red * (d.red - 1)) / (d.total * (d.total - 1))

/-- Theorem stating the probability of the given event -/
theorem prob_event (d : Deck) (h1 : d.total = 52) (h2 : d.queens = 4) (h3 : d.jacks = 4) (h4 : d.red = 26) :
  prob_two_queens d + prob_at_least_one_jack d + prob_two_red d = 89 / 221 := by
  sorry

end NUMINAMATH_CALUDE_prob_event_l3372_337277


namespace NUMINAMATH_CALUDE_people_in_line_l3372_337266

theorem people_in_line (initial_people total_people : ℕ) 
  (h1 : initial_people = 61)
  (h2 : total_people = 83)
  (h3 : total_people > initial_people) :
  total_people - initial_people = 22 := by
sorry

end NUMINAMATH_CALUDE_people_in_line_l3372_337266


namespace NUMINAMATH_CALUDE_cat_and_mouse_positions_after_359_moves_l3372_337211

/-- Represents the possible positions of the cat -/
inductive CatPosition
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

/-- Represents the possible positions of the mouse -/
inductive MousePosition
  | TopMiddle
  | TopRight
  | RightMiddle
  | BottomRight
  | BottomMiddle
  | BottomLeft
  | LeftMiddle
  | TopLeft

/-- Calculate the position of the cat after a given number of moves -/
def catPositionAfterMoves (moves : ℕ) : CatPosition :=
  match moves % 4 with
  | 0 => CatPosition.BottomLeft
  | 1 => CatPosition.TopLeft
  | 2 => CatPosition.TopRight
  | _ => CatPosition.BottomRight

/-- Calculate the position of the mouse after a given number of moves -/
def mousePositionAfterMoves (moves : ℕ) : MousePosition :=
  match moves % 8 with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.TopMiddle
  | 2 => MousePosition.TopRight
  | 3 => MousePosition.RightMiddle
  | 4 => MousePosition.BottomRight
  | 5 => MousePosition.BottomMiddle
  | 6 => MousePosition.BottomLeft
  | _ => MousePosition.LeftMiddle

theorem cat_and_mouse_positions_after_359_moves :
  (catPositionAfterMoves 359 = CatPosition.BottomRight) ∧
  (mousePositionAfterMoves 359 = MousePosition.LeftMiddle) := by
  sorry


end NUMINAMATH_CALUDE_cat_and_mouse_positions_after_359_moves_l3372_337211


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3372_337282

theorem system_solution_ratio (k : ℝ) (x y z : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + k*y + 3*z = 0 →
  3*x + k*y - 2*z = 0 →
  2*x + 4*y - 3*z = 0 →
  x*z / (y^2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3372_337282


namespace NUMINAMATH_CALUDE_jerry_spent_two_tickets_l3372_337249

def tickets_spent (initial_tickets : ℕ) (won_tickets : ℕ) (final_tickets : ℕ) : ℕ :=
  initial_tickets + won_tickets - final_tickets

theorem jerry_spent_two_tickets :
  tickets_spent 4 47 49 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_spent_two_tickets_l3372_337249


namespace NUMINAMATH_CALUDE_marbles_remaining_proof_l3372_337284

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of marbles remaining after sales -/
def remaining_marbles (initial : ℕ) (customers : ℕ) : ℕ :=
  initial - sum_to_n customers

theorem marbles_remaining_proof :
  remaining_marbles 2500 50 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_marbles_remaining_proof_l3372_337284


namespace NUMINAMATH_CALUDE_min_value_m_plus_2n_l3372_337237

theorem min_value_m_plus_2n (m n : ℝ) (h : m - n^2 = 0) : 
  ∀ x y : ℝ, x - y^2 = 0 → m + 2*n ≤ x + 2*y :=
by sorry

end NUMINAMATH_CALUDE_min_value_m_plus_2n_l3372_337237


namespace NUMINAMATH_CALUDE_base8_642_equals_base10_418_l3372_337255

/-- Converts a base-8 number to base-10 -/
def base8_to_base10 (x : ℕ) : ℕ :=
  let d₂ := x / 100
  let d₁ := (x / 10) % 10
  let d₀ := x % 10
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

theorem base8_642_equals_base10_418 : base8_to_base10 642 = 418 := by
  sorry

end NUMINAMATH_CALUDE_base8_642_equals_base10_418_l3372_337255


namespace NUMINAMATH_CALUDE_inequality_proof_l3372_337288

theorem inequality_proof (x y z : ℝ) 
  (hx : 2 < x ∧ x < 4) 
  (hy : 2 < y ∧ y < 4) 
  (hz : 2 < z ∧ z < 4) : 
  x / (y^2 - z) + y / (z^2 - x) + z / (x^2 - y) > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3372_337288


namespace NUMINAMATH_CALUDE_red_triangle_or_blue_quadrilateral_l3372_337261

/-- A type representing the color of an edge --/
inductive Color
| Red
| Blue

/-- A complete graph with 9 vertices --/
def Graph9 := Fin 9 → Fin 9 → Color

/-- A predicate that checks if a graph is complete --/
def is_complete (g : Graph9) : Prop :=
  ∀ i j : Fin 9, i ≠ j → (g i j = Color.Red ∨ g i j = Color.Blue)

/-- A predicate that checks if three vertices form a red triangle --/
def has_red_triangle (g : Graph9) : Prop :=
  ∃ i j k : Fin 9, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    g i j = Color.Red ∧ g j k = Color.Red ∧ g i k = Color.Red

/-- A predicate that checks if four vertices form a blue complete quadrilateral --/
def has_blue_quadrilateral (g : Graph9) : Prop :=
  ∃ i j k l : Fin 9, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    g i j = Color.Blue ∧ g i k = Color.Blue ∧ g i l = Color.Blue ∧
    g j k = Color.Blue ∧ g j l = Color.Blue ∧ g k l = Color.Blue

/-- The main theorem --/
theorem red_triangle_or_blue_quadrilateral (g : Graph9) 
  (h : is_complete g) : has_red_triangle g ∨ has_blue_quadrilateral g := by
  sorry

end NUMINAMATH_CALUDE_red_triangle_or_blue_quadrilateral_l3372_337261


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l3372_337207

/-- The equation of the trajectory of the midpoint of a line segment between a point on a unit circle and a fixed point -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ (px py : ℝ), px^2 + py^2 = 1 ∧ x = (px + 3) / 2 ∧ y = py / 2) ↔ 
  (2*x - 3)^2 + 4*y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l3372_337207


namespace NUMINAMATH_CALUDE_books_together_l3372_337283

/-- The number of books Tim and Sam have together -/
def total_books (tim_books sam_books : ℕ) : ℕ := tim_books + sam_books

/-- Theorem: Tim and Sam have 96 books together -/
theorem books_together : total_books 44 52 = 96 := by sorry

end NUMINAMATH_CALUDE_books_together_l3372_337283


namespace NUMINAMATH_CALUDE_vector_difference_sum_l3372_337258

theorem vector_difference_sum : 
  let v1 : Fin 2 → ℝ := ![5, -8]
  let v2 : Fin 2 → ℝ := ![2, 6]
  let v3 : Fin 2 → ℝ := ![-1, 4]
  let scalar : ℝ := 5
  v1 - scalar • v2 + v3 = ![-6, -34] := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_sum_l3372_337258


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l3372_337264

theorem cubic_root_sum_product (p q r : ℂ) : 
  (6 * p ^ 3 - 5 * p ^ 2 + 13 * p - 10 = 0) →
  (6 * q ^ 3 - 5 * q ^ 2 + 13 * q - 10 = 0) →
  (6 * r ^ 3 - 5 * r ^ 2 + 13 * r - 10 = 0) →
  p * q + q * r + r * p = 13 / 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l3372_337264


namespace NUMINAMATH_CALUDE_gcd_8a_plus_3_5a_plus_2_is_1_l3372_337226

theorem gcd_8a_plus_3_5a_plus_2_is_1 (a : ℕ) : Nat.gcd (8 * a + 3) (5 * a + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8a_plus_3_5a_plus_2_is_1_l3372_337226


namespace NUMINAMATH_CALUDE_range_of_a_l3372_337268

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | -3 ≤ x ∧ x ≤ a}
def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 3*x + 10}
def C (a : ℝ) : Set ℝ := {z | ∃ x ∈ A a, z = 5 - x}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (B a ∩ C a = C a) → (-2/3 ≤ a ∧ a ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3372_337268


namespace NUMINAMATH_CALUDE_highest_power_of_two_dividing_difference_of_fifth_powers_l3372_337276

theorem highest_power_of_two_dividing_difference_of_fifth_powers :
  ∃ k : ℕ, k = 5 ∧ 2^k ∣ (17^5 - 15^5) ∧ ∀ m : ℕ, 2^m ∣ (17^5 - 15^5) → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_highest_power_of_two_dividing_difference_of_fifth_powers_l3372_337276


namespace NUMINAMATH_CALUDE_specific_mountain_depth_l3372_337214

/-- Represents a cone-shaped mountain partially submerged in water -/
structure Mountain where
  totalHeight : ℝ
  baseRadius : ℝ
  aboveWaterVolumeFraction : ℝ

/-- Calculates the depth of the ocean at the base of the mountain -/
def oceanDepth (m : Mountain) : ℝ :=
  m.totalHeight * (1 - (m.aboveWaterVolumeFraction) ^ (1/3))

/-- Theorem stating the ocean depth for the specific mountain described in the problem -/
theorem specific_mountain_depth :
  let m : Mountain := {
    totalHeight := 10000,
    baseRadius := 3000,
    aboveWaterVolumeFraction := 1/10
  }
  oceanDepth m = 5360 := by
  sorry


end NUMINAMATH_CALUDE_specific_mountain_depth_l3372_337214


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l3372_337260

theorem sum_of_four_numbers : 2143 + 3412 + 4213 + 1324 = 11092 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l3372_337260


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3372_337201

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) 
  (h1 : total_students = 485)
  (h2 : boys = 208)
  (h3 : boys < total_students - boys) :
  total_students - boys - boys = 69 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3372_337201


namespace NUMINAMATH_CALUDE_min_zeros_in_interval_l3372_337278

theorem min_zeros_in_interval (f : ℝ → ℝ) 
  (h0 : f 0 = 0)
  (h1 : ∀ x, f (9 + x) = f (9 - x))
  (h2 : ∀ x, f (x - 10) = f (-x - 10)) :
  ∃ n : ℕ, n = 107 ∧ 
    (∀ m : ℕ, (∃ S : Finset ℝ, S.card = m ∧ 
      (∀ x ∈ S, x ∈ Set.Icc 0 2014 ∧ f x = 0)) → m ≥ n) :=
sorry

end NUMINAMATH_CALUDE_min_zeros_in_interval_l3372_337278


namespace NUMINAMATH_CALUDE_parallel_line_perpendicular_line_l3372_337233

-- Define the point P as the intersection of two lines
def P : ℝ × ℝ := (2, 1)

-- Define line l1
def l1 (x y : ℝ) : Prop := 4 * x - y + 1 = 0

-- Define the equation of a line passing through P with slope m
def line_through_P (m : ℝ) (x y : ℝ) : Prop :=
  y - P.2 = m * (x - P.1)

-- Theorem for parallel case
theorem parallel_line :
  ∃ (a b c : ℝ), (∀ x y, a * x + b * y + c = 0 ↔ line_through_P 4 x y) ∧
                 a = 4 ∧ b = -1 ∧ c = -7 :=
sorry

-- Theorem for perpendicular case
theorem perpendicular_line :
  ∃ (a b c : ℝ), (∀ x y, a * x + b * y + c = 0 ↔ line_through_P (-1/4) x y) ∧
                 a = 1 ∧ b = 4 ∧ c = -6 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_perpendicular_line_l3372_337233


namespace NUMINAMATH_CALUDE_closest_to_370_l3372_337245

def calculation : ℝ := 3.1 * 9.1 * (5.92 + 4.08) + 100

def options : List ℝ := [300, 350, 370, 400, 430]

theorem closest_to_370 : 
  ∀ x ∈ options, |calculation - 370| ≤ |calculation - x| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_370_l3372_337245


namespace NUMINAMATH_CALUDE_batting_average_is_62_l3372_337265

/-- Calculates the batting average given the total innings, highest score, score difference, and average excluding extremes. -/
def battingAverage (totalInnings : ℕ) (highestScore : ℕ) (scoreDifference : ℕ) (averageExcludingExtremes : ℕ) : ℚ :=
  let lowestScore := highestScore - scoreDifference
  let totalScoreExcludingExtremes := (totalInnings - 2) * averageExcludingExtremes
  let totalScore := totalScoreExcludingExtremes + highestScore + lowestScore
  totalScore / totalInnings

/-- Theorem stating that under the given conditions, the batting average is 62 runs. -/
theorem batting_average_is_62 :
  battingAverage 46 225 150 58 = 62 := by sorry

end NUMINAMATH_CALUDE_batting_average_is_62_l3372_337265
