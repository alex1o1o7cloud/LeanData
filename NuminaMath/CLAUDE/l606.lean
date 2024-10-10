import Mathlib

namespace characterization_of_n_l606_60633

def invalid_n : Set ℕ := {2, 3, 5, 6, 7, 8, 13, 14, 15, 17, 19, 21, 23, 26, 27, 30, 47, 51, 53, 55, 61}

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (m : ℕ) (a : Fin (m-1) → ℕ), 
    (∀ i : Fin (m-1), 1 ≤ a i ∧ a i ≤ m - 1) ∧
    (∀ i j : Fin (m-1), i ≠ j → a i ≠ a j) ∧
    n = (Finset.univ.sum fun i => a i * (m - a i))

theorem characterization_of_n (n : ℕ) :
  n > 0 → (satisfies_condition n ↔ n ∉ invalid_n) := by sorry

end characterization_of_n_l606_60633


namespace fourth_term_is_375_l606_60611

/-- A geometric sequence of positive integers with first term 3 and third term 75 -/
structure GeometricSequence where
  a : ℕ+  -- first term
  r : ℕ+  -- common ratio
  third_term_eq : a * r^2 = 75
  first_term_eq : a = 3

/-- The fourth term of the geometric sequence is 375 -/
theorem fourth_term_is_375 (seq : GeometricSequence) : seq.a * seq.r^3 = 375 := by
  sorry

end fourth_term_is_375_l606_60611


namespace perimeterDifference_l606_60601

/-- Calculates the perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Calculates the perimeter of an L-shaped formation (2x2 square missing 1x1 square) -/
def lShapePerimeter : ℕ := 5

/-- Calculates the perimeter of Figure 1 (composite of 3x1 rectangle and L-shape) -/
def figure1Perimeter : ℕ :=
  rectanglePerimeter 3 1 + lShapePerimeter

/-- Calculates the perimeter of Figure 2 (6x2 rectangle) -/
def figure2Perimeter : ℕ :=
  rectanglePerimeter 6 2

/-- The main theorem stating the positive difference in perimeters -/
theorem perimeterDifference :
  (max figure1Perimeter figure2Perimeter) - (min figure1Perimeter figure2Perimeter) = 3 := by
  sorry

end perimeterDifference_l606_60601


namespace coupon_difference_l606_60674

/-- Represents the savings from a coupon given a price -/
def coupon_savings (price : ℝ) : (ℝ → ℝ) → ℝ := fun f => f price

/-- Coupon A: 20% off the listed price -/
def coupon_a (price : ℝ) : ℝ := 0.2 * price

/-- Coupon B: flat $40 off -/
def coupon_b (_ : ℝ) : ℝ := 40

/-- Coupon C: 30% off the amount by which the listed price exceeds $120 plus an additional $20 -/
def coupon_c (price : ℝ) : ℝ := 0.3 * (price - 120) + 20

/-- The proposition that Coupon A is at least as good as Coupon B or C for a given price -/
def coupon_a_best (price : ℝ) : Prop :=
  coupon_savings price coupon_a ≥ max (coupon_savings price coupon_b) (coupon_savings price coupon_c)

theorem coupon_difference :
  ∃ (x y : ℝ),
    x > 120 ∧
    y > 120 ∧
    x ≤ y ∧
    coupon_a_best x ∧
    coupon_a_best y ∧
    (∀ p, x < p ∧ p < y → coupon_a_best p) ∧
    (∀ p, p < x → ¬coupon_a_best p) ∧
    (∀ p, p > y → ¬coupon_a_best p) ∧
    y - x = 100 :=
  sorry

end coupon_difference_l606_60674


namespace coinciding_rest_days_count_l606_60698

/-- Chris's schedule cycle length -/
def chris_cycle : ℕ := 6

/-- Dana's schedule cycle length -/
def dana_cycle : ℕ := 6

/-- Number of days Chris works in his cycle -/
def chris_work_days : ℕ := 4

/-- Number of days Dana works in her cycle -/
def dana_work_days : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 1200

/-- The number of times Chris and Dana have rest-days on the same day -/
def coinciding_rest_days : ℕ := total_days / chris_cycle

theorem coinciding_rest_days_count :
  coinciding_rest_days = 200 :=
sorry

end coinciding_rest_days_count_l606_60698


namespace min_fleet_size_10x10_l606_60630

/-- A ship is a figure made up of unit squares connected by common edges -/
def Ship : Type := Unit

/-- A fleet is a set of ships where no two ships contain squares that share a common vertex -/
def Fleet : Type := Set Ship

/-- The size of the grid -/
def gridSize : ℕ := 10

/-- The minimum number of squares in a fleet to which no new ship can be added -/
def minFleetSize (n : ℕ) : ℕ :=
  if n % 3 = 0 then (n / 3) ^ 2
  else (n / 3 + 1) ^ 2

theorem min_fleet_size_10x10 :
  minFleetSize gridSize = 16 := by sorry

end min_fleet_size_10x10_l606_60630


namespace bouncing_ball_height_l606_60608

/-- Represents the height of a bouncing ball -/
def BouncingBall (h : ℝ) : Prop :=
  -- The ball rebounds to 50% of its previous height
  let h₁ := h / 2
  let h₂ := h₁ / 2
  -- The total travel distance when it touches the floor for the third time is 200 cm
  h + 2 * h₁ + 2 * h₂ = 200

/-- Theorem stating that the original height of the ball is 80 cm -/
theorem bouncing_ball_height :
  ∃ h : ℝ, BouncingBall h ∧ h = 80 :=
sorry

end bouncing_ball_height_l606_60608


namespace winnie_lollipop_distribution_l606_60622

/-- Winnie's lollipop distribution problem -/
theorem winnie_lollipop_distribution 
  (total_lollipops : ℕ) 
  (num_friends : ℕ) 
  (h1 : total_lollipops = 72 + 89 + 23 + 316) 
  (h2 : num_friends = 14) : 
  total_lollipops % num_friends = 10 := by
  sorry

end winnie_lollipop_distribution_l606_60622


namespace arithmetic_sequence_equidistant_sum_l606_60628

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_equidistant_sum
  (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 4 + a 8 = 16 → a 2 + a 10 = 16 :=
by
  sorry

end arithmetic_sequence_equidistant_sum_l606_60628


namespace watch_price_proof_l606_60638

/-- The sticker price of the watch in dollars -/
def stickerPrice : ℝ := 250

/-- The price at store X after discounts -/
def priceX (price : ℝ) : ℝ := 0.8 * price - 50

/-- The price at store Y after discount -/
def priceY (price : ℝ) : ℝ := 0.9 * price

theorem watch_price_proof :
  priceY stickerPrice - priceX stickerPrice = 25 :=
sorry

end watch_price_proof_l606_60638


namespace unique_intersection_l606_60643

/-- The first function f(x) = x^2 - 7x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 7*x + 3

/-- The second function g(x) = -3x^2 + 5x - 6 -/
def g (x : ℝ) : ℝ := -3*x^2 + 5*x - 6

/-- The theorem stating that f and g intersect at exactly one point (3/2, -21/4) -/
theorem unique_intersection :
  ∃! p : ℝ × ℝ, 
    p.1 = 3/2 ∧ 
    p.2 = -21/4 ∧ 
    f p.1 = g p.1 ∧
    ∀ x : ℝ, f x = g x → x = p.1 := by
  sorry

#check unique_intersection

end unique_intersection_l606_60643


namespace min_value_expression_l606_60667

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^4 + b^4 + 16 / (a^2 + b^2)^2 ≥ 4 ∧
  (a^4 + b^4 + 16 / (a^2 + b^2)^2 = 4 ↔ a = b ∧ a = 2^(1/4)) :=
by sorry

end min_value_expression_l606_60667


namespace product_nonnegative_implies_lower_bound_l606_60682

open Real

theorem product_nonnegative_implies_lower_bound (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (∀ x : ℝ, x > 0 → (log (a * x) - 1) * (exp x - b) ≥ 0) →
  a * b ≥ exp 2 :=
by sorry

end product_nonnegative_implies_lower_bound_l606_60682


namespace parabola_sum_l606_60687

/-- Parabola in the first quadrant -/
def parabola (x y : ℝ) : Prop := x^2 = (1/2) * y ∧ x > 0 ∧ y > 0

/-- Point on the parabola -/
def point_on_parabola (a : ℕ → ℝ) (i : ℕ) : Prop :=
  parabola (a i) (2 * (a i)^2)

/-- Tangent line intersection property -/
def tangent_intersection (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, i > 0 → point_on_parabola a i →
    ∃ m b : ℝ, (m * (a (i+1)) + b = 0) ∧
              (∀ x y : ℝ, y - 2*(a i)^2 = m*(x - a i) → parabola x y)

/-- The main theorem -/
theorem parabola_sum (a : ℕ → ℝ) :
  (∀ i : ℕ, i > 0 → point_on_parabola a i) →
  tangent_intersection a →
  a 2 = 32 →
  a 2 + a 4 + a 6 = 42 := by sorry

end parabola_sum_l606_60687


namespace systematic_sampling_probability_l606_60609

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a population with a given size -/
structure Population where
  size : ℕ

/-- Represents a sample drawn from a population -/
structure Sample where
  size : ℕ
  population : Population
  method : SamplingMethod

/-- The probability of an individual being selected in a given sample -/
def selectionProbability (s : Sample) : ℚ :=
  s.size / s.population.size

/-- Theorem: In a population of 2008 parts, after eliminating 8 parts randomly
    and then selecting 20 parts using systematic sampling, the probability
    of each part being selected is 20/2008 -/
theorem systematic_sampling_probability :
  let initialPopulation : Population := ⟨2008⟩
  let eliminatedPopulation : Population := ⟨2000⟩
  let sample : Sample := ⟨20, eliminatedPopulation, SamplingMethod.Systematic⟩
  selectionProbability sample = 20 / 2008 := by
  sorry

end systematic_sampling_probability_l606_60609


namespace wage_increase_percentage_l606_60629

theorem wage_increase_percentage (original_wage new_wage : ℝ) 
  (h1 : original_wage = 28)
  (h2 : new_wage = 42) :
  (new_wage - original_wage) / original_wage * 100 = 50 := by
sorry

end wage_increase_percentage_l606_60629


namespace fair_spending_remainder_l606_60692

/-- Calculates the remaining amount after spending on snacks and games at a fair. -/
theorem fair_spending_remainder (initial_amount snack_cost : ℕ) : 
  initial_amount = 80 →
  snack_cost = 18 →
  initial_amount - (snack_cost + 3 * snack_cost) = 8 := by
  sorry

#check fair_spending_remainder

end fair_spending_remainder_l606_60692


namespace shale_mix_cost_per_pound_l606_60632

/-- Prove that the cost of the shale mix per pound is $5 -/
theorem shale_mix_cost_per_pound
  (limestone_cost : ℝ)
  (total_weight : ℝ)
  (total_cost_per_pound : ℝ)
  (limestone_weight : ℝ)
  (h1 : limestone_cost = 3)
  (h2 : total_weight = 100)
  (h3 : total_cost_per_pound = 4.25)
  (h4 : limestone_weight = 37.5) :
  let shale_weight := total_weight - limestone_weight
  let total_cost := total_weight * total_cost_per_pound
  let limestone_total_cost := limestone_weight * limestone_cost
  let shale_total_cost := total_cost - limestone_total_cost
  shale_total_cost / shale_weight = 5 := by
sorry

end shale_mix_cost_per_pound_l606_60632


namespace first_saline_concentration_l606_60605

theorem first_saline_concentration 
  (desired_concentration : ℝ)
  (total_volume : ℝ)
  (first_volume : ℝ)
  (second_volume : ℝ)
  (second_concentration : ℝ)
  (h1 : desired_concentration = 3.24)
  (h2 : total_volume = 5)
  (h3 : first_volume = 3.6)
  (h4 : second_volume = 1.4)
  (h5 : second_concentration = 9)
  (h6 : total_volume = first_volume + second_volume)
  : ∃ (first_concentration : ℝ),
    first_concentration = 1 ∧
    desired_concentration * total_volume = 
      first_concentration * first_volume + second_concentration * second_volume :=
by sorry

end first_saline_concentration_l606_60605


namespace julia_watch_collection_l606_60665

/-- Proves that the percentage of gold watches in Julia's collection is 9.09% -/
theorem julia_watch_collection (silver : ℕ) (bronze : ℕ) (gold : ℕ) (total : ℕ) : 
  silver = 20 →
  bronze = 3 * silver →
  total = silver + bronze + gold →
  total = 88 →
  (gold : ℝ) / (total : ℝ) * 100 = 9.09 := by
  sorry

end julia_watch_collection_l606_60665


namespace other_root_of_quadratic_l606_60696

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + k * x = -6) ∧ (3 * 3^2 + k * 3 = -6) → 
  (∃ r : ℝ, r ≠ 3 ∧ 3 * r^2 + k * r = -6 ∧ r = 2/3) :=
by sorry

end other_root_of_quadratic_l606_60696


namespace new_room_ratio_l606_60620

/-- The ratio of a new room's size to the combined size of a bedroom and bathroom -/
theorem new_room_ratio (bedroom_size bathroom_size new_room_size : ℝ) 
  (h1 : bedroom_size = 309)
  (h2 : bathroom_size = 150)
  (h3 : new_room_size = 918) :
  new_room_size / (bedroom_size + bathroom_size) = 2 := by
  sorry

end new_room_ratio_l606_60620


namespace quadrilateral_angle_measure_l606_60686

theorem quadrilateral_angle_measure :
  ∀ (a b c d : ℝ),
  a = 50 →
  b = 180 - 30 →
  d = 180 - 40 →
  a + b + c + d = 360 →
  c = 20 :=
by
  sorry

end quadrilateral_angle_measure_l606_60686


namespace complex_sum_equality_l606_60650

theorem complex_sum_equality : 
  let A : ℂ := 3 + 2*I
  let O : ℂ := -3 + I
  let P : ℂ := 1 - 2*I
  let S : ℂ := 4 + 5*I
  let T : ℂ := -1
  A - O + P + S + T = 10 + 4*I :=
by sorry

end complex_sum_equality_l606_60650


namespace complex_in_second_quadrant_l606_60688

theorem complex_in_second_quadrant :
  let z : ℂ := Complex.mk (Real.cos 2) (Real.sin 3)
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_in_second_quadrant_l606_60688


namespace interest_rate_problem_l606_60695

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Problem statement -/
theorem interest_rate_problem (principal interest time : ℚ) 
  (h1 : principal = 4000)
  (h2 : interest = 640)
  (h3 : time = 2)
  (h4 : simple_interest principal (8 : ℚ) time = interest) :
  8 = (interest * 100) / (principal * time) :=
by sorry

end interest_rate_problem_l606_60695


namespace plate_cup_cost_l606_60613

/-- Given the cost of 20 plates and 40 cups, calculate the cost of 100 plates and 200 cups -/
theorem plate_cup_cost (plate_cost cup_cost : ℝ) : 
  20 * plate_cost + 40 * cup_cost = 1.50 → 
  100 * plate_cost + 200 * cup_cost = 7.50 := by
  sorry

end plate_cup_cost_l606_60613


namespace solution_interval_l606_60672

theorem solution_interval (x : ℝ) : (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ x ∈ Set.Ici (-4) ∩ Set.Iio (-3/2) := by
  sorry

end solution_interval_l606_60672


namespace complement_union_problem_l606_60659

def U : Finset Nat := {0, 1, 2, 3, 4, 5}
def A : Finset Nat := {1, 2, 3, 5}
def B : Finset Nat := {2, 4}

theorem complement_union_problem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end complement_union_problem_l606_60659


namespace angle_between_vectors_l606_60631

/-- Given two vectors a and b in ℝ², where a is perpendicular to b,
    prove that the angle between (a - b) and b is 150°. -/
theorem angle_between_vectors (a b : ℝ × ℝ) (h_a : a = (Real.sqrt 3, 1))
    (h_b : b.2 = -3) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
    let diff := (a.1 - b.1, a.2 - b.2)
    Real.arccos ((diff.1 * b.1 + diff.2 * b.2) / 
      (Real.sqrt (diff.1^2 + diff.2^2) * Real.sqrt (b.1^2 + b.2^2))) =
    150 * π / 180 := by
  sorry

end angle_between_vectors_l606_60631


namespace logarithm_simplification_l606_60617

open Real

theorem logarithm_simplification (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a ≠ 1) :
  let log_a := fun x => log x / log a
  let log_ab := fun x => log x / log (a * b)
  (log_a b + log_a (b^(1/(2*log b / log (a^2)))))/(log_a b - log_ab b) *
  (log_ab b * log_a b)/(b^(2*log b * log_a b) - 1) = 1 / (log_a b - 1) := by
  sorry

end logarithm_simplification_l606_60617


namespace gauss_family_mean_age_l606_60653

def gauss_family_ages : List ℕ := [7, 7, 7, 14, 15]

theorem gauss_family_mean_age :
  (gauss_family_ages.sum / gauss_family_ages.length : ℚ) = 10 := by
  sorry

end gauss_family_mean_age_l606_60653


namespace triangular_array_coins_l606_60689

theorem triangular_array_coins (N : ℕ) : 
  (N * (N + 1)) / 2 = 2010 → N = 63 ∧ (N / 10 + N % 10 = 9) := by
  sorry

end triangular_array_coins_l606_60689


namespace function_characterization_l606_60699

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the property of the function
def SatisfiesProperty (f : RealFunction) : Prop :=
  ∀ x : ℝ, f x = f (x/2) + (x/2) * (deriv f x)

-- State the theorem
theorem function_characterization :
  ∀ f : RealFunction, SatisfiesProperty f →
  ∃ c b : ℝ, ∀ x : ℝ, f x = c * x + b :=
by sorry

end function_characterization_l606_60699


namespace hyperbola_line_intersection_l606_60679

/-- Hyperbola C: x²/a² - y²/b² = 1 -/
def Hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Line l: y = kx + m -/
def Line (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m

/-- Point on a line -/
def PointOnLine (k m x y : ℝ) : Prop :=
  Line k m x y

/-- Point on the hyperbola -/
def PointOnHyperbola (a b x y : ℝ) : Prop :=
  Hyperbola a b x y

/-- Midpoint of two points -/
def Midpoint (x1 y1 x2 y2 xm ym : ℝ) : Prop :=
  xm = (x1 + x2) / 2 ∧ ym = (y1 + y2) / 2

/-- kAB · kOM = 3/4 condition -/
def SlopeProduct (xa ya xb yb xm ym : ℝ) : Prop :=
  ((yb - ya) / (xb - xa)) * (ym / xm) = 3/4

/-- Circle passing through three points -/
def CircleThroughPoints (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  ∃ (xc yc r : ℝ), (x1 - xc)^2 + (y1 - yc)^2 = r^2 ∧
                   (x2 - xc)^2 + (y2 - yc)^2 = r^2 ∧
                   (x3 - xc)^2 + (y3 - yc)^2 = r^2

theorem hyperbola_line_intersection
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a^2 / b^2 = 4/3)  -- Derived from eccentricity √7/2
  (k m : ℝ)
  (xa ya xb yb : ℝ)
  (h4 : PointOnHyperbola a b xa ya)
  (h5 : PointOnHyperbola a b xb yb)
  (h6 : PointOnLine k m xa ya)
  (h7 : PointOnLine k m xb yb)
  (xm ym : ℝ)
  (h8 : Midpoint xa ya xb yb xm ym)
  (h9 : SlopeProduct xa ya xb yb xm ym)
  (h10 : ¬(PointOnLine k m 2 0))
  (h11 : CircleThroughPoints xa ya xb yb 2 0) :
  PointOnLine k m 14 0 :=
sorry

end hyperbola_line_intersection_l606_60679


namespace savings_calculation_l606_60614

theorem savings_calculation (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : 
  income = 19000 → 
  income_ratio = 5 → 
  expenditure_ratio = 4 → 
  income - (income * expenditure_ratio / income_ratio) = 3800 := by
sorry

end savings_calculation_l606_60614


namespace jason_money_last_week_l606_60606

/-- Given information about Fred and Jason's money before and after washing cars,
    prove how much money Jason had last week. -/
theorem jason_money_last_week
  (fred_money_last_week : ℕ)
  (fred_money_now : ℕ)
  (jason_money_now : ℕ)
  (fred_earned : ℕ)
  (h1 : fred_money_last_week = 19)
  (h2 : fred_money_now = 40)
  (h3 : jason_money_now = 69)
  (h4 : fred_earned = 21)
  (h5 : fred_money_now = fred_money_last_week + fred_earned) :
  jason_money_now - fred_earned = 48 :=
by sorry

end jason_money_last_week_l606_60606


namespace least_positive_integer_multiple_of_53_l606_60675

theorem least_positive_integer_multiple_of_53 : 
  ∃ (x : ℕ+), (x = 4) ∧ 
  (∀ (y : ℕ+), y < x → ¬(53 ∣ (3*y)^2 + 2*41*3*y + 41^2)) ∧ 
  (53 ∣ (3*x)^2 + 2*41*3*x + 41^2) := by
sorry

end least_positive_integer_multiple_of_53_l606_60675


namespace system_of_equations_solutions_l606_60670

theorem system_of_equations_solutions :
  (∃ x y : ℝ, y = 2*x - 3 ∧ 2*x + y = 5 ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℝ, 3*x + 4*y = 5 ∧ 5*x - 2*y = 17 ∧ x = 3 ∧ y = -1) := by
  sorry

end system_of_equations_solutions_l606_60670


namespace divisibility_by_13_l606_60623

theorem divisibility_by_13 (N : ℕ) (x : ℕ) : 
  (N = 2 * 10^2022 + x * 10^2000 + 23) →
  (N % 13 = 0) →
  (x = 3) :=
by sorry

end divisibility_by_13_l606_60623


namespace tangent_angle_at_x_1_l606_60648

/-- The angle of inclination of the tangent to the curve y = x³ - 2x + m at x = 1 is 45° -/
theorem tangent_angle_at_x_1 (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 2*x + m
  let f' : ℝ → ℝ := λ x => 3*x^2 - 2
  let slope : ℝ := f' 1
  Real.arctan slope = π/4 := by sorry

end tangent_angle_at_x_1_l606_60648


namespace stewart_farm_ratio_l606_60680

theorem stewart_farm_ratio : 
  ∀ (sheep horses : ℕ) (horse_food_per_day total_horse_food : ℕ),
    sheep = 8 →
    horse_food_per_day = 230 →
    total_horse_food = 12880 →
    horses * horse_food_per_day = total_horse_food →
    sheep.gcd horses = 1 →
    sheep / horses = 1 / 7 := by
  sorry

end stewart_farm_ratio_l606_60680


namespace prob_two_twos_in_five_rolls_l606_60644

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll : ℚ := 1 / 6

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of times we want to roll the specific number -/
def target_rolls : ℕ := 2

/-- The probability of rolling a specific number exactly k times in n rolls of a fair six-sided die -/
def prob_specific_rolls (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (prob_single_roll ^ k) * ((1 - prob_single_roll) ^ (n - k))

theorem prob_two_twos_in_five_rolls :
  prob_specific_rolls num_rolls target_rolls = 625 / 3888 := by
  sorry

end prob_two_twos_in_five_rolls_l606_60644


namespace square_perimeter_doubled_l606_60602

theorem square_perimeter_doubled (area : ℝ) (h : area = 900) : 
  let side_length := Real.sqrt area
  let initial_perimeter := 4 * side_length
  let doubled_perimeter := 2 * initial_perimeter
  doubled_perimeter = 240 := by sorry

end square_perimeter_doubled_l606_60602


namespace tom_average_increase_l606_60661

def tom_scores : List ℝ := [92, 89, 91, 93]

theorem tom_average_increase :
  let first_three := tom_scores.take 3
  let all_four := tom_scores
  let avg_first_three := first_three.sum / first_three.length
  let avg_all_four := all_four.sum / all_four.length
  avg_all_four - avg_first_three = 0.58 := by
  sorry

end tom_average_increase_l606_60661


namespace inequality_always_true_l606_60666

theorem inequality_always_true (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - a * x + 1 > 0) ↔ (-2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2) := by
  sorry

end inequality_always_true_l606_60666


namespace second_meeting_time_l606_60671

-- Define the number of rounds Charging Bull completes in an hour
def charging_bull_rounds_per_hour : ℕ := 40

-- Define the time Racing Magic takes to complete one round (in seconds)
def racing_magic_time_per_round : ℕ := 150

-- Define the number of seconds in an hour
def seconds_per_hour : ℕ := 3600

-- Define the function to calculate the meeting time in minutes
def meeting_time : ℕ :=
  let racing_magic_rounds_per_hour := seconds_per_hour / racing_magic_time_per_round
  let lcm_rounds := Nat.lcm racing_magic_rounds_per_hour charging_bull_rounds_per_hour
  let hours_to_meet := lcm_rounds / racing_magic_rounds_per_hour
  hours_to_meet * 60

-- Theorem statement
theorem second_meeting_time :
  meeting_time = 300 := by sorry

end second_meeting_time_l606_60671


namespace twentieth_century_power_diff_l606_60607

def is_20th_century (year : ℕ) : Prop := 1900 ≤ year ∧ year ≤ 1999

def is_power_diff (year : ℕ) : Prop :=
  ∃ (n k : ℕ), year = 2^n - 2^k

theorem twentieth_century_power_diff :
  {year : ℕ | is_20th_century year ∧ is_power_diff year} = {1984, 1920} := by
  sorry

end twentieth_century_power_diff_l606_60607


namespace sum_abcd_equals_negative_ten_thirds_l606_60691

theorem sum_abcd_equals_negative_ten_thirds
  (a b c d : ℚ)
  (h : a + 1 = b + 2 ∧ b + 2 = c + 3 ∧ c + 3 = d + 4 ∧ d + 4 = a + b + c + d + 5) :
  a + b + c + d = -10/3 :=
by sorry

end sum_abcd_equals_negative_ten_thirds_l606_60691


namespace reflection_line_sum_l606_60683

/-- Given that (2,3) is reflected across y = mx + b to (10,7), prove m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = (2 + 10) / 2 ∧ y = (3 + 7) / 2 ∧ y = m * x + b) →
  (m = -(10 - 2) / (7 - 3)) →
  m + b = 15 := by
  sorry

end reflection_line_sum_l606_60683


namespace atLeastOneWhite_oneRedOneBlack_mutually_exclusive_not_exhaustive_l606_60660

-- Define the set of possible ball colors
inductive Color
| Red
| White
| Black

-- Define the bag contents
def bag : Multiset Color :=
  Multiset.replicate 3 Color.Red + Multiset.replicate 2 Color.White + Multiset.replicate 1 Color.Black

-- Define a draw as a pair of colors
def Draw := (Color × Color)

-- Define the event "At least one white ball"
def atLeastOneWhite (draw : Draw) : Prop :=
  draw.1 = Color.White ∨ draw.2 = Color.White

-- Define the event "one red ball and one black ball"
def oneRedOneBlack (draw : Draw) : Prop :=
  (draw.1 = Color.Red ∧ draw.2 = Color.Black) ∨ (draw.1 = Color.Black ∧ draw.2 = Color.Red)

-- Theorem stating that the events are mutually exclusive but not exhaustive
theorem atLeastOneWhite_oneRedOneBlack_mutually_exclusive_not_exhaustive :
  (∀ (draw : Draw), ¬(atLeastOneWhite draw ∧ oneRedOneBlack draw)) ∧
  (∃ (draw : Draw), ¬atLeastOneWhite draw ∧ ¬oneRedOneBlack draw) :=
sorry

end atLeastOneWhite_oneRedOneBlack_mutually_exclusive_not_exhaustive_l606_60660


namespace households_without_car_or_bike_l606_60664

/-- Prove that the number of households without either a car or a bike is 11 -/
theorem households_without_car_or_bike (total : ℕ) (car_and_bike : ℕ) (car : ℕ) (bike_only : ℕ)
  (h_total : total = 90)
  (h_car_and_bike : car_and_bike = 18)
  (h_car : car = 44)
  (h_bike_only : bike_only = 35) :
  total - (car + bike_only) = 11 := by
  sorry

end households_without_car_or_bike_l606_60664


namespace triangle_inequalities_l606_60655

/-- Given a triangle with sides a, b, c, prove two inequalities about its radii and semiperimeter. -/
theorem triangle_inequalities 
  (a b c r R r_a r_b r_c p S : ℝ) 
  (h1 : 4 * R + r = r_a + r_b + r_c)
  (h2 : R - 2 * r ≥ 0)
  (h3 : r_a + r_b + r_c = p * r * (1 / (p - a) + 1 / (p - b) + 1 / (p - c)))
  (h4 : 1 / (p - a) + 1 / (p - b) + 1 / (p - c) = (a * b + b * c + c * a - p ^ 2) / S)
  (h5 : p = (a + b + c) / 2)
  (h6 : 2 * (a * b + b * c + c * a) - (a ^ 2 + b ^ 2 + c ^ 2) ≥ 4 * Real.sqrt 3 * S)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ R > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ p > 0 ∧ S > 0) :
  (5 * R - r ≥ Real.sqrt 3 * p) ∧ 
  (4 * R - r_a ≥ (p - a) * (Real.sqrt 3 + (a ^ 2 + (b - c) ^ 2) / (2 * S))) := by
  sorry


end triangle_inequalities_l606_60655


namespace xingyou_age_l606_60645

theorem xingyou_age : ℕ :=
  let current_age : ℕ := sorry
  let current_height : ℕ := sorry
  have h1 : current_age = current_height := by sorry
  have h2 : current_age + 3 = 2 * current_height := by sorry
  have h3 : current_age = 3 := by sorry
  3

#check xingyou_age

end xingyou_age_l606_60645


namespace k_value_when_A_is_quadratic_binomial_C_value_when_k_is_negative_one_l606_60676

-- Define the polynomials A and B
def A (k : ℝ) (x : ℝ) : ℝ := -2 * x^2 - (k - 1) * x + 1
def B (x : ℝ) : ℝ := -2 * (x^2 - x + 2)

-- Define what it means for a polynomial to be a quadratic binomial
def is_quadratic_binomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ x, p x = a * x^2 + b * x + c

-- Theorem 1: When A is a quadratic binomial, k = 1
theorem k_value_when_A_is_quadratic_binomial :
  ∀ k : ℝ, is_quadratic_binomial (A k) → k = 1 :=
sorry

-- Theorem 2: When k = -1 and C + 2A = B, then C = 2x^2 - 2x - 6
theorem C_value_when_k_is_negative_one :
  ∀ C : ℝ → ℝ, (∀ x, C x + 2 * A (-1) x = B x) →
  ∀ x, C x = 2 * x^2 - 2 * x - 6 :=
sorry

end k_value_when_A_is_quadratic_binomial_C_value_when_k_is_negative_one_l606_60676


namespace checkerboard_square_selection_l606_60658

theorem checkerboard_square_selection (b : ℕ) : 
  let n := 2 * b + 1
  (n^2 * (n - 1)) / 2 = n * (n - 1) * n / 2 - n * (n - 1) / 2 :=
by sorry

end checkerboard_square_selection_l606_60658


namespace smallest_n_with_partial_divisibility_l606_60673

theorem smallest_n_with_partial_divisibility : ∃ (n : ℕ), n > 0 ∧
  (∀ (m : ℕ), m > 0 → m < n →
    (∃ (k : ℕ), k > 0 ∧ k ≤ m ∧ (m * (m + 1)) % k = 0) ∧
    (∀ (k : ℕ), k > 0 ∧ k ≤ m → (m * (m + 1)) % k = 0)) ∧
  (∃ (k : ℕ), k > 0 ∧ k ≤ n ∧ (n * (n + 1)) % k = 0) ∧
  (∃ (k : ℕ), k > 0 ∧ k ≤ n ∧ (n * (n + 1)) % k ≠ 0) ∧
  n = 4 :=
sorry

end smallest_n_with_partial_divisibility_l606_60673


namespace remainder_of_M_div_500_l606_60669

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def product_of_factorials : ℕ := (List.range 50).foldl (fun acc n => acc * factorial (n + 1)) 1

def trailing_zeros (n : ℕ) : ℕ := 
  if n = 0 then 0 else (n.digits 10).takeWhile (· = 0) |>.length

def M : ℕ := trailing_zeros product_of_factorials

theorem remainder_of_M_div_500 : M % 500 = 21 := by sorry

end remainder_of_M_div_500_l606_60669


namespace valid_rearrangements_count_valid_rearrangements_count_is_360_l606_60668

/-- Represents the word to be rearranged -/
def word : String := "REPRESENT"

/-- Counts the occurrences of a character in a string -/
def count_char (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

/-- The number of vowels in the word -/
def num_vowels : Nat :=
  count_char word 'E'

/-- The number of consonants in the word -/
def num_consonants : Nat :=
  word.length - num_vowels

/-- The number of unique consonants in the word -/
def num_unique_consonants : Nat :=
  (word.toList.filter (λ c => c ≠ 'E') |>.eraseDups).length

/-- The main theorem stating the number of valid rearrangements -/
theorem valid_rearrangements_count : Nat :=
  (Nat.factorial num_consonants) / (Nat.factorial (num_consonants - num_unique_consonants + 1))

/-- The proof of the main theorem -/
theorem valid_rearrangements_count_is_360 : valid_rearrangements_count = 360 := by
  sorry

end valid_rearrangements_count_valid_rearrangements_count_is_360_l606_60668


namespace special_hexagon_area_l606_60618

/-- A hexagon with specific side lengths that can be divided into a rectangle and two triangles -/
structure SpecialHexagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side6 : ℝ
  rectangle_width : ℝ
  rectangle_height : ℝ
  triangle_base : ℝ
  triangle_height : ℝ
  side1_eq : side1 = 20
  side2_eq : side2 = 15
  side3_eq : side3 = 22
  side4_eq : side4 = 27
  side5_eq : side5 = 18
  side6_eq : side6 = 15
  rectangle_width_eq : rectangle_width = 18
  rectangle_height_eq : rectangle_height = 22
  triangle_base_eq : triangle_base = 18
  triangle_height_eq : triangle_height = 15

/-- The area of the special hexagon is 666 square units -/
theorem special_hexagon_area (h : SpecialHexagon) : 
  h.rectangle_width * h.rectangle_height + 2 * (1/2 * h.triangle_base * h.triangle_height) = 666 := by
  sorry

end special_hexagon_area_l606_60618


namespace tan_3_expression_zero_l606_60641

theorem tan_3_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := by
  sorry

end tan_3_expression_zero_l606_60641


namespace set_operations_and_subset_l606_60642

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- State the theorem
theorem set_operations_and_subset :
  (A ∩ B = {x | 3 ≤ x ∧ x < 6}) ∧
  (A ∪ B = {x | 2 < x ∧ x < 9}) ∧
  ({a : ℝ | C a ⊆ B} = {a | 2 ≤ a ∧ a ≤ 8}) :=
by sorry

end set_operations_and_subset_l606_60642


namespace sequence_proof_l606_60625

theorem sequence_proof (a : Fin 8 → ℕ) 
  (h1 : ∀ i : Fin 6, a i + a (i + 1) + a (i + 2) = 100)
  (h2 : a 0 = 20)
  (h3 : a 7 = 16) :
  a = ![20, 16, 64, 20, 16, 64, 20, 16] := by
sorry

end sequence_proof_l606_60625


namespace x_gt_one_necessary_not_sufficient_for_x_gt_two_l606_60616

theorem x_gt_one_necessary_not_sufficient_for_x_gt_two :
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 2)) :=
by sorry

end x_gt_one_necessary_not_sufficient_for_x_gt_two_l606_60616


namespace simplify_xy_expression_l606_60647

theorem simplify_xy_expression (x y : ℝ) : 4 * x * y - 2 * x * y = 2 * x * y := by
  sorry

end simplify_xy_expression_l606_60647


namespace geometric_sequence_ratio_l606_60693

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The main theorem -/
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_prod : a 7 * a 11 = 6)
  (h_sum : a 4 + a 14 = 5) :
  a 20 / a 10 = 3/2 ∨ a 20 / a 10 = 2/3 :=
sorry

end geometric_sequence_ratio_l606_60693


namespace video_game_lives_l606_60690

/-- 
Given a video game scenario where:
- x is the initial number of lives
- y is the number of power-ups collected
- Each power-up gives 5 extra lives
- The player lost 13 lives
- After these events, the player ended up with 70 lives

Prove that the initial number of lives (x) is equal to 83 minus 5 times 
the number of power-ups collected (y).
-/
theorem video_game_lives (x y : ℤ) : 
  (x - 13 + 5 * y = 70) → (x = 83 - 5 * y) := by
  sorry

end video_game_lives_l606_60690


namespace urn_probability_l606_60656

def Urn := Nat × Nat -- (white balls, black balls)

def initial_urn : Urn := (2, 1)

def operation (u : Urn) : Urn → Prop :=
  fun u' => (u'.1 = u.1 + 1 ∧ u'.2 = u.2) ∨ (u'.1 = u.1 ∧ u'.2 = u.2 + 1)

def final_urn (u : Urn) : Prop := u.1 = 4 ∧ u.2 = 4

def probability_of_drawing_white (u : Urn) : ℚ := u.1 / (u.1 + u.2)

def probability_of_drawing_black (u : Urn) : ℚ := u.2 / (u.1 + u.2)

theorem urn_probability : 
  ∃ (u₁ u₂ u₃ u₄ : Urn),
    operation initial_urn u₁ ∧
    operation u₁ u₂ ∧
    operation u₂ u₃ ∧
    operation u₃ u₄ ∧
    final_urn u₄ ∧
    (probability_of_drawing_white initial_urn *
     probability_of_drawing_white u₁ *
     probability_of_drawing_black u₂ *
     probability_of_drawing_black u₃ +
     probability_of_drawing_white initial_urn *
     probability_of_drawing_black u₁ *
     probability_of_drawing_white u₂ *
     probability_of_drawing_black u₃ +
     probability_of_drawing_white initial_urn *
     probability_of_drawing_black u₁ *
     probability_of_drawing_black u₂ *
     probability_of_drawing_white u₃ +
     probability_of_drawing_black initial_urn *
     probability_of_drawing_white u₁ *
     probability_of_drawing_white u₂ *
     probability_of_drawing_black u₃ +
     probability_of_drawing_black initial_urn *
     probability_of_drawing_white u₁ *
     probability_of_drawing_black u₂ *
     probability_of_drawing_white u₃ +
     probability_of_drawing_black initial_urn *
     probability_of_drawing_black u₁ *
     probability_of_drawing_white u₂ *
     probability_of_drawing_white u₃) = 3/5 := by
  sorry

end urn_probability_l606_60656


namespace waiter_tip_calculation_l606_60600

theorem waiter_tip_calculation (total_customers : ℕ) (non_tipping_customers : ℕ) (total_tip : ℚ) :
  total_customers = 7 →
  non_tipping_customers = 5 →
  total_tip = 6 →
  (total_tip / (total_customers - non_tipping_customers) : ℚ) = 3 :=
by sorry

end waiter_tip_calculation_l606_60600


namespace inequality_and_equality_condition_l606_60640

theorem inequality_and_equality_condition (a b c : ℝ) (h : a * b * c = 1 / 8) :
  a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ≥ 15 / 16 ∧
  (a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 = 15 / 16 ↔ a = 1 / 2 ∧ b = 1 / 2 ∧ c = 1 / 2) :=
by sorry

end inequality_and_equality_condition_l606_60640


namespace quadratic_equation_equivalence_l606_60649

theorem quadratic_equation_equivalence (x : ℝ) :
  let k : ℝ := 0.32653061224489793
  (2 * k * x^2 + 7 * k * x + 2 = 0) ↔ 
  (0.65306122448979586 * x^2 + 2.2857142857142865 * x + 2 = 0) :=
by sorry

end quadratic_equation_equivalence_l606_60649


namespace power_function_through_point_l606_60636

theorem power_function_through_point (a : ℝ) :
  (2 : ℝ) ^ a = Real.sqrt 2 → a = 1 / 2 := by
  sorry

end power_function_through_point_l606_60636


namespace candidate_vote_percentage_l606_60684

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 309400) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) * 100 = 65 := by
  sorry

end candidate_vote_percentage_l606_60684


namespace min_distance_point_to_line_l606_60603

/-- The minimum distance between a point (1,0) and the line x - y + 5 = 0 is 3√2 -/
theorem min_distance_point_to_line : 
  let F : ℝ × ℝ := (1, 0)
  let line (x y : ℝ) : Prop := x - y + 5 = 0
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 ∧ 
    ∀ (P : ℝ × ℝ), line P.1 P.2 → Real.sqrt ((F.1 - P.1)^2 + (F.2 - P.2)^2) ≥ d :=
by sorry

end min_distance_point_to_line_l606_60603


namespace cos_195_plus_i_sin_195_to_60_l606_60662

-- Define DeMoivre's Theorem
axiom deMoivre (θ : ℝ) (n : ℕ) : 
  (Complex.exp (Complex.I * θ)) ^ n = Complex.exp (Complex.I * (n * θ))

-- Define the problem
theorem cos_195_plus_i_sin_195_to_60 :
  (Complex.exp (Complex.I * (195 * π / 180))) ^ 60 = -1 := by sorry

end cos_195_plus_i_sin_195_to_60_l606_60662


namespace sum_of_integers_between_1_and_10_l606_60678

theorem sum_of_integers_between_1_and_10 : 
  (Finset.range 8).sum (fun i => i + 2) = 44 := by
  sorry

end sum_of_integers_between_1_and_10_l606_60678


namespace xxyy_perfect_square_l606_60681

theorem xxyy_perfect_square : 
  ∃! (x y : Nat), x < 10 ∧ y < 10 ∧ 
  (1100 * x + 11 * y = 88 * 88) := by
sorry

end xxyy_perfect_square_l606_60681


namespace factorial_less_than_power_l606_60604

theorem factorial_less_than_power (n : ℕ) (h : n > 1) : 
  Nat.factorial n < ((n + 1) / 2 : ℚ) ^ n := by
  sorry

end factorial_less_than_power_l606_60604


namespace hyperbola_properties_l606_60624

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  17 * x^2 - 16 * x * y + 4 * y^2 - 34 * x + 16 * y + 13 = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (1, 1)
def point_B : ℝ × ℝ := (1, -1)

-- Define the center
def center : ℝ × ℝ := (1, 0)

-- Define the conjugate axis equations
def conjugate_axis_eq (x y : ℝ) : Prop :=
  y = (13 + 5 * Real.sqrt 17) / 16 * (x - 1) ∨
  y = (13 - 5 * Real.sqrt 17) / 16 * (x - 1)

theorem hyperbola_properties :
  (hyperbola_eq point_A.1 point_A.2) ∧
  (hyperbola_eq point_B.1 point_B.2) →
  (∃ (x y : ℝ), hyperbola_eq x y ∧ conjugate_axis_eq x y) ∧
  (center.1 = 1 ∧ center.2 = 0) := by
  sorry

end hyperbola_properties_l606_60624


namespace game_probabilities_and_earnings_l606_60685

/-- Represents the outcome of drawing balls -/
inductive DrawOutcome
  | AllSameColor
  | DifferentColors

/-- Represents the game setup -/
structure GameSetup :=
  (total_balls : Nat)
  (white_balls : Nat)
  (yellow_balls : Nat)
  (same_color_payout : Int)
  (diff_color_payment : Int)
  (draws_per_day : Nat)
  (days_per_month : Nat)

/-- Calculates the probability of drawing 3 white balls -/
def prob_three_white (setup : GameSetup) : Rat :=
  sorry

/-- Calculates the probability of drawing 2 yellow and 1 white ball -/
def prob_two_yellow_one_white (setup : GameSetup) : Rat :=
  sorry

/-- Calculates the expected monthly earnings -/
def expected_monthly_earnings (setup : GameSetup) : Int :=
  sorry

/-- Main theorem stating the probabilities and expected earnings -/
theorem game_probabilities_and_earnings (setup : GameSetup)
  (h1 : setup.total_balls = 6)
  (h2 : setup.white_balls = 3)
  (h3 : setup.yellow_balls = 3)
  (h4 : setup.same_color_payout = -5)
  (h5 : setup.diff_color_payment = 1)
  (h6 : setup.draws_per_day = 100)
  (h7 : setup.days_per_month = 30) :
  prob_three_white setup = 1/20 ∧
  prob_two_yellow_one_white setup = 1/10 ∧
  expected_monthly_earnings setup = 1200 :=
sorry

end game_probabilities_and_earnings_l606_60685


namespace adult_ticket_price_l606_60621

/-- Proves that the price of an adult ticket is $32 given the specified conditions -/
theorem adult_ticket_price
  (num_adults : ℕ)
  (num_children : ℕ)
  (total_amount : ℕ)
  (h_adults : num_adults = 400)
  (h_children : num_children = 200)
  (h_total : total_amount = 16000)
  (h_price_ratio : ∃ (child_price : ℕ), 
    total_amount = num_adults * (2 * child_price) + num_children * child_price) :
  ∃ (adult_price : ℕ), adult_price = 32 ∧
    total_amount = num_adults * adult_price + num_children * (adult_price / 2) :=
by sorry

end adult_ticket_price_l606_60621


namespace james_pizza_slices_l606_60619

theorem james_pizza_slices :
  let total_slices : ℕ := 20
  let tom_slices : ℕ := 5
  let alice_slices : ℕ := 3
  let bob_slices : ℕ := 4
  let friends_slices : ℕ := tom_slices + alice_slices + bob_slices
  let remaining_slices : ℕ := total_slices - friends_slices
  let james_slices : ℕ := remaining_slices / 2
  james_slices = 4 := by sorry

end james_pizza_slices_l606_60619


namespace decreasing_function_positive_l606_60677

/-- A function f is decreasing on ℝ if for all x₁ < x₂, f(x₁) > f(x₂) -/
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

/-- The condition that f'(x) satisfies f(x) / f''(x) < 1 - x -/
def DerivativeCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, DifferentiableAt ℝ f x ∧ DifferentiableAt ℝ (deriv f) x ∧
    f x / (deriv (deriv f) x) < 1 - x

theorem decreasing_function_positive
  (f : ℝ → ℝ)
  (h_decreasing : DecreasingOn f)
  (h_condition : DerivativeCondition f) :
  ∀ x, f x > 0 := by
  sorry

end decreasing_function_positive_l606_60677


namespace octahedron_non_prime_sum_pairs_l606_60646

-- Define the type for die faces
def DieFace := Fin 8

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := sorry

-- Define a function to get the value on a die face
def faceValue (face : DieFace) : ℕ := face.val + 1

-- Define a function to check if the sum of two face values is not prime
def sumNotPrime (face1 face2 : DieFace) : Prop :=
  ¬(isPrime (faceValue face1 + faceValue face2))

-- The main theorem
theorem octahedron_non_prime_sum_pairs :
  ∃ (pairs : Finset (DieFace × DieFace)),
    pairs.card = 8 ∧
    (∀ (pair : DieFace × DieFace), pair ∈ pairs → sumNotPrime pair.1 pair.2) ∧
    (∀ (face1 face2 : DieFace), 
      face1 ≠ face2 → sumNotPrime face1 face2 → 
      (face1, face2) ∈ pairs ∨ (face2, face1) ∈ pairs) :=
sorry

end octahedron_non_prime_sum_pairs_l606_60646


namespace rug_inner_length_is_four_l606_60627

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the rug with three colored regions -/
structure Rug where
  inner : Rectangle
  middle : Rectangle
  outer : Rectangle

/-- Checks if three real numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem rug_inner_length_is_four (rug : Rug) : 
  rug.inner.width = 2 ∧ 
  rug.middle.length = rug.inner.length + 4 ∧ 
  rug.middle.width = rug.inner.width + 4 ∧
  rug.outer.length = rug.middle.length + 4 ∧
  rug.outer.width = rug.middle.width + 4 ∧
  isArithmeticProgression (area rug.inner) (area rug.middle - area rug.inner) (area rug.outer - area rug.middle) →
  rug.inner.length = 4 := by
sorry

end rug_inner_length_is_four_l606_60627


namespace dot_product_OA_OB_line_l_equations_l606_60639

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point M
def M : ℝ × ℝ := (6, 0)

-- Define line l passing through M and intersecting the parabola
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m*y + 6

-- Define points A and B as intersections of line l and the parabola
def intersect_points (m : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Theorem for the dot product of OA and OB
theorem dot_product_OA_OB (m : ℝ) :
  let ((x₁, y₁), (x₂, y₂)) := intersect_points m
  (x₁ * x₂ + y₁ * y₂ : ℝ) = 12 := sorry

-- Theorem for the equations of line l given the area of triangle OAB
theorem line_l_equations :
  (∃ m : ℝ, let ((x₁, y₁), (x₂, y₂)) := intersect_points m
   (1/2 : ℝ) * 6 * |y₁ - y₂| = 12 * Real.sqrt 10) →
  (∃ l₁ l₂ : ℝ → ℝ → Prop,
    (∀ x y, l₁ x y ↔ x + 2*y - 6 = 0) ∧
    (∀ x y, l₂ x y ↔ x - 2*y - 6 = 0) ∧
    (∀ x y, line_l 2 x y ↔ l₁ x y) ∧
    (∀ x y, line_l (-2) x y ↔ l₂ x y)) := sorry

end dot_product_OA_OB_line_l_equations_l606_60639


namespace parabola_equation_l606_60652

/-- Represents a parabola with specific properties -/
structure Parabola where
  -- Equation coefficients
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  -- c is positive
  c_pos : c > 0
  -- GCD of absolute values of coefficients is 1
  gcd_one : Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Nat.gcd (Int.natAbs d) (Nat.gcd (Int.natAbs e) (Int.natAbs f))))) = 1
  -- Passes through (2,6)
  passes_through : a * 2^2 + b * 2 * 6 + c * 6^2 + d * 2 + e * 6 + f = 0
  -- Focus y-coordinate is 2
  focus_y : ∃ (x : ℚ), a * x^2 + b * x * 2 + c * 2^2 + d * x + e * 2 + f = 0
  -- Axis of symmetry parallel to x-axis
  sym_axis_parallel : b = 0
  -- Vertex on y-axis
  vertex_on_y : ∃ (y : ℚ), a * 0^2 + b * 0 * y + c * y^2 + d * 0 + e * y + f = 0

/-- The parabola equation matches the given form -/
theorem parabola_equation (p : Parabola) : p.a = 0 ∧ p.b = 0 ∧ p.c = 1 ∧ p.d = -8 ∧ p.e = -4 ∧ p.f = 4 := by
  sorry

end parabola_equation_l606_60652


namespace play_role_assignment_l606_60663

def number_of_assignments (men women : ℕ) (male_roles female_roles either_roles : ℕ) : ℕ :=
  men * women * (Nat.choose (men + women - male_roles - female_roles) either_roles)

theorem play_role_assignment :
  number_of_assignments 4 7 1 1 4 = 3528 := by
  sorry

end play_role_assignment_l606_60663


namespace banana_arrangements_count_l606_60615

/-- The number of distinct arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct arrangements of the letters in "BANANA" is 60 -/
theorem banana_arrangements_count : banana_arrangements = 60 := by
  sorry

end banana_arrangements_count_l606_60615


namespace probability_not_adjacent_seats_l606_60694

-- Define the number of seats
def num_seats : ℕ := 10

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- Define the number of ways two people can sit next to each other in a row of seats
def adjacent_seats (n : ℕ) : ℕ := n - 1

-- Theorem statement
theorem probability_not_adjacent_seats :
  let total_ways := choose num_seats 2
  let adjacent_ways := adjacent_seats num_seats
  (total_ways - adjacent_ways) / total_ways = 4 / 5 := by sorry

end probability_not_adjacent_seats_l606_60694


namespace givenEquationIsQuadratic_l606_60626

/-- Represents a polynomial equation with one variable -/
structure PolynomialEquation :=
  (a b c : ℝ)

/-- Defines a quadratic equation with one variable -/
def IsQuadraticOneVariable (eq : PolynomialEquation) : Prop :=
  eq.a ≠ 0

/-- The specific equation we're considering -/
def givenEquation : PolynomialEquation :=
  { a := 1, b := 1, c := 3 }

/-- Theorem stating that the given equation is a quadratic equation with one variable -/
theorem givenEquationIsQuadratic : IsQuadraticOneVariable givenEquation := by
  sorry


end givenEquationIsQuadratic_l606_60626


namespace combined_new_wattage_l606_60654

def original_wattages : List ℝ := [60, 80, 100, 120]

def increase_percentage : ℝ := 0.25

def increased_wattage (w : ℝ) : ℝ := w * (1 + increase_percentage)

theorem combined_new_wattage :
  (original_wattages.map increased_wattage).sum = 450 := by
  sorry

end combined_new_wattage_l606_60654


namespace expression_simplification_l606_60657

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 - 3) : 
  (a^2 - 4*a + 4) / (a^2 - 4) / ((a - 2) / (a^2 + 2*a)) + 3 = Real.sqrt 3 := by
  sorry

end expression_simplification_l606_60657


namespace kyunghwan_spent_most_l606_60634

def initial_amount : ℕ := 20000

def seunga_remaining : ℕ := initial_amount / 4
def kyunghwan_remaining : ℕ := initial_amount / 8
def doyun_remaining : ℕ := initial_amount / 5

def seunga_spent : ℕ := initial_amount - seunga_remaining
def kyunghwan_spent : ℕ := initial_amount - kyunghwan_remaining
def doyun_spent : ℕ := initial_amount - doyun_remaining

theorem kyunghwan_spent_most : 
  kyunghwan_spent > seunga_spent ∧ kyunghwan_spent > doyun_spent :=
by sorry

end kyunghwan_spent_most_l606_60634


namespace hiker_catches_cyclist_l606_60637

/-- Proves that a hiker catches up to a cyclist in 30 minutes under specific conditions -/
theorem hiker_catches_cyclist (hiker_speed cyclist_speed : ℝ) (cyclist_travel_time : ℝ) : 
  hiker_speed = 4 →
  cyclist_speed = 24 →
  cyclist_travel_time = 5 / 60 →
  let cyclist_distance := cyclist_speed * cyclist_travel_time
  let catchup_time := cyclist_distance / hiker_speed
  catchup_time * 60 = 30 := by sorry

end hiker_catches_cyclist_l606_60637


namespace at_least_one_multiple_of_three_l606_60610

theorem at_least_one_multiple_of_three (a b : ℤ) : 
  (a + b) % 3 = 0 ∨ (a * b) % 3 = 0 ∨ (a - b) % 3 = 0 := by
sorry

end at_least_one_multiple_of_three_l606_60610


namespace circle_line_intersection_equivalence_l606_60612

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Reflection of a point over a line -/
def reflect_point (p : ℝ × ℝ) (l : Line) : ℝ × ℝ := sorry

/-- Reflection of a circle over a line -/
def reflect_circle (c : Circle) (l : Line) : Circle := sorry

/-- Intersection points of a circle and a line -/
def circle_line_intersection (c : Circle) (l : Line) : Set (ℝ × ℝ) := sorry

/-- Intersection points of two circles -/
def circle_circle_intersection (c1 c2 : Circle) : Set (ℝ × ℝ) := sorry

/-- Main theorem -/
theorem circle_line_intersection_equivalence 
  (k : Circle) (e : Line) (O A B : ℝ × ℝ) :
  O ≠ A ∧ O ≠ B ∧ A ≠ B ∧  -- O is not on line e
  e.point1 = A ∧ e.point2 = B ∧ 
  k.center = O →
  circle_line_intersection k e = circle_circle_intersection k (reflect_circle k e) := by
  sorry

end circle_line_intersection_equivalence_l606_60612


namespace ten_person_handshake_count_l606_60651

/-- Represents a group of people with distinct heights -/
structure HeightGroup where
  n : ℕ
  heights : Fin n → ℕ
  distinct_heights : ∀ i j, i ≠ j → heights i ≠ heights j

/-- The number of handshakes in a height group -/
def handshake_count (group : HeightGroup) : ℕ :=
  (group.n * (group.n - 1)) / 2

/-- Theorem: In a group of 10 people with distinct heights, where each person
    only shakes hands with those taller than themselves, the total number of
    handshakes is 45. -/
theorem ten_person_handshake_count :
  ∀ (group : HeightGroup), group.n = 10 → handshake_count group = 45 := by
  sorry

end ten_person_handshake_count_l606_60651


namespace partial_fraction_decomposition_l606_60635

theorem partial_fraction_decomposition (x : ℝ) (A B C : ℝ) :
  (1 : ℝ) / (x^3 - 7*x^2 + 10*x + 24) = A / (x - 2) + B / (x - 6) + C / (x - 6)^2 →
  x^3 - 7*x^2 + 10*x + 24 = (x - 2) * (x - 6)^2 →
  A = 1/16 := by sorry

end partial_fraction_decomposition_l606_60635


namespace hyperbola_eccentricity_l606_60697

/-- The eccentricity of a hyperbola given its equation and a point it passes through -/
theorem hyperbola_eccentricity (m : ℝ) (h : 2 - 4 / m = 1) : 
  Real.sqrt (1 + 4 / 2) = Real.sqrt 3 := by sorry

end hyperbola_eccentricity_l606_60697
