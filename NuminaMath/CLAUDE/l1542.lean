import Mathlib

namespace NUMINAMATH_CALUDE_loan_payoff_period_l1542_154241

-- Define the costs and monthly payment difference
def house_cost : ℕ := 480000
def trailer_cost : ℕ := 120000
def monthly_payment_diff : ℕ := 1500

-- Define the theorem
theorem loan_payoff_period :
  ∃ (n : ℕ), 
    n * trailer_cost = (n * monthly_payment_diff + trailer_cost) * house_cost ∧
    n = 20 * 12 :=
by sorry

end NUMINAMATH_CALUDE_loan_payoff_period_l1542_154241


namespace NUMINAMATH_CALUDE_at_least_one_shot_hit_l1542_154216

theorem at_least_one_shot_hit (p q : Prop) : 
  (p ∨ q) ↔ ¬(¬p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_at_least_one_shot_hit_l1542_154216


namespace NUMINAMATH_CALUDE_sum_simplification_l1542_154243

theorem sum_simplification :
  (-1)^2002 + (-1)^2003 + 2^2004 - 2^2003 = 2^2003 := by
  sorry

end NUMINAMATH_CALUDE_sum_simplification_l1542_154243


namespace NUMINAMATH_CALUDE_annual_music_cost_l1542_154289

/-- Calculates the annual cost of music for John given his monthly music consumption, average song length, and price per song. -/
theorem annual_music_cost 
  (monthly_hours : ℕ) 
  (song_length_minutes : ℕ) 
  (price_per_song : ℚ) : 
  monthly_hours = 20 → 
  song_length_minutes = 3 → 
  price_per_song = 1/2 → 
  (monthly_hours * 60 / song_length_minutes) * price_per_song * 12 = 2400 := by
sorry

end NUMINAMATH_CALUDE_annual_music_cost_l1542_154289


namespace NUMINAMATH_CALUDE_vector_sum_problem_l1542_154293

theorem vector_sum_problem (x y : ℝ) : 
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (2, y)
  (a.1 + b.1, a.2 + b.2) = (1, -1) → x + y = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_problem_l1542_154293


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1542_154296

def divisors : List Nat := [5, 7, 11, 13, 17, 19]

theorem least_addition_for_divisibility (x : Nat) : 
  (∀ d ∈ divisors, (5432 + x) % d = 0) ∧
  (∀ y < x, ∃ d ∈ divisors, (5432 + y) % d ≠ 0) →
  x = 1611183 := by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1542_154296


namespace NUMINAMATH_CALUDE_trains_crossing_time_l1542_154248

/-- Proves that two trains of equal length traveling in opposite directions will cross each other in 12 seconds -/
theorem trains_crossing_time (length : ℝ) (time1 time2 : ℝ) 
  (h1 : length = 120)
  (h2 : time1 = 10)
  (h3 : time2 = 15) : 
  (2 * length) / (length / time1 + length / time2) = 12 := by
  sorry


end NUMINAMATH_CALUDE_trains_crossing_time_l1542_154248


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1542_154267

theorem fixed_point_on_line (m : ℝ) : 
  (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1542_154267


namespace NUMINAMATH_CALUDE_mexican_olympiad_1988_l1542_154291

theorem mexican_olympiad_1988 (f : ℕ+ → ℕ+) 
  (h : ∀ m n : ℕ+, f (f m + f n) = m + n) : 
  f 1988 = 1988 := by sorry

end NUMINAMATH_CALUDE_mexican_olympiad_1988_l1542_154291


namespace NUMINAMATH_CALUDE_dish_price_theorem_l1542_154228

/-- The original price of a dish that satisfies the given conditions -/
def original_price : ℝ := 40

/-- John's total payment -/
def john_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * price

/-- Jane's total payment -/
def jane_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * (0.9 * price)

/-- Theorem stating that the original price satisfies the given conditions -/
theorem dish_price_theorem : 
  john_payment original_price = jane_payment original_price + 0.60 := by
  sorry

#eval original_price

end NUMINAMATH_CALUDE_dish_price_theorem_l1542_154228


namespace NUMINAMATH_CALUDE_greatest_number_l1542_154229

def octal_to_decimal (n : ℕ) : ℕ := 3 * 8^1 + 2 * 8^0

def base5_to_decimal (n : ℕ) : ℕ := 1 * 5^2 + 1 * 5^1 + 1 * 5^0

def binary_to_decimal (n : ℕ) : ℕ := 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0

def base6_to_decimal (n : ℕ) : ℕ := 5 * 6^1 + 4 * 6^0

theorem greatest_number : 
  binary_to_decimal 101010 > octal_to_decimal 32 ∧
  binary_to_decimal 101010 > base5_to_decimal 111 ∧
  binary_to_decimal 101010 > base6_to_decimal 54 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_l1542_154229


namespace NUMINAMATH_CALUDE_monotonic_increasing_implies_a_eq_neg_six_l1542_154283

/-- The function f(x) defined as the absolute value of 2x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + a|

/-- The property of f being monotonically increasing on [3, +∞) -/
def monotonic_increasing_from_three (a : ℝ) : Prop :=
  ∀ x y, 3 ≤ x → x ≤ y → f a x ≤ f a y

/-- Theorem stating that a must be -6 for f to be monotonically increasing on [3, +∞) -/
theorem monotonic_increasing_implies_a_eq_neg_six :
  ∃ a, monotonic_increasing_from_three a ↔ a = -6 :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_implies_a_eq_neg_six_l1542_154283


namespace NUMINAMATH_CALUDE_local_max_value_is_four_l1542_154252

/-- The function f(x) = x^3 - 3ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + 2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem local_max_value_is_four (a : ℝ) :
  (∃ x : ℝ, IsLocalMin (f a) 1) →
  (∃ x : ℝ, IsLocalMax (f a) x ∧ f a x = 4) :=
by sorry

end NUMINAMATH_CALUDE_local_max_value_is_four_l1542_154252


namespace NUMINAMATH_CALUDE_min_trig_fraction_l1542_154221

theorem min_trig_fraction :
  (∀ x : ℝ, (Real.sin x)^6 + (Real.cos x)^6 + 1 ≥ 5/6 * ((Real.sin x)^4 + (Real.cos x)^4 + 1)) ∧
  (∃ x : ℝ, (Real.sin x)^6 + (Real.cos x)^6 + 1 = 5/6 * ((Real.sin x)^4 + (Real.cos x)^4 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_min_trig_fraction_l1542_154221


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1542_154260

theorem right_triangle_segment_ratio (x y z u v : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
    (h4 : x^2 + y^2 = z^2) (h5 : x * z = u * (u + v)) (h6 : y * z = v * (u + v)) (h7 : 3 * y = 4 * x) :
    9 * v = 16 * u := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1542_154260


namespace NUMINAMATH_CALUDE_interest_equality_second_sum_l1542_154233

/-- Given a total sum divided into two parts, where the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years
    at 5% per annum, prove that the second part is equal to 1680 rupees. -/
theorem interest_equality_second_sum (total : ℚ) (first_part : ℚ) (second_part : ℚ) :
  total = 2730 →
  total = first_part + second_part →
  (first_part * 3 * 8) / 100 = (second_part * 5 * 3) / 100 →
  second_part = 1680 := by
  sorry

end NUMINAMATH_CALUDE_interest_equality_second_sum_l1542_154233


namespace NUMINAMATH_CALUDE_bridge_concrete_total_l1542_154222

/-- The amount of concrete needed for a bridge -/
structure BridgeConcrete where
  roadway_deck : ℕ
  single_anchor : ℕ
  num_anchors : ℕ
  supporting_pillars : ℕ

/-- The total amount of concrete needed for the bridge -/
def total_concrete (b : BridgeConcrete) : ℕ :=
  b.roadway_deck + b.single_anchor * b.num_anchors + b.supporting_pillars

/-- Theorem: The total amount of concrete needed for the bridge is 4800 tons -/
theorem bridge_concrete_total :
  let b : BridgeConcrete := {
    roadway_deck := 1600,
    single_anchor := 700,
    num_anchors := 2,
    supporting_pillars := 1800
  }
  total_concrete b = 4800 := by sorry

end NUMINAMATH_CALUDE_bridge_concrete_total_l1542_154222


namespace NUMINAMATH_CALUDE_binary_sum_equals_decimal_l1542_154239

/-- Converts a binary number represented as a sum of powers of 2 to its decimal equivalent -/
def binary_to_decimal (powers : List Nat) : Nat :=
  powers.foldl (fun acc p => acc + 2^p) 0

theorem binary_sum_equals_decimal : 
  let a := binary_to_decimal [0, 1, 2, 3, 4, 5, 6, 7, 8]  -- 111111111₂
  let b := binary_to_decimal [2, 3, 4, 5]                 -- 110110₂
  a + b = 571 := by sorry

end NUMINAMATH_CALUDE_binary_sum_equals_decimal_l1542_154239


namespace NUMINAMATH_CALUDE_min_gold_chips_l1542_154269

/-- Represents a box of chips with gold, silver, and bronze chips. -/
structure ChipBox where
  gold : ℕ
  silver : ℕ
  bronze : ℕ

/-- Checks if a ChipBox satisfies the given conditions. -/
def isValidChipBox (box : ChipBox) : Prop :=
  box.bronze ≥ 2 * box.silver ∧
  box.bronze ≤ box.gold / 4 ∧
  box.silver + box.bronze ≥ 75

/-- Theorem stating the minimum number of gold chips in a valid ChipBox. -/
theorem min_gold_chips (box : ChipBox) :
  isValidChipBox box → box.gold ≥ 200 := by
  sorry

#check min_gold_chips

end NUMINAMATH_CALUDE_min_gold_chips_l1542_154269


namespace NUMINAMATH_CALUDE_justice_plants_l1542_154219

theorem justice_plants (ferns palms succulents total_wanted : ℕ) : 
  ferns = 3 → palms = 5 → succulents = 7 → total_wanted = 24 →
  total_wanted - (ferns + palms + succulents) = 9 := by
sorry

end NUMINAMATH_CALUDE_justice_plants_l1542_154219


namespace NUMINAMATH_CALUDE_partnership_profit_theorem_l1542_154232

/-- Calculates the total profit for a partnership given investments and one partner's profit share -/
def calculate_total_profit (tom_investment : ℕ) (jose_investment : ℕ) (tom_months : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : ℕ :=
  let tom_total := tom_investment * tom_months
  let jose_total := jose_investment * jose_months
  let ratio_sum := (tom_total / (tom_total.gcd jose_total)) + (jose_total / (tom_total.gcd jose_total))
  (ratio_sum * jose_profit) / (jose_total / (tom_total.gcd jose_total))

theorem partnership_profit_theorem (tom_investment jose_investment tom_months jose_months jose_profit : ℕ) 
  (h1 : tom_investment = 30000)
  (h2 : jose_investment = 45000)
  (h3 : tom_months = 12)
  (h4 : jose_months = 10)
  (h5 : jose_profit = 40000) :
  calculate_total_profit tom_investment jose_investment tom_months jose_months jose_profit = 72000 := by
  sorry

#eval calculate_total_profit 30000 45000 12 10 40000

end NUMINAMATH_CALUDE_partnership_profit_theorem_l1542_154232


namespace NUMINAMATH_CALUDE_third_pair_weight_l1542_154276

def dumbbell_system (weight1 weight2 weight3 : ℕ) : Prop :=
  weight1 * 2 + weight2 * 2 + weight3 * 2 = 32

theorem third_pair_weight :
  ∃ (weight3 : ℕ), dumbbell_system 3 5 weight3 ∧ weight3 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_third_pair_weight_l1542_154276


namespace NUMINAMATH_CALUDE_fruit_shop_problem_l1542_154278

/-- Fruit shop problem -/
theorem fruit_shop_problem 
  (may_total : ℝ) 
  (may_cost_A may_cost_B : ℝ)
  (june_cost_A june_cost_B : ℝ)
  (june_increase : ℝ)
  (june_total_quantity : ℝ)
  (h_may_total : may_total = 1700)
  (h_may_cost_A : may_cost_A = 8)
  (h_may_cost_B : may_cost_B = 18)
  (h_june_cost_A : june_cost_A = 10)
  (h_june_cost_B : june_cost_B = 20)
  (h_june_increase : june_increase = 300)
  (h_june_total_quantity : june_total_quantity = 120) :
  ∃ (may_quantity_A may_quantity_B : ℝ),
    may_quantity_A * may_cost_A + may_quantity_B * may_cost_B = may_total ∧
    may_quantity_A * june_cost_A + may_quantity_B * june_cost_B = may_total + june_increase ∧
    may_quantity_A = 100 ∧
    may_quantity_B = 50 ∧
    (∃ (june_quantity_A : ℝ),
      june_quantity_A ≤ 3 * (june_total_quantity - june_quantity_A) ∧
      june_quantity_A * june_cost_A + (june_total_quantity - june_quantity_A) * june_cost_B = 1500 ∧
      ∀ (other_june_quantity_A : ℝ),
        other_june_quantity_A ≤ 3 * (june_total_quantity - other_june_quantity_A) →
        other_june_quantity_A * june_cost_A + (june_total_quantity - other_june_quantity_A) * june_cost_B ≥ 1500) :=
by sorry

end NUMINAMATH_CALUDE_fruit_shop_problem_l1542_154278


namespace NUMINAMATH_CALUDE_odd_function_2019_l1542_154223

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_2019 (f : ℝ → ℝ) 
  (h_odd : IsOdd f)
  (h_sym : ∀ x, f (1 + x) = f (1 - x))
  (h_f1 : f 1 = 9) :
  f 2019 = -9 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_2019_l1542_154223


namespace NUMINAMATH_CALUDE_smallest_max_sum_l1542_154264

theorem smallest_max_sum (a b c d e : ℕ+) 
  (sum_eq : a + b + c + d + e = 3060)
  (ae_lower_bound : a + e ≥ 1300) :
  let M := max (a + b) (max (b + c) (max (c + d) (d + e)))
  ∀ (a' b' c' d' e' : ℕ+),
    a' + b' + c' + d' + e' = 3060 →
    a' + e' ≥ 1300 →
    max (a' + b') (max (b' + c') (max (c' + d') (d' + e'))) ≥ 1174 :=
by sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l1542_154264


namespace NUMINAMATH_CALUDE_sqrt_six_over_sqrt_two_equals_sqrt_three_l1542_154213

theorem sqrt_six_over_sqrt_two_equals_sqrt_three :
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_over_sqrt_two_equals_sqrt_three_l1542_154213


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1542_154206

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 - 3*i) / (2 + 5*i) = (-13 : ℝ) / 29 - (11 : ℝ) / 29 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1542_154206


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l1542_154270

theorem longest_side_of_triangle (x : ℝ) : 
  7 + (x + 4) + (2 * x + 1) = 36 → 
  max 7 (max (x + 4) (2 * x + 1)) = 17 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l1542_154270


namespace NUMINAMATH_CALUDE_smartphone_cost_smartphone_cost_proof_l1542_154246

theorem smartphone_cost (initial_savings : ℕ) (saving_months : ℕ) (weeks_per_month : ℕ) (weekly_savings : ℕ) : ℕ :=
  let total_weeks := saving_months * weeks_per_month
  let total_savings := weekly_savings * total_weeks
  initial_savings + total_savings

#check smartphone_cost 40 2 4 15 = 160

theorem smartphone_cost_proof :
  smartphone_cost 40 2 4 15 = 160 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_cost_smartphone_cost_proof_l1542_154246


namespace NUMINAMATH_CALUDE_min_distance_after_11_hours_l1542_154218

/-- Represents the turtle's movement on a 2D plane -/
structure TurtleMovement where
  speed : ℝ
  duration : ℕ

/-- Calculates the minimum possible distance from the starting point -/
def minDistanceFromStart (movement : TurtleMovement) : ℝ :=
  sorry

/-- Theorem stating the minimum distance for the given conditions -/
theorem min_distance_after_11_hours :
  let movement : TurtleMovement := ⟨5, 11⟩
  minDistanceFromStart movement = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_after_11_hours_l1542_154218


namespace NUMINAMATH_CALUDE_rotation_of_point_transformed_curve_equation_l1542_154266

def rotation_pi_over_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), p.1)

def transformation_T2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.2)

def compose_transformations (f g : ℝ × ℝ → ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  f (g p)

def parabola (x : ℝ) : ℝ := x^2

theorem rotation_of_point :
  rotation_pi_over_2 (2, 1) = (-1, 2) := by sorry

theorem transformed_curve_equation (x y : ℝ) :
  (∃ t : ℝ, compose_transformations transformation_T2 rotation_pi_over_2 (t, parabola t) = (x, y)) ↔
  y - x = y^2 := by sorry

end NUMINAMATH_CALUDE_rotation_of_point_transformed_curve_equation_l1542_154266


namespace NUMINAMATH_CALUDE_circle_radius_l1542_154249

/-- The circle C is defined by the equation x^2 + y^2 - 4x - 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The radius of a circle is the distance from its center to any point on the circle -/
def is_radius (r : ℝ) (center : ℝ × ℝ) (equation : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, equation x y → (x - center.1)^2 + (y - center.2)^2 = r^2

/-- The radius of the circle C defined by x^2 + y^2 - 4x - 2y + 1 = 0 is equal to 2 -/
theorem circle_radius : ∃ center, is_radius 2 center circle_equation := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1542_154249


namespace NUMINAMATH_CALUDE_lower_right_is_two_l1542_154294

/-- Represents a 5x5 grid of integers -/
def Grid := Fin 5 → Fin 5 → Nat

/-- Check if all numbers in a list are distinct -/
def allDistinct (l : List Nat) : Prop := l.Nodup

/-- Check if a grid satisfies the row constraint -/
def validRows (g : Grid) : Prop :=
  ∀ i, allDistinct [g i 0, g i 1, g i 2, g i 3, g i 4]

/-- Check if a grid satisfies the column constraint -/
def validColumns (g : Grid) : Prop :=
  ∀ j, allDistinct [g 0 j, g 1 j, g 2 j, g 3 j, g 4 j]

/-- Check if all numbers in the grid are between 1 and 5 -/
def validNumbers (g : Grid) : Prop :=
  ∀ i j, 1 ≤ g i j ∧ g i j ≤ 5

/-- Check if the sum of the first row is 15 -/
def firstRowSum15 (g : Grid) : Prop :=
  g 0 0 + g 0 1 + g 0 2 + g 0 3 + g 0 4 = 15

/-- Check if the given numbers in the grid match the problem description -/
def matchesGivenNumbers (g : Grid) : Prop :=
  g 0 0 = 1 ∧ g 0 2 = 3 ∧ g 0 3 = 4 ∧
  g 1 0 = 5 ∧ g 1 2 = 1 ∧ g 1 4 = 3 ∧
  g 2 1 = 4 ∧ g 2 3 = 5 ∧
  g 3 0 = 4

theorem lower_right_is_two (g : Grid) 
  (hrows : validRows g)
  (hcols : validColumns g)
  (hnums : validNumbers g)
  (hsum : firstRowSum15 g)
  (hgiven : matchesGivenNumbers g) :
  g 4 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lower_right_is_two_l1542_154294


namespace NUMINAMATH_CALUDE_arithmetic_progression_reciprocals_implies_squares_l1542_154217

theorem arithmetic_progression_reciprocals_implies_squares
  (a b c : ℝ)
  (h : ∃ (k : ℝ), (1 / (a + c)) - (1 / (a + b)) = k ∧ (1 / (b + c)) - (1 / (a + c)) = k) :
  ∃ (r : ℝ), b^2 - a^2 = r ∧ c^2 - b^2 = r :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_reciprocals_implies_squares_l1542_154217


namespace NUMINAMATH_CALUDE_democrats_ratio_l1542_154257

/-- Proves that the ratio of democrats to total participants is 1:3 -/
theorem democrats_ratio (total : ℕ) (female_democrats : ℕ) :
  total = 810 →
  female_democrats = 135 →
  let female := 2 * female_democrats
  let male := total - female
  let male_democrats := male / 4
  let total_democrats := female_democrats + male_democrats
  (total_democrats : ℚ) / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_democrats_ratio_l1542_154257


namespace NUMINAMATH_CALUDE_intersection_seq_100th_term_l1542_154210

def geometric_seq (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

def arithmetic_seq (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

def intersection_seq (n : ℕ) : ℝ := 2^(4 * n - 3)

theorem intersection_seq_100th_term :
  intersection_seq 100 = 2^397 :=
by sorry

end NUMINAMATH_CALUDE_intersection_seq_100th_term_l1542_154210


namespace NUMINAMATH_CALUDE_family_meeting_impossible_l1542_154207

theorem family_meeting_impossible (n : ℕ) (h : n = 9) :
  ¬ ∃ (handshakes : ℕ), 2 * handshakes = n * 3 :=
by
  sorry

end NUMINAMATH_CALUDE_family_meeting_impossible_l1542_154207


namespace NUMINAMATH_CALUDE_shooting_outcomes_l1542_154251

/-- Represents the number of shots -/
def num_shots : ℕ := 6

/-- Represents the number of hits we're interested in -/
def num_hits : ℕ := 3

/-- Calculates the total number of possible outcomes for n shots -/
def total_outcomes (n : ℕ) : ℕ := 2^n

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of outcomes with exactly k hits out of n shots -/
def outcomes_with_k_hits (n k : ℕ) : ℕ := choose n k

/-- Calculates the number of outcomes with exactly k hits and exactly 2 consecutive hits out of n shots -/
def outcomes_with_k_hits_and_2_consecutive (n k : ℕ) : ℕ := choose (n - k + 1) 2

theorem shooting_outcomes :
  (total_outcomes num_shots = 64) ∧
  (outcomes_with_k_hits num_shots num_hits = 20) ∧
  (outcomes_with_k_hits_and_2_consecutive num_shots num_hits = 6) := by
  sorry

end NUMINAMATH_CALUDE_shooting_outcomes_l1542_154251


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1542_154259

theorem inequality_solution_set (x : ℝ) : 
  (x - 5) / (x + 1) ≤ 0 ∧ x + 1 ≠ 0 ↔ x ∈ Set.Ioc (-1) 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1542_154259


namespace NUMINAMATH_CALUDE_functional_equation_zero_value_l1542_154274

theorem functional_equation_zero_value 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * y) = f x + f y) : 
  f 0 = 0 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_zero_value_l1542_154274


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1542_154268

theorem regular_polygon_sides (n₁ n₂ : ℕ) : 
  n₁ % 2 = 0 → 
  n₂ % 2 = 0 → 
  (n₁ - 2) * 180 + (n₂ - 2) * 180 = 1800 → 
  ((n₁ = 4 ∧ n₂ = 10) ∨ (n₁ = 10 ∧ n₂ = 4) ∨ (n₁ = 6 ∧ n₂ = 8) ∨ (n₁ = 8 ∧ n₂ = 6)) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1542_154268


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1542_154255

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c - k }

theorem parabola_shift_theorem (x y : ℝ) :
  let original := Parabola.mk 3 0 0
  let shifted := shift_parabola original 1 2
  y = 3 * x^2 → y = 3 * (x - 1)^2 - 2 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1542_154255


namespace NUMINAMATH_CALUDE_middle_term_is_plus_minus_six_l1542_154288

/-- The coefficient of the middle term in the expansion of (a ± 3b)² -/
def middle_term_coefficient (a b : ℝ) : Set ℝ :=
  {x : ℝ | ∃ (sign : ℝ) (h : sign = 1 ∨ sign = -1), 
    (a + sign * 3 * b)^2 = a^2 + x * a * b + 9 * b^2}

/-- Theorem stating that the coefficient of the middle term is either 6 or -6 -/
theorem middle_term_is_plus_minus_six (a b : ℝ) : 
  middle_term_coefficient a b = {6, -6} := by
sorry

end NUMINAMATH_CALUDE_middle_term_is_plus_minus_six_l1542_154288


namespace NUMINAMATH_CALUDE_a_1_greater_than_one_l1542_154280

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

theorem a_1_greater_than_one (seq : ArithmeticSequence)
  (sum_condition : seq.a 1 + seq.a 2 + seq.a 5 > 13)
  (geometric_condition : seq.a 2 ^ 2 = seq.a 1 * seq.a 5) :
  seq.a 1 > 1 := by
  sorry

end NUMINAMATH_CALUDE_a_1_greater_than_one_l1542_154280


namespace NUMINAMATH_CALUDE_sin_780_degrees_l1542_154258

theorem sin_780_degrees : Real.sin (780 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_780_degrees_l1542_154258


namespace NUMINAMATH_CALUDE_profit_maximum_l1542_154254

/-- The profit function for a product with selling price m -/
def profit (m : ℝ) : ℝ := (m - 8) * (900 - 15 * m)

/-- The expression claimed to represent the maximum profit -/
def maxProfitExpr (m : ℝ) : ℝ := -15 * (m - 34)^2 + 10140

theorem profit_maximum :
  ∃ m₀ : ℝ, 
    (∀ m : ℝ, profit m ≤ profit m₀) ∧ 
    (∀ m : ℝ, maxProfitExpr m = profit m) ∧
    (maxProfitExpr m₀ = profit m₀) :=
sorry

end NUMINAMATH_CALUDE_profit_maximum_l1542_154254


namespace NUMINAMATH_CALUDE_line_translation_theorem_l1542_154253

/-- Represents a line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Translates a line vertically and horizontally -/
def translateLine (l : Line) (vertical : ℝ) (horizontal : ℝ) : Line :=
  { slope := l.slope,
    yIntercept := l.yIntercept - vertical - l.slope * horizontal }

theorem line_translation_theorem :
  let initialLine : Line := { slope := 2, yIntercept := 1 }
  let translatedLine := translateLine initialLine 3 2
  translatedLine = { slope := 2, yIntercept := -6 } := by sorry

end NUMINAMATH_CALUDE_line_translation_theorem_l1542_154253


namespace NUMINAMATH_CALUDE_reciprocal_sum_problem_l1542_154277

theorem reciprocal_sum_problem (x y z : ℝ) 
  (h1 : 1/x + 1/y + 1/z = 2) 
  (h2 : 1/x^2 + 1/y^2 + 1/z^2 = 1) : 
  1/(x*y) + 1/(y*z) + 1/(z*x) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_problem_l1542_154277


namespace NUMINAMATH_CALUDE_cos_double_angle_for_tan_two_l1542_154279

theorem cos_double_angle_for_tan_two (θ : Real) (h : Real.tan θ = 2) : 
  Real.cos (2 * θ) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_for_tan_two_l1542_154279


namespace NUMINAMATH_CALUDE_all_but_one_are_sum_of_two_primes_l1542_154261

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p + q = n

theorem all_but_one_are_sum_of_two_primes :
  ∀ k : ℕ, k > 0 → is_sum_of_two_primes (1 + 10 * k) :=
by sorry

end NUMINAMATH_CALUDE_all_but_one_are_sum_of_two_primes_l1542_154261


namespace NUMINAMATH_CALUDE_cube_surface_area_ratio_l1542_154262

theorem cube_surface_area_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (ratio : a = 7 * b) :
  (6 * a^2) / (6 * b^2) = 49 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_ratio_l1542_154262


namespace NUMINAMATH_CALUDE_binary_subtraction_example_l1542_154292

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0 -/
def BinaryNum := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : ℕ) : BinaryNum :=
  sorry

/-- Converts a binary number to its natural number representation -/
def fromBinary (b : BinaryNum) : ℕ :=
  sorry

/-- Performs binary subtraction -/
def binarySubtract (a b : BinaryNum) : BinaryNum :=
  sorry

theorem binary_subtraction_example :
  binarySubtract (toBinary 27) (toBinary 5) = toBinary 22 :=
sorry

end NUMINAMATH_CALUDE_binary_subtraction_example_l1542_154292


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1542_154295

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 → 
  (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ) → 
  (x : ℕ) + (y : ℕ) = 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1542_154295


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l1542_154299

theorem sum_of_roots_cubic (x₁ x₂ x₃ k m : ℝ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)
  (h_root₁ : 2 * x₁^3 - k * x₁ = m)
  (h_root₂ : 2 * x₂^3 - k * x₂ = m)
  (h_root₃ : 2 * x₃^3 - k * x₃ = m) :
  x₁ + x₂ + x₃ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l1542_154299


namespace NUMINAMATH_CALUDE_prime_sum_fraction_l1542_154236

theorem prime_sum_fraction (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r)
  (hdistinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (ha : ∃ (a : ℕ), a = (p + q) / r + (q + r) / p + (r + p) / q) :
  ∃ (a : ℕ), a = 7 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_fraction_l1542_154236


namespace NUMINAMATH_CALUDE_increasing_continuous_function_intermediate_values_l1542_154226

theorem increasing_continuous_function_intermediate_values 
  (f : ℝ → ℝ) (M N : ℝ) :
  (∀ x y, x ∈ Set.Icc 0 2 → y ∈ Set.Icc 0 2 → x < y → f x < f y) →
  ContinuousOn f (Set.Icc 0 2) →
  f 0 = M →
  f 2 = N →
  M > 0 →
  N > 0 →
  (∃ x₁ ∈ Set.Icc 0 2, f x₁ = (M + N) / 2) ∧
  (∃ x₂ ∈ Set.Icc 0 2, f x₂ = Real.sqrt (M * N)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_continuous_function_intermediate_values_l1542_154226


namespace NUMINAMATH_CALUDE_x_plus_y_value_l1542_154200

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 3000)
  (eq2 : x + 3000 * Real.sin y = 2999)
  (y_range : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2999 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l1542_154200


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_parallelism_l1542_154265

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity_parallelism
  (m n : Line) (α β : Plane)
  (h_different_lines : m ≠ n)
  (h_different_planes : α ≠ β)
  (h_m_perp_α : perpendicular m α)
  (h_n_parallel_β : parallel_line_plane n β)
  (h_α_parallel_β : parallel_plane α β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_parallelism_l1542_154265


namespace NUMINAMATH_CALUDE_unique_function_property_l1542_154250

theorem unique_function_property (f : ℕ → ℕ) :
  (f 1 > 0) ∧
  (∀ m n : ℕ, f (m^2 + n^2) = (f m)^2 + (f n)^2) →
  (∀ n : ℕ, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_property_l1542_154250


namespace NUMINAMATH_CALUDE_median_is_2040201_l1542_154208

/-- The list of numbers containing integers from 1 to 2020, their squares, and their cubes -/
def numberList : List ℕ := 
  (List.range 2020).map (λ x => x + 1) ++
  (List.range 2020).map (λ x => (x + 1)^2) ++
  (List.range 2020).map (λ x => (x + 1)^3)

/-- The length of the number list -/
def listLength : ℕ := 6060

/-- The position of the lower median element -/
def lowerMedianPos : ℕ := listLength / 2

/-- The position of the upper median element -/
def upperMedianPos : ℕ := lowerMedianPos + 1

/-- The lower median element -/
def lowerMedian : ℕ := 2020^2

/-- The upper median element -/
def upperMedian : ℕ := 1^3

/-- The median of the number list -/
def median : ℕ := (lowerMedian + upperMedian) / 2

/-- Theorem stating that the median of the number list is 2040201 -/
theorem median_is_2040201 : median = 2040201 := by
  sorry

end NUMINAMATH_CALUDE_median_is_2040201_l1542_154208


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1542_154234

def M : Set ℝ := {x | |x| = 1}
def N : Set ℝ := {x | x^2 ≠ x}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1542_154234


namespace NUMINAMATH_CALUDE_floor_ceil_calculation_l1542_154224

theorem floor_ceil_calculation : 
  ⌊(18 : ℝ) / 5 * (-33 : ℝ) / 4⌋ - ⌈(18 : ℝ) / 5 * ⌈(-33 : ℝ) / 4⌉⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_calculation_l1542_154224


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1542_154287

/-- Given a quadratic function y = -3x^2 + 6x + 4, prove that its maximum value is 7 -/
theorem quadratic_max_value :
  let f : ℝ → ℝ := λ x ↦ -3 * x^2 + 6 * x + 4
  ∃ x_max : ℝ, ∀ x : ℝ, f x ≤ f x_max ∧ f x_max = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1542_154287


namespace NUMINAMATH_CALUDE_pears_picked_total_l1542_154245

/-- The number of pears Mike picked -/
def mike_pears : ℕ := 8

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 7

/-- The total number of pears picked -/
def total_pears : ℕ := mike_pears + jason_pears

theorem pears_picked_total : total_pears = 15 := by
  sorry

end NUMINAMATH_CALUDE_pears_picked_total_l1542_154245


namespace NUMINAMATH_CALUDE_tan_3_negative_l1542_154240

theorem tan_3_negative : Real.tan 3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_3_negative_l1542_154240


namespace NUMINAMATH_CALUDE_wage_increase_hours_decrease_l1542_154263

theorem wage_increase_hours_decrease (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  let new_wage := 1.5 * w
  let new_hours := h / 1.5
  let percent_decrease := 100 * (1 - 1 / 1.5)
  new_wage * new_hours = w * h ∧ 
  100 * (h - new_hours) / h = percent_decrease := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_hours_decrease_l1542_154263


namespace NUMINAMATH_CALUDE_sin_690_degrees_l1542_154230

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l1542_154230


namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l1542_154273

-- Problem 1
theorem calculation_proof : 
  Real.sqrt 4 + 2 * Real.sin (45 * π / 180) - (π - 3)^0 + |Real.sqrt 2 - 2| = 3 := by sorry

-- Problem 2
theorem inequality_system_solution (x : ℝ) : 
  (2 * (x + 2) - x ≤ 5 ∧ (4 * x + 1) / 3 > x - 1) ↔ (-4 < x ∧ x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l1542_154273


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1542_154235

-- Define the quadratic function
def quadratic (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

-- Define the transformed quadratic function
def transformed_quadratic (m h k : ℝ) (x : ℝ) : ℝ := m * (x - h)^2 + k

-- State the theorem
theorem quadratic_transformation (p q r : ℝ) :
  (∃ m k : ℝ, ∀ x : ℝ, quadratic p q r x = transformed_quadratic 5 3 15 x) →
  (∃ m k : ℝ, ∀ x : ℝ, quadratic (4*p) (4*q) (4*r) x = transformed_quadratic m 3 k x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1542_154235


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l1542_154285

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ
  side : ℝ

/-- The theorem stating the relationship between the trapezoid's properties -/
theorem isosceles_trapezoid_side_length 
  (t : IsoscelesTrapezoid) 
  (h1 : t.base1 = 9) 
  (h2 : t.base2 = 15) 
  (h3 : t.area = 48) : 
  t.side = 5 := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l1542_154285


namespace NUMINAMATH_CALUDE_unique_solution_prime_power_equation_l1542_154212

theorem unique_solution_prime_power_equation :
  ∀ (p q : ℕ) (n m : ℕ),
    Prime p → Prime q → n ≥ 2 → m ≥ 2 →
    (p^n = q^m + 1 ∨ p^n = q^m - 1) →
    (p = 2 ∧ n = 3 ∧ q = 3 ∧ m = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_prime_power_equation_l1542_154212


namespace NUMINAMATH_CALUDE_total_students_l1542_154247

theorem total_students (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 21) 
  (h2 : rank_from_left = 11) : 
  rank_from_right + rank_from_left - 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l1542_154247


namespace NUMINAMATH_CALUDE_equation_solution_l1542_154209

theorem equation_solution (x : Real) :
  8.414 * Real.cos x + Real.sqrt (3/2 - Real.cos x ^ 2) - Real.cos x * Real.sqrt (3/2 - Real.cos x ^ 2) = 1 ↔
  (∃ k : ℤ, x = 2 * Real.pi * ↑k) ∨
  (∃ k : ℤ, x = Real.pi / 4 + 2 * Real.pi * ↑k) ∨
  (∃ k : ℤ, x = 3 * Real.pi / 4 + 2 * Real.pi * ↑k) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1542_154209


namespace NUMINAMATH_CALUDE_tinas_earnings_l1542_154203

/-- Calculates the total earnings for a worker given their hourly rate, hours worked per day, 
    number of days worked, and regular hours per day before overtime. -/
def calculate_earnings (hourly_rate : ℚ) (hours_per_day : ℕ) (days_worked : ℕ) (regular_hours : ℕ) : ℚ :=
  let regular_pay := hourly_rate * regular_hours * days_worked
  let overtime_hours := if hours_per_day > regular_hours then hours_per_day - regular_hours else 0
  let overtime_rate := hourly_rate * (1 + 1/2)
  let overtime_pay := overtime_rate * overtime_hours * days_worked
  regular_pay + overtime_pay

/-- Theorem stating that Tina's earnings for 5 days of work at 10 hours per day 
    with an $18.00 hourly rate is $990.00. -/
theorem tinas_earnings : 
  calculate_earnings 18 10 5 8 = 990 := by
  sorry

end NUMINAMATH_CALUDE_tinas_earnings_l1542_154203


namespace NUMINAMATH_CALUDE_animal_sanctuary_l1542_154238

theorem animal_sanctuary (total : ℕ) (difference : ℕ) : total = 450 ∧ difference = 75 → ∃ (dogs cats : ℕ), cats = dogs + difference ∧ dogs + cats = total ∧ cats = 262 := by
  sorry

end NUMINAMATH_CALUDE_animal_sanctuary_l1542_154238


namespace NUMINAMATH_CALUDE_team_combinations_theorem_l1542_154284

/-- The number of ways to select k elements from n elements --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid team combinations --/
def validCombinations (totalMale totalFemale teamSize : ℕ) : ℕ :=
  (choose totalMale 1 * choose totalFemale 2) +
  (choose totalMale 2 * choose totalFemale 1)

theorem team_combinations_theorem :
  validCombinations 5 4 3 = 70 := by sorry

end NUMINAMATH_CALUDE_team_combinations_theorem_l1542_154284


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1542_154244

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |2*x - 5|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 2 x ≥ 5} = {x : ℝ | x ≤ 2 ∨ x ≥ 8/3} := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | a > 0 ∧ ∀ x ∈ Set.Icc a (2*a - 2), f a x ≤ |x + 4|} = Set.Ioo 2 (13/5) := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1542_154244


namespace NUMINAMATH_CALUDE_local_value_of_four_l1542_154211

/-- The local value of a digit in a number. -/
def local_value (digit : ℕ) (place : ℕ) : ℕ := digit * (10 ^ place)

/-- The sum of local values of all digits in 2345. -/
def total_sum : ℕ := 2345

/-- The local values of digits 2, 3, and 5 in 2345. -/
def known_values : ℕ := local_value 2 3 + local_value 3 2 + local_value 5 0

/-- The local value of the remaining digit (4) in 2345. -/
def remaining_value : ℕ := total_sum - known_values

theorem local_value_of_four :
  remaining_value = local_value 4 1 :=
sorry

end NUMINAMATH_CALUDE_local_value_of_four_l1542_154211


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1542_154275

theorem right_triangle_side_length (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) (h5 : c = 17) (h6 : a = 15) : b = 8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1542_154275


namespace NUMINAMATH_CALUDE_vanessa_score_in_game_l1542_154214

/-- Calculates Vanessa's score in a basketball game -/
def vanessaScore (totalScore : ℕ) (otherPlayersCount : ℕ) (otherPlayersAverage : ℕ) : ℕ :=
  totalScore - (otherPlayersCount * otherPlayersAverage)

/-- Theorem stating Vanessa's score given the game conditions -/
theorem vanessa_score_in_game : 
  vanessaScore 60 7 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_score_in_game_l1542_154214


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1542_154227

theorem polynomial_factorization :
  ∀ x : ℂ, x^15 + x^10 + 1 = (x^3 - 1) * (x^12 + x^9 + x^6 + x^3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1542_154227


namespace NUMINAMATH_CALUDE_local_minimum_implies_a_equals_one_max_a_for_positive_f_main_result_l1542_154298

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - a * x / (x + 1)

-- Theorem for part (I)
theorem local_minimum_implies_a_equals_one (a : ℝ) :
  (∀ x, x > -1 → f a x ≥ f a 0) → a = 1 := by sorry

-- Theorem for part (II)
theorem max_a_for_positive_f (a : ℝ) :
  (∀ x, x > 0 → f a x > 0) → a ≤ 1 := by sorry

-- Theorem combining both parts
theorem main_result :
  (∃ a : ℝ, (∀ x, x > -1 → f a x ≥ f a 0) ∧ 
   (∀ a', (∀ x, x > 0 → f a' x > 0) → a' ≤ a)) ∧
  (∃ a : ℝ, a = 1 ∧ (∀ x, x > -1 → f a x ≥ f a 0) ∧ 
   (∀ a', (∀ x, x > 0 → f a' x > 0) → a' ≤ a)) := by sorry

end NUMINAMATH_CALUDE_local_minimum_implies_a_equals_one_max_a_for_positive_f_main_result_l1542_154298


namespace NUMINAMATH_CALUDE_problem_solution_l1542_154231

theorem problem_solution (p_xavier p_yvonne p_zelda : ℚ) 
  (h_xavier : p_xavier = 1/5)
  (h_yvonne : p_yvonne = 1/2)
  (h_zelda : p_zelda = 5/8)
  (h_independent : True) -- Assumption of independence
  : p_xavier * p_yvonne * (1 - p_zelda) = 3/80 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1542_154231


namespace NUMINAMATH_CALUDE_corveus_sleep_hours_l1542_154202

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the doctor's recommended hours of sleep per day -/
def recommended_sleep_per_day : ℕ := 6

/-- Represents the sleep deficit in hours per week -/
def sleep_deficit_per_week : ℕ := 14

/-- Calculates Corveus's actual sleep hours per day -/
def actual_sleep_per_day : ℚ :=
  (recommended_sleep_per_day * days_in_week - sleep_deficit_per_week) / days_in_week

/-- Proves that Corveus sleeps 4 hours per day given the conditions -/
theorem corveus_sleep_hours :
  actual_sleep_per_day = 4 := by sorry

end NUMINAMATH_CALUDE_corveus_sleep_hours_l1542_154202


namespace NUMINAMATH_CALUDE_fifteen_students_prefer_dogs_l1542_154297

/-- Represents the preferences of students in a class survey -/
structure ClassPreferences where
  total_students : ℕ
  dogs_videogames_chocolate : Rat
  dogs_videogames_vanilla : Rat
  dogs_movies_chocolate : Rat
  dogs_movies_vanilla : Rat
  cats_movies_chocolate : Rat
  cats_movies_vanilla : Rat
  cats_videogames_chocolate : Rat
  cats_videogames_vanilla : Rat

/-- Theorem stating that 15 students prefer dogs given the survey results -/
theorem fifteen_students_prefer_dogs (prefs : ClassPreferences) : 
  prefs.total_students = 30 ∧
  prefs.dogs_videogames_chocolate = 25/100 ∧
  prefs.dogs_videogames_vanilla = 5/100 ∧
  prefs.dogs_movies_chocolate = 10/100 ∧
  prefs.dogs_movies_vanilla = 10/100 ∧
  prefs.cats_movies_chocolate = 15/100 ∧
  prefs.cats_movies_vanilla = 10/100 ∧
  prefs.cats_videogames_chocolate = 5/100 ∧
  prefs.cats_videogames_vanilla = 10/100 →
  (prefs.dogs_videogames_chocolate + prefs.dogs_videogames_vanilla + 
   prefs.dogs_movies_chocolate + prefs.dogs_movies_vanilla) * prefs.total_students = 15 := by
  sorry


end NUMINAMATH_CALUDE_fifteen_students_prefer_dogs_l1542_154297


namespace NUMINAMATH_CALUDE_stratified_sample_size_is_15_l1542_154281

/-- Represents the number of workers in each age group -/
structure WorkerGroups where
  young : Nat
  middle_aged : Nat
  older : Nat

/-- Calculates the total sample size for a stratified sample -/
def stratified_sample_size (workers : WorkerGroups) (young_sample : Nat) : Nat :=
  let total_workers := workers.young + workers.middle_aged + workers.older
  let sampling_ratio := workers.young / young_sample
  total_workers / sampling_ratio

/-- Theorem: The stratified sample size for the given worker distribution is 15 -/
theorem stratified_sample_size_is_15 :
  let workers : WorkerGroups := ⟨35, 25, 15⟩
  stratified_sample_size workers 7 = 15 := by
  sorry

#eval stratified_sample_size ⟨35, 25, 15⟩ 7

end NUMINAMATH_CALUDE_stratified_sample_size_is_15_l1542_154281


namespace NUMINAMATH_CALUDE_inverse_proportion_quadrants_l1542_154204

/-- An inverse proportion function passing through (3, -5) is in the second and fourth quadrants -/
theorem inverse_proportion_quadrants :
  ∀ k : ℝ,
  (∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = k / x) ∧ f 3 = -5) →
  (∀ x y : ℝ, x ≠ 0 ∧ y = k / x → (x > 0 ∧ y < 0) ∨ (x < 0 ∧ y > 0)) :=
by sorry


end NUMINAMATH_CALUDE_inverse_proportion_quadrants_l1542_154204


namespace NUMINAMATH_CALUDE_sum_c_n_d_n_over_8_n_l1542_154256

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the sequences c_n and d_n
def c_n_d_n (n : ℕ) : ℂ := (3 + 2 * i) ^ n

-- Define c_n as the real part of c_n_d_n
def c_n (n : ℕ) : ℝ := (c_n_d_n n).re

-- Define d_n as the imaginary part of c_n_d_n
def d_n (n : ℕ) : ℝ := (c_n_d_n n).im

-- State the theorem
theorem sum_c_n_d_n_over_8_n :
  ∑' n, (c_n n * d_n n) / (8 : ℝ) ^ n = 6 / 17 := by sorry

end NUMINAMATH_CALUDE_sum_c_n_d_n_over_8_n_l1542_154256


namespace NUMINAMATH_CALUDE_average_string_length_l1542_154220

theorem average_string_length : 
  let string1 : ℚ := 2
  let string2 : ℚ := 5
  let string3 : ℚ := 7
  let total_length : ℚ := string1 + string2 + string3
  let num_strings : ℕ := 3
  (total_length / num_strings) = 14 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_string_length_l1542_154220


namespace NUMINAMATH_CALUDE_shaded_triangles_area_sum_l1542_154272

/-- The sum of areas of shaded triangles in an infinite geometric series --/
theorem shaded_triangles_area_sum (x y z : ℝ) (h1 : x = 8) (h2 : y = 8) (h3 : z = 8) 
  (h4 : x^2 = y^2 + z^2) : 
  let initial_area := (1/2) * y * z
  let first_shaded_area := (1/4) * initial_area
  let ratio := (1/4 : ℝ)
  (initial_area * ratio) / (1 - ratio) = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_triangles_area_sum_l1542_154272


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l1542_154237

theorem circle_diameter_from_area (A : Real) (π : Real) (h : π > 0) :
  A = 225 * π → ∃ d : Real, d > 0 ∧ A = π * (d / 2)^2 ∧ d = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l1542_154237


namespace NUMINAMATH_CALUDE_existence_of_n_l1542_154215

theorem existence_of_n (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_cd : c * d = 1) : 
  ∃ n : ℤ, a * b ≤ (n : ℝ)^2 ∧ (n : ℝ)^2 ≤ (a + c) * (b + d) :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_l1542_154215


namespace NUMINAMATH_CALUDE_constant_distance_l1542_154225

/-- Ellipse E with eccentricity 1/2 and area of triangle F₁PF₂ equal to 3 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : a^2/4 + b^2/3 = 1

/-- Point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2/E.a^2 + y^2/E.b^2 = 1

/-- Point on the line y = 2√3 -/
structure PointOnLine where
  x : ℝ
  y : ℝ
  h : y = 2 * Real.sqrt 3

/-- The theorem to be proved -/
theorem constant_distance (E : Ellipse) (M : PointOnEllipse E) (N : PointOnLine) 
  (h : (M.x * N.x + M.y * N.y) / (M.x^2 + M.y^2).sqrt / (N.x^2 + N.y^2).sqrt = 0) :
  ((M.y * N.x - M.x * N.y)^2 / ((M.x - N.x)^2 + (M.y - N.y)^2)).sqrt = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_distance_l1542_154225


namespace NUMINAMATH_CALUDE_min_hypotenuse_max_inscribed_circle_radius_l1542_154242

/-- A right-angled triangle with perimeter 1 meter -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  hypotenuse : ℝ
  perimeter_eq_one : a + b + hypotenuse = 1
  right_angle : a^2 + b^2 = hypotenuse^2
  positive : 0 < a ∧ 0 < b ∧ 0 < hypotenuse

/-- The minimum length of the hypotenuse in a right-angled triangle with perimeter 1 meter -/
theorem min_hypotenuse (t : RightTriangle) : t.hypotenuse ≥ Real.sqrt 2 - 1 := by sorry

/-- The maximum radius of the inscribed circle in a right-angled triangle with perimeter 1 meter -/
theorem max_inscribed_circle_radius (t : RightTriangle) : 
  t.a * t.b / (t.a + t.b + t.hypotenuse) ≤ 3/2 - Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_min_hypotenuse_max_inscribed_circle_radius_l1542_154242


namespace NUMINAMATH_CALUDE_vector_collinearity_l1542_154205

/-- Given vectors a, b, and c in ℝ², prove that if (a + 2b) is collinear with (3a - c),
    then the y-component of b equals -79/14. -/
theorem vector_collinearity (a b c : ℝ × ℝ) (h : a = (2, -3) ∧ b.1 = 4 ∧ c = (-1, 1)) :
  (∃ (k : ℝ), k • (a + 2 • b) = 3 • a - c) → b.2 = -79/14 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1542_154205


namespace NUMINAMATH_CALUDE_highest_percentage_increase_survey_d_l1542_154290

structure Survey where
  customers : ℕ
  responses : ℕ

def response_rate (s : Survey) : ℚ :=
  s.responses / s.customers

def percentage_change (a b : ℚ) : ℚ :=
  (b - a) / a * 100

theorem highest_percentage_increase_survey_d (survey_a survey_b survey_c survey_d : Survey)
  (ha : survey_a = { customers := 100, responses := 15 })
  (hb : survey_b = { customers := 120, responses := 27 })
  (hc : survey_c = { customers := 140, responses := 39 })
  (hd : survey_d = { customers := 160, responses := 56 }) :
  let change_ab := percentage_change (response_rate survey_a) (response_rate survey_b)
  let change_ac := percentage_change (response_rate survey_a) (response_rate survey_c)
  let change_ad := percentage_change (response_rate survey_a) (response_rate survey_d)
  change_ad > change_ab ∧ change_ad > change_ac := by
  sorry

end NUMINAMATH_CALUDE_highest_percentage_increase_survey_d_l1542_154290


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1542_154282

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, |x| < 0) ↔ (∃ x : ℝ, |x| ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1542_154282


namespace NUMINAMATH_CALUDE_negation_of_sum_equals_one_l1542_154286

theorem negation_of_sum_equals_one (a b : ℝ) :
  ¬(a + b = 1) ↔ (a + b > 1 ∨ a + b < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_sum_equals_one_l1542_154286


namespace NUMINAMATH_CALUDE_cube_surface_area_l1542_154271

/-- The surface area of a cube with volume 64 cubic cm is 96 square cm. -/
theorem cube_surface_area (cube_volume : ℝ) (h : cube_volume = 64) : 
  6 * (cube_volume ^ (1/3))^2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1542_154271


namespace NUMINAMATH_CALUDE_problem_solution_l1542_154201

theorem problem_solution (x y z : ℝ) 
  (h1 : x + y = 5) 
  (h2 : z^2 = x*y + y - 9) : 
  x + 2*y + 3*z = 8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1542_154201
