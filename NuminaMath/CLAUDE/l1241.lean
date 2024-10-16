import Mathlib

namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_monotonic_decreasing_interval_condition_not_always_above_line_l1241_124146

-- Define the function f(x) = x^3 - ax - 1
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

-- Statement 1
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → a ≤ 0 :=
sorry

-- Statement 2
theorem monotonic_decreasing_interval_condition (a : ℝ) :
  (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f a x > f a y) → a ≥ 3 :=
sorry

-- Statement 3
theorem not_always_above_line (a : ℝ) :
  ∃ x : ℝ, f a x < a :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_monotonic_decreasing_interval_condition_not_always_above_line_l1241_124146


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1241_124168

def M : Set ℕ := {0, 1, 3}

def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_of_M_and_N : M ∪ N = {0, 1, 3, 9} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1241_124168


namespace NUMINAMATH_CALUDE_root_between_consecutive_integers_l1241_124166

theorem root_between_consecutive_integers :
  ∃ (A B : ℤ), B = A + 1 ∧
  ∃ (x : ℝ), A < x ∧ x < B ∧ x^3 + 5*x^2 - 3*x + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_between_consecutive_integers_l1241_124166


namespace NUMINAMATH_CALUDE_fibonacci_periodicity_last_digit_2020th_fibonacci_l1241_124139

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

def last_digit (n : ℕ) : ℕ := n % 10

theorem fibonacci_periodicity (n : ℕ) : last_digit (fibonacci n) = last_digit (fibonacci (n % 60)) := by sorry

theorem last_digit_2020th_fibonacci : last_digit (fibonacci 2020) = 0 := by sorry

end NUMINAMATH_CALUDE_fibonacci_periodicity_last_digit_2020th_fibonacci_l1241_124139


namespace NUMINAMATH_CALUDE_sundae_cost_theorem_l1241_124155

def sundae_cost (monday_sundaes tuesday_sundaes : ℕ)
  (monday_mms monday_gummies monday_marshmallows : ℕ)
  (tuesday_mms tuesday_gummies tuesday_marshmallows : ℕ)
  (mms_per_pack gummies_per_pack marshmallows_per_pack : ℕ)
  (mms_pack_cost gummies_pack_cost marshmallows_pack_cost : ℚ) : ℚ :=
  let total_mms := monday_sundaes * monday_mms + tuesday_sundaes * tuesday_mms
  let total_gummies := monday_sundaes * monday_gummies + tuesday_sundaes * tuesday_gummies
  let total_marshmallows := monday_sundaes * monday_marshmallows + tuesday_sundaes * tuesday_marshmallows
  let mms_packs := (total_mms + mms_per_pack - 1) / mms_per_pack
  let gummies_packs := (total_gummies + gummies_per_pack - 1) / gummies_per_pack
  let marshmallows_packs := (total_marshmallows + marshmallows_per_pack - 1) / marshmallows_per_pack
  mms_packs * mms_pack_cost + gummies_packs * gummies_pack_cost + marshmallows_packs * marshmallows_pack_cost

theorem sundae_cost_theorem :
  sundae_cost 40 20 6 4 8 10 5 12 40 30 50 2 (3/2) 1 = 95/2 :=
by sorry

end NUMINAMATH_CALUDE_sundae_cost_theorem_l1241_124155


namespace NUMINAMATH_CALUDE_abs_neg_five_eq_five_l1241_124158

theorem abs_neg_five_eq_five : |(-5 : ℤ)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_eq_five_l1241_124158


namespace NUMINAMATH_CALUDE_last_three_average_l1241_124169

theorem last_three_average (a b c d : ℝ) : 
  (a + b + c) / 3 = 6 →
  a + d = 11 →
  d = 4 →
  (b + c + d) / 3 = 5 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l1241_124169


namespace NUMINAMATH_CALUDE_closest_point_parabola_to_line_l1241_124106

/-- The point (1, 1) on the parabola y^2 = x is the closest point to the line x - 2y + 4 = 0 -/
theorem closest_point_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2^2 = p.1}
  let line := {p : ℝ × ℝ | p.1 - 2*p.2 + 4 = 0}
  let distance (p : ℝ × ℝ) := |p.1 - 2*p.2 + 4| / Real.sqrt 5
  ∀ p ∈ parabola, distance (1, 1) ≤ distance p :=
by sorry

end NUMINAMATH_CALUDE_closest_point_parabola_to_line_l1241_124106


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1241_124122

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / y ≥ 9 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / x + 4 / y = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1241_124122


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l1241_124175

/-- 
Given a parabola y = ax^2 + 6 that is tangent to the line y = 2x - 3,
prove that the value of the constant a is 1/9.
-/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 6 = 2 * x - 3 ∧ 
   ∀ y : ℝ, y ≠ x → a * y^2 + 6 ≠ 2 * y - 3) →
  a = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l1241_124175


namespace NUMINAMATH_CALUDE_large_stores_count_l1241_124149

/-- Represents the total number of stores -/
def total_stores : ℕ := 1500

/-- Represents the sample size -/
def sample_size : ℕ := 90

/-- Represents the ratio of large stores -/
def large_ratio : ℕ := 3

/-- Represents the ratio of medium stores -/
def medium_ratio : ℕ := 5

/-- Represents the ratio of small stores -/
def small_ratio : ℕ := 7

/-- Calculates the number of large stores in the sample -/
def large_stores_in_sample : ℕ :=
  (sample_size * large_ratio) / (large_ratio + medium_ratio + small_ratio)

theorem large_stores_count :
  large_stores_in_sample = 18 := by sorry

end NUMINAMATH_CALUDE_large_stores_count_l1241_124149


namespace NUMINAMATH_CALUDE_max_candies_is_18_l1241_124174

/-- Represents the candy store's pricing structure and Maria's budget -/
structure CandyStore where
  single_price : ℕ := 2
  pack4_price : ℕ := 6
  pack7_price : ℕ := 10
  pack7_discount : ℕ := 3
  budget : ℕ := 25

/-- Calculates the maximum number of candies that can be purchased -/
def max_candies (store : CandyStore) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of candies Maria can buy is 18 -/
theorem max_candies_is_18 (store : CandyStore) : max_candies store = 18 :=
  sorry

end NUMINAMATH_CALUDE_max_candies_is_18_l1241_124174


namespace NUMINAMATH_CALUDE_small_pizza_price_is_two_l1241_124118

/-- The price of a small pizza given the conditions of the problem -/
def small_pizza_price (large_pizza_price : ℕ) (total_sales : ℕ) (small_pizzas_sold : ℕ) (large_pizzas_sold : ℕ) : ℕ :=
  (total_sales - large_pizza_price * large_pizzas_sold) / small_pizzas_sold

/-- Theorem stating that the price of a small pizza is $2 under the given conditions -/
theorem small_pizza_price_is_two :
  small_pizza_price 8 40 8 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_small_pizza_price_is_two_l1241_124118


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1241_124147

theorem arithmetic_sequence_length (a₁ : ℝ) (d : ℝ) (aₙ : ℝ) (n : ℕ) :
  a₁ = 3.5 ∧ d = 4 ∧ aₙ = 55.5 ∧ aₙ = a₁ + (n - 1) * d → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1241_124147


namespace NUMINAMATH_CALUDE_multiple_with_binary_digits_l1241_124159

theorem multiple_with_binary_digits (n : ℕ+) : ∃ m : ℕ,
  (n : ℕ) ∣ m ∧
  (Nat.digits 2 m).length ≤ n ∧
  ∀ d ∈ Nat.digits 2 m, d = 0 ∨ d = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiple_with_binary_digits_l1241_124159


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_l1241_124165

/-- Given parametric equations x = 1 + 2cosθ and y = 2sinθ, 
    prove they are equivalent to the Cartesian equation (x-1)² + y² = 4 -/
theorem parametric_to_cartesian :
  ∀ (x y θ : ℝ), 
  x = 1 + 2 * Real.cos θ ∧ 
  y = 2 * Real.sin θ → 
  (x - 1)^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_l1241_124165


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1241_124124

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 48 → (∀ x y : ℕ, x * y = 48 → x + y ≤ heart + club) → heart + club = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1241_124124


namespace NUMINAMATH_CALUDE_tunnel_crossing_possible_l1241_124107

/-- Represents a friend with their crossing time -/
structure Friend where
  name : String
  time : Nat

/-- Represents a crossing of the tunnel -/
inductive Crossing
  | Forward : List Friend → Crossing
  | Backward : Friend → Crossing

/-- Calculates the time taken for a crossing -/
def crossingTime (c : Crossing) : Nat :=
  match c with
  | Crossing.Forward friends => friends.map Friend.time |>.maximum?.getD 0
  | Crossing.Backward friend => friend.time

/-- The tunnel crossing problem -/
def tunnelCrossing (friends : List Friend) : Prop :=
  ∃ (crossings : List Crossing),
    -- All friends have crossed
    (crossings.filter (λ c => match c with
      | Crossing.Forward _ => true
      | Crossing.Backward _ => false
    )).bind (λ c => match c with
      | Crossing.Forward fs => fs
      | Crossing.Backward _ => []
    ) = friends
    ∧
    -- The total time is exactly 17 minutes
    (crossings.map crossingTime).sum = 17
    ∧
    -- Each crossing involves at most two friends
    ∀ c ∈ crossings, match c with
      | Crossing.Forward fs => fs.length ≤ 2
      | Crossing.Backward _ => true

theorem tunnel_crossing_possible : 
  let friends := [
    { name := "One", time := 1 },
    { name := "Two", time := 2 },
    { name := "Five", time := 5 },
    { name := "Ten", time := 10 }
  ]
  tunnelCrossing friends :=
by
  sorry


end NUMINAMATH_CALUDE_tunnel_crossing_possible_l1241_124107


namespace NUMINAMATH_CALUDE_xyz_value_l1241_124191

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 12)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 16) :
  x * y * z = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l1241_124191


namespace NUMINAMATH_CALUDE_division_remainder_3005_95_l1241_124150

theorem division_remainder_3005_95 : 3005 % 95 = 60 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_3005_95_l1241_124150


namespace NUMINAMATH_CALUDE_canoe_rowing_probability_l1241_124160

/-- The probability of rowing a canoe given certain conditions on oar functionality and weather -/
theorem canoe_rowing_probability :
  let p_left_works : ℚ := 3/5  -- Probability left oar works
  let p_right_works : ℚ := 3/5  -- Probability right oar works
  let p_weather : ℚ := 1/4  -- Probability of adverse weather
  let p_oar_works_in_weather (p : ℚ) : ℚ := 1 - 2 * (1 - p)  -- Probability oar works in adverse weather
  
  let p_both_work_no_weather : ℚ := p_left_works * p_right_works
  let p_both_work_weather : ℚ := p_oar_works_in_weather p_left_works * p_oar_works_in_weather p_right_works
  
  let p_row : ℚ := p_both_work_no_weather * (1 - p_weather) + p_both_work_weather * p_weather

  p_row = 7/25 := by sorry

end NUMINAMATH_CALUDE_canoe_rowing_probability_l1241_124160


namespace NUMINAMATH_CALUDE_ratio_fourth_term_l1241_124134

theorem ratio_fourth_term (x y : ℝ) (hx : x = 0.8571428571428571) :
  (0.75 : ℝ) / x = 7 / y → y = 8 := by
sorry

end NUMINAMATH_CALUDE_ratio_fourth_term_l1241_124134


namespace NUMINAMATH_CALUDE_bryan_bus_time_l1241_124163

/-- Represents the travel time for Bryan's commute -/
structure CommuteTimes where
  walkToStation : ℕ  -- Time to walk from house to bus station
  walkToWork : ℕ     -- Time to walk from bus station to work
  totalYearlyTime : ℕ -- Total yearly commute time in hours
  daysWorked : ℕ     -- Number of days worked per year

/-- Calculates the one-way bus ride time in minutes -/
def onewayBusTime (c : CommuteTimes) : ℕ :=
  let totalDailyTime := (c.totalYearlyTime * 60) / c.daysWorked
  let totalWalkTime := 2 * (c.walkToStation + c.walkToWork)
  (totalDailyTime - totalWalkTime) / 2

/-- Theorem stating that Bryan's one-way bus ride time is 20 minutes -/
theorem bryan_bus_time :
  let c := CommuteTimes.mk 5 5 365 365
  onewayBusTime c = 20 := by
  sorry

end NUMINAMATH_CALUDE_bryan_bus_time_l1241_124163


namespace NUMINAMATH_CALUDE_final_amounts_l1241_124196

/-- Represents a person with their current amount of money -/
structure Person where
  name : String
  amount : ℚ

/-- Represents the state of all persons involved in the transactions -/
structure State where
  michael : Person
  thomas : Person
  emily : Person

/-- Performs the series of transactions described in the problem -/
def performTransactions (initial : State) : State :=
  let s1 := { initial with
    michael := { initial.michael with amount := initial.michael.amount * (1 - 0.3) },
    thomas := { initial.thomas with amount := initial.thomas.amount + initial.michael.amount * 0.3 }
  }
  let s2 := { s1 with
    thomas := { s1.thomas with amount := s1.thomas.amount * (1 - 0.25) },
    emily := { s1.emily with amount := s1.emily.amount + s1.thomas.amount * 0.25 }
  }
  let s3 := { s2 with
    emily := { s2.emily with amount := (s2.emily.amount - 10) / 2 },
    michael := { s2.michael with amount := s2.michael.amount + (s2.emily.amount - 10) / 2 }
  }
  s3

/-- The main theorem stating the final amounts after transactions -/
theorem final_amounts (initial : State)
  (h_michael : initial.michael.amount = 42)
  (h_thomas : initial.thomas.amount = 17)
  (h_emily : initial.emily.amount = 30) :
  let final := performTransactions initial
  final.michael.amount = 43.1 ∧
  final.thomas.amount = 22.2 ∧
  final.emily.amount = 13.7 := by
  sorry


end NUMINAMATH_CALUDE_final_amounts_l1241_124196


namespace NUMINAMATH_CALUDE_min_value_of_f_l1241_124193

def f (x : ℝ) : ℝ := |2*x + 1| + |x - 1|

theorem min_value_of_f :
  ∃ (min_val : ℝ), min_val = 3/2 ∧ ∀ (x : ℝ), f x ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1241_124193


namespace NUMINAMATH_CALUDE_pig_bacon_profit_l1241_124176

def average_pig_bacon : ℝ := 20
def average_type_a_bacon : ℝ := 12
def average_type_b_bacon : ℝ := 8
def type_a_price : ℝ := 6
def type_b_price : ℝ := 4
def this_pig_size_ratio : ℝ := 0.5
def this_pig_type_a_ratio : ℝ := 0.75
def this_pig_type_b_ratio : ℝ := 0.25
def type_a_cost : ℝ := 1.5
def type_b_cost : ℝ := 0.8

theorem pig_bacon_profit : 
  let this_pig_bacon := average_pig_bacon * this_pig_size_ratio
  let this_pig_type_a := this_pig_bacon * this_pig_type_a_ratio
  let this_pig_type_b := this_pig_bacon * this_pig_type_b_ratio
  let revenue := this_pig_type_a * type_a_price + this_pig_type_b * type_b_price
  let cost := this_pig_type_a * type_a_cost + this_pig_type_b * type_b_cost
  revenue - cost = 41.75 := by
sorry

end NUMINAMATH_CALUDE_pig_bacon_profit_l1241_124176


namespace NUMINAMATH_CALUDE_store_b_earns_more_l1241_124131

/-- Represents the total value of goods sold by each store in yuan -/
def total_sales : ℕ := 1000000

/-- Represents the discount rate offered by store A -/
def discount_rate : ℚ := 1/10

/-- Represents the cost of a lottery ticket in yuan -/
def ticket_cost : ℕ := 100

/-- Represents the number of tickets in a batch -/
def tickets_per_batch : ℕ := 10000

/-- Represents the prize structure for store B -/
structure PrizeStructure where
  first_prize : ℕ × ℕ  -- (number of prizes, value of each prize)
  second_prize : ℕ × ℕ
  third_prize : ℕ × ℕ
  fourth_prize : ℕ × ℕ
  fifth_prize : ℕ × ℕ

/-- The actual prize structure used by store B -/
def store_b_prizes : PrizeStructure := {
  first_prize := (5, 1000),
  second_prize := (10, 500),
  third_prize := (20, 200),
  fourth_prize := (40, 100),
  fifth_prize := (5000, 10)
}

/-- Calculates the total prize value for a given prize structure -/
def total_prize_value (ps : PrizeStructure) : ℕ :=
  ps.first_prize.1 * ps.first_prize.2 +
  ps.second_prize.1 * ps.second_prize.2 +
  ps.third_prize.1 * ps.third_prize.2 +
  ps.fourth_prize.1 * ps.fourth_prize.2 +
  ps.fifth_prize.1 * ps.fifth_prize.2

/-- Theorem stating that store B earns at least 32,000 yuan more than store A -/
theorem store_b_earns_more :
  ∃ (x : ℕ), x ≥ 32000 ∧
  (total_sales - (total_prize_value store_b_prizes) * (total_sales / (tickets_per_batch * ticket_cost))) =
  (total_sales * (1 - discount_rate)).floor + x :=
by sorry


end NUMINAMATH_CALUDE_store_b_earns_more_l1241_124131


namespace NUMINAMATH_CALUDE_fraction_order_l1241_124125

theorem fraction_order (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hac : a < c) (hbd : b > d) :
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d ∧
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < (a + c) / (b - d) ∧
  (c - a) / (b + d) < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d ∧
  (c - a) / (b + d) < (a + c) / (b + d) ∧ (a + c) / (b + d) < (a + c) / (b - d) ∧
  (c - a) / (b + d) < (c - a) / (b - d) ∧ (c - a) / (b - d) < (a + c) / (b - d) :=
by sorry

end NUMINAMATH_CALUDE_fraction_order_l1241_124125


namespace NUMINAMATH_CALUDE_total_miles_driven_l1241_124188

theorem total_miles_driven (darius_miles julia_miles : ℕ) 
  (h1 : darius_miles = 679) 
  (h2 : julia_miles = 998) : 
  darius_miles + julia_miles = 1677 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_driven_l1241_124188


namespace NUMINAMATH_CALUDE_jame_practice_weeks_l1241_124197

def regular_cards_per_tear : ℕ := 30
def thick_cards_per_tear : ℕ := 25
def cards_per_regular_deck : ℕ := 52
def cards_per_thick_deck : ℕ := 55
def tears_per_week : ℕ := 4
def regular_decks_bought : ℕ := 27
def thick_decks_bought : ℕ := 14

def total_cards : ℕ := regular_decks_bought * cards_per_regular_deck + thick_decks_bought * cards_per_thick_deck

def cards_torn_per_week : ℕ := (regular_cards_per_tear + thick_cards_per_tear) * (tears_per_week / 2)

theorem jame_practice_weeks :
  (total_cards / cards_torn_per_week : ℕ) = 19 := by sorry

end NUMINAMATH_CALUDE_jame_practice_weeks_l1241_124197


namespace NUMINAMATH_CALUDE_inscribed_circumscribed_inequality_l1241_124104

/-- A polygon inscribed in one circle and circumscribed around another -/
structure InscribedCircumscribedPolygon where
  /-- Area of the inscribing circle -/
  A : ℝ
  /-- Area of the polygon -/
  B : ℝ
  /-- Area of the circumscribed circle -/
  C : ℝ
  /-- The inscribing circle has positive area -/
  hA : 0 < A
  /-- The polygon has positive area -/
  hB : 0 < B
  /-- The circumscribed circle has positive area -/
  hC : 0 < C
  /-- The polygon's area is less than or equal to the inscribing circle's area -/
  hAB : B ≤ A
  /-- The circumscribed circle's area is less than or equal to the polygon's area -/
  hBC : C ≤ B

/-- The inequality holds for any inscribed-circumscribed polygon configuration -/
theorem inscribed_circumscribed_inequality (p : InscribedCircumscribedPolygon) : 2 * p.B ≤ p.A + p.C := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circumscribed_inequality_l1241_124104


namespace NUMINAMATH_CALUDE_combined_diving_depths_l1241_124182

theorem combined_diving_depths (ron_height : ℝ) (water_depth : ℝ) : 
  ron_height = 12 →
  water_depth = 5 * ron_height →
  let dean_height := ron_height - 11
  let sam_height := dean_height + 2
  let ron_dive := ron_height / 2
  let sam_dive := sam_height
  let dean_dive := dean_height + 3
  ron_dive + sam_dive + dean_dive = 13 := by sorry

end NUMINAMATH_CALUDE_combined_diving_depths_l1241_124182


namespace NUMINAMATH_CALUDE_circle_common_chord_l1241_124135

theorem circle_common_chord (a : ℝ) (h : a > 0) : 
  ∃ (x y : ℝ), 
    (x^2 + y^2 = 4 ∧ 
     x^2 + y^2 + 2*x + 2*a*y - 6 = 0 ∧ 
     ∃ (x₁ y₁ x₂ y₂ : ℝ), 
       (x₁^2 + y₁^2 = 4 ∧ 
        x₁^2 + y₁^2 + 2*x₁ + 2*a*y₁ - 6 = 0 ∧
        x₂^2 + y₂^2 = 4 ∧ 
        x₂^2 + y₂^2 + 2*x₂ + 2*a*y₂ - 6 = 0 ∧
        (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12)) →
    a = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_common_chord_l1241_124135


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l1241_124105

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (x > -1 ∧ x < 1) ↔ (∀ y : ℝ, y > x → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l1241_124105


namespace NUMINAMATH_CALUDE_ab_equals_one_l1241_124117

theorem ab_equals_one (θ : ℝ) (a b : ℝ) 
  (h1 : a * Real.sin θ + Real.cos θ = 1)
  (h2 : b * Real.sin θ - Real.cos θ = 1) :
  a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_one_l1241_124117


namespace NUMINAMATH_CALUDE_rectangle_area_error_percentage_l1241_124138

/-- Given a rectangle where one side is measured 16% in excess and the other side is measured 5% in deficit, 
    the error percentage in the calculated area is 10.2%. -/
theorem rectangle_area_error_percentage (L W : ℝ) (L_positive : L > 0) (W_positive : W > 0) : 
  let actual_area := L * W
  let measured_length := L * (1 + 16/100)
  let measured_width := W * (1 - 5/100)
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  let error_percentage := (error / actual_area) * 100
  error_percentage = 10.2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percentage_l1241_124138


namespace NUMINAMATH_CALUDE_complex_number_modulus_one_l1241_124164

theorem complex_number_modulus_one (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  Complex.abs z = 1 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_one_l1241_124164


namespace NUMINAMATH_CALUDE_histogram_frequency_l1241_124195

theorem histogram_frequency (sample_size : ℕ) (num_groups : ℕ) (class_interval : ℕ) (rectangle_height : ℝ) : 
  sample_size = 100 →
  num_groups = 10 →
  class_interval = 10 →
  rectangle_height = 0.03 →
  (rectangle_height * class_interval * sample_size : ℝ) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_histogram_frequency_l1241_124195


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_negative_one_l1241_124153

theorem sum_of_a_and_b_is_negative_one :
  ∀ (a b : ℝ) (S T : ℕ → ℝ),
  (∀ n, S n = 2^n + a) →  -- Sum of geometric sequence
  (∀ n, T n = n^2 - 2*n + b) →  -- Sum of arithmetic sequence
  a + b = -1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_negative_one_l1241_124153


namespace NUMINAMATH_CALUDE_fifth_term_is_nine_l1241_124108

/-- An arithmetic sequence with first term 1 and common difference 2 -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  1 + (n - 1) * 2

/-- The fifth term of the arithmetic sequence is 9 -/
theorem fifth_term_is_nine : arithmetic_sequence 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_nine_l1241_124108


namespace NUMINAMATH_CALUDE_george_socks_problem_l1241_124119

theorem george_socks_problem (initial_socks : ℕ) (new_socks : ℕ) (final_socks : ℕ) 
  (h1 : initial_socks = 28)
  (h2 : new_socks = 36)
  (h3 : final_socks = 60) :
  initial_socks - (initial_socks + new_socks - final_socks) + new_socks = final_socks ∧ 
  initial_socks + new_socks - final_socks = 4 :=
by sorry

end NUMINAMATH_CALUDE_george_socks_problem_l1241_124119


namespace NUMINAMATH_CALUDE_price_reduction_equation_l1241_124178

theorem price_reduction_equation (x : ℝ) : 
  (∃ (original_price final_price : ℝ),
    original_price = 28 ∧ 
    final_price = 16 ∧ 
    final_price = original_price * (1 - x)^2) →
  28 * (1 - x)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l1241_124178


namespace NUMINAMATH_CALUDE_square_divides_power_plus_one_l1241_124143

theorem square_divides_power_plus_one (n : ℕ+) : n^2 ∣ 2^(n : ℕ) + 1 ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_divides_power_plus_one_l1241_124143


namespace NUMINAMATH_CALUDE_minimum_value_and_range_proof_l1241_124129

theorem minimum_value_and_range_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (min : ℝ), min = 9 ∧ (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + b' = 1 → 1/a' + 4/b' ≥ min) ∧
    (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1/a₀ + 4/b₀ = min)) ∧
  (∀ x : ℝ, (1/a + 4/b ≥ |2*x - 1| - |x + 1|) → -7 ≤ x ∧ x ≤ 11) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_and_range_proof_l1241_124129


namespace NUMINAMATH_CALUDE_wait_hare_is_random_l1241_124120

-- Define the type for events
inductive Event
| StrongYouth
| ScoopMoon
| WaitHare
| GreenMountains

-- Define what it means for an event to be random
def isRandom (e : Event) : Prop :=
  match e with
  | Event.WaitHare => True
  | _ => False

-- Theorem statement
theorem wait_hare_is_random :
  isRandom Event.WaitHare :=
sorry

end NUMINAMATH_CALUDE_wait_hare_is_random_l1241_124120


namespace NUMINAMATH_CALUDE_ap_eq_aq_l1241_124171

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A circle in the Euclidean plane -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- A line in the Euclidean plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Definition of an acute-angled triangle -/
def isAcuteAngled (A B C : Point) : Prop :=
  sorry

/-- Definition of a circle with a given diameter -/
def circleWithDiameter (P Q : Point) : Circle :=
  sorry

/-- Definition of the intersection of a line and a circle -/
def lineCircleIntersection (l : Line) (c : Circle) : Set Point :=
  sorry

theorem ap_eq_aq 
  (A B C : Point)
  (h_acute : isAcuteAngled A B C)
  (circle_AC : Circle)
  (circle_AB : Circle)
  (h_circle_AC : circle_AC = circleWithDiameter A C)
  (h_circle_AB : circle_AB = circleWithDiameter A B)
  (F : Point)
  (h_F : F ∈ lineCircleIntersection (Line.mk 0 1 0) circle_AC)
  (E : Point)
  (h_E : E ∈ lineCircleIntersection (Line.mk 1 0 0) circle_AB)
  (BE CF : Line)
  (P : Point)
  (h_P : P ∈ lineCircleIntersection BE circle_AC)
  (Q : Point)
  (h_Q : Q ∈ lineCircleIntersection CF circle_AB) :
  (A.x - P.x)^2 + (A.y - P.y)^2 = (A.x - Q.x)^2 + (A.y - Q.y)^2 :=
sorry

end NUMINAMATH_CALUDE_ap_eq_aq_l1241_124171


namespace NUMINAMATH_CALUDE_product_of_three_integers_l1241_124130

theorem product_of_three_integers : (-3 : ℤ) * (-4 : ℤ) * (-1 : ℤ) = -12 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_integers_l1241_124130


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1241_124180

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem quadratic_function_properties :
  (∀ x, f x = x^2 - 2*x + 3) ∧
  (f 0 = 3) ∧
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∀ x ≤ 1, ∀ y ≥ x, f x ≥ f y) ∧
  (∀ x ≥ 1, ∀ y ≥ x, f x ≤ f y) ∧
  (∀ x, f x ≥ 2) ∧
  (f 1 = 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1241_124180


namespace NUMINAMATH_CALUDE_paths_from_A_to_E_l1241_124141

/-- The number of paths between two consecutive points -/
def paths_between_consecutive : ℕ := 2

/-- The number of direct paths from A to E -/
def direct_paths : ℕ := 1

/-- The number of intermediate points between A and E -/
def intermediate_points : ℕ := 4

/-- The total number of paths from A to E -/
def total_paths : ℕ := paths_between_consecutive ^ intermediate_points + direct_paths

theorem paths_from_A_to_E : total_paths = 17 := by sorry

end NUMINAMATH_CALUDE_paths_from_A_to_E_l1241_124141


namespace NUMINAMATH_CALUDE_base_8_23456_equals_10030_l1241_124161

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ (digits.length - 1 - i))) 0

theorem base_8_23456_equals_10030 :
  base_8_to_10 [2, 3, 4, 5, 6] = 10030 := by
  sorry

end NUMINAMATH_CALUDE_base_8_23456_equals_10030_l1241_124161


namespace NUMINAMATH_CALUDE_circle_area_sum_l1241_124127

/-- The sum of the areas of an infinite sequence of circles with decreasing radii -/
theorem circle_area_sum : 
  let r : ℕ → ℝ := λ n => 2 * (1/3)^(n-1)
  let area : ℕ → ℝ := λ n => π * (r n)^2
  (∑' n, area n) = 9*π/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_sum_l1241_124127


namespace NUMINAMATH_CALUDE_race_time_difference_l1241_124109

/-- Proves the time difference between Malcolm and Joshua finishing a race --/
theorem race_time_difference 
  (malcolm_speed : ℝ) 
  (joshua_speed : ℝ) 
  (race_distance : ℝ) 
  (h1 : malcolm_speed = 5)
  (h2 : joshua_speed = 7)
  (h3 : race_distance = 12) :
  joshua_speed * race_distance - malcolm_speed * race_distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l1241_124109


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1241_124189

/-- The atomic weight of Carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in the compound -/
def carbon_count : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def oxygen_count : ℕ := 1

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight : ℝ := carbon_weight * carbon_count + oxygen_weight * oxygen_count

theorem compound_molecular_weight :
  molecular_weight = 28.01 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1241_124189


namespace NUMINAMATH_CALUDE_product_of_divisors_implies_n_l1241_124136

/-- The product of all positive divisors of a natural number -/
def divisor_product (n : ℕ) : ℕ := sorry

/-- The number of positive divisors of a natural number -/
def divisor_count (n : ℕ) : ℕ := sorry

theorem product_of_divisors_implies_n (N : ℕ) :
  divisor_product N = 2^120 * 3^60 * 5^90 → N = 18000 := by sorry

end NUMINAMATH_CALUDE_product_of_divisors_implies_n_l1241_124136


namespace NUMINAMATH_CALUDE_equal_quantities_after_transfer_l1241_124132

def container_problem (initial_A initial_B initial_C transfer : ℝ) : Prop :=
  let final_B := initial_B + transfer
  let final_C := initial_C - transfer
  initial_A = 1184 ∧
  initial_B = 0.375 * initial_A ∧
  initial_C = initial_A - initial_B ∧
  transfer = 148 →
  final_B = final_C

theorem equal_quantities_after_transfer :
  ∃ (initial_A initial_B initial_C transfer : ℝ),
    container_problem initial_A initial_B initial_C transfer :=
  sorry

end NUMINAMATH_CALUDE_equal_quantities_after_transfer_l1241_124132


namespace NUMINAMATH_CALUDE_bearings_count_proof_l1241_124116

/-- The number of machines -/
def num_machines : ℕ := 10

/-- The normal cost per ball bearing in cents -/
def normal_cost : ℕ := 100

/-- The sale price per ball bearing in cents -/
def sale_price : ℕ := 75

/-- The additional discount rate for bulk purchase -/
def bulk_discount : ℚ := 1/5

/-- The amount saved in cents by buying during the sale -/
def amount_saved : ℕ := 12000

/-- The number of ball bearings per machine -/
def bearings_per_machine : ℕ := 30

theorem bearings_count_proof :
  ∃ (x : ℕ),
    x = bearings_per_machine ∧
    (num_machines * normal_cost * x) -
    (num_machines * sale_price * x * (1 - bulk_discount)) =
    amount_saved :=
sorry

end NUMINAMATH_CALUDE_bearings_count_proof_l1241_124116


namespace NUMINAMATH_CALUDE_soap_brand_usage_l1241_124115

theorem soap_brand_usage (total : ℕ) (neither : ℕ) (only_e : ℕ) (both : ℕ) :
  total = 200 →
  neither = 80 →
  only_e = 60 →
  total = neither + only_e + both + 3 * both →
  both = 15 := by
sorry

end NUMINAMATH_CALUDE_soap_brand_usage_l1241_124115


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1241_124142

theorem rectangular_to_polar_conversion :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 3 * Real.sqrt 2 ∧ θ = π / 4 ∧
  r * Real.cos θ = 3 ∧ r * Real.sin θ = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1241_124142


namespace NUMINAMATH_CALUDE_validSchedules_eq_1296_l1241_124187

/-- Represents a chess tournament between two universities -/
structure ChessTournament where
  university1 : Fin 3 → Type
  university2 : Fin 3 → Type
  rounds : Fin 6 → Fin 3 → (Fin 3 × Fin 3)
  no_immediate_repeat : ∀ (r : Fin 5) (i : Fin 3),
    (rounds r i).1 ≠ (rounds (r + 1) i).1 ∨ (rounds r i).2 ≠ (rounds (r + 1) i).2

/-- The number of valid tournament schedules -/
def validSchedules : ℕ := sorry

/-- Theorem stating the number of valid tournament schedules is 1296 -/
theorem validSchedules_eq_1296 : validSchedules = 1296 := by sorry

end NUMINAMATH_CALUDE_validSchedules_eq_1296_l1241_124187


namespace NUMINAMATH_CALUDE_sqrt_7_simplest_l1241_124177

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ 
  (∀ (z : ℕ), z > 1 → ¬(∃ (w : ℝ), y = z * w ^ 2)) ∧
  y ≠ 1

theorem sqrt_7_simplest : 
  is_simplest_quadratic_radical (Real.sqrt 7) ∧
  ¬(is_simplest_quadratic_radical (Real.sqrt 4)) ∧
  ¬(is_simplest_quadratic_radical (Real.sqrt (1/4))) ∧
  ¬(is_simplest_quadratic_radical (Real.sqrt 27)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_7_simplest_l1241_124177


namespace NUMINAMATH_CALUDE_box_side_length_l1241_124173

/-- Given the total volume needed, cost per box, and minimum total cost,
    calculate the length of one side of a cubic box. -/
theorem box_side_length 
  (total_volume : ℝ) 
  (cost_per_box : ℝ) 
  (min_total_cost : ℝ) 
  (h1 : total_volume = 1920000) 
  (h2 : cost_per_box = 0.5) 
  (h3 : min_total_cost = 200) : 
  ∃ (side_length : ℝ), abs (side_length - 16.89) < 0.01 := by
  sorry

#check box_side_length

end NUMINAMATH_CALUDE_box_side_length_l1241_124173


namespace NUMINAMATH_CALUDE_betty_boxes_theorem_l1241_124145

/-- The number of boxes Betty uses in an average harvest -/
def num_boxes : ℕ := 20

/-- The capacity of each box in parsnips -/
def box_capacity : ℕ := 20

/-- The fraction of boxes that are full -/
def full_box_fraction : ℚ := 3/4

/-- The fraction of boxes that are half-full -/
def half_full_box_fraction : ℚ := 1/4

/-- The number of parsnips in a half-full box -/
def half_full_box_content : ℕ := box_capacity / 2

/-- The total number of parsnips in an average harvest -/
def total_parsnips : ℕ := 350

/-- Theorem stating that the number of boxes used is correct given the conditions -/
theorem betty_boxes_theorem : 
  (↑num_boxes * full_box_fraction * ↑box_capacity : ℚ) + 
  (↑num_boxes * half_full_box_fraction * ↑half_full_box_content : ℚ) = ↑total_parsnips :=
by sorry

end NUMINAMATH_CALUDE_betty_boxes_theorem_l1241_124145


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l1241_124156

theorem fraction_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l1241_124156


namespace NUMINAMATH_CALUDE_cindy_hit_nine_l1241_124157

/-- Represents a player in the dart-throwing contest -/
inductive Player
| Alice
| Ben
| Cindy
| Dave
| Ellen

/-- Represents the score of a single dart throw -/
def DartScore := Fin 15

/-- Represents the scores of three dart throws for a player -/
def PlayerScores := Fin 3 → DartScore

/-- The total score for a player is the sum of their three dart scores -/
def totalScore (scores : PlayerScores) : Nat :=
  (scores 0).val + (scores 1).val + (scores 2).val

/-- The scores for each player -/
def playerTotalScores : Player → Nat
| Player.Alice => 24
| Player.Ben => 13
| Player.Cindy => 19
| Player.Dave => 28
| Player.Ellen => 30

/-- Predicate to check if a player's scores contain a specific value -/
def containsScore (scores : PlayerScores) (n : DartScore) : Prop :=
  ∃ i, scores i = n

/-- Statement: Cindy is the only player who hit the region worth 9 points -/
theorem cindy_hit_nine :
  ∃! p : Player, ∃ scores : PlayerScores,
    totalScore scores = playerTotalScores p ∧
    containsScore scores ⟨9, by norm_num⟩ ∧
    p = Player.Cindy :=
by
  sorry

end NUMINAMATH_CALUDE_cindy_hit_nine_l1241_124157


namespace NUMINAMATH_CALUDE_angle_A_in_triangle_l1241_124123

-- Define the triangle ABC
structure Triangle where
  A : Real
  b : Real
  c : Real
  S : Real

-- State the theorem
theorem angle_A_in_triangle (abc : Triangle) (h1 : abc.b = 8) (h2 : abc.c = 8 * Real.sqrt 3) 
  (h3 : abc.S = 16 * Real.sqrt 3) : 
  abc.A = π / 6 ∨ abc.A = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_in_triangle_l1241_124123


namespace NUMINAMATH_CALUDE_logical_equivalence_l1241_124151

theorem logical_equivalence (R S T : Prop) :
  (R → ¬S ∧ ¬T) ↔ (S ∨ T → ¬R) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l1241_124151


namespace NUMINAMATH_CALUDE_tan_beta_value_l1241_124110

theorem tan_beta_value (α β : ℝ) 
  (h1 : Real.tan α = 1 / 2)
  (h2 : Real.tan (α - β / 2) = 1 / 3) :
  Real.tan β = 7 / 24 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l1241_124110


namespace NUMINAMATH_CALUDE_solution_2015_squared_l1241_124186

theorem solution_2015_squared : 
  ∃ x : ℚ, (2015 + x)^2 = x^2 ∧ x = -2015/2 := by
sorry

end NUMINAMATH_CALUDE_solution_2015_squared_l1241_124186


namespace NUMINAMATH_CALUDE_total_interest_received_l1241_124101

/-- Simple interest calculation function -/
def simple_interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time / 100

theorem total_interest_received (loan_b_principal loan_c_principal : ℕ) 
  (loan_b_time loan_c_time : ℕ) (interest_rate : ℚ) : 
  loan_b_principal = 5000 →
  loan_c_principal = 3000 →
  loan_b_time = 2 →
  loan_c_time = 4 →
  interest_rate = 15 →
  simple_interest loan_b_principal interest_rate loan_b_time + 
  simple_interest loan_c_principal interest_rate loan_c_time = 3300 := by
sorry

end NUMINAMATH_CALUDE_total_interest_received_l1241_124101


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l1241_124113

/-- The number of dogwood trees in the park after planting -/
def total_trees (current : ℕ) (today : ℕ) (tomorrow : ℕ) : ℕ :=
  current + today + tomorrow

/-- Theorem stating the total number of dogwood trees after planting -/
theorem dogwood_tree_count :
  total_trees 39 41 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l1241_124113


namespace NUMINAMATH_CALUDE_prime_sequence_l1241_124102

def f (p : ℕ) (x : ℕ) : ℕ := x^2 + x + p

theorem prime_sequence (p : ℕ) :
  (∀ k : ℕ, k ≤ Real.sqrt (p / 3) → Nat.Prime (f p k)) →
  (∀ n : ℕ, n ≤ p - 2 → Nat.Prime (f p n)) :=
sorry

end NUMINAMATH_CALUDE_prime_sequence_l1241_124102


namespace NUMINAMATH_CALUDE_smallest_positive_congruence_l1241_124162

theorem smallest_positive_congruence :
  ∃ (n : ℕ), n > 0 ∧ n < 13 ∧ -1234 ≡ n [ZMOD 13] ∧
  ∀ (m : ℕ), m > 0 ∧ m < 13 ∧ -1234 ≡ m [ZMOD 13] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_congruence_l1241_124162


namespace NUMINAMATH_CALUDE_yellow_last_probability_l1241_124181

/-- Represents a bag of marbles -/
structure Bag where
  yellow : ℕ
  blue : ℕ
  white : ℕ
  black : ℕ
  green : ℕ
  red : ℕ

/-- The probability of drawing a yellow marble as the last marble -/
def last_yellow_probability (bagA bagB bagC bagD : Bag) : ℚ :=
  sorry

/-- The theorem stating the probability of drawing a yellow marble last -/
theorem yellow_last_probability :
  let bagA : Bag := { yellow := 0, blue := 0, white := 5, black := 5, green := 0, red := 0 }
  let bagB : Bag := { yellow := 8, blue := 6, white := 0, black := 0, green := 0, red := 0 }
  let bagC : Bag := { yellow := 3, blue := 7, white := 0, black := 0, green := 0, red := 0 }
  let bagD : Bag := { yellow := 0, blue := 0, white := 0, black := 0, green := 4, red := 6 }
  last_yellow_probability bagA bagB bagC bagD = 73 / 140 := by
  sorry

end NUMINAMATH_CALUDE_yellow_last_probability_l1241_124181


namespace NUMINAMATH_CALUDE_geometric_sum_problem_l1241_124114

-- Define the sum of a geometric sequence
def GeometricSum (n : ℕ) := ℝ

-- State the theorem
theorem geometric_sum_problem :
  ∀ (S : ℕ → ℝ),
  (S 2 = 4) →
  (S 4 = 6) →
  (S 6 = 7) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_problem_l1241_124114


namespace NUMINAMATH_CALUDE_A_is_singleton_floor_sum_property_l1241_124190

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the set A
def A : Set ℝ :=
  {x | x^2 - (floor x : ℝ) - 1 = 0 ∧ -1 < x ∧ x < 2}

-- Theorem 1: A is a singleton set
theorem A_is_singleton : ∃! x, x ∈ A := by sorry

-- Theorem 2: Floor function property
theorem floor_sum_property (x : ℝ) :
  (floor x : ℝ) + (floor (x + 1/2) : ℝ) = (floor (2*x) : ℝ) := by sorry

end NUMINAMATH_CALUDE_A_is_singleton_floor_sum_property_l1241_124190


namespace NUMINAMATH_CALUDE_set_operation_result_l1241_124133

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 6}
def B : Set Nat := {1, 2}

theorem set_operation_result (C : Set Nat) (h : C ⊆ U) : 
  (C ∪ A) ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l1241_124133


namespace NUMINAMATH_CALUDE_exists_shorter_representation_l1241_124140

def repeatedSevens (n : ℕ) : ℕ := 
  (7 * (10^n - 1)) / 9

def validExpression (expr : ℕ → ℕ) : Prop :=
  ∃ (k : ℕ), expr k = repeatedSevens k ∧ 
  (∀ m : ℕ, m ≤ k → expr m ≠ repeatedSevens m)

theorem exists_shorter_representation : 
  ∃ (n : ℕ) (expr : ℕ → ℕ), n > 2 ∧ validExpression expr ∧ 
  (∀ k : ℕ, k ≥ n → expr k < repeatedSevens k) :=
sorry

end NUMINAMATH_CALUDE_exists_shorter_representation_l1241_124140


namespace NUMINAMATH_CALUDE_smallest_N_and_digit_sum_l1241_124137

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem smallest_N_and_digit_sum :
  ∃ N : ℕ, 
    (∀ k : ℕ, k < N → k * (k + 1) ≤ 10^6) ∧
    N * (N + 1) > 10^6 ∧
    N = 1000 ∧
    sum_of_digits N = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_N_and_digit_sum_l1241_124137


namespace NUMINAMATH_CALUDE_total_savings_ten_sets_l1241_124144

/-- The cost of 2 packs of milk -/
def cost_two_packs : ℚ := 2.50

/-- The cost of an individual pack of milk -/
def cost_individual : ℚ := 1.30

/-- The number of sets being purchased -/
def num_sets : ℕ := 10

/-- The number of packs in each set -/
def packs_per_set : ℕ := 2

/-- Theorem stating the total savings from buying ten sets of 2 packs of milk -/
theorem total_savings_ten_sets : 
  (num_sets * packs_per_set) * (cost_individual - cost_two_packs / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_ten_sets_l1241_124144


namespace NUMINAMATH_CALUDE_min_dimension_sum_for_2310_volume_l1241_124154

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- The volume of a box given its dimensions -/
def volume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- The sum of the dimensions of a box -/
def dimensionSum (d : BoxDimensions) : ℕ := d.length + d.width + d.height

/-- Theorem stating that the minimum sum of dimensions for a box with volume 2310 is 52 -/
theorem min_dimension_sum_for_2310_volume :
  (∃ (d : BoxDimensions), volume d = 2310) →
  (∀ (d : BoxDimensions), volume d = 2310 → dimensionSum d ≥ 52) ∧
  (∃ (d : BoxDimensions), volume d = 2310 ∧ dimensionSum d = 52) :=
sorry

end NUMINAMATH_CALUDE_min_dimension_sum_for_2310_volume_l1241_124154


namespace NUMINAMATH_CALUDE_min_fencing_cost_problem_l1241_124185

/-- Represents the cost of fencing materials in rupees per meter -/
structure FencingMaterial where
  cost : ℚ

/-- Represents a rectangular field -/
structure RectangularField where
  length : ℚ
  width : ℚ
  area : ℚ

/-- Calculates the minimum fencing cost for a rectangular field -/
def minFencingCost (field : RectangularField) (materials : List FencingMaterial) : ℚ :=
  sorry

/-- Theorem stating the minimum fencing cost for the given problem -/
theorem min_fencing_cost_problem :
  let field : RectangularField := {
    length := 108,
    width := 81,
    area := 8748
  }
  let materials : List FencingMaterial := [
    { cost := 0.25 },
    { cost := 0.35 },
    { cost := 0.40 }
  ]
  minFencingCost field materials = 87.75 := by sorry

end NUMINAMATH_CALUDE_min_fencing_cost_problem_l1241_124185


namespace NUMINAMATH_CALUDE_monthly_interest_advantage_l1241_124192

theorem monthly_interest_advantage (p : ℝ) (n : ℕ) (hp : p > 0) (hn : n > 0) :
  (1 + p / (12 * 100)) ^ (6 * n) > (1 + p / (2 * 100)) ^ n :=
sorry

end NUMINAMATH_CALUDE_monthly_interest_advantage_l1241_124192


namespace NUMINAMATH_CALUDE_sum_of_incircle_areas_l1241_124179

/-- Given a triangle ABC with side lengths a, b, c and inradius r, 
    the sum of the areas of its incircle and the incircles of the three smaller triangles 
    formed by tangent lines to the incircle parallel to the sides of ABC 
    is equal to (7πr²)/4. -/
theorem sum_of_incircle_areas (a b c r : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) :
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  r = K / s →
  (π * r^2) + 3 * (π * (r/2)^2) = (7 * π * r^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_incircle_areas_l1241_124179


namespace NUMINAMATH_CALUDE_boxes_needed_to_sell_l1241_124103

def total_chocolate_bars : ℕ := 710
def chocolate_bars_per_box : ℕ := 5

theorem boxes_needed_to_sell (total : ℕ) (per_box : ℕ) :
  total = total_chocolate_bars →
  per_box = chocolate_bars_per_box →
  total / per_box = 142 := by
  sorry

end NUMINAMATH_CALUDE_boxes_needed_to_sell_l1241_124103


namespace NUMINAMATH_CALUDE_henrys_age_l1241_124170

/-- Given that the sum of Henry and Jill's present ages is 48, and 9 years ago Henry was twice the age of Jill, 
    prove that Henry's present age is 29 years. -/
theorem henrys_age (henry_age jill_age : ℕ) 
  (sum_condition : henry_age + jill_age = 48)
  (past_condition : henry_age - 9 = 2 * (jill_age - 9)) : 
  henry_age = 29 := by
  sorry

end NUMINAMATH_CALUDE_henrys_age_l1241_124170


namespace NUMINAMATH_CALUDE_interest_rate_is_twelve_percent_l1241_124112

/-- Calculate the interest rate given principal, time, and interest amount -/
def calculate_interest_rate (principal : ℕ) (time : ℕ) (interest : ℕ) : ℚ :=
  (interest * 100) / (principal * time)

/-- Theorem: The interest rate is 12% given the problem conditions -/
theorem interest_rate_is_twelve_percent (principal : ℕ) (time : ℕ) (interest : ℕ)
  (h1 : principal = 9200)
  (h2 : time = 3)
  (h3 : interest = principal - 5888) :
  calculate_interest_rate principal time interest = 12 := by
  sorry

#eval calculate_interest_rate 9200 3 (9200 - 5888)

end NUMINAMATH_CALUDE_interest_rate_is_twelve_percent_l1241_124112


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a5_value_l1241_124199

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a5_value
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_nonzero : ∃ n : ℕ, a n ≠ 0)
  (h_eq : a 5 ^ 2 - a 3 - a 7 = 0) :
  a 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a5_value_l1241_124199


namespace NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1241_124126

-- Define sets A and B
def A : Set ℝ := {y | ∃ x, 0 ≤ x ∧ x ≤ 3 ∧ y = (x - 1)^2 + 1}
def B : Set ℝ := {y | ∃ x, 1 ≤ x ∧ x ≤ 3 ∧ y = x^2 + 1}

-- Define set difference
def setDifference (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

-- Define symmetric difference
def symmetricDifference (X Y : Set ℝ) : Set ℝ := 
  (setDifference X Y) ∪ (setDifference Y X)

-- Theorem statement
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {y | (1 ≤ y ∧ y < 2) ∨ (5 < y ∧ y ≤ 10)} := by
  sorry

end NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1241_124126


namespace NUMINAMATH_CALUDE_two_zeros_cubic_l1241_124152

def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

theorem two_zeros_cubic (c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f c x = 0 ∧ f c y = 0 ∧ ∀ z : ℝ, f c z = 0 → z = x ∨ z = y) ↔ c = -2 ∨ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_zeros_cubic_l1241_124152


namespace NUMINAMATH_CALUDE_sin_75_degrees_l1241_124172

theorem sin_75_degrees : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_degrees_l1241_124172


namespace NUMINAMATH_CALUDE_portraits_not_taken_l1241_124100

theorem portraits_not_taken (total_students : ℕ) (before_lunch : ℕ) (after_lunch : ℕ) : 
  total_students = 24 → 
  before_lunch = total_students / 3 →
  after_lunch = 10 →
  total_students - (before_lunch + after_lunch) = 6 := by
sorry

end NUMINAMATH_CALUDE_portraits_not_taken_l1241_124100


namespace NUMINAMATH_CALUDE_prakash_copies_five_pages_l1241_124184

/-- Represents the number of pages a person can copy in a given time -/
structure CopyingRate where
  pages : ℕ
  hours : ℕ

/-- Subash's copying rate -/
def subash_rate : CopyingRate := ⟨50, 10⟩

/-- Combined copying rate of Subash and Prakash -/
def combined_rate : CopyingRate := ⟨300, 40⟩

/-- Calculate the number of pages Prakash can copy in 2 hours -/
def prakash_pages : ℕ :=
  let subash_40_hours := (subash_rate.pages * combined_rate.hours) / subash_rate.hours
  let prakash_40_hours := combined_rate.pages - subash_40_hours
  (prakash_40_hours * 2) / combined_rate.hours

theorem prakash_copies_five_pages : prakash_pages = 5 := by
  sorry

end NUMINAMATH_CALUDE_prakash_copies_five_pages_l1241_124184


namespace NUMINAMATH_CALUDE_annuity_duration_exists_l1241_124121

/-- The duration of the original annuity in years -/
def original_duration : ℝ := 20

/-- The interest rate as a decimal -/
def interest_rate : ℝ := 0.04

/-- The equation that the new duration must satisfy -/
def annuity_equation (x : ℝ) : Prop :=
  Real.exp x = 2 * Real.exp original_duration / (Real.exp original_duration + 1)

/-- Theorem stating the existence of a solution to the annuity equation -/
theorem annuity_duration_exists :
  ∃ x : ℝ, annuity_equation x :=
sorry

end NUMINAMATH_CALUDE_annuity_duration_exists_l1241_124121


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l1241_124167

theorem probability_at_least_one_correct (n : ℕ) (k : ℕ) :
  n > 0 → k > 0 →
  let p := 1 - (1 - 1 / n) ^ k
  p = 11529 / 15625 ↔ n = 5 ∧ k = 6 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l1241_124167


namespace NUMINAMATH_CALUDE_dragon_can_be_defeated_l1241_124111

/-- Represents the possible number of heads a warrior can chop off in one strike -/
inductive Strike
  | thirtythree
  | twentyone
  | seventeen
  | one

/-- Represents the state of the dragon -/
structure DragonState where
  heads : ℕ

/-- Applies a strike to the dragon state -/
def applyStrike (s : Strike) (d : DragonState) : DragonState :=
  match s with
  | Strike.thirtythree => ⟨d.heads + 48 - 33⟩
  | Strike.twentyone => ⟨d.heads - 21⟩
  | Strike.seventeen => ⟨d.heads + 14 - 17⟩
  | Strike.one => ⟨d.heads + 349 - 1⟩

/-- Represents a sequence of strikes -/
def StrikeSequence := List Strike

/-- Applies a sequence of strikes to the dragon state -/
def applySequence (seq : StrikeSequence) (d : DragonState) : DragonState :=
  seq.foldl (fun state strike => applyStrike strike state) d

/-- The theorem stating that the dragon can be defeated -/
theorem dragon_can_be_defeated : 
  ∃ (seq : StrikeSequence), (applySequence seq ⟨2000⟩).heads = 0 := by
  sorry

end NUMINAMATH_CALUDE_dragon_can_be_defeated_l1241_124111


namespace NUMINAMATH_CALUDE_water_evaporation_l1241_124128

/-- Given a bowl with 10 ounces of water, if 2% of the original amount evaporates
    over 50 days, then the amount of water evaporated each day is 0.04 ounces. -/
theorem water_evaporation (initial_water : ℝ) (days : ℕ) (evaporation_rate : ℝ) :
  initial_water = 10 →
  days = 50 →
  evaporation_rate = 0.02 →
  (initial_water * evaporation_rate) / days = 0.04 :=
by sorry

end NUMINAMATH_CALUDE_water_evaporation_l1241_124128


namespace NUMINAMATH_CALUDE_grand_total_profit_is_8515_l1241_124148

/-- Represents a type of necklace with its properties -/
structure NecklaceType where
  charms : ℕ
  charmCost : ℕ
  sellingPrice : ℕ

/-- Calculates the profit for a single necklace of a given type -/
def profit (n : NecklaceType) : ℕ :=
  n.sellingPrice - n.charms * n.charmCost

/-- Calculates the total profit for a given number of necklaces of a specific type -/
def totalProfit (n : NecklaceType) (quantity : ℕ) : ℕ :=
  quantity * profit n

/-- Theorem stating that the grand total profit is $8515 -/
theorem grand_total_profit_is_8515 :
  let typeA : NecklaceType := ⟨8, 10, 125⟩
  let typeB : NecklaceType := ⟨12, 18, 280⟩
  let typeC : NecklaceType := ⟨15, 12, 350⟩
  totalProfit typeA 45 + totalProfit typeB 35 + totalProfit typeC 25 = 8515 := by
  sorry

#eval let typeA : NecklaceType := ⟨8, 10, 125⟩
      let typeB : NecklaceType := ⟨12, 18, 280⟩
      let typeC : NecklaceType := ⟨15, 12, 350⟩
      totalProfit typeA 45 + totalProfit typeB 35 + totalProfit typeC 25

end NUMINAMATH_CALUDE_grand_total_profit_is_8515_l1241_124148


namespace NUMINAMATH_CALUDE_total_kids_l1241_124194

theorem total_kids (girls : ℕ) (boys : ℕ) (h1 : girls = 3) (h2 : boys = 6) :
  girls + boys = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_kids_l1241_124194


namespace NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisibility_l1241_124183

theorem product_of_three_consecutive_integers_divisibility
  (k : ℤ)
  (n : ℤ)
  (h1 : n = k * (k + 1) * (k + 2))
  (h2 : 5 ∣ n) :
  (6 ∣ n) ∧
  (10 ∣ n) ∧
  (15 ∣ n) ∧
  (30 ∣ n) ∧
  ∃ m : ℤ, n = m ∧ ¬(20 ∣ m) := by
sorry

end NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisibility_l1241_124183


namespace NUMINAMATH_CALUDE_robin_bracelet_cost_l1241_124198

def cost_per_bracelet : ℕ := 2

def friend_names : List String := ["Jessica", "Tori", "Lily", "Patrice"]

def total_letters (names : List String) : ℕ :=
  names.map String.length |>.sum

def total_cost (names : List String) (cost : ℕ) : ℕ :=
  (total_letters names) * cost

theorem robin_bracelet_cost :
  total_cost friend_names cost_per_bracelet = 44 := by
  sorry

end NUMINAMATH_CALUDE_robin_bracelet_cost_l1241_124198
