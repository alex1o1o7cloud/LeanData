import Mathlib

namespace chantel_final_bracelets_l3062_306292

def bracelets_made_first_period : ℕ := 5 * 2
def bracelets_given_school : ℕ := 3
def bracelets_made_second_period : ℕ := 4 * 3
def bracelets_given_soccer : ℕ := 6

theorem chantel_final_bracelets :
  bracelets_made_first_period - bracelets_given_school + bracelets_made_second_period - bracelets_given_soccer = 13 := by
  sorry

end chantel_final_bracelets_l3062_306292


namespace jons_payment_per_visit_l3062_306253

/-- Represents the payment structure for Jon's website -/
structure WebsitePayment where
  visits_per_hour : ℕ
  hours_per_day : ℕ
  days_per_month : ℕ
  monthly_revenue : ℚ

/-- Calculates the payment per visit given the website payment structure -/
def payment_per_visit (wp : WebsitePayment) : ℚ :=
  wp.monthly_revenue / (wp.visits_per_hour * wp.hours_per_day * wp.days_per_month)

/-- Theorem stating that Jon's payment per visit is $0.10 -/
theorem jons_payment_per_visit :
  let wp : WebsitePayment := {
    visits_per_hour := 50,
    hours_per_day := 24,
    days_per_month := 30,
    monthly_revenue := 3600
  }
  payment_per_visit wp = 1/10 := by
  sorry

end jons_payment_per_visit_l3062_306253


namespace inequality_proof_l3062_306271

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c = 1) : 
  Real.sqrt (a * b / (c + a * b)) + 
  Real.sqrt (b * c / (a + b * c)) + 
  Real.sqrt (c * a / (b + c * a)) ≤ 3 / 2 := by
  sorry

end inequality_proof_l3062_306271


namespace triangle_abc_properties_l3062_306296

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a^2 + c^2 = b^2 + Real.sqrt 2 * a * c →
  c = Real.sqrt 3 + 1 →
  Real.sin A = 1/2 →
  (B = π/4 ∧ 1/2 * a * b * Real.sin C = (Real.sqrt 3 + 1) / 2) :=
by sorry

end triangle_abc_properties_l3062_306296


namespace work_completion_time_l3062_306247

/-- The time needed to complete the work -/
def complete_work (p q : ℝ) (t : ℝ) : Prop :=
  let work_p := t / p
  let work_q := (t - 16) / q
  work_p + work_q = 1

theorem work_completion_time :
  ∀ p q : ℝ,
  p > 0 → q > 0 →
  complete_work p q 40 →
  complete_work q q 24 →
  ∃ t : ℝ, t > 0 ∧ complete_work p q t ∧ t = 25 := by
sorry

end work_completion_time_l3062_306247


namespace circplus_two_three_one_l3062_306214

/-- Definition of the ⊕ operation -/
def circplus (a b c : ℝ) : ℝ := b^2 - 4*a*c + c^2

/-- Theorem: The value of ⊕(2, 3, 1) is 2 -/
theorem circplus_two_three_one : circplus 2 3 1 = 2 := by
  sorry

end circplus_two_three_one_l3062_306214


namespace product_of_five_consecutive_not_square_l3062_306273

theorem product_of_five_consecutive_not_square :
  ∀ a : ℕ, a > 0 →
    ¬∃ n : ℕ, a * (a + 1) * (a + 2) * (a + 3) * (a + 4) = n^2 :=
by sorry

end product_of_five_consecutive_not_square_l3062_306273


namespace wall_clock_ring_interval_l3062_306272

/-- Represents a wall clock that rings multiple times a day at equal intervals -/
structure WallClock where
  rings_per_day : ℕ
  minutes_per_day : ℕ

/-- Calculates the time between two consecutive rings in minutes -/
def time_between_rings (clock : WallClock) : ℕ :=
  clock.minutes_per_day / (clock.rings_per_day - 1)

theorem wall_clock_ring_interval :
  let clock : WallClock := { rings_per_day := 6, minutes_per_day := 24 * 60 }
  time_between_rings clock = 288 := by
  sorry

end wall_clock_ring_interval_l3062_306272


namespace chess_tournament_girls_l3062_306260

theorem chess_tournament_girls (n : ℕ) (x : ℕ) : 
  (n > 0) →  -- number of girls is positive
  (2 * n * x + 16 = (n + 2) * (n + 1)) →  -- total points equation
  (x > 0) →  -- each girl's score is positive
  (n = 7 ∨ n = 14) := by
  sorry

end chess_tournament_girls_l3062_306260


namespace count_eight_to_800_l3062_306203

/-- Count of digit 8 in a single number -/
def count_eight (n : ℕ) : ℕ := sorry

/-- Sum of count_eight for all numbers from 1 to n -/
def sum_count_eight (n : ℕ) : ℕ := sorry

/-- The count of the digit 8 in all integers from 1 to 800 is 161 -/
theorem count_eight_to_800 : sum_count_eight 800 = 161 := by sorry

end count_eight_to_800_l3062_306203


namespace g_of_8_l3062_306209

theorem g_of_8 (g : ℝ → ℝ) (h : ∀ x, x ≠ 2 → g x = (7 * x + 3) / (x - 2)) :
  g 8 = 59 / 6 := by
  sorry

end g_of_8_l3062_306209


namespace equality_from_cubic_relations_equality_from_mixed_cubic_relations_l3062_306291

theorem equality_from_cubic_relations (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * (b^3 + c^3) = b * (c^3 + a^3) ∧ b * (c^3 + a^3) = c * (a^3 + b^3)) → 
  (a = b ∧ b = c) :=
by sorry

theorem equality_from_mixed_cubic_relations (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * (a^3 + b^3) = b * (b^3 + c^3) ∧ b * (b^3 + c^3) = c * (c^3 + a^3)) → 
  (a = b ∧ b = c) :=
by sorry

end equality_from_cubic_relations_equality_from_mixed_cubic_relations_l3062_306291


namespace f_passes_through_point_two_zero_l3062_306231

-- Define the function f
def f (x : ℝ) : ℝ := x - 2

-- Theorem statement
theorem f_passes_through_point_two_zero : f 2 = 0 := by
  sorry

end f_passes_through_point_two_zero_l3062_306231


namespace pam_total_fruits_l3062_306275

-- Define the given conditions
def pam_apple_bags : ℕ := 6
def pam_orange_bags : ℕ := 4
def gerald_apple_bags : ℕ := 5
def gerald_orange_bags : ℕ := 4
def gerald_apples_per_bag : ℕ := 30
def gerald_oranges_per_bag : ℕ := 25
def pam_apple_ratio : ℕ := 3
def pam_orange_ratio : ℕ := 2

-- Theorem to prove
theorem pam_total_fruits :
  pam_apple_bags * (pam_apple_ratio * gerald_apples_per_bag) +
  pam_orange_bags * (pam_orange_ratio * gerald_oranges_per_bag) = 740 := by
  sorry


end pam_total_fruits_l3062_306275


namespace shortest_distance_point_l3062_306242

/-- Given points A and B, find the point P on the y-axis that minimizes AP + BP -/
theorem shortest_distance_point (A B P : ℝ × ℝ) : 
  A = (3, 2) →
  B = (1, -2) →
  P.1 = 0 →
  P = (0, -1) →
  ∀ Q : ℝ × ℝ, Q.1 = 0 → 
    Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) + Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) ≤ 
    Real.sqrt ((A.1 - Q.1)^2 + (A.2 - Q.2)^2) + Real.sqrt ((B.1 - Q.1)^2 + (B.2 - Q.2)^2) :=
by sorry


end shortest_distance_point_l3062_306242


namespace min_value_and_y_l3062_306259

theorem min_value_and_y (x y z : ℝ) (h : 2*x - 3*y + z = 3) :
  ∃ (min_val : ℝ), 
    (∀ x' y' z' : ℝ, 2*x' - 3*y' + z' = 3 → x'^2 + (y' - 1)^2 + z'^2 ≥ min_val) ∧
    (x^2 + (y - 1)^2 + z^2 = min_val ↔ y = -2/7) ∧
    min_val = 18/7 :=
sorry

end min_value_and_y_l3062_306259


namespace distance_to_work_is_18_l3062_306287

/-- The distance Esther drives to work -/
def distance_to_work : ℝ := 18

/-- The average speed to work in miles per hour -/
def speed_to_work : ℝ := 45

/-- The average speed from work in miles per hour -/
def speed_from_work : ℝ := 30

/-- The total commute time in hours -/
def total_commute_time : ℝ := 1

/-- Theorem stating that the distance to work is 18 miles given the conditions -/
theorem distance_to_work_is_18 :
  (distance_to_work / speed_to_work) + (distance_to_work / speed_from_work) = total_commute_time :=
by sorry

end distance_to_work_is_18_l3062_306287


namespace triangle_cosine_value_l3062_306264

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b - c = 1/4 * a and 2 * sin B = 3 * sin C, then cos A = -1/4 -/
theorem triangle_cosine_value (a b c A B C : ℝ) :
  b - c = (1/4) * a →
  2 * Real.sin B = 3 * Real.sin C →
  Real.cos A = -(1/4) :=
by sorry

end triangle_cosine_value_l3062_306264


namespace min_value_quadratic_l3062_306289

theorem min_value_quadratic (x y : ℝ) : 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 ≥ 3 := by
  sorry

end min_value_quadratic_l3062_306289


namespace dice_probability_l3062_306239

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 8

/-- The number of sides showing numbers less than or equal to 4 -/
def favorableOutcomes : ℕ := 4

/-- The number of dice required to show numbers less than or equal to 4 -/
def requiredSuccesses : ℕ := 4

/-- The probability of rolling a number less than or equal to 4 on a single die -/
def singleDieProbability : ℚ := favorableOutcomes / numSides

theorem dice_probability :
  Nat.choose numDice requiredSuccesses *
  singleDieProbability ^ requiredSuccesses *
  (1 - singleDieProbability) ^ (numDice - requiredSuccesses) =
  35 / 128 :=
sorry

end dice_probability_l3062_306239


namespace quadratic_equations_solutions_l3062_306211

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁^2 - 3 * x₁ - 2 = 0 ∧ x₁ = -1/2) ∧
                (2 * x₂^2 - 3 * x₂ - 2 = 0 ∧ x₂ = 2)) ∧
  (∃ y₁ y₂ : ℝ, (2 * y₁^2 - 3 * y₁ - 1 = 0 ∧ y₁ = (3 + Real.sqrt 17) / 4) ∧
                (2 * y₂^2 - 3 * y₂ - 1 = 0 ∧ y₂ = (3 - Real.sqrt 17) / 4)) :=
by sorry

end quadratic_equations_solutions_l3062_306211


namespace gcd_of_three_numbers_l3062_306288

theorem gcd_of_three_numbers : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end gcd_of_three_numbers_l3062_306288


namespace tangent_and_common_point_l3062_306268

/-- The line l: y = kx - 3k + 2 -/
def line (k : ℝ) (x : ℝ) : ℝ := k * x - 3 * k + 2

/-- The curve C: (x-1)² + (y+1)² = 4 where -1 ≤ x ≤ 1 -/
def curve (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 1)^2 = 4 ∧ -1 ≤ x ∧ x ≤ 1

theorem tangent_and_common_point (k : ℝ) :
  (∃ x, curve x (line k x) ∧
    ∀ x', x' ≠ x → ¬ curve x' (line k x')) ↔
  k = 5/12 ∨ (1/2 < k ∧ k ≤ 5/2) :=
sorry

end tangent_and_common_point_l3062_306268


namespace monkeys_eating_birds_l3062_306270

theorem monkeys_eating_birds (initial_monkeys initial_birds : ℕ) 
  (h1 : initial_monkeys = 6)
  (h2 : initial_birds = 6)
  (h3 : ∃ (monkeys_ate : ℕ), 
    (initial_monkeys : ℚ) / (initial_monkeys + initial_birds - monkeys_ate) = 3/5) :
  ∃ (monkeys_ate : ℕ), monkeys_ate = 2 := by
sorry

end monkeys_eating_birds_l3062_306270


namespace total_strikes_is_180_l3062_306241

/-- Calculates the total number of strikes made by a clock in a 24-hour period. -/
def total_strikes : ℕ :=
  let hourly_strikes := 12 * 13 / 2 * 2  -- Sum of 1 to 12, twice
  let half_hour_strikes := 24            -- One strike every half hour (excluding full hours)
  hourly_strikes + half_hour_strikes

/-- Theorem stating that the total number of strikes in a 24-hour period is 180. -/
theorem total_strikes_is_180 : total_strikes = 180 := by
  sorry

end total_strikes_is_180_l3062_306241


namespace cafeteria_apples_l3062_306254

/-- The number of apples handed out to students -/
def apples_handed_out : ℕ := 8

/-- The number of apples needed for each pie -/
def apples_per_pie : ℕ := 9

/-- The number of pies that could be made with the remaining apples -/
def pies_made : ℕ := 6

/-- The initial number of apples in the cafeteria -/
def initial_apples : ℕ := 62

theorem cafeteria_apples :
  initial_apples = apples_handed_out + apples_per_pie * pies_made :=
by sorry

end cafeteria_apples_l3062_306254


namespace smallest_divisor_after_391_l3062_306250

/-- Given an even 4-digit number m where 391 is a divisor,
    the smallest possible divisor of m greater than 391 is 441 -/
theorem smallest_divisor_after_391 (m : ℕ) (h1 : 1000 ≤ m) (h2 : m < 10000) 
    (h3 : Even m) (h4 : m % 391 = 0) : 
  ∃ (d : ℕ), d ∣ m ∧ d > 391 ∧ d ≥ 441 ∧ ∀ (x : ℕ), x ∣ m → x > 391 → x ≥ d :=
by sorry

end smallest_divisor_after_391_l3062_306250


namespace min_value_of_f_l3062_306293

def f (x : ℝ) := |3 - x| + |x - 2|

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x, f x = m) ∧ m = 1 := by
  sorry

end min_value_of_f_l3062_306293


namespace range_of_m_l3062_306295

-- Define the conditions
def p (x : ℝ) : Prop := -x^2 + 8*x + 20 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  ((∀ x, q x m → p x) ∧ (∃ x, p x ∧ ¬q x m) → m ≥ 9) ∧
  ((∀ x, ¬q x m → ¬p x) ∧ (∃ x, ¬p x ∧ q x m) → 0 < m ∧ m ≤ 3) :=
by sorry

end range_of_m_l3062_306295


namespace parabola_fv_unique_value_l3062_306236

/-- A parabola with vertex V and focus F -/
structure Parabola where
  V : ℝ × ℝ
  F : ℝ × ℝ

/-- A point on a parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_fv_unique_value (p : Parabola) 
  (A B : PointOnParabola p)
  (h1 : distance A.point p.F = 25)
  (h2 : distance A.point p.V = 24)
  (h3 : distance B.point p.F = 9) :
  distance p.F p.V = 9 := sorry

end parabola_fv_unique_value_l3062_306236


namespace candy_bar_cost_is_one_l3062_306220

/-- The cost of a candy bar given initial and remaining amounts -/
def candy_bar_cost (initial_amount : ℝ) (remaining_amount : ℝ) : ℝ :=
  initial_amount - remaining_amount

/-- Theorem: The candy bar costs $1 given the conditions -/
theorem candy_bar_cost_is_one :
  let initial_amount : ℝ := 4
  let remaining_amount : ℝ := 3
  candy_bar_cost initial_amount remaining_amount = 1 := by
sorry

end candy_bar_cost_is_one_l3062_306220


namespace vegetable_baskets_l3062_306217

/-- Calculates the number of baskets needed to store vegetables --/
theorem vegetable_baskets
  (keith_turnips : ℕ)
  (alyssa_turnips : ℕ)
  (sean_carrots : ℕ)
  (turnips_per_basket : ℕ)
  (carrots_per_basket : ℕ)
  (h1 : keith_turnips = 6)
  (h2 : alyssa_turnips = 9)
  (h3 : sean_carrots = 5)
  (h4 : turnips_per_basket = 5)
  (h5 : carrots_per_basket = 4) :
  (((keith_turnips + alyssa_turnips) + turnips_per_basket - 1) / turnips_per_basket) +
  ((sean_carrots + carrots_per_basket - 1) / carrots_per_basket) = 5 :=
by sorry

end vegetable_baskets_l3062_306217


namespace difference_le_two_l3062_306210

/-- Represents a right-angled triangle with integer sides -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h_right_angle : a ^ 2 + b ^ 2 = c ^ 2
  h_ordered : a < b ∧ b < c
  h_coprime : Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c

/-- The difference between the hypotenuse and the middle side -/
def difference (t : RightTriangle) : ℕ := t.c - t.b

/-- Theorem: For a right-angled triangle with integer sides a, b, c where
    a < b < c, a, b, c are pairwise co-prime, and (c - b) divides a,
    then (c - b) ≤ 2 -/
theorem difference_le_two (t : RightTriangle) (h_divides : t.a % (difference t) = 0) :
  difference t ≤ 2 := by
  sorry

end difference_le_two_l3062_306210


namespace midpoint_triangle_area_l3062_306233

/-- The area of the nth triangle formed by repeatedly connecting midpoints -/
def triangleArea (n : ℕ) : ℚ :=
  (1 / 4 : ℚ) ^ (n - 1) * (3 / 2 : ℚ)

/-- The original right triangle ABC with sides 3, 4, and 5 -/
structure OriginalTriangle where
  sideA : ℕ := 3
  sideB : ℕ := 4
  sideC : ℕ := 5

theorem midpoint_triangle_area (t : OriginalTriangle) (n : ℕ) (h : n ≥ 1) :
  triangleArea n = (1 / 4 : ℚ) ^ (n - 1) * (3 / 2 : ℚ) :=
by sorry

end midpoint_triangle_area_l3062_306233


namespace light_bulb_packs_theorem_l3062_306257

/-- Calculates the number of light bulb packs needed given the number of bulbs required in each room -/
def light_bulb_packs_needed (bedroom bathroom kitchen basement : ℕ) : ℕ :=
  let total_without_garage := bedroom + bathroom + kitchen + basement
  let garage := total_without_garage / 2
  let total := total_without_garage + garage
  (total + 1) / 2

/-- Theorem stating that given the specific number of light bulbs needed in each room,
    the number of packs needed is 6 -/
theorem light_bulb_packs_theorem :
  light_bulb_packs_needed 2 1 1 4 = 6 := by
  sorry

#eval light_bulb_packs_needed 2 1 1 4

end light_bulb_packs_theorem_l3062_306257


namespace number_of_boys_l3062_306221

/-- Proves that the number of boys is 15 given the problem conditions -/
theorem number_of_boys (men women boys : ℕ) (total_earnings men_wage : ℕ) : 
  (5 * men = women) → 
  (women = boys) → 
  (total_earnings = 180) → 
  (men_wage = 12) → 
  (5 * men * men_wage + women * (total_earnings - 5 * men * men_wage) / (women + boys) + 
   boys * (total_earnings - 5 * men * men_wage) / (women + boys) = total_earnings) →
  boys = 15 := by
sorry

end number_of_boys_l3062_306221


namespace factorization_equality_l3062_306283

theorem factorization_equality (x y z : ℝ) :
  x^2 - 4*y^2 - z^2 + 4*y*z = (x + 2*y - z) * (x - 2*y + z) := by
  sorry

end factorization_equality_l3062_306283


namespace point_P_in_second_quadrant_l3062_306279

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : CartesianPoint) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: Point P(-3, 2) is in the second quadrant -/
theorem point_P_in_second_quadrant :
  let P : CartesianPoint := ⟨-3, 2⟩
  is_in_second_quadrant P := by
  sorry

end point_P_in_second_quadrant_l3062_306279


namespace quadratic_real_roots_condition_l3062_306290

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, (a - 6) * x^2 - 8 * x + 9 = 0) ↔ (a ≤ 70 / 9 ∧ a ≠ 6) :=
by sorry

end quadratic_real_roots_condition_l3062_306290


namespace train_speed_conversion_l3062_306294

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The speed of the train in meters per second -/
def train_speed_mps : ℝ := 30.002399999999998

/-- Theorem stating that the train's speed in km/h is equal to 108.00863999999999 -/
theorem train_speed_conversion :
  train_speed_mps * mps_to_kmph = 108.00863999999999 := by
  sorry

end train_speed_conversion_l3062_306294


namespace least_sum_m_n_l3062_306281

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (m.val > 0 ∧ n.val > 0) ∧
  (Nat.gcd (m.val + n.val) 210 = 1) ∧
  (∃ (k : ℕ), m.val ^ m.val = k * (n.val ^ n.val)) ∧
  (¬ ∃ (l : ℕ), m.val = l * n.val) ∧
  (m.val + n.val = 407) ∧
  (∀ (p q : ℕ+), 
    (p.val > 0 ∧ q.val > 0) →
    (Nat.gcd (p.val + q.val) 210 = 1) →
    (∃ (k : ℕ), p.val ^ p.val = k * (q.val ^ q.val)) →
    (¬ ∃ (l : ℕ), p.val = l * q.val) →
    (p.val + q.val ≥ 407)) :=
by sorry

end least_sum_m_n_l3062_306281


namespace set_B_equals_expected_l3062_306205

def A : Set Int := {-3, -2, -1, 1, 2, 3, 4}

def f (a : Int) : Int := Int.natAbs a

def B : Set Int := f '' A

theorem set_B_equals_expected : B = {1, 2, 3, 4} := by
  sorry

end set_B_equals_expected_l3062_306205


namespace cos_20_cos_385_minus_cos_70_sin_155_l3062_306206

theorem cos_20_cos_385_minus_cos_70_sin_155 :
  Real.cos (20 * π / 180) * Real.cos (385 * π / 180) - 
  Real.cos (70 * π / 180) * Real.sin (155 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end cos_20_cos_385_minus_cos_70_sin_155_l3062_306206


namespace parallelogram_altitude_l3062_306286

/-- Represents a parallelogram ABCD with altitudes DE and DF -/
structure Parallelogram where
  -- Lengths of sides and segments
  DC : ℝ
  EB : ℝ
  DE : ℝ
  -- Condition that ABCD is a parallelogram
  is_parallelogram : True

/-- Theorem: In a parallelogram ABCD with given conditions, DF = 7 -/
theorem parallelogram_altitude (p : Parallelogram)
  (h1 : p.DC = 15)
  (h2 : p.EB = 5)
  (h3 : p.DE = 7) :
  ∃ DF : ℝ, DF = 7 :=
by sorry

end parallelogram_altitude_l3062_306286


namespace subset_implies_m_equals_three_l3062_306229

theorem subset_implies_m_equals_three (A B : Set ℕ) (m : ℕ) :
  A = {1, 3} →
  B = {1, 2, m} →
  A ⊆ B →
  m = 3 := by
sorry

end subset_implies_m_equals_three_l3062_306229


namespace only_two_consecutive_primes_l3062_306246

theorem only_two_consecutive_primes : ∀ p : ℕ, 
  (Nat.Prime p ∧ Nat.Prime (p + 1)) → p = 2 := by
  sorry

end only_two_consecutive_primes_l3062_306246


namespace factorization_of_2m_squared_minus_8_l3062_306252

theorem factorization_of_2m_squared_minus_8 (m : ℝ) : 2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := by
  sorry

end factorization_of_2m_squared_minus_8_l3062_306252


namespace retest_probability_l3062_306208

theorem retest_probability (total : ℕ) (p_physics : ℚ) (p_chemistry : ℚ) (p_biology : ℚ) :
  total = 50 →
  p_physics = 9 / 50 →
  p_chemistry = 1 / 5 →
  p_biology = 11 / 50 →
  let p_one_subject := p_physics + p_chemistry + p_biology
  let p_more_than_one := 1 - p_one_subject
  p_more_than_one = 2 / 5 := by
  sorry

#eval (2 : ℚ) / 5 -- This should output 0.4

end retest_probability_l3062_306208


namespace sock_pairs_count_l3062_306228

def white_socks : ℕ := 5
def brown_socks : ℕ := 4
def blue_socks : ℕ := 3

def different_color_pairs_with_blue : ℕ := (blue_socks * white_socks) + (blue_socks * brown_socks)

theorem sock_pairs_count : different_color_pairs_with_blue = 27 := by
  sorry

end sock_pairs_count_l3062_306228


namespace compare_M_and_N_range_of_m_l3062_306267

-- Problem 1
theorem compare_M_and_N : ∀ x : ℝ, 2 * x^2 + 1 > x^2 + 2*x - 1 := by sorry

-- Problem 2
theorem range_of_m : 
  (∀ m : ℝ, (∀ x : ℝ, 2*m ≤ x ∧ x ≤ m+1 → -1 ≤ x ∧ x ≤ 1) → -1/2 ≤ m ∧ m ≤ 0) := by sorry

end compare_M_and_N_range_of_m_l3062_306267


namespace percentage_of_value_in_quarters_l3062_306297

theorem percentage_of_value_in_quarters : 
  let num_dimes : ℕ := 40
  let num_quarters : ℕ := 30
  let num_nickels : ℕ := 10
  let value_dime : ℕ := 10
  let value_quarter : ℕ := 25
  let value_nickel : ℕ := 5
  let total_value := num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel
  let value_in_quarters := num_quarters * value_quarter
  (value_in_quarters : ℚ) / total_value * 100 = 62.5 := by
sorry

end percentage_of_value_in_quarters_l3062_306297


namespace max_dot_product_l3062_306227

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the center and left focus
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define the dot product of OF and OP
def dot_product (x y : ℝ) : ℝ := (x + 1) * x + y * y

-- Theorem statement
theorem max_dot_product :
  ∀ x y : ℝ, is_on_ellipse x y →
  ∀ x' y' : ℝ, is_on_ellipse x' y' →
  dot_product x y ≤ 6 :=
sorry

end max_dot_product_l3062_306227


namespace chess_tournament_players_l3062_306266

/-- Chess tournament with specific conditions -/
structure ChessTournament where
  n : ℕ
  total_score : ℕ
  two_player_score : ℕ
  avg_score_others : ℕ
  odd_players : Odd n
  two_player_score_eq : two_player_score = 16
  even_avg_score : Even avg_score_others
  total_score_eq : total_score = n * (n - 1)

/-- Theorem stating that under given conditions, the number of players is 9 -/
theorem chess_tournament_players (t : ChessTournament) : t.n = 9 := by
  sorry

end chess_tournament_players_l3062_306266


namespace shoes_outside_library_l3062_306269

/-- The number of shoes for a group of people, given the number of people and shoes per person. -/
def total_shoes (num_people : ℕ) (shoes_per_person : ℕ) : ℕ :=
  num_people * shoes_per_person

/-- Theorem: For a group of 10 people, where each person wears 2 shoes,
    the total number of shoes when everyone takes them off is 20. -/
theorem shoes_outside_library :
  total_shoes 10 2 = 20 := by
  sorry

end shoes_outside_library_l3062_306269


namespace min_gumballs_for_four_same_color_gumball_problem_solution_l3062_306249

/-- Represents the number of gumballs of each color -/
structure GumballCounts where
  green : Nat
  red : Nat
  white : Nat
  blue : Nat

/-- The minimum number of gumballs needed to ensure getting four of the same color -/
def minGumballsForFourSameColor (counts : GumballCounts) : Nat :=
  13

/-- Theorem stating that for the given gumball counts, 
    the minimum number of gumballs needed to ensure 
    getting four of the same color is 13 -/
theorem min_gumballs_for_four_same_color 
  (counts : GumballCounts) 
  (h1 : counts.green = 12) 
  (h2 : counts.red = 10) 
  (h3 : counts.white = 9) 
  (h4 : counts.blue = 11) : 
  minGumballsForFourSameColor counts = 13 := by
  sorry

/-- Main theorem that proves the result for the specific problem instance -/
theorem gumball_problem_solution : 
  ∃ (counts : GumballCounts), 
    counts.green = 12 ∧ 
    counts.red = 10 ∧ 
    counts.white = 9 ∧ 
    counts.blue = 11 ∧ 
    minGumballsForFourSameColor counts = 13 := by
  sorry

end min_gumballs_for_four_same_color_gumball_problem_solution_l3062_306249


namespace unit_circle_solutions_eq_parameterized_solutions_l3062_306212

noncomputable section

variable (F : Type*) [Field F]

/-- The set of solutions to x^2 + y^2 = 1 in a field F where 1 + 1 ≠ 0 -/
def UnitCircleSolutions (F : Type*) [Field F] (h : (1 : F) + 1 ≠ 0) : Set (F × F) :=
  {p : F × F | p.1^2 + p.2^2 = 1}

/-- The parameterized set of solutions -/
def ParameterizedSolutions (F : Type*) [Field F] : Set (F × F) :=
  {p : F × F | ∃ r : F, r^2 ≠ -1 ∧ 
    p = ((r^2 - 1) / (r^2 + 1), 2*r / (r^2 + 1))} ∪ {(1, 0)}

/-- Theorem stating that the solutions to x^2 + y^2 = 1 are exactly the parameterized solutions -/
theorem unit_circle_solutions_eq_parameterized_solutions 
  (h : (1 : F) + 1 ≠ 0) : 
  UnitCircleSolutions F h = ParameterizedSolutions F :=
by sorry

end

end unit_circle_solutions_eq_parameterized_solutions_l3062_306212


namespace packages_sold_correct_l3062_306222

/-- The number of packages of gaskets sold during a week -/
def packages_sold : ℕ := 66

/-- The price per package of gaskets -/
def price_per_package : ℚ := 20

/-- The discount factor for packages in excess of 10 -/
def discount_factor : ℚ := 4/5

/-- The total payment received for the gaskets -/
def total_payment : ℚ := 1096

/-- Calculates the total cost for the given number of packages -/
def total_cost (n : ℕ) : ℚ :=
  if n ≤ 10 then n * price_per_package
  else 10 * price_per_package + (n - 10) * (discount_factor * price_per_package)

/-- Theorem stating that the number of packages sold satisfies the given conditions -/
theorem packages_sold_correct : 
  total_cost packages_sold = total_payment := by sorry

end packages_sold_correct_l3062_306222


namespace total_cost_is_18_l3062_306244

-- Define the cost of a single soda
def soda_cost : ℝ := 1

-- Define the cost of a single soup
def soup_cost : ℝ := 3 * soda_cost

-- Define the cost of a sandwich
def sandwich_cost : ℝ := 3 * soup_cost

-- Define the total cost
def total_cost : ℝ := 3 * soda_cost + 2 * soup_cost + sandwich_cost

-- Theorem statement
theorem total_cost_is_18 : total_cost = 18 := by
  sorry

end total_cost_is_18_l3062_306244


namespace sum_divisible_by_addends_l3062_306219

theorem sum_divisible_by_addends : 
  ∃ (a b c : ℕ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a + b + c) % a = 0 ∧ 
    (a + b + c) % b = 0 ∧ 
    (a + b + c) % c = 0 :=
by sorry

end sum_divisible_by_addends_l3062_306219


namespace inequality_solution_set_l3062_306298

def inequality (a x : ℝ) : Prop := (a + 1) * x - 3 < x - 1

def solution_set (a : ℝ) : Set ℝ :=
  if a < 0 then {x | x < 2/a ∨ x > 1}
  else if a = 0 then {x | x > 1}
  else if 0 < a ∧ a < 2 then {x | 1 < x ∧ x < 2/a}
  else if a = 2 then ∅
  else {x | 2/a < x ∧ x < 1}

theorem inequality_solution_set (a : ℝ) :
  {x : ℝ | inequality a x} = solution_set a :=
  sorry

end inequality_solution_set_l3062_306298


namespace twentieth_meeting_at_D_l3062_306237

/-- Represents a meeting point in the pool lane -/
inductive MeetingPoint
| C
| D

/-- Represents an athlete swimming in the pool lane -/
structure Athlete where
  speed : ℝ
  speed_positive : speed > 0

/-- Represents a swimming scenario with two athletes -/
structure SwimmingScenario where
  athlete1 : Athlete
  athlete2 : Athlete
  different_speeds : athlete1.speed ≠ athlete2.speed
  first_meeting : MeetingPoint
  second_meeting : MeetingPoint
  first_meeting_is_C : first_meeting = MeetingPoint.C
  second_meeting_is_D : second_meeting = MeetingPoint.D

/-- The theorem stating that the 20th meeting occurs at point D -/
theorem twentieth_meeting_at_D (scenario : SwimmingScenario) :
  (fun n => if n % 2 = 0 then MeetingPoint.D else MeetingPoint.C) 20 = MeetingPoint.D :=
sorry

end twentieth_meeting_at_D_l3062_306237


namespace intersection_distance_sum_l3062_306235

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, -4)

-- Define the theorem
theorem intersection_distance_sum :
  ∃ (M N : ℝ × ℝ),
    curve_C M.1 M.2 ∧
    curve_C N.1 N.2 ∧
    line M.1 M.2 ∧
    line N.1 N.2 ∧
    M ≠ N ∧
    Real.sqrt ((M.1 - point_P.1)^2 + (M.2 - point_P.2)^2) +
    Real.sqrt ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2) =
    12 * Real.sqrt 2 :=
sorry

end intersection_distance_sum_l3062_306235


namespace sum_squared_distances_coinciding_centroids_l3062_306240

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  leg_length_pos : leg_length > 0

/-- The sum of squared distances between vertices of two triangles -/
def sum_squared_distances (et : EquilateralTriangle) (irt : IsoscelesRightTriangle) : ℝ := 
  3 * et.side_length^2 + 4 * irt.leg_length^2

theorem sum_squared_distances_coinciding_centroids 
  (et : EquilateralTriangle) 
  (irt : IsoscelesRightTriangle) :
  sum_squared_distances et irt = 3 * et.side_length^2 + 4 * irt.leg_length^2 := by
  sorry

end sum_squared_distances_coinciding_centroids_l3062_306240


namespace expression_value_l3062_306282

theorem expression_value : 
  let x : ℤ := 25
  let y : ℤ := 30
  let z : ℤ := 10
  (x - (y - z)) - ((x - y) - z) = 20 := by
  sorry

end expression_value_l3062_306282


namespace vet_donation_portion_is_about_third_l3062_306215

/-- Represents the adoption event scenario at an animal shelter --/
structure AdoptionEvent where
  dog_fee : ℕ
  cat_fee : ℕ
  dog_adoptions : ℕ
  cat_adoptions : ℕ
  vet_donation : ℕ

/-- Calculates the portion of fees donated by the vet --/
def donation_portion (event : AdoptionEvent) : ℚ :=
  let total_fees := event.dog_fee * event.dog_adoptions + event.cat_fee * event.cat_adoptions
  (event.vet_donation : ℚ) / total_fees

/-- Theorem stating that the portion of fees donated is approximately 33.33% --/
theorem vet_donation_portion_is_about_third (event : AdoptionEvent) 
    (h1 : event.dog_fee = 15)
    (h2 : event.cat_fee = 13)
    (h3 : event.dog_adoptions = 8)
    (h4 : event.cat_adoptions = 3)
    (h5 : event.vet_donation = 53) :
    ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |donation_portion event - 1/3| < ε := by
  sorry

#eval donation_portion { dog_fee := 15, cat_fee := 13, dog_adoptions := 8, cat_adoptions := 3, vet_donation := 53 }

end vet_donation_portion_is_about_third_l3062_306215


namespace sum_of_digits_problem_l3062_306216

def S (n : ℕ) : ℕ := sorry  -- Definition of S(n) as sum of digits

theorem sum_of_digits_problem (n : ℕ) (h : n + S n = 2009) : n = 1990 := by
  sorry

end sum_of_digits_problem_l3062_306216


namespace solve_for_a_l3062_306218

theorem solve_for_a (a : ℝ) (h : 2 * 2^2 * a = 2^6) : a = 8 := by
  sorry

end solve_for_a_l3062_306218


namespace segment_length_l3062_306224

/-- Given two points P and Q on a line segment AB, where:
    - P and Q are on the same side of the midpoint of AB
    - P divides AB in the ratio 3:5
    - Q divides AB in the ratio 4:5
    - PQ = 3
    Prove that the length of AB is 43.2 -/
theorem segment_length (A B P Q : Real) (h1 : P ∈ Set.Icc A B) (h2 : Q ∈ Set.Icc A B)
    (h3 : (P - A) / (B - A) = 3 / 8) (h4 : (Q - A) / (B - A) = 4 / 9) (h5 : Q - P = 3) :
    B - A = 43.2 := by
  sorry

end segment_length_l3062_306224


namespace two_distinct_values_of_T_l3062_306265

theorem two_distinct_values_of_T (n : ℤ) : 
  let i : ℂ := Complex.I
  let T : ℂ := i^(2*n) + i^(-2*n) + Real.cos (n * Real.pi)
  ∃ (a b : ℂ), ∀ (m : ℤ), 
    (let T_m : ℂ := i^(2*m) + i^(-2*m) + Real.cos (m * Real.pi)
     T_m = a ∨ T_m = b) ∧ a ≠ b :=
sorry

end two_distinct_values_of_T_l3062_306265


namespace visitor_growth_l3062_306261

/-- Represents the growth of visitors at a tourist attraction from January to March. -/
theorem visitor_growth (initial_visitors final_visitors : ℕ) (x : ℝ) :
  initial_visitors = 60000 →
  final_visitors = 150000 →
  (initial_visitors : ℝ) / 10000 * (1 + x)^2 = (final_visitors : ℝ) / 10000 →
  6 * (1 + x)^2 = 15 := by
  sorry

#check visitor_growth

end visitor_growth_l3062_306261


namespace complex_number_in_second_quadrant_l3062_306243

def i : ℂ := Complex.I

theorem complex_number_in_second_quadrant :
  let z : ℂ := (1 + i) * i
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_second_quadrant_l3062_306243


namespace train_length_l3062_306278

/-- Given a train traveling at constant speed through a tunnel, this theorem
    proves the length of the train based on the given conditions. -/
theorem train_length
  (tunnel_length : ℝ)
  (total_time : ℝ)
  (light_time : ℝ)
  (h1 : tunnel_length = 310)
  (h2 : total_time = 18)
  (h3 : light_time = 8)
  (h4 : total_time > 0)
  (h5 : light_time > 0)
  (h6 : light_time < total_time) :
  ∃ (train_length : ℝ),
    train_length = 248 ∧
    (tunnel_length + train_length) / total_time = train_length / light_time :=
by sorry


end train_length_l3062_306278


namespace sixth_member_income_l3062_306202

theorem sixth_member_income
  (family_size : ℕ)
  (average_income : ℕ)
  (income1 income2 income3 income4 income5 : ℕ)
  (h1 : family_size = 6)
  (h2 : average_income = 12000)
  (h3 : income1 = 11000)
  (h4 : income2 = 15000)
  (h5 : income3 = 10000)
  (h6 : income4 = 9000)
  (h7 : income5 = 13000) :
  average_income * family_size - (income1 + income2 + income3 + income4 + income5) = 14000 := by
  sorry

end sixth_member_income_l3062_306202


namespace coin_value_difference_max_value_achievable_min_value_achievable_l3062_306204

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime

/-- The value of a coin in cents --/
def coinValue : CoinType → Nat
  | CoinType.Penny => 1
  | CoinType.Nickel => 5
  | CoinType.Dime => 10

/-- A distribution of coins --/
structure CoinDistribution where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  total_coins : pennies + nickels + dimes = 3030
  at_least_one : pennies ≥ 1 ∧ nickels ≥ 1 ∧ dimes ≥ 1

/-- The total value of a coin distribution in cents --/
def totalValue (d : CoinDistribution) : Nat :=
  d.pennies * coinValue CoinType.Penny +
  d.nickels * coinValue CoinType.Nickel +
  d.dimes * coinValue CoinType.Dime

/-- The maximum possible value for any valid coin distribution --/
def maxValue : Nat := 30286

/-- The minimum possible value for any valid coin distribution --/
def minValue : Nat := 3043

theorem coin_value_difference :
  maxValue - minValue = 27243 :=
by
  sorry

theorem max_value_achievable (d : CoinDistribution) :
  totalValue d ≤ maxValue :=
by
  sorry

theorem min_value_achievable (d : CoinDistribution) :
  totalValue d ≥ minValue :=
by
  sorry

end coin_value_difference_max_value_achievable_min_value_achievable_l3062_306204


namespace bobby_toy_cars_increase_l3062_306207

/-- The annual percentage increase in Bobby's toy cars -/
def annual_increase : ℝ := 0.5

theorem bobby_toy_cars_increase :
  let initial_cars : ℝ := 16
  let years : ℕ := 3
  let final_cars : ℝ := 54
  initial_cars * (1 + annual_increase) ^ years = final_cars :=
by sorry

end bobby_toy_cars_increase_l3062_306207


namespace houses_in_block_l3062_306248

theorem houses_in_block (junk_mails_per_house : ℕ) (total_junk_mails_per_block : ℕ) 
  (h1 : junk_mails_per_house = 2) 
  (h2 : total_junk_mails_per_block = 14) : 
  total_junk_mails_per_block / junk_mails_per_house = 7 := by
  sorry

end houses_in_block_l3062_306248


namespace expected_black_pairs_standard_deck_l3062_306225

/-- A standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (black_cards : ℕ)
  (red_cards : ℕ)
  (h_total : total_cards = black_cards + red_cards)

/-- A standard 104-card deck -/
def standard_deck : Deck :=
  { total_cards := 104,
    black_cards := 52,
    red_cards := 52,
    h_total := rfl }

/-- The expected number of pairs of adjacent black cards when dealt in a line -/
def expected_black_pairs (d : Deck) : ℚ :=
  (d.black_cards - 1 : ℚ) * (d.black_cards - 1) / (d.total_cards - 1)

theorem expected_black_pairs_standard_deck :
  expected_black_pairs standard_deck = 2601 / 103 :=
sorry

end expected_black_pairs_standard_deck_l3062_306225


namespace coins_fit_in_new_box_l3062_306251

/-- Represents a rectangular box -/
structure Box where
  width : ℝ
  height : ℝ

/-- Represents a collection of coins -/
structure CoinCollection where
  maxDiameter : ℝ

/-- Check if a coin collection can fit in a box -/
def canFitIn (coins : CoinCollection) (box : Box) : Prop :=
  box.width * box.height ≥ 0 -- This is a simplification, as we don't know the exact arrangement

/-- Theorem: If coins fit in the original box, they can fit in the new box -/
theorem coins_fit_in_new_box 
  (coins : CoinCollection)
  (originalBox : Box)
  (newBox : Box)
  (h1 : coins.maxDiameter ≤ 10)
  (h2 : originalBox.width = 30 ∧ originalBox.height = 70)
  (h3 : newBox.width = 40 ∧ newBox.height = 60)
  (h4 : canFitIn coins originalBox) :
  canFitIn coins newBox :=
by
  sorry

#check coins_fit_in_new_box

end coins_fit_in_new_box_l3062_306251


namespace minimum_bailing_rate_l3062_306256

/-- Represents the problem of determining the minimum bailing rate for a leaking boat --/
theorem minimum_bailing_rate 
  (distance_to_shore : ℝ) 
  (water_intake_rate : ℝ) 
  (max_water_capacity : ℝ) 
  (rowing_speed : ℝ) 
  (h1 : distance_to_shore = 2) 
  (h2 : water_intake_rate = 15) 
  (h3 : max_water_capacity = 50) 
  (h4 : rowing_speed = 3) : 
  ∃ (min_bailing_rate : ℝ), 
    13 < min_bailing_rate ∧ 
    min_bailing_rate ≤ 14 ∧ 
    (distance_to_shore / rowing_speed) * water_intake_rate - 
      (distance_to_shore / rowing_speed) * min_bailing_rate ≤ max_water_capacity :=
by sorry

end minimum_bailing_rate_l3062_306256


namespace diamond_three_four_l3062_306200

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem diamond_three_four : diamond 3 4 = 36 := by
  sorry

end diamond_three_four_l3062_306200


namespace rectangular_garden_perimeter_l3062_306255

theorem rectangular_garden_perimeter 
  (x y : ℝ) 
  (diagonal_squared : x^2 + y^2 = 900)
  (area : x * y = 240) : 
  2 * (x + y) = 4 * Real.sqrt 345 := by
sorry

end rectangular_garden_perimeter_l3062_306255


namespace pascal_triangle_row20_symmetry_l3062_306262

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of elements in row n of Pascal's triangle -/
def rowLength (n : ℕ) : ℕ := n + 1

theorem pascal_triangle_row20_symmetry :
  let n := 20
  let k := 5
  let row_length := rowLength n
  binomial n (k - 1) = binomial n (row_length - k) ∧
  binomial n (k - 1) = 4845 := by
  sorry

end pascal_triangle_row20_symmetry_l3062_306262


namespace parallel_vectors_imply_k_value_l3062_306230

/-- Given vectors a, b, and c in R², prove that if (a - c) is parallel to b, then k = 5 -/
theorem parallel_vectors_imply_k_value (a b c : ℝ × ℝ) (k : ℝ) :
  a = (3, 1) →
  b = (1, 3) →
  c = (k, 7) →
  (∃ (t : ℝ), a - c = t • b) →
  k = 5 := by
  sorry

end parallel_vectors_imply_k_value_l3062_306230


namespace y_investment_is_75000_l3062_306280

/-- Represents the investment and profit scenario of a business --/
structure BusinessScenario where
  x_investment : ℕ
  z_investment : ℕ
  z_join_time : ℕ
  z_profit_share : ℕ
  total_profit : ℕ
  total_duration : ℕ

/-- Calculates Y's investment given a business scenario --/
def calculate_y_investment (scenario : BusinessScenario) : ℕ :=
  sorry

/-- Theorem stating that Y's investment is 75000 for the given scenario --/
theorem y_investment_is_75000 (scenario : BusinessScenario) 
  (h1 : scenario.x_investment = 36000)
  (h2 : scenario.z_investment = 48000)
  (h3 : scenario.z_join_time = 4)
  (h4 : scenario.z_profit_share = 4064)
  (h5 : scenario.total_profit = 13970)
  (h6 : scenario.total_duration = 12) :
  calculate_y_investment scenario = 75000 :=
sorry

end y_investment_is_75000_l3062_306280


namespace chantal_profit_l3062_306285

def sweater_profit (balls_per_sweater : ℕ) (yarn_cost : ℕ) (sell_price : ℕ) (num_sweaters : ℕ) : ℕ :=
  let total_balls := balls_per_sweater * num_sweaters
  let total_cost := total_balls * yarn_cost
  let total_revenue := sell_price * num_sweaters
  total_revenue - total_cost

theorem chantal_profit :
  sweater_profit 4 6 35 28 = 308 := by
  sorry

end chantal_profit_l3062_306285


namespace max_visible_cubes_eq_400_l3062_306234

/-- The size of the cube's edge -/
def cube_size : ℕ := 12

/-- The number of unit cubes visible on a single face -/
def face_count : ℕ := cube_size ^ 2

/-- The number of unit cubes overcounted on each edge -/
def edge_overcount : ℕ := cube_size - 1

/-- The maximum number of visible unit cubes from a single point -/
def max_visible_cubes : ℕ := 3 * face_count - 3 * edge_overcount + 1

theorem max_visible_cubes_eq_400 : max_visible_cubes = 400 := by
  sorry

end max_visible_cubes_eq_400_l3062_306234


namespace josanna_minimum_score_l3062_306238

def current_scores : List ℕ := [75, 85, 65, 95, 70]
def increase_amount : ℕ := 10

def minimum_next_score (scores : List ℕ) (increase : ℕ) : ℕ :=
  let current_sum := scores.sum
  let current_count := scores.length
  let current_avg := current_sum / current_count
  let target_avg := current_avg + increase
  let total_count := current_count + 1
  target_avg * total_count - current_sum

theorem josanna_minimum_score :
  minimum_next_score current_scores increase_amount = 138 := by
  sorry

end josanna_minimum_score_l3062_306238


namespace distance_to_directrix_l3062_306201

/-- A parabola C is defined by the equation y² = 2px. -/
structure Parabola where
  p : ℝ

/-- A point on a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The theorem states that for a parabola C where point A(1, √5) lies on it,
    the distance from A to the directrix of C is 9/4. -/
theorem distance_to_directrix (C : Parabola) (A : Point) :
  A.x = 1 →
  A.y = Real.sqrt 5 →
  A.y ^ 2 = 2 * C.p * A.x →
  (A.x + C.p / 2) = 9 / 4 := by
  sorry

end distance_to_directrix_l3062_306201


namespace carl_accident_cost_l3062_306276

/-- Carl's car accident cost calculation -/
theorem carl_accident_cost (property_damage medical_bills : ℕ) 
  (h1 : property_damage = 40000)
  (h2 : medical_bills = 70000)
  (carl_percentage : ℚ)
  (h3 : carl_percentage = 1/5) :
  carl_percentage * (property_damage + medical_bills : ℚ) = 22000 := by
sorry

end carl_accident_cost_l3062_306276


namespace agent_007_encryption_possible_l3062_306223

theorem agent_007_encryption_possible : ∃ (m n : ℕ), (1 : ℚ) / m + (1 : ℚ) / n = (7 : ℚ) / 100 := by
  sorry

end agent_007_encryption_possible_l3062_306223


namespace function_inequality_l3062_306226

open Set
open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h1 : ∀ x, HasDerivAt f (f' x) x) 
  (h2 : ∀ x, f x + f' x > 2) (h3 : f 0 = 2021) :
  ∀ x, f x > 2 + 2019 / exp x ↔ x > 0 := by
sorry

end function_inequality_l3062_306226


namespace darwin_money_left_l3062_306277

theorem darwin_money_left (initial_amount : ℝ) (gas_fraction : ℝ) (food_fraction : ℝ) : 
  initial_amount = 600 →
  gas_fraction = 1/3 →
  food_fraction = 1/4 →
  initial_amount - (gas_fraction * initial_amount) - (food_fraction * (initial_amount - gas_fraction * initial_amount)) = 300 := by
sorry

end darwin_money_left_l3062_306277


namespace total_eggs_l3062_306213

def eggs_club_house : ℕ := 12
def eggs_park : ℕ := 5
def eggs_town_hall : ℕ := 3

theorem total_eggs : eggs_club_house + eggs_park + eggs_town_hall = 20 := by
  sorry

end total_eggs_l3062_306213


namespace right_triangle_hypotenuse_l3062_306299

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 :=
by
  sorry

end right_triangle_hypotenuse_l3062_306299


namespace winner_ate_15_ounces_l3062_306284

-- Define the weights of each ravioli type
def meat_ravioli_weight : ℝ := 1.5
def pumpkin_ravioli_weight : ℝ := 1.25
def cheese_ravioli_weight : ℝ := 1

-- Define the quantities eaten by Javier
def javier_meat_count : ℕ := 5
def javier_pumpkin_count : ℕ := 2
def javier_cheese_count : ℕ := 4

-- Define the quantity eaten by Javier's brother
def brother_pumpkin_count : ℕ := 12

-- Calculate total weight eaten by Javier
def javier_total_weight : ℝ := 
  meat_ravioli_weight * javier_meat_count +
  pumpkin_ravioli_weight * javier_pumpkin_count +
  cheese_ravioli_weight * javier_cheese_count

-- Calculate total weight eaten by Javier's brother
def brother_total_weight : ℝ := pumpkin_ravioli_weight * brother_pumpkin_count

-- Theorem: The winner ate 15 ounces of ravioli
theorem winner_ate_15_ounces : 
  max javier_total_weight brother_total_weight = 15 := by sorry

end winner_ate_15_ounces_l3062_306284


namespace negation_of_zero_product_implication_l3062_306274

theorem negation_of_zero_product_implication :
  (∀ x y : ℝ, xy = 0 → x = 0 ∨ y = 0) ↔
  (∀ x y : ℝ, xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) :=
by sorry

end negation_of_zero_product_implication_l3062_306274


namespace minimum_teacher_time_l3062_306245

def student_time (explanation_time : ℕ) (completion_time : ℕ) : ℕ :=
  explanation_time + completion_time

theorem minimum_teacher_time 
  (student_A : ℕ) 
  (student_B : ℕ) 
  (student_C : ℕ) 
  (explanation_time : ℕ) 
  (h1 : student_A = student_time explanation_time 13)
  (h2 : student_B = student_time explanation_time 10)
  (h3 : student_C = student_time explanation_time 16)
  (h4 : explanation_time = 3) :
  3 * explanation_time + 2 * student_B + student_A + student_C = 90 :=
sorry

end minimum_teacher_time_l3062_306245


namespace tangent_circle_equation_l3062_306232

/-- A circle passing through two points and tangent to a line -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_origin : center.1^2 + center.2^2 = radius^2
  passes_point : (center.1 - 4)^2 + center.2^2 = radius^2
  tangent_to_line : |center.2 - 1| = radius

/-- The equation of the circle -/
def circle_equation (c : TangentCircle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem tangent_circle_equation :
  ∀ (c : TangentCircle),
    ∀ (x y : ℝ),
      circle_equation c x y ↔ (x - 2)^2 + (y + 3/2)^2 = 25/4 :=
by sorry

end tangent_circle_equation_l3062_306232


namespace sharon_drive_distance_l3062_306263

theorem sharon_drive_distance :
  let usual_time : ℝ := 180
  let snowstorm_time : ℝ := 300
  let speed_decrease : ℝ := 30
  let distance : ℝ := 157.5
  let usual_speed : ℝ := distance / usual_time
  let snowstorm_speed : ℝ := usual_speed - speed_decrease / 60
  (distance / 2) / usual_speed + (distance / 2) / snowstorm_speed = snowstorm_time :=
by sorry

end sharon_drive_distance_l3062_306263


namespace range_of_m_l3062_306258

theorem range_of_m (x m : ℝ) : 
  (∀ x, (1 < x ∧ x < m - 2) → (1 < x ∧ x < 4)) ∧ 
  (∃ x, (1 < x ∧ x < 4) ∧ ¬(1 < x ∧ x < m - 2)) → 
  m ∈ Set.Ioi 6 := by
sorry

end range_of_m_l3062_306258
