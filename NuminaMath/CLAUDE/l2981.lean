import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_binomial_ratio_l2981_298182

theorem consecutive_binomial_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k+1) : ℚ) = 1/3 ∧
  (n.choose (k+1) : ℚ) / (n.choose (k+2) : ℚ) = 3/5 →
  n + k = 8 := by
sorry

end NUMINAMATH_CALUDE_consecutive_binomial_ratio_l2981_298182


namespace NUMINAMATH_CALUDE_john_miles_conversion_l2981_298178

/-- Converts a base-7 number represented as a list of digits to its base-10 equivalent -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base-7 representation of the number of miles John cycled -/
def johnMilesBase7 : List Nat := [6, 1, 5, 3]

theorem john_miles_conversion :
  base7ToBase10 johnMilesBase7 = 1287 := by
  sorry

end NUMINAMATH_CALUDE_john_miles_conversion_l2981_298178


namespace NUMINAMATH_CALUDE_expression_equality_l2981_298139

theorem expression_equality : (3^1003 + 5^1003)^2 - (3^1003 - 5^1003)^2 = 4 * 15^1003 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2981_298139


namespace NUMINAMATH_CALUDE_total_dress_designs_l2981_298173

/-- The number of fabric colors available -/
def num_colors : ℕ := 5

/-- The number of patterns available -/
def num_patterns : ℕ := 4

/-- The number of sleeve length options -/
def num_sleeve_lengths : ℕ := 2

/-- Each dress design requires exactly one color, one pattern, and one sleeve length -/
axiom dress_design_requirement : True

/-- The total number of different dress designs possible -/
def total_designs : ℕ := num_colors * num_patterns * num_sleeve_lengths

/-- Theorem stating that the total number of different dress designs is 40 -/
theorem total_dress_designs : total_designs = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l2981_298173


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l2981_298101

/-- Given a geometric sequence with first term a₁ = 1, 
    the minimum value of 3a₂ + 7a₃ is -27/28 -/
theorem min_value_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 1 = 1 →                     -- first term is 1
  ∃ m : ℝ, m = -27/28 ∧ ∀ r : ℝ, 3 * (a 2) + 7 * (a 3) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l2981_298101


namespace NUMINAMATH_CALUDE_polynomial_identity_l2981_298112

/-- 
Given a natural number n, define a bivariate polynomial P(x, y) 
that satisfies P(u+v, w) + P(v+w, u) + P(w+u, v) = 0 for all u, v, w.
-/
theorem polynomial_identity (n : ℕ) :
  ∃ P : ℝ → ℝ → ℝ, 
    (∀ x y : ℝ, P x y = (x + y)^(n-1) * (x - 2*y)) ∧
    (∀ u v w : ℝ, P (u+v) w + P (v+w) u + P (w+u) v = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2981_298112


namespace NUMINAMATH_CALUDE_sum_squares_equals_two_l2981_298164

theorem sum_squares_equals_two (x y z : ℝ) 
  (sum_eq : x + y = 2) 
  (product_eq : x * y = z^2 + 1) : 
  x^2 + y^2 + z^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_squares_equals_two_l2981_298164


namespace NUMINAMATH_CALUDE_danivan_drugstore_inventory_l2981_298130

/-- Calculates the final inventory of hand sanitizer bottles at Danivan Drugstore after a week -/
def final_inventory (initial_inventory : ℕ) (monday_sales : ℕ) (tuesday_sales : ℕ) (daily_sales_wed_to_sun : ℕ) (saturday_delivery : ℕ) : ℕ :=
  initial_inventory - (monday_sales + tuesday_sales + 5 * daily_sales_wed_to_sun) + saturday_delivery

/-- Theorem stating that the final inventory is 1555 bottles given the specific conditions -/
theorem danivan_drugstore_inventory : 
  final_inventory 4500 2445 900 50 650 = 1555 := by
  sorry

end NUMINAMATH_CALUDE_danivan_drugstore_inventory_l2981_298130


namespace NUMINAMATH_CALUDE_train_speed_calculation_train_speed_result_l2981_298141

/-- Calculates the speed of a train given its length, the time it takes to pass a walking man, and the man's speed. -/
theorem train_speed_calculation (train_length : ℝ) (passing_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_ms := relative_speed + man_speed_ms
  let train_speed_kmh := train_speed_ms * (3600 / 1000)
  train_speed_kmh

/-- The speed of the train is approximately 63.0036 km/hr given the specified conditions. -/
theorem train_speed_result :
  ∃ ε > 0, |train_speed_calculation 900 53.99568034557235 3 - 63.0036| < ε :=
sorry

end NUMINAMATH_CALUDE_train_speed_calculation_train_speed_result_l2981_298141


namespace NUMINAMATH_CALUDE_wheat_purchase_proof_l2981_298189

/-- The cost of wheat in cents per pound -/
def wheat_cost : ℚ := 72

/-- The cost of oats in cents per pound -/
def oats_cost : ℚ := 36

/-- The total amount of wheat and oats bought in pounds -/
def total_amount : ℚ := 30

/-- The total amount spent in cents -/
def total_spent : ℚ := 1620

/-- The amount of wheat bought in pounds -/
def wheat_amount : ℚ := 15

theorem wheat_purchase_proof :
  ∃ (oats_amount : ℚ),
    wheat_amount + oats_amount = total_amount ∧
    wheat_cost * wheat_amount + oats_cost * oats_amount = total_spent :=
by sorry

end NUMINAMATH_CALUDE_wheat_purchase_proof_l2981_298189


namespace NUMINAMATH_CALUDE_product_of_terms_l2981_298176

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b₁ r : ℝ), ∀ n, b n = b₁ * r^(n - 1)

/-- Main theorem -/
theorem product_of_terms (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n, a n ≠ 0) →
  2 * (a 2) - (a 7)^2 + 2 * (a 12) = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 3 * b 11 = 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_terms_l2981_298176


namespace NUMINAMATH_CALUDE_smallest_gcd_yz_l2981_298123

theorem smallest_gcd_yz (x y z : ℕ+) (h1 : Nat.gcd x.val y.val = 224) (h2 : Nat.gcd x.val z.val = 546) :
  ∃ (y' z' : ℕ+), Nat.gcd y'.val z'.val = 14 ∧ 
  (∀ (a b : ℕ+), Nat.gcd x.val a.val = 224 → Nat.gcd x.val b.val = 546 → 
    Nat.gcd a.val b.val ≥ 14) :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_yz_l2981_298123


namespace NUMINAMATH_CALUDE_polynomial_coefficient_values_l2981_298108

theorem polynomial_coefficient_values (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^3 * (x + 2)^2 = x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅) →
  a₄ = 16 ∧ a₅ = 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_values_l2981_298108


namespace NUMINAMATH_CALUDE_triangle_height_l2981_298121

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 4.5 → area = 13.5 → area = (base * height) / 2 → height = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l2981_298121


namespace NUMINAMATH_CALUDE_lisa_spoon_count_l2981_298129

theorem lisa_spoon_count :
  let num_children : ℕ := 6
  let baby_spoons_per_child : ℕ := 4
  let decorative_spoons : ℕ := 4
  let large_spoons : ℕ := 20
  let dessert_spoons : ℕ := 10
  let teaspoons : ℕ := 25
  
  let total_baby_spoons := num_children * baby_spoons_per_child
  let total_special_spoons := total_baby_spoons + decorative_spoons
  let total_new_spoons := large_spoons + dessert_spoons + teaspoons
  let total_spoons := total_special_spoons + total_new_spoons

  total_spoons = 83 := by
sorry

end NUMINAMATH_CALUDE_lisa_spoon_count_l2981_298129


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l2981_298146

theorem sine_cosine_inequality (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ c < 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l2981_298146


namespace NUMINAMATH_CALUDE_smallest_three_digit_product_l2981_298134

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

theorem smallest_three_digit_product (n : ℕ) (a b : ℕ) :
  is_three_digit n →
  is_prime a →
  is_prime b →
  is_prime (10*a + b) →
  a < 10 →
  b < 10 →
  n = a * b * (10*a + b) →
  (∀ m : ℕ, is_three_digit m →
    (∃ x y : ℕ, is_prime x ∧ is_prime y ∧ is_prime (10*x + y) ∧
      x < 10 ∧ y < 10 ∧ m = x * y * (10*x + y)) →
    n ≤ m) →
  n = 138 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_product_l2981_298134


namespace NUMINAMATH_CALUDE_total_cars_is_43_l2981_298106

/-- The number of cars owned by each person -/
structure CarOwnership where
  cathy : ℕ
  lindsey : ℕ
  carol : ℕ
  susan : ℕ
  erica : ℕ

/-- Conditions for car ownership -/
def validCarOwnership (c : CarOwnership) : Prop :=
  c.lindsey = c.cathy + 4 ∧
  c.susan = c.carol - 2 ∧
  c.carol = 2 * c.cathy ∧
  c.erica = c.lindsey + (c.lindsey / 4) ∧
  c.cathy = 5

/-- The total number of cars owned by all people -/
def totalCars (c : CarOwnership) : ℕ :=
  c.cathy + c.lindsey + c.carol + c.susan + c.erica

/-- Theorem stating that the total number of cars is 43 -/
theorem total_cars_is_43 (c : CarOwnership) (h : validCarOwnership c) : totalCars c = 43 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_is_43_l2981_298106


namespace NUMINAMATH_CALUDE_emma_bank_balance_emma_final_balance_l2981_298175

theorem emma_bank_balance (initial_balance : ℝ) (shoe_percentage : ℝ) 
  (tuesday_deposit_percentage : ℝ) (wednesday_deposit_percentage : ℝ) 
  (final_withdrawal_percentage : ℝ) : ℝ :=
  let shoe_cost := initial_balance * shoe_percentage
  let monday_balance := initial_balance - shoe_cost
  let tuesday_deposit := shoe_cost * tuesday_deposit_percentage
  let tuesday_balance := monday_balance + tuesday_deposit
  let wednesday_deposit := shoe_cost * wednesday_deposit_percentage
  let wednesday_balance := tuesday_balance + wednesday_deposit
  let final_withdrawal := wednesday_balance * final_withdrawal_percentage
  let final_balance := wednesday_balance - final_withdrawal
  final_balance
  
theorem emma_final_balance : 
  emma_bank_balance 1200 0.08 0.25 1.5 0.05 = 1208.40 := by
  sorry

end NUMINAMATH_CALUDE_emma_bank_balance_emma_final_balance_l2981_298175


namespace NUMINAMATH_CALUDE_sequence_periodicity_l2981_298148

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1)| - a n

theorem sequence_periodicity (a : ℕ → ℝ) (h : sequence_property a) :
  ∃ m₀ : ℕ, ∀ m ≥ m₀, a (m + 9) = a m :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l2981_298148


namespace NUMINAMATH_CALUDE_ellipse_and_line_equations_l2981_298125

-- Define the ellipse G
def ellipse_G (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = Real.sqrt 6 / 3

-- Define the right focus
def right_focus (x y : ℝ) : Prop := x = 2 * Real.sqrt 2 ∧ y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := ∃ (m : ℝ), y = x + m

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_G A.1 A.2 ∧ ellipse_G B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

-- Define the isosceles triangle
def isosceles_triangle (A B : ℝ × ℝ) : Prop :=
  ∃ (P : ℝ × ℝ), P = (-3, 2) ∧
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2

-- Theorem statement
theorem ellipse_and_line_equations :
  ∀ (A B : ℝ × ℝ) (e : ℝ),
  ellipse_G A.1 A.2 ∧ ellipse_G B.1 B.2 ∧
  eccentricity e ∧
  right_focus (2 * Real.sqrt 2) 0 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  intersection_points A B ∧
  isosceles_triangle A B →
  (∀ (x y : ℝ), ellipse_G x y ↔ x^2 / 12 + y^2 / 4 = 1) ∧
  (∀ (x y : ℝ), line_l x y ↔ x - y + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_equations_l2981_298125


namespace NUMINAMATH_CALUDE_candidate_b_votes_l2981_298147

/-- Proves that candidate B received 4560 valid votes given the election conditions -/
theorem candidate_b_votes (total_eligible : Nat) (abstention_rate : Real) (invalid_vote_rate : Real) 
  (c_vote_percentage : Real) (a_vote_reduction : Real) 
  (h1 : total_eligible = 12000)
  (h2 : abstention_rate = 0.1)
  (h3 : invalid_vote_rate = 0.2)
  (h4 : c_vote_percentage = 0.05)
  (h5 : a_vote_reduction = 0.2) : 
  ∃ (b_votes : Nat), b_votes = 4560 := by
  sorry

end NUMINAMATH_CALUDE_candidate_b_votes_l2981_298147


namespace NUMINAMATH_CALUDE_joined_right_triangles_square_areas_l2981_298107

theorem joined_right_triangles_square_areas 
  (AB BC CD : ℝ) 
  (h_AB : AB^2 = 49) 
  (h_BC : BC^2 = 25) 
  (h_CD : CD^2 = 64) 
  (h_ABC_right : AB^2 + BC^2 = AC^2) 
  (h_ACD_right : CD^2 + AD^2 = AC^2) : 
  AD^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_joined_right_triangles_square_areas_l2981_298107


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2981_298118

theorem p_sufficient_not_necessary_for_q :
  ∃ (x y : ℝ), 
    (((x - 2) * (y - 5) ≠ 0) → (x ≠ 2 ∨ y ≠ 5)) ∧
    ¬(((x ≠ 2 ∨ y ≠ 5) → ((x - 2) * (y - 5) ≠ 0))) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2981_298118


namespace NUMINAMATH_CALUDE_product_expansion_l2981_298128

theorem product_expansion (x : ℝ) : 
  (2 * x^2 - 3 * x + 4) * (2 * x^2 + 3 * x + 4) = 4 * x^4 + 7 * x^2 + 16 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2981_298128


namespace NUMINAMATH_CALUDE_smallest_unstuck_perimeter_l2981_298161

/-- A rectangle inscribed in a larger rectangle. -/
structure InscribedRectangle where
  outer_width : ℝ
  outer_height : ℝ
  inner_width : ℝ
  inner_height : ℝ
  is_inscribed : inner_width ≤ outer_width ∧ inner_height ≤ outer_height

/-- An unstuck inscribed rectangle can be rotated slightly within the larger rectangle. -/
def is_unstuck (r : InscribedRectangle) : Prop := sorry

/-- The perimeter of a rectangle. -/
def perimeter (width height : ℝ) : ℝ := 2 * (width + height)

/-- The theorem to be proved. -/
theorem smallest_unstuck_perimeter :
  ∃ (r : InscribedRectangle),
    r.outer_width = 8 ∧
    r.outer_height = 6 ∧
    is_unstuck r ∧
    (∀ (s : InscribedRectangle),
      s.outer_width = 8 ∧
      s.outer_height = 6 ∧
      is_unstuck s →
      perimeter r.inner_width r.inner_height ≤ perimeter s.inner_width s.inner_height) ∧
    perimeter r.inner_width r.inner_height = Real.sqrt 448 := by sorry

end NUMINAMATH_CALUDE_smallest_unstuck_perimeter_l2981_298161


namespace NUMINAMATH_CALUDE_saras_cake_price_l2981_298122

/-- Sara's cake selling problem -/
theorem saras_cake_price (cakes_per_day : ℕ) (working_days : ℕ) (weeks : ℕ) (total_revenue : ℕ) :
  cakes_per_day = 4 →
  working_days = 5 →
  weeks = 4 →
  total_revenue = 640 →
  total_revenue / (cakes_per_day * working_days * weeks) = 8 := by
  sorry

#check saras_cake_price

end NUMINAMATH_CALUDE_saras_cake_price_l2981_298122


namespace NUMINAMATH_CALUDE_leftover_value_is_650_l2981_298157

/-- Represents the number of coins in a roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents a person's coin collection --/
structure CoinCollection where
  quarters : Nat
  dimes : Nat

/-- Calculates the value of coins in dollars --/
def coinValue (quarters dimes : Nat) : Rat :=
  (quarters * 25 + dimes * 10) / 100

/-- Calculates the number and value of leftover coins --/
def leftoverCoins (emily jack : CoinCollection) (roll : RollSize) : Rat :=
  let totalQuarters := emily.quarters + jack.quarters
  let totalDimes := emily.dimes + jack.dimes
  let leftoverQuarters := totalQuarters % roll.quarters
  let leftoverDimes := totalDimes % roll.dimes
  coinValue leftoverQuarters leftoverDimes

/-- The main theorem --/
theorem leftover_value_is_650 :
  let roll : RollSize := { quarters := 45, dimes := 60 }
  let emily : CoinCollection := { quarters := 105, dimes := 215 }
  let jack : CoinCollection := { quarters := 140, dimes := 340 }
  leftoverCoins emily jack roll = 13/2 := by sorry

end NUMINAMATH_CALUDE_leftover_value_is_650_l2981_298157


namespace NUMINAMATH_CALUDE_money_division_l2981_298158

theorem money_division (total : ℕ) (p q r : ℕ) (h1 : p + q + r = total) (h2 : p = 3 * (total / 22)) (h3 : q = 7 * (total / 22)) (h4 : r = 12 * (total / 22)) (h5 : r - q = 5500) : q - p = 4400 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l2981_298158


namespace NUMINAMATH_CALUDE_jerry_gabriel_toy_difference_l2981_298184

theorem jerry_gabriel_toy_difference (jerry gabriel jaxon : ℕ) 
  (h1 : jerry > gabriel)
  (h2 : gabriel = 2 * jaxon)
  (h3 : jaxon = 15)
  (h4 : jerry + gabriel + jaxon = 83) :
  jerry - gabriel = 8 := by
  sorry

end NUMINAMATH_CALUDE_jerry_gabriel_toy_difference_l2981_298184


namespace NUMINAMATH_CALUDE_ratio_problem_l2981_298132

theorem ratio_problem (x y z : ℝ) 
  (h : y / z = z / x ∧ z / x = x / y ∧ x / y = 1 / 2) : 
  (x / (y * z)) / (y / (z * x)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2981_298132


namespace NUMINAMATH_CALUDE_final_rope_length_l2981_298162

def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]
def knot_loss : ℝ := 1.2
def num_knots : ℕ := rope_lengths.length - 1

theorem final_rope_length :
  (rope_lengths.sum - num_knots * knot_loss : ℝ) = 35 := by
  sorry

end NUMINAMATH_CALUDE_final_rope_length_l2981_298162


namespace NUMINAMATH_CALUDE_propositions_p_and_not_q_l2981_298193

theorem propositions_p_and_not_q :
  (∃ x₀ : ℝ, Real.log x₀ ≥ x₀ - 1) ∧
  ¬(∀ θ : ℝ, Real.sin θ + Real.cos θ < 1) :=
by sorry

end NUMINAMATH_CALUDE_propositions_p_and_not_q_l2981_298193


namespace NUMINAMATH_CALUDE_brand_z_fraction_l2981_298174

/-- Represents the state of the fuel tank -/
structure TankState where
  z : ℚ  -- Amount of brand Z gasoline
  y : ℚ  -- Amount of brand Y gasoline

/-- Fills the tank with brand Y gasoline when it's partially empty -/
def fillWithY (s : TankState) (emptyFraction : ℚ) : TankState :=
  { z := s.z, y := s.y + emptyFraction }

/-- Fills the tank with brand Z gasoline when it's partially empty -/
def fillWithZ (s : TankState) (emptyFraction : ℚ) : TankState :=
  { z := s.z + emptyFraction, y := s.y }

/-- Empties the tank by a given fraction -/
def emptyTank (s : TankState) (emptyFraction : ℚ) : TankState :=
  { z := s.z * (1 - emptyFraction), y := s.y * (1 - emptyFraction) }

/-- The final state of the tank after the described filling process -/
def finalState : TankState :=
  let s1 := { z := 1, y := 0 }
  let s2 := fillWithY (emptyTank s1 (3/4)) (3/4)
  let s3 := fillWithZ (emptyTank s2 (1/2)) (1/2)
  fillWithY (emptyTank s3 (1/2)) (1/2)

/-- The fraction of brand Z gasoline in the final state is 5/16 -/
theorem brand_z_fraction :
  finalState.z / (finalState.z + finalState.y) = 5/16 := by sorry

end NUMINAMATH_CALUDE_brand_z_fraction_l2981_298174


namespace NUMINAMATH_CALUDE_function_root_implies_a_range_l2981_298104

theorem function_root_implies_a_range (a : ℝ) :
  (∃ x₀ : ℝ, -1 < x₀ ∧ x₀ < 1 ∧ 3 * a * x₀ - 2 * a + 1 = 0) →
  a < -1 ∨ a > 1/5 :=
by sorry

end NUMINAMATH_CALUDE_function_root_implies_a_range_l2981_298104


namespace NUMINAMATH_CALUDE_circle_C_and_line_l_properties_l2981_298160

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (0, 2)

-- Define the line y = x
def line_y_eq_x (x : ℝ) : ℝ := x

-- Define the line l: y = kx + 1
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the line m: y = -2x + 4
def line_m (x : ℝ) : ℝ := -2 * x + 4

-- Define circle P
def circle_P : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 5 * p.1^2 + 5 * p.2^2 - 16 * p.1 - 8 * p.2 + 12 = 0}

theorem circle_C_and_line_l_properties :
  ∃ (center : ℝ × ℝ) (P Q : ℝ × ℝ) (k : ℝ),
    center.2 = line_y_eq_x center.1 ∧
    point_A ∈ circle_C ∧
    point_B ∈ circle_C ∧
    P ∈ circle_C ∧
    Q ∈ circle_C ∧
    P.2 = line_l k P.1 ∧
    Q.2 = line_l k Q.1 ∧
    dot_product P Q = -2 →
    (∀ (p : ℝ × ℝ), p ∈ circle_C ↔ p.1^2 + p.2^2 = 4) ∧
    k = 0 ∧
    ∃ (E F : ℝ × ℝ),
      E ∈ circle_C ∧
      F ∈ circle_C ∧
      E.2 = line_m E.1 ∧
      F.2 = line_m F.1 ∧
      (2, 0) ∈ circle_P :=
by sorry

end NUMINAMATH_CALUDE_circle_C_and_line_l_properties_l2981_298160


namespace NUMINAMATH_CALUDE_unique_number_property_l2981_298131

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 3 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l2981_298131


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l2981_298152

theorem earth_inhabitable_fraction :
  let earth_surface := 1
  let land_fraction := (1 : ℚ) / 3
  let inhabitable_land_fraction := (2 : ℚ) / 3
  inhabitable_land_fraction * land_fraction * earth_surface = (2 : ℚ) / 9 :=
by sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l2981_298152


namespace NUMINAMATH_CALUDE_largest_integer_product_12_l2981_298102

theorem largest_integer_product_12 (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a * b * c * d * e = 12 →
  max a (max b (max c (max d e))) = 3 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_product_12_l2981_298102


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_f_l2981_298133

-- Define the function f
noncomputable def f (x : ℝ) : ℤ :=
  if x > -1 then Int.ceil (1 / (x + 1))
  else if x < -1 then Int.floor (1 / (x + 1))
  else 0  -- Arbitrary value for x = -1, as f is not defined there

-- Theorem statement
theorem zero_not_in_range_of_f :
  ¬ ∃ (x : ℝ), f x = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_f_l2981_298133


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2981_298124

theorem cubic_equation_solution (y : ℝ) :
  (((30 * y + (30 * y + 27) ^ (1/3 : ℝ)) ^ (1/3 : ℝ)) = 15) → y = 1674/15 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2981_298124


namespace NUMINAMATH_CALUDE_intersection_condition_implies_a_geq_5_l2981_298196

open Set Real

theorem intersection_condition_implies_a_geq_5 (a : ℝ) :
  let A := {x : ℝ | x ≤ a}
  let B := {x : ℝ | x^2 - 5*x < 0}
  A ∩ B = B → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_implies_a_geq_5_l2981_298196


namespace NUMINAMATH_CALUDE_complex_equality_l2981_298136

theorem complex_equality (a : ℝ) : 
  (Complex.re ((1 + 2*Complex.I) * (a + Complex.I)) = Complex.im ((1 + 2*Complex.I) * (a + Complex.I))) → 
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_l2981_298136


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2981_298172

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) < 0
def solution_set_f_neg (x : ℝ) : Prop := x < -1 ∨ x > 1/2

-- Define the solution set of f(10^x) > 0
def solution_set_f_exp (x : ℝ) : Prop := x < -Real.log 2 / Real.log 10

-- Theorem statement
theorem solution_set_equivalence :
  (∀ x, f x < 0 ↔ solution_set_f_neg x) →
  (∀ x, f (10^x) > 0 ↔ solution_set_f_exp x) :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2981_298172


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2981_298150

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (0 ≤ x ∧ x < 1) ↔ (x/2 + a ≥ 2 ∧ 2*x - b < 3)) → 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2981_298150


namespace NUMINAMATH_CALUDE_fraction_value_l2981_298169

theorem fraction_value (x y : ℚ) (hx : x = 12) (hy : y = -6) :
  (3 * x + y) / (x - y) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2981_298169


namespace NUMINAMATH_CALUDE_chess_tournament_solutions_l2981_298149

def chess_tournament (x : ℕ) : Prop :=
  ∃ y : ℕ,
    -- Two 7th graders scored 8 points in total
    8 + x * y = (x + 2) * (x + 1) / 2 ∧
    -- y is the number of points each 8th grader scored
    y > 0

theorem chess_tournament_solutions :
  ∀ x : ℕ, chess_tournament x ↔ (x = 7 ∨ x = 14) :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_solutions_l2981_298149


namespace NUMINAMATH_CALUDE_cafe_tables_theorem_l2981_298179

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- Calculates the number of tables needed given the number of people and people per table --/
def tablesNeeded (people : Nat) (peoplePerTable : Nat) : Nat :=
  people / peoplePerTable

theorem cafe_tables_theorem (seatingCapacity : Nat) (peoplePerTable : Nat) :
  seatingCapacity = 312 ∧ peoplePerTable = 3 →
  tablesNeeded (base7ToBase10 seatingCapacity) peoplePerTable = 52 := by
  sorry

#eval base7ToBase10 312  -- Should output 156
#eval tablesNeeded 156 3  -- Should output 52

end NUMINAMATH_CALUDE_cafe_tables_theorem_l2981_298179


namespace NUMINAMATH_CALUDE_real_roots_quadratic_complex_coeff_l2981_298151

theorem real_roots_quadratic_complex_coeff (i : ℂ) (m : ℝ) :
  (∃ x : ℝ, x^2 - (2*i - 1)*x + 3*m - i = 0) → m = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_complex_coeff_l2981_298151


namespace NUMINAMATH_CALUDE_A_in_third_quadrant_l2981_298167

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The given point A -/
def A : Point :=
  { x := -2, y := -3 }

/-- Theorem stating that point A is in the third quadrant -/
theorem A_in_third_quadrant : isInThirdQuadrant A := by
  sorry

end NUMINAMATH_CALUDE_A_in_third_quadrant_l2981_298167


namespace NUMINAMATH_CALUDE_max_k_value_l2981_298159

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * (x / y + y / x)) :
  k ≤ 3/7 := by
  sorry

end NUMINAMATH_CALUDE_max_k_value_l2981_298159


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l2981_298143

open Real

theorem function_inequality_implies_parameter_bound (a : ℝ) : 
  (∀ x > 0, 2 * x * log x ≥ -x^2 + a*x - 3) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l2981_298143


namespace NUMINAMATH_CALUDE_unique_solution_exists_l2981_298163

theorem unique_solution_exists : ∃! (x y : ℝ), 
  0.75 * x - 0.40 * y = 0.20 * 422.50 ∧ 
  0.30 * x + 0.50 * y = 0.35 * 530 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l2981_298163


namespace NUMINAMATH_CALUDE_polynomial_value_l2981_298198

theorem polynomial_value (p q : ℝ) : 
  (2*p - q + 3)^2 + 6*(2*p - q + 3) + 6 = (p + 4*q)^2 + 6*(p + 4*q) + 6 →
  p - 5*q + 3 ≠ 0 →
  (5*(p + q + 1))^2 + 6*(5*(p + q + 1)) + 6 = 46 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l2981_298198


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2981_298185

/-- Hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_ecc : e = 5/3

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a hyperbola and a point, check if the point is on the hyperbola -/
def is_on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Given a hyperbola, return its foci -/
def foci (h : Hyperbola) : (Point × Point) :=
  let c := h.a * h.e
  (Point.mk (-c) 0, Point.mk c 0)

/-- Given two points, check if they are perpendicular with respect to the origin -/
def are_perpendicular (p1 p2 : Point) : Prop :=
  p1.x * p2.x + p1.y * p2.y = 0

/-- Main theorem -/
theorem hyperbola_equation (h : Hyperbola) (p : Point) :
  let (f1, f2) := foci h
  (is_on_hyperbola h p) ∧
  (p.x = -3 ∧ p.y = -4) ∧
  (are_perpendicular (Point.mk (p.x - f1.x) (p.y - f1.y)) (Point.mk (p.x - f2.x) (p.y - f2.y))) →
  h.a = 3 ∧ h.b = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2981_298185


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2981_298126

theorem hyperbola_focal_length (m : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/m = 1) → 
  (∃ c : ℝ, c = 5) →
  m = 16 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2981_298126


namespace NUMINAMATH_CALUDE_beths_school_students_l2981_298186

theorem beths_school_students (beth paul : ℕ) : 
  beth = 4 * paul →  -- Beth's school has 4 times as many students as Paul's
  beth + paul = 5000 →  -- Total students in both schools is 5000
  beth = 4000 :=  -- Prove that Beth's school has 4000 students
by
  sorry

end NUMINAMATH_CALUDE_beths_school_students_l2981_298186


namespace NUMINAMATH_CALUDE_three_squares_side_length_l2981_298111

theorem three_squares_side_length (x : ℝ) :
  let middle := x + 17
  let right := middle - 6
  x + middle + right = 52 →
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_three_squares_side_length_l2981_298111


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l2981_298180

theorem consecutive_integers_cube_sum : 
  ∀ x : ℕ, x > 0 → 
  (x - 1) * x * (x + 1) = 12 * (3 * x) →
  (x - 1)^3 + x^3 + (x + 1)^3 = 3 * (37 : ℝ).sqrt^3 + 6 * (37 : ℝ).sqrt :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l2981_298180


namespace NUMINAMATH_CALUDE_two_by_two_squares_count_l2981_298154

theorem two_by_two_squares_count (grid_size : ℕ) (cuts : ℕ) (figures : ℕ) 
  (h1 : grid_size = 100)
  (h2 : cuts = 10000)
  (h3 : figures = 2500) : 
  ∃ (x : ℕ), x = 2300 ∧ 
  (8 * x + 10 * (figures - x) = 4 * grid_size + 2 * cuts) := by
  sorry

#check two_by_two_squares_count

end NUMINAMATH_CALUDE_two_by_two_squares_count_l2981_298154


namespace NUMINAMATH_CALUDE_rower_downstream_speed_l2981_298195

/-- Calculates the downstream speed of a rower given their upstream and still water speeds -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem stating that given the specified upstream and still water speeds, 
    the downstream speed is 48 kmph -/
theorem rower_downstream_speed :
  downstream_speed 32 40 = 48 := by
  sorry

end NUMINAMATH_CALUDE_rower_downstream_speed_l2981_298195


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l2981_298144

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube given its dimensions -/
def surfaceArea (c : CubeDimensions) : ℝ :=
  6 * c.length * c.width

/-- Represents the dimensions of a corner cube to be removed -/
structure CornerCubeDimensions where
  side : ℝ

/-- Theorem stating that removing corner cubes does not change the surface area -/
theorem surface_area_unchanged 
  (original : CubeDimensions) 
  (corner : CornerCubeDimensions) 
  (h1 : original.length = original.width ∧ original.width = original.height)
  (h2 : original.length = 5)
  (h3 : corner.side = 2) : 
  surfaceArea original = surfaceArea original := by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l2981_298144


namespace NUMINAMATH_CALUDE_projection_problem_l2981_298190

/-- Given two vectors that project onto the same vector, prove the resulting projection vector --/
theorem projection_problem (a b v p : ℝ × ℝ) : 
  a = (-3, 2) →
  b = (4, 5) →
  (∃ (k₁ k₂ : ℝ), p = k₁ • v ∧ p = k₂ • v) →
  p = (-69/58, 161/58) := by
  sorry

end NUMINAMATH_CALUDE_projection_problem_l2981_298190


namespace NUMINAMATH_CALUDE_spurs_basketball_count_l2981_298155

theorem spurs_basketball_count :
  let num_players : ℕ := 22
  let basketballs_per_player : ℕ := 11
  num_players * basketballs_per_player = 242 :=
by sorry

end NUMINAMATH_CALUDE_spurs_basketball_count_l2981_298155


namespace NUMINAMATH_CALUDE_line_equation_final_line_equation_l2981_298115

/-- The ellipse with equation x²/9 + y²/4 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The line passing through (1, 1) with slope k -/
def line (x y k : ℝ) : Prop := y = k * (x - 1) + 1

/-- The point (1, 1) is on the line -/
def point_on_line (k : ℝ) : Prop := line 1 1 k

/-- The point (1, 1) is the midpoint of the chord intercepted by the line from the ellipse -/
def is_midpoint (k : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ,
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line x₁ y₁ k ∧ line x₂ y₂ k ∧
    (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 1

theorem line_equation :
  ∀ k : ℝ, point_on_line k → is_midpoint k → k = -4/9 :=
sorry

theorem final_line_equation :
  ∀ x y : ℝ, line x y (-4/9) ↔ 4*x + 9*y = 13 :=
sorry

end NUMINAMATH_CALUDE_line_equation_final_line_equation_l2981_298115


namespace NUMINAMATH_CALUDE_boxes_remaining_l2981_298137

theorem boxes_remaining (total : ℕ) (filled : ℕ) (h1 : total = 13) (h2 : filled = 8) :
  total - filled = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_remaining_l2981_298137


namespace NUMINAMATH_CALUDE_well_digging_payment_l2981_298191

/-- The total payment for two workers digging a well over three days -/
def total_payment (hourly_rate : ℕ) (day1_hours day2_hours day3_hours : ℕ) (num_workers : ℕ) : ℕ :=
  hourly_rate * (day1_hours + day2_hours + day3_hours) * num_workers

/-- Theorem stating that the total payment for the given scenario is $660 -/
theorem well_digging_payment :
  total_payment 10 10 8 15 2 = 660 := by
  sorry

end NUMINAMATH_CALUDE_well_digging_payment_l2981_298191


namespace NUMINAMATH_CALUDE_cubic_solution_sum_l2981_298120

theorem cubic_solution_sum (a b c : ℝ) : 
  (a^3 - 4*a^2 + 7*a = 15) ∧ 
  (b^3 - 4*b^2 + 7*b = 15) ∧ 
  (c^3 - 4*c^2 + 7*c = 15) →
  a*b/c + b*c/a + c*a/b = 49/15 := by
sorry

end NUMINAMATH_CALUDE_cubic_solution_sum_l2981_298120


namespace NUMINAMATH_CALUDE_distance_sum_l2981_298117

/-- Represents a segment with a midpoint and a point Q -/
structure Segment where
  length : ℝ
  q_distance : ℝ

/-- The problem setup -/
def problem_setup (cd : Segment) (cd_prime : Segment) : Prop :=
  cd.length = 10 ∧
  cd_prime.length = 16 ∧
  cd.q_distance = 3 ∧
  cd.q_distance = 2 * (cd_prime.length / 2 - (cd_prime.length / 2 - cd_prime.q_distance))

/-- The theorem to prove -/
theorem distance_sum (cd : Segment) (cd_prime : Segment) 
  (h : problem_setup cd cd_prime) : 
  cd.q_distance + cd_prime.q_distance = 7 := by
  sorry


end NUMINAMATH_CALUDE_distance_sum_l2981_298117


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l2981_298171

theorem min_value_of_expression (x y : ℝ) : (x*y - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, (x*y - 1)^2 + (x + y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l2981_298171


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_zero_l2981_298194

/-- Two real numbers are opposite if their sum is zero. -/
def are_opposite (a b : ℝ) : Prop := a + b = 0

/-- If a and b are opposite numbers, then their sum is zero. -/
theorem opposite_numbers_sum_zero (a b : ℝ) (h : are_opposite a b) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_zero_l2981_298194


namespace NUMINAMATH_CALUDE_sum_C_D_equals_28_l2981_298142

theorem sum_C_D_equals_28 (C D : ℝ) :
  (∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-4 * x^2 + 18 * x + 32) / (x - 3)) →
  C + D = 28 := by
sorry

end NUMINAMATH_CALUDE_sum_C_D_equals_28_l2981_298142


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_48_l2981_298109

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_non_factor_product_of_48 (x y : ℕ) : 
  x ≠ y → 
  x > 0 → 
  y > 0 → 
  is_factor x 48 → 
  is_factor y 48 → 
  ¬ is_factor (x * y) 48 → 
  ∀ a b : ℕ, a ≠ b → a > 0 → b > 0 → is_factor a 48 → is_factor b 48 → ¬ is_factor (a * b) 48 → x * y ≤ a * b →
  x * y = 32 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_48_l2981_298109


namespace NUMINAMATH_CALUDE_pentagon_angles_count_l2981_298105

/-- Represents a sequence of 5 interior angles of a convex pentagon --/
structure PentagonAngles where
  angles : Fin 5 → ℕ
  sum_540 : (angles 0) + (angles 1) + (angles 2) + (angles 3) + (angles 4) = 540
  increasing : ∀ i j, i < j → angles i < angles j
  smallest_ge_60 : angles 0 ≥ 60
  largest_lt_150 : angles 4 < 150
  arithmetic : ∃ d : ℕ, ∀ i : Fin 4, angles (i + 1) = angles i + d
  not_equiangular : ¬ (∀ i j, angles i = angles j)

/-- The number of valid PentagonAngles --/
def validPentagonAnglesCount : ℕ := 5

theorem pentagon_angles_count :
  {s : Finset PentagonAngles | s.card = validPentagonAnglesCount} ≠ ∅ :=
sorry

end NUMINAMATH_CALUDE_pentagon_angles_count_l2981_298105


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2981_298187

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 * a 3 + 2 * a 3 * a 5 + a 5 * a 7 = 4) →
  (a 2 + a 6 = 2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2981_298187


namespace NUMINAMATH_CALUDE_xyz_sum_l2981_298100

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 147)
  (eq2 : y^2 + y*z + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 156) :
  x*y + y*z + x*z = 42 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l2981_298100


namespace NUMINAMATH_CALUDE_mask_production_growth_rate_equation_l2981_298114

/-- Represents the monthly average growth rate of mask production from January to March. -/
def monthly_growth_rate : ℝ → Prop :=
  λ x => (160000 : ℝ) * (1 + x)^2 = 250000

/-- The equation representing the monthly average growth rate of mask production
    from January to March is 16(1+x)^2 = 25, given initial production of 160,000 masks
    in January and 250,000 masks in March. -/
theorem mask_production_growth_rate_equation :
  ∃ x : ℝ, monthly_growth_rate x ∧ 16 * (1 + x)^2 = 25 :=
sorry

end NUMINAMATH_CALUDE_mask_production_growth_rate_equation_l2981_298114


namespace NUMINAMATH_CALUDE_pure_imaginary_modulus_l2981_298138

theorem pure_imaginary_modulus (b : ℝ) : 
  let z : ℂ := (3 + b * Complex.I) * (1 + Complex.I) - 2
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs z = 4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_modulus_l2981_298138


namespace NUMINAMATH_CALUDE_linear_regression_equation_l2981_298113

/-- Linear regression equation for given points -/
theorem linear_regression_equation (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁ = 3 ∧ y₁ = 10)
  (h₂ : x₂ = 7 ∧ y₂ = 20)
  (h₃ : x₃ = 11 ∧ y₃ = 24) :
  ∃ (a b : ℝ), a = 5.75 ∧ b = 1.75 ∧ 
    (∀ (x y : ℝ), (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (x = x₃ ∧ y = y₃) → y = a + b * x) :=
sorry

end NUMINAMATH_CALUDE_linear_regression_equation_l2981_298113


namespace NUMINAMATH_CALUDE_pythagorean_triple_with_24_and_7_l2981_298156

theorem pythagorean_triple_with_24_and_7 : 
  ∃ (x : ℕ), x > 0 ∧ x^2 + 7^2 = 24^2 ∨ x^2 = 24^2 + 7^2 → x = 25 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_with_24_and_7_l2981_298156


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2981_298192

theorem inequality_solution_set (x : ℝ) :
  ((x + 1) / (3 - x) < 0) ↔ (x < -1 ∨ x > 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2981_298192


namespace NUMINAMATH_CALUDE_distinct_arrangements_statistics_l2981_298168

def word_length : ℕ := 10
def letter_counts : List ℕ := [3, 2, 2, 1, 1]

theorem distinct_arrangements_statistics :
  (word_length.factorial) / ((letter_counts.map Nat.factorial).prod) = 75600 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_statistics_l2981_298168


namespace NUMINAMATH_CALUDE_problem_statement_l2981_298166

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem problem_statement :
  (∀ a, a ∈ M → a ∈ N) ∧ 
  (∃ a, a ∈ M ∧ a ∉ N) ∧
  (∀ a b, (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M)) ∧
  (¬(∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2981_298166


namespace NUMINAMATH_CALUDE_new_quadratic_from_original_l2981_298135

theorem new_quadratic_from_original (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 9 * x₁ + 8 = 0) ∧ (2 * x₂^2 - 9 * x₂ + 8 = 0) →
  ∃ y₁ y₂ : ℝ, 
    (36 * y₁^2 - 161 * y₁ + 34 = 0) ∧
    (36 * y₂^2 - 161 * y₂ + 34 = 0) ∧
    (y₁ = 1 / (x₁ + x₂) ∨ y₁ = (x₁ - x₂)^2) ∧
    (y₂ = 1 / (x₁ + x₂) ∨ y₂ = (x₁ - x₂)^2) ∧
    y₁ ≠ y₂ :=
by sorry

end NUMINAMATH_CALUDE_new_quadratic_from_original_l2981_298135


namespace NUMINAMATH_CALUDE_grape_crates_count_l2981_298153

/-- Proves that the number of grape crates is 13 given the total number of crates and the number of mango and passion fruit crates. -/
theorem grape_crates_count (total_crates mango_crates passion_fruit_crates : ℕ) 
  (h1 : total_crates = 50)
  (h2 : mango_crates = 20)
  (h3 : passion_fruit_crates = 17) :
  total_crates - (mango_crates + passion_fruit_crates) = 13 := by
  sorry

end NUMINAMATH_CALUDE_grape_crates_count_l2981_298153


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2981_298177

theorem arithmetic_expression_equality : 70 + (105 / 15) + (19 * 11) - 250 - (360 / 12) = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2981_298177


namespace NUMINAMATH_CALUDE_phi_range_for_increasing_f_l2981_298165

open Real

/-- Given f(x) = sin(x) - 2cos(x - φ)sin(φ) is increasing on [3π, 7π/2] and 0 < φ < π, 
    then π/2 ≤ φ ≤ 3π/4 -/
theorem phi_range_for_increasing_f (φ : ℝ) : 
  (0 < φ) → (φ < π) → 
  (∀ x ∈ Set.Icc (3 * π) ((7 * π) / 2), 
    Monotone (fun x => sin x - 2 * cos (x - φ) * sin φ)) →
  (π / 2 ≤ φ) ∧ (φ ≤ (3 * π) / 4) := by
  sorry


end NUMINAMATH_CALUDE_phi_range_for_increasing_f_l2981_298165


namespace NUMINAMATH_CALUDE_bike_vs_drive_time_difference_l2981_298183

theorem bike_vs_drive_time_difference 
  (normal_drive_time : ℝ) 
  (normal_drive_speed : ℝ) 
  (bike_route_reduction : ℝ) 
  (min_bike_speed : ℝ) 
  (max_bike_speed : ℝ) 
  (h1 : normal_drive_time = 45) 
  (h2 : normal_drive_speed = 40) 
  (h3 : bike_route_reduction = 0.2) 
  (h4 : min_bike_speed = 12) 
  (h5 : max_bike_speed = 16) : 
  ∃ (time_difference : ℝ), time_difference = 75 := by
sorry

end NUMINAMATH_CALUDE_bike_vs_drive_time_difference_l2981_298183


namespace NUMINAMATH_CALUDE_valid_systematic_sample_l2981_298181

/-- Represents a systematic sample -/
structure SystematicSample where
  population : ℕ
  sampleSize : ℕ
  startPoint : ℕ
  interval : ℕ

/-- Checks if a given list is a valid systematic sample -/
def isValidSystematicSample (sample : List ℕ) (s : SystematicSample) : Prop :=
  sample.length = s.sampleSize ∧
  sample.all (·≤ s.population) ∧
  sample.all (·> 0) ∧
  ∀ i, i < sample.length - 1 → sample[i + 1]! - sample[i]! = s.interval

theorem valid_systematic_sample :
  let sample := [3, 13, 23, 33, 43]
  let s : SystematicSample := {
    population := 50,
    sampleSize := 5,
    startPoint := 3,
    interval := 10
  }
  isValidSystematicSample sample s := by
  sorry

end NUMINAMATH_CALUDE_valid_systematic_sample_l2981_298181


namespace NUMINAMATH_CALUDE_tangent_parabola_difference_l2981_298110

/-- Given a parabola y = x^2 + ax + b and a tangent line y = kx + 1 at point (1, 3),
    prove that a - b = -2 -/
theorem tangent_parabola_difference (a b k : ℝ) : 
  (∀ x, x^2 + a*x + b = k*x + 1 → 2*x + a = k) →  -- Tangency condition
  1^2 + a*1 + b = 3 →                             -- Point (1, 3) is on the parabola
  k*1 + 1 = 3 →                                   -- Point (1, 3) is on the tangent line
  a - b = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parabola_difference_l2981_298110


namespace NUMINAMATH_CALUDE_inequality_with_negative_multiplication_l2981_298199

theorem inequality_with_negative_multiplication 
  (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a * c < b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_with_negative_multiplication_l2981_298199


namespace NUMINAMATH_CALUDE_range_of_a_l2981_298170

/-- Given that |a-2x| > x-1 for all x in [0,2], prove that a is in (-∞,2) ∪ (5,+∞) -/
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → |a - 2*x| > x - 1) → 
  a ∈ Set.Ioi 5 ∪ Set.Iic 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2981_298170


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l2981_298127

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 12 = 0

-- Define point D
def point_D : ℝ × ℝ := (-2, 0)

-- Define the line l passing through D
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2)

-- Define the isosceles right triangle condition
def is_isosceles_right_triangle (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 2 * ((A.1 - C.1)^2 + (A.2 - C.2)^2)

-- Define the intersection points condition
def are_intersection_points (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
  line_l k A.1 A.2 ∧ line_l k B.1 B.2

-- Theorem statement
theorem circle_and_line_problem :
  ∃ (A B C : ℝ × ℝ) (k : ℝ),
    C = (0, 4) ∧
    are_intersection_points A B k ∧
    is_isosceles_right_triangle A B C →
    (∀ (x y : ℝ), (x + 1)^2 + (y - 2)^2 = 5 ↔ 
      ∃ (P Q : ℝ × ℝ), P = C ∧ Q = point_D ∧ 
      (x - (P.1 + Q.1)/2)^2 + (y - (P.2 + Q.2)/2)^2 = 
      ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) / 4) ∧
    (k = 1 ∨ k = 7) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_problem_l2981_298127


namespace NUMINAMATH_CALUDE_factor_tree_proof_l2981_298116

theorem factor_tree_proof (X Y Z F : ℕ) : 
  X = Y * Z → 
  Y = 7 * 11 → 
  Z = 7 * F → 
  F = 11 * 2 → 
  X = 11858 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_proof_l2981_298116


namespace NUMINAMATH_CALUDE_c_share_correct_l2981_298188

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_share (total_profit : ℕ) (investments : List ℕ) (partner_index : ℕ) : ℕ :=
  sorry

theorem c_share_correct (total_profit : ℕ) (investments : List ℕ) :
  total_profit = 90000 →
  investments = [30000, 45000, 50000] →
  calculate_share total_profit investments 2 = 36000 :=
by sorry

end NUMINAMATH_CALUDE_c_share_correct_l2981_298188


namespace NUMINAMATH_CALUDE_cube_of_product_l2981_298197

theorem cube_of_product (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_product_l2981_298197


namespace NUMINAMATH_CALUDE_mary_donated_books_l2981_298119

def books_donated (initial_books : ℕ) (monthly_books : ℕ) (bookstore_books : ℕ) 
  (yard_sale_books : ℕ) (daughter_books : ℕ) (mother_books : ℕ) 
  (sold_books : ℕ) (final_books : ℕ) : ℕ :=
  initial_books + (monthly_books * 12) + bookstore_books + yard_sale_books + 
  daughter_books + mother_books - sold_books - final_books

theorem mary_donated_books : 
  books_donated 72 1 5 2 1 4 3 81 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mary_donated_books_l2981_298119


namespace NUMINAMATH_CALUDE_jules_starting_fee_is_two_l2981_298145

/-- Calculates the starting fee per walk for Jules' dog walking service -/
def starting_fee_per_walk (total_vacation_cost : ℚ) (family_members : ℕ) 
  (price_per_block : ℚ) (dogs_walked : ℕ) (total_blocks : ℕ) : ℚ :=
  let individual_contribution := total_vacation_cost / family_members
  let earnings_from_blocks := price_per_block * total_blocks
  let total_starting_fees := individual_contribution - earnings_from_blocks
  total_starting_fees / dogs_walked

/-- Proves that Jules' starting fee per walk is $2 given the problem conditions -/
theorem jules_starting_fee_is_two :
  starting_fee_per_walk 1000 5 (5/4) 20 128 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jules_starting_fee_is_two_l2981_298145


namespace NUMINAMATH_CALUDE_vector_equality_l2981_298140

/-- Given vectors a, b, and c in ℝ², prove that c = a - b -/
theorem vector_equality (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (2, -1)) 
  (hc : c = (-1, 2)) : 
  c = a - b := by sorry

end NUMINAMATH_CALUDE_vector_equality_l2981_298140


namespace NUMINAMATH_CALUDE_last_day_pages_for_specific_book_l2981_298103

/-- Calculates the number of pages read on the last day to complete a book -/
def pages_on_last_day (total_pages : ℕ) (pages_per_day : ℕ) (break_interval : ℕ) : ℕ :=
  let pages_per_cycle := pages_per_day * (break_interval - 1)
  let full_cycles := (total_pages / pages_per_cycle : ℕ)
  let pages_read_in_full_cycles := full_cycles * pages_per_cycle
  total_pages - pages_read_in_full_cycles

theorem last_day_pages_for_specific_book :
  pages_on_last_day 575 37 3 = 57 := by
  sorry

end NUMINAMATH_CALUDE_last_day_pages_for_specific_book_l2981_298103
