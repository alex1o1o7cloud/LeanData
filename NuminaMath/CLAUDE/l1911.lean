import Mathlib

namespace NUMINAMATH_CALUDE_adam_has_more_apples_l1911_191181

/-- The number of apples Jackie has -/
def jackies_apples : ℕ := 6

/-- The number of apples Adam has -/
def adams_apples : ℕ := jackies_apples + 3

theorem adam_has_more_apples : adams_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_adam_has_more_apples_l1911_191181


namespace NUMINAMATH_CALUDE_angle_ADB_is_right_angle_l1911_191156

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Checks if a triangle is isosceles with two sides equal -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2

/-- Checks if a point lies on a circle -/
def Circle.contains (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Calculates the angle between three points -/
def angle (A B C : Point) : ℝ := sorry

/-- Main theorem -/
theorem angle_ADB_is_right_angle 
  (t : Triangle) 
  (c : Circle) 
  (D : Point) 
  (h1 : t.isIsosceles)
  (h2 : c.center = t.C)
  (h3 : c.radius = 15)
  (h4 : c.contains t.B)
  (h5 : ∃ k : ℝ, D.x = t.C.x + k * (t.C.x - t.A.x) ∧ D.y = t.C.y + k * (t.C.y - t.A.y))
  (h6 : c.contains D) :
  angle t.A D t.B = 90 := by sorry

end NUMINAMATH_CALUDE_angle_ADB_is_right_angle_l1911_191156


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1911_191152

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - 3*x + 2 = 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1911_191152


namespace NUMINAMATH_CALUDE_horner_method_f_neg_four_l1911_191129

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 12 - 8x^2 + 6x^4 + 5x^5 + 3x^6 -/
def f (x : ℤ) : ℤ := 12 - 8*x^2 + 6*x^4 + 5*x^5 + 3*x^6

theorem horner_method_f_neg_four :
  horner_eval [3, 5, 6, 0, -8, 0, 12] (-4) = f (-4) ∧ f (-4) = -845 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_neg_four_l1911_191129


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l1911_191111

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x, x^2 + 9*x + 18 = (x + a) * (x + b)) →
  (∀ x, x^2 + 19*x + 90 = (x + b) * (x + c)) →
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l1911_191111


namespace NUMINAMATH_CALUDE_sum_first_n_integers_remainder_l1911_191123

theorem sum_first_n_integers_remainder (n : ℕ+) :
  let sum := n.val * (n.val + 1) / 2
  sum % n.val = if n.val % 2 = 1 then 0 else n.val / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_n_integers_remainder_l1911_191123


namespace NUMINAMATH_CALUDE_percent_value_in_quarters_l1911_191115

def num_dimes : ℕ := 75
def num_quarters : ℕ := 30
def value_dime : ℕ := 10
def value_quarter : ℕ := 25

def total_value : ℕ := num_dimes * value_dime + num_quarters * value_quarter
def value_in_quarters : ℕ := num_quarters * value_quarter

theorem percent_value_in_quarters :
  (value_in_quarters : ℚ) / total_value * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_percent_value_in_quarters_l1911_191115


namespace NUMINAMATH_CALUDE_negation_of_implication_l1911_191175

theorem negation_of_implication :
  (¬(x = 3 → x^2 - 2*x - 3 = 0)) ↔ (x = 3 ∧ x^2 - 2*x - 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1911_191175


namespace NUMINAMATH_CALUDE_cyclist_speed_ratio_l1911_191151

theorem cyclist_speed_ratio :
  ∀ (v₁ v₂ : ℝ),
  v₁ > v₂ →
  v₁ + v₂ = 20 →
  v₁ - v₂ = 5 →
  v₁ / v₂ = 5 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_ratio_l1911_191151


namespace NUMINAMATH_CALUDE_car_arrival_time_difference_l1911_191139

/-- Theorem: Difference in arrival times for two cars traveling the same distance at different speeds -/
theorem car_arrival_time_difference 
  (distance : ℝ) 
  (speed_slow : ℝ) 
  (speed_fast : ℝ) 
  (h_distance : distance = 4.333329)
  (h_speed_slow : speed_slow = 72)
  (h_speed_fast : speed_fast = 78) :
  (distance / speed_slow) - (distance / speed_fast) = 0.004629369 := by
sorry

end NUMINAMATH_CALUDE_car_arrival_time_difference_l1911_191139


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l1911_191198

theorem sqrt_50_between_consecutive_integers_product : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 50 ∧ 
  Real.sqrt 50 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l1911_191198


namespace NUMINAMATH_CALUDE_max_intersections_is_19_l1911_191188

/-- The maximum number of intersection points between three circles -/
def max_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between a line and three circles -/
def max_line_circle_intersections : ℕ := 6

/-- The number of lines -/
def num_lines : ℕ := 2

/-- The number of intersection points between two lines -/
def line_line_intersections : ℕ := 1

/-- The maximum number of intersection points between 3 circles and 2 straight lines on a plane -/
def max_total_intersections : ℕ :=
  max_circle_intersections + 
  (num_lines * max_line_circle_intersections) + 
  line_line_intersections

theorem max_intersections_is_19 : 
  max_total_intersections = 19 := by sorry

end NUMINAMATH_CALUDE_max_intersections_is_19_l1911_191188


namespace NUMINAMATH_CALUDE_carbonic_acid_weight_l1911_191106

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.008

/-- Atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.011

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- Number of Hydrogen atoms in Carbonic acid -/
def num_H : ℕ := 2

/-- Number of Carbon atoms in Carbonic acid -/
def num_C : ℕ := 1

/-- Number of Oxygen atoms in Carbonic acid -/
def num_O : ℕ := 3

/-- Number of moles of Carbonic acid -/
def num_moles : ℝ := 8

/-- Molecular weight of Carbonic acid in g/mol -/
def molecular_weight_H2CO3 : ℝ := 
  num_H * atomic_weight_H + num_C * atomic_weight_C + num_O * atomic_weight_O

/-- Total weight of given moles of Carbonic acid in grams -/
def total_weight : ℝ := num_moles * molecular_weight_H2CO3

theorem carbonic_acid_weight : total_weight = 496.192 := by
  sorry

end NUMINAMATH_CALUDE_carbonic_acid_weight_l1911_191106


namespace NUMINAMATH_CALUDE_one_is_monomial_l1911_191164

/-- Definition of a monomial as an algebraic expression with one term -/
def isMonomial (expr : ℕ → ℚ) : Prop :=
  ∃ (c : ℚ) (n : ℕ), ∀ (k : ℕ), expr k = if k = n then c else 0

/-- Theorem: 1 is a monomial -/
theorem one_is_monomial : isMonomial (fun _ ↦ 1) := by
  sorry

end NUMINAMATH_CALUDE_one_is_monomial_l1911_191164


namespace NUMINAMATH_CALUDE_prescription_rebate_calculation_l1911_191136

/-- Calculates the mail-in rebate amount for a prescription purchase -/
def calculate_rebate (original_cost cashback_percent final_cost : ℚ) : ℚ :=
  let cashback := original_cost * (cashback_percent / 100)
  let cost_after_cashback := original_cost - cashback
  cost_after_cashback - final_cost

theorem prescription_rebate_calculation :
  let original_cost : ℚ := 150
  let cashback_percent : ℚ := 10
  let final_cost : ℚ := 110
  calculate_rebate original_cost cashback_percent final_cost = 25 := by
  sorry

#eval calculate_rebate 150 10 110

end NUMINAMATH_CALUDE_prescription_rebate_calculation_l1911_191136


namespace NUMINAMATH_CALUDE_pqr_product_l1911_191146

theorem pqr_product (p q r : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 → 
  p + q + r = 27 → 
  1 / p + 1 / q + 1 / r + 432 / (p * q * r) = 1 → 
  p * q * r = 1380 := by
sorry

end NUMINAMATH_CALUDE_pqr_product_l1911_191146


namespace NUMINAMATH_CALUDE_dividend_calculation_l1911_191199

theorem dividend_calculation (divisor quotient remainder dividend : ℕ) : 
  divisor = 3 →
  quotient = 7 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  dividend = 23 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1911_191199


namespace NUMINAMATH_CALUDE_inequality_proof_l1911_191183

theorem inequality_proof (x y z : ℝ) 
  (h1 : y ≥ 2 * z) 
  (h2 : 2 * z ≥ 4 * x)
  (h3 : 2 * (x^3 + y^3 + z^3) + 15 * (x * y^2 + y * z^2 + z * x^2) ≥ 
        16 * (x^2 * y + y^2 * z + z^2 * x) + 2 * x * y * z) :
  4 * x + y ≥ 4 * z := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1911_191183


namespace NUMINAMATH_CALUDE_inverse_f_at_317_l1911_191101

def f (x : ℝ) : ℝ := 5 * x^3 - 3

theorem inverse_f_at_317 : f⁻¹ 317 = 4 := by sorry

end NUMINAMATH_CALUDE_inverse_f_at_317_l1911_191101


namespace NUMINAMATH_CALUDE_triple_base_double_exponent_l1911_191148

theorem triple_base_double_exponent (a b x : ℝ) (h1 : b ≠ 0) :
  let r := (3 * a) ^ (2 * b)
  r = a ^ b * x ^ b → x = 9 * a := by
sorry

end NUMINAMATH_CALUDE_triple_base_double_exponent_l1911_191148


namespace NUMINAMATH_CALUDE_first_number_value_l1911_191196

theorem first_number_value (x y z : ℤ) 
  (sum_xy : x + y = 31)
  (sum_yz : y + z = 47)
  (sum_xz : x + z = 52)
  (condition : y + z = x + 16) :
  x = 31 := by
sorry

end NUMINAMATH_CALUDE_first_number_value_l1911_191196


namespace NUMINAMATH_CALUDE_number_of_orders_is_1536_l1911_191132

/-- Represents the number of letters --/
def n : ℕ := 10

/-- Represents the number of letters that can be in the stack (excluding 9 and 10) --/
def m : ℕ := 8

/-- Calculates the number of different orders for typing the remaining letters --/
def number_of_orders : ℕ :=
  Finset.sum (Finset.range (m + 1)) (λ k => (Nat.choose m k) * (k + 2))

/-- Theorem stating that the number of different orders is 1536 --/
theorem number_of_orders_is_1536 : number_of_orders = 1536 := by
  sorry

end NUMINAMATH_CALUDE_number_of_orders_is_1536_l1911_191132


namespace NUMINAMATH_CALUDE_subset_implies_a_zero_l1911_191110

theorem subset_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {1, 2, a}
  let B : Set ℝ := {2, a^2 + 1}
  B ⊆ A → a = 0 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_zero_l1911_191110


namespace NUMINAMATH_CALUDE_min_value_implications_l1911_191140

theorem min_value_implications (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, |x + a| + |x - b| ≥ 2) : 
  (1 / a + 1 / b + 1 / (a * b) ≥ 3) ∧ 
  (∀ t : ℝ, Real.sin t ^ 4 / a + Real.cos t ^ 4 / b ≥ 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_implications_l1911_191140


namespace NUMINAMATH_CALUDE_min_value_of_s_l1911_191147

theorem min_value_of_s (a b : ℤ) :
  let s := a^3 + b^3 - 60*a*b*(a + b)
  s ≥ 2012 → s ≥ 2015 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_s_l1911_191147


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1911_191145

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ = 3 and a₃ = 5, prove that a₅ = 7 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a1 : a 1 = 3) 
  (h_a3 : a 3 = 5) : 
  a 5 = 7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1911_191145


namespace NUMINAMATH_CALUDE_noahs_age_ratio_l1911_191134

theorem noahs_age_ratio (joe_age : ℕ) (noah_future_age : ℕ) (years_to_future : ℕ) :
  joe_age = 6 →
  noah_future_age = 22 →
  years_to_future = 10 →
  ∃ k : ℕ, k * joe_age = noah_future_age - years_to_future →
  (noah_future_age - years_to_future) / joe_age = 2 := by
sorry

end NUMINAMATH_CALUDE_noahs_age_ratio_l1911_191134


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1911_191167

theorem quadratic_real_roots (a : ℝ) : 
  a > 1 → ∃ x : ℝ, x^2 - (2*a + 1)*x + a^2 = 0 :=
by
  sorry

#check quadratic_real_roots

end NUMINAMATH_CALUDE_quadratic_real_roots_l1911_191167


namespace NUMINAMATH_CALUDE_certain_number_value_l1911_191180

def is_smallest_multiplier (n : ℕ) (x : ℝ) : Prop :=
  n > 0 ∧ ∃ (y : ℕ), n * x = y^2 ∧
  ∀ (m : ℕ), m > 0 → m < n → ¬∃ (z : ℕ), m * x = z^2

theorem certain_number_value (n : ℕ) (x : ℝ) :
  is_smallest_multiplier n x → n = 3 → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_value_l1911_191180


namespace NUMINAMATH_CALUDE_kelly_chickens_count_l1911_191149

/-- The number of chickens Kelly has -/
def number_of_chickens : ℕ := 8

/-- The number of eggs each chicken lays per day -/
def eggs_per_chicken_per_day : ℕ := 3

/-- The price of a dozen eggs in dollars -/
def price_per_dozen : ℕ := 5

/-- The total amount Kelly makes in 4 weeks in dollars -/
def total_earnings : ℕ := 280

/-- The number of days in 4 weeks -/
def days_in_four_weeks : ℕ := 28

theorem kelly_chickens_count :
  number_of_chickens * eggs_per_chicken_per_day * days_in_four_weeks / 12 * price_per_dozen = total_earnings :=
sorry

end NUMINAMATH_CALUDE_kelly_chickens_count_l1911_191149


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l1911_191159

theorem arithmetic_sequence_20th_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 3 →                            -- first term is 3
    a 1 = 7 →                            -- second term is 7
    a 19 = 79 :=                         -- 20th term (index 19) is 79
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l1911_191159


namespace NUMINAMATH_CALUDE_molar_mass_calculation_l1911_191135

/-- Given that 3 moles of a substance weigh 264 grams, prove that its molar mass is 88 grams/mole -/
theorem molar_mass_calculation (total_weight : ℝ) (num_moles : ℝ) (h1 : total_weight = 264) (h2 : num_moles = 3) :
  total_weight / num_moles = 88 := by
  sorry

end NUMINAMATH_CALUDE_molar_mass_calculation_l1911_191135


namespace NUMINAMATH_CALUDE_upward_translation_4_units_l1911_191122

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A translation in 2D space -/
structure Translation2D where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (p : Point2D) (t : Translation2D) : Point2D :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem upward_translation_4_units 
  (M : Point2D)
  (N : Point2D)
  (h1 : M.x = -1 ∧ M.y = -1)
  (h2 : N.x = -1 ∧ N.y = 3) :
  ∃ (t : Translation2D), t.dx = 0 ∧ t.dy = 4 ∧ applyTranslation M t = N :=
sorry

end NUMINAMATH_CALUDE_upward_translation_4_units_l1911_191122


namespace NUMINAMATH_CALUDE_shopkeeper_payment_l1911_191193

/-- The total cost of a purchase given the quantity and price per unit -/
def totalCost (quantity : ℕ) (pricePerUnit : ℕ) : ℕ :=
  quantity * pricePerUnit

/-- The problem statement -/
theorem shopkeeper_payment : 
  let grapeQuantity : ℕ := 8
  let grapePrice : ℕ := 70
  let mangoQuantity : ℕ := 9
  let mangoPrice : ℕ := 50
  let grapeCost := totalCost grapeQuantity grapePrice
  let mangoCost := totalCost mangoQuantity mangoPrice
  grapeCost + mangoCost = 1010 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_payment_l1911_191193


namespace NUMINAMATH_CALUDE_nine_knights_in_room_l1911_191150

/-- Represents a person on the island, either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- The total number of people in the room -/
def totalPeople : Nat := 15

/-- Represents the statements made by each person -/
structure Statements where
  sixLiars : Bool  -- "Among my acquaintances in this room, there are exactly six liars"
  noMoreThanSevenKnights : Bool  -- "Among my acquaintances in this room, there are no more than seven knights"

/-- Returns true if the statements are consistent with the person's type and the room's composition -/
def statementsAreConsistent (p : Person) (statements : Statements) (knightCount : Nat) : Bool :=
  match p with
  | Person.Knight => statements.sixLiars = (totalPeople - knightCount - 1 = 6) ∧ 
                     statements.noMoreThanSevenKnights = (knightCount - 1 ≤ 7)
  | Person.Liar => statements.sixLiars ≠ (totalPeople - knightCount - 1 = 6) ∧ 
                   statements.noMoreThanSevenKnights ≠ (knightCount - 1 ≤ 7)

/-- The main theorem: there are exactly 9 knights in the room -/
theorem nine_knights_in_room : 
  ∃ (knightCount : Nat), knightCount = 9 ∧ 
  (∀ (p : Person) (s : Statements), statementsAreConsistent p s knightCount) ∧
  knightCount + (totalPeople - knightCount) = totalPeople :=
sorry

end NUMINAMATH_CALUDE_nine_knights_in_room_l1911_191150


namespace NUMINAMATH_CALUDE_cubic_monomial_properties_l1911_191186

/-- A cubic monomial with coefficient -2 using only variables x and y -/
def cubic_monomial (x y : ℝ) : ℝ := -2 * x^2 * y

theorem cubic_monomial_properties (x y : ℝ) :
  ∃ (a b c : ℕ), a + b + c = 3 ∧ cubic_monomial x y = -2 * x^a * y^b := by
  sorry

end NUMINAMATH_CALUDE_cubic_monomial_properties_l1911_191186


namespace NUMINAMATH_CALUDE_last_painted_cell_l1911_191160

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def is_painted (n : ℕ) : Prop := ∃ k : ℕ, n = triangular_number k

def covers_all_columns (n : ℕ) : Prop :=
  ∀ i : ℕ, i > 0 → i ≤ 8 → ∃ k : ℕ, k ≤ n ∧ is_painted k ∧ k % 8 = i

theorem last_painted_cell :
  ∃ n : ℕ, n = 120 ∧ is_painted n ∧ covers_all_columns n ∧
  ∀ m : ℕ, m < n → ¬(covers_all_columns m) :=
sorry

end NUMINAMATH_CALUDE_last_painted_cell_l1911_191160


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l1911_191127

theorem angle_sum_around_point (x : ℝ) : 
  (3 * x + 7 * x + 4 * x + x = 360) → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l1911_191127


namespace NUMINAMATH_CALUDE_largest_root_ratio_l1911_191187

def f (x : ℝ) : ℝ := 1 - x - 4*x^2 + x^4

def g (x : ℝ) : ℝ := 16 - 8*x - 16*x^2 + x^4

def largest_root (p : ℝ → ℝ) : ℝ := sorry

theorem largest_root_ratio : 
  largest_root g / largest_root f = 2 := by sorry

end NUMINAMATH_CALUDE_largest_root_ratio_l1911_191187


namespace NUMINAMATH_CALUDE_first_interest_rate_is_five_percent_l1911_191189

-- Define the total amount, amounts lent at each rate, and the known interest rate
def total_amount : ℝ := 2500
def amount_first_rate : ℝ := 2000
def amount_second_rate : ℝ := total_amount - amount_first_rate
def second_rate : ℝ := 6

-- Define the total yearly annual income
def total_income : ℝ := 130

-- Define the first interest rate as a variable
variable (first_rate : ℝ)

-- Theorem statement
theorem first_interest_rate_is_five_percent :
  (amount_first_rate * first_rate / 100 + amount_second_rate * second_rate / 100 = total_income) →
  first_rate = 5 := by
sorry

end NUMINAMATH_CALUDE_first_interest_rate_is_five_percent_l1911_191189


namespace NUMINAMATH_CALUDE_greater_number_proof_l1911_191173

theorem greater_number_proof (a b : ℝ) (h_sum : a + b = 36) (h_diff : a - b = 8) (h_greater : a > b) : a = 22 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l1911_191173


namespace NUMINAMATH_CALUDE_eighth_term_is_22_l1911_191118

/-- An arithmetic sequence with a_1 = 1 and sum of first 5 terms = 35 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 1 + a 2 + a 3 + a 4 + a 5 = 35)

/-- The 8th term of the sequence is 22 -/
theorem eighth_term_is_22 (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  a 8 = 22 := by
  sorry


end NUMINAMATH_CALUDE_eighth_term_is_22_l1911_191118


namespace NUMINAMATH_CALUDE_chimps_in_old_cage_l1911_191153

/-- The number of chimps staying in the old cage is equal to the total number of chimps minus the number of chimps being moved. -/
theorem chimps_in_old_cage (total_chimps moving_chimps : ℕ) :
  total_chimps ≥ moving_chimps →
  total_chimps - moving_chimps = total_chimps - moving_chimps :=
by
  sorry

#check chimps_in_old_cage 45 18

end NUMINAMATH_CALUDE_chimps_in_old_cage_l1911_191153


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1911_191195

-- Define the universal set U
def U : Finset ℕ := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset ℕ := {1, 4}

-- Define set N
def N : Finset ℕ := {1, 3, 5}

-- Theorem statement
theorem intersection_complement_equality :
  N ∩ (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1911_191195


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1911_191166

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℝ) (d S : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  (∀ n, a n > 0) →  -- positivity condition
  (∀ n, S * n = (n / 2) * (2 * a 1 + (n - 1) * d)) →  -- sum formula
  (a 2) * (a 2 + S * 5) = (S * 3) ^ 2 →  -- geometric sequence condition
  d / a 1 = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1911_191166


namespace NUMINAMATH_CALUDE_sine_periodicity_l1911_191102

theorem sine_periodicity (m : ℝ) (h : Real.sin (5.1 * π / 180) = m) :
  Real.sin (365.1 * π / 180) = m := by
  sorry

end NUMINAMATH_CALUDE_sine_periodicity_l1911_191102


namespace NUMINAMATH_CALUDE_not_prime_2011_2111_plus_2500_l1911_191184

theorem not_prime_2011_2111_plus_2500 : ¬ Nat.Prime (2011 * 2111 + 2500) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_2011_2111_plus_2500_l1911_191184


namespace NUMINAMATH_CALUDE_no_real_solution_quadratic_l1911_191142

theorem no_real_solution_quadratic : ¬ ∃ x : ℝ, x^2 + 3*x + 3 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_no_real_solution_quadratic_l1911_191142


namespace NUMINAMATH_CALUDE_great_pyramid_sum_height_width_l1911_191168

/-- The Great Pyramid of Giza's dimensions -/
def great_pyramid (H W : ℕ) : Prop :=
  H = 500 + 20 ∧ W = H + 234

/-- Theorem: The sum of the height and width of the Great Pyramid of Giza is 1274 feet -/
theorem great_pyramid_sum_height_width :
  ∀ H W : ℕ, great_pyramid H W → H + W = 1274 :=
by
  sorry

#check great_pyramid_sum_height_width

end NUMINAMATH_CALUDE_great_pyramid_sum_height_width_l1911_191168


namespace NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l1911_191124

def alphabet_size : ℕ := 26
def unique_letters_in_mathematics : ℕ := 8

theorem probability_of_letter_in_mathematics :
  (unique_letters_in_mathematics : ℚ) / (alphabet_size : ℚ) = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l1911_191124


namespace NUMINAMATH_CALUDE_samantha_routes_count_l1911_191165

/-- Represents a location on a grid -/
structure Location :=
  (x : Int) (y : Int)

/-- Represents Central Park -/
structure CentralPark :=
  (sw : Location) (ne : Location)

/-- Calculates the number of shortest paths between two locations on a grid -/
def gridPaths (start finish : Location) : Nat :=
  let dx := (finish.x - start.x).natAbs
  let dy := (finish.y - start.y).natAbs
  Nat.choose (dx + dy) dx

/-- The number of diagonal paths through Central Park -/
def parkPaths : Nat := 2

/-- Theorem stating the number of shortest routes from Samantha's house to her school -/
theorem samantha_routes_count (park : CentralPark) 
  (home : Location) 
  (school : Location) 
  (home_to_sw : home.x = park.sw.x - 3 ∧ home.y = park.sw.y - 2)
  (school_to_ne : school.x = park.ne.x + 3 ∧ school.y = park.ne.y + 3) :
  gridPaths home park.sw * parkPaths * gridPaths park.ne school = 400 := by
  sorry

end NUMINAMATH_CALUDE_samantha_routes_count_l1911_191165


namespace NUMINAMATH_CALUDE_minimal_adjective_f_25_l1911_191178

/-- A function g: ℤ → ℤ is adjective if g(m) + g(n) > max(m², n²) for any integers m and n -/
def Adjective (g : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, g m + g n > max (m ^ 2) (n ^ 2)

/-- The sum of f(1) to f(30) -/
def SumF (f : ℤ → ℤ) : ℤ :=
  (Finset.range 30).sum (fun i => f (i + 1))

/-- f is an adjective function that minimizes SumF -/
def IsMinimalAdjective (f : ℤ → ℤ) : Prop :=
  Adjective f ∧ ∀ g : ℤ → ℤ, Adjective g → SumF f ≤ SumF g

theorem minimal_adjective_f_25 (f : ℤ → ℤ) (hf : IsMinimalAdjective f) : f 25 ≥ 498 := by
  sorry

end NUMINAMATH_CALUDE_minimal_adjective_f_25_l1911_191178


namespace NUMINAMATH_CALUDE_sphere_surface_area_equals_volume_l1911_191174

/-- For a sphere with radius 3, its surface area is numerically equal to its volume. -/
theorem sphere_surface_area_equals_volume :
  let r : ℝ := 3
  let surface_area : ℝ := 4 * Real.pi * r^2
  let volume : ℝ := (4/3) * Real.pi * r^3
  surface_area = volume := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_equals_volume_l1911_191174


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l1911_191144

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (6 * x) % 35 = 17 % 35 ∧
  ∀ (y : ℕ), y > 0 ∧ (6 * y) % 35 = 17 % 35 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l1911_191144


namespace NUMINAMATH_CALUDE_problem_solution_l1911_191112

theorem problem_solution (w y z x : ℕ) 
  (hw : w = 50)
  (hz : z = 2 * w + 3)
  (hy : y = z + 5)
  (hx : x = 2 * y + 4) :
  x = 220 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1911_191112


namespace NUMINAMATH_CALUDE_loop_requirement_correct_l1911_191161

/-- Represents a mathematical operation that may or may not require a loop statement --/
inductive MathOperation
  | GeometricSum
  | CompareNumbers
  | PiecewiseFunction
  | LargestNaturalNumber

/-- Determines if a given mathematical operation requires a loop statement --/
def requires_loop (op : MathOperation) : Prop :=
  match op with
  | MathOperation.GeometricSum => true
  | MathOperation.CompareNumbers => false
  | MathOperation.PiecewiseFunction => false
  | MathOperation.LargestNaturalNumber => true

theorem loop_requirement_correct :
  (requires_loop MathOperation.GeometricSum) ∧
  (¬requires_loop MathOperation.CompareNumbers) ∧
  (¬requires_loop MathOperation.PiecewiseFunction) ∧
  (requires_loop MathOperation.LargestNaturalNumber) :=
by sorry

#check loop_requirement_correct

end NUMINAMATH_CALUDE_loop_requirement_correct_l1911_191161


namespace NUMINAMATH_CALUDE_triangle_ABC_is_obtuse_l1911_191190

theorem triangle_ABC_is_obtuse (A B C : Real) (hA : A = 10) (hB : B = 60) 
  (hsum : A + B + C = 180) : C > 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_is_obtuse_l1911_191190


namespace NUMINAMATH_CALUDE_paper_count_l1911_191116

theorem paper_count (initial_math initial_science used_math used_science received_math given_science : ℕ) :
  initial_math = 220 →
  initial_science = 150 →
  used_math = 95 →
  used_science = 68 →
  received_math = 30 →
  given_science = 15 →
  (initial_math - used_math + received_math) + (initial_science - used_science - given_science) = 222 := by
  sorry

end NUMINAMATH_CALUDE_paper_count_l1911_191116


namespace NUMINAMATH_CALUDE_unique_remainder_modulo_8_and_13_l1911_191121

theorem unique_remainder_modulo_8_and_13 : 
  ∃! n : ℕ, n < 180 ∧ n % 8 = 2 ∧ n % 13 = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_remainder_modulo_8_and_13_l1911_191121


namespace NUMINAMATH_CALUDE_power_windows_count_l1911_191179

theorem power_windows_count (total : ℕ) (power_steering : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 65)
  (h2 : power_steering = 45)
  (h3 : both = 17)
  (h4 : neither = 12) :
  total - neither - (power_steering - both) = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_windows_count_l1911_191179


namespace NUMINAMATH_CALUDE_abs_sum_equals_five_l1911_191158

theorem abs_sum_equals_five (a b c : ℝ) 
  (h1 : a^2 - b*c = 14)
  (h2 : b^2 - c*a = 14)
  (h3 : c^2 - a*b = -3) :
  |a + b + c| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_equals_five_l1911_191158


namespace NUMINAMATH_CALUDE_group_size_l1911_191131

theorem group_size (total : ℕ) 
  (h1 : (total : ℚ) / 5 = (0.12 * total + 64 : ℚ)) : total = 800 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l1911_191131


namespace NUMINAMATH_CALUDE_route_down_is_24_miles_l1911_191177

/-- A hiking trip up and down a mountain -/
structure HikingTrip where
  rate_up : ℝ
  time_up : ℝ
  rate_down_factor : ℝ

/-- The length of the route down the mountain -/
def route_down_length (trip : HikingTrip) : ℝ :=
  trip.rate_up * trip.rate_down_factor * trip.time_up

/-- Theorem: The length of the route down the mountain is 24 miles -/
theorem route_down_is_24_miles (trip : HikingTrip)
  (h1 : trip.rate_up = 8)
  (h2 : trip.time_up = 2)
  (h3 : trip.rate_down_factor = 1.5) :
  route_down_length trip = 24 := by
  sorry

end NUMINAMATH_CALUDE_route_down_is_24_miles_l1911_191177


namespace NUMINAMATH_CALUDE_total_shoes_needed_l1911_191154

def num_dogs : ℕ := 3
def num_cats : ℕ := 2
def num_ferrets : ℕ := 1
def paws_per_animal : ℕ := 4

theorem total_shoes_needed : 
  (num_dogs + num_cats + num_ferrets) * paws_per_animal = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_needed_l1911_191154


namespace NUMINAMATH_CALUDE_division_sum_equals_two_l1911_191108

theorem division_sum_equals_two : (101 : ℚ) / 101 + (99 : ℚ) / 99 = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_sum_equals_two_l1911_191108


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l1911_191113

theorem min_values_xy_and_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 9/y = 1) : xy ≥ 36 ∧ x + y ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l1911_191113


namespace NUMINAMATH_CALUDE_turtle_race_time_difference_l1911_191197

theorem turtle_race_time_difference (greta_time gloria_time : ℕ) 
  (h1 : greta_time = 6)
  (h2 : gloria_time = 8)
  (h3 : gloria_time = 2 * (gloria_time / 2)) :
  greta_time - (gloria_time / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_turtle_race_time_difference_l1911_191197


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_2023_l1911_191103

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_2023 :
  units_digit (factorial_sum 2023) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_2023_l1911_191103


namespace NUMINAMATH_CALUDE_library_book_count_l1911_191194

/-- The number of books in the library after taking out and bringing back some books -/
def final_book_count (initial : ℕ) (taken_out : ℕ) (brought_back : ℕ) : ℕ :=
  initial - taken_out + brought_back

/-- Theorem: Given 336 initial books, 124 taken out, and 22 brought back, there are 234 books now -/
theorem library_book_count : final_book_count 336 124 22 = 234 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l1911_191194


namespace NUMINAMATH_CALUDE_right_triangle_legs_l1911_191170

theorem right_triangle_legs (a Δ : ℝ) (ha : a > 0) (hΔ : Δ > 0) :
  ∃ x y : ℝ,
    x > 0 ∧ y > 0 ∧
    x^2 + y^2 = a^2 ∧
    x * y / 2 = Δ ∧
    x = (Real.sqrt (a^2 + 4*Δ) + Real.sqrt (a^2 - 4*Δ)) / 2 ∧
    y = (Real.sqrt (a^2 + 4*Δ) - Real.sqrt (a^2 - 4*Δ)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l1911_191170


namespace NUMINAMATH_CALUDE_smallest_gcd_with_lcm_condition_l1911_191114

theorem smallest_gcd_with_lcm_condition (x y : ℕ) 
  (h : Nat.lcm x y = (x - y)^2) : 
  Nat.gcd x y ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_with_lcm_condition_l1911_191114


namespace NUMINAMATH_CALUDE_field_trip_adults_l1911_191172

/-- Field trip problem -/
theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) :
  van_capacity = 5 →
  num_students = 12 →
  num_vans = 3 →
  (num_vans * van_capacity - num_students : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_adults_l1911_191172


namespace NUMINAMATH_CALUDE_misread_number_calculation_l1911_191138

theorem misread_number_calculation (n : ℕ) (initial_avg correct_avg wrong_num : ℚ) 
  (h1 : n = 10)
  (h2 : initial_avg = 23)
  (h3 : correct_avg = 24)
  (h4 : wrong_num = 26) : 
  ∃ (actual_num : ℚ), 
    (n : ℚ) * correct_avg - (n : ℚ) * initial_avg = actual_num - wrong_num ∧ 
    actual_num = 36 := by
  sorry

end NUMINAMATH_CALUDE_misread_number_calculation_l1911_191138


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_equals_one_two_l1911_191171

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define set P
def P : Finset Nat := {1, 2, 3, 4}

-- Define set Q
def Q : Finset Nat := {3, 4, 5}

-- Theorem statement
theorem intersection_P_complement_Q_equals_one_two :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_equals_one_two_l1911_191171


namespace NUMINAMATH_CALUDE_dividend_calculation_l1911_191157

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 16)
  (h_quotient : quotient = 9)
  (h_remainder : remainder = 5) :
  divisor * quotient + remainder = 149 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1911_191157


namespace NUMINAMATH_CALUDE_light_travel_distance_l1911_191120

/-- The distance light travels in one year in a vacuum (in miles) -/
def light_speed_vacuum : ℝ := 5870000000000

/-- The factor by which light speed is reduced in the medium -/
def speed_reduction_factor : ℝ := 2

/-- The number of years we're considering -/
def years : ℝ := 1000

/-- The theorem stating the distance light travels in the given conditions -/
theorem light_travel_distance :
  (light_speed_vacuum / speed_reduction_factor) * years = 2935 * (10 ^ 12) := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l1911_191120


namespace NUMINAMATH_CALUDE_simplify_expression_l1911_191163

theorem simplify_expression (x : ℝ) :
  4 * x - 8 * x^2 + 10 - (5 - 4 * x + 8 * x^2) = -16 * x^2 + 8 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1911_191163


namespace NUMINAMATH_CALUDE_ratio_calculation_l1911_191155

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l1911_191155


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1911_191182

theorem units_digit_of_expression : ∃ n : ℕ, (9 * 19 * 1989 - 9^3) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1911_191182


namespace NUMINAMATH_CALUDE_adam_ferris_wheel_cost_l1911_191162

/-- The amount Adam spent on the ferris wheel ride -/
def ferris_wheel_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * ticket_price

/-- Theorem: Adam spent 81 dollars on the ferris wheel ride -/
theorem adam_ferris_wheel_cost :
  ferris_wheel_cost 13 4 9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_adam_ferris_wheel_cost_l1911_191162


namespace NUMINAMATH_CALUDE_cos_sum_fifteenths_l1911_191126

theorem cos_sum_fifteenths : 
  Real.cos (4 * Real.pi / 15) + Real.cos (10 * Real.pi / 15) + Real.cos (14 * Real.pi / 15) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_fifteenths_l1911_191126


namespace NUMINAMATH_CALUDE_max_min_values_l1911_191125

theorem max_min_values (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 = 2) :
  (a + b + c ≤ Real.sqrt 6) ∧
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 * Real.sqrt 6 / 4) := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_l1911_191125


namespace NUMINAMATH_CALUDE_bobby_total_consumption_l1911_191141

def bobby_consumption (initial_candy : ℕ) (additional_candy : ℕ) (candy_fraction : ℚ) 
                      (chocolate : ℕ) (chocolate_fraction : ℚ) : ℚ :=
  initial_candy + candy_fraction * additional_candy + chocolate_fraction * chocolate

theorem bobby_total_consumption : 
  bobby_consumption 28 42 (3/4) 63 (1/2) = 91 := by
  sorry

end NUMINAMATH_CALUDE_bobby_total_consumption_l1911_191141


namespace NUMINAMATH_CALUDE_rhombus_area_l1911_191105

/-- The area of a rhombus with diagonals of 6cm and 8cm is 24cm². -/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) :
  (1 / 2) * d1 * d2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1911_191105


namespace NUMINAMATH_CALUDE_spring_sales_five_million_l1911_191109

/-- Represents the annual pizza sales of a restaurant in millions -/
def annual_sales : ℝ := 20

/-- Represents the winter pizza sales of the restaurant in millions -/
def winter_sales : ℝ := 4

/-- Represents the percentage of annual sales that occur in winter -/
def winter_percentage : ℝ := 0.20

/-- Represents the percentage of annual sales that occur in summer -/
def summer_percentage : ℝ := 0.30

/-- Represents the percentage of annual sales that occur in fall -/
def fall_percentage : ℝ := 0.25

/-- Theorem stating that spring sales are 5 million pizzas -/
theorem spring_sales_five_million :
  winter_sales = winter_percentage * annual_sales →
  ∃ (spring_percentage : ℝ),
    spring_percentage + winter_percentage + summer_percentage + fall_percentage = 1 ∧
    spring_percentage * annual_sales = 5 := by
  sorry

end NUMINAMATH_CALUDE_spring_sales_five_million_l1911_191109


namespace NUMINAMATH_CALUDE_work_completion_time_l1911_191107

/-- The time required for x to complete a work given the combined time of x and y, and the time for y alone. -/
theorem work_completion_time (combined_time y_time : ℝ) (h1 : combined_time > 0) (h2 : y_time > 0) :
  let x_rate := 1 / combined_time - 1 / y_time
  x_rate > 0 → 1 / x_rate = y_time := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1911_191107


namespace NUMINAMATH_CALUDE_jadens_estimate_l1911_191191

theorem jadens_estimate (p q δ γ : ℝ) 
  (h1 : p > q) 
  (h2 : q > 0) 
  (h3 : δ > γ) 
  (h4 : γ > 0) : 
  (p + δ) - (q - γ) > p - q := by
  sorry

end NUMINAMATH_CALUDE_jadens_estimate_l1911_191191


namespace NUMINAMATH_CALUDE_vectors_perpendicular_if_sum_norm_eq_diff_norm_l1911_191130

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vectors_perpendicular_if_sum_norm_eq_diff_norm 
  (a b : ℝ × ℝ) (h : ‖a + b‖ = ‖a - b‖) : 
  angle_between_vectors a b = π / 2 := by sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_if_sum_norm_eq_diff_norm_l1911_191130


namespace NUMINAMATH_CALUDE_cube_root_approximation_l1911_191119

-- Define k as the real cube root of 2
noncomputable def k : ℝ := Real.rpow 2 (1/3)

-- Define the inequality function
def inequality (A B C a b c : ℤ) (x : ℚ) : Prop :=
  x ≥ 0 → |((A * x^2 + B * x + C) / (a * x^2 + b * x + c) : ℝ) - k| < |x - k|

-- State the theorem
theorem cube_root_approximation :
  ∀ x : ℚ, inequality 2 2 2 1 2 2 x := by sorry

end NUMINAMATH_CALUDE_cube_root_approximation_l1911_191119


namespace NUMINAMATH_CALUDE_profit_ratio_theorem_l1911_191104

/-- Represents a partner's investment details -/
structure Partner where
  investment : ℚ
  time : ℕ

/-- Calculates the profit factor for a partner -/
def profitFactor (p : Partner) : ℚ :=
  p.investment * p.time

/-- Theorem: Given the investment ratio and time periods, prove the profit ratio -/
theorem profit_ratio_theorem (p q : Partner) 
  (h1 : p.investment / q.investment = 7 / 5)
  (h2 : p.time = 20)
  (h3 : q.time = 40) :
  profitFactor p / profitFactor q = 7 / 10 := by
  sorry

#check profit_ratio_theorem

end NUMINAMATH_CALUDE_profit_ratio_theorem_l1911_191104


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l1911_191133

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 16 = 48 → Nat.gcd n 16 = 18 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l1911_191133


namespace NUMINAMATH_CALUDE_expression_evaluation_l1911_191169

theorem expression_evaluation : 
  Real.sqrt 3 + 3 + (1 / (Real.sqrt 3 + 3))^2 + 1 / (3 - Real.sqrt 3) = Real.sqrt 3 + 3 + 5/6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1911_191169


namespace NUMINAMATH_CALUDE_largest_integer_not_exceeding_a_n_l1911_191117

/-- Sequence a_n defined recursively -/
def a (a₀ : ℕ) : ℕ → ℚ
  | 0 => a₀
  | n + 1 => (a a₀ n)^2 / ((a a₀ n) + 1)

/-- Theorem stating the largest integer not exceeding a_n is a - n -/
theorem largest_integer_not_exceeding_a_n (a₀ : ℕ) (n : ℕ) 
  (h : n ≤ a₀/2 + 1) : 
  ⌊a a₀ n⌋ = a₀ - n := by sorry

end NUMINAMATH_CALUDE_largest_integer_not_exceeding_a_n_l1911_191117


namespace NUMINAMATH_CALUDE_percentage_of_liars_l1911_191192

theorem percentage_of_liars (truth_speakers : ℝ) (both_speakers : ℝ) (truth_or_lie_prob : ℝ) :
  truth_speakers = 0.3 →
  both_speakers = 0.1 →
  truth_or_lie_prob = 0.4 →
  ∃ (lie_speakers : ℝ), lie_speakers = 0.2 ∧ 
    truth_or_lie_prob = truth_speakers + lie_speakers - both_speakers :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_liars_l1911_191192


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l1911_191128

theorem circle_area_from_circumference (c : ℝ) (h : c = 24) :
  let r := c / (2 * Real.pi)
  (Real.pi * r * r) = 144 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l1911_191128


namespace NUMINAMATH_CALUDE_base5_division_theorem_l1911_191137

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem base5_division_theorem :
  let dividend := [4, 3, 0, 2]  -- 2034₅ in reverse order
  let divisor := [3, 2]         -- 23₅ in reverse order
  let quotient := [0, 4]        -- 40₅ in reverse order
  (base5ToBase10 dividend) / (base5ToBase10 divisor) = base5ToBase10 quotient :=
by sorry

end NUMINAMATH_CALUDE_base5_division_theorem_l1911_191137


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l1911_191185

/-- Given a box with 10 balls, where 1 is yellow, 3 are green, and the rest are red,
    the probability of randomly drawing a red ball is 3/5. -/
theorem probability_of_red_ball (total_balls : ℕ) (yellow_balls : ℕ) (green_balls : ℕ) :
  total_balls = 10 →
  yellow_balls = 1 →
  green_balls = 3 →
  (total_balls - yellow_balls - green_balls : ℚ) / total_balls = 3 / 5 := by
  sorry

#check probability_of_red_ball

end NUMINAMATH_CALUDE_probability_of_red_ball_l1911_191185


namespace NUMINAMATH_CALUDE_f_nonnegative_and_a_range_f_unique_zero_l1911_191143

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 1 / (x + a)

theorem f_nonnegative_and_a_range (a : ℝ) (h : a > 0) :
  (∀ x > 0, f a x ≥ 0) ∧ a ≥ 1 := by sorry

theorem f_unique_zero (a : ℝ) (h : 0 < a ∧ a ≤ 2/3) :
  ∃! x, x > -a ∧ f a x = 0 := by sorry

end NUMINAMATH_CALUDE_f_nonnegative_and_a_range_f_unique_zero_l1911_191143


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l1911_191176

theorem sum_of_fourth_powers (x y : ℕ+) : x^4 + y^4 = 4721 → x + y = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l1911_191176


namespace NUMINAMATH_CALUDE_m_range_when_p_false_l1911_191100

theorem m_range_when_p_false :
  (¬∀ x : ℝ, ∃ m : ℝ, 4*x - 2*x + 1 + m = 0) →
  {m : ℝ | ∃ x : ℝ, 4*x - 2*x + 1 + m ≠ 0} = Set.Iio 1 := by
  sorry

end NUMINAMATH_CALUDE_m_range_when_p_false_l1911_191100
