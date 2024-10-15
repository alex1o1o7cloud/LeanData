import Mathlib

namespace NUMINAMATH_CALUDE_dividing_trapezoid_mn_length_l542_54285

/-- A trapezoid with bases a and b, and a segment MN parallel to the bases that divides the area in half -/
structure DividingTrapezoid (a b : ℝ) where
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (mn_length : ℝ)
  (mn_divides_area : mn_length ^ 2 = (a ^ 2 + b ^ 2) / 2)

/-- The length of MN in a DividingTrapezoid is √((a² + b²) / 2) -/
theorem dividing_trapezoid_mn_length (a b : ℝ) (t : DividingTrapezoid a b) :
  t.mn_length = Real.sqrt ((a ^ 2 + b ^ 2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_dividing_trapezoid_mn_length_l542_54285


namespace NUMINAMATH_CALUDE_total_waiting_time_bounds_l542_54236

/-- 
Represents the total waiting time for a queue with Slowpokes and Quickies.
m: number of Slowpokes
n: number of Quickies
a: time taken by a Quickie
b: time taken by a Slowpoke
-/
def TotalWaitingTime (m n : ℕ) (a b : ℝ) : Prop :=
  let total := m + n
  ∀ (t_min t_max t_exp : ℝ),
    b > a →
    t_min = a * (n.choose 2) + a * m * n + b * (m.choose 2) →
    t_max = a * (n.choose 2) + b * m * n + b * (m.choose 2) →
    t_exp = (total.choose 2 : ℝ) * (b * m + a * n) / total →
    (t_min ≤ t_exp ∧ t_exp ≤ t_max)

theorem total_waiting_time_bounds {m n : ℕ} {a b : ℝ} :
  TotalWaitingTime m n a b :=
sorry

end NUMINAMATH_CALUDE_total_waiting_time_bounds_l542_54236


namespace NUMINAMATH_CALUDE_principal_amount_proof_l542_54269

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Problem statement --/
theorem principal_amount_proof :
  let final_amount : ℝ := 8820
  let rate : ℝ := 0.05
  let time : ℕ := 2
  ∃ (principal : ℝ), principal = 8000 ∧ compound_interest principal rate time = final_amount := by
sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l542_54269


namespace NUMINAMATH_CALUDE_refrigerator_price_calculation_l542_54286

/-- The purchase price of the refrigerator in rupees -/
def refrigerator_price : ℝ := 15000

/-- The purchase price of the mobile phone in rupees -/
def mobile_price : ℝ := 8000

/-- The loss percentage on the refrigerator -/
def refrigerator_loss_percent : ℝ := 0.03

/-- The profit percentage on the mobile phone -/
def mobile_profit_percent : ℝ := 0.10

/-- The overall profit in rupees -/
def overall_profit : ℝ := 350

theorem refrigerator_price_calculation :
  refrigerator_price * (1 - refrigerator_loss_percent) +
  mobile_price * (1 + mobile_profit_percent) =
  refrigerator_price + mobile_price + overall_profit := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_price_calculation_l542_54286


namespace NUMINAMATH_CALUDE_find_c_l542_54266

theorem find_c (p q : ℝ → ℝ) (c : ℝ) : 
  (∀ x, p x = 4 * x - 3) →
  (∀ x, q x = 5 * x - c) →
  p (q 3) = 53 →
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_find_c_l542_54266


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l542_54296

-- Define set A
def A : Set ℝ := {x | |x| < 1}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Ioo (-1 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l542_54296


namespace NUMINAMATH_CALUDE_smallest_prime_divides_infinitely_many_and_all_l542_54246

def a (n : ℕ) : ℕ := 4^(2*n+1) + 3^(n+2)

def is_divisible_by (m n : ℕ) : Prop := ∃ k, m = n * k

def divides_infinitely_many (p : ℕ) : Prop :=
  ∀ N, ∃ n ≥ N, is_divisible_by (a n) p

def divides_all (p : ℕ) : Prop :=
  ∀ n, n ≥ 1 → is_divisible_by (a n) p

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m, 1 < m → m < p → ¬(is_divisible_by p m)

theorem smallest_prime_divides_infinitely_many_and_all :
  ∃ (p q : ℕ),
    is_prime p ∧
    is_prime q ∧
    divides_infinitely_many p ∧
    divides_all q ∧
    (∀ p', is_prime p' → divides_infinitely_many p' → p ≤ p') ∧
    (∀ q', is_prime q' → divides_all q' → q ≤ q') ∧
    p = 5 ∧
    q = 13 :=
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divides_infinitely_many_and_all_l542_54246


namespace NUMINAMATH_CALUDE_product_inequality_l542_54268

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1) * (b + 1) * (a + c) * (b + c) ≥ 16 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l542_54268


namespace NUMINAMATH_CALUDE_locus_of_point_l542_54252

/-- Given three lines in a plane not passing through the origin, prove the locus of a point P
    satisfying certain conditions. -/
theorem locus_of_point (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) :
  let l₁ : ℝ × ℝ → Prop := λ (x, y) ↦ a₁ * x + b₁ * y + c₁ = 0
  let l₂ : ℝ × ℝ → Prop := λ (x, y) ↦ a₂ * x + b₂ * y + c₂ = 0
  let l₃ : ℝ × ℝ → Prop := λ (x, y) ↦ a₃ * x + b₃ * y + c₃ = 0
  let origin : ℝ × ℝ := (0, 0)
  ∀ (l : Set (ℝ × ℝ)) (A B C : ℝ × ℝ),
    (∀ p ∈ l, ∃ t : ℝ, p = (t * (A.1 - origin.1), t * (A.2 - origin.2))) →
    l₁ A ∧ l₂ B ∧ l₃ C →
    A ∈ l ∧ B ∈ l ∧ C ∈ l →
    (∀ P ∈ l, P ≠ origin →
      let ρ₁ := Real.sqrt ((A.1 - origin.1)^2 + (A.2 - origin.2)^2)
      let ρ₂ := Real.sqrt ((B.1 - origin.1)^2 + (B.2 - origin.2)^2)
      let ρ₃ := Real.sqrt ((C.1 - origin.1)^2 + (C.2 - origin.2)^2)
      let ρ  := Real.sqrt ((P.1 - origin.1)^2 + (P.2 - origin.2)^2)
      1 / ρ₁ + 1 / ρ₂ + 1 / ρ₃ = 1 / ρ) →
    ∀ (x y : ℝ),
      (x, y) ∈ l ↔ (a₁ / c₁ + a₂ / c₂ + a₃ / c₃) * x + (b₁ / c₁ + b₂ / c₂ + b₃ / c₃) * y + 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_locus_of_point_l542_54252


namespace NUMINAMATH_CALUDE_total_options_is_twenty_l542_54279

/-- The number of high-speed trains from location A to location B -/
def num_trains : ℕ := 5

/-- The number of ferries from location B to location C -/
def num_ferries : ℕ := 4

/-- The total number of travel options from location A to location C -/
def total_options : ℕ := num_trains * num_ferries

/-- Theorem stating that the total number of travel options is 20 -/
theorem total_options_is_twenty : total_options = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_options_is_twenty_l542_54279


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_of_geometric_sequences_l542_54251

theorem sum_of_common_ratios_of_geometric_sequences 
  (k a₂ a₃ b₂ b₃ p r : ℝ) 
  (h1 : k ≠ 0)
  (h2 : p ≠ 1)
  (h3 : r ≠ 1)
  (h4 : p ≠ r)
  (h5 : a₂ = k * p)
  (h6 : a₃ = k * p^2)
  (h7 : b₂ = k * r)
  (h8 : b₃ = k * r^2)
  (h9 : a₃ - b₃ = 3 * (a₂ - b₂)) :
  p + r = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_of_geometric_sequences_l542_54251


namespace NUMINAMATH_CALUDE_intersection_point_value_l542_54225

theorem intersection_point_value (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (x - a + 2) * (x^2 - a*x - 2) ≥ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_value_l542_54225


namespace NUMINAMATH_CALUDE_miss_both_mutually_exclusive_not_contradictory_l542_54271

-- Define the sample space for two shots
inductive ShotOutcome
| HitBoth
| HitFirst
| HitSecond
| MissBoth

-- Define the events
def hit_exactly_once (outcome : ShotOutcome) : Prop :=
  outcome = ShotOutcome.HitFirst ∨ outcome = ShotOutcome.HitSecond

def miss_both (outcome : ShotOutcome) : Prop :=
  outcome = ShotOutcome.MissBoth

-- Theorem stating that "Miss both times" is mutually exclusive but not contradictory to "hit exactly once"
theorem miss_both_mutually_exclusive_not_contradictory :
  (∀ outcome : ShotOutcome, ¬(hit_exactly_once outcome ∧ miss_both outcome)) ∧
  (∃ outcome : ShotOutcome, hit_exactly_once outcome ∨ miss_both outcome) :=
sorry

end NUMINAMATH_CALUDE_miss_both_mutually_exclusive_not_contradictory_l542_54271


namespace NUMINAMATH_CALUDE_unit_digit_of_x_is_six_l542_54289

theorem unit_digit_of_x_is_six :
  let x : ℤ := (-2)^1988
  ∃ k : ℤ, x = 10 * k + 6 :=
by sorry

end NUMINAMATH_CALUDE_unit_digit_of_x_is_six_l542_54289


namespace NUMINAMATH_CALUDE_pricing_equation_l542_54213

/-- 
Given an item with:
- cost price x (in yuan)
- markup percentage m (as a decimal)
- discount percentage d (as a decimal)
- final selling price s (in yuan)

This theorem states that the equation relating these values is:
x * (1 + m) * (1 - d) = s
-/
theorem pricing_equation (x m d s : ℝ) 
  (markup : m = 0.3)
  (discount : d = 0.2)
  (selling_price : s = 2080) :
  x * (1 + m) * (1 - d) = s :=
sorry

end NUMINAMATH_CALUDE_pricing_equation_l542_54213


namespace NUMINAMATH_CALUDE_choir_arrangement_l542_54250

theorem choir_arrangement (n : ℕ) : 
  (∃ k : ℕ, n = 9 * k) ∧ 
  (∃ k : ℕ, n = 10 * k) ∧ 
  (∃ k : ℕ, n = 11 * k) ↔ 
  n ≥ 990 ∧ n % 990 = 0 :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_l542_54250


namespace NUMINAMATH_CALUDE_smallest_circle_theorem_l542_54200

/-- Given two circles in the xy-plane, this function returns the equation of the circle 
    with the smallest area that passes through their intersection points. -/
def smallest_circle_through_intersections (c1 c2 : ℝ × ℝ → Prop) : ℝ × ℝ → Prop :=
  sorry

/-- The first given circle -/
def circle1 (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 + 4*p.1 + p.2 = -1

/-- The second given circle -/
def circle2 (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 + 2*p.1 + 2*p.2 + 1 = 0

/-- The resulting circle with the smallest area -/
def result_circle (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 + (6/5)*p.1 + (12/5)*p.2 + 1 = 0

theorem smallest_circle_theorem :
  smallest_circle_through_intersections circle1 circle2 = result_circle :=
sorry

end NUMINAMATH_CALUDE_smallest_circle_theorem_l542_54200


namespace NUMINAMATH_CALUDE_kia_vehicles_count_l542_54212

theorem kia_vehicles_count (total : ℕ) (dodge : ℕ) (hyundai : ℕ) (kia : ℕ) : 
  total = 400 →
  dodge = total / 2 →
  hyundai = dodge / 2 →
  kia = total - (dodge + hyundai) →
  kia = 100 := by
  sorry

end NUMINAMATH_CALUDE_kia_vehicles_count_l542_54212


namespace NUMINAMATH_CALUDE_sin_cos_identity_l542_54207

theorem sin_cos_identity (x : ℝ) : 
  Real.sin x * Real.cos x + Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3 / 2 = 
  Real.sin (2 * x + π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l542_54207


namespace NUMINAMATH_CALUDE_triangle_altitude_specific_triangle_altitude_l542_54242

/-- The altitude of a triangle given its area and base -/
theorem triangle_altitude (area : ℝ) (base : ℝ) (h_area : area > 0) (h_base : base > 0) :
  area = (1/2) * base * (2 * area / base) :=
by sorry

/-- The altitude of a specific triangle with area 800 and base 40 -/
theorem specific_triangle_altitude :
  let area : ℝ := 800
  let base : ℝ := 40
  let altitude : ℝ := 2 * area / base
  altitude = 40 :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_specific_triangle_altitude_l542_54242


namespace NUMINAMATH_CALUDE_b_not_two_l542_54201

theorem b_not_two (b : ℝ) (h : ∀ x : ℝ, x ∈ Set.Icc 0 1 → |x + b| ≤ 2) : b ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_b_not_two_l542_54201


namespace NUMINAMATH_CALUDE_painting_price_l542_54288

theorem painting_price (purchase_price : ℝ) (original_price : ℝ) : 
  purchase_price = 200 → 
  purchase_price = (1/4) * original_price → 
  original_price = 800 := by
sorry

end NUMINAMATH_CALUDE_painting_price_l542_54288


namespace NUMINAMATH_CALUDE_divisibility_by_three_l542_54230

theorem divisibility_by_three (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ)
  (h1 : A ^ 2 + B ^ 2 = A * B)
  (h2 : IsUnit (B * A - A * B)) :
  3 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l542_54230


namespace NUMINAMATH_CALUDE_divisors_of_8_factorial_l542_54231

theorem divisors_of_8_factorial (n : ℕ) : n = 8 → (Finset.card (Nat.divisors (Nat.factorial n))) = 96 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_8_factorial_l542_54231


namespace NUMINAMATH_CALUDE_unshaded_perimeter_l542_54261

/-- Given an L-shaped region formed by two adjoining rectangles with the following properties:
  - The total area of the L-shape is 240 square inches
  - The area of the shaded region is 65 square inches
  - The total length of the combined rectangles is 20 inches
  - The total width at the widest point is 12 inches
  - The width of the inner shaded rectangle is 5 inches
  - All rectangles contain right angles

  This theorem proves that the perimeter of the unshaded region is 64 inches. -/
theorem unshaded_perimeter (total_area : ℝ) (shaded_area : ℝ) (total_length : ℝ) (total_width : ℝ) (inner_width : ℝ)
  (h_total_area : total_area = 240)
  (h_shaded_area : shaded_area = 65)
  (h_total_length : total_length = 20)
  (h_total_width : total_width = 12)
  (h_inner_width : inner_width = 5) :
  2 * ((total_width - inner_width) + (total_area - shaded_area) / (total_width - inner_width)) = 64 :=
by sorry

end NUMINAMATH_CALUDE_unshaded_perimeter_l542_54261


namespace NUMINAMATH_CALUDE_soccer_team_lineup_combinations_l542_54287

def choose (n k : ℕ) : ℕ := Nat.choose n k

def total_players : ℕ := 18
def twins : ℕ := 2
def lineup_size : ℕ := 8
def defenders : ℕ := 5

theorem soccer_team_lineup_combinations : 
  (choose 2 1 * choose 5 3 * choose 11 4) +
  (choose 2 2 * choose 5 3 * choose 11 3) +
  (choose 2 1 * choose 5 4 * choose 11 3) +
  (choose 2 2 * choose 5 4 * choose 11 2) +
  (choose 2 1 * choose 5 5 * choose 11 2) +
  (choose 2 2 * choose 5 5 * choose 11 1) = 3602 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_lineup_combinations_l542_54287


namespace NUMINAMATH_CALUDE_book_pages_calculation_l542_54265

theorem book_pages_calculation (pages_read : ℕ) (fraction_read : ℚ) (h1 : pages_read = 16) (h2 : fraction_read = 0.4) : 
  (pages_read : ℚ) / fraction_read = 40 := by
sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l542_54265


namespace NUMINAMATH_CALUDE_commercials_time_l542_54284

/-- Given a total time and a ratio of music to commercials, 
    calculate the number of minutes of commercials played. -/
theorem commercials_time (total_time : ℕ) (music_ratio commercial_ratio : ℕ) 
  (h1 : total_time = 112)
  (h2 : music_ratio = 9)
  (h3 : commercial_ratio = 5) :
  (total_time * commercial_ratio) / (music_ratio + commercial_ratio) = 40 := by
  sorry

#check commercials_time

end NUMINAMATH_CALUDE_commercials_time_l542_54284


namespace NUMINAMATH_CALUDE_pentagon_distance_equality_l542_54278

/-- A regular pentagon with vertices A1, A2, A3, A4, A5 -/
structure RegularPentagon where
  A1 : ℝ × ℝ
  A2 : ℝ × ℝ
  A3 : ℝ × ℝ
  A4 : ℝ × ℝ
  A5 : ℝ × ℝ
  is_regular : True  -- We assume this property without defining it explicitly

/-- The circumcircle of the regular pentagon -/
def circumcircle (p : RegularPentagon) : Set (ℝ × ℝ) :=
  sorry  -- Definition of the circumcircle

/-- The arc A1A5 of the circumcircle -/
def arcA1A5 (p : RegularPentagon) : Set (ℝ × ℝ) :=
  sorry  -- Definition of the arc A1A5

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry  -- Definition of Euclidean distance

/-- Statement of the theorem -/
theorem pentagon_distance_equality (p : RegularPentagon) (B : ℝ × ℝ)
    (h1 : B ∈ arcA1A5 p)
    (h2 : distance B p.A1 < distance B p.A5) :
    distance B p.A1 + distance B p.A3 + distance B p.A5 =
    distance B p.A2 + distance B p.A4 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_distance_equality_l542_54278


namespace NUMINAMATH_CALUDE_tangent_circles_existence_l542_54235

-- Define the necessary geometric objects
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the tangency relations
def isTangentToCircle (c1 c2 : Circle) : Prop :=
  sorry

def isTangentToLine (c : Circle) (l : Line) : Prop :=
  sorry

def isOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  sorry

-- Theorem statement
theorem tangent_circles_existence
  (C : Circle) (l : Line) (M : ℝ × ℝ) 
  (h : isOnLine M l) :
  ∃ (C' C'' : Circle),
    (isTangentToCircle C' C ∧ isTangentToLine C' l ∧ isOnLine M l) ∧
    (isTangentToCircle C'' C ∧ isTangentToLine C'' l ∧ isOnLine M l) ∧
    (C' ≠ C'') :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_existence_l542_54235


namespace NUMINAMATH_CALUDE_die_roll_probability_l542_54281

theorem die_roll_probability : 
  let n : ℕ := 8  -- number of rolls
  let p_even : ℚ := 1/2  -- probability of rolling an even number
  let p_odd : ℚ := 1 - p_even  -- probability of rolling an odd number
  1 - p_odd^n = 255/256 :=
by sorry

end NUMINAMATH_CALUDE_die_roll_probability_l542_54281


namespace NUMINAMATH_CALUDE_condition_relationship_l542_54262

theorem condition_relationship (x : ℝ) :
  (∀ x, x > 1 → 1 / x < 1) ∧ 
  (∃ x, 1 / x < 1 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l542_54262


namespace NUMINAMATH_CALUDE_bill_harry_nuts_ratio_l542_54275

theorem bill_harry_nuts_ratio : 
  ∀ (sue_nuts harry_nuts bill_nuts : ℕ),
    sue_nuts = 48 →
    harry_nuts = 2 * sue_nuts →
    bill_nuts + harry_nuts = 672 →
    bill_nuts = 6 * harry_nuts :=
by
  sorry

end NUMINAMATH_CALUDE_bill_harry_nuts_ratio_l542_54275


namespace NUMINAMATH_CALUDE_inverse_of_17_mod_43_l542_54273

theorem inverse_of_17_mod_43 :
  ∃ x : ℕ, x < 43 ∧ (17 * x) % 43 = 1 :=
by
  use 6
  sorry

end NUMINAMATH_CALUDE_inverse_of_17_mod_43_l542_54273


namespace NUMINAMATH_CALUDE_abc_value_l542_54243

noncomputable def A (x : ℝ) : ℝ := ∑' k, x^(3*k) / (3*k).factorial
noncomputable def B (x : ℝ) : ℝ := ∑' k, x^(3*k+1) / (3*k+1).factorial
noncomputable def C (x : ℝ) : ℝ := ∑' k, x^(3*k+2) / (3*k+2).factorial

theorem abc_value (x : ℝ) (hx : x > 0) :
  (A x)^3 + (B x)^3 + (C x)^3 + 8*(A x)*(B x)*(C x) = 2014 →
  (A x)*(B x)*(C x) = 183 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l542_54243


namespace NUMINAMATH_CALUDE_xyz_value_l542_54220

-- Define a geometric sequence of 5 terms
def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ b = a * q ∧ c = b * q ∧ d = c * q ∧ e = d * q

-- State the theorem
theorem xyz_value (x y z : ℝ) 
  (h : is_geometric_sequence (-1) x y z (-4)) : x * y * z = -8 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l542_54220


namespace NUMINAMATH_CALUDE_angle_bisector_implies_line_AC_l542_54223

-- Define points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-1, 2)

-- Define the angle bisector equation
def angle_bisector (x y : ℝ) : Prop := y = x + 1

-- Define the equation of line AC
def line_AC (x y : ℝ) : Prop := x - 2*y - 1 = 0

theorem angle_bisector_implies_line_AC :
  ∀ C : ℝ × ℝ,
  angle_bisector C.1 C.2 →
  line_AC C.1 C.2 :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_implies_line_AC_l542_54223


namespace NUMINAMATH_CALUDE_product_of_digits_of_largest_valid_number_l542_54264

/-- A function that returns true if the digits of a natural number are in strictly increasing order --/
def strictly_increasing_digits (n : ℕ) : Prop := sorry

/-- A function that returns the sum of the squares of the digits of a natural number --/
def sum_of_squared_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the product of the digits of a natural number --/
def product_of_digits (n : ℕ) : ℕ := sorry

/-- The largest natural number whose digits are in strictly increasing order and whose digits' squares sum to 50 --/
def largest_valid_number : ℕ := sorry

theorem product_of_digits_of_largest_valid_number : 
  strictly_increasing_digits largest_valid_number ∧ 
  sum_of_squared_digits largest_valid_number = 50 ∧
  product_of_digits largest_valid_number = 36 ∧
  ∀ m : ℕ, 
    strictly_increasing_digits m ∧ 
    sum_of_squared_digits m = 50 → 
    m ≤ largest_valid_number :=
sorry

end NUMINAMATH_CALUDE_product_of_digits_of_largest_valid_number_l542_54264


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_18_factorial_l542_54298

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def largest_power_of_three_dividing (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc + (i + 1).log 3) 0

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_three_dividing_18_factorial :
  ones_digit (3^(largest_power_of_three_dividing (factorial 18))) = 1 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_18_factorial_l542_54298


namespace NUMINAMATH_CALUDE_wrapping_paper_division_l542_54228

theorem wrapping_paper_division (total_used : ℚ) (num_presents : ℕ) 
  (h1 : total_used = 1/2)
  (h2 : num_presents = 5) :
  total_used / num_presents = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_division_l542_54228


namespace NUMINAMATH_CALUDE_total_red_cards_l542_54210

/-- The number of decks the shopkeeper has -/
def num_decks : ℕ := 7

/-- The number of red cards in one deck -/
def red_cards_per_deck : ℕ := 26

/-- Theorem: The total number of red cards the shopkeeper has is 182 -/
theorem total_red_cards : num_decks * red_cards_per_deck = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_red_cards_l542_54210


namespace NUMINAMATH_CALUDE_smallest_prime_8_less_than_odd_square_l542_54218

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem smallest_prime_8_less_than_odd_square : 
  ∀ n : ℕ, 
    n > 0 → 
    is_prime n → 
    (∃ m : ℕ, 
      m ≥ 16 ∧ 
      is_perfect_square (n + 8) ∧ 
      is_odd (n + 8) ∧ 
      n + 8 = m * m) → 
    n ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_8_less_than_odd_square_l542_54218


namespace NUMINAMATH_CALUDE_parabola_translation_l542_54280

/-- Represents a parabola of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k + dy }

theorem parabola_translation (p : Parabola) (dx dy : ℝ) :
  p.a = 2 ∧ p.h = 4 ∧ p.k = 3 ∧ dx = 4 ∧ dy = 3 →
  let p' := translate p dx dy
  p'.a = 2 ∧ p'.h = 0 ∧ p'.k = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l542_54280


namespace NUMINAMATH_CALUDE_noah_class_size_l542_54263

theorem noah_class_size (n : ℕ) (noah_rank_best : ℕ) (noah_rank_worst : ℕ) 
  (h1 : noah_rank_best = 40)
  (h2 : noah_rank_worst = 40)
  (h3 : n = noah_rank_best + noah_rank_worst - 1) :
  n = 79 := by
  sorry

end NUMINAMATH_CALUDE_noah_class_size_l542_54263


namespace NUMINAMATH_CALUDE_root_implies_m_value_l542_54227

theorem root_implies_m_value (m : ℚ) : 
  (∃ x : ℚ, x^2 - 6*x - 3*m - 5 = 0) ∧ 
  ((-1 : ℚ)^2 - 6*(-1) - 3*m - 5 = 0) → 
  m = 2/3 := by
sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l542_54227


namespace NUMINAMATH_CALUDE_chord_equation_of_ellipse_l542_54209

/-- Given an ellipse and a point on a bisecting chord, prove the equation of the line containing the chord. -/
theorem chord_equation_of_ellipse (x y : ℝ) :
  let ellipse := fun (x y : ℝ) => x^2/16 + y^2/4 = 1
  let P := (-2, 1)
  let chord_bisector := fun (x y : ℝ) => ∃ (x1 y1 x2 y2 : ℝ),
    ellipse x1 y1 ∧ ellipse x2 y2 ∧ 
    x = (x1 + x2)/2 ∧ y = (y1 + y2)/2
  chord_bisector P.1 P.2 →
  x - 2*y + 4 = 0 := by sorry

end NUMINAMATH_CALUDE_chord_equation_of_ellipse_l542_54209


namespace NUMINAMATH_CALUDE_race_time_problem_l542_54292

/-- Given two racers A and B, where their speeds are in the ratio 3:4 and A takes 30 minutes more
    than B to reach the destination, prove that A takes 120 minutes to reach the destination. -/
theorem race_time_problem (v_A v_B : ℝ) (t_A t_B : ℝ) (D : ℝ) :
  v_A / v_B = 3 / 4 →  -- speeds are in ratio 3:4
  t_A = t_B + 30 →     -- A takes 30 minutes more than B
  D = v_A * t_A →      -- distance = speed * time for A
  D = v_B * t_B →      -- distance = speed * time for B
  t_A = 120 :=         -- A takes 120 minutes
by sorry

end NUMINAMATH_CALUDE_race_time_problem_l542_54292


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l542_54229

theorem unique_modular_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ 12019 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l542_54229


namespace NUMINAMATH_CALUDE_jose_investment_is_45000_l542_54256

/-- Represents the investment and profit scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment based on the given conditions --/
def calculate_jose_investment (shop : ShopInvestment) : ℕ :=
  let tom_investment_months := shop.tom_investment * 12
  let jose_investment_months := (12 - shop.jose_join_delay) * (shop.total_profit - shop.jose_profit) * 10 / shop.jose_profit
  jose_investment_months / (12 - shop.jose_join_delay)

/-- Theorem stating that Jose's investment is 45000 given the specified conditions --/
theorem jose_investment_is_45000 (shop : ShopInvestment)
  (h1 : shop.tom_investment = 30000)
  (h2 : shop.jose_join_delay = 2)
  (h3 : shop.total_profit = 36000)
  (h4 : shop.jose_profit = 20000) :
  calculate_jose_investment shop = 45000 := by
  sorry

#eval calculate_jose_investment ⟨30000, 2, 36000, 20000⟩

end NUMINAMATH_CALUDE_jose_investment_is_45000_l542_54256


namespace NUMINAMATH_CALUDE_equation_solution_l542_54203

theorem equation_solution (x : ℝ) : 
  (21 / (x^2 - 9) - 3 / (x - 3) = 2) ↔ (x = 5 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l542_54203


namespace NUMINAMATH_CALUDE_largest_multiple_of_12_using_all_digits_l542_54253

/-- A function that checks if a number uses each digit from 0 to 9 exactly once -/
def usesAllDigitsOnce (n : ℕ) : Prop := sorry

/-- A function that returns the largest number that can be formed using each digit from 0 to 9 exactly once and is a multiple of 12 -/
def largestMultipleOf12UsingAllDigits : ℕ := sorry

theorem largest_multiple_of_12_using_all_digits :
  largestMultipleOf12UsingAllDigits = 987654320 ∧
  usesAllDigitsOnce largestMultipleOf12UsingAllDigits ∧
  largestMultipleOf12UsingAllDigits % 12 = 0 ∧
  ∀ m : ℕ, usesAllDigitsOnce m ∧ m % 12 = 0 → m ≤ largestMultipleOf12UsingAllDigits :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_12_using_all_digits_l542_54253


namespace NUMINAMATH_CALUDE_set_operations_l542_54217

open Set

def A : Set ℝ := {x | x ≤ 5}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 8}

theorem set_operations :
  (A ∩ B = {x | -3 < x ∧ x ≤ 5}) ∧
  (A ∪ B = {x | x ≤ 8}) ∧
  (A ∪ (𝒰 \ B) = {x | x ≤ 5 ∨ x > 8}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l542_54217


namespace NUMINAMATH_CALUDE_dogwood_trees_planting_l542_54260

theorem dogwood_trees_planting (initial_trees : ℕ) (planted_today : ℕ) (final_total : ℕ) 
  (h1 : initial_trees = 39)
  (h2 : planted_today = 41)
  (h3 : final_total = 100) :
  final_total - (initial_trees + planted_today) = 20 :=
by sorry

end NUMINAMATH_CALUDE_dogwood_trees_planting_l542_54260


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l542_54205

theorem sum_of_coefficients_zero (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x, (1 - 4*x)^10 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                       a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁/2 + a₂/2^2 + a₃/2^3 + a₄/2^4 + a₅/2^5 + a₆/2^6 + a₇/2^7 + a₈/2^8 + a₉/2^9 + a₁₀/2^10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l542_54205


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l542_54239

theorem absolute_value_equation_product (x₁ x₂ : ℝ) : 
  (|3 * x₁ - 5| = 40) ∧ (|3 * x₂ - 5| = 40) ∧ (x₁ ≠ x₂) →
  x₁ * x₂ = -175 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l542_54239


namespace NUMINAMATH_CALUDE_bridge_building_time_l542_54249

/-- If a crew of m workers can build a bridge in d days, then a crew of 2m workers can build the same bridge in d/2 days. -/
theorem bridge_building_time (m d : ℝ) (h1 : m > 0) (h2 : d > 0) :
  let initial_crew := m
  let initial_time := d
  let new_crew := 2 * m
  let new_time := d / 2
  initial_crew * initial_time = new_crew * new_time :=
by sorry

end NUMINAMATH_CALUDE_bridge_building_time_l542_54249


namespace NUMINAMATH_CALUDE_clean_city_people_l542_54219

/-- The number of people working together to clean the city -/
def total_people (group_A group_B group_C group_D group_E group_F group_G group_H : ℕ) : ℕ :=
  group_A + group_B + group_C + group_D + group_E + group_F + group_G + group_H

/-- Theorem stating the total number of people cleaning the city -/
theorem clean_city_people :
  ∃ (group_A group_B group_C group_D group_E group_F group_G group_H : ℕ),
    group_A = 54 ∧
    group_B = group_A - 17 ∧
    group_C = 2 * group_B ∧
    group_D = group_A / 3 ∧
    group_E = group_C + (group_C / 4) ∧
    group_F = group_D / 2 ∧
    group_G = (group_A + group_B + group_C) - ((group_A + group_B + group_C) * 3 / 10) ∧
    group_H = group_F + group_G ∧
    total_people group_A group_B group_C group_D group_E group_F group_G group_H = 523 :=
by sorry

end NUMINAMATH_CALUDE_clean_city_people_l542_54219


namespace NUMINAMATH_CALUDE_repeated_digit_sum_tower_exp_l542_54226

-- Define the function for the tower of exponents
def tower_exp (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 6^(tower_exp n)

-- Define the repeated digit sum operation (conceptually)
def repeated_digit_sum (n : ℕ) : ℕ := n % 11

-- State the theorem
theorem repeated_digit_sum_tower_exp : 
  repeated_digit_sum (7^(tower_exp 5)) = 4 := by sorry

end NUMINAMATH_CALUDE_repeated_digit_sum_tower_exp_l542_54226


namespace NUMINAMATH_CALUDE_gumball_cost_l542_54270

/-- Given that Melanie sells 4 gumballs for a total of 32 cents, prove that each gumball costs 8 cents. -/
theorem gumball_cost (num_gumballs : ℕ) (total_cents : ℕ) (h1 : num_gumballs = 4) (h2 : total_cents = 32) :
  total_cents / num_gumballs = 8 := by
  sorry

end NUMINAMATH_CALUDE_gumball_cost_l542_54270


namespace NUMINAMATH_CALUDE_turtle_reaches_waterhole_in_28_minutes_l542_54206

/-- Represents the scenario with two lion cubs and a turtle moving towards a watering hole -/
structure WaterholeProblem where
  /-- Distance of the first lion cub from the watering hole in minutes -/
  lion1_distance : ℝ
  /-- Speed multiplier of the second lion cub compared to the first -/
  lion2_speed_multiplier : ℝ
  /-- Distance of the turtle from the watering hole in minutes -/
  turtle_distance : ℝ

/-- Calculates the time it takes for the turtle to reach the watering hole after meeting the lion cubs -/
def timeToWaterhole (problem : WaterholeProblem) : ℝ :=
  sorry

/-- Theorem stating that given the specific problem conditions, the turtle reaches the watering hole 28 minutes after meeting the lion cubs -/
theorem turtle_reaches_waterhole_in_28_minutes :
  let problem : WaterholeProblem :=
    { lion1_distance := 5
      lion2_speed_multiplier := 1.5
      turtle_distance := 30 }
  timeToWaterhole problem = 28 :=
sorry

end NUMINAMATH_CALUDE_turtle_reaches_waterhole_in_28_minutes_l542_54206


namespace NUMINAMATH_CALUDE_inequality_holds_l542_54272

theorem inequality_holds (x : ℝ) : (1 : ℝ) / (x^2 + 1) > (1 : ℝ) / (x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l542_54272


namespace NUMINAMATH_CALUDE_union_of_A_and_B_intersection_condition_l542_54297

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 2| < a}
def B : Set ℝ := {x : ℝ | (2*x - 1) / (x + 2) < 1}

-- Part 1
theorem union_of_A_and_B :
  A 2 ∪ B = {x : ℝ | -2 < x ∧ x < 4} := by sorry

-- Part 2
theorem intersection_condition (a : ℝ) :
  A a ∩ B = A a ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_intersection_condition_l542_54297


namespace NUMINAMATH_CALUDE_quadratic_roots_and_fraction_l542_54254

theorem quadratic_roots_and_fraction (a b p q : ℝ) : 
  (∃ (x : ℂ), x^2 + p*x + q = 0 ∧ (x = 2 + a*I ∨ x = b + I)) →
  (a = -1 ∧ b = 2 ∧ p = -4 ∧ q = 5) ∧
  (a + b*I) / (p + q*I) = 3/41 + 6/41*I :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_fraction_l542_54254


namespace NUMINAMATH_CALUDE_intersection_when_a_half_range_of_a_for_empty_intersection_l542_54233

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem for part I
theorem intersection_when_a_half : 
  A (1/2) ∩ B = {x | 0 < x ∧ x < 1} := by sorry

-- Theorem for part II
theorem range_of_a_for_empty_intersection :
  ∀ a : ℝ, (A a).Nonempty → (A a ∩ B = ∅) → 
    ((-2 < a ∧ a ≤ -1/2) ∨ a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_half_range_of_a_for_empty_intersection_l542_54233


namespace NUMINAMATH_CALUDE_trigonometric_identity_l542_54282

theorem trigonometric_identity (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi) (h3 : -Real.sin α = 2 * Real.cos α) :
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l542_54282


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l542_54283

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l542_54283


namespace NUMINAMATH_CALUDE_jake_present_weight_l542_54294

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 156

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 224 - jake_weight

/-- Theorem stating Jake's present weight is 156 pounds -/
theorem jake_present_weight : jake_weight = 156 := by
  have h1 : jake_weight - 20 = 2 * sister_weight := by sorry
  have h2 : jake_weight + sister_weight = 224 := by sorry
  sorry

#check jake_present_weight

end NUMINAMATH_CALUDE_jake_present_weight_l542_54294


namespace NUMINAMATH_CALUDE_bert_sandwiches_l542_54224

/-- The number of sandwiches remaining after two days of eating -/
def sandwiches_remaining (initial : ℕ) : ℕ :=
  initial - (initial / 2) - (initial / 2 - 2)

/-- Theorem stating that given 12 initial sandwiches, 2 remain after two days of eating -/
theorem bert_sandwiches : sandwiches_remaining 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_bert_sandwiches_l542_54224


namespace NUMINAMATH_CALUDE_hyperbola_center_trajectory_l542_54299

/-- The equation of the trajectory of the center of a hyperbola -/
theorem hyperbola_center_trajectory 
  (x y m : ℝ) 
  (h : x^2 - y^2 - 6*m*x - 4*m*y + 5*m^2 - 1 = 0) : 
  2*x + 3*y = 0 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_trajectory_l542_54299


namespace NUMINAMATH_CALUDE_parallelogram_area_from_rectangle_l542_54215

theorem parallelogram_area_from_rectangle (rectangle_width rectangle_length parallelogram_height : ℝ) 
  (hw : rectangle_width = 8)
  (hl : rectangle_length = 10)
  (hh : parallelogram_height = 9) :
  rectangle_width * parallelogram_height = 72 := by
  sorry

#check parallelogram_area_from_rectangle

end NUMINAMATH_CALUDE_parallelogram_area_from_rectangle_l542_54215


namespace NUMINAMATH_CALUDE_max_gcd_sum_1085_l542_54258

theorem max_gcd_sum_1085 :
  ∃ (m n : ℕ+), m + n = 1085 ∧ 
  ∀ (a b : ℕ+), a + b = 1085 → Nat.gcd a b ≤ Nat.gcd m n ∧
  Nat.gcd m n = 217 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_sum_1085_l542_54258


namespace NUMINAMATH_CALUDE_matches_played_eq_teams_minus_one_l542_54255

/-- Represents an elimination tournament. -/
structure EliminationTournament where
  num_teams : ℕ
  no_replays : Bool

/-- The number of matches played in an elimination tournament. -/
def matches_played (t : EliminationTournament) : ℕ := sorry

/-- Theorem stating that in an elimination tournament with no replays, 
    the number of matches played is one less than the number of teams. -/
theorem matches_played_eq_teams_minus_one (t : EliminationTournament) 
  (h : t.no_replays = true) : matches_played t = t.num_teams - 1 := by sorry

end NUMINAMATH_CALUDE_matches_played_eq_teams_minus_one_l542_54255


namespace NUMINAMATH_CALUDE_tims_total_expense_l542_54232

/-- Calculates Tim's total out-of-pocket expense for medical visits -/
theorem tims_total_expense (tims_visit_cost : ℝ) (tims_insurance_coverage : ℝ) 
  (cats_visit_cost : ℝ) (cats_insurance_coverage : ℝ) 
  (h1 : tims_visit_cost = 300)
  (h2 : tims_insurance_coverage = 0.75 * tims_visit_cost)
  (h3 : cats_visit_cost = 120)
  (h4 : cats_insurance_coverage = 60) : 
  tims_visit_cost - tims_insurance_coverage + cats_visit_cost - cats_insurance_coverage = 135 := by
  sorry


end NUMINAMATH_CALUDE_tims_total_expense_l542_54232


namespace NUMINAMATH_CALUDE_max_value_circle_center_l542_54259

/-- Circle C with center (a,b) and radius 1 -/
def Circle (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = 1}

/-- Region Ω -/
def Ω : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 - 7 ≤ 0 ∧ p.1 - p.2 + 3 ≥ 0 ∧ p.2 ≥ 0}

/-- The maximum value of a^2 + b^2 given the conditions -/
theorem max_value_circle_center (a b : ℝ) :
  (a, b) ∈ Ω →
  b = 1 →
  (∃ (x : ℝ), (x, 0) ∈ Circle a b) →
  a^2 + b^2 ≤ 37 :=
sorry

end NUMINAMATH_CALUDE_max_value_circle_center_l542_54259


namespace NUMINAMATH_CALUDE_multiples_of_ten_not_twenty_l542_54295

def count_numbers (n : ℕ) : ℕ :=
  (n.div 10 + 1).div 2

theorem multiples_of_ten_not_twenty (upper_bound : ℕ) (h : upper_bound = 500) :
  count_numbers upper_bound = 25 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_ten_not_twenty_l542_54295


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l542_54248

/-- The y-coordinate of the intersection point of perpendicular tangents on y = 4x^2 -/
theorem perpendicular_tangents_intersection (a b : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    A.2 = 4 * A.1^2 ∧ 
    B.2 = 4 * B.1^2 ∧ 
    A.1 = a ∧ 
    B.1 = b ∧ 
    (8 * a) * (8 * b) = -1) → 
  ∃ P : ℝ × ℝ, 
    (P.1 = (a + b) / 2) ∧ 
    (P.2 = -2) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l542_54248


namespace NUMINAMATH_CALUDE_digit_value_difference_l542_54237

/-- The numeral we are working with -/
def numeral : ℕ := 657903

/-- The digit we are focusing on -/
def digit : ℕ := 7

/-- The position of the digit in the numeral (counting from right, starting at 0) -/
def position : ℕ := 4

/-- The local value of a digit in a given position -/
def local_value (d : ℕ) (pos : ℕ) : ℕ := d * (10 ^ pos)

/-- The face value of a digit -/
def face_value (d : ℕ) : ℕ := d

/-- The difference between local value and face value -/
def value_difference (d : ℕ) (pos : ℕ) : ℕ := local_value d pos - face_value d

theorem digit_value_difference :
  value_difference digit position = 69993 := by sorry

end NUMINAMATH_CALUDE_digit_value_difference_l542_54237


namespace NUMINAMATH_CALUDE_ten_balls_distribution_l542_54241

/-- The number of ways to distribute n identical balls into 3 boxes numbered 1, 2, and 3,
    where each box must contain at least as many balls as its number. -/
def distributionWays (n : ℕ) : ℕ :=
  let remainingBalls := n - (1 + 2 + 3)
  (remainingBalls + 3 - 1).choose 2

/-- Theorem: There are 15 ways to distribute 10 identical balls into 3 boxes numbered 1, 2, and 3,
    where each box must contain at least as many balls as its number. -/
theorem ten_balls_distribution : distributionWays 10 = 15 := by
  sorry

#eval distributionWays 10  -- Should output 15

end NUMINAMATH_CALUDE_ten_balls_distribution_l542_54241


namespace NUMINAMATH_CALUDE_smallest_common_multiple_l542_54214

theorem smallest_common_multiple : ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 ∧ 6 ∣ m ∧ 8 ∣ m ∧ 12 ∣ m → n ≤ m) ∧ 
  6 ∣ n ∧ 8 ∣ n ∧ 12 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_l542_54214


namespace NUMINAMATH_CALUDE_farm_birds_l542_54274

theorem farm_birds (chickens ducks turkeys : ℕ) : 
  ducks = 2 * chickens →
  turkeys = 3 * ducks →
  chickens + ducks + turkeys = 1800 →
  chickens = 200 := by
sorry

end NUMINAMATH_CALUDE_farm_birds_l542_54274


namespace NUMINAMATH_CALUDE_tangent_and_trigonometric_identity_l542_54216

theorem tangent_and_trigonometric_identity (α : ℝ) 
  (h : Real.tan (α + π / 3) = 2 * Real.sqrt 3) : 
  (Real.tan (α - 2 * π / 3) = 2 * Real.sqrt 3) ∧ 
  (2 * Real.sin α ^ 2 - Real.cos α ^ 2 = -43 / 52) := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_trigonometric_identity_l542_54216


namespace NUMINAMATH_CALUDE_determinant_of_2x2_matrix_l542_54240

theorem determinant_of_2x2_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![9, 5; -3, 4]
  Matrix.det A = 51 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_2x2_matrix_l542_54240


namespace NUMINAMATH_CALUDE_odd_function_monotonicity_l542_54257

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 1 / (4^x + 1)

theorem odd_function_monotonicity (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = -1/2 ∧ ∀ x₁ x₂, x₁ < x₂ → f a x₁ > f a x₂) := by sorry

end NUMINAMATH_CALUDE_odd_function_monotonicity_l542_54257


namespace NUMINAMATH_CALUDE_subscription_total_l542_54211

-- Define the subscription amounts for a, b, and c
def subscription (a b c : ℕ) : Prop :=
  a = b + 4000 ∧ b = c + 5000

-- Define the total profit and b's share
def profit_share (total_profit b_profit : ℕ) : Prop :=
  total_profit = 30000 ∧ b_profit = 10200

-- Define the total subscription
def total_subscription (a b c : ℕ) : ℕ :=
  a + b + c

-- Theorem statement
theorem subscription_total 
  (a b c : ℕ) 
  (h1 : subscription a b c) 
  (h2 : profit_share 30000 10200) :
  total_subscription a b c = 14036 :=
sorry

end NUMINAMATH_CALUDE_subscription_total_l542_54211


namespace NUMINAMATH_CALUDE_mod_37_5_l542_54277

theorem mod_37_5 : 37 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_37_5_l542_54277


namespace NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l542_54245

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- State the theorem
theorem fourth_power_of_cube_of_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 4 = 244140625 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l542_54245


namespace NUMINAMATH_CALUDE_probability_x_less_than_2y_l542_54247

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

-- Define the region where x < 2y
def region : Set (ℝ × ℝ) :=
  {p ∈ rectangle | p.1 < 2 * p.2}

-- Define the probability measure on the rectangle
noncomputable def prob : MeasureTheory.Measure (ℝ × ℝ) :=
  sorry

-- State the theorem
theorem probability_x_less_than_2y :
  prob region / prob rectangle = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_less_than_2y_l542_54247


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l542_54291

/-- Given an ellipse where the length of the major axis is twice the length of the minor axis,
    prove that its eccentricity is √3/2. -/
theorem ellipse_eccentricity (a b : ℝ) (h : a = 2 * b) (h_pos : a > 0) :
  let c := Real.sqrt (a^2 - b^2)
  c / a = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l542_54291


namespace NUMINAMATH_CALUDE_rationalize_denominator_1_l542_54208

theorem rationalize_denominator_1 (a b c : ℝ) :
  a / (b - Real.sqrt c + a) = (a * (b + a + Real.sqrt c)) / ((b + a)^2 - c) :=
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_1_l542_54208


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l542_54204

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

theorem geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_decreasing : is_decreasing_sequence a)
  (h_third_term : a 3 = 18)
  (h_fourth_term : a 4 = 12) :
  a 1 = 40.5 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l542_54204


namespace NUMINAMATH_CALUDE_original_class_size_l542_54222

/-- Proves that the original number of students in a class is 12, given the conditions of the problem. -/
theorem original_class_size (initial_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  initial_avg = 40 →
  new_students = 12 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ (original_size : ℕ),
    original_size * initial_avg + new_students * new_avg = (original_size + new_students) * (initial_avg - avg_decrease) ∧
    original_size = 12 := by
  sorry

end NUMINAMATH_CALUDE_original_class_size_l542_54222


namespace NUMINAMATH_CALUDE_total_money_made_l542_54221

/-- Represents the amount of water collected per inch of rain -/
def gallons_per_inch : ℝ := 15

/-- Represents the rainfall on Monday in inches -/
def monday_rain : ℝ := 4

/-- Represents the rainfall on Tuesday in inches -/
def tuesday_rain : ℝ := 3

/-- Represents the rainfall on Wednesday in inches -/
def wednesday_rain : ℝ := 2.5

/-- Represents the selling price per gallon on Monday -/
def monday_price : ℝ := 1.2

/-- Represents the selling price per gallon on Tuesday -/
def tuesday_price : ℝ := 1.5

/-- Represents the selling price per gallon on Wednesday -/
def wednesday_price : ℝ := 0.8

/-- Theorem stating the total money James made from selling water -/
theorem total_money_made : 
  (gallons_per_inch * monday_rain * monday_price) +
  (gallons_per_inch * tuesday_rain * tuesday_price) +
  (gallons_per_inch * wednesday_rain * wednesday_price) = 169.5 := by
  sorry

end NUMINAMATH_CALUDE_total_money_made_l542_54221


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l542_54267

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (25 - a) + b / (65 - b) + c / (60 - c) = 7) :
  5 / (25 - a) + 13 / (65 - b) + 12 / (60 - c) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l542_54267


namespace NUMINAMATH_CALUDE_square_difference_of_constrained_integers_l542_54276

theorem square_difference_of_constrained_integers (x y : ℕ+) 
  (h1 : 56 ≤ (x:ℝ) + y ∧ (x:ℝ) + y ≤ 59)
  (h2 : (0.9:ℝ) < (x:ℝ) / y ∧ (x:ℝ) / y < 0.91) :
  (y:ℤ)^2 - (x:ℤ)^2 = 177 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_constrained_integers_l542_54276


namespace NUMINAMATH_CALUDE_first_competitor_distance_l542_54238

/-- The long jump competition with four competitors -/
structure LongJumpCompetition where
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ

/-- The conditions of the long jump competition -/
def validCompetition (c : LongJumpCompetition) : Prop :=
  c.second = c.first + 1 ∧
  c.third = c.second - 2 ∧
  c.fourth = c.third + 3 ∧
  c.fourth = 24

/-- Theorem: In a valid long jump competition, the first competitor jumped 22 feet -/
theorem first_competitor_distance (c : LongJumpCompetition) 
  (h : validCompetition c) : c.first = 22 := by
  sorry

#check first_competitor_distance

end NUMINAMATH_CALUDE_first_competitor_distance_l542_54238


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l542_54290

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l542_54290


namespace NUMINAMATH_CALUDE_f_max_min_l542_54244

noncomputable def f (x : ℝ) : ℝ := 2^(x+2) - 3 * 4^x

theorem f_max_min :
  ∀ x ∈ Set.Icc (-1 : ℝ) 0,
    f x ≤ 4/3 ∧ f x ≥ 1 ∧
    (∃ x₁ ∈ Set.Icc (-1 : ℝ) 0, f x₁ = 4/3) ∧
    (∃ x₂ ∈ Set.Icc (-1 : ℝ) 0, f x₂ = 1) :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_l542_54244


namespace NUMINAMATH_CALUDE_yolanda_free_throws_l542_54234

/-- Calculates the average number of free throws per game given the total points,
    number of games, and average two-point and three-point baskets per game. -/
def avg_free_throws (total_points : ℕ) (num_games : ℕ) 
                    (avg_two_point : ℕ) (avg_three_point : ℕ) : ℕ :=
  let avg_points_per_game := total_points / num_games
  let points_from_two_point := avg_two_point * 2
  let points_from_three_point := avg_three_point * 3
  avg_points_per_game - (points_from_two_point + points_from_three_point)

theorem yolanda_free_throws : 
  avg_free_throws 345 15 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_yolanda_free_throws_l542_54234


namespace NUMINAMATH_CALUDE_stadium_fee_difference_l542_54202

/-- Calculates the difference in total fees collected between full capacity and 3/4 capacity for a stadium. -/
def fee_difference (capacity : ℕ) (entry_fee : ℕ) : ℕ :=
  capacity * entry_fee - (capacity * 3 / 4) * entry_fee

/-- Proves that the fee difference for a stadium with 2000 capacity and $20 entry fee is $10,000. -/
theorem stadium_fee_difference :
  fee_difference 2000 20 = 10000 := by
  sorry

#eval fee_difference 2000 20

end NUMINAMATH_CALUDE_stadium_fee_difference_l542_54202


namespace NUMINAMATH_CALUDE_park_nests_l542_54293

/-- Calculates the minimum number of nests required for birds in a park -/
def minimum_nests (sparrows pigeons starlings robins : ℕ) 
  (sparrow_nests pigeon_nests starling_nests robin_nests : ℕ) : ℕ :=
  sparrows * sparrow_nests + pigeons * pigeon_nests + 
  starlings * starling_nests + robins * robin_nests

/-- Theorem stating the minimum number of nests required for the given bird populations -/
theorem park_nests : 
  minimum_nests 5 3 6 2 1 2 3 4 = 37 := by
  sorry

#eval minimum_nests 5 3 6 2 1 2 3 4

end NUMINAMATH_CALUDE_park_nests_l542_54293
