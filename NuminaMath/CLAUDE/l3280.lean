import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l3280_328081

-- Define propositions P and Q
def P (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (P a ∨ Q a) ∧ ¬(P a ∧ Q a) →
  ((-1 < a ∧ a ≤ 1) ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3280_328081


namespace NUMINAMATH_CALUDE_roots_sum_quotient_and_reciprocal_l3280_328068

theorem roots_sum_quotient_and_reciprocal (a b : ℝ) : 
  (a^2 + 10*a + 5 = 0) → 
  (b^2 + 10*b + 5 = 0) → 
  (a ≠ 0) → 
  (b ≠ 0) → 
  a/b + b/a = 18 := by sorry

end NUMINAMATH_CALUDE_roots_sum_quotient_and_reciprocal_l3280_328068


namespace NUMINAMATH_CALUDE_range_of_b_for_two_intersection_points_l3280_328087

/-- The range of b for which there are exactly two points P satisfying the given conditions -/
theorem range_of_b_for_two_intersection_points (b : ℝ) : 
  (∃! (P₁ P₂ : ℝ × ℝ), 
    P₁ ≠ P₂ ∧ 
    (P₁.1 + Real.sqrt 3 * P₁.2 = b) ∧ 
    (P₂.1 + Real.sqrt 3 * P₂.2 = b) ∧ 
    ((P₁.1 - 4)^2 + P₁.2^2 = 4 * (P₁.1^2 + P₁.2^2)) ∧
    ((P₂.1 - 4)^2 + P₂.2^2 = 4 * (P₂.1^2 + P₂.2^2))) ↔ 
  (-20/3 < b ∧ b < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_b_for_two_intersection_points_l3280_328087


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3280_328074

theorem geometric_sequence_common_ratio
  (x : ℝ)
  (h : ∃ r : ℝ, (x + Real.log 2 / Real.log 27) * r = x + Real.log 2 / Real.log 9 ∧
                (x + Real.log 2 / Real.log 9) * r = x + Real.log 2 / Real.log 3) :
  ∃ r : ℝ, r = 3 ∧
    (x + Real.log 2 / Real.log 27) * r = x + Real.log 2 / Real.log 9 ∧
    (x + Real.log 2 / Real.log 9) * r = x + Real.log 2 / Real.log 3 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3280_328074


namespace NUMINAMATH_CALUDE_remainder_of_9876543210_div_101_l3280_328056

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 68 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_9876543210_div_101_l3280_328056


namespace NUMINAMATH_CALUDE_local_minimum_of_f_l3280_328035

def f (x : ℝ) := x^3 - 3*x

theorem local_minimum_of_f :
  ∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_of_f_l3280_328035


namespace NUMINAMATH_CALUDE_average_weight_group_B_proof_l3280_328005

/-- The average weight of additional friends in Group B -/
def average_weight_group_B : ℝ := 141

theorem average_weight_group_B_proof
  (initial_group : ℕ) (additional_group : ℕ) (group_A : ℕ) (group_B : ℕ)
  (avg_weight_increase : ℝ) (avg_weight_gain_A : ℝ) (final_avg_weight : ℝ)
  (h1 : initial_group = 50)
  (h2 : additional_group = 40)
  (h3 : group_A = 20)
  (h4 : group_B = 20)
  (h5 : avg_weight_increase = 12)
  (h6 : avg_weight_gain_A = 15)
  (h7 : final_avg_weight = 46)
  (h8 : additional_group = group_A + group_B) :
  average_weight_group_B = 141 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_group_B_proof_l3280_328005


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2024_l3280_328055

theorem units_digit_of_7_power_2024 : 7^2024 ≡ 1 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2024_l3280_328055


namespace NUMINAMATH_CALUDE_max_beads_in_pile_l3280_328089

/-- Represents a pile of beads -/
structure BeadPile :=
  (size : ℕ)
  (has_lighter_bead : Bool)

/-- Represents a balance scale measurement -/
inductive Measurement
  | Balanced
  | Unbalanced

/-- A function that performs a measurement on a subset of beads -/
def perform_measurement (subset_size : ℕ) : Measurement :=
  sorry

/-- A function that represents the algorithm to find the lighter bead -/
def find_lighter_bead (pile : BeadPile) (max_measurements : ℕ) : Bool :=
  sorry

/-- Theorem stating the maximum number of beads in the pile -/
theorem max_beads_in_pile :
  ∀ (pile : BeadPile),
    pile.has_lighter_bead →
    (∃ (algorithm : BeadPile → ℕ → Bool),
      (∀ p, algorithm p 2 = find_lighter_bead p 2) →
      algorithm pile 2 = true) →
    pile.size ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_beads_in_pile_l3280_328089


namespace NUMINAMATH_CALUDE_remainder_492381_div_6_l3280_328094

theorem remainder_492381_div_6 : 492381 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_492381_div_6_l3280_328094


namespace NUMINAMATH_CALUDE_expression_evaluation_l3280_328083

theorem expression_evaluation : 
  (121 * (1/13 - 1/17) + 169 * (1/17 - 1/11) + 289 * (1/11 - 1/13)) / 
  (11 * (1/13 - 1/17) + 13 * (1/17 - 1/11) + 17 * (1/11 - 1/13)) = 41 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3280_328083


namespace NUMINAMATH_CALUDE_biased_coin_flip_l3280_328010

theorem biased_coin_flip (h : ℝ) : 
  0 < h → h < 1 →
  (4 : ℝ) * h * (1 - h)^3 = 6 * h^2 * (1 - h)^2 →
  (6 : ℝ) * (2/5)^2 * (3/5)^2 = 216/625 :=
by sorry

end NUMINAMATH_CALUDE_biased_coin_flip_l3280_328010


namespace NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l3280_328061

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) :
  ∃ m : ℤ, (n - 1) * n * (n + 1) * (n + 2) + 1 = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l3280_328061


namespace NUMINAMATH_CALUDE_highest_frequency_count_l3280_328022

theorem highest_frequency_count (total_sample : ℕ) (num_groups : ℕ) 
  (cumulative_freq_seven : ℚ) (a : ℕ) (r : ℕ) : 
  total_sample = 100 →
  num_groups = 10 →
  cumulative_freq_seven = 79/100 →
  r > 1 →
  a + a * r + a * r^2 = total_sample - (cumulative_freq_seven * total_sample).num →
  (∃ (max_freq : ℕ), max_freq = max a (max (a * r) (a * r^2)) ∧ max_freq = 12) :=
sorry

end NUMINAMATH_CALUDE_highest_frequency_count_l3280_328022


namespace NUMINAMATH_CALUDE_gcd_8675309_7654321_l3280_328095

theorem gcd_8675309_7654321 : Nat.gcd 8675309 7654321 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8675309_7654321_l3280_328095


namespace NUMINAMATH_CALUDE_A_in_second_quadrant_l3280_328079

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The point A with coordinates (-3, 4) -/
def A : Point :=
  { x := -3, y := 4 }

/-- Theorem stating that point A is in the second quadrant -/
theorem A_in_second_quadrant : second_quadrant A := by
  sorry

end NUMINAMATH_CALUDE_A_in_second_quadrant_l3280_328079


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3280_328020

/-- The repeating decimal 0.37268̄ expressed as a fraction -/
def repeating_decimal : ℚ := 371896 / 99900

/-- The decimal representation of 0.37268̄ -/
def decimal_representation : ℚ := 37 / 100 + 268 / 99900

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = decimal_representation := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3280_328020


namespace NUMINAMATH_CALUDE_direct_proportion_increasing_iff_m_gt_two_l3280_328054

/-- A direct proportion function y = (m-2)x where y increases as x increases -/
def direct_proportion_increasing (m : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (m - 2) * x₁ < (m - 2) * x₂

theorem direct_proportion_increasing_iff_m_gt_two :
  ∀ m : ℝ, direct_proportion_increasing m ↔ m > 2 :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_increasing_iff_m_gt_two_l3280_328054


namespace NUMINAMATH_CALUDE_opposite_face_is_U_l3280_328065

-- Define the faces of the cube
inductive Face : Type
  | P | Q | R | S | T | U

-- Define the property of being adjacent in the net
def adjacent_in_net : Face → Face → Prop :=
  sorry

-- Define the property of being opposite in the cube
def opposite_in_cube : Face → Face → Prop :=
  sorry

-- State the theorem
theorem opposite_face_is_U :
  (adjacent_in_net Face.P Face.Q) →
  (adjacent_in_net Face.P Face.R) →
  (adjacent_in_net Face.P Face.S) →
  (¬adjacent_in_net Face.P Face.T ∨ ¬adjacent_in_net Face.P Face.U) →
  opposite_in_cube Face.P Face.U :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_face_is_U_l3280_328065


namespace NUMINAMATH_CALUDE_plains_total_area_l3280_328043

/-- The total area of two plains given their individual areas -/
def total_area (area_A area_B : ℝ) : ℝ := area_A + area_B

/-- Theorem: Given the conditions, the total area of both plains is 350 square miles -/
theorem plains_total_area :
  ∀ (area_A area_B : ℝ),
  area_B = 200 →
  area_A = area_B - 50 →
  total_area area_A area_B = 350 := by
sorry

end NUMINAMATH_CALUDE_plains_total_area_l3280_328043


namespace NUMINAMATH_CALUDE_pairing_theorem_l3280_328070

/-- The number of ways to pair 2n points on a circle with n non-intersecting chords -/
def pairings (n : ℕ) : ℚ :=
  1 / (n + 1 : ℚ) * (Nat.choose (2 * n) n : ℚ)

/-- Theorem stating that the number of ways to pair 2n points on a circle
    with n non-intersecting chords is equal to (1 / (n+1)) * binomial(2n, n) -/
theorem pairing_theorem (n : ℕ) (h : n ≥ 1) :
  pairings n = 1 / (n + 1 : ℚ) * (Nat.choose (2 * n) n : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_pairing_theorem_l3280_328070


namespace NUMINAMATH_CALUDE_parabola_circle_problem_l3280_328008

/-- Parabola in the Cartesian coordinate system -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h_p_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x

/-- Circle in the Cartesian coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The setup of the problem -/
structure ParabolaCircleSetup where
  C : Parabola
  Q : Circle
  h_Q_passes_O : Q.center.1^2 + Q.center.2^2 = Q.radius^2
  h_Q_passes_F : (Q.center.1 - C.p/2)^2 + Q.center.2^2 = Q.radius^2
  h_Q_center_directrix : Q.center.1 + C.p/2 = 3/2

/-- The theorem to be proved -/
theorem parabola_circle_problem (setup : ParabolaCircleSetup) :
  -- 1. The equation of parabola C is y^2 = 4x
  setup.C.p = 2 ∧
  -- 2. For any point M(t, 4) on C and chords MD and ME with MD ⊥ ME,
  --    the line DE passes through the fixed point (8, -4)
  ∀ t : ℝ, setup.C.eq t 4 →
    ∀ D E : ℝ × ℝ, setup.C.eq D.1 D.2 → setup.C.eq E.1 E.2 →
      (t - D.1) * (t - E.1) + (4 - D.2) * (4 - E.2) = 0 →
        ∃ m : ℝ, (D.1 = m * (D.2 + 4) + 8 ∧ E.1 = m * (E.2 + 4) + 8) ∨
                 (D.1 = m * (D.2 - 4) + 4 ∧ E.1 = m * (E.2 - 4) + 4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_circle_problem_l3280_328008


namespace NUMINAMATH_CALUDE_octal_245_equals_decimal_165_l3280_328085

/-- Converts an octal number to decimal --/
def octal_to_decimal (a b c : ℕ) : ℕ := c * 8^2 + b * 8^1 + a * 8^0

/-- Proves that 245 in octal is equal to 165 in decimal --/
theorem octal_245_equals_decimal_165 : octal_to_decimal 5 4 2 = 165 := by
  sorry

end NUMINAMATH_CALUDE_octal_245_equals_decimal_165_l3280_328085


namespace NUMINAMATH_CALUDE_bat_survey_result_l3280_328028

theorem bat_survey_result (total : ℕ) 
  (blind_percent : ℚ) (deaf_percent : ℚ) (deaf_count : ℕ) 
  (h1 : blind_percent = 784/1000) 
  (h2 : deaf_percent = 532/1000) 
  (h3 : deaf_count = 33) : total = 79 :=
by
  sorry

end NUMINAMATH_CALUDE_bat_survey_result_l3280_328028


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3280_328000

/-- Given two hyperbolas x²/9 - y²/16 = 1 and y²/25 - x²/M = 1,
    prove that M = 225/16 for the hyperbolas to have the same asymptotes -/
theorem hyperbolas_same_asymptotes :
  ∀ M : ℝ,
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / M = 1) →
  (∀ k : ℝ, (∃ x y : ℝ, y = k * x ∧ x^2 / 9 - y^2 / 16 = 1) ↔
            (∃ x y : ℝ, y = k * x ∧ y^2 / 25 - x^2 / M = 1)) →
  M = 225 / 16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3280_328000


namespace NUMINAMATH_CALUDE_polynomial_coefficients_sum_l3280_328048

theorem polynomial_coefficients_sum 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h : ∀ x, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) : 
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 1) ∧ (a₀ + a₂ + a₄ + a₆ = 365) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_sum_l3280_328048


namespace NUMINAMATH_CALUDE_g_composition_of_three_l3280_328082

-- Define the function g
def g (x : ℝ) : ℝ := 7 * x - 3

-- State the theorem
theorem g_composition_of_three : g (g (g 3)) = 858 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l3280_328082


namespace NUMINAMATH_CALUDE_unique_set_l3280_328097

def is_valid_set (A : Finset ℤ) : Prop :=
  A.card = 4 ∧
  ∀ (subset : Finset ℤ), subset ⊆ A → subset.card = 3 →
    (subset.sum id) ∈ ({-1, 5, 3, 8} : Finset ℤ)

theorem unique_set :
  ∃! (A : Finset ℤ), is_valid_set A ∧ A = {-3, 0, 2, 6} :=
sorry

end NUMINAMATH_CALUDE_unique_set_l3280_328097


namespace NUMINAMATH_CALUDE_restaurant_theorem_l3280_328098

def restaurant_problem (expenditures : List ℝ) : Prop :=
  let n := 6
  let avg := (List.sum (List.take n expenditures)) / n
  let g_spent := avg - 5
  let h_spent := 2 * (avg - g_spent)
  let total_spent := (List.sum expenditures) + g_spent + h_spent
  expenditures.length = 8 ∧
  List.take n expenditures = [13, 17, 9, 15, 11, 20] ∧
  total_spent = 104.17

theorem restaurant_theorem (expenditures : List ℝ) :
  restaurant_problem expenditures :=
sorry

end NUMINAMATH_CALUDE_restaurant_theorem_l3280_328098


namespace NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l3280_328015

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the specific bridge length problem -/
theorem specific_bridge_length : 
  bridge_length 200 60 25 = 216.75 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l3280_328015


namespace NUMINAMATH_CALUDE_proposition_equivalence_l3280_328037

theorem proposition_equivalence (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, x^2 - a ≤ 0) ↔ a ≥ 9 := by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l3280_328037


namespace NUMINAMATH_CALUDE_checkout_lane_shoppers_l3280_328062

theorem checkout_lane_shoppers (total_shoppers : ℕ) (avoid_fraction : ℚ) : 
  total_shoppers = 480 →
  avoid_fraction = 5/8 →
  total_shoppers - (total_shoppers * avoid_fraction).floor = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_checkout_lane_shoppers_l3280_328062


namespace NUMINAMATH_CALUDE_sugar_spilled_correct_l3280_328004

/-- The amount of sugar Pamela spilled on the floor -/
def sugar_spilled (original : ℝ) (left : ℝ) : ℝ := original - left

/-- Theorem stating that the amount of sugar spilled is correct -/
theorem sugar_spilled_correct (original left : ℝ) 
  (h1 : original = 9.8)
  (h2 : left = 4.6) : 
  sugar_spilled original left = 5.2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_spilled_correct_l3280_328004


namespace NUMINAMATH_CALUDE_triangle_properties_l3280_328011

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem states properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * (Real.cos ((t.C - t.A) / 2))^2 * Real.cos t.A - 
        Real.sin (t.C - t.A) * Real.sin t.A + 
        Real.cos (t.B + t.C) = 1/3)
  (h2 : t.c = 2 * Real.sqrt 2) : 
  Real.sin t.C = (2 * Real.sqrt 2) / 3 ∧ 
  ∃ (max_area : ℝ), max_area = 2 * Real.sqrt 2 ∧ 
    ∀ (area : ℝ), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3280_328011


namespace NUMINAMATH_CALUDE_complement_of_union_is_four_l3280_328019

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3}

theorem complement_of_union_is_four :
  (U \ (A ∪ B)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_is_four_l3280_328019


namespace NUMINAMATH_CALUDE_abc_product_l3280_328023

theorem abc_product (a b c : ℝ) 
  (h1 : a - b = 4)
  (h2 : a^2 + b^2 = 18)
  (h3 : a + b + c = 8) :
  a * b * c = 92 - 50 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l3280_328023


namespace NUMINAMATH_CALUDE_fuel_station_problem_l3280_328033

/-- Represents the problem of determining the number of mini-vans filled up at a fuel station. -/
theorem fuel_station_problem (service_cost truck_count mini_van_tank truck_tank_ratio total_cost fuel_cost : ℚ) :
  service_cost = 210/100 →
  fuel_cost = 70/100 →
  truck_count = 2 →
  mini_van_tank = 65 →
  truck_tank_ratio = 220/100 →
  total_cost = 3472/10 →
  ∃ (mini_van_count : ℚ),
    mini_van_count = 3 ∧
    mini_van_count * (service_cost + mini_van_tank * fuel_cost) +
    truck_count * (service_cost + (mini_van_tank * truck_tank_ratio) * fuel_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_fuel_station_problem_l3280_328033


namespace NUMINAMATH_CALUDE_ashley_wedding_champagne_servings_l3280_328046

/-- The number of servings in one bottle of champagne for Ashley's wedding toast. -/
def servings_per_bottle (guests : ℕ) (glasses_per_guest : ℕ) (total_bottles : ℕ) : ℕ :=
  (guests * glasses_per_guest) / total_bottles

/-- Theorem stating that there are 6 servings in one bottle of champagne for Ashley's wedding toast. -/
theorem ashley_wedding_champagne_servings :
  servings_per_bottle 120 2 40 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ashley_wedding_champagne_servings_l3280_328046


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l3280_328014

theorem prime_sum_theorem (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p + q = r → 1 < p → p < q → p = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l3280_328014


namespace NUMINAMATH_CALUDE_unique_positive_root_implies_a_less_than_neg_one_l3280_328002

/-- Given two functions f and g, if their difference has a unique positive root,
    then the parameter a in f must be less than -1 -/
theorem unique_positive_root_implies_a_less_than_neg_one
  (f g : ℝ → ℝ)
  (h : ∀ x : ℝ, f x = 2 * a * x^3 + 3)
  (k : ∀ x : ℝ, g x = 3 * x^2 + 2)
  (unique_root : ∃! x₀ : ℝ, x₀ > 0 ∧ f x₀ = g x₀) :
  a < -1 :=
sorry

end NUMINAMATH_CALUDE_unique_positive_root_implies_a_less_than_neg_one_l3280_328002


namespace NUMINAMATH_CALUDE_book_cost_l3280_328039

/-- If two identical books cost $36 in total, then eight of these books will cost $144. -/
theorem book_cost (two_books_cost : ℕ) (h : two_books_cost = 36) : 
  (8 * (two_books_cost / 2) = 144) :=
sorry

end NUMINAMATH_CALUDE_book_cost_l3280_328039


namespace NUMINAMATH_CALUDE_absolute_value_complex_l3280_328052

theorem absolute_value_complex : Complex.abs (-1 + (2/3) * Complex.I) = Real.sqrt 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_complex_l3280_328052


namespace NUMINAMATH_CALUDE_tangent_product_simplification_l3280_328029

theorem tangent_product_simplification :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_simplification_l3280_328029


namespace NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l3280_328006

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- Variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_unique_parameters
  (X : BinomialDistribution)
  (h_expectation : expectation X = 8)
  (h_variance : variance X = 1.6) :
  X.n = 10 ∧ X.p = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l3280_328006


namespace NUMINAMATH_CALUDE_three_integers_divisibility_l3280_328092

theorem three_integers_divisibility (x y z : ℕ+) :
  (x ∣ y + z) ∧ (y ∣ x + z) ∧ (z ∣ x + y) →
  (∃ a : ℕ+, (x = a ∧ y = a ∧ z = a) ∨
             (x = a ∧ y = a ∧ z = 2 * a) ∨
             (x = a ∧ y = 2 * a ∧ z = 3 * a)) :=
by sorry

end NUMINAMATH_CALUDE_three_integers_divisibility_l3280_328092


namespace NUMINAMATH_CALUDE_basketball_tryouts_l3280_328072

theorem basketball_tryouts (girls boys called_back : ℕ) 
  (h1 : girls = 42)
  (h2 : boys = 80)
  (h3 : called_back = 25) :
  girls + boys - called_back = 97 := by
sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l3280_328072


namespace NUMINAMATH_CALUDE_suzy_books_wednesday_morning_l3280_328009

/-- The number of books Suzy had at the end of Friday -/
def friday_end : ℕ := 80

/-- The number of books returned on Friday -/
def friday_returned : ℕ := 7

/-- The number of books checked out on Thursday -/
def thursday_checked_out : ℕ := 5

/-- The number of books returned on Thursday -/
def thursday_returned : ℕ := 23

/-- The number of books checked out on Wednesday -/
def wednesday_checked_out : ℕ := 43

/-- The number of books Suzy had on Wednesday morning -/
def wednesday_morning : ℕ := friday_end + friday_returned + thursday_checked_out - thursday_returned + wednesday_checked_out

theorem suzy_books_wednesday_morning : wednesday_morning = 98 := by
  sorry

end NUMINAMATH_CALUDE_suzy_books_wednesday_morning_l3280_328009


namespace NUMINAMATH_CALUDE_orphan_house_donation_percentage_l3280_328013

def total_income : ℝ := 400000
def children_percentage : ℝ := 0.2
def num_children : ℕ := 3
def wife_percentage : ℝ := 0.25
def remaining_amount : ℝ := 60000
def final_amount : ℝ := 40000

theorem orphan_house_donation_percentage :
  let children_share := children_percentage * num_children * total_income
  let wife_share := wife_percentage * total_income
  let donation_amount := remaining_amount - final_amount
  (donation_amount / remaining_amount) * 100 = 100/3 :=
by sorry

end NUMINAMATH_CALUDE_orphan_house_donation_percentage_l3280_328013


namespace NUMINAMATH_CALUDE_kitten_weight_l3280_328034

/-- Given the weights of a kitten and two dogs satisfying certain conditions,
    prove that the kitten weighs 6 pounds. -/
theorem kitten_weight (x y z : ℝ) 
  (h1 : x + y + z = 36)
  (h2 : x + z = 2*y)
  (h3 : x + y = z) :
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_kitten_weight_l3280_328034


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l3280_328001

theorem sqrt_equation_solutions :
  ∀ x : ℚ, (Real.sqrt (9 * x - 4) + 16 / Real.sqrt (9 * x - 4) = 9) ↔ (x = 68/9 ∨ x = 5/9) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l3280_328001


namespace NUMINAMATH_CALUDE_complement_union_problem_l3280_328038

def U : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {1, 3, 4}
def B : Finset Nat := {2, 4}

theorem complement_union_problem :
  (U \ A) ∪ B = {2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_problem_l3280_328038


namespace NUMINAMATH_CALUDE_set_equality_l3280_328069

-- Define the universal set I
def I : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {3, 4, 5}

-- Define set N
def N : Set Nat := {1, 3, 6}

-- Define the set we want to prove equal to (C_I M) ∩ (C_I N)
def target_set : Set Nat := {2, 7}

-- Theorem statement
theorem set_equality : 
  target_set = (I \ M) ∩ (I \ N) := by sorry

end NUMINAMATH_CALUDE_set_equality_l3280_328069


namespace NUMINAMATH_CALUDE_power_of_two_equality_l3280_328012

theorem power_of_two_equality (M : ℕ) : (32^3) * (16^3) = 2^M → M = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l3280_328012


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3280_328091

def A : Set ℤ := {1, 2}
def B : Set ℤ := {-1, 1, 4}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3280_328091


namespace NUMINAMATH_CALUDE_least_possible_BC_l3280_328058

theorem least_possible_BC (AB AC DC BD BC : ℕ) : 
  AB = 7 → 
  AC = 15 → 
  DC = 11 → 
  BD = 25 → 
  BC > AC - AB → 
  BC > BD - DC → 
  BC ≥ 14 ∧ ∀ n : ℕ, (n ≥ 14 → n ≥ BC) → BC = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_least_possible_BC_l3280_328058


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_450_l3280_328051

/-- The number of perfect square factors of 450 -/
def perfect_square_factors_of_450 : ℕ :=
  (Finset.filter (fun n => n^2 ∣ 450) (Finset.range (450 + 1))).card

/-- Theorem: The number of perfect square factors of 450 is 4 -/
theorem count_perfect_square_factors_450 : perfect_square_factors_of_450 = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_450_l3280_328051


namespace NUMINAMATH_CALUDE_P_no_real_roots_l3280_328050

/-- Recursive definition of the polynomial sequence P_n(x) -/
def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | n + 1, x => x^(11 * (n + 1)) - P n x

/-- Theorem stating that P_n(x) has no real roots for all n ≥ 0 -/
theorem P_no_real_roots : ∀ (n : ℕ) (x : ℝ), P n x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_P_no_real_roots_l3280_328050


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3280_328073

theorem quadratic_equation_roots (c : ℝ) :
  (2 + Real.sqrt 3 : ℝ) ^ 2 - 4 * (2 + Real.sqrt 3) + c = 0 →
  (2 - Real.sqrt 3 : ℝ) ^ 2 - 4 * (2 - Real.sqrt 3) + c = 0 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3280_328073


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3280_328030

/-- Given that the midpoint of (k, 0) and (b, 0) is (-1, 0),
    prove that the line y = kx + b passes through (1, -2) -/
theorem line_passes_through_point
  (k b : ℝ) -- k and b are real numbers
  (h : (k + b) / 2 = -1) -- midpoint condition
  : k * 1 + b = -2 := by -- line passes through (1, -2)
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3280_328030


namespace NUMINAMATH_CALUDE_softball_team_ratio_l3280_328044

/-- Represents a co-ed softball team with different skill levels -/
structure SoftballTeam where
  beginnerMen : ℕ
  beginnerWomen : ℕ
  intermediateMen : ℕ
  intermediateWomen : ℕ
  advancedMen : ℕ
  advancedWomen : ℕ

/-- Theorem stating the ratio of men to women on the softball team -/
theorem softball_team_ratio (team : SoftballTeam) : 
  team.beginnerMen = 2 ∧ 
  team.beginnerWomen = 4 ∧
  team.intermediateMen = 3 ∧
  team.intermediateWomen = 5 ∧
  team.advancedMen = 1 ∧
  team.advancedWomen = 3 →
  (team.beginnerMen + team.intermediateMen + team.advancedMen) * 2 = 
  (team.beginnerWomen + team.intermediateWomen + team.advancedWomen) := by
  sorry

#check softball_team_ratio

end NUMINAMATH_CALUDE_softball_team_ratio_l3280_328044


namespace NUMINAMATH_CALUDE_triangle_equation_solution_l3280_328059

theorem triangle_equation_solution (a b c : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a < b + c) :
  let p := (a + b + c) / 2
  let x := a * b * c / (2 * Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  b * Real.sqrt (x^2 - c^2) + c * Real.sqrt (x^2 - b^2) = a * x := by
sorry

end NUMINAMATH_CALUDE_triangle_equation_solution_l3280_328059


namespace NUMINAMATH_CALUDE_root_of_fifth_unity_l3280_328060

theorem root_of_fifth_unity {p q r s t m : ℂ} (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + t = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + t * m + p = 0) :
  m^5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_of_fifth_unity_l3280_328060


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3280_328063

theorem binomial_coefficient_equality (x : ℕ) : 
  Nat.choose 28 x = Nat.choose 28 (2 * x - 1) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3280_328063


namespace NUMINAMATH_CALUDE_cubic_factorization_l3280_328096

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3280_328096


namespace NUMINAMATH_CALUDE_part_one_part_two_l3280_328099

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- Part I
theorem part_one (a : ℝ) :
  (∀ x : ℝ, f a x + f a (x - 2) ≥ 1) → (a ≥ 1/2 ∨ a ≤ -1/2) :=
sorry

-- Part II
theorem part_two (a b c : ℝ) :
  f a ((a - 1) / a) + f a ((b - 1) / a) + f a ((c - 1) / a) = 4 →
  (f a ((a^2 - 1) / a) + f a ((b^2 - 1) / a) + f a ((c^2 - 1) / a) ≥ 16/3 ∧
   ∃ x y z : ℝ, f a ((x^2 - 1) / a) + f a ((y^2 - 1) / a) + f a ((z^2 - 1) / a) = 16/3) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3280_328099


namespace NUMINAMATH_CALUDE_inequality_proof_l3280_328024

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a + b + c) / 3 - (a * b * c) ^ (1/3) ≤ max ((a^(1/2) - b^(1/2))^2) (max ((b^(1/2) - c^(1/2))^2) ((c^(1/2) - a^(1/2))^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3280_328024


namespace NUMINAMATH_CALUDE_number_times_one_fourth_squared_l3280_328064

theorem number_times_one_fourth_squared (x : ℝ) : x * (1/4)^2 = 4^3 → x = 1024 := by
  sorry

end NUMINAMATH_CALUDE_number_times_one_fourth_squared_l3280_328064


namespace NUMINAMATH_CALUDE_linear_system_solution_l3280_328084

/-- Given a system of linear equations and a condition on its solution, prove the value of k. -/
theorem linear_system_solution (x y k : ℝ) : 
  3 * x + 2 * y = k + 1 →
  2 * x + 3 * y = k →
  x + y = 3 →
  k = 7 := by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l3280_328084


namespace NUMINAMATH_CALUDE_distance_to_place_l3280_328021

/-- Proves that the distance to a place is 144 km given the rowing speed, current speed, and total round trip time. -/
theorem distance_to_place (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) :
  rowing_speed = 10 →
  current_speed = 2 →
  total_time = 30 →
  (total_time * (rowing_speed + current_speed) * (rowing_speed - current_speed)) / (2 * rowing_speed) = 144 := by
sorry

end NUMINAMATH_CALUDE_distance_to_place_l3280_328021


namespace NUMINAMATH_CALUDE_face_value_of_shares_l3280_328086

/-- Calculates the face value of shares given investment details -/
theorem face_value_of_shares
  (investment : ℝ)
  (quoted_price : ℝ)
  (dividend_rate : ℝ)
  (annual_income : ℝ)
  (h1 : investment = 4940)
  (h2 : quoted_price = 9.5)
  (h3 : dividend_rate = 0.14)
  (h4 : annual_income = 728)
  : ∃ (face_value : ℝ),
    face_value = 10 ∧
    annual_income = (investment / quoted_price) * (dividend_rate * face_value) :=
by sorry

end NUMINAMATH_CALUDE_face_value_of_shares_l3280_328086


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3280_328080

theorem quadratic_roots_condition (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + x - m = 0 ∧ y^2 + y - m = 0) → m > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3280_328080


namespace NUMINAMATH_CALUDE_fathers_with_full_time_jobs_l3280_328078

theorem fathers_with_full_time_jobs 
  (total_parents : ℝ) 
  (mothers_ratio : ℝ) 
  (mothers_full_time_ratio : ℝ) 
  (no_full_time_ratio : ℝ) 
  (h1 : mothers_ratio = 0.6) 
  (h2 : mothers_full_time_ratio = 5/6) 
  (h3 : no_full_time_ratio = 0.2) : 
  (total_parents * (1 - mothers_ratio) * 3/4) = 
  (total_parents * (1 - no_full_time_ratio) - total_parents * mothers_ratio * mothers_full_time_ratio) := by
sorry

end NUMINAMATH_CALUDE_fathers_with_full_time_jobs_l3280_328078


namespace NUMINAMATH_CALUDE_interval_necessary_not_sufficient_l3280_328071

theorem interval_necessary_not_sufficient :
  ¬(∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 ↔ (x - 5) * (x + 1) < 0) ∧
  (∀ x : ℝ, (x - 5) * (x + 1) < 0 → -1 ≤ x ∧ x ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_interval_necessary_not_sufficient_l3280_328071


namespace NUMINAMATH_CALUDE_inequality_proof_l3280_328036

theorem inequality_proof (x : ℝ) (h1 : (3/2 : ℝ) ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3280_328036


namespace NUMINAMATH_CALUDE_triangle_longest_side_l3280_328032

theorem triangle_longest_side (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (ratio : a / 5 = b / 6 ∧ b / 6 = c / 7)
  (perimeter : a + b + c = 720) :
  c = 280 := by
sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l3280_328032


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3280_328093

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3280_328093


namespace NUMINAMATH_CALUDE_max_value_2x_minus_y_l3280_328045

theorem max_value_2x_minus_y (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0) 
  (h2 : y + 1 ≥ 0) 
  (h3 : x + y + 1 ≤ 0) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ (x' y' : ℝ), 
    x' - y' + 1 ≥ 0 → y' + 1 ≥ 0 → x' + y' + 1 ≤ 0 → 2*x' - y' ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_minus_y_l3280_328045


namespace NUMINAMATH_CALUDE_melissas_fabric_l3280_328053

/-- The amount of fabric Melissa has given her work hours and dress requirements -/
theorem melissas_fabric (fabric_per_dress : ℝ) (hours_per_dress : ℝ) (total_work_hours : ℝ) :
  fabric_per_dress = 4 →
  hours_per_dress = 3 →
  total_work_hours = 42 →
  (total_work_hours / hours_per_dress) * fabric_per_dress = 56 := by
  sorry

end NUMINAMATH_CALUDE_melissas_fabric_l3280_328053


namespace NUMINAMATH_CALUDE_height_on_hypotenuse_of_right_triangle_l3280_328047

theorem height_on_hypotenuse_of_right_triangle (a b h c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → (1/2) * a * b = (1/2) * c * h → h = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_height_on_hypotenuse_of_right_triangle_l3280_328047


namespace NUMINAMATH_CALUDE_orange_cost_calculation_l3280_328031

/-- Given the cost of 5 dozen oranges, calculate the cost of 8 dozen oranges at the same rate -/
theorem orange_cost_calculation (cost_five_dozen : ℝ) : cost_five_dozen = 42 →
  (8 : ℝ) * (cost_five_dozen / 5) = 67.2 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_calculation_l3280_328031


namespace NUMINAMATH_CALUDE_projection_v_onto_w_l3280_328017

def v : Fin 2 → ℝ := ![3, -1]
def w : Fin 2 → ℝ := ![4, 2]

theorem projection_v_onto_w :
  (((v • w) / (w • w)) • w) = ![2, 1] := by sorry

end NUMINAMATH_CALUDE_projection_v_onto_w_l3280_328017


namespace NUMINAMATH_CALUDE_max_squares_covered_two_inch_card_l3280_328025

/-- Represents a square card -/
structure Card where
  side_length : ℝ

/-- Represents a checkerboard square -/
structure BoardSquare where
  side_length : ℝ

/-- Calculates the maximum number of board squares that can be covered by a card -/
def max_squares_covered (card : Card) (board_square : BoardSquare) : ℕ :=
  sorry

/-- Theorem stating the maximum number of squares covered by a 2-inch card on a board of 1-inch squares -/
theorem max_squares_covered_two_inch_card :
  let card := Card.mk 2
  let board_square := BoardSquare.mk 1
  max_squares_covered card board_square = 16 := by
    sorry

end NUMINAMATH_CALUDE_max_squares_covered_two_inch_card_l3280_328025


namespace NUMINAMATH_CALUDE_min_value_inequality_l3280_328007

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * ((1 / (x + y)) + (1 / (y + z)) + (1 / (z + x))) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3280_328007


namespace NUMINAMATH_CALUDE_smallest_norm_v_l3280_328090

open Real
open Vector

/-- Given a vector v such that ||v + (-2, 4)|| = 10, the smallest possible value of ||v|| is 10 - 2√5 -/
theorem smallest_norm_v (v : ℝ × ℝ) (h : ‖v + (-2, 4)‖ = 10) :
  ∀ w : ℝ × ℝ, ‖w + (-2, 4)‖ = 10 → ‖v‖ ≤ ‖w‖ ∧ ‖v‖ = 10 - 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_norm_v_l3280_328090


namespace NUMINAMATH_CALUDE_equal_angles_necessary_not_sufficient_l3280_328041

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  sorry -- Definition of a square

-- Define the property of having equal interior angles
def has_equal_interior_angles (q : Quadrilateral) : Prop :=
  sorry -- Definition of equal interior angles

theorem equal_angles_necessary_not_sufficient :
  (∀ q : Quadrilateral, is_square q → has_equal_interior_angles q) ∧
  (∃ q : Quadrilateral, has_equal_interior_angles q ∧ ¬is_square q) :=
sorry

end NUMINAMATH_CALUDE_equal_angles_necessary_not_sufficient_l3280_328041


namespace NUMINAMATH_CALUDE_roses_to_sister_l3280_328018

-- Define the initial number of roses
def initial_roses : ℕ := 20

-- Define the number of roses given to mother
def roses_to_mother : ℕ := 6

-- Define the number of roses given to grandmother
def roses_to_grandmother : ℕ := 9

-- Define the number of roses Ian kept for himself
def roses_kept : ℕ := 1

-- Theorem to prove
theorem roses_to_sister : 
  initial_roses - (roses_to_mother + roses_to_grandmother + roses_kept) = 4 := by
  sorry

end NUMINAMATH_CALUDE_roses_to_sister_l3280_328018


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_is_80_l3280_328027

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : Nat
  width : Nat
  depth : Nat

/-- Calculates the smallest number of identical cubes needed to fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : Nat :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem stating that the smallest number of cubes to fill the given box is 80 -/
theorem smallest_number_of_cubes_is_80 :
  smallestNumberOfCubes ⟨30, 48, 12⟩ = 80 := by
  sorry

#eval smallestNumberOfCubes ⟨30, 48, 12⟩

end NUMINAMATH_CALUDE_smallest_number_of_cubes_is_80_l3280_328027


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3280_328026

theorem regular_polygon_sides (interior_angle : ℝ) (n : ℕ) :
  interior_angle = 144 →
  (n : ℝ) * (180 - interior_angle) = 360 →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3280_328026


namespace NUMINAMATH_CALUDE_range_of_f_l3280_328076

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -1 ≤ y ∧ y ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l3280_328076


namespace NUMINAMATH_CALUDE_solve_for_y_l3280_328042

theorem solve_for_y (x y : ℝ) (h1 : x^2 + 3*x + 7 = y - 5) (h2 : x = -4) : y = 16 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3280_328042


namespace NUMINAMATH_CALUDE_corn_spacing_theorem_l3280_328067

/-- Calculates the space required for each seed in a row of corn. -/
def space_per_seed (row_length_feet : ℕ) (seeds_per_row : ℕ) : ℕ :=
  (row_length_feet * 12) / seeds_per_row

/-- Theorem: Given a row length of 120 feet and 80 seeds per row, 
    the space required for each seed is 18 inches. -/
theorem corn_spacing_theorem : space_per_seed 120 80 = 18 := by
  sorry

end NUMINAMATH_CALUDE_corn_spacing_theorem_l3280_328067


namespace NUMINAMATH_CALUDE_quadratic_max_value_l3280_328040

theorem quadratic_max_value :
  let f : ℝ → ℝ := fun z ↦ -4 * z^2 + 20 * z - 6
  ∃ (max : ℝ), max = 19 ∧ ∀ z, f z ≤ max :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l3280_328040


namespace NUMINAMATH_CALUDE_reflected_ray_passes_through_C_l3280_328077

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the reflected ray equation
def reflected_ray_equation (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Theorem statement
theorem reflected_ray_passes_through_C : 
  ∃ C : ℝ × ℝ, C.1 = 1 ∧ C.2 = 4 ∧ reflected_ray_equation C.1 C.2 := by sorry

end NUMINAMATH_CALUDE_reflected_ray_passes_through_C_l3280_328077


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l3280_328088

theorem divisibility_by_twelve (n : ℕ) : 
  (713 * 10 + n ≥ 1000) ∧ 
  (713 * 10 + n < 10000) ∧ 
  (713 * 10 + n) % 12 = 0 ↔ 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l3280_328088


namespace NUMINAMATH_CALUDE_fraction_subtraction_complex_fraction_division_l3280_328075

-- Define a and b as real numbers
variable (a b : ℝ)

-- Assumption that a ≠ b
variable (h : a ≠ b)

-- First theorem
theorem fraction_subtraction : (b / (a - b)) - (a / (a - b)) = -1 := by sorry

-- Second theorem
theorem complex_fraction_division : 
  ((a^2 - a*b) / a^2) / ((a / b) - (b / a)) = b / (a + b) := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_complex_fraction_division_l3280_328075


namespace NUMINAMATH_CALUDE_number_wall_top_value_l3280_328057

/-- Represents a number wall pyramid --/
structure NumberWall :=
  (bottom_left : ℕ)
  (bottom_middle : ℕ)
  (bottom_right : ℕ)

/-- Calculates the value at the top of the number wall pyramid --/
def top_value (wall : NumberWall) : ℕ :=
  let m := wall.bottom_left + wall.bottom_middle
  let n := wall.bottom_middle + wall.bottom_right
  let left_mid := wall.bottom_left + m
  let right_mid := m + n
  let left_top := left_mid + right_mid
  let right_top := right_mid + wall.bottom_right
  2 * (left_top + right_top)

/-- Theorem stating that the top value of the given number wall is 320 --/
theorem number_wall_top_value :
  ∃ (wall : NumberWall), wall.bottom_left = 20 ∧ wall.bottom_middle = 34 ∧ wall.bottom_right = 44 ∧ top_value wall = 320 :=
sorry

end NUMINAMATH_CALUDE_number_wall_top_value_l3280_328057


namespace NUMINAMATH_CALUDE_equation_transformation_correct_l3280_328003

theorem equation_transformation_correct (x : ℝ) :
  (x + 1) / 2 - 1 = (2 * x - 1) / 3 ↔ 3 * (x + 1) - 6 = 2 * (2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_correct_l3280_328003


namespace NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l3280_328049

theorem largest_n_for_trig_inequality :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≤ Real.sqrt n / 2) ∧
  (∀ (m : ℕ), m > n → ∃ (x : ℝ), (Real.sin x)^m + (Real.cos x)^m > Real.sqrt m / 2) ∧
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l3280_328049


namespace NUMINAMATH_CALUDE_distance_after_eight_hours_l3280_328016

/-- The distance between two trains after a given time -/
def distance_between_trains (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed2 - speed1) * time

/-- Theorem: The distance between two trains after 8 hours -/
theorem distance_after_eight_hours :
  distance_between_trains 11 31 8 = 160 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_eight_hours_l3280_328016


namespace NUMINAMATH_CALUDE_trig_system_relation_l3280_328066

/-- Given a system of trigonometric equations, prove the relationship between a, b, and c -/
theorem trig_system_relation (x y a b c : ℝ) 
  (h1 : Real.sin x + Real.sin y = 2 * a)
  (h2 : Real.cos x + Real.cos y = 2 * b)
  (h3 : Real.tan x + Real.tan y = 2 * c) :
  a * (b + a * c) = c * (a^2 + b^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_system_relation_l3280_328066
