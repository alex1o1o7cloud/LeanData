import Mathlib

namespace NUMINAMATH_CALUDE_sword_length_difference_is_23_l1907_190709

/-- The length difference between June's and Christopher's swords -/
def sword_length_difference : ℕ → ℕ → ℕ → ℕ
  | christopher_length, jameson_diff, june_diff =>
    let jameson_length := 2 * christopher_length + jameson_diff
    let june_length := jameson_length + june_diff
    june_length - christopher_length

theorem sword_length_difference_is_23 :
  sword_length_difference 15 3 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_sword_length_difference_is_23_l1907_190709


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l1907_190718

theorem cube_plus_reciprocal_cube (x : ℝ) (h1 : x > 0) (h2 : (x + 1/x)^2 = 25) :
  x^3 + 1/x^3 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l1907_190718


namespace NUMINAMATH_CALUDE_machine_production_l1907_190770

/-- The number of shirts produced by a machine given its production rate and working time -/
def shirts_produced (rate : ℕ) (time : ℕ) : ℕ := rate * time

/-- Theorem: A machine producing 2 shirts per minute for 6 minutes makes 12 shirts -/
theorem machine_production : shirts_produced 2 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_machine_production_l1907_190770


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_property_l1907_190717

/-- Given a geometric sequence with first term a₁ and common ratio q,
    the sum of the 4th, 5th, and 6th terms squared equals the product of
    the sum of the 1st, 2nd, and 3rd terms and the sum of the 7th, 8th, and 9th terms. -/
theorem geometric_sequence_sum_property (a₁ q : ℝ) :
  (a₁ * q^3 + a₁ * q^4 + a₁ * q^5)^2 = (a₁ + a₁ * q + a₁ * q^2) * (a₁ * q^6 + a₁ * q^7 + a₁ * q^8) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_property_l1907_190717


namespace NUMINAMATH_CALUDE_max_baggies_l1907_190751

def chocolate_chip_cookies : ℕ := 2
def oatmeal_cookies : ℕ := 16
def cookies_per_bag : ℕ := 3

theorem max_baggies : 
  (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_bag = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_baggies_l1907_190751


namespace NUMINAMATH_CALUDE_farmer_apples_l1907_190796

theorem farmer_apples (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 924 → given_away = 639 → remaining = initial - given_away → remaining = 285 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l1907_190796


namespace NUMINAMATH_CALUDE_container_volume_ratio_l1907_190787

theorem container_volume_ratio : 
  ∀ (A B : ℝ), A > 0 → B > 0 →
  (3/5 * A + 1/4 * B = 4/5 * B) →
  A / B = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l1907_190787


namespace NUMINAMATH_CALUDE_sqrt_2_times_sqrt_24_between_6_and_7_l1907_190765

theorem sqrt_2_times_sqrt_24_between_6_and_7 : 6 < Real.sqrt 2 * Real.sqrt 24 ∧ Real.sqrt 2 * Real.sqrt 24 < 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_times_sqrt_24_between_6_and_7_l1907_190765


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l1907_190755

theorem no_positive_integer_solution (p : ℕ) (x y : ℕ) (hp : p > 3) (hp_prime : Nat.Prime p) (hx : p ∣ x) :
  ¬(x^2 - 1 = y^p) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l1907_190755


namespace NUMINAMATH_CALUDE_inhomogeneous_system_solution_l1907_190734

variable (α₁ α₂ α₃ : ℝ)

def X_gen : Fin 4 → ℝ := fun i =>
  match i with
  | 0 => -7 * α₁ + 8 * α₂ - 9 * α₃ + 4
  | 1 => α₁
  | 2 => α₂
  | 3 => α₃

theorem inhomogeneous_system_solution :
  X_gen α₁ α₂ α₃ 0 + 7 * X_gen α₁ α₂ α₃ 1 - 8 * X_gen α₁ α₂ α₃ 2 + 9 * X_gen α₁ α₂ α₃ 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_inhomogeneous_system_solution_l1907_190734


namespace NUMINAMATH_CALUDE_fraction_equality_l1907_190757

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (4 * a + b) / (a - 4 * b) = 3) : 
  (a + 4 * b) / (4 * a - b) = 9 / 53 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1907_190757


namespace NUMINAMATH_CALUDE_distance_between_trees_l1907_190723

/-- Given a yard of length 325 meters with 26 trees planted at equal distances,
    including one tree at each end, the distance between consecutive trees is 13 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) : 
  yard_length = 325 → num_trees = 26 → (yard_length / (num_trees - 1 : ℝ)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l1907_190723


namespace NUMINAMATH_CALUDE_store_optimal_plan_l1907_190792

/-- Represents the types of soccer balls -/
inductive BallType
| A
| B

/-- Represents the store's inventory and pricing -/
structure Store where
  cost_price : BallType → ℕ
  selling_price : BallType → ℕ
  budget : ℕ

/-- Represents the purchase plan -/
structure PurchasePlan where
  num_A : ℕ
  num_B : ℕ

def Store.is_valid (s : Store) : Prop :=
  s.cost_price BallType.A = s.cost_price BallType.B + 40 ∧
  480 / s.cost_price BallType.A = 240 / s.cost_price BallType.B ∧
  s.budget = 4000 ∧
  s.selling_price BallType.A = 100 ∧
  s.selling_price BallType.B = 55

def PurchasePlan.is_valid (p : PurchasePlan) (s : Store) : Prop :=
  p.num_A ≥ p.num_B ∧
  p.num_A * s.cost_price BallType.A + p.num_B * s.cost_price BallType.B ≤ s.budget

def PurchasePlan.profit (p : PurchasePlan) (s : Store) : ℤ :=
  (s.selling_price BallType.A - s.cost_price BallType.A) * p.num_A +
  (s.selling_price BallType.B - s.cost_price BallType.B) * p.num_B

theorem store_optimal_plan (s : Store) (h : s.is_valid) :
  ∃ (p : PurchasePlan), 
    p.is_valid s ∧ 
    s.cost_price BallType.A = 80 ∧ 
    s.cost_price BallType.B = 40 ∧
    p.num_A = 34 ∧ 
    p.num_B = 32 ∧
    ∀ (p' : PurchasePlan), p'.is_valid s → p.profit s ≥ p'.profit s :=
sorry

end NUMINAMATH_CALUDE_store_optimal_plan_l1907_190792


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l1907_190725

theorem sqrt_twelve_minus_sqrt_three_equals_sqrt_three : 
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l1907_190725


namespace NUMINAMATH_CALUDE_star_seven_five_l1907_190753

/-- The star operation for positive integers -/
def star (a b : ℕ+) : ℚ :=
  (a * b - (a - b)) / (a + b)

/-- Theorem stating that 7 ★ 5 = 11/4 -/
theorem star_seven_five : star 7 5 = 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_star_seven_five_l1907_190753


namespace NUMINAMATH_CALUDE_irreducible_fraction_l1907_190730

theorem irreducible_fraction (n : ℤ) : Int.gcd (39*n + 4) (26*n + 3) = 1 := by sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l1907_190730


namespace NUMINAMATH_CALUDE_emilee_earnings_l1907_190719

/-- Proves that Emilee earns $25 given the conditions of the problem -/
theorem emilee_earnings (total : ℕ) (terrence_earnings : ℕ) (jermaine_extra : ℕ) :
  total = 90 →
  terrence_earnings = 30 →
  jermaine_extra = 5 →
  total = terrence_earnings + (terrence_earnings + jermaine_extra) + (total - (terrence_earnings + (terrence_earnings + jermaine_extra))) →
  (total - (terrence_earnings + (terrence_earnings + jermaine_extra))) = 25 := by
  sorry

#check emilee_earnings

end NUMINAMATH_CALUDE_emilee_earnings_l1907_190719


namespace NUMINAMATH_CALUDE_total_fish_is_36_l1907_190790

/-- The total number of fish caught by Carla, Kyle, and Tasha -/
def total_fish (carla_fish kyle_fish : ℕ) : ℕ :=
  carla_fish + kyle_fish + kyle_fish

/-- Theorem: Given the conditions, the total number of fish caught is 36 -/
theorem total_fish_is_36 (carla_fish kyle_fish : ℕ) 
  (h1 : carla_fish = 8)
  (h2 : kyle_fish = 14) :
  total_fish carla_fish kyle_fish = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_is_36_l1907_190790


namespace NUMINAMATH_CALUDE_modulus_of_z_l1907_190705

theorem modulus_of_z (z : ℂ) : z = Complex.I * (2 - Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1907_190705


namespace NUMINAMATH_CALUDE_polynomial_value_at_zero_l1907_190733

theorem polynomial_value_at_zero (p : Polynomial ℝ) : 
  (Polynomial.degree p = 7) →
  (∀ n : Nat, n ≤ 7 → p.eval (3^n) = (3^n)⁻¹) →
  p.eval 0 = 19682 / 6561 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_zero_l1907_190733


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_squared_l1907_190752

theorem imaginary_part_of_one_plus_i_squared (z : ℂ) : z = 1 + Complex.I → Complex.im (z^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_squared_l1907_190752


namespace NUMINAMATH_CALUDE_sticker_cost_theorem_l1907_190767

/-- Calculates the total cost of buying stickers on two days -/
def total_sticker_cost (day1_packs : ℕ) (day1_price : ℚ) (day1_discount : ℚ)
                       (day2_packs : ℕ) (day2_price : ℚ) (day2_tax : ℚ) : ℚ :=
  let day1_cost := day1_packs * day1_price * (1 - day1_discount)
  let day2_cost := day2_packs * day2_price * (1 + day2_tax)
  day1_cost + day2_cost

/-- Theorem stating the total cost of buying stickers on two days -/
theorem sticker_cost_theorem :
  total_sticker_cost 15 (5/2) (1/10) 25 3 (1/20) = 225/2 := by
  sorry

end NUMINAMATH_CALUDE_sticker_cost_theorem_l1907_190767


namespace NUMINAMATH_CALUDE_derivative_at_three_l1907_190714

/-- Given a function f with f(x) = 3x^2 + 2xf'(1) for all x, prove that f'(3) = 6 -/
theorem derivative_at_three (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x^2 + 2 * x * (deriv f 1)) :
  deriv f 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_three_l1907_190714


namespace NUMINAMATH_CALUDE_positive_divisors_of_90_l1907_190737

theorem positive_divisors_of_90 : Finset.card (Nat.divisors 90) = 12 := by
  sorry

end NUMINAMATH_CALUDE_positive_divisors_of_90_l1907_190737


namespace NUMINAMATH_CALUDE_enrollment_increase_l1907_190731

/-- Theorem: Enrollment Increase Calculation

Given:
- Enrollment at the beginning of 1992 was 20% greater than at the beginning of 1991
- Enrollment at the beginning of 1993 was 26% greater than at the beginning of 1991

Prove:
The percent increase in enrollment from the beginning of 1992 to the beginning of 1993 is 5%
-/
theorem enrollment_increase (e : ℝ) : 
  let e_1992 := 1.20 * e
  let e_1993 := 1.26 * e
  (e_1993 - e_1992) / e_1992 * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_enrollment_increase_l1907_190731


namespace NUMINAMATH_CALUDE_ramesh_refrigerator_price_l1907_190713

/-- The price Ramesh paid for a refrigerator given specific conditions -/
theorem ramesh_refrigerator_price (P : ℝ) 
  (h1 : 1.1 * P = 17600)  -- Selling price for 10% profit without discount
  (h2 : 0.2 * P = P - 0.8 * P)  -- 20% discount on labelled price
  (h3 : 125 = 125)  -- Transport cost
  (h4 : 250 = 250)  -- Installation cost
  : 0.8 * P + 125 + 250 = 13175 := by
  sorry

end NUMINAMATH_CALUDE_ramesh_refrigerator_price_l1907_190713


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1907_190784

def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, 2, 3; 0, 1, 4; 5, 0, 1]

theorem matrix_equation_solution :
  ∃ (p q r : ℤ), 
    B^3 + p • B^2 + q • B + r • (1 : Matrix (Fin 3) (Fin 3) ℤ) = (0 : Matrix (Fin 3) (Fin 3) ℤ) ∧
    p = -41 ∧ q = -80 ∧ r = -460 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1907_190784


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1907_190727

theorem min_value_quadratic (y : ℝ) : 
  (5 * y^2 + 5 * y + 4 = 9) → 
  (∀ z : ℝ, 5 * z^2 + 5 * z + 4 = 9 → y ≤ z) → 
  y = (-1 - Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1907_190727


namespace NUMINAMATH_CALUDE_square_cross_section_cylinder_volume_l1907_190722

/-- A cylinder with a square cross-section and lateral surface area 4π has volume 2π -/
theorem square_cross_section_cylinder_volume (h : ℝ) (r : ℝ) : 
  h > 0 → r > 0 → h = 2*r → h * (4*r) = 4*π → π * r^2 * h = 2*π := by
  sorry

end NUMINAMATH_CALUDE_square_cross_section_cylinder_volume_l1907_190722


namespace NUMINAMATH_CALUDE_sequences_sum_product_l1907_190745

/-- Two sequences satisfying the given conditions -/
def Sequences (α β γ : ℕ) (a b : ℕ → ℕ) : Prop :=
  (α < γ) ∧ 
  (α * γ = β^2 + 1) ∧
  (a 0 = 1) ∧ 
  (b 0 = 1) ∧
  (∀ n, a (n + 1) = α * a n + β * b n) ∧
  (∀ n, b (n + 1) = β * a n + γ * b n)

/-- The main theorem to be proved -/
theorem sequences_sum_product (α β γ : ℕ) (a b : ℕ → ℕ) 
  (h : Sequences α β γ a b) :
  ∀ m n : ℕ, a (m + n) + b (m + n) = a m * a n + b m * b n :=
sorry

end NUMINAMATH_CALUDE_sequences_sum_product_l1907_190745


namespace NUMINAMATH_CALUDE_long_tennis_players_l1907_190768

theorem long_tennis_players (total : ℕ) (football : ℕ) (both : ℕ) (neither : ℕ) :
  total = 36 →
  football = 26 →
  both = 17 →
  neither = 7 →
  ∃ (long_tennis : ℕ), long_tennis = 20 ∧ 
    total = football + long_tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_long_tennis_players_l1907_190768


namespace NUMINAMATH_CALUDE_multiples_between_2000_and_3000_l1907_190795

def count_multiples (lower upper lcm : ℕ) : ℕ :=
  (upper / lcm) - ((lower - 1) / lcm)

theorem multiples_between_2000_and_3000 : count_multiples 2000 3000 72 = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiples_between_2000_and_3000_l1907_190795


namespace NUMINAMATH_CALUDE_sample_correlation_strength_theorem_l1907_190738

/-- Sample correlation coefficient -/
def sample_correlation_coefficient (data : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Strength of linear relationship -/
def linear_relationship_strength (r : ℝ) : ℝ :=
  sorry

theorem sample_correlation_strength_theorem (data : Set (ℝ × ℝ)) :
  let r := sample_correlation_coefficient data
  ∀ s : ℝ, s ∈ Set.Icc (-1 : ℝ) 1 →
    linear_relationship_strength r = linear_relationship_strength (abs r) ∧
    (abs r > abs s → linear_relationship_strength r > linear_relationship_strength s) :=
  sorry

end NUMINAMATH_CALUDE_sample_correlation_strength_theorem_l1907_190738


namespace NUMINAMATH_CALUDE_larger_number_problem_l1907_190747

theorem larger_number_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 4 * y = 6 * x) (h4 : x + y = 50) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1907_190747


namespace NUMINAMATH_CALUDE_max_x_value_l1907_190726

theorem max_x_value (x : ℝ) : 
  ((5*x - 25)/(4*x - 5))^2 + ((5*x - 25)/(4*x - 5)) = 18 → x ≤ 55/29 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l1907_190726


namespace NUMINAMATH_CALUDE_factorization_sum_l1907_190788

theorem factorization_sum (a b : ℤ) : 
  (∀ x : ℝ, 24 * x^2 - 50 * x - 84 = (6 * x + a) * (4 * x + b)) → 
  a + 2 * b = -17 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l1907_190788


namespace NUMINAMATH_CALUDE_plan_a_monthly_fee_is_9_l1907_190786

/-- Represents the cost per text message for Plan A -/
def plan_a_cost_per_text : ℚ := 25 / 100

/-- Represents the cost per text message for the other plan -/
def other_plan_cost_per_text : ℚ := 40 / 100

/-- Represents the number of text messages at which both plans cost the same -/
def equal_cost_messages : ℕ := 60

/-- The monthly fee for Plan A makes both plans cost the same at 60 messages -/
theorem plan_a_monthly_fee_is_9 :
  ∃ (monthly_fee : ℚ),
    plan_a_cost_per_text * equal_cost_messages + monthly_fee =
    other_plan_cost_per_text * equal_cost_messages ∧
    monthly_fee = 9 := by
  sorry

end NUMINAMATH_CALUDE_plan_a_monthly_fee_is_9_l1907_190786


namespace NUMINAMATH_CALUDE_tan_product_seventh_pi_l1907_190772

theorem tan_product_seventh_pi : 
  Real.tan (π / 7) * Real.tan (2 * π / 7) * Real.tan (3 * π / 7) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_seventh_pi_l1907_190772


namespace NUMINAMATH_CALUDE_circle_with_n_integer_points_l1907_190700

/-- A point on the coordinate plane with rational x-coordinate and irrational y-coordinate -/
structure SpecialPoint where
  x : ℚ
  y : ℝ
  y_irrational : Irrational y

/-- The number of integer points inside a circle -/
def IntegerPointsInside (center : ℝ × ℝ) (radius : ℝ) : ℕ :=
  sorry

/-- Theorem: For any non-negative integer n, there exists a circle on the coordinate plane
    that contains exactly n integer points in its interior -/
theorem circle_with_n_integer_points (n : ℕ) :
  ∃ (center : ℝ × ℝ) (radius : ℝ), IntegerPointsInside center radius = n :=
sorry

end NUMINAMATH_CALUDE_circle_with_n_integer_points_l1907_190700


namespace NUMINAMATH_CALUDE_yoque_monthly_payment_l1907_190759

/-- Calculates the monthly payment for a loan with interest -/
def monthly_payment (principal : ℚ) (months : ℕ) (interest_rate : ℚ) : ℚ :=
  (principal * (1 + interest_rate)) / months

/-- Proves that the monthly payment is $15 given the problem conditions -/
theorem yoque_monthly_payment :
  let principal : ℚ := 150
  let months : ℕ := 11
  let interest_rate : ℚ := 1/10
  monthly_payment principal months interest_rate = 15 := by
sorry

end NUMINAMATH_CALUDE_yoque_monthly_payment_l1907_190759


namespace NUMINAMATH_CALUDE_theorem_3_squeeze_theorem_l1907_190704

-- Theorem 3
theorem theorem_3 (v u : ℕ → ℝ) (n_0 : ℕ) 
  (h_v : ∀ ε > 0, ∃ N, ∀ n ≥ N, |v n| ≤ ε) 
  (h_u : ∀ n ≥ n_0, |u n| ≤ |v n|) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n| ≤ ε :=
sorry

-- Squeeze Theorem
theorem squeeze_theorem (u v w : ℕ → ℝ) (l : ℝ) (n_0 : ℕ)
  (h_u : ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - l| ≤ ε)
  (h_w : ∀ ε > 0, ∃ N, ∀ n ≥ N, |w n - l| ≤ ε)
  (h_v : ∀ n ≥ n_0, u n ≤ v n ∧ v n ≤ w n) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |v n - l| ≤ ε :=
sorry

end NUMINAMATH_CALUDE_theorem_3_squeeze_theorem_l1907_190704


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1907_190710

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | x^2 ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1907_190710


namespace NUMINAMATH_CALUDE_vector_properties_l1907_190748

noncomputable section

def a (x : ℝ) : ℝ × ℝ := (Real.cos (3/2 * x), Real.sin (3/2 * x))
def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), Real.sin (x/2))

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def vector_sum (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def f (m x : ℝ) : ℝ := m * vector_magnitude (vector_sum (a x) (b x)) - dot_product (a x) (b x)

theorem vector_properties (m : ℝ) :
  (dot_product (a (π/4)) (b (π/4)) = Real.sqrt 2 / 2) ∧
  (vector_magnitude (vector_sum (a (π/4)) (b (π/4))) = Real.sqrt (2 + Real.sqrt 2)) ∧
  (∀ x ∈ Set.Icc 0 π, 
    (m > 2 → f m x ≤ 2*m - 3) ∧
    (0 ≤ m ∧ m ≤ 2 → f m x ≤ m^2/2 - 1) ∧
    (m < 0 → f m x ≤ -1)) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l1907_190748


namespace NUMINAMATH_CALUDE_lee_weight_l1907_190758

/-- Given Anna's and Lee's weights satisfying certain conditions, prove Lee's weight is 144 pounds. -/
theorem lee_weight (anna lee : ℝ) 
  (h1 : anna + lee = 240)
  (h2 : lee - anna = lee / 3) : 
  lee = 144 := by sorry

end NUMINAMATH_CALUDE_lee_weight_l1907_190758


namespace NUMINAMATH_CALUDE_product_equals_one_l1907_190708

theorem product_equals_one (n : ℕ) (x : ℕ → ℝ) (f : ℕ → ℝ) :
  n > 2 →
  (∀ i j, i % n = j % n → x i = x j) →
  (∀ i, f i = x i + x i * x (i + 1) + x i * x (i + 1) * x (i + 2) + 
    x i * x (i + 1) * x (i + 2) * x (i + 3) + 
    x i * x (i + 1) * x (i + 2) * x (i + 3) * x (i + 4) + 
    x i * x (i + 1) * x (i + 2) * x (i + 3) * x (i + 4) * x (i + 5) +
    x i * x (i + 1) * x (i + 2) * x (i + 3) * x (i + 4) * x (i + 5) * x (i + 6) +
    x i * x (i + 1) * x (i + 2) * x (i + 3) * x (i + 4) * x (i + 5) * x (i + 6) * x (i + 7)) →
  (∀ i j, f i = f j) →
  (∃ i j, x i ≠ x j) →
  (x 1 * x 2 * x 3 * x 4 * x 5 * x 6 * x 7 * x 8) = 1 := by
sorry

end NUMINAMATH_CALUDE_product_equals_one_l1907_190708


namespace NUMINAMATH_CALUDE_line_contains_point_l1907_190764

theorem line_contains_point (m : ℚ) : 
  (2 * m - 3 * (-1) = 5 * 3 + 1) ↔ (m = 13 / 2) := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l1907_190764


namespace NUMINAMATH_CALUDE_business_partnership_problem_l1907_190754

/-- A business partnership problem -/
theorem business_partnership_problem 
  (a_investment : ℕ) 
  (total_duration : ℕ) 
  (b_join_time : ℕ) 
  (profit_ratio_a : ℕ) 
  (profit_ratio_b : ℕ) 
  (h1 : a_investment = 3500)
  (h2 : total_duration = 12)
  (h3 : b_join_time = 8)
  (h4 : profit_ratio_a = 2)
  (h5 : profit_ratio_b = 3) : 
  ∃ (b_investment : ℕ), 
    b_investment = 1575 ∧ 
    (a_investment * total_duration) / (b_investment * (total_duration - b_join_time)) = 
    profit_ratio_a / profit_ratio_b :=
sorry

end NUMINAMATH_CALUDE_business_partnership_problem_l1907_190754


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1907_190712

theorem polynomial_division_remainder : ∃ (Q : Polynomial ℤ) (R : Polynomial ℤ),
  (X : Polynomial ℤ)^50 = (X^2 - 5*X + 6) * Q + R ∧
  (Polynomial.degree R < 2) ∧
  R = (3^50 - 2^50) * X + (2^50 - 2 * 3^50) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1907_190712


namespace NUMINAMATH_CALUDE_polynomial_equality_l1907_190706

/-- Given two polynomials p(x) = 2x^2 + 5x - 2 and q(x) = 2x^2 + 5x + 4,
    prove that the polynomial r(x) = 10x + 6 satisfies p(x) + r(x) = q(x) for all x. -/
theorem polynomial_equality (x : ℝ) :
  (2 * x^2 + 5 * x - 2) + (10 * x + 6) = 2 * x^2 + 5 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1907_190706


namespace NUMINAMATH_CALUDE_arnolds_mileage_calculation_l1907_190749

/-- Calculates the total monthly driving mileage for Arnold given his car efficiencies and gas spending --/
def arnolds_mileage (efficiency1 efficiency2 efficiency3 : ℚ) (gas_price : ℚ) (monthly_spend : ℚ) : ℚ :=
  let total_cars := 3
  let inverse_efficiency := (1 / efficiency1 + 1 / efficiency2 + 1 / efficiency3) / total_cars
  monthly_spend / (gas_price * inverse_efficiency)

/-- Theorem stating that Arnold's total monthly driving mileage is (56 * 450) / 43 miles --/
theorem arnolds_mileage_calculation :
  let efficiency1 := 50
  let efficiency2 := 10
  let efficiency3 := 15
  let gas_price := 2
  let monthly_spend := 56
  arnolds_mileage efficiency1 efficiency2 efficiency3 gas_price monthly_spend = 56 * 450 / 43 := by
  sorry

end NUMINAMATH_CALUDE_arnolds_mileage_calculation_l1907_190749


namespace NUMINAMATH_CALUDE_bakers_cakes_l1907_190716

/-- Baker's cake selling problem -/
theorem bakers_cakes (initial_cakes bought_cakes sold_difference : ℕ) 
  (h1 : initial_cakes = 8)
  (h2 : bought_cakes = 139)
  (h3 : sold_difference = 6) :
  bought_cakes + sold_difference = 145 := by
  sorry

#check bakers_cakes

end NUMINAMATH_CALUDE_bakers_cakes_l1907_190716


namespace NUMINAMATH_CALUDE_sequence_equality_l1907_190746

theorem sequence_equality (n : ℕ+) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l1907_190746


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1907_190711

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality : 
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1907_190711


namespace NUMINAMATH_CALUDE_distance_between_points_l1907_190777

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 17)
  let p2 : ℝ × ℝ := (10, 3)
  let distance := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  distance = 2 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1907_190777


namespace NUMINAMATH_CALUDE_jason_total_games_l1907_190780

/-- The number of football games Jason attended or plans to attend each month from January to July -/
def games_per_month : List Nat := [11, 17, 16, 20, 14, 14, 14]

/-- The total number of games Jason will have attended by the end of July -/
def total_games : Nat := games_per_month.sum

theorem jason_total_games : total_games = 106 := by
  sorry

end NUMINAMATH_CALUDE_jason_total_games_l1907_190780


namespace NUMINAMATH_CALUDE_count_with_zero_eq_952_l1907_190760

/-- A function that checks if a positive integer contains the digit 0 in its base-ten representation -/
def containsZero (n : ℕ+) : Bool :=
  sorry

/-- The count of positive integers less than or equal to 2500 that contain the digit 0 -/
def countWithZero : ℕ :=
  sorry

/-- Theorem stating that the count of positive integers less than or equal to 2500 
    containing the digit 0 is 952 -/
theorem count_with_zero_eq_952 : countWithZero = 952 := by
  sorry

end NUMINAMATH_CALUDE_count_with_zero_eq_952_l1907_190760


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l1907_190742

/-- The cost of an adult ticket to the circus -/
def adult_ticket_cost : ℕ := 2

/-- The number of people in Mary's group -/
def total_people : ℕ := 4

/-- The number of children in Mary's group -/
def num_children : ℕ := 3

/-- The cost of a child's ticket -/
def child_ticket_cost : ℕ := 1

/-- The total amount Mary paid -/
def total_paid : ℕ := 5

theorem circus_ticket_cost :
  adult_ticket_cost = total_paid - (num_children * child_ticket_cost) :=
by sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l1907_190742


namespace NUMINAMATH_CALUDE_sevenPointFourSix_eq_fraction_l1907_190715

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℕ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + (x.repeatingPart : ℚ) / (99 : ℚ)

/-- The repeating decimal 7.464646... -/
def sevenPointFourSix : RepeatingDecimal :=
  { integerPart := 7, repeatingPart := 46 }

theorem sevenPointFourSix_eq_fraction :
  toRational sevenPointFourSix = 739 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sevenPointFourSix_eq_fraction_l1907_190715


namespace NUMINAMATH_CALUDE_sequence_properties_l1907_190794

/-- Sequence a_n is a first-degree function of n -/
def a (n : ℕ) : ℝ := 2 * n + 1

/-- Sequence b_n is composed of a_2, a_4, a_6, a_8, ... -/
def b (n : ℕ) : ℝ := a (2 * n)

theorem sequence_properties :
  (a 1 = 3) ∧ 
  (a 10 = 21) ∧ 
  (∀ n : ℕ, a_2009 = 4019) ∧
  (∀ n : ℕ, b n = 4 * n + 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1907_190794


namespace NUMINAMATH_CALUDE_alpha_value_l1907_190729

theorem alpha_value (α : Real) (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : ∀ x, (Real.sin α) ^ (x^2 - 2*x + 3) ≤ 1/4) : α = 5*π/6 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l1907_190729


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l1907_190739

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l1907_190739


namespace NUMINAMATH_CALUDE_marys_bedrooms_l1907_190721

/-- Represents the number of rooms in a house -/
structure House where
  bedrooms : ℕ
  kitchen : Unit
  livingRoom : Unit

/-- Represents a vacuum cleaner -/
structure VacuumCleaner where
  batteryLife : ℕ  -- in minutes
  chargingTimes : ℕ

/-- Represents the time it takes to vacuum a room -/
def roomVacuumTime : ℕ := 4

theorem marys_bedrooms (h : House) (v : VacuumCleaner)
    (hv : v.batteryLife = 10 ∧ v.chargingTimes = 2) :
    h.bedrooms = 5 := by
  sorry

end NUMINAMATH_CALUDE_marys_bedrooms_l1907_190721


namespace NUMINAMATH_CALUDE_least_time_six_horses_meet_l1907_190707

def horse_lap_time (k : ℕ) : ℕ := k + 1

def is_at_start (t : ℕ) (k : ℕ) : Prop :=
  t % (horse_lap_time k) = 0

def at_least_six_at_start (t : ℕ) : Prop :=
  ∃ (s : Finset ℕ), s.card ≥ 6 ∧ s ⊆ Finset.range 8 ∧ ∀ k ∈ s, is_at_start t k

theorem least_time_six_horses_meet :
  ∃ (T : ℕ), T > 0 ∧ at_least_six_at_start T ∧
  ∀ (t : ℕ), t > 0 ∧ t < T → ¬(at_least_six_at_start t) ∧
  T = 420 :=
sorry

end NUMINAMATH_CALUDE_least_time_six_horses_meet_l1907_190707


namespace NUMINAMATH_CALUDE_basketball_win_rate_l1907_190736

theorem basketball_win_rate (games_won_first_half : ℕ) (total_games : ℕ) (desired_win_rate : ℚ) (games_to_win : ℕ) : 
  games_won_first_half = 30 →
  total_games = 80 →
  desired_win_rate = 3/4 →
  games_to_win = 30 →
  (games_won_first_half + games_to_win : ℚ) / total_games = desired_win_rate :=
by
  sorry

#check basketball_win_rate

end NUMINAMATH_CALUDE_basketball_win_rate_l1907_190736


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l1907_190766

open Set Real

theorem negation_of_existence_inequality :
  (¬ ∃ x : ℝ, x^2 - 5*x - 6 < 0) ↔ (∀ x : ℝ, x^2 - 5*x - 6 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l1907_190766


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l1907_190776

theorem consecutive_even_numbers_sum (x : ℤ) : 
  (x % 2 = 0) →  -- x is even
  (x + (x + 2) + (x + 4) = x + 18) →  -- sum condition
  (x + 4 = 10)  -- largest number is 10
  := by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l1907_190776


namespace NUMINAMATH_CALUDE_value_of_T_l1907_190774

theorem value_of_T : ∃ T : ℚ, (1/2 : ℚ) * (1/7 : ℚ) * T = (1/3 : ℚ) * (1/5 : ℚ) * 60 ∧ T = 56 := by
  sorry

end NUMINAMATH_CALUDE_value_of_T_l1907_190774


namespace NUMINAMATH_CALUDE_cranberry_juice_ounces_l1907_190771

/-- Given a can of cranberry juice that sells for 84 cents with a cost of 7 cents per ounce,
    prove that the can contains 12 ounces of juice. -/
theorem cranberry_juice_ounces (total_cost : ℕ) (cost_per_ounce : ℕ) (h1 : total_cost = 84) (h2 : cost_per_ounce = 7) :
  total_cost / cost_per_ounce = 12 := by
sorry

end NUMINAMATH_CALUDE_cranberry_juice_ounces_l1907_190771


namespace NUMINAMATH_CALUDE_negative_root_implies_a_less_than_negative_three_l1907_190703

theorem negative_root_implies_a_less_than_negative_three (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (a - 3)) → a < -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_root_implies_a_less_than_negative_three_l1907_190703


namespace NUMINAMATH_CALUDE_exists_negative_fraction_abs_lt_four_l1907_190799

theorem exists_negative_fraction_abs_lt_four : ∃ (a b : ℤ), b ≠ 0 ∧ (a / b : ℚ) < 0 ∧ |a / b| < 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_negative_fraction_abs_lt_four_l1907_190799


namespace NUMINAMATH_CALUDE_average_first_twelve_even_numbers_l1907_190781

-- Define the first 12 even numbers
def firstTwelveEvenNumbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

-- Theorem to prove
theorem average_first_twelve_even_numbers :
  (List.sum firstTwelveEvenNumbers) / (List.length firstTwelveEvenNumbers) = 13 := by
  sorry


end NUMINAMATH_CALUDE_average_first_twelve_even_numbers_l1907_190781


namespace NUMINAMATH_CALUDE_slope_of_line_l1907_190798

theorem slope_of_line (x y : ℝ) (h : x/3 + y/2 = 1) : 
  ∃ m b : ℝ, y = m*x + b ∧ m = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l1907_190798


namespace NUMINAMATH_CALUDE_alpha_value_l1907_190743

theorem alpha_value (α : ℝ) :
  (6 * Real.sqrt 3) / (3 * Real.sqrt 2 + 2 * Real.sqrt 3) = 3 * Real.sqrt α - 6 →
  α = 6 :=
by sorry

end NUMINAMATH_CALUDE_alpha_value_l1907_190743


namespace NUMINAMATH_CALUDE_earliest_meet_time_proof_l1907_190720

def charlie_lap_time : ℕ := 5
def alex_lap_time : ℕ := 8
def taylor_lap_time : ℕ := 10

def earliest_meet_time : ℕ := 40

theorem earliest_meet_time_proof :
  lcm (lcm charlie_lap_time alex_lap_time) taylor_lap_time = earliest_meet_time :=
by sorry

end NUMINAMATH_CALUDE_earliest_meet_time_proof_l1907_190720


namespace NUMINAMATH_CALUDE_round_trip_time_l1907_190735

/-- Calculates the total time for a round trip between two towns given the speeds and initial travel time. -/
theorem round_trip_time (speed_to_b : ℝ) (speed_to_a : ℝ) (time_to_b : ℝ) : 
  speed_to_b > 0 → speed_to_a > 0 → time_to_b > 0 →
  speed_to_b = 100 → speed_to_a = 150 → time_to_b = 3 →
  time_to_b + (speed_to_b * time_to_b) / speed_to_a = 5 := by
  sorry

#check round_trip_time

end NUMINAMATH_CALUDE_round_trip_time_l1907_190735


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l1907_190785

theorem complex_modulus_theorem : 
  let i : ℂ := Complex.I
  let T : ℂ := (1 + i)^19 - (1 - i)^19
  Complex.abs T = 512 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l1907_190785


namespace NUMINAMATH_CALUDE_combination_equality_l1907_190728

theorem combination_equality (x : ℕ) : 
  (Nat.choose 10 x = Nat.choose 10 (3*x - 2)) → (x = 1 ∨ x = 3) := by
sorry

end NUMINAMATH_CALUDE_combination_equality_l1907_190728


namespace NUMINAMATH_CALUDE_base12_addition_l1907_190750

/-- Represents a digit in base 12 --/
inductive Digit12 : Type
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

/-- Represents a number in base 12 as a list of digits --/
def Base12 := List Digit12

/-- Convert a base 12 number to its decimal representation --/
def toDecimal (n : Base12) : Nat := sorry

/-- Convert a decimal number to its base 12 representation --/
def fromDecimal (n : Nat) : Base12 := sorry

/-- Addition operation for base 12 numbers --/
def addBase12 (a b : Base12) : Base12 := sorry

/-- The main theorem --/
theorem base12_addition :
  let n1 : Base12 := [Digit12.D5, Digit12.A, Digit12.D3]
  let n2 : Base12 := [Digit12.D2, Digit12.B, Digit12.D8]
  addBase12 n1 n2 = [Digit12.D8, Digit12.D9, Digit12.D6] := by sorry

end NUMINAMATH_CALUDE_base12_addition_l1907_190750


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1907_190732

theorem arithmetic_mean_problem (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 84 → a = 32 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1907_190732


namespace NUMINAMATH_CALUDE_magazine_selection_count_l1907_190756

def total_magazines : ℕ := 8
def literature_magazines : ℕ := 3
def math_magazines : ℕ := 5
def magazines_to_select : ℕ := 3

theorem magazine_selection_count :
  (Nat.choose math_magazines magazines_to_select) +
  (Nat.choose math_magazines (magazines_to_select - 1)) +
  (Nat.choose math_magazines (magazines_to_select - 2)) +
  (if literature_magazines ≥ magazines_to_select then 1 else 0) = 26 := by
  sorry

end NUMINAMATH_CALUDE_magazine_selection_count_l1907_190756


namespace NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l1907_190769

-- Define the ellipse properties
def ellipse_axis_sum : ℝ := 18
def ellipse_focal_length : ℝ := 6

-- Define the reference ellipse for the hyperbola
def reference_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the point Q
def point_Q : ℝ × ℝ := (2, 1)

-- Theorem for the ellipse equation
theorem ellipse_equation (x y : ℝ) :
  (x^2 / 25 + y^2 / 16 = 1) ∨ (x^2 / 16 + y^2 / 25 = 1) :=
sorry

-- Theorem for the hyperbola equation
theorem hyperbola_equation (x y : ℝ) :
  x^2 / 2 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l1907_190769


namespace NUMINAMATH_CALUDE_problem_solution_l1907_190740

/-- Represents the color of a ball -/
inductive Color
| White
| Black

/-- Represents a pair of drawn balls -/
structure DrawnBalls :=
  (first second : Color)

/-- The sample space of all possible outcomes when drawing two balls without replacement -/
def Ω : Finset DrawnBalls := sorry

/-- Event A: drawing two balls of the same color -/
def A : Set DrawnBalls := {db | db.first = db.second}

/-- Event B: the first ball drawn is white -/
def B : Set DrawnBalls := {db | db.first = Color.White}

/-- Event C: the second ball drawn is white -/
def C : Set DrawnBalls := {db | db.second = Color.White}

/-- Event D: drawing two balls of different colors -/
def D : Set DrawnBalls := {db | db.first ≠ db.second}

/-- The probability measure on the sample space -/
noncomputable def P : Set DrawnBalls → ℝ := sorry

theorem problem_solution :
  (P B = 1/2) ∧
  (P (A ∩ B) = P A * P B) ∧
  (A ∪ D = Set.univ) ∧ (A ∩ D = ∅) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1907_190740


namespace NUMINAMATH_CALUDE_smallest_a_value_l1907_190763

def rectangle_vertices : List (ℝ × ℝ) := [(34, 0), (41, 0), (34, 9), (41, 9)]

def line_equation (a : ℝ) (x : ℝ) : ℝ := a * x

def divides_rectangle (a : ℝ) : Prop :=
  ∃ (area1 area2 : ℝ), area1 = 2 * area2 ∧
  area1 + area2 = 63 ∧
  (∃ (x1 y1 x2 y2 : ℝ),
    ((x1, y1) ∈ rectangle_vertices ∨ (x1 ∈ Set.Icc 34 41 ∧ y1 = line_equation a x1)) ∧
    ((x2, y2) ∈ rectangle_vertices ∨ (x2 ∈ Set.Icc 34 41 ∧ y2 = line_equation a x2)) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2))

theorem smallest_a_value :
  ∀ ε > 0, divides_rectangle (0.08 + ε) → divides_rectangle 0.08 ∧ ¬divides_rectangle (0.08 - ε) := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_value_l1907_190763


namespace NUMINAMATH_CALUDE_exists_vertex_with_positive_product_l1907_190702

-- Define a polyhedron type
structure Polyhedron where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  marks : (Nat × Nat) → Int
  vertex_count : vertices.card = 101
  edge_marks : ∀ e ∈ edges, marks e = 1 ∨ marks e = -1

-- Define the product of marks at a vertex
def product_at_vertex (p : Polyhedron) (v : Nat) : Int :=
  (p.edges.filter (λ e => e.1 = v ∨ e.2 = v)).prod p.marks

-- Theorem statement
theorem exists_vertex_with_positive_product (p : Polyhedron) :
  ∃ v ∈ p.vertices, product_at_vertex p v = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_vertex_with_positive_product_l1907_190702


namespace NUMINAMATH_CALUDE_subtract_problem_l1907_190783

theorem subtract_problem (x : ℤ) : x - 46 = 15 → x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_subtract_problem_l1907_190783


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l1907_190761

theorem product_zero_implies_factor_zero (a b : ℝ) : a * b = 0 → a = 0 ∨ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l1907_190761


namespace NUMINAMATH_CALUDE_goals_theorem_l1907_190773

/-- The total number of goals scored in the league against Barca -/
def total_goals : ℕ := 300

/-- The number of players who scored goals -/
def num_players : ℕ := 2

/-- The number of goals scored by each player -/
def goals_per_player : ℕ := 30

/-- The percentage of total goals scored by the two players -/
def percentage : ℚ := 1/5

theorem goals_theorem (h1 : num_players * goals_per_player = (percentage * total_goals).num) :
  total_goals = 300 := by
  sorry

end NUMINAMATH_CALUDE_goals_theorem_l1907_190773


namespace NUMINAMATH_CALUDE_pie_chart_angle_l1907_190762

theorem pie_chart_angle (percentage : ℝ) (angle : ℝ) :
  percentage = 0.15 →
  angle = percentage * 360 →
  angle = 54 := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_angle_l1907_190762


namespace NUMINAMATH_CALUDE_fifth_row_solution_l1907_190775

/-- Represents the possible values in the grid -/
inductive GridValue
  | Two
  | Zero
  | One
  | Five
  | Blank

/-- Represents a 5x5 grid -/
def Grid := Fin 5 → Fin 5 → GridValue

/-- Check if a grid satisfies the row constraint -/
def satisfiesRowConstraint (g : Grid) : Prop :=
  ∀ row, ∃! i, g row i = GridValue.Two ∧
         ∃! i, g row i = GridValue.Zero ∧
         ∃! i, g row i = GridValue.One ∧
         ∃! i, g row i = GridValue.Five

/-- Check if a grid satisfies the column constraint -/
def satisfiesColumnConstraint (g : Grid) : Prop :=
  ∀ col, ∃! i, g i col = GridValue.Two ∧
         ∃! i, g i col = GridValue.Zero ∧
         ∃! i, g i col = GridValue.One ∧
         ∃! i, g i col = GridValue.Five

/-- Check if a grid satisfies the diagonal constraint -/
def satisfiesDiagonalConstraint (g : Grid) : Prop :=
  ∀ i j, i < 4 → j < 4 →
    (g i j ≠ GridValue.Blank → g (i+1) (j+1) ≠ g i j) ∧
    (g i (j+1) ≠ GridValue.Blank → g (i+1) j ≠ g i (j+1))

/-- The main theorem stating the solution for the fifth row -/
theorem fifth_row_solution (g : Grid) 
  (hrow : satisfiesRowConstraint g)
  (hcol : satisfiesColumnConstraint g)
  (hdiag : satisfiesDiagonalConstraint g) :
  g 4 0 = GridValue.One ∧
  g 4 1 = GridValue.Five ∧
  g 4 2 = GridValue.Blank ∧
  g 4 3 = GridValue.Blank ∧
  g 4 4 = GridValue.Two :=
sorry

end NUMINAMATH_CALUDE_fifth_row_solution_l1907_190775


namespace NUMINAMATH_CALUDE_minimum_box_cost_greenville_box_cost_l1907_190779

/-- The minimum amount spent on boxes for packaging a fine arts collection -/
theorem minimum_box_cost (box_length box_width box_height : ℝ) 
  (box_cost : ℝ) (collection_volume : ℝ) : ℝ :=
  let box_volume := box_length * box_width * box_height
  let num_boxes := collection_volume / box_volume
  num_boxes * box_cost

/-- The specific case for Greenville State University -/
theorem greenville_box_cost : 
  minimum_box_cost 20 20 12 0.40 2400000 = 200 := by
  sorry

end NUMINAMATH_CALUDE_minimum_box_cost_greenville_box_cost_l1907_190779


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l1907_190744

-- Define the function f
def f (x : ℤ) : ℤ := 3 * x + 2

-- Define the k-fold composition of f
def f_comp (k : ℕ) : ℤ → ℤ :=
  match k with
  | 0 => id
  | n + 1 => f ∘ (f_comp n)

-- State the theorem
theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ+, (1988 : ℤ) ∣ (f_comp 100 m.val) :=
sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l1907_190744


namespace NUMINAMATH_CALUDE_triangle_toothpick_count_l1907_190701

/-- The number of small equilateral triangles in the base row -/
def base_triangles : ℕ := 10

/-- The number of additional toothpicks in the isosceles row compared to the last equilateral row -/
def extra_isosceles_toothpicks : ℕ := 9

/-- The total number of small equilateral triangles in the main part of the large triangle -/
def total_equilateral_triangles : ℕ := (base_triangles * (base_triangles + 1)) / 2

/-- The number of toothpicks needed for the described triangle construction -/
def total_toothpicks : ℕ := 
  let equilateral_toothpicks := (3 * total_equilateral_triangles + 1) / 2
  let boundary_toothpicks := 2 * base_triangles + extra_isosceles_toothpicks
  equilateral_toothpicks + extra_isosceles_toothpicks + boundary_toothpicks - base_triangles

theorem triangle_toothpick_count : total_toothpicks = 110 := by sorry

end NUMINAMATH_CALUDE_triangle_toothpick_count_l1907_190701


namespace NUMINAMATH_CALUDE_expression_evaluation_l1907_190789

theorem expression_evaluation :
  let a : ℚ := -1/3
  let expr := (3 - a) / (2*a - 4) / (a + 2 - 5/(a - 2))
  expr = 3/16 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1907_190789


namespace NUMINAMATH_CALUDE_club_M_members_eq_five_l1907_190724

/-- The number of people who joined club M in a company with the following conditions:
  - There are 60 people in total
  - There are 3 clubs: M, S, and Z
  - 18 people joined S
  - 11 people joined Z
  - Members of M did not join any other club
  - At most 26 people did not join any club
-/
def club_M_members : ℕ := by
  -- Define the total number of people
  let total_people : ℕ := 60
  -- Define the number of people in club S
  let club_S_members : ℕ := 18
  -- Define the number of people in club Z
  let club_Z_members : ℕ := 11
  -- Define the maximum number of people who didn't join any club
  let max_no_club : ℕ := 26
  
  -- The actual proof would go here
  sorry

theorem club_M_members_eq_five : club_M_members = 5 := by
  sorry

end NUMINAMATH_CALUDE_club_M_members_eq_five_l1907_190724


namespace NUMINAMATH_CALUDE_unique_integer_prime_expressions_l1907_190793

theorem unique_integer_prime_expressions : ∃! n : ℤ, 
  Nat.Prime (Int.natAbs (n^3 - 4*n^2 + 3*n - 35)) ∧ 
  Nat.Prime (Int.natAbs (n^2 + 4*n + 8)) ∧ 
  n = 5 := by sorry

end NUMINAMATH_CALUDE_unique_integer_prime_expressions_l1907_190793


namespace NUMINAMATH_CALUDE_bisecting_line_min_value_bisecting_line_min_value_achievable_l1907_190741

/-- A line that bisects the circumference of a circle --/
structure BisectingLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  bisects : ∀ (x y : ℝ), a * x + 2 * b * y - 2 = 0 → 
    (x^2 + y^2 - 4*x - 2*y - 8 = 0 → 
      ∃ (d : ℝ), d > 0 ∧ (x - 2)^2 + (y - 1)^2 = d^2)

/-- The theorem stating the minimum value of 1/a + 2/b --/
theorem bisecting_line_min_value (l : BisectingLine) :
  (1 / l.a + 2 / l.b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

/-- The theorem stating that the minimum value is achievable --/
theorem bisecting_line_min_value_achievable :
  ∃ (l : BisectingLine), 1 / l.a + 2 / l.b = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_min_value_bisecting_line_min_value_achievable_l1907_190741


namespace NUMINAMATH_CALUDE_center_is_five_l1907_190778

/-- Represents a 3x3 array of integers -/
def Array3x3 := Fin 3 → Fin 3 → ℕ

/-- Checks if two positions in the array are adjacent -/
def is_adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Checks if the array contains all numbers from 1 to 9 -/
def contains_all_numbers (a : Array3x3) : Prop :=
  ∀ n : Fin 9, ∃ i j : Fin 3, a i j = n + 1

/-- Checks if consecutive numbers are adjacent in the array -/
def consecutive_adjacent (a : Array3x3) : Prop :=
  ∀ n : Fin 8, ∃ i₁ j₁ i₂ j₂ : Fin 3,
    a i₁ j₁ = n + 1 ∧ a i₂ j₂ = n + 2 ∧ is_adjacent (i₁, j₁) (i₂, j₂)

/-- The sum of corner numbers is 20 -/
def corner_sum_20 (a : Array3x3) : Prop :=
  a 0 0 + a 0 2 + a 2 0 + a 2 2 = 20

/-- The product of top-left and bottom-right corner numbers is 9 -/
def corner_product_9 (a : Array3x3) : Prop :=
  a 0 0 * a 2 2 = 9

theorem center_is_five (a : Array3x3)
  (h1 : contains_all_numbers a)
  (h2 : consecutive_adjacent a)
  (h3 : corner_sum_20 a)
  (h4 : corner_product_9 a) :
  a 1 1 = 5 :=
sorry

end NUMINAMATH_CALUDE_center_is_five_l1907_190778


namespace NUMINAMATH_CALUDE_power_minus_ten_over_nine_equals_ten_l1907_190797

theorem power_minus_ten_over_nine_equals_ten : (10^2 - 10) / 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_minus_ten_over_nine_equals_ten_l1907_190797


namespace NUMINAMATH_CALUDE_number_puzzle_l1907_190791

theorem number_puzzle (x : ℚ) : (x / 4) * 12 = 9 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1907_190791


namespace NUMINAMATH_CALUDE_spinner_final_direction_l1907_190782

-- Define the possible directions
inductive Direction
| North
| East
| South
| West

-- Define a function to calculate the final direction
def finalDirection (initialDir : Direction) (clockwiseRev : ℚ) (counterClockwiseRev : ℚ) : Direction :=
  sorry

-- Theorem statement
theorem spinner_final_direction :
  finalDirection Direction.North (7/2) (21/4) = Direction.East :=
sorry

end NUMINAMATH_CALUDE_spinner_final_direction_l1907_190782
