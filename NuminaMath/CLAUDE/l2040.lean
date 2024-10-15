import Mathlib

namespace NUMINAMATH_CALUDE_x_less_than_neg_two_sufficient_not_necessary_for_x_leq_zero_l2040_204045

theorem x_less_than_neg_two_sufficient_not_necessary_for_x_leq_zero :
  (∀ x : ℝ, x < -2 → x ≤ 0) ∧
  (∃ x : ℝ, x ≤ 0 ∧ x ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_neg_two_sufficient_not_necessary_for_x_leq_zero_l2040_204045


namespace NUMINAMATH_CALUDE_charles_housesitting_hours_l2040_204006

/-- Proves that Charles housesat for 10 hours given the conditions of the problem -/
theorem charles_housesitting_hours : 
  let housesitting_rate : ℝ := 15
  let dog_walking_rate : ℝ := 22
  let num_dogs : ℕ := 3
  let total_earnings : ℝ := 216
  ∃ (h : ℝ), h * housesitting_rate + (num_dogs : ℝ) * dog_walking_rate = total_earnings ∧ h = 10 := by
  sorry

end NUMINAMATH_CALUDE_charles_housesitting_hours_l2040_204006


namespace NUMINAMATH_CALUDE_hemisphere_volume_l2040_204068

/-- The volume of a hemisphere with diameter 8 cm is (128/3)π cubic centimeters. -/
theorem hemisphere_volume (π : ℝ) (hemisphere_diameter : ℝ) (hemisphere_volume : ℝ → ℝ → ℝ) :
  hemisphere_diameter = 8 →
  hemisphere_volume π hemisphere_diameter = (128 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_volume_l2040_204068


namespace NUMINAMATH_CALUDE_coprime_condition_l2040_204083

theorem coprime_condition (a b c d : ℤ) (h : Nat.gcd a.natAbs b.natAbs = 1 ∧ 
  Nat.gcd c.natAbs d.natAbs = 1 ∧ Nat.gcd a.natAbs c.natAbs = 1) : 
  (∀ (p : ℕ), Nat.Prime p → p ∣ (a * d - b * c).natAbs → (p ∣ a.natAbs ∨ p ∣ c.natAbs)) ↔ 
  (∀ (n : ℤ), Nat.gcd (a * n + b).natAbs (c * n + d).natAbs = 1) := by
sorry

end NUMINAMATH_CALUDE_coprime_condition_l2040_204083


namespace NUMINAMATH_CALUDE_claras_weight_l2040_204043

theorem claras_weight (alice_weight clara_weight : ℝ) 
  (h1 : alice_weight + clara_weight = 240)
  (h2 : clara_weight - alice_weight = (2/3) * clara_weight) : 
  clara_weight = 180 := by
sorry

end NUMINAMATH_CALUDE_claras_weight_l2040_204043


namespace NUMINAMATH_CALUDE_zacks_marbles_l2040_204090

theorem zacks_marbles (initial_marbles : ℕ) (kept_marbles : ℕ) (num_friends : ℕ) : 
  initial_marbles = 65 → 
  kept_marbles = 5 → 
  num_friends = 3 → 
  (initial_marbles - kept_marbles) / num_friends = 20 := by
sorry

end NUMINAMATH_CALUDE_zacks_marbles_l2040_204090


namespace NUMINAMATH_CALUDE_divisor_of_sum_l2040_204048

theorem divisor_of_sum (n : ℕ) (a : ℕ) (d : ℕ) : 
  n = 425897 → a = 7 → d = 7 → (n + a) % d = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_sum_l2040_204048


namespace NUMINAMATH_CALUDE_f_equals_g_l2040_204098

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l2040_204098


namespace NUMINAMATH_CALUDE_middle_circle_radius_l2040_204002

/-- Configuration of five circles tangent to each other and two parallel lines -/
structure CircleConfiguration where
  /-- Radius of the smallest circle -/
  r_min : ℝ
  /-- Radius of the largest circle -/
  r_max : ℝ
  /-- Radius of the middle circle -/
  r_mid : ℝ

/-- The theorem stating the relationship between the radii of the circles -/
theorem middle_circle_radius (c : CircleConfiguration) 
  (h_min : c.r_min = 12)
  (h_max : c.r_max = 24) :
  c.r_mid = 12 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_middle_circle_radius_l2040_204002


namespace NUMINAMATH_CALUDE_bill_caroline_age_difference_l2040_204024

theorem bill_caroline_age_difference (bill_age caroline_age : ℕ) : 
  bill_age + caroline_age = 26 →
  bill_age = 17 →
  ∃ x : ℕ, bill_age = 2 * caroline_age - x →
  2 * caroline_age - bill_age = 1 := by
sorry

end NUMINAMATH_CALUDE_bill_caroline_age_difference_l2040_204024


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2040_204087

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 2) - x^2

theorem function_inequality_implies_a_bound :
  ∀ a : ℝ,
  (∀ p q : ℝ, 0 < q ∧ q < p ∧ p < 1 →
    (f a (p + 1) - f a (q + 1)) / (p - q) > 2) →
  a ≥ 24 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2040_204087


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l2040_204017

theorem gcd_of_squares_sum : Nat.gcd 
  (122^2 + 234^2 + 346^2 + 458^2) 
  (121^2 + 233^2 + 345^2 + 457^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l2040_204017


namespace NUMINAMATH_CALUDE_min_value_theorem_l2040_204061

theorem min_value_theorem (α₁ α₂ : ℝ) 
  (h : (2 + Real.sin α₁)⁻¹ + (2 + Real.sin (2 * α₂))⁻¹ = 2) : 
  ∃ (k₁ k₂ : ℤ), ∀ (α₁' α₂' : ℝ), 
    (2 + Real.sin α₁')⁻¹ + (2 + Real.sin (2 * α₂'))⁻¹ = 2 →
    |10 * Real.pi - α₁' - α₂'| ≥ |10 * Real.pi - ((-π/2 : ℝ) + 2 * ↑k₁ * π) - ((-π/4 : ℝ) + ↑k₂ * π)| ∧
    |10 * Real.pi - ((-π/2 : ℝ) + 2 * ↑k₁ * π) - ((-π/4 : ℝ) + ↑k₂ * π)| = π/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2040_204061


namespace NUMINAMATH_CALUDE_fourth_root_equation_l2040_204074

theorem fourth_root_equation (y : ℝ) :
  (y * (y^5)^(1/2))^(1/4) = 4 → y = 2^(16/7) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_l2040_204074


namespace NUMINAMATH_CALUDE_base8_to_base10_547_l2040_204010

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base-8 representation of the number --/
def base8Number : List Nat := [7, 4, 5]

theorem base8_to_base10_547 :
  base8ToBase10 base8Number = 359 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base10_547_l2040_204010


namespace NUMINAMATH_CALUDE_hannahs_brothers_l2040_204022

theorem hannahs_brothers (num_brothers : ℕ) : num_brothers = 3 :=
  by
  -- Hannah has some brothers
  have h1 : num_brothers > 0 := by sorry
  
  -- All her brothers are 8 years old
  let brother_age := 8
  
  -- Hannah is 48 years old
  let hannah_age := 48
  
  -- Hannah's age is twice the sum of her brothers' ages
  have h2 : hannah_age = 2 * (num_brothers * brother_age) := by sorry
  
  -- Proof that num_brothers = 3
  sorry

end NUMINAMATH_CALUDE_hannahs_brothers_l2040_204022


namespace NUMINAMATH_CALUDE_base8_237_equals_base10_159_l2040_204039

/-- Converts a three-digit number from base 8 to base 10 -/
def base8ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The base 8 number 237 is equal to 159 in base 10 -/
theorem base8_237_equals_base10_159 : base8ToBase10 2 3 7 = 159 := by
  sorry

end NUMINAMATH_CALUDE_base8_237_equals_base10_159_l2040_204039


namespace NUMINAMATH_CALUDE_train_speed_l2040_204015

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 135 →
  bridge_length = 240.03 →
  crossing_time = 30 →
  (((train_length + bridge_length) / crossing_time) * 3.6) = 45.0036 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2040_204015


namespace NUMINAMATH_CALUDE_circle_rotation_invariance_l2040_204071

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a rotation
def rotate (θ : ℝ) (O : ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem circle_rotation_invariance (S : Circle) (θ : ℝ) (O : ℝ × ℝ) :
  ∃ (S' : Circle), S'.radius = S.radius ∧
    (∀ (p : ℝ × ℝ), (p.1 - S.center.1)^2 + (p.2 - S.center.2)^2 = S.radius^2 →
      let p' := rotate θ O p
      (p'.1 - S'.center.1)^2 + (p'.2 - S'.center.2)^2 = S'.radius^2) :=
sorry

end NUMINAMATH_CALUDE_circle_rotation_invariance_l2040_204071


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_l2040_204035

theorem consecutive_odd_numbers (n : ℕ) 
  (h_avg : (27 + 27 - 2 * (n - 1)) / 2 = 24) 
  (h_largest : 27 = 27 - 2 * (n - 1) + 2 * (n - 1)) : n = 4 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_l2040_204035


namespace NUMINAMATH_CALUDE_remaining_money_l2040_204044

/-- Calculates the remaining money after purchasing bread and peanut butter -/
theorem remaining_money 
  (bread_cost : ℝ) 
  (peanut_butter_cost : ℝ) 
  (initial_money : ℝ) 
  (num_loaves : ℕ) : 
  bread_cost = 2.25 →
  peanut_butter_cost = 2 →
  initial_money = 14 →
  num_loaves = 3 →
  initial_money - (num_loaves * bread_cost + peanut_butter_cost) = 5.25 := by
sorry

end NUMINAMATH_CALUDE_remaining_money_l2040_204044


namespace NUMINAMATH_CALUDE_train_speed_with_stoppages_train_problem_l2040_204016

/-- Calculates the speed of a train including stoppages -/
theorem train_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (total_time : ℝ) :
  speed_without_stoppages * (total_time - stoppage_time) / total_time = 
  speed_without_stoppages * (1 - stoppage_time / total_time) := by
  sorry

/-- The speed of a train including stoppages, given its speed without stoppages and stoppage time -/
theorem train_problem 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) :
  speed_without_stoppages = 45 →
  stoppage_time = 1/3 →
  speed_without_stoppages * (1 - stoppage_time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_with_stoppages_train_problem_l2040_204016


namespace NUMINAMATH_CALUDE_capital_growth_l2040_204004

def capital_sequence : ℕ → ℝ
  | 0 => 60
  | n + 1 => 1.5 * capital_sequence n - 15

theorem capital_growth (n : ℕ) :
  -- a₁ = 60
  capital_sequence 0 = 60 ∧
  -- {aₙ - 3} forms a geometric sequence
  (∀ k : ℕ, capital_sequence (k + 1) - 3 = 1.5 * (capital_sequence k - 3)) ∧
  -- By the end of 2026 (6 years from 2021), the remaining capital will exceed 210 million yuan
  ∃ m : ℕ, m ≤ 6 ∧ capital_sequence m > 210 :=
by sorry

end NUMINAMATH_CALUDE_capital_growth_l2040_204004


namespace NUMINAMATH_CALUDE_angle_bisector_inequalities_l2040_204080

/-- Given a triangle with side lengths a, b, and c, and semiperimeter p,
    prove properties about the lengths of its angle bisectors. -/
theorem angle_bisector_inequalities
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (p : ℝ) (hp : p = (a + b + c) / 2)
  (l_a l_b l_c : ℝ)
  (hl_a : l_a^2 ≤ p * (p - a))
  (hl_b : l_b^2 ≤ p * (p - b))
  (hl_c : l_c^2 ≤ p * (p - c)) :
  (l_a^2 + l_b^2 + l_c^2 ≤ p^2) ∧
  (l_a + l_b + l_c ≤ Real.sqrt 3 * p) := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_inequalities_l2040_204080


namespace NUMINAMATH_CALUDE_two_not_units_digit_of_square_l2040_204053

def units_digit (n : ℕ) : ℕ := n % 10

theorem two_not_units_digit_of_square : ∀ n : ℕ, units_digit (n^2) ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_two_not_units_digit_of_square_l2040_204053


namespace NUMINAMATH_CALUDE_problem_solution_l2040_204007

theorem problem_solution (p q : ℝ) (h1 : p > 1) (h2 : q > 1)
  (h3 : 1/p + 1/q = 1) (h4 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2040_204007


namespace NUMINAMATH_CALUDE_quadratic_coincidence_l2040_204012

/-- A quadratic function with vertex at the origin -/
def QuadraticAtOrigin (a : ℝ) : ℝ → ℝ := λ x ↦ a * x^2

/-- The translated quadratic function -/
def TranslatedQuadratic : ℝ → ℝ := λ x ↦ 2 * x^2 + x - 1

/-- Theorem stating that if a quadratic function with vertex at the origin
    can be translated to coincide with y = 2x² + x - 1,
    then its analytical expression is y = 2x² -/
theorem quadratic_coincidence (a : ℝ) :
  (∃ h k : ℝ, ∀ x, QuadraticAtOrigin a (x - h) + k = TranslatedQuadratic x) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_coincidence_l2040_204012


namespace NUMINAMATH_CALUDE_smallest_whole_dollar_price_with_tax_l2040_204041

theorem smallest_whole_dollar_price_with_tax (n : ℕ) (x : ℕ) : n = 21 ↔ 
  n > 0 ∧ 
  x > 0 ∧
  (105 * x) % 100 = 0 ∧
  (105 * x) / 100 = n ∧
  ∀ m : ℕ, m > 0 → m < n → ¬∃ y : ℕ, y > 0 ∧ (105 * y) % 100 = 0 ∧ (105 * y) / 100 = m :=
sorry

end NUMINAMATH_CALUDE_smallest_whole_dollar_price_with_tax_l2040_204041


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2040_204054

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 2 + Complex.I) :
  z = 1/2 + 3/2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2040_204054


namespace NUMINAMATH_CALUDE_divisibility_problem_l2040_204033

theorem divisibility_problem (a b : ℕ) :
  (∃ k : ℕ, a = k * (b + 1)) ∧
  (∃ m : ℕ, 43 = m * (a + b)) →
  ((a = 22 ∧ b = 21) ∨
   (a = 33 ∧ b = 10) ∨
   (a = 40 ∧ b = 3) ∨
   (a = 42 ∧ b = 1)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2040_204033


namespace NUMINAMATH_CALUDE_cubic_polynomial_factor_property_l2040_204028

/-- Given a cubic polynomial 2x³ - hx + k where x + 2 and x - 1 are factors, 
    prove that |2h-3k| = 0 -/
theorem cubic_polynomial_factor_property (h k : ℝ) : 
  (∀ x, (x + 2) * (x - 1) ∣ (2 * x^3 - h * x + k)) → 
  |2 * h - 3 * k| = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_factor_property_l2040_204028


namespace NUMINAMATH_CALUDE_expression_value_l2040_204097

theorem expression_value (x y : ℝ) (hx : x = 1) (hy : y = -2) :
  3 * y^2 - x^2 + 2 * (2 * x^2 - 3 * x * y) - 3 * (x^2 + y^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2040_204097


namespace NUMINAMATH_CALUDE_tom_completion_time_l2040_204096

/-- Represents the duration of a combined BS and Ph.D. program -/
structure Program where
  bs_duration : ℕ
  phd_duration : ℕ

/-- Calculates the time taken by a student to complete the program given a completion ratio -/
def completion_time (p : Program) (ratio : ℚ) : ℚ :=
  ratio * (p.bs_duration + p.phd_duration)

theorem tom_completion_time :
  let p : Program := { bs_duration := 3, phd_duration := 5 }
  let ratio : ℚ := 3/4
  completion_time p ratio = 6 := by
  sorry

end NUMINAMATH_CALUDE_tom_completion_time_l2040_204096


namespace NUMINAMATH_CALUDE_sum_of_composite_functions_l2040_204037

def p (x : ℝ) : ℝ := |x| - 3

def q (x : ℝ) : ℝ := -|x|

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_composite_functions :
  (x_values.map (λ x => q (p x))).sum = -15 := by sorry

end NUMINAMATH_CALUDE_sum_of_composite_functions_l2040_204037


namespace NUMINAMATH_CALUDE_flower_shop_bouquets_l2040_204027

theorem flower_shop_bouquets (roses_per_bouquet : ℕ) 
  (rose_bouquets_sold daisy_bouquets_sold total_flowers : ℕ) :
  roses_per_bouquet = 12 →
  rose_bouquets_sold = 10 →
  daisy_bouquets_sold = 10 →
  total_flowers = 190 →
  total_flowers = roses_per_bouquet * rose_bouquets_sold + 
    (total_flowers - roses_per_bouquet * rose_bouquets_sold) →
  rose_bouquets_sold + daisy_bouquets_sold = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_flower_shop_bouquets_l2040_204027


namespace NUMINAMATH_CALUDE_smallest_b_for_equation_exists_solution_unique_smallest_solution_l2040_204084

theorem smallest_b_for_equation (A B : ℕ) : 
  (360 / (A * A * A / B) = 5) → B ≥ 3 :=
by
  sorry

theorem exists_solution : 
  ∃ (A B : ℕ), (360 / (A * A * A / B) = 5) ∧ B = 3 :=
by
  sorry

theorem unique_smallest_solution (A B : ℕ) : 
  (360 / (A * A * A / B) = 5) → B ≥ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_b_for_equation_exists_solution_unique_smallest_solution_l2040_204084


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2040_204075

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 1 - Complex.I → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2040_204075


namespace NUMINAMATH_CALUDE_z_sum_zero_implies_x_squared_minus_y_squared_eq_neg_three_z_times_one_plus_i_purely_imaginary_implies_modulus_eq_two_sqrt_two_l2040_204034

-- Define complex numbers z₁ and z₂
def z₁ (x : ℝ) : ℂ := (2 * x + 1) + 2 * Complex.I
def z₂ (x y : ℝ) : ℂ := -x - y * Complex.I

-- Theorem 1
theorem z_sum_zero_implies_x_squared_minus_y_squared_eq_neg_three
  (x y : ℝ) (h : z₁ x + z₂ x y = 0) :
  x^2 - y^2 = -3 := by sorry

-- Theorem 2
theorem z_times_one_plus_i_purely_imaginary_implies_modulus_eq_two_sqrt_two
  (x : ℝ) (h : (Complex.I + 1) * z₁ x = Complex.I * (Complex.im ((Complex.I + 1) * z₁ x))) :
  Complex.abs (z₁ x) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_z_sum_zero_implies_x_squared_minus_y_squared_eq_neg_three_z_times_one_plus_i_purely_imaginary_implies_modulus_eq_two_sqrt_two_l2040_204034


namespace NUMINAMATH_CALUDE_cos_neg_600_degrees_l2040_204064

theorem cos_neg_600_degrees : Real.cos ((-600 : ℝ) * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_600_degrees_l2040_204064


namespace NUMINAMATH_CALUDE_value_of_y_l2040_204029

theorem value_of_y (x y : ℚ) : 
  x = 51 → x^3*y - 2*x^2*y + x*y = 127500 → y = 1/51 := by sorry

end NUMINAMATH_CALUDE_value_of_y_l2040_204029


namespace NUMINAMATH_CALUDE_cubic_function_property_l2040_204020

theorem cubic_function_property (p q r s : ℝ) : 
  let g := fun (x : ℝ) => p * x^3 + q * x^2 + r * x + s
  (g (-1) = 2) → (g (-2) = -1) → (g 1 = -2) → 
  (9*p - 3*q + 3*r - s = -2) := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2040_204020


namespace NUMINAMATH_CALUDE_num_terms_xyz_4_is_15_l2040_204082

/-- The number of terms in the expansion of (x+y+z)^4 -/
def num_terms_xyz_4 : ℕ := sorry

/-- Theorem stating that the number of terms in (x+y+z)^4 is 15 -/
theorem num_terms_xyz_4_is_15 : num_terms_xyz_4 = 15 := by sorry

end NUMINAMATH_CALUDE_num_terms_xyz_4_is_15_l2040_204082


namespace NUMINAMATH_CALUDE_rotation_volume_sum_l2040_204036

/-- Given a square ABCD with side length a and a point M at distance b from its center,
    the sum of volumes of solids obtained by rotating triangles ABM, BCM, CDM, and DAM
    around lines AB, BC, CD, and DA respectively is equal to 3a^3/8 -/
theorem rotation_volume_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  let square := {A : ℝ × ℝ | ∃ (x y : ℝ), x ∈ [0, a] ∧ y ∈ [0, a] ∧ A = (x, y)}
  let center := (a/2, a/2)
  let M : ℝ × ℝ := sorry -- A point at distance b from the center
  let volume_sum := sorry -- Sum of volumes of rotated triangles
  volume_sum = 3 * a^3 / 8 := by
sorry

end NUMINAMATH_CALUDE_rotation_volume_sum_l2040_204036


namespace NUMINAMATH_CALUDE_solution_of_system_l2040_204078

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x^((Real.log y)^(Real.log (Real.log x))) = 10^(y^2)
def equation2 (x y : ℝ) : Prop := y^((Real.log x)^(Real.log (Real.log y))) = y^y

-- State the theorem
theorem solution_of_system :
  ∃ (x y : ℝ), x > 1 ∧ y > 1 ∧ equation1 x y ∧ equation2 x y ∧ x = 10^(10^10) ∧ y = 10^10 :=
sorry

end NUMINAMATH_CALUDE_solution_of_system_l2040_204078


namespace NUMINAMATH_CALUDE_divisibility_by_three_l2040_204042

theorem divisibility_by_three (a b c : ℤ) (h : (9 : ℤ) ∣ (a^3 + b^3 + c^3)) :
  (3 : ℤ) ∣ a ∨ (3 : ℤ) ∣ b ∨ (3 : ℤ) ∣ c :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l2040_204042


namespace NUMINAMATH_CALUDE_x_less_than_y_less_than_zero_l2040_204089

theorem x_less_than_y_less_than_zero (x y : ℝ) 
  (h1 : 2 * x - 3 * y > 6 * x) 
  (h2 : 3 * x - 4 * y < 2 * y - x) : 
  x < y ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_y_less_than_zero_l2040_204089


namespace NUMINAMATH_CALUDE_first_account_interest_rate_l2040_204019

/-- Proves that the interest rate of the first account is 0.02 given the problem conditions --/
theorem first_account_interest_rate :
  ∀ (r : ℝ),
    r > 0 →
    r < 1 →
    1000 * r + 1800 * 0.04 = 92 →
    r = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_first_account_interest_rate_l2040_204019


namespace NUMINAMATH_CALUDE_lisas_eggs_per_child_l2040_204094

theorem lisas_eggs_per_child (breakfasts_per_year : ℕ) (num_children : ℕ) 
  (husband_eggs : ℕ) (self_eggs : ℕ) (total_eggs : ℕ) :
  breakfasts_per_year = 260 →
  num_children = 4 →
  husband_eggs = 3 →
  self_eggs = 2 →
  total_eggs = 3380 →
  ∃ (eggs_per_child : ℕ), 
    eggs_per_child = 2 ∧
    total_eggs = breakfasts_per_year * (num_children * eggs_per_child + husband_eggs + self_eggs) :=
by sorry

end NUMINAMATH_CALUDE_lisas_eggs_per_child_l2040_204094


namespace NUMINAMATH_CALUDE_stick_marking_underdetermined_l2040_204001

/-- Represents the length of a portion of the stick -/
structure Portion where
  length : ℚ
  isValid : 0 < length ∧ length ≤ 1

/-- Represents the configuration of markings on the stick -/
structure StickMarkings where
  fifthPortions : ℕ
  xPortions : ℕ
  xLength : ℚ
  totalLength : ℚ
  validTotal : fifthPortions + xPortions = 8
  validLength : fifthPortions * (1/5) + xPortions * xLength = totalLength

/-- Theorem stating that the problem is underdetermined -/
theorem stick_marking_underdetermined :
  ∀ (m : StickMarkings),
    m.totalLength = 1 →
    ∃ (m' : StickMarkings),
      m'.totalLength = 1 ∧
      m'.fifthPortions ≠ m.fifthPortions ∧
      m'.xLength ≠ m.xLength :=
sorry

end NUMINAMATH_CALUDE_stick_marking_underdetermined_l2040_204001


namespace NUMINAMATH_CALUDE_crab_meat_cost_per_pound_l2040_204003

/-- The cost of crab meat per pound given Johnny's crab dish production and expenses -/
theorem crab_meat_cost_per_pound 
  (dishes_per_day : ℕ) 
  (meat_per_dish : ℚ) 
  (weekly_expense : ℕ) 
  (closed_days : ℕ) : 
  dishes_per_day = 40 → 
  meat_per_dish = 3/2 → 
  weekly_expense = 1920 → 
  closed_days = 3 → 
  (weekly_expense : ℚ) / ((7 - closed_days) * dishes_per_day * meat_per_dish) = 8 := by
  sorry

end NUMINAMATH_CALUDE_crab_meat_cost_per_pound_l2040_204003


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2040_204030

theorem triangle_ABC_properties (A B C : ℝ) :
  0 < A ∧ A < 2 * π / 3 →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  Real.cos C + (Real.cos A - Real.sqrt 3 * Real.sin A) * Real.cos B = 0 →
  Real.sin (A - π / 3) = 3 / 5 →
  B = π / 3 ∧ Real.sin (2 * C) = (24 + 7 * Real.sqrt 3) / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2040_204030


namespace NUMINAMATH_CALUDE_smallest_n_same_last_two_digits_l2040_204092

theorem smallest_n_same_last_two_digits : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬(107 * m ≡ m [ZMOD 100])) ∧ 
  (107 * n ≡ n [ZMOD 100]) ∧
  n = 50 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_same_last_two_digits_l2040_204092


namespace NUMINAMATH_CALUDE_xiaoming_money_problem_l2040_204047

theorem xiaoming_money_problem (price_left : ℕ) (price_right : ℕ) :
  price_right = price_left - 1 →
  12 * price_left = 14 * price_right →
  12 * price_left = 84 :=
by sorry

end NUMINAMATH_CALUDE_xiaoming_money_problem_l2040_204047


namespace NUMINAMATH_CALUDE_madison_distance_l2040_204059

/-- Represents the travel from Gardensquare to Madison -/
structure Journey where
  time : ℝ
  speed : ℝ
  mapScale : ℝ

/-- Calculates the distance on the map given a journey -/
def mapDistance (j : Journey) : ℝ :=
  j.time * j.speed * j.mapScale

/-- Theorem stating that the distance on the map is 5 inches -/
theorem madison_distance (j : Journey) 
  (h1 : j.time = 5)
  (h2 : j.speed = 60)
  (h3 : j.mapScale = 0.016666666666666666) : 
  mapDistance j = 5 := by
  sorry

end NUMINAMATH_CALUDE_madison_distance_l2040_204059


namespace NUMINAMATH_CALUDE_ratio_problem_l2040_204057

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2040_204057


namespace NUMINAMATH_CALUDE_relation_between_exponents_l2040_204055

/-- Given real numbers a, b, c, d, x, y, p satisfying certain equations,
    prove that y = (3 * p^2) / 2 -/
theorem relation_between_exponents
  (a b c d x y p : ℝ)
  (h1 : a^x = c^(3*p))
  (h2 : c^(3*p) = b^2)
  (h3 : c^y = b^p)
  (h4 : b^p = d^3)
  : y = (3 * p^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_relation_between_exponents_l2040_204055


namespace NUMINAMATH_CALUDE_equation_solution_l2040_204025

theorem equation_solution : ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -1 ∧
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2040_204025


namespace NUMINAMATH_CALUDE_power_equality_l2040_204067

theorem power_equality (q : ℕ) : 81^7 = 3^q → q = 28 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2040_204067


namespace NUMINAMATH_CALUDE_exists_special_quadratic_trinomial_l2040_204093

/-- A quadratic trinomial function -/
def QuadraticTrinomial := ℝ → ℝ

/-- The n-th composition of a function with itself -/
def compose_n_times (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
| 0 => id
| n + 1 => f ∘ (compose_n_times f n)

/-- The number of distinct real roots of a function -/
noncomputable def num_distinct_real_roots (f : ℝ → ℝ) : ℕ := sorry

/-- The main theorem statement -/
theorem exists_special_quadratic_trinomial :
  ∃ (f : QuadraticTrinomial),
    ∀ (n : ℕ), num_distinct_real_roots (compose_n_times f n) = 2 * n :=
sorry

end NUMINAMATH_CALUDE_exists_special_quadratic_trinomial_l2040_204093


namespace NUMINAMATH_CALUDE_absolute_value_equation_l2040_204032

theorem absolute_value_equation (x : ℝ) (h : |2 - x| = 2 + |x|) : |2 - x| = 2 - x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l2040_204032


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2040_204005

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  a^2 + b^2 + c^2 ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2040_204005


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l2040_204058

theorem right_triangle_third_side_product (a b c : ℝ) : 
  a = 4 → b = 5 → (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  (c = 3 ∨ c = Real.sqrt 41) → 
  3 * Real.sqrt 41 = (if c = 3 then 3 else Real.sqrt 41) * (if c = Real.sqrt 41 then Real.sqrt 41 else 3) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l2040_204058


namespace NUMINAMATH_CALUDE_students_present_l2040_204008

theorem students_present (total : ℕ) (absent_fraction : ℚ) : total = 28 → absent_fraction = 2/7 → total - (absent_fraction * total).floor = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_present_l2040_204008


namespace NUMINAMATH_CALUDE_no_interchange_possible_l2040_204077

/-- Represents the circular arrangement of three tiles -/
inductive CircularArrangement
  | ABC
  | BCA
  | CAB

/-- Represents a move that slides a tile to an adjacent vacant space -/
inductive Move
  | Left
  | Right

/-- Applies a move to a circular arrangement -/
def applyMove (arr : CircularArrangement) (m : Move) : CircularArrangement :=
  match arr, m with
  | CircularArrangement.ABC, Move.Right => CircularArrangement.BCA
  | CircularArrangement.BCA, Move.Right => CircularArrangement.CAB
  | CircularArrangement.CAB, Move.Right => CircularArrangement.ABC
  | CircularArrangement.ABC, Move.Left => CircularArrangement.CAB
  | CircularArrangement.BCA, Move.Left => CircularArrangement.ABC
  | CircularArrangement.CAB, Move.Left => CircularArrangement.BCA

/-- Applies a sequence of moves to a circular arrangement -/
def applyMoves (arr : CircularArrangement) (moves : List Move) : CircularArrangement :=
  match moves with
  | [] => arr
  | m :: ms => applyMoves (applyMove arr m) ms

/-- Theorem stating that it's impossible to interchange 1 and 3 -/
theorem no_interchange_possible (moves : List Move) :
  applyMoves CircularArrangement.ABC moves ≠ CircularArrangement.BCA :=
sorry

end NUMINAMATH_CALUDE_no_interchange_possible_l2040_204077


namespace NUMINAMATH_CALUDE_right_triangle_area_l2040_204091

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b = 24 →
  c = 24 →
  a^2 + c^2 = (24 + b)^2 →
  (1/2) * a * c = 216 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2040_204091


namespace NUMINAMATH_CALUDE_ariella_daniella_savings_difference_l2040_204066

theorem ariella_daniella_savings_difference :
  ∀ (ariella_initial daniella_savings : ℝ),
    daniella_savings = 400 →
    ariella_initial + ariella_initial * 0.1 * 2 = 720 →
    ariella_initial > daniella_savings →
    ariella_initial - daniella_savings = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_ariella_daniella_savings_difference_l2040_204066


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2040_204018

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 ≥ 0) ↔ -2 ≤ m ∧ m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2040_204018


namespace NUMINAMATH_CALUDE_roots_equation_sum_l2040_204060

theorem roots_equation_sum (a b : ℝ) : 
  (a^2 + a - 2022 = 0) → 
  (b^2 + b - 2022 = 0) → 
  (a ≠ b) →
  (a^2 + 2*a + b = 2021) := by
sorry

end NUMINAMATH_CALUDE_roots_equation_sum_l2040_204060


namespace NUMINAMATH_CALUDE_shoes_sold_day1_l2040_204095

/-- Represents the sales data for a shoe store --/
structure ShoeSales where
  shoe_price : ℕ
  boot_price : ℕ
  day1_shoes : ℕ
  day1_boots : ℕ
  day2_shoes : ℕ
  day2_boots : ℕ

/-- Theorem stating the number of shoes sold on day 1 given the sales conditions --/
theorem shoes_sold_day1 (s : ShoeSales) : 
  s.boot_price = s.shoe_price + 15 →
  s.day1_shoes * s.shoe_price + s.day1_boots * s.boot_price = 460 →
  s.day2_shoes * s.shoe_price + s.day2_boots * s.boot_price = 560 →
  s.day1_boots = 16 →
  s.day2_shoes = 8 →
  s.day2_boots = 32 →
  s.day1_shoes = 94 := by
  sorry

#check shoes_sold_day1

end NUMINAMATH_CALUDE_shoes_sold_day1_l2040_204095


namespace NUMINAMATH_CALUDE_initial_people_count_initial_people_count_proof_l2040_204069

theorem initial_people_count : ℕ → Prop :=
  fun n => 
    (n / 3 : ℚ) / 2 = 15 → n = 90

-- The proof goes here
theorem initial_people_count_proof : initial_people_count 90 := by
  sorry

end NUMINAMATH_CALUDE_initial_people_count_initial_people_count_proof_l2040_204069


namespace NUMINAMATH_CALUDE_die_roll_probability_l2040_204013

def roll_die : ℕ := 6
def num_trials : ℕ := 6
def min_success : ℕ := 5
def success_probability : ℚ := 1/3

theorem die_roll_probability : 
  (success_probability ^ num_trials) + 
  (Nat.choose num_trials min_success * success_probability ^ min_success * (1 - success_probability) ^ (num_trials - min_success)) = 13/729 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l2040_204013


namespace NUMINAMATH_CALUDE_parallelogram_sum_impossibility_l2040_204031

theorem parallelogram_sum_impossibility :
  ¬ ∃ (a b h : ℕ+), (b * h : ℕ) + 2 * a + 2 * b + 6 = 102 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_sum_impossibility_l2040_204031


namespace NUMINAMATH_CALUDE_flower_cost_proof_l2040_204011

/-- Proves that if Lilly saves $2 per day for 22 days and can buy 11 flowers with her savings, then each flower costs $4. -/
theorem flower_cost_proof (days : ℕ) (daily_savings : ℚ) (num_flowers : ℕ) 
  (h1 : days = 22) 
  (h2 : daily_savings = 2) 
  (h3 : num_flowers = 11) : 
  (days * daily_savings) / num_flowers = 4 := by
  sorry

#check flower_cost_proof

end NUMINAMATH_CALUDE_flower_cost_proof_l2040_204011


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_negative_eight_l2040_204040

/-- An odd function satisfying f(x-4) = -f(x) -/
def OddPeriodicFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x - 4) = -f x)

theorem sum_of_roots_equals_negative_eight
  (f : ℝ → ℝ) (m : ℝ) (x₁ x₂ x₃ x₄ : ℝ)
  (hf : OddPeriodicFunction f)
  (hm : m > 0)
  (h_roots : x₁ ∈ Set.Icc (-8 : ℝ) 8 ∧
             x₂ ∈ Set.Icc (-8 : ℝ) 8 ∧
             x₃ ∈ Set.Icc (-8 : ℝ) 8 ∧
             x₄ ∈ Set.Icc (-8 : ℝ) 8)
  (h_distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)
  (h_eq : f x₁ = m ∧ f x₂ = m ∧ f x₃ = m ∧ f x₄ = m) :
  x₁ + x₂ + x₃ + x₄ = -8 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_roots_equals_negative_eight_l2040_204040


namespace NUMINAMATH_CALUDE_betty_order_cost_l2040_204065

/-- The total cost of Betty's order -/
def total_cost (slippers_quantity : ℕ) (slippers_price : ℚ) 
               (lipstick_quantity : ℕ) (lipstick_price : ℚ)
               (hair_color_quantity : ℕ) (hair_color_price : ℚ) : ℚ :=
  slippers_quantity * slippers_price + 
  lipstick_quantity * lipstick_price + 
  hair_color_quantity * hair_color_price

/-- Theorem stating that Betty's total order cost is $44 -/
theorem betty_order_cost : 
  total_cost 6 (5/2) 4 (5/4) 8 3 = 44 := by
  sorry

#eval total_cost 6 (5/2) 4 (5/4) 8 3

end NUMINAMATH_CALUDE_betty_order_cost_l2040_204065


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2040_204099

theorem not_sufficient_not_necessary (p q : Prop) : 
  (¬(p ∧ q → p ∨ q)) ∧ (¬(p ∨ q → ¬(p ∧ q))) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2040_204099


namespace NUMINAMATH_CALUDE_taxi_theorem_l2040_204063

def taxi_distances : List ℤ := [9, -3, -5, 4, -8, 6]

def fuel_consumption : ℚ := 0.08
def gasoline_price : ℚ := 6
def starting_price : ℚ := 6
def additional_charge : ℚ := 1.5
def starting_distance : ℕ := 3

def total_distance (distances : List ℤ) : ℕ :=
  (distances.map (Int.natAbs)).sum

def fuel_cost (distance : ℕ) : ℚ :=
  distance * fuel_consumption * gasoline_price

def segment_income (distance : ℕ) : ℚ :=
  if distance ≤ starting_distance then
    starting_price
  else
    starting_price + (distance - starting_distance) * additional_charge

def total_income (distances : List ℤ) : ℚ :=
  (distances.map (Int.natAbs)).map segment_income |>.sum

def net_income (distances : List ℤ) : ℚ :=
  total_income distances - fuel_cost (total_distance distances)

theorem taxi_theorem :
  total_distance taxi_distances = 35 ∧
  fuel_cost (total_distance taxi_distances) = 16.8 ∧
  net_income taxi_distances = 44.7 := by
  sorry

end NUMINAMATH_CALUDE_taxi_theorem_l2040_204063


namespace NUMINAMATH_CALUDE_radius_ratio_in_regular_hexagonal_pyramid_l2040_204073

/-- A regular hexagonal pyramid with a circumscribed sphere and an inscribed sphere. -/
structure RegularHexagonalPyramid where
  /-- The radius of the circumscribed sphere -/
  R_c : ℝ
  /-- The radius of the inscribed sphere -/
  R_i : ℝ
  /-- The center of the circumscribed sphere lies on the surface of the inscribed sphere -/
  center_on_surface : R_c = R_i + R_i

/-- The ratio of the radius of the circumscribed sphere to the radius of the inscribed sphere
    in a regular hexagonal pyramid where the center of the circumscribed sphere lies on
    the surface of the inscribed sphere is equal to 1 + √(7/3). -/
theorem radius_ratio_in_regular_hexagonal_pyramid (p : RegularHexagonalPyramid) :
  p.R_c / p.R_i = 1 + Real.sqrt (7/3) := by
  sorry

end NUMINAMATH_CALUDE_radius_ratio_in_regular_hexagonal_pyramid_l2040_204073


namespace NUMINAMATH_CALUDE_morning_travel_time_l2040_204038

/-- Proves that the time taken to move in the morning is 1 hour -/
theorem morning_travel_time (v_morning v_afternoon : ℝ) (time_diff : ℝ) 
  (h1 : v_morning = 20)
  (h2 : v_afternoon = 10)
  (h3 : time_diff = 1) :
  ∃ (t_morning : ℝ), t_morning = 1 ∧ t_morning * v_morning = (t_morning + time_diff) * v_afternoon :=
by sorry

end NUMINAMATH_CALUDE_morning_travel_time_l2040_204038


namespace NUMINAMATH_CALUDE_female_officers_count_l2040_204086

/-- The total number of police officers on duty that night -/
def total_on_duty : ℕ := 160

/-- The fraction of officers on duty that were female -/
def female_fraction : ℚ := 1/2

/-- The percentage of all female officers that were on duty -/
def female_on_duty_percentage : ℚ := 16/100

/-- The total number of female officers on the police force -/
def total_female_officers : ℕ := 500

theorem female_officers_count :
  total_female_officers = 
    (total_on_duty * female_fraction) / female_on_duty_percentage := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l2040_204086


namespace NUMINAMATH_CALUDE_triangle_properties_l2040_204009

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.c - t.b = 2 * t.b * Real.cos t.A) :
  (t.a = 2 * Real.sqrt 6 ∧ t.b = 3 → t.c = 5) ∧
  (t.C = Real.pi / 2 → t.B = Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2040_204009


namespace NUMINAMATH_CALUDE_symbol_values_l2040_204081

theorem symbol_values (star circle ring : ℤ) 
  (h1 : star + ring = 46)
  (h2 : star + circle = 91)
  (h3 : circle + ring = 63) :
  star = 37 ∧ circle = 54 ∧ ring = 9 := by
  sorry

end NUMINAMATH_CALUDE_symbol_values_l2040_204081


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2040_204051

universe u

def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U : 
  (U \ A) = {0, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2040_204051


namespace NUMINAMATH_CALUDE_problem_solution_l2040_204085

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 2 * m * x - 1

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (3 * f m x + 4) / (x - 2)

theorem problem_solution (m : ℝ) :
  m > 0 →
  (∀ x, f m x < 0 ↔ -3 < x ∧ x < 1) →
  (∃ min_g : ℝ, ∀ x > 2, g m x ≥ min_g ∧ ∃ x₀ > 2, g m x₀ = min_g) ∧
  min_g = 12 ∧
  (∃ x₁ x₂ : ℝ, x₁ ∈ [-3, 0] ∧ x₂ ∈ [-3, 0] ∧ |f m x₁ - f m x₂| ≥ 4 → m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2040_204085


namespace NUMINAMATH_CALUDE_grade12_population_l2040_204014

/-- Represents the number of students in each grade -/
structure GradePopulation where
  grade10 : Nat
  grade11 : Nat
  grade12 : Nat

/-- Represents the number of students sampled from each grade -/
structure SampleSize where
  grade10 : Nat
  total : Nat

/-- Check if the sampling is proportional to the population -/
def isProportionalSampling (pop : GradePopulation) (sample : SampleSize) : Prop :=
  sample.grade10 * (pop.grade10 + pop.grade11 + pop.grade12) = 
  sample.total * pop.grade10

theorem grade12_population 
  (pop : GradePopulation)
  (sample : SampleSize)
  (h1 : pop.grade10 = 1000)
  (h2 : pop.grade11 = 1200)
  (h3 : sample.total = 66)
  (h4 : sample.grade10 = 20)
  (h5 : isProportionalSampling pop sample) :
  pop.grade12 = 1100 := by
  sorry

#check grade12_population

end NUMINAMATH_CALUDE_grade12_population_l2040_204014


namespace NUMINAMATH_CALUDE_infinite_perfect_squares_in_sequence_l2040_204056

theorem infinite_perfect_squares_in_sequence :
  ∃ f : ℕ → ℕ × ℕ, 
    (∀ i : ℕ, (f i).1^2 = 1 + 17 * (f i).2^2) ∧ 
    (∀ i j : ℕ, i ≠ j → f i ≠ f j) := by
  sorry

end NUMINAMATH_CALUDE_infinite_perfect_squares_in_sequence_l2040_204056


namespace NUMINAMATH_CALUDE_cos_135_degrees_l2040_204062

theorem cos_135_degrees : Real.cos (135 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l2040_204062


namespace NUMINAMATH_CALUDE_project_profit_analysis_l2040_204000

/-- Represents the net profit of a project in millions of yuan -/
def net_profit (n : ℕ+) : ℚ :=
  100 * n - (4 * n^2 + 40 * n) - 144

/-- Represents the average annual profit of a project in millions of yuan -/
def avg_annual_profit (n : ℕ+) : ℚ :=
  net_profit n / n

theorem project_profit_analysis :
  ∀ n : ℕ+,
  (net_profit n = -4 * (n - 3) * (n - 12)) ∧
  (net_profit n > 0 ↔ 3 < n ∧ n < 12) ∧
  (∀ m : ℕ+, avg_annual_profit m ≤ avg_annual_profit 6) := by
  sorry

#check project_profit_analysis

end NUMINAMATH_CALUDE_project_profit_analysis_l2040_204000


namespace NUMINAMATH_CALUDE_complement_of_A_l2040_204052

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

-- Theorem statement
theorem complement_of_A : 
  (U \ A) = {x : ℝ | x < -1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2040_204052


namespace NUMINAMATH_CALUDE_number_division_proof_l2040_204070

theorem number_division_proof (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- Ensure all parts are positive
  a / 5 = b / 7 ∧ a / 5 = c / 4 ∧ a / 5 = d / 8 →  -- Parts are proportional
  c = 60 →  -- Smallest part is 60
  a + b + c + d = 360 :=  -- Total number is 360
by sorry

end NUMINAMATH_CALUDE_number_division_proof_l2040_204070


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2040_204026

theorem least_subtraction_for_divisibility :
  ∃ (n : ℕ), n ≤ 5 ∧ (∀ m : ℕ, m < n → ¬(37 ∣ (5000 - m))) ∧ (37 ∣ (5000 - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2040_204026


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2040_204072

theorem imaginary_part_of_complex_number (i : ℂ) (h : i^2 = -1) :
  let z := i^2 / (1 - i)
  (z.im : ℝ) = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2040_204072


namespace NUMINAMATH_CALUDE_ttakji_square_arrangement_l2040_204088

/-- The number of ttakjis on one side of the large square -/
def n : ℕ := 61

/-- The number of ttakjis on the perimeter of the large square -/
def perimeter_ttakjis : ℕ := 240

theorem ttakji_square_arrangement :
  (4 * n - 4 = perimeter_ttakjis) ∧ (n^2 = 3721) := by sorry

end NUMINAMATH_CALUDE_ttakji_square_arrangement_l2040_204088


namespace NUMINAMATH_CALUDE_marie_binders_count_l2040_204049

theorem marie_binders_count :
  let notebooks_count : ℕ := 4
  let stamps_per_notebook : ℕ := 20
  let stamps_per_binder : ℕ := 50
  let kept_fraction : ℚ := 1/4
  let stamps_given_away : ℕ := 135
  ∃ binders_count : ℕ,
    (notebooks_count * stamps_per_notebook + binders_count * stamps_per_binder) * (1 - kept_fraction) = stamps_given_away ∧
    binders_count = 2 :=
by sorry

end NUMINAMATH_CALUDE_marie_binders_count_l2040_204049


namespace NUMINAMATH_CALUDE_probability_theorem_l2040_204079

/-- Represents the number of buttons in a jar -/
structure JarContents where
  red : ℕ
  blue : ℕ

/-- Represents the state of both jars -/
structure JarState where
  jarA : JarContents
  jarB : JarContents

def initial_jarA : JarContents := { red := 6, blue := 14 }

def initial_jarB : JarContents := { red := 0, blue := 0 }

def initial_state : JarState := { jarA := initial_jarA, jarB := initial_jarB }

def buttons_removed (state : JarState) : ℕ :=
  initial_jarA.red + initial_jarA.blue - (state.jarA.red + state.jarA.blue)

def same_number_removed (state : JarState) : Prop :=
  state.jarB.red = state.jarB.blue

def fraction_remaining (state : JarState) : ℚ :=
  (state.jarA.red + state.jarA.blue) / (initial_jarA.red + initial_jarA.blue)

def probability_both_red (state : JarState) : ℚ :=
  (state.jarA.red / (state.jarA.red + state.jarA.blue)) *
  (state.jarB.red / (state.jarB.red + state.jarB.blue))

theorem probability_theorem (final_state : JarState) :
  buttons_removed final_state > 0 ∧
  same_number_removed final_state ∧
  fraction_remaining final_state = 5/7 →
  probability_both_red final_state = 3/28 := by
  sorry

#check probability_theorem

end NUMINAMATH_CALUDE_probability_theorem_l2040_204079


namespace NUMINAMATH_CALUDE_fraction_of_single_men_l2040_204050

theorem fraction_of_single_men (total : ℝ) (h1 : total > 0) : 
  let women := 0.64 * total
  let men := total - women
  let married := 0.60 * total
  let married_women := 0.75 * women
  let married_men := married - married_women
  let single_men := men - married_men
  single_men / men = 2/3 := by sorry

end NUMINAMATH_CALUDE_fraction_of_single_men_l2040_204050


namespace NUMINAMATH_CALUDE_volunteer_selection_theorem_l2040_204076

/-- The number of volunteers --/
def n : ℕ := 5

/-- The number of roles to be filled --/
def k : ℕ := 4

/-- The number of ways to arrange k people in k positions --/
def arrange (k : ℕ) : ℕ := Nat.factorial k

/-- The number of ways to choose k people from n people --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of ways to select and arrange volunteers for roles --/
def totalWays : ℕ :=
  arrange (k - 1) + choose (n - 1) (k - 1) * (k - 1) * arrange (k - 1)

theorem volunteer_selection_theorem : totalWays = 96 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_theorem_l2040_204076


namespace NUMINAMATH_CALUDE_set_equality_implies_a_value_l2040_204021

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {a, a^2}
def B (b : ℝ) : Set ℝ := {1, b}

-- State the theorem
theorem set_equality_implies_a_value (a b : ℝ) :
  A a = B b → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_value_l2040_204021


namespace NUMINAMATH_CALUDE_only_vinyl_chloride_and_benzene_planar_l2040_204046

/-- Represents an organic compound -/
inductive OrganicCompound
| Propylene
| VinylChloride
| Benzene
| Toluene

/-- Predicate to check if all atoms in a compound are on the same plane -/
def all_atoms_on_same_plane (c : OrganicCompound) : Prop :=
  match c with
  | OrganicCompound.Propylene => False
  | OrganicCompound.VinylChloride => True
  | OrganicCompound.Benzene => True
  | OrganicCompound.Toluene => False

/-- Theorem stating that only vinyl chloride and benzene have all atoms on the same plane -/
theorem only_vinyl_chloride_and_benzene_planar :
  ∀ c : OrganicCompound, all_atoms_on_same_plane c ↔ (c = OrganicCompound.VinylChloride ∨ c = OrganicCompound.Benzene) :=
by
  sorry


end NUMINAMATH_CALUDE_only_vinyl_chloride_and_benzene_planar_l2040_204046


namespace NUMINAMATH_CALUDE_evaluate_expression_l2040_204023

theorem evaluate_expression : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2040_204023
