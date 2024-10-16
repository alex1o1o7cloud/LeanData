import Mathlib

namespace NUMINAMATH_CALUDE_octal_sum_equals_2351_l3639_363971

/-- Converts a base-8 number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a decimal number to its base-8 representation as a list of digits -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- Theorem stating that the sum of 1457₈ and 672₈ in base 8 is 2351₈ -/
theorem octal_sum_equals_2351 :
  let a := octal_to_decimal [7, 5, 4, 1]  -- 1457₈
  let b := octal_to_decimal [2, 7, 6]     -- 672₈
  decimal_to_octal (a + b) = [1, 5, 3, 2] -- 2351₈
  := by sorry

end NUMINAMATH_CALUDE_octal_sum_equals_2351_l3639_363971


namespace NUMINAMATH_CALUDE_hyperbola_equation_y_axis_l3639_363926

/-- Given a hyperbola with foci on the y-axis, ratio of real to imaginary axis 2:3, 
    and passing through (√6, 2), prove its equation is y²/1 - x²/3 = 3 -/
theorem hyperbola_equation_y_axis (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : 3 * a = 2 * b) (h4 : 4 / a^2 - 6 / b^2 = 1) :
  ∃ (k : ℝ), k * (y^2 / 1 - x^2 / 3) = 3 := by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_y_axis_l3639_363926


namespace NUMINAMATH_CALUDE_shells_added_l3639_363934

/-- The amount of shells added to Jovana's bucket -/
theorem shells_added (initial_amount final_amount : ℝ) 
  (h1 : initial_amount = 5.75)
  (h2 : final_amount = 28.3) : 
  final_amount - initial_amount = 22.55 := by
  sorry

end NUMINAMATH_CALUDE_shells_added_l3639_363934


namespace NUMINAMATH_CALUDE_frog_game_result_l3639_363979

def frog_A_jump : ℕ := 10
def frog_B_jump : ℕ := 15
def trap_interval : ℕ := 12

def first_trap (jump_distance : ℕ) : ℕ :=
  (trap_interval / jump_distance) * jump_distance

theorem frog_game_result :
  let first_frog_trap := min (first_trap frog_A_jump) (first_trap frog_B_jump)
  let other_frog_distance := if first_frog_trap = first_trap frog_B_jump
                             then (first_frog_trap / frog_B_jump) * frog_A_jump
                             else (first_frog_trap / frog_A_jump) * frog_B_jump
  (trap_interval - (other_frog_distance % trap_interval)) % trap_interval = 8 := by
  sorry

end NUMINAMATH_CALUDE_frog_game_result_l3639_363979


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l3639_363994

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l3639_363994


namespace NUMINAMATH_CALUDE_legos_set_cost_l3639_363956

theorem legos_set_cost (total_earnings : ℕ) (num_cars : ℕ) (car_price : ℕ) (legos_price : ℕ) :
  total_earnings = 45 →
  num_cars = 3 →
  car_price = 5 →
  total_earnings = num_cars * car_price + legos_price →
  legos_price = 30 := by
sorry

end NUMINAMATH_CALUDE_legos_set_cost_l3639_363956


namespace NUMINAMATH_CALUDE_swimmers_speed_l3639_363931

/-- Proves that a swimmer's speed in still water is 12 km/h given the conditions -/
theorem swimmers_speed (v s : ℝ) (h1 : s = 4) (h2 : (v - s)⁻¹ = 2 * (v + s)⁻¹) : v = 12 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_speed_l3639_363931


namespace NUMINAMATH_CALUDE_insect_jumps_l3639_363949

theorem insect_jumps (s : ℝ) (h_s : 1/2 < s ∧ s < 1) (c : ℝ) (h_c : 0 ≤ c ∧ c ≤ 1) :
  ∀ ε > 0, ∃ (n : ℕ) (x : ℕ → ℝ),
    (x 0 = 0 ∨ x 0 = 1) ∧
    (∀ i, i < n → (x (i + 1) = x i * s ∨ x (i + 1) = (x i - 1) * s + 1)) ∧
    |x n - c| < ε :=
by sorry

end NUMINAMATH_CALUDE_insect_jumps_l3639_363949


namespace NUMINAMATH_CALUDE_eventually_constant_l3639_363942

/-- S(n) is defined as n - m^2, where m is the greatest integer with m^2 ≤ n -/
def S (n : ℕ) : ℕ :=
  n - (Nat.sqrt n) ^ 2

/-- The sequence a_k is defined recursively -/
def a (A : ℕ) : ℕ → ℕ
  | 0 => A
  | k + 1 => a A k + S (a A k)

/-- The main theorem stating the condition for the sequence to be eventually constant -/
theorem eventually_constant (A : ℕ) :
  (∃ k : ℕ, ∀ n ≥ k, a A n = a A k) ↔ ∃ m : ℕ, A = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_eventually_constant_l3639_363942


namespace NUMINAMATH_CALUDE_point_reflection_y_axis_l3639_363946

/-- Given a point P(2,1) in the Cartesian coordinate system, 
    its coordinates with respect to the y-axis are (-2,1). -/
theorem point_reflection_y_axis : 
  let P : ℝ × ℝ := (2, 1)
  let reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
  reflect_y P = (-2, 1) := by sorry

end NUMINAMATH_CALUDE_point_reflection_y_axis_l3639_363946


namespace NUMINAMATH_CALUDE_zero_not_necessarily_in_2_5_l3639_363997

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f having its only zero in (1,3)
def has_only_zero_in_open_interval (f : ℝ → ℝ) : Prop :=
  ∃ x, 1 < x ∧ x < 3 ∧ f x = 0 ∧ ∀ y, f y = 0 → (1 < y ∧ y < 3)

-- State the theorem
theorem zero_not_necessarily_in_2_5 
  (h : has_only_zero_in_open_interval f) : 
  ¬(∀ f, has_only_zero_in_open_interval f → ∃ x, 2 < x ∧ x < 5 ∧ f x = 0) :=
sorry

end NUMINAMATH_CALUDE_zero_not_necessarily_in_2_5_l3639_363997


namespace NUMINAMATH_CALUDE_max_value_cyclic_expression_l3639_363972

theorem max_value_cyclic_expression (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 27/8 ∧ 
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 3 ∧
    (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) = 27/8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cyclic_expression_l3639_363972


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3639_363905

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  q > 1 →  -- common ratio > 1
  4 * (a 2005)^2 - 8 * (a 2005) + 3 = 0 →  -- a₂₀₀₅ is a root
  4 * (a 2006)^2 - 8 * (a 2006) + 3 = 0 →  -- a₂₀₀₆ is a root
  a 2007 + a 2008 = 18 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3639_363905


namespace NUMINAMATH_CALUDE_download_speed_scientific_notation_l3639_363910

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The theoretical download speed of the Huawei phone MateX on a 5G network in B/s -/
def download_speed : ℕ := 603000000

/-- Converts a natural number to its scientific notation representation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem download_speed_scientific_notation :
  to_scientific_notation download_speed = ScientificNotation.mk 6.03 8 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_download_speed_scientific_notation_l3639_363910


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l3639_363954

theorem sqrt_product_simplification :
  Real.sqrt (12 + 1/9) * Real.sqrt 3 = Real.sqrt 327 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l3639_363954


namespace NUMINAMATH_CALUDE_polygon_diagonals_l3639_363983

theorem polygon_diagonals (n : ℕ) (h : n = 150) : (n * (n - 3)) / 2 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l3639_363983


namespace NUMINAMATH_CALUDE_exists_polygon_different_centers_l3639_363941

/-- A polygon is represented by a list of its vertices --/
def Polygon := List (ℝ × ℝ)

/-- Calculate the center of gravity of a polygon's vertices --/
noncomputable def centerOfGravityVertices (p : Polygon) : ℝ × ℝ := sorry

/-- Calculate the center of gravity of a polygon plate --/
noncomputable def centerOfGravityPlate (p : Polygon) : ℝ × ℝ := sorry

/-- The theorem stating that there exists a polygon where the centers of gravity don't coincide --/
theorem exists_polygon_different_centers : 
  ∃ (p : Polygon), centerOfGravityVertices p ≠ centerOfGravityPlate p := by sorry

end NUMINAMATH_CALUDE_exists_polygon_different_centers_l3639_363941


namespace NUMINAMATH_CALUDE_replaced_person_weight_l3639_363944

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (num_persons : ℕ) (avg_weight_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  new_person_weight - (num_persons * avg_weight_increase)

/-- Theorem stating the weight of the replaced person under the given conditions -/
theorem replaced_person_weight :
  weight_of_replaced_person 5 10.0 90 = 40.0 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l3639_363944


namespace NUMINAMATH_CALUDE_train_passing_time_l3639_363932

/-- Proves that the time for train A to pass train B is 7.5 seconds given the conditions -/
theorem train_passing_time (length_A length_B : ℝ) (time_B_passes_A : ℝ) 
  (h1 : length_A = 150)
  (h2 : length_B = 200)
  (h3 : time_B_passes_A = 10) :
  (length_A / (length_B / time_B_passes_A)) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l3639_363932


namespace NUMINAMATH_CALUDE_salary_percent_increase_l3639_363900

def salary_increase : ℝ := 5000
def new_salary : ℝ := 25000

theorem salary_percent_increase :
  let original_salary := new_salary - salary_increase
  let percent_increase := (salary_increase / original_salary) * 100
  percent_increase = 25 := by
sorry

end NUMINAMATH_CALUDE_salary_percent_increase_l3639_363900


namespace NUMINAMATH_CALUDE_not_divisible_by_121_l3639_363904

theorem not_divisible_by_121 (n : ℤ) : ¬(121 ∣ (n^2 + 3*n + 5)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_121_l3639_363904


namespace NUMINAMATH_CALUDE_almond_walnut_ratio_is_five_to_two_l3639_363922

/-- Represents a mixture of nuts with almonds and walnuts. -/
structure NutMixture where
  total_weight : ℝ
  almond_weight : ℝ
  almond_parts : ℝ
  walnut_parts : ℝ

/-- The ratio of almonds to walnuts in the mixture. -/
def almond_walnut_ratio (mix : NutMixture) : ℝ × ℝ :=
  (mix.almond_parts, mix.walnut_parts)

theorem almond_walnut_ratio_is_five_to_two
  (mix : NutMixture)
  (h1 : mix.total_weight = 350)
  (h2 : mix.almond_weight = 250)
  (h3 : mix.walnut_parts = 2)
  (h4 : mix.almond_parts * mix.walnut_parts = mix.almond_weight * mix.walnut_parts) :
  almond_walnut_ratio mix = (5, 2) := by
  sorry

end NUMINAMATH_CALUDE_almond_walnut_ratio_is_five_to_two_l3639_363922


namespace NUMINAMATH_CALUDE_residential_ratio_is_half_l3639_363948

/-- Represents a building with residential, office, and restaurant units. -/
structure Building where
  total_units : ℕ
  restaurant_units : ℕ
  office_units : ℕ
  residential_units : ℕ

/-- The ratio of residential units to total units in a building. -/
def residential_ratio (b : Building) : ℚ :=
  b.residential_units / b.total_units

/-- Theorem stating the residential ratio for a specific building configuration. -/
theorem residential_ratio_is_half (b : Building) 
    (h1 : b.total_units = 300)
    (h2 : b.restaurant_units = 75)
    (h3 : b.office_units = b.restaurant_units)
    (h4 : b.residential_units = b.total_units - (b.restaurant_units + b.office_units)) :
    residential_ratio b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_residential_ratio_is_half_l3639_363948


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3639_363986

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_a5 : a 5 = 2)
  (h_a9 : a 9 = 32) :
  a 4 * a 10 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3639_363986


namespace NUMINAMATH_CALUDE_judes_current_age_jude_is_two_years_old_l3639_363967

/-- Proves Jude's current age given Heath's current age and their future age relationship -/
theorem judes_current_age (heath_current_age : ℕ) (future_years : ℕ) (future_age_ratio : ℕ) : ℕ :=
  let heath_future_age := heath_current_age + future_years
  let jude_future_age := heath_future_age / future_age_ratio
  let age_difference := heath_future_age - jude_future_age
  heath_current_age - age_difference

/-- The main theorem that proves Jude's current age is 2 years old -/
theorem jude_is_two_years_old : judes_current_age 16 5 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_judes_current_age_jude_is_two_years_old_l3639_363967


namespace NUMINAMATH_CALUDE_ten_satisfies_divisor_condition_l3639_363960

theorem ten_satisfies_divisor_condition : ∃ (n : ℕ), 
  n > 1 ∧ 
  (∀ (d : ℕ), d > 1 → d ∣ n → ∃ (a r : ℕ), r > 1 ∧ d = a^r + 1) ∧
  n = 10 :=
by sorry

end NUMINAMATH_CALUDE_ten_satisfies_divisor_condition_l3639_363960


namespace NUMINAMATH_CALUDE_dividend_calculation_l3639_363940

/-- The dividend calculation problem -/
theorem dividend_calculation (divisor quotient remainder : ℝ) 
  (h_divisor : divisor = 176.22471910112358)
  (h_quotient : quotient = 89)
  (h_remainder : remainder = 14) :
  divisor * quotient + remainder = 15697.799999999998 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3639_363940


namespace NUMINAMATH_CALUDE_pentagon_lcm_problem_l3639_363916

/-- Given five distinct natural numbers on the vertices of a pentagon,
    if the LCM of each pair of adjacent numbers is the same for all sides,
    then the smallest possible value for this common LCM is 30. -/
theorem pentagon_lcm_problem (a b c d e : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e →
  ∃ L : ℕ, L > 0 ∧
    Nat.lcm a b = L ∧
    Nat.lcm b c = L ∧
    Nat.lcm c d = L ∧
    Nat.lcm d e = L ∧
    Nat.lcm e a = L →
  (∀ M : ℕ, M > 0 ∧
    (∃ x y z w v : ℕ, x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧ y ≠ z ∧ y ≠ w ∧ y ≠ v ∧ z ≠ w ∧ z ≠ v ∧ w ≠ v ∧
      Nat.lcm x y = M ∧
      Nat.lcm y z = M ∧
      Nat.lcm z w = M ∧
      Nat.lcm w v = M ∧
      Nat.lcm v x = M) →
    M ≥ 30) :=
by sorry

end NUMINAMATH_CALUDE_pentagon_lcm_problem_l3639_363916


namespace NUMINAMATH_CALUDE_x_minus_y_equals_negative_twelve_l3639_363980

theorem x_minus_y_equals_negative_twelve (x y : ℝ) 
  (hx : 2 = 0.25 * x) (hy : 2 = 0.10 * y) : x - y = -12 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_negative_twelve_l3639_363980


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l3639_363991

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x > 5}

-- State the theorem
theorem intersection_of_A_and_complement_of_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 2 < x ∧ x ≤ 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l3639_363991


namespace NUMINAMATH_CALUDE_sculpture_height_proof_l3639_363913

/-- The height of the sculpture in inches -/
def sculpture_height : ℝ := 34

/-- The height of the base in inches -/
def base_height : ℝ := 4

/-- The combined height of the sculpture and base in feet -/
def total_height_feet : ℝ := 3.1666666666666665

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

theorem sculpture_height_proof :
  sculpture_height = total_height_feet * feet_to_inches - base_height :=
by sorry

end NUMINAMATH_CALUDE_sculpture_height_proof_l3639_363913


namespace NUMINAMATH_CALUDE_tim_final_soda_cans_l3639_363982

/-- Calculates the final number of soda cans Tim has -/
def final_soda_cans (initial : ℕ) (taken : ℕ) : ℕ :=
  let remaining := initial - taken
  let bought := remaining / 2
  remaining + bought

/-- Proves that Tim ends up with 24 cans of soda given the initial conditions -/
theorem tim_final_soda_cans : final_soda_cans 22 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_tim_final_soda_cans_l3639_363982


namespace NUMINAMATH_CALUDE_lcm_of_12_18_24_l3639_363968

theorem lcm_of_12_18_24 : Nat.lcm 12 (Nat.lcm 18 24) = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_18_24_l3639_363968


namespace NUMINAMATH_CALUDE_least_non_factor_non_prime_l3639_363981

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem least_non_factor_non_prime : 
  ∃ (n : ℕ), n > 0 ∧ ¬(factorial 30 % n = 0) ∧ ¬(is_prime n) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → (factorial 30 % m = 0) ∨ (is_prime m)) ∧ n = 961 := by
  sorry

end NUMINAMATH_CALUDE_least_non_factor_non_prime_l3639_363981


namespace NUMINAMATH_CALUDE_peanut_butter_cookie_probability_l3639_363951

/-- The probability of selecting a peanut butter cookie -/
def peanut_butter_probability (peanut_butter_cookies : ℕ) (chocolate_chip_cookies : ℕ) (lemon_cookies : ℕ) : ℚ :=
  peanut_butter_cookies / (peanut_butter_cookies + chocolate_chip_cookies + lemon_cookies)

theorem peanut_butter_cookie_probability :
  peanut_butter_probability 70 50 20 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_cookie_probability_l3639_363951


namespace NUMINAMATH_CALUDE_farmland_zones_properties_l3639_363945

/-- Represents the farmland areas and drone spraying capabilities in two zones -/
structure FarmlandZones where
  zone_a : ℝ  -- Farmland area in Zone A
  zone_b : ℝ  -- Farmland area in Zone B
  spray_a : ℝ  -- Average spray area per sortie in Zone A
  spray_b : ℝ  -- Average spray area per sortie in Zone B

/-- Theorem stating the properties of the farmland zones based on given conditions -/
theorem farmland_zones_properties :
  ∀ (fz : FarmlandZones),
  fz.zone_a = fz.zone_b + 10000 →  -- Condition 1
  0.8 * fz.zone_a = fz.zone_b →  -- Conditions 2 and 3 combined
  (fz.zone_b / fz.spray_b) = 1.2 * (fz.zone_a / fz.spray_a) →  -- Condition 4
  fz.spray_a = fz.spray_b + 50 / 3 →  -- Condition 5
  fz.zone_a = 50000 ∧ fz.zone_b = 40000 ∧ fz.spray_a = 100 := by
  sorry  -- Proof omitted


end NUMINAMATH_CALUDE_farmland_zones_properties_l3639_363945


namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l3639_363989

/-- Proves that the initial number of bananas per child is 2 --/
theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) : 
  total_children = 780 →
  absent_children = 390 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ), 
    (total_children - absent_children) * (initial_bananas + extra_bananas) = total_children * initial_bananas ∧
    initial_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_per_child_l3639_363989


namespace NUMINAMATH_CALUDE_fifteenth_term_of_geometric_sequence_l3639_363955

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometricSequenceTerm (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

theorem fifteenth_term_of_geometric_sequence :
  let a := 5
  let r := (1/2 : ℚ)
  let n := 15
  geometricSequenceTerm a r n = 5/16384 := by
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_geometric_sequence_l3639_363955


namespace NUMINAMATH_CALUDE_even_sum_difference_l3639_363939

def sum_even_range (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  n * (a + b) / 2

theorem even_sum_difference : sum_even_range 102 150 - sum_even_range 2 50 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_difference_l3639_363939


namespace NUMINAMATH_CALUDE_sector_area_l3639_363921

theorem sector_area (r : ℝ) (α : ℝ) (h1 : r = 3) (h2 : α = 2) :
  (1 / 2 : ℝ) * r^2 * α = 9 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3639_363921


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3639_363908

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

/-- The problem statement -/
theorem symmetric_points_sum (m n : ℝ) 
  (h : symmetric_wrt_origin (-3, m) (n, 2)) : 
  m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3639_363908


namespace NUMINAMATH_CALUDE_symmetry_across_y_eq_neg_x_l3639_363985

/-- Given two lines in the xy-plane, this function checks if they are symmetrical across y = -x -/
def are_symmetrical_lines (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), line1 x y ↔ line2 y x

/-- The original line: √3x + y + 1 = 0 -/
def original_line (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y + 1 = 0

/-- The proposed symmetrical line: x + √3y - 1 = 0 -/
def symmetrical_line (x y : ℝ) : Prop :=
  x + Real.sqrt 3 * y - 1 = 0

/-- Theorem stating that the symmetrical_line is indeed symmetrical to the original_line across y = -x -/
theorem symmetry_across_y_eq_neg_x :
  are_symmetrical_lines original_line symmetrical_line :=
sorry

end NUMINAMATH_CALUDE_symmetry_across_y_eq_neg_x_l3639_363985


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3639_363902

structure Plane
structure Line

-- Define the perpendicular relationship between planes
def perp_planes (α β : Plane) : Prop := sorry

-- Define the intersection of two planes
def intersect_planes (α β : Plane) : Line := sorry

-- Define a line parallel to a plane
def parallel_line_plane (a : Line) (α : Plane) : Prop := sorry

-- Define a line perpendicular to a plane
def perp_line_plane (b : Line) (β : Plane) : Prop := sorry

-- Define a line perpendicular to another line
def perp_lines (b l : Line) : Prop := sorry

theorem perpendicular_lines_from_perpendicular_planes 
  (α β : Plane) (a b l : Line) 
  (h1 : perp_planes α β) 
  (h2 : intersect_planes α β = l) 
  (h3 : parallel_line_plane a α) 
  (h4 : perp_line_plane b β) : 
  perp_lines b l := sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3639_363902


namespace NUMINAMATH_CALUDE_haley_final_lives_l3639_363962

/-- Calculates the final number of lives in a video game scenario. -/
def final_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Proves that for the given scenario, the final number of lives is 46. -/
theorem haley_final_lives : final_lives 14 4 36 = 46 := by
  sorry

end NUMINAMATH_CALUDE_haley_final_lives_l3639_363962


namespace NUMINAMATH_CALUDE_decimal_89_to_binary_l3639_363936

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_89_to_binary :
  decimal_to_binary 89 = [1, 0, 1, 1, 0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_decimal_89_to_binary_l3639_363936


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3639_363950

theorem complex_fraction_sum (a b : ℝ) : 
  (1 + 2*I) / (1 + I) = a + b*I → a + b = 2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3639_363950


namespace NUMINAMATH_CALUDE_first_car_departure_time_l3639_363999

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv : minutes < 60

/-- Represents a car with its speed -/
structure Car where
  speed : ℝ  -- speed in miles per hour

def problem (first_car : Car) (second_car : Car) (trip_distance : ℝ) (time_difference : ℝ) (meeting_time : Time) : Prop :=
  first_car.speed = 30 ∧
  second_car.speed = 60 ∧
  trip_distance = 80 ∧
  time_difference = 1/6 ∧  -- 10 minutes in hours
  meeting_time.hours = 10 ∧
  meeting_time.minutes = 30

theorem first_car_departure_time 
  (first_car : Car) (second_car : Car) (trip_distance : ℝ) (time_difference : ℝ) (meeting_time : Time) :
  problem first_car second_car trip_distance time_difference meeting_time →
  ∃ (departure_time : Time), 
    departure_time.hours = 10 ∧ departure_time.minutes = 10 :=
sorry

end NUMINAMATH_CALUDE_first_car_departure_time_l3639_363999


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_parallel_l3639_363901

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- Given vectors a = (m,1) and b = (n,1), m/n = 1 is a sufficient but not necessary condition for a ∥ b -/
theorem sufficient_not_necessary_parallel (m n : ℝ) :
  (m / n = 1 → parallel (m, 1) (n, 1)) ∧
  ¬(parallel (m, 1) (n, 1) → m / n = 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_parallel_l3639_363901


namespace NUMINAMATH_CALUDE_sum_of_b_values_l3639_363914

/-- The sum of the two values of b for which the equation 3x^2 + bx + 6x + 7 = 0 has only one solution for x -/
theorem sum_of_b_values (b₁ b₂ : ℝ) : 
  (∃! x, 3 * x^2 + b₁ * x + 6 * x + 7 = 0) →
  (∃! x, 3 * x^2 + b₂ * x + 6 * x + 7 = 0) →
  b₁ + b₂ = -12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_b_values_l3639_363914


namespace NUMINAMATH_CALUDE_dans_minimum_spending_l3639_363978

/-- Given Dan's purchases and spending information, prove he spent at least $9 -/
theorem dans_minimum_spending (chocolate_cost candy_cost difference : ℕ) 
  (h1 : chocolate_cost = 7)
  (h2 : candy_cost = 2)
  (h3 : chocolate_cost = candy_cost + difference)
  (h4 : difference = 5) : 
  chocolate_cost + candy_cost ≥ 9 := by
  sorry

#check dans_minimum_spending

end NUMINAMATH_CALUDE_dans_minimum_spending_l3639_363978


namespace NUMINAMATH_CALUDE_bobs_smallest_number_l3639_363996

def is_valid_bob_number (alice_num bob_num : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → p ∣ alice_num → p ∣ bob_num

def has_additional_prime_factor (alice_num bob_num : ℕ) : Prop :=
  ∃ q : ℕ, q.Prime ∧ q ∣ bob_num ∧ ¬(q ∣ alice_num)

theorem bobs_smallest_number (alice_num : ℕ) (bob_num : ℕ) :
  alice_num = 36 →
  is_valid_bob_number alice_num bob_num →
  has_additional_prime_factor alice_num bob_num →
  (∀ n : ℕ, n < bob_num →
    ¬(is_valid_bob_number alice_num n ∧ has_additional_prime_factor alice_num n)) →
  bob_num = 30 :=
sorry

end NUMINAMATH_CALUDE_bobs_smallest_number_l3639_363996


namespace NUMINAMATH_CALUDE_parabola_symmetry_l3639_363920

theorem parabola_symmetry (x₁ x₂ y₁ y₂ m : ℝ) : 
  y₁ = 2 * x₁^2 →
  y₂ = 2 * x₂^2 →
  y₁ + y₂ = x₁ + x₂ + 2*m →
  x₁ * x₂ = -1/2 →
  m = 3/2 := by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l3639_363920


namespace NUMINAMATH_CALUDE_fraction_equality_l3639_363961

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) 
  (h2 : (2 * a) / (3 * b) + (a + 12 * b) / (3 * b + 12 * a) = 5 / 3) : 
  a / b = -93 / 49 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3639_363961


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3639_363974

/-- A geometric sequence with first term a and common ratio q -/
def geometricSequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

/-- A sequence is monotonically increasing -/
def MonotonicallyIncreasing (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, s n ≤ s (n + 1)

/-- The condition q > 1 is neither necessary nor sufficient for a geometric sequence to be monotonically increasing -/
theorem geometric_sequence_increasing_condition (a q : ℝ) :
  ¬(((q > 1) ↔ MonotonicallyIncreasing (geometricSequence a q))) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3639_363974


namespace NUMINAMATH_CALUDE_roots_difference_squared_l3639_363970

theorem roots_difference_squared (α β : ℝ) : 
  α^2 - 3*α + 2 = 0 → β^2 - 3*β + 2 = 0 → α ≠ β → (α - β)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_difference_squared_l3639_363970


namespace NUMINAMATH_CALUDE_probability_n_power_16_mod_6_equals_1_l3639_363998

theorem probability_n_power_16_mod_6_equals_1 (N : ℕ) (h : 1 ≤ N ∧ N ≤ 2000) :
  (Nat.card {n : ℕ | 1 ≤ n ∧ n ≤ 2000 ∧ n^16 % 6 = 1}) / 2000 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_n_power_16_mod_6_equals_1_l3639_363998


namespace NUMINAMATH_CALUDE_age_problem_l3639_363975

theorem age_problem (age1 age2 : ℕ) : 
  age1 + age2 = 63 →
  age1 = 2 * (age2 - (age1 - age2)) →
  (age1 = 36 ∧ age2 = 27) :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l3639_363975


namespace NUMINAMATH_CALUDE_sequence_solution_l3639_363965

def sequence_problem (b : Fin 6 → ℝ) : Prop :=
  (∀ n : Fin 3, b (2 * n) = b (2 * n - 1) ^ 2) ∧
  (∀ n : Fin 2, b (2 * n + 1) = (b (2 * n) * b (2 * n - 1)) ^ 2) ∧
  b 6 = 65536 ∧ b 5 = 256 ∧ b 4 = 16 ∧
  (∀ i : Fin 6, 0 ≤ b i)

theorem sequence_solution (b : Fin 6 → ℝ) (h : sequence_problem b) : b 1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_solution_l3639_363965


namespace NUMINAMATH_CALUDE_yellow_green_difference_l3639_363917

/-- The number of buttons purchased by a tailor -/
def total_buttons : ℕ := 275

/-- The number of green buttons purchased -/
def green_buttons : ℕ := 90

/-- The number of blue buttons purchased -/
def blue_buttons : ℕ := green_buttons - 5

/-- The number of yellow buttons purchased -/
def yellow_buttons : ℕ := total_buttons - green_buttons - blue_buttons

/-- Theorem stating the difference between yellow and green buttons -/
theorem yellow_green_difference : 
  yellow_buttons - green_buttons = 10 := by sorry

end NUMINAMATH_CALUDE_yellow_green_difference_l3639_363917


namespace NUMINAMATH_CALUDE_curve_E_and_min_distance_l3639_363925

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 5 = 0

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 91 = 0

/-- Definition of curve E as the locus of centers of moving circles -/
def E (x y : ℝ) : Prop := ∃ (r : ℝ), 
  (∀ (x₁ y₁ : ℝ), C₁ x₁ y₁ → (x - x₁)^2 + (y - y₁)^2 = (r + 2)^2) ∧
  (∀ (x₂ y₂ : ℝ), C₂ x₂ y₂ → (x - x₂)^2 + (y - y₂)^2 = (10 - r)^2)

/-- The right focus of curve E -/
def F : ℝ × ℝ := (3, 0)

/-- Theorem stating the equation of curve E and the minimum value of |PO|²+|PF|² -/
theorem curve_E_and_min_distance : 
  (∀ x y : ℝ, E x y ↔ x^2/36 + y^2/27 = 1) ∧
  (∃ min : ℝ, min = 45 ∧ 
    ∀ x y : ℝ, E x y → x^2 + y^2 + (x - F.1)^2 + (y - F.2)^2 ≥ min) :=
sorry

end NUMINAMATH_CALUDE_curve_E_and_min_distance_l3639_363925


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l3639_363966

-- Define the slopes of two lines
def slope1 (k : ℝ) := k
def slope2 : ℝ := 2

-- Define the condition for perpendicular lines
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_slope (k : ℝ) :
  perpendicular (slope1 k) slope2 → k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l3639_363966


namespace NUMINAMATH_CALUDE_video_game_lives_l3639_363927

/-- Calculates the total lives after completing all levels in a video game -/
def total_lives (initial : ℝ) (hard_part : ℝ) (next_level : ℝ) (extra_challenge1 : ℝ) (extra_challenge2 : ℝ) : ℝ :=
  initial + hard_part + next_level + extra_challenge1 + extra_challenge2

/-- Theorem stating that the total lives after completing all levels is 261.0 -/
theorem video_game_lives :
  let initial : ℝ := 143.0
  let hard_part : ℝ := 14.0
  let next_level : ℝ := 27.0
  let extra_challenge1 : ℝ := 35.0
  let extra_challenge2 : ℝ := 42.0
  total_lives initial hard_part next_level extra_challenge1 extra_challenge2 = 261.0 := by
  sorry


end NUMINAMATH_CALUDE_video_game_lives_l3639_363927


namespace NUMINAMATH_CALUDE_mailing_cost_calculation_l3639_363952

/-- Calculates the total cost of mailing letters and packages -/
def total_mailing_cost (letter_cost package_cost : ℚ) (num_letters : ℕ) : ℚ :=
  let num_packages := num_letters - 2
  letter_cost * num_letters + package_cost * num_packages

/-- Theorem: Given the conditions, the total mailing cost is $4.49 -/
theorem mailing_cost_calculation :
  total_mailing_cost (37/100) (88/100) 5 = 449/100 := by
sorry

end NUMINAMATH_CALUDE_mailing_cost_calculation_l3639_363952


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l3639_363963

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
def circle2 (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

-- Define the line
def line (x y : ℝ) : Prop := 2*x + y = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l3639_363963


namespace NUMINAMATH_CALUDE_amy_albums_count_l3639_363947

/-- The number of photos Amy uploaded to Facebook -/
def total_photos : ℕ := 180

/-- The number of photos in each album -/
def photos_per_album : ℕ := 20

/-- The number of albums Amy created -/
def num_albums : ℕ := total_photos / photos_per_album

theorem amy_albums_count : num_albums = 9 := by
  sorry

end NUMINAMATH_CALUDE_amy_albums_count_l3639_363947


namespace NUMINAMATH_CALUDE_complex_quotient_real_l3639_363990

theorem complex_quotient_real (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (∃ (r : ℝ), z₁ / z₂ = r) → a = -3/2 := by sorry

end NUMINAMATH_CALUDE_complex_quotient_real_l3639_363990


namespace NUMINAMATH_CALUDE_cos_neg_nineteen_pi_sixths_l3639_363915

theorem cos_neg_nineteen_pi_sixths :
  Real.cos (-19 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_nineteen_pi_sixths_l3639_363915


namespace NUMINAMATH_CALUDE_bicycle_shop_inventory_l3639_363923

/-- Represents the bicycle shop inventory problem --/
theorem bicycle_shop_inventory
  (initial_stock : ℕ)
  (weekly_addition : ℕ)
  (weeks : ℕ)
  (final_stock : ℕ)
  (h1 : initial_stock = 51)
  (h2 : weekly_addition = 3)
  (h3 : weeks = 4)
  (h4 : final_stock = 45) :
  initial_stock + weekly_addition * weeks - final_stock = 18 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_shop_inventory_l3639_363923


namespace NUMINAMATH_CALUDE_sqrt_fraction_equals_half_l3639_363918

theorem sqrt_fraction_equals_half : 
  Real.sqrt ((16^6 + 8^8) / (16^3 + 8^9)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equals_half_l3639_363918


namespace NUMINAMATH_CALUDE_counterexample_exists_l3639_363911

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem counterexample_exists : ∃ n : ℕ, 
  (sum_of_digits n) % 9 = 0 ∧ 
  n % 3 = 0 ∧ 
  n % 9 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3639_363911


namespace NUMINAMATH_CALUDE_min_teachers_for_school_l3639_363992

/-- Represents the number of teachers for each subject -/
structure TeacherCounts where
  english : ℕ
  history : ℕ
  geography : ℕ

/-- Calculates the minimum number of teachers required -/
def min_teachers (counts : TeacherCounts) : ℕ :=
  let geography_and_english := min counts.geography counts.english
  let remaining_english := counts.english - geography_and_english
  let history_and_english := min remaining_english counts.history
  let remaining_history := counts.history - history_and_english
  geography_and_english + remaining_history

/-- Theorem stating the minimum number of teachers required -/
theorem min_teachers_for_school (counts : TeacherCounts) 
  (h1 : counts.english = 9)
  (h2 : counts.history = 7)
  (h3 : counts.geography = 6) :
  min_teachers counts = 10 := by
  sorry

#eval min_teachers { english := 9, history := 7, geography := 6 }

end NUMINAMATH_CALUDE_min_teachers_for_school_l3639_363992


namespace NUMINAMATH_CALUDE_johns_dancing_time_l3639_363912

theorem johns_dancing_time (john_initial : ℝ) (john_after : ℝ) (james : ℝ) 
  (h1 : john_after = 5)
  (h2 : james = john_initial + 1 + john_after + (1/3) * (john_initial + 1 + john_after))
  (h3 : john_initial + john_after + james = 20) :
  john_initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_johns_dancing_time_l3639_363912


namespace NUMINAMATH_CALUDE_theatre_sales_calculation_l3639_363976

/-- Calculates the total sales amount for a theatre performance given ticket prices and quantities sold. -/
theorem theatre_sales_calculation 
  (price1 price2 : ℚ) 
  (total_tickets sold1 : ℕ) 
  (h1 : price1 = 4.5)
  (h2 : price2 = 6)
  (h3 : total_tickets = 380)
  (h4 : sold1 = 205) :
  price1 * sold1 + price2 * (total_tickets - sold1) = 1972.5 :=
by sorry

end NUMINAMATH_CALUDE_theatre_sales_calculation_l3639_363976


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l3639_363938

/-- For any positive real number a, the function f(x) = 2 + a^(x-1) always passes through the point (1, 3) -/
theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ 2 + a^(x - 1)
  f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l3639_363938


namespace NUMINAMATH_CALUDE_blue_pens_count_l3639_363964

theorem blue_pens_count (total_pens : ℕ) (red_pens : ℕ) (black_pens : ℕ) (blue_pens : ℕ) 
  (h1 : total_pens = 31)
  (h2 : total_pens = red_pens + black_pens + blue_pens)
  (h3 : black_pens = red_pens + 5)
  (h4 : blue_pens = 2 * black_pens) :
  blue_pens = 18 := by
  sorry

end NUMINAMATH_CALUDE_blue_pens_count_l3639_363964


namespace NUMINAMATH_CALUDE_inequality_proof_l3639_363937

theorem inequality_proof (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b * c > 0) :
  a * b + b * c + c * a < Real.sqrt (a * b * c) / 2 + 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3639_363937


namespace NUMINAMATH_CALUDE_inequality_relation_l3639_363959

theorem inequality_relation (x y : ℝ) :
  (x^3 + x > x^2*y + y) → (x - y > -1) ∧
  ¬(∀ x y : ℝ, x - y > -1 → x^3 + x > x^2*y + y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relation_l3639_363959


namespace NUMINAMATH_CALUDE_light_glows_165_times_l3639_363907

/-- Represents the glow pattern of the light in seconds -/
def glowPattern : List Nat := [15, 25, 35, 45]

/-- Calculates the total seconds in the glow pattern -/
def patternDuration : Nat := glowPattern.sum

/-- Converts a time in hours, minutes, seconds to total seconds -/
def timeToSeconds (hours minutes seconds : Nat) : Nat :=
  hours * 3600 + minutes * 60 + seconds

/-- Calculates the duration between two times in seconds -/
def durationBetween (startHours startMinutes startSeconds endHours endMinutes endSeconds : Nat) : Nat :=
  timeToSeconds endHours endMinutes endSeconds - timeToSeconds startHours startMinutes startSeconds

/-- Calculates the number of complete cycles in a given duration -/
def completeCycles (duration : Nat) : Nat :=
  duration / patternDuration

/-- Calculates the remaining seconds after complete cycles -/
def remainingSeconds (duration : Nat) : Nat :=
  duration % patternDuration

/-- Counts the number of glows in the remaining seconds -/
def countRemainingGlows (seconds : Nat) : Nat :=
  glowPattern.foldl (fun count interval => if seconds ≥ interval then count + 1 else count) 0

/-- Theorem: The light glows 165 times between 1:57:58 AM and 3:20:47 AM -/
theorem light_glows_165_times : 
  (completeCycles (durationBetween 1 57 58 3 20 47) * glowPattern.length) + 
  countRemainingGlows (remainingSeconds (durationBetween 1 57 58 3 20 47)) = 165 := by
  sorry


end NUMINAMATH_CALUDE_light_glows_165_times_l3639_363907


namespace NUMINAMATH_CALUDE_vector_operations_l3639_363930

def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (-1, 3)

theorem vector_operations :
  (a.1 + b.1, a.2 + b.2) = (1, 3) ∧
  (a.1 - b.1, a.2 - b.2) = (3, -3) := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l3639_363930


namespace NUMINAMATH_CALUDE_line_xz_plane_intersection_l3639_363977

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- The xz-plane -/
def xzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Check if a point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p.x = l.p1.x + t * (l.p2.x - l.p1.x) ∧
            p.y = l.p1.y + t * (l.p2.y - l.p1.y) ∧
            p.z = l.p1.z + t * (l.p2.z - l.p1.z)

theorem line_xz_plane_intersection :
  let l : Line3D := { p1 := ⟨2, -1, 3⟩, p2 := ⟨6, 7, -2⟩ }
  let p : Point3D := ⟨2.5, 0, 2.375⟩
  pointOnLine p l ∧ p ∈ xzPlane :=
by sorry

end NUMINAMATH_CALUDE_line_xz_plane_intersection_l3639_363977


namespace NUMINAMATH_CALUDE_employee_pay_l3639_363906

theorem employee_pay (total_pay : ℚ) (a_pay : ℚ) (b_pay : ℚ) :
  total_pay = 570 →
  a_pay = 1.5 * b_pay →
  total_pay = a_pay + b_pay →
  b_pay = 228 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_l3639_363906


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l3639_363958

theorem camping_trip_percentage
  (total_students : ℕ)
  (h1 : (14 : ℚ) / 100 * total_students = (25 : ℚ) / 100 * (56 : ℚ) / 100 * total_students)
  (h2 : (75 : ℚ) / 100 * (56 : ℚ) / 100 * total_students + (14 : ℚ) / 100 * total_students = (56 : ℚ) / 100 * total_students) :
  (56 : ℚ) / 100 * total_students = (56 : ℚ) / 100 * total_students :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l3639_363958


namespace NUMINAMATH_CALUDE_find_x_l3639_363993

theorem find_x : ∃ x : ℝ,
  (24 + 35 + 58) / 3 = ((19 + 51 + x) / 3) + 6 → x = 29 :=
by sorry

end NUMINAMATH_CALUDE_find_x_l3639_363993


namespace NUMINAMATH_CALUDE_family_savings_l3639_363928

def initial_savings : ℕ := 1147240
def income : ℕ := 509600
def expenses : ℕ := 276000

theorem family_savings : initial_savings + income - expenses = 1340840 := by
  sorry

end NUMINAMATH_CALUDE_family_savings_l3639_363928


namespace NUMINAMATH_CALUDE_color_infinite_lines_parallelogram_property_coloring_theorem_l3639_363969

-- Define the color type
inductive Color where
  | White : Color
  | Red : Color
  | Black : Color

-- Define the coloring function
def f : ℤ × ℤ → Color :=
  sorry

-- Condition 1: Each color appears on infinitely many horizontal lines
theorem color_infinite_lines :
  ∀ c : Color, ∃ (s : Set ℤ), Set.Infinite s ∧
    ∀ y ∈ s, ∃ (t : Set ℤ), Set.Infinite t ∧
      ∀ x ∈ t, f (x, y) = c :=
  sorry

-- Condition 2: Parallelogram property
theorem parallelogram_property :
  ∀ a b c : ℤ × ℤ,
    f a = Color.White → f b = Color.Red → f c = Color.Black →
    ∃ d : ℤ × ℤ, f d = Color.Red ∧ a + c = b + d :=
  sorry

-- Main theorem combining both conditions
theorem coloring_theorem :
  ∃ (f : ℤ × ℤ → Color),
    (∀ c : Color, ∃ (s : Set ℤ), Set.Infinite s ∧
      ∀ y ∈ s, ∃ (t : Set ℤ), Set.Infinite t ∧
        ∀ x ∈ t, f (x, y) = c) ∧
    (∀ a b c : ℤ × ℤ,
      f a = Color.White → f b = Color.Red → f c = Color.Black →
      ∃ d : ℤ × ℤ, f d = Color.Red ∧ a + c = b + d) :=
  sorry

end NUMINAMATH_CALUDE_color_infinite_lines_parallelogram_property_coloring_theorem_l3639_363969


namespace NUMINAMATH_CALUDE_water_added_to_bowl_l3639_363943

theorem water_added_to_bowl (C : ℝ) (h1 : C > 0) : 
  (C / 2 + (14 - C / 2) = 0.7 * C) → (14 - C / 2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_bowl_l3639_363943


namespace NUMINAMATH_CALUDE_bookstore_calculator_sales_l3639_363909

theorem bookstore_calculator_sales
  (price1 : ℕ) (price2 : ℕ) (total_sales : ℕ) (quantity2 : ℕ)
  (h1 : price1 = 15)
  (h2 : price2 = 67)
  (h3 : total_sales = 3875)
  (h4 : quantity2 = 35)
  (h5 : ∃ quantity1 : ℕ, price1 * quantity1 + price2 * quantity2 = total_sales) :
  ∃ total_quantity : ℕ, total_quantity = quantity2 + (total_sales - price2 * quantity2) / price1 ∧
                        total_quantity = 137 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_calculator_sales_l3639_363909


namespace NUMINAMATH_CALUDE_clara_weight_l3639_363933

/-- Given two positive real numbers representing weights in pounds,
    prove that one of them (Clara's weight) is equal to 960/7 pounds,
    given the specified conditions. -/
theorem clara_weight (alice_weight clara_weight : ℝ) 
  (h1 : alice_weight > 0)
  (h2 : clara_weight > 0)
  (h3 : alice_weight + clara_weight = 240)
  (h4 : clara_weight - alice_weight = alice_weight / 3) :
  clara_weight = 960 / 7 := by
  sorry

end NUMINAMATH_CALUDE_clara_weight_l3639_363933


namespace NUMINAMATH_CALUDE_max_ladles_l3639_363924

/-- Represents the cost of a pan in dollars -/
def pan_cost : ℕ := 3

/-- Represents the cost of a pot in dollars -/
def pot_cost : ℕ := 5

/-- Represents the cost of a ladle in dollars -/
def ladle_cost : ℕ := 9

/-- Represents the total amount Sarah will spend in dollars -/
def total_spend : ℕ := 100

/-- Represents the minimum number of each item Sarah must buy -/
def min_items : ℕ := 2

theorem max_ladles :
  ∃ (p q l : ℕ),
    p ≥ min_items ∧
    q ≥ min_items ∧
    l ≥ min_items ∧
    pan_cost * p + pot_cost * q + ladle_cost * l = total_spend ∧
    l = 9 ∧
    ∀ (p' q' l' : ℕ),
      p' ≥ min_items →
      q' ≥ min_items →
      l' ≥ min_items →
      pan_cost * p' + pot_cost * q' + ladle_cost * l' = total_spend →
      l' ≤ l :=
by sorry

end NUMINAMATH_CALUDE_max_ladles_l3639_363924


namespace NUMINAMATH_CALUDE_arctan_sum_equation_n_unique_l3639_363929

/-- The positive integer n satisfying the equation arctan(1/2) + arctan(1/3) + arctan(1/7) + arctan(1/n) = π/4 -/
def n : ℕ := 7

/-- The equation that n satisfies -/
theorem arctan_sum_equation : 
  Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/n) = π/4 := by
  sorry

/-- Proof that n is the unique positive integer satisfying the equation -/
theorem n_unique : 
  ∀ m : ℕ, m > 0 → 
  (Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/m) = π/4) → 
  m = n := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_n_unique_l3639_363929


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l3639_363973

/-- The cost price of one meter of cloth given the selling price and profit per meter -/
theorem cost_price_per_meter 
  (total_meters : ℕ) 
  (selling_price : ℚ) 
  (profit_per_meter : ℚ) : 
  total_meters = 80 → 
  selling_price = 6900 → 
  profit_per_meter = 20 → 
  (selling_price - total_meters * profit_per_meter) / total_meters = 66.25 := by
sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l3639_363973


namespace NUMINAMATH_CALUDE_square_equation_solution_l3639_363957

theorem square_equation_solution :
  ∀ x : ℝ, x^2 = 16 ↔ x = -4 ∨ x = 4 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3639_363957


namespace NUMINAMATH_CALUDE_unique_solution_equation_l3639_363953

theorem unique_solution_equation (x : ℝ) : 
  (8^x * (3*x + 1) = 4) ↔ (x = 1/3) := by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l3639_363953


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l3639_363984

theorem dining_bill_calculation (total_spent : ℝ) (tip_rate : ℝ) (tax_rate : ℝ) 
  (h_total : total_spent = 132)
  (h_tip : tip_rate = 0.20)
  (h_tax : tax_rate = 0.10) :
  ∃ (original_price : ℝ),
    original_price = 100 ∧
    total_spent = original_price * (1 + tax_rate) * (1 + tip_rate) := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l3639_363984


namespace NUMINAMATH_CALUDE_stating_policeman_speed_is_10_l3639_363988

/-- Represents the chase scenario between a policeman and a thief -/
structure ChaseScenario where
  initial_distance : ℝ  -- Initial distance in meters
  thief_speed : ℝ       -- Thief's speed in km/hr
  thief_distance : ℝ    -- Distance thief runs before being caught in meters
  policeman_speed : ℝ   -- Policeman's speed in km/hr

/-- 
Theorem stating that given the specific conditions of the chase,
the policeman's speed must be 10 km/hr
-/
theorem policeman_speed_is_10 (chase : ChaseScenario) 
  (h1 : chase.initial_distance = 100)
  (h2 : chase.thief_speed = 8)
  (h3 : chase.thief_distance = 400) :
  chase.policeman_speed = 10 := by
  sorry

#check policeman_speed_is_10

end NUMINAMATH_CALUDE_stating_policeman_speed_is_10_l3639_363988


namespace NUMINAMATH_CALUDE_power_of_product_squared_l3639_363903

theorem power_of_product_squared (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_squared_l3639_363903


namespace NUMINAMATH_CALUDE_triangle_area_l3639_363987

theorem triangle_area (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  a = Real.sqrt 2 → 
  A = π / 4 → 
  B = π / 3 → 
  C = π - A - B →
  S = (1 / 2) * a * b * Real.sin C →
  S = (3 + Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3639_363987


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3639_363995

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 10) :
  (1 / x + 1 / y) ≥ 2 / 5 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 10 ∧ 1 / x + 1 / y = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3639_363995


namespace NUMINAMATH_CALUDE_vector_inequality_l3639_363919

/-- Given vectors u, v, and w in ℝ², prove that w ≠ u - 3v -/
theorem vector_inequality (u v w : ℝ × ℝ) 
  (hu : u = (3, -6)) 
  (hv : v = (4, 2)) 
  (hw : w = (-12, -6)) : 
  w ≠ u - 3 • v := by sorry

end NUMINAMATH_CALUDE_vector_inequality_l3639_363919


namespace NUMINAMATH_CALUDE_complex_2_minus_3i_in_fourth_quadrant_l3639_363935

def is_in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_2_minus_3i_in_fourth_quadrant :
  is_in_fourth_quadrant (2 - 3*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_2_minus_3i_in_fourth_quadrant_l3639_363935
