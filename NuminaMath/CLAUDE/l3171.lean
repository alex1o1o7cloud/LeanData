import Mathlib

namespace NUMINAMATH_CALUDE_integral_x_cubed_plus_one_l3171_317159

theorem integral_x_cubed_plus_one : ∫ x in (-2)..2, (x^3 + 1) = 4 := by sorry

end NUMINAMATH_CALUDE_integral_x_cubed_plus_one_l3171_317159


namespace NUMINAMATH_CALUDE_product_mod_seven_l3171_317171

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026 * 2027) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l3171_317171


namespace NUMINAMATH_CALUDE_root_parity_l3171_317124

theorem root_parity (n : ℤ) (x₁ x₂ : ℤ) : 
  x₁^2 + (4*n + 1)*x₁ + 2*n = 0 ∧ 
  x₂^2 + (4*n + 1)*x₂ + 2*n = 0 → 
  (Odd x₁ ∧ Even x₂) ∨ (Even x₁ ∧ Odd x₂) := by
sorry

end NUMINAMATH_CALUDE_root_parity_l3171_317124


namespace NUMINAMATH_CALUDE_monitor_width_l3171_317140

theorem monitor_width (width height diagonal : ℝ) : 
  width / height = 16 / 9 →
  width ^ 2 + height ^ 2 = diagonal ^ 2 →
  diagonal = 24 →
  width = 384 / Real.sqrt 337 :=
by sorry

end NUMINAMATH_CALUDE_monitor_width_l3171_317140


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l3171_317193

theorem x_squared_mod_25 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 25]) 
  (h2 : 4 * x ≡ 20 [ZMOD 25]) : 
  x^2 ≡ 4 [ZMOD 25] := by
sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l3171_317193


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3171_317196

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, ((a - 8) * x > a - 8) ↔ (x < 1)) → (a < 8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3171_317196


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_sum_of_divisors_450_l3171_317119

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of distinct prime factors of the sum of positive divisors of 450 is 3 -/
theorem distinct_prime_factors_of_sum_of_divisors_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_sum_of_divisors_450_l3171_317119


namespace NUMINAMATH_CALUDE_log_xy_value_l3171_317184

theorem log_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.log (x^3 * y^5) = 2)
  (h2 : Real.log (x^4 * y^2) = 2)
  (h3 : Real.log (x^2 * y^7) = 3) :
  Real.log (x * y) = 4/7 := by
sorry

end NUMINAMATH_CALUDE_log_xy_value_l3171_317184


namespace NUMINAMATH_CALUDE_union_equals_N_l3171_317122

def M : Set ℝ := {x | x^2 - x < 0}
def N : Set ℝ := {x | -3 < x ∧ x < 3}

theorem union_equals_N : M ∪ N = N := by sorry

end NUMINAMATH_CALUDE_union_equals_N_l3171_317122


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3171_317192

/-- Given a triangle DEF where the measure of angle D is 75 degrees,
    and the measure of angle E is 18 degrees more than four times the measure of angle F,
    prove that the measure of angle F is 17.4 degrees. -/
theorem triangle_angle_measure (D E F : ℝ) : 
  D = 75 →
  E = 4 * F + 18 →
  D + E + F = 180 →
  F = 17.4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3171_317192


namespace NUMINAMATH_CALUDE_count_multiples_of_30_l3171_317150

def smallest_square_multiple_of_30 : ℕ := 900
def smallest_fourth_power_multiple_of_30 : ℕ := 810000

theorem count_multiples_of_30 : 
  (smallest_fourth_power_multiple_of_30 / 30) - (smallest_square_multiple_of_30 / 30) + 1 = 26971 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_30_l3171_317150


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3171_317109

/-- Proves that a train with given length and speed takes a specific time to cross a bridge -/
theorem train_bridge_crossing_time
  (train_length : Real)
  (train_speed_kmh : Real)
  (total_length : Real)
  (h1 : train_length = 100)
  (h2 : train_speed_kmh = 45)
  (h3 : total_length = 275) :
  let train_speed_ms : Real := train_speed_kmh * 1000 / 3600
  let bridge_length : Real := total_length - train_length
  let distance_to_cross : Real := train_length + bridge_length
  let time_to_cross : Real := distance_to_cross / train_speed_ms
  time_to_cross = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3171_317109


namespace NUMINAMATH_CALUDE_inequality_solution_l3171_317156

-- Define the function f(x) = 1/√(x+1)
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x + 1)

-- Define the solution set
def solution_set : Set ℝ := Set.Icc 0 (1/2)

-- Define the inequality
def inequality (l k x : ℝ) : Prop := 
  1 - l * x ≤ f x ∧ f x ≤ 1 - k * x

-- Theorem statement
theorem inequality_solution (l k : ℝ) : 
  (∀ x ∈ solution_set, inequality l k x) ↔ (l = 1/2 ∧ k = 2 - 2 * Real.sqrt 6 / 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3171_317156


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3171_317143

/-- Given a point P in polar coordinates, find its symmetric point with respect to the pole -/
theorem symmetric_point_coordinates (r : ℝ) (θ : ℝ) :
  let P : ℝ × ℝ := (r, θ)
  let symmetric_polar : ℝ × ℝ := (r, θ + π)
  let symmetric_cartesian : ℝ × ℝ := (r * Real.cos (θ + π), r * Real.sin (θ + π))
  (P = (2, -5 * π / 3) →
   symmetric_polar = (2, -2 * π / 3) ∧
   symmetric_cartesian = (-1, -Real.sqrt 3)) := by
  sorry

#check symmetric_point_coordinates

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3171_317143


namespace NUMINAMATH_CALUDE_room_length_proof_l3171_317120

theorem room_length_proof (L : ℝ) : 
  L > 0 → -- Ensure length is positive
  ((L + 4) * 16 - L * 12 = 136) → -- Area of veranda equation
  L = 18 := by
sorry

end NUMINAMATH_CALUDE_room_length_proof_l3171_317120


namespace NUMINAMATH_CALUDE_power_sum_l3171_317107

theorem power_sum (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l3171_317107


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_312_l3171_317155

def sum_of_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

theorem sum_of_binary_digits_312 : sum_of_binary_digits 312 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_312_l3171_317155


namespace NUMINAMATH_CALUDE_board_length_proof_l3171_317189

theorem board_length_proof :
  ∀ (short_piece long_piece total_length : ℝ),
  short_piece > 0 →
  long_piece = 2 * short_piece →
  long_piece = 46 →
  total_length = short_piece + long_piece →
  total_length = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_board_length_proof_l3171_317189


namespace NUMINAMATH_CALUDE_octagon_dual_reflection_area_l3171_317176

/-- The area of the region bounded by 8 arcs created by dual reflection over consecutive sides of a regular octagon inscribed in a circle -/
theorem octagon_dual_reflection_area (s : ℝ) (h : s = 2) :
  let r := 1 / Real.sin (22.5 * π / 180)
  let sector_area := π * r^2 / 8
  let dual_reflected_sector_area := 2 * sector_area
  8 * dual_reflected_sector_area = 2 * (1 / Real.sin (22.5 * π / 180))^2 * π :=
by sorry

end NUMINAMATH_CALUDE_octagon_dual_reflection_area_l3171_317176


namespace NUMINAMATH_CALUDE_triangle_area_l3171_317185

/-- The area of a triangle with base 12 cm and height 7 cm is 42 square centimeters. -/
theorem triangle_area : 
  let base : ℝ := 12
  let height : ℝ := 7
  (1 / 2 : ℝ) * base * height = 42 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3171_317185


namespace NUMINAMATH_CALUDE_additional_track_length_l3171_317168

/-- Calculates the additional track length required when changing the grade of a railroad track. -/
theorem additional_track_length
  (elevation : ℝ)
  (initial_grade : ℝ)
  (final_grade : ℝ)
  (h1 : elevation = 1200)
  (h2 : initial_grade = 0.04)
  (h3 : final_grade = 0.03) :
  (elevation / final_grade) - (elevation / initial_grade) = 10000 :=
by sorry

end NUMINAMATH_CALUDE_additional_track_length_l3171_317168


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l3171_317144

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l3171_317144


namespace NUMINAMATH_CALUDE_special_numbers_characterization_l3171_317139

/-- Definition of partial numbers for a natural number -/
def partialNumbers (n : ℕ) : Set ℕ :=
  sorry

/-- Predicate to check if all partial numbers of a natural number are prime -/
def allPartialNumbersPrime (n : ℕ) : Prop :=
  ∀ m ∈ partialNumbers n, Nat.Prime m

/-- The set of natural numbers whose partial numbers are all prime -/
def specialNumbers : Set ℕ :=
  {n : ℕ | allPartialNumbersPrime n}

/-- Theorem stating that the set of natural numbers whose partial numbers
    are all prime is exactly {2, 3, 5, 7, 23, 37, 53, 73} -/
theorem special_numbers_characterization :
  specialNumbers = {2, 3, 5, 7, 23, 37, 53, 73} :=
sorry

end NUMINAMATH_CALUDE_special_numbers_characterization_l3171_317139


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l3171_317165

theorem marble_fraction_after_tripling (total : ℚ) (h1 : total > 0) : 
  let initial_blue := (4/7) * total
  let initial_red := total - initial_blue
  let new_blue := 3 * initial_blue
  let new_total := new_blue + initial_red
  initial_red / new_total = 1/5 := by
sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l3171_317165


namespace NUMINAMATH_CALUDE_undefined_rational_function_l3171_317161

theorem undefined_rational_function (x : ℝ) :
  (x^2 - 24*x + 144 = 0) ↔ (x = 12) :=
by sorry

end NUMINAMATH_CALUDE_undefined_rational_function_l3171_317161


namespace NUMINAMATH_CALUDE_rectangle_division_l3171_317154

theorem rectangle_division (n : ℕ+) 
  (h1 : ∃ a : ℕ+, n = a * a)
  (h2 : ∃ b : ℕ+, n = (n + 98) * b * b) :
  (∃ x y : ℕ+, n = x * y ∧ ((x = 3 ∧ y = 42) ∨ (x = 6 ∧ y = 21) ∨ (x = 24 ∧ y = 48))) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_division_l3171_317154


namespace NUMINAMATH_CALUDE_special_complex_sum_l3171_317167

-- Define the complex function f
def f (z : ℂ) : ℂ := z^2 - 19*z

-- Define the condition for a right triangle
def is_right_triangle (z : ℂ) : Prop :=
  (f z - z) • (f (f z) - f z) = 0

-- Define the structure of z
structure SpecialComplex where
  m : ℕ+
  n : ℕ+
  z : ℂ
  h : z = m + Real.sqrt n + 11*Complex.I

-- State the theorem
theorem special_complex_sum (sc : SpecialComplex) (h : is_right_triangle sc.z) :
  sc.m + sc.n = 230 :=
sorry

end NUMINAMATH_CALUDE_special_complex_sum_l3171_317167


namespace NUMINAMATH_CALUDE_lucky_larry_problem_l3171_317117

theorem lucky_larry_problem (a b c d e : ℤ) : 
  a = 3 → b = 6 → c = 2 → d = 5 →
  (a - b + c - d + e = a - (b + (c - (d - e)))) → e = 3 := by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_problem_l3171_317117


namespace NUMINAMATH_CALUDE_polygon_sides_l3171_317182

theorem polygon_sides (n : ℕ) : n > 2 →
  (n - 2) * 180 = 3 * 360 - 180 →
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l3171_317182


namespace NUMINAMATH_CALUDE_mairead_running_distance_l3171_317121

theorem mairead_running_distance (run walk jog : ℝ) : 
  walk = (3/5) * run → 
  jog = 5 * walk → 
  run + walk + jog = 184 → 
  run = 40 := by
  sorry

end NUMINAMATH_CALUDE_mairead_running_distance_l3171_317121


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l3171_317123

-- Define a random variable following a normal distribution
def normal_distribution (μ : ℝ) (σ : ℝ) (x : ℝ) : ℝ := sorry

-- Define the cumulative distribution function (CDF) for the normal distribution
def normal_cdf (μ : ℝ) (σ : ℝ) (x : ℝ) : ℝ := sorry

theorem normal_distribution_probability 
  (ξ : ℝ → ℝ) -- ξ is a function representing the random variable
  (σ : ℝ) -- standard deviation
  (h1 : σ > 0) -- condition that σ is positive
  (h2 : ∀ x, ξ x = normal_distribution 1 σ x) -- ξ follows N(1, σ²)
  (h3 : normal_cdf 1 σ 1 - normal_cdf 1 σ 0 = 0.4) -- P(0 < ξ < 1) = 0.4
  : normal_cdf 1 σ 2 - normal_cdf 1 σ 0 = 0.8 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l3171_317123


namespace NUMINAMATH_CALUDE_sin_integral_minus_two_to_two_l3171_317188

theorem sin_integral_minus_two_to_two : ∫ x in (-2)..2, Real.sin x = 0 := by sorry

end NUMINAMATH_CALUDE_sin_integral_minus_two_to_two_l3171_317188


namespace NUMINAMATH_CALUDE_largest_square_tile_size_l3171_317126

theorem largest_square_tile_size (length width : ℕ) (h1 : length = 378) (h2 : width = 595) :
  ∃ (tile_size : ℕ), tile_size = Nat.gcd length width ∧ tile_size = 7 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_tile_size_l3171_317126


namespace NUMINAMATH_CALUDE_prime_power_constraints_l3171_317141

theorem prime_power_constraints (a b m n : ℕ) : 
  a > 1 → b > 1 → m > 1 → n > 1 → 
  Nat.Prime (a^n - 1) → Nat.Prime (b^m + 1) → 
  (∃ k : ℕ, m = 2^k) ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_power_constraints_l3171_317141


namespace NUMINAMATH_CALUDE_range_of_k_l3171_317131

theorem range_of_k (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   2*n - 1 < x₁ ∧ x₁ < 2*n + 1 ∧
   2*n - 1 < x₂ ∧ x₂ < 2*n + 1 ∧
   |x₁ - 2*n| = k * Real.sqrt x₁ ∧
   |x₂ - 2*n| = k * Real.sqrt x₂) →
  (0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_l3171_317131


namespace NUMINAMATH_CALUDE_drummer_stick_sets_l3171_317133

/-- Calculates the total number of drum stick sets used by a drummer over multiple nights. -/
theorem drummer_stick_sets (sets_per_show : ℕ) (sets_tossed : ℕ) (nights : ℕ) : 
  sets_per_show = 5 → sets_tossed = 6 → nights = 30 → 
  (sets_per_show + sets_tossed) * nights = 330 := by
  sorry

#check drummer_stick_sets

end NUMINAMATH_CALUDE_drummer_stick_sets_l3171_317133


namespace NUMINAMATH_CALUDE_smallest_value_x_plus_inv_x_l3171_317157

theorem smallest_value_x_plus_inv_x (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ y : ℝ, y = x + 1/x ∧ y ≥ -Real.sqrt 13 ∧ (∀ z : ℝ, z = x + 1/x → z ≥ y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_x_plus_inv_x_l3171_317157


namespace NUMINAMATH_CALUDE_birthday_gift_savings_l3171_317101

/-- Calculates the total amount saved for a mother's birthday gift based on orange sales --/
def total_savings (liam_oranges : ℕ) (liam_price : ℚ) (claire_oranges : ℕ) (claire_price : ℚ)
  (jake_oranges : ℕ) (jake_bundles : ℕ) (jake_price1 : ℚ) (jake_price2 : ℚ) (jake_discount : ℚ) : ℚ :=
  let liam_earnings := (liam_oranges / 2 : ℚ) * liam_price
  let claire_earnings := (claire_oranges : ℚ) * claire_price
  let jake_earnings1 := (jake_bundles / 2 : ℚ) * jake_price1
  let jake_earnings2 := (jake_bundles / 2 : ℚ) * jake_price2
  let jake_total := jake_earnings1 + jake_earnings2
  let jake_discount_amount := jake_total * jake_discount
  let jake_earnings := jake_total - jake_discount_amount
  liam_earnings + claire_earnings + jake_earnings

/-- Theorem stating that the total savings for the mother's birthday gift is $117.88 --/
theorem birthday_gift_savings :
  total_savings 40 (5/2) 30 (6/5) 50 10 3 (9/2) (3/20) = 11788/100 := by
  sorry

end NUMINAMATH_CALUDE_birthday_gift_savings_l3171_317101


namespace NUMINAMATH_CALUDE_cody_purchase_tax_rate_l3171_317172

/-- Proves that the tax rate is 5% given the conditions of Cody's purchase --/
theorem cody_purchase_tax_rate 
  (initial_purchase : ℝ)
  (post_tax_discount : ℝ)
  (cody_payment : ℝ)
  (h1 : initial_purchase = 40)
  (h2 : post_tax_discount = 8)
  (h3 : cody_payment = 17)
  : ∃ (tax_rate : ℝ), 
    tax_rate = 0.05 ∧ 
    (initial_purchase + initial_purchase * tax_rate - post_tax_discount) / 2 = cody_payment :=
by sorry

end NUMINAMATH_CALUDE_cody_purchase_tax_rate_l3171_317172


namespace NUMINAMATH_CALUDE_complement_union_M_N_l3171_317173

-- Define the universe U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) ≠ 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ≠ p.1 + 1}

-- Theorem statement
theorem complement_union_M_N : 
  (U \ (M ∪ N)) = {(2, 3)} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l3171_317173


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l3171_317148

theorem unique_positive_integer_solution :
  ∃! (x : ℕ+), (4 * (x - 1) : ℝ) < 3 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l3171_317148


namespace NUMINAMATH_CALUDE_max_travel_distance_is_3_4_l3171_317164

/-- Represents the taxi fare structure and travel constraints -/
structure TaxiRide where
  initialFare : ℝ
  initialDistance : ℝ
  additionalFarePerUnit : ℝ
  additionalDistanceUnit : ℝ
  tip : ℝ
  totalBudget : ℝ
  timeLimit : ℝ
  averageSpeed : ℝ

/-- Calculates the maximum distance that can be traveled given the taxi fare structure and constraints -/
def maxTravelDistance (ride : TaxiRide) : ℝ :=
  sorry

/-- Theorem stating that the maximum travel distance is approximately 3.4 miles -/
theorem max_travel_distance_is_3_4 (ride : TaxiRide) 
  (h1 : ride.initialFare = 4)
  (h2 : ride.initialDistance = 3/4)
  (h3 : ride.additionalFarePerUnit = 0.3)
  (h4 : ride.additionalDistanceUnit = 0.1)
  (h5 : ride.tip = 3)
  (h6 : ride.totalBudget = 15)
  (h7 : ride.timeLimit = 45/60)
  (h8 : ride.averageSpeed = 30) :
  ∃ ε > 0, abs (maxTravelDistance ride - 3.4) < ε :=
sorry

end NUMINAMATH_CALUDE_max_travel_distance_is_3_4_l3171_317164


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3171_317190

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number (1-m²) + (1+m)i where m is a real number -/
def complexNumber (m : ℝ) : ℂ :=
  ⟨1 - m^2, 1 + m⟩

theorem necessary_but_not_sufficient :
  (∀ m : ℝ, IsPurelyImaginary (complexNumber m) → m = 1 ∨ m = -1) ∧
  (∃ m : ℝ, (m = 1 ∨ m = -1) ∧ ¬IsPurelyImaginary (complexNumber m)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3171_317190


namespace NUMINAMATH_CALUDE_power_sum_simplification_l3171_317183

theorem power_sum_simplification (n : ℕ) : (-3)^n + 2*(-3)^(n-1) = -(-3)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_simplification_l3171_317183


namespace NUMINAMATH_CALUDE_water_depth_multiple_of_height_l3171_317180

theorem water_depth_multiple_of_height (ron_height : ℕ) (water_depth : ℕ) :
  ron_height = 13 →
  water_depth = 208 →
  ∃ k : ℕ, water_depth = k * ron_height →
  water_depth / ron_height = 16 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_multiple_of_height_l3171_317180


namespace NUMINAMATH_CALUDE_qt_squared_eq_three_l3171_317108

-- Define the points
variable (X Y Z W P Q R S T U : ℝ × ℝ)

-- Define the square XYZW
def is_square (X Y Z W : ℝ × ℝ) : Prop := sorry

-- Define that P and S lie on XZ and XW respectively
def on_line (P X Z : ℝ × ℝ) : Prop := sorry
def on_line' (S X W : ℝ × ℝ) : Prop := sorry

-- Define XP = XS = √3
def distance_eq_sqrt3 (X P S : ℝ × ℝ) : Prop := sorry

-- Define Q and R lie on YZ and YW respectively
def on_line'' (Q Y Z : ℝ × ℝ) : Prop := sorry
def on_line''' (R Y W : ℝ × ℝ) : Prop := sorry

-- Define T and U lie on PS
def on_line'''' (T P S : ℝ × ℝ) : Prop := sorry
def on_line''''' (U P S : ℝ × ℝ) : Prop := sorry

-- Define QT ⊥ PS and RU ⊥ PS
def perpendicular (Q T P S : ℝ × ℝ) : Prop := sorry
def perpendicular' (R U P S : ℝ × ℝ) : Prop := sorry

-- Define areas of the shapes
def area_eq_1_5 (X P S : ℝ × ℝ) : Prop := sorry
def area_eq_1_5' (Y Q T P : ℝ × ℝ) : Prop := sorry
def area_eq_1_5'' (W S U R : ℝ × ℝ) : Prop := sorry
def area_eq_1_5''' (Y R U T Q : ℝ × ℝ) : Prop := sorry

-- The theorem to prove
theorem qt_squared_eq_three 
  (h1 : is_square X Y Z W)
  (h2 : on_line P X Z)
  (h3 : on_line' S X W)
  (h4 : distance_eq_sqrt3 X P S)
  (h5 : on_line'' Q Y Z)
  (h6 : on_line''' R Y W)
  (h7 : on_line'''' T P S)
  (h8 : on_line''''' U P S)
  (h9 : perpendicular Q T P S)
  (h10 : perpendicular' R U P S)
  (h11 : area_eq_1_5 X P S)
  (h12 : area_eq_1_5' Y Q T P)
  (h13 : area_eq_1_5'' W S U R)
  (h14 : area_eq_1_5''' Y R U T Q) :
  (Q.1 - T.1)^2 + (Q.2 - T.2)^2 = 3 := by sorry

end NUMINAMATH_CALUDE_qt_squared_eq_three_l3171_317108


namespace NUMINAMATH_CALUDE_population_growth_proof_l3171_317177

theorem population_growth_proof (growth_rate_1 : ℝ) (growth_rate_2 : ℝ) : 
  growth_rate_1 = 0.2 →
  growth_rate_2 = growth_rate_1 + 0.3 * growth_rate_1 →
  (1 + growth_rate_1) * (1 + growth_rate_2) - 1 = 0.512 :=
by
  sorry

#check population_growth_proof

end NUMINAMATH_CALUDE_population_growth_proof_l3171_317177


namespace NUMINAMATH_CALUDE_initial_puppies_count_l3171_317112

/-- The number of puppies Alyssa had initially --/
def initial_puppies : ℕ := 12

/-- The number of puppies Alyssa gave away --/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has left --/
def puppies_left : ℕ := 5

/-- Theorem stating that the initial number of puppies is equal to
    the sum of puppies given away and puppies left --/
theorem initial_puppies_count :
  initial_puppies = puppies_given_away + puppies_left := by
  sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l3171_317112


namespace NUMINAMATH_CALUDE_sarah_mia_games_together_l3171_317160

def total_players : ℕ := 12
def players_per_game : ℕ := 6

theorem sarah_mia_games_together :
  let total_combinations := Nat.choose total_players players_per_game
  let games_per_player := total_combinations / 2
  let other_players := total_players - 2
  games_per_player * (players_per_game - 1) / other_players = 210 := by
  sorry

end NUMINAMATH_CALUDE_sarah_mia_games_together_l3171_317160


namespace NUMINAMATH_CALUDE_negative_half_power_twenty_times_negative_two_power_twentysix_l3171_317147

theorem negative_half_power_twenty_times_negative_two_power_twentysix :
  -0.5^20 * (-2)^26 = -64 := by
  sorry

end NUMINAMATH_CALUDE_negative_half_power_twenty_times_negative_two_power_twentysix_l3171_317147


namespace NUMINAMATH_CALUDE_range_of_a_l3171_317105

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 3*a < x ∧ x < a ∧ a < 0}
def B : Set ℝ := {x | x < -4 ∨ x ≥ -2}

-- Define the conditions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

-- State the theorem
theorem range_of_a :
  (∀ x, p x a → q x) ∧ 
  (∃ x, ¬p x a ∧ q x) →
  (a ≤ -4 ∨ (-2/3 ≤ a ∧ a < 0)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3171_317105


namespace NUMINAMATH_CALUDE_hot_dog_cost_l3171_317199

/-- The cost of a hot dog given the conditions of the concession stand problem -/
theorem hot_dog_cost (soda_cost : ℝ) (total_revenue : ℝ) (total_items : ℕ) (hot_dogs_sold : ℕ) :
  soda_cost = 0.50 →
  total_revenue = 78.50 →
  total_items = 87 →
  hot_dogs_sold = 35 →
  ∃ (hot_dog_cost : ℝ), 
    hot_dog_cost * hot_dogs_sold + soda_cost * (total_items - hot_dogs_sold) = total_revenue ∧
    hot_dog_cost = 1.50 := by
  sorry


end NUMINAMATH_CALUDE_hot_dog_cost_l3171_317199


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l3171_317102

theorem quadratic_function_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a < 0) 
  (hx : x₁ < x₂) 
  (hsum : x₁ + x₂ = 0) : 
  2 * a * x₁^2 - a * x₁ + 1 < 2 * a * x₂^2 - a * x₂ + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l3171_317102


namespace NUMINAMATH_CALUDE_min_value_theorem_l3171_317130

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3171_317130


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3171_317137

theorem quadratic_root_value (a : ℝ) : (1 : ℝ)^2 + a * 1 + 4 = 0 → a = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3171_317137


namespace NUMINAMATH_CALUDE_sinusoidal_vertical_shift_l3171_317111

theorem sinusoidal_vertical_shift 
  (A B C D : ℝ) 
  (h_max : ∀ x, A * Real.sin (B * x + C) + D ≤ 5)
  (h_min : ∀ x, A * Real.sin (B * x + C) + D ≥ -3)
  (h_max_achieved : ∃ x, A * Real.sin (B * x + C) + D = 5)
  (h_min_achieved : ∃ x, A * Real.sin (B * x + C) + D = -3) :
  D = 1 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_vertical_shift_l3171_317111


namespace NUMINAMATH_CALUDE_fraction_inequality_l3171_317106

theorem fraction_inequality (x : ℝ) :
  0 ≤ x ∧ x ≤ 3 →
  (3 * x + 2 < 2 * (5 * x - 4) ↔ 10 / 7 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3171_317106


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3171_317158

theorem quadratic_factorization (y : ℝ) : y^2 + 14*y + 40 = (y + 4) * (y + 10) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3171_317158


namespace NUMINAMATH_CALUDE_function_properties_l3171_317166

/-- Given f(x) = a(x+b)(x+c) and g(x) = xf(x) where a ≠ 0 and a, b, c ∈ ℝ,
    prove the following statements -/
theorem function_properties :
  ∃ (a b c : ℝ), a ≠ 0 ∧
    (∀ x, (a * (1 + x) * (x + b) * (x + c) = 0) ↔ (a * (1 - x) * (x + b) * (x + c) = 0)) ∧
    (∀ x, (2 * a * x = -(2 * a * (-x))) ∧ (a * (3 * x^2 + 2 * (b + c) * x + b * c) = a * (3 * (-x)^2 + 2 * (b + c) * (-x) + b * c))) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3171_317166


namespace NUMINAMATH_CALUDE_speed_increase_reduces_time_l3171_317132

/-- Given a 600-mile trip at 50 mph, prove that increasing speed by 25 mph reduces travel time by 4 hours -/
theorem speed_increase_reduces_time : ∀ (distance : ℝ) (initial_speed : ℝ) (speed_increase : ℝ),
  distance = 600 →
  initial_speed = 50 →
  speed_increase = 25 →
  distance / initial_speed - distance / (initial_speed + speed_increase) = 4 :=
by
  sorry

#check speed_increase_reduces_time

end NUMINAMATH_CALUDE_speed_increase_reduces_time_l3171_317132


namespace NUMINAMATH_CALUDE_painted_faces_difference_l3171_317197

/-- Represents a 3D cube structure --/
structure CubeStructure where
  length : Nat
  width : Nat
  height : Nat

/-- Counts cubes with exactly n painted faces in the structure --/
def countPaintedFaces (cs : CubeStructure) (n : Nat) : Nat :=
  sorry

/-- The main theorem to be proved --/
theorem painted_faces_difference (cs : CubeStructure) :
  cs.length = 7 → cs.width = 7 → cs.height = 3 →
  countPaintedFaces cs 3 - countPaintedFaces cs 2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_painted_faces_difference_l3171_317197


namespace NUMINAMATH_CALUDE_softball_team_size_l3171_317118

theorem softball_team_size :
  ∀ (men women : ℕ),
  women = men + 4 →
  (men : ℚ) / (women : ℚ) = 5555555555555556 / 10000000000000000 →
  men + women = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_size_l3171_317118


namespace NUMINAMATH_CALUDE_tournament_result_l3171_317100

-- Define the type for teams
inductive Team : Type
| A | B | C | D

-- Define the type for match results
inductive MatchResult : Type
| Win | Loss

-- Define a function to represent the number of wins for each team
def wins : Team → Nat
| Team.A => 2
| Team.B => 0
| Team.C => 1
| Team.D => 3

-- Define a function to represent the number of losses for each team
def losses : Team → Nat
| Team.A => 1
| Team.B => 3
| Team.C => 2
| Team.D => 0

-- Theorem statement
theorem tournament_result :
  (∀ t : Team, wins t + losses t = 3) ∧
  (wins Team.A + wins Team.B + wins Team.C + wins Team.D = 6) ∧
  (losses Team.A + losses Team.B + losses Team.C + losses Team.D = 6) :=
by sorry

end NUMINAMATH_CALUDE_tournament_result_l3171_317100


namespace NUMINAMATH_CALUDE_camp_kids_count_l3171_317114

theorem camp_kids_count (total : ℕ) (soccer : ℕ) (morning : ℕ) (afternoon : ℕ) :
  soccer = total / 2 →
  morning = soccer / 4 →
  afternoon = 750 →
  afternoon = soccer * 3 / 4 →
  total = 2000 := by
sorry

end NUMINAMATH_CALUDE_camp_kids_count_l3171_317114


namespace NUMINAMATH_CALUDE_class_size_calculation_l3171_317151

theorem class_size_calculation (tables : Nat) (students_per_table : Nat)
  (bathroom_girls : Nat) (canteen_multiplier : Nat)
  (group1 : Nat) (group2 : Nat) (group3 : Nat) (group4 : Nat)
  (germany : Nat) (france : Nat) (norway : Nat) (italy : Nat) (spain : Nat) (australia : Nat) :
  tables = 6 →
  students_per_table = 3 →
  bathroom_girls = 5 →
  canteen_multiplier = 5 →
  group1 = 4 →
  group2 = 5 →
  group3 = 6 →
  group4 = 3 →
  germany = 3 →
  france = 4 →
  norway = 3 →
  italy = 2 →
  spain = 2 →
  australia = 1 →
  (tables * students_per_table + bathroom_girls + bathroom_girls * canteen_multiplier +
   group1 + group2 + group3 + group4 +
   germany + france + norway + italy + spain + australia) = 81 :=
by sorry

end NUMINAMATH_CALUDE_class_size_calculation_l3171_317151


namespace NUMINAMATH_CALUDE_min_total_distance_l3171_317138

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 13 ∧ dist B C = 14 ∧ dist C A = 15

-- Define the total distance function
def TotalDistance (A B C P : ℝ × ℝ) : ℝ :=
  dist A P + 5 * dist B P + 4 * dist C P

-- State the theorem
theorem min_total_distance (A B C : ℝ × ℝ) (h : Triangle A B C) :
  ∀ P : ℝ × ℝ, TotalDistance A B C P ≥ 69 ∧
  (TotalDistance A B C B = 69) :=
by sorry

end NUMINAMATH_CALUDE_min_total_distance_l3171_317138


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l3171_317162

theorem quadratic_form_minimum : ∀ x y : ℝ, 
  2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x - 6 * y + 1 ≥ -3 ∧ 
  (2 * (3/2)^2 + 4 * (3/2) * (1/2) + 5 * (1/2)^2 - 4 * (3/2) - 6 * (1/2) + 1 = -3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l3171_317162


namespace NUMINAMATH_CALUDE_reciprocal_of_opposite_of_negative_l3171_317195

theorem reciprocal_of_opposite_of_negative : 
  (1 / -(- -3)) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_opposite_of_negative_l3171_317195


namespace NUMINAMATH_CALUDE_exists_permutation_with_unique_sums_l3171_317191

/-- A permutation of numbers 1 to 10 -/
def Permutation := Fin 10 → Fin 10

/-- Function to check if a permutation results in unique adjacent sums when arranged in a circle -/
def has_unique_adjacent_sums (p : Permutation) : Prop :=
  ∀ i j : Fin 10, i ≠ j → 
    (p i + p ((i + 1) % 10) : ℕ) ≠ (p j + p ((j + 1) % 10) : ℕ)

/-- Theorem stating that there exists a permutation with unique adjacent sums -/
theorem exists_permutation_with_unique_sums : 
  ∃ p : Permutation, Function.Bijective p ∧ has_unique_adjacent_sums p :=
sorry

end NUMINAMATH_CALUDE_exists_permutation_with_unique_sums_l3171_317191


namespace NUMINAMATH_CALUDE_min_value_of_f_l3171_317175

def f (x a b : ℝ) : ℝ := (x + a + b) * (x + a - b) * (x - a + b) * (x - a - b)

theorem min_value_of_f (a b : ℝ) : 
  ∃ (m : ℝ), ∀ (x : ℝ), f x a b ≥ m ∧ ∃ (x₀ : ℝ), f x₀ a b = m ∧ m = -4 * a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3171_317175


namespace NUMINAMATH_CALUDE_opposite_side_length_l3171_317115

/-- Represents a right triangle with one acute angle of 30 degrees and hypotenuse of 10 units -/
structure RightTriangle30 where
  -- The hypotenuse length is 10 units
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = 10
  -- One acute angle is 30 degrees (π/6 radians)
  acute_angle : ℝ
  acute_angle_eq : acute_angle = π/6

/-- 
Theorem: In a right triangle with one acute angle of 30° and a hypotenuse of 10 units, 
the length of the side opposite to the 30° angle is 5 units.
-/
theorem opposite_side_length (t : RightTriangle30) : 
  Real.sin t.acute_angle * t.hypotenuse = 5 := by
  sorry


end NUMINAMATH_CALUDE_opposite_side_length_l3171_317115


namespace NUMINAMATH_CALUDE_roots_fourth_power_sum_lower_bound_l3171_317179

theorem roots_fourth_power_sum_lower_bound (p : ℝ) (hp : p ≠ 0) :
  let x₁ := (-p + Real.sqrt (p^2 + 2/p^2)) / 2
  let x₂ := (-p - Real.sqrt (p^2 + 2/p^2)) / 2
  x₁^4 + x₂^4 ≥ 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_roots_fourth_power_sum_lower_bound_l3171_317179


namespace NUMINAMATH_CALUDE_profit_share_ratio_l3171_317104

theorem profit_share_ratio (total_profit : ℝ) (difference : ℝ) 
  (h_total : total_profit = 1000)
  (h_diff : difference = 200) :
  ∃ (x y : ℝ), 
    x + y = total_profit ∧ 
    x - y = difference ∧ 
    x / total_profit = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l3171_317104


namespace NUMINAMATH_CALUDE_initial_amount_calculation_l3171_317134

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Final amount after simple interest -/
def final_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + simple_interest principal rate time

/-- Theorem: Initial amount calculation for given simple interest scenario -/
theorem initial_amount_calculation (rate : ℝ) (time : ℝ) (final : ℝ) 
  (h_rate : rate = 0.04)
  (h_time : time = 5)
  (h_final : final = 900) :
  ∃ (principal : ℝ), final_amount principal rate time = final ∧ principal = 750 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_calculation_l3171_317134


namespace NUMINAMATH_CALUDE_stream_speed_l3171_317169

/-- Proves that the speed of a stream is 3 kmph given the conditions of boat travel -/
theorem stream_speed (boat_speed : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : downstream_time = 1)
  (h3 : upstream_time = 1.5) :
  ∃ stream_speed : ℝ, 
    stream_speed = 3 ∧ 
    (boat_speed + stream_speed) * downstream_time = (boat_speed - stream_speed) * upstream_time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l3171_317169


namespace NUMINAMATH_CALUDE_permutation_sum_theorem_combination_sum_theorem_l3171_317174

-- Define the permutation function
def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Define the combination function
def C (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem permutation_sum_theorem :
  A 5 1 + A 5 2 + A 5 3 + A 5 4 + A 5 5 = 325 := by sorry

theorem combination_sum_theorem (m : ℕ) (h1 : m > 1) (h2 : C 5 m = C 5 (2*m - 1)) :
  C 6 m + C 6 (m+1) + C 7 (m+2) + C 8 (m+3) = 126 := by sorry

end NUMINAMATH_CALUDE_permutation_sum_theorem_combination_sum_theorem_l3171_317174


namespace NUMINAMATH_CALUDE_theater_empty_seats_l3171_317194

/-- Given a theater with total seats and occupied seats, calculate the number of empty seats. -/
def empty_seats (total_seats occupied_seats : ℕ) : ℕ :=
  total_seats - occupied_seats

/-- Theorem: In a theater with 750 seats and 532 people watching, there are 218 empty seats. -/
theorem theater_empty_seats :
  empty_seats 750 532 = 218 := by
  sorry

end NUMINAMATH_CALUDE_theater_empty_seats_l3171_317194


namespace NUMINAMATH_CALUDE_permutations_theorem_l3171_317187

-- Define the number of books
def n : ℕ := 30

-- Define the function to calculate the number of permutations where two specific objects are not adjacent
def permutations_not_adjacent (n : ℕ) : ℕ := 28 * Nat.factorial (n - 1)

-- Theorem statement
theorem permutations_theorem :
  permutations_not_adjacent n = (n - 2) * Nat.factorial (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_permutations_theorem_l3171_317187


namespace NUMINAMATH_CALUDE_fourth_root_sqrt_five_squared_l3171_317116

theorem fourth_root_sqrt_five_squared : 
  ((5 ^ (1 / 2)) ^ 5) ^ (1 / 4) ^ 2 = 5 * (5 ^ (1 / 4)) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_sqrt_five_squared_l3171_317116


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3171_317198

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (9 + 3 * z) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3171_317198


namespace NUMINAMATH_CALUDE_white_area_theorem_l3171_317136

/-- A painting with two white squares in a gray field -/
structure Painting where
  s : ℝ
  gray_area : ℝ
  total_side_length : ℝ
  smaller_square_side : ℝ
  larger_square_side : ℝ

/-- The theorem stating the area of the white part given the conditions -/
theorem white_area_theorem (p : Painting) 
    (h1 : p.total_side_length = 6 * p.s)
    (h2 : p.smaller_square_side = p.s)
    (h3 : p.larger_square_side = 2 * p.s)
    (h4 : p.gray_area = 62) : 
  ∃ (white_area : ℝ), white_area = 10 := by
  sorry

end NUMINAMATH_CALUDE_white_area_theorem_l3171_317136


namespace NUMINAMATH_CALUDE_other_person_age_is_six_l3171_317186

-- Define Noah's current age
def noah_current_age : ℕ := 22 - 10

-- Define the relationship between Noah's age and the other person's age
def other_person_age : ℕ := noah_current_age / 2

-- Theorem to prove
theorem other_person_age_is_six : other_person_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_other_person_age_is_six_l3171_317186


namespace NUMINAMATH_CALUDE_problem_solution_l3171_317142

def problem (a b m : ℝ × ℝ) : Prop :=
  let midpoint := λ (x y : ℝ × ℝ) => ((x.1 + y.1) / 2, (x.2 + y.2) / 2)
  m = midpoint (2 • a) (2 • b) ∧
  m = (4, 6) ∧
  a.1 * b.1 + a.2 * b.2 = 10 →
  a.1^2 + a.2^2 + b.1^2 + b.2^2 = 32

theorem problem_solution (a b m : ℝ × ℝ) :
  problem a b m := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3171_317142


namespace NUMINAMATH_CALUDE_arrangement_count_l3171_317152

/-- Represents the number of boys -/
def num_boys : Nat := 2

/-- Represents the number of girls -/
def num_girls : Nat := 3

/-- Represents the total number of students -/
def total_students : Nat := num_boys + num_girls

/-- Represents that the girls are adjacent -/
def girls_adjacent : Prop := True

/-- Represents that boy A is to the left of boy B -/
def boy_A_left_of_B : Prop := True

/-- The number of different arrangements -/
def num_arrangements : Nat := 18

theorem arrangement_count :
  girls_adjacent →
  boy_A_left_of_B →
  num_arrangements = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l3171_317152


namespace NUMINAMATH_CALUDE_f_4_eq_7_solutions_l3171_317127

/-- The function f(x) = x^2 - 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- The fourth composition of f -/
def f_4 (x : ℝ) : ℝ := f (f (f (f x)))

/-- The theorem stating that there are exactly 5 distinct real solutions to f⁴(c) = 7 -/
theorem f_4_eq_7_solutions :
  ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ c : ℝ, c ∈ s ↔ f_4 c = 7 := by sorry

end NUMINAMATH_CALUDE_f_4_eq_7_solutions_l3171_317127


namespace NUMINAMATH_CALUDE_evaluate_expression_l3171_317125

theorem evaluate_expression (x y z : ℚ) :
  x = 1/3 → y = 2/3 → z = -9 → x^2 * y^3 * z = -8/27 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3171_317125


namespace NUMINAMATH_CALUDE_no_real_solutions_quadratic_l3171_317149

theorem no_real_solutions_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x - k ≠ 0) ↔ k < -9/4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_quadratic_l3171_317149


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3171_317135

/-- A quadratic function f(x) = x^2 + mx + n with roots -2 and -1 -/
def f (m n : ℝ) (x : ℝ) : ℝ := x^2 + m*x + n

theorem quadratic_function_properties (m n : ℝ) :
  (∀ x, f m n x = 0 ↔ x = -2 ∨ x = -1) →
  (m = 3 ∧ n = 2) ∧
  (∀ x ∈ Set.Icc (-5 : ℝ) 5,
    f m n x ≥ -1/4 ∧
    f m n x ≤ 42 ∧
    (∃ x₁ ∈ Set.Icc (-5 : ℝ) 5, f m n x₁ = -1/4) ∧
    (∃ x₂ ∈ Set.Icc (-5 : ℝ) 5, f m n x₂ = 42)) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l3171_317135


namespace NUMINAMATH_CALUDE_three_function_properties_l3171_317178

theorem three_function_properties :
  (∃ f : ℝ → ℝ, ∀ x : ℝ, f x - (deriv f) x = f (-x) - (deriv f) (-x)) ∧
  (∃ f : ℝ → ℝ, (∀ x : ℝ, (deriv f) x ≠ 0) ∧ (∀ x : ℝ, f x = (deriv f) x)) ∧
  (∃ f : ℝ → ℝ, (∀ x : ℝ, (deriv f) x ≠ 0) ∧ (∀ x : ℝ, f x = -(deriv f) x)) :=
by sorry

end NUMINAMATH_CALUDE_three_function_properties_l3171_317178


namespace NUMINAMATH_CALUDE_curve_intersection_distance_l3171_317146

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - x + y = 0

def C₂ (t x y : ℝ) : Prop := x = 1/2 - (Real.sqrt 2 / 2) * t ∧ y = (Real.sqrt 2 / 2) * t

-- Theorem statement
theorem curve_intersection_distance : 
  -- The polar curve ρ = cos θ - sin θ is equivalent to C₁
  (∀ (ρ θ : ℝ), ρ = Real.cos θ - Real.sin θ ↔ C₁ (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  -- The distance between intersection points is √6/2
  (∃ (t₁ t₂ : ℝ), 
    (C₁ (1/2 - (Real.sqrt 2 / 2) * t₁) ((Real.sqrt 2 / 2) * t₁)) ∧
    (C₁ (1/2 - (Real.sqrt 2 / 2) * t₂) ((Real.sqrt 2 / 2) * t₂)) ∧
    (C₂ t₁ (1/2 - (Real.sqrt 2 / 2) * t₁) ((Real.sqrt 2 / 2) * t₁)) ∧
    (C₂ t₂ (1/2 - (Real.sqrt 2 / 2) * t₂) ((Real.sqrt 2 / 2) * t₂)) ∧
    (t₁ ≠ t₂) ∧
    ((1/2 - (Real.sqrt 2 / 2) * t₁ - (1/2 - (Real.sqrt 2 / 2) * t₂))^2 + 
     ((Real.sqrt 2 / 2) * t₁ - (Real.sqrt 2 / 2) * t₂)^2 = 3/2)) := by
  sorry

end NUMINAMATH_CALUDE_curve_intersection_distance_l3171_317146


namespace NUMINAMATH_CALUDE_fraction_reducible_by_11_l3171_317129

theorem fraction_reducible_by_11 (k : ℕ) 
  (h : (k^2 - 5*k + 8) % 11 = 0 ∨ (k^2 + 6*k + 19) % 11 = 0) : 
  (k^2 - 5*k + 8) % 11 = 0 ∧ (k^2 + 6*k + 19) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_reducible_by_11_l3171_317129


namespace NUMINAMATH_CALUDE_john_pill_schedule_l3171_317181

/-- The number of pills John takes per week -/
def pills_per_week : ℕ := 28

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of hours between each pill John takes -/
def hours_between_pills : ℚ :=
  (days_per_week * hours_per_day) / pills_per_week

theorem john_pill_schedule :
  hours_between_pills = 6 := by sorry

end NUMINAMATH_CALUDE_john_pill_schedule_l3171_317181


namespace NUMINAMATH_CALUDE_equation_describes_spiral_l3171_317163

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The equation r * θ = c -/
def spiralEquation (p : CylindricalPoint) (c : ℝ) : Prop :=
  p.r * p.θ = c

/-- A spiral in cylindrical coordinates -/
def isSpiral (S : Set CylindricalPoint) : Prop :=
  ∃ c : ℝ, ∀ p ∈ S, spiralEquation p c

/-- The shape described by r * θ = c is a spiral -/
theorem equation_describes_spiral (c : ℝ) :
  isSpiral {p : CylindricalPoint | spiralEquation p c} :=
sorry

end NUMINAMATH_CALUDE_equation_describes_spiral_l3171_317163


namespace NUMINAMATH_CALUDE_max_value_problem_l3171_317110

theorem max_value_problem (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 6 * y ≤ 12) : 
  ∀ a b : ℝ, 4 * a + 3 * b ≤ 10 → 3 * a + 6 * b ≤ 12 → 2 * x + y ≥ 2 * a + b :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l3171_317110


namespace NUMINAMATH_CALUDE_range_of_cosine_function_l3171_317128

theorem range_of_cosine_function (f : ℝ → ℝ) (x : ℝ) :
  (f = λ x => 3 * Real.cos (2 * x + π / 3)) →
  (x ∈ Set.Icc 0 (π / 3)) →
  ∃ y ∈ Set.Icc (-3) (3 / 2), f x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_cosine_function_l3171_317128


namespace NUMINAMATH_CALUDE_second_number_is_40_l3171_317113

theorem second_number_is_40 (a b c : ℝ) 
  (sum_eq : a + b + c = 120)
  (ratio_ab : a / b = 3 / 4)
  (ratio_bc : b / c = 4 / 5)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0) :
  b = 40 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_40_l3171_317113


namespace NUMINAMATH_CALUDE_toy_store_revenue_ratio_l3171_317170

/-- Given a toy store's revenue data for three months, prove that January's revenue is 1/5 of November's revenue. -/
theorem toy_store_revenue_ratio :
  ∀ (nov dec jan : ℝ),
  nov > 0 →
  nov = (2/5) * dec →
  dec = (25/6) * ((nov + jan) / 2) →
  jan = (1/5) * nov :=
by sorry

end NUMINAMATH_CALUDE_toy_store_revenue_ratio_l3171_317170


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3171_317103

theorem perfect_square_trinomial (a k : ℝ) : 
  (∃ b : ℝ, a^2 + 2*k*a + 9 = (a + b)^2) → (k = 3 ∨ k = -3) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3171_317103


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3171_317145

theorem polynomial_factorization (x : ℝ) : 
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3)*(8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3171_317145


namespace NUMINAMATH_CALUDE_E_is_integer_l3171_317153

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The expression E as defined in the problem -/
def E (n k : ℕ) : ℚ :=
  ((n - 2*k - 2) : ℚ) / ((k + 2) : ℚ) * binomial n k

theorem E_is_integer (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  ∃ (m : ℤ), E n k = m :=
sorry

end NUMINAMATH_CALUDE_E_is_integer_l3171_317153
