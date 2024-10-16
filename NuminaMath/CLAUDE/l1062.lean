import Mathlib

namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1062_106268

-- Problem 1
theorem problem_1 (a b : ℝ) : (a * b) ^ 6 / (a * b) ^ 2 * (a * b) ^ 4 = a ^ 8 * b ^ 8 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (3 * x ^ 3) ^ 2 * x ^ 5 - (-x ^ 2) ^ 6 / x = 8 * x ^ 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1062_106268


namespace NUMINAMATH_CALUDE_parabola_point_focus_distance_l1062_106253

theorem parabola_point_focus_distance (p m : ℝ) : 
  p > 0 → 
  m^2 = 2*p*4 → 
  (4 + p/2)^2 + m^2 = (17/4)^2 → 
  p = 1/2 ∧ (m = 2 ∨ m = -2) := by
sorry

end NUMINAMATH_CALUDE_parabola_point_focus_distance_l1062_106253


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l1062_106243

theorem largest_divisor_of_difference_of_squares (m n : ℤ) :
  (∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) →  -- m and n are even
  n < m →                              -- n is less than m
  (∀ x : ℤ, x ∣ (m^2 - n^2) → x ≤ 8) ∧ -- 8 is an upper bound for divisors
  8 ∣ (m^2 - n^2)                      -- 8 divides m^2 - n^2
  := by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l1062_106243


namespace NUMINAMATH_CALUDE_sale_price_determination_l1062_106258

/-- Proves that the sale price of each machine is $10,000 given the commission structure and total commission --/
theorem sale_price_determination (commission_rate_first_100 : ℝ) (commission_rate_after_100 : ℝ) 
  (total_machines : ℕ) (machines_at_first_rate : ℕ) (total_commission : ℝ) :
  commission_rate_first_100 = 0.03 →
  commission_rate_after_100 = 0.04 →
  total_machines = 130 →
  machines_at_first_rate = 100 →
  total_commission = 42000 →
  ∃ (sale_price : ℝ), 
    sale_price = 10000 ∧
    (machines_at_first_rate : ℝ) * commission_rate_first_100 * sale_price + 
    ((total_machines - machines_at_first_rate) : ℝ) * commission_rate_after_100 * sale_price = 
    total_commission :=
by
  sorry

#check sale_price_determination

end NUMINAMATH_CALUDE_sale_price_determination_l1062_106258


namespace NUMINAMATH_CALUDE_quartic_factorization_and_solutions_l1062_106242

theorem quartic_factorization_and_solutions :
  ∃ (x₁ x₂ x₃ x₄ : ℂ),
    (∀ x : ℂ, x^4 + 1 = (x^2 + Real.sqrt 2 * x + 1) * (x^2 - Real.sqrt 2 * x + 1)) ∧
    x₁ = -Real.sqrt 2 / 2 + Complex.I * Real.sqrt 2 / 2 ∧
    x₂ = -Real.sqrt 2 / 2 - Complex.I * Real.sqrt 2 / 2 ∧
    x₃ = Real.sqrt 2 / 2 + Complex.I * Real.sqrt 2 / 2 ∧
    x₄ = Real.sqrt 2 / 2 - Complex.I * Real.sqrt 2 / 2 ∧
    {x | x^4 + 1 = 0} = {x₁, x₂, x₃, x₄} := by
  sorry

end NUMINAMATH_CALUDE_quartic_factorization_and_solutions_l1062_106242


namespace NUMINAMATH_CALUDE_units_digit_of_sum_units_digit_of_35_power_87_plus_93_power_49_l1062_106261

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to get the units digit of a power of 5
def unitsDigitPowerOf5 (n : ℕ) : ℕ := 5

-- Define a function to get the units digit of a power of 3
def unitsDigitPowerOf3 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0 -- This case should never occur

theorem units_digit_of_sum (a b : ℕ) :
  unitsDigit (a + b) = unitsDigit (unitsDigit a + unitsDigit b) :=
sorry

theorem units_digit_of_35_power_87_plus_93_power_49 :
  unitsDigit ((35 ^ 87) + (93 ^ 49)) = 8 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_units_digit_of_35_power_87_plus_93_power_49_l1062_106261


namespace NUMINAMATH_CALUDE_marys_income_percentage_l1062_106201

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * (1 - 0.5))
  (h2 : mary = tim * (1 + 0.6)) :
  mary = juan * 0.8 := by sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l1062_106201


namespace NUMINAMATH_CALUDE_line_on_plane_perp_other_plane_implies_planes_perp_l1062_106266

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- The line lies on the plane -/
def lies_on (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- The line is perpendicular to the plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two planes are perpendicular -/
def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem line_on_plane_perp_other_plane_implies_planes_perp
  (l : Line3D) (α β : Plane3D) :
  lies_on l α → perpendicular_line_plane l β → perpendicular_planes α β :=
by sorry

end NUMINAMATH_CALUDE_line_on_plane_perp_other_plane_implies_planes_perp_l1062_106266


namespace NUMINAMATH_CALUDE_union_when_a_is_3_union_equals_B_iff_l1062_106252

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a^2 - 2}
def B : Set ℝ := Set.Ioo 1 5

-- Statement 1: When a = 3, A ∪ B = (1, 7]
theorem union_when_a_is_3 : 
  A 3 ∪ B = Set.Ioc 1 7 := by sorry

-- Statement 2: A ∪ B = B if and only if a ∈ (2, √7)
theorem union_equals_B_iff (a : ℝ) : 
  A a ∪ B = B ↔ a ∈ Set.Ioo 2 (Real.sqrt 7) := by sorry

end NUMINAMATH_CALUDE_union_when_a_is_3_union_equals_B_iff_l1062_106252


namespace NUMINAMATH_CALUDE_sticker_count_l1062_106223

theorem sticker_count (stickers_per_page : ℕ) (number_of_pages : ℕ) : 
  stickers_per_page = 10 → number_of_pages = 22 → stickers_per_page * number_of_pages = 220 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_l1062_106223


namespace NUMINAMATH_CALUDE_kendras_earnings_theorem_l1062_106228

/-- Kendra's total earnings in 2014 and 2015 -/
def kendras_total_earnings (laurel_2014 : ℝ) : ℝ :=
  let kendra_2014 := laurel_2014 - 8000
  let kendra_2015 := laurel_2014 * 1.2
  kendra_2014 + kendra_2015

/-- Theorem stating Kendra's total earnings given Laurel's 2014 earnings -/
theorem kendras_earnings_theorem (laurel_2014 : ℝ) 
  (h : laurel_2014 = 30000) : 
  kendras_total_earnings laurel_2014 = 58000 := by
  sorry

end NUMINAMATH_CALUDE_kendras_earnings_theorem_l1062_106228


namespace NUMINAMATH_CALUDE_fraction_nonzero_digits_l1062_106255

def fraction : ℚ := 120 / (2^4 * 5^8)

def count_nonzero_decimal_digits (q : ℚ) : ℕ :=
  -- Function to count non-zero digits after the decimal point
  sorry

theorem fraction_nonzero_digits :
  count_nonzero_decimal_digits fraction = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_nonzero_digits_l1062_106255


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1062_106208

theorem no_integer_solutions : ¬∃ (x y z : ℤ), 
  (x^2 - 3*x*y + 2*y^2 - z^2 = 39) ∧ 
  (-x^2 + 6*y*z + 2*z^2 = 40) ∧ 
  (x^2 + x*y + 8*z^2 = 96) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1062_106208


namespace NUMINAMATH_CALUDE_intersection_of_A_and_union_of_B_C_l1062_106273

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem intersection_of_A_and_union_of_B_C : A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_union_of_B_C_l1062_106273


namespace NUMINAMATH_CALUDE_crayons_per_box_l1062_106249

theorem crayons_per_box (total_crayons : ℕ) (num_boxes : ℕ) (h1 : total_crayons = 56) (h2 : num_boxes = 8) :
  total_crayons / num_boxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_box_l1062_106249


namespace NUMINAMATH_CALUDE_coral_reef_age_conversion_l1062_106271

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : Nat) : Nat :=
  let d0 := octal % 10
  let d1 := (octal / 10) % 10
  let d2 := (octal / 100) % 10
  let d3 := (octal / 1000) % 10
  d0 * 8^0 + d1 * 8^1 + d2 * 8^2 + d3 * 8^3

theorem coral_reef_age_conversion :
  octal_to_decimal 3456 = 1838 := by
  sorry

end NUMINAMATH_CALUDE_coral_reef_age_conversion_l1062_106271


namespace NUMINAMATH_CALUDE_root_product_cubic_l1062_106288

theorem root_product_cubic (a b c : ℂ) : 
  (∀ x : ℂ, 3 * x^3 - 8 * x^2 + 5 * x - 9 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  a * b * c = 3 := by
sorry

end NUMINAMATH_CALUDE_root_product_cubic_l1062_106288


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l1062_106219

/-- Given an initial angle of 60 degrees and a clockwise rotation of 300 degrees,
    the resulting positive acute angle is 120 degrees. -/
theorem rotated_angle_measure (initial_angle rotation_angle : ℝ) : 
  initial_angle = 60 →
  rotation_angle = 300 →
  (360 - (rotation_angle - initial_angle)) % 360 = 120 :=
by sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l1062_106219


namespace NUMINAMATH_CALUDE_remainder_theorem_l1062_106241

theorem remainder_theorem (P D E Q R M S C : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = E * M + S)
  (h3 : R < D)
  (h4 : S < E) :
  ∃ K, P = K * (D * E) + (S * D + R + C) ∧ S * D + R + C < D * E :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1062_106241


namespace NUMINAMATH_CALUDE_minimum_fuse_length_l1062_106267

theorem minimum_fuse_length (safe_distance : ℝ) (personnel_speed : ℝ) (fuse_burn_speed : ℝ) :
  safe_distance = 70 →
  personnel_speed = 7 →
  fuse_burn_speed = 10.3 →
  ∃ x : ℝ, x > 103 ∧ x / fuse_burn_speed > safe_distance / personnel_speed :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_fuse_length_l1062_106267


namespace NUMINAMATH_CALUDE_fraction_inequality_l1062_106247

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a * d > b * c) 
  (h2 : a / b > c / d) 
  (hb : b > 0) 
  (hd : d > 0) : 
  a / b > (a + c) / (b + d) ∧ (a + c) / (b + d) > c / d := by
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1062_106247


namespace NUMINAMATH_CALUDE_inverse_division_identity_l1062_106236

theorem inverse_division_identity (x : ℝ) (hx : x ≠ 0) : 1 / x⁻¹ = x := by
  sorry

end NUMINAMATH_CALUDE_inverse_division_identity_l1062_106236


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1755_l1062_106279

theorem largest_prime_factor_of_1755 : ∃ p : Nat, Nat.Prime p ∧ p ∣ 1755 ∧ ∀ q : Nat, Nat.Prime q → q ∣ 1755 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1755_l1062_106279


namespace NUMINAMATH_CALUDE_third_generation_tail_length_l1062_106256

/-- The tail length increase factor between generations -/
def increase_factor : ℝ := 1.25

/-- The initial tail length of the first generation in centimeters -/
def initial_length : ℝ := 16

/-- The tail length of a given generation -/
def tail_length (generation : ℕ) : ℝ :=
  initial_length * (increase_factor ^ generation)

/-- Theorem: The tail length of the third generation is 25 cm -/
theorem third_generation_tail_length :
  tail_length 2 = 25 := by sorry

end NUMINAMATH_CALUDE_third_generation_tail_length_l1062_106256


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1062_106220

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  X^5 + 3 = (X + 1)^2 * q + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1062_106220


namespace NUMINAMATH_CALUDE_words_lost_in_oz_l1062_106229

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 69

/-- The number of letters prohibited -/
def prohibited_letters : ℕ := 1

/-- The maximum word length -/
def max_word_length : ℕ := 2

/-- Calculate the number of words lost due to letter prohibition -/
def words_lost (alphabet_size : ℕ) (prohibited_letters : ℕ) (max_word_length : ℕ) : ℕ :=
  prohibited_letters + 
  (alphabet_size * prohibited_letters + alphabet_size * prohibited_letters - prohibited_letters * prohibited_letters)

theorem words_lost_in_oz : 
  words_lost alphabet_size prohibited_letters max_word_length = 138 := by
  sorry

end NUMINAMATH_CALUDE_words_lost_in_oz_l1062_106229


namespace NUMINAMATH_CALUDE_computer_price_increase_l1062_106222

theorem computer_price_increase (d : ℝ) (h1 : 2 * d = 580) : 
  d * (1 + 0.3) = 377 := by sorry

end NUMINAMATH_CALUDE_computer_price_increase_l1062_106222


namespace NUMINAMATH_CALUDE_tom_running_distance_l1062_106281

/-- Calculates the total distance run in a week given the number of days, hours per day, and speed. -/
def weekly_distance (days : ℕ) (hours_per_day : ℝ) (speed : ℝ) : ℝ :=
  days * hours_per_day * speed

/-- Proves that running 5 days a week, 1.5 hours each day, at 8 mph results in 60 miles per week. -/
theorem tom_running_distance :
  weekly_distance 5 1.5 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_tom_running_distance_l1062_106281


namespace NUMINAMATH_CALUDE_tan_five_pi_over_four_l1062_106254

theorem tan_five_pi_over_four : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_over_four_l1062_106254


namespace NUMINAMATH_CALUDE_ladder_problem_l1062_106224

theorem ladder_problem (h1 h2 l : Real) 
  (hyp1 : h1 = 12)
  (hyp2 : h2 = 9)
  (hyp3 : l = 15) :
  Real.sqrt (l^2 - h1^2) + Real.sqrt (l^2 - h2^2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l1062_106224


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l1062_106240

/-- A trapezoid with given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ

/-- The property that the midpoint segment length is half the difference of base lengths -/
def midpoint_property (t : Trapezoid) : Prop :=
  t.midpoint_segment = (t.longer_base - t.shorter_base) / 2

theorem trapezoid_shorter_base (t : Trapezoid) 
  (h1 : t.longer_base = 103)
  (h2 : t.midpoint_segment = 5)
  (h3 : midpoint_property t) :
  t.shorter_base = 93 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l1062_106240


namespace NUMINAMATH_CALUDE_angle_inequality_theorem_l1062_106232

theorem angle_inequality_theorem (θ : Real) : 
  (π / 2 < θ ∧ θ < π) ↔ 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^3 * Real.cos θ + x * (1 - x) - (1 - x)^3 * Real.sin θ < 0) ∧
  (∀ φ : Real, 0 ≤ φ ∧ φ ≤ 2*π ∧ φ ≠ θ → 
    ∃ y : Real, 0 ≤ y ∧ y ≤ 1 ∧ y^3 * Real.cos φ + y * (1 - y) - (1 - y)^3 * Real.sin φ ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_theorem_l1062_106232


namespace NUMINAMATH_CALUDE_power_of_two_equation_l1062_106217

theorem power_of_two_equation (k : ℤ) : 
  2^2000 - 2^1999 - 2^1998 + 2^1997 = k * 2^1997 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l1062_106217


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l1062_106290

theorem average_of_remaining_numbers 
  (total : ℝ) 
  (group1 : ℝ) 
  (group2 : ℝ) 
  (h1 : total = 6 * 3.95) 
  (h2 : group1 = 2 * 3.8) 
  (h3 : group2 = 2 * 3.85) : 
  (total - group1 - group2) / 2 = 4.2 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l1062_106290


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1062_106234

/-- A hyperbola with given eccentricity and distance from focus to asymptote -/
structure Hyperbola where
  e : ℝ  -- eccentricity
  b : ℝ  -- distance from focus to asymptote

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : Prop :=
  ∃ (x y : ℝ), y^2 / 2 - x^2 / 4 = 1

/-- Theorem: For a hyperbola with eccentricity √3 and distance from focus to asymptote 2,
    the standard equation is y²/2 - x²/4 = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_e : h.e = Real.sqrt 3) 
    (h_b : h.b = 2) : 
    standard_equation h := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1062_106234


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_sqrt_three_l1062_106207

theorem reciprocal_of_negative_sqrt_three :
  ((-Real.sqrt 3)⁻¹ : ℝ) = -(Real.sqrt 3 / 3) := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_sqrt_three_l1062_106207


namespace NUMINAMATH_CALUDE_mixed_doubles_count_l1062_106211

/-- The number of ways to form a mixed doubles team -/
def mixedDoublesTeams (numMales : ℕ) (numFemales : ℕ) : ℕ :=
  numMales * numFemales

/-- Theorem: The number of ways to form a mixed doubles team
    with 5 male players and 4 female players is 20 -/
theorem mixed_doubles_count :
  mixedDoublesTeams 5 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_mixed_doubles_count_l1062_106211


namespace NUMINAMATH_CALUDE_greatest_k_value_l1062_106284

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 73) →
  k ≤ Real.sqrt 105 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l1062_106284


namespace NUMINAMATH_CALUDE_perpendicular_condition_l1062_106218

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp_plane : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Theorem statement
theorem perpendicular_condition 
  (α β : Plane) (l : Line) 
  (h_subset : subset l α) :
  (∀ β, perp_line_plane l β → perp_plane α β) ∧ 
  (∃ β, perp_plane α β ∧ ¬perp_line_plane l β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l1062_106218


namespace NUMINAMATH_CALUDE_range_of_3x_minus_y_l1062_106262

theorem range_of_3x_minus_y (x y : ℝ) 
  (h1 : -1 ≤ x + y ∧ x + y ≤ 1) 
  (h2 : 1 ≤ x - y ∧ x - y ≤ 3) : 
  1 ≤ 3*x - y ∧ 3*x - y ≤ 7 := by
sorry

end NUMINAMATH_CALUDE_range_of_3x_minus_y_l1062_106262


namespace NUMINAMATH_CALUDE_number_placement_theorem_l1062_106270

-- Define the function f that maps integer coordinates to natural numbers
def f : ℤ × ℤ → ℕ := fun (x, y) => Nat.gcd (Int.natAbs x) (Int.natAbs y)

-- Define the property that every natural number appears at some point
def surjective (f : ℤ × ℤ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), f (x, y) = n

-- Define the property of periodicity along a line
def periodic_along_line (f : ℤ × ℤ → ℕ) (a b c : ℤ) : Prop :=
  c ≠ 0 →
  (∃ (x₁ y₁ x₂ y₂ : ℤ), (a * x₁ + b * y₁ = c) ∧ (a * x₂ + b * y₂ = c) ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) →
  ∃ (dx dy : ℤ), ∀ (x y : ℤ), 
    (a * x + b * y = c) → f (x, y) = f (x + dx, y + dy)

theorem number_placement_theorem :
  surjective f ∧ 
  (∀ (a b c : ℤ), periodic_along_line f a b c) :=
sorry

end NUMINAMATH_CALUDE_number_placement_theorem_l1062_106270


namespace NUMINAMATH_CALUDE_joes_juices_l1062_106257

/-- The number of juices Joe bought at the market -/
def num_juices : ℕ := 7

/-- The cost of a single orange in dollars -/
def orange_cost : ℚ := 4.5

/-- The cost of a single juice in dollars -/
def juice_cost : ℚ := 0.5

/-- The cost of a single jar of honey in dollars -/
def honey_cost : ℚ := 5

/-- The cost of two plants in dollars -/
def two_plants_cost : ℚ := 18

/-- The total amount Joe spent at the market in dollars -/
def total_spent : ℚ := 68

/-- The number of oranges Joe bought -/
def num_oranges : ℕ := 3

/-- The number of jars of honey Joe bought -/
def num_honey : ℕ := 3

/-- The number of plants Joe bought -/
def num_plants : ℕ := 4

theorem joes_juices :
  num_juices * juice_cost = 
    total_spent - (num_oranges * orange_cost + num_honey * honey_cost + (num_plants / 2) * two_plants_cost) :=
by sorry

end NUMINAMATH_CALUDE_joes_juices_l1062_106257


namespace NUMINAMATH_CALUDE_max_value_trig_product_l1062_106275

theorem max_value_trig_product (x y z : ℝ) :
  (Real.sin (3 * x) + Real.sin (2 * y) + Real.sin z) *
  (Real.cos (3 * x) + Real.cos (2 * y) + Real.cos z) ≤ 4.5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_product_l1062_106275


namespace NUMINAMATH_CALUDE_number_of_friends_l1062_106292

theorem number_of_friends (total_cards : ℕ) (cards_per_friend : ℕ) (h1 : total_cards = 455) (h2 : cards_per_friend = 91) :
  total_cards / cards_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_friends_l1062_106292


namespace NUMINAMATH_CALUDE_emily_sixth_quiz_score_l1062_106296

def emily_scores : List ℕ := [85, 90, 88, 92, 98]
def desired_mean : ℕ := 92
def num_quizzes : ℕ := 6

theorem emily_sixth_quiz_score :
  ∃ (sixth_score : ℕ),
    (emily_scores.sum + sixth_score) / num_quizzes = desired_mean ∧
    sixth_score = 99 := by
  sorry

end NUMINAMATH_CALUDE_emily_sixth_quiz_score_l1062_106296


namespace NUMINAMATH_CALUDE_panthers_games_count_l1062_106248

theorem panthers_games_count : ∀ (initial_games : ℕ) (initial_wins : ℕ),
  initial_wins = (60 * initial_games) / 100 →
  (initial_wins + 4) * 2 = initial_games + 8 →
  initial_games + 8 = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_panthers_games_count_l1062_106248


namespace NUMINAMATH_CALUDE_graduation_messages_l1062_106202

/-- 
Proves that for a class with x students, where each student writes a message 
for every other student, and the total number of messages is 930, 
the equation x(x-1) = 930 holds true.
-/
theorem graduation_messages (x : ℕ) 
  (h1 : x > 0) 
  (h2 : ∀ (s : ℕ), s < x → s.succ ≤ x → x - 1 = (x - s.succ) + s) 
  (h3 : (x * (x - 1)) = 930) : 
  x * (x - 1) = 930 := by
  sorry

end NUMINAMATH_CALUDE_graduation_messages_l1062_106202


namespace NUMINAMATH_CALUDE_semicircle_radius_in_isosceles_triangle_exists_isosceles_triangle_with_inscribed_semicircle_l1062_106221

/-- An isosceles triangle with a semicircle inscribed along its base -/
structure IsoscelesTriangleWithInscribedSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ

/-- Theorem stating the relationship between the triangle's dimensions and the semicircle's radius -/
theorem semicircle_radius_in_isosceles_triangle 
  (triangle : IsoscelesTriangleWithInscribedSemicircle) 
  (h_base : triangle.base = 20) 
  (h_height : triangle.height = 18) : 
  triangle.radius = 90 / Real.sqrt 106 := by
  sorry

/-- Existence of the isosceles triangle with the given properties -/
theorem exists_isosceles_triangle_with_inscribed_semicircle :
  ∃ (triangle : IsoscelesTriangleWithInscribedSemicircle), 
    triangle.base = 20 ∧ 
    triangle.height = 18 ∧ 
    triangle.radius = 90 / Real.sqrt 106 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_in_isosceles_triangle_exists_isosceles_triangle_with_inscribed_semicircle_l1062_106221


namespace NUMINAMATH_CALUDE_ellipse_midpoint_property_l1062_106265

noncomputable section

-- Define the ellipse C
def C : Set (ℝ × ℝ) := {p | p.1^2 / 3 + p.2^2 = 1}

-- Define vertices A₁ and A₂
def A₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
def A₂ : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the line x = -2√3
def line_P : Set (ℝ × ℝ) := {p | p.1 = -2 * Real.sqrt 3}

-- Main theorem
theorem ellipse_midpoint_property 
  (P : ℝ × ℝ) 
  (h_P : P ∈ line_P ∧ P.2 ≠ 0) 
  (M N : ℝ × ℝ) 
  (h_M : M ∈ C ∧ ∃ t : ℝ, M = (1 - t) • P + t • A₂) 
  (h_N : N ∈ C ∧ ∃ s : ℝ, N = (1 - s) • P + s • A₁) 
  (Q : ℝ × ℝ) 
  (h_Q : Q = (M + N) / 2) : 
  2 * dist A₁ Q = dist M N :=
sorry

end

end NUMINAMATH_CALUDE_ellipse_midpoint_property_l1062_106265


namespace NUMINAMATH_CALUDE_max_large_chips_l1062_106289

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_large_chips :
  ∀ (small large prime : ℕ),
    small + large = 61 →
    small = large + prime →
    is_prime prime →
    large ≤ 29 :=
by sorry

end NUMINAMATH_CALUDE_max_large_chips_l1062_106289


namespace NUMINAMATH_CALUDE_equation_solution_l1062_106293

theorem equation_solution : 
  ∃! x : ℚ, (x^2 - 4*x + 3)/(x^2 - 6*x + 5) = (x^2 - 3*x - 10)/(x^2 - 2*x - 15) ∧ x = -19/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1062_106293


namespace NUMINAMATH_CALUDE_not_ideal_implies_cool_or_windy_l1062_106238

-- Define the conditions for an ideal picnicking day
def ideal_picnicking (temperature : ℝ) (windy : Bool) : Prop :=
  temperature ≥ 70 ∧ ¬windy

-- Define the theorem
theorem not_ideal_implies_cool_or_windy :
  ∀ (temperature : ℝ) (windy : Bool),
    ¬(ideal_picnicking temperature windy) →
    (temperature < 70 ∨ windy) := by
  sorry

end NUMINAMATH_CALUDE_not_ideal_implies_cool_or_windy_l1062_106238


namespace NUMINAMATH_CALUDE_q_squared_minus_one_div_fifteen_l1062_106297

/-- The largest prime with 2011 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2011 digits -/
axiom q_digits : ∃ (n : ℕ), 10^2010 ≤ q ∧ q < 10^2011

/-- q is the largest prime with 2011 digits -/
axiom q_largest : ∀ (p : ℕ), Nat.Prime p → (∃ (n : ℕ), 10^2010 ≤ p ∧ p < 10^2011) → p ≤ q

theorem q_squared_minus_one_div_fifteen : 15 ∣ (q^2 - 1) := by sorry

end NUMINAMATH_CALUDE_q_squared_minus_one_div_fifteen_l1062_106297


namespace NUMINAMATH_CALUDE_inequality_ratio_l1062_106291

theorem inequality_ratio (a b : ℝ) (h1 : a < b) (h2 : b < 0) : b / a < a / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_ratio_l1062_106291


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1062_106272

def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 - (2*a + 3) * x + 6 > 0

def solution_set (a : ℝ) : Set ℝ :=
  if a < 0 then Set.Ioo (3/a) 2
  else if a = 0 then Set.Iio 2
  else if 0 < a ∧ a < 3/2 then Set.Iio 2 ∪ Set.Ioi (3/a)
  else if a = 3/2 then Set.Iio 2 ∪ Set.Ioi 2
  else Set.Iio (3/a) ∪ Set.Ioi 2

theorem quadratic_inequality_solution (a : ℝ) :
  {x | quadratic_inequality a x} = solution_set a :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1062_106272


namespace NUMINAMATH_CALUDE_bulbs_chosen_l1062_106226

theorem bulbs_chosen (total : ℕ) (defective : ℕ) (prob : ℝ) :
  total = 20 →
  defective = 4 →
  prob = 0.368421052631579 →
  (∃ n : ℕ, n = 2 ∧ 1 - ((total - defective : ℝ) / total) ^ n = prob) :=
by sorry

end NUMINAMATH_CALUDE_bulbs_chosen_l1062_106226


namespace NUMINAMATH_CALUDE_find_c_value_l1062_106233

theorem find_c_value (a b c : ℝ) 
  (eq1 : 2 * a + 3 = 5)
  (eq2 : b - a = 1)
  (eq3 : c = 2 * b) : 
  c = 4 := by sorry

end NUMINAMATH_CALUDE_find_c_value_l1062_106233


namespace NUMINAMATH_CALUDE_product_of_special_ratio_numbers_l1062_106230

theorem product_of_special_ratio_numbers (x y : ℝ) 
  (h : ∃ (k : ℝ), k > 0 ∧ x - y = k ∧ x + y = 2*k ∧ x^2 * y^2 = 18*k) : 
  x * y = 16 := by
sorry

end NUMINAMATH_CALUDE_product_of_special_ratio_numbers_l1062_106230


namespace NUMINAMATH_CALUDE_digital_root_theorem_l1062_106246

/-- Digital root of a natural number -/
def digitalRoot (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

/-- List of digital roots of first n natural numbers -/
def digitalRootList (n : ℕ) : List ℕ :=
  List.map digitalRoot (List.range n)

theorem digital_root_theorem :
  let l := digitalRootList 20092009
  (l.count 4 > l.count 5) ∧
  (l.count 9 = 2232445) ∧
  (digitalRoot (3^2009) = 9) ∧
  (digitalRoot (17^2009) = 8) := by
  sorry


end NUMINAMATH_CALUDE_digital_root_theorem_l1062_106246


namespace NUMINAMATH_CALUDE_second_class_average_mark_l1062_106298

theorem second_class_average_mark (students1 students2 : ℕ) (avg1 avg_total : ℚ) :
  students1 = 30 →
  students2 = 50 →
  avg1 = 40 →
  avg_total = 58.75 →
  (students1 : ℚ) * avg1 + (students2 : ℚ) * ((students1 + students2 : ℚ) * avg_total - (students1 : ℚ) * avg1) / (students2 : ℚ) =
    (students1 + students2 : ℚ) * avg_total →
  ((students1 + students2 : ℚ) * avg_total - (students1 : ℚ) * avg1) / (students2 : ℚ) = 70 :=
by sorry

end NUMINAMATH_CALUDE_second_class_average_mark_l1062_106298


namespace NUMINAMATH_CALUDE_smallest_k_for_mutual_criticism_l1062_106204

/-- Represents a group of deputies and their criticisms. -/
structure DeputyGroup where
  n : ℕ  -- Number of deputies
  k : ℕ  -- Number of deputies each deputy criticizes

/-- Defines when a DeputyGroup has mutual criticism. -/
def has_mutual_criticism (g : DeputyGroup) : Prop :=
  g.n * g.k > (g.n.choose 2)

/-- The smallest k that guarantees mutual criticism in a group of 15 deputies. -/
theorem smallest_k_for_mutual_criticism :
  ∃ k : ℕ, k = 8 ∧
  (∀ g : DeputyGroup, g.n = 15 → g.k ≥ k → has_mutual_criticism g) ∧
  (∀ k' : ℕ, k' < k → ∃ g : DeputyGroup, g.n = 15 ∧ g.k = k' ∧ ¬has_mutual_criticism g) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_mutual_criticism_l1062_106204


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1062_106269

/-- Given a configuration of squares and a rectangle, prove the ratio of rectangle's length to width -/
theorem rectangle_ratio (s : ℝ) (h1 : s > 0) : 
  let large_square_side := 3 * s
  let small_square_side := s
  let rectangle_width := s
  let rectangle_length := large_square_side
  (rectangle_length / rectangle_width : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1062_106269


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1062_106237

/-- A decreasing geometric sequence with first term 4, second term 2, and third term 1 -/
def geometric_sequence (n : ℕ) : ℚ :=
  4 * (1/2)^(n-1)

/-- The common ratio of the geometric sequence -/
def q : ℚ := 1/2

/-- The sum of the first n terms of the geometric sequence -/
def S (n : ℕ) : ℚ :=
  4 * (1 - (1/2)^n) / (1 - 1/2)

/-- The main theorem stating that S_8 / (1 - q^4) = 17/2 -/
theorem geometric_sequence_sum : S 8 / (1 - q^4) = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1062_106237


namespace NUMINAMATH_CALUDE_tshirts_bought_l1062_106277

/-- Given the price conditions for pants and t-shirts, 
    prove the number of t-shirts that can be bought with 800 Rs. -/
theorem tshirts_bought (pants_price t_shirt_price : ℕ) : 
  (3 * pants_price + 6 * t_shirt_price = 1500) →
  (pants_price + 12 * t_shirt_price = 1500) →
  (800 / t_shirt_price = 8) := by
  sorry

end NUMINAMATH_CALUDE_tshirts_bought_l1062_106277


namespace NUMINAMATH_CALUDE_norbs_age_l1062_106295

def guesses : List Nat := [26, 30, 35, 39, 42, 43, 45, 47, 49, 52, 53]

def isPrime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

def countLowerGuesses (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (· < age)).length

def countOffByOne (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (fun g => g = age - 1 || g = age + 1)).length

theorem norbs_age :
  ∃ (age : Nat),
    age ∈ guesses ∧
    isPrime age ∧
    countLowerGuesses age guesses ≥ guesses.length / 2 ∧
    countOffByOne age guesses = 3 ∧
    age = 47 :=
  by sorry

end NUMINAMATH_CALUDE_norbs_age_l1062_106295


namespace NUMINAMATH_CALUDE_functional_equation_defined_everywhere_l1062_106245

noncomputable def f (c : ℝ) : ℝ → ℝ :=
  λ x => if x = 0 then c
         else if x = 1 then 3 - 2*c
         else (-x^3 + 3*x^2 + 2) / (3*x*(1-x))

theorem functional_equation (c : ℝ) :
  ∀ x : ℝ, x ≠ 0 → f c x + 2 * f c ((x - 1) / x) = 3 * x :=
by sorry

theorem defined_everywhere (c : ℝ) :
  ∀ x : ℝ, ∃ y : ℝ, f c x = y :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_defined_everywhere_l1062_106245


namespace NUMINAMATH_CALUDE_digital_signal_probability_l1062_106287

theorem digital_signal_probability (p_receive_0_given_send_0 : ℝ) 
                                   (p_receive_1_given_send_0 : ℝ)
                                   (p_receive_1_given_send_1 : ℝ)
                                   (p_receive_0_given_send_1 : ℝ)
                                   (p_send_0 : ℝ)
                                   (p_send_1 : ℝ)
                                   (h1 : p_receive_0_given_send_0 = 0.9)
                                   (h2 : p_receive_1_given_send_0 = 0.1)
                                   (h3 : p_receive_1_given_send_1 = 0.95)
                                   (h4 : p_receive_0_given_send_1 = 0.05)
                                   (h5 : p_send_0 = 0.5)
                                   (h6 : p_send_1 = 0.5) :
  p_send_0 * p_receive_1_given_send_0 + p_send_1 * p_receive_1_given_send_1 = 0.525 :=
by sorry

end NUMINAMATH_CALUDE_digital_signal_probability_l1062_106287


namespace NUMINAMATH_CALUDE_floor_abs_negative_l1062_106283

theorem floor_abs_negative : ⌊|(-47.6:ℝ)|⌋ = 47 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_l1062_106283


namespace NUMINAMATH_CALUDE_inequality_proof_l1062_106214

theorem inequality_proof (b : ℝ) : (3*b - 1)*(4*b + 1) > (2*b + 1)*(5*b - 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1062_106214


namespace NUMINAMATH_CALUDE_inequality_proof_l1062_106209

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1062_106209


namespace NUMINAMATH_CALUDE_product_of_one_plus_tangents_l1062_106200

theorem product_of_one_plus_tangents (A B C : Real) : 
  A = π / 12 →  -- 15°
  B = π / 6 →   -- 30°
  A + B + C = π / 2 →  -- 90°
  (1 + Real.tan A) * (1 + Real.tan B) * (1 + Real.tan C) = (2 * Real.sqrt 3 + 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_tangents_l1062_106200


namespace NUMINAMATH_CALUDE_factor_expression_l1062_106205

theorem factor_expression (x : ℝ) : x * (x + 3) + (x + 3) = (x + 1) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1062_106205


namespace NUMINAMATH_CALUDE_total_cost_is_985_l1062_106274

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 1.75

/-- The additional cost of a train ride compared to a bus ride -/
def train_additional_cost : ℝ := 6.35

/-- The total cost of one train ride and one bus ride -/
def total_cost : ℝ := bus_cost + (bus_cost + train_additional_cost)

theorem total_cost_is_985 : total_cost = 9.85 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_985_l1062_106274


namespace NUMINAMATH_CALUDE_same_angle_from_P_l1062_106206

-- Define the basic geometric objects
structure Point : Type :=
  (x : ℝ) (y : ℝ)

structure Circle : Type :=
  (center : Point) (radius : ℝ)

-- Define the given conditions
def P : Point := sorry
def Q : Point := sorry
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry

def circle1 : Circle := sorry
def circle2 : Circle := sorry

-- Define the properties of the configuration
def circles_intersect (c1 c2 : Circle) (P Q : Point) : Prop := sorry
def line_perpendicular_to_PQ (A Q B : Point) : Prop := sorry
def Q_between_A_and_B (A Q B : Point) : Prop := sorry
def tangent_intersection (A B C : Point) (c1 c2 : Circle) : Prop := sorry

-- Define the angle between three points
def angle (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem same_angle_from_P :
  circles_intersect circle1 circle2 P Q →
  line_perpendicular_to_PQ A Q B →
  Q_between_A_and_B A Q B →
  tangent_intersection A B C circle1 circle2 →
  angle B P Q = angle Q P A := by sorry

end NUMINAMATH_CALUDE_same_angle_from_P_l1062_106206


namespace NUMINAMATH_CALUDE_repeating_decimal_length_seven_twelfths_l1062_106264

theorem repeating_decimal_length_seven_twelfths :
  ∃ (d : ℕ) (n : ℕ), 
    7 * (10^n) ≡ d [MOD 12] ∧ 
    7 * (10^(n+1)) ≡ d [MOD 12] ∧ 
    0 < d ∧ d < 12 ∧
    n = 1 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_length_seven_twelfths_l1062_106264


namespace NUMINAMATH_CALUDE_triangle_consecutive_integers_l1062_106250

theorem triangle_consecutive_integers (n : ℕ) :
  let a := n - 1
  let b := n
  let c := n + 1
  let s := (a + b + c) / 2
  let area := n + 2
  (area : ℝ)^2 = s * (s - a) * (s - b) * (s - c) → n = 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_consecutive_integers_l1062_106250


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1062_106203

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (x + y : ℝ) = -b / a :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 - 7*x + 2 - 11
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (x + y : ℝ) = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1062_106203


namespace NUMINAMATH_CALUDE_student_correct_answers_l1062_106282

/-- Represents a multiple-choice test with scoring rules -/
structure MCTest where
  total_questions : ℕ
  correct_points : ℕ
  incorrect_points : ℕ

/-- Represents a student's test result -/
structure TestResult where
  test : MCTest
  total_score : ℤ

/-- Calculates the number of correctly answered questions -/
def correct_answers (result : TestResult) : ℕ :=
  sorry

/-- Theorem stating the problem and its solution -/
theorem student_correct_answers
  (test : MCTest)
  (result : TestResult)
  (h1 : test.total_questions = 25)
  (h2 : test.correct_points = 4)
  (h3 : test.incorrect_points = 1)
  (h4 : result.test = test)
  (h5 : result.total_score = 85) :
  correct_answers result = 22 := by
  sorry

end NUMINAMATH_CALUDE_student_correct_answers_l1062_106282


namespace NUMINAMATH_CALUDE_fraction_of_books_sold_l1062_106244

/-- Prove that the fraction of books sold is 2/3, given the conditions -/
theorem fraction_of_books_sold (price : ℝ) (unsold : ℕ) (total_received : ℝ) :
  price = 4.25 →
  unsold = 30 →
  total_received = 255 →
  (total_received / price) / ((total_received / price) + unsold) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_books_sold_l1062_106244


namespace NUMINAMATH_CALUDE_max_true_statements_l1062_106259

theorem max_true_statements (x : ℝ) : ∃ (n : ℕ), n ≤ 3 ∧
  n = (Bool.toNat (0 < x^2 ∧ x^2 < 4) +
       Bool.toNat (x^2 > 4) +
       Bool.toNat (-2 < x ∧ x < 0) +
       Bool.toNat (0 < x ∧ x < 2) +
       Bool.toNat (0 < x - x^2 ∧ x - x^2 < 4)) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l1062_106259


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l1062_106213

theorem complex_expression_evaluation :
  ∀ (c d : ℂ), c = 7 - 3*I → d = 2 + 5*I → 3*c - 4*d = 13 - 29*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l1062_106213


namespace NUMINAMATH_CALUDE_pine_cones_on_roof_l1062_106276

/-- Calculates the weight of pine cones on a roof given the number of trees, 
    pine cones per tree, percentage on roof, and weight per pine cone. -/
theorem pine_cones_on_roof 
  (num_trees : ℕ) 
  (cones_per_tree : ℕ) 
  (percent_on_roof : ℚ) 
  (weight_per_cone : ℕ) 
  (h1 : num_trees = 8)
  (h2 : cones_per_tree = 200)
  (h3 : percent_on_roof = 30 / 100)
  (h4 : weight_per_cone = 4) :
  (num_trees * cones_per_tree : ℚ) * percent_on_roof * weight_per_cone = 1920 :=
sorry

end NUMINAMATH_CALUDE_pine_cones_on_roof_l1062_106276


namespace NUMINAMATH_CALUDE_coefficient_x10_expansion_l1062_106299

/-- The coefficient of x^10 in the expansion of ((1+x+x^2)(1-x)^10) is 36 -/
theorem coefficient_x10_expansion : 
  let f : ℕ → ℤ := fun n => 
    (Finset.range (n + 1)).sum (fun k => 
      (-1)^k * (Nat.choose n k) * (Finset.range 3).sum (fun i => Nat.choose 2 i * k^(2-i)))
  f 10 = 36 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x10_expansion_l1062_106299


namespace NUMINAMATH_CALUDE_systematic_sampling_first_two_samples_l1062_106212

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  population_size : Nat
  sample_size : Nat
  last_sampled : Nat

/-- Calculates the interval size for systematic sampling -/
def interval_size (s : SystematicSampling) : Nat :=
  s.population_size / s.sample_size

/-- Calculates the start of the last interval -/
def last_interval_start (s : SystematicSampling) : Nat :=
  s.population_size - (interval_size s) + 1

/-- Calculates the position within an interval -/
def position_in_interval (s : SystematicSampling) : Nat :=
  s.last_sampled - (last_interval_start s) + 1

/-- Calculates the nth sampled number -/
def nth_sample (s : SystematicSampling) (n : Nat) : Nat :=
  (n - 1) * (position_in_interval s)

theorem systematic_sampling_first_two_samples
  (s : SystematicSampling)
  (h1 : s.population_size = 8000)
  (h2 : s.sample_size = 50)
  (h3 : s.last_sampled = 7900) :
  (nth_sample s 1 = 60) ∧ (nth_sample s 2 = 220) := by
  sorry

#check systematic_sampling_first_two_samples

end NUMINAMATH_CALUDE_systematic_sampling_first_two_samples_l1062_106212


namespace NUMINAMATH_CALUDE_gcf_of_4620_and_10780_l1062_106285

theorem gcf_of_4620_and_10780 : Nat.gcd 4620 10780 = 1540 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_4620_and_10780_l1062_106285


namespace NUMINAMATH_CALUDE_phone_number_probability_l1062_106260

/-- The set of possible area codes -/
def areaCodes : Finset ℕ := {407, 410, 415}

/-- The set of digits for the remaining part of the phone number -/
def remainingDigits : Finset ℕ := {0, 1, 2, 3, 4}

/-- The total number of digits in the phone number -/
def totalDigits : ℕ := 8

/-- The number of digits after the area code -/
def remainingDigitsCount : ℕ := 5

theorem phone_number_probability :
  (1 : ℚ) / (areaCodes.card * Nat.factorial remainingDigitsCount) =
  (1 : ℚ) / 360 := by sorry

end NUMINAMATH_CALUDE_phone_number_probability_l1062_106260


namespace NUMINAMATH_CALUDE_oscar_age_is_6_l1062_106280

/-- Christina's age in 5 years -/
def christina_age_in_5_years : ℕ := 80 / 2

/-- Christina's current age -/
def christina_current_age : ℕ := christina_age_in_5_years - 5

/-- Oscar's age in 15 years -/
def oscar_age_in_15_years : ℕ := (3 * christina_current_age) / 5

/-- Oscar's current age -/
def oscar_current_age : ℕ := oscar_age_in_15_years - 15

theorem oscar_age_is_6 : oscar_current_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_oscar_age_is_6_l1062_106280


namespace NUMINAMATH_CALUDE_partition_properties_l1062_106210

/-- P k l n is the number of partitions of n into no more than k parts, each not exceeding l -/
def P (k l n : ℕ) : ℕ := sorry

/-- The four properties of the partition function P -/
theorem partition_properties (k l n : ℕ) :
  (P k l n - P k (l-1) n = P (k-1) l (n-l)) ∧
  (P k l n - P (k-1) l n = P k (l-1) (n-k)) ∧
  (P k l n = P l k n) ∧
  (P k l n = P k l (k*l-n)) := by sorry

end NUMINAMATH_CALUDE_partition_properties_l1062_106210


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1062_106239

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 11 →
  a 3 = 19 →
  a 6 = 43 →
  a 4 + a 5 = 62 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1062_106239


namespace NUMINAMATH_CALUDE_mike_seeds_count_mike_total_seeds_l1062_106294

theorem mike_seeds_count : ℕ → Prop :=
  fun total_seeds =>
    let seeds_left : ℕ := 20
    let seeds_right : ℕ := 2 * seeds_left
    let seeds_joining : ℕ := 30
    let seeds_remaining : ℕ := 30
    total_seeds = seeds_left + seeds_right + seeds_joining + seeds_remaining

theorem mike_total_seeds :
  ∃ (total_seeds : ℕ), mike_seeds_count total_seeds ∧ total_seeds = 120 := by
  sorry

end NUMINAMATH_CALUDE_mike_seeds_count_mike_total_seeds_l1062_106294


namespace NUMINAMATH_CALUDE_count_propositions_is_two_l1062_106263

-- Define a type for statements
inductive Statement
| EmptySetProperSubset
| QuadraticInequality
| PerpendicularLinesQuestion
| NaturalNumberEven

-- Define a function to check if a statement is a proposition
def isProposition (s : Statement) : Bool :=
  match s with
  | Statement.EmptySetProperSubset => true
  | Statement.QuadraticInequality => false
  | Statement.PerpendicularLinesQuestion => false
  | Statement.NaturalNumberEven => true

-- Define a function to count propositions
def countPropositions (statements : List Statement) : Nat :=
  statements.filter isProposition |>.length

-- Theorem to prove
theorem count_propositions_is_two :
  let statements := [Statement.EmptySetProperSubset, Statement.QuadraticInequality,
                     Statement.PerpendicularLinesQuestion, Statement.NaturalNumberEven]
  countPropositions statements = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_propositions_is_two_l1062_106263


namespace NUMINAMATH_CALUDE_coupon_value_l1062_106286

/-- Calculates the value of each coupon given the original price, discount percentage, number of bottles, and total cost after discounts and coupons. -/
theorem coupon_value (original_price discount_percent bottles total_cost : ℝ) : 
  original_price = 15 →
  discount_percent = 20 →
  bottles = 3 →
  total_cost = 30 →
  (original_price * (1 - discount_percent / 100) * bottles - total_cost) / bottles = 2 := by
  sorry

end NUMINAMATH_CALUDE_coupon_value_l1062_106286


namespace NUMINAMATH_CALUDE_exam_results_l1062_106251

theorem exam_results (failed_hindi : ℝ) (failed_english : ℝ) (failed_both : ℝ)
  (h1 : failed_hindi = 34)
  (h2 : failed_english = 44)
  (h3 : failed_both = 22) :
  100 - (failed_hindi + failed_english - failed_both) = 44 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l1062_106251


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1062_106227

-- Define the quadrilateral ABCD
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the point of intersection of diagonals
def P (q : Quadrilateral) : Point := sorry

-- Define the distances from A, B, and P to line CD
def a (q : Quadrilateral) : ℝ := sorry
def b (q : Quadrilateral) : ℝ := sorry
def p (q : Quadrilateral) : ℝ := sorry

-- Define the length of side CD
def CD (q : Quadrilateral) : ℝ := sorry

-- Define the area of the quadrilateral
def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area (q : Quadrilateral) : 
  area q = (a q * b q * CD q) / (2 * p q) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1062_106227


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l1062_106225

def U : Set Int := {-1, 2, 4}
def A : Set Int := {-1, 4}

theorem complement_of_A_wrt_U :
  (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l1062_106225


namespace NUMINAMATH_CALUDE_seed_mixture_percentage_l1062_106235

/-- Given two seed mixtures X and Y, and their combination, 
    this theorem proves that the percentage of mixture X in the final mixture is 1/3. -/
theorem seed_mixture_percentage (x y : ℝ) :
  x ≥ 0 → y ≥ 0 →  -- Ensure non-negative weights
  0.40 * x + 0.25 * y = 0.30 * (x + y) →  -- Ryegrass balance equation
  x / (x + y) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_seed_mixture_percentage_l1062_106235


namespace NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l1062_106278

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- Proves that 12 kilometers per second is equal to 43,200 kilometers per hour -/
theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 12 = 43200 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l1062_106278


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1062_106216

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem perpendicular_vectors (x : ℝ) :
  (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2 = 0) →
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1062_106216


namespace NUMINAMATH_CALUDE_max_value_x2_plus_y2_l1062_106215

theorem max_value_x2_plus_y2 (x y : ℝ) (h : 3 * x^2 + 2 * y^2 = 2 * x) :
  ∃ (M : ℝ), M = 4/9 ∧ x^2 + y^2 ≤ M ∧ ∃ (x₀ y₀ : ℝ), 3 * x₀^2 + 2 * y₀^2 = 2 * x₀ ∧ x₀^2 + y₀^2 = M :=
sorry

end NUMINAMATH_CALUDE_max_value_x2_plus_y2_l1062_106215


namespace NUMINAMATH_CALUDE_quadratic_complex_roots_l1062_106231

theorem quadratic_complex_roots (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ a b : ℝ, b ≠ 0 ∧ (∀ x : ℂ, x^2 + p*x + q = 0 → x = Complex.mk a b ∨ x = Complex.mk a (-b))) →
  (∀ x : ℂ, x^2 + p*x + q = 0 → x.re = 1/2) →
  p = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_complex_roots_l1062_106231
