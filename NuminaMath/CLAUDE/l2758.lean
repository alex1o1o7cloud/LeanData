import Mathlib

namespace NUMINAMATH_CALUDE_digital_earth_capabilities_l2758_275887

-- Define the possible capabilities
inductive Capability
  | ReceiveDistanceEducation
  | ShopOnline
  | SeekMedicalAdviceOnline
  | TravelAroundWorld

-- Define Digital Earth
def DigitalEarth : Type := Set Capability

-- Define the correct set of capabilities
def CorrectCapabilities : Set Capability :=
  {Capability.ReceiveDistanceEducation, Capability.ShopOnline, Capability.SeekMedicalAdviceOnline}

-- Theorem stating that Digital Earth capabilities are exactly the correct ones
theorem digital_earth_capabilities :
  ∃ (de : DigitalEarth), de = CorrectCapabilities :=
sorry

end NUMINAMATH_CALUDE_digital_earth_capabilities_l2758_275887


namespace NUMINAMATH_CALUDE_parabola_equation_from_dot_product_parabola_equation_is_x_squared_eq_8y_l2758_275828

/-- A parabola with vertex at the origin and focus on the positive y-axis -/
structure UpwardParabola where
  focus : ℝ
  focus_positive : 0 < focus

/-- The equation of an upward parabola given its focus -/
def parabola_equation (p : UpwardParabola) (x y : ℝ) : Prop :=
  x^2 = 2 * p.focus * y

/-- A line passing through two points on a parabola -/
structure IntersectingLine (p : UpwardParabola) where
  slope : ℝ
  intersects_parabola : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola_equation p x₁ y₁ ∧
    parabola_equation p x₂ y₂ ∧
    y₁ = slope * x₁ + p.focus ∧
    y₂ = slope * x₂ + p.focus

/-- The main theorem -/
theorem parabola_equation_from_dot_product
  (p : UpwardParabola)
  (l : IntersectingLine p)
  (h : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola_equation p x₁ y₁ ∧
    parabola_equation p x₂ y₂ ∧
    y₁ = l.slope * x₁ + p.focus ∧
    y₂ = l.slope * x₂ + p.focus ∧
    x₁ * x₂ + y₁ * y₂ = -12) :
  p.focus = 4 :=
sorry

/-- The equation of the parabola is x² = 8y -/
theorem parabola_equation_is_x_squared_eq_8y
  (p : UpwardParabola)
  (h : p.focus = 4) :
  ∀ x y, parabola_equation p x y ↔ x^2 = 8*y :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_from_dot_product_parabola_equation_is_x_squared_eq_8y_l2758_275828


namespace NUMINAMATH_CALUDE_angle_on_ray_l2758_275826

/-- Given an angle α where its initial side coincides with the non-negative half-axis of the x-axis
    and its terminal side lies on the ray 4x - 3y = 0 (x ≤ 0), cos α - sin α = 1/5 -/
theorem angle_on_ray (α : Real) : 
  (∃ (x y : Real), x ≤ 0 ∧ 4 * x - 3 * y = 0 ∧ 
   x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
   y = Real.sin α * Real.sqrt (x^2 + y^2)) → 
  Real.cos α - Real.sin α = 1/5 := by
sorry

end NUMINAMATH_CALUDE_angle_on_ray_l2758_275826


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l2758_275816

/-- Given a circle with diameter endpoints (2, 0) and (2, -2), its equation is (x - 2)² + (y + 1)² = 1 -/
theorem circle_equation_from_diameter (x y : ℝ) :
  let endpoint1 : ℝ × ℝ := (2, 0)
  let endpoint2 : ℝ × ℝ := (2, -2)
  (x - 2)^2 + (y + 1)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l2758_275816


namespace NUMINAMATH_CALUDE_b_grazing_months_l2758_275827

/-- Represents the number of months B put his oxen for grazing -/
def b_months : ℕ := sorry

/-- Total rent of the pasture in Rs. -/
def total_rent : ℕ := 280

/-- C's share of the rent in Rs. -/
def c_share : ℕ := 72

/-- Calculates the total oxen-months for all farmers -/
def total_oxen_months : ℕ := 10 * 7 + 12 * b_months + 15 * 3

/-- Theorem stating that B put his oxen for grazing for 5 months -/
theorem b_grazing_months : b_months = 5 := by
  sorry

end NUMINAMATH_CALUDE_b_grazing_months_l2758_275827


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l2758_275883

theorem arctan_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.arctan (2 / x) + Real.arctan (1 / x^2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l2758_275883


namespace NUMINAMATH_CALUDE_complex_modulus_l2758_275877

theorem complex_modulus (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2758_275877


namespace NUMINAMATH_CALUDE_system_solution_l2758_275880

theorem system_solution :
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ),
    (x₁ + x₂ + x₃ = 6) ∧
    (x₂ + x₃ + x₄ = 9) ∧
    (x₃ + x₄ + x₅ = 3) ∧
    (x₄ + x₅ + x₆ = -3) ∧
    (x₅ + x₆ + x₇ = -9) ∧
    (x₆ + x₇ + x₈ = -6) ∧
    (x₇ + x₈ + x₁ = -2) ∧
    (x₈ + x₁ + x₂ = 2) ∧
    x₁ = 1 ∧ x₂ = 2 ∧ x₃ = 3 ∧ x₄ = 4 ∧
    x₅ = -4 ∧ x₆ = -3 ∧ x₇ = -2 ∧ x₈ = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2758_275880


namespace NUMINAMATH_CALUDE_sqrt_16_minus_2_squared_equals_zero_l2758_275834

theorem sqrt_16_minus_2_squared_equals_zero : 
  Real.sqrt 16 - 2^2 = 0 := by sorry

end NUMINAMATH_CALUDE_sqrt_16_minus_2_squared_equals_zero_l2758_275834


namespace NUMINAMATH_CALUDE_compute_expression_l2758_275847

/-- Operation Δ: a Δ b = a × 100...00 (b zeros) + b -/
def delta (a b : ℕ) : ℕ := a * (10^b) + b

/-- Operation □: a □ b = a × 10 + b -/
def square (a b : ℕ) : ℕ := a * 10 + b

/-- Theorem: 2018 □ (123 Δ 4) = 1250184 -/
theorem compute_expression : square 2018 (delta 123 4) = 1250184 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l2758_275847


namespace NUMINAMATH_CALUDE_digit_product_puzzle_l2758_275840

theorem digit_product_puzzle :
  ∀ (A B C D : Nat),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
    (10 * A + B) * (10 * C + B) = 111 * D →
    10 * A + B < 10 * C + B →
    A = 2 ∧ B = 7 ∧ C = 3 ∧ D = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_product_puzzle_l2758_275840


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l2758_275863

/-- 
Given an arithmetic sequence where:
  - The first term is 2
  - The common difference is 4
Prove that the 150th term of this sequence is 598
-/
theorem arithmetic_sequence_150th_term : 
  ∀ (a : ℕ → ℕ), 
  (a 1 = 2) →  -- First term is 2
  (∀ n, a (n + 1) = a n + 4) →  -- Common difference is 4
  a 150 = 598 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l2758_275863


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l2758_275860

theorem greatest_three_digit_number : ∃ n : ℕ,
  n = 989 ∧
  n < 1000 ∧
  ∃ k : ℕ, n = 7 * k + 2 ∧
  ∃ m : ℕ, n = 4 * m + 1 ∧
  ∀ x : ℕ, x < 1000 → (∃ a : ℕ, x = 7 * a + 2) → (∃ b : ℕ, x = 4 * b + 1) → x ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l2758_275860


namespace NUMINAMATH_CALUDE_day_after_53_days_l2758_275841

-- Define the days of the week
inductive DayOfWeek
  | Friday
  | Saturday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday

-- Define a function to advance the day by one
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday

-- Define a function to advance the day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem day_after_53_days : advanceDay DayOfWeek.Friday 53 = DayOfWeek.Tuesday := by
  sorry

end NUMINAMATH_CALUDE_day_after_53_days_l2758_275841


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_twelve_l2758_275896

theorem last_digit_of_one_over_two_to_twelve (n : ℕ) : 
  n = 12 → (1 : ℚ) / (2^n : ℚ) * 10^n % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_twelve_l2758_275896


namespace NUMINAMATH_CALUDE_intended_profit_percentage_l2758_275897

/-- Given a cost price, labeled price, and selling price satisfying certain conditions,
    prove that the intended profit percentage is 1/3. -/
theorem intended_profit_percentage
  (C L S : ℝ)  -- Cost price, Labeled price, Selling price
  (P : ℝ)      -- Intended profit percentage (as a decimal)
  (h1 : L = C * (1 + P))        -- Labeled price condition
  (h2 : S = 0.90 * L)           -- 10% discount condition
  (h3 : S = 1.17 * C)           -- 17% actual profit condition
  : P = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_intended_profit_percentage_l2758_275897


namespace NUMINAMATH_CALUDE_computer_price_increase_l2758_275886

/-- The new price of a computer after a 30% increase, given initial conditions -/
theorem computer_price_increase (b : ℝ) (h : 2 * b = 540) : b * 1.3 = 351 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l2758_275886


namespace NUMINAMATH_CALUDE_total_fish_bought_l2758_275891

theorem total_fish_bought (goldfish : ℕ) (blue_fish : ℕ) (angelfish : ℕ) (neon_tetras : ℕ)
  (h1 : goldfish = 23)
  (h2 : blue_fish = 15)
  (h3 : angelfish = 8)
  (h4 : neon_tetras = 12) :
  goldfish + blue_fish + angelfish + neon_tetras = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_bought_l2758_275891


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2758_275862

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (1 - 1 / (x + 1)) * ((x^2 - 1) / x) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2758_275862


namespace NUMINAMATH_CALUDE_remainder_squared_multiply_l2758_275888

theorem remainder_squared_multiply (n a b : ℤ) : 
  n > 0 → b = 3 → a * b ≡ 1 [ZMOD n] → a^2 * b ≡ a [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_remainder_squared_multiply_l2758_275888


namespace NUMINAMATH_CALUDE_mets_fans_count_l2758_275866

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  dodgers : ℕ
  red_sox : ℕ
  cubs : ℕ

/-- The total number of baseball fans in the town -/
def total_fans : ℕ := 585

/-- Checks if the given fan counts satisfy the specified ratios -/
def satisfies_ratios (fc : FanCounts) : Prop :=
  3 * fc.mets = 2 * fc.yankees ∧
  3 * fc.dodgers = fc.mets ∧
  4 * fc.red_sox = 5 * fc.mets ∧
  2 * fc.cubs = fc.mets

/-- Checks if the sum of all fan counts equals the total number of fans -/
def sums_to_total (fc : FanCounts) : Prop :=
  fc.yankees + fc.mets + fc.dodgers + fc.red_sox + fc.cubs = total_fans

/-- The main theorem stating that there are 120 NY Mets fans -/
theorem mets_fans_count :
  ∃ (fc : FanCounts), satisfies_ratios fc ∧ sums_to_total fc ∧ fc.mets = 120 :=
sorry

end NUMINAMATH_CALUDE_mets_fans_count_l2758_275866


namespace NUMINAMATH_CALUDE_digit_sum_in_multiplication_l2758_275825

theorem digit_sum_in_multiplication (c d : ℕ) : 
  c < 10 → d < 10 → 
  (30 + c) * (10 * d + 5) = 185 →
  5 * c = 15 →
  c + d = 11 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_in_multiplication_l2758_275825


namespace NUMINAMATH_CALUDE_sine_amplitude_l2758_275899

theorem sine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.sin (b * x) ≤ 3) ∧ (∃ x, a * Real.sin (b * x) = 3) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_sine_amplitude_l2758_275899


namespace NUMINAMATH_CALUDE_g_composition_of_3_l2758_275815

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 2*n + 1 else 4*n - 3

theorem g_composition_of_3 : g (g (g 3)) = 241 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_3_l2758_275815


namespace NUMINAMATH_CALUDE_incorrect_arrangements_count_l2758_275858

/-- The number of letters in the word --/
def word_length : ℕ := 4

/-- The total number of possible arrangements of the letters --/
def total_arrangements : ℕ := Nat.factorial word_length

/-- The number of correct arrangements (always 1 for a single word) --/
def correct_arrangements : ℕ := 1

/-- Theorem: The number of incorrect arrangements of a 4-letter word is 23 --/
theorem incorrect_arrangements_count :
  total_arrangements - correct_arrangements = 23 := by sorry

end NUMINAMATH_CALUDE_incorrect_arrangements_count_l2758_275858


namespace NUMINAMATH_CALUDE_laundry_time_proof_l2758_275831

/-- Proves that the time to wash one load of laundry is 45 minutes -/
theorem laundry_time_proof (wash_time : ℕ) : 
  (2 * wash_time + 75 = 165) → wash_time = 45 := by
  sorry

end NUMINAMATH_CALUDE_laundry_time_proof_l2758_275831


namespace NUMINAMATH_CALUDE_arrange_seven_white_five_black_l2758_275870

/-- The number of ways to arrange white and black balls with no adjacent black balls -/
def arrangeBalls (white black : ℕ) : ℕ :=
  Nat.choose (white + black - black + 1) (black + 1)

/-- Theorem stating that arranging 7 white and 5 black balls with no adjacent black balls results in 56 ways -/
theorem arrange_seven_white_five_black :
  arrangeBalls 7 5 = 56 := by
  sorry

#eval arrangeBalls 7 5

end NUMINAMATH_CALUDE_arrange_seven_white_five_black_l2758_275870


namespace NUMINAMATH_CALUDE_power_of_three_and_seven_hundreds_digit_l2758_275875

theorem power_of_three_and_seven_hundreds_digit : 
  ∃ (a b : ℕ), 
    100 ≤ 3^a ∧ 3^a < 1000 ∧
    100 ≤ 7^b ∧ 7^b < 1000 ∧
    (3^a / 100 % 10 = 7) ∧ (7^b / 100 % 10 = 7) := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_and_seven_hundreds_digit_l2758_275875


namespace NUMINAMATH_CALUDE_percent_change_condition_l2758_275876

theorem percent_change_condition (a b r N : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < r ∧ 0 < N ∧ r < 50 →
  (N * (1 + a / 100) * (1 - b / 100) ≤ N * (1 + r / 100) ↔ a - b - a * b / 100 ≤ r) :=
by sorry

end NUMINAMATH_CALUDE_percent_change_condition_l2758_275876


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_number_l2758_275817

def class_size : ℕ := 48
def sample_size : ℕ := 4
def interval : ℕ := class_size / sample_size

def is_valid_sample (s : Finset ℕ) : Prop :=
  s.card = sample_size ∧ 
  ∀ x ∈ s, 1 ≤ x ∧ x ≤ class_size ∧
  ∃ k : ℕ, x = 1 + k * interval

theorem systematic_sample_fourth_number :
  ∀ s : Finset ℕ, is_valid_sample s →
  (5 ∈ s ∧ 29 ∈ s ∧ 41 ∈ s) →
  17 ∈ s :=
by sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_number_l2758_275817


namespace NUMINAMATH_CALUDE_max_parts_three_planes_l2758_275845

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this statement

/-- The number of parts that a set of planes divides 3D space into -/
def num_parts (planes : List Plane3D) : ℕ :=
  sorry -- Definition would go here

/-- The maximum number of parts that three planes can divide 3D space into -/
theorem max_parts_three_planes :
  ∃ (planes : List Plane3D), planes.length = 3 ∧ 
  ∀ (other_planes : List Plane3D), other_planes.length = 3 →
  num_parts other_planes ≤ num_parts planes ∧ num_parts planes = 8 :=
sorry

end NUMINAMATH_CALUDE_max_parts_three_planes_l2758_275845


namespace NUMINAMATH_CALUDE_emerson_rowing_distance_l2758_275894

/-- The total distance covered by Emerson on his rowing trip -/
def total_distance (first_part second_part third_part : ℕ) : ℕ :=
  first_part + second_part + third_part

/-- Theorem stating that Emerson's total rowing distance is 39 miles -/
theorem emerson_rowing_distance :
  total_distance 6 15 18 = 39 := by
  sorry

end NUMINAMATH_CALUDE_emerson_rowing_distance_l2758_275894


namespace NUMINAMATH_CALUDE_expression_equals_one_l2758_275878

theorem expression_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a + b - c = 0) :
  (a^2 * b^2) / ((a^2 + b*c) * (b^2 + a*c)) +
  (a^2 * c^2) / ((a^2 + b*c) * (c^2 + a*b)) +
  (b^2 * c^2) / ((b^2 + a*c) * (c^2 + a*b)) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2758_275878


namespace NUMINAMATH_CALUDE_z_pure_imaginary_iff_a_eq_neg_two_l2758_275864

-- Define the complex number z as a function of real number a
def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)

-- Define what it means for a complex number to be purely imaginary
def is_pure_imaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

-- Theorem statement
theorem z_pure_imaginary_iff_a_eq_neg_two :
  ∀ a : ℝ, is_pure_imaginary (z a) ↔ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_z_pure_imaginary_iff_a_eq_neg_two_l2758_275864


namespace NUMINAMATH_CALUDE_roots_equation_l2758_275842

theorem roots_equation (n r s c d : ℝ) : 
  (c^2 - n*c + 6 = 0) →
  (d^2 - n*d + 6 = 0) →
  ((c^2 + 1/d)^2 - r*(c^2 + 1/d) + s = 0) →
  ((d^2 + 1/c)^2 - r*(d^2 + 1/c) + s = 0) →
  s = n + 217/6 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_l2758_275842


namespace NUMINAMATH_CALUDE_proposition_truth_l2758_275846

theorem proposition_truth (p q : Prop) (hp : p) (hq : ¬q) : (¬p) ∨ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l2758_275846


namespace NUMINAMATH_CALUDE_unique_stutterer_square_l2758_275852

/-- A function that checks if a number is a stutterer (first two digits are the same and last two digits are the same) --/
def is_stutterer (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 = (n / 100) % 10) ∧
  ((n / 10) % 10 = n % 10)

/-- The theorem stating that 7744 is the only four-digit stutterer number that is a perfect square --/
theorem unique_stutterer_square : ∀ n : ℕ, 
  is_stutterer n ∧ ∃ k : ℕ, n = k^2 ↔ n = 7744 :=
sorry

end NUMINAMATH_CALUDE_unique_stutterer_square_l2758_275852


namespace NUMINAMATH_CALUDE_set_operations_l2758_275800

def A : Set ℤ := {x | -6 ≤ x ∧ x ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

theorem set_operations :
  (A ∪ (B ∩ C) = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6}) ∧
  (A ∩ (A \ (B ∩ C)) = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 4, 5, 6}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2758_275800


namespace NUMINAMATH_CALUDE_bus_car_length_ratio_l2758_275832

theorem bus_car_length_ratio : 
  ∀ (red_bus_length orange_car_length yellow_bus_length : ℝ),
  red_bus_length = 4 * orange_car_length →
  red_bus_length = 48 →
  yellow_bus_length = red_bus_length - 6 →
  yellow_bus_length / orange_car_length = 7 / 2 := by
sorry

end NUMINAMATH_CALUDE_bus_car_length_ratio_l2758_275832


namespace NUMINAMATH_CALUDE_c_is_largest_l2758_275803

-- Define the numbers as real numbers
def a : ℝ := 7.25678
def b : ℝ := 7.256777777777777 -- Approximation of 7.256̄7
def c : ℝ := 7.257676767676767 -- Approximation of 7.25̄76
def d : ℝ := 7.275675675675675 -- Approximation of 7.2̄756
def e : ℝ := 7.275627562756275 -- Approximation of 7.̄2756

-- Theorem stating that c (7.25̄76) is the largest
theorem c_is_largest : 
  c > a ∧ c > b ∧ c > d ∧ c > e :=
sorry

end NUMINAMATH_CALUDE_c_is_largest_l2758_275803


namespace NUMINAMATH_CALUDE_rachel_brownies_l2758_275848

def brownies_baked (brought_to_school left_at_home : ℕ) : ℕ :=
  brought_to_school + left_at_home

theorem rachel_brownies : 
  brownies_baked 16 24 = 40 := by
  sorry

end NUMINAMATH_CALUDE_rachel_brownies_l2758_275848


namespace NUMINAMATH_CALUDE_income_comparison_l2758_275861

/-- Given that Mart's income is 60% more than Tim's income, and Tim's income is 50% less than Juan's income, 
    prove that Mart's income is 80% of Juan's income. -/
theorem income_comparison (tim juan mart : ℝ) 
  (h1 : mart = tim + 0.6 * tim) 
  (h2 : tim = juan - 0.5 * juan) : 
  mart = 0.8 * juan := by sorry

end NUMINAMATH_CALUDE_income_comparison_l2758_275861


namespace NUMINAMATH_CALUDE_max_value_trig_product_max_value_trig_product_achievable_l2758_275811

theorem max_value_trig_product (x y z : ℝ) :
  (Real.sin x + Real.sin (2 * y) + Real.sin (3 * z)) *
  (Real.cos x + Real.cos (2 * y) + Real.cos (3 * z)) ≤ 4.5 :=
by sorry

theorem max_value_trig_product_achievable :
  ∃ x y z : ℝ,
    (Real.sin x + Real.sin (2 * y) + Real.sin (3 * z)) *
    (Real.cos x + Real.cos (2 * y) + Real.cos (3 * z)) = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_product_max_value_trig_product_achievable_l2758_275811


namespace NUMINAMATH_CALUDE_students_above_eight_l2758_275814

theorem students_above_eight (total : ℕ) (below_eight : ℕ) (eight : ℕ) (above_eight : ℕ) : 
  total = 80 →
  below_eight = total / 4 →
  eight = 36 →
  above_eight = 2 * eight / 3 →
  above_eight = 24 := by
sorry

end NUMINAMATH_CALUDE_students_above_eight_l2758_275814


namespace NUMINAMATH_CALUDE_tucker_tissues_left_l2758_275807

/-- The number of tissues left after buying boxes and using some. -/
def tissues_left (tissues_per_box : ℕ) (boxes_bought : ℕ) (tissues_used : ℕ) : ℕ :=
  tissues_per_box * boxes_bought - tissues_used

/-- Theorem: Given 160 tissues per box, 3 boxes bought, and 210 tissues used, 270 tissues are left. -/
theorem tucker_tissues_left :
  tissues_left 160 3 210 = 270 := by
  sorry

end NUMINAMATH_CALUDE_tucker_tissues_left_l2758_275807


namespace NUMINAMATH_CALUDE_abs_fraction_inequality_l2758_275889

theorem abs_fraction_inequality (x : ℝ) :
  |((3 - x) / 4)| < 1 ↔ 2 < x ∧ x < 7 := by sorry

end NUMINAMATH_CALUDE_abs_fraction_inequality_l2758_275889


namespace NUMINAMATH_CALUDE_domain_of_f_x_squared_l2758_275869

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_x_plus_1 (f : ℝ → ℝ) : Set ℝ := Set.Icc (-2) 3

-- Define the property that f(x+1) has domain [-2, 3]
def f_x_plus_1_domain (f : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ domain_f_x_plus_1 f ↔ f (x + 1) ≠ 0

-- Theorem statement
theorem domain_of_f_x_squared (f : ℝ → ℝ) 
  (h : f_x_plus_1_domain f) : 
  {x : ℝ | f (x^2) ≠ 0} = Set.Icc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_domain_of_f_x_squared_l2758_275869


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2758_275895

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 - a*b + b^2 = 0) :
  (a^6 + b^6) / (a + b)^6 = 1/18 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2758_275895


namespace NUMINAMATH_CALUDE_cubic_sum_ratio_l2758_275836

theorem cubic_sum_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^6 + y^6 + z^6) / (x*y*z * (x*y + x*z + y*z)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_ratio_l2758_275836


namespace NUMINAMATH_CALUDE_sunflower_seed_contest_l2758_275865

/-- The total number of seeds eaten by five players in a sunflower seed eating contest -/
def total_seeds (player1 player2 player3 player4 player5 : ℕ) : ℕ :=
  player1 + player2 + player3 + player4 + player5

/-- Theorem stating the total number of seeds eaten by the five players -/
theorem sunflower_seed_contest : ∃ (player1 player2 player3 player4 player5 : ℕ),
  player1 = 78 ∧
  player2 = 53 ∧
  player3 = player2 + 30 ∧
  player4 = 2 * player3 ∧
  player5 = (player1 + player2 + player3 + player4) / 4 ∧
  total_seeds player1 player2 player3 player4 player5 = 475 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_seed_contest_l2758_275865


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l2758_275867

theorem smaller_number_in_ratio (a b c d x y : ℝ) : 
  0 < a → a < b → 0 < d → d < c →
  x > 0 → y > 0 →
  x / y = a / b →
  x + y = c - d →
  d = 2 * x - y →
  min x y = (2 * a * c - b * c) / (3 * (2 * a - b)) := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l2758_275867


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_plus_ten_l2758_275890

theorem gcf_lcm_sum_plus_ten (a b : ℕ) (h1 : a = 8) (h2 : b = 12) :
  Nat.gcd a b + Nat.lcm a b + 10 = 38 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_plus_ten_l2758_275890


namespace NUMINAMATH_CALUDE_coins_value_l2758_275813

/-- Represents the total value of coins in cents -/
def total_value (total_coins : ℕ) (nickels : ℕ) : ℕ :=
  let dimes : ℕ := total_coins - nickels
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  nickels * nickel_value + dimes * dime_value

/-- Proves that given 50 total coins, 30 of which are nickels, the total value is $3.50 -/
theorem coins_value : total_value 50 30 = 350 := by
  sorry

end NUMINAMATH_CALUDE_coins_value_l2758_275813


namespace NUMINAMATH_CALUDE_divisibility_condition_l2758_275837

def a (n : ℕ) : ℕ := 3 * 4^n

theorem divisibility_condition (n : ℕ) :
  (∀ m : ℕ, 1992 ∣ (m^(a n + 6) - m^(a n + 4) - m^5 + m^3)) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2758_275837


namespace NUMINAMATH_CALUDE_area_ratio_triangle_to_hexagon_l2758_275853

/-- A regular hexagon with vertices labeled A, B, C, D, E, F. -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- The area of a regular hexagon. -/
def area_hexagon (h : RegularHexagon) : ℝ := sorry

/-- Triangle formed by connecting every second vertex of the hexagon. -/
def triangle_ACE (h : RegularHexagon) : sorry := sorry

/-- The area of triangle ACE. -/
def area_triangle_ACE (h : RegularHexagon) : ℝ := sorry

/-- Theorem stating that the ratio of the area of triangle ACE to the area of the regular hexagon is 2/3. -/
theorem area_ratio_triangle_to_hexagon (h : RegularHexagon) :
  (area_triangle_ACE h) / (area_hexagon h) = 2/3 := by sorry

end NUMINAMATH_CALUDE_area_ratio_triangle_to_hexagon_l2758_275853


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2758_275879

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ = 129 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2758_275879


namespace NUMINAMATH_CALUDE_pyramid_volume_l2758_275881

theorem pyramid_volume (base_length : Real) (base_width : Real) (height : Real) :
  base_length = 1 → base_width = 1/4 → height = 1 →
  (1/3) * (base_length * base_width) * height = 1/12 := by
sorry

end NUMINAMATH_CALUDE_pyramid_volume_l2758_275881


namespace NUMINAMATH_CALUDE_ellipse_k_value_l2758_275885

/-- An ellipse with equation x^2 + ky^2 = 1, where k is a positive real number -/
structure Ellipse (k : ℝ) : Type :=
  (eq : ∀ x y : ℝ, x^2 + k * y^2 = 1)

/-- The focus of the ellipse is on the y-axis -/
def focus_on_y_axis (e : Ellipse k) : Prop :=
  k < 1

/-- The length of the major axis is twice that of the minor axis -/
def major_axis_twice_minor (e : Ellipse k) : Prop :=
  2 * (1 / Real.sqrt k) = 4

/-- Theorem: For an ellipse with the given properties, k = 1/4 -/
theorem ellipse_k_value (k : ℝ) (e : Ellipse k) 
  (h1 : focus_on_y_axis e) (h2 : major_axis_twice_minor e) : k = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_value_l2758_275885


namespace NUMINAMATH_CALUDE_square_root_calculations_l2758_275830

theorem square_root_calculations :
  (3 * Real.sqrt 8 - Real.sqrt 32 = 2 * Real.sqrt 2) ∧
  (Real.sqrt 6 * Real.sqrt 2 / Real.sqrt 3 = 2) ∧
  ((Real.sqrt 24 + Real.sqrt (1/6)) / Real.sqrt 3 = 13 * Real.sqrt 2 / 6) ∧
  (Real.sqrt 27 - Real.sqrt 12) / Real.sqrt 3 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_root_calculations_l2758_275830


namespace NUMINAMATH_CALUDE_star_associative_l2758_275812

variable {U : Type*}

def star (X Y : Set U) : Set U := X ∩ Y

theorem star_associative (X Y Z : Set U) : star (star X Y) Z = (X ∩ Y) ∩ Z := by
  sorry

end NUMINAMATH_CALUDE_star_associative_l2758_275812


namespace NUMINAMATH_CALUDE_dress_design_combinations_l2758_275820

theorem dress_design_combinations (num_colors num_patterns : ℕ) :
  num_colors = 4 →
  num_patterns = 5 →
  num_colors * num_patterns = 20 :=
by sorry

end NUMINAMATH_CALUDE_dress_design_combinations_l2758_275820


namespace NUMINAMATH_CALUDE_cos_12_18_minus_sin_12_18_l2758_275819

theorem cos_12_18_minus_sin_12_18 :
  Real.cos (12 * π / 180) * Real.cos (18 * π / 180) - 
  Real.sin (12 * π / 180) * Real.sin (18 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_12_18_minus_sin_12_18_l2758_275819


namespace NUMINAMATH_CALUDE_round_trip_speed_ratio_l2758_275824

/-- Proves that given a round trip with specific conditions, the ratio of return speed to outward speed is 2 --/
theorem round_trip_speed_ratio
  (distance : ℝ)
  (total_time : ℝ)
  (return_speed : ℝ)
  (h_distance : distance = 35)
  (h_total_time : total_time = 6)
  (h_return_speed : return_speed = 17.5)
  : (return_speed / ((2 * distance) / total_time - return_speed)) = 2 := by
  sorry

#check round_trip_speed_ratio

end NUMINAMATH_CALUDE_round_trip_speed_ratio_l2758_275824


namespace NUMINAMATH_CALUDE_tan_two_alpha_l2758_275839

theorem tan_two_alpha (α β : ℝ) (h1 : Real.tan (α - β) = -3/2) (h2 : Real.tan (α + β) = 3) :
  Real.tan (2 * α) = 3/11 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_alpha_l2758_275839


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l2758_275844

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l2758_275844


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2758_275851

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 5*x + 15/2*a > 0) ↔ a > 5/6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2758_275851


namespace NUMINAMATH_CALUDE_common_tangent_sum_l2758_275882

/-- Given two functions f and g with a common tangent at (0, m), prove a + b = 1 -/
theorem common_tangent_sum (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.cos x
  let g : ℝ → ℝ := λ x ↦ x^2 + b*x + 1
  let f' : ℝ → ℝ := λ x ↦ -a * Real.sin x
  let g' : ℝ → ℝ := λ x ↦ 2*x + b
  (∃ m : ℝ, f 0 = m ∧ g 0 = m ∧ f' 0 = g' 0) →
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l2758_275882


namespace NUMINAMATH_CALUDE_three_number_sum_l2758_275868

theorem three_number_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  (a + b + c) / 3 = a + 5 → 
  (a + b + c) / 3 = c - 20 → 
  b = 10 → 
  a + b + c = -15 := by
  sorry

end NUMINAMATH_CALUDE_three_number_sum_l2758_275868


namespace NUMINAMATH_CALUDE_soccer_expansion_l2758_275801

/-- The total number of kids playing soccer after expansion -/
def total_kids (initial : ℕ) (friends_per_kid : ℕ) : ℕ :=
  initial + initial * friends_per_kid

/-- Theorem stating that with 14 initial kids and 3 friends per kid, the total is 56 -/
theorem soccer_expansion : total_kids 14 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_soccer_expansion_l2758_275801


namespace NUMINAMATH_CALUDE_complete_square_factorization_l2758_275823

theorem complete_square_factorization (a : ℝ) :
  a^2 - 6*a + 8 = (a - 4) * (a - 2) := by sorry

end NUMINAMATH_CALUDE_complete_square_factorization_l2758_275823


namespace NUMINAMATH_CALUDE_sum_of_threes_plus_product_of_fours_l2758_275809

theorem sum_of_threes_plus_product_of_fours (m n : ℕ) :
  (List.replicate m 3).sum + (List.replicate n 4).prod = 3 * m + 4^n := by
  sorry

end NUMINAMATH_CALUDE_sum_of_threes_plus_product_of_fours_l2758_275809


namespace NUMINAMATH_CALUDE_cookie_number_proof_l2758_275804

/-- The smallest positive integer satisfying the given conditions -/
def smallest_cookie_number : ℕ := 2549

/-- Proof that the smallest_cookie_number satisfies all conditions -/
theorem cookie_number_proof :
  smallest_cookie_number % 6 = 5 ∧
  smallest_cookie_number % 8 = 6 ∧
  smallest_cookie_number % 10 = 9 ∧
  ∃ k : ℕ, k * k = smallest_cookie_number ∧
  ∀ n : ℕ, n > 0 ∧ n < smallest_cookie_number →
    ¬(n % 6 = 5 ∧ n % 8 = 6 ∧ n % 10 = 9 ∧ ∃ m : ℕ, m * m = n) :=
by sorry

end NUMINAMATH_CALUDE_cookie_number_proof_l2758_275804


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_angles_l2758_275857

-- Define a cyclic quadrilateral
def CyclicQuadrilateral (a b c d : ℝ) : Prop :=
  a + c = 180 ∧ b + d = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

-- Define an arithmetic progression
def ArithmeticProgression (a b c d : ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ b - a = r ∧ c - b = r ∧ d - c = r

-- Define a geometric progression
def GeometricProgression (a b c d : ℝ) : Prop :=
  ∃ (q : ℝ), q ≠ 1 ∧ b / a = q ∧ c / b = q ∧ d / c = q

theorem cyclic_quadrilateral_angles :
  (∃ (a b c d : ℝ), CyclicQuadrilateral a b c d ∧ ArithmeticProgression a b c d) ∧
  (¬ ∃ (a b c d : ℝ), CyclicQuadrilateral a b c d ∧ GeometricProgression a b c d) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_angles_l2758_275857


namespace NUMINAMATH_CALUDE_greatest_three_digit_base_8_divisible_by_7_l2758_275808

def base_8_to_decimal (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

def is_three_digit_base_8 (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_three_digit_base_8_divisible_by_7 :
  ∀ n : Nat, is_three_digit_base_8 n → (base_8_to_decimal n) % 7 = 0 →
  n ≤ 777 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_base_8_divisible_by_7_l2758_275808


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2758_275871

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (q : ℚ), ∀ (n : ℕ), a (n + 1) = a n * q

theorem geometric_sequence_sum 
  (a : ℕ → ℚ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 2 * a 5 = -3/4) 
  (h_sum : a 2 + a 3 + a 4 + a 5 = 5/4) : 
  1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 = -5/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2758_275871


namespace NUMINAMATH_CALUDE_special_function_value_l2758_275855

/-- A function satisfying certain properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 1) = f (1 - x)) ∧ 
  (∀ x, f (x + 2) = f (x + 1) - f x) ∧
  (f 1 = 1/2)

/-- Theorem stating that for any function satisfying the special properties, f(2024) = 1/4 -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) : f 2024 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l2758_275855


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l2758_275893

theorem square_garden_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 200 →
  side^2 = area →
  perimeter = 4 * side →
  perimeter = 40 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l2758_275893


namespace NUMINAMATH_CALUDE_board_number_remainder_l2758_275843

theorem board_number_remainder (n a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  n = 103 * c + d ∧ 
  a + d = 20 ∧ 
  b < 102 →
  b = 20 := by sorry

end NUMINAMATH_CALUDE_board_number_remainder_l2758_275843


namespace NUMINAMATH_CALUDE_valid_grid_count_l2758_275859

/-- A type representing a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → Nat

/-- Checks if a grid is valid according to the problem rules -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j, j < 2 → g i j < g i (j+1)) ∧  -- rows in ascending order
  (∀ i j, i < 2 → g i j < g (i+1) j) ∧  -- columns in ascending order
  (∀ i j, g i j ∈ Finset.range 9) ∧     -- numbers from 1 to 9
  (g 0 0 = 1) ∧ (g 1 1 = 4) ∧ (g 2 2 = 9)  -- pre-filled numbers

/-- The set of all valid grids -/
def valid_grids : Finset Grid :=
  sorry

theorem valid_grid_count : Finset.card valid_grids = 12 := by
  sorry

end NUMINAMATH_CALUDE_valid_grid_count_l2758_275859


namespace NUMINAMATH_CALUDE_thirtieth_triangular_and_sum_l2758_275806

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem thirtieth_triangular_and_sum :
  (triangular_number 30 = 465) ∧
  (triangular_number 30 + triangular_number 31 = 961) := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_and_sum_l2758_275806


namespace NUMINAMATH_CALUDE_donna_additional_flyers_eq_five_l2758_275898

/-- The number of flyers Maisie dropped off -/
def maisie_flyers : ℕ := 33

/-- The total number of flyers Donna dropped off -/
def donna_total_flyers : ℕ := 71

/-- The number of additional flyers Donna dropped off -/
def donna_additional_flyers : ℕ := donna_total_flyers - 2 * maisie_flyers

theorem donna_additional_flyers_eq_five : donna_additional_flyers = 5 := by
  sorry

end NUMINAMATH_CALUDE_donna_additional_flyers_eq_five_l2758_275898


namespace NUMINAMATH_CALUDE_smallest_beta_l2758_275892

theorem smallest_beta (α β : ℕ) (h1 : α > 0) (h2 : β > 0) 
  (h3 : (16 : ℚ) / 37 < α / β) (h4 : α / β < (7 : ℚ) / 16) : 
  (∀ γ : ℕ, γ > 0 → ∃ δ : ℕ, δ > 0 ∧ (16 : ℚ) / 37 < δ / γ ∧ δ / γ < (7 : ℚ) / 16 → γ ≥ 23) ∧ 
  (∃ ε : ℕ, ε > 0 ∧ (16 : ℚ) / 37 < ε / 23 ∧ ε / 23 < (7 : ℚ) / 16) :=
sorry

end NUMINAMATH_CALUDE_smallest_beta_l2758_275892


namespace NUMINAMATH_CALUDE_train_length_l2758_275884

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 40 → time = 27 → speed * time * (1000 / 3600) = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2758_275884


namespace NUMINAMATH_CALUDE_election_votes_l2758_275856

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ) :
  total_votes = 7500 →
  invalid_percent = 1/5 →
  winner_percent = 11/20 →
  ∃ (other_candidate_votes : ℕ), other_candidate_votes = 2700 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l2758_275856


namespace NUMINAMATH_CALUDE_triangle_side_length_l2758_275873

theorem triangle_side_length (b : ℝ) (B : ℝ) (A : ℝ) (a : ℝ) :
  b = 5 → B = π / 4 → Real.sin A = 1 / 3 → a = 5 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2758_275873


namespace NUMINAMATH_CALUDE_product_of_odot_l2758_275810

def A : Finset Int := {-2, 1}
def B : Finset Int := {-1, 2}

def odot (A B : Finset Int) : Finset Int :=
  (A.product B).image (fun (x : Int × Int) => x.1 * x.2)

theorem product_of_odot :
  (odot A B).prod id = 8 := by sorry

end NUMINAMATH_CALUDE_product_of_odot_l2758_275810


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2758_275849

/-- An arithmetic sequence with common difference d ≠ 0 -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y * y = x * z

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence (a 5) (a 9) (a 15)) :
  a 9 / a 5 = 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2758_275849


namespace NUMINAMATH_CALUDE_power_function_even_l2758_275818

-- Define the power function f
def f (x : ℝ) : ℝ := x^(2/3)

-- Theorem statement
theorem power_function_even : 
  (f 8 = 4) → (∀ x : ℝ, f (-x) = f x) :=
by
  sorry

end NUMINAMATH_CALUDE_power_function_even_l2758_275818


namespace NUMINAMATH_CALUDE_money_distribution_l2758_275854

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 1000)
  (ac_sum : A + C = 700)
  (bc_sum : B + C = 600) :
  C = 300 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l2758_275854


namespace NUMINAMATH_CALUDE_square_field_diagonal_l2758_275872

theorem square_field_diagonal (area : ℝ) (diagonal : ℝ) : 
  area = 50 → diagonal = 10 := by sorry

end NUMINAMATH_CALUDE_square_field_diagonal_l2758_275872


namespace NUMINAMATH_CALUDE_total_cost_of_bottle_caps_l2758_275833

/-- The cost of a single bottle cap in dollars -/
def bottle_cap_cost : ℕ := 2

/-- The number of bottle caps -/
def num_bottle_caps : ℕ := 6

/-- Theorem: The total cost of 6 bottle caps is $12 -/
theorem total_cost_of_bottle_caps : 
  bottle_cap_cost * num_bottle_caps = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_bottle_caps_l2758_275833


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_sum_l2758_275805

theorem rectangular_prism_diagonals_sum 
  (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 50) 
  (h2 : x*y + y*z + z*x = 47) : 
  4 * Real.sqrt (x^2 + y^2 + z^2) = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_sum_l2758_275805


namespace NUMINAMATH_CALUDE_empty_subset_of_A_l2758_275874

def A : Set ℤ := {x | 0 < x ∧ x < 3}

theorem empty_subset_of_A : ∅ ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_empty_subset_of_A_l2758_275874


namespace NUMINAMATH_CALUDE_sequence_difference_sum_l2758_275835

-- Define the arithmetic sequences
def seq1 : List Nat := List.range 93 |>.map (fun i => i + 1981)
def seq2 : List Nat := List.range 93 |>.map (fun i => i + 201)

-- Define the sum of each sequence
def sum1 : Nat := seq1.sum
def sum2 : Nat := seq2.sum

-- Theorem statement
theorem sequence_difference_sum : sum1 - sum2 = 165540 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_sum_l2758_275835


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l2758_275829

theorem exam_maximum_marks :
  ∀ (max_marks : ℕ) (paul_marks : ℕ) (failing_margin : ℕ),
    paul_marks = 50 →
    failing_margin = 10 →
    paul_marks + failing_margin = max_marks / 2 →
    max_marks = 120 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l2758_275829


namespace NUMINAMATH_CALUDE_net_calorie_deficit_l2758_275821

/-- Calculates the net calorie deficit for a round trip walk and candy bar consumption. -/
theorem net_calorie_deficit
  (distance : ℝ)
  (calorie_burn_rate : ℝ)
  (candy_bar_calories : ℝ)
  (h1 : distance = 3)
  (h2 : calorie_burn_rate = 150)
  (h3 : candy_bar_calories = 200) :
  distance * calorie_burn_rate - candy_bar_calories = 250 :=
by sorry

end NUMINAMATH_CALUDE_net_calorie_deficit_l2758_275821


namespace NUMINAMATH_CALUDE_existence_of_special_number_l2758_275850

theorem existence_of_special_number (P : Finset Nat) (h_prime : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : Nat,
    (∀ p ∈ P, ∃ a b : Nat, a > 0 ∧ b > 0 ∧ x = a^p + b^p) ∧
    (∀ p : Nat, p ∉ P → Nat.Prime p → ¬∃ a b : Nat, a > 0 ∧ b > 0 ∧ x = a^p + b^p) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l2758_275850


namespace NUMINAMATH_CALUDE_sin_300_degrees_l2758_275802

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l2758_275802


namespace NUMINAMATH_CALUDE_system_is_linear_l2758_275822

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants and at least one of a or b is non-zero. --/
def IsLinearEquation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y = a * x + b * y - c

/-- A system of two linear equations --/
def IsSystemOfTwoLinearEquations (f g : ℝ → ℝ → ℝ) : Prop :=
  IsLinearEquation f ∧ IsLinearEquation g

/-- The given system of equations --/
def f (x y : ℝ) : ℝ := 4 * x - y - 1
def g (x y : ℝ) : ℝ := y - 2 * x - 3

theorem system_is_linear : IsSystemOfTwoLinearEquations f g := by
  sorry

end NUMINAMATH_CALUDE_system_is_linear_l2758_275822


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l2758_275838

-- Define the points
def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (4, 8)
def C : ℝ → ℝ × ℝ := λ x => (5, x)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r.1 - p.1 = t * (q.1 - p.1) ∧ r.2 - p.2 = t * (q.2 - p.2)

-- Theorem statement
theorem collinear_points_x_value :
  ∀ x : ℝ, collinear A B (C x) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_x_value_l2758_275838
