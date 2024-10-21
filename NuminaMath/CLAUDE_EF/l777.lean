import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_monomials_make_perfect_square_l777_77753

-- Define a polynomial type
def MyPolynomial (α : Type*) := List (ℕ × α)

-- Define a monomial type
def MyMonomial (α : Type*) := ℕ × α

-- Function to check if a polynomial is a perfect square binomial
def is_perfect_square_binomial (p : MyPolynomial ℝ) : Prop :=
  ∃ a b : ℝ, p = [(2, a^2), (1, 2*a*b), (0, b^2)]

-- The original polynomial
def original_poly : MyPolynomial ℝ := [(2, 4), (0, 1)]

-- Theorem statement
theorem three_monomials_make_perfect_square :
  ∃! (monomials : Finset (MyMonomial ℝ)),
    (∀ m ∈ monomials, is_perfect_square_binomial (m :: original_poly)) ∧
    monomials.card = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_monomials_make_perfect_square_l777_77753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_is_thursday_l777_77774

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Check if a number is prime -/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m, 1 < m → m < n → ¬(n % m = 0)

/-- Theorem: In a month where three Sundays fall on prime-numbered dates, 
    the 7th day of the month is a Thursday -/
theorem seventh_is_thursday 
  (month : List Date) 
  (three_prime_sundays : ∃ (d1 d2 d3 : Date), 
    d1 ∈ month ∧ d2 ∈ month ∧ d3 ∈ month ∧
    d1.dayOfWeek = DayOfWeek.Sunday ∧ 
    d2.dayOfWeek = DayOfWeek.Sunday ∧ 
    d3.dayOfWeek = DayOfWeek.Sunday ∧
    isPrime d1.day ∧ isPrime d2.day ∧ isPrime d3.day ∧
    d1.day ≠ d2.day ∧ d2.day ≠ d3.day ∧ d1.day ≠ d3.day) :
  ∃ (d : Date), d ∈ month ∧ d.day = 7 ∧ d.dayOfWeek = DayOfWeek.Thursday := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_is_thursday_l777_77774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l777_77722

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log10 (a * x - 3)

theorem max_k_value (a : ℝ) (h : f a 2 = 0) :
  (∃ k : ℕ, k > 0 ∧ 
    (∃ x : ℝ, x ∈ Set.Icc 3 4 ∧ 2 * f a x > log10 (k * x^2)) ∧
    (∀ m : ℕ, m > k → 
      ∀ x : ℝ, x ∈ Set.Icc 3 4 → 2 * f a x ≤ log10 (m * x^2))) →
  (∃ k : ℕ, k = 1 ∧
    (∃ x : ℝ, x ∈ Set.Icc 3 4 ∧ 2 * f a x > log10 (k * x^2)) ∧
    (∀ m : ℕ, m > k → 
      ∀ x : ℝ, x ∈ Set.Icc 3 4 → 2 * f a x ≤ log10 (m * x^2))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l777_77722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_grid_l777_77750

/-- A type representing a 2011 x 2011 grid filled with numbers from 1 to 2011^2 -/
def Grid := Fin 2011 → Fin 2011 → Fin (2011^2)

/-- Predicate checking if a grid is valid (numbers increase strictly across rows and down columns) -/
def is_valid_grid (g : Grid) : Prop :=
  ∀ (i j k l : Fin 2011), i < j → g i k < g j k ∧
  ∀ (i j k l : Fin 2011), k < l → g i k < g i l

/-- Predicate checking if two grids are different -/
def are_different_grids (g1 g2 : Grid) : Prop :=
  ∃ (i j : Fin 2011), g1 i j ≠ g2 i j

/-- Predicate checking if two grids have two integers in the same column of one grid that occur in the same row of the other grid -/
def have_matching_integers (g1 g2 : Grid) : Prop :=
  ∃ (i j k l : Fin 2011), i ≠ k ∧ g1 i j = g2 k l ∧ g1 i l = g2 k j

/-- The main theorem -/
theorem existence_of_special_grid :
  ∃ (special_grid : Grid),
    is_valid_grid special_grid ∧
    ∀ (other_grid : Grid),
      is_valid_grid other_grid →
      are_different_grids special_grid other_grid →
      have_matching_integers special_grid other_grid :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_grid_l777_77750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_and_range_properties_l777_77715

variable (x : Fin 6 → ℝ)

def is_ordered (x : Fin 6 → ℝ) : Prop :=
  ∀ i j : Fin 6, i < j → x i ≤ x j

noncomputable def median (x : Fin 6 → ℝ) : ℝ :=
  (x 2 + x 3) / 2

noncomputable def range (x : Fin 6 → ℝ) : ℝ :=
  x 5 - x 0

noncomputable def subset_median (x : Fin 6 → ℝ) : ℝ :=
  (x 1 + x 2) / 2

noncomputable def subset_range (x : Fin 6 → ℝ) : ℝ :=
  x 4 - x 1

theorem median_and_range_properties
  (h : is_ordered x) :
  subset_median x = median x ∧
  subset_range x ≤ range x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_and_range_properties_l777_77715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_palindromes_l777_77787

/-- A function that checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Bool :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100 = n % 10)

/-- A function that checks if a number is between 200 and 600 -/
def isBetween200And600 (n : ℕ) : Bool :=
  200 ≤ n ∧ n < 600

/-- A function that checks if the middle digit of a three-digit number is even -/
def hasEvenMiddleDigit (n : ℕ) : Bool :=
  ((n / 10) % 10) % 2 = 0

/-- The main theorem stating that there are 20 three-digit palindromes
    between 200 and 600 with an even middle digit -/
theorem count_palindromes : 
  (Finset.filter (λ n : ℕ => isThreeDigitPalindrome n ∧ 
                              isBetween200And600 n ∧ 
                              hasEvenMiddleDigit n) 
                 (Finset.range 1000)).card = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_palindromes_l777_77787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_perpendicular_lines_l777_77768

-- Define the ellipse C₁
noncomputable def C₁ (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle C₂
noncomputable def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the eccentricity of an ellipse
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the area of a quadrilateral given its diagonals and the sine of the angle between them
noncomputable def quadrilateralArea (d1 d2 sinθ : ℝ) : ℝ := (1/2) * d1 * d2 * sinθ

theorem ellipse_and_perpendicular_lines (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 2*a = 4) 
  (h4 : eccentricity a b = 1/2) (k : ℝ) (h5 : k > 0) :
  (∀ x y, C₁ x y a b ↔ x^2/4 + y^2/3 = 1) ∧ 
  (∃ AB CD sinθ, quadrilateralArea AB CD sinθ = (12/7) * Real.sqrt 14 → k = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_perpendicular_lines_l777_77768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_right_if_altitude_ratio_condition_l777_77739

/-- Predicate indicating that h₁, h₂, and h₃ are altitudes of a triangle -/
def IsAltitude (h₁ h₂ h₃ : ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    h₁ * a = h₂ * b ∧ h₁ * a = h₃ * c ∧
    a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate indicating that a triangle with altitudes h₁, h₂, and h₃ is right -/
def IsRightTriangle (h₁ h₂ h₃ : ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    h₁ * a = h₂ * b ∧ h₁ * a = h₃ * c ∧
    a^2 = b^2 + c^2

/-- A triangle with altitudes h₁, h₂, and h₃ is right if (h₁/h₂)² + (h₁/h₃)² = 1 -/
theorem triangle_is_right_if_altitude_ratio_condition 
  (h₁ h₂ h₃ : ℝ) 
  (h_positive : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0)
  (h_altitudes : IsAltitude h₁ h₂ h₃)
  (h_condition : (h₁/h₂)^2 + (h₁/h₃)^2 = 1) :
  IsRightTriangle h₁ h₂ h₃ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_right_if_altitude_ratio_condition_l777_77739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_with_unit_modulus_l777_77710

theorem sum_of_roots_with_unit_modulus : ∃ (S : Finset ℂ), 
  (∀ z ∈ S, z^2009 + z^2008 + 1 = 0 ∧ Complex.abs z = 1) ∧
  (∀ z : ℂ, z^2009 + z^2008 + 1 = 0 ∧ Complex.abs z = 1 → z ∈ S) ∧
  S.sum id = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_with_unit_modulus_l777_77710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_arithmetic_l777_77763

noncomputable def geometricSequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

noncomputable def sumGeometricSequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_arithmetic (a₁ : ℝ) (q : ℝ) :
  (∀ n : ℕ, 2 * (sumGeometricSequence a₁ q (n + 1)) = 
    (sumGeometricSequence a₁ q n) + (sumGeometricSequence a₁ q (n + 2))) →
  q = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_arithmetic_l777_77763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l777_77711

-- Define the points A and B
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (8, 6)

-- Define the reflection line y = mx + b
def reflection_line (m b : ℝ) (x : ℝ) : ℝ := m * x + b

-- State that B is the reflection of A across the line y = mx + b
def is_reflection (m b : ℝ) : Prop :=
  ∃ (x y : ℝ), reflection_line m b x = y ∧ 
  ((x - A.1)^2 + (y - A.2)^2 = (B.1 - x)^2 + (B.2 - y)^2) ∧
  2 * x = A.1 + B.1 ∧ 2 * y = A.2 + B.2

-- Theorem to prove
theorem reflection_sum : 
  ∃ (m b : ℝ), is_reflection m b ∧ m + b = 12.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l777_77711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_rods_l777_77775

theorem quadrilateral_rods (rods : Finset ℕ) : 
  rods = Finset.range 40 →
  (5 ∈ rods ∧ 10 ∈ rods ∧ 20 ∈ rods) →
  (Finset.card (rods.filter (λ x ↦ x ≠ 5 ∧ x ≠ 10 ∧ x ≠ 20 ∧ 
    5 + 10 + 20 > x ∧ x > 20 - (5 + 10) ∧ x > 0))) = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_rods_l777_77775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_20_at_30_over_7_l777_77771

-- Define the functions h and f
noncomputable def f (x : ℝ) : ℝ := 30 / (x + 2)
noncomputable def h (x : ℝ) : ℝ := 4 * (Function.invFun f x)

-- State the theorem
theorem h_equals_20_at_30_over_7 : h (30 / 7) = 20 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_20_at_30_over_7_l777_77771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l777_77759

open Real

/-- A function satisfying the given conditions -/
def f (a : ℝ) : ℝ → ℝ := sorry

/-- The function f is odd -/
axiom f_odd (a : ℝ) : ∀ x, f a (-x) + f a x = 0

/-- The function f has the form 1 + a^x for positive x -/
axiom f_pos (a : ℝ) : ∀ x > 0, f a x = 1 + a^x

/-- The value of f at -1 is -3/2 -/
axiom f_neg_one (a : ℝ) : f a (-1) = -3/2

/-- Given the conditions on f, prove that a = 1/2 -/
theorem a_value : ∃ a : ℝ, (∀ x, f a (-x) + f a x = 0) ∧
                           (∀ x > 0, f a x = 1 + a^x) ∧
                           (f a (-1) = -3/2) →
                           a = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l777_77759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l777_77772

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + (1/2) * x^2

/-- The theorem statement -/
theorem range_of_a (a : ℝ) (h_a_pos : a > 0) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 2) →
  a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l777_77772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_squared_divisible_by_240_l777_77760

theorem least_k_squared_divisible_by_240 :
  ∀ k : ℕ, k^2 % 240 = 0 → k ≥ 60 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_squared_divisible_by_240_l777_77760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l777_77734

/-- The sequence a_n defined recursively -/
def a : ℕ → ℚ
| 0 => 3
| n + 1 => (3 * a n - 4) / (a n - 1)

/-- The theorem stating the closed form of a_n -/
theorem a_closed_form (n : ℕ) : a n = (2 * n.succ + 1) / n.succ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l777_77734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_three_A_given_at_least_two_A_is_53_188_l777_77736

def string_length : ℕ := 5
def num_letters : ℕ := 4

def prob_at_least_three_A_given_at_least_two_A : ℚ :=
  let total_strings := (num_letters : ℚ) ^ string_length
  let strings_with_at_least_two_A : ℚ := (
    (Nat.choose string_length 2 : ℚ) * ((num_letters - 1) : ℚ) ^ 3 +
    (Nat.choose string_length 3 : ℚ) * ((num_letters - 1) : ℚ) ^ 2 +
    (Nat.choose string_length 4 : ℚ) * ((num_letters - 1) : ℚ) ^ 1 +
    (Nat.choose string_length 5 : ℚ) * ((num_letters - 1) : ℚ) ^ 0
  )
  let strings_with_at_least_three_A : ℚ := (
    (Nat.choose string_length 3 : ℚ) * ((num_letters - 1) : ℚ) ^ 2 +
    (Nat.choose string_length 4 : ℚ) * ((num_letters - 1) : ℚ) ^ 1 +
    (Nat.choose string_length 5 : ℚ) * ((num_letters - 1) : ℚ) ^ 0
  )
  strings_with_at_least_three_A / strings_with_at_least_two_A

theorem prob_at_least_three_A_given_at_least_two_A_is_53_188 :
  prob_at_least_three_A_given_at_least_two_A = 53 / 188 := by
  sorry

#eval prob_at_least_three_A_given_at_least_two_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_three_A_given_at_least_two_A_is_53_188_l777_77736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_range_l777_77773

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

noncomputable def S_n (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + ((n - 1) : ℝ) * d) / 2

theorem common_difference_range :
  ∀ d : ℝ,
    (∀ n : ℕ, S_n (-20) d n ≥ S_n (-20) d 6) ∧
    (∀ n : ℕ, n ≠ 6 → S_n (-20) d n > S_n (-20) d 6) →
    10/3 < d ∧ d < 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_range_l777_77773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_chords_is_78_75_degrees_l777_77757

/-- Represents a point on a circle -/
structure CirclePoint where
  angle : ℝ

/-- Represents a circle divided into four arcs -/
structure DividedCircle where
  A : CirclePoint
  B : CirclePoint
  C : CirclePoint
  D : CirclePoint

/-- The ratio of arc lengths in the divided circle -/
noncomputable def arcLengthRatio (circle : DividedCircle) : Prop :=
  let AB := (circle.B.angle - circle.A.angle + 2 * Real.pi) % (2 * Real.pi)
  let BC := (circle.C.angle - circle.B.angle + 2 * Real.pi) % (2 * Real.pi)
  let CD := (circle.D.angle - circle.C.angle + 2 * Real.pi) % (2 * Real.pi)
  let DA := (circle.A.angle - circle.D.angle + 2 * Real.pi) % (2 * Real.pi)
  AB / BC = 2 / 3 ∧ BC / CD = 3 / 5 ∧ CD / DA = 5 / 6

/-- The angle between two chords in a circle -/
noncomputable def angleBetweenChords (circle : DividedCircle) : ℝ :=
  let AB := (circle.B.angle - circle.A.angle + 2 * Real.pi) % (2 * Real.pi)
  let CD := (circle.D.angle - circle.C.angle + 2 * Real.pi) % (2 * Real.pi)
  (AB + CD) / 2

/-- Theorem: The angle between chords AC and BD is 78.75 degrees -/
theorem angle_between_chords_is_78_75_degrees (circle : DividedCircle) 
  (h : arcLengthRatio circle) : 
  angleBetweenChords circle = 78.75 * Real.pi / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_chords_is_78_75_degrees_l777_77757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_2770_l777_77737

def smart_tv_price : ℚ := 800
def soundbar_price : ℚ := 350
def bluetooth_speaker_price : ℚ := 100

def smart_tv_quantity : ℕ := 2
def soundbar_quantity : ℕ := 4
def bluetooth_speaker_quantity : ℕ := 6

def smart_tv_discount : ℚ := 20 / 100
def soundbar_discount : ℚ := 15 / 100

def bluetooth_speaker_promo (quantity : ℕ) : ℕ := (quantity + 1) / 2

noncomputable def total_cost : ℚ :=
  let smart_tv_cost := smart_tv_quantity * smart_tv_price * (1 - smart_tv_discount)
  let soundbar_cost := soundbar_quantity * soundbar_price * (1 - soundbar_discount)
  let bluetooth_speaker_cost := (bluetooth_speaker_promo bluetooth_speaker_quantity : ℚ) * bluetooth_speaker_price
  smart_tv_cost + soundbar_cost + bluetooth_speaker_cost

theorem total_cost_is_2770 : ⌊total_cost⌋ = 2770 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_2770_l777_77737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_investment_l777_77728

/-- Partnership investment problem -/
theorem partnership_investment (x : ℝ) (m : ℝ) : 
  x > 0 → -- A's investment is positive
  (x * 12 / (x * 12 + 2 * x * 6 + m * x * 4)) * 19200 = 6400 → -- A's share condition
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_investment_l777_77728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisor_four_consecutive_integers_l777_77703

theorem greatest_divisor_four_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (a : ℕ), a > 0 → 
    (∃ (k : ℕ), k ∈ Finset.range 4 ∧ 2 ∣ (a + k)) ∧
    (∃ (k : ℕ), k ∈ Finset.range 4 ∧ 3 ∣ (a + k)) ∧
    (∃ (k : ℕ), k ∈ Finset.range 4 ∧ 4 ∣ (a + k)) ∧
    (12 ∣ (a * (a+1) * (a+2) * (a+3))) ∧
    (∀ (m : ℕ), m > 12 → ¬(∀ (b : ℕ), b > 0 → m ∣ (b * (b+1) * (b+2) * (b+3))))) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisor_four_consecutive_integers_l777_77703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_new_config_value_l777_77748

/-- The energy stored in a configuration of three point charges, where two are at the vertices of a right triangle with sides of length 1, and one is at the midpoint of the hypotenuse. -/
noncomputable def energy_new_config (initial_energy : ℝ) : ℝ :=
  let energy_unit := initial_energy / 3
  let hypotenuse := Real.sqrt 2
  let midpoint_distance := hypotenuse / 2
  energy_unit + 2 * (energy_unit * 2 / midpoint_distance)

/-- Theorem stating that the energy in the new configuration is 4 + 8√2 Joules -/
theorem energy_new_config_value :
  energy_new_config 12 = 4 + 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_new_config_value_l777_77748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l777_77724

noncomputable def evaluate_expression (x a y z c d : ℝ) : ℝ :=
  (2 * x^3 - 3 * a^4) / (y^2 + 4 * z^5) + c^4 - d^2

theorem expression_evaluation :
  let x : ℝ := 0.66
  let a : ℝ := 0.1
  let y : ℝ := 0.66
  let z : ℝ := 0.1
  let c : ℝ := 0.066
  let d : ℝ := 0.1
  |evaluate_expression x a y z c d - 1.309091916| < 1e-9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l777_77724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circles_intersection_length_l777_77747

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The intersection points of two circles -/
noncomputable def circleIntersection (c1 c2 : Circle) : Set Point := sorry

/-- The length of segment Q'R' given three circles -/
noncomputable def segmentLength (p q r : Circle) : ℝ := sorry

theorem three_circles_intersection_length
  (p q r : Circle)
  (s : ℝ)
  (h1 : 2 < s)
  (h2 : s < 3)
  (h3 : p.radius = s ∧ q.radius = s ∧ r.radius = s)
  (h4 : distance p.center q.center = 3)
  (h5 : distance q.center r.center = 3)
  (h6 : distance r.center p.center = 3) :
  segmentLength p q r = 1.5 + Real.sqrt (6 * (s^2 - 2.25)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circles_intersection_length_l777_77747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_radius_l777_77744

/-- Two concentric circles with given properties -/
structure ConcentricCircles where
  r : ℝ  -- radius of the smaller circle
  R : ℝ  -- radius of the larger circle
  h : R = 4 * r  -- ratio of radii is 1:4

/-- Chord properties -/
structure Chord (c : ConcentricCircles) where
  A : ℝ × ℝ  -- point A as coordinates
  B : ℝ × ℝ  -- point B as coordinates
  C : ℝ × ℝ  -- point C as coordinates
  hAC : dist A C = 2 * c.R  -- AC is a diameter of the larger circle
  hBC : dist B C ≤ 2 * c.R  -- BC is a chord of the larger circle
  hTangent : dist (0, 0) B = c.R + c.r  -- BC is tangent to the smaller circle
  hAB : dist A B = 16  -- AB = 16

theorem larger_circle_radius (c : ConcentricCircles) (ch : Chord c) : c.R = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_radius_l777_77744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_isosceles_triangles_all_triangles_valid_no_other_triangles_l777_77762

/-- Represents a triangle formed by matchsticks -/
structure MatchstickTriangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ

/-- Checks if a triangle is valid according to the triangle inequality -/
def isValidTriangle (t : MatchstickTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : MatchstickTriangle) : Prop :=
  t.side1 = t.side2 ∨ t.side1 = t.side3 ∨ t.side2 = t.side3

/-- The set of all valid isosceles triangles formed with 14 matchsticks -/
def validIsoscelesTriangles : Set MatchstickTriangle :=
  {t | t.side1 + t.side2 + t.side3 = 14 ∧
       isValidTriangle t ∧
       isIsosceles t}

/-- List of all valid isosceles triangles formed with 14 matchsticks -/
def validIsoscelesTrianglesList : List MatchstickTriangle :=
  [⟨4, 4, 6⟩, ⟨5, 5, 4⟩, ⟨6, 6, 2⟩]

theorem count_isosceles_triangles :
  validIsoscelesTrianglesList.length = 3 := by
  rfl

theorem all_triangles_valid :
  ∀ t ∈ validIsoscelesTrianglesList, t ∈ validIsoscelesTriangles := by
  sorry

theorem no_other_triangles :
  ∀ t ∈ validIsoscelesTriangles, t ∈ validIsoscelesTrianglesList := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_isosceles_triangles_all_triangles_valid_no_other_triangles_l777_77762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_to_fraction_l777_77717

theorem periodic_decimal_to_fraction :
  (∃ (x : ℚ), x = 2 + (6/99)) ∧ (2/99 = 2 / 99) →
  (∃ (y : ℚ), y = 2 + (6/99) ∧ y = 68 / 33) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_to_fraction_l777_77717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_tan_theta_l777_77754

noncomputable def f (x θ : ℝ) : ℝ := Real.sqrt 3 * Real.cos (3 * x - θ) - Real.sin (3 * x - θ)

theorem odd_function_implies_tan_theta (θ : ℝ) :
  (∀ x, f x θ = -f (-x) θ) → Real.tan θ = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_tan_theta_l777_77754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8_625_l777_77781

noncomputable section

-- Define the slopes and intersection point
def slope1 : ℝ := 1/3
def slope2 : ℝ := 3
def intersection : ℝ × ℝ := (1, 1)

-- Define the lines
noncomputable def line1 (x : ℝ) : ℝ := slope1 * (x - intersection.1) + intersection.2
noncomputable def line2 (x : ℝ) : ℝ := slope2 * (x - intersection.1) + intersection.2
def line3 (x y : ℝ) : Prop := x + y = 8

-- Define the area of a triangle given three points
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Theorem statement
theorem triangle_area_is_8_625 :
  ∃ (A B C : ℝ × ℝ),
    A = intersection ∧
    line3 B.1 B.2 ∧
    line3 C.1 C.2 ∧
    line1 B.1 = B.2 ∧
    line2 C.1 = C.2 ∧
    triangleArea A B C = 8.625 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8_625_l777_77781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonal_length_l777_77756

/-- A cyclic quadrilateral with side lengths a, b, c, d -/
structure CyclicQuadrilateral where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  less_than_20 : a < 20 ∧ b < 20 ∧ c < 20 ∧ d < 20
  product_equality : a * b = c * d

/-- The diagonal length of a cyclic quadrilateral -/
noncomputable def diagonal_length (q : CyclicQuadrilateral) : ℝ :=
  Real.sqrt ((q.a^2 + q.b^2 + q.c^2 + q.d^2) / 2)

/-- The theorem stating the maximum possible diagonal length -/
theorem max_diagonal_length :
  ∃ (q : CyclicQuadrilateral), ∀ (p : CyclicQuadrilateral),
    diagonal_length q ≥ diagonal_length p ∧ diagonal_length q = Real.sqrt 405 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonal_length_l777_77756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l777_77755

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

structure Ellipse :=
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)

noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

noncomputable def focus_distance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

noncomputable def vertex_to_focus (e : Ellipse) (s : ℝ) : ℝ :=
  if s = -1 then e.a - focus_distance e
  else if s = 1 then e.a + focus_distance e
  else 0

theorem ellipse_properties (e : Ellipse) 
  (h3 : Real.sqrt ((vertex_to_focus e 1) * (vertex_to_focus e (-1))) = Real.sqrt 3)
  (h4 : eccentricity e = 1/2) :
  ∃ (S : Set (ℝ × ℝ)) (area_range : Set ℝ),
    (∀ x y, (x, y) ∈ S ↔ ellipse_equation 2 (Real.sqrt 3) x y) ∧
    (∀ A B : ℝ × ℝ, A ∈ S → B ∈ S → A ≠ B → 
      (∃ k m : ℝ, m ≠ 0 ∧ 
        (∀ x : ℝ, (x, k * x + m) ∈ S → x = A.1 ∨ x = B.1) ∧
        (∃ r : ℝ, (A.2 / A.1) * (B.2 / B.1) = (k * r)^2)) →
      ∃ area : ℝ, area ∈ area_range ∧ 
        area = (1/2) * abs (A.1 * B.2 - A.2 * B.1)) ∧
    area_range = Set.Ioo 0 (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l777_77755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_symmetry_f_zero_f_max_value_l777_77797

-- Define the function f(x) as noncomputable due to Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

-- Statement 1: The smallest positive period of f(x) is π
theorem f_period : ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧ T = Real.pi :=
sorry

-- Statement 2: f(x) is symmetric about the line x = π/12
theorem f_symmetry : ∀ (x : ℝ), f (Real.pi / 6 - x) = f (Real.pi / 6 + x) :=
sorry

-- Statement 3: f(π/3) = 0
theorem f_zero : f (Real.pi / 3) = 0 :=
sorry

-- Statement 4: The maximum value of f(x) is 2
theorem f_max_value : (∀ (x : ℝ), f x ≤ 2) ∧ (∃ (x : ℝ), f x = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_symmetry_f_zero_f_max_value_l777_77797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equilateral_triangle_l777_77745

-- Define the line
def line (a x y : ℝ) : Prop := a * x + y - 2 = 0

-- Define the circle
def circle_eq (a x y : ℝ) : Prop := (x - 1)^2 + (y - a)^2 = 4

-- Define the center of the circle
def circle_center (a : ℝ) : ℝ × ℝ := (1, a)

-- Define the equilateral triangle property
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

-- Main theorem
theorem intersection_equilateral_triangle (a : ℝ) :
  (∃ A B : ℝ × ℝ, line a A.1 A.2 ∧ line a B.1 B.2 ∧ 
              circle_eq a A.1 A.2 ∧ circle_eq a B.1 B.2 ∧ 
              is_equilateral_triangle A B (circle_center a)) →
  a = 4 + Real.sqrt 15 ∨ a = 4 - Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equilateral_triangle_l777_77745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l777_77782

def z (m : ℝ) : ℂ := (1 + Complex.I) * m^2 + (5 - 2*Complex.I) * m + (6 - 15*Complex.I)

theorem complex_number_properties (m : ℝ) :
  ((z m).im = 0 ↔ m = 5 ∨ m = -3) ∧
  ((z m).im ≠ 0 ↔ m ≠ 5 ∧ m ≠ -3) ∧
  ((z m).re = 0 ↔ m = -2) ∧
  (((z m).re < 0 ∧ (z m).im < 0) ↔ -3 < m ∧ m < -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l777_77782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_well_defined_l777_77789

theorem log_function_well_defined (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.log ((5 - a) * (x^2 + 1)) / Real.log (a - 2)) ↔ (a = 5/2 ∨ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_well_defined_l777_77789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_questionnaires_calculation_l777_77766

noncomputable def category_questionnaires (response_rate : ℚ) (min_responses : ℕ) : ℕ :=
  Int.natAbs (Int.ceil ((min_responses : ℚ) / response_rate))

theorem minimum_questionnaires_calculation :
  let response_rate_A : ℚ := 65/100
  let response_rate_B : ℚ := 70/100
  let response_rate_C : ℚ := 85/100
  let min_responses_A : ℕ := 150
  let min_responses_B : ℕ := 100
  let min_responses_C : ℕ := 50
  let questionnaires_A := category_questionnaires response_rate_A min_responses_A
  let questionnaires_B := category_questionnaires response_rate_B min_responses_B
  let questionnaires_C := category_questionnaires response_rate_C min_responses_C
  let total_questionnaires := questionnaires_A + questionnaires_B + questionnaires_C
  questionnaires_A = 231 ∧
  questionnaires_B = 143 ∧
  questionnaires_C = 59 ∧
  total_questionnaires = 433 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_questionnaires_calculation_l777_77766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_theorem_l777_77794

/-- Represents a square pyramid -/
structure SquarePyramid where
  baseEdge : ℝ
  altitude : ℝ

/-- Calculates the volume of a square pyramid -/
noncomputable def pyramidVolume (p : SquarePyramid) : ℝ :=
  (1 / 3) * p.baseEdge^2 * p.altitude

/-- Represents a frustum formed by cutting a square pyramid -/
structure Frustum where
  originalPyramid : SquarePyramid
  smallerPyramid : SquarePyramid

/-- Calculates the volume of a frustum -/
noncomputable def frustumVolume (f : Frustum) : ℝ :=
  pyramidVolume f.originalPyramid - pyramidVolume f.smallerPyramid

theorem frustum_volume_theorem (f : Frustum) 
  (h1 : f.originalPyramid.baseEdge = 16)
  (h2 : f.originalPyramid.altitude = 10)
  (h3 : f.smallerPyramid.baseEdge = 8)
  (h4 : f.smallerPyramid.altitude = 5) :
  frustumVolume f = 2240 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_theorem_l777_77794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_senior_mean_score_l777_77709

theorem senior_mean_score 
  (total_students : ℕ) 
  (overall_mean : ℝ) 
  (senior_count : ℝ) 
  (non_senior_count : ℝ) 
  (senior_mean : ℝ) 
  (non_senior_mean : ℝ) :
  total_students = 200 →
  overall_mean = 120 →
  non_senior_count = senior_count + 0.8 * senior_count →
  senior_mean = 1.6 * non_senior_mean →
  senior_count + non_senior_count = ↑total_students →
  senior_count * senior_mean + non_senior_count * non_senior_mean = ↑total_students * overall_mean →
  senior_mean = 159 := by
  sorry

#check senior_mean_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_senior_mean_score_l777_77709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l777_77708

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ  -- Length of semi-major axis
  b : ℝ  -- Length of semi-minor axis
  c : ℝ  -- Distance from center to focus

/-- Properties of the specific ellipse in the problem -/
noncomputable def problem_ellipse : Ellipse where
  a := 6 / Real.sqrt 5
  b := 4 / Real.sqrt 5
  c := 2

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- Equation of an ellipse centered at the origin -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x^2 / e.a^2) + (y^2 / e.b^2) = 1

/-- Main theorem about the specific ellipse -/
theorem ellipse_properties (e : Ellipse) (h : e = problem_ellipse) :
  eccentricity e = Real.sqrt 5 / 3 ∧
  ∀ x y, ellipse_equation e x y ↔ (5/36) * y^2 + (5/16) * x^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l777_77708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_results_l777_77735

noncomputable section

structure Competition where
  p_preliminary : ℝ
  p_semifinal : ℝ
  p_final : ℝ

def three_stage_competition : Competition :=
  { p_preliminary := 2/3
  , p_semifinal := 1/2
  , p_final := 1/3 }

def prob_eliminated_semifinal (c : Competition) : ℝ :=
  c.p_preliminary * (1 - c.p_semifinal)

def prob_distribution (c : Competition) : Fin 3 → ℝ
| 0 => 1 - c.p_preliminary
| 1 => c.p_preliminary * (1 - c.p_semifinal)
| 2 => c.p_preliminary * c.p_semifinal

def expected_value (c : Competition) : ℝ :=
  (1 * (1 - c.p_preliminary)) +
  (2 * c.p_preliminary * (1 - c.p_semifinal)) +
  (3 * c.p_preliminary * c.p_semifinal)

theorem competition_results (c : Competition) (hc : c = three_stage_competition) :
  (prob_eliminated_semifinal c = 1/3) ∧
  (∀ i : Fin 3, prob_distribution c i = 1/3) ∧
  (expected_value c = 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_results_l777_77735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sequence_inequality_l777_77731

noncomputable def x : ℕ → ℝ
  | 0 => 4  -- Arbitrary value > 3 for n = 0 case
  | 1 => 4  -- Arbitrary value > 3 for n = 1 case
  | n + 2 => (3 * (x (n + 1))^2 - x (n + 1)) / (4 * (x (n + 1) - 1))

theorem x_sequence_inequality : ∀ n : ℕ, n ≥ 1 → 3 < x (n + 1) ∧ x (n + 1) < x n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sequence_inequality_l777_77731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l777_77721

noncomputable def g (t : ℝ) : ℝ := (t^2 + 3/4 * t) / (t^2 + 1)

theorem g_range :
  Set.range g = Set.Icc (-1/8 : ℝ) (9/8 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l777_77721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_truncated_pyramid_volume_l777_77723

/-- The volume of a right square truncated pyramid -/
noncomputable def truncated_pyramid_volume (diagonal : ℝ) (base1 : ℝ) (base2 : ℝ) : ℝ :=
  let S1 := base1 * base1
  let S2 := base2 * base2
  let h := Real.sqrt (diagonal^2 - (base1 - base2)^2 * 2)
  (h / 3) * (S1 + S2 + Real.sqrt (S1 * S2))

/-- Theorem stating the volume of a specific right square truncated pyramid -/
theorem specific_truncated_pyramid_volume :
  truncated_pyramid_volume 18 14 10 = 872 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_truncated_pyramid_volume_l777_77723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_track_length_l777_77780

/-- Represents the properties of a train journey -/
structure TrainJourney where
  distance : ℝ
  speed : ℝ
  time : ℝ

/-- Calculates the time taken for a journey given distance and speed -/
noncomputable def calculateTime (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Calculates the distance traveled given speed and time -/
noncomputable def calculateDistance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Rounds a real number to the nearest integer -/
noncomputable def roundToNearestInt (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem second_train_track_length
  (train1 : TrainJourney)
  (train2 : TrainJourney)
  (h1 : train1.distance = 200)
  (h2 : train1.speed = 50)
  (h3 : train2.speed = 80)
  (h4 : roundToNearestInt ((calculateTime train1.distance train1.speed + calculateTime train2.distance train2.speed) / 2) = 4) :
  train2.distance = 320 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_track_length_l777_77780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l777_77792

def Ball := Fin 6
def Box := Fin 6

def ValidArrangement (arrangement : Box → Ball) : Prop :=
  ∃ (matching : Finset Box), matching.card = 3 ∧
    (∀ b ∈ matching, arrangement b = b) ∧
    (∀ b ∉ matching, arrangement b ≠ b)

-- Add this line to provide the necessary instance
instance : Fintype { arrangement : Box → Ball | ValidArrangement arrangement } :=
  by sorry

theorem arrangement_count :
  Fintype.card { arrangement : Box → Ball | ValidArrangement arrangement } = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l777_77792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_change_l777_77767

/-- The perimeter of both the rectangle and the triangle -/
def perimeter : ℝ := 150

/-- The length of the rectangular garden -/
def rectangle_length : ℝ := 60

/-- The width of the rectangular garden -/
def rectangle_width : ℝ := 15

/-- The side length of the equilateral triangle -/
noncomputable def triangle_side : ℝ := perimeter / 3

/-- The area of the rectangular garden -/
def rectangle_area : ℝ := rectangle_length * rectangle_width

/-- The area of the equilateral triangle -/
noncomputable def triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side ^ 2

theorem garden_area_change :
  triangle_area - rectangle_area = 625 * Real.sqrt 3 - 900 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_change_l777_77767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l777_77765

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + 2 * x - 5

theorem root_in_interval (k : ℤ) (x₀ : ℝ) (h₁ : f x₀ = 0) (h₂ : x₀ > k ∧ x₀ < k + 1) : k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l777_77765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l777_77764

theorem triangle_perimeter (a b c : ℝ) (h1 : a - b = 4) (h2 : a + c = 2*b) 
  (h3 : ∃ θ : ℝ, θ = 120 * Real.pi / 180 ∧ a^2 = b^2 + c^2 - 2*b*c*(Real.cos θ)) : 
  a + b + c = 30 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l777_77764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l777_77788

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def frac (x : ℝ) : ℝ := x - ↑(floor x)

theorem system_solution :
  ∃! (x y z : ℝ),
    x + floor y + frac z = 1.1 ∧
    frac x + y + floor z = 2.2 ∧
    floor x + frac y + z = 3.3 ∧
    x = 1 ∧ y = 0.2 ∧ z = 2.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l777_77788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l777_77752

noncomputable def snowball_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem snowman_volume :
  let v1 := snowball_volume 4
  let v2 := snowball_volume 6
  let v3 := snowball_volume 8
  let vc := cylinder_volume 3 5
  v1 + v2 + v3 + vc = 1101 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l777_77752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l777_77716

theorem trigonometric_identity (α β : ℝ) 
  (h : (Real.cos α)^4 / (Real.cos β)^2 + (Real.sin α)^4 / (Real.sin β)^2 = 1) :
  (Real.cos β)^4 / (Real.cos α)^2 + (Real.sin β)^4 / (Real.sin α)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l777_77716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_squared_l777_77704

-- Define the variables as noncomputable
noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 12
noncomputable def b : ℝ := -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 12
noncomputable def c : ℝ := Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 12
noncomputable def d : ℝ := -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 12

-- State the theorem
theorem sum_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = 12 * (Real.sqrt 35 - 5)^2 / 1225 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_squared_l777_77704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_three_l777_77740

/-- The length of the tangent from a point to a circle -/
noncomputable def tangent_length (px py cx cy r : ℝ) : ℝ :=
  Real.sqrt ((px - cx)^2 + (py - cy)^2 - r^2)

/-- Theorem: The length of the tangent from P(1, -2) to the circle (x+1)^2 + (y-1)^2 = 4 is 3 -/
theorem tangent_length_is_three :
  tangent_length 1 (-2) (-1) 1 2 = 3 := by
  -- Expand the definition of tangent_length
  unfold tangent_length
  -- Simplify the expression under the square root
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_three_l777_77740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l777_77738

-- Define the binomial expansion
noncomputable def binomial_expansion (x : ℝ) : ℝ := (x^(1/3) - 1/x)^8

-- Define the sum of binomial coefficients
def sum_of_coefficients : ℕ := 256

-- Theorem statement
theorem constant_term_of_binomial_expansion :
  ∃ (c : ℝ), c = 28 ∧ 
  (∀ x : ℝ, x ≠ 0 → ∃ (f : ℝ → ℝ), binomial_expansion x = f x + c + (1/f x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l777_77738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l777_77795

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The right focus of an ellipse -/
noncomputable def Ellipse.right_focus (e : Ellipse) : ℝ × ℝ :=
  (e.a * e.eccentricity, 0)

/-- The right directrix of an ellipse -/
noncomputable def Ellipse.right_directrix (e : Ellipse) : ℝ :=
  e.a / e.eccentricity

/-- A point on the ellipse -/
structure EllipsePoint (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The perpendicular bisector of a line segment passes through a point -/
def perpendicular_bisector_through (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 
  (C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2)

theorem ellipse_eccentricity_half (e : Ellipse) :
  ∃ (P : EllipsePoint e), 
    perpendicular_bisector_through 
      (e.right_directrix, 0) 
      (P.x, P.y) 
      e.right_focus →
  e.eccentricity = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l777_77795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_of_b_l777_77796

-- Define the ages as functions from ℕ to ℕ
def a : ℕ → ℕ := sorry
def b : ℕ → ℕ := sorry
def c : ℕ → ℕ := sorry
def d : ℕ → ℕ := sorry

-- State the theorem
theorem age_of_b : 
  (∀ x, a x = b x + 2) →  -- a is two years older than b
  (∀ x, b x = 2 * c x) →    -- b is twice as old as c
  (∀ x, d x = a x / 2) →  -- d is half the age of a
  (∃ x, a x + b x + c x + d x = 33) →  -- total age is 33
  ∃ x, b x = 10 :=
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_of_b_l777_77796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l777_77727

noncomputable def total_distance : ℝ := 500
noncomputable def distance_leg1 : ℝ := 150
noncomputable def distance_leg2 : ℝ := 150
noncomputable def distance_leg3 : ℝ := 200
noncomputable def speed_leg1 : ℝ := 50
noncomputable def speed_leg2 : ℝ := 30
noncomputable def speed_leg3 : ℝ := 60
noncomputable def inclination_factor : ℝ := 0.9

noncomputable def time_leg1 : ℝ := distance_leg1 / speed_leg1
noncomputable def time_leg2 : ℝ := distance_leg2 / speed_leg2
noncomputable def time_leg3 : ℝ := distance_leg3 / (speed_leg3 * inclination_factor)

noncomputable def total_time : ℝ := time_leg1 + time_leg2 + time_leg3

noncomputable def average_speed : ℝ := total_distance / total_time

theorem journey_average_speed : 
  ∀ ε > 0, |average_speed - 42.74| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l777_77727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l777_77742

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := 2*x - Real.log x

-- Define the point of tangency
def point : ℝ × ℝ := (1, 2)

-- Define the slope of the tangent line
noncomputable def m : ℝ := 2 - 1 / point.1

-- Define the equation of the tangent line
noncomputable def tangent_line (x : ℝ) : ℝ := m * (x - point.1) + point.2

-- Theorem statement
theorem tangent_line_equation :
  ∀ x : ℝ, tangent_line x = x + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l777_77742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l777_77778

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given condition in the problem -/
def condition (t : Triangle) : Prop :=
  t.a * Real.cos t.B + t.b * Real.cos t.A = 2 * t.c * Real.cos t.C

theorem triangle_theorem (t : Triangle) (h : condition t) :
  t.C = Real.pi / 3 ∧ (t.a = 5 ∧ t.b = 8 → t.c = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l777_77778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_people_lying_l777_77786

inductive Person : Type
  | A | B | C | D

def isLying : Person → Prop := sorry

def statement (p : Person) : Prop :=
  match p with
  | Person.A => (∃! q, isLying q)
  | Person.B => (∃ q₁ q₂, q₁ ≠ q₂ ∧ isLying q₁ ∧ isLying q₂ ∧ ∀ r, isLying r → r = q₁ ∨ r = q₂)
  | Person.C => (∃ q₁ q₂ q₃, q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₂ ≠ q₃ ∧ 
          isLying q₁ ∧ isLying q₂ ∧ isLying q₃ ∧ 
          ∀ r, isLying r → r = q₁ ∨ r = q₂ ∨ r = q₃)
  | Person.D => (∀ q, isLying q)

theorem three_people_lying :
  ∃ q₁ q₂ q₃, q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₂ ≠ q₃ ∧ 
  isLying q₁ ∧ isLying q₂ ∧ isLying q₃ ∧ 
  ∀ r, isLying r → r = q₁ ∨ r = q₂ ∨ r = q₃ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_people_lying_l777_77786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_difference_l777_77713

theorem sqrt_sum_difference (x : ℝ) : 
  Real.sqrt 25 + ((-27) ^ (1/3 : ℝ)) - |Real.sqrt 3 - 2| = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_difference_l777_77713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_of_a_10_l777_77726

-- Define the sequence
def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else (2*n - 1) + Finset.sum (Finset.range (n-1)) (fun i => (2*n - 1) + 2*(i+1))

-- Define the first number in the sum for any n
def first_number (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 2 * (Finset.sum (Finset.range (n-1)) (fun i => i + 1))

-- Theorem statement
theorem first_number_of_a_10 : first_number 10 = 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_of_a_10_l777_77726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_double_money_l777_77779

/-- Represents the currencies in Limpopo --/
inductive Currency : Type
| Banana : Currency
| Coconut : Currency
| Raccoon : Currency
| Dollar : Currency

/-- Represents the amount of a specific currency --/
structure Amount where
  value : ℚ
  currency : Currency

/-- Defines the exchange rates between currencies --/
def exchangeRate (from_currency to_currency : Currency) : ℚ :=
  match from_currency, to_currency with
  | Currency.Raccoon, Currency.Banana => 6
  | Currency.Raccoon, Currency.Coconut => 11
  | Currency.Dollar, Currency.Coconut => 10
  | Currency.Coconut, Currency.Dollar => 1 / 15
  | Currency.Banana, Currency.Coconut => 1 / 2
  | _, _ => 0  -- Represents prohibited or undefined exchanges

/-- Represents a single exchange operation --/
structure Exchange where
  from_amount : Amount
  to_currency : Currency

/-- Represents a sequence of exchange operations --/
def ExchangeSequence := List Exchange

/-- Determines if an exchange sequence is valid according to the rules --/
def isValidExchangeSequence (seq : ExchangeSequence) : Prop :=
  sorry

/-- Calculates the final amount after applying an exchange sequence --/
def applyExchangeSequence (initial : Amount) (seq : ExchangeSequence) : Amount :=
  sorry

/-- The main theorem to be proved --/
theorem can_double_money :
  ∃ (seq : ExchangeSequence),
    isValidExchangeSequence seq ∧
    (applyExchangeSequence ⟨100, Currency.Dollar⟩ seq).value ≥ 200 ∧
    (applyExchangeSequence ⟨100, Currency.Dollar⟩ seq).currency = Currency.Dollar :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_double_money_l777_77779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_imaginary_part_is_zero_l777_77770

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := (1 - i) / (i * (1 + i))

theorem z_imaginary_part_is_zero : Complex.im z = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_imaginary_part_is_zero_l777_77770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_matrix_exists_l777_77719

theorem scaling_matrix_exists : ∃ (M : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
    M • A = !![5 * A 0 0, 5 * A 0 1; 3 * A 1 0, 3 * A 1 1] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_matrix_exists_l777_77719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l777_77776

/-- An ellipse with equation x^2/8 + y^2/4 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}

/-- The left focus of the ellipse -/
def leftFocus : ℝ × ℝ := (-2, 0)

/-- The right focus of the ellipse -/
def rightFocus : ℝ × ℝ := (2, 0)

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity : ℝ := Real.sqrt 2 / 2

/-- The length of the minor axis of the ellipse -/
def minorAxisLength : ℝ := 4

/-- The maximum area of the triangle formed by any point on the ellipse and the two foci -/
def maxTriangleArea : ℝ := 4

theorem ellipse_properties :
  eccentricity = Real.sqrt 2 / 2 ∧
  minorAxisLength = 4 ∧
  maxTriangleArea = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l777_77776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_multiples_l777_77751

def first_30_naturals : Set ℕ := {n : ℕ | n ≥ 1 ∧ n ≤ 30}

def multiples_of_4_or_15 (n : ℕ) : Bool := n % 4 = 0 ∨ n % 15 = 0

theorem probability_of_multiples : 
  (Finset.filter (λ n => multiples_of_4_or_15 n) (Finset.range 30)).card / 30 = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_multiples_l777_77751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_candle_half_length_of_other_l777_77720

/-- Represents a candle with its initial length and burn time. -/
structure Candle where
  initialLength : ℝ
  burnTime : ℝ

/-- Calculates the length of a candle after a given time. -/
noncomputable def remainingLength (c : Candle) (t : ℝ) : ℝ :=
  c.initialLength - (c.initialLength / c.burnTime) * t

/-- The problem setup with two candles. -/
def candleProblem :=
  let thinCandle : Candle := { initialLength := 24, burnTime := 4 }
  let thickCandle : Candle := { initialLength := 24, burnTime := 6 }
  (thinCandle, thickCandle)

/-- Theorem stating that one candle will be half the length of the other after 3 hours. -/
theorem one_candle_half_length_of_other :
  let (thin, thick) := candleProblem
  ∃ t : ℝ, t = 3 ∧
    (remainingLength thin t = 2 * remainingLength thick t ∨
     remainingLength thick t = 2 * remainingLength thin t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_candle_half_length_of_other_l777_77720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l777_77784

noncomputable def f (x : ℝ) := Real.sin x - x

theorem f_inequality_solution_set :
  ∀ x : ℝ, f (x + 2) + f (1 - 2*x) < 0 ↔ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l777_77784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_300_l777_77793

theorem divisors_multiple_of_300 (n : ℕ) : 
  n = 2^12 * 3^9 * 5^5 → 
  (Finset.filter (λ d ↦ d ∣ n ∧ 300 ∣ d) (Finset.range (n + 1))).card = 396 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_300_l777_77793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_congruence_l777_77707

theorem product_congruence (n : ℕ) (a b c : ZMod n) : 
  0 < n →
  IsUnit a →
  IsUnit b →
  IsUnit c →
  a = b⁻¹ →
  c = a⁻¹ →
  (a * b) * c = c := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_congruence_l777_77707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_difference_proof_l777_77743

def yellow_weight : ℝ := 0.6
def green_weight : ℝ := 0.4
def red_weight : ℝ := 0.8
def blue_weight : ℝ := 0.5

def block_weights : List ℝ := [yellow_weight, green_weight, red_weight, blue_weight]

theorem weight_difference_proof :
  (List.maximum block_weights).getD 0 - (List.minimum block_weights).getD 0 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_difference_proof_l777_77743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_ratios_l777_77706

noncomputable def room_length : ℝ := 20.5
noncomputable def room_width : ℝ := 12.3

noncomputable def perimeter_feet : ℝ := 2 * (room_length + room_width)
noncomputable def perimeter_yards : ℝ := perimeter_feet / 3

theorem room_ratios :
  (room_length / perimeter_feet = 20.5 / 65.6) ∧
  (perimeter_yards = 21.8667) ∧
  (room_length / perimeter_yards = 20.5 / 21.8667) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_ratios_l777_77706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_artwork_arrangements_l777_77749

/-- Represents the number of different calligraphy works -/
def num_calligraphy : ℕ := 2

/-- Represents the number of different painting works -/
def num_paintings : ℕ := 2

/-- Represents the number of iconic architectural designs -/
def num_architectural : ℕ := 1

/-- Represents the total number of artwork pieces -/
def total_artwork : ℕ := num_calligraphy + num_paintings + num_architectural

/-- Represents the condition that calligraphy works must be adjacent -/
def calligraphy_adjacent : Prop := True

/-- Represents the condition that painting works must not be adjacent -/
def paintings_not_adjacent : Prop := True

/-- Represents the number of arrangements given the conditions -/
def number_of_arrangements (n : ℕ) : ℕ := sorry

/-- The main theorem stating the number of possible arrangements -/
theorem artwork_arrangements :
  (num_calligraphy = 2) →
  (num_paintings = 2) →
  (num_architectural = 1) →
  (total_artwork = 5) →
  calligraphy_adjacent →
  paintings_not_adjacent →
  (∃ n : ℕ, n = 96 ∧ n = number_of_arrangements total_artwork) :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

#check artwork_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_artwork_arrangements_l777_77749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_mother_is_jones_l777_77769

-- Define the set of mothers and daughters
inductive Mother | Jones | White | Brown | Smith
inductive Daughter | Mary | Nora | Gladys | Hilda

-- Define the function that maps daughters to their mothers
def mother_of : Daughter → Mother
| Daughter.Mary => Mother.Jones
| Daughter.Nora => Mother.Brown
| Daughter.Gladys => Mother.White
| Daughter.Hilda => Mother.Smith

-- Define the function that gives the amount of ribbon bought by each person
def ribbon_bought : Mother ⊕ Daughter → ℕ
| Sum.inl Mother.Jones => 20
| Sum.inl Mother.White => 12
| Sum.inl Mother.Brown => 18
| Sum.inl Mother.Smith => 8
| Sum.inr Daughter.Mary => 10
| Sum.inr Daughter.Nora => 9
| Sum.inr Daughter.Gladys => 6
| Sum.inr Daughter.Hilda => 4

-- Define the conditions
axiom mother_daughter_relation :
  ∀ (m : Mother), ∃ (d : Daughter), mother_of d = m

axiom mother_bought_double :
  ∀ (d : Daughter), ribbon_bought (Sum.inr d) * 2 = ribbon_bought (Sum.inl (mother_of d))

axiom price_equals_length :
  ∀ (p : Mother ⊕ Daughter), ribbon_bought p = ribbon_bought p

axiom jones_white_difference :
  ribbon_bought (Sum.inl Mother.Jones) = ribbon_bought (Sum.inl Mother.White) + 76

axiom nora_brown_difference :
  ribbon_bought (Sum.inr Daughter.Nora) + 3 = ribbon_bought (Sum.inl Mother.Brown) / 2

axiom gladys_hilda_difference :
  ribbon_bought (Sum.inr Daughter.Gladys) = ribbon_bought (Sum.inr Daughter.Hilda) + 2

axiom hilda_smith_difference :
  ribbon_bought (Sum.inr Daughter.Hilda) + 48 = ribbon_bought (Sum.inl Mother.Smith)

-- Theorem to prove
theorem mary_mother_is_jones :
  mother_of Daughter.Mary = Mother.Jones := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_mother_is_jones_l777_77769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_adjacent_face_diagonals_perpendicular_l777_77701

/-- A cube is a three-dimensional shape with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- A face of a cube -/
structure Face (c : Cube) where
  -- We don't need to define the specifics of a face for this problem

/-- A face diagonal is a line segment connecting opposite corners of a face -/
structure FaceDiagonal (c : Cube) where
  face : Face c
  -- We don't need to define other specifics of a face diagonal for this problem

/-- A line in 3D space -/
structure Line where
  -- We don't need to define the specifics of a line for this problem

/-- Two faces are adjacent if they share an edge -/
def adjacent_faces (c : Cube) (f1 f2 : Face c) : Prop :=
  sorry

/-- The angle between two lines -/
def angle_between (l1 l2 : Line) : ℝ :=
  sorry

/-- Convert a face diagonal to a line -/
def FaceDiagonal.toLine (d : FaceDiagonal c) : Line :=
  sorry

theorem cube_adjacent_face_diagonals_perpendicular (c : Cube) 
  (f1 f2 : Face c) (d1 : FaceDiagonal c) (d2 : FaceDiagonal c) :
  adjacent_faces c f1 f2 →
  d1.face = f1 →
  d2.face = f2 →
  angle_between (d1.toLine) (d2.toLine) = 90 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_adjacent_face_diagonals_perpendicular_l777_77701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_with_given_sums_l777_77732

def is_valid_set (A : Finset ℕ) : Prop :=
  A.card = 5 ∧ ∃ x₁ x₂ x₃ x₄ x₅, A = {x₁, x₂, x₃, x₄, x₅} ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅

def two_element_sums (A : Finset ℕ) : Finset ℕ :=
  (A.powerset.filter (fun s ↦ s.card = 2)).image (fun s ↦ s.sum id)

theorem unique_set_with_given_sums :
  ∀ A : Finset ℕ,
    is_valid_set A →
    two_element_sums A = {4, 5, 6, 7, 8, 9, 10, 12, 13, 14} →
    A = {1, 3, 4, 5, 9} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_with_given_sums_l777_77732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_parallel_difference_magnitude_l777_77746

def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

theorem perpendicular_condition (x : ℝ) :
  (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0 → x = -1 ∨ x = 3 := by sorry

theorem parallel_difference_magnitude (x : ℝ) :
  (∃ (k : ℝ), a x = k • (b x)) →
  ‖a x - b x‖ = 2 ∨ ‖a x - b x‖ = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_parallel_difference_magnitude_l777_77746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_start_distance_transitive_l777_77714

/-- Represents a runner in a race -/
structure Runner where
  speed : ℚ
  deriving Repr

/-- Calculates the start distance one runner can give another in a 1000-meter race -/
def start_distance (r1 r2 : Runner) : ℚ :=
  1000 - (1000 * r2.speed / r1.speed)

/-- Theorem stating the transitive property of start distances -/
theorem start_distance_transitive (a b c : Runner) 
  (hab : start_distance a b = 100)
  (hbc : start_distance b c = 55.56) :
  start_distance a c = 150 := by
  sorry

#eval "Proof skipped with sorry"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_start_distance_transitive_l777_77714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_camp_total_l777_77785

theorem boys_camp_total (total : ℕ) : 
  (total * 20 / 100 : ℕ) > 0 →  -- Ensure non-zero division
  (total * 20 / 100 * 30 / 100 : ℕ) < (total * 20 / 100 : ℕ) →  -- Ensure valid percentage
  (total * 20 / 100 - total * 20 / 100 * 30 / 100 : ℕ) = 63 →
  total = 450 := by
  sorry

#check boys_camp_total

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_camp_total_l777_77785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_reduction_correct_bicycle_a_reduction_bicycle_b_reduction_bicycle_c_reduction_l777_77700

/-- Given an original price and three consecutive percentage reductions,
    calculates the equivalent overall percentage reduction. -/
noncomputable def overallReduction (originalPrice : ℝ) (reduction1 reduction2 reduction3 : ℝ) : ℝ :=
  1 - (1 - reduction1 / 100) * (1 - reduction2 / 100) * (1 - reduction3 / 100)

/-- Theorem stating that the overall reduction calculated by the function
    is equal to the actual percentage reduction from the original price to the final price. -/
theorem overall_reduction_correct (originalPrice : ℝ) (reduction1 reduction2 reduction3 : ℝ)
    (h1 : originalPrice > 0) (h2 : reduction1 ≥ 0) (h3 : reduction2 ≥ 0) (h4 : reduction3 ≥ 0) :
  let finalPrice := originalPrice * (1 - reduction1 / 100) * (1 - reduction2 / 100) * (1 - reduction3 / 100)
  overallReduction originalPrice reduction1 reduction2 reduction3 = (originalPrice - finalPrice) / originalPrice :=
by
  sorry

/-- Verifies the overall reduction for Bicycle A -/
theorem bicycle_a_reduction :
  abs (overallReduction 600 20 15 10 - 0.388) < 0.001 :=
by
  sorry

/-- Verifies the overall reduction for Bicycle B -/
theorem bicycle_b_reduction :
  abs (overallReduction 800 25 20 5 - 0.43) < 0.001 :=
by
  sorry

/-- Verifies the overall reduction for Bicycle C -/
theorem bicycle_c_reduction :
  abs (overallReduction 1000 30 10 25 - 0.5275) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_reduction_correct_bicycle_a_reduction_bicycle_b_reduction_bicycle_c_reduction_l777_77700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_I_problem_II_l777_77790

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 : ℝ)^(a*x) - 2
def g (a : ℝ) (x : ℝ) : ℝ := a*(x-2*a)*(x+2-a)

-- Theorem I
theorem problem_I (a : ℝ) (h : a ≠ 0) :
  ({x : ℝ | f a x * g a x = 0} = {1, 2}) → a = 1 := by sorry

-- Theorem II
theorem problem_II (a : ℝ) (h : a ≠ 0) :
  ({x : ℝ | f a x < 0 ∨ g a x < 0} = Set.univ) → -Real.sqrt 2 / 2 < a ∧ a < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_I_problem_II_l777_77790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_miles_reading_problem_l777_77758

noncomputable def graphic_novel_reading_speed (total_day_fraction : ℚ) 
  (novel_speed comic_speed : ℕ) (total_pages : ℕ) : ℚ :=
  let total_hours : ℚ := 24 * total_day_fraction
  let type_hours : ℚ := total_hours / 3
  let novel_pages : ℚ := (novel_speed : ℚ) * type_hours
  let comic_pages : ℚ := (comic_speed : ℚ) * type_hours
  let graphic_pages : ℚ := (total_pages : ℚ) - novel_pages - comic_pages
  graphic_pages / type_hours

theorem miles_reading_problem :
  graphic_novel_reading_speed (1/6) 21 45 128 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_miles_reading_problem_l777_77758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l777_77798

/-- Triangle DEF with given properties -/
structure Triangle where
  DE : ℝ
  EF : ℝ
  FD : ℝ
  de_length : DE = 16
  ef_fd_ratio : EF / FD = 25 / 24

/-- The area of a triangle given its side lengths -/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  let s := (t.DE + t.EF + t.FD) / 2
  Real.sqrt (s * (s - t.DE) * (s - t.EF) * (s - t.FD))

/-- The maximum area of the triangle is 446.25 -/
theorem max_triangle_area :
  ∀ t : Triangle, triangle_area t ≤ 446.25 ∧
  ∃ t : Triangle, triangle_area t = 446.25 := by
  sorry

#check max_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l777_77798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_lineup_count_l777_77729

/-- Represents a person in the lineup --/
inductive Person
| M1 | M2 | M3 | W1 | W2 | W3

/-- Represents a position in the lineup --/
def Position := Fin 6

/-- Checks if a person is a man --/
def is_man (p : Person) : Bool :=
  match p with
  | Person.M1 | Person.M2 | Person.M3 => true
  | _ => false

/-- Checks if a person is a woman --/
def is_woman (p : Person) : Bool :=
  match p with
  | Person.W1 | Person.W2 | Person.W3 => true
  | _ => false

/-- Checks if two people are a forbidden pair --/
def is_forbidden_pair (p1 p2 : Person) : Bool :=
  match p1, p2 with
  | Person.M1, Person.W1 | Person.W1, Person.M1 => true
  | Person.M2, Person.W2 | Person.W2, Person.M2 => true
  | Person.M3, Person.W3 | Person.W3, Person.M3 => true
  | _, _ => false

/-- Represents a lineup of people --/
def Lineup := Position → Person

/-- Checks if a lineup is valid according to the problem constraints --/
def is_valid_lineup (l : Lineup) : Prop :=
  (∀ i : Position, (i.val % 2 = 0 → is_man (l i)) ∧ (i.val % 2 = 1 → is_woman (l i))) ∧
  (∀ i j : Position, i.val + 1 = j.val → ¬is_forbidden_pair (l i) (l j))

/-- The number of valid lineups --/
def num_valid_lineups : ℕ := 6

theorem valid_lineup_count :
  (∃ (S : Finset Lineup), (∀ l ∈ S, is_valid_lineup l) ∧ S.card = num_valid_lineups) := by
  sorry

#check valid_lineup_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_lineup_count_l777_77729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l777_77783

/-- A random variable following a normal distribution -/
structure NormalRandomVariable (μ σ : ℝ) where
  pdf : ℝ → ℝ
  is_normal : ∀ x, pdf x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

/-- Cumulative distribution function for a normal random variable -/
noncomputable def cdf (ξ : NormalRandomVariable μ σ) (x : ℝ) : ℝ :=
  ∫ y in Set.Iic x, ξ.pdf y

/-- Theorem stating the relationship between CDF values for a normal distribution -/
theorem normal_distribution_probability (σ : ℝ) (hσ : σ > 0) :
  ∃ ξ : NormalRandomVariable 2 σ,
    cdf ξ 4 = 0.8 → cdf ξ 2 - cdf ξ 0 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l777_77783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l777_77799

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1

-- Define point A
def point_A : ℝ × ℝ := (-2, 2)

-- Define the left focus F
def left_focus : ℝ × ℝ := (-3, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the objective function to be minimized
noncomputable def objective (B : ℝ × ℝ) : ℝ :=
  distance point_A B + (5/3) * distance B left_focus

-- State the theorem
theorem min_distance_point :
  ∃ (B : ℝ × ℝ), is_on_ellipse B.1 B.2 ∧
    (∀ (C : ℝ × ℝ), is_on_ellipse C.1 C.2 → objective B ≤ objective C) ∧
    B = (-5 * Real.sqrt 3 / 2, 2) := by
  sorry

#check min_distance_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l777_77799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_individual_is_14_l777_77702

def population : Set Nat := {n | 1 ≤ n ∧ n ≤ 20}

def random_table : List (List Nat) := [
  [7816, 6572, 0802, 6314, 0702, 4369, 9728, 0198],
  [3204, 9234, 4935, 8200, 3623, 4869, 6938, 7481]
]

def is_valid (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 20

def select_individuals (table : List (List Nat)) : List Nat :=
  sorry

theorem fourth_individual_is_14 :
  (select_individuals random_table).get? 3 = some 14 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_individual_is_14_l777_77702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_ratio_l777_77733

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then n * a₁ 
  else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sum_ratio 
  (a₁ q : ℝ) 
  (h₁ : q ≠ 1) 
  (h₂ : geometric_sum a₁ q 4 / geometric_sum a₁ q 2 = 3) :
  geometric_sum a₁ q 6 / geometric_sum a₁ q 4 = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_ratio_l777_77733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_stack_count_is_190_times_3_cubed_l777_77777

/-- Represents a crate with three possible height orientations -/
inductive CrateOrientation
  | Three
  | Four
  | Six

/-- The height of a crate given its orientation -/
def crateHeight (o : CrateOrientation) : ℕ :=
  match o with
  | .Three => 3
  | .Four => 4
  | .Six => 6

/-- A stack of crates -/
def CrateStack := List CrateOrientation

/-- The total height of a stack of crates -/
def stackHeight (stack : CrateStack) : ℕ :=
  stack.map crateHeight |>.sum

/-- The number of valid stacks of 10 crates with a total height of 41 feet -/
def validStackCount : ℕ := sorry

theorem valid_stack_count_is_190_times_3_cubed :
  validStackCount = 190 * 3^3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_stack_count_is_190_times_3_cubed_l777_77777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_depends_on_a_d_n_l777_77761

/-- Sum of k terms of an arithmetic progression with first term a and common difference d -/
noncomputable def sum_ap (a d : ℝ) (k : ℕ) : ℝ := k / 2 * (2 * a + (k - 1) * d)

/-- R is defined as the difference between s₃, s₂, and s₁ -/
noncomputable def R (a d : ℝ) (n : ℕ) : ℝ :=
  sum_ap a d (5 * n) - sum_ap a d (3 * n) - sum_ap a d n

theorem R_depends_on_a_d_n (a d : ℝ) (n : ℕ) :
  ∃ (f : ℝ → ℝ → ℕ → ℝ), R a d n = f a d n := by
  use fun a d n => n * a * (2 - d) + 15 * d * n^2
  sorry  -- The actual proof is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_depends_on_a_d_n_l777_77761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_b_existence_l777_77705

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem integer_b_existence (b : ℤ) : 
  (∃ x : ℝ, x > 0 ∧ (1 : ℝ) / b = 1 / (floor (2 * x)) + 1 / (floor (5 * x))) ↔ 
  (b = 3 ∨ ∃ k : ℕ, b = 10 * k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_b_existence_l777_77705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_C_squared_minus_D_squared_l777_77718

/-- The minimum value of C² - D² is 36, where
    C = √(u + 3) + √(v + 6) + √(w + 15) and
    D = √(u + 2) + √(v + 2) + √(w + 2),
    for any nonnegative real numbers u, v, and w. -/
theorem min_value_C_squared_minus_D_squared (u v w : ℝ) 
  (hu : u ≥ 0) (hv : v ≥ 0) (hw : w ≥ 0) :
  (Real.sqrt (u + 3) + Real.sqrt (v + 6) + Real.sqrt (w + 15))^2 -
  (Real.sqrt (u + 2) + Real.sqrt (v + 2) + Real.sqrt (w + 2))^2 ≥ 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_C_squared_minus_D_squared_l777_77718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equilateral_triangle_l777_77730

-- Define the line
def line (a x y : ℝ) : Prop := a * x + y - 2 = 0

-- Define the circle
def circleEq (a x y : ℝ) : Prop := (x - 1)^2 + (y - a)^2 = 4

-- Define the center of the circle
def center (a : ℝ) : ℝ × ℝ := (1, a)

-- Define the intersection points
def intersectionPoints (a : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), line a A.1 A.2 ∧ circleEq a A.1 A.2 ∧
                   line a B.1 B.2 ∧ circleEq a B.1 B.2 ∧
                   A ≠ B

-- Define an equilateral triangle
def isEquilateral (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 =
  (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 =
  (C.1 - A.1)^2 + (C.2 - A.2)^2

-- Main theorem
theorem intersection_equilateral_triangle (a : ℝ) :
  intersectionPoints a →
  (∃ (A B : ℝ × ℝ), isEquilateral A B (center a)) →
  a = 4 + Real.sqrt 15 ∨ a = 4 - Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equilateral_triangle_l777_77730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_exponent_l777_77712

theorem power_equality_implies_exponent (q : ℝ) : (16 : ℝ)^15 = (4 : ℝ)^q → q = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_exponent_l777_77712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjusted_retail_price_correct_l777_77741

/-- The retail price of a shirt after adjustment -/
noncomputable def adjusted_retail_price (m : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  m * (1 + a / 100) * (b / 100)

/-- Theorem: The adjusted retail price is correct -/
theorem adjusted_retail_price_correct (m : ℝ) (a : ℝ) (b : ℝ) :
  adjusted_retail_price m a b = m * (1 + a / 100) * (b / 100) := by
  -- Unfold the definition of adjusted_retail_price
  unfold adjusted_retail_price
  -- The equation is now trivially true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjusted_retail_price_correct_l777_77741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_sum_l777_77725

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Point on a parabola -/
def PointOnParabola (p : Parabola) (q : ℝ × ℝ) : Prop :=
  p.equation q.1 q.2

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem statement -/
theorem parabola_min_distance_sum (p : Parabola) (m : ℝ × ℝ) :
  p.equation = (fun x y => x^2 = 2*y) →
  m = (3, 5) →
  (∃ (q : ℝ × ℝ), PointOnParabola p q ∧
    ∀ (r : ℝ × ℝ), PointOnParabola p r →
      distance m q + distance q p.focus ≤ distance m r + distance r p.focus) →
  (∃ (q : ℝ × ℝ), PointOnParabola p q ∧
    distance m q + distance q p.focus = 11/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_sum_l777_77725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_speed_l777_77791

/-- Calculates the speed given distance and time -/
noncomputable def calculate_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem james_speed :
  let distance : ℝ := 80
  let time : ℝ := 5
  calculate_speed distance time = 16 := by
  -- Unfold the definition of calculate_speed
  unfold calculate_speed
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_speed_l777_77791
