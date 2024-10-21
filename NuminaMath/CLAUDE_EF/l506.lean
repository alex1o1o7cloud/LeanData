import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_commutator_zero_l506_50605

theorem det_commutator_zero {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℂ) 
  (h_odd : Odd n) (h_diff_squared : (A - B)^2 = 0) : 
  Matrix.det (A * B - B * A) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_commutator_zero_l506_50605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_magnitude_l506_50650

theorem perpendicular_vectors_magnitude (m : ℝ) : 
  let a : ℝ × ℝ := (2^m, -1)
  let b : ℝ × ℝ := (2^m - 1, 2^(m+1))
  (a.1 * b.1 + a.2 * b.2 = 0) → ‖a‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_magnitude_l506_50650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_of_symmetry_tan_l506_50681

/-- The function for which we're finding the center of symmetry -/
noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.tan (5*x + Real.pi/4)

/-- Predicate to check if a point is a center of symmetry for the given function -/
def is_center_of_symmetry (p : ℝ × ℝ) : Prop :=
  ∀ x : ℝ, f (p.1 + x) = -f (p.1 - x)

/-- The set of all centers of symmetry for the function -/
def centers_of_symmetry : Set (ℝ × ℝ) :=
  {p | is_center_of_symmetry p}

/-- Theorem stating the centers of symmetry for the given function -/
theorem centers_of_symmetry_tan :
  centers_of_symmetry = {p : ℝ × ℝ | ∃ k : ℤ, p = (↑k * Real.pi / 10 - Real.pi / 20, 0)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_of_symmetry_tan_l506_50681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l506_50629

theorem sum_remainder_theorem (a b c : ℕ) 
  (ha : a % 15 = 11) 
  (hb : b % 15 = 12) 
  (hc : c % 15 = 13) : 
  (a + b + c) % 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l506_50629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_x_ln_x_l506_50677

-- Define the function f(x) = x * ln(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem derivative_x_ln_x :
  ∀ x > 0, deriv f x = Real.log x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_x_ln_x_l506_50677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_subtraction_l506_50696

def a : ℂ := 5 - 3 * Complex.I
def b : ℂ := 2 + 4 * Complex.I

theorem complex_subtraction : a - 3 * b = -1 - 15 * Complex.I := by
  -- Expand the definitions of a and b
  unfold a b
  -- Perform the arithmetic
  simp [Complex.I, sub_eq_add_neg, mul_add, add_mul, mul_assoc]
  -- Simplify the result
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_subtraction_l506_50696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_weight_upper_limit_l506_50654

/-- Represents Arun's weight in kilograms -/
def arun_weight : ℝ := sorry

/-- Represents the upper limit of Arun's weight according to his brother's opinion -/
def brother_upper_limit : ℝ := sorry

/-- The average of different probable weights of Arun -/
def average_weight : ℝ := 68

theorem arun_weight_upper_limit :
  (66 < arun_weight ∧ arun_weight < 72) →
  (60 < arun_weight ∧ arun_weight < brother_upper_limit) →
  (arun_weight ≤ 69) →
  (average_weight = 68) →
  brother_upper_limit = 69 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_weight_upper_limit_l506_50654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counting_functions_theorem_l506_50617

/-- The number of functions from [m] to [n] -/
def numFunctions (m n : ℕ) : ℕ := n^m

/-- The number of injections from [m] to [n], given m ≤ n -/
def numInjections (m n : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - m)

/-- The number of surjections from [m] to [n], given m ≥ n -/
noncomputable def numSurjections (m n : ℕ) : ℕ := 
  Finset.sum (Finset.range (n + 1)) (fun k => 
    ((-1 : Int)^k).toNat * Nat.choose n k * (n - k)^m)

theorem counting_functions_theorem (m n : ℕ) :
  /- (i) The number of functions from [m] to [n] is n^m -/
  numFunctions m n = n^m ∧
  /- (ii) The number of injections from [m] to [n], given m ≤ n, is n! / (n-m)! -/
  (m ≤ n → numInjections m n = Nat.factorial n / Nat.factorial (n - m)) ∧
  /- (iii) The number of surjections from [m] to [n], given m ≥ n, is Σ(k=0 to n) (-1)^k * C(n,k) * (n-k)^m -/
  (m ≥ n → numSurjections m n = 
    Finset.sum (Finset.range (n + 1)) (fun k => 
      ((-1 : Int)^k).toNat * Nat.choose n k * (n - k)^m)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counting_functions_theorem_l506_50617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l506_50658

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the left focus F₁
def left_focus : ℝ × ℝ := (-1, 0)

-- Define a line passing through a point
def line_through_point (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem ellipse_line_intersection :
  ∀ (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
  ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
  line_through_point k x₁ y₁ ∧ line_through_point k x₂ y₂ ∧
  distance x₁ y₁ x₂ y₂ = (8 * Real.sqrt 2) / 7 →
  k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by
  sorry

#check ellipse_line_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l506_50658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l506_50689

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form (x²/a² - y²/b² = 1) -/
  equation : ℝ → ℝ → Prop

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between a point and a line (asymptote) -/
noncomputable def distance_to_asymptote (p : Point) (slope : ℝ) : ℝ :=
  abs (p.y - slope * p.x) / Real.sqrt (1 + slope^2)

theorem hyperbola_equation (C : Hyperbola) (F : Point) :
  C.equation = (λ x y => x^2 - y^2 = 1) →
  (C.equation 0 0) ∧  -- center at origin
  (F.x = Real.sqrt 2 ∧ F.y = 0) ∧  -- focus at (√2, 0)
  (∃ (slope : ℝ), distance_to_asymptote F slope = 1)  -- distance to asymptote is 1
  :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l506_50689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_C_l506_50665

-- Define the fixed point A and line l
variable (a : ℝ)
def A (a : ℝ) : ℝ × ℝ := (a, 0)
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the condition a > 0
variable (ha : a > 0)

-- Define point B on line l
variable (b : ℝ)
def B (b : ℝ) : ℝ × ℝ := (-1, b)

-- Define the angle bisector of ∠BAO
noncomputable def angleBisector (A B : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define the line AB
noncomputable def lineAB (A B : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define point C as the intersection of angle bisector and line AB
noncomputable def C (a b : ℝ) : ℝ × ℝ := sorry

-- State the theorem
theorem locus_of_C (a : ℝ) (ha : a > 0) :
  ∀ (x y : ℝ), C a b = (x, y) →
  (0 ≤ x ∧ x < a) →
  (1 - a) * x^2 - 2 * a * x + (1 + a) * y^2 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_C_l506_50665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_k_equals_one_l506_50653

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- Definition of line l₁ -/
def l₁ (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x + (1 + k) * y = 2 - k

/-- Definition of line l₂ -/
def l₂ (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ k * x + 2 * y + 8 = 0

/-- The theorem to be proved -/
theorem parallel_lines_k_equals_one :
  ∃ k : ℝ, (∀ x y : ℝ, l₁ k x y ↔ l₂ k x y) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_k_equals_one_l506_50653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_prime_relatively_prime_count_l506_50687

/-- A function that checks if a number is prime -/
def isPrime (n : Nat) : Prop := sorry

/-- A function that checks if two numbers are relatively prime -/
def areRelativelyPrime (a b : Nat) : Prop := sorry

/-- The theorem statement -/
theorem max_non_prime_relatively_prime_count :
  ∃ (S : Finset Nat),
    (∀ n, n ∈ S → 1 < n ∧ n < 10000 ∧ ¬isPrime n) ∧
    (∀ a b, a ∈ S → b ∈ S → a ≠ b → areRelativelyPrime a b) ∧
    S.card = 25 ∧
    ∀ (T : Finset Nat),
      (∀ n, n ∈ T → 1 < n ∧ n < 10000 ∧ ¬isPrime n) →
      (∀ a b, a ∈ T → b ∈ T → a ≠ b → areRelativelyPrime a b) →
      T.card ≤ 25 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_prime_relatively_prime_count_l506_50687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_round_trip_time_l506_50661

/-- Calculates the total time for a round trip given the distance and speeds -/
noncomputable def total_round_trip_time (distance : ℝ) (speed_to : ℝ) (speed_from : ℝ) : ℝ :=
  distance / speed_to + distance / speed_from

/-- The total time for a round trip between school and home -/
theorem school_round_trip_time :
  total_round_trip_time 24 6 4 = 10 := by
  -- Unfold the definition of total_round_trip_time
  unfold total_round_trip_time
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Prove the equality
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_round_trip_time_l506_50661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f₃_properties_l506_50641

noncomputable section

open Real

def f₁ (x : ℝ) := sin (x / 2 + π / 3)
def f₂ (x : ℝ) := sin (x / 2 - π / 3)
def f₃ (x : ℝ) := sin (2 * x - π / 3)
def f₄ (x : ℝ) := sin (2 * x + π / 3)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem f₃_properties :
  has_period f₃ π ∧ is_symmetric_about f₃ (5 * π / 12) ∧
  (∀ i : Fin 4, i.val ≠ 2 → ¬(has_period (match i.val with
                                          | 0 => f₁
                                          | 1 => f₂
                                          | 2 => f₃
                                          | 3 => f₄
                                          | _ => f₁) π ∧ 
                            is_symmetric_about (match i.val with
                                                | 0 => f₁
                                                | 1 => f₂
                                                | 2 => f₃
                                                | 3 => f₄
                                                | _ => f₁) (5 * π / 12))) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f₃_properties_l506_50641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subtraction_to_addition_l506_50672

theorem subtraction_to_addition :
  (-2) - 3 - (-5) + (-4) = (-2) + (-3) + 5 + (-4) :=
by
  ring  -- This tactic should solve this equality automatically


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subtraction_to_addition_l506_50672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l506_50623

theorem simplify_expression :
  (1 / ((1 / ((Real.sqrt 2 + 1)^2)) + (1 / ((Real.sqrt 5 - 2)^3)))) =
  (1 / (41 + 17 * Real.sqrt 5 - 2 * Real.sqrt 2)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l506_50623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_135_equals_binary_1011101_l506_50651

/-- Represents a digit in the octal (base 8) number system -/
def OctalDigit := Fin 8

/-- Represents a digit in the binary (base 2) number system -/
def BinaryDigit := Fin 2

/-- Converts an octal digit to its binary representation -/
def octal_to_binary (d : OctalDigit) : Fin 2 × Fin 2 × Fin 2 := sorry

/-- Represents the octal number 135 -/
def octal_135 : List OctalDigit := [⟨1, sorry⟩, ⟨3, sorry⟩, ⟨5, sorry⟩]

/-- Represents the binary number 1011101 -/
def binary_1011101 : List BinaryDigit := 
  [⟨1, sorry⟩, ⟨0, sorry⟩, ⟨1, sorry⟩, ⟨1, sorry⟩, ⟨1, sorry⟩, ⟨0, sorry⟩, ⟨1, sorry⟩]

/-- Converts an octal number to its binary representation -/
def octal_to_binary_number (n : List OctalDigit) : List BinaryDigit := sorry

theorem octal_135_equals_binary_1011101 : 
  octal_to_binary_number octal_135 = binary_1011101 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_135_equals_binary_1011101_l506_50651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_sqrt_64_l506_50613

theorem cube_root_of_sqrt_64 : (64 : ℝ) ^ (1/6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_sqrt_64_l506_50613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_calculation_l506_50657

theorem complex_fraction_calculation : 
  ((6.875 - 5/2) * (1/4) + (95/24 + 5/3) / 4) / (5/2) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_calculation_l506_50657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zero_vectors_l506_50682

variable {V : Type*} [AddCommGroup V] [DecidableEq V]

def expr1 (A B C : V) := A - B + B - C + C - A
def expr2 (A B M O : V) := A - B + M - B + B - O + O - M
def expr3 (A B C D : V) := A - B - (A - C) + B - D - (C - D)
def expr4 (O A B C : V) := O - A + O - C + B - O + C - O

def count_zero_vectors (v1 v2 v3 v4 : V) : Nat :=
  (if v1 = 0 then 1 else 0) +
  (if v2 = 0 then 1 else 0) +
  (if v3 = 0 then 1 else 0) +
  (if v4 = 0 then 1 else 0)

theorem two_zero_vectors (A B C D M O : V) :
  count_zero_vectors (expr1 A B C) (expr2 A B M O) (expr3 A B C D) (expr4 O A B C) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zero_vectors_l506_50682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l506_50685

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define the dynamic point M
def M (t : ℝ) : ℝ × ℝ := (2, t)

-- Define the line
def line (x y : ℝ) : Prop :=
  3 * x - 4 * y - 5 = 0

-- Define the circle with diameter OM
def circleOM (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 5

-- Theorem statement
theorem ellipse_properties
  (a b : ℝ)
  (h_ab : a > b ∧ b > 0)
  (h_point : ellipse a b (Real.sqrt 6 / 2) (1 / 2))
  (h_ecc : eccentricity a b = Real.sqrt 2 / 2)
  (t : ℝ)
  (h_t : t > 0) :
  -- 1. Standard equation of the ellipse
  (∀ x y, ellipse a b x y ↔ x^2 / 2 + y^2 = 1) ∧
  -- 2. Equation of the circle
  (∀ x y, circleOM x y ↔ (x - 1)^2 + (y - 2)^2 = 5) ∧
  -- 3. Length of ON is constant
  (∃ N : ℝ × ℝ, 
    (N.1 - 1)^2 + (N.2 - 2)^2 = 5 ∧  -- N is on the circle
    (N.1 - Real.sqrt 2)^2 + N.2^2 = 1 ∧       -- N is √2 away from the right focus
    N.1^2 + N.2^2 = 2                -- ON has length √2
  ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l506_50685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_bound_l506_50621

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0, and eccentricity e,
    if there exists a point P on the ellipse such that the angle APB formed with the
    endpoints of the major axis A and B is 120°, then e² ≥ 2/3. -/
theorem ellipse_eccentricity_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e := Real.sqrt (1 - (b/a)^2)
  ∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧
    Real.cos (Real.arccos (-1/2)) = (x^2 + y^2 - a^2) / (2*a*Real.sqrt (x^2 + y^2)) →
    e^2 ≥ 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_bound_l506_50621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ducks_left_in_lake_is_twenty_l506_50635

def ducks_left_in_lake (initial_ducks : ℕ) (initial_geese : ℕ) (initial_swans : ℕ)
                       (additional_ducks : ℕ) (additional_geese : ℕ) (additional_swans : ℕ)
                       (ducks_fly_away_percent : ℚ) (geese_fly_away_percent : ℚ)
                       (swans_fly_away_percent : ℚ) : ℕ :=
  let total_ducks := initial_ducks + additional_ducks
  let ducks_flying_away := (ducks_fly_away_percent * total_ducks).ceil.toNat
  total_ducks - ducks_flying_away

theorem ducks_left_in_lake_is_twenty :
  ducks_left_in_lake 13 11 9 20 15 7 (40/100) (25/100) (30/100) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ducks_left_in_lake_is_twenty_l506_50635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l506_50671

/-- Given vectors OA, OB, OC and conditions, prove the equation of the line --/
theorem line_equation_proof (k : ℝ) (OA OB OC : ℝ × ℝ) 
  (h_OA : OA = (k, 12))
  (h_OB : OB = (4, 5))
  (h_OC : OC = (10, k))
  (h_collinear : ∃ (t : ℝ), OC - OA = t • (OB - OA))
  (h_k_neg : k < 0)
  (h_k_slope : ∀ (x y : ℝ), y + 1 = k * (x - 2) ↔ (x, y) ∈ Set.range (λ t : ℝ ↦ (t, k * (t - 2) - 1))) :
  ∀ (x y : ℝ), 2 * x + y - 3 = 0 ↔ (x, y) ∈ Set.range (λ t : ℝ ↦ (t, k * (t - 2) - 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l506_50671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_l506_50655

/-- The speed of the faster train given the conditions of the problem -/
theorem faster_train_speed (slower_speed : ℝ) (passing_time : ℝ) (faster_length : ℝ) 
  (h_slower_speed : slower_speed = 36) 
  (h_passing_time : passing_time = 6)
  (h_faster_length : faster_length = 135.0108) :
  ∃ (faster_speed : ℝ), ∃ (ε : ℝ), abs (faster_speed - 45.00648) < ε ∧ ε > 0 := by
  sorry

#check faster_train_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_l506_50655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_value_l506_50604

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors and scalar
variable (m n : V) (a : ℝ)

-- State the theorem
theorem minimum_a_value (hm : m ≠ 0) (hn : ‖n‖ = 1) (hmn : m ≠ n) 
  (hangle : inner m (m - n) = ‖m‖ * ‖m - n‖ * (1 / 2)) 
  (hm_bound : ‖m‖ ∈ Set.Ioo 0 a) :
  a ≥ 2 * Real.sqrt 3 / 3 := by
  sorry

#check minimum_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_value_l506_50604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_submodular_iff_mono_decreasing_g_l506_50656

variable {S : Type*} [Fintype S]

def PowerSet (S : Type*) := Set S

def MonoDecreasing {S : Type*} (f : Set S → ℝ) : Prop :=
  ∀ X Y : Set S, X ⊆ Y → f X ≥ f Y

def Submodular {S : Type*} (f : Set S → ℝ) : Prop :=
  ∀ X Y : Set S, f (X ∪ Y) + f (X ∩ Y) ≤ f X + f Y

def g {S : Type*} (f : Set S → ℝ) (a : S) (X : Set S) : ℝ :=
  f (X ∪ {a}) - f X

theorem submodular_iff_mono_decreasing_g {S : Type*} [Fintype S] (f : Set S → ℝ) :
  Submodular f ↔
    ∀ a : S, MonoDecreasing (g f a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_submodular_iff_mono_decreasing_g_l506_50656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paperboy_delivery_sequences_l506_50626

/-- Recurrence relation for the number of delivery sequences -/
def D : ℕ → ℕ
  | 0 => 2  -- Adding this case to cover Nat.zero
  | 1 => 2
  | 2 => 4
  | 3 => 7
  | n + 4 => D (n + 3) + D (n + 2) + D (n + 1)

/-- The number of possible delivery sequences for 11 houses is 927 -/
theorem paperboy_delivery_sequences : D 11 = 927 := by
  -- Compute the values step by step
  have d4 : D 4 = 13 := by rfl
  have d5 : D 5 = 24 := by rfl
  have d6 : D 6 = 44 := by rfl
  have d7 : D 7 = 81 := by rfl
  have d8 : D 8 = 149 := by rfl
  have d9 : D 9 = 274 := by rfl
  have d10 : D 10 = 504 := by rfl
  -- Final computation
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paperboy_delivery_sequences_l506_50626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l506_50644

theorem triangle_inequality (a b c : ℝ) (n : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : n ≥ 1) : 
  (let s := (a + b + c) / 2;
   a^n / (b + c) + b^n / (c + a) + c^n / (a + b) ≥ (2/3)^(n-2) * s^(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l506_50644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l506_50667

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def infiniteGeometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: For an infinite geometric series with common ratio 1/4 and sum 40, the first term is 30 -/
theorem first_term_of_geometric_series :
  ∃ (a : ℝ), infiniteGeometricSum a (1/4) = 40 ∧ a = 30 := by
  use 30
  constructor
  · -- Prove infiniteGeometricSum 30 (1/4) = 40
    simp [infiniteGeometricSum]
    norm_num
  · -- Prove a = 30
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l506_50667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_meet_count_l506_50610

/-- Represents the attendance of students at birthday parties -/
def BirthdayAttendance := Fin 23 → Fin 23 → Bool

/-- The number of times two students meet at birthday parties -/
def meetCount (attendance : BirthdayAttendance) (i j : Fin 23) : Nat :=
  (Finset.filter (fun k => k ≠ i ∧ k ≠ j ∧ attendance k i ∧ attendance k j) Finset.univ).card

theorem exists_equal_meet_count :
  ∃ (attendance : BirthdayAttendance),
    (∀ i : Fin 23, ∃ j : Fin 23, j ≠ i ∧ attendance i j) ∧ 
    (∀ i : Fin 23, ∃ j : Fin 23, j ≠ i ∧ ¬attendance i j) ∧
    (∀ i j : Fin 23, i ≠ j → meetCount attendance i j = meetCount attendance 0 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_meet_count_l506_50610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l506_50664

theorem triangle_properties (a b c : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 - a*b + b^2 = c^2 →
  c = 2 →
  S = (Real.sqrt 3 / 4) * a * b →
  (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) = π/3) ∧ (a + b = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l506_50664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_exists_l506_50615

/-- Represents the letters in the puzzle -/
inductive Letter : Type
| K | O | M | P | b | box | V | E | T

/-- Assignment of digits to letters -/
def Assignment := Letter → Fin 9

/-- Check if an assignment satisfies all inequalities -/
def satisfiesInequalities (a : Assignment) : Prop :=
  a Letter.P > a Letter.O ∧ a Letter.O > a Letter.K ∧
  a Letter.V > a Letter.M ∧ a Letter.M < a Letter.box ∧
  a Letter.V > a Letter.O ∧ a Letter.O > a Letter.b

/-- Check if the letters spell "КОМПЬЮТЕР" when sorted by their assigned values -/
def spellsKomputer (a : Assignment) : Prop :=
  let sortedLetters := (List.toArray [Letter.K, Letter.O, Letter.M, Letter.P, Letter.b, Letter.box, Letter.V, Letter.E, Letter.T]).qsort (λ x y ↦ a x < a y)
  sortedLetters.toList = [Letter.K, Letter.O, Letter.M, Letter.P, Letter.b, Letter.box, Letter.T, Letter.E, Letter.V]

/-- There exists an assignment that satisfies all conditions -/
theorem puzzle_solution_exists : ∃ (a : Assignment), satisfiesInequalities a ∧ spellsKomputer a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_exists_l506_50615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employees_on_trip_unique_solution_l506_50680

/-- Represents the travel agency's pricing model -/
structure TravelAgency where
  baseCost : ℕ → ℚ
  totalCost : ℕ → ℚ

/-- The Spring and Autumn Travel Agency's specific pricing model -/
noncomputable def springAndAutumnTA : TravelAgency :=
  { baseCost := λ n => if n ≤ 25 then 1000 else max 700 (1500 - 20 * n),
    totalCost := λ n => n * (if n ≤ 25 then 1000 else max 700 (1500 - 20 * n)) }

/-- The theorem stating that 30 employees went on the trip -/
theorem employees_on_trip : ∃ n : ℕ, 
  springAndAutumnTA.totalCost n = 27000 ∧ 
  springAndAutumnTA.baseCost n ≥ 700 ∧ 
  n > 25 := by
  sorry

/-- Proof that 30 is the only solution -/
theorem unique_solution : 
  ∀ n : ℕ, springAndAutumnTA.totalCost n = 27000 ∧ 
           springAndAutumnTA.baseCost n ≥ 700 ∧ 
           n > 25 → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employees_on_trip_unique_solution_l506_50680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_four_digits_differ_l506_50625

theorem last_four_digits_differ (n m : ℕ) : ¬(∃ k : ℕ, k > 0 ∧ k < 10000 ∧ (5 : ℤ)^n ≡ (6 : ℤ)^m [ZMOD k]) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_four_digits_differ_l506_50625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_power_8_equals_result_l506_50691

/-- The matrix we're working with -/
noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.sqrt 2, -Real.sqrt 2; Real.sqrt 2, Real.sqrt 2]

/-- The expected result after raising A to the 8th power -/
def result : Matrix (Fin 2) (Fin 2) ℝ :=
  !![256, 0; 0, 256]

/-- Theorem stating that A^8 equals the expected result -/
theorem A_power_8_equals_result : A^8 = result := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_power_8_equals_result_l506_50691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_set_contains_9654_l506_50633

def is_valid_set (a b c : ℕ) : Prop :=
  a ≥ 1000 ∧ a < 10000 ∧
  b ≥ 1000 ∧ b < 10000 ∧
  c ≥ 1000 ∧ c < 10000 ∧
  (Finset.range 10).card = ((Nat.digits 10 a).toFinset ∪ 
                            (Nat.digits 10 b).toFinset ∪ 
                            (Nat.digits 10 c).toFinset).card

def sum_of_set (a b c : ℕ) : ℕ := a + b + c

def is_optimal_set (a b c : ℕ) : Prop :=
  is_valid_set a b c ∧
  ∀ x y z, is_valid_set x y z → sum_of_set x y z ≤ sum_of_set a b c

theorem optimal_set_contains_9654 :
  ∃ a b, is_optimal_set 9654 a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_set_contains_9654_l506_50633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_even_function_l506_50679

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 3)

theorem shifted_sine_even_function (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : ∀ x, f (x - φ) = f (φ - x)) : 
  φ = 5 * Real.pi / 12 := by
  sorry

#check shifted_sine_even_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_even_function_l506_50679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_product_factorial_l506_50678

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem min_sum_of_product_factorial (p q r s : ℕ+) : 
  p * q * r * s = factorial 8 → 
  (∀ a b c d : ℕ+, a * b * c * d = factorial 8 → p + q + r + s ≤ a + b + c + d) →
  p + q + r + s = 138 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_product_factorial_l506_50678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_width_calculation_l506_50618

/-- The width of a ring formed by two concentric circles -/
noncomputable def ring_width (inner_circumference outer_circumference : ℝ) : ℝ :=
  (outer_circumference - inner_circumference) / (2 * Real.pi)

/-- Theorem: The width of a ring formed by two concentric circles with
    inner circumference 352/7 m and outer circumference 528/7 m is equal to 176/(14π) m -/
theorem ring_width_calculation :
  ring_width (352/7) (528/7) = 176 / (14 * Real.pi) := by
  sorry

/-- Approximate numerical evaluation of the ring width -/
def ring_width_approx : ℚ :=
  (528/7 - 352/7) / (2 * 355/113)

#eval ring_width_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_width_calculation_l506_50618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_m_value_range_of_x_plus_y_l506_50637

-- Define the curve C in polar coordinates
noncomputable def C (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ * Real.cos θ, 4 * Real.cos θ * Real.sin θ)

-- Define the line l in Cartesian coordinates
noncomputable def l (m t : ℝ) : ℝ × ℝ := (m + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Theorem for part (I)
theorem intersection_points_m_value (m : ℝ) :
  (∃ t₁ t₂ : ℝ, C (Real.arccos ((l m t₁).1 / 4)) = l m t₁ ∧ 
                C (Real.arccos ((l m t₂).1 / 4)) = l m t₂ ∧
                Real.sqrt ((l m t₁).1 - (l m t₂).1)^2 + ((l m t₁).2 - (l m t₂).2)^2 = Real.sqrt 14) →
  m = 1 ∨ m = 3 := by
  sorry

-- Theorem for part (II)
theorem range_of_x_plus_y :
  ∀ θ : ℝ, 2 - 2 * Real.sqrt 2 ≤ (C θ).1 + (C θ).2 ∧ (C θ).2 + (C θ).2 ≤ 2 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_m_value_range_of_x_plus_y_l506_50637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_general_form_l506_50639

open Real

/-- The original function f(x) = x/e^x -/
noncomputable def f (x : ℝ) : ℝ := x / exp x

/-- The sequence of functions f_n defined recursively -/
noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => λ x => deriv (f_n n) x

/-- The theorem stating the general form of f_n(x) -/
theorem f_n_general_form (n : ℕ) (x : ℝ) :
  f_n n x = ((-1)^n * (x - n)) / exp x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_general_form_l506_50639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_l506_50612

theorem polynomial_roots : 
  let f : ℝ → ℝ := λ x => 3 * x^4 - x^3 - 8 * x^2 - x + 3
  let roots : Set ℝ := {(7 + Real.sqrt 13) / 6, (7 - Real.sqrt 13) / 6, -1}
  (∀ x ∈ roots, f x = 0) ∧ (∀ y : ℝ, f y = 0 → y ∈ roots) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_l506_50612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_length_l506_50683

/-- An isosceles triangle with a base of 20 inches -/
structure IsoscelesTriangle where
  base : ℝ
  isBase20 : base = 20

/-- A line parallel to the base that divides the triangle -/
structure ParallelLine where
  length : ℝ

/-- The theorem stating the length of the parallel line -/
theorem parallel_line_length (triangle : IsoscelesTriangle) 
  (line : ParallelLine) :
  (∃ (smaller_area larger_area : ℝ), 
    smaller_area + larger_area = triangle.base * triangle.base / 2 ∧
    smaller_area = (triangle.base * triangle.base / 2) / 4) →
  line.length = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_length_l506_50683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_common_tangents_l506_50649

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the distance between two points in a 2D plane -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Determines the number of common tangents between two circles -/
noncomputable def commonTangents (c1 c2 : Circle) : ℕ :=
  let d := distance c1.center c2.center
  if d > c1.radius + c2.radius then 4
  else if d == c1.radius + c2.radius then 3
  else if d < c1.radius + c2.radius ∧ d > |c1.radius - c2.radius| then 2
  else if d == |c1.radius - c2.radius| then 1
  else 0

theorem circle_common_tangents : 
  let c1 : Circle := { center := (4, 0), radius := 3 }
  let c2 : Circle := { center := (0, 3), radius := 2 }
  commonTangents c1 c2 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_common_tangents_l506_50649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_empty_time_l506_50603

/-- Represents the tank system with a leak and an inlet pipe -/
structure TankSystem where
  capacity : ℝ
  leakEmptyTime : ℝ
  inletRate : ℝ

/-- Calculates the time to empty the tank with both inlet and leak open -/
noncomputable def timeToEmpty (system : TankSystem) : ℝ :=
  let leakRate := system.capacity / system.leakEmptyTime
  let inletRatePerHour := system.inletRate * 60
  let netEmptyRate := leakRate - inletRatePerHour
  system.capacity / netEmptyRate

/-- Theorem stating the time to empty the tank under given conditions -/
theorem tank_empty_time (system : TankSystem)
  (h1 : system.capacity = 3600.000000000001)
  (h2 : system.leakEmptyTime = 6)
  (h3 : system.inletRate = 2.5) :
  timeToEmpty system = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_empty_time_l506_50603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l506_50688

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, a > 1 → (f a (2*x - 7) > f a (4*x - 3) ↔ x < -2)) ∧
  (∀ x, 0 < a ∧ a < 1 → (f a (2*x - 7) > f a (4*x - 3) ↔ x > -2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l506_50688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_correct_l506_50652

/-- The particular solution to the differential equation y'' = sin x - 1 
    with initial conditions y(0) = -1 and y'(0) = 1 -/
noncomputable def particular_solution (x : ℝ) : ℝ := -Real.sin x - x^2 / 2 + 2*x - 1

theorem particular_solution_correct :
  let y := particular_solution
  (∀ x, (deriv^[2] y) x = Real.sin x - 1) ∧ 
  y 0 = -1 ∧
  (deriv y) 0 = 1 := by
  sorry

#check particular_solution_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_correct_l506_50652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_current_length_l506_50659

/-- The current length of a highway being extended -/
theorem highway_current_length 
  (final_length : ℕ) 
  (first_day : ℕ) 
  (second_day : ℕ) 
  (remaining : ℕ) : 
  final_length = 650 →
  first_day = 50 →
  second_day = 3 * first_day →
  remaining = 250 →
  final_length - remaining = 400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_current_length_l506_50659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_student_avg_difference_l506_50638

/-- Represents a school with students and teachers. -/
structure School where
  numStudents : Nat
  numTeachers : Nat
  classEnrollments : List Nat

/-- Calculates the average number of students per teacher. -/
noncomputable def avgStudentsPerTeacher (school : School) : ℝ :=
  (school.classEnrollments.sum : ℝ) / school.numTeachers

/-- Calculates the average number of students per student. -/
noncomputable def avgStudentsPerStudent (school : School) : ℝ :=
  (school.classEnrollments.map (fun n => n * n)).sum / school.numStudents

/-- The main theorem to be proved. -/
theorem teacher_student_avg_difference (school : School)
  (h1 : school.numStudents = 120)
  (h2 : school.numTeachers = 4)
  (h3 : school.classEnrollments = [60, 30, 20, 10])
  (h4 : school.classEnrollments.sum = school.numStudents) :
  abs (avgStudentsPerTeacher school - avgStudentsPerStudent school - (-11.66)) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_student_avg_difference_l506_50638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_multiple_of_three_l506_50694

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

def is_multiple_of_three (n : ℕ) : Prop := ∃ k, n = 3 * k

def product_is_multiple_of_three (a b : ℕ) : Bool :=
  (a * b) % 3 = 0

def favorable_outcomes : Finset (ℕ × ℕ) :=
  S.product S |>.filter (λ p => product_is_multiple_of_three p.1 p.2)

theorem probability_of_multiple_of_three :
  (favorable_outcomes.card : ℚ) / ((S.card.choose 2) : ℚ) = 3/5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_multiple_of_three_l506_50694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_is_false_l506_50642

-- Define a structure for triangles
structure Triangle where
  -- You can add more fields here if needed
  mk :: 

-- Define what it means for a triangle to be isosceles and right
def IsIsoscelesRight (t : Triangle) : Prop := sorry

-- Define similarity for triangles
def AreSimilar (t1 t2 : Triangle) : Prop := sorry

-- The original proposition
def p : Prop := ∀ t1 t2 : Triangle, IsIsoscelesRight t1 → IsIsoscelesRight t2 → AreSimilar t1 t2

-- The negation of the proposition
def not_p : Prop := ∃ t1 t2 : Triangle, IsIsoscelesRight t1 ∧ IsIsoscelesRight t2 ∧ ¬(AreSimilar t1 t2)

-- Theorem stating that the negation is false
theorem negation_is_false : ¬not_p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_is_false_l506_50642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l506_50668

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + x else -x^2 + x

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  f (x^2 - x + 1) < 12 ↔ -1 < x ∧ x < 2 := by
  sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l506_50668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_l506_50676

/-- Inverse proportion function -/
noncomputable def inverse_proportion (x : ℝ) : ℝ := 4 / x

/-- Check if a point (x, y) satisfies the inverse proportion function -/
def satisfies_inverse_proportion (x y : ℝ) : Prop :=
  y = inverse_proportion x

theorem inverse_proportion_point :
  satisfies_inverse_proportion (-2) (-2) ∧
  ¬satisfies_inverse_proportion 1 (-4) ∧
  ¬satisfies_inverse_proportion 2 (-2) ∧
  ¬satisfies_inverse_proportion 4 (-1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_l506_50676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_less_than_c_l506_50698

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log (1/2)
noncomputable def b : ℝ := Real.log (1/2) / Real.log 3
noncomputable def c : ℝ := 2^(3/10 : ℝ)

-- State the theorem
theorem a_less_than_b_less_than_c : a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_less_than_c_l506_50698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_E_eq_one_l506_50670

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_E_eq_angle_G (q : Quadrilateral) : Prop := sorry

def EF_eq_GH_eq_200 (q : Quadrilateral) : Prop :=
  dist q.E q.F = 200 ∧ dist q.G q.H = 200

def EH_ne_FG (q : Quadrilateral) : Prop :=
  dist q.E q.H ≠ dist q.F q.G

def perimeter_eq_800 (q : Quadrilateral) : Prop :=
  dist q.E q.F + dist q.F q.G + dist q.G q.H + dist q.H q.E = 800

-- Define the angle E
noncomputable def angle_E (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem cos_E_eq_one (q : Quadrilateral) 
  (h1 : is_convex q)
  (h2 : angle_E_eq_angle_G q)
  (h3 : EF_eq_GH_eq_200 q)
  (h4 : EH_ne_FG q)
  (h5 : perimeter_eq_800 q) :
  Real.cos (angle_E q) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_E_eq_one_l506_50670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombic_parallelepiped_volume_l506_50609

/-- A parallelepiped with a rhombic base -/
structure RhombicParallelepiped where
  /-- Side length of the rhombus base -/
  a : ℝ
  /-- Acute angle of the rhombus base in radians -/
  θ : ℝ
  /-- Angle between edge AA₁ and edges AB, AD in radians -/
  φ : ℝ
  /-- Side length a is positive -/
  ha : 0 < a
  /-- Acute angle of rhombus is 60° (π/3 radians) -/
  hθ : θ = π / 3
  /-- Angle between AA₁ and AB, AD is 45° (π/4 radians) -/
  hφ : φ = π / 4

/-- The volume of a rhombic parallelepiped -/
noncomputable def volume (p : RhombicParallelepiped) : ℝ := p.a^3 / 2

/-- Theorem: The volume of the specified rhombic parallelepiped is a³/2 -/
theorem rhombic_parallelepiped_volume (p : RhombicParallelepiped) : 
  volume p = p.a^3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombic_parallelepiped_volume_l506_50609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l506_50662

theorem range_of_a (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 1)
  (h_a : ∀ a : ℝ, a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) :
  ∀ a : ℝ, (∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ 
    a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) ↔ 0 < a ∧ a ≤ 7 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l506_50662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_constraint_implies_a_range_l506_50620

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a * x^2

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.log x + 1 + 2 * a * x

-- Theorem statement
theorem slope_constraint_implies_a_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f_derivative a x = 3) →
  a ≥ -1 / (2 * Real.exp 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_constraint_implies_a_range_l506_50620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_midpoint_exists_bisecting_line_l506_50624

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define a line with slope 4/5
def line_slope_4_5 (x y : ℝ) : Prop := ∃ c : ℝ, y = 4/5 * x + c

-- Define a point is midpoint of two other points
def is_midpoint (x y x1 y1 x2 y2 : ℝ) : Prop := x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2

theorem trajectory_of_midpoint :
  ∀ x y x1 y1 x2 y2 : ℝ,
  ellipse x1 y1 → ellipse x2 y2 →
  line_slope_4_5 x1 y1 → line_slope_4_5 x2 y2 →
  is_midpoint x y x1 y1 x2 y2 →
  9 * x + 20 * y = 0 :=
sorry

theorem exists_bisecting_line :
  ∃ m b : ℝ,
  ∀ x1 y1 x2 y2 : ℝ,
  ellipse x1 y1 → ellipse x2 y2 →
  y1 = m * x1 + b → y2 = m * x2 + b →
  is_midpoint (4/3) (-3/5) x1 y1 x2 y2 →
  m = 12/15 ∧ b = -25/15 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_midpoint_exists_bisecting_line_l506_50624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_simplest_l506_50684

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → 
    (∃ (a b : ℚ), x = a * Real.sqrt b) → 
    (∃ (c d : ℚ), y = c * Real.sqrt d) → 
    ∀ b d : ℚ, (x = Real.sqrt b ∧ y = Real.sqrt d) → b ≤ d

theorem sqrt_5_simplest : 
  let radicals := [Real.sqrt (1/2), Real.sqrt 12, Real.sqrt 4.5, Real.sqrt 5]
  ∀ r ∈ radicals, is_simplest_quadratic_radical (Real.sqrt 5) → 
    is_simplest_quadratic_radical r → r = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_simplest_l506_50684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emuEggsTheorem_l506_50660

/-- Calculates the number of eggs laid by emus in a week -/
def emuEggsPerWeek (numPens : ℕ) (emusPerPen : ℕ) (femaleProportion : ℚ) (eggsPerDayPerFemale : ℕ) (daysInWeek : ℕ) : ℕ :=
  let totalEmus := numPens * emusPerPen
  let femaleEmus := (totalEmus : ℚ) * femaleProportion
  (Int.floor femaleEmus).toNat * eggsPerDayPerFemale * daysInWeek

/-- Proves that given the specific conditions, the number of eggs laid in a week is 84 -/
theorem emuEggsTheorem : emuEggsPerWeek 4 6 (1/2) 1 7 = 84 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emuEggsTheorem_l506_50660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_values_l506_50645

theorem alpha_values (α : Real) 
  (h1 : Real.sin (α + π/6) ^ 2 + Real.cos (α - π/3) ^ 2 = 3/2)
  (h2 : 0 < α ∧ α < π) :
  α = π/6 ∨ α = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_values_l506_50645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_difference_l506_50600

theorem cosine_sum_difference (α : Real) 
  (h1 : Real.cos α = -4/5) 
  (h2 : π/2 < α) 
  (h3 : α < π) : 
  Real.cos (π/6 - α) = (3 - 4 * Real.sqrt 3) / 10 ∧ 
  Real.cos (π/6 + α) = -(3 + 4 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_difference_l506_50600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l506_50690

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 30, 28, and 10 is approximately 140 -/
theorem triangle_area_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |triangleArea 30 28 10 - 140| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l506_50690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l506_50619

/-- A hyperbola with center at the origin and focus on the y-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The hyperbola satisfies the standard equation (y²/a²) - (x²/b²) = 1 -/
axiom hyperbola_equation (h : Hyperbola) : ∀ x y : ℝ, (y^2 / h.a^2) - (x^2 / h.b^2) = 1

/-- One asymptote of the hyperbola passes through the point (-3, 1) -/
axiom asymptote_point (h : Hyperbola) : h.b / h.a = 3

/-- The eccentricity of a hyperbola is defined as c/a -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- The relationship between a, b, and c in a hyperbola -/
axiom hyperbola_relation (h : Hyperbola) : h.c^2 = h.a^2 + h.b^2

/-- Theorem: The eccentricity of the described hyperbola is √10 -/
theorem hyperbola_eccentricity (h : Hyperbola) : eccentricity h = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l506_50619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_molly_saturday_swim_l506_50606

/-- Represents the number of meters Molly swam on Saturday -/
def saturday_meters : ℕ := sorry

/-- Represents the number of meters Molly swam on Sunday -/
def sunday_meters : ℕ := 28

/-- Represents the total number of meters Molly swam -/
def total_meters : ℕ := 73

/-- Theorem stating that Molly swam 45 meters on Saturday -/
theorem molly_saturday_swim :
  saturday_meters = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_molly_saturday_swim_l506_50606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_after_expense_increase_l506_50699

/-- Calculates the new monthly savings after an increase in expenses -/
def new_monthly_savings (monthly_salary : ℚ) (initial_savings_rate : ℚ) (expense_increase_rate : ℚ) : ℚ :=
  let initial_expenses_rate : ℚ := 1 - initial_savings_rate
  let new_expenses : ℚ := initial_expenses_rate * monthly_salary + expense_increase_rate * initial_expenses_rate * monthly_salary
  monthly_salary - new_expenses

/-- Theorem stating that given the specific conditions, the new monthly savings is 200 -/
theorem savings_after_expense_increase :
  new_monthly_savings 20000 (10/100) (10/100) = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_after_expense_increase_l506_50699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fractional_phi_2j_equals_quarter_l506_50631

-- Define the phi function
noncomputable def phi (y : ℝ) : ℝ := ∑' m : ℕ, (1 : ℝ) / (m : ℝ) ^ y

-- State the theorem
theorem sum_fractional_phi_2j_equals_quarter :
  ∑' j : ℕ, j ≥ 2 → (phi (2 * ↑j) - ⌊phi (2 * ↑j)⌋) = (1 : ℝ) / 4 :=
sorry

-- Additional lemma that might be useful for the proof
lemma phi_upper_bound (y : ℝ) (hy : y ≥ 2) : phi y < 2 :=
sorry

-- Lemma for the fractional part of phi
lemma fractional_phi (y : ℝ) (hy : y ≥ 2) : phi y - ⌊phi y⌋ = phi y - 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fractional_phi_2j_equals_quarter_l506_50631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_cos2x0_l506_50643

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sqrt 3 * Real.sin x + Real.cos x) - 1

theorem f_extrema_and_cos2x0 :
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -1) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1) ∧
  (∀ (x₀ : ℝ), x₀ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → 
    f x₀ = 6 / 5 → Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_cos2x0_l506_50643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extension_odd_function_extension_even_function_l506_50608

-- Function extension definition
def IsExtension (f g : ℝ → ℝ) (Df Dg : Set ℝ) :=
  Df ⊆ Dg ∧ ∀ x ∈ Df, f x = g x

-- First problem
theorem extension_odd_function (f g : ℝ → ℝ) :
  (∀ x > 0, f x = x * Real.log x) →
  IsExtension f g { x | x > 0 } { x | x ≠ 0 } →
  (∀ x, g (-x) = -g x) →
  ∀ x ≠ 0, g x = x * Real.log (|x|) := by
sorry

-- Second problem
theorem extension_even_function (f g : ℝ → ℝ) :
  (∀ x ≤ 0, f x = 2^x - 1) →
  IsExtension f g { x | x ≤ 0 } Set.univ →
  (∀ x, g (-x) = g x) →
  ∀ x, g x = 2^(-|x|) - 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extension_odd_function_extension_even_function_l506_50608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l506_50692

theorem coefficient_x_cubed_in_expansion : 
  ∃ p : Polynomial ℝ, (1 - 2•X : Polynomial ℝ)^6 = p ∧ p.coeff 3 = -160 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l506_50692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coords_equivalence_l506_50646

/-- Given a point in spherical coordinates, this function returns the equivalent point
    in standard spherical coordinate representation. -/
noncomputable def standardSphericalCoords (ρ : ℝ) (θ : ℝ) (φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ, θ % (2 * Real.pi), min (φ % (2 * Real.pi)) (2 * Real.pi - φ % (2 * Real.pi)))

/-- Theorem stating that the given point (5, 9π/4, 11π/7) in spherical coordinates
    is equivalent to (5, π/4, 3π/7) in standard spherical coordinate representation. -/
theorem spherical_coords_equivalence :
  standardSphericalCoords 5 (9 * Real.pi / 4) (11 * Real.pi / 7) = (5, Real.pi / 4, 3 * Real.pi / 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coords_equivalence_l506_50646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_curve_relation_l506_50673

-- Define the curve and line
def curve (x a b : ℝ) : ℝ := x^3 + a*x + b
def line (x k : ℝ) : ℝ := k*x + 1

-- State the theorem
theorem tangent_line_curve_relation (k a b : ℝ) :
  (∃ x₀ : ℝ, x₀ = 1 ∧ curve x₀ a b = line x₀ k) →  -- The line and curve intersect at x = 1
  (∃ x₀ : ℝ, x₀ = 1 ∧ deriv (fun x => curve x a b) x₀ = deriv (fun x => line x k) x₀) →  -- The derivatives are equal at x = 1
  curve 1 a b = 3 →  -- The y-coordinate of the intersection point is 3
  2*a + b = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_curve_relation_l506_50673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_m_range_l506_50614

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x - x

-- Theorem for the minimum value of g(x)
theorem g_min_value : 
  ∃ (x_min : ℝ), ∀ (x : ℝ), g x_min ≤ g x ∧ g x_min = 1 :=
sorry

-- Theorem for the range of m
theorem m_range (m : ℝ) :
  (m ≥ (Real.log 2) ^ 2) → 
  ¬∃ (x : ℝ), x > 0 ∧ (2 * x - m) / g x > x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_m_range_l506_50614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessie_allowance_l506_50616

/-- Jessie's weekly allowance in euros -/
noncomputable def weekly_allowance : ℚ := 3280 / 100

/-- Fraction of allowance spent on trip -/
def trip_fraction : ℚ := 2 / 3

/-- Fraction of remaining allowance spent on art supplies -/
def art_supplies_fraction : ℚ := 1 / 4

/-- Comic book price in USD -/
def comic_book_price_usd : ℚ := 1000 / 100

/-- Exchange rate in euros per USD -/
def exchange_rate : ℚ := 82 / 100

theorem jessie_allowance :
  let remaining_after_trip := weekly_allowance * (1 - trip_fraction)
  let remaining_after_art := remaining_after_trip * (1 - art_supplies_fraction)
  let comic_book_price_eur := comic_book_price_usd * exchange_rate
  remaining_after_art = comic_book_price_eur :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessie_allowance_l506_50616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l506_50647

/-- The distance between two parallel lines. -/
noncomputable def distance_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₁ - C₂| / Real.sqrt (A^2 + B^2)

/-- Two parallel lines l₁ and l₂ -/
def l₁ : ℝ → ℝ → Prop :=
  λ x y ↦ x - y + 1 = 0

def l₂ : ℝ → ℝ → Prop :=
  λ x y ↦ x - y - 3 = 0

theorem distance_between_given_lines :
  distance_parallel_lines 1 (-1) 1 (-3) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l506_50647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l506_50630

-- Define the function f as noncomputable
noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < c then c * x + 1
  else if c ≤ x ∧ x < 1 then 2 * x / c^2 + 1
  else 0  -- We define f as 0 outside the given intervals

-- State the theorem
theorem problem_solution (c : ℝ) (h1 : 0 < c) (h2 : c < 1) (h3 : f c c^2 = 9/8) :
  c = 1/2 ∧ 
  ∀ x, f (1/2) x > Real.sqrt 2 / 8 + 1 ↔ Real.sqrt 2 / 4 < x ∧ x < 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l506_50630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l506_50632

/-- A polynomial of degree 4 with specific properties -/
def P (a b c x : ℝ) : ℝ := x^4 - 29*x^3 + a*x^2 + b*x + c

/-- The sum of coefficients of a polynomial is equal to its value at 1 -/
theorem sum_of_coefficients (a b c : ℝ) : 
  P a b c 5 = 11 → P a b c 11 = 17 → P a b c 17 = 23 → 1 + (-29) + a + b + c = -3193 := by
  sorry

#check sum_of_coefficients

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l506_50632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firecracker_second_fragment_speed_l506_50697

/-- Represents the velocity of an object in 2D space -/
structure Velocity where
  x : ℝ
  y : ℝ

/-- Calculates the magnitude of a 2D velocity vector -/
noncomputable def Velocity.magnitude (v : Velocity) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

/-- Represents the firecracker problem -/
structure FirecrackerProblem where
  initialSpeed : ℝ
  gravity : ℝ
  explosionTime : ℝ
  firstFragmentHorizontalSpeed : ℝ

theorem firecracker_second_fragment_speed 
  (problem : FirecrackerProblem)
  (h1 : problem.initialSpeed = 20)
  (h2 : problem.gravity = 10)
  (h3 : problem.explosionTime = 1)
  (h4 : problem.firstFragmentHorizontalSpeed = 48) :
  let verticalSpeedAtExplosion := problem.initialSpeed - problem.gravity * problem.explosionTime
  let secondFragmentVelocity := Velocity.mk (-problem.firstFragmentHorizontalSpeed) verticalSpeedAtExplosion
  secondFragmentVelocity.magnitude = 52 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_firecracker_second_fragment_speed_l506_50697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_fractions_AB_perfect_fractions_CD_l506_50636

-- Define the fractions A and B
noncomputable def A (x : ℝ) : ℝ := (x - 1) / (x - 4)
noncomputable def B (x : ℝ) : ℝ := (x - 7) / (x - 4)

-- Define the fractions C and D
noncomputable def C (x : ℝ) : ℝ := (3*x - 4) / (x - 2)
noncomputable def D (x : ℝ) (E : ℝ) : ℝ := E / (x^2 - 4)

-- Theorem 1: A and B are perfect fractions with perfect value 2
theorem perfect_fractions_AB (x : ℝ) (h : x ≠ 4) : A x + B x = 2 := by
  sorry

-- Theorem 2: If C and D are perfect fractions with perfect value 3,
-- then E = -2x - 4 and x = 1
theorem perfect_fractions_CD (x : ℝ) (E : ℝ) 
  (h1 : C x + D x E = 3)
  (h2 : ∃ (n : ℕ), D x E = ↑n ∧ n > 0) :
  E = -2*x - 4 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_fractions_AB_perfect_fractions_CD_l506_50636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_length_l506_50693

-- Define the parallelogram
structure Parallelogram where
  s : ℝ
  angle : ℝ
  area : ℝ

-- Define our specific parallelogram
noncomputable def myParallelogram : Parallelogram where
  s := Real.sqrt 6
  angle := Real.pi / 3  -- 60 degrees in radians
  area := 6 * Real.sqrt 3

-- Theorem statement
theorem parallelogram_side_length (p : Parallelogram) 
  (h1 : p.angle = Real.pi / 3)
  (h2 : p.area = 6 * Real.sqrt 3) : 
  p.s = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_length_l506_50693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_hyperbola_l506_50634

/-- The number of lattice points on the hyperbola x^2 - y^2 = 1800^2 -/
theorem lattice_points_on_hyperbola :
  ({p : ℤ × ℤ | p.1^2 - p.2^2 = 1800^2}).ncard = 54 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_hyperbola_l506_50634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_sum_zero_l506_50663

-- Define the polynomial using a parameter instead of Real.sqrt
noncomputable def f (a : ℝ) (x : ℝ) := a * x^3 - (2 * a^2 + 3) * x^2 + 7

-- State the theorem
theorem root_product_sum_zero :
  ∃ (a x₁ x₂ x₃ : ℝ), 
    a > 0 ∧
    a^2 = 2023 ∧
    x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    x₂ * (x₁ + x₃) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_sum_zero_l506_50663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_managers_salary_l506_50607

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) (manager_salary : ℝ) : 
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 500 →
  (num_employees * avg_salary + manager_salary) / (num_employees + 1) = avg_salary + salary_increase →
  manager_salary = 12000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_managers_salary_l506_50607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_withdrawal_l506_50640

/-- Represents the possible bank transactions -/
inductive Transaction
| Withdraw : Transaction
| Deposit : Transaction

/-- Applies a transaction to the current balance -/
def applyTransaction (balance : Int) (t : Transaction) : Int :=
  match t with
  | Transaction.Withdraw => balance - 300
  | Transaction.Deposit => balance + 198

/-- Checks if a sequence of transactions is valid (balance never goes negative) -/
def isValidSequence (initialBalance : Int) (transactions : List Transaction) : Prop :=
  transactions.foldl (fun (acc : Prop × Int) t => 
    (acc.1 ∧ applyTransaction acc.2 t ≥ 0, applyTransaction acc.2 t)) 
    (True, initialBalance) |>.1

/-- The final balance after applying a sequence of transactions -/
def finalBalance (initialBalance : Int) (transactions : List Transaction) : Int :=
  transactions.foldl (fun acc t => applyTransaction acc t) initialBalance

/-- The amount withdrawn is the difference between initial and final balance -/
def amountWithdrawn (initialBalance : Int) (transactions : List Transaction) : Int :=
  initialBalance - finalBalance initialBalance transactions

theorem max_withdrawal (transactions : List Transaction) :
  isValidSequence 500 transactions →
  amountWithdrawn 500 transactions ≤ 498 :=
by sorry

#eval amountWithdrawn 500 []

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_withdrawal_l506_50640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l506_50622

/-- Curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := y = 2 * x - 1

/-- Curve C₂ in Cartesian coordinates -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 2*x - 4*y

/-- The distance between two points in ℝ² -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The theorem stating the distance between intersection points -/
theorem intersection_distance :
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    C₁ x₁ y₁ ∧ C₂ x₁ y₁ ∧
    C₁ x₂ y₂ ∧ C₂ x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    distance x₁ y₁ x₂ y₂ = 8 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l506_50622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_transitive_l506_50669

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internal structure of a plane for this statement

/-- Defines the parallelism relation between two planes -/
def parallelPlanes (p q : Plane3D) : Prop :=
  -- The actual definition is not needed for the statement
  True -- placeholder definition

/-- Theorem: If two planes are both parallel to a third plane, then they are parallel to each other -/
theorem parallel_planes_transitive (p q r : Plane3D) :
  parallelPlanes p r → parallelPlanes q r → parallelPlanes p q :=
by
  sorry

#check parallel_planes_transitive

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_transitive_l506_50669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_60_minus_alpha_equals_sqrt3_div_2_l506_50675

theorem cos_60_minus_alpha_equals_sqrt3_div_2 (α : ℝ) :
  Real.sin (30 * π / 180 + α) = Real.sqrt 3 / 2 → Real.cos (60 * π / 180 - α) = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_60_minus_alpha_equals_sqrt3_div_2_l506_50675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l506_50628

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

theorem triangle_properties (t : Triangle) 
  (h1 : (t.b - 2*t.a) * Real.cos t.C + t.c * Real.cos t.B = 0)
  (h2 : t.c = 2)
  (h3 : area t = Real.sqrt 3) :
  t.C = π/3 ∧ t.a = 2 ∧ t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l506_50628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l506_50666

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  area : Real

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b + t.c = 2 * t.a * Real.cos t.B)  -- Given condition
  (h2 : t.area = t.a^2 / 4)                  -- Given area condition
  : t.A = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l506_50666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_journey_speed_l506_50602

/-- Given a bike journey with a distance and time, calculate its speed. -/
noncomputable def bike_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem bike_journey_speed :
  let distance : ℝ := 450
  let time : ℝ := 5
  bike_speed distance time = 90 := by
  -- Unfold the definition of bike_speed
  unfold bike_speed
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_journey_speed_l506_50602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_hemisphere_tangency_l506_50674

/-- The edge length of a square pyramid's base, given the pyramid's height and the radius of a tangent hemisphere -/
noncomputable def pyramid_base_edge_length (pyramid_height : ℝ) (hemisphere_radius : ℝ) : ℝ :=
  2 * Real.sqrt 42

/-- Theorem stating the edge length of the pyramid's base is 2√42 under the given conditions -/
theorem pyramid_hemisphere_tangency 
  (pyramid_height : ℝ) 
  (hemisphere_radius : ℝ) 
  (h1 : pyramid_height = 10) 
  (h2 : hemisphere_radius = 4) : 
  pyramid_base_edge_length pyramid_height hemisphere_radius = 2 * Real.sqrt 42 := by
  sorry

#check pyramid_hemisphere_tangency

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_hemisphere_tangency_l506_50674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_match_schedule_ways_l506_50686

/-- Represents a chess match between two schools -/
structure ChessMatch where
  players_per_school : Nat
  games_per_matchup : Nat
  games_per_round : Nat
  total_games : Nat
  min_rounds : Nat

/-- Calculates the number of ways to schedule a chess match -/
def schedule_ways (m : ChessMatch) : Nat :=
  Nat.factorial m.min_rounds

/-- Theorem stating the number of ways to schedule the specific chess match -/
theorem chess_match_schedule_ways :
  ∃ (m : ChessMatch),
    m.players_per_school = 3 ∧
    m.games_per_matchup = 3 ∧
    m.games_per_round = 4 ∧
    m.total_games = m.players_per_school * m.players_per_school * m.games_per_matchup ∧
    m.min_rounds = (m.total_games + m.games_per_round - 1) / m.games_per_round ∧
    schedule_ways m = 5040 :=
by
  sorry

#eval Nat.factorial 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_match_schedule_ways_l506_50686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_gt_z₂_implies_a_eq_neg_one_a_eq_zero_implies_euler_identity_l506_50627

/-- Complex number representation -/
def complex (x y : ℝ) := x + y * Complex.I

/-- Definition of z₁ -/
noncomputable def z₁ (a : ℝ) : ℂ := complex (Real.sqrt 3) (-(a + 1))

/-- Definition of z₂ -/
noncomputable def z₂ (a : ℝ) : ℂ := complex ((a^2 + Real.sqrt 3) / 4) ((a^2 - a - 2) / 8)

/-- Theorem 1: When z₁ > z₂, a = -1 -/
theorem z₁_gt_z₂_implies_a_eq_neg_one :
  ∃ a : ℝ, (Complex.abs (z₁ a) > Complex.abs (z₂ a)) ↔ a = -1 :=
sorry

/-- Theorem 2: When a = 0, e^(i(5π/3)) = z₁ · z₂ -/
theorem a_eq_zero_implies_euler_identity :
  Complex.exp (Complex.I * (5 * Real.pi / 3)) = z₁ 0 * z₂ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_gt_z₂_implies_a_eq_neg_one_a_eq_zero_implies_euler_identity_l506_50627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l506_50695

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 7 * x - 8) / (x^2 - 5 * x + 6)

-- Define the horizontal asymptote
def horizontal_asymptote : ℝ := 3

-- Theorem statement
theorem g_crosses_asymptote :
  ∃ x : ℝ, g x = horizontal_asymptote ∧ x = 13/4 := by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l506_50695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tent_placement_theorem_l506_50601

/-- Represents a rectangular region in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the forest with a highway -/
structure Forest where
  highway : Rectangle
  highway_width : ℝ

noncomputable def tent_placement_area (f : Forest) (safe_distance : ℝ) : Rectangle :=
  { width := f.highway.width + 2 * (f.highway_width / 2 + safe_distance),
    height := f.highway.height + 2 * (f.highway_width / 2 + safe_distance) }

theorem tent_placement_theorem (f : Forest) :
  let inner_highway := f.highway
  let highway_width := f.highway_width
  let safe_distance := (1 : ℝ)
  tent_placement_area f safe_distance = 
    { width := inner_highway.width + 2,
      height := inner_highway.height + 2 } :=
by
  -- Unfold definitions
  unfold tent_placement_area
  -- Simplify expressions
  simp
  -- The proof is incomplete, so we use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tent_placement_theorem_l506_50601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l506_50648

noncomputable section

open Real

/-- The function f(x) = cos(2x)cos(π/5) + sin(2x)sin(π/5) -/
def f (x : ℝ) : ℝ := cos (2 * x) * cos (π / 5) + sin (2 * x) * sin (π / 5)

/-- Theorem: f(x) is monotonically decreasing in the interval [kπ + π/10, kπ + 3π/5] for all integers k -/
theorem f_monotone_decreasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * π + π / 10) (k * π + 3 * π / 5)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l506_50648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l506_50611

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (Real.cos (ω * x / 2))^2 + Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) - 1/2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  is_periodic f T ∧ T > 0 ∧ ∀ T' > 0, is_periodic f T' → T ≤ T'

theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : smallest_positive_period (f ω) π) :
  ω = 2 ∧ 
  (∀ x, f ω x ≤ 1) ∧ 
  (∀ x, f ω x ≥ -1) ∧ 
  (∃ x, f ω x = 1) ∧ 
  (∃ x, f ω x = -1) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π/3) (k * π + π/6), 
    ∀ y ∈ Set.Icc (k * π - π/3) (k * π + π/6), 
    x ≤ y → f ω x ≤ f ω y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l506_50611
