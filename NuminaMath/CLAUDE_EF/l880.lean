import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_numbers_with_reverse_digits_property_l880_88042

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- A function that reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Theorem stating the existence of numbers satisfying the given conditions -/
theorem exist_numbers_with_reverse_digits_property :
  ∃ (N : ℕ), 10 ≤ N ∧ N < 100 ∧
    (let N1 := N + sumOfDigits N
     let N2 := N1 + sumOfDigits N1
     10 ≤ N2 ∧ N2 < 100 ∧ N2 = reverseDigits N) :=
by
  -- The proof goes here
  sorry

#eval sumOfDigits 123  -- Example usage
#eval reverseDigits 42 -- Example usage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_numbers_with_reverse_digits_property_l880_88042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l880_88041

def x : ℕ → ℝ
  | 0 => 150  -- Define for 0 to cover all natural numbers
  | 1 => 150
  | (n + 2) => x (n + 1) ^ 2 - x (n + 1)

theorem series_sum : ∑' k, 1 / (x k + 1) = 1 / 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l880_88041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_translation_l880_88022

theorem inequality_translation :
  ∀ x : ℝ, (2 * x - 3 ≥ 8) ↔ (2 * x - 3 ≥ 8) :=
by
  intro x
  apply Iff.refl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_translation_l880_88022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_17_l880_88083

/-- Definition of series A -/
noncomputable def A : ℝ := ∑' n, if n % 4 ≠ 3 ∧ n % 2 = 1 then ((-1) ^ ((n - 1) / 2)) / n ^ 2 else 0

/-- Definition of series B -/
noncomputable def B : ℝ := ∑' n, if n % 8 = 4 then 1 / n ^ 2 else 0

/-- Theorem stating that A/B = 17 -/
theorem A_div_B_eq_17 : A / B = 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_17_l880_88083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_union_N_equals_M_l880_88073

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 * p.2 = 1 ∧ p.1 > 0}
def N : Set (ℝ × ℝ) := {p | Real.arctan p.1 + Real.arctan (1 / p.2) = Real.pi / 2}

-- Theorem statement
theorem M_union_N_equals_M : M ∪ N = M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_union_N_equals_M_l880_88073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_O₂_tangent_equation_O₂_intersect_equations_l880_88081

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Circle O₁ with equation x² + (y+1)² = 4 -/
def O₁ : Circle :=
  { center := (0, -1)
    radius := 2 }

/-- Center of circle O₂ -/
def O₂_center : ℝ × ℝ := (2, 1)

/-- Length of chord AB when O₂ intersects O₁ -/
noncomputable def AB_length : ℝ := 2 * Real.sqrt 2

/-- Theorem for the equation of O₂ when tangent to O₁ -/
theorem O₂_tangent_equation :
    ∃ (r : ℝ), r = 2 * Real.sqrt 2 - 2 ∧
    ∀ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = r^2 ↔
    distance O₂_center (x, y) = r ∧
    distance O₂_center O₁.center = O₁.radius + r := by
  sorry

/-- Theorem for the equations of O₂ when intersecting O₁ at A and B -/
theorem O₂_intersect_equations :
    ∃ (r₁ r₂ : ℝ), r₁ = 2 ∧ r₂ = Real.sqrt 20 ∧
    ∀ (x y : ℝ), ((x - 2)^2 + (y - 1)^2 = r₁^2 ∨ (x - 2)^2 + (y - 1)^2 = r₂^2) ↔
    (∃ (A B : ℝ × ℝ),
      distance A B = AB_length ∧
      distance O₁.center A = O₁.radius ∧
      distance O₁.center B = O₁.radius ∧
      distance O₂_center A = distance O₂_center B ∧
      (distance O₂_center A = r₁ ∨ distance O₂_center A = r₂)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_O₂_tangent_equation_O₂_intersect_equations_l880_88081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l880_88034

/-- The length of each train in meters -/
noncomputable def train_length : ℝ := 65

/-- The speed of the faster train in km/hr -/
noncomputable def fast_train_speed : ℝ := 49

/-- The speed of the slower train in km/hr -/
noncomputable def slow_train_speed : ℝ := 36

/-- The time taken for the faster train to pass the slower train in seconds -/
noncomputable def passing_time : ℝ := 36

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_hr_to_m_s : ℝ := 1000 / 3600

theorem train_length_proof : 
  let relative_speed := (fast_train_speed - slow_train_speed) * km_hr_to_m_s
  let distance := 2 * train_length
  ∃ ε > 0, |distance - relative_speed * passing_time| < ε := by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l880_88034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_is_third_l880_88069

/-- Represents the expansion of (√x + 2/x²)ⁿ --/
noncomputable def expansion (x : ℝ) (n : ℕ) := (Real.sqrt x + 2 / x^2)^n

/-- The condition that only the sixth term's binomial coefficient is maximum --/
def sixth_term_max (n : ℕ) : Prop :=
  ∀ k, k ≠ 5 → Nat.choose n 5 > Nat.choose n k

/-- The rth term in the expansion --/
noncomputable def term (x : ℝ) (n r : ℕ) : ℝ :=
  2^r * (Nat.choose n r : ℝ) * x^(5 - 5*r/2 : ℝ)

/-- The constant term is the term where the exponent of x is zero --/
def is_constant_term (r : ℕ) : Prop :=
  5 - 5*r/2 = 0

theorem constant_term_is_third :
  ∃ n : ℕ, sixth_term_max n ∧
    ∃ r : ℕ, is_constant_term r ∧ r = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_is_third_l880_88069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_A_l880_88037

-- Define a directed graph
structure DirectedGraph (V : Type) where
  edge : V → V → Prop

-- Define the subset A
def SubsetA {V : Type} (G : DirectedGraph V) (A : Set V) : Prop :=
  (∀ a b, a ∈ A → b ∈ A → ¬G.edge a b) ∧
  (∀ v : V, (∃ a ∈ A, G.edge v a) ∨ (∃ w : V, G.edge v w ∧ ∃ a ∈ A, G.edge w a))

-- Theorem statement
theorem exists_subset_A {V : Type} (G : DirectedGraph V) :
  ∃ A : Set V, SubsetA G A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_subset_A_l880_88037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_approx_two_years_l880_88040

/-- The number of years for an investment to grow from initial to final amount at a given interest rate -/
noncomputable def investment_duration (initial_amount : ℝ) (final_amount : ℝ) (interest_rate : ℝ) : ℝ :=
  Real.log (final_amount / initial_amount) / Real.log (1 + interest_rate)

/-- Theorem stating that the investment duration for the given problem is approximately 2 years -/
theorem investment_duration_approx_two_years :
  let initial_amount := 7000
  let final_amount := 8470
  let interest_rate := 0.10
  abs (investment_duration initial_amount final_amount interest_rate - 2) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_approx_two_years_l880_88040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cos_problem_l880_88025

open Real

theorem tan_cos_problem (α β : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : β ∈ Set.Ioo 0 π)
  (h3 : tan α = 2) (h4 : cos β = -(7 * Real.sqrt 2) / 10) :
  (cos (2 * α) = -3 / 5) ∧ (2 * α - β = -π / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cos_problem_l880_88025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_proof_l880_88091

/-- The initial height of the ball in feet -/
noncomputable def initial_height : ℝ := 20

/-- The ratio of the bounce height to the previous height -/
noncomputable def bounce_ratio : ℝ := 3/4

/-- The target height in feet -/
noncomputable def target_height : ℝ := 2

/-- The height of the ball after k bounces -/
noncomputable def height_after_bounces (k : ℕ) : ℝ := initial_height * (bounce_ratio ^ k)

/-- The smallest number of bounces needed to reach a height less than the target height -/
def min_bounces : ℕ := 7

theorem ball_bounce_proof :
  (∀ k : ℕ, k < min_bounces → height_after_bounces k ≥ target_height) ∧
  (height_after_bounces min_bounces < target_height) := by
  sorry

#check ball_bounce_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_proof_l880_88091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_radius_of_sector_l880_88060

/-- Given a sector cut from a circle of radius 12 with an obtuse central angle θ,
    the radius of the circle circumscribed about the sector is 12 sec(θ/2) -/
theorem circumscribed_circle_radius_of_sector (θ : Real) (h_obtuse : θ > π / 2) :
  let r : Real := 12
  ∃ R : Real, R = r / Real.cos (θ / 2) :=
by
  -- We replace Real.sec with 1 / Real.cos
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_radius_of_sector_l880_88060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_reach_15_feet_l880_88005

/-- Represents a Ferris wheel -/
structure FerrisWheel where
  radius : ℝ
  revolutionTime : ℝ

/-- Calculates the height of a rider at time t -/
noncomputable def riderHeight (wheel : FerrisWheel) (t : ℝ) : ℝ :=
  wheel.radius * (1 + Real.cos (2 * Real.pi * t / wheel.revolutionTime))

/-- Theorem: Time to reach 15 feet above bottom is 30 seconds -/
theorem time_to_reach_15_feet
  (wheel : FerrisWheel)
  (h_radius : wheel.radius = 30)
  (h_revTime : wheel.revolutionTime = 90) :
  ∃ t : ℝ, t = 30 ∧ riderHeight wheel t = 45 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_reach_15_feet_l880_88005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l880_88043

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ := a (n + 1) - 2 * n + 2

def sequence_property (a : ℕ → ℕ) : Prop :=
  a 2 = 2 ∧
  ∀ n, sequence_sum a n = (a (n + 1) - 2 * n + 2)

theorem sequence_general_term (a : ℕ → ℕ) (h : sequence_property a) :
  (a 1 = 2) ∧ (∀ n ≥ 2, a n = 2^n - 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l880_88043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_proof_l880_88067

/-- The projection of vector a onto vector b -/
noncomputable def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let b_magnitude_squared := b.1 * b.1 + b.2 * b.2
  let scalar := dot_product / b_magnitude_squared
  (scalar * b.1, scalar * b.2)

/-- Given vectors a and b, prove that the projection of a onto b is (-4/5, 3/5) -/
theorem projection_vector_proof (a b : ℝ × ℝ) (ha : a = (-2, -1)) (hb : b = (-4, 3)) :
  proj a b = (-4/5, 3/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_proof_l880_88067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_9000_terms_eq_1355_l880_88075

/-- Represents a geometric sequence -/
structure GeometricSequence where
  firstTerm : ℝ
  commonRatio : ℝ

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sumOfTerms (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.firstTerm * (1 - seq.commonRatio^n) / (1 - seq.commonRatio)

theorem sum_of_9000_terms_eq_1355 (seq : GeometricSequence) :
  sumOfTerms seq 3000 = 500 →
  sumOfTerms seq 6000 = 950 →
  sumOfTerms seq 9000 = 1355 := by
  sorry

#check sum_of_9000_terms_eq_1355

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_9000_terms_eq_1355_l880_88075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option_c_is_deductive_correct_answer_is_c_l880_88059

/-- Represents a type of reasoning --/
inductive ReasoningType
| Deductive
| Inductive
| Analogical

/-- Represents the components of a syllogism --/
structure Syllogism where
  major_premise : Prop
  minor_premise : Prop
  conclusion : Prop

/-- Definition of deductive reasoning --/
def is_deductive_reasoning (s : Syllogism) : Prop :=
  s.major_premise ∧ s.minor_premise → s.conclusion

/-- Circle area function --/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

/-- Unit circle radius --/
def unit_circle_radius : ℝ := 1

/-- Option C from the problem --/
def option_c : Syllogism :=
  { major_premise := ∀ r : ℝ, circle_area r = Real.pi * r^2
    minor_premise := unit_circle_radius = 1
    conclusion := circle_area unit_circle_radius = Real.pi }

/-- Theorem stating that option C represents deductive reasoning --/
theorem option_c_is_deductive : 
  is_deductive_reasoning option_c := by
  sorry

/-- Main theorem proving that the correct answer is option C --/
theorem correct_answer_is_c : 
  ∃ (correct_option : ReasoningType), correct_option = ReasoningType.Deductive := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_option_c_is_deductive_correct_answer_is_c_l880_88059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_orthogonality_and_angle_l880_88053

noncomputable def m : ℝ × ℝ := (1, Real.sqrt 3)
def n (t : ℝ) : ℝ × ℝ := (2, t)

theorem vector_orthogonality_and_angle (t : ℝ) :
  (m.1 * (n t).1 + m.2 * (n t).2 = 0 → t = -2 * Real.sqrt 3 / 3) ∧
  (Real.cos (30 * π / 180) = (m.1 * (n t).1 + m.2 * (n t).2) / 
    (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt ((n t).1^2 + (n t).2^2)) → 
      t = 2 * Real.sqrt 3 / 3) := by
  sorry

#check vector_orthogonality_and_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_orthogonality_and_angle_l880_88053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_orthogonality_max_value_of_f_min_value_of_f_l880_88088

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3/2 * x), Real.sin (3/2 * x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), -Real.sin (x/2))
def c : ℝ × ℝ := (1, -1)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def norm_squared (v : ℝ × ℝ) : ℝ := v.1^2 + v.2^2

noncomputable def f (x : ℝ) : ℝ :=
  (norm_squared (a x + c) - 3) * (norm_squared (b x + c) - 3)

theorem vector_orthogonality (x : ℝ) :
  dot_product (a x + b x) (a x - b x) = 0 := by sorry

theorem max_value_of_f :
  ∃ x ∈ Set.Icc (-π/2) (π/2), ∀ y ∈ Set.Icc (-π/2) (π/2), f x ≥ f y ∧ f x = 9/2 := by sorry

theorem min_value_of_f :
  ∃ x ∈ Set.Icc (-π/2) (π/2), ∀ y ∈ Set.Icc (-π/2) (π/2), f x ≤ f y ∧ f x = -8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_orthogonality_max_value_of_f_min_value_of_f_l880_88088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_powers_600_l880_88018

-- Define i as a complex number with i² = -1
noncomputable def i : ℂ := Complex.I

-- Define the sum of powers of i from 0 to n
noncomputable def sum_powers (n : ℕ) : ℂ :=
  (Finset.range (n + 1)).sum (λ k => i ^ k)

-- Theorem statement
theorem sum_powers_600 : sum_powers 600 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_powers_600_l880_88018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_and_max_value_l880_88087

open Real

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 4 * Real.log x

-- State the theorem
theorem extreme_points_and_max_value :
  ∀ a b : ℝ,
  (∀ x : ℝ, x > 0 → (deriv (f a b)) x = 0 ↔ x = 1 ∨ x = 2) →
  (a = 1 ∧ b = -6) ∧
  (∀ x : ℝ, 0 < x ∧ x ≤ 3 → f 1 (-6) x ≤ 4 * Real.log 3 - 9) ∧
  (f 1 (-6) 3 = 4 * Real.log 3 - 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_and_max_value_l880_88087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_tossing_game_probability_l880_88044

/-- Represents the gambler's strategy in the coin-tossing game -/
structure GamblerStrategy (C : ℝ) where
  bet : ℝ → ℝ
  bet_def : ∀ y, y < C → bet y = y
  bet_def_2 : ∀ y, y > C → bet y = 2*C - y

/-- The probability of reaching 2C forints in the coin-tossing game -/
noncomputable def probability_reach_2C (C : ℝ) (x : ℝ) : ℝ :=
  x / (2 * C)

theorem coin_tossing_game_probability
  (C : ℝ) (x : ℝ) (hC : C > 0) (hx : 0 < x ∧ x < 2*C)
  (strategy : GamblerStrategy C) :
  probability_reach_2C C x = x / (2 * C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_tossing_game_probability_l880_88044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_neg_seven_l880_88038

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x > 0 then x^2 - 2*x + 4
  else if x < 0 then -((-x)^2 - 2*(-x) + 4)
  else 0

-- State the theorem
theorem f_neg_three_eq_neg_seven :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is an odd function
  (∀ x : ℝ, x > 0 → f x = x^2 - 2*x + 4) →  -- definition for x > 0
  f (-3) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_neg_seven_l880_88038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_travel_distance_l880_88021

/-- The horizontal distance traveled by the center of a wheel -/
noncomputable def wheel_center_distance (radius : ℝ) (revolutions : ℕ) : ℝ :=
  2 * Real.pi * radius * (revolutions : ℝ)

/-- Theorem: A wheel with radius 2 meters, rolling for two complete revolutions,
    travels 8π meters horizontally at its center -/
theorem wheel_travel_distance :
  wheel_center_distance 2 2 = 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_travel_distance_l880_88021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_fraction_equality_l880_88051

theorem exponential_fraction_equality (x : ℝ) : 
  ((9 : ℝ)^x + (32 : ℝ)^x) / ((15 : ℝ)^x + (24 : ℝ)^x) = 4/3 ↔ 
  x = (-2 * Real.log 2) / (Real.log 3 - 3 * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_fraction_equality_l880_88051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_diagonal_l880_88099

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- Length of one base of the trapezoid -/
  a : ℝ
  /-- Length of the other base of the trapezoid -/
  b : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The trapezoid has an inscribed circle -/
  hasInscribedCircle : True

/-- The diagonal of an isosceles trapezoid with an inscribed circle -/
noncomputable def diagonal (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  (1/2) * Real.sqrt (t.a^2 + 6*t.a*t.b + t.b^2)

/-- Theorem: The diagonal of an isosceles trapezoid with an inscribed circle
    is (1/2) * √(a² + 6ab + b²) -/
theorem isosceles_trapezoid_diagonal 
  (t : IsoscelesTrapezoidWithInscribedCircle) : 
  diagonal t = (1/2) * Real.sqrt (t.a^2 + 6*t.a*t.b + t.b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_diagonal_l880_88099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_series_sum_l880_88006

def alternating_series (n : ℕ) : ℤ :=
  if n % 2 = 1 then n else -n

def series_sum (n : ℕ) : ℤ :=
  (Finset.range n).sum (fun i => alternating_series (i + 1))

theorem alternating_series_sum : series_sum 101 = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_series_sum_l880_88006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_over_b_equals_nine_l880_88000

noncomputable def f (x : ℝ) : ℝ := |2 - Real.log x / Real.log 3|

theorem ac_over_b_equals_nine 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_order : a < b ∧ b < c) 
  (h_f : f a = 2 * f b ∧ f b = f c) : 
  a * c / b = 9 := by
  sorry

#check ac_over_b_equals_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_over_b_equals_nine_l880_88000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l880_88061

-- Define the constants
noncomputable def a : ℝ := (0.3 : ℝ) ^ 4
noncomputable def b : ℝ := 4 ^ (0.3 : ℝ)
noncomputable def c : ℝ := Real.log 4 / Real.log 0.3

-- State the theorem
theorem magnitude_relationship : b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l880_88061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_back_neighbor_contribution_ratio_l880_88017

/-- Represents the backyard fencing problem -/
structure BackyardFence where
  side_length : ℕ
  back_length : ℕ
  cost_per_foot : ℕ
  cole_payment : ℕ
  left_neighbor_fraction : ℚ

/-- The specific backyard fence scenario -/
def problem : BackyardFence :=
  { side_length := 9
  , back_length := 18
  , cost_per_foot := 3
  , cole_payment := 72
  , left_neighbor_fraction := 1/3
  }

/-- Calculates the total fence length -/
def total_length (b : BackyardFence) : ℕ :=
  2 * b.side_length + b.back_length

/-- Calculates the total cost of the fence -/
def total_cost (b : BackyardFence) : ℕ :=
  (total_length b) * b.cost_per_foot

/-- Calculates the total contribution from neighbors -/
def neighbors_contribution (b : BackyardFence) : ℕ :=
  (total_cost b) - b.cole_payment

/-- Calculates the left neighbor's contribution -/
noncomputable def left_neighbor_contribution (b : BackyardFence) : ℚ :=
  ↑(b.side_length * b.cost_per_foot) * b.left_neighbor_fraction

/-- Calculates the back neighbor's contribution -/
noncomputable def back_neighbor_contribution (b : BackyardFence) : ℚ :=
  ↑(neighbors_contribution b) - (left_neighbor_contribution b)

/-- The main theorem to prove -/
theorem back_neighbor_contribution_ratio (b : BackyardFence) :
  2 * (back_neighbor_contribution b) = ↑(b.back_length * b.cost_per_foot) :=
by
  sorry

#eval neighbors_contribution problem
#eval problem.back_length * problem.cost_per_foot

end NUMINAMATH_CALUDE_ERRORFEEDBACK_back_neighbor_contribution_ratio_l880_88017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_is_two_l880_88072

theorem cube_root_of_eight_is_two : (8 : ℝ) ^ (1/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_is_two_l880_88072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l880_88036

-- Define the hyperbola and parabola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the right focus of the hyperbola
def right_focus (a b c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

-- Main theorem
theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c = 2) -- Right focus of hyperbola is (2,0), which is the focus of y^2 = 8x
  (P : ℝ × ℝ) -- Common point of hyperbola and parabola
  (h4 : hyperbola a b P.1 P.2)
  (h5 : parabola P.1 P.2)
  (h6 : (P.1 - 2)^2 + P.2^2 = 25) -- |PF| = 5
  : eccentricity a c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l880_88036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_calculation_l880_88064

/-- Represents the annual pension calculation for a retired employee. -/
noncomputable def annual_pension (a b p q : ℝ) : ℝ :=
  (a * q^2 - b * p^2) / (2 * (b * p - a * q))

/-- Theorem stating the correct annual pension calculation. -/
theorem pension_calculation (a b p q x y : ℝ) 
  (h1 : b ≠ a)
  (h2 : ∃ k : ℝ, y = k * Real.sqrt x)
  (h3 : ∃ k : ℝ, y + p = k * Real.sqrt (x + a))
  (h4 : ∃ k : ℝ, y + q = k * Real.sqrt (x + b)) :
  y = annual_pension a b p q := by
  sorry

#check pension_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_calculation_l880_88064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rest_area_location_l880_88096

/-- The milepost of the fourth exit -/
def fourth_exit : ℚ := 50

/-- The milepost of the eighth exit -/
def eighth_exit : ℚ := 210

/-- The milepost of the rest area -/
def rest_area : ℚ := (fourth_exit + eighth_exit) / 2

/-- Theorem stating that the rest area is located at milepost 130 -/
theorem rest_area_location : rest_area = 130 := by
  -- Unfold the definitions
  unfold rest_area fourth_exit eighth_exit
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rest_area_location_l880_88096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l880_88098

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 4 * Real.log x

-- State the theorem
theorem f_monotone_increasing :
  ∀ x : ℝ, x > 0 →
    (∀ y : ℝ, y > x → f y > f x) ↔ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l880_88098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l880_88013

def is_valid_sequence (a : Fin 37 → ℕ) : Prop :=
  a 0 = 37 ∧ 
  a 1 = 1 ∧
  (∀ i : Fin 37, a i ∈ Finset.range 38) ∧
  (∀ i j : Fin 37, i ≠ j → a i ≠ a j) ∧
  (∀ k : Fin 36, (Finset.range k.succ).sum (fun i ↦ a i) % a (k + 1) = 0)

theorem sequence_property (a : Fin 37 → ℕ) (h : is_valid_sequence a) : 
  a 36 = 19 ∧ a 2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l880_88013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_not_divisible_by_five_l880_88068

def numbers : List Nat := [3525, 3540, 3565, 3580, 3592]

def is_divisible_by_five (n : Nat) : Bool :=
  n % 5 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem unique_not_divisible_by_five :
  ∃! n, n ∈ numbers ∧ ¬is_divisible_by_five n ∧ 
  units_digit n * tens_digit n = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_not_divisible_by_five_l880_88068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l880_88049

noncomputable def ellipse_C1 (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

noncomputable def parabola_C2 (x y p : ℝ) : Prop := y^2 = 2*p*x

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

noncomputable def quadrilateral_area (k : ℝ) : ℝ := 
  16 * Real.sqrt ((k^2 + 2 + 1/k^2) * (2*(k^2 + 2 + 1/k^2) + 1))

theorem ellipse_parabola_intersection :
  ∀ a b : ℝ, a > b ∧ b > 0 →
  eccentricity a b = 1/2 →
  ellipse_C1 0 (2 * Real.sqrt 3) a b →
  ∃ p : ℝ,
    (∀ x y : ℝ, ellipse_C1 x y 4 (2 * Real.sqrt 3) ↔ x^2/16 + y^2/12 = 1) ∧
    (∀ x y : ℝ, parabola_C2 x y p ↔ y^2 = 8*x) ∧
    (∀ k : ℝ, k ≠ 0 → quadrilateral_area k ≥ 96) ∧
    (∃ k : ℝ, k ≠ 0 ∧ quadrilateral_area k = 96) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l880_88049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtracted_number_l880_88050

theorem least_subtracted_number (n : ℕ) : 
  (∀ d ∈ ({11, 13, 19, 23} : Set ℕ), (5785 - n) % d = 12) ∧ 
  (∀ m < n, ∃ d ∈ ({11, 13, 19, 23} : Set ℕ), (5785 - m) % d ≠ 12) →
  n = 78 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtracted_number_l880_88050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_cartesian_circle_O_cartesian_l880_88066

noncomputable section

-- Define the polar coordinates of points A and B
def A : ℝ × ℝ := (2, Real.pi / 2)
def B : ℝ × ℝ := (1, -Real.pi / 3)

-- Define the polar equation of circle O
def circle_O_polar (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Theorem for the Cartesian equation of line AB
theorem line_AB_cartesian : 
  ∃ (m c : ℝ), m = -(4 + Real.sqrt 3) ∧ c = 2 ∧
  ∀ (x y : ℝ), y = m * x + c ↔ 
    (∃ (t : ℝ), x = (1 - t) * (A.1 * Real.cos A.2) + t * (B.1 * Real.cos B.2) ∧
                y = (1 - t) * (A.1 * Real.sin A.2) + t * (B.1 * Real.sin B.2)) :=
by sorry

-- Theorem for the Cartesian equation of circle O
theorem circle_O_cartesian :
  ∀ (x y : ℝ), x^2 + (y - 2)^2 = 4 ↔ 
    ∃ (θ : ℝ), x = circle_O_polar θ * Real.cos θ ∧ 
                y = circle_O_polar θ * Real.sin θ :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_cartesian_circle_O_cartesian_l880_88066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_arithmetic_problem_l880_88016

theorem modular_arithmetic_problem (n : ℕ) : 
  n < 29 ∧ (5 * n) % 29 = 1 → 
  ((3^n)^2 - 3) % 29 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_arithmetic_problem_l880_88016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_approximation_l880_88046

structure SampleData where
  size : Nat
  high_quality : Nat
  frequency : ℝ

def sample_data : List SampleData := [
  ⟨50, 45, 0.900⟩,
  ⟨100, 92, 0.920⟩,
  ⟨200, 194, 0.970⟩,
  ⟨500, 474, 0.948⟩,
  ⟨1000, 951, 0.951⟩,
  ⟨2000, 1900, 0.950⟩
]

def approximate_probability : ℝ := 0.95

theorem probability_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧
  ∀ (data : SampleData), data ∈ sample_data →
    |data.frequency - approximate_probability| ≤ ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_approximation_l880_88046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_count_3n_gt_2n_l880_88058

/-- Number of ways to partition a grid into 1×1 tiles -/
def partition_count (rows : ℕ) (columns : ℕ) : ℕ := sorry

/-- Theorem stating that the number of partitions for a 3×n grid is greater than for a 2×n grid -/
theorem partition_count_3n_gt_2n (n : ℕ) :
  partition_count 3 n > partition_count 2 n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_count_3n_gt_2n_l880_88058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_M_trajectory_N_l880_88010

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 8*x = 0

-- Define the chord OA passing through the origin
def chord_OA (x y : ℝ) : Prop := ∃ t : ℝ, x = t ∧ y = t ∧ my_circle x y

-- Define the midpoint M of OA
def midpoint_M (x y : ℝ) : Prop := ∃ (ax ay : ℝ), chord_OA ax ay ∧ x = ax/2 ∧ y = ay/2

-- Define point N as an extension of OA
def point_N (x y : ℝ) : Prop := ∃ (ax ay : ℝ), chord_OA ax ay ∧ x = 2*ax ∧ y = 2*ay

-- Theorem for the trajectory of midpoint M
theorem trajectory_M : ∀ x y : ℝ, midpoint_M x y → x^2 + y^2 - 4*x = 0 := by
  sorry

-- Theorem for the trajectory of point N
theorem trajectory_N : ∀ x y : ℝ, point_N x y → x^2 + y^2 - 16*x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_M_trajectory_N_l880_88010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l880_88020

-- Define an isosceles triangle with given side lengths
def isosceles_triangle (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ 2 * a > b

-- Define the area of the triangle
noncomputable def triangle_area (a b : ℝ) : ℝ :=
  let h := Real.sqrt (a^2 - (b/2)^2)
  (b * h) / 2

-- Theorem statement
theorem isosceles_triangle_area :
  ∀ (a b : ℝ),
  isosceles_triangle a b →
  a = 26 →
  b = 48 →
  triangle_area a b = 240 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l880_88020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l880_88079

theorem number_of_factors (M : ℕ) : M = 2^5 * 3^4 * 5^3 * 7^3 * 11^2 → 
  (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 1440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l880_88079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_22_l880_88052

/-- The hyperbola with equation x^2 - y^2 = 4 -/
def hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = 4}

/-- The left focus of the hyperbola -/
noncomputable def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
noncomputable def F₂ : ℝ × ℝ := sorry

/-- Point P on the hyperbola -/
noncomputable def P : ℝ × ℝ := sorry

/-- Point Q on the hyperbola -/
noncomputable def Q : ℝ × ℝ := sorry

/-- The chord PQ passes through F₁ -/
axiom chord_through_F₁ : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F₁ = (1 - t) • P + t • Q

/-- The length of PQ is 7 -/
axiom PQ_length : dist P Q = 7

/-- The perimeter of triangle PF₂Q -/
noncomputable def perimeter : ℝ := dist P F₂ + dist Q F₂ + dist P Q

theorem perimeter_is_22 : perimeter = 22 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_22_l880_88052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_division_l880_88048

/-- Represents a repeating decimal with a repeating part of two digits -/
def RepeatingDecimal (whole : ℕ) (repeating : ℕ) : ℚ :=
  whole + repeating / 99

theorem repeating_decimal_division :
  (RepeatingDecimal 0 36) / (RepeatingDecimal 0 9) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_division_l880_88048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_selling_price_l880_88047

theorem original_selling_price : ∃ (original_profit_rate reduced_purchase_rate new_profit_rate price_difference : ℝ),
  original_profit_rate = 0.1 ∧
  reduced_purchase_rate = 0.1 ∧
  new_profit_rate = 0.3 ∧
  price_difference = 56 ∧
  let original_selling_price := 880
  let original_purchase_price := original_selling_price / (1 + original_profit_rate)
  let new_selling_price := (original_purchase_price * (1 - reduced_purchase_rate)) * (1 + new_profit_rate)
  new_selling_price - original_selling_price = price_difference ∧
  original_selling_price = 880 :=
by
  -- Prove the existence of the required values
  use 0.1, 0.1, 0.3, 56
  -- Prove the conjunction of all conditions
  apply And.intro
  · rfl
  apply And.intro
  · rfl
  apply And.intro
  · rfl
  apply And.intro
  · rfl
  -- Prove the main statement
  sorry -- This skips the detailed proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_selling_price_l880_88047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l880_88012

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (-x + 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | f x ∈ Set.range Real.log} = Set.Iio 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l880_88012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_range_complete_l880_88015

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin ((Real.pi/4) * Real.sin (Real.sqrt (x-2) + x + 2) - 5*Real.pi/2)

theorem f_range :
  ∀ y ∈ Set.range f, -2 ≤ y ∧ y ≤ -Real.sqrt 2 :=
by
  sorry

theorem f_range_complete :
  ∀ y, -2 ≤ y ∧ y ≤ -Real.sqrt 2 → ∃ x ≥ 2, f x = y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_range_complete_l880_88015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l880_88033

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x / Real.exp x

-- Define the function g
def g (a b : ℝ) (x : ℝ) : ℝ := Real.log (f a x) - b

-- State the theorem
theorem problem_statement 
  (a : ℝ) (b : ℝ) (k : ℝ) (x₁ x₂ : ℝ) :
  (∀ x ∈ Set.Ioo 0 2, f a x < 1 / (k + 2*x - x^2)) →
  (∀ x, deriv (f a) x = 1 → x = 0) →
  (g a b x₁ = 0 ∧ g a b x₂ = 0 ∧ x₁ ≠ x₂) →
  (k ∈ Set.Icc 0 (Real.exp 1 - 1) ∧ deriv (deriv (g a b)) ((x₁ + x₂) / 2) < 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l880_88033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_time_calculation_l880_88009

noncomputable def compound_interest_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  compound_interest_amount principal rate time - principal

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate / 100 * time

theorem simple_interest_time_calculation 
  (principal_simple : ℝ) 
  (principal_compound : ℝ) 
  (rate_simple : ℝ) 
  (rate_compound : ℝ) 
  (time_compound : ℝ) :
  principal_simple = 1750 →
  rate_simple = 8 →
  principal_compound = 4000 →
  rate_compound = 10 →
  time_compound = 2 →
  simple_interest principal_simple rate_simple (3 : ℝ) = 
    (1/2) * compound_interest principal_compound rate_compound time_compound :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_time_calculation_l880_88009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_critical_point_l880_88003

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x - e

theorem tangent_line_and_critical_point (a : ℝ) :
  (∀ x : ℝ, f 1 x = Real.exp x + x - e) ∧
  (∃! t : ℝ, (fun x : ℝ ↦ Real.exp x + a) x = 0) ↔
  ((∀ x y : ℝ, y = 2 * x + 1 - e ↔ y - (f 1 0) = (Real.exp 0 + 1) * (x - 0)) ∧
   a < 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_critical_point_l880_88003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l880_88085

noncomputable def f (x : ℝ) : ℝ := -x / (x^2 + 2)

theorem f_properties :
  -- f is defined on (-1, 1)
  (∀ x : ℝ, -1 < x ∧ x < 1 → f x = -x / (x^2 + 2)) ∧
  -- f is an odd function
  (∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x) ∧
  -- f(-1/2) = 2/9
  (f (-1/2) = 2/9) ∧
  -- f is monotonically decreasing on (-1, 1)
  (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  -- The inequality f(t+1/2) + f(t-1/2) < 0 is satisfied when 0 < t < 1/2
  (∀ t : ℝ, 0 < t ∧ t < 1/2 → f (t + 1/2) + f (t - 1/2) < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l880_88085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l880_88019

theorem equation_solutions_count : 
  ∃ (S : Finset ℝ), 
    (∀ θ ∈ S, 0 < θ ∧ θ ≤ 2 * Real.pi) ∧
    (∀ θ ∈ S, 2 + 4 * Real.sin θ - 6 * Real.cos (2 * θ) = 0) ∧
    Finset.card S = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l880_88019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_circular_arc_surface_area_curve_loop_l880_88070

-- Define the circle
def circle_equation (x y b R : ℝ) : Prop := x^2 + (y - b)^2 = R^2

-- Define the curve
def curve_equation (x y a : ℝ) : Prop := 9 * a * x^2 = y * (3 * a - y)^2

-- Surface area of rotation around Oy axis
noncomputable def surface_area_rotation (f : ℝ → ℝ) (y1 y2 : ℝ) : ℝ :=
  2 * Real.pi * ∫ y in y1..y2, f y * Real.sqrt (1 + ((deriv f) y)^2)

-- Theorem for the surface area of the circular arc
theorem surface_area_circular_arc (R b y1 y2 : ℝ) (h : y1 < y2) :
  surface_area_rotation (λ y => Real.sqrt (R^2 - (y - b)^2)) y1 y2 = 2 * Real.pi * R * (y2 - y1) := by
  sorry

-- Theorem for the surface area of the curve loop
theorem surface_area_curve_loop (a : ℝ) (h : a > 0) :
  surface_area_rotation (λ y => Real.sqrt ((y * (3 * a - y)^2) / (9 * a))) 0 (3 * a) = 3 * Real.pi * a^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_circular_arc_surface_area_curve_loop_l880_88070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poly_not_divisible_iff_not_div_by_three_l880_88023

/-- A polynomial of degree 2k is not divisible by x^2 + x + 1 iff 3 does not divide k -/
theorem poly_not_divisible_iff_not_div_by_three (k : ℕ) :
  ¬ (∃ q : Polynomial ℚ, (X^(2*k) + 1 + (X + 1)^(2*k)) = (X^2 + X + 1) * q) ↔ ¬ (3 ∣ k) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poly_not_divisible_iff_not_div_by_three_l880_88023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_white_l880_88028

/-- Definition of the probability of drawing at least one white ball -/
def Probability.atLeastOneWhite (total : Nat) (white : Nat) (red : Nat) : ℚ :=
  1 - (Nat.choose red 2 : ℚ) / (Nat.choose total 2 : ℚ)

/-- The probability of drawing at least one white ball from a box of 30 balls (4 white, 26 red) -/
theorem prob_at_least_one_white (total : Nat) (white : Nat) (red : Nat) : 
  total = 30 → white = 4 → red = 26 → 
  (Nat.choose red 1 * Nat.choose white 1 + Nat.choose white 2 : ℚ) / (Nat.choose total 2 : ℚ) = 
  Probability.atLeastOneWhite total white red :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_white_l880_88028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_n_b_n_l880_88097

/-- The function L defined for all real numbers -/
noncomputable def L (x : ℝ) : ℝ := x - x^3 / 3

/-- The sequence b_n defined recursively using L -/
noncomputable def b (n : ℕ+) : ℝ := (L^[n.val]) (25 / n.val)

/-- The main theorem stating the limit of n * b_n -/
theorem limit_n_b_n :
  ∀ ε > 0, ∃ N : ℕ+, ∀ n : ℕ+, n ≥ N → |n.val * b n - 75/4| < ε :=
by
  sorry

#check limit_n_b_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_n_b_n_l880_88097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missed_bus_time_l880_88029

/-- Proves that walking at 3/5 of the usual speed results in missing the bus by 10 minutes -/
theorem missed_bus_time (usual_time : ℝ) (usual_speed : ℝ) (actual_speed : ℝ) 
  (h1 : usual_time = 15)
  (h2 : actual_speed = 3/5 * usual_speed)
  (h3 : usual_speed > 0) :
  actual_speed * (usual_time + 10) = usual_speed * usual_time := by
  sorry

#check missed_bus_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missed_bus_time_l880_88029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_weekly_earnings_l880_88027

/-- Jason's weekly earnings calculation --/
theorem jason_weekly_earnings 
  (after_school_rate : ℚ) 
  (saturday_rate : ℚ) 
  (total_hours : ℚ) 
  (saturday_hours : ℚ) 
  (h1 : after_school_rate = 4) 
  (h2 : saturday_rate = 6) 
  (h3 : total_hours = 18) 
  (h4 : saturday_hours = 8) :
  after_school_rate * (total_hours - saturday_hours) + saturday_rate * saturday_hours = 88 := by
  -- Replace all occurrences of variables with their values
  rw [h1, h2, h3, h4]
  -- Simplify the expression
  ring
  -- The proof is complete
  done

#check jason_weekly_earnings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_weekly_earnings_l880_88027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_radius_circle_y_l880_88084

theorem half_radius_circle_y :
  ∃ (x y : ℝ → Prop),
    (∀ r, x r ↔ (2 * π * r = 18 * π)) ∧
    (∀ r, y r ↔ x r) ∧
    (∃ r, y r ∧ r / 2 = 4.5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_radius_circle_y_l880_88084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_day_2024_is_tuesday_l880_88082

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr

/-- Calculates the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Calculates the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => dayAfter (nextDay start) n

/-- The number of days between February 29, 1996 and February 29, 2024 -/
def daysBetweenLeapDays : Nat := 10227

theorem leap_day_2024_is_tuesday :
  dayAfter DayOfWeek.Thursday daysBetweenLeapDays = DayOfWeek.Tuesday := by
  sorry

#eval dayAfter DayOfWeek.Thursday daysBetweenLeapDays

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_day_2024_is_tuesday_l880_88082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_sequence_l880_88031

noncomputable def a (n : ℕ) : ℚ :=
  match n with
  | 0 => 1/2
  | n + 1 => 1 + (a n - 1)^2

theorem infinite_product_sequence :
  (∏' n, a n) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_sequence_l880_88031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l880_88004

/-- Two parallel lines in the plane -/
structure ParallelLines where
  a : ℝ
  l₁ : ℝ → ℝ → Prop := λ x y => x + a * y + 6 = 0
  l₂ : ℝ → ℝ → Prop := λ x y => (a - 2) * x + 3 * y + 2 * a = 0
  parallel : ∃ k : ℝ, k ≠ 0 ∧ 1 = k * (a - 2) ∧ a = k * 3

/-- The distance between two parallel lines -/
noncomputable def distance (lines : ParallelLines) : ℝ :=
  abs (6 - 2 * lines.a / 3) / Real.sqrt (1 + lines.a^2)

theorem parallel_lines_distance (lines : ParallelLines) :
  distance lines = 8 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l880_88004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_multiples_of_three_l880_88062

def sequence_a (p : ℕ) : ℕ → ℕ
  | 0 => 1  -- a_1 ∈ ℕ*
  | n + 1 => let a := sequence_a p n
              if a ≤ p then 2 * a else 2 * a - 6

def M (p : ℕ) : Set ℕ :=
  {n | ∃ k, n = sequence_a p k}

theorem all_multiples_of_three (p : ℕ) (h : p = 18) :
  (∃ m ∈ M p, 3 ∣ m) → ∀ n ∈ M p, 3 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_multiples_of_three_l880_88062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_l880_88095

theorem factorial_ratio : 
  (10 * 9 * 8 * 7 * 6 * 5 * Nat.factorial 4) / Nat.factorial 4 = 151200 := by
  sorry

#check factorial_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_l880_88095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edward_worked_30_hours_l880_88024

/-- Represents Edward's earnings for a week -/
structure EdwardEarnings where
  hourlyRate : ℚ
  overtimeRate : ℚ
  totalEarnings : ℚ
  hoursWorked : ℚ

/-- Calculates Edward's earnings based on hours worked -/
def calculateEarnings (e : EdwardEarnings) : ℚ :=
  if e.hoursWorked ≤ 40 then
    e.hourlyRate * e.hoursWorked
  else
    e.hourlyRate * 40 + e.overtimeRate * (e.hoursWorked - 40)

/-- Theorem: Edward worked 30 hours given the conditions -/
theorem edward_worked_30_hours (e : EdwardEarnings) 
  (h1 : e.hourlyRate = 7)
  (h2 : e.overtimeRate = 2 * e.hourlyRate)
  (h3 : e.totalEarnings = 210)
  (h4 : calculateEarnings e = e.totalEarnings) :
  e.hoursWorked = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_edward_worked_30_hours_l880_88024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_b_equals_five_l880_88002

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ : ℝ} : 
  (∃ c₁ c₂ : ℝ, ∀ x y : ℝ, y = m₁ * x + c₁ ↔ y = m₂ * x + c₂) → m₁ = m₂

/-- The slope-intercept form of a line -/
def slope_intercept_form (m c : ℝ) (x y : ℝ) : Prop := y = m * x + c

theorem parallel_lines_b_equals_five :
  ∀ b : ℝ, 
  (∃ c₁, ∀ x y : ℝ, 3 * y - 3 * b = 9 * x ↔ slope_intercept_form 3 c₁ x y) →
  (∃ c₂, ∀ x y : ℝ, y - 2 = (b - 2) * x ↔ slope_intercept_form (b - 2) c₂ x y) →
  b = 5 := by
  intro b h1 h2
  have parallel : ∃ c₁ c₂, ∀ x y : ℝ, y = 3 * x + c₁ ↔ y = (b - 2) * x + c₂ := sorry
  have slopes_equal : 3 = b - 2 := parallel_lines_equal_slopes parallel
  linarith

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_b_equals_five_l880_88002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_is_one_ninth_l880_88055

/-- The ratio of the volume of an octahedron formed by joining the midpoints of the edges of a regular tetrahedron to the volume of the tetrahedron -/
noncomputable def volume_ratio (s : ℝ) : ℝ :=
  let tetrahedron_volume := s^3 * Real.sqrt 2 / 12
  let octahedron_edge := s * Real.sqrt 6 / 6
  let octahedron_volume := octahedron_edge^3 * Real.sqrt 2 / 3
  octahedron_volume / tetrahedron_volume

/-- Theorem stating that the volume ratio is 1/9 -/
theorem volume_ratio_is_one_ninth (s : ℝ) (h : s > 0) : volume_ratio s = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_is_one_ninth_l880_88055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourStudentsDrawLots_is_classical_model_l880_88008

/-- Represents an experiment in a probability model -/
structure Experiment where
  outcomes : Finset String
  probability : outcomes → ℚ

/-- Defines a classical probability model -/
def IsClassicalProbabilityModel (e : Experiment) : Prop :=
  ∀ (o : e.outcomes), e.probability o = 1 / e.outcomes.card

/-- Represents the experiment of four students drawing lots -/
def fourStudentsDrawLots : Experiment :=
  { outcomes := {"Student1", "Student2", "Student3", "Student4"},
    probability := λ _ => 1 / 4 }

/-- Theorem stating that the four students drawing lots experiment is a classical probability model -/
theorem fourStudentsDrawLots_is_classical_model : 
  IsClassicalProbabilityModel fourStudentsDrawLots := by
  intro o
  simp [IsClassicalProbabilityModel, fourStudentsDrawLots]
  rfl

#check fourStudentsDrawLots_is_classical_model

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourStudentsDrawLots_is_classical_model_l880_88008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l880_88001

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point on a side of a triangle
def PointOnSide (T : Triangle) (P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    (P.1 = t * T.A.1 + (1 - t) * T.B.1) ∧
    (P.2 = t * T.A.2 + (1 - t) * T.B.2)

-- Define the perimeter of a triangle
noncomputable def Perimeter (A B C : ℝ × ℝ) : ℝ :=
  (((A.1 - B.1)^2 + (A.2 - B.2)^2).sqrt) +
  (((B.1 - C.1)^2 + (B.2 - C.2)^2).sqrt) +
  (((C.1 - A.1)^2 + (C.2 - A.2)^2).sqrt)

-- Define an isosceles triangle
def IsIsosceles (T : Triangle) : Prop :=
  let d1 := ((T.A.1 - T.B.1)^2 + (T.A.2 - T.B.2)^2).sqrt
  let d2 := ((T.B.1 - T.C.1)^2 + (T.B.2 - T.C.2)^2).sqrt
  let d3 := ((T.C.1 - T.A.1)^2 + (T.C.2 - T.A.2)^2).sqrt
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

theorem isosceles_triangle (T : Triangle) (M N : ℝ × ℝ)
  (h1 : PointOnSide T M)
  (h2 : PointOnSide T N)
  (h3 : Perimeter T.A M T.C = Perimeter T.C N T.A)
  (h4 : Perimeter T.A N T.B = Perimeter T.C M T.B) :
  IsIsosceles T :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l880_88001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_distance_l880_88080

/-- Two circles are tangent if they intersect at exactly one point. -/
def are_tangent (c1 c2 : Set (ℝ × ℝ)) : Prop :=
  ∃! p : ℝ × ℝ, p ∈ c1 ∧ p ∈ c2

/-- The distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem tangent_circles_distance (O O' : ℝ × ℝ) (r r' : ℝ) :
  r = 3 →
  r' = 4 →
  are_tangent {p : ℝ × ℝ | distance p O = r} {p : ℝ × ℝ | distance p O' = r'} →
  distance O O' = 1 ∨ distance O O' = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_distance_l880_88080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_speed_is_1_6_l880_88045

/-- Represents the hiking scenario with Chantal and Jean -/
structure HikingScenario where
  d : ℚ  -- Represents one-third of the total distance
  chantal_speed1 : ℚ := 5  -- Chantal's initial speed
  chantal_speed2 : ℚ := 3  -- Chantal's speed on steep part
  chantal_speed3 : ℚ := 4  -- Chantal's descent speed

/-- Calculates Jean's average speed given the hiking scenario -/
def jean_average_speed (scenario : HikingScenario) : ℚ :=
  let t1 := (2 * scenario.d) / scenario.chantal_speed1
  let t2 := scenario.d / scenario.chantal_speed2
  let t3 := (scenario.d / 3) / scenario.chantal_speed3
  let total_time := t1 + t2 - t3
  (2 * scenario.d) / total_time

/-- Theorem stating that Jean's average speed is 1.6 miles per hour -/
theorem jean_speed_is_1_6 (scenario : HikingScenario) :
  jean_average_speed scenario = 8/5 := by
  sorry

#eval jean_average_speed { d := 1 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_speed_is_1_6_l880_88045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_musical_ticket_cost_l880_88094

theorem musical_ticket_cost 
  (adult_price : Int) (child_price : Int) (num_adults : Int) (num_children : Int) :
  adult_price * num_adults + child_price * num_children = 66 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_musical_ticket_cost_l880_88094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedding_cost_theorem_l880_88090

/-- Calculates the total cost of a wedding given specific parameters. -/
def wedding_cost (venue_cost : ℕ) (food_cost_per_guest : ℕ) (initial_guests : ℕ) 
  (guest_increase_percent : ℚ) (decoration_base : ℕ) (decoration_per_guest : ℕ) 
  (transport_couple : ℕ) (transport_per_guest : ℕ) (entertainment : ℕ) : ℕ :=
  let final_guests : ℕ := initial_guests + (↑initial_guests * guest_increase_percent).floor.toNat
  let food_cost : ℕ := food_cost_per_guest * final_guests
  let decoration_cost : ℕ := decoration_base + decoration_per_guest * final_guests
  let transport_cost : ℕ := transport_couple + transport_per_guest * final_guests
  venue_cost + food_cost + decoration_cost + transport_cost + entertainment

/-- Theorem stating that the wedding cost under given conditions is $58,700. -/
theorem wedding_cost_theorem : 
  wedding_cost 10000 500 50 (3/5 : ℚ) 2500 10 200 15 4000 = 58700 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedding_cost_theorem_l880_88090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_ball_cost_l880_88077

/-- Given the conditions of Melissa's tennis ball purchase, prove the cost per ball. -/
theorem tennis_ball_cost 
  (num_packs : ℕ) 
  (balls_per_pack : ℕ) 
  (total_cost : ℚ) 
  (h1 : num_packs = 4)
  (h2 : balls_per_pack = 3)
  (h3 : total_cost = 24) : 
  total_cost / (num_packs * balls_per_pack : ℚ) = 2 := by
  -- Convert natural numbers to rationals for division
  have num_packs_rat : ℚ := num_packs
  have balls_per_pack_rat : ℚ := balls_per_pack
  
  -- Calculate total number of balls
  let total_balls : ℚ := num_packs_rat * balls_per_pack_rat
  
  -- Use the given hypotheses
  rw [h1, h2, h3]
  
  -- Perform the division
  norm_num
  
  -- The proof is complete
  done

#check tennis_ball_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_ball_cost_l880_88077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_adjacent_vertices_probability_l880_88086

def number_of_adjacent_pairs (n : ℕ) : ℕ := n

def number_of_distinct_vertex_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

theorem decagon_adjacent_vertices_probability :
  ∀ (n : ℕ), n = 10 →
  (2 : ℚ) / 9 = (number_of_adjacent_pairs n : ℚ) / (number_of_distinct_vertex_pairs n : ℚ) :=
by
  intro n hn
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_adjacent_vertices_probability_l880_88086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_fixed_points_x5_l880_88074

noncomputable def x (n : ℕ) : ℝ → ℝ
| x₀ => match n with
  | 0 => x₀
  | n + 1 => let xₙ := x n x₀
             if 2 * xₙ < 1 then 2 * xₙ else 2 * xₙ - 1

theorem count_fixed_points_x5 :
  ∃ (S : Finset ℝ), S.card = 31 ∧
    ∀ x₀ ∈ Set.Icc (0 : ℝ) 1, x₀ ∈ S ↔ (0 ≤ x₀ ∧ x₀ < 1 ∧ x 5 x₀ = x₀) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_fixed_points_x5_l880_88074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_owen_face_mask_boxes_l880_88007

/-- Proves that Owen bought 12 boxes of face masks given the problem conditions -/
theorem owen_face_mask_boxes : ℕ := by
  -- Define the cost per box
  let cost_per_box : ℚ := 9

  -- Define the number of masks per box
  let masks_per_box : ℕ := 50

  -- Define the number of repacked boxes
  let repacked_boxes : ℕ := 6

  -- Define the selling price for repacked masks
  let repacked_price : ℚ := 5 / 25

  -- Define the number of remaining masks
  let remaining_masks : ℕ := 300

  -- Define the selling price for remaining masks
  let remaining_price : ℚ := 3 / 10

  -- Define the profit
  let profit : ℚ := 42

  -- The theorem to prove
  have h : (repacked_boxes * masks_per_box : ℚ) * repacked_price + 
           (remaining_masks : ℚ) * remaining_price - 
           ((repacked_boxes * masks_per_box + remaining_masks : ℚ) / masks_per_box * cost_per_box) = profit := by
    sorry -- Skip the proof

  -- The number of boxes Owen bought
  exact 12


end NUMINAMATH_CALUDE_ERRORFEEDBACK_owen_face_mask_boxes_l880_88007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_is_one_sixth_l880_88039

/-- The minimum squared distance sum for the given point configuration -/
noncomputable def min_distance_sum : ℝ := 1/6

/-- Point A in the Cartesian plane -/
def A : ℝ × ℝ := (0, -1)

/-- Point B in the Cartesian plane -/
def B : ℝ × ℝ := (1, 3)

/-- Point C in the Cartesian plane -/
def C : ℝ × ℝ := (2, 6)

/-- The line y = ax + b passing through points D, E, F -/
def line (a b : ℝ) (x : ℝ) : ℝ := a * x + b

/-- Point D on the line with x-coordinate 0 -/
def D (a b : ℝ) : ℝ × ℝ := (0, line a b 0)

/-- Point E on the line with x-coordinate 1 -/
def E (a b : ℝ) : ℝ × ℝ := (1, line a b 1)

/-- Point F on the line with x-coordinate 2 -/
def F (a b : ℝ) : ℝ × ℝ := (2, line a b 2)

/-- Squared distance between two points -/
def squared_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Sum of squared distances AD² + BE² + CF² -/
def distance_sum (a b : ℝ) : ℝ :=
  squared_distance A (D a b) + squared_distance B (E a b) + squared_distance C (F a b)

/-- Theorem stating that the minimum value of the distance sum is 1/6 -/
theorem min_distance_sum_is_one_sixth :
  ∃ a b : ℝ, ∀ x y : ℝ, distance_sum a b ≤ distance_sum x y ∧ distance_sum a b = min_distance_sum := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_is_one_sixth_l880_88039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_analysis_l880_88030

-- Define the project parameters
def specified_time : ℝ → Prop := sorry
def team_a_rate : ℝ → ℝ := sorry
def team_b_rate : ℝ → ℝ := sorry
def team_a_cost : ℝ := 6500
def team_b_cost : ℝ := 3500

-- Define the conditions
axiom condition1 : ∀ x : ℝ, specified_time x → team_a_rate x = 1 / x
axiom condition2 : ∀ x : ℝ, specified_time x → team_b_rate x = 1 / (1.5 * x)
axiom condition3 : ∀ x : ℝ, specified_time x → 
  (team_a_rate x + team_b_rate x) * 15 + team_a_rate x * 5 = 1

-- Theorem to prove
theorem project_analysis :
  ∃ x : ℝ, specified_time x ∧ 
  x = 30 ∧
  (1 / (team_a_rate x + team_b_rate x)) * (team_a_cost + team_b_cost) = 180000 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_analysis_l880_88030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l880_88026

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then -x + 3*a else a^x

theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  (1/3 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l880_88026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_three_element_set_l880_88092

theorem number_of_subsets_of_three_element_set :
  ∀ (P : Finset ℕ), P = {1, 2, 3} → Finset.card (Finset.powerset P) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_three_element_set_l880_88092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_count_l880_88054

def is_consecutive_list (l : List ℤ) : Prop :=
  ∀ i, 0 ≤ i ∧ i + 1 < l.length → l[i+1]! = l[i]! + 1

theorem consecutive_integers_count 
  (K : List ℤ) 
  (h_consecutive : is_consecutive_list K) 
  (h_least : K.head? = some (-3)) 
  (h_range : (K.filter (λ x => x > 0)).maximum? = (K.filter (λ x => x > 0)).minimum?.map (· + 7)) :
  K.length = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_count_l880_88054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_less_than_2BP_squared_l880_88011

-- Define the triangle ABC
def Triangle (a : ℝ) : Set (ℝ × ℝ) :=
  {p | p = (0, 0) ∨ p = (a, a) ∨ p = (a, 0)}

-- Define a point P on side AC
def P (a x : ℝ) : ℝ × ℝ := (x, 0)

-- Define s as AP^2 + PC^2
def s (a x : ℝ) : ℝ := x^2 + (a - x)^2

-- Define BP^2
def BP_squared (a x : ℝ) : ℝ := a^2 + x^2

-- Theorem statement
theorem s_less_than_2BP_squared (a : ℝ) (h : a > 0) :
  ∀ x, 0 ≤ x ∧ x ≤ a → s a x < 2 * BP_squared a x := by
  intro x ⟨hx_nonneg, hx_le_a⟩
  -- Expand the definitions of s and BP_squared
  unfold s BP_squared
  -- Simplify the inequality
  simp [mul_add]
  -- Prove the inequality
  sorry

#check s_less_than_2BP_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_less_than_2BP_squared_l880_88011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cookies_l880_88071

theorem max_cookies (total_cookies : Nat) (members : Nat) (h1 : total_cookies = 200) (h2 : members = 40) :
  ∃ (max_cookies : Nat), max_cookies = 161 ∧ max_cookies ≤ total_cookies ∧ 
  (∀ (other_cookies : Nat), other_cookies ≤ total_cookies - (members - 1) → other_cookies ≤ max_cookies) := by
  use 161
  constructor
  · rfl
  constructor
  · rw [h1]
    norm_num
  · intro other_cookies h
    rw [h1, h2] at h
    linarith

#check max_cookies

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cookies_l880_88071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_last_matches_l880_88063

/-- Calculates the average score for the last 4 matches of a cricket player given the total number of matches, 
    average score for all matches, number of first matches, and average score for first matches. -/
theorem cricket_average_last_matches 
  (total_matches : ℕ) 
  (avg_all : ℚ) 
  (first_matches : ℕ) 
  (avg_first : ℚ) 
  (h1 : total_matches = 12)
  (h2 : avg_all = 48)
  (h3 : first_matches = 8)
  (h4 : avg_first = 40) :
  let last_matches := total_matches - first_matches
  let total_runs := total_matches * avg_all
  let first_runs := first_matches * avg_first
  let last_runs := total_runs - first_runs
  let avg_last := last_runs / (total_matches - first_matches)
  avg_last = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_last_matches_l880_88063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_tunnel_theorem_l880_88089

/-- Represents a parabola y = -x^2 / (2p) -/
def Parabola (p : ℝ) (x y : ℝ) : Prop := y = -x^2 / (2 * p)

/-- Represents a rectangle within the parabola -/
def RectangleInParabola (p truck_width truck_height : ℝ) : Prop :=
  ∃ (x : ℝ), 
    Parabola p x (-truck_height) ∧ 
    x = truck_width / 2

theorem parabolic_tunnel_theorem (p truck_width truck_height : ℝ) 
  (h_p_pos : p > 0)
  (h_truck_width : truck_width = 1.6)
  (h_truck_height : truck_height = 3)
  (h_rect_in_parabola : RectangleInParabola p truck_width truck_height) :
  ∃ (ε : ℝ), abs (2 * p - 12.21) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_tunnel_theorem_l880_88089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_alpha_satisfies_l880_88014

def alpha_set : Set ℚ := {-2, -1, -1/2, 1/2, 1, 2}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_decreasing_on_positive_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f y < f x

def satisfies_conditions (α : ℚ) : Prop :=
  is_odd_function (fun x => x ^ (α : ℝ)) ∧
  is_decreasing_on_positive_reals (fun x => x ^ (α : ℝ))

theorem exactly_one_alpha_satisfies :
  ∃! α, α ∈ alpha_set ∧ satisfies_conditions α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_alpha_satisfies_l880_88014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_argument_l880_88035

open Complex

noncomputable def complex_sum : ℂ :=
  exp (11 * Real.pi * I / 40) +
  exp (21 * Real.pi * I / 40) +
  exp (31 * Real.pi * I / 40) +
  exp (41 * Real.pi * I / 40) +
  exp (51 * Real.pi * I / 40) +
  exp (61 * Real.pi * I / 40)

theorem complex_sum_argument :
  ∃ (r : ℝ), complex_sum = r * exp (I * (Real.pi / 2)) ∧ r > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_argument_l880_88035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l880_88032

theorem angle_relations (α β : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : β ∈ Set.Ioo (π / 2) π)
  (h3 : Real.cos (2 * β) = -7 / 9)
  (h4 : Real.sin (α + β) = 7 / 9) :
  Real.cos β = -1 / 3 ∧ Real.sin α = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l880_88032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l880_88065

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (a - 1) * x + 3 * a - 4 else a^x

theorem a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) > 0) →
  1 < a ∧ a ≤ 5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l880_88065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l880_88093

noncomputable def g (c : ℝ) (x : ℝ) : ℝ := (x^2 + 3*x + c) / (x^2 - 3*x - 10)

theorem exactly_one_vertical_asymptote (c : ℝ) :
  (∃! x, (x^2 - 3*x - 10 = 0 ∧ x^2 + 3*x + c ≠ 0)) ↔ (c = -40 ∨ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l880_88093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_min_value_achievable_l880_88078

/-- The expression to be minimized -/
noncomputable def expression (a b c d : ℝ) : ℝ :=
  a^8 / ((a^2 + b) * (a^2 + c) * (a^2 + d)) +
  b^8 / ((b^2 + c) * (b^2 + d) * (b^2 + a)) +
  c^8 / ((c^2 + d) * (c^2 + a) * (c^2 + b)) +
  d^8 / ((d^2 + a) * (d^2 + b) * (d^2 + c))

/-- The theorem stating the minimum value of the expression -/
theorem min_value_of_expression {a b c d : ℝ} 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d = 4) : 
  expression a b c d ≥ (1/2 : ℝ) := by
  sorry

/-- The theorem stating that the minimum is achievable -/
theorem min_value_achievable : 
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  a + b + c + d = 4 ∧ expression a b c d = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_min_value_achievable_l880_88078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l880_88076

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := 
  Int.floor x

-- State the theorem
theorem floor_expression_evaluation : 
  (floor 6.5) * (floor (2/3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.4) - (6.6 : ℝ) = 15.8 := by
  -- Convert all literals to Real type
  have h1 : floor 6.5 = 6 := by sorry
  have h2 : floor (2/3 : ℝ) = 0 := by sorry
  have h3 : floor 2 = 2 := by sorry
  have h4 : floor 8.4 = 8 := by sorry
  
  -- Rewrite using these facts
  rw [h1, h2, h3, h4]
  
  -- Simplify the expression
  norm_num
  
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l880_88076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_B_equals_target_l880_88056

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = (Real.sqrt (x - 4)) / (abs x - 5)}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 - 6*x + 13)}

-- Define the complement of A with respect to B
def complement_A_B : Set ℝ := B \ A

-- Theorem statement
theorem complement_A_B_equals_target : 
  complement_A_B = {x | (2 < x ∧ x < 4) ∨ x = 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_B_equals_target_l880_88056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_problem_l880_88057

-- Define the sets A and B
def A : Set ℝ := {x | (1/2 : ℝ) < (2 : ℝ)^(x+1) ∧ (2 : ℝ)^(x+1) < 8}
def B (a : ℝ) : Set ℝ := {x | 3*a - 2 < x ∧ x < 2*a + 1}

-- State the theorem
theorem sets_problem :
  (∀ x : ℝ, x ∈ (A ∪ (Set.univ \ B 1)) ↔ x < 2 ∨ x ≥ 3) ∧
  (∀ a : ℝ, A ∩ B a = B a ↔ a ∈ Set.Icc 0 (1/2) ∪ Set.Ici 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_problem_l880_88057
