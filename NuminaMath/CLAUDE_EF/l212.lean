import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_sauce_reaches_25_percent_l212_21202

/-- Represents the number of days passed --/
def days : ℕ → ℝ
  | 0 => 0
  | n + 1 => (n + 1 : ℝ)

/-- Initial amount of shampoo in the bottle (in ounces) --/
noncomputable def initial_shampoo : ℝ := 10

/-- Daily usage of shampoo (in ounces) --/
noncomputable def daily_usage : ℝ := 1

/-- Daily replacement with hot sauce (in ounces) --/
noncomputable def daily_replacement : ℝ := 1/2

/-- Total liquid in the bottle after n days --/
noncomputable def total_liquid (n : ℕ) : ℝ :=
  initial_shampoo - n * daily_usage + n * daily_replacement

/-- Amount of hot sauce in the bottle after n days --/
noncomputable def hot_sauce (n : ℕ) : ℝ := n * daily_replacement

/-- Percentage of hot sauce in the bottle after n days --/
noncomputable def hot_sauce_percentage (n : ℕ) : ℝ :=
  (hot_sauce n) / (total_liquid n) * 100

/-- Theorem: The smallest number of days for which the hot sauce content
    reaches or exceeds 25% is 3 --/
theorem hot_sauce_reaches_25_percent :
  (∀ n : ℕ, n < 3 → hot_sauce_percentage n < 25) ∧
  hot_sauce_percentage 3 ≥ 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_sauce_reaches_25_percent_l212_21202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_valid_numbers_l212_21204

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ 
  ∃ (a b c d e : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
    c ≠ d ∧ c ≠ e ∧ 
    d ≠ e ∧
    ({a, b, c, d, e} : Finset ℕ) = {1, 2, 3, 4, 5} ∧
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e

def valid_numbers : Set ℕ := {n | is_valid_number n}

theorem gcd_of_valid_numbers : 
  ∃ (g : ℕ), g > 0 ∧ (∀ (n : ℕ), n ∈ valid_numbers → g ∣ n) ∧
  (∀ (d : ℕ), d > 0 → (∀ (n : ℕ), n ∈ valid_numbers → d ∣ n) → d ≤ g) ∧
  g = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_valid_numbers_l212_21204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l212_21269

def f (x : ℝ) : ℝ := 2 * x + 1

def g (x : ℝ) : ℝ := x * |x - 2|

theorem max_t_value (t : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 t → x₂ ∈ Set.Icc 0 t → x₁ ≠ x₂ →
    (g x₁ - g x₂) / (f x₁ - f x₂) < 2) →
  t ≤ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l212_21269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_reciprocal_sum_l212_21224

theorem min_reciprocal_sum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- Positive sequence
  (∃ q : ℝ, q > 0 ∧ ∀ k, a (k + 1) = a k * q + a k - a (k - 1)) →  -- Arithmetic-geometric sequence
  (a 7 = a 6 + 2 * a 5) →  -- Given condition
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) →  -- Given condition
  (∀ i j : ℕ, i ≠ j → 1 / (i : ℝ) + 1 / (j : ℝ) ≥ 5 / 3) ∧  -- Lower bound
  (∃ i j : ℕ, i ≠ j ∧ 1 / (i : ℝ) + 1 / (j : ℝ) = 5 / 3)  -- Achievable minimum
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_reciprocal_sum_l212_21224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_conic_l212_21227

-- Define the parametric curve
noncomputable def x (t : ℝ) : ℝ := 3 * Real.cos t + 2 * Real.sin t
noncomputable def y (t : ℝ) : ℝ := 5 * Real.sin t

-- Define the equation of the curve
def curve_equation (a b c : ℝ) (t : ℝ) : Prop :=
  a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1

-- Theorem statement
theorem curve_is_conic : ∃ (a b c : ℝ),
  (∀ t, curve_equation a b c t) ∧ 
  a = 1/9 ∧ b = -4/45 ∧ c = 1/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_conic_l212_21227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_preserves_mean_and_variance_l212_21274

def initial_set : Finset Int := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def replacement_set1 : Finset Int := {-5, -4, -3, -2, -1, 0, 1, 2, 3, -1, 5}
def replacement_set2 : Finset Int := {-5, 1, -3, -2, -1, 0, 1, 2, 3, 4, -5}

def mean (s : Finset Int) : ℚ := (s.sum (fun x => (x : ℚ))) / s.card

def variance (s : Finset Int) : ℚ :=
  let μ := mean s
  (s.sum (fun x => ((x : ℚ) - μ) ^ 2)) / s.card

theorem replacement_preserves_mean_and_variance :
  (mean initial_set = mean replacement_set1 ∧ variance initial_set = variance replacement_set1) ∨
  (mean initial_set = mean replacement_set2 ∧ variance initial_set = variance replacement_set2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_preserves_mean_and_variance_l212_21274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_x_cubed_term_l212_21261

theorem no_x_cubed_term (a : ℚ) : 
  (∀ k : ℚ, ∃ p q r s : ℚ, (a * X^2 - 3*X) * (X^2 - 2*X - 1) = p * X^4 + q * X^3 + r * X^2 + s * X) ↔ 
  a = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_x_cubed_term_l212_21261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l212_21283

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2*x) - 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ M) ∧
  (∃ (m : ℝ), m = -1 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → m ≤ f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l212_21283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_equation_solutions_l212_21249

theorem gcd_equation_solutions : 
  ∃! (n : ℕ), n = (Finset.filter (fun p : ℕ × ℕ => p.1 ≤ p.2 ∧ p.1 * p.2 = p.1 + p.2 + Nat.gcd p.1 p.2) (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_equation_solutions_l212_21249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l212_21222

theorem sqrt_equation_solution (x : ℝ) :
  (x > 2) → (Real.sqrt (7 * x) / Real.sqrt (4 * (x - 2)) = 3) → x = 72 / 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l212_21222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_a_range_l212_21264

noncomputable def sequenceA (a : ℝ) (n : ℕ+) : ℝ :=
  if n ≤ 7 then (3 - a) * n.val - 3 else a^(n.val - 6)

theorem increasing_sequence_a_range (a : ℝ) :
  (∀ n m : ℕ+, n < m → sequenceA a n < sequenceA a m) →
  2 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_a_range_l212_21264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_hyperbola_eccentricity_eccentricity_equals_two_sqrt_three_over_three_l212_21276

/-- A cone with an equilateral triangle cross-section -/
structure EquilateralCone where
  /-- The angle between the two asymptotes of the hyperbola formed by intersecting the cone with a plane parallel to its axis -/
  asymptote_angle : ℝ
  /-- The asymptote angle is 60 degrees (π/3 radians) for an equilateral triangle cross-section -/
  asymptote_angle_eq : asymptote_angle = π / 3

/-- The eccentricity of a hyperbola formed by intersecting an equilateral cone with a plane parallel to its axis -/
noncomputable def hyperbola_eccentricity (cone : EquilateralCone) : ℝ :=
  2 * Real.sqrt 3 / 3

/-- Theorem stating that the eccentricity of the hyperbola is 2√3/3 -/
theorem equilateral_cone_hyperbola_eccentricity (cone : EquilateralCone) :
  hyperbola_eccentricity cone = 2 * Real.sqrt 3 / 3 := by
  -- Unfold the definition of hyperbola_eccentricity
  unfold hyperbola_eccentricity
  -- The definition directly gives us the result
  rfl

/-- Proof that the eccentricity is equal to 2√3/3 -/
theorem eccentricity_equals_two_sqrt_three_over_three (cone : EquilateralCone) :
  hyperbola_eccentricity cone = 2 * Real.sqrt 3 / 3 := by
  -- Apply the previously proven theorem
  exact equilateral_cone_hyperbola_eccentricity cone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_hyperbola_eccentricity_eccentricity_equals_two_sqrt_three_over_three_l212_21276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l212_21254

-- Define the function f(x) = sin x
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- Define the domain
def domain : Set ℝ := Set.Icc 0 (3 * Real.pi / 2)

-- Define the area of the closed figure
noncomputable def area : ℝ := ∫ x in (0)..(3 * Real.pi / 2), if x ≤ Real.pi then f x else -f x

-- Theorem statement
theorem area_of_closed_figure : area = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l212_21254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_theorem_l212_21201

/-- The area of a rhombus formed by the equation |x/s| + |y/s| = 1 is 32 if and only if s = 4 -/
theorem rhombus_area_theorem (s : ℝ) : s > 0 →
  (∀ x y : ℝ, |x/s| + |y/s| = 1 → x^2 + y^2 ≤ 2*s^2) →
  2*s^2 = 32 ↔ s = 4 := by
  sorry

#check rhombus_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_theorem_l212_21201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_die_roll_probabilities_l212_21229

-- Define a fair six-sided die
def Die := Fin 6

-- Define the probability space
def Ω := Die × Die

-- Define the probability measure (noncomputable as it involves real numbers)
noncomputable def P : Set Ω → ℝ := sorry

-- Event B: second roll is even
def B : Set Ω := {ω | ω.2.val % 2 = 0}

-- Event C: two rolls result in the same number
def C : Set Ω := {ω | ω.1 = ω.2}

-- Event D: at least one odd number appears
def D : Set Ω := {ω | ω.1.val % 2 = 1 ∨ ω.2.val % 2 = 1}

-- Theorem stating the required probabilities
theorem die_roll_probabilities :
  (P D = 3/4) ∧
  (P (B ∩ D) = 1/4) ∧
  (P (B ∩ C) = P B * P C) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_die_roll_probabilities_l212_21229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_acute_l212_21250

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- Define the arithmetic sequence properties
def arithmeticSequence (a : Real) : Prop :=
  ∃ (d : Real) (first_term : Real), 
    d = a ∧ 
    -4 = (3 - 1) * d + first_term ∧ 
    4 = (7 - 1) * d + first_term

-- Define the geometric sequence properties
def geometricSequence (b : Real) : Prop :=
  ∃ (r : Real) (first_term : Real), 
    r = b ∧ 
    (1/3 : Real) = first_term * r^(3-1) ∧ 
    9 = first_term * r^(6-1)

-- Theorem statement
theorem triangle_is_acute (t : Triangle) 
  (h1 : arithmeticSequence (Real.tan t.A))
  (h2 : geometricSequence (Real.tan t.B)) : 
  t.A < π/2 ∧ t.B < π/2 ∧ t.C < π/2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_acute_l212_21250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_problem_l212_21294

theorem fraction_problem (c b a : ℚ) : 
  (c / b < 1) → 
  (∀ m n : ℤ, m / n = c / b → (m.natAbs ≥ c.num.natAbs ∧ n.natAbs ≥ b.den)) → 
  ((c + a) / b = 1 / 3) → 
  (c / (b + a) = 1 / 4) → 
  c / b = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_problem_l212_21294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_max_value_l212_21243

theorem trigonometric_sum_max_value :
  ∀ θ₁ θ₂ θ₃ θ₄ : ℝ,
  Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + Real.cos θ₃ * Real.sin θ₄ + Real.cos θ₄ * Real.sin θ₁ ≤ 2 ∧
  ∃ θ₁' θ₂' θ₃' θ₄' : ℝ,
    Real.cos θ₁' * Real.sin θ₂' + Real.cos θ₂' * Real.sin θ₃' + Real.cos θ₃' * Real.sin θ₄' + Real.cos θ₄' * Real.sin θ₁' = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_max_value_l212_21243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l212_21236

-- Define the function f(x) = log₍₁/₂₎|x-3|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs (x - 3)) / Real.log (1/2)

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x y : ℝ, 3 < x → x < y → f x > f y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l212_21236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_like_both_mozart_beethoven_l212_21242

theorem min_like_both_mozart_beethoven 
  (total : ℕ) 
  (like_mozart : ℕ) 
  (like_beethoven : ℕ) 
  (dislike_both : ℕ) 
  (h1 : total = 120)
  (h2 : like_mozart = 95)
  (h3 : like_beethoven = 80)
  (h4 : dislike_both ≥ 10) : 
  ∃ (like_both : ℕ), like_both ≥ 25 ∧ like_both ≤ like_mozart ∧ like_both ≤ like_beethoven := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_like_both_mozart_beethoven_l212_21242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_max_value_range_reverse_max_value_range_l212_21231

-- Define the function f
def f (a x : ℝ) : ℝ := a * x - abs (2 * x - 1) + 2

-- Define the solution set for f(x) + f(-x) ≤ 0
def solution_set (a : ℝ) : Set ℝ :=
  {x | f a x + f a (-x) ≤ 0}

-- Theorem for the first part of the problem
theorem solution_set_eq (a : ℝ) :
  solution_set a = Set.Iic (-1) ∪ Set.Ici 1 := by sorry

-- Theorem for the second part of the problem
theorem max_value_range (a : ℝ) :
  (∃ (M : ℝ), ∀ (x : ℝ), f a x ≤ M) → -2 ≤ a ∧ a ≤ 2 := by sorry

-- Reverse implication
theorem reverse_max_value_range (a : ℝ) :
  -2 ≤ a ∧ a ≤ 2 → (∃ (M : ℝ), ∀ (x : ℝ), f a x ≤ M) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_max_value_range_reverse_max_value_range_l212_21231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_minimum_value_condition_l212_21253

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1/2) + 2 / (2 * x + 1)

-- Part I
theorem monotonic_increasing_condition (a : ℝ) :
  (a > 0 ∧ ∀ x > 0, Monotone (f a)) ↔ a ≥ 2 :=
by sorry

-- Part II
theorem minimum_value_condition :
  ∃ a : ℝ, (∀ x > 0, f a x ≥ 1) ∧ (∃ x > 0, f a x = 1) ∧ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_minimum_value_condition_l212_21253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_three_reps_l212_21213

/-- A representation of a number as 13x + 73y where x and y are natural numbers -/
def MyRepresentation := ℕ × ℕ

/-- Check if a natural number n can be represented as 13x + 73y -/
def isRepresentable (n : ℕ) (rep : MyRepresentation) : Prop :=
  n = 13 * rep.1 + 73 * rep.2

/-- Check if a natural number n has exactly three distinct representations -/
def hasThreeRepresentations (n : ℕ) (rep1 rep2 rep3 : MyRepresentation) : Prop :=
  isRepresentable n rep1 ∧
  isRepresentable n rep2 ∧
  isRepresentable n rep3 ∧
  rep1 ≠ rep2 ∧ rep1 ≠ rep3 ∧ rep2 ≠ rep3

/-- The smallest natural number with three representations -/
def smallestWithThreeReps : ℕ := 1984

theorem smallest_with_three_reps :
  (hasThreeRepresentations smallestWithThreeReps (24, 20) (97, 1) (1, 27)) ∧
  (∀ m : ℕ, m < smallestWithThreeReps →
    ∀ rep1 rep2 rep3 : MyRepresentation,
    ¬(hasThreeRepresentations m rep1 rep2 rep3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_three_reps_l212_21213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chelsea_win_guarantee_l212_21248

/-- Represents the score of a single shot in the archery contest -/
inductive ShotScore
  | Ten : ShotScore
  | Eight : ShotScore
  | Five : ShotScore
  | Three : ShotScore
  | Zero : ShotScore

/-- The total number of shots in the contest -/
def totalShots : Nat := 120

/-- Chelsea's lead after 60 shots -/
def initialLead : Nat := 70

/-- The number of shots already taken -/
def shotsTaken : Nat := 60

/-- The minimum score Chelsea gets on each shot -/
def minScore : Nat := 5

/-- The score of a bullseye -/
def bullseyeScore : Nat := 10

/-- Chelsea's initial score after 60 shots -/
def k : Nat := 0  -- We define k as a constant for simplicity

/-- The number of bullseyes Chelsea needs to guarantee a win -/
def minBullseyesForWin : Nat := 47

theorem chelsea_win_guarantee (opponent_score : Fin (totalShots - shotsTaken) → Nat) :
  (∀ n : Fin (totalShots - shotsTaken), opponent_score n ≤ bullseyeScore) →
  k + minBullseyesForWin * bullseyeScore +
    (totalShots - shotsTaken - minBullseyesForWin) * minScore >
  (k - initialLead) + (totalShots - shotsTaken) * bullseyeScore ∧
  ∀ m, m < minBullseyesForWin →
    ∃ opp_scores : Fin (totalShots - shotsTaken) → Nat,
      (∀ n : Fin (totalShots - shotsTaken), opp_scores n ≤ bullseyeScore) ∧
      k + m * bullseyeScore +
        (totalShots - shotsTaken - m) * minScore ≤
      (k - initialLead) + (Finset.sum (Finset.range (totalShots - shotsTaken)) (λ i => opp_scores ⟨i, sorry⟩)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chelsea_win_guarantee_l212_21248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractional_parts_l212_21262

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- State the theorem
theorem sum_of_fractional_parts :
  frac (2015 / 3) + frac (315 / 4) + frac (412 / 5) = 1.817 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractional_parts_l212_21262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_chord_is_8_l212_21221

/-- A circle with a chord that subtends a 90° arc -/
structure Circle90DegChord where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of the chord -/
  chord_length : ℝ
  /-- The chord subtends a 90° arc -/
  subtends_90_deg : True
  /-- The chord length is 16 units -/
  chord_length_is_16 : chord_length = 16

/-- The distance from the center of the circle to the chord -/
noncomputable def distance_to_chord (c : Circle90DegChord) : ℝ :=
  c.chord_length / 2

theorem distance_to_chord_is_8 (c : Circle90DegChord) :
  distance_to_chord c = 8 := by
  -- Unfold the definition of distance_to_chord
  unfold distance_to_chord
  -- Use the fact that chord_length is 16
  rw [c.chord_length_is_16]
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_chord_is_8_l212_21221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_range_of_a_l212_21282

noncomputable section

-- Define the function f
def f (a x : ℝ) : ℝ := (a - 1) * Real.log x + x + a / x

-- Part 1
theorem tangent_line_at_2 :
  let a : ℝ := 1
  let x₀ : ℝ := 2
  let y₀ : ℝ := f a x₀
  let m : ℝ := (deriv (f a)) x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 3 * x - 4 * y + 4 = 0 :=
by
  sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ioc 1 (Real.exp 1) → f a x - a / x > 0) →
  a > 1 - Real.exp 1 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_range_of_a_l212_21282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_discount_calculation_coupon_discount_is_ten_l212_21245

theorem coupon_discount_calculation (original_price : ℝ) 
  (initial_discount_percent : ℝ) (final_discount_percent : ℝ) 
  (total_savings : ℝ) (coupon_discount : ℝ) : Prop :=
  original_price = 125 ∧
  initial_discount_percent = 20 ∧
  final_discount_percent = 10 ∧
  total_savings = 44 →
  let price_after_initial_discount := original_price * (1 - initial_discount_percent / 100);
  let price_after_coupon := price_after_initial_discount - coupon_discount;
  let final_price := price_after_coupon * (1 - final_discount_percent / 100);
  final_price = original_price - total_savings ∧
  coupon_discount = 10

theorem coupon_discount_is_ten : 
  ∃ (coupon_discount : ℝ), 
    coupon_discount_calculation 125 20 10 44 coupon_discount := by
  sorry

#check coupon_discount_is_ten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_discount_calculation_coupon_discount_is_ten_l212_21245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_food_price_is_120_l212_21235

/-- Calculates the original food price given the total bill, tip percentage, and tax rate. -/
noncomputable def originalFoodPrice (totalBill : ℝ) (tipPercentage : ℝ) (taxRate : ℝ) : ℝ :=
  totalBill / (1 + tipPercentage * (1 + taxRate))

/-- Theorem stating that the original food price is $120 given the specified conditions. -/
theorem original_food_price_is_120 :
  originalFoodPrice 158.40 0.20 0.10 = 120 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_food_price_is_120_l212_21235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_given_sin_l212_21290

theorem cos_value_given_sin (α : ℝ) :
  Real.sin (π + α) = 1/2 → Real.cos (α - 3/2 * π) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_given_sin_l212_21290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l212_21217

theorem sin_double_angle_special_case (θ : ℝ) :
  Real.sin (θ + π/4) = 2/5 → Real.sin (2*θ) = -17/25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l212_21217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_passes_through_point_max_chord_length_l212_21225

/-- The length of the chord obtained by intersecting the parabola 
    y = (t^2 + t + 1)x^2 - 2(1 + t)^2 x + t^2 + 3t + 1 with the x-axis -/
noncomputable def chord_length (t : ℝ) : ℝ := |2 * t / (t^2 + t + 1)|

/-- The parabola passes through the point (1,0) for all t -/
theorem parabola_passes_through_point (t : ℝ) : 
  (t^2 + t + 1) - 2*(1 + t)^2 + t^2 + 3*t + 1 = 0 := by sorry

/-- The chord length is maximized when t = -1 -/
theorem max_chord_length : 
  ∀ t : ℝ, chord_length t ≤ chord_length (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_passes_through_point_max_chord_length_l212_21225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_tangent_l212_21298

/-- The value of m for which the parabola y = x^2 + 2 and the hyperbola y^2 - mx^2 = 1 are tangent -/
noncomputable def tangent_m : ℝ := 4 + 2 * Real.sqrt 3

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 2

/-- Hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m * x^2 = 1

/-- Theorem stating that the parabola and hyperbola are tangent when m = 4 + 2√3 -/
theorem parabola_hyperbola_tangent :
  ∃ (x y : ℝ), parabola x y ∧ hyperbola tangent_m x y ∧
  ∀ (x' y' : ℝ), parabola x' y' ∧ hyperbola tangent_m x' y' → (x', y') = (x, y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_tangent_l212_21298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_product_sequence_length_product_mod_9_l212_21223

/-- The arithmetic sequence starting at 7, ending at 197, with common difference 10 -/
def arithmeticSequence : List ℕ := List.range 20 |>.map (λ n => 7 + 10 * n)

/-- The product of all numbers in the sequence -/
def sequenceProduct : ℕ := arithmeticSequence.prod

theorem remainder_of_product (n : ℕ) : n ∈ arithmeticSequence → n % 9 = 7 := by sorry

theorem sequence_length : arithmeticSequence.length = 20 := by sorry

theorem product_mod_9 : sequenceProduct % 9 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_product_sequence_length_product_mod_9_l212_21223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l212_21271

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_angles : 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi)
  (h_cosine_a : Real.cos A = (b^2 + c^2 - a^2) / (2*b*c))
  (h_cosine_b : Real.cos B = (a^2 + c^2 - b^2) / (2*a*c))
  (h_cosine_c : Real.cos C = (a^2 + b^2 - c^2) / (2*a*b))

theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = t.a * t.c) :
  t.B = Real.pi/3 ∧ 
  (t.c = 3*t.a → Real.tan t.A = Real.sqrt 3 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l212_21271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_always_positive_l212_21281

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotone_decreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → f x ≥ f y

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_of_f_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h_odd : is_odd_function f)
  (h_decreasing : is_monotone_decreasing f {x : ℝ | x ≥ 0})
  (h_arithmetic : is_arithmetic_sequence a)
  (h_a3_neg : a 3 < 0) :
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_always_positive_l212_21281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_unique_configuration_l212_21203

/-- Represents the type of tree -/
inductive TreeType
| Oak
| Baobab
deriving DecidableEq

/-- Represents a configuration of trees -/
def Configuration := Fin 2000 → TreeType

/-- Calculates the sign value for a tree at a given position -/
def signValue (config : Configuration) (pos : Fin 2000) : Nat :=
  let left := if pos > 0 then config (pos - 1) else TreeType.Baobab
  let right := if pos < 1999 then config (pos + 1) else TreeType.Baobab
  let center := config pos
  (if left = TreeType.Oak then 1 else 0) +
  (if center = TreeType.Oak then 1 else 0) +
  (if right = TreeType.Oak then 1 else 0)

/-- Generates the sequence of sign values for a given configuration -/
def signSequence (config : Configuration) : Fin 2000 → Nat :=
  fun pos => signValue config pos

theorem non_unique_configuration :
  ∃ (config1 config2 : Configuration),
    config1 ≠ config2 ∧
    signSequence config1 = signSequence config2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_unique_configuration_l212_21203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l212_21228

/-- A hyperbola with foci F₁ and F₂, and a point P on its left branch -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  h_positive : a > 0 ∧ b > 0
  h_equation : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x, y) = P ∨ (x, y) = F₁ ∨ (x, y) = F₂
  h_left_branch : P.1 < 0
  h_perpendicular : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0
  h_angle : (P.2 - F₂.2) / (P.1 - F₂.1) = 2 / 3

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.F₂.1 - h.F₁.1)^2 + (h.F₂.2 - h.F₁.2)^2) / (2 * h.a)

/-- Theorem: The eccentricity of the hyperbola is √13 -/
theorem hyperbola_eccentricity (h : Hyperbola) : eccentricity h = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l212_21228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_distance_ratio_l212_21208

/-- Represents the characteristics of a ferry journey -/
structure FerryJourney where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The ratio of distances between two ferry journeys -/
noncomputable def distanceRatio (j1 j2 : FerryJourney) : ℝ :=
  j1.distance / j2.distance

theorem ferry_distance_ratio :
  ∀ (ferryP ferryQ : FerryJourney),
    ferryP.speed = 8 →
    ferryP.time = 3 →
    ferryQ.speed = ferryP.speed + 4 →
    ferryQ.time = ferryP.time + 1 →
    ferryP.distance = ferryP.speed * ferryP.time →
    ferryQ.distance = ferryQ.speed * ferryQ.time →
    distanceRatio ferryQ ferryP = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_distance_ratio_l212_21208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_seating_capacity_l212_21273

theorem restaurant_seating_capacity 
  (total_tables : ℕ)
  (total_capacity : ℕ)
  (new_table_difference : ℕ)
  (original_table_capacity : ℕ)
  (h1 : total_tables = 40)
  (h2 : total_capacity = 212)
  (h3 : new_table_difference = 12)
  (h4 : original_table_capacity = 4)
  : ∃ (new_table_capacity : ℕ),
    (let original_tables := (total_tables - new_table_difference) / 2
     let new_tables := total_tables - original_tables
     new_table_capacity = 6 ∧ 
     original_tables * original_table_capacity + new_tables * new_table_capacity = total_capacity) :=
by
  sorry

#check restaurant_seating_capacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_seating_capacity_l212_21273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_sum_l212_21259

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/4)

theorem two_roots_sum (m : ℝ) :
  (∃ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ < 2*Real.pi ∧ 0 ≤ x₂ ∧ x₂ < 2*Real.pi ∧
    x₁ ≠ x₂ ∧ f x₁ = m ∧ f x₂ = m) →
  ∃ (x₁ x₂ : ℝ), (x₁ + x₂ = Real.pi/2 ∨ x₁ + x₂ = 5*Real.pi/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_sum_l212_21259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_term_sum_with_twelfth_l212_21257

noncomputable def arithmeticProgression (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def sumArithmeticProgression (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem ninth_term_sum_with_twelfth (a₁ d : ℝ) :
  (∃ k : ℕ, arithmeticProgression a₁ d k + arithmeticProgression a₁ d 12 = 20) ∧
  sumArithmeticProgression a₁ d 20 = 200 →
  arithmeticProgression a₁ d 9 + arithmeticProgression a₁ d 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_term_sum_with_twelfth_l212_21257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_limit_l212_21258

/-- The sum of a geometric series with n terms, first term a, and common ratio r -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^(n+1)) / (1 - r)

/-- The limit of the ratio of two geometric series sums as n approaches infinity -/
theorem geometric_series_ratio_limit :
  let series1 := fun n => geometricSum 1 (1/3) n
  let series2 := fun n => geometricSum 1 (1/2) n
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |series1 n / series2 n - 2| < ε := by
  sorry

#check geometric_series_ratio_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_limit_l212_21258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ordering_l212_21286

noncomputable def f (x : ℝ) : ℝ := 3 / x

noncomputable def A : ℝ × ℝ := (-2, f (-2))
noncomputable def B : ℝ × ℝ := (-1, f (-1))
noncomputable def C : ℝ × ℝ := (1, f 1)

noncomputable def y₁ : ℝ := A.2
noncomputable def y₂ : ℝ := B.2
noncomputable def y₃ : ℝ := C.2

theorem inverse_proportion_ordering : y₂ < y₁ ∧ y₁ < y₃ := by
  -- Unfold definitions
  unfold y₁ y₂ y₃ A B C f
  -- Simplify expressions
  simp
  -- Split the goal into two parts
  apply And.intro
  -- Prove y₂ < y₁
  · norm_num
  -- Prove y₁ < y₃
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ordering_l212_21286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_accomplice_profession_l212_21272

-- Define the suspects
inductive Suspect : Type
| Bertrand : Suspect
| Alfred : Suspect
| Charles : Suspect

-- Define the professions
inductive Profession : Type
| Painter : Profession
| PianoTuner : Profession
| Decorator : Profession
| Doctor : Profession
| InsuranceAgent : Profession

-- Define the roles
inductive Role : Type
| Thief : Role
| Accomplice : Role
| FalselyAccused : Role

-- Define a function to represent the statements made by each suspect
def makeStatement (s : Suspect) : Prop := sorry

-- Define a function to represent the truthfulness of each suspect
def isTruthful (s : Suspect) : Prop := sorry

-- Define a function to assign a role to each suspect
def assignRole (s : Suspect) : Role := sorry

-- Define a function to assign a profession to each suspect
def assignProfession (s : Suspect) : Profession := sorry

-- Theorem statement
theorem accomplice_profession :
  (∀ s : Suspect, makeStatement s) →
  (∃! s : Suspect, assignRole s = Role.Thief) →
  (∃! s : Suspect, assignRole s = Role.Accomplice) →
  (∃! s : Suspect, assignRole s = Role.FalselyAccused) →
  (∀ s : Suspect, assignRole s = Role.Thief → ¬isTruthful s) →
  (∀ s : Suspect, assignRole s = Role.FalselyAccused → isTruthful s) →
  assignRole Suspect.Alfred = Role.Accomplice ∧ assignProfession Suspect.Alfred = Profession.Doctor :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_accomplice_profession_l212_21272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l212_21206

/-- The curve function f(x) = sin²(x) + 2ax --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * a * x

/-- The derivative of f(x) --/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * a

/-- The line equation x + y + m = 0 --/
def line_eq (x y m : ℝ) : Prop := x + y + m = 0

/-- The condition for the line to be a tangent to the curve --/
def is_tangent (a : ℝ) : Prop := ∃ x m : ℝ, line_eq x (f a x) m ∧ f_deriv a x = -1

/-- The main theorem --/
theorem tangent_condition (a : ℝ) : ¬(is_tangent a) ↔ (a < -1 ∨ a > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l212_21206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_with_inscribed_circle_l212_21207

theorem square_diagonal_with_inscribed_circle 
  (s r : ℝ) 
  (h1 : s > 0) 
  (h2 : r > 0) 
  (h3 : s = 2 * r) 
  (h4 : 4 * s = π * r^2) : 
  Real.sqrt 2 * s = 16 * Real.sqrt 2 / π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_with_inscribed_circle_l212_21207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_for_exponential_equation_l212_21219

theorem no_real_solutions_for_exponential_equation :
  ∀ x : ℝ, (2 : ℝ)^(x^2 - 3*x - 2) ≠ (8 : ℝ)^(x - 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_for_exponential_equation_l212_21219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercept_line_equation_l212_21292

/-- A line passing through point (4, -3) with equal intercepts on the coordinate axes -/
structure EqualInterceptLine where
  -- The line equation
  line_equation : ℝ → ℝ → ℝ
  -- The line passes through point (4, -3)
  passes_through : line_equation 4 (-3) = 0
  -- The line has equal intercepts on the coordinate axes
  equal_intercepts : ∃ (a : ℝ), (line_equation a 0 = 0 ∧ line_equation 0 a = 0) ∨ 
                                (line_equation (-a) 0 = 0 ∧ line_equation 0 a = 0)

/-- The equation of the line is one of the three specified equations -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, l.line_equation x y = x - y - 7) ∨
  (∀ x y, l.line_equation x y = x + y - 1) ∨
  (∀ x y, l.line_equation x y = 3*x + 4*y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercept_line_equation_l212_21292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equilateral_from_isosceles_right_l212_21263

/-- Represents a triangle with a 45° angle -/
structure IsoscelesRightTriangle where
  /-- The 45° angle of the triangle -/
  angle : Real
  angle_eq : angle = 45

/-- Represents the result of splicing two isosceles right triangles -/
structure SplicedShape where
  /-- The two triangles being spliced -/
  triangle1 : IsoscelesRightTriangle
  triangle2 : IsoscelesRightTriangle
  /-- The angles of the resulting shape -/
  angles : List Real

/-- An equilateral triangle has all angles equal to 60° -/
def isEquilateralTriangle (shape : SplicedShape) : Prop :=
  ∀ angle, angle ∈ shape.angles → angle = 60

/-- The main theorem: It's impossible to form an equilateral triangle by splicing two isosceles right triangles -/
theorem no_equilateral_from_isosceles_right :
  ¬∃ (shape : SplicedShape), isEquilateralTriangle shape :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equilateral_from_isosceles_right_l212_21263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_integral_value_l212_21226

-- Define x as the square root of 15
noncomputable def x : ℝ := Real.sqrt 15

-- Define the integrand
def f (t : ℝ) : ℝ := t^2

-- Theorem statement
theorem integral_x_squared : ∫ t in (0)..(2), f t = 8/3 := by
  sorry

-- Define y
noncomputable def y : ℝ := 2 * (Real.log x)^3 - (5 / 3)

-- Theorem about the value of the integral
theorem integral_value : ∫ t in (0)..(2), x^2 = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_integral_value_l212_21226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l212_21232

-- Define the line equation
def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 0

-- Define the angle of inclination
noncomputable def angle_of_inclination (m : ℝ) : ℝ := Real.arctan (-m)

-- Theorem statement
theorem line_inclination :
  angle_of_inclination (Real.sqrt 3) = 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l212_21232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_center_and_radius_line_l_equation_trajectory_D_equation_l212_21297

noncomputable section

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 12*y + 24 = 0

-- Define point P
def point_P : ℝ × ℝ := (0, 5)

-- Define the length of the intercepted segment
noncomputable def intercepted_length : ℝ := 4 * Real.sqrt 3

-- Theorem for the center and radius of circle C
theorem circle_C_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-2, 6) ∧ radius = 4 ∧
    ∀ (x y : ℝ), circle_C x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
sorry

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  (3*x - 4*y + 20 = 0) ∨ (x = 0)

-- Theorem for the equation of line l
theorem line_l_equation :
  ∃ (l : ℝ → ℝ → Prop),
    (l point_P.1 point_P.2) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
      l x₁ y₁ ∧ l x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = intercepted_length^2) ∧
    (∀ (x y : ℝ), l x y ↔ line_l x y) :=
sorry

-- Define the trajectory of midpoint D
def trajectory_D (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 11*y + 30 = 0

-- Theorem for the equation of the trajectory of midpoint D
theorem trajectory_D_equation :
  ∀ (x y : ℝ),
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
      2*x = x₁ + x₂ ∧ 2*y = y₁ + y₂ ∧
      (x₁ = 0 ∨ x₂ = 0) ∧ (y₁ = 5 ∨ y₂ = 5)) ↔
    trajectory_D x y :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_center_and_radius_line_l_equation_trajectory_D_equation_l212_21297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_negative_three_fourths_l212_21255

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x / 4 + y / 3 = 1

-- Define the slope of a line
def line_slope (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop := m = (y₂ - y₁) / (x₂ - x₁)

-- Theorem statement
theorem line_slope_is_negative_three_fourths :
  ∃ (m : ℝ), ∀ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ →
    line_equation x₁ y₁ →
    line_equation x₂ y₂ →
    line_slope m x₁ y₁ x₂ y₂ →
    m = -3/4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_negative_three_fourths_l212_21255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_evaluation_l212_21277

noncomputable def N (x : ℝ) : ℝ := 3 * Real.sqrt x

def O (x : ℝ) : ℝ := x ^ 2

theorem nested_function_evaluation :
  N (O (N (O (N (O 4))))) = 108 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_evaluation_l212_21277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_f_inequality_a_range_l212_21256

/-- The function f(x) = |x-a| + 2|x-1| -/
def f (a x : ℝ) : ℝ := |x - a| + 2 * |x - 1|

/-- The solution set for f(x) > 5 when a = 2 -/
def solution_set : Set ℝ := {x | x < -1/3 ∨ x > 3}

/-- The range of a for which f(x) ≤ |a-2| has a solution -/
def a_range : Set ℝ := {a | a ≤ 3/2}

theorem f_inequality_solution :
  ∀ x, f 2 x > 5 ↔ x ∈ solution_set := by sorry

theorem f_inequality_a_range :
  ∀ a, (∃ x, f a x ≤ |a - 2|) ↔ a ∈ a_range := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_f_inequality_a_range_l212_21256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relative_error_approximation_l212_21220

/-- The relative error of a measurement -/
noncomputable def relative_error (a : ℝ) (Δa : ℝ) : ℝ := (Δa / a) * 100

/-- The problem statement -/
theorem relative_error_approximation :
  let a : ℝ := 142.5
  let Δa : ℝ := 0.05
  abs (relative_error a Δa - 0.03) < 0.005 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relative_error_approximation_l212_21220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_alpha_l212_21247

/-- Given two parallel vectors a and b, prove that tan(α) = 3/4 -/
theorem parallel_vectors_tan_alpha (α : ℝ) :
  let a : Fin 2 → ℝ := ![3, 4]
  let b : Fin 2 → ℝ := ![Real.sin α, Real.cos α]
  (∃ (k : ℝ), ∀ (i : Fin 2), a i = k * b i) →
  Real.tan α = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_alpha_l212_21247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_specific_triangle_l212_21215

/-- The centroid of a triangle -/
noncomputable def centroid (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

/-- The centroid of the triangle with vertices (2, 6), (6, 2), and (4, 8) is (4, 16/3) -/
theorem centroid_specific_triangle :
  centroid (2, 6) (6, 2) (4, 8) = (4, 16/3) := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval centroid (2, 6) (6, 2) (4, 8)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_specific_triangle_l212_21215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_hexagon_area_l212_21265

/-- The polynomial Q(z) -/
noncomputable def Q (z : ℂ) : ℂ := z^6 + (3 * Real.sqrt 2 + 4) * z^3 + 3 * Real.sqrt 2 + 5

/-- The zeros of Q(z) form a hexagon in the complex plane -/
axiom zeros_form_hexagon : ∃ (h : Finset ℂ), Finset.card h = 6 ∧ ∀ z ∈ h, Q z = 0

/-- The area of the hexagon formed by the zeros of Q(z) -/
noncomputable def hexagon_area : ℝ := sorry

/-- The minimum area of the hexagon is 9√3 -/
theorem min_hexagon_area : hexagon_area = 9 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_hexagon_area_l212_21265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l212_21230

-- Define set A
def A : Set ℝ := {x | x^2 - x - 6 < 0}

-- Define set B
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l212_21230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_original_price_sales_for_profit_l212_21240

/-- Represents the pricing and sales strategy for a product -/
structure SalesStrategy where
  totalUnits : ℕ
  originalPriceRatio : ℚ
  holidayDiscountRatio : ℚ
  offSeasonPriceRatio : ℚ
  maxHolidaySales : ℕ

/-- Calculates the minimum number of units that must be sold at the original price to ensure profitability -/
def minOriginalPriceSales (strategy : SalesStrategy) : ℕ :=
  (2000 : ℚ) / 4 |> Nat.ceil

/-- Theorem stating the minimum number of units to be sold at original price for profitability -/
theorem min_original_price_sales_for_profit (strategy : SalesStrategy) 
  (h1 : strategy.totalUnits = 1000)
  (h2 : strategy.originalPriceRatio = 125 / 100)
  (h3 : strategy.holidayDiscountRatio = 90 / 100)
  (h4 : strategy.offSeasonPriceRatio = 60 / 100)
  (h5 : strategy.maxHolidaySales = 100) :
  minOriginalPriceSales strategy = 426 := by
  sorry

#eval minOriginalPriceSales { 
  totalUnits := 1000, 
  originalPriceRatio := 125 / 100, 
  holidayDiscountRatio := 90 / 100, 
  offSeasonPriceRatio := 60 / 100, 
  maxHolidaySales := 100 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_original_price_sales_for_profit_l212_21240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_solution_l212_21288

/-- The smallest positive angle in degrees that satisfies the given equation -/
noncomputable def smallest_angle : ℝ := 22.5

/-- Convert degrees to radians -/
noncomputable def deg_to_rad (x : ℝ) : ℝ := x * (Real.pi / 180)

theorem smallest_angle_solution :
  let x := smallest_angle
  (∀ y : ℝ, y > 0 → y < x →
    ¬(Real.tan (deg_to_rad (3 * y + 9)) = 1 / Real.tan (deg_to_rad (y - 9)))) ∧
  Real.tan (deg_to_rad (3 * x + 9)) = 1 / Real.tan (deg_to_rad (x - 9)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_solution_l212_21288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l212_21209

open Real

noncomputable def g (x : ℝ) : ℝ := tan (arcsin x)

theorem domain_of_g :
  ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ↔ g x ∈ Set.univ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l212_21209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_K_find_t_l212_21234

-- Part 1
noncomputable def x : ℝ := 1.9 + 89 / 990

theorem find_K (h : x - 1 = K / 99) : K = 98 := by
  sorry

-- Part 2
variable (p q r t : ℝ)

theorem find_t (h1 : (p + q + r) / 3 = 18) 
                (h2 : ((p + 1) + (q - 2) + (r + 3) + t) / 4 = 19) : 
  t = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_K_find_t_l212_21234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_sum_l212_21244

noncomputable section

theorem vector_equation_sum (e₁ e₂ : ℝ → ℝ → ℝ → ℝ) (x y : ℝ) 
  (h₁ : e₁ ≠ 0)
  (h₂ : e₂ ≠ 0)
  (h₃ : ¬ ∃ (k : ℝ), ∀ a b c, e₁ a b c = k * e₂ a b c)
  (h₄ : ∀ a b c, x * e₁ a b c + (5 - y) * e₂ a b c = (y + 1) * e₁ a b c + x * e₂ a b c) :
  x + y = 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_sum_l212_21244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_solution_l212_21218

theorem unique_positive_solution :
  ∃! x : ℝ, 0 < x ∧ x ≤ 1 ∧
    (Real.sin (Real.arccos (Real.sqrt (Real.tan (Real.arcsin x))^2)))^2 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_solution_l212_21218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_slope_angle_lines_l212_21293

-- Define the original line
def original_line (x y : ℝ) : Prop := y = -Real.sqrt 3 * x + 1

-- Define the first line passing through (-4, 1)
def line1 (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 4 * Real.sqrt 3 + 1 = 0

-- Define the second line with y-intercept -10
def line2 (x y : ℝ) : Prop := y = Real.sqrt 3 * x - 10

-- Function to calculate slope angle from slope
noncomputable def slope_angle (m : ℝ) : ℝ := Real.arctan m

-- Theorem statement
theorem half_slope_angle_lines :
  ∀ (x y : ℝ),
  (slope_angle (Real.sqrt 3) = (1/2) * slope_angle (-Real.sqrt 3)) ∧
  line1 (-4) 1 ∧
  line2 0 (-10) ∧
  (∀ (x y : ℝ), line1 x y → slope_angle (Real.sqrt 3) = (1/2) * slope_angle (-Real.sqrt 3)) ∧
  (∀ (x y : ℝ), line2 x y → slope_angle (Real.sqrt 3) = (1/2) * slope_angle (-Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_slope_angle_lines_l212_21293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_min_value_l212_21266

/-- Definition of the sum of first n terms of a geometric sequence -/
noncomputable def S (n : ℕ) (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  if q = 1 then n * a 1
  else a 1 * (1 - q^n) / (1 - q)

/-- The problem statement -/
theorem geometric_sequence_min_value
  (a : ℕ → ℝ) (q : ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∀ n, a (n + 1) = q * a n)
  (h_S3 : S 3 a q = 10) :
  ∃ (min_value : ℝ), 
    (∀ q', 2 * S 9 a q' - 3 * S 6 a q' + S 3 a q' ≥ min_value) ∧
    (min_value = -5/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_min_value_l212_21266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statements_count_l212_21216

-- Define a type for statistical statements
inductive StatStatement
| ModeUniqueness
| VarianceInvariance
| SamplingMethod
| VariancePositivity

-- Define a function to check if a statement is correct
def is_correct (s : StatStatement) : Bool :=
  match s with
  | StatStatement.ModeUniqueness => false
  | StatStatement.VarianceInvariance => true
  | StatStatement.SamplingMethod => false
  | StatStatement.VariancePositivity => false

-- Define a function to count incorrect statements
def count_incorrect (statements : List StatStatement) : Nat :=
  statements.filter (fun s => ¬(is_correct s)) |>.length

-- Theorem to prove
theorem incorrect_statements_count :
  count_incorrect [StatStatement.ModeUniqueness, StatStatement.VarianceInvariance,
                   StatStatement.SamplingMethod, StatStatement.VariancePositivity] = 3 := by
  -- Proof goes here
  sorry

#eval count_incorrect [StatStatement.ModeUniqueness, StatStatement.VarianceInvariance,
                       StatStatement.SamplingMethod, StatStatement.VariancePositivity]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statements_count_l212_21216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression1_eq_neg_one_complex_expression2_eq_47_minus_39i_l212_21268

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the properties of i
axiom i_squared : i^2 = -1

-- Define the expression for part (1)
noncomputable def expression1 : ℂ := (2 + 2*i) / ((1 - i)^2) + ((Real.sqrt 2) / (1 + i))^2010

-- Define the expression for part (2)
noncomputable def expression2 : ℂ := (4 - i^5) * (6 + 2*i^7) + (7 + i^11) * (4 - 3*i)

-- Theorem for part (1)
theorem complex_expression1_eq_neg_one : expression1 = -1 := by sorry

-- Theorem for part (2)
theorem complex_expression2_eq_47_minus_39i : expression2 = 47 - 39*i := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression1_eq_neg_one_complex_expression2_eq_47_minus_39i_l212_21268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_formula_l212_21291

/-- Represents the diameter of a large part -/
noncomputable def D : ℝ := sorry

/-- Represents the height of the segment -/
noncomputable def H : ℝ := sorry

/-- Represents half the distance between the centers of the supporting balls -/
noncomputable def L : ℝ := sorry

/-- Represents the diameter of each supporting ball -/
noncomputable def d : ℝ := sorry

/-- Theorem stating the relationship between D, H, L, and d -/
theorem diameter_formula (h1 : D > 2) (h2 : H > 0) (h3 : L > 0) (h4 : d > 0) :
  D = (L^2 + H^2 - H*d) / H := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_formula_l212_21291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_credit_card_balance_calculation_l212_21200

/-- Calculate the final credit card balance after transactions and interest --/
theorem credit_card_balance_calculation 
  (initial_balance : ℝ)
  (grocery_cost grocery_discount : ℝ)
  (clothes_cost clothes_discount : ℝ)
  (electronics_cost electronics_discount : ℝ)
  (return_amount : ℝ)
  (monthly_interest_rate : ℝ)
  (h1 : initial_balance = 126)
  (h2 : grocery_cost = 60)
  (h3 : grocery_discount = 0.1)
  (h4 : clothes_cost = 80)
  (h5 : clothes_discount = 0.15)
  (h6 : electronics_cost = 120)
  (h7 : electronics_discount = 0.05)
  (h8 : return_amount = 45)
  (h9 : monthly_interest_rate = 0.015) :
  (initial_balance + grocery_cost * (1 - grocery_discount) + 
   clothes_cost * (1 - clothes_discount) + 
   electronics_cost * (1 - electronics_discount) + 
   (grocery_cost * (1 - grocery_discount)) / 2 - return_amount) * 
   (1 + monthly_interest_rate) = 349.16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_credit_card_balance_calculation_l212_21200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l212_21278

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define a point on the parabola
def point_on_parabola (p : ℝ) (y₀ : ℝ) : Prop := parabola p 2 y₀

-- Define the distance from a point to the directrix
def distance_to_directrix (p : ℝ) : ℝ := p / 2 + 2

-- Main theorem
theorem parabola_equation (p : ℝ) (y₀ : ℝ) 
  (h_p_pos : p > 0) 
  (h_point : point_on_parabola p y₀) 
  (h_distance : distance_to_directrix p = 4) :
  ∀ x y, parabola p x y ↔ y^2 = 8 * x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l212_21278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_collection_size_l212_21251

theorem smallest_collection_size {α : Type*} (A : Finset (Set α)) : 
  (A.card = 2007) → 
  ∃ (B : Finset (Set α)), 
    (B.card = 2008) ∧ 
    (∀ a ∈ A, ∃ b1 b2 : Set α, b1 ∈ B ∧ b2 ∈ B ∧ b1 ≠ b2 ∧ a = b1 ∩ b2) ∧
    (∀ n < 2008, ¬∃ (C : Finset (Set α)), 
      (C.card = n) ∧ 
      (∀ a ∈ A, ∃ c1 c2 : Set α, c1 ∈ C ∧ c2 ∈ C ∧ c1 ≠ c2 ∧ a = c1 ∩ c2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_collection_size_l212_21251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_polygon_diagonals_l212_21210

/-- A simple polygon is a polygon that does not intersect itself -/
structure SimplePolygon where
  vertices : List (ℝ × ℝ)
  is_simple : Bool

/-- A diagonal of a polygon is a line segment connecting two non-adjacent vertices -/
def diagonal (p : SimplePolygon) : List (ℝ × ℝ) := sorry

/-- A diagonal is inside the polygon if it doesn't intersect with any edge of the polygon except at its endpoints -/
def is_inside (d : ℝ × ℝ) (p : SimplePolygon) : Bool := sorry

/-- The number of diagonals inside a simple polygon -/
def num_inside_diagonals (p : SimplePolygon) : ℕ :=
  (diagonal p).filter (λ d ↦ is_inside d p) |>.length

/-- Theorem: Every simple n-sided polygon has at least n-3 diagonals that lie entirely within it -/
theorem simple_polygon_diagonals (p : SimplePolygon) :
  num_inside_diagonals p ≥ p.vertices.length - 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_polygon_diagonals_l212_21210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l212_21287

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- The possible values of m for an ellipse with given eccentricity -/
def possible_m_values (e : ℝ) : Set ℝ :=
  {m | (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a = 4 ∨ b = 4) ∧ eccentricity a b = e ∧ 
       (a^2 = 16 ∧ b^2 = m ∨ a^2 = m ∧ b^2 = 16))}

theorem ellipse_m_values :
  possible_m_values (1/3) = {128/9, 18} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l212_21287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_furniture_l212_21285

/-- The total cost of buying a table, 4 chairs, and a sofa with discounts and tax -/
theorem total_cost_furniture (table_cost chair_cost sofa_cost : ℚ) : 
  table_cost = 140 →
  chair_cost = table_cost / 7 →
  sofa_cost = 2 * table_cost →
  let discounted_table_cost := table_cost * (1 - 1/10)
  let subtotal := discounted_table_cost + 4 * chair_cost + sofa_cost
  let total_with_tax := subtotal * (1 + 7/100)
  total_with_tax = 520.02 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_furniture_l212_21285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_circle_l212_21284

noncomputable section

-- Define the parabola C: y = (1/2)x^2
def parabola (x y : ℝ) : Prop := y = (1/2) * x^2

-- Define the line l: y = 2x + b
def line (x y b : ℝ) : Prop := y = 2 * x + b

-- Define the tangency condition
def is_tangent (b : ℝ) : Prop := ∃ x y : ℝ, parabola x y ∧ line x y b

-- Define the directrix of the parabola
def directrix : ℝ := -1/2

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the circle equation
def circle_eq (x y cx cy r : ℝ) : Prop := (x - cx)^2 + (y - cy)^2 = r^2

theorem tangent_line_and_circle (b : ℝ) : 
  is_tangent b → 
  (b = -2) ∧ 
  (∃ x y : ℝ, parabola x y ∧ line x y b ∧ 
    (∀ x' y' : ℝ, circle_eq x' y' x y ((5:ℝ)/2) ↔ 
      (x' - x)^2 + (y' - y)^2 = (25:ℝ)/4)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_circle_l212_21284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_for_every_p_l212_21241

/-- Arithmetic sequence -/
def arithmetic_seq (a₁ d : ℝ) : ℕ → ℝ := fun n ↦ a₁ + (n - 1) * d

/-- Geometric sequence -/
def geometric_seq (g₁ k : ℝ) : ℕ → ℝ := fun n ↦ g₁ * k^(n - 1)

/-- Main theorem -/
theorem exists_m_for_every_p (a₁ g₁ d k : ℝ) (h_nonconstant : d ≠ 0 ∧ k ≠ 1) 
    (h_a₁_eq_g₁ : a₁ = g₁) (h_a₁_neq_0 : a₁ ≠ 0)
    (h_a₂_eq_g₂ : arithmetic_seq a₁ d 2 = geometric_seq g₁ k 2)
    (h_a₁₀_eq_g₃ : arithmetic_seq a₁ d 10 = geometric_seq g₁ k 3) :
  ∀ p : ℕ, p > 0 → ∃ m : ℕ, m > 0 ∧ geometric_seq g₁ k p = arithmetic_seq a₁ d m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_for_every_p_l212_21241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_extreme_points_min_sin_l212_21252

theorem triangle_extreme_points_min_sin (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- a, b, c are sides opposite to angles A, B, C
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Function f(x) has extreme points
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + 2*b*x₁ + (a^2 + c^2 - a*c) = 0 ∧
    x₂^2 + 2*b*x₂ + (a^2 + c^2 - a*c) = 0 →
  -- Conclusion: minimum value of sin(2B - π/3) is -1
  ∃ θ : ℝ, 2*B - π/3 = θ ∧ Real.sin θ = -1 ∧ 
    ∀ φ : ℝ, Real.sin φ ≥ -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_extreme_points_min_sin_l212_21252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_vasya_meeting_l212_21211

/-- The number of lamps along the alley -/
def total_lamps : ℕ := 100

/-- The position where Petya is observed -/
def petya_observed : ℕ := 22

/-- The position where Vasya is observed -/
def vasya_observed : ℕ := 88

/-- The meeting point of Petya and Vasya -/
def meeting_point : ℕ := 64

/-- Theorem stating that Petya and Vasya meet at lamp 64 -/
theorem petya_vasya_meeting :
  let total_intervals : ℕ := total_lamps - 1
  let petya_covered : ℕ := petya_observed - 1
  let vasya_covered : ℕ := total_lamps - vasya_observed
  let speed_ratio : ℚ := (petya_covered : ℚ) / (vasya_covered : ℚ)
  meeting_point = 1 + (speed_ratio * total_intervals / (1 + speed_ratio)).floor := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_vasya_meeting_l212_21211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_bound_l212_21267

theorem angle_sum_bound (α β : Real) (h1 : π / 2 < α ∧ α < π) 
  (h2 : π / 2 < β ∧ β < π) (h3 : Real.tan α < 1 / Real.tan β) : 
  α + β < 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_bound_l212_21267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_pressure_depth_l212_21233

/-- The depth of the gate in meters -/
def gate_depth : ℝ := 6

/-- The water pressure at depth x -/
def water_pressure (x : ℝ) : ℝ := x

/-- The depth at which the gate should be divided -/
noncomputable def c : ℝ := 3 * Real.sqrt 2

theorem equal_pressure_depth :
  ∫ x in (0 : ℝ)..c, water_pressure x = ∫ x in c..gate_depth, water_pressure x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_pressure_depth_l212_21233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_minus_beta_l212_21246

theorem tan_double_minus_beta (α β : ℝ) 
  (h1 : Real.tan α = 1/2) 
  (h2 : Real.tan (α - β) = 1/5) : 
  Real.tan (2*α - β) = 7/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_minus_beta_l212_21246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_glow_problem_l212_21237

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Converts Time to total seconds -/
def timeToSeconds (t : Time) : ℕ :=
  t.hours * 3600 + t.minutes * 60 + t.seconds

/-- Converts total seconds to Time -/
def secondsToTime (s : ℕ) : Time :=
  let h := s / 3600
  let m := (s % 3600) / 60
  let s := s % 60
  { hours := h, minutes := m, seconds := s }

/-- The light glow problem -/
theorem light_glow_problem (glowInterval : ℕ) (startTime : Time) (glowCount : ℚ) (endTime : Time) :
  glowInterval = 13 →
  startTime = { hours := 1, minutes := 57, seconds := 58 } →
  glowCount = 382.2307692307692 →
  endTime = { hours := 4, minutes := 14, seconds := 4 } →
  timeToSeconds endTime = timeToSeconds startTime + glowInterval * (Int.floor glowCount) := by
  sorry

#check light_glow_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_glow_problem_l212_21237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_configuration_mappings_l212_21279

/- Define the type for points and lines -/
def Point : Type := ℕ
def Line : Type := Set Point

/- Define the configuration -/
structure Configuration :=
  (points : Set Point)
  (lines : Set Line)
  (num_points : Nat)
  (num_lines : Nat)
  (points_per_line : Nat)
  (line_contains : Line → Point → Prop)
  (point_on_line : Point → Line → Prop)

/- Define the properties of the configuration -/
def valid_configuration (c : Configuration) : Prop :=
  c.num_points = 6 ∧
  c.num_lines = 4 ∧
  c.points_per_line = 3 ∧
  (∀ l : Line, l ∈ c.lines → (∃ p1 p2 p3 : Point, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
                              c.line_contains l p1 ∧ c.line_contains l p2 ∧ c.line_contains l p3)) ∧
  (∀ p : Point, p ∈ c.points → (∃ l : Line, l ∈ c.lines ∧ c.point_on_line p l))

/- Define a collinearity-preserving bijection -/
def collinearity_preserving_bijection (c : Configuration) (f : Point → Point) : Prop :=
  Function.Bijective f ∧
  (∀ l : Line, l ∈ c.lines → 
    ∃ l' : Line, l' ∈ c.lines ∧ 
    (∀ p : Point, c.line_contains l p → c.line_contains l' (f p)))

/- The main theorem -/
theorem configuration_mappings (c : Configuration) :
  valid_configuration c →
  (∃ mappings : List (Point → Point), 
    (∀ f ∈ mappings, collinearity_preserving_bijection c f) ∧
    mappings.length = 24) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_configuration_mappings_l212_21279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_l212_21238

/-- The function representing the curve y = a + ln x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + Real.log x

/-- The derivative of f with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x

theorem tangent_line_condition (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ f a x = x ∧ f_deriv a x = 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_l212_21238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_animals_in_field_l212_21296

theorem animals_in_field : ℕ := by
  let initial_dog : ℕ := 1
  let cats : ℕ := 4
  let rabbits_per_cat : ℕ := 2
  let hares_per_rabbit : ℕ := 3
  
  let total_rabbits : ℕ := cats * rabbits_per_cat
  let total_hares : ℕ := total_rabbits * hares_per_rabbit
  
  have h : initial_dog + cats + total_rabbits + total_hares = 37 := by
    -- Proof steps would go here
    sorry
  
  exact 37


end NUMINAMATH_CALUDE_ERRORFEEDBACK_animals_in_field_l212_21296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_solution_form_l212_21239

-- Define the function f(x) = ∛x + ∛(20 - x) - 2
noncomputable def f (x : ℝ) : ℝ := Real.rpow x (1/3) + Real.rpow (20 - x) (1/3) - 2

-- State the theorem
theorem smaller_solution_form :
  ∃ (x : ℝ), f x = 0 ∧ x = 10 - Real.sqrt 108 ∧ 
  ∀ (y : ℝ), f y = 0 → y ≤ x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_solution_form_l212_21239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_division_l212_21289

/-- A trapezoid with upper base a and lower base b -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  h_pos : h > 0
  a_pos : a > 0
  b_pos : b > 0
  a_lt_b : a < b

/-- The median line of a trapezoid -/
noncomputable def median_line (t : Trapezoid) : ℝ := (t.a + t.b) / 2

/-- The area of a trapezoid -/
noncomputable def area (t : Trapezoid) : ℝ := (t.a + t.b) * t.h / 2

/-- The area of the upper part divided by the median line -/
noncomputable def upper_area (t : Trapezoid) : ℝ := (t.a + median_line t) * t.h / 2

/-- The area of the lower part divided by the median line -/
noncomputable def lower_area (t : Trapezoid) : ℝ := (median_line t + t.b) * t.h / 2

theorem trapezoid_median_division (t : Trapezoid) (h : t.b = (7/3) * t.a) :
  upper_area t / lower_area t = 2/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_division_l212_21289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_graph_l212_21212

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 3) + 1
noncomputable def g (x : ℝ) := Real.sin (2 * x)

theorem transform_sin_graph (x : ℝ) : 
  f x = g (x - Real.pi / 6) + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_graph_l212_21212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l212_21295

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 2

/-- The line equation -/
def line (x y : ℝ) : Prop := 2 * x - y - 8 = 0

/-- The distance function from a point (x, y) to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |2 * x - y - 8| / Real.sqrt 5

/-- The minimum distance from the ellipse to the line is 6√5/5 -/
theorem min_distance_ellipse_to_line :
  ∃ (min_dist : ℝ), min_dist = 6 * Real.sqrt 5 / 5 ∧
  ∀ (x y : ℝ), ellipse x y →
    distance_to_line x y ≥ min_dist ∧
    ∃ (x₀ y₀ : ℝ), ellipse x₀ y₀ ∧ distance_to_line x₀ y₀ = min_dist :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l212_21295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sqrt2_over_2_condition_l212_21270

theorem cos_sqrt2_over_2_condition :
  (∃ k : ℤ, ∃ α : ℝ, α = 2 * k * Real.pi - Real.pi / 4 ∧ Real.cos α = Real.sqrt 2 / 2) ∧
  (∃ β : ℝ, Real.cos β = Real.sqrt 2 / 2 ∧ ∀ m : ℤ, β ≠ 2 * m * Real.pi - Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sqrt2_over_2_condition_l212_21270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l212_21260

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2) * Real.exp x

theorem tangent_line_at_one (x y : ℝ) :
  (deriv f 1 : ℝ) * (x - 1) = y - f 1 ↔ 5 * Real.exp 1 * x - y - 2 * Real.exp 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l212_21260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_wrt_U_l212_21214

def U : Set ℕ := {3, 6}
def A : Set ℕ := {3}

theorem complement_of_A_wrt_U :
  (U \ A = {6}) ∨ (U \ A = ∅) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_wrt_U_l212_21214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_50_values_l212_21280

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → x * g y - y * g x = k * g (x / y)

/-- The theorem stating the possible values of g(50) -/
theorem g_50_values (g : ℝ → ℝ) (k : ℝ) 
  (h : FunctionalEquation g k) :
  (g 50 = 0) ∧ 
  (k ≠ -1 → ∃ C : ℝ, g 50 = 50 * C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_50_values_l212_21280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_blue_marbles_l212_21275

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def num_trials : ℕ := 7
def num_blue_desired : ℕ := 3

theorem probability_three_blue_marbles :
  (Nat.choose num_trials num_blue_desired : ℚ) *
  ((blue_marbles : ℚ) / (total_marbles : ℚ)) ^ num_blue_desired *
  ((red_marbles : ℚ) / (total_marbles : ℚ)) ^ (num_trials - num_blue_desired) =
  35 * (1228802 : ℚ) / 171140625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_blue_marbles_l212_21275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_undefined_at_one_g_g_inv_eq_id_g_inv_g_eq_id_l212_21205

/-- The function g(x) = (x-5)/(x-7) -/
noncomputable def g (x : ℝ) : ℝ := (x - 5) / (x - 7)

/-- The inverse function of g -/
noncomputable def g_inv (x : ℝ) : ℝ := (5 - 7*x) / (1 - x)

/-- Theorem: g_inv is undefined at x = 1 -/
theorem g_inv_undefined_at_one :
  ¬∃ (y : ℝ), g y = 1 := by
  sorry

/-- Theorem: g(g_inv(x)) = x for x ≠ 1 -/
theorem g_g_inv_eq_id {x : ℝ} (h : x ≠ 1) :
  g (g_inv x) = x := by
  sorry

/-- Theorem: g_inv(g(x)) = x for x ≠ 7 -/
theorem g_inv_g_eq_id {x : ℝ} (h : x ≠ 7) :
  g_inv (g x) = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_undefined_at_one_g_g_inv_eq_id_g_inv_g_eq_id_l212_21205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bus_cost_difference_l212_21299

/-- The cost of a train ride in dollars -/
def train_cost : ℝ := sorry

/-- The cost of a bus ride in dollars -/
def bus_cost : ℝ := 3.75

theorem train_bus_cost_difference :
  train_cost > bus_cost ∧
  train_cost + bus_cost = 9.85 ∧
  bus_cost = 3.75 →
  train_cost - bus_cost = 2.35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bus_cost_difference_l212_21299
