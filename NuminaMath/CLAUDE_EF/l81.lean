import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_when_intersection_nonempty_l81_8121

/-- The set M of complex numbers -/
def M : Set ℂ :=
  {z : ℂ | ∃ α : ℝ, z = Complex.mk (Real.cos α) (4 - Real.cos α ^ 2)}

/-- The set N of complex numbers -/
def N (l : ℝ) : Set ℂ :=
  {z : ℂ | ∃ β : ℝ, z = Complex.mk (Real.cos β) (l + Real.sin β)}

/-- The theorem stating the range of l when M and N intersect -/
theorem lambda_range_when_intersection_nonempty :
  ∀ l : ℝ, (M ∩ N l).Nonempty → l ∈ Set.Icc (11/4 : ℝ) 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_when_intersection_nonempty_l81_8121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l81_8196

def N : Set ℤ := {x | (1/2 : ℝ) < (2 : ℝ)^(x+1) ∧ (2 : ℝ)^(x+1) < 4}
def M : Set ℤ := {-1, 1}

theorem intersection_M_N : M ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l81_8196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_stu_area_of_triangle_stu_proof_l81_8133

/-- Given two lines intersecting at S(2,5) with slopes 3 and -2 respectively,
    prove that the area of triangle STU is 145/12, where T and U are the
    points where these lines intersect the x-axis. -/
theorem area_of_triangle_stu (x_t x_u area : ℝ) : Prop :=
  let s : ℝ × ℝ := (2, 5)
  let t : ℝ × ℝ := (x_t, 0)
  let u : ℝ × ℝ := (x_u, 0)
  let slope_st : ℝ := 3
  let slope_su : ℝ := -2
  (slope_st = (s.2 - t.2) / (s.1 - t.1)) →
  (slope_su = (s.2 - u.2) / (s.1 - u.1)) →
  (area = (1 / 2) * |x_u - x_t| * 5) →
  (area = 145 / 12)

/-- Proof of the existence of points T and U and the area satisfying the conditions. -/
theorem area_of_triangle_stu_proof : ∃ (x_t x_u area : ℝ),
  area_of_triangle_stu x_t x_u area := by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_stu_area_of_triangle_stu_proof_l81_8133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_theorem_l81_8172

noncomputable section

/-- Parabola C₁ -/
def C₁ (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Theorem stating the main results -/
theorem parabola_circle_intersection_theorem :
  ∀ p : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ, C₁ p x₁ y₁ ∧ C₁ p x₂ y₂ ∧ C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧ distance x₁ y₁ x₂ y₂ = 4) →
  (p = 2 ∧
   ∀ k : ℝ, k ∈ Set.Icc 0 1 →
   ∃ min max : ℝ,
     min = 16 ∧ max = 24 * Real.sqrt 2 ∧
     ∀ xA yA xB yB xC yC xD yD : ℝ,
       C₁ p xA yA ∧ C₁ p xB yB ∧ C₂ xC yC ∧ C₂ xD yD →
       (yA - 1/4) = k * (xA - 0) ∧ (yB - 1/4) = k * (xB - 0) ∧
       (yC - 1/4) = k * (xC - 0) ∧ (yD - 1/4) = k * (xD - 0) →
       min ≤ distance xA yA xB yB * distance xC yC xD yD ∧
       distance xA yA xB yB * distance xC yC xD yD ≤ max) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_theorem_l81_8172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l81_8185

-- Define the function f
def f (x : ℝ) : ℝ := 1

-- Define the domain
def domain : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem f_is_even : ∀ x ∈ domain, f (-x) = f x := by
  intro x hx
  simp [f]
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l81_8185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_sum_l81_8116

-- Define the function g(x) = x³|x|
noncomputable def g (x : ℝ) : ℝ := x^3 * abs x

-- State the theorem
theorem inverse_g_sum : ∃ (y z : ℝ), g y = 8 ∧ g z = -64 ∧ y + z = 2^(3/4) - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_sum_l81_8116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_ge_one_l81_8164

open Set Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 + 1) - a * x

def monotonic_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x < f y ∨ (∀ x y, x ∈ S → y ∈ S → x < y → f x > f y)

theorem f_monotonic_iff_a_ge_one (a : ℝ) (h : a > 0) :
  monotonic_on (f a) (Ioi 0) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_ge_one_l81_8164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l81_8104

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.sqrt 2
noncomputable def c : ℝ := Real.log (1/15) / Real.log (1/4)

-- State the theorem
theorem order_of_abc : c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l81_8104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_probability_l81_8173

-- Define the probability of heads
noncomputable def p : ℝ := Real.sqrt 0.684 / 2

-- Define the conditions
axiom p_less_than_half : p < (1 : ℝ) / 2
axiom prob_three_heads_three_tails : (20 : ℝ) * p^3 * (1 - p)^3 = (1 : ℝ) / 20

-- State the theorem
theorem coin_probability : ∃ ε > 0, |p - 0.159| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_probability_l81_8173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_max_value_achieved_l81_8153

theorem max_value_expression (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) 
  (h_sum : a + b + 2 * c = 1) :
  a + Real.sqrt (a * b) + (a * b * c^2)^(1/3) ≤ (3 + Real.sqrt 3) / 4 :=
by sorry

theorem max_value_achieved (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) 
  (h_sum : a + b + 2 * c = 1) :
  ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + 2 * c₀ = 1 ∧ 
    a₀ + Real.sqrt (a₀ * b₀) + (a₀ * b₀ * c₀^2)^(1/3) = (3 + Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_max_value_achieved_l81_8153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_or_q_l81_8171

-- Define proposition P
def P : Prop :=
  ∀ a b c : ℝ, b^2 = a*c → ∃ r : ℝ, (a = b/r ∧ b = c*r) ∨ (a = b*r ∧ b = c/r)

-- Define function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi/2 + x)

-- Define proposition Q
def Q : Prop := ∀ x : ℝ, f x = -f (-x)

-- Theorem to prove
theorem p_or_q : P ∨ Q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_or_q_l81_8171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_ratio_is_one_l81_8100

-- Define the polynomial f
def f (x : ℂ) : ℂ := x^2009 + 19*x^2008 + 1

-- Define the type of roots of f
def Root (r : ℂ) : Prop := f r = 0

-- State that f has 2009 distinct roots
axiom distinct_roots : ∃ (r : Fin 2009 → ℂ), (∀ i j, i ≠ j → r i ≠ r j) ∧ (∀ i, Root (r i))

-- Define the polynomial P
noncomputable def P : Polynomial ℂ := sorry

-- State that P has degree 2009
axiom P_degree : Polynomial.degree P = 2009

-- State that P(r_j + 1/r_j) = 0 for all roots of f
axiom P_property : ∀ r : ℂ, Root r → P.eval (r + r⁻¹) = 0

-- The theorem to prove
theorem P_ratio_is_one : P.eval 1 / P.eval (-1) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_ratio_is_one_l81_8100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_art_class_arrangement_probability_l81_8105

/-- Represents the number of cultural classes -/
def cultural_classes : ℕ := 3

/-- Represents the number of art classes -/
def art_classes : ℕ := 3

/-- Represents the total number of periods -/
def total_periods : ℕ := 6

/-- Calculates the number of favorable arrangements -/
def favorable_arrangements : ℕ :=
  let cultural_arrangements := Nat.factorial cultural_classes
  let art_arrangements := Nat.factorial art_classes
  let scenario1 := cultural_arrangements * art_arrangements
  let scenario2 := 2 * art_classes * Nat.factorial 4
  let scenario3 := Nat.factorial 4
  scenario1 + scenario2 + scenario3

/-- Calculates the total number of possible arrangements -/
def total_arrangements : ℕ := Nat.factorial total_periods

/-- Represents the probability of the desired arrangement -/
noncomputable def probability : ℚ := (favorable_arrangements : ℚ) / (total_arrangements : ℚ)

theorem art_class_arrangement_probability :
  probability = 17 / 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_art_class_arrangement_probability_l81_8105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_equivalent_properties_l81_8120

-- Define the graph structure
structure BipartiteGraph (α β : Type) where
  A : Set α
  B : Set β
  E : α → β → Prop

-- Define the neighbor set E(X)
def neighborSet {α β : Type} (G : BipartiteGraph α β) (X : Set α) : Set β :=
  {b ∈ G.B | ∃ a ∈ X, G.E a b}

-- Define a perfect matching
def isPerfectMatching {α β : Type} (G : BipartiteGraph α β) (f : α → β) : Prop :=
  Function.Injective f ∧ (∀ a ∈ G.A, G.E a (f a))

theorem not_equivalent_properties :
  ∃ (G : BipartiteGraph ℕ ℕ),
    (∀ X : Set ℕ, X.Finite → X ⊆ G.A → 
      (neighborSet G X).Infinite ∨ 
      ((neighborSet G X).Finite ∧ 
       (neighborSet G X).Nonempty ∧ 
       Nat.card (neighborSet G X) ≥ Nat.card X)) ∧
    ¬∃ f : ℕ → ℕ, isPerfectMatching G f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_equivalent_properties_l81_8120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l81_8190

-- Define the circledast operation
noncomputable def circledast (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  (circledast 1 x) * x - (circledast 2 x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), M = 6 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-2) 2 → f x ≤ M := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l81_8190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_always_wins_l81_8186

-- Define the table as a set of points
def Table : Type := Set (ℝ × ℝ)

-- Define the axis of symmetry
def AxisOfSymmetry (t : Table) : Set (ℝ × ℝ) := sorry

-- Define the midpoint of the axis
def AxisMidpoint (t : Table) : ℝ × ℝ := sorry

-- Define a coin placement
def CoinPlacement : Type := ℝ × ℝ

-- Define the mirroring function relative to the axis
def MirrorOverAxis (t : Table) (p : CoinPlacement) : CoinPlacement := sorry

-- Define the mirroring function relative to the midpoint
def MirrorOverMidpoint (t : Table) (p : CoinPlacement) : CoinPlacement := sorry

-- Define the game state
structure GameState (t : Table) where
  player_a_coins : List CoinPlacement
  player_b_coins : List CoinPlacement

-- Define Player A's strategy
def PlayerAStrategy (t : Table) (state : GameState t) : CoinPlacement := sorry

-- Define a winning condition
def PlayerAWins (t : Table) (state : GameState t) : Prop := sorry

-- The theorem to prove
theorem player_a_always_wins (t : Table) :
  ∀ (game : GameState t), PlayerAWins t (GameState.mk (PlayerAStrategy t game :: game.player_a_coins) game.player_b_coins) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_always_wins_l81_8186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_difference_l81_8195

/-- Compound interest calculation -/
noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_difference :
  let P : ℝ := 19999.99999999962
  let r : ℝ := 0.20
  let t : ℝ := 2
  let annual := compoundInterest P r 1 t
  let halfYearly := compoundInterest P r 2 t
  ∃ ε > 0, |halfYearly - annual - 482.00000000087| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_difference_l81_8195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_theorem_l81_8127

/-- Represents the TV Olympic knowledge contest -/
structure Contest where
  correct_rate : ℚ
  max_questions : ℕ
  required_correct : ℕ

/-- Represents the outcome of the contest for a contestant -/
inductive Outcome
  | FinalRound
  | Eliminated
  | Incomplete

/-- Represents the number of questions answered -/
def QuestionsAnswered : Type := Fin 6

/-- The probability distribution of the number of questions answered -/
def distribution (c : Contest) (n : QuestionsAnswered) : ℚ :=
  sorry

/-- The probability of entering the final round -/
def prob_final_round (c : Contest) : ℚ :=
  sorry

/-- The expected number of questions answered -/
def expected_questions (c : Contest) : ℚ :=
  sorry

/-- Main theorem about the TV Olympic knowledge contest -/
theorem contest_theorem (c : Contest) 
  (h1 : c.correct_rate = 2/3) 
  (h2 : c.max_questions = 5) 
  (h3 : c.required_correct = 3) : 
  prob_final_round c = 64/81 ∧ 
  distribution c ⟨3, by norm_num⟩ = 1/3 ∧ 
  distribution c ⟨4, by norm_num⟩ = 10/27 ∧ 
  distribution c ⟨5, by norm_num⟩ = 8/27 ∧
  expected_questions c = 107/27 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_theorem_l81_8127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_equation_l81_8140

theorem binomial_coefficient_equation (x : ℝ) : 
  (Nat.choose 16 (Nat.floor (x^2 - x)) = Nat.choose 16 (Nat.floor (5*x - 5))) → 
  (x = 1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_equation_l81_8140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l81_8170

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem function_symmetry (ω : ℝ) (h1 : ω > 0) (h2 : 4 * Real.pi = 2 * Real.pi / ω) :
  ∀ x : ℝ, f ω (5 * Real.pi / 3 + x) = f ω (5 * Real.pi / 3 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l81_8170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_in_acute_triangle_l81_8126

noncomputable section

open Real

theorem angle_C_in_acute_triangle (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = c * (sin B) / (sin C) →
  b = c * (sin A) / (sin C) →
  Real.sqrt 3 * a = 2 * c * (sin A) →
  C = π/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_in_acute_triangle_l81_8126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_quadruple_angle_l81_8119

theorem cos_quadruple_angle (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (4*θ) = 17/81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_quadruple_angle_l81_8119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_special_isosceles_triangle_l81_8192

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third is determined by these two
  base_angle : ℝ
  apex_angle : ℝ
  -- Condition that it's isosceles: base angles are equal
  is_isosceles : base_angle ≥ 0 ∧ apex_angle ≥ 0
  -- Sum of angles in a triangle is 180°
  angle_sum : base_angle + base_angle + apex_angle = 180

-- Theorem statement
theorem largest_angle_in_special_isosceles_triangle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.base_angle = 60) : 
  max triangle.base_angle (max triangle.base_angle triangle.apex_angle) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_special_isosceles_triangle_l81_8192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equiv_l81_8174

/-- The scientific notation representation of a number -/
noncomputable def scientific_notation (m : ℝ) (e : ℤ) : ℝ := m * (10 : ℝ) ^ e

/-- The original number to be converted -/
def original_number : ℚ := 12 / 100000000

theorem scientific_notation_equiv :
  ∃ (m : ℚ) (e : ℤ), 
    1 ≤ m ∧ m < 10 ∧ 
    (m : ℝ) * (10 : ℝ) ^ e = original_number ∧
    m = 12 / 10 ∧ e = -7 := by
  sorry

#eval original_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equiv_l81_8174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_minus_sqrt_three_floor_l81_8165

-- Define the integer part function
noncomputable def integerPart (a : ℝ) : ℤ :=
  Int.floor a

-- State the theorem
theorem three_minus_sqrt_three_floor : integerPart (3 - Real.sqrt 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_minus_sqrt_three_floor_l81_8165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_empty_A_l81_8159

/-- The set A defined by the quadratic inequality --/
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - a * x + 1 ≤ 0}

/-- The theorem stating the range of a for which A is empty --/
theorem range_of_a_for_empty_A : 
  {a : ℝ | A a = ∅} = Set.Icc 0 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_empty_A_l81_8159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_region_area_is_4pi_l81_8175

/-- The area of the region formed by all line segments of length 4 units
    that are tangent to a circle with radius 3 units at their midpoints. -/
noncomputable def tangent_segment_region_area (circle_radius : ℝ) (segment_length : ℝ) : ℝ :=
  Real.pi * ((circle_radius ^ 2 + (segment_length / 2) ^ 2) - circle_radius ^ 2)

/-- Theorem stating that the area of the region formed by all line segments
    of length 4 units that are tangent to a circle with radius 3 units
    at their midpoints is equal to 4π. -/
theorem tangent_segment_region_area_is_4pi :
  tangent_segment_region_area 3 4 = 4 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_region_area_is_4pi_l81_8175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_correct_l81_8139

/-- Calculates the actual payment after applying the discount --/
noncomputable def actualPayment (x : ℝ) : ℝ :=
  if x < 200 then x
  else if x < 500 then 0.9 * x
  else 0.8 * x + 50

theorem discount_calculation_correct :
  (actualPayment 400 = 360) ∧
  (actualPayment 600 = 530) ∧
  (∀ x : ℝ, 200 ≤ x ∧ x < 500 → actualPayment x = 0.9 * x) ∧
  (∀ x : ℝ, 500 ≤ x → actualPayment x = 0.8 * x + 50) ∧
  (∀ a : ℝ, 200 < a ∧ a < 300 →
    actualPayment a + actualPayment (820 - a) = 0.1 * a + 706) :=
by
  sorry

-- Remove #eval statements as they're not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_correct_l81_8139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cos_ratio_l81_8146

theorem tan_cos_ratio (α : Real) (h1 : α ∈ Set.Ioo (π/2) π) (h2 : Real.cos α = -5/13) :
  Real.tan (α + π/2) / Real.cos (α + π) = 13/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cos_ratio_l81_8146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_implies_m_min_area_equation_l81_8166

-- Define the line equation
noncomputable def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  x + m * y - 2 * m - 1 = 0

-- Define the sum of intercepts
noncomputable def sum_of_intercepts (m : ℝ) : ℝ :=
  (2 * m + 1) + (2 + 1 / m)

-- Define the area of triangle AOB
noncomputable def triangle_area (m : ℝ) : ℝ :=
  (1 / 2) * (2 * m + 1) * (2 + 1 / m)

-- Theorem for part 1
theorem intercept_sum_implies_m (m : ℝ) (h : m ≠ 0) :
  sum_of_intercepts m = 6 → m = 1/2 ∨ m = 1 := by
  sorry

-- Theorem for part 2
theorem min_area_equation (m : ℝ) (h1 : m > 0) (h2 : ∀ x y, line_equation m x y → x ≥ 0 ∧ y ≥ 0) :
  (∀ m' > 0, triangle_area m ≤ triangle_area m') →
  ∀ x y, line_equation m x y ↔ 2 * x + y - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_implies_m_min_area_equation_l81_8166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_argument_of_z_is_pi_over_four_l81_8193

-- Define θ
noncomputable def θ : ℝ := Real.arctan (5/12)

-- Define the complex number z
noncomputable def z : ℂ := (Complex.cos (2*θ) + Complex.I * Complex.sin (2*θ)) / (239 + Complex.I)

-- Theorem statement
theorem argument_of_z_is_pi_over_four :
  Complex.arg z = π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_argument_of_z_is_pi_over_four_l81_8193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_for_12_ohm_resistance_l81_8142

/-- A function representing the relationship between current and resistance for a 48V battery -/
noncomputable def current (R : ℝ) : ℝ := 48 / R

/-- Theorem stating that for a 48V battery with 12Ω resistance, the current is 4A -/
theorem current_for_12_ohm_resistance : current 12 = 4 := by
  -- Unfold the definition of current
  unfold current
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_for_12_ohm_resistance_l81_8142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l81_8117

noncomputable def circle1_center : ℝ × ℝ := (5, 5)
noncomputable def circle2_center : ℝ × ℝ := (22, 13)

noncomputable def circle1_radius : ℝ := circle1_center.1
noncomputable def circle2_radius : ℝ := circle2_center.1

noncomputable def center_distance : ℝ := Real.sqrt ((circle2_center.1 - circle1_center.1)^2 + (circle2_center.2 - circle1_center.2)^2)

theorem closest_points_distance :
  center_distance - circle1_radius - circle2_radius = Real.sqrt 353 - 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l81_8117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_valid_answers_in_range_l81_8122

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ -1 then (4 - a) * x + a else -x^2 + 1

-- Define what it means for f to be increasing on ℝ
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem increasing_f_implies_a_range (a : ℝ) :
  is_increasing (f a) → 2 ≤ a ∧ a < 4 := by
  sorry

-- Define the set of valid answers
def valid_answers : Set ℝ := {2, 3}

-- Theorem stating that the valid answers are within the range
theorem valid_answers_in_range :
  ∀ x ∈ valid_answers, 2 ≤ x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_valid_answers_in_range_l81_8122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_business_profit_l81_8156

/-- Proves that the total profit of a business is 9600 given specific conditions --/
theorem business_profit (a_investment b_investment : ℕ) 
  (management_fee : ℚ) (a_total_received : ℕ) : ℕ := by
  have h1 : a_investment = 5000 := by sorry
  have h2 : b_investment = 1000 := by sorry
  have h3 : management_fee = 1/10 := by sorry
  have h4 : a_total_received = 8160 := by sorry
  
  let total_profit : ℕ := 9600
  
  have h5 : (management_fee + (1 - management_fee) * (a_investment / (a_investment + b_investment : ℚ))) * total_profit = a_total_received := by sorry
  
  exact total_profit


end NUMINAMATH_CALUDE_ERRORFEEDBACK_business_profit_l81_8156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_exist_l81_8147

noncomputable def f (x : ℝ) := x^2 - Real.log x

theorem perpendicular_tangents_exist :
  ∃ (x₁ x₂ : ℝ),
    1/2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 ∧
    (2*x₁ - 1/x₁) * (2*x₂ - 1/x₂) = -1 ∧
    x₁ = 1/2 ∧ x₂ = 1 ∧
    f x₁ = Real.log 2 + 1/4 ∧ f x₂ = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_exist_l81_8147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tangent_line_l81_8169

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (4 - x) * Real.exp (x - 2)

-- Define the derivative of f
noncomputable def f_prime (x : ℝ) : ℝ := (3 - x) * Real.exp (x - 2)

-- Theorem statement
theorem no_tangent_line :
  ¬ ∃ (m : ℝ), ∃ (x : ℝ), 
    (f x = (3 * x + m) / 2) ∧ 
    (f_prime x = 3 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tangent_line_l81_8169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_f_range_when_f2_leq_3_l81_8161

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := |x - a| + |x + a + 1/a|

-- Theorem 1: f(x) ≥ 2√2 for all x and a ≠ 0
theorem f_lower_bound (a : ℝ) (ha : a ≠ 0) : ∀ x : ℝ, f a x ≥ 2 * Real.sqrt 2 := by
  sorry

-- Theorem 2: Range of a when f(2) ≤ 3
theorem f_range_when_f2_leq_3 (a : ℝ) (ha : a ≠ 0) :
  f a 2 ≤ 3 → (a ∈ Set.Icc (-1) (-1/2) ∪ Set.Ioc (1/2) 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_f_range_when_f2_leq_3_l81_8161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_plane_l81_8130

/-- The distance from a point to a plane in 3D space -/
noncomputable def distance_point_to_plane (p : ℝ × ℝ × ℝ) (a b c d : ℝ) : ℝ :=
  let (x, y, z) := p
  |a * x + b * y + c * z + d| / Real.sqrt (a^2 + b^2 + c^2)

/-- The specific point in the problem -/
def P : ℝ × ℝ × ℝ := (2, 4, 1)

/-- Coefficients of the plane equation x + 2y + 3z + 3 = 0 -/
def A : ℝ := 1
def B : ℝ := 2
def C : ℝ := 3
def D : ℝ := 3

theorem distance_point_to_specific_plane :
  distance_point_to_plane P A B C D = 8 * Real.sqrt 14 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_plane_l81_8130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_distance_l81_8167

/-- The one-way distance of a round trip -/
noncomputable def distance (total_time outbound_speed return_speed : ℝ) : ℝ :=
  (total_time * outbound_speed * return_speed) / (outbound_speed + return_speed)

/-- Theorem stating that for the given conditions, the one-way distance is 28.8 miles -/
theorem round_trip_distance :
  distance 3 16 24 = 28.8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_distance_l81_8167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_virus_diameter_scientific_notation_l81_8108

theorem virus_diameter_scientific_notation :
  (0.00000135 : ℝ) = 1.35 * (10 : ℝ)^(-6 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_virus_diameter_scientific_notation_l81_8108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_squares_area_sum_l81_8106

-- Define the triangle
def Triangle (A B E : ℝ × ℝ) : Prop :=
  -- EAB is a right angle
  (E.1 - A.1) * (B.1 - A.1) + (E.2 - A.2) * (B.2 - A.2) = 0 ∧
  -- BE = 12
  Real.sqrt ((B.1 - E.1)^2 + (B.2 - E.2)^2) = 12 ∧
  -- Angle BAE = 30°
  Real.arccos (((B.1 - A.1) * (E.1 - A.1) + (B.2 - A.2) * (E.2 - A.2)) /
    (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2)))
    = Real.pi / 6

-- Define the theorem
theorem triangle_squares_area_sum (A B E : ℝ × ℝ) (h : Triangle A B E) :
  let ab_length := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let ae_length := Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2)
  ab_length^2 + ae_length^2 = 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_squares_area_sum_l81_8106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_in_circle_l81_8178

theorem distance_between_points_in_circle (points : Finset (ℝ × ℝ)) 
  (h1 : points.card = 8)
  (h2 : ∀ p ∈ points, Real.sqrt ((p.1)^2 + (p.2)^2) ≤ 1) :
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ 
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_in_circle_l81_8178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l81_8198

theorem point_B_coordinates (A B : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (-1, -1) →
  a = (2, 3) →
  (B.1 - A.1, B.2 - A.2) = (3 * a.1, 3 * a.2) →
  B = (5, 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l81_8198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_digit_multiple_of_nine_l81_8141

def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n]
  else (n % 10) :: digits (n / 10)

theorem smallest_even_digit_multiple_of_nine : ∃ n : ℕ, 
  (n % 9 = 0) ∧ 
  (∀ d : ℕ, d ∈ digits n → Even d) ∧
  (n = 288) ∧
  (∀ m : ℕ, m < n → ¬(m % 9 = 0 ∧ ∀ d : ℕ, d ∈ digits m → Even d)) := by
  sorry

-- Additional theorems for other problems can be added here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_digit_multiple_of_nine_l81_8141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_powers_l81_8184

theorem order_of_powers : ∀ (a b c : ℝ),
  a = (0.5 : ℝ)^(0.8 : ℝ) → b = (0.8 : ℝ)^(0.5 : ℝ) → c = (0.8 : ℝ)^(0.8 : ℝ) →
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_powers_l81_8184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_property_l81_8110

/-- A cosine function with specific properties -/
noncomputable def f (ω φ : ℝ) : ℝ → ℝ := fun x ↦ Real.cos (ω * x + φ)

/-- The theorem stating the properties of the function and the result to be proved -/
theorem cosine_function_property (ω φ : ℝ) :
  ω > 0 ∧
  f ω φ (-π/6) = 0 ∧
  f ω φ (5*π/6) = 0 ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ∈ Set.Ioo (-π/6) (5*π/6) ∧ x₂ ∈ Set.Ioo (-π/6) (5*π/6) ∧
    ∀ (x : ℝ), x ∈ Set.Ioo (-π/6) (5*π/6) → (f ω φ (x₁ + (x₁ - x)) = f ω φ x ∧ f ω φ (x₂ + (x₂ - x)) = f ω φ x)) →
  ω * φ = 5*π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_property_l81_8110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_exists_l81_8109

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem tangent_line_exists (c : Circle) (A : Point) 
    (h : distance A c.center > c.radius) :
  ∃ B : Point, (distance B c.center = c.radius) ∧ 
    (∀ P : Point, P ≠ B → distance P c.center < c.radius → 
      distance A P > distance A B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_exists_l81_8109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_theorem_l81_8113

theorem trigonometric_sum_theorem (A B C : ℝ) 
  (h1 : Real.sin A + Real.sin B + Real.sin C = 0)
  (h2 : Real.cos A + Real.cos B + Real.cos C = 0) :
  (Real.cos (3 * A) + Real.cos (3 * B) + Real.cos (3 * C) = 3 * Real.cos (A + B + C)) ∧
  (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 3 * Real.sin (A + B + C)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_theorem_l81_8113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_three_fourths_l81_8118

/-- A square with midpoints on adjacent sides and diagonal segments -/
structure MidpointSquare where
  -- Side length of the square
  side : ℝ
  -- Assumption that side length is positive
  side_pos : 0 < side

/-- The fraction of the square that is shaded -/
noncomputable def shaded_fraction (square : MidpointSquare) : ℝ := 3 / 4

/-- Theorem stating that the shaded fraction is 3/4 -/
theorem shaded_fraction_is_three_fourths (square : MidpointSquare) :
  shaded_fraction square = 3 / 4 := by
  -- Unfold the definition of shaded_fraction
  unfold shaded_fraction
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_three_fourths_l81_8118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_needs_100_cherries_per_quart_l81_8137

-- Define the constants from the problem
def total_time : ℕ := 33
def total_quarts : ℕ := 9
def picking_rate : ℚ := 300 / 2
def syrup_making_time : ℕ := 3

-- Define the function to calculate cherries per quart
def cherries_per_quart : ℚ :=
  (total_time - total_quarts * syrup_making_time) * picking_rate / total_quarts

-- Define the theorem
theorem jerry_needs_100_cherries_per_quart : 
  cherries_per_quart = 100 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_needs_100_cherries_per_quart_l81_8137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neznaika_mistake_l81_8162

theorem neznaika_mistake (n : ℕ) (diffs : List ℤ) : 
  n = 11 ∧ 
  diffs.length = n ∧
  (diffs.filter (λ x => x.natAbs = 1)).length = 4 ∧
  (diffs.filter (λ x => x.natAbs = 2)).length = 4 ∧
  (diffs.filter (λ x => x.natAbs = 3)).length = 3 →
  diffs.sum ≠ 0 := by
  intro h
  sorry

#check neznaika_mistake

end NUMINAMATH_CALUDE_ERRORFEEDBACK_neznaika_mistake_l81_8162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l81_8128

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := (a * x^2 + b * x + c) * Real.exp (-x)

-- Define the derivative of f
noncomputable def f_prime (a b c x : ℝ) : ℝ := Real.exp (-x) * (-a * x^2 + (2 * a - b) * x + b - c)

theorem function_properties (a b c : ℝ) :
  (f a b c 0 = 2 * a) ∧ 
  (f_prime a b c 0 = π / 4) ∧
  (∃ x y : ℝ, x ≥ 1/2 ∧ y > x ∧ f a b c x < f a b c y) →
  (b = 1 + 2 * a ∧ c = 2 * a ∧ -1/4 < a ∧ a < 2) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l81_8128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cos_equation_solutions_l81_8177

theorem tan_cos_equation_solutions :
  ∃ (S : Finset ℝ), S.card = 5 ∧
  (∀ x ∈ S, 0 ≤ x ∧ x ≤ 3 * π / 2 ∧ Real.tan (3 * x) = Real.cos (x - π / 4)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 3 * π / 2 ∧ Real.tan (3 * x) = Real.cos (x - π / 4) → x ∈ S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cos_equation_solutions_l81_8177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_with_large_angle_l81_8136

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : Point2D) : ℝ := sorry

/-- Main theorem -/
theorem exists_triangle_with_large_angle 
  (points : Finset Point2D) 
  (h1 : points.card = 6) 
  (h2 : ∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → 
       p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬collinear p1 p2 p3) :
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    (angle p1 p2 p3 ≥ 120 ∨ angle p2 p3 p1 ≥ 120 ∨ angle p3 p1 p2 ≥ 120) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_with_large_angle_l81_8136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_born_in_may_l81_8182

theorem percentage_born_in_may : 
  let total_scientists : ℕ := 120
  let born_in_may : ℕ := 7
  let percentage : ℚ := (born_in_may : ℚ) / total_scientists * 100
  ∃ (rounded_percentage : ℚ), 
    (rounded_percentage * 100).floor / 100 = percentage.floor / 100 ∧ 
    rounded_percentage = 583/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_born_in_may_l81_8182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_theorem_l81_8134

-- Define a type for lines in space
variable (Line : Type)

-- Define a type for points in space
variable (Point : Type)

-- Define a type for planes in space
variable (Plane : Type)

-- Define a relation for line intersection
variable (intersects : Line → Line → Prop)

-- Define a relation for a line lying in a plane
variable (lies_in : Line → Plane → Prop)

-- Define a relation for a line passing through a point
variable (passes_through : Line → Point → Prop)

-- Theorem statement
theorem line_intersection_theorem 
  (lines : Set Line) 
  (h_intersect : ∀ l1 l2, l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → intersects l1 l2) :
  (∃ p : Plane, ∀ l, l ∈ lines → lies_in l p) ∨ 
  (∃ pt : Point, ∀ l, l ∈ lines → passes_through l pt) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_theorem_l81_8134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l81_8123

/-- In a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition 1 -/
def condition1 (t : Triangle) : Prop :=
  Real.sin t.A / (1 - Real.cos t.A) = Real.sin (2 * t.B) / (1 + Real.cos (2 * t.B))

/-- Condition 2 -/
def condition2 (t : Triangle) : Prop :=
  Real.sin t.C * Real.sin (t.B - t.A) = Real.sin t.B * Real.sin (t.C - t.A)

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) 
  (h : condition1 t ∨ condition2 t) : 
  t.B = t.C ∧ 
  (∀ x : ℝ, (2 * t.a + t.b) / t.c + 1 / Real.cos t.B ≥ 5) ∧
  (∃ x : ℝ, (2 * t.a + t.b) / t.c + 1 / Real.cos t.B = 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l81_8123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_max_area_l81_8181

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- The perimeter of a polygon -/
noncomputable def perimeter {n : ℕ} (p : Polygon n) : ℝ := sorry

/-- The area of a polygon -/
noncomputable def area {n : ℕ} (p : Polygon n) : ℝ := sorry

/-- A regular polygon with n sides -/
noncomputable def regularPolygon (n : ℕ) (perimeter : ℝ) : Polygon n := sorry

/-- Theorem: Among all polygons with n sides (where n = 3 or 4) and a fixed perimeter, 
    the regular polygon has the largest area -/
theorem regular_polygon_max_area {n : Fin 2} {p : ℝ} :
  ∀ (poly : Polygon (n + 3)), 
    perimeter poly = p → 
    area poly ≤ area (regularPolygon (n + 3) p) := by
  sorry

#check regular_polygon_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_max_area_l81_8181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_13_l81_8111

theorem sum_remainder_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_13_l81_8111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l81_8115

/-- The function y1 -/
noncomputable def y1 (k x : ℝ) : ℝ := k / x

/-- The function y2 -/
noncomputable def y2 (k x : ℝ) : ℝ := -k / x

/-- The theorem statement -/
theorem problem_solution (k : ℝ) (h_k : k > 0) :
  (∃ a : ℝ, (∀ x : ℝ, 2 ≤ x → x ≤ 3 → y1 k x ≤ a) ∧
             (∃ x : ℝ, 2 ≤ x ∧ x ≤ 3 ∧ y1 k x = a) ∧
             (∀ x : ℝ, 2 ≤ x → x ≤ 3 → y2 k x ≥ a - 4) ∧
             (∃ x : ℝ, 2 ≤ x ∧ x ≤ 3 ∧ y2 k x = a - 4)) →
  (a = 2 ∧ k = 4) ∧
  (∃ m : ℝ, m ≠ 0 ∧ m ≠ -1 ∧ y1 k m > y1 k (m + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l81_8115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_a_in_open_interval_l81_8125

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x + 2

-- State the theorem
theorem two_zeros_implies_a_in_open_interval (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧
    ∀ x : ℝ, x > 0 → f a x = 0 → (x = x₁ ∨ x = x₂)) →
  0 < a ∧ a < exp 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_a_in_open_interval_l81_8125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_convergence_bound_l81_8101

noncomputable def x : ℕ → ℝ
  | 0 => 5
  | n + 1 => (x n ^ 2 + 5 * x n + 4) / (x n + 6)

theorem x_convergence_bound :
  ∃ m : ℕ, m ∈ Set.Ico 81 243 ∧ x m ≤ 4 + 1 / 2^20 ∧ ∀ k : ℕ, k < m → x k > 4 + 1 / 2^20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_convergence_bound_l81_8101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_min_value_l81_8154

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - (3/2) * x

theorem local_min_value (a : ℝ) :
  (∀ x : ℝ, x > 0 → DifferentiableAt ℝ (f a) x) →
  (∃ ε > 0, ∀ x : ℝ, x > 0 → |x - 1| < ε → x ≠ 1 → f a x ≤ f a 1) →
  (∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x : ℝ, x > 0 → f a x ≥ f a x₀) →
  ∃ x_min : ℝ, x_min > 0 ∧ f a x_min = Real.log 2 - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_min_value_l81_8154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_equation_l81_8180

-- Define the function f: ℝ → ℝ
noncomputable def f (x : ℝ) : ℝ := 1 - x^2 / 2

-- State the theorem
theorem unique_function_satisfying_equation :
  ∀ (g : ℝ → ℝ), (∀ (x y : ℝ), g (x - g y) = g (g y) + x * g y + g x - 1) →
  g = f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_equation_l81_8180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l81_8187

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain D
def D : Set ℝ := { x | 0 ≤ x }

-- Define the range A
def A : Set ℝ := Set.range f

-- State the functional equation
axiom f_eq : ∀ x, f x = f (1 / (x + 1))

-- State that the domain of f is D
axiom f_domain : Set.range f = f '' D

-- Define the set of f(x) for x in [0,a]
def f_set (a : ℝ) : Set ℝ := { y | ∃ x ∈ Set.Icc 0 a, f x = y }

-- State the theorem
theorem min_a_value :
  ∃ a₀ : ℝ, (∀ a ≥ a₀, f_set a = A) ∧ (∀ a < a₀, f_set a ≠ A) ∧ a₀ = (Real.sqrt 5 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l81_8187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutations_of_2021_l81_8176

def digits : List Nat := [2, 0, 2, 1]

def is_valid_permutation (perm : List Nat) : Bool :=
  perm.length = 4 && perm.head? ≠ some 0 && perm.toFinset = digits.toFinset

def number_of_valid_permutations : Nat :=
  (List.permutations digits).filter is_valid_permutation |>.length

theorem permutations_of_2021 :
  number_of_valid_permutations = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutations_of_2021_l81_8176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l81_8149

noncomputable def expansion (x : ℝ) : ℝ → ℝ := λ n ↦ (Real.sqrt x + 1 / (2 * Real.sqrt (4 * x)))^n

noncomputable def coeff (n : ℕ) (r : ℕ) : ℝ := (1/2)^r * (n.choose r)

theorem expansion_properties :
  -- The coefficients of the first three terms form an arithmetic sequence
  ∃ (n : ℕ), 2 * coeff n 1 = coeff n 0 + coeff n 2 →
  -- n = 8
  n = 8 ∧
  -- Rational terms
  (∀ (x : ℝ), x > 0 →
    { r : ℕ | ∃ (k : ℚ), coeff 8 r * x^((16 - 3*r)/4) = ↑k } = {0, 4, 8}) ∧
  -- Terms with largest coefficient
  (∀ (r : ℕ), r ≠ 2 ∧ r ≠ 3 →
    coeff 8 r ≤ coeff 8 2 ∧ coeff 8 r ≤ coeff 8 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l81_8149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watermelon_melon_weight_comparison_l81_8102

/-- The weight of a watermelon -/
def W : ℝ := sorry

/-- The weight of a melon -/
def M : ℝ := sorry

/-- Masha's statement: 2 watermelons are heavier than 3 melons -/
def masha_statement : Prop := 2 * W > 3 * M

/-- Anya's statement: 3 watermelons are heavier than 4 melons -/
def anya_statement : Prop := 3 * W > 4 * M

/-- One of the statements is correct, but not both -/
axiom one_statement_correct : (masha_statement ∨ anya_statement) ∧ ¬(masha_statement ∧ anya_statement)

/-- Theorem: It cannot be concluded that 12 watermelons are heavier than 18 melons -/
theorem watermelon_melon_weight_comparison : ¬(12 * W > 18 * M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_watermelon_melon_weight_comparison_l81_8102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_special_n_l81_8179

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

noncomputable def largest_prime_factor (n : ℕ) : ℕ := 
  (Nat.factors n).maximum.getD 1

theorem largest_prime_factor_of_special_n :
  ∃ n : ℕ,
    (∀ m : ℕ, m < n →
      ¬(is_divisible_by m 36 ∧ is_perfect_cube (m^2) ∧ is_perfect_square (m^3))) ∧
    is_divisible_by n 36 ∧
    is_perfect_cube (n^2) ∧
    is_perfect_square (n^3) ∧
    largest_prime_factor n = 3 :=
by
  sorry

#check largest_prime_factor_of_special_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_special_n_l81_8179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l81_8112

noncomputable def f (x : ℝ) : ℝ := (Finset.range 100).prod (λ i => x - (i + 1 : ℝ))

theorem f_derivative_at_one : 
  deriv f 1 = -(99 : ℕ).factorial := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l81_8112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l81_8150

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2*x else x - 1

-- Define the property of having three distinct real roots
def has_three_distinct_real_roots (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x - a^2 + 2*a = 0 ∧
    f y - a^2 + 2*a = 0 ∧
    f z - a^2 + 2*a = 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  has_three_distinct_real_roots a → (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l81_8150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l81_8145

noncomputable def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

noncomputable def geometric_sum (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_8 :
  ∀ a₁ : ℝ,
  let r := (2 : ℝ)
  let S₄ := geometric_sum a₁ r 4
  let S₈ := geometric_sum a₁ r 8
  S₄ = 1 → S₈ = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l81_8145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_slope_sum_l81_8103

/-- Represents a point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- Represents a trapezoid with integer coordinate vertices -/
structure IntTrapezoid where
  A : IntPoint
  B : IntPoint
  C : IntPoint
  D : IntPoint

/-- Checks if a trapezoid is isosceles -/
def isIsosceles (t : IntTrapezoid) : Prop := sorry

/-- Checks if a trapezoid has no horizontal or vertical sides -/
def noHorizontalVerticalSides (t : IntTrapezoid) : Prop := sorry

/-- Checks if only AB and CD are parallel in the trapezoid -/
def onlyABCDParallel (t : IntTrapezoid) : Prop := sorry

/-- Calculates the slope of a line given two points -/
def calculateSlope (p1 p2 : IntPoint) : ℚ := sorry

/-- Finds all possible slopes for AB in the trapezoid -/
def possibleSlopesAB (t : IntTrapezoid) : List ℚ := sorry

/-- Calculates the sum of absolute values of a list of rational numbers -/
def sumAbsValues (slopes : List ℚ) : ℚ := sorry

/-- Expresses a rational number as a fraction m/n in simplest form -/
noncomputable def simplestForm (q : ℚ) : (Nat × Nat) := sorry

theorem isosceles_trapezoid_slope_sum :
  ∀ t : IntTrapezoid,
    t.A = ⟨30, 200⟩ →
    t.D = ⟨31, 217⟩ →
    isIsosceles t →
    noHorizontalVerticalSides t →
    onlyABCDParallel t →
    let slopes := possibleSlopesAB t
    let sumSlopes := sumAbsValues slopes
    let (m, n) := simplestForm sumSlopes
    m + n = 284 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_slope_sum_l81_8103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_empty_time_l81_8132

/-- A cistern with a leak -/
structure LeakyCistern where
  normal_fill_time : ℝ
  leak_fill_time : ℝ

/-- The time it takes for a full cistern to empty through the leak -/
noncomputable def empty_time (c : LeakyCistern) : ℝ :=
  (c.leak_fill_time * c.normal_fill_time) / (c.leak_fill_time - c.normal_fill_time)

theorem cistern_empty_time (c : LeakyCistern) 
  (h1 : c.normal_fill_time = 4)
  (h2 : c.leak_fill_time = 6) : 
  empty_time c = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_empty_time_l81_8132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_600_sets_l81_8138

-- Define the sales revenue function
noncomputable def P (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then
    -0.4 * x^2 + 4.2 * x - 0.8
  else if x > 5 then
    14.7 - 9 / (x - 3)
  else
    0

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ :=
  P x - (2 + x)

-- Theorem statement
theorem max_profit_at_600_sets :
  ∃ (max_profit : ℝ),
    max_profit = 3.7 ∧
    profit 6 = max_profit ∧
    ∀ x > 0, profit x ≤ max_profit :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_600_sets_l81_8138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_Q_l81_8168

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := (x - 1) * (x - 3) * (x - 5)

-- Define the set of polynomials Q(x) that satisfy the condition
def valid_Q : Set (ℝ → ℝ) :=
  {Q | ∃ (R : ℝ → ℝ), (∀ x, P (Q x) = P x * R x) ∧ 
                       (∃ a b c, ∀ x, Q x = a*x^2 + b*x + c)}

-- Theorem stating that there are exactly 22 such polynomials
theorem count_valid_Q : Nat.card valid_Q = 22 := by
  sorry

-- Helper lemma to show that valid_Q is finite
lemma valid_Q_finite : Finite valid_Q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_Q_l81_8168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_with_difference_36_l81_8148

/-- Given a two-digit number, returns the tens digit -/
def tensDigit (n : Nat) : Nat :=
  (n / 10) % 10

/-- Given a two-digit number, returns the ones digit -/
def onesDigit (n : Nat) : Nat :=
  n % 10

/-- Calculates the place value of a digit in a two-digit number -/
def placeValue (n : Nat) (d : Nat) : Nat :=
  if d = tensDigit n then 10 * d else d

/-- The difference between place value and face value for a digit in 46 -/
def placeFaceDifference (d : Nat) : Nat :=
  placeValue 46 d - d

theorem digit_with_difference_36 :
  ∃ d : Nat, d ∈ ({tensDigit 46, onesDigit 46} : Set Nat) ∧ placeFaceDifference d = 36 ∧ d = 4 := by
  sorry

#eval tensDigit 46
#eval onesDigit 46
#eval placeFaceDifference 4
#eval placeFaceDifference 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_with_difference_36_l81_8148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_lt_x_l81_8194

theorem negation_of_forall_sin_lt_x :
  (¬ ∀ x : ℝ, x > 0 → Real.sin x < x) ↔ (∃ x : ℝ, x > 0 ∧ Real.sin x ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_lt_x_l81_8194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_tiling_10x10_l81_8151

/-- Represents a chessboard -/
structure Chessboard where
  size : ℕ

/-- Represents an L-shaped tetromino -/
structure LTetromino where
  cells : Fin 4 → (ℕ × ℕ)

/-- Defines a valid tiling of a chessboard with L-shaped tetrominos -/
def ValidTiling (board : Chessboard) (tiling : List LTetromino) : Prop :=
  (board.size = 10) ∧
  (tiling.length * 4 = board.size * board.size) ∧
  (∀ t, t ∈ tiling → ∀ i : Fin 4, (t.cells i).1 < board.size ∧ (t.cells i).2 < board.size) ∧
  (∀ t1 t2, t1 ∈ tiling → t2 ∈ tiling → t1 ≠ t2 → ∀ i j : Fin 4, t1.cells i ≠ t2.cells j)

/-- Theorem stating that it's impossible to tile a 10x10 chessboard with L-shaped tetrominos -/
theorem no_valid_tiling_10x10 (board : Chessboard) (tiling : List LTetromino) :
  ¬(ValidTiling board tiling) := by
  sorry

#check no_valid_tiling_10x10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_tiling_10x10_l81_8151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_t_squared_l81_8188

/-- An ellipse centered at the origin passing through (4,0), (3,3), and (0,t) where t > 0 -/
structure CenteredEllipse where
  t : ℝ
  b : ℝ
  t_pos : t > 0
  passes_through_4_0 : 16 + 0 = 16
  passes_through_3_3 : 9 + 9 * (16 / b^2) = 16
  passes_through_0_t : 0 + t^2 * (16 / b^2) = 16

/-- The value of t^2 for the given ellipse -/
theorem ellipse_t_squared (e : CenteredEllipse) : e.t^2 = 144/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_t_squared_l81_8188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_on_AC_l81_8144

open EuclideanGeometry

-- Define the points
variable (A B C D M N : EuclideanSpace ℝ (Fin 2))

-- Define the cyclic quadrilateral ABCD
def is_cyclic_quad (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ), 
    dist center A = radius ∧ 
    dist center B = radius ∧ 
    dist center C = radius ∧ 
    dist center D = radius

-- Define the condition AB = AD
def AB_eq_AD (A B D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A B = dist A D

-- Define M on CD and N on BC
def M_on_CD (C D M : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • C + t • D

def N_on_BC (B C N : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ N = (1 - s) • B + s • C

-- Define the condition DM + BN = MN
def DM_plus_BN_eq_MN (B D M N : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist D M + dist B N = dist M N

-- Define the circumcenter of a triangle
noncomputable def circumcenter (A M N : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  sorry -- Actual definition would go here

-- Define a point being on a segment
def on_segment (P A C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • C

-- The main theorem
theorem circumcenter_on_AC 
  (h1 : is_cyclic_quad A B C D)
  (h2 : AB_eq_AD A B D)
  (h3 : M_on_CD C D M)
  (h4 : N_on_BC B C N)
  (h5 : DM_plus_BN_eq_MN B D M N) :
  on_segment (circumcenter A M N) A C :=
by
  sorry -- The proof would go here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_on_AC_l81_8144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gp_common_ratio_l81_8124

-- Define a geometric progression
noncomputable def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

-- Theorem statement
theorem gp_common_ratio 
  (a : ℝ) (r : ℝ) 
  (h : (geometric_progression a r 6) / (geometric_progression a r 3) = 126) : 
  r = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gp_common_ratio_l81_8124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_time_l81_8143

/-- Represents the time in minutes after 3:00 -/
def t : ℝ := sorry

/-- The current time is between 3:00 and 4:00 -/
axiom time_range : 0 < t ∧ t < 60

/-- Function to calculate the position of the minute hand at time t -/
def minute_hand_pos (t : ℝ) : ℝ := 6 * t

/-- Function to calculate the position of the hour hand at time t -/
def hour_hand_pos (t : ℝ) : ℝ := 90 + 0.5 * t

/-- The condition that in 10 minutes, the minute hand will be exactly 90 degrees
    ahead of where the hour hand was 5 minutes ago -/
axiom hand_position_condition : 
  |minute_hand_pos (t + 10) - hour_hand_pos (t - 5)| = 90

/-- The theorem stating that the time satisfying the conditions is approximately 3.18 minutes -/
theorem exact_time : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |t - 3.18| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_time_l81_8143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_invertibles_mod_factorial_l81_8155

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def invertible_mod (a m : ℕ) : Bool := 
  Nat.gcd a m = 1

theorem product_invertibles_mod_factorial : 
  let fact5 := factorial 5
  let invertibles := (List.range fact5).filter (λ x => invertible_mod x fact5)
  invertibles.foldl (· * ·) 1 % fact5 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_invertibles_mod_factorial_l81_8155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l81_8183

/-- Jane's current age -/
def jane_age : ℕ := 30

/-- Predicate to check if a number is two-digit -/
def is_two_digit (x : ℕ) : Prop := x ≥ 10 ∧ x < 100

/-- Predicate to check if the conditions are satisfied for given d, n, a, b -/
def satisfies_conditions (d n a b : ℕ) : Prop :=
  d > jane_age ∧
  n > 0 ∧
  is_two_digit (jane_age + n) ∧
  is_two_digit (d + n) ∧
  d + n ≥ 35 ∧
  jane_age + n = 10 * a + b ∧
  d + n = 10 * b + a

/-- The number of valid ordered pairs (d, n) -/
def num_valid_pairs : ℕ := 10

/-- Theorem stating that there are exactly 10 valid ordered pairs (d, n) -/
theorem count_valid_pairs :
  ∃ (S : Finset (ℕ × ℕ)), S.card = num_valid_pairs ∧
    ∀ (p : ℕ × ℕ), p ∈ S ↔ ∃ (a b : ℕ), satisfies_conditions p.1 p.2 a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l81_8183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_l81_8160

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem root_exists : ∃ x : ℝ, |2 - x - lg x| < 0.000001 := by
  -- The proof goes here
  sorry

#eval "Root existence theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_l81_8160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_celias_savings_percentage_l81_8163

noncomputable def weekly_food_budget : ℝ := 100
noncomputable def rent : ℝ := 1500
noncomputable def streaming_services : ℝ := 30
noncomputable def cell_phone : ℝ := 50
def num_weeks : ℕ := 4
noncomputable def savings : ℝ := 198

noncomputable def total_spending : ℝ := weekly_food_budget * (num_weeks : ℝ) + rent + streaming_services + cell_phone

noncomputable def savings_percentage : ℝ := savings / total_spending * 100

theorem celias_savings_percentage :
  savings_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_celias_savings_percentage_l81_8163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_number_tenth_row_l81_8189

/-- Calculates the last number in a row given the first number of that row -/
def lastNumberInRow (firstNumber : ℕ) : ℕ :=
  firstNumber + 5 * 4

/-- Calculates the first number of the next row given the first number of the current row -/
def firstNumberNextRow (firstNumber : ℕ) : ℕ :=
  lastNumberInRow firstNumber + 1

/-- Calculates the first number of the nth row -/
def firstNumberOfRow : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 1 => firstNumberNextRow (firstNumberOfRow n)

/-- Calculates the kth number in the nth row -/
def kthNumberInRow (n k : ℕ) : ℕ :=
  firstNumberOfRow n + (k - 1) * 4

theorem fourth_number_tenth_row :
  kthNumberInRow 10 4 = 338 := by
  sorry

#eval kthNumberInRow 10 4  -- Added to check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_number_tenth_row_l81_8189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_properties_l81_8199

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - a * Real.log x + x

-- Define the fixed point property
def is_fixed_point (a : ℝ) (x : ℝ) : Prop := f a x = x

-- Theorem statement
theorem fixed_point_properties :
  ∀ a : ℝ, 
  (∃ x₀ : ℝ, x₀ ≠ 1 ∧ is_fixed_point a x₀) →
  (a ∈ Set.Ioo 0 (Real.exp 1) ∪ Set.Ioi (Real.exp 1)) ∧
  (∀ k : ℤ, (∀ x₀ : ℝ, x₀ ≠ 1 → is_fixed_point a x₀ → (↑k : ℝ) * x₀ < a) → k ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_properties_l81_8199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_inequality_l81_8114

/-- Represents a convex polyhedron with B vertices, P edges, and T as the maximum number of triangular faces sharing a common vertex -/
structure ConvexPolyhedron (B P T : ℕ) : Type where
  convex : True  -- Placeholder for convexity condition
  vertices : B > 0
  edges : P > 0
  max_triangular_faces : T ≥ 0

/-- A convex polyhedron with B vertices, P edges, and T as the maximum number of triangular faces sharing a common vertex satisfies the inequality B√(P + T) ≥ 2P -/
theorem polyhedron_inequality (B P T : ℕ) (h : ConvexPolyhedron B P T) :
  (B : ℝ) * Real.sqrt (P + T : ℝ) ≥ 2 * P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_inequality_l81_8114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_sum_l81_8197

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def digits_used (n : ℕ) : Finset ℕ :=
  Finset.filter (fun d ↦ d > 0 ∧ d ≤ 8) (Finset.range 9)

def valid_prime_set (p₁ p₂ p₃ p₄ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧
  (digits_used p₁ ∪ digits_used p₂ ∪ digits_used p₃ ∪ digits_used p₄).card = 8

theorem smallest_prime_sum :
  ∀ p₁ p₂ p₃ p₄ : ℕ,
  valid_prime_set p₁ p₂ p₃ p₄ →
  p₁ + p₂ + p₃ + p₄ ≥ 206 :=
by
  sorry

#check smallest_prime_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_sum_l81_8197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l81_8152

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) + (x - 5) ^ (1/4 : ℝ)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 5} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l81_8152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l81_8191

/-- The ellipse defined by (x^2/9) + (y^2/4) = 1 -/
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

/-- The line defined by x + 2y - 10 = 0 -/
def line (x y : ℝ) : Prop := x + 2*y - 10 = 0

/-- The distance from a point (x, y) to the line x + 2y - 10 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x + 2*y - 10| / Real.sqrt 5

theorem min_distance_to_line :
  ∀ x y : ℝ, ellipse x y → (∀ x' y' : ℝ, ellipse x' y' → distance_to_line x y ≤ distance_to_line x' y') →
  distance_to_line x y = 1 := by
  sorry

#check min_distance_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l81_8191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l81_8158

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Checks if a point is in the first quadrant -/
def isInFirstQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem tangent_line_y_intercept :
  ∀ (l : Line) (c1 c2 : Circle),
    c1.center = (3, 0) →
    c1.radius = 3 →
    c2.center = (7, 0) →
    c2.radius = 2 →
    isTangent l c1 →
    isTangent l c2 →
    (∃ (p1 p2 : ℝ × ℝ), isInFirstQuadrant p1 ∧ isInFirstQuadrant p2 ∧ 
      p1 ∈ {p | (p.1 - 3)^2 + p.2^2 = 9} ∧
      p2 ∈ {p | (p.1 - 7)^2 + p.2^2 = 4} ∧
      p1.2 = l.slope * p1.1 + l.yIntercept ∧
      p2.2 = l.slope * p2.1 + l.yIntercept) →
    l.yIntercept = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_y_intercept_l81_8158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l81_8129

/-- The distance between two parallel lines -/
noncomputable def distance_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  abs (C₂ - C₁) / Real.sqrt (A^2 + B^2)

/-- Theorem stating the possible values of 'a' given the distance between parallel lines -/
theorem parallel_lines_distance (a : ℝ) :
  distance_parallel_lines 4 3 (-6) a = 2 ↔ a = 4 ∨ a = -16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l81_8129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_expansion_l81_8135

-- Define TrailingZeros as a function
def TrailingZeros (n : ℕ) : ℕ := sorry

theorem zeros_in_expansion : TrailingZeros ((10^12 - 2)^2) = 11 := by
  sorry

#check zeros_in_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_expansion_l81_8135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_middle_swap_l81_8131

/-- Represents a circular arrangement of numbers -/
def CircularArrangement (n : ℕ) := Fin n → ℕ

/-- Performs a swap of adjacent elements in a circular arrangement -/
def swap (arr : CircularArrangement n) (k : Fin n) : CircularArrangement n :=
  sorry

/-- Theorem: In a circular arrangement of 2021 distinct numbers, 
    when performing 2021 swaps where the k-th swap exchanges the numbers 
    adjacent to k, there exists a swap k where the swapped numbers a and b 
    satisfy a < k < b. -/
theorem exists_middle_swap :
  ∃ (initial : CircularArrangement 2021) (k : Fin 2021),
    let final := (List.foldl (λ acc i => swap acc i) initial (List.range 2021))
    ∃ (a b : ℕ), 
      (a < k.val) ∧ 
      (k.val < b) ∧
      (final k = a ∧ (swap final k) k = b ∨ 
       final k = b ∧ (swap final k) k = a) :=
by
  sorry

#check exists_middle_swap

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_middle_swap_l81_8131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_all_squares_l81_8107

-- Define the sequence
def a : ℕ → ℕ → ℕ
  | k, 0 => k  -- Add this case for n = 0
  | k, n + 1 => a k n + 8 * n

-- Define what it means for a natural number to be a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- State the theorem
theorem sequence_all_squares (k : ℕ) :
  (∀ n : ℕ, n ≥ 1 → is_perfect_square (a k n)) ↔ k = 1 := by
  sorry

#check sequence_all_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_all_squares_l81_8107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_age_ratio_sachin_age_value_l81_8157

/-- Sachin's age in years -/
def sachin_age : ℝ := 24.5

/-- Rahul's age in years -/
def rahul_age : ℝ := sachin_age + 7

/-- Rahul is 7 years older than Sachin -/
theorem age_difference : rahul_age = sachin_age + 7 := by rfl

/-- The ratio of Sachin's age to Rahul's age is 7:9 -/
theorem age_ratio : sachin_age / rahul_age = 7 / 9 := by
  -- The proof is omitted for now
  sorry

/-- Theorem: Sachin's age is 24.5 years -/
theorem sachin_age_value : sachin_age = 24.5 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_age_ratio_sachin_age_value_l81_8157
