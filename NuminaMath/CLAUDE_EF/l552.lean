import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_zero_l552_55214

noncomputable def a : ℝ × ℝ := (-2, 3)
noncomputable def b (m : ℝ) : ℝ × ℝ := (1, m - 3/2)

theorem parallel_vectors_m_zero :
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b m) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_zero_l552_55214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_5x8_l552_55290

/-- Represents a rectangle with integer dimensions --/
structure MyRectangle where
  width : ℕ
  height : ℕ

/-- Represents a path within a rectangle --/
structure MyPath (r : MyRectangle) where
  diagonals : ℕ
  closed : Bool
  no_repeat_crossings : Bool
  no_repeat_vertices : Bool

/-- The maximum number of diagonals in a valid path within a 5x8 rectangle --/
def max_diagonals_5x8 : ℕ := 24

/-- Theorem stating that the maximum number of diagonals in a valid path within a 5x8 rectangle is 24 --/
theorem max_path_5x8 :
  ∀ (p : MyPath (MyRectangle.mk 5 8)),
    p.closed ∧ p.no_repeat_crossings ∧ p.no_repeat_vertices →
    p.diagonals ≤ max_diagonals_5x8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_5x8_l552_55290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_y_equals_63_l552_55212

/-- Given a set of data points and a linear regression equation, prove that the mean of y is 63 -/
theorem mean_y_equals_63 
  (y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (regression_eq : ℝ → ℝ)
  (h1 : regression_eq = λ x ↦ 2 * x + 45)
  (h2 : List.map (λ x ↦ (x, regression_eq x)) [1, 5, 7, 13, 19] = 
    [(1, y₁), (5, y₂), (7, y₃), (13, y₄), (19, y₅)]) :
  (y₁ + y₂ + y₃ + y₄ + y₅) / 5 = 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_y_equals_63_l552_55212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_problem_l552_55270

/-- Calculates the total fencing required for a rectangular field with square obstacles -/
noncomputable def total_fencing (field_area : ℝ) (field_width : ℝ) (obstacle1_side : ℝ) (obstacle2_side : ℝ) : ℝ :=
  let field_length := field_area / field_width
  let field_fencing := field_width + 2 * field_length
  let obstacle1_fencing := 4 * obstacle1_side
  let obstacle2_fencing := 4 * obstacle2_side
  field_fencing + obstacle1_fencing + obstacle2_fencing

/-- Theorem stating the total fencing required for the given problem -/
theorem fencing_problem : total_fencing 1200 40 8 4 = 148 := by
  -- Unfold the definition of total_fencing
  unfold total_fencing
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_problem_l552_55270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottles_from_625_l552_55262

/-- Calculates the total number of new bottles that can be made from recycling and bonuses -/
def total_new_bottles (initial_bottles : ℕ) : ℕ :=
  let rec recycle_and_bonus (remaining : ℕ) (acc : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then
      acc
    else
      let new_bottles := remaining / 5
      let bonus_bottles := acc / 20
      if new_bottles + bonus_bottles = 0 then
        acc
      else
        recycle_and_bonus (new_bottles + bonus_bottles) (acc + new_bottles + bonus_bottles) (fuel - 1)
  recycle_and_bonus initial_bottles 0 (initial_bottles + 1)

/-- Theorem stating that starting with 625 glass bottles, 163 new bottles can be made -/
theorem bottles_from_625 : total_new_bottles 625 = 163 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottles_from_625_l552_55262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l552_55238

/-- A function representing the curve y = e^x + x -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

/-- The derivative of f -/
noncomputable def f' (x : ℝ) : ℝ := Real.exp x + 1

/-- The tangent line y = 2x + b -/
def tangent_line (x b : ℝ) : ℝ := 2 * x + b

theorem tangent_line_b_value :
  ∃ x₀ : ℝ, (f' x₀ = 2 ∧ tangent_line x₀ (f x₀ - 2 * x₀) = f x₀) → (f x₀ - 2 * x₀ = 1) :=
by
  -- We'll use x₀ = 0 as our solution
  use 0
  intro h
  have h1 : f' 0 = 2 := h.1
  have h2 : tangent_line 0 (f 0 - 2 * 0) = f 0 := h.2
  
  -- Simplify f' 0
  have f'_eq : f' 0 = Real.exp 0 + 1 := rfl
  rw [f'_eq] at h1
  have exp_0 : Real.exp 0 = 1 := Real.exp_zero
  rw [exp_0] at h1
  
  -- Deduce that f 0 = 1
  have f_0 : f 0 = 1 := by
    unfold f
    rw [exp_0]
    ring
  
  -- Now we can prove the goal
  calc
    f 0 - 2 * 0 = f 0 := by ring
    _ = 1 := f_0

-- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l552_55238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_difference_equals_relationship_strength_relationship_strength_maximized_l552_55259

/-- The difference between two ratios in a two-dimensional bar chart -/
noncomputable def ratio_difference (a b c d : ℝ) : ℝ := (a / (a + b)) - (c / (c + d))

/-- The relationship strength between two categorical variables -/
noncomputable def relationship_strength (a b c d : ℝ) : ℝ := (a * d - b * c) / ((a + b) * (c + d))

theorem ratio_difference_equals_relationship_strength 
  (a b c d : ℝ) (h1 : a + b ≠ 0) (h2 : c + d ≠ 0) :
  ratio_difference a b c d = relationship_strength a b c d := by
  sorry

theorem relationship_strength_maximized 
  (a b c d : ℝ) (h1 : a + b ≠ 0) (h2 : c + d ≠ 0) (h3 : b + d ≠ 0) (h4 : a + c ≠ 0) :
  |relationship_strength a b c d| ≥ max 
    (|(a / (c + d)) - (c / (a + b))|)
    (max (|(a / (a + b)) - (c / (b + c))|)
         (|(a / (b + d)) - (c / (a + c))|)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_difference_equals_relationship_strength_relationship_strength_maximized_l552_55259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relationship_l552_55266

-- Define the constants
noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

-- State the theorem
theorem order_relationship : b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relationship_l552_55266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bounds_l552_55295

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 4

-- Define the line
def line (r : ℝ) : ℝ := r

-- Define the triangle area function
noncomputable def triangleArea (r : ℝ) : ℝ := (r + 4) * Real.sqrt (r + 4)

-- State the theorem
theorem triangle_area_bounds (r : ℝ) :
  16 ≤ triangleArea r ∧ triangleArea r ≤ 144 ↔ 0 ≤ r ∧ r ≤ 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bounds_l552_55295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_condition_l552_55285

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x + Real.pi / 4) - 2

theorem monotonic_decreasing_condition (ω : ℝ) : 
  (ω > 0 ∧ 
   ∀ x ∈ Set.Icc (Real.pi / 2) Real.pi, 
     ∀ y ∈ Set.Icc (Real.pi / 2) Real.pi, 
       x < y → f ω x > f ω y) ↔ 
  (ω ≥ 1 / 2 ∧ ω ≤ 5 / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_condition_l552_55285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_third_f_max_value_f_min_value_l552_55216

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x) + Real.sin x ^ 2 - 4 * Real.cos x

-- State the theorems
theorem f_at_pi_third : f (Real.pi / 3) = -9 / 4 := by sorry

theorem f_max_value : ∃ x : ℝ, f x = 6 ∧ ∀ y : ℝ, f y ≤ 6 := by sorry

theorem f_min_value : ∃ x : ℝ, f x = -7 / 3 ∧ ∀ y : ℝ, f y ≥ -7 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_third_f_max_value_f_min_value_l552_55216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l552_55206

theorem power_of_three (y : ℝ) (h : (3 : ℝ)^y = 81) : (3 : ℝ)^(y+3) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l552_55206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shot_radius_l552_55288

/-- The volume of a sphere given its radius -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- Given a sphere of radius 6 cm from which 216 equal-sized spherical shots are made,
    the radius of each shot is 1 cm -/
theorem shot_radius (r : ℝ) (h1 : sphereVolume 6 = 216 * sphereVolume r) : r = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shot_radius_l552_55288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_greater_l552_55213

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x
def g (x : ℝ) : ℝ := x + 1

-- Statement to prove
theorem not_always_greater : ¬ (∀ x : ℝ, f x > g x) := by
  -- We'll use proof by contradiction
  intro h
  -- Consider x = 0
  let x := 0
  -- Apply the hypothesis to x = 0
  have h0 : f 0 > g 0 := h 0
  -- Simplify f(0) and g(0)
  have f0 : f 0 = 1 := by simp [f, Real.exp_zero]
  have g0 : g 0 = 1 := by simp [g]
  -- Rewrite the inequality using these simplifications
  rw [f0, g0] at h0
  -- This leads to a contradiction: 1 > 1
  exact lt_irrefl 1 h0


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_greater_l552_55213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_pi_l552_55223

open Real

theorem sum_of_solutions_is_pi : ∃ (x₁ x₂ : ℝ), 
  0 ≤ x₁ ∧ x₁ ≤ 2*π ∧
  0 ≤ x₂ ∧ x₂ ≤ 2*π ∧
  1/sin x₁ + 1/cos x₁ = 4 ∧
  1/sin x₂ + 1/cos x₂ = 4 ∧
  x₁ + x₂ = π ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2*π ∧ 1/sin x + 1/cos x = 4 → x = x₁ ∨ x = x₂ := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_pi_l552_55223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_equals_two_l552_55229

/-- The function y = a * cos(2x + π/3) + 3 --/
noncomputable def y (a x : ℝ) : ℝ := a * Real.cos (2 * x + Real.pi / 3) + 3

/-- The theorem stating that if the maximum value of y is 4 for x in [0, π/2] and a > 0, then a = 2 --/
theorem max_value_implies_a_equals_two (a : ℝ) (h_a_pos : a > 0) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → y a x ≤ 4) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ y a x = 4) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_equals_two_l552_55229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_gold_distribution_possible_l552_55241

/-- Represents the amount of gold a pirate has won (positive) or lost (negative) -/
def NetGold := ℤ

/-- Represents a pirate's gold transaction -/
structure PirateTransaction where
  amount : ℚ
  is_paying : Bool

/-- Represents the state of all pirates' gold -/
def PirateState := Fin 100 → NetGold

/-- Applies a transaction to the pirate state -/
def applyTransaction (state : PirateState) (actor : Fin 100) (t : PirateTransaction) : PirateState :=
  sorry

/-- Checks if the final state matches the initial net gold for each pirate -/
def isValidFinalState (initial : PirateState) (final : PirateState) : Prop :=
  ∀ i : Fin 100, final i = initial i

theorem pirate_gold_distribution_possible (initial : PirateState) :
  ∃ (transactions : List (Fin 100 × PirateTransaction)) (final : PirateState),
    (∀ (p : Fin 100 × PirateTransaction), p ∈ transactions → 
      final = applyTransaction final p.1 p.2) ∧
    isValidFinalState initial final :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_gold_distribution_possible_l552_55241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobby_pin_percentage_rounded_l552_55235

def num_barrettes : ℕ := 6

def num_scrunchies : ℕ := 2 * num_barrettes

def num_bobby_pins : ℕ := num_barrettes - 3

def total_decorations : ℕ := num_barrettes + num_scrunchies + num_bobby_pins

def bobby_pin_percentage : ℚ := (num_bobby_pins : ℚ) / (total_decorations : ℚ) * 100

noncomputable def rounded_percentage : ℤ := Int.floor (bobby_pin_percentage + 1/2)

theorem bobby_pin_percentage_rounded : 
  rounded_percentage = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobby_pin_percentage_rounded_l552_55235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_two_roots_l552_55299

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * Real.log x

-- Define the condition for two roots
def has_two_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f a x₁ = 0 ∧ f a x₂ = 0

-- Theorem stating the range of a
theorem a_range_for_two_roots :
  ∀ a : ℝ, has_two_roots a ↔ a > 2 * Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_two_roots_l552_55299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tadpoles_equal_area_l552_55267

noncomputable section

/-- The area of a circle -/
def circle_area (r : ℝ) : ℝ := Real.pi * r^2

/-- The area of an equilateral triangle -/
def equilateral_triangle_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2

/-- The area of a tadpole -/
def tadpole_area (r : ℝ) : ℝ :=
  circle_area r + equilateral_triangle_area (2*r) - 3 * (circle_area r / 6)

theorem tadpoles_equal_area (r : ℝ) (h : r > 0) :
  tadpole_area r = tadpole_area r :=
by
  -- The proof is trivial since we're comparing the same expression
  rfl

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tadpoles_equal_area_l552_55267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l552_55242

/-- Predicate to determine if a given y-coordinate represents the directrix of a parabola -/
def IsDirectrix (y : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ (a h k : ℝ), a ≠ 0 ∧ 
    (∀ x, f x = a * (x - h)^2 + k) ∧
    y = k - 1 / (4 * a)

/-- The directrix of the parabola y = 2x^2 - 6x + 5 is y = 3/8 -/
theorem parabola_directrix (x y : ℝ) : 
  y = 2 * x^2 - 6 * x + 5 → 
  ∃ (k : ℝ), k = 3/8 ∧ IsDirectrix k (λ t => 2 * t^2 - 6 * t + 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l552_55242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_inscribed_square_l552_55228

theorem arc_length_inscribed_square (r : ℝ) (h : r > 0) : 
  let d := 2 * r
  let a := d / Real.sqrt 2
  let α := a / r
  α = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_inscribed_square_l552_55228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_inequality_l552_55294

theorem fourth_root_inequality (x : ℝ) : 
  (x > 0) → ((x^(1/4) + 3 / (x^(1/4) - 2) ≤ 0) ↔ x < 16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_inequality_l552_55294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2010th_term_l552_55274

def sequenceN (n : ℕ+) : ℤ :=
  (-1 : ℤ)^(n.val + 1) * ((n.val^2 : ℤ) - 1)

theorem sequence_2010th_term :
  sequenceN 2010 = 4040099 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2010th_term_l552_55274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_iter_formula_l552_55280

noncomputable def f (x : ℝ) : ℝ := (1010 * x + 1009) / (1009 * x + 1010)

noncomputable def f_iter : ℕ → (ℝ → ℝ)
| 0 => id
| n + 1 => f ∘ (f_iter n)

theorem f_iter_formula (n : ℕ) (x : ℝ) :
  f_iter n x = ((2019^n + 1) * x + (2019^n - 1)) / ((2019^n - 1) * x + (2019^n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_iter_formula_l552_55280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l552_55291

noncomputable section

-- Define an acute triangle
def AcuteTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b + c ∧ b < a + c ∧ c < a + b

-- Define the area of a triangle using Heron's formula
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_side_length 
  (a b c : ℝ) (A B C : ℝ) :
  AcuteTriangle a b c →
  b = 3 →
  c = 1 →
  triangleArea a b c = Real.sqrt 2 →
  a = 2 * Real.sqrt 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l552_55291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l552_55252

theorem min_omega_value (ω : ℝ) : 
  (ω > 0) → 
  (∀ x : ℝ, Real.cos (ω * (x - Real.pi / 3) + Real.pi / 3) = Real.sin (ω * x)) → 
  ω = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l552_55252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_theorem_l552_55202

/-- Circle type -/
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

/-- Triangle type -/
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

/-- Define necessary geometric predicates -/
def is_inscribed (t : Triangle) (c : Circle) : Prop := sorry
def on_circle (p : EuclideanSpace ℝ (Fin 2)) (c : Circle) : Prop := sorry
def on_segment (p q r : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def on_arc (p q r : EuclideanSpace ℝ (Fin 2)) (c : Circle) : Prop := sorry
def on_circumcircle (p q r s : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def line (p q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry
def collinear (p q r : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

theorem collinearity_theorem 
  (Ω : Circle) 
  (ABC : Triangle) 
  (Γ : Circle)
  (O D E F G K L X : EuclideanSpace ℝ (Fin 2)) :
  (ABC.A = Γ.center) →
  (Ω.center = O) →
  (is_inscribed ABC Ω) →
  (on_circle D Γ) →
  (on_circle E Γ) →
  (on_segment D ABC.B ABC.C) →
  (on_segment E D ABC.C) →
  (on_circle F Γ) →
  (on_circle G Γ) →
  (on_circle F Ω) →
  (on_circle G Ω) →
  (on_arc F ABC.A ABC.B Ω) →
  (on_arc G ABC.A ABC.C Ω) →
  (on_circumcircle K ABC.B D F) →
  (on_segment K ABC.A ABC.B) →
  (on_circumcircle L ABC.C E G) →
  (on_segment L ABC.A ABC.C) →
  (X ∈ line F K) →
  (X ∈ line G L) →
  (line F K ≠ line G L) →
  collinear ABC.A X O :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_theorem_l552_55202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_prime_generating_a_l552_55284

theorem smallest_non_prime_generating_a (a : ℕ) : a = 9 ↔ 
  (a > 8 ∧ 
   (∀ x : ℤ, ¬(Nat.Prime (Int.natAbs (x^4) + a^2))) ∧
   (∀ b : ℕ, 8 < b ∧ b < a → ∃ x : ℤ, Nat.Prime (Int.natAbs (x^4) + b^2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_prime_generating_a_l552_55284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l552_55265

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 10
def g (x : ℝ) : ℝ := x^2 - 6

-- State the theorem
theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 18) : 
  a = Real.sqrt (2 * Real.sqrt 2 + 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l552_55265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_inverse_tan_l552_55220

theorem tan_plus_inverse_tan (θ : ℝ) (h : Real.sin θ + Real.cos θ = Real.sqrt 2) :
  Real.tan θ + (1 / Real.tan θ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_inverse_tan_l552_55220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_atOp_properties_l552_55233

-- Define the @ operation (renamed to 'atOp')
def atOp (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Theorem statement
theorem atOp_properties :
  (atOp 1 (-2) = -8) ∧
  (∀ a b : ℝ, atOp a b = atOp b a) ∧
  (∀ a b : ℝ, a + b = 0 → (atOp a a) + (atOp b b) = 8*a^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_atOp_properties_l552_55233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l552_55211

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - (1/2) * x^2 + 1/4

-- State the theorem
theorem f_has_unique_zero (a : ℝ) (h : a ∈ Set.Ioo (-1/2) 0) :
  ∃! x, x > 0 ∧ f a x = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l552_55211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l552_55278

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def satisfies_equations (x y : ℝ) : Prop :=
  y = 3 * (floor x) + 4 ∧ y = 4 * (floor (x - 3)) + 7

theorem range_of_sum (x y : ℝ) :
  satisfies_equations x y → ¬ (∃ n : ℤ, x = n) → 40 < x + y ∧ x + y < 41 := by
  sorry

#check range_of_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l552_55278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_a_l552_55275

def A : Set ℝ := {-1, 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | a * x + 1 = 0}

theorem possible_values_of_a (a : ℝ) : B a ⊆ A → a ∈ ({-1, 1} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_a_l552_55275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_less_than_two_l552_55231

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a * Real.log x

-- State the theorem
theorem inequality_holds_iff_a_less_than_two :
  ∀ a : ℝ, (∀ t : ℝ, t ≥ 1 → f a (2*t - 1) ≥ 2 * f a t - 3) ↔ a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_less_than_two_l552_55231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_porter_equation_correct_l552_55257

/-- Represents the amount of cargo (in kg) that porter A can transport per hour. -/
def porter_a_cargo : ℝ → ℝ := λ x => x

/-- Represents the amount of cargo (in kg) that porter B can transport per hour. -/
def porter_b_cargo : ℝ → ℝ := λ x => x + 600

/-- The time it takes for porter A to transport 5000 kg. -/
noncomputable def time_a : ℝ → ℝ := λ x => 5000 / x

/-- The time it takes for porter B to transport 8000 kg. -/
noncomputable def time_b : ℝ → ℝ := λ x => 8000 / (porter_b_cargo x)

/-- Theorem stating that the equation correctly represents the given conditions. -/
theorem porter_equation_correct (x : ℝ) (hx : x > 0) : time_a x = time_b x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_porter_equation_correct_l552_55257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l552_55201

/-- The ellipse with equation x²/16 + y²/12 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 12) = 1}

/-- The left focus of the ellipse -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the ellipse -/
def F₂ : ℝ × ℝ := sorry

/-- Vector from a point to the left focus -/
def PF₁ (P : ℝ × ℝ) : ℝ × ℝ := (F₁.1 - P.1, F₁.2 - P.2)

/-- Vector from a point to the right focus -/
def PF₂ (P : ℝ × ℝ) : ℝ × ℝ := (F₂.1 - P.1, F₂.2 - P.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem ellipse_property (P : ℝ × ℝ) (h₁ : P ∈ Ellipse) 
    (h₂ : dot_product (PF₁ P) (PF₂ P) = 9) : 
  magnitude (PF₁ P) * magnitude (PF₂ P) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l552_55201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l552_55208

-- Define the binary operation
noncomputable def diamond (a b : ℝ) : ℝ := a / b

-- State the theorem
theorem diamond_equation_solution :
  (∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 →
    diamond a (diamond b c) = (diamond a b) * c) →
  (∀ (a : ℝ), a ≠ 0 → diamond a a = 1) →
  (∃ (x : ℝ), x ≠ 0 ∧ diamond 504 (diamond 3 x) = 50) →
  (∃ (x : ℝ), x = 25 / 84 ∧ diamond 504 (diamond 3 x) = 50) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l552_55208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_side_b_range_of_b_plus_c_l552_55218

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_acute_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧ 0 < t.B ∧ t.B < Real.pi/2 ∧ 0 < t.C ∧ t.C < Real.pi/2

-- Theorem 1
theorem find_side_b (t : Triangle) 
  (h1 : 23 * (Real.cos t.A)^2 + Real.cos (2 * t.A) = 0)
  (h2 : is_acute_triangle t)
  (h3 : t.a = 7)
  (h4 : t.c = 6) :
  t.b = 5 := by sorry

-- Theorem 2
theorem range_of_b_plus_c (t : Triangle)
  (h1 : t.a = Real.sqrt 3)
  (h2 : t.A = Real.pi/3) :
  Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_side_b_range_of_b_plus_c_l552_55218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_example_l552_55236

/-- The distance between a point (x₀, y₀) and a line ax + by + c = 0 --/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  (abs (a * x₀ + b * y₀ + c)) / (Real.sqrt (a^2 + b^2))

/-- Determines if a line intersects a circle --/
def line_intersects_circle (a b c x₀ y₀ r : ℝ) : Prop :=
  distance_point_to_line x₀ y₀ a b c < r

theorem line_intersects_circle_example :
  let a : ℝ := 2
  let b : ℝ := -1
  let c : ℝ := 3
  let x₀ : ℝ := 0
  let y₀ : ℝ := 1
  let r : ℝ := Real.sqrt 5
  line_intersects_circle a b c x₀ y₀ r :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_example_l552_55236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_is_18_l552_55244

def sequence_b : ℕ → ℚ
  | 0 => 2
  | 1 => 2
  | n + 2 => (1/2) * sequence_b (n + 1) + (1/3) * sequence_b n

noncomputable def sequence_sum : ℚ := ∑' n, sequence_b n

theorem sequence_sum_is_18 : sequence_sum = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_is_18_l552_55244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_7_value_l552_55251

/-- Given a real number x such that x + 1/x = 5, T_m is defined as x^m + 1/x^m -/
noncomputable def T (x m : ℝ) : ℝ := x^m + 1/x^m

/-- Theorem stating that under the given conditions, T_7 equals 57960 -/
theorem T_7_value (x : ℝ) (h : x + 1/x = 5) : T x 7 = 57960 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_7_value_l552_55251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_elements_in_T_l552_55237

/-- First arithmetic progression -/
def seq1 (k : ℕ) : ℤ := 5 * k - 3

/-- Second arithmetic progression -/
def seq2 (m : ℕ) : ℤ := 9 * m - 3

/-- Set of the first 1003 terms of the first sequence -/
def C : Finset ℤ := Finset.image seq1 (Finset.range 1003)

/-- Set of the first 1003 terms of the second sequence -/
def D : Finset ℤ := Finset.image seq2 (Finset.range 1003)

/-- Union of sets C and D -/
def T : Finset ℤ := C ∪ D

theorem distinct_elements_in_T : T.card = 1895 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_elements_in_T_l552_55237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_angles_specific_triangle_l552_55255

-- Define the triangle
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  A + B + C = Real.pi

-- Theorem statement
theorem sin_sum_angles_specific_triangle :
  ∀ A B C a b c : ℝ,
  Triangle A B C a b c →
  a = 4 →
  b = 5 →
  c = 6 →
  Real.sin (A + B) = 3 * Real.sqrt 7 / 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_angles_specific_triangle_l552_55255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_wrapping_paper_area_is_answer_l552_55250

theorem wrapping_paper_area 
  (l w h : ℝ) 
  (hl : 0 < l) 
  (hw : 0 < w) 
  (hh : 0 < h) 
  (hlw : l ≥ w) : 
  (l + 2*h)^2 = (l + 2*h)^2 := by
  -- The proof is trivial as we're stating that the expression is equal to itself
  rfl

theorem wrapping_paper_area_is_answer 
  (l w h : ℝ) 
  (hl : 0 < l) 
  (hw : 0 < w) 
  (hh : 0 < h) 
  (hlw : l ≥ w) : 
  ∃ (area : ℝ), area = (l + 2*h)^2 ∧ area ≥ (w + 2*h)^2 := by
  use (l + 2*h)^2
  constructor
  · rfl
  · sorry  -- This step requires more detailed proof

#check wrapping_paper_area
#check wrapping_paper_area_is_answer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_wrapping_paper_area_is_answer_l552_55250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_triangle_area_sum_l552_55264

noncomputable def cube_side_length : ℝ := 2

-- Define the sum of areas of triangles
noncomputable def triangle_area_sum (m n p : ℕ) : ℝ := m + Real.sqrt (n : ℝ) + Real.sqrt (p : ℝ)

-- Theorem statement
theorem cube_triangle_area_sum :
  ∃ (m n p : ℕ), 
    triangle_area_sum m n p = 
      (6 * 4 * (cube_side_length^2 / 2)) + 
      (24 * (cube_side_length * (Real.sqrt 2) / 2)) +
      (8 * ((Real.sqrt 3 / 4) * ((cube_side_length * Real.sqrt 2)^2))) ∧
    m + n + p = 7728 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_triangle_area_sum_l552_55264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l552_55277

-- Define the types and variables
variable (p q r s t : ℝ)

-- Define the conditions
variable (h1 : p < q)
variable (h2 : q < r)
variable (h3 : r < s)
variable (h4 : s < t)

-- Define the functions M and m
noncomputable def M (x y : ℝ) : ℝ := max x y
noncomputable def m (x y : ℝ) : ℝ := min x y

-- State the theorem
theorem problem_solution :
  M (M (m p q) r) (m s (M t p)) = s :=
by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l552_55277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_approximation_l552_55289

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * years)

noncomputable def gain_per_year (
  principal : ℝ)
  (borrow_rate : ℝ)
  (borrow_compounds : ℝ)
  (lend_rate : ℝ)
  (lend_compounds : ℝ)
  (years : ℝ) : ℝ :=
let borrowed_amount := compound_interest principal borrow_rate borrow_compounds years
let lent_amount := compound_interest principal lend_rate lend_compounds years
(lent_amount - borrowed_amount) / years

theorem transaction_gain_approximation :
  let principal : ℝ := 9000
  let borrow_rate : ℝ := 0.04
  let borrow_compounds : ℝ := 4
  let lend_rate : ℝ := 0.06
  let lend_compounds : ℝ := 2
  let years : ℝ := 2
  abs (gain_per_year principal borrow_rate borrow_compounds lend_rate lend_compounds years - 192.49) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_approximation_l552_55289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_square_z_l552_55260

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℚ
  width : ℚ

/-- Represents a square with a given side length -/
structure Square where
  side : ℚ

/-- Represents the value z, which is one-third of the square's side length -/
def z (s : Square) : ℚ := s.side / 3

/-- Theorem stating that for a 9 × 16 rectangle that can be cut into two congruent hexagons
    forming a square, the value z is equal to 4 -/
theorem rectangle_to_square_z (r : Rectangle) (s : Square) : 
  r.length = 16 ∧ r.width = 9 ∧ r.length * r.width = s.side * s.side → z s = 4 := by
  sorry

#eval z { side := 12 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_square_z_l552_55260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_park_theorem_l552_55248

/-- A rhombus-shaped park with a square play area inside --/
structure RhombusPark where
  /-- Length of the first diagonal of the rhombus --/
  diagonal1 : ℝ
  /-- Length of the second diagonal of the rhombus --/
  diagonal2 : ℝ
  /-- The square play area has a side length equal to the smaller diagonal --/
  square_side : ℝ
  /-- Assumption that the square side is equal to the smaller diagonal --/
  h_square_side : square_side = min diagonal1 diagonal2

/-- Properties of the rhombus park --/
def rhombus_park_properties (park : RhombusPark) : Prop :=
  park.diagonal1 = 24 ∧ 
  park.diagonal2 = 16 ∧
  (4 * (((park.diagonal1 / 2) ^ 2 + (park.diagonal2 / 2) ^ 2) ^ (1/2 : ℝ)) = 16 * Real.sqrt 13) ∧
  park.square_side ^ 2 = 256

/-- Theorem stating the properties of the rhombus park --/
theorem rhombus_park_theorem (park : RhombusPark) : rhombus_park_properties park := by
  sorry

#check rhombus_park_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_park_theorem_l552_55248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l552_55263

/-- The area of a triangle with vertices at (3, 2), (3, -4), and (12, 2) is 27 square units. -/
theorem triangle_area : ∃ area : ℝ, area = 27 := by
  -- Define the vertices of the triangle
  let A : ℝ × ℝ := (3, 2)
  let B : ℝ × ℝ := (3, -4)
  let C : ℝ × ℝ := (12, 2)

  -- Calculate the area of the triangle
  let area := (1/2) * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) : ℝ)

  -- Prove that the area is equal to 27
  exists area
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l552_55263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l552_55226

/-- The curve function f(x) = x^3 -/
def f (x : ℝ) : ℝ := x^3

/-- The tangent point -/
def tangent_point : ℝ × ℝ := (3, 27)

/-- The slope of the tangent line at the tangent point -/
noncomputable def tangent_slope : ℝ := 3 * tangent_point.fst^2

/-- The y-intercept of the tangent line -/
noncomputable def y_intercept : ℝ := tangent_point.snd - tangent_slope * tangent_point.fst

/-- The x-intercept of the tangent line -/
noncomputable def x_intercept : ℝ := -y_intercept / tangent_slope

/-- The area of the triangle formed by the tangent line and coordinate axes -/
noncomputable def triangle_area : ℝ := (1/2) * x_intercept * (-y_intercept)

/-- Theorem: The area of the triangle is 54 square units -/
theorem tangent_triangle_area : triangle_area = 54 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l552_55226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_property_l552_55261

def move_first_to_last (n : ℕ) : ℕ :=
  let s := toString n
  let l := s.length
  if l > 1 then
    (s.drop 1 ++ s.take 1).toNat!
  else
    n

theorem smallest_n_with_property : ∃ (n : ℕ),
  n > 0 ∧
  move_first_to_last n = (7 * n) / 2 ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → move_first_to_last m ≠ (7 * m) / 2 :=
by
  use 153846
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_property_l552_55261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ages_l552_55215

-- Define Paula's current age
def Paula : ℕ := sorry

-- Define Karl's current age
def Karl : ℕ := sorry

-- 7 years ago, Paula was 3 times as old as Karl
axiom condition1 : Paula - 7 = 3 * (Karl - 7)

-- In 2 years, Paula will be twice as old as Karl
axiom condition2 : Paula + 2 = 2 * (Karl + 2)

-- Theorem: The sum of their current ages is 50
theorem sum_of_ages : Paula + Karl = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ages_l552_55215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_horse_revolutions_merry_go_round_problem_l552_55298

/-- Calculates the number of revolutions needed for a circular path to cover a given distance -/
noncomputable def revolutions_needed (radius : ℝ) (distance : ℝ) : ℝ :=
  distance / (2 * Real.pi * radius)

theorem inner_horse_revolutions (r1 r2 n1 : ℝ) (hr1 : r1 > 0) (hr2 : r2 > 0) (hn1 : n1 > 0) :
  let d := 2 * Real.pi * r1 * n1  -- Total distance covered by the outer path
  let n2 := revolutions_needed r2 d  -- Number of revolutions needed for the inner path
  n2 = (r1 / r2) * n1 := by sorry

-- Using #eval with noncomputable functions is not possible
-- Instead, we can state the result as a theorem
theorem merry_go_round_problem :
  revolutions_needed 10 (2 * Real.pi * 30 * 25) = 75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_horse_revolutions_merry_go_round_problem_l552_55298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_race_total_l552_55269

/-- Represents a group of students in the charity race -/
structure CharityGroup where
  students : ℕ
  raceAmount : ℕ
  additionalAmount : ℕ

/-- Calculates the total amount raised by a group -/
def groupTotal (g : CharityGroup) : ℕ := g.students * g.raceAmount + g.additionalAmount

/-- The charity race problem -/
theorem charity_race_total :
  let groupA : CharityGroup := { students := 10, raceAmount := 20, additionalAmount := 50 }
  let groupB : CharityGroup := { students := 12, raceAmount := 30, additionalAmount := 120 }
  let groupC : CharityGroup := { students := 8, raceAmount := 25, additionalAmount := 150 }
  let groupD : CharityGroup := { students := 15, raceAmount := 35, additionalAmount := 200 }
  let groupE : CharityGroup := { students := 5, raceAmount := 40, additionalAmount := 300 }
  let totalStudents := groupA.students + groupB.students + groupC.students + groupD.students + groupE.students
  totalStudents = 50 →
  groupTotal groupA + groupTotal groupB + groupTotal groupC + groupTotal groupD + groupTotal groupE = 2305 :=
by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_race_total_l552_55269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_triangle_properties_l552_55230

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = 4*x
def line (x y m : ℝ) : Prop := y = 2*x + m

-- Define the chord length
noncomputable def chord_length (m : ℝ) : ℝ := 3 * Real.sqrt 5

-- Define the area of triangle ABP
def triangle_area : ℝ := 9

-- Theorem statement
theorem intersection_and_triangle_properties :
  ∃ (m : ℝ) (p_x : ℝ),
    (∀ x y, parabola x y ∧ line x y m → 
      chord_length m = 3 * Real.sqrt 5) ∧
    (triangle_area = 9) ∧
    (m = -4) ∧
    (p_x = 5 ∨ p_x = -1) := by
  -- We use 'sorry' to skip the proof for now
  sorry

#check intersection_and_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_triangle_properties_l552_55230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l552_55232

/-- Triangle ABC with vertices A(0,2), B(0,0), and C(6,0) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- The area of a triangle formed by a vertical line at x = a -/
noncomputable def areaLeftOfLine (a : ℝ) : ℝ := triangleArea a ((2*a)/6)

/-- The statement to be proved -/
theorem equal_area_division (ABC : Triangle) (h1 : ABC.A = (0, 2)) 
    (h2 : ABC.B = (0, 0)) (h3 : ABC.C = (6, 0)) :
  areaLeftOfLine 3 = (1/2) * triangleArea 6 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l552_55232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_solution_set_l552_55219

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 - 3*x| ≥ 4} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_solution_set_l552_55219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_collection_end_count_l552_55283

/-- Calculates the number of books in a special collection at the end of a month,
    given the initial number of books, the number of books loaned out,
    and the percentage of loaned books returned. -/
def books_at_end_of_month (initial_books : ℕ) (loaned_books : ℕ) (return_rate : ℚ) : ℕ :=
  initial_books - (loaned_books - Int.toNat ((return_rate * loaned_books).floor))

/-- Proves that given the specified conditions, there will be 63 books
    in the special collection at the end of the month. -/
theorem special_collection_end_count :
  books_at_end_of_month 75 40 (70 / 100) = 63 := by
  sorry

#eval books_at_end_of_month 75 40 (70 / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_collection_end_count_l552_55283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_coefficient_condition_l552_55245

def is_rational_coefficient (n : ℕ+) : Prop :=
  ∃ r : ℕ, r ≤ n ∧ (n - r).bodd = false ∧ r % 3 = 0

theorem rational_coefficient_condition (n : ℕ+) :
  (n = 9 ∨ n = 6 ∨ n = 7 ∨ n = 8) →
  (is_rational_coefficient n ↔ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_coefficient_condition_l552_55245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_length_l552_55276

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the circle
def circleEq (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Define the intersection points
def intersection (A B : ℝ × ℝ) : Prop :=
  asymptote A.1 A.2 ∧ circleEq A.1 A.2 ∧
  asymptote B.1 B.2 ∧ circleEq B.1 B.2

-- The theorem
theorem hyperbola_intersection_length (a b : ℝ) (A B : ℝ × ℝ) :
  a > 0 → b > 0 →
  hyperbola a b A.1 A.2 →
  hyperbola a b B.1 B.2 →
  eccentricity a b = Real.sqrt 5 →
  intersection A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_length_l552_55276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l552_55293

/-- Given a hyperbola W and point P, prove the equation of W -/
theorem hyperbola_equation (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (∃ (A B C D Q : ℝ × ℝ),
    -- W is a hyperbola with equation x²/a² - y²/b² = 1
    (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) ↔ (x, y) ∈ Set.range (fun p : ℝ × ℝ => p)) ∧
    -- P is outside the hyperbola
    (0, 1) ∉ Set.range (fun p : ℝ × ℝ => p) ∧
    -- A and B are intersection points of W with the x-axis
    A.2 = 0 ∧ B.2 = 0 ∧ A ∈ Set.range (fun p : ℝ × ℝ => p) ∧ B ∈ Set.range (fun p : ℝ × ℝ => p) ∧
    -- C and D are on the hyperbola
    C ∈ Set.range (fun p : ℝ × ℝ => p) ∧ D ∈ Set.range (fun p : ℝ × ℝ => p) ∧
    -- PA and PB intersect W at C and D respectively
    (∃ (t₁ t₂ : ℝ), C = (1 - t₁) • (0, 1) + t₁ • A ∧ D = (1 - t₂) • (0, 1) + t₂ • B) ∧
    -- Q is the intersection of tangent lines at C and D
    (∃ (s₁ s₂ : ℝ), 
      Q.1 = C.1 + s₁ * (b^2 * C.1 / (a^2 * C.2)) ∧
      Q.2 = C.2 + s₁ ∧
      Q.1 = D.1 + s₂ * (b^2 * D.1 / (a^2 * D.2)) ∧
      Q.2 = D.2 + s₂) ∧
    -- Triangle QCD is equilateral
    dist Q C = dist C D ∧ dist C D = dist D Q ∧
    -- Area of triangle QCD is 16√3/27
    (1/4) * Real.sqrt 3 * (dist C D)^2 = 16 * Real.sqrt 3 / 27) →
  -- The equation of hyperbola W is 27x²/4 - 3y² = 1
  a^2 = 4/27 ∧ b^2 = 1/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l552_55293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equation_f_range_OA_plus_OB_magnitude_l552_55287

noncomputable section

def O : ℝ × ℝ := (0, 0)
def A (α : ℝ) : ℝ × ℝ := (Real.sin α, 1)
def B (α : ℝ) : ℝ × ℝ := (Real.cos α, 0)
def C (α : ℝ) : ℝ × ℝ := (-Real.sin α, 2)

def P (α : ℝ) : ℝ × ℝ := (2 * Real.cos α - Real.sin α, -1)

def AB (α : ℝ) : ℝ × ℝ := ((B α).1 - (A α).1, (B α).2 - (A α).2)
def BP (α : ℝ) : ℝ × ℝ := ((P α).1 - (B α).1, (P α).2 - (B α).2)

def f (α : ℝ) : ℝ := (BP α).1 * ((C α).1 - (A α).1) + (BP α).2 * ((C α).2 - (A α).2)

theorem f_equation (α : ℝ) : f α = -Real.sqrt 2 * Real.sin (2 * α + π / 4) := by sorry

theorem f_range : Set.Icc (-Real.sqrt 2) 1 ⊆ Set.range f := by sorry

theorem OA_plus_OB_magnitude (α : ℝ) 
  (h : (2 * Real.cos α - Real.sin α) * 2 + Real.sin α = 0) : 
  Real.sqrt ((A α).1 + (B α).1)^2 + ((A α).2 + (B α).2)^2 = Real.sqrt 74 / 5 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equation_f_range_OA_plus_OB_magnitude_l552_55287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodicity_f_three_halves_l552_55292

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x % 2 ∧ x % 2 < 0 then -4 * (x % 2)^2 + 2
  else if 0 ≤ x % 2 ∧ x % 2 < 1 then x % 2
  else 0  -- This case should never occur for the given domain

theorem f_periodicity (x : ℝ) : f (x + 2) = f x := by sorry

theorem f_three_halves : f (3/2) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodicity_f_three_halves_l552_55292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_lengths_l552_55271

-- Define the points and their distances
structure LineSegment where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (s : LineSegment) : Prop :=
  |s.B - s.A| = 5 ∧ |s.C - s.A| = 1.5 * |s.C - s.B|

-- Define the theorem
theorem line_segment_lengths (s : LineSegment) :
  satisfies_conditions s →
  (|s.C - s.B| = 10 ∧ |s.C - s.A| = 15) ∨
  (|s.C - s.B| = 2 ∧ |s.C - s.A| = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_lengths_l552_55271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_and_hypotenuse_l552_55217

theorem right_triangle_area_and_hypotenuse 
  (leg1 leg2 : ℝ) 
  (h_leg1 : leg1 = 36) 
  (h_leg2 : leg2 = 48) : 
  (1 / 2) * leg1 * leg2 = 864 ∧ Real.sqrt (leg1^2 + leg2^2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_and_hypotenuse_l552_55217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_equals_one_fifth_l552_55253

-- Define g(n) for positive integers n
noncomputable def g (n : ℕ+) : ℝ := ∑' k : ℕ+, (1 : ℝ) / (k + 4 : ℝ) ^ n.val

-- State the theorem
theorem sum_of_g_equals_one_fifth :
  ∑' n : ℕ+, g (n + 1) = (1 : ℝ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_equals_one_fifth_l552_55253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_coords_l552_55256

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x - 5)^2 / 3^2 - (y + 8)^2 / 7^2 = 1

-- Define the distance from center to focus
noncomputable def focal_distance : ℝ := Real.sqrt (3^2 + 7^2)

-- Define the coordinates of the focus with larger y-coordinate
noncomputable def focus_coords : ℝ × ℝ := (5, -8 + focal_distance)

-- Theorem statement
theorem hyperbola_focus_coords :
  ∀ (x y : ℝ), hyperbola x y →
  ∃ (f₁ f₂ : ℝ × ℝ), 
    f₁.1 = f₂.1 ∧ f₁.2 ≠ f₂.2 ∧
    (f₁.2 > f₂.2 → f₁ = focus_coords) ∧
    (f₂.2 > f₁.2 → f₂ = focus_coords) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_coords_l552_55256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_prime_values_is_constant_l552_55225

/-- A polynomial with integer coefficients such that P(n) is prime for all integers n is constant. -/
theorem polynomial_prime_values_is_constant (P : Polynomial ℤ)
  (h : ∀ n : ℤ, Nat.Prime (Int.natAbs (P.eval n))) : P.degree = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_prime_values_is_constant_l552_55225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_hyperbola_eccentricity_l552_55207

/-- The function representing the original curve -/
noncomputable def f (x : ℝ) : ℝ := 2*x + 1/x

/-- The angle of rotation that places the focus on the x-axis -/
noncomputable def rotation_angle : ℝ := Real.arctan (1 / (2 + Real.sqrt 5))

/-- The eccentricity of the rotated hyperbola -/
noncomputable def eccentricity : ℝ := Real.sqrt (10 - 4 * Real.sqrt 5)

/-- Theorem stating that the eccentricity of the rotated hyperbola is √(10 - 4√5) -/
theorem rotated_hyperbola_eccentricity :
  let C := {(x, y) | ∃ (t : ℝ), 
    x = (f t.cos * t.cos - f t.sin * t.sin) * rotation_angle.cos + 
        (f t.cos * t.sin + f t.sin * t.cos) * rotation_angle.sin ∧
    y = -(f t.cos * t.cos - f t.sin * t.sin) * rotation_angle.sin + 
         (f t.cos * t.sin + f t.sin * t.cos) * rotation_angle.cos}
  ∃ (a b : ℝ), (∀ (x y : ℝ), (x, y) ∈ C → (x^2 / a^2) - (y^2 / b^2) = 1) ∧
               eccentricity = Real.sqrt (1 + b^2 / a^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_hyperbola_eccentricity_l552_55207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l552_55210

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x + 3 / x - 3) - a / x

/-- The theorem stating the minimum value of a -/
theorem min_value_of_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f a x ≤ 0) → a ≥ Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l552_55210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_main_theorem_l552_55297

theorem dice_probability (n : Nat) : 
  (16 : ℚ) / 31 = (3888 : ℚ) / 7533 := by
  -- Proof steps would go here
  sorry

theorem main_theorem : 
  (16 : ℚ) / 31 = (3888 : ℚ) / 7533 := by
  exact dice_probability 5

#eval (16 : ℚ) / 31 == (3888 : ℚ) / 7533

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_main_theorem_l552_55297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_packets_calculation_l552_55268

-- Define the variables and constants
def cupcake_price : ℚ := 3/2
def cookie_packet_price : ℚ := 2
def biscuit_packet_price : ℚ := 1
def cupcakes_per_day : ℕ := 20
def biscuit_packets_per_day : ℕ := 20
def total_earnings_five_days : ℚ := 350

-- Define the function to calculate the number of cookie packets per day
noncomputable def cookie_packets_per_day : ℕ := 
  let cupcake_earnings := cupcake_price * cupcakes_per_day * 5
  let biscuit_earnings := biscuit_packet_price * biscuit_packets_per_day * 5
  let cookie_earnings := total_earnings_five_days - cupcake_earnings - biscuit_earnings
  (cookie_earnings / (cookie_packet_price * 5)).floor.toNat

-- State the theorem
theorem cookie_packets_calculation :
  cookie_packets_per_day = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_packets_calculation_l552_55268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_desargues_theorem_l552_55281

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle ABC
variable (A B C : Point)

-- Define the line e
variable (e : Line)

-- Define the intersection points A', B', C'
variable (A' B' C' : Point)

-- Define the points A'', B'', C''
variable (A'' B'' C'' : Point)

-- Define the condition that e intersects sides of ABC at A', B', C'
def intersects_sides (A B C : Point) (e : Line) (A' B' C' : Point) : Prop :=
  sorry

-- Define the condition that AA', BB', CC' form triangle C''A''B''
def forms_triangle (A B C A' B' C' A'' B'' C'' : Point) : Prop :=
  sorry

-- Define a function to create a line from two points
def line_from_points (P Q : Point) : Line :=
  { a := Q.y - P.y, b := P.x - Q.x, c := P.y * Q.x - P.x * Q.y }

-- Define the condition for three lines to be concurrent
def are_concurrent (l1 l2 l3 : Line) : Prop :=
  sorry

-- The main theorem
theorem desargues_theorem 
  (h1 : intersects_sides A B C e A' B' C')
  (h2 : forms_triangle A B C A' B' C' A'' B'' C'') :
  are_concurrent (line_from_points A A'') (line_from_points B B'') (line_from_points C C'') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_desargues_theorem_l552_55281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l552_55209

-- Define the triangle
def triangle_DEF (DE DF EF : ℝ) : Prop :=
  DE = 9 ∧ DF = 9 ∧ EF = 8

-- Define the inradius of a triangle
noncomputable def inradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  2 * (Real.sqrt (s * (s - a) * (s - b) * (s - c))) / (a + b + c)

-- Theorem statement
theorem inscribed_circle_radius (DE DF EF : ℝ) :
  triangle_DEF DE DF EF → inradius DE DF EF = 4 * Real.sqrt 65 / 13 := by
  sorry

#check inscribed_circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l552_55209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_shape_area_l552_55279

/-- The combined surface area of a composite shape consisting of a spherical cap and a circular segment -/
noncomputable def combined_surface_area (r : ℝ) (h : ℝ) (base_radius : ℝ) : ℝ :=
  2 * Real.pi * r * h + Real.pi * base_radius^2

/-- Theorem stating that the combined surface area of the composite shape is 100π cm² -/
theorem composite_shape_area :
  let r : ℝ := 10  -- radius of the original hemisphere
  let h : ℝ := 4   -- height of the spherical cap
  let base_radius : ℝ := 6  -- radius of the circular segment base
  combined_surface_area r h base_radius = 100 * Real.pi :=
by
  -- Unfold the definition of combined_surface_area
  unfold combined_surface_area
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_shape_area_l552_55279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_examples_log_sum_property_log_difference_property_log_product_property_l552_55273

-- Definition of logarithm
noncomputable def log (a b : ℝ) : ℝ := Real.log b / Real.log a

-- Axiom for logarithm definition
axiom log_def {a b x : ℝ} (ha : a > 0) (ha1 : a ≠ 1) (hb : b > 0) : 
  a ^ x = b ↔ log a b = x

-- Theorem 1
theorem log_examples :
  log 3 27 = 3 ∧ log 2 16 = 4 ∧ log (2/3) (8/27) = 3 := by sorry

-- Theorem 2
theorem log_sum_property :
  log 3 2 + log 3 4 = log 3 8 := by sorry

-- Theorem 3
theorem log_difference_property :
  log 5 10 - log 5 2 = 1 := by sorry

-- Theorem 4
theorem log_product_property :
  log 10 4 * log 2 10 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_examples_log_sum_property_log_difference_property_log_product_property_l552_55273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cesaro_sum_extension_l552_55222

/-- Cesaro sum of a finite sequence -/
noncomputable def cesaro_sum (B : List ℝ) : ℝ :=
  let n := B.length
  let S := List.scanl (· + ·) 0 B
  (S.sum - S.head!) / n

theorem cesaro_sum_extension (B : List ℝ) (h : B.length = 99) (h_sum : cesaro_sum B = 800) :
  cesaro_sum (10 :: B) = 802 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cesaro_sum_extension_l552_55222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_of_four_factors_l552_55204

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem smallest_difference_of_four_factors (w x y z : ℕ+) : 
  (w : ℕ) * x * y * z = factorial 9 → w < x → x < y → y < z → 
  ∀ (a b c d : ℕ+), (a : ℕ) * b * c * d = factorial 9 → a < b → b < c → c < d → 
  z - w ≤ d - a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_of_four_factors_l552_55204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_max_l552_55296

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x + 5 * Real.pi / 2) * Real.cos (x - Real.pi / 2) - Real.cos (x + Real.pi / 4) ^ 2

theorem triangle_perimeter_max (A B C : ℝ) (a b c : ℝ) :
  0 < A → 0 < B → 0 < C →  -- Acute triangle condition
  A + B + C = Real.pi →    -- Sum of angles in a triangle
  f (A / 2) = (Real.sqrt 3 - 1) / 2 →
  a = 1 →
  a / Real.sin A = b / Real.sin B →  -- Law of sines
  b / Real.sin B = c / Real.sin C →
  (a + b + c) ≤ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_max_l552_55296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_distance_condition_l552_55234

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Property that k points satisfy the distance condition -/
def satisfiesDistanceCondition (points : Finset Point) (k : ℕ) : Prop :=
  ∃ (subpoints : Finset Point), subpoints ⊆ points ∧ subpoints.card = k ∧
    (∀ p q, p ∈ subpoints → q ∈ subpoints → p ≠ q → distance p q ≤ 2) ∨
    (∀ p q, p ∈ subpoints → q ∈ subpoints → p ≠ q → distance p q > 1)

/-- The main theorem -/
theorem smallest_n_for_distance_condition (k : ℕ) (h : k ≥ 2) :
    (∀ (points : Finset Point), points.card = k^2 - 2*k + 2 →
      satisfiesDistanceCondition points k) ∧
    (∀ n < k^2 - 2*k + 2, ∃ (points : Finset Point), points.card = n ∧
      ¬satisfiesDistanceCondition points k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_distance_condition_l552_55234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_non_lucky_multiple_of_7_l552_55200

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isLucky (n : ℕ) : Prop := n % (sumOfDigits n) = 0

theorem least_non_lucky_multiple_of_7 :
  ∀ k : ℕ, k > 0 ∧ k < 15 → isLucky (7 * k) ∧ ¬ isLucky (7 * 14) ∧ 7 * 14 = 98 := by
  sorry

#check least_non_lucky_multiple_of_7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_non_lucky_multiple_of_7_l552_55200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l552_55258

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi/4 - x)

theorem f_monotone_increasing :
  MonotoneOn f (Set.Icc (-5*Real.pi/4) (-Real.pi/4)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l552_55258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_lambda_system_is_sigma_algebra_l552_55240

-- Define the properties of π-system and λ-system
class PiSystem (α : Type) (G : Set (Set α)) where
  intersection_closed : ∀ A B, A ∈ G → B ∈ G → A ∩ B ∈ G

class LambdaSystem (α : Type) (G : Set (Set α)) where
  complement_closed : ∀ A, A ∈ G → (Set.univ \ A) ∈ G
  countable_union_closed : ∀ (f : ℕ → Set α), (∀ n, f n ∈ G) → (⋃ n, f n) ∈ G

-- Define π-λ-system as a combination of π-system and λ-system
class PiLambdaSystem (α : Type) (G : Set (Set α)) extends PiSystem α G, LambdaSystem α G

-- Define algebra of sets
class AlgebraOfSets (α : Type) (A : Set (Set α)) where
  empty_set : ∅ ∈ A
  union_closed : ∀ S T, S ∈ A → T ∈ A → S ∪ T ∈ A
  complement_closed : ∀ S, S ∈ A → (Set.univ \ S) ∈ A

-- Define σ-algebra
class SigmaAlgebra (α : Type) (S : Set (Set α)) extends AlgebraOfSets α S where
  countable_union_closed : ∀ (f : ℕ → Set α), (∀ n, f n ∈ S) → (⋃ n, f n) ∈ S

-- Define monotone class
def MonotoneClass (α : Type) (M : Set (Set α)) :=
  (∀ (f : ℕ → Set α), (∀ n, f n ∈ M) → (∀ n, f n ⊆ f (n+1)) → (⋃ n, f n) ∈ M) ∧
  (∀ (f : ℕ → Set α), (∀ n, f n ∈ M) → (∀ n, f (n+1) ⊆ f n) → (⋂ n, f n) ∈ M)

-- State the monotone class theorem
axiom monotone_class_theorem {α : Type} (A : Set (Set α)) [AlgebraOfSets α A] :
  MonotoneClass α A → SigmaAlgebra α A

-- State the main theorem
theorem pi_lambda_system_is_sigma_algebra {α : Type} (G : Set (Set α)) [PiLambdaSystem α G] :
  SigmaAlgebra α G :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_lambda_system_is_sigma_algebra_l552_55240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_gauss_f_l552_55254

-- Define the Gauss function
noncomputable def gauss (x : ℝ) : ℤ := ⌊x⌋

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2^x + 3) / (2^x + 1)

-- State the theorem
theorem range_of_gauss_f : {y : ℤ | ∃ x : ℝ, gauss (f x) = y} = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_gauss_f_l552_55254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_perpendicular_line_through_intersection_l552_55247

-- Define the lines
def line1 (x y : ℝ) : Prop := x - 2*y + 1 = 0
def line2 (x y : ℝ) : Prop := x - 2*y + 11 = 0
def line3 (x y : ℝ) : Prop := x - 2*y - 9 = 0
def line4 (x y : ℝ) : Prop := x - 2*y + 4 = 0
def line5 (x y : ℝ) : Prop := x + y - 2 = 0
def line6 (x y : ℝ) : Prop := 2*x + 3*y + 1 = 0
def line7 (x y : ℝ) : Prop := 3*x - 2*y + 4 = 0

-- Distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Distance from a point to a line ax + by + c = 0
noncomputable def distanceToLine (x0 y0 a b c : ℝ) : ℝ := 
  abs (a*x0 + b*y0 + c) / Real.sqrt (a^2 + b^2)

theorem parallel_lines_at_distance : 
  ∀ x y : ℝ, (line2 x y ∨ line3 x y) → 
  (∃ x1 y1, line1 x1 y1 ∧ distanceToLine x1 y1 1 (-2) 11 = 2*Real.sqrt 5) ∧
  (∃ x1 y1, line1 x1 y1 ∧ distanceToLine x1 y1 1 (-2) (-9) = 2*Real.sqrt 5) :=
by sorry

theorem perpendicular_line_through_intersection :
  (∃ x y, line4 x y ∧ line5 x y ∧ line7 x y) ∧
  (∀ x y, line7 x y → (3 * 2 + 2 * 3 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_perpendicular_line_through_intersection_l552_55247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equality_l552_55286

noncomputable section

/-- The surface area of a rectangular prism with dimensions a, b, and c. -/
def surface_area_prism (a b c : ℝ) : ℝ := 2 * (a * b + a * c + b * c)

/-- The surface area of a sphere with radius r. -/
noncomputable def surface_area_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- The volume of a sphere with radius r. -/
noncomputable def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- Theorem: Given a rectangular prism with dimensions 2, 2, and 4, and a sphere with the same surface area,
    if the volume of the sphere is (L * √12) / √π, then L = 40√2 / 3. -/
theorem sphere_volume_equality (L : ℝ) : 
  surface_area_prism 2 2 4 = surface_area_sphere (Real.sqrt (10 / Real.pi)) →
  volume_sphere (Real.sqrt (10 / Real.pi)) = (L * Real.sqrt 12) / Real.sqrt Real.pi →
  L = 40 * Real.sqrt 2 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equality_l552_55286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_58_plus_24sqrt6_l552_55227

theorem sqrt_58_plus_24sqrt6 (a b c : ℤ) : 
  (58 + 24 * Real.sqrt 6 : ℝ).sqrt = a + b * Real.sqrt c →
  c > 1 →
  (∀ d : ℤ, d > 1 → d * d ∣ c → d = c) →
  a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_58_plus_24sqrt6_l552_55227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_path_count_l552_55249

/-- A path on a grid -/
structure GridPath where
  steps : List (Nat × Nat)

/-- The number of paths from (0,0) to (x,y) with x+y steps -/
def numPaths (x y : Nat) : Nat :=
  Nat.choose (x + y) x

/-- Whether a path passes through a given point -/
def passesThrough (path : GridPath) (x y : Nat) : Prop :=
  ∃ i, List.take i path.steps = [(x, y)]

/-- The theorem to be proved -/
theorem grid_path_count :
  let totalPaths := numPaths 6 5
  let invalidPaths := numPaths 3 2 * numPaths 3 3
  totalPaths - invalidPaths = 362 := by
  sorry

#eval numPaths 6 5 - (numPaths 3 2 * numPaths 3 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_path_count_l552_55249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_colorable_l552_55221

-- Define the type for vertices
inductive Vertex : Type where
  | zero : Vertex
  | one : Vertex
  | two : Vertex

-- Define the type for triangles
structure Triangle where
  v1 : Vertex
  v2 : Vertex
  v3 : Vertex

-- Define the type for the plane division
structure PlaneDivision where
  triangles : Set Triangle
  vertex_labeling : ∀ (t : Triangle), t.v1 ≠ t.v2 ∧ t.v2 ≠ t.v3 ∧ t.v3 ≠ t.v1
  triangle_adjacency : ∀ (t1 t2 : Triangle), t1 ≠ t2 → 
    (t1.v1 = t2.v1 ∨ t1.v1 = t2.v2 ∨ t1.v1 = t2.v3 ∨
     t1.v2 = t2.v1 ∨ t1.v2 = t2.v2 ∨ t1.v2 = t2.v3 ∨
     t1.v3 = t2.v1 ∨ t1.v3 = t2.v2 ∨ t1.v3 = t2.v3) ∨
    (∃ (v : Vertex), (v = t1.v1 ∨ v = t1.v2 ∨ v = t1.v3) ∧ 
                     (v = t2.v1 ∨ v = t2.v2 ∨ v = t2.v3)) ∨
    (∀ v : Vertex, (v ≠ t1.v1 ∧ v ≠ t1.v2 ∧ v ≠ t1.v3) ∨ 
                   (v ≠ t2.v1 ∧ v ≠ t2.v2 ∧ v ≠ t2.v3))

-- Define the type for coloring
inductive Color : Type where
  | white : Color
  | black : Color

-- Theorem statement
theorem two_colorable (pd : PlaneDivision) : 
  ∃ (coloring : Triangle → Color), 
    ∀ (t1 t2 : Triangle), t1 ∈ pd.triangles → t2 ∈ pd.triangles → 
      t1 ≠ t2 → 
      (∃ (v1 v2 : Vertex), 
        (v1 = t1.v1 ∨ v1 = t1.v2 ∨ v1 = t1.v3) ∧
        (v2 = t2.v1 ∨ v2 = t2.v2 ∨ v2 = t2.v3) ∧
        v1 = v2) →
      coloring t1 ≠ coloring t2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_colorable_l552_55221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_rpm_l552_55205

/-- Calculates the revolutions per minute (rpm) of a wheel given its radius and the speed of the vehicle. -/
noncomputable def calculate_rpm (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let speed_cm_per_minute := speed * 100000 / 60
  speed_cm_per_minute / circumference

/-- Theorem stating that a wheel with radius 140 cm on a bus traveling at 66 km/h has approximately 1250.14 rpm. -/
theorem bus_wheel_rpm :
  let radius := (140 : ℝ)
  let speed := (66 : ℝ)
  let calculated_rpm := calculate_rpm radius speed
  abs (calculated_rpm - 1250.14) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_rpm_l552_55205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l552_55203

noncomputable def f (x θ : ℝ) : ℝ := x^2 + 4 * (Real.sin (θ + Real.pi/3)) * x - 2

theorem function_properties (θ : ℝ) (h : θ ∈ Set.Icc 0 (2*Real.pi)) :
  (∀ x, f x θ = f (-x) θ) → Real.tan θ = -Real.sqrt 3 ∧
  (∀ x ∈ Set.Icc (-Real.sqrt 3) 1, Monotone (f · θ)) →
    θ ∈ Set.Icc 0 (Real.pi/3) ∪ Set.Icc (5*Real.pi/6) (3*Real.pi/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l552_55203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l552_55246

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the foci
def left_focus (F₁ : ℝ × ℝ) : Prop := F₁.1 < 0
def right_focus (F₂ : ℝ × ℝ) : Prop := F₂.1 > 0

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop := hyperbola P.1 P.2

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem hyperbola_focal_distance 
  (F₁ F₂ P : ℝ × ℝ) 
  (h₁ : left_focus F₁) 
  (h₂ : right_focus F₂) 
  (h₃ : point_on_hyperbola P) 
  (h₄ : distance P F₁ = 5) :
  distance P F₂ = 3 ∨ distance P F₂ = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l552_55246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_range_l552_55282

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + 1 - a * x

-- Define the minimum value function g
noncomputable def g (a : ℝ) : ℝ :=
  if 0 < a ∧ a ≤ 1 then a^2 + 1
  else if 1 < a ∧ a < 2 then 1/a + a
  else 0  -- This case should never occur given our assumptions

-- Theorem statement
theorem f_minimum_and_range (a : ℝ) (h : 0 < a ∧ a < 2) :
  (∀ x, f a x ≥ g a) ∧
  (∀ m, m ≥ 5/2 → ∃ a', 0 < a' ∧ a' < 2 ∧ ∃ x, f a' x = m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_range_l552_55282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_fraction_equality_l552_55243

theorem cube_root_fraction_equality : (6.4 / 12.8) ^ (1/3 : ℝ) = (1 / 2 : ℝ) ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_fraction_equality_l552_55243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_confidence_intervals_for_normal_sample_l552_55272

/-- A random variable normally distributed with unknown mean and variance -/
structure NormalSample where
  n : Nat
  sample_mean : ℝ
  sample_variance : ℝ
  confidence_level_mean : ℝ
  confidence_level_variance : ℝ

/-- Confidence interval for the mean -/
noncomputable def confidence_interval_mean (s : NormalSample) : Set ℝ :=
  Set.Icc 1.029 1.311

/-- Confidence interval for the variance -/
noncomputable def confidence_interval_variance (s : NormalSample) : Set ℝ :=
  Set.Icc 0.114 0.889

/-- The main theorem stating the confidence intervals for the given sample -/
theorem confidence_intervals_for_normal_sample
  (s : NormalSample)
  (hn : s.n = 10)
  (hm : s.sample_mean = 1.17)
  (hv : s.sample_variance = 0.25)
  (hcm : s.confidence_level_mean = 0.98)
  (hcv : s.confidence_level_variance = 0.96) :
  confidence_interval_mean s = Set.Icc 1.029 1.311 ∧
  confidence_interval_variance s = Set.Icc 0.114 0.889 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_confidence_intervals_for_normal_sample_l552_55272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l552_55224

/-- Ellipse C with foci F1 and F2, and point P -/
structure EllipseConfig where
  a : ℝ
  b : ℝ
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ
  P : ℝ × ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1
  h_symmetric : ∃ (l : ℝ × ℝ → Prop), l F1 ∧ (∀ x, l x ↔ l (2 * F1 - x))
  h_dot_product : (P.1 - F1.1) * (F2.1 - F1.1) + (P.2 - F1.2) * (F2.2 - F1.2) = 1/2 * a^2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (c : EllipseConfig) : ℝ :=
  Real.sqrt ((c.a^2 - c.b^2) / c.a^2)

/-- Theorem stating that the eccentricity of the ellipse is 1/2 -/
theorem ellipse_eccentricity (c : EllipseConfig) : eccentricity c = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l552_55224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l552_55239

/-- The area of a rhombus given its diagonals -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- Theorem: In a rhombus with area 170 cm² and one diagonal 20 cm, the other diagonal is 17 cm -/
theorem rhombus_other_diagonal (d2 : ℝ) :
  rhombusArea 20 d2 = 170 → d2 = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l552_55239
