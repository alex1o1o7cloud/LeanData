import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equality_condition_l1241_124135

theorem cosine_equality_condition (x y : ℝ) : 
  (x = y → Real.cos x = Real.cos y) ∧ 
  ∃ a b : ℝ, Real.cos a = Real.cos b ∧ a ≠ b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equality_condition_l1241_124135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_equals_three_l1241_124178

/-- The displacement function for a point mass moving in a straight line -/
noncomputable def displacement (t : ℝ) : ℝ := t^2 + 1

/-- The average speed of the point mass over a time interval -/
noncomputable def averageSpeed (t1 t2 : ℝ) : ℝ :=
  (displacement t2 - displacement t1) / (t2 - t1)

/-- Theorem: The average speed of the point mass over the time interval [1, 2] is 3 -/
theorem average_speed_equals_three :
  averageSpeed 1 2 = 3 := by
  -- Unfold the definitions
  unfold averageSpeed displacement
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_equals_three_l1241_124178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_final_results_l1241_124101

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The distance from the center to a focus of an ellipse -/
noncomputable def Ellipse.focalDistance (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

/-- The condition that |AB| = √3|OF| for the ellipse -/
def Ellipse.abOfCondition (e : Ellipse) : Prop :=
  Real.sqrt (e.a^2 + e.b^2) = Real.sqrt 3 * e.focalDistance

/-- The condition that the area of triangle AOB is √2 -/
def Ellipse.aobAreaCondition (e : Ellipse) : Prop :=
  (1/2) * e.a * e.b = Real.sqrt 2

/-- The main theorem about the ellipse -/
theorem ellipse_theorem (e : Ellipse) 
  (h_ab : e.abOfCondition) 
  (h_area : e.aobAreaCondition) : 
  (e.a = 2 ∧ e.b = Real.sqrt 2) ∧ 
  (∃ (x : ℝ), x = Real.sqrt 2 ∧ 
    (∀ (y : ℝ), y ≠ x ∧ y ≠ -x → 
      ¬(∃ (k₁ k₂ : ℝ), 
        ((y^2 - 4) * k₁^2 - 4 * y * k₁ + 2 = 0) ∧
        ((y^2 - 4) * k₂^2 - 4 * y * k₂ + 2 = 0) ∧
        k₁ * k₂ = -1))) := by
  sorry

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) : Prop :=
  ∀ (x y : ℝ), x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The points satisfying the perpendicular tangents condition -/
def perpendicular_tangent_points (e : Ellipse) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 2 ∧ (p.1 = Real.sqrt 2 ∨ p.1 = -Real.sqrt 2)}

/-- Theorem stating the final results -/
theorem final_results (e : Ellipse)
  (h_ab : e.abOfCondition)
  (h_area : e.aobAreaCondition) :
  ellipse_equation e ∧ perpendicular_tangent_points e = {(Real.sqrt 2, 2), (-Real.sqrt 2, 2)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_final_results_l1241_124101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_price_70_cans_l1241_124194

/-- Calculates the price of a given number of soda cans with a discount applied to 24-can cases -/
noncomputable def discounted_soda_price (regular_price : ℚ) (discount_percent : ℚ) (num_cans : ℕ) : ℚ :=
  let discounted_price := regular_price * (1 - discount_percent / 100)
  let full_cases := num_cans / 24
  let remaining_cans := num_cans % 24
  (full_cases * 24 + remaining_cans) * discounted_price

/-- Theorem stating that the price of 70 cans of soda with the given conditions is $28.875 -/
theorem soda_price_70_cans :
  discounted_soda_price (55/100) 25 70 = 28875/1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_price_70_cans_l1241_124194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gender_understanding_related_probability_three_out_of_five_understand_l1241_124189

-- Define the sample data
def total_sample : Nat := 400
def female_understand : Nat := 140
def female_not_understand : Nat := 60
def male_understand : Nat := 180
def male_not_understand : Nat := 20

-- Define the chi-square formula
noncomputable def chi_square (a b c d : Nat) : Real :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : Real) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value
def critical_value : Real := 10.828

-- Define the theorem for the relationship between gender and understanding
theorem gender_understanding_related :
  chi_square female_understand female_not_understand male_understand male_not_understand > critical_value := by
  sorry

-- Define the probability of understanding based on the sample
noncomputable def prob_understand : Real := 320 / 400

-- Define the theorem for the probability of exactly 3 out of 5 understanding
theorem probability_three_out_of_five_understand :
  (Nat.choose 5 3 : Real) * prob_understand^3 * (1 - prob_understand)^2 = 128 / 625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gender_understanding_related_probability_three_out_of_five_understand_l1241_124189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_params_l1241_124124

/-- A function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- ExtremeValue predicate -/
def ExtremeValue (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f x ≤ f y ∨ f x ≥ f y

/-- The theorem stating that if f(x) has an extreme value of 10 at x = 1,
    then a = 4 and b = -11 -/
theorem extreme_value_implies_params
  (a b : ℝ)
  (h1 : f a b 1 = 10)
  (h2 : ExtremeValue (f a b) 1) :
  a = 4 ∧ b = -11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_params_l1241_124124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_triangle_count_is_4200_l1241_124154

/-- A circle with 10 points and chords connecting every pair of points -/
structure CircleWithChords where
  points : Finset (ℝ × ℝ)
  is_on_circle : ∀ p ∈ points, p.1^2 + p.2^2 = 1
  point_count : points.card = 10

/-- No three chords intersect in a single point inside the circle -/
axiom no_triple_intersection (c : CircleWithChords) :
  ∀ p q r : ℝ × ℝ, p ∈ c.points → q ∈ c.points → r ∈ c.points →
    p ≠ q → q ≠ r → p ≠ r →
    ¬∃ x : ℝ × ℝ, x.1^2 + x.2^2 < 1 ∧
      (∃ a b d : ℝ, x = a • p + (1 - a) • q ∧
                    x = b • q + (1 - b) • r ∧
                    x = d • r + (1 - d) • p)

/-- The number of triangles with all three vertices in the interior of the circle -/
def interior_triangle_count (c : CircleWithChords) : ℕ := sorry

/-- The main theorem: There are 4200 interior triangles -/
theorem interior_triangle_count_is_4200 (c : CircleWithChords) :
  interior_triangle_count c = 4200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_triangle_count_is_4200_l1241_124154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_area_ratio_l1241_124121

/-- Represents a triangle -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- The area of a triangle -/
noncomputable def Triangle.area (t : Triangle) : ℝ := (1 / 2) * t.base * t.height

/-- Applies the oblique projection to a triangle -/
noncomputable def oblique_projection (t : Triangle) : Triangle :=
  { base := t.base,
    height := (1 / 2) * t.height * Real.sqrt 2 / 2 }

/-- The theorem stating the relationship between the areas of the original and projected triangles -/
theorem oblique_projection_area_ratio (t : Triangle) :
  (oblique_projection t).area = (Real.sqrt 2 / 4) * t.area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_area_ratio_l1241_124121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ana_age_l1241_124112

/-- Ana's age this year -/
def A : ℕ := sorry

/-- Bonita's age this year -/
def B : ℕ := sorry

/-- Number of years Ana and Bonita were born apart -/
def n : ℕ := sorry

/-- Ana's age is n years more than Bonita's -/
axiom age_difference : A = B + n

/-- Last year, Ana was 7 times as old as Bonita -/
axiom last_year : A - 1 = 7 * (B - 1)

/-- This year, Ana's age is four times Bonita's age -/
axiom this_year : A = 4 * B

theorem ana_age : A = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ana_age_l1241_124112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1241_124102

noncomputable def f (a b x : ℝ) : ℝ := -2 * a * Real.sin (2 * x + Real.pi / 6) + 2 * a + b

noncomputable def g (a b x : ℝ) : ℝ := f a b (x + Real.pi / 2)

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f x < f y

def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f y < f x

theorem function_properties (a b : ℝ) :
  (a > 0) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → -5 ≤ f a b x ∧ f a b x ≤ 1) →
  (∀ x : ℝ, Real.log (g a b x) > 0) →
  (a = 2 ∧ b = -5) ∧
  (∀ k : ℤ, is_increasing (g a b) (k * Real.pi) (k * Real.pi + Real.pi / 6)) ∧
  (∀ k : ℤ, is_decreasing (g a b) (k * Real.pi + Real.pi / 6) (k * Real.pi + Real.pi / 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1241_124102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l1241_124148

theorem sqrt_equation_solution (a b : ℕ+) :
  (Real.sqrt 30 - Real.sqrt 18) * (3 * Real.sqrt a.val + Real.sqrt b.val) = 12 →
  a = 2 ∧ b = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l1241_124148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_b_values_l1241_124105

-- Define the piecewise function f
noncomputable def f (n : ℝ) : ℝ :=
  if n < 1 then n^2 - 6 else 3*n - 15

-- Theorem statement
theorem difference_of_b_values (b₁ b₂ : ℝ) :
  (f (-1) + f 1 + f b₁ = 0) →
  (f (-1) + f 1 + f b₂ = 0) →
  b₁ ≠ b₂ →
  |b₁ - b₂| = Real.sqrt 23 + 32/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_b_values_l1241_124105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1241_124175

/-- Given an ellipse (C) with foci at (±√5, 0) and vertices at (±3, 0), 
    prove that its equation is x²/9 + y²/4 = 1 -/
theorem ellipse_equation (C : Set (ℝ × ℝ)) 
  (h_foci : Set.range (λ t : ℝ ↦ (t * Real.sqrt 5, 0)) ∩ C = {(Real.sqrt 5, 0), (-Real.sqrt 5, 0)})
  (h_vertices : Set.range (λ t : ℝ ↦ (3 * t, 0)) ∩ C = {(3, 0), (-3, 0)}) :
  C = {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / 4 = 1} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1241_124175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_ratio_l1241_124156

/-- Square with side length 2 -/
def Square : Set (ℝ × ℝ) :=
  {p | (0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2) ∧ (p.1 = 0 ∨ p.1 = 2 ∨ p.2 = 0 ∨ p.2 = 2)}

/-- Path of a particle moving on the square perimeter -/
noncomputable def ParticlePath (start : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  sorry

/-- Midpoint of two points -/
noncomputable def Midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

/-- Path traced by the midpoint of the segment connecting two particles -/
noncomputable def MidpointPath (t : ℝ) : ℝ × ℝ :=
  Midpoint (ParticlePath (0, 0) t) (ParticlePath (1, 2) t)

/-- Area enclosed by the midpoint path -/
noncomputable def EnclosedArea : ℝ :=
  sorry

/-- Theorem: The ratio of the enclosed area to the square area is 1/2 -/
theorem enclosed_area_ratio :
  EnclosedArea / 4 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_ratio_l1241_124156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1241_124109

def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

theorem quadratic_function_properties :
  ∀ a b : ℝ,
  (f a b (-1) = 0 ∧ f a b 3 = 0) →
  (∀ x : ℝ, f a b x = -x^2 + 2*x + 3) ∧
  ({x : ℝ | f a b x ≤ 3} = {x : ℝ | x ≤ 0 ∨ x ≥ 2}) ∧
  (∀ x : ℝ, -1 ≤ f a b (Real.sin x) ∧ f a b (Real.sin x) ≤ 4) ∧
  (∃ x y : ℝ, f a b (Real.sin x) = 0 ∧ f a b (Real.sin y) = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1241_124109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_purple_cards_l1241_124145

/-- Represents the number of cards of each color -/
structure CardCount where
  red : ℕ
  blue : ℕ
  purple : ℕ

/-- Represents the exchange rules -/
inductive Exchange
  | RedToBlueAndPurple : Exchange
  | BlueToRedAndPurple : Exchange

/-- Applies an exchange to a card count -/
def applyExchange (cc : CardCount) (e : Exchange) : Option CardCount :=
  match e with
  | Exchange.RedToBlueAndPurple =>
      if cc.red ≥ 2 then some ⟨cc.red - 2, cc.blue + 1, cc.purple + 1⟩ else none
  | Exchange.BlueToRedAndPurple =>
      if cc.blue ≥ 3 then some ⟨cc.red + 1, cc.blue - 3, cc.purple + 1⟩ else none

/-- The initial card count -/
def initialCards : CardCount := ⟨100, 100, 0⟩

/-- Applies a list of exchanges to a card count -/
def applyExchanges (cc : CardCount) (exchanges : List Exchange) : CardCount :=
  exchanges.foldl (fun acc e => (applyExchange acc e).getD acc) cc

/-- Theorem stating the maximum number of purple cards obtainable -/
theorem max_purple_cards :
  ∃ (final : CardCount) (exchanges : List Exchange),
    (applyExchanges initialCards exchanges = final) ∧
    (∀ other : CardCount,
      (∃ otherExchanges : List Exchange,
        applyExchanges initialCards otherExchanges = other) →
      other.purple ≤ final.purple) ∧
    final.purple = 138 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_purple_cards_l1241_124145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_is_x_function_linear_is_x_function_quadratic_not_x_function_sin_cos_x_function_range_piecewise_x_function_and_increasing_l1241_124116

-- Definition of an X-function
def is_x_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) ≠ -f x

-- Statement 1
theorem exp_is_x_function :
  is_x_function (λ x => Real.exp x) := by sorry

-- Statement 2
theorem linear_is_x_function :
  is_x_function (λ x => x + 1) := by sorry

-- Statement 3
theorem quadratic_not_x_function :
  ¬ is_x_function (λ x => x^2 + 2*x - 3) := by sorry

-- Statement 4
theorem sin_cos_x_function_range (a : ℝ) :
  is_x_function (λ x => Real.sin x + Real.cos x + a) ↔ a < -1 ∨ a > 1 := by sorry

-- Statement 5
noncomputable def piecewise_function (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 1 else x

theorem piecewise_x_function_and_increasing :
  is_x_function piecewise_function ∧
  StrictMono piecewise_function := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_is_x_function_linear_is_x_function_quadratic_not_x_function_sin_cos_x_function_range_piecewise_x_function_and_increasing_l1241_124116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_problem_milk_problem_solution_l1241_124174

/-- Represents the problem of finding the initial amount of milk --/
theorem milk_problem (initial_milk : ℝ) : 
  (20 / 1.5 * initial_milk) / (initial_milk + 15) = 32 / 3 →
  initial_milk = 60 := by
  sorry

/-- Helper function to calculate the value per litre of the mixture --/
noncomputable def mixture_value_per_litre (initial_milk : ℝ) : ℝ :=
  (20 / 1.5 * initial_milk) / (initial_milk + 15)

/-- The main theorem stating the solution to the milk problem --/
theorem milk_problem_solution :
  ∃ (initial_milk : ℝ), 
    mixture_value_per_litre initial_milk = 32 / 3 ∧
    initial_milk = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_problem_milk_problem_solution_l1241_124174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_angle_l1241_124115

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the foci
def foci (c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-c, 0), (c, 0))

-- Define the line l
def line_l (a : ℝ) (x : ℝ) : Prop := x = a

-- Define the angle between three points
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem ellipse_max_angle (a b : ℝ) :
  a > b ∧ b > 0 →
  let c := 1
  let (f1, f2) := foci c
  let A := (a, 0)
  (∀ x y, ellipse a b x y → x ≤ a) →
  (∀ p : ℝ × ℝ, line_l a p.1 → angle f1 p f2 ≤ π/4) →
  ∃ p : ℝ × ℝ, line_l a p.1 ∧ angle f1 p f2 = π/4 ∧ (p = (Real.sqrt 2, 1) ∨ p = (Real.sqrt 2, -1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_angle_l1241_124115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_arrangement_l1241_124176

/-- The radius of a circle around which three congruent parabolas are arranged -/
noncomputable def circle_radius : ℝ := 1/4

/-- A parabola in the arrangement -/
def parabola (r : ℝ) : ℝ → ℝ := λ x ↦ x^2 + r

/-- The tangent line between two parabolas -/
def tangent_line : ℝ → ℝ := λ x ↦ x

theorem parabola_circle_arrangement (r : ℝ) :
  (∃ x : ℝ, parabola r x = tangent_line x) ∧  -- Parabola and line are tangent
  (∀ x : ℝ, parabola r x ≥ tangent_line x) ∧  -- Parabola is above or on the line
  (r > 0) →                                   -- Radius is positive
  r = circle_radius := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_arrangement_l1241_124176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_minimizes_cost_l1241_124104

/-- The annual purchase amount in tons -/
noncomputable def annual_purchase : ℝ := 600

/-- The shipping cost per purchase in yuan -/
noncomputable def shipping_cost : ℝ := 60000

/-- The storage cost coefficient in yuan per ton -/
noncomputable def storage_cost_coeff : ℝ := 4000

/-- The total cost function -/
noncomputable def total_cost (x : ℝ) : ℝ := (annual_purchase / x) * shipping_cost + storage_cost_coeff * x

/-- The optimal purchase amount that minimizes the total cost -/
noncomputable def optimal_purchase : ℝ := 30

theorem optimal_purchase_minimizes_cost :
  ∀ x > 0, total_cost optimal_purchase ≤ total_cost x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_minimizes_cost_l1241_124104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_floor_l1241_124123

theorem circle_area_floor : ⌊(π : Real) * 15^2⌋ = 706 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_floor_l1241_124123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_triangles_theorem_l1241_124195

-- Define the triangle ABC
variable (A B C : ℂ)

-- Define the external points P, Q, R
variable (P Q R : ℂ)

-- Define the angles given in the problem
noncomputable def angle_PBC : ℝ := Real.pi / 4  -- 45°
noncomputable def angle_CAQ : ℝ := Real.pi / 4  -- 45°
noncomputable def angle_BCP : ℝ := Real.pi / 6  -- 30°
noncomputable def angle_QCA : ℝ := Real.pi / 6  -- 30°
noncomputable def angle_ABR : ℝ := Real.pi / 12 -- 15°
noncomputable def angle_BAR : ℝ := Real.pi / 12 -- 15°

-- Define the conditions for the external triangles
def external_triangle_condition (A B C P Q R : ℂ) : Prop :=
  Complex.arg (P - B) - Complex.arg (C - B) = angle_PBC ∧
  Complex.arg (Q - C) - Complex.arg (A - C) = angle_CAQ ∧
  Complex.arg (C - B) - Complex.arg (P - B) = angle_BCP ∧
  Complex.arg (A - C) - Complex.arg (Q - C) = angle_QCA ∧
  Complex.arg (R - A) - Complex.arg (B - A) = angle_ABR ∧
  Complex.arg (B - A) - Complex.arg (R - A) = angle_BAR

-- State the theorem
theorem external_triangles_theorem (A B C P Q R : ℂ) 
  (h : external_triangle_condition A B C P Q R) :
  Complex.arg (Q - R) - Complex.arg (P - R) = Real.pi / 2 ∧ 
  Complex.abs (Q - R) = Complex.abs (P - R) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_triangles_theorem_l1241_124195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_projection_l1241_124117

/-- Given that the projection of (0,0) on a line l is (2,3), 
    prove that the equation of line l is 2x + 3y - 13 = 0 -/
theorem line_equation_from_projection 
  (l : Set (ℝ × ℝ)) 
  (h_proj : ∃ (proj : ℝ × ℝ → ℝ × ℝ), proj (0, 0) = (2, 3) ∧ ∀ p, proj p ∈ l) : 
  ∃ (a b c : ℝ), (a, b) ≠ (0, 0) ∧ 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ a * x + b * y + c = 0) ∧
  (a = 2 ∧ b = 3 ∧ c = -13) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_projection_l1241_124117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_equation_l1241_124196

-- Define the circles and points
def C1 (p : ℝ × ℝ) : Prop := (p.1 + 2)^2 + (p.2 - 3)^2 = 5
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (-1, 1)

-- Define the property of C2 intersecting with C1 at points A and B
def C2_intersects_C1 (C2 : (ℝ × ℝ) → Prop) : Prop :=
  C2 A ∧ C2 B ∧ C1 A ∧ C1 B

-- Define the parallelogram property
def is_parallelogram (C1_center C2_center : ℝ × ℝ) : Prop :=
  (C1_center.1 + C2_center.1) / 2 = (A.1 + B.1) / 2 ∧
  (C1_center.2 + C2_center.2) / 2 = (A.2 + B.2) / 2

-- Theorem statement
theorem C2_equation (C2 : (ℝ × ℝ) → Prop) (C2_center : ℝ × ℝ) :
  C2_intersects_C1 C2 →
  is_parallelogram (-2, 3) C2_center →
  (∀ p : ℝ × ℝ, C2 p ↔ (p.1 - 1)^2 + p.2^2 = 5) :=
by
  sorry

#check C2_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_equation_l1241_124196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_l1241_124134

theorem game_probability (alex_prob kate_prob jenna_prob : ℚ) : 
  alex_prob = 3/5 →
  kate_prob = 2 * jenna_prob →
  alex_prob + kate_prob + jenna_prob = 1 →
  (alex_prob^4 * kate_prob^3 * jenna_prob * (Nat.choose 8 4 * Nat.choose 4 3 : ℚ)) = 576/1500 := by
  sorry

#eval (Nat.choose 8 4 * Nat.choose 4 3 : ℚ)  -- This line is just to check if the multinomial coefficient is correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_l1241_124134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_parallel_to_given_line_l1241_124151

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 - 2*x + y^2 = 0

-- Define the parallel line equation
def parallel_line_eq (x y : ℝ) : Prop := x + 2*y = 0

-- Define the center of the circle
noncomputable def circle_center : ℝ × ℝ := (1, 0)

-- Define the slope of the parallel line
noncomputable def parallel_slope : ℝ := -1/2

-- Define the equation of the line we want to prove
def target_line_eq (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Theorem statement
theorem line_through_circle_center_parallel_to_given_line :
  ∀ x y : ℝ,
  circle_eq x y →
  parallel_line_eq x y →
  target_line_eq x y :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_parallel_to_given_line_l1241_124151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_iff_f_geq_half_implies_a_eq_half_l1241_124146

/-- The function f(x) = |x - 1| + a|x - 2| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + a * |x - 2|

/-- The function f(x) has a minimum value if and only if -1 ≤ a ≤ 1 -/
theorem f_has_minimum_iff (a : ℝ) : 
  (∃ m : ℝ, ∀ x : ℝ, m ≤ f a x) ↔ -1 ≤ a ∧ a ≤ 1 := by
  sorry

/-- If f(x) ≥ 1/2 for all x ∈ ℝ, then a = 1/2 -/
theorem f_geq_half_implies_a_eq_half (a : ℝ) :
  (∀ x : ℝ, (1/2 : ℝ) ≤ f a x) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_iff_f_geq_half_implies_a_eq_half_l1241_124146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_correct_l1241_124144

/-- The minimum area of a rectangle containing the figure determined by the given inequalities -/
noncomputable def min_area (a : ℝ) : ℝ :=
  if a ≤ 0 then (1 - a) * Real.sqrt (1 - a)
  else if 0 < a ∧ a < 1/2 then 1 - 2*a
  else 0

/-- The system of inequalities defining the figure -/
def figure (x y a : ℝ) : Prop :=
  y ≤ -x^2 ∧ y ≥ x^2 - 2*x + a

/-- Theorem stating that min_area gives the correct minimum area -/
theorem min_area_correct (a : ℝ) :
  ∀ S : ℝ, (∃ x₁ x₂ y₁ y₂ : ℝ,
    (∀ x y, figure x y a → x₁ ≤ x ∧ x ≤ x₂ ∧ y₁ ≤ y ∧ y ≤ y₂) ∧
    S = (x₂ - x₁) * (y₂ - y₁)) →
  S ≥ min_area a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_correct_l1241_124144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l1241_124141

-- Define the binary operation ⊗ as noncomputable
noncomputable def otimes (a b : ℝ) : ℝ := (a^2 + b^2) / (a^2 - b^2)

-- Theorem statement
theorem otimes_calculation : otimes (otimes 8 6) 2 = 821 / 429 := by
  -- Expand the definition of otimes
  unfold otimes
  -- Perform algebraic simplifications
  simp [pow_two]
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l1241_124141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1241_124126

noncomputable def z : ℂ := (2 - Complex.I) / (1 + Complex.I)

theorem z_in_fourth_quadrant : 
  z.re > 0 ∧ z.im < 0 :=
by
  -- Simplify the complex number
  have h : z = (1/2 : ℝ) - ((3/2 : ℝ) * Complex.I) := by
    -- Proof of this equality goes here
    sorry
  
  -- Show that the real part is positive
  have h_re : z.re > 0 := by
    -- Proof that the real part is 1/2 > 0 goes here
    sorry
  
  -- Show that the imaginary part is negative
  have h_im : z.im < 0 := by
    -- Proof that the imaginary part is -3/2 < 0 goes here
    sorry
  
  -- Combine the results
  exact ⟨h_re, h_im⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1241_124126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_park_area_l1241_124158

/-- A triangle in a city layout -/
structure CityTriangle where
  /-- Point A: origin -/
  A : ℝ × ℝ
  /-- Point B: 5 miles north of A -/
  B : ℝ × ℝ
  /-- Point E: 4 miles east of A -/
  E : ℝ × ℝ
  /-- B is 5 miles north of A -/
  hB : B.2 - A.2 = 5
  /-- E is 4 miles east of A -/
  hE : E.1 - A.1 = 4
  /-- Sunflower Street (AE) is perpendicular to Oak Road (AB) -/
  hPerp : (E.1 - A.1) * (B.1 - A.1) + (E.2 - A.2) * (B.2 - A.2) = 0

/-- The area of the triangular park is 10 square miles -/
theorem triangular_park_area (t : CityTriangle) : 
  abs ((t.B.1 - t.A.1) * (t.E.2 - t.A.2) - (t.E.1 - t.A.1) * (t.B.2 - t.A.2)) / 2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_park_area_l1241_124158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sine_equality_l1241_124187

theorem contrapositive_sine_equality (x y : ℝ) :
  (¬(Real.sin x = Real.sin y) → ¬(x = y)) ↔ (x = y → Real.sin x = Real.sin y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sine_equality_l1241_124187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1241_124142

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(3x+1)
def domain_f_3x_plus_1 : Set ℝ := Set.Icc (-2) 4

-- State the theorem
theorem domain_of_f (x : ℝ) : 
  (∀ x ∈ domain_f_3x_plus_1, f (3 * x + 1) = f (3 * x + 1)) →
  (f x = f x ↔ x ∈ Set.Icc (-5) 13) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1241_124142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_value_l1241_124153

/-- The central angle of the lateral surface of a cone with base radius 1 and height 2, when unfolded. -/
noncomputable def cone_central_angle : ℝ :=
  (2 * Real.sqrt 5 / 5) * Real.pi

/-- Theorem stating that the central angle of the lateral surface of a cone
    with base radius 1 and height 2, when unfolded, is (2√5/5)π radians. -/
theorem cone_central_angle_value (r h : ℝ) (hr : r = 1) (hh : h = 2) :
  cone_central_angle = (2 * Real.sqrt 5 / 5) * Real.pi :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_value_l1241_124153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_57_l1241_124103

-- Define the problem parameters
def initial_reading : ℕ := 34543
def final_reading : ℕ := 34943
def first_day_hours : ℕ := 3
def second_day_hours : ℕ := 4

-- Define the total distance and time
def total_distance : ℕ := final_reading - initial_reading
def total_time : ℕ := first_day_hours + second_day_hours

-- Define the average speed
noncomputable def average_speed : ℚ := (total_distance : ℚ) / total_time

-- Theorem statement
theorem average_speed_approx_57 :
  ∃ ε > 0, |((average_speed : ℝ) - 57)| < ε := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_57_l1241_124103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_in_cones_l1241_124107

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone : Cone
  intersectionDistance : ℝ

/-- Calculates the maximum squared radius of a sphere within two intersecting cones -/
noncomputable def maxSphereRadiusSquared (ic : IntersectingCones) : ℝ :=
  let hypotenuse := Real.sqrt (ic.cone.baseRadius^2 + ic.cone.height^2)
  let remainingHeight := hypotenuse - ic.intersectionDistance
  (ic.cone.baseRadius * remainingHeight / ic.cone.height)^2

/-- The main theorem stating the maximum squared radius of the sphere -/
theorem max_sphere_radius_squared_in_cones :
  let ic : IntersectingCones := {
    cone := { baseRadius := 5, height := 12 },
    intersectionDistance := 4
  }
  maxSphereRadiusSquared ic = 2025 / 169 := by
  sorry

#eval (2025 : Nat) + 169

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_in_cones_l1241_124107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_february_first_is_friday_l1241_124150

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | .Sunday => .Monday
  | .Monday => .Tuesday
  | .Tuesday => .Wednesday
  | .Wednesday => .Thursday
  | .Thursday => .Friday
  | .Friday => .Saturday
  | .Saturday => .Sunday

/-- Function to get the day of the week n days after a given day -/
def daysAfter (d : DayOfWeek) : ℕ → DayOfWeek
  | 0 => d
  | n + 1 => nextDay (daysAfter d n)

/-- Function to get the day of the week n days before a given day -/
def daysBefore (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
  daysAfter d ((7 - (n % 7)) % 7)

theorem february_first_is_friday (h : DayOfWeek.Wednesday = daysBefore DayOfWeek.Wednesday 12) :
  DayOfWeek.Friday = daysBefore DayOfWeek.Wednesday 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_february_first_is_friday_l1241_124150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disc_thickness_from_sphere_l1241_124188

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of a cylinder (disc) with radius r and height h -/
noncomputable def disc_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The thickness of a disc formed from a sphere with constant volume -/
theorem disc_thickness_from_sphere (r_sphere r_disc : ℝ) (h_pos : 0 < r_sphere) (d_pos : 0 < r_disc) :
  let h_disc := sphere_volume r_sphere / (Real.pi * r_disc^2)
  sphere_volume r_sphere = disc_volume r_disc h_disc ∧ h_disc = 9 / 25 := by
  sorry

#check disc_thickness_from_sphere 3 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disc_thickness_from_sphere_l1241_124188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_satisfies_conditions_l1241_124168

/-- The point M that is equidistant from A and B and lies on the y-axis -/
noncomputable def M : ℝ × ℝ × ℝ := (0, 2/3, 0)

/-- Point A -/
def A : ℝ × ℝ × ℝ := (3, 2, 0)

/-- Point B -/
def B : ℝ × ℝ × ℝ := (2, -1, 2)

/-- Distance function in 3D space -/
noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2.1 - q.2.1)^2 + (p.2.2 - q.2.2)^2)

theorem M_satisfies_conditions : 
  M.1 = 0 ∧ M.2.2 = 0 ∧ distance M A = distance M B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_satisfies_conditions_l1241_124168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1241_124198

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on an ellipse -/
structure PointOnEllipse (ε : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / ε.a^2 + y^2 / ε.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (ε : Ellipse) : ℝ :=
  Real.sqrt (1 - ε.b^2 / ε.a^2)

/-- The statement to be proved -/
theorem ellipse_eccentricity_range (ε : Ellipse) 
  (m n : ℝ) (h_m_geq_2n : m ≥ 2 * n)
  (h_m_max : ∀ (P : PointOnEllipse ε), 
    (abs (P.x + ε.a * eccentricity ε) * abs (P.x - ε.a * eccentricity ε) + P.y^2) ≤ m)
  (h_n_min : ∀ (P : PointOnEllipse ε), 
    ((P.x + ε.a * eccentricity ε) * (P.x - ε.a * eccentricity ε) + P.y^2) ≥ n) :
  (1/2 : ℝ) ≤ eccentricity ε ∧ eccentricity ε < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1241_124198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_correct_l1241_124172

/-- A cylinder with equal base diameter and height -/
structure EqualDiameterHeightCylinder where
  /-- The lateral surface area of the cylinder -/
  S : ℝ
  /-- Assumption that S is positive -/
  S_pos : S > 0

/-- The volume of a cylinder with equal base diameter and height -/
noncomputable def cylinderVolume (c : EqualDiameterHeightCylinder) : ℝ :=
  (c.S / 4) * Real.sqrt (c.S / Real.pi)

/-- Theorem stating that the volume formula is correct -/
theorem volume_formula_correct (c : EqualDiameterHeightCylinder) :
  cylinderVolume c = (c.S / 4) * Real.sqrt (c.S / Real.pi) :=
by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_correct_l1241_124172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cylinder_ratio_l1241_124179

/-- A sphere inscribed in a right circular cylinder -/
structure InscribedSphere :=
  (d : ℝ) -- diameter of the sphere and height of the cylinder
  (h_positive : d > 0)

/-- Volume of a sphere -/
noncomputable def sphere_volume (s : InscribedSphere) : ℝ := 
  (4 / 3) * Real.pi * (s.d / 2)^3

/-- Volume of a cylinder -/
noncomputable def cylinder_volume (s : InscribedSphere) : ℝ := 
  Real.pi * (s.d / 2)^2 * s.d

/-- The ratio of the volume of the inscribed sphere to the volume of the cylinder -/
noncomputable def volume_ratio (s : InscribedSphere) : ℝ := 
  sphere_volume s / cylinder_volume s

/-- Theorem: The ratio of the volume of the inscribed sphere to the volume of the cylinder is 2/3 -/
theorem inscribed_sphere_cylinder_ratio (s : InscribedSphere) : 
  volume_ratio s = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cylinder_ratio_l1241_124179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l1241_124185

theorem smallest_positive_z (x z : ℝ) : 
  Real.sin x = 0 → 
  Real.cos (x + z) = -1/2 → 
  (∀ w, 0 < w ∧ w < z → ¬(Real.sin x = 0 ∧ Real.cos (x + w) = -1/2)) → 
  z = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l1241_124185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_correct_l1241_124180

/-- The time taken for two people walking in opposite directions on a circular track to meet -/
noncomputable def meetingTime (trackCircumference : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  trackCircumference / (speed1 + speed2)

/-- Theorem: The meeting time is correct for the given problem -/
theorem meeting_time_correct (trackCircumference : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : trackCircumference = 1000)
  (h2 : speed1 = 20000 / 60)  -- 20 km/hr converted to m/min
  (h3 : speed2 = 13000 / 60)  -- 13 km/hr converted to m/min
  : ∃ ε > 0, |meetingTime trackCircumference speed1 speed2 - 1.82| < ε := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_correct_l1241_124180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_triangle_area_specific_l1241_124138

noncomputable section

variable (a b c : ℝ)
variable (A B C : ℝ)

def is_triangle (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def is_isosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def triangle_area (a b C : ℝ) : ℝ :=
  1/2 * a * b * Real.sin C

theorem isosceles_triangle (h : is_triangle a b c) (h1 : a * Real.sin A = b * Real.sin B) :
  is_isosceles a b c := by
  sorry

theorem triangle_area_specific (h : is_triangle a b c) (h1 : a + b = a * b) (h2 : c = 2) (h3 : C = π/3) :
  triangle_area a b C = Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_triangle_area_specific_l1241_124138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_13_l1241_124177

theorem three_digit_numbers_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 900 ∪ Finset.range 100)).card = 68 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_13_l1241_124177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_function_l1241_124143

theorem min_value_sin_cos_function :
  ∃ (x : ℝ), ∀ (y : ℝ), Real.sin y + Real.cos y - Real.sin y * Real.cos y ≥ 
    Real.sin x + Real.cos x - Real.sin x * Real.cos x ∧
  Real.sin x + Real.cos x - Real.sin x * Real.cos x = -1/2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_function_l1241_124143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1241_124173

theorem remainder_theorem (n : ℕ) : (4 * 6^n + 5^(n-1)) % 20 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1241_124173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l1241_124157

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 4)

theorem f_monotone_increasing_interval :
  ∃ (a b : ℝ), a = -Real.pi/8 ∧ b = 3*Real.pi/8 ∧
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l1241_124157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_odd_m_l1241_124130

theorem no_solutions_odd_m :
  ¬∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ (5 : ℚ) / m + (3 : ℚ) / n = 1 ∧ m % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_odd_m_l1241_124130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sophies_doughnuts_l1241_124114

theorem sophies_doughnuts :
  ∀ (num_doughnuts : ℕ),
    5 * 2 + num_doughnuts * 1 + 4 * 2 + 15 * 3 / 5 = 33 →
    num_doughnuts = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sophies_doughnuts_l1241_124114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_squared_inequality_l1241_124186

theorem contrapositive_squared_inequality :
  (∀ a b : ℝ, a > b → a^2 > b^2) ↔ (∀ a b : ℝ, a^2 ≤ b^2 → a ≤ b) :=
by
  apply Iff.intro
  · intro h a b h_sq
    contrapose! h_sq
    exact h a b h_sq
  · intro h a b h_gt
    contrapose! h_gt
    exact h a b h_gt

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_squared_inequality_l1241_124186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_train_length_l1241_124111

/-- The length of the platform in meters -/
noncomputable def platform_length : ℝ := 300

/-- The speed of train A in kilometers per hour -/
noncomputable def speed_A : ℝ := 72

/-- The speed of train B in kilometers per hour -/
noncomputable def speed_B : ℝ := 90

/-- The time taken by train A to cross the platform in seconds -/
noncomputable def time_A : ℝ := 30

/-- The time taken by train B to cross the platform in seconds -/
noncomputable def time_B : ℝ := 24

/-- Conversion factor from km/h to m/s -/
noncomputable def kmh_to_ms : ℝ := 5 / 18

/-- Theorem stating that the combined length of the two trains is 600 meters -/
theorem combined_train_length : 
  let length_A := speed_A * kmh_to_ms * time_A - platform_length
  let length_B := speed_B * kmh_to_ms * time_B - platform_length
  length_A + length_B = 600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_train_length_l1241_124111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_students_count_l1241_124197

theorem school_students_count : ℕ := by
  -- Define the total number of students
  let total_students : ℕ := 925

  -- Monday's condition
  have monday_condition : (total_students - 370) * 5 = total_students * 3 := by sorry

  -- Tuesday's condition
  have tuesday_condition : (total_students - 150) * 7 = total_students * 4 := by sorry

  -- Wednesday's condition
  have wednesday_condition : (total_students - 490) * 3 = total_students * 2 := by sorry

  -- Thursday's condition
  have thursday_condition : (total_students - 250) * 7 = total_students * 5 := by sorry

  -- Friday's condition
  have friday_condition : (total_students - 340) * 4 = total_students * 1 := by sorry

  -- Theorem: The total number of students is 925
  exact total_students


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_students_count_l1241_124197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_normal_correctness_l1241_124133

-- Curve 1
def curve1 (t : ℝ) : ℝ × ℝ × ℝ := (t^3, t^2, t)

-- Curve 2
def curve2 (z : ℝ) : ℝ × ℝ × ℝ := (z^4, z^2, z)

-- Tangent line for curve 1
def tangentLine1 (p : ℝ × ℝ × ℝ) : ℝ → ℝ × ℝ × ℝ :=
  λ t ↦ (3*t - 1, -2*t + 1, t - 1)

-- Normal plane for curve 1
def normalPlane1 (x y z : ℝ) : Prop :=
  3*x - 2*y + z + 6 = 0

-- Tangent line for curve 2
def tangentLine2 (p : ℝ × ℝ × ℝ) : ℝ → ℝ × ℝ × ℝ :=
  λ t ↦ (32*t + 16, 4*t + 4, t + 2)

-- Normal plane for curve 2
def normalPlane2 (x y z : ℝ) : Prop :=
  32*x + 4*y + z - 530 = 0

theorem curve_tangent_normal_correctness :
  (∀ t, tangentLine1 (curve1 (-1)) t = curve1 t) ∧
  (∀ x y z, normalPlane1 x y z ↔ (x, y, z) ∈ {p | ∃ t, p = tangentLine1 (curve1 (-1)) t}ᶜ) ∧
  (∀ t, tangentLine2 (curve2 2) t = curve2 t) ∧
  (∀ x y z, normalPlane2 x y z ↔ (x, y, z) ∈ {p | ∃ t, p = tangentLine2 (curve2 2) t}ᶜ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_normal_correctness_l1241_124133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1241_124163

/-- Parabola C defined by y^2 = 4x -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus F of the parabola C -/
def F : ℝ × ℝ := (1, 0)

/-- Point B -/
def B : ℝ × ℝ := (3, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_theorem (A : ℝ × ℝ) 
  (h1 : A ∈ C) 
  (h2 : distance A F = distance B F) : 
  distance A B = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1241_124163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l1241_124183

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The auxiliary circle of an ellipse -/
noncomputable def auxiliary_circle (e : Ellipse) : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = e.a^2 + e.b^2}

/-- The area of a triangle given the length of its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ :=
  (1/2) * base * height

theorem ellipse_max_triangle_area (e : Ellipse) (l : Line) :
  e.b = 1 →
  e.a^2 = 3 →
  ∃ (A B : ℝ × ℝ) (C D : ℝ × ℝ),
    A ∈ {p | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1} ∧
    B ∈ {p | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1} ∧
    C ∈ auxiliary_circle e ∧
    D ∈ auxiliary_circle e ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = 13 →
    (∀ (A' B' : ℝ × ℝ),
      A' ∈ {p | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1} →
      B' ∈ {p | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1} →
      triangle_area ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2).sqrt (3/2).sqrt ≤ Real.sqrt 3 / 2) ∧
    (∃ (A'' B'' : ℝ × ℝ),
      A'' ∈ {p | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1} ∧
      B'' ∈ {p | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1} ∧
      triangle_area ((A''.1 - B''.1)^2 + (A''.2 - B''.2)^2).sqrt (3/2).sqrt = Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l1241_124183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_ratio_is_three_halves_l1241_124127

/-- Two lines with the same non-zero y-intercept -/
structure TwoLines where
  y_intercept : ℝ
  slope1 : ℝ
  slope2 : ℝ
  x_intercept1 : ℝ
  x_intercept2 : ℝ
  y_intercept_nonzero : y_intercept ≠ 0
  slope1_is_8 : slope1 = 8
  slope2_is_12 : slope2 = 12

/-- The ratio of x-intercepts for two lines with specific slopes and same y-intercept -/
noncomputable def x_intercept_ratio (lines : TwoLines) : ℝ := lines.x_intercept1 / lines.x_intercept2

/-- Theorem stating that the ratio of x-intercepts is 3/2 -/
theorem x_intercept_ratio_is_three_halves (lines : TwoLines) : 
  x_intercept_ratio lines = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_ratio_is_three_halves_l1241_124127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l1241_124140

noncomputable def initial_number : ℂ := -6 - 2*Complex.I

noncomputable def rotation_angle : ℝ := 30 * Real.pi / 180

def scale_factor : ℝ := 3

theorem complex_transformation (z : ℂ) :
  scale_factor * (Complex.exp (rotation_angle * Complex.I) * z) = -3 - 9 * Real.sqrt 3 - 12 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l1241_124140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2024_value_l1241_124113

def sequence_a : ℕ → ℤ
  | 0 => 0
  | n + 1 => -Int.natAbs (sequence_a n + n.succ)

theorem a_2024_value : sequence_a 2024 = -1012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2024_value_l1241_124113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_l1241_124190

/-- A parabola with vertex (3/4, -25/16) and equation y = ax^2 + bx + c,
    where a > 0 and a + b + c is an integer. -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : a > 0
  vertex_y : ∃ n : ℤ, a + b + c = n

/-- The smallest possible value of a for the given parabola is 9/7. -/
theorem smallest_a (p : Parabola) : ∀ a' : ℝ, a' ≥ 9/7 → p.a ≥ a' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_l1241_124190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_leisure_time_l1241_124160

theorem madeline_leisure_time : 
  (let total_hours_per_week : ℕ := 168
   let class_hours : ℕ := 18
   let homework_hours : ℕ := 4 * 7
   let extracurricular_hours : ℕ := 3 * 3
   let tutoring_hours : ℕ := 1 * 2
   let work_hours : ℕ := 20
   let sleep_hours : ℕ := 8 * 5 + 10 * 2
   let exercise_hours : ℕ := 1 * 7
   let commute_hours : ℕ := 2 * 7
   let errands_hours : ℕ := 5
   let total_occupied_hours : ℕ := class_hours + homework_hours + extracurricular_hours + 
                                   tutoring_hours + work_hours + sleep_hours + 
                                   exercise_hours + commute_hours + errands_hours
   total_hours_per_week - total_occupied_hours) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_leisure_time_l1241_124160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_over_3_b_minus_a_range_l1241_124162

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ)
variable (a b c : ℝ)

-- Define vectors p and q
def p (A B : ℝ) : ℝ × ℝ := (2 * Real.sin A, Real.cos (A - B))
def q (B : ℝ) : ℝ × ℝ := (Real.sin B, -1)

-- Define the dot product of p and q
def p_dot_q (A B : ℝ) : ℝ := (p A B).1 * (q B).1 + (p A B).2 * (q B).2

-- Theorem for part I
theorem angle_C_is_pi_over_3 (h : p_dot_q A B = 1/2) : C = π/3 := by sorry

-- Theorem for part II
theorem b_minus_a_range (h1 : C = π/3) (h2 : c = Real.sqrt 3) : 
  -Real.sqrt 3 < b - a ∧ b - a < Real.sqrt 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_over_3_b_minus_a_range_l1241_124162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_acb_is_60_degrees_l1241_124165

-- Define a structure for a triangle with altitudes
structure TriangleWithAltitudes where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  acute : Bool

-- Define a function to calculate the angle ACB in degrees
noncomputable def angleACB (t : TriangleWithAltitudes) : ℝ := sorry

-- Define a function to check if the vector equation holds
def vectorEquationHolds (t : TriangleWithAltitudes) : Prop :=
  9 • (t.D.1 - t.A.1, t.D.2 - t.A.2) + 
  4 • (t.E.1 - t.B.1, t.E.2 - t.B.2) + 
  7 • (t.F.1 - t.C.1, t.F.2 - t.C.2) = (0, 0)

-- Theorem statement
theorem angle_acb_is_60_degrees (t : TriangleWithAltitudes) :
  t.acute ∧ vectorEquationHolds t → angleACB t = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_acb_is_60_degrees_l1241_124165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l1241_124108

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem min_distance_between_circles :
  ∃ (min_dist : ℝ),
    (∀ (x1 y1 x2 y2 : ℝ),
      circle_O x1 y1 → circle_C x2 y2 →
      distance x1 y1 x2 y2 ≥ min_dist) ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      circle_O x1 y1 ∧ circle_C x2 y2 ∧
      distance x1 y1 x2 y2 = min_dist) ∧
    min_dist = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l1241_124108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1241_124155

-- Define the function f(a, b, c) = (a - c)^2 + (b + 2c)^2
def f (a b c : ℝ) : ℝ := (a - c)^2 + (b + 2*c)^2

-- State the theorem
theorem min_value_theorem (a b : ℝ) (h : a^2 - 4 * Real.log a - b = 0) :
  ∃ (m : ℝ), m = 9/5 ∧ ∀ (c : ℝ), f a b c ≥ m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1241_124155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_complex_number_l1241_124120

theorem square_of_complex_number : 
  ∀ (z i : ℂ), z = 5 - 3 * i → i^2 = -1 → z^2 = 16 - 30 * i := by
  intros z i hz hi
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_complex_number_l1241_124120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1241_124119

/-- The speed of a train given its length, time to cross a man, and the man's speed -/
noncomputable def train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) : ℝ :=
  let relative_speed := train_length / crossing_time
  let man_speed_ms := man_speed * (1000 / 3600)
  (relative_speed - man_speed_ms) * (3600 / 1000)

/-- Theorem stating that the train speed is approximately 70 km/h given the problem conditions -/
theorem train_speed_problem : 
  let train_length := (125 : ℝ)
  let crossing_time := (6 : ℝ)
  let man_speed := (5 : ℝ)
  abs (train_speed train_length crossing_time man_speed - 70) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1241_124119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_count_l1241_124170

/-- Represents the number of coins of each type in the bag -/
def num_coins : ℕ := sorry

/-- The total value of coins in rupees -/
def total_value : ℚ := 105

/-- Theorem stating that if the total value of coins is 105 rupees, 
    and there are equal numbers of one rupee, 50 paise, and 25 paise coins, 
    then there are 60 coins of each type -/
theorem coin_count : 
  (1 : ℚ) * num_coins + (1/2 : ℚ) * num_coins + (1/4 : ℚ) * num_coins = total_value → 
  num_coins = 60 := by
  sorry

#check coin_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_count_l1241_124170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_officers_count_l1241_124118

/-- Represents the total number of police officers on the force -/
def total_officers : ℕ := sorry

/-- Represents the number of female officers on the force -/
def female_officers : ℕ := sorry

/-- Represents the number of officers on duty during the special event -/
def officers_on_duty : ℕ := 195

/-- Represents the number of male officers pulled from regular duties -/
def male_officers_pulled : ℕ := 15

/-- The percentage of female officers on the force -/
def female_percentage : ℚ := 40 / 100

/-- The percentage of female officers on duty during the special event -/
def female_on_duty_percentage : ℚ := 24 / 100

theorem female_officers_count :
  female_officers = 750 := by
  have h1 : officers_on_duty = 195 := rfl
  have h2 : male_officers_pulled = 15 := rfl
  have h3 : female_officers = (female_percentage * total_officers).floor := sorry
  have h4 : (female_on_duty_percentage * female_officers).floor = officers_on_duty - male_officers_pulled := sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_officers_count_l1241_124118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_equality_implies_y_value_proof_l1241_124171

def mean_equality_implies_y_value : ℝ → Prop :=
  fun y ↦ (4 + 6 + 20) / 3 = (14 + y) / 2 → y = 6

theorem mean_equality_implies_y_value_proof : ∀ y, mean_equality_implies_y_value y :=
  fun y h => by
    -- The proof steps would go here
    sorry

#check mean_equality_implies_y_value
#check mean_equality_implies_y_value_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_equality_implies_y_value_proof_l1241_124171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_ratio_implies_k_value_l1241_124193

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define line l1
def line1 (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Define line l2
def line2 (k x y : ℝ) : Prop := y = k * x - 1

-- Define the length of the chord intercepted by circle C on line l1
def chord_length_l1 : ℝ := 2

-- Define the length of the chord intercepted by circle C on line l2
noncomputable def chord_length_l2 (k : ℝ) : ℝ := 2 * Real.sqrt (4 - (2*k - 1)^2 / (k^2 + 1))

-- Theorem statement
theorem chord_ratio_implies_k_value :
  ∀ k : ℝ, chord_length_l2 k = 2 * chord_length_l1 → k = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_ratio_implies_k_value_l1241_124193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_circle_center_no_tangent_line_chord_length_when_k_is_one_l1241_124199

-- Define the line l
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y - k = 0

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Statement 1
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line k 1 0 := by sorry

-- Statement 2
theorem circle_center :
  ∃ x y : ℝ, x = 2 ∧ y = 1 ∧ ∀ x' y' : ℝ, circle_M x' y' ↔ (x' - x)^2 + (y' - y)^2 = 4 := by sorry

-- Statement 3
theorem no_tangent_line :
  ¬∃ k : ℝ, abs (2*k - 1 - k) / Real.sqrt (k^2 + 1) = 2 := by sorry

-- Statement 4
theorem chord_length_when_k_is_one :
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    line 1 x₁ y₁ ∧ circle_M x₁ y₁ ∧
    line 1 x₂ y₂ ∧ circle_M x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_circle_center_no_tangent_line_chord_length_when_k_is_one_l1241_124199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_eq_l1241_124106

/-- The eccentricity of the ellipse -/
noncomputable def e : ℝ := 1 / Real.sqrt 5

/-- The equation of the hyperbola that shares foci with the ellipse -/
def hyperbola_eq (x y : ℝ) : Prop := x^2 / 3 - y^2 / 2 = 1

/-- The standard form of an ellipse equation -/
def ellipse_eq (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating the standard equation of the ellipse -/
theorem ellipse_standard_eq : 
  ∃ (a b : ℝ), (∀ x y, ellipse_eq a b x y ↔ ellipse_eq 5 (Real.sqrt 20) x y) ∧ 
  (∀ x y, hyperbola_eq x y → (x^2 + y^2 = a^2 * e^2 + b^2)) := by
  sorry

#check ellipse_standard_eq

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_eq_l1241_124106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_function_from_A_to_B_l1241_124166

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1}

def f (x : ℤ) : ℤ := x^2

theorem f_is_function_from_A_to_B :
  (∀ x, x ∈ A → f x ∈ B) ∧ 
  (∀ x, x ∈ A → ∃! y, y ∈ B ∧ f x = y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_function_from_A_to_B_l1241_124166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_mn_l1241_124129

theorem product_mn (m n : ℤ) 
  (h1 : -m = 2) 
  (h2 : m.natAbs = 8) 
  (h3 : m + n > 0) : 
  m * n = -16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_mn_l1241_124129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_diophantine_equation_l1241_124167

theorem infinite_solutions_diophantine_equation 
  (l m n : ℕ+) 
  (h1 : Nat.gcd (l * m) n = 1) 
  (h2 : Nat.gcd (l * n) m = 1) 
  (h3 : Nat.gcd (m * n) l = 1) : 
  ∃ (S : Set (ℕ+ × ℕ+ × ℕ+)), 
    (Set.Infinite S) ∧ 
    (∀ (x y z : ℕ+), (x, y, z) ∈ S → x^(l : ℕ) + y^(m : ℕ) = z^(n : ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_diophantine_equation_l1241_124167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_coordinates_X_is_negative_four_thirds_l1241_124131

-- Define points as pairs of real numbers
def Point := ℝ × ℝ

-- Define the given points
noncomputable def Y : Point := (2, 6)
noncomputable def Z : Point := (0, -6)

-- Define the ratios
noncomputable def ratio_XZ_XY : ℝ := 1/3
noncomputable def ratio_ZY_XY : ℝ := 2/3

-- Function to calculate the sum of coordinates of a point
noncomputable def sum_coordinates (p : Point) : ℝ := p.1 + p.2

-- Theorem statement
theorem sum_coordinates_X_is_negative_four_thirds :
  ∃ X : Point,
    (sum_coordinates X = -4/3) ∧
    (ratio_XZ_XY = 1/3) ∧
    (ratio_ZY_XY = 2/3) ∧
    (Y = (2, 6)) ∧
    (Z = (0, -6)) :=
by
  -- Construct X using the section formula
  let X : Point := ((2*0 + 1*2)/(1+2), (2*(-6) + 1*6)/(1+2))
  
  -- Prove the existence of X and its properties
  use X
  apply And.intro
  · -- Prove sum_coordinates X = -4/3
    simp [sum_coordinates]
    norm_num
  · -- Prove the remaining conditions
    apply And.intro
    · -- ratio_XZ_XY = 1/3
      rfl
    apply And.intro
    · -- ratio_ZY_XY = 2/3
      rfl
    apply And.intro
    · -- Y = (2, 6)
      rfl
    · -- Z = (0, -6)
      rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_coordinates_X_is_negative_four_thirds_l1241_124131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1241_124164

open BigOperators

theorem inequality_proof (n : ℕ) (a b : Fin n → ℝ) 
  (h1 : ∀ i, 1 ≤ a i ∧ a i ≤ 2) 
  (h2 : ∀ i, 1 ≤ b i ∧ b i ≤ 2) 
  (h3 : ∑ i, (a i)^2 = ∑ i, (b i)^2) : 
  ∑ i, (a i)^3 / (b i) ≤ (17/10) * ∑ i, (a i)^2 := by
  sorry

/-- Necessary and sufficient conditions for equality --/
def equality_conditions (n : ℕ) (a b : Fin n → ℝ) : Prop :=
  Even n ∧ 
  (∃ s : Fin n → Bool, 
    (∀ i, s i = true → a i = 1 ∧ b i = 2) ∧
    (∀ i, s i = false → a i = 2 ∧ b i = 1) ∧
    (∑ i, if s i then (1 : ℕ) else 0) = n / 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1241_124164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1241_124161

-- Define the complex number i
def i : ℂ := Complex.I

-- Define a as a real number
variable (a : ℝ)

-- Define the condition that (2 - i) / (a + i) is a pure imaginary number
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

-- Define z as a complex number
noncomputable def z (a : ℝ) : ℂ := 4 * a + Complex.I * (Real.sqrt 2)

-- State the theorem
theorem modulus_of_z : 
  is_pure_imaginary ((2 - i) / (Complex.ofReal a + i)) → Complex.abs (z a) = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1241_124161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1241_124122

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sin ((1/5) * x + (13*Real.pi)/6)
noncomputable def g (x : ℝ) : ℝ := f (x - (10*Real.pi)/3)

-- State the theorem
theorem g_properties :
  (∀ x, g x = -Real.cos ((1/5) * x)) ∧
  (∃ p > 0, ∀ x, g (x + p) = g x ∧ ∀ q, 0 < q ∧ q < p → ∃ y, g (y + q) ≠ g y) ∧
  (∀ x, g (-x) = g x) ∧
  (¬ ∀ x, g (Real.pi/2 + x) = g (Real.pi/2 - x)) ∧
  (∀ x ∈ Set.Icc Real.pi (2*Real.pi), ∀ y ∈ Set.Icc Real.pi (2*Real.pi), x < y → g x < g y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1241_124122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1241_124152

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : (Real.sin t.A + Real.sin t.B) * (t.a - t.b) = t.c * (Real.sin t.C - Real.sqrt 3 * Real.sin t.B)) :
  t.A = π / 6 ∧ 
  (Real.cos t.B = -1/7 → 
   ∃ (D : ℝ) (BD : ℝ), 
   BD = 7 * Real.sqrt 7 / 3 ∧ 
   t.c = 7 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1241_124152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1241_124132

noncomputable def f (a b x : ℝ) : ℝ := (a * x) / (Real.exp x + 1) + b * Real.exp (-x)

theorem function_properties (a b : ℝ) :
  f a b 0 = 1 ∧
  (deriv (f a b)) 0 = -(1/2) →
  (a = 1 ∧ b = 1) ∧
  (∀ k x : ℝ, x ≠ 0 →
    f a b x > x / (Real.exp x - 1) + k * Real.exp (-x) →
    k ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1241_124132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cosine_factors_l1241_124136

/-- 
Given that there exist positive integers a, b, and c such that 
the equation sin²x + sin²(3x) + sin²(5x) + sin²(7x) = 2 
reduces to cos(ax) * cos(bx) * cos(cx) = 0, 
prove that a + b + c = 14 
-/
theorem sum_of_cosine_factors (a b c : ℕ+) 
  (h : ∀ x : ℝ, Real.sin x ^ 2 + Real.sin (3*x) ^ 2 + Real.sin (5*x) ^ 2 + Real.sin (7*x) ^ 2 = 2 → 
               Real.cos (a * x) * Real.cos (b * x) * Real.cos (c * x) = 0) : 
  a + b + c = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cosine_factors_l1241_124136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_slope_range_l1241_124128

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote angle condition
def asymptote_angle (a b : ℝ) : Prop := b / a = Real.tan (30 * Real.pi / 180)

-- Define the focal length condition
def focal_length (c : ℝ) : Prop := c = 4 * Real.sqrt 2

-- Define the slope of line F₁E
noncomputable def slope_F1E (t : ℝ) : ℝ := -t / (t^2 + 6)

theorem ellipse_and_slope_range :
  ∀ (a b c : ℝ),
  (a > b) → (b > 0) →
  (∀ x y, ellipse_C a b x y) →
  (∀ x y, hyperbola a b x y) →
  (asymptote_angle a b) →
  (focal_length c) →
  (∃ (x y : ℝ), ellipse_C 6 2 x y) ∧
  (∀ (k : ℝ), (∃ (t : ℝ), k = slope_F1E t) → -Real.sqrt 6 / 12 ≤ k ∧ k ≤ Real.sqrt 6 / 12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_slope_range_l1241_124128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l1241_124181

noncomputable def g (x : ℝ) : ℝ := (x - 5) / (x - 6)

-- State the theorem
theorem inverse_g_undefined_at_one :
  ∀ x : ℝ, x ≠ 1 → ∃ y : ℝ, g y = x :=
by
  sorry

#check inverse_g_undefined_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l1241_124181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_is_9600_l1241_124110

/-- Profit calculation for a business partnership --/
def profit_calculation (a_investment b_investment : ℕ) 
  (a_management_share : ℚ) (a_received : ℕ) : ℚ :=
  let total_investment : ℚ := a_investment + b_investment
  let a_capital_ratio : ℚ := a_investment / total_investment
  let remaining_profit_share : ℚ := 1 - a_management_share
  let a_total_share : ℚ := a_management_share + a_capital_ratio * remaining_profit_share
  (a_received : ℚ) / a_total_share

/-- The total profit of the business is 9600 --/
theorem total_profit_is_9600 :
  profit_calculation 20000 25000 (1/10) 4800 = 9600 := by
  sorry

-- Example evaluation (note: this will not produce an exact result due to rational arithmetic)
#eval profit_calculation 20000 25000 (1/10) 4800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_is_9600_l1241_124110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l1241_124191

/-- The rational function under consideration -/
noncomputable def f (x : ℝ) : ℝ := (3*x^2 - x - 4) / (x - 2)

/-- The slope of the slant asymptote -/
def m : ℝ := 3

/-- The y-intercept of the slant asymptote -/
def b : ℝ := 5

/-- Theorem stating that the sum of the slope and y-intercept of the slant asymptote is 8 -/
theorem slant_asymptote_sum : m + b = 8 := by
  -- Unfold the definitions of m and b
  unfold m b
  -- Perform the addition
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l1241_124191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l1241_124139

-- Define the point P
noncomputable def P : ℝ × ℝ := (4/5, -3/5)

-- Theorem statement
theorem angle_properties (α : ℝ) 
  (h : P.1 = Real.cos α ∧ P.2 = Real.sin α) : 
  Real.cos α = 4/5 ∧ 
  (Real.sin (π/2 - α) / Real.sin (α + π)) * (Real.tan (α - π) / Real.cos (3*π - α)) = 5/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l1241_124139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1241_124192

open Real

noncomputable def f (x : ℝ) : ℝ := cos (2 * x + π / 6)

def is_period (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem smallest_positive_period_of_f :
  ∃ T > 0, is_period T ∧ ∀ S, 0 < S → is_period S → T ≤ S :=
by
  -- We claim that π is the smallest positive period
  use π
  
  sorry  -- The proof is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1241_124192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diff_intersection_y_coords_is_zero_l1241_124100

-- Define the two functions
def f (x : ℝ) : ℝ := 4 - x^2 + x^4
def g (x : ℝ) : ℝ := 2 + x^2 + x^4

-- Define the set of intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Define the y-coordinates of the intersection points
def intersection_y_coords : Set ℝ := {y : ℝ | ∃ x, x ∈ intersection_points ∧ y = f x}

-- Statement: The maximum difference between any two y-coordinates of intersection points is 0
theorem max_diff_intersection_y_coords_is_zero :
  ∀ y₁ y₂, y₁ ∈ intersection_y_coords → y₂ ∈ intersection_y_coords → |y₁ - y₂| ≤ 0 :=
by
  sorry

#check max_diff_intersection_y_coords_is_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diff_intersection_y_coords_is_zero_l1241_124100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_perimeter_theorem_l1241_124169

/-- Represents the growth operation on a polygon edge -/
noncomputable def growth_operation (side_length : ℝ) : ℝ := (4 / 3) * side_length

/-- Calculates the perimeter after n growth operations -/
noncomputable def perimeter_after_growth (initial_side_length : ℝ) (n : ℕ) : ℝ :=
  3 * initial_side_length * (4 / 3) ^ n

theorem growth_perimeter_theorem (initial_side_length : ℝ) :
  initial_side_length = 9 →
  perimeter_after_growth initial_side_length 2 = 48 ∧
  perimeter_after_growth initial_side_length 4 = 85 + 1/3 :=
by
  intro h
  apply And.intro
  · -- Proof for 2 growth operations
    sorry
  · -- Proof for 4 growth operations
    sorry

#check growth_perimeter_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_perimeter_theorem_l1241_124169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_square_l1241_124149

theorem floor_ceil_fraction_square : 
  ⌊⌈((15 : ℚ) / 8)^2⌉ + (19 : ℚ) / 5⌋ = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_square_l1241_124149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l1241_124184

theorem increasing_function_a_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = Real.log x + x^2 - a * x) →
  (∀ x > 0, Monotone f) →
  a ≤ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l1241_124184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_chain_theorem_l1241_124125

/-- The length of the longest factor chain of a natural number -/
def L (x : ℕ) : ℕ := sorry

/-- The number of longest factor chains of a natural number -/
def R (x : ℕ) : ℕ := sorry

/-- Theorem about L and R for a specific form of x -/
theorem factor_chain_theorem (k m n : ℕ) :
  let x := 5^k * 31^m * 1990^n
  (L x = k + m + 3*n) ∧
  (R x = Nat.factorial (k + m + 3*n) / (Nat.factorial (k + n) * Nat.factorial m * (Nat.factorial n * Nat.factorial n))) := by
  sorry

#check factor_chain_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_chain_theorem_l1241_124125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_2n_B_n_plus_3_is_integer_l1241_124147

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

/-- Sequence A_n -/
def A (k : ℕ+) : ℕ → ℕ
| 0 => k
| 1 => k
| (n + 2) => A k n * A k (n + 1)

/-- Sequence B_n -/
noncomputable def B (k : ℕ+) : ℕ → ℚ
| 0 => 1
| 1 => k
| (n + 2) => (B k (n + 1) ^ 3 + 1) / B k n

/-- Main theorem -/
theorem A_2n_B_n_plus_3_is_integer (k : ℕ+) (n : ℕ) :
  ∃ m : ℤ, (A k (2 * n) : ℚ) * B k (n + 3) = m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_2n_B_n_plus_3_is_integer_l1241_124147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_workout_equivalence_l1241_124137

/-- The number of repetitions in Laura's usual routine -/
def usual_reps : ℕ := 30

/-- The weight of each dumbbell in Laura's usual routine (in pounds) -/
def usual_weight : ℚ := 10

/-- The weight of each new dumbbell (in pounds) -/
def new_weight : ℚ := 8

/-- The number of dumbbells used in each repetition -/
def num_dumbbells : ℕ := 2

/-- Calculates the total weight lifted in Laura's usual routine -/
noncomputable def total_weight : ℚ := num_dumbbells * usual_weight * usual_reps

/-- The number of repetitions needed with the new dumbbells to lift the same total weight -/
noncomputable def new_reps : ℚ := total_weight / (num_dumbbells * new_weight)

theorem laura_workout_equivalence :
  new_reps = 75/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_workout_equivalence_l1241_124137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_cos_pi_x_power_l1241_124182

theorem limit_cos_pi_x_power :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ →
    |((Real.cos (Real.pi * x)) ^ (1 / (x * Real.sin (Real.pi * x)))) - Real.exp (-Real.pi / 2)| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_cos_pi_x_power_l1241_124182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l1241_124159

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the distance function between a point and a line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - y - 4| / Real.sqrt 2

-- Statement of the theorem
theorem min_distance_curve_to_line :
  ∃ (x y : ℝ),
    curve_C x y ∧
    distance_to_line x y = Real.sqrt 2 ∧
    (∀ (x' y' : ℝ), curve_C x' y' → distance_to_line x' y' ≥ Real.sqrt 2) ∧
    x = 1 + Real.sqrt 2 ∧
    y = 1 - Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l1241_124159
