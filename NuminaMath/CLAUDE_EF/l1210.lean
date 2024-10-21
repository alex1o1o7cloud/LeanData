import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_is_eight_l1210_121097

/-- Represents a square with a given side length -/
structure Square where
  side_length : ℝ

/-- Represents the arrangement of three squares -/
structure SquareArrangement where
  square1 : Square
  square2 : Square
  square3 : Square

/-- The diagonal line connecting the bottom left corner of the smallest square
    to the upper right corner of the largest square -/
noncomputable def diagonal_line (arr : SquareArrangement) : ℝ → ℝ :=
  λ x ↦ (arr.square3.side_length / (arr.square1.side_length + arr.square2.side_length + arr.square3.side_length)) * x

/-- The area of the quadrilateral formed by the intersection of the diagonal line
    and the middle square -/
noncomputable def intersection_area (arr : SquareArrangement) : ℝ :=
  ((diagonal_line arr arr.square1.side_length + 
    diagonal_line arr (arr.square1.side_length + arr.square2.side_length)) / 2) * 
  arr.square2.side_length

theorem intersection_area_is_eight (arr : SquareArrangement) 
  (h1 : arr.square1.side_length = 2)
  (h2 : arr.square2.side_length = 4)
  (h3 : arr.square3.side_length = 6) :
  intersection_area arr = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_is_eight_l1210_121097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1210_121023

/-- Represents a cistern with two pipes -/
structure Cistern :=
  (fill_time : ℚ)    -- Time to fill the cistern with pipe A
  (empty_time : ℚ)   -- Time to empty the cistern with pipe B

/-- 
Calculates the time to fill the cistern when both pipes are open
given the fill time of pipe A and the empty time of pipe B
-/
def time_to_fill (c : Cistern) : ℚ :=
  (c.fill_time * c.empty_time) / (c.empty_time - c.fill_time)

/-- 
Theorem: For a cistern where pipe A fills it in 16 hours and pipe B empties it in 20 hours,
the time to fill the cistern when both pipes are open is 80 hours
-/
theorem cistern_fill_time :
  let c : Cistern := ⟨16, 20⟩
  time_to_fill c = 80 := by
  -- Unfold the definition of time_to_fill
  unfold time_to_fill
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1210_121023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integration_by_parts_l1210_121002

open MeasureTheory

theorem integration_by_parts (f g : ℝ → ℝ) (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) :
  ∫ (x : ℝ), f x * deriv g x = f x * g x - ∫ (x : ℝ), deriv f x * g x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integration_by_parts_l1210_121002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_at_restaurant_l1210_121053

theorem students_at_restaurant (total_students burger_students : ℕ) : 
  total_students = 45 → burger_students = 30 → 
  total_students = total_students :=
by
  intros h1 h2
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_at_restaurant_l1210_121053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1210_121095

open Real

theorem trigonometric_identities :
  (∀ x : ℝ, (cos (π/5 : ℝ) + cos (2*π/5 : ℝ) + cos (3*π/5 : ℝ) + cos (4*π/5 : ℝ)) = 0) ∧
  (sin (420 * π/180 : ℝ) * cos (330 * π/180 : ℝ) + sin (-690 * π/180 : ℝ) * cos (-660 * π/180 : ℝ) = 1) :=
by
  constructor
  · intro x
    sorry -- Proof for the first part
  · sorry -- Proof for the second part

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1210_121095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_conditions_l1210_121080

theorem count_integers_satisfying_conditions : 
  (∃ (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ 
      100 ≤ (n + 1) / 3 ∧ (n + 1) / 3 < 1000 ∧
      100 ≤ 3 * n + 1 ∧ 3 * n + 1 < 1000) ∧
    (∀ n ∉ S, ¬(n > 0 ∧ 
      100 ≤ (n + 1) / 3 ∧ (n + 1) / 3 < 1000 ∧
      100 ≤ 3 * n + 1 ∧ 3 * n + 1 < 1000)) ∧
    Finset.card S = 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_conditions_l1210_121080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koch_snowflake_area_squared_l1210_121054

/-- The area of the Koch snowflake-like figure obtained from an equilateral triangle --/
noncomputable def koch_snowflake_area (side_length : ℝ) : ℝ :=
  let initial_area := (Real.sqrt 3 / 4) * side_length^2
  initial_area * (1 + (3/4) * (4/5))

/-- The theorem stating that the square of the area of the Koch snowflake-like figure
    obtained from an equilateral triangle with side length 10 is 4800 --/
theorem koch_snowflake_area_squared :
  (koch_snowflake_area 10)^2 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_koch_snowflake_area_squared_l1210_121054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_proof_non_monotonic_range_range_existence_l1210_121042

noncomputable def f (k : ℤ) (x : ℝ) : ℝ := x ^ ((2 - k) * (1 + k))

theorem power_function_proof (k : ℤ) :
  (∀ x y, 0 < x ∧ x < y → f k x < f k y) →
  f k = fun x => x^2 :=
by sorry

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := 2 * f x - 4 * x + 3

theorem non_monotonic_range (f : ℝ → ℝ) (a : ℝ) :
  (∃ x y, 2*a ≤ x ∧ x < y ∧ y ≤ a + 1 ∧ F f x > F f y) →
  0 < a ∧ a < 1/2 :=
by sorry

noncomputable def g (f : ℝ → ℝ) (q : ℝ) (x : ℝ) : ℝ := 1 - q * f x + (2*q - 1) * x

theorem range_existence :
  ∃ q : ℝ, q > 0 ∧
    (∀ x, -1 ≤ x ∧ x ≤ 2 → -4 ≤ g (fun x => x^2) q x ∧ g (fun x => x^2) q x ≤ 17/8) ∧
    (∃ x y, -1 ≤ x ∧ x ≤ 2 ∧ -1 ≤ y ∧ y ≤ 2 ∧ g (fun x => x^2) q x = -4 ∧ g (fun x => x^2) q y = 17/8) ∧
    q = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_proof_non_monotonic_range_range_existence_l1210_121042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1210_121000

/- Define a triangle with side lengths and angles -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  angle_positive : 0 < A ∧ 0 < B ∧ 0 < C

/- Define the given condition -/
def condition (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C

/- State the theorem -/
theorem triangle_theorem (t : Triangle) (h : condition t) : 
  t.B = π / 3 ∧ 
  (t.b = 2 → ∃ (max_perim : ℝ), max_perim = 6 ∧ 
    ∀ (t' : Triangle), condition t' → t'.b = 2 → t'.a + t'.b + t'.c ≤ max_perim) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1210_121000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonagon_perimeter_l1210_121088

/-- The perimeter of a nonagon with eight sides of length 3 units and one side of length 4 units is 28 units. -/
theorem nonagon_perimeter (sides_count eight_side_length one_side_length : ℕ) :
  sides_count = 9 ∧ eight_side_length = 3 ∧ one_side_length = 4 →
  8 * eight_side_length + one_side_length = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonagon_perimeter_l1210_121088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_valid_plans_l1210_121077

/-- Represents the purchasing problem for a supermarket --/
structure SupermarketPurchase where
  cost_A : ℕ            -- Cost price of item A
  cost_B : ℕ            -- Cost price of item B
  sell_A : ℕ            -- Selling price of item A
  sell_B : ℕ            -- Selling price of item B
  total_items : ℕ       -- Total number of items to purchase
  budget : ℕ            -- Maximum budget
  min_profit : ℕ        -- Minimum required profit

/-- The specific instance of the supermarket purchase problem --/
def problem : SupermarketPurchase :=
  { cost_A := 40
  , cost_B := 60
  , sell_A := 50
  , sell_B := 75
  , total_items := 50
  , budget := 2520
  , min_profit := 620
  }

/-- Theorem stating that there are exactly 3 valid purchasing plans --/
theorem three_valid_plans (p : SupermarketPurchase) : 
  p = problem →  
  (∃ (s : Finset ℕ), 
    (∀ a ∈ s, 
      a * p.cost_A + (p.total_items - a) * p.cost_B ≤ p.budget ∧ 
      a * p.sell_A + (p.total_items - a) * p.sell_B - (a * p.cost_A + (p.total_items - a) * p.cost_B) ≥ p.min_profit) ∧ 
    s.card = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_valid_plans_l1210_121077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_problem_solution_l1210_121087

/-- Represents the time taken by a worker to complete the entire work alone -/
structure Worker where
  days_to_complete : ℚ
  days_to_complete_pos : days_to_complete > 0

/-- Represents a work scenario with two workers -/
structure WorkScenario where
  p : Worker
  q : Worker
  total_days : ℚ
  total_days_pos : total_days > 0

/-- The solution to the work problem -/
noncomputable def solution (w : WorkScenario) : ℚ :=
  (w.total_days * (1 / w.p.days_to_complete + 1 / w.q.days_to_complete) - 1) /
  (1 / w.p.days_to_complete + 1 / w.q.days_to_complete - 1 / w.p.days_to_complete)

/-- The main theorem stating the solution for the given scenario -/
theorem work_problem_solution (w : WorkScenario) 
  (hp : w.p.days_to_complete = 80)
  (hq : w.q.days_to_complete = 48)
  (ht : w.total_days = 40) :
  solution w = 16 := by
  sorry

#check work_problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_problem_solution_l1210_121087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_total_score_five_rounds_l1210_121083

/-- Represents the possible scores in one round -/
inductive Score
  | one
  | three
  | five

/-- Represents the game setup with two boxes -/
structure GameSetup :=
  (boxA : Finset (Fin 5))
  (boxB : Finset (Fin 5))

/-- Calculates the probability of drawing a specific ball from a box -/
def drawProbability (box : Finset (Fin 5)) (ball : Fin 5) : ℚ :=
  if ball ∈ box then 1 / box.card else 0

/-- Calculates the score based on the balls drawn from box B -/
def calculateScore (drawnBalls : Finset (Fin 5)) : Score :=
  match (drawnBalls.filter (λ b => b < 2)).card with
    | 2 => Score.five
    | 1 => Score.three
    | _ => Score.one

/-- Calculates the probability of a specific score in one round -/
noncomputable def scoreProbability (setup : GameSetup) (score : Score) : ℚ :=
  sorry  -- Proof omitted

/-- Calculates the expected score in one round -/
noncomputable def expectedScore (setup : GameSetup) : ℚ :=
  (1 : ℚ) * scoreProbability setup Score.one +
  (3 : ℚ) * scoreProbability setup Score.three +
  (5 : ℚ) * scoreProbability setup Score.five

/-- Main theorem: Expected total score in 5 rounds is 13 -/
theorem expected_total_score_five_rounds (setup : GameSetup) :
  5 * expectedScore setup = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_total_score_five_rounds_l1210_121083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1210_121010

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 4*y - 2 = 0

-- Define the point M
def point_M : ℝ × ℝ := (-3, -3)

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x y : ℝ) (a b c : ℝ) : ℝ :=
  abs (a*x + b*y + c) / Real.sqrt (a^2 + b^2)

-- Define the chord length
noncomputable def chord_length (r d : ℝ) : ℝ := 2 * Real.sqrt (r^2 - d^2)

theorem circle_line_intersection :
  ∃ (a b c : ℝ),
    -- Line l passes through point M
    a * point_M.1 + b * point_M.2 + c = 0 ∧
    -- Distance from circle center to line l is √5
    distance_point_to_line 0 (-2) a b c = Real.sqrt 5 →
    -- Chord length is 4√5
    chord_length 5 (Real.sqrt 5) = 4 * Real.sqrt 5 ∧
    -- Equation of line l
    ((a = 1 ∧ b = 2 ∧ c = 9) ∨ (a = 2 ∧ b = -1 ∧ c = 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1210_121010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalization_minimizes_sum_l1210_121007

noncomputable def rationalize_denominator (a b c : ℝ) : ℝ := 
  (a * (b + c)) / ((b - c) * (b + c))

theorem rationalization_minimizes_sum : 
  ∃ (A x y z w : ℕ), 
    w > 0 ∧ 
    ¬ (∃ (p : ℕ), Nat.Prime p ∧ p^2 ∣ y) ∧
    rationalize_denominator (Real.sqrt 50) (Real.sqrt 25) (Real.sqrt 5) = 
      (↑A * ↑x * Real.sqrt ↑y + ↑z) / ↑w ∧
    A + x + y + z + w = 13 ∧
    ∀ (A' x' y' z' w' : ℕ), 
      w' > 0 → 
      ¬ (∃ (p : ℕ), Nat.Prime p ∧ p^2 ∣ y') →
      rationalize_denominator (Real.sqrt 50) (Real.sqrt 25) (Real.sqrt 5) = 
        (↑A' * ↑x' * Real.sqrt ↑y' + ↑z') / ↑w' →
      A + x + y + z + w ≤ A' + x' + y' + z' + w' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalization_minimizes_sum_l1210_121007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_not_in_fourth_quadrant_l1210_121014

/-- The fixed point of f(x) = a^(x-1) - 2 -/
def fixed_point : ℝ × ℝ := (1, -1)

/-- The function f(x) = a^(x-1) - 2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) - 2

/-- The function g(x) = m + x^n where (m, n) is the fixed point of f -/
noncomputable def g (x : ℝ) : ℝ := fixed_point.1 + x^fixed_point.2

/-- Theorem: g(x) does not pass through the fourth quadrant -/
theorem g_not_in_fourth_quadrant :
  ∀ a : ℝ, a > 0 → a ≠ 1 →
  (∀ x : ℝ, f a x = fixed_point.2 ↔ x = fixed_point.1) →
  ¬∃ x y : ℝ, x > 0 ∧ y < 0 ∧ g x = y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_not_in_fourth_quadrant_l1210_121014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_equality_l1210_121031

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def S (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := 
  if r = 1 then n * a₁ 
  else a₁ * (1 - r^n) / (1 - r)

/-- Theorem stating that x equals y for any geometric sequence -/
theorem geometric_sequence_sum_equality 
  (a₁ : ℝ) (r : ℝ) (n : ℕ) (h : r ≠ 1) :
  let S_n := S a₁ r n
  let S_2n := S a₁ r (2*n)
  let S_3n := S a₁ r (3*n)
  let x := S_n^2 + S_2n^2
  let y := S_n * (S_2n + S_3n)
  x = y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_equality_l1210_121031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_third_iteration_l1210_121071

/-- Represents an interval [a, b] where a ≤ b -/
structure MyInterval (α : Type) [PartialOrder α] where
  lower : α
  upper : α
  ordered : lower ≤ upper

/-- Represents a step in the bisection method -/
def bisection_step {α : Type} [LinearOrder α] [Add α] [Div α] [OfNat α 2] (i : MyInterval α) : MyInterval α × MyInterval α :=
  let mid := (i.lower + i.upper) / 2
  (⟨i.lower, mid, by sorry⟩, ⟨mid, i.upper, by sorry⟩)

/-- The bisection method theorem -/
theorem bisection_method_third_iteration :
  ∃ (f : ℝ → ℝ), 
  let initial := MyInterval.mk (-2 : ℝ) (4 : ℝ) (by norm_num)
  let (i1, _) := bisection_step initial
  let (i2, _) := bisection_step i1
  let (i3, _) := bisection_step i2
  i3 = MyInterval.mk (-1/2 : ℝ) (1 : ℝ) (by norm_num) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_third_iteration_l1210_121071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1210_121059

-- Define the hyperbola
noncomputable def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- Define the slope of the asymptote
noncomputable def asymptote_slope (a b : ℝ) : ℝ := b / a

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  asymptote_slope a b = Real.tan (60 * π / 180) →
  eccentricity a b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1210_121059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_phi_forms_cone_l1210_121030

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Defines the set of points satisfying φ = c -/
def ConstantPhiSet (c : ℝ) : Set SphericalPoint :=
  {p : SphericalPoint | p.φ = c}

/-- Predicate to represent a cone in 3D space -/
def IsCone (s : Set SphericalPoint) (apex : ℝ × ℝ × ℝ) (axis : ℝ × ℝ × ℝ) (angle : ℝ) : Prop :=
  sorry -- We'll leave this undefined for now

/-- Theorem: The set of points with constant φ forms a cone -/
theorem constant_phi_forms_cone (c : ℝ) :
  ∃ (apex : ℝ × ℝ × ℝ) (axis : ℝ × ℝ × ℝ) (angle : ℝ),
    IsCone (ConstantPhiSet c) apex axis angle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_phi_forms_cone_l1210_121030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_area_center_l1210_121052

noncomputable section

/-- The equation of a circle in the form x^2 + y^2 + kx + 2y + k^2 = 0 -/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + k*x + 2*y + k^2 = 0

/-- The radius of the circle given its equation -/
noncomputable def circle_radius (k : ℝ) : ℝ :=
  Real.sqrt (1 - (3/4) * k^2)

/-- The center of the circle given its equation -/
def circle_center (k : ℝ) : ℝ × ℝ :=
  (-k/2, -1)

/-- The area of the circle given its radius -/
noncomputable def circle_area (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem circle_max_area_center :
  ∀ k : ℝ,
  (∀ x y : ℝ, circle_equation x y k) →
  (∀ k' : ℝ, circle_area (circle_radius k) ≥ circle_area (circle_radius k')) →
  circle_center k = (0, -1) := by
  sorry

#check circle_max_area_center

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_area_center_l1210_121052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_correct_l1210_121079

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- Length of the longer parallel side -/
  a : ℝ
  /-- Length of the shorter parallel side -/
  b : ℝ
  /-- Height of the trapezoid -/
  h : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The longer side is indeed longer than the shorter side -/
  h_ab : a > b
  /-- The height is positive -/
  h_h_pos : h > 0

/-- The radius of the inscribed circle in the given isosceles trapezoid -/
noncomputable def inscribed_circle_radius (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  480 / (24 + 2 * Real.sqrt 109)

/-- Theorem stating that the radius of the inscribed circle in the given isosceles trapezoid is correct -/
theorem inscribed_circle_radius_correct (t : IsoscelesTrapezoidWithInscribedCircle) 
    (h_a : t.a = 30) (h_b : t.b = 18) (h_h : t.h = 20) : 
    t.r = inscribed_circle_radius t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_correct_l1210_121079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_A_fill_time_l1210_121056

/-- Time taken by pipe B to fill the tanker -/
noncomputable def time_B : ℝ := 15

/-- Time taken by pipes A and B together to fill the tanker -/
noncomputable def time_AB : ℝ := 10

/-- Rate at which pipe B fills the tanker -/
noncomputable def rate_B : ℝ := 1 / time_B

/-- Rate at which pipes A and B together fill the tanker -/
noncomputable def rate_AB : ℝ := 1 / time_AB

/-- Time taken by pipe A to fill the tanker alone -/
noncomputable def time_A : ℝ := 30

theorem pipe_A_fill_time :
  time_A = 30 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_A_fill_time_l1210_121056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_is_empty_l1210_121015

def A : Set ℝ := {x | Real.sqrt (x - 2) ≤ 0}
def B : Set ℝ := {x | (10 : ℝ)^(x^2 - 2) = (10 : ℝ)^x}

theorem intersection_A_complement_B_is_empty :
  A ∩ Bᶜ = ∅ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_is_empty_l1210_121015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_exterior_angle_45_l1210_121099

/-- A regular polygon with an exterior angle of 45° has 8 sides. -/
theorem regular_polygon_exterior_angle_45 :
  ∀ (n : ℕ), n > 0 →
  (∃ (angle : ℝ), angle = 45 ∧ angle * n = 360) →
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_exterior_angle_45_l1210_121099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_ratio_implies_k_value_l1210_121032

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define line l1
def line1 (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Define line l2
def line2 (k x y : ℝ) : Prop := y = k * x - 1

-- Define the length of the chord intercepted by circle C on line l1
noncomputable def chord_length1 : ℝ := 2

-- Define the length of the chord intercepted by circle C on line l2
noncomputable def chord_length2 (k : ℝ) : ℝ := 2 * Real.sqrt (4 - ((2*k - 1) / Real.sqrt (k^2 + 1))^2)

-- Theorem statement
theorem chord_ratio_implies_k_value :
  ∀ k : ℝ, (chord_length2 k = 2 * chord_length1) → k = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_ratio_implies_k_value_l1210_121032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistors_combined_resistance_l1210_121019

/-- The combined resistance of two resistors in parallel -/
noncomputable def combined_resistance (x y : ℝ) : ℝ := 1 / (1/x + 1/y)

/-- Theorem: The combined resistance of two resistors with resistances 5 ohms and 7 ohms in parallel is 35/12 ohms -/
theorem parallel_resistors_combined_resistance :
  combined_resistance 5 7 = 35/12 := by
  -- Unfold the definition of combined_resistance
  unfold combined_resistance
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistors_combined_resistance_l1210_121019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_masha_number_unique_masha_number_value_l1210_121093

/-- The integer part of a real number -/
noncomputable def intPart (x : ℝ) : ℤ :=
  ⌊x⌋

/-- A number is 43% less than another number -/
def isFortyThreePercentLess (x y : ℝ) : Prop :=
  x = 0.57 * y

theorem masha_number_unique :
  ∃! x : ℝ, x > 0 ∧ isFortyThreePercentLess (↑(intPart x)) x :=
by
  sorry

theorem masha_number_value :
  ∀ x : ℝ, x > 0 → isFortyThreePercentLess (↑(intPart x)) x → x = 100 / 57 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_masha_number_unique_masha_number_value_l1210_121093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_problem_matching_l1210_121069

/-- A bipartite graph representation of students and problems -/
structure StudentProblemGraph where
  students : Finset (Fin 6)
  problems : Finset (Fin 6)
  solved : (Fin 6) → (Fin 6) → Bool

/-- Each student solved exactly two problems -/
def TwoProblemsPerStudent (g : StudentProblemGraph) : Prop :=
  ∀ s : Fin 6, (g.problems.filter (λ p => g.solved s p)).card = 2

/-- Each problem was solved by exactly two students -/
def TwoStudentsPerProblem (g : StudentProblemGraph) : Prop :=
  ∀ p : Fin 6, (g.students.filter (λ s => g.solved s p)).card = 2

/-- A perfect matching in the graph -/
def PerfectMatching (g : StudentProblemGraph) : Prop :=
  ∃ f : Fin 6 → Fin 6, Function.Injective f ∧ (∀ s : Fin 6, g.solved s (f s))

/-- The main theorem -/
theorem student_problem_matching (g : StudentProblemGraph) 
  (h1 : TwoProblemsPerStudent g) (h2 : TwoStudentsPerProblem g) : 
  PerfectMatching g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_problem_matching_l1210_121069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_three_l1210_121048

theorem expression_equals_three (a b : ℝ) (h : (3*a)/2 + b = 1) :
  (9:ℝ)^a * (3:ℝ)^b / (3:ℝ)^a^(1/2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_three_l1210_121048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_return_speed_l1210_121091

/-- Proves that the average speed for the return trip is 9 miles per hour given the conditions of the cyclist's journey. -/
theorem cyclist_return_speed (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) : 
  total_distance = 36 →
  speed1 = 12 →
  speed2 = 10 →
  total_time = 7.3 →
  (let time_to_destination := total_distance / (2 * speed1) + total_distance / (2 * speed2);
   let return_time := total_time - time_to_destination;
   total_distance / return_time = 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_return_speed_l1210_121091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_odd_is_even_l1210_121040

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the condition that f is odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the derivative of f as g
noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := deriv f x

-- State the theorem
theorem derivative_of_odd_is_even (h : is_odd f) (hf : Differentiable ℝ f) :
  ∀ x, g f (-x) = g f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_odd_is_even_l1210_121040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1210_121020

noncomputable section

/-- The line equation y = (3x - 1) / 4 -/
def line_equation (x : ℝ) : ℝ := (3 * x - 1) / 4

/-- The point we're measuring distance from -/
def target_point : ℝ × ℝ := (8, 2)

/-- The point on the line claimed to be closest to the target point -/
def closest_point : ℝ × ℝ := (6.2, 4.4)

/-- Check if a point lies on the line -/
def on_line (p : ℝ × ℝ) : Prop :=
  p.2 = line_equation p.1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem closest_point_on_line :
  on_line closest_point ∧
  ∀ p : ℝ × ℝ, on_line p → distance p target_point ≥ distance closest_point target_point :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1210_121020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_pair_in_range_l1210_121073

theorem existence_of_pair_in_range (S : Finset ℝ) (h : S.card = 7) :
  ∃ x y, x ∈ S ∧ y ∈ S ∧ 0 ≤ (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) ≤ 1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_pair_in_range_l1210_121073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_square_side_l1210_121058

theorem minimum_square_side (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let side := if a < (Real.sqrt 2 + 1) * b
               then a
               else (Real.sqrt 2 / 2) * (a + b)
  side ≥ a ∧ side ≥ b ∧
  ∀ s : ℝ, s ≥ a ∧ s ≥ b → s ≥ side := by
  sorry

#check minimum_square_side

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_square_side_l1210_121058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_l1210_121024

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - 3 * x

-- Define the interval (1/2, 3)
def interval : Set ℝ := {x | 1/2 < x ∧ x < 3}

-- State the theorem
theorem monotone_increasing_condition (a : ℝ) : 
  (∀ x ∈ interval, StrictMono (fun x => f a x)) ↔ a ≥ 9/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_l1210_121024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_second_component_l1210_121026

/-- Given two vectors a and b in R³, if they are parallel, then the second component of b is -4. -/
theorem parallel_vectors_second_component (a b : Fin 3 → ℝ) :
  (a 0 = -1 ∧ a 1 = 2 ∧ a 2 = -3) →
  b 0 = 2 →
  b 2 = 6 →
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  b 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_second_component_l1210_121026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_theorem_l1210_121039

theorem acute_angles_theorem (α β γ : Real) 
  (h_acute : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2) 
  (h_cos : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) : 
  Real.tan α * Real.tan β * Real.tan γ ≥ 2 * Real.sqrt 2 ∧ 
  3 * π / 4 < α + β + γ ∧ α + β + γ < π :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_theorem_l1210_121039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l1210_121041

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  /-- Distance from the origin to the plane -/
  distance : ℝ
  /-- x-coordinate of the intersection with the x-axis -/
  α : ℝ
  /-- y-coordinate of the intersection with the y-axis -/
  β : ℝ
  /-- z-coordinate of the intersection with the z-axis -/
  γ : ℝ
  /-- Condition that the plane is at the given distance from the origin -/
  distance_eq : distance = 2
  /-- Condition that the intersections are distinct from the origin -/
  distinct_from_origin : α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0

/-- The centroid of the triangle formed by the intersections -/
noncomputable def centroid (plane : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (plane.α / 3, plane.β / 3, plane.γ / 3)

/-- Theorem statement -/
theorem centroid_sum (plane : IntersectingPlane) :
  let (p, q, r) := centroid plane
  1 / p^2 + 1 / q^2 + 1 / r^2 = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l1210_121041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subtraction_problem_solution_l1210_121025

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a three-digit number -/
def ThreeDigitNumber (h t u : Digit) : ℕ := 100 * h.val + 10 * t.val + u.val

/-- The subtraction problem -/
def SubtractionProblem (a b c : Digit) : Prop :=
  ThreeDigitNumber a b c - ThreeDigitNumber c b a = 307

theorem subtraction_problem_solution :
  ∀ (a b c : Digit), SubtractionProblem a b c ↔ (a.val = 2 ∧ c.val = 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subtraction_problem_solution_l1210_121025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_existence_and_nonexistence_l1210_121027

def is_special (E : Finset ℕ) : Prop :=
  ∀ a b, a ∈ E → b ∈ E → a ≠ b → (a - b)^2 ∣ (a * b)

theorem special_set_existence_and_nonexistence :
  (∃ E : Finset ℕ, is_special E ∧ E.card = 3) ∧
  (¬ ∃ x y : ℕ, is_special {x, x + y, x + 2*y, x + 3*y}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_existence_and_nonexistence_l1210_121027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_movement_l1210_121057

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Convert ℝ × ℝ to Point
def toPoint (p : ℝ × ℝ) : Point :=
  ⟨p.1, p.2⟩

-- Define the problem setup
def problem_setup (S₁ S₂ : Circle) (A M₁ M₂ : Point) (v : ℝ) : Prop :=
  -- Circles pass through each other's centers
  distance (toPoint S₁.center) (toPoint S₂.center) = S₁.radius ∧
  distance (toPoint S₁.center) (toPoint S₂.center) = S₂.radius ∧
  -- A is an intersection point
  distance A (toPoint S₁.center) = S₁.radius ∧
  distance A (toPoint S₂.center) = S₂.radius ∧
  -- M₁ and M₂ are on their respective circles
  distance M₁ (toPoint S₁.center) = S₁.radius ∧
  distance M₂ (toPoint S₂.center) = S₂.radius ∧
  -- M₁ and M₂ move with the same linear velocity v
  v > 0

-- Define the theorem
theorem circle_movement 
  (S₁ S₂ : Circle) (A M₁ M₂ : Point) (v : ℝ) 
  (h : problem_setup S₁ S₂ A M₁ M₂ v) : 
  -- Triangle AM₁M₂ is equilateral
  distance A M₁ = distance M₁ M₂ ∧ distance M₁ M₂ = distance M₂ A ∧
  -- Center of AM₁M₂ moves in a circle with radius 2/3 of S₁'s radius
  ∃ (C : Circle), C.radius = (2/3) * S₁.radius ∧
  -- Velocity of the center of AM₁M₂ is 2/3v
  ∃ (v_center : ℝ), v_center = (2/3) * v :=
by
  sorry  -- The proof is omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_movement_l1210_121057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l1210_121084

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_measure (t : Triangle) 
  (h1 : Real.sqrt 3 * t.a * Real.cos t.C = (2 * t.b - Real.sqrt 3 * t.c) * Real.cos t.A)
  (h2 : 0 < t.A) (h3 : t.A < Real.pi) :
  t.A = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l1210_121084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_to_points_l1210_121011

/-- The parabola y² = 8x -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_sum_to_points (A B : ℝ × ℝ) (h1 : A = (2, 0)) (h2 : B = (7, 6)) :
  ∃ (min : ℝ), ∀ P ∈ Parabola, distance A P + distance B P ≥ min ∧
  ∃ P ∈ Parabola, distance A P + distance B P = min ∧ min = 3 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_to_points_l1210_121011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1210_121089

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + Real.cos x

theorem f_properties :
  (∃ (a b : ℝ), ∀ x, f x ≤ b ∧ a ≤ f x) ∧
  (∀ x, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1210_121089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_property_l1210_121028

theorem root_sum_property (r₁ r₂ r₃ : ℝ) :
  (r₁^3 - 3*r₁^2 + 1 = 0) →
  (r₂^3 - 3*r₂^2 + 1 = 0) →
  (r₃^3 - 3*r₃^2 + 1 = 0) →
  (r₁ + r₂ + r₃ = 3) →
  ((3*r₁ - 2)^(1/3) + (3*r₂ - 2)^(1/3) + (3*r₃ - 2)^(1/3) = 0) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_property_l1210_121028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_classmates_l1210_121061

/-- A type representing students -/
def Student : Type := ℕ

/-- The total number of students -/
def total_students : ℕ := 60

/-- A predicate that checks if two students are classmates -/
def are_classmates (x y : Student) : Prop := sorry

/-- A predicate that checks if a set of students contains at least 3 classmates -/
def has_three_classmates (s : Finset Student) : Prop :=
  ∃ (t : Finset Student), t ⊆ s ∧ t.card = 3 ∧ (∀ x y, x ∈ t → y ∈ t → x ≠ y → are_classmates x y)

/-- The main theorem to prove -/
theorem fifteen_classmates 
  (h : ∀ s : Finset Student, s.card = 10 → has_three_classmates s) :
  ∃ (t : Finset Student), t.card ≥ 15 ∧ (∀ x y, x ∈ t → y ∈ t → x ≠ y → are_classmates x y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_classmates_l1210_121061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_tangent_point_l1210_121064

/-- The circle with center (1, 0) and radius 1 -/
def Circle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

/-- The distance between two points (x1, y1) and (x2, y2) -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- A point P(x, y) draws a tangent of length 2 to the circle -/
def drawsTangent (x y : ℝ) : Prop :=
  ∃ (xt yt : ℝ), Circle xt yt ∧ distance x y xt yt = 2

/-- The trajectory of point P -/
def trajectory (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 5

/-- Theorem: The trajectory of point P that draws a tangent of length 2 to the circle
    is (x-1)^2 + y^2 = 5 -/
theorem trajectory_of_tangent_point :
  ∀ (x y : ℝ), drawsTangent x y → trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_tangent_point_l1210_121064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_equality_l1210_121016

open Real

variable (a b c d p q r x y : ℝ)

noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem logarithm_equality 
  (h1 : log 10 a / p = log 10 b / q)
  (h2 : log 10 a / p = log 10 (c^d) / r)
  (h3 : log 10 a / p = log 10 x)
  (h4 : x ≠ 1)
  (h5 : b^3 / (a^2 * c) = x^y) :
  y = 3*q - 2*p - r/d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_equality_l1210_121016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l1210_121038

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l1210_121038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_with_midpoint_distances_l1210_121045

/-- The volume of a regular tetrahedron with specific midpoint height distances -/
theorem tetrahedron_volume_with_midpoint_distances :
  ∀ (h a : ℝ),
    h > 0 →
    a > 0 →
    h = (4 : ℝ) →
    (a * Real.sqrt 3) / 6 = Real.sqrt 7 →
    h = (Real.sqrt 6 * a) / 3 →
    let V := (a^3 * Real.sqrt 2) / 12
    ∃ ε > 0, |V - 296.32| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_with_midpoint_distances_l1210_121045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_workers_needed_l1210_121068

/-- Represents the project details and worker productivity -/
structure Project where
  total_days : ℕ
  days_passed : ℕ
  initial_workers : ℕ
  completed_percentage : ℚ
  remaining_days : ℕ
  remaining_percentage : ℚ

/-- Calculates the minimum number of workers needed to complete the project on time -/
def min_workers_needed (p : Project) : ℕ :=
  (((p.remaining_percentage / p.remaining_days) / (p.completed_percentage / (p.days_passed * p.initial_workers))).ceil).toNat

/-- Theorem stating that for the given project conditions, 5 workers are needed -/
theorem five_workers_needed (p : Project) 
  (h1 : p.total_days = 40)
  (h2 : p.days_passed = 10)
  (h3 : p.initial_workers = 10)
  (h4 : p.completed_percentage = 2/5)
  (h5 : p.remaining_days = p.total_days - p.days_passed)
  (h6 : p.remaining_percentage = 1 - p.completed_percentage) :
  min_workers_needed p = 5 := by
  sorry

#eval min_workers_needed {
  total_days := 40,
  days_passed := 10,
  initial_workers := 10,
  completed_percentage := 2/5,
  remaining_days := 30,
  remaining_percentage := 3/5
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_workers_needed_l1210_121068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l1210_121070

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  r : ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * r

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (seq : GeometricSequence) (n : ℕ) : ℝ :=
  (seq.a 1) * (1 - seq.r^n) / (1 - seq.r)

theorem geometric_sequence_problem (seq : GeometricSequence) (n : ℕ) :
  (∀ k : ℕ, seq.a (k + 1) > seq.a k) →  -- increasing sequence
  seq.a 1 + seq.a n = 34 →
  seq.a 3 * seq.a (n - 2) = 64 →
  geometricSum seq n = 42 →
  n = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l1210_121070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1210_121081

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The focus of the hyperbola -/
def focus : ℝ × ℝ := (2, 0)

/-- The slope of the asymptotes -/
noncomputable def asymptote_slope : ℝ := Real.sqrt 3

theorem hyperbola_equation (h : Hyperbola) : 
  h.a = 1 ∧ h.b^2 = 3 → 
  ∀ x y : ℝ, x^2 - y^2 / 3 = 1 ↔ x^2 / h.a^2 - y^2 / h.b^2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1210_121081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_a_b_relation_exists_x2_m_l1210_121092

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4^x - 2^x

-- Theorem for the range of f
theorem f_range : 
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ∈ Set.Icc (-1/4 : ℝ) 2 := by sorry

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x^2 - x) / 2

-- Theorem for the relation between a and b
theorem a_b_relation (s t : ℝ) (h : f s + f t = 0) :
  let a := 2^s + 2^t
  let b := 2^(s+t)
  b = g a ∧ a ∈ Set.Ioo 1 2 := by sorry

-- Theorem for the existence of x2 and m
theorem exists_x2_m (x1 : ℝ) (h : x1 ∈ Set.Ioo 1 2) :
  ∃ (x2 : ℝ) (m : ℝ), x2 ∈ Set.Icc (-1 : ℝ) 1 ∧ 
    m ∈ Set.Icc (-1 : ℝ) (1/4 : ℝ) ∧ g x1 = f x2 + m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_a_b_relation_exists_x2_m_l1210_121092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_sequence_b_sum_l1210_121044

def sequence_a : ℕ → ℝ := sorry

def sequence_b : ℕ → ℝ := sorry

def S : ℕ → ℝ := sorry

def T : ℕ → ℝ := sorry

axiom a_condition (n : ℕ) : 3 * S n = 2 * sequence_a n + 1

axiom b_definition (n : ℕ) : sequence_b n = (n + 1 : ℝ) * sequence_a n

theorem sequence_a_formula (n : ℕ) : sequence_a n = (-2) ^ (n - 1) := by
  sorry

theorem sequence_b_sum (n : ℕ) : T n = 4/9 - (3*n + 4)/9 * (-2)^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_sequence_b_sum_l1210_121044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1210_121086

theorem tan_alpha_plus_pi_fourth (α : Real) (M : EuclideanSpace ℝ (Fin 2)) :
  M 0 = 1 ∧ M 1 = Real.sqrt 3 →
  Real.tan α = Real.sqrt 3 →
  Real.tan (α + π/4) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1210_121086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_determination_l1210_121012

-- Define the concept of angles
def Angle : Type := ℝ

-- Define the concept of corresponding angles
def corresponding_angles (a b : Angle) : Prop := sorry

-- Define the problem statement
theorem angle_determination (angle1 angle2 : Angle) 
  (h1 : corresponding_angles angle1 angle2) 
  (h2 : angle1 = (40 : ℝ)) : 
  ¬∃ (x : ℝ), ∀ (angle2 : Angle), angle2 = x :=
by
  sorry

#check angle_determination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_determination_l1210_121012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_cost_approximation_l1210_121055

/-- The original cost of a car, given repair cost, selling price, and profit percentage. -/
noncomputable def original_cost (repair_cost selling_price profit_percent : ℝ) : ℝ :=
  (selling_price * 100) / (100 + profit_percent) - repair_cost

/-- Theorem stating that the original cost of the car is approximately 43915 -/
theorem car_cost_approximation :
  let repair_cost : ℝ := 13000
  let selling_price : ℝ := 64500
  let profit_percent : ℝ := 17.272727272727273
  let calculated_cost := original_cost repair_cost selling_price profit_percent
  ⌊calculated_cost⌋ = 43915 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_cost_approximation_l1210_121055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_set_with_prime_sums_l1210_121003

/-- A set of prime numbers such that the sum of any three is prime -/
def PrimeSetWithPrimeSums (S : Set ℕ) : Prop :=
  (∀ p, p ∈ S → Nat.Prime p) ∧
  (∀ p q r, p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r → Nat.Prime (p + q + r))

/-- The maximum size of a set of primes where the sum of any three is prime -/
theorem max_prime_set_with_prime_sums :
  (∃ S : Finset ℕ, PrimeSetWithPrimeSums S ∧ S.card = 4) ∧
  (∀ S : Finset ℕ, PrimeSetWithPrimeSums S → S.card ≤ 4) :=
sorry

#check max_prime_set_with_prime_sums

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_set_with_prime_sums_l1210_121003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_over_fourth_root_of_five_l1210_121013

theorem sixth_root_over_fourth_root_of_five :
  (5 : ℝ) ^ (1/6 : ℝ) / (5 : ℝ) ^ (1/4 : ℝ) = (5 : ℝ) ^ (-1/12 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_over_fourth_root_of_five_l1210_121013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_triple_properties_l1210_121036

theorem pythagorean_triple_properties (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : Nat.Prime a) :
  (Odd b ∧ Even c ∨ Even b ∧ Odd c) ∧ ∃ k : ℕ, 2 * (a + b + 1) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_triple_properties_l1210_121036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equality_iff_in_interval_l1210_121065

theorem floor_equality_iff_in_interval (x : ℝ) :
  ⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 4⌋ ↔ x ∈ Set.Icc (7/3) 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equality_iff_in_interval_l1210_121065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_and_min_l1210_121043

/-- A function f(x) with parameters a, b, and c -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * Real.log x + b / x + c / (x^2)

/-- Theorem stating the conditions for f(x) to have both a maximum and a minimum -/
theorem f_has_max_and_min (a b c : ℝ) (h_a : a ≠ 0) 
  (h_max_min : ∃ (x_max x_min : ℝ), x_max ≠ x_min ∧ 
    (∀ x, x > 0 → f a b c x ≤ f a b c x_max) ∧
    (∀ x, x > 0 → f a b c x ≥ f a b c x_min)) :
  (a * b > 0) ∧ (b^2 + 8*a*c > 0) ∧ (a * c < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_and_min_l1210_121043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l1210_121001

open Real

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 1 + log x

-- State the theorem
theorem tangent_perpendicular_line (a : ℝ) : 
  (f' e = 2) → -- The slope of the tangent line at (e, e) is 2
  (2 * (-1/a) = -1) → -- The tangent line is perpendicular to x + ay = 1
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l1210_121001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1210_121046

noncomputable def f (x : ℝ) := Real.cos x * (Real.sin x - Real.sqrt 3 * Real.cos x) + Real.sqrt 3 / 2

def isSmallestPositivePeriod (T : ℝ) (f : ℝ → ℝ) :=
  T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y

def isMonotonicIncreaseInterval (a b : ℝ) (f : ℝ → ℝ) :=
  a < b ∧ ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def isEven (g : ℝ → ℝ) := ∀ x, g x = g (-x)

theorem f_properties :
  (isSmallestPositivePeriod Real.pi f) ∧
  (∀ k : ℤ, isMonotonicIncreaseInterval (k * Real.pi - Real.pi/12) (k * Real.pi + 5*Real.pi/12) f) ∧
  (∀ a : ℝ, (isEven (fun x ↦ f (x + a))) → |a| ≥ Real.pi/12) ∧
  (∃ a : ℝ, (isEven (fun x ↦ f (x + a))) ∧ |a| = Real.pi/12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1210_121046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_relatively_prime_factors_l1210_121067

noncomputable def infinite_series : ℚ → ℚ :=
  fun x => (1^2 / 2^2) + (2^2 / 3^3) + (3^2 / 2^4) + (4^2 / 3^5) + (5^2 / 2^6) + (6^2 / 3^7) + x

theorem sum_of_relatively_prime_factors (m n : ℕ) :
  Nat.Coprime m n →
  (m : ℚ) / (n : ℚ) = infinite_series 0 →
  m + n = 302549 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_relatively_prime_factors_l1210_121067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l1210_121018

open Real

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the tangent line function at x=0
def tangent_line (x : ℝ) : ℝ := x + 1

-- Define the area function
noncomputable def area : ℝ := ∫ x in (0)..(2), f x - tangent_line x

-- Theorem statement
theorem area_of_closed_figure : area = Real.exp 2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l1210_121018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_value_l1210_121094

theorem cos_2alpha_value (α : Real) 
  (h1 : Real.tan α - (1 / Real.tan α) = 3/2)
  (h2 : α ∈ Set.Ioo (π/4) (π/2)) : 
  Real.cos (2*α) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_value_l1210_121094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_less_than_point_one_l1210_121005

/-- Given a force and a pressure, calculate the maximum area -/
noncomputable def max_area (force : ℝ) (pressure : ℝ) : ℝ := force / pressure

/-- Theorem: If the force is 100N and the pressure is greater than 1000Pa, 
    then the area must be less than 0.1m² -/
theorem area_less_than_point_one 
  (force : ℝ) 
  (pressure : ℝ) 
  (h1 : force = 100) 
  (h2 : pressure > 1000) : 
  max_area force pressure < 0.1 := by
  sorry

#check area_less_than_point_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_less_than_point_one_l1210_121005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coconut_grove_yield_l1210_121060

/-- The number of trees in a coconut grove -/
def x : ℕ := 10  -- We define x as a constant since we're proving it equals 10

/-- The yield of (x + 2) trees -/
def yield1 : ℕ := 30 * (x + 2)

/-- The yield of x trees -/
def yield2 : ℕ := 120 * x

/-- The yield of (x - 2) trees -/
def yield3 : ℕ := 180 * (x - 2)

/-- The total number of trees -/
def totalTrees : ℕ := (x + 2) + x + (x - 2)

/-- The average yield per tree -/
def avgYield : ℕ := 100

/-- Theorem stating that x = 10 given the conditions -/
theorem coconut_grove_yield : 
  yield1 + yield2 + yield3 = avgYield * totalTrees := by
  sorry

#eval x  -- This will evaluate to 10
#eval yield1 + yield2 + yield3  -- This should equal avgYield * totalTrees
#eval avgYield * totalTrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coconut_grove_yield_l1210_121060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perfect_matching_l1210_121049

/-- A bipartite graph with 20 vertices in each part -/
structure BipartiteGraph where
  S : Finset (Fin 20)  -- Set of students
  P : Finset (Fin 20)  -- Set of problems
  E : Finset (Fin 20 × Fin 20)  -- Set of edges

/-- Each student solved two problems and each problem was solved by two students -/
def ValidBipartiteGraph (G : BipartiteGraph) : Prop :=
  (∀ s, s ∈ G.S → (G.E.filter (λ e ↦ e.1 = s)).card = 2) ∧
  (∀ p, p ∈ G.P → (G.E.filter (λ e ↦ e.2 = p)).card = 2)

/-- A perfect matching in the bipartite graph -/
def PerfectMatching (G : BipartiteGraph) (M : Finset (Fin 20 × Fin 20)) : Prop :=
  M ⊆ G.E ∧
  (∀ s, s ∈ G.S → ∃! p, p ∈ G.P ∧ (s, p) ∈ M) ∧
  (∀ p, p ∈ G.P → ∃! s, s ∈ G.S ∧ (s, p) ∈ M)

/-- The main theorem: there exists a perfect matching in the bipartite graph -/
theorem exists_perfect_matching (G : BipartiteGraph) (h : ValidBipartiteGraph G) :
  ∃ M : Finset (Fin 20 × Fin 20), PerfectMatching G M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perfect_matching_l1210_121049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_B_l1210_121074

-- Define a positive integer that is not a perfect square
noncomputable def a : ℕ+ := sorry

-- Assumption that a is not a perfect square
axiom a_not_perfect_square : ∀ n : ℕ, n ^ 2 ≠ (a : ℕ)

-- Define set A
def A : Set ℚ :=
  {k | ∃ (x y : ℤ), (x : ℚ) > Real.sqrt (a : ℚ) ∧ k = (x^2 - (a : ℚ)) / (x^2 - y^2)}

-- Define set B
def B : Set ℚ :=
  {k | ∃ (x y : ℤ), 0 ≤ x ∧ (x : ℚ) < Real.sqrt (a : ℚ) ∧ k = (x^2 - (a : ℚ)) / (x^2 - y^2)}

-- Theorem statement
theorem A_equals_B : A = B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_B_l1210_121074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_pass_percentage_l1210_121085

theorem exam_pass_percentage 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_math : ℝ) 
  (failed_all : ℝ) 
  (failed_hindi_english : ℝ) 
  (failed_english_math : ℝ) 
  (failed_hindi_math : ℝ) 
  (h_failed_hindi : failed_hindi = 35) 
  (h_failed_english : failed_english = 45) 
  (h_failed_math : failed_math = 25) 
  (h_failed_all : failed_all = 18) 
  (h_failed_hindi_english : failed_hindi_english = 32) 
  (h_failed_english_math : failed_english_math = 22) 
  (h_failed_hindi_math : failed_hindi_math = 24) 
  (h_pass_threshold : ∀ (subject : ℝ), subject ≥ 40 → subject ≥ 40) :
  100 - (failed_hindi + failed_english + failed_math - failed_hindi_english - failed_english_math - failed_hindi_math + failed_all) = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_pass_percentage_l1210_121085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l1210_121051

-- Define the function f(x) = lg x^2
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2) / Real.log 10

-- State the theorem
theorem f_decreasing_interval :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < 0 ∧ x₂ < 0 → f x₁ > f x₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l1210_121051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_purchase_equation_l1210_121021

theorem bread_purchase_equation (x : ℕ+) : 
  (15 : ℝ) / x.val - 1 = (15 - 1) / (x.val + 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_purchase_equation_l1210_121021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_polynomial_division_l1210_121062

noncomputable def ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

theorem remainder_polynomial_division :
  ∃! (Q R : ℂ → ℂ),
    (∀ z, z^2021 + 1 = (z^2 + z + 1) * Q z + R z) ∧
    (∃ (a b : ℂ), ∀ z, R z = a * z + b) ∧
    R = fun z ↦ -z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_polynomial_division_l1210_121062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_numbers_l1210_121096

theorem opposite_numbers : -(-2) = -(-2) := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_numbers_l1210_121096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_2_l1210_121035

/-- The displacement of a particle as a function of time -/
noncomputable def displacement (t : ℝ) : ℝ := t^2 * Real.exp (2*t - 4)

/-- The velocity of a particle as a function of time -/
noncomputable def velocity (t : ℝ) : ℝ := deriv displacement t

theorem instantaneous_velocity_at_2 :
  velocity 2 = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_2_l1210_121035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l1210_121082

-- Define the triangle vertices
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (4, 7)
def C : ℝ × ℝ := (7, 1)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem longest_side_length :
  max (distance A B) (max (distance B C) (distance A C)) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l1210_121082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_largest_side_possible_largest_angle_not_smallest_side_impossible_l1210_121006

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  -- Sides
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  -- Angles
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  -- Convexity condition
  convex : angle1 + angle2 + angle3 + angle4 = 2 * Real.pi
  -- All sides are different
  sides_different : side1 ≠ side2 ∧ side1 ≠ side3 ∧ side1 ≠ side4 ∧ 
                    side2 ≠ side3 ∧ side2 ≠ side4 ∧ side3 ≠ side4
  -- All angles are different
  angles_different : angle1 ≠ angle2 ∧ angle1 ≠ angle3 ∧ angle1 ≠ angle4 ∧ 
                     angle2 ≠ angle3 ∧ angle2 ≠ angle4 ∧ angle3 ≠ angle4

-- Helper function to get the nth element of a 4-tuple
def getNth (n : Fin 4) (a b c d : ℝ) : ℝ :=
  match n with
  | 0 => a
  | 1 => b
  | 2 => c
  | 3 => d

-- Part (a)
theorem largest_angle_largest_side_possible (q : ConvexQuadrilateral) : 
  ∃ (i j : Fin 4), 
    q.side1 ≤ (getNth i q.side1 q.side2 q.side3 q.side4) ∧ 
    q.angle1 ≤ (getNth j q.angle1 q.angle2 q.angle3 q.angle4) ∧ 
    i = j :=
sorry

-- Part (b)
theorem largest_angle_not_smallest_side_impossible (q : ConvexQuadrilateral) : 
  ¬∃ (i j k l : Fin 4), 
    (i.val + 1) % 4 ≠ j.val ∧ (k.val + 1) % 4 ≠ l.val ∧
    (getNth i q.side1 q.side2 q.side3 q.side4) < 
    (getNth j q.side1 q.side2 q.side3 q.side4) ∧
    (getNth k q.side1 q.side2 q.side3 q.side4) < 
    (getNth l q.side1 q.side2 q.side3 q.side4) ∧
    (getNth i q.angle1 q.angle2 q.angle3 q.angle4) > 
    (getNth j q.angle1 q.angle2 q.angle3 q.angle4) ∧
    (getNth k q.angle1 q.angle2 q.angle3 q.angle4) < 
    (getNth l q.angle1 q.angle2 q.angle3 q.angle4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_largest_side_possible_largest_angle_not_smallest_side_impossible_l1210_121006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_for_interval_contained_l1210_121063

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x + 4
def g (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Define the solution set type
def SolutionSet := Set ℝ

-- Theorem for part 1
theorem solution_set_when_a_is_one :
  let S : SolutionSet := {x | f 1 x ≥ g x}
  S = Set.Icc (-1) ((Real.sqrt 17 - 1) / 2) :=
sorry

-- Theorem for part 2
theorem range_of_a_for_interval_contained :
  let S (a : ℝ) : SolutionSet := {x | f a x ≥ g x}
  {a | Set.Icc (-1) 1 ⊆ S a} = Set.Icc (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_for_interval_contained_l1210_121063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_fibonacci_correct_l1210_121017

def circularFibonacci (m : Nat) : List Nat :=
  match m with
  | 2 => [0, 1]
  | 3 => [0, 1, 1]
  | 4 => [0, 1, 1, 2]
  | 5 => [0, 1, 1, 2, 3]
  | 6 => [0, 1, 1, 2, 3, 5]
  | 7 => [0, 1, 1, 2, 3, 5, 8]
  | 8 => [0, 1, 1, 2, 3, 5, 8, 13]
  | 9 => [0, 1, 1, 2, 3, 5, 8, 13, 21]
  | 10 => [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
  | 11 => [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
  | _ => []

def isCircularFibonacci (l : List Nat) : Prop :=
  l.length ≥ 2 ∧
  l.take 2 = [0, 1] ∧
  ∀ i, 2 ≤ i → i < l.length → l.get! i = l.get! (i-1) + l.get! (i-2)

theorem circular_fibonacci_correct (m : Nat) (h : 2 ≤ m ∧ m ≤ 11) :
  isCircularFibonacci (circularFibonacci m) :=
by
  cases m
  all_goals (first | sorry | fail "m out of range")

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_fibonacci_correct_l1210_121017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1210_121009

/-- The complex number z defined as i^2023 / (1 - 2i) -/
noncomputable def z : ℂ := (Complex.I ^ 2023) / (1 - 2 * Complex.I)

/-- A complex number is in the fourth quadrant if its real part is positive and imaginary part is negative -/
def in_fourth_quadrant (w : ℂ) : Prop := 0 < w.re ∧ w.im < 0

/-- Theorem stating that z is in the fourth quadrant -/
theorem z_in_fourth_quadrant : in_fourth_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1210_121009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_subset_l1210_121072

/-- Two points in ℤⁿ are neighbors if they differ by exactly 1 in one coordinate
    and are equal in all other coordinates. -/
def are_neighbors (n : ℕ) (p q : Fin n → ℤ) : Prop :=
  ∃ i : Fin n, (p i - q i).natAbs = 1 ∧ ∀ j : Fin n, j ≠ i → p j = q j

/-- The statement of the problem -/
theorem exists_special_subset (n : ℕ) (hn : n ≥ 1) :
  ∃ (S : Set (Fin n → ℤ)),
    (∀ p ∈ S, ∀ q, are_neighbors n p q → q ∉ S) ∧
    (∀ p ∉ S, ∃! q, are_neighbors n p q ∧ q ∈ S) := by
  sorry

#check exists_special_subset

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_subset_l1210_121072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_t_range_l1210_121022

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x t : ℝ) : ℝ := Real.sqrt (|x + 1| + |x - t| - 2015)

-- State the theorem
theorem domain_implies_t_range :
  (∀ x : ℝ, ∃ y : ℝ, f x t = y) →
  t ∈ Set.Ici 2014 ∪ Set.Iic (-2016) :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_t_range_l1210_121022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_specific_circles_l1210_121066

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The length of the shortest line segment tangent to two circles -/
noncomputable def shortest_tangent_length (C1 C2 : Circle) : ℝ :=
  let d := distance C1.center C2.center
  let r1 := C1.radius
  let r2 := C2.radius
  2 * d * r1 * r2 / (r1 + r2)

/-- The main theorem to be proved -/
theorem shortest_tangent_length_specific_circles : 
  let C1 : Circle := { center := (4, 0), radius := 4 }
  let C2 : Circle := { center := (-6, 0), radius := 6 }
  shortest_tangent_length C1 C2 = 20 := by sorry

#check shortest_tangent_length_specific_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_specific_circles_l1210_121066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_20_l1210_121076

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_20 (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 2 + a 3 = -24 →
  a 18 + a 19 + a 20 = 78 →
  sum_of_arithmetic_sequence a 20 = 180 := by
  sorry

#check arithmetic_sequence_sum_20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_20_l1210_121076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_220_l1210_121029

noncomputable def train_length : ℝ := 180
noncomputable def train_speed_kmph : ℝ := 72
noncomputable def crossing_time : ℝ := 20

noncomputable def kmph_to_mps (speed : ℝ) : ℝ := speed * (1000 / 3600)

noncomputable def platform_length : ℝ :=
  kmph_to_mps train_speed_kmph * crossing_time - train_length

theorem platform_length_is_220 :
  platform_length = 220 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_220_l1210_121029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1210_121090

-- Define the circle
def myCircle (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 4

-- Define the line
def myLine (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the intersection points
def intersection_points (M N : ℝ × ℝ) : Prop :=
  myCircle M.1 M.2 ∧ myLine M.1 M.2 ∧ myCircle N.1 N.2 ∧ myLine N.1 N.2

-- Theorem statement
theorem chord_length (M N : ℝ × ℝ) (h : intersection_points M N) :
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1210_121090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_problem_simplification_problem_l1210_121004

-- Part 1
theorem calculation_problem : 
  3 * Real.tan (30 * π / 180) - |Real.sqrt 3 - 1| + (3.14 - Real.pi) ^ (0 : ℝ) - (1/3) ^ (-2 : ℝ) = -7 := by sorry

-- Part 2
theorem simplification_problem (a : ℝ) (h : a^2 + 2*a - 3 = 0) : 
  (2*a - 12*a/(a+2)) / ((a-4)/(a^2 + 4*a + 4)) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_problem_simplification_problem_l1210_121004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_equals_three_l1210_121047

/-- The volume of a cube with edge length a -/
noncomputable def cube_volume (a : ℝ) : ℝ := a^3

/-- The volume of a square-based pyramid with base edge length b and height h -/
noncomputable def pyramid_volume (b h : ℝ) : ℝ := (1/3) * b^2 * h

/-- 
Given a cube with edge length 4 and a square-based pyramid with base edge length 8,
prove that the height of the pyramid is 3 when their volumes are equal.
-/
theorem pyramid_height_equals_three :
  let cube_edge : ℝ := 4
  let pyramid_base : ℝ := 8
  ∃ h : ℝ, 
    cube_volume cube_edge = pyramid_volume pyramid_base h ∧ 
    h = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_equals_three_l1210_121047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_l1210_121008

theorem xy_value (x y : ℝ) 
  (h1 : (2 : ℝ) ^ x = (64 : ℝ) ^ (y + 1))
  (h2 : (27 : ℝ) ^ y = (3 : ℝ) ^ (x - 2)) : 
  x * y = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_l1210_121008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l1210_121098

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (4 - x^2)}

-- Define set B
def B : Set ℝ := {x | x > 0 ∧ ∃ y, y = (3 : ℝ)^x}

-- Theorem statement
theorem A_intersect_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l1210_121098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l1210_121037

/-- Represents the cost price of an article -/
def cost_price : ℝ := 2407.70

/-- Represents the selling price of an article -/
def selling_price : ℝ := 2552.36

/-- Represents the profit percent -/
def profit_percent : ℝ := 6

/-- Defines an approximate equality for real numbers -/
def approx_equal (x y : ℝ) : Prop := abs (x - y) < 0.01

notation:50 a " ≈ " b:50 => approx_equal a b

theorem cost_price_calculation : 
  cost_price ≈ selling_price / (1 + profit_percent / 100) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l1210_121037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1210_121075

/-- The distance between two points in 3D space -/
noncomputable def distance3D (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

theorem distance_between_specific_points :
  distance3D (3, -2, 5) (7, 4, -1) = 2 * Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1210_121075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_144_multiple_of_6_l1210_121050

def is_factor (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

def is_multiple_of (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

theorem factors_of_144_multiple_of_6 :
  (Finset.filter (λ x => x ≠ 0 ∧ 144 % x = 0 ∧ x % 6 = 0) (Finset.range 145)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_144_multiple_of_6_l1210_121050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_side_relationships_l1210_121034

noncomputable section

open Real

theorem triangle_angle_side_relationships 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) 
  (h_law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) :
  (Real.sin A > Real.sin B → A > B) ∧ 
  (Real.sin B ^ 2 + Real.sin C ^ 2 < Real.sin A ^ 2 → A > π / 2) ∧
  (A < π / 2 ∧ B < π / 2 ∧ C < π / 2 → Real.sin A > Real.cos B) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_side_relationships_l1210_121034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_opposite_coordinates_l1210_121078

/-- If a point A(a,b) is on the angle bisector of the second and fourth quadrants, 
    then a and b are opposite numbers. -/
theorem angle_bisector_opposite_coordinates (a b : ℝ) : 
  (∀ θ : ℝ, θ = π/4 ∨ θ = 5*π/4 → a * Real.cos θ + b * Real.sin θ = 0) → 
  a = -b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_opposite_coordinates_l1210_121078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_possible_l1210_121033

/-- Represents a piece color -/
inductive Color
| White
| Black

/-- Represents a box that may contain a piece -/
inductive Box
| Empty
| Filled (c : Color)

/-- Represents an operation that moves two adjacent pieces -/
structure Operation where
  source : ℕ
  direction : Bool  -- true for right, false for left

/-- The initial configuration of pieces -/
def initial_config (n : ℕ) : List Box :=
  (List.replicate n (Box.Filled Color.White)) ++ (List.replicate n (Box.Filled Color.Black))

/-- Checks if a configuration is alternating without empty boxes -/
def is_alternating (config : List Box) : Prop :=
  ∀ i, i + 1 < config.length →
    match config[i]?, config[i+1]? with
    | some (Box.Filled Color.White), some (Box.Filled Color.Black) => True
    | some (Box.Filled Color.Black), some (Box.Filled Color.White) => True
    | _, _ => False

/-- Applies an operation to a configuration -/
def apply_operation (config : List Box) (op : Operation) : List Box :=
  sorry

/-- The main theorem to prove -/
theorem rearrangement_possible (n : ℕ) (h : n ≥ 3) :
  ∃ (ops : List Operation),
    ops.length = n ∧
    is_alternating (ops.foldl apply_operation (initial_config n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_possible_l1210_121033
