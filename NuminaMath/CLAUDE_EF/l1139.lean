import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_l1139_113942

theorem subset_condition (a : ℝ) : 
  let A := {x : ℝ | a * x = x^2}
  let B := ({0, 1, 2} : Set ℝ)
  A ⊆ B ↔ a ∈ ({0, 1, 2} : Set ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_l1139_113942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l1139_113907

/-- The time taken to fill the tank without leakage -/
def fill_time : ℝ → ℝ := λ t => t

/-- The time taken to empty the tank due to leakage -/
def empty_time : ℝ := 70

/-- The time taken to fill the tank with leakage -/
def fill_time_with_leakage (t : ℝ) : ℝ := 7 * t

theorem pipe_fill_time :
  ∃ t : ℝ, t > 0 ∧ fill_time t = 60 ∧
  (fill_time_with_leakage t * (1 / t - 1 / empty_time) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l1139_113907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_mastermind_l1139_113913

-- Define the suspects
inductive Suspect : Type
  | A | B | C | D

-- Define a function to represent each suspect's statement
def statement (s : Suspect) (mastermind : Suspect) : Prop :=
  match s with
  | Suspect.A => mastermind = Suspect.C
  | Suspect.B => mastermind ≠ Suspect.B
  | Suspect.C => mastermind ≠ Suspect.C
  | Suspect.D => mastermind = Suspect.C  -- Equivalent to A's statement

-- Define the condition that only one statement is true
def only_one_true (mastermind : Suspect) : Prop :=
  ∃! s : Suspect, statement s mastermind

-- Theorem statement
theorem unique_mastermind :
  ∃! mastermind : Suspect, only_one_true mastermind :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_mastermind_l1139_113913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_quadrilateral_ratio_l1139_113958

open Real

noncomputable section

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a Line type
structure Line where
  a : Point
  b : Point

-- Define a Semicircle type
structure Semicircle where
  center : Point
  radius : ℝ
  radius_pos : radius > 0

-- Define a Quadrilateral type
structure Quadrilateral (ω : Semicircle) where
  X : Point
  A : Point
  B : Point
  Y : Point
  on_semicircle : True  -- Placeholder for the actual condition
  diameter : True       -- Placeholder for the actual condition

-- Define distance function
def dist (p q : Point) : ℝ := sorry

-- Define intersection of two lines
def intersect (l1 l2 : Line) : Point := sorry

-- Define perpendicular lines
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define a point being on a line
def on_line (p : Point) (l : Line) : Prop := sorry

-- Define membership in a semicircle
instance : Membership Point Semicircle where
  mem := λ p ω => dist p ω.center = ω.radius

theorem semicircle_quadrilateral_ratio 
  (ω : Semicircle) 
  (quad : Quadrilateral ω) 
  (P : Point) 
  (C : Point) 
  (Z : Point) 
  (Q : Point) :
  C ∈ ω →
  P = intersect (Line.mk quad.A quad.Y) (Line.mk quad.B quad.X) →
  on_line Z (Line.mk quad.X quad.Y) →
  perpendicular (Line.mk P Z) (Line.mk quad.X quad.Y) →
  perpendicular (Line.mk quad.X C) (Line.mk quad.A Z) →
  Q = intersect (Line.mk quad.A quad.Y) (Line.mk quad.X C) →
  dist quad.B quad.Y / dist quad.X P + dist C quad.Y / dist quad.X Q = 
  dist quad.A quad.Y / dist quad.A quad.X := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_quadrilateral_ratio_l1139_113958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_shape_circle_covering_l1139_113957

/-- A convex shape in 2D space -/
structure ConvexShape where
  -- We don't need to define the internal structure of the shape for this statement

/-- Represents the property of a shape covering another shape -/
def covers (F : ConvexShape) (S : Set (ℝ × ℝ)) : Prop :=
  sorry -- Definition of covering

/-- A circle in 2D space -/
def Circle (R : ℝ) : Set (ℝ × ℝ) :=
  sorry -- Definition of a circle with radius R

/-- A semicircle in 2D space -/
def Semicircle (R : ℝ) : Set (ℝ × ℝ) :=
  sorry -- Definition of a semicircle with radius R

/-- Two copies of a shape -/
def TwoCopies (F : ConvexShape) : ConvexShape :=
  sorry -- Definition of two copies of F as a single ConvexShape

theorem convex_shape_circle_covering (R : ℝ) :
  ∃ F : ConvexShape,
    (¬ covers F (Semicircle R)) ∧
    (covers (TwoCopies F) (Circle R)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_shape_circle_covering_l1139_113957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_and_b_range_l1139_113960

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 3 * x - 1 / x + b * Real.log x

theorem local_minimum_and_b_range :
  (∃ (x : ℝ), IsLocalMin (f (-4)) x ∧ f (-4) x = 2) ∧
  (∀ b : ℝ, (∃ x ∈ Set.Icc 1 (Real.exp 1), 4 * x - 1 / x - f b x < -(1 + b) / x) ↔
    b ∈ Set.Ioi ((Real.exp 2 + 1) / (Real.exp 1 - 1)) ∪ Set.Iio (-2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_and_b_range_l1139_113960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_min_a_for_bounded_slope_real_roots_count_l1139_113946

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Theorem 1: Monotonicity of f when a = 1
theorem f_monotonicity :
  ∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ →
  (1 < x₁ ∧ x₁ < x₂ → f 1 x₁ < f 1 x₂) ∧
  (0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f 1 x₁ > f 1 x₂) :=
sorry

-- Theorem 2: Minimum value of a when tangent slope ≤ 1/2
theorem min_a_for_bounded_slope :
  ∀ a : ℝ, a > 0 →
  (∀ x : ℝ, x > 0 → (deriv (f a)) x ≤ 1/2) →
  a ≥ 1/2 :=
sorry

-- Theorem 3: Number of real roots
theorem real_roots_count (a b : ℝ) :
  a > 0 →
  (∀ x : ℝ, x > 0 → f a x = (x^3 + 2*(b*x + a))/(2*x) - 1/2) →
  ((b < 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = (x₁^3 + 2*(b*x₁ + a))/(2*x₁) - 1/2 ∧
                          f a x₂ = (x₂^3 + 2*(b*x₂ + a))/(2*x₂) - 1/2) ∧
   (b = 0 → ∃! x : ℝ, f a x = (x^3 + 2*(b*x + a))/(2*x) - 1/2) ∧
   (b > 0 → ¬∃ x : ℝ, f a x = (x^3 + 2*(b*x + a))/(2*x) - 1/2)) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_min_a_for_bounded_slope_real_roots_count_l1139_113946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_total_approximation_l1139_113930

-- Define the compound interest function
noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r) ^ t

-- Define the problem parameters
def total_investment : ℝ := 7000
def P1 : ℝ := 2000
def P2 : ℝ := 2500
def P3 : ℝ := total_investment - P1 - P2
def r1 : ℝ := 0.06
def r2 : ℝ := 0.085
def r3 : ℝ := 0.07
def t : ℝ := 18

-- Define the theorem
theorem investment_total_approximation :
  let A1 := compound_interest P1 r1 t
  let A2 := compound_interest P2 r2 t
  let A3 := compound_interest P3 r3 t
  abs ((A1 + A2 + A3) - 24605.11) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_total_approximation_l1139_113930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2_implies_cos2theta_tan_theta_minus_pi4_eq_neg_one_fifth_l1139_113978

theorem tan_2_implies_cos2theta_tan_theta_minus_pi4_eq_neg_one_fifth
  (θ : ℝ)
  (h : Real.tan θ = 2) :
  Real.cos (2 * θ) * Real.tan (θ - Real.pi / 4) = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2_implies_cos2theta_tan_theta_minus_pi4_eq_neg_one_fifth_l1139_113978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_categorization_l1139_113935

-- Define the given numbers
noncomputable def numbers : List ℚ := [-8/5, -5/6, 89/10, -7, 1/12, 0, 25]

-- Define the sets
def positive_numbers : Set ℚ := {x | x ∈ numbers ∧ x > 0}
def negative_fractions : Set ℚ := {x | x ∈ numbers ∧ x < 0 ∧ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b ∧ x ≠ ↑(Int.floor x)}
def negative_integers : Set ℚ := {x | x ∈ numbers ∧ x < 0 ∧ x = ↑(Int.floor x)}

-- Theorem to prove
theorem correct_categorization :
  positive_numbers = {89/10, 1/12, 25} ∧
  negative_fractions = {-8/5, -5/6} ∧
  negative_integers = {-7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_categorization_l1139_113935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dave_age_is_five_thirds_l1139_113944

/-- Charlie's age in years -/
def charlie_age : ℝ := sorry

/-- Dave's age in years -/
def dave_age : ℝ := sorry

/-- Eleanor's age in years -/
def eleanor_age : ℝ := sorry

/-- Charlie's age is four times Dave's age -/
axiom charlie_dave_relation : charlie_age = 4 * dave_age

/-- Eleanor is five years older than Dave -/
axiom eleanor_dave_relation : eleanor_age = dave_age + 5

/-- Charlie and Eleanor are twins -/
axiom charlie_eleanor_twins : charlie_age = eleanor_age

theorem dave_age_is_five_thirds : dave_age = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dave_age_is_five_thirds_l1139_113944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minus_median_equals_26_l1139_113917

def weights : List ℚ := [120, 10, 12, 14]

def average (xs : List ℚ) : ℚ := (xs.sum) / xs.length

noncomputable def median (xs : List ℚ) : ℚ :=
  let sorted := xs.toArray.qsort (· ≤ ·)
  let n := sorted.size
  if n % 2 = 0 then
    (sorted[n / 2 - 1]! + sorted[n / 2]!) / 2
  else
    sorted[n / 2]!

theorem average_minus_median_equals_26 :
  average weights - median weights = 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minus_median_equals_26_l1139_113917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_surface_area_in_tetrahedron_l1139_113989

/-- Represents a tetrahedron with a ball inside and partially filled with water -/
structure TetrahedronWithBall where
  edge_length : ℝ
  water_volume_ratio : ℝ
  ball_touches_sides : Prop

/-- Calculates the surface area of the ball inside the tetrahedron -/
noncomputable def ball_surface_area (t : TetrahedronWithBall) : ℝ :=
  2 * Real.pi / 3

/-- Theorem stating the surface area of the ball under given conditions -/
theorem ball_surface_area_in_tetrahedron 
  (t : TetrahedronWithBall) 
  (h1 : t.edge_length = 4)
  (h2 : t.water_volume_ratio = 7/8)
  (h3 : t.ball_touches_sides) :
  ball_surface_area t = 2 * Real.pi / 3 := by
  sorry

#check ball_surface_area_in_tetrahedron

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_surface_area_in_tetrahedron_l1139_113989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_sum_zero_l1139_113908

noncomputable def v (x : ℝ) : ℝ := x + 2 * Real.sin (x * Real.pi / 2)

theorem v_sum_zero : v (-3.14) + v (-0.95) + v 0.95 + v 3.14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_sum_zero_l1139_113908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_color_2017_all_even_blue_two_is_blue_l1139_113911

-- Define the color properties as axioms
axiom is_blue : ℕ → Prop
axiom is_red : ℕ → Prop

-- Define the coloring rules
axiom blue_sum : ∀ a b : ℕ, is_blue a → is_blue b → is_blue (a + b)
axiom red_product : ∀ a b : ℕ, is_red a → is_red b → is_red (a * b)

-- Both colors are used
axiom both_colors_used : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ is_blue a ∧ is_red b

-- 1024 is blue
axiom blue_1024 : is_blue 1024

-- Theorem stating that 2017 is either red or blue
theorem color_2017 : is_red 2017 ∨ is_blue 2017 := by
  sorry

-- Additional theorem: all even numbers are blue
theorem all_even_blue : ∀ n : ℕ, Even n → is_blue n := by
  sorry

-- Theorem: 2 is blue
theorem two_is_blue : is_blue 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_color_2017_all_even_blue_two_is_blue_l1139_113911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1139_113901

theorem triangle_proof (A B C a b c : ℝ) : 
  -- Given conditions
  (Real.sqrt 3 * c) / Real.cos C = a / Real.cos (3 * Real.pi / 2 + A) →
  c / a = 2 →
  b = 4 * Real.sqrt 3 →
  -- Conclusions
  C = Real.pi / 6 ∧ 
  (1 / 2) * a * b * Real.sin C = 2 * Real.sqrt 15 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1139_113901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_size_l1139_113953

theorem min_intersection_size (A B C : Finset ℕ) : 
  (A.card = 50 ∧ B.card = 50 ∧ C.card = 50) →
  ((A.card : ℝ) + (B.card : ℝ) + (C.card : ℝ) = 1.5 * ((A ∪ B ∪ C).card : ℝ)) →
  48 ≤ (A ∩ B ∩ C).card := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_size_l1139_113953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l1139_113926

-- Define variables as real numbers
variable (x y k : ℝ)

-- Define the system of linear equations
def eq1 (x y : ℝ) : Prop := 2 * x - 3 * y = -2
def eq2 (x y k : ℝ) : Prop := x - 2 * y = k

-- Define the constraint
def constraint (x y : ℝ) : Prop := x - y < 0

-- Define the additional inequality
def inequality (x k : ℝ) : Prop := (2 * k + 1) * x < 2 * k + 1

-- Theorem statement
theorem system_solution :
  ∀ x y k : ℝ,
  eq1 x y → eq2 x y k → constraint x y →
  (k > -2) ∧
  (inequality x k → (x > 1) → k = -1 ∨ k < -2) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l1139_113926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_inverse_arithmetic_b_formula_T_formula_l1139_113947

-- Define the sequences a_n, b_n, and c_n
def a : ℕ → ℝ := sorry

def b : ℕ → ℝ := sorry

def c : ℕ → ℝ := sorry

-- Define S_n as the sum of the first n terms of a_n
def S : ℕ → ℝ := sorry

-- Define T_n as the sum of the first n terms of c_n
def T : ℕ → ℝ := sorry

-- Conditions
axiom sum_condition : ∀ n : ℕ, S n + a n = 2
axiom b_initial : b 1 = a 1
axiom b_recurrence : ∀ n : ℕ, n ≥ 2 → b n = (3 * b (n-1)) / (b (n-1) + 3)
axiom c_definition : ∀ n : ℕ, c n = a n / b n

-- Theorems to prove
theorem a_formula : ∀ n : ℕ, a n = 1 / 2^(n-1) := by sorry

theorem b_inverse_arithmetic : ∀ n : ℕ, (1 / b (n+1)) - (1 / b n) = 1 / 3 := by sorry

theorem b_formula : ∀ n : ℕ, b n = 3 / (n + 2) := by sorry

theorem T_formula : ∀ n : ℕ, T n = 8/3 - (n + 4) / (3 * 2^(n-1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_inverse_arithmetic_b_formula_T_formula_l1139_113947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1139_113985

theorem function_inequality (a x₁ x₂ : ℝ) (ha : a > 0) (hx : x₁ < x₂) (hsum : x₁ + x₂ = 0) :
  let f := λ (x : ℝ) => a * x^2 + 2 * a * x + 4
  f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1139_113985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_l1139_113912

/-- Square ABCD with A and B on the x-axis and C and D on the parabola y = x^2 - 4 -/
structure SquareABCD where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  on_x_axis : A.2 = 0 ∧ B.2 = 0
  on_parabola : C.2 = C.1^2 - 4 ∧ D.2 = D.1^2 - 4
  is_square : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
              (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2 ∧
              (A.1 - B.1)^2 + (A.2 - B.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2

/-- The area of square ABCD is 24 - 8√5 -/
theorem square_area (s : SquareABCD) : 
  (s.A.1 - s.B.1)^2 + (s.A.2 - s.B.2)^2 = 24 - 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_l1139_113912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_quadrilateral_angle_sum_l1139_113965

-- Define the parallelogram
structure Parallelogram where
  a : ℝ
  b : ℝ
  sum_eq : a + b = 180
  ratio_eq : a / b = 4 / 11

-- Define the quadrilateral
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  sum_eq : a + b + c + d = 360
  ratio_ab : a / b = 5 / 6
  ratio_bc : b / c = 6 / 7
  ratio_cd : c / d = 7 / 12

-- Theorem statement
theorem parallelogram_quadrilateral_angle_sum
  (p : Parallelogram) (q : Quadrilateral) :
  min p.a p.b + (max (min q.a (max q.b q.c)) (min (max q.a q.b) q.c)) = 132 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_quadrilateral_angle_sum_l1139_113965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_shift_l1139_113925

/-- Given a parabola y = x^2 - 3, shifting it 2 units left and 4 units up results in y = (x + 2)^2 + 1 -/
theorem parabola_shift (x y : ℝ) : 
  (y = x^2 - 3) → 
  (y + 4 = ((x + 2)^2 - 3)) →
  y = (x + 2)^2 + 1 := by
  intros h1 h2
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_shift_l1139_113925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l1139_113969

/-- The slope of the first line -/
noncomputable def m₁ : ℝ := 3

/-- The y-intercept of the first line -/
noncomputable def b₁ : ℝ := -4

/-- A point on the second line -/
noncomputable def p : ℝ × ℝ := (4, 2)

/-- The slope of the second line (perpendicular to the first line) -/
noncomputable def m₂ : ℝ := -1 / m₁

/-- The intersection point to be proven -/
noncomputable def intersection_point : ℝ × ℝ := (2.2, 2.6)

theorem intersection_point_correct :
  let f₁ (x : ℝ) := m₁ * x + b₁
  let f₂ (x : ℝ) := m₂ * (x - p.1) + p.2
  (∃ (x : ℝ), f₁ x = f₂ x ∧ (x, f₁ x) = intersection_point) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l1139_113969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_teachers_l1139_113980

theorem stratified_sampling_teachers (total : ℕ) (senior : ℕ) (intermediate : ℕ) (junior : ℕ) 
  (sample_size : ℕ) (h1 : total = 300) (h2 : senior = 90) (h3 : intermediate = 150) 
  (h4 : junior = 60) (h5 : sample_size = 40) (h6 : total = senior + intermediate + junior) :
  (sample_size * senior) / total = 12 ∧ 
  (sample_size * intermediate) / total = 20 ∧ 
  (sample_size * junior) / total = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_teachers_l1139_113980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oscars_age_l1139_113915

theorem oscars_age : ℕ := by
  -- Define Christina's age in 5 years
  let christina_age_in_5_years := 80 / 2

  -- Calculate Christina's current age
  let christina_current_age := christina_age_in_5_years - 5

  -- Calculate Oscar's age in 15 years
  let oscar_age_in_15_years := 3 * christina_current_age / 5

  -- Calculate Oscar's current age
  let oscar_current_age := oscar_age_in_15_years - 15

  -- Assert that Oscar's current age is 6
  have h : oscar_current_age = 6 := by sorry

  -- Return Oscar's current age
  exact oscar_current_age


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oscars_age_l1139_113915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_cutting_problem_l1139_113906

/-- The length of rope remaining after n cuts, where each cut removes 1/3 of the current length -/
noncomputable def remainingLength (n : ℕ) : ℝ :=
  (2/3) ^ n

/-- The problem statement -/
theorem rope_cutting_problem :
  let initialLength : ℝ := 1
  let numCuts : ℕ := 6
  remainingLength numCuts = (2/3) ^ 6 := by
  -- Unfold the definition of remainingLength
  unfold remainingLength
  -- The equality now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_cutting_problem_l1139_113906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_x_value_l1139_113961

/-- Given two vectors a and b in ℝ², where a = (x, 1) and b = (3, -2),
    and a is perpendicular to b, then x = 2/3. -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (3, -2)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_x_value_l1139_113961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_solid_area_greater_than_cylinder_l1139_113902

/-- Represents a right circular cylinder with inscribed hexagons -/
structure HexagonInscribedCylinder where
  radius : ℝ
  height : ℝ
  num_sections : ℕ
  rotation_angle : ℝ

/-- Calculates the surface area of the cylinder -/
noncomputable def cylinder_surface_area (c : HexagonInscribedCylinder) : ℝ :=
  2 * Real.pi * c.radius * (c.radius + c.height)

/-- Calculates the surface area of the solid formed by hexagon vertices -/
noncomputable def hexagon_solid_surface_area (c : HexagonInscribedCylinder) : ℝ :=
  sorry -- Actual calculation would go here

/-- The main theorem to be proved -/
theorem hexagon_solid_area_greater_than_cylinder 
  (c : HexagonInscribedCylinder) 
  (h1 : c.radius = 0.5) 
  (h2 : c.height = 1)
  (h3 : c.num_sections = 101)
  (h4 : c.rotation_angle = Real.pi / 6) : 
  hexagon_solid_surface_area c > cylinder_surface_area c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_solid_area_greater_than_cylinder_l1139_113902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_decreasing_function_negative_range_l1139_113974

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

theorem even_decreasing_function_negative_range
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_decreasing : is_decreasing_on f (Set.Iic 0))
  (h_f2 : f 2 = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-2) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_decreasing_function_negative_range_l1139_113974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l1139_113941

/-- Given an article with a marked price and selling conditions, calculate the cost price -/
theorem cost_price_calculation (marked_price : ℝ) 
  (h1 : abs (marked_price - 65.25) < 0.01) 
  (h2 : ∀ (cost_price : ℝ), 0.91 * marked_price = 1.25 * cost_price) :
  ∃ (cost_price : ℝ), abs (cost_price - 47.50) < 0.01 := by
  sorry

#check cost_price_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l1139_113941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_speed_theorem_l1139_113996

-- Define constants
def wheel_circumference_feet : ℝ := 9
def feet_per_km : ℝ := 3280.84
def seconds_per_hour : ℝ := 3600
def speed_increase_kmh : ℝ := 3
def time_decrease_seconds : ℝ := 0.2

-- Define the original speed in km/h
noncomputable def original_speed : ℝ := 
  let wheel_circumference_km := wheel_circumference_feet / feet_per_km
  let t := wheel_circumference_km * seconds_per_hour / 7.38 -- Using approximate value to break recursion
  (3 * t * seconds_per_hour - 0.6) / 0.2

-- Theorem statement
theorem wheel_speed_theorem :
  abs (original_speed - 7.38) < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_speed_theorem_l1139_113996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_cubed_minus_3A_l1139_113959

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 3]

theorem det_A_cubed_minus_3A : Matrix.det (A^3 - 3 • A) = -340 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_cubed_minus_3A_l1139_113959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indians_drinking_tea_count_l1139_113956

/-- Represents the nationalities of people in the café. -/
inductive Nationality
| Indian
| Turk
deriving DecidableEq

/-- Represents the drinks available in the café. -/
inductive Drink
| Tea
| Coffee
deriving DecidableEq

/-- Represents a person in the café. -/
structure Person where
  nationality : Nationality
  drink : Drink
deriving DecidableEq

/-- The café scenario with given conditions. -/
structure CafeScenario where
  people : Finset Person
  total_count : Nat
  coffee_yes_count : Nat
  turk_yes_count : Nat
  rain_yes_count : Nat
  h_total_count : people.card = total_count
  h_total_55 : total_count = 55
  h_coffee_yes : coffee_yes_count = 44
  h_turk_yes : turk_yes_count = 33
  h_rain_yes : rain_yes_count = 22

  h_truth_telling : ∀ p : Person, p ∈ people →
    ((p.nationality = Nationality.Indian ∧ p.drink = Drink.Tea) ∨
     (p.nationality = Nationality.Turk ∧ p.drink = Drink.Coffee))
    ↔ ((p.drink = Drink.Coffee → p ∈ (people.filter (λ x => x.drink = Drink.Coffee))) ∧
       (p.nationality = Nationality.Turk → p ∈ (people.filter (λ x => x.nationality = Nationality.Turk))) ∧
       p ∈ (people.filter (λ _ => true)))

/-- The main theorem stating that the number of Indians drinking tea is zero. -/
theorem indians_drinking_tea_count (scenario : CafeScenario) :
  (scenario.people.filter (λ p => p.nationality = Nationality.Indian ∧ p.drink = Drink.Tea)).card = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indians_drinking_tea_count_l1139_113956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_theta_l1139_113954

theorem sin_double_theta (θ : ℝ) (h : Real.cos θ + Real.sin θ = 3/2) : Real.sin (2 * θ) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_theta_l1139_113954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_at_sqrt2_distance_only_two_points_at_sqrt2_distance_l1139_113977

noncomputable section

-- Define the line using a parameter t
def line (t : ℝ) : ℝ × ℝ := (-2 - Real.sqrt 2 * t, 3 + Real.sqrt 2 * t)

-- Define point A
def point_A : ℝ × ℝ := (-2, 3)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_points_at_sqrt2_distance :
  ∀ t : ℝ, distance (line t) point_A = Real.sqrt 2 ↔ t = Real.sqrt 2 / 2 ∨ t = -Real.sqrt 2 / 2 :=
by sorry

theorem only_two_points_at_sqrt2_distance :
  {p : ℝ × ℝ | ∃ t : ℝ, line t = p ∧ distance p point_A = Real.sqrt 2} = {(-3, 4), (-1, 2)} :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_at_sqrt2_distance_only_two_points_at_sqrt2_distance_l1139_113977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_has_twelve_edges_l1139_113916

-- Define a cube
structure Cube where
  -- No specific properties needed for this problem

-- Theorem stating that a cube has 12 edges
theorem cube_has_twelve_edges : Nat := 12

#check cube_has_twelve_edges

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_has_twelve_edges_l1139_113916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_values_l1139_113928

theorem sin_2alpha_values (α : ℝ) :
  Real.sin (α - π / 4) = -Real.cos (2 * α) →
  Real.sin (2 * α) = -1/2 ∨ Real.sin (2 * α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_values_l1139_113928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_for_all_in_S_inequality_holds_for_lambda_in_range_lambda_range_is_tight_l1139_113973

-- Define the set of real numbers a and b satisfying |a| < 1 and |b| < 1
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1| < 1 ∧ |p.2| < 1}

-- Statement 1
theorem inequality_holds_for_all_in_S : ∀ (p : ℝ × ℝ), p ∈ S → |1 - p.1 * p.2| / |p.1 - p.2| > 1 := by sorry

-- Statement 2
theorem inequality_holds_for_lambda_in_range : ∀ (l : ℝ), l ∈ Set.Icc (-1) 1 → 
  ∀ (p : ℝ × ℝ), p ∈ S → |1 - p.1 * p.2 * l| / |p.1 * l - p.2| > 1 := by sorry

-- Statement 3
theorem lambda_range_is_tight : ∀ (l : ℝ), 
  (∀ (p : ℝ × ℝ), p ∈ S → |1 - p.1 * p.2 * l| / |p.1 * l - p.2| > 1) → 
  l ∈ Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_for_all_in_S_inequality_holds_for_lambda_in_range_lambda_range_is_tight_l1139_113973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l1139_113920

theorem cosine_identity (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.cos (θ + π / 12) = Real.sqrt 3 / 3) : 
  Real.cos (5 * π / 12 - θ) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l1139_113920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ages_l1139_113909

/-- Represents the ages of family members -/
structure FamilyAges where
  grace : ℝ
  father : ℝ
  grandmother : ℝ
  mother : ℝ
  brother : ℝ
  sister : ℝ

/-- Calculates the ages of family members based on given conditions -/
noncomputable def calculateAges : FamilyAges :=
  let mother := (80 : ℝ)
  let grandmother := 3 * mother
  let grace := (3/8) * grandmother
  let father := (7/12) * grandmother
  let brother := (2/5) * grace
  let sister := (3/7) * brother
  { grace := grace
  , father := father
  , grandmother := grandmother
  , mother := mother
  , brother := brother
  , sister := sister }

/-- Theorem stating the correctness of the calculated ages -/
theorem correct_ages :
  let ages := calculateAges
  ages.grace = 90 ∧
  ages.father = 140 ∧
  ages.brother = 36 ∧
  ages.sister = 15.4286 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ages_l1139_113909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_increasing_interval_l1139_113982

open Real Set

-- Define the interval where sin(x) is increasing
def sinIncreasingInterval (k : ℤ) : Set ℝ :=
  Icc (2 * ↑k * π - π / 2) (2 * ↑k * π + π / 2)

-- Define the interval where cos(x) is increasing
def cosIncreasingInterval (k : ℤ) : Set ℝ :=
  Icc (2 * ↑k * π - π) (2 * ↑k * π)

-- Define the interval where both sin(x) and cos(x) are increasing
def bothIncreasingInterval (k : ℤ) : Set ℝ :=
  Icc (2 * ↑k * π - π / 2) (2 * ↑k * π)

-- Theorem stating that the intersection of sin and cos increasing intervals
-- is equal to the interval where both are increasing
theorem both_increasing_interval (k : ℤ) :
  (sinIncreasingInterval k) ∩ (cosIncreasingInterval k) = bothIncreasingInterval k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_increasing_interval_l1139_113982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_problem_l1139_113979

def A : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 2]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 0, 1]
def v : Matrix (Fin 2) (Fin 1) ℝ := !![2; -4]

theorem matrix_problem (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M * A = B) : 
  M ^ 2 = 1 ∧ M ^ 2014 * v = v := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_problem_l1139_113979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_tangent_l1139_113952

/-- The value of n for which the parabola y = x^2 + 5 and the hyperbola y^2 - nx^2 = 4 are tangent -/
noncomputable def tangent_n : ℝ :=
  10 + Real.sqrt 84

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ :=
  x^2 + 5

/-- The hyperbola equation -/
def hyperbola (n x y : ℝ) : Prop :=
  y^2 - n * x^2 = 4

/-- Theorem stating that if the parabola and hyperbola are tangent, then n = 10 + √84 -/
theorem parabola_hyperbola_tangent :
  ∃ x y : ℝ, parabola x = y ∧ hyperbola tangent_n x y ∧
  ∀ x' y' : ℝ, parabola x' = y' → hyperbola tangent_n x' y' → x' = x ∧ y' = y := by
  sorry

#check parabola_hyperbola_tangent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_tangent_l1139_113952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_new_salary_l1139_113993

/-- Calculates the new salary after a percentage increase --/
noncomputable def new_salary (original : ℚ) (percentage_increase : ℚ) : ℚ :=
  original * (1 + percentage_increase / 100)

/-- Proves that John's new salary is $68 after a 13.333333333333334% raise from $60 --/
theorem john_new_salary :
  let original_salary : ℚ := 60
  let percentage_increase : ℚ := 13333333333333334 / 1000000000000000
  ⌊new_salary original_salary percentage_increase⌋ = 68 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_new_salary_l1139_113993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_coverage_l1139_113992

/-- The number of radars -/
def n : ℕ := 7

/-- The radius of each radar's coverage circle in km -/
def r : ℝ := 41

/-- The width of the coverage ring in km -/
def w : ℝ := 18

/-- The maximum distance from the center to each radar -/
noncomputable def max_distance : ℝ := 40 / Real.sin (Real.pi / n)

/-- The area of the coverage ring -/
noncomputable def coverage_area : ℝ := 1440 * Real.pi / Real.tan (Real.pi / n)

/-- Theorem stating the maximum distance and coverage area for the given configuration -/
theorem radar_coverage :
  (∀ (d : ℝ), d ≤ max_distance → ∃ (i : Fin n), d ≤ r ∧ d ≥ r - w) ∧
  (coverage_area = Real.pi * ((max_distance + w / 2)^2 - (max_distance - w / 2)^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_coverage_l1139_113992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_loci_l1139_113999

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- A point M outside the circle -/
def M : ℝ × ℝ := sorry

/-- A moving chord AB within the circle -/
def AB (c : Circle) : Set (ℝ × ℝ) := sorry

/-- The foot of the perpendicular from M to chord AB -/
def H (c : Circle) : Set (ℝ × ℝ) := sorry

/-- The intersection point of tangents drawn at A and B -/
def P (c : Circle) : Set (ℝ × ℝ) := sorry

/-- Predicate stating that AB subtends a right angle at M -/
def subtendsRightAngle (c : Circle) : Prop := sorry

/-- The locus of the midpoint of AB -/
def midpointLocus (c : Circle) : Set (ℝ × ℝ) := sorry

/-- Predicate stating that a set of points forms a circle -/
def isCircle : Set (ℝ × ℝ) → Prop := sorry

/-- Predicate stating that three circles are coaxial with a limiting point -/
def areCoaxial (c1 c2 c3 : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop := sorry

theorem chord_loci (c : Circle) 
  (h : subtendsRightAngle c) : 
  isCircle (midpointLocus c) ∧ 
  isCircle (H c) ∧ 
  isCircle (P c) ∧ 
  areCoaxial (midpointLocus c) (H c) (P c) M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_loci_l1139_113999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_17_over_70_l1139_113949

theorem digit_150_of_17_over_70 (n : ℕ) (h : n = 150) : 
  (((17 : ℚ) / 70 - ((17 : ℚ) / 70).floor) * 10^n).floor % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_17_over_70_l1139_113949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_liar_ratio_l1139_113918

/-- Represents a person, either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- Represents the table arrangement -/
def TableArrangement := List Person

/-- Checks if a person is a knight -/
def isKnight (p : Person) : Bool :=
  match p with
  | Person.Knight => true
  | Person.Liar => false

/-- Gets the neighbors of a person in a circular arrangement -/
def getNeighbors (arrangement : TableArrangement) (index : Nat) : Option (Person × Person) := do
  let n := arrangement.length
  let left ← arrangement.get? ((index - 1 + n) % n)
  let right ← arrangement.get? ((index + 1) % n)
  some (left, right)

/-- Checks if a person's statement is consistent with their type and neighbors -/
def isConsistent (arrangement : TableArrangement) (index : Nat) : Bool := 
  match arrangement.get? index, getNeighbors arrangement index with
  | some person, some (left, right) => 
    let knightCount := if isKnight left then 1 else 0 + if isKnight right then 1 else 0
    (isKnight person ∧ knightCount = 1) ∨ (¬isKnight person ∧ knightCount ≠ 1)
  | _, _ => false

/-- Checks if all statements in the arrangement are consistent -/
def isValidArrangement (arrangement : TableArrangement) : Bool :=
  List.range arrangement.length |>.all (isConsistent arrangement)

/-- Counts the number of knights in the arrangement -/
def countKnights (arrangement : TableArrangement) : Nat :=
  arrangement.filter isKnight |>.length

/-- Counts the number of liars in the arrangement -/
def countLiars (arrangement : TableArrangement) : Nat :=
  arrangement.length - countKnights arrangement

theorem knight_liar_ratio (arrangement : TableArrangement) :
  arrangement.length > 2 ∧ 
  isValidArrangement arrangement ∧ 
  countKnights arrangement > 0 ∧ 
  countLiars arrangement > 0 →
  countKnights arrangement = 2 * countLiars arrangement := by
  sorry

#eval isValidArrangement [Person.Knight, Person.Knight, Person.Liar]
#eval countKnights [Person.Knight, Person.Knight, Person.Liar]
#eval countLiars [Person.Knight, Person.Knight, Person.Liar]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_liar_ratio_l1139_113918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_edges_is_twelve_l1139_113988

/-- Represents a rectangular picture frame -/
structure Frame where
  frame_width : ℝ
  interior_edge : ℝ
  frame_area : ℝ

/-- Calculates the sum of the interior edges of a frame -/
noncomputable def sum_interior_edges (f : Frame) : ℝ :=
  2 * (f.interior_edge + (f.frame_area - 2 * f.frame_width * f.interior_edge - 2 * f.frame_width ^ 2) / 
      (2 * f.frame_width + f.interior_edge))

/-- Theorem: The sum of interior edges for the given frame is 12 inches -/
theorem sum_interior_edges_is_twelve (f : Frame) 
  (h1 : f.frame_width = 1.5)
  (h2 : f.interior_edge = 4.5)
  (h3 : f.frame_area = 27) :
  sum_interior_edges f = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_edges_is_twelve_l1139_113988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_13_14_15_triangle_l1139_113900

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a

/-- The length of the altitude to the longest side of a triangle -/
noncomputable def altitude_to_longest_side (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  let area := Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))
  2 * area / max t.a (max t.b t.c)

/-- Theorem stating that for a triangle with sides 13, 14, and 15, 
    the altitude to the longest side is 168/15 -/
theorem altitude_13_14_15_triangle :
  ∃ t : Triangle, t.a = 13 ∧ t.b = 14 ∧ t.c = 15 ∧ 
  altitude_to_longest_side t = 168 / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_13_14_15_triangle_l1139_113900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_free_fall_impossibility_l1139_113948

/-- Proves the impossibility of the given scenario in free fall. -/
theorem free_fall_impossibility (g : ℝ) (h₀ h₁ h₂ : ℝ) (v₁ v₂ : ℝ) 
  (h₀_pos : h₀ > 0)
  (h₁_pos : h₁ > 0)
  (h₂_nonneg : h₂ ≥ 0)
  (h_order : h₀ > h₁ ∧ h₁ > h₂)
  (g_pos : g > 0)
  (v₁_def : v₁^2 = 2 * g * (h₀ - h₁))
  (v₂_def : v₂ = 2 * v₁)
  (h₀_val : h₀ = 10)
  (h₁_val : h₁ = 5)
  (h₂_val : h₂ = 0) :
  False := by
  sorry

#check free_fall_impossibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_free_fall_impossibility_l1139_113948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_right_triangle_properties_l1139_113991

/-- A non-right triangle -/
structure NonRightTriangle where
  angles : Fin 3 → ℝ
  sum_angles : (angles 0) + (angles 1) + (angles 2) = Real.pi
  all_positive : ∀ i, 0 < angles i
  non_right : ∀ i, angles i ≠ Real.pi/2

/-- The sequence of triangles formed by the feet of altitudes -/
noncomputable def altitude_triangle_sequence (T : NonRightTriangle) : ℕ → NonRightTriangle
  | 0 => T
  | n + 1 => sorry  -- Definition of the next triangle in the sequence

/-- T1 is acute-angled -/
def is_T1_acute (T : NonRightTriangle) : Prop :=
  (∀ i, Real.pi/4 < T.angles i ∧ T.angles i < Real.pi/2) ∨
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧
    T.angles i < Real.pi/4 ∧ T.angles j < Real.pi/4 ∧ Real.pi/2 < T.angles k ∧ T.angles k < 3*Real.pi/4)

/-- A right triangle appears in the sequence -/
def exists_right_triangle_in_sequence (T : NonRightTriangle) : Prop :=
  ∃ n s : ℕ, ∃ i : Fin 3, T.angles i = (Real.pi * s) / (2^n)

/-- T3 is similar to T -/
def is_T3_similar_to_T (T : NonRightTriangle) : Prop :=
  (∀ i, Real.pi/4 < T.angles i ∧ T.angles i < Real.pi/2) ∨
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧
    T.angles i < Real.pi/4 ∧ T.angles j < Real.pi/4 ∧ Real.pi/2 < T.angles k ∧ T.angles k < 3*Real.pi/4)

/-- Number of non-similar triangles T for which Tn is similar to T -/
def count_similar_triangles (n : ℕ) : ℕ := 2^(2*n) - 2^n

theorem non_right_triangle_properties (T : NonRightTriangle) :
  (is_T1_acute T) ∧
  (exists_right_triangle_in_sequence T) ∧
  (is_T3_similar_to_T T) ∧
  (∀ n : ℕ, ∃ m : ℕ, m = count_similar_triangles n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_right_triangle_properties_l1139_113991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_district_a_private_schools_l1139_113950

/-- Represents the number of schools in Veenapaniville --/
structure SchoolCounts where
  total : Nat
  public : Nat
  parochial : Nat
  privateIndependent : Nat
  districtA : Nat
  districtB : Nat
  districtBPrivateIndependent : Nat

/-- The given conditions for the school counts in Veenapaniville --/
def veenapaniville : SchoolCounts := {
  total := 50,
  public := 25,
  parochial := 16,
  privateIndependent := 9,
  districtA := 18,
  districtB := 17,
  districtBPrivateIndependent := 2
}

/-- Theorem stating that District A has 2 private independent schools --/
theorem district_a_private_schools (v : SchoolCounts) (h : v = veenapaniville) :
  v.privateIndependent - (v.districtBPrivateIndependent + (v.total - v.districtA - v.districtB) / 3) = 2 := by
  sorry

#check district_a_private_schools

end NUMINAMATH_CALUDE_ERRORFEEDBACK_district_a_private_schools_l1139_113950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_second_quadrant_l1139_113910

theorem cosine_second_quadrant (α : Real) (h1 : Real.sin α = 5/13) (h2 : π/2 < α ∧ α < π) :
  Real.cos α = -12/13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_second_quadrant_l1139_113910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_Q_plus_S_l1139_113963

def R (k : ℕ) : ℚ := (10^k - 1) / 9

def Q : ℚ := R 32 / R 8

def S : ℕ := 100000000

noncomputable def count_zeros (n : ℚ) : ℕ := sorry

theorem zeros_in_Q_plus_S : count_zeros (Q + S) = 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_Q_plus_S_l1139_113963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_value_decrease_l1139_113937

theorem card_value_decrease (initial_value : ℝ) (h : initial_value > 0) : 
  (initial_value * (1 - 0.2) * (1 - 0.3) - initial_value) / initial_value = -0.44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_value_decrease_l1139_113937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_welders_count_l1139_113927

/-- The number of welders initially working on the order -/
def initial_welders : ℕ := 10

/-- The number of days needed to complete the order initially -/
def initial_days : ℕ := 8

/-- The number of welders who leave after the first day -/
def leaving_welders : ℕ := 9

/-- The number of additional days needed by the remaining welders -/
def additional_days : ℕ := 28

/-- The fraction of work completed on the first day -/
def first_day_work : ℚ := 1 / initial_days

/-- The fraction of work remaining after the first day -/
def remaining_work : ℚ := 1 - first_day_work

theorem welders_count :
  (initial_welders : ℚ) * (1 : ℚ) = 
  (initial_welders - leaving_welders : ℚ) * (additional_days : ℚ) ∧
  initial_welders > leaving_welders := by
  sorry

#check welders_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_welders_count_l1139_113927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_value_l1139_113923

-- Define the integer and decimal parts of 2+√2 and 4-√2
noncomputable def a : ℤ := ⌊2 + Real.sqrt 2⌋
noncomputable def b : ℝ := 2 + Real.sqrt 2 - a
noncomputable def c : ℤ := ⌊4 - Real.sqrt 2⌋
noncomputable def d : ℝ := 4 - Real.sqrt 2 - c

-- State the theorem
theorem fraction_value : (b + d) / (a * c) = 1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_value_l1139_113923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1139_113905

noncomputable def f (A ω φ x : ℝ) := A * Real.sin (ω * x + φ)

theorem function_properties 
  (A ω φ : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : 0 < φ ∧ φ < π) 
  (h4 : f A ω φ (π/12) = 4) 
  (h5 : f A ω φ (5*π/12) = -4) :
  (∃ (s : Set ℝ), s = {x | x ∈ Set.union (Set.Icc 0 (π/12)) (Set.Icc (5*π/12) (3*π/4))} ∧ 
    ∀ (x y : ℝ), x ∈ s → y ∈ s → x < y → f A ω φ x < f A ω φ y) ∧
  (∀ (α : ℝ), α ∈ Set.Ioo 0 π → f A ω φ (2*α/3 + π/12) = 2 → 
    α = π/6 ∨ α = 5*π/6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1139_113905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_conditions_l1139_113933

theorem square_conditions (n : ℕ+) :
  (∃ k m : ℤ, (2 * n + 1 : ℤ) = k^2 ∧ (3 * n + 1 : ℤ) = m^2) ↔
  (∃ a b : ℤ, (n + 1 : ℤ) = a^2 + (a + 1)^2 ∧ (n + 1 : ℤ) = b^2 + 2*(b + 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_conditions_l1139_113933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₂_in_fourth_quadrant_z₂_equals_z₁_times_ni_l1139_113970

-- Define complex numbers z₁ and z₂
def z₁ (m : ℝ) : ℂ := m + Complex.I
def z₂ (m : ℝ) : ℂ := m + (m - 2) * Complex.I

-- Theorem 1: z₂ lies in the fourth quadrant iff 0 < m < 2
theorem z₂_in_fourth_quadrant (m : ℝ) :
  (0 < m ∧ m < 2) ↔ (Complex.re (z₂ m) > 0 ∧ Complex.im (z₂ m) < 0) :=
sorry

-- Theorem 2: z₂ = z₁ * ni iff (m = 1 and n = -1) or (m = -2 and n = 2)
theorem z₂_equals_z₁_times_ni (m n : ℝ) :
  z₂ m = z₁ m * (n * Complex.I) ↔ (m = 1 ∧ n = -1) ∨ (m = -2 ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₂_in_fourth_quadrant_z₂_equals_z₁_times_ni_l1139_113970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_and_adjacent_line_l1139_113914

open Real

/-- The logarithmic function shifted by 1 -/
noncomputable def f (x : ℝ) : ℝ := log (x + 1)

/-- The fractional function -/
noncomputable def g (x : ℝ) : ℝ := x / (x + 1)

/-- The tangent line to f at x = 0 -/
def tangent_f (x : ℝ) : ℝ := x

/-- The tangent line to g at x = 0 -/
def tangent_g (x : ℝ) : ℝ := x

theorem common_tangent_and_adjacent_line :
  (∀ x, f x ≤ tangent_f x) ∧
  (∀ x, g x ≤ tangent_g x) ∧
  (∀ x, tangent_f x = tangent_g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_and_adjacent_line_l1139_113914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1139_113938

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 3 = 0

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 2*x

-- Define point P
def point_P : ℝ × ℝ := (1, -2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ curve_C A.1 A.2 ∧
    line_l B.1 B.2 ∧ curve_C B.1 B.2 ∧
    A ≠ B ∧
    distance point_P A + distance point_P B = 6 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1139_113938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_equals_pi_over_four_l1139_113964

theorem angle_sum_equals_pi_over_four (θ φ : Real) :
  0 < θ ∧ θ < π / 2 →
  0 < φ ∧ φ < π / 2 →
  Real.tan θ = 1 / 3 →
  Real.sin φ = 1 / 3 →
  θ + 2 * φ = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_equals_pi_over_four_l1139_113964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l1139_113983

theorem coefficient_x_squared_in_expansion : 
  Finset.sum (Finset.range 6) (fun k => 
    (Nat.choose 5 k) * (2^(5-k)) * ((-3)^k : ℤ) * 
    if k = 3 then 1 else 0) = -1080 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l1139_113983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_candidates_with_disjoint_solutions_l1139_113921

/-- Represents a candidate in the math test -/
structure Candidate where
  id : Nat

/-- Represents a problem in the math test -/
structure Problem where
  id : Nat

/-- Represents the math test setup -/
structure MathTest where
  candidates : Finset Candidate
  problems : Finset Problem
  solved : Problem → Finset Candidate

/-- The theorem statement -/
theorem two_candidates_with_disjoint_solutions (test : MathTest)
  (h_candidate_count : test.candidates.card = 200)
  (h_problem_count : test.problems.card = 6)
  (h_max_solvers : ∀ p ∈ test.problems, (test.solved p).card < 80) :
  ∃ c1 c2, c1 ∈ test.candidates ∧ c2 ∈ test.candidates ∧ c1 ≠ c2 ∧
    ∀ p ∈ test.problems, ¬(c1 ∈ test.solved p ∧ c2 ∈ test.solved p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_candidates_with_disjoint_solutions_l1139_113921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mitzi_japan_spending_l1139_113945

/-- Calculates the remaining money and its dollar equivalent after Mitzi's purchases in Japan --/
theorem mitzi_japan_spending (initial_yen : ℕ) (ticket_cost : ℕ) (food_cost : ℕ) (tshirt_cost : ℕ)
  (souvenir_cost : ℕ) (discount_percent : ℚ) (exchange_rate : ℚ) :
  initial_yen = 10000 ∧ 
  ticket_cost = 3000 ∧ 
  food_cost = 2500 ∧ 
  tshirt_cost = 1500 ∧ 
  souvenir_cost = 2200 ∧ 
  discount_percent = 20 / 100 ∧
  exchange_rate = 110 / 1 →
  let discounted_souvenir := souvenir_cost - (souvenir_cost * discount_percent).floor
  let total_spent := ticket_cost + food_cost + tshirt_cost + discounted_souvenir
  let remaining_yen := initial_yen - total_spent
  let remaining_dollars := (remaining_yen : ℚ) / exchange_rate
  remaining_yen = 1240 ∧ 
  (remaining_dollars * 100).floor / 100 = 1127 / 100 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mitzi_japan_spending_l1139_113945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_in_year_l1139_113967

/-- The probability of rolling a number other than 1 on a fair six-sided die -/
def p : ℚ := 5/6

/-- The number of days in a non-leap year -/
def days : ℕ := 365

/-- The expected number of rolls on a single day -/
noncomputable def expected_rolls_per_day : ℚ := 1 / p

/-- The expected total number of rolls in a non-leap year -/
noncomputable def expected_total_rolls : ℚ := expected_rolls_per_day * days

/-- Theorem: The expected number of die rolls in a non-leap year is 438 -/
theorem expected_rolls_in_year : 
  ⌊expected_total_rolls⌋ = 438 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_in_year_l1139_113967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1139_113972

theorem calculation_proof :
  let expr1 := (2.25 ^ (1/2)) - (9.6 ^ 0) - ((-3.375) ^ (-2/3)) + ((1.5) ^ (-2))
  let expr2 := (Real.log 25 / Real.log 2) * (Real.log (2 * Real.sqrt 2) / Real.log 3) * (Real.log 9 / Real.log 5)
  expr1 = 1/2 ∧ expr2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1139_113972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_12_l1139_113940

noncomputable def f (x : ℝ) : ℝ := (Real.cos (Real.pi / 4 + x))^2 - (Real.cos (Real.pi / 4 - x))^2

theorem f_pi_12 : f (Real.pi / 12) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_12_l1139_113940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_permutation_sum_l1139_113984

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∀ d, d ∈ Nat.digits 10 n → d ≠ 0

noncomputable def sum_of_permutations (n : ℕ) : ℕ :=
  sorry -- Implementation of sum of permutations

theorem three_digit_permutation_sum (n : ℕ) :
  is_valid_number n →
  sum_of_permutations n = 444 →
  n = 112 ∨ n = 121 ∨ n = 211 ∨ n = 444 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_permutation_sum_l1139_113984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_m_eq_one_l1139_113968

/-- The function f(x) defined on (1, +∞) with parameter m > 0 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m / (x - 1)

/-- The theorem stating that if f has a minimum value of 3 on (1, +∞), then m = 1 -/
theorem min_value_implies_m_eq_one (m : ℝ) (h_m : m > 0) :
  (∀ x > 1, f m x ≥ 3) ∧ (∃ x > 1, f m x = 3) → m = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_m_eq_one_l1139_113968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_units_l1139_113955

-- Define the fixed cost
def fixed_cost : ℝ := 20000

-- Define the variable cost per unit
def variable_cost_per_unit : ℝ := 100

-- Define the revenue function
noncomputable def revenue (x : ℝ) : ℝ :=
  if x ≤ 390 then -x^3 / 900 + 400 * x else 90090

-- Define the total cost function
noncomputable def total_cost (x : ℝ) : ℝ := fixed_cost + variable_cost_per_unit * x

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ := revenue x - total_cost x

-- Theorem statement
theorem max_profit_units : 
  ∃ (x : ℝ), x = 300 ∧ 
  (∀ (y : ℝ), y ≥ 0 → profit y ≤ profit x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_units_l1139_113955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l1139_113924

theorem tan_difference (α β : Real) (h1 : Real.tan α = 9) (h2 : Real.tan β = 6) :
  Real.tan (α - β) = 3 / 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l1139_113924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninas_house_height_l1139_113939

/-- The height of Nina's house, given shadow lengths and a reference object --/
noncomputable def house_height (house_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ) : ℝ :=
  (house_shadow * tree_height) / tree_shadow

/-- Rounds a real number to the nearest integer --/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

/-- Theorem: Nina's house height is approximately 56 feet --/
theorem ninas_house_height :
  round_to_nearest (house_height 75 15 20) = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninas_house_height_l1139_113939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_sum_divisible_l1139_113976

theorem odd_power_sum_divisible (k : ℕ) (x y : ℤ) :
  ∃ q : ℤ, x^(2*k + 1) + y^(2*k + 1) = (x + y) * q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_sum_divisible_l1139_113976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_21_l1139_113997

def my_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => my_sequence n + (n + 2)

theorem sixth_term_is_21 : my_sequence 5 = 21 := by
  rw [my_sequence]
  simp
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_21_l1139_113997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_tax_rate_is_ten_percent_l1139_113975

/-- Represents the tax rates and spending percentages for Jill's shopping trip --/
structure ShoppingTrip where
  clothing_spend : ℚ
  food_spend : ℚ
  other_spend : ℚ
  clothing_tax : ℚ
  food_tax : ℚ
  total_tax : ℚ

/-- Calculates the tax rate on other items given the shopping trip details --/
def calculate_other_tax_rate (trip : ShoppingTrip) : ℚ :=
  ((trip.total_tax - (trip.clothing_spend * trip.clothing_tax)) / trip.other_spend) * 100

/-- Theorem stating that the tax rate on other items is 10% --/
theorem other_tax_rate_is_ten_percent (trip : ShoppingTrip) :
  trip.clothing_spend = 45 →
  trip.food_spend = 45 →
  trip.other_spend = 10 →
  trip.clothing_tax = 5 / 100 →
  trip.food_tax = 0 →
  trip.total_tax = 325 / 100 →
  calculate_other_tax_rate trip = 10 := by
  sorry

#eval calculate_other_tax_rate {
  clothing_spend := 45,
  food_spend := 45,
  other_spend := 10,
  clothing_tax := 5 / 100,
  food_tax := 0,
  total_tax := 325 / 100
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_tax_rate_is_ten_percent_l1139_113975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_slope_l1139_113903

-- Define the curves C₁ and C₂
def C₁ (x : ℝ) : ℝ := x^2
def C₂ (x : ℝ) : ℝ := x^3

-- Define the derivative of C₁ and C₂
def C₁' (x : ℝ) : ℝ := 2*x
def C₂' (x : ℝ) : ℝ := 3*x^2

-- Define what it means for a line to be tangent to a curve at a point
def is_tangent_to (m : ℝ) (c : ℝ → ℝ) (c' : ℝ → ℝ) (x : ℝ) : Prop :=
  c x = m * x + c 0 ∧ c' x = m

-- Theorem statement
theorem common_tangent_slope :
  ∀ (m : ℝ) (x₁ x₂ : ℝ),
    is_tangent_to m C₁ C₁' x₁ ∧ is_tangent_to m C₂ C₂' x₂ →
    m = 0 ∨ m = 64/27 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_slope_l1139_113903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1139_113994

/-- The eccentricity of a hyperbola with the given properties is √5 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let C : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let F₁ : ℝ × ℝ := (-Real.sqrt (a^2 + b^2), 0)
  let F₂ : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)
  let O : ℝ × ℝ := (0, 0)
  let asymptote := λ x => -b * x / a
  ∃ (D : ℝ × ℝ), 
    (D.1 - F₁.1) * (asymptote D.1 - D.2) = 0 ∧  -- D is on the perpendicular line from F₁ to the asymptote
    (D.1 - F₂.1)^2 + (D.2 - F₂.2)^2 = 8 * ((D.1 - O.1)^2 + (D.2 - O.2)^2) →  -- |DF₂| = 2√2|OD|
  (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1139_113994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angles_l1139_113934

theorem quadrilateral_angles (A B C D : ℝ) : 
  A + B + C + D = 360 →  -- Sum of angles in a quadrilateral
  (A : ℝ) / 1 = B / 3 ∧ B / 3 = C / 5 ∧ C / 5 = D / 6 →  -- Given ratio of angles
  A = 24 ∧ D = 144 :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angles_l1139_113934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_products_l1139_113922

def is_valid_product (p : ℕ) : Prop :=
  ∃ (x y z : ℕ),
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    p = (100 * x + 10 * y + z) * (100 * y + 10 * x + z) ∧
    ∃ (o : ℕ), o ≠ x ∧ o ≠ y ∧ o ≠ z ∧
    (p / 10000 % 10) ∈ ({x, y, z, o} : Set ℕ) ∧
    (p / 1000 % 10) ∈ ({x, y, z, o} : Set ℕ) ∧
    (p / 100 % 10) ∈ ({x, y, z, o} : Set ℕ) ∧
    (p / 10 % 10) ∈ ({x, y, z, o} : Set ℕ) ∧
    p / 100000 = p % 10 ∧
    p / 100000 ∉ ({x, y, z, o} : Set ℕ)

theorem valid_products :
  ∀ p : ℕ, is_valid_product p ↔ p = 169201 ∨ p = 193501 := by
  sorry

#check valid_products

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_products_l1139_113922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_height_calculation_l1139_113995

/-- Given a lamp post height, its shadow length, and a dog's shadow length,
    calculate the height of the dog using the principle of similar triangles. -/
noncomputable def dog_height (lamp_height : ℝ) (lamp_shadow : ℝ) (dog_shadow : ℝ) : ℝ :=
  (lamp_height / lamp_shadow) * dog_shadow

/-- Theorem stating that a dog casting a 6-inch shadow next to a 50-foot lamp post
    with an 8-foot shadow is 37.5 inches tall. -/
theorem dog_height_calculation :
  let lamp_height : ℝ := 50 * 12  -- Convert 50 feet to inches
  let lamp_shadow : ℝ := 8 * 12   -- Convert 8 feet to inches
  let dog_shadow : ℝ := 6         -- Dog's shadow in inches
  dog_height lamp_height lamp_shadow dog_shadow = 37.5 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_height_calculation_l1139_113995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leader_boy_probability_l1139_113966

/-- Represents a student in the group -/
inductive Student
| Girl
| Boy
deriving DecidableEq

/-- The group of students -/
def group : Finset Student := sorry

/-- The number of students in the group -/
def groupSize : ℕ := 5

/-- The number of girls in the group -/
def girlCount : ℕ := 3

/-- The number of boys in the group -/
def boyCount : ℕ := 2

/-- A selection of two students from the group -/
def Selection := (Student × Student)

/-- The set of all possible selections -/
def allSelections : Finset Selection := sorry

/-- Predicate for a selection where the leader (first student) is a boy -/
def leaderIsBoy (s : Selection) : Prop := s.1 = Student.Boy

/-- The probability that the leader is a boy -/
noncomputable def probLeaderIsBoy : ℚ := sorry

theorem leader_boy_probability : 
  group.card = groupSize ∧ 
  (group.filter (· = Student.Girl)).card = girlCount ∧
  (group.filter (· = Student.Boy)).card = boyCount ∧
  allSelections = group.product group →
  probLeaderIsBoy = 2 / 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leader_boy_probability_l1139_113966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_and_g_range_l1139_113981

/-- The function f(x) satisfying the given conditions -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

/-- The function g(x) defined in terms of f -/
noncomputable def g (x : ℝ) : ℝ := (6 * Real.cos x ^ 4 - Real.sin x ^ 2 - 1) / (f (x / 2 + Real.pi / 6) ^ 2 - 2)

/-- Theorem stating that f satisfies the given conditions and g has the specified range -/
theorem f_satisfies_conditions_and_g_range : 
  (∀ x, f x ≤ 2) ∧ 
  (f (Real.pi / 6) = 2) ∧
  (∀ x y, x < y ∧ f x = 0 ∧ f y = 0 ∧ (∀ z, x < z ∧ z < y → f z ≠ 0) → y - x = Real.pi / 2) ∧
  (Set.range g = Set.Icc 1 (7 / 4) ∪ Set.Ioo (7 / 4) (5 / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_and_g_range_l1139_113981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_print_rolls_sold_l1139_113990

/-- Proves that the number of print rolls sold is 210 given the conditions of the problem -/
theorem print_rolls_sold :
  ∃ (solid_rolls print_rolls : ℕ),
    solid_rolls + print_rolls = 480 ∧
    4 * solid_rolls + 6 * print_rolls = 2340 ∧
    print_rolls = 210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_print_rolls_sold_l1139_113990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_quadrilaterals_with_equidistant_point_l1139_113936

/-- A quadrilateral type -/
inductive QuadrilateralType
| Square
| Rectangle
| KiteWithTwoAdjacentCongruentAngles
| AnyCyclicQuadrilateral
| CyclicTrapezoid

/-- Function indicating if a quadrilateral type has a point equidistant from all vertices when cyclic -/
def has_equidistant_point (q : QuadrilateralType) : Bool :=
  match q with
  | QuadrilateralType.Square => true
  | QuadrilateralType.Rectangle => true
  | QuadrilateralType.KiteWithTwoAdjacentCongruentAngles => false
  | QuadrilateralType.AnyCyclicQuadrilateral => true
  | QuadrilateralType.CyclicTrapezoid => true

/-- The list of all quadrilateral types -/
def quadrilateral_types : List QuadrilateralType :=
  [QuadrilateralType.Square, QuadrilateralType.Rectangle, 
   QuadrilateralType.KiteWithTwoAdjacentCongruentAngles,
   QuadrilateralType.AnyCyclicQuadrilateral, QuadrilateralType.CyclicTrapezoid]

theorem count_quadrilaterals_with_equidistant_point :
  (quadrilateral_types.filter has_equidistant_point).length = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_quadrilaterals_with_equidistant_point_l1139_113936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1139_113998

/-- Given an ellipse and a hyperbola that share the same foci, 
    prove that the length of the major axis of the ellipse is 8. -/
theorem ellipse_major_axis_length 
  (k : ℝ) 
  (ellipse : (ℝ × ℝ) → Prop)
  (hyperbola : (ℝ × ℝ) → Prop)
  (ellipse_eq : ∀ (x y : ℝ), ellipse (x, y) ↔ x^2 / k + y^2 / 9 = 1)
  (hyperbola_eq : ∀ (x y : ℝ), hyperbola (x, y) ↔ x^2 / 4 - y^2 / 3 = 1)
  (shared_foci : ∃ (c : ℝ), 
    (∀ (x y : ℝ), ellipse (x, y) → x^2 + y^2 ≤ c^2) ∧ 
    (∀ (x y : ℝ), hyperbola (x, y) → x^2 - y^2 = c^2)) :
  2 * Real.sqrt k = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1139_113998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_odd_prime_factor_of_2023_4_plus_1_l1139_113951

theorem least_odd_prime_factor_of_2023_4_plus_1 : 
  ∃ (p : ℕ), Nat.Prime p ∧ Odd p ∧ p ∣ (2023^4 + 1) ∧ 
  ∀ (q : ℕ), Nat.Prime q → Odd q → q ∣ (2023^4 + 1) → p ≤ q ∧ p = 17 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_odd_prime_factor_of_2023_4_plus_1_l1139_113951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_expression_l1139_113919

/-- Given a triangle ABC with sides AB = 7, AC = 8, and BC = 5, 
    the expression (cos((A - B)/2) / sin(C/2)) - (sin((A - B)/2) / cos(C/2)) equals 16/7 -/
theorem triangle_trig_expression (A B C : ℝ) : 
  let a : ℝ := 7
  let b : ℝ := 8
  let c : ℝ := 5
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧  -- positive side lengths
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧  -- triangle inequality
  (A + B + C = π) ∧  -- sum of angles in a triangle
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) →  -- law of sines
  (Real.cos ((A - B)/2) / Real.sin (C/2)) - (Real.sin ((A - B)/2) / Real.cos (C/2)) = 16/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_expression_l1139_113919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_and_beta_values_l1139_113971

-- Define the vectors a and b as functions of x
noncomputable def a (x : Real) : Real × Real := (2 * Real.sin x, Real.sin x + Real.cos x)
noncomputable def b (x : Real) : Real × Real := (Real.cos x, Real.sqrt 3 * (Real.sin x - Real.cos x))

-- Define the function f as the dot product of a and b
noncomputable def f (x : Real) : Real := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- State the theorem
theorem alpha_and_beta_values (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi/2)
  (h2 : f (α/2) = -1)
  (h3 : 0 < β ∧ β < Real.pi/2)
  (h4 : Real.cos (α + β) = -1/3) :
  α = Real.pi/6 ∧ Real.sin β = (2 * Real.sqrt 6 + 1) / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_and_beta_values_l1139_113971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_limited_intersection_l1139_113931

/-- The set A containing elements from 0 to 9 -/
def A : Finset ℕ := Finset.range 10

/-- A family of non-empty subsets of A -/
def FamilyOfSubsets (k : ℕ) := Fin k → {B : Finset ℕ // B ⊆ A ∧ B.Nonempty}

/-- The condition that any two distinct subsets in the family have at most two elements in their intersection -/
def ValidFamily (k : ℕ) (F : FamilyOfSubsets k) :=
  ∀ i j, i ≠ j → ((F i).1 ∩ (F j).1).card ≤ 2

/-- The theorem stating the maximum number of subsets satisfying the condition -/
theorem max_subsets_with_limited_intersection :
  (∃ k : ℕ, ∃ F : FamilyOfSubsets k, ValidFamily k F) ∧
  (∀ m : ℕ, ∀ G : FamilyOfSubsets m, ValidFamily m G → m ≤ 175) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_limited_intersection_l1139_113931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_n_for_solution_l1139_113962

/-- Represents the system of equations for n variables -/
def system_of_equations (n : ℕ) (x : ℕ → ℤ) : Prop :=
  (2 * x 1 - x 2 = 1) ∧
  (∀ i, 2 ≤ i ∧ i ≤ n - 1 → -x (i-1) + 2 * x i - x (i+1) = 1) ∧
  (-x (n-1) + 2 * x n = 1)

/-- The main theorem stating that if there's a solution to the system, n must be even -/
theorem even_n_for_solution (n : ℕ) (x : ℕ → ℤ) :
  n > 1 → system_of_equations n x → Even n := by
  sorry

/-- Helper lemma: x_i can be expressed in terms of x_1 -/
lemma x_i_in_terms_of_x_1 (n : ℕ) (x : ℕ → ℤ) (i : ℕ) :
  system_of_equations n x → i ≤ n → x i = i * x 1 - i * (i - 1) / 2 := by
  sorry

/-- Helper lemma: x_1 must be an integer solution of a specific equation -/
lemma x_1_equation (n : ℕ) (x : ℕ → ℤ) :
  system_of_equations n x → ∃ k : ℤ, 2 * (n + 1) * k = 3 * n^2 - 5 * n + 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_n_for_solution_l1139_113962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_f_is_four_l1139_113929

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 - Real.log x / Real.log 2

-- State the theorem
theorem root_of_f_is_four :
  ∀ a : ℝ, a > 0 → f a = 0 → a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_f_is_four_l1139_113929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_in_still_water_l1139_113986

/-- The speed of a man rowing in still water, given his downstream performance and current speed. -/
theorem mans_speed_in_still_water 
  (current_speed : ℝ)
  (distance_downstream : ℝ)
  (time_downstream : ℝ)
  (h1 : current_speed = 5)
  (h2 : distance_downstream = 60 / 1000)  -- Convert 60 meters to kilometers
  (h3 : time_downstream = 10.799136069114471 / 3600)  -- Convert seconds to hours
  : ∃ (speed_still_water : ℝ), 
    (speed_still_water + current_speed) * time_downstream = distance_downstream ∧
    abs (speed_still_water - 15.0008) < 0.0001 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_in_still_water_l1139_113986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_selection_theorem_l1139_113943

/-- A vector in a 2D plane -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The length of a 2D vector -/
noncomputable def Vector2D.length (v : Vector2D) : ℝ := Real.sqrt (v.x^2 + v.y^2)

/-- The sum of a list of 2D vectors -/
def vectorSum (vs : List Vector2D) : Vector2D :=
  { x := vs.map (·.x) |>.sum,
    y := vs.map (·.y) |>.sum }

theorem vector_selection_theorem (vectors : List Vector2D) 
  (h : (vectors.map Vector2D.length).sum = 4) :
  ∃ (subset : List Vector2D), subset ⊆ vectors ∧ 
    (vectorSum subset).length > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_selection_theorem_l1139_113943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangle_perimeters_l1139_113932

/-- The side length of the first equilateral triangle in centimeters -/
noncomputable def initial_side_length : ℝ := 60

/-- The ratio of side lengths between consecutive triangles -/
noncomputable def side_ratio : ℝ := 1 / 2

/-- The sum of the geometric series representing the ratio of perimeters -/
noncomputable def perimeter_ratio_sum : ℝ := 2

/-- The number of sides in an equilateral triangle -/
def triangle_sides : ℕ := 3

/-- 
Theorem: The sum of the perimeters of an infinite series of equilateral triangles, 
where each triangle is formed by joining the midpoints of the previous triangle's sides, 
and the first triangle has sides of length 60 cm, is equal to 360 cm.
-/
theorem sum_of_triangle_perimeters : 
  (triangle_sides : ℝ) * initial_side_length * perimeter_ratio_sum = 360 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangle_perimeters_l1139_113932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l1139_113904

/-- The function f(x) parameterized by m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 1/x - m/x^2 - x/3

/-- Theorem stating the condition for f to have two zeros -/
theorem f_has_two_zeros (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0) ↔ 0 < m ∧ m < 2/3 := by
  sorry

#check f_has_two_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l1139_113904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_increase_is_ten_percent_l1139_113987

/-- Represents the financial situation of Paulson -/
structure FinancialSituation where
  income : ℝ
  expenditure_percentage : ℝ
  income_increase_percentage : ℝ
  savings_increase_percentage : ℝ

/-- Calculates the percentage increase in expenditure -/
noncomputable def calculate_expenditure_increase (fs : FinancialSituation) : ℝ :=
  let original_expenditure := fs.income * fs.expenditure_percentage
  let original_savings := fs.income - original_expenditure
  let new_income := fs.income * (1 + fs.income_increase_percentage)
  let new_savings := original_savings * (1 + fs.savings_increase_percentage)
  let new_expenditure := new_income - new_savings
  (new_expenditure / original_expenditure - 1) * 100

/-- Theorem stating that given the conditions, the expenditure increase is 10% -/
theorem expenditure_increase_is_ten_percent (fs : FinancialSituation)
  (h1 : fs.expenditure_percentage = 0.75)
  (h2 : fs.income_increase_percentage = 0.20)
  (h3 : fs.savings_increase_percentage = 0.4999999999999996)
  : calculate_expenditure_increase fs = 10 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_increase_is_ten_percent_l1139_113987
