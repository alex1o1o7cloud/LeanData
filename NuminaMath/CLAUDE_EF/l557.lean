import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l557_55754

/-- A rational function with two specified vertical asymptotes -/
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ := (x - 3) / (x^2 + c*x + d)

/-- The sum of coefficients c and d is -3 given the vertical asymptotes -/
theorem asymptote_sum (c d : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → g c d x ≠ 0) ∧ 
  (∀ x : ℝ, x ≠ -3 → g c d x ≠ 0) ∧ 
  (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |g c d x| > 1/ε) ∧
  (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x + 3| ∧ |x + 3| < δ → |g c d x| > 1/ε) →
  c + d = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l557_55754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grazing_months_l557_55703

/-- Represents the number of months A put his oxen for grazing -/
def x : ℝ := sorry

/-- The total rent of the pasture -/
def total_rent : ℝ := 175

/-- C's share of the rent -/
def c_share : ℝ := 44.99999999999999

/-- The number of oxen A put for grazing -/
def a_oxen : ℕ := 10

/-- The number of oxen B put for grazing -/
def b_oxen : ℕ := 12

/-- The number of months B put his oxen for grazing -/
def b_months : ℕ := 5

/-- The number of oxen C put for grazing -/
def c_oxen : ℕ := 15

/-- The number of months C put his oxen for grazing -/
def c_months : ℕ := 3

theorem grazing_months :
  (a_oxen : ℝ) * x + (b_oxen : ℝ) * b_months + (c_oxen : ℝ) * c_months = total_rent ∧
  (c_oxen : ℝ) * c_months = c_share →
  x = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grazing_months_l557_55703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l557_55745

noncomputable section

variable (e₁ e₂ : ℝ × ℝ)

-- Define e₁ and e₂ as mutually perpendicular unit vectors
axiom e₁_unit : ‖e₁‖ = 1
axiom e₂_unit : ‖e₂‖ = 1
axiom e₁_perp_e₂ : e₁ • e₂ = 0

-- Define the vectors v₁ and v₂
def v₁ (e₁ e₂ : ℝ × ℝ) : ℝ × ℝ := (Real.sqrt 3 * e₁.1 - e₂.1, Real.sqrt 3 * e₁.2 - e₂.2)
def v₂ (e₁ e₂ : ℝ × ℝ) : ℝ × ℝ := (Real.sqrt 3 * e₁.1 + e₂.1, Real.sqrt 3 * e₁.2 + e₂.2)

-- State the theorem
theorem angle_between_vectors (e₁ e₂ : ℝ × ℝ) : 
  let θ := Real.arccos ((v₁ e₁ e₂).1 * (v₂ e₁ e₂).1 + (v₁ e₁ e₂).2 * (v₂ e₁ e₂).2) / 
    (Real.sqrt ((v₁ e₁ e₂).1^2 + (v₁ e₁ e₂).2^2) * Real.sqrt ((v₂ e₁ e₂).1^2 + (v₂ e₁ e₂).2^2))
  θ = π / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l557_55745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_set_is_parabola_l557_55712

noncomputable def vertex_of_parabola (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  (x, a * x^2 + b * x + c)

theorem vertex_set_is_parabola (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  {p : ℝ × ℝ | ∃ b : ℝ, p = vertex_of_parabola a b c} =
  {p : ℝ × ℝ | p.2 = -a * p.1^2 + c} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_set_is_parabola_l557_55712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_and_symmetric_point_l557_55778

/-- The projection of a point onto the xOy plane -/
def projection_xOy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2.1, 0)

/-- The symmetric point with respect to the xOy plane -/
def symmetric_xOy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2.1, -p.2.2)

theorem projection_and_symmetric_point :
  let P : ℝ × ℝ × ℝ := (2, 3, 4)
  (projection_xOy P = (2, 3, 0)) ∧
  (symmetric_xOy P = (2, 3, -4)) := by
  sorry

#eval projection_xOy (2, 3, 4)
#eval symmetric_xOy (2, 3, 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_and_symmetric_point_l557_55778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_formula_l557_55727

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => λ x => f (f_n n x)

theorem f_n_formula (n : ℕ) (x : ℝ) (h1 : n ≥ 2) (h2 : x > 0) :
  f_n n x = x / (n * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_formula_l557_55727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l557_55765

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧
    T = Real.pi ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2),
      f x ≤ 2 ∧
      f x ≥ -1 ∧
      f (Real.pi / 6) = 2 ∧
      f (Real.pi / 2) = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l557_55765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l557_55729

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_sum_magnitude (a b : V) 
  (ha : ‖a‖ = 2)
  (hb : ‖b‖ = Real.sqrt 2)
  (hangle : Real.arccos (inner a b / (‖a‖ * ‖b‖)) = π / 4) :
  ‖a + b‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l557_55729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_satisfies_conditions_l557_55753

/-- Point type representing a 2D point with real coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on a line segment defined by two other points -/
def isOnSegment (p1 p2 p : Point) : Prop :=
  distance p1 p + distance p p2 = distance p1 p2

/-- The main theorem to prove -/
theorem point_D_satisfies_conditions : 
  let P : Point := ⟨-2, 1⟩
  let Q : Point := ⟨4, 9⟩
  let D : Point := ⟨2.5, 7⟩
  (isOnSegment P Q D) ∧ (distance P D = 2 * distance D Q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_satisfies_conditions_l557_55753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l557_55723

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

-- State the theorem
theorem decreasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ≥ 1/7 ∧ a < 1/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l557_55723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_l557_55791

def sample (x : ℝ) : List ℝ := [2, 3, x, 6, 8]

theorem sample_variance (x : ℝ) 
  (h_avg : (List.sum (sample x)) / 5 = 5) :
  let variance := (List.sum (List.map (λ y => (y - 5)^2) (sample x))) / 5
  variance = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_l557_55791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hawks_win_probability_l557_55708

-- Define the number of games
def num_games : ℕ := 5

-- Define the probability of Hawks winning a single game
def hawks_win_prob : ℚ := 4/5

-- Define the minimum number of games Hawks need to win
def min_wins : ℕ := 4

-- Define the probability of Hawks winning at least 4 out of 5 games
def hawks_win_at_least_4 : ℚ := 73728/100000

-- Theorem statement
theorem hawks_win_probability : 
  (Finset.sum (Finset.range (num_games - min_wins + 1))
    (λ k ↦ (Nat.choose num_games (num_games - k)) * 
          (hawks_win_prob ^ (num_games - k)) * 
          ((1 - hawks_win_prob) ^ k))) = hawks_win_at_least_4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hawks_win_probability_l557_55708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_to_circle_under_inversion_l557_55770

/-- An inversion is a transformation of the plane. -/
structure Inversion where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

/-- A line in the plane. -/
structure Line where
  point : EuclideanSpace ℝ (Fin 2)
  direction : EuclideanSpace ℝ (Fin 2)

/-- A circle in the plane. -/
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

/-- The image of a point under an inversion. -/
def image (inv : Inversion) (p : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- Membership of a point in a line. -/
def Line.mem (p : EuclideanSpace ℝ (Fin 2)) (l : Line) : Prop :=
  sorry

/-- Membership of a point in a circle. -/
def Circle.mem (p : EuclideanSpace ℝ (Fin 2)) (c : Circle) : Prop :=
  sorry

/-- Theorem: Under an inversion, a line not passing through the center
    is transformed into a circle passing through the center. -/
theorem line_to_circle_under_inversion
  (inv : Inversion) (l : Line) (h : l.point ≠ inv.center) :
  ∃ (c : Circle), c.center ≠ inv.center ∧ 
  (∀ (p : EuclideanSpace ℝ (Fin 2)), l.mem p → c.mem (image inv p)) ∧
  c.mem inv.center :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_to_circle_under_inversion_l557_55770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_in_triangle_ABC_l557_55737

theorem angle_B_in_triangle_ABC (A B : Real) : 
  0 < A ∧ A < π → 
  0 < B ∧ B < (3 * π) / 4 →
  Real.sin A + Real.cos A = Real.sqrt 2 → 
  Real.sqrt 3 * Real.cos A = -(Real.sqrt 2 * Real.cos (π / 2 + B)) → 
  B = π / 3 ∨ B = 2 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_in_triangle_ABC_l557_55737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l557_55713

theorem trigonometric_identities (α : Real) 
  (h1 : Real.sin α = 4/5) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.sin (2*α) = -24/25 ∧ 
  Real.cos (α + π/4) = -7*Real.sqrt 2/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l557_55713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l557_55730

-- Define the domain D
variable (D : Type)

-- Define functions f and g
variable (f g : D → ℝ)

-- Define the parameter a
variable (a : ℝ)

-- Define the parameter p
variable (p : ℝ)

-- Theorem statement
theorem range_of_a (h1 : a > 0)
  (h2 : ∀ x : D, -(2*a + 3) ≤ f x ∧ f x ≤ a + 6)
  (h3 : ∀ x : D, a^2 ≤ g x)
  (h4 : ∃ x₁ x₂ : D, |f x₁ - g x₂| < p) :
  -1 < a ∧ a < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l557_55730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_P_exists_l557_55797

-- Define the circles and point M
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_C (x y : ℝ) : Prop := (x-0)^2 + (y-4)^2 = 1
def point_M : ℝ × ℝ := (2, 0)

-- Define a line through C(0,4) intersecting circle O
def line_through_C (k : ℝ) (x y : ℝ) : Prop := y = k*x + 4

-- Define the existence of circle P
def exists_circle_P : Prop :=
  ∃ (k : ℝ) (A B : ℝ × ℝ),
    -- A and B are on circle O
    circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
    -- A and B are on the line through C
    line_through_C k A.1 A.2 ∧ line_through_C k B.1 B.2 ∧
    -- Circle P passes through M and has diameter AB
    (A.1 - point_M.1)*(B.1 - point_M.1) + (A.2 - point_M.2)*(B.2 - point_M.2) = 0

-- Theorem statement
theorem circle_P_exists : exists_circle_P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_P_exists_l557_55797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_function_l557_55707

-- Define the function f(x) = √(x+2)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2)

-- Theorem statement
theorem domain_of_sqrt_function :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≥ -2} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_function_l557_55707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_locus_is_ellipse_l557_55720

/-- A convex trapezoid with fixed longer base AB and moving shorter base CD -/
structure ConvexTrapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  ab_length : ℝ
  cd_length : ℝ
  perimeter : ℝ
  is_convex : Bool
  ab_fixed : Bool
  cd_parallel : Bool
  cd_shorter : cd_length < ab_length
  constant_cd : Bool
  constant_perimeter : Bool

/-- The intersection point of the extensions of the non-parallel sides -/
noncomputable def intersection_point (t : ConvexTrapezoid) : ℝ × ℝ := sorry

/-- An ellipse with foci A and B -/
structure Ellipse where
  A : ℝ × ℝ
  B : ℝ × ℝ
  major_axis : ℝ

/-- Membership of a point in an ellipse -/
def point_in_ellipse (p : ℝ × ℝ) (e : Ellipse) : Prop :=
  let d1 := Real.sqrt ((p.1 - e.A.1)^2 + (p.2 - e.A.2)^2)
  let d2 := Real.sqrt ((p.1 - e.B.1)^2 + (p.2 - e.B.2)^2)
  d1 + d2 = e.major_axis

/-- The theorem stating that the locus of intersection points forms an ellipse -/
theorem intersection_locus_is_ellipse (t : ConvexTrapezoid) :
  ∃ e : Ellipse, ∀ p : ℝ × ℝ, p = intersection_point t → point_in_ellipse p e := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_locus_is_ellipse_l557_55720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_existence_of_c_l557_55719

noncomputable def sequence_a (b : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => Real.sqrt ((sequence_a b n)^2 - 2*(sequence_a b n) + 2) + b

theorem sequence_a_general_term (n : ℕ) :
  sequence_a 1 n = Real.sqrt (n.pred : ℝ) + 1 := by sorry

theorem existence_of_c :
  ∃ c : ℝ, ∀ n : ℕ, sequence_a (-1) (2*n) < c ∧ c < sequence_a (-1) (2*n + 1) := by
  use 1/4
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_existence_of_c_l557_55719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l557_55726

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x * e^x) / (e^(ax) - 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (x * Real.exp x) / (Real.exp (a * x) - 1)

/-- If f(x) = (x * e^x) / (e^(ax) - 1) is an even function, then a = 2 -/
theorem f_even_implies_a_eq_two :
  ∀ a : ℝ, IsEven (f a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l557_55726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_range_of_a_l557_55706

-- Define the function f
noncomputable def f (m n x : ℝ) : ℝ := (m * x + n) / (x^2 + 1)

-- Define the theorem
theorem odd_function_values (m n : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f m n x = -f m n (-x)) →  -- f is odd on [-1, 1]
  f m n 1 = 1 →                                       -- f(1) = 1
  m = 2 ∧ n = 0 :=                                    -- Conclusion: m = 2 and n = 0
by
  sorry

-- Define the theorem for part (2)
theorem range_of_a (a : ℝ) :
  (f 2 0 (a - 1) + f 2 0 (a^2 - 1) < 0) ↔ (0 ≤ a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_range_of_a_l557_55706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worst_player_is_grandson_l557_55761

-- Define the players
inductive Player
| Grandfather
| Son
| Granddaughter
| Grandson

-- Define the sex
inductive Sex
| Male
| Female

-- Define the function to get the sex of a player
def sex : Player → Sex
| Player.Grandfather => Sex.Male
| Player.Son => Sex.Male
| Player.Granddaughter => Sex.Female
| Player.Grandson => Sex.Male

-- Define the function to get the age of a player
def age : Player → ℕ := sorry

-- Define the function to determine if two players are twins
def isTwin : Player → Player → Prop := sorry

-- Define the worst and best players
def worst : Player := sorry
def best : Player := sorry

-- State the theorem
theorem worst_player_is_grandson :
  (∃ p : Player, isTwin worst p) ∧ 
  (sex worst ≠ sex best) ∧
  (age worst = age best) →
  worst = Player.Grandson :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worst_player_is_grandson_l557_55761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_square_area_ratio_l557_55733

theorem fence_square_area_ratio : 
  ∀ s : ℝ, s > 0 → 
  (4 * s^2) / ((4 * s)^2) = (1 / 4 : ℝ) := by
  intro s hs
  -- Simplify the expression
  have h1 : (4 * s^2) / ((4 * s)^2) = (4 * s^2) / (16 * s^2) := by ring
  -- Further simplification
  have h2 : (4 * s^2) / (16 * s^2) = 1 / 4 := by
    field_simp [hs]
    ring
  -- Combine the steps
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_square_area_ratio_l557_55733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_one_l557_55704

-- Define the function g
def g : ℝ → ℝ := sorry

-- Define the function f in terms of g
def f (x : ℝ) : ℝ := g x + 2

-- State the theorem
theorem f_neg_two_equals_one
  (h_odd : ∀ x, g (-x) = -g x)  -- g is an odd function
  (h_f_two : f 2 = 3)           -- f(2) = 3
  : f (-2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_one_l557_55704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_iff_a_eq_one_l557_55717

/-- The function f(x) = (x^2 * 2^x) / (4^(ax) + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 * Real.rpow 2 x) / (Real.rpow 4 (a*x) + 1)

/-- f is an even function if f(-x) = f(x) for all x -/
def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem f_even_iff_a_eq_one (a : ℝ) : 
  is_even_function (f a) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_iff_a_eq_one_l557_55717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arccos_sqrt3_over_2_l557_55700

theorem arccos_sqrt3_over_2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arccos_sqrt3_over_2_l557_55700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l557_55798

-- Define the function f(x) = e^x - cos(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.cos x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := Real.exp x + Real.sin x

-- Theorem statement
theorem tangent_line_at_zero :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l557_55798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_example_custom_op_equation_solution_l557_55777

-- Define the custom operation
noncomputable def custom_op (a b : ℝ) : ℝ := 1 / b - 1 / a

-- Theorem 1: 2※(-2) = -1
theorem custom_op_example : custom_op 2 (-2) = -1 := by
  -- Unfold the definition of custom_op
  unfold custom_op
  -- Simplify the expression
  simp [div_neg, neg_div]
  -- Perform the arithmetic
  ring

-- Theorem 2: The solution to 2※(2x-1) = 1 is x = 5/6
theorem custom_op_equation_solution :
  ∃ x : ℝ, custom_op 2 (2*x - 1) = 1 ∧ x = 5/6 := by
  -- Provide the witness
  use 5/6
  constructor
  · -- Prove that custom_op 2 (2*(5/6) - 1) = 1
    unfold custom_op
    simp
    -- Perform the arithmetic
    ring
  · -- Prove that 5/6 = 5/6
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_example_custom_op_equation_solution_l557_55777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_implies_omega_range_l557_55786

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x / 2) ^ 2 + (1/2) * Real.sin (ω * x) - 1/2

theorem no_zeros_implies_omega_range (ω : ℝ) (h1 : ω > 0) :
  (∀ x ∈ Set.Ioo π (2*π), f ω x ≠ 0) →
  ω ∈ Set.Ioc 0 (1/8) ∪ Set.Icc (1/4) (5/8) :=
by
  sorry

#check no_zeros_implies_omega_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_implies_omega_range_l557_55786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_painting_cost_l557_55739

/-- Calculates the sum of digits in a number --/
def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Calculates the nth term of an arithmetic sequence --/
def arithmeticSequenceTerm (a1 d n : ℕ) : ℕ := a1 + d * (n - 1)

/-- Calculates the cost of painting house numbers for one side of the street --/
def costOfPaintingSide (a1 d n : ℕ) : ℕ := 
  List.range n |>.map (fun i => sumOfDigits (arithmeticSequenceTerm a1 d (i + 1))) |>.sum

theorem total_painting_cost : 
  let southSideCost := costOfPaintingSide 5 7 30
  let northSideCost := costOfPaintingSide 2 7 30
  southSideCost + northSideCost = 161 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_painting_cost_l557_55739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_attendance_l557_55744

theorem exam_attendance :
  let math_pass_rate : ℚ := 35/100
  let math_fail_count : ℕ := 546
  let science_pass_rate : ℚ := 42/100
  let science_fail_count : ℕ := 458
  let english_pass_rate : ℚ := 38/100
  let english_fail_count : ℕ := 490

  let math_total : ℕ := (math_fail_count : ℚ) / (1 - math_pass_rate) |>.ceil.toNat
  let science_total : ℕ := (science_fail_count : ℚ) / (1 - science_pass_rate) |>.ceil.toNat
  let english_total : ℕ := (english_fail_count : ℚ) / (1 - english_pass_rate) |>.ceil.toNat

  math_total = 840 ∧ science_total = 790 ∧ english_total = 790 :=
by
  sorry

#check exam_attendance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_attendance_l557_55744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l557_55711

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 3√7/4 under the given conditions. -/
theorem triangle_area (a b c : ℝ) (A B C : Real) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths for an acute triangle
  Real.sin B * Real.sin C / Real.sin A = 3 * Real.sqrt 7 / 2 →
  b = 4 * a →
  a + c = 5 →
  (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l557_55711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l557_55794

theorem inequality_proof (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  Real.sqrt (Real.sin x) * (Real.tan x) ^ (1/4) +
  Real.sqrt (Real.cos x) * (1 / Real.tan x) ^ (1/4) ≥ 8 ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l557_55794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_positive_integers_l557_55776

def M : Set ℕ := sorry

axiom M_contains_2018 : 2018 ∈ M

axiom M_closed_under_divisors :
  ∀ m : ℕ, m ∈ M → ∀ d : ℕ, d > 0 ∧ d ∣ m → d ∈ M

axiom M_closed_under_km_plus_one :
  ∀ k m : ℕ, k ∈ M → m ∈ M → 1 < k → k < m → k * m + 1 ∈ M

theorem M_equals_positive_integers : M = {n : ℕ | n > 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_positive_integers_l557_55776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_budget_percentage_is_95_percent_l557_55769

/-- Represents a sales representative's earnings and allocations --/
structure SalesRep where
  hourlyRate : ℚ
  commissionRate : ℚ
  hoursWorked : ℚ
  totalSales : ℚ
  insuranceAllocation : ℚ

/-- Calculates the percentage of total earnings allocated for monthly budget --/
def monthlyBudgetPercentage (rep : SalesRep) : ℚ :=
  let basicSalary := rep.hourlyRate * rep.hoursWorked
  let commission := rep.commissionRate * rep.totalSales
  let totalEarnings := basicSalary + commission
  let monthlyBudget := totalEarnings - rep.insuranceAllocation
  (monthlyBudget / totalEarnings) * 100

/-- Theorem stating that given the conditions, the monthly budget percentage is 95% --/
theorem monthly_budget_percentage_is_95_percent :
  let kristy : SalesRep := {
    hourlyRate := 15/2,
    commissionRate := 4/25,
    hoursWorked := 160,
    totalSales := 25000,
    insuranceAllocation := 260
  }
  monthlyBudgetPercentage kristy = 95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_budget_percentage_is_95_percent_l557_55769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_less_than_half_outside_l557_55752

/-- Represents a triangle with an inscribed circle and a circumscribed square -/
structure InscribedCircleTriangle where
  /-- The triangle -/
  triangle : Set (ℝ × ℝ)
  /-- The inscribed circle -/
  circle : Set (ℝ × ℝ)
  /-- The circumscribed square -/
  square : Set (ℝ × ℝ)
  /-- The circle is inscribed in the triangle -/
  circle_inscribed : circle ⊆ triangle
  /-- The square is circumscribed around the circle -/
  square_circumscribed : circle ⊆ square

/-- The perimeter of the square -/
noncomputable def square_perimeter (config : InscribedCircleTriangle) : ℝ := 
  sorry

/-- The length of the square's perimeter that lies outside the triangle -/
noncomputable def outside_perimeter (config : InscribedCircleTriangle) : ℝ := 
  sorry

/-- Theorem: Less than half of the square's perimeter lies outside the triangle -/
theorem less_than_half_outside (config : InscribedCircleTriangle) :
  outside_perimeter config < (square_perimeter config) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_less_than_half_outside_l557_55752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_distribution_problem_l557_55750

theorem money_distribution_problem (n : ℕ) : 3 * n + n * (n - 1) / 2 = 100 * n → n = 195 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_distribution_problem_l557_55750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_tangent_impossibility_l557_55736

theorem sine_tangent_impossibility : ∀ θ : Real,
  (Real.sin θ ≠ 0.27413 ∨ Real.tan θ ≠ 0.25719) ∧
  (Real.sin θ ≠ 0.25719 ∨ Real.tan θ ≠ 0.27413) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_tangent_impossibility_l557_55736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l557_55772

-- Define the logarithms
noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 14 / Real.log 7

-- State the theorem
theorem log_inequality : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l557_55772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABP_area_range_l557_55787

/-- The line equation x + y - 6 = 0 -/
def line (x y : ℝ) : Prop := x + y - 6 = 0

/-- The ellipse equation x^2 + 2y^2 = 6 -/
def ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 6

/-- Point A is the intersection of the line with the x-axis -/
noncomputable def point_A : ℝ × ℝ := (6, 0)

/-- Point B is the intersection of the line with the y-axis -/
noncomputable def point_B : ℝ × ℝ := (0, 6)

/-- Point P lies on the ellipse -/
def point_P (x y : ℝ) : Prop := ellipse x y

/-- The area of triangle ABP given coordinates of P -/
noncomputable def triangle_area (px py : ℝ) : ℝ := 
  abs ((px * 6 + py * 6) / 2)

/-- The theorem stating the range of possible areas for triangle ABP -/
theorem triangle_ABP_area_range :
  ∀ px py : ℝ, point_P px py →
  9 ≤ triangle_area px py ∧ triangle_area px py ≤ 27 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABP_area_range_l557_55787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_b_greater_c_l557_55762

-- Define the constants
noncomputable def a : ℝ := 3^(1/5)
noncomputable def b : ℝ := Real.log 4 / Real.log 6
noncomputable def c : ℝ := Real.log 2 / Real.log 3

-- State the theorem
theorem a_greater_b_greater_c : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_b_greater_c_l557_55762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_from_sine_cosine_progressions_l557_55766

open Real

theorem equal_angles_from_sine_cosine_progressions 
  (a b c : ℝ) 
  (acute_a : 0 < a ∧ a < π / 2) 
  (acute_b : 0 < b ∧ b < π / 2) 
  (acute_c : 0 < c ∧ c < π / 2) 
  (sine_arithmetic : Real.sin b = (Real.sin a + Real.sin c) / 2) 
  (cosine_geometric : (Real.cos b)^2 = Real.cos a * Real.cos c) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_from_sine_cosine_progressions_l557_55766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l557_55764

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- A circle with center (2, 0) and radius √3 -/
def target_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 3

/-- Asymptote of the hyperbola is tangent to the target circle -/
def asymptote_tangent_to_circle (h : Hyperbola) : Prop :=
  ∃ (k : ℝ), (∀ x y, y = k * x → target_circle x y) ∧ 
              (k = h.b / h.a ∨ k = -h.b / h.a)

/-- The main theorem to prove -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_tangent : asymptote_tangent_to_circle h) : 
  eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l557_55764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_square_semicircle_l557_55780

/-- A predicate stating that points P, Q, and N form a semicircle with PQ as diameter. -/
def is_semicircle (P Q N : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- A predicate stating that points P, Q, R, and S form a square. -/
def is_square (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- A predicate stating that square PQRS is tangent to the semicircle with diameter PQ. -/
def tangent_square_to_semicircle (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- The length of a line segment between two points. -/
def segment_length (P Q : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- The proportion of an arc PN to the full semicircle arc PQ. -/
def arc_proportion (P N Q : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- Given a semicircle with diameter PQ of length 10 and a tangential square PQRS,
    if point N is located at three-quarters of the arc PQ starting from P,
    then the length of segment NR is 5√3. -/
theorem tangent_square_semicircle (P Q R S N : EuclideanSpace ℝ (Fin 2))
    (h1 : is_semicircle P Q N)
    (h2 : is_square P Q R S)
    (h3 : tangent_square_to_semicircle P Q R S)
    (h4 : segment_length P Q = 10)
    (h5 : arc_proportion P N Q = 3/4) :
  segment_length N R = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_square_semicircle_l557_55780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l557_55779

theorem solve_exponential_equation (x : ℝ) :
  (5 : ℝ) ^ (2 * x) = Real.sqrt 125 → x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l557_55779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_helen_washing_time_l557_55790

/-- The number of weeks in a year -/
def weeks_in_year : ℕ := 52

/-- The frequency of washing pillowcases in weeks -/
def washing_frequency : ℕ := 4

/-- The time it takes to wash pillowcases in minutes -/
def washing_time : ℕ := 30

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem: Helen spends 6.5 hours per year washing her pillowcases -/
theorem helen_washing_time : 
  (weeks_in_year / washing_frequency : ℚ) * washing_time / minutes_per_hour = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_helen_washing_time_l557_55790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_circumference_l557_55709

noncomputable def jack_circumference : ℝ := 12

noncomputable def charlie_circumference (jack : ℝ) : ℝ := (jack / 2) + 9

noncomputable def bill_circumference (charlie : ℝ) : ℝ := (2 / 3) * charlie

noncomputable def maya_circumference (jack charlie : ℝ) : ℝ := (jack + charlie) / 2

noncomputable def thomas_circumference (bill : ℝ) : ℝ := (2 * bill) - 3

theorem combined_circumference :
  let jack := jack_circumference
  let charlie := charlie_circumference jack
  let bill := bill_circumference charlie
  let maya := maya_circumference jack charlie
  let thomas := thomas_circumference bill
  jack + charlie + bill + maya + thomas = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_circumference_l557_55709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_product_l557_55743

/-- X represents the variable x in the monomial -/
noncomputable def X : ℤ → ℤ := sorry

/-- Y represents the variable y in the monomial -/
noncomputable def Y : ℤ → ℤ := sorry

/-- Given two monomials that are like terms, prove their product -/
theorem monomial_product (a b : ℤ) 
  (h : a - 2*b = 3 ∧ 2*a + b = 8*b) : 
  ((-2 : ℤ) * X (a - 2*b) * Y (2*a + b)) * (X 3 * Y (8*b)) =
  (-2 : ℤ) * X 6 * Y 32 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_product_l557_55743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tims_average_sleep_l557_55728

-- Define the sleep hours for each day of the week
def weekday_sleep : List Float := [6, 6, 10, 10, 8]
def weekend_sleep : List Float := [9, 9]

-- Define the power nap duration
def power_nap : Float := 0.5

-- Define the number of workdays
def workdays : Nat := 5

-- Define the number of days in a week
def days_in_week : Nat := 7

-- Theorem statement
theorem tims_average_sleep :
  let total_weekday_sleep := (weekday_sleep.sum + workdays.toFloat * power_nap)
  let total_weekend_sleep := weekend_sleep.sum
  let total_sleep := total_weekday_sleep + total_weekend_sleep
  (total_sleep / days_in_week.toFloat) = 60.5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tims_average_sleep_l557_55728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_elements_in_set_l557_55760

theorem divisible_elements_in_set (p : ℕ) (hp : p.Prime) (hp5 : p > 5) :
  ∃ a b, a ∈ {x | ∃ n : ℕ, x = p - n^2 ∧ n^2 < p} ∧
         b ∈ {x | ∃ n : ℕ, x = p - n^2 ∧ n^2 < p} ∧
         a ∣ b ∧ 1 < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_elements_in_set_l557_55760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l557_55725

open Set

def solution_set : Set ℝ := Iic 4 ∪ Ioi 5

theorem inequality_equivalence (x : ℝ) (h : x ≠ 4) :
  (x - 2) / (x - 4) ≤ 3 ↔ x ∈ solution_set :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l557_55725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_additional_discount_l557_55715

/-- Represents the additional discount percentage during a special sale -/
noncomputable def additional_discount (list_price : ℝ) (typical_discount_min : ℝ) (typical_discount_max : ℝ) (lowest_sale_price_percent : ℝ) : ℝ :=
  let price_after_max_discount := list_price * (1 - typical_discount_max)
  let lowest_sale_price := list_price * lowest_sale_price_percent
  (price_after_max_discount - lowest_sale_price) / price_after_max_discount * 100

/-- Theorem stating the additional discount during the sale -/
theorem sale_additional_discount :
  let list_price : ℝ := 80
  let typical_discount_min : ℝ := 0.30
  let typical_discount_max : ℝ := 0.50
  let lowest_sale_price_percent : ℝ := 0.40
  additional_discount list_price typical_discount_min typical_discount_max lowest_sale_price_percent = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_additional_discount_l557_55715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l557_55740

-- Define the parametric equations as noncomputable due to Real.sqrt
noncomputable def x (t : ℝ) : ℝ := -2 - Real.sqrt 2 * t
noncomputable def y (t : ℝ) : ℝ := 3 + Real.sqrt 2 * t

-- State the theorem
theorem line_equation :
  ∀ (t : ℝ), x t + y t - 1 = 0 := by
  intro t
  -- Expand the definitions of x and y
  unfold x y
  -- Simplify the expression
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l557_55740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_length_l557_55758

/-- Calculates the number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a : Int) (l : Int) (d : Int) : Nat :=
  Int.toNat ((l - a) / d + 1)

/-- The number of terms in the arithmetic sequence from -53 to 87 with common difference 5 -/
theorem sequence_length : arithmeticSequenceLength (-53) 87 5 = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_length_l557_55758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_class_average_l557_55732

def overall_average (sections : List (Nat × Nat × Nat)) : ℚ :=
  let total_marks := sections.foldl (fun acc (students, mean_mark, _) => acc + students * mean_mark) 0
  let total_students := sections.foldl (fun acc (students, _, _) => acc + students) 0
  (total_marks : ℚ) / total_students

theorem chemistry_class_average :
  let sections := [(55, 50, 1), (35, 60, 2), (45, 55, 3), (42, 45, 4)]
  abs (overall_average sections - 52.09) < 0.01 := by
  sorry

#eval overall_average [(55, 50, 1), (35, 60, 2), (45, 55, 3), (42, 45, 4)]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_class_average_l557_55732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_condition_and_max_value_l557_55785

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - 3*x

def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x - 3

theorem f_increasing_condition_and_max_value :
  (∀ a : ℝ, (∀ x ≥ 1, (∀ y ≥ x, f a y ≥ f a x)) ↔ a ≤ 0) ∧
  (∀ a : ℝ, f_deriv a (-1/3) = 0 → ∃ M : ℝ, M = -6 ∧ ∀ x ∈ Set.Icc 1 4, f a x ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_condition_and_max_value_l557_55785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l557_55781

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  a : ℝ  -- Length of one non-parallel side
  b : ℝ  -- Length of the other non-parallel side
  c : ℝ  -- Length of one base
  d : ℝ  -- Length of the other base
  h : ℝ  -- Height of the trapezoid
  θ : ℝ  -- Intersection angle between the sides forming the bases

/-- The area of an isosceles trapezoid -/
noncomputable def area (t : IsoscelesTrapezoid) : ℝ := (1/2) * (t.c + t.d) * t.h

/-- Theorem stating that the area of an isosceles trapezoid is (1/2) * (c + d) * h -/
theorem isosceles_trapezoid_area (t : IsoscelesTrapezoid) :
  area t = (1/2) * (t.c + t.d) * t.h := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l557_55781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_eval_l557_55784

/-- The function g as defined in the problem -/
noncomputable def g (a b c : ℝ) : ℝ := (c^2 + a^2) / (c - b)

/-- Theorem stating that g(2, -3, 1) = 5/4 -/
theorem g_eval : g 2 (-3) 1 = 5/4 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expression
  simp [pow_two]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_eval_l557_55784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l557_55701

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2) / (2 + x)

theorem f_range :
  ∀ y ∈ Set.range f,
  (0 ≤ y ∧ y ≤ Real.sqrt 3 / 3) ∧
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 0 ∧
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l557_55701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l557_55738

/-- The line x + √3y - 4 = 0 is tangent to the circle x^2 + y^2 = 4 -/
theorem line_tangent_to_circle :
  ∃ (x y : ℝ), 
    (x + Real.sqrt 3 * y - 4 = 0) ∧ 
    (x^2 + y^2 = 4) ∧
    (∀ (x' y' : ℝ), x' + Real.sqrt 3 * y' - 4 = 0 → x'^2 + y'^2 = 4 → (x', y') = (x, y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l557_55738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_99_minutes_l557_55702

/-- Represents a cyclist with their speeds on different terrains -/
structure Cyclist where
  uphill_speed : ℚ
  downhill_speed : ℚ
  flat_speed : ℚ

/-- Represents a route with distances for different terrains -/
structure Route where
  uphill_distance : ℚ
  downhill_distance : ℚ
  flat_distance : ℚ

/-- Calculates the time taken by a cyclist to complete a route -/
def time_taken (c : Cyclist) (r : Route) : ℚ :=
  r.uphill_distance / c.uphill_speed +
  r.downhill_distance / c.downhill_speed +
  r.flat_distance / c.flat_speed

/-- The main theorem to prove -/
theorem time_difference_is_99_minutes
  (joey : Cyclist)
  (sue : Cyclist)
  (route : Route)
  (h1 : joey.uphill_speed = 6)
  (h2 : joey.downhill_speed = 25)
  (h3 : joey.flat_speed = 15)
  (h4 : sue.uphill_speed = 12)
  (h5 : sue.downhill_speed = 35)
  (h6 : sue.flat_speed = 25)
  (h7 : route.uphill_distance = 12)
  (h8 : route.downhill_distance = 10)
  (h9 : route.flat_distance = 20) :
  (time_taken joey route - time_taken sue route) * 60 = 99 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_99_minutes_l557_55702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_zero_l557_55771

/-- The area of a triangle with vertices (4,1,2), (2,5,3), and (8,13,10) is 0 -/
theorem triangle_area_zero : 
  let a : Fin 3 → ℝ := ![4, 1, 2]
  let b : Fin 3 → ℝ := ![2, 5, 3]
  let c : Fin 3 → ℝ := ![8, 13, 10]
  let area := (1/4) * Real.sqrt (
    ((a 0) * ((b 1) - (c 1)) + (b 0) * ((c 1) - (a 1)) + (c 0) * ((a 1) - (b 1)))^2 +
    ((a 1) * ((b 2) - (c 2)) + (b 1) * ((c 2) - (a 2)) + (c 1) * ((a 2) - (b 2)))^2 +
    ((a 2) * ((b 0) - (c 0)) + (b 2) * ((c 0) - (a 0)) + (c 2) * ((a 0) - (b 0)))^2
  )
  area = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_zero_l557_55771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_l557_55751

/-- The function g(x) as defined in the problem -/
noncomputable def g (x : ℝ) : ℝ := x + (2*x)/(x^2 + 2) + (x*(x + 5))/(x^2 + 3) + (3*(x + 3))/(x*(x^2 + 3))

/-- Theorem stating that g(x) has a minimum value of 6 for all positive real x -/
theorem g_min_value (x : ℝ) (h : x > 0) : g x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_l557_55751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_relation_l557_55793

/-- A tetrahedron in 3D space -/
structure Tetrahedron where
  -- Define the structure of a tetrahedron
  mk :: -- Add constructor

/-- A quadrilateral in 3D space -/
structure Quadrilateral where
  -- Define the structure of a quadrilateral
  mk :: -- Add constructor

/-- Volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Construct a tetrahedron from a quadrilateral -/
noncomputable def tetrahedron_from_quadrilateral (q : Quadrilateral) : Tetrahedron :=
  sorry

/-- Check if the sides of a quadrilateral are perpendicular to the faces of a tetrahedron -/
def Quadrilateral.sides_perpendicular_to_faces (q : Quadrilateral) (t : Tetrahedron) : Prop :=
  sorry

/-- Check if the side lengths of a quadrilateral are equal to the face areas of a tetrahedron -/
def Quadrilateral.side_lengths_equal_face_areas (q : Quadrilateral) (t : Tetrahedron) : Prop :=
  sorry

/-- Given a tetrahedron ABCD and a spatial quadrilateral KLMN, prove that the volume of KLMN
    is 3/4 of the volume of ABCD under certain conditions. -/
theorem tetrahedron_volume_relation 
  (ABCD : Tetrahedron) 
  (KLMN : Quadrilateral) 
  (h1 : KLMN.sides_perpendicular_to_faces ABCD)
  (h2 : KLMN.side_lengths_equal_face_areas ABCD) :
  volume (tetrahedron_from_quadrilateral KLMN) = (3/4) * volume ABCD := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_relation_l557_55793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_g_of_3_l557_55796

noncomputable def g (x : ℝ) : ℝ := -1 / (x^2)

theorem nested_g_of_3 : g (g (g (g (g 3)))) = -(1 / 3^64) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_g_of_3_l557_55796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_zero_l557_55749

-- Define the function f
def f (x : ℚ) : ℚ :=
  if x ≤ 0 then 4 * x else 2 * x

-- Theorem statement
theorem f_sum_equals_zero : f (-1) + f 2 = 0 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the if-then-else expressions
  simp
  -- Perform arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_zero_l557_55749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l557_55755

-- Define the function f(x) = ln x + x - 4
noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

-- State the theorem
theorem zero_point_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l557_55755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_bounds_l557_55705

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*(x - 1)

-- Define the circle (renamed to avoid conflict)
def circleC (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- Define a point on the parabola
def point_on_parabola (x₀ y₀ : ℝ) : Prop := parabola x₀ y₀

-- Define the distance |AB|
noncomputable def distance_AB (x₀ y₀ : ℝ) : ℝ :=
  Real.sqrt (((2*y₀)/(x₀ + 2))^2 + 4*x₀/(x₀ + 2))

-- Theorem statement
theorem tangent_intersection_bounds (x₀ y₀ : ℝ) :
  point_on_parabola x₀ y₀ →
  x₀ ≥ 1 →
  2 * Real.sqrt 3 / 3 ≤ distance_AB x₀ y₀ ∧ distance_AB x₀ y₀ ≤ Real.sqrt 39 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_bounds_l557_55705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_product_l557_55722

theorem max_value_of_sum_product (a b c d : ℕ) : 
  a ∈ ({1, 2, 3, 4} : Set ℕ) → b ∈ ({1, 2, 3, 4} : Set ℕ) → 
  c ∈ ({1, 2, 3, 4} : Set ℕ) → d ∈ ({1, 2, 3, 4} : Set ℕ) →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (∀ x : ℕ, x ∈ ({a, b, c, d} : Set ℕ) → x ∈ ({1, 2, 3, 4} : Set ℕ)) →
  (∀ x : ℕ, x ∈ ({1, 2, 3, 4} : Set ℕ) → x ∈ ({a, b, c, d} : Set ℕ)) →
  (a * b + b * c + c * d + d * a) ≤ 25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_product_l557_55722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_restoration_percentage_l557_55768

theorem price_restoration_percentage (original_price : ℝ) (decrease_percentage : ℝ) 
  (h1 : original_price = 120)
  (h2 : decrease_percentage = 15) : 
  ∃ ε > 0, |((original_price / (original_price * (1 - decrease_percentage / 100))) - 1) * 100 - 17.647| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_restoration_percentage_l557_55768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_path_l557_55721

/-- A regular polygon with 2n + 1 sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n + 1) → ℝ × ℝ
  regular : ∀ i j : Fin (2*n + 1), dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

/-- All diagonals of a polygon -/
def allDiagonals (p : RegularPolygon n) : Set (ℝ × ℝ × ℝ × ℝ) :=
  { (a, b, c, d) | ∃ (i j : Fin (2*n + 1)), i ≠ j ∧ (a, b) = p.vertices i ∧ (c, d) = p.vertices j }

/-- A path in the polygon -/
def PolygonPath (p : RegularPolygon n) := List (Fin (2*n + 1))

/-- Check if a path traverses each line exactly once -/
def isValidPath (p : RegularPolygon n) (path : PolygonPath p) : Prop :=
  sorry

/-- Main theorem: there exists a path that traverses each line exactly once -/
theorem exists_valid_path (n : ℕ) (p : RegularPolygon n) :
  ∃ (path : PolygonPath p), isValidPath p path :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_path_l557_55721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_relation_l557_55734

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid G of the triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define the side lengths of the triangle
noncomputable def side_lengths (t : Triangle) : ℝ × ℝ × ℝ := sorry

-- Define the circumradius of the triangle
noncomputable def circumradius (t : Triangle) : ℝ := sorry

-- Define the squared distance between two points
def dist_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Define the norm squared of a point
def norm_squared (p : ℝ × ℝ) : ℝ :=
  p.1^2 + p.2^2

-- Theorem statement
theorem centroid_distance_relation (t : Triangle) (P : ℝ × ℝ) :
  let G := centroid t
  let (a, b, c) := side_lengths t
  let R := circumradius t
  dist_squared P t.A + dist_squared P t.B + dist_squared P t.C - dist_squared P G =
    2 * norm_squared P + 2 * R^2 + (a^2 + b^2 + c^2) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_relation_l557_55734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_at_N_MN_passes_through_S_locus_of_R_l557_55710

-- Define the basic geometric objects
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

structure Square where
  center : Point
  side : ℝ

-- Define the given segment AB and point M
noncomputable def A : Point := ⟨0, 0⟩
noncomputable def B (b : ℝ) : Point := ⟨b, 0⟩
noncomputable def M (m : ℝ) : Point := ⟨m, 0⟩

-- Define the squares AMCD and BMEF
noncomputable def square1 (m : ℝ) : Square := ⟨⟨m/2, m/2⟩, m⟩
noncomputable def square2 (m b : ℝ) : Square := ⟨⟨(m+b)/2, (b-m)/2⟩, b-m⟩

-- Define the circumcircles and their centers
noncomputable def P (m : ℝ) : Point := (square1 m).center
noncomputable def Q (m b : ℝ) : Point := (square2 m b).center
noncomputable def circle1 (m : ℝ) : Circle := ⟨P m, m/Real.sqrt 2⟩
noncomputable def circle2 (m b : ℝ) : Circle := ⟨Q m b, (b-m)/Real.sqrt 2⟩

-- Define point N (the other intersection of the circles)
noncomputable def N (m b : ℝ) : Point := sorry

-- Define lines FA and BC
noncomputable def FA (m b : ℝ) : Line := sorry
noncomputable def BC (m b : ℝ) : Line := sorry

-- Define the fixed point S
noncomputable def S (b : ℝ) : Point := sorry

-- Define the midpoint of PQ
noncomputable def R (m b : ℝ) : Point := ⟨(m + b/2)/2, (2*m + b)/4⟩

theorem intersection_at_N (m b : ℝ) (h : 0 ≤ m ∧ m ≤ b) : 
  (FA m b).a * (N m b).x + (FA m b).b * (N m b).y + (FA m b).c = 0 ∧ 
  (BC m b).a * (N m b).x + (BC m b).b * (N m b).y + (BC m b).c = 0 := 
sorry

theorem MN_passes_through_S (m b : ℝ) (h : 0 ≤ m ∧ m ≤ b) : 
  ∃ t : ℝ, S b = ⟨(M m).x + t * ((N m b).x - (M m).x), (M m).y + t * ((N m b).y - (M m).y)⟩ := 
sorry

theorem locus_of_R (m b : ℝ) (h : 0 ≤ m ∧ m ≤ b) : 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R m b = ⟨b/4 + t*b/2, b/4 + t*b/2⟩ := 
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_at_N_MN_passes_through_S_locus_of_R_l557_55710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l557_55782

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_specific_vectors :
  let a : ℝ × ℝ := (3, 0)
  let b : ℝ × ℝ := (-5, 5)
  angle_between a b = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l557_55782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_l557_55747

/-- The time (in seconds) it takes for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  train_length / (train_speed_kmh * 1000 / 3600)

/-- Theorem: A train of length 140 m traveling at 108 km/hr takes approximately 4.67 seconds to cross an electric pole -/
theorem train_crossing_pole : 
  ∃ (ε : ℝ), ε > 0 ∧ |train_crossing_time 140 108 - 4.67| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_l557_55747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_implies_a_range_l557_55724

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a / x + 2 else -x^2 + 2*x

theorem function_monotonicity_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  a ∈ Set.Icc (-1) 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_implies_a_range_l557_55724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_four_equals_105_l557_55788

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + x + 1

-- Define the properties of g
noncomputable def g : ℝ → ℝ := sorry

-- g is a cubic polynomial (this is implied by its definition in terms of f's roots)
axiom g_cubic : ∃ (a b c d : ℝ), ∀ x, g x = a*x^3 + b*x^2 + c*x + d

-- g(0) = -1
axiom g_zero : g 0 = -1

-- The roots of g are the squares of the roots of f
axiom g_roots : ∀ x, f x = 0 → g (x^2) = 0

-- Theorem to prove
theorem g_four_equals_105 : g 4 = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_four_equals_105_l557_55788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_truncated_cone_volume_l557_55783

/-- The volume of a truncated right circular cone -/
noncomputable def truncatedConeVolume (R h r : ℝ) : ℝ :=
  (1/3) * Real.pi * h * (R^2 + r^2 + R*r)

/-- Theorem: Volume of a specific truncated right circular cone -/
theorem specific_truncated_cone_volume :
  truncatedConeVolume 10 8 5 = (1400/3) * Real.pi := by
  sorry

#check specific_truncated_cone_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_truncated_cone_volume_l557_55783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_2008_power_2008_l557_55746

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The cycle of units digits for powers of 8 -/
def cycleOfEight : List ℕ := [8, 4, 2, 6]

theorem units_digit_of_2008_power_2008 :
  unitsDigit (2008^2008) = 6 :=
by
  have h1 : unitsDigit 2008 = 8 := by sorry
  have h2 : ∀ k : ℕ, unitsDigit (8^k) = cycleOfEight[k % 4]'(by sorry) := by sorry
  have h3 : 2008 % 4 = 0 := by sorry
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof
-- #eval unitsDigit (2008^2008)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_2008_power_2008_l557_55746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l557_55741

noncomputable def f (x : ℝ) := 2 * Real.cos x + 3

theorem tangent_line_at_zero (x y : ℝ) :
  f 0 = 5 →
  (∀ h : ℝ, h ≠ 0 → (f h - f 0) / h = 0) →
  y - 5 = 0 ↔ y = f x ∧ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l557_55741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l557_55731

-- Define the function
noncomputable def f (x : ℝ) : ℝ := |Real.sin x| * Real.cos x + |Real.cos x| * Real.sin x

-- State the theorem
theorem range_of_f : Set.range f = Set.Icc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l557_55731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marlas_errand_time_calculation_l557_55759

theorem marlas_errand_time_calculation : 
  (20 : ℕ) + 30 + 15 + 10 + 5 + 25 + 70 + 30 + 40 + 20 = 265 := by
  -- Driving to school (20) + Bus to store (30) + Grocery shopping (15) +
  -- Walking to gas station (10) + Filling gas (5) + Bicycle to school (25) +
  -- Parent-teacher night (70) + Coffee with friend (30) + Subway home (40) +
  -- Driving home (20)
  norm_num

#check marlas_errand_time_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marlas_errand_time_calculation_l557_55759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l557_55763

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

-- Define the derivative of f
def f_prime (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Define the function g
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := (f_prime a b x) * Real.exp (-x)

theorem function_properties :
  ∃ (a b : ℝ),
    f_prime a b 1 = 2*a ∧
    f_prime a b 2 = -b ∧
    a = -3/2 ∧
    b = -3 ∧
    (∀ x : ℝ, g a b x ≥ -3) ∧
    g a b 0 = -3 ∧
    (∀ x : ℝ, g a b x ≤ 15 * Real.exp (-3)) ∧
    g a b 3 = 15 * Real.exp (-3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l557_55763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_polar_representations_l557_55742

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Converts a polar point to Cartesian coordinates -/
noncomputable def polarToCartesian (p : PolarPoint) : ℝ × ℝ :=
  (p.r * Real.cos p.θ, p.r * Real.sin p.θ)

/-- The theorem stating that the two polar representations are equivalent -/
theorem equivalent_polar_representations :
  let p1 : PolarPoint := ⟨-1, 5 * π / 6⟩
  let p2 : PolarPoint := ⟨1, 11 * π / 6⟩
  polarToCartesian p1 = polarToCartesian p2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_polar_representations_l557_55742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fractional_solution_l557_55735

theorem no_fractional_solution (x y : ℚ) : 
  (∃ m n : ℤ, (13 : ℚ) * x + (4 : ℚ) * y = m ∧ (10 : ℚ) * x + (3 : ℚ) * y = n) → 
  (∃ a b : ℤ, x = a ∧ y = b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fractional_solution_l557_55735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cube_painting_l557_55799

/-- A cube painting configuration -/
structure CubePainting where
  blue : Fin 6
  red : Finset (Fin 6)
  green : Finset (Fin 6)

/-- Predicate to check if a face is visible from a vertex -/
def visibleFromVertex (v : Fin 8) (f : Fin 6) : Prop := sorry

/-- The set of all valid cube paintings -/
def ValidCubePaintings : Set CubePainting :=
  { p : CubePainting |
    p.blue ∉ p.red ∧ p.blue ∉ p.green ∧
    p.red.card = 3 ∧
    p.green.card = 2 ∧
    p.red ∩ p.green = ∅ ∧
    p.red ∪ p.green ∪ {p.blue} = Finset.univ ∧
    ∃ (v : Fin 8), ∀ (f : Fin 6), f ∈ p.red → visibleFromVertex v f }

/-- Theorem stating that there exists a unique valid cube painting -/
theorem unique_cube_painting :
  ∃! (p : CubePainting), p ∈ ValidCubePaintings := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cube_painting_l557_55799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_g_l557_55789

noncomputable def f (x : ℝ) : ℝ := (1 - x^2) / (1 + x^2)

noncomputable def g (x : ℝ) : ℝ := x - Real.sqrt (1 - 2*x)

theorem range_of_f :
  Set.range f = Set.Ioc (-1) 1 := by sorry

theorem range_of_g :
  Set.range g = Set.Iic (-1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_g_l557_55789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_P_equation_l557_55756

-- Define the curves C₁ and C₂
noncomputable def C₁ (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ ∧ 0 ≤ θ ∧ θ < Real.pi / 2
noncomputable def C₂ (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 3

-- Define the intersection point
noncomputable def intersection_point : ℝ × ℝ := (2 * Real.sqrt 3, Real.pi / 6)

-- Define the relationship between Q and P
noncomputable def Q_P_relation (ρ_Q θ_Q ρ_P θ_P : ℝ) : Prop :=
  C₁ ρ_Q θ_Q ∧ ρ_Q = (2/3) * ρ_P ∧ θ_Q = θ_P

-- Define the polar coordinate equation of P
noncomputable def P_equation (ρ θ : ℝ) : Prop := ρ = 10 * Real.cos θ ∧ 0 ≤ θ ∧ θ < Real.pi / 2

theorem intersection_and_P_equation :
  (∀ ρ θ, C₁ ρ θ ∧ C₂ ρ θ ↔ (ρ, θ) = intersection_point) ∧
  (∀ ρ_Q θ_Q ρ_P θ_P, Q_P_relation ρ_Q θ_Q ρ_P θ_P → P_equation ρ_P θ_P) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_P_equation_l557_55756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_operations_l557_55767

theorem arithmetic_operations :
  ((-9) - (-7) + (-6) - 5 = -13) ∧
  ((-5/12 + 2/3 - 3/4) * (-12) = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_operations_l557_55767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_not_perfect_square_l557_55757

theorem gcd_not_perfect_square (m n : ℕ) 
  (h_one_multiple : (m % 3 = 0 ∧ n % 3 ≠ 0) ∨ (m % 3 ≠ 0 ∧ n % 3 = 0))
  (h_positive : m > 0 ∧ n > 0) :
  ¬ ∃ k : ℕ, (Nat.gcd (m^2 + n^2 + 2) (m^2 * n^2 + 3) = k^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_not_perfect_square_l557_55757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toilet_paper_production_ratio_l557_55775

/-- Proves that the ratio of increased production to original production is 3:1 -/
theorem toilet_paper_production_ratio 
  (original_daily_production : ℕ)
  (total_march_production : ℕ)
  (days_in_march : ℕ)
  (h1 : original_daily_production = 7000)
  (h2 : total_march_production = 868000)
  (h3 : days_in_march = 31) :
  (total_march_production - original_daily_production * days_in_march) / 
  (original_daily_production * days_in_march) = 3 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toilet_paper_production_ratio_l557_55775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l557_55795

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 4 / (x - 2)

-- State the theorem
theorem f_minimum_value :
  (∀ x > 2, f x ≥ 6) ∧ (∃ x > 2, f x = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l557_55795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_sail_speed_theorem_l557_55716

/-- A sailboat with two different sized sails -/
structure Sailboat where
  big_sail : ℝ  -- Size of the bigger sail in square feet
  small_sail : ℝ  -- Size of the smaller sail in square feet
  big_sail_speed : ℝ  -- Speed with the bigger sail in MPH
  distance : ℝ  -- Distance to travel in miles
  time_difference : ℝ  -- Time difference between sails in hours

/-- The speed of the sailboat with the smaller sail -/
noncomputable def small_sail_speed (s : Sailboat) : ℝ :=
  s.distance / (s.distance / s.big_sail_speed + s.time_difference)

theorem small_sail_speed_theorem (s : Sailboat) 
  (h1 : s.big_sail = 24)
  (h2 : s.small_sail = 12)
  (h3 : s.big_sail_speed = 50)
  (h4 : s.distance = 200)
  (h5 : s.time_difference = 6) :
  small_sail_speed s = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_sail_speed_theorem_l557_55716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l557_55748

/-- Two vectors are orthogonal if and only if their dot product is zero -/
def are_orthogonal (v w : Fin 3 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2) = 0

theorem perpendicular_lines (c : ℝ) :
  let line1_dir : Fin 3 → ℝ := ![((3 * c - 10) / 4), -3, 2]
  let line2_dir : Fin 3 → ℝ := ![4, c, 5]
  are_orthogonal line1_dir line2_dir := by
  sorry

#check perpendicular_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l557_55748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l557_55714

/-- The volume of a right pyramid with a square base, given specific conditions -/
theorem pyramid_volume (total_surface_area : ℝ) (base_area : ℝ) (triangular_face_area : ℝ) :
  total_surface_area = 768 →
  triangular_face_area = base_area / 3 →
  total_surface_area = base_area + 4 * triangular_face_area →
  ∃ (volume : ℝ), ∃ (h : volume > 0), abs (volume - 853.56) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l557_55714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_basketball_score_l557_55792

theorem leo_basketball_score :
  ∀ (x y : ℚ),
  x + y = 40 →
  (∀ z : ℚ, 0 ≤ z ∧ z ≤ 40 → 0.75 * z + 0.8 * (40 - z) ≤ 32) ∧
  (∃ w : ℚ, 0 ≤ w ∧ w ≤ 40 ∧ 0.75 * w + 0.8 * (40 - w) = 32) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_basketball_score_l557_55792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l557_55773

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides

-- Define the problem conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a^2 + t.b^2 - (t.a * Real.cos t.B + t.b * Real.cos t.A)^2 = 2 * t.a * t.b * Real.cos t.B

-- Define the angle range condition
def angle_range_condition (t : Triangle) : Prop :=
  0 < t.B ∧ t.B < Real.pi ∧ 0 < t.C ∧ t.C < Real.pi

-- Theorem statement
theorem isosceles_triangle (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : angle_range_condition t) : 
  t.B = t.C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l557_55773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2y7_eq_20_l557_55718

/-- The coefficient of x²y⁷ in the expansion of ((x+y)(x-y)⁸) -/
def coefficient_x2y7 : ℤ := 20

theorem coefficient_x2y7_eq_20 : coefficient_x2y7 = 20 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2y7_eq_20_l557_55718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l557_55774

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - 8 * Real.log x

-- State the theorem
theorem f_monotone_increasing :
  ∀ x y, 2 ≤ x ∧ x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l557_55774
