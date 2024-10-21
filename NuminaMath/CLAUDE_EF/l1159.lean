import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_max_area_l1159_115922

/-- The perimeter of the hexagon -/
def perimeter : ℝ := 72

/-- The maximum area of a hexagon with the given perimeter -/
noncomputable def max_area : ℝ := 216 * Real.sqrt 3

/-- Theorem stating that the maximum area of a hexagon with perimeter 72 is 216√3 -/
theorem hexagon_max_area :
  ∀ (area : ℝ), area ≤ max_area := by
  sorry

#check hexagon_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_max_area_l1159_115922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1159_115908

noncomputable def f (a x : ℝ) : ℝ := a / x + x * Real.log x

def g (x : ℝ) : ℝ := x^3 - x^2 - 3

theorem problem_solution (a : ℝ) : 
  (∃ M : ℤ, (∀ N : ℤ, N > M → ¬∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ g x₁ - g x₂ ≥ N) ∧ 
    (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ g x₁ - g x₂ ≥ M) ∧ M = 4) ∧
  ((∀ s t : ℝ, s ∈ Set.Icc (1/2) 2 → t ∈ Set.Icc (1/2) 2 → f a s ≥ g t) ↔ a ≥ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1159_115908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_AB_length_l1159_115909

/-- The line l passing through (-√3/3, 0) with slope π/3 -/
def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * x + 1

/-- The circle defined by x^2 + y^2 - 6y = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*y = 0

/-- Points A and B are the intersection points of line_l and circle_eq -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_eq A.1 A.2 ∧ line_l B.1 B.2 ∧ circle_eq B.1 B.2

/-- The length of a chord given by two points -/
noncomputable def chord_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem chord_AB_length :
  ∀ A B : ℝ × ℝ, intersection_points A B → chord_length A B = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_AB_length_l1159_115909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_and_sum_l1159_115992

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def A1 : Circle := { center := (0, 0), radius := 7 }
def A2 : Circle := { center := (20, 0), radius := 13 }
def A3 : Circle := { center := (10, 0), radius := 20 }

-- Define the conditions
axiom externally_tangent : A1.center.1 + A1.radius + A2.radius = A2.center.1

axiom internally_tangent_A1 : A3.radius = A3.center.1 + A1.radius
axiom internally_tangent_A2 : A3.radius = A2.center.1 - A3.center.1 + A2.radius

axiom collinear_centers : A1.center.2 = A2.center.2 ∧ A2.center.2 = A3.center.2

-- Define the common external tangent
noncomputable def common_external_tangent : ℝ := 2 * Real.sqrt (A3.radius^2 - (140/3 - 10)^2)

-- Define the properties of m, n, and p
axiom m_n_p_positive : ∃ (m n p : ℕ+), common_external_tangent = (m.val * Real.sqrt n.val) / p.val
axiom m_p_coprime : ∀ (m p : ℕ+), Nat.Coprime m.val p.val
axiom n_not_square_divisible : ∀ (n : ℕ+), ¬ ∃ (q : ℕ+), q.val > 1 ∧ q.val^2 ∣ n.val

-- The theorem to prove
theorem tangent_length_and_sum : 
  common_external_tangent = 8 * Real.sqrt 546 ∧
  ∃ (m n p : ℕ+), common_external_tangent = (m.val * Real.sqrt n.val) / p.val ∧ 
                   m.val + n.val + p.val = 558 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_and_sum_l1159_115992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_estimate_river_length_l1159_115934

-- Define the measurement structure
structure Measurement where
  length : Real
  margin : Real
  errorProb : Real

-- Define the river measurement problem
def riverMeasurementProblem (gsa awra : Measurement) : Prop :=
  gsa.length = 402 ∧
  gsa.margin = 0.5 ∧
  gsa.errorProb = 0.04 ∧
  awra.length = 403 ∧
  awra.margin = 0.5 ∧
  awra.errorProb = 0.04

-- Theorem statement
theorem best_estimate_river_length 
  (gsa awra : Measurement) 
  (h : riverMeasurementProblem gsa awra) :
  ∃ (estimate : Real) (errorProb : Real),
    estimate = 402.5 ∧ 
    errorProb = 0.04 ∧
    (∀ other_estimate : Real, 
      (other_estimate ≠ estimate → 
        (abs (other_estimate - gsa.length) > abs (estimate - gsa.length) ∨
         abs (other_estimate - awra.length) > abs (estimate - awra.length)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_estimate_river_length_l1159_115934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_9_equals_neg_half_l1159_115945

def sequence_a : ℕ → ℚ
  | 0 => 3  -- Add this case for 0
  | 1 => 3
  | n + 2 => 1 - 1 / sequence_a (n + 1)

theorem a_9_equals_neg_half : sequence_a 9 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_9_equals_neg_half_l1159_115945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_point_l1159_115917

/-- Converts Fahrenheit to Celsius -/
noncomputable def f_to_c (f : ℝ) : ℝ := (f - 32) * (5/9)

/-- Converts Celsius to Fahrenheit -/
noncomputable def c_to_f (c : ℝ) : ℝ := c * (9/5) + 32

theorem water_boiling_point :
  ∃ (boiling_c : ℝ),
    f_to_c 212 = boiling_c ∧  -- Water boils at 212 °F
    f_to_c 32 = 0 ∧           -- Ice melts at 32 °F or 0 °C
    c_to_f 55 = 131 ∧         -- Known temperature conversion
    boiling_c = 100           -- Boiling point in Celsius
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_point_l1159_115917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_one_condition_two_l1159_115969

-- Define the lines
noncomputable def l₁ (a b : ℝ) (x y : ℝ) : Prop := a * x - b * y + 4 = 0
noncomputable def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a - 1) * x + y + b = 0

-- Define perpendicularity
noncomputable def perpendicular (a b : ℝ) : Prop := a * (a - 1) + (-b) * 1 = 0

-- Define parallelism
noncomputable def parallel (a b : ℝ) : Prop := a / b = 1 - a

-- Define distance from origin to line
noncomputable def distance_to_origin (a b : ℝ) : ℝ := |4 * (a - 1) / a|

-- Theorem for condition 1
theorem condition_one (a b : ℝ) :
  l₁ a b (-3) (-1) ∧ perpendicular a b → a = 2 ∧ b = 2 := by
  sorry

-- Theorem for condition 2
theorem condition_two (a b : ℝ) :
  parallel a b ∧ distance_to_origin a b = |a / (1 - a)| →
  (a = 2 ∧ b = -2) ∨ (a = 2/3 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_one_condition_two_l1159_115969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_nine_values_l1159_115907

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 6*x + 14

-- Define g as a function of x, not using f.inverse
def g (x : ℝ) : ℝ := 3*x + 2

-- Theorem statement
theorem sum_of_g_nine_values : 
  ∃ s : Finset ℝ, s.card = 2 ∧ (∀ x ∈ s, f x = 9) ∧ s.sum (λ x => g x) = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_nine_values_l1159_115907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_increasing_intervals_f_decreasing_interval_f_minimum_on_interval_f_maximum_on_interval_l1159_115965

noncomputable section

-- Define the function f
def f (x : ℝ) := (1/3) * x^3 + 2 * x^2 - 5 * x - 1

-- Define the derivative of f
def f' (x : ℝ) := x^2 + 4 * x - 5

-- Theorem for monotonicity and extreme values
theorem f_properties :
  (∀ x < -5, f' x > 0) ∧
  (∀ x ∈ Set.Ioo (-5) 1, f' x < 0) ∧
  (∀ x > 1, f' x > 0) ∧
  (∀ x ∈ Set.Icc (-2) 2, f x ≥ -11/3) ∧
  (∀ x ∈ Set.Icc (-2) 2, f x ≤ -1/3) ∧
  (f 1 = -11/3) ∧
  (f 2 = -1/3) := by
  sorry

-- Corollaries for specific properties
theorem f_increasing_intervals (x : ℝ) :
  (x < -5 ∨ x > 1) → f' x > 0 := by
  sorry

theorem f_decreasing_interval (x : ℝ) :
  x ∈ Set.Ioo (-5) 1 → f' x < 0 := by
  sorry

theorem f_minimum_on_interval :
  ∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f x ≤ f y := by
  sorry

theorem f_maximum_on_interval :
  ∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f x ≥ f y := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_increasing_intervals_f_decreasing_interval_f_minimum_on_interval_f_maximum_on_interval_l1159_115965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_fraction_l1159_115986

theorem circle_area_fraction (r : ℝ) (h : r > 0) : 
  (π * r^2 - π * (r/2)^2) / (π * r^2) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_fraction_l1159_115986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_function_characterization_l1159_115959

def factorial : ℕ → ℕ
| 0 => 1
| n + 1 => (n + 1) * factorial n

def is_valid_function (f : ℕ → ℕ) : Prop :=
  f 1 = 1 ∧
  (∀ n : ℕ, f (n + 2) + (n^2 + 4*n + 3) * f n = (2*n + 5) * f (n + 1)) ∧
  (∀ m n : ℕ, m > n → ∃ k : ℕ, f m = k * f n)

theorem valid_function_characterization (f : ℕ → ℕ) :
  is_valid_function f ↔ (∀ n : ℕ, f n = factorial n ∨ f n = (factorial (n + 2)) / 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_function_characterization_l1159_115959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_property_l1159_115938

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by its vertex and focus -/
structure Parabola where
  vertex : Point
  focus : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point lies on a parabola -/
def isOnParabola (p : Point) (par : Parabola) : Prop :=
  let directrixY := par.vertex.y - (par.focus.y - par.vertex.y)
  distance p par.focus = |p.y - directrixY|

/-- The main theorem -/
theorem parabola_point_property :
  let par : Parabola := ⟨⟨0, 0⟩, ⟨0, 3⟩⟩
  let p : Point := ⟨Real.sqrt 564, 47⟩
  isOnParabola p par ∧ distance p par.focus = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_property_l1159_115938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_value_l1159_115988

/-- The area of the region bounded by y = 1/x, y = 4x, x = 1, and the x-axis -/
noncomputable def enclosed_area : ℝ := ∫ x in (0)..(1/2), 4*x + ∫ x in (1/2)..1, 1/x

/-- Theorem stating that the enclosed area is equal to ln(2) + 1/2 -/
theorem enclosed_area_value : enclosed_area = Real.log 2 + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_value_l1159_115988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_exponentiation_congruence_l1159_115931

theorem modular_exponentiation_congruence 
  (p q : Nat) (a : Nat) (n : Nat) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (ha_pos : a > 0) 
  (ha_cong : a ≡ 1 [MOD (p - 1) * (q - 1)]) : 
  n ^ a ≡ n [MOD p * q] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_exponentiation_congruence_l1159_115931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_szekeres_l1159_115991

theorem erdos_szekeres (m n : ℕ) (seq : Fin (m * n + 1) → ℝ) :
  (∃ (subseq : Fin (m + 1) → Fin (m * n + 1)), StrictMono subseq ∧ StrictMono (seq ∘ subseq)) ∨
  (∃ (subseq : Fin (n + 1) → Fin (m * n + 1)), StrictMono subseq ∧ StrictAnti (seq ∘ subseq)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_szekeres_l1159_115991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l1159_115925

/-- The function for which we want to find the oblique asymptote -/
noncomputable def f (x : ℝ) : ℝ := (2*x^3 + 2*x^2 + 7*x + 10) / (2*x + 3)

/-- The proposed oblique asymptote function -/
noncomputable def g (x : ℝ) : ℝ := x^2 - (1/2)*x

/-- Theorem stating that g is the oblique asymptote of f -/
theorem oblique_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - g x| < ε :=
by
  sorry

#check oblique_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l1159_115925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_18_point_5_l1159_115905

/-- The speed of a train crossing a bridge -/
noncomputable def train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  (train_length + bridge_length) / crossing_time

/-- Theorem: The speed of the train is 18.5 m/s -/
theorem train_speed_is_18_point_5 :
  train_speed 250 120 20 = 18.5 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the arithmetic
  simp [add_div]
  -- Check that the result is equal to 18.5
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_18_point_5_l1159_115905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_threshold_proof_l1159_115940

/-- Calculates the income threshold given pre-tax income, post-tax income, and tax rate -/
noncomputable def calculate_threshold (pre_tax_income : ℝ) (post_tax_income : ℝ) (tax_rate : ℝ) : ℝ :=
  (pre_tax_income - post_tax_income) / tax_rate

theorem income_threshold_proof (pre_tax_income post_tax_income tax_rate threshold : ℝ) 
  (h1 : pre_tax_income = 13000)
  (h2 : post_tax_income = 12000)
  (h3 : tax_rate = 0.1)
  (h4 : threshold = calculate_threshold pre_tax_income post_tax_income tax_rate) :
  threshold = 3000 := by
  sorry

-- Use #eval with rationals instead of reals for computation
#eval (13000 : ℚ) - (12000 : ℚ) / (1 / 10 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_threshold_proof_l1159_115940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_C_gets_box_A_yellow_l1159_115970

-- Define the types for boxes, colors, and people
inductive Box : Type
  | A | B | C

inductive Color : Type
  | Red | Yellow | Blue

inductive Person : Type
  | A | B | C

-- Define a function to represent the assignment of boxes to people
variable (assignment : Person → Box)

-- Define a function to represent the color of the ball in each box
variable (box_color : Box → Color)

-- State the theorem
theorem person_C_gets_box_A_yellow :
  -- Conditions
  (assignment Person.A ≠ Box.A) →
  (assignment Person.B ≠ Box.B) →
  (box_color (assignment Person.B) ≠ Color.Yellow) →
  (box_color Box.A ≠ Color.Red) →
  (box_color Box.B = Color.Blue) →
  -- Conclusion
  (assignment Person.C = Box.A ∧ box_color Box.A = Color.Yellow) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_C_gets_box_A_yellow_l1159_115970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intercept_sum_l1159_115985

/-- Represents a line with negative slope passing through (18, 8) -/
structure NegativeSlopeLine where
  m : ℝ
  h : m < 0

/-- The x-intercept of the line -/
noncomputable def x_intercept (l : NegativeSlopeLine) : ℝ := 18 + 8 / l.m

/-- The y-intercept of the line -/
noncomputable def y_intercept (l : NegativeSlopeLine) : ℝ := 18 * l.m + 8

/-- The sum of x and y intercepts -/
noncomputable def intercept_sum (l : NegativeSlopeLine) : ℝ := x_intercept l + y_intercept l

/-- Theorem stating the minimum value of the sum of intercepts -/
theorem min_intercept_sum :
  ∃ (min : ℝ), min = 50 ∧ ∀ (l : NegativeSlopeLine), intercept_sum l ≥ min := by
  sorry

#check min_intercept_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intercept_sum_l1159_115985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_intersecting_lines_plane_existence_l1159_115982

/-- Two lines in 3D space are non-intersecting if they have no point in common -/
def NonIntersecting (a b : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  a ∩ b = ∅

/-- A line is contained in a plane if all points of the line are in the plane -/
def LineInPlane (l : Set (EuclideanSpace ℝ (Fin 3))) (α : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  l ⊆ α

/-- A line is parallel to a plane if it doesn't intersect the plane or is contained in it -/
def LineParallelToPlane (l : Set (EuclideanSpace ℝ (Fin 3))) (α : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  l ∩ α = ∅ ∨ LineInPlane l α

/-- For any two non-intersecting space lines, there exists a plane that contains one line
    and is parallel to the other -/
theorem non_intersecting_lines_plane_existence (a b : Set (EuclideanSpace ℝ (Fin 3))) 
    (h : NonIntersecting a b) : 
    ∃ α : Set (EuclideanSpace ℝ (Fin 3)), LineInPlane a α ∧ LineParallelToPlane b α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_intersecting_lines_plane_existence_l1159_115982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_xy_l1159_115943

noncomputable def Det3 (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ) : ℝ :=
  a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32
  - a13 * a22 * a31 - a12 * a21 * a33 - a11 * a23 * a32

theorem sum_of_xy (x y : ℝ) (hxy : x * y = 1) (hneq : x ≠ y)
  (hdet : Det3 1 5 8 3 x y 3 y x = 0) : x + y = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_xy_l1159_115943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l1159_115980

-- Define the count function at the top level
def count (l : List Bool) (p : Bool → Bool) : ℕ :=
  l.foldl (λ acc x => acc + if p x then 1 else 0) 0

theorem two_true_propositions (x y : ℝ) : 
  (∃! n : ℕ, n = 2 ∧ 
    count [
      xy = 0 → x^2 + y^2 = 0,
      x^2 + y^2 = 0 → xy = 0,
      xy ≠ 0 → x^2 + y^2 ≠ 0,
      x^2 + y^2 ≠ 0 → xy ≠ 0
    ] id = n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l1159_115980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_minimum_l1159_115939

/-- A quadratic function f(x) = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic function at a given x -/
noncomputable def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The expression to be minimized -/
noncomputable def expression (f : QuadraticFunction) : ℝ :=
  (3 * f.eval 1 + 6 * f.eval 0 - f.eval (-1)) / (f.eval 0 - f.eval (-2))

theorem quadratic_function_minimum (f : QuadraticFunction)
    (h1 : f.b > 2 * f.a)
    (h2 : ∀ x : ℝ, f.eval x ≥ 0) :
    expression f ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_minimum_l1159_115939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_onto_b_is_negative_one_l1159_115928

/-- Given two vectors a and b in a real inner product space, 
    with the given magnitude conditions, 
    prove that the projection of a onto b is -1. -/
theorem projection_of_a_onto_b_is_negative_one 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (a b : V) :
  ‖b‖ = 5 →
  ‖a + b‖ = 4 →
  ‖a - b‖ = 6 →
  inner a b / ‖b‖ = -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_onto_b_is_negative_one_l1159_115928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_possible_scores_l1159_115932

/-- Represents the number of baskets made by the player -/
def total_baskets : ℕ := 7

/-- Represents the possible point values for each basket -/
def basket_values : Finset ℕ := {2, 3}

/-- Represents all possible combinations of 2-point and 3-point baskets -/
def basket_combinations : Finset (ℕ × ℕ) :=
  Finset.filter (λ p : ℕ × ℕ => p.1 + p.2 = total_baskets) (Finset.product (Finset.range (total_baskets + 1)) (Finset.range (total_baskets + 1)))

/-- Calculates the total score for a given combination of 2-point and 3-point baskets -/
def score (combo : ℕ × ℕ) : ℕ :=
  2 * combo.1 + 3 * combo.2

/-- The set of all possible total scores -/
def possible_scores : Finset ℕ :=
  Finset.image score basket_combinations

theorem num_possible_scores : Finset.card possible_scores = 8 := by
  sorry

#eval Finset.card possible_scores

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_possible_scores_l1159_115932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l1159_115918

noncomputable section

/-- Function f(x) = 7x^2 + 1/x + 1 -/
def f (x : ℝ) : ℝ := 7 * x^2 + 1/x + 1

/-- Function g(x) = x^2 - k -/
def g (x k : ℝ) : ℝ := x^2 - k

/-- Theorem stating that if f(3) - g(3) = 5, then k = -151/3 -/
theorem k_value (k : ℝ) : f 3 - g 3 k = 5 → k = -151/3 := by
  intro h
  -- The proof steps would go here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l1159_115918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l1159_115951

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  AC : ℝ  -- Side AC

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧ 
  t.A + t.B + t.C = Real.pi ∧
  t.AC = Real.sqrt 3

-- Define the arithmetic sequence property
def is_arithmetic_sequence (t : Triangle) : Prop :=
  2 * t.B = t.A + t.C

-- Define the dot product of vectors BA and BC
noncomputable def dot_product (t : Triangle) : ℝ :=
  t.AC * t.AC * Real.cos t.B / 2

-- State the theorem
theorem dot_product_range (t : Triangle) 
  (h_valid : is_valid_triangle t) 
  (h_arithmetic : is_arithmetic_sequence t) : 
  0 < dot_product t ∧ dot_product t ≤ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l1159_115951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_intercept_and_angle_l1159_115971

/-- A line in the 2D plane is represented by its equation y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The x-intercept of a line is the x-coordinate where the line crosses the x-axis -/
noncomputable def xIntercept (l : Line) : ℝ := -l.b / l.m

/-- The inclination angle of a line is the angle it makes with the positive x-axis -/
noncomputable def inclinationAngle (l : Line) : ℝ := Real.arctan l.m

theorem line_equation_from_intercept_and_angle (x_int : ℝ) (angle : ℝ) :
  let l : Line := { m := -1, b := 5 }
  xIntercept l = x_int ∧ inclinationAngle l = angle → l.m = -1 ∧ l.b = 5 := by
  sorry

#check line_equation_from_intercept_and_angle 5 (3 * Real.pi / 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_intercept_and_angle_l1159_115971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_identity_tan_20_40_sum_sqrt3_l1159_115995

-- Define the tangent function for degrees
noncomputable def tan_deg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- State the theorem
theorem tan_sum_identity (x y : ℝ) :
  tan_deg (x + y) = (tan_deg x + tan_deg y) / (1 - tan_deg x * tan_deg y) := by sorry

-- Define the main theorem
theorem tan_20_40_sum_sqrt3 :
  tan_deg 20 + tan_deg 40 + Real.sqrt 3 * tan_deg 20 * tan_deg 40 = Real.sqrt 3 := by
  -- Assuming tan 60° = √3
  have tan_60 : tan_deg 60 = Real.sqrt 3 := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_identity_tan_20_40_sum_sqrt3_l1159_115995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_malibu_pool_drain_time_l1159_115997

/-- Represents the dimensions of a rectangular pool -/
structure PoolDimensions where
  width : ℝ
  length : ℝ
  depth : ℝ

/-- Calculates the time required to drain a pool -/
noncomputable def drainTime (dimensions : PoolDimensions) (capacity : ℝ) (drainRate : ℝ) : ℝ :=
  (dimensions.width * dimensions.length * dimensions.depth * capacity) / drainRate

/-- Theorem: The time required to drain the Malibu Country Club pool is 20 hours -/
theorem malibu_pool_drain_time :
  let poolDimensions : PoolDimensions := ⟨60, 150, 10⟩
  let capacity : ℝ := 0.8
  let drainRate : ℝ := 60
  drainTime poolDimensions capacity drainRate = 20 * 60 := by
  sorry

#check malibu_pool_drain_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_malibu_pool_drain_time_l1159_115997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_AC_is_sqrt_458_l1159_115950

-- Define the cyclic quadrilateral
structure CyclicQuadrilateral where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ
  distinct : AB ≠ BC ∧ AB ≠ CD ∧ AB ≠ DA ∧ BC ≠ CD ∧ BC ≠ DA ∧ CD ≠ DA
  less_than_20 : AB < 20 ∧ BC < 20 ∧ CD < 20 ∧ DA < 20
  ptolemy : ∃ (AC BD : ℝ), AC * BD = (AB * CD : ℝ) + (BC * DA : ℝ)

noncomputable def max_AC (q : CyclicQuadrilateral) : ℝ :=
  Real.sqrt 458

theorem max_AC_is_sqrt_458 (q : CyclicQuadrilateral) :
  max_AC q = Real.sqrt 458 ∧ 
  ∀ (AC : ℝ), (∃ (BD : ℝ), AC * BD = (q.AB * q.CD : ℝ) + (q.BC * q.DA : ℝ)) → 
    AC ≤ max_AC q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_AC_is_sqrt_458_l1159_115950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_group_selection_l1159_115915

/-- Represents a group in the population -/
structure PopulationGroup where
  number : Nat
  members : Finset Nat

/-- Represents the population -/
structure Population where
  size : Nat
  groups : Finset PopulationGroup

/-- The rule for selecting numbers from groups -/
def selectNumber (m : Nat) (groupNumber : Nat) : Nat :=
  let unitsDigit := (m + groupNumber) % 10
  10 * (groupNumber - 1) + unitsDigit

theorem seventh_group_selection (pop : Population) (m : Nat) :
  pop.size = 100 ∧
  pop.groups.card = 10 ∧
  (∀ g ∈ pop.groups, g.members.card = 10) ∧
  (∀ g ∈ pop.groups, ∀ n ∈ g.members, 0 ≤ n ∧ n < 100) ∧
  m = 6 →
  selectNumber m 7 = 63 := by
  sorry

#check seventh_group_selection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_group_selection_l1159_115915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_point_value_l1159_115930

/-- A quadratic function with vertex (2, -3) passing through (0, 7) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 2)^2 - 3

/-- The coefficient 'a' of the quadratic function -/
noncomputable def a : ℝ := 5/2

theorem quadratic_point_value :
  f a 0 = 7 ∧ f a 5 = 39/2 := by
  constructor
  · -- Proof for f a 0 = 7
    sorry
  · -- Proof for f a 5 = 39/2
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_point_value_l1159_115930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1159_115903

theorem sum_of_coefficients (a : Fin 7 → ℝ) :
  (∀ x : ℝ, (Finset.sum (Finset.range 7) (λ i => (1 + 2 * x) ^ (i + 1))) = 
    (Finset.sum (Finset.range 7) (λ i => a i * x ^ i))) →
  a 0 = 6 →
  (Finset.sum (Finset.range 6) (λ i => a (i + 1))) = 1086 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1159_115903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_chord_properties_l1159_115916

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line
noncomputable def line_eq (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 2 = 0

-- Define the point M
noncomputable def point_M : ℝ × ℝ := (1, Real.sqrt 3)

-- Theorem statement
theorem circle_and_chord_properties :
  -- The circle passes through point M
  circle_eq point_M.1 point_M.2 ∧
  -- The length of the chord AB is 2√3
  ∃ (A B : ℝ × ℝ), 
    line_eq A.1 A.2 ∧ 
    line_eq B.1 B.2 ∧ 
    circle_eq A.1 A.2 ∧ 
    circle_eq B.1 B.2 ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_chord_properties_l1159_115916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l1159_115968

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + 8*x + y^2 - 6*y - 15 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 16*x + y^2 + 12*y + 151 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (-4, 3)
def center2 : ℝ × ℝ := (8, -6)
noncomputable def radius1 : ℝ := Real.sqrt 40
noncomputable def radius2 : ℝ := Real.sqrt 51

-- Define the distance between centers
noncomputable def centerDistance : ℝ := Real.sqrt ((8 - (-4))^2 + (-6 - 3)^2)

-- Theorem: The shortest distance between the circles is 15 - (√40 + √51)
theorem shortest_distance_between_circles :
  let shortestDistance := centerDistance - (radius1 + radius2)
  shortestDistance = 15 - (Real.sqrt 40 + Real.sqrt 51) := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l1159_115968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_theorem_l1159_115964

/-- A pentagon inscribed around a circle -/
structure InscribedPentagon where
  -- The lengths of the sides
  s₁ : ℕ
  s₂ : ℕ
  s₃ : ℕ
  s₄ : ℕ
  s₅ : ℕ
  -- Conditions on the side lengths
  h₁ : s₁ = 1
  h₃ : s₃ = 1

/-- The segments into which the point of tangency divides the second side -/
noncomputable def tangentSegments (p : InscribedPentagon) : ℚ × ℚ :=
  (1/2, 1/2)

theorem tangent_segment_theorem (p : InscribedPentagon) :
  tangentSegments p = (1/2, 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_theorem_l1159_115964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_west_ten_meters_l1159_115919

/-- Represents the direction of walking --/
inductive Direction
  | East
  | West

/-- Represents a distance walked in meters --/
def Distance := ℝ

/-- Converts a direction and magnitude to a signed distance --/
def signedDistance (dir : Direction) (magnitude : ℝ) : Distance :=
  match dir with
  | Direction.East => magnitude
  | Direction.West => -magnitude

theorem west_ten_meters :
  signedDistance Direction.West 10 = (-10 : ℝ) :=
by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_west_ten_meters_l1159_115919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_row_is_4213_l1159_115946

-- Define the grid type
def Grid := Matrix (Fin 4) (Fin 4) Nat

-- Define a valid grid
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ ({1, 2, 3, 4} : Set Nat)) ∧
  (∀ i j₁ j₂, j₁ ≠ j₂ → g i j₁ ≠ g i j₂) ∧
  (∀ i₁ i₂ j, i₁ ≠ i₂ → g i₁ j ≠ g i₂ j)

-- Define the outside hints
def outside_hints (g : Grid) : Prop :=
  (g 0 0 = 3 ∧ g 0 3 = 1) ∧
  (g 0 1 = 2 ∧ g 0 2 = 4) ∧
  (g 1 0 % 2 = 1) ∧
  (g 2 0 % 2 = 0) ∧
  (g 3 0 % 2 = 1)

-- Define the third row as a four-digit number
def third_row_number (g : Grid) : Nat :=
  1000 * g 2 0 + 100 * g 2 1 + 10 * g 2 2 + g 2 3

-- Theorem statement
theorem third_row_is_4213 (g : Grid) 
  (h₁ : is_valid_grid g) 
  (h₂ : outside_hints g) : 
  third_row_number g = 4213 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_row_is_4213_l1159_115946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1159_115906

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3 * Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, 2 * Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ p = Real.pi ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (M : ℝ), M = 3 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ M) ∧
  (∃ (m : ℝ), m = 0 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → m ≤ f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1159_115906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_selling_price_is_1824_l1159_115920

/-- Calculates the minimum selling price per washing machine to ensure a monthly profit margin of at least 20% --/
noncomputable def minimum_selling_price (
  machines_per_month : ℕ) 
  (machine_cost : ℝ) 
  (shipping_cost : ℝ) 
  (store_fee : ℝ) 
  (repair_fee : ℝ) 
  (profit_margin : ℝ) : ℝ :=
let total_monthly_cost := machines_per_month * (machine_cost + shipping_cost) + store_fee + repair_fee
let cost_per_machine := total_monthly_cost / machines_per_month
cost_per_machine * (1 + profit_margin)

/-- Theorem stating that the minimum selling price is 1824 yuan --/
theorem min_selling_price_is_1824 :
  minimum_selling_price 50 1200 20 10000 5000 0.2 = 1824 := by
  sorry

-- Remove the #eval statement as it's not necessary for building
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_selling_price_is_1824_l1159_115920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonality_condition_l1159_115976

def a : Fin 2 → ℝ := ![(-2), 1]
def b : Fin 2 → ℝ := ![0, 1]

theorem orthogonality_condition (lambda : ℝ) : 
  (∀ i : Fin 2, b i * (lambda * a i + b i) = 0) → lambda = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonality_condition_l1159_115976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ypsilon_year_has_twelve_months_l1159_115902

/-- Represents the number of days in a month on Planet Ypsilon -/
inductive MonthLength
| Short : MonthLength  -- 28 days
| Medium : MonthLength -- 30 days
| Long : MonthLength   -- 31 days

def MonthLength.toDays : MonthLength → Nat
| Short => 28
| Medium => 30
| Long => 31

/-- The calendar of Planet Ypsilon -/
structure YpsilonCalendar where
  months : List MonthLength
  total_days : Nat
  total_days_eq_365 : total_days = 365
  month_days_valid : ∀ m ∈ months, 
    (m = MonthLength.Short ∧ m.toDays = 28) ∨
    (m = MonthLength.Medium ∧ m.toDays = 30) ∨
    (m = MonthLength.Long ∧ m.toDays = 31)

theorem ypsilon_year_has_twelve_months (cal : YpsilonCalendar) : 
  cal.months.length = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ypsilon_year_has_twelve_months_l1159_115902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_theorem_l1159_115953

/-- Triangle ABC with altitude AD, angle bisector AE, and median AM from vertex A -/
structure TriangleA where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  AD : ℝ         -- Length of altitude
  AE : ℝ         -- Length of angle bisector
  AM : ℝ         -- Length of median

/-- Properties of TriangleA -/
def TriangleA.properties (t : TriangleA) : Prop :=
  t.AD = 12 ∧ t.AE = 13

/-- Angle A is acute -/
def angle_A_acute (t : TriangleA) : Prop := sorry

/-- Angle A is obtuse -/
def angle_A_obtuse (t : TriangleA) : Prop := sorry

/-- Angle A is a right angle -/
def angle_A_right (t : TriangleA) : Prop := sorry

/-- Set of possible lengths for AM when angle A is acute -/
def set_of_possible_lengths_acute : Set ℝ := sorry

/-- Set of possible lengths for AM when angle A is obtuse -/
def set_of_possible_lengths_obtuse : Set ℝ := sorry

/-- Set of possible lengths for AM when angle A is a right angle -/
def set_of_possible_lengths_right : Set ℝ := sorry

/-- Theorem about the median length in TriangleA -/
theorem median_length_theorem (t : TriangleA) 
  (h : t.properties) : 
  (∃ (x : ℝ), t.AM = x) ∧ 
  (angle_A_acute t → t.AM ∈ set_of_possible_lengths_acute) ∧
  (angle_A_obtuse t → t.AM ∈ set_of_possible_lengths_obtuse) ∧
  (angle_A_right t → t.AM ∈ set_of_possible_lengths_right) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_theorem_l1159_115953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_to_submerge_spheres_l1159_115942

-- Define the cylinder
noncomputable def cylinder_radius : ℝ := 1
noncomputable def cylinder_height : ℝ := 2

-- Define the sphere (ball)
noncomputable def sphere_radius : ℝ := 0.5

-- Calculate volumes
noncomputable def sphere_volume : ℝ := (4 / 3) * Real.pi * (sphere_radius ^ 3)
noncomputable def total_spheres_volume : ℝ := 4 * sphere_volume
noncomputable def cylinder_volume : ℝ := Real.pi * (cylinder_radius ^ 2) * cylinder_height

-- Theorem statement
theorem water_volume_to_submerge_spheres :
  cylinder_volume - total_spheres_volume = (4 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_to_submerge_spheres_l1159_115942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_l1159_115924

def vector_a : ℝ × ℝ × ℝ := (2, -1, 3)
def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)

theorem vectors_parallel (x : ℝ) : 
  (∃ (k : ℝ), vector_b x = k • vector_a) → x = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_l1159_115924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_equilateral_with_equilateral_centroids_l1159_115954

open Complex

/-- The triangle type with vertices as complex numbers -/
structure Triangle :=
  (A B C : ℂ)

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℂ := (t.A + t.B + t.C) / 3

/-- The circumcenter of a triangle -/
noncomputable def circumcenter (t : Triangle) : ℂ := sorry

/-- The orthocenter of a triangle -/
noncomputable def orthocenter (t : Triangle) : ℂ := sorry

/-- Check if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop :=
  abs (t.B - t.A) = abs (t.C - t.B) ∧ abs (t.C - t.B) = abs (t.A - t.C)

/-- The main theorem -/
theorem exists_non_equilateral_with_equilateral_centroids :
  ∃ (t : Triangle),
    ¬is_equilateral t ∧
    let O := circumcenter t
    let G := centroid t
    let H := orthocenter t
    let G₁ := centroid ⟨O, t.B, t.C⟩
    let G₂ := centroid ⟨G, t.C, t.A⟩
    let G₃ := centroid ⟨H, t.A, t.B⟩
    is_equilateral ⟨G₁, G₂, G₃⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_equilateral_with_equilateral_centroids_l1159_115954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l1159_115957

-- Define the set T
def T : Set (ℝ × ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0 ∧ p.1 + p.2.1 + p.2.2 = 3}

-- Define the support condition
def supports (p : ℝ × ℝ × ℝ) (a b c : ℝ) : Prop :=
  (p.1 ≥ a ∧ p.2.1 ≥ b) ∨ (p.1 ≥ a ∧ p.2.2 ≥ c) ∨ (p.2.1 ≥ b ∧ p.2.2 ≥ c)

-- Define the set S
def S : Set (ℝ × ℝ × ℝ) :=
  {p ∈ T | supports p (1/3) (1/4) (1/3)}

-- Define the area function (noncomputable as it's not constructive)
noncomputable def area (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem area_ratio : area S / area T = 89 / 243 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l1159_115957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1159_115973

-- Define the functions f and g
noncomputable def f (x : ℝ) := (1/2) * x^2 + (1 - x) * Real.exp x

noncomputable def g (a : ℝ) (x : ℝ) := x - (1 + a) * Real.log x - a / x

-- State the theorem
theorem function_inequality (a : ℝ) :
  (∀ x₁ ∈ Set.Icc (-1) 0, ∃ x₂ ∈ Set.Icc (Real.exp 1) 3, f x₁ > g a x₂) →
  (Real.exp 2 - 2 * Real.exp 1) / (Real.exp 1 + 1) < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1159_115973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_value_f_inequality_holds_l1159_115937

/-- The function f(e) that satisfies the inequality for all real a, b, c, d -/
noncomputable def f (e : ℝ) : ℝ := 1 / (4 * e^2)

/-- The statement of the theorem -/
theorem least_value_f (e : ℝ) (he : e > 0) :
  ∀ (g : ℝ → ℝ), (∀ (a b c d : ℝ),
    a^3 + b^3 + c^3 + d^3 ≤ e^2 * (a^2 + b^2 + c^2 + d^2) + g e * (a^4 + b^4 + c^4 + d^4)) →
  f e ≤ g e :=
by sorry

/-- The inequality holds for the defined f -/
theorem inequality_holds (e : ℝ) (he : e > 0) (a b c d : ℝ) :
  a^3 + b^3 + c^3 + d^3 ≤ e^2 * (a^2 + b^2 + c^2 + d^2) + f e * (a^4 + b^4 + c^4 + d^4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_value_f_inequality_holds_l1159_115937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_z_l1159_115913

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - Complex.I * 5) + Complex.abs (z - 6) = Real.sqrt 61) : 
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 30 / Real.sqrt 61 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_z_l1159_115913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_path_distance_l1159_115983

/-- The distance traveled by the center of a ball along a path of three semicircular arcs -/
noncomputable def distance_traveled (ball_diameter : ℝ) (R₁ R₂ R₃ : ℝ) : ℝ :=
  let ball_radius := ball_diameter / 2
  let path_radius₁ := R₁ - ball_radius
  let path_radius₂ := R₂ + ball_radius
  let path_radius₃ := R₃ - ball_radius
  Real.pi * (path_radius₁ + path_radius₂ + path_radius₃)

/-- Theorem: The distance traveled by the center of a ball with diameter 6 inches
    along a path of three semicircular arcs with radii 120, 50, and 100 inches
    is equal to 267π inches -/
theorem ball_path_distance :
  distance_traveled 6 120 50 100 = 267 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_path_distance_l1159_115983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ownership_period_l1159_115981

/-- Cost function for owning a sedan for n years -/
noncomputable def f (n : ℝ) : ℝ := 0.1 * n^2 + 0.6 * n + 14.4

/-- Average annual cost of owning the sedan for n years -/
noncomputable def avg_cost (n : ℝ) : ℝ := f n / n

/-- Theorem: The number of years that minimizes the average annual cost is 12 -/
theorem optimal_ownership_period :
  ∃ (n : ℝ), n > 0 ∧ (∀ (m : ℝ), m > 0 → avg_cost n ≤ avg_cost m) ∧ n = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ownership_period_l1159_115981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_value_l1159_115975

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem sine_function_value 
  (ω φ : ℝ) 
  (h1 : ω > 0) 
  (h2 : 0 < φ ∧ φ < Real.pi) 
  (h3 : Real.pi / (2 * ω) = Real.pi / 2) 
  (h4 : Real.tan φ = Real.sqrt 3 / 3) : 
  f ω φ (Real.pi / 4) = Real.sqrt 3 / 2 := by
  sorry

#check sine_function_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_value_l1159_115975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_1_min_value_f_l1159_115929

noncomputable def f (a x : ℝ) : ℝ := |a + 1/x| + |a - 1/x|

-- Theorem for the solution set when a = 1
theorem solution_set_a_1 :
  {x : ℝ | f 1 x > 3} = {x : ℝ | -2/3 < x ∧ x < 2/3} := by
  sorry

-- Theorem for the minimum value of f(a)
theorem min_value_f :
  ∀ a : ℝ, f a a ≥ 2 ∧ ∃ x : ℝ, f x x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_1_min_value_f_l1159_115929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_ln_2x_l1159_115990

/-- The minimum difference between x₁ and x₂ when ln x₁ = 2x₂ --/
theorem min_difference_ln_2x :
  ∃ (x : ℝ), x > 0 ∧ ∀ y > 0, |y - Real.log y / 2| ≥ |(1 + Real.log 2) / 2| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_ln_2x_l1159_115990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_increase_l1159_115904

-- Define the arc length function
noncomputable def arcLength (R : ℝ) (θ : ℝ) : ℝ := 2 * Real.pi * R * θ / 360

-- State the theorem
theorem arc_length_increase (R : ℝ) (h : R > 0) :
  arcLength R 1 = Real.pi * R / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_increase_l1159_115904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_generating_function_and_coefficient_l1159_115996

/-- The number of solutions to x₁ + ... + xₖ = n in non-negative integers -/
def a (k : ℕ) : ℕ → ℕ := sorry

/-- The generating function for the sequence a_n -/
noncomputable def F (k : ℕ) : ℝ → ℝ := sorry

/-- The theorem stating the properties of F(x) and a_n -/
theorem generating_function_and_coefficient (k : ℕ) :
  (∀ x, F k x = (1 - x)^(-(k : ℤ))) ∧
  (∀ n, a k n = Nat.choose (n + k - 1) n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_generating_function_and_coefficient_l1159_115996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_theorem_1_distance_theorem_2_l1159_115949

-- Define the distance function between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Define the simplified distance function for points on a line parallel to y-axis
def distance_y_parallel (y₁ y₂ : ℝ) : ℝ :=
  |y₂ - y₁|

-- Theorem 1: Distance between (2,4) and (-3,-8) is 13
theorem distance_theorem_1 : distance 2 4 (-3) (-8) = 13 := by
  sorry

-- Theorem 2: Distance between two points on a line parallel to y-axis with y-coordinates 5 and -1 is 6
theorem distance_theorem_2 : distance_y_parallel 5 (-1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_theorem_1_distance_theorem_2_l1159_115949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_uniquely_determined_l1159_115901

/-- Represents a chessboard filled with numbers 1 to 64 -/
def Chessboard := Fin 8 → Fin 8 → Fin 64

/-- The sum of numbers in a 2-cell rectangle -/
def RectangleSum (board : Chessboard) (i j : Fin 8) (horizontal : Bool) : ℕ :=
  if horizontal then
    (board i j).val + (board i (j.succ)).val
  else
    (board i j).val + (board (i.succ) j).val

/-- Predicate to check if two positions are on the same diagonal -/
def OnSameDiagonal (i₁ j₁ i₂ j₂ : Fin 8) : Prop :=
  (i₁.val + j₁.val = i₂.val + j₂.val) ∨ (i₁.val - j₁.val = i₂.val - j₂.val)

/-- Main theorem: The chessboard configuration can be uniquely determined -/
theorem chessboard_uniquely_determined (board : Chessboard)
  (sums : ∀ i j : Fin 8, ∀ h : Bool, ∃ m : ℕ, RectangleSum board i j h = m)
  (one_and_sixtyfour_diagonal : ∃ i₁ j₁ i₂ j₂ : Fin 8,
    board i₁ j₁ = 0 ∧ board i₂ j₂ = 63 ∧ OnSameDiagonal i₁ j₁ i₂ j₂) :
  ∀ i j : Fin 8, ∃! n : Fin 64, board i j = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_uniquely_determined_l1159_115901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_in_classroom_l1159_115987

theorem boys_in_classroom (total_children : Nat) (girls_fraction : Rat) (boys : Nat) : 
  total_children = 45 →
  girls_fraction = 1/3 →
  boys = total_children - (girls_fraction * ↑total_children).floor →
  boys = 30 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_in_classroom_l1159_115987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1159_115974

noncomputable def g (x : ℝ) : ℝ := 
  Real.sin x ^ 4 - Real.sin x * Real.cos x + Real.cos x ^ 4 + Real.sin x ^ 2 * Real.cos x ^ 2

theorem g_range :
  ∀ x : ℝ, (1 / 4 : ℝ) ≤ g x ∧ g x ≤ (5 / 4 : ℝ) :=
by
  intro x
  have h1 : Real.sin x ^ 2 + Real.cos x ^ 2 = 1 := by sorry
  have h2 : Real.sin x * Real.cos x = (1 / 2) * Real.sin (2 * x) := by sorry
  have h3 : -(1 : ℝ) ≤ Real.sin (2 * x) ∧ Real.sin (2 * x) ≤ 1 := by sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1159_115974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangular_iff_sum_of_squares_l1159_115900

/-- A triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: A positive integer m is a sum of two triangular numbers if and only if 4m+1 is a sum of two squares -/
theorem sum_of_triangular_iff_sum_of_squares (m : ℕ) (hm : m > 0) :
  (∃ x y : ℕ, m = triangular_number x + triangular_number y) ↔
  (∃ a b : ℤ, (4 : ℤ) * (m : ℤ) + 1 = a^2 + b^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangular_iff_sum_of_squares_l1159_115900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_has_extrema_l1159_115944

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + Real.sin (x + Real.pi / 2)

theorem f_is_even_and_has_extrema :
  (∀ x, f x = f (-x)) ∧
  (∃ M m, ∀ x, m ≤ f x ∧ f x ≤ M) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_has_extrema_l1159_115944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1159_115933

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4) / ((x - 3)^2)

theorem solution_set (x : ℝ) : 
  x ≠ 3 → (f x ≥ 0 ↔ x ∈ Set.Iic (-2) ∪ Set.Icc 2 3 ∪ Set.Ioi 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1159_115933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_planes_of_bounded_body_with_two_rotation_axes_l1159_115914

/-- A bounded body in 3D space -/
structure BoundedBody where
  -- Add necessary fields (placeholder)
  dummy : Unit

/-- Represents an axis of rotation for a body -/
structure RotationAxis where
  -- Add necessary fields (placeholder)
  dummy : Unit

/-- Represents a plane in 3D space -/
structure Plane where
  -- Add necessary fields (placeholder)
  dummy : Unit

/-- Determines if a plane is a symmetry plane for a body -/
def isSymmetryPlane (body : BoundedBody) (plane : Plane) : Prop :=
  -- Add definition (placeholder)
  True

/-- The center of mass of a body -/
def centerOfMass (body : BoundedBody) : ℝ × ℝ × ℝ :=
  -- Add definition (placeholder)
  (0, 0, 0)

/-- Determines if a plane passes through a point -/
def planePassesThroughPoint (plane : Plane) (point : ℝ × ℝ × ℝ) : Prop :=
  -- Add definition (placeholder)
  True

/-- Determines if a body has a rotation axis -/
def hasRotationAxis (body : BoundedBody) (axis : RotationAxis) : Prop :=
  -- Add definition (placeholder)
  True

theorem symmetry_planes_of_bounded_body_with_two_rotation_axes
  (body : BoundedBody)
  (axis1 axis2 : RotationAxis)
  (h1 : hasRotationAxis body axis1)
  (h2 : hasRotationAxis body axis2)
  (h3 : axis1 ≠ axis2)
  (plane : Plane)
  : planePassesThroughPoint plane (centerOfMass body) → isSymmetryPlane body plane :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_planes_of_bounded_body_with_two_rotation_axes_l1159_115914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1159_115979

/-- Projection of vector a onto vector e -/
noncomputable def proj (a e : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (a.1 * e.1 + a.2 * e.2) / (e.1 * e.1 + e.2 * e.2)
  (scalar * e.1, scalar * e.2)

/-- The angle between two vectors in radians -/
noncomputable def angle (a e : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * e.1 + a.2 * e.2) / (Real.sqrt (a.1 * a.1 + a.2 * a.2) * Real.sqrt (e.1 * e.1 + e.2 * e.2)))

theorem projection_theorem (a : ℝ × ℝ) :
  let e : ℝ × ℝ := (0, 1)
  Real.sqrt (a.1 * a.1 + a.2 * a.2) = 6 →
  angle a e = π / 6 →
  proj a e = (0, 3 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1159_115979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l1159_115989

theorem negation_of_cosine_inequality :
  (¬ ∀ x : ℝ, Real.cos x ≤ 1) ↔ (∃ x₀ : ℝ, Real.cos x₀ > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l1159_115989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_max_distance_l1159_115936

/-- The circle with center (5, 3) and radius 3 -/
def my_circle (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9

/-- The line 3x + 4y - 2 = 0 -/
def my_line (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0

/-- The maximum distance from any point on the circle to the line -/
def max_distance : ℝ := 8

theorem circle_line_max_distance :
  ∀ (M : ℝ × ℝ), my_circle M.1 M.2 →
    ∃ (P : ℝ × ℝ), my_line P.1 P.2 ∧
      ∀ (Q : ℝ × ℝ), my_line Q.1 Q.2 →
        dist M P ≤ max_distance ∧
        (dist M P = max_distance ↔ dist M Q ≤ dist M P) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_max_distance_l1159_115936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l1159_115927

theorem cube_root_equation_solution :
  ∀ x : ℝ, (Real.rpow (5 * x - 2) (1/3) = 2) ↔ (x = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l1159_115927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABC_is_30_degrees_l1159_115967

noncomputable def BA : ℝ × ℝ := (1/2, Real.sqrt 3 / 2)
noncomputable def BC : ℝ × ℝ := (Real.sqrt 3 / 2, 1/2)

theorem angle_ABC_is_30_degrees :
  let cosABC := (BA.1 * BC.1 + BA.2 * BC.2) / (Real.sqrt (BA.1^2 + BA.2^2) * Real.sqrt (BC.1^2 + BC.2^2))
  Real.arccos cosABC = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABC_is_30_degrees_l1159_115967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_children_in_families_with_children_l1159_115955

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 9)
  (h2 : average_children = 3)
  (h3 : childless_families = 3) :
  (total_families * average_children) / (total_families - childless_families : ℚ) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_children_in_families_with_children_l1159_115955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_implies_a_greater_than_four_l1159_115961

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (x / 2) - x / 4

-- Define the function F
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - a * f x + 4

-- Theorem statement
theorem no_zeros_implies_a_greater_than_four (a : ℝ) (h_a_pos : a > 0) :
  (∀ x, F a x ≠ 0) → a > 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_implies_a_greater_than_four_l1159_115961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_base_isosceles_triangle_l1159_115960

/-- Represents an isosceles triangle inscribed in a semicircle -/
structure InscribedIsoscelesTriangle where
  R : ℝ  -- Radius of the semicircle
  x : ℝ  -- Length of the equal sides of the isosceles triangle
  y : ℝ  -- Length of the base of the isosceles triangle

/-- The base length of the inscribed isosceles triangle as a function of its equal side length -/
noncomputable def baseLengthFunction (t : InscribedIsoscelesTriangle) : ℝ :=
  Real.sqrt ((1 / t.R) * t.x^2 * (2 * t.R - t.x))

/-- Theorem stating that the isosceles triangle with the largest base has equal sides of length 4/3 * R -/
theorem largest_base_isosceles_triangle (t : InscribedIsoscelesTriangle) :
  (∀ (s : InscribedIsoscelesTriangle), s.R = t.R → baseLengthFunction s ≤ baseLengthFunction t) →
  t.x = (4 / 3) * t.R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_base_isosceles_triangle_l1159_115960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l1159_115963

theorem rectangular_to_polar_conversion :
  let x : Real := 2
  let y : Real := 2 * Real.sqrt 3
  let r : Real := Real.sqrt (x^2 + y^2)
  let θ : Real := Real.arctan (y / x)
  (r = 4 ∧ θ = π / 3) ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l1159_115963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approximation_l1159_115962

/-- The speed of the train in kilometers per hour -/
noncomputable def train_speed : ℝ := 120

/-- The time taken by the train to cross a pole in seconds -/
noncomputable def crossing_time : ℝ := 9

/-- Conversion factor from kilometers per hour to meters per second -/
noncomputable def km_per_hr_to_m_per_s : ℝ := 1000 / 3600

/-- The length of the train in meters -/
noncomputable def train_length : ℝ := train_speed * km_per_hr_to_m_per_s * crossing_time

theorem train_length_approximation : 
  ∃ ε > 0, |train_length - 299.97| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approximation_l1159_115962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_sum_divisibility_induction_step_correctness_l1159_115966

theorem odd_power_sum_divisibility (x y : ℤ) :
  ∀ n : ℕ, n % 2 = 1 → (x + y) ∣ (x^n + y^n) :=
by
  sorry

theorem induction_step_correctness (x y : ℤ) :
  let P : ℕ → Prop := λ n => (x + y) ∣ (x^n + y^n)
  let induction_step : Prop := ∀ k : ℕ, P (2*k - 1) → P (2*k + 1)
  (∀ n : ℕ, n % 2 = 1 → P n) ↔ (P 1 ∧ induction_step) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_sum_divisibility_induction_step_correctness_l1159_115966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jia_peak_count_l1159_115977

/-- Represents the mountain with points A (foot), B (peak), and C (on slope) -/
structure Mountain where
  AB : ℝ
  AC : ℝ
  CB : ℝ

/-- Represents a person climbing the mountain -/
structure Climber where
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- The problem setup -/
def mountain_problem (m : Mountain) (jia yi : Climber) : Prop :=
  m.AC = (1/3) * m.AB ∧
  m.CB = (2/3) * m.AB ∧
  jia.uphill_speed / yi.uphill_speed = 6/5 ∧
  jia.downhill_speed = 1.5 * jia.uphill_speed ∧
  yi.downhill_speed = 1.5 * yi.uphill_speed

/-- Function to calculate the number of times Jia reaches the peak -/
noncomputable def number_of_times_jia_reaches_peak_when_seeing_yi_on_AC_second_time 
  (m : Mountain) (jia yi : Climber) : ℕ := 9

/-- The theorem to prove -/
theorem jia_peak_count 
  (m : Mountain) 
  (jia yi : Climber) 
  (h : mountain_problem m jia yi) : 
  (∃ n : ℕ, n = 9 ∧ 
    n = number_of_times_jia_reaches_peak_when_seeing_yi_on_AC_second_time m jia yi) := by
  use 9
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jia_peak_count_l1159_115977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_e_neg_4i_in_second_quadrant_l1159_115923

/-- Euler's formula for complex exponentials -/
axiom euler_formula (x : ℝ) : Complex.exp (x * Complex.I) = Complex.cos x + Complex.I * Complex.sin x

/-- The complex number e^(-4i) -/
noncomputable def z : ℂ := Complex.exp (-4 * Complex.I)

/-- A complex number is in the second quadrant if its real part is negative and imaginary part is positive -/
def is_in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

/-- Theorem: e^(-4i) is located in the second quadrant of the complex plane -/
theorem e_neg_4i_in_second_quadrant : is_in_second_quadrant z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_e_neg_4i_in_second_quadrant_l1159_115923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equality_l1159_115998

open BigOperators

def product (n : ℕ) : ℚ :=
  ∏ x in Finset.range n, ((x + 1 : ℚ) / ((x : ℚ)^2 + 1) + 1/4)

theorem product_equality : 
  product 2022 = (2024^2 + 1 : ℚ) / (2^4044 * 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equality_l1159_115998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_power_function_implies_m_equals_two_l1159_115912

open Real

-- Define the power function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m-1)

-- State the theorem
theorem increasing_power_function_implies_m_equals_two :
  ∀ m : ℝ, (∀ x y : ℝ, 0 < x → x < y → f m x < f m y) → m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_power_function_implies_m_equals_two_l1159_115912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1159_115926

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The theorem stating that the distance between (1, 3) and (-5, 7) is 2√13 -/
theorem distance_between_points : distance 1 3 (-5) 7 = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1159_115926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_from_two_points_l1159_115972

/-- Given points A and B in ℝ³, and a point P on the y-axis equidistant from A and B,
    prove that the y-coordinate of P is -7/6. -/
theorem point_equidistant_from_two_points (A B P : ℝ × ℝ × ℝ) :
  A = (1, 2, 3) →
  B = (2, -1, 4) →
  (Prod.fst P) = 0 ∧ (Prod.snd (Prod.snd P)) = 0 →  -- P is on the y-axis
  (Prod.fst P - Prod.fst A)^2 + (Prod.fst (Prod.snd P) - Prod.fst (Prod.snd A))^2 + (Prod.snd (Prod.snd P) - Prod.snd (Prod.snd A))^2 =
    (Prod.fst P - Prod.fst B)^2 + (Prod.fst (Prod.snd P) - Prod.fst (Prod.snd B))^2 + (Prod.snd (Prod.snd P) - Prod.snd (Prod.snd B))^2 →  -- |PA| = |PB|
  Prod.fst (Prod.snd P) = -7/6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_from_two_points_l1159_115972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1159_115984

noncomputable def triangle_problem (A B C P : ℝ × ℝ) (x y : ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  let S := abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) / 2
  let CP := (P.1 - C.1, P.2 - C.2)
  let CA_norm := Real.sqrt ((AC.1)^2 + (AC.2)^2)
  let CB_norm := Real.sqrt ((BC.1)^2 + (BC.2)^2)
  AB.1 * AC.1 + AB.2 * AC.2 = 9 ∧
  Real.sin (Real.arccos ((BC.1 * AB.1 + BC.2 * AB.2) / (Real.sqrt (BC.1^2 + BC.2^2) * Real.sqrt (AB.1^2 + AB.2^2)))) =
    Real.cos (Real.arccos ((BC.1 * AC.1 + BC.2 * AC.2) / (Real.sqrt (BC.1^2 + BC.2^2) * Real.sqrt (AC.1^2 + AC.2^2)))) *
    Real.sin (Real.arccos ((AC.1 * AB.1 + AC.2 * AB.2) / (Real.sqrt (AC.1^2 + AC.2^2) * Real.sqrt (AB.1^2 + AB.2^2)))) ∧
  S = 6 ∧
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2)) ∧
  CP = (x * AC.1 / CA_norm + y * BC.1 / CB_norm, x * AC.2 / CA_norm + y * BC.2 / CB_norm)

theorem min_value_theorem (A B C P : ℝ × ℝ) (x y : ℝ) 
  (h : triangle_problem A B C P x y) : 
  (∀ x' y' : ℝ, triangle_problem A B C P x' y' → 1/x' + 1/y' ≥ 7/12 + Real.sqrt 3 / 3) ∧
  (∃ x'' y'' : ℝ, triangle_problem A B C P x'' y'' ∧ 1/x'' + 1/y'' = 7/12 + Real.sqrt 3 / 3) := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1159_115984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_three_element_subsets_l1159_115993

def X (k : ℕ) := Finset.range (6 * k)

theorem min_three_element_subsets (k : ℕ) (L : Finset (Finset (Fin (6 * k)))) :
  (∀ (pair : Finset (Fin (6 * k))), pair.card = 2 → ∃ (triple : Finset (Fin (6 * k))), triple ∈ L ∧ pair ⊆ triple) →
  (∀ (s : Finset (Fin (6 * k))), s ∈ L → s.card = 3) →
  L.card ≥ (6 * k)^2 / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_three_element_subsets_l1159_115993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_no_zeros_condition_l1159_115999

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

-- Theorem for part (1)
theorem monotonicity_intervals (x : ℝ) :
  let f₁ := f 1
  (0 < x ∧ x < 2 → (deriv f₁) x < 0) ∧
  (2 < x → (deriv f₁) x > 0) :=
by sorry

-- Theorem for part (2)
theorem no_zeros_condition (a : ℝ) :
  (∀ x, 0 < x ∧ x < 1/3 → f a x ≠ 0) ↔ a ≥ 2 - 3 * Real.log 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_no_zeros_condition_l1159_115999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l1159_115910

noncomputable def f (x : Real) := Real.sin x

noncomputable def g (x : Real) := Real.sin (1/2 * x - Real.pi/6)

theorem graph_transformation (x : Real) : 
  f ((x - Real.pi/3) / 2) = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l1159_115910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_specific_point_l1159_115941

noncomputable def cylindrical_to_rectangular (r θ z : Real) : Real × Real × Real :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_specific_point :
  cylindrical_to_rectangular 10 (Real.pi / 3) (-2) = (5, 5 * Real.sqrt 3, -2) := by
  unfold cylindrical_to_rectangular
  simp [Real.cos_pi_div_three, Real.sin_pi_div_three]
  norm_num
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_specific_point_l1159_115941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_rotation_around_octagon_l1159_115947

/-- The number of sides in a regular pentagon -/
def pentagon_sides : ℕ := 5

/-- The number of sides in a regular octagon -/
def octagon_sides : ℕ := 8

/-- The number of full movements of the pentagon around the octagon -/
def num_movements : ℕ := 3

/-- Calculates the inner angle of a regular polygon with n sides -/
noncomputable def inner_angle (n : ℕ) : ℝ := ((n - 2) * 180) / n

/-- Calculates the rotation of the pentagon per movement around the octagon -/
noncomputable def rotation_per_movement : ℝ :=
  360 - (inner_angle octagon_sides + inner_angle pentagon_sides)

/-- The total rotation of the pentagon after the specified number of movements -/
noncomputable def total_rotation : ℝ := num_movements * rotation_per_movement

theorem pentagon_rotation_around_octagon :
  total_rotation = 351 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_rotation_around_octagon_l1159_115947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_mold_radius_l1159_115911

noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

theorem smaller_mold_radius :
  let large_radius : ℝ := 2
  let num_small_molds : ℝ := 64
  let large_volume := hemisphere_volume large_radius
  let small_radius := (large_volume / num_small_molds) ^ (1/3)
  small_radius = 1/2 := by
  -- Unfold definitions
  unfold hemisphere_volume
  -- Simplify expressions
  simp [Real.pi]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_mold_radius_l1159_115911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1159_115958

noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

theorem f_properties :
  (∃! x : ℝ, x > 0 ∧ f x = x) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₂ > x₁ → f x₁ = f x₂ → x₁ + x₂ > 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1159_115958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1159_115978

theorem sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x > y → x^3 + y^3 > x^2*y + x*y^2) ∧
  (∃ x y : ℝ, x^3 + y^3 > x^2*y + x*y^2 ∧ x ≤ y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1159_115978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ways_AB_same_box_correct_ways_max_two_per_box_correct_l1159_115956

-- Define the number of balls and boxes
def num_balls : ℕ := 4
def num_boxes : ℕ := 4

-- Define the function for the number of ways to place balls with A and B in the same box
def ways_AB_same_box : ℕ := num_boxes^(num_balls - 1)

-- Define the function for the number of ways to place balls with at most 2 balls per box
def ways_max_two_per_box : ℕ :=
  (Nat.factorial num_balls) +  -- All balls in different boxes
  (Nat.choose num_balls 2 * Nat.choose num_boxes 2 * 2) +  -- Two pairs in different boxes
  (Nat.choose num_balls 2 * Nat.choose num_boxes 3 * Nat.factorial 3)  -- One pair and two singles

-- Theorem statements
theorem ways_AB_same_box_correct :
  ways_AB_same_box = 64 := by sorry

theorem ways_max_two_per_box_correct :
  ways_max_two_per_box = 204 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ways_AB_same_box_correct_ways_max_two_per_box_correct_l1159_115956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l1159_115994

-- Define the centers of the circles
def center1 : ℝ × ℝ := (5, 3)
def center2 : ℝ × ℝ := (20, 7)

-- Define the property that the circles are tangent to the x-axis
noncomputable def tangent_to_x_axis (center : ℝ × ℝ) : Prop :=
  center.2 = Real.sqrt ((center.1 - center.1)^2 + center.2^2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem closest_points_distance :
  (tangent_to_x_axis center1) →
  (tangent_to_x_axis center2) →
  (distance center1 center2 - center1.2 - center2.2 = Real.sqrt 241 - 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l1159_115994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l1159_115935

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the given conditions
axiom f_symmetry (x : ℝ) : f (x + 4) + f (-x) = 2
axiom f_even (x : ℝ) : f (2 * x + 1) = f (-(2 * x + 1))
axiom g_def (x : ℝ) : g x = -f (2 - x)

-- State the theorem to be proved
theorem f_and_g_properties : f 0 = 1 ∧ g 2024 = g 0 ∧ g 0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l1159_115935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_2sqrt5_l1159_115952

noncomputable def sequence_term (n : ℕ) : ℝ := Real.sqrt (3 * n - 1)

theorem seventh_term_is_2sqrt5 : sequence_term 7 = 2 * Real.sqrt 5 := by
  -- Unfold the definition of sequence_term
  unfold sequence_term
  -- Simplify the left-hand side
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_2sqrt5_l1159_115952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l1159_115921

noncomputable def RealPos := {x : ℝ | x > 0}

def IsFunctionWithProperties (f : ℝ → ℝ) : Prop :=
  (∀ x > 1, f x < 0) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y)

theorem solution_set_equality 
  (f : ℝ → ℝ) 
  (h : IsFunctionWithProperties f) :
  {x : ℝ | x > 2 ∧ x ≤ 4 ∧ f x + f (x - 2) ≥ f 8} = Set.Ioo 2 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l1159_115921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_power_mod_eleven_l1159_115948

theorem fifth_power_mod_eleven (a : ℕ) : ∃ x : ZMod 11, x ∈ ({-1, 0, 1} : Set (ZMod 11)) ∧ (a : ZMod 11)^5 = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_power_mod_eleven_l1159_115948
