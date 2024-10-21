import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_sign_distance_l1129_112911

/-- Represents a town in our problem. -/
structure Town where
  name : String

/-- Represents a path between two towns. -/
structure PathBetween (A B : Town) where
  distance : ℝ

/-- The shortest path property -/
def is_shortest_path (A B C : Town) (path_AB : PathBetween A B) (path_BC : PathBetween B C) (path_AC : PathBetween A C) :=
  path_AC.distance = path_AB.distance + path_BC.distance

/-- Main theorem: Given the conditions, prove the distance from Betown to the point is 2 km -/
theorem broken_sign_distance 
  (Atown Betown Cetown : Town)
  (path_AB : PathBetween Atown Betown)
  (path_BC : PathBetween Betown Cetown)
  (path_AC : PathBetween Atown Cetown)
  (h_shortest : is_shortest_path Atown Betown Cetown path_AB path_BC path_AC)
  (h_AB_distance : path_AB.distance = 4)
  (h_AP_distance : (PathBetween.mk 6 : PathBetween Atown (Town.mk "Point")).distance = 6) :
  (PathBetween.mk 2 : PathBetween Betown (Town.mk "Point")).distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_sign_distance_l1129_112911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_equidistant_equidistant_point_is_circumcenter_l1129_112931

/-- A triangle in a 2D plane -/
structure Triangle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] where
  A : V
  B : V
  C : V

/-- The perpendicular bisector of a line segment -/
def perpBisector {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (P Q : V) : Set V :=
  {X : V | ‖X - P‖ = ‖X - Q‖}

/-- The intersection point of the three perpendicular bisectors of a triangle -/
noncomputable def circumcenter {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (t : Triangle V) : V :=
  sorry

/-- The theorem stating that the circumcenter is equidistant from all vertices -/
theorem circumcenter_equidistant {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (t : Triangle V) :
  let O := circumcenter t
  ‖O - t.A‖ = ‖O - t.B‖ ∧ ‖O - t.B‖ = ‖O - t.C‖ := by
  sorry

/-- The theorem stating that any point equidistant from all vertices is the circumcenter -/
theorem equidistant_point_is_circumcenter {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (t : Triangle V) (P : V) :
  (‖P - t.A‖ = ‖P - t.B‖ ∧ ‖P - t.B‖ = ‖P - t.C‖) → P = circumcenter t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_equidistant_equidistant_point_is_circumcenter_l1129_112931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_value_l1129_112924

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_slope {a b c d e f : ℝ} :
  (∀ x y : ℝ, a * x + b * y + e = 0 ↔ c * x + d * y + f = 0) → a / c = b / d

/-- Given two parallel lines l₁: ax - 2y - 1 = 0 and l₂: 6x - 4y + 1 = 0, prove that a = 3 -/
theorem parallel_lines_value (a : ℝ) :
  (∀ x y : ℝ, a * x - 2 * y - 1 = 0 ↔ 6 * x - 4 * y + 1 = 0) → a = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_value_l1129_112924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_cartesian_equation_l1129_112907

open Real

-- Define the curve C
noncomputable def C (θ : ℝ) : ℝ × ℝ := (cos θ, cos (2 * θ) + 2)

-- State the theorem
theorem curve_C_cartesian_equation :
  ∀ x y : ℝ, (∃ θ : ℝ, C θ = (x, y)) ↔ y = 2 * x^2 + 1 ∧ -1 ≤ x ∧ x ≤ 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_cartesian_equation_l1129_112907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1129_112993

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

-- Theorem statement
theorem hyperbola_eccentricity :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ x y, hyperbola_equation x y ↔ x^2/a^2 - y^2/b^2 = 1) ∧
  eccentricity a b = 2 := by
  -- Proof goes here
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1129_112993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koch_snowflake_area_l1129_112979

/-- Represents the area of the shape enclosed by the nth curve in the sequence -/
noncomputable def S (n : ℕ) : ℝ :=
  8/5 - 3/5 * (4/9)^n

/-- The limit of S(n) as n approaches infinity -/
noncomputable def S_limit : ℝ := 8/5

/-- Theorem stating the properties of the Koch snowflake area sequence -/
theorem koch_snowflake_area (n : ℕ) :
  /- Initial condition: S₀ = 1 -/
  S 0 = 1 ∧
  /- Recursive relation: Sₖ₊₁ = Sₖ + 3 * 4ᵏ * (1/3)^(2k+2) -/
  (∀ k, S (k + 1) = S k + 3 * 4^k * (1/3)^(2*k + 2)) ∧
  /- Limit of S(n) as n approaches infinity -/
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |S n - S_limit| < ε :=
by sorry

#check koch_snowflake_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_koch_snowflake_area_l1129_112979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l1129_112908

-- Define the function
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 3 * Real.cos (x + φ) - 1

-- State the theorem
theorem symmetry_implies_phi (φ : ℝ) :
  (φ ∈ Set.Icc 0 Real.pi) →
  (∀ x : ℝ, f φ (Real.pi/3 - x) = f φ (Real.pi/3 + x)) →
  φ = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l1129_112908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1129_112936

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 1)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x + 1)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) ∧
  f x = 9/4 ∧
  ∀ y ∈ Set.Icc (-Real.pi/2) (Real.pi/2), f y ≤ f x ∧
  (x = Real.pi/3 ∨ x = -Real.pi/3) := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1129_112936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequences_grid_l1129_112900

/- Define M as a parameter -/
variable (M : ℚ)

/- Define the arithmetic sequences -/
def row_sequence (n : ℕ) : ℚ := 25 + (n - 1) * (-17/3)
def column1_sequence (n : ℕ) : ℚ := 12 + (n - 1) * 4
def column2_sequence (M : ℚ) (n : ℕ) : ℚ := M + (n - 1) * (-115/6)

/- State the theorem -/
theorem arithmetic_sequences_grid (M : ℚ) :
  row_sequence 1 = 25 ∧
  column1_sequence 2 = 16 ∧
  column1_sequence 3 = 20 ∧
  column2_sequence M 5 = -20 →
  M = 75/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequences_grid_l1129_112900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_study_time_allocation_l1129_112937

/-- Represents the time allocation for studying different question types on a citizenship test. -/
structure StudyTime where
  totalQuestions : ℕ
  multipleChoiceQuestions : ℕ
  fillInBlankQuestions : ℕ
  essayQuestions : ℕ
  multipleChoiceTime : ℕ  -- in minutes
  fillInBlankTime : ℕ     -- in minutes
  essayTime : ℕ           -- in minutes
  totalStudyTime : ℕ      -- in hours

/-- Theorem stating that the time allocated to each question type is approximately 16.67 hours. -/
theorem study_time_allocation (st : StudyTime)
    (h1 : st.totalQuestions = 90)
    (h2 : st.multipleChoiceQuestions = 30)
    (h3 : st.fillInBlankQuestions = 30)
    (h4 : st.essayQuestions = 30)
    (h5 : st.multipleChoiceTime = 15)
    (h6 : st.fillInBlankTime = 25)
    (h7 : st.essayTime = 45)
    (h8 : st.totalStudyTime = 50) :
    ∃ (timePerType : ℚ), abs (timePerType - 50 / 3) < 0.01 ∧ abs (timePerType - 16.67) < 0.01 := by
  sorry

#eval (50 : ℚ) / 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_study_time_allocation_l1129_112937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_range_l1129_112980

noncomputable def f (x : ℝ) := (1/2) * x^2 - 9 * Real.log x

def has_critical_point_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a ≤ x ∧ x ≤ b ∧ deriv f x = 0

theorem critical_point_range (a : ℝ) :
  has_critical_point_in_interval f (a - 1) (a + 1) ↔ 2 < a ∧ a < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_range_l1129_112980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1129_112934

noncomputable section

variable (ω : ℝ)
axiom ω_bounds : 0 < ω ∧ ω < 1

def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin (ω * x), Real.sin (ω * x) + Real.cos (ω * x))
def b (x : ℝ) : ℝ × ℝ := (Real.cos (ω * x), Real.sqrt 3 * (Real.sin (ω * x) - Real.cos (ω * x)))
def f (x : ℝ) : ℝ := (a ω x).1 * (b ω x).1 + (a ω x).2 * (b ω x).2

theorem problem_solution :
  (∀ x, f ω x = f ω (5 * Real.pi / 3 - x)) →
  (∃ A c, f ω A = 0 ∧ c = 3 ∧ A^2 + c^2 - (A * c)^2 = 13) →
  (∀ x, f ω x = 2 * Real.sin (x - Real.pi / 3)) ∧
  (∃ b, b = 4 ∧ ∃ A c, A^2 + c^2 - b^2 = 2 * b * c * Real.cos A) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1129_112934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_band_and_chorus_not_orchestra_l1129_112940

/-- Represents the number of students in a musical group or combination of groups. -/
structure StudentCount where
  count : ℕ

/-- The total number of students in the school. -/
def total_students : StudentCount := ⟨300⟩

/-- The number of students in the band. -/
def band_students : StudentCount := ⟨100⟩

/-- The number of students in the chorus. -/
def chorus_students : StudentCount := ⟨120⟩

/-- The number of students in the orchestra. -/
def orchestra_students : StudentCount := ⟨60⟩

/-- The number of students in at least one of band, chorus, or orchestra. -/
def students_in_at_least_one : StudentCount := ⟨200⟩

/-- The number of students in all three groups. -/
def students_in_all_three : StudentCount := ⟨10⟩

/-- Theorem stating that the number of students in both band and chorus but not in orchestra is 30. -/
theorem students_in_band_and_chorus_not_orchestra : ∃ (x : StudentCount), x.count = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_band_and_chorus_not_orchestra_l1129_112940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_shots_problem_l1129_112995

theorem jack_shots_problem (initial_shots : ℕ) (initial_success_rate : ℚ) 
  (additional_shots : ℕ) (new_success_rate : ℚ) 
  (h1 : initial_shots = 30)
  (h2 : initial_success_rate = 60 / 100)
  (h3 : additional_shots = 10)
  (h4 : new_success_rate = 62 / 100) : ℕ := by
  -- Define the number of successful shots in the last 10
  let last_10_success := 7

  -- The theorem states that the last_10_success must be 7
  have : last_10_success = 7 := by sorry

  exact last_10_success


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_shots_problem_l1129_112995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_8_50_equals_log_2_50_div_3_l1129_112945

theorem log_8_50_equals_log_2_50_div_3 : 
  Real.log 50 / Real.log 8 = (Real.log 50 / Real.log 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_8_50_equals_log_2_50_div_3_l1129_112945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1129_112950

/-- An ellipse with center at the origin, major axis 6, and eccentricity √5/3 -/
structure Ellipse where
  center : ℝ × ℝ
  major_axis : ℝ
  eccentricity : ℝ
  h_center : center = (0, 0)
  h_major_axis : major_axis = 6
  h_eccentricity : eccentricity = Real.sqrt 5 / 3

/-- A point on the ellipse -/
structure PointOnEllipse (C : Ellipse) where
  point : ℝ × ℝ
  h_on_ellipse : (point.1^2 / 9) + (point.2^2 / 4) = 1

/-- The distance from a point to the right focus -/
noncomputable def distToRightFocus (C : Ellipse) (P : PointOnEllipse C) : ℝ :=
  sorry

/-- The distance from a point to the right directrix -/
noncomputable def distToRightDirectrix (C : Ellipse) (P : PointOnEllipse C) : ℝ :=
  sorry

theorem ellipse_properties (C : Ellipse) :
  (∀ P : ℝ × ℝ, (P.1^2 / 9) + (P.2^2 / 4) = 1 ↔ ∃ Q : PointOnEllipse C, Q.point = P) ∧
  (∀ P : PointOnEllipse C, distToRightFocus C P = 4 →
    distToRightDirectrix C P = (6 / 5) * Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1129_112950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_T_bounds_l1129_112998

/-- Positive sequence a_n satisfying the given conditions -/
def a (n : ℕ+) : ℝ := 2 * n.val - 1

/-- Sum of the first n terms of sequence a_n -/
noncomputable def S (n : ℕ+) : ℝ := (n.val * (2 * n.val - 1)) / 2

/-- Condition relating S_n and a_n -/
axiom S_a_relation (n : ℕ+) : 4 * S n - 1 = a n ^ 2 + 2 * a n

/-- Sequence b_n defined in terms of a_n -/
noncomputable def b (n : ℕ+) : ℝ := 1 / (a n * (a n + 2))

/-- Sum of the first n terms of sequence b_n -/
noncomputable def T (n : ℕ+) : ℝ := (1 / 2) * (1 - 1 / (2 * n.val + 1))

/-- Theorem: General formula for a_n -/
theorem a_formula (n : ℕ+) : a n = 2 * n - 1 := by
  rfl

/-- Theorem: Bounds for T_n -/
theorem T_bounds (n : ℕ+) : 1/3 ≤ T n ∧ T n < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_T_bounds_l1129_112998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_concavity_and_inflection_points_l1129_112917

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x^2)

noncomputable def f'' (x : ℝ) : ℝ := (4 * x^2 - 2) * Real.exp (-x^2)

theorem gaussian_concavity_and_inflection_points :
  -- Concave up on (-∞, -1/√2) and (1/√2, ∞)
  (∀ x < -1 / Real.sqrt 2, f'' x > 0) ∧
  (∀ x > 1 / Real.sqrt 2, f'' x > 0) ∧
  -- Concave down on (-1/√2, 1/√2)
  (∀ x ∈ Set.Ioo (-1 / Real.sqrt 2) (1 / Real.sqrt 2), f'' x < 0) ∧
  -- Inflection points
  f'' (-1 / Real.sqrt 2) = 0 ∧
  f'' (1 / Real.sqrt 2) = 0 ∧
  f (-1 / Real.sqrt 2) = 1 / Real.sqrt (Real.exp 1) ∧
  f (1 / Real.sqrt 2) = 1 / Real.sqrt (Real.exp 1) := by
  sorry

#check gaussian_concavity_and_inflection_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_concavity_and_inflection_points_l1129_112917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_zero_l1129_112912

/-- Function to calculate the volume of a tetrahedron given its vertices -/
def volume_tetrahedron (v1 v2 v3 v4 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- The volume of a tetrahedron with given vertices is 0 if the vertices are coplanar -/
theorem tetrahedron_volume_zero (v1 v2 v3 v4 : ℝ × ℝ × ℝ) : 
  v1 = (5, 8, 10) → 
  v2 = (10, 10, 17) → 
  v3 = (4, 45, 46) → 
  v4 = (2, 5, 4) → 
  volume_tetrahedron v1 v2 v3 v4 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_zero_l1129_112912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_lender_interest_l1129_112967

theorem money_lender_interest (n : ℕ) : n = 7 → 
  (1000 : ℚ) = 800 * (3 / 100) * n + 1000 * (9 / 200) * (n + 2) + 1400 * (1 / 20) * (n - 1) := by
  intro h
  rw [h]
  norm_num
  ring
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_lender_interest_l1129_112967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_identity_l1129_112963

theorem functional_equation_identity (f : ℕ+ → ℕ+) 
  (h : ∀ n : ℕ+, f (f (f n)) + f (f n) + f n = 3 * n) : 
  ∀ n : ℕ+, f n = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_identity_l1129_112963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_g_composed_l1129_112902

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x - 5)

theorem smallest_x_in_domain_of_g_composed (x : ℝ) : 
  (∀ y, y < x → g (g y) ≠ 0 ∨ g (g y) > 0) → x = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_g_composed_l1129_112902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l1129_112981

theorem cube_root_sum_equals_one : (8 + 3 * Real.sqrt 21) ^ (1/3 : ℝ) + (8 - 3 * Real.sqrt 21) ^ (1/3 : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l1129_112981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_color_condition_l1129_112918

/-- Represents a color on the board -/
inductive Color
| Black
| White

/-- Represents a move on the board -/
structure Move where
  row : Nat
  col : Nat

/-- Represents the board state -/
def Board (n : Nat) := Fin n → Fin n → Color

/-- Applies a move to the board -/
def applyMove (n : Nat) (board : Board n) (move : Move) : Board n :=
  sorry

/-- Checks if all squares on the board have the same color -/
def allSameColor (n : Nat) (board : Board n) : Prop :=
  sorry

/-- Theorem stating the condition for achieving a uniform color board -/
theorem uniform_color_condition (n : Nat) (h : n ≥ 3) :
  (∃ (moves : List Move), ∀ (initial : Board n),
    ∃ (final : Board n), (List.foldl (applyMove n) initial moves = final) ∧ allSameColor n final) ↔
  ∃ (k : Nat), n = 4 * k ∧ k > 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_color_condition_l1129_112918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zahra_problem_l1129_112951

def iteration (n : ℕ) : ℕ := n / 2

def process (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => iteration (process start n)

theorem zahra_problem (start : ℕ) (h : start = 128) :
  (∃ n : ℕ, process start n ≤ 2 ∧ ∀ m : ℕ, m < n → process start m > 2) ∧
  (∀ n : ℕ, n < 6 → process start n > 2) ∧
  process start 6 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zahra_problem_l1129_112951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_inverse_function_l1129_112949

-- Define the logarithm function base 2
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the symmetry condition
def symmetric_about_xy (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem symmetry_implies_inverse_function
    (f : ℝ → ℝ)
    (h : symmetric_about_xy f (λ x ↦ log2 (x + 1))) :
    ∀ x, f x = 2^x - 1 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_inverse_function_l1129_112949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1129_112946

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, (2 : ℝ)^x = 5) ↔ (∃ x₀ : ℝ, (2 : ℝ)^x₀ ≠ 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1129_112946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_8_sided_die_l1129_112943

def standard_8_sided_die : Finset ℕ := Finset.range 8

theorem expected_value_8_sided_die :
  (Finset.sum standard_8_sided_die (λ i => (i + 1) * (1 / 8 : ℚ))) = (9 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_8_sided_die_l1129_112943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisible_by_1947_l1129_112952

def prod_even (n : ℕ) : ℕ := (Finset.range (n / 2)).prod (λ i => 2 * (i + 1))

theorem smallest_n_divisible_by_1947 : 
  ∀ n : ℕ, n % 2 = 0 → n > 0 → (prod_even n % 1947 = 0 ↔ n ≥ 3894) :=
by
  sorry

#eval prod_even 3894 % 1947

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisible_by_1947_l1129_112952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_polygon_arrangement_exists_l1129_112978

-- Define a polygon
structure Polygon where
  vertices : List (ℝ × ℝ)

-- Define equality for polygons
def equal_polygons (p1 p2 : Polygon) : Prop :=
  ∃ (translation : ℝ × ℝ), p1.vertices = p2.vertices.map (λ (x, y) ↦ (x + translation.1, y + translation.2))

-- Define the property of no interior points in common
def no_interior_overlap (p1 p2 : Polygon) : Prop :=
  sorry  -- Definition of no interior overlap

-- Define the property of sharing a boundary segment
def share_boundary (p1 p2 : Polygon) : Prop :=
  sorry  -- Definition of sharing a boundary segment

-- The main theorem
theorem four_polygon_arrangement_exists : 
  ∃ (p1 p2 p3 p4 : Polygon),
    equal_polygons p1 p2 ∧ equal_polygons p1 p3 ∧ equal_polygons p1 p4 ∧
    no_interior_overlap p1 p2 ∧ no_interior_overlap p1 p3 ∧ no_interior_overlap p1 p4 ∧
    no_interior_overlap p2 p3 ∧ no_interior_overlap p2 p4 ∧ no_interior_overlap p3 p4 ∧
    share_boundary p1 p2 ∧ share_boundary p1 p3 ∧ share_boundary p1 p4 ∧
    share_boundary p2 p3 ∧ share_boundary p2 p4 ∧ share_boundary p3 p4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_polygon_arrangement_exists_l1129_112978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l1129_112959

/-- Definition of the ellipse C -/
def ellipse_C (x y m : ℝ) : Prop := x^2/m^2 + y^2 = 1

/-- Definition of the line l -/
def line_l (x y t : ℝ) : Prop := y = x + t ∧ t > 0

/-- Definition of a point being inside a circle -/
def inside_circle (O A B : ℝ × ℝ) : Prop :=
  let OA := (A.1 - O.1, A.2 - O.2)
  let OB := (B.1 - O.1, B.2 - O.2)
  OA.1 * OB.1 + OA.2 * OB.2 < 0

/-- Main theorem -/
theorem ellipse_and_line_intersection
  (m : ℝ) (e : ℝ) (A B : ℝ × ℝ) (t : ℝ)
  (h_ellipse : ∀ x y, ellipse_C x y m ↔ x^2/m^2 + y^2 = 1)
  (h_eccentricity : e = Real.sqrt 2/2)
  (h_line : line_l A.1 A.2 t ∧ line_l B.1 B.2 t)
  (h_intersection : ellipse_C A.1 A.2 m ∧ ellipse_C B.1 B.2 m)
  (h_inside : inside_circle (0, 0) A B) :
  (∀ x y, ellipse_C x y m ↔ x^2/2 + y^2 = 1) ∧
  (0 < t ∧ t < 2*Real.sqrt 3/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l1129_112959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_expected_shots_l1129_112904

/-- The expected number of shots needed to reduce arrows by one -/
noncomputable def expected_shots_per_arrow : ℝ := 10 / 7

/-- The initial number of arrows -/
def initial_arrows : ℕ := 14

/-- The probability of hitting a cone -/
def hit_probability : ℝ := 0.1

/-- The number of additional arrows received for each hit -/
def additional_arrows_per_hit : ℕ := 3

/-- The expected total number of shots -/
noncomputable def expected_total_shots : ℝ := initial_arrows * expected_shots_per_arrow

theorem ivan_expected_shots :
  expected_total_shots = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_expected_shots_l1129_112904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_plus_A_implies_sin_pi_half_plus_A_l1129_112976

theorem cos_pi_plus_A_implies_sin_pi_half_plus_A 
  (A : ℝ) (h : Real.cos (Real.pi + A) = -1/2) : 
  Real.sin (Real.pi/2 + A) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_plus_A_implies_sin_pi_half_plus_A_l1129_112976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a1a3_l1129_112957

/-- Arithmetic sequence definition -/
def IsArithmeticSeq (f : ℕ → ℝ) (start finish : ℕ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, start ≤ n ∧ n < finish → f (n + 1) - f n = d

/-- Given a sequence of positive terms, prove the minimum value of a_1 * a_3 -/
theorem min_value_a1a3 (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_a2 : a 2 = 6)
  (h_arith : IsArithmeticSeq (λ n ↦ 1 / (a n + n)) 1 3) :
  ∃ (min : ℝ), (∀ a₁ a₃, a₁ > 0 ∧ a₃ > 0 ∧ 
    IsArithmeticSeq (λ n ↦ 1 / (a n + n)) 1 3 → a₁ * a₃ ≥ min) ∧
  min = 19 + 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a1a3_l1129_112957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_a_fill_time_l1129_112930

/-- Represents the rate at which a pipe fills a tank in tanks per hour -/
def PipeRate := ℝ

/-- Represents the time taken to fill a tank in hours -/
def FillTime := ℝ

/-- Given three pipes A, B, and C that fill a tank in 8 hours,
    where C is twice as fast as B and B is twice as fast as A,
    prove that pipe A alone will take 56 hours to fill the tank -/
theorem pipe_a_fill_time
  (a b c : ℝ)
  (h1 : a + b + c = 1 / 8)
  (h2 : b = 2 * a)
  (h3 : c = 2 * b)
  : (1 : ℝ) / a = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_a_fill_time_l1129_112930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_two_range_l1129_112901

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*x - 1 else -2*x + 6

-- Theorem statement
theorem f_greater_than_two_range :
  {t : ℝ | f t > 2} = {t : ℝ | t < 0 ∨ t > 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_two_range_l1129_112901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cone_volume_is_minimum_l1129_112916

/-- The minimum volume of a cone enclosing a sphere of radius r -/
noncomputable def min_cone_volume (r : ℝ) : ℝ := (8 * Real.pi * r^3) / 3

/-- Theorem stating that min_cone_volume gives the minimum volume of a cone enclosing a sphere of radius r -/
theorem min_cone_volume_is_minimum (r : ℝ) (h : r > 0) :
  ∀ (V : ℝ), (∃ (R m : ℝ), V = (1/3) * Real.pi * R^2 * m ∧ R^2 + (m - r)^2 = (m + r)^2) →
  V ≥ min_cone_volume r := by
  sorry

#check min_cone_volume
#check min_cone_volume_is_minimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cone_volume_is_minimum_l1129_112916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_seven_l1129_112984

theorem subsets_containing_seven (S : Finset ℕ) (h : S = {1, 2, 3, 4, 5, 6, 7}) :
  (S.powerset.filter (λ A => 7 ∈ A)).card = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_seven_l1129_112984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_ratio_l1129_112921

/-- Represents an octagon composed of unit squares, a rectangle, and a triangle. -/
structure Octagon where
  unit_squares : ℕ
  rectangle_height : ℚ
  triangle_base : ℚ
  triangle_height : ℚ

/-- Calculates the area of the rectangle in the octagon. -/
def rectangle_area (o : Octagon) : ℚ :=
  o.triangle_base * o.rectangle_height

/-- Calculates the area of the triangle in the octagon. -/
def triangle_area (o : Octagon) : ℚ :=
  (1 / 2) * o.triangle_base * o.triangle_height

/-- Theorem stating the ratio of triangle area to rectangle area in the octagon. -/
theorem octagon_area_ratio (o : Octagon) 
    (h1 : o.unit_squares = 12)
    (h2 : o.triangle_base = 6)
    (h3 : o.rectangle_height = 1)
    (h4 : triangle_area o + rectangle_area o + o.unit_squares = 27) :
  triangle_area o / rectangle_area o = 3/2 := by
  sorry

#eval (3 : ℚ) / 2  -- This will output 1.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_ratio_l1129_112921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_ratio_l1129_112965

noncomputable def sine (x : ℝ) : ℝ := Real.sin x
noncomputable def sine_period : ℝ := 2 * Real.pi

noncomputable def sine_60 : ℝ := Real.sin (Real.pi / 3)

noncomputable def intersection_point (n : ℤ) : ℝ := Real.pi / 6 + n * sine_period

noncomputable def distance_p (n : ℤ) : ℝ := intersection_point (n + 1) - intersection_point n
noncomputable def distance_q (n : ℤ) : ℝ := intersection_point (n + 2) - intersection_point (n + 1)

theorem intersection_distance_ratio :
  ∀ n : ℤ, distance_p n / distance_q n = 1 / 2 := by
  intro n
  -- Expand definitions
  unfold distance_p distance_q intersection_point
  -- Simplify the expression
  simp [sine_period]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_ratio_l1129_112965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_twenty_percent_l1129_112974

/-- Calculates the discount percentage for an annual subscription. -/
noncomputable def annual_subscription_discount_percentage 
  (monthly_cost : ℚ) (discounted_annual_cost : ℚ) : ℚ :=
  let annual_cost := monthly_cost * 12
  let discount_amount := annual_cost - discounted_annual_cost
  (discount_amount / annual_cost) * 100

/-- Theorem stating that the discount percentage is 20% for the given subscription costs. -/
theorem discount_percentage_is_twenty_percent :
  annual_subscription_discount_percentage 10 96 = 20 := by
  -- Unfold the definition and simplify
  unfold annual_subscription_discount_percentage
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_twenty_percent_l1129_112974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2017_is_zero_l1129_112915

/-- An arithmetic sequence -/
def arithmetic_sequence : ℕ → ℝ := sorry

/-- Sum of first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- Point A on the coordinate plane -/
def A : ℝ × ℝ := (arithmetic_sequence 1009, 1)

/-- Point B on the coordinate plane -/
def B : ℝ × ℝ := (2, -1)

/-- Point C on the coordinate plane -/
def C : ℝ × ℝ := (2, 2)

/-- Origin -/
def O : ℝ × ℝ := (0, 0)

/-- Vector OA -/
def OA : ℝ × ℝ := (A.1 - O.1, A.2 - O.2)

/-- Vector OB -/
def OB : ℝ × ℝ := (B.1 - O.1, B.2 - O.2)

/-- Vector OC -/
def OC : ℝ × ℝ := (C.1 - O.1, C.2 - O.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem: Given the conditions, prove that S_2017 = 0 -/
theorem sum_2017_is_zero : 
  dot_product OA OC = dot_product OB OC → S 2017 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2017_is_zero_l1129_112915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_line_l1129_112927

-- Define the curve and line functions
noncomputable def curve (x : ℝ) : ℝ := 3 * Real.sqrt x
def line (x : ℝ) : ℝ := x + 2

-- Define the intersection points
def x₁ : ℝ := 1
def x₂ : ℝ := 4

-- Define the enclosed area
noncomputable def enclosed_area : ℝ := ∫ x in x₁..x₂, curve x - line x

-- Theorem statement
theorem area_enclosed_by_curve_and_line :
  enclosed_area = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_line_l1129_112927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_supplies_count_l1129_112983

/-- Calculates the number of supplies left after a series of events --/
def supplies_left (students : ℕ) (paper_per_student : ℕ) (glue_bottles : ℕ) 
  (loss_fraction : ℚ) (additional_paper : ℕ) : ℕ :=
  let initial_paper := students * paper_per_student
  let initial_supplies := initial_paper + glue_bottles
  let lost_supplies := (loss_fraction * initial_supplies).floor.toNat
  let remaining_supplies := initial_supplies - lost_supplies
  remaining_supplies + additional_paper

/-- Proves that given the specific conditions, the final number of supplies is 44 --/
theorem final_supplies_count : 
  supplies_left 15 5 10 (2/3) 15 = 44 := by
  -- Expand the definition of supplies_left
  unfold supplies_left
  -- Perform the calculations
  simp [Rat.floor, Int.toNat]
  -- The proof is completed
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_supplies_count_l1129_112983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_16_seconds_l1129_112994

/-- The time taken for a train to cross a platform -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) (platform_length : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem stating that the time taken for a specific train to cross a specific platform is approximately 16 seconds -/
theorem train_crossing_time_approx_16_seconds :
  let train_length := (140 : ℝ)
  let train_speed_kmph := (84 : ℝ)
  let platform_length := (233.3632 : ℝ)
  let crossing_time := train_crossing_time train_length train_speed_kmph platform_length
  ∃ ε > 0, abs (crossing_time - 16) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_16_seconds_l1129_112994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l1129_112919

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + 3*y^2 = 4

-- Define the point of interest
def point_of_interest (x y : ℝ) : Prop := curve x y ∧ y = 1 ∧ x < 0

-- Theorem statement
theorem tangent_and_normal_equations 
  (x y : ℝ) 
  (h : point_of_interest x y) :
  (∃ (m b : ℝ), ∀ (x' y' : ℝ), y' = m*x' + b ↔ 
    (x' - x)*(2*x) + (y' - y)*(6*y) = 0) ∧
  (∃ (m' b' : ℝ), ∀ (x' y' : ℝ), y' = m'*x' + b' ↔ 
    (x' - x)*(6*y) - (y' - y)*(2*x) = 0) ∧
  (m = 1/3 ∧ b = 4/3) ∧
  (m' = -3 ∧ b' = -2) := by
  sorry

#check tangent_and_normal_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l1129_112919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_from_square_l1129_112938

/-- Properties of a cylinder formed by rotating a square --/
theorem cylinder_from_square (side_length : ℝ) (side_length_pos : 0 < side_length) :
  let radius := side_length / 2
  let height := side_length
  let volume := π * radius^2 * height
  let surface_area := 2 * π * radius^2 + 2 * π * radius * height
  side_length = 20 →
  volume = 2000 * π ∧ surface_area = 600 * π := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_from_square_l1129_112938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_log_proper_divisors_million_l1129_112903

noncomputable def million : ℕ := 1000000

-- Define the sum of logarithms of proper divisors
noncomputable def sum_log_proper_divisors (n : ℕ) : ℝ :=
  (Finset.filter (λ d => d ≠ 1 ∧ d ≠ n) (Nat.divisors n)).sum (λ d => Real.log (d : ℝ) / Real.log 10)

theorem sum_log_proper_divisors_million :
  ⌊sum_log_proper_divisors million⌋ = 141 ∧ 
  sum_log_proper_divisors million - ⌊sum_log_proper_divisors million⌋ < 0.5 := by
  sorry

#check sum_log_proper_divisors_million

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_log_proper_divisors_million_l1129_112903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_sum_eq_pi_fourth_l1129_112962

theorem arctan_sum_eq_pi_fourth (n : ℕ+) : 
  (Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/(n : ℝ)) = π/4) ↔ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_sum_eq_pi_fourth_l1129_112962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_partition_exists_l1129_112941

/-- Represents a square divided into a 4x4 grid -/
structure GridSquare :=
  (grid : Matrix (Fin 4) (Fin 4) Bool)

/-- Represents a partition of the GridSquare -/
structure Partition :=
  (parts : Fin 4 → Set (Fin 4 × Fin 4))

/-- Checks if a partition is valid according to the problem conditions -/
def is_valid_partition (s : GridSquare) (p : Partition) : Prop :=
  -- Each part contains exactly one shaded square
  (∀ i : Fin 4, ∃! (x y : Fin 4), (x, y) ∈ p.parts i ∧ s.grid x y = true) ∧
  -- All parts have the same shape and size
  (∀ i j : Fin 4, ∃ f : (Fin 4 × Fin 4) → (Fin 4 × Fin 4), 
    Function.Bijective f ∧ (∀ x, x ∈ p.parts i ↔ f x ∈ p.parts j))

/-- The main theorem to prove -/
theorem square_partition_exists (s : GridSquare) 
  (h : ∀ i : Fin 4, ∃! (x y : Fin 4), x.val / 2 = i.val / 2 ∧ y.val / 2 = i.val % 2 ∧ s.grid x y = true) :
  ∃ p : Partition, is_valid_partition s p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_partition_exists_l1129_112941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_interval_of_increase_l1129_112968

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 6

-- Define what it means for f to be an even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the interval of increase
def interval_of_increase (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

-- State the theorem
theorem even_function_interval_of_increase :
  ∃ a : ℝ, is_even_function (f a) →
  interval_of_increase (f a) { x : ℝ | x ≥ 0 } :=
by
  -- We know a = 0 for f to be even
  use 0
  intro h_even
  -- Now we need to prove the interval of increase
  unfold interval_of_increase
  intros x y hx hy hxy
  -- The function is now x^2 + 6
  have h_f : f 0 = fun x => x^2 + 6 := by
    ext x
    simp [f]
  -- Use this fact and prove the inequality
  rw [h_f]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_interval_of_increase_l1129_112968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_for_101st_heaviest_l1129_112954

/-- Represents a coin with a unique weight -/
structure Coin where
  weight : ℕ
  deriving Ord

/-- Represents a two-pan balance scale -/
def BalanceScale := Coin → Coin → Bool

/-- Finds the nth heaviest coin among a list of coins using a balance scale -/
def findNthHeaviest (coins : List Coin) (n : ℕ) (scale : BalanceScale) : ℕ :=
  sorry

/-- Theorem: 8 weighings are needed to find the 101st heaviest coin among 201 coins -/
theorem min_weighings_for_101st_heaviest 
  (silver_coins : List Coin) 
  (gold_coins : List Coin) 
  (scale : BalanceScale) 
  (h1 : silver_coins.length = 100)
  (h2 : gold_coins.length = 101)
  (h3 : ∀ c1 c2, c1 ∈ silver_coins ++ gold_coins → c2 ∈ silver_coins ++ gold_coins → c1 ≠ c2 → c1.weight ≠ c2.weight) :
  findNthHeaviest (silver_coins ++ gold_coins) 101 scale = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_for_101st_heaviest_l1129_112954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_solutions_l1129_112989

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 2*x - 2 else 2*x - 3

-- State the theorem
theorem f_equals_one_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 1 ∧ f x₂ = 1 ∧
  ∀ (x : ℝ), f x = 1 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_solutions_l1129_112989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1129_112925

/-- Given a hyperbola C with point A and asymptotes, prove its standard equation -/
theorem hyperbola_equation (C : Set (ℝ × ℝ)) (A : ℝ × ℝ) 
  (h1 : A ∈ C)
  (h2 : A = (2 * Real.sqrt 2, 2))
  (h3 : ∀ (x y : ℝ), (x, y) ∈ C → (y = (1/2) * x ∨ y = -(1/2) * x) → 
    ∃ (t : ℝ), t ≠ 0 ∧ ((x, y) = (t, (1/2) * t) ∨ (x, y) = (t, -(1/2) * t))) :
  ∀ (x y : ℝ), (x, y) ∈ C ↔ y^2 / 2 - x^2 / 8 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1129_112925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_symmetric_product_l1129_112991

theorem complex_symmetric_product : 
  ∀ z₁ z₂ : ℂ, 
  z₁ = 2 + I → 
  z₂ = -z₁.re + I * z₁.im → 
  z₁ * z₂ = -5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_symmetric_product_l1129_112991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l1129_112970

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.exp (2 * x)
def g (m : ℝ) (x : ℝ) : ℝ := m * x + 1

-- State the theorem
theorem function_range_theorem (m : ℝ) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, g m x₀ = f x₁) →
  m ∈ Set.Iic (1 - Real.exp 2) ∪ Set.Ici (Real.exp 2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l1129_112970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_y_axis_l1129_112905

/-- The ellipse with equation x²/4 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}

/-- The foci of the ellipse -/
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 3, 0)

/-- A point M on the ellipse -/
noncomputable def M : ℝ × ℝ := sorry

/-- The dot product of vectors MF1 and MF2 is zero -/
axiom h_perpendicular : (M.1 - F1.1) * (M.1 - F2.1) + (M.2 - F1.2) * (M.2 - F2.2) = 0

/-- M is on the ellipse -/
axiom h_on_ellipse : M ∈ Ellipse

/-- The theorem to be proved -/
theorem distance_to_y_axis :
  |M.1| = 2 * Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_y_axis_l1129_112905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_permutations_l1129_112923

/-- Represents a 6-digit integer with unique digits --/
def SixDigitInt := Fin 6 → Fin 10

/-- Checks if two SixDigitInt are permutations of each other --/
def isPermutation (x y : SixDigitInt) : Prop := ∃ (σ : Equiv (Fin 6) (Fin 6)), ∀ i, x i = y (σ i)

/-- Represents the set of letters A, B, C --/
inductive Letter | A | B | C

/-- Associates a letter with a digit --/
def LetterDigit := Letter → Fin 10

/-- Calculates the sum of digits in a SixDigitInt --/
def sumDigits (x : SixDigitInt) : ℕ := (x 0).val + (x 1).val + (x 2).val + (x 3).val + (x 4).val + (x 5).val

theorem max_sum_of_permutations (a b c : SixDigitInt) (l : LetterDigit) :
  (∀ i j, i ≠ j → a i ≠ a j) →
  (∀ i j, i ≠ j → b i ≠ b j) →
  (∀ i j, i ≠ j → c i ≠ c j) →
  isPermutation a b →
  isPermutation b c →
  isPermutation c a →
  (∀ d : Fin 10, ∃ i, a i = d ∨ b i = d ∨ c i = d ∨ ∃ lt, l lt = d) →
  (∀ lt : Letter, l lt ≠ 0) →
  (∀ lt1 lt2, lt1 ≠ lt2 → l lt1 ≠ l lt2) →
  (∃ i, a i = l Letter.A) →
  (∃ i, b i = l Letter.B) →
  (∃ i, c i = l Letter.C) →
  (sumDigits a + sumDigits b + sumDigits c : ℕ) ≤ 1436649 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_permutations_l1129_112923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_mpg_l1129_112948

/-- Calculates the average miles per gallon for a car trip --/
def average_mpg (initial_odometer final_odometer : ℕ) 
                (initial_gas mid_trip_gas final_gas : ℕ) : ℚ :=
  let total_distance := final_odometer - initial_odometer
  let total_gas := initial_gas + mid_trip_gas + final_gas
  (total_distance : ℚ) / total_gas

/-- Rounds a rational number to the nearest tenth --/
def round_to_tenth (x : ℚ) : ℚ :=
  ⌊(x * 10 + 1/2)⌋ / 10

theorem car_trip_mpg : 
  let initial_odometer : ℕ := 56300
  let final_odometer : ℕ := 57200
  let initial_gas : ℕ := 8  -- 2 gallons already in tank + 8 added
  let mid_trip_gas : ℕ := 18
  let final_gas : ℕ := 25
  round_to_tenth (average_mpg initial_odometer final_odometer initial_gas mid_trip_gas final_gas) = 88/5 := by
  sorry

#eval round_to_tenth (average_mpg 56300 57200 8 18 25)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_mpg_l1129_112948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_stream_range_optimal_z_l1129_112928

/-- The range of a water stream from a cylindrical vessel -/
noncomputable def stream_range (h z : ℝ) : ℝ := 2 * Real.sqrt (z * (h - z))

/-- Theorem: The range of the water stream is maximized when z = h/2 -/
theorem max_stream_range (h : ℝ) (h_pos : h > 0) :
  ∃ (z : ℝ), z ∈ Set.Icc 0 h ∧ 
  ∀ (y : ℝ), y ∈ Set.Icc 0 h → stream_range h z ≥ stream_range h y :=
by
  sorry

/-- Corollary: The optimal z is exactly h/2 -/
theorem optimal_z (h : ℝ) (h_pos : h > 0) :
  ∃! (z : ℝ), z ∈ Set.Icc 0 h ∧ 
  ∀ (y : ℝ), y ∈ Set.Icc 0 h → stream_range h z ≥ stream_range h y ∧
  z = h / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_stream_range_optimal_z_l1129_112928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l1129_112992

def z (a : ℝ) : ℂ := (a^2 - 1) + (a - 2) * Complex.I

theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → z a = Complex.I * (z a).im) ∧
  (∃ a : ℝ, a ≠ 1 ∧ z a = Complex.I * (z a).im) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l1129_112992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_is_45_degrees_l1129_112961

/-- The angle of inclination of a line with equation ax + by + c = 0 -/
noncomputable def angle_of_inclination (a b c : ℝ) : ℝ := Real.arctan (a / b)

/-- The line equation 2x - 2y + 1 = 0 -/
def line_equation (x y : ℝ) : Prop := 2 * x - 2 * y + 1 = 0

theorem angle_of_inclination_is_45_degrees :
  angle_of_inclination 2 (-2) 1 = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_is_45_degrees_l1129_112961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_variance_greater_than_girls_l1129_112942

noncomputable def boys_scores : List ℝ := [86, 94, 88, 92, 90]
noncomputable def girls_scores : List ℝ := [88, 93, 93, 88, 93]

noncomputable def variance (scores : List ℝ) : ℝ :=
  let mean := scores.sum / scores.length
  (scores.map (fun x => (x - mean) ^ 2)).sum / scores.length

theorem boys_variance_greater_than_girls :
  variance boys_scores > variance girls_scores := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_variance_greater_than_girls_l1129_112942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1129_112986

-- Define the lines and point
def l₁ (x y : ℝ) : Prop := 4 * x + y = 0
def l₂ (x y : ℝ) : Prop := x + y - 1 = 0
def P : ℝ × ℝ := (3, -2)

-- Define the circle
def circleEq (C : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  (x - C.1)^2 + (y - C.2)^2 = r^2

-- Theorem statement
theorem circle_equation :
  ∃ C : ℝ × ℝ, ∃ r : ℝ,
    (l₁ C.1 C.2) ∧  -- Center C is on l₁
    (l₂ P.1 P.2) ∧  -- Point P is on l₂
    (∀ x y : ℝ, l₂ x y → ((x - P.1) * (y - C.2) = (x - C.1) * (y - P.2))) ∧  -- Circle is tangent to l₂ at P
    (∀ x y : ℝ, circleEq C r x y ↔ (x - 1)^2 + (y + 4)^2 = 8) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1129_112986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiply_24_12_l1129_112909

theorem multiply_24_12 : 24 * 12 = 288 := by
  have h1 : 12 = 10 + 2 := by rfl
  have h2 : 24 * 2 = 48 := by rfl
  have h3 : 24 * 10 = 240 := by rfl
  calc
    24 * 12 = 24 * (10 + 2) := by rw [h1]
    _ = 24 * 10 + 24 * 2 := by rw [mul_add]
    _ = 240 + 48 := by rw [h3, h2]
    _ = 288 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiply_24_12_l1129_112909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_zero_equally_spaced_l1129_112929

/-- Two complex numbers are equally spaced on the unit circle if their arguments differ by π -/
def equally_spaced (z₁ z₂ : ℂ) : Prop :=
  ∃ θ : ℝ, z₁ = Complex.exp (Complex.I * θ) ∧ z₂ = Complex.exp (Complex.I * (θ + Real.pi))

theorem complex_sum_zero_equally_spaced :
  ∀ z₁ z₂ : ℂ,
    Complex.abs z₁ = 1 →
    Complex.abs z₂ = 1 →
    z₁ + z₁^2 + z₂ + z₂^2 = 0 →
    equally_spaced z₁ z₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_zero_equally_spaced_l1129_112929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1129_112972

-- Define the expression
noncomputable def f (x : ℝ) : ℝ := (Real.log (5 - x)) / Real.sqrt (x - 2)

-- Define a predicate for when the function is well-defined
def isWellDefined (x : ℝ) : Prop := 2 < x ∧ x < 5

-- Theorem statement
theorem f_domain : 
  {x : ℝ | isWellDefined x} = Set.Ioo 2 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1129_112972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_10_l1129_112973

-- Define the selling price and cost price
noncomputable def selling_price : ℝ := 100
noncomputable def cost_price : ℝ := 90.91

-- Define profit
noncomputable def profit : ℝ := selling_price - cost_price

-- Define profit percentage
noncomputable def profit_percentage : ℝ := (profit / cost_price) * 100

-- Theorem to prove
theorem profit_percentage_is_10 : 
  profit_percentage = 10 := by
  -- Unfold definitions
  unfold profit_percentage profit selling_price cost_price
  -- Perform numerical calculations
  norm_num
  -- Complete the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_10_l1129_112973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ellipse_area_l1129_112990

-- Define the ellipse equation
def ellipse_eq (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle equations
def circle1_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4
def circle2_eq (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

-- Define the area of an ellipse
noncomputable def ellipse_area (a b : ℝ) : ℝ := Real.pi * a * b

-- Theorem statement
theorem smallest_ellipse_area :
  ∀ a b : ℝ, a > 0 → b > 0 →
  (∃ x y : ℝ, ellipse_eq a b x y ∧ circle1_eq x y) →
  (∃ x y : ℝ, ellipse_eq a b x y ∧ circle2_eq x y) →
  ellipse_area a b ≥ 4 * Real.pi := by
  sorry

#check smallest_ellipse_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ellipse_area_l1129_112990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_diagonal_ratio_l1129_112971

/-- Predicate stating that a quadrilateral ABCD is inscribed in a circle -/
def QuadrilateralInscribed 
  (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ), 
    dist center A = radius ∧
    dist center B = radius ∧
    dist center C = radius ∧
    dist center D = radius

/-- Given an inscribed quadrilateral ABCD, the ratio of its diagonals AC and BD
    is equal to the ratio of the sums of the products of opposite sides. -/
theorem inscribed_quadrilateral_diagonal_ratio 
  (A B C D : EuclideanSpace ℝ (Fin 2)) 
  (h : QuadrilateralInscribed A B C D) :
  (dist A C) / (dist B D) = 
  ((dist A B) * (dist A D) + (dist C B) * (dist C D)) / 
  ((dist B A) * (dist B C) + (dist D A) * (dist D C)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_diagonal_ratio_l1129_112971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l1129_112926

theorem sin_cos_identity : Real.sin (Real.pi + 2) - Real.cos (Real.pi / 2 + 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l1129_112926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_monotonicity_l1129_112944

open Real

-- Define the function f(x) = ln x - x
noncomputable def f (x : ℝ) : ℝ := log x - x

-- State the theorem
theorem tangent_and_monotonicity :
  -- The derivative at x = 1 is zero
  (deriv f) 1 = 0 ∧
  -- The function is increasing on (0, 1)
  (∀ y : ℝ, 0 < y ∧ y < 1 → (deriv f) y > 0) ∧
  -- The function is decreasing on (1, +∞)
  (∀ z : ℝ, z > 1 → (deriv f) z < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_monotonicity_l1129_112944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mandm_probability_l1129_112999

/-- Represents the number of M&Ms received from a single coin insertion -/
inductive MandMResult
| one
| two

/-- Calculates the probability of having exactly n M&Ms after some number of coin insertions -/
def prob_n_mandms : ℕ → ℚ
| 0 => 1
| 1 => 1/2
| n+2 => (prob_n_mandms (n+1) * 1/2) + (prob_n_mandms n * 1/2)

/-- The probability of getting exactly 7 M&Ms when inserting coins until at least 6 M&Ms are obtained -/
def prob_seven_mandms : ℚ := (prob_n_mandms 5) * (1/2)

theorem mandm_probability : prob_seven_mandms = 21/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mandm_probability_l1129_112999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_ratio_triangle_l1129_112977

theorem largest_angle_in_ratio_triangle : 
  ∀ (a b c : ℝ),
    a > 0 → b > 0 → c > 0 →
    a + b + c = 180 →
    (a : ℝ) / 3 = (b : ℝ) / 4 → (b : ℝ) / 4 = (c : ℝ) / 5 →
    max a (max b c) = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_ratio_triangle_l1129_112977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brixton_average_temperature_l1129_112985

/-- The average daily high temperature in Brixton for a week -/
def average_temperature (temperatures : List Float) : Float :=
  temperatures.sum / temperatures.length.toFloat

/-- Theorem: The average daily high temperature in Brixton for the given week is 59.2°F -/
theorem brixton_average_temperature :
  let temperatures : List Float := [55, 68, 63, 60, 50]
  average_temperature temperatures = 59.2 := by
  sorry

#eval average_temperature [55, 68, 63, 60, 50]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brixton_average_temperature_l1129_112985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patio_coverage_l1129_112966

/-- The minimum number of square tiles needed to cover a square patio -/
def min_tiles (patio_side : ℚ) (tile_side : ℚ) : ℕ :=
  ⌈(patio_side / tile_side)^2⌉.toNat

/-- Proof that 900 tiles of 20cm x 20cm are needed to cover a 6m x 6m patio -/
theorem patio_coverage : min_tiles 6 0.2 = 900 := by
  sorry

#eval min_tiles 6 0.2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_patio_coverage_l1129_112966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_knights_is_six_l1129_112910

/-- Represents a person at the table, either a knight or a liar -/
inductive Person
| Knight
| Liar
deriving BEq, Repr

/-- Represents the number of coins a person has -/
def CoinCount := Fin 3

/-- Represents the table of people -/
def Table := Vector Person 10

/-- Checks if a person's statement is consistent with their type and coin count -/
def consistent_statement (p : Person) (own_coins : CoinCount) (right_neighbor_coins : CoinCount) : Prop :=
  match p with
  | Person.Knight => own_coins.val > right_neighbor_coins.val
  | Person.Liar => own_coins.val ≤ right_neighbor_coins.val

/-- Checks if a given table configuration is valid according to the problem rules -/
def valid_table (t : Table) (coin_distribution : Vector CoinCount 10) : Prop :=
  ∀ i : Fin 10, consistent_statement (t.get i) (coin_distribution.get i) (coin_distribution.get ((i + 1) % 10))

/-- The main theorem: The maximum number of knights in a valid table configuration is 6 -/
theorem max_knights_is_six :
  ∀ t : Table, ∀ cd : Vector CoinCount 10,
    valid_table t cd →
    (t.toList.count Person.Knight ≤ 6) ∧ 
    (∃ t' : Table, ∃ cd' : Vector CoinCount 10, valid_table t' cd' ∧ t'.toList.count Person.Knight = 6) :=
by sorry

#check max_knights_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_knights_is_six_l1129_112910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_zone_score_l1129_112996

/-- Represents a circular target with zones -/
structure Target where
  numZones : Nat
  bullseyeRadius : ℝ
  bullseyeScore : ℝ

/-- Calculates the area of a circle given its radius -/
noncomputable def circleArea (radius : ℝ) : ℝ := Real.pi * radius^2

/-- Calculates the area of a ring given its inner and outer radii -/
noncomputable def ringArea (innerRadius outerRadius : ℝ) : ℝ :=
  circleArea outerRadius - circleArea innerRadius

/-- Calculates the score of a zone based on its area and the bullseye score -/
noncomputable def zoneScore (zoneArea bullseyeArea bullseyeScore : ℝ) : ℝ :=
  (bullseyeArea / zoneArea) * bullseyeScore

/-- Main theorem: The score of the blue (second-to-last) zone is 45 points -/
theorem blue_zone_score (t : Target) (h1 : t.numZones = 5) (h2 : t.bullseyeScore = 315) :
  zoneScore (ringArea (3 * t.bullseyeRadius) (4 * t.bullseyeRadius))
            (circleArea t.bullseyeRadius)
            t.bullseyeScore = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_zone_score_l1129_112996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l1129_112906

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the intersection points
structure IntersectionPoint (k : ℝ) where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y
  on_line : line_through_focus k x y

-- Theorem statement
theorem distance_between_intersection_points 
  (k : ℝ) (P1 P2 : IntersectionPoint k) 
  (h : P1.y + P2.y = 6) : 
  Real.sqrt ((P1.x - P2.x)^2 + (P1.y - P2.y)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l1129_112906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_coords_l1129_112955

/-- Parabola type -/
structure Parabola where
  a : ℝ
  h : a ≠ 0

/-- Circle type -/
structure Circle where
  r : ℝ

/-- Point on a parabola -/
noncomputable def parabola_point (p : Parabola) (x : ℝ) : ℝ × ℝ :=
  (x, x^2 + p.a * x - 5)

/-- Secant line slope between two points on a parabola -/
noncomputable def secant_slope (p : Parabola) (x₁ x₂ : ℝ) : ℝ :=
  let (_, y₁) := parabola_point p x₁
  let (_, y₂) := parabola_point p x₂
  (y₂ - y₁) / (x₂ - x₁)

/-- Tangent point to both parabola and circle -/
noncomputable def tangent_point (p : Parabola) : ℝ × ℝ :=
  (-1, -p.a - 4)

/-- Vertex of a parabola -/
noncomputable def parabola_vertex (p : Parabola) : ℝ × ℝ :=
  (-p.a / 2, (-p.a / 2)^2 + p.a * (-p.a / 2) - 5)

/-- Main theorem -/
theorem parabola_vertex_coords (p : Parabola) (c : Circle) :
  secant_slope p (-4) 2 = p.a - 2 →
  (∃ (k : ℝ), k * (tangent_point p).1 - (tangent_point p).2 - 6 = 0 ∧
              k = secant_slope p (-4) 2) →
  5 * (tangent_point p).1^2 + 5 * (tangent_point p).2^2 = 36 →
  parabola_vertex p = (-2, -9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_coords_l1129_112955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_profit_l1129_112913

/-- Represents the profit calculation for a school club selling candy bars -/
theorem candy_bar_profit
  (total_bars : ℕ)
  (buy_price : ℚ)
  (sell_price : ℚ)
  (bars_per_buy : ℕ)
  (bars_per_sell : ℕ)
  (h1 : total_bars = 800)
  (h2 : buy_price = 3)
  (h3 : sell_price = 2)
  (h4 : bars_per_buy = 4)
  (h5 : bars_per_sell = 3) :
  let cost_per_bar : ℚ := buy_price / bars_per_buy
  let revenue_per_bar : ℚ := sell_price / bars_per_sell
  let total_cost : ℚ := cost_per_bar * total_bars
  let total_revenue : ℚ := revenue_per_bar * total_bars
  let profit : ℚ := total_revenue - total_cost
  ∃ (ε : ℚ), abs (profit - (-67)) < ε ∧ ε < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_profit_l1129_112913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_approx_l1129_112922

/-- Calculates the average speed for a round trip in a multi-story building --/
noncomputable def average_speed_round_trip (floors : ℕ) (steps_per_floor : ℕ) (speed_up : ℝ) (speed_down : ℝ) (step_height_inches : ℝ) : ℝ :=
  let total_steps := (floors * steps_per_floor : ℝ)
  let time_up := total_steps / speed_up
  let time_down := total_steps / speed_down
  let total_time := time_up + time_down
  let building_height_feet := (floors * steps_per_floor : ℝ) * step_height_inches / 12
  let total_distance_miles := (2 * building_height_feet) / 5280
  let total_time_hours := total_time / 3600
  total_distance_miles / total_time_hours

/-- The average speed for the round trip is approximately 1.486 miles per hour --/
theorem round_trip_speed_approx (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ |average_speed_round_trip 50 14 3 5 7 - 1.486| < δ ∧ δ < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_approx_l1129_112922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walker_speed_l1129_112933

/-- A track with semicircular ends and straight sides -/
structure Track where
  innerRadius : ℝ
  width : ℝ
  straightLength : ℝ

/-- The time difference between walking the outer and inner edges of the track -/
noncomputable def timeDifference (track : Track) (speed : ℝ) : ℝ :=
  (2 * track.straightLength + 2 * Real.pi * (track.innerRadius + track.width)) / speed -
  (2 * track.straightLength + 2 * Real.pi * track.innerRadius) / speed

/-- The theorem stating that the walker's speed is π/3 given the track conditions -/
theorem walker_speed (track : Track) (h1 : track.width = 8) (h2 : timeDifference track (Real.pi/3) = 48) :
  ∃ (speed : ℝ), speed = Real.pi/3 ∧ timeDifference track speed = 48 := by
  sorry

#check walker_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walker_speed_l1129_112933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_arithmetic_sequence_l1129_112920

def sequence_a : ℕ → ℝ := sorry

def S : ℕ → ℝ := sorry

axiom a_1 : sequence_a 1 = 1

axiom point_on_line : ∀ n : ℕ, 2 * sequence_a (n + 1) + S n - 2 = 0

theorem sequence_formula : ∀ n : ℕ, sequence_a n = (1/2) ^ (n - 1) := by
  sorry

def lambda : ℝ := 2

theorem arithmetic_sequence :
  ∃ d : ℝ, ∀ n : ℕ, (S (n + 1) + lambda * (n + 1) + lambda / (2 ^ (n + 1))) - 
                    (S n + lambda * n + lambda / (2 ^ n)) = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_arithmetic_sequence_l1129_112920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l1129_112947

/-- The line equation mx + y - 2m - 1 = 0 -/
def line_eq (m : ℝ) (x y : ℝ) : Prop :=
  m * x + y - 2 * m - 1 = 0

/-- The circle equation x^2 + y^2 - 2x - 4y = 0 -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y = 0

/-- The line passes through the point (2, 1) -/
def line_passes_through_point (m : ℝ) : Prop :=
  line_eq m 2 1

theorem shortest_chord_length (m : ℝ) :
  line_passes_through_point m →
  (∀ x y : ℝ, line_eq m x y → circle_eq x y → 
    ∀ m' : ℝ, line_passes_through_point m' →
      ∃ x' y' : ℝ, line_eq m' x' y' ∧ circle_eq x' y' →
        (x - 2)^2 + (y - 1)^2 ≤ (x' - 2)^2 + (y' - 1)^2) →
  m = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l1129_112947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_implies_a_range_l1129_112964

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 4*a) / Real.log 0.5

-- Define the interval [2, +∞)
def interval : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem f_monotone_decreasing_implies_a_range (a : ℝ) :
  (∀ x y, x ∈ interval → y ∈ interval → x < y → f a x > f a y) →
  a ∈ Set.Ioo (-2 : ℝ) 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_implies_a_range_l1129_112964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_similar_noncongruent_trihedral_angles_l1129_112939

-- Define geometric figures
class GeometricFigure (α : Type) where
  -- Add necessary fields if needed

-- Define similarity for geometric figures
def isSimilar {α : Type} [GeometricFigure α] (a b : α) : Prop :=
  sorry

-- Define congruence for geometric figures
def isCongruent {α : Type} [GeometricFigure α] (a b : α) : Prop :=
  sorry

-- Define trihedral angle
structure TrihedralAngle where
  -- Add necessary fields if needed

-- Define triangle
structure Triangle where
  -- Add necessary fields if needed

-- Instance of GeometricFigure for TrihedralAngle
instance : GeometricFigure TrihedralAngle :=
{ }

-- Instance of GeometricFigure for Triangle
instance : GeometricFigure Triangle :=
{ }

-- Define triangle similarity conditions
def triangleSimilarityAA (t1 t2 : Triangle) : Prop :=
  sorry

def triangleSimilaritySAS (t1 t2 : Triangle) : Prop :=
  sorry

def triangleSimilaritySSS (t1 t2 : Triangle) : Prop :=
  sorry

-- Theorem: No conditions for similar but not congruent trihedral angles
theorem no_similar_noncongruent_trihedral_angles :
  ∀ (a b : TrihedralAngle),
    isSimilar a b → isCongruent a b :=
  by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_similar_noncongruent_trihedral_angles_l1129_112939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_distant_coprime_point_l1129_112982

noncomputable def is_coprime (x y : ℕ+) : Prop := Nat.gcd x.val y.val = 1

def coprime_points : Set (ℕ+ × ℕ+) := {p | is_coprime p.1 p.2}

noncomputable def euclidean_distance (p1 p2 : ℕ+ × ℕ+) : ℝ :=
  Real.sqrt ((p1.1.val - p2.1.val : ℝ)^2 + (p1.2.val - p2.2.val : ℝ)^2)

theorem exists_distant_coprime_point :
  ∃ p ∈ coprime_points, ∀ q ∈ coprime_points, p ≠ q → euclidean_distance p q > 2020 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_distant_coprime_point_l1129_112982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_rep_2023_l1129_112958

def binary_representation (n : ℕ) : List ℕ :=
  sorry

theorem binary_rep_2023 :
  let rep := binary_representation 2023
  (∀ i j, i ≠ j → rep.get! i ≠ rep.get! j) ∧
  (rep.sum = 2023) ∧
  (rep.map Nat.log2).sum = 48 ∧
  (∀ other_rep : List ℕ,
    (∀ i j, i ≠ j → other_rep.get! i ≠ other_rep.get! j) →
    other_rep.sum = 2023 →
    (other_rep.map Nat.log2).sum ≥ 48) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_rep_2023_l1129_112958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l1129_112969

noncomputable def binomial_expansion (x : ℝ) : ℝ := (1 / (2 * x) - Real.sqrt x) ^ 9

theorem constant_term_of_expansion :
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 →
    ∃ (p : ℝ → ℝ), binomial_expansion x = c + x * (p x)
    ∧ c = 21 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l1129_112969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_triangle_side_c_l1129_112956

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 - 1/2

def I : Set ℝ := Set.Icc (-Real.pi/12) (5*Real.pi/12)

theorem f_extrema :
  ∃ (max_x min_x : ℝ), max_x ∈ I ∧ min_x ∈ I ∧
  (∀ x ∈ I, f x ≤ f max_x) ∧
  (∀ x ∈ I, f x ≥ f min_x) ∧
  max_x = Real.pi/3 ∧ min_x = -Real.pi/12 := by sorry

theorem triangle_side_c :
  ∀ (A B C : ℝ) (a b c : ℝ),
  a = 2 * Real.sqrt 3 →
  b = 6 →
  f (A/2) = -1 →
  c = 2 * Real.sqrt 3 ∨ c = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_triangle_side_c_l1129_112956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_product_implies_divides_factor_l1129_112914

theorem prime_divides_product_implies_divides_factor
  (p : ℕ) (hp : Nat.Prime p) (a : List ℕ) :
  p ∣ a.prod → ∃ x ∈ a, p ∣ x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_product_implies_divides_factor_l1129_112914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_series_a_sum_series_b_sum_series_c_l1129_112975

-- Define the sum of an infinite geometric series
noncomputable def geometric_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

-- Theorem for the sum of the first series
theorem sum_series_a : geometric_sum (1/2) (1/2) = 1 := by
  -- Proof steps would go here
  sorry

-- Theorem for the sum of the second series
theorem sum_series_b : geometric_sum (1/3) (1/3) = 1/2 := by
  -- Proof steps would go here
  sorry

-- Theorem for the sum of the third series
theorem sum_series_c : geometric_sum (1/4) (1/4) = 1/3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_series_a_sum_series_b_sum_series_c_l1129_112975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tina_works_ten_hours_l1129_112987

/-- Tina's work schedule and pay structure -/
structure WorkSchedule where
  regularWage : ℚ
  overtimeThreshold : ℚ
  daysWorked : ℕ
  totalEarnings : ℚ

/-- Calculate total earnings based on hours worked per day -/
noncomputable def calculateEarnings (schedule : WorkSchedule) (hoursPerDay : ℚ) : ℚ :=
  let regularHours := min hoursPerDay schedule.overtimeThreshold
  let overtimeHours := max (hoursPerDay - schedule.overtimeThreshold) 0
  let regularPay := regularHours * schedule.regularWage
  let overtimePay := overtimeHours * (schedule.regularWage * (3/2))
  (regularPay + overtimePay) * schedule.daysWorked

/-- Theorem stating that Tina works 10 hours per day -/
theorem tina_works_ten_hours (schedule : WorkSchedule)
  (h1 : schedule.regularWage = 18)
  (h2 : schedule.overtimeThreshold = 8)
  (h3 : schedule.daysWorked = 5)
  (h4 : schedule.totalEarnings = 990)
  : ∃ (hoursPerDay : ℚ), hoursPerDay = 10 ∧ calculateEarnings schedule hoursPerDay = schedule.totalEarnings := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tina_works_ten_hours_l1129_112987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_roots_comparison_l1129_112997

/-- Define a sequence that represents the nested square roots -/
noncomputable def nestedRoots : ℕ → ℝ
  | 0 => 0
  | n + 1 => 
    if n % 2 = 0 then Real.sqrt (17 * nestedRoots n)
    else Real.sqrt (13 * nestedRoots n)

/-- The theorem stating that 17 * (13/17)^(1/3) is greater than the nested square root expression with 2018 roots -/
theorem nested_roots_comparison :
  17 * (13/17)^(1/3 : ℝ) > nestedRoots 2018 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_roots_comparison_l1129_112997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_1959_l1129_112953

theorem divisibility_by_1959 (n : ℕ) : ∃ k : ℤ, 5^(8*n) - 2^(4*n) * 7^(2*n) = 1959 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_1959_l1129_112953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersecting_line_l1129_112988

/-- Given a parabola and a line intersecting it, if the line from the origin to one intersection point is perpendicular to the line from the origin to the other intersection point, then the line has a specific equation. -/
theorem parabola_intersecting_line (k : ℝ) : 
  let parabola := λ x y : ℝ ↦ y^2 = 2*x
  let line := λ x y : ℝ ↦ y = k*x + 2
  let O := (0, 0)
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    parabola x₁ y₁ ∧ 
    parabola x₂ y₂ ∧
    line x₁ y₁ ∧ 
    line x₂ y₂ ∧
    (x₁ - 0) * (x₂ - 0) + (y₁ - 0) * (y₂ - 0) = 0 →
  k = -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersecting_line_l1129_112988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l1129_112935

/-- The distance between two parallel lines given by their general equations -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the parallel lines 3x + 4y - 9 = 0 and 6x + 8y + 2 = 0 is 11/5 -/
theorem distance_specific_parallel_lines :
  distance_between_parallel_lines 3 4 (-9) 2 = 11/5 := by
  sorry

-- Remove the #eval statement as it's causing issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l1129_112935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curl_A₁_curl_A₂_l1129_112932

open Real

-- Define the vector field A for part (a)
def A₁ (x y z : ℝ) : ℝ × ℝ × ℝ := (x, -z^2, y^2)

-- Define the vector field A for part (b)
def A₂ (x y z : ℝ) : ℝ × ℝ × ℝ := (y*z, x*z, x*y)

-- Define the curl operation
noncomputable def curl (f : ℝ → ℝ → ℝ → ℝ × ℝ × ℝ) : ℝ → ℝ → ℝ → ℝ × ℝ × ℝ :=
  λ x y z ↦ (
    (deriv (λ y ↦ (f x y z).2.2) y) - (deriv (λ z ↦ (f x y z).2.1) z),
    (deriv (λ z ↦ (f x y z).1) z) - (deriv (λ x ↦ (f x y z).2.2) x),
    (deriv (λ x ↦ (f x y z).2.1) x) - (deriv (λ y ↦ (f x y z).1) y)
  )

-- Theorem for part (a)
theorem curl_A₁ (x y z : ℝ) :
  curl A₁ x y z = (2*y + 2*z, 0, 0) := by sorry

-- Theorem for part (b)
theorem curl_A₂ (x y z : ℝ) :
  curl A₂ x y z = (0, 0, 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curl_A₁_curl_A₂_l1129_112932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_determinant_l1129_112960

def A : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

theorem rotation_matrix_determinant :
  ∀ (a b c d : ℝ),
  ((!![a, b; 1, 2] : Matrix (Fin 2) (Fin 2) ℝ) * A = !![3, 4; c, d]) →
  a * d - b * c = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_determinant_l1129_112960
