import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_25_term_l959_95920

theorem base_25_term (x y : ℝ) : 
  (5 : ℝ)^(x+1) * (4 : ℝ)^(y-1) = (25 : ℝ)^x * (64 : ℝ)^y ∧ x + y = 0.5 → (25 : ℝ)^x = (5 : ℝ)^(2*x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_25_term_l959_95920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_division_l959_95943

theorem cube_division (edge : ℕ) (N : ℕ) : edge = 5 →
  (∃ (sizes : List ℕ), 
    (∀ s, s ∈ sizes → s > 0 ∧ s ≤ edge) ∧ 
    (∃ a b, a ∈ sizes ∧ b ∈ sizes ∧ a ≠ b) ∧
    (List.sum (List.map (λ s => s^3) sizes) = edge^3) ∧
    (List.length sizes = N)) →
  N ≤ 118 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_division_l959_95943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l959_95917

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity_range 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (h_hyperbola : ∀ x y, x^2/a^2 - y^2/b^2 = 1 → (x, y) ∈ Set.range (λ p : ℝ × ℝ => p)) 
  (F₁ F₂ : ℝ × ℝ)
  (h_foci : ‖F₁ - F₂‖ = 2*c)
  (A : ℝ × ℝ)
  (h_A : A ∈ Set.range (λ p : ℝ × ℝ => p) ∧ A.1 = c ∧ A.2 > 0)
  (Q : ℝ × ℝ)
  (h_Q : Q = (c, 3*a/2))
  (h_F₂Q_gt_F₂A : ‖F₂ - Q‖ > ‖F₂ - A‖)
  (h_P : ∀ P : ℝ × ℝ, P ∈ Set.range (λ p : ℝ × ℝ => p) → P.1 > 0 → 
    ‖P - F₁‖ + ‖P - Q‖ > 3/2 * ‖F₁ - F₂‖) :
  let e := c/a
  1 < e ∧ e < 7/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l959_95917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_triangle_area_l959_95901

/-- Triangle properties -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def triangle_conditions (t : Triangle) : Prop :=
  t.a^2 - t.a * t.b - 2 * t.b^2 = 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.a / Real.sin t.A = t.c / Real.sin t.C

/-- Theorem 1: If B = π/6, then C = π/3 -/
theorem angle_C_value (t : Triangle) (h : triangle_conditions t) (hB : t.B = Real.pi/6) :
  t.C = Real.pi/3 := by sorry

/-- Theorem 2: If C = 2π/3 and c = 14, then the area is 28√3 -/
theorem triangle_area (t : Triangle) (h : triangle_conditions t) (hC : t.C = 2*Real.pi/3) (hc : t.c = 14) :
  (1/2) * t.a * t.b * Real.sin t.C = 28 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_triangle_area_l959_95901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_at_zero_l959_95923

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

-- State the theorem
theorem f_minimum_at_zero (a : ℝ) :
  (∀ x : ℝ, f a x ≥ f a 0) ↔ (0 ≤ a ∧ a ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_at_zero_l959_95923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_PQT_PTR_l959_95936

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  -- No specific conditions needed for the triangle

-- Define point T on side QR
def T_on_QR (P Q R T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ T = (1 - t) • Q + t • R

-- Define the lengths QT and TR
def QT_length (Q T : ℝ × ℝ) : ℝ := 6
def TR_length (T R : ℝ × ℝ) : ℝ := 10

-- Define the areas of triangles PQT and PTR
noncomputable def area_PQT (P Q T : ℝ × ℝ) : ℝ := sorry
noncomputable def area_PTR (P T R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio_PQT_PTR (P Q R T : ℝ × ℝ) 
  (tri : Triangle P Q R) 
  (t_on_qr : T_on_QR P Q R T) 
  (qt_len : QT_length Q T = 6) 
  (tr_len : TR_length T R = 10) : 
  (area_PQT P Q T) / (area_PTR P T R) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_PQT_PTR_l959_95936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_lead_calculation_l959_95948

/-- Represents the race scenario with two runners, Sunny and Windy -/
structure RaceScenario where
  a : ℝ  -- Length of the first race in meters
  e : ℝ  -- Distance Sunny finishes ahead of Windy in the first race
  sunny_speed_increase : ℝ  -- Sunny's speed increase in the second race

/-- Calculates Sunny's lead at the end of the second race -/
noncomputable def sunny_lead (scenario : RaceScenario) : ℝ :=
  (2 * scenario.e^2) / (scenario.a + scenario.e)

/-- Theorem stating that Sunny's lead at the end of the second race is 2e²/(a+e) meters -/
theorem sunny_lead_calculation (scenario : RaceScenario) 
  (h1 : scenario.a > 0)
  (h2 : scenario.e > 0)
  (h3 : scenario.sunny_speed_increase = 0.2) :
  sunny_lead scenario = (2 * scenario.e^2) / (scenario.a + scenario.e) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_lead_calculation_l959_95948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationship_theorem_l959_95903

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines the relationship between two lines -/
inductive LineRelationship
  | Intersecting
  | Perpendicular
  | Coincident
  | Parallel

/-- Determines the relationship between two lines based on the parameter a -/
noncomputable def lineRelationship (a : ℝ) : LineRelationship :=
  if a ≠ -1 ∧ a ≠ 2 then LineRelationship.Intersecting
  else if a = 2/3 then LineRelationship.Perpendicular
  else if a = 2 then LineRelationship.Coincident
  else if a = -1 then LineRelationship.Parallel
  else LineRelationship.Intersecting

theorem line_relationship_theorem (a : ℝ) :
  let l₁ : Line := ⟨a, 2, 6⟩
  let l₂ : Line := ⟨1, a-1, a^2-1⟩
  (lineRelationship a = LineRelationship.Intersecting → a ≠ -1 ∧ a ≠ 2) ∧
  (lineRelationship a = LineRelationship.Perpendicular → a = 2/3) ∧
  (lineRelationship a = LineRelationship.Coincident → a = 2) ∧
  (lineRelationship a = LineRelationship.Parallel → a = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationship_theorem_l959_95903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_g_l959_95969

-- Define the function f(x) = b + a * sin(x)
noncomputable def f (a b x : ℝ) : ℝ := b + a * Real.sin x

-- Define the function g(x) = tan((3a + b)x)
noncomputable def g (a b x : ℝ) : ℝ := Real.tan ((3*a + b) * x)

-- State the theorem
theorem smallest_positive_period_of_g (a b : ℝ) :
  a < 0 ∧ 
  (∀ x, f a b x ≤ -1) ∧
  (∃ x, f a b x = -1) ∧
  (∀ x, f a b x ≥ -5) ∧
  (∃ x, f a b x = -5) →
  (let period := Real.pi / 9
   ∀ x, g a b (x + period) = g a b x ∧
   ∀ p, 0 < p ∧ p < period → ∃ x, g a b (x + p) ≠ g a b x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_g_l959_95969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_beta_success_ratio_l959_95916

/-- Represents a participant's score in a two-day contest -/
structure ContestScore where
  day1_score : ℕ
  day1_attempted : ℕ
  day2_score : ℕ
  day2_attempted : ℕ

/-- The total score of a participant -/
def total_score (s : ContestScore) : ℕ := s.day1_score + s.day2_score

/-- The total points attempted by a participant -/
def total_attempted (s : ContestScore) : ℕ := s.day1_attempted + s.day2_attempted

/-- The success ratio of a participant for a given day -/
noncomputable def day_success_ratio (score : ℕ) (attempted : ℕ) : ℚ :=
  if attempted = 0 then 0 else (score : ℚ) / (attempted : ℚ)

/-- The overall success ratio of a participant -/
noncomputable def overall_success_ratio (s : ContestScore) : ℚ :=
  day_success_ratio (total_score s) (total_attempted s)

theorem max_beta_success_ratio (alpha beta : ContestScore) :
  alpha.day1_score = 192 →
  alpha.day1_attempted = 360 →
  alpha.day2_score = 168 →
  alpha.day2_attempted = 240 →
  beta.day1_score > 0 →
  beta.day2_score > 0 →
  day_success_ratio beta.day1_score beta.day1_attempted < day_success_ratio alpha.day1_score alpha.day1_attempted →
  day_success_ratio beta.day2_score beta.day2_attempted < day_success_ratio alpha.day2_score alpha.day2_attempted →
  total_attempted beta ≤ 550 →
  total_attempted beta ≤ 600 →
  overall_success_ratio beta ≤ 274 / 550 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_beta_success_ratio_l959_95916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_ratio_l959_95904

noncomputable section

open Real

structure Quadrilateral (A B C D : ℝ × ℝ) : Prop where
  right_angle_A : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0
  right_angle_C : (B.1 - C.1) * (D.1 - C.1) + (B.2 - C.2) * (D.2 - C.2) = 0

def similar_triangles (A B C D E F : ℝ × ℝ) : Prop :=
  let ratio := dist A B / dist D E
  dist A C / dist D F = ratio ∧ dist B C / dist E F = ratio

def point_interior (E A B C D : ℝ × ℝ) : Prop :=
  ∃ (α β γ δ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0 ∧ α + β + γ + δ = 1 ∧
  E = (α * A.1 + β * B.1 + γ * C.1 + δ * D.1, α * A.2 + β * B.2 + γ * C.2 + δ * D.2)

def area (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

theorem quadrilateral_ratio (A B C D E : ℝ × ℝ) 
  (h1 : Quadrilateral A B C D)
  (h2 : similar_triangles A B C A C D)
  (h3 : dist A B > dist A D)
  (h4 : point_interior E A B C D)
  (h5 : similar_triangles A B E C E D)
  (h6 : area B D E = 20 * area C E D) :
  dist A B^2 = (dist A D^2) / 399 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_ratio_l959_95904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l959_95996

theorem cube_root_equation_solution (x : ℝ) (h1 : x > 0) :
  (((1.5 - x^3) ^ (1/3 : ℝ)) + ((1.5 + x^3) ^ (1/3 : ℝ)) = 1) →
  x^3 = Real.sqrt (136/27) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l959_95996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l959_95977

theorem geometric_series_sum : ∀ (a r : ℝ) (n : ℕ),
  a > 0 → r > 1 →
  a * r^(n-1) = 2048 →
  (Finset.range n).sum (λ i => a * r^i) = 4095 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l959_95977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l959_95961

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ 3 then 2*x - x^2
  else if x ≥ -2 ∧ x ≤ 0 then x^2 + 6*x
  else 0  -- This else case is added to make the function total

theorem range_of_f :
  Set.range f = Set.Icc (-8 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l959_95961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coprime_integers_l959_95929

noncomputable def series_term (n : ℕ) : ℚ :=
  if n % 2 = 1 then
    (n^2 : ℚ) / (2^(n + 1) : ℚ)
  else
    (n^2 : ℚ) / (3^(3*n/2) : ℚ)

noncomputable def series_sum : ℚ := ∑' n, series_term n

theorem sum_of_coprime_integers (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : Nat.Coprime a b) 
  (h4 : (a : ℚ) / (b : ℚ) = series_sum) : 
  a + b = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coprime_integers_l959_95929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_difference_of_points_l959_95939

-- Define the curve
def curve (x y : ℝ) : Prop := y^2 + x^4 = 2 * x^2 * y + 1

-- Define the points
def point_a (a : ℝ) : Prop := curve (Real.sqrt (Real.exp 1)) a
def point_b (b : ℝ) : Prop := curve (Real.sqrt (Real.exp 1)) b

-- Theorem statement
theorem abs_difference_of_points (a b : ℝ) 
  (ha : point_a a) (hb : point_b b) (hab : a ≠ b) : 
  |a - b| = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_difference_of_points_l959_95939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l959_95914

/-- The curve function f(x) = e^x + 2 -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2

/-- The point P on the curve -/
def P : ℝ × ℝ := (0, 3)

/-- The tangent line equation: ax + by + c = 0 -/
structure TangentLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The tangent line to f at P is x - y + 3 = 0 -/
theorem tangent_line_at_P : 
  ∃ (t : TangentLine), t.a = 1 ∧ t.b = -1 ∧ t.c = 3 ∧
  (∀ (x y : ℝ), y = f x → (x = P.1 ∧ y = P.2) → t.a * x + t.b * y + t.c = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l959_95914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_ratio_l959_95994

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (P Q R : ℝ × ℝ) : ℝ := sorry

/-- Helper function to find the midpoint of a line segment -/
def Midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateral (P Q R : ℝ × ℝ) : Prop := sorry

/-- An equilateral triangle with midpoints and area ratio -/
theorem equilateral_triangle_area_ratio 
  (A B C D E F G H : ℝ × ℝ) 
  (h_equilateral : IsEquilateral A B C)
  (h_D : D = Midpoint A B)
  (h_E : E = Midpoint B C)
  (h_F : F = Midpoint C A)
  (h_G : G = Midpoint D F)
  (h_H : H = Midpoint F E) :
  (area_triangle F G H + (area_triangle D F E) / 2) / 
  (area_triangle A B C - (area_triangle F G H + (area_triangle D F E) / 2)) = 3 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_ratio_l959_95994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_l959_95913

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equality (n p : ℝ) (hn : 0 < n) (hp : 0 < p) :
  log10 (log10 (log10 ((10 : ℝ)^((10 : ℝ)^(n*p))))) = log10 (n*p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_l959_95913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_and_sin_alpha_l959_95989

/-- The function f(x) = sin(x) + 2cos(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * Real.cos x

/-- The maximum value of f(x) -/
noncomputable def f_max : ℝ := Real.sqrt 5

/-- Theorem stating the maximum value of f(x) and the value of sin(α) when f(x) is maximum -/
theorem f_maximum_and_sin_alpha :
  (∀ x : ℝ, f x ≤ f_max) ∧
  (∃ α : ℝ, f α = f_max ∧ Real.sin α = 1 / Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_and_sin_alpha_l959_95989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_work_time_l959_95926

/-- The number of days it takes Michael and Adam to complete the work together -/
noncomputable def total_days : ℝ := 20

/-- The number of days Michael and Adam work together before Michael stops -/
noncomputable def days_worked_together : ℝ := 16

/-- The number of days it takes Adam to complete the remaining work alone -/
noncomputable def adams_remaining_days : ℝ := 10

/-- The fraction of work completed by Michael and Adam working together in one day -/
noncomputable def combined_work_rate : ℝ := 1 / total_days

/-- The fraction of work completed by Adam in one day -/
noncomputable def adams_work_rate : ℝ := (1 - (combined_work_rate * days_worked_together)) / adams_remaining_days

/-- The number of days it would take Michael to complete the work alone -/
noncomputable def michaels_days : ℝ := 100 / 3

theorem michael_work_time :
  michaels_days = 1 / (combined_work_rate - adams_work_rate) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_work_time_l959_95926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_down_payment_contribution_exists_l959_95949

theorem car_down_payment_contribution_exists : ∃ (X Y Z : ℚ), 
  (0.35 * X + 0.25 * Y + 0.20 * Z ≤ 3500) ∧ 
  (⌊0.35 * X⌋ + ⌊0.25 * Y⌋ + ⌊0.20 * Z⌋ ≤ 3500) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_down_payment_contribution_exists_l959_95949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_sin_l959_95918

theorem axis_of_symmetry_sin (ω φ : ℝ) (h_ω : ω > 0) (h_φ : |φ| < π) :
  (∀ x, Real.sin (ω * x + φ) = Real.sin (2 * (x + π / 6))) →
  (∃ x₀, x₀ = π / 12 ∧ ∀ x, |x - x₀| < |x + x₀| → |x| > |x₀|) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_sin_l959_95918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l959_95950

/-- Represents a rhombus with given area and diagonal ratio -/
structure Rhombus where
  area : ℝ
  diagonalRatio : ℝ
  area_positive : area > 0
  ratio_positive : diagonalRatio > 0

/-- The length of the longer diagonal of a rhombus -/
noncomputable def longerDiagonal (r : Rhombus) : ℝ :=
  2 * Real.sqrt ((2 * r.area * r.diagonalRatio) / (1 + r.diagonalRatio))

/-- Theorem: For a rhombus with area 150 and diagonal ratio 4/3, the longer diagonal is 20 -/
theorem rhombus_longer_diagonal :
  let r : Rhombus := ⟨150, 4/3, by norm_num, by norm_num⟩
  longerDiagonal r = 20 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l959_95950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_equality_l959_95940

theorem triangle_angle_equality (A B C : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_equation : 1 - Real.cos A * Real.cos B - (Real.cos (C/2))^2 = 0) : 
  A = B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_equality_l959_95940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l959_95921

noncomputable def f (a b x : ℝ) : ℝ := a * (1/4)^(abs x) + b

theorem function_properties (a b : ℝ) :
  (∀ x, f a b x ≤ 3) ∧ 
  (∀ ε > 0, ∃ x, 3 - f a b x < ε) ∧ 
  f a b 0 = 0 →
  (a = -3 ∧ b = 3) ∧
  f a b (1/2) = 3/2 ∧
  (∀ x, 0 ≤ f a b x ∧ f a b x < 3) ∧
  (∀ x, f a b x > f a b (2*x+1) → -1 < x ∧ x < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l959_95921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_distance_walked_l959_95900

/-- Calculates the distance walked given the total travel time, walking speed, and bicycling speed -/
noncomputable def distance_walked (total_time : ℝ) (walking_speed : ℝ) (bicycling_speed : ℝ) : ℝ :=
  let total_distance := total_time * (walking_speed * bicycling_speed) / (walking_speed + bicycling_speed)
  total_distance / 2

theorem linda_distance_walked :
  let total_time : ℝ := 36 / 60  -- 36 minutes converted to hours
  let walking_speed : ℝ := 4     -- km/h
  let bicycling_speed : ℝ := 12  -- km/h
  distance_walked total_time walking_speed bicycling_speed = 1.8 := by
  sorry

-- Use a computable version for evaluation
def distance_walked_float (total_time : Float) (walking_speed : Float) (bicycling_speed : Float) : Float :=
  let total_distance := total_time * (walking_speed * bicycling_speed) / (walking_speed + bicycling_speed)
  total_distance / 2

#eval distance_walked_float 0.6 4 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_distance_walked_l959_95900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_range_l959_95908

-- Define a quadratic function
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_symmetry : ∀ x, f (x + 2) = f (2 - x)
axiom f_inequality : ∀ a, f a ≤ f 0 ∧ f 0 ≤ f 1

-- Theorem statement
theorem quadratic_range (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f (x + 2) = f (2 - x)) ∧ (f a ≤ f 0 ∧ f 0 ≤ f 1)) → 
  (a ≤ 0 ∨ a ≥ 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_range_l959_95908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_l959_95934

/-- Parabola type representing x^2 = 2py -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

noncomputable def Parabola.focus (c : Parabola) : Point :=
  ⟨0, c.p / 2⟩

noncomputable def Parabola.directrix (c : Parabola) : ℝ :=
  -c.p / 2

def Parabola.contains (c : Parabola) (p : Point) : Prop :=
  p.x^2 = 2 * c.p * p.y

theorem parabola_min_distance (c : Parabola) (p : Point) :
  c.focus = Point.mk 0 4 →
  p = Point.mk 2 3 →
  (∃ (m : Point), c.contains m ∧
    ∀ (n : Point), c.contains n →
      distance m c.focus + distance m p ≤ distance n c.focus + distance n p) →
  (∃ (m : Point), c.contains m ∧
    distance m c.focus + distance m p = 7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_l959_95934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_red_ball_is_three_l959_95942

/-- Represents a two-digit number between 01 and 33 -/
def RedBallNumber := {n : ℕ // 1 ≤ n ∧ n ≤ 33}

/-- The random number table -/
def randomTable : List (List ℕ) := [[2976, 3413, 2814, 2641], [8303, 9822, 5888, 2410]]

/-- Extracts two-digit numbers from a list of four-digit numbers -/
def extractTwoDigitNumbers (row : List ℕ) : List ℕ :=
  row.bind (λ n => [n / 100, n % 100])

/-- Selects valid red ball numbers from a list of two-digit numbers -/
def selectRedBallNumbers (numbers : List ℕ) : List RedBallNumber :=
  numbers.filterMap (λ n => if 1 ≤ n ∧ n ≤ 33 then some ⟨n, by sorry⟩ else none)

/-- Provide an instance of Inhabited for RedBallNumber -/
instance : Inhabited RedBallNumber := ⟨⟨1, by sorry⟩⟩

/-- The main theorem -/
theorem fourth_red_ball_is_three :
  let twoDigitNumbers := extractTwoDigitNumbers (randomTable.head!)
  let redBallNumbers := selectRedBallNumbers twoDigitNumbers
  (redBallNumbers.get! 3).val = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_red_ball_is_three_l959_95942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_externally_l959_95976

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 9
def C2 (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 4

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (2, -1)
def center2 : ℝ × ℝ := (-1, 3)
def radius1 : ℝ := 3
def radius2 : ℝ := 2

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ :=
  Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)

-- Theorem: The circles are tangent externally
theorem circles_tangent_externally :
  distance_between_centers = radius1 + radius2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_externally_l959_95976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_region_bisecting_line_slope_l959_95905

/-- F-shaped region in the xy-plane -/
structure FRegion where
  vertices : List (ℝ × ℝ) := [(0,0), (0,4), (4,4), (4,2), (7,2), (7,0)]

/-- Line through the origin with slope m -/
def Line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1}

/-- Area of a region -/
noncomputable def area (r : Set (ℝ × ℝ)) : ℝ := sorry

/-- The region above a line -/
def regionAbove (l : Set (ℝ × ℝ)) (r : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p ∈ r | ∃ (q : ℝ × ℝ), q ∈ l ∧ p.2 ≥ q.2}

/-- The region below a line -/
def regionBelow (l : Set (ℝ × ℝ)) (r : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p ∈ r | ∃ (q : ℝ × ℝ), q ∈ l ∧ p.2 ≤ q.2}

/-- Convert a list of points to a set -/
def listToSet (l : List (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p | p ∈ l}

/-- Theorem: The slope of the line dividing the F-shaped region into two equal areas is 5/8 -/
theorem f_region_bisecting_line_slope (f : FRegion) :
  ∃ (m : ℝ), m = 5/8 ∧
    area (regionAbove (Line m) (listToSet f.vertices)) =
    area (regionBelow (Line m) (listToSet f.vertices)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_region_bisecting_line_slope_l959_95905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equidistant_from_B_and_C_l959_95993

/-- The distance between two points in 3D space -/
noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- Point A coordinates -/
noncomputable def A : ℝ × ℝ × ℝ := (0, 11/6, 0)

/-- Point B coordinates -/
def B : ℝ × ℝ × ℝ := (-2, -4, 6)

/-- Point C coordinates -/
def C : ℝ × ℝ × ℝ := (7, 2, 5)

/-- Theorem stating that A is equidistant from B and C -/
theorem A_equidistant_from_B_and_C :
  distance A.1 A.2.1 A.2.2 B.1 B.2.1 B.2.2 =
  distance A.1 A.2.1 A.2.2 C.1 C.2.1 C.2.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equidistant_from_B_and_C_l959_95993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_production_growth_rate_l959_95951

/-- Represents the average monthly growth rate of mask production -/
def x : Real := sorry

/-- Initial production in February 2022 (in hundreds of thousands) -/
def initial_production : Real := 180

/-- Final production in April 2022 (in hundreds of thousands) -/
def final_production : Real := 461

/-- Number of months between February and April -/
def months_between : Nat := 2

/-- Theorem stating the correct equation for the growth rate -/
theorem mask_production_growth_rate : 
  initial_production * (1 + x) ^ months_between = final_production := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_production_growth_rate_l959_95951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_selection_for_difference_two_l959_95988

theorem min_selection_for_difference_two (n : ℕ) (h : n = 20) :
  ∃ k : ℕ, k ≤ n ∧
    (∀ S : Finset ℕ, S.card = k ∧ S ⊆ Finset.range n →
      ∃ a b, a ∈ S ∧ b ∈ S ∧ a - b = 2) ∧
    (∀ T : Finset ℕ, T.card < k ∧ T ⊆ Finset.range n →
      ∃ U : Finset ℕ, U.card = T.card ∧ U ⊆ Finset.range n ∧
        ∀ a b, a ∈ U → b ∈ U → a - b ≠ 2) ∧
  k = 11 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_selection_for_difference_two_l959_95988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_OAB_in_inscribed_decagon_is_18_degrees_l959_95980

/-- The measure of angle OAB in a regular decagon inscribed in a circle,
    where O is the center and A and B are adjacent vertices. -/
noncomputable def angle_OAB_in_inscribed_decagon : ℝ := 18

/-- The number of sides in a decagon. -/
def decagon_sides : ℕ := 10

/-- The measure of the interior angle of a regular decagon. -/
noncomputable def interior_angle_decagon : ℝ := ((decagon_sides - 2) * 180) / decagon_sides

/-- The measure of the central angle subtended by a side of the inscribed decagon. -/
noncomputable def central_angle_decagon : ℝ := interior_angle_decagon

theorem angle_OAB_in_inscribed_decagon_is_18_degrees :
  angle_OAB_in_inscribed_decagon = (180 - central_angle_decagon) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_OAB_in_inscribed_decagon_is_18_degrees_l959_95980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dataset_mean_correction_l959_95941

theorem dataset_mean_correction (initial_observations : ℕ) (initial_mean : ℝ)
  (missing_observations : ℕ) (incorrect_increments : ℕ) (incorrect_decrements : ℕ)
  (increment_difference : ℝ) (decrement_difference : ℝ) :
  let total_sum := initial_observations * initial_mean
  let adjusted_sum_missing := total_sum - (initial_mean * missing_observations)
  let adjusted_sum_increments := adjusted_sum_missing - (incorrect_increments * increment_difference)
  let adjusted_sum_decrements := adjusted_sum_increments + (incorrect_decrements * decrement_difference)
  let updated_observations := initial_observations - missing_observations
  let updated_mean := adjusted_sum_decrements / updated_observations
  initial_observations = 60 →
  initial_mean = 210 →
  missing_observations = 2 →
  incorrect_increments = 3 →
  incorrect_decrements = 6 →
  increment_difference = 10 →
  decrement_difference = 6 →
  abs (updated_mean - 210.1034) < 0.0001 :=
by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dataset_mean_correction_l959_95941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_regression_implies_zero_correlation_l959_95956

/-- The correlation coefficient between two variables -/
noncomputable def correlation_coefficient (X Y : Type) [NormedAddCommGroup X] [InnerProductSpace ℝ X] 
  [NormedAddCommGroup Y] [InnerProductSpace ℝ Y] : ℝ := 
  sorry

/-- The regression coefficient in simple linear regression -/
noncomputable def regression_coefficient (X Y : Type) [NormedAddCommGroup X] [InnerProductSpace ℝ X] 
  [NormedAddCommGroup Y] [InnerProductSpace ℝ Y] : ℝ := 
  sorry

/-- Theorem: If the regression coefficient is zero, then the correlation coefficient is also zero -/
theorem zero_regression_implies_zero_correlation 
  (X Y : Type) [NormedAddCommGroup X] [InnerProductSpace ℝ X] 
  [NormedAddCommGroup Y] [InnerProductSpace ℝ Y] :
  regression_coefficient X Y = 0 → correlation_coefficient X Y = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_regression_implies_zero_correlation_l959_95956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_matching_difference_l959_95958

/-- A circle with marked points and chords. -/
structure MarkedCircle where
  n : ℕ
  points : Fin (2 * n) → Point
  is_valid_chord : (Fin (2 * n) × Fin (2 * n)) → Prop

/-- A matching on a marked circle. -/
def Matching (mc : MarkedCircle) := 
  { m : Fin mc.n → Fin (2 * mc.n) × Fin (2 * mc.n) // 
    (∀ i j, i ≠ j → (m i).1 ≠ (m j).1 ∧ (m i).1 ≠ (m j).2 ∧ 
                    (m i).2 ≠ (m j).1 ∧ (m i).2 ≠ (m j).2) ∧
    (∀ i, mc.is_valid_chord (m i)) }

/-- The number of intersection points of chords in a matching. -/
noncomputable def num_intersections (mc : MarkedCircle) (m : Matching mc) : ℕ := sorry

/-- An even matching has an even number of intersection points. -/
def is_even_matching (mc : MarkedCircle) (m : Matching mc) : Prop :=
  Even (num_intersections mc m)

/-- An odd matching has an odd number of intersection points. -/
def is_odd_matching (mc : MarkedCircle) (m : Matching mc) : Prop :=
  Odd (num_intersections mc m)

/-- Assume that Matching mc is finite -/
instance (mc : MarkedCircle) : Fintype (Matching mc) := sorry

/-- Assume that is_even_matching is decidable -/
instance (mc : MarkedCircle) : DecidablePred (is_even_matching mc) := sorry

/-- Assume that is_odd_matching is decidable -/
instance (mc : MarkedCircle) : DecidablePred (is_odd_matching mc) := sorry

/-- The main theorem: the difference between even and odd matchings is always 1. -/
theorem even_odd_matching_difference (mc : MarkedCircle) :
  (Finset.filter (is_even_matching mc) (Finset.univ : Finset (Matching mc))).card -
  (Finset.filter (is_odd_matching mc) (Finset.univ : Finset (Matching mc))).card = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_matching_difference_l959_95958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_zero_l959_95947

theorem sin_alpha_zero (α β : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2))
  (h2 : β ∈ Set.Ioo (π/2) π)
  (h3 : Real.sin (α + β) = 3/5)
  (h4 : Real.cos β = -4/5) : 
  Real.sin α = 0 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_zero_l959_95947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_l959_95902

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

def binary_to_decimal (bits : List Nat) : Nat :=
  base_to_decimal bits 2

theorem smallest_number :
  let a := 85
  let b := base_to_decimal [2, 1, 0] 6
  let c := base_to_decimal [1, 0, 0, 0] 7
  let d := binary_to_decimal [1, 0, 1, 0, 1, 1]
  d < a ∧ d < b ∧ d < c := by
  sorry

#eval base_to_decimal [2, 1, 0] 6
#eval base_to_decimal [1, 0, 0, 0] 7
#eval binary_to_decimal [1, 0, 1, 0, 1, 1]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_l959_95902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_range_on_interval_l959_95967

noncomputable def f (x : ℝ) := Real.sin x

theorem sin_range_on_interval :
  Set.Icc (-π/4 : ℝ) (3*π/4) ⊆ f ⁻¹' (Set.Icc (-Real.sqrt 2 / 2) 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_range_on_interval_l959_95967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_door_opening_probability_l959_95953

/-- The number of keys -/
def total_keys : ℕ := 5

/-- The number of keys that can open the door -/
def working_keys : ℕ := 2

/-- The attempt number we're interested in -/
def target_attempt : ℕ := 3

/-- The probability of succeeding in opening the door exactly on the third attempt -/
def success_probability : ℚ := 1 / 5

theorem door_opening_probability :
  (Nat.choose total_keys target_attempt * Nat.choose working_keys 1 : ℚ) / 
  (Nat.choose total_keys target_attempt : ℚ) = success_probability := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_door_opening_probability_l959_95953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_path_length_approx_l959_95907

-- Define the given constants
noncomputable def α : ℝ := 42 + 21 / 60 + 13 / 3600  -- Convert angle to decimal degrees
def a : ℝ := 10  -- Time delay in seconds
def b : ℝ := 2.5  -- Thunder duration in seconds
def v : ℝ := 333  -- Speed of sound in m/s

-- Define the function to calculate the lightning path length
noncomputable def lightning_path_length (α a b v : ℝ) : ℝ :=
  let AB := v * a
  let AC := v * (a + b)
  let cos_α := Real.cos (α * Real.pi / 180)  -- Convert degrees to radians
  Real.sqrt (AB^2 + AC^2 - 2 * AB * AC * cos_α)

-- State the theorem
theorem lightning_path_length_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (lightning_path_length α a b v - 2815.75) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_path_length_approx_l959_95907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_half_l959_95992

noncomputable def f (x : ℝ) : ℝ := 1 - Real.sin x + Real.log ((1 - x) / (1 + x)) / Real.log 5

theorem f_sum_half : f (1/2) + f (-1/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_half_l959_95992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l959_95933

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) : ℕ → ℕ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

theorem sixth_term_value (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) :
  a 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l959_95933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l959_95960

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 2, 1; 2, 0, 2; 1, 2, 0]

theorem matrix_equation_solution :
  let s : ℝ := -10
  let t : ℝ := -8
  let u : ℝ := -36
  B^3 + s • B^2 + t • B + u • (1 : Matrix (Fin 3) (Fin 3) ℝ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l959_95960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_j_in_terms_of_h_l959_95910

-- Define the domain of h
def DomainH : Set ℝ := { x | -4 ≤ x ∧ x ≤ 4 }

-- Define h as a function on its domain
def h : DomainH → ℝ := sorry

-- Define j as a function derived from h
def j (x : ℝ) : ℝ := h ⟨6 - x, sorry⟩

-- Theorem statement
theorem j_in_terms_of_h : 
  ∀ x, j x = h ⟨6 - x, sorry⟩ := by
  intro x
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_j_in_terms_of_h_l959_95910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_is_simplest_l959_95962

-- Define the square roots
noncomputable def sqrt_3 : ℝ := Real.sqrt 3
noncomputable def sqrt_0_5 : ℝ := Real.sqrt 0.5
noncomputable def sqrt_2_5 : ℝ := Real.sqrt (2/5)
noncomputable def sqrt_8 : ℝ := Real.sqrt 8

-- Define a function to check if a square root is in its simplest form
def is_simplest (x : ℝ) : Prop := ∀ y : ℝ, y^2 = x → y = Real.sqrt x

-- Theorem statement
theorem sqrt_3_is_simplest :
  is_simplest sqrt_3 ∧
  ¬(is_simplest sqrt_0_5) ∧
  ¬(is_simplest sqrt_2_5) ∧
  ¬(is_simplest sqrt_8) := by
  sorry

#check sqrt_3_is_simplest

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_is_simplest_l959_95962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l959_95983

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x + x^2

/-- The theorem stating the solution set of the inequality -/
theorem inequality_solution_set :
  {x : ℝ | f (Real.log x) + f (Real.log (1/x)) < 2 * f 1} = Set.Ioo (Real.exp (-1)) (Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l959_95983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_squared_plus_b_squared_l959_95932

/-- A function f(x) = e^x + ax + b with a root in [1,3] -/
noncomputable def f (a b : ℝ) : ℝ → ℝ := fun x ↦ Real.exp x + a * x + b

/-- The existence of a root in [1,3] -/
def has_root_in_interval (a b : ℝ) : Prop :=
  ∃ t : ℝ, t ∈ Set.Icc 1 3 ∧ f a b t = 0

/-- The minimum value of a^2 + b^2 is e^2 / 2 -/
theorem min_value_of_a_squared_plus_b_squared (a b : ℝ) 
  (h : has_root_in_interval a b) : 
  a^2 + b^2 ≥ Real.exp 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_squared_plus_b_squared_l959_95932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_file_size_is_90MB_l959_95909

/-- Represents the download characteristics of a file -/
structure DownloadInfo where
  initial_rate : ℝ  -- Initial download rate in MB/s
  initial_size : ℝ  -- Size at which the rate changes in MB
  final_rate : ℝ    -- Final download rate in MB/s
  total_time : ℝ    -- Total download time in seconds

/-- Calculates the total size of a file given its download information -/
noncomputable def calculateFileSize (info : DownloadInfo) : ℝ :=
  let initial_time := info.initial_size / info.initial_rate
  let remaining_time := info.total_time - initial_time
  info.initial_size + info.final_rate * remaining_time

/-- Theorem stating that a file with given download characteristics has a size of 90 MB -/
theorem file_size_is_90MB (info : DownloadInfo) 
    (h1 : info.initial_rate = 5)
    (h2 : info.initial_size = 60)
    (h3 : info.final_rate = 10)
    (h4 : info.total_time = 15) :
    calculateFileSize info = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_file_size_is_90MB_l959_95909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_condition_l959_95931

theorem no_function_satisfies_condition : ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = x - y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_condition_l959_95931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_matrix_l959_95964

def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

theorem transformation_matrix :
  (rotation_matrix * dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ) = !![0, -2; 2, 0] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_matrix_l959_95964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_a1_values_l959_95975

/-- An arithmetic progression with integer terms -/
structure ArithmeticProgression where
  a : ℕ → ℤ
  d : ℤ
  is_ap : ∀ n, a (n + 1) = a n + d
  is_increasing : d > 0

/-- The sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℤ :=
  (n * (2 * ap.a 1 + (n - 1) * ap.d)) / 2

/-- The theorem stating the possible values of a₁ given the conditions -/
theorem possible_a1_values (ap : ArithmeticProgression) :
  let S := sum_n ap 7
  (ap.a 7 * ap.a 12 > S + 20 ∧ ap.a 9 * ap.a 10 < S + 44) →
  ap.a 1 ∈ ({-9, -8, -7, -6, -4, -3, -2, -1} : Set ℤ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_a1_values_l959_95975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_pair_equal_l959_95970

noncomputable section

-- Define the four pairs of functions
def f1_1 (x : ℝ) : ℝ := (x + 3) * (x - 5) / (x + 3)
def f1_2 (x : ℝ) : ℝ := x - 5

def f2_1 (x : ℝ) : ℝ := Real.sqrt (x + 1) * Real.sqrt (x - 1)
def f2_2 (x : ℝ) : ℝ := Real.sqrt ((x + 1) * (x - 1))

def f3_1 (x : ℝ) : ℝ := x
def f3_2 (x : ℝ) : ℝ := Real.sqrt (x^2)

def f4_1 (x : ℝ) : ℝ := x * (x - 1)^(1/3)
def f4_2 (x : ℝ) : ℝ := x * (x - 1)^(1/3)

end noncomputable section

-- Define a predicate for function equality
def FunctionEqual (f g : ℝ → ℝ) : Prop :=
  (∀ x, f x = g x) ∧ (∀ x, (∃ y, f x = y) ↔ (∃ y, g x = y))

-- Theorem statement
theorem exactly_one_pair_equal :
  (FunctionEqual f1_1 f1_2 ∧ ¬FunctionEqual f2_1 f2_2 ∧ ¬FunctionEqual f3_1 f3_2 ∧ FunctionEqual f4_1 f4_2) ∨
  (¬FunctionEqual f1_1 f1_2 ∧ FunctionEqual f2_1 f2_2 ∧ ¬FunctionEqual f3_1 f3_2 ∧ ¬FunctionEqual f4_1 f4_2) ∨
  (¬FunctionEqual f1_1 f1_2 ∧ ¬FunctionEqual f2_1 f2_2 ∧ FunctionEqual f3_1 f3_2 ∧ ¬FunctionEqual f4_1 f4_2) ∨
  (¬FunctionEqual f1_1 f1_2 ∧ ¬FunctionEqual f2_1 f2_2 ∧ ¬FunctionEqual f3_1 f3_2 ∧ FunctionEqual f4_1 f4_2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_pair_equal_l959_95970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_search_theorem_l959_95925

/-- Represents a grid cell -/
structure Cell where
  row : ℕ
  col : ℕ

/-- Represents the grid and the numbers written on it -/
def Grid (n : ℕ) := Cell → ℕ

/-- Predicate to check if two cells share a common side -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ (c1.col = c2.col + 1 ∨ c2.col = c1.col + 1)) ∨
  (c1.col = c2.col ∧ (c1.row = c2.row + 1 ∨ c2.row = c1.row + 1))

/-- The main theorem to be proved -/
theorem grid_search_theorem (k : ℕ) (hk : k ≥ 1) :
  let n := 2^k - 1
  ∀ (grid : Grid n),
    (∀ i, 1 ≤ i ∧ i < n^2 →
      ∃ c1 c2 : Cell, grid c1 = i ∧ grid c2 = i+1 ∧ adjacent c1 c2) →
    ∃ (strategy : ℕ → Cell),
      (∃ t < 3*n, grid (strategy t) = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_search_theorem_l959_95925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_remaining_l959_95972

noncomputable def initial_rope_length : ℝ := 60

noncomputable def remaining_after_allan (total : ℝ) : ℝ := total * (2/3)

noncomputable def remaining_after_jack (after_allan : ℝ) : ℝ := after_allan * (1/2)

noncomputable def remaining_after_maria (after_jack : ℝ) : ℝ := after_jack * (1/4)

noncomputable def remaining_after_mike (after_maria : ℝ) : ℝ := after_maria * (4/5)

theorem rope_remaining :
  remaining_after_mike (remaining_after_maria (remaining_after_jack (remaining_after_allan initial_rope_length))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_remaining_l959_95972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_theorem_l959_95946

-- Function 1
noncomputable def f1 (x : ℝ) : ℝ := (Real.sqrt (3 - x)) / (x + 1) + (x - 2)^0

-- Function 2
noncomputable def f2 (x : ℝ) : ℝ := (Real.sqrt (x + 4) + Real.sqrt (1 - x)) / x

-- Domain of f1
def domain_f1 : Set ℝ := {x | x < -1 ∨ (-1 < x ∧ x < 2) ∨ (2 < x ∧ x ≤ 3)}

-- Domain of f2
def domain_f2 : Set ℝ := {x | -4 ≤ x ∧ x < 0} ∪ {x | 0 < x ∧ x ≤ 1}

theorem domain_theorem :
  (∀ x, x ∈ domain_f1 ↔ (3 - x ≥ 0 ∧ x + 1 ≠ 0 ∧ x - 2 ≠ 0)) ∧
  (∀ x, x ∈ domain_f2 ↔ (x + 4 ≥ 0 ∧ 1 - x ≥ 0 ∧ x ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_theorem_l959_95946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l959_95927

/-- The function g as defined in the problem -/
noncomputable def g (a b c : ℝ) : ℝ := a / (a + b) + b / (b + c) + c / (c + a)

/-- The theorem stating the range of g -/
theorem g_range (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 1 2 ∧ g a b c = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l959_95927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l959_95963

noncomputable def f (x : ℝ) : ℝ := (15 * x^5 + 7 * x^3 + 10 * x^2 + 6 * x + 2) / (4 * x^5 + 3 * x^3 + 11 * x^2 + 4 * x + 1)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N : ℝ, ∀ x > N, |f x - 15/4| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l959_95963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_purchase_amount_l959_95991

/-- Represents the discount policy of the supermarket -/
noncomputable def discount (amount : ℝ) : ℝ :=
  if amount ≥ 200 ∧ amount < 500 then 0.05
  else if amount ≥ 500 then 0.1
  else 0

/-- Represents the total discount when combining purchases -/
noncomputable def combinedDiscount (amounts : List ℝ) : ℝ :=
  discount (amounts.sum) * amounts.sum

/-- Represents the total discount when purchasing separately -/
noncomputable def separateDiscount (amounts : List ℝ) : ℝ :=
  amounts.map (λ a => discount a * a) |>.sum

theorem second_purchase_amount (x : ℝ) :
  let first := (5/8) * x
  let second := 270 - (5/8) * x
  let third := x
  200 ≤ first + second ∧ first + second < 500 →
  combinedDiscount [first, second] - separateDiscount [first, second] = 13.5 →
  combinedDiscount [first, second, third] - separateDiscount [first, second, third] = 39.4 →
  230 ≤ x ∧ x < 320 →
  second = 115 := by
  sorry

#check second_purchase_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_purchase_amount_l959_95991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l959_95912

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the problem conditions
def parallel_vectors (t : Triangle) : Prop :=
  t.a * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.A

-- Theorem statement
theorem triangle_problem (t : Triangle) 
  (h1 : parallel_vectors t) 
  (h2 : t.a = Real.sqrt 7) 
  (h3 : t.b = 2) : 
  t.A = π / 3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l959_95912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_polynomial_bound_l959_95982

open Real Set

noncomputable def T (n : ℕ) (x : ℝ) : ℝ := (1 / 2^n) * ((x + Real.sqrt (1 - x^2))^n + (x - Real.sqrt (1 - x^2))^n)

theorem monic_polynomial_bound (n : ℕ) (p : Polynomial ℝ) :
  Polynomial.Monic p →
  Polynomial.degree p = n →
  (∀ x : ℝ, x ∈ Icc (-1) 1 → p.eval x > -(1 / 2^(n-1))) →
  ∃ x₀ : ℝ, x₀ ∈ Icc (-1) 1 ∧ p.eval x₀ ≥ 1 / 2^(n-1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_polynomial_bound_l959_95982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_color_change_interval_is_ten_minutes_l959_95924

/-- Represents the duration of the sunset in hours -/
def sunset_duration : ℚ := 2

/-- Represents the number of color changes during the sunset -/
def color_changes : ℕ := 12

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the time interval between color changes -/
noncomputable def color_change_interval : ℚ :=
  (sunset_duration * minutes_per_hour) / color_changes

theorem color_change_interval_is_ten_minutes :
  color_change_interval = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_color_change_interval_is_ten_minutes_l959_95924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_vector_product_l959_95984

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  let side_length := 2
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = side_length^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = side_length^2 ∧
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = side_length^2 ∧
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = side_length^2

-- Define the midpoint E of CD
def Midpoint (C D E : ℝ × ℝ) : Prop :=
  E.1 = (C.1 + D.1) / 2 ∧ E.2 = (C.2 + D.2) / 2

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Theorem statement
theorem square_vector_product (A B C D E : ℝ × ℝ) :
  Square A B C D → Midpoint C D E →
  dot_product (E.1 - A.1, E.2 - A.2) (D.1 - B.1, D.2 - B.2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_vector_product_l959_95984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_properties_l959_95911

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

-- Define a rhombus with vertices on the ellipse and diagonals intersecting at the origin
def rhombus_on_ellipse (A B C D : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ellipse C.1 C.2 ∧ ellipse D.1 D.2 ∧
  (A.1 + C.1 = 0) ∧ (A.2 + C.2 = 0) ∧ (B.1 + D.1 = 0) ∧ (B.2 + D.2 = 0)

-- Define the area of a rhombus
noncomputable def rhombus_area (A B C D : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) * Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) / 2

-- Define the circle
def fixed_circle (x y : ℝ) : Prop := x^2 + y^2 = 9/10

-- Theorem statement
theorem rhombus_properties :
  (∀ A B C D : ℝ × ℝ, rhombus_on_ellipse A B C D →
    rhombus_area A B C D ≥ 18/5) ∧
  (∀ A B C D : ℝ × ℝ, rhombus_on_ellipse A B C D →
    ∃ P : ℝ × ℝ, fixed_circle P.1 P.2 ∧
      (P.1 - A.1)^2 + (P.2 - A.2)^2 = (9/10)^2 ∧
      (P.1 - B.1)^2 + (P.2 - B.2)^2 = (9/10)^2 ∧
      (P.1 - C.1)^2 + (P.2 - C.2)^2 = (9/10)^2 ∧
      (P.1 - D.1)^2 + (P.2 - D.2)^2 = (9/10)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_properties_l959_95911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_angle_relation_max_sum_sqrt_linear_function_arithmetic_sequence_l959_95990

-- Proposition 1
theorem sine_angle_relation (A B : ℝ) (hsin : Real.sin A > Real.sin B) : A > B := by sorry

-- Proposition 2
theorem max_sum_sqrt (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 4 → Real.sqrt (x + 3) + Real.sqrt (y + 2) ≤ Real.sqrt (a + 3) + Real.sqrt (b + 2)) ∧
  Real.sqrt (a + 3) + Real.sqrt (b + 2) = 3 * Real.sqrt 2 := by sorry

-- Proposition 3
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem linear_function_arithmetic_sequence (f : ℝ → ℝ) (hf : ∃ m b : ℝ, ∀ x, f x = m * x + b) :
  IsArithmeticSequence (λ n => f (n : ℝ)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_angle_relation_max_sum_sqrt_linear_function_arithmetic_sequence_l959_95990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_intersection_implies_m_range_l959_95928

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 1/x - m/x^2 - x/3

theorem one_intersection_implies_m_range (m : ℝ) :
  (∃! x, f m x = 0) → m ∈ Set.Iic 0 ∪ {2/3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_intersection_implies_m_range_l959_95928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_to_line_l959_95979

/-- Theorem: Parabola tangent to line at specific point --/
theorem parabola_tangent_to_line (p q : ℝ) : 
  (∀ x, x^2 + p*x + q = 2*x - 3 → x = 2) ∧ 
  (2^2 + p*2 + q = 1) ∧
  (2*2 + p = 2) → 
  p = -2 ∧ q = 1 := by
  sorry

#check parabola_tangent_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_to_line_l959_95979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l959_95974

/-- The circle with center (-4, -1) and radius 3 -/
def myCircle (x y : ℝ) : Prop := (x + 4)^2 + (y + 1)^2 = 9

/-- Point P -/
def P : ℝ × ℝ := (2, 3)

/-- A point is on the circle -/
def onCircle (p : ℝ × ℝ) : Prop := myCircle p.1 p.2

/-- Line equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point is on a line -/
def onLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Tangent line to the circle at a point -/
def isTangent (l : Line) (p : ℝ × ℝ) : Prop :=
  onCircle p ∧ onLine p l ∧ ∀ q, onCircle q → onLine q l → q = p

theorem tangent_line_equation :
  ∃ A B : ℝ × ℝ,
    onCircle A ∧ onCircle B ∧
    (∃ lPA lPB : Line, isTangent lPA A ∧ isTangent lPB B) →
    onLine A (Line.mk 6 4 19) ∧
    onLine B (Line.mk 6 4 19) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l959_95974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_sum_l959_95922

/-- Given two unit vectors a and b with an angle of 60° between them, 
    the magnitude of 3a + b is sqrt(13). -/
theorem magnitude_of_vector_sum (a b : ℝ × ℝ) : 
  (‖a‖ = 1) → 
  (‖b‖ = 1) → 
  (a • b = 1/2) → 
  ‖3 • a + b‖ = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_sum_l959_95922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beaver_to_moose_ratio_l959_95919

def canada_population (moose beaver human : ℕ) : Prop :=
  ∃ (k : ℕ), k > 0 ∧ beaver = k * moose ∧ human = 19 * beaver

theorem beaver_to_moose_ratio :
  ∀ (moose beaver human : ℕ),
  canada_population moose beaver human →
  human = 38000000 →
  moose = 1000000 →
  (beaver : ℚ) / (moose : ℚ) = 2 := by
  sorry

#check beaver_to_moose_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beaver_to_moose_ratio_l959_95919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_factors_with_small_difference_l959_95915

theorem no_factors_with_small_difference (n : ℕ) : 
  ¬∃ (a b : ℕ), (n^2 + 3*n + 3 : ℕ) = a * b ∧ (a : ℝ) - b < 2 * Real.sqrt (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_factors_with_small_difference_l959_95915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_quadratic_l959_95971

noncomputable def root1 (p : ℝ) : ℝ := p + Real.sqrt (4 - p^2)
noncomputable def root2 (p : ℝ) : ℝ := p - Real.sqrt (4 - p^2)

theorem root_difference_quadratic (p : ℝ) : 
  let r := max (root1 p) (root2 p)
  let s := min (root1 p) (root2 p)
  r - s = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_quadratic_l959_95971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_salary_calculation_l959_95987

def initial_salary : ℚ := 30000
def raise_percentage : ℚ := 1 / 10
def tax_rate : ℚ := 13 / 100

def calculate_new_salary (initial : ℚ) (raise : ℚ) (tax : ℚ) : ℤ :=
  let after_raise := initial * (1 + raise)
  let before_tax := after_raise / (1 - tax)
  Int.floor (before_tax + 1/2)

theorem new_salary_calculation :
  calculate_new_salary initial_salary raise_percentage tax_rate = 37931 := by
  sorry

#eval calculate_new_salary initial_salary raise_percentage tax_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_salary_calculation_l959_95987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l959_95938

-- Define the triangle ABC
def triangle_ABC (a : ℝ) : ℝ × ℝ × ℝ := (a, a + 1, a + 2)

-- Define the sine law condition
def sine_law_condition (a : ℝ) (A C : ℝ) : Prop :=
  2 * Real.sin C = 3 * Real.sin A

-- Define the area of the triangle
noncomputable def triangle_area (a : ℝ) : ℝ := (15 * Real.sqrt 7) / 4

-- Define the condition for an obtuse triangle
def is_obtuse_triangle (a : ℝ) : Prop :=
  let (x, y, z) := triangle_ABC a
  (x^2 + y^2 < z^2) ∨ (x^2 + z^2 < y^2) ∨ (y^2 + z^2 < x^2)

theorem triangle_properties :
  ∃ (a : ℝ) (A B C : ℝ),
    (triangle_ABC a).1 > 0 ∧
    sine_law_condition a A C ∧
    triangle_area a = (15 * Real.sqrt 7) / 4 ∧
    is_obtuse_triangle a ∧
    a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l959_95938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_profit_l959_95986

/-- Represents the financial details of an electronic item --/
structure ElectronicItem where
  costPrice : ℝ
  sellingPrice : ℝ
  taxRate : ℝ
  discountRate : ℝ
  shippingCost : ℝ

/-- Calculates the final selling price after applying tax, discount, and shipping cost --/
def finalSellingPrice (item : ElectronicItem) : ℝ :=
  item.sellingPrice * (1 + item.taxRate - item.discountRate) + item.shippingCost

/-- Theorem: The store owner makes a profit of 408 Rs --/
theorem store_profit : 
  let radio1 : ElectronicItem := { costPrice := 490, sellingPrice := 465.50, taxRate := 0, discountRate := 0, shippingCost := 0 }
  let tv : ElectronicItem := { costPrice := 12000, sellingPrice := 11400, taxRate := 0.1, discountRate := 0, shippingCost := 0 }
  let speaker : ElectronicItem := { costPrice := 1200, sellingPrice := 1150, taxRate := 0, discountRate := 0.05, shippingCost := 0 }
  let radio2 : ElectronicItem := { costPrice := 600, sellingPrice := 550, taxRate := 0, discountRate := 0, shippingCost := 50 }
  let items := [radio1, tv, speaker, radio2]
  (items.map finalSellingPrice).sum - (items.map (·.costPrice)).sum = 408 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_profit_l959_95986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dislike_but_enjoy_approx_40_69_percent_l959_95935

/-- Represents the percentage of students who enjoy dancing -/
noncomputable def enjoy_dancing : ℝ := 0.7

/-- Represents the percentage of students who dislike dancing -/
noncomputable def dislike_dancing : ℝ := 1 - enjoy_dancing

/-- Represents the percentage of students who enjoy dancing and honestly claim they like it -/
noncomputable def honest_enjoy : ℝ := 0.75

/-- Represents the percentage of students who dislike dancing and correctly state they dislike it -/
noncomputable def honest_dislike : ℝ := 0.85

/-- Calculates the percentage of students who state they dislike dancing but actually enjoy it -/
noncomputable def dislike_but_enjoy : ℝ :=
  (enjoy_dancing * (1 - honest_enjoy)) / 
  (enjoy_dancing * (1 - honest_enjoy) + dislike_dancing * honest_dislike)

theorem dislike_but_enjoy_approx_40_69_percent :
  ∃ ε > 0, |dislike_but_enjoy - 0.4069| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dislike_but_enjoy_approx_40_69_percent_l959_95935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_extrema_f_non_negative_condition_l959_95999

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (a - 1) / x - Real.log x - 1

-- Part I
theorem f_monotonicity_and_extrema (h : 0 < x) :
  let f₁ := f 1
  (x < 1 → (deriv f₁) x < 0) ∧
  (1 < x → 0 < (deriv f₁) x) ∧
  (∀ y > 0, f₁ y ≥ f₁ 1) ∧
  (f₁ 1 = 0) ∧
  (∀ M : ℝ, ∃ y > 0, M < f₁ y) :=
by sorry

-- Part II
theorem f_non_negative_condition (h : 0 < a) :
  (∀ x ≥ 1, 0 ≤ f a x) ↔ 1 ≤ a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_extrema_f_non_negative_condition_l959_95999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_side_b_value_l959_95981

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_acute_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def sine_rule_condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a = 2 * t.b * Real.sin t.A

def area_condition (t : Triangle) : Prop :=
  1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 3

def side_condition (t : Triangle) : Prop :=
  t.a^2 + t.c^2 = 7

-- State the theorems
theorem angle_B_value (t : Triangle) 
  (h1 : is_acute_triangle t) 
  (h2 : sine_rule_condition t) : 
  t.B = Real.pi/3 := by sorry

theorem side_b_value (t : Triangle) 
  (h1 : is_acute_triangle t) 
  (h2 : sine_rule_condition t)
  (h3 : area_condition t)
  (h4 : side_condition t) : 
  t.b = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_side_b_value_l959_95981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l959_95955

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- An ellipse in the 2D plane -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- A line in the 2D plane -/
structure Line where
  m : ℝ
  c : ℝ

/-- Check if a point is inside an ellipse -/
def isInside (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 < 1

/-- Check if a point is on an ellipse -/
def isOn (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if a point is on a line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.c

/-- Check if a line intersects an ellipse at two points -/
def intersectsAtTwoPoints (l : Line) (e : Ellipse) : Prop :=
  ∃ (p1 p2 : Point), p1 ≠ p2 ∧ isOn p1 e ∧ isOn p2 e ∧ isOnLine p1 l ∧ isOnLine p2 l

/-- Check if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (p1 p2 : Point) : Prop :=
  m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2

theorem line_equation (e : Ellipse) (m : Point) (l : Line) :
  e.a = 2 ∧ e.b = Real.sqrt 3 ∧
  m.x = 1 ∧ m.y = 1 ∧
  isInside m e ∧
  intersectsAtTwoPoints l e ∧
  (∃ (p1 p2 : Point), isOn p1 e ∧ isOn p2 e ∧ isOnLine p1 l ∧ isOnLine p2 l ∧ isMidpoint m p1 p2) →
  ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = -7 ∧ ∀ (x y : ℝ), isOnLine (Point.mk x y) l ↔ a * x + b * y + c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l959_95955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_h_l959_95978

/-- Given a function h such that h(3x + 2) = 5x - 6 for all x,
    prove that the unique solution to h(x) = x is x = 14 -/
theorem unique_fixed_point_of_h (h : ℝ → ℝ) 
    (h_def : ∀ x, h (3*x + 2) = 5*x - 6) :
    (∃! x, h x = x) ∧ (∀ x, h x = x → x = 14) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_h_l959_95978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orphanage_parent_assignment_l959_95973

structure Orphanage :=
  (V : Type*)
  (E : V → V → Prop)
  (symmetric : ∀ x y, E x y → E y x)
  (irreflexive : ∀ x, ¬E x x)
  (friend_enemy_condition : ∀ x a b c, E x a → E x b → E x c → 
    (¬E a b ∧ ¬E b c ∧ E a c) ∨ (¬E a b ∧ E b c ∧ ¬E a c) ∨ (E a b ∧ ¬E b c ∧ ¬E a c) ∨ (E a b ∧ E b c ∧ E a c))

def valid_parent_assignment (O : Orphanage) (P : O.V → Set (Type*)) : Prop :=
  (∀ x, ∃ p1 p2, P x = {p1, p2} ∧ p1 ≠ p2) ∧ 
  (∀ x y, O.E x y → ∃! p, p ∈ P x ∧ p ∈ P y) ∧
  (∀ x y, ¬O.E x y → ∀ p, ¬(p ∈ P x ∧ p ∈ P y)) ∧
  (∀ p1 p2 p3 x y z, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → 
    ¬(p1 ∈ P x ∧ p2 ∈ P x ∧ p2 ∈ P y ∧ p3 ∈ P y ∧ p3 ∈ P z ∧ p1 ∈ P z))

theorem orphanage_parent_assignment (O : Orphanage) : 
  ∃ P : O.V → Set (Type*), valid_parent_assignment O P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orphanage_parent_assignment_l959_95973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_line_segment_l959_95957

def point1 : ℝ × ℝ := (2, 1)
def point2 : ℝ × ℝ := (-2, -2)

noncomputable def distance_sum (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - point1.1)^2 + (p.2 - point1.2)^2) +
  Real.sqrt ((p.1 - point2.1)^2 + (p.2 - point2.2)^2)

def locus := {p : ℝ × ℝ | distance_sum p = 5}

theorem locus_is_line_segment :
  ∃ (a b : ℝ × ℝ), locus = {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • a + t • b} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_line_segment_l959_95957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l959_95998

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 5*x + 6) / Real.log (1/2)

def domain : Set ℝ := {x | x < 2 ∨ x > 3}

theorem f_monotone_decreasing :
  ∀ x₁ x₂, x₁ ∈ domain → x₂ ∈ domain → x₁ < x₂ → x₁ > 3 → x₂ > 3 → f x₁ > f x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l959_95998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_operations_l959_95944

theorem polynomial_operations (x y : ℝ) : 
  let p₁ := x
  let p₂ := 2*x + y
  let M (n : ℕ) := (1/2 + 2^(n-1))*(3*x + y)
  (∃ (polynomials : List ℝ), polynomials.length = 9) ∧ 
  (x = 1 ∧ y = 2 → M 4 = 42.5) ∧
  (3*x + y = 1 → M 13 - M 11 = 3072) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_operations_l959_95944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quentin_finishes_first_l959_95945

/-- Represents the area of a lawn -/
structure LawnArea where
  size : ℝ
  size_pos : size > 0

/-- Represents the mowing rate of a lawn mower -/
structure MowingRate where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a person with their lawn and mower -/
structure Person where
  name : String
  lawn : LawnArea
  mower : MowingRate

/-- Calculates the time taken to mow a lawn -/
noncomputable def mowingTime (p : Person) : ℝ := p.lawn.size / p.mower.rate

theorem quentin_finishes_first (q p r : Person)
  (h_q_name : q.name = "Quentin")
  (h_p_name : p.name = "Paul")
  (h_r_name : r.name = "Rachel")
  (h_p_lawn : p.lawn.size = 3 * q.lawn.size)
  (h_r_lawn : r.lawn.size = 2 * p.lawn.size)
  (h_r_mower : r.mower.rate = 2 * p.mower.rate)
  (h_p_mower : p.mower.rate = 2 * q.mower.rate) :
  mowingTime q < mowingTime p ∧ mowingTime q < mowingTime r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quentin_finishes_first_l959_95945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_slope_sum_l959_95954

/-- Defines circle w1 -/
def w1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 10*x - 24*y - 87 = 0

/-- Defines circle w2 -/
def w2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 10*x - 24*y + 153 = 0

/-- Defines a circle externally tangent to w2 and internally tangent to w1 -/
def tangent_circle (x y r : ℝ) : Prop :=
  r + 4 = Real.sqrt ((x - 5)^2 + (y - 12)^2) ∧
  16 - r = Real.sqrt ((x + 5)^2 + (y - 12)^2)

/-- Main theorem -/
theorem tangent_circle_slope_sum :
  ∃ (p q : ℕ), Nat.Coprime p q ∧
    (∀ (x y r a : ℝ),
      tangent_circle x y r →
      y = a * x →
      (a : ℝ)^2 ≤ (p : ℝ) / (q : ℝ)) ∧
    (∃ (x y r : ℝ),
      tangent_circle x y r ∧
      y = ((p : ℝ) / (q : ℝ)).sqrt * x) ∧
    p + q = 169 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_slope_sum_l959_95954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_path_all_colors_once_l959_95995

-- Define a graph
structure Graph (V : Type) where
  edges : V → V → Prop

-- Define a proper coloring of a graph
def ProperColoring {V : Type} (G : Graph V) (k : ℕ) (f : V → Fin k) : Prop :=
  ∀ v w : V, G.edges v w → f v ≠ f w

-- Define the minimality of the coloring
def MinimalColoring {V : Type} (G : Graph V) (k : ℕ) : Prop :=
  (∃ f : V → Fin k, ProperColoring G k f) ∧
  (∀ j < k, ¬∃ f : V → Fin j, ProperColoring G j f)

-- Define a path in the graph
def GraphPath {V : Type} (G : Graph V) (p : List V) : Prop :=
  ∀ i, i + 1 < p.length → G.edges (p.get ⟨i, by sorry⟩) (p.get ⟨i+1, by sorry⟩)

-- Define the property that all colors appear exactly once in a path
def AllColorsOnceInPath {V : Type} (G : Graph V) (k : ℕ) (f : V → Fin k) (p : List V) : Prop :=
  (∀ c : Fin k, ∃! v, v ∈ p ∧ f v = c) ∧ GraphPath G p

-- The main theorem
theorem exists_path_all_colors_once 
  {V : Type} (G : Graph V) (k : ℕ) (h : MinimalColoring G k) :
  ∃ (f : V → Fin k) (p : List V), 
    ProperColoring G k f ∧ AllColorsOnceInPath G k f p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_path_all_colors_once_l959_95995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_configurations_l959_95968

/-- Represents the state of the circle of numbers -/
def CircleState := Fin 300 → ℤ

/-- The initial state of the circle -/
def initial_state : CircleState :=
  fun i => if i = 0 then 1 else 0

/-- Performs the first type of operation on the circle -/
def operation1 (state : CircleState) : CircleState :=
  fun i =>
    state i - state ((i + 299) % 300) - state ((i + 1) % 300)

/-- Performs the second type of operation on the circle -/
def operation2 (state : CircleState) (i j : Fin 300) (add : Bool) : CircleState :=
  fun k =>
    if k = i || k = j
    then state k + (if add then 1 else -1)
    else state k

/-- Checks if the state has two consecutive 1's and 298 zeros -/
def has_two_ones (state : CircleState) : Prop :=
  ∃ i : Fin 300, state i = 1 ∧ state ((i + 1) % 300) = 1 ∧
    (∀ j : Fin 300, j ≠ i ∧ j ≠ ((i + 1) % 300) → state j = 0)

/-- Checks if the state has three consecutive 1's and 297 zeros -/
def has_three_ones (state : CircleState) : Prop :=
  ∃ i : Fin 300, state i = 1 ∧ state ((i + 1) % 300) = 1 ∧ state ((i + 2) % 300) = 1 ∧
    (∀ j : Fin 300, j ≠ i ∧ j ≠ ((i + 1) % 300) ∧ j ≠ ((i + 2) % 300) → state j = 0)

/-- The main theorem stating the impossibility of achieving the desired configurations -/
theorem impossible_configurations :
  ∀ (state : CircleState),
  (∃ (n : ℕ) (ops : List (CircleState → CircleState)),
   state = List.foldl (fun s f => f s) initial_state ops) →
  ¬(has_two_ones state ∨ has_three_ones state) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_configurations_l959_95968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l959_95966

theorem sum_remainder_theorem (a b c d : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l959_95966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l959_95965

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_and_range_proof 
  (A ω φ : ℝ) 
  (h_A : A > 0) 
  (h_ω : ω > 0) 
  (h_φ : abs φ < π) 
  (h_max : f A ω φ (π/12) = 3) 
  (h_min : f A ω φ (7*π/12) = -3) :
  (∃ m : ℝ, ∀ x, x ∈ Set.Icc (-π/3) (π/6) →
    (∃ x₁ x₂, x₁ ∈ Set.Icc (-π/3) (π/6) ∧ x₂ ∈ Set.Icc (-π/3) (π/6) ∧ x₁ ≠ x₂ ∧ 
      2 * f A ω φ x₁ + 1 - m = 0 ∧ 
      2 * f A ω φ x₂ + 1 - m = 0) ↔ 
    m ∈ Set.Icc (3 * Real.sqrt 3 + 1) 7) ∧
  (∀ x : ℝ, f A ω φ x = 3 * Real.sin (2 * x + π/3)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l959_95965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_definition_l959_95937

def charity : String := "charitable organization"

theorem charity_definition : charity = "charitable organization" := by
  -- The proof is trivial since we defined charity as "charitable organization"
  rfl

#check charity_definition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_definition_l959_95937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_theorem_l959_95952

open Complex

def det (a b c d : ℂ) : ℂ := a * d - b * c

def z₁ : ℂ := det 1 2 I (I^2018)

theorem circle_theorem (z : ℂ) :
  abs (z - z₁) = 4 ↔ 
  ∃ (t : ℝ), z = (-1 - 2*I) + 4 * exp (I * t) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_theorem_l959_95952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_factor_is_4_l959_95930

-- Define the original function q
noncomputable def q (w d z : ℝ) : ℝ := 5 * w / (4 * d * z^2)

-- Define the new function q_new after changes
noncomputable def q_new (w d z F_w : ℝ) : ℝ := 5 * (F_w * w) / (4 * (2 * d) * (3 * z)^2)

theorem w_factor_is_4 (w d z : ℝ) (h_positive : d > 0 ∧ z ≠ 0) :
  ∃ F_w : ℝ, q_new w d z F_w = 0.2222222222222222 * q w d z ∧ F_w = 4 := by
  -- We'll use F_w = 4 as our witness
  use 4
  
  -- Split the goal into two parts
  constructor
  
  -- Part 1: Show that q_new w d z 4 = 0.2222222222222222 * q w d z
  · -- Expand definitions and simplify
    simp [q, q_new]
    -- The rest of the proof would go here, but we'll use sorry for now
    sorry
  
  -- Part 2: Show that F_w = 4
  · -- This is trivially true by our choice of F_w
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_factor_is_4_l959_95930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_in_triangle_l959_95997

/-- The length of an arc enclosed within a triangle, given the base length and base angles. -/
theorem arc_length_in_triangle (a α β : ℝ) (h_pos : 0 < a) (h_ang : 0 < α ∧ 0 < β ∧ α + β < π) :
  ∃ l : ℝ, l = (a * (π - α - β) * Real.sin α * Real.sin β) / Real.sin (α + β) := by
  sorry

#check arc_length_in_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_in_triangle_l959_95997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l959_95959

-- Define the hyperbola E
noncomputable def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a : ℝ) : ℝ :=
  Real.sqrt (a^2 + 1) / a

-- Define the area of triangle ABF₁
noncomputable def triangle_area (a c m : ℝ) : ℝ :=
  2 * Real.sqrt ((12 * m^2 + 12) / (m^2 - 3)^2)

theorem hyperbola_properties (a : ℝ) (h1 : a > 0) (h2 : eccentricity a = 2 * Real.sqrt 3 / 3) :
  -- 1. The equation of E is x²/3 - y² = 1
  (a = Real.sqrt 3) ∧
  -- 2. The minimum area of triangle ABF₁ is 4√3/3
  (∀ m : ℝ, triangle_area a (2 * a) m ≥ 4 * Real.sqrt 3 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l959_95959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commute_time_difference_l959_95985

-- Define constants
noncomputable def distance_to_work : ℝ := 1.5
noncomputable def walking_speed : ℝ := 3
noncomputable def train_speed : ℝ := 20
noncomputable def additional_train_time : ℝ := 10.5

-- Define functions to calculate time
noncomputable def walking_time (d : ℝ) (s : ℝ) : ℝ :=
  d / s * 60

noncomputable def train_travel_time (d : ℝ) (s : ℝ) : ℝ :=
  d / s * 60

noncomputable def total_train_time (d : ℝ) (s : ℝ) (a : ℝ) : ℝ :=
  train_travel_time d s + a

-- Theorem statement
theorem commute_time_difference :
  walking_time distance_to_work walking_speed - 
  total_train_time distance_to_work train_speed additional_train_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commute_time_difference_l959_95985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x5y2_l959_95906

def polynomial_term (n : ℕ) (x y : ℕ) : ℕ := sorry

theorem coefficient_x5y2 :
  polynomial_term 5 5 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x5y2_l959_95906
