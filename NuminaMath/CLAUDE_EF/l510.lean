import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rajesh_distance_l510_51079

theorem rajesh_distance (hiro_distance rajesh_distance : ℕ) 
  (rajesh_walks_less : hiro_distance * 4 - 10 = rajesh_distance)
  (total_distance : hiro_distance + rajesh_distance = 25) :
  rajesh_distance = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rajesh_distance_l510_51079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_sale_charity_amount_l510_51095

/-- Calculates the amount each charity receives from a cookie sale -/
theorem cookie_sale_charity_amount 
  (dozen_count : ℕ) 
  (cookies_per_dozen : ℕ) 
  (selling_price : ℚ) 
  (production_cost : ℚ) 
  (charity_count : ℕ) 
  (h1 : dozen_count = 6)
  (h2 : cookies_per_dozen = 12)
  (h3 : selling_price = 3/2)
  (h4 : production_cost = 1/4)
  (h5 : charity_count = 2) :
  (dozen_count * cookies_per_dozen * selling_price - 
   dozen_count * cookies_per_dozen * production_cost) / charity_count = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_sale_charity_amount_l510_51095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indeterminate_correctness_l510_51029

/-- Represents the correctness of a student's answer -/
inductive Correctness
| Right
| Wrong
| Unknown

/-- Represents a student -/
structure Student where
  name : String
  statement : Bool
  correctness : Correctness

/-- The problem setup -/
def problem_setup : (Student × Student × Student) :=
  ({ name := "A", statement := false, correctness := Correctness.Unknown },
   { name := "B", statement := true, correctness := Correctness.Unknown },
   { name := "C", statement := false, correctness := Correctness.Unknown })

/-- The teacher's statement -/
def teacher_statement (students : Student × Student × Student) : Prop :=
  ∃ (s1 s2 : Student), s1 ≠ s2 ∧ 
    s1.correctness = Correctness.Right ∧ 
    s2.correctness = Correctness.Wrong ∧
    (∀ s, s ≠ s1 ∧ s ≠ s2 → s.correctness = Correctness.Unknown)

/-- The theorem stating that it's impossible to determine who got it right -/
theorem indeterminate_correctness (students : Student × Student × Student) 
  (h : teacher_statement students) : 
  ¬∃ (s : Student), s.correctness = Correctness.Right ∧ 
    (∀ s', s' ≠ s → s'.correctness ≠ Correctness.Right) :=
by
  sorry

#check indeterminate_correctness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indeterminate_correctness_l510_51029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_properties_l510_51068

noncomputable def Z : ℂ := 2 / (-1 + Complex.I)

theorem Z_properties : Z^2 = 2 * Complex.I ∧ Z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_properties_l510_51068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_when_a_is_one_f_monotone_increasing_condition_l510_51061

-- Define the function f(x) as noncomputable due to its dependency on Real.log
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x + a / (x^2)

-- Theorem for the minimum value when a = 1
theorem f_min_value_when_a_is_one :
  ∃ (x_min : ℝ), x_min > 0 ∧ ∀ (x : ℝ), x > 0 → f 1 x_min ≤ f 1 x ∧ f 1 x_min = 0 := by
  sorry

-- Theorem for the monotonically increasing condition
theorem f_monotone_increasing_condition (a : ℝ) :
  (∀ (x y : ℝ), 1 ≤ x ∧ x < y → f a x < f a y) ↔ -2 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_when_a_is_one_f_monotone_increasing_condition_l510_51061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_trigonometric_sum_l510_51027

/-- Given an angle α whose terminal side passes through the point P(-4m, 3m) where m < 0,
    prove that 2sin(α) + cos(α) = -2/5 -/
theorem angle_terminal_side_trigonometric_sum (m : ℝ) (α : ℝ) (h1 : m < 0) :
  let x : ℝ := -4 * m
  let y : ℝ := 3 * m
  2 * Real.sin α + Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_trigonometric_sum_l510_51027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_area_in_rectangle_largest_circle_area_in_18_14_rectangle_l510_51048

/-- Define the area of a circle -/
noncomputable def area_circle (r : ℝ) : ℝ := Real.pi * r^2

/-- The area of the largest circle inside a rectangle -/
theorem largest_circle_area_in_rectangle (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let r := min a b / 2
  area_circle r = Real.pi * r^2 :=
by sorry

/-- The specific case for the given rectangle dimensions -/
theorem largest_circle_area_in_18_14_rectangle :
  let r := 14 / 2
  area_circle r = Real.pi * r^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_area_in_rectangle_largest_circle_area_in_18_14_rectangle_l510_51048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l510_51005

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

/-- Two lines in the form Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

theorem distance_between_specific_lines (l₁ l₂ : Line)
  (h_parallel : l₁.A = l₂.A ∧ l₁.B = l₂.B)
  (h_l₁ : l₁ = ⟨2, 1, -1⟩)
  (h_l₂ : l₂ = ⟨2, 1, 1⟩) :
  distance_between_parallel_lines l₁.A l₁.B l₁.C l₂.C = 2 * Real.sqrt 5 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l510_51005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l510_51045

/-- The inclination angle of a line with equation √3x + y + 2024 = 0 is 2π/3 -/
theorem line_inclination_angle : 
  ∃ θ : ℝ, θ = 2 * Real.pi / 3 ∧ 
    (∀ x y : ℝ, Real.sqrt 3 * x + y + 2024 = 0 → Real.tan θ = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l510_51045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_equals_five_l510_51049

noncomputable def S (n : ℕ) : ℝ := sorry
noncomputable def a (n : ℕ) : ℝ := sorry

axiom sequence_condition (n : ℕ) : 
  a n * (2 + Real.sin (n * Real.pi / 2)) = n * (2 + Real.cos (n * Real.pi))

axiom sum_condition (n : ℕ) (a b : ℝ) : 
  S (4 * n) = a * n^2 + b * n

theorem a_minus_b_equals_five (a b : ℝ) 
  (h : ∀ n, S (4 * n) = a * n^2 + b * n) : a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_equals_five_l510_51049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_eighth_lt_pi_eighth_l510_51053

theorem sin_pi_eighth_lt_pi_eighth : Real.sin (π / 8) < π / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_eighth_lt_pi_eighth_l510_51053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bounds_l510_51056

/-- Represents the percentage of students who changed their answer -/
def x : ℝ := 0  -- Initialize with a default value

/-- Initial percentage of students who liked math -/
def initial_like : ℝ := 50

/-- Initial percentage of students who didn't like math -/
def initial_dislike : ℝ := 50

/-- Final percentage of students who liked math -/
def final_like : ℝ := 70

/-- Final percentage of students who didn't like math -/
def final_dislike : ℝ := 30

/-- The sum of percentages in each survey should be 100% -/
axiom initial_sum : initial_like + initial_dislike = 100
axiom final_sum : final_like + final_dislike = 100

/-- The percentage of students who changed their answer is non-negative -/
axiom x_non_negative : x ≥ 0

/-- The percentage of students who changed their answer cannot exceed 100% -/
axiom x_upper_bound : x ≤ 100

/-- Theorem stating the bounds on the percentage of students who changed their answer -/
theorem x_bounds : 20 ≤ x ∧ x ≤ 80 := by
  sorry

#check x_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bounds_l510_51056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_fourth_in_f_prime_l510_51044

noncomputable section

/-- The function f(x) as defined in the problem -/
def f (x : ℝ) : ℝ := (x + 1) * (x^2 + 2) * (x^3 + 3)

/-- The derivative of f(x) -/
def f' : ℝ → ℝ := deriv f

/-- Theorem stating that the coefficient of x^4 in f'(x) is 5 -/
theorem coefficient_of_x_fourth_in_f_prime :
  ∃ (a b c d : ℝ), f' = fun x ↦ 6 * x^5 + 5 * x^4 + a * x^3 + b * x^2 + c * x + d :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_fourth_in_f_prime_l510_51044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shuttlecock_kicks_l510_51032

theorem shuttlecock_kicks 
  (total_kicks : ℕ) 
  (hong_kicks : ℕ) 
  (fang_kicks : ℕ) 
  (ying_kicks : ℕ) 
  (ping_kicks : ℕ) 
  (h1 : total_kicks = 280)
  (h2 : 2 * hong_kicks = 3 * fang_kicks)
  (h3 : 2 * hong_kicks = ying_kicks)
  (h4 : 2 * hong_kicks = 5 * ping_kicks)
  (h5 : total_kicks = hong_kicks + fang_kicks + ying_kicks + ping_kicks) :
  hong_kicks = 40 ∧ fang_kicks = 60 ∧ ying_kicks = 80 ∧ ping_kicks = 100 := by
  sorry

#check shuttlecock_kicks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shuttlecock_kicks_l510_51032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_side_length_proof_l510_51052

/-- The length of a side of a regular decagon inscribed in a unit circle -/
noncomputable def regular_decagon_side_length : ℝ := (Real.sqrt 5 - 1) / 2

/-- Theorem: The length of each side of a regular decagon inscribed in a circle with radius 1 is (√5 - 1) / 2 -/
theorem regular_decagon_side_length_proof (n : ℕ) (h : n = 10) :
  let r : ℝ := 1 -- radius of the circle
  let θ : ℝ := 2 * Real.pi / n -- central angle of the decagon
  2 * r * Real.sin (θ / 2) = regular_decagon_side_length :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_side_length_proof_l510_51052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_composite_mersenne_under_300_l510_51099

/-- Definition of a Mersenne number -/
def mersenne_number (n : ℕ) : ℕ := 2^n - 1

/-- A number is composite if it has a factor other than 1 and itself -/
def isComposite (n : ℕ) : Prop := ∃ m, 1 < m ∧ m < n ∧ n % m = 0

theorem largest_composite_mersenne_under_300 :
  ∃ n : ℕ, mersenne_number n = 255 ∧
    isComposite (mersenne_number n) ∧
    (∀ m : ℕ, mersenne_number m < 300 → isComposite (mersenne_number m) → mersenne_number m ≤ 255) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_composite_mersenne_under_300_l510_51099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l510_51070

theorem trig_simplification (α : ℝ) (h : Real.cos α ≠ 0) :
  (Real.tan α + 1 / Real.tan α) * (1 / 2 * Real.sin (2 * α)) - 2 * (Real.cos α)^2 = -Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l510_51070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l510_51098

theorem sequence_inequality (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a n > 0)
  (h2 : ∀ n : ℕ, (a n)^2 ≤ a n - a (n + 1)) :
  ∀ n : ℕ, a n < 1 / (n : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l510_51098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_trajectory_l510_51086

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents an equilateral cone -/
structure EquilateralCone where
  slantHeight : ℝ
  apex : Point2D
  base : Point2D

/-- The trajectory of point P on the base of an equilateral cone -/
def trajectoryEquation (p : Point2D) : Prop :=
  p.y^2 = -3 * p.x + 3/2

/-- Theorem stating the trajectory of point P on the base of an equilateral cone -/
theorem cone_trajectory (cone : EquilateralCone) (p : Point2D) :
  cone.slantHeight = 2 →
  let m : Point2D := { x := -1/2, y := 0 }
  let angle_mp_sa : ℝ := 60 * Real.pi / 180  -- 60 degrees in radians
  (p.x - m.x)^2 + p.y^2 = (cone.slantHeight / 2)^2 →  -- MP = SA/2
  (p.x - m.x)^2 + p.y^2 + 1 - 2 * Real.sqrt ((p.x - m.x)^2 + p.y^2) * Real.cos angle_mp_sa = cone.slantHeight^2 →  -- Law of cosines
  trajectoryEquation p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_trajectory_l510_51086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_less_than_threshold_l510_51020

def numbers : List ℚ := [4/5, 1/2, 9/10, 1/3]
def threshold : ℚ := 3/5

theorem largest_less_than_threshold :
  (numbers.filter (fun x => x < threshold)).argmax id = some (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_less_than_threshold_l510_51020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carolyn_sum_is_twelve_l510_51015

def game_numbers : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_valid_move (numbers: List Nat) (n: Nat) : Bool :=
  n ∈ numbers ∧ ∃ m ∈ numbers, m ≠ n ∧ n % m = 0

def carolyn_move (numbers: List Nat) : Option Nat :=
  numbers.find? (is_valid_move numbers)

def paul_move (numbers: List Nat) (n: Nat) : List Nat :=
  numbers.filter (λ m => ¬(n % m = 0) ∧ m ≠ n)

def game_step (numbers: List Nat) : List Nat :=
  match carolyn_move numbers with
  | some n => paul_move numbers n
  | none => []

def game_play (numbers: List Nat) : List Nat :=
  let rec aux (current: List Nat) (removed: List Nat) (fuel: Nat) : List Nat :=
    match fuel with
    | 0 => removed
    | fuel + 1 =>
      match carolyn_move current with
      | some n => aux (game_step current) (n :: removed) fuel
      | none => removed
  aux numbers [] numbers.length

theorem carolyn_sum_is_twelve :
  (game_play (paul_move game_numbers 3)).sum = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carolyn_sum_is_twelve_l510_51015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_l510_51040

/-- Given an infinite geometric series {a_n} with common ratio q,
    if a_1 = lim_{n → ∞} (a_3 + a_4 + ...), then q = (-1 + √5) / 2 -/
theorem geometric_series_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric series condition
  (a 1 = ∑' n, a (n + 2)) →  -- a_1 = lim_{n → ∞} (a_3 + a_4 + ...)
  q = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_l510_51040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_even_and_range_l510_51067

noncomputable def f (x : ℝ) : ℝ := Real.sin x

theorem sine_even_and_range (θ : ℝ) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi)) :
  (∀ x, f (x + θ) = f (-x + θ)) →
  (θ = Real.pi / 2 ∨ θ = 3 * Real.pi / 2) ∧
  Set.range (fun x ↦ (f (x + Real.pi / 12))^2 + (f (x + Real.pi / 4))^2) =
    Set.Icc (1 - Real.sqrt 3 / 2) (1 + Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_even_and_range_l510_51067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_proof_l510_51073

open Matrix

theorem matrix_transformation_proof :
  ∃ (M : Matrix (Fin 3) (Fin 3) ℝ),
    ∀ (N : Matrix (Fin 3) (Fin 3) ℝ),
      let result := M * N
      (∀ j, result 1 j = 2 * N 2 j) ∧
      (∀ j, result 2 j = 3 * N 1 j) ∧
      (∀ j, result 3 j = 3 * N 3 j) :=
by
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![0, 2, 0; 3, 0, 0; 0, 0, 3]
  exists M
  intro N
  simp [Matrix.mul_apply]
  sorry

#check matrix_transformation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_proof_l510_51073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_relation_l510_51046

-- Define the coefficients of the quadratic equation
variable (a b c : ℝ)

-- Define the roots of the quadratic equation
variable (α β : ℝ)

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- State that α and β are roots of the equation
axiom root_α : quadratic_equation a b c α
axiom root_β : quadratic_equation a b c β

-- State that one root is triple the other
axiom triple_root : β = 3 * α

-- Vieta's formulas
axiom vieta_sum : α + β = -b / a
axiom vieta_product : α * β = c / a

-- Theorem to prove
theorem root_relation : 3 * b^2 = 16 * a * c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_relation_l510_51046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_parallelepiped_volume_l510_51026

/-- The volume of a special rectangular parallelepiped -/
theorem special_parallelepiped_volume 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_angle : Real.cos (60 * Real.pi / 180) = 1 / 2)
  (h_diag : (a^2 + b^2 + a*b).sqrt = ((a^2 + b^2 - a*b) + ((Real.sqrt 6 * a * b) / 2)^2).sqrt) :
  (Real.sqrt 6 * a^2 * b^2) / 2 = 
    (a * b * Real.sin (60 * Real.pi / 180)) * ((Real.sqrt 6 * a * b) / 2) := by
  sorry

#check special_parallelepiped_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_parallelepiped_volume_l510_51026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_congruent_modulo_l510_51001

noncomputable def sequenceA (i : ℕ) : ℕ :=
  match i with
  | 0 => 2
  | n + 1 => 2^(sequenceA n)

theorem eventually_congruent_modulo (n : ℕ) (hn : n ≥ 1) :
  ∃ j : ℕ, ∀ k : ℕ, k ≥ j → sequenceA k ≡ sequenceA j [MOD n] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_congruent_modulo_l510_51001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_moving_circle_center_l510_51000

/-- Given a circle C and a fixed point A, we define the trajectory of the center M of a moving circle
    that passes through A and is tangent to C from the inside. -/
theorem trajectory_of_moving_circle_center (x y : ℝ) :
  let C : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + p.2^2 = 4}
  let A : ℝ × ℝ := (-3, 0)
  (∀ (M : ℝ × ℝ), M ∉ C → 
    (∃ (r : ℝ), r > 0 ∧ 
      (∀ (p : ℝ × ℝ), (p.1 - M.1)^2 + (p.2 - M.2)^2 = r^2 → 
        (p = A ∨ (∃ (q : ℝ × ℝ), q ∈ C ∧ (q.1 - M.1)^2 + (q.2 - M.2)^2 = r^2))))) →
  (x^2 - y^2/8 = 1 ∧ x ≥ 1) ↔ (x, y) ∈ {M | M ∉ C ∧ 
    (∃ (r : ℝ), r > 0 ∧ 
      (∀ (p : ℝ × ℝ), (p.1 - x)^2 + (p.2 - y)^2 = r^2 → 
        (p = A ∨ (∃ (q : ℝ × ℝ), q ∈ C ∧ (q.1 - x)^2 + (q.2 - y)^2 = r^2))))} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_moving_circle_center_l510_51000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l510_51043

open Real

variable (a b c A B C : ℝ)

-- Define the triangle ABC
def is_triangle (a b c A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

-- Define the given conditions
def given_conditions (a b c A B C : ℝ) : Prop :=
  is_triangle a b c A B C ∧
  (a + 2*c) * cos B + b * cos A = 0 ∧
  b = 3 ∧
  a + b + c = 3 + 2 * sqrt 3

-- Define the theorem
theorem triangle_properties 
  (h : given_conditions a b c A B C) :
  B = 2*Real.pi/3 ∧ 
  a + c = 2 * sqrt 3 ∧
  a * c = 3 ∧
  (1/2) * b * c * sin A = 3 * sqrt 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l510_51043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l510_51060

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2) / h.a

/-- Theorem: If one asymptote of the hyperbola is y = √3x, then its eccentricity is 2 -/
theorem hyperbola_eccentricity_is_two (h : Hyperbola) 
  (h_asymptote : h.b / h.a = Real.sqrt 3) : eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l510_51060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_at_12_ohms_l510_51008

/-- The voltage of the battery in volts -/
noncomputable def V : ℝ := 48

/-- The resistance in ohms -/
noncomputable def R : ℝ := 12

/-- The current in amperes as a function of resistance -/
noncomputable def I (r : ℝ) : ℝ := V / r

/-- Theorem: When the resistance is 12Ω, the current is 4A -/
theorem current_at_12_ohms : I R = 4 := by
  -- Unfold the definitions of I, V, and R
  unfold I V R
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_at_12_ohms_l510_51008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_equals_zero_l510_51089

theorem cos_2beta_equals_zero 
  (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.sin α = Real.sqrt 5 / 5) 
  (h4 : Real.sin (α - β) = -(Real.sqrt 10) / 10) : 
  Real.cos (2 * β) = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_equals_zero_l510_51089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_34_and_35_l510_51074

noncomputable def floor (z : ℝ) : ℤ := Int.floor z

theorem x_plus_y_between_34_and_35 
  (x y : ℝ) 
  (h1 : y = 3 * (floor x) + 2)
  (h2 : y = 4 * (floor (x - 3)) + 6)
  (h3 : x ≠ ↑(floor x)) : 
  34 < x + y ∧ x + y < 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_34_and_35_l510_51074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_implies_angle_identity_l510_51041

theorem point_on_line_implies_angle_identity (α : ℝ) :
  (Real.sin α = -2 * Real.cos α) → (Real.sin (2 * α) + 2 * Real.cos (2 * α) = -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_implies_angle_identity_l510_51041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l510_51013

-- Define the circles
def circle_M (x y : ℝ) : Prop := (x - 3/2)^2 + y^2 = 23/4
def circle_N (x y : ℝ) : Prop := (x + 5)^2 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem max_distance_MN :
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_M x1 y1 ∧ circle_N x2 y2 ∧
    (∀ (x3 y3 x4 y4 : ℝ),
      circle_M x3 y3 → circle_N x4 y4 →
      distance x1 y1 x2 y2 ≥ distance x3 y3 x4 y4) ∧
    distance x1 y1 x2 y2 = (15 + Real.sqrt 23) / 2 := by
  sorry

#check max_distance_MN

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l510_51013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_over_a_l510_51069

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.log x + (Real.exp 1 - a) * x - b

-- State the theorem
theorem min_b_over_a (a b : ℝ) :
  (∀ x > 0, f a b x ≤ 0) →
  (∃ c, c = -1 / Real.exp 1 ∧ ∀ k, (∃ a' b', k = b' / a' ∧ ∀ x > 0, f a' b' x ≤ 0) → k ≥ c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_over_a_l510_51069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_form_case_l510_51094

/-- Represents a quadrilateral ABCD with a circle of radius r -/
structure QuadrilateralWithCircle where
  AB : ℝ
  CD : ℝ
  r : ℕ

/-- The total area of the quadrilateral ABCD plus the area of the circle -/
noncomputable def total_area (q : QuadrilateralWithCircle) : ℝ :=
  (q.AB^2 + q.AB * q.CD) / 2 + Real.pi * (q.r^2 : ℝ)

/-- Theorem stating that the case AB = 4, CD = 1, r = 2 results in the simplest form -/
theorem simplest_form_case :
  ∃ (a b : ℕ), total_area ⟨4, 1, 2⟩ = a + b * Real.pi ∧
  ∀ (q : QuadrilateralWithCircle), 
    q ∈ [⟨4, 1, 2⟩, ⟨6, 2, 3⟩, ⟨8, 3, 4⟩, ⟨10, 4, 5⟩, ⟨12, 5, 6⟩] →
    ∃ (c d : ℕ), total_area q = c + d * Real.pi ∧ b ≤ d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_form_case_l510_51094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_equals_4035_l510_51059

noncomputable def f (A w φ x : ℝ) : ℝ := A * (Real.cos (w * x + φ))^2 + 1

theorem sum_f_equals_4035 (A w φ : ℝ) (hA : A > 0) (hw : w > 0) (hφ : 0 < φ ∧ φ < Real.pi / 2)
  (hmax : ∀ x, f A w φ x ≤ 3)
  (hy_intersect : f A w φ 0 = 2)
  (hsymmetry : ∃ T > 0, ∀ x, f A w φ (x + T) = f A w φ x ∧ T = 2) :
  (Finset.range 2018).sum (fun i ↦ f A w φ (i + 1)) = 4035 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_equals_4035_l510_51059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_min_f_g_l510_51028

/-- Definition of the minimum function -/
noncomputable def my_min (x y : ℝ) : ℝ := if x ≤ y then x else y

/-- The problem statement -/
theorem max_value_of_min_f_g :
  ∀ (f g : ℝ → ℝ),
  (∀ x, f x + g x = 2 * x / (x^2 + 1)) →
  (∃ (M : ℝ), M = 1/2 ∧ ∀ x, my_min (f x) (g x) ≤ M ∧ ∃ x₀, my_min (f x₀) (g x₀) = M) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_min_f_g_l510_51028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_circle_centers_l510_51088

/-- The greatest possible distance between the centers of two circles in a rectangle --/
theorem greatest_distance_between_circle_centers
  (rect_width : ℝ) (rect_height : ℝ) (circle_diameter : ℝ)
  (h_width : rect_width = 15)
  (h_height : rect_height = 16)
  (h_diameter : circle_diameter = 7) :
  Real.sqrt ((rect_width - circle_diameter) ^ 2 + (rect_height - circle_diameter) ^ 2) = Real.sqrt 145 :=
by
  -- Replace this with the actual proof steps
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_circle_centers_l510_51088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_arrangements_l510_51087

/-- The number of students -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of competitions -/
def num_competitions : ℕ := 4

/-- The number of competitions Student A cannot participate in -/
def restricted_competitions : ℕ := 2

/-- The number of different arrangements for the competitions -/
def num_arrangements : ℕ := 72

theorem competition_arrangements :
  (total_students.choose selected_students * (num_competitions - restricted_competitions) * 
   Nat.factorial (selected_students - 1)) +
  ((total_students - 1).choose selected_students * Nat.factorial num_competitions) = num_arrangements := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_arrangements_l510_51087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_is_twenty_l510_51085

/-- The cost to feed chickens given the total number of birds, fraction of ducks, and cost per chicken. -/
def cost_to_feed_chickens (total_birds : ℕ) (duck_fraction : ℚ) (cost_per_chicken : ℕ) : ℕ :=
  Int.toNat ((total_birds : ℚ) * (1 - duck_fraction) * cost_per_chicken).floor

/-- Theorem stating that the cost to feed chickens is $20 under the given conditions. -/
theorem cost_is_twenty :
  cost_to_feed_chickens 15 (1/3) 2 = 20 := by
  -- Unfold the definition of cost_to_feed_chickens
  unfold cost_to_feed_chickens
  -- Simplify the arithmetic
  simp
  -- The proof is completed
  rfl

#eval cost_to_feed_chickens 15 (1/3) 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_is_twenty_l510_51085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_five_digit_divisible_by_first_five_primes_l510_51090

theorem smallest_five_digit_divisible_by_first_five_primes : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- Five-digit number
  (∀ p : ℕ, p ∈ [2, 3, 5, 7, 11] → n % p = 0) ∧  -- Divisible by first five primes
  (∀ m : ℕ, m ≥ 10000 ∧ m < n →
    ∃ p : ℕ, p ∈ [2, 3, 5, 7, 11] ∧ m % p ≠ 0) ∧  -- Smallest such number
  n = 11550 :=  -- The answer is 11550
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_five_digit_divisible_by_first_five_primes_l510_51090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_christopher_natasha_difference_l510_51017

/-- The number of joggers Tyson bought -/
def T : ℕ := 6

/-- The number of joggers Martha bought -/
def M : ℕ := T - (2 * T) / 3

/-- The number of joggers Alexander bought -/
def A : ℕ := (3 * T) / 2

/-- The number of joggers Christopher bought -/
def C : ℕ := 18 * (A - M)

/-- The number of joggers Natasha bought -/
def N : ℕ := (4 * C) / 5

theorem christopher_natasha_difference : C - N = 26 := by
  -- Unfold definitions
  unfold C N A M T
  -- Perform arithmetic calculations
  norm_num
  -- QED
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_christopher_natasha_difference_l510_51017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l510_51062

-- Define the two equations
def equation1 (x y : ℝ) : Prop := x^2 + 4*y = 20
def equation2 (x y : ℝ) : Prop := 2*x + y = 20

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ equation1 x y ∧ equation2 x y}

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_distance :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ distance p1 p2 = 4 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l510_51062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_store_soda_correct_l510_51039

def grocery_store_soda (apples : ℕ) (diet_soda : ℕ) (total_bottles_difference : ℕ) : ℕ :=
  total_bottles_difference + apples - diet_soda

theorem grocery_store_soda_correct (apples diet_soda total_bottles_difference : ℕ) :
  grocery_store_soda apples diet_soda total_bottles_difference =
  total_bottles_difference + apples - diet_soda :=
by
  rfl

#eval grocery_store_soda 36 54 98

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_store_soda_correct_l510_51039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_approx_10_l510_51071

/-- The radius of the base of a cone, given its slant height and curved surface area. -/
noncomputable def cone_base_radius (slant_height : ℝ) (curved_surface_area : ℝ) : ℝ :=
  curved_surface_area / (Real.pi * slant_height)

/-- Theorem: The radius of the base of a cone with slant height 21 cm and curved surface area 659.7344572538566 cm² is approximately 10 cm. -/
theorem cone_radius_approx_10 :
  let r := cone_base_radius 21 659.7344572538566
  ∃ ε > 0, abs (r - 10) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_approx_10_l510_51071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l510_51014

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x)

theorem axis_of_symmetry :
  ∃ (k : ℤ), ∀ (x : ℝ), f (x + (-π/12 : ℝ)) = f (-x + (-π/12 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l510_51014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_equal_distances_l510_51072

/-- Ellipse C with equation x²/3 + y² = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- Line y = kx + m -/
def line (k m x y : ℝ) : Prop := y = k * x + m

/-- Point P is on ellipse C -/
def on_ellipse_C (P : ℝ × ℝ) : Prop := ellipse_C P.1 P.2

/-- Point P is on line -/
def on_line (k m : ℝ) (P : ℝ × ℝ) : Prop := line k m P.1 P.2

/-- Distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- Lower vertex of ellipse C -/
def lower_vertex : ℝ × ℝ := (0, -1)

/-- Theorem: Range of m for equal distances -/
theorem range_of_m_for_equal_distances :
  ∀ k m : ℝ,
  (∃ M N : ℝ × ℝ,
    on_ellipse_C M ∧ on_ellipse_C N ∧
    on_line k m M ∧ on_line k m N ∧
    M ≠ N ∧
    distance lower_vertex M = distance lower_vertex N) →
  m ∈ Set.Icc (1/2) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_equal_distances_l510_51072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l510_51055

/-- Represents a cistern with two pipes -/
structure Cistern where
  fill_time : ℝ  -- Time to fill the cistern with pipe A
  empty_time : ℝ  -- Time to empty the cistern with pipe B

/-- Calculates the time to fill the cistern when both pipes are open -/
noncomputable def time_to_fill (c : Cistern) : ℝ :=
  (c.fill_time * c.empty_time) / (c.empty_time - c.fill_time)

/-- Theorem stating that for the given cistern, it takes 80 hours to fill when both pipes are open -/
theorem cistern_fill_time :
  let c : Cistern := { fill_time := 16, empty_time := 20 }
  time_to_fill c = 80 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l510_51055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_paycheck_calculation_l510_51003

/-- Represents Jim's bi-weekly paycheck calculation -/
noncomputable def jim_paycheck (gross_pay : ℝ) (retirement_rate : ℝ) (tax_deduction : ℝ) 
  (monthly_healthcare : ℝ) (monthly_gym : ℝ) (monthly_bonus : ℝ) 
  (bonus_tax_rate : ℝ) (exchange_rate : ℝ) : ℝ :=
  let retirement_contribution := gross_pay * retirement_rate
  let bi_weekly_fees := (monthly_healthcare + monthly_gym) / 2
  let net_bonus := monthly_bonus * (1 - bonus_tax_rate) / 2
  let net_pay := gross_pay - retirement_contribution - tax_deduction - bi_weekly_fees + net_bonus
  net_pay / exchange_rate

/-- Theorem stating that Jim's bi-weekly paycheck in euros is approximately €686.96 -/
theorem jim_paycheck_calculation :
  ∃ ε > 0, |jim_paycheck 1120 0.25 100 200 50 500 0.30 1.15 - 686.96| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_paycheck_calculation_l510_51003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_reflection_path_l510_51037

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define the foci (we don't need to specify their exact coordinates)
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ

-- Axiom stating that A and B are the foci of the ellipse
axiom are_foci : ∀ (x y : ℝ), is_on_ellipse x y → 
  (Real.sqrt ((x - A.1)^2 + (y - A.2)^2) + Real.sqrt ((x - B.1)^2 + (y - B.2)^2)) = 8

-- Define the reflection property
def reflects (p q r : ℝ × ℝ) : Prop :=
  is_on_ellipse q.1 q.2 ∧ 
  (Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) + Real.sqrt ((q.1 - r.1)^2 + (q.2 - r.2)^2)) =
  (Real.sqrt ((p.1 - A.1)^2 + (p.2 - A.2)^2) + Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
   Real.sqrt ((B.1 - r.1)^2 + (B.2 - r.2)^2))

-- Theorem statement
theorem shortest_reflection_path :
  ∀ (P Q : ℝ × ℝ), reflects A P B → reflects B Q A → 
  (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) + 
   Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) + 
   Real.sqrt ((B.1 - Q.1)^2 + (B.2 - Q.2)^2) + 
   Real.sqrt ((Q.1 - A.1)^2 + (Q.2 - A.2)^2)) = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_reflection_path_l510_51037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle1_equation_circle2_equation_l510_51006

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the two circles
noncomputable def circle1 : Circle := { center := (-2, 3), radius := 5 }
noncomputable def circle2 : Circle := { center := (1/3, -1), radius := 1/2 }

-- Function to generate the equation of a circle
noncomputable def circleEquation (c : Circle) : ℝ → ℝ → ℝ := 
  fun x y ↦ (x - c.center.1)^2 + (y - c.center.2)^2 - c.radius^2

-- Theorem for the first circle
theorem circle1_equation : 
  ∀ x y, circleEquation circle1 x y = 0 ↔ x^2 + y^2 + 4*x - 6*y - 12 = 0 :=
by sorry

-- Theorem for the second circle
theorem circle2_equation : 
  ∀ x y, circleEquation circle2 x y = 0 ↔ 36*x^2 + 36*y^2 - 24*x + 72*y + 31 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle1_equation_circle2_equation_l510_51006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_for_horizontal_asymptote_l510_51042

/-- A polynomial function -/
structure MyPolynomial where
  degree : ℕ

/-- A rational function -/
structure RationalFunction where
  numerator : MyPolynomial
  denominator : MyPolynomial

/-- Condition for a rational function to have a horizontal asymptote -/
def has_horizontal_asymptote (f : RationalFunction) : Prop :=
  f.numerator.degree ≤ f.denominator.degree

/-- The specific rational function from the problem -/
def specific_function (p : MyPolynomial) : RationalFunction :=
  { numerator := p,
    denominator := { degree := 4 } }

/-- Theorem stating the maximum degree of p(x) for a horizontal asymptote -/
theorem max_degree_for_horizontal_asymptote :
  ∀ p : MyPolynomial, has_horizontal_asymptote (specific_function p) → p.degree ≤ 4 :=
by
  intro p h
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_for_horizontal_asymptote_l510_51042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_less_than_twice_sine_l510_51023

theorem sine_double_angle_less_than_twice_sine (α : Real) 
  (h : 0 < α ∧ α < π / 4) : 
  Real.sin (2 * α) < 2 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_less_than_twice_sine_l510_51023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_parabola_area_l510_51064

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define the probability function (this is a placeholder and needs to be properly defined)
noncomputable def probability_in_area (f : ℝ → ℝ) (c m1 m2 : ℝ) : ℝ :=
  (∫ x in m1..m2, min (f x) c - 0) / ((m2 - m1) * c)

-- State the theorem
theorem probability_in_parabola_area (a b m c : ℝ) : 
  (∀ x, f a b x ≥ 0) →  -- Value range of f(x) is [0, +∞)
  (∀ x, f a b x < c ↔ m < x ∧ x < m + 6) →  -- Solution set of f(x) < c is (m, m+6)
  (probability_in_area (f a b) c m (m + 6) = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_parabola_area_l510_51064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_intersection_range_l510_51033

/-- The line equation x + my = 2 + m -/
def line_eq (m : ℝ) (x y : ℝ) : Prop :=
  x + m * y = 2 + m

/-- The circle equation x^2 + y^2 - 2x - 2y + 1 = 0 -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- The line intersects with the circle for all non-zero real m -/
theorem line_intersects_circle :
  ∀ m : ℝ, m ≠ 0 → ∃ x y : ℝ, line_eq m x y ∧ circle_eq x y :=
by
  sorry

/-- The range of m for which the line intersects the circle is (-∞, 0) ∪ (0, +∞) -/
theorem intersection_range :
  ∀ m : ℝ, (∃ x y : ℝ, line_eq m x y ∧ circle_eq x y) ↔ m ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_intersection_range_l510_51033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_split_implies_length_ratio_l510_51082

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point C is on the line segment AB -/
def isOnLineSegment (A B C : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C.x = A.x + t * (B.x - A.x) ∧ C.y = A.y + t * (B.y - A.y)

/-- The ratio in which C splits the x-coordinates of A and B -/
noncomputable def xRatio (A B C : Point) : ℝ := (C.x - A.x) / (B.x - C.x)

/-- The ratio in which C splits the y-coordinates of A and B -/
noncomputable def yRatio (A B C : Point) : ℝ := (C.y - A.y) / (B.y - C.y)

/-- The main theorem -/
theorem ratio_split_implies_length_ratio 
  (A B C : Point) 
  (h1 : isOnLineSegment A B C) 
  (h2 : xRatio A B C = 3) 
  (h3 : yRatio A B C = 3) : 
  distance A C / distance C B = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_split_implies_length_ratio_l510_51082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_72π_l510_51081

open Real

-- Define the length of each segment
noncomputable def segment_length : ℝ := 4

-- Define the number of segments
def num_segments : ℕ := 6

-- Define the radius of the large semicircle
noncomputable def large_radius : ℝ := segment_length * (num_segments / 2 : ℝ)

-- Define the area of the shaded region
noncomputable def shaded_area : ℝ := π * large_radius^2 / 2

-- Theorem statement
theorem shaded_area_equals_72π : shaded_area = 72 * π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_72π_l510_51081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patient_weight_calculation_l510_51004

/-- The patient's body weight in pounds -/
noncomputable def patient_weight : ℚ := 120

/-- The prescribed dosage in cubic centimeters -/
noncomputable def prescribed_dosage : ℚ := 12

/-- The typical dosage in cubic centimeters per 15 pounds of body weight -/
noncomputable def typical_dosage_rate : ℚ := 2 / 15

/-- The prescribed dosage is 25% less than the typical dosage -/
noncomputable def dosage_reduction_factor : ℚ := 3 / 4

theorem patient_weight_calculation :
  prescribed_dosage = dosage_reduction_factor * typical_dosage_rate * patient_weight :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_patient_weight_calculation_l510_51004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_l510_51011

-- Define the ceiling function as noncomputable
noncomputable def ceiling (x : ℝ) : ℤ := ⌈x⌉

-- Define the problem conditions
def problem_conditions (x : ℝ) : Prop :=
  ceiling (2 * x + 1) = 5 ∧ ceiling (2 - 3 * x) = -3

-- State the theorem
theorem x_range (x : ℝ) :
  problem_conditions x → (5/3 : ℝ) ≤ x ∧ x < 2 := by
  sorry

#check x_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_l510_51011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ironman_age_l510_51022

/-- Given the ages of superheroes, prove Ironman's age --/
theorem ironman_age (thor captain_america peter_parker doctor_strange ironman : ℕ) : 
  thor = 13 * captain_america →
  captain_america = 7 * peter_parker →
  4 * peter_parker = doctor_strange →
  doctor_strange = captain_america + 87 →
  ironman = peter_parker + 32 →
  thor = 1456 →
  ironman = 82 := by
  intro h1 h2 h3 h4 h5 h6
  -- We'll use these hypotheses to derive Ironman's age
  -- The proof steps would go here, but for now we'll use sorry
  sorry

#check ironman_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ironman_age_l510_51022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_4_range_of_a_when_f_geq_4_l510_51075

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for the first part of the problem
theorem solution_set_when_a_is_4 :
  {x : ℝ | f 4 x ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_when_f_geq_4 :
  (∀ x : ℝ, f a x ≥ 4) → a ∈ Set.Iic (-3) ∪ Set.Ici 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_4_range_of_a_when_f_geq_4_l510_51075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monster_perimeter_value_l510_51054

/-- The perimeter of a shape consisting of two sides of an equilateral triangle
    with side length 3 cm and a 90° arc of a circle inscribed in this triangle -/
noncomputable def monster_perimeter : ℝ :=
  let triangle_side : ℝ := 3
  let triangle_perimeter : ℝ := 2 * triangle_side
  let inscribed_circle_radius : ℝ := triangle_side / Real.sqrt 3
  let arc_length : ℝ := (Real.pi / 2) * inscribed_circle_radius
  triangle_perimeter + arc_length

/-- Theorem stating that the monster_perimeter is equal to 6 + (π√3)/2 cm -/
theorem monster_perimeter_value : 
  monster_perimeter = 6 + (Real.pi * Real.sqrt 3) / 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monster_perimeter_value_l510_51054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l510_51092

open Real

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧ A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2

-- Define the arithmetic sequence property
def arithmetic_sequence (A B C : ℝ) : Prop :=
  2 * B = A + C

-- Define vector m
noncomputable def m (A : ℝ) : ℝ × ℝ :=
  (2 * sin A, 2 * sin (A + Real.pi/12))

-- Define vector n
noncomputable def n (A : ℝ) : ℝ × ℝ :=
  (sin A, cos (A + Real.pi/12))

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem triangle_ABC_properties {A B C : ℝ} 
  (h_triangle : triangle_ABC A B C) 
  (h_arithmetic : arithmetic_sequence A B C) :
  (Real.pi/6 < A ∧ A < Real.pi/2) ∧
  (∃ (x : ℝ), 3/2 < x ∧ x ≤ 2 ∧ dot_product (m A) (n A) = x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l510_51092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_implies_odd_numbers_l510_51076

theorem odd_sum_implies_odd_numbers (n : ℕ) (sum : ℕ) (nums : Fin n → ℕ) 
  (h1 : n = 49) 
  (h2 : sum = 2401) 
  (h3 : sum = (Finset.sum Finset.univ (λ i => nums i))) : 
  ∀ i, Odd (nums i) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_implies_odd_numbers_l510_51076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_192_sqrt_2_l510_51065

/-- A rectangle in 3D space with a point above it forming a pyramid -/
structure PyramidOverRectangle where
  /-- The length of PN, which is perpendicular to the rectangle -/
  h : ℕ
  /-- The length of NP -/
  b : ℕ
  /-- PQRS is a rectangle -/
  is_rectangle : True
  /-- PN is perpendicular to the plane of PQRS -/
  is_perpendicular : True
  /-- NP, NQ, and NR are consecutive even positive integers -/
  consecutive_even : b % 2 = 0 ∧ b > 0

/-- The volume of the pyramid NPQRS -/
noncomputable def pyramid_volume (p : PyramidOverRectangle) : ℝ :=
  192 * Real.sqrt 2

/-- Theorem stating that the volume of the pyramid NPQRS is 192√2 -/
theorem pyramid_volume_is_192_sqrt_2 (p : PyramidOverRectangle) :
  pyramid_volume p = 192 * Real.sqrt 2 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_192_sqrt_2_l510_51065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_one_l510_51002

-- Define a 4x4 grid
def Grid := Fin 4 → Fin 4 → Fin 4

-- Define a valid grid
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j < 4) ∧
  (∀ i, Function.Injective (g i)) ∧
  (∀ j, Function.Injective (λ i ↦ g i j))

-- Define the initial conditions
def initial_conditions (g : Grid) : Prop :=
  g 0 0 = 0 ∧ g 0 3 = 1 ∧ g 1 1 = 1 ∧ g 2 2 = 2

-- Theorem statement
theorem lower_right_is_one (g : Grid) 
  (h1 : is_valid_grid g) 
  (h2 : initial_conditions g) : 
  g 3 3 = 0 := by
  sorry

#check lower_right_is_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_one_l510_51002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_RST_collinear_l510_51097

/-- A point in the plane -/
structure Point : Type :=
  (x : ℝ) (y : ℝ)

/-- A circle in the plane -/
structure Circle : Type :=
  (center : Point) (radius : ℝ)

/-- Cyclic quadrilateral ABCD with diagonals intersecting at E -/
structure CyclicQuadrilateral : Type :=
  (A B C D E : Point)

/-- Circle Γ internally tangent to arc BC and tangent to BE and CE -/
noncomputable def circleGamma (ABCD : CyclicQuadrilateral) : Circle :=
  sorry

/-- Point T where Γ is tangent to arc BC -/
noncomputable def T (ABCD : CyclicQuadrilateral) : Point :=
  sorry

/-- Intersection of angle bisectors of ∠ABC and ∠BCD -/
noncomputable def R (ABCD : CyclicQuadrilateral) : Point :=
  sorry

/-- Incenter of triangle BCE -/
noncomputable def S (ABCD : CyclicQuadrilateral) : Point :=
  sorry

/-- Three points are collinear -/
def collinear (P Q R : Point) : Prop :=
  sorry

/-- Main theorem -/
theorem RST_collinear (ABCD : CyclicQuadrilateral) : 
  collinear (R ABCD) (S ABCD) (T ABCD) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_RST_collinear_l510_51097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iodine_mass_percentage_different_from_reference_l510_51066

-- Define the atomic masses
noncomputable def atomic_mass_Al : ℝ := 26.98
noncomputable def atomic_mass_I : ℝ := 126.90

-- Define the molar mass of AlI3
noncomputable def molar_mass_AlI3 : ℝ := atomic_mass_Al + 3 * atomic_mass_I

-- Define the mass percentage of iodine
noncomputable def mass_percentage_I : ℝ := (3 * atomic_mass_I / molar_mass_AlI3) * 100

-- Theorem statement
theorem iodine_mass_percentage_different_from_reference :
  mass_percentage_I ≠ 6.62 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_iodine_mass_percentage_different_from_reference_l510_51066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kings_tour_length_bounds_l510_51036

/-- Represents a king's move on a chessboard -/
inductive KingMove
| Straight
| Diagonal

/-- Represents a king's tour on a chessboard -/
def KingsTour := List KingMove

/-- The length of a king's tour -/
noncomputable def tourLength (tour : KingsTour) : ℝ :=
  tour.foldl (fun acc move => acc + match move with
    | KingMove.Straight => 1
    | KingMove.Diagonal => Real.sqrt 2) 0

theorem kings_tour_length_bounds :
  ∀ (tour : KingsTour),
    (tourLength tour ≥ 64) ∧
    (tourLength tour ≤ 28 + 36 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kings_tour_length_bounds_l510_51036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_value_l510_51007

theorem cos_product_value (x : ℝ) (h : Real.sin x = 3 * Real.sin (x - π/2)) :
  Real.cos x * Real.cos (x + π/2) = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_value_l510_51007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l510_51083

-- Define the function f(x) = x - a * ln(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x

-- State the theorem
theorem f_properties (a : ℝ) :
  -- Domain of f is (0, +∞)
  (∀ x > 0, f a x ∈ Set.Ioi 0) ∧
  -- When a = 2, the tangent line equation at x = 1 is x + y - 2 = 0
  (a = 2 → (let y := f 2 1; ∀ x y, y - 1 = -(x - 1) ↔ x + y - 2 = 0)) ∧
  -- For a ≤ 0, f(x) is strictly increasing on (0, +∞)
  (a ≤ 0 → StrictMono (f a)) ∧
  -- For a > 0, f(x) has a minimum value of a - a * ln(a) at x = a
  (a > 0 → (∃ x > 0, ∀ y > 0, f a x ≤ f a y) ∧
           (let x_min := a; f a x_min = a - a * Real.log a)) ∧
  -- For a > 0, f(x) has no maximum value
  (a > 0 → ¬∃ x > 0, ∀ y > 0, f a y ≤ f a x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l510_51083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l510_51031

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then (2 - a) * x + 2 else 2^x - 5 * a

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem f_increasing_iff_a_in_range (a : ℝ) :
  increasing_function (f a) ↔ a ∈ Set.Icc (-1/2) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l510_51031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_non_square_product_minus_one_l510_51084

theorem exist_non_square_product_minus_one (d : ℕ) 
  (h_pos : d > 0) (h_neq_2 : d ≠ 2) (h_neq_5 : d ≠ 5) (h_neq_13 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧ ¬∃ (k : ℕ), a * b - 1 = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_non_square_product_minus_one_l510_51084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_l510_51012

/-- The number of terms in the expansion of a product of two sums is equal to
    the product of the number of terms in each sum. -/
theorem expansion_terms_count (m n : ℕ) : m > 0 → n > 0 → m * n =
  (Finset.univ : Finset (Fin m × Fin n)).card := by
  intro hm hn
  rw [Finset.card_univ, Fintype.card_prod]
  simp [Fintype.card_fin]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_l510_51012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_on_c_l510_51009

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![x, 2]
def b : Fin 2 → ℝ := ![2, 1]
def c (x : ℝ) : Fin 2 → ℝ := ![3, x]

-- Define the parallel condition
def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ i, u i = k * v i

-- Define the dot product
def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

-- Define the magnitude of a vector
noncomputable def magnitude (u : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((u 0)^2 + (u 1)^2)

-- Define the projection
noncomputable def projection (u v : Fin 2 → ℝ) : ℝ :=
  (dot_product u v) / (magnitude v)

-- Theorem statement
theorem projection_of_a_on_c (x : ℝ) :
  parallel (a x) b →
  projection (a x) (c x) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_on_c_l510_51009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_le_semiperimeter_l510_51018

/-- A convex figure in the plane -/
structure ConvexFigure where
  dummy : Unit

/-- The area of a convex figure -/
noncomputable def area (f : ConvexFigure) : ℝ := sorry

/-- The semiperimeter of a convex figure -/
noncomputable def semiperimeter (f : ConvexFigure) : ℝ := sorry

/-- Predicate to check if a point is a lattice point -/
def isLatticePoint (p : ℝ × ℝ) : Prop :=
  ∃ (m n : ℤ), p = (↑m, ↑n)

/-- Predicate to check if a point is inside a convex figure -/
def isInside (p : ℝ × ℝ) (f : ConvexFigure) : Prop := sorry

/-- The main theorem -/
theorem area_le_semiperimeter (f : ConvexFigure) 
  (h : ∀ p, isLatticePoint p → ¬isInside p f) : 
  area f ≤ semiperimeter f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_le_semiperimeter_l510_51018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_net_profit_calculation_l510_51096

/-- Given an article with a purchase price, overhead percentage, and markup,
    calculate the net profit. -/
noncomputable def calculate_net_profit (purchase_price overhead_percent markup : ℚ) : ℚ :=
  markup - (overhead_percent / 100 * purchase_price)

/-- Theorem stating that given the specific values in the problem,
    the net profit is $32.80 -/
theorem net_profit_calculation :
  calculate_net_profit 48 15 40 = 328/10 := by
  -- Unfold the definition of calculate_net_profit
  unfold calculate_net_profit
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_net_profit_calculation_l510_51096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_theorem_shape_C_theorem_l510_51063

-- Define the fixed points and the moving point
def A₁ (a : ℝ) : ℝ × ℝ := (-a, 0)
def A₂ (a : ℝ) : ℝ × ℝ := (a, 0)
def M (x y : ℝ) : ℝ × ℝ := (x, y)

-- Define the slope product condition
def slope_product (a m : ℝ) (x y : ℝ) : Prop :=
  (y / (x + a)) * (y / (x - a)) = m

-- Define the curve C
def curve_C (a m : ℝ) (x y : ℝ) : Prop :=
  m * x^2 - y^2 = m * a^2

-- Define the shape of C based on m
noncomputable def shape_of_C (m : ℝ) : String :=
  if m < -1 then "ellipse with foci on y-axis"
  else if m = -1 then "circle"
  else if -1 < m ∧ m < 0 then "ellipse with foci on x-axis"
  else if m > 0 then "hyperbola with foci on x-axis"
  else "undefined"

-- Theorem statement
theorem curve_C_theorem (a m : ℝ) (h₁ : a > 0) (h₂ : m ≠ 0) :
  ∀ x y : ℝ, x ≠ a ∧ x ≠ -a →
  slope_product a m x y ↔ curve_C a m x y := by
  sorry

-- Theorem for the shape of C
theorem shape_C_theorem (a m : ℝ) (h₁ : a > 0) (h₂ : m ≠ 0) :
  ∀ x y : ℝ, curve_C a m x y → shape_of_C m ≠ "undefined" := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_theorem_shape_C_theorem_l510_51063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_two_sides_l510_51038

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the lengths of sides
noncomputable def side_length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

noncomputable def AB (t : Triangle) : ℝ := side_length t.A t.B
noncomputable def BC (t : Triangle) : ℝ := side_length t.B t.C
noncomputable def AC (t : Triangle) : ℝ := side_length t.A t.C

-- Define the angle at A
noncomputable def angle_A (t : Triangle) : ℝ :=
  Real.arccos ((AB t)^2 + (AC t)^2 - (BC t)^2) / (2 * AB t * AC t)

-- Theorem statement
theorem max_sum_of_two_sides (t : Triangle) :
  angle_A t = π / 3 → BC t = Real.sqrt 3 →
  AB t + AC t ≤ 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_two_sides_l510_51038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_rectangle_l510_51091

/-- Rectangle EFGH with given coordinates -/
structure Rectangle where
  E : ℝ × ℝ
  F : ℝ × ℝ
  H : ℝ × ℝ

/-- Calculate the area of the rectangle -/
def rectangleArea (r : Rectangle) : ℝ := sorry

/-- The specific rectangle from the problem -/
noncomputable def specificRectangle : Rectangle :=
  { E := (3, -5)
    F := (1003, 95)
    H := (5, -25) }

/-- Theorem stating that the area of the specific rectangle is 202000 -/
theorem area_of_specific_rectangle :
  rectangleArea specificRectangle = 202000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_rectangle_l510_51091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_road_trip_mpg_l510_51034

/-- Represents Jenna's road trip --/
structure RoadTrip where
  speed1 : ℚ  -- Speed for the first part of the trip (mph)
  time1 : ℚ   -- Time for the first part of the trip (hours)
  speed2 : ℚ  -- Speed for the second part of the trip (mph)
  time2 : ℚ   -- Time for the second part of the trip (hours)
  gasCost : ℚ -- Cost of gas per gallon ($)
  totalGasCost : ℚ -- Total amount spent on gas for the trip ($)

/-- Calculates the miles per gallon for a given road trip --/
def milesPerGallon (trip : RoadTrip) : ℚ :=
  let totalDistance := trip.speed1 * trip.time1 + trip.speed2 * trip.time2
  let gallonsUsed := trip.totalGasCost / trip.gasCost
  totalDistance / gallonsUsed

/-- Theorem stating that Jenna's road trip results in 30 miles per gallon --/
theorem jenna_road_trip_mpg :
  let trip : RoadTrip := {
    speed1 := 60
    time1 := 2
    speed2 := 50
    time2 := 3
    gasCost := 2
    totalGasCost := 18
  }
  milesPerGallon trip = 30 := by
  sorry

#eval milesPerGallon {
  speed1 := 60
  time1 := 2
  speed2 := 50
  time2 := 3
  gasCost := 2
  totalGasCost := 18
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_road_trip_mpg_l510_51034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equal_implies_sides_equal_cos_equal_implies_sides_equal_cos_double_equal_implies_sides_equal_l510_51051

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions for a valid triangle
def ValidTriangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi

-- Theorem stating that if sin A = sin B, then a = b
theorem sin_equal_implies_sides_equal (t : Triangle) (h : ValidTriangle t) :
  Real.sin t.A = Real.sin t.B → t.a = t.b := by sorry

-- Theorem stating that if cos A = cos B, then a = b
theorem cos_equal_implies_sides_equal (t : Triangle) (h : ValidTriangle t) :
  Real.cos t.A = Real.cos t.B → t.a = t.b := by sorry

-- Theorem stating that if cos 2A = cos 2B, then a = b
theorem cos_double_equal_implies_sides_equal (t : Triangle) (h : ValidTriangle t) :
  Real.cos (2 * t.A) = Real.cos (2 * t.B) → t.a = t.b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equal_implies_sides_equal_cos_equal_implies_sides_equal_cos_double_equal_implies_sides_equal_l510_51051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l510_51019

theorem cos_alpha_value (α : ℝ) 
  (h1 : Real.cos (α - π/6) = 15/17) 
  (h2 : α ∈ Set.Ioo (π/6) (π/2)) : 
  Real.cos α = (15 * Real.sqrt 3 - 8) / 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l510_51019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_cover_and_independent_set_l510_51047

/-- A directed graph. -/
structure DirectedGraph (V : Type) where
  edge : V → V → Prop

/-- A path in a directed graph. -/
def GraphPath (G : DirectedGraph V) : List V → Prop
  | [] => true
  | [_] => true
  | (v :: w :: rest) => G.edge v w ∧ GraphPath G (w :: rest)

/-- A path cover of a directed graph. -/
def PathCover (G : DirectedGraph V) (P : Set (List V)) : Prop :=
  (∀ p ∈ P, GraphPath G p) ∧ (∀ v : V, ∃ p ∈ P, v ∈ p)

/-- An independent set in a directed graph. -/
def IndependentSet (G : DirectedGraph V) (S : Set V) : Prop :=
  ∀ u v, u ∈ S → v ∈ S → u ≠ v → ¬G.edge u v ∧ ¬G.edge v u

/-- Main theorem: Every directed graph has a path cover and an independent set
    with the specified properties. -/
theorem path_cover_and_independent_set (G : DirectedGraph V) :
  ∃ (P : Set (List V)) (vP : List V → V),
    PathCover G P ∧
    IndependentSet G (Set.image vP P) ∧
    ∀ p ∈ P, vP p ∈ p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_cover_and_independent_set_l510_51047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_labeling_exists_l510_51058

-- Define a type for points in space
variable (Point : Type)

-- Define a function to represent the angle between three points
variable (angle : Point → Point → Point → ℝ)

-- Define the theorem
theorem angle_labeling_exists
  (n : ℕ)
  (points : Finset Point)
  (h_card : points.card = n)
  (h_angle : ∀ p q r, p ∈ points → q ∈ points → r ∈ points →
    p ≠ q → q ≠ r → p ≠ r →
    (angle p q r > 120 ∨ angle q r p > 120 ∨ angle r p q > 120)) :
  ∃ f : Fin n → Point,
    Function.Injective f ∧
    (∀ i j k : Fin n, i < j → j < k → angle (f i) (f j) (f k) > 120) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_labeling_exists_l510_51058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_perpendicular_lines_l510_51024

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  (h : 0 < b) (h' : b < a)

/-- A point on an ellipse -/
structure PointOnEllipse {a b : ℝ} (e : Ellipse a b) where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / a^2 + y^2 / b^2 = 1

/-- The focal length of an ellipse -/
noncomputable def focal_length {a b : ℝ} (e : Ellipse a b) : ℝ :=
  Real.sqrt (a^2 - b^2)

/-- The two foci of an ellipse -/
noncomputable def foci {a b : ℝ} (e : Ellipse a b) : ℝ × ℝ × ℝ × ℝ :=
  let c := focal_length e
  (c, 0, -c, 0)

/-- Theorem: Area of triangle formed by point and foci -/
theorem area_of_triangle_with_perpendicular_lines 
  {a b : ℝ} (e : Ellipse a b) (p : PointOnEllipse e) :
  let (x1, y1, x2, y2) := foci e
  (p.x - x1) * (p.x - x2) + (p.y - y1) * (p.y - y2) = 0 →
  abs ((x1 - x2) * p.y) / 2 = b * focal_length e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_perpendicular_lines_l510_51024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_mapping_theorem_l510_51080

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a type for triangles
structure Triangle :=
  (A B C : Point)

-- Define a type for circles
structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define a function type that maps triangles to circles
def TriangleToCircle := Triangle → Circle

-- Define what it means for a triangle to be non-degenerate
def NonDegenerateTriangle (t : Triangle) : Prop :=
  t.A ≠ t.B ∧ t.B ≠ t.C ∧ t.C ≠ t.A

-- Define what it means for four points to be in general position
def GeneralPosition (A B C D : Point) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Define the property of circles having a common point
def HaveCommonPoint (c1 c2 c3 c4 : Circle) : Prop :=
  ∃ (p : Point), (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
                 (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2 ∧
                 (p.1 - c3.center.1)^2 + (p.2 - c3.center.2)^2 = c3.radius^2 ∧
                 (p.1 - c4.center.1)^2 + (p.2 - c4.center.2)^2 = c4.radius^2

-- Define the nine-point circle of a triangle
noncomputable def NinePointCircle (t : Triangle) : Circle :=
  sorry -- Definition of nine-point circle

-- State the theorem
theorem triangle_mapping_theorem (f : TriangleToCircle) :
  (∀ t : Triangle, NonDegenerateTriangle t → (f t).radius > 0) →
  (∀ A B C D : Point, GeneralPosition A B C D →
    HaveCommonPoint (f ⟨A, B, C⟩) (f ⟨B, C, D⟩) (f ⟨C, D, A⟩) (f ⟨D, A, B⟩)) →
  ∀ t : Triangle, NonDegenerateTriangle t → f t = NinePointCircle t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_mapping_theorem_l510_51080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extreme_values_and_a_range_l510_51035

noncomputable def f (x : ℝ) := x^2 * Real.exp x

theorem f_extreme_values_and_a_range :
  (∃ (x_max : ℝ), x_max = -2 ∧ f x_max = 4 / Real.exp 2 ∧ ∀ x, f x ≤ f x_max) ∧
  (∃ (x_min : ℝ), x_min = 0 ∧ f x_min = 0 ∧ ∀ x, f x_min ≤ f x) ∧
  (∀ a : ℝ, (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧
    f x₁ = a * x₁ ∧ f x₂ = a * x₂ ∧ f x₃ = a * x₃) →
    -1 / Real.exp 1 < a ∧ a < 0) :=
by
  sorry

#check f_extreme_values_and_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extreme_values_and_a_range_l510_51035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_logs_l510_51093

theorem sum_of_reciprocal_logs (x y : ℝ) (h1 : (2 : ℝ)^x = 10) (h2 : (5 : ℝ)^y = 10) : 
  1/x + 1/y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_logs_l510_51093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_donation_multiple_l510_51078

/-- Represents the charity organization's food distribution problem -/
theorem charity_donation_multiple :
  -- Initial number of boxes
  ∀ (initial_boxes : ℕ),
  -- Cost of food per box
  ∀ (food_cost : ℚ),
  -- Cost of additional supplies per box
  ∀ (supplies_cost : ℚ),
  -- Total number of boxes after donation
  ∀ (total_boxes : ℕ),
  -- Conditions
  initial_boxes = 400 →
  food_cost = 80 →
  supplies_cost = 165 →
  total_boxes = 2000 →
  -- Conclusion: The donation multiple is 4
  (((total_boxes : ℚ) - (initial_boxes : ℚ)) * (food_cost + supplies_cost)) /
  ((initial_boxes : ℚ) * (food_cost + supplies_cost)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_donation_multiple_l510_51078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l510_51010

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define for 0 to avoid missing case
  | 1 => 1
  | n + 2 => sequence_a (n + 1) / (sequence_a (n + 1) + 2)

theorem a_10_value : sequence_a 10 = 1 / 1023 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l510_51010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_even_function_inequality_l510_51050

-- Define a 2-periodic even function
def is_2periodic_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x ∧ f (-x) = f x

-- Define the specific function for x ∈ [0, 1]
noncomputable def f_specific (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x^(1/110 : ℝ) else 0

-- Main theorem
theorem periodic_even_function_inequality 
  (f : ℝ → ℝ) 
  (h1 : is_2periodic_even f) 
  (h2 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = f_specific x) :
  f (101/17) < f (104/15) ∧ f (104/15) < f (98/19) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_even_function_inequality_l510_51050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_problem_solution_l510_51025

/-- Two ships traveling towards each other with constant speeds -/
structure ShipProblem where
  /-- Speed of the first ship (in distance units per hour) -/
  v1 : ℝ
  /-- Speed of the second ship (in distance units per hour) -/
  v2 : ℝ
  /-- Total distance between the two ports -/
  d : ℝ
  /-- Time taken by first ship to reach second port after meeting -/
  t1 : ℝ
  /-- Time taken by second ship to reach first port after meeting -/
  t2 : ℝ
  /-- Speeds are positive -/
  h1 : v1 > 0
  /-- Speeds are positive -/
  h2 : v2 > 0
  /-- Distance is positive -/
  h3 : d > 0
  /-- Times are positive -/
  h4 : t1 > 0
  /-- Times are positive -/
  h5 : t2 > 0
  /-- Constraint from problem statement -/
  h6 : t1 = 16
  /-- Constraint from problem statement -/
  h7 : t2 = 25

/-- The theorem stating the solution to the ship problem -/
theorem ship_problem_solution (p : ShipProblem) :
  (p.d / p.v1 = 36) ∧ (p.d / p.v2 = 45) := by
  sorry

/-- The specific instance of the problem as given -/
noncomputable def given_problem : ShipProblem := {
  v1 := 1 / 36
  v2 := 1 / 45
  d := 1
  t1 := 16
  t2 := 25
  h1 := by norm_num
  h2 := by norm_num
  h3 := by norm_num
  h4 := by norm_num
  h5 := by norm_num
  h6 := rfl
  h7 := rfl
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_problem_solution_l510_51025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_l510_51077

def A : ℂ := 3 - 2*Complex.I
def M : ℂ := -3 + 2*Complex.I
def S : ℂ := 2*Complex.I
def P : ℝ := -1
def R : ℂ := A * M

theorem complex_expression_equality :
  A - M + S - (P : ℂ) + R = 2 + 10*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_l510_51077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l510_51030

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l510_51030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_from_surface_area_l510_51057

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 →
  let a := (s / 6).sqrt
  s = 6 * a^2 →
  a^3 = (10 : ℝ)^3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_from_surface_area_l510_51057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sqrt_count_l510_51021

theorem ceiling_sqrt_count : 
  (Finset.filter (fun x : ℕ => ⌈Real.sqrt (x : ℝ)⌉ = 20) (Finset.range 400 \ Finset.range 361)).card = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sqrt_count_l510_51021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_property_l510_51016

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define points A, B, C, D, and P
variable (A B C D P : ℝ × ℝ)

-- Define that AB is a diameter of the circle
variable (h_diameter : A ∈ circle ∧ B ∈ circle ∧ IsCircleDiameter circle A B)

-- Define that AC and BD are chords of the circle
variable (h_chord_AC : A ∈ circle ∧ C ∈ circle)
variable (h_chord_BD : B ∈ circle ∧ D ∈ circle)

-- Define that P is the intersection point of AC and BD inside the circle
variable (h_intersection : P ∈ circle ∧ P ∈ Set.Icc A C ∧ P ∈ Set.Icc B D)

-- State the theorem
theorem chord_intersection_property :
  ‖A - B‖^2 = ‖A - C‖ * ‖A - P‖ + ‖B - D‖ * ‖B - P‖ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_property_l510_51016
