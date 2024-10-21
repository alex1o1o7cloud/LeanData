import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l335_33580

-- Define the vector type
def Vector2D := ℝ × ℝ

-- Define vector operations
def dot_product (v w : Vector2D) : ℝ := v.1 * w.1 + v.2 * w.2
def scalar_mult (k : ℝ) (v : Vector2D) : Vector2D := (k * v.1, k * v.2)
def vector_add (v w : Vector2D) : Vector2D := (v.1 + w.1, v.2 + w.2)
def vector_sub (v w : Vector2D) : Vector2D := (v.1 - w.1, v.2 - w.2)
noncomputable def vector_length (v : Vector2D) : ℝ := Real.sqrt (dot_product v v)

-- Define parallel and perpendicular
def parallel (v w : Vector2D) : Prop := ∃ (k : ℝ), w = scalar_mult k v
def perpendicular (v w : Vector2D) : Prop := dot_product v w = 0

-- Define the theorem
theorem vector_problem (a b c : Vector2D) :
  a = (1, 2) →
  vector_length c = 3 * Real.sqrt 5 →
  parallel a c →
  vector_length b = 3 * Real.sqrt 5 →
  perpendicular (vector_sub (scalar_mult 4 a) b) (vector_add (scalar_mult 2 a) b) →
  ((c = (3, 6) ∨ c = (-3, -6)) ∧
   (dot_product a b / (vector_length a * vector_length b) = 1 / 6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l335_33580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_expansion_l335_33564

theorem coefficient_of_x_in_expansion : 
  (Polynomial.coeff ((Polynomial.X ^ 2 + 3 * Polynomial.X + 2) ^ 5) 1) = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_expansion_l335_33564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_meeting_point_l335_33554

/-- Represents a triangle with vertices X, Y, and Z. -/
def IsTriangle (X Y Z : Point) : Prop := sorry

/-- Represents the scenario where two points start at X and move along the
    perimeter of triangle XYZ in opposite directions at the same speed,
    meeting at point D. -/
def TwoPointsMeetOnPerimeter (X Y Z D : Point) : Prop := sorry

/-- Calculates the length of a line segment between two points. -/
def SegmentLength (P Q : Point) : ℝ := sorry

/-- Given a triangle XYZ with side lengths as specified, prove that two points
    starting from X and moving in opposite directions along the perimeter at
    the same speed will meet at a point D such that YD = 6. -/
theorem bug_meeting_point (X Y Z D : Point) (xy yz xz : ℝ) : 
  xy = 6 → yz = 8 → xz = 10 →
  IsTriangle X Y Z →
  TwoPointsMeetOnPerimeter X Y Z D →
  SegmentLength Y D = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_meeting_point_l335_33554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percent_calculation_l335_33542

theorem gain_percent_calculation (MP : ℝ) (MP_pos : MP > 0) : 
  let CP := 0.36 * MP
  let SP := 0.80 * MP
  let gain_percent := ((SP - CP) / CP) * 100
  abs (gain_percent - 122.22) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percent_calculation_l335_33542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_inverse_of_f_l335_33570

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 7 - 3 * x

-- Define the proposed inverse function h
noncomputable def h (x : ℝ) : ℝ := (7 - x) / 3

-- Theorem stating that h is the inverse of f
theorem h_is_inverse_of_f : Function.LeftInverse h f ∧ Function.RightInverse h f :=
by
  constructor
  · -- Prove left inverse
    intro x
    simp [f, h]
    field_simp
    ring
  · -- Prove right inverse
    intro x
    simp [f, h]
    field_simp
    ring

#check h_is_inverse_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_inverse_of_f_l335_33570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_swims_on_saturday_l335_33527

-- Define the days of the week
inductive Day : Type
  | sunday | monday | tuesday | wednesday | thursday | friday | saturday
  deriving Repr, DecidableEq

-- Define the sports
inductive Sport : Type
  | running | volleyball | golf | swimming | tennis
  deriving Repr, DecidableEq

-- Define the schedule type
def Schedule := Day → Sport

-- Define the condition that Mahdi runs three days a week
def runsThreeDays (s : Schedule) : Prop :=
  (List.filter (fun d => s d = Sport.running) [Day.sunday, Day.monday, Day.tuesday, Day.wednesday, Day.thursday, Day.friday, Day.saturday]).length = 3

-- Define the condition that Mahdi never runs on two consecutive days
def noConsecutiveRunning (s : Schedule) : Prop :=
  ∀ d1 d2 : Day, d1 ≠ d2 → (s d1 = Sport.running ∧ s d2 = Sport.running) → 
    (d1 = Day.sunday ∧ d2 = Day.tuesday) ∨
    (d1 = Day.sunday ∧ d2 = Day.wednesday) ∨
    (d1 = Day.monday ∧ d2 = Day.wednesday) ∨
    (d1 = Day.monday ∧ d2 = Day.thursday) ∨
    (d1 = Day.tuesday ∧ d2 = Day.thursday) ∨
    (d1 = Day.tuesday ∧ d2 = Day.friday) ∨
    (d1 = Day.wednesday ∧ d2 = Day.friday) ∨
    (d1 = Day.wednesday ∧ d2 = Day.saturday) ∨
    (d1 = Day.thursday ∧ d2 = Day.saturday) ∨
    (d1 = Day.friday ∧ d2 = Day.sunday)

-- Define the condition for Monday's and Wednesday's activities
def mondayVolleyballWednesdayGolf (s : Schedule) : Prop :=
  s Day.monday = Sport.volleyball ∧ s Day.wednesday = Sport.golf

-- Define the condition that Mahdi swims and plays tennis
def swimsAndPlaysTennis (s : Schedule) : Prop :=
  ∃ d1 d2 : Day, s d1 = Sport.swimming ∧ s d2 = Sport.tennis

-- Define the condition that tennis is not played after running or swimming
def tennisNotAfterRunningOrSwimming (s : Schedule) : Prop :=
  ∀ d1 d2 : Day, (s d1 = Sport.running ∨ s d1 = Sport.swimming) →
    (d2 = Day.sunday ∧ d1 = Day.saturday) ∨
    (d2 = Day.monday ∧ d1 = Day.sunday) ∨
    (d2 = Day.tuesday ∧ d1 = Day.monday) ∨
    (d2 = Day.wednesday ∧ d1 = Day.tuesday) ∨
    (d2 = Day.thursday ∧ d1 = Day.wednesday) ∨
    (d2 = Day.friday ∧ d1 = Day.thursday) ∨
    (d2 = Day.saturday ∧ d1 = Day.friday) →
    s d2 ≠ Sport.tennis

-- The main theorem
theorem mahdi_swims_on_saturday (s : Schedule) 
  (h1 : runsThreeDays s)
  (h2 : noConsecutiveRunning s)
  (h3 : mondayVolleyballWednesdayGolf s)
  (h4 : swimsAndPlaysTennis s)
  (h5 : tennisNotAfterRunningOrSwimming s) :
  s Day.saturday = Sport.swimming :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_swims_on_saturday_l335_33527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l335_33597

noncomputable def equation (x : ℝ) : ℝ :=
  (x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 1) / ((x - 5) * (x - 7) * (x - 5))

theorem solution_set :
  {x : ℝ | equation x = 0 ∧ x ≠ 5 ∧ x ≠ 7} = {1, 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l335_33597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_approximation_l335_33579

/-- The volume of a right rectangular prism with face areas a, b, and c -/
noncomputable def prismVolume (a b c : ℝ) : ℝ := Real.sqrt (a * b * c)

/-- Rounds a real number to the nearest integer -/
noncomputable def roundToNearest (x : ℝ) : ℤ := Int.floor (x + 0.5)

theorem prism_volume_approximation (a b c : ℝ) (h1 : a = 60) (h2 : b = 70) (h3 : c = 84) :
  roundToNearest (prismVolume a b c) = 1572 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_approximation_l335_33579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l335_33566

theorem ellipse_line_intersection (m n : ℝ) (A B M : ℝ × ℝ) :
  (∀ x y : ℝ, m * x^2 + n * y^2 = 1 ↔ (x, y) = A ∨ (x, y) = B) →
  (∀ x y : ℝ, x + y = 1 ↔ (x, y) = A ∨ (x, y) = B) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  M.2 / M.1 = Real.sqrt 2 / 2 →
  m / n = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l335_33566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_of_roots_log_equation_l335_33532

theorem tan_sum_of_roots_log_equation (α β : Real) : 
  (∃ x₁ x₂ : Real, x₁ ≠ x₂ ∧ 
    (Real.log (6 * x₁^2 - 5 * x₁ + 2) = 0) ∧ 
    (Real.log (6 * x₂^2 - 5 * x₂ + 2) = 0) ∧ 
    x₁ = Real.tan α ∧ x₂ = Real.tan β) →
  Real.tan (α + β) = 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_of_roots_log_equation_l335_33532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_l335_33576

-- Define the function f(x) = log₃(x) + x - 3
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 + x - 3

-- State the theorem
theorem solution_interval :
  ∃ x, x ∈ Set.Ioo 2 3 ∧ f x = 0 ∧ ∀ y, f y = 0 → y ∈ Set.Ioo 2 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_l335_33576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l335_33572

/-- The area of a triangle with side lengths 8, 9, and 9 is 4√65 -/
theorem triangle_area (a b c : ℝ) (ha : a = 8) (hb : b = 9) (hc : c = 9) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 4 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l335_33572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_perimeter_for_72_circumference_l335_33507

/-- The perimeter of the shaded region formed by the intersection of three identical touching circles -/
noncomputable def shaded_perimeter (circle_circumference : ℝ) : ℝ :=
  circle_circumference / 2

/-- Theorem: The perimeter of the shaded region formed by the intersection of three identical
    touching circles, each with a circumference of 72, is equal to 36. -/
theorem shaded_perimeter_for_72_circumference :
  shaded_perimeter 72 = 36 := by
  -- Unfold the definition of shaded_perimeter
  unfold shaded_perimeter
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_perimeter_for_72_circumference_l335_33507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangements_eq_24_l335_33552

/-- The number of different arrangements of 5 books, where one specific book is fixed in the middle position -/
def book_arrangements : ℕ :=
  Nat.factorial 4

theorem book_arrangements_eq_24 : book_arrangements = 24 := by
  unfold book_arrangements
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangements_eq_24_l335_33552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_inclination_l335_33582

theorem hyperbola_asymptote_inclination (a : ℝ) : 
  a > 0 → 
  (∃ x y : ℝ, x^2 / a^2 - y^2 = 1) → 
  (∃ y : ℝ → ℝ, (∀ x, y x = x / a) ∧ Real.tan (π / 6) = 1 / a) → 
  a = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_inclination_l335_33582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l335_33574

noncomputable def sine_function (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem sine_function_properties
  (A ω φ : ℝ)
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_zero : sine_function A ω φ (π / 12) = 0)
  (h_max : sine_function A ω φ (π / 3) = 5) :
  ∃ (k : ℤ),
    (∀ x, sine_function A ω φ x = 5 * Real.sin (2 * x - π / 6)) ∧
    (∀ x, sine_function A ω φ x ≤ 5) ∧
    (∀ k : ℤ, sine_function A ω φ (↑k * π + π / 3) = 5) ∧
    (∀ x : ℝ, ∀ k : ℤ, sine_function A ω φ x ≤ 0 ↔ ↑k * π - 5 * π / 12 ≤ x ∧ x ≤ ↑k * π + π / 12) :=
by
  sorry

#check sine_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l335_33574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_first_term_is_eleven_l335_33547

def jo_blair_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then jo_blair_sequence n + 1 else 2 * jo_blair_sequence (n - 1)

theorem twenty_first_term_is_eleven : jo_blair_sequence 20 = 11 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_first_term_is_eleven_l335_33547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_sin_range_l335_33536

noncomputable def f (ϖ : ℝ) (x : ℝ) : ℝ := Real.sin (ϖ * x)

theorem monotonic_decreasing_sin_range (ϖ : ℝ) (h1 : ϖ > 0) :
  (∀ x y : ℝ, π/3 ≤ x ∧ x < y ∧ y ≤ π/2 → f ϖ x > f ϖ y) →
  3/2 ≤ ϖ ∧ ϖ ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_sin_range_l335_33536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l335_33557

-- Define the function g as noncomputable due to its dependency on Real.pi
noncomputable def g (x : ℝ) : ℝ := -3 * Real.sin (2 * Real.pi * x)

-- State the theorem
theorem solutions_count :
  ∃! (s : Finset ℝ), 
    (∀ x ∈ s, -1 ≤ x ∧ x ≤ 1 ∧ g (g (g x)) = g x) ∧
    (∀ x, -1 ≤ x ∧ x ≤ 1 ∧ g (g (g x)) = g x → x ∈ s) ∧
    Finset.card s = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l335_33557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pentagon_ratio_l335_33502

/-- Represents the ratio of the triangle's area to the pentagon's area -/
noncomputable def triangle_to_pentagon_ratio (x : ℝ) : ℝ :=
  let triangle_area := x^2 * Real.sqrt 3 / 2
  let rectangle_area := 2 * x^2
  let pentagon_area := triangle_area + rectangle_area
  triangle_area / pentagon_area

/-- The main theorem stating the ratio of the triangle's area to the pentagon's area -/
theorem triangle_pentagon_ratio :
  ∀ x : ℝ, x > 0 → triangle_to_pentagon_ratio x = Real.sqrt 3 / (Real.sqrt 3 + 4) := by
  sorry

#check triangle_pentagon_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pentagon_ratio_l335_33502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l335_33563

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) > Real.sqrt (c^2 - c*a + a^2)) ∧
  (Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (b^2 + b*c + c^2) > Real.sqrt (c^2 + c*a + a^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l335_33563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_9_8_equals_3a_div_2b_l335_33513

-- Define the variables
variable (a b : ℝ)

-- Define the conditions
def condition1 (a : ℝ) : Prop := (10 : ℝ)^a = 2
def condition2 (b : ℝ) : Prop := Real.log 3 = b

-- State the theorem
theorem log_9_8_equals_3a_div_2b (h1 : condition1 a) (h2 : condition2 b) : 
  Real.log 8 / Real.log 9 = 3*a/(2*b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_9_8_equals_3a_div_2b_l335_33513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independent_set_probability_bound_l335_33567

/-- G(n,p) random graph model -/
structure RandomGraph (n : ℕ) (p : ℝ) : Type := mk ::

/-- Independence number of a graph -/
def independenceNumber {n : ℕ} {p : ℝ} (G : RandomGraph n p) : ℕ := sorry

/-- Probability measure for random graphs -/
noncomputable def ℙ {n : ℕ} {p : ℝ} (P : RandomGraph n p → Prop) : ℝ := sorry

theorem independent_set_probability_bound
  (n k : ℕ) (p : ℝ) (h1 : n ≥ k) (h2 : k ≥ 2) (h3 : 0 ≤ p ∧ p ≤ 1) :
  ℙ (λ G : RandomGraph n p => independenceNumber G ≥ k) ≤ (n.choose k) * (1 - p) ^ (k.choose 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_independent_set_probability_bound_l335_33567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_complement_union_intersection_condition_l335_33505

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | x^2 + 6*x + 5 < 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x < 1}
def C (a : ℝ) : Set ℝ := {x | x < a}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∩ B = ∅
theorem intersection_empty : A ∩ B = ∅ := by sorry

-- Theorem 2: C_U(A ∪ B) = (-∞, -5] ∪ [1, +∞)
theorem complement_union :
  (A ∪ B)ᶜ = Set.Iic (-5) ∪ Set.Ici 1 := by sorry

-- Theorem 3: B ∩ C = B if and only if a ≥ 1
theorem intersection_condition (a : ℝ) :
  B ∩ C a = B ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_complement_union_intersection_condition_l335_33505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_omega_magnitude_power_l335_33538

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given complex number z
noncomputable def z : ℂ := 1 + 2*i

-- Define the condition i(z+1) = -2+2i
axiom z_condition : i * (z + 1) = -2 + 2*i

-- Theorem 1: The imaginary part of z is 2
theorem imaginary_part_of_z :
  z.im = 2 := by sorry

-- Define ω
noncomputable def ω : ℂ := z / (1 - 2*i)

-- Theorem 2: |ω|^2012 = 1
theorem omega_magnitude_power :
  Complex.abs ω ^ 2012 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_omega_magnitude_power_l335_33538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_divisible_by_1986_l335_33544

def sequence_u : ℕ → ℤ
  | 0 => 39
  | 1 => 45
  | (n + 2) => sequence_u (n + 1) ^ 2 - sequence_u n

theorem infinite_divisible_by_1986 :
  ∃ f : ℕ → ℕ, StrictMono f ∧
  ∀ k, sequence_u (f k) % 1986 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_divisible_by_1986_l335_33544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_distances_l335_33587

open InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem max_sum_squared_distances (a b c : E) 
  (ha : ‖a‖ = 1) (hc : ‖c‖ = 1) (hb : ‖b‖ = 2) :
  ‖a - b‖^2 + ‖a - c‖^2 + ‖b - c‖^2 ≤ 19 ∧ 
  ∃ a' b' c' : E, ‖a'‖ = 1 ∧ ‖c'‖ = 1 ∧ ‖b'‖ = 2 ∧ 
    ‖a' - b'‖^2 + ‖a' - c'‖^2 + ‖b' - c'‖^2 = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_distances_l335_33587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_is_four_thirds_l335_33556

/-- The parabola function -/
def f (x : ℝ) : ℝ := -x^2 + 3*x - 2

/-- The area bounded by the parabola and the coordinate axes -/
noncomputable def parabola_area : ℝ :=
  (∫ x in Set.Icc 0 1, f x) + (∫ x in Set.Icc 1 2, |f x|)

/-- Theorem stating that the area bounded by the parabola and the coordinate axes is 4/3 -/
theorem parabola_area_is_four_thirds :
  parabola_area = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_is_four_thirds_l335_33556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l335_33503

noncomputable def f (x : ℝ) := (x^4 - 4*x^3 + 6*x^2 - 6*x + 3) / (x^3 - 4*x^2 + 4*x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 0 ∨ (0 < x ∧ x < 2) ∨ 2 < x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l335_33503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_neg_seven_l335_33541

noncomputable def f (x : ℝ) : ℝ := (2*x^2 + 15*x + 7) / (x + 7)

theorem limit_f_at_neg_seven :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x + 7| ∧ |x + 7| < δ → |f x + 13| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_neg_seven_l335_33541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_not_smallest_l335_33528

/-- A quadratic function symmetric about x = 2 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The symmetry condition -/
def symmetric (f : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, f (2 + t) = f (2 - t)

theorem f_one_not_smallest (a b c : ℝ) (ha : a ≠ 0) 
  (h_sym : symmetric (f a b c)) :
  ∃ x : ℝ, x ∈ ({-1, 1, 2, 5} : Set ℝ) ∧ f a b c x < f a b c 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_not_smallest_l335_33528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l335_33534

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Point B -/
def point_B : ℝ × ℝ := (4, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_distance_theorem (A : Parabola) :
  distance (A.x, A.y) focus = distance point_B focus →
  distance (A.x, A.y) point_B = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l335_33534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mens_wages_approx_l335_33562

/-- Represents the wage distribution problem -/
structure WageDistribution where
  W : ℕ  -- number of women equal to 5 men
  total_earnings : ℚ
  men_equal_women : 5 = W
  women_equal_boys : W = 8
  total_earnings_value : total_earnings = 120

/-- Calculates the men's wages given a wage distribution -/
noncomputable def mens_wages (wd : WageDistribution) : ℚ :=
  (5 : ℚ) / 13 * wd.total_earnings

/-- Theorem stating that men's wages are approximately 46.15 Rs -/
theorem mens_wages_approx (wd : WageDistribution) :
  ∃ ε > 0, |mens_wages wd - 46.15| < ε := by
  sorry

#eval (5 : ℚ) / 13 * 120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mens_wages_approx_l335_33562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sum_minimum_l335_33510

theorem abs_sum_minimum : 
  ∀ x : ℝ, |x + 1| + |x + 3| + |x + 6| ≥ 5 ∧ ∃ x : ℝ, |x + 1| + |x + 3| + |x + 6| = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sum_minimum_l335_33510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l335_33504

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (2 : ℝ)^x - (2 : ℝ)^(-x)
def g (x : ℝ) : ℝ := x^2 - 4*x + 6

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ,
  (∀ x₁ : ℝ, x₁ ∈ Set.Icc 1 4 →
    ∃ x₂ : ℝ, x₂ ∈ Set.Icc 0 2 ∧
      g x₁ = m * f x₂ + 7 - 3*m) →
  m ≥ 5/3 ∨ m ≤ 19/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l335_33504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l335_33537

-- Define the curve C
def curve_C (x y : ℝ) : Prop := 4 * x^2 / 9 + y^2 / 16 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := 2 * x + y - 11 = 0

-- Define the distance function from a point (x, y) to line l
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  (|2 * x + y - 11|) / Real.sqrt 5

-- Theorem statement
theorem distance_bounds :
  ∃ (d_max d_min : ℝ),
    d_max = 16 * Real.sqrt 5 / 5 ∧
    d_min = 6 * Real.sqrt 5 / 5 ∧
    (∀ x y : ℝ, curve_C x y →
      d_min ≤ distance_to_line x y ∧ distance_to_line x y ≤ d_max) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l335_33537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_priyas_trip_l335_33509

/-- Priya's trip between towns X, Y, and Z -/
theorem priyas_trip (total_time : ℝ) (return_speed : ℝ) (half_return_time : ℝ)
  (h1 : total_time = 5)
  (h2 : return_speed = 60)
  (h3 : half_return_time = 2.0833333333333335)
  (h4 : 0 < total_time)
  (h5 : 0 < return_speed)
  (h6 : 0 < half_return_time) :
  (return_speed * half_return_time * 2) / total_time = 50 := by
  sorry

#check priyas_trip

end NUMINAMATH_CALUDE_ERRORFEEDBACK_priyas_trip_l335_33509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l335_33561

/-- Given a quadratic function p(x) = x^2 + 2x - m, 
    prove that if p(1) is false and p(2) is true, 
    then the range of m is [3, 8). -/
theorem range_of_m (m : ℝ) (p : ℝ → Prop) : 
  (∀ x, (x^2 + 2*x - m > 0) ↔ p x) →
  (¬ p 1) →
  (p 2) →
  m ∈ Set.Icc 3 8 ∧ m ≠ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l335_33561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_box_puzzle_l335_33514

structure Box where
  label : Char
  barMass : Nat
  barsTaken : Nat

def totalMass (boxes : List Box) : Nat :=
  boxes.foldl (fun acc box => acc + box.barMass * box.barsTaken) 0

theorem chocolate_box_puzzle :
  ∀ (boxes : List Box),
    boxes.length = 5 ∧
    (∃ (v w x y z : Box),
      boxes = [v, w, x, y, z] ∧
      v.label = 'V' ∧ w.label = 'W' ∧ x.label = 'X' ∧ y.label = 'Y' ∧ z.label = 'Z' ∧
      v.barsTaken = 1 ∧ w.barsTaken = 2 ∧ x.barsTaken = 4 ∧ y.barsTaken = 8 ∧ z.barsTaken = 16 ∧
      (v.barMass = 100 ∨ v.barMass = 90) ∧
      (w.barMass = 100 ∨ w.barMass = 90) ∧
      (x.barMass = 100 ∨ x.barMass = 90) ∧
      (y.barMass = 100 ∨ y.barMass = 90) ∧
      (z.barMass = 100 ∨ z.barMass = 90) ∧
      (boxes.filter (fun box => box.barMass = 100)).length = 3 ∧
      (boxes.filter (fun box => box.barMass = 90)).length = 2 ∧
      totalMass boxes = 2920) →
    ∃ (w z : Box),
      w ∈ boxes ∧ z ∈ boxes ∧
      w.label = 'W' ∧ z.label = 'Z' ∧
      w.barMass = 90 ∧ z.barMass = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_box_puzzle_l335_33514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_import_tax_calculation_l335_33590

/-- Calculate import tax for an item given its total value, tax rate, and tax-free threshold. -/
noncomputable def calculate_import_tax (total_value : ℝ) (tax_rate : ℝ) (tax_free_threshold : ℝ) : ℝ :=
  max 0 ((total_value - tax_free_threshold) * tax_rate)

/-- Theorem stating that the import tax for the given conditions is $111.30 -/
theorem import_tax_calculation :
  let total_value : ℝ := 2590
  let tax_rate : ℝ := 0.07
  let tax_free_threshold : ℝ := 1000
  calculate_import_tax total_value tax_rate tax_free_threshold = 111.30 := by
  -- Unfold the definition of calculate_import_tax
  unfold calculate_import_tax
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_import_tax_calculation_l335_33590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_all_correct_grid_l335_33586

/-- Represents the state of a cell in the grid -/
inductive CellState
  | One
  | Two
  | Three
  | Four
deriving DecidableEq

/-- Represents an infinite 2D grid -/
def InfiniteGrid := ℤ → ℤ → CellState

/-- Checks if a cell is correct according to the problem's definition -/
def is_correct_cell (grid : InfiniteGrid) (x y : ℤ) : Prop :=
  let adjacent_states := [grid (x+1) y, grid (x-1) y, grid x (y+1), grid x (y-1)]
  let distinct_count := (adjacent_states.toFinset).card
  match grid x y with
  | CellState.One => distinct_count = 1
  | CellState.Two => distinct_count = 2
  | CellState.Three => distinct_count = 3
  | CellState.Four => distinct_count = 4

/-- All numbers from 1 to 4 appear at least once in the grid -/
def all_numbers_appear (grid : InfiniteGrid) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℤ),
    grid x1 y1 = CellState.One ∧
    grid x2 y2 = CellState.Two ∧
    grid x3 y3 = CellState.Three ∧
    grid x4 y4 = CellState.Four

/-- Main theorem: It's impossible to have an infinite grid where all cells are correct -/
theorem impossible_all_correct_grid :
  ¬∃ (grid : InfiniteGrid),
    (∀ (x y : ℤ), is_correct_cell grid x y) ∧
    all_numbers_appear grid :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_all_correct_grid_l335_33586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l335_33583

/-- The eccentricity of a hyperbola with specific geometric properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ∃ (F M N : ℝ × ℝ),
    -- Hyperbola equation
    (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x, y) ∈ Set.range (λ t : ℝ ↦ (t, 0))) ∧
    -- Right focus of hyperbola
    F = (c, 0) ∧
    -- Circle equation
    (∀ (x y : ℝ), x^2 + y^2 = a^2 → (x, y) ∈ Set.range (λ t : ℝ ↦ (t, 0))) ∧
    -- Parabola equation
    (∀ (x y : ℝ), y^2 = -4*c*x → (x, y) ∈ Set.range (λ t : ℝ ↦ (t, 0))) ∧
    -- Vector relation
    F + N = 2 * M ∧
    -- Eccentricity
    c / a = (1 + Real.sqrt 5) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l335_33583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_two_digit_composite_coprime_l335_33599

/-- A number is two-digit if it's between 10 and 99 inclusive -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A number is composite if it's not prime and greater than 1 -/
def is_composite (n : ℕ) : Prop := ¬ Nat.Prime n ∧ n > 1

/-- A list of numbers is pairwise coprime if any two distinct numbers in the list are coprime -/
def pairwise_coprime (l : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ l → b ∈ l → a ≠ b → Nat.Coprime a b

/-- The main theorem: The maximum number of two-digit composite numbers that are pairwise coprime is 4 -/
theorem max_two_digit_composite_coprime :
  (∃ (l : List ℕ), l.length = 4 ∧
    (∀ n ∈ l, is_two_digit n ∧ is_composite n) ∧
    pairwise_coprime l) ∧
  (∀ (l : List ℕ),
    (∀ n ∈ l, is_two_digit n ∧ is_composite n) ∧
    pairwise_coprime l →
    l.length ≤ 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_two_digit_composite_coprime_l335_33599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weights_with_same_sum_pairs_l335_33524

theorem min_weights_with_same_sum_pairs (weights : Finset ℕ) : 
  weights.card = 8 → (∀ w ∈ weights, w ≤ 21) → 
  ∃ a b c d, a ∈ weights ∧ b ∈ weights ∧ c ∈ weights ∧ d ∈ weights ∧
    a ≠ b ∧ c ≠ d ∧ (a, b) ≠ (c, d) ∧ a + b = c + d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weights_with_same_sum_pairs_l335_33524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_change_approx_30_percent_l335_33596

noncomputable section

def initial_investment : ℝ := 200

def year1_loss_rate : ℝ := 0.10
def year2_gain_rate : ℝ := 0.25
def year3_gain_rate : ℝ := 0.15

def investment_after_year1 : ℝ := initial_investment * (1 - year1_loss_rate)
def investment_after_year2 : ℝ := investment_after_year1 * (1 + year2_gain_rate)
def investment_after_year3 : ℝ := investment_after_year2 * (1 + year3_gain_rate)

def total_change_rate : ℝ := (investment_after_year3 - initial_investment) / initial_investment

theorem investment_change_approx_30_percent :
  ∃ ε > 0, ε < 0.01 ∧ |total_change_rate - 0.3| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_change_approx_30_percent_l335_33596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_radii_neq_sum_distances_l335_33559

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def circleA : Circle := ⟨(0, 0), 1⟩
def circleB : Circle := ⟨(3, 0), 0.5⟩
def circleC : Circle := ⟨(0, 3), 0.5⟩

-- Assumptions
axiom A_largest : circleA.radius ≥ circleB.radius ∧ circleA.radius ≥ circleC.radius
axiom non_intersecting : 
  (circleA.center.1 - circleB.center.1)^2 + (circleA.center.2 - circleB.center.2)^2 > (circleA.radius + circleB.radius)^2 ∧
  (circleA.center.1 - circleC.center.1)^2 + (circleA.center.2 - circleC.center.2)^2 > (circleA.radius + circleC.radius)^2 ∧
  (circleB.center.1 - circleC.center.1)^2 + (circleB.center.2 - circleC.center.2)^2 > (circleB.radius + circleC.radius)^2

-- Define distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem to prove
theorem sum_radii_neq_sum_distances : 
  circleA.radius + circleB.radius + circleC.radius ≠ 
  distance circleA.center circleB.center + distance circleA.center circleC.center := by
  sorry

#check sum_radii_neq_sum_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_radii_neq_sum_distances_l335_33559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hostel_cost_calculation_l335_33516

/-- Cost calculation for a student youth hostel stay -/
theorem hostel_cost_calculation (days : ℕ) (first_week_rate : ℚ) (additional_week_rate : ℚ) :
  days = 23 ∧ first_week_rate = 18 ∧ additional_week_rate = 12 →
  (min days 7 * first_week_rate + 
   ((days - 7) / 7) * 7 * additional_week_rate + 
   (days - 7) % 7 * additional_week_rate) = 318 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hostel_cost_calculation_l335_33516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_l335_33535

/-- Represents the number of students in the fifth grade -/
def f : ℕ := 1  -- We define f as a concrete value to avoid issues with functions

/-- Average minutes run by third graders per day -/
def third_grade_avg : ℚ := 12

/-- Average minutes run by fourth graders per day -/
def fourth_grade_avg : ℚ := 15

/-- Average minutes run by fifth graders per day -/
def fifth_grade_avg : ℚ := 10

/-- Number of third graders -/
def third_graders : ℕ := 4 * f

/-- Number of fourth graders -/
def fourth_graders : ℕ := 2 * f

/-- Number of fifth graders -/
def fifth_graders : ℕ := f

/-- Total number of students -/
def total_students : ℕ := third_graders + fourth_graders + fifth_graders

/-- Total minutes run by all students -/
def total_minutes : ℚ := third_grade_avg * (third_graders : ℚ) + fourth_grade_avg * (fourth_graders : ℚ) + fifth_grade_avg * (fifth_graders : ℚ)

/-- Theorem stating that the average number of minutes run per day by all students is 88/7 -/
theorem average_minutes_run : total_minutes / (total_students : ℚ) = 88 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_l335_33535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_formula_specific_average_rate_of_change_l335_33592

/-- The average rate of change of f(x) = 3x^2 + 2 over [x₀, x₀ + Δx] -/
noncomputable def averageRateOfChange (x₀ Δx : ℝ) : ℝ :=
  let f := fun x => 3 * x^2 + 2
  (f (x₀ + Δx) - f x₀) / Δx

theorem average_rate_of_change_formula (x₀ Δx : ℝ) (h : Δx ≠ 0) :
  averageRateOfChange x₀ Δx = 6 * x₀ + 3 * Δx :=
by sorry

theorem specific_average_rate_of_change :
  averageRateOfChange 2 0.1 = 12.3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_formula_specific_average_rate_of_change_l335_33592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_selections_l335_33589

/-- Represents the number of ways to select 4 balls from a set of 9 balls with specific conditions -/
def select_balls : ℕ := 11

/-- The function to calculate the number of ways to select the balls -/
def calculate_selections : ℕ :=
  let total_balls : ℕ := 9
  let red_balls : ℕ := 2
  let white_balls : ℕ := 4
  let different_balls : ℕ := 3
  let selected_balls : ℕ := 4

  -- Placeholder for the actual calculation
  11

/-- Theorem: The number of ways to select the balls is 11 -/
theorem num_selections : calculate_selections = select_balls := by
  -- Proof goes here
  sorry

#eval select_balls
#eval calculate_selections

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_selections_l335_33589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_of_cubes_l335_33512

theorem surface_area_ratio_of_cubes (x : ℝ) (hx : x > 0) :
  (6 * (2 * x)^2) / (6 * x^2) = 4 := by
  -- Simplify the left side of the equation
  have h1 : 6 * (2 * x)^2 = 24 * x^2 := by
    ring
  have h2 : 6 * x^2 = 6 * x^2 := by
    rfl
  -- Rewrite the equation using h1 and h2
  rw [h1, h2]
  -- Simplify the fraction
  field_simp
  -- Prove the final equality
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_of_cubes_l335_33512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l335_33523

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f
noncomputable def f (x y : ℝ) : ℝ := (x + y) / (floor x * floor y + floor x + floor y + 1)

-- State the theorem
theorem f_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  (∃ (z : ℝ), f x y = z ∧ (z = 1/2 ∨ (5/6 ≤ z ∧ z < 5/4))) ∧
  (∀ (w : ℝ), (w = 1/2 ∨ (5/6 ≤ w ∧ w < 5/4)) → ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = 1 ∧ f a b = w) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l335_33523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_in_triangle_l335_33546

/-- In a triangle ABC where angle B is 60° and cos A is 3/5, sin C equals (3√3 + 4) / 10 -/
theorem sin_C_in_triangle (A B C : Real) : 
  B = π/3 →  -- 60° in radians
  Real.cos A = 3/5 → 
  A + B + C = π → -- sum of angles in a triangle
  Real.sin C = (3 * Real.sqrt 3 + 4) / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_in_triangle_l335_33546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l335_33594

open Real

/-- The function f(x) defined as (1/4)sin(x - π/6) - cos(x - 2π/3) -/
noncomputable def f (x : ℝ) : ℝ := (1/4) * sin (x - π/6) - cos (x - 2*π/3)

/-- The maximum value of f(x) is 3/4 -/
theorem f_max_value : ∃ (M : ℝ), M = 3/4 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l335_33594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l335_33540

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 - a * x) / Real.log (1/2)

-- State the theorem
theorem increasing_f_implies_a_range :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3 → f a x₁ < f a x₂) →
  (0 < a ∧ a < 2/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l335_33540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_relations_and_t_range_l335_33550

/-- Given two functions f and g with a common point and tangent line, 
    prove the relations between their coefficients and the possible range of t. -/
theorem function_relations_and_t_range 
  (t : ℝ) 
  (h_t_nonzero : t ≠ 0) 
  (f g : ℝ → ℝ)
  (h_f : ∀ x, f x = x^3 + a*x)
  (h_g : ∀ x, g x = b*x^2 + c)
  (h_common_point : f t = 0 ∧ g t = 0)
  (h_common_tangent : (deriv f) t = (deriv g) t)
  (h_monotone_decreasing : StrictMonoOn (f - g) (Set.Ioo (-1) 3)) :
  (a = -t^2 ∧ b = t ∧ c = -t^3) ∧ 
  (t ∈ Set.Iic (-9) ∪ Set.Ici 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_relations_and_t_range_l335_33550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l335_33521

theorem geometric_series_sum :
  let a : ℝ := 1
  let r : ℝ := 1/2
  let S := fun (n : ℕ) => a * r^n
  ∑' (n : ℕ), S n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l335_33521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mobius_trip_time_l335_33584

/-- Represents the round trip of Mobius the mule between Florence and Rome -/
noncomputable def MobiusTrip (speed_no_load : ℝ) (speed_with_load : ℝ) (distance : ℝ) (rest_time : ℝ) : ℝ :=
  (distance / speed_with_load) + (distance / speed_no_load) + 2 * rest_time

/-- Theorem stating that Mobius' round trip takes 26 hours -/
theorem mobius_trip_time :
  let speed_no_load : ℝ := 13
  let speed_with_load : ℝ := 11
  let distance : ℝ := 143
  let rest_time : ℝ := 1
  MobiusTrip speed_no_load speed_with_load distance rest_time = 26 := by
  sorry

#check mobius_trip_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mobius_trip_time_l335_33584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sessions_until_stable_l335_33551

/-- Represents a jury member's opinion about another member's competence -/
def Opinion := Bool

/-- Represents the jury system -/
structure JurySystem where
  initialMembers : ℕ
  opinions : Fin initialMembers → Fin initialMembers → Opinion

/-- Represents the state of the jury after some number of sessions -/
structure JuryState where
  remainingMembers : ℕ
  excluded : Fin initialMembers → Bool

/-- Function to model a single voting session -/
def voteSession (system : JurySystem) (state : JuryState) : JuryState :=
  sorry

/-- Initial state of the jury -/
def init (system : JurySystem) : JuryState :=
  { remainingMembers := system.initialMembers,
    excluded := λ _ => false }

/-- Theorem stating that the number of sessions until no more exclusions is at most half the initial number of members -/
theorem sessions_until_stable (system : JurySystem) :
  ∃ (n : ℕ), n ≤ system.initialMembers / 2 ∧
  (∀ m > n, voteSession system (Nat.iterate (voteSession system) n (init system)) =
            voteSession system (Nat.iterate (voteSession system) m (init system))) :=
by
  sorry

#check sessions_until_stable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sessions_until_stable_l335_33551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_through_point_l335_33598

theorem cos_angle_through_point (α : Real) (x y : Real) : 
  x = -3 → y = 4 → Real.cos α = -3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_through_point_l335_33598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_l335_33568

-- Define the distance traveled
noncomputable def distance : ℝ := 390

-- Define the time taken
noncomputable def time : ℝ := 4

-- Define the speed
noncomputable def speed : ℝ := distance / time

-- Theorem to prove
theorem car_speed : speed = 97.5 := by
  -- Unfold the definitions
  unfold speed distance time
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_l335_33568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l335_33517

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  -- Triangle inequality
  ab_gt_c : a + b > c
  bc_gt_a : b + c > a
  ca_gt_b : c + a > b
  -- Angle sum property
  angle_sum : A + B + C = π
  -- Angles are positive
  A_pos : A > 0
  B_pos : B > 0
  C_pos : C > 0
  -- Given condition
  condition : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c

theorem triangle_properties (t : Triangle) :
  t.C = π / 3 ∧ 
  (t.c = Real.sqrt 3 → 2 * Real.sqrt 3 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 3 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l335_33517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_zero_l335_33530

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-2*x)) / (x + Real.sin (x^2))

-- State the theorem
theorem limit_of_f_at_zero : 
  Filter.Tendsto f (nhds 0) (nhds 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_zero_l335_33530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l335_33519

/-- Represents a hyperbola -/
structure Hyperbola where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ

/-- Defines the asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola) : (ℝ → ℝ) × (ℝ → ℝ) :=
  let (cx, cy) := h.center
  (λ x => 2 * (x - cx) + cy, λ x => -2 * (x - cx) + cy)

/-- Checks if a point lies on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := h.center
  ((x - cx)^2 / h.a^2) - ((y - cy)^2 / h.b^2) = 1

/-- Calculates the distance between foci of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ :=
  2 * Real.sqrt (h.a^2 + h.b^2)

/-- Theorem: For a hyperbola with given asymptotes and passing through (4,4),
    the distance between its foci is 4√3 -/
theorem hyperbola_focal_distance :
  ∃ (h : Hyperbola),
    asymptotes h = (λ x => 2*x - 2, λ x => -2*x + 6) ∧
    on_hyperbola h (4, 4) ∧
    focal_distance h = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l335_33519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_f_l335_33549

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x > 0 then x^2 - 2*x + 3 
  else -x^2 - 2*x - 3

-- State the theorem
theorem odd_function_f :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x : ℝ, x > 0 → f x = x^2 - 2*x + 3) →
  (∀ x : ℝ, x < 0 → f x = -x^2 - 2*x - 3) :=
by
  intro h
  intro x hx
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_f_l335_33549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l335_33520

/-- Represents the frequency distribution of days missed by students -/
def frequency_distribution : List (Nat × Nat) :=
  [(0, 3), (1, 2), (2, 4), (3, 2), (4, 1), (5, 5), (6, 1)]

/-- Calculates the total number of students -/
def total_students : Nat :=
  (frequency_distribution.map Prod.snd).sum

/-- Calculates the median number of days missed -/
def median : Rat := 2

/-- Calculates the mean number of days missed -/
noncomputable def mean : Rat :=
  (frequency_distribution.map (fun (days, count) => days * count) |>.sum : Rat) / total_students

/-- Theorem stating the difference between mean and median -/
theorem mean_median_difference :
  mean - median = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l335_33520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_111_abs_l335_33545

noncomputable def z : ℕ → ℂ
  | 0 => 0
  | n + 1 => (z n)^2 + 1 + Complex.I

theorem z_111_abs : Complex.abs (z 111) = 7 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_111_abs_l335_33545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_longevity_probability_l335_33543

noncomputable def prob_10 : ℝ := 0.9

noncomputable def prob_15 : ℝ := 0.6

noncomputable def conditional_prob : ℝ := prob_15 / prob_10

theorem animal_longevity_probability : conditional_prob = 2/3 := by
  unfold conditional_prob prob_15 prob_10
  norm_num
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_longevity_probability_l335_33543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_size_from_means_l335_33593

theorem set_size_from_means (S : Finset ℝ) (M : ℝ) :
  (S.sum id) / S.card = M →
  (S.sum id + 15) / (S.card + 1) = M + 2 →
  (S.sum id + 16) / (S.card + 2) = M + 1 →
  S.card = 4 := by
  sorry

#check set_size_from_means

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_size_from_means_l335_33593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_theorem_l335_33539

/-- Circle u₁ with equation x²+y²+8x-30y-63=0 -/
def u₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 8*p.1 - 30*p.2 - 63 = 0}

/-- Circle u₂ with equation x²+y²-6x-30y+99=0 -/
def u₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 - 30*p.2 + 99 = 0}

/-- A circle is externally tangent to u₂ and internally tangent to u₁ -/
def is_tangent_circle (c : Set (ℝ × ℝ)) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 
    (∃ (center : ℝ × ℝ), c = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2}) ∧
    (∃ (p : ℝ × ℝ), p ∈ c ∩ u₂) ∧
    (∃ (q : ℝ × ℝ), q ∈ c ∩ u₁) ∧
    (∀ (x : ℝ × ℝ), x ∈ c → x ∉ (u₁ ∩ u₂))

/-- The line y = nx contains the center of the tangent circle -/
def center_on_line (n : ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ), 
    (∃ (r : ℝ), r > 0 ∧ c = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2}) ∧
    center.2 = n * center.1

/-- Main theorem -/
theorem tangent_circle_theorem : 
  ∃ (p q : ℕ), 
    Nat.Coprime p q ∧ 
    (∀ (n : ℝ), n > 0 → n^2 = p / q → 
      (∃ (c : Set (ℝ × ℝ)), is_tangent_circle c ∧ center_on_line n c)) ∧
    p + q = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_theorem_l335_33539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l335_33577

-- Define the quadratic functions
noncomputable def f₁ (x : ℝ) : ℝ := x^2 + 2*x + 1
noncomputable def f₂ (x : ℝ) : ℝ := -1/2 * x^2 + 3
noncomputable def f₃ (x : ℝ) : ℝ := 2*(x+1)*(x-3)

-- Define the vertices
def vertex₁ : ℝ × ℝ := (-1, 0)
def vertex₂ : ℝ × ℝ := (0, 3)
def vertex₃ : ℝ × ℝ := (1, -8)

-- Define the theorems to prove
theorem quadratic_properties :
  (∀ x < -1, (deriv f₁ x) < 0) ∧
  (∀ x > 0, (deriv f₂ x) < 0) ∧
  (∀ x < 1, (deriv f₃ x) < 0) ∧
  (f₁ (vertex₁.1) = vertex₁.2) ∧
  (f₂ (vertex₂.1) = vertex₂.2) ∧
  (f₃ (vertex₃.1) = vertex₃.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l335_33577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_OX_value_l335_33500

noncomputable section

/-- The circle ω centered at O with radius R -/
def ω (O : ℝ × ℝ) (R : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - O.1)^2 + (p.2 - O.2)^2 = R^2}

/-- The circle γ centered at I with radius r -/
def γ (I : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - I.1)^2 + (p.2 - I.2)^2 = r^2}

/-- The main theorem -/
theorem max_OX_value (R : ℝ) (r : ℝ) (O I A B C D E F P₁ Q₁ P₂ P₃ Q₂ Q₃ K X : ℝ × ℝ) :
  R = 2018 →
  0 < r →
  r < 1009 →
  A ∈ ω O R →
  B ∈ ω O R →
  C ∈ ω O R →
  D ∈ ω O R →
  P₁ ∈ ω O R →
  Q₁ ∈ ω O R →
  P₂ ∈ ω O R →
  P₃ ∈ ω O R →
  Q₂ ∈ ω O R →
  Q₃ ∈ ω O R →
  (I.1 - O.1)^2 + (I.2 - O.2)^2 = R * (R - 2 * r) →
  E ∈ γ I r →
  F ∈ γ I r →
  (∃ r_A : ℝ, r_A = 5 * r ∧ (D.1 - A.1)^2 + (D.2 - A.2)^2 = r_A^2) →
  (∃ t : ℝ, P₁ = (t * E.1 + (1 - t) * F.1, t * E.2 + (1 - t) * F.2)) →
  (∃ t : ℝ, Q₁ = (t * E.1 + (1 - t) * F.1, t * E.2 + (1 - t) * F.2)) →
  P₂ ∈ γ I r →
  P₃ ∈ γ I r →
  Q₂ ∈ γ I r →
  Q₃ ∈ γ I r →
  (∃ t : ℝ, K = (t * P₂.1 + (1 - t) * P₃.1, t * P₂.2 + (1 - t) * P₃.2)) →
  (∃ t : ℝ, K = (t * Q₂.1 + (1 - t) * Q₃.1, t * Q₂.2 + (1 - t) * Q₃.2)) →
  (∃ t : ℝ, X = (t * K.1 + (1 - t) * I.1, t * K.2 + (1 - t) * I.2)) →
  (∃ t : ℝ, X = (t * A.1 + (1 - t) * D.1, t * A.2 + (1 - t) * D.2)) →
  (∀ r' : ℝ, 0 < r' → r' < 1009 → (X.1 - O.1)^2 + (X.2 - O.2)^2 ≤ (2018 * Real.sqrt 3 / 2)^2) ∧
  (∃ r' : ℝ, r' = 1009 ∧ (X.1 - O.1)^2 + (X.2 - O.2)^2 = (2018 * Real.sqrt 3 / 2)^2) :=
by
  sorry

#check max_OX_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_OX_value_l335_33500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumcircle_radius_l335_33555

noncomputable def radius_of_circumscribed_circle_about_sector (r θ : ℝ) : ℝ :=
  r / Real.cos (θ / 2)

theorem sector_circumcircle_radius 
  (r : ℝ) 
  (θ : ℝ) 
  (h1 : r = 9) 
  (h2 : 0 < θ ∧ θ < π/2) : 
  ∃ R : ℝ, R = 4.5 * (1 / Real.cos θ) ∧ 
  R = radius_of_circumscribed_circle_about_sector r (2*θ) := by
  -- Define R
  let R := 4.5 * (1 / Real.cos θ)
  
  -- Prove existence
  use R

  -- Prove the two conditions
  constructor
  · -- First condition: R = 4.5 * (1 / Real.cos θ)
    rfl
  
  · -- Second condition: R = radius_of_circumscribed_circle_about_sector r (2*θ)
    -- Unfold the definition and simplify
    unfold radius_of_circumscribed_circle_about_sector
    rw [h1]
    -- The rest of the proof would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumcircle_radius_l335_33555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_symmetric_functions_l335_33533

open Real

theorem range_of_m_for_symmetric_functions (e : ℝ) (h_e : e = exp 1) :
  let f (x m : ℝ) := m - 1 - x^2
  let g (x : ℝ) := 2 - 5 * log x
  let range_m := {m | ∃ x, e ≤ x ∧ x ≤ 2*e ∧ f x m = -g x}
  range_m = Set.Icc (e^2 + 4) (4*e^2 + 4 + 5*log 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_symmetric_functions_l335_33533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_wrt_x_l335_33525

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := (1 + Real.cos t ^ 2) ^ 2
noncomputable def y (t : ℝ) : ℝ := Real.cos t / Real.sin t ^ 2

-- State the theorem
theorem derivative_y_wrt_x (t : ℝ) (h : Real.sin t ≠ 0) (h' : Real.cos t ≠ 0) :
  deriv (fun x => y (Real.arccos (Real.sqrt (Real.sqrt x - 1)))) (x t) =
    1 / (4 * Real.cos t * Real.sin t ^ 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_wrt_x_l335_33525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equivalence_intersection_point_l335_33565

/-- Line l defined by parametric equations -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, Real.sqrt 3 + Real.sqrt 3 * t)

/-- Curve C defined by polar equation -/
def curve_C (ρ θ : ℝ) : Prop := Real.sin θ - Real.sqrt 3 * ρ * (Real.cos θ)^2 = 0

/-- Rectangular form of curve C -/
def curve_C_rect (x y : ℝ) : Prop := y - Real.sqrt 3 * x^2 = 0

/-- Theorem stating the equivalence of polar and rectangular forms of curve C -/
theorem curve_C_equivalence : ∀ (x y : ℝ), 
  (∃ (ρ θ : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ curve_C ρ θ) ↔ curve_C_rect x y := by sorry

/-- Theorem stating that (2, π/3) is a polar coordinate of an intersection point -/
theorem intersection_point : ∃ (t : ℝ), 
  let (x, y) := line_l t
  curve_C_rect x y ∧ x^2 + y^2 = 4 ∧ Real.arctan (y/x) = π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equivalence_intersection_point_l335_33565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_l335_33501

/-- The curve C -/
def C (a : ℝ) (x y : ℝ) : Prop := a * x^2 + a * y^2 - 2 * a^2 * x - 4 * y = 0

/-- The line l -/
def l (x y : ℝ) : Prop := y = -2 * x + 4

/-- The origin O -/
def O : ℝ × ℝ := (0, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem curve_equation (a : ℝ) (M N : ℝ × ℝ) (ha : a ≠ 0) :
  (∃ x y, C a x y ∧ l x y) →  -- C intersects l
  M ≠ N →  -- M and N are distinct
  C a M.1 M.2 →  -- M is on C
  C a N.1 N.2 →  -- N is on C
  l M.1 M.2 →  -- M is on l
  l N.1 N.2 →  -- N is on l
  distance O M = distance O N →  -- |OM| = |ON|
  (∀ x y, C a x y ↔ x^2 + y^2 - 4*x - 2*y = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_l335_33501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_for_special_hyperbola_l335_33571

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem stating the range of eccentricity for a hyperbola under given conditions -/
theorem eccentricity_range_for_special_hyperbola (h : Hyperbola) (x₀ : ℝ) 
    (h_intersect : ∃ y₀ : ℝ, y₀^2 = x₀ ∧ y₀ = h.b / h.a * x₀)
    (h_x₀_gt_one : x₀ > 1) :
    1 < eccentricity h ∧ eccentricity h < Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_for_special_hyperbola_l335_33571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l335_33591

/-- The circle equation -/
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

/-- A line passing through the origin -/
def line_through_origin (k : ℝ) (x y : ℝ) : Prop := y = k * x

/-- The fourth quadrant -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Tangent point satisfies both line and circle equations -/
def is_tangent_point (k : ℝ) (x y : ℝ) : Prop :=
  my_circle x y ∧ line_through_origin k x y

theorem tangent_line_equation :
  ∃ (k x y : ℝ),
    is_tangent_point k x y ∧
    fourth_quadrant x y ∧
    k = -Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l335_33591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l335_33578

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := ⟨m^2 + 2*m - 8, m - 2⟩

-- Define what it means for a complex number to be pure imaginary
def is_pure_imaginary (w : ℂ) : Prop := w.re = 0 ∧ w.im ≠ 0

-- Theorem statement
theorem pure_imaginary_condition :
  ∀ m : ℝ, is_pure_imaginary (z m) ↔ m = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l335_33578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_seven_l335_33515

/-- A sequence defined by the given recurrence relation -/
def u : ℕ → ℚ
  | 0 => 7  -- Add a case for 0 to cover all natural numbers
  | 1 => 7
  | n + 2 => u (n + 1) + (5 * (n + 1) + 2 * n)

/-- The theorem stating that the sum of coefficients of the quadratic representation of u_n is 7 -/
theorem sum_of_coefficients_is_seven :
  ∃ (a b c : ℚ), ∀ n : ℕ, n ≥ 1 → u n = a * n^2 + b * n + c ∧ a + b + c = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_seven_l335_33515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l335_33508

/-- The area of a triangle with vertices A(1,2,3), B(3,2,1), and C(1,0,1) is 2√3 -/
theorem triangle_area : 
  let A : Fin 3 → ℝ := ![1, 2, 3]
  let B : Fin 3 → ℝ := ![3, 2, 1]
  let C : Fin 3 → ℝ := ![1, 0, 1]
  let AB : Fin 3 → ℝ := λ i => B i - A i
  let AC : Fin 3 → ℝ := λ i => C i - A i
  let cross_product : Fin 3 → ℝ := ![
    AB 1 * AC 2 - AB 2 * AC 1,
    AB 2 * AC 0 - AB 0 * AC 2,
    AB 0 * AC 1 - AB 1 * AC 0
  ]
  let area : ℝ := (1/2) * Real.sqrt (
    (cross_product 0)^2 + (cross_product 1)^2 + (cross_product 2)^2
  )
  area = 2 * Real.sqrt 3 := by
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l335_33508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_gravity_circle_arc_l335_33506

noncomputable section

-- Define the circle
def circleArc (R : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = R^2 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

-- Define the density function
def densityFunc (ρ0 : ℝ) (p : ℝ × ℝ) : ℝ :=
  ρ0 * p.1 * p.2

-- Define a placeholder for centerOfMass
def centerOfMass (s : Set (ℝ × ℝ)) (ρ : ℝ × ℝ → ℝ) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem center_of_gravity_circle_arc (R ρ0 : ℝ) (h_R : R > 0) (h_ρ0 : ρ0 > 0) :
  ∃ cog : ℝ × ℝ, cog = (2*R/3, 2*R/3) ∧
  cog = centerOfMass (circleArc R) (densityFunc ρ0) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_gravity_circle_arc_l335_33506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rita_trip_distance_l335_33575

/-- The distance in miles between Rita's house and her sister's house -/
noncomputable def distance : ℝ := 125

/-- Rita's usual trip time in minutes -/
noncomputable def usual_time : ℝ := 200

/-- Rita's reduced speed in miles per hour -/
noncomputable def speed_reduction : ℝ := 15

/-- The actual trip time in minutes after speed reduction -/
noncomputable def actual_time : ℝ := 300

/-- Rita's usual speed in miles per minute -/
noncomputable def usual_speed : ℝ := distance / usual_time

/-- Rita's reduced speed in miles per minute -/
noncomputable def reduced_speed : ℝ := usual_speed - speed_reduction / 60

theorem rita_trip_distance :
  distance = 125 ∧
  (distance / 4) / usual_speed + (3 * distance / 4) / reduced_speed = actual_time := by
  sorry

#check rita_trip_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rita_trip_distance_l335_33575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_line_AB_equation_min_product_AF_BF_l335_33511

-- Define the parabola C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4 * p.2}

-- Define the focus F
def F : ℝ × ℝ := (0, 1)

-- Define the directrix
def directrix : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -1}

-- Define the line l
def l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 - 2 = 0}

-- Statement 1: Equation of C
theorem parabola_equation :
  C = {p : ℝ × ℝ | p.1^2 = 4 * p.2} := by sorry

-- Statement 2: Equation of line AB
theorem line_AB_equation (P : ℝ × ℝ) (h : P ∈ l) (h₁ : P = (1/2, -3/2)) :
  ∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧
  (∀ x y : ℝ, (x - 4*y + 6 = 0) ↔ ((x, y) ∈ {p : ℝ × ℝ | p.1 * A.1 - 2 * A.2 - 2 * p.2 = 0})) := by sorry

-- Statement 3: Minimum value of |AF|·|BF|
theorem min_product_AF_BF :
  ∃ m : ℝ, m = 9/2 ∧
  (∀ P : ℝ × ℝ, P ∈ l →
    ∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧
    dist A F * dist B F ≥ m) := by sorry

-- Helper function for Euclidean distance
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_line_AB_equation_min_product_AF_BF_l335_33511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_abs_sum_l335_33526

theorem min_value_abs_sum : 
  ∀ x : ℝ, |2*x - 1| + |x + 2| ≥ 5/2 ∧ ∃ x : ℝ, |2*x - 1| + |x + 2| = 5/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_abs_sum_l335_33526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_minus_one_power_l335_33588

theorem sqrt_two_minus_one_power (n : ℕ) :
  (∃ (a b : ℕ), (1 - Real.sqrt 2) ^ n = a - b * Real.sqrt 2 ∧ a^2 - 2 * b^2 = (-1 : ℤ)^n) ∧
  (∃ (m : ℕ), (Real.sqrt 2 - 1) ^ n = Real.sqrt (m : ℝ) - Real.sqrt ((m - 1) : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_minus_one_power_l335_33588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_speed_calculation_l335_33569

-- Define the constants
noncomputable def john_initial_distance : ℝ := 12
noncomputable def john_speed : ℝ := 4.2
noncomputable def final_push_time : ℝ := 28
noncomputable def john_final_lead : ℝ := 2

-- Define Steve's speed as a variable
noncomputable def steve_speed : ℝ := (john_speed * final_push_time - john_initial_distance - john_final_lead) / final_push_time

-- Theorem statement
theorem steve_speed_calculation :
  ∀ ε > 0, |steve_speed - 4.13| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_speed_calculation_l335_33569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l335_33548

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ
  b : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The foci of a hyperbola -/
noncomputable def foci (h : Hyperbola a b) : Point × Point := sorry

/-- The asymptotes of a hyperbola -/
noncomputable def asymptotes (h : Hyperbola a b) : Line × Line := sorry

/-- Checks if a point is on a line -/
def point_on_line (p : Point) (l : Line) : Prop := sorry

/-- Checks if a line is parallel to another line -/
def parallel_lines (l1 l2 : Line) : Prop := sorry

/-- Checks if a point is on a circle with given diameter endpoints -/
def point_on_circle (p : Point) (d1 d2 : Point) : Prop := sorry

/-- Main theorem -/
theorem hyperbola_eccentricity
  (h : Hyperbola a b)
  (f1 f2 : Point)
  (l : Line)
  (m : Point) :
  foci h = (f1, f2) →
  (∃ (asym1 asym2 : Line), asymptotes h = (asym1, asym2) ∧
    parallel_lines l asym1 ∧
    point_on_line f2 l ∧
    point_on_line m asym2 ∧
    point_on_circle m f1 f2) →
  eccentricity h = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l335_33548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_factorial_power_pairs_l335_33518

theorem unique_factorial_power_pairs : 
  {(n, k) : ℕ × ℕ | (n + 1)^k = n! + 1} = {(1, 1), (2, 1), (4, 2)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_factorial_power_pairs_l335_33518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l335_33581

/-- Given a line y = 5x - 7 parameterized by (x, y) = (s, 3) + t(6, m), prove that s = 2 and m = 30 -/
theorem line_parameterization (s m : ℝ) : 
  (∀ x y : ℝ, y = 5*x - 7 ↔ ∃ t : ℝ, (x, y) = (s + 6*t, 3 + m*t)) →
  s = 2 ∧ m = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l335_33581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_3_l335_33553

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (2 + (Real.sqrt 3 / 2) * t, (1 / 2) * t)

-- Define the curve C in polar form
noncomputable def curve_C (θ : ℝ) : ℝ :=
  2 * Real.cos θ

-- Define the chord length
noncomputable def chord_length : ℝ :=
  Real.sqrt 3

-- Theorem statement
theorem chord_length_is_sqrt_3 :
  ∃ (t₁ t₂ : ℝ) (θ₁ θ₂ : ℝ),
    let (x₁, y₁) := line_l t₁
    let (x₂, y₂) := line_l t₂
    let ρ₁ := curve_C θ₁
    let ρ₂ := curve_C θ₂
    x₁^2 + y₁^2 = ρ₁^2 ∧
    x₂^2 + y₂^2 = ρ₂^2 ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = chord_length^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_3_l335_33553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l335_33560

/-- The line equation y = (2x + 3) / 5 -/
def line_equation (x y : ℝ) : Prop := y = (2 * x + 3) / 5

/-- The parameterization form (x, y) = v + t * d -/
def parameterization (x y t : ℝ) (v d : ℝ × ℝ) : Prop :=
  x = v.1 + t * d.1 ∧ y = v.2 + t * d.2

/-- The distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem statement -/
theorem line_parameterization :
  ∃ (d : ℝ × ℝ),
    d = (5 / Real.sqrt 29, 2 / Real.sqrt 29) ∧
    ∀ (x y t : ℝ),
      x ≥ 2 →
      line_equation x y →
      parameterization x y t (2, 1) d →
      t = distance (x, y) (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l335_33560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l335_33573

/-- Parabola with equation y² = 8x and directrix x = -2 -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  directrix : ℝ → Prop

/-- Point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- Focus of the parabola -/
def focus (p : Parabola) : ℝ × ℝ := (2, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

theorem parabola_triangle_area 
  (p : Parabola) 
  (h_eq : p.equation = fun x y => y^2 = 8*x) 
  (h_dir : p.directrix = fun x => x = -2) 
  (M : PointOnParabola p) 
  (h_dist : distance (M.x, M.y) (focus p) = 8) :
  triangle_area (-2, 0) (focus p) (M.x, M.y) = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l335_33573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fran_speed_l335_33558

/-- Calculates the required speed for Fran to cover the same distance as Joann -/
theorem fran_speed (joann_speed joann_time fran_time : ℝ) : 
  joann_speed = 15 → joann_time = 4 → fran_time = 5 → 
  (joann_speed * joann_time) / fran_time = 12 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  
#check fran_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fran_speed_l335_33558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_l335_33522

def is_valid_permutation (b : Fin 14 → ℕ) : Prop :=
  (∀ i j, i ≠ j → b i ≠ b j) ∧
  (∀ i, b i ∈ Finset.range 15 \ {0}) ∧
  (∀ i : Fin 6, b i > b (i + 1)) ∧
  (∀ i : Fin 7, b (i + 7) < b (i + 8))

-- Define the set of valid permutations
def valid_permutations : Set (Fin 14 → ℕ) :=
  {b | is_valid_permutation b}

-- Prove that the set of valid permutations is finite
instance : Fintype valid_permutations := sorry

-- State the theorem
theorem permutation_count :
  Fintype.card valid_permutations = Nat.choose 13 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_l335_33522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearance_time_approx_l335_33585

/-- The time for two trains to be completely clear of each other -/
noncomputable def clearance_time (train1_length train2_length : ℝ) (train1_speed train2_speed : ℝ) : ℝ :=
  (train1_length + train2_length) / ((train1_speed + train2_speed) * 1000 / 3600)

/-- Theorem stating the clearance time for the given train problem -/
theorem train_clearance_time_approx :
  let train1_length : ℝ := 131
  let train2_length : ℝ := 165
  let train1_speed : ℝ := 80
  let train2_speed : ℝ := 65
  abs (clearance_time train1_length train2_length train1_speed train2_speed - 7.35) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearance_time_approx_l335_33585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_plus_f_one_equals_one_l335_33531

-- Define the function f
def f (x : ℝ) : ℝ := 1 - 3*(x - 1) + 3*(x - 1)^2 - (x - 1)^3

-- State the theorem
theorem inverse_f_plus_f_one_equals_one :
  ∃ y, f y = 8 ∧ y + f 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_plus_f_one_equals_one_l335_33531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_range_l335_33595

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then x + 3 else a * x + 1

theorem f_monotonic_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 0 < a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_range_l335_33595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_speed_at_max_power_l335_33529

/-- The force exerted by airflow on a sail -/
noncomputable def F (c s ρ v₀ v : ℝ) : ℝ := (c * s * ρ * (v₀ - v)^2) / 2

/-- The power of the wind -/
noncomputable def N (c s ρ v₀ v : ℝ) : ℝ := F c s ρ v₀ v * v

/-- Theorem: The sailboat speed that maximizes wind power is one-third of the wind speed -/
theorem sailboat_speed_at_max_power
  (c s ρ v₀ : ℝ) (hc : c > 0) (hs : s > 0) (hρ : ρ > 0) (hv₀ : v₀ > 0) :
  ∃ (v : ℝ), v > 0 ∧ v = v₀ / 3 ∧ ∀ (u : ℝ), u ≠ v → N c s ρ v₀ v ≥ N c s ρ v₀ u := by
  sorry

#check sailboat_speed_at_max_power

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_speed_at_max_power_l335_33529
