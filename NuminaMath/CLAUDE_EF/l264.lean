import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_x_less_y_specific_l264_26419

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability of selecting a point (x, y) from a rectangle such that x < y --/
noncomputable def prob_x_less_y (r : Rectangle) : ℝ :=
  let area_triangle := (min r.x_max (r.y_max - r.y_min))^2 / 2
  let area_rectangle := (r.x_max - r.x_min) * (r.y_max - r.y_min)
  area_triangle / area_rectangle

/-- The theorem stating the probability for the specific rectangle in the problem --/
theorem prob_x_less_y_specific : 
  prob_x_less_y ⟨0, 4, 0, 3, by norm_num, by norm_num⟩ = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_x_less_y_specific_l264_26419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_N_starts_with_9_l264_26463

-- Define a function to calculate the sum of digits
def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

-- Define the property of N
def is_valid_N (N : Nat) : Prop :=
  sum_of_digits N = 2020 ∧ ∀ m : Nat, m < N → sum_of_digits m ≠ 2020

-- Theorem statement
theorem smallest_N_starts_with_9 :
  ∃ N : Nat, is_valid_N N ∧ (N.repr.get 0 = '9') :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_N_starts_with_9_l264_26463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sqrt_five_is_seventh_term_l264_26468

noncomputable def my_sequence (n : ℕ+) : ℝ := Real.sqrt (3 * (n : ℝ) - 1)

theorem two_sqrt_five_is_seventh_term :
  my_sequence 7 = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sqrt_five_is_seventh_term_l264_26468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l264_26465

-- Define the function pairs
noncomputable def f_A (x : ℝ) : ℝ := Real.sqrt ((x - 1)^2)
def g_A (x : ℝ) : ℝ := x - 1

def f_B (x : ℝ) : ℝ := x - 1
def g_B (t : ℝ) : ℝ := t - 1

noncomputable def f_C (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)
noncomputable def g_C (x : ℝ) : ℝ := Real.sqrt (x + 1) * Real.sqrt (x - 1)

def f_D (x : ℝ) : ℝ := x
noncomputable def g_D (x : ℝ) : ℝ := x^2 / x

-- Theorem statement
theorem function_equality :
  (∀ x, f_A x = g_A x) = False ∧
  (∀ x, f_B x = g_B x) = True ∧
  (∀ x, f_C x = g_C x) = False ∧
  (∀ x, f_D x = g_D x) = False :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l264_26465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_length_in_triangle_l264_26461

/-- Triangle PQR with given side lengths --/
structure Triangle (PQ PR QR : ℝ) where
  positive_sides : 0 < PQ ∧ 0 < PR ∧ 0 < QR
  triangle_inequality : PQ + PR > QR ∧ PR + QR > PQ ∧ QR + PQ > PR

/-- Point on a line segment --/
def PointOnSegment (P Q M : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • P + t • Q

/-- Parallel line segments --/
def Parallel (AB CD : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (A, B) := AB
  let (C, D) := CD
  ∃ k : ℝ, B - A = k • (D - C)

/-- Midpoint of a line segment --/
def Midpoint (A B M : ℝ × ℝ) : Prop :=
  M = (A + B) / 2

/-- Height of a triangle --/
noncomputable def Height (P Q R : ℝ × ℝ) : ℝ :=
  sorry -- Definition of height

/-- Foot of the height in a triangle --/
noncomputable def HeightFoot (P Q R : ℝ × ℝ) : ℝ × ℝ :=
  sorry -- Definition of the foot of the height

/-- Theorem: MN length in triangle PQR --/
theorem mn_length_in_triangle (P Q R M N : ℝ × ℝ) :
  let t := Triangle 24 26 30
  PointOnSegment P Q M →
  PointOnSegment P R N →
  Parallel (M, N) (Q, R) →
  (let h := Height P Q R
   ∃ H : ℝ × ℝ, Midpoint P H (HeightFoot P Q R) ∧ PointOnSegment M N H) →
  ‖M - N‖ = 15 := by
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_length_in_triangle_l264_26461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_5_mod_31_l264_26448

theorem inverse_of_5_mod_31 : ∃ x : ℕ, x < 31 ∧ (5 * x) % 31 = 1 :=
by
  use 25
  constructor
  · norm_num
  · norm_num

#eval (5 * 25) % 31

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_5_mod_31_l264_26448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_10_l264_26467

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (6 * x + 3) / (x - 2)

-- State the theorem
theorem f_at_10 : f 10 = 63 / 8 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform arithmetic calculations
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_10_l264_26467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_cube_edge_length_l264_26426

-- Define the rectangular parallelepiped
def rectangular_parallelepiped (a b c : ℝ) : Prop :=
  0 < a ∧ a < b ∧ b < c

-- Define the volume difference function
noncomputable def volume_difference (a b c x : ℝ) : ℝ :=
  if x ≤ a then a * b * c - x^3
  else if x ≤ b then a * b * c + (x - a) * x^2 - a * x^2
  else if x ≤ c then x^3 + a * b * (c - x) - a * b * x
  else x^3 - a * b * c

-- State the theorem
theorem optimal_cube_edge_length (a b c : ℝ) 
  (h : rectangular_parallelepiped a b c) :
  ∃ (x : ℝ), ∀ (y : ℝ), 
    volume_difference a b c x ≤ volume_difference a b c y ∧
    x = min b (4 * a / 3) := by
  sorry

#check optimal_cube_edge_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_cube_edge_length_l264_26426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_critical_points_inequality_l264_26424

/-- The function f(x) = x^2 + a*ln(x+1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * Real.log (x + 1)

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2 * x + a / (x + 1)

theorem f_critical_points_inequality (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
  f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0 →
  (0 : ℝ) < f a x₂ / x₁ ∧ f a x₂ / x₁ < -1/2 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_critical_points_inequality_l264_26424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_smaller_sphere_l264_26421

/-- The weight of a hollow sphere given its radius and weight constant -/
noncomputable def sphereWeight (radius : ℝ) (weightConst : ℝ) : ℝ :=
  weightConst * (4 * Real.pi * radius ^ 2)

/-- Theorem: Weight of a smaller sphere given two spheres' radii and the larger sphere's weight -/
theorem weight_of_smaller_sphere 
  (r1 r2 w2 : ℝ) 
  (h1 : r1 > 0) 
  (h2 : r2 > 0) 
  (h3 : w2 > 0) 
  (h4 : r1 = 0.15) 
  (h5 : r2 = 0.3) 
  (h6 : w2 = 32) :
  ∃ (weightConst : ℝ), 
    sphereWeight r2 weightConst = w2 ∧ 
    sphereWeight r1 weightConst = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_smaller_sphere_l264_26421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fixed_point_and_vertex_path_l264_26495

noncomputable def f (x k : ℝ) : ℝ := (k - 1) * x^2 - 2 * k * x + k + 1

noncomputable def vertex (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  (x, a * x^2 + b * x + c)

theorem parabola_fixed_point_and_vertex_path :
  (∀ k : ℝ, f 1 k = 0) ∧
  (∀ k : ℝ, k ≠ 1 → 
    let v := vertex (k - 1) (-2 * k) (k + 1)
    v.fst + v.snd = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fixed_point_and_vertex_path_l264_26495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joan_balloons_l264_26476

/-- Given that Melanie has 41 blue balloons and the total number of blue balloons
    between Joan and Melanie is 81, prove that Joan has 40 blue balloons. -/
theorem joan_balloons (melanie_balloons : ℕ) (total_balloons : ℕ) (joan_balloons : ℕ)
    (h1 : melanie_balloons = 41)
    (h2 : total_balloons = 81)
    (h3 : total_balloons = melanie_balloons + joan_balloons) :
    joan_balloons = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joan_balloons_l264_26476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l264_26451

def sequenceProperty (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n ≠ 0 → n * (a (n + 1) - a n) = a n + 1

theorem sequence_inequality (a : ℝ) :
  (∃ seq : ℕ → ℝ, sequenceProperty seq ∧
    ∀ t n, t ∈ Set.Icc 0 1 → n ≠ 0 →
      seq (n + 1) / (n + 1 : ℝ) < -2 * t^2 - (a + 1) * t + a^2 - a + 3) →
  a ≤ -1 ∨ a ≥ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l264_26451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l264_26462

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 4 + Real.pi / 4) + 2

-- State the theorem
theorem period_of_f :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f (y + q) ≠ f y :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l264_26462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l264_26420

open Real

def f (x : ℝ) := 3 - 4*x - 2*x^2

theorem function_properties :
  (∃ (m : ℝ), IsGreatest {y | ∃ x, f x = y} m ∧ m = 5) ∧
  (¬∃ (m : ℝ), IsLeast {y | ∃ x, f x = y} m) ∧
  (∃ (m M : ℝ), IsGreatest {f x | x ∈ Set.Icc (-3) 2} M ∧ M = 5 ∧
                IsLeast {f x | x ∈ Set.Icc (-3) 2} m ∧ m = -13) ∧
  (¬(∃ (m M : ℝ), IsGreatest {f x | x ∈ Set.Icc 1 2} M ∧ M = -3 ∧
                   IsLeast {f x | x ∈ Set.Icc 1 2} m ∧ m = -13)) ∧
  (∃ (M : ℝ), IsGreatest {f x | x ∈ Set.Ici 0} M ∧ M = 3) ∧
  (¬∃ (m : ℝ), IsLeast {f x | x ∈ Set.Ici 0} m) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l264_26420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_investment_problem_l264_26471

def investment_problem (A_investment : ℝ) (total_profit : ℝ) (A_profit : ℝ) (total_months : ℕ) (B_start_month : ℕ) : Prop :=
  ∃ (B_investment : ℝ),
    A_investment * (total_months : ℝ) / (A_investment * (total_months : ℝ) + B_investment * ((total_months - B_start_month) : ℝ)) = A_profit / total_profit ∧
    B_investment = 200

theorem solve_investment_problem :
  investment_problem 400 100 80 12 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_investment_problem_l264_26471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_form_basis_iff_not_collinear_range_of_m_for_basis_l264_26409

/-- Two vectors form a basis for ℝ² if and only if they are not collinear -/
theorem vectors_form_basis_iff_not_collinear (a b : ℝ × ℝ) :
  (∀ c : ℝ × ℝ, ∃! p : ℝ × ℝ, c = p.1 • a + p.2 • b) ↔ ¬ (∃ k : ℝ, b = k • a) :=
sorry

/-- The range of m for which (1, 3) and (m, 2m-3) form a basis for ℝ² -/
theorem range_of_m_for_basis : ∀ m : ℝ,
  (∀ c : ℝ × ℝ, ∃! p : ℝ × ℝ, c = p.1 • (1, 3) + p.2 • (m, 2*m-3)) ↔ m ≠ -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_form_basis_iff_not_collinear_range_of_m_for_basis_l264_26409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_4_l264_26470

noncomputable section

def f : ℝ → ℝ := sorry

axiom f_equation (x : ℝ) : f x = 2 * f (8 - x) - x^2 + 11*x - 18

def tangent_line (x : ℝ) := x - 14

theorem tangent_line_at_4 :
  ∀ x : ℝ, (tangent_line x - f 4) = (deriv f 4) * (x - 4) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_4_l264_26470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_painted_faces_l264_26457

/-- Represents a 3D coordinate within the block --/
structure Coordinate where
  x : Fin 6
  y : Fin 6
  z : Fin 2
deriving Fintype

/-- Counts the number of painted faces for a cube at a given coordinate --/
def countPaintedFaces (c : Coordinate) : Nat :=
  (if c.x = 0 ∨ c.x = 5 then 1 else 0) +
  (if c.y = 0 ∨ c.y = 5 then 1 else 0) +
  (if c.z = 0 ∨ c.z = 1 then 1 else 0)

/-- Checks if a number is even --/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/-- The main theorem --/
theorem count_even_painted_faces :
  (Finset.univ.filter (fun c => isEven (countPaintedFaces c))).card = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_painted_faces_l264_26457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_APF_l264_26404

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/8 = 1

-- Define the right focus F
def F : ℝ × ℝ := (3, 0)

-- Define point A
noncomputable def A : ℝ × ℝ := (0, 6 * Real.sqrt 6)

-- Define a point P on the left branch of the hyperbola
variable (P : ℝ × ℝ)

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Function to calculate perimeter of triangle APF
noncomputable def perimeter (P : ℝ × ℝ) : ℝ := 
  distance A P + distance P F + distance A F

-- Theorem statement
theorem min_perimeter_APF : 
  ∃ (min_perim : ℝ), 
    (∀ Q : ℝ × ℝ, hyperbola Q.1 Q.2 → Q.1 < 0 → perimeter Q ≤ min_perim) ∧ 
    min_perim = 32 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_APF_l264_26404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_range_l264_26488

theorem quadratic_roots_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + (m-2)*x + (5-m) = 0 → x > 2) ↔ m ∈ Set.Ioo (-5) (-4) ∪ {-4} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_range_l264_26488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_CD_length_approx_10_l264_26473

/-- Represents a quadrilateral with diagonals intersecting at a point -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)

/-- Calculate the distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem about the length of CD in a specific quadrilateral -/
theorem CD_length_approx_10 (ABCD : Quadrilateral) : 
  distance ABCD.B ABCD.O = 5 →
  distance ABCD.O ABCD.D = 7 →
  distance ABCD.A ABCD.O = 9 →
  distance ABCD.O ABCD.C = 4 →
  distance ABCD.A ABCD.B = 7 →
  ∃ ε > 0, |distance ABCD.C ABCD.D - 10| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_CD_length_approx_10_l264_26473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_staircase_handrail_length_l264_26452

/-- The length of a handrail on a spiral staircase -/
noncomputable def handrail_length (radius : ℝ) (rise : ℝ) (angle : ℝ) : ℝ :=
  Real.sqrt (rise^2 + (angle / 360 * 2 * Real.pi * radius)^2)

/-- Theorem: The length of a handrail on a spiral staircase with given parameters is approximately 17.7 feet -/
theorem spiral_staircase_handrail_length :
  let radius : ℝ := 3
  let rise : ℝ := 15
  let angle : ℝ := 180
  abs (handrail_length radius rise angle - 17.7) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_staircase_handrail_length_l264_26452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l264_26498

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.sqrt 3 * (Real.cos x)^2

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := Real.sin (2*x + Real.pi/3) - Real.sqrt 3 / 2

/-- Theorem stating that the minimum positive k for which f(x) = g(x - k) is π/3 -/
theorem min_shift_value :
  ∃ (k : ℝ), k > 0 ∧ (∀ x, f x = g (x - k)) ∧
  (∀ k' > 0, (∀ x, f x = g (x - k')) → k ≤ k') ∧
  k = Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l264_26498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_squared_l264_26438

-- Define the circles
def circle1 (p : ℝ × ℝ) : Prop :=
  (p.1 - 2)^2 + (p.2 + 1)^2 = 16

def circle2 (p : ℝ × ℝ) : Prop :=
  (p.1 - 2)^2 + (p.2 - 5)^2 = 10

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | circle1 p ∧ circle2 p}

-- Theorem statement
theorem intersection_distance_squared :
  ∀ A B : ℝ × ℝ, A ∈ intersection_points → B ∈ intersection_points → A ≠ B →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 15 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_squared_l264_26438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_equality_l264_26459

-- Define the quadratic roots
noncomputable def root1 : ℝ := 4 * Real.sqrt 2
noncomputable def root2 (a : ℝ) : ℝ := 3 * Real.sqrt (3 - 2 * a)

-- Define the condition for roots being of the same type
def same_type (r1 r2 : ℝ) : Prop := sorry

-- Define the condition for root2 being in simplest form
def simplest_form (r : ℝ → ℝ) : Prop := sorry

-- Theorem statement
theorem quadratic_roots_equality (a : ℝ) : 
  same_type root1 (root2 a) ∧ simplest_form root2 → a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_equality_l264_26459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_l264_26435

-- Define the ellipse E: 3x^2 + y^2 = 3
def E (x y : ℝ) : Prop := 3 * x^2 + y^2 = 3

-- Define the hyperbola H: xy = 3/4
def H (x y : ℝ) : Prop := x * y = 3/4

-- Define the region R
def R (x y : ℝ) : Prop := 3 * x^2 + y^2 ≤ 3 ∧ x * y ≥ 3/4

-- Theorem statement
theorem intersection_and_area :
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    E x₁ y₁ ∧ H x₁ y₁ ∧
    E x₂ y₂ ∧ H x₂ y₂ ∧
    E x₃ y₃ ∧ H x₃ y₃ ∧
    E x₄ y₄ ∧ H x₄ y₄ ∧
    (x₁ = 1/2 ∧ y₁ = 3/2 ∨
     x₁ = -1/2 ∧ y₁ = -3/2 ∨
     x₁ = Real.sqrt 3 / 2 ∧ y₁ = Real.sqrt 3 / 2 ∨
     x₁ = -Real.sqrt 3 / 2 ∧ y₁ = -Real.sqrt 3 / 2) ∧
    (x₂ = 1/2 ∧ y₂ = 3/2 ∨
     x₂ = -1/2 ∧ y₂ = -3/2 ∨
     x₂ = Real.sqrt 3 / 2 ∧ y₂ = Real.sqrt 3 / 2 ∨
     x₂ = -Real.sqrt 3 / 2 ∧ y₂ = -Real.sqrt 3 / 2) ∧
    (x₃ = 1/2 ∧ y₃ = 3/2 ∨
     x₃ = -1/2 ∧ y₃ = -3/2 ∨
     x₃ = Real.sqrt 3 / 2 ∧ y₃ = Real.sqrt 3 / 2 ∨
     x₃ = -Real.sqrt 3 / 2 ∧ y₃ = -Real.sqrt 3 / 2) ∧
    (x₄ = 1/2 ∧ y₄ = 3/2 ∨
     x₄ = -1/2 ∧ y₄ = -3/2 ∨
     x₄ = Real.sqrt 3 / 2 ∧ y₄ = Real.sqrt 3 / 2 ∨
     x₄ = -Real.sqrt 3 / 2 ∧ y₄ = -Real.sqrt 3 / 2) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
  (∃ (A : ℝ), A = Real.pi / 4 - 3/2 * Real.log 3 ∧
    A = ∫ (x : ℝ) in (1/2)..(Real.sqrt 3 / 2), 2 * (Real.sqrt (3 - 3 * x^2) - 3 / (4 * x))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_l264_26435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_perpendicular_line_l264_26483

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The x-intercept of a line -/
noncomputable def xIntercept (l : Line) : ℝ := -l.yIntercept / l.slope

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def perpendicular (l1 l2 : Line) : Prop := l1.slope * l2.slope = -1

theorem x_intercept_of_perpendicular_line (l1 l2 : Line) :
  l1.slope = 1/2 →
  l1.yIntercept = -2 →
  perpendicular l1 l2 →
  l2.yIntercept = 5 →
  xIntercept l2 = -5/2 := by
  sorry

#check x_intercept_of_perpendicular_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_perpendicular_line_l264_26483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_midpoint_diagonals_equal_opposite_sides_l264_26406

/-- A quadrilateral with diagonals intersecting at their midpoints has equal opposite sides -/
theorem quadrilateral_midpoint_diagonals_equal_opposite_sides 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
  (A B C D O : V)
  (h_midpoint_AC : O = midpoint ℝ A C)
  (h_midpoint_BD : O = midpoint ℝ B D) :
  (dist A B = dist C D) ∧ (dist A D = dist B C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_midpoint_diagonals_equal_opposite_sides_l264_26406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_4410_undefined_l264_26491

theorem tan_4410_undefined : ¬∃ x : ℝ, Real.tan (4410 * π / 180) = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_4410_undefined_l264_26491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_most_stable_l264_26412

/-- Represents a person in the shooting test -/
inductive Person
| A
| B
| C

/-- Returns the variance of a person's shooting performance -/
noncomputable def variance (p : Person) : ℝ :=
  match p with
  | Person.A => 0.45
  | Person.B => 0.42
  | Person.C => 0.51

/-- Defines stability as the inverse of variance -/
noncomputable def stability (p : Person) : ℝ := 1 / variance p

/-- States that person B has the most stable performance -/
theorem b_most_stable :
  ∀ p : Person, p ≠ Person.B → stability Person.B > stability p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_most_stable_l264_26412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_relation_l264_26487

theorem sine_cosine_relation (α : Real) (h : Real.cos α = 1/3) :
  Real.sin (α + 3*Real.pi/2) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_relation_l264_26487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l264_26410

theorem no_valid_n : ¬∃ (n : ℕ), 
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l264_26410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_Z_l264_26455

theorem imaginary_part_of_Z : ∀ (Z : ℂ),
  Z * Complex.I = ((Complex.I + 1) / (Complex.I - 1)) ^ 2018 →
  Z.im = (5 : ℝ) ^ 1009 * Real.sin (2018 * Real.arctan 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_Z_l264_26455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trail_blazers_third_place_l264_26428

-- Define the teams and people
inductive Team : Type
| Warriors | Nuggets | Jazz | TrailBlazers | Rockets

inductive Person : Type
| A | B | C | D | E

-- Define the ranking as a function from Team to ℕ
def Ranking : Type := Team → ℕ

-- Define a predicate for a valid ranking (1 to 5)
def ValidRanking (r : Ranking) : Prop :=
  (∀ t : Team, r t ≥ 1 ∧ r t ≤ 5) ∧
  (∀ t1 t2 : Team, t1 ≠ t2 → r t1 ≠ r t2)

-- Define the guesses made by each person
def Guesses (r : Ranking) : Prop :=
  (r Team.Warriors = 1 ∨ r Team.Nuggets = 3) ∧  -- A's guesses
  (r Team.Warriors = 3 ∨ r Team.Jazz = 5) ∧     -- B's guesses
  (r Team.Rockets = 4 ∨ r Team.Warriors = 2) ∧  -- C's guesses
  (r Team.Nuggets = 2 ∨ r Team.TrailBlazers = 5) ∧  -- D's guesses
  (r Team.TrailBlazers = 3 ∨ r Team.Rockets = 5)    -- E's guesses

-- Define that each person guessed half correctly
def HalfCorrect (r : Ranking) : Prop :=
  (r Team.Warriors = 1) ≠ (r Team.Nuggets = 3) ∧
  (r Team.Warriors = 3) ≠ (r Team.Jazz = 5) ∧
  (r Team.Rockets = 4) ≠ (r Team.Warriors = 2) ∧
  (r Team.Nuggets = 2) ≠ (r Team.TrailBlazers = 5) ∧
  (r Team.TrailBlazers = 3) ≠ (r Team.Rockets = 5)

-- Define that each place was guessed correctly by someone
def EachPlaceGuessed (r : Ranking) : Prop :=
  (r Team.Warriors = 1 ∨ r Team.Warriors = 2 ∨ r Team.Warriors = 3) ∧
  (r Team.Nuggets = 2 ∨ r Team.Nuggets = 3) ∧
  (r Team.Jazz = 5) ∧
  (r Team.TrailBlazers = 3 ∨ r Team.TrailBlazers = 5) ∧
  (r Team.Rockets = 4 ∨ r Team.Rockets = 5)

-- Theorem stating that Trail Blazers secured 3rd place
theorem trail_blazers_third_place :
  ∀ r : Ranking, ValidRanking r → Guesses r → HalfCorrect r → EachPlaceGuessed r →
  r Team.TrailBlazers = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trail_blazers_third_place_l264_26428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l264_26407

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 - x + 1 / (16 * a) > 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x > 0 → Real.sqrt (3 * x + 1) < 1 + a * x

-- Define the theorem
theorem a_range (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → (3/2 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l264_26407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l264_26489

noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x - 3) * (x - 5) / ((x - 2) * (x - 4) * (x - 6))

def solution_set : Set ℝ := {x | x < 1 ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ 6 < x}

theorem inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l264_26489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_has_period_pi_l264_26423

/-- The function f(x) defined as sin(π/4 + x) * sin(π/4 - x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi/4 + x) * Real.sin (Real.pi/4 - x)

/-- Theorem stating that f(x) is an even function -/
theorem f_is_even : ∀ x : ℝ, f x = f (-x) := by sorry

/-- Theorem stating that f(x) has a period of π -/
theorem f_has_period_pi : ∀ x : ℝ, f (x + Real.pi) = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_has_period_pi_l264_26423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_product_nonzero_l264_26414

theorem negation_of_forall_product_nonzero {R : Type*} [Field R] :
  (¬ ∀ (x y : R), x * y ≠ 0) ↔ (∃ (x y : R), x * y = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_product_nonzero_l264_26414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_angles_polygon_l264_26413

/-- The sum of the interior angles of an n-sided polygon is (n-2) * 180°. -/
theorem sum_interior_angles_polygon (n : ℕ) (h : n ≥ 3) :
  (n - 2) * 180 = (n - 2) * 180 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_angles_polygon_l264_26413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_pyramid_l264_26492

/-- Represents a pyramid with a square base and congruent lateral faces -/
structure Pyramid where
  base_side : ℕ  -- Side length of the square base
  height : ℕ     -- Height of the pyramid
  slant_height : ℕ  -- Slant height (height of each triangular face)

/-- Calculates the lateral edge length of the pyramid -/
noncomputable def lateral_edge (p : Pyramid) : ℝ :=
  Real.sqrt (p.slant_height^2 + (p.base_side / 2)^2)

/-- Calculates the total surface area of the pyramid -/
def total_area (p : Pyramid) : ℚ :=
  p.base_side^2 + 2 * p.base_side * p.slant_height

/-- Calculates the volume of the pyramid -/
def volume (p : Pyramid) : ℚ :=
  (1 / 3) * p.base_side^2 * p.height

theorem no_integer_pyramid :
  ∀ (p : Pyramid),
    ¬(∃ (n : ℕ), lateral_edge p = n ∧ ∃ (m : ℕ), total_area p = m ∧ ∃ (k : ℕ), volume p = k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_pyramid_l264_26492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_BDF_l264_26402

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- The angle between three points in 3D space -/
noncomputable def angle (p q r : Point3D) : ℝ := sorry

/-- The cross product of two vectors in 3D space -/
def cross_product (v w : Point3D) : Point3D := sorry

/-- The magnitude of a vector in 3D space -/
noncomputable def magnitude (v : Point3D) : ℝ := sorry

/-- Whether a plane is perpendicular to a line segment -/
def is_perpendicular (p q r s t : Point3D) : Prop := sorry

theorem area_triangle_BDF 
  (A B C D E F : Point3D)
  (h1 : distance A B = 3 ∧ distance B C = 3 ∧ distance C D = 3 ∧ 
        distance D E = 3 ∧ distance E F = 3 ∧ distance F A = 3)
  (h2 : angle A B C = 2 * π / 3 ∧ angle C D E = 2 * π / 3 ∧ angle E F A = 2 * π / 3)
  (h3 : is_perpendicular A B C D E) :
  ∃ (area : ℝ), area = (1/2) * magnitude (cross_product 
    ⟨B.x - D.x, B.y - D.y, B.z - D.z⟩ 
    ⟨B.x - F.x, B.y - F.y, B.z - F.z⟩) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_BDF_l264_26402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_value_l264_26415

theorem find_m_value (U A : Set ℝ) (m : ℝ) : 
  U = {4, m^2 + 2*m - 3, 19} →
  A = {5} →
  U \ A = {|4*m - 3|, 4} →
  m = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_value_l264_26415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_travels_400_feet_l264_26430

/-- The distance Lindy travels when Jack and Christina meet -/
noncomputable def lindyDistance (initialDistance : ℝ) (jackSpeed christinaSpeed lindySpeed : ℝ) : ℝ :=
  let timeToMeet := initialDistance / (jackSpeed + christinaSpeed)
  lindySpeed * timeToMeet

/-- Theorem stating that Lindy travels 400 feet when Jack and Christina meet -/
theorem lindy_travels_400_feet :
  lindyDistance 240 3 3 10 = 400 := by
  -- Unfold the definition of lindyDistance
  unfold lindyDistance
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_travels_400_feet_l264_26430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_implies_zero_l264_26422

noncomputable def cube_root_2 : ℝ := Real.rpow 2 (1/3)
noncomputable def cube_root_4 : ℝ := Real.rpow 4 (1/3)

noncomputable def expansion (n : ℕ) : ℝ × ℝ × ℝ :=
  let x := (1 + 4 * cube_root_2 - 4 * cube_root_4) ^ n
  (x, x * cube_root_2, x * cube_root_4)

theorem expansion_implies_zero (n : ℕ) (a b c : ℝ) :
  expansion n = (a, b, c) →
  c = 0 →
  n = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_implies_zero_l264_26422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_solution_exists_solution_no_greater_solution_greatest_integer_is_seven_l264_26499

theorem greatest_integer_solution (x : ℤ) : (3 * (x - 2).natAbs + 9 ≤ 24) → x ≤ 7 :=
by sorry

theorem exists_solution : ∃ x : ℤ, 3 * (x - 2).natAbs + 9 ≤ 24 ∧ x = 7 :=
by sorry

theorem no_greater_solution (y : ℤ) : y > 7 → ¬(3 * (y - 2).natAbs + 9 ≤ 24) :=
by sorry

theorem greatest_integer_is_seven :
  (∃ x : ℤ, 3 * (x - 2).natAbs + 9 ≤ 24) →
  (∃ m : ℤ, (3 * (m - 2).natAbs + 9 ≤ 24) ∧ 
            (∀ y : ℤ, 3 * (y - 2).natAbs + 9 ≤ 24 → y ≤ m) ∧
            m = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_solution_exists_solution_no_greater_solution_greatest_integer_is_seven_l264_26499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_b_sequence_l264_26440

noncomputable def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

noncomputable def geometric_sum (a₁ : ℝ) (r : ℝ) : ℝ := a₁ / (1 - r)

noncomputable def b_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := geometric_sequence a₁ r (2 * n)

theorem sum_of_b_sequence :
  ∀ (r : ℝ),
  (geometric_sum 3 r = 9) →
  (geometric_sum (b_sequence 3 r 1) ((geometric_sequence 3 r 2) / (geometric_sequence 3 r 1)) = 18/5) :=
by
  sorry

#check sum_of_b_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_b_sequence_l264_26440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_polygon_equal_distances_l264_26433

/-- A polygon with colored vertices -/
structure ColoredPolygon where
  vertices : Finset (ℝ × ℝ)
  red : Finset (ℝ × ℝ)
  blue : Finset (ℝ × ℝ)
  vertex_count : vertices.card = 100
  red_count : red.card = 10
  blue_count : blue.card = 10
  red_subset : red ⊆ vertices
  blue_subset : blue ⊆ vertices

/-- Distance between two points -/
noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

/-- The main theorem -/
theorem colored_polygon_equal_distances (p : ColoredPolygon) :
  ∃ (a b c d : ℝ × ℝ),
    a ∈ p.blue ∧ b ∈ p.blue ∧ c ∈ p.red ∧ d ∈ p.red ∧
    distance a b = distance c d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_polygon_equal_distances_l264_26433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem1_proof_l264_26460

theorem problem1_proof : 2 * (Real.sqrt 3 - 1) - |Real.sqrt 3 - 2| - 4 = 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem1_proof_l264_26460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l264_26432

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) / (x - 3)

theorem function_properties (a : ℝ) :
  f a 0 = -1 →
  (a = 3 ∧
   ∀ x, x ≠ 3 → f a x = 1 + 6 / (x - 3) ∧
   ∀ x₁ x₂, x₁ > 3 → x₂ > 3 → x₁ > x₂ → f a x₁ < f a x₂) :=
by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l264_26432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_optimal_choice_l264_26439

-- Define the probability function for Carol's win
noncomputable def carol_win_probability (c : ℝ) : ℝ :=
  if c ≤ 1/4 then c
  else if c ≤ 1/2 then 8*c - 8*c^2 - 1
  else 1 - c

-- State the theorem
theorem carol_optimal_choice :
  ∃ (c : ℝ), c = 3/8 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 →
  carol_win_probability x ≤ carol_win_probability c := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_optimal_choice_l264_26439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l264_26417

theorem triangle_side_lengths (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  Real.sin C = Real.sqrt 10 / 4 ∧
  a = 2 ∧
  2 * Real.sin A = Real.sin C →
  c = 4 ∧ (b = Real.sqrt 6 ∨ b = 2 * Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l264_26417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_common_difference_for_special_arithmetic_progression_l264_26490

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def num_divisors (n : ℕ) : ℕ := (Finset.filter (λ d ↦ n % d = 0) (Finset.range (n + 1))).card

def arithmetic_progression (a d n : ℕ) : ℕ := a + n * d

theorem smallest_common_difference_for_special_arithmetic_progression :
  ∃ d : ℕ,
    (∀ n : ℕ, is_divisible_by (num_divisors (arithmetic_progression 16 d n)) 5) ∧
    (∀ d' : ℕ, d' < d →
      ¬(∀ n : ℕ, is_divisible_by (num_divisors (arithmetic_progression 16 d' n)) 5)) ∧
    d = 32 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_common_difference_for_special_arithmetic_progression_l264_26490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_blue_is_seven_fiftieths_l264_26496

/-- The number of tiles in the box -/
def total_tiles : ℕ := 100

/-- A tile is blue if its number is congruent to 3 mod 7 -/
def is_blue (n : ℕ) : Prop := n % 7 = 3

/-- The set of blue tiles -/
def blue_tiles : Finset ℕ := Finset.filter (fun n => n % 7 = 3) (Finset.range total_tiles)

/-- The probability of choosing a blue tile -/
noncomputable def prob_blue : ℚ := (blue_tiles.card : ℚ) / total_tiles

theorem prob_blue_is_seven_fiftieths : prob_blue = 7 / 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_blue_is_seven_fiftieths_l264_26496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_monochromatic_isosceles_triangles_l264_26481

-- Define a circle
def Circle := Set (ℝ × ℝ)

-- Define a color type
inductive Color
| One
| Two

-- Define a coloring function for points on the circle
def Coloring (c : Circle) := (ℝ × ℝ) → Color

-- Define an isosceles triangle
def IsoscelesTriangle (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

-- Define a predicate for a triangle being inscribed in a circle
def InscribedTriangle (c : Circle) (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

-- Main theorem
theorem infinite_monochromatic_isosceles_triangles 
  (c : Circle) (coloring : Coloring c) : 
  ∃ (triangles : Set (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ)), 
    (Set.Infinite triangles) ∧ 
    (∀ t ∈ triangles, 
      let (x1, y1, x2, y2, x3, y3) := t;
      IsoscelesTriangle (x1, y1) (x2, y2) (x3, y3) ∧
      InscribedTriangle c (x1, y1) (x2, y2) (x3, y3) ∧
      coloring (x1, y1) = coloring (x2, y2) ∧
      coloring (x2, y2) = coloring (x3, y3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_monochromatic_isosceles_triangles_l264_26481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_area_of_triangle_l264_26445

noncomputable def point : Type := ℝ × ℝ

def A : point := (0, 6)
def B : point := (8, 0)
def O : point := (0, 0)

noncomputable def distance (p q : point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def triangleArea (p q r : point) : ℝ :=
  (1/2) * |p.1 * (q.2 - r.2) + q.1 * (r.2 - p.2) + r.1 * (p.2 - q.2)|

theorem distance_and_area_of_triangle :
  distance A B = 10 ∧ triangleArea O A B = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_area_of_triangle_l264_26445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_reduction_l264_26479

theorem village_population_reduction (initial_population : ℕ) 
  (death_rate : ℚ) (current_population : ℕ) 
  (h1 : initial_population = 3800)
  (h2 : death_rate = 1/10)
  (h3 : current_population = 2907)
  : (initial_population * (1 - death_rate) - current_population) / 
    (initial_population * (1 - death_rate)) = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_reduction_l264_26479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_formaldehyde_content_scientific_notation_l264_26405

/-- Represents parts per million as a real number between 0 and 1 -/
def parts_per_million (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

/-- Represents a number in scientific notation -/
noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

theorem formaldehyde_content_scientific_notation :
  ∀ x : ℝ, parts_per_million x → x = 75 / 1000000 →
  ∃ a n, scientific_notation a n = x ∧ a = 7.5 ∧ n = -5 := by
  sorry

#check formaldehyde_content_scientific_notation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_formaldehyde_content_scientific_notation_l264_26405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_implies_a_geq_one_roots_condition_implies_inequality_l264_26400

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x - 1

-- Theorem 1: If f(x) is monotonically decreasing on [1, +∞), then a ≥ 1
theorem monotonic_decreasing_implies_a_geq_one (a : ℝ) :
  (∀ x y, 1 ≤ x ∧ x < y → f a y ≤ f a x) → a ≥ 1 := by
  sorry

-- Theorem 2: If f(x) + 2 = 0 has two real roots x₁ and x₂, and x₂ > 2x₁, then x₁x₂² > 32/e³
theorem roots_condition_implies_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  f a x₁ + 2 = 0 →
  f a x₂ + 2 = 0 →
  x₂ > 2 * x₁ →
  x₁ * x₂^2 > 32 / Real.exp 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_implies_a_geq_one_roots_condition_implies_inequality_l264_26400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l264_26485

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := 1
def c : ℝ := 1

noncomputable def e : ℝ := Real.sqrt 2 / 2

def focal_distance : ℝ := 2

def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def is_on_line (x y m : ℝ) : Prop := x - y + m = 0

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem ellipse_and_line_intersection :
  a > b ∧ b > 0 ∧
  e = c / a ∧
  focal_distance = 2 * c ∧
  (∀ x y, is_on_ellipse x y ↔ x^2 / a^2 + y^2 / b^2 = 1) →
  (∃ m : ℝ,
    ∃ x₁ y₁ x₂ y₂ : ℝ,
      is_on_ellipse x₁ y₁ ∧
      is_on_ellipse x₂ y₂ ∧
      is_on_line x₁ y₁ m ∧
      is_on_line x₂ y₂ m ∧
      (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
      distance x₁ y₁ x₂ y₂ = 2 ∧
      (m = Real.sqrt 3 / 2 ∨ m = -Real.sqrt 3 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l264_26485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_proof_l264_26441

/-- Given a triangle ABC with point P on BC and Q the midpoint of AC, prove that BC = (-6, 21) -/
theorem triangle_vector_proof (A B C P Q : ℝ × ℝ) : 
  P.1 = B.1 + 2/3 * (C.1 - B.1) ∧ 
  P.2 = B.2 + 2/3 * (C.2 - B.2) ∧ 
  Q.1 = (A.1 + C.1) / 2 ∧ 
  Q.2 = (A.2 + C.2) / 2 ∧
  (P.1 - A.1, P.2 - A.2) = (4, 3) ∧ 
  (P.1 - Q.1, P.2 - Q.2) = (1, 5) → 
  (C.1 - B.1, C.2 - B.2) = (-6, 21) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_proof_l264_26441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_height_interval_l264_26464

/-- The height of a ball thrown vertically upward -/
def ballHeight (x : ℝ) : ℝ := 10 * x - 4.9 * x^2

/-- The equation representing when the ball is at 5 meters -/
def heightEq (x : ℝ) : Prop := ballHeight x = 5

theorem ball_height_interval : 
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
  heightEq x₁ ∧ heightEq x₂ ∧ 
  abs (x₂ - x₁ - 0.28) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_height_interval_l264_26464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_increase_factor_l264_26431

/-- Represents the labor market for dwarves and elves -/
structure LaborMarket where
  dwarves_supply : ℝ → ℝ
  elves_supply : ℝ → ℝ
  dwarves_demand : ℝ → ℝ
  elves_demand : ℝ → ℝ

/-- The labor market before the king's intervention -/
noncomputable def initial_market : LaborMarket :=
  { dwarves_supply := λ L => 1 + L / 3
    elves_supply := λ L => 3 + L
    dwarves_demand := λ L => 10 - 2 * L / 3
    elves_demand := λ L => 18 - 2 * L }

/-- Calculates the equilibrium wage and labor for a given group -/
noncomputable def equilibrium (supply demand : ℝ → ℝ) : ℝ × ℝ :=
  sorry

/-- Calculates the new equilibrium wage after the king's intervention -/
noncomputable def new_equilibrium (market : LaborMarket) : ℝ :=
  sorry

/-- Theorem: The wage increase factor for the lower-paid group (dwarves) is 1.25 -/
theorem wage_increase_factor (market : LaborMarket) :
  let (w_d, _) := equilibrium market.dwarves_supply market.dwarves_demand
  let w_new := new_equilibrium market
  w_new / w_d = 1.25 := by
  sorry

#eval "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_increase_factor_l264_26431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l264_26446

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4) + Real.cos (2 * x + Real.pi / 4)

theorem f_properties :
  (∀ x y, x ∈ Set.Ioo 0 (Real.pi / 2) → y ∈ Set.Ioo 0 (Real.pi / 2) → x < y → f y < f x) ∧
  (∀ x, f (Real.pi / 2 + x) = f (Real.pi / 2 - x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l264_26446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_equals_2_sqrt_41_l264_26494

noncomputable def start_point : ℝ × ℝ := (-3, 6)
noncomputable def end_point : ℝ × ℝ := (6, -3)
noncomputable def intermediate_point : ℝ × ℝ := (1, 1)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem total_distance_equals_2_sqrt_41 :
  distance start_point intermediate_point + distance intermediate_point end_point = 2 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_equals_2_sqrt_41_l264_26494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_12_plus_a_l264_26425

theorem remainder_of_12_plus_a (a : ℕ) (h : 17 * a ≡ 1 [ZMOD 31]) :
  (12 + a) % 31 = 23 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_12_plus_a_l264_26425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_plants_count_l264_26458

theorem garden_plants_count (total_rows : ℕ) (columns_per_row : ℕ) 
  (corn_columns_in_divided_row : ℕ) (tomato_columns_in_divided_row : ℕ) 
  (h1 : total_rows = 96) 
  (h2 : columns_per_row = 24) 
  (h3 : corn_columns_in_divided_row = 12) 
  (h4 : tomato_columns_in_divided_row = 12) : 
  (2 * (total_rows / 3) * columns_per_row + 
   2 * (total_rows / 3) * corn_columns_in_divided_row) = 2304 := by
  sorry

#eval 2 * (96 / 3) * 24 + 2 * (96 / 3) * 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_plants_count_l264_26458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l264_26449

theorem product_remainder (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l264_26449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l264_26416

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - 3*y^2 - 8*x - 18*y - 8 = 0

/-- The x-coordinate of the focus -/
noncomputable def focus_x : ℝ := 4 + Real.sqrt (76/3)

/-- The y-coordinate of the focus -/
def focus_y : ℝ := -3

/-- Theorem stating that the given point is a focus of the hyperbola -/
theorem is_focus_of_hyperbola : 
  ∃ (c : ℝ), c > 0 ∧ 
  ∀ (x y : ℝ), hyperbola_equation x y ↔ 
    ((x - 4)^2 / (c^2) - (y + 3)^2 / (c^2 / 3) = 1) ∧
    (focus_x - 4)^2 - (focus_y + 3)^2 = c^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l264_26416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_son_work_time_is_twelve_l264_26429

/-- The time taken by the son to complete the work alone, given the man's and combined work rates -/
noncomputable def son_work_time (man_time : ℝ) (combined_time : ℝ) : ℝ :=
  1 / (1 / combined_time - 1 / man_time)

/-- Theorem stating that the son can complete the work alone in 12 days -/
theorem son_work_time_is_twelve :
  son_work_time 4 3 = 12 := by
  -- Unfold the definition of son_work_time
  unfold son_work_time
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_son_work_time_is_twelve_l264_26429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_n_plus_one_l264_26450

theorem polynomial_value_at_n_plus_one (n : ℕ) (p : ℝ → ℝ) :
  (∀ (k : ℕ), k ≤ n → p k = k / (k + 1)) →
  (∃ (c : Polynomial ℝ), ∀ (x : ℝ), p x = c.eval x ∧ c.degree ≤ n) →
  p (n + 1) = if n % 2 = 1 then 1 else n / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_n_plus_one_l264_26450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l264_26475

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

-- Define the theorem
theorem problem_solution (m : ℝ) (A B C : ℝ) :
  (∀ x ∈ Set.Icc m (π / 2), f x ∈ Set.Icc (-Real.sqrt 3) 2) →
  f (A / 4) = 2 →
  Real.sin B = (3 * Real.sqrt 3 / 4) * Real.cos C →
  m = -π / 3 ∧ Real.sin C = Real.sqrt 21 / 7 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l264_26475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l264_26418

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 8 = 1

-- Define the focus of the hyperbola
noncomputable def focus : ℝ := 2 * Real.sqrt 6

-- Define the line passing through the focus and perpendicular to the real axis
def line (x : ℝ) : Prop := x = focus

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | hyperbola p.1 p.2 ∧ line p.1}

-- Theorem stating the distance between intersection points is 4
theorem intersection_distance :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l264_26418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_values_l264_26477

/-- The function representing the number at position (i, j) in the table -/
def a : ℕ → ℕ → ℕ := sorry

/-- The rule for odd-indexed rows and columns -/
axiom odd_rule (i j : ℕ) (h : Odd i) : a i j = i^2 - (j - 1)

/-- The rule for even-indexed rows and columns -/
axiom even_rule (i j : ℕ) (h : Even i) : a i j = (i - 1)^2 + j

/-- The main theorem to prove -/
theorem table_values : (a 12 8 = 140) ∧ (a 8 4 = 60) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_values_l264_26477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_order_ratio_l264_26456

theorem sock_order_ratio : 
  ∀ (g : ℚ) (y : ℚ),
  y > 0 →
  g > 0 →
  0.9 * (3 * g * y + 3 * y) = 1.2 * (0.9 * (9 * y + g * y)) →
  (3 : ℚ) / g = 3 / 4 := by
  intros g y hy hg heq
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_order_ratio_l264_26456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sum_l264_26403

-- Define x as noncomputable
noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 61) / 2 + 3 / 2)

-- Define the equation
def equation (a b c : ℕ+) : Prop :=
  x^100 = 2*x^98 + 18*x^96 + 15*x^94 - x^50 + (a : ℝ)*x^46 + (b : ℝ)*x^44 + (c : ℝ)*x^42

-- Theorem statement
theorem solution_sum :
  ∃! (a b c : ℕ+), equation a b c ∧ (a : ℕ) + (b : ℕ) + (c : ℕ) = 91 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sum_l264_26403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_and_reciprocal_of_abs_neg_one_third_l264_26408

theorem opposite_and_reciprocal_of_abs_neg_one_third :
  (∃ x : ℚ, x = |(-1/3 : ℚ)| ∧ 
    (∃ y : ℚ, y = -x ∧ x + y = 0) ∧
    (∃ z : ℚ, z * x = 1)) ∧
  -|(-1/3 : ℚ)| = -1/3 ∧
  |(-1/3 : ℚ)|⁻¹ = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_and_reciprocal_of_abs_neg_one_third_l264_26408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l264_26427

theorem trig_expression_simplification (A : ℝ) (h1 : Real.cos A ≠ 0) (h2 : Real.sin A ≠ 0) :
  (1 - (Real.cos A / Real.sin A) + (1 / Real.sin A)) * (1 - (Real.sin A / Real.cos A) + (1 / Real.cos A)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l264_26427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_with_specific_ends_and_alternating_l264_26469

/-- The number of ways to arrange n distinct objects -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange 4 boys and 4 girls in a row -/
def total_people : ℕ := 8

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 4

theorem arrangement_with_specific_ends_and_alternating :
  (-- Arrangement with A and B at ends
   2 * arrangements (total_people - 2) = 1440) ∧
  (-- Alternating arrangement
   arrangements num_boys * (arrangements (num_girls + 1) / arrangements 1) = 2880) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_with_specific_ends_and_alternating_l264_26469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eightieth_element_value_l264_26466

def sequence_value (row : ℕ) : ℕ := 3 * row

def row_length (row : ℕ) : ℕ := row^3

def total_elements_before_row (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) row_length

theorem eightieth_element_value :
  ∃ (row : ℕ), 
    total_elements_before_row row < 80 ∧ 
    total_elements_before_row (row + 1) ≥ 80 ∧
    sequence_value row = 12 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eightieth_element_value_l264_26466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_difference_l264_26474

theorem permutation_difference (n : ℕ) 
  (h1 : 0 ≤ (n : ℤ) + 3 ∧ (n : ℤ) + 3 ≤ 2*(n : ℤ)) 
  (h2 : 0 ≤ (n : ℤ) + 1 ∧ (n : ℤ) + 1 ≤ 4) : 
  (Nat.descFactorial (2*n) (n + 3)) - (Nat.descFactorial 4 (n + 1)) = 696 := by
  sorry

#check permutation_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_difference_l264_26474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_rectangle_area_l264_26442

/-- A rectangle with a circle tangent to three sides and passing through the diagonal midpoint -/
structure TangentCircleRectangle where
  -- The rectangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- The circle
  center : ℝ × ℝ
  radius : ℝ
  -- Conditions
  is_rectangle : (A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ C.2 = D.2)
  tangent_AB : dist center (A.1, center.2) = radius
  tangent_AD : dist center (center.1, A.2) = radius
  tangent_CD : dist center C = radius
  midpoint_condition : dist center ((A.1 + C.1)/2, (A.2 + C.2)/2) = radius

/-- The area of a TangentCircleRectangle is 8r^2 -/
theorem tangent_circle_rectangle_area (tcr : TangentCircleRectangle) :
  abs ((tcr.B.1 - tcr.A.1) * (tcr.C.2 - tcr.A.2)) = 8 * tcr.radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_rectangle_area_l264_26442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l264_26480

noncomputable section

open Real

-- Define the original function
def f (x : ℝ) : ℝ := cos (x - π/3)

-- Define the transformed function
def g (x : ℝ) : ℝ := cos (x/2 - π/4)

-- Define the transformation
def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2 * (x + π/6) - π/3)

-- Theorem statement
theorem axis_of_symmetry :
  (∀ x : ℝ, transform f x = g x) →
  (∀ x : ℝ, g (π/2 + x) = g (π/2 - x)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l264_26480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l264_26444

noncomputable def z (m : ℝ) : ℂ := (m * (m + 2)) / (m - 1) + (m^2 + 2*m - 3) * Complex.I

theorem complex_number_properties :
  ∀ m : ℝ,
  (z m ∈ Set.range (Complex.ofReal) ↔ m = -3) ∧
  (z m ∈ {w : ℂ | w.re = 0 ∧ w.im ≠ 0} ↔ m = 0 ∨ m = -2) ∧
  (¬∃ m : ℝ, z m ∈ {w : ℂ | w.re = 0 ∧ w.im > 0}) ∧
  (z m ∈ {w : ℂ | w.re < 0 ∧ w.im > 0} ↔ m < -3) ∧
  (z m ∈ {w : ℂ | w.re + w.im = -3} ↔ m = 0 ∨ m = -2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l264_26444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_children_count_l264_26482

theorem zoo_children_count : ℕ := by
  let num_adults : ℕ := 10
  let child_ticket_price : ℕ := 10
  let adult_ticket_price : ℕ := 16
  let total_cost : ℕ := 220
  let num_children : ℕ := (total_cost - num_adults * adult_ticket_price) / child_ticket_price
  have h : num_children = 6 := by
    -- Proof goes here
    sorry
  exact num_children

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_children_count_l264_26482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_trajectory_l264_26478

/-- The trajectory of a point P(x,y) satisfying PA² = 3PB², where A(-1,0) and B(1,0) -/
theorem point_trajectory (x y : ℝ) : 
  (((x + 1)^2 + y^2) = 3 * ((x - 1)^2 + y^2)) → 
  (x^2 + y^2 - 4*x + 1 = 0) := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_trajectory_l264_26478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sqrt_expressions_l264_26436

theorem complex_sqrt_expressions : 
  (Real.sqrt 45 + Real.sqrt 18) - (Real.sqrt 8 - Real.sqrt 125) = 8 * Real.sqrt 5 + Real.sqrt 2 ∧
  Real.sqrt 48 / (-Real.sqrt 3) - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = -4 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sqrt_expressions_l264_26436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l264_26497

/-- Parallelogram ABCD with given vertices -/
structure Parallelogram where
  A : ℝ × ℝ := (2, -3)
  B : ℝ × ℝ := (14, 9)
  C : ℝ × ℝ := (5, 12)
  D : ℝ × ℝ := (11, -6)

/-- The intersection point of the diagonals in the parallelogram -/
noncomputable def diagonalIntersection (p : Parallelogram) : ℝ × ℝ :=
  ((p.A.1 + p.C.1) / 2, (p.A.2 + p.C.2) / 2)

/-- The area of the parallelogram -/
noncomputable def parallelogramArea (p : Parallelogram) : ℝ :=
  (1/2) * abs (
    p.A.1 * p.B.2 + p.B.1 * p.C.2 + p.C.1 * p.D.2 + p.D.1 * p.A.2 -
    (p.A.2 * p.B.1 + p.B.2 * p.C.1 + p.C.2 * p.D.1 + p.D.2 * p.A.1)
  )

theorem parallelogram_properties (p : Parallelogram) :
  diagonalIntersection p = (8, 3) ∧ parallelogramArea p = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l264_26497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_track_arrangements_l264_26447

/-- The number of ways to arrange n athletes on n tracks with exactly k athletes on their numbered track. -/
def arrangements (n k : ℕ) : ℕ := sorry

/-- The number of derangements of n elements. -/
def derangement (n : ℕ) : ℕ := sorry

theorem athlete_track_arrangements :
  arrangements 5 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_track_arrangements_l264_26447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PQT_twice_OPQR_l264_26411

/-- Square OPQR with O at origin and Q at (3,3) -/
structure Square :=
  (O : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (is_origin : O = (0, 0))
  (is_diagonal : Q = (3, 3))

/-- Point T -/
def T : ℝ × ℝ := (3, 12)

/-- Area of a triangle given three points -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- Theorem: Area of PQT is twice the area of OPQR -/
theorem area_PQT_twice_OPQR (s : Square) :
  let P := (3, 0)
  let area_OPQR := 9
  triangle_area P s.Q T = 2 * area_OPQR := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PQT_twice_OPQR_l264_26411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_eq_seven_plus_six_root_two_l264_26493

/-- Represents a hyperbola with center (h, k), focus (3, 10), and vertex (3, 4) -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  focus_y : ℝ
  vertex_y : ℝ
  center_eq : (h, k) = (3, 1)
  focus_eq : (h, focus_y) = (3, 10)
  vertex_eq : (h, vertex_y) = (3, 4)

/-- The sum of h, k, a, and b for the given hyperbola -/
noncomputable def hyperbola_sum (hyp : Hyperbola) : ℝ :=
  let a := |hyp.k - hyp.vertex_y|
  let c := |hyp.k - hyp.focus_y|
  let b := Real.sqrt (c^2 - a^2)
  hyp.h + hyp.k + a + b

/-- Theorem stating that the sum h + k + a + b equals 7 + 6√2 for the given hyperbola -/
theorem hyperbola_sum_eq_seven_plus_six_root_two (hyp : Hyperbola) :
  hyperbola_sum hyp = 7 + 6 * Real.sqrt 2 := by
  sorry

#eval "Hyperbola sum theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_eq_seven_plus_six_root_two_l264_26493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l264_26454

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.sqrt x
  else if x ≥ 1 then 2 * (x - 1)
  else 0  -- We need to define f for all real numbers

-- State the theorem
theorem function_property (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : f a = f (a + 1)) :
  f (1 / a) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l264_26454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_theorem_l264_26434

/-- The radius of the wheel in centimeters -/
noncomputable def wheel_radius : ℝ := 24.2

/-- The number of revolutions the wheel completes -/
def num_revolutions : ℕ := 500

/-- Converts centimeters to meters -/
noncomputable def cm_to_m (cm : ℝ) : ℝ := cm / 100

/-- Calculates the circumference of a circle given its radius -/
noncomputable def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius

/-- Theorem: The total distance covered by the wheel is approximately 760 meters -/
theorem wheel_distance_theorem :
  ∃ (distance : ℝ), abs (distance - 760) < 1 ∧
  distance = num_revolutions * circumference (cm_to_m wheel_radius) := by
  sorry

#eval num_revolutions -- This line is added to ensure there's some computable content

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_theorem_l264_26434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_16_divisors_l264_26443

open Nat

def count_divisors (n : ℕ) : ℕ := (divisors n).card

theorem smallest_with_16_divisors :
  ∀ n : ℕ, n > 0 → count_divisors n = 16 → n ≥ 216 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_16_divisors_l264_26443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_through_given_points_l264_26437

/-- Theorem: The slope of a line passing through (1,0) and (-2,3) is -1 -/
theorem line_slope_through_given_points :
  (3 - 0) / ((-2) - 1) = -1 := by
  -- Simplify the fraction
  have h1 : (3 - 0) / ((-2) - 1) = 3 / (-3) := by
    ring
  -- Evaluate the fraction
  have h2 : 3 / (-3) = -1 := by
    norm_num
  -- Combine the steps
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_through_given_points_l264_26437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l264_26401

-- Define the quadratic function f
noncomputable def f : ℝ → ℝ := sorry

-- State the conditions
axiom condition1 : ∀ x : ℝ, f (x + 1) - f x = 2 * x
axiom condition2 : ∀ x : ℝ, f x ≥ x^2 - x + 1
axiom condition3 : ∀ x : ℝ, x ≥ 0 → f x ≤ 2^x

-- State the theorem
theorem quadratic_function_properties :
  (∀ x : ℝ, f x = x^2 - x + 1) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f x > 2 * x + m) ↔ m < -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l264_26401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l264_26453

/-- Representation of an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  isArithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 0 + seq.a (n - 1)) / 2

theorem arithmetic_sequence_sum_property 
  (seq : ArithmeticSequence) (m n : ℕ) :
  m ≠ n →
  sumFirstN seq m = n^2 →
  sumFirstN seq n = m^2 →
  sumFirstN seq (n + m) = -(m + n)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l264_26453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_banks_for_10million_l264_26486

/-- The minimum number of banks needed to fully insure a given amount -/
def min_banks_needed (total_amount : ℕ) (max_insured_per_bank : ℕ) : ℕ :=
  (total_amount + max_insured_per_bank - 1) / max_insured_per_bank

/-- Theorem stating the minimum number of banks needed for the given problem -/
theorem min_banks_for_10million (total_amount : ℕ) (max_insured_per_bank : ℕ) 
  (h1 : total_amount = 10000000)
  (h2 : max_insured_per_bank = 1400000) :
  min_banks_needed total_amount max_insured_per_bank = 8 := by
  sorry

#eval min_banks_needed 10000000 1400000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_banks_for_10million_l264_26486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_alternating_sums_for_eight_l264_26472

/-- Alternating sum of a subset of {1, 2, ..., n} -/
def alternatingSum (S : Finset Nat) : Int :=
  sorry

/-- The set {1, 2, ..., n} -/
def fullSet (n : Nat) : Finset Nat :=
  Finset.range n

/-- Sum of all alternating sums for subsets of {1, 2, ..., n} -/
def sumOfAlternatingSums (n : Nat) : Int :=
  sorry

/-- Extra value added for subsets of size 3 -/
def extraForSizeThree : Nat := 3

theorem sum_of_alternating_sums_for_eight :
  sumOfAlternatingSums 8 + (Finset.card (Finset.filter (fun S => Finset.card S = 3) (Finset.powerset (fullSet 8)))) * extraForSizeThree = 1192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_alternating_sums_for_eight_l264_26472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duct_tape_strands_l264_26484

theorem duct_tape_strands : ℕ := by
  -- Define Hannah's cutting rate
  let hannah_rate : ℕ := 8
  -- Define son's cutting rate
  let son_rate : ℕ := 3
  -- Define time taken to free younger son
  let time_taken : ℕ := 2
  -- Define total strands
  let total_strands : ℕ := (hannah_rate + son_rate) * time_taken
  -- Theorem statement
  have h : total_strands = 22 := by
    rfl
  exact total_strands


end NUMINAMATH_CALUDE_ERRORFEEDBACK_duct_tape_strands_l264_26484
