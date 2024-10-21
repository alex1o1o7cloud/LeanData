import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_height_l1035_103559

/-- The height of a right circular cylinder inscribed in a sphere -/
noncomputable def cylinder_height (sphere_radius cylinder_radius : ℝ) : ℝ :=
  2 * Real.sqrt (sphere_radius ^ 2 - cylinder_radius ^ 2)

/-- Theorem: The height of a specific inscribed cylinder -/
theorem inscribed_cylinder_height :
  let sphere_radius : ℝ := 6
  let cylinder_radius : ℝ := 3
  cylinder_height sphere_radius cylinder_radius = 6 * Real.sqrt 3 := by
  sorry

#check inscribed_cylinder_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_height_l1035_103559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_11_terms_l1035_103551

def arithmetic_sequence (n : ℕ) : ℤ := 1 - 2 * n

def S (n : ℕ) : ℚ := (arithmetic_sequence 1 + arithmetic_sequence n) * n / 2

def sequence_Sn_div_n (n : ℕ) : ℚ := S n / n

theorem sum_of_first_11_terms :
  (Finset.range 11).sum (λ i => sequence_Sn_div_n (i + 1)) = -66 := by
  sorry

#eval (Finset.range 11).sum (λ i => sequence_Sn_div_n (i + 1))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_11_terms_l1035_103551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_expressions_l1035_103546

theorem square_root_expressions : 
  ((Real.sqrt 2 + 2 * Real.sqrt 3)^2 = 14 + 4 * Real.sqrt 6) ∧
  (Real.sqrt 3 * (Real.sqrt 2 - Real.sqrt 3) - Real.sqrt 24 - |Real.sqrt 6 - 3| = -6) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_expressions_l1035_103546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_midpoint_trajectory_l1035_103594

/-- Given an ellipse C with foci at (-2, 0) and (2, 0) and eccentricity √2/2,
    prove that the trajectory of the midpoint M of QP is x²/2 + y²/4 = 1,
    where P is any point on C and Q is the intersection of the perpendicular
    line from P to the y-axis. -/
theorem ellipse_midpoint_trajectory 
  (C : Set (ℝ × ℝ)) 
  (h_foci : ∀ (x y : ℝ), (x, y) ∈ C ↔ 
    Real.sqrt ((x + 2)^2 + y^2) + Real.sqrt ((x - 2)^2 + y^2) = 4 * Real.sqrt 2)
  (h_eccentricity : (2 : ℝ) / (2 * Real.sqrt 2) = Real.sqrt 2 / 2)
  (P : ℝ × ℝ)
  (h_P_on_C : P ∈ C)
  (Q : ℝ × ℝ)
  (h_Q_on_yaxis : Q.1 = 0)
  (h_PQ_perpendicular : (P.1 - Q.1) * (P.2 - Q.2) = 0)
  (M : ℝ × ℝ)
  (h_M_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  M.1^2/2 + M.2^2/4 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_midpoint_trajectory_l1035_103594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_failing_percentage_l1035_103583

/-- Represents a class of students -/
structure MyClass where
  total_students : ℕ
  failing_students : ℕ

/-- The percentage of failing students in a class -/
def failing_percentage (c : MyClass) : ℚ :=
  (c.failing_students : ℚ) / (c.total_students : ℚ) * 100

theorem current_failing_percentage
  (c : MyClass)
  (h1 : failing_percentage {total_students := c.total_students, failing_students := c.failing_students - 1} = 24)
  (h2 : failing_percentage {total_students := c.total_students - 1, failing_students := c.failing_students - 1} = 25) :
  failing_percentage c = 24 := by
  sorry

#check current_failing_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_failing_percentage_l1035_103583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_tile_modified_chessboard_l1035_103543

/-- Represents a chessboard square --/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the color of a square --/
inductive Color
  | Black
  | White

/-- Determines the color of a square based on its coordinates --/
def squareColor (s : Square) : Color :=
  if (s.row.val + s.col.val) % 2 = 0 then Color.Black else Color.White

/-- Represents the modified chessboard with opposite corners removed --/
def ModifiedChessboard : Set Square :=
  { s : Square | s ≠ ⟨0, 0⟩ ∧ s ≠ ⟨7, 7⟩ }

/-- Represents a 1x2 domino placement on the chessboard --/
structure Domino where
  square1 : Square
  square2 : Square
  adjacent : (square1.row = square2.row ∧ square1.col.val + 1 = square2.col.val) ∨
             (square1.col = square2.col ∧ square1.row.val + 1 = square2.row.val)

/-- The main theorem stating that it's impossible to tile the modified chessboard with dominoes --/
theorem impossible_to_tile_modified_chessboard :
  ¬ ∃ (tiling : Set Domino),
    (∀ s, s ∈ ModifiedChessboard → ∃ d, d ∈ tiling ∧ (s = d.square1 ∨ s = d.square2)) ∧
    (∀ d1 d2, d1 ∈ tiling → d2 ∈ tiling → d1 ≠ d2 → 
      d1.square1 ≠ d2.square1 ∧ d1.square1 ≠ d2.square2 ∧
      d1.square2 ≠ d2.square1 ∧ d1.square2 ≠ d2.square2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_tile_modified_chessboard_l1035_103543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_solutions_l1035_103553

/-- The system of equations -/
def system (x y z w : ℝ) : Prop :=
  x = z + w + z*w - z*w*x ∧
  y = w + x + w*x - w*x*y ∧
  z = x + y + x*y - x*y*z ∧
  w = y + z + y*z - y*z*w

/-- The set of all solutions to the system -/
def solution_set : Set (ℝ × ℝ × ℝ × ℝ) :=
  {p | system p.1 p.2.1 p.2.2.1 p.2.2.2}

/-- There exists a finite subset of the solution set with exactly 5 elements -/
theorem five_solutions :
  ∃ (S : Set (ℝ × ℝ × ℝ × ℝ)), S ⊆ solution_set ∧ Finite S ∧ Nat.card S = 5 ∧ 
  ∀ p ∈ solution_set, ∃ q ∈ S, p = q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_solutions_l1035_103553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_from_perpendicular_lines_parallel_planes_from_perpendicular_parallel_lines_l1035_103518

-- Define the types for planes and lines in 3D space
variable (P L : Type)

-- Define the relationships between lines and planes
variable (perpendicular_line_plane : L → P → Prop)
variable (perpendicular_line_line : L → L → Prop)
variable (parallel_line_line : L → L → Prop)
variable (plane_perpendicular : P → P → Prop)
variable (plane_parallel : P → P → Prop)
variable (line_in_plane : L → P → Prop)

-- Theorem for proposition ②
theorem perpendicular_planes_from_perpendicular_lines 
  (α β : P) (l m : L) 
  (h1 : perpendicular_line_plane l α) 
  (h2 : perpendicular_line_plane m β) 
  (h3 : perpendicular_line_line l m) : 
  plane_perpendicular α β :=
sorry

-- Theorem for proposition ③
theorem parallel_planes_from_perpendicular_parallel_lines 
  (α β : P) (m n : L) 
  (h1 : perpendicular_line_plane m α) 
  (h2 : perpendicular_line_plane n β) 
  (h3 : parallel_line_line m n) : 
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_from_perpendicular_lines_parallel_planes_from_perpendicular_parallel_lines_l1035_103518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_properties_l1035_103566

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => if x > 0 then 2 / x - 1 else -2 / x - 1

-- State the theorem
theorem even_function_properties :
  (∀ x, f x = f (-x)) →  -- f is an even function
  (∀ x > 0, f x = 2 / x - 1) →  -- given definition for x > 0
  (f (-1) = 1) ∧  -- prove f(-1) = 1
  (∀ x < 0, f x = -2 / x - 1)  -- prove the expression for x < 0
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_properties_l1035_103566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1035_103521

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x) + Real.cos (Real.sin x)

-- Theorem statement
theorem f_properties :
  (∃ x : ℝ, f x > Real.sqrt 2) ∧
  (∀ x : ℝ, f (x - 2 * Real.pi) = f x) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 Real.pi → f (x + Real.pi) > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1035_103521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squares_l1035_103527

/-- Represents a quadrilateral ABCD in a plane -/
structure Quadrilateral :=
  (A B C D : EuclideanSpace ℝ (Fin 2))

/-- The condition that AB = CD -/
def equal_sides (q : Quadrilateral) : Prop :=
  ‖q.A - q.B‖ = ‖q.C - q.D‖

/-- The condition that AD = DC = CB = 1 -/
def unit_sides (q : Quadrilateral) : Prop :=
  ‖q.A - q.D‖ = 1 ∧ ‖q.D - q.C‖ = 1 ∧ ‖q.C - q.B‖ = 1

/-- The area of triangle ABD -/
noncomputable def S (q : Quadrilateral) : ℝ :=
  abs (Matrix.det ![q.A - q.D, q.B - q.D]) / 2

/-- The area of triangle BCD -/
noncomputable def T (q : Quadrilateral) : ℝ :=
  abs (Matrix.det ![q.B - q.D, q.C - q.D]) / 2

/-- The theorem stating the maximum value of S^2 + T^2 -/
theorem max_sum_squares (q : Quadrilateral) 
  (h1 : equal_sides q) (h2 : unit_sides q) :
  ∃ (max : ℝ), ∀ (q' : Quadrilateral), 
    equal_sides q' → unit_sides q' → 
    (S q')^2 + (T q')^2 ≤ max ∧ 
    max = (1 : ℝ) / 2 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squares_l1035_103527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_problem_l1035_103505

/-- An arithmetic sequence {a_n} -/
def a : ℕ → ℝ := sorry

/-- A geometric sequence {b_n} -/
def b : ℕ → ℝ := sorry

theorem arithmetic_geometric_sequence_problem 
  (h1 : a 1 + a 2 = 10)
  (h2 : a 4 - a 3 = 2)
  (h3 : b 2 = a 3)
  (h4 : b 3 = a 7) :
  (∀ n : ℕ, a n = 2 * n + 2) ∧ 
  (a 63 = b 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_problem_l1035_103505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_common_points_l1035_103569

noncomputable section

-- Define the basic geometric objects
variable (A B C D X Y : EuclideanPlane)
variable (O O₁ : EuclideanPlane)

-- Define the cyclic quadrilateral ABCD
def is_cyclic_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define the arbitrary circle passing through C and D
def circle_through_C_D (C D X Y : EuclideanPlane) : Prop := sorry

-- Define the circles CAY and CBX
def circle_CAY (C A Y : EuclideanPlane) : Set EuclideanPlane := sorry
def circle_CBX (C B X : EuclideanPlane) : Set EuclideanPlane := sorry

-- Define the line PQ
def line_PQ (O O₁ : EuclideanPlane) : Set EuclideanPlane := sorry

-- Define the perpendicular line from C to PQ
def perp_line_C_to_PQ (C : EuclideanPlane) (pq : Set EuclideanPlane) : Set EuclideanPlane := sorry

-- The main theorem
theorem locus_of_common_points 
  (h1 : is_cyclic_quadrilateral A B C D)
  (h2 : circle_through_C_D C D X Y) :
  ∃ (pq : Set EuclideanPlane), 
    (circle_CAY C A Y ∩ circle_CBX C B X) = perp_line_C_to_PQ C pq :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_common_points_l1035_103569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_length_arithmetic_progression_without_digit_l1035_103542

-- Define p_adic_digits as a function
def p_adic_digits (p : ℕ) (n : ℕ) : List ℕ :=
  sorry -- Definition of p-adic digits representation

theorem maximal_length_arithmetic_progression_without_digit (p : ℕ) (k : ℕ) 
  (h_prime : Nat.Prime p) (h_p_ge_5 : p ≥ 5) (h_k_digit : k < p) :
  ∃ (seq : ℕ → ℕ) (d : ℕ),
    d ≠ 0 ∧
    (∀ n, seq (n + 1) - seq n = d) ∧
    (∀ n < p - 1, ∀ digit, digit ∈ p_adic_digits p (seq n) → digit ≠ k) ∧
    (∀ m : ℕ, ∀ seq' : ℕ → ℕ, ∀ d' : ℕ,
      d' ≠ 0 →
      (∀ n, seq' (n + 1) - seq' n = d') →
      (∀ n < m, ∀ digit, digit ∈ p_adic_digits p (seq' n) → digit ≠ k) →
      m ≤ p - 1) :=
by
  sorry -- Proof to be implemented


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_length_arithmetic_progression_without_digit_l1035_103542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_ratio_l1035_103513

/-- A line y = b - 2x where 0 < b < 7 -/
structure Line where
  b : ℝ
  h1 : 0 < b
  h2 : b < 7

/-- Point P where the line intersects the y-axis -/
noncomputable def P (l : Line) : ℝ × ℝ := (0, l.b)

/-- Point S where the line intersects x = 4 -/
noncomputable def S (l : Line) : ℝ × ℝ := (4, l.b - 8)

/-- Point Q where the line intersects the x-axis -/
noncomputable def Q (l : Line) : ℝ × ℝ := (l.b / 2, 0)

/-- Area of triangle QOP -/
noncomputable def area_QOP (l : Line) : ℝ := (l.b^2) / 4

/-- Area of triangle QRS -/
noncomputable def area_QRS (l : Line) : ℝ := ((8 - l.b) * (l.b - 8)) / 4

/-- The main theorem -/
theorem line_intersection_ratio (l : Line) :
  (area_QRS l) / (area_QOP l) = 4 / 9 → l.b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_ratio_l1035_103513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1035_103589

open Real

-- Define the function f(x) = ln x - kx
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := log x - k * x

theorem f_properties :
  (∀ x > 0, f 1 x ≤ 0) ∧
  (∀ k > (1 / exp 1), ∀ x > 0, f k x ≠ 0) ∧
  (∀ k ≤ (1 / exp 1), ∃ x > 0, f k x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1035_103589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decomposition_l1035_103580

noncomputable def f : ℝ → ℝ := λ x ↦ Real.log (10^x + 1)

noncomputable def g : ℝ → ℝ := λ x ↦ x/2

noncomputable def h : ℝ → ℝ := λ x ↦ Real.log (10^x + 1) - x/2

theorem f_decomposition :
  (∀ x : ℝ, f x = g x + h x) ∧
  (∀ x : ℝ, g (-x) = -g x) ∧
  (∀ x : ℝ, h (-x) = h x) := by
  sorry

#check f_decomposition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decomposition_l1035_103580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_l1035_103562

theorem absolute_value_nested (x : ℤ) (h : x = -2023) : 
  (abs ((abs ((abs (abs x - 2*x)) - abs x)) - x)) = 6069 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_l1035_103562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ella_finished_on_saturday_l1035_103534

def days_to_read (n : ℕ) : ℕ := n + 1

def total_days (n : ℕ) : ℕ := Finset.sum (Finset.range n) (fun k => days_to_read (k + 1))

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ := (start_day + days_passed) % 7

theorem ella_finished_on_saturday (n : ℕ) (h : n = 20) :
  day_of_week 0 (total_days n) = 6 := by
  sorry

#eval total_days 20
#eval day_of_week 0 (total_days 20)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ella_finished_on_saturday_l1035_103534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1035_103536

/-- A function that determines if an angle is in the third quadrant -/
def is_third_quadrant (α : ℝ) : Prop :=
  Real.sin α < 0 ∧ Real.cos α < 0

/-- Theorem: If sin(α) * cos(α) > 0 and sin(α) + cos(α) < 0, then α is in the third quadrant -/
theorem angle_in_third_quadrant (α : ℝ) 
  (h1 : Real.sin α * Real.cos α > 0) 
  (h2 : Real.sin α + Real.cos α < 0) : 
  is_third_quadrant α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1035_103536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_m_range_l1035_103508

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x / (x + 1)

-- Theorem for monotonicity
theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Theorem for the range of m
theorem m_range : ∀ m : ℝ, f (2*m - 1) > f (1 - m) ↔ 2/3 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_m_range_l1035_103508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_inequality_l1035_103528

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- Theorem 1
theorem m_value (m : ℝ) : 
  (∀ x, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-1 : ℝ) 1) → m = 1 := by sorry

-- Theorem 2
theorem inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / a + 1 / (2 * b) + 1 / (3 * c) = 1 → a + 2 * b + 3 * c ≥ 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_inequality_l1035_103528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_contact_probability_l1035_103530

/-- The probability that at least one tourist from the first group can contact at least one tourist from the second group -/
theorem tourist_contact_probability (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) : 
  1 - (1 - p) ^ 42 = 1 - (1 - p) ^ 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_contact_probability_l1035_103530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_correct_l1035_103597

/-- Calculates the total milk production under variable conditions -/
noncomputable def totalMilkProduction (a b c d e f k : ℝ) : ℝ :=
  (b * d) / (a * c) * (f * k + e - f)

/-- Theorem stating that the calculated total milk production is correct -/
theorem milk_production_correct (a b c d e f k : ℝ) 
  (ha : a > 0) (hc : c > 0) (he : e ≥ 0) (hf : 0 ≤ f ∧ f ≤ e) (hk : k ≥ 0) :
  totalMilkProduction a b c d e f k = (b * d) / (a * c) * (f * k + e - f) := by
  -- Unfold the definition of totalMilkProduction
  unfold totalMilkProduction
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_correct_l1035_103597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chastity_lollipops_l1035_103535

/-- Calculates the number of lollipops Chastity bought given the conditions of her purchase. -/
theorem chastity_lollipops : 
  (15 - 5 - 2 * 2) / (3/2 : ℚ) = 4 := by
  -- Convert the decimal representations to rational numbers
  have h1 : (15 : ℚ) - (5 : ℚ) - (2 : ℚ) * (2 : ℚ) = (6 : ℚ) := by norm_num
  have h2 : (6 : ℚ) / (3/2 : ℚ) = 4 := by norm_num
  rw [h1, h2]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chastity_lollipops_l1035_103535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_distances_l1035_103532

/-- Ellipse parametric equations -/
noncomputable def ellipse (θ : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 * Real.cos θ, Real.sin θ)

/-- Line equation in polar form -/
def line (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.cos (θ + Real.pi / 3) = 3 * Real.sqrt 6

/-- Distance function from a point to the line -/
noncomputable def distance (x y : ℝ) : ℝ :=
  (|x - Real.sqrt 3 * y - 3 * Real.sqrt 6|) / 2

/-- Theorem stating the maximum and minimum distances -/
theorem ellipse_line_distances :
  (∃ θ : ℝ, distance (ellipse θ).1 (ellipse θ).2 = 2 * Real.sqrt 6) ∧
  (∃ θ : ℝ, distance (ellipse θ).1 (ellipse θ).2 = Real.sqrt 6) ∧
  (∀ θ : ℝ, Real.sqrt 6 ≤ distance (ellipse θ).1 (ellipse θ).2 ∧
            distance (ellipse θ).1 (ellipse θ).2 ≤ 2 * Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_distances_l1035_103532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_euclidean_complexity_l1035_103584

/-- Modified Euclidean algorithm for computing the GCD of two positive integers -/
def modifiedEuclidean (a b : ℕ) : ℕ := sorry

/-- Time complexity function for the modified Euclidean algorithm -/
noncomputable def timeComplexity (a b : ℕ) : ℝ := sorry

/-- Big O notation for comparing functions -/
def bigO (f g : ℕ → ℝ) : Prop := sorry

/-- Theorem stating the time complexity of the modified Euclidean algorithm -/
theorem modified_euclidean_complexity (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  bigO (λ n ↦ timeComplexity a b) (λ n ↦ (Real.log (a : ℝ))^2 + (Real.log (b : ℝ))^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_euclidean_complexity_l1035_103584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trisected_segment_length_l1035_103572

/-- Defines the trisection of a line segment AD by points B and C, resulting in points E and F. -/
def trisect_segment (A D B C E F : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (dist A E = dist E F) ∧ (dist E F = dist F D) ∧
  (B = E) ∧ (C = F)

/-- Defines the midpoint M of a line segment AD. -/
def is_midpoint (M A D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (dist A M = dist M D) ∧ (dist A M + dist M D = dist A D)

/-- Given a line segment AD trisected by points E and F, with M as the midpoint of AD and ME = 5, prove that the length of AD is 30. -/
theorem trisected_segment_length (A D E F M : EuclideanSpace ℝ (Fin 2)) (B C : EuclideanSpace ℝ (Fin 2)) : 
  trisect_segment A D B C E F →
  is_midpoint M A D →
  dist M E = 5 →
  dist A D = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trisected_segment_length_l1035_103572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_plus_icing_area_eq_24_l1035_103598

/-- Represents a cubical cake with given edge length -/
structure CubicalCake where
  edgeLength : ℝ

/-- Represents a triangular piece of the cake -/
structure TriangularPiece (cake : CubicalCake) where
  -- Additional properties of the triangular piece could be defined here

/-- Calculates the volume of a triangular piece of the cake -/
noncomputable def volumeOfPiece (cake : CubicalCake) (piece : TriangularPiece cake) : ℝ :=
  (cake.edgeLength ^ 2 / 4) * cake.edgeLength

/-- Calculates the area of icing on a triangular piece (top and bottom) -/
noncomputable def areaOfIcing (cake : CubicalCake) (piece : TriangularPiece cake) : ℝ :=
  cake.edgeLength ^ 2 / 2

/-- Theorem stating that the sum of volume and icing area of a triangular piece is 24 -/
theorem volume_plus_icing_area_eq_24 (cake : CubicalCake) (piece : TriangularPiece cake) 
    (h : cake.edgeLength = 4) : 
    volumeOfPiece cake piece + areaOfIcing cake piece = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_plus_icing_area_eq_24_l1035_103598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_solution_l1035_103548

noncomputable def f (t : ℝ) : ℝ := (t^3 + 12*t) / (3*t^2 + 4)

theorem triplet_solution (x y z : ℝ) :
  y = f x ∧ z = f y ∧ x = f z →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 2 ∧ y = 2 ∧ z = 2) ∨ (x = -2 ∧ y = -2 ∧ z = -2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_solution_l1035_103548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_to_base4_conversion_l1035_103539

/-- Converts a binary number (represented as a list of bits) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The original binary number 10111001₂ -/
def original_binary : List Bool := [true, false, true, true, true, false, false, true]

/-- Flips the third bit from the right in a binary number -/
def flip_third_from_right (bits : List Bool) : List Bool :=
  let n := bits.length
  if n ≥ 3 then
    bits.take (n - 3) ++ [!(bits.get ⟨n - 3, by sorry⟩)] ++ bits.drop (n - 2)
  else
    bits

/-- The flipped binary number 10111101₂ -/
def flipped_binary : List Bool := flip_third_from_right original_binary

/-- The expected base 4 result -/
def expected_base4 : List ℕ := [2, 3, 3, 1]

theorem binary_to_base4_conversion :
  decimal_to_base4 (binary_to_decimal flipped_binary) = expected_base4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_to_base4_conversion_l1035_103539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_a_part_b_l1035_103561

-- Define the triangle ABC in the real plane
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the circumcircle and its radius
variable (circle : Sphere (EuclideanSpace ℝ (Fin 2)) ℝ)
variable (R : ℝ)

-- Define point P on the circumcircle
variable (P : circle.sphere)

-- Define perpendiculars and distance
noncomputable def PA₁ : ℝ := sorry
noncomputable def PB₁ : ℝ := sorry
noncomputable def d : ℝ := sorry

-- Define angle α
noncomputable def α : ℝ := sorry

-- State the theorems to be proved
theorem part_a : PA₁ * (dist P A) = 2 * R * d := by sorry

theorem part_b : Real.cos α = (dist P A) / (2 * R) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_a_part_b_l1035_103561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_impossibility_l1035_103549

/-- Represents a point on a 2D plane --/
structure Point where
  x : Int
  y : Int

/-- Represents a move in the grasshopper game --/
def move (p₁ p₂ : Point) : Point :=
  { x := 2 * p₂.x - p₁.x,
    y := 2 * p₂.y - p₁.y }

/-- Represents the state of the grasshopper game --/
structure GameState where
  points : List Point

/-- Checks if a point is reachable from the initial state --/
def is_reachable (initial : GameState) (target : Point) : Prop :=
  ∃ (final : GameState),
    (∀ p, p ∈ final.points → p = target ∨ p ∈ initial.points) ∧
    (∀ p, p ∈ final.points → p ∉ initial.points →
      ∃ p₁ p₂, p₁ ∈ initial.points ∧ p₂ ∈ initial.points ∧ p = move p₁ p₂)

/-- The main theorem statement --/
theorem grasshopper_impossibility :
  let initial : GameState := { points := [{ x := 0, y := 0 }, { x := 0, y := 1 }, { x := 1, y := 0 }] }
  let target : Point := { x := 1, y := 1 }
  ¬ is_reachable initial target := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_impossibility_l1035_103549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1035_103503

theorem trigonometric_identities (α : ℝ) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : Real.sin α = 3 / 5) : 
  (Real.sin α ^ 2 + Real.sin (2 * α)) / (Real.cos α ^ 2 + Real.cos (2 * α)) = -15 / 23 ∧ 
  Real.tan (α - 5 * π / 4) = -7 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1035_103503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_phase_shift_l1035_103512

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def g (ω x : ℝ) : ℝ := Real.sin (ω * x)

theorem period_and_phase_shift (ω φ : ℝ) :
  ω > 0 ∧ 
  abs φ < π / 2 ∧
  (∀ x : ℝ, f ω φ (x + π) = f ω φ x) ∧
  (∀ x : ℝ, g ω x = f ω φ (x - π / 6)) →
  φ = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_phase_shift_l1035_103512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_rotated_angle_l1035_103524

theorem sine_of_rotated_angle (α : ℝ) :
  α ∈ Set.Ioo 0 (π / 2) →
  Real.cos α = 1 / 3 →
  Real.sin (α + π / 3) = (2 * Real.sqrt 2 + Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_rotated_angle_l1035_103524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_internally_l1035_103582

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Two circles are tangent internally if the distance between their centers
    is equal to the difference of their radii -/
def tangentInternally (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center = abs (c2.radius - c1.radius)

theorem circles_tangent_internally : 
  let c1 : Circle := ⟨(3, -2), 1⟩
  let c2 : Circle := ⟨(7, 1), 6⟩
  tangentInternally c1 c2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_internally_l1035_103582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_correct_l1035_103576

/-- The radius of two circles that are externally tangent to each other and internally tangent to the ellipse x^2 + 5y^2 = 6 -/
noncomputable def circle_radius : ℝ := 2 * Real.sqrt 6 / 5

/-- The equation of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 + 5*y^2 = 6

/-- The equation of the right circle -/
def is_on_right_circle (x y : ℝ) : Prop := (x - circle_radius)^2 + y^2 = circle_radius^2

/-- The circles are externally tangent to each other -/
axiom circles_externally_tangent : ∃ x y : ℝ, is_on_right_circle x y ∧ is_on_right_circle (-x) y

/-- The circles are internally tangent to the ellipse -/
axiom circles_internally_tangent_to_ellipse : 
  ∀ x y : ℝ, is_on_right_circle x y → is_on_ellipse x y

/-- The theorem stating that the radius of the circles is 2√6/5 -/
theorem circle_radius_is_correct : circle_radius = 2 * Real.sqrt 6 / 5 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_correct_l1035_103576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_integer_sum_equals_power_l1035_103525

theorem unique_positive_integer_sum_equals_power : ∃! (n : ℕ), n > 0 ∧ (
  (Finset.range (n - 1)).sum (λ i => (i + 2) * 3^(i + 2)) + n * 3^n = 3^(n + 8)
) ∧ n = 515 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_integer_sum_equals_power_l1035_103525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_value_of_d_l1035_103556

theorem least_value_of_d : 
  ∃ (min_d : ℝ), min_d = -1 ∧ 
    (∀ d : ℝ, |((3 - 2*d) / 5) + 2| ≤ 3 → d ≥ min_d) ∧
    |((3 - 2*min_d) / 5) + 2| ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_value_of_d_l1035_103556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_spacing_theorem_l1035_103586

/-- The distance between equally spaced trees along a road -/
noncomputable def distance_between_trees (road_length : ℝ) (num_trees : ℕ) : ℝ :=
  road_length / (num_trees - 1 : ℝ)

/-- Theorem: The distance between equally spaced trees along a 239.66-meter road
    with 24 trees (including one at each end) is approximately 10.42 meters -/
theorem tree_spacing_theorem :
  let road_length : ℝ := 239.66
  let num_trees : ℕ := 24
  abs (distance_between_trees road_length num_trees - 10.42) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_spacing_theorem_l1035_103586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camera_price_increase_l1035_103554

/-- Calculates the percentage increase between two prices -/
noncomputable def percentageIncrease (oldPrice newPrice : ℝ) : ℝ :=
  (newPrice - oldPrice) / oldPrice * 100

theorem camera_price_increase :
  let oldCameraPrice : ℝ := 4000
  let lensOriginalPrice : ℝ := 400
  let lensDiscount : ℝ := 200
  let totalPaid : ℝ := 5400
  let newCameraPrice : ℝ := totalPaid - (lensOriginalPrice - lensDiscount)
  percentageIncrease oldCameraPrice newCameraPrice = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_camera_price_increase_l1035_103554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_good_power_numbers_fifty_not_good_power_number_smallest_good_power_number_after_70_infinitely_many_good_power_numbers_l1035_103593

def sequenceCustom : ℕ → ℕ
  | 0 => 1
  | n + 1 => if n + 1 = 2^(Nat.log2 (n + 1)) then 2^(Nat.log2 (n + 1)) else sequenceCustom n

def S (n : ℕ) : ℕ := (Finset.range n).sum (fun i => sequenceCustom i)

def is_good_power_number (m : ℕ) : Prop :=
  ∃ p : ℕ, S m = 2^p

theorem first_three_good_power_numbers :
  is_good_power_number 1 ∧ is_good_power_number 2 ∧ is_good_power_number 3 ∧
  ∀ m < 3, is_good_power_number m → m = 1 ∨ m = 2 ∨ m = 3 := by
  sorry

theorem fifty_not_good_power_number :
  ¬ is_good_power_number 50 := by
  sorry

theorem smallest_good_power_number_after_70 :
  is_good_power_number 95 ∧ ∀ m, 70 < m → m < 95 → ¬ is_good_power_number m := by
  sorry

theorem infinitely_many_good_power_numbers :
  ∀ N, ∃ m > N, is_good_power_number m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_good_power_numbers_fifty_not_good_power_number_smallest_good_power_number_after_70_infinitely_many_good_power_numbers_l1035_103593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_cubic_l1035_103575

/-- Given a cubic function f(x) = ax³ + x + 1, prove that if its tangent line at
    (1, f(1)) passes through (2, 7), then a = 1. -/
theorem tangent_line_cubic (a : ℝ) : 
  (let f : ℝ → ℝ := λ x => a * x^3 + x + 1
   let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 1
   let tangent_line : ℝ → ℝ := λ x => f 1 + f' 1 * (x - 1)
   tangent_line 2 = 7) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_cubic_l1035_103575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1035_103545

-- Define the hyperbola
def hyperbola (b : ℝ) (x y : ℝ) : Prop := x^2 - y^2 / b^2 = 1 ∧ b > 0

-- Define the foci
def foci (F1 F2 : ℝ × ℝ) : Prop := ∃ c : ℝ, F1 = (-c, 0) ∧ F2 = (c, 0)

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define perpendicularity of two lines
def perpendicular (P F1 F2 : ℝ × ℝ) : Prop :=
  (F1.1 - P.1) * (F2.1 - P.1) + (F1.2 - P.2) * (F2.2 - P.2) = 0

-- State the theorem
theorem hyperbola_eccentricity (b : ℝ) (P F1 F2 : ℝ × ℝ) :
  hyperbola b P.1 P.2 →
  foci F1 F2 →
  distance P F1 + distance P F2 = 6 →
  perpendicular P F1 F2 →
  ∃ (a c : ℝ), c / a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1035_103545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonius_circle_bisector_l1035_103531

/-- The Apollonius circle C defined by fixed points A(-2,0) and B(4,0), 
    and the condition |PA|/|PB| = 1/2 for any point P on C -/
def ApolloniusCircle : Set (ℝ × ℝ) :=
  {P | let (x, y) := P
       (x + 2)^2 + y^2 = ((x - 4)^2 + y^2) / 4}

/-- Point A in the Cartesian plane -/
def A : ℝ × ℝ := (-2, 0)

/-- Point B in the Cartesian plane -/
def B : ℝ × ℝ := (4, 0)

/-- Origin O in the Cartesian plane -/
def O : ℝ × ℝ := (0, 0)

/-- Predicate to check if three points are collinear -/
def areCollinear (P Q R : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  let (x₃, y₃) := R
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Ray PO bisects ∠APB -/
def RayPOBisectsAngleAPB (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x + 2) * (x - 4) + y^2 = 0

theorem apollonius_circle_bisector :
  ∀ P ∈ ApolloniusCircle, ¬areCollinear P A B → RayPOBisectsAngleAPB P :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonius_circle_bisector_l1035_103531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_l1035_103578

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a
noncomputable def g (x : ℝ) : ℝ := x + 4 / x

theorem min_value_condition (a : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 3, ∃ x₂ ∈ Set.Icc 1 4, f a x₁ ≥ g x₂) →
  a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_l1035_103578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1035_103504

def S (n : ℕ) : ℕ := 2 * n^2 + 1

def a : ℕ → ℕ
| 0     => 3  -- Adding case for 0
| 1     => 3
| (n+2) => 4*(n+2) - 2

theorem sequence_general_term (n : ℕ) :
  Finset.sum (Finset.range n) (fun i => a (i + 1)) = S n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1035_103504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_time_sum_l1035_103501

theorem paint_time_sum (expert_rate amateur_rate beginner_rate : ℚ)
  (m n : ℕ) (hm : m > 0) (hn : n > 0) (hco : Nat.Coprime m n) :
  expert_rate = 1 →
  amateur_rate = (1 : ℚ) / 2 →
  beginner_rate = (1 : ℚ) / 3 →
  (expert_rate + amateur_rate + beginner_rate) * (m : ℚ) / (n : ℚ) = 3 →
  m + n = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_time_sum_l1035_103501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_half_l1035_103533

/-- A function f is odd if f(-x) = -f(x) for all x in the domain of f -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = x / ((2x+1)(x-a)) -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ x / ((2 * x + 1) * (x - a))

/-- If f(x) = x / ((2x+1)(x-a)) is an odd function, then a = 1/2 -/
theorem odd_function_implies_a_half (a : ℝ) :
  IsOdd (f a) → a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_half_l1035_103533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_line_perp_plane_implies_planes_perp_l1035_103514

-- Define the necessary types
variable (Point Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Theorem 1
theorem parallel_transitive (a b l : Line) :
  parallel a l → parallel b l → parallel a b := by sorry

-- Theorem 2
theorem line_perp_plane_implies_planes_perp (a : Line) (β γ : Plane) :
  in_plane a β → perpendicular a γ → plane_perpendicular β γ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_line_perp_plane_implies_planes_perp_l1035_103514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_sqrt_45000_l1035_103558

/-- A trapezoid with specific measurements -/
structure Trapezoid where
  longSide : ℝ
  shortSide : ℝ
  rightAngle : ℝ
  fortyFiveAngle : ℝ

/-- Two guards walking around the trapezoid -/
structure Guards where
  trapezoid : Trapezoid
  distance : ℝ  -- Distance between guards along the perimeter

/-- Maximum straight-line distance between guards -/
noncomputable def maxDistance (g : Guards) : ℝ := Real.sqrt 45000

/-- Theorem: The maximum straight-line distance between guards is √45000 meters -/
theorem max_distance_is_sqrt_45000 (g : Guards) 
  (h1 : g.trapezoid.longSide = 200)
  (h2 : g.trapezoid.shortSide = 100)
  (h3 : g.trapezoid.rightAngle = 90)
  (h4 : g.trapezoid.fortyFiveAngle = 45)
  (h5 : g.distance = (g.trapezoid.longSide + g.trapezoid.shortSide + 
         Real.sqrt 2 * g.trapezoid.shortSide + g.trapezoid.shortSide) / 2) :
  maxDistance g = Real.sqrt 45000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_sqrt_45000_l1035_103558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l1035_103573

/-- The sum of the infinite series ∑(k/(4^k)) for k from 1 to infinity -/
noncomputable def infiniteSeries : ℝ := ∑' k, (k : ℝ) / (4 ^ k)

/-- Theorem stating that the sum of the infinite series is equal to 4/9 -/
theorem infiniteSeries_sum : infiniteSeries = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l1035_103573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l1035_103510

/-- The area of a rectangle inscribed in an isosceles triangle -/
theorem inscribed_rectangle_area (b h x : ℝ) (θ : ℝ) : 
  b > 0 → h > 0 → x > 0 → x < h → 0 < θ → θ < π/2 →
  b * x * (h - x) / h = b * x * (h - x) / h := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l1035_103510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equality_periodic_l1035_103519

theorem tan_equality_periodic (n : ℤ) : 
  -180 < n ∧ n < 180 → Real.tan (n * Real.pi / 180) = Real.tan (860 * Real.pi / 180) → n = -40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equality_periodic_l1035_103519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_side_length_l1035_103526

/-- The side length of the nth inscribed equilateral triangle -/
noncomputable def L (n : ℕ+) : ℝ :=
  2 * n.val / 3

/-- The x-coordinate of the nth point on the x-axis -/
noncomputable def x_coord (n : ℕ+) : ℝ :=
  (n.val * (n.val + 1)) / 3

/-- The parabola function -/
noncomputable def parabola (x : ℝ) : ℝ :=
  Real.sqrt x

theorem inscribed_triangle_side_length (n : ℕ+) :
  L n = 2 * n.val / 3 := by
  -- Proof goes here
  sorry

#check inscribed_triangle_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_side_length_l1035_103526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_properties_l1035_103563

-- Define the slope angle type
def SlopeAngle := { α : Real // 0 ≤ α ∧ α < Real.pi }

-- Define a straight line in a Cartesian coordinate system
structure StraightLine where
  slope_angle : SlopeAngle
  has_defined_slope : Bool

-- Theorem statement
theorem slope_angle_properties :
  -- 1) The range of slope angles is [0, π)
  (∀ (α : Real), (∃ (l : StraightLine), l.slope_angle.val = α) ↔ (0 ≤ α ∧ α < Real.pi)) ∧
  -- 2) All straight lines have a slope angle
  (∀ (l : StraightLine), ∃ (α : SlopeAngle), l.slope_angle = α) ∧
  -- 3) There exist lines without a defined slope
  (∃ (l : StraightLine), ¬l.has_defined_slope) ∧
  -- 4) Non-monotonic relationship between slope angle and slope
  (∃ (l1 l2 : StraightLine), 
    l1.slope_angle.val < l2.slope_angle.val ∧ 
    l1.has_defined_slope ∧ 
    l2.has_defined_slope ∧
    (∃ (s1 s2 : Real), s1 > s2 ∧ 
      s1 = Real.tan l1.slope_angle.val ∧ 
      s2 = Real.tan l2.slope_angle.val)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_properties_l1035_103563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_enclosing_sphere_radius_is_correct_l1035_103587

/-- The radius of a sphere that encloses four spheres of radius 2 arranged in a tetrahedral configuration -/
noncomputable def enclosing_sphere_radius : ℝ := 2 + Real.sqrt 6

/-- The setup of four spheres in a tetrahedral configuration -/
structure TetrahedralSpheres where
  sphere_radius : ℝ
  edge_length : ℝ
  tetrahedral : edge_length = 2 * sphere_radius

/-- The theorem stating the radius of the smallest enclosing sphere -/
theorem smallest_enclosing_sphere_radius 
  (setup : TetrahedralSpheres) 
  (h_radius : setup.sphere_radius = 2) :
  enclosing_sphere_radius = 
    setup.sphere_radius + (setup.edge_length * Real.sqrt 6) / 4 := by
  sorry

/-- The main theorem proving that the radius of the smallest enclosing sphere is 2 + √6 -/
theorem enclosing_sphere_radius_is_correct 
  (setup : TetrahedralSpheres) 
  (h_radius : setup.sphere_radius = 2) :
  enclosing_sphere_radius = 2 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_enclosing_sphere_radius_is_correct_l1035_103587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_floor_sum_l1035_103571

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x / (1 + a^x)

theorem range_of_floor_sum (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  Set.range (λ x : ℝ ↦ ⌊f a x - 1/2⌋ + ⌊f a (-x) - 1/2⌋) = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_floor_sum_l1035_103571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1035_103502

noncomputable def f (x : ℝ) : ℝ := (1/2)^x

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  f ((a + b) / 2) ≤ f (Real.sqrt (a * b)) ∧ f (Real.sqrt (a * b)) ≤ f (2 * a * b / (a + b)) := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1035_103502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sum_theorem_l1035_103507

-- Define a convex curve
class ConvexCurve (α : Type*) [AddCommGroup α] [Inhabited α]

-- Define a polygon
class Polygon (α : Type*) [AddCommGroup α] [Inhabited α]

-- Define the property of being circumscribed
def Circumscribed {α : Type*} [AddCommGroup α] [Inhabited α] (P : Polygon α) (K : ConvexCurve α) : Prop := sorry

-- Define the property of having pairwise parallel and equally directed sides
def ParallelEquallyDirectedSides {α : Type*} [AddCommGroup α] [Inhabited α] (P₁ P₂ : Polygon α) : Prop := sorry

-- Define the sum operation for convex curves
def ConvexCurve.add {α : Type*} [AddCommGroup α] [Inhabited α] (K₁ K₂ : ConvexCurve α) : ConvexCurve α := sorry

-- Define the sum operation for polygons
def Polygon.add {α : Type*} [AddCommGroup α] [Inhabited α] (P₁ P₂ : Polygon α) : Polygon α := sorry

-- The main theorem
theorem circumscribed_sum_theorem 
  {α : Type*} [AddCommGroup α] [Inhabited α]
  (K₁ K₂ : ConvexCurve α) (L₁ L₂ : Polygon α) :
  Circumscribed L₁ K₁ → 
  Circumscribed L₂ K₂ → 
  ParallelEquallyDirectedSides L₁ L₂ → 
  Circumscribed (Polygon.add L₁ L₂) (ConvexCurve.add K₁ K₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sum_theorem_l1035_103507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_bound_l1035_103500

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A rectangle with width 1 and height 2 -/
def rectangle : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 2}

theorem min_distance_bound (points : Finset Point) 
    (h1 : points.card = 6)
    (h2 : ∀ p, p ∈ points → p ∈ rectangle) :
    ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ distance p q ≤ Real.sqrt 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_bound_l1035_103500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l1035_103592

-- Define the function f(x) = x · ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := Real.log x + 1

-- Theorem statement
theorem tangent_line_intersection (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧
    -- The tangent line equation
    (f_derivative 1 * (x - 1) = x^2 + a - f 1) ∧
    -- The tangent line touches the curve y = x^2 + a at exactly one point
    (∀ y : ℝ, y ≠ x → f_derivative 1 * (y - 1) ≠ y^2 + a - f 1)) →
  a = -3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l1035_103592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_square_l1035_103590

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- Represents a triangle -/
structure Triangle where
  -- Define triangle properties as needed
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Area of a square -/
def Square.area (s : Square) : ℝ := s.side * s.side

/-- Predicate indicating if a rectangle can be cut into two triangles -/
def Rectangle.can_form_triangles (r : Rectangle) (t1 t2 : Triangle) : Prop :=
  sorry

/-- Predicate indicating if a square can be formed from two triangles -/
def Square.can_form_from_triangles (s : Square) (t1 t2 : Triangle) : Prop :=
  sorry

/-- Predicate indicating if two triangles are congruent -/
def Triangle.congruent (t1 t2 : Triangle) : Prop :=
  sorry

/-- Theorem stating that a 10x20 rectangle can be cut into two congruent triangles
    that can form a square with side length 10√2 -/
theorem rectangle_to_square (r : Rectangle) (s : Square) : 
  r.width = 10 ∧ r.height = 20 →
  s.side = 10 * Real.sqrt 2 →
  ∃ (t1 t2 : Triangle), 
    Triangle.congruent t1 t2 ∧
    Rectangle.area r = Square.area s ∧
    Rectangle.can_form_triangles r t1 t2 ∧
    Square.can_form_from_triangles s t1 t2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_square_l1035_103590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_value_l1035_103555

/-- The circle equation: (x-a)^2 + (y-a)^2 = 1 -/
def circleEq (x y a : ℝ) : Prop := (x - a)^2 + (y - a)^2 = 1

/-- The line equation: y = 3x -/
def lineEq (x y : ℝ) : Prop := y = 3 * x

/-- The area of triangle CPQ -/
noncomputable def triangleArea (a : ℝ) : ℝ := (Real.sqrt (10 * a^2 - 4 * a^4)) / 5

theorem max_area_value (a : ℝ) (h : a > 0) :
  (∀ b, b > 0 → triangleArea b ≤ triangleArea a) ↔ a = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_value_l1035_103555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1035_103574

/-- The area of a triangle with base 4.5 and height 6 is 13.5 -/
theorem triangle_area (base height : Real)
  (h1 : base = 4.5)
  (h2 : height = 6) :
  (base * height) / 2 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1035_103574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_naomi_website_count_l1035_103517

/-- The time Katherine takes to develop a website -/
def katherine_time : ℚ := 20

/-- The additional time factor for Naomi compared to Katherine -/
def naomi_factor : ℚ := 5/4

/-- The total time Naomi spent developing websites -/
def naomi_total_time : ℚ := 750

/-- The number of websites Naomi developed -/
def naomi_websites : ℕ := 30

theorem naomi_website_count :
  (naomi_total_time / (katherine_time * naomi_factor)).floor = naomi_websites := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_naomi_website_count_l1035_103517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_lives_in_sixth_entrance_l1035_103523

/-- Represents a point in a 2D grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a house entrance -/
structure Entrance where
  location : Point
  number : ℕ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt (((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2) : ℝ)

/-- The layout of the houses -/
def houseLayout : List Entrance := sorry

/-- Vasya's entrance -/
def vasyaEntrance : Entrance :=
  { location := { x := 4, y := 0 }, number := 4 }

/-- Find the shortest path between two entrances -/
noncomputable def shortestPath (e1 e2 : Entrance) : ℝ := sorry

/-- Check if the path is the same going around both sides of Petya's house -/
def samePathBothSides (petyaEntrance : Entrance) : Prop := sorry

theorem petya_lives_in_sixth_entrance :
  ∃ (petyaEntrance : Entrance),
    petyaEntrance.number = 6 ∧
    petyaEntrance ∈ houseLayout ∧
    vasyaEntrance ∈ houseLayout ∧
    samePathBothSides petyaEntrance := by
  sorry

#check petya_lives_in_sixth_entrance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_lives_in_sixth_entrance_l1035_103523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_shifted_l1035_103529

-- Define the given function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4)

-- Define the reference function
noncomputable def g (x : ℝ) : ℝ := Real.sin x + Real.cos x

-- Theorem stating the equivalence
theorem f_eq_g_shifted :
  ∀ x : ℝ, f x = g (x - Real.pi / 4) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_shifted_l1035_103529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_25_percent_l1035_103522

/-- Represents a rectangle formed by overlapping squares -/
structure OverlappingSquaresRectangle where
  square_side : ℚ
  rectangle_width : ℚ
  rectangle_length : ℚ
  overlap_length : ℚ

/-- Calculate the percentage of shaded area in the rectangle -/
def shaded_area_percentage (r : OverlappingSquaresRectangle) : ℚ :=
  (r.overlap_length * r.rectangle_width) / (r.rectangle_width * r.rectangle_length) * 100

/-- Theorem stating that the shaded area percentage is 25% -/
theorem shaded_area_is_25_percent (r : OverlappingSquaresRectangle) 
    (h1 : r.square_side = 10)
    (h2 : r.rectangle_width = 10)
    (h3 : r.rectangle_length = 20)
    (h4 : r.overlap_length = 5) : 
  shaded_area_percentage r = 25 := by
  unfold shaded_area_percentage
  simp [h1, h2, h3, h4]
  norm_num

#eval shaded_area_percentage { square_side := 10, rectangle_width := 10, rectangle_length := 20, overlap_length := 5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_25_percent_l1035_103522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_cost_is_one_l1035_103552

/-- A business selling charm bracelets -/
structure BraceletBusiness where
  selling_price : ℚ
  bead_cost : ℚ
  bracelets_sold : ℕ
  total_profit : ℚ

/-- The cost of string for each bracelet -/
def string_cost (b : BraceletBusiness) : ℚ :=
  b.selling_price - b.bead_cost - b.total_profit / b.bracelets_sold

/-- Theorem stating that the string cost is $1 under given conditions -/
theorem string_cost_is_one (b : BraceletBusiness) 
  (h1 : b.selling_price = 6)
  (h2 : b.bead_cost = 3)
  (h3 : b.bracelets_sold = 25)
  (h4 : b.total_profit = 50) : 
  string_cost b = 1 := by
  sorry

/-- Compute the string cost for the given example -/
def example_string_cost : ℚ :=
  string_cost { selling_price := 6, bead_cost := 3, bracelets_sold := 25, total_profit := 50 }

#eval example_string_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_cost_is_one_l1035_103552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cuttable_prism_l1035_103547

/-- A right rectangular prism with integral side lengths -/
structure RectPrism where
  a : ℕ
  b : ℕ
  c : ℕ
  h_order : a ≤ b ∧ b ≤ c

/-- The volume of a rectangular prism -/
def Volume (Q : RectPrism) : ℚ := (Q.a : ℚ) * Q.b * Q.c

/-- Similarity relation between two rectangular prisms -/
def SimilarTo (Q1 Q2 : RectPrism) : Prop :=
  ∃ (k : ℚ), k > 0 ∧ 
    (Q1.a : ℚ) = k * Q2.a ∧
    (Q1.b : ℚ) = k * Q2.b ∧
    (Q1.c : ℚ) = k * Q2.c

/-- Predicate for a prism that can be cut as described in the problem -/
def CanBeCut (Q : RectPrism) : Prop :=
  ∃ (Q' : RectPrism), SimilarTo Q' Q ∧ Volume Q' = (1/4 : ℚ) * Volume Q

/-- The main theorem -/
theorem unique_cuttable_prism :
  ∃! (Q : RectPrism), Q.b = 2000 ∧ CanBeCut Q :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cuttable_prism_l1035_103547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_eight_l1035_103595

theorem power_product_eight (a b : ℕ) (h : (2^a)^b = 2^2) : 2^a * 2^b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_eight_l1035_103595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l1035_103516

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  /-- Distance from the origin to the plane -/
  distance : ℝ
  /-- x-coordinate of the intersection with the x-axis -/
  a : ℝ
  /-- y-coordinate of the intersection with the y-axis -/
  b : ℝ
  /-- z-coordinate of the intersection with the z-axis -/
  c : ℝ
  /-- The plane does not pass through the origin -/
  origin_distinct : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
  /-- The distance equation holds -/
  distance_eq : 1 / a^2 + 1 / b^2 + 1 / c^2 = 1 / distance^2

/-- The centroid of a triangle formed by the intersections -/
noncomputable def centroid (plane : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (plane.a / 3, plane.b / 3, plane.c / 3)

/-- The main theorem -/
theorem centroid_sum (plane : IntersectingPlane) (h : plane.distance = 2) :
    let (p, q, r) := centroid plane
    1 / p^2 + 1 / q^2 + 1 / r^2 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l1035_103516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_purely_imaginary_alpha_l1035_103567

theorem complex_purely_imaginary_alpha (α : ℝ) :
  (Complex.I * (1 - Real.cos α) = Complex.mk (Real.sin α) (1 - Real.cos α)) →
  ∃ k : ℤ, α = (2 * k + 1) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_purely_imaginary_alpha_l1035_103567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_on_hypotenuse_of_right_triangle_l1035_103557

/-- Given a right triangle with legs of lengths 3 and 4, the height on the hypotenuse is 12/5 -/
theorem height_on_hypotenuse_of_right_triangle (a b c h : ℝ) : 
  (a = 3 ∧ b = 4) →  -- Lengths of legs
  (c^2 = a^2 + b^2) →  -- Pythagorean theorem (right triangle)
  (h * c / 2 = a * b / 2) →  -- Area equality
  h = 12/5 := by 
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_on_hypotenuse_of_right_triangle_l1035_103557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_original_price_l1035_103511

/-- The original price of a cup of coffee, given the promotion rules and average price -/
theorem coffee_original_price : ∃ (original_price : ℝ),
  let avg_price : ℝ := 19
  let third_cup_price : ℝ := 3
  let total_cost : ℝ := 3 * avg_price
  original_price + original_price / 2 + third_cup_price = total_cost ∧
  original_price = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_original_price_l1035_103511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_sqrt_2_l1035_103550

noncomputable def point_A : ℝ × ℝ := (1, 2)
noncomputable def point_B : ℝ × ℝ := (2, 3)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt (((p2.1 - p1.1) ^ 2) + ((p2.2 - p1.2) ^ 2))

theorem distance_AB_is_sqrt_2 :
  distance point_A point_B = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_sqrt_2_l1035_103550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_parabola_l1035_103585

/-- The polar equation of the curve -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.sin θ)^2 - 2 * Real.cos θ = 0

/-- The Cartesian equation of a parabola -/
def is_parabola (x y : ℝ) : Prop :=
  ∃ (a : ℝ), a ≠ 0 ∧ y^2 = 2 * a * x

/-- The theorem stating that the given polar equation represents a parabola -/
theorem polar_equation_is_parabola :
  ∀ (x y : ℝ), (∃ (ρ θ : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ polar_equation ρ θ) →
  is_parabola x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_parabola_l1035_103585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_S₁_plus_S₂_l1035_103564

-- Define the parabola
noncomputable def C (x : ℝ) : ℝ := x^2

-- Define the tangent line
noncomputable def l (t x : ℝ) : ℝ := 2*t*x - t^2

-- Define S₁
noncomputable def S₁ (t : ℝ) : ℝ := (7*t^3)/12

-- Define S₂
noncomputable def S₂ (t : ℝ) : ℝ := 1/3 - t^3/3 - t + t^2

-- Theorem statement
theorem min_S₁_plus_S₂ :
  ∃ (min : ℝ), min = 1/3 ∧
  ∀ t : ℝ, 0 < t → t < 1 →
  S₁ t + S₂ t ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_S₁_plus_S₂_l1035_103564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_both_curves_l1035_103560

open Real

theorem tangent_line_to_both_curves :
  ∃ (x₀ : ℝ),
    (x₀ + 1 = exp x₀ ∧ deriv exp x₀ = 1) ∧
    (x₀ + 1 = -x₀^2 / 4 ∧ deriv (fun x => -x^2 / 4) x₀ = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_both_curves_l1035_103560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_units_digit_three_l1035_103577

def IntSet : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 12) (Finset.range 13)

def unitsDigitSum (a b : ℕ) : ℕ := (a + b) % 10

def favorableOutcomes : Finset (ℕ × ℕ) :=
  Finset.filter (λ p : ℕ × ℕ => p.1 ∈ IntSet ∧ p.2 ∈ IntSet ∧ unitsDigitSum p.1 p.2 = 3)
    (Finset.product IntSet IntSet)

theorem probability_units_digit_three :
  (favorableOutcomes.card : ℚ) / ((IntSet.card ^ 2) : ℚ) = 1/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_units_digit_three_l1035_103577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tiling_l1035_103568

/-- Represents a square on the chessboard -/
inductive Square
| White
| Black

/-- Represents the modified chessboard -/
def ModifiedChessboard : Type := Fin 8 → Fin 8 → Option Square

/-- Creates the initial 8x8 chessboard with alternating colors -/
def initialChessboard : Fin 8 → Fin 8 → Square :=
  fun row col => if (row.val + col.val) % 2 = 0 then Square.White else Square.Black

/-- Creates the modified chessboard with top-left and bottom-right squares removed -/
def createModifiedChessboard : ModifiedChessboard :=
  fun row col =>
    if (row = 0 && col = 0) || (row = 7 && col = 7) then
      none
    else
      some (initialChessboard row col)

/-- Counts the number of white and black squares in the modified chessboard -/
def countSquares (board : ModifiedChessboard) : Nat × Nat := Id.run do
  let mut white := 0
  let mut black := 0
  for row in [0:8] do
    for col in [0:8] do
      match board ⟨row, sorry⟩ ⟨col, sorry⟩ with
      | some Square.White => white := white + 1
      | some Square.Black => black := black + 1
      | none => pure ()
  (white, black)

/-- Represents a domino placement on the chessboard -/
structure DominoPlacement where
  row1 : Fin 8
  col1 : Fin 8
  row2 : Fin 8
  col2 : Fin 8
  adjacent : (row1 = row2 ∧ (col1.val + 1 = col2.val ∨ col2.val + 1 = col1.val)) ∨
             (col1 = col2 ∧ (row1.val + 1 = row2.val ∨ row2.val + 1 = row1.val))

/-- Theorem: It is impossible to tile the modified chessboard with dominos -/
theorem impossible_tiling (board : ModifiedChessboard)
  (h_board : board = createModifiedChessboard)
  (h_count : countSquares board = (30, 32)) :
  ¬∃ (placements : List DominoPlacement),
    (placements.length = 31) ∧
    (∀ p ∈ placements, ∃ (s1 s2 : Square),
      board p.row1 p.col1 = some s1 ∧
      board p.row2 p.col2 = some s2 ∧
      s1 ≠ s2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tiling_l1035_103568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_tax_percentage_problem_l1035_103565

/-- Calculates the sales tax percentage given the food bill, tip percentage, and total bill amount. -/
noncomputable def calculate_sales_tax_percentage (food_bill : ℝ) (tip_percentage : ℝ) (total_bill : ℝ) : ℝ :=
  ((total_bill - food_bill - (tip_percentage / 100) * food_bill) / food_bill) * 100

/-- Theorem stating that given the specific values in the problem, the sales tax percentage is approximately 9.17%. -/
theorem sales_tax_percentage_problem :
  let food_bill : ℝ := 30
  let tip_percentage : ℝ := 10
  let total_bill : ℝ := 35.75
  let calculated_tax_percentage := calculate_sales_tax_percentage food_bill tip_percentage total_bill
  ∃ ε > 0, |calculated_tax_percentage - 9.17| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_tax_percentage_problem_l1035_103565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1035_103544

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- A point on the ellipse -/
structure EllipsePoint (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1
  h_above_x : y > 0

/-- The ratio of distances from a point to the foci -/
noncomputable def lambda_ratio (E : Ellipse) (P : EllipsePoint E) : ℝ := 
  let c := Real.sqrt (E.a^2 - E.b^2)
  let PF2 := Real.sqrt ((P.x - c)^2 + P.y^2)
  let PF1 := Real.sqrt ((P.x + c)^2 + P.y^2)
  PF2 / PF1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ :=
  Real.sqrt (1 - E.b^2 / E.a^2)

/-- The main theorem about the eccentricity range -/
theorem eccentricity_range (E : Ellipse) (P : EllipsePoint E) 
  (h_perp : (P.x - Real.sqrt (E.a^2 - E.b^2))^2 + P.y^2 = E.b^4 / E.a^2)
  (h_lambda : lambda_ratio E P ∈ Set.Icc (1/3) (1/2)) :
  eccentricity E ∈ Set.Icc (Real.sqrt 3 / 3) (Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1035_103544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_conditions_system_no_solution_conditions_l1035_103579

/-- Geometric sequence with first term 1 and common ratio q -/
def geometric_sequence (q : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => q * geometric_sequence q n

/-- The system of linear equations -/
def system_has_solution (q : ℝ) : Prop :=
  ∃ x y : ℝ, (geometric_sequence q 0) * x + (geometric_sequence q 2) * y = 3 ∧
             (geometric_sequence q 1) * x + (geometric_sequence q 3) * y = -2

theorem system_solution_conditions (q : ℝ) :
  (system_has_solution q ∧ ∀ x' y' : ℝ, 
    (geometric_sequence q 0) * x' + (geometric_sequence q 2) * y' = 3 ∧
    (geometric_sequence q 1) * x' + (geometric_sequence q 3) * y' = -2 →
    ∃ t : ℝ, x' = t * x ∧ y' = t * y) ↔ q = -2/3 :=
by sorry

theorem system_no_solution_conditions (q : ℝ) :
  ¬(system_has_solution q) ↔ q ≠ -2/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_conditions_system_no_solution_conditions_l1035_103579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_for_special_angle_l1035_103537

theorem sin_minus_cos_for_special_angle (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_for_special_angle_l1035_103537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_digits_same_remainder_l1035_103538

theorem same_digits_same_remainder (a b : ℕ) : 
  (1000 ≤ a) ∧ (a < 10000) ∧ (1000 ≤ b) ∧ (b < 10000) ∧ 
  (∃ m c d u : ℕ, m < 10 ∧ c < 10 ∧ d < 10 ∧ u < 10 ∧
    a = 1000 * m + 100 * c + 10 * d + u ∧
    b = 1000 * m + 100 * c + 10 * d + u ∨
    b = 1000 * m + 100 * c + 10 * u + d ∨
    b = 1000 * m + 100 * d + 10 * c + u ∨
    b = 1000 * m + 100 * d + 10 * u + c ∨
    b = 1000 * m + 100 * u + 10 * c + d ∨
    b = 1000 * m + 100 * u + 10 * d + c ∨
    b = 1000 * c + 100 * m + 10 * d + u ∨
    b = 1000 * c + 100 * m + 10 * u + d ∨
    b = 1000 * c + 100 * d + 10 * m + u ∨
    b = 1000 * c + 100 * d + 10 * u + m ∨
    b = 1000 * c + 100 * u + 10 * m + d ∨
    b = 1000 * c + 100 * u + 10 * d + m ∨
    b = 1000 * d + 100 * m + 10 * c + u ∨
    b = 1000 * d + 100 * m + 10 * u + c ∨
    b = 1000 * d + 100 * c + 10 * m + u ∨
    b = 1000 * d + 100 * c + 10 * u + m ∨
    b = 1000 * d + 100 * u + 10 * m + c ∨
    b = 1000 * d + 100 * u + 10 * c + m ∨
    b = 1000 * u + 100 * m + 10 * c + d ∨
    b = 1000 * u + 100 * m + 10 * d + c ∨
    b = 1000 * u + 100 * c + 10 * m + d ∨
    b = 1000 * u + 100 * c + 10 * d + m ∨
    b = 1000 * u + 100 * d + 10 * m + c ∨
    b = 1000 * u + 100 * d + 10 * c + m) →
  a % 9 = b % 9 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_digits_same_remainder_l1035_103538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_in_third_quadrant_l1035_103599

theorem cos_value_in_third_quadrant (θ : ℝ) 
  (h1 : Real.sin θ = -1/3) 
  (h2 : θ ∈ Set.Ioo π (3*π/2)) : 
  Real.cos θ = -2*Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_in_third_quadrant_l1035_103599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_a_b_l1035_103541

/-- τ(k) denotes the number of all positive divisors of k, including 1 and k -/
def tau (k : ℕ) : ℕ := (Nat.divisors k).card

/-- The main theorem -/
theorem equal_a_b (a b : ℕ+) : 
  (∀ n : ℕ+, tau (tau (a.val * n.val)) = tau (tau (b.val * n.val))) → a = b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_a_b_l1035_103541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l1035_103509

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a : ℝ), (x - a)^2 + y^2 = 4 ∧ a ∈ Set.Icc (-1 : ℝ) 1

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x = 0 ∨ 3*x + 4*y - 8 = 0

-- Main theorem
theorem circle_and_line_properties :
  ∃ (x_c y_c : ℝ),
    -- C's center is on x-axis
    circle_C x_c 0 ∧
    -- C passes through A(-1,0) and B(1,2)
    circle_C (-1) 0 ∧ circle_C 1 2 ∧
    -- l passes through P(0,2)
    line_l 0 2 ∧
    -- l intersects C at M and N
    ∃ (x_m y_m x_n y_n : ℝ),
      circle_C x_m y_m ∧ circle_C x_n y_n ∧
      line_l x_m y_m ∧ line_l x_n y_n ∧
      -- |MN| = 2√3
      (x_m - x_n)^2 + (y_m - y_n)^2 = 12 ∧
    -- Standard equation of C
    (∀ x y : ℝ, circle_C x y ↔ (x - 1)^2 + y^2 = 4) ∧
    -- Equation of l
    (∀ x y : ℝ, line_l x y ↔ (x = 0 ∨ 3*x + 4*y - 8 = 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l1035_103509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_transformation_l1035_103591

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := Real.sin x

theorem f_transformation (x : ℝ) : f x = g ((x + 3 * Real.pi / 4) / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_transformation_l1035_103591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1035_103570

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/2)^x + (1/3)^x

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (∀ x ≤ 1, f x - m ≥ 0) ↔ m ≤ 5/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1035_103570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_problem_l1035_103515

-- Define the set M
def M : Set ℕ := {0, 1}

-- Define set A
def A : Set (ℕ × ℕ) := {p | p.1 ∈ M ∧ p.2 ∈ M}

-- Define set B
def B : Set (ℕ × ℕ) := {p | p.2 = (1 : ℤ) - p.1}

-- Theorem statement
theorem sets_problem :
  (A = {(0,0), (0,1), (1,0), (1,1)}) ∧
  (A ∩ B = {(1,0), (0,1)}) ∧
  (Set.powerset (A ∩ B) = {∅, {(1,0)}, {(0,1)}, {(1,0), (0,1)}}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_problem_l1035_103515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1035_103506

noncomputable def point1 : ℝ × ℝ := (2, -3)
noncomputable def point2 : ℝ × ℝ := (-4, 6)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_between_points :
  distance point1 point2 = 3 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1035_103506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_term_coefficient_l1035_103596

/-- A fifth-degree polynomial with leading coefficient 1 -/
def fifth_degree_poly (a b c d e : ℝ) : ℝ → ℝ := 
  λ x ↦ x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- The polynomial satisfies f(n) = 8n for n = 1, 2, 3, 4, 5 -/
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → f (n : ℝ) = 8 * n

theorem linear_term_coefficient 
  (a b c d e : ℝ) 
  (h : satisfies_condition (fifth_degree_poly a b c d e)) :
  d = 282 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_term_coefficient_l1035_103596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_f_inequality_solution_l1035_103540

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then (2 : ℝ)^(-2*x) else 2*x + 2

-- Theorem for f[f(-2)] = 34
theorem f_composition : f (f (-2)) = 34 := by sorry

-- Define the solution set
def solution_set : Set ℝ := {x | x ≤ -1 ∨ x ≥ 0}

-- Theorem for the solution set of f(x) ≥ 2
theorem f_inequality_solution : {x : ℝ | f x ≥ 2} = solution_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_f_inequality_solution_l1035_103540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_is_integer_l1035_103581

theorem fraction_is_integer (n : ℤ) : 
  n ≠ -2 → (∃ (k : ℤ), (n^3 + 8) = k * (n^2 - 4)) ↔ n ∈ ({0, 1, 3, 4, 6} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_is_integer_l1035_103581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_zero_l1035_103588

open Real

noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ := λ i j =>
  match i, j with
  | 0, 0 => cos (π / 4)
  | 0, 1 => cos (π / 2)
  | 0, 2 => cos (3 * π / 4)
  | 1, 0 => cos π
  | 1, 1 => cos (5 * π / 4)
  | 1, 2 => cos (3 * π / 2)
  | 2, 0 => cos (7 * π / 4)
  | 2, 1 => cos (2 * π)
  | 2, 2 => cos (9 * π / 4)

-- Theorem statement
theorem det_A_eq_zero : Matrix.det A = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_zero_l1035_103588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l1035_103520

/-- Two perpendicular lines with direction vectors (5, -2) and (b, 6) imply b = 12/5 -/
theorem perpendicular_lines_b_value (b : ℝ) :
  let v1 : Fin 2 → ℝ := ![5, -2]
  let v2 : Fin 2 → ℝ := ![b, 6]
  (v1 • v2 = 0) →
  b = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l1035_103520
