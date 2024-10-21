import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1059_105934

def arithmetic_sequence (a₁ : ℤ) (d : ℕ+) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℕ+) (n : ℕ) : ℤ :=
  n * a₁ + n * (n - 1) * d / 2

theorem arithmetic_sequence_problem :
  let a₁ : ℤ := -2
  ∀ d : ℕ+,
    (arithmetic_sequence a₁ d 5 = 30 →
      ∀ n : ℕ, arithmetic_sequence a₁ d n = 8 * n - 10) ∧
    (∃! L : List (ℕ+ × ℕ),
      L.length = 3 ∧
      (∀ p : ℕ+ × ℕ, p ∈ L → sum_arithmetic_sequence a₁ p.1 p.2 = 10) ∧
      L = [(⟨14, by norm_num⟩, 2), (⟨3, by norm_num⟩, 4), (⟨2, by norm_num⟩, 5)]) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1059_105934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_N_coordinates_l1059_105994

structure Point where
  x : ℝ
  y : ℝ

noncomputable def LineSegment (A B : Point) : ℝ :=
  Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

def ParallelToYAxis (A B : Point) : Prop :=
  A.x = B.x

theorem point_N_coordinates (M N : Point) :
  LineSegment M N = 4 →
  ParallelToYAxis M N →
  M.x = -1 →
  M.y = 2 →
  (N.x = -1 ∧ N.y = -2) ∨ (N.x = -1 ∧ N.y = 6) := by
  sorry

#check point_N_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_N_coordinates_l1059_105994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l1059_105948

noncomputable def q (x : ℝ) : ℝ := (8/3)*x^3 - (16/3)*x^2 - (40/3)*x + 16

theorem q_satisfies_conditions :
  (∀ x, (x - 3) * (x - 1) * (x + 2) ∣ q x) ∧
  (∃ a b c d : ℝ, ∀ x, q x = a*x^3 + b*x^2 + c*x + d) ∧
  q 4 = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l1059_105948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axes_distance_l1059_105911

noncomputable def f (x : ℝ) := Real.sin (2/3 * x + Real.pi/2) + Real.sin (2/3 * x)

theorem symmetry_axes_distance :
  let period := 2 * Real.pi / (2/3)
  (period / 2 : ℝ) = 3 * Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axes_distance_l1059_105911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_cross_product_equality_l1059_105902

open Real

/-- Definition of vector a -/
def a : Real × Real × Real := (2, -1, 1)

/-- Definition of vector b -/
def b : Real × Real × Real := (-1, 3, 0)

/-- Definition of vector v -/
def v : Real × Real × Real := (1, 2, 1)

/-- Function to compute cross product of two 3D vectors -/
def cross (u v : Real × Real × Real) : Real × Real × Real :=
  let (u₁, u₂, u₃) := u
  let (v₁, v₂, v₃) := v
  (u₂ * v₃ - u₃ * v₂, u₃ * v₁ - u₁ * v₃, u₁ * v₂ - u₂ * v₁)

/-- Theorem stating the equality of cross products -/
theorem vector_cross_product_equality :
  (cross v a = cross b a) ∧ (cross v b = cross a b) :=
by
  apply And.intro
  · -- Proof for (cross v a = cross b a)
    sorry
  · -- Proof for (cross v b = cross a b)
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_cross_product_equality_l1059_105902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_state_B_selection_percentage_l1059_105982

theorem state_B_selection_percentage
  (total_candidates : ℕ)
  (state_A_percentage : ℚ)
  (additional_selected_B : ℕ)
  (h1 : total_candidates = 8000)
  (h2 : state_A_percentage = 6 / 100)
  (h3 : additional_selected_B = 80) :
  (↑((state_A_percentage * ↑total_candidates).floor + additional_selected_B) / ↑total_candidates : ℚ) = 7 / 100 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_state_B_selection_percentage_l1059_105982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_congruent_squares_on_7x7_grid_l1059_105905

/-- Represents a square on a grid --/
structure Square where
  size : ℕ
  is_diagonal : Bool

/-- Calculates the number of squares of a given size on a grid --/
def count_squares (grid_size : ℕ) (square : Square) : ℕ :=
  if square.is_diagonal then
    if square.size = 1 then (grid_size - 1)^2
    else if square.size = 2 then (grid_size - 2)^2
    else 0
  else
    (grid_size - square.size + 1)^2

/-- Theorem: The number of non-congruent squares on a 7x7 grid is 200 --/
theorem non_congruent_squares_on_7x7_grid :
  (List.sum (List.map (λ i => count_squares 7 ⟨i, false⟩) [1, 2, 3, 4, 5, 6])) +
  (count_squares 7 ⟨1, true⟩ + count_squares 7 ⟨2, true⟩) = 200 := by
  sorry

#eval (List.sum (List.map (λ i => count_squares 7 ⟨i, false⟩) [1, 2, 3, 4, 5, 6])) +
      (count_squares 7 ⟨1, true⟩ + count_squares 7 ⟨2, true⟩)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_congruent_squares_on_7x7_grid_l1059_105905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_two_copresidents_l1059_105923

-- Define the number of students in each club
def students_club1 : ℕ := 6
def students_club2 : ℕ := 9
def students_club3 : ℕ := 10

-- Define the number of co-presidents in each club
def copresidents : ℕ := 3

-- Define the number of students to be selected
def selected : ℕ := 3

-- Define the probability calculation function
noncomputable def probability : ℚ :=
  (1 : ℚ) / 3 * (
    (Nat.choose copresidents 2 * Nat.choose (students_club1 - copresidents) 1) / Nat.choose students_club1 selected +
    (Nat.choose copresidents 2 * Nat.choose (students_club2 - copresidents) 1) / Nat.choose students_club2 selected +
    (Nat.choose copresidents 2 * Nat.choose (students_club3 - copresidents) 1) / Nat.choose students_club3 selected
  )

-- Theorem statement
theorem probability_of_two_copresidents : 
  ∃ ε > 0, |probability - 0.27977| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_two_copresidents_l1059_105923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1059_105968

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 1 else Real.cos x

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1059_105968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_division_example_l1059_105961

/-- Represents a number in base 5 --/
structure Base5 where
  value : ℕ

/-- Converts a base 5 number to a natural number --/
def to_nat (n : Base5) : ℕ := sorry

/-- Converts a natural number to a base 5 number --/
def from_nat (n : ℕ) : Base5 := ⟨n⟩

/-- Division operation for base 5 numbers --/
def base5_div (a b : Base5) : Base5 := sorry

/-- Theorem: 1324₅ ÷ 12₅ = 111₅ in base 5 --/
theorem base5_division_example : 
  base5_div (from_nat 1324) (from_nat 12) = from_nat 111 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_division_example_l1059_105961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_fifth_term_sum_l1059_105964

def geometricSequence (a₀ : ℝ) (r : ℝ) : ℕ → ℝ := fun n ↦ a₀ * r^n

theorem fourth_fifth_term_sum :
  let seq := geometricSequence 4096 (1/4)
  seq 3 + seq 4 = 80 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_fifth_term_sum_l1059_105964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_minus_pi_4_l1059_105967

theorem tan_x_minus_pi_4 (x : ℝ) (h1 : x ∈ Set.Ioo 0 π) (h2 : Real.cos (2 * x - π / 2) = Real.sin x ^ 2) :
  Real.tan (x - π / 4) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_minus_pi_4_l1059_105967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_customers_sitting_alone_l1059_105993

/-- Represents a restaurant with tables and customers -/
structure Restaurant :=
  (tables : ℕ)
  (initial_customers : ℕ)
  (customers_left : ℕ)
  (group_sizes : List ℕ)

/-- Counts the number of customers sitting alone -/
def count_alone (r : Restaurant) : ℕ :=
  r.group_sizes.filter (· = 1) |>.length

/-- Theorem stating the number of customers sitting alone in the given scenario -/
theorem customers_sitting_alone (r : Restaurant) 
  (h1 : r.tables = 6)
  (h2 : r.initial_customers = 14)
  (h3 : r.customers_left = 5)
  (h4 : r.group_sizes = [3, 1, 2, 4]) :
  count_alone r = 1 := by
  sorry

#check customers_sitting_alone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_customers_sitting_alone_l1059_105993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coefficient_and_odd_sum_l1059_105916

def f (m n : ℕ) (x : ℝ) : ℝ := (1 + x)^m + (1 + x)^n

/-- Coefficient of x^2 in the expansion of f(x) -/
noncomputable def coefficient_x_squared (m n : ℕ) : ℝ :=
  (m * (m - 1) / 2 : ℝ) + 2 * n * (n - 1)

/-- Sum of coefficients of odd powers of x in the expansion of f(x) -/
noncomputable def sum_odd_coefficients (m n : ℕ) : ℝ :=
  ((2^m + 3^n) - (-1)) / 2

theorem min_coefficient_and_odd_sum :
  (∃ (m n : ℕ), 
    coefficient_x_squared m n = 22 ∧ 
    (∀ (m' n' : ℕ), coefficient_x_squared m' n' ≥ 22) ∧
    sum_odd_coefficients m n = 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coefficient_and_odd_sum_l1059_105916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_minus_dA_star_l1059_105957

open Matrix

theorem det_A_minus_dA_star {n : Type*} [Fintype n] [DecidableEq n]
  (A : Matrix n n ℝ) (d : ℝ) :
  det A = d ∧ d ≠ 0 ∧ det (A + d • Aᵀ) = 0 →
  det (A - d • Aᵀ) = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_minus_dA_star_l1059_105957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_three_element_set_l1059_105979

theorem proper_subsets_of_three_element_set :
  let S : Finset Int := {-1, 0, 1}
  (Finset.card (Finset.powerset S \ {S})) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_three_element_set_l1059_105979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l1059_105975

theorem triangle_area_proof (a b c : ℝ) (h_perimeter : a + b + c = 2 * Real.sqrt 2 + Real.sqrt 5)
  (h_proportions : ∃ (k : ℝ), a = k * (Real.sqrt 2 - 1) ∧ b = k * Real.sqrt 5 ∧ c = k * (Real.sqrt 2 + 1)) :
  Real.sqrt (1/4 * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2)) = Real.sqrt 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l1059_105975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_x_value_l1059_105915

/-- Represents a systematic sample from a class -/
structure SystematicSample where
  total_students : Nat
  sample_size : Nat
  seat_numbers : Fin 4 → Nat

/-- Checks if a given systematic sample is valid -/
def is_valid_sample (s : SystematicSample) : Prop :=
  s.total_students = 52 ∧
  s.sample_size = 4 ∧
  s.seat_numbers 0 = 6 ∧
  s.seat_numbers 2 = 30 ∧
  s.seat_numbers 3 = 42

/-- The theorem stating that X must be 18 in a valid systematic sample -/
theorem systematic_sample_x_value (s : SystematicSample) 
  (h : is_valid_sample s) : s.seat_numbers 1 = 18 := by
  sorry

#check systematic_sample_x_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_x_value_l1059_105915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_m_for_odd_decreasing_function_l1059_105984

/-- An odd, decreasing function on the real numbers. -/
def OddDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x y, x < y → f x > f y)

/-- The theorem statement -/
theorem existence_of_m_for_odd_decreasing_function 
  (f : ℝ → ℝ) (h : OddDecreasingFunction f) :
  ∃ m : ℝ, m < 1 ∧ 
    ∀ θ : ℝ, θ ∈ Set.Icc (-Real.pi/3) (Real.pi/2) → 
      f (4*m - 2*m*(Real.cos θ)) + f (2*(Real.cos θ)^2 - 4) > f 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_m_for_odd_decreasing_function_l1059_105984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_property_l1059_105926

noncomputable section

/-- A cubic function with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + b * x + 1

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + b

/-- Predicate to check if three numbers form an arithmetic or geometric progression -/
def formsProgression (x y z : ℝ) : Prop :=
  (y - x = z - y) ∨ (y^2 = x*z)

theorem cubic_function_property (a b : ℝ) (x₁ x₂ : ℝ) :
  a > 0 → b > 0 →
  x₁ ≠ x₂ →
  f' a b x₁ = 0 →
  f' a b x₂ = 0 →
  formsProgression x₁ x₂ 2 ∨ formsProgression x₁ 2 x₂ ∨ formsProgression x₂ x₁ 2 ∨ formsProgression x₂ 2 x₁ →
  a + b = 13/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_property_l1059_105926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_empty_l1059_105909

-- Define the universal set I and sets P and Q
variable {α : Type*} (I P Q : Set α)

-- State the conditions
variable (h1 : P ⊂ Q)
variable (h2 : Q ⊂ I)
variable (h3 : P.Nonempty)
variable (h4 : Q.Nonempty)

-- Define the theorem
theorem intersection_complement_empty :
  P ∩ (I \ Q) = ∅ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_empty_l1059_105909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_equidistant_l1059_105976

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
noncomputable def distancePointLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- The theorem to be proved -/
theorem unique_line_equidistant :
  ∃! l : Line,
    (distancePointLine ⟨0, 0⟩ l = 1) ∧
    (distancePointLine ⟨-4, -3⟩ l = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_equidistant_l1059_105976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_on_curve_l1059_105943

/-- The curve on which the point (x,y) lies -/
def curve (x y : ℝ) : Prop := 3 * x^2 + 2 * Real.sqrt 3 * x * y + y^2 = 1

/-- The distance of a point (x,y) from the origin -/
noncomputable def distance (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- Theorem stating that the minimum distance from the origin to any point on the curve is 1/2 -/
theorem min_distance_on_curve : 
  ∀ x y : ℝ, curve x y → distance x y ≥ (1/2 : ℝ) ∧ ∃ x₀ y₀ : ℝ, curve x₀ y₀ ∧ distance x₀ y₀ = (1/2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_on_curve_l1059_105943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_formula_l1059_105928

/-- An arithmetic sequence is a sequence where the difference between any term and its preceding term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- A second-order arithmetic sequence is a sequence where the differences between successive terms form an arithmetic sequence. -/
def is_second_order_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence (λ n ↦ a (n + 1) - a n)

/-- The specific second-order arithmetic sequence 1, 3, 7, 13, 21, ... -/
def special_sequence : ℕ → ℝ
| 0 => 1  -- Adding case for 0
| 1 => 1
| 2 => 3
| 3 => 7
| 4 => 13
| 5 => 21
| n + 6 => special_sequence (n + 5) + 2 * (n + 5)

theorem special_sequence_formula :
  is_second_order_arithmetic_sequence special_sequence ∧
  ∀ n : ℕ, special_sequence n = 1 + n * (n - 1) :=
by sorry

#check special_sequence_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_formula_l1059_105928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_arithmetic_sum_1_to_30_l1059_105942

/-- Represents a number in base 8 -/
structure OctalNumber where
  value : Nat

/-- Converts an octal number to decimal -/
def octal_to_decimal (n : OctalNumber) : Nat :=
  sorry

/-- Converts a decimal number to octal -/
def decimal_to_octal (n : Nat) : OctalNumber :=
  sorry

/-- Calculates the sum of an arithmetic series in base 8 -/
def octal_arithmetic_sum (first last : OctalNumber) : OctalNumber :=
  sorry

theorem octal_arithmetic_sum_1_to_30 :
  octal_arithmetic_sum (OctalNumber.mk 1) (OctalNumber.mk 30) = OctalNumber.mk 454 :=
by sorry

/-- Instance for OfNat OctalNumber -/
instance : OfNat OctalNumber n where
  ofNat := OctalNumber.mk n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_arithmetic_sum_1_to_30_l1059_105942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1059_105970

-- Define the constants
noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

-- State the theorem
theorem order_of_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1059_105970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotated_triangle_theorem_l1059_105932

/-- Helper function to calculate the volume of the rotated triangle -/
noncomputable def volume_of_rotated_triangle (R α AB : ℝ) : ℝ := 
  (2/3) * Real.pi * R^3 * Real.sin (2*α) * Real.sin (4*α)

/-- Given a semicircle with diameter AB of length 2R, and a chord CD parallel to AB,
    where the inscribed angle subtended by arc AC is α (with AC < AD),
    the volume of the solid formed by rotating triangle ACD around diameter AB
    is (2/3) * π * R^3 * sin(2α) * sin(4α) -/
theorem volume_of_rotated_triangle_theorem (R α : ℝ) (h_α : 0 < α ∧ α < Real.pi/2) :
  let AB : ℝ := 2 * R
  let volume := volume_of_rotated_triangle R α AB
  volume = (2/3) * Real.pi * R^3 * Real.sin (2*α) * Real.sin (4*α)
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotated_triangle_theorem_l1059_105932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_five_l1059_105960

def A : Finset Int := {-1, 5, 1}

theorem subsets_containing_five :
  (A.powerset.filter (fun s => 5 ∈ s)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_five_l1059_105960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l1059_105987

/-- Given an angle α and a point P on its terminal side, prove the value of m. -/
theorem find_m (α : ℝ) (m : ℝ) : 
  (∃ P : ℝ × ℝ, P = (-8 * m, -3)) →  -- Point P on terminal side of α
  Real.cos α = -4/5 →                -- Given cosine value
  m = 1/2 := by                      -- Conclusion to prove
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l1059_105987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_difference_l1059_105958

-- Define the radii of the two circles
noncomputable def r1 : ℝ := 25
noncomputable def d2 : ℝ := 15

-- Define the area difference function
noncomputable def area_difference (r1 d2 : ℝ) : ℝ :=
  Real.pi * r1^2 - Real.pi * (d2/2)^2

-- State the theorem
theorem circle_area_difference :
  area_difference r1 d2 = 568.75 * Real.pi := by
  -- Expand the definition of area_difference
  unfold area_difference
  -- Simplify the expression
  simp [r1, d2]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_difference_l1059_105958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_sum_l1059_105959

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 3 * x^4)

theorem f_range_and_sum :
  ∃ (a b : ℝ), Set.range f = Set.Ioc a b ∧ a + b = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_sum_l1059_105959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l1059_105986

noncomputable def f (x : ℝ) := Real.rpow x (3⁻¹ : ℝ) * Real.rpow x (3⁻¹ : ℝ) * Real.rpow x (3⁻¹ : ℝ)
def g (x : ℝ) := x

theorem f_equiv_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l1059_105986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_sum_pi_half_l1059_105914

theorem arctan_sum_pi_half (m n : ℕ+) : 
  (m = 3 ∧ n = 4) ∨ (m = 4 ∧ n = 3) ↔ 
  Real.arctan (1 / 3) + Real.arctan (1 / 4) + Real.arctan (1 / m.val) + Real.arctan (1 / n.val) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_sum_pi_half_l1059_105914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_coefficient_in_expansion_l1059_105981

/-- Given that the sum of the binomial coefficients of the last three terms 
    in the expansion of (1-3x)^n is equal to 121, 
    the term with the smallest coefficient in the expansion is C_{15}^{11}(-3)^11*x^11 -/
theorem smallest_coefficient_in_expansion (n : ℕ) : 
  (Nat.choose n (n-2) + Nat.choose n (n-1) + Nat.choose n n = 121) → 
  (∃ k : ℕ, ∀ j : ℕ, j ≤ n → 
    |Int.ofNat (Nat.choose n k) * (-3 : ℤ)^k| ≤ |Int.ofNat (Nat.choose n j) * (-3 : ℤ)^j| ∧
    k = 11 ∧ n = 15) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_coefficient_in_expansion_l1059_105981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_FCH_cube_exists_angle_FCH_45_angle_FCH_range_l1059_105954

-- Define a rectangular parallelepiped
structure RectParallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the angle FCH
noncomputable def angle_FCH (p : RectParallelepiped) : ℝ :=
  Real.arccos ((- p.a^2 + p.b^2 - p.c^2) / (p.a^2 + p.b^2 + p.c^2))

-- Theorem for part (a)
theorem angle_FCH_cube (p : RectParallelepiped) (h_cube : p.a = p.b ∧ p.b = p.c) :
  angle_FCH p = Real.arccos (-1/3) := by sorry

-- Theorem for part (b)
theorem exists_angle_FCH_45 :
  ∃ p : RectParallelepiped, angle_FCH p = π/4 := by sorry

-- Theorem for part (c)
theorem angle_FCH_range (p : RectParallelepiped) :
  0 ≤ angle_FCH p ∧ angle_FCH p ≤ π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_FCH_cube_exists_angle_FCH_45_angle_FCH_range_l1059_105954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_lines_l1059_105955

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point A
def point_A : ℝ × ℝ := (-1, -6)

-- Define a line passing through point A
def line_through_A (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m ∧ point_A.2 = k * point_A.1 + m

-- Define the condition for a circle to pass through the vertex of the parabola
def circle_through_vertex (P Q : ℝ × ℝ) : Prop :=
  (P.1 + Q.1)^2 + (P.2 + Q.2)^2 = (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Main theorem
theorem parabola_intersection_lines :
  ∃! L : Set (ℝ → ℝ → Prop), 
    (∀ l ∈ L, ∃ k m, ∀ x y, l x y ↔ line_through_A k m x y) ∧ 
    (∀ l ∈ L, ∃ P Q : ℝ × ℝ, 
      parabola P.1 P.2 ∧ parabola Q.1 Q.2 ∧
      l P.1 P.2 ∧ l Q.1 Q.2 ∧
      circle_through_vertex P Q) ∧
    (L = {λ x y => y = 6*x} ∪ {λ x y => 6*x - 5*y - 24 = 0}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_lines_l1059_105955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_time_in_storm_l1059_105937

/-- Represents the velocity of an object in miles per minute -/
structure Velocity where
  x : ℝ
  y : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circular storm -/
structure Storm where
  center : Point
  radius : ℝ
  velocity : Velocity

/-- Represents a car -/
structure Car where
  position : Point
  velocity : Velocity

def car_storm_intersection_time (car : Car) (storm : Storm) : ℝ :=
  sorry

theorem average_time_in_storm (car : Car) (storm : Storm) : 
  let t₁ := car_storm_intersection_time car storm
  let t₂ := car_storm_intersection_time car storm
  (t₁ + t₂) / 2 = 198 :=
by
  sorry

def main : IO Unit :=
  let car : Car := {
    position := { x := 0, y := 0 },
    velocity := { x := 3/4, y := 0 }
  }
  let storm : Storm := {
    center := { x := 0, y := 130 },
    radius := 60,
    velocity := { x := -(1/3) * Real.sqrt 5, y := -(1/3) * Real.sqrt 5 }
  }
  IO.println s!"Average time in storm: 198 minutes"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_time_in_storm_l1059_105937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1059_105995

noncomputable def f (x : ℝ) : ℝ := Real.pi / 2 + Real.arcsin (3 * x)

def f_domain : Set ℝ := Set.Icc (-1/3) (1/3)

noncomputable def g (x : ℝ) : ℝ := -1/3 * Real.cos x

def g_domain : Set ℝ := Set.Icc 0 Real.pi

theorem inverse_function_theorem (x : ℝ) (hx : x ∈ f_domain) :
  g (f x) = x ∧ f (g x) = x ∧ Set.range f = g_domain := by
  sorry

#check inverse_function_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1059_105995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l1059_105903

theorem cone_volume_ratio : 
  ∃ (h_A r_A h_B r_B V_A V_B : ℝ),
    h_A = 30 ∧
    r_A = 15 ∧
    h_B = r_A ∧
    r_B = 2 * h_A ∧
    V_A = (1/3) * Real.pi * r_A^2 * h_A ∧
    V_B = (1/3) * Real.pi * r_B^2 * h_B ∧
    V_A / V_B = 1/24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l1059_105903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_total_points_l1059_105988

theorem test_total_points : 110 = (
  let total_problems : ℕ := 30
  let computation_problems : ℕ := 20
  let word_problems : ℕ := total_problems - computation_problems
  let computation_points : ℕ := 3
  let word_points : ℕ := 5
  computation_problems * computation_points + word_problems * word_points
) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_total_points_l1059_105988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_sufficient_not_necessary_for_g_increasing_l1059_105977

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x
def g (a : ℝ) (x : ℝ) : ℝ := (2-a)*x^3

-- State the theorem
theorem f_decreasing_sufficient_not_necessary_for_g_increasing
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  (∀ x y : ℝ, x < y → g a x < g a y) ∧
  ¬(∀ x y : ℝ, x < y → g a x < g a y →
    ∀ x y : ℝ, x < y → f a x > f a y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_sufficient_not_necessary_for_g_increasing_l1059_105977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_value_l1059_105953

/-- A spatial quadrilateral formed by folding a unit square along its diagonal -/
structure FoldedSquare where
  /-- The side length of the original square -/
  side_length : ℝ
  /-- Assertion that the side length is 1 -/
  side_length_is_one : side_length = 1
  /-- The two parts of the folded square lie on perpendicular planes -/
  perpendicular_planes : True

/-- The radius of the inscribed sphere in the folded square -/
noncomputable def inscribed_sphere_radius (fs : FoldedSquare) : ℝ :=
  Real.sqrt 2 - Real.sqrt 6 / 2

/-- Theorem stating that the radius of the inscribed sphere is √2 - √6/2 -/
theorem inscribed_sphere_radius_value (fs : FoldedSquare) :
  inscribed_sphere_radius fs = Real.sqrt 2 - Real.sqrt 6 / 2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_value_l1059_105953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1059_105951

-- Define the parabola C: x^2 = 4y
def C (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line l: y = kx - 1
def l (k x y : ℝ) : Prop := y = k*x - 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the condition for F being outside the circle with diameter AB
def F_outside_circle (x1 y1 x2 y2 : ℝ) : Prop :=
  x1*x2 + y1*y2 - (y1 + y2) + 1 > 0

theorem parabola_line_intersection 
  (x1 y1 x2 y2 k : ℝ) 
  (h1 : C x1 y1) 
  (h2 : C x2 y2) 
  (h3 : l k x1 y1) 
  (h4 : l k x2 y2) 
  (h5 : x1 ≠ x2) :
  (distance x1 y1 x2 y2 = 4 * Real.sqrt 15 → k = 2 ∨ k = -2) ∧
  (F_outside_circle x1 y1 x2 y2 → k > 1 ∧ k < Real.sqrt 2 ∨ k < -1 ∧ k > -Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1059_105951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l1059_105944

/-- Predicate to determine if a given equation represents an ellipse -/
def IsEllipse (x y : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ x^2/a^2 + y^2/b^2 = 1

/-- Given an equation representing an ellipse, this theorem proves the range of k. -/
theorem ellipse_k_range :
  ∀ k : ℝ,
  (∀ x y : ℝ, (x^2 / (2 - k) + y^2 / (3 + k) = 1) → IsEllipse x y) →
  -3 < k ∧ k < 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l1059_105944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_and_min_modulus_l1059_105940

/-- The equation of the locus of the centroid and its minimum modulus -/
theorem centroid_locus_and_min_modulus 
  (θ : ℝ) (S : ℝ) (h₁ : 0 < θ) (h₂ : θ < π / 2) (h₃ : S > 0) :
  ∃ (Z : ℂ),
    let x := Z.re
    let y := Z.im
    (9 * x^2 / Real.cos θ^2 - 9 * y^2 / Real.sin θ^2 = 8 * S / Real.sin (2 * θ)) ∧
    (Complex.abs Z ≥ 2 / 3 * Real.sqrt (S * Real.tan θ⁻¹)) ∧
    (∃ (Z₀ : ℂ), Complex.abs Z₀ = 2 / 3 * Real.sqrt (S * Real.tan θ⁻¹)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_and_min_modulus_l1059_105940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_mpg_is_48_l1059_105929

/-- Represents the fuel efficiency and tank capacity of a car -/
structure CarFuelData where
  highway_miles_per_tankful : ℝ
  city_miles_per_tankful : ℝ
  city_mpg_difference : ℝ

/-- Calculates the city miles per gallon given car fuel data -/
noncomputable def city_mpg (data : CarFuelData) : ℝ :=
  let highway_mpg := data.highway_miles_per_tankful / (data.city_miles_per_tankful / (data.highway_miles_per_tankful / (data.highway_miles_per_tankful - data.city_mpg_difference)))
  highway_mpg - data.city_mpg_difference

/-- Theorem stating that for the given car fuel data, the city mpg is 48 -/
theorem car_city_mpg_is_48 (data : CarFuelData) 
    (h1 : data.highway_miles_per_tankful = 462)
    (h2 : data.city_miles_per_tankful = 336)
    (h3 : data.city_mpg_difference = 18) :
  city_mpg data = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_mpg_is_48_l1059_105929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_numbers_product_l1059_105962

theorem two_numbers_product (x y : ℝ) : 
  (x - y) / (x + y) = 1/5 ∧ (x * y) / (x + y) = 4 → x * y = 200 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_numbers_product_l1059_105962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_root_in_interval_l1059_105933

-- Define the function f(x) = log₂(x) + x - 5
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + x - 5

-- Theorem statement
theorem exists_root_in_interval :
  ∃ x : ℝ, 3 < x ∧ x < 4 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_root_in_interval_l1059_105933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ngon_labeling_divisibility_l1059_105939

/-- 
Theorem: For a regular n-gon (n ≥ 4), the sum of labels on the sides and n-3 diagonals 
is divisible by n-2 if and only if n is even or n = 4k+1 for some integer k.
-/
theorem ngon_labeling_divisibility (n : ℕ) (h : n ≥ 4) :
  (∃ (diag_sum : ℕ), (n * (n + 1) / 2 + diag_sum) % (n - 2) = 0) ↔ 
  (∃ (k : ℕ), n = 2 * k ∨ n = 4 * k + 1) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ngon_labeling_divisibility_l1059_105939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whisky_replacement_theorem_l1059_105906

/-- The percentage of alcohol in the whisky that replaced the original whisky -/
noncomputable def replacing_whisky_alcohol_percentage (
  original_alcohol_percentage : ℝ
  ) (replaced_quantity : ℝ
  ) (new_mixture_alcohol_percentage : ℝ
  ) : ℝ :=
  ((new_mixture_alcohol_percentage - original_alcohol_percentage * (1 - replaced_quantity))
   / replaced_quantity)

/-- Theorem stating that given the conditions of the whisky replacement problem,
    the alcohol percentage of the replacing whisky is approximately 19% -/
theorem whisky_replacement_theorem :
  let original_alcohol_percentage : ℝ := 0.40
  let replaced_quantity : ℝ := 0.7619047619047619
  let new_mixture_alcohol_percentage : ℝ := 0.24
  ∃ ε > 0, ε < 0.01 ∧
  |replacing_whisky_alcohol_percentage original_alcohol_percentage replaced_quantity new_mixture_alcohol_percentage - 0.19| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_whisky_replacement_theorem_l1059_105906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_coeff_terms_l1059_105965

/-- The exponent of the binomial expansion (x^2 - 1/x)^n -/
def n : ℕ := 8

/-- The general term of the expansion (x^2 - 1/x)^n -/
def general_term (r : ℕ) : ℤ × ℤ := 
  ((-1)^r * (n.choose r), 16 - 3*r)

/-- Theorem stating the maximum and minimum coefficient terms in the expansion -/
theorem max_min_coeff_terms :
  (∃ r : ℕ, general_term r = (70, 4)) ∧
  (∃ r : ℕ, general_term r = (-56, 7)) ∧
  (∃ r : ℕ, general_term r = (-56, 1)) ∧
  (∀ r : ℕ, (general_term r).fst.natAbs ≤ 70) ∧
  (∀ r : ℕ, (general_term r).fst ≠ 0 → (general_term r).fst.natAbs ≥ 56) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_coeff_terms_l1059_105965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_angle_l1059_105919

noncomputable def Parabola (x y : ℝ) : Prop := y^2 = 3 * x

noncomputable def Focus : ℝ × ℝ := (3/4, 0)

noncomputable def DistanceFA (A : ℝ × ℝ) : Prop := 
  Real.sqrt ((A.1 - Focus.1)^2 + (A.2 - Focus.2)^2) = 3

noncomputable def AngleFA (A : ℝ × ℝ) : ℝ := 
  Real.arctan ((A.2 - Focus.2) / (A.1 - Focus.1))

theorem parabola_focus_angle {A : ℝ × ℝ} 
  (h1 : Parabola A.1 A.2) 
  (h2 : DistanceFA A) : 
  AngleFA A = π/3 ∨ AngleFA A = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_angle_l1059_105919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1059_105971

-- Problem 1
theorem problem_1 : -2^(0 : ℤ) + 4^(-1 : ℤ) * (-1)^(2009 : ℤ) * (-1/2 : ℚ)^(-2 : ℤ) = -17/16 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (x + 1)^2 - (x - 1)*(x + 2) = 3*x - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1059_105971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1059_105991

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x - π / 3)

theorem omega_range (ω : ℝ) :
  ω > 0 →
  (∀ x ∈ Set.Icc 0 π, f ω x ∈ Set.Icc (-Real.sqrt 3 / 2) 1) →
  ω ∈ Set.Icc (5 / 6) (5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1059_105991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_product_of_two_primes_above_30_l1059_105950

/-- Two distinct primes greater than 30 -/
def p1 : ℕ := 31
def p2 : ℕ := 37

/-- The product of p1 and p2 -/
def product : ℕ := p1 * p2

theorem least_product_of_two_primes_above_30 :
  (Nat.Prime p1 ∧ Nat.Prime p2 ∧ p1 > 30 ∧ p2 > 30 ∧ p1 ≠ p2) →
  (∀ q1 q2 : ℕ, Nat.Prime q1 → Nat.Prime q2 → q1 > 30 → q2 > 30 → q1 ≠ q2 → q1 * q2 ≥ product) →
  product = 1147 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_product_of_two_primes_above_30_l1059_105950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_to_53_l1059_105925

theorem sum_to_53 (S : Finset ℕ) (h1 : S.card = 53) (h2 : S.sum id ≤ 1990) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = 53 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_to_53_l1059_105925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_integer_satisfying_inequality_l1059_105998

theorem smallest_even_integer_satisfying_inequality :
  ∃ x : ℕ, 
    (∀ y : ℕ, Even y ∧ y < 3*y - 10 → x ≤ y) ∧
    Even x ∧
    x < 3*x - 10 ∧
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_integer_satisfying_inequality_l1059_105998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_position_1011_l1059_105900

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | BADC
  | DCBA
  | CBAD

-- Define the transformation function
def transform (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.CBAD
  | SquarePosition.CBAD => SquarePosition.ABCD

-- Define the theorem
theorem square_position_1011 :
  (Nat.iterate transform 1011 SquarePosition.ABCD) = SquarePosition.DCBA :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_position_1011_l1059_105900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_good_fourth_powers_l1059_105999

/-- A natural number is good if it can be represented as the sum of two coprime natural numbers,
    where one has an odd number of prime factors and the other has an even number of prime factors. -/
def IsGood (n : ℕ) : Prop :=
  ∃ (x y : ℕ), n = x + y ∧ Nat.Coprime x y ∧
    Odd (Finset.card (Nat.factors x).toFinset) ∧ Even (Finset.card (Nat.factors y).toFinset)

/-- There exists an infinite sequence of natural numbers such that the fourth power of each
    number in the sequence is good. -/
theorem infinitely_many_good_fourth_powers :
  ∃ (f : ℕ → ℕ), ∀ (k : ℕ), IsGood ((f k) ^ 4) ∧ ∀ (i j : ℕ), i < j → f i < f j :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_good_fourth_powers_l1059_105999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_values_l1059_105997

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / 2 + Real.log (Real.sqrt x)

-- Define the theorem
theorem range_of_t_values (a b ta tb : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : ta ≤ tb) 
  (h4 : ∀ x ∈ Set.Icc a b, f x ∈ Set.Icc ta tb) :
  ∃ t : ℝ, t ∈ Set.Ioo (1/2) ((1 + Real.exp 1) / (2 * Real.exp 1)) ∧
    ∃ x ∈ Set.Icc a b, f x = t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_values_l1059_105997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_mowing_time_l1059_105910

/-- Represents the mowing problem for Jerry --/
structure MowingProblem where
  total_acres : ℝ
  riding_mower_ratio : ℝ
  first_push_mower_ratio : ℝ
  riding_mower_speed : ℝ
  first_push_mower_speed : ℝ
  second_push_mower_speed : ℝ

/-- Calculates the total mowing time for Jerry --/
noncomputable def total_mowing_time (p : MowingProblem) : ℝ :=
  let riding_acres := p.riding_mower_ratio * p.total_acres
  let remaining_acres := p.total_acres - riding_acres
  let first_push_acres := p.first_push_mower_ratio * remaining_acres
  let second_push_acres := remaining_acres - first_push_acres
  
  let riding_time := riding_acres / p.riding_mower_speed
  let first_push_time := first_push_acres / p.first_push_mower_speed
  let second_push_time := second_push_acres / p.second_push_mower_speed
  
  riding_time + first_push_time + second_push_time

/-- Theorem stating that Jerry's total mowing time is approximately 13.69 hours --/
theorem jerry_mowing_time :
  let p : MowingProblem := {
    total_acres := 20,
    riding_mower_ratio := 3/5,
    first_push_mower_ratio := 1/3,
    riding_mower_speed := 2.5,
    first_push_mower_speed := 1.2,
    second_push_mower_speed := 0.8
  }
  abs (total_mowing_time p - 13.69) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_mowing_time_l1059_105910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_c_l1059_105945

theorem existence_of_c : ∃ c : ℝ, 
  (3 * (Int.floor c)^2 + 23 * (Int.floor c) - 75 = 0) ∧ 
  (4 * (c - Int.floor c)^2 - 19 * (c - Int.floor c) + 3 = 0) ∧ 
  (c = -11.84) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_c_l1059_105945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_is_one_l1059_105949

/-- Pentagon with specific side lengths -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  ab_length : dist A B = 1
  de_length : dist D E = 1

/-- Function to calculate the area of a pentagon -/
noncomputable def area (p : Pentagon) : ℝ := sorry

/-- The area of a pentagon with sides AB and DE equal to 1 is 1 -/
theorem pentagon_area_is_one (p : Pentagon) : area p = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_is_one_l1059_105949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_inscribed_triangle_l1059_105972

theorem max_sum_squared_inscribed_triangle (r : ℝ) (h : r > 0) :
  ∃ (A B C : EuclideanSpace ℝ (Fin 2)),
    (∀ P ∈ ({A, B, C} : Set (EuclideanSpace ℝ (Fin 2))), ‖P‖ = r) →
    ‖A - B‖ = 2 * r →
    A ≠ C ∧ B ≠ C →
    (‖A - C‖ + ‖B - C‖)^2 ≤ 8 * r^2 ∧
    ∃ C', (‖A - C'‖ + ‖B - C'‖)^2 = 8 * r^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_inscribed_triangle_l1059_105972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_example_l1059_105918

/-- The time (in seconds) for a train to cross an electric pole -/
noncomputable def trainCrossingTime (trainLength : ℝ) (trainSpeed : ℝ) : ℝ :=
  trainLength / (trainSpeed * 1000 / 3600)

/-- Theorem: A train of length 100 meters traveling at 144 km/hr takes 2.5 seconds to cross an electric pole -/
theorem train_crossing_time_example :
  trainCrossingTime 100 144 = 2.5 := by
  -- Unfold the definition of trainCrossingTime
  unfold trainCrossingTime
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_example_l1059_105918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1059_105974

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed_kmh : ℝ) (crossing_time_s : ℝ) (bridge_length_m : ℝ) :
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  bridge_length_m = 250.03 →
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time_s
  let train_length := total_distance - bridge_length_m
  ∃ ε > 0, |train_length - 124.97| < ε := by
  intro h1 h2 h3
  -- The proof steps would go here
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1059_105974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l1059_105983

noncomputable def f (x a m : ℝ) : ℝ := (1/2) * x^2 - (a + m) * x + a * Real.log x

theorem problem_1 (a m : ℝ) :
  (deriv (f · a m)) 1 = 0 → m = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l1059_105983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_and_minimum_l1059_105904

noncomputable section

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 / a)^2 + (p.2 / b)^2 = 1}

-- Define the curve C (first quadrant part of the ellipse)
def CurveC (a b : ℝ) : Set (ℝ × ℝ) :=
  {p ∈ Ellipse a b | p.1 > 0 ∧ p.2 > 0}

-- Define the tangent line at a point on the ellipse
def TangentLine (a b : ℝ) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | (q.2 - p.2) = -(a^2 * p.1) / (b^2 * p.2) * (q.1 - p.1)}

-- Define point M
def PointM (a b : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (b^2 / p.1, a^2 / p.2)

theorem ellipse_trajectory_and_minimum (a b : ℝ) 
  (h1 : a^2 - b^2 = 3) 
  (h2 : Real.sqrt 3 / a = Real.sqrt 3 / 2) :
  let M := {m : ℝ × ℝ | ∃ p ∈ CurveC a b, m = PointM a b p}
  (∀ m ∈ M, 1 / m.1^2 + 4 / m.2^2 = 1 ∧ m.1 > 1 ∧ m.2 > 2) ∧
  (∀ m ∈ M, ‖m‖ ≥ 3) ∧
  (∃ m ∈ M, ‖m‖ = 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_and_minimum_l1059_105904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_tangent_to_circle_l1059_105917

-- Define the line l1
def l1 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 16

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- State the theorem
theorem min_distance_tangent_to_circle :
  ∀ (xP yP xM yM : ℝ),
  l1 xP yP →
  C xM yM →
  (∀ (t : ℝ), C (xP + t * (xM - xP)) (yP + t * (yM - yP)) → t = 0 ∨ t = 1) →
  (∀ (x y : ℝ), l1 x y → distance x y xM yM ≥ distance xP yP xM yM) →
  distance xP yP xM yM = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_tangent_to_circle_l1059_105917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_volume_l1059_105908

-- Define the right square pyramid
structure RightSquarePyramid where
  baseEdgeLength : ℝ
  sideEdgeLength : ℝ

-- Define the sphere
structure Sphere where
  radius : ℝ

-- Define the volume of a sphere
noncomputable def sphereVolume (s : Sphere) : ℝ := (4 / 3) * Real.pi * s.radius ^ 3

-- Theorem statement
theorem pyramid_sphere_volume 
  (p : RightSquarePyramid) 
  (s : Sphere) 
  (h1 : p.baseEdgeLength = 4) 
  (h2 : p.sideEdgeLength = 2 * Real.sqrt 6) 
  (h3 : s.radius = 3) : 
  sphereVolume s = 36 * Real.pi := by
  sorry

#check pyramid_sphere_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_volume_l1059_105908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wholesale_price_calculation_l1059_105985

/-- Proves that the wholesale price of a machine is $90 given the retail price,
    discount percentage, and profit percentage. -/
theorem wholesale_price_calculation (retail_price : ℝ) (discount_percent : ℝ) (profit_percent : ℝ)
    (h1 : retail_price = 120)
    (h2 : discount_percent = 0.1)
    (h3 : profit_percent = 0.2) :
    (retail_price * (1 - discount_percent)) / (1 + profit_percent) = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wholesale_price_calculation_l1059_105985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l1059_105927

/-- Proves that a train of given length crossing a bridge of given length in a given time has a specific speed -/
theorem train_speed_proof (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 120)
  (h2 : bridge_length = 255)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l1059_105927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l1059_105912

/-- A trinomial is a perfect square if it can be expressed as (y + a)^2 for some constant a -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ y : ℝ, a * y^2 + b * y + c = (y + k)^2

/-- Given that y^2 + my + 9 is a perfect square trinomial, m must be either 6 or -6 -/
theorem perfect_square_trinomial_m_value (m : ℝ) :
  is_perfect_square_trinomial 1 m 9 → m = 6 ∨ m = -6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l1059_105912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_in_interval_l1059_105941

-- Define the function f(x) = (1/2)^x
noncomputable def f (x : ℝ) : ℝ := (1/2)^x

-- State the theorem
theorem min_value_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) (-1 : ℝ) ∧ 
  (∀ x ∈ Set.Icc (-2 : ℝ) (-1 : ℝ), f c ≤ f x) ∧ 
  f c = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_in_interval_l1059_105941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quasiperfect_is_odd_square_l1059_105969

-- Define the sum of divisors function
def sigma (n : ℕ) : ℕ := (Finset.sum (Nat.divisors n) id)

-- Define quasiperfect numbers
def is_quasiperfect (n : ℕ) : Prop := sigma n = 2 * n + 1

-- Theorem statement
theorem quasiperfect_is_odd_square (N : ℕ) (h : is_quasiperfect N) :
  ∃ m : ℕ, N = m^2 ∧ m % 2 = 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quasiperfect_is_odd_square_l1059_105969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_30_60_90_divisible_into_three_congruent_l1059_105913

/-- Predicate for a set of points forming a right triangle -/
def IsRightTriangle (T : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate for a triangle having a specific angle (in degrees) -/
def TriangleAngle (T : Set (ℝ × ℝ)) (angle : ℝ) : Prop := sorry

/-- Predicate for two triangles being congruent -/
def IsCongruentTriangle (T1 T2 : Set (ℝ × ℝ)) : Prop := sorry

/-- A right triangle with angles 30°, 60°, and 90° can be divided into three congruent triangles -/
theorem right_triangle_30_60_90_divisible_into_three_congruent :
  ∃ (T : Set (ℝ × ℝ)) (A B C : Set (ℝ × ℝ)),
    IsRightTriangle T ∧
    TriangleAngle T 30 ∧
    TriangleAngle T 60 ∧
    TriangleAngle T 90 ∧
    IsCongruentTriangle A B ∧
    IsCongruentTriangle B C ∧
    IsCongruentTriangle C A ∧
    A ∪ B ∪ C = T :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_30_60_90_divisible_into_three_congruent_l1059_105913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionArea_formula_l1059_105922

/-- Regular quadrilateral pyramid with square base -/
structure RegularQuadPyramid where
  a : ℝ  -- side length of the square base
  b : ℝ  -- length of the lateral edge
  a_pos : 0 < a
  b_pos : 0 < b

/-- The area of the cross-section of a regular quadrilateral pyramid -/
noncomputable def crossSectionArea (p : RegularQuadPyramid) : ℝ :=
  (p.a * p.b * Real.sqrt 2) / 4

/-- Theorem: The area of the cross-section of a regular quadrilateral pyramid
    by a plane passing through the diagonal of the base and parallel to a lateral edge
    is equal to (a * b * √2) / 4 -/
theorem crossSectionArea_formula (p : RegularQuadPyramid) :
  crossSectionArea p = (p.a * p.b * Real.sqrt 2) / 4 := by
  -- Unfold the definition of crossSectionArea
  unfold crossSectionArea
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionArea_formula_l1059_105922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_min_value_l1059_105978

theorem cos_min_value (x : ℝ) (h : x ∈ Set.Icc (π/6) ((2/3)*π)) :
  ∃ y, y = Real.cos (x - π/8) ∧ y ≥ (1/2) ∧ ∀ z, z = Real.cos (x - π/8) → y ≤ z :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_min_value_l1059_105978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_constant_sum_l1059_105956

/-- The ellipse C -/
noncomputable def C (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

/-- The fixed point Q -/
noncomputable def Q : ℝ × ℝ := (6 * Real.sqrt 5 / 5, 0)

/-- A point is on the ellipse C -/
def onEllipse (p : ℝ × ℝ) : Prop := C p.1 p.2

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem to be proved -/
theorem ellipse_chord_constant_sum :
  ∀ A B : ℝ × ℝ,
  onEllipse A → onEllipse B →
  (∃ t : ℝ, (1 - t) • A + t • Q = B) →
  1 / distance Q A ^ 2 + 1 / distance Q B ^ 2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_constant_sum_l1059_105956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_intersection_point_ratio_l1059_105907

def Parabola (x y : ℝ) : Prop := y^2 = 4*x

def Line (x y m : ℝ) : Prop := y = x - m

def DistanceToFocus (x : ℝ) : ℝ := x - 1

theorem parabola_intersection_theorem (m : ℝ) :
  (∃ A B : ℝ × ℝ,
    Parabola A.1 A.2 ∧ Parabola B.1 B.2 ∧
    Line A.1 A.2 m ∧ Line B.1 B.2 m ∧
    A.1 > 1 ∧ A.2 > 0 ∧ B.1 > 1 ∧ B.2 < 0 ∧
    DistanceToFocus A.1 + DistanceToFocus B.1 = 10) →
  m = 2 :=
by sorry

theorem intersection_point_ratio (m : ℝ) (A B P : ℝ × ℝ) :
  Parabola A.1 A.2 ∧ Parabola B.1 B.2 ∧
  Line A.1 A.2 m ∧ Line B.1 B.2 m ∧
  P.2 = 0 ∧
  (A.1 - P.1)^2 + (A.2 - P.2)^2 + (B.1 - P.1)^2 + (B.2 - P.2)^2 = 288 →
  ∃ l : ℝ, l = 2 ∧ A.1 - P.1 = l * (P.1 - B.1) ∧ A.2 - P.2 = l * (P.2 - B.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_intersection_point_ratio_l1059_105907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_sixty_cents_l1059_105996

-- Define coin types
inductive Coin
| Penny
| Nickel
| Dime
| Quarter

def coin_value : Coin → ℕ
| Coin.Penny => 1
| Coin.Nickel => 5
| Coin.Dime => 10
| Coin.Quarter => 25

def coin_count : Coin → ℕ
| Coin.Dime => 2
| _ => 1

def is_valid_selection (selection : List Coin) : Prop :=
  (selection.map coin_count).sum = 6

def selection_value (selection : List Coin) : ℕ :=
  (selection.map coin_value).sum

theorem impossible_sixty_cents :
  ¬∃ (selection : List Coin), is_valid_selection selection ∧ selection_value selection = 60 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_sixty_cents_l1059_105996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_plane_segments_l1059_105930

-- Define the ratio of line segments
variable (k : ℝ) 

-- Define the angles
noncomputable def x (k : ℝ) : ℝ := 2 * Real.arccos ((k + Real.sqrt (k^2 + 4)) / 4)

-- State the theorem
theorem parallel_plane_segments (k : ℝ) :
  (2 / (2 * Real.sqrt 3) < k ∧ k < 2 * Real.sqrt 3 / 3) ↔
  (∃ (a : ℝ), a > 0 ∧
    ∃ (x : ℝ), 0 < x ∧ x ≤ π/2 ∧
      Real.sin x = k * Real.sin ((2/3) * x) ∧
      x = 2 * Real.arccos ((k + Real.sqrt (k^2 + 4)) / 4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_plane_segments_l1059_105930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workday_meetings_percentage_l1059_105924

/-- The duration of Makarla's workday in hours -/
noncomputable def workday_hours : ℝ := 8

/-- The duration of the first meeting in minutes -/
noncomputable def first_meeting_duration : ℝ := 30

/-- The duration of the second meeting in minutes -/
noncomputable def second_meeting_duration : ℝ := 1.5 * first_meeting_duration

/-- The duration of the third meeting in minutes -/
noncomputable def third_meeting_duration : ℝ := first_meeting_duration + second_meeting_duration

/-- The total duration of all meetings in minutes -/
noncomputable def total_meeting_duration : ℝ := first_meeting_duration + second_meeting_duration + third_meeting_duration

/-- The percentage of the workday spent in meetings -/
noncomputable def meeting_percentage : ℝ := (total_meeting_duration / (workday_hours * 60)) * 100

theorem workday_meetings_percentage :
  meeting_percentage = 31.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workday_meetings_percentage_l1059_105924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_degrees_l1059_105992

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![cos θ, -sin θ],
    ![sin θ,  cos θ]]

noncomputable def angle : ℝ := 150 * π / 180

theorem rotation_150_degrees :
  rotation_matrix angle = ![![-Real.sqrt 3 / 2, -1 / 2],
                            ![1 / 2, -Real.sqrt 3 / 2]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_degrees_l1059_105992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_four_even_numbers_with_divisible_sums_l1059_105973

theorem exist_four_even_numbers_with_divisible_sums :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    Even a ∧ Even b ∧ Even c ∧ Even d ∧
    (a + b + c) % d = 0 ∧
    (a + b + d) % c = 0 ∧
    (a + c + d) % b = 0 ∧
    (b + c + d) % a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_four_even_numbers_with_divisible_sums_l1059_105973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_10_l1059_105952

open BigOperators Finset

def factors (n : ℕ) : Finset ℕ := (Finset.range n).filter (λ d => d > 0 ∧ n % d = 0)

def lessThan10 (s : Finset ℕ) : Finset ℕ := s.filter (λ x => x < 10)

theorem probability_factor_less_than_10 :
  let allFactors := factors 90
  let factorsLessThan10 := lessThan10 allFactors
  (factorsLessThan10.card : ℚ) / (allFactors.card : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_10_l1059_105952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1059_105946

noncomputable section

variable (a b : ℝ × ℝ)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt (dot_product v v)

noncomputable def angle_between (v w : ℝ × ℝ) : ℝ := Real.arccos (dot_product v w / (vector_length v * vector_length w))

theorem vector_problem (h1 : vector_length a = 1) 
                       (h2 : dot_product a b = 1/2) 
                       (h3 : dot_product (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) = 1/2) : 
  angle_between a b = π/4 ∧ 
  dot_product (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) / 
  (vector_length (a.1 + b.1, a.2 + b.2) * vector_length (a.1 - b.1, a.2 - b.2)) = Real.sqrt 5 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1059_105946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_m_value_l1059_105936

theorem quadratic_function_m_value (m : ℝ) : 
  (∃ (y : ℝ → ℝ), y = (λ x => (m + 1) * x^(m^2 + 3*m + 4)) ∧ 
   ∃ (a b c : ℝ), y = (λ x => a*x^2 + b*x + c) ∧ a ≠ 0) ∧ 
  m + 1 ≠ 0 → 
  m = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_m_value_l1059_105936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_sqrt_equation_l1059_105980

theorem max_sum_of_sqrt_equation (x y : ℤ) : 
  (31 * 38 * Real.sqrt (x : ℝ)) + Real.sqrt (y : ℝ) = Real.sqrt 2009 →
  x + y ≤ 1517 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_sqrt_equation_l1059_105980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_has_two_elements_l1059_105921

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (2 * x + 9) / x

-- Define the sequence of functions gₙ
noncomputable def g_n : ℕ → (ℝ → ℝ)
  | 0 => g
  | n + 1 => g ∘ g_n n

-- Define the set T
def T : Set ℝ := {x | ∃ n : ℕ, n > 0 ∧ g_n n x = x}

-- Theorem statement
theorem T_has_two_elements : ∃ (s : Finset ℝ), s.card = 2 ∧ ↑s = T := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_has_two_elements_l1059_105921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_area_l1059_105963

noncomputable def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

noncomputable def point_M (x₀ y₀ : ℝ) : ℝ := -2*y₀/(x₀ - 2)

noncomputable def point_N (x₀ y₀ : ℝ) : ℝ := -x₀/(y₀ - 1)

noncomputable def area_ABNM (x₀ y₀ : ℝ) : ℝ := 
  1/2 * (2 + point_N x₀ y₀) * (1 + point_M x₀ y₀)

theorem constant_area : 
  ∀ x₀ y₀ : ℝ, 
    ellipse x₀ y₀ → 
    third_quadrant x₀ y₀ → 
    area_ABNM x₀ y₀ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_area_l1059_105963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l1059_105901

/-- The slope of a line parallel to 3x - 6y = 12 -/
def slope_of_parallel_line : ℚ := 1 / 2

/-- Given a line with equation 3x - 6y = 12, prove that the slope of any parallel line is 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 12) → (slope_of_parallel_line = 1 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l1059_105901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_inequality_l1059_105990

/-- Sum of digits function -/
def sumOfDigits : ℕ → ℕ := sorry

/-- Property of sum of digits for addition -/
axiom sum_of_digits_add (a b : ℕ) : sumOfDigits (a + b) ≤ sumOfDigits a + sumOfDigits b

/-- Theorem: Sum of digits of k is not greater than 8 times the sum of digits of 8k -/
theorem sum_of_digits_inequality (k : ℕ) : sumOfDigits k ≤ 8 * sumOfDigits (8 * k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_inequality_l1059_105990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilution_problem_l1059_105947

/-- The amount of water needed to dilute an alcohol solution -/
noncomputable def water_needed (initial_volume : ℝ) (initial_concentration : ℝ) (target_concentration : ℝ) : ℝ :=
  (initial_volume * initial_concentration / target_concentration) - initial_volume

/-- Theorem: The amount of water needed to dilute 12 ounces of a 60% alcohol solution to a 40% alcohol solution is 6 ounces -/
theorem dilution_problem :
  water_needed 12 0.6 0.4 = 6 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval water_needed 12 0.6 0.4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilution_problem_l1059_105947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_workers_count_l1059_105935

/-- Represents the number of workers in a company -/
def total_workers : ℕ := sorry

/-- Represents the number of women workers in the company -/
def women_workers : ℕ := sorry

/-- Represents the fraction of workers without a retirement plan -/
def no_retirement_plan_fraction : ℚ := sorry

/-- Represents the fraction of workers without a retirement plan who are women -/
def women_no_plan_fraction : ℚ := sorry

/-- Represents the fraction of workers with a retirement plan who are men -/
def men_with_plan_fraction : ℚ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem company_workers_count 
  (h1 : no_retirement_plan_fraction = 1 / 3)
  (h2 : women_no_plan_fraction = 1 / 5)
  (h3 : men_with_plan_fraction = 2 / 5)
  (h4 : women_workers = 140) :
  total_workers - women_workers = 1120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_workers_count_l1059_105935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_job_earnings_l1059_105938

/-- Proves that working approximately 29 hours per week for the last 7 weeks,
    given the conditions of missed work and wage increase, will result in total earnings of $4,500 -/
theorem summer_job_earnings 
  (total_weeks : ℕ) 
  (planned_hours_per_week : ℕ) 
  (target_earnings : ℚ) 
  (missed_weeks : ℕ) 
  (increased_wage_weeks : ℕ) 
  (wage_increase_factor : ℚ) 
  (h1 : total_weeks = 14)
  (h2 : planned_hours_per_week = 25)
  (h3 : target_earnings = 4500)
  (h4 : missed_weeks = 3)
  (h5 : increased_wage_weeks = 4)
  (h6 : wage_increase_factor = 3/2)
  : ∃ (hours_per_week : ℚ), 
    (hours_per_week ≥ 28 ∧ hours_per_week ≤ 30) ∧ 
    (increased_wage_weeks * wage_increase_factor * planned_hours_per_week * (target_earnings / (total_weeks * planned_hours_per_week)) + 
     (total_weeks - missed_weeks - increased_wage_weeks) * hours_per_week * (target_earnings / (total_weeks * planned_hours_per_week))) = target_earnings :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_job_earnings_l1059_105938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_always_positive_l1059_105931

theorem function_always_positive (k : ℝ) (x : ℝ) : 
  k ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → 
  (x^2 + (k - 4) * x - 2 * k + 4 > 0 ↔ x < 1 ∨ x > 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_always_positive_l1059_105931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l1059_105920

open Real

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x - π / 4)

-- Define the shifted function
noncomputable def g (x : ℝ) : ℝ := 3 * sin (2 * x)

-- Theorem statement
theorem shift_equivalence :
  ∀ x : ℝ, f x = g (x - π / 8) :=
by
  intro x
  simp [f, g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l1059_105920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_stamps_is_eight_l1059_105966

/-- Represents the number of ways to make 50 cents using 5-cent and 7-cent stamps -/
def stamp_combinations : Set (ℕ × ℕ) :=
  {p | p.1 * 5 + p.2 * 7 = 50}

/-- The total number of stamps used in a combination -/
def total_stamps (p : ℕ × ℕ) : ℕ := p.1 + p.2

/-- Theorem stating that the minimum number of stamps to make 50 cents is 8 -/
theorem min_stamps_is_eight :
  ∃ (p : ℕ × ℕ), p ∈ stamp_combinations ∧
    (∀ (p' : ℕ × ℕ), p' ∈ stamp_combinations → total_stamps p ≤ total_stamps p') ∧
    total_stamps p = 8 := by
  sorry

#check min_stamps_is_eight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_stamps_is_eight_l1059_105966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_40_50_70_l1059_105989

/-- The area of a triangle given its three side lengths -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with side lengths 40, 50, and 70 is equal to 80√15 -/
theorem triangle_area_40_50_70 :
  triangle_area 40 50 70 = 80 * Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_40_50_70_l1059_105989
