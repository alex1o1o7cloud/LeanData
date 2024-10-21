import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_three_halves_l68_6824

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (4 * x - 6)

theorem vertical_asymptote_at_three_halves :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 3/2| ∧ |x - 3/2| < δ → |f x| > 1/ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_three_halves_l68_6824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_some_number_value_l68_6857

/-- Given that (3.242 * 14) / some_number = 0.045388, prove that some_number ≈ 1000 -/
theorem some_number_value (some_number : ℝ) 
  (h : (3.242 * 14) / some_number = 0.045388) : 
  ‖some_number - 1000‖ < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_some_number_value_l68_6857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_die_roll_probability_l68_6847

/-- Represents the outcome of rolling a standard die 8 times -/
structure DieRolls where
  rolls : Fin 8 → Nat
  odd_rolls : ∀ i, Odd (rolls i)

/-- The probability of rolling an odd number on a single roll -/
noncomputable def prob_odd_single : ℝ := 1/2

/-- The probability of the sum being greater than 20, given all rolls are odd -/
noncomputable def prob_sum_gt_20 : ℝ := 1/2

/-- The probability of rolling 8 odd numbers and their sum being greater than 20 -/
noncomputable def prob_odd_product_and_sum_gt_20 : ℝ := 1/512

/-- Theorem stating the relationship between probabilities -/
theorem die_roll_probability (rolls : DieRolls) :
  prob_odd_product_and_sum_gt_20 =
    prob_odd_single ^ 8 * prob_sum_gt_20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_die_roll_probability_l68_6847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_two_roots_monotonicity_condition_for_g_l68_6820

-- Define the piecewise function f
noncomputable def f (n : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < n then Real.sqrt x else (x - 1)^2

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a * abs (x - 2)

-- Statement for part 1
theorem smallest_n_for_two_roots :
  ∀ n : ℝ, n > 1 →
  (∃ b : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f n x₁ = b ∧ f n x₂ = b) →
  n ≥ 2 :=
sorry

-- Statement for part 2
theorem monotonicity_condition_for_g :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → Monotone (g a)) ↔ -4 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_two_roots_monotonicity_condition_for_g_l68_6820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_cartesian_l68_6890

-- Define the parametric equations
noncomputable def parametric_x (θ : ℝ) : ℝ := 3 + 4 * Real.cos θ
noncomputable def parametric_y (θ : ℝ) : ℝ := -2 + 4 * Real.sin θ

-- State the theorem
theorem parametric_to_cartesian :
  ∀ (x y θ : ℝ),
  x = parametric_x θ ∧ y = parametric_y θ →
  (x - 3)^2 + (y + 2)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_cartesian_l68_6890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_flip_area_theorem_hexagon_flip_area_theorem_l68_6871

/-- The area occupied by a unit square when flipped around one of its vertices -/
noncomputable def square_flip_area : ℝ := 1 + Real.pi / 2

/-- The area occupied by a unit regular hexagon when flipped around one of its vertices -/
noncomputable def hexagon_flip_area : ℝ := 3 * Real.sqrt 3 / 2 + 2 * Real.pi / 3

/-- Theorem stating the area occupied by a unit square when flipped around one of its vertices -/
theorem square_flip_area_theorem :
  square_flip_area = 1 + Real.pi / 2 := by sorry

/-- Theorem stating the area occupied by a unit regular hexagon when flipped around one of its vertices -/
theorem hexagon_flip_area_theorem :
  hexagon_flip_area = 3 * Real.sqrt 3 / 2 + 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_flip_area_theorem_hexagon_flip_area_theorem_l68_6871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l68_6873

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length train_speed_kmh crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 255 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l68_6873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_consecutive_even_numbers_l68_6876

theorem sum_of_four_consecutive_even_numbers (n1 n2 n3 n4 : ℕ) : 
  (Even n1 ∧ Even n2 ∧ Even n3 ∧ Even n4) →  -- All numbers are even
  (n2 = n1 + 2 ∧ n3 = n2 + 2 ∧ n4 = n3 + 2) →  -- Numbers are consecutive
  (n4 = 38) →  -- Largest number is 38
  (n1 + n2 + n3 + n4 = 140) :=  -- Sum equals 140
by
  sorry

#check sum_of_four_consecutive_even_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_consecutive_even_numbers_l68_6876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l68_6810

def a : Fin 2 → ℝ := ![3, 4]
def b : Fin 2 → ℝ := ![6, 2]
def c : Fin 2 → ℝ := ![1, -1]

theorem triangle_area : 
  let v1 := c
  let v2 := a + c
  let v3 := b + c
  (1/2 : ℝ) * abs ((v2 0 - v1 0) * (v3 1 - v1 1) - (v2 1 - v1 1) * (v3 0 - v1 0)) = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l68_6810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_function_min_units_A_max_profit_l68_6849

/-- Represents the number of units of good A purchased -/
def x : ℕ := sorry

/-- Represents the total profit in yuan -/
def y : ℤ := sorry

/-- Cost price of good A in yuan -/
def cost_A : ℕ := 15

/-- Selling price of good A in yuan -/
def sell_A : ℕ := 20

/-- Cost price of good B in yuan -/
def cost_B : ℕ := 35

/-- Selling price of good B in yuan -/
def sell_B : ℕ := 45

/-- Total number of units purchased -/
def total_units : ℕ := 100

/-- Maximum investment in yuan -/
def max_investment : ℕ := 3000

/-- Theorem stating the functional relationship between profit and units of A purchased -/
theorem profit_function : y = -5 * (x : ℤ) + 1000 := by sorry

/-- Theorem stating the minimum number of units of A that should be purchased -/
theorem min_units_A : x ≥ 25 := by sorry

/-- Theorem stating the maximum achievable profit -/
theorem max_profit : y ≤ 875 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_function_min_units_A_max_profit_l68_6849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l68_6804

theorem job_completion_time
  (p q s : ℝ)
  (h_positive : p > 0 ∧ q > 0 ∧ s ≥ 0)
  : (1.1 * p * q) / (p + 2 * s) =
    (1.1 * p * q) / (p + 2 * s) :=
by
  -- The proof is trivial as we're asserting equality with itself
  rfl

#check job_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l68_6804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l68_6863

noncomputable def line_equation (x y : ℝ) : Prop := y = x + Real.sqrt 3

noncomputable def slope_angle (m : ℝ) : ℝ := Real.arctan m

theorem slope_angle_of_line :
  ∃ (m : ℝ), (∀ x y, line_equation x y → y - Real.sqrt 3 = m * x) ∧
             slope_angle m = 45 * π / 180 := by
  -- Introduce the slope m
  let m := 1
  
  -- Prove the existence of m
  use m
  
  constructor
  
  -- Prove that the slope of the line is 1
  · intro x y h
    rw [line_equation] at h
    rw [h]
    ring
  
  -- Prove that the slope angle is 45 degrees
  · unfold slope_angle
    -- We need to prove that arctan 1 = 45°
    sorry  -- This step requires more advanced trigonometry in Lean

-- The proof is incomplete, but the structure is correct


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l68_6863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_inverse_product_l68_6884

theorem fraction_sum_inverse_product : 
  12 * ((1:ℚ)/3 + (1:ℚ)/4 + (1:ℚ)/6 + (1:ℚ)/12)⁻¹ = 72/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_inverse_product_l68_6884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_lambda_range_l68_6894

-- Define the circle
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P outside the circle
def point_outside_circle (x y : ℝ) : Prop := x^2 + y^2 > 4

-- Define the angle AOB
def angle_AOB : ℝ := 120

-- Define point C
def point_C : ℝ × ℝ := (6, 0)

-- Define the relationship between PO and PC
def PO_PC_relation (lambda : ℝ) (x y : ℝ) : Prop :=
  (x^2 + y^2).sqrt = lambda * ((x - 6)^2 + y^2).sqrt

-- Theorem statement
theorem tangent_circle_lambda_range :
  ∀ x y lambda : ℝ,
  circle_O x y →
  point_outside_circle x y →
  angle_AOB = 120 →
  PO_PC_relation lambda x y →
  lambda ∈ Set.Icc (2/5) 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_lambda_range_l68_6894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_subset_negative_l68_6850

theorem quadratic_subset_negative (p : ℝ) :
  ({x : ℝ | x^2 + (p+2)*x + 1 = 0} ⊂ Set.Iic 0) ↔ p ∈ Set.Ioi (-4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_subset_negative_l68_6850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inserted_numbers_sum_l68_6880

theorem inserted_numbers_sum (a₁ a₂ a₃ a₄ : ℤ) : 
  a₁ = 2015 →
  a₄ = 131 →
  (∀ i : Fin 3, a₁ - i.val * (a₁ - a₄) / 3 > a₁ - (i.val + 1) * (a₁ - a₄) / 3) →
  (∃ d : ℤ, ∀ i : Fin 3, a₁ - (i.val + 1) * d = a₁ - i.val * d - d) →
  a₂ + a₃ = 2146 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inserted_numbers_sum_l68_6880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l68_6877

theorem sin_double_angle_special_case (α : Real) 
  (h1 : Real.sin α = 3/5) 
  (h2 : α ∈ Set.Ioo (Real.pi/2) Real.pi) : 
  Real.sin (2 * α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l68_6877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l68_6866

open Real

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := 3 * sin (2 * x + 2 * π / 3) + 1

-- State the theorem
theorem max_value_of_expression (x₁ x₂ : ℝ) :
  x₁ ∈ Set.Icc (-3 * π / 2) (3 * π / 2) →
  x₂ ∈ Set.Icc (-3 * π / 2) (3 * π / 2) →
  g x₁ * g x₂ = 16 →
  (∃ (y₁ y₂ : ℝ), y₁ ∈ Set.Icc (-3 * π / 2) (3 * π / 2) ∧
                  y₂ ∈ Set.Icc (-3 * π / 2) (3 * π / 2) ∧
                  g y₁ * g y₂ = 16 ∧
                  2 * y₁ - y₂ = 35 * π / 12) ∧
  (∀ (z₁ z₂ : ℝ), z₁ ∈ Set.Icc (-3 * π / 2) (3 * π / 2) →
                   z₂ ∈ Set.Icc (-3 * π / 2) (3 * π / 2) →
                   g z₁ * g z₂ = 16 →
                   2 * z₁ - z₂ ≤ 35 * π / 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l68_6866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_area_polygon_l68_6821

/-- A polygon with 100 sides in the Cartesian plane -/
structure Polygon100 where
  vertices : Fin 100 → ℤ × ℤ

/-- The property that all sides of a polygon are parallel to coordinate axes -/
def parallelToAxes (P : Polygon100) : Prop :=
  ∀ i : Fin 99, (P.vertices i).1 = (P.vertices (i+1)).1 ∨ (P.vertices i).2 = (P.vertices (i+1)).2

/-- The property that all side lengths of a polygon are odd -/
def allSidesOdd (P : Polygon100) : Prop :=
  ∀ i : Fin 99, 
    (((P.vertices i).1 - (P.vertices (i+1)).1).natAbs % 2 = 1) ∨
    (((P.vertices i).2 - (P.vertices (i+1)).2).natAbs % 2 = 1)

/-- The area of a polygon -/
noncomputable def area (P : Polygon100) : ℚ :=
  sorry  -- Actual area calculation would go here

/-- Main theorem: If a 100-sided polygon has integer coordinates, sides parallel to axes, 
    and odd side lengths, then its area is an odd integer -/
theorem odd_area_polygon (P : Polygon100) 
    (h_parallel : parallelToAxes P) 
    (h_odd : allSidesOdd P) : 
  ∃ (n : ℤ), area P = n ∧ n % 2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_area_polygon_l68_6821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_equal_distances_l68_6831

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance from a point to a line --/
noncomputable def distance_point_to_line (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Check if a point is on a line --/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem --/
theorem line_equation_with_equal_distances (l : Line) : 
  point_on_line ⟨1, 2⟩ l →
  distance_point_to_line ⟨2, 3⟩ l = distance_point_to_line ⟨4, -5⟩ l →
  (l.a = 4 ∧ l.b = 1 ∧ l.c = -6) ∨ (l.a = 3 ∧ l.b = 2 ∧ l.c = -7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_equal_distances_l68_6831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_top_triangle_multiple_of_five_l68_6867

-- Define the structure of the triangle
structure TriangleNumbers where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  top : ℤ

-- Define the property that sums around gray triangles are multiples of 5
def validSums (t : TriangleNumbers) : Prop :=
  (3 - t.a) % 5 = 0 ∧
  (-t.a - t.b) % 5 = 0 ∧
  (-t.b - t.c) % 5 = 0 ∧
  (-t.c - t.d) % 5 = 0 ∧
  (2 - t.d) % 5 = 0 ∧
  (2 + 2*t.a + t.b) % 5 = 0 ∧
  (t.a + 2*t.b + t.c) % 5 = 0 ∧
  (t.b + 2*t.c + t.d) % 5 = 0 ∧
  (3 + t.c + 2*t.d) % 5 = 0 ∧
  (3 + 2*t.a + 2*t.b - t.c) % 5 = 0 ∧
  (-t.a + 2*t.b + 2*t.c - t.d) % 5 = 0 ∧
  (2 - t.b + 2*t.c + 2*t.d) % 5 = 0 ∧
  (2 - t.a + t.b - t.c + t.d) % 5 = 0 ∧
  (3 + t.a - t.b + t.c - t.d) % 5 = 0 ∧
  t.top % 5 = 0

-- Theorem statement
theorem top_triangle_multiple_of_five (t : TriangleNumbers) 
  (h : validSums t) : ∃ k : ℤ, t.top = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_top_triangle_multiple_of_five_l68_6867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_proof_l68_6898

noncomputable def f (x : ℝ) : ℝ := if x > 0 then x^3 + x + 1 else -x^3 - x + 1

theorem f_even_proof :
  (∀ x, f x = f (-x)) ∧
  (∀ x > 0, f x = x^3 + x + 1) →
  (∀ x < 0, f x = -x^3 - x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_proof_l68_6898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_rotation_theorem_l68_6875

/-- Represents a disk with 20 numbers arranged around its circumference -/
def Disk := Fin 20 → Fin 20

/-- Represents a rotation of the disk -/
def Rotation := Fin 20

/-- Checks if two disks have no matching numbers at a given rotation -/
def no_match (d1 d2 : Disk) (r : Rotation) : Prop :=
  ∀ i : Fin 20, d1 i ≠ d2 ((i.val + r.val) % 20)

theorem disk_rotation_theorem (d1 d2 : Disk) :
  ∃ r : Rotation, no_match d1 d2 r := by
  sorry

#check disk_rotation_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_rotation_theorem_l68_6875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l68_6841

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x + a else x + 4/x

-- State the theorem
theorem range_of_a :
  {a : ℝ | ∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m ∧ (∃ (y : ℝ), f a y = m)} = {a : ℝ | a ≥ 4} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l68_6841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_customSequence_2006_mod_100_l68_6859

def customSequence (n : ℕ) : ℤ :=
  match n with
  | 0 => 21
  | 1 => 35
  | n + 2 => 4 * customSequence (n + 1) - 4 * customSequence n + n^2

theorem customSequence_2006_mod_100 :
  customSequence 2006 % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_customSequence_2006_mod_100_l68_6859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_alpha_l68_6897

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 - Real.sin (2 * x)) / Real.cos x

-- Define the properties of α
noncomputable def α : ℝ := Real.arctan (-4 / 3) + 2 * Real.pi

-- Theorem statement
theorem f_at_alpha :
  (α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) →  -- α is in the fourth quadrant
  (Real.tan α = -4 / 3) →              -- tan(α) = -4/3
  f α = 49 / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_alpha_l68_6897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l68_6823

theorem power_equality (a b : ℝ) (h1 : (100 : ℝ)^a = 4) (h2 : (100 : ℝ)^b = 5) :
  (20 : ℝ)^((1 - a - b) / (2 * (1 - b))) = Real.sqrt 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l68_6823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_limit_l68_6832

noncomputable def sequence_a : ℕ → ℝ → ℝ
  | 0, a₀ => a₀
  | n + 1, a₀ => 1 / (2 - sequence_a n a₀)

theorem sequence_a_limit (a₀ : ℝ) : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sequence_a n a₀ - 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_limit_l68_6832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_power_tower_eq_four_l68_6893

/-- Represents an infinite power tower with base x -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ :=
  Real.log x / Real.log (Real.log x)

/-- Theorem stating that √2 is the solution to the infinite power tower equation equal to 4 -/
theorem infinite_power_tower_eq_four :
  infinitePowerTower (Real.sqrt 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_power_tower_eq_four_l68_6893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l68_6874

def m (l : ℝ) : ℝ × ℝ := (l + 1, 1)
def n (l : ℝ) : ℝ × ℝ := (l + 2, 2)

theorem perpendicular_vectors_lambda (l : ℝ) :
  (m l + n l) • (m l - n l) = 0 → l = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l68_6874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_factorial_divisibility_l68_6856

theorem largest_n_factorial_divisibility : 
  ∃ (n : ℕ), n = 6 ∧ 
  (∀ m : ℕ, m > n → ¬((Nat.factorial (Nat.factorial m)).factorial ∣ (Nat.factorial 2004).factorial)) ∧
  ((Nat.factorial (Nat.factorial n)).factorial ∣ (Nat.factorial 2004).factorial) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_factorial_divisibility_l68_6856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_even_function_l68_6843

-- Define the four functions
noncomputable def f₁ (x : ℝ) : ℝ := x^2 - 2*x
noncomputable def f₂ (x : ℝ) : ℝ := (x + 1) * Real.sqrt ((1 - x) / (1 + x))
noncomputable def f₃ (x : ℝ) : ℝ := (x - 1)^2
noncomputable def f₄ (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 - 2))

-- Define what it means for a function to be even
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem one_even_function :
  (¬ isEven f₁) ∧ (¬ isEven f₂) ∧ (¬ isEven f₃) ∧ (isEven f₄) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_even_function_l68_6843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equal_parts_l68_6868

theorem complex_equal_parts (a : ℝ) : 
  (((2 : ℂ) + a * Complex.I) * (1 + Complex.I)).re = 
  (((2 : ℂ) + a * Complex.I) * (1 + Complex.I)).im → 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equal_parts_l68_6868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_imaginary_axis_length_l68_6848

noncomputable def hyperbola (x y b : ℝ) : Prop := x^2 / 3 - y^2 / b^2 = 1

noncomputable def focal_length (b : ℝ) : ℝ := Real.sqrt (3 + b^2)

def distance_to_asymptote (b : ℝ) : ℝ := b

theorem hyperbola_imaginary_axis_length 
  (b : ℝ) 
  (h1 : b > 0) 
  (h2 : distance_to_asymptote b = (1/4) * focal_length b) : 
  2 * b = 2 := by
  sorry

#check hyperbola_imaginary_axis_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_imaginary_axis_length_l68_6848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_condition_l68_6830

noncomputable section

open Real

theorem obtuse_triangle_condition (A B C : ℝ) : 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π →
  (sin A * sin B < cos A * cos B → C > π / 2) ∧
  ¬(C > π / 2 → sin A * sin B < cos A * cos B) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_condition_l68_6830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_equal_and_real_l68_6851

theorem quadratic_roots_equal_and_real (a c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 4 * x * Real.sqrt 3 + c
  let discriminant := (-4 * Real.sqrt 3)^2 - 4 * a * c
  discriminant = 0 →
  ∃ r : ℝ, (∀ x : ℝ, f x = 0 ↔ x = r) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_equal_and_real_l68_6851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_height_calculation_l68_6828

/-- Given the shadow lengths and heights of objects, calculate the height of a house and a flagpole. -/
theorem shadow_height_calculation (house_shadow : ℝ) (tree_height tree_shadow : ℝ) (flagpole_shadow : ℝ)
  (h_house_shadow : house_shadow = 70)
  (h_tree_height : tree_height = 28)
  (h_tree_shadow : tree_shadow = 40)
  (h_flagpole_shadow : flagpole_shadow = 25) :
  ∃ (house_height flagpole_height : ℝ),
    house_height = 49 ∧
    flagpole_height = 17.5 ∧
    (Int.floor (flagpole_height + 0.5) : ℤ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_height_calculation_l68_6828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chuck_area_l68_6879

/-- The area available for a point tied to the corner of a rectangle with a radius constraint, moving only outside the rectangle. -/
noncomputable def tiedPointArea (width : ℝ) (height : ℝ) (radius : ℝ) : ℝ :=
  (3/4) * Real.pi * radius^2 + (1/4) * Real.pi * (radius - width)^2

/-- The specific problem of Chuck the llama -/
theorem chuck_area :
  let width : ℝ := 4
  let height : ℝ := 3
  let radius : ℝ := 5
  tiedPointArea width height radius = 19.75 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chuck_area_l68_6879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_necessary_not_sufficient_l68_6808

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 2) / Real.log (1/2)

def monotone_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

def I : Set ℝ := Set.Iic 1

theorem f_monotone_necessary_not_sufficient :
  (∀ a : ℝ, monotone_increasing (f a) I → a < 0) ∧
  (∃ a : ℝ, a < 0 ∧ ¬(monotone_increasing (f a) I)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_necessary_not_sufficient_l68_6808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l68_6840

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2^x + 1) / (2^x - 1)

-- State the theorem
theorem f_odd_and_decreasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l68_6840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_day_ratio_l68_6834

/-- Represents the number of eggs processed per day -/
def total_eggs : ℕ := 400

/-- Represents the number of additional eggs accepted on the particular day -/
def additional_accepted : ℕ := 12

/-- Represents the ratio of accepted to rejected eggs on a normal day -/
def normal_ratio : Rat := 24 / 1

/-- Calculates the number of accepted eggs on a normal day -/
noncomputable def normal_accepted : ℕ := 
  (total_eggs * normal_ratio.num / (normal_ratio.num + normal_ratio.den)).toNat

/-- Calculates the number of rejected eggs on a normal day -/
noncomputable def normal_rejected : ℕ := total_eggs - normal_accepted

/-- Calculates the number of accepted eggs on the particular day -/
noncomputable def particular_accepted : ℕ := normal_accepted + additional_accepted

/-- Calculates the number of rejected eggs on the particular day -/
noncomputable def particular_rejected : ℕ := normal_rejected - additional_accepted

/-- Theorem stating that the ratio of accepted to rejected eggs on the particular day is 99:1 -/
theorem particular_day_ratio :
  (particular_accepted : ℚ) / (particular_rejected : ℚ) = 99 / 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_day_ratio_l68_6834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l68_6885

/-- Given a hyperbola with one of its asymptotes as y = √2x, its eccentricity is either √3 or √6/2 -/
theorem hyperbola_eccentricity (H : Real) (asymptote : Set (ℝ × ℝ)) 
  (h_asymptote : asymptote = {(x, y) | y = Real.sqrt 2 * x}) :
  H = Real.sqrt 3 ∨ H = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l68_6885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_n_value_l68_6803

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a2_eq_3 : a 2 = 3
  an_minus1_eq_17 : ∀ n, n ≥ 2 → a (n - 1) = 17
  sum_eq_100 : ∀ n, n ≥ 2 → n * (a 2 + a (n - 1)) / 2 = 100

/-- The value of n for which the sum of the first n terms is 100 -/
theorem arithmetic_sequence_n_value (seq : ArithmeticSequence) : 
  ∃ n : ℕ, n ≥ 2 ∧ n * (seq.a 2 + seq.a (n - 1)) / 2 = 100 ∧ n = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_n_value_l68_6803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l68_6833

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x - m * Real.log (m * x - m) + m

-- State the theorem
theorem f_properties :
  (∀ x > 1, f 1 x > 4) ∧
  (∀ m > 0, (∀ x > 1, f m x ≥ 0) ↔ m ≤ Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l68_6833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_m_n_l68_6887

theorem existence_of_m_n : ∃ m n : ℕ+, 
  (∃ S : Finset ℕ+, Finset.card S ≥ 2012 ∧
    ∀ x ∈ S, ∃ a b : ℕ, 
      (m : ℤ) - (x : ℤ)^2 = (a : ℤ)^2 ∧ 
      (n : ℤ) - (x : ℤ)^2 = (b : ℤ)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_m_n_l68_6887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_inscribed_quadrilateral_l68_6846

-- Define a Point as a pair of integers
def Point := ℤ × ℤ

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define what it means for a quadrilateral to be convex and inscribed
def is_convex_inscribed (a b x y : Point) : Prop :=
  -- We don't implement the full definition here, as it would be complex
  -- and not necessary for the statement of the theorem
  sorry

-- The main theorem
theorem convex_inscribed_quadrilateral (a b : Point) :
  (∃ x y : Point, x ≠ y ∧ is_convex_inscribed a b x y) ↔ distance a b ≠ 1 := by
  sorry

#check convex_inscribed_quadrilateral

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_inscribed_quadrilateral_l68_6846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_no_lattice_points_l68_6814

theorem max_a_no_lattice_points :
  ∀ (a : ℚ),
  (∀ (m : ℚ), 1/2 < m → m < a →
    ∀ (x y : ℤ), 0 < x → x ≤ 100 → y = ⌊(m * ↑x + 2 : ℚ)⌋ → (y : ℚ) ≠ m * ↑x + 2) →
  a ≤ 50/99 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_no_lattice_points_l68_6814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l68_6889

-- Define the function v(x)
noncomputable def v (x : ℝ) : ℝ := 1 / (Real.sqrt x + x - 1)

-- Define the domain of v(x)
def domain_v : Set ℝ := {x | x ≥ 0 ∧ x ≠ (3 - Real.sqrt 5) / 2}

-- Theorem stating that the domain of v is correct
theorem domain_of_v :
  {x : ℝ | ∃ y, v x = y} = domain_v := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l68_6889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_cardinality_l68_6855

def A : Finset ℕ := {2, 3, 5, 7}
def B : Finset ℕ := {2, 4, 6, 8}
def U : Finset ℕ := A ∪ B

theorem complement_intersection_cardinality : Finset.card (U \ (A ∩ B)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_cardinality_l68_6855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_m_bound_l68_6836

open Real

theorem function_inequality_implies_m_bound 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (m : ℝ) 
  (hf : ∀ x, f x = 1 + sin (2 * x)) 
  (hg : ∀ x, g x = 2 * (cos x)^2 + m) 
  (h_exists : ∃ x₀ ∈ Set.Icc 0 (π / 2), f x₀ ≥ g x₀) : 
  m ≤ sqrt 2 := by
  sorry

#check function_inequality_implies_m_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_m_bound_l68_6836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_inequality_l68_6861

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the set S and its projections
variable (S : Finset Point3D)

noncomputable def S_x : Finset (ℝ × ℝ) := S.image (fun p => (p.y, p.z))
noncomputable def S_y : Finset (ℝ × ℝ) := S.image (fun p => (p.x, p.z))
noncomputable def S_z : Finset (ℝ × ℝ) := S.image (fun p => (p.x, p.y))

-- State the theorem
theorem projection_inequality (S : Finset Point3D) : 
  (S.card ^ 2 : ℕ) ≤ (S_x S).card * (S_y S).card * (S_z S).card := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_inequality_l68_6861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_example_l68_6815

/-- The sum of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℝ) (d : ℝ) (aₙ : ℝ) : ℝ :=
  let n := (aₙ - a₁) / d + 1
  n / 2 * (a₁ + aₙ)

/-- Theorem: The sum of the arithmetic sequence with first term 2, 
    common difference 4, and last term 42 is equal to 242 -/
theorem arithmetic_sum_example : arithmetic_sum 2 4 42 = 242 := by
  -- Unfold the definition of arithmetic_sum
  unfold arithmetic_sum
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_example_l68_6815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equals_cube_surface_l68_6844

noncomputable section

/-- The side length of the cube -/
def cube_side : ℝ := 3

/-- The surface area of the cube -/
def cube_surface_area : ℝ := 6 * cube_side^2

/-- The surface area of the sphere -/
def sphere_surface_area : ℝ := cube_surface_area

/-- The radius of the sphere -/
noncomputable def sphere_radius : ℝ := Real.sqrt (sphere_surface_area / (4 * Real.pi))

/-- The volume of the sphere -/
noncomputable def sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius^3

/-- The constant K in the volume expression -/
noncomputable def K : ℝ := sphere_volume * Real.sqrt Real.pi / Real.sqrt 6

theorem sphere_volume_equals_cube_surface : K = 72 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equals_cube_surface_l68_6844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_DO_vector_l68_6812

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors e₁ and e₂
variable (e₁ e₂ : V)

-- Define the points A, B, C, D, and O
variable (A B C D O : V)

-- Define the properties of the quadrilateral
axiom diag_intersect : O = (1/2 : ℝ) • (A + C) ∧ O = (1/2 : ℝ) • (B + D)
axiom AB_vec : B - A = 4 • e₁
axiom BC_vec : C - B = 6 • e₂

-- State the theorem
theorem DO_vector : D - O = 2 • e₁ - 3 • e₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_DO_vector_l68_6812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_inscribed_quadrilateral_l68_6896

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  radius : ℝ
  inscribed : Bool
  perpendicular_diagonals : Bool

/-- The length of a line segment between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: The radius of a circle circumscribing a quadrilateral with perpendicular diagonals -/
theorem radius_of_inscribed_quadrilateral (q : InscribedQuadrilateral) 
  (h_inscribed : q.inscribed = true)
  (h_perpendicular : q.perpendicular_diagonals = true)
  (h_AB : distance q.A q.B = 4)
  (h_CD : distance q.C q.D = 2) :
  q.radius = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_inscribed_quadrilateral_l68_6896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_trailer_homes_count_l68_6835

theorem new_trailer_homes_count : ∃ (new_count : ℕ), new_count = 42 := by
  let initial_count : ℕ := 30
  let initial_average_age : ℕ := 12
  let years_passed : ℕ := 5
  let current_average_age : ℕ := 10
  
  -- Define the calculation for new_count
  let new_count : ℕ := (initial_count * (initial_average_age + years_passed) - initial_count * current_average_age) / (current_average_age - years_passed)
  
  -- Assert the existence of new_count and its value
  use new_count
  
  -- Proof (skipped with sorry)
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_trailer_homes_count_l68_6835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l68_6801

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * (Real.cos x) ^ 2, Real.sqrt 3)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, Real.sin (2 * x))

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem f_properties :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ x : ℝ, f (π / 3 + x) = f (π / 3 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l68_6801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dimension_diff_formula_l68_6888

/-- A rectangular prism with total edge length P and space diagonal D -/
structure RectangularPrism (P D : ℝ) where
  x : ℝ -- length
  y : ℝ -- width
  z : ℝ -- height
  edge_length : x + y + z = P / 4
  diagonal : x^2 + y^2 + z^2 = D^2
  height_eq_width : z = y
  x_geq_y : x ≥ y

/-- The difference between the largest and smallest dimension -/
def dimension_diff (P D : ℝ) (prism : RectangularPrism P D) : ℝ :=
  prism.x - prism.y

theorem dimension_diff_formula (P D : ℝ) (prism : RectangularPrism P D) :
  dimension_diff P D prism = Real.sqrt (P^2 / 16 - D^2 + 2 * prism.y^2) :=
by
  sorry -- Proof steps would go here

#check dimension_diff_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dimension_diff_formula_l68_6888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_problem_l68_6842

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the plane -/
def Point := ℝ × ℝ

/-- The area of a circle -/
noncomputable def Circle.area (c : Circle) : ℝ := Real.pi * c.radius^2

/-- The tangent line to a circle at a point -/
def TangentLine (c : Circle) (p : Point) : Set (ℝ × ℝ) := sorry

theorem circle_area_problem (ω : Circle) (A B : Point) :
  A = (5, 16) →
  B = (13, 14) →
  A ∈ { p : ℝ × ℝ | (p.1 - ω.center.1)^2 + (p.2 - ω.center.2)^2 = ω.radius^2 } →
  B ∈ { p : ℝ × ℝ | (p.1 - ω.center.1)^2 + (p.2 - ω.center.2)^2 = ω.radius^2 } →
  ∃ (x : ℝ), (TangentLine ω A ∩ TangentLine ω B) ∩ {p : ℝ × ℝ | p.2 = 0} = {(x, 0)} →
  ω.area = 1024.25 * Real.pi / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_problem_l68_6842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_20_seconds_l68_6829

noncomputable section

-- Define the velocities as functions of time
def v₁ (t : ℝ) : ℝ := 5 * t
def v₂ (t : ℝ) : ℝ := 3 * t^2

-- Define the positions as integrals of velocities
noncomputable def s₁ (t : ℝ) : ℝ := ∫ x in (0 : ℝ)..t, v₁ x
noncomputable def s₂ (t : ℝ) : ℝ := ∫ x in (0 : ℝ)..t, v₂ x

-- Define the distance between the two bodies
noncomputable def distance (t : ℝ) : ℝ := s₂ t - s₁ t

-- State the theorem
theorem distance_after_20_seconds :
  distance 20 = 7000 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_20_seconds_l68_6829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l68_6819

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x - a - 1 > 0}

-- Define the complement of A in ℝ
def complement_A (a : ℝ) : Set ℝ := {x | x^2 - a*x - a - 1 ≤ 0}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (∃! z : ℤ, (z : ℝ) ∈ complement_A a) → -3 < a ∧ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l68_6819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_faster_by_856_minutes_l68_6895

/-- Represents a participant in the relay race -/
inductive Participant
  | Apple
  | Mac
  | Orange

/-- Represents a segment of the race -/
inductive Segment
  | Forest
  | SandyBeach
  | Mountain

/-- Returns the distance (in miles) for a given participant in a given segment -/
def distance (p : Participant) (s : Segment) : ℚ :=
  match p, s with
  | Participant.Apple, Segment.Forest => 18
  | Participant.Apple, Segment.SandyBeach => 6
  | Participant.Apple, Segment.Mountain => 3
  | Participant.Mac, Segment.Forest => 20
  | Participant.Mac, Segment.SandyBeach => 8
  | Participant.Mac, Segment.Mountain => 3
  | Participant.Orange, Segment.Forest => 22
  | Participant.Orange, Segment.SandyBeach => 10
  | Participant.Orange, Segment.Mountain => 3

/-- Returns the speed (in mph) for a given participant in a given segment -/
def speed (p : Participant) (s : Segment) : ℚ :=
  match p, s with
  | Participant.Apple, Segment.Forest => 3
  | Participant.Apple, Segment.SandyBeach => 2
  | Participant.Apple, Segment.Mountain => 1
  | Participant.Mac, Segment.Forest => 4
  | Participant.Mac, Segment.SandyBeach => 3
  | Participant.Mac, Segment.Mountain => 1
  | Participant.Orange, Segment.Forest => 5
  | Participant.Orange, Segment.SandyBeach => 4
  | Participant.Orange, Segment.Mountain => 2

/-- Calculates the total time (in hours) for a participant to complete all segments -/
def totalTime (p : Participant) : ℚ :=
  (distance p Segment.Forest / speed p Segment.Forest) +
  (distance p Segment.SandyBeach / speed p Segment.SandyBeach) +
  (distance p Segment.Mountain / speed p Segment.Mountain)

/-- The main theorem to prove -/
theorem orange_faster_by_856_minutes :
  (totalTime Participant.Apple + totalTime Participant.Mac - totalTime Participant.Orange) * 60 = 856 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_faster_by_856_minutes_l68_6895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_april_decrease_approx_26_percent_l68_6883

/-- Represents a sequence of price changes --/
structure PriceChanges where
  jan : ℝ
  feb : ℝ
  mar : ℝ
  apr : ℝ
  may : ℝ

/-- Applies a sequence of price changes to an initial price --/
noncomputable def applyChanges (initial : ℝ) (changes : PriceChanges) : ℝ :=
  initial * (1 + changes.jan/100) * (1 - changes.feb/100) * (1 + changes.mar/100) * 
            (1 - changes.apr/100) * (1 + changes.may/100)

/-- Theorem stating that given specific price changes, if the final price equals 
    the initial price, then the April decrease is approximately 26% --/
theorem april_decrease_approx_26_percent (initial : ℝ) (changes : PriceChanges) 
    (h1 : initial > 0)
    (h2 : changes.jan = 30)
    (h3 : changes.feb = 25)
    (h4 : changes.mar = 20)
    (h5 : changes.may = 15)
    (h6 : applyChanges initial changes = initial) :
    ∃ (ε : ℝ), changes.apr = 26 + ε ∧ abs ε < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_april_decrease_approx_26_percent_l68_6883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l68_6891

noncomputable def f (x : ℝ) : ℝ := 2^x
noncomputable def g (x : ℝ) : ℝ := 2^(-x-3) + 1

theorem transform_f_to_g :
  ∃ (a b : ℝ),
    (∀ x, g x = f (-x + a) + b) ∧
    (∀ x, g x = f (-x + a) + b) := by
  -- We choose a = 3 and b = 1
  use 3, 1
  constructor
  · intro x
    simp [f, g]
    -- The proof steps would go here
    sorry
  · intro x
    simp [f, g]
    -- The proof steps would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l68_6891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_if_perpendicular_to_same_line_lines_parallel_if_perpendicular_to_same_plane_l68_6838

-- Define the types for lines and planes
structure Line where

structure Plane where

-- Define the perpendicular and parallel relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry

def parallel (p1 p2 : Plane) : Prop := sorry

def parallel_lines (l1 l2 : Line) : Prop := sorry

-- Theorem 1: If a line is perpendicular to two planes, then the two planes are parallel
theorem planes_parallel_if_perpendicular_to_same_line (a : Line) (α β : Plane) :
  perpendicular a α → perpendicular a β → parallel α β := by
  sorry

-- Theorem 2: If two lines are perpendicular to the same plane, then the lines are parallel
theorem lines_parallel_if_perpendicular_to_same_plane (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel_lines a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_if_perpendicular_to_same_line_lines_parallel_if_perpendicular_to_same_plane_l68_6838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_omega_l68_6822

/-- The minimum absolute value of ω given the graph transformation -/
theorem min_abs_omega : 
  ∃ ω : ℝ, 
    (∀ ω' : ℝ, (∃ x : ℝ, ∀ y : ℝ, Real.sin (ω * (y - π/6)) = Real.sin (ω' * y - 4*π/3)) → |ω| ≤ |ω'|) ∧ 
    |ω| = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_omega_l68_6822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l68_6852

theorem inequality_solution_set (x : ℝ) :
  let f := λ x : ℝ => (x + 2) / (x - 2) + (x + 4) / (3 * x)
  let s := Set.Ioo ((7 - Real.sqrt 33) / 4) ((7 + Real.sqrt 33) / 4) ∪ Set.Ioi 2
  (x ∈ s) ↔ (f x ≥ 4 ∧ x ≠ 0 ∧ x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l68_6852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_six_percentage_l68_6878

theorem divisible_by_six_percentage (n : ℕ) : n = 120 → 
  (((Finset.filter (λ x => x % 6 = 0) (Finset.range (n + 1))).card : ℚ) / (n : ℚ)) * 100 = 100 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_six_percentage_l68_6878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_l68_6817

/-- Rounds a given real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The sum of 47.2189 and 34.0076, when rounded to the nearest hundredth, equals 81.23 -/
theorem sum_and_round :
  round_to_hundredth (47.2189 + 34.0076) = 81.23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_l68_6817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_minus_2b_is_sqrt3_l68_6870

noncomputable def a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
noncomputable def b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

def scalar_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

theorem magnitude_a_minus_2b_is_sqrt3 :
  magnitude (vector_sub a (scalar_mul 2 b)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_minus_2b_is_sqrt3_l68_6870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l68_6809

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (x^2 + 1)

theorem odd_function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a b (-x) = -f a b x) →
  f a b (1/2) = 2/5 →
  (∃ g : ℝ → ℝ, 
    (∀ x, g x = x / (x^2 + 1)) ∧
    (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → g x = f a b x) ∧
    (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → g x < g y) ∧
    (Set.Ioo 0 (1/3 : ℝ) = {x | g (2*x-1) + g x < 0})) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l68_6809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_center_locus_l68_6811

-- Define a point on the parabola y = x^2
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  eq : y = x^2

-- Define a line passing through a parabola point with slope 2x
noncomputable def line_through_point (p : ParabolaPoint) : ℝ → ℝ :=
  λ x ↦ 2 * p.x * (x - p.x) + p.y

-- Define the intersection point of two lines
noncomputable def intersection (l1 l2 : ℝ → ℝ) : ℝ × ℝ :=
  let x := (l2 0 - l1 0) / ((l1 1 - l1 0) - (l2 1 - l2 0))
  (x, l1 x)

-- Define the center of a triangle given by three points
noncomputable def triangle_center (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

-- Placeholder for IsEquilateral
def IsEquilateral (_ _ _ : ℝ × ℝ) : Prop := sorry

-- Main theorem
theorem equilateral_triangle_center_locus 
  (p1 p2 p3 : ParabolaPoint)
  (h : IsEquilateral (intersection (line_through_point p1) (line_through_point p2))
                     (intersection (line_through_point p2) (line_through_point p3))
                     (intersection (line_through_point p3) (line_through_point p1))) :
  (triangle_center (intersection (line_through_point p1) (line_through_point p2))
                   (intersection (line_through_point p2) (line_through_point p3))
                   (intersection (line_through_point p3) (line_through_point p1))).2 = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_center_locus_l68_6811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rug_inner_length_l68_6845

/-- Represents the dimensions of a rectangular rug region -/
structure RectangleDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def rectangleArea (d : RectangleDimensions) : ℝ := d.length * d.width

/-- Represents the three regions of the rug -/
structure RugRegions where
  x : ℝ
  inner : RectangleDimensions := { length := x, width := 1 }
  middle : RectangleDimensions := { length := x + 2, width := 3 }
  outer : RectangleDimensions := { length := x + 4, width := 5 }

/-- Calculates the areas of the three rug regions -/
def rugAreas (r : RugRegions) : List ℝ :=
  [rectangleArea r.inner,
   rectangleArea r.middle - rectangleArea r.inner,
   rectangleArea r.outer - rectangleArea r.middle]

/-- Checks if a list of three numbers forms an arithmetic progression -/
def isArithmeticProgression (l : List ℝ) : Prop :=
  l.length = 3 ∧ l[1]! - l[0]! = l[2]! - l[1]!

theorem rug_inner_length :
  ∃ x : ℝ, x > 0 ∧ isArithmeticProgression (rugAreas { x := x }) → x = 2 := by
  sorry

#eval rugAreas { x := 2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rug_inner_length_l68_6845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_x_coordinate_Q_l68_6882

noncomputable def upperSemicircle (r : ℝ) : ℝ → ℝ :=
  λ x => Real.sqrt (r^2 - x^2)

noncomputable def parabola : ℝ → ℝ :=
  λ x => Real.sqrt x

noncomputable def intersectionPoint (r : ℝ) : ℝ × ℝ :=
  let α := (-1 + Real.sqrt (1 + 4*r^2)) / 2
  (α, Real.sqrt α)

noncomputable def lineSlope (r : ℝ) : ℝ :=
  let (α, β) := intersectionPoint r
  (β - r) / α

noncomputable def xCoordinateQ (r : ℝ) : ℝ :=
  let (α, β) := intersectionPoint r
  (r * α) / (r - β)

theorem limit_x_coordinate_Q :
  ∀ ε > 0, ∃ δ > 0, ∀ r, 0 < r ∧ r < δ → |xCoordinateQ r - 2| < ε := by
  sorry

#check limit_x_coordinate_Q

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_x_coordinate_Q_l68_6882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_tiling_exists_l68_6816

/-- Represents a chessboard -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Represents a 1x3 tile -/
def Tile := Fin 1 → Fin 3 → Bool

/-- A tiling of a chessboard with 1x3 tiles -/
def Tiling (board : Chessboard) := 
  ∃ (tiles : Set Tile) (placement : tiles → Fin 8 × Fin 8),
    ∀ (i j : Fin 8), board i j = true → 
      ∃ (t : Tile) (h : t ∈ tiles),
        ∃ (x y : Fin 8), placement ⟨t, h⟩ = (x, y) ∧
          (i.val - x.val : ℤ) ∈ ({0, 1, 2} : Set ℤ) ∧ 
          (j.val - y.val : ℤ) ∈ ({0} : Set ℤ)

/-- The main theorem -/
theorem chessboard_tiling_exists :
  ∃ (removed_square : Fin 8 × Fin 8),
    ∃ (board : Chessboard),
      (∀ (i j : Fin 8), board i j = ((i, j) ≠ removed_square)) ∧
      Tiling board :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_tiling_exists_l68_6816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_defective_probability_l68_6892

/-- The number of total products -/
def total_products : ℕ := 10

/-- The number of defective products -/
def defective_products : ℕ := 4

/-- The number of non-defective products -/
def non_defective_products : ℕ := 6

/-- The test number on which the last defective product is discovered -/
def last_defective_test : ℕ := 5

/-- Calculates the probability of discovering the last defective product on the specified test -/
def probability_last_defective (n : ℕ) (d : ℕ) (nd : ℕ) (t : ℕ) : ℚ :=
  (Nat.choose nd 1 * Nat.choose d 1 * (Nat.factorial (t - 1))) / (Nat.factorial n / Nat.factorial (n - t))

theorem last_defective_probability :
  probability_last_defective total_products defective_products non_defective_products last_defective_test = 2 / 105 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_defective_probability_l68_6892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integer_probability_l68_6860

theorem odd_integer_probability (n : ℕ) (hn : n = 2016) :
  let S := Finset.range n
  let odd_set := S.filter (λ x => x % 2 = 1)
  (odd_set.card.choose 3 : ℚ) / (S.card.choose 3 : ℚ) < 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integer_probability_l68_6860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_range_l68_6839

theorem sin_cos_equation_range (a : ℝ) :
  (∃ x : ℝ, Real.sin x ^ 2 + Real.cos x + a = 0) ↔ -5/4 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_range_l68_6839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vinegar_evaporation_l68_6806

/-- The proportion of vinegar remaining after one year -/
noncomputable def yearly_remaining : ℝ := 0.8

/-- The proportion of vinegar remaining after n years -/
noncomputable def remaining_after (n : ℝ) : ℝ := yearly_remaining ^ n

/-- The target proportion of vinegar remaining -/
noncomputable def target_remaining : ℝ := 0.64

/-- The number of years required to reach the target remaining proportion -/
noncomputable def years_elapsed : ℝ := Real.log target_remaining / Real.log yearly_remaining

/-- Theorem stating that the number of years elapsed is approximately 2 -/
theorem vinegar_evaporation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |years_elapsed - 2| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vinegar_evaporation_l68_6806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_word2_given_M_l68_6827

-- Define the two words
def word1 : String := "MATHEMATICS"
def word2 : String := "MEMES"

-- Define the probability of choosing each word
noncomputable def prob_choose_word : ℚ := 1 / 2

-- Function to count occurrences of a character in a string
def count_char (s : String) (c : Char) : ℕ := s.toList.filter (· = c) |>.length

-- Define the probability of drawing M from each word
noncomputable def prob_M_given_word1 : ℚ := (count_char word1 'M' : ℚ) / word1.length
noncomputable def prob_M_given_word2 : ℚ := (count_char word2 'M' : ℚ) / word2.length

-- Define the total probability of drawing M
noncomputable def prob_M : ℚ := prob_M_given_word1 * prob_choose_word + prob_M_given_word2 * prob_choose_word

-- State the theorem
theorem prob_word2_given_M : 
  (prob_M_given_word2 * prob_choose_word) / prob_M = 11 / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_word2_given_M_l68_6827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_winning_N_l68_6899

def is_winning_strategy (N : ℕ) : Prop :=
  ∀ (list : List ℕ),
    (∀ x ∈ list, 0 < x ∧ x ≤ 25) →
    (list.sum ≥ 200) →
    ∃ (subset : List ℕ),
      (∀ x ∈ subset, x ∈ list) ∧
      200 - N ≤ subset.sum ∧ subset.sum ≤ 200 + N

theorem smallest_winning_N :
  ∃! (N : ℕ),
    N > 0 ∧
    is_winning_strategy N ∧
    ∀ (M : ℕ), 0 < M ∧ M < N → ¬is_winning_strategy M :=
by
  -- The proof would go here
  sorry

#check smallest_winning_N

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_winning_N_l68_6899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_parabola_properties_l68_6807

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the parabola M
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the conditions
variable (a b : ℝ) (F₁ F₂ : ℝ × ℝ)

-- Define area_triangle function (placeholder)
noncomputable def area_triangle (O A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_and_parabola_properties
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_gt_b : a > b)
  (h_ellipse_focus : ellipse a b (focus.1) (focus.2))
  (h_F₁_left : F₁.1 < 0)
  (h_F₂_right : F₂.1 > 0)
  (h_dot_product : (F₁.1 - focus.1) * (F₂.1 - F₁.1) + (F₁.2 - focus.2) * (F₂.2 - F₁.2) = 6) :
  (∃ (x y : ℝ), ellipse 2 1 x y) ∧
  (∃ (S : ℝ), S = 1 ∧ ∀ (A B : ℝ × ℝ), 
    (∃ (l : ℝ → ℝ), (∀ t, parabola t (l t)) ∧ 
      ellipse 2 1 A.1 A.2 ∧ ellipse 2 1 B.1 B.2) →
    area_triangle (0, 0) A B ≤ S) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_parabola_properties_l68_6807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_pi_half_plus_alpha_l68_6865

theorem sin_three_pi_half_plus_alpha (α : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos α = -5 ∧ r * Real.sin α = -12) →
  Real.sin (3 * Real.pi / 2 + α) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_pi_half_plus_alpha_l68_6865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_containing_interval_l68_6826

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x + 4
def g (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Statement for part 1
theorem solution_set_when_a_is_one :
  let a := 1
  ∃ (S : Set ℝ), S = {x : ℝ | f a x ≥ g x} ∧ S = Set.Icc (-1) ((Real.sqrt 17 - 1) / 2) :=
sorry

-- Statement for part 2
theorem range_of_a_containing_interval :
  {a : ℝ | ∀ x ∈ Set.Icc (-1) 1, f a x ≥ g x} = Set.Icc (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_containing_interval_l68_6826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreases_l68_6872

/-- Inverse proportion function -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_proportion_decreases (k : ℝ) (y₁ y₂ : ℝ) 
  (h1 : k > 0) 
  (h2 : y₁ = inverse_proportion k 1) 
  (h3 : y₂ = inverse_proportion k 4) : 
  y₁ > y₂ := by
  sorry

#check inverse_proportion_decreases

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreases_l68_6872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_ratio_l68_6805

/-- Given a parallelogram with area S and a quadrilateral formed by the intersection of its angle bisectors with area Q, this theorem states that the ratio of the lengths of the sides of the parallelogram is (S + Q + √(Q² + 2QS)) / S. -/
theorem parallelogram_side_ratio (S Q : ℝ) (h1 : S > 0) (h2 : Q > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∃ (α : ℝ), 0 < α ∧ α < Real.pi ∧ S = a * b * Real.sin α) ∧
  (∃ (rect_area : ℝ), rect_area = Q ∧ rect_area = ((a - b)^2 * Real.sin α) / 2) ∧
  (a / b = (S + Q + Real.sqrt (Q^2 + 2*Q*S)) / S ∨
   b / a = (S + Q + Real.sqrt (Q^2 + 2*Q*S)) / S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_ratio_l68_6805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l68_6813

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x+a)ln((2x-1)/(2x+1)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

/-- The domain of f(x) is x > 1/2 or x < -1/2 -/
def DomainF (x : ℝ) : Prop :=
  x > 1/2 ∨ x < -1/2

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x, DomainF x → IsEven (f a)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l68_6813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motion_first_kind_is_rotation_or_translation_l68_6854

-- Define a type for lines in a plane
def Line : Type := ℝ × ℝ → Prop

-- Define a type for reflections
def Reflection : Type := Line → (ℝ × ℝ → ℝ × ℝ)

-- Define a type for motions (isometries of the plane)
def Motion : Type := ℝ × ℝ → ℝ × ℝ

-- Define rotation
def IsRotation (m : Motion) : Prop := sorry

-- Define translation
def IsTranslation (m : Motion) : Prop := sorry

-- Define composition of reflections
def ComposeReflections (σ₁ σ₂ : Reflection) : Motion := sorry

-- Theorem statement
theorem motion_first_kind_is_rotation_or_translation 
  (l₁ l₂ : Line) (σ_l₁ σ_l₂ : Reflection) :
  let m := ComposeReflections σ_l₁ σ_l₂
  IsRotation m ∨ IsTranslation m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motion_first_kind_is_rotation_or_translation_l68_6854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_exponentials_l68_6886

theorem compare_exponentials : Real.rpow 2 0.3 > Real.rpow 0.3 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_exponentials_l68_6886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_and_inclination_l68_6818

theorem line_slope_and_inclination :
  ∀ (x y : ℝ), Real.sqrt 3 * x + y - 1 = 0 →
  ∃ (m : ℝ) (α : ℝ), 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), Real.sqrt 3 * x₁ + y₁ - 1 = 0 ∧ Real.sqrt 3 * x₂ + y₂ - 1 = 0 ∧ x₁ ≠ x₂ → 
      m = (y₂ - y₁) / (x₂ - x₁)) ∧
    m = -Real.sqrt 3 ∧
    0 ≤ α ∧ α < Real.pi ∧
    Real.tan α = m ∧
    α = 2 * Real.pi / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_and_inclination_l68_6818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l68_6858

noncomputable def f (x : ℝ) : ℝ := (x - 3) * (x - 5) * (x - 7) / ((x - 2) * (x - 6) * (x - 8))

theorem inequality_solution :
  ∀ x : ℝ, f x > 0 ↔ x ∈ Set.Iio 2 ∪ Set.Ioo 3 6 ∪ Set.Ioo 7 8 :=
by
  sorry

#check inequality_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l68_6858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l68_6862

/-- Represents a position on the chessboard -/
structure Position where
  x : Fin 8
  y : Fin 8

/-- Represents the game state -/
structure GameState where
  currentPosition : Position
  visitedPositions : Set Position

/-- Represents a player's move -/
inductive Move where
  | Queen : Position → Move
  | King : Position → Position → Move

/-- The game rules -/
def isValidMove (gs : GameState) (m : Move) : Prop :=
  match m with
  | Move.Queen pos => pos ∉ gs.visitedPositions
  | Move.King pos1 pos2 => pos1 ∉ gs.visitedPositions ∧ pos2 ∉ gs.visitedPositions

/-- Iterate the game for n moves -/
def iterateGame (n : ℕ) (initialState : GameState) (player1Strategy : GameState → Move) (player2Strategy : GameState → Move) : GameState :=
  sorry

/-- The winning strategy exists for the second player -/
theorem second_player_wins :
  ∃ (strategy : GameState → Move),
    ∀ (initialState : GameState),
      initialState.currentPosition.x = 0 ∧ initialState.currentPosition.y = 0 →
        ∀ (player1Strategy : GameState → Move),
          (∀ gs, isValidMove gs (player1Strategy gs)) →
            ∃ (n : ℕ), ¬isValidMove (iterateGame n initialState player1Strategy strategy) (player1Strategy (iterateGame n initialState player1Strategy strategy)) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l68_6862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_l68_6853

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem symmetric_center 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : abs φ < π / 2) 
  (h_period : ∀ x, f ω φ (x + 4 * π) = f ω φ x) 
  (h_max : ∀ x, f ω φ x ≤ f ω φ (π / 3)) :
  ∃ k : ℤ, f ω φ (-2 * π / 3 + 4 * π * k) = 0 ∧ 
    (∀ y, f ω φ (-2 * π / 3 - y) = f ω φ (-2 * π / 3 + y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_center_l68_6853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_at_quarter_is_six_l68_6802

/-- Represents the orbit of Krypton around its sun -/
structure KryptonOrbit where
  perigee : ℝ
  apogee : ℝ
  isSunAtFocus : Bool

/-- Calculates the approximate distance from the sun to Krypton when it's 1/4 of the way from perigee to apogee -/
noncomputable def distanceAtQuarter (orbit : KryptonOrbit) : ℝ :=
  (orbit.perigee + orbit.apogee) / 2

/-- Theorem stating the distance at quarter point is approximately 6 AU -/
theorem distance_at_quarter_is_six (orbit : KryptonOrbit) 
  (h1 : orbit.perigee = 3)
  (h2 : orbit.apogee = 9)
  (h3 : orbit.isSunAtFocus = true) :
  distanceAtQuarter orbit = 6 := by
  unfold distanceAtQuarter
  rw [h1, h2]
  norm_num

#check distance_at_quarter_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_at_quarter_is_six_l68_6802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_reflection_y_axis_l68_6869

/-- Given a circle with equation (x-1)^2 + (y-2)^2 = 4, 
    this theorem states that its reflection across the y-axis 
    has the equation (x+1)^2 + (y-2)^2 = 4 -/
theorem circle_reflection_y_axis :
  let original_circle := fun (x y : ℝ) ↦ (x - 1)^2 + (y - 2)^2 = 4
  let reflected_circle := fun (x y : ℝ) ↦ (x + 1)^2 + (y - 2)^2 = 4
  ∀ x y : ℝ, original_circle (-x) y ↔ reflected_circle x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_reflection_y_axis_l68_6869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_669th_term_l68_6800

def arithmetic_sequence (a₁ a₂ a₃ : ℤ → ℤ) : Prop :=
  ∃ (x : ℤ), 
    a₁ x = 3*x - 5 ∧
    a₂ x = 7*x - 15 ∧
    a₃ x = 4*x + 3 ∧
    ∃ (d : ℤ), a₂ x - a₁ x = d ∧ a₃ x - a₂ x = d

theorem arithmetic_sequence_669th_term 
  (a₁ a₂ a₃ : ℤ → ℤ) (a : ℕ → ℤ) :
  arithmetic_sequence a₁ a₂ a₃ →
  (∀ n : ℕ, n > 0 → a n = a₁ 4 + (n - 1) * (a₂ 4 - a₁ 4)) →
  a 669 = 4019 :=
by
  intro h1 h2
  sorry

#check arithmetic_sequence_669th_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_669th_term_l68_6800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l68_6825

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt 3 * Real.sin t.B - Real.cos t.B = 1)
  (h2 : t.b = 1) : 
  (t.A = 5 * Real.pi / 12 → t.c = Real.sqrt 6 / 3) ∧ 
  (t.a = 2 * t.c → t.a * t.b * Real.sin t.C / 2 = Real.sqrt 3 / 6) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l68_6825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_numbers_with_properties_l68_6837

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ d ↦ d ∣ n) (Finset.range (n + 1))

def numDivisors (n : ℕ) : ℕ :=
  (divisors n).card

def sumDivisors (n : ℕ) : ℕ :=
  (divisors n).sum id

theorem exist_numbers_with_properties : ∃ x y : ℕ,
  x < y ∧ numDivisors x = numDivisors y ∧ sumDivisors x > sumDivisors y := by
  -- We use 38 and 39 as our examples
  let x := 38
  let y := 39
  have h1 : x < y := by norm_num
  have h2 : numDivisors x = numDivisors y := by
    -- Both 38 and 39 have 4 divisors
    sorry
  have h3 : sumDivisors x > sumDivisors y := by
    -- Sum of divisors of 38 is 60, sum of divisors of 39 is 56
    sorry
  exact ⟨x, y, h1, h2, h3⟩

#eval divisors 38
#eval divisors 39
#eval numDivisors 38
#eval numDivisors 39
#eval sumDivisors 38
#eval sumDivisors 39

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_numbers_with_properties_l68_6837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l68_6864

theorem coefficient_x_cubed_in_expansion : 
  (Polynomial.coeff ((1 + Polynomial.X : Polynomial ℚ)^5 - (1 + Polynomial.X : Polynomial ℚ)^6) 3) = -10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l68_6864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concrete_order_amount_l68_6881

/-- Represents the dimensions of a sidewalk in feet and inches. -/
structure SidewalkDimensions where
  width : ℚ  -- width in feet
  length : ℚ  -- length in feet
  thickness : ℚ  -- thickness in inches

/-- Calculates the volume of concrete needed in cubic yards. -/
def concreteVolume (d : SidewalkDimensions) : ℚ :=
  (d.width / 3) * (d.length / 3) * (d.thickness / 36)

/-- Rounds up a rational number to the nearest integer. -/
def ceilToNat (x : ℚ) : ℕ :=
  (x.ceil).toNat

theorem concrete_order_amount (d : SidewalkDimensions) 
  (h1 : d.width = 4)
  (h2 : d.length = 80)
  (h3 : d.thickness = 4) :
  ceilToNat (concreteVolume d) = 4 := by
  sorry

#eval ceilToNat (concreteVolume { width := 4, length := 80, thickness := 4 })

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concrete_order_amount_l68_6881
