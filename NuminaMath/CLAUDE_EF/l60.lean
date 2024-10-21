import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l60_6019

noncomputable def f (a : ℕ+) (x : ℝ) : ℝ := x + 2 * (a : ℝ) / x

theorem problem_solution (a : ℕ+) (h : (a : ℝ)^2 - (a : ℝ) < 2) :
  (a = 1) ∧
  (Set.range (f a) = {y | y ≤ -2 * Real.sqrt 2 ∨ y ≥ 2 * Real.sqrt 2}) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l60_6019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_x2_minus_2y2_eq_1_l60_6036

theorem infinitely_many_solutions_x2_minus_2y2_eq_1 :
  ∃ f : ℕ → ℕ × ℕ, Function.Injective f ∧
    ∀ n : ℕ, let (x, y) := f n; x^2 - 2*y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_x2_minus_2y2_eq_1_l60_6036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_start_number_l60_6059

theorem range_start_number (end_number : ℕ) (count_even_not_div_3 : ℕ) :
  end_number = 140 →
  count_even_not_div_3 = 44 →
  ∃ start_number : ℕ,
    start_number = 10 ∧
    (Finset.filter (λ n => n % 2 = 0 ∧ n % 3 ≠ 0) (Finset.Icc start_number end_number)).card = count_even_not_div_3 :=
by
  intro h_end h_count
  use 10
  apply And.intro rfl
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_start_number_l60_6059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_eq_sin_2x_plus_pi_4_l60_6002

theorem cos_2x_eq_sin_2x_plus_pi_4 (x : ℝ) : 
  Real.cos (2 * x) = Real.sin (2 * (x + Real.pi / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_eq_sin_2x_plus_pi_4_l60_6002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l60_6009

def A : Set ℤ := {x | |x| < 4}
def B : Set ℤ := {x | x ≥ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l60_6009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_implies_a_range_l60_6055

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := 2^x - a^2 - a

-- State the theorem
theorem zero_in_interval_implies_a_range (a : ℝ) :
  (a > 0) →
  (∃ x : ℝ, x ≤ 1 ∧ f x a = 0) →
  0 < a ∧ a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_implies_a_range_l60_6055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_for_inequality_l60_6045

theorem smallest_b_for_inequality : 
  ∃ b : ℕ, b = 4 ∧ ∀ n : ℕ, 27^n > 3^9 ↔ n ≥ b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_for_inequality_l60_6045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandbag_filling_percentage_l60_6004

/-- Represents the weight of the sandbag contents in pounds -/
structure SandbagContents where
  sand : ℝ
  filling : ℝ

/-- Calculates the percentage of filling material in the sandbag contents -/
noncomputable def fillingPercentage (contents : SandbagContents) : ℝ :=
  contents.filling / (contents.sand + contents.filling)

theorem sandbag_filling_percentage :
  ∀ (contents : SandbagContents),
    contents.sand + contents.filling = 30 →
    contents.filling = 1.4 * contents.sand →
    fillingPercentage contents = 7 / 12 := by
  sorry

#eval "Sandbag filling percentage theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandbag_filling_percentage_l60_6004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_trajectory_l60_6022

noncomputable def distance_point_to_point (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

def distance_point_to_vertical_line (x y a : ℝ) : ℝ :=
  |x - a|

def on_parabola (x y : ℝ) : Prop :=
  y^2 = 8*x

theorem parabola_trajectory (x y : ℝ) :
  distance_point_to_point x y 2 0 = distance_point_to_vertical_line x y (-2) →
  on_parabola x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_trajectory_l60_6022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l60_6040

/-- Curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 2*ρ*Real.cos θ - 2 = 0

/-- Line l in polar coordinates -/
def line_l (ρ θ : ℝ) : Prop :=
  ρ * Real.cos (θ - Real.pi/6) = 3*Real.sqrt 3

/-- Ray OT -/
def ray_OT (ρ θ : ℝ) : Prop :=
  θ = Real.pi/3 ∧ ρ > 0

/-- Point A: intersection of curve C and ray OT -/
noncomputable def point_A : ℝ × ℝ :=
  (2, Real.pi/3)

/-- Point B: intersection of line l and ray OT -/
noncomputable def point_B : ℝ × ℝ :=
  (6, Real.pi/3)

/-- Length of segment AB -/
noncomputable def length_AB : ℝ :=
  point_B.1 - point_A.1

theorem intersection_segment_length :
  curve_C point_A.1 point_A.2 ∧
  line_l point_B.1 point_B.2 ∧
  ray_OT point_A.1 point_A.2 ∧
  ray_OT point_B.1 point_B.2 →
  length_AB = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l60_6040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_downstream_time_difference_l60_6057

/-- Given a boatsman's speed in still water and a river's speed, calculates the
    difference in time taken to travel a fixed distance upstream versus downstream. -/
noncomputable def time_difference (boatsman_speed river_speed distance : ℝ) : ℝ :=
  let downstream_speed := boatsman_speed + river_speed
  let upstream_speed := boatsman_speed - river_speed
  distance / upstream_speed - distance / downstream_speed

/-- Proves that the difference in time taken to travel 40 km upstream versus downstream
    is 6 hours, given a boatsman's speed of 7 km/hr in still water and a river speed of 3 km/hr. -/
theorem upstream_downstream_time_difference :
  time_difference 7 3 40 = 6 := by
  -- Unfold the definition of time_difference
  unfold time_difference
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_downstream_time_difference_l60_6057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emmas_age_ratio_l60_6012

theorem emmas_age_ratio (E M : ℚ) : 
  (∃ (children_ages : Fin 4 → ℚ), 
    E = (Finset.sum (Finset.univ : Finset (Fin 4)) (λ i => children_ages i)) ∧ 
    (E - M) = 3 * ((Finset.sum (Finset.univ : Finset (Fin 4)) (λ i => children_ages i)) - 4 * M)) → 
  E / M = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emmas_age_ratio_l60_6012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l60_6088

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)

def IsSymmetryCenter (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (p.1 + x) = f (p.1 - x)

theorem symmetry_center_of_f (φ : ℝ) (h1 : |φ| < π/2) (h2 : f 0 φ = Real.sqrt 3) :
  IsSymmetryCenter (f · φ) (-π/6, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l60_6088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_ellipse_final_equation_l60_6054

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The theorem stating the properties of the given ellipse -/
theorem ellipse_properties (e : Ellipse) : 
  let f1 : Point := ⟨-1, 0⟩
  let f2 : Point := ⟨1, 0⟩
  let p : Point := ⟨4/3, 1/3⟩
  ellipse_equation e f1 ∧ 
  ellipse_equation e f2 ∧ 
  ellipse_equation e p ∧
  distance f1 f2 = 2 →
  e.a = Real.sqrt 2 ∧ e.b = 1 :=
by sorry

/-- The main theorem to prove -/
theorem ellipse_final_equation (e : Ellipse) :
  let f1 : Point := ⟨-1, 0⟩
  let f2 : Point := ⟨1, 0⟩
  let p : Point := ⟨4/3, 1/3⟩
  ellipse_equation e f1 ∧ 
  ellipse_equation e f2 ∧ 
  ellipse_equation e p ∧
  distance f1 f2 = 2 →
  ∀ (x y : ℝ), x^2 / 2 + y^2 = 1 ↔ ellipse_equation e ⟨x, y⟩ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_ellipse_final_equation_l60_6054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_resistor_combinations_count_l60_6067

/-- Represents the possible combination methods for resistors -/
inductive CombinationMethod
  | Series
  | Parallel
  | ShortCircuit

/-- Represents a resistor with a rational resistance value -/
structure Resistor where
  resistance : ℚ
deriving BEq, Repr

/-- Combines two resistors using a given method -/
def combineResistors (method : CombinationMethod) (r1 r2 : Resistor) : Resistor :=
  match method with
  | CombinationMethod.Series => ⟨r1.resistance + r2.resistance⟩
  | CombinationMethod.Parallel => ⟨(r1.resistance * r2.resistance) / (r1.resistance + r2.resistance)⟩
  | CombinationMethod.ShortCircuit => if r1.resistance ≤ r2.resistance then r1 else r2

/-- Performs a single combination step on a list of resistors -/
def combinationStep (resistors : List Resistor) : List Resistor :=
  match resistors with
  | [] => []
  | [r] => [r]
  | r1 :: r2 :: rest =>
      (combineResistors CombinationMethod.Series r1 r2) ::
      (combineResistors CombinationMethod.Parallel r1 r2) ::
      (combineResistors CombinationMethod.ShortCircuit r1 r2) ::
      rest

/-- Performs n combination steps on a list of resistors -/
def performCombinations (n : ℕ) (resistors : List Resistor) : List Resistor :=
  match n with
  | 0 => resistors
  | n + 1 => performCombinations n (combinationStep resistors)

/-- The main theorem statement -/
theorem resistor_combinations_count :
  let initialResistors := List.replicate 24 ⟨1⟩
  let finalResistors := performCombinations 23 initialResistors
  List.length (List.eraseDups finalResistors) = 1015080877 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_resistor_combinations_count_l60_6067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_six_l60_6007

-- Define the rectangle and points
structure Rectangle where
  width : ℝ
  height : ℝ
  x : ℝ
  y : ℝ

def Point : Type := ℝ × ℝ

-- Define the rectangle DEFA
def DEFA : Rectangle := ⟨2, 5, 0, 0⟩

-- Define points D, E, F, A, C, B
def D (r : Rectangle) : Point := (r.x, r.y + r.height)
def E (r : Rectangle) : Point := (r.x, r.y)
def F (r : Rectangle) : Point := (r.x + r.width, r.y)
def A (r : Rectangle) : Point := (r.x + r.width, r.y + r.height)
def C (r : Rectangle) : Point := (r.x, r.y + r.height - 2)
def B (r : Rectangle) : Point := (r.x + r.width, r.y + r.height - 2)

-- Define the function to calculate the area of a triangle using Shoelace formula
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  1/2 * abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p1.2) - (p1.2 * p2.1 + p2.2 * p3.1 + p3.2 * p1.1))

-- Define the total shaded area
noncomputable def shadedArea (r : Rectangle) : ℝ :=
  triangleArea (E r) (C r) (F r) + triangleArea (E r) (F r) (B r)

-- Theorem statement
theorem shaded_area_is_six :
  shadedArea DEFA = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_six_l60_6007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l60_6086

/-- An arithmetic sequence satisfying the given conditions -/
noncomputable def arithmetic_sequence (n : ℕ) : ℝ :=
  3 * n - 1

/-- The b_n sequence defined in terms of a_n -/
noncomputable def b_sequence (n : ℕ) : ℝ :=
  1 / (arithmetic_sequence n * arithmetic_sequence (n + 1))

/-- The sum of the first n terms of b_sequence -/
noncomputable def T (n : ℕ) : ℝ :=
  n / (6 * n + 4)

theorem arithmetic_sequence_properties :
  (arithmetic_sequence 1 + arithmetic_sequence 2 = 7) ∧
  (arithmetic_sequence 3 = 8) ∧
  (∀ n : ℕ, arithmetic_sequence n = 3 * n - 1) ∧
  (∀ n : ℕ, T n = n / (6 * n + 4)) ∧
  (T 1 * T 10 = T 2 * T 2) ∧
  (∀ m n : ℕ, 1 < m → m < n → T 1 * T n = T m * T m → m = 2 ∧ n = 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l60_6086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_factors_multiples_l60_6031

theorem division_factors_multiples : ∀ a b c : ℕ, 
  a / b = c → 
  (c ∣ a ∧ b ∣ a) ∧ 
  (a % c = 0 ∧ a % b = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_factors_multiples_l60_6031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_center_l60_6011

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 + Real.cos x ^ 2 + (1/4) * Real.sin (2*x) * Real.cos (2*x)

theorem f_symmetry_center : 
  ∀ (x : ℝ), f (x + π/8) + f (-x - π/8) = 2 * f (-π/16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_center_l60_6011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_inequality_condition_l60_6010

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x - 3*a|

-- Theorem for part I
theorem min_value_condition (a : ℝ) :
  (∃ (x : ℝ), f a x = 2 ∧ ∀ (y : ℝ), f a y ≥ 2) ↔ (a = 1 ∨ a = -1) := by sorry

-- Theorem for part II
theorem inequality_condition (m : ℝ) :
  (∀ (x : ℝ), ∃ (a : ℝ), a ∈ Set.Icc (-1) 1 ∧ m^2 - |m| - f a x < 0) ↔ 
  (-2 < m ∧ m < 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_inequality_condition_l60_6010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_eq_2592_l60_6050

def is_valid_number (n : ℕ) : Bool :=
  2000 ≤ n && n ≤ 9999 && 
  (n % 2 = 0) &&
  (let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
   List.Nodup digits)

def count_valid_numbers : ℕ := 
  (List.range 8000).filter (λ n => is_valid_number (n + 2000)) |>.length

theorem count_valid_numbers_eq_2592 : count_valid_numbers = 2592 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_eq_2592_l60_6050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l60_6078

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos (2 * x)

theorem f_properties :
  (∃ p > 0, ∀ x, f (x + p) = f x ∧ ∀ q ∈ Set.Ioo 0 p, ∃ y, f (y + q) ≠ f y) ∧
  (∀ y, f (π/3 + y) = f (π/3 - y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l60_6078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l60_6025

noncomputable def f (x : ℝ) := Real.sqrt (x * (60 - x)) + Real.sqrt (x * (5 - x))

theorem max_value_of_f :
  ∃ (x₀ : ℝ), x₀ ∈ Set.Icc 0 5 ∧ 
  (∀ x ∈ Set.Icc 0 5, f x ≤ f x₀) ∧
  x₀ = 60 / 13 ∧ f x₀ = 10 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l60_6025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l60_6041

/-- Given a parabola and a hyperbola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, y^2 = 4*x → (1 : ℝ) = a) →  -- Focus of parabola is right vertex of hyperbola
  (∀ x y : ℝ, y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) →  -- Asymptote equations
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l60_6041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_not_in_third_quadrant_l60_6072

-- Define the two lines
noncomputable def line1 (m : ℝ) (x : ℝ) : ℝ := x + 2*m
noncomputable def line2 (x : ℝ) : ℝ := -x + 4

-- Define the intersection point
noncomputable def intersection (m : ℝ) : ℝ × ℝ :=
  let x := (4 - 2*m) / 2
  let y := line1 m x
  (x, y)

-- Theorem: The intersection point is never in the third quadrant
theorem intersection_not_in_third_quadrant (m : ℝ) :
  let (x, y) := intersection m
  ¬(x < 0 ∧ y < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_not_in_third_quadrant_l60_6072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l60_6048

def spinner_numbers : List ℕ := [3, 6, 1, 4, 5, 2, 7, 9]

def count_primes (l : List ℕ) : ℕ :=
  l.filter (fun n => Nat.Prime n) |>.length

theorem spinner_prime_probability : 
  (count_primes spinner_numbers : ℚ) / (spinner_numbers.length : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l60_6048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_garden_area_l60_6056

/-- Represents a circular garden with a straight path through its center -/
structure GardenWithPath where
  diameter : ℝ
  pathWidth : ℝ

/-- Calculates the remaining area of the garden not covered by the path -/
noncomputable def remainingArea (g : GardenWithPath) : ℝ :=
  (Real.pi * (g.diameter / 2)^2) - (g.diameter * g.pathWidth)

/-- Theorem stating the remaining area for a specific garden configuration -/
theorem specific_garden_area :
  let g : GardenWithPath := { diameter := 10, pathWidth := 4 }
  remainingArea g = 25 * Real.pi - 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_garden_area_l60_6056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_l60_6073

def z₁ : ℂ := 3 + 2*Complex.I
def z₂ (m : ℝ) : ℂ := 1 + m*Complex.I

theorem pure_imaginary_product (m : ℝ) : 
  (z₁ * z₂ m).re = 0 → m = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_l60_6073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_spray_economy_l60_6042

/-- Represents the size of a cleaning spray container -/
inductive Size
| Compact
| Regular
| Jumbo

/-- Represents the cost and quantity of a cleaning spray container -/
structure Container where
  size : Size
  cost : ℚ
  quantity : ℚ

/-- Calculates the cost per unit quantity -/
def costPerUnit (c : Container) : ℚ := c.cost / c.quantity

/-- Determines if one container is more economical than another -/
def moreEconomical (c1 c2 : Container) : Prop :=
  costPerUnit c1 < costPerUnit c2

theorem cleaning_spray_economy (c : Container) (r : Container) (j : Container)
  (hc : c.size = Size.Compact)
  (hr : r.size = Size.Regular)
  (hj : j.size = Size.Jumbo)
  (h_r_cost : r.cost = 14/10 * c.cost)
  (h_j_quantity : j.quantity = 25/10 * c.quantity)
  (h_r_quantity : r.quantity = 75/100 * j.quantity)
  (h_j_cost : j.cost = 12/10 * r.cost) :
  moreEconomical j r ∧ moreEconomical r c := by
  sorry

#check cleaning_spray_economy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_spray_economy_l60_6042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_unit_vector_coordinates_l60_6063

-- Define points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

-- Define vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the magnitude of vector AB
noncomputable def magnitudeAB : ℝ := Real.sqrt (vectorAB.1^2 + vectorAB.2^2)

-- Define the unit vector in the opposite direction of AB
noncomputable def oppositeUnitVector : ℝ × ℝ := (-vectorAB.1 / magnitudeAB, -vectorAB.2 / magnitudeAB)

-- Theorem statement
theorem opposite_unit_vector_coordinates :
  oppositeUnitVector = (-3/5, 4/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_unit_vector_coordinates_l60_6063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_drained_in_four_minutes_l60_6074

/-- Represents the fraction of a tub's content drained in a given time -/
noncomputable def fraction_drained (time : ℝ) (total_time : ℝ) : ℝ :=
  time / total_time

/-- The time it takes to drain a certain fraction of the tub's content -/
def drain_time : ℝ := 4

/-- The additional time it takes to completely empty the tub -/
def additional_time : ℝ := 1

/-- The total time it takes to completely empty the tub -/
def total_drain_time : ℝ := drain_time + additional_time

theorem fraction_drained_in_four_minutes :
  fraction_drained drain_time total_drain_time = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_drained_in_four_minutes_l60_6074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_identification_l60_6065

-- Define the concept of a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the given functions
noncomputable def f1 (x : ℝ) : ℝ := 2^x
noncomputable def f2 (x : ℝ) : ℝ := x^2
noncomputable def f3 (x : ℝ) : ℝ := 1/x
noncomputable def f4 (x : ℝ) : ℝ := x^2 + 1
noncomputable def f5 (x : ℝ) : ℝ := 3/(x^2)

-- Theorem statement
theorem power_function_identification :
  isPowerFunction f2 ∧ isPowerFunction f3 ∧
  ¬isPowerFunction f1 ∧ ¬isPowerFunction f4 ∧ ¬isPowerFunction f5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_identification_l60_6065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gumball_average_range_l60_6082

theorem gumball_average_range (x : ℕ → ℝ) (h : max (x 1) (x 0) - min (x 0) (x 1) = 18) :
  let avg := λ i => (16 + 12 + x i) / 3
  avg 1 - avg 0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gumball_average_range_l60_6082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person2_more_heads_prob_l60_6068

/-- The probability of getting heads in a single fair coin flip -/
noncomputable def probHeads : ℝ := 1/2

/-- The number of coin flips for Person 1 -/
def flips1 : ℕ := 10

/-- The number of coin flips for Person 2 -/
def flips2 : ℕ := 11

/-- The probability that Person 2 gets more heads than Person 1 -/
noncomputable def probPerson2MoreHeads : ℝ := 1/2

/-- Theorem: The probability that Person 2 gets more heads in 11 flips
    than Person 1 in 10 flips is 1/2 -/
theorem person2_more_heads_prob :
  probPerson2MoreHeads = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_person2_more_heads_prob_l60_6068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_t_equals_two_inequality_for_ln_x_inequality_implies_upper_bound_l60_6032

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def F (t : ℝ) (x : ℝ) : ℝ := t * f x

def g (x : ℝ) : ℝ := x^2 - 1

-- Part 1
theorem common_tangent_implies_t_equals_two :
  ∀ t : ℝ, (∃ k : ℝ, (deriv (F t)) 1 = (deriv g) 1) → t = 2 := by sorry

-- Part 2
theorem inequality_for_ln_x :
  ∀ x : ℝ, x > 0 → |f x - x| > f x / x + 1/2 := by sorry

-- Part 3
theorem inequality_implies_upper_bound :
  ∀ a m x : ℝ, m ∈ Set.Icc 0 (3/2) → x ∈ Set.Icc 1 (Real.exp 2) →
  (m * f x ≥ a + x) → a ≤ -(Real.exp 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_t_equals_two_inequality_for_ln_x_inequality_implies_upper_bound_l60_6032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_max_value_l60_6091

theorem trig_expression_max_value (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (Real.sin x + (1 / Real.sin x) + Real.tan x)^2 + (Real.cos x + (1 / Real.cos x) + (1 / Real.tan x))^2 ≤ 20 ∧
  ∃ y, 0 < y ∧ y < π / 2 ∧ (Real.sin y + (1 / Real.sin y) + Real.tan y)^2 + (Real.cos y + (1 / Real.cos y) + (1 / Real.tan y))^2 = 20 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_max_value_l60_6091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_iff_a_in_range_l60_6023

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4*a*x else (2*a + 3)*x - 4*a + 5

theorem increasing_f_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (1/2 ≤ a ∧ a ≤ 3/2) := by
  sorry

#check increasing_f_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_iff_a_in_range_l60_6023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_period_divisor_at_40_smallest_divisor_is_40_l60_6092

def f (z : ℂ) : ℂ := z^11 + z^10 + z^8 + z^6 + z^3 + z + 1

theorem smallest_divisor_period : 
  ∀ k : ℕ, (∀ z : ℂ, f z ∣ (z^k - 1)) → k ≥ 40 :=
by sorry

theorem divisor_at_40 : 
  ∀ z : ℂ, f z ∣ (z^40 - 1) :=
by sorry

theorem smallest_divisor_is_40 : 
  (∀ z : ℂ, f z ∣ (z^40 - 1)) ∧ 
  (∀ k : ℕ, k < 40 → ∃ z : ℂ, ¬(f z ∣ (z^k - 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_period_divisor_at_40_smallest_divisor_is_40_l60_6092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_similarity_l60_6085

/-- An isosceles triangle with a specified angle --/
structure IsoscelesTriangle :=
  (angle : ℝ)
  (is_isosceles : True)  -- We assume the triangle is isosceles

/-- Predicate to check if two isosceles triangles are necessarily similar --/
def necessarily_similar (t1 t2 : IsoscelesTriangle) : Prop :=
  (t1.angle = 60 ∧ t2.angle = 60) ∨
  (t1.angle = 90 ∧ t2.angle = 90) ∨
  (t1.angle = 120 ∧ t2.angle = 120)

/-- Theorem stating that among 60°, 45°, 90°, and 120°, only 45° isosceles triangles are not necessarily similar --/
theorem isosceles_similarity :
  ∀ (t1 t2 : IsoscelesTriangle),
    (t1.angle ∈ ({60, 45, 90, 120} : Set ℝ) ∧ t2.angle ∈ ({60, 45, 90, 120} : Set ℝ)) →
    (¬ necessarily_similar t1 t2 ↔ (t1.angle = 45 ∧ t2.angle = 45)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_similarity_l60_6085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l60_6030

-- Define the center and radius of the circle
def center : ℝ × ℝ := (-3, 4)
noncomputable def radius : ℝ := Real.sqrt 3

-- State the theorem
theorem circle_equation (x y : ℝ) :
  (x - center.fst)^2 + (y - center.snd)^2 = radius^2 ↔ 
  (x + 3)^2 + (y - 4)^2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l60_6030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l60_6018

def s : ℕ → ℚ
  | 0 => 2  -- Add a case for 0 to cover all natural numbers
  | 1 => 2
  | n + 2 => if (n + 2) % 3 = 0 then 2 + s ((n + 2) / 3) else 2 / s (n + 1)

theorem sequence_problem (n : ℕ) (h : n > 0) (h_eq : s n = 13 / 29) : n = 154305 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l60_6018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_circles_construction_l60_6016

/-- Regular pentagon with vertices P and R -/
structure RegularPentagon where
  P : EuclideanSpace ℝ (Fin 2)
  R : EuclideanSpace ℝ (Fin 2)

/-- Circle in a 2D plane -/
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

/-- Construction of circles k1 to k6 -/
def constructCircles (pentagon : RegularPentagon) : List Circle :=
  sorry

/-- Theorem: The remaining two circles have centers P and R with radius ((1 + √5) / 2) * PR -/
theorem remaining_circles_construction (pentagon : RegularPentagon) :
  let circles := constructCircles pentagon
  let PR := dist pentagon.P pentagon.R
  let goldenRatio := (1 + Real.sqrt 5) / 2
  ∃ (k7 k8 : Circle),
    k7.center = pentagon.P ∧
    k8.center = pentagon.R ∧
    k7.radius = goldenRatio * PR ∧
    k8.radius = goldenRatio * PR :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_circles_construction_l60_6016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_equals_zero_l60_6099

-- Define the function f(x) = x^2 - ln(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := 2*x - 1/x

-- Theorem statement
theorem tangent_line_b_equals_zero (x : ℝ) (h1 : x > 0) (h2 : f' x = 1) :
  ∃ b : ℝ, f x = x - b ∧ b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_equals_zero_l60_6099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_750_equals_half_l60_6066

-- Define the period of the sine function
def sine_period : ℝ := 360

-- Define the angle in degrees
def angle : ℝ := 750

-- Define the relationship between 750° and 30°
axiom angle_decomposition : angle = 2 * sine_period + 30

-- Define the value of sin 30°
axiom sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2

-- Theorem to prove
theorem sin_750_equals_half : Real.sin (angle * Real.pi / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_750_equals_half_l60_6066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l60_6035

/-- A point on a parabola with specific properties -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2*x
  focus_axis_ratio : (x + 1/2) / y = 9/4
  focus_distance : x + 1/2 > 2

/-- The distance from a point on a parabola to the origin -/
noncomputable def distance_to_origin (p : ParabolaPoint) : ℝ :=
  Real.sqrt (p.x^2 + p.y^2)

/-- Theorem: The distance from the specified point to the origin is 4√5 -/
theorem parabola_point_distance (p : ParabolaPoint) : 
  distance_to_origin p = 4 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l60_6035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_relationships_l60_6098

-- Define a structure for Triangle
structure Triangle where
  -- You can add more properties here if needed
  mk :: (dummy : Unit)

-- Define the propositions
def congruent (t1 t2 : Triangle) : Prop := sorry
def equal_areas (t1 t2 : Triangle) : Prop := sorry

-- Define the original proposition
def original_proposition : Prop :=
  ∀ t1 t2 : Triangle, congruent t1 t2 → equal_areas t1 t2

-- Define the converse, inverse, and contrapositive
def converse : Prop :=
  ∀ t1 t2 : Triangle, equal_areas t1 t2 → congruent t1 t2

def inverse : Prop :=
  ∀ t1 t2 : Triangle, ¬(congruent t1 t2) → ¬(equal_areas t1 t2)

def contrapositive : Prop :=
  ∀ t1 t2 : Triangle, ¬(equal_areas t1 t2) → ¬(congruent t1 t2)

-- Theorem stating the relationships
theorem proposition_relationships :
  (original_proposition ↔ contrapositive) ∧
  (converse ↔ inverse) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_relationships_l60_6098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_some_are_equilateral_l60_6080

-- Define a type for triangles
variable (Triangle : Type)

-- Define a predicate for equilateral triangles
variable (isEquilateral : Triangle → Prop)

-- Define the proposition P
def P (Triangle : Type) (isEquilateral : Triangle → Prop) : Prop := 
  ∃ t : Triangle, isEquilateral t

-- Theorem: The negation of P is equivalent to "No triangles are equilateral"
theorem negation_of_some_are_equilateral (Triangle : Type) (isEquilateral : Triangle → Prop) :
  ¬(P Triangle isEquilateral) ↔ ∀ t : Triangle, ¬(isEquilateral t) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_some_are_equilateral_l60_6080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_NC_l60_6015

noncomputable section

open Real

-- Define the square and semicircle
def square_side_length : ℝ := 8

-- Define the angle PNQ
def angle_PNQ : ℝ := π / 3  -- 60° in radians

-- Define the coordinates of points
def P : ℝ × ℝ := (0, 0)
def Q : ℝ × ℝ := (square_side_length, 0)
def C : ℝ × ℝ := (square_side_length / 2, 0)
def N : ℝ × ℝ := (3 * square_side_length / 4, square_side_length * sqrt 3 / 4)

-- Theorem statement
theorem length_of_NC :
  let square_side_length : ℝ := 8
  let angle_PNQ : ℝ := π / 3
  let P : ℝ × ℝ := (0, 0)
  let Q : ℝ × ℝ := (square_side_length, 0)
  let C : ℝ × ℝ := (square_side_length / 2, 0)
  let N : ℝ × ℝ := (3 * square_side_length / 4, square_side_length * sqrt 3 / 4)
  sqrt ((N.1 - C.1)^2 + (N.2 - C.2)^2) = 2 * sqrt 3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_NC_l60_6015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invariant_interval_implies_m_eq_neg_one_l60_6089

/-- A function with an invariant interval -/
def HasInvariantInterval (f : ℝ → ℝ) (A : Set ℝ) : Prop :=
  (∀ x ∈ A, f x ∈ A) ∧ (∀ y ∈ A, ∃ x ∈ A, f x = y)

/-- The function g(x) = -x + m + exp(x) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := -x + m + Real.exp x

/-- The interval [0, +∞) -/
def nonNegativeReals : Set ℝ := {x : ℝ | x ≥ 0}

theorem invariant_interval_implies_m_eq_neg_one :
  (HasInvariantInterval (g m) nonNegativeReals) → m = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_invariant_interval_implies_m_eq_neg_one_l60_6089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l60_6097

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 3*a) / Real.log (Real.sin 1)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ici 2, StrictMonoOn (fun x => -(f a x)) (Set.Ici 2)) →
  a ∈ Set.Ioc (-4) 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l60_6097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apex_angle_is_sixty_degrees_l60_6047

/-- Represents a cone with its lateral surface unfolded into a semicircle -/
structure UnfoldedCone where
  /-- The radius of the semicircle formed by unfolding the lateral surface -/
  semicircle_radius : ℝ
  /-- The radius of the base of the cone -/
  base_radius : ℝ
  /-- The height of the cone -/
  height : ℝ
  /-- The lateral surface unfolds into a semicircle -/
  unfolded_is_semicircle : semicircle_radius = π * base_radius

/-- The apex angle of a cone (in radians) -/
noncomputable def apex_angle (cone : UnfoldedCone) : ℝ :=
  2 * Real.arcsin (cone.base_radius / cone.semicircle_radius)

/-- Theorem stating that the apex angle of a cone with its lateral surface
    unfolded into a semicircle is π/3 radians (60 degrees) -/
theorem apex_angle_is_sixty_degrees (cone : UnfoldedCone) :
  apex_angle cone = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apex_angle_is_sixty_degrees_l60_6047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l60_6071

theorem tangent_triangle_area (a : ℝ) : a > 0 → 
  let f (x : ℝ) := x^(-(1/2 : ℝ))
  let f' (x : ℝ) := -(1/2 : ℝ) * x^(-(3/2 : ℝ))
  let slope := f' a
  let y_intercept := f a - slope * a
  let x_intercept := -y_intercept / slope
  let triangle_area := 1/2 * x_intercept * y_intercept
  triangle_area = 18 → a = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l60_6071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AP_length_approx_l60_6084

noncomputable def triangle_AP_length (AB : ℝ) (angleA angleB : ℝ) : ℝ :=
  let BC := AB * Real.sin (angleA * Real.pi / 180) / Real.sin ((180 - angleA - angleB) * Real.pi / 180)
  let AH := AB * Real.sin (angleA * Real.pi / 180) / Real.cos (angleB * Real.pi / 180)
  let HM := BC / 2
  let PH := HM / 2
  Real.sqrt (AH^2 + PH^2)

theorem triangle_AP_length_approx :
  ∃ (AB angleA angleB : ℝ),
    AB = 12 ∧
    angleA = 45 ∧
    angleB = 60 ∧
    abs (triangle_AP_length AB angleA angleB - 17.52) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AP_length_approx_l60_6084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_is_30_l60_6064

/-- The number of students in a class with given initial average, corrected average, and a mark correction. -/
def number_of_students (initial_avg : ℚ) (corrected_avg : ℚ) (wrong_mark : ℚ) (correct_mark : ℚ) : ℕ :=
  Int.natAbs (⌊(wrong_mark - correct_mark) / (initial_avg - corrected_avg)⌋₊)

/-- Theorem stating that the number of students in the class is 30 -/
theorem class_size_is_30 :
  number_of_students 60 57.5 90 15 = 30 := by
  sorry

#eval number_of_students 60 57.5 90 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_is_30_l60_6064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_and_linear_system_l60_6060

theorem calculation_and_linear_system :
  (- Real.sqrt 16 / abs (-2) + (27 : ℝ) ^ (1/3) = 1) ∧
  (∃ x y : ℝ, x = 2 ∧ y = -3 ∧ 2 * x - y = 7 ∧ 3 * x + 2 * y = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_and_linear_system_l60_6060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_m_l60_6026

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b^2 / a^2))

/-- Theorem: For a hyperbola with equation x^2/(m+9) + y^2/9 = 1 and eccentricity 2, m = -36 -/
theorem hyperbola_eccentricity_m (m : ℝ) :
  (∀ x y : ℝ, x^2/(m+9) + y^2/9 = 1) →
  eccentricity 3 (Real.sqrt (-(m+9))) = 2 →
  m = -36 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_m_l60_6026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l60_6077

/-- The function for which we're finding vertical asymptotes -/
noncomputable def f (x : ℝ) : ℝ := (4 * x^2 - 7) / (2 * x^2 + 7 * x + 3)

/-- The denominator of the function -/
noncomputable def denom (x : ℝ) : ℝ := 2 * x^2 + 7 * x + 3

theorem vertical_asymptotes_sum :
  ∃ (a b : ℝ), (∀ x, denom x = 0 ↔ x = a ∨ x = b) ∧ a + b = -3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l60_6077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_21_is_zero_l60_6076

def sequence_u (u₁ u₂ : ℕ) : ℕ → ℕ
  | 0 => u₁
  | 1 => u₂
  | (n + 2) => min (Int.natAbs (sequence_u u₁ u₂ (n + 1) - sequence_u u₁ u₂ n)) 
                   (Int.natAbs (sequence_u u₁ u₂ (n + 1) - sequence_u u₁ u₂ n))

theorem u_21_is_zero (u₁ u₂ : ℕ) (h₁ : 1 ≤ u₁ ∧ u₁ ≤ 10000) (h₂ : 1 ≤ u₂ ∧ u₂ ≤ 10000) :
  sequence_u u₁ u₂ 21 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_21_is_zero_l60_6076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l60_6093

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * x^2

-- State the theorem
theorem f_monotonic_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1/2 → f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l60_6093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_spending_calculation_l60_6081

/-- Represents the cumulative spending in millions of dollars -/
def CumulativeSpending : ℝ → ℝ := sorry

/-- The cumulative spending at the end of February -/
def feb_spending : ℝ := 2.8

/-- The cumulative spending at the end of May -/
def may_spending : ℝ := 5.6

/-- The spending during March, April, and May -/
def spring_spending : ℝ := may_spending - feb_spending

theorem spring_spending_calculation :
  spring_spending = 2.8 := by
  -- Unfold the definition of spring_spending
  unfold spring_spending
  -- Unfold the definitions of may_spending and feb_spending
  unfold may_spending feb_spending
  -- Perform the subtraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_spending_calculation_l60_6081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_profit_l60_6083

/-- Revenue function for iPhone production --/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 40 then 400 - 6*x
  else if x > 40 then 7400/x - 40000/(x^2)
  else 0

/-- Annual profit function for iPhone production --/
noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 40 then -6*x^2 + 384*x - 40
  else if x > 40 then -40000/x - 16*x + 7360
  else 0

/-- Theorem stating the maximum annual profit --/
theorem max_annual_profit :
  ∃ (x : ℝ), x = 32 ∧ W x = 6104 ∧ ∀ (y : ℝ), W y ≤ W x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_profit_l60_6083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l60_6024

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (1 + (b/a)^2)

/-- A line with slope m and y-intercept c -/
structure Line (m c : ℝ) where

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (l₁ : Line m₁ c₁) (l₂ : Line m₂ c₂) : Prop :=
  m₁ * m₂ = -1

theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (l : Line 2 1) (asymptote : Line (b/a) 0)
  (perp : perpendicular asymptote l) :
  eccentricity h = Real.sqrt 5 / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l60_6024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l60_6006

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = -1 / Real.exp 1 :=
by
  -- We'll use x = 1/e as the minimizer
  let x := 1 / Real.exp 1
  
  -- Prove that x > 0
  have x_pos : x > 0 := by
    apply div_pos
    · exact zero_lt_one
    · exact Real.exp_pos 1
  
  -- Use x as our witness
  use x
  
  constructor
  · exact x_pos
  
  intro y y_pos
  
  sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l60_6006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_function_l60_6037

theorem exists_special_function : ∃ (f : ℚ → Fin 2), 
  (∀ x y : ℚ, x ≠ y → (x * y = 1 → f x ≠ f y)) ∧
  (∀ x y : ℚ, x ≠ y → (x + y = 0 ∨ x + y = 1 → f x ≠ f y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_function_l60_6037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_equals_four_l60_6062

noncomputable section

def total_amount : ℝ := 3.5

-- Define a function to represent rounding to the nearest integer
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  if x - ⌊x⌋ < 0.5 then ⌊x⌋ else ⌈x⌉

-- Define the probability space
def Ω : Set (ℝ × ℝ) := {p | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 = total_amount}

-- Define the event where the sum of rounded numbers equals 4
def E : Set (ℝ × ℝ) := {p ∈ Ω | round_to_nearest p.1 + round_to_nearest p.2 = 4}

-- State the theorem
theorem probability_sum_equals_four :
  (MeasureTheory.volume E) / (MeasureTheory.volume Ω) = 6/7 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_equals_four_l60_6062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_a_l60_6046

noncomputable def f (x : ℝ) : ℝ := Real.cos (x^2) - (Real.sqrt 3 / 2) * Real.sin (2 * x) - 1/2

theorem min_side_a (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi →
  f A + 1 = 0 →
  b + c = 2 →
  0 < b ∧ 0 < c →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  a ≥ 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_a_l60_6046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_afternoon_snowfall_rate_l60_6044

/-- Snowfall rates and durations for different time segments -/
structure SnowfallData where
  mon_morning_inches : ℚ
  mon_morning_hours : ℚ
  mon_afternoon_inches : ℚ
  mon_afternoon_hours : ℚ
  tue_morning_cm : ℚ
  tue_morning_hours : ℚ
  tue_afternoon_mm : ℚ
  tue_afternoon_hours : ℚ

/-- Conversion rates between different units -/
structure ConversionRates where
  inch_to_cm : ℚ
  cm_to_inch : ℚ
  mm_to_inch : ℚ

/-- Calculate the average snowfall rate for Tuesday afternoon -/
noncomputable def average_snowfall_rate_tue_afternoon (data : SnowfallData) (rates : ConversionRates) : ℚ :=
  (data.tue_afternoon_mm * rates.mm_to_inch) / data.tue_afternoon_hours

/-- Theorem stating that the average snowfall rate for Tuesday afternoon is 0.1968505 inches/hour -/
theorem tuesday_afternoon_snowfall_rate 
  (data : SnowfallData) 
  (rates : ConversionRates) 
  (h1 : data.mon_morning_inches = 125/1000) 
  (h2 : data.mon_morning_hours = 2)
  (h3 : data.mon_afternoon_inches = 1/2) 
  (h4 : data.mon_afternoon_hours = 3)
  (h5 : data.tue_morning_cm = 135/100) 
  (h6 : data.tue_morning_hours = 4)
  (h7 : data.tue_afternoon_mm = 25) 
  (h8 : data.tue_afternoon_hours = 5)
  (h9 : rates.inch_to_cm = 254/100)
  (h10 : rates.cm_to_inch = 393701/1000000)
  (h11 : rates.mm_to_inch = 393701/10000000) :
  average_snowfall_rate_tue_afternoon data rates = 1968505/10000000 := by
  sorry

#eval (25 : ℚ) * (393701 : ℚ) / (10000000 : ℚ) / (5 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_afternoon_snowfall_rate_l60_6044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_start_time_l60_6075

/-- The time B starts cycling after A starts walking -/
noncomputable def time_difference (a_speed b_speed : ℝ) (meeting_distance : ℝ) : ℝ :=
  meeting_distance / (a_speed + b_speed)

/-- Theorem stating that B starts cycling 2 hours after A, given the conditions -/
theorem cycling_start_time (a_speed b_speed meeting_distance : ℝ) 
  (ha : a_speed = 10)
  (hb : b_speed = 20)
  (hm : meeting_distance = 60) :
  time_difference a_speed b_speed meeting_distance = 2 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_start_time_l60_6075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_Q_equal_at_one_l60_6001

noncomputable def P (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 4

noncomputable def Q (x : ℝ) : ℝ :=
  let coeffs := [1, -2, 3, -4]
  let mean := (coeffs.sum) / (coeffs.length : ℝ)
  mean * x^3 + mean * x^2 + mean * x + mean

theorem P_Q_equal_at_one : P 1 = Q 1 ∧ P 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_Q_equal_at_one_l60_6001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_list_eventually_stable_l60_6017

/-- The operation that replaces two numbers with their GCD and LCM -/
def replace_operation (a b : ℕ) : ℕ × ℕ :=
  (Nat.gcd a b, Nat.lcm a b)

/-- A list is stable if no further operations can change it -/
def is_stable (list : List ℕ) : Prop :=
  ∀ a b, a ∈ list → b ∈ list → replace_operation a b = (a, b)

/-- Iterate the replace operation on a list given a sequence of pairs to replace -/
def iterate_replace (initial_list : List ℕ) (operation_sequence : Fin n → ℕ × ℕ) : List ℕ :=
  sorry

/-- The main theorem stating that any list will eventually become stable -/
theorem list_eventually_stable (initial_list : List ℕ) : 
  ∃ (n : ℕ) (final_list : List ℕ), 
    (∃ (operation_sequence : Fin n → ℕ × ℕ), 
      final_list = iterate_replace initial_list operation_sequence) ∧ 
    is_stable final_list :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_list_eventually_stable_l60_6017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_formula_l60_6034

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def a : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => a (n + 1) + a n

theorem fibonacci_formula (n : ℕ) :
  fibonacci n = a (n.pred) * fibonacci 0 + a n * fibonacci 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_formula_l60_6034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_polygon_area_ratio_l60_6020

/-- A parallelogram with side lengths 3 and 5 -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  is_side1 : side1 = 3
  is_side2 : side2 = 5

/-- The polygon formed by the bisectors of the parallelogram's internal angles -/
def BisectorPolygon (p : Parallelogram) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem stating the ratio of areas -/
theorem bisector_polygon_area_ratio (p : Parallelogram) :
  (area (BisectorPolygon p)) / (area (Set.range (λ x : ℝ × ℝ => x))) = 2 / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_polygon_area_ratio_l60_6020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_total_amount_l60_6008

/-- Represents the available paper money denominations in Ft -/
inductive Denomination : Type
  | twenty : Denomination
  | fifty : Denomination
  | hundred : Denomination
  | fiveHundred : Denomination

/-- The value of a denomination in Ft -/
def denominationValue : Denomination → Nat
  | Denomination.twenty => 20
  | Denomination.fifty => 50
  | Denomination.hundred => 100
  | Denomination.fiveHundred => 500

/-- A group of people with their money -/
structure MoneyGroup where
  size : Nat
  requiredContribution : Nat
  totalAmount : Nat

/-- The problem statement -/
theorem minimum_total_amount (g : MoneyGroup) (d : List Denomination) :
  g.size = 10 ∧
  g.requiredContribution = 30 ∧
  d = [Denomination.twenty, Denomination.fifty, Denomination.hundred, Denomination.fiveHundred] →
  g.totalAmount ≥ 560 ∧ 
  (∀ t : Nat, t ≥ 560 → 
    ∃ (distribution : List Nat), 
      distribution.length = g.size ∧
      distribution.sum = t ∧
      (∀ amount ∈ distribution, ∃ (bills : List Denomination), (bills.map denominationValue).sum = amount)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_total_amount_l60_6008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_two_l60_6049

/-- The length of the chord obtained when the line θ = π/6 intersects the circle ρ = 2cos(θ - π/6) -/
def chord_length : ℝ := 2

/-- The polar equation of the line -/
def line_equation (θ : ℝ) : Prop := θ = Real.pi / 6

/-- The polar equation of the circle -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos (θ - Real.pi / 6)

/-- The theorem stating that the chord length is 2 -/
theorem chord_length_is_two :
  ∀ ρ θ : ℝ, line_equation θ → circle_equation ρ θ → ρ = chord_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_two_l60_6049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_specific_l60_6038

/-- The area of an ellipse with given major axis endpoints and a point on the ellipse -/
noncomputable def ellipse_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  let center_x := (x1 + x2) / 2
  let center_y := (y1 + y2) / 2
  let a := ((x2 - x1)^2 + (y2 - y1)^2).sqrt / 2
  let b_squared := 36 * 100 / 19
  Real.pi * a * (b_squared.sqrt)

/-- Theorem: The area of the ellipse with major axis endpoints (-5, 0) and (15, 0),
    passing through the point (14, 6), is equal to (600 π) / √19 -/
theorem ellipse_area_specific : 
  ellipse_area (-5) 0 15 0 14 6 = 600 * Real.pi / Real.sqrt 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_specific_l60_6038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l60_6051

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: The sum of the first 5 terms of the geometric series with first term 1/4 and common ratio 1/4 is 1023/3072 -/
theorem geometric_series_sum :
  geometricSum (1/4 : ℝ) (1/4 : ℝ) 5 = 1023/3072 := by
  -- Expand the definition of geometricSum
  unfold geometricSum
  -- Simplify the expression
  simp [pow_succ]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l60_6051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_product_l60_6028

def regular_pentagon (Q : Fin 5 → ℂ) : Prop :=
  ∃ (center : ℂ) (r : ℝ), ∀ k : Fin 5, Complex.abs (Q k - center) = r ∧
    (Q k - center) = (Q 0 - center) * (Complex.exp (2 * Real.pi * Complex.I * (k : ℝ) / 5))

theorem pentagon_product (Q : Fin 5 → ℂ) :
  regular_pentagon Q → Q 0 = 6 → Q 2 = 8 →
  (Q 0) * (Q 1) * (Q 2) * (Q 3) * (Q 4) = 16806 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_product_l60_6028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_cat_walk_time_l60_6095

/-- The total time for Jenny's cat walking process -/
noncomputable def total_time (resistance_time : ℝ) (distance : ℝ) (rate : ℝ) : ℝ :=
  resistance_time + distance / rate

/-- Theorem: The total time for Jenny's cat walking process is 28 minutes -/
theorem jenny_cat_walk_time :
  total_time 20 64 8 = 28 := by
  -- Unfold the definition of total_time
  unfold total_time
  -- Simplify the arithmetic
  simp [add_comm, div_eq_mul_inv]
  -- Prove the equality
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_cat_walk_time_l60_6095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_two_three_l60_6087

noncomputable def f (x : ℝ) := (1/2) * Real.exp x + x - 6

theorem zero_of_f_in_interval_two_three :
  ∃ x, x ∈ Set.Ioo 2 3 ∧ f x = 0 := by
  sorry

#check zero_of_f_in_interval_two_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_two_three_l60_6087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coloring_l60_6013

/-- The number of ways to color n sectors of a circle using m colors -/
def colorings (n m : ℕ) : ℤ :=
  (m - 1)^n + (-1)^n * (m - 1)

/-- The actual number of valid colorings (this is a placeholder for the true definition) -/
def number_of_valid_colorings (n m : ℕ) : ℤ :=
  colorings n m  -- For now, we define it as equal to colorings

/-- Theorem stating the number of valid colorings for a circle with n sectors and m colors -/
theorem circle_coloring (n m : ℕ) (hn : n ≥ 2) (hm : m ≥ 2) :
  colorings n m = number_of_valid_colorings n m :=
by
  -- The proof would go here, but for now we use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coloring_l60_6013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_l60_6079

/-- Given two distinct real numbers a and b, sets M and N, and a mapping f,
    prove that a + b = 4 -/
theorem sum_of_roots (a b : ℝ) (ha_ne_b : a ≠ b) : 
  let M : Set ℝ := {a^2 - 4*a, -1}
  let N : Set ℝ := {b^2 - 4*b + 1, -2}
  ∃ (f : M → N), (∀ x : M, (f x : ℝ) = (x : ℝ)) → a + b = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_l60_6079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dans_car_efficiency_l60_6096

/-- Represents the fuel efficiency of a car in miles per gallon -/
def FuelEfficiency : Type := ℝ

/-- Calculates the fuel efficiency of a car given the distance traveled, 
    the amount spent on gas, and the cost per gallon of gas -/
noncomputable def calculate_fuel_efficiency (distance : ℝ) (amount_spent : ℝ) (cost_per_gallon : ℝ) : FuelEfficiency :=
  distance / (amount_spent / cost_per_gallon)

theorem dans_car_efficiency :
  let distance : ℝ := 304
  let amount_spent : ℝ := 38
  let cost_per_gallon : ℝ := 4
  calculate_fuel_efficiency distance amount_spent cost_per_gallon = (32 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dans_car_efficiency_l60_6096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cs_value_l60_6090

noncomputable def cs (x : Real) : Real := x.cos / x.sin

theorem cs_value (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) (h2 : Real.sin α - Real.cos α = Real.sqrt 2) :
  cs (α - Real.pi/4) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cs_value_l60_6090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_rectangle_with_100_similar_unique_parts_l60_6000

/-- A rectangle represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Similarity relation between two rectangles -/
def similar (r1 r2 : Rectangle) : Prop :=
  r1.width / r1.height = r2.width / r2.height

/-- A partition of a rectangle into smaller rectangles -/
structure Partition where
  original : Rectangle
  parts : Finset Rectangle

/-- Predicate to check if all rectangles in a partition are similar to the original -/
def all_similar (p : Partition) : Prop :=
  ∀ r ∈ p.parts, similar p.original r

/-- Predicate to check if all rectangles in a partition are unique -/
def all_unique (p : Partition) : Prop :=
  ∀ r1 r2, r1 ∈ p.parts → r2 ∈ p.parts → r1 ≠ r2 → r1.width ≠ r2.width ∨ r1.height ≠ r2.height

/-- The main theorem stating the existence of a rectangle with the desired properties -/
theorem exists_rectangle_with_100_similar_unique_parts :
  ∃ p : Partition, p.parts.card = 100 ∧ all_similar p ∧ all_unique p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_rectangle_with_100_similar_unique_parts_l60_6000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_consecutive_pairs_in_A_or_B_l60_6021

-- Define the relation between two non-negative integers
def related (a b : ℕ) : Prop := ∃ (n : ℕ), a + b = n ∧ (∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1)

-- Define the properties of sets A and B
structure RelatedSets :=
  (A B : Set ℕ)
  (A_infinite : Set.Infinite A)
  (B_infinite : Set.Infinite B)
  (related_AB : ∀ a b, a ∈ A → b ∈ B → related a b)
  (related_to_all_A_in_B : ∀ c, (∀ a, a ∈ A → related c a) → c ∈ B)
  (related_to_all_B_in_A : ∀ c, (∀ b, b ∈ B → related c b) → c ∈ A)

-- Define what it means for a set to have infinitely many consecutive pairs
def has_infinite_consecutive_pairs (S : Set ℕ) :=
  ∀ n : ℕ, ∃ m ≥ n, m ∈ S ∧ (m + 1) ∈ S

-- State the theorem
theorem infinite_consecutive_pairs_in_A_or_B (rs : RelatedSets) :
  has_infinite_consecutive_pairs rs.A ∨ has_infinite_consecutive_pairs rs.B :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_consecutive_pairs_in_A_or_B_l60_6021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_minus_pi_sixth_l60_6053

theorem cos_theta_minus_pi_sixth (θ : ℝ) (h1 : Real.cos θ = -5/13) (h2 : θ ∈ Set.Ioo π (3*π/2)) :
  Real.cos (θ - π/6) = -(5*Real.sqrt 3 + 12) / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_minus_pi_sixth_l60_6053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sum_given_prime_product_l60_6029

/-- A standard six-sided die -/
def Die : Type := Fin 6

/-- The result of rolling 5 dice -/
def Roll : Type := Fin 5 → Die

/-- The product of the values shown on a roll -/
def roll_product (r : Roll) : ℕ := (Finset.univ.prod (λ i => (r i).val.succ))

/-- The sum of the values shown on a roll -/
def roll_sum (r : Roll) : ℕ := (Finset.univ.sum (λ i => (r i).val.succ))

/-- A roll results in a prime product -/
def is_prime_product (r : Roll) : Prop := Nat.Prime (roll_product r)

theorem dice_sum_given_prime_product (r : Roll) :
  is_prime_product r → (roll_sum r % 2 = 1) = false :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sum_given_prime_product_l60_6029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_voting_l60_6027

noncomputable def vote : Fin b → Fin a → Bool := sorry

theorem competition_voting (a b k : ℕ) (h_b_odd : Odd b) (h_b_ge_3 : b ≥ 3) : 
  (∀ i j : Fin b, i ≠ j → ∃ S : Finset (Fin a), S.card ≤ k ∧ 
    (∀ x : Fin a, x ∉ S → (vote i x ≠ vote j x))) →
  (k : ℚ) / a ≥ (b - 1 : ℚ) / (2 * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_voting_l60_6027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l60_6033

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- State the theorem
theorem odd_function_properties (a b : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f a b x = -f a b (-x)) →  -- f is odd on (-1,1)
  f a b (1/2) = 2/5 →                                  -- f(1/2) = 2/5
  (∃ g : ℝ → ℝ, 
    (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f a b x = g x) ∧        -- f equals g on (-1,1)
    (∀ x ∈ Set.Ioo (-1 : ℝ) 1, g x = x / (1 + x^2)) ∧  -- g(x) = x/(1+x^2)
    StrictMono g ∧                                     -- g is strictly increasing
    {t : ℝ | f a b (t-1) + f a b t < 0} = Set.Ioo 0 (1/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l60_6033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_ratio_1_simplify_ratio_2_simplify_ratio_3_l60_6043

-- Define a function to represent a ratio
def Ratio (a b : ℤ) : ℚ := a / b

-- Define equality for ratios
def ratio_eq (r₁ r₂ : ℚ) : Prop := r₁ = r₂

-- Theorem for the first ratio
theorem simplify_ratio_1 : ratio_eq (Ratio 32 48) (Ratio 2 3) := by
  sorry

-- Theorem for the second ratio
theorem simplify_ratio_2 : ratio_eq (Ratio 15 300) (Ratio 1 20) := by
  sorry

-- Theorem for the third ratio
theorem simplify_ratio_3 : ratio_eq (Ratio 11 12) (Ratio 22 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_ratio_1_simplify_ratio_2_simplify_ratio_3_l60_6043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_slant_height_l60_6014

/-- Represents a frustum of a cone -/
structure Frustum where
  upper_radius : ℝ
  lower_radius : ℝ
  slant_height : ℝ

/-- Calculates the area of a circular base -/
noncomputable def base_area (radius : ℝ) : ℝ := Real.pi * radius^2

/-- Calculates the lateral surface area of a frustum -/
noncomputable def lateral_area (f : Frustum) : ℝ :=
  Real.pi * (f.upper_radius + f.lower_radius) * f.slant_height

/-- Calculates the total base area of a frustum -/
noncomputable def total_base_area (f : Frustum) : ℝ :=
  base_area f.upper_radius + base_area f.lower_radius

theorem frustum_slant_height :
  ∀ f : Frustum,
    f.upper_radius = 2 →
    f.lower_radius = 6 →
    lateral_area f = total_base_area f →
    f.slant_height = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_slant_height_l60_6014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_difference_l60_6070

theorem log_power_difference (c d : ℝ) (hc : c = Real.log 25) (hd : d = Real.log 49) :
  (5 : ℝ)^(d/c) - (7 : ℝ)^(c/d) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_difference_l60_6070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l60_6003

/-- A line passing through a point (x₀, y₀) can be represented by the equation:
    A(x - x₀) + B(y - y₀) = 0, where (A, B) is a normal vector to the line. -/
def line_through_point (A B x₀ y₀ : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ A * (x - x₀) + B * (y - y₀) = 0

/-- Two lines Ax + By + C = 0 and Dx + Ey + F = 0 are perpendicular if and only if AD + BE = 0 -/
def perpendicular (A B D E : ℝ) : Prop :=
  A * D + B * E = 0

theorem perpendicular_line_equation :
  let given_line := λ x y ↦ 3 * x - y + 8 = 0
  let perpendicular_line := line_through_point 1 3 2 1
  perpendicular 3 (-1) 1 3 ∧
  perpendicular_line 2 1 ∧
  ∀ x y, perpendicular_line x y ↔ x + 3 * y - 5 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l60_6003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_percentage_is_88_l60_6005

/-- A triangle with sides of length 5 units -/
structure LargeTriangle where
  side_length : ℝ
  side_length_eq : side_length = 5

/-- The division of the large triangle into smaller triangles -/
structure TriangleDivision (T : LargeTriangle) where
  total_small_triangles : ℕ
  shaded_small_triangles : ℕ
  total_eq : total_small_triangles = 25
  shaded_eq : shaded_small_triangles = 22

/-- The percentage of shaded area in the triangle -/
noncomputable def shaded_percentage (T : LargeTriangle) (D : TriangleDivision T) : ℝ :=
  (D.shaded_small_triangles : ℝ) / (D.total_small_triangles : ℝ) * 100

/-- Theorem stating that the shaded percentage is 88% -/
theorem shaded_percentage_is_88 (T : LargeTriangle) (D : TriangleDivision T) :
  shaded_percentage T D = 88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_percentage_is_88_l60_6005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base12_addition_proof_l60_6069

/-- Represents a single digit in base 12 --/
def Base12Digit := Fin 12

/-- Converts a natural number to its base 12 representation --/
def toBase12 (n : ℕ) : List Base12Digit := sorry

/-- Converts a base 12 representation to a natural number --/
def fromBase12 (digits : List Base12Digit) : ℕ := sorry

/-- Adds two numbers represented in base 12 --/
def addBase12 (a b : List Base12Digit) : List Base12Digit := sorry

/-- Helper function to create a Base12Digit from a natural number --/
def mkBase12Digit (n : ℕ) : Base12Digit :=
  ⟨n % 12, by
    apply Nat.mod_lt
    exact Nat.zero_lt_succ 11⟩

theorem base12_addition_proof :
  let a := [mkBase12Digit 7, mkBase12Digit 10, mkBase12Digit 3]  -- 7A3₁₂
  let b := [mkBase12Digit 2, mkBase12Digit 11, mkBase12Digit 8]  -- 2B8₁₂
  addBase12 a b = [mkBase12Digit 10, mkBase12Digit 1, mkBase12Digit 11]  -- A1B₁₂
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base12_addition_proof_l60_6069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_pencil_pen_difference_l60_6094

structure Student : Type

def brought_pencil (s : Student) : Prop := sorry
def brought_pen (s : Student) : Prop := sorry

theorem math_competition_pencil_pen_difference
  (total_students : ℕ)
  (forgot_pencil : ℕ)
  (forgot_pen : ℕ)
  (h_total : total_students = 20)
  (h_forgot_pencil : forgot_pencil = 12)
  (h_forgot_pen : forgot_pen = 2)
  (h_pencil_implies_pen : ∀ s : Student, brought_pencil s → brought_pen s) :
  (total_students - forgot_pen) - (total_students - forgot_pencil) =
  (total_students - forgot_pencil) - (total_students - forgot_pen) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_pencil_pen_difference_l60_6094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y2z_in_expansion_l60_6058

theorem coefficient_x3y2z_in_expansion : ℕ := by
  -- Define the multinomial coefficient
  let multinomial_coeff (n : ℕ) (k₁ k₂ k₃ : ℕ) : ℕ := 
    Nat.factorial n / (Nat.factorial k₁ * Nat.factorial k₂ * Nat.factorial k₃)

  -- Define the specific case we're interested in
  let n : ℕ := 6
  let k₁ : ℕ := 3
  let k₂ : ℕ := 2
  let k₃ : ℕ := 1

  -- State that the sum of k₁, k₂, and k₃ equals n
  have sum_equals_n : k₁ + k₂ + k₃ = n := by sorry

  -- The coefficient is equal to the multinomial coefficient
  have coefficient_equals_multinomial : 
    multinomial_coeff n k₁ k₂ k₃ = 60 := by sorry

  -- Return the final result
  exact 60

/- The coefficient of x^3y^2z in the expansion of (x+y+z)^6 is 60 -/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y2z_in_expansion_l60_6058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_l60_6052

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then
    Real.exp (Real.sin (x^(3/2) * Real.sin (2/x))) - 1 + x^2
  else
    0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_l60_6052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_m_greater_than_three_l60_6061

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem inequality_holds_iff_m_greater_than_three (m : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 4, f x ^ 2 - f (x ^ 2) - m < 0) ↔ m > 3 :=
by
  sorry

#check inequality_holds_iff_m_greater_than_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_m_greater_than_three_l60_6061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_student_count_l60_6039

/-- The number of students per row on the first day -/
def initial_row_count : ℕ := 18

/-- The number of days they can arrange differently -/
def arrangement_days : ℕ := 12

/-- The function to count the number of divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The theorem stating that 72 is the smallest number satisfying the conditions -/
theorem smallest_student_count : 
  72 = (Finset.filter (λ x => x % initial_row_count = 0 ∧ count_divisors x = arrangement_days + 1) 
    (Finset.range 1000)).min' (by sorry) := by
  sorry

#check smallest_student_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_student_count_l60_6039
