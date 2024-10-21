import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_3sin_max_l128_12834

theorem cos_plus_3sin_max (x : ℝ) : Real.cos x + 3 * Real.sin x ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_3sin_max_l128_12834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_calculation_l128_12821

/-- Given a discount percentage and a desired profit percentage, 
    calculate the markup percentage for the marked price. -/
noncomputable def calculate_markup (discount_percent : ℝ) (profit_percent : ℝ) : ℝ :=
  let selling_price_ratio := 1 + profit_percent / 100
  let discount_ratio := 1 - discount_percent / 100
  (selling_price_ratio / discount_ratio - 1) * 100

/-- The theorem states that given the specific discount and profit percentages,
    the calculated markup is approximately 30%. -/
theorem markup_calculation :
  let discount : ℝ := 16.92307692307692
  let profit : ℝ := 8
  let calculated_markup := calculate_markup discount profit
  ∀ ε > 0, |calculated_markup - 30| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_calculation_l128_12821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_inequality_fib_between_powers_l128_12895

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_inequality (n : ℕ) (h : n ≥ 3) : fib (n + 3) < 5 * fib n := by
  sorry

theorem fib_between_powers (n : ℕ) (h : n > 0) (k : ℕ) :
  ∀ s : Finset ℕ, (∀ m ∈ s, n^k < fib m ∧ fib m < n^(k+1)) → s.card ≤ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_inequality_fib_between_powers_l128_12895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l128_12833

open Complex Real

noncomputable def initial_number : ℂ := -4 + 6*I
noncomputable def rotation_angle : ℝ := π/3  -- 60° in radians
def dilation_factor : ℝ := 2

theorem complex_transformation :
  (2 * (initial_number * (Complex.exp (rotation_angle * I)))) = -22 + (6 - 4 * sqrt 3) * I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l128_12833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_460_l128_12891

-- Define the set of angles with the same terminal side as a given angle
def same_terminal_side (θ : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = θ + k * 360}

-- Theorem statement
theorem same_terminal_side_460 :
  same_terminal_side (-460) = {α : ℝ | ∃ k : ℤ, α = k * 360 + 260} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_460_l128_12891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recliner_sales_increase_l128_12865

def price_drop_percentage : ℝ := 20
def gross_increase_percentage : ℝ := 36

theorem recliner_sales_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (new_quantity : ℝ) 
  (h1 : original_price > 0) 
  (h2 : original_quantity > 0) :
  let new_price := original_price * (1 - price_drop_percentage / 100)
  let gross_increase_factor := 1 + gross_increase_percentage / 100
  new_price * new_quantity = gross_increase_factor * (original_price * original_quantity) →
  (new_quantity / original_quantity - 1) * 100 = 70 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_recliner_sales_increase_l128_12865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firecracker_thrower_in_302_l128_12816

-- Define the set of boys
inductive Boy : Type
| buster : Boy
| oak : Boy
| marco : Boy
| knobby : Boy
| malf : Boy

-- Define the set of rooms
inductive Room : Type
| r302 : Room
| r401 : Room
| r502 : Room

-- Define a function to assign rooms to boys
def room_of : Boy → Room
| Boy.buster => Room.r502
| Boy.oak => Room.r401
| Boy.marco => Room.r401
| Boy.knobby => Room.r302
| Boy.malf => Room.r302

-- Define a predicate for telling the truth
def tells_truth : Boy → Prop := sorry

-- Define a predicate for throwing the firecracker
def threw_firecracker : Boy → Prop := sorry

-- Theorem statement
theorem firecracker_thrower_in_302 :
  (∃! b : Boy, ¬(tells_truth b)) →
  (∀ b : Boy, ¬(tells_truth b) ↔ threw_firecracker b) →
  (tells_truth Boy.buster → ¬(threw_firecracker Boy.buster)) →
  (tells_truth Boy.oak → ¬(threw_firecracker Boy.oak) ∧ tells_truth Boy.marco) →
  (tells_truth Boy.marco → ¬(threw_firecracker Boy.oak) ∧ ¬(∃ b : Boy, room_of b = Room.r502 ∧ threw_firecracker b) ∧ ¬(threw_firecracker Boy.marco)) →
  (tells_truth Boy.knobby → ∃ b : Boy, room_of b = Room.r502 ∧ threw_firecracker b) →
  (tells_truth Boy.malf → ∃ b : Boy, room_of b = Room.r502 ∧ threw_firecracker b) →
  ∃ b : Boy, threw_firecracker b ∧ room_of b = Room.r302 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_firecracker_thrower_in_302_l128_12816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_lower_bound_l128_12850

/-- M_(n,k,h) represents the number of k-element subsets of an n-element set 
    that contain at least one h-element subset -/
def M (n k h : ℕ) : ℕ := sorry

/-- The ceiling function -/
noncomputable def ceiling (x : ℚ) : ℕ := Int.toNat (Int.ceil x)

/-- Theorem stating the lower bound for M_(n,k,h) -/
theorem M_lower_bound {n k h : ℕ} (h_le_k : h ≤ k) (k_le_n : k ≤ n) :
  M n k h ≥ ceiling (n / (n - h : ℚ) * 
    ceiling ((n - 1) / ((n - h - 1) : ℚ) * 
      ceiling ((k + 1) / ((k - h + 1) : ℚ)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_lower_bound_l128_12850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l128_12871

/-- A right triangle with specific median lengths and area has a hypotenuse of √550.4 -/
theorem right_triangle_hypotenuse (a b : ℝ) : 
  (a > 0) → 
  (b > 0) → 
  (a^2 + b^2/4 = 36) → 
  (b^2 + a^2/4 = 136) → 
  (2 * a * b = 48) → 
  (Real.sqrt ((2*a)^2 + (2*b)^2) = Real.sqrt 550.4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l128_12871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_five_l128_12814

/-- The cost function for building a dormitory and road -/
noncomputable def f (x : ℝ) : ℝ := 600 / (x + 5) + 5 + 6 * x

/-- The theorem stating the minimum value of the cost function -/
theorem min_cost_at_five :
  ∀ x ∈ Set.Icc (0 : ℝ) 8, f 5 ≤ f x ∧ f 5 = 95 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_five_l128_12814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l128_12897

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point of tangency
def point : ℝ × ℝ := (2, 8)

-- Theorem statement
theorem tangent_line_at_point :
  ∃ (m b : ℝ), 
    (∀ x, (m * x + b) = f point.fst + (deriv f point.fst) * (x - point.fst)) ∧
    m = 12 ∧ 
    b = -16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l128_12897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrow_construction_l128_12859

-- Define a triangle
structure Triangle where
  side_length : ℝ

-- Define the shapes that can be formed
inductive Shape
  | LargeTriangle
  | Parallelogram
  | Trapezoid
  | HalfHexagon
deriving DecidableEq

-- Define the arrow shape
structure Arrow where
  shapes : List Shape

-- Theorem statement
theorem arrow_construction (t1 t2 t3 : Triangle) 
  (h : t1.side_length = t2.side_length ∧ t2.side_length = t3.side_length) :
  ∃ (arrow : Arrow), 
    arrow.shapes.length = 4 ∧ 
    arrow.shapes.toFinset.card = 4 ∧
    Shape.LargeTriangle ∈ arrow.shapes ∧
    Shape.Parallelogram ∈ arrow.shapes ∧
    Shape.Trapezoid ∈ arrow.shapes ∧
    Shape.HalfHexagon ∈ arrow.shapes := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrow_construction_l128_12859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_vertex_distance_l128_12875

/-- A parabola with vertex V and focus F -/
structure Parabola where
  V : ℝ × ℝ
  F : ℝ × ℝ

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_focus_vertex_distance 
  (p : Parabola) 
  (A : ℝ × ℝ) 
  (h1 : distance A p.F = 24) 
  (h2 : distance A p.V = 25) : 
  distance p.F p.V = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_vertex_distance_l128_12875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l128_12847

theorem triangle_angle_measure (a b c : ℝ) (h : a^2 - b^2 - c^2 + Real.sqrt 3 * b * c = 0) :
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l128_12847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_calculation_l128_12863

theorem complex_fraction_calculation : ((((4:ℚ)-1)⁻¹ - 1)⁻¹ - 1)⁻¹ - 1 = -7/5 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_calculation_l128_12863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_8_equals_13_over_3_l128_12829

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := (3 * x + 2) / (x - 2)

-- Theorem statement
theorem g_of_8_equals_13_over_3 : g 8 = 13 / 3 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expression
  simp [mul_add, add_div]
  -- Perform arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_8_equals_13_over_3_l128_12829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_f_nonneg_def_solution_set_eq_open_interval_l128_12889

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 - 4*x else (λ y => y^2 - 4*y) (-x)

theorem f_even : ∀ x, f x = f (-x) := by sorry

theorem f_nonneg_def : ∀ x ≥ 0, f x = x^2 - 4*x := by sorry

theorem solution_set_eq_open_interval :
  {x : ℝ | f (x + 2) < 5} = Set.Ioo (-3) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_f_nonneg_def_solution_set_eq_open_interval_l128_12889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l128_12823

/-- The function f(x) = sin(ωx + φ) -/
noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

/-- The theorem stating the minimum value of ω -/
theorem min_omega :
  ∀ ω φ : ℝ,
  ω > 0 →
  (∀ x : ℝ, f ω φ (x - π/32) = f ω φ (π/32 - x)) →
  f ω φ (-π/32) = 0 →
  (∃ x₀ : ℝ, ∀ x : ℝ, f ω φ x₀ ≤ f ω φ x ∧ f ω φ x ≤ f ω φ (x₀ + π/8)) →
  ω ≥ 8 :=
by
  sorry

#check min_omega

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l128_12823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_max_value_achievable_l128_12818

theorem max_value_theorem (y : ℝ) (h : y > 0) :
  (y^2 + 4 - Real.sqrt (y^4 + 16)) / y ≤ 2 * Real.sqrt 2 - 2 :=
by sorry

theorem max_value_achievable :
  ∃ y : ℝ, y > 0 ∧ (y^2 + 4 - Real.sqrt (y^4 + 16)) / y = 2 * Real.sqrt 2 - 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_max_value_achievable_l128_12818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_area_strip_circle_intersection_max_area_l128_12884

/-- A convex curve in 2D space -/
structure ConvexCurve where
  -- Add necessary fields and conditions
  -- (left as a placeholder for now)

/-- A convex figure in 2D space -/
structure ConvexFigure where
  -- Add necessary fields and conditions
  -- (left as a placeholder for now)

/-- The diameter of a convex curve or figure -/
noncomputable def diameter {T : Type*} (C : T) : ℝ := sorry

/-- The width of a convex figure -/
noncomputable def width (F : ConvexFigure) : ℝ := sorry

/-- The area enclosed by a convex curve or figure -/
noncomputable def area {T : Type*} (C : T) : ℝ := sorry

/-- A circle with given diameter -/
noncomputable def circleShape (d : ℝ) : ConvexCurve := sorry

/-- The intersection of a strip and a circle -/
noncomputable def stripCircleIntersection (w d : ℝ) : ConvexFigure := sorry

/-- Theorem: Among all convex curves with a diameter of 1, 
    the area enclosed by a circle is maximal -/
theorem circle_max_area (C : ConvexCurve) (h : diameter C = 1) : 
  area C ≤ area (circleShape 1) := by sorry

/-- Theorem: Among all convex figures with diameter D and width Δ, 
    the area enclosed by the intersection of a strip of width Δ 
    and a circle of diameter D is maximal -/
theorem strip_circle_intersection_max_area (F : ConvexFigure) (D Δ : ℝ) 
  (h1 : diameter F = D) (h2 : width F = Δ) : 
  area F ≤ area (stripCircleIntersection Δ D) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_area_strip_circle_intersection_max_area_l128_12884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_theorem_l128_12862

-- Define the circles and points
variable (K₁ K₂ K₃ : Set (ℝ × ℝ))
variable (O₁ O₂ O₃ P A B C X Y Z : ℝ × ℝ)

-- State the conditions
axiom centers : (O₁ ∈ K₁) ∧ (O₂ ∈ K₂) ∧ (O₃ ∈ K₃)
axiom common_point : P ∈ K₁ ∧ P ∈ K₂ ∧ P ∈ K₃
axiom intersections : 
  (K₁ ∩ K₂ = {P, A}) ∧ (K₂ ∩ K₃ = {P, B}) ∧ (K₃ ∩ K₁ = {P, C})
axiom X_on_K₁ : X ∈ K₁
axiom Y_def : Y ∈ K₂ ∧ Y ≠ A ∧ ∃ t : ℝ, Y = t • X + (1 - t) • A
axiom Z_def : Z ∈ K₃ ∧ Z ≠ C ∧ ∃ s : ℝ, Z = s • X + (1 - s) • C

-- Define area function (placeholder)
noncomputable def area_triangle (a b c : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem circles_theorem :
  (∃ t : ℝ, Y = t • B + (1 - t) • Z) ∧ 
  (area_triangle X Y Z ≤ 4 * area_triangle O₁ O₂ O₃) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_theorem_l128_12862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_and_q_l128_12803

theorem proposition_p_and_q : 
  (∃ x : ℝ, Real.sin x < 1) ∧ (∀ x : ℝ, Real.exp (abs x) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_and_q_l128_12803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l128_12852

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

theorem geometric_sequence_properties (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : |a 2 - a 1| = 2) 
  (h3 : a 1 * a 2 * a 3 = 8) :
  ∃ q S_5 : ℝ, q = a 2 / a 1 ∧ 
              S_5 = a 1 * (1 - q^5) / (1 - q) ∧
              q = 1/2 ∧ S_5 = 31/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l128_12852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_writable_composite_l128_12827

/-- A function that represents the set of numbers that can be written on the board -/
def Writable : Set ℕ → Prop := sorry

/-- The initial number on the board -/
def initial_number : ℕ := 2022

/-- The rule for writing numbers on the board -/
axiom write_rule (n d : ℕ) :
  d > 1 → d ∣ n → Writable {n} → Writable {n + d}

/-- The initial number is writable -/
axiom initial_writable : Writable {initial_number}

/-- Definition of a composite number -/
def is_composite (n : ℕ) : Prop := ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b

/-- The main theorem -/
theorem largest_non_writable_composite :
  (∀ n, is_composite n ∧ n > 2033 → Writable {n}) ∧
  is_composite 2033 ∧ ¬Writable {2033} := by
  sorry

#check largest_non_writable_composite

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_writable_composite_l128_12827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l128_12885

def number_list : List ℕ := [37, 39, 41, 43, 45]

def is_prime (n : ℕ) : Bool :=
  n > 1 && (Nat.factors n).length == 1

def prime_numbers : List ℕ := number_list.filter is_prime

theorem arithmetic_mean_of_primes :
  (prime_numbers.sum : ℚ) / prime_numbers.length = 121 / 3 := by
  -- Proof goes here
  sorry

#eval prime_numbers
#eval prime_numbers.sum
#eval prime_numbers.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l128_12885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l128_12808

/-- The curve C₁ in the Cartesian coordinate system -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The line l in the polar coordinate system -/
noncomputable def l (ρ θ : ℝ) : Prop := ρ * (Real.cos θ - Real.sin θ) = 4

/-- The curve C₂ obtained by stretching C₁ -/
def C₂ (x' y' : ℝ) : Prop := x'^2 / 4 + y'^2 / 3 = 1

/-- The point P -/
def P : ℝ × ℝ := (1, 2)

/-- The line l₁ passing through P and parallel to l -/
noncomputable def l₁ (t : ℝ) : ℝ × ℝ := (1 + Real.sqrt 2 / 2 * t, 2 + Real.sqrt 2 / 2 * t)

/-- The theorem stating that the product of distances from P to intersection points is 2 -/
theorem intersection_distance_product :
  ∃ (t₁ t₂ : ℝ), C₂ (l₁ t₁).1 (l₁ t₁).2 ∧ C₂ (l₁ t₂).1 (l₁ t₂).2 ∧
  ((l₁ t₁).1 - P.1)^2 + ((l₁ t₁).2 - P.2)^2 *
  ((l₁ t₂).1 - P.1)^2 + ((l₁ t₂).2 - P.2)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l128_12808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_eq_third_height_l128_12843

/-- A triangle with sides in arithmetic progression -/
structure ArithmeticTriangle where
  a : ℝ
  d : ℝ
  side1 : ℝ := a
  side2 : ℝ := a + d
  side3 : ℝ := a + 2*d
  a_pos : 0 < a
  d_pos : 0 < d

/-- The height of the triangle corresponding to the middle side -/
noncomputable def ArithmeticTriangle.height (t : ArithmeticTriangle) : ℝ := sorry

/-- The radius of the inscribed circle of the triangle -/
noncomputable def ArithmeticTriangle.inradius (t : ArithmeticTriangle) : ℝ := sorry

/-- Theorem: In a triangle with sides in arithmetic progression, 
    the radius of the inscribed circle is 1/3 of the height 
    corresponding to the middle side of the progression -/
theorem inradius_eq_third_height (t : ArithmeticTriangle) : 
  t.inradius = (1/3) * t.height := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_eq_third_height_l128_12843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l128_12866

/-- Regular triangular prism with base side length a -/
structure RegularTriangularPrism (a : ℝ) where
  base_side_length : a > 0

/-- Plane passing through the center of the base and symmetry centers of two lateral faces -/
structure CrossSectionPlane (α : ℝ) where
  angle_with_base : 0 < α ∧ α < Real.pi / 2

/-- Area of the cross-section in a regular triangular prism -/
noncomputable def cross_section_area (a : ℝ) (α : ℝ) (prism : RegularTriangularPrism a) (plane : CrossSectionPlane α) : ℝ :=
  (a^2 * Real.sqrt 3) / (12 * Real.cos α)

/-- Theorem: The area of the cross-section in a regular triangular prism -/
theorem cross_section_area_theorem (a : ℝ) (α : ℝ) (prism : RegularTriangularPrism a) (plane : CrossSectionPlane α) :
  cross_section_area a α prism plane = (a^2 * Real.sqrt 3) / (12 * Real.cos α) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l128_12866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l128_12844

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (2 * x) - 1, Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem triangle_side_length (A B C : ℝ) (hf : f A = 1) (hb : B = 1) 
  (harea : (1/2) * B * C * Real.sin A = Real.sqrt 3 / 2) :
  A^2 + B^2 + C^2 - 2*B*C*Real.cos A = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l128_12844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_properties_l128_12835

/-- Given a triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The dot product of two vectors -/
def dotProduct (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

/-- The magnitude of a vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

/-- The projection of one vector onto another -/
noncomputable def projection (u v : ℝ × ℝ) : ℝ :=
  (dotProduct u v) / (magnitude v)

theorem triangle_vector_properties (t : Triangle) 
    (h1 : t.a = 4) 
    (h2 : t.b = 6) 
    (h3 : t.C = 60 * π / 180) : 
    ∃ (BC CA : ℝ × ℝ), 
      dotProduct BC CA = -12 ∧ 
      projection CA BC = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_properties_l128_12835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l128_12879

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

noncomputable def y (b x : ℝ) : ℝ := log b (x^2 + 1)

theorem log_properties (b : ℝ) (h : b > 1) :
  (y b 0 = 0) ∧
  (y b 1 = log b 2) ∧
  (∃ (r : ℝ), y b (-1) = r) ∧
  (∀ x, 0 ≤ x → x < 1 → y b x > 0 ∧ Monotone (fun x => y b x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l128_12879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tree_types_l128_12839

/-- Represents a tree type -/
inductive TreeType
| Apple
| Pear
| Plum
| Apricot
| Cherry
| Almond
deriving DecidableEq

/-- Represents a position in the orchard -/
structure Position where
  x : ℕ
  y : ℕ
deriving DecidableEq

/-- Represents a tree placement in the orchard -/
structure TreePlacement where
  treeType : TreeType
  position : Position
deriving DecidableEq

/-- Checks if three positions form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : Position) : Prop := sorry

/-- Checks if a list of tree placements satisfies the orchard conditions -/
def isValidOrchard (placements : List TreePlacement) : Prop :=
  (placements.length = 18) ∧
  (∀ t : TreeType, (placements.filter (λ p => p.treeType = t)).length = 3) ∧
  (∃ triangles : List (Position × Position × Position),
    triangles.length = 6 ∧
    (∀ triangle ∈ triangles, let (p1, p2, p3) := triangle; isEquilateralTriangle p1 p2 p3) ∧
    (∀ triangle ∈ triangles,
      let (p1, p2, p3) := triangle
      ∃ t1 t2 t3 : TreeType,
        t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧
        (TreePlacement.mk t1 p1 ∈ placements) ∧
        (TreePlacement.mk t2 p2 ∈ placements) ∧
        (TreePlacement.mk t3 p3 ∈ placements)))

theorem max_tree_types :
  ∃ (placements : List TreePlacement),
    isValidOrchard placements ∧
    (∀ (placements' : List TreePlacement),
      isValidOrchard placements' →
      (placements'.map (λ p => p.treeType)).toFinset.card ≤ 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tree_types_l128_12839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l128_12826

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi :=
by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l128_12826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_five_l128_12836

/-- Represents the outcome of rolling a fair six-sided die twice -/
structure DiceRoll where
  first : Fin 6
  second : Fin 6
deriving Fintype

/-- The sum of the numbers facing up on two dice rolls -/
def sum_of_rolls (roll : DiceRoll) : Nat :=
  roll.first.val + 1 + roll.second.val + 1

/-- The set of all possible outcomes when rolling a die twice -/
def all_outcomes : Finset DiceRoll :=
  Finset.univ

/-- The set of outcomes where the sum of the rolls is 5 -/
def sum_five_outcomes : Finset DiceRoll :=
  Finset.filter (fun roll => sum_of_rolls roll = 5) all_outcomes

/-- The probability of an event occurring when rolling a die twice -/
def probability (event : Finset DiceRoll) : Rat :=
  event.card / all_outcomes.card

theorem probability_sum_five :
  probability sum_five_outcomes = 1 / 9 := by
  sorry

#eval probability sum_five_outcomes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_five_l128_12836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l128_12882

/-- Given a quadratic function z = -x^2 + cx + d with roots -6 and 2, its vertex is (-2, 16) -/
theorem quadratic_vertex (c d : ℝ) : 
  (∀ x, -x^2 + c*x + d = -(x + 6)*(x - 2)) →
  (let f := fun x : ℝ => -x^2 + c*x + d
   ∃ vertex : ℝ, IsLocalMax f vertex ∧ vertex = -2 ∧ f vertex = 16) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l128_12882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_holds_iff_m_greater_than_neg_quarter_l128_12815

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

-- State the theorem
theorem f_inequality_holds_iff_m_greater_than_neg_quarter (m : ℝ) :
  (∀ x ≥ 1, f (x + 2*m) + m * f x > 0) ↔ m > -1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_holds_iff_m_greater_than_neg_quarter_l128_12815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_secant_l128_12858

noncomputable section

open Real Set

theorem circle_radius_from_secant (O P Q R : EuclideanSpace ℝ (Fin 2)) (r : ℝ) : 
  (dist O P = 15) →
  (dist P Q = 11) →
  (dist Q R = 8) →
  (∃ (C : Set (EuclideanSpace ℝ (Fin 2))), Metric.sphere O r = C ∧ Q ∈ C ∧ R ∈ C ∧ P ∉ C) →
  r = 4 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_secant_l128_12858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_range_of_m_l128_12802

open Real

variable (a b c x : ℝ)

-- Define the conditions
def positive_abc (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_abc (a b c : ℝ) : Prop := a + b + c = 3

-- Part 1
theorem inequality_proof {a b c : ℝ} (h1 : positive_abc a b c) (h2 : sum_abc a b c) :
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2 :=
sorry

-- Part 2
theorem range_of_m {a b c : ℝ} (h1 : positive_abc a b c) (h2 : sum_abc a b c) :
  ∃ m : ℝ, (∀ x : ℝ, -x^2 + m*x + 2 ≤ a^2 + b^2 + c^2) ∧ m ∈ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_range_of_m_l128_12802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_perimeter_specific_l128_12878

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  base_side_length : ℝ
  height : ℝ

/-- Midpoint triangle in a right prism -/
noncomputable def midpoint_triangle_perimeter (prism : RightPrism) : ℝ :=
  let half_base := prism.base_side_length / 2
  let half_height := prism.height / 2
  5 + 2 * Real.sqrt (half_base^2 + half_height^2)

/-- Theorem: The perimeter of the midpoint triangle in a right prism 
    with base side length 10 and height 18 is 5 + 2√106 -/
theorem midpoint_triangle_perimeter_specific :
  midpoint_triangle_perimeter ⟨10, 18⟩ = 5 + 2 * Real.sqrt 106 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_perimeter_specific_l128_12878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_to_second_hour_increase_l128_12896

/-- Represents a 3-hour bike ride with given conditions -/
structure BikeRide where
  second_hour_distance : ℚ
  third_hour_percentage_increase : ℚ
  total_distance : ℚ

/-- Calculates the percentage increase between two distances -/
def percentage_increase (d1 d2 : ℚ) : ℚ :=
  (d2 - d1) / d1 * 100

/-- Theorem stating the percentage increase from first to second hour -/
theorem first_to_second_hour_increase (ride : BikeRide)
  (h1 : ride.second_hour_distance = 12)
  (h2 : ride.third_hour_percentage_increase = 25)
  (h3 : ride.total_distance = 37) :
  percentage_increase 
    (ride.total_distance - ride.second_hour_distance - (ride.second_hour_distance * (1 + ride.third_hour_percentage_increase / 100)))
    ride.second_hour_distance = 20 := by
  sorry

#check first_to_second_hour_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_to_second_hour_increase_l128_12896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_for_sixty_degree_angle_l128_12869

/-- Definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of an ellipse -/
structure Ellipse where
  foci : Point × Point
  eccentricity : ℝ

/-- Definition of a hyperbola -/
structure Hyperbola where
  foci : Point × Point
  eccentricity : ℝ

/-- Definition of related curves -/
structure RelatedCurves where
  foci : Point × Point
  ellipse : Ellipse
  hyperbola : Hyperbola
  foci_same : ellipse.foci = foci ∧ hyperbola.foci = foci
  reciprocal_eccentricities : ellipse.eccentricity * hyperbola.eccentricity = 1

/-- Angle between three points -/
def angle (A B C : Point) : ℝ := sorry

/-- Check if a point is in the first quadrant -/
def Point.isInFirstQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y > 0

/-- The main theorem -/
theorem ellipse_eccentricity_for_sixty_degree_angle (rc : RelatedCurves) 
  (P : Point) (h_intersection : sorry) 
  (h_first_quadrant : P.isInFirstQuadrant) 
  (h_angle : angle rc.foci.1 P rc.foci.2 = 60 * π / 180) : 
  rc.ellipse.eccentricity = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_for_sixty_degree_angle_l128_12869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_duration_l128_12872

/-- The number of days A takes to finish the work -/
noncomputable def a_days : ℝ := 9

/-- The number of days B worked before leaving -/
noncomputable def b_worked : ℝ := 10

/-- The number of days A takes to finish the remaining work after B left -/
noncomputable def a_remaining : ℝ := 3

/-- The number of days B takes to finish the entire work -/
noncomputable def b_days : ℝ := 31/3

theorem b_work_duration :
  ∃ (total_work : ℝ),
  total_work > 0 ∧
  total_work = b_worked * (total_work / b_days) + a_remaining * (total_work / a_days) := by
  sorry

#check b_work_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_duration_l128_12872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_chord_property_l128_12893

/-- A hexagon inscribed in a circle with specific side lengths -/
structure InscribedHexagon where
  sides₁ : ℝ  -- Length of three consecutive sides
  sides₂ : ℝ  -- Length of the other three consecutive sides
  is_inscribed : Bool  -- Indicates if the hexagon is inscribed in a circle

/-- The chord that divides the hexagon into two trapezoids -/
noncomputable def dividing_chord (h : InscribedHexagon) : ℝ := sorry

/-- Theorem stating the properties of the dividing chord -/
theorem dividing_chord_property (h : InscribedHexagon) 
  (h_inscribed : h.is_inscribed = true) 
  (h_sides : h.sides₁ = 4 ∧ h.sides₂ = 6) :
  ∃ (p q : ℕ), 
    Nat.Coprime p q ∧ 
    dividing_chord h = p / q ∧ 
    p + q = 799 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_chord_property_l128_12893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equality_l128_12832

variable (a b c : ℝ × ℝ × ℝ)

theorem dot_product_equality 
  (eq_zero : (3 : ℝ) • a + (4 : ℝ) • b + (5 : ℝ) • c = (0, 0, 0))
  (norm_a : ‖a‖ = 1)
  (norm_b : ‖b‖ = 1)
  (norm_c : ‖c‖ = 1) :
  b • (a + c) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equality_l128_12832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_is_144_l128_12830

/-- The radius of the initial sphere in centimeters -/
def sphere_radius : ℝ := 12

/-- The radius of the wire's cross-section in centimeters -/
def wire_radius : ℝ := 4

/-- The volume of a sphere given its radius -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

/-- The volume of a cylinder given its radius and height -/
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The theorem stating that the length of the wire is 144 cm -/
theorem wire_length_is_144 :
  ∃ (h : ℝ), sphere_volume sphere_radius = cylinder_volume wire_radius h ∧ h = 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_is_144_l128_12830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l128_12873

-- Define the line equation
def line_equation (x y m : ℝ) : ℝ := 3 * y - 2 * x - m

-- Define the condition for points being on opposite sides of the line
def opposite_sides (m : ℝ) : Prop :=
  (line_equation 2 1 m) * (line_equation 5 (-1) m) < 0

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, opposite_sides m ↔ -13 < m ∧ m < -1 := by
  intro m
  apply Iff.intro
  · intro h
    sorry -- Proof for the forward direction
  · intro h
    sorry -- Proof for the reverse direction


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l128_12873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_overlap_length_l128_12819

/-- The horizontal length in meters of overlapping paper pieces -/
noncomputable def horizontalLength (paperWidth : ℝ) (numPieces : ℕ) (overlap : ℝ) : ℝ :=
  (paperWidth + (numPieces - 1 : ℝ) * (paperWidth - overlap)) / 100

/-- Theorem stating the horizontal length of 15 pieces of 25 cm wide paper with 0.5 cm overlap -/
theorem paper_overlap_length :
  horizontalLength 25 15 0.5 = 3.68 := by
  -- Unfold the definition of horizontalLength
  unfold horizontalLength
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_overlap_length_l128_12819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pass_level_game_l128_12880

/-- A fair six-faced die -/
inductive Die : Type
| one | two | three | four | five | six

/-- The number of points on a face of the die -/
def Die.points : Die → ℕ
| Die.one   => 1
| Die.two   => 2
| Die.three => 3
| Die.four  => 4
| Die.five  => 5
| Die.six   => 6

/-- The condition for passing a level -/
def passLevel (n : ℕ) (rolls : List Die) : Prop :=
  rolls.length = n ∧ (rolls.map Die.points).sum > 2^n

/-- The maximum number of levels that can be passed -/
def maxLevels : ℕ := 4

/-- The probability of passing the first three levels -/
def probFirstThreeLevels : ℚ := 100 / 243

/-- A placeholder for the probability measure -/
axiom Prob : (List Die → Prop) → ℚ

/-- The main theorem stating the maximum number of levels and the probability of passing the first three -/
theorem pass_level_game :
  (∀ n : ℕ, n > maxLevels → ¬∃ rolls : List Die, passLevel n rolls) ∧
  (Prob (λ rolls => ∃ r₁ r₂ r₃ : List Die, passLevel 1 r₁ ∧ passLevel 2 r₂ ∧ passLevel 3 r₃) = probFirstThreeLevels) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pass_level_game_l128_12880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_max_sum_l128_12822

/-- Circle C in polar coordinates -/
def circle_C (ρ θ : ℝ) : Prop :=
  ρ^2 = 4*ρ*(Real.cos θ + Real.sin θ) - 6

/-- Parametric equations of circle C -/
noncomputable def parametric_C (θ : ℝ) : ℝ × ℝ :=
  (2 + Real.sqrt 2 * Real.cos θ, 2 + Real.sqrt 2 * Real.sin θ)

/-- The sum x + y for a point on circle C -/
noncomputable def sum_xy (θ : ℝ) : ℝ :=
  let (x, y) := parametric_C θ
  x + y

theorem circle_C_max_sum :
  (∀ θ : ℝ, sum_xy θ ≤ 6) ∧
  (∃ θ : ℝ, sum_xy θ = 6) ∧
  (sum_xy (π/4) = 6) ∧
  (parametric_C (π/4) = (3, 3)) := by
  sorry

#eval "Circle C theorem compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_max_sum_l128_12822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l128_12861

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1

-- Define the maximum value M(a) on [1,3]
noncomputable def M (a : ℝ) : ℝ := max (f a 1) (f a 3)

-- Define the minimum value N(a) on [1,3]
noncomputable def N (a : ℝ) : ℝ := min (f a 1) (f a 3)

-- Define g(a) = M(a) - N(a)
noncomputable def g (a : ℝ) : ℝ := M a - N a

-- Theorem statement
theorem min_value_of_g :
  ∀ a : ℝ, 1/3 ≤ a ∧ a ≤ 1 → g a ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l128_12861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l128_12849

noncomputable section

def I : Set ℝ := Set.Icc (-1) 1

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_domain : ∀ x, x ∈ I → f x ∈ I)
  (h_f1 : f 1 = 1)
  (h_ineq : ∀ a b, a ∈ I → b ∈ I → a + b ≠ 0 → (f a + f b) / (a + b) > 0) :
  (∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) ∧
  (∀ m : ℝ, (∃ x, x ∈ I ∧ ∀ a, a ∈ I → f x ≥ m^2 - 2*a*m - 2) → m ∈ I) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l128_12849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_value_l128_12845

/-- Given three points A(1,3), B(5,8), and C(29,a) that are collinear, 
    prove that a = 38 --/
theorem collinear_points_value (a : ℝ) : 
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (5, 8)
  let C : ℝ × ℝ := (29, a)
  (B.2 - A.2) * (C.1 - B.1) = (C.2 - B.2) * (B.1 - A.1) → a = 38 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_value_l128_12845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_set_arithmetic_mean_l128_12809

theorem square_set_arithmetic_mean (p : ℕ) (M : Finset ℕ) : 
  Nat.Prime p → 
  p % 2 = 1 → 
  Finset.card M = (p^2 + 1) / 2 → 
  (∀ m ∈ M, ∃ k : ℕ, m = k^2) → 
  ∃ S : Finset ℕ, S ⊆ M ∧ Finset.card S = p ∧ (∃ n : ℕ, (Finset.sum S id) / p = n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_set_arithmetic_mean_l128_12809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_three_l128_12870

noncomputable def f (x : ℝ) : ℝ := Real.exp (-5 * x) + 2

theorem tangent_line_at_zero_three :
  let p : ℝ × ℝ := (0, 3)
  let tangent_line (x : ℝ) := -5 * x + 3
  (∀ x, tangent_line x = f p.1 + (deriv f p.1) * (x - p.1)) ∧
  f p.1 = p.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_three_l128_12870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_expense_calculation_l128_12874

noncomputable def monthly_salary : ℝ := 5000
noncomputable def tax_rate : ℝ := 0.10
noncomputable def late_rent_fraction : ℝ := 3/5
def late_rent_months : ℕ := 2

theorem rent_expense_calculation :
  let after_tax_salary := monthly_salary * (1 - tax_rate)
  let total_late_rent := after_tax_salary * late_rent_fraction
  let monthly_rent := total_late_rent / late_rent_months
  monthly_rent = 1350 := by
  -- Unfold definitions
  unfold monthly_salary tax_rate late_rent_fraction late_rent_months
  -- Perform calculations
  -- This is where we would normally write the proof steps
  -- For now, we'll use sorry to skip the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_expense_calculation_l128_12874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l128_12868

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (w : ℂ), Complex.abs w = 3 ∧ 
    Complex.abs ((1 + 2 * Complex.I) * w^4 - w^6) = 81 * Real.sqrt 5 * (9 - Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l128_12868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_parabola_l128_12848

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 6 * Real.tan θ * (1 / Real.cos θ)

-- Define the Cartesian equation of a parabola
def parabola_equation (x y : ℝ) : Prop := x^2 = 6 * y

-- Theorem statement
theorem polar_equation_is_parabola :
  ∀ (r θ x y : ℝ), 
    polar_equation r θ → 
    x = r * Real.cos θ → 
    y = r * Real.sin θ → 
    parabola_equation x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_parabola_l128_12848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l128_12824

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then 1 / x else |x + 1|

theorem a_range (a : ℝ) :
  (∀ x y, x < y ∧ y < a → f a x > f a y) ∧
  (∀ x y, a < x ∧ x < y → f a x < f a y) →
  a ∈ Set.Icc (-1) 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l128_12824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_joined_after_four_months_l128_12817

/-- Represents the number of months after which z joined the business -/
def months_after_start : ℕ → Prop := sorry

/-- The investment amounts of x, y, and z -/
def investment_x : ℕ := 36000
def investment_y : ℕ := 42000
def investment_z : ℕ := 48000

/-- The total profit and z's share in the profit -/
def total_profit : ℕ := 13860
def z_profit_share : ℕ := 4032

/-- The theorem stating that z joined 4 months after x and y started the business -/
theorem z_joined_after_four_months :
  ∃ (m : ℕ), months_after_start m ∧
  (z_profit_share : ℚ) / (total_profit - z_profit_share : ℚ) =
  (investment_z * (12 - m) : ℚ) / ((investment_x + investment_y) * 12 : ℚ) →
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_joined_after_four_months_l128_12817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_zeros_bound_l128_12881

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sin (Real.pi * x) - 3 * (x - 1) * Real.log (x + 1) - m

theorem tangent_line_and_zeros_bound (m : ℝ) :
  (∃ x₁ x₂, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ f m x₁ = 0 ∧ f m x₂ = 0 ∧ x₁ ≠ x₂) →
  (∀ x, (deriv (f 0)) 0 * x = (Real.pi + 3) * x) ∧
  ∃ x₁ x₂, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ f m x₁ = 0 ∧ f m x₂ = 0 ∧ 
    |x₁ - x₂| ≤ 1 - (2 * m) / (Real.pi + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_zeros_bound_l128_12881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_on_negative_reals_a_range_when_bounded_l128_12813

-- Define the function f(x) as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + a * (1/2)^x + (1/4)^x

-- Part 1: Unboundedness on (-∞, 0) when a = 1
theorem unbounded_on_negative_reals :
  ¬ ∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), x < 0 → |f 1 x| ≤ M :=
by sorry

-- Part 2: Range of a when bounded on [0, +∞) with upper bound 3
theorem a_range_when_bounded (a : ℝ) :
  (∀ (x : ℝ), x ≥ 0 → |f a x| ≤ 3) → a ∈ Set.Icc (-5) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_on_negative_reals_a_range_when_bounded_l128_12813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_in_terms_of_a_and_b_l128_12860

theorem sin_x_in_terms_of_a_and_b 
  (x a b : ℝ) 
  (h1 : Real.tan x = (2 * a * b) / (a^2 - b^2))
  (h2 : a > b)
  (h3 : b > 0)
  (h4 : 0 < x)
  (h5 : x < Real.pi/2) : 
  Real.sin x = (2 * a * b) / (a^2 + b^2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_in_terms_of_a_and_b_l128_12860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_n_for_triangle_inequality_l128_12855

-- Define a triangle with the given conditions
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  angle_sum : angleA + angleC = 2 * angleB

-- Define the property we want to prove
def satisfies_inequality (n : ℕ) (t : Triangle) : Prop :=
  t.a^n + t.c^n ≤ 2 * t.b^n

-- State the theorem
theorem greatest_n_for_triangle_inequality :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (t : Triangle), satisfies_inequality n t) ∧
  (∀ (m : ℕ), m > n → ∃ (t : Triangle), ¬satisfies_inequality m t) :=
by
  -- The proof goes here
  sorry

#check greatest_n_for_triangle_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_n_for_triangle_inequality_l128_12855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_six_digit_l128_12838

def integers_with_six (n : ℕ) : Bool :=
  (n ≥ 10 && n ≤ 99) && (n % 10 = 6 || n / 10 = 6)

theorem probability_of_six_digit : 
  (Finset.filter (fun n => integers_with_six n) (Finset.range 90)).card / 90 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_six_digit_l128_12838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_C_l128_12800

noncomputable section

/-- The function f(x) as defined in the problem -/
def f (x φ : Real) : Real :=
  2 * Real.sin x * Real.cos (φ / 2) ^ 2 + Real.cos x * Real.sin φ - Real.sin x

/-- Triangle ABC with given properties -/
structure Triangle where
  a : Real
  b : Real
  A : Real
  h_a : a = 1
  h_b : b = Real.sqrt 2
  h_fA : f A (π / 2) = Real.sqrt 3 / 2

/-- The main theorem to prove -/
theorem triangle_angle_C (t : Triangle) : 
  t.A + Real.arcsin (t.b * Real.sin t.A / t.a) + Real.arccos (t.a / t.b * Real.cos t.A) = π ∧ 
  (Real.arccos (t.a / t.b * Real.cos t.A) = 7 * π / 12 ∨ 
   Real.arccos (t.a / t.b * Real.cos t.A) = π / 12) := by
  sorry

/-- Helper lemma: f(x) takes its minimum at x = π -/
lemma f_min_at_pi (φ : Real) (h : 0 < φ ∧ φ < π) : 
  ∀ x, f π φ ≤ f x φ := by
  sorry

/-- Helper lemma: φ = π/2 -/
lemma phi_value (φ : Real) (h : 0 < φ ∧ φ < π) : 
  (∀ x, f π φ ≤ f x φ) → φ = π / 2 := by
  sorry

/-- Helper lemma: Simplified f(x) = cos(x) when φ = π/2 -/
lemma f_simplified (x : Real) : f x (π / 2) = Real.cos x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_C_l128_12800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_divisible_numbers_exist_l128_12831

theorem consecutive_divisible_numbers_exist : 
  ∃ n : ℕ, 
    (∀ i : ℕ, i ∈ Finset.range 99 → (n + i) % (100 - i) = 0) ∧ 
    n % 100 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_divisible_numbers_exist_l128_12831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l128_12807

theorem divisibility_property (A : Finset ℕ) : 
  A.card = 26 → 
  (∀ S : Finset ℕ, S ⊆ A → S.card = 6 → ∃ x y : ℕ, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ x ∣ y) → 
  ∃ S : Finset ℕ, S ⊆ A ∧ S.card = 6 ∧ ∃ x ∈ S, ∀ y ∈ S, y ≠ x → x ∣ y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l128_12807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_science_only_students_l128_12867

theorem science_only_students 
  (total : ℕ) 
  (science : ℕ) 
  (history : ℕ) 
  (h1 : total = 120) 
  (h2 : science = 85) 
  (h3 : history = 65) 
  (h4 : ∀ s, s ∈ Finset.range total → 
       (s < science ∨ s < history)) : 
  science - (science + history - total) = 55 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_science_only_students_l128_12867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l128_12804

-- Define the function f
noncomputable def f (k : ℤ) (x : ℝ) : ℝ := x^(-k^2 + k + 2)

-- Define the function g
noncomputable def g (p : ℝ) (x : ℝ) : ℝ := 1 - p * (x^2) + (2*p - 1) * x

-- Theorem statement
theorem function_properties :
  (∃ k : ℤ, f k 2 < f k 3 ∧ (k = 0 ∨ k = 1)) ∧
  (∃! p : ℝ, p > 0 ∧
    (∀ x, x ∈ Set.Icc (-1 : ℝ) 2 → g p x ∈ Set.Icc (-4 : ℝ) (17/8)) ∧
    (∃ x y, x ∈ Set.Icc (-1 : ℝ) 2 ∧ y ∈ Set.Icc (-1 : ℝ) 2 ∧ g p x = -4 ∧ g p y = 17/8) ∧
    p = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l128_12804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l128_12888

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the point of tangency
def point : ℝ × ℝ := (1, 0)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), (∀ (x y : ℝ), y = m * x + b ↔ x - y - 1 = 0) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - point.fst| < δ → 
    |f x - (m * x + b) - point.snd| < ε * |x - point.fst|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l128_12888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_B_is_three_l128_12811

def B : Set ℤ := {n : ℤ | ∃ x : ℤ, n = (2*x - 1) + (2*x + 1) + (2*x + 3)}

theorem gcd_of_B_is_three : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ (n.natAbs)) ∧ 
  (∀ m : ℕ, (∀ n ∈ B, m ∣ (n.natAbs)) → m ∣ d) ∧ d = 3 := by
  sorry

#check gcd_of_B_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_B_is_three_l128_12811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_cycle_four_l128_12820

/-- A graph representing students and their acquaintances -/
structure StudentGraph where
  -- The number of students (vertices)
  n : ℕ
  -- The adjacency relation (edge existence)
  adj : Fin n → Fin n → Bool
  -- Adjacency is symmetric (knowing is mutual)
  adj_symm : ∀ i j, adj i j = adj j i
  -- Each student knows at least 45 others
  min_degree : ∀ i, (Finset.univ.filter (λ j => adj i j)).card ≥ 45

/-- A cycle of length 4 in the graph -/
def CycleFour (G : StudentGraph) (a b c d : Fin G.n) : Prop :=
  G.adj a b ∧ G.adj b c ∧ G.adj c d ∧ G.adj d a

/-- Main theorem: There exists a cycle of length 4 in the student graph -/
theorem exists_cycle_four (G : StudentGraph) (h : G.n = 2021) : 
  ∃ (a b c d : Fin G.n), CycleFour G a b c d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_cycle_four_l128_12820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_machines_l128_12842

/-- Represents the problem of determining the number of closed machines in a factory. -/
theorem closed_machines (total_machines : ℕ) (annual_output : ℝ) (profit_percentage : ℝ) 
  (profit_decrease : ℝ) (closed_machines : ℕ) : 
  total_machines = 14 →
  annual_output = 70000 →
  profit_percentage = 0.125 →
  profit_decrease = 0.125 →
  (annual_output * (total_machines - closed_machines : ℝ) / total_machines) * profit_percentage = 
    annual_output * profit_percentage * (1 - profit_decrease) →
  closed_machines = 2 := by
  sorry

/-- Represents the relationship between output and number of machines. -/
noncomputable def output_proportion (total_machines : ℕ) (annual_output : ℝ) (active_machines : ℕ) : ℝ :=
  annual_output * (active_machines : ℝ) / total_machines

/-- Represents the profit calculation based on output and profit percentage. -/
noncomputable def profit_calculation (output : ℝ) (profit_percentage : ℝ) : ℝ :=
  output * profit_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_machines_l128_12842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_inequality_l128_12812

noncomputable def f (x : ℝ) : ℝ := 1 / x

noncomputable def average_rate_of_change (f : ℝ → ℝ) (a b : ℝ) : ℝ := (f b - f a) / (b - a)

noncomputable def k1 : ℝ := average_rate_of_change f 1 2
noncomputable def k2 : ℝ := average_rate_of_change f 2 3
noncomputable def k3 : ℝ := average_rate_of_change f 3 4

theorem average_rate_of_change_inequality : k3 < k2 ∧ k2 < k1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_inequality_l128_12812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_siding_cost_l128_12837

/-- Calculates the cost of siding for Sandy's playhouse --/
theorem siding_cost 
  (wall_width : ℝ) (wall_height : ℝ) (roof_base : ℝ) (roof_height : ℝ)
  (siding_width : ℝ) (siding_height : ℝ) (siding_cost : ℝ) : 
  wall_width = 8 ∧ wall_height = 6 ∧ roof_base = 8 ∧ roof_height = 5 ∧
  siding_width = 10 ∧ siding_height = 12 ∧ siding_cost = 30 →
  ⌈((2 * wall_width * wall_height + roof_base * roof_height) / (siding_width * siding_height))⌉ * siding_cost = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_siding_cost_l128_12837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_implies_base_inequality_l128_12899

theorem log_inequality_implies_base_inequality (m n : ℝ) : 
  (Real.log 9 / Real.log m < Real.log 9 / Real.log n) ∧ (Real.log 9 / Real.log n < 0) → 
  0 < m ∧ m < n ∧ n < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_implies_base_inequality_l128_12899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_minimum_value_condition_l128_12883

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1/2) + 2 / (2 * x + 1)

/-- Theorem for part (I) -/
theorem monotone_increasing_condition (a : ℝ) :
  (a > 0) →
  (∀ x ∈ Set.Ioi 0, Monotone (f a)) →
  a ≥ 2 :=
sorry

/-- Theorem for part (II) -/
theorem minimum_value_condition (a : ℝ) :
  (∃ x ∈ Set.Ioi 0, f a x = 1) ∧
  (∀ x ∈ Set.Ioi 0, f a x ≥ 1) ↔
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_minimum_value_condition_l128_12883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_minimize_distance_to_center_l128_12887

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (3 + (1/2) * t, (Real.sqrt 3 / 2) * t)

-- Define the circle C in polar form
def circle_C_polar (θ : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin θ

-- Define the circle C in Cartesian form
def circle_C_cartesian (x y : ℝ) : Prop := x^2 + (y - Real.sqrt 3)^2 = 3

-- Define the center of circle C
def center_C : ℝ × ℝ := (0, Real.sqrt 3)

-- Define the distance function between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

end noncomputable section

-- Statement 1: Prove that the Cartesian equation of circle C is correct
theorem circle_C_equation : ∀ θ : ℝ, 
  let (x, y) := (circle_C_polar θ * Real.cos θ, circle_C_polar θ * Real.sin θ)
  circle_C_cartesian x y := by sorry

-- Statement 2: Prove that P(3, 0) minimizes the distance to the center of C
theorem minimize_distance_to_center : 
  ∀ t : ℝ, distance (line_l 0) center_C ≤ distance (line_l t) center_C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_minimize_distance_to_center_l128_12887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_theorem_l128_12828

/-- Calculates the speed of a car given distance and time -/
noncomputable def calculate_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem car_speed_theorem (distance : ℝ) (time : ℝ) 
  (h1 : distance = 642) 
  (h2 : time = 6.5) : 
  ∃ (speed : ℝ), abs (calculate_speed distance time - speed) < 0.5 ∧ speed = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_theorem_l128_12828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_A_in_triangle_l128_12825

theorem tan_A_in_triangle (a b c : ℝ) (A B C : ℝ) : 
  0 < A → A < Real.pi / 2 →  -- A is acute
  b = 3 * a * Real.sin B →  -- given condition
  Real.tan A = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_A_in_triangle_l128_12825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_score_difference_l128_12851

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  total_innings : ℕ
  total_runs : ℕ
  avg_all : ℚ
  avg_excluding_extremes : ℚ
  highest_score : ℕ

/-- Calculates the difference between highest and lowest scores -/
def score_difference (stats : BatsmanStats) : ℤ :=
  stats.highest_score - (stats.total_runs - stats.highest_score - 
    (stats.avg_excluding_extremes * (stats.total_innings - 2 : ℚ)).floor)

/-- Theorem stating the difference between highest and lowest scores -/
theorem batsman_score_difference (stats : BatsmanStats) 
  (h1 : stats.total_innings = 46)
  (h2 : stats.avg_all = 60)
  (h3 : stats.avg_excluding_extremes = 58)
  (h4 : stats.highest_score = 199)
  (h5 : stats.total_runs = (stats.avg_all * stats.total_innings).floor) :
  score_difference stats = 190 := by
  sorry

#eval score_difference { 
  total_innings := 46, 
  total_runs := 2760, 
  avg_all := 60, 
  avg_excluding_extremes := 58, 
  highest_score := 199 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_score_difference_l128_12851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l128_12864

noncomputable def vector_a : ℝ × ℝ := (1, Real.sqrt 3)
def vector_b : ℝ × ℝ := (3, 0)

theorem angle_between_vectors : 
  Real.arccos ((vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) / 
    (Real.sqrt (vector_a.1^2 + vector_a.2^2) * Real.sqrt (vector_b.1^2 + vector_b.2^2))) = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l128_12864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_value_l128_12853

open BigOperators

theorem smallest_n_value (m n : ℕ) (h : 3^m * n = (7 * 6 * 5 * 4 * 3 * 2 * 1) + (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) + (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) : 
  ∃ (k : ℕ), k ≥ n ∧ k = 560 ∧ (∃ (l : ℕ), 3^l * k = (7 * 6 * 5 * 4 * 3 * 2 * 1) + (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) + (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_value_l128_12853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_monotone_decreasing_range_l128_12801

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

-- Define has_extremum
def has_extremum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- Define is_tangent_line_at
def is_tangent_line_at (f : ℝ → ℝ) (p : ℝ × ℝ) (q : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ), DifferentiableAt ℝ f p.1 ∧ 
    (deriv f p.1 = m) ∧ 
    q.2 - p.2 = m * (q.1 - p.1)

-- Part 1: Tangent line at (0, f(0))
theorem tangent_line_at_zero (a : ℝ) :
  (∃ (x : ℝ), x = -3 ∧ has_extremum (f a) x) →
  ∃ (m b : ℝ), m = 3 ∧ b = -9 ∧ ∀ (x y : ℝ), y = m*x + b ↔ is_tangent_line_at (f a) (0, f a 0) (x, y) :=
sorry

-- Part 2: Range of a for monotonically decreasing f on [1, 2]
theorem monotone_decreasing_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, StrictMonoOn (fun x ↦ -(f a x)) (Set.Icc 1 2)) →
  a ≤ -15/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_monotone_decreasing_range_l128_12801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_subtraction_l128_12898

def is_valid_digit (d : Nat) : Prop := d > 0 ∧ d < 10

def three_digit_number (a b c : Nat) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c

theorem three_digit_subtraction
  (a b c : Nat)
  (h1 : three_digit_number a b c)
  (h2 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h3 : (100 * c + 10 * b + a) - (100 * a + 10 * b + c) = 5)
  : b ∈ ({2, 3, 4, 5, 7, 8, 9} : Set Nat) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_subtraction_l128_12898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_and_perpendicular_tangents_l128_12876

noncomputable def curve (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (ω * x) - Real.cos (ω * x)

noncomputable def curve_derivative (ω : ℝ) (x : ℝ) : ℝ := 
  ω * (Real.sqrt 2 * Real.cos (ω * x) + Real.sin (ω * x))

def is_center_of_symmetry (ω : ℝ) (x : ℝ) : Prop :=
  curve ω x = 0 ∧ ∃ k : ℤ, ω * x = Real.pi * k + Real.arctan (Real.sqrt 2 / 2)

theorem curve_symmetry_and_perpendicular_tangents (ω : ℝ) :
  ω > 0 →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    is_center_of_symmetry ω x₁ ∧
    is_center_of_symmetry ω x₂ ∧
    curve_derivative ω x₁ * curve_derivative ω x₂ = -1 →
  ω = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_and_perpendicular_tangents_l128_12876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equals_three_l128_12854

/-- Define the nested radical function recursively -/
noncomputable def nestedRadical : ℕ → ℝ
  | 0 => Real.sqrt (1 + 2018 * 2020)
  | n + 1 => Real.sqrt (1 + (2017 - n) * nestedRadical n)

/-- The main theorem stating that the nested radical equals 3 -/
theorem nested_radical_equals_three : nestedRadical 2017 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equals_three_l128_12854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_ratio_l128_12892

/-- Represents a trapezium ABCD with AB parallel to DC -/
structure Trapezium where
  a : ℝ  -- Length of AD
  b : ℝ  -- Length of AB
  h : a < b  -- Condition that a < b

/-- The ratio of the area of triangle DOC to the area of trapezium ABCD -/
noncomputable def area_ratio : ℝ := 2 / 9

theorem trapezium_ratio (T : Trapezium) : 
  area_ratio = 2 / 9 → T.a / T.b = (2 + 3 * Real.sqrt 2) / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_ratio_l128_12892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l128_12846

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the points D, E, F
variable (D E F : ℝ × ℝ)

-- Define the lengths of segments
noncomputable def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
noncomputable def area_triangle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define the area of a quadrilateral
noncomputable def area_quadrilateral (p q r s : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem length_AB (t : Triangle) 
  (h1 : length t.A D = 2 * length t.B D)
  (h2 : length t.A D = length E t.C)
  (h3 : length t.B t.C = 18)
  (h4 : area_triangle t.A F t.C = area_quadrilateral D t.B E F) :
  length t.A t.B = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l128_12846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sin_minus_x_over_x_cubed_limit_ln_minus_x_plus_x_squared_over_x_cubed_l128_12877

-- Part a
theorem limit_sin_minus_x_over_x_cubed :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → 
    |((Real.sin x - x) / x^3) + 1/6| < ε := sorry

-- Part b
theorem limit_ln_minus_x_plus_x_squared_over_x_cubed :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → 
    |((Real.log (1+x) - x + x^2/2) / x^3) - 1/3| < ε := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sin_minus_x_over_x_cubed_limit_ln_minus_x_plus_x_squared_over_x_cubed_l128_12877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_roots_l128_12890

theorem compare_roots : 
  (4 : ℝ)^((1/4) : ℝ) > (5 : ℝ)^((1/5) : ℝ) ∧ 
  (5 : ℝ)^((1/5) : ℝ) > max ((7 : ℝ)^((1/7) : ℝ)) ((12 : ℝ)^((1/12) : ℝ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_roots_l128_12890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_half_sum_roots_l128_12805

theorem tangent_half_sum_roots (a : ℝ) (α β : ℝ) : 
  a > 1 →
  α ∈ Set.Ioo (-Real.pi/2) (Real.pi/2) →
  β ∈ Set.Ioo (-Real.pi/2) (Real.pi/2) →
  (Real.tan α)^2 + 4*a*(Real.tan α) + 3*a + 1 = 0 →
  (Real.tan β)^2 + 4*a*(Real.tan β) + 3*a + 1 = 0 →
  Real.tan ((α + β) / 2) = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_half_sum_roots_l128_12805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l128_12857

-- Define the points and line
def A : ℝ × ℝ := (1, 0)
def l (x : ℝ) : ℝ := 2 * x - 4

-- Define point R on line l
def R (m : ℝ) : ℝ × ℝ := (m, l m)

-- Define point P
variable (P : ℝ × ℝ)

-- State the theorem
theorem trajectory_of_P (m : ℝ) (h : R m - A = P - R m) :
  P.2 = 2 * P.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l128_12857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heather_distance_is_correct_l128_12806

/-- The distance between Stacy and Heather's starting points in miles -/
noncomputable def initial_distance : ℝ := 40

/-- Heather's walking speed in miles per hour -/
noncomputable def heather_speed : ℝ := 5

/-- The time difference between Stacy and Heather's start times in hours -/
noncomputable def time_difference : ℝ := 24 / 60

/-- Stacy's walking speed in miles per hour -/
noncomputable def stacy_speed : ℝ := heather_speed + 1

/-- The time it takes for Heather and Stacy to meet -/
noncomputable def meeting_time : ℝ := (initial_distance - stacy_speed * time_difference) / (heather_speed + stacy_speed)

/-- The distance Heather has walked when they meet -/
noncomputable def heather_distance : ℝ := heather_speed * meeting_time

theorem heather_distance_is_correct : 
  ∃ ε > 0, abs (heather_distance - 17.09) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heather_distance_is_correct_l128_12806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_on_cube_l128_12840

/-- Predicate to check if a face is not visited by a given path --/
def face_not_visited (path_length : ℝ) (face : Fin 6) : Prop :=
  sorry

/-- The shortest path length for a fly traversing all faces of a cube --/
theorem shortest_path_on_cube (a : ℝ) (h : a > 0) :
  ∃ (path_length : ℝ),
    path_length = 3 * a * Real.sqrt 2 ∧
    ∀ (other_path : ℝ),
      (other_path ≥ path_length ∨
       ∃ (face : Fin 6), face_not_visited other_path face) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_on_cube_l128_12840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_monotonicity_depends_on_a_max_value_of_sum_l128_12856

noncomputable section

variable (a b x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := -2 * a * Real.log x + 2 * (a + 1) * x - x^2

theorem tangent_line_parallel (h : a > 0) :
  (∃ (a : ℝ), ∀ x ∈ Set.Ioi 0, 
    (deriv (f a)) x = 1 → x = 2 → a = 3) := sorry

theorem monotonicity_depends_on_a (h : a > 0) :
  ∃ (P : ℝ → Prop), ∀ x ∈ Set.Ioi 0, 
    Monotone (f a) ↔ P a := sorry

theorem max_value_of_sum (h : a > 0) :
  (∀ x ∈ Set.Ioi 0, f a x ≥ -x^2 + 2*a*x + b) →
  a + b ≤ 2 * Real.sqrt (Real.exp 1) := sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_monotonicity_depends_on_a_max_value_of_sum_l128_12856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l128_12841

theorem division_remainder_problem (a b : ℕ) 
  (h1 : a - b = 1365)
  (h2 : a = 1620)
  (h3 : a / b = 6) : 
  a % b = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l128_12841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_plus_area_sum_l128_12810

/-- A parallelogram with vertices at (1, 3), (5, 3), (7, 8), and (3, 8) -/
structure Parallelogram where
  v1 : ℝ × ℝ := (1, 3)
  v2 : ℝ × ℝ := (5, 3)
  v3 : ℝ × ℝ := (7, 8)
  v4 : ℝ × ℝ := (3, 8)

/-- The perimeter of the parallelogram -/
noncomputable def perimeter (p : Parallelogram) : ℝ :=
  let d1 := ((p.v2.1 - p.v1.1)^2 + (p.v2.2 - p.v1.2)^2).sqrt
  let d2 := ((p.v4.1 - p.v1.1)^2 + (p.v4.2 - p.v1.2)^2).sqrt
  2 * (d1 + d2)

/-- The area of the parallelogram -/
def area (p : Parallelogram) : ℝ :=
  let base := p.v2.1 - p.v1.1
  let height := p.v4.2 - p.v1.2
  base * height

/-- Theorem: The sum of the perimeter and area of the given parallelogram is 28 + 2√29 -/
theorem perimeter_plus_area_sum (p : Parallelogram) : perimeter p + area p = 28 + 2 * Real.sqrt 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_plus_area_sum_l128_12810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dmitry_arrival_probability_l128_12886

/-- Represents the arrival times of family members -/
structure ArrivalTimes (m : ℝ) where
  x : ℝ
  y : ℝ
  z : ℝ
  x_bounds : 0 < x ∧ x < m
  y_z_bounds : 0 < y ∧ y < z ∧ z < m

/-- The probability that Dmitry arrived before his father -/
noncomputable def probability_dmitry_before_father (m : ℝ) : ℝ := 2/3

/-- Theorem stating that the probability of Dmitry arriving before his father is 2/3 -/
theorem dmitry_arrival_probability (m : ℝ) (h : m > 0) :
  probability_dmitry_before_father m = 2/3 := by
  -- The proof goes here
  sorry

#check dmitry_arrival_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dmitry_arrival_probability_l128_12886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_ratio_l128_12894

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if a^2 + b^2 = 2c^2, then (sin^2 A + sin^2 B) / sin^2 C = 2 -/
theorem triangle_sine_ratio (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  a^2 + b^2 = 2 * c^2 →
  (Real.sin A)^2 + (Real.sin B)^2 = 2 * (Real.sin C)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_ratio_l128_12894
