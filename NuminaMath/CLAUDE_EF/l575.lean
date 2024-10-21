import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equal_parts_l575_57512

theorem complex_equal_parts (a : ℝ) : 
  (Complex.re ((1 + a * Complex.I) * (2 + Complex.I)) = Complex.im ((1 + a * Complex.I) * (2 + Complex.I))) → 
  a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equal_parts_l575_57512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l575_57574

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 1 / (3 * x + b)

-- Define the inverse function
noncomputable def f_inv (x : ℝ) : ℝ := (1 - 3 * x) / (3 * x)

-- Theorem statement
theorem inverse_function_condition (b : ℝ) :
  (∀ x, f_inv (f b x) = x) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l575_57574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decaf_percentage_theorem_l575_57553

/-- Represents the coffee stock and its composition -/
structure CoffeeStock where
  total : ℝ
  typeA_percent : ℝ
  typeB_percent : ℝ
  typeC_percent : ℝ
  typeA_decaf_percent : ℝ
  typeB_decaf_percent : ℝ
  typeC_decaf_percent : ℝ

/-- Calculates the percentage of decaffeinated coffee in the updated stock -/
noncomputable def decaf_percentage (initial : CoffeeStock) (additional : CoffeeStock) : ℝ :=
  let total_weight := initial.total + additional.total
  let initial_decaf := initial.total * (
    initial.typeA_percent * initial.typeA_decaf_percent +
    initial.typeB_percent * initial.typeB_decaf_percent +
    initial.typeC_percent * initial.typeC_decaf_percent
  )
  let additional_decaf := additional.total * (
    additional.typeA_percent * additional.typeA_decaf_percent +
    additional.typeB_percent * additional.typeB_decaf_percent +
    additional.typeC_percent * additional.typeC_decaf_percent
  )
  let total_decaf := initial_decaf + additional_decaf
  (total_decaf / total_weight) * 100

/-- Theorem stating that the percentage of decaffeinated coffee in the updated stock is approximately 26.39% -/
theorem decaf_percentage_theorem (initial : CoffeeStock) (additional : CoffeeStock)
  (h1 : initial.total = 1200)
  (h2 : initial.typeA_percent = 0.3)
  (h3 : initial.typeB_percent = 0.5)
  (h4 : initial.typeC_percent = 0.2)
  (h5 : initial.typeA_decaf_percent = 0.1)
  (h6 : initial.typeB_decaf_percent = 0.25)
  (h7 : initial.typeC_decaf_percent = 0.55)
  (h8 : additional.total = 200)
  (h9 : additional.typeA_percent = 0.45)
  (h10 : additional.typeB_percent = 0.3)
  (h11 : additional.typeC_percent = 0.25)
  (h12 : additional.typeA_decaf_percent = 0.1)
  (h13 : additional.typeB_decaf_percent = 0.25)
  (h14 : additional.typeC_decaf_percent = 0.55) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |decaf_percentage initial additional - 26.39| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decaf_percentage_theorem_l575_57553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_szekeres_theorem_l575_57523

theorem erdos_szekeres_theorem (n : ℕ) (a : ℕ → ℝ) :
  ∃ (s : Finset ℕ),
    s.card = n + 1 ∧
    (∀ i j, i ∈ s → j ∈ s → i < j → a i < a j) ∨
    (∀ i j, i ∈ s → j ∈ s → i < j → a i > a j) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_szekeres_theorem_l575_57523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_point_l575_57531

-- Define the points
noncomputable def A : ℝ × ℝ := (0, -2)
noncomputable def B : ℝ × ℝ := (6, 0)
noncomputable def C : ℝ → ℝ × ℝ := λ a => (0, a)

-- Define the midpoint of AB
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the slope of AB
noncomputable def slope_AB : ℝ := (B.2 - A.2) / (B.1 - A.1)

-- Define the slope of the perpendicular bisector
noncomputable def slope_perp_bisector : ℝ := -1 / slope_AB

-- Define the equation of the perpendicular bisector
def perp_bisector_eq (x y : ℝ) : Prop :=
  y - D.2 = slope_perp_bisector * (x - D.1)

-- State the theorem
theorem perpendicular_bisector_point (a : ℝ) :
  perp_bisector_eq (C a).1 (C a).2 → a = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_point_l575_57531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sixth_minus_tan_squared_l575_57593

theorem tan_sixth_minus_tan_squared (x : ℝ) 
  (h : ∃ r : ℝ, (Real.cos x) * r = Real.tan x ∧ (Real.tan x) * r = Real.sin x) : 
  (Real.tan x) ^ 6 - (Real.tan x) ^ 2 = - (Real.cos x) * (Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sixth_minus_tan_squared_l575_57593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_phi_value_l575_57560

noncomputable def f (φ : Real) (x : Real) : Real := Real.sin (2 * x + φ)

theorem possible_phi_value :
  ∃ (φ : Real),
    (∀ (x : Real), f φ (x + π/8) = Real.sin (2*x + π/4 + φ)) ∧
    (∀ (x : Real), f φ (x + π/8) = f φ (-x + π/8)) ∧
    φ = π/4 :=
by sorry

#check possible_phi_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_phi_value_l575_57560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_at_critical_point_l575_57500

/-- The quadratic function f(x) = 3x^2 + 9x - 7 -/
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 9 * x - 7

/-- The critical point of f -/
noncomputable def critical_point : ℝ := -3/2

theorem f_minimum_at_critical_point :
  ∀ x : ℝ, f x ≥ f critical_point := by
  sorry

#check f_minimum_at_critical_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_at_critical_point_l575_57500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_total_payment_l575_57569

-- Define constants
def gum_packs : ℕ := 5
def candy_bars : ℕ := 4
def chip_bags : ℕ := 2
def candy_bar_price : ℚ := 3/2  -- 1.5 as a rational number
def discount_rate : ℚ := 1/10   -- 0.1 as a rational number

-- Define price relations
def gum_price : ℚ := candy_bar_price / 2
def chip_price : ℚ := candy_bar_price * 2

-- Calculate total cost before discount
def total_cost_before_discount : ℚ :=
  gum_packs * gum_price + candy_bars * candy_bar_price + chip_bags * chip_price

-- Calculate discount amount
def discount_amount : ℚ := total_cost_before_discount * discount_rate

-- Calculate total cost after discount
def total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount

-- Theorem to prove
theorem john_total_payment :
  (total_cost_after_discount * 100).floor / 100 = 1417 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_total_payment_l575_57569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_d_230_l575_57590

theorem lcm_d_230 (d : ℕ) (h1 : ¬ 2 ∣ d) (h2 : d > 0) (h3 : ¬ 3 ∣ 230) :
  Nat.lcm d 230 = 230 * d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_d_230_l575_57590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segment_length_l575_57529

/-- Triangle ABC with given side lengths --/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB_length : dist A B = 24)
  (AC_length : dist A C = 26)
  (BC_length : dist B C = 30)

/-- The centroid of a triangle --/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- A line segment parallel to BC containing the centroid --/
structure ParallelSegment (t : Triangle) :=
  (D E : ℝ × ℝ)
  (on_AB : D.1 = t.A.1 + r * (t.B.1 - t.A.1) ∧ D.2 = t.A.2 + r * (t.B.2 - t.A.2))
  (on_AC : E.1 = t.A.1 + s * (t.C.1 - t.A.1) ∧ E.2 = t.A.2 + s * (t.C.2 - t.A.2))
  (parallel_to_BC : (E.2 - D.2) / (E.1 - D.1) = (t.C.2 - t.B.2) / (t.C.1 - t.B.1))
  (contains_centroid : ∃ (k : ℝ), centroid t = (D.1 + k * (E.1 - D.1), D.2 + k * (E.2 - D.2)))
  (r s : ℝ)

/-- The main theorem --/
theorem parallel_segment_length (t : Triangle) (seg : ParallelSegment t) :
  dist seg.D seg.E = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segment_length_l575_57529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_distances_l575_57525

/-- The ellipse with equation x²/4 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}

/-- The foci of the ellipse -/
noncomputable def Foci : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-Real.sqrt 3, 0), (Real.sqrt 3, 0))

/-- A point on the ellipse -/
noncomputable def P : ℝ × ℝ := sorry

/-- Assertion that P is on the ellipse -/
axiom P_on_ellipse : P ∈ Ellipse

/-- The midpoint of PF₁ -/
noncomputable def Midpoint : ℝ × ℝ :=
  ((P.1 + Foci.1.1) / 2, (P.2 + Foci.1.2) / 2)

/-- Assertion that the midpoint of PF₁ is on the y-axis -/
axiom midpoint_on_yaxis : Midpoint.1 = 0

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The statement to be proved -/
theorem ratio_of_distances : 
  distance P Foci.2 / distance P Foci.1 = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_distances_l575_57525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l575_57544

/-- The function g as defined in the problem -/
noncomputable def g (n : ℝ) : ℝ := (1/4) * n * (n+1) * (n+2) * (n+3)

/-- The main theorem to prove -/
theorem g_difference (r : ℝ) : g r - g (r-1) = r * (r+1) * (r+2) := by
  -- Expand the definition of g
  unfold g
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l575_57544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_segment_AB_l575_57598

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + p.2^2 = (p.1 + 2)^2}

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define the line l
def l : Set (ℝ × ℝ) :=
  {p | p.2 = p.1 - 1}

-- Theorem statement
theorem curve_C_and_segment_AB :
  (∀ p ∈ C, (p.1 - 1)^2 + p.2^2 = (p.1 + 2)^2) →
  (C = {p : ℝ × ℝ | p.2^2 = 4 * p.1}) ∧
  (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ A ∈ l ∧ B ∈ l ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_segment_AB_l575_57598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_decreasing_range_l575_57554

/-- A function f: ℝ → ℝ is decreasing if for all x, y ∈ ℝ, x < y implies f(x) > f(y) -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

/-- The exponential function with base (a+1) -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ (a + 1) ^ x

theorem exponential_decreasing_range (a : ℝ) :
  DecreasingFunction (f a) → -1 < a ∧ a < 0 := by
  sorry

#check exponential_decreasing_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_decreasing_range_l575_57554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_fixed_point_zeros_sum_greater_than_two_l575_57586

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (|x - 1|) - a / x

def hasTwoZeros (a : ℝ) (x₁ x₂ : ℝ) : Prop :=
  f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ ≠ x₂

theorem tangent_line_fixed_point (a : ℝ) :
  ∃ (m b : ℝ), ∀ x, m * x + b = f a x + (deriv (f a)) 2 * (x - 2) - f a 2 →
  m * 4 + b = 2 := by sorry

theorem zeros_sum_greater_than_two (a : ℝ) (x₁ x₂ : ℝ) :
  a < 0 → hasTwoZeros a x₁ x₂ → x₁ + x₂ > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_fixed_point_zeros_sum_greater_than_two_l575_57586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_exists_l575_57513

noncomputable def x : ℕ → ℝ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 4 * x n + 3) / (x n + 7)

theorem smallest_m_exists :
  ∃ m : ℕ, m ∈ Set.Icc 201 300 ∧
    x m ≤ 3 + 1 / (2^30) ∧
    ∀ k : ℕ, k < m → x k > 3 + 1 / (2^30) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_exists_l575_57513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_is_96_div_7_l575_57555

/-- The average number of minutes run per day by students in three grades -/
noncomputable def average_minutes_run (e : ℝ) : ℝ :=
  let sixth_graders := e / 4
  let seventh_graders := e / 2
  let eighth_graders := e
  let total_students := sixth_graders + seventh_graders + eighth_graders
  let total_minutes := 8 * sixth_graders + 12 * seventh_graders + 16 * eighth_graders
  total_minutes / total_students

/-- Theorem stating that the average number of minutes run per day is 96/7 -/
theorem average_minutes_run_is_96_div_7 :
  ∀ e : ℝ, e > 0 → average_minutes_run e = 96 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_is_96_div_7_l575_57555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_distance_constant_l575_57545

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the point C
def point_C (c : ℝ) : ℝ × ℝ := (0, c)

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem chord_distance_constant :
  ∃ c : ℝ, ∀ xa ya xb yb : ℝ,
    circle_eq xa ya → circle_eq xb yb →
    let (cx, cy) := point_C c
    (cy - ya) * (xb - xa) = (cy - yb) * (xa - cx) →
    Real.sqrt (distance cx cy xa ya ^2 + distance cx cy xb yb ^2) = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_distance_constant_l575_57545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l575_57502

noncomputable def f (x : ℝ) := Real.sin x * Real.cos (x - Real.pi / 2)

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧
    (∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q)) ∧
  (∃ c : ℝ, ∀ x : ℝ, f (2 * c - x) = f x) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l575_57502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spectral_density_correlation_function_relationship_l575_57591

/-- Spectral density function -/
noncomputable def spectral_density (D α ω : ℝ) : ℝ := 
  (D * α) / (Real.pi * (α^2 + ω^2))

/-- Correlation function -/
noncomputable def correlation_function (D α τ : ℝ) : ℝ := 
  D * Real.exp (-α * abs τ)

/-- Theorem stating the relationship between spectral density and correlation function -/
theorem spectral_density_correlation_function_relationship 
  (D α : ℝ) (h_D : D > 0) (h_α : α > 0) :
  ∀ τ : ℝ, 
    correlation_function D α τ = 
    ∫ (ω : ℝ), spectral_density D α ω * Complex.exp (Complex.I * τ * ω) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spectral_density_correlation_function_relationship_l575_57591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_nearest_square_is_two_l575_57576

-- Define the ceiling function as noncomputable
noncomputable def ceiling (x : ℝ) : ℤ := Int.ceil x

-- Define the distance to nearest square function
def distToNearestSquare (x : ℤ) : ℕ :=
  Int.natAbs (x - Int.natAbs (Int.sqrt x)^2)

-- State the theorem
theorem distance_to_nearest_square_is_two :
  ∃ (A : ℝ), ∀ (n : ℕ), distToNearestSquare (ceiling (A^n)) = 2 := by
  -- The proof is omitted and replaced with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_nearest_square_is_two_l575_57576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_25360000_l575_57552

-- Define the given condition
axiom cube_root_25_36 : Real.rpow 25.36 (1/3) = 2.938

-- State the theorem to be proved
theorem cube_root_25360000 : Real.rpow 25360000 (1/3) = 293.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_25360000_l575_57552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_gross_profit_l575_57584

-- Define the cost of the jacket
def cost : ℚ := 42

-- Define the markup percentage based on cost
noncomputable def markup_percentage (c : ℚ) : ℚ :=
  if c < 50 then 30/100
  else if c ≤ 80 then 25/100
  else 20/100

-- Define the original selling price
noncomputable def original_selling_price (c : ℚ) : ℚ :=
  c / (1 - markup_percentage c)

-- Define the first discount percentage
def first_discount_percentage : ℚ := 15/100

-- Define the second discount percentage
def second_discount_percentage : ℚ := 20/100

-- Define the final selling price after both discounts
noncomputable def final_selling_price (c : ℚ) : ℚ :=
  let original_price := original_selling_price c
  let after_first_discount := original_price * (1 - first_discount_percentage)
  after_first_discount * (1 - second_discount_percentage)

-- Define the gross profit
noncomputable def gross_profit (c : ℚ) : ℚ :=
  final_selling_price c - c

-- Theorem statement
theorem merchant_gross_profit :
  gross_profit cost = -(6/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_gross_profit_l575_57584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_sqrt_plus_three_l575_57540

theorem definite_integral_sqrt_plus_three : 
  ∫ x in (Set.Icc 0 1), (Real.sqrt x + 3) = 11/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_sqrt_plus_three_l575_57540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_60_l575_57543

/-- The speed of a train in km/hr, given its length and time to cross a pole -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: The speed of a train is 60 km/hr -/
theorem train_speed_is_60 (length time : ℝ) 
  (h1 : length = 50) 
  (h2 : time = 3) : 
  train_speed length time = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_60_l575_57543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_not_parabola_l575_57510

-- Define the circle
def Circle (O : EuclideanSpace ℝ (Fin 2)) (r : ℝ) := 
  {P : EuclideanSpace ℝ (Fin 2) | ‖P - O‖ = r}

-- Define the perpendicular bisector of a segment
def PerpendicularBisector (A B : EuclideanSpace ℝ (Fin 2)) := 
  {P : EuclideanSpace ℝ (Fin 2) | ‖P - A‖ = ‖P - B‖}

-- Define the line through two points
def Line (A B : EuclideanSpace ℝ (Fin 2)) := 
  {P : EuclideanSpace ℝ (Fin 2) | ∃ t : ℝ, P = A + t • (B - A)}

-- Define the locus of point P
def Locus (O A : EuclideanSpace ℝ (Fin 2)) := 
  {P : EuclideanSpace ℝ (Fin 2) | ∃ B : EuclideanSpace ℝ (Fin 2), 
    B ∈ Circle O 1 ∧ P ∈ PerpendicularBisector A B ∧ P ∈ Line O B}

-- Define IsParabola (this is a placeholder definition)
def IsParabola (S : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- State the theorem
theorem locus_not_parabola (O A : EuclideanSpace ℝ (Fin 2)) :
  ¬ IsParabola (Locus O A) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_not_parabola_l575_57510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_kiosk_placement_l575_57557

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Helper function to calculate Euclidean distance between two points -/
noncomputable def dist (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The main theorem about optimal placement of three kiosks in a circular area -/
theorem optimal_kiosk_placement (R : ℝ) (h : R > 0) :
  ∃ (p₁ p₂ p₃ : Point),
    (∀ q : Point, (q.x^2 + q.y^2 ≤ R^2) →
      min (dist q p₁) (min (dist q p₂) (dist q p₃)) ≤ R * Real.sqrt 3 / 2) ∧
    (∀ (p₁' p₂' p₃' : Point),
      (∃ q : Point, (q.x^2 + q.y^2 ≤ R^2) ∧
        min (dist q p₁') (min (dist q p₂') (dist q p₃')) > R * Real.sqrt 3 / 2) →
      ∃ q : Point, (q.x^2 + q.y^2 ≤ R^2) ∧
        min (dist q p₁') (min (dist q p₂') (dist q p₃')) >
        min (dist q p₁) (min (dist q p₂) (dist q p₃))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_kiosk_placement_l575_57557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_eighth_power_l575_57563

theorem fourth_root_sixteen_to_eighth_power : (16 : ℝ) ^ ((1/4 : ℝ) * 8) = 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_eighth_power_l575_57563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l575_57559

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := y^2 / 3 - x^2 / 2 = 1

/-- The asymptotic line equation -/
def asymptotic_line_eq (x y : ℝ) : Prop := y = Real.sqrt 6 / 2 * x ∨ y = -(Real.sqrt 6 / 2 * x)

/-- Theorem: The asymptotic line equation of the given hyperbola -/
theorem hyperbola_asymptote :
  ∀ x y : ℝ, hyperbola_eq x y → asymptotic_line_eq x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l575_57559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_B_l575_57511

-- Define the weights of A, B, and C as variables
variable (A B C : ℝ)

-- State the conditions as axioms
axiom avg_ABC : (A + B + C) / 3 = 45
axiom avg_AB : (A + B) / 2 = 40
axiom avg_BC : (B + C) / 2 = 43

-- State the theorem to be proved
theorem weight_of_B : B = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_B_l575_57511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_11236_l575_57522

theorem largest_prime_factor_of_11236 :
  (Nat.factors 11236).maximum? = some 53 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_11236_l575_57522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l575_57595

/-- Two circles with equations x^2 + y^2 + 2ax + a^2 - 4 = 0 and x^2 + y^2 - 4by - 1 + 4b^2 = 0 -/
def circle1 (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*a*x + a^2 - 4 = 0

def circle2 (b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

/-- The two circles have exactly three common tangents -/
def have_three_common_tangents (a b : ℝ) : Prop := sorry

/-- The minimum value of 4/a^2 + 1/b^2 -/
noncomputable def min_value (a b : ℝ) : ℝ := (4 / a^2) + (1 / b^2)

theorem min_value_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : have_three_common_tangents a b) : 
  ∃ (m : ℝ), m = 16/9 ∧ ∀ (x : ℝ), min_value a b ≥ x → x ≤ m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l575_57595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_from_dot_product_equality_l575_57539

/-- 
Given a triangle ABC where the dot product of AB and BC equals the dot product of BC and CA,
prove that the triangle is isosceles.
-/
theorem isosceles_triangle_from_dot_product_equality 
  (A B C : ℝ × ℝ) 
  (h : (B.1 - A.1, B.2 - A.2) • (C.1 - B.1, C.2 - B.2) = 
       (C.1 - B.1, C.2 - B.2) • (A.1 - C.1, A.2 - C.2)) :
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = 
  (B.1 - A.1)^2 + (B.2 - A.2)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_from_dot_product_equality_l575_57539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hole_radius_in_rectangular_solid_l575_57588

/-- The radius of a cylindrical hole drilled through a rectangular solid -/
noncomputable def hole_radius (l w h : ℝ) : ℝ :=
  let surface_area_before := 2 * (l * w + l * h + w * h)
  let r := h / 2
  let surface_area_after := surface_area_before - 2 * (Real.pi * r^2) + 2 * Real.pi * r * h
  r

theorem hole_radius_in_rectangular_solid :
  hole_radius 3 8 9 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hole_radius_in_rectangular_solid_l575_57588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l575_57548

noncomputable def a (x : ℝ) : ℝ × ℝ := (3 * Real.sqrt 3 * Real.sin x, Real.sqrt 3 * Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

noncomputable def g (x : ℝ) : ℝ := f x + 1

theorem min_value_of_g (x : ℝ) (h : x ∈ Set.Icc (-π/3) (π/3)) :
  g x ≥ -1/2 ∧ g (-π/3) = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l575_57548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_payment_duration_l575_57546

/-- Proves that Mike made payments for 12 months given the specified conditions -/
theorem mikes_payment_duration
  (lower_rate : ℕ)
  (higher_rate : ℕ)
  (lower_payments : ℕ)
  (higher_payments : ℕ)
  (total_paid : ℕ)
  (h : lower_rate * lower_payments + higher_rate * higher_payments = total_paid)
  : lower_payments + higher_payments = 12 :=
by
  -- The proof goes here
  sorry

#check mikes_payment_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_payment_duration_l575_57546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_timber_volume_calculation_l575_57581

/-- Calculates the timber volume after two years given initial volume, growth rate, and annual harvest. -/
def timber_volume_after_two_years (a p b : ℝ) : ℝ :=
  a * (1 + p)^2 - (2 + p) * b

/-- Theorem stating that the timber volume after two years is correctly calculated. -/
theorem timber_volume_calculation (a p b : ℝ) :
  timber_volume_after_two_years a p b = a * (1 + p)^2 - (2 + p) * b :=
by
  -- Unfold the definition of timber_volume_after_two_years
  unfold timber_volume_after_two_years
  -- The equality holds by definition
  rfl

#check timber_volume_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_timber_volume_calculation_l575_57581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_royalties_sales_ratio_decrease_l575_57589

/-- Calculates the percentage decrease in the ratio of royalties to sales --/
theorem royalties_sales_ratio_decrease (first_sales : ℝ) (first_royalties : ℝ) 
  (second_sales : ℝ) (second_royalties : ℝ) :
  first_sales = 20 →
  first_royalties = 4 →
  second_sales = 108 →
  second_royalties = 9 →
  let first_ratio := first_royalties / first_sales
  let second_ratio := second_royalties / second_sales
  let percentage_decrease := (first_ratio - second_ratio) / first_ratio * 100
  |percentage_decrease - 58.35| < 0.01 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_royalties_sales_ratio_decrease_l575_57589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l575_57561

-- Define the propositions
def universal_prop := ∀ x : ℝ, 2*x + 5 > 0
def equation_prop := ∀ x : ℝ, x^2 + 5*x = 6

-- Define logical variables
variable (p q : Prop)

-- Theorem statement
theorem problem_statement :
  -- 1. "For all x ∈ ℝ, 2x+5>0" is a universal proposition
  (universal_prop ∧ True) ∧
  -- 2. The negation of "For all x ∈ ℝ, x²+5x=6" is not "There exists x ∉ ℝ such that x²+5x ≠ 6"
  (¬(¬equation_prop ↔ ∃ x : ℝ, x^2 + 5*x ≠ 6)) ∧
  -- 3. |x|=|y| does not always imply x=y
  (∃ x y : ℝ, |x| = |y| ∧ x ≠ y) ∧
  -- 4. If p ∨ q is false, then both p and q are false
  (¬(p ∨ q) → ¬p ∧ ¬q) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l575_57561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statements_l575_57520

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x
noncomputable def g (x : ℝ) : ℝ := x - Real.log x

theorem problem_statements :
  (∀ x > 0, Monotone (λ y ↦ g (Real.exp y))) ∧
  (∀ x > 1, (∀ a ≥ 2/Real.exp 1, f (a*x) ≥ f (Real.log (x^2))) ∧
            (∀ ε > 0, ∃ x > 1, f ((2/Real.exp 1 - ε)*x) < f (Real.log (x^2)))) ∧
  (∀ x₁ x₂ t, t > 2 → x₂ > x₁ → x₁ > 0 → f x₁ = t → g x₂ = t →
    (∀ y₁ y₂ s, s > 2 → y₂ > y₁ → y₁ > 0 → f y₁ = s → g y₂ = s →
      (Real.log t)/(x₂ - x₁) ≥ (Real.log s)/(y₂ - y₁)) ∧
    (Real.log t)/(x₂ - x₁) ≤ 1/Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statements_l575_57520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_sum_of_squared_differences_l575_57580

theorem minimum_sum_of_squared_differences (a : ℝ) : 
  (∀ a₁ a₂ a₃ a₄ : ℝ, ∃ k₁ k₂ k₃ k₄ : ℤ, 
    (a₁ - k₁ - (a₂ - k₂))^2 + (a₁ - k₁ - (a₃ - k₃))^2 + 
    (a₁ - k₁ - (a₄ - k₄))^2 + (a₂ - k₂ - (a₃ - k₃))^2 + 
    (a₂ - k₂ - (a₄ - k₄))^2 + (a₃ - k₃ - (a₄ - k₄))^2 ≤ a) → 
  a ≥ 1.75 := by
  sorry

#check minimum_sum_of_squared_differences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_sum_of_squared_differences_l575_57580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l575_57538

/-- Given that 2m men can complete a job in d days, 
    prove that 2m+2r men will complete the same job in md/(m+r) days -/
theorem job_completion_time 
  (m d r : ℕ) 
  (h : m > 0 ∧ d > 0) 
  (job_time : ℕ → ℝ → ℝ) 
  (h_original : job_time (2 * m) d = d) :
  job_time (2 * m + 2 * r) ((m * d : ℝ) / (m + r : ℝ)) = (m * d : ℝ) / (m + r : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l575_57538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_higher_score_from_lower_average_class_l575_57504

/-- Represents a class with an average score -/
structure ClassScore where
  averageScore : ℝ

/-- Represents a student with a score -/
structure StudentScore where
  score : ℝ

/-- Theorem: Given two classes with different average scores, it's possible for a student
    from the class with lower average to have a higher score than a student from the class
    with higher average -/
theorem possible_higher_score_from_lower_average_class
  (class1 class2 : ClassScore)
  (h_diff : class1.averageScore < class2.averageScore) :
  ∃ (student1 student2 : StudentScore),
    student1.score > student2.score := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_higher_score_from_lower_average_class_l575_57504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l575_57568

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ :=
  Real.sqrt (1 - (b / a)^2)

/-- A circle with radius r -/
structure Circle (r : ℝ) where
  h : r > 0

/-- The theorem states that if the line connecting the right focus and upper vertex of an ellipse
    is tangent to a specific circle, then the eccentricity of the ellipse is either 1/2 or √3/2 -/
theorem ellipse_eccentricity_theorem (a b : ℝ) (e : Ellipse a b) (c : Circle (Real.sqrt ((3 * a^2) / 16))) :
  (∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ 
   (b * x + Real.sqrt (a^2 - b^2) * y - b * Real.sqrt (a^2 - b^2))^2 / 
   (b^2 + (a^2 - b^2)) = (3 * a^2) / 16) →
  eccentricity e = 1/2 ∨ eccentricity e = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l575_57568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equals_sin_l575_57582

theorem cos_shift_equals_sin (φ : ℝ) : 
  -π ≤ φ ∧ φ < π →
  (∀ x, Real.cos (2*(x - π/2) + φ) = Real.sin (2*x + π/3)) →
  φ = 5*π/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equals_sin_l575_57582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l575_57507

theorem power_equation_solution (x : ℝ) : (3 : ℝ)^(x - 2) = (9 : ℝ)^3 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l575_57507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l575_57524

-- Define the ellipse
structure Ellipse where
  focus : ℝ × ℝ
  major_minor_ratio : ℝ

-- Define our specific ellipse
noncomputable def our_ellipse : Ellipse where
  focus := (Real.sqrt 2, 0)
  major_minor_ratio := Real.sqrt 3

-- Define the standard form of an ellipse equation
def is_standard_ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Theorem statement
theorem ellipse_equation (e : Ellipse) (x y : ℝ) :
  e.focus.1 = Real.sqrt 2 ∧ 
  e.focus.2 = 0 ∧ 
  e.major_minor_ratio = Real.sqrt 3 →
  is_standard_ellipse_equation 3 1 x y := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l575_57524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_zero_l575_57585

theorem cubic_function_zero (s : ℝ) : 
  (λ x : ℝ => 3 * x^3 - 2 * x^2 + 4 * x + s) (-1) = 0 ↔ s = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_zero_l575_57585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_diff_is_24_div_7_l575_57521

/-- Represents the probability of the coin being blown away on a single throw -/
noncomputable def p : ℝ := 1 / 25

/-- Represents the expected value of |H - T| -/
noncomputable def expected_diff : ℝ := (1 - p) / Real.sqrt (p * (2 - p))

/-- The main theorem stating that the expected value of |H - T| is 24/7 -/
theorem expected_diff_is_24_div_7 : expected_diff = 24 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_diff_is_24_div_7_l575_57521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l575_57594

/-- The slope of a line parallel to 3x - 6y = 21 is 1/2 -/
theorem parallel_line_slope : ∀ (m b : ℝ),
  (∀ x y : ℝ, 3 * x - 6 * y = 21 → y = m * x + b) →
  m = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l575_57594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_comparison_l575_57503

-- Define the pricing functions for Plan ① and Plan ②
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 3 then 10
  else if x ≤ 5 then 10 + 1.5 * (x - 3)
  else 13 + 2.5 * (x - 5)

noncomputable def g (x : ℝ) : ℝ :=
  let k := ⌊x / 5⌋
  if x ≤ 5 * k + 3 then 13 * k + 10
  else 13 * k + 10 + 1.5 * (x - (5 * k + 3))

-- Theorem statement
theorem taxi_fare_comparison (x : ℝ) (h : x > 0) :
  (x ≤ 5 → f x = g x) ∧
  (x > 5 → f x < g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_comparison_l575_57503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximately_ten_percent_l575_57508

/-- Calculates the compound interest rate given the principal, time, and final amount -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (final_amount : ℝ) : ℝ :=
  ((final_amount / principal) ^ (1 / time)) - 1

/-- Theorem stating that the given financial scenario results in approximately 10% interest rate -/
theorem interest_rate_approximately_ten_percent 
  (principal : ℝ) 
  (time : ℝ) 
  (final_amount : ℝ) 
  (h1 : principal = 4000)
  (h2 : time = 3)
  (h3 : final_amount = 5324.000000000002) :
  ∃ (ε : ℝ), ε > 0 ∧ |calculate_interest_rate principal time final_amount - 0.1| < ε := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use the following instead:
#check calculate_interest_rate 4000 3 5324.000000000002

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximately_ten_percent_l575_57508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_real_domain_l575_57573

/-- The function f(x) = √(ax² - ax + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a * x^2 - a * x + 1)

/-- The domain of f is all real numbers -/
def has_real_domain (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - a * x + 1 ≥ 0

/-- The range of a for which f has a real domain is [0, 4] -/
theorem range_of_a_for_real_domain :
  {a : ℝ | has_real_domain a} = Set.Icc 0 4 := by
  sorry

#check range_of_a_for_real_domain

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_real_domain_l575_57573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_l575_57536

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem tangent_line_perpendicular (a b : ℝ) : 
  (∃ (m : ℝ), (deriv (f a)) 1 = m ∧ m * (-1/3) = -1) → 
  f a 1 = b → 
  2 * a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_l575_57536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squared_distances_l575_57541

/-- Given collinear points A, B, C, D, and F, prove the minimum sum of squared distances to any point P -/
theorem min_sum_squared_distances (A B C D F : ℝ) (P : ℝ) : 
  A < B → B < C → C < D → D < F →
  B - A = 1 →
  C - B = 1 →
  D - C = 3 →
  F - D = 8 →
  (A - P)^2 + (B - P)^2 + (C - P)^2 + (D - P)^2 + (F - P)^2 ≥ 130.8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squared_distances_l575_57541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_even_product_l575_57516

/-- Spinner A with 4 equally likely outcomes -/
def spinner_A : Finset ℕ := {1, 2, 3, 4}

/-- Spinner B with 3 equally likely outcomes -/
def spinner_B : Finset ℕ := {1, 2, 3}

/-- The product of two numbers is even -/
def is_even_product (a b : ℕ) : Prop := Even (a * b)

/-- The probability of an event occurring in a finite sample space -/
def probability (event : Finset (ℕ × ℕ)) (sample_space : Finset (ℕ × ℕ)) : ℚ :=
  (event.card : ℚ) / (sample_space.card : ℚ)

/-- The sample space of all possible outcomes when spinning both spinners -/
def sample_space : Finset (ℕ × ℕ) := spinner_A.product spinner_B

/-- The event of getting an even product when spinning both spinners -/
noncomputable def even_product_event : Finset (ℕ × ℕ) :=
  sample_space.filter (fun p => Even (p.1 * p.2))

theorem probability_of_even_product :
  probability even_product_event sample_space = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_even_product_l575_57516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chosen_number_is_five_l575_57567

theorem chosen_number_is_five : ∃! (x : ℕ), x > 0 ∧ ((10 * x + 5) - x^2) / x - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chosen_number_is_five_l575_57567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l575_57501

-- Define the ellipse parameters
variable (a b : ℝ)

-- Define the conditions
axiom h1 : a > b
axiom h2 : b > 0

-- Define the intersection points
variable (x₁ y₁ x₂ y₂ : ℝ)

-- Define the conditions for the intersection points
axiom intersection_line : y₁ = -x₁ + 1 ∧ y₂ = -x₂ + 1
axiom intersection_ellipse : x₁^2 / a^2 + y₁^2 / b^2 = 1 ∧ x₂^2 / a^2 + y₂^2 / b^2 = 1

-- Define the midpoint condition
axiom midpoint_condition : (x₁ + x₂) / 2 - 2 * (y₁ + y₂) / 2 = 0

-- Define the right focus
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt (a^2 - b^2), 0)

-- Define the symmetric point of the right focus
noncomputable def symmetric_point : ℝ × ℝ := 
  let (fx, fy) := right_focus a b
  let x := (3 * fx + 2 * fy) / 5
  let y := (4 * fx - 2 * fy) / 5
  (x, y)

-- Define the condition for the symmetric point
axiom symmetric_point_on_circle : 
  let (x, y) := symmetric_point a b
  x^2 + y^2 = 4

-- State the theorem
theorem ellipse_properties : 
  -- Eccentricity is √2/2
  Real.sqrt (a^2 - b^2) / a = Real.sqrt 2 / 2 ∧
  -- Equation of the ellipse is x^2/8 + y^2/4 = 1
  a^2 = 8 ∧ b^2 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l575_57501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_to_2_l575_57537

def divisors_of_150 : List Nat := [2, 3, 5, 6, 10, 15, 25, 30, 50, 75, 150]

def has_common_factor (a b : Nat) : Prop :=
  ∃ (f : Nat), f > 1 ∧ f ∣ a ∧ f ∣ b

def valid_arrangement (arr : List Nat) : Prop :=
  arr.length = divisors_of_150.length ∧
  (∀ x ∈ arr, x ∈ divisors_of_150) ∧
  (∀ i, has_common_factor (arr[i]!) (arr[(i + 1) % arr.length]!))

theorem sum_of_adjacent_to_2 (arr : List Nat) (h : valid_arrangement arr) :
  ∃ i, arr[i]! = 2 ∧ arr[(i - 1 + arr.length) % arr.length]! + arr[(i + 1) % arr.length]! = 16 := by
  sorry

#check sum_of_adjacent_to_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_to_2_l575_57537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_prime_l575_57506

-- Define the quadratic function
noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define variables
variable (a b c m₁ m₂ a₂ : ℝ)

-- Define the conditions
axiom a_gt_b : a > b
axiom b_gt_1 : b > 1
axiom f_1_eq_0 : f a b c 1 = 0
axiom condition_eq : a^2 + a₂ * f a b c m₁ + f a b c m₂ - f a b c m₁ * f a b c m₂ = 0

-- Define primality for natural numbers
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a function to convert real numbers to natural numbers
noncomputable def real_to_nat (x : ℝ) : ℕ := 
  if x < 0 then 0 else Int.toNat ⌊x⌋

-- Define the theorem
theorem smallest_k_prime :
  ∃ k : ℕ, k = 3 ∧ 
  (∀ j : ℕ, j < k → ¬(is_prime (real_to_nat (f a b c (m₁ + j))) ∨ is_prime (real_to_nat (f a b c (m₂ + j))))) ∧
  (is_prime (real_to_nat (f a b c (m₁ + k))) ∨ is_prime (real_to_nat (f a b c (m₂ + k)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_prime_l575_57506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l575_57528

/-- The line l in rectangular coordinates -/
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

/-- The circle M in rectangular coordinates -/
def circle_M (x y : ℝ) : Prop := (x - 0)^2 + (y - 2)^2 = 4

/-- The length of the chord AB formed by the intersection of line_l and circle_M -/
noncomputable def chord_length : ℝ := Real.sqrt 14

theorem intersection_chord_length :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    circle_M A.1 A.2 ∧ circle_M B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = chord_length :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l575_57528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_form_C_range_x_plus_y_on_C_l575_57547

-- Define the curve C
noncomputable def C (φ : ℝ) : ℝ × ℝ := (4 * Real.cos φ, 3 * Real.sin φ)

-- Theorem for the general form of curve C
theorem general_form_C :
  ∀ (x y : ℝ), (∃ φ, C φ = (x, y)) ↔ x^2/16 + y^2/9 = 1 := by
  sorry

-- Theorem for the range of x + y on curve C
theorem range_x_plus_y_on_C :
  ∀ (x y : ℝ), (∃ φ, C φ = (x, y)) → -5 ≤ x + y ∧ x + y ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_form_C_range_x_plus_y_on_C_l575_57547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_below_line_l575_57578

/-- The equation of the line --/
def line_equation (x y : ℝ) : Prop := 16 * x + 144 * y = 1152

/-- The x-intercept of the line --/
def x_intercept : ℝ := 72

/-- The y-intercept of the line --/
def y_intercept : ℝ := 8

/-- The negative reciprocal of the slope of the line --/
def negative_reciprocal_slope : ℝ := 9

/-- The number of squares crossed by the diagonal --/
def squares_crossed_by_diagonal : ℕ := 79

/-- The total number of squares in the rectangle --/
def total_squares : ℕ := 576

/-- Theorem stating the number of squares below the line --/
theorem squares_below_line :
  (total_squares - squares_crossed_by_diagonal : ℚ) / 2 = 248.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_below_line_l575_57578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l575_57592

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) : Type :=
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : a > b)

/-- A point on the ellipse -/
def PointOnEllipse {a b : ℝ} (E : Ellipse a b) : Type :=
  {p : ℝ × ℝ // (p.1 / a) ^ 2 + (p.2 / b) ^ 2 = 1}

/-- The upper vertex of the ellipse -/
def upperVertex {a b : ℝ} (E : Ellipse a b) : ℝ × ℝ := (0, b)

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity {a b : ℝ} (E : Ellipse a b) : ℝ := 
  Real.sqrt (1 - (b / a) ^ 2)

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

/-- The theorem statement -/
theorem ellipse_eccentricity_range {a b : ℝ} (E : Ellipse a b) :
  (∀ (p : PointOnEllipse E), distance p.val (upperVertex E) ≤ 2 * b) →
  0 < eccentricity E ∧ eccentricity E ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l575_57592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_polynomial_unique_solution_l575_57566

theorem prime_polynomial_unique_solution (x : ℕ) :
  x > 0 → Prime (x^5 + x + 1) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_polynomial_unique_solution_l575_57566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l575_57505

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the ellipse -/
def Ellipse.contains (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The line y = mx + c -/
structure Line where
  m : ℝ
  c : ℝ

/-- The length of the line segment intercepted by the ellipse and a line -/
noncomputable def interceptLength (e : Ellipse) (l : Line) : ℝ :=
  sorry -- Definition omitted as it's not directly given in the problem

theorem ellipse_properties (e : Ellipse) 
    (h1 : e.contains ⟨0, Real.sqrt 3⟩) 
    (h2 : e.eccentricity = 1/2) :
  (e.a = 2 ∧ e.b = Real.sqrt 3) ∧ 
  interceptLength e ⟨1, -1⟩ = 24/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l575_57505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_on_x_axis_l575_57571

/-- Parabola type representing y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to calculate the y-coordinate of the vertex of a parabola -/
noncomputable def vertexY (p : Parabola) : ℝ :=
  (4 * p.a * p.c - p.b^2) / (4 * p.a)

/-- Theorem stating that if the vertex of y = x^2 + 2x + c lies on the x-axis, then c = 1 -/
theorem parabola_vertex_on_x_axis (c : ℝ) :
  let p := Parabola.mk 1 2 c
  vertexY p = 0 → c = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_on_x_axis_l575_57571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_sixth_l575_57517

/-- The area enclosed by y = x and y = 2x - x^2 -/
noncomputable def enclosed_area : ℝ := ∫ x in (0)..(1), (x - (2*x - x^2))

/-- Theorem: The area enclosed by y = x and y = 2x - x^2 is 1/6 -/
theorem area_is_one_sixth : enclosed_area = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_sixth_l575_57517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_sum_is_365_5_l575_57534

/-- A hexagon inscribed in a circle with five sides of length 75 and one side of length 27 -/
structure InscribedHexagon where
  /-- The circle in which the hexagon is inscribed -/
  circle : Set ℝ × ℝ
  /-- The vertices of the hexagon -/
  vertices : Fin 6 → ℝ × ℝ
  /-- Five sides have length 75 -/
  five_sides_length : ∀ i : Fin 5, dist (vertices i) (vertices (i.succ)) = 75
  /-- The sixth side has length 27 -/
  sixth_side_length : dist (vertices 5) (vertices 0) = 27

/-- The sum of the lengths of the three diagonals from one vertex of the sixth side -/
def diagonalSum (h : InscribedHexagon) : ℝ :=
  dist (h.vertices 0) (h.vertices 2) + 
  dist (h.vertices 0) (h.vertices 3) + 
  dist (h.vertices 0) (h.vertices 4)

/-- Theorem stating that the sum of the diagonal lengths is 365.5 -/
theorem diagonal_sum_is_365_5 (h : InscribedHexagon) : diagonalSum h = 365.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_sum_is_365_5_l575_57534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_count_l575_57530

theorem integer_solutions_count : 
  ∃! (S : Set (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ S ↔ 2*x^2 + x*y + y^2 - x + 2*y + 1 = 0) ∧ 
    Finite S ∧ 
    Nat.card S = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_count_l575_57530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_coefficients_increasing_function_condition_l575_57597

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x

-- Theorem for part (I)
theorem tangent_line_coefficients (a b : ℝ) :
  (∀ x, deriv (f a) 2 * (x - 2) + f a 2 = x + b) →
  a = -2 ∧ b = -2 * Real.log 2 := by sorry

-- Theorem for part (II)
theorem increasing_function_condition (a : ℝ) :
  (∀ x > 1, Monotone (f a)) →
  a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_coefficients_increasing_function_condition_l575_57597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strongest_goldbach_difference_144_l575_57519

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def goldbach_representation (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≠ q ∧ p + q = n

theorem strongest_goldbach_difference_144 :
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≠ q ∧ p + q = 144 ∧
  ∀ p' q' : ℕ, is_prime p' ∧ is_prime q' ∧ p' ≠ q' ∧ p' + q' = 144 →
  (q - p : ℤ).natAbs ≥ (q' - p' : ℤ).natAbs ∧ (q - p : ℤ).natAbs = 134 :=
sorry

#check strongest_goldbach_difference_144

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strongest_goldbach_difference_144_l575_57519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_P_intersect_Q_l575_57527

def P : Set ℝ := {x | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem complement_P_intersect_Q :
  ∀ x : ℝ, x ∈ (Set.univ \ P) ∩ Q ↔ 1 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_P_intersect_Q_l575_57527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_angle_bisector_lengths_l575_57596

/-- Given a triangle ABC with sides a, b, c, calculate the lengths of external angle bisector segments -/
theorem external_angle_bisector_lengths 
  (a b c : ℝ) 
  (h_a : a = 5 - Real.sqrt 7) 
  (h_b : b = 6) 
  (h_c : c = 5 + Real.sqrt 7) :
  let s := (a + b + c) / 2
  let fa := Real.sqrt ((4 * b * c * (s - b) * (s - c)) / ((c - b) ^ 2))
  let fb := Real.sqrt ((4 * c * a * (s - c) * (s - a)) / ((a - c) ^ 2))
  let fc := Real.sqrt ((4 * a * b * (s - a) * (s - b)) / ((b - a) ^ 2))
  fa = 4 * Real.sqrt 3 ∧ fb = 6 / Real.sqrt 7 ∧ fc = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_angle_bisector_lengths_l575_57596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_problem_l575_57533

/-- The probability of choosing a cultural landscape --/
noncomputable def p : ℝ := 2/3

/-- The probability of choosing a natural landscape --/
noncomputable def q : ℝ := 1/3

/-- The number of people in the sample --/
def n : ℕ := 5

/-- The random variable X representing the number of people choosing a cultural landscape --/
noncomputable def X : Nat → ℝ := sorry

/-- The sequence Pn representing the probability of cumulative score being n points --/
noncomputable def P : ℕ → ℝ
| 0 => 0  -- Adding a case for 0 to ensure all cases are covered
| 1 => 1/3
| 2 => 7/9
| (n + 3) => (1/3) * P (n + 2) + (2/3) * P (n + 1)

theorem tourist_problem :
  (Finset.sum (Finset.range (n + 1)) (fun k => k * Nat.choose n k * p^k * (1 - p)^(n - k)) = 10/3) ∧
  (Finset.sum (Finset.range (n + 1)) (fun k => k^2 * Nat.choose n k * p^k * (1 - p)^(n - k)) - (10/3)^2 = 10/9) ∧
  (∀ m : ℕ, P m = 3/5 - (4/15) * (-2/3)^(m - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_problem_l575_57533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetration_14_converges_aerith_is_correct_l575_57535

/-- Tetration function for base 1.4 -/
noncomputable def tetration (n : ℕ) : ℝ :=
  match n with
  | 0 => 1
  | n + 1 => (1.4 : ℝ) ^ (tetration n)

/-- The statement that tetration of 1.4 converges -/
theorem tetration_14_converges :
  ∃ (L : ℝ), L > 0 ∧ ∀ (n : ℕ), tetration n ≤ L := by
  sorry

/-- Aerith is correct: (1.4)^(1.4)^(1.4) is well-defined -/
theorem aerith_is_correct : ∃ (x : ℝ), x = (1.4 : ℝ) ^ ((1.4 : ℝ) ^ (1.4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetration_14_converges_aerith_is_correct_l575_57535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l575_57542

theorem divisibility_condition (m n : ℕ) (h : n > 1) :
  (2^n - 1) ∣ (2^m + 1) ↔ (n = 2 ∧ m % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l575_57542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l575_57526

noncomputable def f (x : Real) : Real := Real.sqrt 2 * Real.cos x * Real.sin (x + Real.pi/4)

theorem f_properties :
  ∃ (T : Real),
    (∀ (x : Real), f (x + T) = f x) ∧
    (∀ (T' : Real), T' > 0 → (∀ (x : Real), f (x + T') = f x) → T' ≥ T) ∧
    T = Real.pi ∧
    (∀ (x : Real), f x ≤ (Real.sqrt 2 + 1) / 2) ∧
    (∃ (x : Real), f x = (Real.sqrt 2 + 1) / 2) ∧
    (∀ (x y : Real), 0 ≤ x ∧ x < y ∧ y ≤ Real.pi/8 → f x < f y) ∧
    (∀ (x y : Real), Real.pi/8 ≤ x ∧ x < y ∧ y ≤ 5*Real.pi/8 → f x > f y) ∧
    (∀ (x y : Real), 5*Real.pi/8 ≤ x ∧ x < y ∧ y ≤ Real.pi → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l575_57526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_side_range_l575_57532

theorem acute_triangle_side_range :
  ∀ a : ℝ,
  (∃ (A B C : ℝ × ℝ),
    let d := (λ (p q : ℝ × ℝ) ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
    d A B = 1 ∧
    d B C = 2 ∧
    d C A = a ∧
    (d A B)^2 + (d B C)^2 > (d C A)^2 ∧
    (d B C)^2 + (d C A)^2 > (d A B)^2 ∧
    (d C A)^2 + (d A B)^2 > (d B C)^2) ↔
  (Real.sqrt 3 < a ∧ a < Real.sqrt 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_side_range_l575_57532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_correct_sum_of_coefficients_correct_l575_57564

/-- An isosceles trapezoid with given side lengths and inscribed circles --/
structure IsoscelesTrapezoidWithCircles where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  DA : ℝ
  CD : ℝ
  -- Radius of circles centered at A and B
  rAB : ℝ
  -- Radius of circles centered at C and D
  rCD : ℝ
  -- Conditions
  isIsosceles : BC = DA
  validSides : AB > CD

/-- The radius of the circle inside the trapezoid that is tangent to all four circles --/
noncomputable def innerCircleRadius (t : IsoscelesTrapezoidWithCircles) : ℝ :=
  (-84 + 72 * Real.sqrt 5) / 29

/-- Theorem stating that the radius of the inner circle is correct --/
theorem inner_circle_radius_correct (t : IsoscelesTrapezoidWithCircles) 
    (h1 : t.AB = 8) (h2 : t.BC = 7) (h3 : t.CD = 6) (h4 : t.rAB = 4) (h5 : t.rCD = 3) :
    innerCircleRadius t = (-84 + 72 * Real.sqrt 5) / 29 := by
  sorry

/-- The sum of k, m, n, and p in the expression (−k + m√n) / p --/
def sumOfCoefficients : ℕ := 84 + 72 + 5 + 29

/-- Theorem stating that the sum of coefficients is correct --/
theorem sum_of_coefficients_correct :
    sumOfCoefficients = 190 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_correct_sum_of_coefficients_correct_l575_57564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l575_57572

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions of the problem
def problem_conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B * Real.cos t.C + t.b * Real.cos t.A * Real.cos t.C = t.c / 2 ∧
  t.c = Real.sqrt 7 ∧
  t.a + t.b = 5

-- State the theorem
theorem triangle_problem (t : Triangle) (h : problem_conditions t) :
  t.C = π / 3 ∧ 
  (1/2 : Real) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l575_57572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l575_57518

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - 2*x

-- State the theorem
theorem f_upper_bound (a : ℝ) (b : ℝ) :
  a ∈ Set.Icc (-1) 0 →
  (∀ x ∈ Set.Ioo 0 1, f a x < b) →
  b > -3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l575_57518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_f_range_l575_57577

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3/4)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

noncomputable def f (x : ℝ) : ℝ := 2 * ((a x).1 + (b x).1) * (b x).1 + 2 * ((a x).2 + (b x).2) * (b x).2

theorem parallel_vectors_tan (x : ℝ) :
  (∃ k : ℝ, a x = k • b x) → Real.tan (x - π/4) = -7 :=
by sorry

theorem f_range :
  Set.Icc (0 : ℝ) (π/2) ⊆ f ⁻¹' Set.Icc (-1/2) (Real.sqrt 2 - 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_f_range_l575_57577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_range_l575_57515

-- Define the function f and its derivative g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_is_derivative : ∀ x, HasDerivAt f (g x) x
axiom f_1_eq_0 : f 1 = 0
axiom xg_minus_f_neg : ∀ x, x > 0 → x * g x - f x < 0

-- Define the set of x that satisfies f(x) < 0
def S : Set ℝ := {x | f x < 0}

-- State the theorem
theorem f_neg_range : S = Set.Ioo (-1) 0 ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_range_l575_57515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_condition_l575_57579

-- Define the points
variable (O A₁ A₂ A₃ A₄ B₁ B₂ B₃ B₄ N M : EuclideanSpace ℝ (Fin 2))

-- Define the lines
def line (P Q : EuclideanSpace ℝ (Fin 2)) := {X : EuclideanSpace ℝ (Fin 2) | ∃ t : ℝ, X = (1 - t) • P + t • Q}

-- Define the conditions
variable (h₁ : N ∈ line A₁ B₁ ∩ line A₂ B₂)
variable (h₂ : M ∈ line A₃ B₃ ∩ line A₄ B₄)

-- Define collinearity
def collinear (P Q R : EuclideanSpace ℝ (Fin 2)) := 
  ∃ t : ℝ, Q = (1 - t) • P + t • R ∨ R = (1 - t) • P + t • Q

-- Use the built-in distance function
open EuclideanSpace

-- State the theorem
theorem collinearity_condition :
  collinear O N M ↔ 
    (dist O B₁ / dist O B₃) * (dist O B₂ / dist O B₄) * (dist B₃ B₄ / dist B₁ B₂) = 
    (dist O A₁ / dist O A₃) * (dist O A₂ / dist O A₄) * (dist A₃ A₄ / dist A₁ A₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_condition_l575_57579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l575_57550

-- Define the two points
def point1 : ℝ × ℝ := (3, 5)
def point2 : ℝ × ℝ := (0, 1)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem distance_between_points :
  distance point1 point2 = 5 := by
  -- Expand the definition of distance
  unfold distance
  -- Simplify the expression under the square root
  simp [point1, point2]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l575_57550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_equals_ln_2_plus_sqrt_3_l575_57575

-- Define the curve
noncomputable def curve (y : ℝ) : ℝ := Real.log (Real.cos y)

-- Define the arc length function
noncomputable def arcLength (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ y in a..b, Real.sqrt (1 + ((deriv f) y) ^ 2)

-- Theorem statement
theorem curve_length_equals_ln_2_plus_sqrt_3 :
  arcLength curve 0 (π/3) = Real.log (2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_equals_ln_2_plus_sqrt_3_l575_57575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locally_odd_f_locally_odd_g_range_locally_odd_h_range_l575_57562

-- Define the functions
def f (x : ℝ) : ℝ := x^4 + x^3 + x^2 + x - 1

noncomputable def g (m x : ℝ) : ℝ := (4 : ℝ)^x - m * 2^(x+1) + m^2 - 3

noncomputable def h (a b x : ℝ) : ℝ := Real.log (x + 1 + a) + x^2 + x - b

-- Statement 1
theorem locally_odd_f : ∃ x ∈ Set.Icc (-1 : ℝ) 1, f (-x) = -f x := by sorry

-- Statement 2
theorem locally_odd_g_range : 
  Set.Icc (1 - Real.sqrt 3) (2 * Real.sqrt 2) = 
  {m : ℝ | ∃ x : ℝ, g m (-x) = -g m x} := by sorry

-- Statement 3
theorem locally_odd_h_range : 
  Set.Ioo (1 : ℝ) (Real.exp 1 - 1) = 
  {a : ℝ | a > 1 ∧ ∀ b ∈ Set.Icc 1 (3/2), ∃ x ∈ Set.Icc (-1 : ℝ) 1, h a b (-x) = -h a b x} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locally_odd_f_locally_odd_g_range_locally_odd_h_range_l575_57562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_range_a_l575_57565

/-- The function f(x) = xe^x + x^2 + 2x + a -/
noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x + x^2 + 2*x + a

/-- The function g(x) = xe^x + x^2 + 2x -/
noncomputable def g (x : ℝ) : ℝ := x * Real.exp x + x^2 + 2*x

theorem two_zeros_range_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f x a = 0 ∧ f y a = 0) ∧
  (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z → ¬(f x a = 0 ∧ f y a = 0 ∧ f z a = 0)) →
  a < 1 + 1 / Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_range_a_l575_57565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l575_57587

/-- Given an obtuse triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) ∧
  c^2 = a^2 + b^2 - 2 * a * b * (Real.cos C)

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_obtuse : Real.pi/2 < C)
  (h_a : a = 7)
  (h_b : b = 3)
  (h_cos_C : Real.cos C = 11/14) :
  c = 5 ∧ A = 2*Real.pi/3 ∧ Real.sin (2*C - Real.pi/6) = 71/98 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l575_57587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_143_l575_57583

theorem sum_of_divisors_143 : (Finset.sum (Nat.divisors 143) id) = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_143_l575_57583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differentiation_check_l575_57509

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x + 3 / x
noncomputable def g (x : ℝ) : ℝ := (x + 3)^3
noncomputable def h (x : ℝ) : ℝ := 3^x
noncomputable def j (x : ℝ) : ℝ := x^2 * Real.cos x

-- State the theorem
theorem differentiation_check :
  (∃ x, deriv f x ≠ 1 + 3 / x^2) ∧
  (∀ x, deriv g x = 3 * (x + 3)^2) ∧
  (∃ x, deriv h x ≠ 3 * Real.log x) ∧
  (∃ x, deriv j x ≠ -2 * x * Real.sin x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_differentiation_check_l575_57509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l575_57570

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  sum_of_arithmetic_sequence a 10 = 60 →
  a 7 = 7 →
  a 4 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l575_57570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l575_57599

/-- Calculates the length of a train given the speeds of two trains and the time taken for the faster train to cross a man in the slower train. -/
noncomputable def train_length (faster_speed slower_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  let relative_speed := faster_speed - slower_speed
  let relative_speed_mps := relative_speed * (1000 / 3600)
  relative_speed_mps * crossing_time

/-- Theorem stating that given the conditions of the problem, the length of the faster train is 135 meters. -/
theorem faster_train_length :
  let faster_speed := (54 : ℝ)
  let slower_speed := (36 : ℝ)
  let crossing_time := (27 : ℝ)
  train_length faster_speed slower_speed crossing_time = 135 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l575_57599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_selling_price_l575_57549

/-- Calculates the selling price of an item given its cost price and loss percentage. -/
noncomputable def sellingPrice (costPrice : ℝ) (lossPercentage : ℝ) : ℝ :=
  costPrice * (1 - lossPercentage / 100)

/-- Theorem stating that for a radio with a cost price of 4500 and a loss percentage of 28.888888888888886%,
    the selling price is 3200. -/
theorem radio_selling_price :
  sellingPrice 4500 28.888888888888886 = 3200 := by
  -- Unfold the definition of sellingPrice
  unfold sellingPrice
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_selling_price_l575_57549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_single_point_implies_d_eq_12_l575_57558

/-- A conic section represented by the equation 3x^2 + y^2 + 6x - 6y + d = 0 -/
def conic (d : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0

/-- The conic degenerates to a single point -/
def is_single_point (d : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, conic d p.1 p.2

theorem conic_single_point_implies_d_eq_12 :
  ∀ d : ℝ, is_single_point d → d = 12 := by
  sorry

#check conic_single_point_implies_d_eq_12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_single_point_implies_d_eq_12_l575_57558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_speed_calculation_l575_57514

/-- Calculates the harmonic mean of two speeds -/
noncomputable def harmonicMean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

theorem swimming_speed_calculation (swimmingSpeed runningSpeed averageSpeed : ℝ) 
  (h1 : runningSpeed = 9)
  (h2 : averageSpeed = 5)
  (h3 : averageSpeed = harmonicMean swimmingSpeed runningSpeed) :
  swimmingSpeed = 45 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_speed_calculation_l575_57514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_approx_l575_57551

/-- The speed of the man given the train's parameters -/
noncomputable def mans_speed (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let relative_speed := train_length / passing_time
  let mans_speed_mps := relative_speed - train_speed_mps
  mans_speed_mps * 3600 / 1000

/-- Theorem stating the man's speed given the train's parameters -/
theorem mans_speed_approx (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ)
    (h1 : train_length = 330)
    (h2 : train_speed_kmph = 60)
    (h3 : passing_time = 18) :
    ∃ ε > 0, |mans_speed train_length train_speed_kmph passing_time - 5.99| < ε := by
  sorry

/-- Compute an approximation of the man's speed -/
def mans_speed_approx_compute : ℚ :=
  (330 : ℚ) / 18 - 60 * 1000 / 3600

#eval mans_speed_approx_compute

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_approx_l575_57551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l575_57556

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l575_57556
