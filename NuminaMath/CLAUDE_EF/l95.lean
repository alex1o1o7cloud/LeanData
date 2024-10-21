import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l95_9562

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) (D : Real) :
  (2 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C →
  t.A = Real.pi / 3 ∧
  ∃ (area : Real), 
    (2 * D = t.c) → 
    (Real.sqrt ((2 * D)^2 + 4) = 2) → 
    area ≤ (3 * Real.sqrt 3) / 2 ∧
    ∃ (t' : Triangle), t'.A = t.A ∧ area = (t'.b * t'.c * Real.sin t'.A) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l95_9562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l95_9552

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the vertices and focus
def A (a : ℝ) : ℝ × ℝ := (-a, 0)
def B (a : ℝ) : ℝ × ℝ := (a, 0)
def D (b : ℝ) : ℝ × ℝ := (0, b)
def F (c : ℝ) : ℝ × ℝ := (-c, 0)

-- Define the point P on the ellipse
def P (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse a b p.1 p.2}

-- Define the line l (AD)
def line_l (a b : ℝ) (x y : ℝ) : Prop :=
  y/b - x/a = 1

-- Define the point M
noncomputable def M (a b c : ℝ) : ℝ × ℝ := (-c, b*(a-c)/a)

-- Define the point N
noncomputable def N (b : ℝ) : ℝ × ℝ := (0, 2*b/3)

-- State the theorem
theorem ellipse_eccentricity
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : ∃ p ∈ P a b, p.2 = 0 → p.1 = -c)
  (h4 : line_l a b (M a b c).1 (M a b c).2)
  (h5 : ∃ t, (1-t) • (B a) + t • (M a b c) = N b) :
  c/a = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l95_9552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_mean_theorem_l95_9502

/-- Represents a normal distribution with mean μ and standard deviation σ -/
structure NormalDistribution (μ σ : ℝ) where
  pdf : ℝ → ℝ
  is_pdf : ∫ x, pdf x = 1

/-- The cumulative distribution function (CDF) of a normal distribution -/
noncomputable def cdf (nd : NormalDistribution μ σ) (x : ℝ) : ℝ :=
  ∫ y in Set.Iic x, nd.pdf y

theorem normal_distribution_mean_theorem 
  (μ σ : ℝ) (nd : NormalDistribution μ σ) :
  cdf nd 14 = 0.1 → cdf nd 18 = 0.9 → μ = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_mean_theorem_l95_9502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l95_9567

noncomputable def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 2

noncomputable def mean_of_nonzero_coefficients : ℝ :=
  let coeffs := [3, -5, 2]
  (coeffs.sum) / (coeffs.length : ℝ)

def Q (x : ℝ) : ℝ := 0

theorem intersection_point :
  ∃ x : ℝ, x = 1 ∧ P x = Q x := by
  use 1
  constructor
  · rfl
  · sorry  -- The actual computation of P(1) = Q(1) = 0 would go here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l95_9567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcyclist_distance_theorem_l95_9501

/-- Represents the distance traveled by a motorcyclist in a moving column of cars. -/
noncomputable def motorcyclist_distance (L : ℝ) (y : ℝ) : ℝ :=
  L * (1 + Real.sqrt 2)

/-- Theorem stating the distance traveled by the motorcyclist. -/
theorem motorcyclist_distance_theorem (L : ℝ) (y : ℝ) (h1 : L > 0) (h2 : y > 0) :
  ∃ (x : ℝ), x > y ∧ 
  ((L / (x - y)) + (L / (x + y))) = (L / y) ∧
  motorcyclist_distance L y = (L / y) * x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcyclist_distance_theorem_l95_9501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_logarithms_and_powers_l95_9523

theorem ordering_of_logarithms_and_powers : 
  let a := Real.log 6 / Real.log (1/3)
  let b := (1/4 : ℝ) ^ (0.8 : ℝ)
  let c := Real.log π
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_logarithms_and_powers_l95_9523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_line_arrangements_l95_9548

/-- Represents the number of people in each group -/
def num_fathers : Nat := 2
def num_mothers : Nat := 2
def num_children : Nat := 2

/-- The total number of people -/
def total_people : Nat := num_fathers + num_mothers + num_children

/-- Represents the constraint that fathers must be at the beginning and end -/
def fathers_at_ends : Nat := 2

/-- Represents the constraint that children must be together -/
def children_together : Nat := 1

/-- The number of ways to arrange the people under the given constraints -/
def arrangement_count : Nat := 24

theorem zoo_line_arrangements :
  fathers_at_ends * children_together * Nat.factorial (total_people - num_fathers - children_together + 1) = arrangement_count :=
by
  -- Proof goes here
  sorry

#eval fathers_at_ends * children_together * Nat.factorial (total_people - num_fathers - children_together + 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_line_arrangements_l95_9548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_numbers_between_29_and_36_l95_9514

theorem two_digit_numbers_between_29_and_36 : 
  Finset.card (Finset.filter (fun n => 29 < n ∧ n < 36) (Finset.range 100)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_numbers_between_29_and_36_l95_9514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_45_plus_tan_30_l95_9559

theorem cot_45_plus_tan_30 : 
  Real.tan (π/4)⁻¹ + Real.tan (π/6) = (Real.sqrt 18 + Real.sqrt 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_45_plus_tan_30_l95_9559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_l95_9589

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (x + φ)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f φ (2 * x - Real.pi / 3)

theorem symmetric_sine (φ : ℝ) :
  (∀ x : ℝ, g φ x = g φ (-x)) → φ = Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_l95_9589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l95_9512

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp 0 * x + (1/2) * x^2

-- State the theorem
theorem f_derivative_at_one : 
  (deriv f) 1 = Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l95_9512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joanna_money_is_8_l95_9525

/-- Represents the amount of money Joanna has -/
def joanna_money : ℝ := sorry

/-- Represents the amount of money Joanna's brother has -/
noncomputable def brother_money : ℝ := 3 * joanna_money

/-- Represents the amount of money Joanna's sister has -/
noncomputable def sister_money : ℝ := joanna_money / 2

/-- The total amount of money the three of them have -/
def total_money : ℝ := 36

theorem joanna_money_is_8 :
  joanna_money + brother_money + sister_money = total_money →
  joanna_money = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joanna_money_is_8_l95_9525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l95_9538

/-- The function f(x) = x ln x - ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x

/-- The function g(x) = x^3 - x + 6 -/
def g (x : ℝ) : ℝ := x^3 - x + 6

/-- The derivative of g(x) -/
def g' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, 2 * f a x ≤ g' x + 2) ↔ a ∈ Set.Ici (-2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l95_9538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_existence_l95_9535

/-- A polynomial with real coefficients -/
def RealPolynomial (n : ℕ) := ℂ → ℂ

/-- The theorem statement -/
theorem polynomial_root_existence (n : ℕ) (P : RealPolynomial n) :
  (Complex.abs (P Complex.I) < 1) →
  ∃ (a b : ℝ), P (Complex.ofReal a + Complex.I * Complex.ofReal b) = 0 ∧ (a^2 + b^2 + 1)^2 < 4*b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_existence_l95_9535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_base_angle_l95_9599

/-- Regular triangular pyramid -/
structure RegularTriangularPyramid where
  base_side : ℝ
  lateral_edge : ℝ

/-- Angle between a line and a plane -/
noncomputable def angle_line_plane (l : ℝ) (p : ℝ) : ℝ := sorry

/-- Main theorem -/
theorem lateral_edge_base_angle (pyramid : RegularTriangularPyramid) 
  (h : angle_line_plane pyramid.lateral_edge pyramid.lateral_edge = Real.arcsin (Real.sqrt 2 / 3)) :
  angle_line_plane pyramid.lateral_edge pyramid.base_side = Real.arcsin (2 * Real.sqrt 2 / 3) ∨
  angle_line_plane pyramid.lateral_edge pyramid.base_side = Real.arcsin (1 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_base_angle_l95_9599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_vertex_sum_l95_9537

noncomputable section

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (3, 5)
def D : ℝ × ℝ := (11, -3)

-- Define the property that A and D are diagonally opposite
def diagonally_opposite (A D : ℝ × ℝ) : Prop :=
  ∃ (M : ℝ × ℝ), M = ((A.1 + D.1) / 2, (A.2 + D.2) / 2)

-- Define the area of the parallelogram
def parallelogram_area (A B D : ℝ × ℝ) : ℝ :=
  (1/2) * abs (A.1 * (B.2 - D.2) + B.1 * (D.2 - A.2) + D.1 * (A.2 - B.2))

-- Theorem statement
theorem parallelogram_vertex_sum :
  ∀ C : ℝ × ℝ,
  diagonally_opposite A D →
  parallelogram_area A B D = 48 →
  ∃ (x y : ℝ), C = (x, y) ∧ x + y = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_vertex_sum_l95_9537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_area_calculation_l95_9519

/-- The diameter of the circular flowerbed in feet -/
def flowerbed_diameter : ℝ := 16

/-- The width of the walking path in feet -/
def path_width : ℝ := 4

/-- The distance from the center of the flowerbed to the nearest edge of the path in feet -/
def center_to_path : ℝ := 2

/-- The remaining area covered by flowers in square feet -/
noncomputable def remaining_flower_area : ℝ := 36 * Real.pi

theorem flower_area_calculation :
  let flowerbed_radius := flowerbed_diameter / 2
  let total_area := Real.pi * flowerbed_radius ^ 2
  let inner_circle_radius := flowerbed_radius - center_to_path
  let inner_circle_area := Real.pi * inner_circle_radius ^ 2
  let path_area := total_area - inner_circle_area
  total_area - path_area = remaining_flower_area := by
  -- Proof steps would go here
  sorry

#check flower_area_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_area_calculation_l95_9519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_implies_a_value_l95_9570

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x - 1 else 2

-- Define the theorem
theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ici 3 ↔ x * f (x - 1) ≥ a) →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_implies_a_value_l95_9570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l95_9516

theorem largest_lambda : 
  ∃ (lambda_max : ℝ), lambda_max = 2 ∧ 
  (∀ (lambda : ℝ), (∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 + a*b^2 ≥ a*b + lambda*b*c + c*d) → lambda ≤ lambda_max) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l95_9516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_l95_9503

theorem angle_difference (α β : Real) : 
  0 < α ∧ α < Real.pi/2 →
  0 < β ∧ β < Real.pi/2 →
  Real.cos α = 2 * Real.sqrt 5 / 5 →
  Real.cos β = Real.sqrt 10 / 10 →
  α - β = -Real.pi/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_l95_9503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_angle_l95_9530

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 3*x

-- Define the line passing through (3, 0)
def line (t : ℝ) (x y : ℝ) : Prop := x = t*y + 3

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) (t : ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  line t A.1 A.2 ∧ line t B.1 B.2

-- Define the angle between two vectors
def angle_between (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem parabola_line_intersection_angle 
  (t : ℝ) (A B : ℝ × ℝ) (h : intersection_points A B t) : 
  angle_between A B = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_angle_l95_9530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_integral_equality_integral_bound_l95_9583

noncomputable section

variable (f : ℝ → ℝ)
variable (h_cont : ContinuousOn f (Set.Icc 0 1))
variable (h_int : ∀ (k : ℕ), k < n → ∫ x in Set.Icc 0 1, x^k * f x = 0)
variable (h_n : n ≥ 1)
variable (n : ℕ)

/-- The function g as defined in the problem -/
def g (t : ℝ) : ℝ := ∫ x in Set.Icc 0 1, |x - t|^n

/-- The maximum absolute value of f on [0, 1] -/
def M : ℝ := ⨆ x ∈ Set.Icc 0 1, |f x|

/-- The minimum value of g and where it occurs -/
theorem min_value_g :
  ∃ t₀ : ℝ, ∀ t : ℝ, g n t₀ ≤ g n t ∧ g n t₀ = 1 / ((n + 1) * 2^(n+1)) :=
sorry

/-- The equality of integrals as stated in the problem -/
theorem integral_equality (t : ℝ) :
  ∫ x in Set.Icc 0 1, (x - t)^n * f x = ∫ x in Set.Icc 0 1, x^n * f x :=
sorry

/-- The bound on the integral as stated in the problem -/
theorem integral_bound :
  |∫ x in Set.Icc 0 1, x^n * f x| ≤ M f / (2^n * (n + 1)) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_integral_equality_integral_bound_l95_9583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_origin_l95_9553

noncomputable def point_P : ℝ × ℝ := (-1, 2)
def origin : ℝ × ℝ := (0, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_P_to_origin : distance point_P origin = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_origin_l95_9553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sqrt_twelve_count_l95_9592

theorem ceiling_sqrt_twelve_count :
  ∃ (S : Finset ℤ), (∀ x ∈ S, ⌈Real.sqrt x⌉ = 12) ∧ Finset.card S = 23 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sqrt_twelve_count_l95_9592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_divisibility_a_triple_divisibility_l95_9571

noncomputable def a (n : ℕ) : ℚ :=
  15 * n + 2 + (15 * n - 32/100) * 16^(n-1)

theorem a_divisibility (n : ℕ) : 
  ∃ k : ℤ, (a n) = (15^3 : ℚ) * k :=
sorry

theorem a_triple_divisibility : 
  {n : ℕ | (∃ k₁ k₂ k₃ : ℤ, (a n = 1991 * k₁) ∧ 
                            (a (n+1) = 1991 * k₂) ∧ 
                            (a (n+2) = 1991 * k₃))} 
  = {n : ℕ | ∃ k : ℕ, n = 89595 * k} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_divisibility_a_triple_divisibility_l95_9571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l95_9568

/-- Converts meters to miles -/
noncomputable def meters_to_miles (m : ℝ) : ℝ := m * 0.000621371

/-- Converts minutes to hours -/
noncomputable def minutes_to_hours (m : ℝ) : ℝ := m / 60

/-- Calculates speed in miles per hour given distance in meters and time in minutes -/
noncomputable def speed_mph (distance_m : ℝ) (time_min : ℝ) : ℝ :=
  (meters_to_miles distance_m) / (minutes_to_hours time_min)

theorem speed_calculation :
  let distance_m : ℝ := 900
  let time_min : ℝ := 3 + 20 / 60
  abs (speed_mph distance_m time_min - 10.07) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l95_9568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_satisfies_equation_l95_9556

-- Define the function f as noncomputable due to the use of Real.pow
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (2 : ℝ)^(x-1) else x + 3

-- State the theorem
theorem unique_a_satisfies_equation :
  ∃! a : ℝ, f a + f 1 = 0 ∧ a = -4 := by
  -- The proof is omitted and replaced with sorry
  sorry

-- You can add additional lemmas or theorems if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_satisfies_equation_l95_9556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sequences_l95_9576

/-- Represents the number of foreign guests -/
def num_guests : ℕ := 4

/-- Represents the number of security personnel -/
def num_security : ℕ := 2

/-- Represents the total number of people entering -/
def total_people : ℕ := num_guests + num_security

/-- Represents that two specific guests (A and B) must be together -/
def guests_together : Prop := true

/-- Represents that security personnel must be at the beginning and end -/
def security_at_ends : Prop := true

/-- Function to calculate the number of sequences -/
def number_of_sequences (ng : ℕ) (ns : ℕ) (gt : Prop) (se : Prop) : ℕ :=
  sorry -- Implementation details would go here

/-- The theorem stating the total number of possible sequences -/
theorem total_sequences : 
  num_guests = 4 → 
  num_security = 2 → 
  total_people = num_guests + num_security → 
  guests_together → 
  security_at_ends → 
  (∃ n : ℕ, n = 24 ∧ n = number_of_sequences num_guests num_security guests_together security_at_ends) :=
by
  sorry -- Proof details would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sequences_l95_9576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_adjoining_squares_l95_9581

/-- The area of the shaded region formed by two adjoining squares -/
theorem shaded_area_of_adjoining_squares (small_side large_side : ℝ) : 
  small_side = 3 →
  large_side = 10 →
  (small_side^2 - (1 / 2) * ((large_side / (small_side + large_side)) * small_side) * small_side) = 72 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_adjoining_squares_l95_9581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_division_l95_9578

theorem cube_division (edge_length : ℕ) (n : ℕ) : 
  edge_length = 4 →
  ∃ (sizes : List ℕ),
    (sizes.length = n) ∧
    (∀ x, x ∈ sizes → x > 0 ∧ x ≤ edge_length) ∧
    (∃ a b, a ∈ sizes ∧ b ∈ sizes ∧ a ≠ b) ∧
    (sizes.map (λ x => x^3)).sum = edge_length^3 →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_division_l95_9578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_range_l95_9566

theorem sequence_inequality_range (a : ℝ) : 
  (∀ n : ℕ+, (-1 : ℝ)^(n.val + 2013 : ℕ) * a < 2 + (-1 : ℝ)^(n.val + 2014 : ℕ) / (n : ℝ)) ↔ 
  (-2 ≤ a ∧ a < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_range_l95_9566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l95_9513

noncomputable def interval_length (m n : ℝ) : ℝ := n - m

noncomputable def solution_length (a : ℝ) : ℝ := a + 4 / a

theorem quadratic_inequality_solution (a : ℝ) (h : a > 0) :
  -- 1. The length of the solution set interval
  solution_length a = interval_length (-a) (4 / a) ∧
  -- 2. When a = 1, l = 5
  solution_length 1 = 5 ∧
  -- 3. The minimum value of l is 4
  (∀ x > 0, solution_length x ≥ 4) ∧ (∃ y > 0, solution_length y = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l95_9513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l95_9582

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_expression_evaluation :
  (floor 6.5) * (floor (2/3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.3) - (6.6 : ℝ) = 15.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l95_9582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_time_example_l95_9515

/-- The time taken for a train to pass a platform -/
noncomputable def train_pass_time (train_length platform_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: The time taken for a train of length 360 m, moving at a speed of 45 km/hr,
    to pass a platform of length 240 m is 48 seconds. -/
theorem train_pass_time_example :
  train_pass_time 360 240 45 = 48 := by
  -- Unfold the definition of train_pass_time
  unfold train_pass_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_time_example_l95_9515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_16_formula_l95_9554

noncomputable def f (x : ℝ) : ℝ := (1 + x) / (2 - x)

noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => id
  | 1 => f
  | n + 1 => λ x => f (f_n n x)

theorem f_16_formula : f_n 16 = λ x => (x - 1) / x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_16_formula_l95_9554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l95_9560

/-- The distance between two points in a 2D plane. -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem: The distance between points (2, -3) and (9, 6) is √130. -/
theorem distance_between_specific_points :
  distance 2 (-3) 9 6 = Real.sqrt 130 := by
  -- Expand the definition of distance
  unfold distance
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l95_9560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_squared_equals_one_over_288_l95_9549

-- Define g(n) as the sum of 1/k^n for k from 3 to infinity
noncomputable def g (n : ℕ+) : ℝ := ∑' k : ℕ, (1 : ℝ) / ((k + 2 : ℝ) ^ (n : ℝ))

-- Theorem statement
theorem sum_of_g_squared_equals_one_over_288 :
  ∑' n : ℕ+, (g n)^2 = 1 / 288 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_squared_equals_one_over_288_l95_9549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_sine_function_l95_9511

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem symmetry_center_of_sine_function
  (ω φ : ℝ)
  (h_ω_pos : ω > 0)
  (h_φ_bound : |φ| < π / 2)
  (h_period : ∀ x, f ω φ (x + 4 * π) = f ω φ x)
  (h_smallest_period : ∀ T, T > 0 → (∀ x, f ω φ (x + T) = f ω φ x) → T ≥ 4 * π)
  (h_f_value : f ω φ (π / 3) = 1) :
  ∃ k : ℤ, ∀ x, f ω φ (x + (2 * k * π - 2 * π / 3)) = f ω φ (-x + (2 * k * π - 2 * π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_sine_function_l95_9511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_and_tangent_line_l95_9550

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 + 2*y^2 = 4

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the line y = 2
def line_y_2 (y : ℝ) : Prop := y = 2

-- Define perpendicularity
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

theorem ellipse_eccentricity_and_tangent_line :
  -- Eccentricity of ellipse C is √2/2
  (∃ e : ℝ, e = Real.sqrt 2 / 2 ∧
    ∀ x y : ℝ, ellipse_C x y → 
      ∃ a b c : ℝ, a^2 * y^2 + b^2 * x^2 = a^2 * b^2 ∧
                   c^2 = a^2 - b^2 ∧
                   e = c / a) ∧
  -- For any points A on C and B on y = 2 with OA ⊥ OB, AB is tangent to the circle
  (∀ x1 y1 x2 y2 : ℝ,
    ellipse_C x1 y1 →
    line_y_2 y2 →
    perpendicular x1 y1 x2 y2 →
    ∃ t : ℝ, ∀ x y : ℝ,
      (y - y2 = (y1 - y2) / (x1 - x2) * (x - x2)) →
      (x^2 + y^2 ≥ 2 ∨ circle_eq x y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_and_tangent_line_l95_9550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_imply_a_power_b_l95_9597

-- Define the function as noncomputable
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (x - b) / (x + 2)

-- State the theorem
theorem function_properties_imply_a_power_b (a b : ℝ) : 
  b < -2 → 
  (∀ x ∈ Set.Ioo a (b + 4), f b x ∈ Set.Ioi 2) →
  (∀ y > 2, ∃ x ∈ Set.Ioo a (b + 4), f b x = y) →
  a^b = (1/16 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_imply_a_power_b_l95_9597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l95_9517

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (-2^x + b) / (2^x + a)

-- Define what it means for f to be an odd function
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Main theorem
theorem odd_function_properties (a b : ℝ) :
  is_odd_function (f a b) →
  (a = 1 ∧ b = 1) ∧
  (∀ k, (∀ t, f 1 1 (t^2 - 2*t) + f 1 1 (2*t^2 - k) < 0) → k < -1/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l95_9517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_not_always_similar_l95_9506

/-- Two shapes are similar if they have proportional corresponding sides and equal corresponding angles. -/
def are_similar {α : Type} (shape1 shape2 : α) : Prop := sorry

/-- An equilateral triangle has all sides equal and all angles equal to 60°. -/
structure EquilateralTriangle : Type where
  dummy : Unit

/-- An isosceles right triangle has one right angle and two equal acute angles of 45°. -/
structure IsoscelesRightTriangle : Type where
  dummy : Unit

/-- A rectangle has four right angles. -/
structure Rectangle : Type where
  width : ℝ
  height : ℝ

/-- A square has all sides equal and all angles equal to 90°. -/
structure Square : Type where
  side : ℝ

theorem rectangle_not_always_similar :
  (∀ (t1 t2 : EquilateralTriangle), are_similar t1 t2) ∧
  (∀ (t1 t2 : IsoscelesRightTriangle), are_similar t1 t2) ∧
  (∃ (r1 r2 : Rectangle), ¬ are_similar r1 r2) ∧
  (∀ (s1 s2 : Square), are_similar s1 s2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_not_always_similar_l95_9506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_distances_l95_9588

/-- A circle with radius r, where r is an odd number -/
structure Circle where
  r : ℕ
  r_odd : Odd r

/-- A point on the circumference of the circle -/
structure CirclePoint (c : Circle) where
  u : ℕ
  v : ℕ
  p : ℕ
  q : ℕ
  m : ℕ
  n : ℕ
  h_on_circle : u^2 + v^2 = c.r^2
  h_u : u = p^m
  h_v : v = q^n
  h_p_prime : Nat.Prime p
  h_q_prime : Nat.Prime q
  h_u_gt_v : u > v

/-- The theorem to be proved -/
theorem circle_intersection_distances (c : Circle) (P : CirclePoint c) :
  let M := P.u
  let N := P.v
  (c.r - M = 1) ∧
  (c.r + M = 9) ∧
  (c.r + N = 8) ∧
  (c.r - N = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_distances_l95_9588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_max_t_value_exists_max_t_l95_9557

/-- The function f(x) = ln(x) / (x + 1) -/
noncomputable def f (x : ℝ) : ℝ := Real.log x / (x + 1)

/-- The domain of f is x > 0 -/
def f_domain (x : ℝ) : Prop := x > 0

theorem tangent_line_at_one (x y : ℝ) :
  f_domain 1 → x - 2*y - 1 = 0 ↔ y = (f 1) + (x - 1) * ((deriv f) 1) := by
  sorry

theorem max_t_value (t : ℝ) :
  (∀ x, f_domain x → x ≠ 1 → f x - t/x > Real.log x / (x - 1)) →
  t ≤ -1 := by
  sorry

theorem exists_max_t :
  ∃ t, t = -1 ∧
    (∀ x, f_domain x → x ≠ 1 → f x - t/x > Real.log x / (x - 1)) ∧
    (∀ t', t' < t →
      ∃ x, f_domain x ∧ x ≠ 1 ∧ f x - t'/x ≤ Real.log x / (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_max_t_value_exists_max_t_l95_9557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_seven_implies_a_plus_b_l95_9526

/-- Recursive function definition for f_n -/
def f_n (a b : ℝ) : ℕ → (ℝ → ℝ)
  | 0 => λ x => x  -- Base case for n = 0
  | 1 => λ x => a * x - b
  | n + 1 => λ x => f_n a b n (a * x - b)

/-- Theorem statement -/
theorem f_seven_implies_a_plus_b (a b : ℝ) :
  (∀ x, f_n a b 7 x = 128 * x + 381) → a + b = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_seven_implies_a_plus_b_l95_9526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l95_9522

noncomputable section

-- Define the slopes of the two lines
def slope1 (a : ℝ) : ℝ := -a / 2
def slope2 : ℝ := -1

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop := slope1 a * slope2 = -1

-- Theorem statement
theorem perpendicular_lines_a_value :
  ∀ a : ℝ, perpendicular a → a = -2 :=
by
  intro a h
  unfold perpendicular at h
  unfold slope1 at h
  unfold slope2 at h
  field_simp at h
  linarith

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l95_9522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_emeralds_is_twelve_l95_9575

/-- Represents a box of gemstones -/
structure GemBox where
  count : Nat
  type : String

/-- The jeweler's collection of gem boxes -/
def jewelerBoxes : List GemBox := sorry

/-- Theorem: Given the conditions, the total number of emeralds is 12 -/
theorem total_emeralds_is_twelve :
  -- There are 6 boxes in total
  jewelerBoxes.length = 6 →
  -- 2 boxes contain diamonds, 2 contain emeralds, and 2 contain rubies
  (jewelerBoxes.filter (λ b => b.type = "diamond")).length = 2 →
  (jewelerBoxes.filter (λ b => b.type = "emerald")).length = 2 →
  (jewelerBoxes.filter (λ b => b.type = "ruby")).length = 2 →
  -- The total number of rubies is 15 more than the total number of diamonds
  (jewelerBoxes.filter (λ b => b.type = "ruby")).foldr (λ b acc => b.count + acc) 0 =
    (jewelerBoxes.filter (λ b => b.type = "diamond")).foldr (λ b acc => b.count + acc) 0 + 15 →
  -- The total number of emeralds is 12
  (jewelerBoxes.filter (λ b => b.type = "emerald")).foldr (λ b acc => b.count + acc) 0 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_emeralds_is_twelve_l95_9575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_proof_l95_9586

/-- Calculates the initial amount of money given the final amounts and interest rates -/
def calculate_initial_amount (final_amount_1 final_amount_2 : ℚ) 
  (interest_rate_increase : ℚ) (time : ℚ) : ℚ :=
  (final_amount_1 - final_amount_2) / (interest_rate_increase * time / 100)

/-- Proves that the initial amount is 1179.6 given the problem conditions -/
theorem initial_amount_proof : 
  ∃ (initial_amount : ℚ), initial_amount = 1179.6 :=
by
  -- Define the initial amount
  let initial_amount := calculate_initial_amount 670 552.04 2 5
  
  -- Assert that this initial amount exists
  use initial_amount
  
  -- Prove that it equals 1179.6
  -- In a real proof, we would show this step-by-step
  -- For now, we'll use sorry to skip the detailed proof
  sorry

-- Evaluate the result
#eval calculate_initial_amount 670 552.04 2 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_proof_l95_9586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_perimeter_implies_cyclic_not_always_equal_perimeter_point_for_cyclic_l95_9545

/-- Definition of a convex quadrilateral -/
def ConvexQuadrilateral (A B C D : ℝ × ℝ) : Prop :=
sorry

/-- Definition of a cyclic quadrilateral -/
def CyclicQuadrilateral (A B C D : ℝ × ℝ) : Prop :=
sorry

/-- Definition of a triangle -/
def Triangle (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
sorry

/-- Definition of the perimeter of a triangle -/
def Perimeter (T : Set (ℝ × ℝ)) : ℝ :=
sorry

/-- A convex quadrilateral ABCD with a point X such that the perimeters of triangles ABX, BCX, CDX, and DAX are equal is cyclic. -/
theorem equal_perimeter_implies_cyclic 
  (A B C D X : ℝ × ℝ) 
  (convex : ConvexQuadrilateral A B C D) 
  (equal_perimeters : 
    Perimeter (Triangle A B X) = Perimeter (Triangle B C X) ∧
    Perimeter (Triangle B C X) = Perimeter (Triangle C D X) ∧
    Perimeter (Triangle C D X) = Perimeter (Triangle D A X)) :
  CyclicQuadrilateral A B C D :=
by
  sorry

/-- There does not always exist a point X satisfying equal perimeter conditions for a cyclic quadrilateral. -/
theorem not_always_equal_perimeter_point_for_cyclic 
  (A B C D : ℝ × ℝ) 
  (cyclic : CyclicQuadrilateral A B C D) :
  ¬ ∀ X : ℝ × ℝ, 
    Perimeter (Triangle A B X) = Perimeter (Triangle B C X) ∧
    Perimeter (Triangle B C X) = Perimeter (Triangle C D X) ∧
    Perimeter (Triangle C D X) = Perimeter (Triangle D A X) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_perimeter_implies_cyclic_not_always_equal_perimeter_point_for_cyclic_l95_9545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l95_9533

def b : ℕ → ℝ
  | 0 => 3  -- Adding the base case for 0
  | n + 1 => (64 * (b n)^3)^(1/3)

theorem b_50_value : b 50 = 4^49 * 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l95_9533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_EOF_l95_9587

-- Define the line L
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - 2 * p.2 - 3 = 0}

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 3)^2 = 9}

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the intersection points E and F
noncomputable def E : ℝ × ℝ := sorry
noncomputable def F : ℝ × ℝ := sorry

-- Assumptions
axiom E_in_L : E ∈ L
axiom E_in_C : E ∈ C
axiom F_in_L : F ∈ L
axiom F_in_C : F ∈ C
axiom E_ne_F : E ≠ F

-- Theorem statement
theorem area_of_triangle_EOF :
  let triangle_area := abs ((E.1 - O.1) * (F.2 - O.2) - (F.1 - O.1) * (E.2 - O.2)) / 2
  triangle_area = 6 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_EOF_l95_9587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rajit_calc_smaller_l95_9594

/-- Represents a large positive integer -/
def LargePositiveInt : Type := ℕ

/-- Rounds a number up -/
noncomputable def roundUp (n : ℚ) : ℚ := 
  ⌈n⌉

/-- Rounds a number down -/
noncomputable def roundDown (n : ℚ) : ℚ := 
  ⌊n⌋

/-- The exact calculation -/
def exactCalc (a b c d : ℚ) : ℚ := (a + b) / c + d

/-- Rajit's approximate calculation -/
noncomputable def rajitCalc (a b c d : ℚ) : ℚ := 
  (roundUp a + roundUp b) / roundDown c + roundDown d

/-- Theorem stating Rajit's calculation is smaller than the exact calculation -/
theorem rajit_calc_smaller (a b c d : ℚ) 
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  rajitCalc a b c d < exactCalc a b c d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rajit_calc_smaller_l95_9594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l95_9510

theorem unique_solution_exponential_equation :
  ∃! (x y : ℝ), (4 : ℝ)^(x^2 + y) + (4 : ℝ)^(x + y^2) = 2 ∧ x = -1/2 ∧ y = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l95_9510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_switches_in_position_A_l95_9508

/-- Represents a switch with its label components -/
structure Switch where
  x : Fin 6
  y : Fin 6
  z : Fin 6
deriving DecidableEq

/-- The total number of switches -/
def totalSwitches : Nat := 729

/-- The number of positions each switch can be in -/
def numPositions : Nat := 4

/-- The maximum value for x, y, and z in the switch labels -/
def maxExponent : Nat := 5

/-- Checks if one switch's label divides another's -/
def divides (s1 s2 : Switch) : Prop :=
  s1.x ≤ s2.x ∧ s1.y ≤ s2.y ∧ s1.z ≤ s2.z

/-- Represents the 729-step process -/
def process (switches : Fin totalSwitches → Switch) : Prop :=
  ∀ i : Fin totalSwitches, ∀ s : Switch,
    divides (switches i) s → (numPositions - 1) ∣ ((maxExponent + 1 - s.x) *
                                                   (maxExponent + 1 - s.y) *
                                                   (maxExponent + 1 - s.z))

/-- The main theorem to prove -/
theorem switches_in_position_A (switches : Fin totalSwitches → Switch)
  (h : process switches) : 
  (Finset.filter (fun s => (maxExponent + 1 - s.x.val) *
                           (maxExponent + 1 - s.y.val) *
                           (maxExponent + 1 - s.z.val) % numPositions = 0)
    (Finset.image switches Finset.univ)).card = 675 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_switches_in_position_A_l95_9508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_circumference_l95_9544

noncomputable def angle : ℝ := 72 * Real.pi / 180

theorem tangent_circle_circumference 
  (arc_AC arc_BC angle_C : ℝ) 
  (h1 : arc_AC = 15)
  (h2 : arc_BC = 18)
  (h3 : angle_C = angle) :
  ∃ r₃ : ℝ,
    ((37.5 / Real.pi + 45 / Real.pi) / 2) ^ 2 = (37.5 / Real.pi - r₃) * (45 / Real.pi - r₃) ∧
    ∃ C : ℝ, C = 2 * Real.pi * r₃ := by
  sorry

#check tangent_circle_circumference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_circumference_l95_9544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_l95_9577

/-- A sequence {a_n} with given initial condition and recurrence relation -/
def sequence_a : ℕ → ℤ
  | 0 => 1  -- Adding case for 0 to cover all natural numbers
  | 1 => 1
  | n + 2 => 4 * (n + 1) - sequence_a (n + 1)

theorem sequence_a_property (n : ℕ) :
  sequence_a n = 2 * n - 1 ∧ sequence_a 2023 = 4045 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_l95_9577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_of_roots_l95_9543

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 3)

theorem sin_difference_of_roots (x₁ x₂ : ℝ) :
  0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi →
  f x₁ = 1/3 →
  f x₂ = 1/3 →
  Real.sin (x₁ - x₂) = -2 * Real.sqrt 2 / 3 := by
  sorry

#check sin_difference_of_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_of_roots_l95_9543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_rung_length_l95_9595

/-- Calculates the length of each rung in a ladder --/
noncomputable def rung_length (ladder_height_feet : ℝ) (rung_spacing_inches : ℝ) (total_wood_feet : ℝ) : ℝ :=
  let ladder_height_inches := ladder_height_feet * 12
  let num_rungs := ladder_height_inches / rung_spacing_inches
  let total_wood_inches := total_wood_feet * 12
  total_wood_inches / num_rungs

/-- Theorem: Given the specified conditions, each rung of the ladder is 18 inches long --/
theorem ladder_rung_length :
  rung_length 50 6 150 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_rung_length_l95_9595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_inequality_l95_9528

/-- Geometric sequence with positive common ratio -/
noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

/-- Sum of first n terms of geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- Theorem: S_{n+1}a_n > S_na_{n+1} for geometric sequences with positive common ratio -/
theorem geometric_sum_inequality (a₁ : ℝ) (q : ℝ) (n : ℕ) 
    (h₁ : a₁ > 0) (h₂ : q > 0) :
  (geometric_sum a₁ q (n + 1)) * (geometric_sequence a₁ q n) >
  (geometric_sum a₁ q n) * (geometric_sequence a₁ q (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_inequality_l95_9528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_time_l95_9555

/-- Calculates the time taken for a train to cross a platform -/
noncomputable def time_to_cross_platform (train_length : ℝ) (signal_pole_time : ℝ) (platform_length : ℝ) : ℝ :=
  let train_speed := train_length / signal_pole_time
  let total_distance := train_length + platform_length
  total_distance / train_speed

/-- Theorem: A 300 m long train that crosses a signal pole in 18 sec will take approximately 48 sec to cross a 500 m long platform -/
theorem train_crossing_platform_time :
  ∀ (ε : ℝ), ε > 0 →
  |time_to_cross_platform 300 18 500 - 48| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_time_l95_9555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decomposition_and_range_l95_9518

/-- A function that is odd -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function that is even -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function that is increasing on an interval -/
def IncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

/-- A function that is decreasing -/
def Decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem function_decomposition_and_range (a : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x^2 + (a+1)*x + (a+2)
  let p := IncreasingOn f (Set.Ici ((a+1)^2))
  ∃ (g h : ℝ → ℝ),
    (∀ x, f x = g x + h x) ∧
    OddFunction g ∧
    EvenFunction h ∧
    (¬p → False) ∧
    (p ∨ Decreasing g → False) →
    (∀ x, g x = (a+1)*x) ∧
    (∀ x, h x = x^2 + a + 2) ∧
    a ≥ -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decomposition_and_range_l95_9518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_g_evaluation_l95_9520

noncomputable def g (x : ℝ) : ℝ := -1 / (x^2)

theorem nested_g_evaluation :
  g (g (g (g (g 3)))) = -1 / 1853020188851841 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_g_evaluation_l95_9520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematical_ballet_arrangement_l95_9598

/-- The number of boys participating in the ballet -/
def num_boys : ℕ := 5

/-- The distance between each girl and her two designated boys (in meters) -/
def distance : ℝ := 5

/-- The maximum number of girls that can participate in the ballet -/
def max_girls : ℕ := 20

/-- Theorem stating that given 5 boys and the distance condition, 
    the maximum number of girls that can participate is 20 -/
theorem mathematical_ballet_arrangement (n : ℕ) (d : ℝ) (m : ℕ) 
  (h1 : n = num_boys) (h2 : d = distance) (h3 : m = max_girls) : 
  ∃ (arrangement : Finset (ℝ × ℝ)), 
    (∀ girl ∈ arrangement, ∃! (boy1 boy2 : ℝ × ℝ), 
      boy1 ≠ boy2 ∧ 
      (girl.1 - boy1.1)^2 + (girl.2 - boy1.2)^2 = d^2 ∧
      (girl.1 - boy2.1)^2 + (girl.2 - boy2.2)^2 = d^2) ∧
    (Finset.card arrangement = m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematical_ballet_arrangement_l95_9598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l95_9500

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - a * x + (1 - a) / (x + 1)

theorem function_properties (a : ℝ) (h : a ≥ 2) :
  (∃ (x : ℝ), x > -1 ∧ (∀ (y : ℝ), y > -1 → deriv (f a) x = deriv (f a) y → x = y) ∧
    deriv (f a) x = -2 ∧ a = 3) ∧
  (∀ (x y : ℝ), -1 < x ∧ x < y ∧ y < 0 → f a x < f a y) ∧
  (∀ (x y : ℝ), 0 < x ∧ x < y → f a x > f a y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l95_9500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_part1_line_equation_part2_l95_9561

-- Define the ellipse C
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the endpoints of minor axis
def B₁ : ℝ × ℝ := sorry
def B₂ : ℝ × ℝ := sorry

-- Define the line l
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - 1)}

-- Define the intersection points
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Part 1 theorem
theorem ellipse_equation_part1 :
  (Ellipse (2 / Real.sqrt 3) (1 / Real.sqrt 3) = {p : ℝ × ℝ | 3 * p.1 ^ 2 / 4 + 3 * p.2 ^ 2 = 1}) ↔
  (dist F₁ B₁ = dist B₁ B₂ ∧ dist F₁ B₁ = dist F₁ B₂) := by sorry

-- Part 2 theorem
theorem line_equation_part2 :
  (∃ k : ℝ, k = Real.sqrt 7 / 7 ∨ k = -Real.sqrt 7 / 7) ↔
  (Ellipse (Real.sqrt 2) 1 = {p : ℝ × ℝ | p.1 ^ 2 / 2 + p.2 ^ 2 = 1} ∧
   P ∈ Ellipse (Real.sqrt 2) 1 ∧
   Q ∈ Ellipse (Real.sqrt 2) 1 ∧
   F₂ ∈ Line k ∧
   P ∈ Line k ∧
   Q ∈ Line k ∧
   ((P.1 + 1) * (Q.1 + 1) + P.2 * Q.2 = 0)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_part1_line_equation_part2_l95_9561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_165_deg_10_cm_radius_l95_9536

/-- The length of an arc in a circle, given the central angle in degrees and the radius. -/
noncomputable def arcLength (angle : ℝ) (radius : ℝ) : ℝ := (angle * Real.pi * radius) / 180

/-- Theorem: The length of an arc in a circle with radius 10 cm and central angle 165° is 55π/6 cm. -/
theorem arc_length_165_deg_10_cm_radius :
  arcLength 165 10 = (55 * Real.pi) / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_165_deg_10_cm_radius_l95_9536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l95_9524

/-- A quadratic function with specific properties -/
noncomputable def q (x : ℝ) : ℝ := (12 * x^2 - 48) / 5

/-- The function q has vertical asymptotes at x = -2 and x = 2 -/
def has_vertical_asymptotes (f : ℝ → ℝ) : Prop :=
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x + 2| ∧ |x + 2| < δ → |f x| > 1/ε) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |f x| > 1/ε)

/-- Main theorem: q satisfies all given conditions -/
theorem q_satisfies_conditions :
  (∀ x : ℝ, q x = (12 * x^2 - 48) / 5) ∧
  has_vertical_asymptotes q ∧
  q 3 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l95_9524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_wheel_revolutions_l95_9531

/-- Calculates the number of revolutions of the back wheel given the radii of both wheels and the number of revolutions of the front wheel -/
noncomputable def back_wheel_revolutions (front_radius : ℝ) (back_radius : ℝ) (front_revolutions : ℝ) : ℝ :=
  (front_radius * front_revolutions) / back_radius

/-- Theorem stating that for a bicycle with a front wheel of radius 3 feet and a back wheel of radius 6 inches,
    when the front wheel makes 50 revolutions, the back wheel will make 300 revolutions -/
theorem bicycle_wheel_revolutions :
  let front_radius : ℝ := 3  -- in feet
  let back_radius : ℝ := 1/2  -- 6 inches converted to feet
  let front_revolutions : ℝ := 50
  back_wheel_revolutions front_radius back_radius front_revolutions = 300 :=
by
  -- Unfold the definition of back_wheel_revolutions
  unfold back_wheel_revolutions
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_wheel_revolutions_l95_9531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_product_of_roots_l95_9596

theorem unique_solution_for_product_of_roots (x : ℝ) :
  x > 0 ∧ Real.sqrt (12 * x) * Real.sqrt (20 * x) * Real.sqrt (5 * x) * Real.sqrt (30 * x) * Real.sqrt (2 * x) = 60 →
  x = 1 / (2 * (20 : ℝ)^(1/5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_product_of_roots_l95_9596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_is_positive_l95_9509

/-- A line that doesn't pass through the third quadrant -/
structure NonThirdQuadrantLine where
  k : ℝ
  b : ℝ
  not_third_quadrant : k < 0 ∧ b ≥ 0

/-- Four points on a line -/
structure FourPointsOnLine (L : NonThirdQuadrantLine) where
  a : ℝ
  m : ℝ
  e : ℝ
  n : ℝ
  c : ℝ
  d : ℝ
  a_gt_e : a > e
  point_A : m = L.k * a + L.b
  point_B : n = L.k * e + L.b
  point_C : c = L.k * (-m) + L.b
  point_D : d = L.k * (-n) + L.b

/-- The main theorem -/
theorem product_is_positive (L : NonThirdQuadrantLine) (P : FourPointsOnLine L) :
  (P.m - P.n) * (P.c - P.d)^3 > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_is_positive_l95_9509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_max_min_dot_product_l95_9584

/-- Triangle ABC with side lengths a, b, and c -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Circle centered at A with diameter PQ = 2r -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Angle between two vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Vector from point to point -/
def vector (p q : ℝ × ℝ) : ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2)

theorem triangle_circle_max_min_dot_product 
  (abc : Triangle) (circ : Circle) 
  (h_center : circ.center = abc.A) 
  (P Q : ℝ × ℝ) 
  (h_diameter : vector P Q = (2 * circ.radius, 0)) 
  (α : ℝ) 
  (h_angle : angle (vector abc.B P) (vector abc.C Q) = α) :
  let BP := vector abc.B P
  let CQ := vector abc.C Q
  let CB := vector abc.C abc.B
  let AP := vector abc.A P
  (∀ P' Q', 
    vector P' Q' = (2 * circ.radius, 0) → 
    dot_product BP CQ * Real.cos α ≤ dot_product BP' CQ' * Real.cos (angle BP' CQ')) ∧
  (∀ P' Q', 
    vector P' Q' = (2 * circ.radius, 0) → 
    dot_product BP CQ * Real.cos α ≥ dot_product BP' CQ' * Real.cos (angle BP' CQ')) →
  (AP.1 / CB.1 = AP.2 / CB.2 ∨ AP.1 / CB.1 = -AP.2 / CB.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_max_min_dot_product_l95_9584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_properties_l95_9527

noncomputable def Z : ℂ := (1/2)/(1+Complex.I) + (-5/4 + 9/4*Complex.I)

theorem Z_properties :
  (Complex.abs Z = Real.sqrt 5) ∧
  (∃ (p q : ℝ), 2 * Z^2 + p * Z + q = 0 ∧ p = 4 ∧ q = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_properties_l95_9527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_magnitude_l95_9541

-- Define the constants
noncomputable def a : ℝ := 2^(1.2 : ℝ)
noncomputable def b : ℝ := (1/2)^(-(0.2 : ℝ))
noncomputable def c : ℝ := 2 * (Real.log 2 / Real.log 5)

-- State the theorem
theorem order_of_magnitude : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_magnitude_l95_9541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scout_weekend_earnings_l95_9565

noncomputable def base_pay : ℝ := 10.00
def saturday_hours : ℕ := 6
def saturday_deliveries : ℕ := 5
noncomputable def saturday_tip : ℝ := 5.00
def sunday_hours : ℕ := 8
def sunday_deliveries : ℕ := 10
noncomputable def sunday_low_tip : ℝ := 3.00
noncomputable def sunday_high_tip : ℝ := 7.00
noncomputable def overtime_multiplier : ℝ := 1.5

noncomputable def weekend_earnings : ℝ :=
  (base_pay * (saturday_hours : ℝ) + (saturday_deliveries : ℝ) * saturday_tip) +
  (base_pay * overtime_multiplier * (sunday_hours : ℝ) + 
   ((sunday_deliveries / 2 : ℝ) * sunday_low_tip + 
    (sunday_deliveries / 2 : ℝ) * sunday_high_tip))

theorem scout_weekend_earnings :
  weekend_earnings = 255.00 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scout_weekend_earnings_l95_9565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l95_9540

-- Define the quadratic equations
def p (m : ℝ) (x : ℝ) : Prop := x^2 + m*x + 1 = 0
def q (m : ℝ) (x : ℝ) : Prop := 4*x^2 + 4*(m-2)*x + 1 = 0

-- Define the conditions
def has_two_distinct_negative_roots (f : ℝ → Prop) : Prop :=
  ∃ x y, x < 0 ∧ y < 0 ∧ x ≠ y ∧ f x ∧ f y

def has_no_real_roots (f : ℝ → Prop) : Prop :=
  ∀ x, ¬(f x)

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ,
  (has_two_distinct_negative_roots (p m) ∨ has_no_real_roots (q m)) ∧
  ¬(has_two_distinct_negative_roots (p m) ∧ has_no_real_roots (q m)) →
  (1 < m ∧ m ≤ 2) ∨ (3 ≤ m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l95_9540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_children_in_families_with_children_l95_9505

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 10)
  (h2 : total_average = 2)
  (h3 : childless_families = 2) :
  (total_families : ℚ) * total_average / ((total_families : ℚ) - childless_families) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_children_in_families_with_children_l95_9505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_l95_9507

/-- The polar equation of the curve -/
noncomputable def polar_equation (θ : ℝ) : ℝ :=
  1 / (1 - Real.cos θ + Real.sin θ)

/-- The eccentricity of the curve -/
noncomputable def eccentricity : ℝ :=
  Real.sqrt 2

/-- Theorem stating that the curve is a hyperbola -/
theorem curve_is_hyperbola :
  ∃ (ℓ θ₀ : ℝ), ∀ θ,
    polar_equation θ = ℓ / (1 - eccentricity * Real.cos (θ - θ₀)) ∧
    eccentricity > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_l95_9507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l95_9504

theorem trigonometric_identities (α : ℝ) 
  (h1 : Real.sin α = 1/3) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.sin (α - π/6) = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 ∧ 
  Real.cos (2 * α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l95_9504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l95_9546

/-- The function f(x) = -1/2x^2 + 4x - 3ln(x) -/
noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 4*x - 3 * Real.log x

/-- f is non-monotonic on the interval [t, t+1] -/
def is_non_monotonic (f : ℝ → ℝ) (t : ℝ) : Prop :=
  ¬(∀ x y, t ≤ x ∧ x < y ∧ y ≤ t + 1 → f x ≤ f y) ∧
  ¬(∀ x y, t ≤ x ∧ x < y ∧ y ≤ t + 1 → f x ≥ f y)

/-- Main theorem: If f is non-monotonic on [t, t+1], then t is in (0,1) or (2,3) -/
theorem non_monotonic_interval (t : ℝ) :
  is_non_monotonic f t → (0 < t ∧ t < 1) ∨ (2 < t ∧ t < 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l95_9546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l95_9551

/-- The speed of the faster train in kilometers per hour -/
noncomputable def faster_train_speed : ℝ := 72

/-- The speed of the slower train in kilometers per hour -/
noncomputable def slower_train_speed : ℝ := 36

/-- The time taken for the faster train to cross a man in the slower train, in seconds -/
noncomputable def crossing_time : ℝ := 37

/-- Conversion factor from kilometers per hour to meters per second -/
noncomputable def kmph_to_mps : ℝ := 5 / 18

theorem faster_train_length :
  let relative_speed := (faster_train_speed - slower_train_speed) * kmph_to_mps
  relative_speed * crossing_time = 370 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l95_9551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_and_arith_sqrt_l95_9573

theorem square_root_and_arith_sqrt :
  (∀ x : ℝ, x * x = 4/9 → x = 2/3 ∨ x = -2/3) ∧
  (Real.sqrt (Real.sqrt 16) = 4) :=
by
  constructor
  · intro x h
    sorry -- Proof of the first part
  · sorry -- Proof of the second part

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_and_arith_sqrt_l95_9573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_bought_seven_items_l95_9532

/-- Calculates the discounted price for a purchase with the given parameters. -/
noncomputable def discountedPrice (itemPrice : ℚ) (itemCount : ℕ) : ℚ :=
  let totalPrice := itemPrice * itemCount
  if totalPrice ≤ 1000 then totalPrice
  else 1000 + (totalPrice - 1000) * (9/10)

/-- Proves that John bought 7 items given the conditions of the problem. -/
theorem john_bought_seven_items :
  ∃ (itemCount : ℕ), 
    (∀ (n : ℕ), discountedPrice 200 n = 1360 → n = itemCount) ∧ 
    itemCount = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_bought_seven_items_l95_9532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nasa_transport_cost_l95_9574

/-- Calculates the transportation cost for a scientific device -/
noncomputable def transportationCost (weightInGrams : ℝ) (costPerKg : ℝ) (discountPercent : ℝ) (discountThreshold : ℝ) : ℝ :=
  let weightInKg := weightInGrams / 1000
  let baseCost := weightInKg * costPerKg
  if weightInGrams < discountThreshold then
    baseCost * (1 - discountPercent / 100)
  else
    baseCost

/-- The transportation cost for a 400g device is $9000 -/
theorem nasa_transport_cost :
  transportationCost 400 25000 10 500 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nasa_transport_cost_l95_9574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l95_9558

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1
def parabola2 (x : ℝ) : ℝ := x^2 + 4 * x + 4

-- Define the x-coordinates of intersection points
noncomputable def x1 : ℝ := (7 + Real.sqrt 61) / 2
noncomputable def x2 : ℝ := (7 - Real.sqrt 61) / 2

-- Define the y-coordinates of intersection points
noncomputable def y1 : ℝ := parabola1 x1
noncomputable def y2 : ℝ := parabola1 x2

-- Theorem statement
theorem parabola_intersection :
  (parabola1 x1 = parabola2 x1) ∧ 
  (parabola1 x2 = parabola2 x2) ∧
  (∀ x : ℝ, parabola1 x = parabola2 x → x = x1 ∨ x = x2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l95_9558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l95_9564

noncomputable def g (x : ℝ) : ℝ := (3*x - 9)*(2*x - 5) / (2*x + 4)

theorem inequality_solution :
  {x : ℝ | g x ≥ 0} = {x | x < -2} ∪ {x | -2 < x ∧ x ≤ 5/2} ∪ {x | x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l95_9564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_three_halves_l95_9529

theorem complex_expression_equals_three_halves :
  2⁻¹ + (3 - Real.pi)^0 + |2 * Real.sqrt 3 - Real.sqrt 2| + 2 * Real.cos (π / 4) - Real.sqrt 12 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_three_halves_l95_9529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_sea_turtles_l95_9593

theorem baby_sea_turtles (total : ℕ) (h1 : total = 240) : 
  (total - (total / 3) - ((total - (total / 3)) / 5)) = 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_sea_turtles_l95_9593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_selections_5x5_l95_9547

/-- Represents the number of ways to select bricks from rows such that
    adjacent selections are adjacent. -/
def brick_selections (rows columns : Nat) : Nat :=
  let dp : Array (Array Nat) := 
    mkArray rows (mkArray columns 0)
  
  -- Initialize first row
  let dp := dp.set! 0 (mkArray columns 1)
  
  -- Fill the dp table
  let dp := (List.range (rows - 1)).foldl
    (fun acc i =>
      (List.range columns).foldl
        (fun inner_acc j =>
          let value := 
            if j = 0 then 
              (acc.get! i).get! j + (acc.get! i).get! (j + 1)
            else if j = columns - 1 then
              (acc.get! i).get! (j - 1) + (acc.get! i).get! j
            else
              (acc.get! i).get! (j - 1) + (acc.get! i).get! j + (acc.get! i).get! (j + 1)
          inner_acc.set! (i + 1) ((inner_acc.get! (i + 1)).set! j value))
        acc)
    dp

  -- Sum the last row
  (dp.get! (rows - 1)).foldl (fun acc x => acc + x) 0

/-- The theorem stating that the number of valid brick selections for a 5x5 grid is 259. -/
theorem brick_selections_5x5 : brick_selections 5 5 = 259 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_selections_5x5_l95_9547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kinetic_energy_at_4_seconds_l95_9542

/-- The position function of the object -/
noncomputable def s (t : ℝ) : ℝ := 3 * t^2 + t + 4

/-- The velocity function of the object (derivative of s) -/
noncomputable def v (t : ℝ) : ℝ := 6 * t + 1

/-- The mass of the object in kg -/
def m : ℝ := 10

/-- The kinetic energy function -/
noncomputable def kineticEnergy (t : ℝ) : ℝ := (1/2) * m * (v t)^2

theorem kinetic_energy_at_4_seconds :
  kineticEnergy 4 = 3125 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kinetic_energy_at_4_seconds_l95_9542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_in_progressions_l95_9569

-- Define the arithmetic progressions
def arith_prog_1 (n : ℕ) : ℕ := 3 + 4 * n
def arith_prog_2 (n : ℕ) : ℕ := 5 + 6 * n
def arith_prog_3 (n : ℕ) : ℕ := 11 + 10 * n

-- State the theorem
theorem infinitely_many_primes_in_progressions :
  (∃ (f : ℕ → ℕ), StrictMono f ∧ ∀ n, Nat.Prime (arith_prog_1 (f n))) ∧
  (∃ (g : ℕ → ℕ), StrictMono g ∧ ∀ n, Nat.Prime (arith_prog_2 (g n))) ∧
  (∃ (h : ℕ → ℕ), StrictMono h ∧ ∀ n, Nat.Prime (arith_prog_3 (h n))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_in_progressions_l95_9569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x6_in_expansion_l95_9590

theorem coefficient_x6_in_expansion : 
  (Polynomial.coeff (3 * (1 - 3 * X)^7 : Polynomial ℝ) 6) = 567 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x6_in_expansion_l95_9590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distances_to_lila_l95_9534

/-- The vertical and horizontal distances walked together by Mia and Noah to reach Lila -/
def distances_walked_together (mia_x mia_y noah_x noah_y lila_x lila_y : ℚ) : ℚ × ℚ :=
  let meeting_x := (mia_x + noah_x) / 2
  let meeting_y := (mia_y + noah_y) / 2
  (abs (lila_y - meeting_y), abs (lila_x - meeting_x))

/-- Theorem stating the distances walked together by Mia and Noah to reach Lila -/
theorem distances_to_lila : 
  distances_walked_together 3 (-10) 9 18 (12/2) 10 = (6, 0) := by
  sorry

#eval distances_walked_together 3 (-10) 9 18 (12/2) 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distances_to_lila_l95_9534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l95_9579

/-- The time (in seconds) required for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 110 meters traveling at 45 km/hr 
    takes 30 seconds to cross a bridge of length 265 meters -/
theorem train_crossing_bridge_time :
  train_crossing_time 110 45 265 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l95_9579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_x_equals_neg_two_log_two_base_five_l95_9585

-- Define x as given in the problem
noncomputable def x : ℝ := (Real.log 3 / Real.log 9) ^ (Real.log 9 / Real.log 3)

-- Theorem statement
theorem log_x_equals_neg_two_log_two_base_five :
  Real.log x / Real.log 5 = -2 * (Real.log 2 / Real.log 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_x_equals_neg_two_log_two_base_five_l95_9585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_l95_9572

-- Define variables as parameters
variable (p q r s : ℝ)

-- Define the points and midpoints
noncomputable def P : ℝ × ℝ := (p, q)
noncomputable def Q : ℝ × ℝ := (r, s)
noncomputable def N : ℝ × ℝ := ((p + r) / 2, (q + s) / 2)

-- Define the new positions after moving
noncomputable def P' : ℝ × ℝ := (p - 3, q + 5)
noncomputable def Q' : ℝ × ℝ := (r + 4, s - 3)

-- Define the new midpoint
noncomputable def N' : ℝ × ℝ := ((p - 3 + r + 4) / 2, (q + 5 + s - 3) / 2)

-- Theorem statement
theorem midpoint_distance (p q r s : ℝ) :
  Real.sqrt ((N' p q r s).1 - (N p q r s).1)^2 + ((N' p q r s).2 - (N p q r s).2)^2 = Real.sqrt 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_distance_l95_9572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_example_l95_9563

/-- Converts polar coordinates to rectangular coordinates -/
noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- Theorem: The polar coordinates (6, 5π/3) are equivalent to the rectangular coordinates (3, -3√3) -/
theorem polar_to_rectangular_example : 
  polar_to_rectangular 6 (5 * Real.pi / 3) = (3, -3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_example_l95_9563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_exists_l95_9591

-- Define the set of nonzero real numbers
def S : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the properties of the function f
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 1 = 3) ∧
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x + y ≠ 0 → f (1 / (x + y)) = 3 * f (1 / x) + 3 * f (1 / y)) ∧
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x + y ≠ 0 → (x + y + 1) * f (x + y) = x * y * f x * f y)

-- Theorem statement
theorem unique_function_exists :
  ∃! f : ℝ → ℝ, satisfies_conditions f ∧ ∀ x : ℝ, x ≠ 0 → f x = 3 * x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_exists_l95_9591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOC_l95_9580

-- Define the circle Γ centered at O
def Γ : Set (Fin 2 → ℝ) := sorry

-- Define points
variable (P A B C E F O : Fin 2 → ℝ)

-- Define conditions
def is_outside (P : Fin 2 → ℝ) (Γ : Set (Fin 2 → ℝ)) : Prop := sorry
def is_tangent_line (l : Set (Fin 2 → ℝ)) (Γ : Set (Fin 2 → ℝ)) : Prop := sorry
def intersects_at (l₁ l₂ : Set (Fin 2 → ℝ)) (P : Fin 2 → ℝ) : Prop := sorry
def distance (P Q : Fin 2 → ℝ) : ℝ := sorry
def angle (A O B : Fin 2 → ℝ) : ℝ := sorry
def area_triangle (A O C : Fin 2 → ℝ) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_AOC 
  (h_outside : is_outside P Γ)
  (h_tangent_PA : is_tangent_line {x | x = P ∨ x = A} Γ)
  (h_tangent_PB : is_tangent_line {x | x = P ∨ x = B} Γ)
  (h_intersect_PO : intersects_at {x | x = P ∨ x = O} Γ C)
  (h_tangent_C : ∃ l : Set (Fin 2 → ℝ), is_tangent_line l Γ ∧ C ∈ l ∧ E ∈ l ∧ F ∈ l)
  (h_EF_length : distance E F = 8)
  (h_angle_APB : angle A P B = π / 3) :
  area_triangle A O C = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOC_l95_9580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_systematic_sample_l95_9521

/-- Represents a systematic sample of products -/
structure SystematicSample where
  total_products : Nat
  sample_size : Nat
  start : Nat
  interval : Nat

/-- Generates the set of serial numbers for a systematic sample -/
def generate_sample (s : SystematicSample) : Finset Nat :=
  Finset.image (fun i => s.start + i * s.interval) (Finset.range s.sample_size)

/-- Theorem stating that the correct systematic sample for 60 products with 6 samples
    starts at 3 with an interval of 10 -/
theorem correct_systematic_sample :
  let s : SystematicSample := {
    total_products := 60,
    sample_size := 6,
    start := 3,
    interval := 10
  }
  generate_sample s = {3, 13, 23, 33, 43, 53} := by
  sorry

#eval generate_sample {
  total_products := 60,
  sample_size := 6,
  start := 3,
  interval := 10
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_systematic_sample_l95_9521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_bought_three_oranges_l95_9539

-- Define the prices and quantities
def juice_price : ℚ := 1/2
def honey_price : ℚ := 5
def plant_price : ℚ := 9  -- Price per plant (18/2)
def orange_price : ℚ := 9/2

def juice_quantity : ℕ := 7
def honey_quantity : ℕ := 3
def plant_quantity : ℕ := 4

def total_spent : ℚ := 68

-- Define the function to calculate the number of oranges
def calculate_oranges : ℕ :=
  let other_items_cost := juice_price * juice_quantity + 
                          honey_price * honey_quantity + 
                          plant_price * plant_quantity
  let orange_cost := total_spent - other_items_cost
  (orange_cost / orange_price).floor.toNat

-- Theorem statement
theorem joe_bought_three_oranges : calculate_oranges = 3 := by
  -- Unfold the definition of calculate_oranges
  unfold calculate_oranges
  -- Simplify the arithmetic expressions
  simp [juice_price, honey_price, plant_price, orange_price,
        juice_quantity, honey_quantity, plant_quantity, total_spent]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_bought_three_oranges_l95_9539
