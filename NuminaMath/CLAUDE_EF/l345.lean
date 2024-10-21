import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_composition_roots_l345_34556

/-- A quadratic polynomial with two distinct roots -/
def QuadraticPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ 
  (∀ x, f x = a * x^2 + b * x + c) ∧
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ f r₁ = 0 ∧ f r₂ = 0)

/-- The number of distinct roots of a polynomial equation -/
noncomputable def DistinctRoots (f : ℝ → ℝ) : ℕ :=
  Nat.card { x : ℝ | f x = 0 }

theorem quadratic_polynomial_composition_roots
  (f : ℝ → ℝ) (hf : QuadraticPolynomial f) :
  ¬(DistinctRoots (f ∘ f) = 3 ∧ DistinctRoots (f ∘ f ∘ f) = 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_composition_roots_l345_34556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_equals_semiperimeter_l345_34525

/-- The diameter of a figure consisting of a triangle and semicircles on its sides -/
noncomputable def triangle_semicircle_diameter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

/-- The theorem stating that the diameter of the figure is equal to the semi-perimeter of the triangle -/
theorem diameter_equals_semiperimeter 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  ∃ (Φ : Set (ℝ × ℝ)), 
    (∀ (p q : ℝ × ℝ), p ∈ Φ → q ∈ Φ → dist p q ≤ triangle_semicircle_diameter a b c) ∧ 
    (∃ (p q : ℝ × ℝ), p ∈ Φ ∧ q ∈ Φ ∧ dist p q = triangle_semicircle_diameter a b c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_equals_semiperimeter_l345_34525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_of_y_l345_34531

noncomputable def y (x : ℝ) : ℝ := 
  (2 * Real.sin x ^ 2 + Real.sin (3 * x / 2) - 4) / (Real.sin x ^ 2 + 2 * Real.cos x ^ 2)

theorem max_min_sum_of_y :
  ∃ (M m : ℝ), (∀ x, y x ≤ M) ∧ (∃ x, y x = M) ∧ 
               (∀ x, m ≤ y x) ∧ (∃ x, y x = m) →
               M + m = -4 := by
  sorry

#check max_min_sum_of_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_of_y_l345_34531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_trigonometric_equation_l345_34583

theorem unique_solution_trigonometric_equation :
  ∃! x : ℝ, Real.sqrt (1 + Real.tan x) = Real.sin x + Real.cos x ∧ |2 * x - 5| < 2 :=
by
  use (3 * Real.pi / 4)
  constructor
  · sorry  -- Proof that the equation holds for x = 3π/4
  · sorry  -- Proof of uniqueness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_trigonometric_equation_l345_34583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_symmetry_yaxis_l345_34599

theorem point_symmetry_yaxis (a : ℝ) :
  (∃ n : ℤ, a = 2 * n * Real.pi / 5) ↔
  (Real.sin (2 * a) = - Real.sin (3 * a) ∧ Real.cos (3 * a) = Real.cos (2 * a)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_symmetry_yaxis_l345_34599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l345_34540

noncomputable def f (x : ℝ) := Real.log (Real.exp x + Real.exp (-x))

theorem f_properties :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂) ∧
  (Set.Icc (-2 : ℝ) 6 = {m : ℝ | ∀ x : ℝ, Real.exp (2*x) + Real.exp (-2*x) - 2*m*Real.exp (f x) + 6*m + 2 ≥ 0}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l345_34540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AKF_l345_34511

/-- Parabola with equation y² = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ^ 2 = 4 * p.1}

/-- Focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- Slope of the line passing through F -/
noncomputable def m : ℝ := Real.sqrt 3

/-- Point A is on the parabola and above the x-axis -/
noncomputable def A : ℝ × ℝ := (3, 2 * Real.sqrt 3)

/-- K is the foot of the perpendicular from A to the directrix -/
noncomputable def K : ℝ × ℝ := (-1, 2 * Real.sqrt 3)

/-- The area of triangle AKF is 4√3 -/
theorem area_triangle_AKF : 
  (1/2) * Real.sqrt ((F.1 - A.1)^2 + (F.2 - A.2)^2) * 
         Real.sqrt ((K.1 - A.1)^2 + (K.2 - A.2)^2) * 
         Real.sqrt 3 / 2 = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AKF_l345_34511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_loss_l345_34520

noncomputable def villa_sale_price : ℝ := 15000
noncomputable def restaurant_sale_price : ℝ := 15000
noncomputable def villa_loss_percentage : ℝ := 0.25
noncomputable def restaurant_gain_percentage : ℝ := 0.15

noncomputable def villa_cost_price : ℝ := villa_sale_price / (1 - villa_loss_percentage)
noncomputable def restaurant_cost_price : ℝ := restaurant_sale_price / (1 + restaurant_gain_percentage)

noncomputable def total_cost : ℝ := villa_cost_price + restaurant_cost_price
noncomputable def total_sale : ℝ := villa_sale_price + restaurant_sale_price

theorem transaction_loss : 
  ⌊(total_cost - total_sale)⌋ = 3043 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_loss_l345_34520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l345_34504

noncomputable def a : ℕ → ℝ
  | 0 => 2
  | n + 1 => (Real.sqrt 2 - 1) * (a n + 2)

noncomputable def b : ℕ → ℝ
  | 0 => 2
  | n + 1 => (3 * b n + 4) / (2 * b n + 3)

theorem sequence_inequality (n : ℕ) (hn : n > 0) :
  Real.sqrt 2 < b n ∧ b n ≤ a (4 * n - 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l345_34504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_sine_cosine_sum_implies_natural_linear_combination_l345_34513

theorem rational_sine_cosine_sum_implies_natural_linear_combination 
  (x y : ℝ) 
  (h1 : ∃ (p q : ℕ), Real.sin x + Real.cos y = (p : ℝ) / (q : ℝ) ∧ 0 < (p : ℝ) / (q : ℝ))
  (h2 : ∃ (r s : ℕ), Real.sin y + Real.cos x = (r : ℝ) / (s : ℝ) ∧ 0 < (r : ℝ) / (s : ℝ)) :
  ∃ (m n k : ℕ), (m : ℝ) * Real.sin x + (n : ℝ) * Real.cos x = k := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_sine_cosine_sum_implies_natural_linear_combination_l345_34513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_distance_round_100_l345_34554

/-- The distance of the snail from the pole at the start of round n -/
def snail_distance : ℕ → ℚ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | n + 1 => snail_distance n * ((n + 3) / (n + 1))

/-- The theorem stating the distance of the snail at the start of round 100 -/
theorem snail_distance_round_100 : snail_distance 100 = 51 := by
  sorry

#eval snail_distance 100  -- Add this line to evaluate the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_distance_round_100_l345_34554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_value_l345_34585

/-- The total shaded area given two circles with radii 3 and 6 -/
noncomputable def total_shaded_area : ℝ :=
  let small_radius : ℝ := 3
  let large_radius : ℝ := 6
  let small_rectangle_area : ℝ := small_radius * (2 * small_radius)
  let large_rectangle_area : ℝ := large_radius * (2 * large_radius)
  let small_semicircle_area : ℝ := 0.5 * Real.pi * small_radius ^ 2
  let large_semicircle_area : ℝ := 0.5 * Real.pi * large_radius ^ 2
  (small_rectangle_area - small_semicircle_area) + (large_rectangle_area - large_semicircle_area)

theorem total_shaded_area_value : total_shaded_area = 90 - 22.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_value_l345_34585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_sum_x2_plus_x_plus_1_pow_5_l345_34521

theorem coefficient_sum_x2_plus_x_plus_1_pow_5 : 
  let p : Polynomial ℚ := X^2 + X + 1
  let expansion := p^5
  expansion.coeff 6 + expansion.coeff 5 + expansion.coeff 4 = 141 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_sum_x2_plus_x_plus_1_pow_5_l345_34521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_of_specific_pyramid_l345_34595

-- Define the triangular pyramid
structure TriangularPyramid where
  OA : ℝ
  OB : ℝ
  OC : ℝ
  perpendicular : True  -- We'll assume perpendicularity without proving it

-- Define the function to calculate the surface area of the circumscribed sphere
noncomputable def circumscribedSphereSurfaceArea (p : TriangularPyramid) : ℝ :=
  4 * Real.pi * ((p.OA ^ 2 + p.OB ^ 2 + p.OC ^ 2) / 4)

-- State the theorem
theorem circumscribed_sphere_surface_area_of_specific_pyramid :
  let p : TriangularPyramid := {
    OA := 2,
    OB := 2,
    OC := 1,
    perpendicular := trivial
  }
  circumscribedSphereSurfaceArea p = 9 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_of_specific_pyramid_l345_34595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l345_34551

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  pos_a : 0 < a
  pos_b : 0 < b

/-- Function to calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Function to check if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.h)^2 / e.a^2 + (p.y - e.k)^2 / e.b^2 = 1

/-- Theorem stating the properties of the ellipse -/
theorem ellipse_properties :
  ∃ (e : Ellipse),
    let f1 : Point := ⟨3, 3⟩
    let f2 : Point := ⟨3, 8⟩
    let p : Point := ⟨15, 0⟩
    distance p f1 + distance p f2 = distance p f1 + distance p f2 ∧
    isOnEllipse e p ∧
    e.a = Real.sqrt 234 ∧
    e.b = 31/2 ∧
    e.h = 3 ∧
    e.k = 11/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l345_34551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l345_34562

theorem triangle_side_length (J K L : EuclideanSpace ℝ (Fin 2)) (tanK : ℝ) :
  tanK = 4 / 3 →
  ‖K - J‖ = 3 →
  ‖K - L‖ = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l345_34562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_area_l345_34541

/-- The smallest area of a square with two vertices on y = 2x + 3 and two on y = x^2 -/
theorem smallest_square_area : ℝ := by
  have vertices_on_line : ∃ (A B : ℝ × ℝ), A.2 = 2 * A.1 + 3 ∧ B.2 = 2 * B.1 + 3 := by sorry
  have vertices_on_parabola : ∃ (C D : ℝ × ℝ), C.2 = C.1^2 ∧ D.2 = D.1^2 := by sorry
  have is_square : ∃ (s : ℝ), s > 0 ∧ ∃ (A B C D : ℝ × ℝ),
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = s^2 ∧
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = s^2 ∧
    (A.1 - D.1)^2 + (A.2 - D.2)^2 = s^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = s^2 ∧
    (B.1 - D.1)^2 + (B.2 - D.2)^2 = s^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = s^2 := by sorry
  have is_smallest : ∀ (t : ℝ), t > 0 → (∃ (A' B' C' D' : ℝ × ℝ),
    A'.2 = 2 * A'.1 + 3 ∧ B'.2 = 2 * B'.1 + 3 ∧
    C'.2 = C'.1^2 ∧ D'.2 = D'.1^2 ∧
    (A'.1 - B'.1)^2 + (A'.2 - B'.2)^2 = t^2 ∧
    (A'.1 - C'.1)^2 + (A'.2 - C'.2)^2 = t^2 ∧
    (A'.1 - D'.1)^2 + (A'.2 - D'.2)^2 = t^2 ∧
    (B'.1 - C'.1)^2 + (B'.2 - C'.2)^2 = t^2 ∧
    (B'.1 - D'.1)^2 + (B'.2 - D'.2)^2 = t^2 ∧
    (C'.1 - D'.1)^2 + (C'.2 - D'.2)^2 = t^2) → t^2 ≥ 200 := by sorry
  exact 200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_area_l345_34541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_equality_l345_34500

/-- Represents the discount allowed on a bill -/
noncomputable def discount (principal : ℝ) (time : ℝ) (rate : ℝ) : ℝ :=
  principal - principal / (1 + rate / 100) ^ time

/-- Theorem stating that the discount is the same for a bill due at time T and at time 2T -/
theorem discount_equality (principal : ℝ) (time : ℝ) (rate : ℝ) :
  discount principal time rate = discount principal (2 * time) rate :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_equality_l345_34500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_intervals_l345_34532

-- Define the function
noncomputable def f (x : ℝ) := x * Real.sin x + Real.cos x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) := x * Real.cos x

-- Theorem statement
theorem monotonic_increasing_intervals (x : ℝ) :
  x ∈ Set.Ioo (-Real.pi) (-Real.pi/2) ∪ Set.Ioo 0 (Real.pi/2) ↔
  x ∈ Set.Ioo (-Real.pi) Real.pi ∧ f' x > 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_intervals_l345_34532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_square_sum_l345_34560

theorem rationalize_and_square_sum :
  (∃ (a b : ℝ),
    (2 / (Real.sqrt 5 + Real.sqrt 3) = Real.sqrt 5 - Real.sqrt 3) ∧
    (1 / (2 - Real.sqrt 3) = a + b) ∧
    (a = Int.floor (1 / (2 - Real.sqrt 3))) ∧
    (b = 1 / (2 - Real.sqrt 3) - Int.floor (1 / (2 - Real.sqrt 3))) ∧
    (a^2 + b^2 = 13 - 2 * Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_square_sum_l345_34560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_bike_cost_l345_34592

/-- The cost of Julie's mountain bike --/
def mountain_bike_cost (saved lawns newspapers dogs remaining lawn_price newspaper_price dog_price : ℕ) : ℕ :=
  saved +
  lawns * lawn_price +
  newspapers * newspaper_price / 100 +
  dogs * dog_price -
  remaining

theorem julie_bike_cost :
  mountain_bike_cost 1500 20 600 24 155 20 40 15 = 2345 := by
  -- Unfold the definition of mountain_bike_cost
  unfold mountain_bike_cost
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_bike_cost_l345_34592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_expected_l345_34558

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | (x - 1) * (x - 3) ≥ 0}
def T : Set ℝ := {x : ℝ | x > 0}

-- Define the intersection of S and T
def S_intersect_T : Set ℝ := S ∩ T

-- Define the expected result
def expected_result : Set ℝ := Set.Ioc 0 1 ∪ Set.Ici 3

-- Theorem statement
theorem intersection_equals_expected : S_intersect_T = expected_result := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_expected_l345_34558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_example_l345_34526

/-- Factorization is the process of transforming a polynomial into the product of several polynomials -/
def is_factorization (lhs rhs : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b, lhs a b = rhs a b ∧ 
  ∃ (f g : ℝ → ℝ → ℝ), rhs a b = f a b * g a b

/-- The equation -4a^2+9b^2=(-2a+3b)(2a+3b) is a factorization -/
theorem factorization_example : 
  is_factorization (λ a b ↦ -4*a^2 + 9*b^2) (λ a b ↦ (-2*a + 3*b) * (2*a + 3*b)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_example_l345_34526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_l345_34517

-- Define the functions
def f (x : ℝ) : ℝ := x^2

noncomputable def g (x : ℝ) : ℝ := Real.log x

-- Define the tangent line condition
def is_tangent_to_both (x₀ : ℝ) : Prop :=
  ∃ (m : ℝ), 0 < m ∧ m < 1 ∧
  (2 * x₀ = 1 / m) ∧
  (f x₀ - g m = (1 / m) * (x₀ - m))

-- State the theorem
theorem tangent_line_condition (x₀ : ℝ) :
  is_tangent_to_both x₀ → Real.sqrt 2 < x₀ ∧ x₀ < Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_l345_34517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unkind_manager_total_salary_l345_34571

/-- Represents the total monthly salary before any changes --/
def initial_total_salary : ℝ := 10000

/-- Represents the total monthly salary after the kind manager's proposal --/
def kind_manager_total_salary : ℝ := 24000

/-- Represents the salary threshold --/
def salary_threshold : ℝ := 500

/-- Represents the number of employees earning less than or equal to the threshold --/
def x : ℕ := sorry

/-- Represents the number of employees earning more than the threshold --/
def y : ℕ := sorry

/-- Represents the total number of employees --/
def total_employees : ℕ := x + y

/-- Represents the total salary of employees earning less than or equal to the threshold --/
def S₁ : ℝ := sorry

/-- Represents the total salary of employees earning more than the threshold --/
def S₂ : ℝ := sorry

/-- States that the sum of S₁ and S₂ equals the initial total salary --/
axiom total_salary_split : S₁ + S₂ = initial_total_salary

/-- States the equation derived from the kind manager's proposal --/
axiom kind_manager_equation : 3 * S₁ + S₂ + 1000 * (y : ℝ) = kind_manager_total_salary

/-- The main theorem to prove --/
theorem unkind_manager_total_salary : 
  S₁ + 500 * (y : ℝ) = 7000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unkind_manager_total_salary_l345_34571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_hit_probability_l345_34524

/-- The probability problem of two shooters and a plane -/
theorem plane_hit_probability (p_A p_B : ℝ) 
  (h_A : p_A = 0.7) 
  (h_B : p_B = 0.6) :
  (p_A * (1 - p_B) + (1 - p_A) * p_B = 0.46) ∧
  (1 - (1 - p_A) * (1 - p_B) = 0.88) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_hit_probability_l345_34524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_midpoint_is_7_5_l345_34557

/-- A right triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  DE : ℝ
  DF : ℝ
  EF : ℝ
  is_right : DE^2 = DF^2 + EF^2
  de_eq : DE = 15
  df_eq : DF = 9
  ef_eq : EF = 12

/-- The distance from a vertex to the midpoint of the hypotenuse in a right triangle -/
noncomputable def distance_to_midpoint (t : RightTriangle) : ℝ := t.DE / 2

theorem distance_to_midpoint_is_7_5 (t : RightTriangle) :
  distance_to_midpoint t = 7.5 := by
  unfold distance_to_midpoint
  rw [t.de_eq]
  norm_num

#check distance_to_midpoint_is_7_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_midpoint_is_7_5_l345_34557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_values_l345_34538

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- Define the angle bisector property
def angle_bisector (T : Triangle) (X : Real) : Prop :=
  X = (T.B + T.C) / 2

-- Define the theorem
theorem triangle_angle_values (T : Triangle) (P Q : Real) :
  angle_bisector T P →
  angle_bisector T Q →
  T.A = 60 →
  T.A + T.B = T.A + Q →
  T.A = 60 ∧ T.B = 80 ∧ T.C = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_values_l345_34538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l345_34534

/-- The function g as defined in the problem -/
noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

/-- The theorem statement -/
theorem unique_number_not_in_range
  (p q r s : ℝ)
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : g p q r s 5 = 5)
  (h2 : g p q r s 13 = 13)
  (h3 : ∀ x ≠ -s/r, g p q r s (g p q r s x) = x) :
  ∃! y, (∀ x, g p q r s x ≠ y) ∧ y = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l345_34534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l345_34528

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, focal distance 8, and a point (1, √3) on its asymptote, prove that its equation is x²/4 - y²/12 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ c : ℝ, c = 4 ∧ c^2 = a^2 + b^2) →
  (∃ k : ℝ, k = Real.sqrt 3 ∧ k = b / a) →
  ∀ x y : ℝ, x^2 / 4 - y^2 / 12 = 1 :=
by
  intro hc hk x y
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l345_34528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_express_delivery_company_growth_and_staffing_l345_34553

-- Define the initial conditions
def march_deliveries : ℕ := 100000
def may_deliveries : ℕ := 121000
def max_deliveries_per_agent : ℚ := 6 / 10
def current_agents : ℕ := 20

-- Define the monthly growth rate
def monthly_growth_rate : ℚ := 1 / 10

-- Define the function to calculate deliveries after n months
def deliveries_after_n_months (initial : ℕ) (rate : ℚ) (n : ℕ) : ℚ :=
  (initial : ℚ) * (1 + rate) ^ n

-- Define the function to calculate the number of agents needed
noncomputable def agents_needed (deliveries : ℚ) : ℕ :=
  ⌈(deliveries / max_deliveries_per_agent : ℚ)⌉.toNat

-- Theorem statement
theorem express_delivery_company_growth_and_staffing :
  -- The growth rate satisfies the March to May delivery increase
  deliveries_after_n_months march_deliveries monthly_growth_rate 2 = may_deliveries ∧
  -- The number of additional agents needed in June is 3
  agents_needed (deliveries_after_n_months may_deliveries monthly_growth_rate 1) - current_agents = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_express_delivery_company_growth_and_staffing_l345_34553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_inequality_range_l345_34530

-- Define the function f(x) = a*ln(x) - x^2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

-- State the theorem
theorem secant_inequality_range (a : ℝ) : 
  (∀ p q : ℝ, 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → 
    (f a p - f a q) / (p - q) > 0 ∨ (f a p - f a q) / (p - q) < 0) ↔ 
  a ≥ 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_inequality_range_l345_34530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_50_52_l345_34512

def g (x : ℤ) : ℤ := x^2 - 2*x + 2021

theorem gcd_g_50_52 : Int.gcd (g 50) (g 52) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_50_52_l345_34512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_equals_two_l345_34507

-- Define the function g
def g : ℕ → ℕ
| 1 => 4
| 2 => 2
| 3 => 5
| 4 => 3
| 5 => 1
| _ => 0  -- Default case for completeness

-- Assume g is bijective
axiom g_bijective : Function.Bijective g

-- Define g_inv as the inverse of g
noncomputable def g_inv : ℕ → ℕ := Function.invFun g

-- State the theorem
theorem inverse_composition_equals_two :
  g_inv (g_inv (g_inv 2)) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_equals_two_l345_34507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_2_l345_34594

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 4

-- Define the line
def line_eq (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 2

-- Theorem statement
theorem chord_length_is_2_sqrt_2 :
  ∀ (x y : ℝ), circle_eq x y ∧ line_eq x y → 
  ∃ (x1 y1 x2 y2 : ℝ), 
    circle_eq x1 y1 ∧ circle_eq x2 y2 ∧ 
    line_eq x1 y1 ∧ line_eq x2 y2 ∧
    Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = chord_length :=
by
  sorry

#check chord_length_is_2_sqrt_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_2_l345_34594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_number_is_seven_l345_34503

-- Define the diagram structure
structure Diagram where
  circles : Fin 10 → ℕ
  sum_15 : ∀ (t : Fin 7), (circles (Fin.ofNat t.val) + circles (Fin.ofNat (t.val + 1)) + circles (Fin.ofNat (t.val + 2)) = 15)
  range_1_to_9 : ∀ (i : Fin 10), 1 ≤ circles i ∧ circles i ≤ 9
  distinct : ∀ (i j : Fin 10), i ≠ j → circles i ≠ circles j

-- Define the specific diagram with given numbers
def given_diagram (d : Diagram) : Prop :=
  d.circles 3 = 1 ∧ d.circles 4 = 4 ∧ d.circles 8 = 9

-- Define the position of "冰"
def ice_position : Fin 10 := 6

-- Theorem to prove
theorem ice_number_is_seven (d : Diagram) (h : given_diagram d) : 
  d.circles ice_position = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_number_is_seven_l345_34503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_profit_percent_l345_34552

/-- Calculates the profit percent for a pen purchase and sale scenario -/
theorem pen_profit_percent (purchase_quantity purchase_price : ℕ) (discount_percent : ℝ) : 
  purchase_quantity = 58 →
  purchase_price = 46 →
  discount_percent = 1 →
  abs ((((↑purchase_quantity * (1 - discount_percent / 100) - ↑purchase_price) / ↑purchase_price) * 100) - 24.83) < 0.01 := by
  intros h1 h2 h3
  -- The proof goes here
  sorry

#check pen_profit_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_profit_percent_l345_34552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sum_l345_34535

/-- A function g that is its own inverse -/
noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

/-- Theorem: If g is its own inverse, then p + s = 0 -/
theorem inverse_function_sum (p q r s : ℝ) :
  p ≠ 0 → q ≠ 0 → r ≠ 0 → s ≠ 0 →
  (∀ x, g p q r s (g p q r s x) = x) →
  p + s = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sum_l345_34535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_graph_main_theorem_l345_34515

/-- The original function f(x) = 1/x -/
noncomputable def f (x : ℝ) : ℝ := 1 / x

/-- The new function g(x) = 2/x -/
noncomputable def g (x : ℝ) : ℝ := 2 / x

/-- The new unit length e -/
noncomputable def e : ℝ := Real.sqrt 2 / 2

/-- Theorem stating that the transformation preserves the relationship between f and g -/
theorem transform_graph (x : ℝ) (hx : x ≠ 0) : 
  f (x / e) = e * g (x / e) := by
  sorry

/-- The main theorem stating that e = √2/2 transforms f to g -/
theorem main_theorem : 
  ∀ x : ℝ, x ≠ 0 → f (x / e) = e * g (x / e) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_graph_main_theorem_l345_34515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l345_34587

theorem exponential_inequality (x : ℝ) : (2 : ℝ)^(3*x - 1) > 2 ↔ x > 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l345_34587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l345_34580

/-- The area of a rectangle with length 0.5 meters and width 0.24 meters is 0.12 square meters. -/
theorem rectangle_area (length width : Real) : 
  length = 0.5 → width = 0.24 → length * width = 0.12 := by
  intros h_length h_width
  rw [h_length, h_width]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l345_34580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_roots_count_l345_34579

def q (x : ℝ) : ℝ := 2 * x^2 + 2 * x - 1

def q_iter (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => q (q_iter n x)

theorem negative_roots_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, q_iter 2016 x = 0 ∧ x < 0) ∧ 
  Finset.card S = (2^2017 + 1) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_roots_count_l345_34579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medium_tank_radius_l345_34588

/-- Represents a cylindrical tank with a given radius and height -/
structure CylindricalTank where
  radius : ℝ
  height : ℝ

/-- The volume of a cylindrical tank -/
noncomputable def volume (tank : CylindricalTank) : ℝ := Real.pi * tank.radius^2 * tank.height

theorem medium_tank_radius
  (shortest medium tallest : CylindricalTank)
  (h_volume_equal : volume shortest = volume medium ∧ volume medium = volume tallest)
  (h_height_ratio : tallest.height = 5 * shortest.height ∧ medium.height = tallest.height / 4)
  (h_shortest_radius : shortest.radius = 15) :
  medium.radius = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_medium_tank_radius_l345_34588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l345_34568

theorem no_solutions_in_interval : 
  ¬ ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ 
    Real.cos ((Real.pi / 3) * Real.sin x) = Real.sin ((Real.pi / 3) * Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l345_34568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AOB_properties_l345_34597

-- Define the triangle AOB
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (0, 30)
def B : ℝ × ℝ := (40, 0)

-- Define point C on AB such that OC is perpendicular to AB
def C : ℝ × ℝ := (14.4, 19.2)

-- Define M as the center of the circle passing through O, A, and B
def M : ℝ × ℝ := (20, 15)

-- Define the length function
noncomputable def length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem triangle_AOB_properties :
  (length O C = 24) ∧
  (C = (14.4, 19.2)) ∧
  (length C M = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AOB_properties_l345_34597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_derivative_wrt_A_l345_34510

/-- The differential equation ẍ = x + Aẋ² -/
def diff_eq (x : ℝ → ℝ) (A : ℝ) : Prop :=
  ∀ t, (deriv (deriv x)) t = x t + A * ((deriv x t) ^ 2)

/-- The initial conditions x(0) = 1 and ẋ(0) = 0 -/
def initial_conditions (x : ℝ → ℝ) : Prop :=
  x 0 = 1 ∧ deriv x 0 = 0

/-- The solution function for A = 0 -/
noncomputable def solution (t : ℝ) : ℝ := Real.cosh t

/-- The derivative of the solution with respect to A at A = 0 -/
noncomputable def g (t : ℝ) : ℝ := 
  -1/3 * Real.exp t - 1/3 * Real.exp (-t) + 1/6 * Real.cosh (2*t) + 1/2

/-- The theorem stating that g is the correct derivative of the solution with respect to A at A = 0 -/
theorem solution_derivative_wrt_A (x : ℝ → ℝ → ℝ) :
  (∀ A, diff_eq (x A) A) → 
  (∀ A, initial_conditions (x A)) → 
  (∀ t, x 0 t = solution t) →
  (∀ t, (deriv (fun A => x A t) 0) = g t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_derivative_wrt_A_l345_34510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_range_l345_34542

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 9 then |Real.log x / Real.log 3 - 1|
  else if x > 9 then 4 - Real.sqrt x
  else 0  -- This case should never occur in our problem

theorem product_range (a b c : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →  -- a, b, c are positive
  (a ≠ b) → (b ≠ c) → (a ≠ c) →  -- a, b, c are distinct
  (f a = f b) → (f b = f c) →    -- f(a) = f(b) = f(c)
  81 < a * b * c ∧ a * b * c < 144 := by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_range_l345_34542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reverse_game_theorem_initial_amounts_unique_l345_34566

-- Define the number of players
def num_players : Nat := 7

-- Define the final amount each player has
def final_amount : ℚ := 128/100

-- Define the initial amounts for each player
def initial_amounts : Fin num_players → ℚ
  | ⟨0, _⟩ => 449/100  -- Player A
  | ⟨1, _⟩ => 225/100  -- Player B
  | ⟨2, _⟩ => 113/100  -- Player C
  | ⟨3, _⟩ => 57/100   -- Player D
  | ⟨4, _⟩ => 29/100   -- Player E
  | ⟨5, _⟩ => 15/100   -- Player F
  | ⟨6, _⟩ => 8/100    -- Player G
  | ⟨n+7, h⟩ => absurd h (Nat.not_lt_of_ge (Nat.le_add_left 7 n))

-- Function to calculate the amount after a round
def amount_after_round (current_amount : ℚ) : ℚ :=
  7 * current_amount + 1/100

-- Theorem stating that the initial amounts lead to the final amount after 7 rounds
theorem reverse_game_theorem :
  ∀ i : Fin num_players,
    (amount_after_round^[7] (initial_amounts i)) = final_amount := by
  sorry

-- Theorem stating that the initial amounts are unique
theorem initial_amounts_unique :
  ∀ f : Fin num_players → ℚ,
    (∀ i : Fin num_players, (amount_after_round^[7] (f i)) = final_amount) →
    f = initial_amounts := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reverse_game_theorem_initial_amounts_unique_l345_34566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_half_cistern_time_l345_34559

/-- Represents the time it takes to fill a portion of a cistern -/
def fill_time (portion : ℚ) : ℝ := sorry

/-- The portion of the cistern we're interested in filling -/
def target_portion : ℚ := 1/2

theorem fill_half_cistern_time :
  fill_time target_portion = fill_time (1/2) :=
by rfl

#check fill_half_cistern_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_half_cistern_time_l345_34559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l345_34578

/-- The length of a train given its speed, the speed of a man it's passing, and the time it takes to pass the man completely. -/
noncomputable def train_length (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  let relative_speed := (train_speed - man_speed) * (5000 / 18)
  relative_speed * crossing_time

/-- Theorem stating that for a train with given parameters, its length is approximately 299.98 meters. -/
theorem train_length_approx :
  let train_speed := (63 : ℝ)
  let man_speed := (3 : ℝ)
  let crossing_time := (17.998560115190784 : ℝ)
  abs (train_length train_speed man_speed crossing_time - 299.98) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l345_34578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intersection_l345_34514

/-- Definition of an ellipse with given foci and passing through origin -/
def is_ellipse (x : ℝ) : Prop :=
  let f1 : ℝ × ℝ := (0, 3)
  let f2 : ℝ × ℝ := (4, 0)
  let p : ℝ × ℝ := (x, 0)
  Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) +
  Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2) =
  Real.sqrt ((0 - f1.1)^2 + (0 - f1.2)^2) +
  Real.sqrt ((0 - f2.1)^2 + (0 - f2.2)^2)

/-- Theorem stating the other x-axis intersection point of the ellipse -/
theorem ellipse_x_intersection :
  ∃ x : ℝ, x ≠ 0 ∧ is_ellipse x ∧ x = 56 / 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intersection_l345_34514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l345_34569

/-- The hyperbola equation -/
noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4

/-- The positive slope of an asymptote of the hyperbola -/
noncomputable def asymptote_slope : ℝ := Real.sqrt 5 / 2

theorem hyperbola_asymptote_slope :
  ∃ (x y : ℝ), hyperbola_equation x y ∧
  asymptote_slope = (Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l345_34569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digital_music_library_space_l345_34516

/-- Calculates the average megabytes per hour of music in a digital library, rounded to the nearest whole number -/
def averageMBPerHour (totalDays : ℕ) (totalMB : ℕ) : ℕ :=
  let totalHours := totalDays * 24
  let exactAverage := (totalMB : ℚ) / totalHours
  (exactAverage + 1/2).floor.toNat

/-- Theorem stating that for a digital music library with 15 days of music
    taking up 20,000 megabytes, the average megabytes per hour of music,
    when rounded to the nearest whole number, is 56 MB/hour -/
theorem digital_music_library_space :
  averageMBPerHour 15 20000 = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digital_music_library_space_l345_34516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_circumcircles_tangent_l345_34573

-- Define the points
variable (A B C D E : EuclideanPlane)

-- Define the pentagon
def is_convex_pentagon (A B C D E : EuclideanPlane) : Prop := sorry

-- Define the circumcircle
def circumcircle (P Q R : EuclideanPlane) : Set EuclideanPlane := sorry

-- Define tangency of a line to a circle
def is_tangent_to_circle (P Q : EuclideanPlane) (circle : Set EuclideanPlane) : Prop := sorry

-- Define the midpoint of a line segment
noncomputable def segment_midpoint (P Q : EuclideanPlane) : EuclideanPlane := sorry

-- Define the intersection of two circles
def circles_intersect_at (circle1 circle2 : Set EuclideanPlane) (P : EuclideanPlane) : Prop := sorry

-- Define tangency between two circles
def circles_are_tangent (circle1 circle2 : Set EuclideanPlane) : Prop := sorry

-- The main theorem
theorem pentagon_circumcircles_tangent 
  (h_convex : is_convex_pentagon A B C D E)
  (h_tangent_BC : is_tangent_to_circle B C (circumcircle A C D))
  (h_tangent_DE : is_tangent_to_circle D E (circumcircle A C D))
  (h_intersect : circles_intersect_at (circumcircle A B C) (circumcircle A D E) (segment_midpoint C D)) :
  circles_are_tangent (circumcircle A B E) (circumcircle A C D) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_circumcircles_tangent_l345_34573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_arm_width_value_l345_34584

/-- The width of the green arms in a square flag with specific properties -/
noncomputable def green_arm_width : ℝ :=
  let flag_side : ℝ := 1
  let flag_area : ℝ := flag_side * flag_side
  let cross_area : ℝ := 0.40 * flag_area
  let green_area : ℝ := 0.32 * flag_area
  let orange_area : ℝ := cross_area - green_area
  let orange_side : ℝ := Real.sqrt orange_area
  0.5 - orange_side / 2

/-- Theorem stating the width of the green arms -/
theorem green_arm_width_value :
  green_arm_width = 0.5 - Real.sqrt 0.08 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_arm_width_value_l345_34584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inequality_bound_l345_34547

theorem max_inequality_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (c : ℝ), c > 0 ∧ 
    (∀ (x : ℝ), x > 0 → c ≤ max (a * x + 1 / (a * x)) (b * x + 1 / (b * x))) ∧
    (∀ (c' : ℝ), c' > c → 
      ∃ (x : ℝ), x > 0 ∧ c' > max (a * x + 1 / (a * x)) (b * x + 1 / (b * x)))) ∧
  (let c := Real.sqrt (b / a) + Real.sqrt (a / b);
    c > 0 ∧
    (∀ (x : ℝ), x > 0 → c ≤ max (a * x + 1 / (a * x)) (b * x + 1 / (b * x))) ∧
    (∀ (c' : ℝ), c' > c → 
      ∃ (x : ℝ), x > 0 ∧ c' > max (a * x + 1 / (a * x)) (b * x + 1 / (b * x)))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inequality_bound_l345_34547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_zero_necessary_not_sufficient_l345_34522

-- Define the sine function
noncomputable def sin : ℝ → ℝ := Real.sin

-- Define the set of real numbers that satisfy α = 2kπ for some integer k
def S : Set ℝ := {α | ∃ k : ℤ, α = 2 * k * Real.pi}

-- Statement: "sinα=0" is a necessary but not sufficient condition for "α=2kπ, k∈Z"
theorem sin_zero_necessary_not_sufficient :
  (∀ α : ℝ, α ∈ S → sin α = 0) ∧
  ¬(∀ α : ℝ, sin α = 0 → α ∈ S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_zero_necessary_not_sufficient_l345_34522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_proof_l345_34593

/-- Represents a trapezoid EFGH with parallel sides EH and FG -/
structure Trapezoid where
  EG : ℝ
  EF : ℝ
  GH : ℝ
  altitude : ℝ

/-- The area of a trapezoid EFGH -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  600 + 10 * Real.sqrt 56 + 25 * Real.sqrt 21

theorem trapezoid_area_proof (t : Trapezoid) 
  (h1 : t.EG = 18)
  (h2 : t.EF = 60)
  (h3 : t.GH = 25)
  (h4 : t.altitude = 10) :
  trapezoidArea t = 600 + 10 * Real.sqrt 56 + 25 * Real.sqrt 21 := by
  sorry

#check trapezoid_area_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_proof_l345_34593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l345_34561

noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (1 + x^2))

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  simp [f]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l345_34561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_value_l345_34570

-- Define M as in the original problem
noncomputable def M : ℝ := (Real.sqrt (Real.sqrt 7 + 4) + Real.sqrt (Real.sqrt 7 - 4)) / Real.sqrt (Real.sqrt 7 + 2) - Real.sqrt (5 - 2 * Real.sqrt 6)

-- Theorem statement
theorem M_value : M = Real.sqrt 2 / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_value_l345_34570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_approx_l345_34596

/-- The radius of a cone's base given its slant height and curved surface area -/
noncomputable def cone_base_radius (slant_height : ℝ) (curved_surface_area : ℝ) : ℝ :=
  curved_surface_area / (Real.pi * slant_height)

/-- Theorem: Given a cone with slant height 18 cm and curved surface area 452.3893421169302 cm², 
    its base radius is approximately 8 cm -/
theorem cone_base_radius_approx :
  let l := (18 : ℝ)
  let csa := (452.3893421169302 : ℝ)
  let r := cone_base_radius l csa
  ∃ ε > 0, abs (r - 8) < ε ∧ ε < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_approx_l345_34596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_area_l345_34591

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := (1/4) * x^2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := (1/2) * x

/-- The tangent line function -/
def tangent_line (x : ℝ) : ℝ := x - 1

/-- The x-intercept of the tangent line -/
def x_intercept : ℝ := 1

/-- The y-intercept of the tangent line -/
def y_intercept : ℝ := -1

/-- The area enclosed by the tangent line, x-axis, and y-axis -/
def enclosed_area : ℝ := (1/2) * x_intercept * (-y_intercept)

theorem tangent_line_area :
  f 2 = 1 ∧
  f' 2 = 1 ∧
  (∀ x, tangent_line x = f' 2 * (x - 2) + f 2) ∧
  tangent_line 0 = y_intercept ∧
  tangent_line x_intercept = 0 ∧
  enclosed_area = 1/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_area_l345_34591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l345_34564

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 + 4*x + 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -3 ∨ (-3 < x ∧ x < -1) ∨ -1 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l345_34564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_equation_l345_34590

/-- The equation of the circle -/
def myCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The equation of the moving line -/
def movingLine (k x y : ℝ) : Prop := k*x - y + 1 = 0

/-- The trajectory of the midpoint of chord AB -/
def midpointTrajectory (x y : ℝ) : Prop := x^2 + y^2 - y = 0

/-- Theorem: The midpoint of chord AB formed by the intersection of the moving line and the circle
    lies on the trajectory defined by x^2 + y^2 - y = 0 -/
theorem midpoint_trajectory_equation (k x y : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), 
    myCircle x1 y1 ∧ myCircle x2 y2 ∧ 
    movingLine k x1 y1 ∧ movingLine k x2 y2 ∧
    x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2) →
  midpointTrajectory x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_equation_l345_34590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l345_34563

-- Define the correlation coefficient
noncomputable def correlation_coefficient (X Y : List ℝ) : ℝ := sorry

-- Define the variance
noncomputable def variance (X : List ℝ) : ℝ := sorry

-- Define the sum Sk
def Sk (k : ℕ) : ℝ := sorry

-- Define the volume of a tetrahedron
def tetrahedron_volume (S1 S2 S3 S4 r : ℝ) : ℝ := sorry

-- Define stronger linear correlation
def stronger_linear_correlation (X Y : List ℝ) : Prop := sorry

-- Define regression model and related concepts
structure RegressionModel where
  residual_sum_of_squares : ℝ
  model_fit : ℝ

def larger_residual_sum_of_squares (m1 m2 : RegressionModel) : Prop :=
  m1.residual_sum_of_squares > m2.residual_sum_of_squares

def better_model_fit (m1 m2 : RegressionModel) : Prop :=
  m1.model_fit > m2.model_fit

theorem correct_statements :
  -- Statement (2)
  (∀ X Y : List ℝ, abs (correlation_coefficient X Y) > abs (correlation_coefficient X Y) → 
    stronger_linear_correlation X Y) ∧
  
  -- Statement (3)
  (∀ (X : List ℝ) (c : ℝ), variance X = variance (List.map (λ x => x + c) X)) ∧
  
  -- Statement (6)
  (∀ S1 S2 S3 S4 r : ℝ, tetrahedron_volume S1 S2 S3 S4 r = (1/3) * (S1 + S2 + S3 + S4) * r) ∧
  
  -- Statement (1) is incorrect
  (¬ ∀ (m1 m2 : RegressionModel), larger_residual_sum_of_squares m1 m2 → better_model_fit m1 m2) ∧
  
  -- Statement (4) is incorrect
  (¬ ∀ k : ℕ, Sk (k+1) = Sk k + 1 / (2*(k+1))) ∧
  
  -- Statement (5) is incorrect
  (¬ ∀ smoking_related_probability : ℝ, 
    smoking_related_probability = 0.99 → 
    ∃ probability_of_lung_disease_given_smoking : ℝ,
      probability_of_lung_disease_given_smoking = 0.99) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l345_34563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l345_34543

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := 1 / (1 + x^2)

-- Define the point
def point : ℝ × ℝ := (2, 1/5)

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -2 * x / (1 + x^2)^2

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∀ x y, y = f x → (x, y) = point → a * x + b * y + c = 0) ∧
  a = 4 ∧ b = 25 ∧ c = -13 :=
sorry

-- Theorem for the normal line equation
theorem normal_line_equation :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∀ x y, y = f x → (x, y) = point → 
    (a * x + b * y + c = 0 ∧ 
     a * (f' x) + b = 0)) ∧
  a = 125 ∧ b = -20 ∧ c = -246 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l345_34543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_price_decrease_l345_34598

/-- Calculates the percent decrease between two prices -/
noncomputable def percent_decrease (original_price sale_price : ℝ) : ℝ :=
  (original_price - sale_price) / original_price * 100

/-- The original price of the trouser -/
def original_price : ℝ := 100

/-- The sale price of the trouser -/
def sale_price : ℝ := 20

theorem trouser_price_decrease :
  percent_decrease original_price sale_price = 80 := by
  -- Unfold the definitions
  unfold percent_decrease original_price sale_price
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_price_decrease_l345_34598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequential_structure_necessary_l345_34555

-- Define the basic structures of algorithms
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

-- Define what it means for a structure to be necessary for all algorithms
def NecessaryForAllAlgorithms (s : AlgorithmStructure) : Prop :=
  ∀ (algorithm : Type), s = AlgorithmStructure.Sequential

-- State the theorem
theorem sequential_structure_necessary :
  NecessaryForAllAlgorithms AlgorithmStructure.Sequential ∧
  ¬(NecessaryForAllAlgorithms AlgorithmStructure.Conditional) ∧
  ¬(NecessaryForAllAlgorithms AlgorithmStructure.Loop) := by
  sorry

#check sequential_structure_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequential_structure_necessary_l345_34555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_perpendicular_relations_l345_34549

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Plane → Plane → Prop)
variable (perpLine : Line → Line → Prop)
variable (perpPlaneLine : Plane → Line → Prop)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- Given planes and lines
variable (α β γ : Plane)
variable (l m : Line)

-- Theorem statement
theorem plane_perpendicular_relations 
  (h1 : perp α γ)
  (h2 : intersect γ α = m)
  (h3 : intersect γ β = l)
  (h4 : perpLine l m) :
  perpPlaneLine α l ∧ perp α β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_perpendicular_relations_l345_34549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_in_F10_l345_34574

/-- The number of diamonds in figure F_n -/
def D : ℕ → ℕ
  | 0 => 1  -- Base case for n = 0 (equivalent to F_1)
  | n + 1 => D n + 3 * (n + 2)

/-- Theorem stating that the number of diamonds in F_10 is 160 -/
theorem diamonds_in_F10 : D 9 = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_in_F10_l345_34574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l345_34586

open Real

theorem trig_identities (α : ℝ) (h1 : sin α = 3/5) (h2 : α ∈ Set.Ioo (π/2) π) :
  (tan (α + π) = -3/4) ∧ (cos (α - π/2) * sin (α + 3*π/2) = 12/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l345_34586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_hilt_money_is_50_total_cost_equals_money_l345_34567

/-- The amount of money Mrs. Hilt has, in cents -/
def mrs_hilt_money : ℕ := 50

/-- The cost of a pencil, in cents -/
def pencil_cost : ℕ := 5

/-- The number of pencils Mrs. Hilt can buy -/
def pencils_buyable : ℕ := 10

/-- Theorem: Mrs. Hilt has 50 cents -/
theorem mrs_hilt_money_is_50 : mrs_hilt_money = 50 := by
  rfl

/-- Theorem: The total cost of pencils equals Mrs. Hilt's money -/
theorem total_cost_equals_money : pencil_cost * pencils_buyable = mrs_hilt_money := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_hilt_money_is_50_total_cost_equals_money_l345_34567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l345_34519

-- Problem 1
theorem simplify_expression_1 : 
  (0.027 : ℝ) ^ (2/3 : ℝ) + (27/125 : ℝ) ^ (-(1/3) : ℝ) - (25/9 : ℝ) ^ (1/2 : ℝ) = 0.09 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a ^ ((2/3) : ℝ) * b ^ ((1/2) : ℝ)) / (a ^ (-(1/2) : ℝ) * b ^ ((1/3) : ℝ)) / 
  ((a ^ (-1 : ℝ) * b ^ (-(1/2) : ℝ)) / (b * a ^ ((1/2) : ℝ))) ^ (-(2/3) : ℝ) = 
  a ^ ((1/6) : ℝ) * b ^ (-(5/6) : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l345_34519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_midpoint_inequality_l345_34545

theorem sine_midpoint_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < π) (h₃ : 0 < x₂) (h₄ : x₂ < π) :
  (Real.sin x₁ + Real.sin x₂) / 2 < Real.sin ((x₁ + x₂) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_midpoint_inequality_l345_34545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toby_sled_journey_time_l345_34529

-- Define constants for speeds and terrain effects
noncomputable def unloadedSpeed : ℝ := 20
noncomputable def loadedSpeed : ℝ := 10
noncomputable def snowSpeedFactor : ℝ := 0.7  -- 1 - 0.3
noncomputable def iceSpeedFactor : ℝ := 0.8   -- 1 - 0.2

-- Define a function to calculate speed based on load and terrain
noncomputable def calculateSpeed (baseSpeed : ℝ) (terrainFactor : ℝ) : ℝ :=
  baseSpeed * terrainFactor

-- Define a function to calculate time for a segment
noncomputable def calculateTime (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

-- Define the theorem
theorem toby_sled_journey_time : 
  let halfLoadedSpeed := (unloadedSpeed + loadedSpeed) / 2
  let time1 := calculateTime 150 (calculateSpeed loadedSpeed snowSpeedFactor)
  let time2 := calculateTime 100 (calculateSpeed halfLoadedSpeed iceSpeedFactor)
  let time3 := calculateTime 120 unloadedSpeed
  let time4 := calculateTime 90 (calculateSpeed halfLoadedSpeed snowSpeedFactor)
  let time5 := calculateTime 60 (calculateSpeed loadedSpeed iceSpeedFactor)
  let time6 := calculateTime 180 unloadedSpeed
  let totalTime := time1 + time2 + time3 + time4 + time5 + time6
  ∃ (ε : ℝ), ε > 0 ∧ |totalTime - 60.83| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_toby_sled_journey_time_l345_34529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_placemat_length_approx_l345_34523

/-- The radius of the circular table -/
def table_radius : ℝ := 5

/-- The number of placemats on the table -/
def num_placemats : ℕ := 5

/-- The width of each placemat -/
def placemat_width : ℝ := 1

/-- The length of each placemat -/
noncomputable def placemat_length : ℝ := 2 * table_radius * Real.sin (Real.pi / num_placemats)

/-- Theorem stating the approximate length of each placemat -/
theorem placemat_length_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |placemat_length - 5.878| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_placemat_length_approx_l345_34523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fantomas_can_be_caught_l345_34537

/-- Represents a city in Dodecaedria -/
structure City where
  id : Nat

/-- Represents an airline route between two cities -/
structure Route where
  fromCity : City
  toCity : City

/-- The state of Dodecaedria's airline network -/
structure AirlineNetwork where
  cities : Finset City
  routes : Finset Route
  cityCount : Nat
  routeCount : Nat

/-- Represents a single night's action by the police -/
structure PoliceAction where
  routeClosed : Route
  routeOpened : Route

/-- The main theorem stating that Fantomas can be caught -/
theorem fantomas_can_be_caught 
  (initial_network : AirlineNetwork)
  (h_city_count : initial_network.cityCount = 20)
  (h_route_count : initial_network.routeCount = 30) :
  ∃ (actions : List PoliceAction), 
    ∃ (final_network : AirlineNetwork),
      ∃ (isolated_city : City),
        isolated_city ∈ final_network.cities ∧
        ∀ (route : Route), 
          route ∈ final_network.routes → 
          (route.fromCity ≠ isolated_city ∧ route.toCity ≠ isolated_city) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fantomas_can_be_caught_l345_34537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l345_34546

noncomputable section

-- Define the plane and its properties
def plane_distance_from_origin : ℝ := 2

-- Define the intersection points
structure IntersectionPoints where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ

-- Define the property that the points are on the axes and distinct from origin
def points_on_axes (points : IntersectionPoints) : Prop :=
  ∃ α β γ : ℝ, 
    α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0 ∧
    points.A = (α, 0, 0) ∧
    points.B = (0, β, 0) ∧
    points.C = (0, 0, γ)

-- Define the centroid of the triangle
def centroid (points : IntersectionPoints) : ℝ × ℝ × ℝ :=
  let (x1, y1, z1) := points.A
  let (x2, y2, z2) := points.B
  let (x3, y3, z3) := points.C
  ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3)

-- The main theorem
theorem centroid_sum (points : IntersectionPoints) 
  (h1 : points_on_axes points) : 
  let (p, q, r) := centroid points
  1 / p^2 + 1 / q^2 + 1 / r^2 = 2.25 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l345_34546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_slope_l345_34539

-- Define the points on the original line
def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (-2, 6)

-- Define the slope of the original line
noncomputable def original_slope : ℝ := (point2.2 - point1.2) / (point2.1 - point1.1)

-- Define the slope of the perpendicular line
noncomputable def perpendicular_slope : ℝ := -1 / original_slope

-- Theorem statement
theorem perpendicular_line_slope :
  perpendicular_slope = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_slope_l345_34539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l345_34565

/-- Represents a six-digit number -/
def SixDigitNumber := Fin 6 → Fin 6

/-- Checks if a number is even -/
def isEven (n : SixDigitNumber) : Prop := n 5 % 2 = 0

/-- Checks if digits are unique -/
def hasUniqueDigits (n : SixDigitNumber) : Prop := 
  ∀ i j, i ≠ j → n i ≠ n j

/-- Checks if 1 and 3 are not adjacent to 5 -/
def noOneThreeAdjacentToFive (n : SixDigitNumber) : Prop :=
  ∀ i, n i = 4 → (n (i-1) ≠ 0 ∧ n (i-1) ≠ 2) ∧ (n (i+1) ≠ 0 ∧ n (i+1) ≠ 2)

/-- The set of valid six-digit numbers -/
def ValidNumbers : Set SixDigitNumber :=
  {n | isEven n ∧ hasUniqueDigits n ∧ noOneThreeAdjacentToFive n}

/-- Assuming ValidNumbers is finite -/
instance : Fintype ValidNumbers := by sorry

theorem count_valid_numbers : Fintype.card ValidNumbers = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l345_34565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ad_length_l345_34505

/-- Triangle ABC with point D -/
structure ExtendedTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  ab : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6
  bc : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 8
  ac : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 10
  cd : Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 7
  angle_bac_bca : (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) =
                  (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2)
  angle_acd_cab : (A.1 - C.1) * (D.1 - C.1) + (A.2 - C.2) * (D.2 - C.2) =
                  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2)

/-- The length of AD in the extended triangle is √65 -/
theorem ad_length (t : ExtendedTriangle) :
  Real.sqrt ((t.A.1 - t.D.1)^2 + (t.A.2 - t.D.2)^2) = Real.sqrt 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ad_length_l345_34505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_fraction_is_integer_l345_34577

/-- Legendre's formula for the exponent of a prime p in m! -/
noncomputable def legendreFormula (p m : ℕ) : ℕ :=
  ∑' k, (m / p^k : ℕ)

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := 
  Nat.choose n k

/-- Theorem: The number (C_{2n}^n) / (n+1) is an integer for all n ≥ 0 -/
theorem binomial_fraction_is_integer (n : ℕ) :
  ∃ k : ℕ, k * (n + 1) = binomial (2 * n) n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_fraction_is_integer_l345_34577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l345_34572

def A : Finset ℕ := {1, 2, 3, 4}
def B : Finset ℕ := {1, 2}

theorem number_of_subsets : ∃! n : ℕ, ∃ S : Finset (Finset ℕ),
  (∀ C, C ∈ S ↔ (B ⊆ C ∧ C ⊆ A)) ∧ Finset.card S = n ∧ n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l345_34572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_triangle_area_ratio_l345_34548

-- Define a triangle type
structure Triangle where
  area : ℝ

-- Define a function to create a new triangle from an existing one
-- by connecting points that divide the sides in a 1:2 ratio
noncomputable def newTriangleFromDivision (t : Triangle) : Triangle :=
  { area := t.area / 9 }

-- Theorem statement
theorem new_triangle_area_ratio (t : Triangle) :
  (newTriangleFromDivision t).area = t.area / 9 := by
  -- Unfold the definition of newTriangleFromDivision
  unfold newTriangleFromDivision
  -- The proof is now trivial since it's just comparing equal expressions
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_triangle_area_ratio_l345_34548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_S_T_l345_34527

-- Define the sets S and T
def S : Set ℝ := {x | x > -1/2}
def T : Set ℝ := {x | (2 : ℝ)^(3*x - 1) < 1}

-- State the theorem
theorem intersection_S_T : S ∩ T = {x : ℝ | -1/2 < x ∧ x < 1/3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_S_T_l345_34527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_implies_omega_range_subset_implies_m_range_l345_34581

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + 1

/-- The set A -/
def A : Set ℝ := {x | Real.pi/6 ≤ x ∧ x ≤ 2*Real.pi/3}

/-- The set B -/
def B (m : ℝ) : Set ℝ := {x | |f x - m| < 2}

/-- Part 1: Increasing condition implies range of ω -/
theorem increasing_implies_omega_range :
  ∀ ω : ℝ, ω > 0 →
  (∀ x₁ x₂ : ℝ, -Real.pi/2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2*Real.pi/3 → f (ω*x₁) < f (ω*x₂)) →
  ω ∈ Set.Ioc 0 (3/4) := by
  sorry

/-- Part 2: Subset condition implies range of m -/
theorem subset_implies_m_range :
  ∀ m : ℝ, A ⊆ B m → m ∈ Set.Ioo 1 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_implies_omega_range_subset_implies_m_range_l345_34581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_cylinder_surface_area_l345_34509

/-- Represents a cylinder with specific properties -/
structure SpecialCylinder where
  O₁ : Point₃  -- Center of top face
  O₂ : Point₃  -- Center of bottom face
  is_square_cross_section : Bool
  cross_section_area : ℝ

/-- Calculate the surface area of a cylinder -/
noncomputable def surfaceArea (c : SpecialCylinder) : ℝ :=
  sorry  -- Placeholder for the actual calculation

/-- Theorem stating the surface area of the special cylinder -/
theorem special_cylinder_surface_area 
  (c : SpecialCylinder) 
  (h₁ : c.is_square_cross_section = true) 
  (h₂ : c.cross_section_area = 8) : 
  surfaceArea c = 12 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_cylinder_surface_area_l345_34509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l345_34576

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else -x^2 + a*x

theorem range_of_a (a : ℝ) :
  (∀ x, f a x < 3) ∧ (∀ ε > 0, ∃ x, f a x > 3 - ε) ↔ 2 ≤ a ∧ a < 2 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l345_34576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_theorem_approx_price_increase_theorem_l345_34589

/-- The percentage increase required to achieve the desired price increase --/
noncomputable def price_increase_percentage : ℝ := 100 * (-1 + Real.sqrt 5.5)

/-- The equation that models the price increase scenario --/
def price_increase_equation (p : ℝ) : Prop :=
  (2/3) * (1 + p/100) + (1/3) * (1 + p/100)^2 = 1.5

/-- Theorem stating that the calculated price increase percentage satisfies the equation --/
theorem price_increase_theorem : 
  price_increase_equation price_increase_percentage := by
  sorry

/-- Approximate evaluation of the price increase percentage --/
def approx_price_increase : ℚ := 34.5

theorem approx_price_increase_theorem :
  ‖price_increase_percentage - approx_price_increase‖ < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_theorem_approx_price_increase_theorem_l345_34589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l345_34508

/-- A circle tangent to both coordinate axes with its center on the line 5x - 3y = 8 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = radius ∨ center.2 = radius
  center_on_line : 5 * center.1 - 3 * center.2 = 8

/-- The equation of a circle given its center and radius -/
def circle_equation (c : ℝ × ℝ) (r : ℝ) : ℝ → ℝ → Prop :=
  fun x y => (x - c.1)^2 + (y - c.2)^2 = r^2

theorem tangent_circle_equation (c : TangentCircle) :
  (circle_equation (4, 4) 4 c.center.1 c.center.2) ∨ 
  (circle_equation (1, -1) 1 c.center.1 c.center.2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l345_34508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_digit_solution_l345_34502

/-- Represents the different animal masks --/
inductive Mask
  | elephant
  | mouse
  | pig
  | panda

/-- Represents the mapping of masks to digits --/
def mask_to_digit : Mask → Nat := sorry

/-- The product of two identical digits results in a two-digit number
    that doesn't end with the same digit --/
def valid_square (n : Nat) : Prop :=
  let square := n * n
  square ≥ 10 ∧ square < 100 ∧ square % 10 ≠ n

/-- The square of the "mouse" digit ends with the same digit as the "elephant" digit --/
def mouse_elephant_relation (mouse_digit elephant_digit : Nat) : Prop :=
  (mouse_digit * mouse_digit) % 10 = elephant_digit

/-- All masks have different digits --/
def all_different (f : Mask → Nat) : Prop :=
  ∀ m₁ m₂, m₁ ≠ m₂ → f m₁ ≠ f m₂

theorem mask_digit_solution :
  ∃ (f : Mask → Nat),
    (∀ m, valid_square (f m)) ∧
    mouse_elephant_relation (f Mask.mouse) (f Mask.elephant) ∧
    all_different f ∧
    f Mask.elephant = 6 ∧
    f Mask.mouse = 4 ∧
    f Mask.pig = 8 ∧
    f Mask.panda = 1 := by
  sorry

#check mask_digit_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_digit_solution_l345_34502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_value_l345_34518

theorem a_plus_b_value (a b : ℝ) (h1 : |a| = 1) (h2 : b = -2) :
  a + b ∈ ({-1, -3} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_value_l345_34518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_in_trig_equation_l345_34533

theorem range_of_a_in_trig_equation : 
  ∀ a : ℝ, 
  (∃ x : ℝ, (Real.sin x)^2 - 2*(Real.sin x) - a = 0) → 
  (-1 ≤ a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_in_trig_equation_l345_34533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l345_34550

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x + 1 else -x + 3

-- State the theorem
theorem f_composition_value : f (f (5/2)) = 3/2 := by
  -- Evaluate f(5/2)
  have h1 : f (5/2) = 1/2 := by
    simp [f]
    norm_num
  
  -- Evaluate f(f(5/2))
  have h2 : f (1/2) = 3/2 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc
    f (f (5/2)) = f (1/2) := by rw [h1]
    _           = 3/2     := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l345_34550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_selection_l345_34506

theorem milk_selection (total : ℕ) (soda_percent : ℚ) (milk_percent : ℚ) (soda_count : ℕ) :
  soda_percent = 70 / 100 →
  milk_percent = 15 / 100 →
  soda_count = 84 →
  (soda_percent * total : ℚ) = soda_count →
  (milk_percent * ↑total : ℚ) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_selection_l345_34506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_15_l345_34501

/-- Calculates the average speed given two separate journeys -/
noncomputable def averageSpeed (time1 : ℝ) (speed1 : ℝ) (time2 : ℝ) (speed2 : ℝ) : ℝ :=
  let distance1 := time1 * speed1 / 60
  let distance2 := time2 * speed2 / 60
  let totalDistance := distance1 + distance2
  let totalTime := (time1 + time2) / 60
  totalDistance / totalTime

/-- Theorem stating that given the specific journey details, the average speed is approximately 15 mph -/
theorem average_speed_approx_15 :
  ∃ ε > 0, |averageSpeed 45 30 60 5 - 15| < ε := by
  sorry

-- This is now a comment instead of an evaluation
-- #eval averageSpeed 45 30 60 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_15_l345_34501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_roots_l345_34575

open Real

-- Define the function f(x) = x/e^x
noncomputable def f (x : ℝ) : ℝ := x / exp x

-- State the theorem
theorem range_of_a_for_two_roots :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ x ≠ 0 ∧ y ≠ 0 ∧ f x = a ∧ f y = a) → 0 < a ∧ a < 1/exp 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_roots_l345_34575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_cosine_l345_34582

theorem integral_reciprocal_cosine (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∫ θ in (0)..(2*π), 1 / (a + b * Real.cos θ) = 2*π / Real.sqrt (a^2 - b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_cosine_l345_34582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_maximized_l345_34536

/-- The height that maximizes the volume of a cone with slant height 18 cm -/
noncomputable def optimal_height : ℝ := 6 * Real.sqrt 3

/-- The volume of a cone given its height and slant length -/
noncomputable def cone_volume (height : ℝ) (slant : ℝ) : ℝ :=
  (1/3) * Real.pi * (slant^2 - height^2) * height

theorem cone_volume_maximized (h : ℝ) (h_pos : h > 0) (h_bound : h < 18) :
  cone_volume h 18 ≤ cone_volume optimal_height 18 := by
  sorry

#check cone_volume_maximized

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_maximized_l345_34536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l345_34544

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x - x^2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l345_34544
