import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1199_119952

noncomputable def f (x : ℝ) := 3 * x + Real.sin x

theorem inequality_equivalence (m : ℝ) :
  (f (2 * m - 1) + f (3 - m) > 0) ↔ (m > -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1199_119952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_constant_term_expansion_proof_l1199_119902

theorem constant_term_expansion : ℕ := 17

theorem constant_term_expansion_proof :
  let expansion := fun x : ℚ => (x^2 + 2) * (1/x - 1)^6
  let constant_term := 2 * (Nat.choose 6 6) + 1 * (Nat.choose 6 4)
  constant_term = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_constant_term_expansion_proof_l1199_119902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisibility_l1199_119925

theorem factorial_divisibility (n : ℕ) : 2^6 * 3^3 * n = Nat.factorial 10 → n = 2100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisibility_l1199_119925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harvesting_system_correct_l1199_119987

/-- Represents the harvesting rate of a large harvester in hectares per hour -/
def large_harvester_rate : ℝ → Prop := sorry

/-- Represents the harvesting rate of a small harvester in hectares per hour -/
def small_harvester_rate : ℝ → Prop := sorry

/-- The system of equations representing the harvesting problem is correct -/
theorem harvesting_system_correct (x y : ℝ) 
  (hx : large_harvester_rate x) (hy : small_harvester_rate y) : 
  2 * (2 * x + 5 * y) = 3.6 ∧ 5 * (3 * x + 2 * y) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harvesting_system_correct_l1199_119987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_l1199_119942

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem magnitude_of_b (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : ‖a‖ = 1) (h2 : ‖a + b‖ = 1) 
  (h3 : inner a b = -(1/2) * ‖a‖ * ‖b‖) : ‖b‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_l1199_119942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_cylinder_optimal_cylinder_height_l1199_119978

noncomputable section

open Real

/-- A cylinder inscribed in a right circular cone. -/
structure InscribedCylinder (r m : ℝ) where
  x : ℝ  -- distance from the edge of the cone's base to the cylinder's base
  h : 0 < x ∧ x < r  -- x must be positive and less than r

/-- The lateral surface area of an inscribed cylinder. -/
def lateralSurfaceArea (r m : ℝ) (cyl : InscribedCylinder r m) : ℝ :=
  2 * π * (r - cyl.x) * (cyl.x * m / r)

/-- The optimal cylinder has x = r/2. -/
theorem optimal_cylinder (r m : ℝ) (hr : 0 < r) (hm : 0 < m) :
    ∃ (cyl : InscribedCylinder r m), 
      ∀ (cyl' : InscribedCylinder r m), 
        lateralSurfaceArea r m cyl ≥ lateralSurfaceArea r m cyl' ∧ 
        cyl.x = r / 2 := by
  sorry  -- Proof omitted for brevity

/-- The height of the optimal cylinder is m/2. -/
theorem optimal_cylinder_height (r m : ℝ) (hr : 0 < r) (hm : 0 < m) 
    (cyl : InscribedCylinder r m) (hopt : cyl.x = r / 2) :
    cyl.x * m / r = m / 2 := by
  sorry  -- Proof omitted for brevity

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_cylinder_optimal_cylinder_height_l1199_119978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l1199_119914

def distance_AB : ℝ := 10

theorem airplane_altitude 
  (A B C : ℝ × ℝ) 
  (h_distance : (A.1 - B.1)^2 + (A.2 - B.2)^2 = distance_AB^2) 
  (h_north : A.2 = B.2 ∧ A.1 < B.1) 
  (h_west : A.1 = C.1 ∧ B.2 < C.2) 
  (h_angle_A : Real.tan (30 * π / 180) = (C.2 - A.2) / (C.1 - A.1)) 
  (h_angle_B : Real.tan (60 * π / 180) = (C.2 - B.2) / (C.1 - B.1)) 
  : C.2 - A.2 = Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l1199_119914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_distribution_standard_deviation_l1199_119929

-- Define a type for probability distributions
structure Distribution where
  -- The probability density function
  pdf : ℝ → ℝ
  -- Ensure the pdf is non-negative and integrates to 1
  pdf_nonneg : ∀ x, pdf x ≥ 0
  pdf_integral_eq_one : ∫ x in Set.univ, pdf x = 1

-- Define properties of the distribution
def is_symmetric (d : Distribution) (a : ℝ) : Prop :=
  ∀ x, d.pdf (a - x) = d.pdf (a + x)

noncomputable def percent_less_than (d : Distribution) (x : ℝ) : ℝ :=
  100 * ∫ y in Set.Iic x, d.pdf y

noncomputable def percent_between (d : Distribution) (x y : ℝ) : ℝ :=
  100 * ∫ z in Set.Icc x y, d.pdf z

-- State the theorem
theorem symmetric_distribution_standard_deviation 
  (d : Distribution) (a : ℝ) (std_dev : ℝ) :
  is_symmetric d a →
  percent_less_than d (a + std_dev) = 84 →
  percent_between d (a - std_dev) (a + std_dev) = 68 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_distribution_standard_deviation_l1199_119929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_g_l1199_119981

noncomputable def g (x : ℝ) : ℝ := |⌊x⌋| - |⌊2 - x⌋|

theorem symmetry_of_g : ∀ x : ℝ, g x = g (2 - x) := by
  intro x
  simp [g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_g_l1199_119981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_180_l1199_119917

noncomputable def square_area : ℝ := 2025
def rectangle_breadth : ℝ := 10

noncomputable def square_side : ℝ := Real.sqrt square_area
noncomputable def circle_radius : ℝ := square_side
noncomputable def rectangle_length : ℝ := (2 / 5) * circle_radius

theorem rectangle_area_is_180 :
  rectangle_length * rectangle_breadth = 180 := by
  -- Expand definitions
  unfold rectangle_length circle_radius square_side
  -- Simplify expressions
  simp [square_area, rectangle_breadth]
  -- Prove equality
  norm_num
  -- If the above steps are not sufficient, we can use sorry to skip the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_180_l1199_119917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1199_119995

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  B = π / 3 →
  Real.cos A = 4 / 5 →
  b = Real.sqrt 3 →
  Real.sin C = (3 + 4 * Real.sqrt 3) / 10 ∧
  (1 / 2) * b * c * Real.sin A = (36 + 9 * Real.sqrt 3) / 50 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1199_119995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_inequality_range_l1199_119958

theorem cubic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^3 - x^2 - x < m) ↔ m > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_inequality_range_l1199_119958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_product_less_than_one_l1199_119992

open Real

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := a * x - log x
noncomputable def g (x : ℝ) : ℝ := x / exp x

-- Define the equation for the zeros
def equation (x k : ℝ) : Prop := f 1 x + g x = k

-- State the theorem
theorem zeros_product_less_than_one (k : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : equation x₁ k) (h₂ : equation x₂ k) (h₃ : x₁ < x₂) : 
  x₁ * x₂ < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_product_less_than_one_l1199_119992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_100th_term_l1199_119943

def my_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ 
  a 2 = 1 ∧ 
  ∀ n ≥ 2, (a n * a (n-1)) / (a (n-1) - a n) = (a n * a (n+1)) / (a n - a (n+1))

theorem sequence_100th_term (a : ℕ → ℚ) (h : my_sequence a) : a 100 = 1/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_100th_term_l1199_119943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l1199_119912

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / (4 : ℝ) + p.y^2 = 1

/-- The foci of the ellipse -/
noncomputable def foci (e : Ellipse) : (Point × Point) :=
  let c := Real.sqrt (e.a^2 - e.b^2)
  (⟨-c, 0⟩, ⟨c, 0⟩)

/-- Theorem: For the given ellipse, the distance from P to F2 is 7/2 -/
theorem distance_to_focus (e : Ellipse) (p : Point) :
  e.a = 2 →
  e.b = 1 →
  isOnEllipse e p →
  let (f1, f2) := foci e
  p.x = f1.x →
  distance p f2 = 7/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l1199_119912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_interval_l1199_119910

-- Define the distribution function F
noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if x ≤ 3 then x / 3
  else 1

-- Define the probability function P for the interval (a, b]
noncomputable def P (a b : ℝ) : ℝ := F b - F a

-- Theorem statement
theorem probability_in_interval :
  P 2 3 = 1/3 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_interval_l1199_119910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_problem_l1199_119948

/-- The cost structure for apples -/
structure AppleCost where
  l : ℝ  -- cost per kg for first 30 kgs
  q : ℝ  -- cost per kg for additional kgs

/-- Calculate the cost of n kg of apples -/
noncomputable def cost (ac : AppleCost) (n : ℝ) : ℝ :=
  if n ≤ 30 then n * ac.l
  else 30 * ac.l + (n - 30) * ac.q

/-- The apple cost problem -/
theorem apple_cost_problem (ac : AppleCost) :
  cost ac 33 = 360 ∧ cost ac 36 = 420 → cost ac 25 = 250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_problem_l1199_119948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_6_terms_is_84_l1199_119916

/-- A geometric sequence is defined by its first term and common ratio -/
structure GeometricSequence where
  first_term : ℝ
  common_ratio : ℝ

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sum_n_terms (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.first_term * (1 - seq.common_ratio^n) / (1 - seq.common_ratio)

/-- Theorem: If the sum of the first 2 terms of a geometric sequence is 12,
    and the sum of the first 4 terms is 36, then the sum of the first 6 terms is 84 -/
theorem sum_6_terms_is_84 (seq : GeometricSequence) 
    (h1 : sum_n_terms seq 2 = 12)
    (h2 : sum_n_terms seq 4 = 36) :
  sum_n_terms seq 6 = 84 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_6_terms_is_84_l1199_119916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_constant_average_distance_l1199_119903

/-- The constant t for an equilateral triangle with side length 1 -/
noncomputable def t : ℝ := (1 + Real.sqrt 3 / 2) / 3

/-- An equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ
  is_equilateral : ∀ i j : Fin 3, i ≠ j → 
    Real.sqrt ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 1

/-- A point on the perimeter of the triangle -/
def PerimeterPoint (triangle : EquilateralTriangle) := 
  { p : ℝ × ℝ // ∃ i j : Fin 3, i ≠ j ∧ 
    ∃ l : ℝ, 0 ≤ l ∧ l ≤ 1 ∧ 
    p = (l * (triangle.vertices i).1 + (1 - l) * (triangle.vertices j).1,
         l * (triangle.vertices i).2 + (1 - l) * (triangle.vertices j).2) }

/-- The average distance from a point to a set of points -/
noncomputable def averageDistance (p : ℝ × ℝ) (points : List (ℝ × ℝ)) : ℝ :=
  (points.map (λ q => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))).sum / points.length

/-- The main theorem -/
theorem equilateral_triangle_constant_average_distance 
  (triangle : EquilateralTriangle) (points : List (PerimeterPoint triangle)) :
  ∃ (p : PerimeterPoint triangle), averageDistance p.val (points.map Subtype.val) = t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_constant_average_distance_l1199_119903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_theorem_l1199_119919

theorem complex_roots_theorem (z₁ z₂ z₃ : ℂ) 
  (h1 : Complex.abs z₁ = 1)
  (h2 : Complex.abs z₂ = 1)
  (h3 : Complex.abs z₃ = 1)
  (h4 : z₁ + z₂ + z₃ = 1)
  (h5 : z₁ * z₂ * z₃ = 1) :
  ∃ (σ : Equiv.Perm (Fin 3)), 
    (σ.toFun 0 = 0 ∧ z₁ = 1) ∨
    (σ.toFun 0 = 1 ∧ z₁ = Complex.I) ∨
    (σ.toFun 0 = 2 ∧ z₁ = -Complex.I) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_theorem_l1199_119919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1199_119968

theorem line_inclination_angle (x y : ℝ) :
  (Real.sqrt 3 * x - y - 10 = 0) → 
  (Real.arctan (Real.sqrt 3) = 60 * Real.pi / 180) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1199_119968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1199_119904

-- Define the hyperbola
noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define a point on the hyperbola
def point_on_hyperbola (a b X₀ Y₀ : ℝ) : Prop :=
  hyperbola a b X₀ Y₀

-- Define the product of slopes
noncomputable def slope_product (a X₀ Y₀ : ℝ) : ℝ :=
  (Y₀ / (X₀ - a)) * (Y₀ / (X₀ + a))

-- Define the distance from focus to asymptote
noncomputable def focus_asymptote_distance (a b c : ℝ) : ℝ :=
  (b * c) / Real.sqrt (a^2 + b^2)

-- Define eccentricity
noncomputable def eccentricity (a c : ℝ) : ℝ :=
  c / a

-- Theorem statement
theorem hyperbola_properties
  (a b X₀ Y₀ : ℝ)
  (h_pos : a > 0 ∧ b > 0)
  (h_point : point_on_hyperbola a b X₀ Y₀)
  (h_slopes : slope_product a X₀ Y₀ = 144 / 25)
  (h_distance : ∃ c, focus_asymptote_distance a b c = 12) :
  ∃ c,
    eccentricity a c = 13 / 5 ∧
    hyperbola 5 12 X₀ Y₀ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1199_119904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_bounds_l1199_119976

theorem angle_sum_bounds (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < Real.pi / 2)
  (h_acute_β : 0 < β ∧ β < Real.pi / 2)
  (h_acute_γ : 0 < γ ∧ γ < Real.pi / 2)
  (h_cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  3 * Real.pi / 4 < α + β + γ ∧ α + β + γ < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_bounds_l1199_119976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1199_119980

noncomputable def a (x : ℝ) : ℝ × ℝ := (-Real.cos (2 * x), 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (2, 2 - Real.sqrt 3 * Real.sin (2 * x))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 4

noncomputable def g (x : ℝ) : ℝ := -2 * Real.cos x

theorem problem_solution :
  (∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), f x₀ = 2 ∧ ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ f x₀) ∧
  (∀ x, g (x + 2 * Real.pi) = g x) ∧
  (∀ k : ℤ, g (k * Real.pi + Real.pi / 2 + x) = -g (k * Real.pi + Real.pi / 2 - x)) ∧
  (∀ α, α ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2) → f α = -1 → Real.sin (2 * α) = (Real.sqrt 15 + Real.sqrt 3) / 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1199_119980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l1199_119932

/-- Given a train with constant speed that crosses two platforms of different lengths,
    this theorem proves the length of the train. -/
theorem train_length
  (speed : ℝ) -- The constant speed of the train
  (platform1_length : ℝ) -- Length of the first platform
  (platform2_length : ℝ) -- Length of the second platform
  (time1 : ℝ) -- Time to cross the first platform
  (time2 : ℝ) -- Time to cross the second platform
  (train_length : ℝ) -- Length of the train
  (h1 : platform1_length = 350) -- First platform is 350m long
  (h2 : platform2_length = 500) -- Second platform is 500m long
  (h3 : time1 = 15) -- Time to cross first platform is 15 seconds
  (h4 : time2 = 20) -- Time to cross second platform is 20 seconds
  (h5 : speed > 0) -- Speed is positive
  (h6 : (platform1_length + train_length) / time1 = speed) -- Speed equation for first platform
  (h7 : (platform2_length + train_length) / time2 = speed) -- Speed equation for second platform
  : train_length = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l1199_119932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_evaporation_l1199_119998

theorem vessel_evaporation (V : ℝ) (h : V > 0) : 
  let day1_remaining := V - (1/3) * V
  let day2_remaining := day1_remaining - (3/4) * day1_remaining
  day2_remaining = (1/6) * V := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_evaporation_l1199_119998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_F_range_of_a_l1199_119944

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - 2*x - a * Real.log x
def g (x : ℝ) : ℝ := a * x
def F (x : ℝ) : ℝ := f a x + g a x

theorem extreme_values_F (a : ℝ) :
  (∀ x > 0, F a x ≥ a - 1 ∧ (a ≥ 0 → ¬∃ M, ∀ y > 0, F a y ≤ M)) ∧
  ((-2 < a ∧ a < 0) →
    (∃ M, ∀ y > 0, F a y ≤ M ∧ F a (-a/2) = M) ∧
    (∀ x > 0, F a x ≥ a - 1 ∧ F a 1 = a - 1)) ∧
  (a = -2 → ¬∃ m M, ∀ y > 0, m ≤ F a y ∧ F a y ≤ M) ∧
  (a < -2 →
    (∀ x > 0, F a x ≤ a - 1 ∧ F a 1 = a - 1) ∧
    (∀ x > 0, F a x ≥ F a (-a/2) ∧ F a (-a/2) = a - a^2/4 - a * Real.log (-a/2))) :=
sorry

theorem range_of_a :
  (∀ x ≥ (0 : ℝ), Real.sin x / (2 + Real.cos x) ≤ a * x) ↔ a ∈ Set.Ici (1/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_F_range_of_a_l1199_119944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_at_0_equals_1_l1199_119965

open Real

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => sin
| (n + 1) => deriv (f n)

-- State the theorem
theorem f_2017_at_0_equals_1 : f 2017 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_at_0_equals_1_l1199_119965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_60_degrees_b_plus_c_equals_5_l1199_119991

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  (1/2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2 ∧
  t.c + 2 * t.a * Real.cos t.C = 2 * t.b

-- Theorem for part 1
theorem angle_A_is_60_degrees (t : Triangle) 
  (h : triangle_conditions t) : t.A = Real.pi / 3 := by
  sorry

-- Theorem for part 2
theorem b_plus_c_equals_5 (t : Triangle) 
  (h1 : triangle_conditions t) (h2 : t.a = Real.sqrt 7) : t.b + t.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_60_degrees_b_plus_c_equals_5_l1199_119991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1199_119985

noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2)

theorem f_properties :
  let S := Set.Ioo (-1 : ℝ) 1
  (∀ x, x ∈ S → f (-x) = -f x) ∧
  (∀ x1 x2, x1 ∈ S → x2 ∈ S → x1 < x2 → f x1 < f x2) ∧
  {x : ℝ | 0 < x ∧ x < 1/2} = {x : ℝ | f (x-1) + f x < 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1199_119985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equidistant_point_l1199_119997

/-- Parabola defined by y^2 = 4x -/
structure Parabola where
  f : ℝ × ℝ → ℝ
  eq : ∀ (x y : ℝ), f (x, y) = 0 ↔ y^2 = 4*x

/-- Point on a parabola -/
def PointOnParabola (C : Parabola) (A : ℝ × ℝ) : Prop :=
  C.f A = 0

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: For a parabola y^2 = 4x, if a point A on the parabola is equidistant
    from the focus F and point B(3,0), then the distance between A and B is 2√2 -/
theorem parabola_equidistant_point (C : Parabola) (A F : ℝ × ℝ) :
  PointOnParabola C A →
  F = (1, 0) →
  distance A F = distance (3, 0) F →
  distance A (3, 0) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equidistant_point_l1199_119997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_primality_l1199_119955

def sequenceElement (n : ℕ) : ℕ := 
  if n = 1 then 47
  else 47 * (((10 ^ n - 1) / 9))

theorem sequence_primality : ∃! k : ℕ, k > 0 ∧ Nat.Prime (sequenceElement k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_primality_l1199_119955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_range_of_f_l1199_119967

-- Define the function f(x, y) as noncomputable
noncomputable def f (x y : ℝ) : ℝ := (x * y) / (x + y^2)

-- State the theorem
theorem value_range_of_f :
  ∀ x y : ℝ, x + y^2 = 4 →
  ∃ a b : ℝ, a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2 ∧
  (∀ z : ℝ, f x y = z → a ≤ z ∧ z ≤ b) ∧
  (f x y ≠ 1) :=
by
  -- Proof goes here
  sorry

#check value_range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_range_of_f_l1199_119967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1199_119986

noncomputable def f (x θ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x + θ) + Real.cos (2 * x + θ)

theorem function_properties :
  let θ := π / 3
  (∀ x, f x θ = f (-x) θ) ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 4 → f y θ < f x θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1199_119986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_paint_percentage_l1199_119937

/-- Represents a paint mixture with blue, red, and white components. -/
structure PaintMixture where
  blue : ℝ
  red : ℝ
  white : ℝ

/-- Calculates the total amount of paint in the mixture. -/
noncomputable def total_paint (m : PaintMixture) : ℝ := m.blue + m.red + m.white

/-- Calculates the percentage of blue paint in the mixture. -/
noncomputable def blue_percentage (m : PaintMixture) : ℝ :=
  (m.blue / total_paint m) * 100

/-- Theorem: Given a paint mixture with 20% red paint, 140 ounces of blue paint,
    and 20 ounces of white paint, the percentage of blue paint is 70%. -/
theorem blue_paint_percentage :
  ∀ (m : PaintMixture),
    m.blue = 140 ∧
    m.white = 20 ∧
    m.red = 0.2 * total_paint m →
    blue_percentage m = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_paint_percentage_l1199_119937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1199_119940

theorem negation_of_proposition :
  (¬(∀ x : ℝ, x > 0 → x + 1/x ≥ 2)) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + 1/x₀ < 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1199_119940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_street_cost_l1199_119933

/-- Calculates the sum of digits for an arithmetic sequence of house numbers -/
def sumOfDigits (start : ℕ) (count : ℕ) : ℕ :=
  let lastTerm := start + 6 * (count - 1)
  let oneDigit := if start < 10 then 1 else 0
  let twoDigits := (min 99 lastTerm - max 10 start + 1).max 0
  let threeDigits := (lastTerm - max 100 start + 1).max 0
  oneDigit + 2 * twoDigits + 3 * threeDigits

/-- Represents the street with house numbers and calculates the total cost -/
structure Street where
  southStart : ℕ
  northStart : ℕ
  houseCount : ℕ
  totalCost : ℕ

/-- Creates a street with the given parameters -/
def createStreet (southStart northStart houseCount : ℕ) : Street :=
  { southStart := southStart
  , northStart := northStart
  , houseCount := houseCount
  , totalCost := sumOfDigits southStart houseCount + sumOfDigits northStart houseCount }

/-- Theorem stating that the total cost for the given street configuration is 116 dollars -/
theorem street_cost :
  (createStreet 4 5 25).totalCost = 116 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_street_cost_l1199_119933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenters_collinear_l1199_119915

/-- A line in a plane --/
structure Line where
  -- Define a line (this is a placeholder, as the actual definition would depend on how you want to represent lines)
  dummy : Unit

/-- A point in a plane --/
structure Point where
  -- Define a point (this is a placeholder, as the actual definition would depend on how you want to represent points)
  dummy : Unit

/-- The orthocenter of a triangle --/
def orthocenter (a b c : Point) : Point :=
  -- Define the orthocenter (this is a placeholder)
  sorry

/-- Check if three points are collinear --/
def collinear (a b c : Point) : Prop :=
  -- Define collinearity (this is a placeholder)
  sorry

/-- Main theorem: The orthocenters of four triangles formed by four intersecting lines are collinear --/
theorem orthocenters_collinear (l1 l2 l3 l4 : Line) : 
  ∃ (t1 t2 t3 t4 : Point × Point × Point),
    -- t1, t2, t3, t4 are the vertices of the four triangles formed by the intersecting lines
    -- (This condition is simplified and would need to be expanded in a full implementation)
    collinear (orthocenter t1.1 t1.2.1 t1.2.2)
              (orthocenter t2.1 t2.2.1 t2.2.2)
              (orthocenter t3.1 t3.2.1 t3.2.2) ∧
    collinear (orthocenter t1.1 t1.2.1 t1.2.2)
              (orthocenter t2.1 t2.2.1 t2.2.2)
              (orthocenter t4.1 t4.2.1 t4.2.2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenters_collinear_l1199_119915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_P_with_slope_angle_l1199_119999

-- Define the point P
def P : ℝ × ℝ := (-4, 3)

-- Define the slope angle in radians (45° = π/4)
noncomputable def slope_angle : ℝ := Real.pi / 4

-- Define the slope
noncomputable def k : ℝ := Real.tan slope_angle

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y + 7 = 0

-- Theorem statement
theorem line_passes_through_P_with_slope_angle :
  line_equation P.1 P.2 ∧ 
  ∀ (x y : ℝ), line_equation x y → (y - P.2) = k * (x - P.1) := by
  sorry

#check line_passes_through_P_with_slope_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_P_with_slope_angle_l1199_119999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_properties_l1199_119951

-- Define the logarithm with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := log_half x

-- Define the function F
noncomputable def F (x : ℝ) : ℝ := f (x + 1) + f (1 - x)

-- State the theorem
theorem F_properties :
  (∀ x, F (-x) = F x) ∧
  (∀ x, |F x| ≤ 1 ↔ -Real.sqrt 2 / 2 ≤ x ∧ x ≤ Real.sqrt 2 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_properties_l1199_119951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_is_eight_l1199_119977

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![(Real.sqrt 2) / 2, (Real.sqrt 2) / 2],
    ![-(Real.sqrt 2) / 2, (Real.sqrt 2) / 2]]

def is_identity (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  M = ![![1, 0], ![0, 1]]

theorem smallest_power_is_eight :
  (∀ k : ℕ, 0 < k → k < 8 → ¬(is_identity (A ^ k))) ∧
  (is_identity (A ^ 8)) := by
  sorry

#check smallest_power_is_eight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_is_eight_l1199_119977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l1199_119953

/-- The longest side of a triangle with vertices at (2,2), (5,6), and (6,2) has a length of 5 units. -/
theorem longest_side_of_triangle : ∃ (longest_side : ℝ), longest_side = 5 := by
  -- Define the vertices of the triangle
  let v1 : ℝ × ℝ := (2, 2)
  let v2 : ℝ × ℝ := (5, 6)
  let v3 : ℝ × ℝ := (6, 2)

  -- Define a function to calculate the distance between two points
  let dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

  -- Calculate the lengths of all sides
  let side1 := dist v1 v2
  let side2 := dist v2 v3
  let side3 := dist v3 v1

  -- The longest side is 5
  let longest_side := max (max side1 side2) side3
  
  -- Prove that the longest side is 5
  sorry

-- This line is not necessary in a theorem, but if you want to keep it, use #check instead
#check longest_side_of_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l1199_119953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1199_119911

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.c = 1 + Real.sqrt 3 ∧
  t.b = 1 ∧
  Real.sin t.C = Real.sqrt 3 * Real.sin t.A

-- Define the function f(x)
noncomputable def f (x : ℝ) (B : ℝ) : ℝ :=
  2 * Real.sin (2 * x + B) + 4 * (Real.cos x) ^ 2

-- Theorem statement
theorem triangle_problem (t : Triangle) (h : triangle_conditions t) :
  t.B = π / 6 ∧
  Set.Icc (-4 : ℝ) (2 * Real.sqrt 3 + 2) =
    Set.range (fun x => f x t.B) ∩ Set.Icc 0 (π / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1199_119911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_range_l1199_119947

def a (n : ℕ) (p : ℝ) : ℝ := -n + p

def b (n : ℕ) : ℝ := 2^(n-5)

noncomputable def c (n : ℕ) (p : ℝ) : ℝ :=
  if a n p ≤ b n then a n p else b n

theorem p_range (p : ℝ) : 
  (∀ n : ℕ, n ≠ 0 → n ≠ 8 → c 8 p > c n p) → 
  12 < p ∧ p < 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_range_l1199_119947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l1199_119974

/-- Given a differentiable function f : ℝ → ℝ with a tangent line y = 1/2x + 2 at x = 1,
    prove that f(1) + f'(1) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x, f 1 + (deriv f 1) * (x - 1) = 1/2 * x + 2) : 
    f 1 + deriv f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l1199_119974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_specific_triangle_l1199_119959

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 4 ∧ t.c = 2 ∧ t.A = Real.pi / 3

-- Define the circumradius of a triangle
noncomputable def circumradius (t : Triangle) : Real :=
  t.a / (2 * Real.sin t.A)

-- Theorem statement
theorem circumradius_of_specific_triangle :
  ∀ t : Triangle, triangle_conditions t → circumradius t = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_specific_triangle_l1199_119959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_transitivity_l1199_119960

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the perpendicular relation between planes
variable (plane_perp : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_transitivity 
  (m n : Line) (α β : Plane) 
  (h1 : perp_line_plane m α) 
  (h2 : perp_line_plane n β) 
  (h3 : plane_perp α β) : 
  perp_line_line m n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_transitivity_l1199_119960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_and_minimum_l1199_119971

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x - a * (1/2)^x

-- State the theorem
theorem even_function_and_minimum (a : ℝ) :
  (∀ x, f a x = f a (-x)) →  -- f is an even function
  a = -1 ∧                   -- a equals -1
  (∀ x, f (-1) x ≥ 2) ∧      -- minimum value is at least 2
  ∃ x, f (-1) x = 2          -- minimum value of 2 is achieved
  := by sorry

-- Additional lemmas that might be useful for the proof
lemma f_neg_one (x : ℝ) : f (-1) x = 2^x + 2^(-x) := by sorry

lemma f_min_value : ∀ x, f (-1) x ≥ 2 := by sorry

lemma f_achieves_min : ∃ x, f (-1) x = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_and_minimum_l1199_119971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_equality_implies_x_values_l1199_119983

theorem binomial_coefficient_equality_implies_x_values (x : ℝ) :
  (Nat.choose 18 (Int.floor x).toNat = Nat.choose 18 (Int.floor (3*x - 6)).toNat) → (x = 3 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_equality_implies_x_values_l1199_119983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_value_l1199_119949

-- Define the curve function
noncomputable def curve (x : ℝ) : ℝ := x^2 - 3 * Real.log x

-- Define the line function
def line (x m : ℝ) : ℝ := -x + m

-- Theorem statement
theorem tangent_line_m_value :
  ∃ (x₀ : ℝ), x₀ > 0 ∧
  curve x₀ = line x₀ 2 ∧
  (deriv curve) x₀ = (deriv (line · 2)) x₀ := by
  sorry

#check tangent_line_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_value_l1199_119949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_equals_f_0_l1199_119994

open Real

-- Define the function sequence
noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => cos
| (n + 1) => deriv (f n)

-- State the theorem
theorem f_2012_equals_f_0 : f 2012 = f 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_equals_f_0_l1199_119994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l1199_119909

/-- Parabola structure -/
structure Parabola where
  f : ℝ → ℝ
  h : ∀ x, f x = 4 * x

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  h : y^2 = p.f x

/-- Chord of a parabola -/
structure Chord (p : Parabola) where
  a : PointOnParabola p
  b : PointOnParabola p

/-- Circumcircle of a triangle -/
structure Circumcircle where
  center : ℝ × ℝ
  radius : ℝ

/-- Main theorem -/
theorem parabola_chord_length 
  (p : Parabola) 
  (c : Chord p) 
  (f : ℝ × ℝ)
  (circ : Circumcircle)
  (inter : PointOnParabola p) :
  f = (1, 0) →
  (c.a.x - f.1)^2 + (c.a.y - f.2)^2 = (c.b.x - f.1)^2 + (c.b.y - f.2)^2 →
  (inter.x - circ.center.1)^2 + (inter.y - circ.center.2)^2 = circ.radius^2 →
  inter ≠ c.a ∧ inter ≠ c.b →
  (inter.x - f.1)^2 + (inter.y - f.2)^2 = (Real.sqrt 13 - 1)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l1199_119909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l1199_119938

/-- A parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  eq_def : equation = fun x y ↦ y^2 = 4*x

/-- The focus of a parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- A line with slope k passing through (1, 0) -/
structure Line where
  k : ℝ
  equation : ℝ → ℝ → Prop
  eq_def : equation = fun x y ↦ y = k * (x - 1)

/-- Intersection points of a parabola and a line -/
def intersection (p : Parabola) (l : Line) : Set (ℝ × ℝ) :=
  {point | p.equation point.1 point.2 ∧ l.equation point.1 point.2}

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The slope of the line intersecting the parabola y^2 = 4x
    at two points A and B, where |AF| = 3|BF| (F is the focus), is ± √3 -/
theorem parabola_line_intersection_slope (p : Parabola) (l : Line) :
  let f := focus
  let points := intersection p l
  (∃ A B, A ∈ points ∧ B ∈ points ∧ A ≠ B ∧ distance A f = 3 * distance B f) →
  l.k = Real.sqrt 3 ∨ l.k = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l1199_119938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_three_eighths_l1199_119935

/-- A rectangle in the xy-plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The area of a rectangle --/
def rectangle_area (r : Rectangle) : ℝ :=
  (r.x_max - r.x_min) * (r.y_max - r.y_min)

/-- The probability of a point (x,y) in the rectangle satisfying x < y - 1 --/
noncomputable def probability_x_less_y_minus_one (r : Rectangle) : ℝ :=
  let triangle_area := ((r.x_max - (r.y_max - 1)) * r.y_max) / 2
  triangle_area / rectangle_area r

/-- The specific rectangle in the problem --/
def problem_rectangle : Rectangle where
  x_min := 0
  x_max := 4
  y_min := 0
  y_max := 2
  h_x := by norm_num
  h_y := by norm_num

/-- The main theorem --/
theorem probability_is_three_eighths :
  probability_x_less_y_minus_one problem_rectangle = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_three_eighths_l1199_119935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l1199_119900

/-- Circle C₁ with equation x² + y² - 8x - 4y + 11 = 0 -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 4*y + 11 = 0

/-- Circle C₂ with equation x² + y² + 4x + 2y + 1 = 0 -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- The distance between two points (x₁, y₁) and (x₂, y₂) -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem min_distance_between_circles :
  ∀ x₁ y₁ x₂ y₂ : ℝ, C₁ x₁ y₁ → C₂ x₂ y₂ →
  distance x₁ y₁ x₂ y₂ ≥ 3 * Real.sqrt 5 - 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l1199_119900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fruits_is_174_l1199_119961

/-- Represents the contents of a fruit basket -/
structure FruitBasket where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ
  grapes : ℕ

/-- Calculates the total number of fruits in a basket -/
def totalFruits (basket : FruitBasket) : ℕ :=
  basket.apples + basket.oranges + basket.bananas + basket.grapes

/-- Represents the group of 6 fruit baskets -/
def fruitBaskets : List FruitBasket :=
  [
    ⟨9, 15, 14, 12⟩, -- First basket
    ⟨9, 15, 14, 12⟩, -- Second basket
    ⟨9, 15, 14, 12⟩, -- Third basket
    ⟨7, 13, 12, 10⟩, -- Fourth basket
    ⟨12, 10, 14, 12⟩, -- Fifth basket
    ⟨0, 0, 28, 6⟩ -- Sixth basket
  ]

/-- Theorem stating that the total number of fruits in all baskets is 174 -/
theorem total_fruits_is_174 :
  (fruitBaskets.map totalFruits).sum = 174 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fruits_is_174_l1199_119961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bipartite_implies_two_color_partition_l1199_119936

/-- A graph is bipartite if its vertices can be colored with two colors such that
    no two adjacent vertices have the same color. -/
def Bipartite (G : Type*) (adj : G → G → Prop) : Prop :=
  ∃ (c : G → Bool), ∀ (v w : G), v ≠ w → adj v w → c v ≠ c w

/-- A partition of a graph's vertices into two disjoint sets. -/
def TwoColorPartition (G : Type*) (A B : Set G) : Prop :=
  A ∪ B = Set.univ ∧ A ∩ B = ∅

/-- Every edge in the graph connects vertices from different sets of the partition. -/
def ValidPartition (G : Type*) (adj : G → G → Prop) (A B : Set G) : Prop :=
  ∀ (v w : G), adj v w → (v ∈ A ∧ w ∈ B) ∨ (v ∈ B ∧ w ∈ A)

/-- If a graph is bipartite, then its vertices can be partitioned into two disjoint sets
    such that every edge connects a vertex in one set to a vertex in the other set. -/
theorem bipartite_implies_two_color_partition (G : Type*) (adj : G → G → Prop) :
  Bipartite G adj → ∃ (A B : Set G), TwoColorPartition G A B ∧ ValidPartition G adj A B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bipartite_implies_two_color_partition_l1199_119936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_players_quit_l1199_119956

theorem football_players_quit : 10 = 13 - (15 - (16 - 4)) := by
  -- Initial values
  let initial_football_players := 13
  let initial_cheerleaders := 16
  let cheerleaders_quit := 4
  let total_left := 15

  -- Calculate the number of football players who quit
  let football_players_quit := initial_football_players - (total_left - (initial_cheerleaders - cheerleaders_quit))
  
  -- Prove that football_players_quit equals 10
  calc
    football_players_quit = 13 - (15 - (16 - 4)) := rfl
    _ = 13 - (15 - 12) := by rfl
    _ = 13 - 3 := by rfl
    _ = 10 := by rfl

#check football_players_quit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_players_quit_l1199_119956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_circle_l1199_119984

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem max_distance_to_circle : 
  ∃ (max_dist : ℝ), max_dist = 3 ∧ 
  ∀ (P : ℝ × ℝ), circle_C P.1 P.2 → 
  Real.sqrt ((P.1 - point_A.1)^2 + (P.2 - point_A.2)^2) ≤ max_dist :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_circle_l1199_119984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1199_119901

noncomputable def f (x : ℝ) : ℝ := (x^3 - 3*x^2 + 5*x - 2) / (x^2 - 5*x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 2 ∨ (2 < x ∧ x < 3) ∨ 3 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1199_119901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l1199_119941

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(-x)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (2, 1)

def BA : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem propositions_truth : 
  (∀ x, f x = f (-x)) ∧  -- Proposition ①
  (¬ ∀ (a b : ℝ × ℝ) (θ : ℝ), Real.cos θ = dot_product a b / (magnitude a * magnitude b)) ∧  -- Proposition ②
  (dot_product BA (1, 0) / magnitude BA ≠ 4/5) ∧  -- Proposition ③
  ({α : ℝ | ∃ k : ℤ, α = k * π / 2} ≠ {α : ℝ | ∃ k : ℤ, α = π / 2 + k * π}) ∧  -- Proposition ④
  (∀ x, 3 * Real.sin (2 * (x - π/6) + π/3) = 3 * Real.sin (2 * x)) ∧  -- Proposition ⑤
  (¬ ∀ x ∈ Set.Icc 0 π, ∀ ε > 0, Real.sin (x - π/2) < Real.sin (x - π/2 + ε))  -- Proposition ⑥
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l1199_119941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_split_into_four_regions_l1199_119966

-- Define the two lines
noncomputable def line1 (x y : ℝ) : Prop := y = 3 * x
noncomputable def line2 (x y : ℝ) : Prop := x = 3 * y + 2

-- Define the intersection point
noncomputable def intersection : ℝ × ℝ := (-1/4, -3/4)

-- Theorem stating that the lines split the plane into 4 regions
theorem plane_split_into_four_regions :
  ∃ (R1 R2 R3 R4 : Set (ℝ × ℝ)),
    (∀ p : ℝ × ℝ, p ∈ R1 ∨ p ∈ R2 ∨ p ∈ R3 ∨ p ∈ R4) ∧
    (R1 ∩ R2 = ∅) ∧ (R1 ∩ R3 = ∅) ∧ (R1 ∩ R4 = ∅) ∧
    (R2 ∩ R3 = ∅) ∧ (R2 ∩ R4 = ∅) ∧ (R3 ∩ R4 = ∅) ∧
    (∀ p : ℝ × ℝ, p ∈ R1 → ¬(line1 p.1 p.2 ∨ line2 p.1 p.2)) ∧
    (∀ p : ℝ × ℝ, p ∈ R2 → (line1 p.1 p.2 ∧ ¬line2 p.1 p.2) ∨ (¬line1 p.1 p.2 ∧ line2 p.1 p.2)) ∧
    (∀ p : ℝ × ℝ, p ∈ R3 → (line1 p.1 p.2 ∧ line2 p.1 p.2)) ∧
    (∀ p : ℝ × ℝ, p ∈ R4 → ¬(line1 p.1 p.2 ∨ line2 p.1 p.2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_split_into_four_regions_l1199_119966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_G_l1199_119918

noncomputable def F (p q : ℝ) : ℝ := -3*p*q + 2*p*(1-q) + 4*(1-p)*q - 5*(1-p)*(1-q)

noncomputable def G (p : ℝ) : ℝ := ⨆ q ∈ Set.Icc 0 1, F p q

theorem minimize_G :
  ∃ (p : ℝ), p ∈ Set.Icc 0 1 ∧ 
  (∀ (p' : ℝ), p' ∈ Set.Icc 0 1 → G p ≤ G p') ∧
  p = 9/10 := by
  sorry

#check minimize_G

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_G_l1199_119918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1199_119934

/-- The length of a train that crosses an electric pole in a given time at a given speed. -/
noncomputable def train_length (crossing_time : ℝ) (speed_kmh : ℝ) : ℝ :=
  crossing_time * (speed_kmh * (1000 / 3600))

/-- Theorem stating that a train crossing an electric pole in 20 seconds at 90 km/h has a length of 500 meters. -/
theorem train_length_calculation :
  train_length 20 90 = 500 := by
  sorry

/-- Compute an approximation of the train length -/
def train_length_approx (crossing_time : Float) (speed_kmh : Float) : Float :=
  crossing_time * (speed_kmh * (1000 / 3600))

#eval train_length_approx 20 90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1199_119934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_price_increase_l1199_119931

theorem gasoline_price_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (price_increase : ℝ) 
  (quantity_decrease : ℝ) 
  (spending_increase : ℝ)
  (h1 : original_price > 0) 
  (h2 : original_quantity > 0) 
  (h3 : quantity_decrease = 0.16) 
  (h4 : spending_increase = 0.05) :
  price_increase = 0.25 ↔ 
    (original_price * (1 + price_increase) * (original_quantity * (1 - quantity_decrease)) = 
     original_price * original_quantity * (1 + spending_increase)) := by
  sorry

#check gasoline_price_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_price_increase_l1199_119931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_theorem_l1199_119930

/-- The radius of a circumscribed sphere around a regular quadrilateral pyramid -/
noncomputable def circumscribed_sphere_radius (a : ℝ) (α : ℝ) : ℝ :=
  (a * (3 + Real.cos (2 * α))) / (4 * Real.sin (2 * α))

/-- Theorem: The radius of the circumscribed sphere around a regular quadrilateral pyramid -/
theorem circumscribed_sphere_radius_theorem 
  (a : ℝ) (α : ℝ) (h₁ : a > 0) (h₂ : 0 < α ∧ α < Real.pi / 2) :
  ∃ (R : ℝ), R = circumscribed_sphere_radius a α ∧ 
  R = (a * (3 + Real.cos (2 * α))) / (4 * Real.sin (2 * α)) := by
  sorry

#check circumscribed_sphere_radius_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_theorem_l1199_119930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_l1199_119964

theorem tan_half_angle (α : ℝ) 
  (h1 : Real.cos α = Real.sqrt 3 / 3) 
  (h2 : π < α ∧ α < 2 * π) : 
  Real.tan (α / 2) = (Real.sqrt 2 - Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_l1199_119964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l1199_119922

/-- Given a linear function f and its solution set for an inequality,
    prove the solution set for another inequality. -/
theorem solution_set_inequality (a : ℝ) :
  (∀ x, |a * x + 2| < 6 ↔ -1 < x ∧ x < 2) →
  {x : ℝ | a * x + 2 ≤ 1} = Set.Iic (1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l1199_119922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1199_119939

theorem log_inequality (a b c : ℝ) : 
  a = (Real.log 28) / (Real.log 4) →
  b = (Real.log 35) / (Real.log 5) →
  c = (Real.log 42) / (Real.log 6) →
  a > b ∧ b > c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1199_119939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_less_than_25_and_major_l1199_119905

/-- Represents the major programs offered in the graduating class -/
inductive Major
| Science
| Arts
| Business

/-- Represents the gender of students -/
inductive Gender
| Male
| Female

/-- Represents the age category of students -/
inductive AgeCategory
| LessThan25
| AtLeast25

/-- Represents the distribution of students across different categories -/
structure ClassDistribution where
  malePercentage : ℝ
  femalePercentage : ℝ
  sciencePercentage : ℝ
  artsPercentage : ℝ
  businessPercentage : ℝ
  maleScienceAtLeast25Percentage : ℝ
  maleArtsAtLeast25Percentage : ℝ
  maleBusinessAtLeast25Percentage : ℝ
  femaleScienceAtLeast25Percentage : ℝ
  femaleArtsAtLeast25Percentage : ℝ
  femaleBusinessAtLeast25Percentage : ℝ

/-- Calculates the probability of a randomly selected student being less than 25 years old and pursuing a specific major -/
def probabilityLessThan25AndMajor (d : ClassDistribution) (m : Major) : ℝ :=
  match m with
  | Major.Science => 
      d.malePercentage * d.sciencePercentage * (1 - d.maleScienceAtLeast25Percentage) +
      d.femalePercentage * d.sciencePercentage * (1 - d.femaleScienceAtLeast25Percentage)
  | Major.Arts => 
      d.malePercentage * d.artsPercentage * (1 - d.maleArtsAtLeast25Percentage) +
      d.femalePercentage * d.artsPercentage * (1 - d.femaleArtsAtLeast25Percentage)
  | Major.Business => 
      d.malePercentage * d.businessPercentage * (1 - d.maleBusinessAtLeast25Percentage) +
      d.femalePercentage * d.businessPercentage * (1 - d.femaleBusinessAtLeast25Percentage)

theorem probability_less_than_25_and_major (d : ClassDistribution) : 
  d.malePercentage = 0.4 ∧ 
  d.femalePercentage = 0.6 ∧ 
  d.sciencePercentage = 0.3 ∧ 
  d.artsPercentage = 0.45 ∧ 
  d.businessPercentage = 0.25 ∧
  d.maleScienceAtLeast25Percentage = 0.4 ∧
  d.maleArtsAtLeast25Percentage = 0.5 ∧
  d.maleBusinessAtLeast25Percentage = 0.35 ∧
  d.femaleScienceAtLeast25Percentage = 0.3 ∧
  d.femaleArtsAtLeast25Percentage = 0.45 ∧
  d.femaleBusinessAtLeast25Percentage = 0.2 →
  (abs (probabilityLessThan25AndMajor d Major.Science - 0.198) < 0.001 ∧
   abs (probabilityLessThan25AndMajor d Major.Arts - 0.2385) < 0.001 ∧
   abs (probabilityLessThan25AndMajor d Major.Business - 0.185) < 0.001) :=
by sorry

#check probability_less_than_25_and_major

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_less_than_25_and_major_l1199_119905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_and_difference_l1199_119927

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + (1/2) * x^2 - (5/2) * x

-- Define the function g
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := f x - (b - 3/2) * x

-- State the theorem
theorem function_extrema_and_difference 
  (b : ℝ) (x₁ x₂ : ℝ) 
  (h_b : b ≥ 3/2) 
  (h_x : x₁ < x₂) 
  (h_extreme : ∀ x, x ∈ Set.Icc (1/4 : ℝ) 2 → 
    g b x₁ ≥ g b x ∧ g b x₂ ≥ g b x) : 
  (∀ x, x ∈ Set.Icc (1/4 : ℝ) 2 → f x ≤ -Real.log 2 - 9/8) ∧ 
  (∀ x, x ∈ Set.Icc (1/4 : ℝ) 2 → f x ≥ -Real.log 4 - 19/32) ∧ 
  (g b x₁ - g b x₂ ≤ 15/8 - 2 * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_and_difference_l1199_119927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l1199_119972

theorem sqrt_inequality : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l1199_119972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_path_exists_l1199_119954

/-- Represents a chessboard of dimensions m × n -/
structure Chessboard where
  m : ℕ
  n : ℕ

/-- Represents the movement constraints of the rook -/
def validRookPath (board : Chessboard) (path : List (ℕ × ℕ)) : Prop :=
  path.length = board.m * board.n ∧
  (∀ i, i + 1 < path.length →
    (path[i]!.1 = path[i+1]!.1 ∧ path[i]!.2 ≠ path[i+1]!.2) ∨
    (path[i]!.1 ≠ path[i+1]!.1 ∧ path[i]!.2 = path[i+1]!.2)) ∧
  (∀ i j, i < path.length → j < path.length → i ≠ j → path[i]! ≠ path[j]!)

/-- The main theorem stating the condition for a valid rook path -/
theorem rook_path_exists (board : Chessboard) :
  (∃ path : List (ℕ × ℕ), validRookPath board path) ↔ (Even board.m ∨ Even board.n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_path_exists_l1199_119954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_negative_sufficient_not_necessary_l1199_119926

def a (n : ℕ+) (lambda : ℝ) : ℝ := n.val^2 - 2*lambda*n.val

theorem lambda_negative_sufficient_not_necessary
  (lambda : ℝ)
  (h : lambda < 0) :
  (∀ n : ℕ+, a n lambda < a (n + 1) lambda) ∧
  ¬(∀ mu : ℝ, (∀ n : ℕ+, a n mu < a (n + 1) mu) → mu < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_negative_sufficient_not_necessary_l1199_119926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_rank_is_second_l1199_119928

-- Define the students
inductive Student : Type
  | A
  | B
  | C

-- Define a function to represent the score of each student
variable (score : Student → ℕ)

-- Define the statements made by each student
def statement_A (score : Student → ℕ) : Prop := ∀ s : Student, s ≠ Student.A → score Student.A > score s
def statement_B (score : Student → ℕ) : Prop := score Student.B > score Student.C
def statement_C (score : Student → ℕ) : Prop := ∀ s : Student, s ≠ Student.C → score s > score Student.C

-- Theorem to prove
theorem A_rank_is_second (score : Student → ℕ) :
  (∀ s₁ s₂ : Student, s₁ ≠ s₂ → score s₁ ≠ score s₂) →  -- Scores are different
  (statement_A score ∧ statement_B score ∧ ¬statement_C score) ∨          -- Only one statement is wrong
  (statement_A score ∧ ¬statement_B score ∧ statement_C score) ∨
  (¬statement_A score ∧ statement_B score ∧ statement_C score) →
  ∃ s : Student, s ≠ Student.A ∧ score s > score Student.A ∧
        ∀ s' : Student, s' ≠ Student.A → s' ≠ s → score Student.A > score s' :=
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_rank_is_second_l1199_119928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l1199_119993

/-- Triangle PQR with vertices P(-2, 2), Q(8, 2), and R(4, -4) has an area of 30 square units. -/
theorem triangle_PQR_area :
  let P : ℝ × ℝ := (-2, 2)
  let Q : ℝ × ℝ := (8, 2)
  let R : ℝ × ℝ := (4, -4)
  let triangle_area := abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)
  triangle_area = 30 := by
  -- Proof goes here
  sorry

#check triangle_PQR_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l1199_119993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1199_119996

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem intersection_M_N : 
  ∀ x : ℝ, x ∈ M ∩ N ↔ 1 < x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1199_119996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_participant_selection_ways_l1199_119908

/-- The number of ways to select k participants from n with at least d-1 empty seats between each pair --/
def number_of_selection_ways (n k d : ℕ) : ℚ :=
  (n : ℚ) / k * (Nat.choose (n - k * d + k - 1) (k - 1) : ℚ)

theorem participant_selection_ways
  (n k d : ℕ)
  (h1 : n ≥ 4)
  (h2 : k ≥ 2)
  (h3 : d ≥ 2)
  (h4 : k * d ≤ n) :
  number_of_selection_ways n k d = (n : ℚ) / k * (Nat.choose (n - k * d + k - 1) (k - 1) : ℚ) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_participant_selection_ways_l1199_119908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle6_measure_l1199_119990

-- Define the angle measures as real numbers
variable (angle1 angle2 angle5 angle6 : ℝ)

-- Define the parallel lines property
variable (m_parallel_n : Prop)

-- State the theorem
theorem angle6_measure :
  m_parallel_n ∧ 
  angle1 = (1/6) * angle2 ∧ 
  angle5 = angle1 ∧
  angle6 + angle5 = 180 →
  angle6 = 1080 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle6_measure_l1199_119990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_implies_a_bound_l1199_119906

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x - (log x) / x + a

theorem f_zero_implies_a_bound (a : ℝ) :
  (∃ x > 0, f a x = 0) → a ≤ (Real.exp 1)^2 + 1/(Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_implies_a_bound_l1199_119906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1199_119962

theorem product_remainder (a b c : ℕ) : 
  a % 7 = 2 → b % 7 = 3 → c % 7 = 5 → (a * b * c) % 7 = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1199_119962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_evaluation_integral_evaluation_l1199_119946

-- Problem 1
theorem complex_expression_evaluation :
  let i := Complex.I
  ((-1 + i) * i^100 + ((1 - i) / (1 + i))^5)^2017 - ((1 + i) / Real.sqrt 2)^20 = -2 * i := by sorry

-- Problem 2
theorem integral_evaluation :
  ∫ x in Set.Icc (-1 : ℝ) 1, (3 * Real.tan x + Real.sin x - 2 * x^3 + Real.sqrt (16 - (x - 1)^2)) = 4 * Real.pi / 3 + 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_evaluation_integral_evaluation_l1199_119946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_factorial_factor_l1199_119988

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem max_n_factorial_factor (n : ℕ) : 
  (∀ m : ℕ, m ≤ n → (factorial (factorial m) : ℕ).factorial ∣ (factorial 2021).factorial) ∧
  ¬((factorial (factorial (n + 1)) : ℕ).factorial ∣ (factorial 2021).factorial) →
  n = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_factorial_factor_l1199_119988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_log_representation_l1199_119970

-- Define a function that represents the logarithm of a negative number
def log_negative (x : ℝ) : Prop :=
  x < 0 → ∃ (y : ℝ), y > 0 ∧ Real.log y = Real.log (abs x)

-- State the theorem
theorem negative_log_representation :
  ∀ x : ℝ, x < 0 → log_negative x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_log_representation_l1199_119970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_600_actual_payment_160_discount_200_to_500_discount_500_or_more_two_day_shopping_total_discount_250_l1199_119945

-- Define the discount function
noncomputable def discount (x : ℝ) : ℝ :=
  if x < 200 then x
  else if x < 500 then 0.8 * x
  else 0.7 * x + 50

-- Theorem for a 600 yuan purchase
theorem discount_600 : discount 600 = 470 := by sorry

-- Theorem for 160 yuan actual payment
theorem actual_payment_160 (x : ℝ) :
  discount x = 160 → (x = 160 ∨ x = 200) := by sorry

-- Theorem for purchases between 200 and 500 yuan
theorem discount_200_to_500 (x : ℝ) :
  200 ≤ x → x < 500 → discount x = 0.8 * x := by sorry

-- Theorem for purchases of 500 yuan or more
theorem discount_500_or_more (x : ℝ) :
  x ≥ 500 → discount x = 0.7 * x + 50 := by sorry

-- Function for two-day shopping total
noncomputable def two_day_total (a : ℝ) : ℝ :=
  discount a + discount (900 - a)

-- Theorem for two-day shopping total
theorem two_day_shopping (a : ℝ) :
  200 < a → a < 300 → two_day_total a = 0.1 * a + 680 := by sorry

-- Theorem for total discount when a = 250
theorem total_discount_250 :
  900 - two_day_total 250 = 195 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_600_actual_payment_160_discount_200_to_500_discount_500_or_more_two_day_shopping_total_discount_250_l1199_119945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1199_119989

/-- A function that returns true if a number is a three-digit even number with the sum of tens and units digits equal to 12 -/
def isValidNumber (n : ℕ) : Bool :=
  100 ≤ n ∧ n ≤ 999 ∧  -- Three-digit number
  n % 2 = 0 ∧  -- Even number
  (n / 10 % 10 + n % 10 = 12)  -- Sum of tens and units digits is 12

/-- The count of valid numbers as defined by isValidNumber -/
def countValidNumbers : ℕ := (Finset.filter (fun n => isValidNumber n) (Finset.range 900)).card + 100

/-- Theorem stating that the count of valid numbers is 36 -/
theorem count_valid_numbers : countValidNumbers = 36 := by
  sorry

#eval countValidNumbers  -- This line will evaluate and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1199_119989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l1199_119979

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x + 3) / (2 * x - 4)

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem intersection_dot_product :
  ∀ (A B : ℝ × ℝ),
  (∃ (t : ℝ), A.1 = 2 + t ∧ A.2 = 1 + t ∧ A.2 = f A.1) →
  (∃ (s : ℝ), B.1 = 2 + s ∧ B.2 = 1 + s ∧ B.2 = f B.1) →
  ((A.1 - O.1, A.2 - O.2) + (B.1 - O.1, B.2 - O.2)) • (P.1 - O.1, P.2 - O.2) = 10 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l1199_119979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1199_119950

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + 2 * Real.sqrt 3 * (Real.cos x) ^ 2 - Real.sqrt 3

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m * Real.cos (2 * x - Real.pi / 6) - 2 * m + 3

theorem range_of_m (m : ℝ) :
  m > 0 →
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 (Real.pi / 4) ∧ x₂ ∈ Set.Icc 0 (Real.pi / 4) ∧ f x₁ = g m x₂) →
  m ∈ Set.Icc (2 / 3) 2 := by
  sorry

#check range_of_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1199_119950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_diagonal_length_l1199_119969

/-- A cyclic quadrilateral with integer side lengths --/
structure CyclicQuad where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  a_lt_10 : a < 10
  b_lt_10 : b < 10
  c_lt_10 : c < 10
  d_lt_10 : d < 10
  distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  bc_cd_eq_2ab_da : b * c = 2 * a * d

/-- The diagonal length of a cyclic quadrilateral --/
noncomputable def diagonalLength (q : CyclicQuad) : ℝ :=
  Real.sqrt ((q.a ^ 2 + q.b ^ 2 + q.c ^ 2 + q.d ^ 2) / 2)

/-- The theorem stating the largest possible diagonal length --/
theorem largest_diagonal_length :
    ∀ q : CyclicQuad, diagonalLength q ≤ 11 ∧ ∃ q' : CyclicQuad, diagonalLength q' = 11 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_diagonal_length_l1199_119969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_length_l1199_119924

theorem ladder_length (angle : Real) (adjacent : Real) (hypotenuse : Real) : 
  angle = 60 * Real.pi / 180 →
  adjacent = 4.6 →
  Real.cos angle = adjacent / hypotenuse →
  hypotenuse = 9.2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_length_l1199_119924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_existence_l1199_119957

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  a : Point2D
  b : Point2D
  c : Point2D

/-- Checks if a triangle is isosceles and right-angled -/
def isIsoscelesRight (t : Triangle) : Prop := sorry

/-- Rotates a point around another point by a given angle -/
noncomputable def rotatePoint (center : Point2D) (p : Point2D) (angle : ℝ) : Point2D := sorry

/-- Checks if a point is on a line segment between two other points -/
def isOnSegment (p : Point2D) (a : Point2D) (b : Point2D) : Prop := sorry

/-- Main theorem -/
theorem isosceles_right_triangle_existence 
  (abc : Triangle) 
  (ade : Triangle) 
  (h1 : isIsoscelesRight abc) 
  (h2 : isIsoscelesRight ade) 
  (h3 : abc.a = ade.a) -- A is the common point
  : ∀ θ : ℝ, ∃ m : Point2D, 
    let rotated_ade := Triangle.mk abc.a (rotatePoint abc.a ade.b θ) (rotatePoint abc.a ade.c θ)
    isOnSegment m rotated_ade.c abc.c ∧ 
    isIsoscelesRight (Triangle.mk abc.b m rotated_ade.b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_existence_l1199_119957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_A_B_l1199_119907

noncomputable def set_A : Set ℂ := {z : ℂ | z^5 = 32}
noncomputable def set_B : Set ℂ := {z : ℂ | z^3 - 16*z^2 - 32*z + 256 = 0}

noncomputable def max_distance (A B : Set ℂ) : ℝ :=
  ⨆ (a ∈ A) (b ∈ B), Complex.abs (a - b)

theorem max_distance_A_B :
  max_distance set_A set_B = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_A_B_l1199_119907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_length_165_to_45_l1199_119923

def arithmetic_sequence_length (first last step : ℤ) : ℕ :=
  Int.natAbs ((last - first) / step + 1)

theorem sequence_length_165_to_45 :
  arithmetic_sequence_length 165 45 (-3) = 41 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_length_165_to_45_l1199_119923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l1199_119921

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := -x^2 + 3 * Real.log x
def g (x : ℝ) : ℝ := x + 2

-- Define the points as functions
def P (a b : ℝ) : ℝ × ℝ := (a, b)
def Q (c d : ℝ) : ℝ × ℝ := (c, d)

-- State the theorem
theorem min_distance_squared (a b c d : ℝ) :
  (∃ x, P a b = (x, f x)) →  -- P lies on y = -x^2 + 3ln(x)
  (∃ x, Q c d = (x, g x)) →  -- Q lies on y = x + 2
  (∃ m, (a - c)^2 + (b - d)^2 ≥ m ∧ 
    ∀ a' b' c' d', (∃ x, (a', b') = (x, f x)) → 
                   (∃ x, (c', d') = (x, g x)) → 
                   (a' - c')^2 + (b' - d')^2 ≥ m) →
  (a - c)^2 + (b - d)^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l1199_119921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_last_five_digits_l1199_119913

/-- Given a positive integer M where M and M^2 end in the same sequence of five digits in base 10,
    and the first of these five digits is not zero, then the first four digits of this sequence
    form the number 2502. -/
theorem same_last_five_digits (M : ℕ) (hM : M > 0)
  (h1 : ∃ (a b c d e : ℕ), a ≠ 0 ∧ 
    M % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    (M * M) % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e) :
  M % 10000 = 2502 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_last_five_digits_l1199_119913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_l1199_119973

/-- A right triangle with an inscribed square -/
structure RightTriangleWithSquare where
  -- The lengths of the sides of the right triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The side length of the inscribed square
  s : ℝ
  -- Conditions
  right_triangle : a^2 + b^2 = c^2
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  s_positive : s > 0
  -- The square is inscribed in the triangle
  inscribed : s ≤ a ∧ s ≤ b

/-- The specific right triangle with an inscribed square from the problem -/
noncomputable def problemTriangle : RightTriangleWithSquare where
  a := 5
  b := 12
  c := 13
  s := 12/5
  right_triangle := by norm_num
  a_positive := by norm_num
  b_positive := by norm_num
  c_positive := by norm_num
  s_positive := by norm_num
  inscribed := by
    constructor
    · norm_num
    · norm_num

/-- The theorem stating that the side length of the inscribed square is 12/5 -/
theorem inscribed_square_side_length :
  problemTriangle.s = 12/5 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_l1199_119973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dana_has_largest_cookies_l1199_119975

-- Define the cookie shapes and their dimensions
def circle_radius : ℝ := 2
def square_side : ℝ := 3
def rectangle_length : ℝ := 4
def rectangle_width : ℝ := 2
def hexagon_side : ℝ := 1
def triangle_side : ℝ := 4

-- Define the areas of each cookie shape
noncomputable def circle_area : ℝ := Real.pi * circle_radius^2
def square_area : ℝ := square_side^2
def rectangle_area : ℝ := rectangle_length * rectangle_width
noncomputable def hexagon_area : ℝ := (3 * Real.sqrt 3 / 2) * hexagon_side^2
noncomputable def triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2

-- Theorem statement
theorem dana_has_largest_cookies :
  hexagon_area > circle_area ∧
  hexagon_area > square_area ∧
  hexagon_area > rectangle_area ∧
  hexagon_area > triangle_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dana_has_largest_cookies_l1199_119975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_meeting_time_l1199_119920

/-- The time it takes for two runners to meet on a circular track -/
noncomputable def meet_time (track_length : ℝ) (speed_bruce : ℝ) (speed_bhishma : ℝ) : ℝ :=
  track_length / (speed_bruce - speed_bhishma)

theorem first_meeting_time :
  let track_length : ℝ := 600
  let speed_bruce : ℝ := 30
  let speed_bhishma : ℝ := 20
  meet_time track_length speed_bruce speed_bhishma = 60 := by
  -- Unfold the definition of meet_time
  unfold meet_time
  -- Simplify the expression
  simp
  -- The proof is completed
  norm_num
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_meeting_time_l1199_119920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michaels_school_problem_l1199_119982

def michaels_school_students : ℕ := 2667

theorem michaels_school_problem (m n : ℕ) 
  (h1 : m = 5 * n)
  (h2 : m + n + 300 = 3500) :
  m = michaels_school_students := by
  have : n = 533 := by
    -- Proof that n = 533
    sorry
  have : m = 5 * 533 := by
    -- Proof that m = 5 * 533
    sorry
  -- Show that m = 2667
  sorry

#eval michaels_school_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_michaels_school_problem_l1199_119982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_equals_one_l1199_119963

/-- Given a 2x2 matrix B with real entries x and y, if B + B^(-1) = 0, then det B = 1 -/
theorem det_B_equals_one (x y : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, 3; -4, y]
  B + B⁻¹ = 0 → Matrix.det B = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_equals_one_l1199_119963
