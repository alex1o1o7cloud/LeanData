import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_length_approx_l679_67911

/-- Calculates the length of a goods train given the speeds of two trains traveling in opposite directions and the time taken for the goods train to pass a point on the other train. -/
noncomputable def goods_train_length (mans_train_speed goods_train_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := mans_train_speed + goods_train_speed
  let relative_speed_mps := relative_speed * (1000 / 3600)
  relative_speed_mps * passing_time

/-- Theorem stating that under the given conditions, the length of the goods train is approximately 410 meters. -/
theorem goods_train_length_approx :
  let mans_train_speed := (56 : ℝ)
  let goods_train_speed := (42.4 : ℝ)
  let passing_time := (15 : ℝ)
  ∃ ε > 0, |goods_train_length mans_train_speed goods_train_speed passing_time - 410| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_length_approx_l679_67911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enchilada_price_l679_67917

/-- The cost of a taco in dollars -/
def taco_cost : ℚ := sorry

/-- The cost of an enchilada in dollars -/
def enchilada_cost : ℚ := sorry

/-- The first order consists of 2 tacos and 3 enchiladas -/
def order1_cost : ℚ := 2 * taco_cost + 3 * enchilada_cost

/-- The second order consists of 3 tacos and 5 enchiladas -/
def order2_cost : ℚ := 3 * taco_cost + 5 * enchilada_cost

theorem enchilada_price :
  order1_cost = 39/5 ∧ order2_cost = 127/10 → enchilada_cost = 2 := by
  sorry

#check enchilada_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enchilada_price_l679_67917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_vertex_l679_67905

/-- Two parabolas are tangent if they intersect at exactly one point -/
noncomputable def are_tangent (f g : ℝ → ℝ) : Prop :=
  ∃! x, f x = g x

/-- The vertex of a parabola y = ax^2 + bx + c is at (-b/(2a), f(-b/(2a))) -/
noncomputable def vertex (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  (x, a * x^2 + b * x + c)

theorem parabola_tangent_vertex (b c : ℝ) :
  are_tangent (λ x => -x^2 + b*x + c) (λ x => x^2) →
  vertex (-1) b c = (b/2, b^2/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_vertex_l679_67905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_focus_coincides_l679_67992

/-- The conic section Γ -/
def Γ (m : ℝ) (x y : ℝ) : Prop := x^2 / m + y^2 / 5 = 1

/-- The parabola -/
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

/-- The focus of the parabola y² = 8x -/
def parabola_focus : ℝ × ℝ := (2, 0)

/-- A placeholder for the is_focus function, which we'll assume exists -/
def is_focus (Γ : ℝ → ℝ → ℝ → Prop) (m : ℝ) (f : ℝ × ℝ) : Prop := sorry

/-- Theorem: If one focus of Γ coincides with the focus of the parabola y² = 8x, then m = 9 -/
theorem conic_focus_coincides (m : ℝ) (hm1 : m ≠ 0) (hm2 : m ≠ 5) : 
  (∃ (f : ℝ × ℝ), f = parabola_focus ∧ is_focus Γ m f) → m = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_focus_coincides_l679_67992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_comparison_l679_67993

theorem power_comparison : (4 : ℝ)^(1/4) > (5 : ℝ)^(1/5) ∧ (5 : ℝ)^(1/5) > (16 : ℝ)^(1/16) ∧ (16 : ℝ)^(1/16) > (25 : ℝ)^(1/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_comparison_l679_67993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_l679_67963

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- The set of eight spheres, one in each octant -/
def octantSpheres : Set Sphere :=
  { s | s.radius = 2 ∧ 
        (∃ (x y z : ℝ), s.center = (x, y, z) ∧ 
                         (x = 2 ∨ x = -2) ∧ 
                         (y = 2 ∨ y = -2) ∧ 
                         (z = 2 ∨ z = -2)) }

/-- The smallest sphere centered at the origin that contains all octantSpheres -/
noncomputable def enclosingSphere : Sphere :=
  { center := (0, 0, 0),
    radius := 2 * Real.sqrt 3 + 2 }

/-- Theorem stating that the enclosingSphere is the smallest sphere that contains all octantSpheres -/
theorem smallest_enclosing_sphere :
  ∀ s ∈ octantSpheres, (∀ p, ‖p - s.center‖ ≤ s.radius → ‖p‖ ≤ enclosingSphere.radius) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_l679_67963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_phi_l679_67947

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

noncomputable def g (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3 - 2 * φ)

theorem smallest_positive_phi :
  ∃ φ : ℝ, φ > 0 ∧
  (∀ x : ℝ, g x φ = g (-x) φ) ∧
  (∀ ψ : ℝ, ψ > 0 ∧ (∀ x : ℝ, g x ψ = g (-x) ψ) → φ ≤ ψ) ∧
  φ = 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_phi_l679_67947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l679_67956

noncomputable def f (x : ℝ) : ℝ := (Real.sin x - Real.cos x) * Real.sin (2 * x) / Real.sin x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def strictly_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

theorem f_properties :
  (∀ x : ℝ, f x ≠ 0 ↔ x ∉ {y | ∃ k : ℤ, y = k * Real.pi}) ∧
  (∃ p : ℝ, p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q → is_periodic f q → p ≤ q) ∧
  (∀ k : ℤ, strictly_increasing_on f (Set.Icc (k * Real.pi - Real.pi / 8) (k * Real.pi) ∪
                                       Set.Ioo (k * Real.pi) (k * Real.pi + 3 * Real.pi / 8))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l679_67956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_monotonicity_l679_67914

noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + 1

theorem function_and_monotonicity :
  ∃ b c : ℝ,
    (f b c 0 = 1) ∧
    (∃ m : ℝ, 2 = 3*(1^2) + 2*b*1 + c ∧ 5 = b + c + 2) →
    (f b c = λ x ↦ x^3 + 4*x^2 - 9*x + 1) ∧
    (∀ x, x < (-4 - Real.sqrt 41) / 3 → (deriv (f b c)) x > 0) ∧
    (∀ x, (-4 - Real.sqrt 41) / 3 < x ∧ x < (-4 + Real.sqrt 41) / 3 → (deriv (f b c)) x < 0) ∧
    (∀ x, x > (-4 + Real.sqrt 41) / 3 → (deriv (f b c)) x > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_monotonicity_l679_67914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_triangle_l679_67966

structure Grid :=
  (size : ℕ)
  (nodes : Finset (ℕ × ℕ))
  (square_area : ℝ)

def is_collinear (p1 p2 p3 : ℕ × ℕ) : Prop := sorry

noncomputable def triangle_area (p1 p2 p3 : ℕ × ℕ) : ℝ := sorry

def valid_selection (g : Grid) (s : Finset (ℕ × ℕ)) : Prop :=
  s ⊆ g.nodes ∧ 
  s.card = 6 ∧ 
  ∀ p1 p2 p3, p1 ∈ s → p2 ∈ s → p3 ∈ s → p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬is_collinear p1 p2 p3

theorem exists_small_triangle (g : Grid) (s : Finset (ℕ × ℕ)) :
  g.size = 4 ∧ 
  g.nodes.card = 25 ∧ 
  g.square_area = 1 ∧
  valid_selection g s →
  ∃ p1 p2 p3, p1 ∈ s ∧ p2 ∈ s ∧ p3 ∈ s ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ triangle_area p1 p2 p3 ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_triangle_l679_67966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_identity_l679_67903

-- Define Chebyshev polynomials of the first kind
def T : ℕ → (ℝ → ℝ)
| 0 => λ z => 1
| 1 => λ z => z
| (n+2) => λ z => 2*z*(T (n+1) z) - T n z

-- Define Chebyshev polynomials of the second kind
def U : ℕ → (ℝ → ℝ)
| 0 => λ z => 1
| 1 => λ z => 2*z
| (n+2) => λ z => 2*z*(U (n+1) z) - U n z

-- Theorem statement
theorem chebyshev_identity (n : ℕ) (x : ℝ) :
  (Real.cos (n * x) = T n (Real.cos x)) ∧
  (Real.sin (n * x) = Real.sin x * U (n - 1) (Real.cos x)) := by
  sorry

-- Compute the polynomials for n = 0, 1, 2, 3, 4, 5
example : 
  (T 0 = λ z => 1) ∧
  (T 1 = λ z => z) ∧
  (T 2 = λ z => 2*z^2 - 1) ∧
  (T 3 = λ z => 4*z^3 - 3*z) ∧
  (T 4 = λ z => 8*z^4 - 8*z^2 + 1) ∧
  (T 5 = λ z => 16*z^5 - 20*z^3 + 5*z) ∧
  (U 0 = λ z => 1) ∧
  (U 1 = λ z => 2*z) ∧
  (U 2 = λ z => 4*z^2 - 1) ∧
  (U 3 = λ z => 8*z^3 - 4*z) ∧
  (U 4 = λ z => 16*z^4 - 12*z^2 + 1) ∧
  (U 5 = λ z => 32*z^5 - 32*z^3 + 6*z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_identity_l679_67903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l679_67939

noncomputable def triangle_area (r₁ r₂ : ℝ) (θ₁ θ₂ : ℝ) : ℝ :=
  (1/2) * r₁ * r₂ * Real.sin (θ₂ - θ₁)

noncomputable def deg_to_rad (deg : ℝ) : ℝ :=
  deg * (Real.pi / 180)

theorem triangle_area_specific : 
  triangle_area 5 4 (deg_to_rad 109) (deg_to_rad 49) = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l679_67939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l679_67941

noncomputable section

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := (a * Real.log x) / x + b

-- Define the function g(x)
def g (c x : ℝ) : ℝ := Real.log x / Real.log c - x

-- State the theorem
theorem problem_solution :
  ∀ (a b : ℝ),
  (∀ x : ℝ, x > 0 → DifferentiableAt ℝ (f a b) x) →
  (∃ x : ℝ, x > 0 ∧ deriv (f a b) x = 1) →
  f a b 1 = 0 →
  (a = 1 ∧ b = 0) ∧
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f 1 0 x ≥ f 1 0 y) ∧
  f 1 0 (Real.exp 1) = 1 / Real.exp 1 ∧
  (∀ c : ℝ, c > 0 → c ≠ 1 → (∃ x : ℝ, x > 0 ∧ g c x = 0) →
    c ≤ Real.exp (1 / Real.exp 1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l679_67941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_value_l679_67977

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = sin x + m - 1 -/
noncomputable def f (m : ℝ) : ℝ → ℝ := fun x ↦ Real.sin x + m - 1

theorem odd_function_m_value :
  ∀ m : ℝ, IsOdd (f m) → m = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_value_l679_67977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercept_point_l679_67995

noncomputable section

/-- The slope of the line -/
def m : ℝ := -5/3

/-- The y-intercept of the line -/
def b : ℝ := 15

/-- The x-coordinate of point P (x-intercept) -/
noncomputable def p_x : ℝ := b / (-m)

/-- The y-coordinate of point Q (y-intercept) -/
def q_y : ℝ := b

/-- The coordinates of point T on the line segment PQ -/
structure Point where
  x : ℝ
  y : ℝ

/-- T is on the line -/
def on_line (t : Point) : Prop := t.y = m * t.x + b

/-- T is between P and Q -/
def between_p_and_q (t : Point) : Prop := 0 ≤ t.x ∧ t.x ≤ p_x

/-- The area of triangle POQ -/
noncomputable def area_poq : ℝ := (1/2) * p_x * q_y

/-- The area of triangle TOP -/
noncomputable def area_top (t : Point) : ℝ := (1/2) * t.x * t.y

theorem line_intercept_point (t : Point) 
  (h1 : on_line t) 
  (h2 : between_p_and_q t) 
  (h3 : area_poq = 4 * area_top t) : 
  t.x + t.y = 10.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercept_point_l679_67995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_l679_67951

-- Define the period of a sinusoidal function
noncomputable def period (ω : ℝ) : ℝ := 2 * Real.pi / ω

-- Define the given functions
noncomputable def f₁ (x : ℝ) : ℝ := Real.sin (x / 2)
noncomputable def f₂ (x : ℝ) : ℝ := Real.sin x
noncomputable def f₃ (x : ℝ) : ℝ := Real.sin (2 * x)
noncomputable def f₄ (x : ℝ) : ℝ := Real.sin (4 * x)

-- Theorem statement
theorem smallest_period :
  period 4 = Real.pi / 2 ∧
  period 4 < period 2 ∧
  period 4 < period 1 ∧
  period 4 < period (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_l679_67951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exercise_enthusiasm_gender_relationship_l679_67922

/-- Represents the exercise time intervals --/
inductive ExerciseInterval
| ZeroToThree
| ThreeToSix
| SixToNine
| NineToTwelve
| TwelveToFifteen

/-- Represents gender --/
inductive Gender
| Boy
| Girl

/-- Represents enthusiasm level --/
inductive EnthusiasmLevel
| Enthusiast
| NonEnthusiast

/-- Sample data for boys --/
def boySample : List (ExerciseInterval × Nat) :=
  [(ExerciseInterval.ZeroToThree, 3),
   (ExerciseInterval.ThreeToSix, 9),
   (ExerciseInterval.SixToNine, 8),
   (ExerciseInterval.NineToTwelve, 12),
   (ExerciseInterval.TwelveToFifteen, 4)]

/-- Sample data for girls --/
def girlSample : List (ExerciseInterval × Nat) :=
  [(ExerciseInterval.ZeroToThree, 6),
   (ExerciseInterval.ThreeToSix, 10),
   (ExerciseInterval.SixToNine, 5),
   (ExerciseInterval.NineToTwelve, 2),
   (ExerciseInterval.TwelveToFifteen, 1)]

/-- Total number of students in the sample --/
def totalSample : Nat := 60

/-- Critical value for 0.025 significance level --/
def criticalValue : ℝ := 5.024

/-- Function to calculate K^2 --/
noncomputable def calculateKSquared (a b c d : Nat) : ℝ :=
  let n : ℝ := (a + b + c + d : ℝ)
  (n * (a * d - b * c)^2 : ℝ) / ((a + b : ℝ) * (c + d : ℝ) * (a + c : ℝ) * (b + d : ℝ))

/-- Theorem stating that the relationship between exercise enthusiasm and gender is significant --/
theorem exercise_enthusiasm_gender_relationship :
  ∃ (a b c d : Nat),
    a + b + c + d = totalSample ∧
    calculateKSquared a b c d > criticalValue := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exercise_enthusiasm_gender_relationship_l679_67922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_doubled_radius_l679_67926

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem sphere_volume_doubled_radius (r : ℝ) (h : r > 0) : 
  sphere_volume (2 * r) = 8 * sphere_volume r := by
  unfold sphere_volume
  simp [Real.pi]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_doubled_radius_l679_67926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_star_15_equals_5_l679_67958

noncomputable def star (a b : ℝ) : ℝ := (Real.sqrt (a^2 + b)) / (Real.sqrt (a^2 - b))

theorem y_star_15_equals_5 :
  ∃ y : ℝ, star y 15 = 5 ∧ y = Real.sqrt 65 / 2 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_star_15_equals_5_l679_67958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_with_obtuse_angle_a_less_than_half_l679_67918

/-- A line with equation y = mx + b has an obtuse angle of inclination if its slope m < 0 -/
def has_obtuse_angle_of_inclination (m : ℝ) : Prop := m < 0

/-- The slope of a line y = (2a-1)x + 2 -/
def line_slope (a : ℝ) : ℝ := 2*a - 1

theorem line_with_obtuse_angle_a_less_than_half (a : ℝ) :
  has_obtuse_angle_of_inclination (line_slope a) → a < 1/2 := by
  intro h
  unfold has_obtuse_angle_of_inclination at h
  unfold line_slope at h
  linarith

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_with_obtuse_angle_a_less_than_half_l679_67918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_scaling_l679_67987

/-- Represents the recipe and scaling for cookie production -/
structure CookieRecipe where
  initialFlour : ℚ
  initialSugar : ℚ
  initialCookies : ℚ
  newFlour : ℚ

/-- Calculates the number of cookies that can be made with a new amount of flour -/
noncomputable def cookiesWithNewFlour (recipe : CookieRecipe) : ℚ :=
  (recipe.initialCookies / recipe.initialFlour) * recipe.newFlour

/-- Calculates the amount of sugar needed for a new amount of flour -/
noncomputable def sugarForNewFlour (recipe : CookieRecipe) : ℚ :=
  (recipe.initialSugar / recipe.initialFlour) * recipe.newFlour

/-- Theorem stating the correct number of cookies and amount of sugar for the new recipe -/
theorem cookie_scaling (recipe : CookieRecipe) 
    (h1 : recipe.initialFlour = 3)
    (h2 : recipe.initialSugar = 3/2)
    (h3 : recipe.initialCookies = 24)
    (h4 : recipe.newFlour = 5) :
    cookiesWithNewFlour recipe = 40 ∧ sugarForNewFlour recipe = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_scaling_l679_67987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_k_l679_67929

/-- An arithmetic sequence with common difference d and first term 4d -/
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 4 * d + (n - 1 : ℝ) * d

/-- The geometric mean of two real numbers -/
noncomputable def geometric_mean (x y : ℝ) : ℝ := Real.sqrt (x * y)

theorem arithmetic_sequence_k (d : ℝ) (k : ℕ) (h : d ≠ 0) :
  geometric_mean (arithmetic_sequence d 1) (arithmetic_sequence d 6) = arithmetic_sequence d k →
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_k_l679_67929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_for_increasing_sequence_l679_67913

/-- Definition of the sequence a_n -/
noncomputable def a (n : ℕ) (t : ℝ) : ℝ := (2 * n + t^2 - 8) / (n + t)

/-- The sequence {a_n} is increasing -/
def is_increasing (t : ℝ) : Prop :=
  ∀ n : ℕ, a n t < a (n + 1) t

/-- The main theorem: the range of t for which {a_n} is increasing -/
theorem range_of_t_for_increasing_sequence :
  {t : ℝ | is_increasing t} = Set.Ioo (-1 : ℝ) 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_for_increasing_sequence_l679_67913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l679_67955

-- Define the function f
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^α

-- State the theorem
theorem alpha_value (α : ℝ) : 
  (∀ x, deriv (f α) x = α * x^(α - 1)) → 
  deriv (f α) (-1) = -4 → 
  α = 4 := by
  intro h1 h2
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l679_67955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_is_ten_years_l679_67910

/-- Represents the simple interest scenario -/
structure SimpleInterest where
  principal : ℚ
  rate : ℚ
  time : ℚ

/-- Calculates the interest for a given simple interest scenario -/
def calculateInterest (si : SimpleInterest) : ℚ :=
  si.principal * si.rate * si.time / 100

/-- Theorem stating the condition for the time period to be 10 years -/
theorem time_is_ten_years 
  (si : SimpleInterest)
  (h1 : si.principal = 600)
  (h2 : calculateInterest { principal := si.principal, 
                            rate := si.rate + 5, 
                            time := si.time } - 
        calculateInterest si = 300) :
  si.time = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_is_ten_years_l679_67910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_result_is_10000_l679_67969

/-- A circle intersecting the y-axis at two points and tangent to a specific line -/
structure TangentCircle where
  -- The y-coordinates of the intersection points with the y-axis
  a : ℝ
  b : ℝ
  -- The circle is tangent to the line x + 100y = 100 at (100, 0)
  tangent_point : ℝ × ℝ := (100, 0)
  tangent_line : Set (ℝ × ℝ) := {p | p.1 + 100 * p.2 = 100}

/-- The result of ab - a - b for the TangentCircle -/
def circle_result (c : TangentCircle) : ℝ :=
  c.a * c.b - c.a - c.b

/-- Theorem stating that the circle_result is always 10000 -/
theorem circle_result_is_10000 (c : TangentCircle) : circle_result c = 10000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_result_is_10000_l679_67969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_product_l679_67952

/-- Given two integers p and q in base 10, where
    p = 33331111 and q = 77772222, the sum of digits
    of their product in base 10 is 66. -/
theorem sum_of_digits_of_product : ∃ (p q : ℕ), 
  p = 33331111 ∧ 
  q = 77772222 ∧ 
  (Nat.digits 10 (p * q)).sum = 66 := by
  use 33331111, 77772222
  constructor
  · rfl
  constructor
  · rfl
  · sorry  -- The actual computation is left as an exercise


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_product_l679_67952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_distance_squared_l679_67948

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter, orthocenter, and circumradius
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry
noncomputable def circumradius (t : Triangle) : ℝ := sorry

-- Define the side lengths
noncomputable def side_lengths (t : Triangle) : ℝ × ℝ × ℝ := sorry

-- Theorem statement
theorem orthocenter_circumcenter_distance_squared (t : Triangle) :
  let O := circumcenter t
  let H := orthocenter t
  let R := circumradius t
  let (a, b, c) := side_lengths t
  R = 5 ∧ a^2 + b^2 + c^2 = 50 →
  (O.1 - H.1)^2 + (O.2 - H.2)^2 = 175 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_distance_squared_l679_67948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l679_67980

noncomputable def z : ℂ := (4 - 3*Complex.I) / (3 + 4*Complex.I) + 2

theorem z_in_fourth_quadrant : Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l679_67980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l679_67945

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log_equation_solution :
  ∀ x : ℝ, x > 0 → (f (x + 1) + f (x - 3) = 1 ↔ x = 4) :=
by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l679_67945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_difference_l679_67996

theorem cube_root_equation_solution_difference : ∃ (x y : ℝ),
  ((9 - x^2 / 4)^(1/3 : ℝ) = -3) ∧
  ((9 - y^2 / 4)^(1/3 : ℝ) = -3) ∧
  x ≠ y ∧
  |x - y| = 24 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_difference_l679_67996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_solution_base_solution_composition_l679_67973

/-- Represents the components of the cleaning solution --/
structure CleaningSolution where
  chemicalA : ℚ
  water : ℚ
  chemicalB : ℚ
  baseSolution : ℚ

/-- Calculates the required amounts of water and chemical B for the cleaning solution --/
noncomputable def calculateSolution (baseSolutionTotal : ℚ) (chemicalBRate : ℚ) : CleaningSolution :=
  let waterProportion := 2 / 6
  let waterAmount := baseSolutionTotal * waterProportion
  let chemicalBAmount := baseSolutionTotal * chemicalBRate
  { chemicalA := 0, water := waterAmount, chemicalB := chemicalBAmount, baseSolution := baseSolutionTotal }

/-- Theorem stating the correct amounts of water and chemical B needed --/
theorem correct_solution :
  let solution := calculateSolution (72 / 100) (1 / 100)
  solution.water = 24 / 100 ∧ solution.chemicalB = 72 / 10000 := by
  sorry

/-- Verifies the base solution composition --/
theorem base_solution_composition :
  4 / 100 + 2 / 100 = 6 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_solution_base_solution_composition_l679_67973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_zero_solution_l679_67975

theorem inverse_function_zero_solution
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (f : ℝ → ℝ) (hf : ∀ x, f x = 1 / (a * x + b))
  (f_inv : ℝ → ℝ) (hf_inv : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f) :
  f_inv 0 = 1 / b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_zero_solution_l679_67975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_intersection_l679_67936

/-- A parabola in the Cartesian coordinate system -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- A moving line in the Cartesian coordinate system -/
structure MovingLine where
  m : ℝ

/-- The intersection points of a parabola and a moving line -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Point P where y = x intersects the parabola -/
def P : ℝ × ℝ := (2, 2)

/-- Point D where the tangent line at P intersects the moving line -/
def D (l : MovingLine) : ℝ × ℝ := (l.m + 2, 2 * l.m + 2)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The value of t as defined in the problem -/
noncomputable def t (l : MovingLine) (pts : IntersectionPoints) : ℝ :=
  (distance P (D l))^2 / (distance (D l) pts.A * distance (D l) pts.B)

theorem parabola_and_line_intersection
  (C : Parabola) (l : MovingLine) (pts : IntersectionPoints) :
  (pts.A.1 * pts.A.2 + pts.B.1 * pts.B.2 = l.m^2 - 2 * l.m) →
  (C.p = 1 ∧ t l pts = 5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_intersection_l679_67936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_g_inequality_range_l679_67961

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def g (a x : ℝ) : ℝ := -x^2 + a*x - 3

-- Theorem for the minimum value of f
theorem f_min_value : ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = -1 / Real.exp 1 := by
  sorry

-- Theorem for the range of a
theorem g_inequality_range (a : ℝ) : 
  (∀ (x : ℝ), x > 0 → 2 * f x ≥ g a x) ↔ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_g_inequality_range_l679_67961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l679_67900

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 + (2^x - 1)/(2^x + 1) + Real.sin (2*x)

-- State the theorem
theorem range_sum (k : ℝ) (hk : k > 0) :
  ∃ (m n : ℝ), (∀ x ∈ Set.Icc (-k) k, m ≤ f x ∧ f x ≤ n) ∧
  (∀ y : ℝ, (∃ x ∈ Set.Icc (-k) k, f x = y) → m ≤ y ∧ y ≤ n) →
  m + n = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l679_67900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_problem_l679_67908

-- Define the speed of the car
variable (v : ℝ)

-- Define the time it takes to travel 1 km at 60 km/hr
noncomputable def time_at_60 : ℝ := 1 / 60 * 3600

-- Define the time it takes to travel 1 km at v km/hr
noncomputable def time_at_v : ℝ := 1 / v * 3600

-- Theorem statement
theorem car_speed_problem :
  time_at_v v = time_at_60 + 5 ↔ v = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_problem_l679_67908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_unit_square_l679_67925

/-- The probability that a randomly chosen point in the unit square has a y-coordinate
    greater than or equal to its x-coordinate minus 1/4 is 9/32. -/
theorem probability_in_unit_square : 
  ∃ (p : Set (ℝ × ℝ)) (μ : MeasureTheory.Measure (ℝ × ℝ)),
    (∀ x y, (x, y) ∈ p ↔ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1) ∧
    (μ p = 1) ∧
    (μ {q : ℝ × ℝ | q ∈ p ∧ q.2 ≥ q.1 - 1/4} = 9/32) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_unit_square_l679_67925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l679_67994

theorem negation_of_cosine_inequality :
  (¬ (∀ x : ℝ, Real.cos x ≤ 1)) ↔ (∃ x : ℝ, Real.cos x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l679_67994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_shaded_is_correct_l679_67902

/-- Represents a rectangle in the grid --/
structure Rectangle where
  left : Nat
  right : Nat
  top : Nat
  bottom : Nat

/-- The dimensions of the entire grid --/
def gridWidth : Nat := 2004
def gridHeight : Nat := 3

/-- The positions of the shaded squares in each row --/
def shadedLeft : Nat := 1002
def shadedRight : Nat := 1003

/-- Checks if a rectangle contains a shaded square --/
def containsShaded (r : Rectangle) : Prop :=
  (r.left ≤ shadedRight && shadedLeft ≤ r.right) && 
  (r.top ≤ gridHeight && r.bottom ≥ 1)

/-- The total number of possible rectangles --/
def totalRectangles : Nat := gridHeight * (gridWidth.choose 2)

/-- The number of rectangles containing at least one shaded square --/
def shadedRectangles : Nat := gridHeight * shadedLeft * shadedLeft

/-- The probability of choosing a rectangle that does not contain a shaded square --/
noncomputable def probabilityNoShaded : Real :=
  1 - (shadedRectangles : Real) / (totalRectangles : Real)

/-- The main theorem stating the probability --/
theorem probability_no_shaded_is_correct :
  probabilityNoShaded = 0.25 / 1002.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_shaded_is_correct_l679_67902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bouquet_combinations_l679_67953

theorem bouquet_combinations : 
  let total_cost : ℕ := 60
  let rose_cost : ℕ := 4
  let carnation_cost : ℕ := 3
  (Finset.filter (λ pair : ℕ × ℕ => rose_cost * pair.1 + carnation_cost * pair.2 = total_cost) 
    (Finset.product (Finset.range (total_cost / rose_cost + 1)) (Finset.range (total_cost / carnation_cost + 1)))).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bouquet_combinations_l679_67953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_is_21_hours_l679_67960

/-- Represents the time it takes for a pump to fill a tank without a leak -/
def pump_fill_time : ℚ := 3

/-- Represents the time it takes for a pump to fill a tank with a leak present -/
def pump_fill_time_with_leak : ℚ := 7/2

/-- Calculates the time it takes for the leak to empty a full tank -/
def leak_empty_time (pump_fill_time : ℚ) (pump_fill_time_with_leak : ℚ) : ℚ :=
  (pump_fill_time * pump_fill_time_with_leak) / (pump_fill_time_with_leak - pump_fill_time)

theorem leak_empty_time_is_21_hours :
  leak_empty_time pump_fill_time pump_fill_time_with_leak = 21 := by
  unfold leak_empty_time pump_fill_time pump_fill_time_with_leak
  -- The proof steps would go here
  sorry

#eval leak_empty_time pump_fill_time pump_fill_time_with_leak

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_is_21_hours_l679_67960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_radius_l679_67972

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 36 = 6*x + 24*y

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, 12)

-- Define the radius of the circle
noncomputable def circle_radius : ℝ := 3 * Real.sqrt 13

-- Theorem statement
theorem cookie_radius :
  ∀ x y : ℝ, circle_equation x y ↔ 
    (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_radius_l679_67972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_670154500_matches_l679_67968

/-- Function to get the digit at a specific place value in a number -/
def getDigitAtPlace (n : ℕ) (place : ℕ) : ℕ :=
  (n / (10 ^ place)) % 10

/-- The set of numbers to check -/
def numbers : List ℕ := [17003500, 175003050, 13007500, 670154500]

/-- Theorem stating that only 670154500 has 7 in ten-millions place and 5 in hundreds place -/
theorem only_670154500_matches : ∃! n, n ∈ numbers ∧ getDigitAtPlace n 7 = 7 ∧ getDigitAtPlace n 2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_670154500_matches_l679_67968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_2_equals_neg_3_l679_67934

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2*x + 1/x else x^2 - 3*x + 1

-- Theorem statement
theorem f_f_2_equals_neg_3 : f (f 2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_2_equals_neg_3_l679_67934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2018_l679_67954

def sequence_a : ℕ → ℚ
  | 0 => 4/5  -- Add a case for 0 to avoid the "missing cases" error
  | 1 => 4/5
  | n + 1 => 
      let a_n := sequence_a n
      if 0 ≤ a_n ∧ a_n ≤ 1/2 then 2 * a_n
      else if 1/2 < a_n ∧ a_n ≤ 1 then 2 * a_n - 1
      else a_n  -- This case should never occur based on the problem definition

theorem sequence_a_2018 : sequence_a 2018 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2018_l679_67954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_centrally_symmetric_not_necessarily_axially_symmetric_l679_67989

/-- A shape is centrally symmetric if for every point on the shape, 
    there is another point directly opposite to it through the center. -/
def CentrallySymmetric (Shape : Type) : Prop := sorry

/-- A shape is axially symmetric if it can be divided into two identical halves by a line. -/
def AxiallySymmetric (Shape : Type) : Prop := sorry

/-- Definition of a parallelogram -/
structure Parallelogram where
  -- We'll leave the internal structure empty for this example
  mk :: -- Constructor

/-- Theorem: A parallelogram is centrally symmetric but not necessarily axially symmetric -/
theorem parallelogram_centrally_symmetric_not_necessarily_axially_symmetric :
  CentrallySymmetric Parallelogram ∧ 
  ¬(∀ p : Parallelogram, AxiallySymmetric Parallelogram) :=
by
  sorry  -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_centrally_symmetric_not_necessarily_axially_symmetric_l679_67989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_roses_l679_67924

theorem initial_roses (initial_orchids final_orchids final_roses initial_roses : ℕ) 
  (h1 : initial_orchids = 84)
  (h2 : final_orchids = 91)
  (h3 : final_roses = 14)
  (h4 : final_orchids - initial_orchids = final_roses - initial_roses) :
  initial_roses = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_roses_l679_67924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_5_expression_l679_67998

def f (x : ℝ) : ℝ := 2 * x + 1

def f_n : ℕ → (ℝ → ℝ)
  | 0 => λ x => x  -- Base case for n = 0
  | 1 => f
  | n + 1 => λ x => f (f_n n x)

theorem f_5_expression (x : ℝ) : f_n 5 x = 32 * x + 31 := by
  -- Expand the definition of f_n for n = 5
  have h1 : f_n 5 x = f (f_n 4 x) := rfl
  have h2 : f_n 4 x = f (f_n 3 x) := rfl
  have h3 : f_n 3 x = f (f_n 2 x) := rfl
  have h4 : f_n 2 x = f (f_n 1 x) := rfl
  have h5 : f_n 1 x = f x := rfl

  -- Substitute and simplify
  simp [h1, h2, h3, h4, h5, f]
  ring  -- Simplify the algebraic expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_5_expression_l679_67998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a2_plus_b2_l679_67990

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := 2 * a * Real.sqrt x + b - Real.exp (x / 2)

-- State the theorem
theorem min_value_of_a2_plus_b2 (a b x₀ : ℝ) 
  (h1 : f a b x₀ = 0)  -- x₀ is a zero of f
  (h2 : 1/4 ≤ x₀ ∧ x₀ ≤ Real.exp 1)  -- x₀ ∈ [1/4, e]
  : a^2 + b^2 ≥ Real.exp (3/4) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a2_plus_b2_l679_67990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_square_area_theorem_l679_67965

/-- Triangle ABC with angle A = 135/2 degrees, BC = 15, and inscribed square WXYZ -/
structure TriangleWithSquare where
  /-- Side length of BC -/
  bc : ℝ
  /-- Angle A in radians -/
  angleA : ℝ
  /-- Side length of square WXYZ -/
  s : ℝ
  /-- W is on AB, X is on AC, Z is on BC -/
  squareInscribed : Bool
  /-- Triangle ZBW is similar to triangle ABC -/
  trianglesSimilar : Bool
  /-- WZ is not parallel to AC -/
  notParallel : Bool

/-- The maximum area of the inscribed square WXYZ -/
noncomputable def maxSquareArea (t : TriangleWithSquare) : ℝ :=
  (225 * Real.sqrt 2) / 8

/-- Theorem stating the maximum area of the inscribed square -/
theorem max_square_area_theorem (t : TriangleWithSquare) 
  (h1 : t.bc = 15)
  (h2 : t.angleA = 135 * Real.pi / 360)
  (h3 : t.squareInscribed)
  (h4 : t.trianglesSimilar)
  (h5 : t.notParallel) :
  t.s ^ 2 ≤ maxSquareArea t := by
  sorry

#check max_square_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_square_area_theorem_l679_67965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_decreasing_l679_67985

-- Define the circle
def circleEquation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the rotation speed
def rotationPeriod : ℝ := 12

-- Define the initial position
noncomputable def initialX : ℝ := 1/2
noncomputable def initialY : ℝ := Real.sqrt 3 / 2

-- Define the position of point A at time t
noncomputable def positionA (t : ℝ) : ℝ × ℝ := sorry

-- Define monotonically decreasing function on an interval
def monotonicDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem x_coordinate_decreasing :
  (monotonicDecreasing (λ t => (positionA t).1) 0 4) ∧
  (monotonicDecreasing (λ t => (positionA t).1) 10 12) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_decreasing_l679_67985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_approx_l679_67999

/-- Calculates the balance of an account with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Calculates the balance of an account with simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

/-- The positive difference between compound and simple interest balances -/
theorem interest_difference_approx (angela_principal : ℝ) (bob_principal : ℝ) 
    (angela_rate : ℝ) (bob_rate : ℝ) (time : ℕ) :
  angela_principal = 12000 →
  bob_principal = 15000 →
  angela_rate = 0.05 →
  bob_rate = 0.08 →
  time = 25 →
  |simple_interest bob_principal bob_rate time - compound_interest angela_principal angela_rate time - 4363| < 1 := by
  sorry

#check interest_difference_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_approx_l679_67999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_b_is_127_l679_67981

def product_sequence (a b : ℕ) : ℚ :=
  (Finset.range (a - 2)).prod (λ i => (i + 3 : ℚ) / (i + 2))

theorem sum_a_b_is_127 (a b : ℕ) (h : product_sequence a b = 32) : a + b = 127 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_b_is_127_l679_67981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_none_l679_67964

def correct_answer : String := "None"

theorem correct_answer_is_none : correct_answer = "None" := by
  rfl

#check correct_answer_is_none

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_none_l679_67964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_approx_7_2_l679_67938

/-- The height of water in two containers -/
noncomputable def water_height (rectangular_width rectangular_length cylinder_radius total_volume : ℝ) : ℝ :=
  total_volume / (rectangular_width * rectangular_length + Real.pi * cylinder_radius^2)

/-- Proof that the water height is approximately 7.2 cm -/
theorem water_height_approx_7_2 :
  let h := water_height 2 4 1 80
  ∃ ε > 0, abs (h - 7.2) < ε ∧ ε < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_approx_7_2_l679_67938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_existence_l679_67923

theorem divisor_existence (S : Finset ℕ) :
  S.card = 1008 →
  (∀ n, n ∈ S → 1 ≤ n ∧ n ≤ 2014) →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_existence_l679_67923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l679_67928

theorem geometric_series_sum : ∀ (a r : ℝ) (n : ℕ),
  a > 0 → r > 1 →
  a * r^(n-1) = 6561 →
  (Finset.range n).sum (λ i => a * r^i) = 9841 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l679_67928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_sufficient_for_not_cheap_l679_67901

-- Define the universe of goods
variable (Goods : Type)

-- Define the properties
variable (good : Goods → Prop)
variable (cheap : Goods → Prop)

-- Define the statement "Good goods are not cheap"
def good_not_cheap (Goods : Type) (good : Goods → Prop) (cheap : Goods → Prop) : Prop :=
  ∀ x : Goods, good x → ¬(cheap x)

-- Theorem: If "Good goods are not cheap" holds, then "good goods" is a sufficient condition for "not cheap"
theorem good_sufficient_for_not_cheap 
  (Goods : Type) (good : Goods → Prop) (cheap : Goods → Prop)
  (h : good_not_cheap Goods good cheap) : 
  ∀ x : Goods, good x → ¬(cheap x) :=
by
  intro x
  exact h x


end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_sufficient_for_not_cheap_l679_67901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_relationship_l679_67937

/-- Given that x is directly proportional to y², y is inversely proportional to √z,
    and x = 8 when z = 16, prove that x = 2 when z = 64. -/
theorem proportional_relationship (x y z : ℝ) (m n : ℝ) 
    (h1 : x = m * y^2)
    (h2 : y = n / Real.sqrt z)
    (h3 : x = 8)
    (h4 : z = 16) :
    x = 2 ∧ z = 64 → x = 2 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_relationship_l679_67937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_12th_term_l679_67932

/-- Represents a geometric sequence -/
noncomputable def GeometricSequence := ℕ → ℝ

/-- The common ratio of a geometric sequence -/
noncomputable def commonRatio (a : GeometricSequence) : ℝ := a 2 / a 1

/-- Given a geometric sequence where the 4th term is 16 and the 9th term is 512,
    prove that the 12th term is 4096 -/
theorem geometric_sequence_12th_term
  (a : GeometricSequence)
  (h1 : a 4 = 16)
  (h2 : a 9 = 512) :
  a 12 = 4096 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_12th_term_l679_67932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l679_67997

/-- The function f(x) = (3x^2 + 8x + 12) / (3x + 4) -/
noncomputable def f (x : ℝ) : ℝ := (3*x^2 + 8*x + 12) / (3*x + 4)

/-- The oblique asymptote of f(x) -/
noncomputable def oblique_asymptote (x : ℝ) : ℝ := x + 4/3

/-- Theorem stating that the oblique asymptote approaches f(x) as x approaches infinity -/
theorem oblique_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x > M, |f x - oblique_asymptote x| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l679_67997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_l679_67921

theorem plant_arrangement (n m : ℕ) (h1 : n = 5) (h2 : m = 3) :
  (Nat.factorial (n + 1)) * (Nat.factorial m) = 4320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_l679_67921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alfreds_scooter_gain_percent_l679_67919

/-- Calculates the gain percent for Alfred's scooter sale -/
theorem alfreds_scooter_gain_percent :
  ∀ (scooter_cost repair_cost original_accessory_cost discount_rate selling_price : ℝ),
  scooter_cost = 4400 →
  repair_cost = 800 →
  original_accessory_cost = 600 →
  discount_rate = 0.20 →
  selling_price = 5800 →
  let discounted_accessory_cost := original_accessory_cost * (1 - discount_rate)
  let total_cost := scooter_cost + repair_cost + discounted_accessory_cost
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  abs (gain_percent - 2.11) < 0.01 := by
  sorry

#eval "Alfred's scooter gain percent theorem is now defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alfreds_scooter_gain_percent_l679_67919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_theorem_l679_67982

/-- The distance covered by a wheel with given diameter and number of revolutions -/
noncomputable def distance_covered (diameter : ℝ) (revolutions : ℝ) : ℝ :=
  Real.pi * diameter * revolutions

/-- Theorem stating the distance covered by a wheel with specific diameter and revolutions -/
theorem wheel_distance_theorem (diameter : ℝ) (revolutions : ℝ) 
  (h1 : diameter = 15)
  (h2 : revolutions = 11.210191082802547) :
  ∃ ε > 0, |distance_covered diameter revolutions - 528.32| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_theorem_l679_67982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marketers_percentage_value_l679_67940

/-- Represents the percentage of marketers in the company -/
def marketers_percentage : ℝ := sorry

/-- Represents the percentage of engineers in the company -/
def engineers_percentage : ℝ := 0.1

/-- Represents the percentage of managers in the company -/
def managers_percentage : ℝ := 1 - engineers_percentage - marketers_percentage

/-- Average salary of marketers -/
def marketers_salary : ℝ := 50000

/-- Average salary of engineers -/
def engineers_salary : ℝ := 80000

/-- Average salary of managers -/
def managers_salary : ℝ := 370000

/-- Average salary for all employees -/
def average_salary : ℝ := 80000

/-- Theorem stating that the percentage of marketers is approximately 83.47% -/
theorem marketers_percentage_value :
  ∃ ε > 0, |marketers_percentage - 0.8347| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marketers_percentage_value_l679_67940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_of_interest_problem_l679_67984

/-- Calculates the rate of interest given principal, simple interest, and time. -/
noncomputable def calculate_rate_of_interest (principal : ℝ) (simple_interest : ℝ) (time : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * time)

/-- Theorem stating that given the specified conditions, the rate of interest is 22.5%. -/
theorem rate_of_interest_problem (principal : ℝ) (simple_interest : ℝ) (time : ℝ)
  (h1 : principal = 400)
  (h2 : simple_interest = 180)
  (h3 : time = 2) :
  calculate_rate_of_interest principal simple_interest time = 22.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_rate_of_interest 400 180 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_of_interest_problem_l679_67984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_pool_capacity_value_l679_67930

/-- The capacity of a pool given specific filling conditions -/
theorem pool_capacity (t_both : ℝ) (t_first : ℝ) (diff_rate : ℝ) (h1 : t_both = 48) 
  (h2 : t_first = 120) (h3 : diff_rate = 50) : ℝ :=
let C := t_both * t_first * diff_rate / (t_first - t_both)
C

theorem pool_capacity_value : pool_capacity 48 120 50 rfl rfl rfl = 12000 := by
  unfold pool_capacity
  simp
  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_pool_capacity_value_l679_67930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_iff_m_eq_one_l679_67957

noncomputable def i : ℂ := Complex.I

noncomputable def z (m : ℝ) : ℂ := (1 + i) / (1 - i) + m * ((1 - i) / (1 + i))

theorem z_is_real_iff_m_eq_one :
  ∀ m : ℝ, (z m).im = 0 ↔ m = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_iff_m_eq_one_l679_67957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_special_sum_l679_67933

/-- For any integer n ≥ 1, the number of distinct prime factors of 2^(2^n) + 2^(2^(n-1)) + 1 is greater than or equal to n -/
theorem distinct_prime_factors_of_special_sum (n : ℕ) (hn : n ≥ 1) :
  (Nat.factors (2^(2^n) + 2^(2^(n-1)) + 1)).length ≥ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_special_sum_l679_67933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_senior_class_girls_l679_67943

theorem senior_class_girls (boys : ℕ) (girls : ℕ) : 
  boys = 160 →
  (0.75 * (boys : ℝ) + 0.6 * (girls : ℝ)) / ((boys : ℝ) + (girls : ℝ)) = 0.6667 →
  girls = 200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_senior_class_girls_l679_67943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_line_tangent_to_circle_l679_67983

/-- Given an ellipse E with equation x²/a² + y²/b² = 1 where a > b > 0,
    and its major axis is √3 times the length of its minor axis. -/
def Ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ a = Real.sqrt 3 * b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

/-- A line l intersecting the ellipse at points P and Q -/
structure IntersectionLine (a b : ℝ) where
  slope : ℝ
  yIntercept : ℝ

/-- The distance from a point to a line -/
noncomputable def distancePointToLine (x y : ℝ) (l : IntersectionLine a b) : ℝ :=
  abs (y - l.slope * x - l.yIntercept) / Real.sqrt (1 + l.slope^2)

theorem ellipse_eccentricity (a b : ℝ) (h : Ellipse a b) :
  eccentricity a b = Real.sqrt 6 / 3 := by sorry

theorem line_tangent_to_circle 
  (a b : ℝ) (h1 : Ellipse a b) 
  (h2 : Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 2) 
  (l : IntersectionLine a b) 
  (h3 : ∃ (x1 y1 x2 y2 : ℝ), 
    x1^2/a^2 + y1^2/b^2 = 1 ∧ 
    x2^2/a^2 + y2^2/b^2 = 1 ∧ 
    y1 = l.slope * x1 + l.yIntercept ∧ 
    y2 = l.slope * x2 + l.yIntercept ∧ 
    x1 * x2 + y1 * y2 = 0) :
  ∀ (x y : ℝ), x^2 + y^2 = 3/4 → distancePointToLine x y l = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_line_tangent_to_circle_l679_67983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_tan_function_l679_67912

noncomputable def f (x : ℝ) := Real.tan (Real.pi * x / 2 + Real.pi / 3)

theorem domain_of_tan_function :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∀ k : ℤ, x ≠ 2 * k + 1 / 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_tan_function_l679_67912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l679_67974

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem arithmetic_sequence_ratio
  (a₁ d : ℝ)
  (h_d : d ≠ 0)
  (h_eq : arithmetic_sequence a₁ d 8 = 2 * arithmetic_sequence a₁ d 3) :
  arithmetic_sum a₁ d 15 / arithmetic_sum a₁ d 5 = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l679_67974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_useful_items_percentage_l679_67931

/-- Represents the contents of Mason's attic --/
structure AtticContents where
  total : ℕ
  useful : ℕ
  junk : ℕ
  heirlooms : ℕ

/-- The percentage of items that are junk --/
def junkPercentage : ℚ := 70 / 100

/-- Theorem stating that the percentage of useful items in Mason's attic is 20% --/
theorem useful_items_percentage (contents : AtticContents) 
  (h1 : contents.junk = 28)
  (h2 : contents.useful = 8)
  (h3 : contents.junk = (junkPercentage * contents.total) / 1) :
  (contents.useful : ℚ) / contents.total = 20 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_useful_items_percentage_l679_67931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l679_67986

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := x^3 + (2023^x - 1) / (2023^x + 1) + 5

-- State the theorem
theorem max_value_theorem (a b : ℝ) (h : f (2 * a^2) + f (b^2 - 2) = 10) :
  ∃ (M : ℝ), M = (3 * Real.sqrt 2) / 4 ∧ a * Real.sqrt (1 + b^2) ≤ M := by
  -- The proof is omitted and replaced with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l679_67986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l679_67962

/-- The function f(x) defined on the interval [0, 2] -/
noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x + 5

/-- The theorem stating the minimum and maximum values of f(x) on [0, 2] -/
theorem f_min_max :
  ∃ (min max : ℝ), 
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x ≥ min) ∧
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ f x = min) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x ≤ max) ∧
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ f x = max) ∧
    min = 1/2 ∧ max = 5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l679_67962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l679_67949

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 2*a*x - 3 else (4-a)*x + 1

theorem increasing_function_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → a ∈ Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l679_67949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l679_67904

-- Define the train's length in meters
noncomputable def train_length : ℝ := 600

-- Define the train's speed in km/hr
noncomputable def train_speed_kmh : ℝ := 144

-- Define the conversion factor from km/hr to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Define the time to cross the pole in seconds
noncomputable def time_to_cross : ℝ := 15

-- Theorem statement
theorem train_crossing_time :
  train_length / (train_speed_kmh * kmh_to_ms) = time_to_cross :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l679_67904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_share_is_495_l679_67978

/-- Proves that C's share is 495 Rs given the problem conditions -/
theorem c_share_is_495 (total : ℕ) (a b c : ℕ) : 
  total = 1010 →
  a + b + c = total →
  (a - 25 : ℚ) / 3 = (b - 10 : ℚ) / 2 →
  (b - 10 : ℚ) / 2 = (c - 15 : ℚ) / 5 →
  c = 495 := by
  sorry

#check c_share_is_495

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_share_is_495_l679_67978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_on_interval_l679_67988

open Real

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := sin (2 * (x - π / 6))

-- State the theorem
theorem g_min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-π/3) 0 ∧ g x = -1 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-π/3) 0 → g y ≥ -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_on_interval_l679_67988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_correct_l679_67907

/-- A king can move one square in any direction (horizontally, vertically, or diagonally) -/
structure King : Type := mk ::

/-- A chessboard of size n x n -/
def Chessboard (n : ℕ) : Type := Fin n → Fin n → Option King

/-- Two kings threaten each other if they are adjacent or diagonally adjacent -/
def threatens (n : ℕ) (board : Chessboard n) (i j k l : Fin n) : Prop :=
  board i j = some King.mk ∧ board k l = some King.mk ∧
  (((i : ℤ) - (k : ℤ)).natAbs ≤ 1 ∧ ((j : ℤ) - (l : ℤ)).natAbs ≤ 1) ∧
  ¬(i = k ∧ j = l)

/-- A valid placement of kings on the board -/
def valid_placement (n : ℕ) (board : Chessboard n) : Prop :=
  ∀ i j k l, ¬(threatens n board i j k l)

/-- The number of kings on the board -/
def num_kings (n : ℕ) (board : Chessboard n) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin n)) (λ i =>
    Finset.sum (Finset.univ : Finset (Fin n)) (λ j =>
      match board i j with
      | some King.mk => 1
      | none => 0
    )
  ))

/-- The maximum number of kings that can be placed on an n x n chessboard -/
noncomputable def max_kings (n : ℕ) : ℕ :=
  Nat.floor ((n + 1)^2 / 4 : ℚ)

theorem max_kings_correct (n : ℕ) :
  ∃ (board : Chessboard n), valid_placement n board ∧ num_kings n board = max_kings n ∧
  ∀ (board' : Chessboard n), valid_placement n board' → num_kings n board' ≤ max_kings n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_correct_l679_67907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_sum_condition_l679_67906

/-- Represents the letters on the calendar --/
inductive Letter
| A | B | C | P | Q | R | S | T

/-- The date behind a given letter --/
def date (l : Letter) : ℕ := sorry

/-- The conditions of the calendar arrangement --/
axiom date_B_minus_A : date Letter.B = date Letter.A + 22
axiom sum_condition : ∃ l : Letter, date Letter.C + date l = 2 * date Letter.A

/-- The theorem to prove --/
theorem q_satisfies_sum_condition : 
  date Letter.C + date Letter.Q = 2 * date Letter.A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_sum_condition_l679_67906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_men_count_l679_67946

/-- Theorem stating that if M men have provisions for 17 days, and 500 more men make the provisions last for 11.333333333333334 days, then M is approximately 1765. -/
theorem initial_men_count (M : ℕ) : 
  (M : ℚ) * 17 = ((M : ℚ) + 500) * (11333333333333334 / 1000000000000000) → 
  ∃ (ε : ℚ), abs ((M : ℚ) - 1765) ≤ ε ∧ ε < 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_men_count_l679_67946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_digit_l679_67944

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def number (B : ℕ) : ℕ := 303200 + B

theorem unique_prime_digit :
  ∃! B : ℕ, B ∈ ({1, 2, 6, 7, 9} : Set ℕ) ∧ is_prime (number B) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_digit_l679_67944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_cost_price_cost_price_value_l679_67976

/-- Represents the cost price of a radio --/
noncomputable def cost_price : ℝ := 
  (300 - 30) / (1 + 17.64705882352942 / 100)

/-- Represents the overhead expenses --/
def overhead : ℝ := 30

/-- Represents the selling price --/
def selling_price : ℝ := 300

/-- Represents the profit percentage --/
def profit_percentage : ℝ := 17.64705882352942

/-- Theorem stating the relationship between cost price, overhead, selling price, and profit percentage --/
theorem radio_cost_price : 
  cost_price * (1 + profit_percentage / 100) = selling_price - overhead := by
  sorry

/-- The cost price of the radio is approximately 229.41 --/
theorem cost_price_value : 
  ‖cost_price - 229.41‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_cost_price_cost_price_value_l679_67976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equations_two_distinct_zeros_range_l679_67991

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2

-- Part I: Tangent line equations
theorem tangent_line_equations :
  ∃ (m : ℝ), (∀ x y : ℝ, x - y + 1 = 0 → y = f 1 x ∧ (0, 1) ∈ {(x, y) | y = m * x + 1}) ∨
             (∀ x y : ℝ, (Real.exp 1 - 2) * x - y + 1 = 0 → y = f 1 x ∧ (0, 1) ∈ {(x, y) | y = m * x + 1}) :=
by sorry

-- Part II: Range of a for two distinct zeros
theorem two_distinct_zeros_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ a > (Real.exp 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equations_two_distinct_zeros_range_l679_67991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_correct_l679_67959

def Answer : Type := String

def correct_answer : Answer := "B. the;a"

def is_correct (answer : Answer) : Prop :=
  answer = correct_answer

theorem solution_is_correct : is_correct correct_answer := by
  rfl

#check solution_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_correct_l679_67959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parent_son_age_ratio_l679_67920

theorem parent_son_age_ratio : 
  ∀ (parent_age son_age : ℕ),
  parent_age = 54 →
  parent_age - 9 = 5 * (son_age - 9) →
  parent_age = 3 * son_age :=
by
  intros parent_age son_age h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parent_son_age_ratio_l679_67920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l679_67967

/-- The function f(x) = e^x - x - 2 -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := Real.exp x - 1

theorem max_k_value (k : ℤ) (hk : k > 2) :
  ∃ x : ℝ, x > 0 ∧ (k - x) / (x + 1) * f' x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l679_67967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l679_67927

-- Define the curve C
noncomputable def curve_C (α : Real) : Real × Real :=
  (3 + Real.sqrt 10 * Real.cos α, 1 + Real.sqrt 10 * Real.sin α)

-- Define the polar equation of the straight line
def line_polar (θ ρ : Real) : Prop :=
  Real.sin θ - Real.cos θ = 1 / ρ

-- Theorem statement
theorem chord_length :
  ∃ (chord_length : Real),
    (∀ α : Real, curve_C α ∈ {p : Real × Real | (p.1 - 3)^2 + (p.2 - 1)^2 = 10}) ∧
    chord_length = Real.sqrt 22 ∧
    (∃ A B : Real × Real,
      A ∈ {p : Real × Real | (p.1 - 3)^2 + (p.2 - 1)^2 = 10} ∧
      B ∈ {p : Real × Real | (p.1 - 3)^2 + (p.2 - 1)^2 = 10} ∧
      (∃ θ ρ : Real, line_polar θ ρ ∧ A.1 - A.2 + 1 = 0 ∧ B.1 - B.2 + 1 = 0) ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = chord_length^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l679_67927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2013_l679_67971

def my_sequence (a : ℕ → ℤ) : Prop :=
  (∀ n, a (n + 2) = a (n + 1) - a n) ∧ 
  a 1 = 2 ∧ 
  a 2 = 5

theorem sequence_2013 (a : ℕ → ℤ) (h : my_sequence a) : a 2013 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2013_l679_67971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_100th_term_l679_67935

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem sequence_100th_term : arithmetic_sequence 3 (-2) 100 = -195 := by
  unfold arithmetic_sequence
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_100th_term_l679_67935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l679_67970

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | 0 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l679_67970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_a_equals_six_l679_67979

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.sqrt x
  else if x ≥ 1 then 2 * (x - 1)
  else 0  -- Add this to make the function total

-- State the theorem
theorem f_inverse_a_equals_six (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : f a = f (a + 1)) :
  f (1 / a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_a_equals_six_l679_67979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l679_67942

/-- Represents a person's age -/
def Age := ℕ

/-- Represents the number of years in the future -/
def YearsLater := ℕ

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Represents the condition for Chloe's age to be a multiple of Max's age -/
def isMultiple (chloesAge maxAge : ℕ) : Prop := ∃ k : ℕ, chloesAge = k * maxAge

theorem birthday_problem (maxAge joeyAge chloeAge : ℕ) (n : ℕ) :
  maxAge = 3 →
  joeyAge = chloeAge + 2 →
  (∃! count : ℕ, count = 12 ∧ 
    ∀ i : ℕ, i < count → isMultiple (chloeAge + i) (maxAge + i)) →
  (∃ k : ℕ, joeyAge + n = k * (maxAge + n)) →
  (∀ m : ℕ, m < n → ¬∃ k : ℕ, joeyAge + m = k * (maxAge + m)) →
  sumOfDigits (joeyAge + n) = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l679_67942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l679_67909

/-- Simple interest calculation --/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ := (principal * rate * time) / 100

/-- Problem statement --/
theorem interest_rate_calculation (principal simple_interest time : ℝ) 
  (h_principal : principal = 400)
  (h_interest : simple_interest = 100)
  (h_time : time = 2) :
  ∃ rate : ℝ, simple_interest = (principal * rate * time) / 100 ∧ rate = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l679_67909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_implies_n_value_l679_67916

/-- Two vectors in ℝ² -/
def a : Fin 2 → ℝ := ![2, -3]
def b (n : ℝ) : Fin 2 → ℝ := ![1 + n, n]

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), v 0 * w 1 = v 1 * w 0 ∧ (∀ i, k * v i = w i)

theorem parallel_vectors_implies_n_value (n : ℝ) :
  parallel a (b n) → n = -3/5 := by
  sorry

#check parallel_vectors_implies_n_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_implies_n_value_l679_67916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l679_67950

theorem polynomial_factorization (x : ℝ) :
  (λ x => 3 * (1 + 2*x) * (-2*x + 1)) = (λ x => 3 - 12*x^2) := by
  ext x
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l679_67950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tint_percentage_approx_37_l679_67915

/-- Calculates the percentage of red tint in a new mixture after adding more red tint -/
noncomputable def red_tint_percentage (initial_volume : ℝ) (initial_red_percent : ℝ) (added_red_volume : ℝ) : ℝ :=
  let initial_red_volume := initial_volume * (initial_red_percent / 100)
  let new_red_volume := initial_red_volume + added_red_volume
  let new_total_volume := initial_volume + added_red_volume
  (new_red_volume / new_total_volume) * 100

/-- Theorem stating that the new red tint percentage is approximately 37% -/
theorem red_tint_percentage_approx_37 :
  ∃ ε > 0, ε < 0.5 ∧ |red_tint_percentage 30 20 8 - 37| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tint_percentage_approx_37_l679_67915
