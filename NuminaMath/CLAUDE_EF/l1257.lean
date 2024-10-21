import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_is_lipschitz_sqrt_x_plus_one_is_lipschitz_sqrt_x_plus_one_min_lipschitz_l1257_125731

-- Define Lipschitz condition
def is_lipschitz (f : ℝ → ℝ) (k : ℝ) (D : Set ℝ) :=
  ∀ x y, x ∈ D → y ∈ D → |f x - f y| ≤ k * |x - y|

-- Theorem 1: f(x) = x is Lipschitz with k = 1 on ℝ
theorem identity_is_lipschitz :
  is_lipschitz (λ x : ℝ ↦ x) 1 Set.univ :=
sorry

-- Theorem 2: f(x) = √(x+1) is Lipschitz with k = 1/2 on [0, +∞)
theorem sqrt_x_plus_one_is_lipschitz :
  is_lipschitz (λ x : ℝ ↦ Real.sqrt (x + 1)) (1/2) {x : ℝ | x ≥ 0} :=
sorry

-- Theorem 3: 1/2 is the minimum Lipschitz constant for f(x) = √(x+1) on [0, +∞)
theorem sqrt_x_plus_one_min_lipschitz :
  ∀ k : ℝ, k < 1/2 → ¬(is_lipschitz (λ x : ℝ ↦ Real.sqrt (x + 1)) k {x : ℝ | x ≥ 0}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_is_lipschitz_sqrt_x_plus_one_is_lipschitz_sqrt_x_plus_one_min_lipschitz_l1257_125731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_egyptian_rope_triangle_l1257_125778

theorem egyptian_rope_triangle :
  ∀ (a b c : ℝ),
    a = 6 ∧ b = 8 ∧ c = 10 →
    a + b + c = 24 ∧
    a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_egyptian_rope_triangle_l1257_125778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_time_example_l1257_125711

/-- The time (in minutes) it takes for a train to pass through a tunnel -/
noncomputable def train_tunnel_time (train_speed : ℝ) (tunnel_length : ℝ) (train_length : ℝ) : ℝ :=
  (tunnel_length + train_length) / train_speed * 60

/-- Theorem: A train traveling at 75 mph through a 3.5-mile tunnel, with the train being 0.25 miles long,
    takes 3 minutes to pass through the tunnel -/
theorem train_tunnel_time_example : train_tunnel_time 75 3.5 0.25 = 3 := by
  -- Unfold the definition of train_tunnel_time
  unfold train_tunnel_time
  -- Simplify the arithmetic
  simp [div_mul_eq_mul_div, add_div]
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_time_example_l1257_125711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_product_relation_l1257_125797

noncomputable def x : ℕ → ℝ
  | 0 => 2
  | n + 1 => x n ^ 2 + x n

noncomputable def y (n : ℕ) : ℝ := 1 / (1 + x n)

noncomputable def A : ℕ → ℝ
  | 0 => y 0
  | n + 1 => A n + y (n + 1)

noncomputable def B : ℕ → ℝ
  | 0 => y 0
  | n + 1 => B n * y (n + 1)

theorem sum_product_relation (n : ℕ) : 2 * A n + B n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_product_relation_l1257_125797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inscribed_semicircle_l1257_125736

-- Define necessary structures and functions
structure IsRightTriangle (triangle : Set ℝ × Set ℝ × Set ℝ) : Prop

structure IsInscribed (semicircle : Set ℝ) (triangle : Set ℝ × Set ℝ × Set ℝ) : Prop

structure DiameterOnHypotenuse (semicircle : Set ℝ) (triangle : Set ℝ × Set ℝ × Set ℝ) : Prop

structure CenterDividesHypotenuse (semicircle : Set ℝ) (triangle : Set ℝ × Set ℝ × Set ℝ) (segment1 : ℝ) (segment2 : ℝ) : Prop

def TriangleArea (triangle : Set ℝ × Set ℝ × Set ℝ) : ℝ := sorry

def SemicircleLength (semicircle : Set ℝ) : ℝ := sorry

theorem right_triangle_inscribed_semicircle 
  (triangle : Set ℝ × Set ℝ × Set ℝ) 
  (semicircle : Set ℝ) 
  (is_right_triangle : IsRightTriangle triangle) 
  (is_inscribed : IsInscribed semicircle triangle) 
  (diameter_on_hypotenuse : DiameterOnHypotenuse semicircle triangle) 
  (center_divides_hypotenuse : CenterDividesHypotenuse semicircle triangle 15 20) : 
  TriangleArea triangle = 294 ∧ SemicircleLength semicircle = 12 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inscribed_semicircle_l1257_125736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1257_125787

-- Define the polynomial
def f (x : ℝ) : ℝ := 6*x^4 - 14*x^3 - 4*x^2 + 2*x - 26

-- Define the divisor
def g (x : ℝ) : ℝ := 2*x - 6

-- Theorem statement
theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), f = λ x ↦ g x * q x + 52 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1257_125787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_30_60_90_l1257_125717

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  C = (0, 0) ∧
  A.1 = 0 ∧
  B.2 = 0 ∧
  A.2 * B.1 = A.1 * B.2 + A.2 * B.1

-- Define the reflection point P
def ReflectionPoint (A B C P : ℝ × ℝ) : Prop :=
  ∃ (D : ℝ × ℝ),
    ∃ (t : ℝ), D = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2) ∧
    (D.1 - C.1) * (B.1 - A.1) + (D.2 - C.2) * (B.2 - A.2) = 0 ∧
    P.1 - C.1 = 2 * (D.1 - C.1) ∧
    P.2 - C.2 = 2 * (D.2 - C.2)

-- Define collinearity of three points
def AreCollinear (P Q R : ℝ × ℝ) : Prop :=
  (Q.2 - P.2) * (R.1 - Q.1) = (R.2 - Q.2) * (Q.1 - P.1)

-- Main theorem
theorem triangle_angles_30_60_90
  (A B C P : ℝ × ℝ)
  (h_triangle : Triangle A B C)
  (h_reflection : ReflectionPoint A B C P)
  (h_collinear : AreCollinear P ((A.1 + C.1) / 2, (A.2 + C.2) / 2) ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) :
  ∃ (θ : ℝ), 
    θ = Real.pi / 6 ∧
    Real.cos θ = B.1 / Real.sqrt (B.1^2 + A.2^2) ∧
    Real.sin θ = A.2 / Real.sqrt (B.1^2 + A.2^2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_30_60_90_l1257_125717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_original_price_l1257_125724

/-- Proves that given an article sold for Rs. 550 with a loss of 8.333333333333329%, 
    the original price of the article was approximately Rs. 600. -/
theorem article_original_price 
  (selling_price : ℝ) 
  (loss_percent : ℝ) 
  (h_selling_price : selling_price = 550) 
  (h_loss_percent : loss_percent = 8.333333333333329) : 
  ∃ (original_price : ℝ), 
    (original_price * (1 - loss_percent / 100) = selling_price) ∧ 
    (abs (original_price - 600) < 0.01) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_original_price_l1257_125724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ride_time_is_55_seconds_l1257_125785

/-- Represents the time it takes Clea to ride an escalator under different conditions -/
structure EscalatorRide where
  stationary_walk_time : ℝ
  moving_walk_time : ℝ
  maintenance_delay : ℝ

/-- Calculates the time for Clea to ride the escalator without walking -/
noncomputable def ride_time (e : EscalatorRide) : ℝ :=
  let walking_speed := e.stationary_walk_time⁻¹
  let escalator_speed := walking_speed * (e.stationary_walk_time / e.moving_walk_time - 1)
  e.stationary_walk_time * escalator_speed⁻¹ + e.maintenance_delay

/-- Theorem stating that given the conditions, the ride time is 55 seconds -/
theorem ride_time_is_55_seconds (e : EscalatorRide)
  (h1 : e.stationary_walk_time = 75)
  (h2 : e.moving_walk_time = 30)
  (h3 : e.maintenance_delay = 5) :
  ride_time e = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ride_time_is_55_seconds_l1257_125785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_l1257_125720

theorem complex_modulus (z : ℂ) (h : (1 + 2*Complex.I)*z = (1 - Complex.I)) : 
  Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_l1257_125720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_a_div_factorial_l1257_125713

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

def a : ℚ := (((factorial 13)^16 - (factorial 13)^8) : ℚ) / ((factorial 13)^8 + (factorial 13)^4)

theorem units_digit_of_a_div_factorial : units_digit (Int.floor (a / (factorial 13)^4)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_a_div_factorial_l1257_125713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_2014_jiangxi_l1257_125794

theorem problem_2014_jiangxi (a : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ (5 : ℝ)^(abs x)
  let g : ℝ → ℝ := fun x ↦ a*x^2 - x
  (f (g 1) = 1) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_2014_jiangxi_l1257_125794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_red_ball_prob_distribution_X_prob_sum_to_one_l1257_125799

-- Define the total number of balls and the number of red and white balls
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3

-- Define the number of balls drawn
def balls_drawn : ℕ := 3

-- Define the random variable X as the number of red balls drawn
def X : ℕ → ℝ := sorry

-- Define the probability function
def P : Set ℕ → ℝ := sorry

-- Theorem for the probability of drawing exactly one red ball
theorem prob_one_red_ball : P {1} = 3/5 := by sorry

-- Theorem for the probability distribution of X
theorem prob_distribution_X :
  P {0} = 1/10 ∧ P {1} = 3/5 ∧ P {2} = 3/10 := by sorry

-- Theorem to ensure the probabilities sum to 1
theorem prob_sum_to_one :
  P {0} + P {1} + P {2} = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_red_ball_prob_distribution_X_prob_sum_to_one_l1257_125799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_endpoint_l1257_125776

theorem line_segment_endpoint (M A B : ℝ × ℝ) 
  (h_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h_M : M = (2, 3))
  (h_A : A = (7, -4)) :
  B = (-3, 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_endpoint_l1257_125776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_expression_equals_one_l1257_125702

theorem logarithmic_expression_equals_one :
  2 * (Real.log 2 / Real.log 3) - (Real.log (32 / 9) / Real.log 3) + (Real.log 8 / Real.log 3) - 
  2 * (5 ^ (Real.log 3 / Real.log 5)) + 16 ^ (3/4 : Real) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_expression_equals_one_l1257_125702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sequence_property_x_initial_condition_l1257_125727

noncomputable def x : ℕ → ℝ
  | 0 => 10  -- Arbitrary value between 8 and 12
  | n + 1 => (6 * (x n)^2 - 23 * x n) / (7 * (x n - 5))

theorem x_sequence_property :
  ∀ n : ℕ, n ≥ 1 → x n < x (n + 1) ∧ x (n + 1) < 12 :=
by
  sorry

theorem x_initial_condition :
  8 < x 0 ∧ x 0 < 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sequence_property_x_initial_condition_l1257_125727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_angles_l1257_125757

def is_valid_triangle_set (angles : List ℝ) : Prop :=
  angles.length = 9 ∧
  angles.sum = 540 ∧
  (∃ d : ℝ, ∀ i : Fin 8, angles[i.val]! < angles[i.val + 1]! ∧ angles[i.val + 1]! - angles[i.val]! = d) ∧
  42 ∈ angles

theorem triangle_max_angles (angles : List ℝ) :
  is_valid_triangle_set angles →
  (List.maximum? angles = some 78 ∨
   List.maximum? angles = some 84 ∨
   List.maximum? angles = some 96) :=
by
  intro h
  sorry

#check triangle_max_angles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_angles_l1257_125757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1257_125707

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x - 2) + (x - 3) ^ 0

-- Define the domain of f
def domain_f : Set ℝ := {x | x > 2 ∧ x ≠ 3}

-- Theorem stating that domain_f is the correct domain for f
theorem f_domain : ∀ x : ℝ, x ∈ domain_f ↔ (∃ y : ℝ, f x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1257_125707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l1257_125722

-- Define the power function as noncomputable
noncomputable def power_function (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

-- Theorem statement
theorem power_function_value (a : ℝ) :
  (power_function a 2 = Real.sqrt 2) → (power_function a 4 = 2) := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l1257_125722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_increasing_l1257_125772

noncomputable def f (x : ℝ) : ℝ := 1 - Real.log (abs x) / Real.log (1/2)

theorem f_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_increasing_l1257_125772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_bounds_l1257_125756

noncomputable def g (x : ℝ) : ℝ := Real.cos x + 3 * Real.sin x + 4 * (Real.cos x / Real.sin x)

def is_root (x : ℝ) : Prop := g x = 0

def smallest_positive_root (s : ℝ) : Prop :=
  is_root s ∧ s > 0 ∧ ∀ x, 0 < x ∧ x < s → ¬(is_root x)

theorem smallest_positive_root_bounds :
  ∃ s, smallest_positive_root s ∧ 2 ≤ s ∧ s < 3 := by sorry

#check smallest_positive_root_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_bounds_l1257_125756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_l1257_125700

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := deriv f x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (4*x - y - 2 = 0) :=
by
  -- Introduce the variables
  intro x y
  
  -- Define x₀, y₀, and m
  let x₀ := 1
  let y₀ := f x₀
  let m := deriv f x₀
  
  -- Proof steps would go here
  sorry  -- We use sorry to skip the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_l1257_125700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_value_l1257_125771

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2

theorem tangent_slope_implies_a_value :
  ∀ a : ℝ, (deriv (f a)) 2 = -3/2 → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_value_l1257_125771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_on_ellipse_l1257_125734

/-- The ellipse in the rectangular coordinate system xOy -/
def Γ : Set (ℝ × ℝ) := {p | p.1^2 / 3 + p.2^2 = 1}

/-- The left vertex of the ellipse -/
noncomputable def P : ℝ × ℝ := (-Real.sqrt 3, 0)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The vector from P to a point X -/
noncomputable def vector_PX (X : ℝ × ℝ) : ℝ × ℝ := (X.1 - P.1, X.2 - P.2)

theorem dot_product_range_on_ellipse :
  ∀ A B : ℝ × ℝ, A ∈ Γ → B ∈ Γ →
  -1/4 ≤ dot_product (vector_PX A) (vector_PX B) ∧
  dot_product (vector_PX A) (vector_PX B) ≤ 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_on_ellipse_l1257_125734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_bisector_intersection_l1257_125723

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define a perimeter-bisector line
noncomputable def perimeterBisectorLine (t : Triangle) (vertex : ℝ × ℝ) : Line :=
  sorry

-- Define the intersection point of two lines
noncomputable def intersectionPoint (l1 l2 : Line) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem perimeter_bisector_intersection (t : Triangle) :
  let bisectorA := perimeterBisectorLine t t.A
  let bisectorB := perimeterBisectorLine t t.B
  let bisectorC := perimeterBisectorLine t t.C
  let pointAB := intersectionPoint bisectorA bisectorB
  let pointBC := intersectionPoint bisectorB bisectorC
  let pointAC := intersectionPoint bisectorA bisectorC
  pointAB = pointBC ∧ pointBC = pointAC := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_bisector_intersection_l1257_125723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l1257_125773

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define point P
def point_P : ℝ × ℝ := (0, 3)

-- Define the intersection points A and B
def is_intersection (p : ℝ × ℝ) : Prop :=
  line_l p.1 p.2 ∧ curve_C p.1 p.2

-- Theorem statement
theorem intersection_product :
  ∃ (A B : ℝ × ℝ),
    is_intersection A ∧
    is_intersection B ∧
    A ≠ B ∧
    let PA := Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2)
    let PB := Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)
    PA * PB = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l1257_125773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_intersection_line_existence_l1257_125760

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- The minimum distance from a point on the ellipse to a focus -/
noncomputable def min_focal_distance (e : Ellipse) : ℝ := e.a - e.a * eccentricity e

theorem ellipse_properties (e : Ellipse) 
  (h_ecc : eccentricity e = 1/2) 
  (h_min_dist : min_focal_distance e = 1) :
  e.a = 2 ∧ e.b = Real.sqrt 3 := by
  sorry

theorem intersection_line_existence (e : Ellipse) 
  (h_ecc : eccentricity e = 1/2) 
  (h_min_dist : min_focal_distance e = 1) :
  ∃ (k m : ℝ), 
    (∀ x y : ℝ, y = k * x + m → (x^2 / 4 + y^2 / 3 = 1 → 
      ∃ (x' y' : ℝ), y' = k * x' + m ∧ x^2 / 4 + y^2 / 3 = 1 ∧ 
        x * x' + y * y' = 0)) ∧
    (m < -2 * Real.sqrt 21 / 7 ∨ m > 2 * Real.sqrt 21 / 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_intersection_line_existence_l1257_125760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_point_l1257_125710

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 16 = 1

-- Define the left focus
noncomputable def left_focus : ℝ × ℝ := (-2 * Real.sqrt 5, 0)

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 5

-- Define the vertices
def left_vertex : ℝ × ℝ := (-2, 0)
def right_vertex : ℝ × ℝ := (2, 0)

-- Define a line passing through (-4, 0)
def intersecting_line (m : ℝ) (y : ℝ) : ℝ := m * y - 4

-- Define the theorem
theorem hyperbola_intersection_point :
  ∀ (m : ℝ) (y₁ y₂ : ℝ),
  -- Conditions
  (hyperbola_C (intersecting_line m y₁) y₁) ∧
  (hyperbola_C (intersecting_line m y₂) y₂) ∧
  (y₁ > 0) ∧ (intersecting_line m y₁ < 0) ∧  -- M in second quadrant
  (y₁ ≠ y₂) →
  -- Conclusion
  ∃ (x : ℝ),
    x = -1 ∧
    -- P is the intersection of MA₁ and NA₂
    (y₁ / (intersecting_line m y₁ + 2) * (x + 2) =
     y₂ / (intersecting_line m y₂ - 2) * (x - 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_point_l1257_125710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_sequence_l1257_125719

def my_sequence (n : ℕ) : ℚ := (2 * n : ℚ) / (2 * n + 1 : ℚ)

theorem tenth_term_of_sequence : my_sequence 10 = 20 / 21 := by
  -- Unfold the definition of my_sequence
  unfold my_sequence
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_sequence_l1257_125719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_AC_onto_BD_l1257_125763

-- Define the points
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (-2, -1)
def D : ℝ × ℝ := (3, 4)

-- Define the vectors
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def BD : ℝ × ℝ := (D.1 - B.1, D.2 - B.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define vector projection
noncomputable def vector_projection (v w : ℝ × ℝ) : ℝ := (dot_product v w) / (magnitude w)

theorem projection_of_AC_onto_BD :
  vector_projection AC BD = -3 * Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_AC_onto_BD_l1257_125763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_perimeter_shaded_region_perimeter_equals_12π_l1257_125754

/-- The perimeter of the shaded region formed by the arcs connecting the centers of three identical touching circles with diameter 24 is equal to 12π. -/
theorem shaded_region_perimeter (circle_diameter : ℝ) (h1 : circle_diameter = 24) : ℝ :=
  let circle_radius := circle_diameter / 2
  let arc_angle : ℝ := 60 -- in degrees
  let arc_length := 2 * circle_radius * Real.pi * (arc_angle / 360)
  let num_arcs := 3
  num_arcs * arc_length

theorem shaded_region_perimeter_equals_12π :
  shaded_region_perimeter 24 rfl = 12 * Real.pi :=
by
  -- Expand the definition of shaded_region_perimeter
  unfold shaded_region_perimeter
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_perimeter_shaded_region_perimeter_equals_12π_l1257_125754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_given_circle_l1257_125786

/-- The equation of a circle in the xy-plane -/
def circle_equation (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + 4*x + y^2 - 8*y + 16 = 0

/-- The center of a circle given by its equation -/
noncomputable def circle_center (eq : (ℝ × ℝ) → Prop) : ℝ × ℝ := sorry

theorem center_of_given_circle :
  circle_center circle_equation = (-2, 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_given_circle_l1257_125786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_constant_l1257_125775

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incircle
structure Incircle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the A-excircle
structure Excircle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a point on a line
def PointOnLine (A B X : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, X = (1 - t) • A + t • B

-- Define a tangent from a point to a circle (generalized for both Incircle and Excircle)
def Tangent (X : ℝ × ℝ) (circle : Incircle ⊕ Excircle) (Y : ℝ × ℝ) : Prop :=
  let center := match circle with
    | Sum.inl inc => inc.center
    | Sum.inr exc => exc.center
  let radius := match circle with
    | Sum.inl inc => inc.radius
    | Sum.inr exc => exc.radius
  ∃ t : ℝ, Y = X + t • (center - X) ∧ 
             ‖Y - center‖ = radius

-- Main theorem
theorem tangent_sum_constant 
  (ABC : Triangle) 
  (ω : Incircle) 
  (A_excircle : Excircle)
  (A' : ℝ × ℝ)
  (h1 : ‖ABC.A - ABC.B‖ < ‖ABC.A - ABC.C‖)
  (h2 : Tangent A' (Sum.inr A_excircle) ABC.B)
  (h3 : Tangent A' (Sum.inr A_excircle) ABC.C) :
  ∃ k : ℝ, ∀ X : ℝ × ℝ, 
    PointOnLine ABC.A A' X →
    (∀ Y Z : ℝ × ℝ, 
      Tangent X (Sum.inl ω) Y ∧ Tangent X (Sum.inl ω) Z ∧ 
      PointOnLine ABC.B ABC.C Y ∧ PointOnLine ABC.B ABC.C Z →
      ‖X - Y‖ + ‖X - Z‖ = k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_constant_l1257_125775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_and_overlap_implies_omega_two_l1257_125750

noncomputable def original_func (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 6)

noncomputable def shifted_func (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * (x + Real.pi / 3) - Real.pi / 6)

noncomputable def cosine_func (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

theorem shift_and_overlap_implies_omega_two :
  ∃ ω : ℝ, (∀ x : ℝ, shifted_func ω x = cosine_func ω x) ∧ ω = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_and_overlap_implies_omega_two_l1257_125750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_two_equals_five_l1257_125752

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

-- State the theorem
theorem f_of_two_equals_five :
  (∀ x : ℝ, x ≥ 0 → f (Real.sqrt x + 1) = 2 * x + 3) →
  f 2 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_two_equals_five_l1257_125752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_freshmen_musicians_l1257_125715

/-- Given a school with the following conditions:
  * There are 400 total students
  * 50% of freshmen play a musical instrument
  * 25% of non-freshmen do not play a musical instrument
  * 40% of all students do not play a musical instrument

  Prove that 120 non-freshmen play a musical instrument
-/
theorem non_freshmen_musicians (total : ℚ) (freshmen : ℚ) (non_freshmen : ℚ)
  (h_total : total = 400)
  (h_sum : freshmen + non_freshmen = total)
  (h_freshmen_musicians : freshmen / 2 = freshmen - (0.4 * total - 0.25 * non_freshmen))
  (h_non_freshmen_non_musicians : 0.25 * non_freshmen = 0.4 * total - freshmen / 2) :
  0.75 * non_freshmen = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_freshmen_musicians_l1257_125715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_tap_rate_l1257_125716

/-- The rate at which the tap admits water into a leaking cistern -/
noncomputable def tap_rate (capacity : ℝ) (leak_time : ℝ) (total_time : ℝ) : ℝ :=
  capacity / leak_time - capacity / total_time

theorem cistern_tap_rate :
  tap_rate 480 20 24 = 4 := by
  -- Unfold the definition of tap_rate
  unfold tap_rate
  -- Simplify the expression
  simp [div_sub_div]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_tap_rate_l1257_125716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_f_sign_l1257_125749

/-- Given a triangle ABC with sides a, b, c (a ≤ b ≤ c), circumradius R, and inradius r,
    this theorem proves the relationship between f = a + b - 2R - 2r and angle C. -/
theorem triangle_f_sign (a b c R r : ℝ) (A B C : ℝ) 
  (h_ab : a ≤ b) (h_bc : b ≤ c)
  (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c)
  (h_R : 0 < R) (h_r : 0 < r)
  (h_sum : A + B + C = π)
  (h_A : 0 < A) (h_AB : A ≤ B) (h_BC : B ≤ C)
  (h_c : c = 2 * R * Real.sin C)
  (h_r : r = (a + b + c) / (4 * Real.tan (A/2) + 4 * Real.tan (B/2) + 4 * Real.tan (C/2))) :
  let f := a + b - 2*R - 2*r
  (C < π/2 → f > 0) ∧ (C = π/2 → f = 0) ∧ (C > π/2 → f < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_f_sign_l1257_125749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_area_l1257_125733

noncomputable section

/-- The ellipse with equation x²/5 + y²/4 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 / 4 = 1

/-- The right focus F of the ellipse -/
def F : ℝ × ℝ := (1, 0)

/-- The line passing through F with slope 2 -/
def line (x y : ℝ) : Prop := y = 2 * (x - 1)

/-- Point A is on both the ellipse and the line -/
noncomputable def A : ℝ × ℝ := (0, -2)

/-- Point B is on both the ellipse and the line -/
noncomputable def B : ℝ × ℝ := (5/3, 4/3)

/-- The origin O -/
def O : ℝ × ℝ := (0, 0)

/-- The area of triangle OAB -/
def area_OAB : ℝ := 5/3

theorem ellipse_line_intersection_area :
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
  line A.1 A.2 ∧ line B.1 B.2 →
  area_OAB = 5/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_area_l1257_125733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1257_125795

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos (x / 2))^2 + 1

-- State the theorem
theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1257_125795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l1257_125798

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity : ℝ := Real.sqrt 5

/-- The equation of the hyperbola -/
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / m - y^2 / (m^2 + 4) = 1

/-- Theorem: For a hyperbola with the given equation and eccentricity, m equals 2 -/
theorem hyperbola_m_value (m : ℝ) :
  (∃ x y, hyperbola_equation x y m) →
  (∃ a b c, a^2 = m ∧ b^2 = m^2 + 4 ∧ c^2 = m^2 + m + 4 ∧ c / a = eccentricity) →
  m = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l1257_125798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pentagon_properties_l1257_125739

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a pentagon -/
structure Pentagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  O : Point  -- Add the center point O

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Checks if a pentagon is convex -/
def isConvex (p : Pentagon) : Prop := sorry

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Checks if a point is a diagonal of the pentagon -/
def isDiagonal (p : Pentagon) (d : Point) : Prop := sorry

/-- Checks if two points divide each other in the golden ratio -/
def divideInGoldenRatio (p1 p2 : Point) : Prop := sorry

/-- Calculates the area of the pentagon -/
noncomputable def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Calculates the area of the star formed by the pentagon's diagonals -/
noncomputable def starArea (p : Pentagon) : ℝ := sorry

/-- Calculates the area of the inner pentagon formed by the intersection of diagonals -/
noncomputable def innerPentagonArea (p : Pentagon) : ℝ := sorry

/-- Main theorem about the special pentagon -/
theorem special_pentagon_properties (p : Pentagon) 
  (h_convex : isConvex p)
  (h_areas : triangleArea p.A p.B p.C = 1 ∧ 
             triangleArea p.B p.O p.D = 1 ∧
             triangleArea p.O p.D p.E = 1 ∧
             triangleArea p.D p.E p.A = 1 ∧
             triangleArea p.E p.A p.B = 1) :
  (∀ (d1 d2 : Point), isDiagonal p d1 ∧ isDiagonal p d2 → 
    divideInGoldenRatio d1 d2) ∧
  pentagonArea p = (5 + Real.sqrt 5) / 2 ∧
  starArea p = 3 * Real.sqrt 5 - 5 ∧
  innerPentagonArea p = 5 - 2 * Real.sqrt 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pentagon_properties_l1257_125739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_locus_of_point_proof_l1257_125712

def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (1, 0)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def vector_between (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

def is_arithmetic_sequence_decreasing (a b c : ℝ) : Prop :=
  b - a = c - b ∧ b - a < 0

theorem locus_of_point (P : ℝ × ℝ) : Prop :=
  let MP := vector_between M P
  let PN := vector_between P N
  let MN := vector_between M N
  let PM := vector_between P M
  let NM := vector_between N M
  let NP := vector_between N P
  is_arithmetic_sequence_decreasing 
    (dot_product MP MN) 
    (dot_product PM PN) 
    (dot_product NM NP) →
  P.1^2 + P.2^2 = 3 ∧ P.1 > 0

-- Proof
theorem locus_of_point_proof : ∀ P : ℝ × ℝ, locus_of_point P :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_locus_of_point_proof_l1257_125712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_calculation_l1257_125790

/-- Represents a class with juniors and seniors -/
structure ClassComposition where
  total_students : ℕ
  junior_percentage : ℚ
  senior_percentage : ℚ
  overall_average : ℚ
  senior_average : ℚ
  junior_score : ℚ

/-- Theorem stating the conditions and the result to be proved -/
theorem junior_score_calculation (c : ClassComposition) : 
  c.junior_percentage = 1/10 ∧ 
  c.senior_percentage = 9/10 ∧ 
  c.overall_average = 84 ∧ 
  c.senior_average = 83 →
  c.junior_score = 93 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_calculation_l1257_125790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_values_l1257_125726

/-- Represents a stock with its initial investment, annual income, and brokerage fee percentage -/
structure Stock where
  initialInvestment : ℚ
  annualIncome : ℚ
  brokerageFeePercentage : ℚ

/-- Calculates the market value of a stock -/
def marketValue (stock : Stock) : ℚ :=
  stock.initialInvestment + stock.annualIncome - (stock.initialInvestment * stock.brokerageFeePercentage / 100)

theorem stock_market_values :
  let stockA : Stock := { initialInvestment := 6500, annualIncome := 756, brokerageFeePercentage := 1/4 }
  let stockB : Stock := { initialInvestment := 5500, annualIncome := 935, brokerageFeePercentage := 33/100 }
  let stockC : Stock := { initialInvestment := 4000, annualIncome := 1225, brokerageFeePercentage := 1/2 }
  (marketValue stockA = 7223.75) ∧
  (marketValue stockB = 6398.85) ∧
  (marketValue stockC = 5205) := by
  sorry

#eval marketValue { initialInvestment := 6500, annualIncome := 756, brokerageFeePercentage := 1/4 }
#eval marketValue { initialInvestment := 5500, annualIncome := 935, brokerageFeePercentage := 33/100 }
#eval marketValue { initialInvestment := 4000, annualIncome := 1225, brokerageFeePercentage := 1/2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_values_l1257_125726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1257_125714

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem axis_of_symmetry :
  ∃ (k : ℤ), ∀ (x : ℝ), f (-Real.pi / 2 + x) = f (-Real.pi / 2 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1257_125714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coins_identifiable_l1257_125777

/-- Represents the count reported by Baron Munchausen for a given number of counterfeit coins -/
def baronReport (actualCount : ℕ) (exaggeration : ℕ) : ℕ := actualCount + exaggeration

/-- Represents the strategy to determine counterfeit coins -/
structure Strategy where
  identify : (Finset (Fin 100) → ℕ) → Finset (Fin 100)

/-- The strategy is correct if it correctly identifies counterfeit coins for any possible configuration -/
def isCorrectStrategy (s : Strategy) : Prop :=
  ∀ (counterfeitCoins : Finset (Fin 100)) (exaggeration : ℕ),
    let report := λ (subset : Finset (Fin 100)) => 
      baronReport (Finset.card (counterfeitCoins ∩ subset)) exaggeration
    s.identify report = counterfeitCoins

/-- There exists a correct strategy to identify counterfeit coins -/
theorem counterfeit_coins_identifiable :
  ∃ (s : Strategy), isCorrectStrategy s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coins_identifiable_l1257_125777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_domain_l1257_125779

noncomputable def k (x : ℝ) : ℝ := 1 / (x + 9) + 1 / (x^2 + 9) + 1 / (x^5 + 9) + 1 / (x - 9)

def domain_k : Set ℝ := {x | x ≠ -9 ∧ x^5 ≠ -9 ∧ x ≠ 9}

theorem k_domain : 
  {x : ℝ | IsRegular (k x)} = domain_k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_domain_l1257_125779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l1257_125743

/-- Given two non-collinear and non-zero vectors a and b in a real vector space,
    if 8a + kb is collinear with ka + 2b, then k = 4 or k = -4 -/
theorem vector_collinearity (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V) (k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hnc : ¬ ∃ (r : ℝ), a = r • b)
  (hcol : ∃ (t : ℝ), 8 • a + k • b = t • (k • a + 2 • b)) :
  k = 4 ∨ k = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l1257_125743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l1257_125761

/-- The ellipse (x^2)/2 + y^2 = 1 -/
def Ellipse : Set (Real × Real) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- The foci of the ellipse -/
def F₁ : Real × Real := (-1, 0)
def F₂ : Real × Real := (1, 0)

/-- A line that intersects the ellipse at two distinct points -/
structure IntersectingLine where
  slope : Real
  yIntercept : Real
  intersectionPoints : Fin 2 → Real × Real
  distinctPoints : intersectionPoints 0 ≠ intersectionPoints 1
  onEllipse : ∀ i, intersectionPoints i ∈ Ellipse
  notThroughF₁ : ∀ x, (slope * x + yIntercept ≠ 0) ∨ (x ≠ -1)

/-- The distance from F₂ to the line -/
noncomputable def distanceToF₂ (l : IntersectingLine) : Real :=
  abs (l.slope * F₂.1 + l.yIntercept - F₂.2) / Real.sqrt (1 + l.slope^2)

/-- The slopes form an arithmetic sequence -/
def slopesInArithmeticSequence (l : IntersectingLine) : Prop :=
  let s₁ := (l.intersectionPoints 0).2 / ((l.intersectionPoints 0).1 + 1)
  let s₂ := l.slope
  let s₃ := (l.intersectionPoints 1).2 / ((l.intersectionPoints 1).1 + 1)
  s₁ + s₃ = 2 * s₂

/-- The main theorem -/
theorem distance_range (l : IntersectingLine) 
  (h : slopesInArithmeticSequence l) :
  Real.sqrt 3 < distanceToF₂ l ∧ distanceToF₂ l < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l1257_125761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_proof_l1257_125709

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 8 = 0

/-- The circumference of the circle -/
noncomputable def circle_circumference : ℝ := 2 * Real.sqrt 2 * Real.pi

/-- Theorem stating the relationship between the circle equation and its standard form,
    and the correctness of the calculated circumference -/
theorem circle_circumference_proof :
  ∃ (r : ℝ), (∀ (x y : ℝ), circle_equation x y ↔ (x - 1)^2 + (y + 3)^2 = r^2) ∧
  circle_circumference = 2 * Real.pi * r := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_proof_l1257_125709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1257_125774

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - 2 * x

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), max = 1/8 ∧ ∀ (x : ℝ), x ≥ 0 → f x ≤ max :=
by
  -- Use 1/8 as the maximum value
  use 1/8
  constructor
  · -- Prove that max = 1/8
    rfl
  · -- Prove that for all x ≥ 0, f(x) ≤ 1/8
    sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1257_125774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_cost_18_pounds_l1257_125721

/-- The cost of buying oranges given a specific pricing structure -/
noncomputable def orange_cost (regular_price : ℝ) (regular_weight : ℝ) (discount_price : ℝ) (threshold : ℝ) (total_weight : ℝ) : ℝ :=
  let regular_rate := regular_price / regular_weight
  let regular_cost := min threshold total_weight * regular_rate
  let discount_cost := max 0 (total_weight - threshold) * discount_price
  regular_cost + discount_cost

/-- Theorem stating the cost of buying 18 pounds of oranges under the given pricing structure -/
theorem orange_cost_18_pounds :
  orange_cost 6 6 0.90 10 18 = 17.20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_cost_18_pounds_l1257_125721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1257_125741

/-- The parabola y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- Point A -/
def A : ℝ × ℝ := (2, 0)

/-- Point B -/
def B : ℝ × ℝ := (7, 3)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating the minimum value of AP + BP -/
theorem min_distance_sum : 
  ∀ P ∈ Parabola, distance A P + distance B P ≥ 3 * Real.sqrt 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1257_125741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_double_fact_ratio_5_l1257_125769

-- Define the double factorial
def doubleFact : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * doubleFact n

-- Define the sum
def sumDoubleFactRatio (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => (doubleFact (2*i + 1) : ℚ) / (doubleFact (2*i + 2)))

-- Theorem statement
theorem sum_double_fact_ratio_5 :
  sumDoubleFactRatio 5 = 437 / 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_double_fact_ratio_5_l1257_125769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1257_125764

/-- A parabola passing through specific points -/
structure Parabola where
  b : ℝ
  c : ℝ
  passes_through_1_0 : 1 + b + c = 0
  passes_through_0_neg3 : c = -3

/-- The analytical expression of the parabola -/
noncomputable def analytical_expression (p : Parabola) : ℝ → ℝ := fun x ↦ x^2 + p.b * x + p.c

/-- The vertex of the parabola -/
noncomputable def vertex (p : Parabola) : ℝ × ℝ := (-p.b / 2, -(p.b^2 / 4 - p.c))

theorem parabola_properties (p : Parabola) :
  (∀ x, analytical_expression p x = x^2 + 2*x - 3) ∧
  vertex p = (-1, -4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1257_125764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_even_l1257_125770

def total_cards : ℕ := 9
def even_cards : ℕ := 4
def odd_cards : ℕ := 5

theorem prob_at_least_one_even :
  (Nat.choose total_cards 2 - Nat.choose odd_cards 2) / Nat.choose total_cards 2 = 13 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_even_l1257_125770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1257_125745

variable (m x y : ℝ)

def system_equations (m x y : ℝ) : Prop :=
  (2 * x + y = 4 * m) ∧ (x + 2 * y = 2 * m + 1)

theorem problem_solution :
  system_equations m x y →
  ((x + y = 1) → (m = 1/3)) ∧
  ((x - y ≥ -1 ∧ x - y ≤ 5) → (m ≥ 0 ∧ m ≤ 3)) ∧
  ((x - y ≥ -1 ∧ x - y ≤ 5) →
    ((0 ≤ m ∧ m ≤ 3/2 → |m + 2| + |2*m - 3| = 5 - m) ∧
     (3/2 < m ∧ m ≤ 3 → |m + 2| + |2*m - 3| = 3*m - 1))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1257_125745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_l1257_125708

-- Define the circle
def myCircle (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 6 = 0

-- Define the line
def myLine (x y : ℝ) : Prop := x + y - 8 = 0

-- State the theorem
theorem distance_difference :
  ∃ (dmax dmin : ℝ),
    (∀ (x y : ℝ), myCircle x y → 
      ∃ (d : ℝ), (∀ (x' y' : ℝ), myLine x' y' → d ≤ Real.sqrt ((x - x')^2 + (y - y')^2)) ∧
                 (∃ (x' y' : ℝ), myLine x' y' ∧ d = Real.sqrt ((x - x')^2 + (y - y')^2))) ∧
    (∀ (x y : ℝ), myCircle x y → 
      ∃ (d : ℝ), (∀ (x' y' : ℝ), myLine x' y' → Real.sqrt ((x - x')^2 + (y - y')^2) ≤ d) ∧
                 (∃ (x' y' : ℝ), myLine x' y' ∧ d = Real.sqrt ((x - x')^2 + (y - y')^2))) ∧
    dmax - dmin = 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_l1257_125708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_19_l1257_125732

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem tenth_term_is_19 : arithmetic_sequence 1 2 10 = 19 := by
  unfold arithmetic_sequence
  norm_num

#eval arithmetic_sequence 1 2 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_19_l1257_125732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_b_investment_l1257_125789

/-- Calculates B's investment in a partnership given the investments of A and C, 
    the total profit, and A's share of the profit. -/
theorem calculate_b_investment 
  (a_investment : ℝ) 
  (c_investment : ℝ) 
  (total_profit : ℝ) 
  (a_profit_share : ℝ) 
  (h1 : a_investment = 6300)
  (h2 : c_investment = 10500)
  (h3 : total_profit = 12700)
  (h4 : a_profit_share = 3810)
  : (total_profit * a_investment) / a_profit_share - a_investment - c_investment = 13702.36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_b_investment_l1257_125789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_fixed_point_l1257_125738

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- The locus of points P satisfying the distance condition -/
def locus : Set Point := {P | P.x^2 = 4 * P.y}

/-- The point A -/
def A : Point := ⟨0, 1⟩

/-- The point Q -/
def Q : Point := ⟨0, 2⟩

/-- The fixed point R -/
def R : Point := ⟨0, -2⟩

/-- Distance from a point to a horizontal line -/
def distToHorizontalLine (P : Point) (y : ℝ) : ℝ :=
  |P.y - y|

/-- Distance between two points -/
noncomputable def distBetweenPoints (P1 P2 : Point) : ℝ :=
  Real.sqrt ((P1.x - P2.x)^2 + (P1.y - P2.y)^2)

/-- Theorem stating the locus equation and the existence of point R -/
theorem locus_and_fixed_point :
  (∀ P, P ∈ locus ↔ distToHorizontalLine P (-3) = distBetweenPoints P A + 2) ∧
  (∀ l : Line, ∃ M N : Point,
    M ∈ locus ∧ N ∈ locus ∧
    M.y = l.m * M.x + l.b ∧ N.y = l.m * N.x + l.b ∧
    l.b = Q.y ∧
    (M.y - R.y) / (M.x - R.x) + (N.y - R.y) / (N.x - R.x) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_fixed_point_l1257_125738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l1257_125744

-- Define set M
def M : Set ℝ := {x | x / (x - 2) ≤ 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = -x^2 + 3}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l1257_125744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_symmetry_l1257_125735

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the line y = kx
def my_line (k x y : ℝ) : Prop := y = k * x

-- Define the symmetry line
def my_symmetry_line (b x y : ℝ) : Prop := 2 * x + y + b = 0

-- Theorem statement
theorem intersection_symmetry (k b : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    my_circle x₁ y₁ ∧ my_circle x₂ y₂ ∧
    my_line k x₁ y₁ ∧ my_line k x₂ y₂ ∧
    (∃ x_sym y_sym : ℝ, my_symmetry_line b x_sym y_sym ∧
      (x_sym - x₁) = (x₂ - x_sym) ∧
      (y_sym - y₁) = (y₂ - y_sym))) →
  k = 1/2 ∧ b = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_symmetry_l1257_125735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_has_property_P_infinitely_many_composite_with_property_P_l1257_125706

def has_property_P (n : ℕ) : Prop :=
  ∀ a : ℕ, n ∣ (a^n - 1 : ℕ) → n^2 ∣ (a^n - 1 : ℕ)

theorem prime_has_property_P (p : ℕ) (hp : Nat.Prime p) : has_property_P p := by
  sorry

theorem infinitely_many_composite_with_property_P :
  ∃ S : Set ℕ, (∀ n ∈ S, ¬Nat.Prime n ∧ has_property_P n) ∧ Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_has_property_P_infinitely_many_composite_with_property_P_l1257_125706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_adults_is_two_l1257_125781

/-- Represents a group of people at a restaurant --/
structure RestaurantGroup where
  num_children : ℕ
  meal_cost : ℕ
  total_bill : ℕ

/-- Calculates the number of adults in a group --/
def num_adults (g : RestaurantGroup) : ℕ :=
  (g.total_bill - g.num_children * g.meal_cost) / g.meal_cost

/-- Theorem stating that the number of adults in the group is 2 --/
theorem num_adults_is_two (g : RestaurantGroup) 
  (h1 : g.num_children = 5)
  (h2 : g.meal_cost = 3)
  (h3 : g.total_bill = 21) : 
  num_adults g = 2 := by
  sorry

#eval num_adults { num_children := 5, meal_cost := 3, total_bill := 21 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_adults_is_two_l1257_125781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l1257_125788

noncomputable def z (a : ℝ) : ℂ := (1 + 2*Complex.I) / (1 - Complex.I) + a

theorem purely_imaginary_condition (a : ℝ) :
  (z a).re = 0 → a = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l1257_125788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_count_l1257_125705

/-- A point in the 2D plane -/
structure Point where
  x : Int
  y : Int

/-- Determines if a point is on the line segment between two given points -/
def isOnLineSegment (p q r : Point) : Prop :=
  let dx := q.x - p.x
  let dy := q.y - p.y
  (r.x - p.x) * dy == (r.y - p.y) * dx ∧
  min p.x q.x ≤ r.x ∧ r.x ≤ max p.x q.x ∧
  min p.y q.y ≤ r.y ∧ r.y ≤ max p.y q.y

/-- The set of all lattice points on the line segment between two given points -/
def latticePointsOnSegment (p q : Point) : Set Point :=
  {r : Point | isOnLineSegment p q r}

theorem lattice_points_count :
  let p := Point.mk 3 17
  let q := Point.mk 48 281
  ∃ (s : Finset Point), s.card = 4 ∧ ∀ x, x ∈ s ↔ x ∈ latticePointsOnSegment p q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_count_l1257_125705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inclination_theorem_l1257_125729

theorem angle_inclination_theorem (θ : ℝ) :
  Real.tan θ = -2 →
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 1/3 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inclination_theorem_l1257_125729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_rectangle_l1257_125784

/-- Represents a rectangle with given perimeter and one fixed side --/
structure Rectangle where
  perimeter : ℝ
  fixed_side : ℝ

/-- Calculates the area of a rectangle given its perimeter and one fixed side --/
noncomputable def area (rect : Rectangle) : ℝ :=
  let other_side := (rect.perimeter / 2) - rect.fixed_side
  rect.fixed_side * other_side

/-- Theorem: The maximum area of a rectangle with perimeter 30 and one side 7 is 56 --/
theorem max_area_rectangle :
  ∃ (rect : Rectangle), rect.perimeter = 30 ∧ rect.fixed_side = 7 ∧
  ∀ (other : Rectangle), other.perimeter = 30 ∧ other.fixed_side = 7 →
  area other ≤ area rect ∧ area rect = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_rectangle_l1257_125784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_m_range_l1257_125737

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 ≤ x ∧ x ≤ m + 1}

-- State the theorem
theorem subset_implies_m_range (m : ℝ) : B m ⊆ A → m ∈ Set.Ici (-1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_m_range_l1257_125737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l1257_125725

-- Define a power function
noncomputable def powerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_value :
  ∃ α : ℝ, (powerFunction α 2 = Real.sqrt 2 / 2) ∧ (powerFunction α 4 = 1 / 2) := by
  -- Introduce α and prove its existence
  use (-1/2 : ℝ)
  
  -- Split the conjunction
  constructor

  -- Prove the first part: powerFunction (-1/2) 2 = √2 / 2
  · simp [powerFunction]
    -- The rest of the proof would go here
    sorry

  -- Prove the second part: powerFunction (-1/2) 4 = 1 / 2
  · simp [powerFunction]
    -- The rest of the proof would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l1257_125725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_omega_l1257_125730

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x + Real.pi / 4)

theorem smallest_positive_omega :
  ∃ ω : ℝ, ω > 0 ∧
  (∀ x, g ω x = g ω (-x)) ∧
  (∀ ω' : ℝ, ω' > 0 ∧ (∀ x, g ω' x = g ω' (-x)) → ω ≤ ω') ∧
  ω = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_omega_l1257_125730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l1257_125780

/-- The side length of a regular hexagon in centimeters -/
def hexagon_side : ℚ := 6

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The perimeter of the regular hexagon -/
def hexagon_perimeter : ℚ := hexagon_side * hexagon_sides

/-- The side length of the square -/
def square_side : ℚ := hexagon_perimeter / square_sides

theorem square_side_length : square_side = 9 := by
  -- Unfold definitions
  unfold square_side
  unfold hexagon_perimeter
  unfold hexagon_side
  unfold hexagon_sides
  unfold square_sides
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl

#eval square_side

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l1257_125780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anne_wandering_time_l1257_125751

/-- Calculates the time taken for a journey given distance and speed -/
noncomputable def time_taken (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- Proves that Anne's wandering time is 1.5 hours -/
theorem anne_wandering_time :
  let distance : ℝ := 3.0
  let speed : ℝ := 2.0
  time_taken distance speed = 1.5 := by
  -- Unfold the definition of time_taken
  unfold time_taken
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_anne_wandering_time_l1257_125751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geoff_sneaker_spending_l1257_125765

theorem geoff_sneaker_spending (total : ℝ) (tuesday_factor : ℝ) (wednesday_factor : ℝ)
  (h_total : total = 600)
  (h_tuesday : tuesday_factor = 4)
  (h_wednesday : wednesday_factor = 5) :
  total / (1 + tuesday_factor + wednesday_factor) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geoff_sneaker_spending_l1257_125765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_odd_function_l1257_125782

noncomputable def determinant (a₁ a₂ a₃ a₄ : ℝ) : ℝ := a₁ * a₄ - a₂ * a₃

noncomputable def f (x : ℝ) : ℝ := determinant (-Real.sin x) (Real.cos x) 1 (-Real.sqrt 3)

noncomputable def g (m x : ℝ) : ℝ := 2 * Real.sin (x + m - Real.pi / 6)

def is_odd_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem min_shift_for_odd_function :
  (∃ m : ℝ, m > 0 ∧ is_odd_function (g m) ∧ ∀ m' > 0, is_odd_function (g m') → m ≤ m') →
  ∃ m : ℝ, m > 0 ∧ is_odd_function (g m) ∧ ∀ m' > 0, is_odd_function (g m') → m ≤ m' ∧ m = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_odd_function_l1257_125782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1257_125783

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

noncomputable def asymptote_slope (a b : ℝ) : ℝ :=
  b / a

def focus (c : ℝ) : ℝ × ℝ :=
  (-c, 0)

theorem hyperbola_properties (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ a > b ∧
  asymptote_slope a b = 2 ∧
  (∃ (x y : ℝ), y = -2*x - 10 ∧ (x, y) = focus c) ∧
  c^2 = a^2 + b^2 →
  a^2 = 5 ∧ b^2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1257_125783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_shift_l1257_125767

theorem sine_graph_shift (x : ℝ) :
  Real.sin (2 * (x - π / 6) + π / 3) = Real.sin (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_shift_l1257_125767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_on_reals_l1257_125701

/-- The function f(x) defined in terms of a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / Real.log (2^x + 4 * 2^(-x) - a)

/-- Theorem stating that for f to be defined on all real numbers, a must be less than 3 -/
theorem f_defined_on_reals (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_on_reals_l1257_125701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_perpendicular_point_l1257_125768

/-- Given points A, B, and C in a rectangular coordinate system with origin O -/
def A : ℝ × ℝ := (2, 5)
def B : ℝ × ℝ := (3, 1)
def C (x : ℝ) : ℝ × ℝ := (x, 3)

/-- Vector from point P to point Q -/
def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Three points form a triangle if and only if they are not collinear -/
def form_triangle (P Q R : ℝ × ℝ) : Prop :=
  dot_product (vector P Q) (vector Q R) ≠ 0

/-- Point M is on line OC -/
def on_line_OC (M : ℝ × ℝ) (x : ℝ) : Prop :=
  ∃ l : ℝ, M = (l * x, l * 3)

/-- Vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  dot_product v w = 0

theorem triangle_condition (x : ℝ) :
  form_triangle A B (C x) ↔ x ≠ 5/2 := by sorry

theorem perpendicular_point (M : ℝ × ℝ) :
  on_line_OC M 6 ∧ perpendicular (vector M A) (vector M B) ↔
  M = (2, 1) ∨ M = (22/5, 11/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_perpendicular_point_l1257_125768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_premium_ticket_revenue_l1257_125728

theorem premium_ticket_revenue : 
  ∀ (p s x : ℕ),
  p + s = 200 →
  p > 0 →
  s > 0 →
  x > 0 →
  (3/2 : ℚ) * (x : ℚ) * (p : ℚ) + (x : ℚ) * (s : ℚ) = 3500 →
  (3/2 : ℚ) * (x : ℚ) * (p : ℚ) = 1200 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_premium_ticket_revenue_l1257_125728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palace_to_airport_distance_l1257_125742

/-- Represents the travel scenario of Emir Ben Sidi Mohammed --/
structure TravelScenario where
  distance : ℝ  -- Distance in km
  speed : ℝ     -- Speed in km/h
  time : ℝ      -- Time in hours

/-- The travel time given a change in speed --/
noncomputable def time_with_speed_change (scenario : TravelScenario) (speed_change : ℝ) : ℝ :=
  scenario.distance / (scenario.speed + speed_change)

/-- Theorem stating the distance between the palace and the airport --/
theorem palace_to_airport_distance :
  ∃ (scenario : TravelScenario),
    scenario.distance = 20 ∧
    scenario.speed > 0 ∧
    time_with_speed_change scenario 20 = scenario.time - 1/30 ∧
    time_with_speed_change scenario (-20) = scenario.time + 1/20 :=
by
  -- The proof is omitted for now
  sorry

#eval "Palace to airport distance theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_palace_to_airport_distance_l1257_125742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_cars_count_l1257_125746

/-- Represents the car rental problem --/
structure CarRental where
  redCars : ℕ
  whiteCars : ℕ
  redCarRate : ℕ
  whiteCarRate : ℕ
  rentalTime : ℕ
  totalEarnings : ℕ

/-- Calculates the total earnings from car rentals --/
def totalEarnings (cr : CarRental) : ℕ :=
  cr.redCars * cr.redCarRate * cr.rentalTime + cr.whiteCars * cr.whiteCarRate * cr.rentalTime

/-- Theorem stating that given the conditions, there are 2 white cars --/
theorem white_cars_count (cr : CarRental) 
  (h1 : cr.redCars = 3)
  (h2 : cr.redCarRate = 3)
  (h3 : cr.whiteCarRate = 2)
  (h4 : cr.rentalTime = 180)
  (h5 : cr.totalEarnings = 2340)
  (h6 : totalEarnings cr = cr.totalEarnings) :
  cr.whiteCars = 2 := by
  sorry

-- Remove the #eval line as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_cars_count_l1257_125746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1257_125796

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x else -x^2 - 4*x

theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧  -- f is odd
  (∀ x : ℝ, x ≥ 0 → f x = x^2 - 4*x) →  -- given condition
  (f (-3) + f (-2) + f 3 = 4) ∧  -- part 1
  (∀ x : ℝ, x < 0 → f x = -x^2 - 4*x) ∧  -- part 2
  (StrictMonoOn f (Set.Iio (-2))) ∧  -- monotonic on (-∞, -2)
  (StrictMonoOn f (Set.Ioi 2)) :=  -- monotonic on (2, +∞)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1257_125796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twice_x_plus_third_y_correct_l1257_125791

/-- The algebraic expression for "twice of x plus one third of y" -/
noncomputable def twice_x_plus_third_y (x y : ℝ) : ℝ := 2 * x + (1 / 3) * y

/-- Theorem stating that the algebraic expression for "twice of x plus one third of y" is correct -/
theorem twice_x_plus_third_y_correct (x y : ℝ) : 
  twice_x_plus_third_y x y = 2 * x + (1 / 3) * y := by
  -- Unfold the definition of twice_x_plus_third_y
  unfold twice_x_plus_third_y
  -- The equality now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twice_x_plus_third_y_correct_l1257_125791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_probability_above_x_axis_l1257_125753

/-- A trapezoid with vertices P, Q, R, and S -/
structure Trapezoid where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ

/-- The area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  ((t.Q.1 - t.P.1) + (t.R.1 - t.S.1)) * (t.P.2 - t.S.2) / 2

/-- The area of the part of the trapezoid above the x-axis -/
noncomputable def areaAboveXAxis (t : Trapezoid) : ℝ :=
  (t.Q.1 - t.P.1) * t.P.2

/-- The probability of a point being above the x-axis -/
noncomputable def probabilityAboveXAxis (t : Trapezoid) : ℝ :=
  areaAboveXAxis t / trapezoidArea t

theorem trapezoid_probability_above_x_axis :
  let t : Trapezoid := {
    P := (-3, 3),
    Q := (3, 3),
    R := (5, -1),
    S := (-5, -1)
  }
  probabilityAboveXAxis t = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_probability_above_x_axis_l1257_125753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l1257_125758

/-- Proves that given specific interest conditions, if the difference between compound
    and simple interest is 51, then the principal sum is 5100. -/
theorem interest_difference_implies_principal : ∀ (P : ℝ),
  (P * (1 + 10/100)^2 - P) - (P * 10 * 2 / 100) = 51 → P = 5100 := by
  intro P
  intro h
  -- The proof steps would go here
  sorry

#check interest_difference_implies_principal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l1257_125758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l1257_125766

/-- A sequence of digits satisfying the problem conditions -/
def ValidSequence : Type := List Nat

/-- Check if a sequence of digits is valid according to the problem conditions -/
def isValid (seq : ValidSequence) : Prop :=
  ∀ i j, i < j → j < seq.length →
    (seq.get! i ≠ seq.get! (i+1)) ∧  -- Adjacent digits are different
    (∀ k l, k < l → l < seq.length - 1 →
      (seq.get! k, seq.get! (k+1)) ≠ (seq.get! i, seq.get! (i+1)))  -- No repeated adjacent pairs

/-- The set of digits that can be used -/
def Digits : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- All digits in the sequence are from the set of valid digits -/
def usesValidDigits (seq : ValidSequence) : Prop :=
  ∀ i, i < seq.length → seq.get! i ∈ Digits

theorem max_sequence_length :
  ∃ (seq : ValidSequence), isValid seq ∧ usesValidDigits seq ∧
    (∀ (other : ValidSequence), isValid other ∧ usesValidDigits other →
      other.length ≤ seq.length) ∧
    seq.length = 73 := by
  sorry

#check max_sequence_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l1257_125766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_g_increasing_l1257_125793

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x
def g (a : ℝ) (x : ℝ) : ℝ := (2 - a) * x^3

-- State the theorem
theorem f_decreasing_implies_g_increasing (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  (∀ x y : ℝ, x < y → g a x < g a y) ∧
  ¬(∀ a : ℝ, (∀ x y : ℝ, x < y → g a x < g a y) → (∀ x y : ℝ, x < y → f a x > f a y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_g_increasing_l1257_125793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_numbers_l1257_125755

/-- A function that returns the list of digits of a natural number -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- A function that checks if a natural number consists only of the digit 1 -/
def allOnes (n : ℕ) : Prop :=
  ∀ d ∈ digits n, d = 1

theorem existence_of_special_numbers : ∃ (a b : ℕ), 
  (digits a).Perm (digits b) ∧ 
  allOnes (a - b) := by
  sorry

#check existence_of_special_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_numbers_l1257_125755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1257_125703

theorem remainder_problem (y : ℕ) (h : (7 * y) % 31 = 1) : (17 + 2 * y) % 31 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1257_125703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_even_function_l1257_125762

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The decreasing interval of a function f: ℝ → ℝ -/
def DecreasingInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

/-- Given function f(x) = kx^2 + (k-1)x + 3 -/
def f (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + (k - 1) * x + 3

theorem decreasing_interval_of_even_function (k : ℝ) :
  EvenFunction (f k) →
  ∀ x, x ≤ 0 → ∀ y, x < y ∧ y ≤ 0 → (f k) y < (f k) x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_even_function_l1257_125762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_count_l1257_125792

theorem congruent_count : ∃ (count : Nat), count = 111 ∧ 
  count = (Finset.filter (fun x => x % 9 = 7) (Finset.range 1000)).card := by
  use 111
  constructor
  · rfl
  · sorry

#eval (Finset.filter (fun x => x % 9 = 7) (Finset.range 1000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_count_l1257_125792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_amount_l1257_125740

/-- Represents the composition of the initial mixture -/
structure InitialMixture where
  nacl : ℝ
  kcl : ℝ
  sugar : ℝ
  water : ℝ

/-- Represents the additional salts added to the mixture -/
structure AddedSalts where
  nacl : ℝ
  kcl : ℝ

/-- Calculates the initial amount of mixture given the composition, added salts, and new salt content -/
noncomputable def calculate_initial_amount (initial : InitialMixture) (added : AddedSalts) (new_salt_content : ℝ) : ℝ :=
  let total_added := added.nacl + added.kcl
  (new_salt_content * total_added - added.nacl - added.kcl) / (new_salt_content - initial.nacl - initial.kcl)

/-- Theorem stating that the initial amount of mixture is 2730 grams -/
theorem initial_mixture_amount
  (initial : InitialMixture)
  (added : AddedSalts)
  (h1 : initial.nacl = 0.15)
  (h2 : initial.kcl = 0.30)
  (h3 : initial.sugar = 0.35)
  (h4 : initial.water = 0.20)
  (h5 : added.nacl = 50)
  (h6 : added.kcl = 80)
  (h7 : calculate_initial_amount initial added 0.475 = 2730) :
  calculate_initial_amount initial added 0.475 = 2730 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_amount_l1257_125740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1257_125747

/-- Represents the number of boys in the group -/
def boys : ℕ := sorry

/-- Represents the number of girls in the group -/
def girls : ℕ := sorry

/-- The minimum number of slices a boy can eat -/
def boyMin : ℕ := 6

/-- The maximum number of slices a boy can eat -/
def boyMax : ℕ := 7

/-- The minimum number of slices a girl can eat -/
def girlMin : ℕ := 2

/-- The maximum number of slices a girl can eat -/
def girlMax : ℕ := 3

/-- The number of slices in a pizza -/
def slicesPerPizza : ℕ := 12

/-- Four pizzas are never enough -/
axiom fourPizzasNotEnough : boyMin * boys + girlMin * girls > 4 * slicesPerPizza

/-- Five pizzas always have leftovers -/
axiom fivePizzasHaveLeftovers : boyMax * boys + girlMax * girls < 5 * slicesPerPizza

/-- The theorem stating that the only solution is 8 boys and 1 girl -/
theorem unique_solution : boys = 8 ∧ girls = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1257_125747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coles_average_speed_home_l1257_125759

noncomputable def average_speed_to_work : ℝ := 75
noncomputable def total_round_trip_time : ℝ := 2
noncomputable def time_to_work : ℝ := 70 / 60

theorem coles_average_speed_home (distance_to_work : ℝ) :
  distance_to_work = average_speed_to_work * time_to_work →
  distance_to_work / (total_round_trip_time - time_to_work) = 75 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coles_average_speed_home_l1257_125759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_purchasing_problem_l1257_125718

-- Define the types and variables
def basketball_price : ℕ → ℕ := sorry
def soccer_price : ℕ → ℕ := sorry
def basketball_sell_price : ℕ := 150
def soccer_sell_price : ℕ := 110

-- Define the theorem
theorem mall_purchasing_problem 
  (h1 : ∀ x, basketball_price x = soccer_price x + 30)
  (h2 : ∃ q, 360 / soccer_price q = 480 / basketball_price q)
  (h3 : ∀ m, ∃ n, n = m / 3 + 10)
  (h4 : ∀ m n, (basketball_sell_price - basketball_price m) * m + 
                (soccer_sell_price - soccer_price m) * n > 1300)
  (h5 : ∀ n, 120 * n + 90 * (100 - n) ≤ 10350)
  (h6 : ∀ n, n ≥ 43 ∧ n ≤ 100) :
  ∃ (soccer_unit_price basketball_unit_price min_basketballs schemes max_profit_basketballs : ℕ),
    soccer_unit_price = 90 ∧ 
    basketball_unit_price = 120 ∧
    min_basketballs = 33 ∧
    schemes = 3 ∧
    max_profit_basketballs = 45 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_purchasing_problem_l1257_125718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_integer_solutions_l1257_125748

theorem three_integer_solutions (a : ℝ) : 
  (∃! (x₁ x₂ x₃ : ℤ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    (abs (abs (x₁ - 2) - 1) = a) ∧ 
    (abs (abs (x₂ - 2) - 1) = a) ∧ 
    (abs (abs (x₃ - 2) - 1) = a)) ↔ 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_integer_solutions_l1257_125748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l1257_125704

theorem triangle_angle_B (A B : ℝ) (a b : ℝ) : 
  Real.cos A = 13/14 → 7*a = 3*b → (B = π/3 ∨ B = 2*π/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l1257_125704
