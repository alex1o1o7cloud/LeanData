import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_l1112_111213

def is_gcd (p q r : Polynomial ℤ) : Prop :=
  ∃ (s t : Polynomial ℤ), s * p + t * q = r

def is_lcm (p q r : Polynomial ℤ) : Prop :=
  ∃ (s t : Polynomial ℤ), s * p = t * q ∧ s * p = r

theorem sum_abc (a b c : ℤ) :
  let p := Polynomial.X^2 + Polynomial.C a * Polynomial.X + Polynomial.C b
  let q := Polynomial.X^2 + Polynomial.C b * Polynomial.X + Polynomial.C c
  is_gcd p q (Polynomial.X + 1) →
  is_lcm p q (Polynomial.X^3 - 4*Polynomial.X^2 + Polynomial.X + 6) →
  a + b + c = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_l1112_111213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_distance_l1112_111274

/-- Proves that for a round trip flight with given conditions, the distance flown each way is 1500 miles -/
theorem round_trip_distance (total_time outbound_speed return_speed : ℝ) 
  (h_total_time : total_time = 8)
  (h_outbound_speed : outbound_speed = 300)
  (h_return_speed : return_speed = 500) :
  (total_time * outbound_speed * return_speed) / (outbound_speed + return_speed) = 1500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_distance_l1112_111274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_range_l1112_111283

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then -x + 3*a else x^2 - a*x + 1

theorem f_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ (0 ≤ a ∧ a ≤ 1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_range_l1112_111283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exponential_positivity_l1112_111234

theorem negation_of_exponential_positivity :
  (¬ ∀ x : ℝ, (2 : ℝ)^x > 0) ↔ (∃ x : ℝ, (2 : ℝ)^x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exponential_positivity_l1112_111234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1112_111254

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x * (x + 1)
  else -Real.exp (-x) * (-x + 1)

-- State the theorem
theorem f_properties :
  (∀ x, f x > 0 ↔ (x ∈ Set.Ioo (-1) 0 ∪ Set.Ioi 1)) ∧
  (∀ x₁ x₂, |f x₁ - f x₂| < 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1112_111254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l1112_111272

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + x else -3*x

-- State the theorem
theorem f_inequality_range (a : ℝ) :
  a * (f a - f (-a)) > 0 ↔ a < -2 ∨ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l1112_111272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l1112_111216

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def trapeziumArea (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- The theorem to be proved -/
theorem trapezium_other_side_length (t : Trapezium) 
    (h1 : t.side1 = 5)
    (h2 : t.height = 6)
    (h3 : t.area = 27)
    (h4 : trapeziumArea t = t.area) : 
  t.side2 = 4 := by
  sorry

#check trapezium_other_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l1112_111216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_value_l1112_111268

noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f x - m

theorem odd_function_m_value :
  ∃ (m : ℝ), ∀ (x : ℝ), x ≠ 0 → g m x = -g m (-x) → m = 2 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_value_l1112_111268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_two_maxima_l1112_111244

theorem min_omega_two_maxima (ω : ℝ) : 
  (ω > 0) →
  (∀ x, x ∈ Set.Icc 0 1 → ∃ y, y = Real.sin (ω * x)) →
  (∃ x₁ x₂, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ x₁ ≠ x₂ ∧ 
    (∀ x, x ∈ Set.Icc 0 1 → Real.sin (ω * x) ≤ Real.sin (ω * x₁) ∧ 
                             Real.sin (ω * x) ≤ Real.sin (ω * x₂))) →
  ω ≥ 5 * Real.pi / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_two_maxima_l1112_111244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_65_l1112_111271

/-- Represents the speed of a car given fuel efficiency and consumption data -/
noncomputable def car_speed (km_per_liter : ℝ) (gallons_consumed : ℝ) (hours_traveled : ℝ) 
  (liters_per_gallon : ℝ) (km_per_mile : ℝ) : ℝ :=
  let liters_consumed := gallons_consumed * liters_per_gallon
  let km_traveled := liters_consumed * km_per_liter
  let miles_traveled := km_traveled / km_per_mile
  miles_traveled / hours_traveled

/-- Theorem stating that the car's speed is 65 miles per hour -/
theorem car_speed_is_65 : 
  car_speed 40 3.9 5.7 3.8 1.6 = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_65_l1112_111271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1112_111294

open Real BigOperators

theorem trigonometric_equation_solution (x : ℝ) : 
  (∑ k in Finset.range 1010, sin ((2*k + 1) * x)) = (∑ k in Finset.range 1010, cos ((2*k + 1) * x)) ↔ 
  (∃ k : ℤ, (k % 1010 ≠ 0 ∧ x = k * π / 1010) ∨ 
   (∃ m : ℤ, x = (π + 4 * π * m) / 4040)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1112_111294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_72_equals_498_l1112_111222

partial def g : ℤ → ℤ
  | n => if n ≥ 500 then n - 4 else g (g (n + 5))

theorem g_72_equals_498 : g 72 = 498 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_72_equals_498_l1112_111222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l1112_111297

/-- The increase in area when changing a rectangular garden to a circular one with the same perimeter --/
theorem garden_area_increase (width : ℝ) (length : ℝ) (π : ℝ) : 
  width = 20 →
  length = 60 →
  π > 0 →
  let rectangular_area := width * length
  let perimeter := 2 * (width + length)
  let radius := perimeter / (2 * π)
  let circular_area := π * radius^2
  |circular_area - rectangular_area - 837.75| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l1112_111297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1112_111290

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ) + 1

theorem function_properties 
  (A ω φ : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : -π/2 ≤ φ ∧ φ ≤ π/2) 
  (h4 : ∀ x, f A ω φ x = f A ω φ (2*π/3 - x)) 
  (h5 : ∃ x, f A ω φ x = 3 ∧ ∀ y, f A ω φ y ≤ 3) 
  (h6 : ∀ x, f A ω φ x = 3 → f A ω φ (x + π) = 3) :
  (∃ T > 0, ∀ x, f A ω φ (x + T) = f A ω φ x ∧ 
    ∀ S, 0 < S ∧ S < T → ∃ y, f A ω φ (y + S) ≠ f A ω φ y) ∧
  (∀ x, f A ω φ x = 2 * Real.sin (2*x - π/6) + 1) ∧
  (∀ θ, f A ω φ (θ/2 + π/3) = 7/5 → Real.sin θ = 2*Real.sqrt 6/5 ∨ Real.sin θ = -2*Real.sqrt 6/5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1112_111290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closer_points_form_half_plane_l1112_111231

noncomputable section

structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

def distance (p1 p2 : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  ‖p1 - p2‖

def on_circle (p : EuclideanSpace ℝ (Fin 2)) (c : Circle) : Prop :=
  distance p c.center = c.radius

def half_plane (p1 p2 : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {p : EuclideanSpace ℝ (Fin 2) | distance p p1 < distance p p2}

theorem closer_points_form_half_plane (C : Circle) (B D : EuclideanSpace ℝ (Fin 2)) :
  on_circle B C → on_circle D C → B ≠ D →
  {A : EuclideanSpace ℝ (Fin 2) | distance A B < distance A D} = half_plane B D := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closer_points_form_half_plane_l1112_111231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_both_curves_l1112_111232

-- Define the curves and the tangent line
noncomputable def curve1 (x : ℝ) : ℝ := Real.log x + 2
noncomputable def curve2 (x : ℝ) : ℝ := Real.log (x + 1)
def tangentLine (k b x : ℝ) : ℝ := k * x + b

-- Define the tangency condition
def isTangent (f g : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = g x ∧ (deriv f) x = (deriv g) x

-- Theorem statement
theorem tangent_to_both_curves (k b : ℝ) :
  (∃ x₁ > 0, isTangent (curve1) (tangentLine k b) x₁) ∧
  (∃ x₂ > 0, isTangent (curve2) (tangentLine k b) x₂) →
  b = 1 - Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_both_curves_l1112_111232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_is_162_l1112_111252

/-- Given a coplanar arrangement of a square ABCD and a regular pentagon DAEFG sharing side DA,
    the exterior angle DEG measures 162°. -/
def exterior_angle_square_pentagon (angle_deg : ℝ) : Prop :=
  let square_interior_angle : ℝ := 90
  let pentagon_interior_angle : ℝ := 108
  let sum_adjacent_angles : ℝ := square_interior_angle + pentagon_interior_angle
  angle_deg = 360 - sum_adjacent_angles

#check exterior_angle_square_pentagon

/-- The exterior angle DEG measures 162°. -/
theorem exterior_angle_is_162 : exterior_angle_square_pentagon 162 := by
  unfold exterior_angle_square_pentagon
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_is_162_l1112_111252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1112_111288

theorem solve_exponential_equation : ∃ x : ℚ, (3 : ℝ) ^ (2 * x : ℝ) = Real.sqrt 243 ∧ x = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1112_111288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_prime_y_l1112_111299

/-- Sequence x defined recursively -/
def x : ℕ → ℕ → ℕ
  | a, 0 => a
  | a, n + 1 => 2 * x a n + 1

/-- Sequence y defined in terms of x -/
def y (a n : ℕ) : ℕ := 2^(x a n) - 1

/-- Definition of primality -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The main theorem statement -/
theorem largest_k_prime_y : 
  ∀ k : ℕ, k > 2 → 
  ¬∃ a : ℕ, a > 0 ∧ (∀ i : ℕ, i > 0 ∧ i ≤ k → is_prime (y a i)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_prime_y_l1112_111299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_configuration_radius_determination_l1112_111224

/-- Represents a cone on a table -/
structure Cone where
  baseRadius : ℝ

/-- Represents a truncated cone on a table -/
structure TruncatedCone where
  smallerBaseRadius : ℝ

/-- The configuration of cones on the table -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  truncatedCone : TruncatedCone
  touchingCones : Bool
  sharingSlantHeight : Bool

theorem cone_configuration_radius_determination 
  (config : ConeConfiguration) 
  (h1 : config.cone1.baseRadius = 2 * config.truncatedCone.smallerBaseRadius / 15)
  (h2 : config.cone2.baseRadius = 3 * config.truncatedCone.smallerBaseRadius / 15)
  (h3 : config.cone3.baseRadius = 10 * config.truncatedCone.smallerBaseRadius / 15)
  (h4 : config.truncatedCone.smallerBaseRadius = 15)
  (h5 : config.touchingCones = true)
  (h6 : config.sharingSlantHeight = true) :
  config.cone1.baseRadius = 58 / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_configuration_radius_determination_l1112_111224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_vector_expression_l1112_111242

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem max_value_of_vector_expression (a b c : V)
  (ha : ‖a‖ = 2)
  (hb : ‖b‖ = 3)
  (hc : ‖c‖ = 4) :
  ‖a - 3 • b‖^2 + ‖b - 3 • c‖^2 + ‖c - 3 • a‖^2 ≤ 377 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_vector_expression_l1112_111242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_innings_played_l1112_111212

/-- Represents the number of innings played by a cricket player. -/
def innings : ℕ := sorry

/-- Represents the current average runs per innings. -/
def current_average : ℝ := 36

/-- Represents the runs scored in the next innings. -/
def next_innings_runs : ℝ := 120

/-- Represents the increase in average after the next innings. -/
def average_increase : ℝ := 4

/-- Theorem stating that under the given conditions, the player has played 20 innings. -/
theorem innings_played : 
  (current_average * innings + next_innings_runs) / (innings + 1) = current_average + average_increase →
  innings = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_innings_played_l1112_111212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1112_111275

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.sin x - 1) / Real.sqrt (3 - 2 * Real.cos x - 2 * Real.sin x)

-- State the theorem
theorem f_range :
  ∀ y ∈ Set.range f,
  -1 ≤ y ∧ y ≤ 0 ∧
  ∃ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1112_111275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_minimum_distance_l1112_111200

/-- The circle equation: x^2 + y^2 - 4x - 4y + 7 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y + 7 = 0

/-- The line equation: y = kx -/
def line_equation (k x y : ℝ) : Prop :=
  y = k * x

/-- The minimum distance between P and Q is 2√2 - 1 -/
noncomputable def min_distance : ℝ := 2 * Real.sqrt 2 - 1

theorem circle_line_minimum_distance (k : ℝ) :
  (∃ x y : ℝ, circle_equation x y) →
  (∀ x y : ℝ, line_equation k x y → 
    ∃ xp yp : ℝ, circle_equation xp yp ∧ 
    ∀ xq yq : ℝ, line_equation k xq yq → 
    Real.sqrt ((xp - xq)^2 + (yp - yq)^2) ≥ min_distance) →
  (∃ xp yp xq yq : ℝ, circle_equation xp yp ∧ line_equation k xq yq ∧
    Real.sqrt ((xp - xq)^2 + (yp - yq)^2) = min_distance) →
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_minimum_distance_l1112_111200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_subset_with_property_l1112_111239

theorem exists_infinite_subset_with_property
  (f g : ℝ → ℝ) (h : ∀ x : ℝ, f x < g x) :
  ∃ S : Set ℝ, Set.Infinite S ∧ ∀ (x y : ℝ), x ∈ S → y ∈ S → f x < g y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_subset_with_property_l1112_111239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_distance_l1112_111247

noncomputable def initial_height : ℝ := 128
noncomputable def bounce_ratio : ℝ := 1/2
def num_bounces : ℕ := 8

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

noncomputable def total_distance (h : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  2 * geometric_sum h r n - h

theorem ball_bounce_distance :
  total_distance initial_height bounce_ratio num_bounces = 382 := by
  sorry

#eval num_bounces -- This line is added to ensure there's at least one computable definition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_distance_l1112_111247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_at_2000m_l1112_111205

/-- Atmospheric pressure function -/
noncomputable def atmospheric_pressure (a : ℝ) (x : ℝ) : ℝ := 100 * Real.exp (a * x)

/-- Theorem: Given the atmospheric pressure function and the condition at 1000m,
    the pressure at 2000m is 81 kPa -/
theorem pressure_at_2000m (a : ℝ) :
  atmospheric_pressure a 1000 = 90 →
  atmospheric_pressure a 2000 = 81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_at_2000m_l1112_111205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_proof_l1112_111256

-- Define the necessary structures and types
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the necessary propositions
def tangentInternally (c1 c2 : Circle) (p : Point) : Prop :=
  sorry

def diametricallyOpposite (c : Circle) (p1 p2 : Point) : Prop :=
  sorry

def chordTangentToCircle (c1 c2 : Circle) (p1 p2 p3 : Point) : Prop :=
  sorry

def isAngleBisector (t : Triangle) (p : Point) : Prop :=
  sorry

-- The main theorem
theorem angle_bisector_proof (c1 c2 : Circle) (A B C D : Point) :
  tangentInternally c1 c2 A →
  diametricallyOpposite c1 A B →
  chordTangentToCircle c1 c2 B C D →
  isAngleBisector (Triangle.mk A B C) D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_proof_l1112_111256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_coefficients_l1112_111289

/-- The sum of the squares of the coefficients of 6(x^5 + 4x^3 + 2) is 756. -/
theorem sum_of_squares_of_coefficients : (6^2 + 24^2 + 12^2 : ℕ) = 756 := by
  -- Expand the expression
  have h1 : 6^2 + 24^2 + 12^2 = 36 + 576 + 144 := by rfl
  
  -- Compute the sum
  have h2 : 36 + 576 + 144 = 756 := by rfl
  
  -- Combine the steps
  rw [h1, h2]

#eval 6^2 + 24^2 + 12^2  -- This will output 756

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_coefficients_l1112_111289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l1112_111296

theorem exponential_equation_solution : 
  ∃ x : ℝ, (3 : ℝ)^(x + 3) = (81 : ℝ)^x ∧ x = 1 := by 
  use 1
  constructor
  · simp
    norm_num
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l1112_111296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_same_domain_range_as_g_l1112_111255

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := Real.exp (Real.log x)

-- Define the domain and range of g
def domain_g : Set ℝ := {x : ℝ | x > 0}
def range_g : Set ℝ := Set.Ioi 0

-- State the theorem
theorem f_same_domain_range_as_g :
  (∀ x, x ∈ domain_g ↔ f x ≠ 0) ∧
  (Set.range f = range_g) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_same_domain_range_as_g_l1112_111255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cylinder_volume_ratio_l1112_111259

noncomputable section

/-- The volume of a cylinder formed by rolling a rectangle along its side -/
def cylinderVolume (width : ℝ) (height : ℝ) : ℝ :=
  Real.pi * (width / (2 * Real.pi))^2 * height

/-- The ratio of volumes of two cylinders formed from a rectangle -/
def cylinderVolumeRatio (length : ℝ) (width : ℝ) : ℝ :=
  max (cylinderVolume length width) (cylinderVolume width length) /
  min (cylinderVolume length width) (cylinderVolume width length)

theorem rectangle_cylinder_volume_ratio :
  cylinderVolumeRatio 12 7 = 56 / 33 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cylinder_volume_ratio_l1112_111259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_equal_area_l1112_111285

/-- The surface area of a sphere with radius r -/
noncomputable def sphereSurfaceArea (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- The curved surface area of a right circular cylinder with radius r and height h -/
noncomputable def cylinderSurfaceArea (r h : ℝ) : ℝ := 2 * Real.pi * r * h

theorem cylinder_sphere_equal_area (r_sphere r_cylinder h_cylinder : ℝ) :
  r_sphere = 8 →
  h_cylinder = 2 * r_cylinder →
  sphereSurfaceArea r_sphere = cylinderSurfaceArea r_cylinder h_cylinder →
  h_cylinder = 16 ∧ 2 * r_cylinder = 16 := by
  sorry

#check cylinder_sphere_equal_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_equal_area_l1112_111285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_smallest_n_with_some_but_not_all_divisors_l1112_111219

/-- 
Given a positive integer n, this function returns true if n^2 - 2n is divisible by some 
but not all integer values of k where 1 ≤ k ≤ n.
-/
def hasSomeButNotAllDivisors (n : ℕ) : Prop :=
  ∃ k₁ k₂ : ℕ, 1 ≤ k₁ ∧ k₁ ≤ n ∧ 1 ≤ k₂ ∧ k₂ ≤ n ∧ 
    (n^2 - 2*n) % k₁ = 0 ∧ 
    (n^2 - 2*n) % k₂ ≠ 0

/-- 
Theorem stating that 5 is the smallest positive integer n such that n^2 - 2n 
is divisible by some but not all integer values of k when 1 ≤ k ≤ n.
-/
theorem smallest_n_with_some_but_not_all_divisors : 
  (∀ m : ℕ, 0 < m → m < 5 → ¬(hasSomeButNotAllDivisors m)) ∧ 
  hasSomeButNotAllDivisors 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_smallest_n_with_some_but_not_all_divisors_l1112_111219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2015th_term_l1112_111220

def my_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧
  a 2 = 1/2 ∧
  ∀ n ≥ 2, a n * (a (n-1) + a (n+1)) = 2 * a (n+1) * a (n-1)

theorem sequence_2015th_term (a : ℕ → ℚ) (h : my_sequence a) : a 2015 = 1/2015 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2015th_term_l1112_111220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equiangular_star_inner_pentagon_perimeter_l1112_111210

/-- Given a closed five-link broken line forming an equiangular star with outer path length 1,
    the perimeter of the inner pentagon is √5 - 2. -/
theorem equiangular_star_inner_pentagon_perimeter :
  ∀ (outer_path : ℝ) (inner_perimeter : ℝ),
    outer_path = 1 →
    inner_perimeter = Real.sqrt 5 - 2 :=
by
  intros outer_path inner_perimeter h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equiangular_star_inner_pentagon_perimeter_l1112_111210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_minimum_at_one_f_less_than_x_plus_two_l1112_111273

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x + 2 / (a * x)

-- Theorem for part 1
theorem f_local_minimum_at_one :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x > 0 ∧ |x - 1| < ε → f 1 x ≥ f 1 1 ∧ f 1 1 = 3 :=
by sorry

-- Theorem for part 2
theorem f_less_than_x_plus_two (a : ℝ) :
  (∀ (x : ℝ), Real.exp (-1) < x ∧ x < Real.exp 1 → f a x < x + 2) ↔
  (a < 0 ∨ a ≥ 2 * Real.exp 1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_minimum_at_one_f_less_than_x_plus_two_l1112_111273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_l1112_111238

/-- Representation of a line in 3D space -/
structure Line :=
  (point : EuclideanSpace ℝ (Fin 3))
  (direction : EuclideanSpace ℝ (Fin 3))

/-- Representation of a plane in 3D space -/
structure Plane :=
  (point : EuclideanSpace ℝ (Fin 3))
  (normal : EuclideanSpace ℝ (Fin 3))

/-- Two lines are different -/
def lines_different (l m : Line) : Prop := l ≠ m

/-- Two planes are non-coincident -/
def planes_non_coincident (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line) (α : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def parallel_line_plane (l : Line) (α : Plane) : Prop := sorry

/-- A line is contained in a plane -/
def contained_in (l : Line) (α : Plane) : Prop := sorry

/-- Two planes are parallel -/
def parallel_planes (α β : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def perpendicular_planes (α β : Plane) : Prop := sorry

/-- Two lines are parallel -/
def parallel_lines (l m : Line) : Prop := sorry

/-- Two lines are perpendicular -/
def perpendicular_lines (l m : Line) : Prop := sorry

theorem spatial_relationships (l m : Line) (α β : Plane)
  (h1 : lines_different l m) (h2 : planes_non_coincident α β) :
  (parallel_planes α β → perpendicular l α → perpendicular l β) ∧
  ¬(parallel_lines l m → contained_in l α → contained_in m β → parallel_planes α β) ∧
  ¬(perpendicular m α → perpendicular_lines l m → parallel_line_plane l α) ∧
  ¬(perpendicular_planes α β → contained_in l α → contained_in m β → perpendicular_lines l m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_l1112_111238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1112_111243

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1 - Real.sin x) / (5 + 2 * Real.cos x)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc 0 (10/21) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1112_111243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_sum_and_product_l1112_111295

/-- For any positive integer n, there exists a list of n positive integers
    whose sum is a perfect square and whose product is a perfect cube. -/
theorem perfect_sum_and_product (n : ℕ) (hn : 0 < n) :
  ∃ (list : List ℕ),
    List.length list = n ∧
    (∃ (m : ℕ), list.sum = m^2) ∧
    (∃ (k : ℕ), list.prod = k^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_sum_and_product_l1112_111295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_trajectory_l1112_111217

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/16 + y^2/7 = 1

-- Define the distance ratio lambda
def distance_ratio (lambda : ℝ) (x y : ℝ) : Prop := (x^2 + y^2) * lambda^2 = 9*x^2/16 + 7

-- Define the trajectory of point M
def trajectory_M (lambda : ℝ) (x y : ℝ) : Prop :=
  if lambda = 3/4 then
    y^2 = 112/9 ∧ -4 ≤ x ∧ x ≤ 4
  else
    x^2 / (112 / (16*lambda^2 - 9)) + y^2 / (112 / (16*lambda^2)) = 1 ∧ -4 ≤ x ∧ x ≤ 4

theorem ellipse_and_trajectory :
  ∀ (lambda : ℝ), lambda > 0 →
    (∀ (x y : ℝ), ellipse_C x y ∧ distance_ratio lambda x y → trajectory_M lambda x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_trajectory_l1112_111217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operation_count_l1112_111250

/-- Represents a polynomial of degree n -/
def MyPolynomial (α : Type*) := List α

/-- Horner's method for polynomial evaluation -/
def horner_eval {α : Type*} [Ring α] (p : MyPolynomial α) (x : α) : α :=
  p.foldl (fun acc a => acc * x + a) 0

/-- Count of operations in Horner's method -/
def horner_op_count (n : ℕ) : ℕ := 2 * n

theorem horner_method_operation_count (n : ℕ) (α : Type*) [Ring α] (p : MyPolynomial α) (x : α) :
  p.length = n + 1 → horner_op_count n = n + n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operation_count_l1112_111250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_MAB_l1112_111279

-- Define the curves C1 and C2
noncomputable def C1 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 + 2 * Real.sin θ)
noncomputable def C2 (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := C1 (Real.pi / 3)
noncomputable def B : ℝ × ℝ := (C2 (Real.pi / 3) * Real.cos (Real.pi / 3), C2 (Real.pi / 3) * Real.sin (Real.pi / 3))

-- Define point M
def M : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem area_of_triangle_MAB :
  let d := 2 * Real.sin (Real.pi / 3)
  let ab_length := C2 (Real.pi / 3) - 4 * Real.sin (Real.pi / 3)
  (1/2) * d * ab_length = 3 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_MAB_l1112_111279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_must_score_perfect_on_all_remaining_tests_l1112_111269

/-- Represents the mathematics competition preparation scenario for John -/
structure MathCompetition where
  total_tests : ℕ
  goal_percentage : ℚ
  completed_tests : ℕ
  perfect_scores : ℕ

/-- Calculates the maximum number of remaining tests where John can score less than 100 -/
def max_non_perfect_scores (comp : MathCompetition) : ℕ :=
  let remaining_tests := comp.total_tests - comp.completed_tests
  let min_perfect_scores := Nat.ceil (comp.goal_percentage * comp.total_tests : ℚ)
  let additional_perfect_scores_needed := min_perfect_scores - comp.perfect_scores
  (remaining_tests - additional_perfect_scores_needed).max 0

/-- The theorem stating that John can't afford to score less than 100 on any remaining test -/
theorem john_must_score_perfect_on_all_remaining_tests 
  (comp : MathCompetition) 
  (h_total : comp.total_tests = 40)
  (h_goal : comp.goal_percentage = 85 / 100)
  (h_completed : comp.completed_tests = 36)
  (h_perfect : comp.perfect_scores = 30) :
  max_non_perfect_scores comp = 0 := by
  sorry

#eval max_non_perfect_scores {
  total_tests := 40,
  goal_percentage := 85 / 100,
  completed_tests := 36,
  perfect_scores := 30
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_must_score_perfect_on_all_remaining_tests_l1112_111269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1112_111282

/-- Calculates the speed of a train given its length and time to cross a pole --/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that a train with given length and crossing time has a specific speed --/
theorem train_speed_calculation :
  let length := (350 : ℝ) -- meters
  let time := (21 : ℝ) -- seconds
  let calculated_speed := train_speed length time
  ∃ ε > 0, |calculated_speed - 60| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1112_111282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_intersect_is_five_twelfths_l1112_111284

/-- A convex hexagon -/
structure ConvexHexagon where
  -- Add necessary properties here

/-- A diagonal in a convex hexagon -/
structure Diagonal (H : ConvexHexagon) where
  -- Add necessary properties here

/-- Predicate to check if two diagonals intersect within the hexagon -/
def intersect_within (H : ConvexHexagon) (d1 d2 : Diagonal H) : Prop :=
  sorry -- Define the condition for two diagonals to intersect within the hexagon

/-- The number of diagonals in a convex hexagon -/
def num_diagonals : ℕ := 9

/-- The number of pairs of diagonals that intersect within the hexagon -/
def num_intersecting_pairs : ℕ := 15

/-- The probability of two randomly chosen diagonals intersecting within the hexagon -/
def prob_intersect : ℚ :=
  num_intersecting_pairs / (num_diagonals.choose 2)

theorem prob_intersect_is_five_twelfths :
  prob_intersect = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_intersect_is_five_twelfths_l1112_111284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1112_111248

/-- Represents the time (in hours) it takes to fill or empty a cistern using a specific pipe -/
def PipeRate := ℝ

/-- Calculates the time to fill a cistern given the rates of all pipes -/
noncomputable def fillTime (fill_a fill_b fill_c empty_d empty_e : PipeRate) : ℝ :=
  360 / (30 + 24 + 18 - 20 - 15)

/-- Theorem stating that given the specific pipe rates, the cistern will be filled in 360/37 hours -/
theorem cistern_fill_time :
  ∀ (fill_a fill_b fill_c empty_d empty_e : PipeRate),
    fill_a = (12 : ℝ) → fill_b = (15 : ℝ) → fill_c = (20 : ℝ) → empty_d = (18 : ℝ) → empty_e = (24 : ℝ) →
    fillTime fill_a fill_b fill_c empty_d empty_e = 360 / 37 := by
  sorry

#eval (360 : ℚ) / 37

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1112_111248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_determination_of_V_l1112_111280

/-- An arithmetic sequence -/
noncomputable def arithmetic_sequence (b₁ : ℝ) (e : ℝ) (n : ℕ) : ℝ := b₁ + (n - 1 : ℝ) * e

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def U (b₁ : ℝ) (e : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * b₁ + (n - 1 : ℝ) * e) / 2

/-- Sum of U_k for k from 1 to n -/
noncomputable def V (b₁ : ℝ) (e : ℝ) (n : ℕ) : ℝ :=
  ((n : ℝ) * ((n : ℝ) + 1) * (3 * b₁ + ((n : ℝ) - 1) * e)) / 6

theorem unique_determination_of_V (b₁ : ℝ) (e : ℝ) :
  ∃ (u₂₀₂₀ : ℝ), U b₁ e 2020 = u₂₀₂₀ →
  ∀ (n : ℕ), n < 3030 → ¬∃! (v : ℝ), V b₁ e n = v :=
by
  sorry

#check unique_determination_of_V

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_determination_of_V_l1112_111280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_coloring_theorem_l1112_111240

/-- The set of points inside and on the border of a regular hexagon with side length 1 -/
def S : Set (ℝ × ℝ) := sorry

/-- A coloring function that assigns one of three colors to each point in S -/
def coloring (s : S) : Fin 3 := sorry

/-- The distance function between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hexagon_coloring_theorem :
  (∀ ε > 0, ∃ (c : S → Fin 3), ∀ (p q : S), c p = c q → distance p.val q.val < 3/2 + ε) ∧
  (∀ δ < 3/2, ∃ (p q : S), ∀ (c : S → Fin 3), c p = c q → distance p.val q.val ≥ δ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_coloring_theorem_l1112_111240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_decreasing_functions_l1112_111286

noncomputable def f₁ (x : ℝ) : ℝ := 1 / x
noncomputable def f₂ (x : ℝ) : ℝ := -abs x
noncomputable def f₃ (x : ℝ) : ℝ := -2 * x - 1

theorem strictly_decreasing_functions :
  ∀ (f : ℝ → ℝ), f = f₁ ∨ f = f₂ ∨ f = f₃ →
    ∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ < x₂ → f x₁ > f x₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_decreasing_functions_l1112_111286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_l1112_111225

/-- The function f(x) = -√3/3 * x^3 + 2 -/
noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 / 3 * x^3 + 2

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := -Real.sqrt 3 * x^2

/-- The slope angle of the tangent line to f(x) at x = 1 -/
noncomputable def slope_angle : ℝ := 2 * Real.pi / 3

theorem tangent_slope_angle :
  let slope := f' 1
  Real.arctan slope = slope_angle := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_l1112_111225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spirit_water_solution_cost_l1112_111262

/-- Represents the cost of a spirit-water solution -/
noncomputable def solution_cost (spirit_volume : ℝ) (water_volume : ℝ) (cost_per_pure_spirit : ℝ) : ℝ :=
  (spirit_volume / (spirit_volume + water_volume)) * cost_per_pure_spirit

/-- The theorem statement -/
theorem spirit_water_solution_cost :
  ∀ (cost_per_pure_spirit : ℝ),
  solution_cost 1 2 cost_per_pure_spirit = 49.99999999999999 →
  solution_cost 1 1 cost_per_pure_spirit = 75 :=
by
  intro cost_per_pure_spirit
  intro h
  sorry

#check spirit_water_solution_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spirit_water_solution_cost_l1112_111262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_theorem_l1112_111278

def ℕ_plus : Type := {n : Nat // n > 0}

def is_increasing (f : ℕ_plus → Nat) : Prop :=
  ∀ n m : ℕ_plus, n.val > m.val → f n ≥ f m

def satisfies_multiplication (f : ℕ_plus → Nat) : Prop :=
  ∀ n m : ℕ_plus, f ⟨n.val * m.val, Nat.mul_pos n.property m.property⟩ = f n + f m

theorem unique_function_theorem (f : ℕ_plus → Nat) 
  (h1 : is_increasing f) (h2 : satisfies_multiplication f) :
  ∀ n : ℕ_plus, f n = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_theorem_l1112_111278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l1112_111287

-- Define a, b, and c
noncomputable def a : ℝ := 2^(0.2 : ℝ)
noncomputable def b : ℝ := Real.log 2
noncomputable def c : ℝ := Real.log 2 / Real.log 0.3

-- Theorem statement
theorem magnitude_relationship : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l1112_111287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l1112_111215

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the focal length of a hyperbola
noncomputable def focal_length (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem hyperbola_focal_length :
  ∃ (a b : ℝ), a^2 = 3 ∧ b^2 = 1 ∧ focal_length a b = 4 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l1112_111215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nineteenth_term_equals_five_sqrt_three_l1112_111235

noncomputable def my_sequence (n : ℕ) : ℝ := Real.sqrt (3 + 4 * (n - 1))

theorem nineteenth_term_equals_five_sqrt_three :
  my_sequence 19 = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nineteenth_term_equals_five_sqrt_three_l1112_111235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_type_function_statements_l1112_111265

-- Definition of k-type function
def is_k_type_function (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∃ (m n : ℝ), m < n ∧ ∀ x ∈ Set.Icc m n, f x ∈ Set.Icc (k * m) (k * n)

-- Statement 1
def statement1 : Prop :=
  ¬∃ k, is_k_type_function (λ x ↦ 3 - 4 / x) k

-- Statement 2
def statement2 : Prop :=
  is_k_type_function (λ x ↦ -1/2 * x^2 + x) 3 → (∃ m n, m = -4 ∧ n = 0)

-- Statement 3
def statement3 : Prop :=
  is_k_type_function (λ x ↦ |3^x - 1|) 2 → (∃ m n, m + n = 1)

-- Statement 4
def statement4 : Prop :=
  ∀ a ≠ 0, is_k_type_function (λ x ↦ ((a^2 + a) * x - 1) / (a^2 * x)) 1 → 
    (∃ m n, n - m ≤ 2 * Real.sqrt 3 / 3 ∧ 
      ∀ m' n', n' - m' ≤ n - m)

-- Main theorem
theorem k_type_function_statements :
  ¬statement1 ∧ statement2 ∧ statement3 ∧ statement4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_type_function_statements_l1112_111265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_square_divisible_by_five_in_range_l1112_111218

theorem unique_square_divisible_by_five_in_range : 
  ∃! x : ℕ, ∃ y : ℕ, x = y * y ∧ x % 5 = 0 ∧ 50 < x ∧ x < 150 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_square_divisible_by_five_in_range_l1112_111218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_and_range_l1112_111292

-- Define the sets A and M
def A : Set ℝ := {x : ℝ | x^2 ≤ 5*x - 4}
def M (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (a+2)*x + 2*a ≤ 0}

-- Define the conditions p and q
def p (x : ℝ) (a : ℝ) : Prop := x ∈ M a
def q (x : ℝ) : Prop := x ∈ A

-- State the theorem
theorem solution_sets_and_range :
  (A = Set.Icc 1 4) ∧
  (∀ a ≥ 2, (∀ x, p x a → q x) → a ∈ Set.Icc 2 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_and_range_l1112_111292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1112_111208

theorem rationalize_denominator : 
  (30 : ℝ) / (40 : ℝ)^(1/3) = 3 * (1600 : ℝ)^(1/3) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1112_111208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1112_111237

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x) - 3) / (Real.sin x + Real.cos x - 2)

theorem f_range : Set.range f = {y : ℝ | 2 - Real.sqrt 2 ≤ y ∧ y ≤ 2 + Real.sqrt 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1112_111237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_over_sum_positive_l1112_111211

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 - Real.log (Real.sqrt (x^2 + 1) - x)

-- Theorem statement
theorem f_sum_over_sum_positive (a b : ℝ) (h : a + b ≠ 0) : 
  (f a + f b) / (a + b) > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_over_sum_positive_l1112_111211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_selling_price_l1112_111204

/-- Given a bond with face value and interest rate, calculate its selling price --/
theorem bond_selling_price (face_value : ℝ) (interest_rate : ℝ) (interest_to_price_ratio : ℝ) :
  face_value = 5000 →
  interest_rate = 0.06 →
  interest_to_price_ratio = 0.065 →
  ∃ (selling_price : ℝ), 
    (face_value * interest_rate) = (selling_price * interest_to_price_ratio) ∧
    abs (selling_price - 4615.38) < 0.01 := by
  sorry

#check bond_selling_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_selling_price_l1112_111204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_weight_is_90_l1112_111202

/-- Nancy's daily water intake in pounds -/
noncomputable def daily_water_intake : ℝ := 54

/-- The percentage of Nancy's body weight she drinks in water, expressed as a decimal -/
noncomputable def water_percentage : ℝ := 0.60

/-- Nancy's weight in pounds -/
noncomputable def nancy_weight : ℝ := daily_water_intake / water_percentage

theorem nancy_weight_is_90 : nancy_weight = 90 := by
  -- Unfold the definition of nancy_weight
  unfold nancy_weight
  -- Unfold the definitions of daily_water_intake and water_percentage
  unfold daily_water_intake water_percentage
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_weight_is_90_l1112_111202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1112_111236

-- Define the hyperbola
def hyperbola (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote
def asymptote : ℝ → ℝ → Prop :=
  λ x y ↦ y = Real.sqrt 3 * x - 1

-- Define the parabola
def parabola : ℝ → ℝ → Prop :=
  λ x y ↦ y^2 = 8 * Real.sqrt 2 * x

-- Define the directrix of the parabola
def directrix : ℝ → Prop :=
  λ x ↦ x = 2 * Real.sqrt 2

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x₀ y₀ : ℝ), asymptote x₀ y₀ ∧ 
    (∀ (x y : ℝ), hyperbola a b x y → (y = b/a * x ∨ y = -b/a * x))) →
  (∃ (x₁ y₁ : ℝ), hyperbola a b x₁ y₁ ∧ directrix x₁) →
  hyperbola (Real.sqrt 2) (Real.sqrt 6) = hyperbola a b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1112_111236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_implication_count_l1112_111293

theorem implication_count (p q r : Prop) : 
  let statements := [p ∧ ¬q ∧ ¬r, ¬p ∧ ¬q ∧ ¬r, p ∧ ¬q ∧ r, ¬p ∧ q ∧ ¬r]
  (statements.filter (λ s => (s → ((p → q) → r)) = True)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_implication_count_l1112_111293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_with_conditions_l1112_111263

theorem triangle_inequality_with_conditions (x y a b c : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (hxy : (x - 1) * (y - 1) ≥ 1)
  (htriangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * x + b^2 * y > c^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_with_conditions_l1112_111263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l1112_111241

open Real

-- Define proposition p
noncomputable def p : Prop := ∃ x : ℝ, tan x > 1

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := (1/3) * x^2

-- Define proposition q
def q : Prop := ∃ f d : ℝ, 
  (∀ x : ℝ, parabola x = (x - f)^2 / (4 * (f - d))) ∧ 
  (abs (f - d) = 1/6)

-- Theorem to prove
theorem correct_proposition : p ∧ (¬q) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l1112_111241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1112_111257

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def ellipse_set (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | distance P A + distance P B = 2 * distance A B}

theorem ellipse_theorem (A B : ℝ × ℝ) (h : distance A B = 10) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧
  ellipse_set A B = {P | (P.1 - ((A.1 + B.1) / 2))^2 / a^2 + (P.2 - ((A.2 + B.2) / 2))^2 / b^2 = 1} :=
by
  sorry

#check ellipse_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1112_111257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_correct_is_correct_l1112_111226

noncomputable def total_questions : ℕ := 4
noncomputable def known_questions : ℕ := 3
noncomputable def unknown_questions : ℕ := total_questions - known_questions
noncomputable def prob_correct_known : ℝ := 0.8
noncomputable def prob_correct_unknown : ℝ := 0.25
noncomputable def selected_questions : ℕ := 2

noncomputable def prob_both_correct : ℝ :=
  (Nat.choose known_questions selected_questions : ℝ) / (Nat.choose total_questions selected_questions : ℝ) * prob_correct_known ^ 2 +
  (Nat.choose unknown_questions 1 : ℝ) * (Nat.choose known_questions 1 : ℝ) / (Nat.choose total_questions selected_questions : ℝ) * prob_correct_known * prob_correct_unknown

theorem prob_both_correct_is_correct :
  prob_both_correct = 0.42 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_correct_is_correct_l1112_111226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_C_approx_l1112_111266

/-- The molar mass of calcium (Ca) in g/mol -/
noncomputable def molar_mass_Ca : ℝ := 40.08

/-- The molar mass of carbon (C) in g/mol -/
noncomputable def molar_mass_C : ℝ := 12.01

/-- The molar mass of oxygen (O) in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The number of oxygen atoms in calcium carbonate (CaCO3) -/
def num_O_atoms : ℕ := 3

/-- The molar mass of calcium carbonate (CaCO3) in g/mol -/
noncomputable def molar_mass_CaCO3 : ℝ := molar_mass_Ca + molar_mass_C + num_O_atoms * molar_mass_O

/-- The mass percentage of carbon (C) in calcium carbonate (CaCO3) -/
noncomputable def mass_percentage_C : ℝ := (molar_mass_C / molar_mass_CaCO3) * 100

/-- Theorem stating that the mass percentage of carbon in calcium carbonate is approximately 12.01% -/
theorem mass_percentage_C_approx :
  |mass_percentage_C - 12.01| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_C_approx_l1112_111266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_necessary_not_sufficient_l1112_111291

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def are_collinear (a b : V) : Prop :=
  ∃ (k : ℝ), a = k • b ∨ b = k • a

theorem collinearity_necessary_not_sufficient :
  (∀ a b : V, a ≠ 0 → b ≠ 0 → ‖a + b‖ = ‖a‖ + ‖b‖ → are_collinear a b) ∧
  ¬(∀ a b : V, a ≠ 0 → b ≠ 0 → are_collinear a b → ‖a + b‖ = ‖a‖ + ‖b‖) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_necessary_not_sufficient_l1112_111291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l1112_111251

noncomputable def data_set : List ℝ := [-2, -1, 0, 1, 2]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs)^2)).sum / xs.length

theorem variance_of_data_set : variance data_set = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l1112_111251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l1112_111261

/-- The function f(x) = √(1-x^2) -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2)

/-- The function g(x) = -x^2 + mx -/
def g (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x

/-- The tangent line l to f(x) at (0,1) -/
def l : ℝ → ℝ := λ _ ↦ 1

theorem tangent_line_intersection (m : ℝ) : 
  (∀ x, l x = f x → x = 0) ∧ 
  (∃ x, l x = g m x ∧ (∀ y, y ≠ x → l y ≠ g m y)) → 
  m = 2 ∨ m = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l1112_111261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_pi_fifths_radians_to_degrees_l1112_111246

-- Define the conversion factor from radians to degrees
noncomputable def radians_to_degrees (x : ℝ) : ℝ := x * (180 / Real.pi)

-- Theorem statement
theorem eight_pi_fifths_radians_to_degrees :
  radians_to_degrees (8 * Real.pi / 5) = 288 := by
  -- Unfold the definition of radians_to_degrees
  unfold radians_to_degrees
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_pi_fifths_radians_to_degrees_l1112_111246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1112_111221

theorem arithmetic_sequence_problem (x : ℝ) (h : x > 0) :
  let a : ℕ → ℝ := fun n =>
    match n with
    | 0 => 2^2
    | 1 => x^2
    | 2 => 5^2
    | _ => 0  -- Default value for other indices
  let d := a 1 - a 0
  (∀ n : ℕ, a (n + 1) - a n = d) →
  x = Real.sqrt 14.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1112_111221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rotation_pentagon_is_72_l1112_111245

/-- The minimum degree of rotation for a regular pentagon to coincide with its original figure -/
noncomputable def min_rotation_pentagon : ℚ :=
  360 / 5

/-- Theorem stating that the minimum rotation for a regular pentagon is 72 degrees -/
theorem min_rotation_pentagon_is_72 : 
  min_rotation_pentagon = 72 := by
  unfold min_rotation_pentagon
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rotation_pentagon_is_72_l1112_111245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_two_largest_l1112_111233

theorem product_of_two_largest (numbers : List ℕ) (h : numbers = [10, 11, 12, 13]) :
  (List.maximum numbers).bind (λ max => 
    (List.maximum (numbers.filter (· ≠ max))).map (λ second_max => 
      max * second_max
    )
  ) = some 156 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_two_largest_l1112_111233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_abs_diff_l1112_111203

def is_permutation (xs : List ℕ) : Prop :=
  xs.length = 63 ∧ xs.toFinset = Finset.range 63

def sum_abs_diff (xs : List ℕ) : ℕ :=
  (List.range 63).zip xs |>.map (λ (i, x) => Int.natAbs (x - (i + 1))) |>.sum

theorem max_sum_abs_diff :
  ∃ (xs : List ℕ), is_permutation xs ∧
    ∀ (ys : List ℕ), is_permutation ys → sum_abs_diff ys ≤ sum_abs_diff xs ∧
    sum_abs_diff xs = 1984 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_abs_diff_l1112_111203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_equality_implies_power_relation_l1112_111228

theorem gcd_equality_implies_power_relation (m n : ℕ) :
  (∀ k : ℕ, Nat.gcd (11 * k - 1) m = Nat.gcd (11 * k - 1) n) →
  ∃ l : ℕ, m = n * (11 : ℕ) ^ l ∨ n = m * (11 : ℕ) ^ l :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_equality_implies_power_relation_l1112_111228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_intersection_log_curve_l1112_111281

noncomputable def f (x : ℝ) : ℝ := Real.log x / x

theorem max_k_intersection_log_curve :
  (∃ (k : ℝ), ∀ (x : ℝ), x > 0 → k * x = Real.log x) →
  (∃ (k_max : ℝ), k_max = (1 : ℝ) / Real.exp 1 ∧
    ∀ (k : ℝ), (∃ (x : ℝ), x > 0 ∧ k * x = Real.log x) → k ≤ k_max) :=
by
  sorry

#check max_k_intersection_log_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_intersection_log_curve_l1112_111281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1112_111229

/-- Properties of a triangle ABC -/
theorem triangle_properties 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = Real.pi) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) 
  (h_sine_rule : a / Real.sin A = b / Real.sin B) :
  (A > B → Real.sin A > Real.sin B) ∧ 
  (Real.sin (2 * A) = Real.sin (2 * B) → (A = B ∨ A + B = Real.pi / 2)) ∧
  (a * Real.cos B - b * Real.cos A = c → A = Real.pi / 2) ∧
  (B = Real.pi / 3 ∧ a = 2 → Real.sqrt 3 < b ∧ b < 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1112_111229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1112_111206

/-- The equation of a cubic curve -/
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 2)

/-- Theorem: The tangent line equation -/
theorem tangent_line_equation :
  let x₀ := point.1
  let y₀ := point.2
  let m := f' x₀
  (fun x y ↦ y - y₀ = m * (x - x₀)) = (fun x y ↦ y = 3*x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1112_111206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_solution_l1112_111227

/-- A function that represents a polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- Condition that a polynomial is nonconstant -/
def IsNonconstant (f : RealPolynomial) : Prop :=
  ∃ x y, f x ≠ f y

/-- The main theorem -/
theorem polynomial_equation_solution (n : ℕ) (f : Fin (n + 3) → RealPolynomial)
    (hn : n ≥ 0)
    (h_nonconstant : ∀ k, IsNonconstant (f k))
    (h_eq : ∀ (k : Fin (n + 3)) (x : ℝ),
      (f k x) * (f (k.succ) x) = f (k.succ) (f (k.succ.succ) x))
    (h_cyclic : ∀ x, f 0 x = f (Fin.last _) x ∧ f 1 x = f 0 x) :
    ∀ (k : Fin (n + 3)) (x : ℝ), f k x = x^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_solution_l1112_111227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_G_l1112_111230

noncomputable def G : ℂ := (3/5 : ℝ) + (4/5 : ℝ) * Complex.I

theorem reciprocal_of_G :
  let recip := G⁻¹
  (recip.re > 0) ∧ (recip.im < 0) ∧ (Complex.abs recip = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_G_l1112_111230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apprentice_work_time_l1112_111258

/-- Represents the productivity of a worker (work completed per hour) -/
abbrev Productivity := ℝ

/-- Represents the time taken to complete a task in hours -/
abbrev Time := ℝ

/-- Represents the fraction of work completed -/
abbrev WorkFraction := ℝ

theorem apprentice_work_time 
  (master_productivity : Productivity)
  (apprentice_productivity : Productivity)
  (h1 : 7 * master_productivity + 4 * apprentice_productivity = (5 : ℝ) / 9)
  (h2 : 11 * master_productivity + 8 * apprentice_productivity = (17 : ℝ) / 18) :
  (1 : ℝ) / apprentice_productivity = 24 := by
  sorry

#check apprentice_work_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apprentice_work_time_l1112_111258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_class_l1112_111298

theorem students_in_class 
  (student_avg : ℝ) 
  (total_avg : ℝ) 
  (teacher_age : ℕ) 
  (h1 : student_avg = 14)
  (h2 : total_avg = 15)
  (h3 : teacher_age = 45)
  : ∃ (n : ℕ), n > 0 ∧ 
    (n * student_avg + teacher_age : ℝ) / (n + 1 : ℝ) = total_avg ∧
    n = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_class_l1112_111298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_l1112_111253

-- Define the functions f and g
noncomputable def f (x : ℝ) := 1 / Real.sqrt (2 - x)
noncomputable def g (a x : ℝ) := Real.log ((x - a - 1) * (2 * a - x))

-- Define the domain A of f
def A : Set ℝ := {x | x < -1 ∨ x ≥ 1}

-- Define the domain B of g
def B (a : ℝ) : Set ℝ := {x | (x - a - 1) * (2 * a - x) > 0}

-- Theorem statement
theorem domain_and_range :
  (∀ x, x ∈ A ↔ (x < -1 ∨ x ≥ 1)) ∧
  (∀ a, a < 1 → (B a ⊆ A ↔ a ∈ Set.Iic (-2) ∪ Set.Ioc (1/2) 1)) :=
by
  sorry

#check domain_and_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_l1112_111253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1112_111270

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

/-- Theorem: If 8S_6 = 7S_3 for a geometric sequence, then its common ratio is -1/2 -/
theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) :
  (8 * geometric_sum a₁ q 6 = 7 * geometric_sum a₁ q 3) → q = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1112_111270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l1112_111264

theorem monic_quadratic_with_complex_root :
  ∃! p : Polynomial ℝ,
    Polynomial.Monic p ∧ 
    Polynomial.degree p = 2 ∧ 
    (∀ x : ℂ, Polynomial.aeval x p = 0 ↔ x = Complex.mk 2 (-3)) ∧
    p = Polynomial.X^2 - 4 * Polynomial.X + 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l1112_111264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dylan_bike_speed_l1112_111223

/-- Represents the speed of a moving object in km/h -/
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- Proves that given a constant speed, a travel time of 25 hours, and a distance of 1250 kilometers, the speed is equal to 50 km/h -/
theorem dylan_bike_speed :
  let distance : ℝ := 1250
  let time : ℝ := 25
  speed distance time = 50 := by
  -- Unfold the definition of speed
  unfold speed
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dylan_bike_speed_l1112_111223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_at_negative_one_l1112_111277

-- Define the polynomial f(x)
def f (p q r : ℝ) (x : ℝ) : ℝ := x^3 - p*x^2 + q*x - r

-- Define the conditions on p, q, and r
def conditions (p q r : ℝ) : Prop := 0 < p ∧ 0 < q ∧ 0 < r ∧ p < q ∧ q < r

-- Define the theorem
theorem g_at_negative_one 
  (p q r : ℝ) 
  (h_conditions : conditions p q r) 
  (g : ℝ → ℝ)
  (h_g_leading_coeff : ∃ (a : ℝ → ℝ), g = λ x ↦ x^3 + a x^2 + (a x) + (a 1))
  (h_g_roots : ∀ (x : ℝ), f p q r x = 0 ↔ g (1/x) = 0) :
  g (-1) = (1 + p + q - r) / (-r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_at_negative_one_l1112_111277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_transaction_proof_l1112_111249

theorem house_transaction_proof (initial_value loss_percent maintenance_cost gain_percent : ℝ) :
  initial_value = 12000 ∧
  loss_percent = 0.15 ∧
  maintenance_cost = 300 ∧
  gain_percent = 0.2 →
  (let first_sale_price := initial_value * (1 - loss_percent)
   let total_cost_B := first_sale_price + maintenance_cost
   let second_sale_price := first_sale_price * (1 + gain_percent)
   let loss_A := second_sale_price - initial_value
   let gain_B := second_sale_price - total_cost_B
   loss_A = 240 ∧ gain_B = 1740) := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_transaction_proof_l1112_111249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_function_and_inverse_l1112_111276

-- Define the function f
def f (b : ℤ) : ℝ → ℝ := λ x ↦ 3 * x + b

-- State the theorem
theorem intersection_point_of_function_and_inverse (b a : ℤ) :
  (∃ (f_inv : ℝ → ℝ), Function.LeftInverse f_inv (f b) ∧ Function.RightInverse f_inv (f b)) →
  f b (-3) = a →
  f b a = -3 →
  a = -3 :=
by
  intros h1 h2 h3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_function_and_inverse_l1112_111276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l1112_111267

-- Define the hyperbolas H₂
def H₂ (x y : ℝ) : Prop := x^2 / 20 - y^2 / 5 = 1

-- Define that H₁ and H₂ share asymptotes
axiom share_asymptotes : ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), H₂ x y ↔ x^2 / 20 - y^2 / 5 = k

-- Define the point on H₁
noncomputable def point_on_H₁ : ℝ × ℝ := (2 * Real.sqrt 15, Real.sqrt 5)

-- State the theorem
theorem distance_focus_to_asymptote :
  ∃ (H₁ : ℝ → ℝ → Prop),
    (∀ x y, H₁ x y ↔ x^2 / 20 - y^2 / 5 = 2) ∧
    H₁ point_on_H₁.1 point_on_H₁.2 ∧
    (∃ (focus : ℝ × ℝ) (asymptote : ℝ → ℝ),
      (∀ x, asymptote x = x / 2 ∨ asymptote x = -x / 2) ∧
      Real.sqrt ((focus.1 - 0)^2 + (focus.2 - asymptote focus.1)^2) = Real.sqrt 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l1112_111267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1112_111201

/-- Time taken for two trains to cross each other -/
noncomputable def train_crossing_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let total_length := length1 + length2
  let relative_speed := (speed1 + speed2) * (5 / 18)
  total_length / relative_speed

theorem train_crossing_theorem :
  let length1 : ℝ := 180
  let length2 : ℝ := 160
  let speed1 : ℝ := 60
  let speed2 : ℝ := 40
  abs (train_crossing_time length1 length2 speed1 speed2 - 12.24) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1112_111201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l1112_111209

theorem puzzle_solution : ∃ (N E : ℕ), 
  N = 3 ∧ 
  E = 7 ∧ 
  (N : ℚ) / E = 0.428571428571428571 :=
by
  -- Introduce N and E
  use 3, 7
  
  -- Split the goal into three parts
  refine ⟨rfl, rfl, ?_⟩
  
  -- Prove the equality
  norm_num
  
  sorry -- Skip the detailed proof of the repeating decimal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l1112_111209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_and_sine_sum_l1112_111214

open Real
open BigOperators

theorem cosine_sum_and_sine_sum (n : ℕ+) (x : ℝ) (h : ∀ k : ℤ, x ≠ 2 * π * ↑k) :
  (∑ i in Finset.range n, cos (↑i * x)) = (sin ((↑n + 1/2) * x)) / (2 * sin (x/2)) - 1/2 ∧
  (∑ i in Finset.range 2018, ↑(i + 1) * sin (↑(i + 1) * π / 6)) = sqrt 3 - 2015/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_and_sine_sum_l1112_111214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_specific_dataset_l1112_111207

noncomputable def standard_deviation (data : Finset ℝ) : ℝ :=
  Real.sqrt (((data.sum (λ x => x^2)) / data.card) - (data.sum id / data.card)^2)

theorem standard_deviation_specific_dataset :
  ∀ (data : Finset ℝ),
    data.card = 40 →
    data.sum (λ x => x^2) = 56 →
    data.sum id / data.card = Real.sqrt 2 / 2 →
    standard_deviation data = 3 * Real.sqrt 10 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_specific_dataset_l1112_111207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_theorem_l1112_111260

/-- Given a point A in 3D space, this function returns the point symmetric to A with respect to the y-axis. -/
def symmetric_point_y_axis (A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-(A.1), A.2.1, -(A.2.2))

/-- Theorem stating that the point symmetric to A(-3, 1, -4) with respect to the y-axis has coordinates (3, 1, 4). -/
theorem symmetric_point_theorem :
  let A : ℝ × ℝ × ℝ := (-3, 1, -4)
  symmetric_point_y_axis A = (3, 1, 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_theorem_l1112_111260
