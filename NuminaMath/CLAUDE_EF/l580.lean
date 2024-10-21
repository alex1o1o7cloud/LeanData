import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l580_58042

noncomputable def z : ℂ := (3 + Complex.I) / (1 + Complex.I) + 3 * Complex.I

theorem z_in_first_quadrant : 
  0 < z.re ∧ 0 < z.im := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l580_58042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l580_58099

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x * Real.cos x - Real.sin x ^ 2 + 1 / 2

theorem f_increasing_interval (a : ℝ) :
  (∀ x : ℝ, f a (x + π / 3) = f a (π / 3 - x)) →
  ∀ k : ℤ, StrictMonoOn (f a) (Set.Icc (k * π - π / 3) (k * π + π / 6)) := by
  sorry

#check f_increasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l580_58099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_specific_segment_l580_58092

theorem midpoint_of_specific_segment : 
  let z₁ : ℂ := -20 + 8*Complex.I
  let z₂ : ℂ := 10 - 12*Complex.I
  (z₁ + z₂) / 2 = -5 - 2*Complex.I :=
by
  -- Expand the definition of z₁ and z₂
  simp [Complex.I]
  -- Perform the arithmetic
  ring
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_specific_segment_l580_58092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_prime_l580_58067

def recurrence (a b x₀ : ℕ) : ℕ → ℕ
  | 0 => x₀
  | n + 1 => a * recurrence a b x₀ n + b

theorem not_all_prime (a b x₀ : ℕ) (ha : a > 0) (hb : b > 0) (hx₀ : x₀ > 0) :
  ∃ n : ℕ, ¬ Nat.Prime (recurrence a b x₀ n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_prime_l580_58067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crate_height_difference_l580_58060

/-- The diameter of each cylindrical roll in centimeters -/
def roll_diameter : ℝ := 12

/-- The number of rows in Crate A -/
def rows : ℕ := 18

/-- The number of rolls per row in Crate A -/
def rolls_per_row : ℕ := 8

/-- The height of Crate A in centimeters -/
def height_crate_a : ℝ := rows * roll_diameter

/-- The height between two consecutive centers of rolls in hexagonal packing -/
noncomputable def hexagonal_gap : ℝ := (Real.sqrt 3 / 2) * roll_diameter

/-- The height of Crate B in centimeters -/
noncomputable def height_crate_b : ℝ := rows * roll_diameter + (rows - 1) * hexagonal_gap

/-- The positive difference in height between Crate B and Crate A -/
noncomputable def height_difference : ℝ := height_crate_b - height_crate_a

theorem crate_height_difference :
  height_difference = 102 * Real.sqrt 3 - 108 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crate_height_difference_l580_58060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_about_center_l580_58005

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem f_symmetry_about_center : 
  ∀ (x : ℝ), f (π/12 - x) = f (π/12 + x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_about_center_l580_58005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumscribed_sphere_area_l580_58032

/-- The surface area of a sphere circumscribing a regular tetrahedron -/
theorem tetrahedron_circumscribed_sphere_area (a : ℝ) (a_pos : 0 < a) :
  (3 * Real.pi * a^2) / 2 = (3 * Real.pi * a^2) / 2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumscribed_sphere_area_l580_58032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_implies_m_range_l580_58048

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ m + 2}

-- Part 1: Prove that if A ∩ B = [1,3], then m = 3
theorem intersection_implies_m_value (m : ℝ) : A ∩ B m = Set.Icc 1 3 → m = 3 := by
  sorry

-- Part 2: Prove that if A ⊆ ℝ\B, then m > 5 or m < -3
theorem subset_implies_m_range (m : ℝ) : A ⊆ (Set.univ \ B m) → m > 5 ∨ m < -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_implies_m_range_l580_58048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_minus_sum_result_l580_58036

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def primesBetween4And18 : Set ℕ := {p | isPrime p ∧ 4 < p ∧ p < 18}

def operation (p q : ℕ) : ℤ := p * q - (p + q)

theorem prime_product_minus_sum_result :
  ∃ (p q : ℕ), p ∈ primesBetween4And18 ∧ q ∈ primesBetween4And18 ∧ p ≠ q ∧ operation p q = 119 ∧
  (∀ (x y : ℕ), x ∈ primesBetween4And18 → y ∈ primesBetween4And18 → x ≠ y →
    operation x y ∉ ({21, 60, 180, 231} : Set ℤ)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_minus_sum_result_l580_58036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l580_58050

theorem hyperbola_equation (x y : ℝ) : 
  let given_hyperbola := fun (x y : ℝ) => x^2 / 9 - y^2 / 16 = 1
  let target_hyperbola := fun (x y : ℝ) => 4 * x^2 / 9 - y^2 / 4 = 1
  (∀ (x y : ℝ), x^2 / 9 - y^2 / 16 = 0 ↔ 4 * x^2 / 9 - y^2 / 4 = 0) ∧
  target_hyperbola (-3) (2 * Real.sqrt 3) →
  target_hyperbola x y := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l580_58050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_distance_l580_58016

noncomputable def initialHeight : ℝ := 100

noncomputable def bounceRatio : ℝ := 1/2

def numBounces : ℕ := 8

noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

noncomputable def totalDistance : ℝ :=
  2 * geometricSum initialHeight bounceRatio numBounces - initialHeight

theorem ball_bounce_distance :
  ∃ ε > 0, abs (totalDistance - 298.4) < ε := by
  sorry

#eval numBounces -- This will work since numBounces is computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_distance_l580_58016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l580_58057

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^2
def curve2 (x : ℝ) : ℝ := x
def curve3 (x : ℝ) : ℝ := 2*x

-- Define the bounded area
noncomputable def bounded_area : ℝ :=
  (∫ x in Set.Icc 0 1, curve3 x - curve2 x) +
  (∫ x in Set.Icc 1 2, curve3 x - curve1 x)

-- Theorem statement
theorem area_of_bounded_region :
  bounded_area = 7/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l580_58057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l580_58074

/-- Calculates the speed of a train excluding stoppages given its speed including stoppages and stop time per hour. -/
noncomputable def train_speed_excluding_stoppages (speed_with_stops : ℝ) (stop_time : ℝ) : ℝ :=
  speed_with_stops / (1 - stop_time / 60)

/-- Theorem stating that a train traveling at 31 kmph including stoppages and stopping for 18.67 minutes per hour has a speed of approximately 44.98 kmph excluding stoppages. -/
theorem train_speed_theorem :
  let speed_with_stops := (31 : ℝ)
  let stop_time := (18.67 : ℝ)
  let calculated_speed := train_speed_excluding_stoppages speed_with_stops stop_time
  abs (calculated_speed - 44.98) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l580_58074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_over_four_l580_58062

theorem tan_theta_minus_pi_over_four (θ : Real) 
  (h : Real.sin θ - 3 * Real.cos θ = Real.sqrt 10) : 
  Real.tan (θ - π/4) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_over_four_l580_58062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_properties_l580_58052

-- Define the basic structures
structure Parallelepiped where

structure Cuboid extends Parallelepiped where

structure Cube extends Cuboid where

structure RightParallelepiped extends Parallelepiped where

-- Define properties
def has_rectangular_base (p : Parallelepiped) : Prop := sorry

def is_right_quadrilateral_prism (p : Parallelepiped) : Prop := sorry

def has_congruent_edges (p : Parallelepiped) : Prop := sorry

def lateral_edges_perpendicular_to_two_base_edges (p : Parallelepiped) : Prop := sorry

def has_equal_diagonals (p : Parallelepiped) : Prop := sorry

-- State the theorem
theorem parallelepiped_properties :
  ¬(∀ (p : Parallelepiped), has_rectangular_base p → (∃ c : Cuboid, c.toParallelepiped = p)) ∧
  ¬(∀ (p : Parallelepiped), is_right_quadrilateral_prism p ∧ has_congruent_edges p → (∃ c : Cube, c.toCuboid.toParallelepiped = p)) ∧
  ¬(∀ (p : Parallelepiped), lateral_edges_perpendicular_to_two_base_edges p → (∃ r : RightParallelepiped, r.toParallelepiped = p)) ∧
  (∀ (p : Parallelepiped), has_equal_diagonals p → (∃ r : RightParallelepiped, r.toParallelepiped = p)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_properties_l580_58052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_four_powers_l580_58058

theorem fourth_root_of_four_powers : ((4^7 + 4^7 + 4^7 + 4^7 : ℝ) ^ (1/4 : ℝ)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_four_powers_l580_58058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_maximum_l580_58009

-- Define the recursive function g_n
noncomputable def g : ℕ → ℝ → ℝ
| 0, _ => 0  -- Add a base case for 0
| 1, x => Real.sqrt (2 - x)
| (n + 1), x => g n (Real.sqrt ((n + 1)^3 - x))

-- Define the domain of g_n
def domain (n : ℕ) : Set ℝ :=
  {x | ∃ y, g n x = y}

-- State the theorem
theorem g_domain_maximum :
  (∃ N : ℕ, N = 3 ∧ 
    (∀ n > N, domain n = ∅) ∧
    (domain N = {-37})) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_maximum_l580_58009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l580_58017

noncomputable section

-- Define the function
def f (x : ℝ) : ℝ := x - 1/x

-- State the theorem
theorem derivative_of_f (x : ℝ) (h : x ≠ 0) : 
  deriv f x = 1 + 1/x^2 := by
  -- Proof steps would go here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l580_58017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_absolute_value_exists_l580_58034

theorem equal_absolute_value_exists (n : ℕ) (a b c : ℂ) 
  (h1 : n > 1) 
  (h2 : a + b + c = 0) 
  (h3 : a^n + b^n + c^n = 0) : 
  ∃ (x y : ℂ), x ∈ ({a, b, c} : Set ℂ) ∧ y ∈ ({a, b, c} : Set ℂ) ∧ x ≠ y ∧ Complex.abs x = Complex.abs y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_absolute_value_exists_l580_58034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l580_58081

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x < 1/2 ∨ x > 1) → (x^2 - (2*a + 1)*x + a*(a + 1) > 0)) ∧
  (∃ x : ℝ, (x < 1/2 ∨ x > 1) ∧ (x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0)) →
  (0 ≤ a ∧ a ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l580_58081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumscribed_circle_l580_58001

-- Define the sector and its properties
def circle_radius : ℝ := 8
def central_angle (θ : ℝ) : Prop := Real.pi/2 < θ ∧ θ < Real.pi

-- Define the radius of the circumscribed circle
noncomputable def circumscribed_radius (θ : ℝ) : ℝ := 4 * (1 / Real.cos (θ/2))

-- Theorem statement
theorem sector_circumscribed_circle (θ : ℝ) 
  (h : central_angle θ) : 
  circumscribed_radius θ = 4 * (1 / Real.cos (θ/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumscribed_circle_l580_58001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_l580_58078

noncomputable def distance (x y x' y' : ℝ) : ℝ := Real.sqrt ((x - x')^2 + (y - y')^2)

def point_set : List (ℝ × ℝ) := [(1, 5), (2, 1), (4, -3), (7, 0), (-2, -1)]

def reference_point : ℝ × ℝ := (3, 3)

theorem closest_point :
  ∀ p ∈ point_set, 
    distance 2 1 3 3 ≤ distance p.1 p.2 3 3 :=
by
  sorry

#eval point_set
#eval reference_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_l580_58078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l580_58041

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    and a point P on the ellipse satisfying certain conditions,
    prove that the eccentricity of the ellipse is √3 - 1 -/
theorem ellipse_eccentricity (a b : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a > 0 → b > 0 → a > b →
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) →
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0 →
  Real.tan (Real.arctan ((P.2 - F₁.2) / (P.1 - F₁.1)) - Real.arctan ((F₂.2 - F₁.2) / (F₂.1 - F₁.1))) = Real.sqrt 3 / 3 →
  let c := Real.sqrt (a^2 - b^2)
  c / a = Real.sqrt 3 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l580_58041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_solutions_l580_58093

-- Define the function f(x) = x^3 - 2x + 1
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the set of real numbers that satisfy the equation
def S : Set ℝ := {x : ℝ | (f x)^2 = 9}

-- Theorem statement
theorem exactly_two_solutions : ∃ (s : Finset ℝ), s.card = 2 ∧ ∀ x, x ∈ S ↔ x ∈ s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_solutions_l580_58093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sustainable_consumption_is_max_sustainable_l580_58069

/-- The maximum sustainable daily consumption of apples by a bird -/
theorem max_sustainable_consumption : ℝ := 10 / 11
  where
  initial_apples : ℝ := 10
  growth_rate : ℝ := 1.1

  theorem is_max_sustainable (x : ℝ) (h : x > 0) :
    (∀ n : ℕ, 
      let apples_after_n_days : ℝ := (growth_rate * (initial_apples - x))^n * initial_apples
      apples_after_n_days ≥ initial_apples) →
    x ≤ 10 / 11 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sustainable_consumption_is_max_sustainable_l580_58069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_regression_line_regression_line_passes_through_center_regression_line_negative_slope_l580_58043

-- Define the variables and constants
variable (x y : ℝ → ℝ)
def x_mean : ℝ := 4
def y_mean : ℝ := 6.5

-- Define the property of negative correlation
def negatively_correlated (x y : ℝ → ℝ) : Prop :=
  ∀ t, (x t - x_mean) * (y t - y_mean) < 0

-- Define the regression line equation
def regression_line (a : ℝ) : ℝ := -2 * a + 14.5

-- State the theorem
theorem correct_regression_line
  (h1 : negatively_correlated x y)
  (h2 : regression_line x_mean = y_mean) :
  ∀ t, y t = regression_line (x t) := by
  sorry

-- Verify that the regression line passes through the sample center
theorem regression_line_passes_through_center :
  regression_line x_mean = y_mean := by
  unfold regression_line x_mean y_mean
  simp
  norm_num

-- Prove that the regression line has a negative slope
theorem regression_line_negative_slope :
  ∀ a b, a < b → regression_line a > regression_line b := by
  intros a b h
  unfold regression_line
  linarith

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_regression_line_regression_line_passes_through_center_regression_line_negative_slope_l580_58043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_formula_l580_58038

/-- Calculates the total time for a train journey with a stop -/
noncomputable def total_journey_time (p : ℝ) : ℝ :=
  let distance_ab := p
  let distance_bc := 2 * p
  let speed_ab := (50 : ℝ)
  let speed_bc := (80 : ℝ)
  let pause_time := (0.5 : ℝ)
  let time_ab := distance_ab / speed_ab
  let time_bc := distance_bc / speed_bc
  time_ab + pause_time + time_bc

/-- Theorem stating that the total journey time is (9p + 100) / 200 hours -/
theorem journey_time_formula (p : ℝ) :
  total_journey_time p = (9 * p + 100) / 200 := by
  unfold total_journey_time
  -- The actual proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_formula_l580_58038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l580_58063

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in a 2D plane --/
def Point : Type := ℝ × ℝ

/-- Calculate the distance between two points --/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Check if a point is on a circle --/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  distance p c.center = c.radius

/-- The main theorem --/
theorem intersection_points_count 
  (D : Circle) 
  (Q : Point) 
  (h1 : D.radius = 5) 
  (h2 : distance Q D.center = 8) :
  ∃ (P1 P2 : Point), 
    isOnCircle P1 D ∧ 
    isOnCircle P2 D ∧ 
    distance P1 Q = 4 ∧ 
    distance P2 Q = 4 ∧ 
    ∀ (P : Point), isOnCircle P D ∧ distance P Q = 4 → P = P1 ∨ P = P2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l580_58063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_words_without_AMC_l580_58054

/-- The set of letters used to form the words -/
inductive Letter : Type
| A : Letter
| M : Letter
| C : Letter

/-- A six-letter word is represented as a function from Fin 6 to Letter -/
def Word : Type := Fin 6 → Letter

/-- Check if a word contains the substring AMC -/
def containsAMC (w : Word) : Prop :=
  ∃ i : Fin 4, w i = Letter.A ∧ w (i + 1) = Letter.M ∧ w (i + 2) = Letter.C

/-- The set of all possible six-letter words -/
def allWords : Set Word :=
  {w | True}

/-- The set of words not containing the substring AMC -/
def wordsWithoutAMC : Set Word :=
  {w | ¬containsAMC w}

/-- Fintype instance for Word -/
instance : Fintype Word := by
  sorry

/-- DecidablePred instance for wordsWithoutAMC -/
instance : DecidablePred (λ w => w ∈ wordsWithoutAMC) := by
  sorry

/-- The main theorem: there are 622 six-letter words without the substring AMC -/
theorem count_words_without_AMC : Finset.card (Finset.filter (λ w => w ∈ wordsWithoutAMC) (Finset.univ : Finset Word)) = 622 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_words_without_AMC_l580_58054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l580_58025

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem statement
theorem floor_expression_evaluation :
  (floor 6.5) * (floor (2 / 3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.4) - (6.2 : ℝ) = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l580_58025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_volume_at_centroid_l580_58008

/-- A trihedral angle with right dihedral angles -/
structure TrihedralAngle where
  is_right : Bool

/-- A point inside a trihedral angle -/
structure InnerPoint (T : TrihedralAngle) where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane passing through a point inside a trihedral angle -/
structure IntersectingPlane (T : TrihedralAngle) (P : InnerPoint T) where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : x / a + y / b + z / c = 1

/-- The volume of the tetrahedron cut off by the plane -/
noncomputable def tetrahedronVolume (T : TrihedralAngle) (P : InnerPoint T) (Pl : IntersectingPlane T P) : ℝ :=
  (1 / 6) * Pl.a * Pl.b * Pl.c

/-- The centroid of the triangle formed by the intersection of the plane and the trihedral angle -/
def isCentroid (T : TrihedralAngle) (P : InnerPoint T) (Pl : IntersectingPlane T P) : Prop :=
  P.x / Pl.a = P.y / Pl.b ∧ P.y / Pl.b = P.z / Pl.c ∧ P.x / Pl.a + P.y / Pl.b + P.z / Pl.c = 1

/-- The main theorem -/
theorem min_volume_at_centroid (T : TrihedralAngle) (P : InnerPoint T) (Pl : IntersectingPlane T P) :
  T.is_right → isCentroid T P Pl → 
  ∀ (Q : InnerPoint T) (Ql : IntersectingPlane T Q), 
    tetrahedronVolume T P Pl ≤ tetrahedronVolume T Q Ql := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_volume_at_centroid_l580_58008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_is_four_l580_58011

-- Define the piecewise function g
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then a * x + b else 9 - 2 * x

-- State the theorem
theorem sum_of_a_and_b_is_four (a b : ℝ) :
  (∀ x, g a b (g a b x) = x) →   -- g(g(x)) = x for all x
  ContinuousAt (g a b) 3 →       -- g is continuous at x = 3
  a + b = 4 :=                   -- conclusion: a + b = 4
by
  sorry  -- Skip the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_is_four_l580_58011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_of_equation_l580_58077

theorem integer_solutions_of_equation : 
  {x : ℤ | (x - 3 : ℚ) ^ (27 - x ^ 3) = 1} = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_of_equation_l580_58077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temp_drop_notation_l580_58080

/-- Represents a temperature change in Celsius -/
structure TempChange where
  value : ℚ
  is_rise : Bool

/-- Denotes how a temperature change is represented as a string -/
def denote (t : TempChange) : String :=
  if t.is_rise then
    s!"+{t.value}°C"
  else
    s!"-{t.value}°C"

theorem temp_drop_notation (rise : TempChange) (drop : TempChange) 
    (h1 : rise.value = 10) (h2 : rise.is_rise = true) 
    (h3 : drop.value = 5) (h4 : drop.is_rise = false) :
    denote rise = "+10°C" → denote drop = "-5°C" := by
  intro h
  simp [denote, h1, h2, h3, h4]
  rfl

#eval denote { value := 10, is_rise := true }
#eval denote { value := 5, is_rise := false }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temp_drop_notation_l580_58080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l580_58033

def z₁ : ℂ := 1 + Complex.I
def z₂ (m : ℝ) : ℂ := 2 + m * Complex.I

theorem complex_number_problem :
  (∃ m : ℝ, (z₂ m / z₁).im ≠ 0 ∧ (z₂ m / z₁).re = 0 → m = -2) ∧
  (∃ m : ℝ, (z₂ m / z₁).im = 0 → Complex.re (3 * z₁ + Complex.I * z₂ m) + Complex.im (3 * z₁ + Complex.I * z₂ m) = 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l580_58033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_18_plus_2sqrt73_l580_58088

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  -- Base lengths
  EF : ℝ
  GH : ℝ
  -- EF is twice GH
  h_ef_twice_gh : EF = 2 * GH
  -- Area is 72 square units
  h_area : (EF + GH) * (48 / GH) / 2 = 72
  -- Non-negative lengths
  h_ef_nonneg : EF ≥ 0
  h_gh_nonneg : GH ≥ 0

/-- The perimeter of the isosceles trapezoid -/
noncomputable def perimeter (t : IsoscelesTrapezoid) : ℝ :=
  t.EF + t.GH + 2 * Real.sqrt ((48 / t.GH)^2 + (t.GH / 2)^2)

/-- Theorem stating that the perimeter is 18 + 2√73 -/
theorem perimeter_is_18_plus_2sqrt73 (t : IsoscelesTrapezoid) :
  perimeter t = 18 + 2 * Real.sqrt 73 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_18_plus_2sqrt73_l580_58088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_focal_distances_l580_58085

/-- An ellipse with semi-major axis a, semi-minor axis b, and semi-focal distance c -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : c > 0
  h4 : a > b
  h5 : a^2 = b^2 + c^2

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The theorem stating the maximum value of |PF₁| · |PF₂| for any point P on the ellipse -/
theorem max_product_focal_distances (e : Ellipse) :
  ∃ (M : ℝ), M = e.a^2 ∧ ∀ (P : PointOnEllipse e),
    |P.x - (-e.c)| * |P.x - e.c| ≤ M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_focal_distances_l580_58085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_calculation_l580_58006

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

noncomputable def yearly_gain (
  borrowed_amount : ℝ)
  (borrowing_rate : ℝ)
  (lending_rate : ℝ)
  (time : ℝ) : ℝ :=
  let interest_earned := simple_interest borrowed_amount lending_rate time
  let interest_paid := simple_interest borrowed_amount borrowing_rate time
  (interest_earned - interest_paid) / time

theorem gain_calculation :
  yearly_gain 20000 8 9 6 = 200 := by
  unfold yearly_gain simple_interest
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_calculation_l580_58006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segments_intersect_l580_58003

/-- A spatial quadrilateral -/
structure SpatialQuadrilateral where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  D : ℝ × ℝ × ℝ

/-- A point that divides a line segment in a given ratio -/
noncomputable def divideSegment (P Q : ℝ × ℝ × ℝ) (ratio : ℝ) : ℝ × ℝ × ℝ :=
  let (x₁, y₁, z₁) := P
  let (x₂, y₂, z₂) := Q
  ((x₁ + ratio * x₂) / (1 + ratio),
   (y₁ + ratio * y₂) / (1 + ratio),
   (z₁ + ratio * z₂) / (1 + ratio))

/-- Theorem: In a spatial quadrilateral, if points divide the sides in given ratios,
    then the segments connecting these points intersect -/
theorem segments_intersect (ABCD : SpatialQuadrilateral) (α β : ℝ) :
  let K₁ := divideSegment ABCD.A ABCD.B α
  let K₂ := divideSegment ABCD.C ABCD.D α
  let K₃ := divideSegment ABCD.B ABCD.C β
  let K₄ := divideSegment ABCD.A ABCD.D β
  ∃ t s, t ∈ Set.Icc 0 1 ∧ s ∈ Set.Icc 0 1 ∧
    K₁ + t • (K₂ - K₁) = K₃ + s • (K₄ - K₃) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segments_intersect_l580_58003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sin_neg_x_not_cos_x_l580_58059

-- Define the function f(x) = sin(-x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (-x)

-- State the theorem
theorem derivative_of_sin_neg_x_not_cos_x :
  ¬ (∀ x : ℝ, deriv f x = Real.cos x) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sin_neg_x_not_cos_x_l580_58059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l580_58098

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ici (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l580_58098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_our_monomial_l580_58045

/-- The degree of a monomial is the sum of the exponents of its variables. -/
def degree_of_monomial (m : MvPolynomial (Fin 2) ℚ) : ℕ :=
  sorry

/-- The monomial -2/3 * a^3 * b -/
def our_monomial : MvPolynomial (Fin 2) ℚ :=
  sorry

theorem degree_of_our_monomial :
  degree_of_monomial our_monomial = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_our_monomial_l580_58045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_existence_inequality_condition_l580_58071

-- Define the function f(x) as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

-- Part I
theorem zero_point_existence (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f a x = 0) ↔ a < 0 :=
by sorry

-- Part II
theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 2 → f a x ≥ a) ↔ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_existence_inequality_condition_l580_58071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contribution_impossibility_l580_58020

theorem contribution_impossibility (a b c d e : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → e ≥ 0 → 
  a + b + c + d + e > 0 → 
  (∀ (x y : ℝ), (x, y) ∈ [(a, b), (a, c), (a, d), (a, e), (b, c), (b, d), (b, e), (c, d), (c, e), (d, e)] → 
    x + y < (a + b + c + d + e) / 3) → 
  False := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contribution_impossibility_l580_58020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_increase_l580_58026

theorem book_price_increase (P : ℝ) (h : P > 0) : 
  let price_after_first_increase := P * (1 + 0.15)
  let price_after_second_increase := price_after_first_increase * (1 + 0.20)
  let price_after_third_increase := price_after_second_increase * (1 + 0.25)
  price_after_third_increase = P * 1.725 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_increase_l580_58026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sixth_power_l580_58037

theorem cos_sin_sixth_power (θ : ℝ) (h : Real.cos (2 * θ) = 1/4) :
  (Real.cos θ) ^ 6 + (Real.sin θ) ^ 6 = 19/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sixth_power_l580_58037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_from_cosine_and_negative_sine_l580_58055

theorem tangent_from_cosine_and_negative_sine (α : ℝ) 
  (h1 : Real.cos α = 3/5) (h2 : Real.sin α < 0) : Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_from_cosine_and_negative_sine_l580_58055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiple_l580_58049

theorem matrix_scalar_multiple (M : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ v : Fin 3 → ℝ, M.vecMul v = (-4 : ℝ) • v) →
  M = ![![-4, 0, 0], ![0, -4, 0], ![0, 0, -4]] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiple_l580_58049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_direct_proportion_l580_58056

/-- A function f: ℝ → ℝ is a direct proportion function if there exists a constant k such that f(x) = k * x for all x. -/
def IsDirectProportionFunction (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function f(x) = (1/2)x -/
noncomputable def f : ℝ → ℝ := λ x ↦ (1/2) * x

/-- Theorem: The function f(x) = (1/2)x is a direct proportion function -/
theorem f_is_direct_proportion : IsDirectProportionFunction f := by
  use (1/2)
  intro x
  rfl

#check f_is_direct_proportion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_direct_proportion_l580_58056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_ordering_l580_58096

/-- Inverse proportion function -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_proportion_point_ordering :
  ∀ (k : ℝ) (y₁ y₂ : ℝ),
  inverse_proportion k (-2) = y₁ →
  inverse_proportion k 1 = y₂ →
  inverse_proportion k 2 = 1 →
  y₁ < 1 ∧ 1 < y₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_ordering_l580_58096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_points_l580_58035

theorem complex_equality_points : ∃ x : ℝ, 
  let z₁ := Complex.mk (Real.sin x * Real.sin x) (Real.cos (2 * x))
  let z₂ := Complex.mk (Real.sin x * Real.sin x) (Real.cos x)
  z₁ = z₂ ∧ (z₁ = Complex.I ∨ z₁ = Complex.mk (3/4) (-1/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_points_l580_58035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l580_58086

open Real

-- Define the polar coordinate equation of line l
def line_l (ρ θ : ℝ) : Prop := 2 * ρ * sin (θ - π / 6) - 3 * sqrt 3 = 0

-- Define the parametric equation of curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (cos α, sqrt 3 * sin α)

-- Define the distance function from a point to line l
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x - sqrt 3 * y + 3 * sqrt 3) / 2

-- State the theorem
theorem max_distance_to_line :
  ∀ α : ℝ, 
  let (x, y) := curve_C α
  distance_to_line x y ≤ (sqrt 10 + 3 * sqrt 3) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l580_58086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l580_58087

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x^α

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) (α : ℝ) :
  (∃ α, f = power_function α) →  -- f is a power function
  f 2 = Real.sqrt 2 →            -- f passes through (2, √2)
  f (1/9) = 1/3 :=               -- prove f(1/9) = 1/3
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l580_58087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_line_l580_58019

/-- Definition of the ellipse E -/
def ellipse (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (1 - a^2) = 1

/-- Definition of focal distance -/
noncomputable def focalDistance (a : ℝ) : ℝ := 
  Real.sqrt (2 * a^2 - 1)

/-- Theorem about the equation of ellipse E and the fixed line -/
theorem ellipse_and_fixed_line :
  ∃ (a : ℝ), 0 < a ∧ a < 1 ∧ focalDistance a = 1 ∧
  (∀ x y : ℝ, ellipse a x y ↔ 8 * x^2 / 5 + 8 * y^2 / 3 = 1) ∧
  (∀ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ ellipse a x₀ y₀ ∧
    (∃ q : ℝ, q * (x₀ + focalDistance a) * (x₀ - focalDistance a) = -y₀^2) →
    x₀ + y₀ = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_line_l580_58019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_surd_form_l580_58070

-- Define the expression
noncomputable def expression : ℝ := Real.sqrt (2 / Real.sqrt 2)

-- Theorem statement
theorem simplest_surd_form : 
  expression = Real.sqrt (Real.sqrt 2) := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_surd_form_l580_58070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_a_for_g_nonnegative_l580_58027

/-- The base of the natural logarithm -/
noncomputable def e : ℝ := Real.exp 1

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x - (a / 2) * x^2 + a * x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x + (x - 2) * Real.exp x - a * x + a

/-- The function g(x) -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f' a x + 2 - a

/-- The theorem stating that the maximum integer value of a for which g(x) ≥ 0 holds true over ℝ is 1 -/
theorem max_integer_a_for_g_nonnegative :
  (∃ (a : ℤ), ∀ (x : ℝ), g a x ≥ 0) ∧
  (∀ (a : ℤ), (∀ (x : ℝ), g a x ≥ 0) → a ≤ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_a_for_g_nonnegative_l580_58027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_circle_l580_58091

/-- Two non-intersecting circles with centers O₁ and O₂ -/
structure TwoCircles (a : ℝ) where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  r₁ : ℝ
  r₂ : ℝ
  h_distance : dist O₁ O₂ = a
  h_non_intersecting : dist O₁ O₂ > r₁ + r₂

/-- Common external tangent to two circles -/
def common_external_tangent {a : ℝ} (c : TwoCircles a) : Set (ℝ × ℝ) := sorry

/-- Common internal tangent to two circles -/
def common_internal_tangent {a : ℝ} (c : TwoCircles a) : Set (ℝ × ℝ) := sorry

/-- Intersection points of common external and internal tangents -/
def intersection_points {a : ℝ} (c : TwoCircles a) : Set (ℝ × ℝ) :=
  (common_external_tangent c) ∩ (common_internal_tangent c)

/-- Circle with center at the midpoint of O₁ and O₂, and radius a/2 -/
def midpoint_circle {a : ℝ} (c : TwoCircles a) : Set (ℝ × ℝ) :=
  {p | dist p ((c.O₁ + c.O₂) / 2) = a / 2}

theorem intersection_points_on_circle {a : ℝ} (c : TwoCircles a) :
  intersection_points c ⊆ midpoint_circle c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_circle_l580_58091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_intersection_property_l580_58018

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - a * x^2

-- Define the property that any line with slope ≥ 1 intersects f(x) at most once
def intersectsAtMostOnce (a : ℝ) : Prop :=
  ∀ m b : ℝ, m ≥ 1 → (∃! x : ℝ, 0.5 ≤ x ∧ x ≤ 1 ∧ f a x = m * x + b) ∨
                     (∀ x : ℝ, 0.5 ≤ x → x ≤ 1 → f a x ≠ m * x + b)

-- Theorem statement
theorem f_intersection_property (a : ℝ) :
  intersectsAtMostOnce a → a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_intersection_property_l580_58018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_fx1_l580_58097

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(2x+1)
def domain_f2x1 : Set ℝ := Set.Icc 2 3

-- State the theorem
theorem domain_fx1 (h : ∀ x ∈ domain_f2x1, ∃ y, f (2*x+1) = y) :
  {x : ℝ | ∃ y, f (x+1) = y} = Set.Icc 4 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_fx1_l580_58097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_pairs_l580_58095

/-- The set A defined by the condition |x-a| = 1 -/
def A (a : ℝ) : Set ℝ := {x | |x - a| = 1}

/-- The set B containing 1, -3, and b -/
def B (b : ℝ) : Set ℝ := {1, -3, b}

/-- The proposition that there are exactly 4 pairs (a, b) satisfying the conditions -/
theorem exactly_four_pairs : 
  ∃! s : Finset (ℝ × ℝ), 
    (∀ p : ℝ × ℝ, p ∈ s ↔ A p.1 ⊆ B p.2) ∧ 
    s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_pairs_l580_58095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_expressible_is_eleven_l580_58073

def is_expressible (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    n = (2^a - 2^b) / (2^c - 2^d)

theorem smallest_non_expressible_is_eleven :
  (∀ k < 11, is_expressible k) ∧ ¬is_expressible 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_expressible_is_eleven_l580_58073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_ABC_l580_58064

theorem det_product_ABC {n : Type*} [Fintype n] [DecidableEq n]
  (A B C : Matrix n n ℝ) 
  (hA : Matrix.det A = 3)
  (hB : Matrix.det B = 5)
  (hC : Matrix.det C = 4) :
  Matrix.det (A * B * C) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_ABC_l580_58064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_l580_58075

theorem angle_A_value (A B C : Real) (a b : Real) :
  0 < A ∧ A < π/2 →  -- A is acute
  0 < B ∧ B < π/2 →  -- B is acute
  0 < C ∧ C < π/2 →  -- C is acute
  A + B + C = π →    -- Sum of angles in a triangle
  a = Real.sin A / Real.sin C →  -- Law of Sines
  b = Real.sin B / Real.sin C →  -- Law of Sines
  2 * a * Real.sin B = Real.sqrt 3 * b →  -- Given condition
  A = π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_l580_58075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exp_sum_l580_58004

/-- Distance between two points in R^2 -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The theorem statement -/
theorem min_value_of_exp_sum (x y : ℝ) :
  distance x y 0 4 = distance x y (-2) 0 →
  2^x + 4^y ≥ 4 * Real.sqrt 2 ∧
  ∃ x₀ y₀ : ℝ, distance x₀ y₀ 0 4 = distance x₀ y₀ (-2) 0 ∧
             2^x₀ + 4^y₀ = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exp_sum_l580_58004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_geometric_subsequence_l580_58013

-- Define the sequence b_n
noncomputable def b (n : ℕ+) : ℝ := n + Real.sqrt 2

-- Theorem statement
theorem no_geometric_subsequence :
  ∀ p q r : ℕ+, p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (b q)^2 ≠ (b p) * (b r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_geometric_subsequence_l580_58013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_trial_amounts_l580_58000

/-- The golden ratio, approximately 0.618 --/
noncomputable def φ : ℝ := (Real.sqrt 5 - 1) / 2

/-- The 0.618 method for optimization --/
def golden_section_search (a b : ℝ) : Set ℝ :=
  {a + φ * (b - a), b - φ * (b - a)}

/-- Theorem: Second trial amounts using 0.618 method --/
theorem second_trial_amounts :
  golden_section_search 100 200 = {138.2, 161.8} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_trial_amounts_l580_58000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_OB_l580_58030

noncomputable def angle_AOB : ℝ := Real.pi / 4

def length_AB : ℝ := 2

theorem max_length_OB :
  ∀ (O A B : ℝ × ℝ),
  let angle := angle_AOB
  let ab_length := length_AB
  (A.1 - O.1) * (B.2 - O.2) = (A.2 - O.2) * (B.1 - O.1) →
  (B.1 - O.1) * (A.2 - O.2) = (B.2 - O.2) * (A.1 - O.1) →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = ab_length^2 →
  ∃ (C : ℝ × ℝ), 
    (C.1 - O.1)^2 + (C.2 - O.2)^2 ≤ 8 ∧
    ∀ (D : ℝ × ℝ),
      (D.1 - O.1) * (A.2 - O.2) = (D.2 - O.2) * (A.1 - O.1) →
      (D.1 - O.1)^2 + (D.2 - O.2)^2 ≤ (C.1 - O.1)^2 + (C.2 - O.2)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_OB_l580_58030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_l580_58090

-- Define the walking speed
variable (w : ℝ) (hw : w > 0)

-- Define the distances
variable (x y : ℝ) (hx : x > 0) (hy : y > 0)

-- Define the bicycle speed as 5 times the walking speed
def bicycle_speed (w : ℝ) : ℝ := 5 * w

-- Theorem stating the ratio of distances
theorem distance_ratio (w x y : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) :
  (x / w + (x + y) / (bicycle_speed w) = y / w) → (x / y = 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_l580_58090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distances_l580_58082

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if a point lies on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Theorem about the distances from foci to a point on a hyperbola -/
theorem hyperbola_focal_distances 
  (h : Hyperbola) 
  (p : Point) 
  (f1 f2 : Point) 
  (h_eq : h.b^2 = 27)
  (h_a_pos : h.a > 0)
  (h_on_hyperbola : isOnHyperbola h p)
  (h_asymptote_slope : h.b / h.a = Real.tan (π / 3))
  (h_f1_dist : distance p f1 = 7)
  : distance p f2 = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distances_l580_58082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_row_theorem_l580_58051

theorem random_row_theorem (a : Fin 2010 → Fin 2010) (h : Function.Bijective a) :
  ∃ i j : Fin 2010, i ≠ j ∧
    ((a i + i.val = a j + j.val) ∨ (a i + i.val - (a j + j.val) = 2010) ∨ (a j + j.val - (a i + i.val) = 2010)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_row_theorem_l580_58051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pairs_sum_product_l580_58068

theorem unique_pairs_sum_product (S P : ℝ) (h : S^2 ≥ 4*P) :
  let x₁ := (S + Real.sqrt (S^2 - 4*P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4*P)) / 2
  let y₁ := S - x₁
  let y₂ := S - x₂
  ∀ x y : ℝ, x + y = S ∧ x * y = P ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pairs_sum_product_l580_58068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drink_ratio_theorem_l580_58065

/-- Represents the ratio of flavoring to corn syrup to water in a drink formulation -/
structure DrinkRatio where
  flavoring : ℝ
  corn_syrup : ℝ
  water : ℝ

/-- The standard formulation of the drink -/
def standard : DrinkRatio := sorry

/-- The sport formulation of the drink -/
def sport : DrinkRatio := sorry

/-- The amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℝ := 6

/-- The amount of water in the sport formulation (in ounces) -/
def sport_water : ℝ := 90

theorem drink_ratio_theorem :
  -- In the sport formulation, the ratio of flavoring to corn syrup is three times that of the standard formulation
  sport.flavoring / sport.corn_syrup = 3 * (standard.flavoring / standard.corn_syrup) →
  -- In the sport formulation, the ratio of flavoring to water is half that of the standard formulation
  sport.flavoring / sport.water = (1/2) * (standard.flavoring / standard.water) →
  -- The sport formulation contains 6 ounces of corn syrup and 90 ounces of water
  sport.corn_syrup / sport.water = sport_corn_syrup / sport_water →
  -- The ratio of flavoring to corn syrup to water in the standard formulation is 1:7.5:56.25
  (standard.flavoring : ℝ) / standard.corn_syrup = 1 / 7.5 ∧
  (standard.flavoring : ℝ) / standard.water = 1 / 56.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_drink_ratio_theorem_l580_58065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_rate_approx_five_percent_l580_58021

/-- Calculates the annual interest rate given the principal, final amount, and time period. -/
noncomputable def calculate_interest_rate (principal : ℝ) (final_amount : ℝ) (years : ℝ) : ℝ :=
  ((final_amount / principal) ^ (1 / years)) - 1

/-- Theorem stating that the interest rate for the given investment scenario is approximately 5% -/
theorem investment_interest_rate_approx_five_percent :
  let principal := (8000 : ℝ)
  let final_amount := (9724.05 : ℝ)
  let years := (4 : ℝ)
  let calculated_rate := calculate_interest_rate principal final_amount years
  ∃ ε > 0, abs (calculated_rate - 0.05) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_rate_approx_five_percent_l580_58021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l580_58012

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : x^2 = 4*y

/-- The focus of the parabola x^2 = 4y -/
def focus : ℝ × ℝ := (0, 1)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating the relationship between the point's distance to focus and x-axis -/
theorem parabola_point_distance (P : ParabolaPoint) :
  distance (P.x, P.y) focus = 8 → P.y = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l580_58012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_pole_in_room_l580_58024

/-- The length of the longest pole that can be placed in a rectangular room -/
noncomputable def longest_pole (length width height : ℝ) : ℝ :=
  Real.sqrt (length^2 + width^2 + height^2)

theorem longest_pole_in_room :
  longest_pole 12 8 9 = 17 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_pole_in_room_l580_58024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antenna_height_l580_58039

-- Define the antenna and points
structure Antenna :=
  (height : ℝ)

structure Point :=
  (distance : ℝ)
  (angle : ℝ)

-- Define the problem setup
def antenna_problem (a : Antenna) (p1 p2 p3 : Point) : Prop :=
  p1.distance = 100 ∧
  p2.distance = 200 ∧
  p3.distance = 300 ∧
  p1.angle + p2.angle + p3.angle = 90 ∧
  Real.tan p1.angle = a.height / p1.distance ∧
  Real.tan p2.angle = a.height / p2.distance ∧
  Real.tan p3.angle = a.height / p3.distance

-- Theorem statement
theorem antenna_height (a : Antenna) (p1 p2 p3 : Point) :
  antenna_problem a p1 p2 p3 → a.height = 100 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_antenna_height_l580_58039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_5_l580_58023

-- Define the curve C in parametric form
noncomputable def curve_C (φ : Real) : Real × Real :=
  (1 + Real.sqrt 3 * Real.cos φ, Real.sqrt 3 * Real.sin φ)

-- Define the range of the parameter φ
def φ_range (φ : Real) : Prop :=
  0 ≤ φ ∧ φ ≤ Real.pi

-- Define line l₁ in polar form
def line_l1 (θ : Real) : Real → Prop := fun ρ ↦
  2 * ρ * Real.sin (θ + Real.pi / 3) + 3 * Real.sqrt 3 = 0

-- Define line l₂ in polar form
def line_l2 : Real → Prop := fun θ ↦
  θ = Real.pi / 3

-- Define the intersection point P
noncomputable def point_P : Real × Real :=
  (2, Real.pi / 3)

-- Define the intersection point Q
noncomputable def point_Q : Real × Real :=
  (-3, Real.pi / 3)

-- Theorem to prove
theorem length_PQ_is_5 :
  let (ρ₁, θ₁) := point_P
  let (ρ₂, θ₂) := point_Q
  abs (ρ₁ - ρ₂) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_5_l580_58023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_l580_58002

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / Real.exp x

theorem function_properties_and_inequality (a : ℝ) (m n : ℝ) 
  (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, f a x ≤ 1 / Real.exp 1)
  (h3 : m ≠ n)
  (h4 : m = n * Real.exp (m - n)) :
  (0 < a ∧ a ≤ 1) ∧ m + n > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_l580_58002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_hyperbola_problem_l580_58022

/-- Given a right triangle AOB with ∠O = 90°, where A(0, 3) and OP = 4√2 
    (P being the intersection of diagonals of square ABCD constructed externally on AB),
    prove that the value of k in the hyperbolic function y = k/x that passes through point D is 24. -/
theorem square_hyperbola_problem (O A B D P : ℝ × ℝ) (k : ℝ) : 
  O = (0, 0) →
  A = (0, 3) →
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0 →
  ‖P - O‖ = 4 * Real.sqrt 2 →
  D.2 = k / D.1 →
  k = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_hyperbola_problem_l580_58022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l580_58089

-- Define the curves in polar coordinates
noncomputable def curve1 (φ : Real) : Real := Real.sqrt 3 * Real.cos φ
noncomputable def curve2 (φ : Real) : Real := Real.sin φ

-- Define the range of φ
def φ_range : Set Real := { φ | 0 ≤ φ ∧ φ ≤ Real.pi / 2 }

-- Define the area calculation function
noncomputable def area_between_curves : Real :=
  (1 / 2) * ∫ φ in φ_range, (min (curve1 φ) (curve2 φ))^2

-- State the theorem
theorem area_calculation :
  area_between_curves = (5 * Real.pi / 24) - (Real.sqrt 3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l580_58089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_positive_sufficient_a_positive_not_necessary_a_positive_sufficient_not_necessary_l580_58014

/-- The function f(x) = (1/2)x³ + ax + 4 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^3 + a * x + 4

/-- f(x) is monotonically increasing on ℝ -/
def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

/-- "a > 0" is sufficient for f(x) to be monotonically increasing -/
theorem a_positive_sufficient (a : ℝ) (h : a > 0) : is_monotone_increasing (f a) := by
  sorry

/-- "a > 0" is not necessary for f(x) to be monotonically increasing -/
theorem a_positive_not_necessary : ∃ a : ℝ, a ≤ 0 ∧ is_monotone_increasing (f a) := by
  sorry

/-- Main theorem: "a > 0" is sufficient but not necessary for f(x) to be monotonically increasing -/
theorem a_positive_sufficient_not_necessary :
  (∀ a : ℝ, a > 0 → is_monotone_increasing (f a)) ∧
  (∃ a : ℝ, a ≤ 0 ∧ is_monotone_increasing (f a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_positive_sufficient_a_positive_not_necessary_a_positive_sufficient_not_necessary_l580_58014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l580_58046

/-- Definition of synthetic method -/
def synthetic_method : Prop :=
  (∃ x, x = "A cause-and-effect method and a forward reasoning method")

/-- Definition of analytical method -/
def analytical_method : Prop :=
  (∃ x, x = "A cause-seeking method from the effect and a direct proof method")

/-- Definition of contradiction method -/
def contradiction_method : Prop :=
  (∃ x, x = "A method that assumes the negation of the proposition to be true, derives a contradiction, and concludes that the assumption is false")

/-- Set of statements about proof methods -/
def statements : Fin 5 → Prop
| 1 => (∃ x, x = "The synthetic method is a cause-and-effect method")
| 2 => (∃ x, x = "The synthetic method is a forward reasoning method")
| 3 => (∃ x, x = "The analytical method is a cause-seeking method from the effect")
| 4 => (∃ x, x = "The analytical method is an indirect proof method")
| 5 => (∃ x, x = "The contradiction method is a backward reasoning method")

/-- Theorem stating which statements are correct -/
theorem correct_statements :
  {n : Fin 5 | statements n} = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l580_58046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_cosine_sum_l580_58044

/-- A function f(x) = a*sin(x) + cos(x) that is symmetric about the line x = π/4 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x + Real.cos x

/-- The property of f being symmetric about x = π/4 -/
def is_symmetric_about_pi_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (Real.pi/2 - x)

/-- Theorem stating that if f(x) = a*sin(x) + cos(x) is symmetric about x = π/4, then a = 1 -/
theorem symmetric_sine_cosine_sum (a : ℝ) :
  is_symmetric_about_pi_4 (f a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_cosine_sum_l580_58044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_third_l580_58031

theorem tan_alpha_plus_pi_third (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin (2 * α) = Real.cos α) :
  Real.tan (α + π / 3) = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_third_l580_58031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_equals_vertex_sum_l580_58010

/-- An isosceles triangle in the Cartesian plane -/
structure IsoscelesTriangle where
  a : ℝ  -- x-coordinate of the apex
  b : ℝ  -- x-coordinate of one base vertex (equal to the other base vertex)

/-- The sum of x-coordinates of vertices equals 15 -/
noncomputable def vertexSum (t : IsoscelesTriangle) : ℝ := t.a + 2 * t.b

/-- The sum of x-coordinates of midpoints -/
noncomputable def midpointSum (t : IsoscelesTriangle) : ℝ := (t.a + t.b) / 2 + (t.a + t.b) / 2 + t.b

/-- Theorem: If the sum of x-coordinates of vertices is 15, 
    then the sum of x-coordinates of midpoints is also 15 -/
theorem midpoint_sum_equals_vertex_sum (t : IsoscelesTriangle) 
  (h : vertexSum t = 15) : midpointSum t = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_equals_vertex_sum_l580_58010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_F₁PF₂_l580_58015

/-- The ellipse with equation x²/25 + y²/16 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 25) + (p.2^2 / 16) = 1}

/-- The foci of the ellipse -/
def Foci : Set (ℝ × ℝ) := {(3, 0), (-3, 0)}

/-- A point on the ellipse -/
noncomputable def P : ℝ × ℝ := sorry

/-- The angle F₁PF₂ is 60° -/
noncomputable def angle_F₁PF₂ : ℝ := 60 * Real.pi / 180

/-- The theorem stating that the area of triangle F₁PF₂ is 16√3/3 -/
theorem area_triangle_F₁PF₂ : 
  P ∈ Ellipse → 
  ∃ F₁ F₂, F₁ ∈ Foci ∧ F₂ ∈ Foci ∧ 
  Real.sqrt 3 / 2 * ‖F₁ - P‖ * ‖F₂ - P‖ = 16 * Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_F₁PF₂_l580_58015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l580_58047

open Set Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := log (-x^2 + 4*x + 12)

-- Define the domain of f
def domain : Set ℝ := Ioo (-2) 6

-- Theorem stating the interval of increase for f
theorem interval_of_increase :
  ∀ x ∈ domain, ∀ y ∈ domain, x < y → f x < f y ↔ x ∈ Ico (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l580_58047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_to_find_balogh_l580_58094

/-- Represents a guest at the gathering -/
structure Guest where
  id : Nat
  isBalogh : Bool

/-- Represents the knowledge state of guests -/
def knows (a b : Guest) : Bool :=
  b.isBalogh || (¬a.isBalogh && a.id ≠ b.id)

/-- The gathering of guests -/
structure Gathering where
  n : Nat
  guests : Finset Guest
  balogh : Guest
  h_size : guests.card = n
  h_balogh_in : balogh ∈ guests
  h_balogh_unique : ∀ g ∈ guests, g.isBalogh ↔ g = balogh

/-- A question asked by the journalist -/
structure Question where
  askedGuest : Guest
  aboutGuest : Guest

/-- The result of asking a question -/
def askQuestion (g : Gathering) (q : Question) : Bool :=
  knows q.askedGuest q.aboutGuest

/-- The minimum number of questions needed to find Mr. Balogh -/
theorem min_questions_to_find_balogh (g : Gathering) :
  ∃ (questions : List Question), questions.length = g.n - 1 ∧
  (∀ q : Question, q ∈ questions → q.askedGuest ∈ g.guests ∧ q.aboutGuest ∈ g.guests) ∧
  (∀ sequence : List Question, sequence.length < g.n - 1 →
    ∃ g1 g2 : Guest, g1 ≠ g2 ∧ g1 ∈ g.guests ∧ g2 ∈ g.guests ∧
    (∀ q ∈ sequence, askQuestion g q = true → g1 ≠ q.aboutGuest ∧ g2 ≠ q.aboutGuest) ∧
    (∀ q ∈ sequence, askQuestion g q = false → g1 ≠ q.askedGuest ∧ g2 ≠ q.askedGuest)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_to_find_balogh_l580_58094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l580_58028

-- Define a basic structure for points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define structures for Plane and Line
structure Plane where
  normal : Point
  d : ℝ

structure Line where
  direction : Point
  point : Point

-- Define subset relation
def is_subset (l : Line) (p : Plane) : Prop :=
  ∃ (t : ℝ), p.normal.x * (l.point.x + t * l.direction.x) + 
              p.normal.y * (l.point.y + t * l.direction.y) + 
              p.normal.z * (l.point.z + t * l.direction.z) + p.d = 0

-- Define perpendicularity between line and plane
def is_perpendicular_line_plane (l : Line) (p : Plane) : Prop :=
  l.direction.x * p.normal.x + l.direction.y * p.normal.y + l.direction.z * p.normal.z = 0

-- Define perpendicularity between two lines
def is_perpendicular_line_line (l1 l2 : Line) : Prop :=
  l1.direction.x * l2.direction.x + l1.direction.y * l2.direction.y + l1.direction.z * l2.direction.z = 0

-- Main theorem
theorem perpendicular_condition (α : Plane) (a l : Line) 
  (h_subset : is_subset a α) :
  (∀ (α : Plane) (a l : Line), is_perpendicular_line_plane l α → is_perpendicular_line_line l a) ∧
  (∃ (α : Plane) (a l : Line), is_perpendicular_line_line l a ∧ ¬is_perpendicular_line_plane l α) :=
by
  sorry  -- The proof is omitted as per the instruction


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l580_58028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_when_m_neg_one_intersection_A_B_empty_iff_l580_58040

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - x) - 1 / Real.sqrt (x - 1)

-- Define the domain A
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Define the set B
def B (m : ℝ) : Set ℝ := {x | 2 * m ≤ x ∧ x ≤ 1 - m}

-- Theorem for part 1
theorem union_A_B_when_m_neg_one :
  A ∪ B (-1) = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for part 2
theorem intersection_A_B_empty_iff (m : ℝ) :
  A ∩ B m = ∅ ↔ 0 ≤ m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_when_m_neg_one_intersection_A_B_empty_iff_l580_58040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_l580_58007

/-- Represents the time it takes to fill a pool with two valves -/
noncomputable def fillPoolTime (poolCapacity : ℝ) (valve1Time : ℝ) (valve2ExtraRate : ℝ) : ℝ :=
  let valve1Rate := poolCapacity / valve1Time
  let valve2Rate := valve1Rate + valve2ExtraRate
  let combinedRate := valve1Rate + valve2Rate
  poolCapacity / combinedRate

/-- Theorem: Given the specified conditions, the pool fills in 48 minutes with both valves open -/
theorem pool_fill_time :
  fillPoolTime 12000 120 50 = 48 := by
  -- Unfold the definition of fillPoolTime
  unfold fillPoolTime
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_l580_58007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l580_58072

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4ax -/
structure Parabola where
  a : ℝ
  h_a : 0 < a ∧ a < 1

/-- Represents a circle with center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Theorem statement for the parabola and circle intersection problem -/
theorem parabola_circle_intersection
  (p : Parabola)
  (c : Circle)
  (h_center : c.center = Point.mk (p.a + 4) 0)
  (h_radius : c.radius = 4)
  (M N : Point)
  (h_M : M.y ≥ 0 ∧ M.y^2 = 4 * p.a * M.x)
  (h_N : N.y ≥ 0 ∧ N.y^2 = 4 * p.a * N.x)
  (h_circle_M : (M.x - c.center.x)^2 + M.y^2 = c.radius^2)
  (h_circle_N : (N.x - c.center.x)^2 + N.y^2 = c.radius^2)
  (F : Point)
  (h_F : F = Point.mk p.a 0)
  (P : Point)
  (h_P : P = Point.mk ((M.x + N.x) / 2) ((M.y + N.y) / 2)) :
  (((M.x - F.x)^2 + (M.y - F.y)^2).sqrt + ((N.x - F.x)^2 + (N.y - F.y)^2).sqrt = 8) ∧
  ¬∃ (a : ℝ), 0 < a ∧ a < 1 ∧
    2 * ((P.x - F.x)^2 + (P.y - F.y)^2).sqrt =
    ((M.x - F.x)^2 + (M.y - F.y)^2).sqrt + ((N.x - F.x)^2 + (N.y - F.y)^2).sqrt := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l580_58072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l580_58076

/-- The average speed of a car traveling the same distance in both directions -/
noncomputable def average_speed (v1 v2 : ℝ) : ℝ := (2 * v1 * v2) / (v1 + v2)

/-- Theorem: The average speed of a car traveling at 60 km/h in one direction
    and 100 km/h in the opposite direction for the same distance is 75 km/h -/
theorem car_average_speed :
  let v1 : ℝ := 60
  let v2 : ℝ := 100
  average_speed v1 v2 = 75 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l580_58076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_f_l580_58079

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

theorem max_min_difference_f :
  let a : ℝ := -2
  let b : ℝ := 0
  (∀ x ∈ Set.Icc a b, f x ≤ f a ∧ f b ≤ f x) →
  f a - f b = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_f_l580_58079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rectangles_2016_grid_l580_58029

/-- Represents a grid with n rows and n columns -/
structure Grid (n : ℕ) where
  rows : ℕ
  cols : ℕ
  size_eq : rows = n ∧ cols = n

/-- Represents a rectangle on the grid -/
structure Rectangle (n : ℕ) where
  top : ℕ
  left : ℕ
  bottom : ℕ
  right : ℕ
  valid : top ≤ n ∧ left ≤ n ∧ bottom ≤ n ∧ right ≤ n ∧ top < bottom ∧ left < right

/-- Checks if a set of rectangles covers all edges in the grid -/
def covers_all_edges (n : ℕ) (rectangles : Set (Rectangle n)) : Prop :=
  ∀ (i j : ℕ), i < n ∧ j < n → 
    ∃ (r : Rectangle n), r ∈ rectangles ∧ 
      ((r.top = i ∧ r.left ≤ j ∧ j < r.right) ∨
       (r.bottom = i + 1 ∧ r.left ≤ j ∧ j < r.right) ∨
       (r.left = j ∧ r.top ≤ i ∧ i < r.bottom) ∨
       (r.right = j + 1 ∧ r.top ≤ i ∧ i < r.bottom))

/-- The main theorem: The minimum number of rectangles required to cover all edges in a 2016x2016 grid is 2017 -/
theorem min_rectangles_2016_grid :
  ∀ (g : Grid 2016) (rectangles : Finset (Rectangle 2016)),
    covers_all_edges 2016 (↑rectangles) →
    2017 ≤ rectangles.card :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rectangles_2016_grid_l580_58029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l580_58061

/-- The area of a triangle with vertices at (-4, 8), (0, 0), and (-8, 0) is 32 square units. -/
theorem triangle_area : 
  let v1 : ℝ × ℝ := (-4, 8)
  let v2 : ℝ × ℝ := (0, 0)
  let v3 : ℝ × ℝ := (-8, 0)
  let area := abs ((v1.1 * (v2.2 - v3.2) + v2.1 * (v3.2 - v1.2) + v3.1 * (v1.2 - v2.2)) / 2)
  area = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l580_58061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_price_calculation_l580_58066

theorem coffee_price_calculation (num_coffee_customers : ℕ) 
                                 (num_tea_customers : ℕ) 
                                 (tea_price : ℚ) 
                                 (total_revenue : ℚ) 
                                 (h1 : num_coffee_customers = 7)
                                 (h2 : num_tea_customers = 8)
                                 (h3 : tea_price = 4)
                                 (h4 : total_revenue = 67) :
  (total_revenue - num_tea_customers * tea_price) / num_coffee_customers = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_price_calculation_l580_58066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l580_58084

noncomputable def stick_lengths : List ℝ := [2, 3, 4, 5, 6]

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

def can_form_triangle (lengths : List ℝ) (a b c : ℝ) : Prop :=
  ∃ (l₁ l₂ l₃ : List ℝ), 
    l₁ ++ l₂ ++ l₃ = lengths ∧ 
    a = l₁.sum ∧ b = l₂.sum ∧ c = l₃.sum

theorem max_triangle_area :
  ∀ (a b c : ℝ),
    can_form_triangle stick_lengths a b c →
    is_valid_triangle a b c →
    triangle_area a b c ≤ 6 * Real.sqrt 10 := by
  sorry

#check max_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l580_58084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l580_58053

noncomputable def f (x : ℝ) := Real.sqrt (2 * x - 4) + (x - 5) ^ (1/4 : ℝ)

theorem domain_of_f :
  ∀ x : ℝ, x ∈ Set.Ici 5 ↔ (∃ y : ℝ, f y = f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l580_58053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l580_58083

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := (Real.sqrt (3 * a + 2 * b)) ^ 3

-- State the theorem
theorem diamond_equation_solution :
  ∀ y : ℝ, diamond 6 y = 64 → y = -1 := by
  intro y hyp
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l580_58083
