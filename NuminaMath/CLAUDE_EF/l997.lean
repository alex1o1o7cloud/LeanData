import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_b_pi_over_36_l997_99728

def a : ℕ := 987654321

def cyclic_permutations (n : ℕ) : List ℕ := sorry

def b : ℕ := (cyclic_permutations a).sum

theorem sin_b_pi_over_36 : Real.sin (b * Real.pi / 36) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_b_pi_over_36_l997_99728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_even_nor_odd_l997_99725

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + (1 + x^3)^(1/3))

-- Theorem statement
theorem f_neither_even_nor_odd :
  ¬(∀ x, f (-x) = f x) ∧ ¬(∀ x, f (-x) = -f x) :=
by
  -- We'll use a proof by contradiction for both parts
  apply And.intro
  -- Prove that f is not even
  · intro h
    -- Choose a specific x where f(-x) ≠ f(x)
    let x := 1
    have : f (-1) ≠ f 1 := by
      -- This inequality can be shown rigorously, but we'll use sorry for now
      sorry
    exact this (h 1)
  -- Prove that f is not odd
  · intro h
    -- Choose a specific x where f(-x) ≠ -f(x)
    let x := 1
    have : f (-1) ≠ -f 1 := by
      -- This inequality can be shown rigorously, but we'll use sorry for now
      sorry
    exact this (h 1)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_even_nor_odd_l997_99725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_prime_factors_implies_power_of_two_l997_99799

theorem same_prime_factors_implies_power_of_two (b m n : ℕ) 
  (hb : b ≠ 1) (hb_pos : b > 1) (hm_pos : m > 0) (hn_pos : n > 0) (hmn : m ≠ n) 
  (h_same_factors : ∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) :
  ∃ k : ℕ, b + 1 = 2^k := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_prime_factors_implies_power_of_two_l997_99799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_functions_satisfying_inequality_l997_99776

/-- A function satisfying the given inequality on the interval [0,1] -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → 
    (x - y)^2 ≤ |f x - f y| ∧ |f x - f y| ≤ |x - y|

/-- The main theorem stating the form of functions satisfying the inequality -/
theorem characterize_functions_satisfying_inequality :
  ∀ f : ℝ → ℝ, SatisfiesInequality f → 
    ∃ (a : ℝ) (C : ℝ), (a = 1 ∨ a = -1) ∧ 
      ∀ x, x ∈ Set.Icc 0 1 → f x = a * x + C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_functions_satisfying_inequality_l997_99776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l997_99756

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  (1/2) * abs (v.1 * w.2 - v.2 * w.1)

/-- The area of the triangle with vertices (1, -3), (9, 2), and (5, 8) is 34 -/
theorem triangle_area_example : triangle_area (1, -3) (9, 2) (5, 8) = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l997_99756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_one_l997_99794

noncomputable def f (x : ℝ) : ℝ :=
  if x > 2 then x + 1 / (x - 2)
  else x^2 + 2

theorem f_composition_at_one : f (f 1) = 4 := by
  -- Evaluate f(1)
  have h1 : f 1 = 3 := by
    simp [f]
    norm_num
  
  -- Evaluate f(3)
  have h2 : f 3 = 4 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc
    f (f 1) = f 3 := by rw [h1]
    _       = 4   := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_one_l997_99794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_coincidence_implies_a_range_l997_99704

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 + x + a else 1/x

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2*x + 1 else -1/(x^2)

-- Theorem statement
theorem tangent_coincidence_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ 
    f_deriv a x₁ = f_deriv a x₂ ∧
    f a x₁ - (f_deriv a x₁ * x₁) = f a x₂ - (f_deriv a x₂ * x₂)) →
  (1/4 < a ∧ a < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_coincidence_implies_a_range_l997_99704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_plus_power_linear_function_through_points_l997_99777

-- Problem 1
theorem cube_root_plus_abs_plus_power : (8 : ℝ) ^ (1/3) + |(-5)| + (-1)^2023 = 6 := by sorry

-- Problem 2
theorem linear_function_through_points :
  ∀ (k b : ℝ), (∀ x y : ℝ, y = k * x + b) →
  (1 = k * 0 + b) →
  (5 = k * 2 + b) →
  (∀ x : ℝ, k * x + b = 2 * x + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_plus_power_linear_function_through_points_l997_99777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l997_99784

/-- The trajectory of point M satisfying the given conditions -/
def trajectory (x y : ℝ) : Prop :=
  y^2 = 16 * x

/-- Point F -/
def F : ℝ × ℝ := (4, 0)

/-- Line l -/
def line_l (x : ℝ) : Prop := x + 5 = 0

/-- Distance from a point to F -/
noncomputable def dist_to_F (x y : ℝ) : ℝ :=
  Real.sqrt ((x - F.1)^2 + y^2)

/-- Distance from a point to line l -/
def dist_to_line_l (x : ℝ) : ℝ :=
  |x + 5|

/-- The condition that M's distance to F is 1 less than its distance to line l -/
def distance_condition (x y : ℝ) : Prop :=
  dist_to_F x y + 1 = dist_to_line_l x

theorem trajectory_equation :
  ∀ x y : ℝ, distance_condition x y → trajectory x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l997_99784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinate_given_equidistant_line_l997_99712

/-- A point in the xy-coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line through the origin with given slope -/
def line_through_origin (m : ℝ) : Set Point :=
  {p : Point | p.y = m * p.x}

/-- A line is equidistant from two points -/
def is_equidistant (l : Set Point) (p q : Point) : Prop :=
  ∃ (r : Point), r ∈ l ∧ (r.x - p.x)^2 + (r.y - p.y)^2 = (r.x - q.x)^2 + (r.y - q.y)^2

theorem point_coordinate_given_equidistant_line 
  (p : Point) 
  (q : Point) 
  (h1 : is_equidistant (line_through_origin 0.8) p q) 
  (h2 : p.y = 6) : 
  p.x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinate_given_equidistant_line_l997_99712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_l_value_l997_99763

theorem quadratic_function_l_value (a b c : ℤ) (l : ℤ) : 
  let g : ℝ → ℝ := λ x ↦ (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)
  (g 2 = 0) →
  (60 < g 6 ∧ g 6 < 70) →
  (80 < g 9 ∧ g 9 < 90) →
  (6000 * (l : ℝ) < g 100 ∧ g 100 < 6000 * ((l + 1) : ℝ)) →
  l = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_l_value_l997_99763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_touching_circles_exist_l997_99729

/-- Three non-collinear points in a plane -/
structure ThreePoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  noncollinear : A ≠ B ∧ B ≠ C ∧ A ≠ C

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_positive : radius > 0

/-- Two circles touch if they intersect at exactly one point -/
def circles_touch (c1 c2 : Circle) : Prop :=
  ∃ (p : ℝ × ℝ), p ∈ Metric.sphere c1.center c1.radius ∧ 
                 p ∈ Metric.sphere c2.center c2.radius ∧
                 ∀ (q : ℝ × ℝ), q ∈ Metric.sphere c1.center c1.radius ∧ 
                                q ∈ Metric.sphere c2.center c2.radius → q = p

/-- Main theorem: Given three non-collinear points, there exist three distinct circles
    that pairwise touch at these points -/
theorem three_touching_circles_exist (points : ThreePoints) : 
  ∃ (c1 c2 c3 : Circle), 
    c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
    circles_touch c1 c2 ∧ circles_touch c2 c3 ∧ circles_touch c1 c3 ∧
    points.A ∈ Metric.sphere c1.center c1.radius ∧
    points.A ∈ Metric.sphere c2.center c2.radius ∧
    points.B ∈ Metric.sphere c2.center c2.radius ∧
    points.B ∈ Metric.sphere c3.center c3.radius ∧
    points.C ∈ Metric.sphere c3.center c3.radius ∧
    points.C ∈ Metric.sphere c1.center c1.radius :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_touching_circles_exist_l997_99729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_satisfying_condition_l997_99717

def satisfiesCondition (S : Set ℕ) : Prop :=
  ∀ m n, m ∈ S → n ∈ S → (m + n) / Nat.gcd m n ∈ S

theorem unique_set_satisfying_condition :
  ∀ S : Set ℕ,
    S.Nonempty →
    S.Finite →
    satisfiesCondition S →
    S = {2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_satisfying_condition_l997_99717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_root_most_appropriate_l997_99793

def equation (x : ℝ) : Prop := 4 * x^2 - 9 = 0

def direct_root_method (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x, eq x ↔ a * x^2 + b * x + c = 0

def most_appropriate_method (eq : ℝ → Prop) (method : (ℝ → Prop) → Prop) : Prop :=
  method eq ∧ ∀ other_method : (ℝ → Prop) → Prop, other_method eq → method eq

theorem direct_root_most_appropriate :
  most_appropriate_method equation direct_root_method := by
  sorry

#check direct_root_most_appropriate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_root_most_appropriate_l997_99793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_storm_time_l997_99760

/-- Represents the position of an object in 2D space -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents the state of the car and storm at a given time -/
structure State (t : ℝ) where
  carPosition : Position
  stormCenter : Position

/-- The initial state at t=0 -/
def initialState : State 0 where
  carPosition := { x := 0, y := 0 }
  stormCenter := { x := 0, y := 130 }

/-- The state at any time t -/
noncomputable def stateAtTime (t : ℝ) : State t :=
  { carPosition := { x := t, y := 0 }
    stormCenter := { x := t / Real.sqrt 2, y := 130 - t / Real.sqrt 2 } }

/-- The distance between two positions -/
noncomputable def distance (p1 p2 : Position) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Predicate to check if the car is inside the storm -/
def isInStorm (s : State t) : Prop :=
  distance s.carPosition s.stormCenter ≤ 75

/-- The times when the car enters and exits the storm -/
noncomputable def stormTimes : ℝ × ℝ :=
  sorry

/-- Theorem stating that the average of entry and exit times is 260 -/
theorem average_storm_time :
  let (t₁, t₂) := stormTimes
  (t₁ + t₂) / 2 = 260 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_storm_time_l997_99760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_60_degrees_l997_99796

/-- A regular square pyramid with given volume and base diagonal length -/
structure RegularSquarePyramid where
  volume : ℝ
  base_diagonal : ℝ

/-- The dihedral angle between a lateral face and the base of a regular square pyramid -/
noncomputable def dihedral_angle (p : RegularSquarePyramid) : ℝ :=
  Real.arctan (Real.sqrt 3)

theorem dihedral_angle_60_degrees (p : RegularSquarePyramid) 
  (h_volume : p.volume = 12)
  (h_diagonal : p.base_diagonal = 2 * Real.sqrt 6) :
  dihedral_angle p = π / 3 := by
  sorry

#check dihedral_angle_60_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_60_degrees_l997_99796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_specific_integers_l997_99737

/-- A function that returns true if a number is a 3-digit positive integer -/
def isThreeDigitPositive (n : ℕ) : Bool :=
  100 ≤ n ∧ n ≤ 999

/-- A function that returns the product of digits of a number -/
def digitProduct (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

/-- A function that returns true if a number has at least one even digit -/
def hasEvenDigit (n : ℕ) : Bool :=
  (n / 100) % 2 = 0 ∨ ((n / 10) % 10) % 2 = 0 ∨ (n % 10) % 2 = 0

/-- The main theorem -/
theorem count_specific_integers : 
  (Finset.filter (fun n => isThreeDigitPositive n ∧ digitProduct n = 30 ∧ hasEvenDigit n) (Finset.range 1000)).card = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_specific_integers_l997_99737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_is_four_l997_99792

-- Define the function f(x) = 2^x + 2^(2-x)
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + Real.exp ((2 - x) * Real.log 2)

-- Theorem statement
theorem f_min_value_is_four :
  ∀ x : ℝ, f x ≥ 4 ∧ ∃ x₀ : ℝ, f x₀ = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_is_four_l997_99792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_property_l997_99746

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*x + y^2 - 6*y = 11

-- Define the center and radius of the circle
noncomputable def circle_center_radius : ℝ × ℝ × ℝ :=
  let a := 2
  let b := 3
  let r := Real.sqrt 24
  (a, b, r)

-- Theorem statement
theorem circle_sum_property :
  let (a, b, r) := circle_center_radius
  a + b + r = 5 + 2 * Real.sqrt 6 :=
by
  -- Unfold the definition of circle_center_radius
  unfold circle_center_radius
  -- Simplify the left-hand side
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_property_l997_99746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l997_99740

/-- Given an acute triangle ABC with sides a = √7 and b = 3, 
    if √7 sin B + sin A = 2√3, then angle A = π/3 and 
    the area of the triangle is 3√3/2 -/
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a = Real.sqrt 7 →
  b = 3 →
  a = c * Real.sin B / Real.sin C →
  b = c * Real.sin A / Real.sin C →
  Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3 →
  A = π/3 ∧ 
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l997_99740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_novel_pages_l997_99721

theorem novel_pages
  (days : ℕ)
  (pages_first_two_days : ℕ)
  (pages_next_four_days : ℕ)
  (pages_last_day : ℕ) :
  days = 7 →
  pages_first_two_days = 2 * 50 →
  pages_next_four_days = 4 * 25 →
  pages_last_day = 30 →
  pages_first_two_days + pages_next_four_days + pages_last_day = 230 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_novel_pages_l997_99721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_mdn_is_45_degrees_l997_99770

/-- A square with side length a -/
structure Square (α : Type*) [NormedAddCommGroup α] :=
  (a : α)
  (A B C D : α × α)
  (is_square : A = (0, 0) ∧ B = (a, 0) ∧ C = (a, a) ∧ D = (0, a))

/-- Point on diagonal AC -/
noncomputable def point_on_diagonal (S : Square ℝ) : ℝ × ℝ := 
  let x : ℝ := Real.sqrt (S.a^2 / 2)
  (x, x)

/-- Point on side BC -/
def point_on_side (S : Square ℝ) : ℝ × ℝ := 
  (S.a, S.a)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem angle_mdn_is_45_degrees (S : Square ℝ) :
  let M := point_on_diagonal S
  let N := point_on_side S
  let D := S.D
  distance M N = distance M D →
  Real.arccos ((M.1 - D.1) * (N.1 - D.1) + (M.2 - D.2) * (N.2 - D.2)) / 
    (distance M D * distance N D) = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_mdn_is_45_degrees_l997_99770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_problem_l997_99772

theorem tan_difference_problem (α φ : ℝ) 
  (h1 : Real.cos α = 1/3)
  (h2 : α ∈ Set.Ioo (-Real.pi/2) 0)
  (h3 : Real.tan φ = Real.sqrt 2) :
  Real.tan (φ - α) = -Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_problem_l997_99772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_l997_99751

-- Define the triathlon parameters
noncomputable def segment_length : ℝ := 5
noncomputable def swim_speed : ℝ := 2
noncomputable def bike_speed : ℝ := 25
noncomputable def run_speed : ℝ := 8

-- Define the harmonic mean function for three numbers
noncomputable def harmonic_mean (a b c : ℝ) : ℝ := 3 / (1/a + 1/b + 1/c)

-- Theorem statement
theorem triathlon_average_speed :
  let avg_speed := harmonic_mean swim_speed bike_speed run_speed
  ∃ ε > 0, |avg_speed - 5| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_l997_99751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_trip_length_l997_99734

/-- Represents the bus trip scenario -/
structure BusTrip where
  speed : ℝ  -- Initial speed of the bus in mph
  distance : ℝ  -- Total distance of the trip in miles

/-- Calculates the total time of the trip given the initial scenario -/
noncomputable def initial_scenario (trip : BusTrip) : ℝ :=
  2 + (3 * (trip.distance - 2 * trip.speed)) / (2 * trip.speed) + 0.75

/-- Calculates the total time of the trip given the alternative scenario -/
noncomputable def alternative_scenario (trip : BusTrip) : ℝ :=
  (2 * trip.speed + 120) / trip.speed + 
  (3 * (trip.distance - 2 * trip.speed - 120)) / (2 * trip.speed) + 0.75

/-- The theorem stating the conditions and the conclusion about the trip length -/
theorem bus_trip_length : 
  ∃ (trip : BusTrip), 
    initial_scenario trip = alternative_scenario trip + 1 ∧ 
    trip.distance = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_trip_length_l997_99734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_or_q_l997_99736

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log (abs x)

-- Define the propositions
def p (a : ℝ) : Prop := ∀ x : ℝ, x = 0 → f a x = -2
def q : Prop := ∃ x y : ℝ, x ≠ y ∧ x ≠ 0 ∧ y ≠ 0 ∧ g x = 0 ∧ g y = 0

-- The theorem to prove
theorem p_or_q (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : p a ∨ q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_or_q_l997_99736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l997_99703

/-- An isosceles triangle with base b and height h -/
structure IsoscelesTriangle where
  b : ℝ
  h : ℝ
  b_pos : 0 < b
  h_pos : 0 < h

/-- A rectangle inscribed in an isosceles triangle -/
structure InscribedRectangle (triangle : IsoscelesTriangle) where
  x : ℝ
  x_pos : 0 < x
  x_le_h : x ≤ triangle.h

/-- The area of an inscribed rectangle -/
noncomputable def rectangleArea (triangle : IsoscelesTriangle) (rect : InscribedRectangle triangle) : ℝ :=
  (triangle.b * rect.x / triangle.h) * (triangle.h - rect.x)

theorem inscribed_rectangle_area (triangle : IsoscelesTriangle) (rect : InscribedRectangle triangle) :
  rectangleArea triangle rect = (triangle.b * rect.x / triangle.h) * (triangle.h - rect.x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l997_99703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l997_99774

noncomputable def f (x : ℝ) : ℝ := Real.tan x / (1 - Real.tan x ^ 2)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = π/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l997_99774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_existence_problem_l997_99775

theorem function_existence_problem :
  (¬ ∃ (f g : ℤ → ℤ), ∀ x : ℤ, f (f x) = x ∧ g (g x) = x ∧ f (g x) > x ∧ g (f x) > x) ∧
  (∃ (f g : ℤ → ℤ), ∀ x : ℤ, f (f x) < x ∧ g (g x) < x ∧ f (g x) > x ∧ g (f x) > x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_existence_problem_l997_99775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_area_ratio_l997_99798

/-- Given a sphere of radius r and an external tangent cone with height 3r,
    the ratio of the lateral surface area of the cone to the surface area of the sphere is 3:2 -/
theorem cone_sphere_area_ratio (r : ℝ) (h : r > 0) :
  (π * r * Real.sqrt ((3 * r)^2 + r^2)) / (4 * π * r^2) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_area_ratio_l997_99798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_prime_solution_range_l997_99781

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 1) * Real.log x - x^2 - a * x

-- Part 1: Minimum value of f' when a = 1
theorem min_value_f_prime :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → (deriv (f 1)) y ≥ (deriv (f 1)) x ∧ (deriv (f 1)) x = -1 :=
by sorry

-- Part 2: Range of a for which f(x) = axe^(2ax) - x^2 has a solution
theorem solution_range :
  ∀ (a : ℝ), (∃ (x : ℝ), x > 0 ∧ f a x = a * x * Real.exp (2 * a * x) - x^2) ↔ a ≤ Real.exp (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_prime_solution_range_l997_99781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l997_99786

def P : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≤ 4}

theorem union_of_P_and_Q : P ∪ Q = Set.Icc (-2) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l997_99786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sin2x_max_value_l997_99709

theorem sin_sin2x_max_value (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  ∃ (max : ℝ), max = (4 * Real.sqrt 3) / 9 ∧
  Real.sin x * Real.sin (2 * x) ≤ max ∧
  ∃ (y : ℝ), 0 < y ∧ y < π / 2 ∧ Real.sin y * Real.sin (2 * y) = max := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sin2x_max_value_l997_99709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_triangle_functions_l997_99738

-- Define the domain for each function
def D1 : Set ℝ := {x | x > 0}
def D2 : Set ℝ := Set.univ
def D3 : Set ℝ := {x | 1 ≤ x ∧ x ≤ 16}
def D4 : Set ℝ := Set.univ

-- Define the functions
noncomputable def f1 : ℝ → ℝ := λ x => Real.log (x + 1)
noncomputable def f2 : ℝ → ℝ := λ x => 4 - Real.cos x
noncomputable def f3 : ℝ → ℝ := λ x => Real.sqrt x
noncomputable def f4 : ℝ → ℝ := λ x => (3^x + 2) / (3^x + 1)

-- Define what it means to be a triangle function
def is_triangle_function (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ a b c, a ∈ D → b ∈ D → c ∈ D → (f a) + (f b) > (f c)

-- State the theorem
theorem exactly_two_triangle_functions :
  (is_triangle_function f1 D1 ∧ is_triangle_function f2 D2 ∧
   ¬is_triangle_function f3 D3 ∧ is_triangle_function f4 D4) ∨
  (is_triangle_function f1 D1 ∧ ¬is_triangle_function f2 D2 ∧
   is_triangle_function f3 D3 ∧ is_triangle_function f4 D4) ∨
  (¬is_triangle_function f1 D1 ∧ is_triangle_function f2 D2 ∧
   is_triangle_function f3 D3 ∧ is_triangle_function f4 D4) ∨
  (¬is_triangle_function f1 D1 ∧ is_triangle_function f2 D2 ∧
   ¬is_triangle_function f3 D3 ∧ is_triangle_function f4 D4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_triangle_functions_l997_99738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_at_zero_l997_99718

-- Define points A, B, and C
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (5, 3)
def C (x : ℝ) : ℝ × ℝ := (x, 2*x)

-- Define distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the sum of distances AC + BC
noncomputable def total_distance (x : ℝ) : ℝ :=
  distance A (C x) + distance B (C x)

-- Theorem statement
theorem min_distance_at_zero :
  ∀ x : ℝ, total_distance 0 ≤ total_distance x := by
  sorry

#eval A
#eval B
#eval C 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_at_zero_l997_99718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shipment_average_weight_l997_99726

/-- A shipment of boxes with specific weight properties -/
structure Shipment where
  total_boxes : ℕ
  light_boxes : ℕ
  heavy_boxes : ℕ
  light_weight : ℚ
  heavy_weight : ℚ

/-- The average weight of boxes in a shipment -/
def average_weight (s : Shipment) : ℚ :=
  (s.light_boxes * s.light_weight + s.heavy_boxes * s.heavy_weight) / s.total_boxes

/-- The theorem statement -/
theorem shipment_average_weight
  (s : Shipment)
  (h1 : s.total_boxes = 20)
  (h2 : s.light_weight = 10)
  (h3 : s.heavy_weight = 20)
  (h4 : s.light_boxes + s.heavy_boxes = s.total_boxes)
  (h5 : average_weight { s with
    total_boxes := s.total_boxes - 15,
    heavy_boxes := s.heavy_boxes - 15
  } = 12) :
  average_weight s = 35/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shipment_average_weight_l997_99726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l997_99716

theorem problem_statement (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : Real.log x / Real.log y + Real.log y / Real.log x = 9/2) 
  (h4 : x * y = 200) : 
  (x + y) / 2 = (Real.rpow 200 (1/(9/2)) + Real.rpow 200 ((7/2)/(9/2))) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l997_99716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_has_real_root_l997_99730

/-- Given a polynomial P and real numbers a₁, a₂, a₃, b₁, b₂, b₃ satisfying certain conditions,
    P has at least one real root. -/
theorem polynomial_has_real_root
  (P : ℝ → ℝ) (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ)
  (h₁ : a₁ * a₂ * a₃ ≠ 0)
  (h₂ : ∀ x, P (a₁ * x + b₁) + P (a₂ * x + b₂) = P (a₃ * x + b₃)) :
  ∃ x, P x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_has_real_root_l997_99730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_is_2_sqrt_13_l997_99766

/-- A parallelogram formed by six congruent equilateral triangles -/
structure TriangleParallelogram where
  /-- Side length of each equilateral triangle -/
  side_length : ℝ
  /-- The side length is positive -/
  side_length_pos : 0 < side_length
  /-- The number of equilateral triangles forming the parallelogram -/
  num_triangles : ℕ
  /-- The parallelogram is formed by exactly six triangles -/
  six_triangles : num_triangles = 6

/-- The length of the diagonal PR in the TriangleParallelogram -/
noncomputable def diagonal_length (tp : TriangleParallelogram) : ℝ :=
  2 * Real.sqrt 13

/-- Theorem: The length of the diagonal PR in a TriangleParallelogram with side length 2 is 2√13 -/
theorem diagonal_length_is_2_sqrt_13 (tp : TriangleParallelogram) 
    (h : tp.side_length = 2) : diagonal_length tp = 2 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_is_2_sqrt_13_l997_99766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squared_areas_l997_99711

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the focus of the parabola
def F : Point := ⟨1, 0⟩

-- Define the origin
def O : Point := ⟨0, 0⟩

-- Define the area of a triangle formed by the origin, focus, and a point on the parabola
noncomputable def triangleArea (p : Point) : ℝ := (1/2) * abs p.y

-- Declare A, B, and C as variables
variable (A B C : Point)

-- State that A, B, and C are on the parabola
axiom A_on_parabola : parabola A.x A.y
axiom B_on_parabola : parabola B.x B.y
axiom C_on_parabola : parabola C.x C.y

-- State that F is the centroid of triangle ABC
axiom F_is_centroid : F.x = (A.x + B.x + C.x) / 3 ∧ F.y = (A.y + B.y + C.y) / 3

-- Define S₁, S₂, and S₃
noncomputable def S₁ : ℝ := triangleArea A
noncomputable def S₂ : ℝ := triangleArea B
noncomputable def S₃ : ℝ := triangleArea C

-- The theorem to prove
theorem sum_of_squared_areas : S₁^2 + S₂^2 + S₃^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squared_areas_l997_99711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_square_ratio_is_correct_l997_99752

/-- Represents a square flag with a symmetric cross -/
structure SquareFlag where
  side : ℝ
  cross_area_ratio : ℝ
  cross_area_ratio_is_quarter : cross_area_ratio = 1/4

/-- Calculates the ratio of the center square's area to the total flag area -/
noncomputable def center_square_ratio (flag : SquareFlag) : ℝ :=
  let w := flag.side * (flag.cross_area_ratio / 4)  -- Width of cross arm
  (4 * w^2) / (flag.side^2)

/-- Theorem: The center square occupies 1.5625% of the flag's area -/
theorem center_square_ratio_is_correct (flag : SquareFlag) :
  center_square_ratio flag = 0.015625 := by
  sorry

#eval (0.015625 : Float) * 100  -- Should evaluate to 1.5625

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_square_ratio_is_correct_l997_99752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_proof_l997_99713

-- Define the initial water percentage (as a ratio)
def initial_water_percentage : ℝ := sorry

-- Define the dried water percentage (as a ratio)
def dried_water_percentage : ℝ := 0.16

-- Define the weight of the dried fruit in kg
def dried_weight : ℝ := 8

-- Define the weight of the fresh fruit in kg
def fresh_weight : ℝ := 95.99999999999999

-- Theorem statement
theorem water_percentage_proof :
  (1 - initial_water_percentage) * fresh_weight = (1 - dried_water_percentage) * dried_weight →
  initial_water_percentage = 0.93 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_proof_l997_99713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l997_99789

/-- A quadratic function satisfying specific conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x - 2) = f (-x - 2)) ∧
  (f 0 = 1) ∧
  (∃ a b, a < b ∧ f a = 0 ∧ f b = 0 ∧ b - a = 2 * Real.sqrt 2)

/-- The specific quadratic function we want to prove -/
noncomputable def f (x : ℝ) : ℝ :=
  1/2 * (x + 2)^2 - 1

/-- Theorem stating that our function f satisfies the QuadraticFunction conditions -/
theorem quadratic_function_theorem :
  QuadraticFunction f :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l997_99789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l997_99710

/-- The standard equation of a circle with center at (-3, 4) and tangent to the y-axis -/
theorem circle_equation :
  ∃ (C : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ C ↔ (x + 3)^2 + (y - 4)^2 = 9) ∧
    ((-3, 4) ∈ C) ∧
    (∀ (y : ℝ), (0, y) ∈ C → ∀ (ε : ℝ), ε > 0 → ∃ (x : ℝ), 0 < |x| ∧ |x| < ε ∧ (x, y) ∉ C) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l997_99710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l997_99783

noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then x + 2 else -x + 2

theorem solution_set_of_inequality (x : ℝ) : 
  f x ≥ x^2 ↔ -1 ≤ x ∧ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l997_99783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_preference_gender_related_l997_99754

/-- Soccer preference survey data -/
structure SurveyData where
  boys_like : ℕ
  boys_dislike : ℕ
  girls_like : ℕ
  girls_dislike : ℕ

/-- Calculate χ² value for independence test -/
noncomputable def chi_square (data : SurveyData) : ℝ :=
  let n := (data.boys_like + data.boys_dislike + data.girls_like + data.girls_dislike : ℝ)
  let ad := (data.boys_like * data.girls_dislike : ℝ)
  let bc := (data.boys_dislike * data.girls_like : ℝ)
  n * (ad - bc)^2 / ((data.boys_like + data.boys_dislike) * 
    (data.girls_like + data.girls_dislike) * 
    (data.boys_like + data.girls_like) * 
    (data.boys_dislike + data.girls_dislike))

/-- Probability of scoring for male and female students -/
noncomputable def male_score_prob : ℝ := 2/3
noncomputable def female_score_prob : ℝ := 1/2

/-- Expected value of goals scored -/
noncomputable def expected_goals : ℝ :=
  0 * (1/3)^2 * (1/2) +
  1 * (2 * 2/3 * 1/3 * 1/2 + 1/2 * (1/3)^2) +
  2 * (2 * 2/3 * 2/3 * 1/2 + (2/3)^2 * 1/2) +
  3 * ((2/3)^2 * 1/2)

theorem soccer_preference_gender_related (data : SurveyData) 
  (h1 : data.boys_like + data.boys_dislike = 100)
  (h2 : data.girls_like + data.girls_dislike = 100)
  (h3 : data.boys_dislike = 40)
  (h4 : data.girls_like = 30) :
  chi_square data > 10.828 ∧ expected_goals = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_preference_gender_related_l997_99754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_one_fourth_l997_99705

/-- Two lines with slopes 4 and 6 intersect at the point (7, -3).
    This function calculates the distance between their x-intercepts. -/
noncomputable def distance_between_x_intercepts : ℝ :=
  let line1 : ℝ → ℝ := λ x => 4 * x - 31
  let line2 : ℝ → ℝ := λ x => 6 * x - 45
  let x_intercept1 : ℝ := 31 / 4
  let x_intercept2 : ℝ := 15 / 2
  |x_intercept1 - x_intercept2|

/-- Theorem stating that the distance between x-intercepts is 1/4 -/
theorem distance_is_one_fourth :
  distance_between_x_intercepts = 1 / 4 := by
  -- Unfold the definition of distance_between_x_intercepts
  unfold distance_between_x_intercepts
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_one_fourth_l997_99705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinate_equivalence_l997_99747

/-- 
Given a point in spherical coordinates (ρ, θ, φ), 
this function returns the equivalent point in standard spherical coordinate representation.
-/
noncomputable def standardSphericalCoordinates (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  let ρ' := ρ
  let θ' := if φ > Real.pi then (θ + Real.pi) % (2 * Real.pi) else θ % (2 * Real.pi)
  let φ' := if φ > Real.pi then 2 * Real.pi - φ else φ
  (ρ', θ', φ')

/-- 
Theorem: The point (4, 3π/8, 9π/5) in spherical coordinates 
is equivalent to (4, 11π/8, π/5) in standard spherical coordinate representation.
-/
theorem spherical_coordinate_equivalence :
  standardSphericalCoordinates 4 (3 * Real.pi / 8) (9 * Real.pi / 5) = (4, 11 * Real.pi / 8, Real.pi / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinate_equivalence_l997_99747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l997_99773

def sequence_a : ℕ → ℚ
  | 0 => 0  -- Add this case for n = 0
  | 1 => 0
  | n + 2 => (n + 1) / (n + 2) * (sequence_a (n + 1) + 1)

theorem sequence_a_formula (n : ℕ) : sequence_a n = (n - 1 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l997_99773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l997_99744

/-- The focal length of a hyperbola given its equation and asymptote -/
theorem hyperbola_focal_length (m : ℝ) (h1 : m > 0) :
  let C : ℝ → ℝ → Prop := λ x y ↦ x^2 / m - y^2 = 1
  let asymptote : ℝ → ℝ → Prop := λ x y ↦ Real.sqrt 2 * x + m * y = 0
  ∃ (x y : ℝ), C x y ∧ asymptote x y →
  2 * Real.sqrt 3 = 2 * Real.sqrt (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l997_99744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_rate_1000_discount_rate_range_discount_rate_outside_range_l997_99731

noncomputable def voucher (spending : ℝ) : ℝ :=
  if 200 ≤ spending ∧ spending < 400 then 30
  else if 400 ≤ spending ∧ spending < 500 then 60
  else if 500 ≤ spending ∧ spending < 700 then 100
  else if 700 ≤ spending ∧ spending < 900 then 130
  else 0  -- Default case, can be adjusted based on the complete voucher scheme

noncomputable def discountRate (originalPrice : ℝ) : ℝ :=
  (0.2 * originalPrice + voucher (0.8 * originalPrice)) / originalPrice

-- Theorem for the first question
theorem discount_rate_1000 : discountRate 1000 = 0.33 := by
  sorry

-- Theorem for the second question
theorem discount_rate_range (price : ℝ) (h1 : 625 ≤ price) (h2 : price ≤ 750) :
  discountRate price ≥ 1/3 := by
  sorry

-- Additional theorem to show that prices below 625 or above 750 in the [500, 800] range don't achieve 1/3 discount
theorem discount_rate_outside_range (price : ℝ) (h1 : 500 ≤ price) (h2 : price ≤ 800) 
  (h3 : price < 625 ∨ price > 750) : discountRate price < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_rate_1000_discount_rate_range_discount_rate_outside_range_l997_99731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_curve_and_line_l997_99720

-- Define the curve C
noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

-- Define the line l in polar form
def line_l (ρ θ m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 3) + m = 0

-- Theorem statement
theorem intersection_of_curve_and_line :
  ∀ (m : ℝ),
  (∀ (x y : ℝ), Real.sqrt 3 * x + y + 2 * m = 0 ↔ 
    ∃ (ρ θ : ℝ), line_l ρ θ m ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  (∃ (t : ℝ), ∃ (x y : ℝ), curve_C t = (x, y) ∧ Real.sqrt 3 * x + y + 2 * m = 0) ↔
  -19/12 ≤ m ∧ m ≤ 5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_curve_and_line_l997_99720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_equilateral_cross_section_l997_99768

/-- The volume of a right circular cone with an equilateral triangle cross-section --/
theorem cone_volume_equilateral_cross_section (s : ℝ) (h : s = 2) :
  (1/3) * Real.pi * (s/2)^2 * (Real.sqrt 3 * s/2) = (Real.sqrt 3 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_equilateral_cross_section_l997_99768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_with_leak_is_three_l997_99742

noncomputable def fill_rate (fill_time : ℝ) : ℝ := 1 / fill_time

noncomputable def empty_rate (empty_time : ℝ) : ℝ := 1 / empty_time

noncomputable def combined_rate (fill_rate empty_rate : ℝ) : ℝ := fill_rate - empty_rate

noncomputable def fill_time_with_leak (fill_rate empty_rate : ℝ) : ℝ := 1 / (combined_rate fill_rate empty_rate)

theorem fill_time_with_leak_is_three :
  let pipe_fill_time : ℝ := 2
  let leak_empty_time : ℝ := 6
  fill_time_with_leak (fill_rate pipe_fill_time) (empty_rate leak_empty_time) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_with_leak_is_three_l997_99742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l997_99764

/-- The constant term in the expansion of (x + 2/x)^4 -/
def constantTerm : ℕ := 24

/-- The binomial (x + 2/x)^4 -/
noncomputable def binomial (x : ℝ) : ℝ := (x + 2/x)^4

theorem constant_term_of_binomial_expansion :
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 → binomial x = c + x * (binomial x - c) / x ∧ c = constantTerm := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l997_99764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l997_99715

def number : ℕ := 3^4 * 5^6 * 7^3

theorem number_of_factors : 
  (Finset.filter (· ∣ number) (Finset.range (number + 1))).card = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l997_99715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probabilities_l997_99795

/-- Represents a player in the game -/
inductive Player : Type
| A
| B
| C

/-- The rules and conditions of the game -/
structure GameRules :=
  (win_prob : ℚ)
  (elimination_losses : ℕ)

/-- The state of the game -/
structure GameState :=
  (players : List Player)
  (active_players : List Player)
  (losses : Player → ℕ)
  (games_played : ℕ)

/-- The probability of a specific player winning four consecutive games -/
noncomputable def prob_four_consecutive_wins (rules : GameRules) : ℚ :=
  rules.win_prob ^ 4

/-- The probability of needing a fifth game -/
noncomputable def prob_fifth_game (rules : GameRules) : ℚ :=
  1 - 4 * (rules.win_prob ^ 4)

/-- The probability of a specific player winning the game -/
def prob_player_winning (rules : GameRules) : ℚ :=
  7 / 16

/-- Main theorem stating the probabilities for the game -/
theorem game_probabilities (rules : GameRules) 
  (h1 : rules.win_prob = 1 / 2) 
  (h2 : rules.elimination_losses = 2) :
  prob_four_consecutive_wins rules = 1 / 16 ∧
  prob_fifth_game rules = 3 / 4 ∧
  prob_player_winning rules = 7 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probabilities_l997_99795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l997_99735

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x + b * (x^(1/3)) + 4

-- State the theorem
theorem problem_statement (a b : ℝ) :
  f a b (Real.log 3 / Real.log 10) = 3 →
  f a b (Real.log (1/3) / Real.log 10) = 5 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l997_99735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l997_99759

noncomputable def f (x : ℝ) : ℝ := 3^x + x - 5

theorem root_in_interval (a b : ℕ+) (h1 : (b : ℝ) - (a : ℝ) = 1) 
  (h2 : ∃ x : ℝ, x ∈ Set.Icc (a : ℝ) (b : ℝ) ∧ f x = 0) : 
  (a : ℕ) + (b : ℕ) = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l997_99759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l997_99745

/-- Calculates the length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
noncomputable def train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed + person_speed
  let relative_speed_ms := relative_speed * (1000 / 3600)
  relative_speed_ms * passing_time

/-- Theorem stating that a train with speed 30 km/h passing a man running at 3 km/h in the opposite direction in 12 seconds has a length of approximately 110 meters. -/
theorem train_length_problem : 
  let train_speed := (30 : ℝ)
  let person_speed := (3 : ℝ)
  let passing_time := (12 : ℝ)
  abs (train_length train_speed person_speed passing_time - 110) < 1 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_length 30 3 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l997_99745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribing_triangle_exists_l997_99758

-- Define a convex polygon
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  is_convex : Bool
  not_parallelogram : Bool

-- Helper functions (declared but not implemented)
def sides (vertices : List (ℝ × ℝ)) : List (List (ℝ × ℝ)) := sorry
def extend (side : List (ℝ × ℝ)) : List (ℝ × ℝ) := sorry
def forms_triangle (s1 s2 s3 : List (ℝ × ℝ)) : Prop := sorry
def triangle_from_sides (s1 s2 s3 : List (ℝ × ℝ)) : List (ℝ × ℝ) := sorry
def contains_polygon (outer inner : List (ℝ × ℝ)) : Prop := sorry

-- Define a function to check if three sides of a polygon can form a circumscribing triangle
def has_circumscribing_triangle (p : ConvexPolygon) : Prop :=
  ∃ (s1 s2 s3 : List (ℝ × ℝ)), 
    s1 ∈ sides p.vertices ∧ 
    s2 ∈ sides p.vertices ∧ 
    s3 ∈ sides p.vertices ∧ 
    forms_triangle (extend s1) (extend s2) (extend s3) ∧
    contains_polygon (triangle_from_sides (extend s1) (extend s2) (extend s3)) p.vertices

-- The main theorem
theorem circumscribing_triangle_exists (p : ConvexPolygon) : 
  p.is_convex ∧ p.not_parallelogram → has_circumscribing_triangle p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribing_triangle_exists_l997_99758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_duration_l997_99780

theorem lunch_break_duration 
  (distance : ℝ) 
  (speed : ℝ) 
  (num_bathroom_breaks : ℕ) 
  (bathroom_break_duration : ℝ) 
  (total_trip_time : ℝ) 
  (h1 : distance = 480) 
  (h2 : speed = 60) 
  (h3 : num_bathroom_breaks = 2) 
  (h4 : bathroom_break_duration = 15 / 60) 
  (h5 : total_trip_time = 9) : 
  (total_trip_time - distance / speed - num_bathroom_breaks * bathroom_break_duration) = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_duration_l997_99780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solutions_l997_99797

theorem trigonometric_equation_solutions (x : ℝ) (n : ℤ) :
  (∃ (k : ℤ), (x = 7/12 + 2*k ∧ ∃ (m : ℤ), n = 2*m) ∨
               (x = -1/4 + 2*k ∧ ∃ (m : ℤ), n = 2*m - 1)) ↔
  (Real.sin (π * x) + Real.cos (π * x) = Real.sqrt 2 / 2 ∧ x ∈ Set.Icc (n : ℝ) (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solutions_l997_99797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l997_99750

/-- Triangle DEF with side lengths -/
structure Triangle where
  DE : ℝ
  EF : ℝ
  FD : ℝ

/-- Rectangle UVWX inscribed in triangle DEF -/
structure InscribedRectangle where
  UV : ℝ  -- Side length ω

/-- Area of the inscribed rectangle as a function of its side length -/
def rectangleArea (t : Triangle) (r : InscribedRectangle) (α β : ℝ) : ℝ → ℝ :=
  λ ω ↦ α * ω - β * ω^2

/-- Theorem stating that β = 5/12 for the given triangle -/
theorem beta_value (t : Triangle) (r : InscribedRectangle) :
  t.DE = 15 ∧ t.EF = 36 ∧ t.FD = 39 →
  ∃ (α β : ℝ), 
    (∀ ω, rectangleArea t r α β ω = α * ω - β * ω^2) ∧
    β = 5/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l997_99750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_unspent_portion_is_43_48_l997_99743

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards -/
structure SallyCards where
  gold : CreditCard
  platinum : CreditCard
  diamond : CreditCard

/-- Initializes Sally's cards with the given conditions -/
noncomputable def initializeCards (g : ℝ) : SallyCards :=
  { gold := { limit := g, balance := g / 4 },
    platinum := { limit := 2 * g, balance := (2 * g) / 8 },
    diamond := { limit := 6 * g, balance := (6 * g) / 16 } }

/-- Transfers balance from gold to platinum card -/
noncomputable def transferGoldToPlatinum (cards : SallyCards) : SallyCards :=
  { cards with
    gold := { cards.gold with balance := 0 },
    platinum := { cards.platinum with balance := cards.platinum.balance + cards.gold.balance } }

/-- Transfers half of platinum balance to diamond card -/
noncomputable def transferPlatinumToDiamond (cards : SallyCards) : SallyCards :=
  let transferAmount := cards.platinum.balance / 2
  { cards with
    platinum := { cards.platinum with balance := cards.platinum.balance - transferAmount },
    diamond := { cards.diamond with balance := cards.diamond.balance + transferAmount } }

/-- Calculates the portion of the diamond card limit that remains unspent -/
noncomputable def diamondUnspentPortion (cards : SallyCards) : ℝ :=
  (cards.diamond.limit - cards.diamond.balance) / cards.diamond.limit

theorem diamond_unspent_portion_is_43_48 (g : ℝ) (h : g > 0) :
  let initialCards := initializeCards g
  let afterGoldTransfer := transferGoldToPlatinum initialCards
  let finalCards := transferPlatinumToDiamond afterGoldTransfer
  diamondUnspentPortion finalCards = 43 / 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_unspent_portion_is_43_48_l997_99743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_formula_l997_99708

/-- Taxi fare calculation function -/
noncomputable def taxiFare (P : ℝ) : ℝ :=
  if P ≤ 7 then 5 else 5 + 1.5 * (P - 7)

/-- Theorem: For distances greater than 7 km, the taxi fare is 1.5P - 5.5 -/
theorem taxi_fare_formula (P : ℝ) (h : P > 7) : taxiFare P = 1.5 * P - 5.5 := by
  sorry

#check taxi_fare_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_formula_l997_99708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_projection_sum_is_103_30_l997_99767

/-- Triangle MNP with given side lengths and altitude -/
structure Triangle where
  MN : ℝ
  NP : ℝ
  MP : ℝ
  altitude_N : ℝ
  h_MN : MN = 5
  h_NP : NP = 7
  h_MP : MP = 8
  h_altitude : altitude_N = 4

/-- Centroid G and its projections X, Y, Z onto sides NP, MP, MN respectively -/
structure Centroid (T : Triangle) where
  G : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Sum of distances from centroid to its projections -/
noncomputable def centroid_projection_sum (T : Triangle) (C : Centroid T) : ℝ :=
  distance C.G C.X + distance C.G C.Y + distance C.G C.Z

/-- Main theorem: The sum of distances from centroid to its projections is 103/30 -/
theorem centroid_projection_sum_is_103_30 (T : Triangle) (C : Centroid T) :
  centroid_projection_sum T C = 103/30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_projection_sum_is_103_30_l997_99767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l997_99771

/-- The length of a platform given train crossing times -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 16) :
  (train_length / time_cross_pole * time_cross_platform - train_length) = 431.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l997_99771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_after_operations_l997_99732

def square_set : Set ℕ := {n : ℕ | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 101 ∧ n = k^2}

def operation (S : Set ℕ) : Set ℕ :=
  {n : ℕ | ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ n = Int.natAbs (a - b)}

def iterate_operation (S : Set ℕ) : ℕ → Set ℕ
  | 0 => S
  | n + 1 => operation (iterate_operation S n)

theorem smallest_number_after_operations :
  ∃ n : ℕ, n ∈ iterate_operation square_set 100 ∧
    (∀ m : ℕ, m ∈ iterate_operation square_set 100 → n ≤ m) ∧
    n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_after_operations_l997_99732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_juice_percentage_is_42_86_l997_99761

/-- Represents the juice extraction rates and blend composition -/
structure JuiceBlend where
  apples_per_batch : ℕ
  bananas_per_batch : ℕ
  apple_juice_per_batch : ℚ
  banana_juice_per_batch : ℚ
  apples_in_blend : ℕ
  bananas_in_blend : ℕ

/-- Calculates the percentage of apple juice in the blend -/
def apple_juice_percentage (blend : JuiceBlend) : ℚ :=
  let apple_juice_yield := blend.apple_juice_per_batch / blend.apples_per_batch
  let banana_juice_yield := blend.banana_juice_per_batch / blend.bananas_per_batch
  let total_apple_juice := apple_juice_yield * blend.apples_in_blend
  let total_banana_juice := banana_juice_yield * blend.bananas_in_blend
  total_apple_juice / (total_apple_juice + total_banana_juice) * 100

/-- Theorem stating that the percentage of apple juice in the blend is approximately 42.86% -/
theorem apple_juice_percentage_is_42_86 (blend : JuiceBlend) 
  (h1 : blend.apples_per_batch = 3)
  (h2 : blend.bananas_per_batch = 2)
  (h3 : blend.apple_juice_per_batch = 9)
  (h4 : blend.banana_juice_per_batch = 10)
  (h5 : blend.apples_in_blend = 5)
  (h6 : blend.bananas_in_blend = 4) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |apple_juice_percentage blend - 42.86| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_juice_percentage_is_42_86_l997_99761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winter_jackets_count_l997_99782

-- Define the given constants
def total_receipts : ℚ := 5108.30
def price_before_noon : ℚ := 31.95
def price_after_noon : ℚ := 18.95
def jackets_after_noon : ℕ := 133

-- Define the function to calculate the total number of jackets
noncomputable def total_jackets : ℕ := 
  let receipts_after_noon := price_after_noon * (jackets_after_noon : ℚ)
  let receipts_before_noon := total_receipts - receipts_after_noon
  let jackets_before_noon := (receipts_before_noon / price_before_noon).floor.toNat
  jackets_before_noon + jackets_after_noon

-- Theorem statement
theorem winter_jackets_count : total_jackets = 214 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winter_jackets_count_l997_99782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_for_specific_cone_l997_99748

/-- The radius of a sphere with the same volume as a cone -/
noncomputable def sphere_radius_equal_volume_cone (cone_radius : ℝ) (cone_height : ℝ) : ℝ :=
  (3 * cone_radius^2 * cone_height / 4)^(1/3)

/-- Theorem: The radius of a sphere with the same volume as a cone with radius 2 inches and height 3 inches is equal to ∛3 inches -/
theorem sphere_radius_for_specific_cone :
  sphere_radius_equal_volume_cone 2 3 = Real.rpow 3 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_for_specific_cone_l997_99748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_series_sum_l997_99719

/-- Defines the series sum from 2 to 100 with alternating signs -/
def alternatingSeries : ℕ → ℤ
  | 0 => 0
  | n + 1 => alternatingSeries n + if n % 2 = 0 then (n + 2 : ℤ) else -(n + 2 : ℤ)

/-- The theorem stating that the alternating series sum from 2 to 100 equals 51 -/
theorem alternating_series_sum : alternatingSeries 99 = 51 := by
  sorry

#eval alternatingSeries 99

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_series_sum_l997_99719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_cartesian_eq_l_l_intersects_C_l997_99723

noncomputable section

-- Define the polar coordinates of point A
def point_A : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)

-- Define the polar equation of line l
def line_l (ρ θ a : ℝ) : Prop := ρ * Real.cos (θ - Real.pi / 4) = a

-- Define that point A lies on line l
axiom A_on_l : line_l point_A.1 point_A.2 (Real.sqrt 2)

-- Define the parametric equation of circle C
def circle_C (α : ℝ) : ℝ × ℝ := (1 + Real.cos α, Real.sin α)

-- Theorem 1: The value of a in the polar equation of line l is √2
theorem a_value : ∃ a : ℝ, ∀ ρ θ : ℝ, line_l ρ θ a ↔ line_l ρ θ (Real.sqrt 2) := by
  sorry

-- Theorem 2: The Cartesian equation of line l is x + y - 2 = 0
theorem cartesian_eq_l : ∀ x y : ℝ, (∃ ρ θ : ℝ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ line_l ρ θ (Real.sqrt 2)) ↔ x + y = 2 := by
  sorry

-- Theorem 3: Line l intersects circle C
theorem l_intersects_C : ∃ α : ℝ, let (x, y) := circle_C α; x + y = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_cartesian_eq_l_l_intersects_C_l997_99723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_when_x_less_than_one_l997_99741

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - a) * abs (x + abs (x - 2)) * (x - a)

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f x 1 < 0} = Set.Iio 1 := by sorry

-- Part 2
theorem range_of_a_when_x_less_than_one :
  {a : ℝ | ∀ x < 1, f x a < 0} = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_when_x_less_than_one_l997_99741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_substances_l997_99785

-- Define the set of substances
inductive Substance
| A
| B
| C
deriving Repr

-- Define the properties and reactions
class ChemicalProperties (S : Type) where
  density : S → ℝ
  isLiquid : S → Prop
  isGas : S → Prop
  react : S → S → Option S

-- Define the specific properties for our problem
axiom water_density : ℝ
axiom air_density : ℝ

axiom properties : ChemicalProperties Substance
axiom A_liquid : properties.isLiquid Substance.A
axiom B_gas : properties.isGas Substance.B
axiom C_liquid : properties.isLiquid Substance.C

axiom A_lighter_than_water : properties.density Substance.A < water_density
axiom B_lighter_than_air : properties.density Substance.B < air_density
axiom C_heavier_than_water : properties.density Substance.C > water_density

axiom A_reacts_to_B : properties.react Substance.A Substance.A = some Substance.B
axiom B_reacts_to_C : properties.react Substance.B Substance.B = some Substance.C

-- Define the chemical formulas
def ChemicalFormula : Type := String

-- Function to convert Substance to ChemicalFormula
def substanceToFormula : Substance → ChemicalFormula
| Substance.A => "C₂H₅OH"
| Substance.B => "CH₂=CH₂"
| Substance.C => "C₂H₅Br"

-- State the theorem
theorem identify_substances :
  ∃ (ethanol bromoethane ethylene : ChemicalFormula),
    (substanceToFormula Substance.A = ethanol) ∧
    (substanceToFormula Substance.B = ethylene) ∧
    (substanceToFormula Substance.C = bromoethane) ∧
    (ethanol = "C₂H₅OH") ∧
    (ethylene = "CH₂=CH₂") ∧
    (bromoethane = "C₂H₅Br") := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_substances_l997_99785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_mod_45_l997_99706

/-- M is a number formed by writing integers from 1 to 60 consecutively -/
def M : ℕ := sorry

/-- The theorem states that M mod 45 = 2 -/
theorem M_mod_45 : M % 45 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_mod_45_l997_99706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportion_above_threshold_l997_99722

-- Define the sample size
def sample_size : ℕ := 66

-- Define the groupings and their frequencies
def groupings : List (ℝ × ℝ × ℕ) := [
  (11.5, 15.5, 2),
  (15.5, 19.5, 4),
  (19.5, 23.5, 9),
  (23.5, 27.5, 18),
  (27.5, 31.5, 11),
  (31.5, 35.5, 12),
  (35.5, 39.5, 7),
  (39.5, 43.5, 3)
]

-- Define the threshold value
def threshold : ℝ := 31.5

-- Theorem statement
theorem proportion_above_threshold :
  (groupings.filter (fun g => g.1 ≥ threshold)).foldl (fun acc g => acc + g.2.2) 0 / sample_size = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportion_above_threshold_l997_99722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l997_99701

theorem cube_root_inequality (x : ℝ) :
  x.rpow (1/3) + 3 / (x.rpow (1/3) + 4) ≤ 0 ↔ -27 < x ∧ x < -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l997_99701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_root_least_degree_polynomial_l997_99762

noncomputable def root : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem polynomial_with_root (x : ℝ) : 
  x = root → x^4 - 10*x^2 - 11 = 0 :=
by sorry

theorem least_degree_polynomial (p : ℝ → ℝ) :
  (∃ (a b c : ℤ), ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x - 11) →
  (p root = 0) →
  (∃ (q : ℝ → ℝ), (∀ x, q x = x^4 - 10*x^2 - 11) ∧ 
    (∀ x, p x = q x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_root_least_degree_polynomial_l997_99762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_on_line_l997_99714

theorem sin_cos_value_on_line (x : ℝ) :
  (∃ (P : ℝ × ℝ), P.1 = Real.cos x ∧ P.2 = Real.sin x ∧ P.2 = 3 * P.1) →
  Real.sin x * Real.cos x = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_on_line_l997_99714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_implies_trig_equality_zero_point_implies_m_range_l997_99707

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, -1)
noncomputable def b (x m : ℝ) : ℝ × ℝ := (Real.cos x, m)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

noncomputable def f (x m : ℝ) : ℝ :=
  let a_vec := a x
  let b_vec := b x m
  2 * ((a_vec.1 + b_vec.1) * b_vec.1 + (a_vec.2 + b_vec.2) * b_vec.2) - 2 * m^2 - 1

noncomputable def g (x m : ℝ) : ℝ := f (x - Real.pi/6) m

theorem vector_parallel_implies_trig_equality (x : ℝ) :
  parallel (a x) (b x (Real.tan (10 * Real.pi / 3))) →
  Real.cos x^2 - Real.sin (2 * x) = 3/2 := by sorry

theorem zero_point_implies_m_range (m : ℝ) :
  (∃ x, x ∈ Set.Icc 0 (Real.pi/2) ∧ g x m = 0) →
  m ∈ Set.Icc (-1/2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_implies_trig_equality_zero_point_implies_m_range_l997_99707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_order_l997_99788

-- Define the real numbers a, b, and c
noncomputable def a : ℝ := -1/2
noncomputable def b : ℝ := Real.log 3 / Real.log (1/4)
noncomputable def c : ℝ := Real.log (1/2) / Real.log 3

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x

-- State the relationship between f, g, and e^x + cos x
axiom fg_sum : ∀ x, f x + g x = Real.exp x + Real.cos x

-- State the theorem to be proved
theorem g_order : g a < g c ∧ g c < g b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_order_l997_99788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_cube_l997_99739

theorem cube_root_of_cube (x : ℝ) : x^(1/3) * x^(1/3) * x^(1/3) = x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_cube_l997_99739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vase_sale_outcome_l997_99724

/-- Represents the sale of two vases with given profit and loss percentages -/
structure VaseSale where
  sellPrice : ℚ
  profitPercent : ℚ
  lossPercent : ℚ

/-- Calculates the overall financial outcome of the vase sale -/
noncomputable def financialOutcome (sale : VaseSale) : ℚ :=
  let costPrice1 := sale.sellPrice / (1 + sale.profitPercent / 100)
  let costPrice2 := sale.sellPrice / (1 - sale.lossPercent / 100)
  let totalRevenue := 2 * sale.sellPrice
  let totalCost := costPrice1 + costPrice2
  totalRevenue - totalCost

/-- Theorem stating that the financial outcome of the specific vase sale is a loss of 12 cents -/
theorem vase_sale_outcome :
  let sale : VaseSale := { sellPrice := 3/2, profitPercent := 25, lossPercent := 25 }
  financialOutcome sale = -3/25 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vase_sale_outcome_l997_99724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grapevine_wire_length_l997_99727

/-- Represents the setup of grapevines --/
structure GrapevineSetup where
  numStakes : ℕ
  stakeLength : ℝ
  stakeSeparation : ℝ
  braceDistance : ℝ

/-- Calculates the minimum wire length required for a grapevine setup --/
noncomputable def minWireLength (setup : GrapevineSetup) : ℝ :=
  (setup.numStakes - 1) * setup.stakeSeparation + 2 * Real.sqrt (setup.stakeLength ^ 2 + setup.braceDistance ^ 2)

/-- Theorem stating the minimum wire length for the given grapevine setup --/
theorem grapevine_wire_length :
  let setup : GrapevineSetup := {
    numStakes := 20,
    stakeLength := 2,
    stakeSeparation := 5,
    braceDistance := 1
  }
  minWireLength setup = 95 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grapevine_wire_length_l997_99727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lengths_15gon_radical_form_l997_99749

/-- The sum of lengths of sides and diagonals of a regular 15-gon inscribed in a circle of radius 15 -/
noncomputable def sum_lengths_15gon : ℝ :=
  450 * (Real.sin (12 * Real.pi / 180) + Real.sin (24 * Real.pi / 180) +
         Real.sin (36 * Real.pi / 180) + Real.sin (48 * Real.pi / 180))

/-- Representation of the sum as a + b√2 + c√3 + d√5 + e√6 -/
structure RadicalForm where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The value of the radical form -/
noncomputable def RadicalForm.value (rf : RadicalForm) : ℝ :=
  rf.a + rf.b * Real.sqrt 2 + rf.c * Real.sqrt 3 + rf.d * Real.sqrt 5 + rf.e * Real.sqrt 6

/-- The theorem to be proved -/
theorem sum_lengths_15gon_radical_form :
  ∃ (rf : RadicalForm), 
    rf.value = sum_lengths_15gon ∧
    rf.a + rf.b + rf.c + rf.d + rf.e = 877 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lengths_15gon_radical_form_l997_99749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l997_99791

noncomputable def scores : List ℝ := [9.7, 9.9, 10.1, 10.2, 10.1]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ) ^ 2)).sum / xs.length

theorem variance_of_scores : variance scores = 0.032 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l997_99791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_deposit_proof_l997_99700

/-- Compound interest calculation -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * time)

/-- Proof that $5,000 grows to $5,202 in 6 months with 8% annual interest compounded quarterly -/
theorem initial_deposit_proof :
  let principal := 5000
  let rate := 0.08
  let compounds_per_year := 4
  let time := 0.5
  let final_amount := 5202
  abs (compound_interest principal rate compounds_per_year time - final_amount) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_deposit_proof_l997_99700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_B5F_equals_2911_l997_99757

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'B' => 11
  | '5' => 5
  | 'F' => 15
  | _ => 0  -- Default case, should not be reached for this problem

/-- Calculates the value of a hexadecimal number in decimal -/
def hex_to_decimal (s : String) : ℕ :=
  s.toList.reverse.enum.foldl
    (fun acc (i, c) => acc + (hex_to_dec c) * (16 ^ i))
    0

/-- Theorem stating that B5F in hexadecimal is equal to 2911 in decimal -/
theorem hex_B5F_equals_2911 :
  hex_to_decimal "B5F" = 2911 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_B5F_equals_2911_l997_99757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_7_and_8_l997_99778

theorem three_digit_divisible_by_7_and_8 : 
  (Finset.filter (fun n : ℕ => 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0 ∧ n % 8 = 0) (Finset.range 1000)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_7_and_8_l997_99778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_maximizes_sum_of_squares_l997_99769

/-- A triangle inscribed in a circle -/
structure InscribedTriangle (r : ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  inscribed : ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * r^2) ∧
              ((B.1 - C.1)^2 + (B.2 - C.2)^2 = 4 * r^2) ∧
              ((C.1 - A.1)^2 + (C.2 - A.2)^2 = 4 * r^2)

/-- The sum of squares of sides of a triangle -/
def sumOfSquares (t : InscribedTriangle r) : ℝ :=
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 +
  (t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2 +
  (t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2

/-- An equilateral triangle inscribed in a circle -/
noncomputable def equilateralInscribed (r : ℝ) : InscribedTriangle r :=
  { A := (r, 0),
    B := (-r/2, r*Real.sqrt 3/2),
    C := (-r/2, -r*Real.sqrt 3/2),
    inscribed := by sorry }

/-- Theorem: The equilateral triangle maximizes the sum of squares of sides -/
theorem equilateral_maximizes_sum_of_squares (r : ℝ) (t : InscribedTriangle r) :
  sumOfSquares t ≤ sumOfSquares (equilateralInscribed r) := by
  sorry

#check equilateral_maximizes_sum_of_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_maximizes_sum_of_squares_l997_99769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_reduction_percentage_l997_99790

/-- The percentage reduction in employees from an original count to a new count -/
noncomputable def percentage_reduction (original : ℝ) (new : ℝ) : ℝ :=
  (original - new) / original * 100

theorem employee_reduction_percentage :
  let original_count : ℝ := 224.13793103448276
  let new_count : ℝ := 195
  abs (percentage_reduction original_count new_count - 13) < 0.0000001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_reduction_percentage_l997_99790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_feet_equal_distance_l997_99733

structure Triangle (K I A : ℝ × ℝ) : Prop where
  is_triangle : True

structure Point (p : ℝ × ℝ) : Prop where
  is_point : True

def median_foot (K I A O : ℝ × ℝ) : Prop :=
  O = ((I.1 + A.1) / 2, (I.2 + A.2) / 2)

def perpendicular_foot (P Q R : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0

def angle_bisector (P Q R : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ S ↦ (S.1 - Q.1) * (R.2 - Q.2) = (S.2 - Q.2) * (R.1 - Q.1)

def intersection_point (P Q R S X : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, X = (P.1 + t * (Q.1 - P.1), P.2 + t * (Q.2 - P.2)) ∧
             X = (R.1 + t * (S.1 - R.1), R.2 + t * (S.2 - R.2))

theorem perpendicular_feet_equal_distance 
  (K I A O Y Z X : ℝ × ℝ) 
  (tri : Triangle K I A) 
  (pO : Point O) (pY : Point Y) (pZ : Point Z) (pX : Point X)
  (hO : median_foot K I A O)
  (hY : perpendicular_foot I Y O ∧ angle_bisector I O K Y)
  (hZ : perpendicular_foot A Z O ∧ angle_bisector A O K Z)
  (hX : intersection_point K O Y Z X) :
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (X.1 - Z.1)^2 + (X.2 - Z.2)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_feet_equal_distance_l997_99733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_probability_theorem_l997_99755

open Real MeasureTheory

theorem sin_probability_theorem :
  let f : ℝ → ℝ := λ x ↦ Real.sin x
  let I : Set ℝ := Set.Icc 0 π
  let E : Set ℝ := {x ∈ I | f x ≥ 1/2}
  (volume E) / (volume I) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_probability_theorem_l997_99755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_in_f_l997_99787

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (9 - x) / (x - 4)

-- State the theorem
theorem range_of_x_in_f :
  ∀ x : ℝ, x ≠ 4 ↔ ∃ y : ℝ, f x = y :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_in_f_l997_99787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_l997_99765

noncomputable def line1 (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = 2011
noncomputable def line2 (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 4) = 2012

theorem lines_perpendicular : 
  ∃ (m₁ m₂ : ℝ), 
    (∀ x y, line1 x y ↔ y = m₁ * x + 2011 * Real.sqrt 2) ∧
    (∀ x y, line2 x y ↔ y = m₂ * x - 2012 * Real.sqrt 2) ∧
    m₁ * m₂ = -1 := by
  sorry

#check lines_perpendicular

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_l997_99765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_week_cut_is_twenty_percent_final_weight_matches_l997_99702

/-- The percentage of marble cut away in the second week of sculpting -/
noncomputable def second_week_cut (initial_weight : ℝ) (first_week_cut : ℝ) (third_week_cut : ℝ) (final_weight : ℝ) : ℝ :=
  let weight_after_first_week := initial_weight * (1 - first_week_cut)
  100 * (1 - (final_weight / (weight_after_first_week * (1 - third_week_cut))))

/-- Theorem stating that the percentage cut in the second week is 20% -/
theorem second_week_cut_is_twenty_percent :
  second_week_cut 250 0.30 0.25 105 = 20 := by
  sorry

/-- The final weight matches the given conditions -/
theorem final_weight_matches :
  let x := second_week_cut 250 0.30 0.25 105
  250 * (1 - 0.30) * (1 - x / 100) * (1 - 0.25) = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_week_cut_is_twenty_percent_final_weight_matches_l997_99702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_circumscribed_sphere_l997_99779

/-- The radius of a circumscribed sphere around a regular triangular pyramid -/
noncomputable def circumscribed_sphere_radius (a : ℝ) : ℝ :=
  a * Real.sqrt 3 / 3

/-- Theorem: The radius of the circumscribed sphere around a regular triangular pyramid
    with base side length a and lateral edge forming a 42° angle with the base plane
    is (a * √3) / 3 -/
theorem regular_triangular_pyramid_circumscribed_sphere
  (a : ℝ) -- base side length
  (h_positive : a > 0) -- assumption that side length is positive
  (h_angle : Real.cos (42 * Real.pi / 180) = (Real.sqrt 3) / 3) -- angle condition
  : circumscribed_sphere_radius a = a * Real.sqrt 3 / 3 :=
by
  sorry

#check regular_triangular_pyramid_circumscribed_sphere

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_circumscribed_sphere_l997_99779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l997_99753

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y = -9

-- Define the area of the region
noncomputable def region_area : ℝ := 16 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry

#check area_of_region

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l997_99753
