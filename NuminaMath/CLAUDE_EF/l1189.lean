import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_from_distance_sum_l1189_118950

/-- Given two fixed points A and B in a Euclidean space, the set of points P
    such that PA + PB = 2 * AB forms an ellipse -/
theorem ellipse_from_distance_sum {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] 
  (A B : E) : 
  let d := ‖A - B‖
  let S := {P : E | ‖P - A‖ + ‖P - B‖ = 2 * d}
  ∃ (center : E) (a b : ℝ), 
    S = {P : E | ‖P - center‖^2 / a^2 + ‖P - center‖^2 / b^2 = 1 ∧ 
                  a = d ∧ 
                  b = Real.sqrt 3 * d / 2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_from_distance_sum_l1189_118950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_complement_l1189_118938

/-- Represents an angle in degrees and minutes -/
structure Angle where
  degrees : ℕ
  minutes : ℕ

/-- The complement of an angle -/
def Angle.complement (a : Angle) : Angle :=
  { degrees := 90 - a.degrees - 1,
    minutes := 60 - a.minutes }

/-- Given that the complement of α is 125°12', prove that the complement of α is 35°12' -/
theorem complement_of_complement (α : Angle) 
  (h : α.complement = Angle.mk 125 12) : 
  α.complement.complement = Angle.mk 35 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_complement_l1189_118938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_points_distance_l1189_118919

-- Define a regular hexagon with side length 1
def RegularHexagon : Set (ℝ × ℝ) := sorry

-- Define a function that checks if a point is inside the hexagon
def InsideHexagon (p : ℝ × ℝ) : Prop := p ∈ RegularHexagon

-- Define a function to calculate the distance between two points
def Distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hexagon_points_distance 
  (points : Finset (ℝ × ℝ)) 
  (h1 : points.card = 7) 
  (h2 : ∀ p, p ∈ points → InsideHexagon p) : 
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ Distance p1 p2 ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_points_distance_l1189_118919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_of_angles_l1189_118961

theorem cos_sum_of_angles (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : (Real.sin (2*α)) / (Real.cos (2*α) - 1) = (1 - Real.tan β) / (1 + Real.tan β)) :
  Real.cos (α + β) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_of_angles_l1189_118961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1189_118968

-- Define the function g
def g (x : ℝ) : ℝ := -x^2 - 3

-- Define the properties of f
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_min_value_1_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-1) 2, f x ≥ 1 ∧ ∃ y ∈ Set.Icc (-1) 2, f y = 1

-- State the theorem
theorem quadratic_function_theorem (f : ℝ → ℝ) :
  is_quadratic f ∧
  is_odd (λ x ↦ f x + g x) ∧
  has_min_value_1_on_interval f →
  (∀ x, f x = x^2 + 3*x + 3) ∨ (∀ x, f x = x^2 - x + 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1189_118968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_times_prime_divisors_l1189_118998

theorem seven_times_prime_divisors (p : ℕ) (h_prime : Nat.Prime p) :
  let n := 7 * p
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_times_prime_divisors_l1189_118998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_resistance_correct_l1189_118992

/-- Represents the equivalent resistance of an n-stage network as described in Figure 2-16 -/
noncomputable def equivalent_resistance (k : ℕ) : ℝ :=
  (Real.sqrt 3 - 1 - (5 - 3 * Real.sqrt 3) * (7 - 4 * Real.sqrt 3) ^ (k - 1)) /
  (1 - (7 - 4 * Real.sqrt 3) ^ k)

/-- Theorem stating that the equivalent resistance of the n-stage network is correctly computed -/
theorem equivalent_resistance_correct (k : ℕ) :
  equivalent_resistance k = (Real.sqrt 3 - 1 - (5 - 3 * Real.sqrt 3) * (7 - 4 * Real.sqrt 3) ^ (k - 1)) /
                            (1 - (7 - 4 * Real.sqrt 3) ^ k) :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_resistance_correct_l1189_118992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_complex_fraction_l1189_118965

theorem simplify_complex_fraction :
  (1 : ℝ) / (1 / (Real.sqrt 5 + 2) + 3 / (Real.sqrt 7 - 2)) = (Real.sqrt 7 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_complex_fraction_l1189_118965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_condition_positivity_condition_l1189_118960

-- Define the function f(x)
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the interval [5, 20]
def interval : Set ℝ := Set.Icc 5 20

-- Part I: Monotonicity condition
theorem monotonicity_condition (k : ℝ) :
  (∀ x ∈ interval, ∀ y ∈ interval, x < y → f k x < f k y) ∨
  (∀ x ∈ interval, ∀ y ∈ interval, x < y → f k x > f k y) ↔
  k ∈ Set.Iic 40 ∪ Set.Ici 160 :=
sorry

-- Part II: Positivity condition
theorem positivity_condition (k : ℝ) :
  (∀ x ∈ interval, f k x > 0) ↔ k ∈ Set.Iio (92/5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_condition_positivity_condition_l1189_118960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1189_118928

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-8) 3

-- Define the function h in terms of f
def h (x : ℝ) : ℝ := f (-3 * x + 1)

-- Define the domain of h
def domain_h : Set ℝ := Set.Icc (-2/3) 3

-- Theorem statement
theorem domain_of_h : 
  ∀ x : ℝ, x ∈ domain_h ↔ (-3 * x + 1) ∈ domain_f :=
by
  intro x
  sorry

#check domain_of_h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1189_118928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_length_l1189_118987

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p q : Point) : ℝ := 
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

def on_circle (p : Point) (c : Circle) : Prop := 
  distance p ⟨c.center.1, c.center.2⟩ = c.radius

def on_segment (p q r : Point) : Prop := 
  distance p q + distance q r = distance p r

def collinear (p q r : Point) : Prop := 
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

noncomputable def intersect_circle (l1 l2 : Point) (c : Circle) : Point × Point := 
  sorry -- This is a complex calculation that we'll skip for now

theorem circle_intersection_length 
  (c : Circle) 
  (a b c' d p q x y : Point) 
  (hab : distance a b = 13)
  (hcd : distance c' d = 17)
  (hp_on_ab : on_segment a p b)
  (hq_on_cd : on_segment c' q d)
  (hap : distance a p = 7)
  (hcq : distance c' q = 5)
  (hpq : distance p q = 25)
  (ha_on_c : on_circle a c)
  (hb_on_c : on_circle b c)
  (hc_on_c : on_circle c' c)
  (hd_on_c : on_circle d c)
  (hxy_intersect : (x, y) = intersect_circle p q c)
  (hcollinear : collinear x p q ∧ collinear p q y)
  : distance x y = 693 / 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_length_l1189_118987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_lower_bound_l1189_118999

open BigOperators

variable (n : ℕ)
variable (O : ℝ × ℝ)
variable (A : Fin n → ℝ × ℝ)
variable (B : ℝ × ℝ)

def on_unit_circle (O : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - O.1)^2 + (p.2 - O.2)^2 = 1

theorem sum_distances_lower_bound
  (h1 : ∀ i, on_unit_circle O (A i))
  (h2 : (∑ i, (A i).1 - O.1) = 0 ∧ (∑ i, (A i).2 - O.2) = 0) :
  (∑ i, Real.sqrt ((B.1 - (A i).1)^2 + (B.2 - (A i).2)^2)) ≥ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_lower_bound_l1189_118999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l1189_118979

theorem puzzle_solution (a b c d e : ℕ) : 
  a ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  b ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  c ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  d ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  e ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  b ≤ d ∧
  c ≥ a ∧
  a ≤ e ∧
  b ≥ e ∧
  d ≠ 5 →
  a^b + c^d + e = 628 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l1189_118979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1189_118989

/-- A parabola with x-axis as axis of symmetry and origin as vertex -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 4 * a * x

/-- The focus of the parabola -/
def focus (p : Parabola) : ℝ × ℝ := (p.a, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: For a parabola with x-axis as symmetry axis and origin as vertex,
    if a point P(1, m) on the parabola is at distance 3 from the focus,
    then the equation of the parabola is y^2 = 8x -/
theorem parabola_equation (p : Parabola) (m : ℝ) 
    (h1 : p.equation 1 m)
    (h2 : distance (1, m) (focus p) = 3) :
    p.equation = fun x y => y^2 = 8 * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1189_118989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_distance_is_15_l1189_118920

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Calculate the x-intercept of a line. -/
noncomputable def xIntercept (l : Line) : ℝ :=
  let (x, y) := l.point
  x - y / l.slope

/-- Calculate the distance between two points on the x-axis. -/
def distanceOnXAxis (x1 x2 : ℝ) : ℝ :=
  |x1 - x2|

theorem x_intercept_distance_is_15 (line1 line2 : Line)
  (h1 : line1.slope = 2)
  (h2 : line2.slope = -4)
  (h3 : line1.point = (8, 20))
  (h4 : line2.point = (8, 20)) :
  distanceOnXAxis (xIntercept line1) (xIntercept line2) = 15 := by
  sorry

#check x_intercept_distance_is_15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_distance_is_15_l1189_118920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_bus_speeds_l1189_118926

/-- Represents the speeds and travel times of two school buses --/
structure SchoolBusTravel where
  distance_A : ℝ
  distance_B : ℝ
  speed_ratio : ℝ
  time_difference : ℝ

/-- Calculates the speed of School A's bus given the travel parameters --/
noncomputable def speed_A (travel : SchoolBusTravel) : ℝ :=
  travel.distance_A / (travel.distance_B / (travel.speed_ratio * travel.distance_A / travel.distance_B) + travel.time_difference)

/-- Theorem stating the speeds of the school buses --/
theorem school_bus_speeds (travel : SchoolBusTravel) 
  (h1 : travel.distance_A = 240)
  (h2 : travel.distance_B = 270)
  (h3 : travel.speed_ratio = 1.5)
  (h4 : travel.time_difference = 1) :
  speed_A travel = 60 ∧ speed_A travel * travel.speed_ratio = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_bus_speeds_l1189_118926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_equilateral_triangle_in_rectangle_l1189_118923

/-- The maximum area of an equilateral triangle inscribed in a rectangle --/
theorem max_area_equilateral_triangle_in_rectangle 
  (a b : ℝ) 
  (ha : a = 8) 
  (hb : b = 15) : 
  ∃ (s : ℝ), s > 0 ∧ s ≤ a ∧ s * (Real.sqrt 3 / 2) ≤ b ∧ 
  ∀ (t : ℝ), t > 0 → t ≤ a → t * (Real.sqrt 3 / 2) ≤ b → 
  (Real.sqrt 3 / 4) * s^2 ≥ (Real.sqrt 3 / 4) * t^2 ∧
  (Real.sqrt 3 / 4) * s^2 = 16 * Real.sqrt 3 := by
  sorry

#check max_area_equilateral_triangle_in_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_equilateral_triangle_in_rectangle_l1189_118923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l1189_118993

theorem circle_radius (m : ℝ) :
  ∃ (M N : ℝ × ℝ),
    (∀ (x y : ℝ), x^2 + y^2 - 2*x + m*y - 4 = 0 ↔ (x, y) ∈ ({M, N} : Set (ℝ × ℝ))) ∧
    (∃ (t : ℝ), M.1 + N.1 = 2*t ∧ M.2 + N.2 = -t) →
    ∃ (c : ℝ × ℝ), ∀ (p : ℝ × ℝ), 
      p.1^2 + p.2^2 - 2*p.1 + m*p.2 - 4 = 0 ↔ (p.1 - c.1)^2 + (p.2 - c.2)^2 = 3^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l1189_118993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_rate_is_sixty_l1189_118909

/-- Represents the assembly line production scenario -/
structure AssemblyLine where
  initial_rate : ℝ
  initial_order : ℝ
  second_order : ℝ
  average_output : ℝ

/-- Calculates the production rate after speed increase -/
noncomputable def production_rate_after_increase (a : AssemblyLine) : ℝ :=
  let initial_time := a.initial_order / a.initial_rate
  let total_time := (a.initial_order + a.second_order) / a.average_output
  let remaining_time := total_time - initial_time
  a.second_order / remaining_time

/-- Theorem stating that the production rate after speed increase is 60 cogs per hour -/
theorem production_rate_is_sixty (a : AssemblyLine) 
  (h1 : a.initial_rate = 15)
  (h2 : a.initial_order = 60)
  (h3 : a.second_order = 60)
  (h4 : a.average_output = 24) :
  production_rate_after_increase a = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_rate_is_sixty_l1189_118909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l1189_118944

noncomputable def f (x : ℝ) : ℝ := 3^(x^2 - 2*x)

theorem f_decreasing_on_interval :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → x₂ ≤ 1 → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l1189_118944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_7_4_l1189_118995

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => sequence_a n + 1 / ((n + 2) * (n + 1))

theorem a_4_equals_7_4 : sequence_a 3 = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_7_4_l1189_118995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_lists_eq_256_l1189_118917

/-- A list of the first 12 positive integers satisfying the given condition -/
def ValidList : Type :=
  {l : List ℕ // 
    l.length = 12 ∧ 
    l.toFinset = Finset.range 12 ∧
    ∀ i ∈ l.zip (List.range 12), 
      2 ≤ i.2 → 
      ∃ j ∈ l.zip (List.range 12), 
        j.2 < i.2 ∧ (j.1 = i.1 + 2 ∨ j.1 = i.1 - 2)}

/-- The number of valid lists -/
noncomputable def numValidLists : ℕ := sorry

/-- The main theorem stating that the number of valid lists is 256 -/
theorem num_valid_lists_eq_256 : numValidLists = 256 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_lists_eq_256_l1189_118917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l1189_118908

/-- Given a triangle ABC and a point D in its plane, 
    if BC = 3CD, then AD = -1/3 AB + 4/3 AC -/
theorem vector_relation (A B C D : EuclideanSpace ℝ (Fin 2)) :
  (C - B) = 3 • (D - C) →
  (D - A) = -(1/3) • (B - A) + (4/3) • (C - A) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l1189_118908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l1189_118936

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x + 1) / x

-- State the theorem
theorem symmetry_center_of_f :
  ∃ (c : ℝ × ℝ), c = (0, 1) ∧
  ∀ (x y : ℝ), x ≠ 0 → y = f x →
  ∃ (x' y' : ℝ), x' ≠ 0 ∧ y' = f x' ∧
  (x' - c.1 = c.1 - x) ∧ (y' - c.2 = c.2 - y) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l1189_118936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_spiral_strip_length_l1189_118902

/-- The length of a spiral strip on a cylindrical surface -/
noncomputable def spiral_strip_length (circumference height horizontal_shift : ℝ) : ℝ :=
  (((circumference + horizontal_shift)^2 + height^2) : ℝ).sqrt

/-- Theorem: The length of the specific spiral strip described in the problem -/
theorem specific_spiral_strip_length :
  spiral_strip_length 24 7 3 = (778 : ℝ).sqrt := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_spiral_strip_length_l1189_118902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_k_range_l1189_118996

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

-- State the theorem
theorem f_not_monotonic_k_range :
  ∀ k : ℝ, 
    (∃ x y : ℝ, k - 1 < x ∧ x < y ∧ y < k + 1 ∧ 
      ((f x < f y ∧ f y < f x) ∨ (f x > f y ∧ f y > f x))) →
    (1 ≤ k ∧ k < 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_k_range_l1189_118996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_l1189_118975

def primeSequence : List Nat := [7, 17, 37, 47, 67]

theorem sum_of_primes (h : ∀ n ∈ primeSequence, Nat.Prime n) :
  primeSequence.sum = 175 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_l1189_118975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l1189_118983

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  -2 * x^2 + 4 * y^2 - 12 * x - 20 * y + 5 = 0

-- Define a focus point
noncomputable def focus : ℝ × ℝ := (-3, 2.5 + 2 * Real.sqrt 3)

-- Theorem statement
theorem focus_of_hyperbola :
  let (x₀, y₀) := focus
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), hyperbola_eq x y ↔
      ((y - 2.5)^2 / a^2) - ((x + 3)^2 / b^2) = 1 ∧
      (y₀ - 2.5)^2 - (x₀ + 3)^2 = a^2 + b^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l1189_118983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_hot_dogs_is_3258_l1189_118922

-- Define the prices and quantities of hot dog packs
def price_8pack : ℚ := 155 / 100
def price_20pack : ℚ := 305 / 100
def price_250pack : ℚ := 2295 / 100
def budget : ℚ := 300

-- Define the function to calculate the maximum number of hot dogs
def max_hot_dogs : ℕ :=
  let num_250packs := (budget / price_250pack).floor.toNat
  let remaining_budget := budget - (num_250packs : ℚ) * price_250pack
  let num_8packs := (remaining_budget / price_8pack).floor.toNat
  num_250packs * 250 + num_8packs * 8

-- Theorem statement
theorem max_hot_dogs_is_3258 : max_hot_dogs = 3258 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_hot_dogs_is_3258_l1189_118922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_l1189_118954

/-- The length of rectangular place mats arranged on a circular table -/
noncomputable def mat_length (table_radius : ℝ) (num_mats : ℕ) : ℝ :=
  2 * table_radius * Real.sin (Real.pi / (2 * num_mats))

/-- Theorem stating the length of place mats on a circular table -/
theorem place_mat_length :
  let table_radius : ℝ := 6
  let num_mats : ℕ := 8
  mat_length table_radius num_mats = 6 * Real.sqrt (2 - Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_l1189_118954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_x_intercept_l1189_118977

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  x_intercept1 : Point

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The constant sum of distances from any point on the ellipse to the two foci -/
noncomputable def ellipseConstant (e : Ellipse) : ℝ :=
  distance e.x_intercept1 e.focus1 + distance e.x_intercept1 e.focus2

theorem ellipse_other_x_intercept (e : Ellipse) 
    (h1 : e.focus1 = ⟨0, -3⟩)
    (h2 : e.focus2 = ⟨4, 0⟩)
    (h3 : e.x_intercept1 = ⟨0, 0⟩) :
    ∃ (p : Point), p.y = 0 ∧ 
    distance p e.focus1 + distance p e.focus2 = ellipseConstant e ∧
    p = ⟨11/2, 0⟩ := by
  sorry

#check ellipse_other_x_intercept

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_x_intercept_l1189_118977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_steps_for_255_matches_l1189_118904

def matches_per_step (k : ℕ) : ℕ := 2 * k + 1

def total_matches (n : ℕ) : ℕ := Finset.sum (Finset.range n) (λ k => matches_per_step (k + 1))

theorem max_steps_for_255_matches :
  ∀ n : ℕ, total_matches n ≤ 255 → n ≤ 15 :=
by
  sorry

#eval total_matches 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_steps_for_255_matches_l1189_118904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_vectors_lambda_l1189_118930

/-- Three vectors are coplanar if and only if their scalar triple product is zero -/
def coplanar (a b c : ℝ × ℝ × ℝ) : Prop :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  let (c₁, c₂, c₃) := c
  a₁ * (b₂ * c₃ - b₃ * c₂) - a₂ * (b₁ * c₃ - b₃ * c₁) + a₃ * (b₁ * c₂ - b₂ * c₁) = 0

theorem coplanar_vectors_lambda (lambda : ℚ) : 
  let a : ℝ × ℝ × ℝ := (2, -1, 2)
  let b : ℝ × ℝ × ℝ := (-1, 3, -3)
  let c : ℝ × ℝ × ℝ := (13, 6, lambda)
  coplanar a b c → lambda = -57/17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_vectors_lambda_l1189_118930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hollow_cylinder_area_example_l1189_118962

/-- The total surface area of a hollow cylinder -/
noncomputable def hollow_cylinder_surface_area (h : ℝ) (r_outer : ℝ) (r_inner : ℝ) : ℝ :=
  2 * Real.pi * (r_outer^2 - r_inner^2) +  -- Area of two ends
  2 * Real.pi * r_outer * h +              -- External lateral surface area
  2 * Real.pi * r_inner * h                -- Internal lateral surface area

/-- Theorem: The total surface area of a hollow cylinder with height 12, outer radius 5, and inner radius 2 is 210π -/
theorem hollow_cylinder_area_example :
  hollow_cylinder_surface_area 12 5 2 = 210 * Real.pi := by
  -- Unfold the definition of hollow_cylinder_surface_area
  unfold hollow_cylinder_surface_area
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hollow_cylinder_area_example_l1189_118962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_integral_l1189_118978

/-- The volume of a sphere with radius R -/
noncomputable def sphere_volume (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

/-- Theorem: The volume of a sphere with radius R is equal to the integral of π(R² - x²) from -R to R -/
theorem sphere_volume_integral (R : ℝ) (h : R > 0) : 
  sphere_volume R = ∫ x in (-R)..R, Real.pi * (R^2 - x^2) := by
  sorry

#check sphere_volume_integral

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_integral_l1189_118978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_duration_is_five_hours_l1189_118973

/-- A journey with two parts -/
structure Journey where
  total_distance : ℝ
  speed_part1 : ℝ
  speed_part2 : ℝ
  duration_part1 : ℝ

/-- The total duration of a journey -/
noncomputable def total_duration (j : Journey) : ℝ :=
  j.duration_part1 + (j.total_distance - j.speed_part1 * j.duration_part1) / j.speed_part2

/-- Theorem stating that for the given journey, the total duration is 5 hours -/
theorem journey_duration_is_five_hours (j : Journey) 
  (h1 : j.total_distance = 240)
  (h2 : j.speed_part1 = 40)
  (h3 : j.speed_part2 = 60)
  (h4 : j.duration_part1 = 3) :
  total_duration j = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_duration_is_five_hours_l1189_118973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_non_prime_powers_l1189_118913

theorem consecutive_non_prime_powers (n : ℕ+) :
  ∃ N : ℕ, ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ¬∃ (p : ℕ) (m : ℕ), Nat.Prime p ∧ N + k = p ^ m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_non_prime_powers_l1189_118913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ratio_lines_l1189_118997

-- Define the parabola C
noncomputable def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle Ω
noncomputable def circle_Ω (x y : ℝ) : Prop := x^2 + y^2 + 4*x = 0

-- Define a line through the origin
noncomputable def line_through_origin (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define the perpendicularity of two lines
noncomputable def perpendicular (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -1

-- Define the area ratio function
noncomputable def area_ratio (k : ℝ) : ℝ := (1 + k^2)^2 / k^2

theorem min_area_ratio_lines :
  ∀ k : ℝ, k ≠ 0 →
  area_ratio k ≥ 4 ∧
  (area_ratio k = 4 ↔ k = 1 ∨ k = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ratio_lines_l1189_118997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_implies_a_equals_plus_minus_one_l1189_118948

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x - 1 else -x^2 - 2*x

-- Theorem statement
theorem f_equals_one_implies_a_equals_plus_minus_one (a : ℝ) :
  f a = 1 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_implies_a_equals_plus_minus_one_l1189_118948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_triangular_prism_l1189_118910

/-- Right prism with triangular bases -/
structure TriangularPrism where
  a : ℝ  -- Side length of base triangle
  b : ℝ  -- Side length of base triangle
  h : ℝ  -- Height of prism
  θ : ℝ  -- Angle between sides a and b

/-- The sum of areas of three mutually adjacent faces is 30 -/
noncomputable def adjacent_faces_area (p : TriangularPrism) : ℝ :=
  p.a * p.h + p.b * p.h + 1/2 * p.a * p.b * Real.sin p.θ

/-- Volume of the triangular prism -/
noncomputable def volume (p : TriangularPrism) : ℝ :=
  1/2 * p.a * p.b * p.h * Real.sin p.θ

/-- Theorem: The maximum volume of the prism is 10√5 -/
theorem max_volume_triangular_prism :
  ∀ p : TriangularPrism, adjacent_faces_area p = 30 →
  volume p ≤ 10 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_triangular_prism_l1189_118910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_points_theorem_prob_envelope_theorem_l1189_118982

-- Define the game parameters
def num_envelopes : ℕ := 13
def points_to_win : ℕ := 6

-- Define the probability of a team answering correctly
noncomputable def prob_correct : ℝ := 1 / 2

-- Define the function to calculate the expected points for one game
noncomputable def expected_points_one_game : ℝ := 
  6 * (1 - (Nat.choose 12 6 : ℝ) / 2^11)

-- Define the function to calculate the expected points for 100 games
noncomputable def expected_points_100_games : ℝ := 
  100 * expected_points_one_game

-- Define the function to calculate the probability of a specific envelope being played
noncomputable def prob_envelope_played : ℝ := 
  (num_envelopes - 1 : ℝ) / num_envelopes * (1 - (Nat.choose 12 6 : ℝ) / 2^12)

-- Theorem statements
theorem expected_points_theorem : 
  ∃ ε > 0, |expected_points_100_games - 465| < ε := sorry

theorem prob_envelope_theorem : 
  ∃ ε > 0, |prob_envelope_played - 0.715| < ε := sorry

-- We can't use #eval for noncomputable definitions, so we'll use #check instead
#check expected_points_100_games
#check prob_envelope_played

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_points_theorem_prob_envelope_theorem_l1189_118982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_container_radius_l1189_118980

open Real

noncomputable def container_volume (r : ℝ) : ℝ := π * r^2 + (2/3) * π * r^3

noncomputable def container_cost (r : ℝ) : ℝ := 
  30000 * (2 * π * r * (28 / (3 * r^2) - 2/3 * r) + π * r^2) + 
  40000 * (2 * π * r^2)

theorem optimal_container_radius :
  ∃ (r : ℝ), r > 0 ∧ 
    container_volume r = (28/3) * π ∧
    ∀ (s : ℝ), s > 0 ∧ container_volume s = (28/3) * π → 
      container_cost r ≤ container_cost s ∧
    r = (4 : ℝ) ^ (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_container_radius_l1189_118980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_samples_theorem_l1189_118918

-- Define the predicates as functions
def heavy : ℕ → Prop := sorry
def low_sulfur : ℕ → Prop := sorry

theorem oil_samples_theorem (total_samples : ℕ) 
  (h1 : total_samples % 9 = 0)
  (h2 : total_samples ≤ 200 ∧ ∀ n : ℕ, n % 9 = 0 → n ≤ 200 → total_samples ≥ n)
  (prob_heavy : ℚ) (prob_light_low : ℚ)
  (h3 : prob_heavy = 1 / 9)
  (h4 : prob_light_low = 11 / 18)
  (h5 : ∀ sample : ℕ, heavy sample → ¬low_sulfur sample) :
  ∃ high_sulfur_count : ℕ, high_sulfur_count = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_samples_theorem_l1189_118918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_difference_equals_expected_l1189_118971

/-- The diameter of each cylindrical pipe in centimeters -/
noncomputable def pipeDiameter : ℝ := 8

/-- The number of pipes in each crate -/
def numberOfPipes : ℕ := 160

/-- The height of Crate A with pipes stacked directly -/
noncomputable def heightCrateA : ℝ := pipeDiameter * (numberOfPipes / 10 : ℝ)

/-- The vertical distance between centers of pipes in consecutive rows in Crate B -/
noncomputable def verticalDistance : ℝ := pipeDiameter * (Real.sqrt 3 / 2)

/-- The height of Crate B with pipes in staggered formation -/
noncomputable def heightCrateB : ℝ := pipeDiameter + verticalDistance * ((numberOfPipes / 10 : ℝ) - 1)

/-- The positive difference in heights between Crate A and Crate B -/
noncomputable def heightDifference : ℝ := heightCrateA - heightCrateB

theorem height_difference_equals_expected : 
  heightDifference = 120 - 60 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_difference_equals_expected_l1189_118971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_second_quadrant_l1189_118970

theorem sin_double_angle_second_quadrant (α : Real) 
  (h1 : Real.sin α = 3/5) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.sin (2 * α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_second_quadrant_l1189_118970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1189_118914

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 - Real.cos x ^ 4

-- Define the interval
def interval : Set ℝ := {x | -Real.pi/12 ≤ x ∧ x ≤ Real.pi/3}

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ interval ∧ f x = -1 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1189_118914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_equals_sqrt14_l1189_118907

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line
def my_line (t x y : ℝ) : Prop := x = 2 - 1/2 * t ∧ y = -1 + 1/2 * t

-- Define the chord length
noncomputable def chord_length (r d : ℝ) : ℝ := 2 * Real.sqrt (r^2 - d^2)

theorem chord_length_equals_sqrt14 :
  ∃ (x y t : ℝ), my_circle x y ∧ my_line t x y ∧ chord_length 2 (1 / Real.sqrt 2) = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_equals_sqrt14_l1189_118907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_points_circle_theorem_l1189_118939

/-- A set of points in the plane is in convex position if they form the vertices of a convex polygon. -/
def ConvexPosition (S : Set (ℝ × ℝ)) : Prop := sorry

/-- No three points in the set are collinear. -/
def NoThreeCollinear (S : Set (ℝ × ℝ)) : Prop := sorry

/-- No four points in the set are concyclic. -/
def NoFourConcyclic (S : Set (ℝ × ℝ)) : Prop := sorry

/-- A circle contains a point in its interior. -/
def CircleContainsInterior (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop := sorry

/-- The number of points from a set that a circle contains in its interior. -/
noncomputable def PointsContainedInterior (S : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ) : ℕ := sorry

theorem convex_points_circle_theorem (S : Set (ℝ × ℝ)) 
  (h1 : Fintype S)
  (h2 : Fintype.card S = 2021)
  (h3 : ConvexPosition S)
  (h4 : NoThreeCollinear S)
  (h5 : NoFourConcyclic S) :
  ∃ (p q : ℝ × ℝ), p ∈ S ∧ q ∈ S ∧ 
    ∀ (center : ℝ × ℝ) (radius : ℝ),
      CircleContainsInterior center radius p → 
      CircleContainsInterior center radius q → 
      PointsContainedInterior S center radius ≥ 673 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_points_circle_theorem_l1189_118939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_size_for_rotation_l1189_118933

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ

/-- Represents the position of an object in a 2D space -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents a room with a table and a column -/
structure Room where
  table : Dimensions
  column : Dimensions
  column_position : Position

/-- Calculates the diagonal of a rectangular object -/
noncomputable def diagonal (d : Dimensions) : ℝ :=
  Real.sqrt (d.length ^ 2 + d.width ^ 2)

/-- Theorem: Minimum room size for table rotation -/
theorem min_room_size_for_rotation (r : Room) 
  (h1 : r.table = ⟨9, 12⟩) 
  (h2 : r.column = ⟨2, 2⟩) 
  (h3 : r.column_position = ⟨3, 3⟩) : 
  ∃ (s : ℝ), s = 17 ∧ s ≥ diagonal r.table + r.column.width / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_size_for_rotation_l1189_118933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_with_remainder_l1189_118959

theorem least_positive_integer_with_remainder (n : ℕ) : n = 842 ↔ 
  (n > 1) ∧ 
  (∀ d : ℕ, d ∈ ({5, 6, 7, 8, 10} : Set ℕ) → n % d = 2) ∧
  (∀ m : ℕ, m > 1 → (∀ d : ℕ, d ∈ ({5, 6, 7, 8, 10} : Set ℕ) → m % d = 2) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_with_remainder_l1189_118959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_swap_square_difference_l1189_118934

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 10) :: aux (m / 10)
  aux n

def swap_digits (n : ℕ) : Set ℕ :=
  {m | is_three_digit m ∧ digits m ≠ digits n ∧ (∀ i, List.get? (digits m) i ≠ List.get? (digits n) i)}

def is_two_digit_square (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ k, n = k * k

theorem three_digit_swap_square_difference (solution_set : Set ℕ) : 
  {A : ℕ | is_three_digit A ∧ 
           ∃ B ∈ swap_digits A, 
           is_two_digit_square (A - B)} = solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_swap_square_difference_l1189_118934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1189_118905

/-- Represents the speed of a train in km/h -/
abbrev TrainSpeed := ℝ

/-- Represents the time taken for a journey in hours -/
abbrev JourneyTime := ℝ

/-- Represents the distance between two cities in km -/
abbrev Distance := ℝ

/-- The original speed of the train -/
noncomputable def original_speed : TrainSpeed := 60

/-- The increased speed of the train -/
noncomputable def increased_speed : TrainSpeed := original_speed + 30

/-- The time taken at the original speed -/
def original_time : JourneyTime := 3

/-- The time taken at the increased speed -/
def increased_time : JourneyTime := 2

/-- The distance between the two cities -/
noncomputable def distance : Distance := original_speed * original_time

theorem train_speed_theorem : 
  distance = increased_speed * increased_time → original_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1189_118905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_assignment_l1189_118949

-- Define the type for digits (0-9)
def Digit := Fin 10

-- Define the type for the sequence of syllables
structure Syllable where
  first : Char
  second : Char

-- Define the sequence of syllables
def syllableSequence : List Syllable := [
  ⟨'Z', 'U'⟩, ⟨'Z', 'A'⟩, ⟨'N', 'A'⟩, ⟨'N', 'E'⟩, ⟨'L', 'A'⟩,
  ⟨'L', 'U'⟩, ⟨'C', 'I'⟩, ⟨'S', 'A'⟩, ⟨'M', 'U'⟩, ⟨'E', 'L'⟩
]

-- Define the assignment of digits to letters
def Assignment := Char → Digit

-- Define a predicate to check if an assignment is valid
def validAssignment (a : Assignment) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → a c1 ≠ a c2

-- Define a function to convert a syllable to a number
def syllableToNumber (a : Assignment) (s : Syllable) : Nat :=
  10 * (a s.first).val + (a s.second).val

-- Define a predicate to check if the sequence is in ascending order
def ascendingSequence (a : Assignment) : Prop :=
  ∀ i j, i < j → i < syllableSequence.length → j < syllableSequence.length →
    syllableToNumber a (syllableSequence.get ⟨i, by sorry⟩) < syllableToNumber a (syllableSequence.get ⟨j, by sorry⟩)

-- Theorem: There is no valid assignment that satisfies all conditions
theorem no_valid_assignment :
  ¬ ∃ (a : Assignment), validAssignment a ∧ ascendingSequence a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_assignment_l1189_118949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_minus_y_squared_l1189_118943

-- Define x and y as noncomputable
noncomputable def x : ℝ := 2 - Real.sqrt 3
noncomputable def y : ℝ := 2 + Real.sqrt 3

-- Theorem statement
theorem x_squared_minus_y_squared : x^2 - y^2 = -8 * Real.sqrt 3 := by
  -- Expand the definitions of x and y
  unfold x y
  -- Simplify the expression
  simp [pow_two]
  -- Algebraic manipulation
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_minus_y_squared_l1189_118943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_60_l1189_118940

/-- Represents the per capita cost function for the tour group -/
noncomputable def per_capita_cost (x : ℝ) : ℝ :=
  if x > 0 ∧ x ≤ 30 then 900
  else if x > 30 ∧ x ≤ 75 then 1200 - 10 * x
  else 0

/-- Represents the profit function for the travel agency -/
noncomputable def profit (x : ℝ) : ℝ :=
  if x > 0 ∧ x ≤ 30 then 900 * x - 15000
  else if x > 30 ∧ x ≤ 75 then x * (1200 - 10 * x) - 15000
  else 0

/-- Theorem stating that the profit function reaches its maximum at x = 60 -/
theorem max_profit_at_60 :
  ∀ x, x > 0 → x ≤ 75 → profit x ≤ profit 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_60_l1189_118940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_piece_volume_approx_l1189_118990

/-- The volume of a cylinder given its radius and height -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The cheese wheel is a cylinder with diameter 5 cm and thickness 1.5 cm -/
def cheeseWheel : ℝ × ℝ := (2.5, 1.5)

/-- The volume of the removed piece is one-third of the total volume -/
noncomputable def removedPieceVolume (wheel : ℝ × ℝ) : ℝ :=
  (1 / 3) * cylinderVolume wheel.1 wheel.2

/-- The removed piece volume is approximately 5.9 cubic centimeters -/
theorem removed_piece_volume_approx :
  abs (removedPieceVolume cheeseWheel - 5.9) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_piece_volume_approx_l1189_118990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_discs_distance_interval_l1189_118912

/-- Represents a circular disc -/
structure Disc where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Calculates the area of overlap between two discs -/
noncomputable def overlapArea (d1 d2 : Disc) : ℝ := sorry

/-- The main theorem -/
theorem overlapping_discs_distance_interval
  (d1 d2 : Disc)
  (h_radius : d1.radius = 2 ∧ d2.radius = 2)
  (h_overlap : overlapArea d1 d2 + 2 * Real.pi * d1.radius^2 = 6 * Real.pi) :
  161/100 ≤ distance d1.center d2.center ∧ distance d1.center d2.center ≤ 162/100 := by
  sorry

#check overlapping_discs_distance_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_discs_distance_interval_l1189_118912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_price_increase_l1189_118937

/-- Calculates the required additional percentage increase to reach a total increase of 1/3 after three consecutive price increases. -/
theorem gas_price_increase (initial_price : ℝ) (increase1 increase2 increase3 : ℝ) 
  (h1 : increase1 = 0.05) 
  (h2 : increase2 = 0.06) 
  (h3 : increase3 = 0.10) :
  ∃ (additional_increase : ℝ), 
    (initial_price * (1 + increase1) * (1 + increase2) * (1 + increase3) * (1 + additional_increase)) = 
    (initial_price * (4/3)) ∧ 
    (additional_increase > 0.0890 ∧ additional_increase < 0.0892) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_price_increase_l1189_118937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l1189_118963

/-- Represents the score distribution in a math contest -/
structure ScoreDistribution where
  score75 : ℝ
  score85 : ℝ
  score90 : ℝ
  score100 : ℝ
  score110 : ℝ
  sum_to_one : score75 + score85 + score90 + score100 + score110 = 1
  non_negative : score75 ≥ 0 ∧ score85 ≥ 0 ∧ score90 ≥ 0 ∧ score100 ≥ 0 ∧ score110 ≥ 0

/-- Calculates the mean score given a score distribution -/
def mean (d : ScoreDistribution) : ℝ :=
  75 * d.score75 + 85 * d.score85 + 90 * d.score90 + 100 * d.score100 + 110 * d.score110

/-- Determines the median score given a score distribution -/
noncomputable def median (d : ScoreDistribution) : ℝ :=
  if d.score75 + d.score85 > 0.5 then 85
  else if d.score75 + d.score85 + d.score90 > 0.5 then 90
  else if d.score75 + d.score85 + d.score90 + d.score100 > 0.5 then 100
  else 110

/-- The main theorem stating that the difference between mean and median is 3.5 -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score75 = 0.05)
  (h2 : d.score85 = 0.35)
  (h3 : d.score90 = 0.25)
  (h4 : d.score100 = 0.10) :
  |mean d - median d| = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l1189_118963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_square_characterization_l1189_118984

/-- The 2-adic valuation of a natural number -/
def v2 : ℕ → ℕ
  | 0 => 0
  | n + 1 => if (n + 1) % 2 = 0 then v2 ((n + 1) / 2) + 1 else 0

/-- Checks if there exist n consecutive positive integers whose sum is a perfect square -/
def exists_consecutive_sum_square (n : ℕ) : Prop :=
  ∃ k : ℕ, ∃ m : ℕ, n * k + n * (n + 1) / 2 = m^2

/-- Main theorem: characterizes positive integers n for which there exist n consecutive
    positive integers whose sum is a perfect square -/
theorem consecutive_sum_square_characterization (n : ℕ+) :
  exists_consecutive_sum_square n ↔ v2 n = 0 ∨ Odd (v2 n) := by
  sorry

#check consecutive_sum_square_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_square_characterization_l1189_118984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1189_118947

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2*x + 3) / Real.log 0.5

-- Define the domain of f(x)
def domain : Set ℝ := Set.Ioo (-1) 3

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x ∈ domain, ∀ y ∈ domain, x < y → x > 1 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1189_118947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_imply_sum_l1189_118969

noncomputable def f (a b : ℝ) (ha : a ≠ 0) : ℝ → ℝ := λ x ↦ x / (a * x + b)

def passes_through (a b : ℝ) (ha : a ≠ 0) : Prop :=
  f a b ha (-4) = 4

def symmetric_about_y_eq_neg_x (a b : ℝ) (ha : a ≠ 0) : Prop :=
  ∀ x y, f a b ha x = y → f a b ha (-y) = -x

theorem function_properties_imply_sum (a b : ℝ) (ha : a ≠ 0) :
  passes_through a b ha ∧ symmetric_about_y_eq_neg_x a b ha → a + b = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_imply_sum_l1189_118969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_op_nine_three_l1189_118921

-- Define the ⊙ operation
def circle_op : ℕ → ℕ → ℕ := sorry

-- State the given conditions
axiom cond1 : circle_op 2 4 = 8
axiom cond2 : circle_op 4 6 = 14
axiom cond3 : circle_op 5 3 = 13
axiom cond4 : circle_op 8 7 = 23

-- Theorem to prove
theorem circle_op_nine_three : circle_op 9 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_op_nine_three_l1189_118921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_hats_cost_l1189_118974

/-- Represents the size categories of baseball caps -/
inductive HatSize
  | Small
  | Medium
  | Large

/-- Determines the hat size based on head circumference -/
noncomputable def getHatSize (circumference : ℝ) : HatSize :=
  if circumference ≥ 8 ∧ circumference ≤ 15 then HatSize.Small
  else if circumference ≥ 16 ∧ circumference ≤ 22 then HatSize.Medium
  else HatSize.Large

/-- Returns the cost of a hat based on its size -/
def hatCost (size : HatSize) : ℕ :=
  match size with
  | HatSize.Small => 15
  | HatSize.Medium => 20
  | HatSize.Large => 25

/-- Calculates the total cost of hats for Jack and his friends -/
noncomputable def totalCost (jackCircumference : ℝ) : ℕ :=
  let charlieCircumference := (jackCircumference / 2) + 9
  let billCircumference := (2 / 3) * charlieCircumference
  let mayaCircumference := (jackCircumference + charlieCircumference) / 2
  let thomasCircumference := 2 * billCircumference - 3
  
  let jackCost := hatCost (getHatSize jackCircumference)
  let charlieCost := hatCost (getHatSize charlieCircumference)
  let billCost := hatCost (getHatSize billCircumference)
  let mayaCost := hatCost (getHatSize mayaCircumference)
  let thomasCost := hatCost (getHatSize thomasCircumference)
  
  jackCost + charlieCost + billCost + mayaCost + thomasCost

theorem custom_hats_cost (jackCircumference : ℝ) 
    (h : jackCircumference = 12) : totalCost jackCircumference = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_hats_cost_l1189_118974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_less_than_pi_half_l1189_118931

theorem angle_sum_less_than_pi_half 
  (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.cos α > Real.sin β) : 
  α + β < π/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_less_than_pi_half_l1189_118931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_gift_certificate_value_l1189_118994

/-- Calculates the gift certificate value per 100 meters for a race winner -/
theorem race_gift_certificate_value 
  (race_duration : ℕ) 
  (lap_distance : ℕ) 
  (winner_laps : ℕ) 
  (earnings_per_minute : ℚ) 
  (h1 : race_duration = 12)
  (h2 : lap_distance = 100)
  (h3 : winner_laps = 24)
  (h4 : earnings_per_minute = 7)
  : (race_duration * earnings_per_minute) / (winner_laps * lap_distance / 100) = 7/2 := by
  sorry

#check race_gift_certificate_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_gift_certificate_value_l1189_118994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_books_sold_l1189_118901

theorem library_books_sold (total : ℕ) (remaining_fraction : ℚ) (sold : ℕ) : 
  total = 9900 → 
  remaining_fraction = 4/6 → 
  sold = total - (remaining_fraction * ↑total).floor → 
  sold = 3300 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_books_sold_l1189_118901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_is_one_l1189_118952

-- Define the circle
def circle_eq (x y : ℤ) : Prop := x^2 + y^2 = 16

-- Define a point on the circle
structure PointOnCircle where
  x : ℤ
  y : ℤ
  on_circle : circle_eq x y

-- Define the distance between two points
noncomputable def distance (p q : PointOnCircle) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 : ℝ)

-- Define the irrationality of a distance
def is_irrational_distance (p q : PointOnCircle) : Prop :=
  Irrational (distance p q)

-- Theorem statement
theorem max_ratio_is_one
  (p q r s : PointOnCircle)
  (distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (pq_irrational : is_irrational_distance p q)
  (rs_irrational : is_irrational_distance r s) :
  ∃ (ratio : ℝ), ratio ≤ 1 ∧
    ∀ (p' q' r' s' : PointOnCircle),
      p' ≠ q' ∧ r' ≠ s' →
      is_irrational_distance p' q' →
      is_irrational_distance r' s' →
      distance p' q' / distance r' s' ≤ ratio := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_is_one_l1189_118952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_neq_y_l1189_118991

def x : ℕ → ℚ
  | 0 => 1/8  -- Add case for 0
  | 1 => 1/8
  | (n+1) => x n + (x n)^2

def y : ℕ → ℚ
  | 0 => 1/10  -- Add case for 0
  | 1 => 1/10
  | (n+1) => y n + (y n)^2

theorem x_neq_y : ∀ (m n : ℕ), m > 0 → n > 0 → x m ≠ y n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_neq_y_l1189_118991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_revenue_theorem_l1189_118900

/-- Represents the parking revenue calculation for a shopping mall --/
def ParkingRevenue (total_parking : ℕ) (two_wheel_fee : ℕ) (small_car_fee : ℕ) : Prop :=
  ∀ (x y : ℤ),
    (0 : ℤ) ≤ x ∧ x ≤ total_parking →
    y = two_wheel_fee * x + small_car_fee * (total_parking - x) →
    y = -4 * x + 10000

/-- The main theorem about the parking revenue --/
theorem parking_revenue_theorem :
  ParkingRevenue 2000 1 5 := by
  sorry

#check parking_revenue_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_revenue_theorem_l1189_118900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_two_equals_556_div_25_l1189_118929

-- Define the functions t and s
def t (x : ℝ) : ℝ := 5 * x - 14

noncomputable def s (x : ℝ) : ℝ := 
  let y := (x + 14) / 5  -- Inverse function of t
  y^2 + 5*y - 4

-- Theorem statement
theorem s_of_two_equals_556_div_25 : s 2 = 556 / 25 := by
  -- Unfold the definition of s
  unfold s
  -- Simplify the expression
  simp
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_two_equals_556_div_25_l1189_118929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_inverse_is_contrapositive_l1189_118925

-- Define a proposition type
variable (P : Prop)

-- Define logical operations
def negation (p : Prop → Prop) : Prop → Prop :=
  fun a => ¬(p a)

def inverse (p : Prop → Prop) : Prop → Prop :=
  fun a => p (¬a)

def contrapositive (p : Prop → Prop) : Prop → Prop :=
  fun a => p (¬a)

-- State the theorem
theorem negation_inverse_is_contrapositive (p : Prop → Prop) :
  let r := negation p
  let s := inverse r
  s = contrapositive p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_inverse_is_contrapositive_l1189_118925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_t_range_tight_l1189_118924

/-- Definition of a non-degenerate triangle with ordered side lengths -/
structure OrderedTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  a_le_b : a ≤ b
  b_le_c : b ≤ c
  triangle_ineq : a + b > c

/-- Definition of the function t -/
noncomputable def t (triangle : OrderedTriangle) : ℝ :=
  min (triangle.b / triangle.a) (triangle.c / triangle.b)

/-- The theorem stating the range of t -/
theorem t_range (triangle : OrderedTriangle) :
  1 ≤ t triangle ∧ t triangle < (1 + Real.sqrt 5) / 2 := by
  sorry

/-- The theorem stating that the interval is the smallest possible -/
theorem t_range_tight :
  ∀ x, 1 ≤ x ∧ x < (1 + Real.sqrt 5) / 2 →
  ∃ triangle : OrderedTriangle, t triangle = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_t_range_tight_l1189_118924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cbl_min_additional_people_cbl_even_teams_cbl_team_size_cbl_total_members_l1189_118915

/-- The minimum number of additional people needed for the CBL to start playing games -/
def min_additional_people (initial_members : ℕ) (team_size : ℕ) : ℕ :=
  let total_members := (initial_members + team_size - 1) / team_size * team_size
  total_members - initial_members

/-- Theorem stating the minimum number of additional people needed for the CBL to start playing games -/
theorem cbl_min_additional_people :
  min_additional_people 38 9 = 16 := by
  sorry

/-- Proof that the number of teams is even -/
theorem cbl_even_teams :
  (38 + min_additional_people 38 9) / 9 % 2 = 0 := by
  sorry

/-- Proof that each team has exactly 9 people -/
theorem cbl_team_size :
  (38 + min_additional_people 38 9) % 9 = 0 := by
  sorry

/-- Proof that the total number of people is at least 38 -/
theorem cbl_total_members :
  38 + min_additional_people 38 9 ≥ 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cbl_min_additional_people_cbl_even_teams_cbl_team_size_cbl_total_members_l1189_118915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1189_118967

def b (m : ℕ+) : ℕ := 2 * (Nat.sqrt m)

theorem sequence_property (e f g : ℤ) : 
  (∀ m : ℕ+, (b m : ℤ) = e * Int.floor (Real.sqrt (m + f)) + g) →
  e + f + g = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1189_118967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1189_118916

/-- The speed of a train in km/hr, given its length and time to pass a pole -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: The speed of a 160-meter train that takes 8 seconds to pass a pole is 72 km/hr -/
theorem train_speed_theorem :
  train_speed 160 8 = 72 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1189_118916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_about_pi_12_l1189_118964

noncomputable def f (x : ℝ) := 4 * Real.sin (2 * x + Real.pi / 3)

theorem f_symmetry_about_pi_12 :
  ∀ (x : ℝ), f (Real.pi / 6 - x) = f (Real.pi / 6 + x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_about_pi_12_l1189_118964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_condition_l1189_118955

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def b_seq (b k : ℝ) : ℕ → ℝ
  | 0 => b  -- Add case for 0
  | 1 => b
  | 2 => k * b
  | (n + 3) => b_seq b k (n + 1) * b_seq b k (n + 2)

theorem geometric_progression_condition (b k : ℝ) :
  (b > 0) →
  (k > 0) →
  geometric_sequence (b_seq b k) ↔ b = 1 ∧ k = 1 := by
  sorry

#check geometric_progression_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_condition_l1189_118955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_availability_l1189_118927

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

-- Define the people
inductive Person
| Anna
| Bill
| Carl

-- Define the availability function
def isAvailable (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.Anna, Day.Monday => false
  | Person.Anna, Day.Tuesday => true
  | Person.Anna, Day.Wednesday => false
  | Person.Anna, Day.Thursday => true
  | Person.Anna, Day.Friday => true
  | Person.Bill, Day.Monday => true
  | Person.Bill, Day.Tuesday => false
  | Person.Bill, Day.Wednesday => true
  | Person.Bill, Day.Thursday => false
  | Person.Bill, Day.Friday => false
  | Person.Carl, Day.Monday => false
  | Person.Carl, Day.Tuesday => false
  | Person.Carl, Day.Wednesday => true
  | Person.Carl, Day.Thursday => false
  | Person.Carl, Day.Friday => true

-- Define the number of available people for a given day
def availablePeople (d : Day) : Nat :=
  (List.filter (fun p => isAvailable p d) [Person.Anna, Person.Bill, Person.Carl]).length

-- Theorem statement
theorem min_availability :
  (availablePeople Day.Monday = 1) ∧
  (availablePeople Day.Tuesday = 1) ∧
  (availablePeople Day.Thursday = 1) ∧
  (∀ d : Day, availablePeople d ≥ 1) :=
by
  sorry

#eval availablePeople Day.Monday
#eval availablePeople Day.Tuesday
#eval availablePeople Day.Wednesday
#eval availablePeople Day.Thursday
#eval availablePeople Day.Friday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_availability_l1189_118927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_travel_theorem_l1189_118906

/-- The total distance traveled by an ant on three concentric circles -/
noncomputable def ant_travel_distance (r₁ r₂ r₃ : ℝ) : ℝ :=
  -- Arc of the largest circle (1/3 of circumference)
  (1/3) * 2 * Real.pi * r₃ +
  -- Radial distances
  (r₃ - r₂) + (r₂ - r₁) +
  -- Arc of the middle circle (1/3 of circumference)
  (1/3) * 2 * Real.pi * r₂ +
  -- Diameter of the smallest circle
  2 * r₁ +
  -- Half-circumference of the smallest circle
  (1/2) * 2 * Real.pi * r₁

/-- Theorem stating the total distance traveled by the ant -/
theorem ant_travel_theorem (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ = 5) (h₂ : r₂ = 10) (h₃ : r₃ = 15) :
  ant_travel_distance r₁ r₂ r₃ = (65 * Real.pi / 3) + 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_travel_theorem_l1189_118906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_discriminant_l1189_118966

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 3x^2 + (2 + 1/2)x + 1/2 -/
noncomputable def a : ℝ := 3
noncomputable def b : ℝ := 2 + 1/2
noncomputable def c : ℝ := 1/2

theorem quadratic_discriminant :
  discriminant a b c = 1/4 := by
  -- Unfold definitions and simplify
  unfold discriminant a b c
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_discriminant_l1189_118966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_implies_m_l1189_118945

-- Define the circle
noncomputable def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define the line
noncomputable def my_line (x y m : ℝ) : Prop := x - m*y + 1 = 0

-- Define the intersection of the line and circle
noncomputable def intersects (m : ℝ) : Prop :=
  ∃ (x y : ℝ), my_circle x y ∧ my_line x y m

-- Define the area of triangle ABC
noncomputable def triangle_area (m : ℝ) : ℝ := 8/5

-- Theorem statement
theorem intersection_area_implies_m (m : ℝ) :
  intersects m → triangle_area m = 8/5 →
  m = 2 ∨ m = -2 ∨ m = 1/2 ∨ m = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_implies_m_l1189_118945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_direct_proportion_l1189_118972

-- Define a function to represent the given function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := |m + 1| * x^(m^2)

-- Define what it means for a function to be a direct proportion
def is_direct_proportion (g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, g x = k * x

-- Theorem statement
theorem unique_m_for_direct_proportion :
  ∃! m : ℝ, is_direct_proportion (f m) ∧ m = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_direct_proportion_l1189_118972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_theorem_l1189_118958

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Magnitude of a vector -/
noncomputable def magnitude (v : Vector2D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2)

/-- Vector addition -/
def add (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Vector subtraction -/
def sub (v w : Vector2D) : Vector2D :=
  ⟨v.x - w.x, v.y - w.y⟩

/-- Scalar multiplication -/
def scale (k : ℝ) (v : Vector2D) : Vector2D :=
  ⟨k * v.x, k * v.y⟩

theorem vector_magnitude_theorem (a b : Vector2D) 
  (h1 : magnitude a = 1)
  (h2 : magnitude b = 2)
  (h3 : magnitude (add a b) = magnitude (sub a b)) :
  magnitude (sub (scale 2 a) b) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_theorem_l1189_118958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_l1189_118951

/-- A polynomial of degree 4 with coefficients a, b, c -/
def polynomial (a b c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + a*x + c

/-- Predicate to check if a polynomial has exactly three distinct roots of the form tan(y), tan(2y), tan(3y) -/
def has_three_distinct_tan_roots (a b c : ℝ) : Prop :=
  ∃ y : ℝ, (polynomial a b c (Real.tan y) = 0) ∧ 
            (polynomial a b c (Real.tan (2*y)) = 0) ∧ 
            (polynomial a b c (Real.tan (3*y)) = 0) ∧
            (Real.tan y ≠ Real.tan (2*y)) ∧ 
            (Real.tan y ≠ Real.tan (3*y)) ∧ 
            (Real.tan (2*y) ≠ Real.tan (3*y))

/-- The set of real triples (a, b, c) satisfying the conditions -/
def solution_set : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | has_three_distinct_tan_roots p.1 p.2.1 p.2.2}

/-- The main theorem stating that there are exactly 18 real triples (a, b, c) satisfying the conditions -/
theorem count_triples : 
  (∃ (s : Finset (ℝ × ℝ × ℝ)), s.card = 18 ∧ ∀ p, p ∈ s ↔ p ∈ solution_set) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_l1189_118951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1189_118953

-- Problem 1
theorem problem_1 : Real.sqrt 4 - (Real.sqrt 3 - 1)^(0 : ℕ) + 2^(-1 : ℤ) = 1.5 := by sorry

-- Problem 2
theorem problem_2 : Set.Ioo (-2 : ℝ) 5 = {x : ℝ | 1 - 2*x < 5 ∧ (x - 2) / 3 ≤ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1189_118953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_proof_l1189_118932

/-- The smallest positive angle whose terminal side passes through the point (√3, -1) -/
noncomputable def smallest_positive_angle : ℝ := 11 * Real.pi / 6

/-- The x-coordinate of point P -/
noncomputable def x : ℝ := Real.sqrt 3

/-- The y-coordinate of point P -/
def y : ℝ := -1

theorem smallest_angle_proof :
  ∀ α : ℝ,
  (α > 0) →
  (x * Real.cos α = Real.sqrt 3 ∧ x * Real.sin α = -1) →
  α ≥ smallest_positive_angle :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_proof_l1189_118932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l1189_118956

/-- Calculate the percentage of profit given the selling price and cost price -/
noncomputable def percentage_profit (selling_price cost_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

theorem profit_percentage_calculation :
  let selling_price : ℝ := 1200
  let cost_price : ℝ := 1000
  percentage_profit selling_price cost_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l1189_118956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_diagonal_intersection_probability_l1189_118986

/-- A regular decagon is a 10-sided polygon with all sides equal and all angles equal. -/
structure RegularDecagon where
  -- We don't need to define the structure fully, just declare it exists

/-- A diagonal of a regular decagon is a line segment that connects two non-adjacent vertices. -/
def diagonal (d : RegularDecagon) : Type := sorry

/-- The probability that two randomly chosen diagonals of a regular decagon intersect inside the decagon. -/
def intersection_probability (d : RegularDecagon) : ℚ := sorry

/-- Theorem stating that the probability of two randomly chosen diagonals 
    of a regular decagon intersecting inside the decagon is 42/119. -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) : 
  intersection_probability d = 42 / 119 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_diagonal_intersection_probability_l1189_118986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_circle_equation_l1189_118985

-- Define the ⊙ operation
noncomputable def circleOp (a b : ℝ) : ℝ := (Real.sqrt (3 * a + b))^3

-- State the theorem
theorem solve_circle_equation (x : ℝ) :
  circleOp 7 x = 125 → x = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_circle_equation_l1189_118985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1189_118988

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The set of numbers satisfying the given conditions -/
def valid_numbers : Set ℕ := {n | n < 500 ∧ n = 7 * sum_of_digits n ∧ n % 7 = 0}

/-- Predicate for valid numbers -/
def is_valid (n : ℕ) : Prop := n ∈ valid_numbers

/-- Decidability instance for the valid_numbers predicate -/
instance : DecidablePred is_valid :=
  fun n => decidable_of_iff (n < 500 ∧ n = 7 * sum_of_digits n ∧ n % 7 = 0) (by simp [is_valid, valid_numbers])

theorem count_valid_numbers : Finset.card (Finset.filter is_valid (Finset.range 500)) = 6 := by
  sorry

#eval Finset.card (Finset.filter is_valid (Finset.range 500))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1189_118988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_expression_range_l1189_118981

-- Define the expression as a function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) / x

-- State the theorem
theorem meaningful_expression_range :
  (∀ x : ℝ, x > 0 → x - 1 ≥ 0 → f x ∈ Set.range f) →
  (∀ x : ℝ, f x ∈ Set.range f → x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_expression_range_l1189_118981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_value_l1189_118903

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- State the theorem
theorem unique_a_value :
  ∃! a : ℝ, (A a ∩ B).Nonempty ∧ (A a ∩ C) = ∅ ∧ a = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_value_l1189_118903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheater_can_win_l1189_118942

/-- Represents a contestant in the trivia contest -/
structure Contestant where
  id : ℕ
  score : ℤ

/-- Represents the state of the trivia contest -/
structure TriviaContest where
  n : ℕ
  contestants : List Contestant
  cheater : Contestant

/-- The scoring function for the trivia contest -/
def score (correct : Bool) (isCheater : Bool) : ℤ :=
  if correct then 0
  else if isCheater then -1
  else -2

/-- Helper function to get the maximum score from a list of contestants -/
def maxScore (contestants : List Contestant) : ℤ :=
  contestants.foldl (fun acc c => max acc c.score) (-(2^62))

/-- Theorem stating that the cheater can win if leading by 2^(n-1) points -/
theorem cheater_can_win (contest : TriviaContest) 
  (h_n : contest.n > 1)
  (h_lead : contest.cheater.score ≥ 2^(contest.n - 1) + maxScore contest.contestants) :
  ∃ (finalContest : TriviaContest), 
    finalContest.cheater.score > maxScore finalContest.contestants :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheater_can_win_l1189_118942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1189_118911

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if there exist three distinct points P, A, and B on the hyperbola such that
    A and B are symmetric about the origin, and the product of slopes of PA and PB is 3/4,
    then the eccentricity of the hyperbola is √7/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (P A B : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  P ≠ A →
  P ≠ B →
  A ≠ B →
  (P.1^2 / a^2 - P.2^2 / b^2 = 1) →
  (A.1^2 / a^2 - A.2^2 / b^2 = 1) →
  (B.1^2 / a^2 - B.2^2 / b^2 = 1) →
  B = (-A.1, -A.2) →
  ((P.2 - A.2) / (P.1 - A.1)) * ((P.2 - B.2) / (P.1 - B.1)) = 3/4 →
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1189_118911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l1189_118935

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 0 → x + 1/2 > 2

-- Define proposition q
def q : Prop := ∃ x : ℝ, (2 : ℝ)^x < 0

-- Theorem to prove
theorem correct_proposition : ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l1189_118935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_parallel_opposite_l1189_118957

/-- Two vectors in ℝ² -/
noncomputable def a : ℝ × ℝ := (-5, 3/5)
noncomputable def b : ℝ × ℝ := (10, -6/5)

/-- Definition of parallel vectors in opposite directions -/
def parallel_opposite (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ v = (k * w.1, k * w.2)

/-- Theorem: Vectors a and b are parallel and in opposite directions -/
theorem a_b_parallel_opposite : parallel_opposite a b := by
  use -1/2
  constructor
  · exact show -1/2 < 0 from by norm_num
  · ext
    · simp [a, b]
      norm_num
    · simp [a, b]
      norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_parallel_opposite_l1189_118957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l1189_118976

/-- Approximation of the given expression -/
noncomputable def approximation : ℝ := -52.25

/-- The expression to be evaluated -/
noncomputable def expression (x y z w : ℝ) : ℝ :=
  (x^2 - 69.28 * 0.004 * y) / ((0.03 * z) + Real.sin w)

/-- Theorem stating that the expression with given values is approximately equal to the approximation -/
theorem expression_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |expression 3.5 14.7 6.8 (Real.pi / 4) - approximation| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l1189_118976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1189_118946

-- Define the parabola
noncomputable def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define the focus of the parabola
noncomputable def focus : ℝ × ℝ := (1/2, 0)

-- Define the directrix
noncomputable def directrix (x : ℝ) : Prop := x = -1/2

-- Theorem statement
theorem parabola_focus_distance 
  (P : ℝ × ℝ) -- Point on the parabola
  (Q : ℝ × ℝ) -- Point on the directrix
  (h1 : parabola P.1 P.2) -- P is on the parabola
  (h2 : directrix Q.1) -- Q is on the directrix
  (h3 : P.2 = Q.2) -- PQ is parallel to x-axis
  (h4 : dist P Q = dist Q focus) -- |PQ| = |QF|
  : dist P focus = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1189_118946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1189_118941

/-- A rectangle in the coordinate plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- A line in the coordinate plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

/-- The area of a rectangle -/
noncomputable def rectangle_area (r : Rectangle) : ℝ :=
  r.width * r.height

/-- A line divides a rectangle into two equal areas -/
def divides_equally (r : Rectangle) (l : Line) (d : ℝ) : Prop :=
  triangle_area d r.height = (1 / 2) * rectangle_area r

theorem equal_area_division (r : Rectangle) (l : Line) (d : ℝ) :
  r.width = 2 ∧ r.height = 3 →
  l.slope = 3 / d ∧ l.y_intercept = 0 →
  divides_equally r l d ↔ d = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1189_118941
