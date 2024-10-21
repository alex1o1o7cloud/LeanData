import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_belty_position_l931_93118

def letters : List Char := ['B', 'E', 'L', 'T', 'Y']

def is_valid_word (w : List Char) : Bool :=
  w.length = 5 && w.toFinset ⊆ letters.toFinset

def all_words : List (List Char) :=
  letters.permutations.filter is_valid_word

def belty : List Char := ['B', 'E', 'L', 'T', 'Y']

theorem belty_position :
  all_words.indexOf belty = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_belty_position_l931_93118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_l931_93189

theorem sin_difference (α β : ℝ) 
  (h1 : Real.sin α - Real.cos β = 3/4) 
  (h2 : Real.cos α + Real.sin β = -2/5) : 
  Real.sin (α - β) = 511/800 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_l931_93189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_problem_fraction_sum_zero_matrix_problem_conclusion_l931_93128

theorem matrix_problem (x y z : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![x, 2*y, 2*z], ![2*y, z, x], ![2*z, x, y]]
  ¬(IsUnit (Matrix.det M)) →
  x^2 + y^2 + z^2 = 0 →
  x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro M h1 h2
  sorry

theorem fraction_sum_zero (x y z : ℝ) :
  x = 0 ∧ y = 0 ∧ z = 0 →
  x / (2*y + 2*z) + y / (x + z) + z / (x + y) = 0 :=
by
  intro h
  sorry

theorem matrix_problem_conclusion (x y z : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![x, 2*y, 2*z], ![2*y, z, x], ![2*z, x, y]]
  ¬(IsUnit (Matrix.det M)) →
  x^2 + y^2 + z^2 = 0 →
  x / (2*y + 2*z) + y / (x + z) + z / (x + y) = 0 :=
by
  intro M h1 h2
  have h3 : x = 0 ∧ y = 0 ∧ z = 0 := matrix_problem x y z h1 h2
  exact fraction_sum_zero x y z h3


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_problem_fraction_sum_zero_matrix_problem_conclusion_l931_93128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_divisors_420_eq_192_l931_93127

/-- The sum of all odd divisors of 420 -/
def sum_odd_divisors_420 : ℕ :=
  (Finset.filter (fun d => d % 2 = 1) (Nat.divisors 420)).sum id

/-- Theorem: The sum of all odd divisors of 420 is 192 -/
theorem sum_odd_divisors_420_eq_192 : sum_odd_divisors_420 = 192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_divisors_420_eq_192_l931_93127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_objects_to_containers_l931_93144

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers. -/
def number_of_ways_to_distribute (n k : ℕ) : ℕ := k^n

theorem distribute_objects_to_containers (n k : ℕ) : 
  (n = 5 ∧ k = 3) → (number_of_ways_to_distribute n k = 3^5) :=
by
  intro h
  simp [number_of_ways_to_distribute]
  rw [h.1, h.2]
  rfl

#check distribute_objects_to_containers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_objects_to_containers_l931_93144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l931_93123

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + 1 / (x - 1)

-- State the theorem
theorem f_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : 1/2 ≤ a ∧ a < 2) 
  (hx₁ : 0 < x₁ ∧ x₁ < 1/2) 
  (hx₂ : 2 < x₂) : 
  f a x₂ - f a x₁ ≥ log 2 + 3/4 := by
  sorry

#check f_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l931_93123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_factorial_power_minus_one_l931_93109

theorem divisibility_of_factorial_power_minus_one (a n : ℕ) 
  (h : n > 0) (h2 : Nat.gcd a (Nat.factorial n) = 1) : 
  Nat.factorial n ∣ (a ^ Nat.factorial n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_factorial_power_minus_one_l931_93109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patio_herb_yield_l931_93116

-- Define the constants
def steps : ℕ := 25
def feet_per_step : ℝ := 1.5
def pounds_per_sqft : ℝ := 0.75

-- Define the theorem
theorem patio_herb_yield :
  let side_length : ℝ := (steps : ℝ) * feet_per_step
  let area : ℝ := side_length ^ 2
  let total_yield : ℝ := area * pounds_per_sqft
  Int.floor total_yield = 1055 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_patio_herb_yield_l931_93116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_asymptotes_eccentricity_l931_93159

-- Define a hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1

-- Define the property of perpendicular asymptotes
def has_perpendicular_asymptotes (h : Hyperbola) : Prop :=
  h.a = h.b

-- Define eccentricity
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

-- Theorem statement
theorem perpendicular_asymptotes_eccentricity (h : Hyperbola) 
  (perp : has_perpendicular_asymptotes h) : 
  eccentricity h = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_asymptotes_eccentricity_l931_93159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approx_l931_93166

/-- The area of a circular sector with given radius and central angle -/
noncomputable def sectorArea (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  (centralAngle / 360) * Real.pi * radius^2

theorem sector_area_approx :
  let radius : ℝ := 12
  let centralAngle : ℝ := 41
  abs (sectorArea radius centralAngle - 51.57) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approx_l931_93166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_current_l931_93129

/-- A function representing the relationship between current and resistance for a 48V battery. -/
noncomputable def current (resistance : ℝ) : ℝ := 48 / resistance

/-- Theorem stating that for a 48V battery with 12Ω resistance, the current is 4A. -/
theorem battery_current : current 12 = 4 := by
  -- Unfold the definition of current
  unfold current
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_current_l931_93129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l931_93141

-- Define the sets A and B
def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {x | Real.rpow 3 x > 1}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Bᶜ) = {x : ℝ | -2 < x ∧ x ≤ 0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l931_93141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_is_simplest_l931_93160

-- Define a function to check if a square root is in its simplest form
def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y z : ℚ, x = (y : ℝ) * Real.sqrt (z : ℝ) → y = 1 ∧ z = x^2

-- Define the given square roots
noncomputable def sqrt_2 : ℝ := Real.sqrt 2
noncomputable def sqrt_12 : ℝ := Real.sqrt 12
noncomputable def sqrt_half : ℝ := Real.sqrt (1/2)
noncomputable def sqrt_1_5 : ℝ := Real.sqrt 1.5

-- Theorem statement
theorem sqrt_2_is_simplest :
  is_simplest_sqrt sqrt_2 ∧
  ¬ is_simplest_sqrt sqrt_12 ∧
  ¬ is_simplest_sqrt sqrt_half ∧
  ¬ is_simplest_sqrt sqrt_1_5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_is_simplest_l931_93160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_coloring_l931_93147

-- Define the isosceles right triangle
structure IsoscelesRightTriangle where
  side_length : ℝ
  is_unit : side_length = 1

-- Define a color type
inductive Color
  | Red
  | Blue
  | Green
  | Yellow

-- Define a point in the triangle
structure Point where
  x : ℝ
  y : ℝ
  in_triangle : x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 1

-- Define a colored point
structure ColoredPoint where
  point : Point
  color : Color

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- The main theorem
theorem isosceles_right_triangle_coloring 
  (triangle : IsoscelesRightTriangle) 
  (coloring : Point → Color) :
  ∃ (p1 p2 : Point), 
    coloring p1 = coloring p2 ∧ 
    distance p1 p2 ≥ 2 - Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_coloring_l931_93147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pedal_circles_common_point_l931_93168

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Function to check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- Function to generate the pedal circle of a point with respect to a triangle -/
noncomputable def pedal_circle (p a b c : Point) : Circle :=
  sorry -- Definition of pedal circle

/-- Define membership for a point in a circle -/
instance : Membership Point Circle where
  mem p c := (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Theorem stating that the four pedal circles have a common point -/
theorem pedal_circles_common_point (a b c d : Point) :
  ¬collinear a b c → ¬collinear a b d → ¬collinear a c d → ¬collinear b c d →
  ∃ (p : Point),
    p ∈ pedal_circle d a b c ∧
    p ∈ pedal_circle c a b d ∧
    p ∈ pedal_circle b a c d ∧
    p ∈ pedal_circle a b c d :=
by
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pedal_circles_common_point_l931_93168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l931_93148

/-- Calculates simple interest given principal, rate, and time -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest given principal, rate, and time -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

theorem compound_interest_problem (P : ℝ) 
    (h1 : simpleInterest P 5 2 = 600)
    (h2 : P > 0) :
    compoundInterest P 5 2 = 615 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l931_93148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l931_93164

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  focal_sum : ℝ
  eccentricity : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_focal_sum : focal_sum = 4 * Real.sqrt 2
  h_eccentricity : eccentricity = Real.sqrt 3 / 2
  h_equation : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

/-- Properties of the ellipse -/
def ellipse_properties (e : Ellipse) : Prop :=
  -- Standard equation
  (∀ x y : ℝ, x^2 / 8 + y^2 / 2 = 1) ∧
  -- Length of major axis
  (2 * e.a = 4 * Real.sqrt 2) ∧
  -- Coordinates of foci
  (∃ c : ℝ, c = Real.sqrt 6 ∧
    ((-c, 0) ∈ Set.range (λ p : ℝ × ℝ ↦ p)) ∧
    ((c, 0) ∈ Set.range (λ p : ℝ × ℝ ↦ p))) ∧
  -- Equation of directrix
  (∃ k : ℝ, k = 4 * Real.sqrt 6 / 3 ∧
    (∀ x : ℝ, x = k ∨ x = -k))

/-- Main theorem: The ellipse has the specified properties -/
theorem ellipse_theorem (e : Ellipse) : ellipse_properties e := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l931_93164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_bounds_l931_93102

theorem log_sum_bounds (k : ℕ+) :
  k < Real.log 3 / Real.log 2 + Real.log 4 / Real.log 3 ∧
  Real.log 3 / Real.log 2 + Real.log 4 / Real.log 3 < k + 1 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_bounds_l931_93102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_seating_equation_l931_93135

def number_of_people (x : ℕ) : ℕ := 45 * x + 28

theorem bus_seating_equation (x : ℕ) (h : x > 0) : 
  (45 * x + 28 = 50 * (x - 1) - 12) ↔ 
  (45 * x + 28 = number_of_people x ∧ 
   50 * x - 50 = number_of_people x ∧
   number_of_people x > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_seating_equation_l931_93135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_theorem_l931_93190

noncomputable def marguerite_distance : ℝ := 180
noncomputable def marguerite_time : ℝ := 3.6
noncomputable def sam_total_time : ℝ := 4.5
noncomputable def sam_pit_stop : ℝ := 0.5

noncomputable def average_speed : ℝ := marguerite_distance / marguerite_time

noncomputable def sam_driving_time : ℝ := sam_total_time - sam_pit_stop

theorem sam_distance_theorem : average_speed * sam_driving_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_theorem_l931_93190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_sum_diff_vectors_l931_93191

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle_between (u v : V) : ℝ :=
  Real.arccos ((inner u v) / (norm u * norm v))

theorem angle_between_sum_diff_vectors (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : norm (a + b) = norm (a - b) ∧ norm (a + b) = 2 * norm a) : 
  angle_between (a + b) (a - b) = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_sum_diff_vectors_l931_93191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_equals_8_root_390_div_7_l931_93145

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def are_externally_tangent (c1 c2 : Circle) : Prop := sorry

def is_internally_tangent (c1 c2 : Circle) : Prop := sorry

def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

def is_common_external_tangent (l : Set (ℝ × ℝ)) (c1 c2 : Circle) : Prop := sorry

def is_chord (l : Set (ℝ × ℝ)) (c : Circle) : Prop := sorry

theorem chord_length_equals_8_root_390_div_7 
  (c1 c2 c3 : Circle)
  (h1 : are_externally_tangent c1 c2)
  (h2 : is_internally_tangent c1 c3)
  (h3 : is_internally_tangent c2 c3)
  (h4 : c1.radius = 4)
  (h5 : c2.radius = 10)
  (h6 : are_collinear c1.center c2.center c3.center)
  (l : Set (ℝ × ℝ))
  (h7 : is_common_external_tangent l c1 c2)
  (h8 : is_chord l c3) :
  ∃ (length : ℝ), length = 8 * Real.sqrt 390 / 7 ∧ 
    ∀ (p1 p2 : ℝ × ℝ), p1 ∈ l ∧ p2 ∈ l → dist p1 p2 = length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_equals_8_root_390_div_7_l931_93145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_playhouse_siding_cost_l931_93126

/-- Calculates the cost of siding for Sandy's daughter's playhouse -/
def playhouse_siding_cost 
  (wall_width : ℕ) (wall_height : ℕ) (roof_base : ℕ) (roof_height : ℕ)
  (siding_width : ℕ) (siding_height : ℕ) (siding_cost : ℕ) : ℕ :=
  let wall_area := 2 * (wall_width * wall_height)
  let roof_area := (roof_base * roof_height) / 2
  let total_area := wall_area + roof_area
  let siding_area := siding_width * siding_height
  let sections_needed := (total_area + siding_area - 1) / siding_area
  sections_needed * siding_cost

/-- The cost of siding for Sandy's daughter's playhouse is $70 -/
theorem sandy_playhouse_siding_cost : 
  playhouse_siding_cost 10 7 10 6 10 15 35 = 70 := by
  sorry

#eval playhouse_siding_cost 10 7 10 6 10 15 35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_playhouse_siding_cost_l931_93126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AC_equals_three_l931_93187

/-- Given points A and B in 3D space, and C symmetric to B with respect to the y-axis,
    prove that the distance between A and C is 3. -/
theorem distance_AC_equals_three (A B : ℝ × ℝ × ℝ) (h1 : A = (-1, 2, 0)) (h2 : B = (-1, 1, 2)) :
  let C : ℝ × ℝ × ℝ := (1, B.2.1, -B.2.2)
  Real.sqrt ((C.1 - A.1)^2 + (C.2.1 - A.2.1)^2 + (C.2.2 - A.2.2)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AC_equals_three_l931_93187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_numbering_l931_93115

/-- A polygon with 2n+1 vertices -/
structure Polygon (n : ℕ) where
  vertices : Fin (2*n+1) → ℕ
  midpoints : Fin (2*n+1) → ℕ

/-- The assignment of numbers is valid -/
def valid_assignment (n : ℕ) (p : Polygon n) : Prop :=
  (∀ i, p.vertices i ∈ Finset.range (4*n+3)) ∧
  (∀ i, p.midpoints i ∈ Finset.range (4*n+3)) ∧
  (Finset.card (Finset.image p.vertices (Finset.univ) ∪ 
               Finset.image p.midpoints (Finset.univ)) = 4*n+2)

/-- The sum of numbers on each side is constant -/
def constant_sum (n : ℕ) (p : Polygon n) : Prop :=
  ∃ s, ∀ i : Fin (2*n+1), 
    p.vertices i + p.vertices (i.succ) + p.midpoints i = s

theorem polygon_numbering (n : ℕ) :
  ∃ (p : Polygon n), valid_assignment n p ∧ constant_sum n p ∧
    (∀ i : Fin (2*n+1), 
      p.vertices i + p.vertices (i.succ) + p.midpoints i = 5*n+4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_numbering_l931_93115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_property_l931_93119

/-- Auxiliary function to calculate the sum of digits -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Theorem stating the property of digit sums -/
theorem digit_sum_property (N : ℕ) (a b c : ℕ) : 
  (10^2014 ≤ N) ∧ (N < 10^2015) ∧  -- N is a 2015-digit number
  (N % 9 = 0) ∧                    -- N is divisible by 9
  (a = sum_of_digits N) ∧          -- a is the sum of digits of N
  (b = sum_of_digits a) ∧          -- b is the sum of digits of a
  (c = sum_of_digits b)            -- c is the sum of digits of b
  → c = 9 :=
by
  sorry  -- Placeholder for the proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_property_l931_93119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_range_l931_93198

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (1+a)^x

-- State the theorem
theorem monotonic_increasing_range (a : ℝ) :
  a ∈ Set.Ioo 0 1 →
  (∀ x ∈ Set.Ioi 0, Monotone (fun x => f a x)) →
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_range_l931_93198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l931_93170

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

theorem f_monotonic_decreasing :
  (∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ → f x₂ < f x₁) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 1 → f x₂ < f x₁) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l931_93170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ages_l931_93184

/-- Given the ages of Beckett, Olaf, Shannen, Jack, and Emma, prove that their sum is 615 years. -/
theorem sum_of_ages (beckett olaf shannen jack emma : ℕ) : 
  beckett = 12 ∧ 
  beckett * beckett = olaf ∧ 
  shannen = olaf / 2 - 2 ∧ 
  jack = 5 + 2 * (shannen + Int.floor (Real.rpow (olaf : ℝ) (1/3 : ℝ))) ∧ 
  emma = Int.floor ((Real.sqrt (beckett : ℝ) + Real.sqrt (shannen : ℝ)) * ((jack : ℝ) - (olaf : ℝ))) →
  beckett + olaf + shannen + jack + emma = 615 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ages_l931_93184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_left_side_when_n_equals_one_l931_93103

theorem left_side_when_n_equals_one (a : ℝ) (h : a ≠ 1) : 
  (fun n : ℕ+ => (Finset.range (n : ℕ)).sum (fun i => a ^ i)) 1 = 1 + a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_left_side_when_n_equals_one_l931_93103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_three_l931_93196

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- A theorem about the eccentricity of a specific hyperbola configuration -/
theorem hyperbola_eccentricity_sqrt_three (h : Hyperbola) (c : Circle) (m p q : Point) :
  c.center = m →
  m.x^2 / h.a^2 - m.y^2 / h.b^2 = 1 →
  c.radius = |m.y| →
  p.x = 0 ∧ q.x = 0 →
  (m.x - p.x)^2 + (m.y - p.y)^2 = (m.x - q.x)^2 + (m.y - q.y)^2 →
  (m.x - p.x)^2 + (m.y - p.y)^2 = (p.x - q.x)^2 + (p.y - q.y)^2 →
  eccentricity h = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_three_l931_93196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_averaging_properties_l931_93169

-- Define the averaging operation
noncomputable def avg (x y : ℝ) : ℝ := (x + y) / 2

-- Theorem statement
theorem averaging_properties :
  (∀ x y : ℝ, avg x y = avg y x) ∧ 
  (∀ x y z : ℝ, x + avg y z = avg (x + y) (x + z)) ∧ 
  (∃ x y z : ℝ, avg (avg x y) z ≠ avg x (avg y z)) ∧ 
  (∃ x y z : ℝ, avg x (y + z) ≠ avg x y + avg x z) ∧ 
  (∀ i : ℝ, ∃ x : ℝ, avg x i ≠ x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_averaging_properties_l931_93169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l931_93121

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x - a/(2^x)

noncomputable def C1 (a : ℝ) (x : ℝ) : ℝ := 2^(x-2) - a/(2^(x-2))

noncomputable def C2 (a : ℝ) (x : ℝ) : ℝ := a/(2^(x-2)) - 2^(x-2)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := C2 a x + 2

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x / a + g a x

noncomputable def m (a : ℝ) : ℝ := 
  Real.sqrt ((1/a - 1/4) * (4*a - 1)) * 2 + 2

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, F a x ≥ m a) ∧ (m a > 2 + Real.sqrt 7) ↔ 1/2 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l931_93121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l931_93178

theorem simplify_and_rationalize :
  1 / (1 - 1 / (Real.sqrt 5 - 2)) = (1 - Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l931_93178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bead_cuboid_existence_coinciding_endpoints_l931_93181

/-- A unit cube with a hole drilled along its diagonal -/
structure BeadCube where
  -- No additional fields needed as it's a unit cube

/-- A string of bead cubes -/
def BeadString (p q r : ℕ) := Fin (p * q * r) → BeadCube

/-- A rectangular parallelepiped configuration -/
structure Parallelepiped (p q r : ℕ) where
  beads : BeadString p q r
  -- We assume the existence of a function that checks if the configuration is valid

/-- Theorem stating that for any positive integers p, q, and r, 
    it's possible to arrange p × q × r bead cubes into a parallelepiped -/
theorem bead_cuboid_existence (p q r : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  ∃ (config : Parallelepiped p q r), True := by
  sorry

/-- For a parallelepiped, A is the starting vertex and B is the terminal vertex -/
def start_end_vertices (p q r : ℕ) (config : Parallelepiped p q r) : Prop :=
  sorry -- Definition of A and B coinciding

/-- Theorem stating conditions for A and B to coincide -/
theorem coinciding_endpoints (p q r : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  ∃ (config : Parallelepiped p q r), start_end_vertices p q r config ↔ 
    (p % 2 = 0 ∧ q % 2 = 0) ∨ (p % 2 = 0 ∧ r % 2 = 0) ∨ (q % 2 = 0 ∧ r % 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bead_cuboid_existence_coinciding_endpoints_l931_93181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_given_tan_l931_93108

theorem sin_double_angle_given_tan (α : ℝ) (h : Real.tan α = 3) : Real.sin (2 * α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_given_tan_l931_93108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_is_seven_l931_93104

/-- An equilateral triangle with an internal point -/
structure EquilateralTriangleWithInternalPoint where
  /-- The vertices of the triangle -/
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  /-- The internal point -/
  P : ℝ × ℝ
  /-- The triangle is equilateral -/
  equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  /-- The distances from P to the vertices -/
  distPA : (P.1 - A.1)^2 + (P.2 - A.2)^2 = 4^2
  distPB : (P.1 - B.1)^2 + (P.2 - B.2)^2 = 5^2
  distPC : (P.1 - C.1)^2 + (P.2 - C.2)^2 = 3^2

/-- The side length of the equilateral triangle is 7 -/
theorem side_length_is_seven (t : EquilateralTriangleWithInternalPoint) :
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = 7^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_is_seven_l931_93104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l931_93180

/-- The set of foci of the ellipse -/
def set_of_foci_ellipse : Set (ℝ × ℝ) := sorry

/-- The set of foci of the hyperbola -/
def set_of_foci_hyperbola : Set (ℝ × ℝ) := sorry

/-- Predicate to check if a line is an asymptote of the hyperbola -/
def is_asymptote_of_hyperbola : (ℝ × ℝ → ℝ) → Prop := sorry

/-- Predicate to check if an equation represents the hyperbola -/
def is_equation_of_hyperbola : (ℝ × ℝ → ℝ) → Prop := sorry

/-- Given an ellipse and a hyperbola with shared foci, prove the equation of the hyperbola -/
theorem hyperbola_equation (x y : ℝ) :
  (∀ x y, 4 * x^2 + y^2 = 1 → ∃ c, c > 0 ∧ (0, c) ∈ set_of_foci_ellipse ∧ (0, c) ∈ set_of_foci_hyperbola) →
  (∃ k, k > 0 ∧ is_asymptote_of_hyperbola (fun (x, y) ↦ y - k * x)) →
  (is_asymptote_of_hyperbola (fun (x, y) ↦ y - Real.sqrt 2 * x)) →
  is_equation_of_hyperbola (fun (x, y) ↦ 4 * y^2 - 2 * x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l931_93180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_police_catch_thief_l931_93139

/-- Theorem: Police Catching Thief
  Given:
  - Thief's speed is 25 km/hr
  - Police officer's speed is 35 km/hr
  - Police station is 80 km away from the initial location
  - Police officer starts chasing after 1.5 hours
  Prove: The time taken by the police officer to catch the thief is 4.25 hours
-/
theorem police_catch_thief 
  (thief_speed : ℝ) 
  (police_speed : ℝ) 
  (station_distance : ℝ) 
  (delay_time : ℝ) 
  (h1 : thief_speed = 25)
  (h2 : police_speed = 35)
  (h3 : station_distance = 80)
  (h4 : delay_time = 1.5) :
  (station_distance - thief_speed * delay_time) / (police_speed - thief_speed) = 4.25 := by
  sorry

#check police_catch_thief

end NUMINAMATH_CALUDE_ERRORFEEDBACK_police_catch_thief_l931_93139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l931_93124

theorem max_negative_integers (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) :
  (Finset.filter (· < 0) {a, b, c, d, e, f}).card ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l931_93124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_5000_l931_93125

/-- Calculates the principal amount given the simple interest, rate, and time. -/
noncomputable def calculate_principal (interest : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (interest * 100) / (rate * time)

/-- Proves that the principal amount is 5000 given the specified conditions. -/
theorem principal_is_5000 (interest rate time : ℝ) 
    (h_interest : interest = 2000)
    (h_rate : rate = 4)
    (h_time : time = 10) :
    calculate_principal interest rate time = 5000 := by
  sorry

/-- Evaluates the principal amount for the given values. -/
def eval_principal : ℚ :=
  (2000 : ℚ) * 100 / ((4 : ℚ) * 10)

#eval eval_principal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_5000_l931_93125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_unit_interval_l931_93193

-- Define the function f(x) = 1/x
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- State the theorem
theorem f_decreasing_on_unit_interval :
  ∀ x y : ℝ, 0 < x → x < 1 → 0 < y → y < 1 → x < y → f y < f x := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_unit_interval_l931_93193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l931_93153

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 6) + 1 / 2

theorem omega_value (ω : ℝ) (hω : ω > 0) :
  (∃ α β : ℝ, f ω α = -1/2 ∧ f ω β = 1/2 ∧ 
    (∀ γ δ : ℝ, f ω γ = -1/2 → f ω δ = 1/2 → |γ - δ| ≥ 3*Real.pi/4) ∧
    |α - β| = 3*Real.pi/4) →
  ω = 2/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l931_93153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_proof_l931_93165

-- Define points in the Euclidean plane
variable (A B C P : EuclideanSpace ℝ (Fin 2))

-- P is inside triangle ABC
def P_inside_triangle (A B C P : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Vector equation
def vector_equation (A B C P : EuclideanSpace ℝ (Fin 2)) : Prop := 
  A - P + 3 • (B - P) + 4 • (C - P) = 0

-- Area of a triangle
noncomputable def area (X Y Z : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

theorem area_ratio_proof 
  (h_inside : P_inside_triangle A B C P) 
  (h_vector : vector_equation A B C P) :
  area A B C / area A P B = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_proof_l931_93165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_terminating_decimal_l931_93185

theorem smallest_n_for_terminating_decimal : 
  ∃ (n : ℕ+), n = 498 ∧ 
  (∀ (m : ℕ+), m < n → ¬(∃ (k : ℕ+), (m : ℚ) / (m + 127 : ℚ) = (k : ℚ) / (10^(Nat.log 10 k + 1) : ℚ))) ∧
  (∃ (k : ℕ+), (n : ℚ) / (n + 127 : ℚ) = (k : ℚ) / (10^(Nat.log 10 k + 1) : ℚ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_terminating_decimal_l931_93185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_contribution_ratio_l931_93175

/-- The cost of raising a child per year --/
def ChildRaisingCost : ℕ → ℕ
| n => if n < 8 then 10000 else 20000

/-- John's contribution --/
def JohnContribution : ℕ := 265000

/-- University tuition --/
def UniversityTuition : ℕ := 250000

/-- Years before university --/
def YearsBeforeUniversity : ℕ := 18

/-- Theorem stating the ratio of John's contribution to the total cost --/
theorem john_contribution_ratio :
  2 * JohnContribution = (Finset.sum (Finset.range YearsBeforeUniversity) ChildRaisingCost) + UniversityTuition :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_contribution_ratio_l931_93175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_equivalence_l931_93134

theorem condition_equivalence (A B : Prop) 
  (h1 : ¬A → B)  -- B is necessary for ¬A
  (h2 : ¬(B → ¬A))  -- B is not sufficient for ¬A
  : (¬B → A) ∧ ¬(A → ¬B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_equivalence_l931_93134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_from_volume_and_height_cone_radius_specific_case_l931_93192

/-- The volume of a cone with radius r and height h is (1/3) * π * r^2 * h -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem cone_radius_from_volume_and_height 
  (h : ℝ) (V : ℝ) (h_pos : h > 0) (V_pos : V > 0) :
  let r := Real.sqrt (3 * V / (Real.pi * h))
  cone_volume r h = V := by
sorry

/-- For a cone with height 4 cm and volume 12 cm³, the radius of the base is √(9/π) cm -/
theorem cone_radius_specific_case :
  let h : ℝ := 4
  let V : ℝ := 12
  let r := Real.sqrt (9 / Real.pi)
  cone_volume r h = V := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_from_volume_and_height_cone_radius_specific_case_l931_93192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_result_l931_93172

theorem thirteenth_result (total_results : ℕ) (total_average : ℚ) 
  (first_twelve_average : ℚ) (last_twelve_average : ℚ) 
  (h1 : total_results = 25)
  (h2 : total_average = 50)
  (h3 : first_twelve_average = 14)
  (h4 : last_twelve_average = 17)
  : (total_results : ℚ) * total_average = 
    12 * first_twelve_average + 12 * last_twelve_average + 878 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_result_l931_93172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_ordering_l931_93132

/-- Represents a decimal number with up to three decimal places and optional recurring digits -/
inductive Decimal where
  | Terminating (whole : ℕ) (d1 d2 d3 : ℕ)
  | RecurringTwo (whole : ℕ) (d1 : ℕ) (r1 r2 : ℕ)
  | RecurringOne (whole : ℕ) (d1 d2 : ℕ) (r : ℕ)

/-- Converts a Decimal to a rational number -/
def toRational (d : Decimal) : ℚ :=
  match d with
  | Decimal.Terminating w d1 d2 d3 => (w * 1000 + d1 * 100 + d2 * 10 + d3) / 1000
  | Decimal.RecurringTwo w d1 r1 r2 => 
      (w * 100 + d1 * 10 + r1) / 100 + (r2 - r1) / 990
  | Decimal.RecurringOne w d1 d2 r => 
      (w * 100 + d1 * 10 + d2) / 100 + r / 900

theorem decimal_ordering :
  let a := Decimal.Terminating 0 1 2 3
  let b := Decimal.Terminating 0 0 1 2
  let c := Decimal.RecurringTwo 0 1 2 3
  let d := Decimal.RecurringOne 0 1 2 3
  toRational b < toRational c ∧ toRational c < toRational d ∧ toRational d < toRational a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_ordering_l931_93132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_P_two_not_in_P_l931_93137

-- Define the set P
variable (P : Set Int)

-- Define the properties of set P
axiom P_contains_positive : ∃ x : Int, x > 0 ∧ x ∈ P
axiom P_contains_negative : ∃ x : Int, x < 0 ∧ x ∈ P
axiom P_contains_odd : ∃ x : Int, x % 2 ≠ 0 ∧ x ∈ P
axiom P_contains_even : ∃ x : Int, x % 2 = 0 ∧ x ∈ P
axiom one_not_in_P : 1 ∉ P
axiom P_closed_under_addition : ∀ x y : Int, x ∈ P → y ∈ P → (x + y) ∈ P

-- Theorem to prove
theorem zero_in_P_two_not_in_P (P : Set Int) : 0 ∈ P ∧ 2 ∉ P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_P_two_not_in_P_l931_93137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l931_93152

noncomputable def f (x : ℝ) : ℝ := (8/21) * x^4 - (80/21) * x^2 + 24/7

theorem function_properties :
  (∀ x, f x = f (-x)) ∧ 
  f (-1) = 0 ∧
  f (1/2) = 5/2 ∧
  f 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l931_93152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_max_earnings_l931_93177

/-- Calculates the maximum earnings for Mary in a week -/
def maxEarnings (maxHours regularRate regularHours firstOTRate secondOTRate thirdOTRate : ℕ) : ℕ :=
  sorry

theorem mary_max_earnings : 
  maxEarnings 70 8 20 10 12 14 = 840 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_max_earnings_l931_93177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_l931_93130

-- Define points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (5, 7)

-- Define the point that line l passes through
def P : ℝ × ℝ := (-1, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the line equation type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ (x y : ℝ), a * x + b * y + c = 0

-- Theorem statement
theorem line_l_equation :
  ∃ (l : Line),
    (l.a * P.1 + l.b * P.2 + l.c = 0) ∧
    (distance A (l.a, l.b) = distance B (l.a, l.b)) ∧
    ((l.a = 1 ∧ l.b = -1 ∧ l.c = 1) ∨ (l.a = 5 ∧ l.b = -4 ∧ l.c = 5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_l931_93130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_inequality_l931_93199

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / Real.exp x - x + 1

-- Part 1: Tangent line equation
theorem tangent_line_equation (x y : ℝ) :
  f 1 0 = 0 → (deriv (f 1)) 0 = -2 → 2 * x + y - 2 = 0 → y = f 1 x := by sorry

-- Part 2: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x < 0) → a ≤ -1 := by sorry

-- Part 3: Inequality
theorem inequality (x : ℝ) :
  x > 0 → 2 / Real.exp x - 2 < (1/2) * x^2 - x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_inequality_l931_93199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_partition_size_l931_93163

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def valid_partition (n : ℕ) (A B : Set ℕ) : Prop :=
  A ∪ B = Finset.range n.succ ∧
  A ∩ B = ∅ ∧
  (∀ x y, x ∈ A → y ∈ A → x ≠ y → ¬is_perfect_square (x + y)) ∧
  (∀ x y, x ∈ B → y ∈ B → x ≠ y → ¬is_perfect_square (x + y))

theorem max_partition_size :
  (∃ A B : Set ℕ, valid_partition 14 A B) ∧
  (∀ n > 14, ¬∃ A B : Set ℕ, valid_partition n A B) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_partition_size_l931_93163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_partitionable_iff_divisibility_and_inequality_l931_93107

/-- A set is k-partitionable if it can be partitioned into k disjoint subsets with equal sums -/
def IsKPartitionable (k : ℕ) (A : Finset ℕ) : Prop :=
  ∃ (partition : Finset (Finset ℕ)),
    partition.card = k ∧
    (∀ S, S ∈ partition → S ⊆ A) ∧
    (∀ S T, S ∈ partition → T ∈ partition → S ≠ T → S ∩ T = ∅) ∧
    (∀ S T, S ∈ partition → T ∈ partition → (S.sum id) = (T.sum id))

theorem k_partitionable_iff_divisibility_and_inequality
  (n k : ℕ) (h_n : n ≥ 1) :
  let A := Finset.range n
  IsKPartitionable k A ↔ (2 * k ∣ n * (n + 1)) ∧ (2 * k ≤ n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_partitionable_iff_divisibility_and_inequality_l931_93107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_and_sums_l931_93183

-- Define the arithmetic sequence a_n and its sum S_n
def a (n : ℕ) : ℚ := 3 * n
def S (n : ℕ) : ℚ := n * (a 1 + a n) / 2

-- Define the sequence b_n and its sum T_n
def b (n : ℕ) : ℚ := 3 * 2^(n-1)
def T (n : ℕ) : ℚ := 2 * b n - 3

-- Define the sequence c_n
def c (n : ℕ) : ℚ := if n % 2 = 1 then b n else a n

-- Define the sum Q_n of c_n
def Q (n : ℕ) : ℚ := 
  if n % 2 = 0
  then (2^n : ℚ) - 1 + 3/4 * (n^2 : ℚ) + 3/2 * (n : ℚ)
  else (2^(n+1) : ℚ) + 3/4 * (n^2 : ℚ) - 7/4

-- State the theorem
theorem sequences_and_sums :
  (a 2 = 6) ∧
  (S 5 = 45) ∧
  (∀ n, T n - 2 * b n + 3 = 0) →
  (∀ n, a n = 3 * n) ∧
  (∀ n, b n = 3 * 2^(n-1)) ∧
  (∀ n, Q n = if n % 2 = 0
              then (2^n : ℚ) - 1 + 3/4 * (n^2 : ℚ) + 3/2 * (n : ℚ)
              else (2^(n+1) : ℚ) + 3/4 * (n^2 : ℚ) - 7/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_and_sums_l931_93183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_l931_93112

/-- The time taken for hoses A, B, and C to fill the pool together -/
noncomputable def fillTime (rateA rateB rateC : ℝ) : ℝ := 1 / (rateA + rateB + rateC)

/-- The volume of the pool -/
def poolVolume : ℝ := 1

theorem pool_fill_time :
  ∀ (rateA rateB rateC : ℝ),
  rateA > 0 → rateB > 0 → rateC > 0 →
  fillTime rateA rateB 0 = 3 →
  fillTime rateA 0 rateC = 5 →
  fillTime 0 rateB rateC = 4 →
  fillTime rateA rateB rateC = 120 / 47 := by
  sorry

#eval (120 : Float) / 47

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_l931_93112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_A_and_parallel_to_direction_vector_l931_93113

-- Define the vectors
noncomputable def a : ℝ × ℝ := (6, 2)
noncomputable def b : ℝ × ℝ := (-4, 1/2)

-- Define the point A
noncomputable def A : ℝ × ℝ := (3, -1)

-- Define the direction vector
noncomputable def direction_vector : ℝ × ℝ := (a.1 + 2 * b.1, a.2 + 2 * b.2)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x + 3 * y = 3

-- Theorem statement
theorem line_passes_through_A_and_parallel_to_direction_vector :
  line_equation A.1 A.2 ∧
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ (x y : ℝ), line_equation x y → 
    (x - A.1, y - A.2) = (k * direction_vector.1, k * direction_vector.2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_A_and_parallel_to_direction_vector_l931_93113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l931_93110

/-- The sum of the infinite series ∑(1 / (n(n+1)(n+2))) from n=1 to infinity equals 1/2. -/
theorem infinite_series_sum : 
  ∑' n : ℕ, (1 : ℝ) / (n * (n + 1) * (n + 2)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l931_93110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_2_1_and_2_5_l931_93182

-- Define the distribution function F(x)
noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 2 then 0
  else if x ≤ 3 then (x - 2)^2
  else 1

-- Define the probability density function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 0
  else if x ≤ 3 then 2 * (x - 2)
  else 0

-- Theorem statement
theorem probability_between_2_1_and_2_5 :
  ∫ x in 2.1..2.5, f x = 0.24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_2_1_and_2_5_l931_93182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_1891_minutes_l931_93140

/-- Represents the position of a particle -/
structure Position where
  x : ℕ
  y : ℕ

/-- Represents a cycle in the particle's movement -/
inductive MovementCycle
  | Odd
  | Even

/-- Calculates the next position and cycle after a full cycle movement -/
def nextCyclePosition (pos : Position) (c : MovementCycle) : Position × MovementCycle :=
  match c with
  | MovementCycle.Odd => (⟨pos.x + 1, pos.y + 1⟩, MovementCycle.Even)
  | MovementCycle.Even => (⟨pos.x + 1, pos.y + 1⟩, MovementCycle.Odd)

/-- Calculates the final position of the particle after given number of minutes -/
def finalPosition (minutes : ℕ) : Position :=
  sorry

/-- Theorem stating that after 1891 minutes, the particle will be at position (45, 46) -/
theorem particle_position_after_1891_minutes :
  finalPosition 1891 = ⟨45, 46⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_1891_minutes_l931_93140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_ellipse_chord_l931_93174

/-- Given an ellipse and a chord with a specific midpoint, 
    prove the equation of the perpendicular bisector of the chord. -/
theorem perpendicular_bisector_of_ellipse_chord 
  (A B : ℝ × ℝ) -- Points A and B on the ellipse
  (h_ellipse : ∀ (P : ℝ × ℝ), P = A ∨ P = B → P.1^2 / 9 + P.2^2 / 5 = 1) -- A and B satisfy the ellipse equation
  (h_midpoint : (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) -- Midpoint of AB is (1,1)
  : ∃ (a b c : ℝ), a * A.1 + b * A.2 + c = 0 
               ∧ a * B.1 + b * B.2 + c = 0 
               ∧ a = 5 ∧ b = 9 ∧ c = -14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_ellipse_chord_l931_93174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preference_change_difference_l931_93114

/-- Represents the percentage of students preferring a subject -/
def Percentage := Fin 101

/-- Initial percentage of students preferring Science -/
def initial_science : Percentage := ⟨60, by norm_num⟩

/-- Initial percentage of students preferring Math -/
def initial_math : Percentage := ⟨40, by norm_num⟩

/-- Final percentage of students preferring Science -/
def final_science : Percentage := ⟨80, by norm_num⟩

/-- Final percentage of students preferring Math -/
def final_math : Percentage := ⟨20, by norm_num⟩

/-- The minimum possible change in student preferences -/
def min_change : ℕ := 20

/-- The maximum possible change in student preferences -/
def max_change : ℕ := 40

/-- Theorem stating that the difference between max and min change is 20% -/
theorem preference_change_difference :
  max_change - min_change = 20 := by
  rfl

#eval max_change - min_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_preference_change_difference_l931_93114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_partition_parity_l931_93197

theorem rectangle_partition_parity (m n : ℕ) (hm : Odd m) (hn : Odd n) :
  ∃ (x₁ y₁ x₂ y₂ : ℕ),
    x₁ < x₂ ∧ x₂ ≤ m ∧
    y₁ < y₂ ∧ y₂ ≤ n ∧
    (∀ i ∈ ({x₁, m - x₂, y₁, n - y₂} : Set ℕ), Even i ∨ 
     ∀ i ∈ ({x₁, m - x₂, y₁, n - y₂} : Set ℕ), Odd i) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_partition_parity_l931_93197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_travels_100_feet_l931_93157

/-- The distance Lindy travels when Jack and Christina meet -/
noncomputable def lindy_distance (initial_distance : ℝ) (jack_speed christina_speed lindy_speed : ℝ) : ℝ :=
  let meeting_time := initial_distance / (jack_speed + christina_speed)
  lindy_speed * meeting_time

/-- Theorem stating that Lindy travels 100 feet when Jack and Christina meet -/
theorem lindy_travels_100_feet :
  lindy_distance 150 7 8 10 = 100 := by
  -- Unfold the definition of lindy_distance
  unfold lindy_distance
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_travels_100_feet_l931_93157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_fraction_value_l931_93122

theorem trig_fraction_value (α : ℝ) 
  (h1 : Real.sin (2 * Real.pi - α) = 4/5)
  (h2 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_fraction_value_l931_93122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_fraction_is_one_fourth_l931_93138

/-- Represents a cube with side length 2 and three half-shaded faces -/
structure ShadedCube where
  side_length : ℚ
  num_shaded_faces : ℕ
  shaded_fraction : ℚ
  h_side_length : side_length = 2
  h_num_shaded_faces : num_shaded_faces = 3
  h_shaded_fraction : shaded_fraction = 1 / 2

/-- The fraction of the total surface area of the cube that is shaded -/
def shaded_area_fraction (cube : ShadedCube) : ℚ :=
  (cube.num_shaded_faces : ℚ) * cube.shaded_fraction * cube.side_length^2 / (6 * cube.side_length^2)

/-- Theorem stating that the shaded area fraction of the cube is 1/4 -/
theorem shaded_area_fraction_is_one_fourth (cube : ShadedCube) : 
  shaded_area_fraction cube = 1 / 4 := by
  -- Expand the definition of shaded_area_fraction
  unfold shaded_area_fraction
  
  -- Substitute the known values from the ShadedCube structure
  rw [cube.h_side_length, cube.h_num_shaded_faces, cube.h_shaded_fraction]
  
  -- Simplify the expression
  simp [pow_two]
  
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_fraction_is_one_fourth_l931_93138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_multiple_iff_not_div_twenty_l931_93171

/-- A positive integer is alternating if adjacent digits in its decimal representation have different parity. -/
def IsAlternating (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), n.digits 10 = digits ∧
    ∀ (i : ℕ), i + 1 < digits.length →
      (digits[i]! % 2 = 0 ∧ digits[i+1]! % 2 = 1) ∨
      (digits[i]! % 2 = 1 ∧ digits[i+1]! % 2 = 0)

/-- For a positive integer n, there exists a multiple of n that is an alternating number
    if and only if n is not divisible by 20. -/
theorem alternating_multiple_iff_not_div_twenty (n : ℕ) (hn : 0 < n) :
  (∃ (k : ℕ), 0 < k ∧ IsAlternating (k * n)) ↔ ¬(20 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_multiple_iff_not_div_twenty_l931_93171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mats_completed_equals_sum_l931_93133

variable (a b c d e f g h n : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (hd : d > 0)
variable (he : e > 0)
variable (hf : f > 0)
variable (hg : g > 0)
variable (hh : h > 0)
variable (hn : n > 0)

noncomputable def weaver_rate (x : ℝ) : ℝ := 1 / x

noncomputable def total_rate (a b c d e f g h : ℝ) : ℝ := 
  weaver_rate a + weaver_rate b + weaver_rate c + weaver_rate d +
  weaver_rate e + weaver_rate f + weaver_rate g + weaver_rate h

noncomputable def mats_completed (a b c d e f g h n : ℝ) : ℝ := 
  total_rate a b c d e f g h * (2 * n)

theorem mats_completed_equals_sum (a b c d e f g h n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (he : e > 0) (hf : f > 0) (hg : g > 0) (hh : h > 0) (hn : n > 0) :
  mats_completed a b c d e f g h n =
  (1/a + 1/b + 1/c + 1/d + 1/e + 1/f + 1/g + 1/h) * (2 * n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mats_completed_equals_sum_l931_93133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l931_93150

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m1 m2 : ℝ} : m1 = m2 ↔ m1 = m2

/-- Definition of the first line: 3y - 4a = 8x -/
def line1 (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y => 3 * y - 4 * a = 8 * x

/-- Definition of the second line: y - 2 = (a + 4)x -/
def line2 (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y => y - 2 = (a + 4) * x

/-- Slope of the first line -/
noncomputable def slope1 : ℝ := 8 / 3

/-- Slope of the second line -/
def slope2 (a : ℝ) : ℝ := a + 4

theorem parallel_lines_a_value :
  ∀ a : ℝ, slope1 = slope2 a → a = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l931_93150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l931_93151

/-- The volume of a regular tetrahedron with edge length 1 -/
noncomputable def regularTetrahedronVolume : ℝ := Real.sqrt 2 / 12

/-- Theorem: The volume of a regular tetrahedron with edge length 1 is √2/12 -/
theorem regular_tetrahedron_volume :
  regularTetrahedronVolume = Real.sqrt 2 / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l931_93151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l931_93162

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x) / (-1)

theorem f_monotone_decreasing :
  ∀ a b : ℝ, 2 < a ∧ a < b → f b < f a :=
by
  intros a b h
  sorry

#check f_monotone_decreasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l931_93162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_mean_median_l931_93105

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def sequence_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_mean_median
  (a₁ : ℝ)
  (a₂₀ : ℝ)
  (h₁ : a₁ = 4)
  (h₂ : a₂₀ = 42) :
  let d := (a₂₀ - a₁) / 19
  let mean := sequence_sum a₁ d 20 / 20
  let median := (arithmetic_sequence a₁ d 10 + arithmetic_sequence a₁ d 11) / 2
  mean = 23 ∧ median = 23 := by
  sorry

#check arithmetic_sequence_mean_median

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_mean_median_l931_93105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l931_93155

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : Real.cos (2 * α) + Real.sin α * (2 * Real.sin α - 1) = 2/5)
  (h2 : α ∈ Set.Ioo (Real.pi/2) Real.pi) :
  Real.tan (α + Real.pi/4) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l931_93155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_amount_proof_l931_93179

/-- The annual interest rate A charges B -/
def rate_A_to_B : ℚ := 10 / 100

/-- The annual interest rate B charges C -/
def rate_B_to_C : ℚ := 115 / 1000

/-- The number of years for the loan -/
def years : ℕ := 3

/-- B's gain over the loan period -/
def B_gain : ℚ := 180

/-- The amount lent by A to B -/
def amount_lent : ℚ := 1333333 / 1000

theorem loan_amount_proof :
  ∃ (P : ℚ), P > 0 ∧ 
    (rate_B_to_C * P * (years : ℚ) - rate_A_to_B * P * (years : ℚ) = B_gain) ∧
    P = amount_lent :=
by sorry

#eval amount_lent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_amount_proof_l931_93179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_exists_but_not_nat_pos_l931_93154

/-- A set of positive real numbers satisfying certain closure properties -/
structure SpecialSet where
  S : Set ℝ
  positive : ∀ x ∈ S, x > 0
  contains_one : 1 ∈ S
  closed_under_add : ∀ x y, x ∈ S → y ∈ S → x + y ∈ S
  closed_under_mul : ∀ x y, x ∈ S → y ∈ S → x * y ∈ S

/-- A subset of S that generates all elements of S \ {1} uniquely -/
structure GeneratingSubset (S : SpecialSet) where
  P : Set ℝ
  subset_of_S : P ⊆ S.S
  generates : ∀ x ∈ S.S, x ≠ 1 → ∃! (factors : Multiset ℝ), (∀ f ∈ factors, f ∈ P) ∧ (factors.prod = x)

/-- The statement that a SpecialSet with a GeneratingSubset exists but is not equal to ℕ+ -/
theorem special_set_exists_but_not_nat_pos : ∃ (S : SpecialSet) (G : GeneratingSubset S), ∃ x, x ∈ S.S ∧ x ∉ Set.Ioi (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_exists_but_not_nat_pos_l931_93154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l931_93173

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 - 4*x else ((-x)^2 - 4*(-x))

theorem solution_set_of_inequality (h : ∀ x, f (-x) = f x) :
  {x : ℝ | f (x + 2) < 5} = Set.Ioo (-7) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l931_93173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_has_winning_strategy_l931_93188

/-- The set of available digits for the game -/
def AvailableDigits : Set Nat := {1, 2, 3, 4, 5}

/-- The number of digits in the final number N -/
def TotalDigits : Nat := 2005

/-- A strategy for player B that responds to A's choice -/
def BStrategy (a_choice : Nat) : Nat :=
  6 - a_choice

/-- Represents a game state after each turn -/
structure GameState where
  turn : Nat
  digit_sum : Nat

/-- The game progression function -/
def play_turn (state : GameState) (a_choice : Nat) : GameState :=
  { turn := state.turn + 2,
    digit_sum := state.digit_sum + a_choice + BStrategy a_choice }

/-- The final game state after all turns -/
def final_state (state : GameState) : Prop :=
  state.turn = TotalDigits

/-- The winning condition for player A -/
def A_wins (state : GameState) : Prop :=
  state.digit_sum % 9 = 0

/-- The main theorem stating that B has a winning strategy -/
theorem B_has_winning_strategy :
  ∀ (initial_state : GameState),
  ∀ (game : Nat → GameState),
  ∀ (a_choice : Nat → Nat),
  (game 0 = initial_state) →
  (∀ n, game (n + 1) = play_turn (game n) (a_choice n)) →
  (∀ n, a_choice n ∈ AvailableDigits) →
  (∃ n, final_state (game n)) →
  ¬(A_wins (game (TotalDigits - 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_has_winning_strategy_l931_93188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_width_l931_93158

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r^(n-1)

def sum_geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then n * a₁ else a₁ * (1 - r^n) / (1 - r)

def S (n : ℕ) : ℚ := sum_geometric_sequence (3/2) (-1/2) n

theorem min_interval_width :
  ∃ (s t : ℚ), (∀ (n : ℕ), n ≥ 1 → S n - (S n)⁻¹ ∈ Set.Icc s t) ∧
    (∀ (s' t' : ℚ), (∀ (n : ℕ), n ≥ 1 → S n - (S n)⁻¹ ∈ Set.Icc s' t') →
      t - s ≤ t' - s') ∧
    t - s = 17/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_width_l931_93158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAB_l931_93111

-- Define the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Define the polar coordinate system
structure PolarPoint where
  r : ℝ
  θ : ℝ

-- Define line l
def line_l (t : ℝ) : Point :=
  { x := -1 + t, y := 1 + t }

-- Define curve C
def curve_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 5

-- Define point P
noncomputable def point_P : PolarPoint :=
  { r := 2 * Real.sqrt 2, θ := 7 * Real.pi / 4 }

-- Define line l'
def line_l' (x : ℝ) : ℝ :=
  x

-- Theorem statement
theorem area_of_triangle_PAB :
  ∃ (A B : Point),
    curve_C A.x A.y ∧
    curve_C B.x B.y ∧
    A.y = line_l' A.x ∧
    B.y = line_l' B.x ∧
    (1/2 : ℝ) * Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) * (2 * Real.sqrt 2) = 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAB_l931_93111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_symmetric_point_l931_93120

/-- Given a point M in 3D space, find the coordinates of the projection of M's symmetric point about the y-axis on the xOz plane. -/
theorem projection_of_symmetric_point (M : ℝ × ℝ × ℝ) (h : M = (4, 5, 6)) :
  let N := (-(M.fst), M.snd.fst, -(M.snd.snd))  -- Symmetric point about y-axis
  let P := (N.1, 0, N.2.2)      -- Projection on xOz plane
  P = (-4, 0, -6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_symmetric_point_l931_93120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_price_l931_93161

/-- Given a selling price and a markup percentage, calculate the cost price. -/
noncomputable def cost_price (selling_price : ℝ) (markup_percentage : ℝ) : ℝ :=
  selling_price / (1 + markup_percentage / 100)

/-- Theorem: The cost price of a computer table sold for 7350 with a 10% markup is approximately 6681.82 -/
theorem computer_table_cost_price :
  let selling_price : ℝ := 7350
  let markup_percentage : ℝ := 10
  abs (cost_price selling_price markup_percentage - 6681.82) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_price_l931_93161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_correct_l931_93167

/-- A quadrilateral in the complex plane -/
structure Quadrilateral :=
  (A B C D : ℂ)

/-- The given quadrilateral with specified points -/
noncomputable def given_quadrilateral : Quadrilateral :=
  { A := 0,
    B := 3 + 2*Complex.I,
    D := 2 - 4*Complex.I,
    C := 5 - 2*Complex.I }

/-- Theorem stating that the complex number corresponding to point C is correct -/
theorem point_C_correct (q : Quadrilateral) (h1 : q.A = 0) (h2 : q.B = 3 + 2*Complex.I) (h3 : q.D = 2 - 4*Complex.I) :
  q.C = 5 - 2*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_correct_l931_93167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l931_93106

noncomputable def f (x : ℝ) := Real.exp x

theorem tangent_line_at_zero (x y : ℝ) :
  (HasDerivAt f 1 0) →
  (f 0 = 1) →
  (x - y + 1 = 0 ↔ y = 1 * x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l931_93106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l931_93117

theorem remainder_problem (p q r s : ℕ) 
  (hp : p % 18 = 8)
  (hq : q % 18 = 11)
  (hr : r % 18 = 14)
  (hs : s % 18 = 15) :
  (3 * (p + q + r + s)) % 18 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l931_93117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_function_l931_93143

theorem max_value_sin_function (x : ℝ) :
  x ∈ Set.Icc 0 Real.pi →
  (Real.sqrt 2 * Real.sin (x - Real.pi/4)) ≤ Real.sqrt 2 :=
by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_function_l931_93143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_infinite_non_factorable_polynomials_l931_93186

/-- A structure representing a partition of positive integers -/
structure PositiveIntegerPartition (k : ℕ) where
  subsets : Fin k → Set ℕ+
  pairwise_disjoint : ∀ i j, i ≠ j → Disjoint (subsets i) (subsets j)
  cover_all : (⋃ i, subsets i) = Set.univ

/-- Definition of a non-factorable polynomial -/
def IsNonFactorable (p : Polynomial ℤ) : Prop :=
  ∀ q r : Polynomial ℤ, p = q * r → q.degree = 0 ∨ r.degree = 0

/-- Main theorem statement -/
theorem existence_of_infinite_non_factorable_polynomials
  {k n : ℕ} (hk : k > 1) (hn : n > 1)
  (partition : PositiveIntegerPartition k) :
  ∃ i : Fin k, ∃ f : ℕ → Polynomial ℤ,
    (∀ m, (f m).degree = n) ∧
    (∀ m, IsNonFactorable (f m)) ∧
    (∀ m j, ∃ a : ℕ+, a ∈ partition.subsets i ∧ (f m).coeff j = a) ∧
    (∀ m j l, j ≠ l → (f m).coeff j ≠ (f m).coeff l) ∧
    (∀ m₁ m₂, m₁ ≠ m₂ → f m₁ ≠ f m₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_infinite_non_factorable_polynomials_l931_93186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_is_280_l931_93194

/-- The total distance of a race where A beats B by 56 meters or 7 seconds,
    and A's time over the course is 28 seconds. -/
noncomputable def race_distance : ℝ := 280

/-- A's time to complete the race in seconds -/
noncomputable def a_time : ℝ := 28

/-- The distance by which A beats B in meters -/
noncomputable def beat_distance : ℝ := 56

/-- The time by which A beats B in seconds -/
noncomputable def beat_time : ℝ := 7

/-- A's speed in meters per second -/
noncomputable def a_speed : ℝ := race_distance / a_time

/-- B's speed in meters per second -/
noncomputable def b_speed : ℝ := beat_distance / beat_time

theorem race_distance_is_280 :
  (race_distance - beat_distance) / a_time = b_speed ∧
  race_distance / a_time = a_speed ∧
  race_distance = 280 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_is_280_l931_93194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_n_values_l931_93136

-- Define the piecewise function f
noncomputable def f (n : ℝ) (x : ℝ) : ℝ :=
  if x < n then x^2 + 2 else 2*x + 5

-- Define continuity at a point
def continuous_at_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → |f y - f x| < ε

-- Theorem statement
theorem sum_of_n_values :
  (∃ n : ℝ, ∀ x : ℝ, f n x = if x < n then x^2 + 2 else 2*x + 5) →
  (∀ n : ℝ, ∀ x : ℝ, continuous_at_point (f n) x) →
  (∃ n₁ n₂ : ℝ, (∀ n : ℝ, (∀ x : ℝ, continuous_at_point (f n) x) → (n = n₁ ∨ n = n₂)) ∧ n₁ + n₂ = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_n_values_l931_93136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l931_93101

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3) + Real.tan (5 * Real.pi / 6) * Real.cos (2 * x)

theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  -- Equation of symmetry axis
  (∀ (k : ℤ), ∃ (x : ℝ), x = k * Real.pi / 2 + Real.pi / 6 ∧ ∀ (y : ℝ), f (x - y) = f (x + y)) ∧
  -- Range of f(x) on (0, π/2)
  (∀ (y : ℝ), y ∈ Set.Ioo (-Real.sqrt 3 / 6) (Real.sqrt 3 / 3) ↔ ∃ (x : ℝ), x ∈ Set.Ioo 0 (Real.pi / 2) ∧ f x = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l931_93101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_initial_age_l931_93195

/-- Represents the age and height of a tree over time -/
structure TreeGrowth where
  initialAge : ℕ
  initialHeight : ℕ
  growthRate : ℕ

/-- Proves that the initial age of the tree is 1 year -/
theorem tree_initial_age (t : TreeGrowth) 
  (h1 : t.initialHeight = 5)
  (h2 : t.growthRate = 3)
  (h3 : t.initialHeight + 7 * t.growthRate = 23) :
  t.initialAge = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_initial_age_l931_93195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l931_93176

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := Set.Icc (-2) 2

-- Define the properties of f
axiom f_even : ∀ x, x ∈ domain → f x = f (-x)
axiom f_monotone : ∀ x y, x ∈ Set.Icc (-2) 0 → y ∈ Set.Icc (-2) 0 → x ≤ y → f x ≤ f y

-- Define the solution set
def solution_set : Set ℝ := Set.union (Set.Icc (-3) (-2)) (Set.Icc 0 1)

-- State the theorem
theorem solution_set_correct :
  ∀ x, x ∈ domain → (f (x + 1) ≤ f (-1) ↔ x ∈ solution_set) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l931_93176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_digits_for_four_thirteenths_l931_93149

theorem sum_of_repeating_digits_for_four_thirteenths :
  ∃ (a b : ℕ), (4 : ℚ) / 13 = 
    (∑' n, (10 * a + b : ℚ) / 100^(n + 1)) ∧ 
    a + b = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_digits_for_four_thirteenths_l931_93149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_polar_equation_l931_93100

/-- Given a point P with polar coordinates (π, π), the equation of the line 
    passing through P and perpendicular to the polar axis in polar coordinates 
    is ρ = -π / cos θ. -/
theorem perpendicular_line_polar_equation (P : Prod ℝ ℝ) 
    (h : P = (π, π)) : 
  ∀ θ ρ : ℝ, (ρ * Real.cos θ = -π) ↔ (ρ = -π / Real.cos θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_polar_equation_l931_93100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l931_93156

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x + 7 + a) / (x + 1)

/-- Theorem stating that if f(x) ≥ 4 for all positive integers x, then a ≥ 1/3 -/
theorem f_lower_bound (a : ℝ) : 
  (∀ x : ℕ+, f a (x : ℝ) ≥ 4) → a ≥ 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l931_93156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_price_equality_l931_93146

/-- Represents the price of a big fish today -/
noncomputable def big_fish_today : ℝ := sorry

/-- Represents the price of a small fish today -/
noncomputable def small_fish_today : ℝ := sorry

/-- Represents the price of a big fish yesterday -/
noncomputable def big_fish_yesterday : ℝ := sorry

/-- Represents the price of a small fish yesterday -/
noncomputable def small_fish_yesterday : ℝ := sorry

/-- The first condition: three big fish and one small fish today cost as much as five big fish yesterday -/
axiom condition1 : 3 * big_fish_today + small_fish_today = 5 * big_fish_yesterday

/-- The second condition: two big fish and one small fish today cost as much as three big fish and one small fish yesterday -/
axiom condition2 : 2 * big_fish_today + small_fish_today = 3 * big_fish_yesterday + small_fish_yesterday

/-- The theorem to be proved -/
theorem fish_price_equality : 
  big_fish_today + 2 * small_fish_today = 5 * small_fish_yesterday := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_price_equality_l931_93146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangents_coincidence_l931_93142

/-- A parabola represented by the equation y = x^2 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2}

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The set of points on a circle -/
def Circle.toSet (c : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2}

/-- Tangent line to the parabola at a point -/
def tangentParabola (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.2 - p.2 = 2 * p.1 * (q.1 - p.1)}

/-- Tangent line to a circle at a point -/
def tangentCircle (c : Circle) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | (q.1 - p.1) * (p.1 - c.center.1) + (q.2 - p.2) * (p.2 - c.center.2) = 0}

/-- The main theorem -/
theorem parabola_circle_tangents_coincidence :
  ∃ (c : Circle) (A B : ℝ × ℝ),
    A ∈ Parabola ∩ c.toSet ∧
    B ∈ Parabola ∩ c.toSet ∧
    A ≠ B ∧
    (∀ p : ℝ × ℝ, p ∈ Parabola ∩ c.toSet → p = A ∨ p = B) ∧
    tangentParabola A = tangentCircle c A ∧
    tangentParabola B ≠ tangentCircle c B :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangents_coincidence_l931_93142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_minimized_l931_93131

/-- Triangle ABC with sides AB=5, BC=4, CA=3 -/
structure Triangle where
  AB : ℝ
  BC : ℝ
  CA : ℝ
  ab_eq : AB = 5
  bc_eq : BC = 4
  ca_eq : CA = 3

/-- Area of triangle formed by ants' positions after t seconds -/
noncomputable def A (triangle : Triangle) (t : ℝ) : ℝ :=
  (1 / 10) * (47 * t - 12 * t^2)

/-- Theorem stating that A(t) is minimized when t = 47/24 -/
theorem area_minimized (triangle : Triangle) :
  ∀ t : ℝ, 0 < t → t < 3 →
    A triangle (47 / 24) ≤ A triangle t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_minimized_l931_93131
