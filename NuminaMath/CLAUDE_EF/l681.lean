import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_g_value_l681_68147

-- Define the tetrahedron
def Tetrahedron (E F G H : ℝ × ℝ × ℝ) : Prop :=
  norm (E - H) = 20 ∧ norm (F - G) = 20 ∧
  norm (E - G) = 26 ∧ norm (F - H) = 26 ∧
  norm (E - F) = 30 ∧ norm (G - H) = 30

-- Define the function g
def g (E F G H Y : ℝ × ℝ × ℝ) : ℝ :=
  norm (E - Y) + norm (F - Y) + norm (G - Y) + norm (H - Y)

-- Theorem statement
theorem min_g_value {E F G H : ℝ × ℝ × ℝ} (tet : Tetrahedron E F G H) :
  ∃ (Y : ℝ × ℝ × ℝ), ∀ (Z : ℝ × ℝ × ℝ), g E F G H Y ≤ g E F G H Z ∧ g E F G H Y = 2 * Real.sqrt 1323 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_g_value_l681_68147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PMN_l681_68187

/-- An ellipse with focus on the y-axis, major axis length 4, and eccentricity √3/2 -/
structure Ellipse where
  focus_on_y_axis : ℝ
  major_axis_length : ℝ
  eccentricity : ℝ
  h_major_axis : major_axis_length = 4
  h_eccentricity : eccentricity = Real.sqrt 3 / 2

/-- A line with equation y = kx + 1 -/
structure Line where
  k : ℝ
  b : ℝ
  h_b : b = 1

/-- A point P with coordinates (0, -3) -/
def point_P : ℝ × ℝ := (0, -3)

/-- The maximum area of triangle PMN -/
noncomputable def max_area (e : Ellipse) (l : Line) : ℝ := 2 * Real.sqrt 3

/-- Theorem stating the maximum area of triangle PMN -/
theorem max_area_triangle_PMN (e : Ellipse) (l : Line) :
  max_area e l = 2 * Real.sqrt 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PMN_l681_68187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_operation_result_l681_68158

-- Define the star operation
noncomputable def star (a b : ℝ) : ℝ := (a + b) / (a - b)

-- Theorem statement
theorem star_operation_result : star (star 2 5) (-1) = 5/2 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_operation_result_l681_68158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_monotonic_f_l681_68118

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x)

-- State the theorem
theorem omega_range_for_monotonic_f :
  ∀ ω : ℝ, 
    ω > 0 →
    (∀ x y : ℝ, π / 6 ≤ x ∧ x < y ∧ y ≤ π / 4 → f ω x < f ω y) →
    (ω ∈ Set.Ioc 0 (2/3) ∪ Set.Icc 7 (26/3)) :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_monotonic_f_l681_68118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_negative_420_degrees_sec_negative_420_degrees_result_l681_68155

theorem sec_negative_420_degrees : Real.cos ((-420 : ℝ) * π / 180) = 1 / 2 := by
  sorry

theorem sec_negative_420_degrees_result : 1 / Real.cos ((-420 : ℝ) * π / 180) = 2 := by
  have h : Real.cos ((-420 : ℝ) * π / 180) = 1 / 2 := sec_negative_420_degrees
  rw [h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_negative_420_degrees_sec_negative_420_degrees_result_l681_68155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_relation_l681_68175

/-- A geometric sequence with first term 1 and common ratio 2/3 -/
noncomputable def geometric_sequence (n : ℕ) : ℝ :=
  (2/3) ^ (n - 1)

/-- Sum of the first n terms of the geometric sequence -/
noncomputable def geometric_sum (n : ℕ) : ℝ :=
  (1 - (2/3)^n) / (1 - 2/3)

/-- Theorem stating the relationship between the sum and the nth term -/
theorem geometric_sum_relation (n : ℕ) :
  geometric_sum n = 3 - 2 * geometric_sequence n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_relation_l681_68175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_B_is_1700_l681_68101

/-- Given the investments and profit difference between A and C, calculate B's profit share -/
def calculate_profit_share_B (investment_A investment_B investment_C : ℚ) 
  (profit_diff_A_C : ℚ) : ℚ :=
  let total_investment := investment_A + investment_B + investment_C
  let profit_ratio_A := investment_A / total_investment
  let profit_ratio_B := investment_B / total_investment
  let profit_ratio_C := investment_C / total_investment
  let total_profit := profit_diff_A_C * total_investment / (profit_ratio_C - profit_ratio_A)
  profit_ratio_B * total_profit

/-- Prove that B's profit share is 1700 given the specific investments and profit difference -/
theorem profit_share_B_is_1700 :
  calculate_profit_share_B 8000 10000 12000 680 = 1700 := by
  sorry

#eval calculate_profit_share_B 8000 10000 12000 680

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_B_is_1700_l681_68101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gamma_length_l681_68110

noncomputable section

variable (α β γ : ℝ × ℝ)

def isPerpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def isUnitVector (v : ℝ × ℝ) : Prop :=
  v.1 * v.1 + v.2 * v.2 = 1

def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def vectorLength (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem max_gamma_length
  (h1 : isPerpendicular α β)
  (h2 : isUnitVector α)
  (h3 : isUnitVector β)
  (h4 : dotProduct (5 • α - 2 • γ) (12 • β - 2 • γ) = 0) :
  vectorLength γ ≤ 13/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gamma_length_l681_68110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_equation_solutions_l681_68193

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the golden ratio and its conjugate
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def φ_conj : ℝ := (1 - Real.sqrt 5) / 2

theorem fibonacci_equation_solutions :
  ∀ x : ℝ, x^2010 = (fib 2009 : ℝ) * x + (fib 2008 : ℝ) ↔ x = φ ∨ x = φ_conj := by
  sorry

#check fibonacci_equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_equation_solutions_l681_68193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l681_68137

noncomputable def f (x α β : ℝ) : ℝ := Real.sin (x - α) * Real.cos (x - β)

theorem f_properties (α β : ℝ) :
  (∃ p : ℝ, p > 0 ∧ p = π ∧ ∀ x : ℝ, f x α β = f (x + p) α β) ∧
  (∀ x : ℝ, f (α + x) α β = f (α - x) α β) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l681_68137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_minimum_value_in_interval_l681_68115

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (x/2) * Real.cos (x/2) - Real.sqrt 2 * (Real.sin (x/2))^2

-- Theorem for the smallest positive period
theorem smallest_positive_period : 
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧ 
  p = 2 * Real.pi := by
  sorry

-- Theorem for the minimum value in the interval [-π, 0]
theorem minimum_value_in_interval : 
  ∃ (m : ℝ), (∀ (x : ℝ), -Real.pi ≤ x ∧ x ≤ 0 → f x ≥ m) ∧ 
  (∃ (x : ℝ), -Real.pi ≤ x ∧ x ≤ 0 ∧ f x = m) ∧ 
  m = -1 - Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_minimum_value_in_interval_l681_68115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positional_relationships_of_lines_l681_68148

-- Define the three-dimensional space
structure Space3D where
  -- (We don't need to define the specifics of the 3D space for this problem)

-- Define a line in 3D space
structure Line3D (S : Space3D) where
  -- (We don't need to define the specifics of a line for this problem)

-- Define the positional relationships
inductive PositionalRelationship
  | Parallel
  | Intersecting
  | SkewLines

-- State the theorem
theorem positional_relationships_of_lines (S : Space3D) (l1 l2 : Line3D S) :
  ∃ (r : PositionalRelationship), r = PositionalRelationship.Parallel ∨ 
                                   r = PositionalRelationship.Intersecting ∨ 
                                   r = PositionalRelationship.SkewLines :=
by
  -- Introduce the existential quantifier
  use PositionalRelationship.Parallel
  -- Prove that the chosen relationship satisfies the condition
  apply Or.inl
  rfl
  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positional_relationships_of_lines_l681_68148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_monotonic_f_l681_68166

noncomputable def f (x : ℝ) := 3 * Real.sin (x + Real.pi / 10) - 2

theorem max_a_for_monotonic_f :
  ∀ a : ℝ, 
  (∀ x y : ℝ, Real.pi / 2 ≤ x ∧ x < y ∧ y ≤ a → (f x < f y ∨ f x > f y)) →
  a ≤ 7 * Real.pi / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_monotonic_f_l681_68166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l681_68157

/-- Geometric sequence with first term 4 and common ratio q -/
noncomputable def geometric_sequence (q : ℝ) : ℕ → ℝ :=
  fun n => 4 * q ^ (n - 1)

/-- Sum of the first n terms of the geometric sequence -/
noncomputable def sum_geometric (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then 4 * n else 4 * (1 - q^n) / (1 - q)

/-- The sequence {S_n + 2} is geometric -/
def sum_plus_two_is_geometric (q : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → (sum_geometric q (n + 1) + 2) * (sum_geometric q (n - 1) + 2) = (sum_geometric q n + 2)^2

theorem geometric_sequence_property (q : ℝ) :
  sum_plus_two_is_geometric q → q = 3 := by
  sorry

#check geometric_sequence_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l681_68157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_with_area_l681_68127

/-- Given a line L1 with equation 4x - 3y + 5 = 0, prove that the line L2 perpendicular to L1
    that forms a triangle with the coordinate axes having an area of 24 has the equation
    3x + 4y ± 12√2 = 0 -/
theorem perpendicular_line_with_area (L1 L2 : Set (ℝ × ℝ)) (m : ℝ) :
  (∀ x y, (x, y) ∈ L1 ↔ 4 * x - 3 * y + 5 = 0) →
  (∀ x y, (x, y) ∈ L2 ↔ 3 * x + 4 * y + m = 0) →
  (∀ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ L1 → (x₂, y₂) ∈ L2 → (4 * (x₂ - x₁) + 3 * (y₂ - y₁) = 0)) →
  abs (m * m / 24) = 24 →
  m = 12 * Real.sqrt 2 ∨ m = -12 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_with_area_l681_68127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_is_nine_l681_68100

/-- A frustum with given top and bottom base areas and a parallel intersection plane -/
structure Frustum where
  top_area : ℝ
  bottom_area : ℝ
  intersection_plane : ℝ
  h1 : top_area = 1
  h2 : bottom_area = 16
  h3 : intersection_plane > 0
  h4 : intersection_plane < 1

/-- The ratio of distances from the intersection plane to the top and bottom bases -/
def distance_ratio : ℝ := 2

/-- The area of the intersection plane -/
noncomputable def intersection_area (f : Frustum) : ℝ :=
  ((Real.sqrt f.top_area + 2 * Real.sqrt f.bottom_area) / 3) ^ 2

/-- Theorem stating that the area of the intersection is 9 -/
theorem intersection_area_is_nine (f : Frustum) :
  intersection_area f = 9 := by
  sorry

#eval distance_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_is_nine_l681_68100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dany_farm_consumption_l681_68169

/-- Represents the number of bushels consumed by a group of animals per day -/
def animalConsumption (numAnimals : ℕ) (bushelsPerAnimal : ℚ) : ℚ :=
  (numAnimals : ℚ) * bushelsPerAnimal

/-- Calculates the total bushels consumed by all animals on Dany's farm for one day -/
def totalBushelsConsumed (cowConsumption sheepConsumption chickenConsumption pigConsumption horseConsumption : ℚ) : ℚ :=
  cowConsumption + sheepConsumption + chickenConsumption + pigConsumption + horseConsumption

theorem dany_farm_consumption :
  let cowConsumption := animalConsumption 5 (35/10)
  let sheepConsumption := animalConsumption 4 (175/100)
  let chickenConsumption := animalConsumption 8 (125/100)
  let pigConsumption := animalConsumption 6 (45/10)
  let horseConsumption := animalConsumption 2 (575/100)
  totalBushelsConsumed cowConsumption sheepConsumption chickenConsumption pigConsumption horseConsumption = 73 := by
  sorry

#eval totalBushelsConsumed 
  (animalConsumption 5 (35/10))
  (animalConsumption 4 (175/100))
  (animalConsumption 8 (125/100))
  (animalConsumption 6 (45/10))
  (animalConsumption 2 (575/100))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dany_farm_consumption_l681_68169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_factorial_is_seven_factorial_l681_68188

def b : ℕ := 9

theorem unknown_factorial_is_seven_factorial :
  ∀ n : ℕ,
  (Nat.gcd (Nat.factorial (b - 2)) (Nat.factorial n) = 5040) →
  (Nat.gcd (Nat.factorial n) (Nat.factorial (b + 4)) = 5040) →
  Nat.factorial n = Nat.factorial 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_factorial_is_seven_factorial_l681_68188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l681_68179

theorem binomial_expansion_properties :
  ∀ (a : ℕ → ℝ),
  (∀ x : ℝ, (1 + 2*x)^10 = a 0 + (Finset.range 11).sum (λ i ↦ a (i+1) * x^(i+1))) →
  ((Finset.range 10).sum (λ i ↦ a (i+1)) = 3^10 - 1) ∧ (a 2 = 9 * a 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l681_68179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lighthouse_pentagon_perimeter_l681_68144

/-- Represents a point on the circular lake shore -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the pentagon formed by the lighthouses -/
structure Pentagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point

/-- The diameter of the circular lake -/
def lake_diameter : ℝ := 10

/-- Checks if two points form a diameter of the circle -/
def is_diameter (p1 p2 : Point) : Prop := 
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = lake_diameter^2

/-- Calculates the angle between three points -/
noncomputable def angle (center p1 p2 : Point) : ℝ := sorry

/-- Checks if lights appear equally spaced from a given point -/
def equally_spaced (center : Point) (p1 p2 p3 p4 : Point) : Prop := 
  ∃ θ : ℝ, θ > 0 ∧ 
    ∀ (i j : Fin 4), i < j → 
      (angle center (([p1, p2, p3, p4].get i)) (([p1, p2, p3, p4].get j))) = (j - i) * θ

/-- Calculates the distance between two points -/
noncomputable def dist (p1 p2 : Point) : ℝ := 
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the perimeter of the pentagon -/
noncomputable def perimeter (p : Pentagon) : ℝ := 
  dist p.A p.B + dist p.B p.C + dist p.C p.D + dist p.D p.E + dist p.E p.A

/-- Main theorem -/
theorem lighthouse_pentagon_perimeter (p : Pentagon) 
  (h1 : is_diameter p.A p.D)
  (h2 : is_diameter p.B p.E)
  (h3 : equally_spaced p.A p.B p.C p.D p.E) :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ perimeter p = m + Real.sqrt n ∧ m + n = 95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lighthouse_pentagon_perimeter_l681_68144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l681_68196

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 1

theorem f_properties :
  (∀ x : ℝ, f x ≤ 1) ∧ 
  (∃ x : ℝ, f x = 1) ∧
  (∀ x : ℝ, f x ≥ 5/6) ∧
  (∃ x : ℝ, f x = 5/6) ∧
  (deriv f (3/2) = 3/4) ∧
  (∫ x in (0)..(3/2), 1 - f x = 9/64) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l681_68196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_DBCE_l681_68161

/-- Represents a triangle in the diagram -/
structure Triangle where
  base : ℝ
  height : ℝ
  area : ℝ

/-- The main triangle ABC -/
def ABC : Triangle := { base := 0, height := 0, area := 72 }

/-- One of the smallest triangles -/
def SmallTriangle : Triangle := { base := 0, height := 0, area := 2 }

/-- Triangle ADE composed of 3 smallest triangles -/
def ADE : Triangle := { base := 0, height := 0, area := 6 }

/-- Trapezoid DBCE -/
def DBCE : Triangle := { base := 0, height := 0, area := 0 }

/-- All triangles are similar to ABC -/
axiom all_similar : ∀ t : Triangle, ∃ k : ℝ, t.base = k * ABC.base ∧ t.height = k * ABC.height

/-- ABC is isosceles with AB = AC -/
axiom ABC_isosceles : ABC.base = ABC.height

/-- There are 6 smallest triangles -/
axiom six_small_triangles : ∃ (ts : Finset Triangle), ts.card = 6 ∧ ∀ t ∈ ts, t = SmallTriangle

/-- The area of DBCE is the difference between ABC and ADE -/
axiom DBCE_area : DBCE.area = ABC.area - ADE.area

/-- The main theorem: The area of trapezoid DBCE is 66 -/
theorem area_of_DBCE : DBCE.area = 66 := by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_DBCE_l681_68161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_logarithm_proposition_l681_68185

theorem negation_logarithm_proposition (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  (¬ ∃ x > 1, Real.log x / Real.log a > 0) ↔ (∀ x > 1, Real.log x / Real.log a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_logarithm_proposition_l681_68185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_collection_bounds_l681_68136

/-- A structure representing a collection of sets with specific properties. -/
structure SetCollection (n : ℕ) where
  m : ℕ
  sets : Fin m → Finset (Fin n)
  set_size : ∀ i, (sets i).card = 3
  set_subset : ∀ i, sets i ⊆ Finset.univ
  intersection_size : ∀ i j, i < j → (sets i ∩ sets j).card ≤ 1

/-- The main theorem stating the bounds on m for a valid SetCollection. -/
theorem set_collection_bounds (n : ℕ) (h : n ≥ 3) :
  (∃ sc : SetCollection n, sc.m ≤ n * (n - 1) / 6) ∧
  (∃ sc : SetCollection n, sc.m ≥ (n - 1) * (n - 2) / 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_collection_bounds_l681_68136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l681_68195

theorem diophantine_equation_solution :
  {(x, y) : ℤ × ℤ | x^2 - 3*y^2 + 2*x*y - 2*x - 10*y + 20 = 0} =
  {(19, -7), (-15, 5), (7, 5), (-3, -7)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l681_68195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_special_polynomials_l681_68102

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Check if a polynomial has a coefficient with absolute value greater than 2015 -/
def has_large_coeff (p : IntPolynomial) : Prop :=
  ∃ n : ℕ, |p n| > 2015

/-- Check if all coefficients of a polynomial have absolute value ≤ 1 -/
def all_small_coeffs (p : IntPolynomial) : Prop :=
  ∀ n : ℕ, |p n| ≤ 1

/-- Multiplication of two integer polynomials -/
def mult_poly (f g : IntPolynomial) : IntPolynomial :=
  λ n => (Finset.range (n + 1)).sum (λ i => f i * g (n - i))

/-- Theorem: There exist two polynomials with integer coefficients satisfying the given conditions -/
theorem exist_special_polynomials : ∃ f g : IntPolynomial,
  has_large_coeff f ∧ has_large_coeff g ∧ all_small_coeffs (mult_poly f g) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_special_polynomials_l681_68102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_intersection_ratio_l681_68162

/-- A parabola in a 2D plane --/
structure Parabola where
  -- Add necessary fields for a parabola
  mk ::

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space --/
structure Line where
  -- Add necessary fields for a line
  mk ::

/-- Represents a chord of the parabola --/
structure Chord where
  start : Point
  endpoint : Point

/-- Checks if a line is tangent to a parabola --/
def isTangent (l : Line) (p : Parabola) : Prop := sorry

/-- Checks if two lines are parallel --/
def isParallel (l1 l2 : Line) : Prop := sorry

/-- Checks if a point lies on a line --/
def pointOnLine (pt : Point) (l : Line) : Prop := sorry

/-- Calculates the ratio in which a point divides a line segment --/
noncomputable def divisionRatio (p1 p2 p : Point) : ℝ := sorry

/-- Converts a chord to a line --/
def chordToLine (c : Chord) : Line := sorry

/-- Main theorem --/
theorem parabola_chord_intersection_ratio 
  (p : Parabola) 
  (t1 t2 : Line) 
  (p1 p2 : Point) 
  (c1 c2 : Chord) 
  (n : Point) :
  isTangent t1 p → 
  isTangent t2 p → 
  pointOnLine p1 t1 → 
  pointOnLine p2 t2 → 
  isParallel (chordToLine c1) t2 → 
  isParallel (chordToLine c2) t1 → 
  p1 = c1.start → 
  p2 = c2.start → 
  pointOnLine n (chordToLine c1) → 
  pointOnLine n (chordToLine c2) → 
  divisionRatio c1.start c1.endpoint n = 1/3 ∧ 
  divisionRatio c2.start c2.endpoint n = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_intersection_ratio_l681_68162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l681_68112

/-- Proves that the downstream distance is 80 km given the conditions of the boat problem -/
theorem boat_downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (upstream_distance : ℝ) 
  (downstream_distance : ℝ)
  (h1 : boat_speed = 30)
  (h2 : stream_speed = 10)
  (h3 : upstream_distance = 40)
  (h4 : (upstream_distance / (boat_speed - stream_speed)) = 
        (downstream_distance / (boat_speed + stream_speed))) :
  downstream_distance = 80 := by
  sorry

#check boat_downstream_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l681_68112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l681_68197

-- Define the function f(x) = a^x - b
noncomputable def f (a b x : ℝ) : ℝ := a^x - b

-- Define what it means for a function to not pass through the second quadrant
def not_in_second_quadrant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < 0 → y > 0 → f x ≠ y

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → not_in_second_quadrant (f a b)) ∧
  (∃ a b : ℝ, a > 0 ∧ a ≠ 1 ∧ not_in_second_quadrant (f a b) ∧ ¬(a > 1 ∧ b > 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l681_68197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l681_68170

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) - 1
noncomputable def g (x : ℝ) : ℝ := Real.log (x + 1)

-- Define the solution set
def solution_set : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | f (g x) - g (f x) ≤ 1} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l681_68170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elvie_age_is_10_l681_68103

-- Define the ages as natural numbers
def elvie_age (e : ℕ) : Prop := true
def arielle_age (a : ℕ) : Prop := true

-- Define the condition that relates their ages
def age_relation (e a : ℕ) : Prop := e + a + e * a = 131

-- State the theorem
theorem elvie_age_is_10 :
  ∀ e a : ℕ,
  elvie_age e →
  arielle_age a →
  age_relation e a →
  a = 11 →
  e = 10 :=
by
  intro e a he ha hrel h11
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elvie_age_is_10_l681_68103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_find_b_l681_68178

-- Define the vector dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the function f
noncomputable def f (A : ℝ) : ℝ := dot_product (1, 1) (-Real.cos A, Real.sin A)

-- Theorem for the range of f
theorem range_of_f :
  ∀ A, 0 < A → A < Real.pi → -1 < f A ∧ f A ≤ Real.sqrt 2 := by
  sorry

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Theorem for finding b
theorem find_b (t : Triangle) (h1 : f t.A = Real.sqrt 2 / 2)
  (h2 : t.C = Real.pi / 3) (h3 : t.c = Real.sqrt 6) : t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_find_b_l681_68178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_properties_l681_68120

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 1 = 0

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := ⟨-1, -1⟩
noncomputable def B : ℝ × ℝ := ⟨1, -1⟩

-- Theorem stating the properties of the common chord AB
theorem common_chord_properties :
  (∃ (a b : ℝ), circle1 a b ∧ circle2 a b) →
  (∀ (x y : ℝ), (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) → y = -1) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 := by
  sorry

#check common_chord_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_properties_l681_68120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_problem_l681_68125

theorem tan_double_angle_problem (α : Real) 
  (h1 : Real.sin (π - α) = 3 * Real.sqrt 10 / 10)
  (h2 : 0 < α ∧ α < π / 2) :
  Real.tan (2 * α) = -3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_problem_l681_68125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_imply_nm_equals_nine_l681_68153

/-- Two algebraic terms are considered "like terms" if they have the same variables raised to the same powers. -/
def are_like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (a b : ℕ), ∃ (c : ℚ), term1 a b = c * term2 a b

/-- The first term in our problem -/
def term1 (n : ℕ) (a b : ℕ) : ℚ := (1/5 : ℚ) * a^(n+1) * b^n

/-- The second term in our problem -/
def term2 (m : ℕ) (a b : ℕ) : ℚ := (-3 : ℚ) * a^(2*m) * b^3

theorem like_terms_imply_nm_equals_nine (n m : ℕ) :
  are_like_terms (term1 n) (term2 m) → n^m = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_imply_nm_equals_nine_l681_68153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosh_sinh_identity_f_inequality_equiv_l681_68183

-- Define cosh and sinh as noncomputable
noncomputable def cosh (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2
noncomputable def sinh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

-- Define f as noncomputable
noncomputable def f (x : ℝ) : ℝ := sinh x * cosh x

-- Theorem statements
theorem cosh_sinh_identity : ∀ x : ℝ, (cosh x)^2 - (sinh x)^2 = 1 := by sorry

theorem f_inequality_equiv : ∀ x : ℝ, f x < (1 - Real.exp 4) / (4 * Real.exp 2) ↔ x < -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosh_sinh_identity_f_inequality_equiv_l681_68183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_gender_probability_l681_68159

theorem same_gender_probability (school_A_male school_A_female school_B_male school_B_female : ℕ) 
  (h1 : school_A_male = 2)
  (h2 : school_A_female = 1)
  (h3 : school_B_male = 1)
  (h4 : school_B_female = 2) :
  (school_A_male * school_B_male + school_A_female * school_B_female : ℚ) / 
  ((school_A_male + school_A_female) * (school_B_male + school_B_female)) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_gender_probability_l681_68159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_three_less_than_reciprocal_l681_68171

theorem negative_three_less_than_reciprocal : 
  (-3 : ℚ) < (-3 : ℚ)⁻¹ ∧ (-3 : ℚ) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_three_less_than_reciprocal_l681_68171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_stationary_points_order_l681_68106

-- Define the concept of "new stationary point"
def new_stationary_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = (deriv^[2] f) x₀

-- Define the functions
def g (x : ℝ) : ℝ := 2 * x

noncomputable def h (x : ℝ) : ℝ := Real.log x

def φ (x : ℝ) : ℝ := x^3

-- State the theorem
theorem new_stationary_points_order :
  ∃ (a b c : ℝ),
    (new_stationary_point g a) ∧
    (new_stationary_point h b) ∧
    (new_stationary_point φ c) ∧
    (c > b) ∧ (b > a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_stationary_points_order_l681_68106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l681_68149

/-- Theorem: Eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (F₁ F₂ P O : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}) →
  F₁.1 < 0 →
  F₂.1 > 0 →
  P.1 > 0 →
  O = (0, 0) →
  (P - O + (F₂ - O)) • (F₂ - P) = 0 →
  2 * ‖P - F₁‖ = 3 * ‖P - F₂‖ →
  let c := ‖F₂ - O‖
  let e := c / a
  e = Real.sqrt 13 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l681_68149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_l681_68177

noncomputable section

/-- Curve C₁ in parametric form -/
def C₁ (t : ℝ) : ℝ × ℝ :=
  (t + 1/t, t - 1/t)

/-- Curve C₂ in parametric form -/
def C₂ (a θ : ℝ) : ℝ × ℝ :=
  (a * Real.cos θ, Real.sin θ)

/-- Foci of C₂ -/
def foci (a : ℝ) : Set (ℝ × ℝ) :=
  {(Real.sqrt (a^2 - 1), 0), (-Real.sqrt (a^2 - 1), 0)}

/-- C₁ passes through the foci of C₂ -/
def passes_through_foci (a : ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, C₁ t₁ ∈ foci a ∧ C₁ t₂ ∈ foci a ∧ t₁ ≠ t₂

theorem curve_intersection (a : ℝ) (h : a > 1) :
  passes_through_foci a → a = Real.sqrt 5 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_l681_68177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_price_decrease_l681_68150

theorem rice_price_decrease (initial_price : ℝ) (money : ℝ) :
  initial_price > 0 →
  money > 0 →
  let new_price := 0.8 * initial_price;
  let new_amount := 25;
  money = new_price * new_amount →
  let original_amount := money / initial_price;
  original_amount = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_price_decrease_l681_68150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_area_eq_four_times_perimeter_l681_68182

theorem right_triangles_area_eq_four_times_perimeter :
  ∃! n : ℕ, n > 0 ∧ 
  (∃ S : Finset (ℕ × ℕ × ℕ), S.card = n ∧
    ∀ (t : ℕ × ℕ × ℕ), t ∈ S → 
      let (a, b, c) := t
      a > 0 ∧ b > 0 ∧ c > 0 ∧
      a * a + b * b = c * c ∧
      a * b = 8 * (a + b + c) ∧
      a < b ∧ b < c) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_area_eq_four_times_perimeter_l681_68182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_radii_l681_68119

/-- Given a triangle with perimeter 2s and angles α and β, this theorem proves the formulas
    for the radius of the circumscribed circle and the radius of the inscribed circle. -/
theorem triangle_circle_radii (s α β : ℝ) (h1 : 0 < α) (h2 : 0 < β) (h3 : α + β < π) :
  let γ : ℝ := π - α - β
  ∃ (r ρ : ℝ),
    r = s / (Real.sin α + Real.sin β + Real.sin γ) ∧
    ρ = s / (Real.tan (α/2)⁻¹ + Real.tan (β/2)⁻¹ + Real.tan (γ/2)⁻¹) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_radii_l681_68119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_satisfying_inequality_l681_68164

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℕ, x > 0 → (x : ℝ)^4 / (x : ℝ)^2 < 18 → x ≤ 4 ∧ 4^4 / 4^2 < 18 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_satisfying_inequality_l681_68164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_has_no_4_level_ideal_interval_l681_68123

open Real Set

-- Define the concept of k-level "ideal interval"
def is_ideal_interval (f : ℝ → ℝ) (k : ℝ) (a b : ℝ) : Prop :=
  a < b ∧
  (∀ x, x ∈ Icc a b → f x ∈ Icc (k * a) (k * b)) ∧
  (∀ x y, x ∈ Icc a b → y ∈ Icc a b → x < y → f x < f y)

-- State the theorem
theorem tan_has_no_4_level_ideal_interval :
  ¬ ∃ a b : ℝ, -π/2 < a ∧ b < π/2 ∧ is_ideal_interval tan 4 a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_has_no_4_level_ideal_interval_l681_68123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l681_68180

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 + (Real.sin x) / (1 + Real.cos x)

-- Define the sequence of positive zeros
noncomputable def positive_zeros : ℕ → ℝ
| 0 => 4 * Real.pi / 3
| n + 1 => positive_zeros n + 2 * Real.pi

-- Define the third positive zero
noncomputable def x₃ : ℝ := positive_zeros 2

-- Define α
noncomputable def α : ℝ := 12 * x₃ + 201

-- Theorem statement
theorem sin_alpha_value : Real.sin α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l681_68180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l681_68151

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 7768) :
  Int.gcd (4 * b ^ 2 + 37 * b + 72) (3 * b + 8) = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l681_68151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_same_acquaintance_count_exists_valid_16_person_meeting_l681_68174

/-- Represents a person in the meeting -/
structure Person where
  id : Nat

/-- Represents the meeting with n people -/
structure Meeting where
  n : Nat
  people : Finset Person
  acquaintance : Person → Person → Bool
  h_n_gt_one : n > 1
  h_people_count : people.card = n
  h_mutual_acquaintances : ∀ p q : Person, p ∈ people → q ∈ people → p ≠ q →
    (people.filter (λ r => acquaintance p r ∧ acquaintance q r)).card = 2

/-- The number of acquaintances for a person -/
def acquaintance_count (m : Meeting) (p : Person) : Nat :=
  (m.people.filter (λ q => m.acquaintance p q)).card

theorem all_same_acquaintance_count (m : Meeting) :
  ∀ p q : Person, p ∈ m.people → q ∈ m.people → acquaintance_count m p = acquaintance_count m q :=
by sorry

theorem exists_valid_16_person_meeting :
  ∃ m : Meeting, m.n = 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_same_acquaintance_count_exists_valid_16_person_meeting_l681_68174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l681_68111

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 / 4 = 1

-- Define the slope of the line
def line_slope : ℝ := 2

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (1, 0)

-- Define the line passing through the right focus
def line (x y : ℝ) : Prop := y = line_slope * (x - right_focus.1)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2

-- Define the length of AB
noncomputable def length_AB (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The theorem to prove
theorem chord_length (A B : ℝ × ℝ) :
  intersection_points A B → length_AB A B = 5 * Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l681_68111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_max_value_of_fraction_existence_of_max_values_l681_68173

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := |1 + 2*x| / |x| + |1 - 2*x| / |x|

-- Theorem for the minimum value of f(x)
theorem min_value_of_f : 
  ∃ (m : ℝ), (∀ x : ℝ, x ≠ 0 → f x ≥ m) ∧ (∃ x : ℝ, x ≠ 0 ∧ f x = m) ∧ m = 4 :=
by sorry

-- Theorem for the maximum value of ab/(a+4b)
theorem max_value_of_fraction :
  ∀ a b : ℝ, a > 0 → b > 0 → a + b = 4 →
  a * b / (a + 4 * b) ≤ 4 / 9 :=
by sorry

-- Theorem for the existence of values achieving the maximum
theorem existence_of_max_values :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 4 ∧ a * b / (a + 4 * b) = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_max_value_of_fraction_existence_of_max_values_l681_68173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_abc_properties_l681_68192

/-- Represents a right triangle ABC with given side lengths -/
structure RightTriangle where
  ac : ℝ
  bc : ℝ
  h_right_angle : ac > 0 ∧ bc > 0 ∧ ac < bc

/-- Calculates the length of side AB in a right triangle -/
noncomputable def length_ab (t : RightTriangle) : ℝ :=
  Real.sqrt (t.bc ^ 2 - t.ac ^ 2)

/-- Calculates the area of a right triangle -/
noncomputable def area (t : RightTriangle) : ℝ :=
  (1 / 2) * t.ac * length_ab t

/-- Main theorem about the specific right triangle -/
theorem right_triangle_abc_properties :
  let t : RightTriangle := ⟨5, 13, by sorry⟩
  length_ab t = 12 ∧ area t = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_abc_properties_l681_68192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l681_68194

/-- The function f(x) = 2x + √(1 - x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * x + Real.sqrt (1 - x)

/-- The domain of f(x) -/
def dom : Set ℝ := {x : ℝ | x ≤ 1}

theorem f_range :
  Set.range f = Set.Iic (17/8) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l681_68194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisor_four_consecutive_integers_l681_68121

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
    12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) ∧
    ∀ m : ℕ, m > 12 →
      ∃ k : ℕ, k > 0 ∧ ¬(m ∣ (k * (k + 1) * (k + 2) * (k + 3))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisor_four_consecutive_integers_l681_68121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l681_68134

/-- Given points A, E, F, and P in 2D space, with vectors EA and EF, prove that λ + μ = 2/3 
    when EP = λEA + μEF. -/
theorem vector_equation_solution (A E F P : ℝ × ℝ) (l m : ℝ) : 
  A = (3, 0) →
  P = (2, 0) →
  (E.1 + 2, E.2 + 1) = A →
  (E.1 + 1, E.2 + 2) = F →
  (P.1 - E.1, P.2 - E.2) = l • (2, 1) + m • (1, 2) →
  l + m = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l681_68134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_sum_l681_68129

/-- Given the expansion of (1+ax+by)^n where a and b are positive integer constants,
    if the sum of the coefficients of terms not containing x is 243, then n = 5 -/
theorem expansion_coefficient_sum (a b : ℕ+) (n : ℕ) : (1 + b.val : ℕ) ^ n = 243 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_sum_l681_68129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_sqrt_l681_68141

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ (y : ℝ) (n : ℕ), x = y * Real.sqrt n → y = 1 ∧ n = Int.floor (x^2)

theorem simplest_sqrt :
  is_simplest_sqrt (Real.sqrt 11) ∧
  ¬ is_simplest_sqrt (Real.sqrt (1/2)) ∧
  ¬ is_simplest_sqrt (Real.sqrt 27) ∧
  ¬ is_simplest_sqrt (Real.sqrt 0.3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_sqrt_l681_68141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumradius_l681_68131

-- Define the radius of the original circle
def originalRadius : ℝ := 4

-- Define the central angle θ
variable (θ : ℝ)

-- Condition that θ is an obtuse angle
def isObtuse (θ : ℝ) : Prop := Real.pi / 2 < θ ∧ θ < Real.pi

-- Define the radius of the circumscribed circle
noncomputable def circumRadius (θ : ℝ) : ℝ := 2 / Real.sin (θ / 2)

-- Theorem statement
theorem sector_circumradius (θ : ℝ) (h : isObtuse θ) : 
  circumRadius θ = 2 * (1 / Real.sin (θ / 2)) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumradius_l681_68131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l681_68113

noncomputable def f (x : ℝ) : ℝ := (x - 1)^0 / Real.sqrt (|x| + x)

def domain_of_f : Set ℝ :=
  Set.Ioo 0 1 ∪ Set.Ioi 1

theorem f_domain : 
  ∀ x : ℝ, f x ≠ 0 ↔ x ∈ domain_of_f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l681_68113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_ones_in_twelve_dice_l681_68145

theorem probability_two_ones_in_twelve_dice :
  let n : ℕ := 12  -- Total number of dice
  let k : ℕ := 2   -- Number of dice showing 1
  let p : ℚ := 1/6 -- Probability of a single die showing 1
  Nat.choose n k * p^k * (1-p)^(n-k) = (Nat.choose 12 2 : ℚ) * (1/6)^2 * (5/6)^10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_ones_in_twelve_dice_l681_68145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l681_68108

noncomputable def f (x : ℝ) := 2 * Real.cos (x - Real.pi / 6)

theorem problem_solution :
  (f Real.pi = -Real.sqrt 3) ∧
  (∀ α : ℝ, α ∈ Set.Ioo (-Real.pi/2) 0 → f (α + 2*Real.pi/3) = 6/5 → f (2*α) = (7*Real.sqrt 3 - 24) / 25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l681_68108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l681_68143

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x^2 + 6)

-- Part 1
theorem part_one (k : ℝ) (h : Set.Iio (-3) ∪ Set.Ioi (-2) = {x | f x > k}) : k = -2/5 := by
  sorry

-- Part 2
theorem part_two (t : ℝ) (h : ∀ x > 0, f x ≤ t) : t ≥ Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l681_68143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_config_l681_68176

/-- Represents a configuration of numbers in the table -/
def TableConfig := Fin 10 → Fin 4 × Fin 3

/-- Checks if a number is in the long column -/
def isInLongColumn (pos : Fin 4 × Fin 3) : Bool :=
  pos.1 = 3 ∨ (pos.1 = 2 ∧ pos.2 = 2)

/-- Calculates the sum of a row or the long column -/
def sumPart (config : TableConfig) (isPartOf : Fin 4 × Fin 3 → Bool) : Nat :=
  (Finset.sum (Finset.range 10) fun i => if isPartOf (config i) then i + 1 else 0)

/-- Checks if the configuration is valid -/
def isValidConfig (config : TableConfig) : Prop :=
  (∀ i j : Fin 10, i ≠ j → config i ≠ config j) ∧
  (sumPart config (fun pos => pos.1 = 0) =
   sumPart config (fun pos => pos.1 = 1) ∧
   sumPart config (fun pos => pos.1 = 0) =
   sumPart config isInLongColumn)

theorem exists_valid_config : ∃ config : TableConfig, isValidConfig config := by
  sorry

#check exists_valid_config

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_config_l681_68176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_converges_l681_68126

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => λ x => x
| n + 1 => λ x => 2 * x^(n + 2) - x^(n + 1) + (1/2) * ∫ t in Set.Icc 0 1, f n t

theorem limit_f_converges :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |f n (1 + 1 / (2 * n)) - Real.exp (1/2)| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_converges_l681_68126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_constant_slope_product_l681_68189

-- Define the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the constant product of slopes
def constant_slope_product (m : ℝ) (A B P : Point) : Prop :=
  let slope_PA := (P.y - A.y) / (P.x - A.x)
  let slope_PB := (P.y - B.y) / (P.x - B.x)
  slope_PA * slope_PB = m

-- Define the possible trajectories
inductive Trajectory
  | Ellipse
  | Hyperbola
  | Parabola
  | Circle
  | StraightLine

-- Define the set of possible trajectories for a point P
def possible_trajectories (P : Point) : Set Trajectory :=
  sorry

-- Theorem statement
theorem trajectory_of_constant_slope_product 
  (A B : Point) (m : ℝ) :
  ∃ (P : Point), constant_slope_product m A B P →
    (Trajectory.Ellipse ∈ possible_trajectories P ∧
     Trajectory.Hyperbola ∈ possible_trajectories P ∧
     Trajectory.Circle ∈ possible_trajectories P ∧
     Trajectory.StraightLine ∈ possible_trajectories P ∧
     Trajectory.Parabola ∉ possible_trajectories P) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_constant_slope_product_l681_68189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l681_68156

noncomputable def f (x : ℝ) := 3 * x + 1 + 9 / (3 * x - 2)

theorem f_max_value :
  (∀ x < 2/3, f x ≤ -3) ∧ (∃ x < 2/3, f x = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l681_68156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_m_value_l681_68104

/-- Linear regression data point -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Calculate the mean of a list of real numbers -/
noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

/-- Calculate ŷ given x and a linear regression equation -/
def predict (x : ℝ) (eq : LinearRegression) : ℝ :=
  eq.slope * x + eq.intercept

theorem linear_regression_m_value
  (data : List DataPoint)
  (eq : LinearRegression)
  (h1 : data.length = 4)
  (h2 : data[0].x = 1 ∧ data[1].x = 2 ∧ data[2].x = 3 ∧ data[3].x = 4)
  (h3 : data[1].y = 3.2 ∧ data[2].y = 4.8 ∧ data[3].y = 7.5)
  (h4 : eq.slope = 2.1 ∧ eq.intercept = -0.25)
  : data[0].y = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_m_value_l681_68104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_growth_bound_l681_68146

/-- Given a function f: ℝ → ℝ where f'(x) < f(x) for all x,
    prove that f(a) < exp(a) * f(0) for any a > 0 -/
theorem function_growth_bound (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : ∀ x, deriv f x < f x) (a : ℝ) (ha : a > 0) : 
    f a < Real.exp a * f 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_growth_bound_l681_68146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_property_l681_68199

open Real Set

theorem sine_function_property (ω : ℝ) (f : ℝ → ℝ) : 
  (ω > 0) →
  (f = λ x ↦ sin (ω * x + π / 4)) →
  (∃ (M : ℝ), ∀ x ∈ Ioo (π / 12) (π / 3), f x ≤ M) →
  (∀ (m : ℝ), ∃ x ∈ Ioo (π / 12) (π / 3), f x < m) →
  (3 / 4 < ω ∧ ω < 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_property_l681_68199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_perimeter_of_triangles_l681_68142

-- Define the triangles
def triangle1_leg1 : ℝ := 15
def triangle1_leg2 : ℝ := 20
def triangle2_leg1 : ℝ := 9
def triangle2_leg2 : ℝ := 12

-- Define the function to calculate the hypotenuse
noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

-- Define the function to calculate the perimeter of a right triangle
noncomputable def triangle_perimeter (a b : ℝ) : ℝ := a + b + hypotenuse a b

-- Theorem statement
theorem combined_perimeter_of_triangles :
  triangle_perimeter triangle1_leg1 triangle1_leg2 +
  triangle_perimeter triangle2_leg1 triangle2_leg2 -
  hypotenuse triangle1_leg1 triangle1_leg2 = 106 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_perimeter_of_triangles_l681_68142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_c_value_l681_68107

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- A configuration of 99 points satisfying the distance condition -/
structure Configuration (c : ℝ) where
  points : Fin 99 → Point3D
  distance_condition : ∀ i : Fin 99, ∃ j k : Fin 99,
    i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    ∀ l : Fin 99, l ≠ i → l ≠ j → l ≠ k →
      distance (points i) (points j) ≤ distance (points i) (points l) ∧
      distance (points i) (points k) ≤ distance (points i) (points l) ∧
      distance (points i) (points k) ≥ c * distance (points i) (points j)

/-- The maximum possible value of c for any valid configuration -/
theorem max_c_value {c : ℝ} (config : Configuration c) :
  c ≤ (1 + Real.sqrt 5) / 2 := by
  sorry

#check max_c_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_c_value_l681_68107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_of_point_on_parabola_l681_68140

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to the directrix of a parabola -/
noncomputable def distance_to_directrix (para : Parabola) (pt : Point) : ℝ :=
  pt.x + para.p / 2

theorem distance_to_directrix_of_point_on_parabola :
  ∀ (para : Parabola) (A : Point),
    A.x = 1 ∧ A.y = Real.sqrt 5 →
    para.eq A.x A.y →
    distance_to_directrix para A = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_of_point_on_parabola_l681_68140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daisy_surname_is_mak_l681_68122

-- Define the set of ladies
inductive Lady
| Ong
| Lim
| Mak
| Nai
| Poh
| Ellie
| Cindy
| Amy
| Beatrice
| Daisy

-- Define the circular seating arrangement
def SeatingArrangement := Fin 5 → Lady

-- Define the condition for a valid seating arrangement
def IsValidSeating (arrangement : SeatingArrangement) : Prop :=
  ∃ i, arrangement i = Lady.Ong ∧ 
       arrangement (i + 1) = Lady.Mak ∧
       arrangement (i - 1) = Lady.Lim ∧
  ∃ j, arrangement j = Lady.Ellie ∧
       arrangement (j + 1) = Lady.Nai ∧
       arrangement (j - 1) = Lady.Cindy ∧
  ∃ k, arrangement k = Lady.Lim ∧
       arrangement (k + 1) = Lady.Amy ∧
       arrangement (k - 1) = Lady.Ellie ∧
  ∃ l, arrangement l = Lady.Beatrice ∧
       arrangement (l + 1) = Lady.Mak ∧
       arrangement (l - 1) = Lady.Poh

-- Theorem: Given the seating conditions, Daisy's surname must be Mak
theorem daisy_surname_is_mak (arrangement : SeatingArrangement) 
  (h : IsValidSeating arrangement) : 
  ∃ i, arrangement i = Lady.Daisy ∧ arrangement (i + 1) = Lady.Mak :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_daisy_surname_is_mak_l681_68122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_chord_properties_l681_68105

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y : ℝ) := x^2 / 3 + y^2 = 1

/-- Eccentricity of the ellipse -/
noncomputable def eccentricity : ℝ := Real.sqrt 6 / 3

/-- Distance from minor axis endpoint to right focus -/
noncomputable def minor_to_focus : ℝ := Real.sqrt 3

/-- Slope of line l -/
def line_slope : ℝ := 1

/-- Left focus x-coordinate -/
noncomputable def left_focus_x : ℝ := -Real.sqrt 2

/-- Theorem stating the properties of the ellipse and the chord AB -/
theorem ellipse_and_chord_properties :
  (∀ x y, ellipse x y ↔ x^2 / 3 + y^2 = 1) ∧
  ∃ A B : ℝ × ℝ,
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    A.2 - B.2 = line_slope * (A.1 - B.1) ∧
    A.2 - left_focus_x = line_slope * (A.1 - left_focus_x) ∧
    B.2 - left_focus_x = line_slope * (B.1 - left_focus_x) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_chord_properties_l681_68105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_expression_l681_68190

noncomputable def cube_root_unity : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

theorem distinct_values_of_expression : 
  ∃ (S : Finset ℂ), (Finset.card S = 6) ∧ 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → 
    (((cube_root_unity^8 + 1)^n) ∈ S)) ∧
  (∀ z : ℂ, z ∈ S → 
    ∃ n : ℕ, 1 ≤ n ∧ n ≤ 100 ∧ 
    z = ((cube_root_unity^8 + 1)^n)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_expression_l681_68190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l681_68154

theorem cos_minus_sin_value (θ α : Real) (k : Real) :
  (∃ x y : Real, x = Real.tan θ ∧ y = (Real.tan θ)⁻¹ ∧ 2 * x^2 - 2 * k * x = 3 - k^2 ∧
                 2 * y^2 - 2 * k * y = 3 - k^2) →
  α < θ ∧ θ < 5 * Real.pi / 4 →
  Real.cos θ - Real.sin θ = -Real.sqrt ((5 - 2 * Real.sqrt 5) / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l681_68154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_fifth_power_decomposition_l681_68160

def B : Matrix (Fin 2) (Fin 2) ℚ := !![2, 3; 4, -1]

theorem B_fifth_power_decomposition :
  B^5 = 625 • B + 1224 • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_fifth_power_decomposition_l681_68160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_sequence_l681_68198

def sequence (k : ℕ) : ℕ → ℕ
  | 0 => 1  -- a_1 = 1
  | 1 => 2020  -- a_2 = 2020 (derived from a_2018 = 2020 due to periodicity)
  | n + 2 => k * sequence k (n + 1) / sequence k n

theorem smallest_k_for_sequence : 
  ∃ k : ℕ, k > 0 ∧ 
  (∀ n : ℕ, sequence k n > 0) ∧
  (∀ n : ℕ, n ≥ 2 → sequence k (n + 1) = k * sequence k n / sequence k (n - 1)) ∧
  sequence k 2017 = 2020 ∧
  (∀ k' : ℕ, k' < k → 
    ¬(∀ n : ℕ, sequence k' n > 0) ∨ 
    ¬(∀ n : ℕ, n ≥ 2 → sequence k' (n + 1) = k' * sequence k' n / sequence k' (n - 1)) ∨
    sequence k' 2017 ≠ 2020) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_sequence_l681_68198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_values_l681_68139

/-- Given a triangle ABC with side lengths a and c, and angle A, 
    prove the values of sin C and cos 2C. -/
theorem triangle_trig_values (a c : ℝ) (A : ℝ) :
  c = 2 → a = Real.sqrt 3 → A = π / 6 →
  ∃ (C : ℝ),
    (Real.sin C = Real.sqrt 3 / 3) ∧
    (Real.cos (2 * C) = 1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_values_l681_68139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_combination_count_l681_68117

/-- The number of valid flower combinations -/
def flower_combinations : ℕ := 17

/-- Represents a combination of flowers -/
structure FlowerCombination where
  roses : ℕ
  carnations : ℕ
  lilies : ℕ

/-- Checks if a flower combination is valid -/
def is_valid_combination (fc : FlowerCombination) : Prop :=
  fc.roses > 0 ∧ fc.carnations > 0 ∧ fc.lilies > 0 ∧
  4 * fc.roses + 3 * fc.carnations + 5 * fc.lilies = 120

/-- The set of all valid flower combinations -/
def valid_combinations : Set FlowerCombination :=
  {fc : FlowerCombination | is_valid_combination fc}

/-- Assume the set of valid combinations is finite -/
instance : Fintype valid_combinations := sorry

theorem flower_combination_count :
  Fintype.card valid_combinations = flower_combinations := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_combination_count_l681_68117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_container_relationship_l681_68167

/-- The relationship between milk quantities in containers after transfer -/
theorem milk_container_relationship (A : ℝ) (h : A > 0) :
  (0.375 * A + 158) / (0.625 * A - 158) =
  (0.375 * A + 158) / (0.625 * A - 158) := by
  -- Define variables for clarity
  let initial_B := 0.375 * A
  let initial_C := 0.625 * A
  let transfer := 158
  let final_B := initial_B + transfer
  let final_C := initial_C - transfer
  
  -- The proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_container_relationship_l681_68167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jorges_acreage_l681_68128

/-- Represents the yield of corn in bushels per acre for good soil -/
noncomputable def good_soil_yield : ℝ := 400

/-- Represents the yield of corn in bushels per acre for clay-rich soil -/
noncomputable def clay_rich_soil_yield : ℝ := good_soil_yield / 2

/-- Represents the total yield of corn in bushels from Jorge's land -/
noncomputable def total_yield : ℝ := 20000

/-- Represents the fraction of Jorge's land that is clay-rich soil -/
noncomputable def clay_rich_fraction : ℝ := 1 / 3

/-- Represents the fraction of Jorge's land that is good soil -/
noncomputable def good_soil_fraction : ℝ := 1 - clay_rich_fraction

/-- Theorem stating that Jorge's total acreage is 60 acres -/
theorem jorges_acreage :
  ∃ (A : ℝ), A > 0 ∧ 
  good_soil_yield * good_soil_fraction * A + 
  clay_rich_soil_yield * clay_rich_fraction * A = total_yield ∧
  A = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jorges_acreage_l681_68128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_47_l681_68184

theorem repeating_decimal_47 : ∃ x : ℚ, x = 47 / 99 ∧ x = ∑' n, 47 * (1 / 100) ^ (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_47_l681_68184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l681_68116

noncomputable section

/-- The volume of the solid formed by rotating the region bounded by y = 2x - x^2, y = -x + 2, and x = 0 about the x-axis -/
def rotationVolume : ℝ := (9 / 5) * Real.pi

/-- The upper bounding function -/
def f (x : ℝ) : ℝ := 2 * x - x^2

/-- The lower bounding function -/
def g (x : ℝ) : ℝ := -x + 2

/-- The left boundary of the region -/
def leftBoundary : ℝ := 0

/-- The right boundary of the region (intersection of f and g) -/
def rightBoundary : ℝ := 1

theorem volume_of_rotation : 
  ∫ x in leftBoundary..rightBoundary, Real.pi * ((f x)^2 - (g x)^2) = rotationVolume := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l681_68116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_log_sum_l681_68124

/-- For a 3-digit number n, if log₃n + log₉n is a whole number and log₉n is a whole number, then n = 9 -/
theorem three_digit_log_sum (n : ℕ) : 
  100 ≤ n ∧ n ≤ 999 →  -- n is a 3-digit number
  ∃ (k : ℤ), (Real.log n / Real.log 3) + (Real.log n / Real.log 9) = k →  -- log₃n + log₉n is a whole number
  ∃ (m : ℤ), Real.log n / Real.log 9 = m →  -- log₉n is a whole number
  n = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_log_sum_l681_68124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_focus_product_l681_68186

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a parabola with equation y² = 4px -/
structure Parabola where
  p : ℝ
  h_pos_p : p > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2) / h.a

/-- The focus of a parabola y² = 4px -/
def parabola_focus (p : Parabola) : ℝ × ℝ := (p.p, 0)

/-- Theorem stating that for a hyperbola with eccentricity 2 and one focus coinciding
    with the focus of the parabola y² = 4x, the product ab equals √3/4 -/
theorem hyperbola_parabola_focus_product (h : Hyperbola) (p : Parabola) :
  eccentricity h = 2 →
  p.p = 1 →
  parabola_focus p = (1, 0) →
  h.a * h.b = Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_focus_product_l681_68186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_bag_weight_l681_68130

theorem empty_bag_weight 
  (total_weight : ℝ) 
  (weight_after_removal : ℝ) 
  (empty_bag_weight : ℝ)
  (h1 : total_weight = 3.4)
  (h2 : weight_after_removal = 2.98)
  (h3 : weight_after_removal = total_weight - 0.2 * (total_weight - empty_bag_weight)) :
  empty_bag_weight = 1.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_bag_weight_l681_68130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l681_68181

noncomputable def f (x : ℝ) := Real.sin (Real.pi / 2 - x) * Real.sin x - Real.sqrt 3 * (Real.cos x) ^ 2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (M : ℝ), M = -Real.sqrt 3 / 2 ∧ ∀ (x : ℝ), f x ≤ M) ∧
  (∀ (x y : ℝ), 5 * Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ 2 * Real.pi / 3 → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l681_68181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_one_is_negative_three_l681_68133

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x ≤ 0 then 2 * x^2 - x
  else -(2 * (-x)^2 - (-x))

-- State the theorem
theorem f_at_one_is_negative_three :
  (∀ x, f (-x) = -f x) → -- f is odd
  f 1 = -3 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_one_is_negative_three_l681_68133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l681_68168

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi/4 * x - Real.pi/6) - 2 * (Real.cos (Real.pi/8 * x))^2 + 1

noncomputable def g (x : ℝ) : ℝ := f (2 - x)

theorem max_value_of_g :
  ∃ (y : ℝ), y ∈ Set.Icc 0 (4/3) ∧ g y = Real.sqrt 3 / 2 ∧ ∀ z ∈ Set.Icc 0 (4/3), g z ≤ g y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l681_68168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l681_68191

/-- The sequence a_n defined by the given recurrence relation -/
def a : ℕ → ℕ
  | 0 => 1  -- Add a case for 0 to avoid missing case error
  | 1 => 2
  | n + 2 => (a (n + 1))^2 - (n + 1) * (a (n + 1)) + 1

/-- The theorem stating that a_n = n + 1 for all positive integers n -/
theorem a_formula (n : ℕ) (h : n > 0) : a n = n + 1 := by
  induction n with
  | zero => contradiction
  | succ n ih =>
    cases n with
    | zero =>
      simp [a]
    | succ n =>
      simp [a]
      sorry  -- The actual proof would go here

#eval a 1  -- Should output 2
#eval a 2  -- Should output 3
#eval a 3  -- Should output 4
#eval a 4  -- Should output 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l681_68191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_speed_l681_68109

/-- Proves that the downstream speed is 45 kmph given the upstream speed,
    still water speed, and the relationship between these speeds. -/
theorem downstream_speed
  (upstream_speed : ℝ)
  (still_water_speed : ℝ)
  (downstream_speed : ℝ)
  (h1 : upstream_speed = 25)
  (h2 : still_water_speed = 35)
  (h3 : still_water_speed = (upstream_speed + downstream_speed) / 2) :
  downstream_speed = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_speed_l681_68109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_change_percentage_difference_l681_68172

/-- Represents the percentage of students who answered "Yes" at the beginning and end of the year --/
structure YesPercentages where
  initial : ℝ
  final : ℝ

/-- Represents the percentage of students who changed their answer --/
def changed_percentage (yp : YesPercentages) (y : ℝ) : Prop :=
  0 ≤ y ∧ y ≤ 100 ∧ 
    ∃ (changed_to_yes changed_to_no : ℝ),
      0 ≤ changed_to_yes ∧ changed_to_yes ≤ yp.initial ∧
      0 ≤ changed_to_no ∧ changed_to_no ≤ 100 - yp.initial ∧
      changed_to_yes - changed_to_no = yp.final - yp.initial ∧
      y = changed_to_yes + changed_to_no

/-- The main theorem stating the difference between max and min possible change percentages --/
theorem change_percentage_difference (yp : YesPercentages) 
    (h1 : yp.initial = 60)
    (h2 : yp.final = 80) :
    ∃ (y_min y_max : ℝ),
      changed_percentage yp y_min ∧
      changed_percentage yp y_max ∧
      (∀ y, changed_percentage yp y → y_min ≤ y ∧ y ≤ y_max) ∧
      y_max - y_min = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_change_percentage_difference_l681_68172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l681_68114

/-- The coefficient of x^3 in the expansion of (x - 2/x)^7 is 84 -/
theorem coefficient_x_cubed_in_expansion : ∃ (c : ℤ), c = 84 ∧ 
  (fun x : ℝ => (x - 2/x)^7) = fun x : ℝ => c * x^3 + ((x - 2/x)^7 - c * x^3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l681_68114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l681_68165

theorem largest_expression (α : Real) (h1 : α = 48 * Real.pi / 180) 
  (h2 : 0 < Real.cos α) (h3 : Real.cos α < 1) 
  (h4 : 0 < Real.sin α) (h5 : Real.sin α < 1) : 
  Real.tan α + (1 / Real.tan α) = max 
    (Real.tan α + (1 / Real.tan α)) 
    (max (Real.sin α + Real.cos α) 
      (max (Real.tan α + Real.cos α) ((1 / Real.tan α) + Real.sin α))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l681_68165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_midpoint_locus_l681_68152

/-- Given a hyperbola x^2 - y^2/4 = 1 with center O, and two perpendicular rays from O
    intersecting the hyperbola at points A and B, the locus of the midpoint P of chord AB
    satisfies the equation: 3(4x^2 - y^2)^2 = 4(16x^2 + y^2) -/
theorem hyperbola_midpoint_locus :
  ∀ (A B P : ℝ × ℝ),
  (A.1^2 - A.2^2/4 = 1) →  -- Point A lies on the hyperbola
  (B.1^2 - B.2^2/4 = 1) →  -- Point B lies on the hyperbola
  (A.1 * B.1 + A.2 * B.2 = 0) →  -- OA ⊥ OB
  (P = ((A.1 + B.1)/2, (A.2 + B.2)/2)) →  -- P is the midpoint of AB
  3 * (4 * P.1^2 - P.2^2)^2 = 4 * (16 * P.1^2 + P.2^2) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_midpoint_locus_l681_68152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_moments_first_three_central_moments_l681_68132

/-- Probability density function for the given random variable X -/
noncomputable def p (x : ℝ) : ℝ :=
  if x ≤ 0 then 0 else Real.exp (-x)

/-- The nth moment of the random variable X -/
noncomputable def moment (n : ℕ) : ℝ :=
  ∫ x in Set.Ioi 0, x^n * p x

/-- Theorem stating the values of the first three moments -/
theorem first_three_moments :
  moment 1 = 1 ∧ moment 2 = 2 ∧ moment 3 = 6 := by
  sorry

/-- The nth central moment of the random variable X -/
noncomputable def central_moment (n : ℕ) : ℝ :=
  ∫ x in Set.Ioi 0, (x - moment 1)^n * p x

/-- Theorem stating the values of the first three central moments -/
theorem first_three_central_moments :
  central_moment 1 = 0 ∧ central_moment 2 = 1 ∧ central_moment 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_moments_first_three_central_moments_l681_68132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equality_l681_68135

theorem infinite_sum_equality (x : ℝ) (h : x > 1) : 
  (∑' n : ℕ, 1 / (x^(3^n) - (1/x)^(3^n))) = 1 / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equality_l681_68135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_A_completes_in_40_days_l681_68138

/-- Represents the total amount of work to be done -/
noncomputable def TotalWork : ℝ := 1

/-- The rate at which worker B completes the work -/
noncomputable def RateB : ℝ := TotalWork / 40

/-- The rate at which worker C completes the work -/
noncomputable def RateC : ℝ := TotalWork / 20

/-- The number of days A worked -/
def DaysA : ℝ := 10

/-- The number of days B worked -/
def DaysB : ℝ := 10

/-- The number of days C worked -/
def DaysC : ℝ := 10

/-- Theorem stating that worker A can complete the work in 40 days -/
theorem worker_A_completes_in_40_days :
  ∃ (rateA : ℝ),
    rateA * DaysA + RateB * DaysB + RateC * DaysC = TotalWork ∧
    rateA * 40 = TotalWork := by
  sorry

#check worker_A_completes_in_40_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_A_completes_in_40_days_l681_68138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_theta_l681_68163

noncomputable def f (x θ : ℝ) : ℝ := Real.sin (2 * x + θ) + Real.sqrt 3 * Real.cos (2 * x + θ)

theorem odd_decreasing_function_theta (θ : ℝ) :
  (∀ x, f x θ = -f (-x) θ) →  -- f is an odd function
  (∀ x y, -π/4 ≤ x ∧ x < y ∧ y ≤ 0 → f y θ < f x θ) →  -- f is decreasing on [-π/4, 0]
  θ = 2*π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_theta_l681_68163
