import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_trinomial_equation_roots_l17_1708

/-- Given three quadratic trinomials, prove that the equation involving their absolute values has exactly 8 distinct real roots -/
theorem quadratic_trinomial_equation_roots 
  (p₁ q₁ p₂ q₂ p₃ q₃ : ℝ) : 
  ∃! (S : Finset ℝ), S.card = 8 ∧ 
    ∀ x ∈ S, |x^2 + p₁ * x + q₁| + |x^2 + p₂ * x + q₂| = |x^2 + p₃ * x + q₃| :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_trinomial_equation_roots_l17_1708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_mean_median_l17_1700

/-- An arithmetic sequence of 20 terms with given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), d ≠ 0 ∧
  (∀ n, a (n + 1) = a n + d) ∧
  a 3 = 8 ∧
  (a 1 + a 2 + a 3 + a 4 = 28)

/-- The mean of the sequence -/
noncomputable def SequenceMean (a : ℕ → ℝ) : ℝ := (a 1 + a 20) / 2

/-- The median of the sequence -/
noncomputable def SequenceMedian (a : ℕ → ℝ) : ℝ := (a 10 + a 11) / 2

/-- Theorem stating that the mean and median are both 23 -/
theorem arithmetic_sequence_mean_median :
  ∀ a : ℕ → ℝ, ArithmeticSequence a →
  SequenceMean a = 23 ∧ SequenceMedian a = 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_mean_median_l17_1700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_range_of_b_plus_c_l17_1762

/-- Triangle ABC with side lengths a, b, c corresponding to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def condition (t : Triangle) : Prop :=
  t.a * Real.cos t.B - (1/2) * t.b = (t.a^2) / t.c - (t.b * Real.sin t.B) / Real.sin t.C

theorem angle_A_value (t : Triangle) (h : condition t) : t.A = π/3 := by
  sorry

theorem range_of_b_plus_c (t : Triangle) (h1 : condition t) (h2 : t.a = Real.sqrt 3) :
  Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_range_of_b_plus_c_l17_1762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_properties_l17_1795

theorem exponential_function_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha_neq_1 : a ≠ 1) :
  let f := λ x : ℝ => b * a^x
  (f 1 = 27 ∧ f (-1) = 3) →
  (a = 3 ∧ b = 9 ∧ ∀ x : ℝ, x ≥ 1 → a^x + b^x ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_properties_l17_1795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_coloring_count_l17_1759

/-- The number of ways to color an n-sided polygon with m colors,
    where adjacent vertices must have different colors. -/
def colorings (n m : ℕ) : ℤ :=
  (m - 1) * ((m - 1)^(n - 1) + (-1)^n)

/-- Theorem stating the number of valid colorings for an n-sided polygon with m colors. -/
theorem polygon_coloring_count (n m : ℕ) (hn : n ≥ 3) (hm : m ≥ 3) :
  colorings n m = (m - 1) * ((m - 1)^(n - 1) + (-1)^n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_coloring_count_l17_1759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_toggle_theorem_l17_1712

/-- Represents the state of a light: off or on -/
inductive LightState
| Off
| On

/-- Represents a position on the 5x5 grid -/
structure Position where
  row : Fin 5
  col : Fin 5

/-- The grid of lights -/
def Grid := Fin 5 → Fin 5 → LightState

/-- Toggles a light and its neighbors in the same row and column -/
def toggle (grid : Grid) (pos : Position) : Grid :=
  sorry

/-- Checks if only one light is on in the grid -/
def onlyOneLightOn (grid : Grid) : Prop :=
  sorry

/-- The set of possible final positions for the single lit light -/
def possibleFinalPositions : Set Position :=
  { ⟨1, 1⟩, ⟨1, 3⟩, ⟨3, 1⟩, ⟨3, 3⟩, ⟨2, 2⟩ }

/-- The main theorem stating that the only possible positions for a single lit light
    are (2,2), (2,4), (4,2), (4,4), and (3,3) -/
theorem light_toggle_theorem (finalGrid : Grid) :
  (∃ initialGrid : Grid, ∀ (i j : Fin 5), initialGrid i j = LightState.Off) →
  (∃ toggleSequence : List Position, finalGrid = toggleSequence.foldl toggle initialGrid) →
  onlyOneLightOn finalGrid →
  ∃ pos ∈ possibleFinalPositions, finalGrid pos.row pos.col = LightState.On :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_toggle_theorem_l17_1712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_is_520_l17_1701

/-- Represents a circular track with two runners -/
structure CircularTrack where
  length : ℝ
  runner1_speed : ℝ
  runner2_speed : ℝ

/-- Calculates the distance run by runner2 at the first meeting -/
noncomputable def distance_runner2_first_meeting (track : CircularTrack) : ℝ :=
  track.length / 2 - 80

/-- Calculates the total distance run by runner2 at the second meeting -/
noncomputable def total_distance_runner2_second_meeting (track : CircularTrack) : ℝ :=
  track.length / 2 + 100

/-- Calculates the total distance run by runner1 at the second meeting -/
noncomputable def total_distance_runner1_second_meeting (track : CircularTrack) : ℝ :=
  track.length / 2 - 100

/-- The theorem stating the length of the track -/
theorem track_length_is_520 (track : CircularTrack) 
  (h1 : track.runner1_speed > 0)
  (h2 : track.runner2_speed > 0)
  (h3 : track.runner1_speed / track.runner2_speed = 80 / (distance_runner2_first_meeting track))
  (h4 : (total_distance_runner1_second_meeting track) / (total_distance_runner2_second_meeting track) = track.runner1_speed / track.runner2_speed) :
  track.length = 520 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_is_520_l17_1701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_l17_1756

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - x

-- State the theorem
theorem parabola_area : 
  (∫ x in (-1)..0, f x) = 5/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_l17_1756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l17_1790

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.cos t.C + (t.c - 2 * t.b) * Real.cos t.A = 0)
  (h2 : (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3)
  (h3 : t.a = 2 * Real.sqrt 3) :
  t.A = π/3 ∧ t.b + t.c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l17_1790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_values_l17_1741

/-- Represents an infinite geometric progression. -/
structure GeometricProgression where
  a : ℝ  -- First term
  r : ℝ  -- Common ratio

/-- The sum to infinity of an infinite geometric progression. -/
noncomputable def sumToInfinity (gp : GeometricProgression) : ℝ :=
  gp.a / (1 - gp.r)

/-- The sum of the first two terms of a geometric progression. -/
noncomputable def sumFirstTwo (gp : GeometricProgression) : ℝ :=
  gp.a + gp.a * gp.r

/-- Theorem stating the possible values for the first term of the geometric progression. -/
theorem first_term_values (gp : GeometricProgression) 
  (h1 : sumToInfinity gp = 8)
  (h2 : sumFirstTwo gp = 5) :
  gp.a = 2 * (4 - Real.sqrt 6) ∨ gp.a = 2 * (4 + Real.sqrt 6) := by
  sorry

#check first_term_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_values_l17_1741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_janos_and_lajos_l17_1703

/-- The distance between János and Lajos --/
noncomputable def distance_between : ℝ := 50

/-- János's walking speed in km/hour --/
noncomputable def janos_speed : ℝ := 5

/-- Lajos's walking speed in km/hour --/
noncomputable def lajos_speed : ℝ := 4.5

/-- Car speed in km/hour --/
noncomputable def car_speed : ℝ := 30

/-- Ratio of János's route length to Lajos's --/
noncomputable def route_ratio : ℝ := 3/2

/-- Fraction of route traveled by car --/
noncomputable def car_fraction : ℝ := 3/5

/-- Time difference in hours between János's and Lajos's arrivals --/
noncomputable def time_difference : ℝ := 1/15

theorem distance_between_janos_and_lajos :
  let lajos_distance := distance_between / 2
  let janos_distance := lajos_distance * route_ratio
  let lajos_time := (car_fraction * lajos_distance / car_speed) +
    ((1 - car_fraction) * lajos_distance / lajos_speed)
  let janos_time := (car_fraction * janos_distance / car_speed) +
    ((1 - car_fraction) * janos_distance / janos_speed)
  janos_time = lajos_time + time_difference ∧
  distance_between = lajos_distance + janos_distance :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_janos_and_lajos_l17_1703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_theta_l17_1761

theorem cos_pi_third_plus_theta (θ : ℝ) :
  Real.cos (π / 6 - θ) = 2 * Real.sqrt 2 / 3 →
  Real.cos (π / 3 + θ) = 1 / 3 ∨ Real.cos (π / 3 + θ) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_theta_l17_1761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_t_range_l17_1728

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def satisfiesConditions (t : ℝ) (tri : Triangle) : Prop :=
  let AB := Real.sqrt ((tri.A.1 - tri.B.1)^2 + (tri.A.2 - tri.B.2)^2)
  let BC := Real.sqrt ((tri.B.1 - tri.C.1)^2 + (tri.B.2 - tri.C.2)^2)
  let AC := Real.sqrt ((tri.A.1 - tri.C.1)^2 + (tri.A.2 - tri.C.2)^2)
  let angleABC := Real.arccos ((AB^2 + BC^2 - AC^2) / (2 * AB * BC))
  angleABC = Real.pi/4 ∧ AC = 1 ∧ BC = t

-- Define the uniqueness condition
def uniqueTriangle (t : ℝ) : Prop :=
  ∃! tri : Triangle, satisfiesConditions t tri

-- The theorem to prove
theorem triangle_t_range :
  ∀ t : ℝ, uniqueTriangle t → t ∈ Set.union (Set.Ioc 0 1) {Real.sqrt 2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_t_range_l17_1728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_acute_triangle_l17_1792

theorem at_least_one_acute_triangle (a : Fin 5 → ℝ) 
  (h_ordered : ∀ i j, i < j → a i ≤ a j)
  (h_triangle : ∀ (i j k : Fin 5), i < j → j < k → 
    a i + a j > a k) :
  ∃ (i j k : Fin 5), i < j ∧ j < k ∧ 
    (a i)^2 + (a j)^2 > (a k)^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_acute_triangle_l17_1792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_odd_exists_l17_1778

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / (3^x + 1) - a

theorem f_decreasing (a : ℝ) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂ :=
by sorry

theorem f_odd_exists :
  ∃ a : ℝ, ∀ x : ℝ, f a (-x) = -(f a x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_odd_exists_l17_1778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inverse_is_irrational_sqrt_l17_1732

/-- A quadratic function from ℝ to ℝ -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- An irrational function involving a square root from ℝ to ℝ -/
noncomputable def IrrationalSqrtFunction (p q r : ℝ) : ℝ → ℝ := λ y => Real.sqrt (p * y + q) + r

theorem quadratic_inverse_is_irrational_sqrt (a b c : ℝ) (ha : a ≠ 0) :
  ∃ p q r : ℝ, ∀ x y : ℝ, y = QuadraticFunction a b c x →
    x = IrrationalSqrtFunction p q r y ∨ x = -IrrationalSqrtFunction p q r y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inverse_is_irrational_sqrt_l17_1732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_microbrewery_output_increase_l17_1785

/-- Represents the percent change in beer output of a microbrewery -/
noncomputable def beer_output_increase (original_output : ℝ) (new_output : ℝ) : ℝ :=
  (new_output - original_output) / original_output * 100

/-- Represents the percent change in working hours -/
def working_hours_decrease : ℝ := 30

/-- Represents the percent increase in output per hour -/
def output_per_hour_increase : ℝ := 171.43

/-- Theorem stating the relationship between changes in working hours, 
    output per hour, and overall beer output -/
theorem microbrewery_output_increase 
  (original_output : ℝ) (original_hours : ℝ) 
  (new_output : ℝ) (new_hours : ℝ) 
  (h1 : new_hours = original_hours * (1 - working_hours_decrease / 100)) 
  (h2 : new_output / new_hours = 
        (original_output / original_hours) * (1 + output_per_hour_increase / 100)) :
  ∃ ε > 0, |beer_output_increase original_output new_output - 90| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_microbrewery_output_increase_l17_1785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l17_1727

/-- Represents a parabola with equation y² = 2px -/
structure Parabola where
  p : ℝ

/-- A point on the parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- The focus of the parabola -/
noncomputable def focus (par : Parabola) : ℝ × ℝ := (par.p / 2, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_focus_distance 
  (par : Parabola) 
  (point : ParabolaPoint) 
  (h1 : point.y^2 = 2 * par.p * point.x) -- Point satisfies parabola equation
  (h2 : point.x = 4) -- x-coordinate is 4
  (h3 : distance (point.x, point.y) (focus par) = 5) -- Distance to focus is 5
  : par.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l17_1727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_implies_range_l17_1798

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- Define the range of a
def range_of_a : Set ℝ := Set.Iic (-1/2) ∪ Set.Ici 2

-- State the theorem
theorem intersection_empty_implies_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a ∈ range_of_a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_implies_range_l17_1798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumscribed_sphere_volume_l17_1750

/-- The volume of the circumscribed sphere of a tetrahedron P-ABC -/
noncomputable def circumscribed_sphere_volume (PA PB PC : ℝ) : ℝ :=
  (4 / 3) * Real.pi * ((PA ^ 2 + PB ^ 2 + PC ^ 2) / 4) ^ (3 / 2)

/-- Theorem: The volume of the circumscribed sphere of tetrahedron P-ABC is 9π/2 -/
theorem tetrahedron_circumscribed_sphere_volume :
  circumscribed_sphere_volume (Real.sqrt 2) (Real.sqrt 3) 2 = (9 * Real.pi) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumscribed_sphere_volume_l17_1750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l17_1714

noncomputable def z : ℂ := (3 + 2*Complex.I) / (2 - 3*Complex.I)

theorem complex_number_properties : 
  (Complex.abs (z - 1 - 2*Complex.I) = Real.sqrt 2) ∧ 
  (Finset.range 2021).sum (λ n => z^(n+1)) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l17_1714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_isosceles_triangles_l17_1702

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Bool :=
  let d1 := squaredDistance t.a t.b
  let d2 := squaredDistance t.b t.c
  let d3 := squaredDistance t.c t.a
  d1 = d2 || d2 = d3 || d3 = d1

/-- The four triangles given in the problem -/
def triangles : List Triangle :=
  [ { a := ⟨1, 1⟩, b := ⟨1, 3⟩, c := ⟨3, 1⟩ }
  , { a := ⟨4, 3⟩, b := ⟨4, 1⟩, c := ⟨6, 3⟩ }
  , { a := ⟨0, 0⟩, b := ⟨2, 4⟩, c := ⟨4, 0⟩ }
  , { a := ⟨2, 2⟩, b := ⟨3, 4⟩, c := ⟨5, 2⟩ }
  ]

/-- The main theorem to prove -/
theorem three_isosceles_triangles :
  (triangles.filter isIsosceles).length = 3 := by
  sorry

#eval (triangles.filter isIsosceles).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_isosceles_triangles_l17_1702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l17_1797

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, Finset.sum (Finset.range n) (λ i => a (2 * i + 1)) = 1 - 2^n) →
  a 2 * a 3 * a 4 = -8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l17_1797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_DE_l17_1757

/-- Given a line segment AB of length 4, with point C on AB, and isosceles triangles ACD and BEC 
    on the same side of AB where AD = DC and CE = EB, the minimum length of segment DE is 2. -/
theorem min_length_DE (A B C D E : EuclideanSpace ℝ (Fin 2)) : 
  (‖B - A‖ = 4) → 
  (∃ t : ℝ, C = (1 - t) • A + t • B ∧ 0 ≤ t ∧ t ≤ 1) →
  (‖A - D‖ = ‖C - D‖) →
  (‖C - E‖ = ‖B - E‖) →
  (∀ D' E' : EuclideanSpace ℝ (Fin 2), 
    (‖A - D'‖ = ‖C - D'‖) → 
    (‖C - E'‖ = ‖B - E'‖) → 
    ‖E - D‖ ≤ ‖E' - D'‖) →
  ‖E - D‖ = 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_DE_l17_1757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_z_l17_1754

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ a then x^2 + 5*x + 2 else x + 2

-- Define g in terms of f
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 2*x

-- State the theorem
theorem range_of_z (a : ℝ) :
  (∃ x y z : ℝ, x < y ∧ y < z ∧ g a x = 0 ∧ g a y = 0 ∧ g a z = 0 ∧
    (∀ w : ℝ, g a w = 0 → w = x ∨ w = y ∨ w = z)) →
  ((2:ℝ)^a ≥ (1/2:ℝ) ∧ (2:ℝ)^a < 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_z_l17_1754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_and_cube_l17_1742

/-- The set of prime numbers on an 8-sided die -/
def primes_on_8_sided_die : Finset Nat :=
  {2, 3, 5, 7}

/-- The set of cube numbers on an 8-sided die -/
def cubes_on_8_sided_die : Finset Nat :=
  {1, 8}

/-- The number of faces on each die -/
def num_faces : Nat := 8

/-- The probability of rolling a prime on the blue die and a cube on the yellow die -/
theorem probability_prime_and_cube : 
  (Finset.card primes_on_8_sided_die * 
   Finset.card cubes_on_8_sided_die) / 
  (num_faces * num_faces) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_and_cube_l17_1742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_geq_one_l17_1786

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + 1

-- State the theorem
theorem decreasing_f_implies_a_geq_one (a : ℝ) (h1 : a > 0) :
  (∀ x ∈ Set.Icc 0 1, StrictMonoOn (fun x => -(f a x)) (Set.Icc 0 1)) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_geq_one_l17_1786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_decimal_digits_l17_1771

def fraction : ℚ := 987654321 / (2^30 * 5^3)

theorem min_decimal_digits (n : ℕ) : n = 30 ↔ 
  (∀ m : ℕ, m < n → ∃ k : ℚ, fraction ≠ k * (10 : ℚ)^(-m : ℤ)) ∧ 
  (∃ k : ℚ, fraction = k * (10 : ℚ)^(-n : ℤ)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_decimal_digits_l17_1771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_can_prevent_divisibility_by_three_l17_1752

/-- Represents a player in the game -/
inductive Player
| Alice
| Bob

/-- Represents the state of the game -/
structure GameState where
  sum : ℤ  -- Current sum modulo 3
  lastDigit : ℤ  -- Last digit chosen
  currentPlayer : Player

/-- Represents a valid move in the game -/
def validMove (s t : GameState) : Prop :=
  (s.currentPlayer ≠ t.currentPlayer) ∧
  (t.lastDigit ∈ ({0, 1, -1} : Set ℤ)) ∧
  (t.lastDigit ≠ s.lastDigit) ∧
  (t.sum ≡ s.sum + t.lastDigit [ZMOD 3])

/-- The theorem to be proved -/
theorem alice_can_prevent_divisibility_by_three :
  ∃ (strategy : GameState → ℤ),
    ∀ (game : Fin 2018 → GameState),
      (game 0 = GameState.mk 0 0 Player.Alice) →
      (∀ n : Fin 2017, validMove (game n) (game (n.val + 1))) →
      (∀ n : Fin 2018, n.val % 2 = 0 → (game n).currentPlayer = Player.Alice) →
      (∀ n : Fin 2018, n.val % 2 = 1 → (game n).currentPlayer = Player.Bob) →
      (∀ n : Fin 2018, n.val % 2 = 0 → (game (n.val + 1)).lastDigit = strategy (game n)) →
      (game 2017).sum ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_can_prevent_divisibility_by_three_l17_1752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_can_is_25_cents_l17_1780

/-- The cost of a 12-pack of soft drinks in dollars -/
def pack_cost : ℚ := 299 / 100

/-- The number of cans in a pack -/
def cans_per_pack : ℕ := 12

/-- The cost per can in dollars -/
def cost_per_can : ℚ := pack_cost / cans_per_pack

/-- Rounds a rational number to the nearest cent -/
def round_to_cent (x : ℚ) : ℚ := (x * 100).floor / 100

theorem cost_per_can_is_25_cents : round_to_cent cost_per_can = 25 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_can_is_25_cents_l17_1780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_law_equilateral_equivalence_l17_1760

theorem sine_law_equilateral_equivalence (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a = 2 * (Real.sin (A/2)) * (Real.sin (B/2)) * (Real.sin (C/2)) / (Real.sin ((A+B)/2))) →
  (b = 2 * (Real.sin (A/2)) * (Real.sin (B/2)) * (Real.sin (C/2)) / (Real.sin ((B+C)/2))) →
  (c = 2 * (Real.sin (A/2)) * (Real.sin (B/2)) * (Real.sin (C/2)) / (Real.sin ((C+A)/2))) →
  (a / Real.sin B = b / Real.sin C ∧ b / Real.sin C = c / Real.sin A) ↔ (A = B ∧ B = C) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_law_equilateral_equivalence_l17_1760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_equals_four_l17_1768

theorem sum_of_roots_equals_four (a b : ℝ) : 
  a ≠ b → 
  ({a^2 - 4*a, -1} : Set ℝ) = ({b^2 - 4*b + 1, -2} : Set ℝ) → 
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_equals_four_l17_1768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l17_1796

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * sin ((π / 2) * x + π / 5)

-- State the theorem
theorem min_distance_theorem (x₁ x₂ : ℝ) :
  (∀ x : ℝ, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  |x₂ - x₁| ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l17_1796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_about_line_symmetric_about_point_increasing_on_interval_not_shifted_graph_l17_1723

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x - Real.pi / 3)

def C := {(x, y) : ℝ × ℝ | y = f x}

-- Statement ①
theorem symmetric_about_line : 
  ∀ (x y : ℝ), (x, y) ∈ C → (11 * Real.pi / 6 - x, y) ∈ C := by sorry

-- Statement ②
theorem symmetric_about_point : 
  ∀ (x y : ℝ), (x, y) ∈ C → (4 * Real.pi / 3 - x, -y) ∈ C := by sorry

-- Statement ④
theorem increasing_on_interval : 
  ∀ (x₁ x₂ : ℝ), -Real.pi/12 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5*Real.pi/12 → f x₁ < f x₂ := by sorry

-- Statement ③ (negation)
theorem not_shifted_graph :
  ∃ (x : ℝ), f (x + Real.pi/3) ≠ 3 * Real.sin (2*x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_about_line_symmetric_about_point_increasing_on_interval_not_shifted_graph_l17_1723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l17_1711

open Real

-- Define the function f(x) = 3 + x ln x
noncomputable def f (x : ℝ) : ℝ := 3 + x * log x

-- State the theorem
theorem f_monotone_increasing :
  StrictMonoOn f (Set.Ioi (1 / exp 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l17_1711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l17_1720

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := 2 * (n : ℝ) - 1

noncomputable def sum_of_terms (m : ℕ) : ℝ := (m : ℝ) * ((arithmetic_sequence 1 + arithmetic_sequence m) / 2)

theorem arithmetic_sequence_property (m : ℕ) :
  sum_of_terms m = (arithmetic_sequence m + arithmetic_sequence (m + 1)) / 2 →
  m = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l17_1720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_graph_translation_l17_1719

noncomputable def f (x : ℝ) := Real.cos (2 * x + Real.pi / 6)

theorem cosine_graph_translation (φ : ℝ) (h1 : φ > 0) :
  (∀ x, f (x + φ) = f (-x - φ)) →
  (∀ ψ, 0 < ψ ∧ ψ < φ → ¬(∀ x, f (x + ψ) = f (-x - ψ))) →
  Real.tan φ = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_graph_translation_l17_1719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l17_1788

/-- A line in 3D space represented by its direction vector -/
structure Line3D where
  direction : Fin 3 → ℝ

/-- A plane in 3D space represented by its normal vector -/
structure Plane3D where
  normal : Fin 3 → ℝ

/-- Dot product of two 3D vectors -/
def dot_product (v1 v2 : Fin 3 → ℝ) : ℝ :=
  (Finset.univ.sum fun i => v1 i * v2 i)

/-- A line is parallel to a plane if and only if 
    the dot product of the line's direction vector 
    and the plane's normal vector is zero -/
theorem line_parallel_to_plane (l : Line3D) (p : Plane3D) :
  (dot_product l.direction p.normal = 0) ↔
  (∃ (k : ℝ), ∀ (i : Fin 3), l.direction i = k * p.normal i) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l17_1788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_tractor_speed_ratio_l17_1746

-- Define the speeds and distances
noncomputable def car_speed : ℝ := 630 / 7
noncomputable def tractor_speed : ℝ := 575 / 23
noncomputable def bike_speed : ℝ := car_speed * (5 / 9)

-- Define the theorem
theorem bike_tractor_speed_ratio :
  bike_speed / tractor_speed = 2 := by
  -- Expand the definitions
  unfold bike_speed car_speed tractor_speed
  -- Perform algebraic simplifications
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_tractor_speed_ratio_l17_1746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l17_1735

/-- Represents a wheel with two circles and five spokes. -/
structure Wheel :=
  (inner_circle : Fin 5)
  (outer_circle : Fin 5)
  (spokes : Fin 5)

/-- Represents the possible moves on the wheel. -/
inductive Move
  | inner_clockwise
  | outer_counterclockwise
  | switch

/-- A path on the wheel is a list of moves. -/
def WheelPath := List Move

/-- Checks if a path is valid according to the rules. -/
def is_valid_path (p : WheelPath) : Bool :=
  sorry

/-- Checks if a path starts and ends at point A. -/
def starts_and_ends_at_A (p : WheelPath) : Bool :=
  sorry

/-- Counts the number of valid paths with 15 steps that start and end at point A. -/
def count_valid_paths (w : Wheel) : Nat :=
  sorry

/-- The main theorem stating that the number of valid paths is 3004. -/
theorem valid_paths_count (w : Wheel) : count_valid_paths w = 3004 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l17_1735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_geometric_sequence_l17_1765

/-- Given a geometric sequence with first term 3 and common ratio 4y, 
    the fifth term is 768y^4 -/
theorem fifth_term_geometric_sequence (y : ℝ) : 
  let a : ℕ → ℝ := fun n => 
    if n = 1 then 3 else (if n > 1 then 3 * (4*y)^(n-1) else 0)
  a 5 = 768 * y^4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_geometric_sequence_l17_1765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_l17_1770

/-- Represents a parabola with equation ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) is on the parabola -/
noncomputable def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Calculates the x-coordinate of the vertex of the parabola -/
noncomputable def Parabola.vertexX (p : Parabola) : ℝ := -p.b / (2 * p.a)

/-- Calculates the y-coordinate of the vertex of the parabola -/
noncomputable def Parabola.vertexY (p : Parabola) : ℝ := p.c - p.b^2 / (4 * p.a)

/-- Checks if the parabola has a vertical axis of symmetry -/
def Parabola.hasVerticalAxis (p : Parabola) : Prop := p.a ≠ 0

theorem parabola_equation_correct : 
  ∃ (p : Parabola), 
    p.a = -1/3 ∧ 
    p.b = 2 ∧ 
    p.c = 2 ∧ 
    p.vertexX = 3 ∧ 
    p.vertexY = 5 ∧ 
    p.hasVerticalAxis ∧ 
    p.contains 0 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_l17_1770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_problem_l17_1739

/-- The average age of a group of people given their total number and sum of ages -/
def average_age (total_people : ℚ) (sum_ages : ℚ) : ℚ :=
  sum_ages / total_people

/-- The problem statement -/
theorem average_age_problem (fifth_graders : ℚ) (parents : ℚ) (grandparents : ℚ)
  (avg_age_fifth_graders : ℚ) (avg_age_parents : ℚ) (avg_age_grandparents : ℚ)
  (h1 : fifth_graders = 40)
  (h2 : parents = 60)
  (h3 : grandparents = 20)
  (h4 : avg_age_fifth_graders = 10)
  (h5 : avg_age_parents = 35)
  (h6 : avg_age_grandparents = 65) :
  average_age (fifth_graders + parents + grandparents)
    (fifth_graders * avg_age_fifth_graders + parents * avg_age_parents + grandparents * avg_age_grandparents) = 95 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_problem_l17_1739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_1100_l17_1784

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Calculates the shaded area in the given configuration -/
noncomputable def shadedArea (squareSide : ℝ) 
  (unshaded1 : Point × Point × Point) 
  (unshaded2 : Point × Point × Point)
  (shaded : Point × Point × Point) : ℝ :=
  let squareArea := squareSide * squareSide
  let unshaded1Area := triangleArea unshaded1.1 unshaded1.2.1 unshaded1.2.2
  let unshaded2Area := triangleArea unshaded2.1 unshaded2.2.1 unshaded2.2.2
  let shadedTriangleArea := triangleArea shaded.1 shaded.2.1 shaded.2.2
  squareArea - unshaded1Area - unshaded2Area + shadedTriangleArea

theorem shaded_area_is_1100 : 
  shadedArea 40 
    (⟨0, 0⟩, ⟨15, 0⟩, ⟨40, 25⟩) 
    (⟨25, 40⟩, ⟨40, 40⟩, ⟨40, 25⟩)
    (⟨0, 0⟩, ⟨0, 15⟩, ⟨15, 0⟩) = 1100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_1100_l17_1784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cute_numbers_l17_1704

def is_cute (n : ℕ+) : Prop :=
  ∀ m : ℕ+, 1 < m → m < n → Nat.Coprime m n → Nat.Prime m

theorem cute_numbers :
  ∀ n : ℕ+, is_cute n ↔ n ∈ ({1, 2, 3, 4, 6, 8, 12, 18, 24, 30} : Finset ℕ+) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cute_numbers_l17_1704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_30_equals_sqrt3_over_3_l17_1764

-- Define the tangent function for 30 degrees
noncomputable def tan_30 : ℝ := 1 / Real.sqrt 3

-- Define the correct answer
noncomputable def correct_answer : ℝ := Real.sqrt 3 / 3

-- Theorem statement
theorem tan_30_equals_sqrt3_over_3 : tan_30 = correct_answer := by
  -- Unfold the definitions
  unfold tan_30 correct_answer
  -- Simplify the expressions
  simp [Real.sqrt_sq, Real.sq_sqrt]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_30_equals_sqrt3_over_3_l17_1764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_polar_circle_l17_1737

-- Define the circle in polar coordinates
noncomputable def polar_circle (ρ θ : ℝ) : Prop := ρ = Real.sin θ

-- Define the center of the circle in polar coordinates
noncomputable def center : ℝ × ℝ := (1/2, Real.pi/2)

-- Theorem statement
theorem center_of_polar_circle :
  ∀ ρ θ : ℝ, polar_circle ρ θ → 
  ∃ x y : ℝ, x^2 + y^2 = ρ^2 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ →
  x^2 + (y - 1/2)^2 = 1/4 := by
  sorry

#check center_of_polar_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_polar_circle_l17_1737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l17_1743

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  2 * Real.sqrt (r^2 - d^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l17_1743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangles_in_park_l17_1781

/-- Represents the dimensions of a square park -/
def ParkDimensions : ℕ := 10

/-- Represents the base of a right triangle zone -/
def TriangleBase : ℕ := 1

/-- Represents the height of a right triangle zone -/
def TriangleHeight : ℕ := 3

/-- Calculates the area of a square given its side length -/
def square_area (side : ℕ) : ℕ := side * side

/-- Calculates the area of a right triangle given its base and height -/
def triangle_area (base height : ℕ) : ℚ := (base * height : ℚ) / 2

/-- Theorem stating the maximum number of complete right triangles that can fit in the park -/
theorem max_triangles_in_park : 
  ⌊(square_area ParkDimensions : ℚ) / (triangle_area TriangleBase TriangleHeight)⌋ = 66 := by
  sorry

#eval ⌊(square_area ParkDimensions : ℚ) / (triangle_area TriangleBase TriangleHeight)⌋

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangles_in_park_l17_1781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_l17_1767

open Real

-- Define the spherical coordinates
noncomputable def ρ : ℝ := 5
noncomputable def θ : ℝ := π / 6
noncomputable def φ : ℝ := π / 3

-- Define the conversion functions
noncomputable def x : ℝ := ρ * sin φ * cos θ
noncomputable def y : ℝ := ρ * sin φ * sin θ
noncomputable def z : ℝ := ρ * cos φ

-- State the theorem
theorem spherical_to_rectangular :
  (x, y, z) = (15 / 4, 5 * Real.sqrt 3 / 4, 5 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_l17_1767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_no_further_moves_l17_1734

/-- Represents a chessboard configuration -/
structure Board (n : ℕ) where
  pieces : Fin n → Fin n → Bool

/-- Represents a move on the chessboard -/
inductive Move (n : ℕ)
  | jump : Fin n → Fin n → Fin n → Fin n → Move n

/-- Applies a move to a board -/
def apply_move {n : ℕ} (b : Board n) (m : Move n) : Board n := sorry

/-- Checks if a move is valid on a given board -/
def is_valid_move {n : ℕ} (b : Board n) (m : Move n) : Prop := sorry

/-- Checks if any move is possible on a given board -/
def can_move {n : ℕ} (b : Board n) : Prop := sorry

/-- Initial board configuration with all cells occupied -/
def initial_board (n : ℕ) : Board n := sorry

/-- Sequence of moves -/
def move_sequence (n : ℕ) := List (Move n)

/-- Applies a sequence of moves to a board -/
def apply_moves {n : ℕ} (b : Board n) (moves : move_sequence n) : Board n := sorry

/-- Theorem: The minimum number of moves to reach a position where no further moves 
    are possible is at least ⌊n^2/3⌋ -/
theorem min_moves_to_no_further_moves (n : ℕ) :
  ∀ (moves : move_sequence n), 
    (∀ i, i < moves.length → is_valid_move (apply_moves (initial_board n) (moves.take i)) (moves.get ⟨i, sorry⟩)) →
    ¬(can_move (apply_moves (initial_board n) moves)) →
    moves.length ≥ ⌊(n^2 : ℚ)/3⌋ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_no_further_moves_l17_1734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse1_properties_ellipse2_properties_l17_1773

-- Part 1
def ellipse1 (x y : ℝ) : Prop := x^2/15 + y^2/10 = 1

theorem ellipse1_properties :
  (∃ x y, ellipse1 x y ∧ x = 3 ∧ y = -2) ∧
  (∀ x y, 4*x^2 + 9*y^2 = 36 → ∃ c, c^2 = 5 ∧ 
    ∀ x' y', ellipse1 x' y' → ∃ a b, a^2 - b^2 = c^2 ∧ x'^2/a^2 + y'^2/b^2 = 1) :=
by sorry

-- Part 2
def ellipse2 (x y : ℝ) : Prop := x^2/16 + y^2/8 = 1

theorem ellipse2_properties :
  (∀ x y, ellipse2 x y → ∃ a b, x^2/a^2 + y^2/b^2 = 1) ∧
  (∃ c > 0, c/4 = Real.sqrt 2 / 2) ∧
  (∃ f₁ f₂ : ℝ × ℝ, f₁.1 = -2 ∧ f₁.2 = 0 ∧ f₂.1 = 2 ∧ f₂.2 = 0) ∧
  (∃ p₁ p₂ : ℝ × ℝ, 
    ellipse2 p₁.1 p₁.2 ∧ 
    ellipse2 p₂.1 p₂.2 ∧ 
    p₁.2 = p₂.2 ∧ 
    dist p₁ (2, 0) + dist p₂ (2, 0) + dist p₁ p₂ = 16) :=
by sorry

noncomputable def dist (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse1_properties_ellipse2_properties_l17_1773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_plane_vectors_l17_1763

-- Define the angle function
noncomputable def angle (v w : ℝ × ℝ) : ℝ := 
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem angle_between_plane_vectors (v w : ℝ × ℝ) (hv : v ≠ (0, 0)) (hw : w ≠ (0, 0)) :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧ angle v w = θ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_plane_vectors_l17_1763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_is_forty_percent_l17_1766

/-- Represents a theater ticket campaign where 5 tickets can be purchased for the price of 3 tickets -/
structure TicketCampaign where
  /-- The price of a single ticket under normal circumstances -/
  normal_price : ℚ
  /-- The price of 5 tickets in the campaign -/
  campaign_price : ℚ
  /-- The campaign price is equal to the price of 3 tickets at normal price -/
  campaign_condition : campaign_price = 3 * normal_price

/-- The savings percentage in the TicketCampaign -/
def savings_percentage (c : TicketCampaign) : ℚ :=
  ((5 * c.normal_price - c.campaign_price) / (5 * c.normal_price)) * 100

/-- Theorem stating that the savings percentage in the TicketCampaign is 40% -/
theorem savings_is_forty_percent (c : TicketCampaign) : savings_percentage c = 40 := by
  sorry

def example_campaign : TicketCampaign := {
  normal_price := 10,
  campaign_price := 30,
  campaign_condition := by rfl
}

#eval savings_percentage example_campaign

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_is_forty_percent_l17_1766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_sum_eq_reciprocal_l17_1787

/-- The sum of 1/(3^a * 4^b * 6^c) over all positive integer triples (a, b, c) where 1 ≤ a < b < c -/
noncomputable def triple_sum : ℝ :=
  ∑' (a : ℕ), ∑' (b : ℕ), ∑' (c : ℕ),
    if 1 ≤ a ∧ a < b ∧ b < c then
      1 / (3^a * 4^b * 6^c : ℝ)
    else
      0

/-- The triple sum is equal to 1/13397 -/
theorem triple_sum_eq_reciprocal : triple_sum = 1 / 13397 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_sum_eq_reciprocal_l17_1787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_min_value_l17_1731

theorem triangle_min_value (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : 0 < A ∧ A < Real.pi)
  (h3 : 0 < B ∧ B < Real.pi) (h4 : 0 < C ∧ C < Real.pi) (h5 : Real.cos B = Real.sqrt 2 / 2) :
  (∀ A' B' C', A' + B' + C' = Real.pi → 0 < A' ∧ A' < Real.pi → 0 < B' ∧ B' < Real.pi → 0 < C' ∧ C' < Real.pi →
    Real.cos B' = Real.sqrt 2 / 2 →
      (Real.tan A' ^ 2 - 3) * Real.sin (2 * C') ≥ 4 * Real.sqrt 2 - 6) ∧
  (∃ A₀ B₀ C₀, A₀ + B₀ + C₀ = Real.pi ∧ 0 < A₀ ∧ A₀ < Real.pi ∧ 0 < B₀ ∧ B₀ < Real.pi ∧ 0 < C₀ ∧ C₀ < Real.pi ∧
    Real.cos B₀ = Real.sqrt 2 / 2 ∧ (Real.tan A₀ ^ 2 - 3) * Real.sin (2 * C₀) = 4 * Real.sqrt 2 - 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_min_value_l17_1731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_theorem_l17_1715

-- Define the trapezoid ABCD
structure Trapezoid (V : Type*) [LinearOrderedField V] :=
  (A B C D : V × V)
  (is_trapezoid : ∃ (m : V), (C.2 - D.2) = m * (A.2 - B.2) ∧ m ≠ 0)

-- Define the points A₀, B₀, C₀, D₀
def divide_segment {V : Type*} [LinearOrderedField V] (A : V × V) (A₁ : V × V) (t : V) : V × V :=
  (A.1 + t * (A₁.1 - A.1), A.2 + t * (A₁.2 - A.2))

variable {V : Type*} [LinearOrderedField V]

def A₀ (ABCD : Trapezoid V) (A₁ : V × V) (t : V) := divide_segment ABCD.A A₁ t
def B₀ (ABCD : Trapezoid V) (B₁ : V × V) (t : V) := divide_segment ABCD.B B₁ t
def C₀ (ABCD : Trapezoid V) (C₁ : V × V) (t : V) := divide_segment ABCD.C C₁ t
def D₀ (ABCD : Trapezoid V) (D₁ : V × V) (t : V) := divide_segment ABCD.D D₁ t

-- State the theorem
theorem trapezoid_theorem (ABCD : Trapezoid V) (t : V) (A₁ B₁ C₁ D₁ : V × V) :
  (∃ (m : V), (C₀ ABCD C₁ t).2 - (D₀ ABCD D₁ t).2 = m * ((A₀ ABCD A₁ t).2 - (B₀ ABCD B₁ t).2) ∧ m ≠ 0) ∧
  ((C₀ ABCD C₁ t).1 - (D₀ ABCD D₁ t).1) / ((A₀ ABCD A₁ t).1 - (B₀ ABCD B₁ t).1) = (ABCD.C.1 - ABCD.D.1) / (ABCD.A.1 - ABCD.B.1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_theorem_l17_1715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_residue_criterion_l17_1725

theorem quadratic_residue_criterion (p : Nat) (n : Int) 
  (h_prime : Nat.Prime p) 
  (h_odd : Odd p) 
  (h_not_div : ¬(p : Int) ∣ n) : 
  (∃ a : Int, n ≡ a^2 [ZMOD p]) ↔ (n^((p-1)/2) ≡ 1 [ZMOD p]) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_residue_criterion_l17_1725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_g_period_l17_1716

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sin x

noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬is_periodic f q

theorem f_period : is_smallest_positive_period f (2 * Real.pi) := by sorry

theorem g_period : is_smallest_positive_period g Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_g_period_l17_1716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K_squared_relatedness_correlation_l17_1776

/-- Represents the value of the K^2 statistic -/
def K_squared : ℝ → ℝ := sorry

/-- Represents the likelihood of two categorical variables being related -/
def relatedness_likelihood : ℝ → ℝ := sorry

/-- States that as K^2 increases, the likelihood of relatedness increases -/
theorem K_squared_relatedness_correlation :
  ∀ (k1 k2 : ℝ), k1 < k2 → relatedness_likelihood (K_squared k1) < relatedness_likelihood (K_squared k2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_K_squared_relatedness_correlation_l17_1776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l17_1747

/-- The area of a triangle with vertices at (2,0), (6,0), and (6,3) is 6 square units -/
theorem triangle_area : ∃ (area : ℝ), area = 6 := by
  -- Define the vertices of the triangle
  let A : ℝ × ℝ := (2, 0)
  let B : ℝ × ℝ := (6, 0)
  let C : ℝ × ℝ := (6, 3)

  -- Calculate the area of the triangle
  let area := (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

  -- Prove that the area is equal to 6
  use area
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l17_1747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_l17_1710

-- Define the basic geometric objects
structure Point where

structure Line where

structure Plane where

-- Define the relationships between geometric objects
def Point.onPlane (p : Point) (π : Plane) : Prop := sorry

def Point.onLine (p : Point) (l : Line) : Prop := sorry

def Line.onPlane (l : Line) (π : Plane) : Prop := sorry

def Plane.intersect (π₁ π₂ : Plane) : Line := sorry

-- Define the statements
def statement1 : Prop :=
  ∀ π₁ π₂ : Plane, ∀ p₁ p₂ p₃ : Point,
    p₁.onPlane π₁ ∧ p₂.onPlane π₁ ∧ p₃.onPlane π₁ ∧
    p₁.onPlane π₂ ∧ p₂.onPlane π₂ ∧ p₃.onPlane π₂ →
    π₁ = π₂

def statement2 : Prop :=
  ∀ l₁ l₂ : Line, ∃ π : Plane, l₁.onPlane π ∧ l₂.onPlane π

def statement3 : Prop :=
  ∀ π₁ π₂ : Plane, ∀ M : Point, ∀ l : Line,
    M.onPlane π₁ ∧ M.onPlane π₂ ∧ Plane.intersect π₁ π₂ = l →
    M.onLine l

def statement4 : Prop :=
  ∀ l₁ l₂ l₃ : Line, ∀ P : Point,
    P.onLine l₁ ∧ P.onLine l₂ ∧ P.onLine l₃ →
    ∃ π : Plane, l₁.onPlane π ∧ l₂.onPlane π ∧ l₃.onPlane π

-- Theorem statement
theorem exactly_one_correct :
  (statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ statement2 ∧ ¬statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ statement4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_l17_1710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_train_problem_l17_1774

/-- Two trains traveling at different speeds -/
theorem train_speed (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x ≠ y

/-- Combined distance covered by two trains in 3 hours -/
noncomputable def combined_distance (x y : ℝ) : ℝ :=
  3 * x + 3 * y

/-- Time taken to travel a given distance at a given speed -/
noncomputable def time_to_travel (distance speed : ℝ) : ℝ :=
  distance / speed

theorem train_problem (x y : ℝ) 
  (h1 : train_speed x y) 
  (h2 : combined_distance x y = 360) :
  x + y = 120 ∧ 
  time_to_travel 240 x = 240 / x ∧ 
  time_to_travel 240 y = 240 / y :=
by
  sorry

#check train_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_train_problem_l17_1774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l17_1783

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the line
def my_line (a x y : ℝ) : Prop := a * x - y + 5 = 0

-- Define the range of a
def valid_range (a : ℝ) : Prop := a < 0 ∨ a > 5/12

-- Define the condition for the perpendicular bisector
def perpendicular_bisector_condition (a : ℝ) : Prop :=
  ∃ (x y : ℝ), my_circle x y ∧ my_line a x y ∧ (x - (-2)) / (y - 4) = -1

-- Main theorem
theorem circle_line_intersection (a : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    my_circle x₁ y₁ ∧ my_circle x₂ y₂ ∧
    my_line a x₁ y₁ ∧ my_line a x₂ y₂) →
  (valid_range a ∧
   (perpendicular_bisector_condition a → a = 3/4)) := by
  sorry

#check circle_line_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l17_1783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_foci_distances_l17_1733

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 = 1

-- Define the foci (we don't need their exact coordinates for this problem)
variable (F₁ F₂ : ℝ × ℝ)

-- Define a point on the ellipse
variable (P : ℝ × ℝ)
variable (h_P : is_on_ellipse P.fst P.snd)

-- Define the distances from P to the foci
noncomputable def dist_PF₁ (P F₁ : ℝ × ℝ) : ℝ := Real.sqrt ((P.fst - F₁.fst)^2 + (P.snd - F₁.snd)^2)
noncomputable def dist_PF₂ (P F₂ : ℝ × ℝ) : ℝ := Real.sqrt ((P.fst - F₂.fst)^2 + (P.snd - F₂.snd)^2)

-- State the theorem
theorem max_product_foci_distances :
  ∃ P : ℝ × ℝ, is_on_ellipse P.fst P.snd ∧ dist_PF₁ P F₁ * dist_PF₂ P F₂ = 8 ∧
  ∀ Q : ℝ × ℝ, is_on_ellipse Q.fst Q.snd → dist_PF₁ Q F₁ * dist_PF₂ Q F₂ ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_foci_distances_l17_1733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_implies_a_half_l17_1736

theorem unique_solution_implies_a_half (a : ℝ) (h_pos : a > 0) : 
  (∃! x : ℝ, x > 0 ∧ x^2 - 2*a*Real.log x - 2*a*x = 0) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_implies_a_half_l17_1736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_door_height_is_three_l17_1709

/-- Represents the dimensions and cost information for a room whitewashing problem -/
structure RoomInfo where
  length : ℝ
  width : ℝ
  height : ℝ
  doorWidth : ℝ
  windowWidth : ℝ
  windowHeight : ℝ
  numWindows : ℕ
  whitewashCost : ℝ
  totalCost : ℝ

/-- Calculates the height of the door given the room information -/
noncomputable def calculateDoorHeight (info : RoomInfo) : ℝ :=
  let wallArea := 2 * (info.length + info.width) * info.height
  let windowArea := info.numWindows * info.windowWidth * info.windowHeight
  let whitewashedArea := (info.totalCost / info.whitewashCost)
  (wallArea - windowArea - whitewashedArea) / info.doorWidth

/-- Theorem stating that the calculated door height is 3 feet for the given room information -/
theorem door_height_is_three (info : RoomInfo) 
    (h1 : info.length = 25)
    (h2 : info.width = 15)
    (h3 : info.height = 12)
    (h4 : info.doorWidth = 6)
    (h5 : info.windowWidth = 4)
    (h6 : info.windowHeight = 3)
    (h7 : info.numWindows = 3)
    (h8 : info.whitewashCost = 2)
    (h9 : info.totalCost = 1812) :
  calculateDoorHeight info = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_door_height_is_three_l17_1709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f1001_l17_1706

-- Define the function f₁
noncomputable def f₁ (x : ℝ) : ℝ := 2/3 - 3/(3*x + 1)

-- Define the recursive function fₙ
noncomputable def f : ℕ → ℝ → ℝ
  | 0, x => x
  | n + 1, x => f₁ (f n x)

-- State the theorem
theorem unique_solution_f1001 :
  ∃! x : ℝ, f 1001 x = x - 3 ∧ x = 5/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f1001_l17_1706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l17_1718

theorem angle_equality (α β : ℝ) : 
  0 < α ∧ α < π/2 → 
  0 < β ∧ β < π/2 → 
  Real.sin (α + β) + Real.sin (α - β) = Real.sin (2 * β) → 
  α = β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l17_1718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l17_1738

-- Define the differential equation
def diff_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (deriv^[2] y) x + 6 * (deriv y) x + 9 * y x = 14 * Real.exp (-3 * x)

-- Define the general solution
noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  (C₁ + C₂ * x + 7 * x^2) * Real.exp (-3 * x)

-- Theorem statement
theorem general_solution_satisfies_diff_eq (C₁ C₂ : ℝ) :
  ∀ x, diff_eq (general_solution C₁ C₂) x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l17_1738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_value_l17_1755

-- Define the set A
def A (x : ℝ) : Set ℝ := {0, 1, Real.log (x^2 + 2) / Real.log 3, x^2 - 3*x}

-- State the theorem
theorem unique_x_value : ∀ x : ℝ, -2 ∈ A x → x = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_value_l17_1755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_function_range_l17_1794

theorem monotone_function_range (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x : ℝ, x > 0 → StrictMono (λ x : ℝ ↦ a^x + (1+a)^x)) →
  a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_function_range_l17_1794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_equals_sqrt_sum_l17_1726

/-- A regular hexagon with side length 3 -/
structure RegularHexagon where
  side_length : ℝ
  side_length_eq : side_length = 3

/-- The area of a regular hexagon -/
noncomputable def area (h : RegularHexagon) : ℝ := 
  (3 * Real.sqrt 3 * h.side_length ^ 2) / 2

/-- The theorem to be proved -/
theorem hexagon_area_equals_sqrt_sum (h : RegularHexagon) : 
  area h = Real.sqrt 54 + Real.sqrt 243 := by
  sorry

#check hexagon_area_equals_sqrt_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_equals_sqrt_sum_l17_1726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_radius_circle_l17_1707

/-- The maximum radius of a circle with center at (0,1) that touches the lines x - y + 2 = 0 and x - 3y = 0 -/
theorem max_radius_circle :
  ∀ r : ℝ, r > 0 →
  (∃ a b : ℝ, a^2 + (b-1)^2 = r^2 ∧ a - b + 2 ≥ 0 ∧ a - 3*b ≤ 0) →
  r ≤ Real.sqrt 2 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_radius_circle_l17_1707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_eq_six_l17_1769

def count_triples : Nat :=
  (Finset.filter (fun (triple : Nat × Nat × Nat) =>
    let (x, y, z) := triple
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    Nat.lcm x y = 180 ∧
    Nat.lcm x z = 504 ∧
    Nat.lcm y z = 1260
  ) (Finset.product (Finset.range 1000) (Finset.product (Finset.range 1000) (Finset.range 1000)))).card

theorem count_triples_eq_six : count_triples = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_eq_six_l17_1769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carols_weight_l17_1730

/-- Given Mildred's weight and the weight difference between Mildred and Carol, 
    prove Carol's weight. -/
theorem carols_weight (mildreds_weight : ℕ) (weight_difference : ℕ) (carols_weight : ℕ)
    (h1 : mildreds_weight = 59)
    (h2 : weight_difference = 50)
    (h3 : mildreds_weight = weight_difference + carols_weight) :
  carols_weight = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carols_weight_l17_1730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_M_inverse_l17_1721

def M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 2, 2]

theorem eigenvalues_of_M_inverse :
  let M_inv := M⁻¹
  ∃ (v₁ v₂ : (Fin 2) → ℝ), v₁ ≠ 0 ∧ v₂ ≠ 0 ∧ v₁ ≠ v₂ ∧
    M_inv.mulVec v₁ = (1 : ℝ) • v₁ ∧
    M_inv.mulVec v₂ = (1/2 : ℝ) • v₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_M_inverse_l17_1721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_side_range_l17_1717

theorem acute_triangle_side_range :
  ∀ a : ℝ,
    (∃ (A B C : ℝ),
      0 < A ∧ A < π / 2 ∧
      0 < B ∧ B < π / 2 ∧
      0 < C ∧ C < π / 2 ∧
      A + B + C = π ∧
      3^2 = 4^2 + a^2 - 2 * 4 * a * (Real.cos C) ∧
      4^2 = 3^2 + a^2 - 2 * 3 * a * (Real.cos B) ∧
      a^2 = 3^2 + 4^2 - 2 * 3 * 4 * (Real.cos A)) ↔
    (Real.sqrt 7 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_side_range_l17_1717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_B_l17_1751

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

def number (B : Nat) : Nat := 303200 + B

theorem unique_prime_B :
  (∃! B : Nat, B ∈ ({1, 4, 5, 7, 9} : Finset Nat) ∧ is_prime (number B)) ∧
  (∀ B : Nat, B ∈ ({1, 4, 5, 7, 9} : Finset Nat) ∧ is_prime (number B) → B = 9) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_B_l17_1751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l17_1705

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1

theorem f_properties :
  (∀ x : ℝ, f x = f (-π/6 - x)) ∧
  (∀ x y : ℝ, -π/6 < x ∧ x < y ∧ y < π/6 → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l17_1705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_property_l17_1753

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola -/
def focus : Point := ⟨1, 0⟩

/-- The directrix of the parabola -/
def directrix : ℝ := -1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a parabola y² = 4x, if a line intersects it at A and B
    such that |AF| = 4|BF|, then |CD| = 5, where C and D are the feet of
    perpendiculars from A and B to the directrix. -/
theorem parabola_intersection_property
  (A B : Point)
  (h_A : A ∈ Parabola)
  (h_B : B ∈ Parabola)
  (h_line : ∃ (m c : ℝ), A.y = m * A.x + c ∧ B.y = m * B.x + c)
  (h_AF_BF : distance A focus = 4 * distance B focus)
  (C : Point)
  (h_C : C.x = directrix ∧ (A.y - C.y) * (A.x - C.x) = 0)
  (D : Point)
  (h_D : D.x = directrix ∧ (B.y - D.y) * (B.x - D.x) = 0) :
  distance C D = 5 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_property_l17_1753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_1_f_property_2_l17_1779

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Theorem 1
theorem f_property_1 (a : ℝ) (x₀ : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a x₀ = 2) :
  f a (3 * x₀) = 8 := by sorry

-- Define the inverse function g(x) = log_a(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem 2
theorem f_property_2 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 4) :
  Set.range (fun x => g a x) ∩ Set.Icc (1/2 : ℝ) 2 = Set.Icc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_1_f_property_2_l17_1779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weeks_to_purchase_car_l17_1713

noncomputable def hourly_rate : ℝ := 20
noncomputable def hours_per_week : ℝ := 52
noncomputable def overtime_threshold : ℝ := 40
noncomputable def overtime_rate : ℝ := 1.5
noncomputable def car_cost : ℝ := 4640

noncomputable def regular_hours : ℝ := min hours_per_week overtime_threshold
noncomputable def overtime_hours : ℝ := max (hours_per_week - overtime_threshold) 0

noncomputable def weekly_earnings : ℝ :=
  regular_hours * hourly_rate + overtime_hours * hourly_rate * overtime_rate

theorem weeks_to_purchase_car :
  ⌈car_cost / weekly_earnings⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weeks_to_purchase_car_l17_1713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_approx_9_seconds_l17_1729

/-- Conversion factor from kmph to m/s -/
noncomputable def kmph_to_ms : ℝ := 1000 / 3600

/-- Length of the first train in meters -/
def train1_length : ℝ := 280

/-- Speed of the first train in kmph -/
def train1_speed_kmph : ℝ := 120

/-- Length of the second train in meters -/
def train2_length : ℝ := 220.04

/-- Speed of the second train in kmph -/
def train2_speed_kmph : ℝ := 80

/-- Time for the trains to cross each other in seconds -/
noncomputable def crossing_time : ℝ :=
  (train1_length + train2_length) /
  ((train1_speed_kmph * kmph_to_ms) + (train2_speed_kmph * kmph_to_ms))

theorem trains_crossing_time_approx_9_seconds :
  ∃ ε > 0, abs (crossing_time - 9) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_approx_9_seconds_l17_1729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_form_l17_1758

-- Define the triangle vertices and point P
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (4, 6)
def P : ℝ × ℝ := (5, 3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem distance_sum_form :
  ∃ (m n a b : ℕ),
    distance P A + distance P B + distance P C = m * Real.sqrt (a : ℝ) + n * Real.sqrt (b : ℝ) ∧
    m + n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_form_l17_1758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_ratio_l17_1777

-- Define a triangle structure
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  ξ : ℝ
  η : ℝ
  ζ : ℝ
  -- Triangle inequality
  hxy : x + y > z
  hyz : y + z > x
  hxz : x + z > y
  -- Angle sum in a triangle is π
  hangle_sum : ξ + η + ζ = Real.pi
  -- Law of sines
  hlaw_of_sines : x / Real.sin ξ = y / Real.sin η
  hlaw_of_sines' : y / Real.sin η = z / Real.sin ζ
  -- Given condition
  hcondition : x^2 + y^2 = 2023 * z^2

theorem triangle_cotangent_ratio (t : Triangle) : 
  (Real.tan t.ζ)⁻¹ / ((Real.tan t.ξ)⁻¹ + (Real.tan t.η)⁻¹) = 1011 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_ratio_l17_1777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l17_1793

open Real

theorem problem_solution (α β : ℝ) 
  (h_α : α ∈ Set.Ioo 0 (π/2))
  (h_β : β ∈ Set.Ioo (π/2) π)
  (h_cos_β : cos β = -1/3)
  (h_sin_α_β : sin (α + β) = (4 - Real.sqrt 2)/6) :
  tan (2*β) = 4*Real.sqrt 2/7 ∧ α = π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l17_1793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l17_1799

theorem trigonometric_identity (α β : ℝ) 
  (h1 : π/2 ≤ β) (h2 : β ≤ α) (h3 : α ≤ 3*π/4)
  (h4 : Real.cos (α - β) = 12/13) (h5 : Real.sin (α + β) = -3/5) :
  Real.sin (2*α) = -56/65 ∧ Real.cos (2*β) = -63/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l17_1799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_about_neg_pi_third_l17_1791

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + 2 * Real.pi / 3)

theorem f_symmetric_about_neg_pi_third :
  ∀ x : ℝ, f ((-Real.pi/3) + x) = f ((-Real.pi/3) - x) :=
by
  intro x
  unfold f
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_about_neg_pi_third_l17_1791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_competition_participants_l17_1744

/-- Represents the number of high school students in the chess competition. -/
def num_high_school_students : ℕ := sorry

/-- Represents the total number of participants in the chess competition. -/
def total_participants : ℕ := num_high_school_students + 2

/-- The total number of games played in the competition. -/
def total_games : ℕ := total_participants * (total_participants - 1) / 2

/-- The total points scored by all participants. -/
def total_points : ℕ := total_games

/-- The points scored by the two junior high students. -/
def junior_high_points : ℕ := 8

/-- The points scored by all high school students. -/
def high_school_points : ℕ := total_points - junior_high_points

/-- Each high school student scores the same number of points. -/
def points_per_high_school_student : ℕ := high_school_points / num_high_school_students

theorem chess_competition_participants :
  (num_high_school_students = 7 ∨ num_high_school_students = 14) ↔
  (high_school_points % num_high_school_students = 0 ∧
   ∀ n : ℕ, n ≠ 7 → n ≠ 14 → n > 2 →
     (total_participants * (total_participants - 1) / 2 - junior_high_points) % n ≠ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_competition_participants_l17_1744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_proof_l17_1775

noncomputable def N : ℚ := 199.99999999999997 / 2.5

theorem fraction_proof (F : ℚ) : 
  (4/5 : ℚ) * F * N = 24 ∧ (250/100 : ℚ) * N = 199.99999999999997 → F = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_proof_l17_1775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digitlength_reaches_four_l17_1722

/-- The digitlength of a positive integer is the total number of letters used in spelling its digits. -/
def digitlength : ℕ+ → ℕ :=
  sorry

/-- The sequence of digitlengths starting from a given positive integer. -/
def digitlength_sequence (start : ℕ+) : ℕ → ℕ
  | 0 => start.val
  | n + 1 => digitlength ⟨digitlength_sequence start n, by
    sorry
  ⟩

/-- For any positive integer, the sequence of digitlengths will eventually reach 4. -/
theorem digitlength_reaches_four (start : ℕ+) : ∃ n : ℕ, digitlength_sequence start n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digitlength_reaches_four_l17_1722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_small_boxes_in_large_box_l17_1772

noncomputable section

/-- Calculates the volume of a box given its dimensions in meters -/
def boxVolume (length width height : ℝ) : ℝ := length * width * height

/-- Converts centimeters to meters -/
def cmToM (cm : ℝ) : ℝ := cm / 100

theorem max_small_boxes_in_large_box :
  let largeBoxLength : ℝ := 6
  let largeBoxWidth : ℝ := 5
  let largeBoxHeight : ℝ := 4
  let smallBoxLength : ℝ := cmToM 60
  let smallBoxWidth : ℝ := cmToM 50
  let smallBoxHeight : ℝ := cmToM 40
  let largeBoxVolume := boxVolume largeBoxLength largeBoxWidth largeBoxHeight
  let smallBoxVolume := boxVolume smallBoxLength smallBoxWidth smallBoxHeight
  ⌊largeBoxVolume / smallBoxVolume⌋ = 1000 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_small_boxes_in_large_box_l17_1772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_behavior_l17_1782

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := x^3 - x^2 - 1/2 * x + 2
def g (x : ℝ) : ℝ := x^2 + 3/2 * x + 4

-- Define the intersection property
def intersects (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = g x

-- Define the "intersects from below upwards" property
def intersects_from_below (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = g x ∧ ∀ y : ℝ, y < x → g y < f y

-- Define the "passes from below upwards on the left side" property
def passes_from_below_left (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = g x ∧ 
    (∀ y : ℝ, y < x → g y < f y) ∧
    (∀ z : ℝ, x < z → f z < g z) ∧
    (x < 0)

-- Theorem statement
theorem intersection_behavior :
  intersects f g →
  intersects_from_below f g →
  passes_from_below_left f g :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_behavior_l17_1782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_height_l17_1745

noncomputable def height_of_max_volume (l : ℝ) : ℝ := Real.sqrt 3 / 3 * l

noncomputable def cone_volume (h l : ℝ) : ℝ := (Real.pi / 3) * (l^2 - h^2) * h

theorem max_volume_height (l : ℝ) (h : ℝ) (hl : l > 0) (hh : 0 < h ∧ h < l) :
  cone_volume h l ≤ cone_volume (height_of_max_volume l) l := by
  sorry

#check max_volume_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_height_l17_1745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blood_expires_same_day_l17_1749

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- Represents the expiration time of blood in seconds -/
def blood_expiration_time : ℕ := (8 : ℕ).factorial

/-- Proves that blood donated at noon expires on the same day -/
theorem blood_expires_same_day : blood_expiration_time < seconds_per_day := by
  -- Unfold the definitions
  unfold blood_expiration_time seconds_per_day
  -- Calculate 8!
  have h1 : (8 : ℕ).factorial = 40320 := rfl
  rw [h1]
  -- Prove the inequality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blood_expires_same_day_l17_1749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_is_two_lines_l17_1789

/-- Point q is between points p and r -/
def Between (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ q = (t * p.1 + (1 - t) * r.1, t * p.2 + (1 - t) * r.2)

/-- A set is a line if it's nonempty and for any three distinct points, one is between the other two -/
def IsLine (s : Set (ℝ × ℝ)) : Prop :=
  s.Nonempty ∧ 
  ∀ p q r : ℝ × ℝ, p ∈ s → q ∈ s → r ∈ s → p ≠ q → q ≠ r → p ≠ r →
  (Between p q r ∨ Between q r p ∨ Between r p q)

/-- The graph of (x + ay)^2 = x^2 + y^2 consists of two lines -/
theorem graph_is_two_lines (a : ℝ) : 
  ∃ (l₁ l₂ : Set (ℝ × ℝ)), IsLine l₁ ∧ IsLine l₂ ∧ 
  {p : ℝ × ℝ | (p.1 + a * p.2)^2 = p.1^2 + p.2^2} = l₁ ∪ l₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_is_two_lines_l17_1789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_problem_l17_1740

/-- Point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Given two points in polar coordinates, compute the distance between them -/
noncomputable def distance (a b : PolarPoint) : ℝ :=
  Real.sqrt (a.r^2 + b.r^2 - 2 * a.r * b.r * Real.cos (a.θ - b.θ))

/-- Given two points in polar coordinates, compute the area of the triangle formed with the pole -/
noncomputable def triangleArea (a b : PolarPoint) : ℝ :=
  (1/2) * a.r * b.r * Real.sin (a.θ - b.θ)

theorem polar_problem (A B : PolarPoint) 
  (hA : A = ⟨2, π/3⟩) (hB : B = ⟨3, 0⟩) : 
  distance A B = Real.sqrt 7 ∧ triangleArea A B = (3 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_problem_l17_1740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_right_triangle_values_l17_1724

def OA : ℝ × ℝ := (2, -1)
def OB : ℝ × ℝ := (3, 2)
def OC (m : ℝ) : ℝ × ℝ := (m, 2*m + 1)

def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)
def AC (m : ℝ) : ℝ × ℝ := ((OC m).1 - OA.1, (OC m).2 - OA.2)
def BC (m : ℝ) : ℝ × ℝ := ((OC m).1 - OB.1, (OC m).2 - OB.2)

def is_triangle (m : ℝ) : Prop :=
  AB.1 * (AC m).2 - AB.2 * (AC m).1 ≠ 0

def is_right_triangle (m : ℝ) : Prop :=
  AB.1 * (AC m).1 + AB.2 * (AC m).2 = 0 ∨
  AB.1 * (BC m).1 + AB.2 * (BC m).2 = 0 ∨
  (AC m).1 * (BC m).1 + (AC m).2 * (BC m).2 = 0

theorem triangle_condition (m : ℝ) :
  is_triangle m ↔ m ≠ 8 :=
sorry

theorem right_triangle_values :
  ∀ m : ℝ, is_right_triangle m ↔ m = -4/7 ∨ m = 6/7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_right_triangle_values_l17_1724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frobenius_theorem_l17_1748

theorem frobenius_theorem (a b n : ℕ) (h_coprime : Nat.Coprime a b) (h_a : a ≥ 1) (h_b : b ≥ 1) (h_n : n ≥ (a - 1) * (b - 1)) :
  ∃ (u v : ℕ), n = u * a + v * b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frobenius_theorem_l17_1748
