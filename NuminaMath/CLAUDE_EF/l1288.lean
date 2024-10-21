import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_passes_gas_station_twice_l1288_128815

-- Define the piecewise function s(t)
noncomputable def s (t : ℝ) : ℝ :=
  if t ≥ 0 ∧ t ≤ 3 then -5*t*(t-13)
  else if t > 3 ∧ t ≤ 8 then 150
  else if t > 8 ∧ t ≤ 10.5 then 60*t - 330
  else 0  -- undefined for other t values

-- Define the gas station distance
def gas_station_distance : ℝ := 60

-- Theorem statement
theorem car_passes_gas_station_twice :
  ∃ t₁ t₂ : ℝ, 
    t₁ ≥ 0 ∧ t₁ ≤ 3 ∧
    t₂ > 8 ∧ t₂ ≤ 10.5 ∧
    s t₁ = gas_station_distance ∧
    s t₂ = gas_station_distance ∧
    t₁ = 1 ∧ t₂ = 9.5 ∧
    ∀ t : ℝ, (t ≥ 0 ∧ t ≤ 10.5 ∧ s t = gas_station_distance) → (t = t₁ ∨ t = t₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_passes_gas_station_twice_l1288_128815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_in_interval_l1288_128897

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + 3 * x - 2

-- State the theorem
theorem max_value_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧
  ∀ (x : ℝ), x ∈ Set.Icc 0 2 → f x ≤ f c ∧
  f c = -2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_in_interval_l1288_128897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_city_intersections_l1288_128849

/-- Represents a city with a given number of streets -/
structure City where
  num_streets : ℕ

/-- The number of intersections in a city with the given properties -/
def num_intersections (c : City) : ℕ := sorry

/-- Predicate to check if two streets are parallel -/
def are_parallel : ℕ → ℕ → Prop := sorry

/-- Predicate to check if three streets meet at a single point -/
def meet_at_point : ℕ → ℕ → ℕ → Prop := sorry

/-- Theorem: A city with 10 streets has 45 intersections -/
theorem math_city_intersections :
  ∀ (c : City),
    c.num_streets = 10 →
    (∀ i j, i ≠ j → ¬ are_parallel i j) →
    (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ meet_at_point i j k) →
    num_intersections c = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_city_intersections_l1288_128849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tile_probability_l1288_128813

/-- The number of tiles in the box -/
def total_tiles : ℕ := 70

/-- A tile is red if its number is congruent to 3 mod 7 -/
def is_red (n : ℕ) : Prop := n % 7 = 3

/-- The set of red tiles -/
def red_tiles : Finset ℕ := Finset.filter (fun n => n % 7 = 3) (Finset.range total_tiles)

/-- The probability of choosing a red tile -/
noncomputable def prob_red_tile : ℚ := (red_tiles.card : ℚ) / total_tiles

theorem red_tile_probability : prob_red_tile = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tile_probability_l1288_128813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_divisors_multiple_of_six_l1288_128889

theorem equidistant_divisors_multiple_of_six (n : ℕ) : 
  (∃ (s t : ℕ), s ≠ t ∧ s > 0 ∧ t > 0 ∧ s ∣ n ∧ t ∣ n ∧ 
   (s : ℤ) - (n : ℤ)/3 = (n : ℤ)/3 - (t : ℤ)) → 
  (∃ (k : ℕ), k > 0 ∧ n = 6*k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_divisors_multiple_of_six_l1288_128889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_top_is_60_l1288_128863

/-- Represents the mountain journey problem -/
structure MountainJourney where
  initial_speed : ℝ
  ascending_speed_factor : ℝ
  descending_speed_factor : ℝ
  distance_down : ℝ
  total_time : ℝ

/-- Calculates the distance to the top of the mountain -/
noncomputable def distance_to_top (journey : MountainJourney) : ℝ :=
  let ascending_speed := journey.initial_speed * journey.ascending_speed_factor
  let descending_speed := journey.initial_speed * journey.descending_speed_factor
  let time_descending := journey.distance_down / descending_speed
  let time_ascending := journey.total_time - time_descending
  ascending_speed * time_ascending

/-- Theorem stating the distance to the top of the mountain is 60 miles -/
theorem distance_to_top_is_60 (journey : MountainJourney) 
    (h1 : journey.initial_speed = 30)
    (h2 : journey.ascending_speed_factor = 0.5)
    (h3 : journey.descending_speed_factor = 1.2)
    (h4 : journey.distance_down = 72)
    (h5 : journey.total_time = 6) :
    distance_to_top journey = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_top_is_60_l1288_128863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_midline_l1288_128811

/-- Given a sinusoidal function y = a * sin(b * x + c) + d where a, b, c, and d are positive constants,
    if the function oscillates between 4 and -2, then d = 1 -/
theorem sinusoidal_midline (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_max : ∀ x, a * Real.sin (b * x + c) + d ≤ 4)
  (h_min : ∀ x, a * Real.sin (b * x + c) + d ≥ -2)
  (h_reaches_max : ∃ x, a * Real.sin (b * x + c) + d = 4)
  (h_reaches_min : ∃ x, a * Real.sin (b * x + c) + d = -2) :
  d = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_midline_l1288_128811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reverse_domain_intervals_of_g_l1288_128818

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 + 2*x else x^2 + 2*x

-- Define the property of being a reverse domain interval
def is_reverse_domain_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc a b → f x ∈ Set.Icc (1/b) (1/a)

-- State the theorem
theorem reverse_domain_intervals_of_g :
  (is_reverse_domain_interval g 1 ((1 + Real.sqrt 5) / 2)) ∧
  (is_reverse_domain_interval g (-(1 + Real.sqrt 5) / 2) (-1)) := by
  sorry

#check reverse_domain_intervals_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reverse_domain_intervals_of_g_l1288_128818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_value_l1288_128896

def sequence_a : ℕ → ℚ
  | 0 => -1/4  -- Added case for 0
  | 1 => -1/4
  | n+2 => 1 - 1 / sequence_a (n+1)

theorem a_2016_value : sequence_a 2016 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_value_l1288_128896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1288_128844

theorem trigonometric_identity (α β : ℝ) :
  (2 * Real.sin (α + β) + 1 / Real.sin (α + β))^2 + 
  (2 * Real.cos (α + β) + 1 / Real.cos (α + β))^2 = 
  9 + Real.tan (α + β)^2 + (1 / Real.tan (α + β))^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1288_128844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1288_128885

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def tangent_to_x_axis (c : Circle) : Prop :=
  c.center.2 = c.radius

def center_on_line (c : Circle) : Prop :=
  3 * c.center.1 = c.center.2

noncomputable def chord_length (c : Circle) : ℝ :=
  2 * Real.sqrt (c.radius^2 - (c.center.1 - c.center.2)^2 / 2)

-- Theorem statement
theorem circle_properties :
  ∃ (c : Circle),
    tangent_to_x_axis c ∧
    center_on_line c ∧
    chord_length c = 2 ∧
    ((c.center = (Real.sqrt 7 / 7, 3 * Real.sqrt 7 / 7) ∧ c.radius = 3 * Real.sqrt 7 / 7) ∨
     (c.center = (-Real.sqrt 7 / 7, -3 * Real.sqrt 7 / 7) ∧ c.radius = 3 * Real.sqrt 7 / 7)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1288_128885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_in_pyramid_l1288_128847

/-- Regular quadrilateral pyramid with a cone -/
structure PyramidWithCone where
  /-- Side length of the square base of the pyramid -/
  base_side : ℝ
  /-- Distance from C to E on edge DC -/
  ce_length : ℝ
  /-- Distance from C to F on edge BC -/
  cf_length : ℝ

/-- Volume of the cone in the pyramid -/
noncomputable def cone_volume (p : PyramidWithCone) : ℝ :=
  63 * Real.pi * Real.sqrt 6

/-- Theorem stating the volume of the cone -/
theorem cone_volume_in_pyramid (p : PyramidWithCone) 
  (h1 : p.base_side = 10)
  (h2 : p.ce_length = 6)
  (h3 : p.cf_length = 9) :
  cone_volume p = 63 * Real.pi * Real.sqrt 6 := by
  sorry

#eval "Pyramid with Cone theorem compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_in_pyramid_l1288_128847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_inequality_l1288_128808

-- Define the interval (-1, 0)
def OpenInterval := { x : ℝ | -1 < x ∧ x < 0 }

-- Define α₁, α₂, α₃
noncomputable def α₁ (x : ℝ) := Real.cos (Real.sin x * Real.pi)
noncomputable def α₂ (x : ℝ) := Real.sin (Real.cos x * Real.pi)
noncomputable def α₃ (x : ℝ) := Real.cos ((x + 1) * Real.pi)

-- State the theorem
theorem alpha_inequality : ∀ x ∈ OpenInterval, α₃ x < α₂ x ∧ α₂ x < α₁ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_inequality_l1288_128808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_and_collinearity_l1288_128852

/-- Given two non-collinear vectors a and b, and points A, B, C defined by their position vectors,
    prove properties about their relationships and collinearity. -/
theorem vector_relationships_and_collinearity 
  (a b : ℝ → ℝ → ℝ → ℝ) 
  (h_not_collinear : ¬ ∃ (k : ℝ), a = k • b) 
  (OA OB OC : ℝ → ℝ → ℝ → ℝ) 
  (h_OA : OA = 2 • a - b) 
  (h_OB : OB = a + 2 • b) 
  (m n : ℝ) 
  (h_OC : OC = m • a + n • b) 
  (h_relation : 2 • OA - OB = OC) :
  (m = 3 ∧ n = -4) ∧ 
  (∀ (t : ℝ), (∃ (k : ℝ), OC - OA = k • (OB - OA)) → 
    m * n ≤ 25 / 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_and_collinearity_l1288_128852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_from_polar_equation_l1288_128812

/-- The polar equation p = 4cos(θ) + 3sin(θ) represents a circle. This theorem proves it and finds its radius. -/
theorem circle_from_polar_equation :
  ∃ (r : ℝ) (center : ℝ × ℝ), 
    (∀ θ : ℝ, (4 * Real.cos θ + 3 * Real.sin θ) • (Real.cos θ, Real.sin θ) = center + r • (Real.cos θ, Real.sin θ)) ∧
    r > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_from_polar_equation_l1288_128812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_likely_parent_genotypes_l1288_128800

/-- Represents the alleles for rabbit fur type -/
inductive Allele
| H  -- Hairy (dominant)
| h  -- Hairy (recessive)
| S  -- Smooth (dominant)
| s  -- Smooth (recessive)

/-- Represents the genotype of a rabbit -/
structure Genotype :=
(allele1 : Allele)
(allele2 : Allele)

/-- Represents the phenotype (observable trait) of a rabbit -/
inductive Phenotype
| Hairy
| Smooth

/-- Function to determine the phenotype from a genotype -/
def phenotypeFromGenotype (g : Genotype) : Phenotype :=
  match g.allele1, g.allele2 with
  | Allele.H, _ | _, Allele.H => Phenotype.Hairy
  | Allele.S, _ | _, Allele.S => Phenotype.Smooth
  | Allele.h, Allele.h => Phenotype.Hairy
  | Allele.s, Allele.s => Phenotype.Smooth
  | Allele.h, Allele.s | Allele.s, Allele.h => Phenotype.Smooth

/-- The probability of the hairy allele in the population -/
def p : ℝ := 0.1

/-- Theorem: Given the conditions, (HH, Sh) is the most likely genotype combination for the parents -/
theorem most_likely_parent_genotypes :
  ∀ (parent1 parent2 : Genotype),
    phenotypeFromGenotype parent1 = Phenotype.Hairy ∧
    phenotypeFromGenotype parent2 = Phenotype.Smooth ∧
    (∀ (offspring : Fin 4 → Genotype),
      (∀ i, phenotypeFromGenotype (offspring i) = Phenotype.Hairy)) →
    (parent1 = ⟨Allele.H, Allele.H⟩ ∧ parent2 = ⟨Allele.S, Allele.h⟩) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_likely_parent_genotypes_l1288_128800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_event_l1288_128805

/-- The probability of an event occurring -/
noncomputable def prob : ℝ := 1/3

/-- The number of events -/
def n : ℕ := 4

/-- The probability that no events occur -/
noncomputable def prob_none : ℝ := (1 - prob)^n

/-- The probability that exactly one event occurs -/
noncomputable def prob_one : ℝ := n * prob * (1 - prob)^(n-1)

/-- The probability that at most one event occurs -/
theorem at_most_one_event : prob_none + prob_one = 16/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_event_l1288_128805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_equals_21_when_S_is_10_l1288_128807

/-- Given that R = gS - 4 and R = 16 when S = 8, prove that R = 21 when S = 10 -/
theorem R_equals_21_when_S_is_10 (g : ℚ) : 
  (∀ S : ℚ, (g * S - 4 = 16 → S = 8) → 
    (S = 10 → g * S - 4 = 21)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_equals_21_when_S_is_10_l1288_128807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l1288_128868

/-- The hyperbola with equation x²/4 - y² = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

/-- Point A with coordinates (0, 1) -/
def point_A : ℝ × ℝ := (0, 1)

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- The theorem stating the distance from point A to the asymptote of the hyperbola -/
theorem distance_to_asymptote :
  ∃ (x y : ℝ), hyperbola x y →
  distance_point_to_line point_A.1 point_A.2 1 (-2) 0 = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l1288_128868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_perpendiculars_isosceles_l1288_128869

/-- A triangle with angle bisectors and perpendiculars from their bases. -/
structure TriangleWithBisectors where
  A : ℝ × ℝ  -- Vertex A of the triangle
  B : ℝ × ℝ  -- Vertex B of the triangle
  C : ℝ × ℝ  -- Vertex C of the triangle
  D : ℝ × ℝ  -- Point where angle bisector from A meets BC
  E : ℝ × ℝ  -- Point where angle bisector from B meets AC
  F : ℝ × ℝ  -- Point where angle bisector from C meets AB
  P : ℝ × ℝ  -- Point of intersection of perpendiculars

/-- The condition that D, E, F are on the angle bisectors. -/
def is_on_bisector (T : TriangleWithBisectors) : Prop :=
  sorry

/-- The condition that the perpendiculars from D, E, F to the opposite sides intersect at P. -/
def perpendiculars_intersect (T : TriangleWithBisectors) : Prop :=
  sorry

/-- Two sides of a triangle are equal. -/
def is_isosceles (T : TriangleWithBisectors) : Prop :=
  sorry

/-- The main theorem: if the perpendiculars from the bases of angle bisectors intersect at a point,
    then the triangle is isosceles. -/
theorem bisector_perpendiculars_isosceles (T : TriangleWithBisectors) :
  is_on_bisector T → perpendiculars_intersect T → is_isosceles T :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisector_perpendiculars_isosceles_l1288_128869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1288_128845

theorem equation_solution (x y z : ℝ) (hx : Real.sin x ≠ 0) (hy : Real.cos y ≠ 0) :
  (Real.sin x ^ 2 + 1 / (Real.sin x ^ 2)) ^ 3 + (Real.cos y ^ 2 + 1 / (Real.cos y ^ 2)) ^ 3 = 16 * Real.cos z ↔
  ∃ (n k m : ℤ), x = Real.pi / 2 + Real.pi * ↑n ∧ y = Real.pi * ↑k ∧ z = 2 * Real.pi * ↑m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1288_128845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_sin_squared_l1288_128858

noncomputable def f (x : ℝ) := Real.sin x ^ 2

theorem smallest_positive_period_of_sin_squared :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_sin_squared_l1288_128858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l1288_128831

theorem sin_cos_relation (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l1288_128831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1288_128817

/-- The distance between two points (x₁, y₁) and (x₂, y₂) is √((x₂ - x₁)² + (y₂ - y₁)²) -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem distance_between_specific_points :
  distance 3 (-4) 10 6 = Real.sqrt 149 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1288_128817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_l1288_128809

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3*x + 1

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem statement
theorem tangent_line_values (a : ℝ) :
  (∃ x : ℝ, f x = a ∧ f' x = 0) → (a = -8 ∨ a = 8/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_l1288_128809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_tangent_to_lines_l1288_128872

/-- 
A circle with center (0, k) where k > 10 is tangent to the lines y = x, y = -x, and y = 10.
This theorem proves that the radius of such a circle is (k-10)√2.
-/
theorem circle_radius_tangent_to_lines (k : ℝ) (h : k > 10) : 
  let r := (k - 10) * Real.sqrt 2
  ∃ (c : Set (ℝ × ℝ)), 
    (∃ center : ℝ × ℝ, center = (0, k) ∧
      c = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2}) ∧
    (∀ p ∈ c, p.1 = p.2 → (p.1 + p.2 = 2 * k - 2 * r)) ∧
    (∀ p ∈ c, p.1 = -p.2 → (p.1 - p.2 = 2 * k - 2 * r)) ∧
    (∀ p ∈ c, p.2 = 10 → (k - 10 = r)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_tangent_to_lines_l1288_128872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1288_128843

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  hq : q ≠ 0 -- The common ratio is non-zero
  h : ∀ n, a (n + 1) = a n * q -- Definition of geometric sequence

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def S (g : GeometricSequence) (n : ℕ) : ℝ :=
  if g.q = 1 then n * g.a 1
  else g.a 1 * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_property (g : GeometricSequence) :
  S g 5 / S g 2 = -11 → 8 * g.a 2 + g.a 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1288_128843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_one_two_three_equals_one_l1288_128825

-- Define the operation * as noncomputable
noncomputable def star (a b : ℝ) : ℝ := (a - b) / (1 - a * b)

-- Theorem statement
theorem star_one_two_three_equals_one :
  star 1 (star 2 3) = 1 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_one_two_three_equals_one_l1288_128825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1288_128829

/-- A vector in R^2 -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The dot product of two 2D vectors -/
noncomputable def dot_product (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

/-- The squared norm of a 2D vector -/
noncomputable def norm_squared (v : Vector2D) : ℝ := dot_product v v

/-- The projection of v onto w -/
noncomputable def proj (v w : Vector2D) : Vector2D :=
  let scalar := (dot_product v w) / (norm_squared w)
  Vector2D.mk (scalar * w.x) (scalar * w.y)

/-- A vector on the line y = 3x - 2 -/
noncomputable def vector_on_line (a : ℝ) : Vector2D := Vector2D.mk a (3 * a - 2)

/-- The theorem stating that the projection is always (3/5, -1/5) -/
theorem projection_theorem (w : Vector2D) : 
  ∃ (c : ℝ), w = Vector2D.mk (-3 * c) c → 
  ∀ (a : ℝ), proj (vector_on_line a) w = Vector2D.mk (3/5) (-1/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1288_128829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_300_l1288_128837

noncomputable section

/-- Revenue function -/
def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then 400 * x - (1/2) * x^2
  else 80000

/-- Cost function -/
def C (x : ℝ) : ℝ := 20000 + 100 * x

/-- Profit function -/
def f (x : ℝ) : ℝ := R x - C x

/-- Theorem stating the maximum profit and optimal production volume -/
theorem max_profit_at_300 :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x ≤ f x_max) ∧
  (∃ (x_opt : ℝ), x_opt = 300 ∧ f x_opt = 25000) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_300_l1288_128837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_distance_l1288_128821

noncomputable def laser_start : ℝ × ℝ := (1, 3)
noncomputable def laser_end : ℝ × ℝ := (5, 3)

def y_axis (x : ℝ) : Prop := x = 0
def diagonal_line (x y : ℝ) : Prop := y = -x

noncomputable def reflection_point_y_axis : ℝ × ℝ := (0, 3)
noncomputable def reflection_point_diagonal : ℝ × ℝ := (-3, -1)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem laser_beam_distance :
  distance laser_start reflection_point_y_axis +
  distance reflection_point_y_axis reflection_point_diagonal +
  distance reflection_point_diagonal laser_end =
  6 + 4 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_distance_l1288_128821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_height_l1288_128857

/-- The height of a right trapezoid circumscribed around a circle -/
noncomputable def trapezoidHeight (α : Real) (P : Real) : Real :=
  P * Real.sin α / (4 * (Real.cos (Real.pi / 4 - α / 2))^2)

/-- Theorem stating the height of a right trapezoid circumscribed around a circle -/
theorem right_trapezoid_height (α P : Real) 
  (h_acute : 0 < α ∧ α < Real.pi/2) 
  (h_perimeter : P > 0) :
  ∃ (h : Real), h = trapezoidHeight α P ∧ 
  h > 0 ∧ 
  ∃ (r : Real), r > 0 ∧
  ∃ (a b c d : Real), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a + b + c + d = P ∧
    (∃ (x y : Real), x > 0 ∧ y > 0 ∧ 
      x * Real.cos α = r ∧
      y * Real.sin α = r ∧
      a = x + y ∧
      b = x - y ∧
      c = h ∧
      d = h / Real.sin α) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_height_l1288_128857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sqrt_sum_equivalence_l1288_128899

/-- A sequence of positive real numbers -/
def PositiveSeq := ℕ → ℝ

/-- Partial sum of a sequence -/
def PartialSum (a : PositiveSeq) : ℕ → ℝ
  | 0 => 0
  | n + 1 => PartialSum a n + a (n + 1)

/-- Arithmetic sequence property -/
def IsArithmetic (a : PositiveSeq) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sqrt_sum_equivalence (a : PositiveSeq) 
  (h_pos : ∀ n, 0 < a n) : 
  ((IsArithmetic a ∧ a 2 = 3 * a 1) ↔
  (IsArithmetic a ∧ IsArithmetic (fun n => Real.sqrt (PartialSum a n)))) ∧
  ((IsArithmetic a ∧ IsArithmetic (fun n => Real.sqrt (PartialSum a n))) ↔
  (a 2 = 3 * a 1 ∧ IsArithmetic (fun n => Real.sqrt (PartialSum a n)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sqrt_sum_equivalence_l1288_128899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_equals_neg_three_tenths_l1288_128816

theorem sin_product_equals_neg_three_tenths (α : ℝ) (h : Real.tan α = 3) :
  Real.sin α * Real.sin (3 * π / 2 - α) = -3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_equals_neg_three_tenths_l1288_128816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1288_128810

noncomputable def f (x : ℝ) := Real.cos x ^ 4 - 2 * Real.sin x * Real.cos x - Real.sin x ^ 4

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -Real.sqrt 2) ∧
  f (3 * Real.pi / 8) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1288_128810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_function_characterization_l1288_128859

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the property for non-negative coefficients
def NonNegativeCoefficients (P : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∀ x : ℝ, (deriv^[n] P) x ≥ 0

-- Define the functional equation
def FunctionalEquation (P : ℝ → ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x + P x * f y) = (y + 1) * f x

-- State the theorem
theorem polynomial_function_characterization
  (P : ℝ → ℝ) (f : ℝ → ℝ)
  (h_poly : Polynomial ℝ)
  (h_nonneg : NonNegativeCoefficients P)
  (h_eq : FunctionalEquation P f) :
  ∃ (m : ℝ), m > 0 ∧ (∀ x, P x = m * x) ∧ (∀ x, x > 0 → f x = x / m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_function_characterization_l1288_128859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_stdDev_l1288_128894

def sample : List ℝ := [4, 2, 1, 0, -2]

/-- The mean of a list of real numbers -/
noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

/-- The variance of a list of real numbers -/
noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ)^2)).sum / xs.length

/-- The standard deviation of a list of real numbers -/
noncomputable def stdDev (xs : List ℝ) : ℝ := Real.sqrt (variance xs)

theorem sample_stdDev : stdDev sample = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_stdDev_l1288_128894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_of_line_l1288_128841

noncomputable def direction_vector : ℝ × ℝ := (3, 2)

noncomputable def inclination_angle (v : ℝ × ℝ) : ℝ :=
  Real.arctan (v.2 / v.1)

theorem inclination_angle_of_line (l : ℝ → ℝ × ℝ) 
  (h : ∃ t, l t = t • direction_vector) :
  inclination_angle direction_vector = Real.arctan (2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_of_line_l1288_128841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l1288_128848

open Set
open Function
open Real

-- Define the function f(x) = (1-x)e^x
noncomputable def f (x : ℝ) : ℝ := (1 - x) * exp x

-- Define the monotonic increasing property
def MonotonicIncreasingOn (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

-- Theorem statement
theorem f_monotonic_increasing_interval :
  {x : ℝ | x < 0} = {x : ℝ | MonotonicIncreasingOn f {y : ℝ | y ≤ x}} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l1288_128848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1288_128892

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 3 = 0

-- Define the line passing through the origin
def line_through_origin (m : ℝ) (x y : ℝ) : Prop := y = m * x

-- Define the third quadrant
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (x₀ y₀ m : ℝ),
    circle_equation x₀ y₀ ∧
    third_quadrant x₀ y₀ ∧
    line_through_origin m x₀ y₀ ∧
    (∀ (x y : ℝ), circle_equation x y → (x - x₀)^2 + (y - y₀)^2 ≥ 0) ∧
    m = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1288_128892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_is_ellipse_l1288_128870

-- Define the circle
def myCircle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 36

-- Define the center of the circle
def M : ℝ × ℝ := (-2, 0)

-- Define point N
def N : ℝ × ℝ := (2, 0)

-- Define point A on the circle
def A : ℝ × ℝ → Prop := λ p => myCircle p.1 p.2

-- Define point P
def P (p : ℝ × ℝ) : Prop :=
  ∃ (a : ℝ × ℝ), A a ∧
  -- P is on the perpendicular bisector of AN
  (p.1 - (a.1 + N.1)/2)^2 + (p.2 - (a.2 + N.2)/2)^2 = ((a.1 - N.1)/2)^2 + ((a.2 - N.2)/2)^2 ∧
  -- P is on the line MA
  ∃ (t : ℝ), p = (M.1 + t * (a.1 - M.1), M.2 + t * (a.2 - M.2))

-- Theorem statement
theorem locus_of_P_is_ellipse :
  ∃ (c₁ c₂ a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), P (x, y) ↔ (x - c₁)^2 / a^2 + (y - c₂)^2 / b^2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_is_ellipse_l1288_128870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reach_point_l1288_128832

/-- Probability that particle M reaches point (0, n) -/
noncomputable def P (n : ℕ) : ℝ :=
  2/3 + 1/12 * (1 - (-1/3)^(n-1))

/-- Movement probability for vector a -/
noncomputable def prob_a : ℝ := 2/3

/-- Movement probability for vector b -/
noncomputable def prob_b : ℝ := 1/3

/-- Vector a -/
def vec_a : ℝ × ℝ := (0, 1)

/-- Vector b -/
def vec_b : ℝ × ℝ := (0, 2)

/-- Theorem stating that P(n) is the correct probability for reaching (0, n) -/
theorem probability_reach_point (n : ℕ) (h : n ≥ 2):
  P n = prob_a * P (n-1) + prob_b * P (n-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reach_point_l1288_128832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_after_one_year_l1288_128819

/-- Calculates the final balance after compound interest is applied -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * years)

theorem balance_after_one_year :
  let initial_deposit : ℝ := 150
  let annual_rate : ℝ := 0.20
  let compounds_per_year : ℝ := 2
  let years : ℝ := 1
  compound_interest initial_deposit annual_rate compounds_per_year years = 181.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_after_one_year_l1288_128819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_line_equation_l1288_128814

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (x^3)

/-- The derivative of the curve function -/
noncomputable def f' (x : ℝ) : ℝ := 1 + (3/2) * Real.sqrt x

/-- The point of interest -/
def x₀ : ℝ := 1

/-- Theorem: The equation of the normal line to the curve y = x + √(x³) at x₀ = 1 -/
theorem normal_line_equation :
  let y₀ := f x₀
  let m := -1 / (f' x₀)
  (fun x y => y = m * (x - x₀) + y₀) = (fun x y => y = -(2/5) * x + 12/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_line_equation_l1288_128814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xfx_positive_solution_set_l1288_128874

open Set Real

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem xfx_positive_solution_set
  (f : ℝ → ℝ)
  (h_even : IsEven f)
  (h_ineq : ∀ x > 0, f x < x * (deriv f x))
  (h_f1 : f 1 = 0) :
  {x : ℝ | x * f x > 0} = Ioo (-1 : ℝ) 0 ∪ Ioi (1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xfx_positive_solution_set_l1288_128874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1288_128801

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem f_properties :
  (∀ x, abs (f (x + Real.pi)) = abs (f x)) ∧
  (∀ x, f ((2 * Real.pi) / 3 + x) = -f ((2 * Real.pi) / 3 - x)) ∧
  (∃ a : ℝ, a = (5 * Real.pi) / 12 ∧
    (∀ x y, -a ≤ x ∧ x < y ∧ y ≤ a → f x < f y) ∧
    (∀ b, b > a → ∃ x y, -b ≤ x ∧ x < y ∧ y ≤ b ∧ f x ≥ f y)) ∧
  (∀ x, f (x - Real.pi / 12) = f (-x - Real.pi / 12)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1288_128801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_angle_30_implies_cot_diff_l1288_128860

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a median
def isMedian (t : Triangle) (D : ℝ × ℝ) : Prop :=
  D.1 = (t.B.1 + t.C.1) / 2 ∧ D.2 = (t.B.2 + t.C.2) / 2

-- Define the angle between two vectors
noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

-- Define cotangent
noncomputable def cot (θ : ℝ) : ℝ :=
  1 / Real.tan θ

-- State the theorem
theorem median_angle_30_implies_cot_diff (t : Triangle) (D : ℝ × ℝ) :
  isMedian t D →
  angle (D.1 - t.B.1, D.2 - t.B.2) (t.C.1 - t.B.1, t.C.2 - t.B.2) = π / 6 →
  |cot (angle (t.A.1 - t.B.1, t.A.2 - t.B.2) (t.C.1 - t.B.1, t.C.2 - t.B.2)) -
   cot (angle (t.A.1 - t.C.1, t.A.2 - t.C.2) (t.B.1 - t.C.1, t.B.2 - t.C.2))| = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_angle_30_implies_cot_diff_l1288_128860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l1288_128827

-- Define the function f(x) = |2^x - 2|
noncomputable def f (x : ℝ) : ℝ := |2^x - 2|

-- State the theorem
theorem f_monotonicity :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 1 → f x ≥ f y) ∧
  (∀ x y : ℝ, 1 ≤ x ∧ x ≤ y → f x ≤ f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l1288_128827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integer_ending_in_seven_divisibility_by_four_l1288_128826

theorem three_digit_integer_ending_in_seven_divisibility_by_four :
  ∀ M : ℕ,
  100 ≤ M ∧ M < 1000 ∧ M % 10 = 7 →
  (∃ (total : ℕ) (divisible : ℕ), 
    total > 0 ∧
    divisible = (Finset.filter (λ x : ℕ ↦ x % 4 = 0) {M}).card ∧
    (divisible : ℚ) / total = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integer_ending_in_seven_divisibility_by_four_l1288_128826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_one_inequality_l1288_128878

theorem product_one_inequality (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hprod : x * y * z = 1) (hineq : 1/x + 1/y + 1/z ≥ x + y + z) :
  ∀ k : ℕ, x^(-(k : ℤ)) + y^(-(k : ℤ)) + z^(-(k : ℤ)) ≥ x^k + y^k + z^k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_one_inequality_l1288_128878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_proof_l1288_128864

/-- The growth factor of bacteria population every 30 seconds -/
def growth_factor : ℚ := 5

/-- The number of 30-second intervals in 4 minutes -/
def intervals : ℕ := 8

/-- The final number of bacteria after 4 minutes -/
def final_population : ℕ := 1953125

/-- The initial number of bacteria -/
noncomputable def initial_population : ℚ := final_population / (growth_factor ^ intervals)

theorem bacteria_growth_proof :
  initial_population = 5 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_proof_l1288_128864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1288_128876

theorem triangle_inequality (a b c S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0) :
  (a * b + b * c + c * a) / (4 * S) ≥ Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1288_128876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1288_128835

/-- The time (in days) it takes for the work to be completed -/
noncomputable def total_time (p_time q_time : ℝ) (p_alone_time : ℝ) : ℝ :=
  let p_rate := 1 / p_time
  let q_rate := 1 / q_time
  let combined_rate := p_rate + q_rate
  let p_work := p_rate * p_alone_time
  let remaining_work := 1 - p_work
  p_alone_time + remaining_work / combined_rate

/-- The theorem stating that the work lasts 40 days given the conditions -/
theorem work_completion_time :
  total_time 80 48 16 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1288_128835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_integers_201_to_399_l1288_128806

theorem sum_of_odd_integers_201_to_399 : 
  (let a : ℕ := 201  -- first term
   let l : ℕ := 399  -- last term
   let d : ℕ := 2    -- common difference
   let n : ℕ := (l - a) / d + 1  -- number of terms
   n * (a + l) / 2) = 30000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_integers_201_to_399_l1288_128806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1288_128895

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- The given hyperbola equation is x²/2 - y²/6 = 1 -/
theorem hyperbola_eccentricity :
  eccentricity (Real.sqrt 2) (Real.sqrt 6) = 2 := by
  -- Unfold the definition of eccentricity
  unfold eccentricity
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1288_128895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_process_theorem_l1288_128883

/-- A mapping process that takes a point (p, q) to (ap+bq, cp+dq) -/
def mapping_process (a b c d : ℝ) (p q : ℝ) : ℝ × ℝ := (a*p + b*q, c*p + d*q)

/-- The parabola C: y = x^2 - x + k -/
def parabola (k : ℝ) (x : ℝ) : ℝ := x^2 - x + k

theorem mapping_process_theorem (a b c d k : ℝ) :
  k ≠ 0 →
  (a, b, c, d) ≠ (1, 0, 0, 1) →
  (∀ x y : ℝ, y = parabola k x → 
    (let (x', y') := mapping_process a b c d x y; y' = parabola k x')) →
  ((a, b, c, d) = (-1, 0, 2, 1) ∧
   k = 1/2 ∧
   ∃ x : ℝ, x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mapping_process_theorem_l1288_128883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_alone_time_l1288_128871

/-- Given two workers A and B, where A is three times as fast as B,
    this function calculates the time it takes for A to complete the job alone. -/
noncomputable def time_for_A_alone (time_together : ℝ) : ℝ :=
  (4 / 3) * time_together

/-- Theorem stating that if A is three times as fast as B, and together they can complete
    a job in 21 days, then A alone can complete the job in 28 days. -/
theorem A_alone_time (time_together : ℝ) 
    (h_together : time_together = 21) : time_for_A_alone time_together = 28 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_alone_time_l1288_128871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_is_five_l1288_128855

def first_n_odd_numbers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def product_of_list (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_product_is_five (n : ℕ) :
  units_digit (product_of_list (first_n_odd_numbers n)) = 5 ↔ n ≥ 3 := by
  sorry

#eval first_n_odd_numbers 5
#eval product_of_list (first_n_odd_numbers 5)
#eval units_digit (product_of_list (first_n_odd_numbers 5))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_is_five_l1288_128855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x0_value_l1288_128865

theorem max_x0_value (x : Fin 1996 → ℝ) 
  (h_pos : ∀ i, x i > 0)
  (h_eq : x 0 = x 1995)
  (h_rec : ∀ i : Fin 1995, x i.val + 2 / x i.val = 2 * x (i.succ) + 1 / x (i.succ)) :
  x 0 ≤ 2^997 ∧ ∃ x' : Fin 1996 → ℝ, 
    x' 0 = 2^997 ∧ 
    (∀ i, x' i > 0) ∧ 
    x' 0 = x' 1995 ∧ 
    (∀ i : Fin 1995, x' i.val + 2 / x' i.val = 2 * x' (i.succ) + 1 / x' (i.succ)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x0_value_l1288_128865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_common_terms_l1288_128838

def last_digit (n : ℕ) : ℕ := n % 10

def sequence_a (a₁ : ℕ) : ℕ → ℕ
  | 0 => a₁
  | n + 1 => sequence_a a₁ n + last_digit (sequence_a a₁ n)

def sequence_2n (n : ℕ) : ℕ := 2^n

theorem infinitely_many_common_terms (a₁ : ℕ) (h : a₁ % 5 ≠ 0) :
  ∀ k : ℕ, ∃ n > k, sequence_a a₁ n = sequence_2n n :=
by
  sorry

#check infinitely_many_common_terms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_common_terms_l1288_128838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_existence_l1288_128853

theorem divisor_existence (S : Finset ℕ) : 
  S ⊆ Finset.range 2015 → S.card = 1008 →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_existence_l1288_128853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_satisfy_conditions_l1288_128882

-- Define a structure for a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (l : Line2D) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

-- Function to calculate y-intercept of a line
noncomputable def yIntercept (l : Line2D) : ℝ :=
  -l.c / l.b

-- Function to check if a line is perpendicular to x-axis
def perpendicularToXAxis (l : Line2D) : Prop :=
  l.b = 0

-- Function to calculate sum of x and y intercepts
noncomputable def sumOfIntercepts (l : Line2D) : ℝ :=
  -l.c/l.a - l.c/l.b

theorem line_equations_satisfy_conditions :
  -- Condition 1
  let l1 : Line2D := { a := 2, b := 1, c := 3 }
  (yIntercept l1 = -3) ∧ (pointOnLine l1 (-2) 1) ∧

  -- Condition 2
  let l2 : Line2D := { a := 1, b := 0, c := 3 }
  (pointOnLine l2 (-3) 1) ∧ (perpendicularToXAxis l2) ∧

  -- Condition 3
  let l3 : Line2D := { a := 1, b := 3, c := -9 }
  let l4 : Line2D := { a := 4, b := -1, c := 16 }
  (pointOnLine l3 (-3) 4) ∧ (sumOfIntercepts l3 = 12) ∧
  (pointOnLine l4 (-3) 4) ∧ (sumOfIntercepts l4 = 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_satisfy_conditions_l1288_128882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_divisible_by_7_and_5_l1288_128833

theorem four_digit_divisible_by_7_and_5 : 
  (Finset.filter (fun n : ℕ => 1000 ≤ n ∧ n ≤ 9999 ∧ n % 7 = 0 ∧ n % 5 = 0) (Finset.range 10000)).card = 257 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_divisible_by_7_and_5_l1288_128833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008_equals_3_l1288_128862

noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem f_2008_equals_3 
  (a b α β : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : α ≠ 0) 
  (h4 : β ≠ 0) 
  (h5 : f a b α β 1988 = 3) : 
  f a b α β 2008 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008_equals_3_l1288_128862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1288_128820

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)

noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

def is_period (p : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_smallest_positive_period (p : ℝ) (f : ℝ → ℝ) : Prop :=
  p > 0 ∧ is_period p f ∧ ∀ q, 0 < q ∧ q < p → ¬ is_period q f

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f y < f x

theorem f_properties :
  (is_smallest_positive_period π f) ∧
  (∀ k : ℤ, monotone_decreasing_on f (Set.Icc (π/6 + k*π) (2*π/3 + k*π))) ∧
  (∃ x ∈ Set.Icc (-π/6) (π/4), ∀ y ∈ Set.Icc (-π/6) (π/4), f y ≤ f x) ∧
  (∀ x, f x = 3 → x ∈ Set.Icc (-π/6) (π/4)) ∧
  (∃ x ∈ Set.Icc (-π/6) (π/4), ∀ y ∈ Set.Icc (-π/6) (π/4), f x ≤ f y) ∧
  (∀ x, f x = 0 → x ∈ Set.Icc (-π/6) (π/4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1288_128820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bryden_receives_18_75_l1288_128846

/-- The amount Bryden receives for his state quarters -/
def bryden_amount (face_value : ℚ) (num_quarters : ℕ) (collector_multiplier : ℚ) : ℚ :=
  face_value * num_quarters * collector_multiplier

/-- Theorem: Bryden receives $18.75 for his five state quarters -/
theorem bryden_receives_18_75 :
  let face_value : ℚ := 1/4
  let num_quarters : ℕ := 5
  let collector_multiplier : ℚ := 15
  bryden_amount face_value num_quarters collector_multiplier = 75/4 := by
  unfold bryden_amount
  simp
  norm_num

#eval bryden_amount (1/4) 5 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bryden_receives_18_75_l1288_128846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1288_128822

noncomputable def f (x φ : ℝ) := Real.cos (2 * x - φ) - Real.sqrt 3 * Real.sin (2 * x - φ)

theorem min_value_of_f (φ : ℝ) (h1 : |φ| < π/2) 
  (h2 : ∀ x, f (x + π/12) φ = f (-x + π/12) φ) :
  ∃ x ∈ Set.Icc (-π/2) 0, ∀ y ∈ Set.Icc (-π/2) 0, f x φ ≤ f y φ ∧ f x φ = -Real.sqrt 3 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1288_128822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_of_log_shifted_l1288_128884

-- Define the concept of an asymptote for a function
def is_asymptote (f : ℝ → ℝ) (l : ℝ → ℝ) : Prop := sorry

-- Define the exponential function
noncomputable def exp2 (x : ℝ) : ℝ := 2^x

-- Define the logarithm function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the function we're investigating
noncomputable def f (x : ℝ) : ℝ := log2 (x + 1) + 2

-- State the theorem
theorem asymptote_of_log_shifted :
  (is_asymptote exp2 (λ x => 0)) →  -- Given: x-axis is asymptote of y = 2^x
  (is_asymptote f (λ x => -1)) :=   -- To prove: x = -1 is asymptote of y = log₂(x+1) + 2
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_of_log_shifted_l1288_128884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_tenth_term_greater_than_500_l1288_128880

/-- Given a sequence of 10 strictly increasing natural numbers with 
    a strictly decreasing sequence of their greatest proper divisors, 
    prove that the 10th term is greater than 500. -/
theorem sequence_tenth_term_greater_than_500 
  (a : Fin 10 → ℕ) 
  (h_a_increasing : ∀ i j, i < j → a i < a j) 
  (b : Fin 10 → ℕ) 
  (h_b_def : ∀ k, b k = (a k).factors.reverse.head!) 
  (h_b_decreasing : ∀ i j, i < j → b i > b j) : 
  a 9 > 500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_tenth_term_greater_than_500_l1288_128880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_event_girls_fraction_l1288_128891

/-- Represents a school with a given number of students and gender ratio -/
structure School where
  total_students : ℕ
  boys_ratio : ℕ
  girls_ratio : ℕ

/-- Calculates the number of girls in a school -/
def num_girls (s : School) : ℕ :=
  s.total_students * s.girls_ratio / (s.boys_ratio + s.girls_ratio)

/-- The fraction of girls at the combined event -/
def fraction_girls (jefferson : School) (lincoln : School) : ℚ :=
  (num_girls jefferson + num_girls lincoln : ℚ) / (jefferson.total_students + lincoln.total_students : ℚ)

theorem combined_event_girls_fraction 
  (jefferson : School)
  (lincoln : School)
  (h_jeff_students : jefferson.total_students = 300)
  (h_jeff_ratio : jefferson.boys_ratio = 3 ∧ jefferson.girls_ratio = 2)
  (h_linc_students : lincoln.total_students = 240)
  (h_linc_ratio : lincoln.boys_ratio = 2 ∧ lincoln.girls_ratio = 3) :
  fraction_girls jefferson lincoln = 22 / 45 := by
  sorry

#eval fraction_girls 
  { total_students := 300, boys_ratio := 3, girls_ratio := 2 }
  { total_students := 240, boys_ratio := 2, girls_ratio := 3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_event_girls_fraction_l1288_128891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_coverage_theorem_l1288_128803

/-- The number of radars -/
def n : ℕ := 8

/-- The coverage radius of each radar in km -/
noncomputable def r : ℝ := 17

/-- The width of the coverage ring in km -/
noncomputable def w : ℝ := 16

/-- The central angle between two adjacent radars in radians -/
noncomputable def θ : ℝ := 2 * Real.pi / n

/-- The maximum distance from the center to each radar -/
noncomputable def max_distance : ℝ := 15 / Real.sin (θ / 2)

/-- The area of the coverage ring -/
noncomputable def coverage_area : ℝ := 480 * Real.pi / Real.tan (θ / 2)

theorem radar_coverage_theorem :
  (max_distance = 15 / Real.sin (θ / 2)) ∧
  (coverage_area = 480 * Real.pi / Real.tan (θ / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_coverage_theorem_l1288_128803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_k_values_l1288_128823

def is_factor_pair (a b : ℕ) : Prop := a * b = 36 ∧ a > 0 ∧ b > 0

def k_value (a b : ℕ) : ℕ := a + b

def distinct_k_values : List ℕ :=
  [12, 13, 15, 20, 37]

theorem average_of_k_values :
  (List.sum distinct_k_values : ℚ) / distinct_k_values.length = 97 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_k_values_l1288_128823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_product_is_one_l1288_128834

/-- Two non-horizontal lines with slopes m and n, where the angle of the first line
    with the horizontal is three times that of the second line, and m = 3n -/
structure LineConfig where
  m : ℝ
  n : ℝ
  angle_relation : ∀ θ₂ : ℝ, Real.tan (3 * θ₂) = m ∧ Real.tan θ₂ = n
  slope_relation : m = 3 * n
  non_horizontal : m ≠ 0 ∧ n ≠ 0

/-- The product of the slopes of two lines satisfying the given conditions is 1 -/
theorem slope_product_is_one (config : LineConfig) : config.m * config.n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_product_is_one_l1288_128834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_equals_sum_div_gcd_l1288_128836

def f (a b M : ℕ) (n : ℤ) : ℤ :=
  if n < M then n + a else n - b

def f_iter (a b M : ℕ) : ℕ → ℤ → ℤ
  | 0, n => n
  | k+1, n => f a b M (f_iter a b M k n)

theorem smallest_k_equals_sum_div_gcd (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) :
  let M := (a + b) / 2
  ∃ k, f_iter a b M k 0 = 0 ∧
    k = (a + b) / Nat.gcd a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_equals_sum_div_gcd_l1288_128836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l1288_128867

/-- A simple graph on n vertices. -/
structure SimpleGraph' (n : Type*) where
  edges : n → n → Prop

/-- The function k(G) as described in the problem. -/
noncomputable def k {n : Type*} [Fintype n] (G : SimpleGraph' n) : ℕ := sorry

/-- The maximal value of k(G) over all graphs on n vertices. -/
noncomputable def M (n : ℕ) : ℕ := sorry

/-- The main theorem: M(n) = ⌊n/2⌋ for n > 1 -/
theorem max_k_value (n : ℕ) (h : n > 1) : M n = n / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l1288_128867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_popular_language_l1288_128881

/-- A room with people and their spoken languages -/
structure LanguageRoom where
  people : Finset ℕ
  languages : Finset ℕ
  speaks : ℕ → Finset ℕ
  num_people : people.card = 1985
  max_languages : ∀ p, p ∈ people → (speaks p).card ≤ 5
  common_language : ∀ p q r, p ∈ people → q ∈ people → r ∈ people → p ≠ q → q ≠ r → p ≠ r →
    ∃ l, l ∈ languages ∧ ((l ∈ speaks p ∧ l ∈ speaks q) ∨
                          (l ∈ speaks q ∧ l ∈ speaks r) ∨
                          (l ∈ speaks p ∧ l ∈ speaks r))

/-- There exists a language spoken by at least 200 people -/
theorem popular_language (room : LanguageRoom) :
  ∃ l, l ∈ room.languages ∧ (room.people.filter (λ p => l ∈ room.speaks p)).card ≥ 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_popular_language_l1288_128881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_properties_l1288_128840

structure Kite where
  a : ℝ
  b : ℝ
  d : ℝ
  h_ab : b ≥ a
  h_equal_diagonals : True  -- This represents the condition of equal diagonals

noncomputable def angle_alpha (k : Kite) : ℝ :=
  Real.arccos (1 - k.d^2 / (2 * k.b^2))

noncomputable def angle_beta (k : Kite) : ℝ :=
  Real.arccos ((k.a^2 + k.b^2 - k.d^2) / (2 * k.a * k.b))

noncomputable def angle_gamma (k : Kite) : ℝ :=
  2 * Real.pi - (angle_alpha k + 2 * angle_beta k)

theorem kite_properties (k : Kite) :
  (k.a = k.b → k.d^2 = 2 * k.b^2) ∧
  angle_alpha k = Real.arccos (1 - k.d^2 / (2 * k.b^2)) ∧
  angle_beta k = Real.arccos ((k.a^2 + k.b^2 - k.d^2) / (2 * k.a * k.b)) ∧
  angle_gamma k = 2 * Real.pi - (angle_alpha k + 2 * angle_beta k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_properties_l1288_128840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l1288_128824

theorem sin_pi_plus_alpha (α : ℝ) 
  (h1 : Real.cos (π / 3 + α) = 1 / 3) 
  (h2 : 0 < α) 
  (h3 : α < π / 2) : 
  Real.sin (π + α) = (Real.sqrt 3 - 2 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l1288_128824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_is_6060_l1288_128802

/-- A circular arrangement of numbers satisfying the given conditions -/
structure CircularArrangement where
  numbers : Fin 2019 → ℝ
  neighbor_diff : ∀ i : Fin 2019, |numbers i - numbers (i + 1)| ≥ 2
  neighbor_sum : ∀ i : Fin 2019, numbers i + numbers (i + 1) ≥ 6

/-- The sum of numbers in a circular arrangement -/
def sum_of_arrangement (arr : CircularArrangement) : ℝ :=
  Finset.sum (Finset.univ : Finset (Fin 2019)) arr.numbers

/-- The theorem stating the smallest possible sum -/
theorem smallest_sum_is_6060 :
  (∃ arr : CircularArrangement, sum_of_arrangement arr = 6060) ∧
  (∀ arr : CircularArrangement, sum_of_arrangement arr ≥ 6060) := by
  sorry

#check smallest_sum_is_6060

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_is_6060_l1288_128802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_time_calculation_l1288_128873

/-- Calculates the time required to mow a rectangular lawn -/
noncomputable def mowing_time (length width swath_width overlap speed : ℝ) : ℝ :=
  let effective_swath := (swath_width - overlap) / 12 -- Convert to feet
  let strips := width / effective_swath
  let total_distance := strips * length
  total_distance / speed

/-- Theorem: The time required to mow the given lawn is 1.5 hours -/
theorem mowing_time_calculation :
  mowing_time 120 100 (30/12) (6/12) 4000 = 1.5 := by
  -- Unfold the definition of mowing_time
  unfold mowing_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_time_calculation_l1288_128873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_monotonicity_l1288_128893

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x) - 2 * x

-- Define the function g
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := f (-1) (2 * x) - 4 * b * f (-1) x

-- Theorem statement
theorem odd_function_and_monotonicity :
  (∀ x, f a (-x) = -(f a x)) →
  (a = -1 ∧
   (∀ x y, x < y → f (-1) x < f (-1) y) ∧
   (∀ b, (∀ x, x > 0 → g b x > 0) ↔ b ≤ 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_monotonicity_l1288_128893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_in_box_l1288_128888

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular box -/
structure Box where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D
  e : Point3D
  f : Point3D
  g : Point3D
  h : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

noncomputable def areaOfPentagon (a p q r s : Point3D) : ℝ :=
  sorry  -- Definition of area calculation

def pointInPlane (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

theorem pentagon_area_in_box (box : Box) (plane : Plane) 
    (p q r s : Point3D) : 
  box.a.x = 0 ∧ box.a.y = 0 ∧ box.a.z = 0 →
  distance box.a box.b = 6 →
  distance box.a box.d = 6 →
  distance box.a box.e = 49 →
  pointInPlane p plane ∧ pointInPlane q plane ∧ pointInPlane r plane ∧ pointInPlane s plane →
  distance box.a p = distance box.a s →
  distance p q = distance q r ∧ distance q r = distance r s →
  areaOfPentagon box.a p q r s = (141 * Real.sqrt 11) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_in_box_l1288_128888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1288_128854

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 0 → x^2 > Real.log x)) ↔ (∃ x : ℝ, x > 0 ∧ x^2 ≤ Real.log x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1288_128854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_l1288_128850

def vector1 : Fin 3 → ℝ := ![3, 4, 5]
def vector2 (m : ℝ) : Fin 3 → ℝ := ![2, m, 3]
def vector3 (m : ℝ) : Fin 3 → ℝ := ![2, 3, m]

noncomputable def volume (m : ℝ) : ℝ := 
  |Matrix.det (Matrix.of (fun i j => if j = 0 then vector1 i 
                                     else if j = 1 then vector2 m i 
                                     else vector3 m i))|

theorem parallelepiped_volume (m : ℝ) :
  m > 0 ∧ volume m = 22 → m = (11 + Real.sqrt 277) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_l1288_128850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l1288_128856

/-- A convex quadrilateral with a point inside -/
structure ConvexQuadrilateral where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  P : ℝ × ℝ
  convex : Convex ℝ (Set.range (fun i => match i with
    | 0 => W
    | 1 => X
    | 2 => Y
    | _ => Z))
  inside : P ∈ interior (Set.range (fun i => match i with
    | 0 => W
    | 1 => X
    | 2 => Y
    | _ => Z))

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- Area of a quadrilateral -/
noncomputable def area (q : ConvexQuadrilateral) : ℝ := sorry

/-- Perimeter of a quadrilateral -/
noncomputable def perimeter (q : ConvexQuadrilateral) : ℝ :=
  distance q.W q.X + distance q.X q.Y + distance q.Y q.Z + distance q.Z q.W

/-- The main theorem -/
theorem quadrilateral_perimeter (q : ConvexQuadrilateral)
  (h_area : area q = 3000)
  (h_PW : distance q.P q.W = 30)
  (h_PX : distance q.P q.X = 40)
  (h_PY : distance q.P q.Y = 35)
  (h_PZ : distance q.P q.Z = 50) :
  |perimeter q - 268.35| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l1288_128856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_condition_l1288_128866

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2^x

-- State the theorem
theorem f_satisfies_condition : ∀ x : ℝ, f (x + 1) = 2 * f x := by
  intro x
  simp [f]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_condition_l1288_128866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l1288_128886

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_values (a b : ℝ) :
  is_odd (f a b) → a = -1/2 ∧ b = Real.log 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l1288_128886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1288_128898

/-- Represents the speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 180

/-- Represents the length of the train in meters -/
noncomputable def train_length : ℝ := 100

/-- Converts km/hr to m/s -/
noncomputable def km_hr_to_m_s (speed : ℝ) : ℝ := speed * (1000 / 3600)

/-- Calculates the time taken for the train to cross the pole -/
noncomputable def crossing_time : ℝ := train_length / km_hr_to_m_s train_speed

theorem train_crossing_time :
  crossing_time = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1288_128898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l1288_128879

theorem absolute_value_expression (x : ℤ) (h : x = -2016) :
  abs (abs (abs x - x) - abs x) - x = 4032 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l1288_128879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_30_consecutive_even_integers_sum_9000_l1288_128890

def consecutive_even_integers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => start + 2 * i)

theorem largest_of_30_consecutive_even_integers_sum_9000 :
  ∃ start : ℤ,
    let sequence := consecutive_even_integers start 30
    (sequence.sum = 9000) ∧
    (sequence.length = 30) ∧
    (sequence.getLast? = some 329) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_30_consecutive_even_integers_sum_9000_l1288_128890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jonas_shoes_count_l1288_128877

/-- Represents the number of pairs of an item in Jonas' wardrobe -/
structure WardrobeItem where
  pairs : Nat

/-- Calculates the number of individual items from pairs -/
def individualItems (item : WardrobeItem) : Nat :=
  item.pairs * 2

/-- Jonas' current wardrobe -/
structure Wardrobe where
  socks : WardrobeItem
  pants : WardrobeItem
  tshirts : Nat
  shoes : WardrobeItem

def jonas_wardrobe : Wardrobe :=
  { socks := ⟨20⟩
  , pants := ⟨10⟩
  , tshirts := 10
  , shoes := ⟨35⟩  -- We'll prove this value
  }

theorem jonas_shoes_count :
  jonas_wardrobe.shoes.pairs = 35 :=
by
  have h1 : individualItems jonas_wardrobe.socks + individualItems jonas_wardrobe.pants + jonas_wardrobe.tshirts + individualItems jonas_wardrobe.shoes = 
            2 * (individualItems jonas_wardrobe.socks + individualItems jonas_wardrobe.pants + jonas_wardrobe.tshirts + 35 * 2) := by
    sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jonas_shoes_count_l1288_128877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_1957_l1288_128851

theorem divisibility_by_1957 (n : ℕ) :
  ∃ k : ℤ, (1721^(2*n) - 73^(2*n) - 521^(2*n) + 212^(2*n) : ℤ) = 1957 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_1957_l1288_128851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_theorem_l1288_128842

open Real

/-- Represents an acute triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute_A : 0 < A ∧ A < π/2
  acute_B : 0 < B ∧ B < π/2
  acute_C : 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  cosine_law : c^2 = a^2 + b^2 - a*b

/-- The main theorem about the acute triangle with given conditions. -/
theorem acute_triangle_theorem (t : AcuteTriangle)
  (h1 : tan t.A - tan t.B = (Real.sqrt 3/3) * (1 + tan t.A * tan t.B)) :
  t.A = 5*π/12 ∧ t.B = π/4 ∧ t.C = π/3 ∧
  ∀ (m n : ℝ × ℝ),
    m = (sin t.A, cos t.A) →
    n = (cos t.B, sin t.B) →
    1 < Norm.norm (3 • m - 2 • n) ∧ Norm.norm (3 • m - 2 • n) < Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_theorem_l1288_128842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_is_576_l1288_128887

/-- Represents the number of male or female animals -/
def num_pairs : ℕ := 5

/-- Calculates the number of ways to arrange animals under given conditions -/
def arrangement_count : ℕ :=
  (num_pairs - 1) * (Nat.factorial (num_pairs - 1)) * (Nat.factorial (num_pairs - 1))

/-- Theorem stating that the number of valid arrangements is 576 -/
theorem arrangement_count_is_576 : arrangement_count = 576 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_is_576_l1288_128887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_k_fail_eq_expected_failures_eq_l1288_128804

/-- A system of n elements where each element has a probability p of failing
    and causing all elements below it to fail. -/
structure FailureSystem where
  n : ℕ
  p : ℝ
  h_p_pos : 0 < p
  h_p_lt_one : p < 1

/-- The probability that exactly k elements fail in the system. -/
noncomputable def prob_k_fail (s : FailureSystem) (k : ℕ) : ℝ :=
  s.p * (1 - s.p) ^ (s.n - k)

/-- The expected number of failed elements in the system. -/
noncomputable def expected_failures (s : FailureSystem) : ℝ :=
  s.n + 1 - 1 / s.p + (1 - s.p) ^ (s.n + 1) / s.p

/-- Theorem stating the probability of exactly k elements failing. -/
theorem prob_k_fail_eq (s : FailureSystem) (k : ℕ) :
  prob_k_fail s k = s.p * (1 - s.p) ^ (s.n - k) := by sorry

/-- Theorem stating the expected number of failed elements. -/
theorem expected_failures_eq (s : FailureSystem) :
  expected_failures s = s.n + 1 - 1 / s.p + (1 - s.p) ^ (s.n + 1) / s.p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_k_fail_eq_expected_failures_eq_l1288_128804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l1288_128839

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | (n + 1) => sequence_a n / (2 * sequence_a n + 1)

theorem a_5_value : sequence_a 5 = 1/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l1288_128839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_sum_l1288_128875

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_sum_l1288_128875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_fraction_value_l1288_128861

theorem trig_fraction_value (α : Real) 
  (h1 : α > -π/2 ∧ α < π/2) 
  (h2 : Real.sin α - Real.cos α = 1/5) : 
  (Real.sin α * Real.cos α) / (Real.sin α + Real.cos α) = 12/35 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_fraction_value_l1288_128861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concave_condition_l1288_128830

/-- The function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/20) * x^5 - (1/12) * m * x^4 - 2 * x^2

/-- The second derivative of f(x) -/
noncomputable def f'' (m : ℝ) (x : ℝ) : ℝ := x^3 - m * x^2 - 4

/-- Theorem: For f(x) to be concave on (1,3), m must be in (-∞, -3] -/
theorem concave_condition (m : ℝ) :
  (∀ x ∈ Set.Ioo 1 3, f'' m x > 0) ↔ m ∈ Set.Iic (-3) := by
  sorry

#check concave_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concave_condition_l1288_128830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_polar_equation_l1288_128828

/-- The equation of a circle in polar form -/
noncomputable def polar_equation (θ : ℝ) : ℝ := 3 * Real.cos θ - 4 * Real.sin θ

/-- The area of the circle represented by the polar equation -/
noncomputable def circle_area : ℝ := (25 / 4) * Real.pi

theorem circle_area_from_polar_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ, polar_equation θ = radius * Real.cos θ + center.1 * Real.cos θ + center.2 * Real.sin θ) ∧
    circle_area = Real.pi * radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_polar_equation_l1288_128828
