import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_covering_convex_polygon_l859_85960

/-- A triangle can cover a convex polygon -/
def covers (t : Set (ℝ × ℝ)) (p : Set (ℝ × ℝ)) : Prop :=
  p ⊆ t

/-- Two triangles are congruent -/
def congruent (t1 t2 : Set (ℝ × ℝ)) : Prop :=
  ∃ f : (ℝ × ℝ) → (ℝ × ℝ), Isometry f ∧ f '' t1 = t2

/-- A set is convex -/
def convex (s : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ × ℝ), x ∈ s → y ∈ s → ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (1 - t) • x + t • y ∈ s

/-- A set is a triangle -/
def is_triangle (t : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c : ℝ × ℝ, t = {a, b, c}

/-- Two line segments are parallel or coincident -/
def parallel_or_coincident (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  ∃ v : ℝ × ℝ, ∀ (x y : ℝ × ℝ), x ∈ s1 → y ∈ s1 → ∃ t : ℝ, y - x = t • v ∧
  ∀ (x y : ℝ × ℝ), x ∈ s2 → y ∈ s2 → ∃ t : ℝ, y - x = t • v

theorem triangle_covering_convex_polygon 
  (ABC : Set (ℝ × ℝ)) (M : Set (ℝ × ℝ)) 
  (h1 : is_triangle ABC) (h2 : convex M) (h3 : covers ABC M) :
  ∃ DEF : Set (ℝ × ℝ), 
    is_triangle DEF ∧ 
    congruent ABC DEF ∧ 
    covers DEF M ∧
    ∃ s1 s2 : Set (ℝ × ℝ), s1 ⊆ DEF ∧ s2 ⊆ M ∧ parallel_or_coincident s1 s2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_covering_convex_polygon_l859_85960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_lap_fraction_l859_85908

/-- Given the conditions of a track running scenario, prove that each lap is 1/6 of a mile. -/
theorem track_lap_fraction (boys_laps girls_extra_laps : ℕ) (girls_total_miles : ℚ)
  (h1 : boys_laps = 34)
  (h2 : girls_extra_laps = 20)
  (h3 : girls_total_miles = 9)
  : (girls_total_miles / (boys_laps + girls_extra_laps : ℚ)) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_lap_fraction_l859_85908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l859_85923

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_calculation (principal time interest : ℝ) 
  (h1 : principal = 800)
  (h2 : time = 4)
  (h3 : interest = 192) :
  simple_interest principal 6 time = interest :=
by
  -- Unfold the definition of simple_interest
  unfold simple_interest
  -- Substitute the known values
  rw [h1, h2, h3]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l859_85923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_eccentricity_l859_85926

-- Define the curve C parametrically
noncomputable def C (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define the eccentricity of an ellipse
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

-- Theorem statement
theorem curve_eccentricity :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ θ : ℝ, (C θ).1^2 / a^2 + (C θ).2^2 / b^2 = 1) ∧
  eccentricity a b = Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_eccentricity_l859_85926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_l859_85943

noncomputable def hyperbola (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 16) - (P.2^2 / 9) = 1

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem hyperbola_distance (P : ℝ × ℝ) :
  hyperbola P →
  distance P (5, 0) = 15 →
  (distance P (-5, 0) = 7 ∨ distance P (-5, 0) = 23) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_l859_85943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_concentration_is_49_percent_l859_85984

/-- Represents a solution with a given volume and concentration -/
structure Solution where
  volume : ℝ
  concentration : ℝ

/-- Calculates the total volume of solute in a solution -/
noncomputable def solute_volume (s : Solution) : ℝ := s.volume * s.concentration / 100

/-- Calculates the percentage concentration of a mixture of solutions -/
noncomputable def mixture_concentration (solutions : List Solution) : ℝ :=
  let total_solute := (solutions.map solute_volume).sum
  let total_volume := (solutions.map (·.volume)).sum
  100 * total_solute / total_volume

/-- The main theorem stating that the mixture concentration is 49% -/
theorem mixture_concentration_is_49_percent : 
  let solutions := [
    ⟨8, 30⟩,
    ⟨5, 50⟩,
    ⟨7, 70⟩
  ]
  mixture_concentration solutions = 49 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_concentration_is_49_percent_l859_85984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simsons_theorem_converse_simsons_theorem_l859_85949

-- Define the circle
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the triangle
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

-- Define predicates
def inscribed_triangle (t : Triangle) (c : Circle) : Prop := sorry

def on_circle (p : EuclideanSpace ℝ (Fin 2)) (c : Circle) : Prop := sorry

def perpendicular (l1 l2 : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

def collinear (p q r : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def concyclic (p q r s : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the Simson's Theorem
theorem simsons_theorem (O : Circle) (ABC : Triangle) (P D E F : EuclideanSpace ℝ (Fin 2)) :
  inscribed_triangle ABC O →
  on_circle P O →
  perpendicular {P, D} {ABC.B, ABC.C} →
  perpendicular {P, E} {ABC.C, ABC.A} →
  perpendicular {P, F} {ABC.A, ABC.B} →
  collinear D E F :=
by sorry

-- Define the Converse of Simson's Theorem
theorem converse_simsons_theorem (O : Circle) (ABC : Triangle) (P D E F : EuclideanSpace ℝ (Fin 2)) :
  inscribed_triangle ABC O →
  perpendicular {P, D} {ABC.B, ABC.C} →
  perpendicular {P, E} {ABC.C, ABC.A} →
  perpendicular {P, F} {ABC.A, ABC.B} →
  collinear D E F →
  concyclic P ABC.A ABC.B ABC.C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simsons_theorem_converse_simsons_theorem_l859_85949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_score_probability_is_35_72_l859_85986

-- Define the dartboard configuration
structure Dartboard where
  outer_radius : ℝ
  inner_radius : ℝ
  num_sections : ℕ

-- Define the point values for each region
def point_values : List ℕ := [1, 2, 2, 2, 1, 1]

-- Define the probability of hitting a region
noncomputable def hit_probability (db : Dartboard) (region : ℕ) : ℝ :=
  if region < db.num_sections then
    (db.inner_radius^2) / (2 * db.outer_radius^2)
  else
    (db.outer_radius^2 - db.inner_radius^2) / (2 * db.outer_radius^2)

-- Define the probability of getting an odd score
noncomputable def odd_score_probability (db : Dartboard) : ℝ :=
  let odd_regions := List.filter (fun x => x % 2 = 1) point_values
  let even_regions := List.filter (fun x => x % 2 = 0) point_values
  let p_odd := List.sum (List.map (hit_probability db) (List.range odd_regions.length))
  let p_even := List.sum (List.map (hit_probability db) (List.range even_regions.length))
  2 * p_odd * p_even

-- Theorem statement
theorem odd_score_probability_is_35_72 (db : Dartboard) :
  db.outer_radius = 6 ∧ db.inner_radius = 3 ∧ db.num_sections = 3 →
  odd_score_probability db = 35 / 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_score_probability_is_35_72_l859_85986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_right_3_equiv_f_x_minus_3_l859_85931

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the transformation
def shift_right (g : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x ↦ g (x - shift)

theorem shift_right_3_equiv_f_x_minus_3 :
  shift_right f 3 = λ x ↦ f (x - 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_right_3_equiv_f_x_minus_3_l859_85931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composed_ten_times_two_equals_two_l859_85979

-- Define the function g based on the graph
noncomputable def g : ℝ → ℝ := fun x =>
  if x = 2 then 2
  else if x = 1 then 7
  else if x = 3 then 5
  else if x = 5 then 3
  else if x = 7 then 1
  else if x = 9 then 0
  else x  -- For all other points, g(x) = x (based on the diagonal line in the graph)

-- Define function composition
def compose (f : ℝ → ℝ) (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ (compose f n)

-- Theorem statement
theorem g_composed_ten_times_two_equals_two :
  compose g 10 2 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composed_ten_times_two_equals_two_l859_85979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_g_piecewise_l859_85994

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := x^2 - t*x + 1

noncomputable def M (t : ℝ) : ℝ := max (f t (-1)) (max (f t 3) (f t (t/2)))
noncomputable def L (t : ℝ) : ℝ := min (f t (-1)) (min (f t 3) (f t (t/2)))

noncomputable def g (t : ℝ) : ℝ := M t - L t

theorem f_inequality_solution (t m : ℝ) (ht : t = 1) (hm : m > 0) :
  ∃ S : Set ℝ, ∀ x : ℝ, x ∈ S ↔ m * (f t x) > x + m - 1 := by
  sorry

theorem g_piecewise (t : ℝ) (ht : t > 0) :
  g t = if t ≤ -2 then -4*t + 8
        else if t ≤ 2 then -t^2/4 - 3*t + 9
        else if t < 6 then t^2/4 + t + 1
        else 4*t - 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_g_piecewise_l859_85994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_mixture_price_l859_85996

theorem rice_mixture_price (price_A price_mix : ℚ) (ratio : ℚ) (price_B : ℚ) : 
  price_A = 31/10 →
  ratio = 7/3 →
  ((3 * price_A + 7 * price_B) / (3 + 7) = price_mix) →
  price_mix = 13/4 →
  ∃ ε > 0, |price_B - 331/100| < ε :=
by sorry

#check rice_mixture_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_mixture_price_l859_85996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_example_l859_85965

theorem cross_product_example :
  let v : Fin 3 → ℝ := ![0, -2, 4]
  let w : Fin 3 → ℝ := ![-1, 0, 6]
  (v 1 * w 2 - v 2 * w 1, v 2 * w 0 - v 0 * w 2, v 0 * w 1 - v 1 * w 0) = (-12, -4, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_example_l859_85965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_pqrs_l859_85983

/-- The area of a square given its vertices. -/
def area_square (A B C D : ℝ × ℝ) : ℝ := sorry

/-- Predicate to check if a triangle is equilateral. -/
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if points are coplanar. -/
def coplanar (A B C D E F G H : ℝ × ℝ) : Prop := sorry

/-- Given a square EFGH with area 25 and equilateral triangles constructed on its sides,
    the area of the square PQRS formed by connecting the outer vertices of these triangles is 50 + 25√3. -/
theorem area_of_pqrs (E F G H P Q R S : ℝ × ℝ) : 
  area_square E F G H = 25 →
  is_equilateral_triangle E P F →
  is_equilateral_triangle F Q G →
  is_equilateral_triangle G R H →
  is_equilateral_triangle H S E →
  coplanar E F G H P Q R S →
  area_square P Q R S = 50 + 25 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_pqrs_l859_85983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cozy_palindromes_characterization_l859_85972

/-- A six-digit number is cozy if AB divides CD and CD divides EF -/
def IsCozy (n : ℕ) : Prop :=
  let a := n / 100000
  let b := (n / 10000) % 10
  let c := (n / 1000) % 10
  let d := (n / 100) % 10
  let e := (n / 10) % 10
  let f := n % 10
  (10 * a + b) ∣ (10 * c + d) ∧ (10 * c + d) ∣ (10 * e + f)

/-- A number is a palindrome if it reads the same forwards and backwards -/
def IsPalindrome (n : ℕ) : Prop :=
  (Nat.digits 10 n).reverse = Nat.digits 10 n

/-- The set of cozy palindromes -/
def CozyPalindromes : Set ℕ :=
  {n : ℕ | 100000 ≤ n ∧ n < 1000000 ∧ IsCozy n ∧ IsPalindrome n}

theorem cozy_palindromes_characterization :
  ∀ n ∈ CozyPalindromes, ∃ d : ℕ, 1 ≤ d ∧ d ≤ 9 ∧ n = d * 111111 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cozy_palindromes_characterization_l859_85972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_sides_l859_85930

/-- Represents a convex polygon with n sides and interior angles in arithmetic progression. -/
structure ConvexPolygon where
  n : ℕ
  smallestAngle : ℚ
  largestAngle : ℚ
  commonDifference : ℚ

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180°. -/
def sumOfInteriorAngles (p : ConvexPolygon) : ℚ :=
  (p.n - 2) * 180

/-- The sum of interior angles calculated using arithmetic progression formula. -/
def sumOfArithmeticProgression (p : ConvexPolygon) : ℚ :=
  p.n * (p.smallestAngle + p.largestAngle) / 2

/-- Theorem stating that a convex polygon with given properties has 8 sides. -/
theorem convex_polygon_sides (p : ConvexPolygon) 
  (h1 : p.smallestAngle = 120)
  (h2 : p.largestAngle = 150)
  (h3 : p.commonDifference = 10)
  (h4 : sumOfInteriorAngles p = sumOfArithmeticProgression p) : 
  p.n = 8 := by
  sorry

#eval ConvexPolygon.n { n := 8, smallestAngle := 120, largestAngle := 150, commonDifference := 10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_sides_l859_85930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_log_function_l859_85963

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) / Real.log 3

theorem domain_of_log_function :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_log_function_l859_85963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_inequality_count_l859_85995

theorem digit_inequality_count : 
  ∃! n : ℕ, n = (Finset.filter (fun d : ℕ => 0 ≤ d ∧ d ≤ 9 ∧ (3 + d * (1/1000 : ℚ)) > (301/100 : ℚ)) (Finset.range 10)).card ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_inequality_count_l859_85995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l859_85952

def total_members : ℕ := 30
def num_boys : ℕ := 12
def num_girls : ℕ := 18
def committee_size : ℕ := 4

theorem committee_probability : 
  (1 : ℚ) - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size : ℚ) / 
    Nat.choose total_members committee_size = 530 / 609 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l859_85952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l859_85999

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  -2 * x^2 + 3 * y^2 - 8 * x - 24 * y + 8 = 0

-- Define the focus coordinates
noncomputable def focus_coordinates : ℝ × ℝ :=
  (-2, 4 + Real.sqrt (80 / 3))

-- Theorem statement
theorem hyperbola_focus :
  ∃ (f : ℝ × ℝ), f = focus_coordinates ∧
  (∀ (x y : ℝ), hyperbola_equation x y → 
    ∃ (c : ℝ), c > 0 ∧
    (x - f.1)^2 / c^2 - (y - f.2)^2 / c^2 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l859_85999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_on_rough_terrain_l859_85958

/-- The total distance covered by a wheel on rough terrain -/
noncomputable def total_distance (radius : ℝ) (revolutions : ℕ) (decrease_percentage : ℝ) : ℝ :=
  2 * Real.pi * radius * (1 - decrease_percentage / 100) * (revolutions : ℝ)

/-- Theorem stating the total distance covered by the wheel -/
theorem wheel_distance_on_rough_terrain :
  let radius : ℝ := 14.6
  let revolutions : ℕ := 100
  let decrease_percentage : ℝ := 12
  abs (total_distance radius revolutions decrease_percentage - 8076.8512) < 0.0001 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_on_rough_terrain_l859_85958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_equal_surface_area_l859_85982

-- Define a regular tetrahedron
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

-- Define a regular octahedron
structure RegularOctahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

-- Function to calculate the surface area of a regular tetrahedron
noncomputable def surface_area_tetrahedron (t : RegularTetrahedron) : ℝ :=
  t.edge_length^2 * Real.sqrt 3

-- Function to calculate the surface area of a regular octahedron
noncomputable def surface_area_octahedron (o : RegularOctahedron) : ℝ :=
  2 * o.edge_length^2 * Real.sqrt 3

-- Function to calculate the volume of a regular tetrahedron
noncomputable def volume_tetrahedron (t : RegularTetrahedron) : ℝ :=
  t.edge_length^3 * Real.sqrt 2 / 12

-- Function to calculate the volume of a regular octahedron
noncomputable def volume_octahedron (o : RegularOctahedron) : ℝ :=
  o.edge_length^3 * Real.sqrt 2 / 3

-- Theorem statement
theorem volume_ratio_equal_surface_area (t : RegularTetrahedron) (o : RegularOctahedron) 
    (h : surface_area_tetrahedron t = surface_area_octahedron o) :
    volume_tetrahedron t / volume_octahedron o = 1 / Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_equal_surface_area_l859_85982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_necessary_not_sufficient_l859_85910

open Real

/-- The function f(x) obtained after shifting sin(2x + φ) left by π/8 units -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := sin (2 * x + π / 4 + φ)

/-- f is an even function -/
def is_even (φ : ℝ) : Prop := ∀ x, f φ x = f φ (-x)

/-- φ = π/4 is necessary but not sufficient for f to be even -/
theorem phi_necessary_not_sufficient :
  (∀ φ, is_even φ → φ = π/4) ∧ 
  ¬(∀ φ, φ = π/4 → is_even φ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_necessary_not_sufficient_l859_85910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendly_group_theorem_l859_85981

/-- A structure representing a group with friendly and hostile relationships -/
structure FriendlyGroup where
  n : ℕ  -- number of people
  q : ℕ  -- number of friendly pairs
  is_valid : q ≤ n.choose 2  -- ensure q is not greater than maximum possible pairs
  is_hostile : Fin n → Fin n → Prop  -- predicate for hostility between two people
  hostile_triple : ∀ (a b c : Fin n), a ≠ b → b ≠ c → a ≠ c → 
    (is_hostile a b) ∨ (is_hostile b c) ∨ (is_hostile a c)

/-- Helper function to count friendly pairs among enemies of a given person -/
def number_of_friendly_pairs_among_enemies (G : FriendlyGroup) (v : Fin G.n) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem friendly_group_theorem (G : FriendlyGroup) :
  ∃ v : Fin G.n, 
    (number_of_friendly_pairs_among_enemies G v) ≤ G.q * (1 - 4 * G.q / G.n^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendly_group_theorem_l859_85981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circles_line_equation_l859_85912

/-- Given two circles that are symmetric with respect to a line, 
    prove that the line has a specific equation. -/
theorem symmetric_circles_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ l → (x^2 + y^2 = 1 ↔ x^2 + y^2 + 4*x - 4*y + 7 = 0)) →
  (∃ (a b c : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ a*x + b*y + c = 0) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x - y + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circles_line_equation_l859_85912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approx_l859_85903

/-- The cost price of an article -/
noncomputable def cost_price : ℝ := sorry

/-- The marked price of an article -/
noncomputable def marked_price : ℝ := sorry

/-- The selling price is 94% of the marked price and 125% of the cost price -/
axiom selling_price_equation : 0.94 * marked_price = 1.25 * cost_price

/-- The selling price is approximately 63.16 -/
axiom selling_price_approx : ∃ ε > 0, |0.94 * marked_price - 63.16| < ε

/-- The cost price is approximately 50.56 -/
theorem cost_price_approx : ∃ ε > 0, |cost_price - 50.56| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approx_l859_85903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_n_solutions_l859_85907

noncomputable def count_solutions (n : ℕ) : ℕ :=
  (2 * Int.floor (Real.sqrt (n : ℝ))).toNat

theorem exact_n_solutions (n : ℕ) : 
  (n = 0 ∨ n = 1 ∨ n = 3 ∨ n = 5) ↔ count_solutions n = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_n_solutions_l859_85907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_two_l859_85933

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_domain : Set.range f = Set.Icc 0 3
axiom f_has_inverse : Function.Bijective f

-- Define the inverse function properties
axiom inverse_property_1 : Set.image f⁻¹ (Set.Ico 0 1) = Set.Ico 1 2
axiom inverse_property_2 : Set.image f⁻¹ (Set.Ioc 2 4) = Set.Ico 0 1

-- Define the existence of a solution
axiom solution_exists : ∃ x₀, f x₀ = x₀

-- State the theorem
theorem unique_solution_is_two :
  ∃! x₀, f x₀ = x₀ ∧ x₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_two_l859_85933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_sum_l859_85993

/-- Represents a rectangle with horizontal and vertical sides -/
structure Rectangle where
  horizontal : ℝ
  vertical : ℝ

/-- Represents a square divided into red and blue rectangles -/
structure DividedSquare where
  side_length : ℝ
  blue_rectangles : List Rectangle
  red_rectangles : List Rectangle

/-- The sum of areas of red rectangles equals the sum of areas of blue rectangles -/
def equal_area_sum (s : DividedSquare) : Prop :=
  (s.blue_rectangles.map (λ r => r.horizontal * r.vertical)).sum =
  (s.red_rectangles.map (λ r => r.horizontal * r.vertical)).sum

/-- The sum of ratios for blue and red rectangles -/
noncomputable def ratio_sum (s : DividedSquare) : ℝ :=
  (s.blue_rectangles.map (λ r => r.vertical / r.horizontal)).sum +
  (s.red_rectangles.map (λ r => r.horizontal / r.vertical)).sum

/-- The theorem stating the minimum ratio sum -/
theorem min_ratio_sum (s : DividedSquare) (h : equal_area_sum s) :
  ratio_sum s ≥ 5/2 ∧ ∃ (s' : DividedSquare), equal_area_sum s' ∧ ratio_sum s' = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_sum_l859_85993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_parallelogram_quadrilateral_l859_85955

/-- Given a parallelogram with adjacent angle ratio 3:8 and a quadrilateral with angle ratio 3:4:7:10,
    the sum of the larger angle of the parallelogram and the largest angle of the quadrilateral
    is approximately 280.91 degrees. -/
theorem angle_sum_parallelogram_quadrilateral :
  ∀ (p_small p_large q1 q2 q3 q4 : ℝ),
  p_small + p_large = 180 →
  p_large / p_small = 8 / 3 →
  q1 + q2 + q3 + q4 = 360 →
  q1 / 3 = q2 / 4 ∧ q2 / 4 = q3 / 7 ∧ q3 / 7 = q4 / 10 →
  abs ((p_large + max q1 (max q2 (max q3 q4))) - 280.91) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_parallelogram_quadrilateral_l859_85955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_square_pairs_l859_85924

/-- For a positive integer n, a_n is defined as (5 * 10^(n-1) - 1)^2 -/
def a_n (n : ℕ) : ℕ := (5 * 10^(n-1) - 1)^2

/-- For a positive integer n, b_n is defined as (10^n - 1)^2 -/
def b_n (n : ℕ) : ℕ := (10^n - 1)^2

/-- The concatenation of two natural numbers -/
def concatenate (a b : ℕ) : ℕ := a * 10^(Nat.log 10 b + 1) + b

/-- Theorem: For all positive integers n, there exist perfect squares a_n and b_n
    such that they have the same number of digits and their concatenation is a perfect square -/
theorem infinite_square_pairs (n : ℕ) (hn : n > 0) :
  ∃ (k : ℕ), (Nat.log 10 (a_n n) = Nat.log 10 (b_n n)) ∧
              (∃ (m : ℕ), concatenate (a_n n) (b_n n) = m^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_square_pairs_l859_85924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l859_85957

/-- The repeating decimal 0.3̄6 is equal to the fraction 11/30 -/
theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 11 / 30) ∧ (x * 30 = 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l859_85957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_tangent_properties_l859_85942

-- Define the points
def E : ℝ × ℝ := (-2, 0)
def F : ℝ × ℝ := (2, 0)
def A : ℝ × ℝ := (2, 1)

-- Define the curve C
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Define the condition for points on curve C
def on_curve (M : ℝ × ℝ) : Prop :=
  (M.1 + 2) * (M.1 - 2) + M.2 * M.2 = -3

-- Define the point P
variable (a b : ℝ)
def P : ℝ × ℝ := (a, b)

-- Define the tangent point Q
variable (Q : ℝ × ℝ)

-- State the theorem
theorem curve_and_tangent_properties :
  (∀ M, M ∈ C → on_curve M) ∧
  (∃ Q ∈ C, ‖P a b - Q‖ = ‖P a b - A‖) →
  (C = {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}) ∧
  (∃ min_length, min_length = 2 / Real.sqrt 5 ∧
    ∀ Q' ∈ C, ‖P a b - Q'‖ ≥ min_length) ∧
  (∃ r_min, (a - 6/5)^2 + (b - 3/5)^2 = r_min^2 ∧
    r_min = 3 / Real.sqrt 5 - 1 ∧
    ∀ r, (∃ Q' ∈ C, ‖P a b - Q'‖ = r) → r ≥ r_min) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_tangent_properties_l859_85942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_047_fraction_l859_85962

def repeating_decimal_to_fraction (a b : ℕ) : ℚ :=
  (a : ℚ) / ((10 ^ b - 1) : ℚ)

theorem decimal_047_fraction :
  ∃ (n : ℕ), repeating_decimal_to_fraction 47 2 = n / 99 ∧ Nat.Coprime n 99 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_047_fraction_l859_85962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_ratio_l859_85947

/-- Right triangle with 60° and 30° acute angles and two inscribed circles -/
structure SpecialTriangle where
  /-- The right angle vertex -/
  B : ℝ × ℝ
  /-- The 60° angle vertex -/
  A : ℝ × ℝ
  /-- The 30° angle vertex -/
  C : ℝ × ℝ
  /-- First circle center -/
  O₁ : ℝ × ℝ
  /-- Second circle center -/
  O₂ : ℝ × ℝ
  /-- Radius of the inscribed circles -/
  r : ℝ
  /-- The triangle is right-angled at B -/
  right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0
  /-- The angle at A is 60° -/
  angle_A_60 : Real.cos (π / 3) = ((C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2)) / 
    (Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2))
  /-- The angle at C is 30° -/
  angle_C_30 : Real.cos (π / 6) = ((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2)) / 
    (Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) * Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2))
  /-- The circles are congruent -/
  circles_congruent : r > 0
  /-- The circles touch the hypotenuse -/
  touch_hypotenuse : (O₁.1 - A.1) * (B.2 - A.2) - (O₁.2 - A.2) * (B.1 - A.1) = r * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) ∧
                     (O₂.1 - A.1) * (B.2 - A.2) - (O₂.2 - A.2) * (B.1 - A.1) = r * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  /-- The circles touch each other -/
  touch_each_other : (O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2 = (2*r)^2
  /-- The circles touch one leg each -/
  touch_legs : (O₁.1 - B.1) * (C.2 - B.2) - (O₁.2 - B.2) * (C.1 - B.1) = r * Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) ∧
               (O₂.1 - A.1) * (C.2 - A.2) - (O₂.2 - A.2) * (C.1 - A.1) = r * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)

/-- The ratio of the smaller leg to the radius of the circles is 2 + √3 -/
theorem special_triangle_ratio (t : SpecialTriangle) :
  Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2) / t.r = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_ratio_l859_85947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sdr_count_l859_85925

/-- Represents the array pattern described in the problem -/
def ArrayPattern (n : ℕ) : List (List ℕ) :=
  [ List.range n,
    [2, 3, 1] ++ List.range (n - 3) |>.map (λ x => x + 4),
    [3, 1, 2] ++ List.range (n - 3) |>.map (λ x => x + 5) ]

/-- Definition of α₁ -/
noncomputable def α₁ : ℝ := (1 + Real.sqrt 5) / 2

/-- Definition of α₂ -/
noncomputable def α₂ : ℝ := (1 - Real.sqrt 5) / 2

/-- The number of SDRs for the given array pattern -/
noncomputable def numSDRs (n : ℕ) : ℝ := 6 * (α₁ ^ (n - 3) + α₂ ^ (n - 3) + 2)

/-- Count of SDRs for a given list of lists -/
def SDRCount (lists : List (List ℕ)) : ℕ := sorry

/-- Theorem stating the number of SDRs for the given array pattern -/
theorem sdr_count (n : ℕ) (h : n ≥ 6) :
  (SDRCount (ArrayPattern n)) = ⌊numSDRs n⌋ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sdr_count_l859_85925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_200_tons_l859_85980

noncomputable section

/-- The price per ton as a function of quantity produced -/
def price (x : ℝ) : ℝ := 24200 - (1/5) * x^2

/-- The cost of production as a function of quantity produced -/
def cost (x : ℝ) : ℝ := 50000 + 200 * x

/-- The profit as a function of quantity produced -/
def profit (x : ℝ) : ℝ := x * price x - cost x

/-- The statement that the maximum profit occurs at 200 tons with a value of 12700000 yuan -/
theorem max_profit_at_200_tons :
  (∀ x : ℝ, x ≥ 0 → profit x ≤ profit 200) ∧
  profit 200 = 12700000 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_200_tons_l859_85980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_coordinates_l859_85988

noncomputable def f (x : ℝ) := 1 - (x + 1)^2 / 3

theorem intersection_sum_coordinates :
  ∃ (a b : ℝ), 
    f a = 2 * f (a - 2) ∧ 
    b = f a ∧
    a + b = -((6 + Real.sqrt 5) / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_coordinates_l859_85988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_participants_l859_85998

theorem competition_participants (total_score : ℚ) (n : ℕ) : 
  (n ≠ 0) →
  (10350 * n ≤ 1000 * total_score) →
  (1000 * total_score < 10450 * n) →
  (10550 * n ≤ 1000 * (total_score + 4)) →
  (1000 * (total_score + 4) < 10650 * n) →
  (14 ≤ n ∧ n ≤ 39) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_participants_l859_85998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percent_l859_85990

noncomputable def profit_percent (purchase_price repair_cost selling_price : ℝ) : ℝ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

theorem car_profit_percent :
  let purchase_price := (42000 : ℝ)
  let repair_cost := (13000 : ℝ)
  let selling_price := (64900 : ℝ)
  profit_percent purchase_price repair_cost selling_price = 18 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percent_l859_85990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_relationship_l859_85934

/-- An inverse proportion function passing through (-2, 3) -/
noncomputable def f (x : ℝ) : ℝ := -6 / x

/-- y₁ is the y-coordinate when x = -3 -/
noncomputable def y₁ : ℝ := f (-3)

/-- y₂ is the y-coordinate when x = 1 -/
noncomputable def y₂ : ℝ := f 1

/-- y₃ is the y-coordinate when x = 2 -/
noncomputable def y₃ : ℝ := f 2

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem inverse_proportion_relationship : y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_relationship_l859_85934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elmo_club_existence_l859_85966

open Function Set Finset

-- Define the graph type
structure Graph (α : Type*) where
  vertices : Set α
  edges : Set (α × α)

-- Define the clique property
def is_clique {α : Type*} (G : Graph α) (S : Set α) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≠ y → (x, y) ∈ G.edges

theorem elmo_club_existence
  (n : ℕ)
  (hn : n ≥ 3)
  (G : Graph (Fin (n^3)))
  (triangle_existence : ∀ (S : Finset (Fin (n^3))), S.card = n → ∃ T ⊆ S, T.card = 3 ∧ is_clique G (T : Set (Fin (n^3))))
  : ∃ S : Set (Fin (n^3)), S.ncard = 5 ∧ is_clique G S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elmo_club_existence_l859_85966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_l859_85938

/-- Represents the number of pages read on a given day -/
def pages_read (total : ℚ) (fraction : ℚ) (additional : ℕ) : ℚ :=
  fraction * total + additional

/-- Represents the number of pages remaining after reading -/
def pages_remaining (total : ℚ) (read : ℚ) : ℚ :=
  total - read

/-- The book reading problem -/
theorem book_pages : ∃ (x : ℚ),
  (let day1_read := pages_read x (1/4) 10
   let day1_remaining := pages_remaining x day1_read
   let day2_read := pages_read day1_remaining (1/5) 20
   let day2_remaining := pages_remaining day1_remaining day2_read
   let day3_read := pages_read day2_remaining (1/2) 25
   let day3_remaining := pages_remaining day2_remaining day3_read
   day3_remaining = 75 ∧ x = 380) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_l859_85938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l859_85927

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 4 = 0

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define the condition for P being inside the circle
def inside_circle (x y : ℝ) : Prop := x^2 + y^2 < 4

-- Define the geometric sequence condition
def geometric_sequence (x y : ℝ) : Prop :=
  ((x + 2)^2 + y^2) * ((x - 2)^2 + y^2) = (x^2 + y^2)^2

-- Define the dot product PA · PB
def dot_product (x y : ℝ) : ℝ := x^2 + y^2 - 4

theorem dot_product_range :
  ∀ x y : ℝ,
  circle_equation x y →
  tangent_line x y →
  inside_circle x y →
  geometric_sequence x y →
  -2 ≤ dot_product x y ∧ dot_product x y < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l859_85927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ada_original_seat_l859_85900

/-- Represents the seats in the row --/
inductive Seat
  | one
  | two
  | three
  | four
  | five
  | six
deriving Repr, DecidableEq

/-- Represents the friends --/
inductive Friend
  | ada
  | bea
  | ceci
  | dee
  | edie
  | fay
deriving Repr, DecidableEq

/-- Represents the seating arrangement --/
def Arrangement := Friend → Seat

/-- Defines the movement of a friend --/
def move (f : Friend) : ℤ :=
  match f with
  | Friend.bea => 1
  | Friend.ceci => -2
  | Friend.dee => 0
  | Friend.edie => 0
  | Friend.fay => 1
  | Friend.ada => 0

/-- Converts a Seat to a natural number --/
def Seat.toNat : Seat → ℕ
  | Seat.one => 1
  | Seat.two => 2
  | Seat.three => 3
  | Seat.four => 4
  | Seat.five => 5
  | Seat.six => 6

/-- The initial arrangement of friends --/
def initial_arrangement : Arrangement := sorry

/-- The final arrangement after movements --/
def final_arrangement : Arrangement := sorry

/-- Theorem stating Ada's original seat --/
theorem ada_original_seat :
  initial_arrangement Friend.ada = Seat.six ∧
  (∀ f : Friend, f ≠ Friend.ada →
    (Seat.toNat (final_arrangement f) - Seat.toNat (initial_arrangement f) = move f)) ∧
  (final_arrangement Friend.ada = Seat.one ∨ final_arrangement Friend.ada = Seat.six) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ada_original_seat_l859_85900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_sine_two_pi_is_min_positive_period_l859_85968

-- Define the sine function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- State the theorem about the minimum positive period of sine
theorem min_positive_period_of_sine :
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = 2 * Real.pi := by
  -- The proof is omitted
  sorry

-- Additional theorem to explicitly state that 2π is the minimum positive period
theorem two_pi_is_min_positive_period :
  let p := 2 * Real.pi
  p > 0 ∧
  (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) := by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_sine_two_pi_is_min_positive_period_l859_85968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_specific_isosceles_triangle_l859_85969

/-- An isosceles triangle with specific side lengths -/
structure IsoscelesTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  -- Isosceles condition
  isIsosceles : ab > 0 ∧ bc > 0

/-- The area of an isosceles triangle -/
noncomputable def areaIsoscelesTriangle (t : IsoscelesTriangle) : ℝ :=
  let height := Real.sqrt (t.ab^2 - (t.bc/2)^2)
  (1/2) * t.bc * height

/-- Theorem: The area of the specific isosceles triangle is 240 -/
theorem area_specific_isosceles_triangle :
  let t : IsoscelesTriangle := ⟨26, 20, by sorry⟩
  areaIsoscelesTriangle t = 240 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_specific_isosceles_triangle_l859_85969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l859_85997

-- Define the sets A and B
def A : Set ℝ := {x | x^3 - 3*x^2 - x + 3 < 0}
def B : Set ℝ := {x | |x + 1/2| ≥ 1}

-- Define the union of sets A and B
def AUnionB : Set ℝ := A ∪ B

-- Define the expected result using Set.Ici and Set.Iic
def ExpectedUnion : Set ℝ := Set.Iic (-1) ∪ Set.Ici (1/2)

-- Theorem statement
theorem union_of_A_and_B : AUnionB = ExpectedUnion := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l859_85997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parentheses_removal_l859_85911

theorem parentheses_removal : 7 - (-5) + (-7) - 3 = 7 + 5 - 7 - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parentheses_removal_l859_85911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_percentage_approx_l859_85919

/-- Represents the cost of items in their original currencies -/
structure OriginalCosts where
  bread : ℝ  -- in yen
  ham : ℝ    -- in euros
  cake : ℝ   -- in kr
  cheese : ℝ  -- in USD

/-- Represents the exchange rates to USD -/
structure ExchangeRates where
  yen_to_usd : ℝ
  euro_to_usd : ℝ
  kr_to_usd : ℝ

/-- Represents the discount percentages -/
structure Discounts where
  ham : ℝ
  cake : ℝ

noncomputable def calculate_total_percentage (costs : OriginalCosts) (rates : ExchangeRates) (discounts : Discounts) : ℝ :=
  sorry

theorem total_percentage_approx (costs : OriginalCosts) (rates : ExchangeRates) (discounts : Discounts) :
  costs.bread = 1000 ∧
  costs.ham = 20 ∧
  costs.cake = 100 ∧
  costs.cheese = 112.5 ∧
  rates.yen_to_usd = 0.010 ∧
  rates.euro_to_usd = 1.1 ∧
  rates.kr_to_usd = 0.15 ∧
  discounts.ham = 0.1 ∧
  discounts.cake = 0.2
  →
  |calculate_total_percentage costs rates discounts - 92.21| < 0.01 :=
by
  sorry

#check total_percentage_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_percentage_approx_l859_85919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l859_85902

-- Define the constants and functions
noncomputable def x : ℝ := (3 + Real.sqrt 8) ^ 500
noncomputable def n : ℤ := ⌊x⌋
noncomputable def f : ℝ := x - n

-- State the theorem
theorem x_times_one_minus_f_equals_one : x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l859_85902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disliked_by_both_percentage_l859_85978

def total_comics : ℕ := 300
def female_like_percent : ℚ := 30 / 100
def male_like_count : ℕ := 120

theorem disliked_by_both_percentage : 
  (total_comics - (female_like_percent * ↑total_comics).floor - male_like_count) / total_comics = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disliked_by_both_percentage_l859_85978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_b_in_range_l859_85950

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x + b / x

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y ∨ (∀ x y, a < x ∧ x < y ∧ y < b → f x > f y)

theorem f_monotonic_iff_b_in_range (b : ℝ) :
  monotonic_on (f b) 1 (Real.exp 1) ↔ b ≤ 1 ∨ b ≥ Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_b_in_range_l859_85950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seq_divisibility_l859_85951

def seq : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => 2 * seq (n + 1) + seq n

theorem seq_divisibility (k n : ℕ) :
  (2^k : ℤ) ∣ (seq n) ↔ 2^k ∣ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seq_divisibility_l859_85951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_y_minus_one_l859_85921

def y : ℕ → ℕ
  | 0 => 217  -- We define y(0) to be 217 to match y₁ in the original problem
  | k + 1 => y k * y k - y k

theorem sum_reciprocal_y_minus_one :
  let series := fun n => 1 / ((y n : ℝ) - 1)
  (∑' n, series (n + 1)) = 1 / (y 0 : ℝ) := by
  sorry

#eval y 0  -- Should output 217
#eval y 1  -- Should output 46872

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_y_minus_one_l859_85921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_InfinitelyManyValidColorings_l859_85976

-- Define a coloring as a function from positive integers to a color (blue or red)
def Coloring := ℕ+ → Bool

-- Define a valid coloring
def IsValidColoring (c : Coloring) : Prop :=
  ∀ a b : ℕ+, c a = c b → (a : ℕ) > 10 * b → c a = c (a - 10 * b)

-- Theorem statement
theorem InfinitelyManyValidColorings : 
  ∃ f : ℕ → Coloring, (∀ i j : ℕ, i ≠ j → f i ≠ f j) ∧ (∀ i : ℕ, IsValidColoring (f i)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_InfinitelyManyValidColorings_l859_85976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_triangle_l859_85956

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_perimeter_of_triangle : 
  ∀ x : ℕ, 
  3 < x → x < 15 → 
  is_valid_triangle 7 8 (x : ℝ) → 
  (∀ y : ℕ, 3 < y → y < 15 → is_valid_triangle 7 8 (y : ℝ) → 
    7 + 8 + (x : ℝ) ≥ 7 + 8 + (y : ℝ)) →
  7 + 8 + (x : ℝ) = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_triangle_l859_85956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_increase_l859_85975

/-- Represents the price of a stock over three years -/
structure StockPrice where
  initial : ℝ
  year1 : ℝ
  year2 : ℝ
  year3 : ℝ

/-- Calculates the percentage change between two prices -/
noncomputable def percentageChange (old : ℝ) (new : ℝ) : ℝ :=
  (new - old) / old * 100

/-- Theorem: Given the specified price changes over three years,
    the percentage increase in the third year is 35% -/
theorem stock_price_increase (p : StockPrice) 
  (h1 : p.year1 = p.initial * 1.20)
  (h2 : p.year2 = p.year1 * 0.75)
  (h3 : p.year3 = p.initial * 1.215) :
  percentageChange p.year2 p.year3 = 35 := by
  sorry

-- Remove the #eval line as it's not necessary for this theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_increase_l859_85975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_segment_length_l859_85906

/-- Represents a convex quadrilateral ABCD with point E at the intersection of diagonals -/
structure ConvexQuadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  convex : Bool
  diagonals_intersect : Bool

/-- The length of a line segment between two points -/
noncomputable def length (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The area of a triangle given three points -/
noncomputable def triangle_area (p q r : ℝ × ℝ) : ℝ := 
  (1/2) * abs ((q.1 - p.1) * (r.2 - p.2) - (r.1 - p.1) * (q.2 - p.2))

theorem quadrilateral_diagonal_segment_length 
  (ABCD : ConvexQuadrilateral)
  (h1 : length ABCD.A ABCD.B = 10)
  (h2 : length ABCD.C ABCD.D = 15)
  (h3 : length ABCD.A ABCD.C = 17)
  (h4 : triangle_area ABCD.A ABCD.E ABCD.D = triangle_area ABCD.B ABCD.E ABCD.C) :
  length ABCD.A ABCD.E = 34/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_segment_length_l859_85906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_six_given_at_least_four_l859_85991

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of getting exactly k heads when flipping n coins -/
noncomputable def prob_exactly (n k : ℕ) : ℚ := (binomial n k : ℚ) / (2^n : ℚ)

/-- The probability of getting at least k heads when flipping n coins -/
noncomputable def prob_at_least (n k : ℕ) : ℚ := 
  Finset.sum (Finset.range (n - k + 1)) (λ i => prob_exactly n (k + i))

/-- The main theorem: probability of 6 heads given at least 4 heads when flipping 10 coins -/
theorem prob_six_given_at_least_four :
  prob_exactly 10 6 / prob_at_least 10 4 = 105 / 424 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_six_given_at_least_four_l859_85991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l859_85920

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 6*y - 39 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (2, 1)
def center2 : ℝ × ℝ := (-1, -3)
def radius1 : ℝ := 2
def radius2 : ℝ := 7

-- Define the distance between the centers
noncomputable def distance : ℝ := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)

-- Theorem: The circles are internally tangent
theorem circles_internally_tangent : distance = radius2 - radius1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l859_85920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_symmetry_l859_85985

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

noncomputable def translated_function (x ρ : ℝ) : ℝ := Real.sin (2 * x + 2 * ρ + Real.pi / 3)

theorem translation_symmetry :
  ∃ ρ : ℝ, 
    (∀ x : ℝ, translated_function x ρ = translated_function (-Real.pi/6 - x) ρ) ∧ 
    (translated_function (-Real.pi/12) ρ = 0) ∧
    (ρ = Real.pi/12) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_symmetry_l859_85985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_3_l859_85992

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 1) / (4*x - 5)

-- State the theorem
theorem f_at_3 : f 3 = 16/7 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [pow_two]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_3_l859_85992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_is_four_l859_85901

/-- Represents a circular sector -/
structure CircularSector where
  radius : ℝ
  arcLength : ℝ
  centralAngle : ℝ

/-- Calculates the perimeter of a circular sector -/
noncomputable def perimeter (s : CircularSector) : ℝ := s.arcLength + 2 * s.radius

/-- Calculates the area of a circular sector -/
noncomputable def area (s : CircularSector) : ℝ := (1/2) * s.radius * s.arcLength

/-- Theorem: A circular sector with perimeter 8 and central angle 2 radians has an area of 4 -/
theorem sector_area_is_four :
  ∀ s : CircularSector,
    perimeter s = 8 →
    s.centralAngle = 2 →
    area s = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_is_four_l859_85901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_sum_l859_85937

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.cos (2 * θ) = 1/2) :
  (Real.sin θ) ^ 6 + (Real.cos θ) ^ 6 = 0.8125 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_sum_l859_85937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_pairs_l859_85941

/-- Represents the total number of socks -/
def total_socks : ℕ := 10

/-- Represents the number of blue sock pairs -/
def blue_pairs : ℕ := 2

/-- Represents the number of red sock pairs -/
def red_pairs : ℕ := 2

/-- Represents the number of yellow sock pairs -/
def yellow_pairs : ℕ := 1

/-- Represents the number of green sock pairs -/
def green_pairs : ℕ := 1

/-- Represents the number of socks selected -/
def selected_socks : ℕ := 5

/-- Calculates the probability of selecting exactly two pairs of socks with the same color -/
theorem probability_two_pairs : 
  (Nat.choose total_socks selected_socks : ℚ)⁻¹ * 
  ((Nat.choose 2 2 : ℚ) * (Nat.choose 2 1 : ℚ) * 
   ((Nat.choose (2 * blue_pairs) 2 : ℚ) * (Nat.choose (2 * red_pairs) 2 : ℚ))) = 3 / 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_pairs_l859_85941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l859_85977

theorem cos_double_angle (α : ℝ) (h : Real.cos α = 1/3) : Real.cos (2*α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l859_85977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l859_85973

/-- The area of a right triangle with base 12 cm and height 9 cm is 54 cm² -/
theorem right_triangle_area : 
  ∀ (A B C : ℝ × ℝ) (base height : ℝ),
  base = 12 →
  height = 9 →
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = base^2 →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = height^2 →
  (C.1 - A.1) * (B.2 - A.2) = (C.2 - A.2) * (B.1 - A.1) →
  (1/2) * base * height = 54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l859_85973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l859_85971

/-- Represents a point in 2D space -/
structure Point where
  x : Real
  y : Real

/-- Checks if angle A in triangle ABC is a right angle -/
def RightAngle (A B C : Point) : Prop := sorry

/-- Calculates the tangent of angle B in triangle ABC -/
def TanAngle (B : Point) : Real := sorry

/-- Checks if an ellipse with foci A and B passes through point C -/
def EllipsePasses (A B C : Point) : Prop := sorry

/-- Calculates the eccentricity of an ellipse with foci A and B passing through C -/
noncomputable def EllipseEccentricity (A B C : Point) : Real := sorry

/-- Given a right-angled triangle ABC with angle A = 90° and tan B = 3/4,
    and an ellipse with foci at A and B passing through C,
    prove that the eccentricity of the ellipse is 1/2 -/
theorem ellipse_eccentricity (A B C : Point) (e : Real) :
  RightAngle A B C →
  TanAngle B = 3/4 →
  EllipsePasses A B C →
  e = EllipseEccentricity A B C →
  e = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l859_85971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l859_85953

theorem function_property (f : ℕ → ℕ) 
  (h1 : ∀ n, f (f n) + f n = 2 * n + 3) 
  (h2 : f 0 = 1) : 
  ∀ n, f n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l859_85953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statements_evaluation_l859_85905

open Real

-- Define the statements
def statement_A : Prop := ∀ α : ℝ, α = 1 → α > Real.pi / 2 ∧ α < Real.pi
def statement_B : Prop := 15 / 60 * 360 = 90
def statement_C : Prop := ∀ α : ℝ, 0 < α ∧ α < Real.pi / 2 → 0 < α / 3 ∧ α / 3 < Real.pi / 2
def statement_D : Prop := ∃! x : ℝ, -Real.pi/2 < x ∧ x < Real.pi/2 ∧ Real.tan x = Real.sin x

theorem statements_evaluation :
  ¬statement_A ∧ statement_B ∧ ¬statement_C ∧ statement_D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_statements_evaluation_l859_85905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_characterization_l859_85940

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The divisibility condition for the polynomial -/
def satisfiesDivisibilityCondition (P : IntPolynomial) : Prop :=
  ∀ (a b : ℤ), (a + 2*b) ∣ (P.eval a + 2 * P.eval b)

/-- The theorem statement -/
theorem polynomial_divisibility_characterization (P : IntPolynomial) :
  satisfiesDivisibilityCondition P ↔ ∃ (k : ℤ), P = Polynomial.monomial 1 k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_characterization_l859_85940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_compensation_l859_85967

/-- Calculates the insurance compensation given the insured amount, deductible percentage, and actual damage. -/
noncomputable def insurance_compensation (insured_amount : ℝ) (deductible_percent : ℝ) (actual_damage : ℝ) : ℝ :=
  let threshold := insured_amount * (deductible_percent / 100)
  if actual_damage < threshold then 0 else actual_damage

/-- Theorem stating that the insurance compensation is 0 for the given scenario. -/
theorem zero_compensation (insured_amount : ℝ) (deductible_percent : ℝ) (actual_damage : ℝ)
  (h1 : insured_amount = 500000)
  (h2 : deductible_percent = 1)
  (h3 : actual_damage = 4000) :
  insurance_compensation insured_amount deductible_percent actual_damage = 0 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_compensation_l859_85967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l859_85922

/-- Represents a regular polygon inscribed in a circle -/
structure RegularPolygon where
  sides : ℕ
  inscribed : Bool
  deriving Repr, DecidableEq

/-- Counts the number of intersections between two regular polygons -/
def countIntersections (p1 p2 : RegularPolygon) : ℕ :=
  2 * min p1.sides p2.sides

/-- The set of regular polygons inscribed in the circle -/
def polygons : List RegularPolygon :=
  [⟨5, true⟩, ⟨6, true⟩, ⟨7, true⟩, ⟨8, true⟩]

/-- Checks if all polygons in the list are inscribed -/
def allInscribed (ps : List RegularPolygon) : Prop :=
  ps.all (·.inscribed)

/-- No two polygons share a vertex -/
axiom no_shared_vertices (ps : List RegularPolygon) : Prop

/-- No three sides intersect at a common point -/
axiom no_triple_intersections (ps : List RegularPolygon) : Prop

/-- Calculates the total number of intersections -/
def totalIntersections (ps : List RegularPolygon) : ℕ :=
  (ps.foldl (fun acc p1 => 
    acc + ps.foldl (fun inner_acc p2 => 
      if p1 ≠ p2 then inner_acc + countIntersections p1 p2 else inner_acc
    ) 0
  ) 0) / 2

/-- The main theorem -/
theorem intersection_count : 
  allInscribed polygons → 
  no_shared_vertices polygons → 
  no_triple_intersections polygons → 
  totalIntersections polygons = 68 := by
  sorry

#eval totalIntersections polygons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l859_85922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_problem_l859_85915

theorem cubic_polynomial_problem (a b c : ℝ) (P : ℝ → ℝ) : 
  (a^3 - 4*a^2 + 7*a - 2 = 0) →
  (b^3 - 4*b^2 + 7*b - 2 = 0) →
  (c^3 - 4*c^2 + 7*c - 2 = 0) →
  (∃ p q r s : ℝ, ∀ x, P x = p*x^3 + q*x^2 + r*x + s) →
  (P a = b + 2*c) →
  (P b = 2*a + c) →
  (P c = 2*a + b) →
  (P (a + b + c) = -20) →
  (∀ x, P x = -10/13*x^3 + 32/13*x^2 - 11/13*x - 92/13) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_problem_l859_85915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lous_shoes_last_week_sales_l859_85936

/-- Prove that Lou's Shoes sold 27 pairs of shoes last week -/
theorem lous_shoes_last_week_sales : ℕ := by
  let monthly_goal : ℕ := 80
  let sold_this_week : ℕ := 12
  let remaining_to_goal : ℕ := 41
  let last_week_sales : ℕ := monthly_goal - sold_this_week - remaining_to_goal
  have h : last_week_sales = 27 := by
    -- Proof goes here
    sorry
  exact last_week_sales


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lous_shoes_last_week_sales_l859_85936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_x_squared_plus_2x_l859_85970

theorem integral_sqrt_minus_x_squared_plus_2x :
  (∫ x in (Set.Icc 0 1), Real.sqrt (-x^2 + 2*x)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_x_squared_plus_2x_l859_85970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_definition_correct_l859_85974

-- Define the types for geometric solids
structure Frustum : Type := (dummy : Unit)
structure Prism : Type := (dummy : Unit)
structure Cone : Type := (dummy : Unit)
structure Sphere : Type := (dummy : Unit)

-- Define the operations that form these solids
noncomputable def cut_pyramid_with_plane : Unit → Option Frustum := sorry
noncomputable def has_parallel_faces_and_quadrilaterals : Unit → Option Prism := sorry
noncomputable def rotate_right_triangle : Unit → Option Cone := sorry
noncomputable def rotate_hemisphere : Unit → Option Sphere := sorry

-- Theorem stating that only the sphere definition is correct
theorem sphere_definition_correct :
  (∃ (p : Unit), cut_pyramid_with_plane p = some (Frustum.mk ())) = False ∧
  (∃ (p : Unit), has_parallel_faces_and_quadrilaterals p = some (Prism.mk ())) = False ∧
  (∃ (t : Unit), rotate_right_triangle t = some (Cone.mk ())) = False ∧
  (∃ (h : Unit), rotate_hemisphere h = some (Sphere.mk ())) = True :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_definition_correct_l859_85974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l859_85939

/-- Reflection of a point across a line --/
noncomputable def reflect (x y m b : ℝ) : ℝ × ℝ := 
  let x' := ((1 - m^2) * x + 2 * m * (y - b)) / (1 + m^2)
  let y' := (2 * m * x - (1 - m^2) * (y - b)) / (1 + m^2) + b
  (x', y')

/-- Theorem: If (-2, 0) reflects to (6, 4) across y = mx + b, then m + b = 4 --/
theorem reflection_sum (m b : ℝ) :
  reflect (-2) 0 m b = (6, 4) → m + b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l859_85939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PRD_area_l859_85989

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- Theorem: The area of triangle PRD is 48 - 4q -/
theorem triangle_PRD_area (q : ℝ) : 
  triangleArea 0 12 4 12 12 0 = 48 - 4*q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PRD_area_l859_85989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_loss_percentage_l859_85916

noncomputable def total_cost : ℝ := 490
noncomputable def cost_book1 : ℝ := 285.8333333333333
noncomputable def gain_percentage : ℝ := 19

noncomputable def cost_book2 : ℝ := total_cost - cost_book1

noncomputable def selling_price_book2 : ℝ := cost_book2 * (1 + gain_percentage / 100)

noncomputable def loss_percentage : ℝ := (cost_book1 - selling_price_book2) / cost_book1 * 100

theorem book_sale_loss_percentage :
  (loss_percentage ≥ 14.9 ∧ loss_percentage ≤ 15.1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_loss_percentage_l859_85916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_VXZ_measure_is_135_l859_85987

/-- A regular octagon is a polygon with 8 sides of equal length and 8 angles of equal measure. -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : ∀ i j : Fin 8, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

/-- The measure of an angle in a regular octagon. -/
def regular_octagon_angle_measure : ℝ := 135

/-- The measure of angle VXZ in a regular octagon, where V, X, and Z are vertices
    with one vertex between V and X. -/
def angle_VXZ_measure (octagon : RegularOctagon) : ℝ := 135

/-- Theorem: In a regular octagon, the measure of angle VXZ is 135°,
    where V, X, and Z are vertices with one vertex between V and X. -/
theorem angle_VXZ_measure_is_135 (octagon : RegularOctagon) :
  angle_VXZ_measure octagon = regular_octagon_angle_measure := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_VXZ_measure_is_135_l859_85987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_derivatives_is_zero_l859_85961

/-- An arithmetic sequence with common difference d -/
def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℕ → ℝ) (x : ℝ) : ℝ :=
  (x - a 1) * (x - a 2) * (x - a 3) * (x - a 4)

/-- The derivative of f(x) -/
noncomputable def f' (a : ℕ → ℝ) : ℝ → ℝ :=
  deriv (f a)

theorem sum_of_derivatives_is_zero (a : ℕ → ℝ) (d : ℝ) :
  arithmeticSequence a d →
  f' a (a 1) + f' a (a 2) + f' a (a 3) + f' a (a 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_derivatives_is_zero_l859_85961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_equal_sum_date_l859_85945

/-- Represents a date in DD.MM.YYYY format -/
structure Date where
  day : Nat
  month : Nat
  year : Nat
  h1 : day ≥ 1 ∧ day ≤ 31
  h2 : month ≥ 1 ∧ month ≤ 12
  h3 : year ≥ 1

/-- Returns true if the date is in the year 2008 -/
def isIn2008 (d : Date) : Prop := d.year = 2008

/-- Returns true if the sum of the first four digits equals the sum of the last four digits -/
def hasEqualSums (d : Date) : Prop :=
  let digits := [d.day / 10, d.day % 10, d.month / 10, d.month % 10,
                 d.year / 1000, (d.year / 100) % 10, (d.year / 10) % 10, d.year % 10]
  (digits[0]! + digits[1]! + digits[2]! + digits[3]! : Nat) =
  (digits[4]! + digits[5]! + digits[6]! + digits[7]! : Nat)

/-- Returns true if d1 is later than d2 -/
def isLaterThan (d1 d2 : Date) : Prop :=
  d1.year > d2.year ∨
  (d1.year = d2.year ∧ d1.month > d2.month) ∨
  (d1.year = d2.year ∧ d1.month = d2.month ∧ d1.day > d2.day)

theorem last_equal_sum_date :
  ∃ (d : Date),
    isIn2008 d ∧
    hasEqualSums d ∧
    d.day = 25 ∧
    d.month = 12 ∧
    (∀ (d' : Date), isIn2008 d' → hasEqualSums d' → isLaterThan d' d → False) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_equal_sum_date_l859_85945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_discontinuous_at_zero_g_discontinuous_at_neg_three_h_continuous_everywhere_j_discontinuous_at_odd_multiples_of_pi_half_l859_85964

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def g (x : ℝ) : ℝ := 3 / (x + 3)
noncomputable def h (x : ℝ) : ℝ := (x^2 - 1) / (x^2 + 1)
noncomputable def j (x : ℝ) : ℝ := Real.tan x

-- Theorems to prove
theorem f_discontinuous_at_zero : 
  ¬ContinuousAt f 0 := by sorry

theorem g_discontinuous_at_neg_three : 
  ¬ContinuousAt g (-3) := by sorry

theorem h_continuous_everywhere : 
  Continuous h := by sorry

theorem j_discontinuous_at_odd_multiples_of_pi_half (k : ℤ) : 
  ¬ContinuousAt j ((2 * k + 1) * Real.pi / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_discontinuous_at_zero_g_discontinuous_at_neg_three_h_continuous_everywhere_j_discontinuous_at_odd_multiples_of_pi_half_l859_85964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_slope_l859_85948

/-- The point from which the light ray is emitted -/
def A : ℝ × ℝ := (-2, 3)

/-- The circle equation: (x-3)^2 + (y-2)^2 = 1 -/
def circleEq (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1

/-- The reflected point of A across the x-axis -/
def A' : ℝ × ℝ := (-2, -3)

/-- The slope of the line on which the reflected ray lies -/
def k : Set ℝ := {slope | 
  (slope = 4/3 ∨ slope = 3/4) ∧
  ∃ x y : ℝ, 
    -- The line passes through A'
    y + 3 = slope * (x + 2) ∧
    -- The line is tangent to the circle
    (3 * slope - 2 + 2 * slope - 3)^2 / (slope^2 + 1) = 1 ∧
    -- The point (x, y) satisfies the circle equation
    circleEq x y}

theorem reflected_ray_slope :
  ∃ slope : ℝ, slope ∈ k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_slope_l859_85948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_union_theorem_l859_85909

theorem subset_union_theorem (n m : ℕ) (X : Finset ℕ) 
  (A : Fin m → Finset ℕ) :
  n > 6 →
  X.card = n →
  (∀ i : Fin m, (A i).card = 5 ∧ A i ⊆ X) →
  (∀ i j : Fin m, i ≠ j → A i ≠ A j) →
  m > n * (n - 1) * (n - 2) * (n - 3) * (4 * n - 15) / 600 →
  ∃ (i : Fin 6 → Fin m), 
    (∀ j k : Fin 6, j < k → i j < i k) ∧
    (Finset.biUnion (Finset.range 6) (fun j => A (i j))).card = 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_union_theorem_l859_85909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circular_arcs_area_l859_85954

/-- The area enclosed by three intersecting circular arcs -/
noncomputable def enclosed_area (r : ℝ) : ℝ :=
  (9 * Real.pi / 2) - (9 * Real.sqrt 3 / 2)

/-- Theorem stating the area enclosed by three intersecting circular arcs -/
theorem three_circular_arcs_area :
  let r : ℝ := 3
  let central_angle : ℝ := Real.pi / 2
  let triangle_side : ℝ := 2 * r
  enclosed_area r = (9 * Real.pi / 2) - (9 * Real.sqrt 3 / 2) := by
  -- The proof goes here
  sorry

#check three_circular_arcs_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circular_arcs_area_l859_85954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l859_85929

theorem tangent_line_equation (e : ℝ) (x y : ℝ → ℝ) :
  (∀ t, y t = (x t)^3 / e) →
  x e = e →
  y e = e^2 →
  ∃ m : ℝ, ∀ t, y t - e^2 = m * (x t - e) ∧ m = 3 * e :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l859_85929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_l859_85917

-- Define the number of sides on each die
def num_sides : ℕ := 8

-- Define the set of even numbers on the blue die
def blue_even : Finset ℕ := {2, 4, 6, 8}

-- Define the set of prime numbers on the yellow die
def yellow_prime : Finset ℕ := {2, 3, 5, 7}

-- Define the total number of possible outcomes
def total_outcomes : ℕ := num_sides * num_sides

-- Define the number of favorable outcomes
def favorable_outcomes : ℕ := (blue_even.card : ℕ) * (yellow_prime.card : ℕ)

-- Theorem statement
theorem dice_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_l859_85917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l859_85928

-- Define the function f(x)
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * ω * x - Real.pi / 4)

-- State the theorem
theorem monotonic_increase_interval 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_max_period : (2 : ℝ) = 2 * Real.pi / (2 * ω)) : 
  ∃ (a b : ℝ), a = -1/4 ∧ b = 3/4 ∧ 
  StrictMonoOn (f ω) (Set.Icc a b) ∧
  Set.Icc a b ⊆ Set.Icc (-1 : ℝ) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l859_85928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_plus_arcsin_l859_85944

theorem max_value_sin_plus_arcsin :
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 →
  |Real.sin x + Real.arcsin x| ≤ Real.sqrt 2 / 2 + Real.pi / 4 ∧
  ∃ y ∈ Set.Icc (-1) 1, |Real.sin y + Real.arcsin y| = Real.sqrt 2 / 2 + Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_plus_arcsin_l859_85944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_discarding_l859_85935

theorem average_after_discarding (numbers : Finset ℝ) (sum : ℝ) : 
  numbers.card = 50 →
  sum = numbers.sum (fun x => x) →
  sum / 50 = 56 →
  45 ∈ numbers →
  55 ∈ numbers →
  let remaining := numbers.erase 45 |>.erase 55
  let new_sum := sum - 45 - 55
  (new_sum / remaining.card : ℝ) = 56.25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_discarding_l859_85935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_bill_amount_l859_85914

/-- The final bill amount after two successive 2% late charges on an original bill of $500 -/
theorem final_bill_amount : ℝ := by
  let original_bill : ℝ := 500
  let late_charge_rate : ℝ := 0.02
  let first_late_charge : ℝ := original_bill * (1 + late_charge_rate)
  let second_late_charge : ℝ := first_late_charge * (1 + late_charge_rate)
  have : second_late_charge = 520.20 := by sorry
  exact 520.20


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_bill_amount_l859_85914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_equals_89_point_5_l859_85932

open Real

-- Define the sequence of angles
noncomputable def angles : List ℝ := List.map (fun k => (2 * k - 1) / 2) (List.range 90)

-- Define the sum of sin^2 for the given angles
noncomputable def sin_squared_sum : ℝ := (angles.map (fun θ => Real.sin (θ * π / 180) ^ 2)).sum

-- Theorem statement
theorem sin_squared_sum_equals_89_point_5 : sin_squared_sum = 89.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_equals_89_point_5_l859_85932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_right_angle_l859_85959

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Theorem: Largest angle in triangle F₁MF₂ is 90° -/
theorem largest_angle_is_right_angle
  (e : Ellipse)
  (f1 f2 m : Point)
  (h1 : e.a = 4 ∧ e.b = 2*Real.sqrt 3)
  (h2 : onEllipse m e)
  (h3 : distance m f1 - distance m f2 = 2) :
  ∃ θ : ℝ, θ = 90 ∧ θ = max (angle f1 m f2) (max (angle m f1 f2) (angle m f2 f1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_right_angle_l859_85959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l859_85918

theorem triangle_area (A B C : ℝ) (a b c : ℝ) (R : ℝ) (S_ABC : ℝ) : 
  A = π / 3 →
  b = 1 →
  R = 1 →
  S_ABC = (1 / 2) * a * b * Real.sin C →
  S_ABC = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l859_85918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inverse_fourth_quadrant_l859_85904

theorem sin_inverse_fourth_quadrant (x : ℝ) :
  Real.sin x = -1/3 ∧ x ∈ Set.Ioo (-Real.pi/2) 0 → x = -Real.arcsin (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inverse_fourth_quadrant_l859_85904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joann_fifth_day_lollipops_l859_85946

/-- The number of lollipops Joann ate on the first day -/
def first_day_lollipops : ℕ := 10

/-- The total number of lollipops Joann ate over seven days -/
def total_lollipops : ℕ := 154

/-- The number of additional lollipops Joann ate each day compared to the previous day -/
def daily_increase : ℕ := 4

/-- The number of days Joann ate lollipops -/
def total_days : ℕ := 7

/-- The number of lollipops Joann ate on the nth day -/
def lollipops_on_day (n : ℕ) : ℕ := first_day_lollipops + daily_increase * (n - 1)

theorem joann_fifth_day_lollipops :
  lollipops_on_day 5 = 26 ∧
  (Finset.sum (Finset.range total_days) (fun i => lollipops_on_day (i + 1))) = total_lollipops :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joann_fifth_day_lollipops_l859_85946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_properties_l859_85913

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with vertex and focus -/
structure Parabola where
  vertex : Point
  focus : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if a point is in the first quadrant -/
def isFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Checks if a point lies on a parabola -/
def liesOnParabola (p : Point) (parab : Parabola) : Prop :=
  let directrixY := parab.vertex.y - (parab.focus.y - parab.vertex.y)
  distance p parab.focus = p.y - directrixY

theorem parabola_point_properties :
  let parab : Parabola := { vertex := { x := 0, y := 0 }, focus := { x := 0, y := 2 } }
  let p : Point := { x := 2 * Real.sqrt 296, y := 148 }
  liesOnParabola p parab ∧ isFirstQuadrant p ∧ distance p parab.focus = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_properties_l859_85913
