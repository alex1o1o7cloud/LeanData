import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l876_87698

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def valid_subset (M : Finset ℕ) : Prop :=
  M ⊆ Finset.range 16 ∧
  ∀ a b c, a ∈ M → b ∈ M → c ∈ M → a ≠ b ∧ b ≠ c ∧ a ≠ c → ¬is_perfect_square (a * b * c)

theorem max_subset_size :
  ∃ M : Finset ℕ, valid_subset M ∧ M.card = 11 ∧
  ∀ N : Finset ℕ, valid_subset N → N.card ≤ 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l876_87698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l876_87624

theorem evaluate_expression (a : ℝ) : (let x := a + 7; x - a + 3) = 10 := by
  -- Introduce x
  let x := a + 7
  -- Expand the expression
  calc
    x - a + 3 = (a + 7) - a + 3 := by rfl
    _ = a - a + 7 + 3 := by ring
    _ = 0 + 7 + 3 := by ring
    _ = 10 := by ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l876_87624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_weight_proof_l876_87694

theorem coin_weight_proof (S : Finset ℕ) (h1 : S.card = 20) 
  (h2 : ∀ n, n ∈ S → 1 ≤ n ∧ n ≤ 20) (h3 : ∀ m n, m ∈ S → n ∈ S → m ≠ n → m ≠ n) :
  ∃ T : Finset ℕ, T ⊆ S ∧ T.card = 2 ∧ (S \ T).card = 18 ∧ 
    ∃ x y, x ∈ T ∧ y ∈ T ∧ x + y ≥ 28 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_weight_proof_l876_87694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_l876_87666

/-- The number of pieces of cherry gum Chewbacca has initially -/
def cherry_gum : ℕ := 18

/-- The number of pieces of grape gum Chewbacca has initially -/
def grape_gum : ℕ := 24

/-- The theorem stating that the number of pieces in each complete pack of gum is 1 -/
theorem gum_pack_size (x : ℕ) :
  (cherry_gum - 2 * x) / grape_gum = cherry_gum / (grape_gum + 3 * x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_l876_87666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_for_given_tax_l876_87690

/-- Represents the tax structure in country X -/
structure TaxStructure where
  lowRate : ℚ  -- tax rate for income up to $5000
  highRate : ℚ  -- tax rate for income over $5000
  threshold : ℚ  -- income threshold for rate change

/-- Calculates the tax for a given income under the given tax structure -/
def calculateTax (ts : TaxStructure) (income : ℚ) : ℚ :=
  if income ≤ ts.threshold then
    ts.lowRate * income
  else
    ts.lowRate * ts.threshold + ts.highRate * (income - ts.threshold)

/-- Theorem stating that given the specific tax structure and tax paid, 
    the corresponding income is $10,500 -/
theorem income_for_given_tax (ts : TaxStructure) 
  (h1 : ts.lowRate = 8/100)
  (h2 : ts.highRate = 10/100)
  (h3 : ts.threshold = 5000)
  (h4 : calculateTax ts 10500 = 950) :
  ∃ (income : ℚ), calculateTax ts income = 950 ∧ income = 10500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_for_given_tax_l876_87690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birth_interval_is_three_l876_87636

/-- Represents the ages of 5 children -/
structure ChildrenAges where
  ages : Fin 5 → ℕ
  youngest_is_8 : ages 0 = 8
  sum_is_70 : (Finset.sum (Finset.univ : Finset (Fin 5)) ages) = 70

/-- The interval between births is the constant difference between consecutive ages -/
def BirthInterval (ca : ChildrenAges) : ℕ :=
  ca.ages 1 - ca.ages 0

/-- The main theorem: given the conditions, the birth interval is 3 years -/
theorem birth_interval_is_three (ca : ChildrenAges) 
  (h_constant : ∀ i : Fin 4, ca.ages (i + 1) - ca.ages i = BirthInterval ca) : 
  BirthInterval ca = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birth_interval_is_three_l876_87636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_with_spheres_l876_87695

/-- The height of a rectangular box containing spheres with specific configurations --/
noncomputable def box_height (box_width box_length : ℝ) (large_sphere_radius small_sphere_radius : ℝ) : ℝ :=
  let small_sphere_center_distance := box_width - 2 * small_sphere_radius
  let center_to_center_distance := Real.sqrt (64 + 1)
  2 * small_sphere_radius + center_to_center_distance - 2 * small_sphere_radius

/-- Theorem stating the height of the box with given specifications --/
theorem box_height_with_spheres :
  box_height 6 6 3 2 = Real.sqrt 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_with_spheres_l876_87695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_GP_length_eq_l876_87681

/-- Rhombus JOHN with given properties -/
structure Rhombus where
  -- Side length
  side : ℝ
  -- Diagonal ON
  diag : ℝ
  -- Assumption that side = 16 and diag = 12
  side_eq : side = 16
  diag_eq : diag = 12

/-- Point G on diagonal JN -/
noncomputable def G (r : Rhombus) : ℝ × ℝ :=
  (5 * Real.sqrt 2, 0)

/-- Point P on diagonal HN -/
noncomputable def P (r : Rhombus) : ℝ × ℝ :=
  (r.side - 10 * Real.sqrt 2, 0)

/-- Length of GP -/
noncomputable def GP_length (r : Rhombus) : ℝ :=
  Real.sqrt ((G r).1 - (P r).1)^2

theorem GP_length_eq (r : Rhombus) : GP_length r = (3 * Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_GP_length_eq_l876_87681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l876_87661

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 2

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := f x - (m - 9)*x

-- Theorem for the properties of f and g
theorem f_and_g_properties :
  (f (-1) = 7) ∧
  (deriv f 3 = 0) ∧
  (∀ m : ℝ,
    (m > -3 →
      (∃ a b : ℝ, a < b ∧
        (StrictMonoOn (g m) (Set.Iio a)) ∧
        (StrictAntiOn (g m) (Set.Icc a b)) ∧
        (StrictMonoOn (g m) (Set.Ioi b)))) ∧
    (m = -3 →
      (StrictMonoOn (g m) Set.univ)) ∧
    (m < -3 →
      (StrictMonoOn (g m) Set.univ))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l876_87661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_sum_157_l876_87632

def is_geometric_progression (s : List Nat) : Prop :=
  s.length > 0 ∧ 
  (∃ q : Nat, q > 1 ∧ 
    ∀ i : Nat, i < s.length - 1 → s.get! (i+1) = s.get! i * q)

theorem geometric_progression_sum_157 :
  ∀ s : List Nat,
    is_geometric_progression s ∧ 
    s.sum = 157 →
    s = [157] ∨ s = [1, 156] ∨ s = [1, 12, 144] :=
by
  intro s hyp
  sorry

#check geometric_progression_sum_157

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_sum_157_l876_87632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_period_l876_87602

/-- The smallest positive period of y = 5tan(2/5x + π/6) is 5π/2 -/
theorem tangent_period : 
  ∃ T : ℝ, T > 0 ∧ T = 5*π/2 ∧ 
  (∀ t : ℝ, 5 * Real.tan (2/5 * (t + T) + π/6) = 5 * Real.tan (2/5 * t + π/6)) ∧
  (∀ S : ℝ, S > 0 ∧ (∀ t : ℝ, 5 * Real.tan (2/5 * (t + S) + π/6) = 5 * Real.tan (2/5 * t + π/6)) → T ≤ S) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_period_l876_87602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_speed_Q_to_B_l876_87689

/-- Represents the distance between towns B and C -/
def D : ℝ := sorry

/-- Represents Tom's speed from Q to B -/
def S : ℝ := sorry

/-- The distance between Q and B -/
def Q_to_B_distance : ℝ := 2 * D

/-- The distance between B and C is 20 mph -/
def B_to_C_speed : ℝ := 20

/-- The distance between Q and B is twice the distance between B and C -/
axiom distance_relation : 2 * D = Q_to_B_distance

/-- The average speed of the whole journey is 36 mph -/
axiom average_speed : 36 = (3 * D) / ((2 * D) / S + D / 20)

/-- Tom's speed from B to C is 20 mph -/
axiom speed_B_to_C : 20 = B_to_C_speed

/-- Theorem stating that Tom's speed from Q to B is 60 mph -/
theorem toms_speed_Q_to_B : S = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_speed_Q_to_B_l876_87689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_two_l876_87644

/-- A square with an inscribed circle and a circumscribed circle -/
structure SquareWithCircles where
  s : ℝ  -- side length of the square
  s_pos : s > 0

namespace SquareWithCircles

/-- Radius of the inscribed circle -/
noncomputable def r_inscribed (sq : SquareWithCircles) : ℝ := sq.s / 2

/-- Radius of the circumscribed circle -/
noncomputable def r_circumscribed (sq : SquareWithCircles) : ℝ := sq.s * Real.sqrt 2 / 2

/-- Area of the inscribed circle -/
noncomputable def area_inscribed (sq : SquareWithCircles) : ℝ := Real.pi * (r_inscribed sq) ^ 2

/-- Area of the circumscribed circle -/
noncomputable def area_circumscribed (sq : SquareWithCircles) : ℝ := Real.pi * (r_circumscribed sq) ^ 2

/-- The theorem stating that the ratio of the areas is 2 -/
theorem area_ratio_is_two (sq : SquareWithCircles) :
  area_circumscribed sq / area_inscribed sq = 2 := by
  sorry

end SquareWithCircles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_two_l876_87644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l876_87657

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the line
def line (k m x y : ℝ) : Prop := y = k*x + m

-- Define the distance from a point to a line
noncomputable def distance_to_line (x₀ y₀ a b c : ℝ) : ℝ := 
  abs (a*x₀ + b*y₀ + c) / Real.sqrt (a^2 + b^2)

theorem ellipse_properties 
  (k m : ℝ) 
  (h_k : k ≠ 0) :
  -- The ellipse has a vertex at (0, -1)
  ellipse 0 (-1) →
  -- The right focus is at distance 3 from the line x - y + 2√2 = 0
  (∃ x₀ : ℝ, distance_to_line x₀ 0 1 (-1) (2*Real.sqrt 2) = 3 ∧ 
    x₀ > 0 ∧ ellipse x₀ 0) →
  -- The ellipse intersects the line y = kx + m at two distinct points
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line k m x₁ y₁ ∧ line k m x₂ y₂) →
  -- When |AM| = |AN|, where A is (0, -1) and M, N are intersection points
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line k m x₁ y₁ ∧ line k m x₂ y₂ →
    (x₁^2 + (y₁ + 1)^2 = x₂^2 + (y₂ + 1)^2)) →
  -- Then m is in the range (1/2, 2)
  1/2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l876_87657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_slope_l876_87656

/-- Given a parabola y² = 4x with focus at (1, 0), if a point A on the parabola
    is at distance 3 from the focus, then the slope of line OA
    (where O is the origin) is ±√2. -/
theorem parabola_slope (A : ℝ × ℝ) : 
  A.2^2 = 4 * A.1 →  -- A is on the parabola
  (A.1 - 1)^2 + A.2^2 = 9 →  -- A is distance 3 from focus (1, 0)
  A.2 / A.1 = Real.sqrt 2 ∨ A.2 / A.1 = -Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_slope_l876_87656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_team_orders_four_runners_l876_87665

def relay_team_orders (n : ℕ) (first_runner : Fin n) (flexible_runner : Fin n) : ℕ :=
  if n ≠ 4 ∨ first_runner = flexible_runner then 0
  else Nat.factorial (n - 2)

theorem relay_team_orders_four_runners :
  ∃ (first_runner flexible_runner : Fin 4),
    first_runner ≠ flexible_runner ∧
    relay_team_orders 4 first_runner flexible_runner = 6 :=
by
  sorry

#eval relay_team_orders 4 0 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_team_orders_four_runners_l876_87665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swallow_oxygen_consumption_l876_87684

theorem swallow_oxygen_consumption 
  (a : ℝ) 
  (h_a : a ≠ 0) 
  (v : ℝ → ℝ) 
  (h_v : ∀ x, v x = a * Real.log (x / 10) / Real.log 2) 
  (h_40 : v 40 = 10) : 
  v 320 = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swallow_oxygen_consumption_l876_87684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_at_distance_l876_87638

noncomputable def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_points_at_distance (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ) (P : ℝ × ℝ) :
  C = Circle center radius →
  P ∉ C →
  (∃ (p : ℝ × ℝ), p ∈ C ∧ distance P p = 3) →
  (∀ (S : Finset (ℝ × ℝ)), ↑S ⊆ C ∧ (∀ p ∈ S, distance P p = 3) → S.card ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_at_distance_l876_87638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_chalk_count_l876_87664

theorem original_chalk_count 
  (total_people : ℕ)
  (chalk_per_person : ℕ)
  (added_chalk : ℕ)
  (lost_chalk : ℕ)
  (original_chalk : ℕ)
  (h1 : total_people = 7)
  (h2 : chalk_per_person = 3)
  (h3 : added_chalk = 12)
  (h4 : lost_chalk = 2)
  (h5 : total_people * chalk_per_person = original_chalk + added_chalk - lost_chalk) :
  original_chalk = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_chalk_count_l876_87664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_division_l876_87615

/-- A tetrahedron with vertices A, B, C, D -/
structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

/-- A plane passing through an edge and the midpoint of the opposite edge -/
structure DividingPlane where
  edge : Segment
  midpoint : Point

/-- The set of all dividing planes for a tetrahedron -/
def dividingPlanes (t : Tetrahedron) : Set DividingPlane := sorry

/-- The number of parts the tetrahedron is divided into by the dividing planes -/
def numParts (t : Tetrahedron) : ℕ := sorry

/-- The volume of a part of the divided tetrahedron -/
noncomputable def partVolume (t : Tetrahedron) (p : Set Point) : ℝ := sorry

/-- The partition of the tetrahedron created by the dividing planes -/
def tetrahedronPartition (t : Tetrahedron) : Set (Set Point) := sorry

theorem tetrahedron_division (t : Tetrahedron) :
  (numParts t = 24) ∧
  (∀ p q : Set Point, p ∈ tetrahedronPartition t ∧ q ∈ tetrahedronPartition t →
    partVolume t p = partVolume t q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_division_l876_87615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_f_eq_l876_87667

def f (x : ℝ) : ℝ := (x - 2)^2 + 2

def translate_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x ↦ f (x + a)
def translate_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x ↦ f x + b

def translated_f : ℝ → ℝ := translate_up (translate_left f 1) 1

theorem translated_f_eq : ∀ x, translated_f x = (x - 1)^2 + 3 := by
  intro x
  unfold translated_f translate_up translate_left f
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_f_eq_l876_87667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l876_87650

/-- A quadrilateral with specific properties -/
structure Quadrilateral :=
  (EF FG EH HG : ℝ)
  (right_angle_F : EF ^ 2 + FG ^ 2 = 16)
  (right_angle_H : EH ^ 2 + HG ^ 2 = 16)
  (distinct_integer_sides : ∃ (a b : ℕ), (EF = a ∨ FG = a ∨ EH = a ∨ HG = a) ∧ 
                                        (EF = b ∨ FG = b ∨ EH = b ∨ HG = b) ∧ 
                                        a ≠ b)

/-- The area of the quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℝ := 2 * Real.sqrt 3 + (3/2) * Real.sqrt 7

/-- Theorem stating the area of the quadrilateral with given properties -/
theorem quadrilateral_area (q : Quadrilateral) : area q = 2 * Real.sqrt 3 + (3/2) * Real.sqrt 7 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l876_87650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l876_87673

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on an ellipse -/
def point_on_ellipse (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The upper vertex of an ellipse -/
def upper_vertex (e : Ellipse) : ℝ × ℝ := (0, e.b)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem statement -/
theorem ellipse_eccentricity_range (e : Ellipse) :
  (∀ (x y : ℝ), point_on_ellipse e x y →
    distance (x, y) (upper_vertex e) ≤ 2 * e.b) →
  0 < eccentricity e ∧ eccentricity e ≤ Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l876_87673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_y_coordinate_less_than_half_l876_87628

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1/2 + Real.log (x / (1 - x)) / Real.log 2

-- Define the theorem
theorem midpoint_y_coordinate_less_than_half 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : y₁ = f x₁) 
  (h₂ : y₂ = f x₂) 
  (h₃ : (x₁ + x₂) / 2 = 1/2) : 
  (y₁ + y₂) / 2 < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_y_coordinate_less_than_half_l876_87628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l876_87691

structure Triangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real

def isAcute (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

def isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

theorem triangle_properties (t : Triangle) :
  (isAcute t → Real.sin t.A > Real.cos t.B) ∧
  (Real.sin t.A ^ 2 + Real.sin t.B ^ 2 + Real.cos t.C ^ 2 < 1 → isObtuse t) ∧
  (t.AB = Real.sqrt 3 ∧ t.AC = 1 ∧ t.B = Real.pi/6 →
    t.AB * t.AC * Real.sin t.B / 2 = Real.sqrt 3 / 4 ∨
    t.AB * t.AC * Real.sin t.B / 2 = Real.sqrt 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l876_87691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l876_87635

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem tangent_line_at_one (x y : ℝ) :
  (y = 2*x - 1) ↔ (y - f 1 = deriv f 1 * (x - 1)) :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l876_87635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_m_range_l876_87697

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 4) ^ 2 - Real.sqrt 3 * Real.cos (2 * ω * x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_properties (ω : ℝ) (h_ω_pos : ω > 0) (h_period : has_period (f ω) Real.pi) :
  ω = 1 ∧ Set.Icc 0 3 = Set.range (f ω) := by
  sorry

theorem m_range (ω : ℝ) (h_ω_pos : ω > 0) (h_period : has_period (f ω) Real.pi) :
  ∀ m : ℝ, (∀ x ∈ Set.Icc (Real.pi/4) (Real.pi/2), |f ω x - m| < 2) → m ∈ Set.Ioo 1 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_m_range_l876_87697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l876_87633

-- Define the three lines
def line1 : ℝ → ℝ := λ _ => 8
def line2 : ℝ → ℝ := λ x => 2 + 2*x
def line3 : ℝ → ℝ := λ x => 2 - 2*x

-- Define the intersection points
def point1 : ℝ × ℝ := (3, 8)
def point2 : ℝ × ℝ := (-3, 8)
def point3 : ℝ × ℝ := (0, 2)

-- Define the triangle
def triangle : Set (ℝ × ℝ) := {point1, point2, point3}

-- Theorem statement
theorem triangle_area : 
  let base := |point1.1 - point2.1|
  let height := |line1 0 - point3.2|
  (1/2 : ℝ) * base * height = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l876_87633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remi_water_spill_correct_l876_87613

def remi_water_spill (bottle_capacity : ℕ) (refills_per_day : ℕ) (days : ℕ) 
  (first_spill : ℕ) (total_drunk : ℕ) : ℕ :=
  let total_without_spills := bottle_capacity * refills_per_day * days
  let second_spill := total_without_spills - first_spill - total_drunk
  second_spill

theorem remi_water_spill_correct (bottle_capacity refills_per_day days first_spill total_drunk : ℕ) :
  remi_water_spill bottle_capacity refills_per_day days first_spill total_drunk =
  bottle_capacity * refills_per_day * days - first_spill - total_drunk :=
by
  unfold remi_water_spill
  simp

#eval remi_water_spill 20 3 7 5 407

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remi_water_spill_correct_l876_87613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l876_87671

/-- A function f is even if f(-x) = f(x) for all x in its domain --/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x * exp(x) / (exp(a*x) - 1) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

/-- If f(x) = x * exp(x) / (exp(a*x) - 1) is an even function, then a = 2 --/
theorem f_even_implies_a_eq_two (a : ℝ) :
  IsEven (f a) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l876_87671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_RPQ_measure_l876_87653

-- Define the points
variable (P Q R S : EuclideanSpace ℝ (Fin 2))

-- Define the angle measure function
noncomputable def angle_measure (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Define the line function
def line (A B : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define z as a real number
variable (z : ℝ)

-- Define the conditions
axiom P_on_RS : P ∈ line R S
axiom QP_bisects_SQR : angle_measure S Q P = angle_measure P Q R
axiom PQ_eq_PR : dist P Q = dist P R
axiom RSQ_angle : angle_measure R S Q = 3 * z
axiom RPQ_angle : angle_measure R P Q = 4 * z

-- Theorem to prove
theorem RPQ_measure :
  angle_measure R P Q = 720 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_RPQ_measure_l876_87653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_expressions_l876_87651

theorem max_prime_expressions (m n : ℕ+) (h : m > n) : 
  (∃ (f : Fin 4 → ℕ), 
    (f 0 = m + n) ∧ 
    (f 1 = m - n) ∧ 
    (f 2 = m * n) ∧ 
    (f 3 = m / n) ∧ 
    (∃ (g : Fin 4 → Bool), (∀ i, g i = Nat.Prime (f i)) ∧ (Finset.sum (Finset.univ : Finset (Fin 4)) (fun i => if g i then 1 else 0) ≤ 3))) ∧ 
  (∀ (f : Fin 4 → ℕ), 
    (f 0 = m + n) → 
    (f 1 = m - n) → 
    (f 2 = m * n) → 
    (f 3 = m / n) → 
    (∀ (g : Fin 4 → Bool), (∀ i, g i = Nat.Prime (f i)) → (Finset.sum (Finset.univ : Finset (Fin 4)) (fun i => if g i then 1 else 0) ≤ 3))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_expressions_l876_87651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_triangle_area_ratio_for_given_conditions_l876_87693

/-- Represents a right triangle divided into a square and two smaller right triangles -/
structure DividedRightTriangle where
  square_side : ℝ
  small_triangle_area_ratio : ℝ
  h_square_side_positive : 0 < square_side
  h_small_triangle_area_ratio_positive : 0 < small_triangle_area_ratio

/-- The ratio of the area of the other small right triangle to the area of the square -/
noncomputable def other_triangle_area_ratio (t : DividedRightTriangle) : ℝ :=
  1 / (4 * t.small_triangle_area_ratio)

theorem other_triangle_area_ratio_for_given_conditions 
  (t : DividedRightTriangle) 
  (h_square_side : t.square_side = 2) : 
  other_triangle_area_ratio t = 1 / (4 * t.small_triangle_area_ratio) := by
  -- Unfold the definition of other_triangle_area_ratio
  unfold other_triangle_area_ratio
  -- The definition matches the right-hand side exactly, so we're done
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_triangle_area_ratio_for_given_conditions_l876_87693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_with_winning_strategy_l876_87610

/-- Represents the state of a card (White or Black) -/
inductive CardState
| White
| Black

/-- Represents the game state -/
structure GameState where
  cards : List CardState
  currentPlayer : Nat

/-- Defines a valid move in the game -/
def validMove (state : GameState) (k : Nat) : Prop :=
  k < state.cards.length - 40 ∧ 
  state.cards[k]? = some CardState.White

/-- Applies a move to the game state -/
def applyMove (state : GameState) (k : Nat) : GameState :=
  { state with 
    cards := state.cards.take k ++ List.replicate 41 CardState.Black ++ state.cards.drop (k + 41),
    currentPlayer := if state.currentPlayer = 1 then 2 else 1 }

/-- Defines if the game has ended -/
def gameEnded (state : GameState) : Prop :=
  ∀ k, ¬validMove state k

/-- The main theorem to be proved -/
theorem game_ends_with_winning_strategy : 
  ∃ (n : Nat), ∀ (initialState : GameState), 
    initialState.cards.length = 2009 ∧ 
    (∀ i, i < 2009 → initialState.cards[i]? = some CardState.White) →
    (∃ (finalState : GameState), 
      (∃ (moves : List Nat), moves.length ≤ n ∧ 
        finalState = moves.foldl applyMove initialState) ∧
      gameEnded finalState) ∧
    (∃ (strategy : GameState → Nat), 
      ∀ (state : GameState), 
        state.currentPlayer = 1 → 
        validMove state (strategy state) ∧
        (gameEnded (applyMove state (strategy state)) ∨
         ¬∃ (opponentMove : Nat), 
           validMove (applyMove state (strategy state)) opponentMove ∧
           ¬gameEnded (applyMove (applyMove state (strategy state)) opponentMove))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_with_winning_strategy_l876_87610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_perpendicular_chord_projection_l876_87620

/-- 
Given a parabola y^2 = 2px (p > 0), if two perpendicular lines through the origin 
intersect the parabola at points A and B, then the locus of the projection M of 
the origin O on AB is described by the equation x^2 - 2px + y^2 = 0 (where x > 0).
-/
theorem parabola_perpendicular_chord_projection (p : ℝ) (hp : p > 0) :
  ∀ (A B M : ℝ × ℝ),
  (fun x => x.2^2 = 2*p*x.1) A ∧
  (fun x => x.2^2 = 2*p*x.1) B ∧
  (A.1 * B.1 + A.2 * B.2 = 0) →
  (M.1 > 0 ∧ ∃ t : ℝ, M = (1-t) • A + t • B) →
  M.1^2 - 2*p*M.1 + M.2^2 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_perpendicular_chord_projection_l876_87620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_minus_reciprocal_l876_87645

open Real

theorem tan_minus_reciprocal (θ : ℝ) : 
  θ ∈ Set.Ioo 0 π → 
  (∃ x y : ℝ, x = sin θ ∧ y = cos θ ∧ 25 * x^2 - 5 * x - 12 = 0 ∧ 25 * y^2 - 5 * y - 12 = 0) →
  tan θ - (tan θ)⁻¹ = -7/12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_minus_reciprocal_l876_87645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l876_87685

-- Define the conditions
noncomputable def a (x : ℝ) : ℝ := x - Real.sqrt 2
noncomputable def b (x : ℝ) : ℝ := x - 1/x
noncomputable def c (x : ℝ) : ℝ := x + 1/x
noncomputable def d (x : ℝ) : ℝ := x^2 + 2 * Real.sqrt 2

-- Define a predicate for whether a real number is an integer
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

-- Define the property that exactly one of a, b, c, d is not an integer
def exactly_one_not_integer (x : ℝ) : Prop :=
  (¬ isInteger (a x) ∧ isInteger (b x) ∧ isInteger (c x) ∧ isInteger (d x)) ∨
  (isInteger (a x) ∧ ¬ isInteger (b x) ∧ isInteger (c x) ∧ isInteger (d x)) ∨
  (isInteger (a x) ∧ isInteger (b x) ∧ ¬ isInteger (c x) ∧ isInteger (d x)) ∨
  (isInteger (a x) ∧ isInteger (b x) ∧ isInteger (c x) ∧ ¬ isInteger (d x))

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, exactly_one_not_integer x ∧ x = Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l876_87685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l876_87640

def A : Set ℕ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℕ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l876_87640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_implies_a_range_l876_87655

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x * Real.log 2) + a

-- State the theorem
theorem zero_in_interval_implies_a_range :
  ∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Ioo (-1) 1 ∧ f x a = 0) → a ∈ Set.Ioo (-3) (1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_implies_a_range_l876_87655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_addition_and_averaging_properties_l876_87627

noncomputable def avg (a b : ℝ) : ℝ := (a + b) / 2

theorem addition_and_averaging_properties :
  (∀ x y z : ℝ, (x + y) + z = x + (y + z)) ∧
  (∀ x y : ℝ, avg x y = avg y x) ∧
  (∀ x y z : ℝ, x + avg y z = avg (x + y) (x + z)) ∧
  (∃ x y z : ℝ, avg x (y + z) ≠ avg x y + avg x z) ∧
  (¬ ∃ e : ℝ, ∀ x : ℝ, avg x e = x) :=
by
  sorry

#check addition_and_averaging_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_addition_and_averaging_properties_l876_87627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractions_converges_to_one_l876_87678

theorem sum_of_fractions_converges_to_one :
  let S := {f : ℚ | ∃ (a b : ℕ), f = (1 : ℚ) / ((↑(a + 1) : ℚ) ^ (b + 1))}
  ∃ (L : ℚ), IsLUB S L ∧ L = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractions_converges_to_one_l876_87678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_hike_l876_87683

/-- Billy's hiking path --/
structure HikingPath where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Calculate the distance between two points --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Billy's hiking theorem --/
theorem billy_hike (path : HikingPath) :
  path.A = (0, 0) →
  path.B = (6, 0) →
  path.C = (6 + 4/Real.sqrt 2, 4/Real.sqrt 2) →
  path.D = (6 + 4/Real.sqrt 2 + 8 * Real.cos (15 * π / 180),
            4/Real.sqrt 2 + 8 * Real.sin (15 * π / 180)) →
  abs (distance path.A path.D - 17.26) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_hike_l876_87683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l876_87675

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_calculation (principal time interest : ℝ) 
  (h_principal : principal = 8925)
  (h_time : time = 5)
  (h_interest : interest = 4016.25) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 9 := by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l876_87675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_probability_l876_87680

def box_C : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 30) (Finset.range 31)
def box_D : Finset ℕ := Finset.filter (λ n => 21 ≤ n ∧ n ≤ 50) (Finset.range 51)

def favorable_C : Finset ℕ := Finset.filter (λ n => n < 20) box_C
def favorable_D : Finset ℕ := Finset.filter (λ n => n % 2 = 1 ∨ n > 40) box_D

theorem tile_probability :
  (Finset.card favorable_C * Finset.card favorable_D : ℚ) / 
  (Finset.card box_C * Finset.card box_D) = 19 / 45 := by
  sorry

#eval Finset.card favorable_C -- Should output 19
#eval Finset.card favorable_D -- Should output 20
#eval Finset.card box_C -- Should output 30
#eval Finset.card box_D -- Should output 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_probability_l876_87680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_initial_stock_l876_87669

/-- Represents the number of cans in Carla's food bank inventory --/
def Inventory := ℕ

/-- Represents the number of people who took food on a given day --/
def People := ℕ

/-- Represents the number of cans taken per person on a given day --/
def CansPerPerson := ℕ

/-- Represents the number of cans restocked after a given day --/
def Restock := ℕ

instance : OfNat People n where
  ofNat := n

instance : OfNat CansPerPerson n where
  ofNat := n

instance : OfNat Restock n where
  ofNat := n

instance : OfNat Inventory n where
  ofNat := n

theorem carla_initial_stock (
  day1_people : People
  ) (day1_cans_per_person : CansPerPerson
  ) (day1_restock : Restock
  ) (day2_people : People
  ) (day2_cans_per_person : CansPerPerson
  ) (day2_restock : Restock
  ) (total_given_away : Inventory
  ) (h1 : day1_people = 500
  ) (h2 : day1_cans_per_person = 1
  ) (h3 : day1_restock = 1500
  ) (h4 : day2_people = 1000
  ) (h5 : day2_cans_per_person = 2
  ) (h6 : day2_restock = 3000
  ) (h7 : total_given_away = 2500
  ) : Inventory :=
by
  sorry

#check carla_initial_stock

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_initial_stock_l876_87669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_18pi_l876_87626

/-- Configuration of circles where two smaller circles touch a larger circle and each other at the center of the larger circle -/
structure CircleConfiguration where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of each smaller circle
  touch_condition : r = R / 2  -- Condition for circles to touch as described

/-- The area of the shaded region in the circle configuration -/
noncomputable def shaded_area (c : CircleConfiguration) : ℝ :=
  Real.pi * c.R^2 - 2 * Real.pi * c.r^2

/-- Theorem stating that for a circle configuration with larger circle radius 6, 
    the shaded area is 18π -/
theorem shaded_area_is_18pi (c : CircleConfiguration) 
    (h : c.R = 6) : shaded_area c = 18 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_18pi_l876_87626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l876_87649

/-- An ellipse C in the Cartesian coordinate system xOy -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the Cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

def Ellipse.equation (C : Ellipse) (P : Point) : Prop :=
  P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1

noncomputable def Ellipse.eccentricity (C : Ellipse) : ℝ :=
  Real.sqrt (C.a^2 - C.b^2) / C.a

def Line.intersect_ellipse (l : Line) (C : Ellipse) : Set Point :=
  {P : Point | C.equation P ∧ P.y = l.slope * P.x + l.intercept}

theorem ellipse_properties (C : Ellipse) (P : Point) (l : Line) :
  C.equation P ∧ 
  P.x = 1 ∧ P.y = 3/2 ∧
  C.eccentricity = 1/2 ∧
  l.slope = Real.sqrt 3 / 2 →
  (C.a = 2 ∧ C.b = Real.sqrt 3) ∧
  (∀ A B, A ∈ l.intersect_ellipse C → B ∈ l.intersect_ellipse C → A.x^2 + A.y^2 + B.x^2 + B.y^2 = 7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l876_87649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l876_87623

theorem sqrt_calculations : 
  (Real.sqrt 27 - Real.sqrt 75 + Real.sqrt 3 = -Real.sqrt 3) ∧
  ((Real.sqrt 5 + Real.sqrt 35) / Real.sqrt 5 = 1 + Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l876_87623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l876_87604

/-- The distance between two points in a 2D plane. -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem: The distance between point A(1,0) and point B(4,4) is 5. -/
theorem distance_A_to_B :
  distance 1 0 4 4 = 5 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression under the square root
  simp
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l876_87604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2A_value_l876_87692

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.sin (x + Real.pi / 4))^2 - (Real.cos x)^2 - (1 + Real.sqrt 3) / 2

theorem cos_2A_value (A : ℝ) (m n : Fin 2 → ℝ) 
  (h_acute : 0 < A ∧ A < Real.pi / 2)
  (h_f : f = f)
  (h_m : m = ![1, 5])
  (h_n : n = ![1, f (Real.pi / 4 - A)])
  (h_perp : m 0 * n 0 + m 1 * n 1 = 0) :
  Real.cos (2 * A) = (4 * Real.sqrt 3 + 3) / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2A_value_l876_87692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_polynomial_l876_87625

theorem divides_polynomial (a : ℤ) : 
  (∀ x : ℤ, (x^2 - x + a) ∣ (x^15 + x + 100)) ↔ a = 2 := by
  sorry

#check divides_polynomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_polynomial_l876_87625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_queen_spade_probability_l876_87605

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
| Hearts | Diamonds | Clubs | Spades

/-- Represents the rank of a card -/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents a playing card -/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- Returns true if the card is a Queen -/
def is_queen (c : Card) : Prop :=
  c.rank = Rank.Queen

/-- Returns true if the card is a Spade -/
def is_spade (c : Card) : Prop :=
  c.suit = Suit.Spades

/-- The number of Queens in a standard deck -/
def queen_count : ℕ := 4

/-- The number of Spades in a standard deck -/
def spade_count : ℕ := 13

/-- Theorem stating the probability of drawing a Queen first and a Spade second -/
theorem queen_spade_probability (d : Deck) :
  (queen_count : ℚ) / 52 * (spade_count : ℚ) / 51 = 289 / 14968 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_queen_spade_probability_l876_87605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_common_one_position_l876_87647

/-- A binary sequence of length n -/
def BinarySeq (n : ℕ) := Fin n → Bool

/-- The set of 2^(n-1) binary sequences -/
def SequenceSet (n : ℕ) := Finset (BinarySeq n)

/-- The property that any three sequences share at least one position with digit 1 -/
def SharesOnePosition (s : SequenceSet n) : Prop :=
  ∀ (x y z : BinarySeq n), x ∈ s.toSet → y ∈ s.toSet → z ∈ s.toSet →
    ∃ (i : Fin n), x i ∧ y i ∧ z i

/-- The main theorem: there exists a unique position where all sequences have digit 1 -/
theorem unique_common_one_position
  (n : ℕ) (s : SequenceSet n)
  (h1 : s.card = 2^(n-1))
  (h2 : SharesOnePosition s) :
  ∃! (i : Fin n), ∀ (x : BinarySeq n), x ∈ s.toSet → x i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_common_one_position_l876_87647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_probabilities_l876_87677

structure BasketballGame where
  player_A_accuracy : ℝ
  player_B_accuracy : ℝ
  first_shot_prob : ℝ

noncomputable def prob_B_second_shot (game : BasketballGame) : ℝ :=
  game.first_shot_prob * (1 - game.player_A_accuracy) + game.first_shot_prob * game.player_B_accuracy

noncomputable def prob_A_ith_shot (game : BasketballGame) (i : ℕ) : ℝ :=
  1/3 + (1/6) * (2/5)^(i-1)

noncomputable def expected_A_shots (game : BasketballGame) (n : ℕ) : ℝ :=
  (5/18) * (1 - (2/5)^n) + n/3

theorem basketball_game_probabilities (game : BasketballGame) 
  (h1 : game.player_A_accuracy = 0.6)
  (h2 : game.player_B_accuracy = 0.8)
  (h3 : game.first_shot_prob = 0.5) :
  prob_B_second_shot game = 0.6 ∧
  (∀ i : ℕ, prob_A_ith_shot game i = 1/3 + (1/6) * (2/5)^(i-1)) ∧
  (∀ n : ℕ, expected_A_shots game n = (5/18) * (1 - (2/5)^n) + n/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_probabilities_l876_87677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_3rd_4th_5th_terms_l876_87612

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem sum_of_3rd_4th_5th_terms 
  (a : ℕ → ℝ) 
  (h1 : geometric_sequence a)
  (h2 : ∀ n, a n > 0)
  (h3 : ∀ n, a (n + 1) = 2 * a n)
  (h4 : a 1 + a 2 + a 3 = 21) :
  a 3 + a 4 + a 5 = 84 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_3rd_4th_5th_terms_l876_87612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l876_87619

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

def transformed_function (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x ↦ f (2 * x)

theorem transformation_result :
  transformed_function original_function = λ x ↦ Real.sin (2 * x + Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l876_87619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_13_l876_87603

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h₁ : d ≠ 0
  h₂ : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def ArithmeticSequence.sum (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_sum_13 (seq : ArithmeticSequence) 
    (h : seq.a 4 ^ 2 + seq.a 6 ^ 2 = seq.a 8 ^ 2 + seq.a 10 ^ 2) : 
    seq.sum 13 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_13_l876_87603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traditional_is_adjective_l876_87672

theorem traditional_is_adjective : 
  "传统的" = "traditional" :=
by
  -- The proof that "传统的" translates to "traditional"
  sorry

#check traditional_is_adjective

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traditional_is_adjective_l876_87672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_for_regular_pyramid_l876_87617

/-- A point in 3D space --/
structure Point :=
  (x y z : ℝ)

/-- A triangle in 3D space --/
structure Triangle :=
  (a b c : Point)

/-- A pyramid is a polyhedron with a polygonal base and triangular faces meeting at a point (the apex) --/
structure Pyramid :=
  (base : Set Point)
  (apex : Point)
  (lateral_faces : Set Triangle)

/-- A regular pyramid has a regular polygon as its base and congruent isosceles triangles as its lateral faces --/
def is_regular_pyramid (p : Pyramid) : Prop := sorry

/-- An equilateral triangle is a triangle with all sides equal --/
def is_equilateral_triangle (t : Triangle) : Prop := sorry

/-- This theorem states that if all lateral faces of a pyramid are equilateral triangles, 
    then the pyramid is regular, but not all regular pyramids have equilateral lateral faces --/
theorem sufficient_not_necessary_condition_for_regular_pyramid (p : Pyramid) :
  (∀ t ∈ p.lateral_faces, is_equilateral_triangle t) → is_regular_pyramid p ∧
  ¬(is_regular_pyramid p → ∀ t ∈ p.lateral_faces, is_equilateral_triangle t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_for_regular_pyramid_l876_87617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l876_87616

/-- The area of a triangle inscribed in a circle, where the vertices divide the circle into three arcs of lengths 5, 7, and 8 -/
noncomputable def triangleArea : ℝ := 138.005 / Real.pi^2

/-- The lengths of the three arcs -/
def arcLengths : List ℝ := [5, 7, 8]

/-- Predicate to check if a set is a circle -/
def IsCircle (s : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate to check if a triangle is inscribed in a circle -/
def IsInscribed (triangle : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) : Prop := sorry

/-- Function to calculate the arc length of a set -/
noncomputable def arcLength (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Function to calculate the area of a set -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem inscribed_triangle_area :
  ∀ (circle : Set (ℝ × ℝ)) (triangle : Set (ℝ × ℝ)),
  IsCircle circle →
  IsInscribed triangle circle →
  (∃ (a b c : Set (ℝ × ℝ)), a ∪ b ∪ c = circle ∧ 
    arcLength a = 5 ∧ arcLength b = 7 ∧ arcLength c = 8) →
  area triangle = triangleArea := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l876_87616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l876_87660

noncomputable def inverse_proportion (x : ℝ) : ℝ := 2 / x

theorem inverse_proportion_properties :
  (∀ x : ℝ, x ≠ 0 → inverse_proportion x = 2 / x) ∧
  (inverse_proportion 1 = 2) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ < x₂ → inverse_proportion x₁ > inverse_proportion x₂) ∧
  (∀ x : ℝ, x > 0 → inverse_proportion x > 0) ∧
  (∀ x : ℝ, x < 0 → inverse_proportion x < 0) ∧
  (∀ x : ℝ, x > 1 → 0 < inverse_proportion x ∧ inverse_proportion x < 2) :=
by
  sorry

#check inverse_proportion_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l876_87660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_line_l876_87618

/-- The distance from the origin to a line ax + by + c = 0 is |c| / √(a^2 + b^2) -/
noncomputable def distanceFromOriginToLine (a b c : ℝ) : ℝ :=
  abs c / Real.sqrt (a^2 + b^2)

/-- The line equation x + 2y - 5 = 0 -/
def lineEquation (x y : ℝ) : Prop :=
  x + 2*y - 5 = 0

theorem distance_to_specific_line :
  distanceFromOriginToLine 1 2 (-5) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_line_l876_87618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polyhedra_similarity_l876_87696

/-- A regular polyhedron -/
structure RegularPolyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))
  isRegular : Prop

/-- The type (kind) of a regular polyhedron -/
inductive PolyhedronType
  | Tetrahedron
  | Cube
  | Octahedron
  | Dodecahedron
  | Icosahedron

/-- Two shapes are similar if one can be transformed into the other via scaling and translation -/
def IsSimilar (p1 p2 : RegularPolyhedron) : Prop := sorry

/-- The type of a regular polyhedron -/
def getPolyhedronType (p : RegularPolyhedron) : PolyhedronType := sorry

/-- The center of a regular polyhedron -/
def Center (p : RegularPolyhedron) : Fin 3 → ℝ := sorry

/-- The number of faces of a regular polyhedron -/
def NumFaces (p : RegularPolyhedron) : ℕ := sorry

/-- A polyhedral angle of a regular polyhedron -/
structure PolyhedralAngle (p : RegularPolyhedron) where
  vertex : Fin 3 → ℝ
  edges : Set (Set (Fin 3 → ℝ))

/-- The set of all polyhedral angles of a regular polyhedron -/
def PolyhedralAngles (p : RegularPolyhedron) : Set (PolyhedralAngle p) := sorry

theorem regular_polyhedra_similarity 
  (p1 p2 : RegularPolyhedron) 
  (h : getPolyhedronType p1 = getPolyhedronType p2) :
  IsSimilar p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polyhedra_similarity_l876_87696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_min_volume_l876_87648

/-- Represents a tetrahedron APQR in a unit cube --/
structure Tetrahedron where
  -- Lengths of AP, AQ, AR respectively
  p : ℝ
  q : ℝ
  r : ℝ
  -- Constraints ensuring the tetrahedron is inside the unit cube
  h_positive : 0 < p ∧ 0 < q ∧ 0 < r
  h_cube : p ≤ 1 ∧ q ≤ 1 ∧ r ≤ 1
  -- Constraint from the geometry of the problem
  h_plane : 1/p + 1/q + 1/r = 1

/-- The volume of the tetrahedron --/
noncomputable def volume (t : Tetrahedron) : ℝ := t.p * t.q * t.r / 6

/-- The theorem to be proved --/
theorem tetrahedron_min_volume :
  ∀ t : Tetrahedron, volume t ≥ 4.5 ∧
  (volume t = 4.5 ↔ t.p = t.q ∧ t.q = t.r) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_min_volume_l876_87648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_intersections_l876_87629

-- Define the circles in polar coordinates
noncomputable def C1 (θ : Real) : Real := 4 * Real.cos θ
noncomputable def C2 (θ : Real) : Real := 2 * Real.sin θ

-- Define the ray OM
def ray_OM (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define the intersection points
noncomputable def point_P (α : Real) : Real := C1 α
noncomputable def point_Q (α : Real) : Real := C2 α

-- State the theorem
theorem max_product_of_intersections :
  ∃ (max : Real), max = 4 ∧
  ∀ α, ray_OM α →
    point_P α * point_Q α ≤ max ∧
    ∃ α₀, ray_OM α₀ ∧ point_P α₀ * point_Q α₀ = max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_intersections_l876_87629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_walking_speed_l876_87686

/-- Calculates the speed in km/hr given the distance in km and time in minutes -/
noncomputable def calculate_speed (distance : ℝ) (time_minutes : ℝ) : ℝ :=
  distance / (time_minutes / 60)

theorem man_walking_speed :
  let distance : ℝ := 7
  let time_minutes : ℝ := 42
  calculate_speed distance time_minutes = 10 := by
  -- Unfold the definition of calculate_speed
  unfold calculate_speed
  -- Perform the calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_walking_speed_l876_87686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_monotonic_condition_l876_87654

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- Define the interval [-5, 5]
def I : Set ℝ := Set.Icc (-5) 5

-- Theorem for part (1)
theorem max_min_values (h : Set.Nonempty I) :
  (∃ x ∈ I, ∀ y ∈ I, f (-1) y ≤ f (-1) x) ∧
  (∃ x ∈ I, f (-1) x = 37) ∧
  (∃ x ∈ I, ∀ y ∈ I, f (-1) x ≤ f (-1) y) ∧
  (∃ x ∈ I, f (-1) x = 1) := by sorry

-- Theorem for part (2)
theorem monotonic_condition (h : Set.Nonempty I) :
  ∀ a : ℝ, (∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f a x < f a y) ∨ 
            (∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f a x > f a y)
  ↔ a ∈ Set.Iic (-5) ∪ Set.Ici 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_monotonic_condition_l876_87654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_points_properties_l876_87699

/-- A line parallel to the Oy axis that cuts a segment of length 3 from the Ox axis -/
def parallel_line : Set (ℝ × ℝ) := {p | p.1 = 3}

/-- Point A -/
def point_A : ℝ × ℝ := (3, 4)

/-- Point B -/
def point_B : ℝ × ℝ := (-3, 2)

/-- Theorem stating the properties of the line and the points -/
theorem line_and_points_properties :
  (∀ p ∈ parallel_line, p.1 = 3) ∧
  (point_A ∈ parallel_line) ∧
  (point_B ∉ parallel_line) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_points_properties_l876_87699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_example_min_distance_curves_range_of_a_range_of_m_l876_87676

-- 1. Polar to Cartesian conversion
noncomputable def polar_to_cartesian (ρ : ℝ) (θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

theorem polar_to_cartesian_example : 
  polar_to_cartesian 2 (π / 6) = (Real.sqrt 3, 1) := by sorry

-- 2. Minimum distance between curves
noncomputable def curve1 (θ : ℝ) : ℝ := 2 / Real.sin θ
noncomputable def curve2 (θ : ℝ) : ℝ := 2 / Real.cos θ

theorem min_distance_curves : 
  ∃ (d : ℝ), d = 1 ∧ ∀ (θ₁ θ₂ : ℝ), 
    ((polar_to_cartesian (curve1 θ₁) θ₁).1 - (polar_to_cartesian (curve2 θ₂) θ₂).1)^2 +
    ((polar_to_cartesian (curve1 θ₁) θ₁).2 - (polar_to_cartesian (curve2 θ₂) θ₂).2)^2 ≥ d^2 := by sorry

-- 3. Range of a for inequality
theorem range_of_a : 
  ∀ (a : ℝ), (∃ (x : ℝ), 4 - x^2 ≥ |x - a| + a) ↔ a ≤ 17/8 := by sorry

-- 4. Range of m given conditions
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m : 
  ∀ (m : ℝ), m > 0 ∧ 
    (∀ (x : ℝ), ¬(p x) → ¬(q x m)) ∧ 
    (∃ (x : ℝ), ¬(p x) ∧ (q x m)) 
  ↔ m ≥ 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_example_min_distance_curves_range_of_a_range_of_m_l876_87676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_divisors_90_l876_87621

def sum_of_odd_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d % 2 = 1) (Nat.divisors n)).sum id

theorem sum_of_odd_divisors_90 :
  sum_of_odd_divisors 90 = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_divisors_90_l876_87621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l876_87687

/-- The function f(x) = (x-1)e^x - (k/2)x^2 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - k / 2 * x^2

/-- Theorem: When k > 0, the function f(x) has exactly one zero in its domain (-∞, +∞) -/
theorem f_has_unique_zero (k : ℝ) (h : k > 0) :
  ∃! x, f k x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l876_87687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tangent_l876_87601

noncomputable section

open Real

theorem parallel_vectors_tangent (θ : ℝ) 
  (h1 : 0 < θ) (h2 : θ < π/2)
  (a : Fin 2 → ℝ) (b : Fin 2 → ℝ)
  (ha : a = λ i => if i = 0 then Real.sin (2*θ) else Real.cos θ)
  (hb : b = λ i => if i = 0 then Real.cos θ else 1)
  (h_parallel : ∃ (k : ℝ), a = k • b) : 
  Real.tan θ = 1/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tangent_l876_87601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_fill_cost_l876_87639

/-- Represents a right circular cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The volume of a cylinder --/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- The cost to fill a cylinder given a base cost and volume --/
noncomputable def fillCost (baseCost : ℝ) (baseVolume : ℝ) (c : Cylinder) : ℝ :=
  (cylinderVolume c / baseVolume) * baseCost

theorem cylinder_fill_cost (r h baseCost : ℝ) :
  let canB := Cylinder.mk r h
  let canC := Cylinder.mk (2*r) (h/2)
  let canA := Cylinder.mk (3*r) (h/3)
  let baseVolume := cylinderVolume canB / 2
  fillCost baseCost baseVolume canC + fillCost baseCost baseVolume canA = 40 :=
by
  sorry

#check cylinder_fill_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_fill_cost_l876_87639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_triangle_inequality_l876_87663

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A lattice triangle defined by three lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- The length of a side of a lattice triangle -/
noncomputable def sideLength (p1 p2 : LatticePoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 : ℝ)

/-- The perimeter of a lattice triangle -/
noncomputable def perimeter (t : LatticeTriangle) : ℝ :=
  sideLength t.A t.B + sideLength t.B t.C + sideLength t.C t.A

theorem lattice_triangle_inequality (t : LatticeTriangle) :
  sideLength t.A t.B > sideLength t.A t.C →
  sideLength t.A t.B - sideLength t.A t.C > 1 / perimeter t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_triangle_inequality_l876_87663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l876_87609

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- State the theorem
theorem f_sum_equals_two :
  (∀ x : ℝ, f (x + 1) = x^2 + x) → f 1 + f (-1) = 2 := by
  intro h
  have h1 : f 1 = 0 := by
    rw [f]
    ring
  have h2 : f (-1) = 2 := by
    rw [f]
    ring
  rw [h1, h2]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l876_87609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2013_equals_4_l876_87674

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Adding this case to handle Nat.zero
  | 1 => 2
  | 2 => 7
  | (n + 3) => (sequence_a (n + 1) * sequence_a (n + 2)) % 10

theorem a_2013_equals_4 : sequence_a 2013 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2013_equals_4_l876_87674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_width_approx_l876_87642

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- A rectangle with golden ratio proportions -/
structure GoldenRectangle where
  length : ℝ
  width : ℝ
  golden_ratio : width / length = 1 / φ

/-- The book's dimensions -/
noncomputable def book : GoldenRectangle where
  length := 20
  width := 20 / φ
  golden_ratio := by sorry

theorem book_width_approx : 
  ∃ ε > 0, ε < 0.01 ∧ |book.width - 12.36| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_width_approx_l876_87642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_value_l876_87614

/-- Given vectors a, b, c in R², if (a + kb) ∥ c, then k = 1/2 -/
theorem parallel_vectors_k_value (a b c : ℝ × ℝ) (k : ℝ) 
  (ha : a = (2, -1)) 
  (hb : b = (1, 1)) 
  (hc : c = (-5, 1)) 
  (h_parallel : ∃ t : ℝ, t ≠ 0 ∧ a + k • b = t • c) : 
  k = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_value_l876_87614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_can_prevent_divisibility_l876_87607

/-- A strategy for Alice in the digit game. -/
def AliceStrategy := Nat → Nat → Nat

/-- The result of the game after all moves are made. -/
structure GameResult where
  finalNumber : Nat
  digits : List Nat

/-- Checks if a number is divisible by 3. -/
def divisibleBy3 (n : Nat) : Prop := n % 3 = 0

/-- Checks if two numbers are in different residue classes modulo 3. -/
def differentResidueClass (a b : Nat) : Prop := a % 3 ≠ b % 3

/-- Simulates the game with Alice's strategy and Bob's best efforts. -/
def playGame (strategy : AliceStrategy) : GameResult :=
  sorry

/-- Alice can prevent Bob from making the number divisible by 3. -/
theorem alice_can_prevent_divisibility : ∃ (strategy : AliceStrategy),
  ¬(divisibleBy3 (playGame strategy).finalNumber) ∧
  (∀ (i : Nat), i < 2017 → differentResidueClass ((playGame strategy).digits.get! i) ((playGame strategy).digits.get! (i+1))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_can_prevent_divisibility_l876_87607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_points_sum_l876_87641

/-- Square with side length 1200 -/
structure Square where
  center : Real × Real
  side_length : ℝ
  is_square : side_length = 1200

/-- Points on the diagonal of the square -/
structure DiagonalPoints (S : Square) where
  G : Real × Real
  H : Real × Real
  on_diagonal : True  -- Simplified condition
  G_between_W_H : True  -- Simplified condition
  WG_less_HG : True  -- Simplified condition
  angle_GOH : True  -- Simplified condition
  GH_length : dist G H = 500

/-- The form of HG as q + r√s -/
structure HGForm (S : Square) (D : DiagonalPoints S) where
  q : ℕ
  r : ℕ
  s : ℕ
  positive : q > 0 ∧ r > 0 ∧ s > 0
  s_not_square : ∀ (p : ℕ), Nat.Prime p → s % (p^2) ≠ 0
  HG_eq : dist D.H (S.center.1 + 1200, S.center.2) = q + r * Real.sqrt s

theorem square_diagonal_points_sum (S : Square) (D : DiagonalPoints S) (F : HGForm S D) :
  F.q + F.r + F.s = 303 := by
  sorry

#check square_diagonal_points_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_points_sum_l876_87641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_2023_divisors_l876_87634

theorem least_integer_with_2023_divisors :
  ∃ (n m k : ℕ),
    (∀ d : ℕ, d ∣ n ↔ d > 0 ∧ d ≤ n) →
    (Finset.card (Finset.filter (· ∣ n) (Finset.range (n + 1))) = 2023) →
    (∀ p : ℕ, p < n → Finset.card (Finset.filter (· ∣ p) (Finset.range (p + 1))) < 2023) →
    n = m * 12^k →
    ¬(12 ∣ m) →
    m = 2^10 * 3^13 →
    k = 3 →
    m + k = 1595347 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_2023_divisors_l876_87634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l876_87643

/-- A power function is of the form f(x) = ax^n where a is a constant and n is a real number -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), ∀ x, f x = a * x ^ n

/-- Given function f(x) = (m-2)x^(m^2-2m) -/
noncomputable def f (m : ℝ) : ℝ → ℝ := fun x ↦ (m - 2) * x ^ (m^2 - 2*m)

theorem power_function_m_value :
  ∀ m : ℝ, isPowerFunction (f m) → m = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l876_87643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_fifth_subset_l876_87622

/-- Definition of a k-th subset of M -/
def kth_subset (M : Finset ℕ) (k : ℕ) : Finset ℕ :=
  M.filter (fun i => ∃ j, j < M.card ∧ (k / 2^j) % 2 = 1)

theorem twenty_fifth_subset (M : Finset ℕ) (h : M.Nonempty) :
  kth_subset M 25 = {1, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_fifth_subset_l876_87622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_window_side_length_l876_87679

/-- Represents a trapezoidal glass pane -/
structure TrapezoidalPane where
  height : ℝ
  longerBase : ℝ
  area : ℝ

/-- Represents an octagonal window -/
structure OctagonalWindow where
  panes : Finset TrapezoidalPane
  borderWidth : ℝ

/-- The side length of an octagonal window from one corner to the directly opposite one -/
noncomputable def octagonSideLength (window : OctagonalWindow) : ℝ :=
  sorry

theorem octagon_window_side_length 
  (window : OctagonalWindow)
  (h1 : window.panes.card = 8)
  (h2 : ∀ p ∈ window.panes, p.height / p.longerBase = 5 / 3)
  (h3 : ∀ p ∈ window.panes, p.area = 120)
  (h4 : window.borderWidth = 3) :
  ∃ ε > 0, |octagonSideLength window - 59| < ε := by
  sorry

#check octagon_window_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_window_side_length_l876_87679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l876_87688

/-- Calculates the distance traveled downstream by a boat given its speed in still water, 
    the speed of the current, and the time traveled. -/
noncomputable def distance_downstream (boat_speed : ℝ) (current_speed : ℝ) (time_minutes : ℝ) : ℝ :=
  (boat_speed + current_speed) * (time_minutes / 60)

/-- Theorem stating that a boat with a speed of 20 km/hr in still water, 
    traveling in a current of 3 km/hr for 24 minutes, will cover 9.2 km downstream. -/
theorem boat_downstream_distance :
  distance_downstream 20 3 24 = 9.2 := by
  -- Unfold the definition of distance_downstream
  unfold distance_downstream
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l876_87688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_e_neg_two_i_in_third_quadrant_l876_87658

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define Euler's formula
axiom eulers_formula (x : ℝ) : Complex.exp (x * i) = Complex.cos x + i * Complex.sin x

-- Define the properties of cos and sin at x = -2
axiom cos_neg_two_negative : Complex.re (Complex.cos (-2 : ℝ)) < 0
axiom sin_neg_two_negative : Complex.re (Complex.sin (-2 : ℝ)) < 0

-- Define what it means for a complex number to be in the third quadrant
def in_third_quadrant (z : ℂ) : Prop :=
  Complex.re z < 0 ∧ Complex.im z < 0

-- State the theorem
theorem e_neg_two_i_in_third_quadrant :
  in_third_quadrant (Complex.exp (-2 * i)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_e_neg_two_i_in_third_quadrant_l876_87658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_specific_case_l876_87631

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a line with equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

noncomputable def Line.slope (l : Line) : ℝ := -l.A / l.B

noncomputable def Hyperbola.asymptote_slope (h : Hyperbola) : ℝ := h.b / h.a

noncomputable def Hyperbola.focus_distance (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2)

theorem hyperbola_specific_case 
  (h : Hyperbola) 
  (l : Line) 
  (h_line_eq : l.A = 1 ∧ l.B = -2 ∧ l.C = -5)
  (h_parallel : l.slope = h.asymptote_slope)
  (h_focus : l.C / l.A = -h.focus_distance) :
  h.a^2 = 20 ∧ h.b^2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_specific_case_l876_87631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_increasing_implies_k_geq_neg_two_l876_87662

/-- The function h(x) -/
noncomputable def h (k : ℝ) (x : ℝ) : ℝ := 2*x - k/x + k/3

/-- h(x) is increasing on (1, +∞) -/
def h_increasing (k : ℝ) : Prop :=
  ∀ x₁ x₂, 1 < x₁ → x₁ < x₂ → h k x₁ < h k x₂

theorem h_increasing_implies_k_geq_neg_two (k : ℝ) :
  h_increasing k → k ≥ -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_increasing_implies_k_geq_neg_two_l876_87662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_expression_change_l876_87670

theorem quadratic_expression_change (x b a : ℝ) (h : a > 0) :
  ∃ (sign₁ sign₂ : ℤ), sign₁ * sign₁ = 1 ∧ sign₂ * sign₂ = 1 ∧
    ((x + sign₁ * a)^2 + b * (x + sign₁ * a) - 6) - (x^2 + b * x - 6) =
    sign₁ * 2 * a * x + a^2 + sign₂ * b * a :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_expression_change_l876_87670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l876_87611

-- Define the curve
noncomputable def x (t : ℝ) : ℝ := t^3 + 1
noncomputable def y (t : ℝ) : ℝ := t^2

-- Define the parameter value
def t₀ : ℝ := -2

-- Define the point on the curve at t₀
noncomputable def x₀ : ℝ := x t₀
noncomputable def y₀ : ℝ := y t₀

-- Define the derivatives
noncomputable def dx_dt (t : ℝ) : ℝ := 3 * t^2
noncomputable def dy_dt (t : ℝ) : ℝ := 2 * t

-- Define the slope of the tangent line at t₀
noncomputable def m_tangent : ℝ := dy_dt t₀ / dx_dt t₀

-- Define the slope of the normal line
noncomputable def m_normal : ℝ := -1 / m_tangent

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∀ x, y₀ + m_tangent * (x - x₀) = -1/3 * x + 5/3 := by sorry

-- Theorem for the normal line equation
theorem normal_line_equation :
  ∀ x, y₀ + m_normal * (x - x₀) = 3 * x + 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l876_87611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l876_87637

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- The theorem states that the point A(0, 0, -7) is equidistant from B(6, -7, 1) and C(-1, 2, 5) -/
theorem equidistant_point :
  let A : Point3D := ⟨0, 0, -7⟩
  let B : Point3D := ⟨6, -7, 1⟩
  let C : Point3D := ⟨-1, 2, 5⟩
  distance A B = distance A C := by
  sorry

#check equidistant_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l876_87637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_m_l876_87659

theorem characterization_of_m (n : ℕ+) :
  {m : ℕ+ | ∃ (f : Polynomial ℤ),
    f.degree = n.val ∧
    f.coeff n.val ≠ 0 ∧
    (Nat.gcd m.val (Finset.gcd (Finset.range (n.val + 1)) (fun i => (f.coeff i).natAbs)) = 1) ∧
    (∀ k : ℤ, (m : ℤ) ∣ f.eval k)}
  =
  {m : ℕ+ | m ∣ n!} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_m_l876_87659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_cups_is_six_l876_87646

/-- Represents the painting rate and order details for Gina's Etsy store --/
structure PaintingOrder where
  rose_rate : ℕ  -- Roses painted per hour
  lily_rate : ℕ  -- Lilies painted per hour
  lily_order : ℕ  -- Number of lily cups ordered
  total_pay : ℕ  -- Total payment for the order in dollars
  hourly_rate : ℕ  -- Gina's hourly rate in dollars

/-- Calculates the number of rose cups in the order --/
def rose_cups_in_order (order : PaintingOrder) : ℕ :=
  let total_hours := order.total_pay / order.hourly_rate
  let lily_hours := order.lily_order / order.lily_rate
  let rose_hours := total_hours - lily_hours
  rose_hours * order.rose_rate

/-- Theorem stating that the number of rose cups in the given order is 6 --/
theorem rose_cups_is_six (order : PaintingOrder) 
  (h1 : order.rose_rate = 6)
  (h2 : order.lily_rate = 7)
  (h3 : order.lily_order = 14)
  (h4 : order.total_pay = 90)
  (h5 : order.hourly_rate = 30) : 
  rose_cups_in_order order = 6 := by
  sorry

#eval rose_cups_in_order { rose_rate := 6, lily_rate := 7, lily_order := 14, total_pay := 90, hourly_rate := 30 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_cups_is_six_l876_87646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_100_chips_l876_87668

/-- Represents a strip of chips -/
def Strip := List Nat

/-- Cost of swapping adjacent chips -/
def adjacent_swap_cost : Nat := 1

/-- Checks if a swap is free (exactly 3 chips between) -/
def is_free_swap (i j : Nat) : Bool :=
  (i < j && j - i = 4) || (j < i && i - j = 4)

/-- Reverses the order of chips on the strip -/
def reverse_strip (s : Strip) : Strip :=
  s.reverse

/-- Calculates the minimum cost to reverse the strip -/
def min_cost_to_reverse (s : Strip) : Nat :=
  sorry

/-- The main theorem stating the minimum cost for a strip of 100 chips -/
theorem min_cost_100_chips :
  ∀ s : Strip, s.length = 100 → min_cost_to_reverse s = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_100_chips_l876_87668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l876_87600

noncomputable def f (x : ℝ) := (Real.sin x + Real.cos x)^2 - Real.sqrt 3 * Real.cos (2 * x)

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), 0 < T' → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
    T = π ∧
    (∀ (x : ℝ), 0 ≤ x → x ≤ π/2 → f x ≤ 3) ∧
    f (5*π/12) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l876_87600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_years_to_half_age_correct_l876_87652

/-- The number of years it takes for a man to be half his father's age -/
noncomputable def years_to_half_age (father_age : ℝ) : ℝ :=
  let man_age := (2/5) * father_age
  (father_age - 2 * man_age) / (1/2)

theorem years_to_half_age_correct (father_age : ℝ) 
  (h : father_age = 50.000000000000014) :
  years_to_half_age father_age = 10.000000000000004 := by
  unfold years_to_half_age
  rw [h]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_years_to_half_age_correct_l876_87652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_expression_eval_l876_87682

theorem ceiling_expression_eval : 
  ⌈Real.sqrt (16/9 : ℝ)⌉ - ⌈(16/9 : ℝ)⌉ + ⌈((16/9 : ℝ)^2)⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_expression_eval_l876_87682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_ends_in_one_l876_87606

def M : Finset Nat := {21, 23, 25, 27, 29}
def N : Finset Nat := Finset.range 21

def ends_in_one (m n : Nat) : Bool :=
  (m^n) % 10 = 1

def count_ends_in_one : Nat :=
  M.sum (λ m => (N.filter (λ n => ends_in_one m (n + 2010))).card)

theorem probability_ends_in_one :
  (count_ends_in_one : Rat) / (M.card * N.card) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_ends_in_one_l876_87606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_equals_train_length_l876_87608

/-- The length of a platform that a train crosses in one minute --/
noncomputable def platform_length (train_speed : ℝ) (train_length : ℝ) : ℝ :=
  train_speed * (1 / 60) - train_length

/-- Theorem stating that the platform length equals the train length --/
theorem platform_equals_train_length (train_speed : ℝ) (train_length : ℝ) 
  (h1 : train_speed = 90) 
  (h2 : train_length = 750) : 
  platform_length train_speed train_length = train_length := by
  sorry

#check platform_equals_train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_equals_train_length_l876_87608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_proof_l876_87630

theorem certain_number_proof (a b n : ℕ) (q : Set ℕ) : 
  (∃ k₁ k₂ : ℕ, a = k₁ * n ∧ b = k₂ * n) →  -- a and b are multiples of n
  (q = Set.Icc a b) →  -- q is the set of consecutive integers between a and b, inclusive
  (Finset.card (Finset.filter (λ x => n ∣ x) (Finset.range (b - a + 1))) = 10) →  -- q contains 10 multiples of n
  (Finset.card (Finset.filter (λ x => 7 ∣ x) (Finset.range (b - a + 1))) = 19) →  -- q contains 19 multiples of 7
  n = 14 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_proof_l876_87630
