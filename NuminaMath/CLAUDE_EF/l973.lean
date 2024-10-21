import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l973_97371

noncomputable def f (x : ℝ) : ℝ := Real.cos x / x

theorem f_is_odd : ∀ x : ℝ, x ≠ 0 → f (-x) = -f x := by
  intros x hx
  simp [f]
  field_simp
  ring
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l973_97371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l973_97382

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-2 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Define the curve C' after transformation
noncomputable def curve_C' (x : ℝ) : ℝ := (9 / 4) * x^2

-- Define point M
def point_M : ℝ × ℝ := (-2, 0)

-- Theorem statement
theorem intersection_distance_product :
  ∃ (A B : ℝ × ℝ),
    (∃ (t_A t_B : ℝ),
      line_l t_A = A ∧
      line_l t_B = B ∧
      curve_C' A.1 = A.2 ∧
      curve_C' B.1 = B.2) ∧
    (point_M.1 - A.1)^2 + (point_M.2 - A.2)^2 *
    (point_M.1 - B.1)^2 + (point_M.2 - B.2)^2 = 9^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l973_97382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_cube_volume_ratio_is_one_sixth_l973_97308

/-- The ratio of the volume of a regular octahedron formed by joining the centers of adjoining faces of a cube to the volume of the cube -/
noncomputable def octahedron_cube_volume_ratio (x : ℝ) : ℝ :=
  let cube_side := 2 * x
  let cube_volume := cube_side ^ 3
  let octahedron_edge := x * Real.sqrt 2
  let octahedron_volume := (octahedron_edge ^ 3 * Real.sqrt 2) / 3
  octahedron_volume / cube_volume

/-- Theorem stating that the ratio of the volume of the octahedron to the volume of the cube is 1/6 -/
theorem octahedron_cube_volume_ratio_is_one_sixth (x : ℝ) (h : x > 0) :
  octahedron_cube_volume_ratio x = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_cube_volume_ratio_is_one_sixth_l973_97308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alok_present_age_l973_97361

-- Define the ages of Bipin, Alok, and Chandan
def bipin_age : ℕ → ℕ := sorry
def alok_age : ℕ → ℕ := sorry
def chandan_age : ℕ → ℕ := sorry

-- Define the conditions
axiom bipin_alok_relation : ∀ t, bipin_age t = 6 * alok_age t
axiom future_relation : ∀ t, bipin_age t + 10 = 2 * (chandan_age t + 10)
axiom chandan_birthday : ∀ t, chandan_age t = 10

-- Theorem to prove
theorem alok_present_age :
  ∃ t, alok_age t = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alok_present_age_l973_97361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_YU_l973_97365

-- Define the rectangle ABCD
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the pyramid Q
structure Pyramid where
  base : Rectangle
  height : ℝ

-- Define the frustum G and smaller pyramid Q'
structure Frustum where
  base : Rectangle
  top : Rectangle
  height : ℝ

-- Define point Y (center of circumsphere of G) and point U (apex of Q)
def Point := ℝ × ℝ × ℝ

def dist (p1 p2 : Point) : ℝ := sorry

noncomputable def rectangle_ABCD : Rectangle := ⟨10, 15⟩

noncomputable def pyramid_Q : Pyramid := ⟨rectangle_ABCD, 30⟩

noncomputable def frustum_G : Frustum := sorry

noncomputable def pyramid_Q' : Pyramid := sorry

noncomputable def point_Y : Point := sorry

noncomputable def point_U : Point := sorry

-- State the theorem
theorem length_YU (h1 : pyramid_Q.base.length = 10)
                  (h2 : pyramid_Q.base.width = 15)
                  (h3 : pyramid_Q.height = 30)
                  (h4 : frustum_G.base = pyramid_Q.base)
                  (h5 : ∃ k, pyramid_Q'.base.length = k * pyramid_Q.base.length ∧ 
                             pyramid_Q'.base.width = k * pyramid_Q.base.width)
                  (h6 : point_U = (0, 0, pyramid_Q.height))
                  (h7 : (pyramid_Q.base.length * pyramid_Q.base.width * pyramid_Q.height) = 
                        9 * (pyramid_Q'.base.length * pyramid_Q'.base.width * pyramid_Q'.height)) :
  dist point_Y point_U = 325 / 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_YU_l973_97365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_needle_endpoints_not_interchangeable_l973_97326

-- Define the complex number ω = e^(πi/4)
noncomputable def ω : ℂ := Complex.exp (Complex.I * Real.pi / 4)

-- Define the sequence of rotations
noncomputable def z : ℕ → ℂ
| 0 => 0
| 1 => 1
| (n + 2) => (1 - ω^(e n)) * z (n + 1) + ω^(e n) * z n
where
  e : ℕ → ℤ := fun _ => 1  -- Arbitrary sequence of integers, simplified for this example

-- Define the number of terms in the expansion of z_n
def A : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => 2 * A (n + 1) + A n

theorem needle_endpoints_not_interchangeable : ∀ k : ℕ, Odd k → z k ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_needle_endpoints_not_interchangeable_l973_97326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_theorem_l973_97352

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ

/-- Checks if a point lies on a given parabola -/
def isOnParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola -/
noncomputable def focus (parabola : Parabola) : Point :=
  { x := parabola.p / 2, y := 0 }

/-- Vector addition -/
def vectorAdd (a b : Point) : Point :=
  { x := a.x + b.x, y := a.y + b.y }

theorem parabola_line_theorem (parabola : Parabola) (A B C : Point) (h1 : isOnParabola A parabola)
    (h2 : isOnParabola B parabola) (h3 : isOnParabola C parabola) (h4 : A.x = 1 ∧ A.y = 2)
    (h5 : vectorAdd (vectorAdd B (Point.mk (-A.x) (-A.y))) (vectorAdd C (Point.mk (-A.x) (-A.y))) =
          vectorAdd (focus parabola) (Point.mk (-A.x) (-A.y))) :
    ∃ (l : Line), l.a = 2 ∧ l.b = -1 ∧ l.c = -1 ∧
    (∀ (p : Point), isOnParabola p parabola → (l.a * p.x + l.b * p.y + l.c = 0 ↔ (p = B ∨ p = C))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_theorem_l973_97352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l973_97334

-- Define a structure for our triangle
structure Triangle where
  t : ℝ  -- area
  r : ℝ  -- radius of inscribed circle
  α : ℝ  -- one angle (in radians)
  s : ℝ  -- semiperimeter
  a : ℝ  -- one side
  bc : ℝ  -- sum of other two sides
  βγ : ℝ  -- sum of other two angles

-- State the theorem
theorem triangle_properties (tri : Triangle) (h_pos_t : tri.t > 0) (h_pos_r : tri.r > 0) 
    (h_angle : 0 < tri.α ∧ tri.α < π) : 
  tri.s = tri.t / tri.r ∧ 
  tri.a = tri.t / tri.r - tri.r * (1 / Real.tan (tri.α / 2)) ∧
  tri.bc = tri.t / tri.r + tri.r * (1 / Real.tan (tri.α / 2)) ∧
  tri.βγ = π - tri.α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l973_97334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_bound_l973_97380

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A chord of a circle --/
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ
  on_circle1 : (endpoint1.1 - c.center.1)^2 + (endpoint1.2 - c.center.2)^2 = c.radius^2
  on_circle2 : (endpoint2.1 - c.center.1)^2 + (endpoint2.2 - c.center.2)^2 = c.radius^2

/-- The length of a chord --/
noncomputable def chord_length (c : Circle) (ch : Chord c) : ℝ :=
  Real.sqrt ((ch.endpoint1.1 - ch.endpoint2.1)^2 + (ch.endpoint1.2 - ch.endpoint2.2)^2)

/-- A chord is a diameter if it passes through the center of the circle --/
def is_diameter (c : Circle) (ch : Chord c) : Prop :=
  (ch.endpoint1.1 - c.center.1) * (ch.endpoint2.1 - c.center.1) +
  (ch.endpoint1.2 - c.center.2) * (ch.endpoint2.2 - c.center.2) = -c.radius^2

theorem chord_length_bound (c : Circle) (ch : Chord c) :
  chord_length c ch ≤ 2 * c.radius ∧
  (chord_length c ch = 2 * c.radius ↔ is_diameter c ch) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_bound_l973_97380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_are_distinct_l973_97395

/-- A polynomial of degree n. -/
structure PolynomialN (R : Type*) [Ring R] (n : ℕ) where
  coeff : Fin (n + 1) → R
  leading_coeff_nonzero : coeff ⟨n, by linarith⟩ ≠ 0

/-- The second derivative of a polynomial. -/
def secondDerivative {R : Type*} [Ring R] {n : ℕ} (p : PolynomialN R n) : PolynomialN R (n - 2) :=
  sorry

/-- The roots of a polynomial. -/
def roots {R : Type*} [Field R] {n : ℕ} (p : PolynomialN R n) : Set R :=
  sorry

/-- The property that all roots are distinct. -/
def hasDistinctRoots {R : Type*} [Field R] {n : ℕ} (p : PolynomialN R n) : Prop :=
  ∀ x y, x ∈ roots p → y ∈ roots p → x = y → x = y

/-- The theorem statement. -/
theorem roots_are_distinct
  {R : Type*} [Field R] {n : ℕ} (p : PolynomialN R n) (q : PolynomialN R 2)
  (h1 : ∀ x, p.coeff x = (secondDerivative p).coeff x * q.coeff x)
  (h2 : ¬ ∀ x y, x ∈ roots p → y ∈ roots p → x = y) :
  hasDistinctRoots p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_are_distinct_l973_97395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_cost_calculation_l973_97357

/-- The cost in dollars to transport 1 kg of equipment to the International Space Station. -/
noncomputable def cost_per_kg : ℝ := 18000

/-- The weight of the scientific instrument in grams. -/
noncomputable def instrument_weight_g : ℝ := 400

/-- The weight of the scientific instrument in kilograms. -/
noncomputable def instrument_weight_kg : ℝ := instrument_weight_g / 1000

/-- The total cost in dollars to transport the scientific instrument. -/
noncomputable def total_cost : ℝ := instrument_weight_kg * cost_per_kg

theorem transportation_cost_calculation :
  total_cost = 7200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_cost_calculation_l973_97357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l973_97374

theorem power_inequality (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : a < Real.exp 1) : a ^ b > b ^ a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l973_97374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l973_97399

noncomputable def f (x : Real) : Real := 
  (Real.sin x + Real.cos x)^2 + Real.cos (2*x) - 1

theorem f_properties :
  ∃ (T : Real), T > 0 ∧ 
  (∀ (x : Real), f (x + T) = f x) ∧
  (∀ (S : Real), S > 0 ∧ (∀ (x : Real), f (x + S) = f x) → S ≥ T) ∧
  T = Real.pi ∧
  (∀ (x : Real), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≤ Real.sqrt 2) ∧
  (∀ (x : Real), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≥ -Real.sqrt 2) ∧
  (∃ (x₁ x₂ : Real), x₁ ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ 
                   x₂ ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ 
                   f x₁ = Real.sqrt 2 ∧ 
                   f x₂ = -Real.sqrt 2) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l973_97399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_eating_out_and_socializing_l973_97302

noncomputable def net_salary : ℝ := 3400
noncomputable def discretionary_income : ℝ := net_salary / 5
noncomputable def vacation_fund : ℝ := 0.3 * discretionary_income
noncomputable def savings : ℝ := 0.2 * discretionary_income
noncomputable def gifts_and_charity : ℝ := 102

noncomputable def eating_out_and_socializing : ℝ := 
  discretionary_income - (vacation_fund + savings + gifts_and_charity)

theorem percentage_eating_out_and_socializing : 
  (eating_out_and_socializing / discretionary_income) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_eating_out_and_socializing_l973_97302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_grass_cutting_height_l973_97369

/-- Represents the grass cutting scenario -/
structure GrassCutting where
  initial_height : ℝ
  growth_rate : ℝ
  cost_per_cut : ℝ
  annual_cost : ℝ

/-- Calculates the height at which the grass is cut -/
noncomputable def cutting_height (gc : GrassCutting) : ℝ :=
  gc.initial_height + (gc.annual_cost / gc.cost_per_cut) * (gc.growth_rate * (12 / (gc.annual_cost / gc.cost_per_cut)))

/-- Theorem stating that for the given conditions, the cutting height is 4 inches -/
theorem john_grass_cutting_height :
  let gc : GrassCutting := {
    initial_height := 2
    growth_rate := 0.5
    cost_per_cut := 100
    annual_cost := 300
  }
  cutting_height gc = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_grass_cutting_height_l973_97369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l973_97317

/-- The probability of drawing the same color for the first and last card from a deck of 5 blue and 5 red cards -/
theorem same_color_probability (blue_cards red_cards : ℕ) (h1 : blue_cards = 5) (h2 : red_cards = 5) :
  let total_cards := blue_cards + red_cards
  let prob_first_blue := blue_cards / total_cards
  let prob_first_red := red_cards / total_cards
  let prob_last_blue_given_first_blue := (blue_cards - 1) / (total_cards - 1)
  let prob_last_red_given_first_red := (red_cards - 1) / (total_cards - 1)
  prob_first_blue * prob_last_blue_given_first_blue + 
  prob_first_red * prob_last_red_given_first_red = 4 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l973_97317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_intersection_x_is_3_l973_97393

/-- The polynomial P(x) -/
def P (x d : ℝ) : ℝ := x^7 - x^6 - 21*x^5 + 35*x^4 + 84*x^3 - 49*x^2 - 70*x + d

/-- The line y = 2x - 3 -/
def L (x : ℝ) : ℝ := 2*x - 3

/-- The theorem stating that the largest x-value of intersection is 3 -/
theorem largest_intersection_x_is_3 (d : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    P x₁ d = L x₁ ∧ P x₂ d = L x₂ ∧ P x₃ d = L x₃ ∧
    ∀ (x : ℝ), P x d = L x → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (∃ (x : ℝ), P x d = L x ∧ ∀ (y : ℝ), P y d = L y → y ≤ x) →
  (∃ (x : ℝ), P x d = L x ∧ ∀ (y : ℝ), P y d = L y → y ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_intersection_x_is_3_l973_97393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_origin_same_side_of_line_l_l973_97354

/-- The line l: x + y - 1 = 0 -/
def line_l (x y : ℝ) : ℝ := x + y - 1

/-- Point A has coordinates (1, a) -/
def point_A (a : ℝ) : ℝ × ℝ := (1, a)

/-- The origin (0, 0) -/
def origin : ℝ × ℝ := (0, 0)

/-- Two points are on the same side of a line if the product of their signed distances is positive -/
def same_side (p q : ℝ × ℝ) : Prop :=
  (line_l p.1 p.2) * (line_l q.1 q.2) > 0

/-- If point A (1, a) and the origin are on the same side of line l: x+y-1=0, then a < 0 -/
theorem point_A_origin_same_side_of_line_l (a : ℝ) :
  same_side (point_A a) origin → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_origin_same_side_of_line_l_l973_97354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_crossing_lighthouse_l973_97345

/-- The time taken for a ship to cross a lighthouse -/
noncomputable def ship_crossing_time (speed_kmh : ℝ) (length_m : ℝ) : ℝ :=
  length_m / (speed_kmh * 1000 / 3600)

/-- Theorem: A ship with speed 18 km/hr and length 100.008 meters takes 20.0016 seconds to cross a lighthouse -/
theorem ship_crossing_lighthouse :
  let speed_kmh : ℝ := 18
  let length_m : ℝ := 100.008
  ship_crossing_time speed_kmh length_m = 20.0016 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_crossing_lighthouse_l973_97345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_l973_97370

def sequence_y (n m : ℕ) : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | k+2 => ((n+m-1) * sequence_y n m (k+1) - (n-(k:ℕ)) * sequence_y n m k + m) / (k+1:ℚ)

theorem sum_of_sequence (n m : ℕ) (h : n > m) :
  ∑' k, sequence_y n m k = 2^(n+m-1) * (n + m - 1/2) := by
  sorry

#check sum_of_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_l973_97370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_integer_in_list_l973_97381

def given_numbers : List ℚ := [0, -1, 2, -3/2]

theorem negative_integer_in_list : ∃ (x : ℤ), (x : ℚ) ∈ given_numbers ∧ x < 0 ∧ 
  (∀ (y : ℤ), (y : ℚ) ∈ given_numbers ∧ y < 0 → y = x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_integer_in_list_l973_97381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_l973_97391

theorem remainder_sum (x y : ℕ) 
  (hx : x % 246 = 37)
  (hy : y % 357 = 53) :
  (x + y + 97) % 123 = 52 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_l973_97391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l973_97313

theorem sin_cos_relation (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l973_97313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_problem_l973_97307

-- Define the triangle DOG
structure Triangle (D O G : Type) where
  dog : D → O → G → Prop

-- Define angle measure
noncomputable def angle_measure (A B C : Type) : ℝ := sorry

-- Define angle bisector
def is_angle_bisector (O S D G : Type) : Prop :=
  angle_measure O S D = angle_measure O S G

-- State the theorem
theorem triangle_angle_problem 
  (D O G S : Type) 
  (dog : Triangle D O G) 
  (h1 : angle_measure D G O = angle_measure D O G)
  (h2 : angle_measure G O D = 45)
  (h3 : is_angle_bisector O S D G) :
  angle_measure D S O = 78.75 := by
  sorry

#check triangle_angle_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_problem_l973_97307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_value_l973_97398

-- Define the polynomial division operation
noncomputable def poly_div (p q : ℝ → ℝ) : (ℝ → ℝ) × ℝ :=
  sorry

-- Define x^6
noncomputable def p (x : ℝ) : ℝ := x^6

-- Define x - 1/3
noncomputable def d (x : ℝ) : ℝ := x - 1/3

-- First division
noncomputable def first_div : (ℝ → ℝ) × ℝ := poly_div p d

-- Extract q_1 and r_1 from first division
noncomputable def q_1 : ℝ → ℝ := (first_div.1)
noncomputable def r_1 : ℝ := (first_div.2)

-- Second division
noncomputable def second_div : (ℝ → ℝ) × ℝ := poly_div q_1 d

-- Extract q_2 and r_2 from second division
noncomputable def q_2 : ℝ → ℝ := (second_div.1)
noncomputable def r_2 : ℝ := (second_div.2)

-- Theorem statement
theorem remainder_value : r_2 = 2/81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_value_l973_97398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_divisors_equal_implies_equal_l973_97377

def product_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem product_of_divisors_equal_implies_equal (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  product_of_divisors a = product_of_divisors b → a = b := by
  sorry

#check product_of_divisors_equal_implies_equal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_divisors_equal_implies_equal_l973_97377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l973_97332

/-- Represents a pyramid with an equilateral triangle base --/
structure EquilateralPyramid where
  -- Side length of the base triangle
  side_length : ℝ
  -- Angle between edge and line to vertex
  angle : ℝ

/-- Volume of an equilateral pyramid --/
noncomputable def volume (p : EquilateralPyramid) : ℝ :=
  (2 * Real.sqrt 3 * Real.tan p.angle) / 3

/-- Theorem stating the volume of the specific pyramid --/
theorem volume_of_specific_pyramid :
  ∀ (p : EquilateralPyramid),
  p.side_length = 2 →
  volume p = (2 * Real.sqrt 3 * Real.tan p.angle) / 3 :=
by
  intro p h
  -- Unfold the definition of volume
  unfold volume
  -- The equality holds by definition
  rfl

#check volume_of_specific_pyramid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l973_97332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_Z_l973_97335

-- Define the complex number z
variable (z : ℂ)

-- Define the set of points Z satisfying the condition |z+i| ≤ 1
def Z : Set ℂ := {z : ℂ | Complex.abs (z + Complex.I) ≤ 1}

-- State the theorem
theorem area_of_Z : MeasureTheory.volume Z = π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_Z_l973_97335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_of_2_equals_3_l973_97321

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(x-1) - 2

-- State the theorem
theorem inverse_f_of_2_equals_3 :
  (∀ x, f (x + 1) = 2^x - 2) → f⁻¹ 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_of_2_equals_3_l973_97321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_deposit_total_amount_l973_97328

/-- Calculate the total amount of principal and interest for a fixed deposit -/
def calculateTotalAmount (initialDeposit : ℝ) (annualInterestRate : ℝ) (monthlySubsidyRate : ℝ) 
  (monthlyInterestRateAfterMaturity : ℝ) (depositYears : ℝ) (extraMonths : ℝ) : ℝ :=
  let amountAtMaturity := initialDeposit * (1 + annualInterestRate * depositYears + 
    monthlySubsidyRate * 12 * depositYears)
  amountAtMaturity * (1 + monthlyInterestRateAfterMaturity * extraMonths)

/-- Theorem stating the correct total amount for the given conditions -/
theorem fixed_deposit_total_amount :
  calculateTotalAmount 1000 0.14 0.07 0.02 3 6 = 4390.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_deposit_total_amount_l973_97328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_billing_l973_97300

-- Define the electricity billing function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 200 then 0.6 * x else 0.65 * x - 10

-- State the theorem
theorem electricity_billing :
  (∀ x : ℝ, x ≥ 0 → x ≤ 200 → f x = 0.6 * x) ∧
  (∀ x : ℝ, x > 200 → f x = 0.65 * x - 10) ∧
  (f 160 = 96) ∧
  (∃ x : ℝ, f x = 146 ∧ x = 240) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_billing_l973_97300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_when_area_is_specific_l973_97343

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2/3 + y^2/2 = 1

-- Define a line passing through (1,0)
def line_through_F (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the area of triangle ΔABO
noncomputable def triangle_area (k : ℝ) : ℝ := 
  2 * Real.sqrt 3 * Real.sqrt ((k^2 * (1 + k^2)) / ((2 + 3*k^2)^2))

-- Theorem statement
theorem line_equation_when_area_is_specific :
  ∀ k : ℝ, triangle_area k = 2 * Real.sqrt 6 / 5 → k = 1 ∨ k = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_when_area_is_specific_l973_97343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solutions_l973_97387

theorem no_integer_solutions :
  ¬∃ (x y : ℤ), (2 : ℝ)^(x + 2) - (3 : ℝ)^(y + 1) = 41 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solutions_l973_97387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l973_97322

theorem sin_minus_cos_value (x : ℝ) (h1 : -π < x) (h2 : x < 0) (h3 : Real.sin x + Real.cos x = 1/5) :
  Real.sin x - Real.cos x = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l973_97322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l973_97346

theorem triangle_shape (A B C : Real) (a b c : Real) : 
  (A > 0) → (B > 0) → (C > 0) → 
  (a > 0) → (b > 0) → (c > 0) → 
  (A + B + C = Real.pi) → 
  (Real.cos A / Real.cos B = c / a) → 
  ((A = B) ∨ (A + B = Real.pi / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l973_97346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l973_97389

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / (4 * h.a^2) - p.y^2 / (2 * h.b^2) = 1

/-- Defines the foci of the hyperbola -/
noncomputable def foci (h : Hyperbola) : (Point × Point) :=
  let c := Real.sqrt (h.a^2 + h.b^2)
  (Point.mk (-c) 0, Point.mk c 0)

/-- Defines a point on the asymptote of the hyperbola -/
def on_asymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y = (h.b / h.a) * p.x

/-- Defines perpendicularity of two vectors -/
def perpendicular (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

/-- Calculates the area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  let a := Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)
  let b := Real.sqrt ((p3.x - p2.x)^2 + (p3.y - p2.y)^2)
  let c := Real.sqrt ((p1.x - p3.x)^2 + (p1.y - p3.y)^2)
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem hyperbola_triangle_area
  (h : Hyperbola)
  (p : Point)
  (h_eq : hyperbola_equation h p)
  (h_asym : on_asymptote h p)
  (h_perp : let (f1, f2) := foci h; perpendicular p f1 f2) :
  triangle_area p (foci h).1 (foci h).2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l973_97389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_rate_l973_97311

/-- Represents the growth factor of an investment over a given time period. -/
noncomputable def growthFactor (principal amount : ℝ) : ℝ := amount / principal

/-- Calculates the number of times an investment triples over a given time period. -/
noncomputable def triplingPeriods (years : ℝ) (x : ℝ) : ℝ := years / (112 / x)

/-- Theorem stating that if money triples every 112/x years at x% interest,
    and $3500 grows to $31500 in 28 years at 8% interest, then x = 8. -/
theorem investment_growth_rate (x : ℝ) :
  (∀ (principal amount years : ℝ), 
    growthFactor principal amount = 3 ^ (triplingPeriods years x)) →
  (growthFactor 3500 31500 = 3 ^ (triplingPeriods 28 8)) →
  x = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_rate_l973_97311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_valid_floors_l973_97348

/-- Represents a rectangular floor with dimensions a and b -/
structure Floor :=
  (a : ℕ)
  (b : ℕ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (b_gt_a : b > a)

/-- Checks if a floor satisfies the painting conditions -/
def satisfiesPaintingConditions (f : Floor) : Prop :=
  let unpaintedArea := f.a * f.b - (f.a - 4) * (f.b - 4)
  3 * unpaintedArea = f.a * f.b

/-- The main theorem stating that there are exactly 4 valid floor configurations -/
theorem exactly_four_valid_floors :
  ∃! (s : Finset Floor), s.card = 4 ∧ 
    (∀ f ∈ s, satisfiesPaintingConditions f) ∧
    (∀ f : Floor, satisfiesPaintingConditions f → f ∈ s) :=
sorry

#check exactly_four_valid_floors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_valid_floors_l973_97348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_stats_from_transformed_l973_97388

-- Define the type for our data set
def DataSet := List ℝ

-- Function to transform the data
def transform (data : DataSet) : DataSet :=
  data.map (λ x => 2 * x - 3)

-- Function to calculate the average of a data set
noncomputable def average (data : DataSet) : ℝ :=
  data.sum / data.length

-- Function to calculate the variance of a data set
noncomputable def variance (data : DataSet) : ℝ :=
  let avg := average data
  (data.map (λ x => (x - avg) ^ 2)).sum / data.length

theorem original_stats_from_transformed (data : DataSet) :
  (average (transform data) = 7) →
  (variance (transform data) = 4) →
  (average data = 5 ∧ variance data = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_stats_from_transformed_l973_97388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_edges_perpendicular_implies_right_equal_body_diagonals_implies_right_l973_97303

-- Define a quadrangular prism
structure QuadrangularPrism where
  -- Add necessary fields here (placeholder)
  base_width : ℝ
  base_length : ℝ
  height : ℝ

-- Define what it means for a quadrangular prism to be right
def is_right_quadrangular_prism (p : QuadrangularPrism) : Prop := 
  -- Definition of a right quadrangular prism (placeholder)
  p.base_width > 0 ∧ p.base_length > 0 ∧ p.height > 0

-- Define a section passing through opposite lateral edges
def opposite_lateral_edge_section (p : QuadrangularPrism) : Type := 
  -- Definition of a section passing through opposite lateral edges (placeholder)
  ℝ × ℝ

-- Define what it means for a section to be perpendicular to the base
def is_perpendicular_to_base (p : QuadrangularPrism) (s : opposite_lateral_edge_section p) : Prop :=
  -- Definition of perpendicularity to base (placeholder)
  true

-- Define body diagonal of a prism
def body_diagonal (p : QuadrangularPrism) : Type :=
  -- Definition of a body diagonal (placeholder)
  ℝ

-- Define equality of body diagonals
def body_diagonals_equal (p : QuadrangularPrism) : Prop :=
  -- Definition of all four body diagonals being pairwise equal (placeholder)
  true

-- Theorem for proposition 2
theorem opposite_edges_perpendicular_implies_right (p : QuadrangularPrism) :
  (∀ s : opposite_lateral_edge_section p, is_perpendicular_to_base p s) →
  is_right_quadrangular_prism p := by
  sorry

-- Theorem for proposition 4
theorem equal_body_diagonals_implies_right (p : QuadrangularPrism) :
  body_diagonals_equal p →
  is_right_quadrangular_prism p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_edges_perpendicular_implies_right_equal_body_diagonals_implies_right_l973_97303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_after_transform_l973_97305

def transform (x : ℝ) : ℝ := 2 * x + 1

-- Define a type for variance function
def VarianceFunction := (Finset ℝ) → ℝ

theorem variance_after_transform (data : Finset ℝ) (variance : VarianceFunction) :
  variance data = 9 →
  variance (data.image transform) = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_after_transform_l973_97305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_proof_l973_97312

theorem right_triangle_proof (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧ 
  A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C ∧
  2 * a * Real.sin C * Real.cos A = c * Real.sin (2 * B) ∧
  A ≠ B →
  A + B = Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_proof_l973_97312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_divisible_by_5_l973_97341

def isDivisibleBy5 (n : ℕ) : Bool := n % 5 = 0

def numbersInRange : List ℕ := (List.range 29).map (· + 6) |>.filter isDivisibleBy5

theorem average_of_numbers_divisible_by_5 : 
  (List.sum numbersInRange) / (List.length numbersInRange) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_divisible_by_5_l973_97341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_quadrilateral_area_l973_97358

-- Define the ellipse C1
noncomputable def C1 (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

-- Define the parabola C2
noncomputable def C2 (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the eccentricity of C1
def eccentricity_C1 : ℚ := 1 / 2

-- Define the y-intercept of C1
noncomputable def y_intercept_C1 : ℝ := 2 * Real.sqrt 3

-- Define the vertex of C2
def vertex_C2 : ℝ × ℝ := (0, 0)

-- Define the focus of C2 (which coincides with the left focus of C1)
def focus_C2 : ℝ × ℝ := (-2, 0)

-- Define a function to calculate the area of the quadrilateral
noncomputable def quadrilateral_area (k : ℝ) : ℝ :=
  16 * Real.sqrt ((k^2 + 2 + 1/k^2) * (2*k^2 + 5 + 2/k^2))

-- State the theorem
theorem min_quadrilateral_area :
  ∃ (min_area : ℝ), min_area = 96 ∧
  ∀ (k : ℝ), k ≠ 0 → quadrilateral_area k ≥ min_area :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_quadrilateral_area_l973_97358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snack_cost_proof_l973_97331

/-- The cost of a snack given the total cost and price difference with juice -/
noncomputable def cost_of_snack (total_cost : ℝ) (price_difference : ℝ) : ℝ :=
  (total_cost - price_difference) / 2

theorem snack_cost_proof (total_cost : ℝ) (price_difference : ℝ) 
  (h1 : total_cost = 4.60)
  (h2 : price_difference = 3) :
  cost_of_snack total_cost price_difference = 0.80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snack_cost_proof_l973_97331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_is_24_over_5_l973_97396

/-- A right triangle with a semicircle inscribed -/
structure RightTriangleWithSemicircle where
  /-- Length of side DE -/
  de : ℝ
  /-- Length of side EF -/
  ef : ℝ
  /-- Angle E is a right angle -/
  angle_e_right : True
  /-- DE is positive -/
  de_pos : de > 0
  /-- EF is positive -/
  ef_pos : ef > 0

/-- The radius of the inscribed semicircle in the right triangle -/
noncomputable def semicircle_radius (t : RightTriangleWithSemicircle) : ℝ :=
  (t.de * t.ef) / (t.de + t.ef + Real.sqrt (t.de^2 + t.ef^2))

/-- Theorem: The radius of the inscribed semicircle in the specified right triangle is 24/5 -/
theorem semicircle_radius_is_24_over_5 (t : RightTriangleWithSemicircle) 
    (h1 : t.de = 15) (h2 : t.ef = 8) : semicircle_radius t = 24 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_is_24_over_5_l973_97396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_philosophy_worldview_relationship_l973_97392

-- Define the basic concepts
def Person : Type := Unit
def Worldview : Type := Unit
def Philosophy : Type := Unit

-- Define properties
def has_worldview (p : Person) : Prop := sorry
def is_philosopher (p : Person) : Prop := sorry
def is_systematized_worldview (phil : Philosophy) : Prop := sorry
def oppose_simplification (phil : Philosophy) : Prop := sorry

-- State the theorem
theorem philosophy_worldview_relationship :
  (∀ p : Person, has_worldview p) →
  (∃ p : Person, ¬is_philosopher p) →
  (∀ phil : Philosophy, is_systematized_worldview phil ∧ oppose_simplification phil) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_philosophy_worldview_relationship_l973_97392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression1_value_expression2_value_l973_97325

-- Define the first expression
noncomputable def expression1 : ℝ := (64 / 125) ^ (1/3)

-- Define the second expression
noncomputable def expression2 : ℝ := Real.sqrt 16

-- Theorem for the first expression
theorem expression1_value : expression1 = 4/5 := by
  sorry

-- Theorem for the second expression
theorem expression2_value : expression2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression1_value_expression2_value_l973_97325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fountain_area_l973_97324

/-- The area of a circular fountain with the given conditions -/
theorem fountain_area : 
  ∀ (A B C D : ℝ × ℝ) (r : ℝ),
  (‖A - B‖ = 16) →  -- Length of plank AB is 16
  (‖C - D‖ = 10) →  -- Length of plank DC is 10
  (D = (A + B) / 2) →  -- D is midpoint of AB
  (‖A - C‖ = r ∧ ‖B - C‖ = r) →  -- A and B are on the circumference
  (π * r^2 = 164 * π) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fountain_area_l973_97324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_range_l973_97384

/-- The inverse proportion function y = 3/x -/
noncomputable def inverse_proportion (x : ℝ) : ℝ := 3 / x

theorem inverse_proportion_range (m y₁ y₂ : ℝ) :
  (inverse_proportion (m - 1) = y₁) →
  (inverse_proportion (m + 1) = y₂) →
  (y₁ > y₂) →
  (m < -1 ∨ m > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_range_l973_97384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_relation_l973_97355

theorem log_relation (a b : ℝ) (h1 : a = Real.log 225 / Real.log 8) (h2 : b = Real.log 15 / Real.log 2) :
  a = (2 / 3) * b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_relation_l973_97355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_when_a_zero_f_negative_when_a_half_and_x_greater_one_l973_97368

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * (x - 1)^2 - x + 1

-- Part 1
theorem f_properties_when_a_zero :
  ∀ x y : ℝ, 
  (0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x < y → f 0 x > f 0 y) ∧
  (1 < x ∧ 1 < y ∧ x < y → f 0 x < f 0 y) ∧
  (∀ z : ℝ, z > 0 → f 0 z ≥ f 0 1) ∧
  (f 0 1 = 0) := by
  sorry

-- Part 2
theorem f_negative_when_a_half_and_x_greater_one :
  ∀ a x : ℝ, a ≥ 1/2 ∧ x > 1 → f a x < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_when_a_zero_f_negative_when_a_half_and_x_greater_one_l973_97368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_root_angle_l973_97333

theorem repeated_root_angle (θ : Real) : 
  0 < θ ∧ θ < π / 2 → -- θ is acute
  (∃ x : Real, (x^2 + 4*x*Real.cos θ + Real.tan (π/2 - θ) = 0) ∧ 
    (∀ y : Real, y^2 + 4*y*Real.cos θ + Real.tan (π/2 - θ) = 0 → y = x)) → -- equation has a repeated root
  θ = π / 12 ∨ θ = 5*π / 12 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_root_angle_l973_97333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_point_l973_97356

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x - 6)

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := Real.exp x * (x - 5)

theorem tangent_line_passes_through_point :
  let x₀ : ℝ := 4
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∃ (b : ℝ), y₀ + m * (2 - x₀) = 0 :=
by
  sorry

#check tangent_line_passes_through_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_point_l973_97356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_right_triangle_l973_97375

/-- A cone is a geometric solid with a circular base and a single vertex, where all points on the base are connected to the vertex by straight lines. -/
structure Cone where
  base : Sphere
  vertex : EuclideanSpace ℝ (Fin 3)
  slant_height : Real

/-- A point on the circumference of the cone's base -/
def point_on_base_circumference (c : Cone) : EuclideanSpace ℝ (Fin 3) :=
  sorry

/-- The center of the cone's base circle -/
def base_center (c : Cone) : EuclideanSpace ℝ (Fin 3) :=
  sorry

/-- The triangle formed by the vertex, a point on the base circumference, and the base center -/
def triangle_in_cone (c : Cone) : Set (EuclideanSpace ℝ (Fin 3)) :=
  {c.vertex, point_on_base_circumference c, base_center c}

/-- Predicate to check if a triangle is right-angled -/
def IsRightTriangle (t : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  sorry

/-- Theorem: The triangle formed by the vertex of a cone, any point on the circumference of the cone's base, and the center of the base circle is a right triangle -/
theorem cone_right_triangle (c : Cone) : 
  IsRightTriangle (triangle_in_cone c) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_right_triangle_l973_97375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l973_97363

noncomputable section

-- Define the original ellipse
def original_ellipse (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define the transformation
def transform (x y : ℝ) : ℝ × ℝ := (x, y/2)

-- Define curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the parametric equation of C
def parametric_C (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 2

-- Define the tangent line to C at a point
def tangent_C (x y dx dy : ℝ) : Prop := 2*x*dx + 2*y*dy = 0

theorem curve_C_properties :
  -- 1. Prove that the equation of curve C is x^2 + y^2 = 1
  (∀ x y : ℝ, original_ellipse x y → curve_C (transform x y).1 (transform x y).2) ∧
  -- 2. Prove that the parametric equation of C is correct
  (∀ α : ℝ, curve_C (parametric_C α).1 (parametric_C α).2) ∧
  -- 3. Prove that the points where tangent is perpendicular to l have the given coordinates
  (∀ x y : ℝ, curve_C x y →
    (∃ dx dy : ℝ, tangent_C x y dx dy ∧ dx * 1 + dy * (-1) = 0) →
    ((x = Real.sqrt 2 / 2 ∧ y = Real.sqrt 2 / 2) ∨
     (x = -Real.sqrt 2 / 2 ∧ y = -Real.sqrt 2 / 2))) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l973_97363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l973_97327

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / 2 = 1

-- Define the asymptote
def asymptote (a : ℝ) (x y : ℝ) : Prop := y = (Real.sqrt 2 / a) * x

-- Define the focus
noncomputable def focus (a : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + 2), 0)

-- Define the distance from a point to a line
noncomputable def distancePointToLine (p : ℝ × ℝ) (m : ℝ) : ℝ := 
  let (x, y) := p
  abs (y - m * x) / Real.sqrt (1 + m^2)

theorem hyperbola_focus_distance (a : ℝ) :
  hyperbola a (Real.sqrt 2) 1 →
  asymptote a (Real.sqrt 2) 1 →
  distancePointToLine (focus a) (Real.sqrt 2 / a) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l973_97327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_between_lines_l973_97394

/-- The cosine of the acute angle between two 2D vectors -/
noncomputable def cosine_angle (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2))

/-- The first line's direction vector -/
def v1 : ℝ × ℝ := (4, 5)

/-- The second line's direction vector -/
def v2 : ℝ × ℝ := (2, -1)

theorem cos_angle_between_lines : cosine_angle v1 v2 = 3 / Real.sqrt 205 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_between_lines_l973_97394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_sum_of_squares_l973_97372

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := |a * Real.sin x + b * Real.cos x - 1| + |b * Real.sin x - a * Real.cos x|

-- State the theorem
theorem max_value_implies_sum_of_squares (a b : ℝ) : 
  (∀ x : ℝ, f a b x ≤ 11) ∧ (∃ x : ℝ, f a b x = 11) → a^2 + b^2 = 50 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_sum_of_squares_l973_97372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rows_in_table_l973_97367

theorem max_rows_in_table (n : ℕ) (hn : n > 0) :
  ∃ (m : ℕ), m = 2^n ∧
  (∃ (t : Fin m → Fin n → ℝ),
    ∀ (i j : Fin m), i ≠ j →
      (∃ (k : Fin n), |t i k - t j k| = 1) ∧
      (∀ (k : Fin n), |t i k - t j k| ≤ 1)) ∧
  (∀ (m' : ℕ), m' > m →
    ¬∃ (t : Fin m' → Fin n → ℝ),
      ∀ (i j : Fin m'), i ≠ j →
        (∃ (k : Fin n), |t i k - t j k| = 1) ∧
        (∀ (k : Fin n), |t i k - t j k| ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rows_in_table_l973_97367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_implies_diameter_l973_97330

/-- The area of the circle in square meters -/
def circle_area : ℝ := 132.73228961416876

/-- The diameter of the circle in meters -/
def circle_diameter : ℝ := 12.998044

/-- Theorem stating that the given area corresponds to the given diameter -/
theorem area_implies_diameter :
  ∃ (d : ℝ), d > 0 ∧ circle_area = Real.pi * (d / 2)^2 ∧ 
  |d - circle_diameter| < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_implies_diameter_l973_97330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l973_97310

theorem point_in_fourth_quadrant (θ : ℝ) 
  (h1 : Real.sin (θ / 2) = 3 / 5) 
  (h2 : Real.cos (θ / 2) = -4 / 5) : 
  Real.cos θ > 0 ∧ Real.sin θ < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l973_97310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_whole_number_to_shaded_area_l973_97329

-- Define the rectangle dimensions
def rectangle_width : ℚ := 5
def rectangle_height : ℚ := 8

-- Define the circle diameter
def circle_diameter : ℚ := 2

-- Define the area of the rectangle
def rectangle_area : ℚ := rectangle_width * rectangle_height

-- Define the radius of the circle
def circle_radius : ℚ := circle_diameter / 2

-- Define the area of the circle (noncomputable due to π)
noncomputable def circle_area : ℝ := Real.pi * (circle_radius : ℝ) ^ 2

-- Define the area of the shaded region (noncomputable due to circle_area)
noncomputable def shaded_area : ℝ := (rectangle_area : ℝ) - circle_area

-- Theorem to prove
theorem closest_whole_number_to_shaded_area : 
  ∃ (n : ℤ), n = 37 ∧ ∀ (m : ℤ), |shaded_area - (n : ℝ)| ≤ |shaded_area - (m : ℝ)| := by
  sorry

#eval rectangle_area
#eval circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_whole_number_to_shaded_area_l973_97329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_final_push_time_l973_97337

/-- The time it takes for John to catch up and overtake Steve in a race -/
noncomputable def race_time (john_speed steve_speed initial_distance final_distance : ℝ) : ℝ :=
  (initial_distance + final_distance) / (john_speed - steve_speed)

/-- Theorem: Given the conditions of the race, John's final push lasts 42.5 seconds -/
theorem johns_final_push_time :
  let john_speed : ℝ := 4.2
  let steve_speed : ℝ := 3.8
  let initial_distance : ℝ := 15
  let final_distance : ℝ := 2
  race_time john_speed steve_speed initial_distance final_distance = 42.5 := by
  -- Unfold the definition of race_time
  unfold race_time
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

#eval (15 + 2) / (4.2 - 3.8)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_final_push_time_l973_97337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l973_97306

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

noncomputable def sum_of_terms (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_max_sum
  (a₁ : ℝ) (d : ℝ) (h_a₁ : a₁ > 0) (h_d : d < 0)
  (h_a₅_a₇ : arithmetic_sequence a₁ d 5 = 3 * arithmetic_sequence a₁ d 7) :
  ∃ n : ℕ, (∀ k : ℕ, sum_of_terms a₁ d n ≥ sum_of_terms a₁ d k) →
    (n = 7 ∨ n = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l973_97306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_visible_sum_is_90_l973_97349

/-- Represents a 3x3x3 cube made of dice -/
structure LargeCube where
  size : Nat
  dice : Array (Array (Array Nat))
  size_eq : size = 3

/-- Represents a single die -/
structure Die where
  faces : Array Nat
  opposite_sum : ∀ i, i < 3 → faces[i]! + faces[5 - i]! = 7

/-- The sum of visible values on the large cube -/
def visibleSum (cube : LargeCube) : Nat :=
  sorry

/-- The minimum possible sum of visible values -/
def minVisibleSum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the minimum visible sum is 90 -/
theorem min_visible_sum_is_90 (cube : LargeCube) : minVisibleSum cube = 90 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_visible_sum_is_90_l973_97349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l973_97376

theorem min_value_expression (a b : ℤ) (ha : 0 < a ∧ a < 10) (hb : 0 < b ∧ b < 10) :
  (∀ x y : ℤ, 0 < x ∧ x < 10 → 0 < y ∧ y < 10 → 3 * x - x * y ≥ 3 * a - a * b) →
  3 * a - a * b = -54 :=
by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l973_97376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_relation_l973_97350

/-- Given real numbers c and d satisfying an infinite geometric series equation,
    prove that another related infinite geometric series equals 9/11. -/
theorem geometric_series_relation (c d : ℝ) 
  (h : (c/d) + (c/d^3) + (c/d^6) + (∑' n, c/d^(3*n)) = 9) :
  (c/(c+2*d)) + (c/(c+2*d)^2) + (c/(c+2*d)^3) + (∑' n, c/(c+2*d)^n) = 9/11 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_relation_l973_97350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_for_increasing_f_l973_97316

noncomputable def f (ω : ℕ+) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem max_omega_for_increasing_f :
  ∀ ω : ℕ+,
  (∀ x ∈ Set.Icc (Real.pi / 6) (Real.pi / 4), Monotone (f ω)) →
  ω ≤ 9 ∧ ∃ (ω' : ℕ+), ω' = 9 ∧ (∀ x ∈ Set.Icc (Real.pi / 6) (Real.pi / 4), Monotone (f ω')) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_for_increasing_f_l973_97316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_p_prob_one_incorrect_two_rounds_l973_97385

-- Define the probabilities
noncomputable def prob_A_correct : ℝ := 1/3
noncomputable def prob_B_correct : ℝ := 1/2  -- We know p = 1/2 from the solution
noncomputable def prob_team_correct_first_round : ℝ := 1/2

-- Theorem for the value of p
theorem value_of_p :
  prob_A_correct * (1 - prob_B_correct) + (1 - prob_A_correct) * prob_B_correct = prob_team_correct_first_round :=
by sorry

-- Theorem for the probability of answering one question incorrectly in two rounds
theorem prob_one_incorrect_two_rounds :
  let prob_team_correct_one_round := 1/2
  let prob_team_correct_two_in_round := prob_A_correct * prob_B_correct
  (prob_team_correct_two_in_round * prob_team_correct_one_round +
   prob_team_correct_one_round * prob_team_correct_two_in_round) = 1/6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_p_prob_one_incorrect_two_rounds_l973_97385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_segment_ratio_is_one_to_nine_l973_97339

/-- Represents a right triangle with legs in the ratio 1:3 -/
structure RightTriangle1to3 where
  a : ℝ  -- shorter leg
  b : ℝ  -- longer leg
  c : ℝ  -- hypotenuse
  right_angle : a^2 + b^2 = c^2
  leg_ratio : b = 3 * a
  a_pos : 0 < a

/-- The ratio of segments on the hypotenuse created by the altitude from the right angle -/
noncomputable def hypotenuse_segment_ratio (t : RightTriangle1to3) : ℝ × ℝ :=
  let d := (t.a * t.c) / t.b  -- length of the shorter segment
  (d, t.c - d)

/-- Theorem stating that the ratio of hypotenuse segments is 1:9 -/
theorem hypotenuse_segment_ratio_is_one_to_nine (t : RightTriangle1to3) :
  let (x, y) := hypotenuse_segment_ratio t
  x / y = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_segment_ratio_is_one_to_nine_l973_97339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_is_14_l973_97309

/-- A triangle with an inscribed circle -/
structure InscribedTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of segment BD -/
  bd : ℝ
  /-- The length of segment DC -/
  dc : ℝ
  /-- Assumption that r, bd, and dc are positive -/
  r_pos : r > 0
  bd_pos : bd > 0
  dc_pos : dc > 0

/-- The longest side of a triangle with an inscribed circle -/
noncomputable def longestSide (t : InscribedTriangle) : ℝ :=
  max (t.bd + t.dc) (max (t.bd + t.dc) (t.bd + t.dc))

/-- Theorem: The longest side of the triangle is 14 units -/
theorem longest_side_is_14 (t : InscribedTriangle)
  (h_r : t.r = 3)
  (h_bd : t.bd = 9)
  (h_dc : t.dc = 5) :
  longestSide t = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_is_14_l973_97309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_ratio_theorem_l973_97344

-- Define the triangle DEF
def triangle_DEF (D E F : ℝ × ℝ) : Prop :=
  dist D E = 18 ∧ dist E F = 28 ∧ dist D F = 34

-- Define a point N on line DF
def point_on_line (D N F : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ N = (1 - t) • D + t • F

-- Define the area of a triangle
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

-- Define the condition for equal incircle radii using areas
def equal_incircle_radii (D E N F : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    r * (dist D E + dist E N + dist D N) = 2 * triangle_area D E N ∧
    r * (dist E F + dist F N + dist E N) = 2 * triangle_area E F N

-- Theorem statement
theorem incircle_ratio_theorem (D E F N : ℝ × ℝ) :
  triangle_DEF D E F →
  point_on_line D N F →
  equal_incircle_radii D E N F →
  (dist D N) / (dist N F) = 101 / 50 := by
  sorry

#check incircle_ratio_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_ratio_theorem_l973_97344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l973_97336

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- State the theorem
theorem function_properties (a : ℝ) 
  (h_odd : ∀ x, f a x = -f a (-x)) :
  (a = 1) ∧ 
  (Set.range (f a) = Set.Ioo (-1) 1) ∧
  (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l973_97336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_l973_97319

-- Define the line l
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 2 * Real.sqrt 3 = 0

-- Define the curve C in polar form
def curve_C_polar (ρ θ : ℝ) : Prop := 2 * Real.cos θ = ρ * (1 - Real.cos θ ^ 2)

-- Define the curve C in Cartesian form
def curve_C_cartesian (x y : ℝ) : Prop := y ^ 2 = 2 * x

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ curve_C_cartesian A.1 A.2 ∧
  line_l B.1 B.2 ∧ curve_C_cartesian B.1 B.2 ∧
  A ≠ B

-- Theorem: The midpoint of AB has coordinates (7/3, √3/3)
theorem midpoint_coordinates (A B : ℝ × ℝ) :
  intersection_points A B →
  (A.1 + B.1) / 2 = 7 / 3 ∧ (A.2 + B.2) / 2 = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_l973_97319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_length_proof_l973_97378

-- Define the original thread length
noncomputable def original_length : ℝ := 12

-- Define the additional fraction needed
def additional_fraction : ℚ := 3/4

-- Theorem to prove the total length required
theorem total_length_proof :
  (original_length : ℝ) + (additional_fraction : ℝ) * original_length = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_length_proof_l973_97378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_time_l973_97364

theorem train_tunnel_time (train_length : ℝ) (train_speed_kmh : ℝ) (tunnel_length_km : ℝ) :
  train_length = 100 →
  train_speed_kmh = 72 →
  tunnel_length_km = 3.5 →
  (tunnel_length_km * 1000 + train_length) / (train_speed_kmh * 1000 / 60) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_time_l973_97364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_with_double_slope_angle_l973_97314

noncomputable section

/-- The slope of a line given its equation in the form ax + by + c = 0 -/
def slopeLine (a b : ℝ) : ℝ := -a / b

/-- The slope angle of a line given its slope -/
noncomputable def slopeAngle (m : ℝ) : ℝ := Real.arctan m

/-- The equation of a line passing through (2, 1) with slope angle twice that of x-y-1=0 -/
theorem line_equation_through_point_with_double_slope_angle :
  let l₁ : ℝ → ℝ → ℝ := λ x y => x - y - 1
  let m₁ := slopeLine 1 (-1)
  let α₁ := slopeAngle m₁
  let α₂ := 2 * α₁
  ∃ (f : ℝ → ℝ → ℝ), (f 2 1 = 0) ∧ (slopeAngle (slopeLine 1 0) = α₂) ∧ (∀ x y, f x y = 0 ↔ x = 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_with_double_slope_angle_l973_97314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_at_least_one_vowel_l973_97360

-- Define the sets
def set1 : Finset Char := {'a', 'b', 'c', 'd', 'e'}
def set2 : Finset Char := {'k', 'l', 'm', 'n', 'o', 'p'}
def set3 : Finset Char := {'r', 's', 't', 'u', 'v'}
def set4 : Finset Char := {'w', 'x', 'y', 'z', 'i'}

-- Define vowels in each set
def vowels1 : Finset Char := {'a', 'e'}
def vowels3 : Finset Char := {'u'}
def vowels4 : Finset Char := {'i'}

-- Define the probability function
def prob (event : Finset (Char × Char × Char × Char)) : ℚ :=
  (event.card : ℚ) / ((set1.card * set2.card * set3.card * set4.card) : ℚ)

-- Define valid sequences
def valid_sequences : Finset (Char × Char × Char × Char) :=
  Finset.filter (fun s => 
    (s.1 ∈ set1 ∧ s.2.1 ∈ set2 ∧ s.2.2.1 ∈ set3 ∧ s.2.2.2 ∈ set4) ∧
    ((s.1 ∉ vowels1 ∧ s.2.2.1 ∉ vowels3 ∧ s.2.2.2 ∉ vowels4) ∨
     (s.1 = 'a' ∧ s.2.2.1 = 'u') ∨
     (s.1 = 'e' ∧ s.2.2.2 = 'i'))
  ) (set1.product (set2.product (set3.product set4)))

-- Define sequences with at least one vowel
def vowel_sequences : Finset (Char × Char × Char × Char) :=
  Finset.filter (fun s => 
    s.1 ∈ vowels1 ∨ s.2.2.1 ∈ vowels3 ∨ s.2.2.2 ∈ vowels4
  ) valid_sequences

-- State the theorem
theorem probability_of_at_least_one_vowel :
  prob vowel_sequences = 58 / 125 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_at_least_one_vowel_l973_97360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coord_l973_97318

/-- A parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  h : ∀ x y, equation x y ↔ y^2 = 4*x

/-- A point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- The focus of the parabola y^2 = 4x -/
def parabola_focus : ℝ × ℝ := (1, 0)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- 
Given a parabola y^2 = 4x and a point M on this parabola,
if the distance from M to the focus is 3,
then the x-coordinate of M is 1.
-/
theorem parabola_point_x_coord (p : Parabola) (M : PointOnParabola p) :
  distance (M.x, M.y) parabola_focus = 3 →
  M.x = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coord_l973_97318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_result_l973_97362

def a : Fin 3 → ℝ := ![(-3), 1, (-1)]
def b : Fin 3 → ℝ := ![1, 3, 5]
def c : Fin 3 → ℝ := ![(-2), (-1), 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

def vector_subtract (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => v i - w i

theorem dot_product_result :
  dot_product (vector_subtract a b) c = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_result_l973_97362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_two_l973_97379

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) := sin x
noncomputable def g (x : ℝ) := sin (x - π/3)

-- Define the domain
def domain : Set ℝ := Set.Icc 0 (2 * π)

-- Define the area of the enclosed shape
noncomputable def enclosed_area : ℝ := ∫ x in (2*π/3)..(5*π/3), g x - f x

-- Theorem statement
theorem enclosed_area_is_two :
  enclosed_area = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_two_l973_97379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l973_97383

noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

theorem area_of_specific_triangle : 
  triangle_area 0 0 12 0 0 5 = 30 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l973_97383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_four_factors_l973_97301

def number_list : List Nat := [14, 21, 28, 35, 42]

def has_four_factors (n : Nat) : Bool :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 4

theorem count_numbers_with_four_factors :
  (number_list.filter has_four_factors).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_four_factors_l973_97301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_bounded_sequence_l973_97340

/-- Definition of the sequence a_n -/
def sequence_a (a₁ : ℚ) : ℕ → ℚ
  | 0 => a₁
  | n + 1 => let a_n := sequence_a a₁ n
              let p_n := a_n.num.natAbs
              let q_n := a_n.den
              (p_n^2 + 2015) / (p_n * q_n)

/-- The main theorem to prove -/
theorem exists_bounded_sequence :
  ∃ (a₁ : ℚ), a₁ > 2015 ∧ ∃ (M : ℚ), ∀ (n : ℕ), sequence_a a₁ n ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_bounded_sequence_l973_97340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abCosC_range_is_correct_l973_97320

/-- Triangle ABC with perimeter 16 and AB = 6 -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  perim : a + b + c = 16
  ab_len : c = 6

/-- The range of ab cos C for the given triangle -/
def abCosC_range (t : Triangle) : Set ℝ :=
  { x | ∃ C, x = t.a * t.b * Real.cos C ∧ 7 ≤ x ∧ x < 16 }

/-- Theorem stating the range of ab cos C -/
theorem abCosC_range_is_correct (t : Triangle) :
  abCosC_range t = Set.Icc 7 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abCosC_range_is_correct_l973_97320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_factors_of_2_pow_30_minus_1_l973_97386

theorem two_digit_factors_of_2_pow_30_minus_1 : 
  (Finset.filter (λ n : ℕ => 10 ≤ n ∧ n ≤ 99 ∧ (2^30 - 1) % n = 0) (Finset.range 100)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_factors_of_2_pow_30_minus_1_l973_97386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_theorem_l973_97373

-- Define the set of angles
def angles : Set ℝ := {60, 120, 180, 240, 300}

-- Convert degrees to radians
noncomputable def to_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)

-- Define the property for an angle
def valid_angle (α : ℝ) : Prop :=
  0 < α ∧ α < 360 ∧ ∃ k : ℤ, (6 * α) = k * 360

theorem angle_theorem :
  ∀ α : ℝ, valid_angle α → (to_radians α) ∈ (Set.image to_radians angles) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_theorem_l973_97373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_satisfies_equation_l973_97353

/-- The smallest positive angle in degrees that satisfies the given equation -/
noncomputable def smallest_angle : ℝ := 22.5

theorem smallest_angle_satisfies_equation :
  let x := smallest_angle
  Real.tan (3 * x * Real.pi / 180) = (Real.cos (x * Real.pi / 180) - Real.sin (x * Real.pi / 180)) / (Real.cos (x * Real.pi / 180) + Real.sin (x * Real.pi / 180)) ∧
  ∀ y : ℝ, 0 < y ∧ y < x →
    Real.tan (3 * y * Real.pi / 180) ≠ (Real.cos (y * Real.pi / 180) - Real.sin (y * Real.pi / 180)) / (Real.cos (y * Real.pi / 180) + Real.sin (y * Real.pi / 180)) :=
by sorry

#check smallest_angle_satisfies_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_satisfies_equation_l973_97353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_five_l973_97304

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g as f(x) + x
def g (x : ℝ) : ℝ := f x + x

-- State the theorem
theorem f_neg_two_equals_five :
  (∀ x, g x = g (-x)) →  -- g is an even function
  f 2 = 1 →              -- given condition
  f (-2) = 5 :=          -- conclusion to prove
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_five_l973_97304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_50_mod_5_l973_97338

def lucas : ℕ → ℕ
  | 0 => 2  -- Adding this case to cover Nat.zero
  | 1 => 1
  | 2 => 3
  | n + 3 => lucas (n + 2) + lucas (n + 1)

theorem lucas_50_mod_5 : lucas 50 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_50_mod_5_l973_97338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l973_97366

-- Define α to be in the second quadrant
noncomputable def α : ℝ := Real.arctan 2

-- Given conditions
axiom tan_α : Real.tan α = 2
axiom sin_α : Real.sin α = 2 * Real.sqrt 5 / 5
axiom α_second_quadrant : Real.pi < α ∧ α < 3 * Real.pi / 2

theorem trigonometric_identities :
  (4 * (Real.sin α)^2 + 2 * Real.sin α * Real.cos α = 4) ∧
  (Real.tan (α + 3 * Real.pi) + Real.sin (9 * Real.pi / 2 + α) / Real.cos (9 * Real.pi / 2 - α) = -(4 * Real.sqrt 5 + 5) / 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l973_97366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_bound_l973_97390

open Real

theorem extreme_point_bound (b : ℝ) (x₁ x₂ : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ log x - x^2
  let g : ℝ → ℝ := λ x ↦ f x + 3/2 * x^2 - (1 - b) * x
  x₁ < x₂ →
  x₁^2 - (1 + b) * x₁ + 1 = 0 →
  x₂^2 - (1 + b) * x₂ + 1 = 0 →
  b ≥ (exp 1)^2 / exp 1 + 1 / exp 1 - 1 →
  x₂ ≥ exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_bound_l973_97390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brand_z_percentage_l973_97359

/-- Represents the capacity of the fuel tank -/
def tank_capacity : ℝ := 100

/-- Represents the amount of brand Z gasoline after initial filling -/
def initial_z : ℝ := tank_capacity

/-- Represents the amount of brand Z gasoline after first partial emptying -/
def z_after_first_empty : ℝ := 0.75 * initial_z

/-- Represents the amount of brand Z gasoline after second partial emptying -/
def z_after_second_empty : ℝ := 0.5 * z_after_first_empty

/-- Represents the amount of brand Z gasoline after third partial emptying -/
def z_after_third_empty : ℝ := 0.5 * z_after_second_empty

/-- Represents the amount of brand Z gasoline after fourth partial emptying -/
def z_after_fourth_empty : ℝ := 0.25 * z_after_third_empty

/-- Represents the amount of brand Z gasoline added in final filling -/
def z_final_fill : ℝ := 0.75 * tank_capacity

/-- Represents the total amount of brand Z gasoline at the end -/
def total_z : ℝ := z_after_fourth_empty + z_final_fill

/-- Theorem stating that the percentage of brand Z gasoline at the end is approximately 79.69% -/
theorem brand_z_percentage : ∃ ε > 0, |((total_z / tank_capacity) * 100) - 79.69| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brand_z_percentage_l973_97359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_double_angle_problem_l973_97342

theorem cosine_double_angle_problem (x y : ℝ) : 
  Real.cos x * Real.cos y + Real.sin x * Real.sin y = 1/3 → 
  Real.cos (2*x - 2*y) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_double_angle_problem_l973_97342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_2a_minus_b_l973_97351

def vector_a : ℝ × ℝ := (-1, 2)
def vector_b : ℝ × ℝ := (1, 3)

theorem magnitude_of_2a_minus_b : 
  Real.sqrt ((2 * vector_a.1 - vector_b.1)^2 + (2 * vector_a.2 - vector_b.2)^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_2a_minus_b_l973_97351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_correct_l973_97347

/-- A geometric progression with given first and fifth terms -/
structure GeometricProgression where
  b1 : ℝ
  b5 : ℝ
  h1 : b1 = Real.sqrt 3
  h5 : b5 = Real.sqrt 243

/-- The common ratio and sixth term of the geometric progression -/
noncomputable def geometric_progression_solution (gp : GeometricProgression) : ℝ × ℝ :=
  let q := Real.sqrt 3
  let b6 := 27
  (q, b6)

/-- Theorem stating the correctness of the solution -/
theorem geometric_progression_correct (gp : GeometricProgression) :
  let (q, b6) := geometric_progression_solution gp
  (q = Real.sqrt 3 ∨ q = -Real.sqrt 3) ∧ (b6 = 27 ∨ b6 = -27) := by
  sorry

#check geometric_progression_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_correct_l973_97347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_lightest_and_heaviest_l973_97397

/-- Represents a stone with a unique weight -/
structure Stone where
  weight : ℕ  -- Changed to ℕ for simplicity

/-- Represents the result of comparing two stones on a balance scale -/
inductive ComparisonResult
  | Left  : ComparisonResult  -- Left stone is heavier
  | Right : ComparisonResult  -- Right stone is heavier

/-- A function that compares two stones and returns the result -/
def compare (s1 s2 : Stone) : ComparisonResult :=
  if s1.weight > s2.weight then ComparisonResult.Left else ComparisonResult.Right

/-- Theorem stating that 3N-2 comparisons are sufficient to find the lightest and heaviest stones -/
theorem find_lightest_and_heaviest (N : ℕ) (stones : Finset Stone)
  (h_count : stones.card = 2 * N)
  (h_distinct : ∀ s1 s2 : Stone, s1 ∈ stones → s2 ∈ stones → s1 ≠ s2 → s1.weight ≠ s2.weight) :
  ∃ (lightest heaviest : Stone) (num_comparisons : ℕ),
    lightest ∈ stones ∧
    heaviest ∈ stones ∧
    (∀ s ∈ stones, lightest.weight ≤ s.weight) ∧
    (∀ s ∈ stones, s.weight ≤ heaviest.weight) ∧
    num_comparisons ≤ 3 * N - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_lightest_and_heaviest_l973_97397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleZeroSum_l973_97323

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a sequence of digits -/
def DigitSequence := List Digit

/-- Represents the possible signs between digits -/
inductive Sign
| Plus
| Minus

/-- Represents a sequence of signs -/
def SignSequence := List Sign

/-- The specific sequence of digits for the number 20222023 -/
def number : DigitSequence := [⟨2, by norm_num⟩, ⟨0, by norm_num⟩, ⟨2, by norm_num⟩, 
                               ⟨2, by norm_num⟩, ⟨2, by norm_num⟩, ⟨0, by norm_num⟩, 
                               ⟨2, by norm_num⟩, ⟨3, by norm_num⟩]

/-- Function to evaluate the expression given a sequence of digits and signs -/
def evaluateExpression (digits : DigitSequence) (signs : SignSequence) : Int :=
  sorry

/-- Theorem stating that it's impossible to find a sign sequence that makes the expression zero -/
theorem impossibleZeroSum : ∀ (signs : SignSequence), 
  evaluateExpression number signs ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleZeroSum_l973_97323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_tower_height_example_l973_97315

/-- Calculates the height of a scaled model tower given the original tower's height and dome volume, and the model's dome volume. -/
noncomputable def scaled_tower_height (original_height : ℝ) (original_volume : ℝ) (model_volume : ℝ) : ℝ :=
  original_height / (original_volume / model_volume) ^ (1/3)

/-- Theorem stating that a 60m tower with 150,000L dome volume scaled to a model with 0.15L dome volume should be 0.6m tall. -/
theorem scaled_tower_height_example : 
  scaled_tower_height 60 150000 0.15 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_tower_height_example_l973_97315
