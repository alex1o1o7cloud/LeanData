import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2010_equals_4_l1037_103726

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 2  -- Define for 0 to avoid missing case
  | 1 => 2
  | 2 => 3
  | n + 3 => (a (n + 2) * a (n + 1)) % 10

-- Define the theorem
theorem a_2010_equals_4 : a 2010 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2010_equals_4_l1037_103726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_dot_product_l1037_103781

-- Define a triangle ABC in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define vector operations
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Theorem statement
theorem triangle_vector_dot_product (t : Triangle) 
  (h1 : vec_length (vec t.B t.C) = 4)
  (h2 : dot_product (vec t.A t.B + vec t.A t.C) (vec t.B t.C) = 0) :
  dot_product (vec t.B t.A) (vec t.B t.C) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_dot_product_l1037_103781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_constant_l1037_103782

/-- Sequence c_n -/
def c : ℕ → ℤ
  | 0 => 1  -- Adding case for 0 to cover all natural numbers
  | 1 => 0  -- Correcting the value for c_1 based on the problem statement
  | 2 => 2005
  | n + 3 => -3 * c (n + 1) - 4 * c (n + 2) + 2008

/-- Sequence a_n -/
def a (n : ℕ) : ℤ :=
  if n ≥ 2 then
    3 * (c (n - 2) - n) * (502 - (n + 1) - c (n + 2)) + 4^n * 2004 * 501
  else
    0  -- Defining a value for n < 2 to make the function total

/-- Theorem stating that a_n is constant for n > 2 -/
theorem a_is_constant : ∀ m n : ℕ, m > 2 → n > 2 → a m = a n := by
  sorry

#check a_is_constant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_constant_l1037_103782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_true_l1037_103789

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- Define the propositions
def prop1 (Line Plane : Type) 
  (parallel : Line → Line → Prop) 
  (parallel_plane : Plane → Plane → Prop) 
  (line_parallel_plane : Line → Plane → Prop) : Prop := 
  ∀ (m n : Line) (α β : Plane), 
    line_parallel_plane n α → line_parallel_plane m β → parallel_plane α β → parallel m n

def prop2 (Line Plane : Type) 
  (perpendicular : Line → Line → Prop) 
  (line_parallel_plane : Line → Plane → Prop) 
  (line_perpendicular_plane : Line → Plane → Prop) : Prop := 
  ∀ (m n : Line) (α : Plane), 
    line_perpendicular_plane m α → line_parallel_plane n α → perpendicular m n

def prop3 (Plane : Type) 
  (parallel_plane : Plane → Plane → Prop) 
  (perpendicular_plane : Plane → Plane → Prop) : Prop := 
  ∀ (α β γ : Plane), 
    perpendicular_plane α γ → perpendicular_plane β γ → parallel_plane α β

def prop4 (Line Plane : Type) 
  (parallel_plane : Plane → Plane → Prop) 
  (line_perpendicular_plane : Line → Plane → Prop) : Prop := 
  ∀ (m : Line) (α β γ : Plane), 
    parallel_plane α β → parallel_plane β γ → line_perpendicular_plane m α → line_perpendicular_plane m γ

-- The main theorem
theorem exactly_two_true : 
  (¬prop1 Line Plane parallel parallel_plane line_parallel_plane ∧ 
   prop2 Line Plane perpendicular line_parallel_plane line_perpendicular_plane ∧ 
   ¬prop3 Plane parallel_plane perpendicular_plane ∧ 
   prop4 Line Plane parallel_plane line_perpendicular_plane) ∧ 
  m ≠ n ∧ α ≠ β ∧ α ≠ γ ∧ β ≠ γ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_true_l1037_103789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_inequality_area_inequality_l1037_103733

open Real

/-- Represents a polygon in 2D space --/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Checks if a polygon is convex --/
def is_convex (n : ℕ) (P : Polygon n) : Prop := sorry

/-- Checks if one polygon is the midpoint polygon of another --/
def is_midpoint_polygon (n : ℕ) (P Q : Polygon n) : Prop := sorry

/-- Calculates the perimeter of a polygon --/
noncomputable def perimeter (n : ℕ) (P : Polygon n) : ℝ := sorry

/-- Calculates the area of a polygon --/
noncomputable def area (n : ℕ) (P : Polygon n) : ℝ := sorry

/-- Theorem for perimeter inequality --/
theorem perimeter_inequality {n : ℕ} (M M' : Polygon n)
    (h1 : n ≥ 3) (h2 : is_convex n M) (h3 : is_midpoint_polygon n M M') :
    perimeter n M > perimeter n M' ∧ perimeter n M' ≥ (1/2) * perimeter n M := by
  sorry

/-- Theorem for area inequality --/
theorem area_inequality {n : ℕ} (M M' : Polygon n)
    (h1 : n ≥ 4) (h2 : is_convex n M) (h3 : is_midpoint_polygon n M M') :
    area n M > area n M' ∧ area n M' ≥ (1/2) * area n M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_inequality_area_inequality_l1037_103733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1037_103793

noncomputable def circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

noncomputable def point_to_line_distance (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

def is_tangent (h k r A B C : ℝ) : Prop :=
  point_to_line_distance h k A B C = r

theorem circle_tangent_to_line (x y : ℝ) :
  let h : ℝ := 2
  let k : ℝ := -1
  let A : ℝ := 3
  let B : ℝ := -4
  let C : ℝ := 5
  let r : ℝ := point_to_line_distance h k A B C
  is_tangent h k r A B C →
  circle_equation h k r x y :=
by
  sorry

#check circle_tangent_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1037_103793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_street_crossing_speed_l1037_103722

/-- Calculates the speed in km/h given distance in meters and time in minutes -/
noncomputable def calculate_speed (distance_m : ℝ) (time_min : ℝ) : ℝ :=
  (distance_m / 1000) / (time_min / 60)

/-- Proves that crossing 708 meters in 8 minutes results in a speed of approximately 5.31 km/h -/
theorem street_crossing_speed : 
  let distance_m : ℝ := 708
  let time_min : ℝ := 8
  let speed := calculate_speed distance_m time_min
  ∃ ε > 0, |speed - 5.31| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_street_crossing_speed_l1037_103722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_after_reflection_l1037_103711

-- Define the coordinates of point F
def F : ℝ × ℝ := (4, 3)

-- Define the reflection of F over the y-axis
def F' : ℝ × ℝ := (-F.1, F.2)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem segment_length_after_reflection :
  distance F F' = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_after_reflection_l1037_103711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_TR_equals_square_side_l1037_103715

-- Define the square PQRS
def square_side : ℝ := 10

-- Define the angles
def angle_SPT : ℝ := 75
def angle_TSP : ℝ := 30

-- Define point T inside the square
structure Point := (x : ℝ) (y : ℝ)

def P : Point := ⟨0, 0⟩
def Q : Point := ⟨square_side, 0⟩
def R : Point := ⟨square_side, square_side⟩
def S : Point := ⟨0, square_side⟩

-- T is inside the square, but we don't know its exact coordinates
noncomputable def T : Point := sorry

-- Define the distance function
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- State the theorem
theorem length_TR_equals_square_side :
  distance T R = square_side := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_TR_equals_square_side_l1037_103715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tires_in_batch_is_15000_l1037_103749

/-- Represents the tire production and sales problem -/
structure TireProduction where
  batchCost : ℚ
  productionCostPerTire : ℚ
  sellingPricePerTire : ℚ
  profitPerTire : ℚ

/-- Calculates the number of tires produced and sold in a batch -/
noncomputable def tiresInBatch (tp : TireProduction) : ℚ :=
  tp.batchCost / (tp.sellingPricePerTire - tp.productionCostPerTire - tp.profitPerTire)

/-- Theorem stating that given the specific conditions, the number of tires in the batch is 15000 -/
theorem tires_in_batch_is_15000 :
  let tp : TireProduction := {
    batchCost := 22500,
    productionCostPerTire := 8,
    sellingPricePerTire := 20,
    profitPerTire := 21/2
  }
  tiresInBatch tp = 15000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tires_in_batch_is_15000_l1037_103749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_formula_l1037_103779

-- Define the regions M and N
def M : Set (ℝ × ℝ) := {p | p.2 ≥ 0 ∧ p.2 ≤ p.1 ∧ p.2 ≤ 2 - p.1}
def N (t : ℝ) : Set (ℝ × ℝ) := {p | t ≤ p.1 ∧ p.1 ≤ t + 1}

-- Define the parameter t
def t_range : Set ℝ := {t | 0 ≤ t ∧ t ≤ 1}

-- Define the common area function
noncomputable def f (t : ℝ) : ℝ := (MeasureTheory.volume (M ∩ N t)).toReal

-- State the theorem
theorem common_area_formula (t : ℝ) (h : t ∈ t_range) : 
  f t = -t^2 + t + 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_formula_l1037_103779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_intercept_triangle_line_equation_l1037_103777

/-- A line with slope 3/4 forming a triangle with the coordinate axes -/
structure TriangleLine where
  slope : ℚ
  yIntercept : ℝ
  slope_eq : slope = 3/4

/-- The area of the triangle formed by the line and the coordinate axes -/
noncomputable def triangleArea (line : TriangleLine) : ℝ :=
  (1/2) * |line.yIntercept| * |(4/3) * line.yIntercept|

/-- Theorem stating that a line with slope 3/4 forming a triangle with area 6 
    must have y-intercept of either 3 or -3 -/
theorem triangle_line_intercept (line : TriangleLine) 
  (area_eq : triangleArea line = 6) : 
  line.yIntercept = 3 ∨ line.yIntercept = -3 := by
  sorry

/-- The main theorem proving that the equation of the line must be 
    either y = 3/4x + 3 or y = 3/4x - 3 -/
theorem triangle_line_equation (line : TriangleLine) 
  (area_eq : triangleArea line = 6) : 
  (∀ x, line.slope * x + line.yIntercept = 3/4 * x + 3) ∨
  (∀ x, line.slope * x + line.yIntercept = 3/4 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_intercept_triangle_line_equation_l1037_103777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_less_than_a_less_than_b_l1037_103765

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the derivative of f
noncomputable def f' : ℝ → ℝ := sorry

-- Assumption: f is differentiable on ℝ
axiom f_differentiable : Differentiable ℝ f

-- Condition: (x-1) · f'(x) - f(x) · (x-1)' > 0 for x ∈ (1, +∞)
axiom condition_holds (x : ℝ) (h : x > 1) : (x - 1) * f' x - f x > 0

-- Define a, b, and c
noncomputable def a : ℝ := f 2
noncomputable def b : ℝ := (1 / 2) * f 3
noncomputable def c : ℝ := (1 / (Real.sqrt 2 - 1)) * f (Real.sqrt 2)

-- Theorem to prove
theorem c_less_than_a_less_than_b : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_less_than_a_less_than_b_l1037_103765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_guessing_game_l1037_103790

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def sequence_range (start : ℕ) : List ℕ :=
  List.range 2005 |>.map (λ x => start + x + 1)

def classify_sequence (seq : List ℕ) : List Prop :=
  seq.map is_prime

theorem number_guessing_game :
  ∀ i j : ℕ, 1 ≤ i → i < j → j ≤ 2006 →
    classify_sequence (sequence_range i) ≠ classify_sequence (sequence_range j) :=
by
  sorry

#check number_guessing_game

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_guessing_game_l1037_103790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_ratio_for_given_cycle_l1037_103747

/-- Represents an ideal gas thermodynamic cycle --/
structure IdealGasCycle where
  T_max : ℝ
  T_min : ℝ
  η : ℝ

/-- The ratio of final to initial absolute temperatures during isochoric heating --/
noncomputable def temperature_ratio (cycle : IdealGasCycle) : ℝ :=
  (cycle.T_max / cycle.T_min) * (1 - cycle.η)

/-- Theorem stating the temperature ratio for the given cycle --/
theorem temperature_ratio_for_given_cycle :
  let cycle : IdealGasCycle := {
    T_max := 900,
    T_min := 350,
    η := 0.4
  }
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |temperature_ratio cycle - 1.54| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_ratio_for_given_cycle_l1037_103747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_bd_ratio_l1037_103729

-- Define the line segment and points
structure LineSegment where
  point : ℝ

def A : LineSegment := ⟨0⟩
def B : LineSegment := ⟨1⟩
def C : LineSegment := ⟨3⟩
def D : LineSegment := ⟨4.875⟩

-- Define the ratios
def ratio_AB_BC : ℚ := 1 / 2
def ratio_BC_CD : ℚ := 8 / 5

-- Define the order of points
axiom order : A.point < B.point ∧ B.point < C.point ∧ C.point < D.point

-- Define the ratios in terms of distances
axiom ratio_def_1 : (B.point - A.point) / (C.point - B.point) = ratio_AB_BC
axiom ratio_def_2 : (C.point - B.point) / (D.point - C.point) = ratio_BC_CD

-- Theorem to prove
theorem ab_bd_ratio : (B.point - A.point) / (D.point - B.point) = 4 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_bd_ratio_l1037_103729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_little_theorem_eulers_theorem_l1037_103700

-- Define Euler's totient function
def phi : ℕ → ℕ := sorry

-- Define the congruence relation
def congruent (a b m : ℕ) : Prop := ∃ k : ℤ, (a : ℤ) - (b : ℤ) = k * (m : ℤ)

-- Fermat's Little Theorem (given)
theorem fermats_little_theorem (p : ℕ) (a : ℕ) (h_prime : Nat.Prime p) (h_coprime : Nat.Coprime a p) :
  congruent (a^(p - 1)) 1 p := sorry

-- Euler's theorem statement
theorem eulers_theorem (m : ℕ) (a : ℕ) (h_pos : m > 0) (h_coprime : Nat.Coprime a m) :
  congruent (a^(phi m)) 1 m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_little_theorem_eulers_theorem_l1037_103700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaohua_home_to_school_distance_l1037_103710

/-- Represents the route from Xiaohua's home to school -/
structure Route where
  flat : ℚ  -- Length of flat road in meters
  slope : ℚ  -- Length of slope (downhill/uphill) in meters

/-- Calculates the time taken for a journey given the route and speeds -/
def journey_time (r : Route) (flat_speed uphill_speed : ℚ) : ℚ :=
  r.flat / flat_speed + r.slope / uphill_speed

/-- The route from Xiaohua's home to school -/
noncomputable def xiaohua_route : Route :=
  { flat := 300, slope := 400 }

theorem xiaohua_home_to_school_distance :
  let r := xiaohua_route
  let flat_speed := 60
  let downhill_speed := 80
  let uphill_speed := 40
  journey_time r flat_speed downhill_speed = 10 ∧
  journey_time r flat_speed uphill_speed = 15 →
  r.flat + r.slope = 700 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaohua_home_to_school_distance_l1037_103710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_sector_area_value_l1037_103795

/-- The area of the region inside an octagon but outside circular sectors --/
noncomputable def octagon_sector_area (side_length : ℝ) (sector_radius : ℝ) : ℝ :=
  let octagon_area := 8 * (side_length^2 / 4) * Real.tan (45 / 2 * Real.pi / 180)
  let sector_area := 8 * (Real.pi * sector_radius^2 * 45 / 360)
  octagon_area - sector_area

/-- Theorem stating the area of the region for the given octagon and sectors --/
theorem octagon_sector_area_value :
  octagon_sector_area 5 4 = 50 - 16 * Real.pi := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval octagon_sector_area 5 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_sector_area_value_l1037_103795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1037_103713

/-- The constant term in the expansion of (x + 1/(3x))^8 is 28 -/
theorem constant_term_binomial_expansion :
  (Finset.sum (Finset.range 9) (fun r => Nat.choose 8 r * (1 / 3 ^ (8 - r)) * (if r = 2 then 1 else 0))) = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1037_103713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_doubles_in_five_years_l1037_103773

/-- The annual interest rate as a decimal -/
noncomputable def r : ℚ := 15 / 100

/-- The number of times interest is compounded per year -/
def n : ℕ := 1

/-- The compound interest factor for one year -/
noncomputable def factor : ℚ := 1 + r / n

/-- The number of years we're proving about -/
def t : ℕ := 5

theorem money_doubles_in_five_years :
  (∀ k : ℕ, k < t → (factor : ℝ) ^ (n * k) ≤ 2) ∧
  (factor : ℝ) ^ (n * t) > 2 := by
  sorry

#check money_doubles_in_five_years

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_doubles_in_five_years_l1037_103773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_center_distance_l1037_103728

/-- Two wheels rolling side-by-side on a flat surface -/
structure TwoWheels where
  radius : ℝ
  revolution : ℝ

/-- The horizontal distance traveled by the center of Wheel A -/
noncomputable def distance (w : TwoWheels) : ℝ := 2 * Real.pi * w.radius * w.revolution

theorem wheel_center_distance (w : TwoWheels) 
  (h1 : w.radius = 1)
  (h2 : w.revolution = 1) : 
  distance w = 2 * Real.pi := by
  sorry

#check wheel_center_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_center_distance_l1037_103728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_K_equals_one_plus_two_ln_two_l1037_103797

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- Define the constant K
def K : ℝ := 1

-- Define the function f_K
noncomputable def f_K (x : ℝ) : ℝ :=
  if f x ≤ K then K else f x

-- State the theorem
theorem integral_f_K_equals_one_plus_two_ln_two :
  ∫ x in (1/4)..2, f_K x = 1 + 2 * Real.log 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_K_equals_one_plus_two_ln_two_l1037_103797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_energy_increase_percentage_l1037_103737

/-- Represents the percentage of new energy vehicles in total production last year -/
def last_year_new_energy_percentage : ℝ := 0.1

/-- Represents the reduction percentage in conventional car production this year -/
def conventional_car_reduction : ℝ := 0.1

/-- Theorem stating that the percentage increase in new energy vehicle production
    should be 90% to maintain the same total production -/
theorem new_energy_increase_percentage (a x : ℝ) :
  (1 - last_year_new_energy_percentage) * a * (1 - conventional_car_reduction) +
  last_year_new_energy_percentage * a * (1 + x) = a →
  x = 0.9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_energy_increase_percentage_l1037_103737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l1037_103798

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  p : Point
  q : Point
  r : Point

/-- Calculates the centroid of a triangle -/
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.p.x + t.q.x + t.r.x) / 3,
    y := (t.p.y + t.q.y + t.r.y) / 3 }

/-- Calculates the area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  (1/2) * abs ((t.p.x * (t.q.y - t.r.y) + t.q.x * (t.r.y - t.p.y) + t.r.x * (t.p.y - t.q.y)))

/-- Checks if a side of a triangle is horizontal -/
def isHorizontal (p1 : Point) (p2 : Point) : Prop :=
  p1.y = p2.y

/-- Applies dilation to a point -/
def dilate (p : Point) (center : Point) (scale : ℝ) : Point :=
  { x := center.x + scale * (p.x - center.x),
    y := center.y + scale * (p.y - center.y) }

/-- Calculates the distance between two points -/
noncomputable def distance (p1 : Point) (p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem to prove -/
theorem farthest_vertex_after_dilation (t : Triangle) :
  centroid t = { x := 4, y := -4 } →
  area t = 9 →
  isHorizontal t.p t.q →
  let t' := Triangle.mk
    (dilate t.p { x := 0, y := 0 } 3)
    (dilate t.q { x := 0, y := 0 } 3)
    (dilate t.r { x := 0, y := 0 } 3)
  let origin := { x := 0, y := 0 }
  let farthest := max (distance t'.p origin) (max (distance t'.q origin) (distance t'.r origin))
  farthest = distance { x := 12, y := -12 + 6 * Real.sqrt 3 } origin :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l1037_103798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_intervals_inequality_condition_l1037_103754

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^3 + a*x^2 - a^2*x + 2

def f_derivative (x : ℝ) : ℝ := 3*x^2 + 2*a*x - a^2

theorem tangent_line_at_one (h : a = 1) :
  ∃ (k m : ℝ), k * x - y - m = 0 ∧ 
  (∀ x, k * x - f a x - m = 0 ↔ x = 1) :=
sorry

theorem monotonicity_intervals (h : a ≠ 0) :
  (∀ x₁ x₂, x₁ < x₂ → x₂ < -a → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, -a < x₁ → x₁ < x₂ → x₂ < a/3 → f a x₁ > f a x₂) ∧
  (∀ x₁ x₂, a/3 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) :=
sorry

theorem inequality_condition :
  (∀ x > 0, 2*x*(Real.log x) ≤ f_derivative a x + a^2 + 1) ↔ 
  a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_intervals_inequality_condition_l1037_103754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1037_103708

noncomputable def given_numbers : List ℝ := [-3, -1/3, -|-3|, Real.pi, -0.3, 0, (16 : ℝ) ^ (1/3), 1.1010010001]

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n
def is_negative_fraction (x : ℝ) : Prop := x < 0 ∧ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b
def is_irrational (x : ℝ) : Prop := ¬∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem number_categorization :
  (∀ x ∈ given_numbers, is_integer x ↔ x ∈ ({-3, -|-3|, 0} : Set ℝ)) ∧
  (∀ x ∈ given_numbers, is_negative_fraction x ↔ x ∈ ({-1/3, -0.3} : Set ℝ)) ∧
  (∀ x ∈ given_numbers, is_irrational x ↔ x ∈ ({Real.pi, (16 : ℝ) ^ (1/3)} : Set ℝ)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1037_103708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_composite_for_all_n_l1037_103739

theorem exists_k_composite_for_all_n : ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n > 0 → ∃ m : ℕ, m > 1 ∧ m < 2^n * k + 1 ∧ (2^n * k + 1) % m = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_composite_for_all_n_l1037_103739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_proof_l1037_103799

-- Define the given constants
noncomputable def total_volume : ℝ := 10
noncomputable def final_alcohol_percentage : ℝ := 45 / 100
noncomputable def first_mixture_volume : ℝ := 2.5
noncomputable def second_mixture_percentage : ℝ := 50 / 100

-- Define the unknown percentage as a variable
noncomputable def first_mixture_percentage : ℝ := 30 / 100

-- Theorem statement
theorem alcohol_mixture_proof :
  first_mixture_percentage * first_mixture_volume +
  second_mixture_percentage * (total_volume - first_mixture_volume) =
  final_alcohol_percentage * total_volume :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_proof_l1037_103799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_statements_l1037_103756

open Real

-- Define the functions
noncomputable def f (x : ℝ) := sin x ^ 4 - cos x ^ 4
noncomputable def g (x : ℝ) := 3 * sin x
noncomputable def h (x : ℝ) := 3 * sin (2 * x)
noncomputable def j (x : ℝ) := sin x

-- State the theorem
theorem trigonometric_statements :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (p : ℝ), p = π ∧ p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  ¬(∃ (k : ℝ), ∀ (x : ℝ), h x = g (x - k)) ∧
  ¬(∀ (x : ℝ), x ∈ Set.Icc 0 π → (deriv j) x ≤ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_statements_l1037_103756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patrol_analysis_l1037_103759

def travel_record : List Int := [9, -8, 6, -15, 6, -14, 4, -3]

theorem patrol_analysis (a : ℝ) (a_pos : a > 0) :
  let final_position := travel_record.sum
  let total_distance := (travel_record.map Int.natAbs).sum
  (final_position = -15 ∧ total_distance = 65) ∧
  (a * (total_distance : ℝ) = 65 * a) := by
  sorry

#check patrol_analysis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_patrol_analysis_l1037_103759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_perfect_squares_l1037_103721

/-- Definition of the nth term of the sequence -/
def x (n : ℕ) : ℕ := 
  4 * (10^(2*n) - 10^n) / 9 + 889

/-- Theorem stating that each term in the sequence is a perfect square -/
theorem sequence_perfect_squares (n : ℕ) : 
  ∃ k : ℕ, x n = k^2 := by
  -- We'll use k = (2 * 10^n + 1) / 3
  let k := (2 * 10^n + 1) / 3
  use k
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_perfect_squares_l1037_103721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_length_l1037_103780

-- Define the right triangle PQR
structure RightTriangle :=
  (PQ : ℝ)
  (PR : ℝ)
  (is_right_angle : PQ > 0 ∧ PR > 0)

-- Define the altitude PS
noncomputable def altitude (t : RightTriangle) : ℝ :=
  Real.sqrt (t.PQ * t.PR)

-- Theorem statement
theorem altitude_length (t : RightTriangle) 
  (h1 : t.PQ = 3) 
  (h2 : t.PR = 4) : 
  altitude t = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_length_l1037_103780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l1037_103719

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 30*x + y^2 - 8*y + 325 = 0

/-- The shortest distance from the origin to the circle -/
noncomputable def shortest_distance : ℝ := Real.sqrt 241 - 2 * Real.sqrt 21

/-- Theorem stating that the shortest_distance is less than or equal to
    the distance from the origin to any point on the circle -/
theorem shortest_distance_to_circle :
  ∀ (x y : ℝ), circle_equation x y →
  shortest_distance ≤ Real.sqrt (x^2 + y^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l1037_103719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rhombus_l1037_103752

/-- The area enclosed by the graph of |2x| + |3y| = 6 is 12 -/
theorem area_of_rhombus :
  ∃ (A : Set (ℝ × ℝ)) (area : ℝ),
    A = {(x, y) | |2*x| + |3*y| = 6} ∧
    area = 12 ∧
    (MeasureTheory.volume A).toReal = area :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rhombus_l1037_103752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_volume_l1037_103701

/-- Represents the initial volume of water in liters -/
def initial_water : ℝ := sorry

/-- Represents the initial volume of milk in liters -/
def initial_milk : ℝ := sorry

/-- Represents the volume of water added in liters -/
def water_added : ℝ := 1.6

/-- The initial ratio of milk to water is 7:1 -/
axiom initial_ratio : initial_milk = 7 * initial_water

/-- The new ratio of milk to water after adding water is 3:1 -/
axiom new_ratio : initial_milk / (initial_water + water_added) = 3

/-- The theorem to prove: the initial volume of the mixture is 9.6 liters -/
theorem initial_mixture_volume : initial_milk + initial_water = 9.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_volume_l1037_103701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_circle_properties_l1037_103757

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 1

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4

-- Define the points
def P : ℝ × ℝ := (-1, 2)
def M : ℝ × ℝ := (2, 1)
def N : ℝ × ℝ := (4, -1)

-- Theorem statement
theorem line_and_circle_properties :
  -- Line l passes through P
  line_l P.1 P.2 ∧
  -- Area of triangle formed by line l and positive semi-axes is 1/2
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ line_l a 0 ∧ line_l 0 b ∧ a * b / 2 = 1/2) ∧
  -- Circle passes through M and N
  my_circle M.1 M.2 ∧ my_circle N.1 N.2 ∧
  -- Circle center is on line l
  (∃ x y : ℝ, line_l x y ∧ ∀ a b : ℝ, my_circle a b ↔ (a - x)^2 + (b - y)^2 = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_circle_properties_l1037_103757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_2x_minus_23_l1037_103748

theorem cube_root_of_2x_minus_23 (x : ℝ) (h : (2 * x - 1).sqrt = 7 ∨ (2 * x - 1).sqrt = -7) : 
  ∃ y : ℝ, y^3 = 2 * x - 23 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_2x_minus_23_l1037_103748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_and_initial_phase_l1037_103709

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 * sin (-x + π/6)

-- State the theorem
theorem phase_and_initial_phase :
  ∃ (phase : ℝ → ℝ) (initial_phase : ℝ),
    (∀ x, f x = 3 * sin (phase x)) ∧
    phase 0 = initial_phase ∧
    (∀ x, phase x = x + 5*π/6) ∧
    initial_phase = 5*π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_and_initial_phase_l1037_103709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_c_range_l1037_103714

-- Define the function f(x)
noncomputable def f (x c : ℝ) : ℝ := x * Real.exp x + c

-- State the theorem
theorem two_zeros_c_range :
  ∀ c : ℝ, (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ c = 0 ∧ f x₂ c = 0) → 0 < c ∧ c < Real.exp (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_c_range_l1037_103714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_inclination_l1037_103723

-- Define a line in 2D space
structure Line2D where
  slope : ℝ
  intercept : ℝ

-- Define the angle of inclination for a line
noncomputable def angleOfInclination (l : Line2D) : ℝ :=
  Real.arctan l.slope * (180 / Real.pi)

-- Define perpendicularity of two lines
def isPerpendicular (l1 l2 : Line2D) : Prop :=
  l1.slope * l2.slope = -1

theorem perpendicular_lines_inclination (l1 l2 : Line2D) :
  isPerpendicular l1 l2 →
  angleOfInclination l2 = 135 →
  angleOfInclination l1 = 45 :=
by
  sorry

#check perpendicular_lines_inclination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_inclination_l1037_103723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_two_l1037_103784

-- Define the circle C in parametric form
noncomputable def circle_C (φ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos φ, Real.sin φ)

-- Define the polar equation of line l
def line_l (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin (θ + Real.pi/3) = 3 * Real.sqrt 3

-- Define the ray OM
def ray_OM (θ : ℝ) : Prop :=
  θ = Real.pi/3

-- Define the polar equation of circle C
def polar_circle_C (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ

-- Theorem statement
theorem length_PQ_is_two :
  ∃ (ρ_P ρ_Q : ℝ),
    polar_circle_C ρ_P (Real.pi/3) ∧
    line_l ρ_Q (Real.pi/3) ∧
    ray_OM (Real.pi/3) ∧
    ρ_Q - ρ_P = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_two_l1037_103784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_function_l1037_103770

noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

def shrink_abscissa (f : ℝ → ℝ) : ℝ → ℝ := fun x ↦ f (2 * x)

def shift_right (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := fun x ↦ f (x - shift)

theorem transform_sin_function :
  ∀ x : ℝ, (shift_right (shrink_abscissa original_function) (π / 3)) x =
  Real.sin (2 * x - 2 * π / 3) := by
  intro x
  simp [shift_right, shrink_abscissa, original_function]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_function_l1037_103770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pentagon_area_theorem_l1037_103788

/-- A trapezoid with inscribed and circumscribed circles -/
structure SpecialTrapezoid where
  /-- Length of the smaller base -/
  small_base : ℝ
  /-- Length of the larger base -/
  large_base : ℝ
  /-- The trapezoid has an inscribed circle -/
  has_inscribed_circle : Prop
  /-- The trapezoid has a circumscribed circle -/
  has_circumscribed_circle : Prop

/-- The area of the special pentagon within the trapezoid -/
noncomputable def special_pentagon_area (t : SpecialTrapezoid) : ℝ :=
  3 * Real.sqrt 15 / 2

/-- Theorem stating the area of the special pentagon in the given trapezoid -/
theorem special_pentagon_area_theorem (t : SpecialTrapezoid) 
    (h1 : t.small_base = 3)
    (h2 : t.large_base = 5) :
    special_pentagon_area t = 3 * Real.sqrt 15 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pentagon_area_theorem_l1037_103788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1037_103778

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of first n terms
noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a8_pos : a 8 > 0)
  (h_a8_a9_neg : a 8 + a 9 < 0) :
  (∀ n > 15, S a n ≤ 0) ∧
  (∃ n ≤ 15, S a n > 0) ∧
  (∀ n ∈ Finset.range 15, S a 8 / a 8 ≥ S a n / a n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1037_103778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ganesh_average_speed_l1037_103771

/-- The overall average speed for a round trip given two different speeds -/
noncomputable def overall_average_speed (v1 v2 : ℝ) : ℝ := (2 * v1 * v2) / (v1 + v2)

/-- Theorem stating that the overall average speed for the given problem is approximately 37.97 km/hr -/
theorem ganesh_average_speed :
  let v1 : ℝ := 43  -- Speed from x to y
  let v2 : ℝ := 34  -- Speed from y to x
  abs (overall_average_speed v1 v2 - 37.97) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ganesh_average_speed_l1037_103771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_rules_2_and_4_are_functions_l1037_103787

-- Define the sets M and N
def M : Set ℤ := {-1, 1, 2, 3}
def N : Set ℤ := {0, 1, 2, 3, 4}

-- Define the correspondence rules
def rule1 (x : ℤ) : ℤ := x^2
def rule2 (x : ℤ) : ℤ := x + 1
noncomputable def rule3 (x : ℤ) : ℚ := (x + 3) / (2 * x - 1)
def rule4 (x : ℤ) : ℤ := (x - 1)^2

-- Define a predicate for a rule being a function from M to N
def is_function_M_to_N (f : ℤ → ℤ) : Prop :=
  ∀ x ∈ M, f x ∈ N

-- Define a predicate for rule3 (which outputs rationals)
def rule3_not_in_N : Prop :=
  ∀ x ∈ M, ∀ y ∈ N, rule3 x ≠ y

-- State the theorem
theorem only_rules_2_and_4_are_functions :
  is_function_M_to_N rule2 ∧ 
  is_function_M_to_N rule4 ∧ 
  ¬is_function_M_to_N rule1 ∧
  rule3_not_in_N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_rules_2_and_4_are_functions_l1037_103787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cell_culture_growth_l1037_103712

/-- Calculates the number of cells after a given number of splitting periods -/
def cell_growth (initial_cells : ℕ) (success_rate : ℚ) (splits : ℕ) : ℕ :=
  ((initial_cells : ℚ) * (2 * success_rate) ^ splits).floor.toNat

theorem cell_culture_growth :
  let initial_cells : ℕ := 5
  let success_rate : ℚ := 3/4
  let doubling_period : ℕ := 3
  let total_days : ℕ := 9
  let splits := total_days / doubling_period
  cell_growth initial_cells success_rate splits = 15 := by
  sorry

#eval cell_growth 5 (3/4) 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cell_culture_growth_l1037_103712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_side_length_l1037_103720

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def trapezium_area (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- Theorem stating the length of the second parallel side of the trapezium -/
theorem trapezium_side_length (t : Trapezium) 
    (h1 : t.side1 = 18)
    (h2 : t.height = 15)
    (h3 : t.area = 300)
    (h4 : trapezium_area t = t.area) :
  t.side2 = 22 := by
  sorry

#eval "Trapezium theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_side_length_l1037_103720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1037_103792

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := min (4 * x + 1) (min (x + 2) (-2 * x + 4))

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 8/3 := by
  -- We'll use x₀ = 2/3 as the point where f attains its maximum
  let x₀ : ℝ := 2/3
  let M : ℝ := 8/3

  -- Prove the three parts of the conjunction
  have h1 : ∀ x, f x ≤ M := by
    sorry -- Proof omitted
  
  have h2 : f x₀ = M := by
    sorry -- Proof omitted

  have h3 : M = 8/3 := by rfl

  -- Combine the proofs
  exact ⟨M, h1, ⟨x₀, h2⟩, h3⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1037_103792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_values_l1037_103763

/-- Represents a pyramid structure where each number is the product of the two above it -/
structure Pyramid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  h : ℝ
  i : ℝ
  j : ℝ
  g : ℝ
  h_eq : h = 16
  i_eq : i = 48
  j_eq : j = 72
  g_eq : g = 8
  d_eq : d = b * a
  e_eq1 : e = b * c
  e_eq2 : e = d * a
  f_eq : f = c * a
  h_prod : h = d * b
  i_prod : i = d * a
  j_prod : j = e * c
  g_prod : g = f * c

/-- The theorem stating the values of a, b, and c in the pyramid -/
theorem pyramid_values (p : Pyramid) : p.a = 3 ∧ p.b = 1 ∧ p.c = (3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_values_l1037_103763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infection_rate_is_seven_exceeds_500_after_three_rounds_l1037_103740

/-- The average number of healthy laying hens each infected laying hen infects in each round -/
def infection_rate : ℕ → ℕ := sorry

/-- The total number of infected laying hens after n rounds of infection -/
def total_infected : ℕ → ℕ := sorry

/-- Given conditions -/
axiom two_rounds_infected : total_infected 2 = 64

/-- Theorem 1: The infection rate is 7 -/
theorem infection_rate_is_seven : infection_rate 0 = 7 := by
  sorry

/-- Theorem 2: The number of infected laying hens after three rounds exceeds 500 -/
theorem exceeds_500_after_three_rounds : total_infected 3 > 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infection_rate_is_seven_exceeds_500_after_three_rounds_l1037_103740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1037_103730

/-- Definition of a geometric sequence -/
def IsGeometricSequence {α : Type*} [Field α] (s : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, s (n + 1) = s n * r

/-- Given that -1, a, b, c, -9 form a geometric sequence, prove that b = -3 and ac = 9 -/
theorem geometric_sequence_properties (a b c : ℝ) 
  (h : IsGeometricSequence (fun n ↦ match n with
    | 0 => -1
    | 1 => a
    | 2 => b
    | 3 => c
    | 4 => -9
    | _ => 0)) : 
  b = -3 ∧ a * c = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1037_103730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_satisfying_conditions_l1037_103742

/-- Two concentric circles with center P and radii r1 and r2 -/
structure ConcentricCircles (P : ℝ × ℝ) (r1 r2 : ℝ) :=
  (r2_gt_r1 : r2 > r1)

/-- Point on the larger circle -/
def PointOnLargerCircle (P : ℝ × ℝ) (r2 : ℝ) (B : ℝ × ℝ) : Prop :=
  (B.1 - P.1)^2 + (B.2 - P.2)^2 = r2^2

/-- Distance between two points -/
noncomputable def Distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- A is closer to B than any other point on C2 -/
def CloserToB (P : ℝ × ℝ) (r2 : ℝ) (A B : ℝ × ℝ) : Prop :=
  ∀ X : ℝ × ℝ, PointOnLargerCircle P r2 X → Distance A B ≤ Distance A X

/-- A is equidistant from P -/
def EquidistantFromP (P : ℝ × ℝ) (A : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, Distance A P = r

/-- Main theorem -/
theorem point_satisfying_conditions (P : ℝ × ℝ) (r1 r2 : ℝ) (C : ConcentricCircles P r1 r2) 
  (B : ℝ × ℝ) (h : PointOnLargerCircle P r2 B) :
  ∀ A : ℝ × ℝ, CloserToB P r2 A B → EquidistantFromP P A → A = B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_satisfying_conditions_l1037_103742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ladder_rungs_l1037_103734

/-- A function representing the possible positions of the monkey on the ladder. -/
def MonkeyPosition (n : ℕ) : ℕ → Prop :=
  λ pos ↦ pos ≤ n

/-- The monkey can move up 16 rungs or down 9 rungs. -/
def MonkeyMove (n : ℕ) (start finish : ℕ) : Prop :=
  (finish = start + 16 ∧ finish ≤ n) ∨ (finish = start - 9 ∧ 0 ≤ finish)

/-- A sequence of moves from the ground to the top and back to the ground. -/
def ValidClimb (n : ℕ) : Prop :=
  ∃ (seq : List ℕ),
    seq.head? = some 0 ∧
    seq.getLast? = some 0 ∧
    n ∈ seq ∧
    ∀ i, i + 1 < seq.length → MonkeyMove n (seq.get! i) (seq.get! (i + 1))

/-- The theorem stating that 24 is the minimum value of n for a valid climb. -/
theorem min_ladder_rungs :
  (ValidClimb 24) ∧ (∀ m < 24, ¬ValidClimb m) := by
  sorry

#check min_ladder_rungs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ladder_rungs_l1037_103734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_to_arithmetic_sequence_l1037_103753

theorem geometric_to_arithmetic_sequence (a q : ℝ) : 
  (∃ d : ℝ, (a * q + 6 - a = d) ∧ (a * q^2 + 3 - (a * q + 6) = d) ∧ (a * q^3 - 96 - (a * q^2 + 3) = d)) →
  a = 1 ∧ q = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_to_arithmetic_sequence_l1037_103753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_l1037_103731

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x > 0}
def B : Set ℝ := {y | ∃ x, y = 2^x}

-- State the theorem
theorem set_equality : (Set.univ \ B) ∪ A = Set.Iic 1 ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_l1037_103731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_cubic_sine_function_l1037_103758

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The domain of f is [n, n+6] -/
def DomainInterval (f : ℝ → ℝ) (n : ℝ) : Prop :=
  ∀ x, f x ≠ 0 → n ≤ x ∧ x ≤ n + 6

theorem odd_cubic_sine_function (a b m n : ℝ) :
  let f := fun x ↦ a * x^3 + b * Real.sin x + m - 3
  IsOdd f ∧ DomainInterval f n → m + n = 8/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_cubic_sine_function_l1037_103758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_probability_l1037_103732

/-- A circle with evenly spaced points -/
structure CircleWithPoints where
  numPoints : ℕ
  points : Fin numPoints → Point

/-- Four points selected from the circle -/
structure SelectedPoints (c : CircleWithPoints) where
  p : Fin c.numPoints
  q : Fin c.numPoints
  r : Fin c.numPoints
  s : Fin c.numPoints
  distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

/-- Predicate to check if two chords intersect -/
def chordsIntersect (c : CircleWithPoints) (sp : SelectedPoints c) : Prop :=
  sorry

/-- The probability of two chords intersecting -/
noncomputable def intersectionProbability (c : CircleWithPoints) : ℚ :=
  sorry

/-- Theorem stating that the probability of chord intersection is 1/3 -/
theorem chord_intersection_probability :
  ∀ (c : CircleWithPoints),
  c.numPoints = 2023 →
  (∃ (points : Finset (Fin c.numPoints)), points.card = 11) →
  intersectionProbability c = 1/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_probability_l1037_103732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1037_103751

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_points : distance (0, 12) (9, 0) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1037_103751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_window_area_ratio_l1037_103725

/-- The ratio of the total width to the width of the semicircle -/
def total_to_semicircle_ratio : ℚ := 5 / 4

/-- The width of the semicircle in inches -/
noncomputable def semicircle_width : ℝ := 40

/-- The radius of the semicircle -/
noncomputable def semicircle_radius : ℝ := semicircle_width / 2

/-- The total width of the window -/
noncomputable def total_width : ℝ := (total_to_semicircle_ratio : ℝ) * semicircle_width

/-- The combined width of the two smaller rectangles -/
noncomputable def small_rectangles_width : ℝ := total_width - semicircle_width

/-- The width of each smaller rectangle -/
noncomputable def small_rectangle_width : ℝ := small_rectangles_width / 2

/-- The height of the smaller rectangles (assumed to be equal to the semicircle radius) -/
noncomputable def small_rectangle_height : ℝ := semicircle_radius

/-- The area of the semicircle -/
noncomputable def semicircle_area : ℝ := Real.pi * semicircle_radius^2 / 2

/-- The total area of the two smaller rectangles -/
noncomputable def small_rectangles_area : ℝ := 2 * small_rectangle_width * small_rectangle_height

/-- The theorem stating that the ratio of the total area of the two smaller rectangles
    to the area of the semicircle is 1/π -/
theorem window_area_ratio :
  small_rectangles_area / semicircle_area = 1 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_window_area_ratio_l1037_103725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_is_essential_singularity_of_f_l1037_103736

open Complex

-- Define the function f(z) = e^(1/z^2)
noncomputable def f (z : ℂ) : ℂ := Complex.exp (1 / z^2)

-- Define what it means for a point to be an essential singularity
def is_essential_singularity (f : ℂ → ℂ) (z₀ : ℂ) : Prop :=
  ∀ w : ℂ, ∃ (U : Set ℂ), IsOpen U ∧ z₀ ∉ U ∧
    ∀ ε > 0, ∃ z ∈ U, Complex.abs (f z - w) < ε

-- Theorem statement
theorem zero_is_essential_singularity_of_f :
  is_essential_singularity f 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_is_essential_singularity_of_f_l1037_103736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_at_5_l1037_103750

/-- A polynomial of degree at most 3 satisfying specific conditions -/
noncomputable def P : ℝ → ℝ := sorry

/-- P is a polynomial of degree at most 3 -/
axiom P_degree : ∃ (a b c d : ℝ), ∀ x, P x = a * x^3 + b * x^2 + c * x + d

/-- P satisfies the given conditions for x = 1, 2, 3, 4 -/
axiom P_conditions : ∀ x ∈ ({1, 2, 3, 4} : Set ℝ), P x = 1 / (1 + x + x^2)

/-- The main theorem: P(5) = -3/91 -/
theorem P_at_5 : P 5 = -3/91 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_at_5_l1037_103750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_eight_l1037_103707

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The common ratio of the geometric sequence -/
def q : ℝ := sorry

/-- The first term of the geometric sequence -/
def a₁ : ℝ := sorry

theorem geometric_sequence_sum_eight (h1 : S 4 = -5) (h2 : S 6 = 21 * S 2) 
  (h3 : ∀ n : ℕ, S n = a₁ * (1 - q^n) / (1 - q)) (h4 : q ≠ 1) : 
  S 8 = -85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_eight_l1037_103707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cos_symmetric_interval_l1037_103786

theorem integral_cos_symmetric_interval (a : ℝ) : 
  (0 ≤ a) ∧ (a < 2 * Real.pi) ∧ 
  (∃ (x y : ℝ), x = 1 ∧ y = Real.sqrt 3 ∧ x * Real.cos a = x ∧ y * Real.sin a = y) →
  ∫ (x : ℝ) in -a..a, Real.cos x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cos_symmetric_interval_l1037_103786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_l1037_103746

theorem trigonometric_expression (x y : ℝ) 
  (h1 : Real.sin x / Real.sin y = 4)
  (h2 : Real.cos x / Real.cos y = -1/3) :
  Real.sin (2*x) / Real.sin (2*y) + (Real.cos (2*x) + Real.cos (2*y)) / Real.cos (2*y) = 158/381 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_l1037_103746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_radius_theorem_l1037_103724

/-- The radius of a circle that is internally tangent to four externally tangent circles of radius 2 -/
noncomputable def large_circle_radius : ℝ := 2 + 2 * Real.sqrt 2

/-- Four circles of radius 2 that are externally tangent to each other -/
structure FourTangentCircles :=
  (radius : ℝ)
  (tangent : Bool)

/-- A larger circle that is internally tangent to the four smaller circles -/
structure LargeCircle :=
  (radius : ℝ)
  (internally_tangent : Bool)

/-- Theorem stating that the radius of the large circle is 2 + 2√2 -/
theorem large_circle_radius_theorem (small_circles : FourTangentCircles) 
  (large_circle : LargeCircle) : 
  small_circles.radius = 2 ∧ 
  small_circles.tangent = true ∧ 
  large_circle.internally_tangent = true →
  large_circle.radius = large_circle_radius :=
by
  sorry

#check large_circle_radius_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_radius_theorem_l1037_103724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_problem_l1037_103741

noncomputable section

-- Define the semicircle C
def C (θ : Real) : Real × Real := (Real.cos θ, Real.sin θ)

-- Define point P on the semicircle
def P (θ : Real) : Real × Real := C θ

-- Define point A
def A : Real × Real := (1, 0)

-- Define point O (origin)
def O : Real × Real := (0, 0)

-- Define point M on ray OP
def M : Real × Real := (Real.pi/6, Real.sqrt 3 * Real.pi/6)

-- Define the length of OM
def OM_length : Real := Real.pi/3

-- Define the length of arc AP
def AP_arc_length : Real := Real.pi/3

-- Theorem statement
theorem semicircle_problem (θ : Real) (h1 : 0 ≤ θ ∧ θ ≤ Real.pi) :
  -- Polar coordinates of M
  (OM_length = Real.pi/3 ∧ Real.sqrt (M.1^2 + M.2^2) = Real.pi/3) ∧
  -- Parametric equation of line AM
  (∃ t : Real, A.1 + (M.1 - A.1) * t = 1 + (Real.pi/6 - 1) * t ∧
            A.2 + (M.2 - A.2) * t = (Real.sqrt 3 * Real.pi / 6) * t) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_problem_l1037_103741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_distance_l1037_103767

/-- Given an angle α with its vertex at the origin, its initial side on the positive x-axis,
    points A(1,a) and B(2,b) on its terminal side, and cos(2α) = 2/3, then |a-b| = √5/5. -/
theorem angle_terminal_side_distance (α : ℝ) (a b : ℝ) 
    (h1 : Real.cos (2 * α) = 2/3)
    (h2 : ∃ (t : ℝ), a = t ∧ b = 2*t) : 
    |a - b| = Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_distance_l1037_103767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_positive_exponential_l1037_103717

theorem negation_of_all_positive_exponential :
  (¬ (∀ x : ℝ, (3 : ℝ)^x > 0)) ↔ (∃ x : ℝ, (3 : ℝ)^x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_positive_exponential_l1037_103717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1037_103718

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x * abs (x - 2)
  else if x < 0 then x * (x + 2)
  else 0  -- Define f(0) = 0 to make it a total function

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is an odd function
  f (-3) = -3 ∧           -- f(-3) = -3
  ∀ x, x < 0 → f x = x * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1037_103718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_radius_squared_l1037_103745

/-- The radius squared of the circle containing all intersection points of two parabolas -/
theorem intersection_circle_radius_squared 
  (x y : ℝ) : 
  (y = (x - 2)^2) ∧ (x + 1 = (y + 2)^2) → 
  (x - 1)^2 + (y + 1)^2 = 1 := by
  intro h
  sorry

#check intersection_circle_radius_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_radius_squared_l1037_103745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_advantageous_discount_l1037_103791

def effective_discount_1 : ℚ := 1 - (1 - 0.2) * (1 - 0.2)
def effective_discount_2 : ℚ := 1 - (1 - 0.08) * (1 - 0.08) * (1 - 0.08)
def effective_discount_3 : ℚ := 1 - (1 - 0.3) * (1 - 0.1)

def is_more_advantageous (n : ℕ) : Prop :=
  (n : ℚ) / 100 > effective_discount_1 ∧
  (n : ℚ) / 100 > effective_discount_2 ∧
  (n : ℚ) / 100 > effective_discount_3

theorem smallest_advantageous_discount :
  ∃ (n : ℕ), n = 38 ∧ is_more_advantageous n ∧ ∀ m, m < n → ¬is_more_advantageous m := by
  sorry

#eval effective_discount_1
#eval effective_discount_2
#eval effective_discount_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_advantageous_discount_l1037_103791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1037_103766

-- Define the equation of the circle
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 2*m^2*y + m^4 + 2*m^2 - m = 0

-- Define the radius of the circle as a function of m
noncomputable def radius (m : ℝ) : ℝ :=
  Real.sqrt (-m^2 + m)

-- Theorem statement
theorem circle_properties :
  (∀ x y m, circle_equation x y m → (0 < m ∧ m < 1)) ∧
  (∃ m, ∀ m', radius m ≥ radius m') ∧
  (∃ m, radius m = 1/2) := by
  sorry

#check circle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1037_103766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1037_103775

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.log 0.3 / Real.log 0.4)
  (hb : b = (0.3 : ℝ)^(0.4 : ℝ))
  (hc : c = (0.4 : ℝ)^(0.3 : ℝ)) :
  a > c ∧ c > b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1037_103775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_after_x_months_l1037_103727

/-- The price of a phone after a certain number of months, given its original price and monthly depreciation rate. -/
noncomputable def phone_price (a : ℝ) (p : ℝ) (x : ℝ) : ℝ :=
  a * (1 - p/100)^x

/-- Theorem stating the price of a phone after x months -/
theorem price_after_x_months
  (a : ℝ) (p : ℝ) (m : ℝ) (x : ℝ)
  (h1 : a > 0) (h2 : 0 < p ∧ p < 100) (h3 : m > 0) (h4 : 0 ≤ x ∧ x ≤ m) :
  ∃ y : ℝ, y = phone_price a p x ∧ y > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_after_x_months_l1037_103727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parameter_l1037_103768

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 + 4*x = 0

-- Define the line
def my_line (a x y : ℝ) : Prop := (1+a)*x + y - 1 = 0

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ (x y : ℝ), my_circle x y ∧ my_line a x y ∧
  ∀ (x' y' : ℝ), my_circle x' y' → my_line a x' y' → (x = x' ∧ y = y')

-- Theorem statement
theorem tangent_line_parameter :
  ∀ a : ℝ, is_tangent a → a = -1/4 :=
by
  sorry

#check tangent_line_parameter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parameter_l1037_103768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qj_length_is_sqrt_11_599_l1037_103783

/-- Triangle PQR with given side lengths and incenter J -/
structure TrianglePQR where
  /-- Side length PQ -/
  pq : ℝ
  /-- Side length PR -/
  pr : ℝ
  /-- Side length QR -/
  qr : ℝ
  /-- Incenter of the triangle -/
  j : ℝ × ℝ
  /-- Condition: PQ = 11 -/
  h_pq : pq = 11
  /-- Condition: PR = 17 -/
  h_pr : pr = 17
  /-- Condition: QR = 10 -/
  h_qr : qr = 10

/-- The length of QJ in the given triangle -/
noncomputable def qj_length (t : TrianglePQR) : ℝ :=
  Real.sqrt 11.599

/-- Theorem: The length of QJ is equal to √11.599 -/
theorem qj_length_is_sqrt_11_599 (t : TrianglePQR) :
  qj_length t = Real.sqrt 11.599 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_qj_length_is_sqrt_11_599_l1037_103783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_percentage_all_amenities_l1037_103716

def wireless_internet_percentage : ℝ := 40
def free_snacks_percentage : ℝ := 70
def in_flight_entertainment_percentage : ℝ := 60
def premium_seating_percentage : ℝ := 50

theorem max_percentage_all_amenities :
  let percentages := [wireless_internet_percentage, free_snacks_percentage,
                      in_flight_entertainment_percentage, premium_seating_percentage]
  (∀ p ∈ percentages, 0 ≤ p ∧ p ≤ 100) →
  ∃ max_percentage : ℝ,
    max_percentage = (List.minimum percentages).getD 0 ∧
    max_percentage ≤ wireless_internet_percentage ∧
    max_percentage ≤ free_snacks_percentage ∧
    max_percentage ≤ in_flight_entertainment_percentage ∧
    max_percentage ≤ premium_seating_percentage :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_percentage_all_amenities_l1037_103716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_min_area_PRN_l1037_103794

noncomputable section

-- Define the points and vectors
def F : ℝ × ℝ := (1/2, 0)
def A : ℝ → ℝ × ℝ := λ x => (x, 0)
def B : ℝ → ℝ × ℝ := λ y => (0, y)
def M : ℝ × ℝ → ℝ × ℝ := λ p => p

-- Define the conditions
def condition_AM_2AB (x y : ℝ) : Prop :=
  M (x, y) - A (-x) = 2 • (B (y/2) - A (-x))

def condition_BA_BF_orthogonal (x y : ℝ) : Prop :=
  let ba := A (-x) - B (y/2)
  let bf := F - B (y/2)
  ba.1 * bf.1 + ba.2 * bf.2 = 0

-- Define the trajectory E
def E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ condition_AM_2AB x y ∧ condition_BA_BF_orthogonal x y}

-- Define the circle
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1}

-- Define the area of triangle PRN
def area_PRN (P : ℝ × ℝ) (R N : ℝ) : ℝ :=
  1/2 * P.1 * |R - N|

-- Define line segment
def LineSegment (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B}

-- Theorem statements
theorem trajectory_equation :
  E = {p : ℝ × ℝ | p.2^2 = 2 * p.1} := by
  sorry

theorem min_area_PRN :
  ∃ P ∈ E, ∀ R N : ℝ, R ≠ N →
    (∃ Q ∈ Circle, Q ∈ LineSegment P (0, R) ∨ Q ∈ LineSegment P (0, N) ∨ Q ∈ LineSegment (0, R) (0, N)) →
    area_PRN P R N ≥ 8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_min_area_PRN_l1037_103794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1037_103785

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance from the center to a focus of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2)

/-- Theorem: For a hyperbola with equation x^2/a^2 - y^2/b^2 = 1, 
    eccentricity 2, and a focal point at (4, 0), 
    the values of a and b are 2 and 2√3 respectively -/
theorem hyperbola_properties :
  ∀ h : Hyperbola, 
    eccentricity h = 2 → 
    focal_distance h = 4 → 
    h.a = 2 ∧ h.b = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1037_103785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_conditions_and_inequality_l1037_103760

noncomputable def f (n m : ℝ) (x : ℝ) : ℝ := (-2^x + n) / (2^(x+1) + m)

theorem odd_function_conditions_and_inequality 
  (n m k : ℝ)
  (h_odd : ∀ x, f n m (-x) = -(f n m x)) 
  (h_ineq : ∀ c ∈ Set.Ioo (-1 : ℝ) 1, f n m (4^c - 2^(c+1)) + f n m (2 * 4^c - k) < 0) :
  (n = 1 ∧ m = 2) ∧ k ≤ -1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_conditions_and_inequality_l1037_103760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_sum_l1037_103743

theorem quadratic_solution_sum (x y : ℝ) (p q r s : ℕ+) :
  x + y = 6 →
  x * y = 6/5 →
  x = (p : ℝ) + (q : ℝ) * Real.sqrt (r : ℝ) / (s : ℝ) ∨
  x = (p : ℝ) - (q : ℝ) * Real.sqrt (r : ℝ) / (s : ℝ) →
  p + q + r + s = 236 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_sum_l1037_103743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_race_result_l1037_103761

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ

/-- Represents a race between two runners -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner

/-- Calculates the distance covered by a runner in a given time -/
def distance_covered (runner : Runner) (time : ℝ) : ℝ :=
  runner.speed * time

/-- Theorem stating the result of the second race -/
theorem second_race_result (first_race : Race) (h1 : first_race.distance = 100) 
    (h2 : distance_covered first_race.runner_a first_race.distance = first_race.distance) 
    (h3 : distance_covered first_race.runner_b first_race.distance = first_race.distance - 10) :
  ∃ (second_race : Race), 
    second_race.distance = 100 ∧ 
    second_race.runner_a = first_race.runner_a ∧ 
    second_race.runner_b = first_race.runner_b ∧ 
    distance_covered second_race.runner_a second_race.distance = 
      distance_covered second_race.runner_b second_race.distance + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_race_result_l1037_103761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_composition_l1037_103772

noncomputable def f (x : ℝ) : ℝ := x + 3

noncomputable def g (x : ℝ) : ℝ := x / 4

theorem compute_composition :
  f (g⁻¹ (f⁻¹ (f⁻¹ (g (f 23))))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_composition_l1037_103772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cobbler_hours_worked_l1037_103738

/-- The cost of making the mold for the shoes -/
noncomputable def mold_cost : ℚ := 250

/-- The cobbler's hourly rate before discount -/
noncomputable def hourly_rate : ℚ := 75

/-- The discount factor applied to the cobbler's work -/
noncomputable def discount_factor : ℚ := 4/5

/-- The total amount Bobby paid -/
noncomputable def total_paid : ℚ := 730

/-- The number of hours the cobbler worked on the shoes -/
noncomputable def hours_worked : ℚ := (total_paid - mold_cost) / (hourly_rate * discount_factor)

theorem cobbler_hours_worked :
  hours_worked = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cobbler_hours_worked_l1037_103738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aisha_age_l1037_103762

/-- Proves that Aisha is 8 years old given the conditions in the problem -/
theorem aisha_age :
  ∀ (ali_age yusaf_age umar_age aisha_age : ℕ),
  ali_age = 8 →
  ali_age = yusaf_age + 3 →
  umar_age = 2 * yusaf_age →
  (aisha_age : ℚ) + (1/2 : ℚ) = ((ali_age : ℚ) + (umar_age : ℚ)) / 2 →
  aisha_age = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aisha_age_l1037_103762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_win_amount_l1037_103764

theorem lottery_win_amount (tax_rate : ℝ) (processing_fee : ℝ) (take_home : ℝ) (win_amount : ℝ) : 
  tax_rate = 0.20 →
  processing_fee = 5 →
  take_home = 35 →
  win_amount - tax_rate * win_amount - processing_fee = take_home →
  win_amount = 50 := by
  sorry

#check lottery_win_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_win_amount_l1037_103764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_available_storage_l1037_103703

/-- Represents the storage space problem in the warehouse --/
structure Warehouse :=
  (second_floor : ℝ)

/-- Calculates the total storage space in the warehouse --/
noncomputable def total_storage (w : Warehouse) : ℝ :=
  2 * w.second_floor + w.second_floor + w.second_floor / 2 + 3 * (w.second_floor / 2)

/-- Calculates the filled storage space in the warehouse --/
noncomputable def filled_storage (w : Warehouse) : ℝ :=
  w.second_floor / 4 + (w.second_floor / 2) / 3 + (3 * (w.second_floor / 2)) / 5

/-- Theorem stating the available storage space in the warehouse --/
theorem available_storage (w : Warehouse) 
  (h : w.second_floor / 4 = 5000) : 
  ∃ (available : ℝ), abs (available - (total_storage w - filled_storage w)) < 0.01 ∧ 
                     abs (available - 85666.67) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_available_storage_l1037_103703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1037_103704

-- Define the slopes of two lines
noncomputable def slope1 (a : ℝ) := -1 / a
noncomputable def slope2 : ℝ := -2 / 3

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines (a : ℝ) :
  perpendicular (slope1 a) slope2 → a = -2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1037_103704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1037_103796

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

noncomputable def sum_of_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => a i)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_sum : ∀ n, S n = sum_of_terms a n)
  (h_a3 : a 3 = 2 * S 2 + 5)
  (h_a4 : a 4 = 2 * S 3 + 5) :
  ∃ q : ℝ, q = 3 ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
by
  sorry

#check geometric_sequence_common_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1037_103796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_not_equal_to_given_set_l1037_103755

def A : Set ℤ := {0, 1, 2}

def f (x : ℤ) : ℤ := (x - 1)^2

def B : Set ℤ := f '' A

theorem B_not_equal_to_given_set : B ≠ {0, -1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_not_equal_to_given_set_l1037_103755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1037_103774

-- Define the common perimeter
variable (P : ℝ) (P_pos : P > 0)

-- Define the side lengths
noncomputable def square_side : ℝ := P / 4
noncomputable def hexagon_side : ℝ := P / 6

-- Define the radii of circumscribed circles
noncomputable def square_circle_radius : ℝ := (P * Real.sqrt 2) / 8
noncomputable def hexagon_circle_radius : ℝ := P / 6

-- Define the areas of circumscribed circles
noncomputable def A : ℝ := Real.pi * (square_circle_radius P) ^ 2
noncomputable def C : ℝ := Real.pi * (hexagon_circle_radius P) ^ 2

-- The theorem to prove
theorem circle_area_ratio :
  A P / C P = 9 / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1037_103774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1037_103705

/-- The equation of a parabola -/
noncomputable def parabola_equation (x : ℝ) : ℝ := (x^2 - 10*x + 21) / 14

/-- The equation of the directrix -/
noncomputable def directrix : ℝ := -53/14

/-- Theorem stating that the given directrix is correct for the parabola -/
theorem parabola_directrix : 
  ∀ x y : ℝ, y = parabola_equation x → 
  ∃ p : ℝ, p > 0 ∧ 
  ∃ h k : ℝ, y = (1/(4*p)) * (x - h)^2 + k ∧
  directrix = k - p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1037_103705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_x_coordinate_l1037_103706

/-- 
Given two lines y = ax + 4 and y = 3x + b, where a + b = 9,
prove that their point of intersection has x-coordinate equal to 1.
-/
theorem intersection_x_coordinate (a b : ℝ) (h : a + b = 9) :
  ∃ x : ℝ, a * x + 4 = 3 * x + b ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_x_coordinate_l1037_103706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l1037_103702

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 8*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- The origin -/
def origin : ℝ × ℝ := (0, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

theorem parabola_triangle_area (P : ParabolaPoint) 
  (h : distance (P.x, P.y) focus = 4) : 
  triangleArea (P.x, P.y) focus origin = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l1037_103702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pyramid_volume_l1037_103735

/-- Represents a pyramid with a square base and height intersecting the diagonal of the base -/
structure SquareBasePyramid where
  base_side : ℝ
  height : ℝ

/-- The perimeter of the diagonal section containing the height of the pyramid -/
noncomputable def diagonal_section_perimeter (p : SquareBasePyramid) : ℝ :=
  p.base_side * Real.sqrt 2 + 2 * Real.sqrt (p.height^2 + (p.base_side / 2)^2)

/-- The volume of the pyramid -/
noncomputable def pyramid_volume (p : SquareBasePyramid) : ℝ :=
  (1 / 3) * p.base_side^2 * p.height

/-- The theorem stating the maximum volume of the pyramid under given conditions -/
theorem max_pyramid_volume :
  ∃ (p : SquareBasePyramid),
    diagonal_section_perimeter p = 5 ∧
    ∀ (q : SquareBasePyramid),
      diagonal_section_perimeter q = 5 →
      pyramid_volume q ≤ pyramid_volume p ∧
      pyramid_volume p = Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pyramid_volume_l1037_103735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_squared_greater_than_two_a_minus_one_l1037_103769

theorem a_squared_greater_than_two_a_minus_one
  (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  a^2 > 2*a - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_squared_greater_than_two_a_minus_one_l1037_103769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_undefined_value_l1037_103776

theorem largest_undefined_value : 
  let f (x : ℝ) := (x + 2) / (4 * x^3 - 40 * x^2 + 36 * x - 8)
  ∀ y : ℝ, (∃ x : ℝ, f x = 0⁻¹) → y ≤ 4 + Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_undefined_value_l1037_103776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_pascal_row_15_l1037_103744

/-- Pascal's triangle coefficient -/
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The 15th row of Pascal's triangle -/
def pascal_row_15 : List ℕ :=
  List.map (binomial 15) (List.range 16)

theorem fifth_number_pascal_row_15 : 
  pascal_row_15[4] = 1365 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_pascal_row_15_l1037_103744
