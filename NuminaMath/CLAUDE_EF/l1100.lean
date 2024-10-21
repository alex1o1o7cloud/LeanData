import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2019_value_l1100_110026

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => (sequenceA n - 1) / (sequenceA n + 1)

theorem sequence_2019_value : sequenceA 2018 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2019_value_l1100_110026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_second_side_l1100_110014

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def trapeziumArea (t : Trapezium) : ℝ :=
  (1 / 2) * (t.side1 + t.side2) * t.height

/-- Theorem: The second parallel side of the trapezium is 14 cm -/
theorem trapezium_second_side (t : Trapezium) 
    (h1 : t.side1 = 24)
    (h2 : t.height = 18)
    (h3 : t.area = 342)
    (h4 : t.area = trapeziumArea t) : 
  t.side2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_second_side_l1100_110014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l1100_110029

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 2021
  | (n + 1) => Real.sqrt (4 + sequence_a n)

theorem sequence_problem (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  sequence_a 5 = Real.sqrt ((m + Real.sqrt n : ℝ) / 2) + Real.sqrt ((m - Real.sqrt n : ℝ) / 2) →
  10 * m + n = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l1100_110029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l1100_110005

/-- The plane equation: 2x - 3y + 4z = 40 -/
def plane_equation (p : ℝ × ℝ × ℝ) : Prop :=
  2 * p.1 - 3 * p.2.1 + 4 * p.2.2 = 40

/-- The given point A -/
def point_A : ℝ × ℝ × ℝ := (2, 3, 1)

/-- The closest point P on the plane to point A -/
noncomputable def point_P : ℝ × ℝ × ℝ := (92/29, 16/29, 145/29)

/-- Theorem stating that point_P is the closest point on the plane to point_A -/
theorem closest_point_on_plane :
  plane_equation point_P ∧
  ∀ q : ℝ × ℝ × ℝ, plane_equation q →
    ‖(point_P.1 - point_A.1, point_P.2.1 - point_A.2.1, point_P.2.2 - point_A.2.2)‖ ≤ 
    ‖(q.1 - point_A.1, q.2.1 - point_A.2.1, q.2.2 - point_A.2.2)‖ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l1100_110005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2008_value_l1100_110006

def sequence_a (n : ℕ) : ℕ := 
  let k := (n - 1) / 40
  let i := (n - 1) % 40 + 1
  75 * k + (if i = 1 then 1
            else if i = 2 then 2
            else if i = 3 then 4
            else if i = 4 then 7
            else if i = 5 then 8
            else if i = 8 then 14
            else 0)  -- We only define the necessary terms for this problem

def is_coprime_to_75 (n : ℕ) : Prop :=
  Nat.Coprime n 75

theorem a_2008_value :
  (∀ n m : ℕ, n < m → sequence_a n < sequence_a m) →
  (∀ n : ℕ, is_coprime_to_75 (sequence_a n)) →
  (∀ k : ℕ, k > 0 → is_coprime_to_75 k → ∃ n : ℕ, sequence_a n = k) →
  sequence_a 2008 = 3764 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2008_value_l1100_110006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_expression_l1100_110066

theorem simplify_sqrt_expression : Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_expression_l1100_110066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_speed_calculation_l1100_110024

/-- The hiker's speed in miles per hour -/
noncomputable def hikerSpeed : ℝ := 7.5

/-- The motor-cyclist's speed in miles per hour -/
noncomputable def motorCyclistSpeed : ℝ := 30

/-- The time in hours the motor-cyclist travels before stopping -/
noncomputable def travelTime : ℝ := 12 / 60

/-- The time in hours the motor-cyclist waits for the hiker -/
noncomputable def waitTime : ℝ := 48 / 60

theorem hiker_speed_calculation :
  hikerSpeed * waitTime = motorCyclistSpeed * travelTime := by
  sorry

#check hiker_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_speed_calculation_l1100_110024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_containers_l1100_110096

theorem orange_containers (total_containers min_oranges max_oranges : ℕ) :
  total_containers = 150 →
  min_oranges = 130 →
  max_oranges = 160 →
  ∃ (n : ℕ), n = 5 ∧
    (∀ (m : ℕ), m > n → 
      ∃ (f : ℕ → ℕ), 
        (∀ i, i < total_containers → min_oranges ≤ f i ∧ f i ≤ max_oranges) ∧
        (∀ k, min_oranges ≤ k ∧ k ≤ max_oranges → 
          (Finset.filter (λ i ↦ f i = k) (Finset.range total_containers)).card < m)) ∧
    (∃ (f : ℕ → ℕ),
      (∀ i, i < total_containers → min_oranges ≤ f i ∧ f i ≤ max_oranges) ∧
      ∃ k, min_oranges ≤ k ∧ k ≤ max_oranges ∧
        (Finset.filter (λ i ↦ f i = k) (Finset.range total_containers)).card ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_containers_l1100_110096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_even_odd_sum_l1100_110031

theorem max_value_of_even_odd_sum (m n : ℕ) 
  (h_even : ∃ (a : Fin m → ℕ), (∀ i, Even (a i)) ∧ (∀ i j, i ≠ j → a i ≠ a j))
  (h_odd : ∃ (b : Fin n → ℕ), (∀ i, Odd (b i)) ∧ (∀ i j, i ≠ j → b i ≠ b j))
  (h_sum : ∃ (a : Fin m → ℕ) (b : Fin n → ℕ), 
    (∀ i, Even (a i)) ∧ (∀ i, Odd (b i)) ∧
    (∀ i j, i ≠ j → a i ≠ a j) ∧ (∀ i j, i ≠ j → b i ≠ b j) ∧
    (Finset.sum (Finset.univ : Finset (Fin m)) a + Finset.sum (Finset.univ : Finset (Fin n)) b = 2015)) :
  (20 * m + 15 * n : ℕ) ≤ 1105 ∧ 
  ∃ (m₀ n₀ : ℕ), 20 * m₀ + 15 * n₀ = 1105 ∧
    ∃ (a : Fin m₀ → ℕ) (b : Fin n₀ → ℕ), 
      (∀ i, Even (a i)) ∧ (∀ i, Odd (b i)) ∧
      (∀ i j, i ≠ j → a i ≠ a j) ∧ (∀ i j, i ≠ j → b i ≠ b j) ∧
      (Finset.sum (Finset.univ : Finset (Fin m₀)) a + Finset.sum (Finset.univ : Finset (Fin n₀)) b = 2015) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_even_odd_sum_l1100_110031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l1100_110092

/-- Given a line l and a circle C in the Cartesian coordinate system:
    - Line l has parametric equation: x = -1 + (√3/2)t, y = (1/2)t
    - Circle C has polar equation: ρ = -4cosθ
    The distance from the center of circle C to line l is 1/2 -/
theorem distance_from_circle_center_to_line (t : ℝ) (θ : ℝ) : 
  let l : ℝ × ℝ → Prop := λ (x, y) ↦ ∃ t, x = -1 + (Real.sqrt 3 / 2) * t ∧ y = (1 / 2) * t
  let C : ℝ × ℝ → Prop := λ (x, y) ↦ Real.sqrt (x^2 + y^2) = -4 * Real.cos θ
  let center : ℝ × ℝ := (-2, 0)
  let distance := λ (p : ℝ × ℝ) (l : ℝ × ℝ → Prop) ↦ 
    |p.1 + 1 - Real.sqrt 3 * p.2| / Real.sqrt 4
  distance center l = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l1100_110092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_36_values_l1100_110011

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 4

noncomputable def g (y : ℝ) : ℝ := 
  let x := Real.sqrt (y + 4) / 2
  x^2 - x + 2

-- Theorem statement
theorem sum_of_g_36_values : 
  ∃ (a b : ℝ), g 36 = a ∧ g 36 = b ∧ a + b = 24 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_36_values_l1100_110011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_l1100_110015

-- Define the given points and line
def A : ℝ × ℝ := (-3, 5)
def B : ℝ × ℝ := (2, 12)
def l (x y : ℝ) : Prop := x - y - 3 = 0

-- Define the symmetric point of A with respect to line l
noncomputable def symmetric_point (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ × ℝ := 
  sorry

-- Define the equation of a line passing through two points
def line_equation (p1 p2 : ℝ × ℝ) : ℝ → ℝ → Prop := 
  λ x y => sorry

-- Theorem statement
theorem reflected_ray_equation :
  let A' := symmetric_point A l
  line_equation A' B = λ x y => x - 2*y + 22 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_l1100_110015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_monotone_intervals_triangle_side_ratio_range_l1100_110027

noncomputable def a (ω x : ℝ) : ℝ × ℝ := (Real.sin (ω * x), Real.cos (ω * x))

noncomputable def b (ω x : ℝ) : ℝ × ℝ := (Real.cos (ω * x), Real.sqrt 3 * Real.cos (ω * x))

noncomputable def f (ω x : ℝ) : ℝ := 
  let av := a ω x
  let bv := b ω x
  (av.1 * bv.1 + av.2 * bv.2) - Real.sqrt 3 / 2 * (av.1 * av.1 + av.2 * av.2)

theorem f_simplification (ω : ℝ) (hω : ω > 0) :
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f ω x = f ω (x + T) ∧ 
  (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f ω x = f ω (x + S)) → T ≤ S) ∧ T = π →
  ∀ x : ℝ, f ω x = Real.sin (2 * x + π / 3) :=
by sorry

theorem f_monotone_intervals (ω : ℝ) (hω : ω > 0) :
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f ω x = f ω (x + T) ∧ 
  (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f ω x = f ω (x + S)) → T ≤ S) ∧ T = π →
  ∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (k * π - 5 * π / 12) (k * π + π / 12)) :=
by sorry

theorem triangle_side_ratio_range (A B : ℝ) (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) 
  (hC : 0 < π - A - B ∧ π - A - B < π / 2) (ω : ℝ) (hω : ω > 0) :
  f ω (A / 2) = Real.sqrt 3 / 2 →
  Real.sqrt 3 / 2 < Real.sin A / Real.sin B ∧ Real.sin A / Real.sin B < Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_monotone_intervals_triangle_side_ratio_range_l1100_110027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1100_110086

/-- Given an ellipse where the major axis length is √2 times the minor axis length,
    the eccentricity is √2/2. -/
theorem ellipse_eccentricity (a b : ℝ) (h : 2*a = Real.sqrt 2 * (2*b)) :
  Real.sqrt (a^2 - b^2) / a = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1100_110086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_percentage_increase_correct_l1100_110020

/-- The total percentage increase of a population over three time periods -/
noncomputable def totalPercentageIncrease (a b c : ℝ) : ℝ :=
  a + b + c + (a * b + a * c + b * c) / 100 + a * b * c / 10000

/-- Theorem stating that the total percentage increase is correct -/
theorem total_percentage_increase_correct (a b c : ℝ) :
  let factor1 := 1 + a / 100
  let factor2 := 1 + b / 100
  let factor3 := 1 + c / 100
  (factor1 * factor2 * factor3 - 1) * 100 = totalPercentageIncrease a b c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_percentage_increase_correct_l1100_110020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l1100_110007

theorem smallest_sum_of_factors (a b : ℕ) (h : (3^6 * 5^3 * 7^2 : ℕ) = a^b) :
  ∃ (a' b' : ℕ), (3^6 * 5^3 * 7^2 : ℕ) = a'^b' ∧ a' + b' ≤ a + b ∧ a' + b' = 317 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l1100_110007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1100_110093

/-- Given an ellipse and a line passing through its focus and vertex, prove the eccentricity --/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let line := fun (x y : ℝ) => x + 2*y = 2
  let focus_vertex_on_line := ∃ (xf yf xv yv : ℝ),
    ellipse xf yf ∧ ellipse xv yv ∧
    line xf yf ∧ line xv yv ∧
    (xf^2 + yf^2 = (a^2 - b^2)) ∧
    (xv = 0 ∨ xv = a ∨ yv = 0 ∨ yv = b)
  focus_vertex_on_line →
  let e := Real.sqrt (1 - b^2/a^2)  -- eccentricity
  e = 2 * Real.sqrt 5 / 5 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1100_110093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kate_average_speed_l1100_110079

/-- Represents a segment of Kate's journey -/
structure Segment where
  duration : ℚ  -- in hours
  speed : ℚ     -- in mph

/-- Calculates the average speed given a list of journey segments -/
def averageSpeed (segments : List Segment) : ℚ :=
  let totalDistance := segments.foldl (fun acc s => acc + s.duration * s.speed) 0
  let totalTime := segments.foldl (fun acc s => acc + s.duration) 0
  totalDistance / totalTime

theorem kate_average_speed :
  let bicycle : Segment := { duration := 20/60, speed := 20 }
  let walk : Segment := { duration := 1, speed := 4 }
  let jog : Segment := { duration := 40/60, speed := 6 }
  let journey := [bicycle, walk, jog]
  averageSpeed journey = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kate_average_speed_l1100_110079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1100_110057

/-- Triangle ABC with side lengths a, b, c opposite angles A, B, C respectively -/
structure Triangle (a b c : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The area of a triangle given two sides and the angle between them -/
noncomputable def area (a b : ℝ) (C : ℝ) : ℝ := (1/2) * a * b * Real.sin C

/-- A triangle is obtuse if one of its angles is greater than π/2 -/
def isObtuse (A B C : ℝ) : Prop := A > Real.pi/2 ∨ B > Real.pi/2 ∨ C > Real.pi/2

theorem triangle_abc_properties (a : ℝ) :
  let b := a + 1
  let c := a + 2
  ∀ (A B C : ℝ), 
    Triangle a b c →
    2 * Real.sin C = 3 * Real.sin A →
    (area a b C = (15 * Real.sqrt 7) / 4) ∧
    (∃ (n : ℕ), n > 0 ∧ isObtuse A B C ∧ a = n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1100_110057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_l1100_110022

theorem fraction_meaningful (a : ℝ) : (a + 2 ≠ 0) ↔ a ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_l1100_110022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l1100_110061

-- Define the function f(x) = 1 - 2sin(πx/2)
noncomputable def f (x : ℝ) : ℝ := 1 - 2 * Real.sin (Real.pi / 2 * x)

-- State the theorem about the minimum and maximum values of f
theorem f_min_max :
  (∀ x : ℝ, f x ≥ -1) ∧ 
  (∃ x : ℝ, f x = -1) ∧
  (∀ x : ℝ, f x ≤ 3) ∧ 
  (∃ x : ℝ, f x = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l1100_110061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_expression_l1100_110042

/-- Given m < 0 and a point M(3m, -m) on the terminal side of angle α, 
    prove that 1 / (2*sin(α)*cos(α) + cos²(α)) = 10/3 -/
theorem angle_expression (m : ℝ) (α : ℝ) (h1 : m < 0) 
  (h2 : Real.sin α = Real.sqrt 10 / 10) (h3 : Real.cos α = -3 / Real.sqrt 10) : 
  1 / (2 * Real.sin α * Real.cos α + (Real.cos α) ^ 2) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_expression_l1100_110042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_inradius_ratio_l1100_110077

noncomputable section

-- Define a right triangle with leg lengths 6 and 8
def right_triangle (a b c : ℝ) : Prop :=
  a = 6 ∧ b = 8 ∧ c^2 = a^2 + b^2

-- Define the radius of the circumcircle
noncomputable def circumradius (c : ℝ) : ℝ := c / 2

-- Define the semiperimeter
noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Define the area of the triangle
noncomputable def area (a b : ℝ) : ℝ := (a * b) / 2

-- Define the radius of the incircle
noncomputable def inradius (a b c : ℝ) : ℝ := area a b / semiperimeter a b c

-- Theorem statement
theorem circumradius_inradius_ratio (a b c : ℝ) :
  right_triangle a b c →
  circumradius c / inradius a b c = 5 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_inradius_ratio_l1100_110077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_count_l1100_110043

/-- Represents a color in the problem -/
inductive Color
| Red
| White
| Blue
| Green

/-- Represents a position in the configuration -/
inductive Position
| LeftTop | LeftBottom | LeftRight
| MiddleTop | MiddleBottom | MiddleRight
| RightTop | RightBottom | RightRight

/-- Defines the adjacency relation between positions -/
def adjacent : Position → Position → Prop :=
  sorry

/-- A coloring is a function from positions to colors -/
def Coloring := Position → Color

/-- A valid coloring is one where adjacent positions have different colors -/
def isValidColoring (c : Coloring) : Prop :=
  ∀ p1 p2 : Position, adjacent p1 p2 → c p1 ≠ c p2

/-- Instance for Fintype Coloring -/
instance : Fintype Coloring :=
  sorry

/-- Instance for DecidablePred isValidColoring -/
instance : DecidablePred isValidColoring :=
  sorry

/-- The main theorem: there are exactly 864 valid colorings -/
theorem valid_colorings_count :
  (Finset.filter isValidColoring (Finset.univ : Finset Coloring)).card = 864 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_count_l1100_110043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reflection_theorem_l1100_110044

structure Triangle (P Q R : ℝ × ℝ) where
  -- Define a triangle with vertices P, Q, R

def median (P Q R M : ℝ × ℝ) : Prop :=
  M.1 = (Q.1 + R.1) / 2 ∧ M.2 = (Q.2 + R.2) / 2

def reflect (P M S T : ℝ × ℝ) : Prop :=
  T.1 - P.1 = P.1 - S.1 ∧ T.2 - P.2 = P.2 - S.2

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem triangle_reflection_theorem 
  (P Q R S T M : ℝ × ℝ) 
  (tri : Triangle P Q R) 
  (med : median P Q R M) 
  (refl : reflect P M S T) 
  (h1 : distance P S = 8) 
  (h2 : distance S R = 16) 
  (h3 : distance Q T = 12) : 
  distance P Q = 8 * Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reflection_theorem_l1100_110044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_x_positivity_l1100_110050

theorem negative_x_positivity (x : ℝ) (h : x < 0) : -x / |x| > 0 ∧ (2 : ℝ)^(-x) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_x_positivity_l1100_110050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1100_110004

noncomputable def f (x : ℝ) : ℝ := (Real.cos (2 * x) + Real.sin (2 * x)) / (Real.cos (2 * x) - Real.sin (2 * x))

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1100_110004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_box_diagonal_sum_l1100_110025

/-- Represents a rectangular box with side lengths x, y, and z. -/
structure RectangularBox where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the total surface area of a rectangular box. -/
def totalSurfaceArea (box : RectangularBox) : ℝ :=
  2 * (box.x * box.y + box.y * box.z + box.z * box.x)

/-- Calculates the sum of the lengths of all edges of a rectangular box. -/
def totalEdgeLength (box : RectangularBox) : ℝ :=
  4 * (box.x + box.y + box.z)

/-- Calculates the sum of the lengths of all interior diagonals of a rectangular box. -/
noncomputable def totalDiagonalLength (box : RectangularBox) : ℝ :=
  4 * Real.sqrt (box.x^2 + box.y^2 + box.z^2)

theorem rectangular_box_diagonal_sum (box : RectangularBox) 
  (h1 : totalSurfaceArea box = 94)
  (h2 : totalEdgeLength box = 48) :
  totalDiagonalLength box = 20 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_box_diagonal_sum_l1100_110025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skating_time_approx_l1100_110089

/-- Represents the problem of Alice skating on a highway stretch --/
structure SkatingProblem where
  highway_length : ℝ
  highway_width : ℝ
  car_spacing : ℝ
  quarter_circle_radius : ℝ
  skating_speed : ℝ

/-- Calculates the time taken for Alice to skate the entire stretch --/
noncomputable def time_taken (p : SkatingProblem) : ℝ :=
  let num_quarter_circles := p.highway_length / p.car_spacing
  let quarter_circle_distance := num_quarter_circles * (Real.pi / 2 * p.quarter_circle_radius)
  let straight_distance := p.highway_length - (num_quarter_circles * 2 * p.quarter_circle_radius)
  let total_distance := straight_distance + quarter_circle_distance
  let distance_miles := total_distance / 5280
  distance_miles / p.skating_speed

/-- The main theorem stating that the time taken is approximately 0.1362 hours --/
theorem skating_time_approx (p : SkatingProblem) 
    (h1 : p.highway_length = 3000)
    (h2 : p.highway_width = 60)
    (h3 : p.car_spacing = 100)
    (h4 : p.quarter_circle_radius = 10)
    (h5 : p.skating_speed = 4) :
    ∃ ε > 0, |time_taken p - 0.1362| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skating_time_approx_l1100_110089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l1100_110067

/-- The distance to the destination given boat speed, stream speed, and round trip time -/
noncomputable def distance_to_destination (boat_speed stream_speed : ℝ) (round_trip_time : ℝ) : ℝ :=
  (round_trip_time * boat_speed * (boat_speed - stream_speed)) / (2 * boat_speed - stream_speed)

/-- Theorem stating that the calculated distance is approximately 7392.92 km -/
theorem distance_calculation (ε : ℝ) (hε : ε > 0) :
  ∃ (d : ℝ), abs (d - distance_to_destination 16 2 937.1428571428571) < ε ∧ abs (d - 7392.92) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l1100_110067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1100_110013

-- Define constants
noncomputable def jogger_speed : ℝ := 8000 / 3600  -- 8 kmph in m/s
noncomputable def train_speed : ℝ := 55000 / 3600  -- 55 kmph in m/s
def train_length : ℝ := 130  -- meters
def initial_distance : ℝ := 340  -- meters

-- Define the theorem
theorem train_passing_time :
  let relative_speed := train_speed - jogger_speed
  let total_distance := initial_distance + train_length
  let passing_time := total_distance / relative_speed
  ∃ (ε : ℝ), ε > 0 ∧ |passing_time - 36| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1100_110013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_to_exp_is_log_minus_one_l1100_110064

/-- A function is symmetric to e^(x+1) with respect to y = x if it's the inverse of e^(x+1) -/
def SymmetricToExp (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f (Real.exp (x + 1)) = x

/-- The main theorem: if f is symmetric to e^(x+1), then f(x) = ln x - 1 for x > 0 -/
theorem symmetric_to_exp_is_log_minus_one (f : ℝ → ℝ) (h : SymmetricToExp f) :
  ∀ x > 0, f x = Real.log x - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_to_exp_is_log_minus_one_l1100_110064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_constant_l1100_110003

noncomputable section

/-- The curve y = x^4 -/
def curve (x : ℝ) : ℝ := x^4

/-- The line y = mx + d passing through (0, d) -/
def line (m d x : ℝ) : ℝ := m * x + d

/-- The sum s = 1/AC^2 + 1/BC^2 -/
noncomputable def s (A B C : ℝ × ℝ) : ℝ :=
  1 / ((A.1 - C.1)^2 + (A.2 - C.2)^2) + 1 / ((B.1 - C.1)^2 + (B.2 - C.2)^2)

/-- The main theorem stating that there exist points A and B such that
    the sum s is constant (0) for the given curve and line -/
theorem intersection_sum_constant (m : ℝ) :
  ∃ A B : ℝ × ℝ,
    curve A.1 = line m (1/4) A.1 ∧
    curve B.1 = line m (1/4) B.1 ∧
    A ≠ B ∧
    s A B (0, 1/4) = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_constant_l1100_110003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l1100_110012

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * Real.sin x + 2

-- State the theorem
theorem f_symmetry (a b m : ℝ) (h : f a b m = -5) : f a b (-m) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l1100_110012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1100_110038

theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ+, ((-1:ℝ)^(n:ℕ) * a < 2 + (-1:ℝ)^((n:ℕ)+1) / (n:ℝ))) ↔ 
  (a ∈ Set.Icc (-2) (3/2) ∧ a ≠ 3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1100_110038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1100_110058

/-- The eccentricity of a hyperbola with the given conditions is √2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let right_focus : ℝ × ℝ := (c, 0)
  let asymptote := fun (x : ℝ) ↦ -b/a * x
  let A : ℝ × ℝ := (3*c/4, b*c/(4*a))
  let B : ℝ × ℝ := (c/2, b*c/(2*a))
  hyperbola A.1 A.2 ∧ 
  (∀ (x : ℝ), A.2 - B.2 = -b/a * (A.1 - B.1)) ∧
  (A.1 - right_focus.1 = B.1 - A.1 ∧ A.2 - right_focus.2 = B.2 - A.2) →
  c / a = Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1100_110058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_capital_multiple_l1100_110010

/-- Represents the capitals and profit distribution problem --/
structure CapitalProblem where
  P : ℝ  -- Capital of p
  Q : ℝ  -- Capital of q
  R : ℝ  -- Capital of r
  k : ℝ  -- Multiple of q's capital
  totalProfit : ℝ  -- Total profit
  rProfit : ℝ  -- r's share of the profit

/-- The theorem stating the conditions and the result to be proved --/
theorem capital_multiple (prob : CapitalProblem) 
  (h1 : 4 * prob.P = prob.k * prob.Q)
  (h2 : 4 * prob.P = 10 * prob.R)
  (h3 : prob.totalProfit = 4650)
  (h4 : prob.rProfit = 900)
  (h5 : prob.rProfit / prob.totalProfit = prob.R / (prob.P + prob.Q + prob.R)) :
  ∃ (ε : ℝ), abs (prob.k - 6.06) < ε ∧ ε > 0 := by
  sorry

#check capital_multiple

end NUMINAMATH_CALUDE_ERRORFEEDBACK_capital_multiple_l1100_110010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_has_eight_dandelions_l1100_110056

/-- Represents the number of sunflowers Carla has -/
def num_sunflowers : ℕ := 6

/-- Represents the number of seeds per sunflower -/
def seeds_per_sunflower : ℕ := 9

/-- Represents the number of seeds per dandelion -/
def seeds_per_dandelion : ℕ := 12

/-- Represents the percentage of seeds that come from dandelions -/
def dandelion_seed_percentage : ℚ := 64 / 100

/-- Calculates the number of dandelions Carla has based on the given conditions -/
noncomputable def calculate_dandelions : ℕ :=
  let total_seeds := (num_sunflowers * seeds_per_sunflower : ℚ) / (1 - dandelion_seed_percentage)
  let dandelion_seeds := total_seeds * dandelion_seed_percentage
  (dandelion_seeds / seeds_per_dandelion).floor.toNat

/-- Theorem stating that Carla has 8 dandelions -/
theorem carla_has_eight_dandelions : calculate_dandelions = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_has_eight_dandelions_l1100_110056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1100_110076

noncomputable def f (x φ : ℝ) := Real.sin (2 * x + φ)

theorem min_value_of_f (φ : ℝ) (h1 : |φ| < π/2) 
  (h2 : ∀ x, f (x + π/6) φ = -f (-x - π/6) φ) :
  (∀ x ∈ Set.Icc 0 (π/2), f x φ ≥ -Real.sqrt 3 / 2) ∧ 
  (∃ x ∈ Set.Icc 0 (π/2), f x φ = -Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1100_110076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_vector_to_slope_intercept_l1100_110069

/-- Given a line in vector form, prove its slope-intercept form coefficients -/
theorem line_vector_to_slope_intercept :
  ∃ (m b : ℝ), m = -1/2 ∧ b = -11/2 ∧
  ∀ x y : ℝ, (-1 : ℝ) * (x - 3) + 2 * (y + 4) = 0 ↔ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_vector_to_slope_intercept_l1100_110069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trig_product_l1100_110095

theorem max_trig_product :
  (∃ (A B C : Real), A ≥ B ∧ B ≥ C ∧ C ≥ π/8 ∧ A + B + C = π/2 ∧
    720 * Real.sin A * Real.cos B * Real.sin C = 180) ∧
  (∀ (A B C : Real), A ≥ B → B ≥ C → C ≥ π/8 → A + B + C = π/2 →
    720 * Real.sin A * Real.cos B * Real.sin C ≤ 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trig_product_l1100_110095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_trempons_l1100_110035

def give_trempons (n : ℚ) : ℚ := n / 2 + 1/2

theorem john_trempons : 
  ∃ x : ℚ, x > 0 ∧ 
  let r1 := x - give_trempons x;
  let r2 := r1 - give_trempons r1;
  let r3 := r2 - give_trempons r2;
  r3 = 0 ∧ x = 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_trempons_l1100_110035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_pdf_is_even_l1100_110023

/-- The probability density function of a normal distribution -/
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  1 / (σ * Real.sqrt (2 * Real.pi)) * Real.exp (-(1 / 2) * ((x - μ) / σ)^2)

/-- A function f is even if f(x) = f(-x) for all x -/
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The normal PDF with μ = 0 and σ = -1 is an even function -/
theorem normal_pdf_is_even :
  is_even_function (normal_pdf 0 (-1)) := by
  intro x
  -- The proof steps would go here
  sorry

#check normal_pdf_is_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_pdf_is_even_l1100_110023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_brand_ratio_l1100_110091

theorem soap_brand_ratio 
  (total_households : ℕ) 
  (neither_brand : ℕ) 
  (only_R : ℕ) 
  (both_brands : ℕ) 
  (h1 : total_households = 200)
  (h2 : neither_brand = 80)
  (h3 : only_R = 60)
  (h4 : both_brands = 40)
  (h5 : neither_brand + only_R + both_brands ≤ total_households) :
  ∃ (only_B : ℕ), 
    only_B = total_households - (neither_brand + only_R + both_brands) ∧
    only_B = 20 ∧
    (only_B : ℚ) / (both_brands : ℚ) = 1 / 2 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_brand_ratio_l1100_110091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_ad_cost_l1100_110017

/-- Represents the cost calculation for magazine advertisements --/
noncomputable def magazine_ad_cost (page_width : ℝ) (page_height : ℝ) 
  (full_page_rate : ℝ) (half_page_rate : ℝ) (quarter_page_rate : ℝ) 
  (discount_4_plus : ℝ) (discount_6_plus : ℝ) 
  (half_page_count : ℕ) (quarter_page_count : ℕ) : ℝ :=
  let full_page_area := page_width * page_height
  let half_page_area := full_page_area / 2
  let quarter_page_area := full_page_area / 4
  let half_page_cost := (half_page_count : ℝ) * half_page_area * half_page_rate
  let quarter_page_cost := (quarter_page_count : ℝ) * quarter_page_area * quarter_page_rate
  let total_cost := half_page_cost + quarter_page_cost
  let total_ad_count := half_page_count + quarter_page_count
  let discount_rate := if total_ad_count ≥ 6 then discount_6_plus
                       else if total_ad_count ≥ 4 then discount_4_plus
                       else 0
  total_cost * (1 - discount_rate)

/-- Proves that Jason's ad order costs $1360.80 --/
theorem jason_ad_cost : 
  magazine_ad_cost 9 12 6.5 8 10 0.1 0.15 1 4 = 1360.80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_ad_cost_l1100_110017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_shop_pricing_l1100_110059

theorem hat_shop_pricing (x : ℝ) (x_pos : x > 0) :
  0.8775 * x = x * 1.3 * 0.75 * 0.9 := by
  -- Proof steps would go here
  sorry

#check hat_shop_pricing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_shop_pricing_l1100_110059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_root_values_l1100_110090

def is_triple_root (p : Polynomial ℤ) (s : ℤ) : Prop :=
  (Polynomial.X - Polynomial.C s)^3 ∣ p

theorem triple_root_values (b₃ b₂ b₁ : ℤ) (s : ℤ) :
  let p := Polynomial.X^4 + Polynomial.C b₃ * Polynomial.X^3 + Polynomial.C b₂ * Polynomial.X^2 + Polynomial.C b₁ * Polynomial.X + Polynomial.C 24
  is_triple_root p s → s ∈ ({-2, -1, 1, 2} : Set ℤ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_root_values_l1100_110090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_60deg_count_optimal_bamboo_angle_l1100_110034

-- Define the angle between sunlight and ground
def sunlight_ground_angle : ℝ → Prop := sorry

-- Define the number of lines on ground forming 60° with sunlight
def lines_60deg_with_sunlight : ℕ → Prop := sorry

-- Define the angle of bamboo pole with ground
def bamboo_ground_angle : ℝ → Prop := sorry

-- Define the length of bamboo pole shadow
def shadow_length : ℝ → Prop := sorry

-- Theorem 1: Number of lines forming 60° with sunlight
theorem lines_60deg_count :
  ∀ θ : ℝ, sunlight_ground_angle θ →
  (lines_60deg_with_sunlight 0 ∨ ∀ n : ℕ, lines_60deg_with_sunlight n) :=
by sorry

-- Theorem 2: Optimal angle for longest shadow
theorem optimal_bamboo_angle :
  sunlight_ground_angle (π / 3) →
  (∀ α : ℝ, bamboo_ground_angle α → 
   shadow_length (π / 6) ≥ shadow_length α) ∧
  bamboo_ground_angle (π / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_60deg_count_optimal_bamboo_angle_l1100_110034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_leq_one_l1100_110009

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (abs (x - a))

-- State the theorem
theorem increasing_f_implies_a_leq_one (a : ℝ) :
  (∀ x y, 1 ≤ x ∧ x < y → f a x < f a y) → a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_leq_one_l1100_110009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1100_110030

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

-- State the theorem
theorem f_monotone_increasing :
  MonotoneOn f (Set.Ioi 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1100_110030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_coefficient_sum_l1100_110002

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The vertices of the hexagon -/
def hexagonVertices : List Point := [
  ⟨0, 1⟩, ⟨1, 2⟩, ⟨2, 2⟩, ⟨2, 1⟩, ⟨3, 1⟩, ⟨2, 0⟩, ⟨0, 1⟩
]

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the perimeter of the hexagon -/
noncomputable def hexagonPerimeter : ℝ :=
  List.sum (List.zipWith distance hexagonVertices (List.tail hexagonVertices))

/-- Theorem: The sum of coefficients in the perimeter expression is 6 -/
theorem hexagon_perimeter_coefficient_sum :
  ∃ (a b c : ℤ), hexagonPerimeter = a + b * Real.sqrt 2 + c * Real.sqrt 5 ∧ a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_coefficient_sum_l1100_110002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_and_minimum_a_l1100_110048

noncomputable def f (a b x : ℝ) : ℝ := b * x / Real.log x - a * x

theorem tangent_condition_and_minimum_a (e : ℝ) (he : e = Real.exp 1) :
  let f := f
  -- Part 1: Tangent condition implies a = 1 and b = 1
  (∀ x y : ℝ, x > 0 ∧ x ≠ 1 →
    (y = f 1 1 x ∧ 
     3 * Real.sqrt e + y - 4 * Real.sqrt e = 0 ∧
     HasDerivAt (f 1 1) (-3) (Real.sqrt e)) →
    1 = 1 ∧ 1 = 1) ∧
  -- Part 2: Minimum value of a when b = 1
  (∃ a : ℝ, a = 1/2 - 1/(4*e^2) ∧
    ∀ a' : ℝ, (∃ x₁ x₂ : ℝ, e ≤ x₁ ∧ x₁ ≤ e^2 ∧ e ≤ x₂ ∧ x₂ ≤ e^2 ∧
      f a' 1 x₁ ≤ (deriv (f a' 1)) x₂ + a') →
    a ≤ a') :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_and_minimum_a_l1100_110048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1100_110094

/-- The function f(t) = 4t^2 / (1 + 4t^2) -/
noncomputable def f (t : ℝ) : ℝ := 4 * t^2 / (1 + 4 * t^2)

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  f x = y ∧ f y = z ∧ f z = x

/-- The theorem stating that the only solutions to the system are (0, 0, 0) and (1/2, 1/2, 1/2) -/
theorem system_solutions :
  ∀ x y z : ℝ, system x y z → (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) := by
  sorry

#check system_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1100_110094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1100_110036

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sin x ^ 2 + 1/2

-- Define the theorem
theorem function_properties :
  -- Part 1: Interval of monotonic increase
  (∀ x ∈ Set.Icc (Real.pi/2) Real.pi, ∀ y ∈ Set.Icc (Real.pi/2) Real.pi,
    x ≤ y → f x ≤ f y) ∧
  -- Part 2: Triangle area
  (∀ A B C : ℝ,
    -- Triangle ABC is acute
    0 < A ∧ A < Real.pi/2 ∧
    0 < B ∧ B < Real.pi/2 ∧
    0 < C ∧ C < Real.pi/2 ∧
    A + B + C = Real.pi →
    -- Side lengths
    Real.sqrt 19 = 2 * (5 * Real.sin B) / Real.sin A →
    -- f(A) = 0
    f A = 0 →
    -- Area of triangle
    (1/2) * 5 * (2 * (5 * Real.sin B) / Real.sin A) * Real.sin C = 15 * Real.sqrt 3 / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1100_110036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_empty_spaces_probability_l1100_110063

def total_spaces : ℕ := 16
def parked_cars : ℕ := 12

def probability_adjacent_empty_spaces : ℚ :=
  1 - (Nat.choose (parked_cars + (total_spaces - parked_cars - 1)) (total_spaces - parked_cars - 1) : ℚ) / (Nat.choose total_spaces parked_cars : ℚ)

theorem adjacent_empty_spaces_probability :
  probability_adjacent_empty_spaces = 17 / 28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_empty_spaces_probability_l1100_110063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_dividing_polynomial_l1100_110041

theorem smallest_prime_dividing_polynomial :
  ∃ (p : ℕ) (n : ℤ), 
    Nat.Prime p ∧ 
    (p : ℤ) ∣ (n^2 + 5*n + 23) ∧ 
    ∀ (q : ℕ), Nat.Prime q → (q : ℤ) ∣ (n^2 + 5*n + 23) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_dividing_polynomial_l1100_110041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_river_crossing_path_l1100_110028

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a vector in 2D space -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Represents a straight river -/
structure River where
  width : ℝ

/-- Translates a point by a vector -/
def translate (p : Point) (v : Vec2D) : Point :=
  ⟨p.x + v.x, p.y + v.y⟩

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The shortest path crossing the river is achieved when the bridge
    is placed at the intersection of AB' and the river bank -/
theorem shortest_river_crossing_path
  (river : River)
  (A B : Point)
  (v : Vec2D)
  (h1 : v.y = 0 ∧ v.x = river.width)  -- v is perpendicular to the river with length equal to river width
  (h2 : A.x < B.x)  -- A and B are on opposite sides of the river
  (h3 : B.x - A.x > river.width)  -- The river is between A and B
  : 
  let B' := translate B v
  let P := Point.mk A.x B'.y
  ∀ Q : Point, Q.x = A.x → distance A P + distance P B ≤ distance A Q + distance Q B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_river_crossing_path_l1100_110028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baseball_card_distribution_l1100_110099

/-- Proves the number of cards in each binder given the total cards and binder distribution --/
theorem baseball_card_distribution (total_cards : ℝ) (num_binders : ℕ) (fewer_cards_diff : ℝ) 
  (h1 : total_cards = 7496.5)
  (h2 : num_binders = 23)
  (h3 : fewer_cards_diff = 27.7) :
  let cards_per_binder := (total_cards + fewer_cards_diff) / num_binders
  (⌊cards_per_binder + 0.5⌋ = 327) ∧ 
  (⌊cards_per_binder - fewer_cards_diff + 0.5⌋ = 299) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_baseball_card_distribution_l1100_110099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_in_scientific_notation_l1100_110032

-- Define scientific notation
noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

-- Define the original number
def original_number : ℕ := 142000

-- Theorem statement
theorem original_number_in_scientific_notation :
  (original_number : ℝ) = scientific_notation 1.42 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_in_scientific_notation_l1100_110032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_construction_l1100_110053

-- Define a line in 2D space
def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define a point on a line
def PointOnLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  p ∈ l

-- Define distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem segment_construction
  (l : Set (ℝ × ℝ))
  (A : ℝ × ℝ)
  (r : ℝ)
  (h1 : ∃ a b c : ℝ, l = Line a b c)
  (h2 : PointOnLine A l)
  (h3 : r > 0) :
  ∃ B : ℝ × ℝ, PointOnLine B l ∧ distance A B = r :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_construction_l1100_110053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_k_l1100_110040

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := -5 * x^5 + 4 * x^3 - 2 * x^2 + 8

-- Define a proposition for the degree of a polynomial
def hasDegree (p : ℝ → ℝ) (n : ℕ) : Prop := sorry

-- Theorem statement
theorem degree_of_k (k : ℝ → ℝ) :
  (∃ c₅ c₄ c₃ c₂ c₁ c₀ : ℝ, k = λ x ↦ c₅ * x^5 + c₄ * x^4 + c₃ * x^3 + c₂ * x^2 + c₁ * x + c₀) →
  hasDegree (λ x ↦ h x + k x) 2 →
  hasDegree k 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_k_l1100_110040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l1100_110071

-- Define the expression
noncomputable def f (y : ℝ) := (Real.log (5 - y)) / Real.sqrt (y - 2)

-- Theorem statement
theorem f_defined_iff (y : ℝ) : 
  (∃ (x : ℝ), f y = x) ↔ (2 < y ∧ y < 5) := by
  sorry

#check f_defined_iff

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l1100_110071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_when_a_zero_a_value_for_given_distance_sum_l1100_110088

noncomputable section

/-- Line l in parametric form -/
def line_l (a t : ℝ) : ℝ × ℝ := (a - t/2, Real.sqrt 3 * t/2)

/-- Curve C in polar form -/
def curve_C (ρ θ : ℝ) : Prop := ρ * (Real.cos θ)^2 = Real.sin θ

/-- Conversion from polar to Cartesian coordinates -/
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_points_when_a_zero :
  ∃ (t1 t2 : ℝ), t1 ≠ t2 ∧
  curve_C (2 * Real.sqrt 3) (2 * Real.pi / 3) ∧
  curve_C 0 0 ∧
  line_l 0 t1 = polar_to_cartesian (2 * Real.sqrt 3) (2 * Real.pi / 3) ∧
  line_l 0 t2 = polar_to_cartesian 0 0 :=
sorry

theorem a_value_for_given_distance_sum :
  ∃ (a t1 t2 : ℝ), t1 ≠ t2 ∧
  curve_C (distance (a, 0) (line_l a t1)) (Real.arctan ((line_l a t1).2 / (line_l a t1).1)) ∧
  curve_C (distance (a, 0) (line_l a t2)) (Real.arctan ((line_l a t2).2 / (line_l a t2).1)) ∧
  distance (a, 0) (line_l a t1) + distance (a, 0) (line_l a t2) = 8 + 2 * Real.sqrt 3 →
  a = 2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_when_a_zero_a_value_for_given_distance_sum_l1100_110088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1100_110021

noncomputable section

-- Define the rectangle
def rectangle_width : ℝ := 3
def rectangle_height : ℝ := 2

-- Define the line
def line_start (c : ℝ) : ℝ × ℝ := (c, 0)
def line_end : ℝ × ℝ := (2, 3)

-- Define the area of the rectangle
def total_area : ℝ := rectangle_width * rectangle_height

-- Define the area of the triangle formed by the line and the x-axis
def triangle_area (c : ℝ) : ℝ := (1/2) * (2 - c) * 3

-- Theorem statement
theorem equal_area_division (c : ℝ) : 
  triangle_area c = total_area / 2 ↔ c = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1100_110021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_face_formula_l1100_110070

/-- A regular triangular pyramid with base side length a and right-angled lateral faces -/
structure RegularTriangularPyramid (a : ℝ) :=
  (base_side_length : a > 0)
  (lateral_faces_right_angle : True)  -- This is a placeholder for the right angle condition

/-- The distance from the center of the base to a lateral face in a regular triangular pyramid -/
noncomputable def distance_center_to_face (a : ℝ) (pyramid : RegularTriangularPyramid a) : ℝ :=
  (Real.sqrt 3 / 2) * a

/-- Theorem stating that the distance from the center of the base to a lateral face
    in a regular triangular pyramid with base side length a is (√3/2)a -/
theorem distance_center_to_face_formula (a : ℝ) (pyramid : RegularTriangularPyramid a) :
  distance_center_to_face a pyramid = (Real.sqrt 3 / 2) * a := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_face_formula_l1100_110070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1100_110065

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

-- Define the domain of the function
def domain : Set ℝ := {x : ℝ | x ≠ -2}

-- State the theorem
theorem f_range : 
  {y : ℝ | ∃ x ∈ domain, f x = y} = Set.Ioi 1 ∪ Set.Iio 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1100_110065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_chord_length_is_two_l1100_110033

/-- Definition of the ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of semi-major axis a -/
noncomputable def semi_major_axis : ℝ := Real.sqrt 2

/-- Definition of semi-minor axis b -/
def semi_minor_axis : ℝ := 1

/-- Definition of linear eccentricity c -/
noncomputable def linear_eccentricity : ℝ := Real.sqrt (semi_major_axis^2 - semi_minor_axis^2)

/-- Definition of focal chord length -/
noncomputable def focal_chord_length : ℝ := 2 * linear_eccentricity

/-- Theorem: The length of the focal chord of the ellipse x^2/2 + y^2 = 1 is 2 -/
theorem focal_chord_length_is_two : focal_chord_length = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_chord_length_is_two_l1100_110033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l1100_110049

/-- An ellipse with three of its axis endpoints -/
structure Ellipse where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ
  h1 : p1 ∈ ({(10, -3), (15, 7), (25, -3)} : Set (ℝ × ℝ))
  h2 : p2 ∈ ({(10, -3), (15, 7), (25, -3)} : Set (ℝ × ℝ))
  h3 : p3 ∈ ({(10, -3), (15, 7), (25, -3)} : Set (ℝ × ℝ))
  h4 : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3

/-- The distance between the foci of the ellipse -/
noncomputable def focalDistance (e : Ellipse) : ℝ := 2 * Real.sqrt (56.25 - 25)

/-- Theorem stating that the focal distance of the given ellipse is 2√(56.25 - 25) -/
theorem ellipse_focal_distance (e : Ellipse) : 
  focalDistance e = 2 * Real.sqrt (56.25 - 25) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l1100_110049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_circle_from_hexagon_vertices_l1100_110037

/-- A regular hexagon in a 2D plane -/
structure RegularHexagon where
  vertices : Finset (ℝ × ℝ)
  regular : vertices.card = 6
  -- Additional properties ensuring regularity could be added here

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle's diameter has endpoints at hexagon vertices -/
def CircleHasDiameterFromHexagonVertices (h : RegularHexagon) (c : Circle) : Prop :=
  ∃ v1 v2, v1 ∈ h.vertices ∧ v2 ∈ h.vertices ∧ 
    v1 ≠ v2 ∧ 
    (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = (2 * c.radius)^2 ∧
    c.center = ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)

/-- Theorem stating that there's only one circle with diameter endpoints at hexagon vertices -/
theorem unique_circle_from_hexagon_vertices (h : RegularHexagon) : 
  ∃! c : Circle, CircleHasDiameterFromHexagonVertices h c :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_circle_from_hexagon_vertices_l1100_110037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_gain_percent_l1100_110073

theorem cycle_gain_percent (cost_price selling_price : ℝ) :
  cost_price = 930 →
  selling_price = 1210 →
  abs (((selling_price - cost_price) / cost_price * 100) - 30.11) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_gain_percent_l1100_110073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_l1100_110054

/-- Represents a parabola in the form y = a(x - h)² + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The vertex of a parabola --/
def Parabola.vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

/-- Checks if a given point is the vertex of a parabola --/
def is_vertex (p : Parabola) (point : ℝ × ℝ) : Prop :=
  p.vertex = point

/-- Constructs a parabola from its equation coefficients --/
noncomputable def parabola_from_equation (a b c : ℝ) : Parabola :=
  { a := a, h := -b / (2 * a), k := c - (b^2) / (4 * a) }

theorem parabola_equation_correct : 
  let p := parabola_from_equation 1 (-4) 5
  is_vertex p (-2, 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_l1100_110054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_distribution_proof_l1100_110051

def apple_distribution (initial_apples : ℚ) : ℕ → ℚ
  | 0 => initial_apples
  | n + 1 => (apple_distribution initial_apples n / 2) - 1/2

theorem apple_distribution_proof (initial_apples : ℚ) :
  apple_distribution initial_apples 5 = 0 ↔ initial_apples = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_distribution_proof_l1100_110051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1100_110046

/-- A function that represents a five-digit number in the form 54a7b -/
def number (a b : ℕ) : ℕ := 54000 + 100 * a + 10 * 7 + b

/-- Predicate to check if a number is in the correct form -/
def is_valid_form (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = number a b

/-- The set of all valid numbers -/
def valid_numbers : Set ℕ :=
  {n : ℕ | is_valid_form n ∧ n % 3 = 0 ∧ n % 5 = 0}

/-- Lemma: valid_numbers is finite -/
lemma valid_numbers_finite : Set.Finite valid_numbers := by
  sorry

/-- Theorem: There are exactly 7 valid numbers -/
theorem count_valid_numbers : Nat.card valid_numbers = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1100_110046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_time_to_school_approx_l1100_110047

/-- Represents a person's walking characteristics and time to school -/
structure Walker where
  steps_per_minute : ℝ
  step_length : ℝ
  time_to_school : ℝ

/-- Calculates the distance to school based on a walker's characteristics -/
def distance_to_school (w : Walker) : ℝ :=
  w.steps_per_minute * w.step_length * w.time_to_school

/-- Theorem stating that Jack's time to school is approximately 13.62 minutes -/
theorem jack_time_to_school_approx (dave : Walker) (jack : Walker)
    (h_dave_steps : dave.steps_per_minute = 85)
    (h_dave_step_length : dave.step_length = 80)
    (h_dave_time : dave.time_to_school = 15)
    (h_jack_steps : jack.steps_per_minute = 104)
    (h_jack_step_length : jack.step_length = 72)
    (h_same_route : distance_to_school dave = distance_to_school jack) :
    ∃ ε > 0, |jack.time_to_school - 13.62| < ε := by
  sorry

#check jack_time_to_school_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_time_to_school_approx_l1100_110047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1100_110078

def a : ℕ → ℚ
  | 0 => -1
  | n + 1 => a n + 1 / ((n + 1) * (n + 2))

theorem a_formula (n : ℕ) : a n = -1 / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1100_110078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1100_110085

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 - x) / Real.log a - Real.log (1 + x) / Real.log a

theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a (3/5) = 2) :
  (∀ x, f a x ≠ 0 → x ∈ Set.Ioo (-1 : ℝ) 1) ∧ 
  (∀ x, f a (-x) = -f a x) ∧
  (∀ x, f a x > 0 ↔ x ∈ Set.Ioo (0 : ℝ) 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1100_110085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_equals_negative_twenty_l1100_110068

-- Define the logarithm with base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_expression_equals_negative_twenty :
  (lg (1/4) - lg 25) / (100 ^ (-(1/2 : ℝ))) = -20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_equals_negative_twenty_l1100_110068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K_is_regular_iff_l1100_110045

-- Define regular polygon
structure RegularPolygon :=
  (vertices : Set (ℝ × ℝ))
  (is_regular : Bool)

-- Define the set of midpoints
def K (M N : RegularPolygon) : Set (ℝ × ℝ) :=
  {p | ∃ a b, a ∈ M.vertices ∧ b ∈ N.vertices ∧ p = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)}

-- Define homothety
def is_homothetic (M N : RegularPolygon) : Prop :=
  ∃ k : ℝ, ∃ c : ℝ × ℝ, ∀ p ∈ M.vertices, ∃ q ∈ N.vertices, q = (k * (p.1 - c.1) + c.1, k * (p.2 - c.2) + c.2)

-- Define rotation followed by translation
noncomputable def is_rotated_translated (M N : RegularPolygon) : Prop :=
  ∃ θ : ℝ, ∃ t : ℝ × ℝ, ∀ p ∈ M.vertices, ∃ q ∈ N.vertices,
    q = (Real.cos θ * p.1 - Real.sin θ * p.2 + t.1, Real.sin θ * p.1 + Real.cos θ * p.2 + t.2)

-- Main theorem
theorem K_is_regular_iff (M N : RegularPolygon) :
  (∃ K_poly : RegularPolygon, K_poly.vertices = K M N ∧ K_poly.is_regular = true) ↔
  (is_homothetic M N ∨ (is_rotated_translated M N ∧ ∃ m : ℕ, m > 2 ∧ θ = Real.pi / m)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_K_is_regular_iff_l1100_110045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l1100_110018

theorem inequality_problem (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (c / a < b / a) ∧
  ((b - a) / c > 0) ∧
  ((a - c) / (a * c) < 0) ∧
  ¬(∀ x y z : ℝ, z < y ∧ y < x ∧ x * z < 0 → y^2 / z > x^2 / z) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l1100_110018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_from_circles_l1100_110055

theorem square_area_from_circles (r R : ℝ) (h1 : R = (7/3) * r) (h2 : 2 * Real.pi * r = 8) :
  let side := 2 * R
  side * side = 3136 / (9 * Real.pi^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_from_circles_l1100_110055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_greatest_prime_divisor_16386_l1100_110060

/-- The greatest prime divisor of a natural number n -/
def greatestPrimeDivisor (n : ℕ) : ℕ :=
  (Nat.factors n).maximum?.getD 1

/-- The sum of digits of a natural number n -/
def sumOfDigits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

/-- Theorem: The sum of the digits of the greatest prime divisor of 16386 is 5 -/
theorem sum_digits_greatest_prime_divisor_16386 :
  sumOfDigits (greatestPrimeDivisor 16386) = 5 := by
  sorry

#eval sumOfDigits (greatestPrimeDivisor 16386)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_greatest_prime_divisor_16386_l1100_110060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_selection_probability_l1100_110062

/-- Probability of selecting two white balls at positions i and j -/
def probability_two_white (N n M i j : ℕ) : ℚ := sorry

/-- Probability of selecting three white balls at positions i, j, and k -/
def probability_three_white (N n M i j k : ℕ) : ℚ := sorry

/-- Probability of selecting white balls in specific positions from an urn -/
theorem urn_selection_probability 
  (N n M : ℕ) 
  (h1 : n ≥ 2) 
  (h2 : N > n) 
  (h3 : M ≤ N) :
  let P_ij := n * (n - 1) / (N * (N - 1) : ℚ)
  let P_ijk := n * (n - 1) * (n - 2) / (N * (N - 1) * (N - 2) : ℚ)
  ∀ (i j k : ℕ), i < j ∧ j < k ∧ k ≤ M →
  (probability_two_white N n M i j = P_ij ∧ 
   probability_three_white N n M i j k = P_ijk) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_selection_probability_l1100_110062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_trig_matrix_l1100_110082

theorem determinant_trig_matrix (α β : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := 
    !![Real.sin α * Real.cos β, -Real.sin α * Real.sin β, Real.cos α;
      -Real.sin β, -Real.cos β, 0;
      Real.cos α * Real.cos β, Real.cos α * Real.sin β, Real.sin α]
  Matrix.det M = Real.sin α ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_trig_matrix_l1100_110082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2A_cos_pi_3_minus_A_l1100_110016

-- Define the slope angle A based on the line equation 4x - 3y + 12 = 0
noncomputable def A : ℝ := Real.arctan (4 / 3)

-- Theorem for tan(2A)
theorem tan_2A : Real.tan (2 * A) = -24 / 7 := by sorry

-- Theorem for cos(π/3 - A)
theorem cos_pi_3_minus_A : Real.cos (π / 3 - A) = (3 + 4 * Real.sqrt 3) / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2A_cos_pi_3_minus_A_l1100_110016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l1100_110075

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 1 → (2 : ℝ)^x > 1) ∧
  (∃ x : ℝ, (2 : ℝ)^x > 1 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l1100_110075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_over_two_l1100_110052

theorem angle_sum_is_pi_over_two
  (α β : ℝ)
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_sin_sq : 3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 = 1)
  (h_sin_double : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_over_two_l1100_110052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_u_l1100_110084

noncomputable def u (x y z : ℝ) : ℝ := 1 / (1 - x^2) + 4 / (4 - y^2) + 9 / (9 - z^2)

theorem min_value_u {x y z : ℝ} (hx : x ∈ Set.Ioo (-1) 1) (hy : y ∈ Set.Ioo (-1) 1) (hz : z ∈ Set.Ioo (-1) 1)
  (h_prod : x * y * z = 1/36) :
  ∀ a b c, a ∈ Set.Ioo (-1) 1 → b ∈ Set.Ioo (-1) 1 → c ∈ Set.Ioo (-1) 1 → a * b * c = 1/36 →
  u x y z ≤ u a b c ∧ u x y z = 108/35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_u_l1100_110084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_altitude_length_l1100_110081

/-- Calculate the foot of the altitude from a point to a line segment --/
def foot_of_altitude (P A B : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Theorem about the length of the altitude from the centroid to a side of a triangle --/
theorem centroid_altitude_length (D E F : ℝ × ℝ) :
  let DE := Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2)
  let DF := Real.sqrt ((D.1 - F.1)^2 + (D.2 - F.2)^2)
  let EF := Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2)
  DE = 14 ∧ DF = 15 ∧ EF = 21 →
  let G := ((D.1 + E.1 + F.1) / 3, (D.2 + E.2 + F.2) / 3)  -- Centroid
  let Q := foot_of_altitude G E F
  Real.sqrt ((G.1 - Q.1)^2 + (G.2 - Q.2)^2) = 100 * Real.sqrt 22 / 63 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_altitude_length_l1100_110081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_nested_calculation_l1100_110080

/-- The ♠ operation for positive real numbers -/
noncomputable def spade (x y : ℝ) : ℝ := x - 1 / y

/-- Theorem stating that ♠(3, ♠(3, 3)) = 21/8 -/
theorem spade_nested_calculation :
  spade 3 (spade 3 3) = 21 / 8 :=
by
  -- Unfold the definition of spade
  unfold spade
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_nested_calculation_l1100_110080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_composition_l1100_110083

/-- A structure representing a special 100-digit number -/
structure SpecialNumber where
  digits : List Nat
  digit_count : digits.length = 100
  only_ones_and_twos : ∀ d ∈ digits, d = 1 ∨ d = 2
  even_between_twos : ∀ i j, i < j → j < digits.length → 
    digits[i]? = some 2 → digits[j]? = some 2 → (j - i - 1) % 2 = 0
  divisible_by_three : (digits.sum) % 3 = 0

/-- Theorem stating the composition of the special number -/
theorem special_number_composition (N : SpecialNumber) :
  (N.digits.filter (· = 1)).length = 98 ∧
  (N.digits.filter (· = 2)).length = 2 := by
  sorry

#check special_number_composition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_composition_l1100_110083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l1100_110000

theorem min_distance_to_origin (a b : ℝ) : 
  (3 * a + 4 * b = 10) →
  (∀ x y : ℝ, 3 * x + 4 * y = 10 → a^2 + b^2 ≤ x^2 + y^2) →
  Real.sqrt (a^2 + b^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l1100_110000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_same_color_unit_distance_l1100_110098

-- Define a type for colors
inductive Color
| Red
| Green
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- State the theorem
theorem exists_same_color_unit_distance :
  ∃ (p q : Point), coloring p = coloring q ∧ distance p q = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_same_color_unit_distance_l1100_110098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1100_110072

/-- The angle between two 2D vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ := 
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt ((v.1^2 + v.2^2) * (w.1^2 + w.2^2))))

/-- Given three non-collinear planar vectors with equal angles between them and specific magnitudes,
    prove that the magnitude of their sum is 2. -/
theorem vector_sum_magnitude (a b c : ℝ × ℝ) : 
  (∃ θ : ℝ, θ ≠ 0 ∧ θ ≠ π ∧ 
    Real.cos (angle a b) = Real.cos θ ∧ 
    Real.cos (angle b c) = Real.cos θ ∧ 
    Real.cos (angle c a) = Real.cos θ) →
  Real.sqrt (a.1^2 + a.2^2) = 1 →
  Real.sqrt (b.1^2 + b.2^2) = 1 →
  Real.sqrt (c.1^2 + c.2^2) = 3 →
  Real.sqrt ((a.1 + b.1 + c.1)^2 + (a.2 + b.2 + c.2)^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1100_110072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coconut_grove_yield_l1100_110087

/-- Represents the yield of trees in a coconut grove -/
structure CoconutGrove (x : ℕ) where
  yield_xplus2  : ℕ   -- Yield of x + 2 trees
  yield_x       : ℕ   -- Yield of x trees
  yield_xminus2 : ℕ   -- Yield of x - 2 trees
  avg_yield     : ℕ   -- Average yield per tree

/-- The coconut grove problem statement -/
theorem coconut_grove_yield (x : ℕ) (grove : CoconutGrove x)
  (h1 : grove.yield_xplus2 = 30 * (x + 2))
  (h2 : grove.yield_x = 120 * x)
  (h3 : grove.avg_yield = 100)
  (h4 : x = 10) :
  grove.yield_xminus2 = 1440 := by
  sorry

#check coconut_grove_yield

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coconut_grove_yield_l1100_110087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_relation_l1100_110074

theorem cos_angle_relation (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 6) = 1 / 3) :
  Real.cos (α - π / 3) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_relation_l1100_110074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l1100_110019

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![cos θ, -sin θ],
    ![sin θ, cos θ]]

def is_identity (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  M = ![![1, 0],
       ![0, 1]]

theorem smallest_rotation_power :
  (∃ (n : ℕ), n > 0 ∧ is_identity ((rotation_matrix (140 * π / 180))^n)) ∧
  (∀ (m : ℕ), m > 0 ∧ is_identity ((rotation_matrix (140 * π / 180))^m) → m ≥ 18) :=
by
  sorry

#check smallest_rotation_power

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l1100_110019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roll_ceiling_l1100_110008

noncomputable def o : ℝ := 2013
noncomputable def l : ℝ := 1 / 50

noncomputable def r : ℝ := (o^34 * l^36) / (o^33 * l^37)

theorem roll_ceiling : ⌈r * l⌉ = 2013 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roll_ceiling_l1100_110008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_or_quadrilateral_l1100_110039

/-- Represents a convex polygon --/
structure ConvexPolygon where
  vertices : ℕ
  convex : vertices ≥ 3

/-- Represents the state of polygons on the table --/
structure TableState where
  polygons : List ConvexPolygon
  total_vertices : ℕ

/-- Performs a straight-line cut on a polygon --/
def cut (state : TableState) : TableState :=
  { polygons := sorry,
    total_vertices := state.total_vertices + 4 }

theorem exists_triangle_or_quadrilateral 
  (initial_state : TableState)
  (h_initial_decagons : initial_state.polygons.length = 10 ∧ 
                        ∀ p ∈ initial_state.polygons, p.vertices = 10)
  (h_initial_vertices : initial_state.total_vertices = 100)
  (h_cuts : ℕ)
  (h_cuts_count : h_cuts = 51)
  (final_state : TableState)
  (h_final_state : final_state = (List.foldr (λ _ s => cut s) initial_state (List.range h_cuts))) :
  ∃ p ∈ final_state.polygons, p.vertices < 5 :=
by
  sorry

#check exists_triangle_or_quadrilateral

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_or_quadrilateral_l1100_110039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_length_l1100_110001

/-- Given an airplane hangar and planes, calculate the length of each plane. -/
theorem plane_length (hangar_length : ℝ) (num_planes : ℕ) 
  (h1 : hangar_length = 300) 
  (h2 : num_planes = 7) :
  ∃ (ε : ℝ), abs (hangar_length / num_planes - 42.86) < ε ∧ ε < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_length_l1100_110001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1100_110097

theorem trajectory_equation (A M : ℝ × ℝ) : 
  (A.1^2 + A.2^2 = 4) →  -- A is on the circle
  (M.1 = A.1) →  -- M and A have the same x-coordinate (line l is perpendicular to x-axis)
  (M.2 = Real.sqrt 3 / 2 * A.2) →  -- DM = (√3/2) * DA
  (M.1^2 / 4 + M.2^2 / 3 = 1) := by
  sorry

#check trajectory_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1100_110097
