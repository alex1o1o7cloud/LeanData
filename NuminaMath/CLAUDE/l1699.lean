import Mathlib

namespace min_bailing_rate_l1699_169987

/-- Minimum bailing rate problem -/
theorem min_bailing_rate (distance : ℝ) (leak_rate : ℝ) (capacity : ℝ) (speed : ℝ) 
  (h1 : distance = 2)
  (h2 : leak_rate = 15)
  (h3 : capacity = 50)
  (h4 : speed = 3) : 
  ∃ (bailing_rate : ℝ), bailing_rate ≥ 14 ∧ 
  (distance / speed * 60 * (leak_rate - bailing_rate) ≤ capacity) := by
  sorry

end min_bailing_rate_l1699_169987


namespace select_two_from_six_l1699_169909

theorem select_two_from_six (n : ℕ) (k : ℕ) : n = 6 → k = 2 → Nat.choose n k = 15 := by
  sorry

end select_two_from_six_l1699_169909


namespace inequality_proof_l1699_169978

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  x * (x - z)^2 + y * (y - z)^2 ≥ (x - z) * (y - z) * (x + y - z) := by
  sorry

end inequality_proof_l1699_169978


namespace fibonacci_parity_l1699_169920

def E : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | n + 3 => E (n + 2) + E (n + 1)

def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem fibonacci_parity : 
  isEven (E 2021) ∧ ¬isEven (E 2022) ∧ ¬isEven (E 2023) := by sorry

end fibonacci_parity_l1699_169920


namespace perimeter_of_isosceles_triangle_l1699_169993

-- Define the condition for x and y
def satisfies_equation (x y : ℝ) : Prop :=
  |x - 4| + Real.sqrt (y - 10) = 0

-- Define an isosceles triangle with side lengths x, y, and y
def isosceles_triangle (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y > y ∧ y + y > x

-- Define the perimeter of the triangle
def triangle_perimeter (x y : ℝ) : ℝ :=
  x + y + y

-- Theorem statement
theorem perimeter_of_isosceles_triangle (x y : ℝ) :
  satisfies_equation x y → isosceles_triangle x y → triangle_perimeter x y = 24 :=
by
  sorry


end perimeter_of_isosceles_triangle_l1699_169993


namespace tenth_digit_of_expression_l1699_169924

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def tenthDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem tenth_digit_of_expression : 
  tenthDigit ((factorial 5 * factorial 5 - factorial 5 * factorial 3) / 5) = 3 := by
  sorry

end tenth_digit_of_expression_l1699_169924


namespace specific_trapezoid_area_l1699_169923

/-- An isosceles trapezoid with given base lengths and angle -/
structure IsoscelesTrapezoid where
  larger_base : ℝ
  smaller_base : ℝ
  angle_at_larger_base : ℝ

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The area of the specific isosceles trapezoid is 15 -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    larger_base := 8,
    smaller_base := 2,
    angle_at_larger_base := Real.pi / 4  -- 45° in radians
  }
  area t = 15 := by
  sorry

end specific_trapezoid_area_l1699_169923


namespace unique_solution_condition_l1699_169952

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 5) * (x - 3) = -15 + k * x) ↔ k = -4 :=
by sorry

end unique_solution_condition_l1699_169952


namespace max_xyz_value_l1699_169926

theorem max_xyz_value (x y z : ℝ) 
  (eq1 : x + x*y + x*y*z = 1)
  (eq2 : y + y*z + x*y*z = 2)
  (eq3 : z + x*z + x*y*z = 4) :
  x*y*z ≤ (5 + Real.sqrt 17) / 2 :=
sorry

end max_xyz_value_l1699_169926


namespace daily_profit_function_l1699_169964

/-- The daily profit function for a product with given cost and sales quantity relation -/
theorem daily_profit_function (x : ℝ) : 
  let cost : ℝ := 8
  let sales_quantity : ℝ → ℝ := λ price => -price + 30
  let profit : ℝ → ℝ := λ price => (price - cost) * (sales_quantity price)
  profit x = -x^2 + 38*x - 240 := by
sorry

end daily_profit_function_l1699_169964


namespace max_a_value_l1699_169985

/-- A lattice point in an xy-coordinate system -/
def LatticePoint (x y : ℤ) : Prop := True

/-- The line equation y = mx + 3 -/
def LineEquation (m : ℚ) (x y : ℤ) : Prop := y = m * x + 3

/-- Predicate for a line not passing through any lattice point in the given range -/
def NoLatticePointIntersection (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x → x ≤ 150 → LatticePoint x y → ¬LineEquation m x y

/-- The theorem statement -/
theorem max_a_value :
  (∀ m : ℚ, 1/3 < m → m < 50/149 → NoLatticePointIntersection m) ∧
  ¬(∀ m : ℚ, 1/3 < m → m < 50/149 + ε → NoLatticePointIntersection m) :=
sorry

end max_a_value_l1699_169985


namespace clock_strikes_l1699_169974

/-- If a clock strikes three times in 12 seconds, it will strike six times in 30 seconds. -/
theorem clock_strikes (strike_interval : ℝ) : 
  (3 * strike_interval = 12) → (6 * strike_interval = 30) := by
  sorry

end clock_strikes_l1699_169974


namespace no_roots_of_composite_l1699_169979

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The condition that f(x) = x has no real roots -/
def no_real_roots (b c : ℝ) : Prop := ∀ x : ℝ, f b c x ≠ x

/-- The theorem stating that if f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_roots_of_composite (b c : ℝ) (h : no_real_roots b c) :
  ∀ x : ℝ, f b c (f b c x) ≠ x := by
  sorry

end no_roots_of_composite_l1699_169979


namespace hypotenuse_of_45_45_90_triangle_l1699_169984

-- Define a right triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  angle_opposite_leg1 : ℝ
  is_right_triangle : leg1^2 + leg2^2 = hypotenuse^2

-- Theorem statement
theorem hypotenuse_of_45_45_90_triangle 
  (triangle : RightTriangle) 
  (h1 : triangle.leg1 = 12)
  (h2 : triangle.angle_opposite_leg1 = 45) :
  triangle.hypotenuse = 12 * Real.sqrt 2 := by
sorry

end hypotenuse_of_45_45_90_triangle_l1699_169984


namespace max_value_in_D_l1699_169914

-- Define the region D
def D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = 0 ∧ p.1 ≤ 1 ∧ p.1 ≥ 0}

-- Define the objective function
def z (p : ℝ × ℝ) : ℝ := p.1 - 2*p.2 + 5

-- Theorem statement
theorem max_value_in_D :
  ∃ (m : ℝ), m = 8 ∧ ∀ p ∈ D, z p ≤ m :=
sorry

end max_value_in_D_l1699_169914


namespace discount_difference_l1699_169928

def initial_amount : ℝ := 12000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def scheme1 (amount : ℝ) : ℝ :=
  apply_discount (apply_discount (apply_discount amount 0.25) 0.15) 0.10

def scheme2 (amount : ℝ) : ℝ :=
  apply_discount (apply_discount (apply_discount amount 0.30) 0.10) 0.05

theorem discount_difference :
  scheme1 initial_amount - scheme2 initial_amount = 297 := by
  sorry

end discount_difference_l1699_169928


namespace max_runs_in_match_l1699_169936

/-- Represents the number of overs in a cricket match -/
def overs : ℕ := 20

/-- Represents the maximum number of runs a batsman can score -/
def max_runs : ℕ := 663

/-- Represents the number of balls in an over -/
def balls_per_over : ℕ := 6

/-- Represents the maximum runs that can be scored off a single ball -/
def max_runs_per_ball : ℕ := 6

/-- Represents the total number of balls in the match -/
def total_balls : ℕ := overs * balls_per_over

/-- Theorem stating that under certain conditions, the maximum runs a batsman can score in the match is 663 -/
theorem max_runs_in_match : 
  ∃ (balls_faced : ℕ) (runs_per_ball : ℕ), 
    balls_faced ≤ total_balls ∧ 
    runs_per_ball ≤ max_runs_per_ball ∧ 
    balls_faced * runs_per_ball = max_runs :=
sorry

end max_runs_in_match_l1699_169936


namespace solution_value_l1699_169994

theorem solution_value (a : ℝ) : (3 * 5 - 2 * a = 7) → a = 4 := by
  sorry

end solution_value_l1699_169994


namespace fraction_addition_l1699_169959

theorem fraction_addition : (3 / 4) / (5 / 8) + 1 / 2 = 17 / 10 := by
  sorry

end fraction_addition_l1699_169959


namespace planar_graph_properties_l1699_169929

structure PlanarGraph where
  s : ℕ  -- number of vertices
  a : ℕ  -- number of edges
  f : ℕ  -- number of faces

def no_triangular_faces (G : PlanarGraph) : Prop :=
  -- This is a placeholder for the condition that no face is a triangle
  True

theorem planar_graph_properties (G : PlanarGraph) :
  (G.s - G.a + G.f = 2) ∧
  (G.a ≤ 3 * G.s - 6) ∧
  (no_triangular_faces G → G.a ≤ 2 * G.s - 4) := by
  sorry

end planar_graph_properties_l1699_169929


namespace sphere_surface_area_l1699_169962

/-- Given a sphere with volume 4√3π, its surface area is 12π -/
theorem sphere_surface_area (V : ℝ) (R : ℝ) (S : ℝ) : 
  V = 4 * Real.sqrt 3 * Real.pi → 
  V = (4 / 3) * Real.pi * R^3 →
  S = 4 * Real.pi * R^2 →
  S = 12 * Real.pi := by
sorry


end sphere_surface_area_l1699_169962


namespace medication_frequency_l1699_169995

/-- The number of times Kara takes her medication per day -/
def medication_times_per_day : ℕ := sorry

/-- The amount of water Kara drinks with each medication dose in ounces -/
def water_per_dose : ℕ := 4

/-- The number of days Kara followed her medication schedule -/
def days_followed : ℕ := 14

/-- The number of doses Kara missed in the two-week period -/
def doses_missed : ℕ := 2

/-- The total amount of water Kara drank with her medication over two weeks in ounces -/
def total_water_consumed : ℕ := 160

theorem medication_frequency :
  medication_times_per_day = 3 :=
by
  have h1 : water_per_dose * (days_followed * medication_times_per_day - doses_missed) = total_water_consumed := sorry
  sorry

end medication_frequency_l1699_169995


namespace smallest_tree_height_l1699_169925

/-- Given three trees with specific height relationships, prove the height of the smallest tree -/
theorem smallest_tree_height (tallest middle smallest : ℝ) : 
  tallest = 108 →
  middle = tallest / 2 - 6 →
  smallest = middle / 4 →
  smallest = 12 := by sorry

end smallest_tree_height_l1699_169925


namespace cousins_arrangement_l1699_169922

/-- The number of ways to arrange cousins in rooms -/
def arrange_cousins (n : ℕ) (m : ℕ) : ℕ :=
  -- n is the number of cousins
  -- m is the number of rooms
  sorry

/-- Theorem: Arranging 5 cousins in 4 rooms with at least one empty room -/
theorem cousins_arrangement :
  arrange_cousins 5 4 = 56 :=
by sorry

end cousins_arrangement_l1699_169922


namespace orthogonal_vectors_l1699_169915

theorem orthogonal_vectors (y : ℝ) : y = 28 / 3 →
  (3 : ℝ) * y + 7 * (-4 : ℝ) = 0 := by
  sorry

end orthogonal_vectors_l1699_169915


namespace ice_cream_distribution_l1699_169903

theorem ice_cream_distribution (nieces : ℚ) (total_sandwiches : ℕ) :
  nieces = 11 ∧ total_sandwiches = 1573 →
  (total_sandwiches : ℚ) / nieces = 143 := by
  sorry

end ice_cream_distribution_l1699_169903


namespace polynomial_ratio_l1699_169921

-- Define the polynomial coefficients
variable (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ)

-- Define the main equation
def main_equation (x : ℚ) : Prop :=
  (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5

-- State the theorem
theorem polynomial_ratio :
  (∀ x, main_equation a₀ a₁ a₂ a₃ a₄ a₅ x) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61 / 60 :=
by sorry

end polynomial_ratio_l1699_169921


namespace sheridan_cats_l1699_169965

/-- The total number of cats Mrs. Sheridan has after buying more -/
def total_cats (initial : Float) (bought : Float) : Float :=
  initial + bought

/-- Theorem stating that Mrs. Sheridan's total number of cats is 54.0 -/
theorem sheridan_cats : total_cats 11.0 43.0 = 54.0 := by
  sorry

end sheridan_cats_l1699_169965


namespace largest_non_expressible_l1699_169961

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_expressible (n : ℕ) : Prop :=
  ∃ a b, a > 0 ∧ is_composite b ∧ n = 36 * a + b

theorem largest_non_expressible : 
  (∀ n > 188, is_expressible n) ∧ ¬is_expressible 188 :=
sorry

end largest_non_expressible_l1699_169961


namespace pepperoni_count_l1699_169939

/-- Represents a pizza with pepperoni slices -/
structure Pizza :=
  (total_slices : ℕ)

/-- Represents a quarter of a pizza -/
def QuarterPizza := Pizza

theorem pepperoni_count (p : Pizza) (q : QuarterPizza) :
  (p.total_slices = 4 * q.total_slices) →
  (q.total_slices = 10) →
  (p.total_slices = 40) := by
  sorry

end pepperoni_count_l1699_169939


namespace no_solution_quadratic_inequality_l1699_169930

theorem no_solution_quadratic_inequality :
  ¬ ∃ x : ℝ, 3 * x^2 + 9 * x ≤ -12 := by
sorry

end no_solution_quadratic_inequality_l1699_169930


namespace decimal_division_equals_forty_l1699_169996

theorem decimal_division_equals_forty : (0.24 : ℚ) / (0.006 : ℚ) = 40 := by
  sorry

end decimal_division_equals_forty_l1699_169996


namespace quadratic_equation_roots_l1699_169918

/-- Theorem: For the quadratic equation x^2 + x - 2 = m, when m > 0, the equation has two distinct real roots. -/
theorem quadratic_equation_roots (m : ℝ) (h : m > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + x₁ - 2 = m ∧ x₂^2 + x₂ - 2 = m :=
sorry

end quadratic_equation_roots_l1699_169918


namespace lcm_gcf_problem_l1699_169901

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 16 = 48 → Nat.gcd n 16 = 8 → n = 24 := by
  sorry

end lcm_gcf_problem_l1699_169901


namespace more_sad_left_l1699_169943

/-- Represents the state of a player in the game -/
inductive PlayerState
| Sad
| Cheerful

/-- Represents the game with its rules and initial state -/
structure Game where
  initial_players : ℕ
  remaining_player : ℕ
  sad_left : ℕ
  cheerful_left : ℕ

/-- The game rules ensure that when only one player remains, more sad players have left than cheerful players -/
theorem more_sad_left (g : Game) 
  (h1 : g.initial_players = 36)
  (h2 : g.remaining_player = 1)
  (h3 : g.sad_left + g.cheerful_left = g.initial_players - g.remaining_player) :
  g.sad_left > g.cheerful_left := by
  sorry

#check more_sad_left

end more_sad_left_l1699_169943


namespace m_plus_n_equals_plus_minus_one_l1699_169963

theorem m_plus_n_equals_plus_minus_one (m n : ℤ) 
  (hm : |m| = 3) 
  (hn : |n| = 2) 
  (hmn : m * n < 0) : 
  m + n = 1 ∨ m + n = -1 := by
sorry

end m_plus_n_equals_plus_minus_one_l1699_169963


namespace skew_lines_theorem_l1699_169946

-- Define the concept of a line in 3D space
structure Line3D where
  -- This is a placeholder definition. In a real scenario, we would need to define
  -- what constitutes a line in 3D space, likely using vectors or points.
  mk :: (dummy : Unit)

-- Define what it means for lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Two lines are skew if they are not coplanar and do not intersect
  sorry

-- Define what it means for lines to be parallel
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Two lines are parallel if they are coplanar and do not intersect
  sorry

-- Define what it means for lines to intersect
def do_intersect (l1 l2 : Line3D) : Prop :=
  -- Two lines intersect if they share a point
  sorry

-- The main theorem
theorem skew_lines_theorem (a b c : Line3D) 
  (h1 : are_skew a b)
  (h2 : are_parallel a c)
  (h3 : ¬do_intersect b c) :
  are_skew b c := by
  sorry

end skew_lines_theorem_l1699_169946


namespace square_difference_252_248_l1699_169940

theorem square_difference_252_248 : 252^2 - 248^2 = 2000 := by
  sorry

end square_difference_252_248_l1699_169940


namespace power_six_mod_five_remainder_six_power_23_mod_five_l1699_169927

theorem power_six_mod_five (n : ℕ) : 6^n ≡ 1 [ZMOD 5] := by sorry

theorem remainder_six_power_23_mod_five : 6^23 ≡ 1 [ZMOD 5] := by sorry

end power_six_mod_five_remainder_six_power_23_mod_five_l1699_169927


namespace constant_polar_angle_forms_cone_l1699_169991

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the set of points satisfying φ = c
def ConstantPolarAngleSet (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

-- Statement: The set of points with constant polar angle forms a cone
theorem constant_polar_angle_forms_cone (c : ℝ) :
  ∃ (cone : Set SphericalCoord), ConstantPolarAngleSet c = cone :=
sorry

end constant_polar_angle_forms_cone_l1699_169991


namespace leigh_has_16_seashells_l1699_169973

/-- The number of seashells Leigh has, given the conditions of the problem -/
def leighs_seashells : ℕ :=
  let mimis_shells := 2 * 12  -- 2 dozen
  let kyles_shells := 2 * mimis_shells  -- twice as many as Mimi
  kyles_shells / 3  -- one-third of Kyle's shells

/-- Theorem stating that Leigh has 16 seashells -/
theorem leigh_has_16_seashells : leighs_seashells = 16 := by
  sorry

end leigh_has_16_seashells_l1699_169973


namespace lines_perp_to_plane_are_parallel_l1699_169954

-- Define the types for lines and planes
variable (L : Type*) [AddCommGroup L] [Module ℝ L]
variable (P : Type*) [AddCommGroup P] [Module ℝ P]

-- Define the perpendicular and parallel relations
variable (perpendicular : L → P → Prop)
variable (parallel : L → L → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel
  (a b : L) (M : P)
  (h1 : perpendicular a M)
  (h2 : perpendicular b M) :
  parallel a b :=
sorry

end lines_perp_to_plane_are_parallel_l1699_169954


namespace reorganize_32_city_graph_l1699_169986

/-- A graph with n vertices, where each pair of vertices is connected by a directed edge. -/
structure DirectedGraph (n : ℕ) where
  edges : Fin n → Fin n → Bool

/-- The number of steps required to reorganize a directed graph with n vertices
    such that the resulting graph has no cycles. -/
def reorganization_steps (n : ℕ) : ℕ :=
  if n ≤ 2 then 0 else 2^(n-2) * (2^n - n - 1)

/-- Theorem stating that for a graph with 32 vertices, it's possible to reorganize
    the edge directions in at most 208 steps to eliminate all cycles. -/
theorem reorganize_32_city_graph :
  reorganization_steps 32 ≤ 208 :=
sorry

end reorganize_32_city_graph_l1699_169986


namespace percent_decrease_l1699_169900

theorem percent_decrease (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 55) : 
  (original_price - sale_price) / original_price * 100 = 45 := by
  sorry

end percent_decrease_l1699_169900


namespace least_six_digit_binary_l1699_169999

theorem least_six_digit_binary : ∃ n : ℕ, n = 32 ∧ 
  (∀ m : ℕ, m < n → (Nat.log 2 m).succ < 6) ∧
  (Nat.log 2 n).succ = 6 :=
sorry

end least_six_digit_binary_l1699_169999


namespace bus_speed_excluding_stoppages_l1699_169912

/-- Given a bus that stops for 15 minutes per hour and has a speed of 48 km/hr including stoppages,
    its speed excluding stoppages is 64 km/hr. -/
theorem bus_speed_excluding_stoppages 
  (stop_time : ℝ) 
  (speed_with_stoppages : ℝ) 
  (h1 : stop_time = 15) 
  (h2 : speed_with_stoppages = 48) : 
  speed_with_stoppages * (60 / (60 - stop_time)) = 64 :=
sorry

end bus_speed_excluding_stoppages_l1699_169912


namespace circle_configuration_implies_zero_area_l1699_169949

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line structure
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def CircleTangentToLine (c : Circle) (l : Line) : Prop :=
  sorry

def CirclesExternallyTangent (c1 c2 : Circle) : Prop :=
  sorry

def PointBetween (p1 p2 p3 : ℝ × ℝ) : Prop :=
  sorry

def TriangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  sorry

theorem circle_configuration_implies_zero_area 
  (P Q R : Circle)
  (l : Line)
  (P' Q' R' : ℝ × ℝ)
  (h1 : P.radius = 2)
  (h2 : Q.radius = 3)
  (h3 : R.radius = 4)
  (h4 : CircleTangentToLine P l)
  (h5 : CircleTangentToLine Q l)
  (h6 : CircleTangentToLine R l)
  (h7 : CirclesExternallyTangent Q P)
  (h8 : CirclesExternallyTangent Q R)
  (h9 : PointBetween P' Q' R')
  (h10 : P' = (P.center.1, l.a * P.center.1 + l.b))
  (h11 : Q' = (Q.center.1, l.a * Q.center.1 + l.b))
  (h12 : R' = (R.center.1, l.a * R.center.1 + l.b)) :
  TriangleArea P.center Q.center R.center = 0 :=
sorry

end circle_configuration_implies_zero_area_l1699_169949


namespace sasha_kolya_distance_l1699_169971

/-- Represents the race scenario with three runners -/
structure RaceScenario where
  race_length : ℝ
  sasha_speed : ℝ
  lesha_speed : ℝ
  kolya_speed : ℝ
  sasha_lesha_gap : ℝ
  lesha_kolya_gap : ℝ
  (sasha_speed_pos : sasha_speed > 0)
  (lesha_speed_pos : lesha_speed > 0)
  (kolya_speed_pos : kolya_speed > 0)
  (race_length_pos : race_length > 0)
  (sasha_lesha_gap_pos : sasha_lesha_gap > 0)
  (lesha_kolya_gap_pos : lesha_kolya_gap > 0)
  (sasha_fastest : sasha_speed > lesha_speed ∧ sasha_speed > kolya_speed)
  (lesha_second : lesha_speed > kolya_speed)
  (sasha_lesha_relation : lesha_speed * race_length = sasha_speed * (race_length - sasha_lesha_gap))
  (lesha_kolya_relation : kolya_speed * race_length = lesha_speed * (race_length - lesha_kolya_gap))

/-- Theorem stating the distance between Sasha and Kolya when Sasha finishes -/
theorem sasha_kolya_distance (scenario : RaceScenario) :
  let sasha_finish_time := scenario.race_length / scenario.sasha_speed
  let kolya_distance := scenario.kolya_speed * sasha_finish_time
  scenario.race_length - kolya_distance = 19 := by sorry

end sasha_kolya_distance_l1699_169971


namespace win_sector_area_l1699_169980

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 6) (h2 : p = 1/3) :
  p * (π * r^2) = 12 * π := by
  sorry

end win_sector_area_l1699_169980


namespace audiobook_disc_content_l1699_169907

theorem audiobook_disc_content (total_time min_per_disc : ℕ) 
  (h1 : total_time = 520) 
  (h2 : min_per_disc = 65) : 
  ∃ (num_discs : ℕ), 
    num_discs > 0 ∧ 
    num_discs * min_per_disc = total_time ∧ 
    ∀ (n : ℕ), n > 0 → n * min_per_disc < total_time → n < num_discs :=
by sorry

end audiobook_disc_content_l1699_169907


namespace correct_statements_l1699_169988

/-- Represents a mathematical statement about proofs and principles -/
inductive MathStatement
  | InductionInfinite
  | ProofStructure
  | TheoremProof
  | AxiomPostulate
  | NoUnprovenConjectures

/-- Determines if a given mathematical statement is correct -/
def is_correct (statement : MathStatement) : Prop :=
  match statement with
  | MathStatement.InductionInfinite => False
  | MathStatement.ProofStructure => True
  | MathStatement.TheoremProof => True
  | MathStatement.AxiomPostulate => True
  | MathStatement.NoUnprovenConjectures => True

/-- Theorem stating that statement A is incorrect while B, C, D, and E are correct -/
theorem correct_statements :
  ¬(is_correct MathStatement.InductionInfinite) ∧
  (is_correct MathStatement.ProofStructure) ∧
  (is_correct MathStatement.TheoremProof) ∧
  (is_correct MathStatement.AxiomPostulate) ∧
  (is_correct MathStatement.NoUnprovenConjectures) :=
sorry

end correct_statements_l1699_169988


namespace binomial_sum_equals_power_of_two_l1699_169935

theorem binomial_sum_equals_power_of_two : 
  3^2006 - Nat.choose 2006 1 * 3^2005 + Nat.choose 2006 2 * 3^2004 - Nat.choose 2006 3 * 3^2003 +
  Nat.choose 2006 4 * 3^2002 - Nat.choose 2006 5 * 3^2001 + 
  -- ... (omitting middle terms for brevity)
  Nat.choose 2006 2004 * 3^2 - Nat.choose 2006 2005 * 3 + 1 = 2^2006 :=
by sorry

end binomial_sum_equals_power_of_two_l1699_169935


namespace equation_of_line_l_equations_of_line_m_l1699_169960

-- Define the slope of line l
def slope_l : ℚ := -3/4

-- Define the equation of the line that point P is on
def line_p (k : ℚ) (x y : ℝ) : Prop := k * x - y + 2 * k + 5 = 0

-- Define point P
def point_p : ℝ × ℝ := (-2, 5)

-- Define the distance from point P to line m
def distance_p_to_m : ℝ := 3

-- Theorem for the equation of line l
theorem equation_of_line_l :
  ∃ (A B C : ℝ), A * point_p.1 + B * point_p.2 + C = 0 ∧
  B ≠ 0 ∧ -A/B = slope_l ∧
  ∀ (x y : ℝ), A * x + B * y + C = 0 ↔ y = slope_l * x + (point_p.2 - slope_l * point_p.1) :=
sorry

-- Theorem for the equations of line m
theorem equations_of_line_m :
  ∃ (b₁ b₂ : ℝ), 
    (∀ (x y : ℝ), y = slope_l * x + b₁ ↔ 
      distance_p_to_m = |slope_l * point_p.1 - point_p.2 + b₁| / Real.sqrt (slope_l^2 + 1)) ∧
    (∀ (x y : ℝ), y = slope_l * x + b₂ ↔ 
      distance_p_to_m = |slope_l * point_p.1 - point_p.2 + b₂| / Real.sqrt (slope_l^2 + 1)) ∧
    b₁ ≠ b₂ :=
sorry

end equation_of_line_l_equations_of_line_m_l1699_169960


namespace range_of_a_l1699_169966

/-- Custom operation ⊗ -/
def custom_op (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of a given the condition -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, custom_op (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end range_of_a_l1699_169966


namespace range_of_m_l1699_169953

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 - m*x + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + m > 0

-- Define the main theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m) → 1 < m ∧ m < 2 :=
sorry

end range_of_m_l1699_169953


namespace intersection_count_504_220_l1699_169975

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A line segment from (0,0) to (a,b) -/
structure LineSegment where
  a : ℤ
  b : ℤ

/-- Count of intersections with squares and circles -/
structure IntersectionCount where
  squares : ℕ
  circles : ℕ

/-- Function to count intersections of a line segment with squares and circles -/
def countIntersections (l : LineSegment) : IntersectionCount :=
  sorry

theorem intersection_count_504_220 :
  let l : LineSegment := ⟨504, 220⟩
  let count : IntersectionCount := countIntersections l
  count.squares + count.circles = 255 := by
  sorry

end intersection_count_504_220_l1699_169975


namespace locomotive_whistle_distance_l1699_169945

/-- The speed of the locomotive in meters per second -/
def locomotive_speed : ℝ := 20

/-- The speed of sound in meters per second -/
def sound_speed : ℝ := 340

/-- The time difference between hearing the whistle and the train's arrival in seconds -/
def time_difference : ℝ := 4

/-- The distance of the locomotive when it started whistling in meters -/
def whistle_distance : ℝ := 85

theorem locomotive_whistle_distance :
  (whistle_distance / locomotive_speed) - time_difference = whistle_distance / sound_speed :=
by sorry

end locomotive_whistle_distance_l1699_169945


namespace downstream_speed_l1699_169956

/-- 
Theorem: Given a man's upstream rowing speed and still water speed, 
we can determine his downstream rowing speed.
-/
theorem downstream_speed 
  (upstream_speed : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : upstream_speed = 22) 
  (h2 : still_water_speed = 32) : 
  ∃ downstream_speed : ℝ, 
    downstream_speed = 2 * still_water_speed - upstream_speed ∧ 
    downstream_speed = 42 := by
  sorry

end downstream_speed_l1699_169956


namespace seating_arrangements_l1699_169972

def number_of_seats : ℕ := 9
def number_of_families : ℕ := 3
def members_per_family : ℕ := 3

theorem seating_arrangements :
  (number_of_seats = number_of_families * members_per_family) →
  (number_of_different_seating_arrangements : ℕ) = (Nat.factorial number_of_families)^(number_of_families + 1) :=
by sorry

end seating_arrangements_l1699_169972


namespace scientific_notation_of_11930000_l1699_169910

/-- Proves that 11,930,000 is equal to 1.193 × 10^7 in scientific notation -/
theorem scientific_notation_of_11930000 : 
  11930000 = 1.193 * (10 : ℝ)^7 := by sorry

end scientific_notation_of_11930000_l1699_169910


namespace triangle_abc_properties_l1699_169976

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C (in radians),
    prove the properties of the triangle given specific conditions. -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  B = π / 3 →
  3 * b * Real.sin A = 2 * c * Real.sin B →
  c = 3 ∧
  b = Real.sqrt 7 ∧
  (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 :=
by sorry

end triangle_abc_properties_l1699_169976


namespace sugar_for_recipe_l1699_169931

/-- The amount of sugar required for a cake recipe -/
theorem sugar_for_recipe (sugar_frosting sugar_cake : ℚ) 
  (h1 : sugar_frosting = 6/10)
  (h2 : sugar_cake = 2/10) :
  sugar_frosting + sugar_cake = 8/10 := by
  sorry

end sugar_for_recipe_l1699_169931


namespace log_base_5_inequality_l1699_169970

theorem log_base_5_inequality (x : ℝ) (h1 : 0 < x) (h2 : Real.log x / Real.log 5 < 1) : 1 < x ∧ x < 5 := by
  sorry

end log_base_5_inequality_l1699_169970


namespace tangent_line_equation_point_B_coordinates_fixed_point_on_AB_l1699_169950

-- Define the parabola Γ
def Γ (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define point D
def D (p x₀ y₀ : ℝ) : Prop := y₀^2 > 2*p*x₀

-- Define tangent line through D intersecting Γ at A and B
def tangent_line (p x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  D p x₀ y₀ ∧ Γ p x₁ y₁ ∧ Γ p x₂ y₂

-- Theorem 1: Line yy₁ = p(x + x₁) is tangent to Γ
theorem tangent_line_equation (p x₀ y₀ x₁ y₁ : ℝ) :
  tangent_line p x₀ y₀ x₁ y₁ x₁ y₁ → ∀ x y, y * y₁ = p * (x + x₁) := by sorry

-- Theorem 2: Coordinates of B when A(4, 4) and D on directrix
theorem point_B_coordinates (p : ℝ) :
  Γ p 4 4 → D p (-p/2) (3/2) → ∃ x₂ y₂, Γ p x₂ y₂ ∧ x₂ = 1/4 ∧ y₂ = -1 := by sorry

-- Theorem 3: AB passes through fixed point when D moves on x + p = 0
theorem fixed_point_on_AB (p x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) :
  tangent_line p x₀ y₀ x₁ y₁ x₂ y₂ → x₀ = -p → 
  ∃ k b, y₁ - y₂ = k * (x₁ - x₂) ∧ y₁ = k * x₁ + b ∧ 0 = k * p + b := by sorry

end tangent_line_equation_point_B_coordinates_fixed_point_on_AB_l1699_169950


namespace perpendicular_vectors_x_value_l1699_169938

/-- Two 2D vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Given vectors a and b, prove that if they are perpendicular, then x = 6 -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  perpendicular a b → x = 6 :=
by
  sorry

#check perpendicular_vectors_x_value

end perpendicular_vectors_x_value_l1699_169938


namespace remainder_3_800_mod_17_l1699_169906

theorem remainder_3_800_mod_17 : 3^800 % 17 = 1 := by
  sorry

end remainder_3_800_mod_17_l1699_169906


namespace hundredth_term_is_9999_l1699_169969

/-- The nth term of the sequence -/
def sequenceTerm (n : ℕ) : ℕ := n^2 - 1

/-- Theorem: The 100th term of the sequence is 9999 -/
theorem hundredth_term_is_9999 : sequenceTerm 100 = 9999 := by
  sorry

end hundredth_term_is_9999_l1699_169969


namespace even_shifted_implies_equality_l1699_169947

def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) = f (1 - x)

theorem even_shifted_implies_equality (f : ℝ → ℝ) 
  (h : is_even_shifted f) : f 0 = f 2 := by
  sorry

end even_shifted_implies_equality_l1699_169947


namespace two_cubed_and_three_squared_are_like_terms_l1699_169957

-- Define what it means for two expressions to be like terms
def are_like_terms (a b : ℕ) : Prop :=
  (∃ (x y : ℕ), a = x ∧ b = y) ∨ (∀ (x y : ℕ), a ≠ x ∧ b ≠ y)

-- Theorem statement
theorem two_cubed_and_three_squared_are_like_terms :
  are_like_terms (2^3) (3^2) :=
sorry

end two_cubed_and_three_squared_are_like_terms_l1699_169957


namespace last_non_zero_digit_30_factorial_l1699_169937

/-- The last non-zero digit of a natural number -/
def lastNonZeroDigit (n : ℕ) : ℕ :=
  n % 10 -- Definition, not from solution steps

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem last_non_zero_digit_30_factorial :
  lastNonZeroDigit (factorial 30) = 8 := by
  sorry

end last_non_zero_digit_30_factorial_l1699_169937


namespace units_digit_of_base_l1699_169913

/-- Given a natural number, return its unit's digit -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The product of the given terms -/
def product (x : ℕ) : ℕ := (x ^ 41) * (41 ^ 14) * (14 ^ 87) * (87 ^ 76)

/-- The theorem stating that if the unit's digit of the product is 4, 
    then the unit's digit of x must be 1 -/
theorem units_digit_of_base (x : ℕ) : 
  unitsDigit (product x) = 4 → unitsDigit x = 1 := by
  sorry

end units_digit_of_base_l1699_169913


namespace jessica_seashells_l1699_169941

theorem jessica_seashells (initial_seashells : ℕ) (given_seashells : ℕ) :
  initial_seashells = 8 →
  given_seashells = 6 →
  initial_seashells - given_seashells = 2 :=
by
  sorry

end jessica_seashells_l1699_169941


namespace cooking_and_weaving_count_l1699_169992

theorem cooking_and_weaving_count (total : ℕ) (yoga cooking weaving cooking_only cooking_and_yoga all : ℕ) 
  (h1 : yoga = 25)
  (h2 : cooking = 15)
  (h3 : weaving = 8)
  (h4 : cooking_only = 2)
  (h5 : cooking_and_yoga = 7)
  (h6 : all = 3) :
  cooking - (cooking_and_yoga + cooking_only) = 6 :=
by sorry

end cooking_and_weaving_count_l1699_169992


namespace grid_paths_7x3_l1699_169917

theorem grid_paths_7x3 : 
  let m : ℕ := 7  -- width of the grid
  let n : ℕ := 3  -- height of the grid
  (Nat.choose (m + n) n) = 120 := by
sorry

end grid_paths_7x3_l1699_169917


namespace aquarium_water_after_45_days_l1699_169904

/-- Calculates the remaining water in an aquarium after a given time period. -/
def remainingWater (initialVolume : ℝ) (lossRate : ℝ) (days : ℝ) : ℝ :=
  initialVolume - lossRate * days

/-- Theorem stating the remaining water volume in the aquarium after 45 days. -/
theorem aquarium_water_after_45_days :
  remainingWater 500 1.2 45 = 446 := by
  sorry

end aquarium_water_after_45_days_l1699_169904


namespace quadratic_inequality_equivalence_l1699_169967

theorem quadratic_inequality_equivalence (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x - 4*a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) := by
  sorry

end quadratic_inequality_equivalence_l1699_169967


namespace sock_pair_combinations_l1699_169911

/-- The number of ways to choose a pair of socks of different colors -/
def different_color_sock_pairs (white brown blue red : ℕ) : ℕ :=
  white * brown + white * blue + white * red +
  brown * blue + brown * red +
  blue * red

/-- Theorem stating the number of ways to choose a pair of socks of different colors
    given the specific quantities of each color -/
theorem sock_pair_combinations :
  different_color_sock_pairs 5 3 3 1 = 50 := by
  sorry

end sock_pair_combinations_l1699_169911


namespace xy_sum_product_l1699_169990

theorem xy_sum_product (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x*y + x + y = 7) : 
  x^2*y + x*y^2 = 196/25 := by
sorry

end xy_sum_product_l1699_169990


namespace matrix_sum_of_squares_l1699_169932

theorem matrix_sum_of_squares (x y z w : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  (B.transpose = 2 • B⁻¹) → x^2 + y^2 + z^2 + w^2 = 4 := by
  sorry

end matrix_sum_of_squares_l1699_169932


namespace binomial_coefficient_sum_l1699_169934

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 :=
by
  sorry

end binomial_coefficient_sum_l1699_169934


namespace reflection_about_y_eq_neg_x_l1699_169916

def reflect_point (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

theorem reflection_about_y_eq_neg_x (x y : ℝ) :
  reflect_point 4 (-3) = (3, -4) := by
  sorry

end reflection_about_y_eq_neg_x_l1699_169916


namespace parabola_focus_focus_of_specific_parabola_l1699_169968

/-- The focus of a parabola y = ax^2 + k is at (0, k - 1/(4a)) when a ≠ 0 -/
theorem parabola_focus (a k : ℝ) (ha : a ≠ 0) :
  let f : ℝ × ℝ := (0, k - 1 / (4 * a))
  ∀ x y : ℝ, y = a * x^2 + k → (x - f.1)^2 + (y - f.2)^2 = (y - k + 1 / (4 * a))^2 / (4 * a^2) :=
sorry

/-- The focus of the parabola y = -2x^2 + 4 is at (0, 33/8) -/
theorem focus_of_specific_parabola :
  let f : ℝ × ℝ := (0, 33/8)
  ∀ x y : ℝ, y = -2 * x^2 + 4 → (x - f.1)^2 + (y - f.2)^2 = (y - 4 + 1/8)^2 / 16 :=
sorry

end parabola_focus_focus_of_specific_parabola_l1699_169968


namespace prime_sum_equality_l1699_169977

theorem prime_sum_equality (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (h_sum : (Finset.range q).sum (λ i => p ^ (i + 1)) = (Finset.range p).sum (λ i => q ^ (i + 1))) : 
  p = q := by
sorry

end prime_sum_equality_l1699_169977


namespace owls_on_fence_l1699_169948

theorem owls_on_fence (initial_owls joining_owls : ℕ) :
  initial_owls = 12 → joining_owls = 7 → initial_owls + joining_owls = 19 := by
  sorry

end owls_on_fence_l1699_169948


namespace xenia_earnings_l1699_169905

/-- Xenia's work and earnings over two weeks -/
theorem xenia_earnings 
  (hours_week1 : ℕ) 
  (hours_week2 : ℕ) 
  (wage : ℚ) 
  (extra_earnings : ℚ) 
  (h1 : hours_week1 = 12)
  (h2 : hours_week2 = 20)
  (h3 : extra_earnings = 36)
  (h4 : wage * (hours_week2 - hours_week1) = extra_earnings) :
  wage * (hours_week1 + hours_week2) = 144 :=
sorry

end xenia_earnings_l1699_169905


namespace subtract_negative_l1699_169942

theorem subtract_negative : 2 - (-3) = 5 := by
  sorry

end subtract_negative_l1699_169942


namespace p_minus_m_equals_2010_l1699_169919

-- Define the set of positive integers
def PositiveInt : Set ℕ := {n : ℕ | n > 0}

-- Define set M
def M : Set ℕ := {x ∈ PositiveInt | 1 ≤ x ∧ x ≤ 2009}

-- Define set P
def P : Set ℕ := {y ∈ PositiveInt | 2 ≤ y ∧ y ≤ 2010}

-- Define the set difference operation
def SetDifference (A B : Set ℕ) : Set ℕ := {x ∈ A | x ∉ B}

-- Theorem statement
theorem p_minus_m_equals_2010 : SetDifference P M = {2010} := by
  sorry

end p_minus_m_equals_2010_l1699_169919


namespace original_price_correct_l1699_169981

/-- The original price of water bottles that satisfies the given conditions --/
def original_price : ℝ :=
  let number_of_bottles : ℕ := 60
  let reduced_price : ℝ := 1.85
  let shortfall : ℝ := 9
  2

theorem original_price_correct :
  let number_of_bottles : ℕ := 60
  let reduced_price : ℝ := 1.85
  let shortfall : ℝ := 9
  (number_of_bottles : ℝ) * original_price = 
    (number_of_bottles : ℝ) * reduced_price + shortfall :=
by
  sorry

#eval original_price

end original_price_correct_l1699_169981


namespace complex_fraction_equality_l1699_169998

/-- Given z = 1 + i and (z^2 + az + b) / (z^2 - z + 1) = 1 - i, where a and b are real numbers, 
    then a = -1 and b = 2. -/
theorem complex_fraction_equality (a b : ℝ) : 
  let z : ℂ := 1 + I
  ((z^2 + a*z + b) / (z^2 - z + 1) = 1 - I) → (a = -1 ∧ b = 2) := by
sorry


end complex_fraction_equality_l1699_169998


namespace min_value_parallel_vectors_l1699_169982

/-- Given two vectors a and b, where a is parallel to b, 
    prove that the minimum value of 3/x + 2/y is 8 -/
theorem min_value_parallel_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, y - 1)
  (∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1) →  -- parallelism condition
  (3 / x + 2 / y) ≥ 8 ∧ 
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 2 / y₀ = 8) :=
by sorry

end min_value_parallel_vectors_l1699_169982


namespace bench_press_changes_l1699_169989

/-- Calculates the final bench press weight after a series of changes -/
def final_bench_press (initial_weight : ℝ) : ℝ :=
  let after_injury := initial_weight * (1 - 0.8)
  let after_recovery := after_injury * (1 + 0.6)
  let after_setback := after_recovery * (1 - 0.2)
  let final_weight := after_setback * 3
  final_weight

/-- Theorem stating that the final bench press weight is 384 pounds -/
theorem bench_press_changes (initial_weight : ℝ) 
  (h : initial_weight = 500) : 
  final_bench_press initial_weight = 384 := by
  sorry

#eval final_bench_press 500

end bench_press_changes_l1699_169989


namespace sum_interior_angles_regular_polygon_l1699_169958

/-- 
For a regular polygon where each exterior angle measures 30 degrees, 
the sum of the measures of the interior angles is 1800 degrees.
-/
theorem sum_interior_angles_regular_polygon : 
  ∀ (n : ℕ) (exterior_angle : ℝ),
  n > 2 → 
  exterior_angle = 30 →
  n * exterior_angle = 360 →
  (n - 2) * 180 = 1800 :=
by
  sorry

end sum_interior_angles_regular_polygon_l1699_169958


namespace johns_weekly_sleep_l1699_169997

/-- The total sleep John got in a week given specific sleep patterns --/
def totalSleepInWeek (daysWithLowSleep : ℕ) (hoursLowSleep : ℕ) 
  (recommendedSleep : ℕ) (percentageNormalSleep : ℚ) : ℚ :=
  (daysWithLowSleep * hoursLowSleep : ℚ) + 
  ((7 - daysWithLowSleep) * (recommendedSleep * percentageNormalSleep))

/-- Theorem stating that John's total sleep for the week is 30 hours --/
theorem johns_weekly_sleep : 
  totalSleepInWeek 2 3 8 (60 / 100) = 30 := by
  sorry

end johns_weekly_sleep_l1699_169997


namespace rectangle_perimeter_l1699_169983

theorem rectangle_perimeter (area : ℝ) (width : ℝ) (length : ℝ) :
  area = 500 →
  length = 2 * width →
  area = length * width →
  2 * (length + width) = 30 * Real.sqrt 10 :=
by sorry

end rectangle_perimeter_l1699_169983


namespace britney_tea_service_l1699_169902

/-- Given a total number of cups and cups per person, calculate the number of people served -/
def people_served (total_cups : ℕ) (cups_per_person : ℕ) : ℕ :=
  total_cups / cups_per_person

/-- Theorem: Britney served 5 people given the conditions -/
theorem britney_tea_service :
  people_served 10 2 = 5 := by
  sorry

end britney_tea_service_l1699_169902


namespace rectangle_area_l1699_169944

theorem rectangle_area (d : ℝ) (w : ℝ) (h : w > 0) : 
  (3 * w)^2 + w^2 = d^2 → 3 * w^2 = 3 * d^2 / 10 :=
sorry

end rectangle_area_l1699_169944


namespace collinear_points_solution_l1699_169955

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The theorem states that if points A(a,2), B(5,1), and C(-4,2a) are collinear, 
    then a = 5 ± √21 -/
theorem collinear_points_solution (a : ℝ) :
  collinear a 2 5 1 (-4) (2*a) → a = 5 + Real.sqrt 21 ∨ a = 5 - Real.sqrt 21 :=
by sorry

end collinear_points_solution_l1699_169955


namespace ohara_triple_49_16_l1699_169933

/-- Definition of O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: If (49, 16, x) is an O'Hara triple, then x = 11 -/
theorem ohara_triple_49_16 (x : ℕ) :
  is_ohara_triple 49 16 x → x = 11 := by
  sorry

end ohara_triple_49_16_l1699_169933


namespace car_speed_problem_l1699_169951

theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 75 →
  average_speed = 82.5 →
  (speed_second_hour + (average_speed * 2 - speed_second_hour)) / 2 = average_speed →
  average_speed * 2 - speed_second_hour = 90 :=
by
  sorry

end car_speed_problem_l1699_169951


namespace inscribed_triangle_angle_l1699_169908

theorem inscribed_triangle_angle (x : ℝ) : 
  let arc_DE := x + 90
  let arc_EF := 2*x + 15
  let arc_FD := 3*x - 30
  -- Sum of arcs is 360°
  arc_DE + arc_EF + arc_FD = 360 →
  -- Triangle inscribed in circle
  -- Interior angles are half the corresponding arc measures
  ∃ (angle : ℝ), (angle = arc_EF / 2 ∨ angle = arc_FD / 2 ∨ angle = arc_DE / 2) ∧ 
  (angle ≥ 68.5 ∧ angle ≤ 69.5) :=
by
  sorry

end inscribed_triangle_angle_l1699_169908
