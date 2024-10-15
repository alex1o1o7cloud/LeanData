import Mathlib

namespace NUMINAMATH_CALUDE_book_reading_ratio_l870_87012

/-- The number of books William read last month -/
def william_last_month : ℕ := 6

/-- The number of books Brad read last month -/
def brad_last_month : ℕ := 3 * william_last_month

/-- The number of books Brad read this month -/
def brad_this_month : ℕ := 8

/-- The difference in total books read between William and Brad over two months -/
def difference_total : ℕ := 4

/-- The number of books William read this month -/
def william_this_month : ℕ := william_last_month + brad_last_month + brad_this_month + difference_total - (brad_last_month + brad_this_month)

theorem book_reading_ratio : 
  william_this_month / brad_this_month = 3 ∧ william_this_month % brad_this_month = 0 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_ratio_l870_87012


namespace NUMINAMATH_CALUDE_managers_salary_managers_salary_proof_l870_87016

/-- The manager's salary problem -/
theorem managers_salary (num_employees : ℕ) (initial_avg_salary : ℕ) (salary_increase : ℕ) : ℕ :=
  let total_initial_salary := num_employees * initial_avg_salary
  let new_avg_salary := initial_avg_salary + salary_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_initial_salary

/-- Proof of the manager's salary -/
theorem managers_salary_proof :
  managers_salary 50 2500 1500 = 79000 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_managers_salary_proof_l870_87016


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l870_87078

theorem trigonometric_system_solution (θ : ℝ) (a b : ℝ) 
  (eq1 : Real.sin θ + Real.cos θ = a)
  (eq2 : Real.sin θ - Real.cos θ = b)
  (eq3 : Real.sin θ * Real.sin θ - Real.cos θ * Real.cos θ - Real.sin θ = -b * b) :
  ((a = Real.sqrt 7 / 2 ∧ b = 1 / 2) ∨
   (a = -Real.sqrt 7 / 2 ∧ b = 1 / 2) ∨
   (a = 1 ∧ b = -1) ∨
   (a = -1 ∧ b = 1)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_system_solution_l870_87078


namespace NUMINAMATH_CALUDE_price_adjustment_solution_l870_87048

/-- Selling prices before and after adjustment in places A and B -/
structure Prices where
  a_before : ℝ
  b_before : ℝ
  a_after : ℝ
  b_after : ℝ

/-- Conditions of the price adjustment problem -/
def PriceAdjustmentConditions (p : Prices) : Prop :=
  p.a_after = p.a_before * 1.1 ∧
  p.b_after = p.b_before - 5 ∧
  p.b_before - p.a_before = 10 ∧
  p.b_after - p.a_after = 1

/-- Theorem stating the solution to the price adjustment problem -/
theorem price_adjustment_solution :
  ∃ (p : Prices), PriceAdjustmentConditions p ∧ p.a_before = 40 ∧ p.b_before = 50 := by
  sorry

end NUMINAMATH_CALUDE_price_adjustment_solution_l870_87048


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l870_87007

/-- Given a geometric sequence {a_n} with a_1 = 1/8 and a_4 = -1, prove that the common ratio q is -2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a1 : a 1 = 1/8) 
  (h_a4 : a 4 = -1) 
  (q : ℚ) : 
  q = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l870_87007


namespace NUMINAMATH_CALUDE_cube_volume_partition_l870_87041

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  sideLength : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Given a cube and a plane passing through the midpoint of one edge and two points
    on opposite edges with the ratio 1:7 from the vertices, the smaller part of the
    volume separated by this plane is 25/192 of the cube's volume. -/
theorem cube_volume_partition (cube : Cube) (plane : Plane)
  (h1 : plane.a * (cube.sideLength / 2) + plane.b * 0 + plane.c * 0 = plane.d)
  (h2 : plane.a * 0 + plane.b * 0 + plane.c * (cube.sideLength / 8) = plane.d)
  (h3 : plane.a * cube.sideLength + plane.b * cube.sideLength + plane.c * (cube.sideLength / 8) = plane.d) :
  ∃ (smallerVolume : ℝ), smallerVolume = (25 / 192) * cube.sideLength ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_partition_l870_87041


namespace NUMINAMATH_CALUDE_discrete_rv_distribution_l870_87061

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  p₁ : ℝ
  h₁ : x₂ > x₁
  h₂ : p₁ = 0.6
  h₃ : p₁ * x₁ + (1 - p₁) * x₂ = 1.4  -- Expected value
  h₄ : p₁ * (x₁ - 1.4)^2 + (1 - p₁) * (x₂ - 1.4)^2 = 0.24  -- Variance

/-- The probability distribution of the discrete random variable -/
def probability_distribution (X : DiscreteRV) : Prop :=
  X.x₁ = 1 ∧ X.x₂ = 2

theorem discrete_rv_distribution (X : DiscreteRV) :
  probability_distribution X := by
  sorry

end NUMINAMATH_CALUDE_discrete_rv_distribution_l870_87061


namespace NUMINAMATH_CALUDE_wall_passing_skill_l870_87005

theorem wall_passing_skill (n : ℕ) (h : 8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n))) :
  n = 63 := by
  sorry

end NUMINAMATH_CALUDE_wall_passing_skill_l870_87005


namespace NUMINAMATH_CALUDE_square_difference_l870_87033

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x - 2) * (x + 2) = 9797 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l870_87033


namespace NUMINAMATH_CALUDE_whale_consumption_increase_l870_87063

/-- Represents the whale's plankton consumption pattern -/
structure WhaleConsumption where
  initial : ℕ  -- Initial consumption in the first hour
  increase : ℕ  -- Constant increase each hour after the first
  duration : ℕ  -- Duration of the feeding frenzy in hours
  total : ℕ     -- Total accumulated consumption
  sixth_hour : ℕ -- Consumption in the sixth hour

/-- Theorem stating the whale's consumption increase -/
theorem whale_consumption_increase 
  (w : WhaleConsumption) 
  (h1 : w.duration = 9)
  (h2 : w.total = 450)
  (h3 : w.sixth_hour = 54)
  (h4 : w.initial + 5 * w.increase = w.sixth_hour)
  (h5 : (w.duration : ℕ) * w.initial + 
        (w.duration * (w.duration - 1) / 2) * w.increase = w.total) : 
  w.increase = 4 := by
  sorry

end NUMINAMATH_CALUDE_whale_consumption_increase_l870_87063


namespace NUMINAMATH_CALUDE_circle_equations_l870_87020

-- Define points
def A : ℝ × ℝ := (6, 5)
def B : ℝ × ℝ := (0, 1)
def P : ℝ × ℝ := (-2, 4)
def Q : ℝ × ℝ := (3, -1)

-- Define the line equation for the center
def center_line (x y : ℝ) : Prop := 3 * x + 10 * y + 9 = 0

-- Define the chord length on x-axis
def chord_length : ℝ := 6

-- Define circle equations
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 13
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 25

theorem circle_equations :
  ∃ (C : ℝ × ℝ),
    (center_line C.1 C.2) ∧
    (circle1 A.1 A.2 ∨ circle2 A.1 A.2) ∧
    (circle1 B.1 B.2 ∨ circle2 B.1 B.2) ∧
    (circle1 P.1 P.2 ∨ circle2 P.1 P.2) ∧
    (circle1 Q.1 Q.2 ∨ circle2 Q.1 Q.2) ∧
    (∃ (x1 x2 : ℝ), x2 - x1 = chord_length ∧
      ((circle1 x1 0 ∧ circle1 x2 0) ∨ (circle2 x1 0 ∧ circle2 x2 0))) :=
by sorry


end NUMINAMATH_CALUDE_circle_equations_l870_87020


namespace NUMINAMATH_CALUDE_absolute_value_of_five_minus_e_l870_87093

-- Define e as a constant approximation
def e : ℝ := 2.71828

-- State the theorem
theorem absolute_value_of_five_minus_e : |5 - e| = 2.28172 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_five_minus_e_l870_87093


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l870_87022

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | |x - 1| ≤ 2}

theorem intersection_of_P_and_Q : P ∩ Q = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l870_87022


namespace NUMINAMATH_CALUDE_probability_equals_fraction_l870_87085

def num_forks : ℕ := 8
def num_spoons : ℕ := 5
def num_knives : ℕ := 7
def total_silverware : ℕ := num_forks + num_spoons + num_knives
def pieces_removed : ℕ := 4

def probability_two_forks_one_spoon_one_knife : ℚ :=
  (Nat.choose num_forks 2 * Nat.choose num_spoons 1 * Nat.choose num_knives 1) /
  Nat.choose total_silverware pieces_removed

theorem probability_equals_fraction :
  probability_two_forks_one_spoon_one_knife = 196 / 969 := by
  sorry

end NUMINAMATH_CALUDE_probability_equals_fraction_l870_87085


namespace NUMINAMATH_CALUDE_point_on_curve_limit_at_one_l870_87030

/-- The curve y = x² + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The point (1, 2) lies on the curve -/
theorem point_on_curve : f 1 = 2 := by sorry

/-- The limit of Δy/Δx as Δx approaches 0 at x = 1 is 2 -/
theorem limit_at_one : 
  ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 
    0 < |h| → |h| < δ → |(f (1 + h) - f 1) / h - 2| < ε := by sorry

end NUMINAMATH_CALUDE_point_on_curve_limit_at_one_l870_87030


namespace NUMINAMATH_CALUDE_matt_points_l870_87000

/-- Calculates the total points scored in basketball given the number of successful 2-point and 3-point shots -/
def total_points (two_point_shots : ℕ) (three_point_shots : ℕ) : ℕ :=
  2 * two_point_shots + 3 * three_point_shots

/-- Theorem stating that four 2-point shots and two 3-point shots result in 14 points -/
theorem matt_points : total_points 4 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_matt_points_l870_87000


namespace NUMINAMATH_CALUDE_circumcircumcircumcoronene_tilings_l870_87065

/-- Represents a tiling of a hexagon with edge length n using diamonds of side 1 -/
def HexagonTiling (n : ℕ) : Type := Unit

/-- The number of valid tilings for a hexagon with edge length n -/
def count_tilings (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of tilings for a hexagon with edge length 5 is 267227532 -/
theorem circumcircumcircumcoronene_tilings :
  count_tilings 5 = 267227532 := by sorry

end NUMINAMATH_CALUDE_circumcircumcircumcoronene_tilings_l870_87065


namespace NUMINAMATH_CALUDE_min_value_y_l870_87073

theorem min_value_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y * Real.log y = Real.exp (2 * x) - y * Real.log (2 * x)) : 
  (∀ z, z > 0 → z * Real.log z = Real.exp (2 * x) - z * Real.log (2 * x) → y ≤ z) ∧ y = Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_y_l870_87073


namespace NUMINAMATH_CALUDE_larger_rectangle_area_larger_rectangle_area_proof_l870_87008

theorem larger_rectangle_area : ℝ → ℝ → ℝ → Prop :=
  fun (small_square_area : ℝ) (small_rect_length : ℝ) (small_rect_width : ℝ) =>
    small_square_area = 25 ∧
    small_rect_length = 3 * Real.sqrt small_square_area ∧
    small_rect_width = Real.sqrt small_square_area ∧
    2 * small_rect_width = small_rect_length →
    small_rect_length * (2 * small_rect_width) = 150

-- The proof goes here
theorem larger_rectangle_area_proof :
  ∃ (small_square_area small_rect_length small_rect_width : ℝ),
    larger_rectangle_area small_square_area small_rect_length small_rect_width :=
by
  sorry

end NUMINAMATH_CALUDE_larger_rectangle_area_larger_rectangle_area_proof_l870_87008


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l870_87046

theorem quadratic_distinct_roots (n : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + n*x + 9 = 0 ∧ y^2 + n*y + 9 = 0) ↔ 
  (n < -6 ∨ n > 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l870_87046


namespace NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l870_87009

/-- Given a geometric sequence {a_n} with common ratio q ≠ 1,
    if a_4, a_3, a_5 form an arithmetic sequence,
    then (a_3 + a_4) / (a_2 + a_3) = -2 -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n : ℕ, a (n + 1) = q * a n) →
  2 * a 3 = a 4 + a 5 →
  (a 3 + a 4) / (a 2 + a 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l870_87009


namespace NUMINAMATH_CALUDE_circle_equation_l870_87052

-- Define the center and radius of the circle
def center : ℝ × ℝ := (2, -1)
def radius : ℝ := 4

-- State the theorem
theorem circle_equation :
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ 
  (x - 2)^2 + (y + 1)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l870_87052


namespace NUMINAMATH_CALUDE_abs_difference_equals_seven_l870_87072

theorem abs_difference_equals_seven (a b : ℝ) 
  (ha : |a| = 4) 
  (hb : |b| = 3) 
  (hab : a * b < 0) : 
  |a - b| = 7 := by
sorry

end NUMINAMATH_CALUDE_abs_difference_equals_seven_l870_87072


namespace NUMINAMATH_CALUDE_brothers_money_distribution_l870_87097

/-- Represents the money distribution among four brothers -/
structure MoneyDistribution where
  john : ℕ
  william : ℕ
  charles : ℕ
  thomas : ℕ

/-- Checks if the given money distribution satisfies all conditions -/
def satisfies_conditions (d : MoneyDistribution) : Prop :=
  d.john + 2 = d.william - 2 ∧
  d.john + 2 = 2 * d.charles ∧
  d.john + 2 = d.thomas / 2 ∧
  d.john + d.william + d.charles + d.thomas = 45

/-- Checks if the given money distribution can be represented with 6 coins -/
def can_be_represented_with_six_coins (d : MoneyDistribution) : Prop :=
  ∃ (j1 j2 w1 w2 c t : ℕ),
    j1 + j2 = d.john ∧
    w1 + w2 = d.william ∧
    c = d.charles ∧
    t = d.thomas

/-- The main theorem stating the unique solution for the brothers' money distribution -/
theorem brothers_money_distribution :
  ∃! (d : MoneyDistribution),
    satisfies_conditions d ∧
    can_be_represented_with_six_coins d ∧
    d.john = 8 ∧ d.william = 12 ∧ d.charles = 5 ∧ d.thomas = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_money_distribution_l870_87097


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l870_87083

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 15 → difference = 1 → friend_cost = total / 2 + difference / 2 → friend_cost = 8 := by
sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l870_87083


namespace NUMINAMATH_CALUDE_division_remainder_problem_l870_87089

theorem division_remainder_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) 
  (h1 : dividend = 161)
  (h2 : divisor = 16)
  (h3 : quotient = 10)
  (h4 : dividend = divisor * quotient + (dividend % divisor)) :
  dividend % divisor = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l870_87089


namespace NUMINAMATH_CALUDE_emily_garden_seeds_l870_87028

theorem emily_garden_seeds (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 41)
  (h2 : small_gardens = 3)
  (h3 : seeds_per_small_garden = 4) :
  total_seeds - (small_gardens * seeds_per_small_garden) = 29 := by
  sorry

end NUMINAMATH_CALUDE_emily_garden_seeds_l870_87028


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l870_87066

theorem quadratic_roots_property (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) → 
  (3 * e^2 + 4 * e - 7 = 0) → 
  (d - 2) * (e - 2) = 13/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l870_87066


namespace NUMINAMATH_CALUDE_subset_condition_l870_87003

def P : Set ℝ := {x | x^2 ≠ 4}
def Q (a : ℝ) : Set ℝ := {x | a * x = 4}

theorem subset_condition (a : ℝ) : Q a ⊆ P ↔ a = 0 ∨ a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l870_87003


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l870_87079

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - 9*x^2 + 24*x > 0 ↔ (0 < x ∧ x < 3) ∨ (x > 8) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l870_87079


namespace NUMINAMATH_CALUDE_waiting_time_problem_l870_87043

/-- Proves that the waiting time for the man to catch up is 25 minutes -/
theorem waiting_time_problem (man_speed woman_speed : ℚ) (stop_time : ℚ) :
  man_speed = 5 →
  woman_speed = 25 →
  stop_time = 5 / 60 →
  let distance_traveled := woman_speed * stop_time
  let catch_up_time := distance_traveled / man_speed
  catch_up_time = 25 / 60 := by sorry

end NUMINAMATH_CALUDE_waiting_time_problem_l870_87043


namespace NUMINAMATH_CALUDE_smallest_c_plus_d_l870_87090

theorem smallest_c_plus_d : ∃ (c d : ℕ+), 
  (3^6 * 7^2 : ℕ) = c^(d:ℕ) ∧ 
  (∀ (c' d' : ℕ+), (3^6 * 7^2 : ℕ) = c'^(d':ℕ) → c + d ≤ c' + d') ∧
  c + d = 1325 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_plus_d_l870_87090


namespace NUMINAMATH_CALUDE_ordering_abc_l870_87042

theorem ordering_abc (a b c : ℝ) : 
  a = 31/32 → b = Real.cos (1/4) → c = 4 * Real.sin (1/4) → c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l870_87042


namespace NUMINAMATH_CALUDE_temperature_difference_l870_87082

def lowest_temp : ℤ := -4
def highest_temp : ℤ := 5

theorem temperature_difference : highest_temp - lowest_temp = 9 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l870_87082


namespace NUMINAMATH_CALUDE_cuboid_gluing_theorem_l870_87024

/-- A cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ+
  width : ℕ+
  height : ℕ+
  different_dimensions : length ≠ width ∧ width ≠ height ∧ height ≠ length

/-- The volume of a cuboid -/
def volume (c : Cuboid) : ℕ := c.length * c.width * c.height

/-- Two cuboids can be glued if they share a face -/
def can_be_glued (c1 c2 : Cuboid) : Prop :=
  (c1.length = c2.length ∧ c1.width = c2.width) ∨
  (c1.length = c2.length ∧ c1.height = c2.height) ∨
  (c1.width = c2.width ∧ c1.height = c2.height)

/-- The resulting cuboid after gluing two cuboids -/
def glued_cuboid (c1 c2 : Cuboid) : Cuboid :=
  if c1.length = c2.length ∧ c1.width = c2.width then
    ⟨c1.length, c1.width, c1.height + c2.height, sorry⟩
  else if c1.length = c2.length ∧ c1.height = c2.height then
    ⟨c1.length, c1.width + c2.width, c1.height, sorry⟩
  else
    ⟨c1.length + c2.length, c1.width, c1.height, sorry⟩

theorem cuboid_gluing_theorem (c1 c2 : Cuboid) :
  volume c1 = 12 →
  volume c2 = 30 →
  can_be_glued c1 c2 →
  let c := glued_cuboid c1 c2
  (c.length = 1 ∧ c.width = 2 ∧ c.height = 21) ∨
  (c.length = 1 ∧ c.width = 3 ∧ c.height = 14) ∨
  (c.length = 1 ∧ c.width = 6 ∧ c.height = 7) :=
by sorry

end NUMINAMATH_CALUDE_cuboid_gluing_theorem_l870_87024


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l870_87026

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 9

-- Define the line l
def line_l (x y m : ℝ) : Prop := x - y + m = 0

-- Define the perpendicularity condition
def perpendicular (xa ya xb yb xc yc : ℝ) : Prop :=
  (xa - xc) * (xb - xc) + (ya - yc) * (yb - yc) = 0

-- State the theorem
theorem circle_intersection_theorem (m : ℝ) :
  (∃ (xa ya xb yb : ℝ),
    circle_C xa ya ∧ circle_C xb yb ∧
    line_l xa ya m ∧ line_l xb yb m ∧
    perpendicular xa ya xb yb 3 1) →
  m = 1 ∨ m = -5 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l870_87026


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_quotient_l870_87064

theorem two_numbers_sum_and_quotient (x y : ℝ) : 
  x > 0 → y > 0 → x + y = 432 → y / x = 5 → x = 72 ∧ y = 360 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_quotient_l870_87064


namespace NUMINAMATH_CALUDE_triangle_side_range_l870_87095

theorem triangle_side_range (a b c : ℝ) : 
  (|a - 3| + (b - 7)^2 = 0) →
  (c ≥ a ∧ c ≥ b) →
  (c < a + b) →
  (7 ≤ c ∧ c < 10) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l870_87095


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l870_87080

/-- Calculates the gain percent when an item is bought and sold at given prices. -/
def gainPercent (costPrice sellingPrice : ℚ) : ℚ :=
  ((sellingPrice - costPrice) / costPrice) * 100

/-- Theorem: The gain percent is 50% when a cycle is bought for Rs. 900 and sold for Rs. 1350. -/
theorem cycle_gain_percent :
  let costPrice : ℚ := 900
  let sellingPrice : ℚ := 1350
  gainPercent costPrice sellingPrice = 50 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l870_87080


namespace NUMINAMATH_CALUDE_min_cans_correct_l870_87094

/-- The volume of soda in a single can (in ounces) -/
def can_volume : ℝ := 15

/-- The conversion factor from liters to ounces -/
def liter_to_ounce : ℝ := 33.814

/-- The required volume of soda (in liters) -/
def required_volume : ℝ := 3.8

/-- The minimum number of cans required to provide at least the required volume of soda -/
def min_cans : ℕ := 9

/-- Theorem stating that the minimum number of cans required to provide at least
    the required volume of soda is 9 -/
theorem min_cans_correct :
  ∀ n : ℕ, (n : ℝ) * can_volume ≥ required_volume * liter_to_ounce → n ≥ min_cans :=
by sorry

end NUMINAMATH_CALUDE_min_cans_correct_l870_87094


namespace NUMINAMATH_CALUDE_special_triangle_third_side_l870_87018

/-- Triangle sides satisfy the given conditions -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  side_condition : Real.sqrt (a - 9) + (b - 2)^2 = 0
  c_odd : ∃ (k : ℤ), c = 2 * k + 1

/-- The third side of the special triangle is 9 -/
theorem special_triangle_third_side (t : SpecialTriangle) : t.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_third_side_l870_87018


namespace NUMINAMATH_CALUDE_goods_train_passing_time_l870_87027

/-- The time taken for a goods train to pass a man in an opposing train -/
theorem goods_train_passing_time (man_speed goods_speed : ℝ) (goods_length : ℝ) : 
  man_speed = 70 →
  goods_speed = 42 →
  goods_length = 280 →
  ∃ t : ℝ, t > 0 ∧ t < 10 ∧ t * (man_speed + goods_speed) * (1000 / 3600) = goods_length :=
by sorry

end NUMINAMATH_CALUDE_goods_train_passing_time_l870_87027


namespace NUMINAMATH_CALUDE_shorter_base_length_l870_87047

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midpoint_segment : ℝ

/-- The trapezoid satisfies the given conditions -/
def trapezoid_conditions (t : Trapezoid) : Prop :=
  t.long_base = 125 ∧ t.midpoint_segment = 5

/-- Theorem: In a trapezoid satisfying the given conditions, the shorter base is 115 -/
theorem shorter_base_length (t : Trapezoid) (h : trapezoid_conditions t) : 
  t.short_base = 115 := by
  sorry

#check shorter_base_length

end NUMINAMATH_CALUDE_shorter_base_length_l870_87047


namespace NUMINAMATH_CALUDE_bryce_raisins_l870_87087

theorem bryce_raisins : ∃ (bryce carter : ℕ), 
  bryce = carter + 8 ∧ 
  carter = bryce / 3 ∧ 
  bryce = 12 := by
sorry

end NUMINAMATH_CALUDE_bryce_raisins_l870_87087


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_specific_root_condition_l870_87038

/-- Represents a quadratic equation of the form x^2 + 2(m+1)x + m^2 - 1 = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 + 2*(m+1)*x + m^2 - 1 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  8*m + 8

/-- Condition for the roots of the quadratic equation -/
def root_condition (x₁ x₂ : ℝ) : Prop :=
  (x₁ - x₂)^2 = 16 - x₁*x₂

theorem quadratic_equation_properties (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ x₁ ≠ x₂) ↔ m ≥ -1 :=
sorry

theorem specific_root_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ 
   x₁ ≠ x₂ ∧ root_condition x₁ x₂) → m = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_specific_root_condition_l870_87038


namespace NUMINAMATH_CALUDE_matt_total_skips_l870_87077

/-- The number of skips per second -/
def skips_per_second : ℕ := 3

/-- The duration of jumping in minutes -/
def jump_duration : ℕ := 10

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Theorem: Given the conditions, Matt's total number of skips is 1800 -/
theorem matt_total_skips :
  skips_per_second * jump_duration * seconds_per_minute = 1800 := by
  sorry

end NUMINAMATH_CALUDE_matt_total_skips_l870_87077


namespace NUMINAMATH_CALUDE_rectangle_area_is_216_l870_87040

/-- Represents a rectangle with given properties -/
structure Rectangle where
  length : ℝ
  breadth : ℝ
  perimeterToBreadthRatio : ℝ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.breadth

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.breadth)

/-- Theorem: A rectangle with length 18 and perimeter to breadth ratio of 5 has an area of 216 -/
theorem rectangle_area_is_216 (r : Rectangle) 
    (h1 : r.length = 18)
    (h2 : r.perimeterToBreadthRatio = 5)
    (h3 : perimeter r / r.breadth = r.perimeterToBreadthRatio) : 
  area r = 216 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_is_216_l870_87040


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_cubic_eq_l870_87019

theorem x_eq_one_sufficient_not_necessary_for_cubic_eq :
  (∀ x : ℝ, x = 1 → x^3 - 2*x + 1 = 0) ∧
  (∃ x : ℝ, x ≠ 1 ∧ x^3 - 2*x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_cubic_eq_l870_87019


namespace NUMINAMATH_CALUDE_red_markers_count_l870_87035

/-- Given a total number of markers and a number of blue markers, 
    calculate the number of red markers. -/
def red_markers (total : ℝ) (blue : ℕ) : ℝ :=
  total - blue

/-- Prove that given 64.0 total markers and 23 blue markers, 
    the number of red markers is 41. -/
theorem red_markers_count : red_markers 64.0 23 = 41 := by
  sorry

end NUMINAMATH_CALUDE_red_markers_count_l870_87035


namespace NUMINAMATH_CALUDE_log_equation_solution_l870_87084

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_solution (a : ℝ) (h : log a - 2 * log 2 = 1) : a = 40 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l870_87084


namespace NUMINAMATH_CALUDE_coprime_powers_of_primes_l870_87062

def valid_n : Set ℕ := {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 18, 20, 24, 30, 42}

def is_power_of_prime (m : ℕ) : Prop :=
  ∃ p k, Nat.Prime p ∧ m = p ^ k

theorem coprime_powers_of_primes (n : ℕ) :
  (∀ m, 0 < m ∧ m < n ∧ Nat.Coprime m n → is_power_of_prime m) ↔ n ∈ valid_n := by
  sorry

end NUMINAMATH_CALUDE_coprime_powers_of_primes_l870_87062


namespace NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l870_87088

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π
  h_sum_angles : A + B + C = π
  h_area : S = (1/2) * b * c * Real.sin A

-- Theorem statement
theorem angle_measure_in_special_triangle (t : Triangle) 
  (h : (t.b + t.c)^2 - t.a^2 = 4 * Real.sqrt 3 * t.S) : 
  t.A = π/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l870_87088


namespace NUMINAMATH_CALUDE_binomial_18_9_l870_87096

theorem binomial_18_9 (h1 : Nat.choose 16 7 = 11440) 
                      (h2 : Nat.choose 16 8 = 12870) 
                      (h3 : Nat.choose 16 9 = 11440) : 
  Nat.choose 18 9 = 48620 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_9_l870_87096


namespace NUMINAMATH_CALUDE_fraction_order_l870_87086

theorem fraction_order : 
  (20 : ℚ) / 15 < 25 / 18 ∧ 25 / 18 < 23 / 16 ∧ 23 / 16 < 21 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l870_87086


namespace NUMINAMATH_CALUDE_trinomial_square_equality_l870_87099

theorem trinomial_square_equality : 
  15^2 + 3^2 + 1^2 + 2*(15*3) + 2*(15*1) + 2*(3*1) = (15 + 3 + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_trinomial_square_equality_l870_87099


namespace NUMINAMATH_CALUDE_min_value_of_function_l870_87055

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  let y := x + 4 / (x - 1)
  (∀ z, z > 1 → y ≤ z + 4 / (z - 1)) ∧ y = 5 ↔ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l870_87055


namespace NUMINAMATH_CALUDE_max_distance_between_circles_l870_87069

/-- Circle C₁ with equation x² + y² + 2x + 8y - 8 = 0 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

/-- Circle C₂ with equation x² + y² - 4x - 5 = 0 -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

/-- The maximum distance between any point on C₁ and any point on C₂ is 13 -/
theorem max_distance_between_circles :
  ∃ (m₁ m₂ n₁ n₂ : ℝ), C₁ m₁ m₂ ∧ C₂ n₁ n₂ ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ → C₂ x₂ y₂ →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ Real.sqrt ((m₁ - n₁)^2 + (m₂ - n₂)^2) ∧
  Real.sqrt ((m₁ - n₁)^2 + (m₂ - n₂)^2) = 13 :=
sorry

end NUMINAMATH_CALUDE_max_distance_between_circles_l870_87069


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l870_87057

theorem algebraic_expression_value (x y : ℝ) (h : x - y - 3 = 0) :
  x^2 - y^2 - 6*y = 9 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l870_87057


namespace NUMINAMATH_CALUDE_polynomial_expansion_problem_l870_87049

theorem polynomial_expansion_problem (p q : ℝ) : 
  p > 0 ∧ q > 0 ∧ p + 2*q = 1 ∧ 
  (45 : ℝ) * p^8 * q^2 = (120 : ℝ) * p^7 * q^3 → 
  p = 4/7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_problem_l870_87049


namespace NUMINAMATH_CALUDE_parallel_planes_transitive_perpendicular_to_line_parallel_l870_87023

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (parallel_to_line : Plane → Line → Prop)
variable (perpendicular_to_line : Plane → Line → Prop)
variable (coincident : Plane → Plane → Prop)

-- Theorem 1
theorem parallel_planes_transitive (α β γ : Plane) :
  parallel α γ → parallel β γ → ¬coincident α β → parallel α β := by sorry

-- Theorem 2
theorem perpendicular_to_line_parallel (α β : Plane) (l : Line) :
  perpendicular_to_line α l → perpendicular_to_line β l → ¬coincident α β → parallel α β := by sorry

end NUMINAMATH_CALUDE_parallel_planes_transitive_perpendicular_to_line_parallel_l870_87023


namespace NUMINAMATH_CALUDE_point_on_bisector_value_l870_87051

/-- A point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The bisector of the angle between the two coordinate axes in the first and third quadrants -/
def isOnBisector (p : Point) : Prop :=
  p.x = p.y

/-- The theorem statement -/
theorem point_on_bisector_value (a : ℝ) :
  let A : Point := ⟨a, 2*a + 3⟩
  isOnBisector A → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_bisector_value_l870_87051


namespace NUMINAMATH_CALUDE_cube_diff_divisibility_l870_87053

theorem cube_diff_divisibility (m n k : ℕ) (hm : Odd m) (hn : Odd n) (hk : k > 0) :
  (2^k ∣ m^3 - n^3) ↔ (2^k ∣ m - n) := by
  sorry

end NUMINAMATH_CALUDE_cube_diff_divisibility_l870_87053


namespace NUMINAMATH_CALUDE_log_inequality_l870_87002

theorem log_inequality (x : ℝ) (h : 2 * Real.log x / Real.log 2 - 1 < 0) : 0 < x ∧ x < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l870_87002


namespace NUMINAMATH_CALUDE_f_properties_l870_87006

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom cond1 : ∀ x, f (10 + x) = f (10 - x)
axiom cond2 : ∀ x, f (20 - x) = -f (20 + x)

-- Define oddness
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

-- Define periodicity
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

-- Theorem statement
theorem f_properties : is_odd f ∧ is_periodic f 20 :=
  sorry

end NUMINAMATH_CALUDE_f_properties_l870_87006


namespace NUMINAMATH_CALUDE_nines_squared_zeros_l870_87054

theorem nines_squared_zeros (n : ℕ) :
  ∃ m : ℕ, (10^9 - 1)^2 = m * 10^8 ∧ m % 10 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_nines_squared_zeros_l870_87054


namespace NUMINAMATH_CALUDE_exists_floating_polyhedron_with_properties_l870_87029

/-- A convex polyhedron floating in water -/
structure FloatingPolyhedron where
  volume : ℝ
  surfaceArea : ℝ
  submergedVolume : ℝ
  surfaceAreaAboveWater : ℝ
  volume_pos : 0 < volume
  surfaceArea_pos : 0 < surfaceArea
  submergedVolume_le_volume : submergedVolume ≤ volume
  surfaceAreaAboveWater_le_surfaceArea : surfaceAreaAboveWater ≤ surfaceArea

/-- Theorem stating the existence of a floating polyhedron with specific properties -/
theorem exists_floating_polyhedron_with_properties :
  ∀ ε > 0, ∃ (P : FloatingPolyhedron),
    P.submergedVolume / P.volume > 1 - ε ∧
    P.surfaceAreaAboveWater / P.surfaceArea > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_exists_floating_polyhedron_with_properties_l870_87029


namespace NUMINAMATH_CALUDE_new_person_weight_l870_87067

/-- Given a group of people where:
  * There are initially 4 persons
  * One person weighing 70 kg is replaced by a new person
  * The average weight increases by 3 kg after the replacement
  * The total combined weight of all five people after the change is 390 kg
  Prove that the weight of the new person is 58 kg -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℕ) 
  (avg_increase : ℕ) (total_weight : ℕ) :
  initial_count = 4 →
  replaced_weight = 70 →
  avg_increase = 3 →
  total_weight = 390 →
  ∃ (new_weight : ℕ),
    new_weight = 58 ∧
    (total_weight - new_weight + replaced_weight) / initial_count = 
    (total_weight - new_weight) / initial_count + avg_increase :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l870_87067


namespace NUMINAMATH_CALUDE_arithmetic_sequence_angles_l870_87074

/-- Given five angles in an arithmetic sequence with the smallest angle 25° and the largest 105°,
    the common difference is 20°. -/
theorem arithmetic_sequence_angles (a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ →  -- ensuring the sequence is increasing
  a₁ = 25 →  -- smallest angle is 25°
  a₅ = 105 →  -- largest angle is 105°
  ∃ d : ℝ, d = 20 ∧  -- common difference exists and equals 20°
    a₂ = a₁ + d ∧ 
    a₃ = a₂ + d ∧ 
    a₄ = a₃ + d ∧ 
    a₅ = a₄ + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_angles_l870_87074


namespace NUMINAMATH_CALUDE_total_cost_theorem_l870_87017

-- Define the given conditions
def cards_per_student : ℕ := 10
def periods_per_day : ℕ := 6
def students_per_class : ℕ := 30
def cards_per_pack : ℕ := 50
def cost_per_pack : ℚ := 3

-- Define the total number of index cards needed
def total_cards_needed : ℕ := cards_per_student * students_per_class * periods_per_day

-- Define the number of packs needed
def packs_needed : ℕ := (total_cards_needed + cards_per_pack - 1) / cards_per_pack

-- State the theorem
theorem total_cost_theorem : 
  cost_per_pack * packs_needed = 108 := by sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l870_87017


namespace NUMINAMATH_CALUDE_unique_k_for_perfect_square_and_cube_l870_87070

theorem unique_k_for_perfect_square_and_cube (Z K : ℤ) 
  (h1 : 700 < Z) (h2 : Z < 1500) (h3 : K > 1) (h4 : Z = K^4) :
  (∃ a b : ℤ, Z = a^2 ∧ Z = b^3) ↔ K = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_perfect_square_and_cube_l870_87070


namespace NUMINAMATH_CALUDE_salary_problem_l870_87060

theorem salary_problem (salary_a salary_b : ℝ) : 
  salary_a + salary_b = 2000 →
  salary_a * 0.05 = salary_b * 0.15 →
  salary_a = 1500 := by
sorry

end NUMINAMATH_CALUDE_salary_problem_l870_87060


namespace NUMINAMATH_CALUDE_julie_newspaper_count_l870_87091

theorem julie_newspaper_count :
  let boxes : ℕ := 2
  let packages_per_box : ℕ := 5
  let sheets_per_package : ℕ := 250
  let sheets_per_newspaper : ℕ := 25
  let total_sheets : ℕ := boxes * packages_per_box * sheets_per_package
  let newspapers : ℕ := total_sheets / sheets_per_newspaper
  newspapers = 100 := by sorry

end NUMINAMATH_CALUDE_julie_newspaper_count_l870_87091


namespace NUMINAMATH_CALUDE_total_sugar_amount_l870_87034

/-- The total amount of sugar the owner started with, given the number of packs,
    weight per pack, and leftover sugar. -/
theorem total_sugar_amount
  (num_packs : ℕ)
  (weight_per_pack : ℕ)
  (leftover_sugar : ℕ)
  (h1 : num_packs = 12)
  (h2 : weight_per_pack = 250)
  (h3 : leftover_sugar = 20) :
  num_packs * weight_per_pack + leftover_sugar = 3020 :=
by sorry

end NUMINAMATH_CALUDE_total_sugar_amount_l870_87034


namespace NUMINAMATH_CALUDE_k_range_l870_87081

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := (3 : ℝ) / (x + 1) < 1

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬q x

-- Theorem statement
theorem k_range (k : ℝ) : 
  necessary_but_not_sufficient (p k) q ↔ k > 2 :=
sorry

end NUMINAMATH_CALUDE_k_range_l870_87081


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l870_87075

/-- The function f(x) = sin(2x) - a*cos(x) is monotonically increasing on [0, π] iff a ≥ 2 -/
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 0 Real.pi, MonotoneOn (fun x => Real.sin (2 * x) - a * Real.cos x) (Set.Icc 0 Real.pi)) ↔ 
  a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l870_87075


namespace NUMINAMATH_CALUDE_tracy_book_collection_l870_87014

theorem tracy_book_collection (x : ℕ) (h : x + 10 * x = 99) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_tracy_book_collection_l870_87014


namespace NUMINAMATH_CALUDE_given_number_eq_scientific_notation_l870_87068

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_abs_coeff : 1 ≤ |coefficient|
  abs_coeff_lt_ten : |coefficient| < 10

/-- The given number in centimeters -/
def given_number : ℝ := 0.0000021

/-- The scientific notation representation of the given number -/
def scientific_notation : ScientificNotation := {
  coefficient := 2.1
  exponent := -6
  one_le_abs_coeff := sorry
  abs_coeff_lt_ten := sorry
}

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_eq_scientific_notation : 
  given_number = scientific_notation.coefficient * (10 : ℝ) ^ scientific_notation.exponent := by
  sorry

end NUMINAMATH_CALUDE_given_number_eq_scientific_notation_l870_87068


namespace NUMINAMATH_CALUDE_scalar_projection_a_onto_b_l870_87059

/-- The scalar projection of vector a (1, 2) onto vector b (3, 4) is 11/5 -/
theorem scalar_projection_a_onto_b :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (3, 4)
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_scalar_projection_a_onto_b_l870_87059


namespace NUMINAMATH_CALUDE_OMM_MOO_not_synonyms_l870_87021

/-- Represents a word in the Ancient Tribe language --/
inductive AncientWord
  | M : AncientWord
  | O : AncientWord
  | append : AncientWord → AncientWord → AncientWord

/-- Counts the number of 'M's in a word --/
def countM : AncientWord → Nat
  | AncientWord.M => 1
  | AncientWord.O => 0
  | AncientWord.append w1 w2 => countM w1 + countM w2

/-- Counts the number of 'O's in a word --/
def countO : AncientWord → Nat
  | AncientWord.M => 0
  | AncientWord.O => 1
  | AncientWord.append w1 w2 => countO w1 + countO w2

/-- Calculates the difference between 'M's and 'O's in a word --/
def letterDifference (w : AncientWord) : Int :=
  (countM w : Int) - (countO w : Int)

/-- Two words are synonyms if their letter differences are equal --/
def areSynonyms (w1 w2 : AncientWord) : Prop :=
  letterDifference w1 = letterDifference w2

/-- Construct the word "OMM" --/
def OMM : AncientWord :=
  AncientWord.append AncientWord.O (AncientWord.append AncientWord.M AncientWord.M)

/-- Construct the word "MOO" --/
def MOO : AncientWord :=
  AncientWord.append AncientWord.M (AncientWord.append AncientWord.O AncientWord.O)

/-- Theorem: "OMM" and "MOO" are not synonyms --/
theorem OMM_MOO_not_synonyms : ¬(areSynonyms OMM MOO) := by
  sorry


end NUMINAMATH_CALUDE_OMM_MOO_not_synonyms_l870_87021


namespace NUMINAMATH_CALUDE_percentage_equality_l870_87039

theorem percentage_equality (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l870_87039


namespace NUMINAMATH_CALUDE_quadratic_function_range_l870_87010

/-- A quadratic function f(x) = a + bx - x^2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a + b * x - x^2

theorem quadratic_function_range (a b m : ℝ) :
  (∀ x, f a b (1 + x) = f a b (1 - x)) →
  (∀ x ≤ 4, Monotone (fun x ↦ f a b (x + m))) →
  m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l870_87010


namespace NUMINAMATH_CALUDE_pie_eating_contest_l870_87011

theorem pie_eating_contest (first_student second_student : ℚ) 
  (h1 : first_student = 7/8)
  (h2 : second_student = 5/6) :
  first_student - second_student = 1/24 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l870_87011


namespace NUMINAMATH_CALUDE_representable_as_product_of_three_l870_87045

theorem representable_as_product_of_three : ∃ (a b c : ℕ), 
  a > 1 ∧ b > 1 ∧ c > 1 ∧ 2^58 + 1 = a * b * c := by
  sorry

end NUMINAMATH_CALUDE_representable_as_product_of_three_l870_87045


namespace NUMINAMATH_CALUDE_personal_planner_cost_proof_l870_87025

/-- The cost of a spiral notebook -/
def spiral_notebook_cost : ℝ := 15

/-- The number of spiral notebooks -/
def num_spiral_notebooks : ℕ := 4

/-- The number of personal planners -/
def num_personal_planners : ℕ := 8

/-- The discount rate -/
def discount_rate : ℝ := 0.2

/-- The total cost after discount -/
def total_cost_after_discount : ℝ := 112

/-- The cost of a personal planner -/
def personal_planner_cost : ℝ := 10

theorem personal_planner_cost_proof :
  let total_cost := spiral_notebook_cost * num_spiral_notebooks + personal_planner_cost * num_personal_planners
  total_cost * (1 - discount_rate) = total_cost_after_discount :=
by sorry

end NUMINAMATH_CALUDE_personal_planner_cost_proof_l870_87025


namespace NUMINAMATH_CALUDE_like_terms_exponent_product_l870_87031

/-- 
Given two algebraic terms are like terms, prove that the product of their exponents is 6.
-/
theorem like_terms_exponent_product (a b : ℝ) (m n : ℕ) :
  (∃ (k : ℝ), 5 * a^3 * b^n = k * (-3 * a^m * b^2)) → m * n = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_product_l870_87031


namespace NUMINAMATH_CALUDE_opposite_of_2023_l870_87032

theorem opposite_of_2023 : 
  (2023 : ℤ) + (-2023) = 0 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l870_87032


namespace NUMINAMATH_CALUDE_remaining_money_l870_87013

def money_spent_on_books : ℝ := 76.8
def money_spent_on_apples : ℝ := 12
def total_money_brought : ℝ := 100

theorem remaining_money :
  total_money_brought - money_spent_on_books - money_spent_on_apples = 11.2 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l870_87013


namespace NUMINAMATH_CALUDE_xyz_value_l870_87092

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l870_87092


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l870_87015

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 + 3 / (a + b + c)^2 ≥ 2 :=
sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 + 3 / (a + b + c)^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l870_87015


namespace NUMINAMATH_CALUDE_prob_at_least_three_same_value_l870_87036

def num_dice : ℕ := 5
def num_sides : ℕ := 8

def prob_at_least_three_same : ℚ :=
  (num_dice.choose 3) * (1 / num_sides^2) * ((num_sides - 1) / num_sides)^2 +
  (num_dice.choose 4) * (1 / num_sides^3) * ((num_sides - 1) / num_sides) +
  (1 / num_sides^4)

theorem prob_at_least_three_same_value :
  prob_at_least_three_same = 526 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_same_value_l870_87036


namespace NUMINAMATH_CALUDE_arithmetic_mean_relation_l870_87044

theorem arithmetic_mean_relation (a b x : ℝ) : 
  (2 * x = a + b) → 
  (2 * x^2 = a^2 - b^2) → 
  (a = -b ∨ a = 3*b) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_relation_l870_87044


namespace NUMINAMATH_CALUDE_num_selection_methods_l870_87001

/-- The number of fleets --/
def num_fleets : ℕ := 7

/-- The total number of vehicles to be selected --/
def total_vehicles : ℕ := 10

/-- The minimum number of vehicles in each fleet --/
def min_vehicles_per_fleet : ℕ := 5

/-- Function to calculate the number of ways to select vehicles --/
def select_vehicles (n f t m : ℕ) : ℕ :=
  Nat.choose n 1 + n * (n - 1) + Nat.choose n 3

/-- Theorem stating the number of ways to select vehicles --/
theorem num_selection_methods :
  select_vehicles num_fleets num_fleets total_vehicles min_vehicles_per_fleet = 84 := by
  sorry


end NUMINAMATH_CALUDE_num_selection_methods_l870_87001


namespace NUMINAMATH_CALUDE_first_train_length_is_30_l870_87058

/-- The length of the second train in meters -/
def second_train_length : ℝ := 180

/-- The time taken by the first train to cross the stationary second train in seconds -/
def time_cross_stationary : ℝ := 18

/-- The length of the platform crossed by the first train in meters -/
def platform_length_first : ℝ := 250

/-- The time taken by the first train to cross its platform in seconds -/
def time_cross_platform_first : ℝ := 24

/-- The length of the platform crossed by the second train in meters -/
def platform_length_second : ℝ := 200

/-- The time taken by the second train to cross its platform in seconds -/
def time_cross_platform_second : ℝ := 22

/-- The length of the first train in meters -/
def first_train_length : ℝ := 30

theorem first_train_length_is_30 :
  (first_train_length + second_train_length) / time_cross_stationary =
  (first_train_length + platform_length_first) / time_cross_platform_first ∧
  first_train_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_first_train_length_is_30_l870_87058


namespace NUMINAMATH_CALUDE_fruit_difference_l870_87056

theorem fruit_difference (watermelons peaches plums : ℕ) : 
  watermelons = 1 →
  peaches > watermelons →
  plums = 3 * peaches →
  watermelons + peaches + plums = 53 →
  peaches - watermelons = 12 :=
by sorry

end NUMINAMATH_CALUDE_fruit_difference_l870_87056


namespace NUMINAMATH_CALUDE_perfect_cube_units_digits_l870_87098

theorem perfect_cube_units_digits : 
  ∃! (s : Finset Nat), 
    (∀ d ∈ s, d < 10) ∧ 
    (∀ n : ℤ, ∃ d ∈ s, (n ^ 3) % 10 = d) ∧
    s.card = 10 :=
by sorry

end NUMINAMATH_CALUDE_perfect_cube_units_digits_l870_87098


namespace NUMINAMATH_CALUDE_line_parameterization_l870_87037

/-- Given a line y = (3/4)x + 2 parameterized by [x; y] = [-8; s] + t[l; -6],
    prove that s = -4 and l = -8 -/
theorem line_parameterization (s l : ℝ) : 
  (∀ x y t : ℝ, y = (3/4) * x + 2 ↔ ∃ t, (x, y) = (-8 + t * l, s + t * (-6))) →
  s = -4 ∧ l = -8 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l870_87037


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l870_87050

theorem existence_of_special_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j : Fin 100, i < j → a i < a j) ∧ 
    (∀ k : Fin 100, 2 ≤ k.val → k.val ≤ 100 → 
      Nat.lcm (a ⟨k.val - 1, sorry⟩) (a k) > Nat.lcm (a k) (a ⟨k.val + 1, sorry⟩)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l870_87050


namespace NUMINAMATH_CALUDE_people_who_left_line_l870_87076

theorem people_who_left_line (initial_people : ℕ) (joined_people : ℕ) (people_who_left : ℕ) : 
  initial_people = 31 → 
  joined_people = 25 → 
  initial_people = (initial_people - people_who_left) + joined_people →
  people_who_left = 25 := by
sorry

end NUMINAMATH_CALUDE_people_who_left_line_l870_87076


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l870_87004

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 2*a| + |x + 3| < 5)) ↔ (a ≤ -4 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l870_87004


namespace NUMINAMATH_CALUDE_turban_price_turban_price_proof_l870_87071

/-- The price of a turban given the following conditions:
  - The total salary for one year is Rs. 90 plus one turban
  - The servant works for 9 months (3/4 of a year)
  - The servant receives Rs. 60 plus the turban for 9 months of work
-/
theorem turban_price : ℝ :=
  let yearly_salary : ℝ → ℝ := λ t => 90 + t
  let worked_fraction : ℝ := 3 / 4
  let received_salary : ℝ → ℝ := λ t => 60 + t
  30

theorem turban_price_proof (t : ℝ) : 
  (let yearly_salary : ℝ → ℝ := λ t => 90 + t
   let worked_fraction : ℝ := 3 / 4
   let received_salary : ℝ → ℝ := λ t => 60 + t
   worked_fraction * yearly_salary t = received_salary t) →
  t = 30 := by
sorry

end NUMINAMATH_CALUDE_turban_price_turban_price_proof_l870_87071
