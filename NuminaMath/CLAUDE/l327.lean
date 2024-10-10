import Mathlib

namespace probability_two_blue_l327_32710

/-- Represents a jar with red and blue buttons -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- Calculates the total number of buttons in a jar -/
def Jar.total (j : Jar) : ℕ := j.red + j.blue

/-- Represents the state of both jars after button removal -/
structure JarState where
  c : Jar
  d : Jar

/-- Defines the initial state of Jar C -/
def initial_jar_c : Jar := { red := 6, blue := 10 }

/-- Defines the button removal process -/
def remove_buttons (j : Jar) (n : ℕ) : JarState :=
  { c := { red := j.red - n, blue := j.blue - n },
    d := { red := n, blue := n } }

/-- Theorem stating the probability of choosing two blue buttons -/
theorem probability_two_blue (n : ℕ) : 
  let initial_total := initial_jar_c.total
  let final_state := remove_buttons initial_jar_c n
  final_state.c.total = (3 * initial_total) / 4 →
  (final_state.c.blue : ℚ) / final_state.c.total * 
  (final_state.d.blue : ℚ) / final_state.d.total = 1 / 3 := by
  sorry

end probability_two_blue_l327_32710


namespace range_of_a_l327_32708

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l327_32708


namespace line_L_equation_ellipse_C_equation_l327_32711

-- Define the line L
def line_L (x y : ℝ) : Prop := x/4 + y/2 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Theorem for line L
theorem line_L_equation :
  ∀ (a b : ℝ), a > 0 ∧ b > 0 →
  (∀ (x y : ℝ), x/a + y/b = 1 → (x = 2 ∧ y = 1)) →
  (1/2 * a * b = 4) →
  (∀ (x y : ℝ), line_L x y ↔ x/a + y/b = 1) :=
sorry

-- Theorem for ellipse C
theorem ellipse_C_equation :
  let e : ℝ := 0.8
  let c : ℝ := 4
  let a : ℝ := c / e
  let b : ℝ := Real.sqrt (a^2 - c^2)
  ∀ (x y : ℝ), ellipse_C x y ↔ x^2/a^2 + y^2/b^2 = 1 :=
sorry

end line_L_equation_ellipse_C_equation_l327_32711


namespace mean_equality_implies_y_equals_six_l327_32725

theorem mean_equality_implies_y_equals_six :
  let mean1 := (4 + 8 + 16) / 3
  let mean2 := (10 + 12 + y) / 3
  mean1 = mean2 → y = 6 :=
by
  sorry

end mean_equality_implies_y_equals_six_l327_32725


namespace minimize_y_l327_32769

/-- The function y in terms of x, a, b, and c -/
def y (x a b c : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + c * x

/-- The theorem stating that (a + b - c/2) / 2 minimizes y -/
theorem minimize_y (a b c : ℝ) :
  ∃ (x : ℝ), ∀ (z : ℝ), y z a b c ≥ y ((a + b - c/2) / 2) a b c :=
sorry

end minimize_y_l327_32769


namespace cricket_average_increase_l327_32792

theorem cricket_average_increase (innings : ℕ) (current_average : ℚ) (next_innings_runs : ℕ) 
  (h1 : innings = 13)
  (h2 : current_average = 22)
  (h3 : next_innings_runs = 92) : 
  let total_runs : ℚ := innings * current_average
  let new_total_runs : ℚ := total_runs + next_innings_runs
  let new_average : ℚ := new_total_runs / (innings + 1)
  new_average - current_average = 5 := by sorry

end cricket_average_increase_l327_32792


namespace distance_on_quadratic_curve_l327_32774

/-- The distance between two points on a quadratic curve -/
theorem distance_on_quadratic_curve (m k a c : ℝ) :
  let y (x : ℝ) := m * x^2 + k
  let point1 := (a, y a)
  let point2 := (c, y c)
  let distance := Real.sqrt ((c - a)^2 + (y c - y a)^2)
  distance = |a - c| * Real.sqrt (1 + m^2 * (c + a)^2) := by
  sorry

end distance_on_quadratic_curve_l327_32774


namespace complex_pure_imaginary_condition_l327_32780

/-- The complex number z is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

/-- Given that (2-ai)/(1+i) is pure imaginary and a is real, prove that a = 2 -/
theorem complex_pure_imaginary_condition (a : ℝ) 
  (h : isPureImaginary ((2 - a * Complex.I) / (1 + Complex.I))) : a = 2 := by
  sorry

end complex_pure_imaginary_condition_l327_32780


namespace coins_sold_proof_l327_32702

def beth_initial_coins : ℕ := 250
def carl_gift_coins : ℕ := 75
def sell_percentage : ℚ := 60 / 100

theorem coins_sold_proof :
  let total_coins := beth_initial_coins + carl_gift_coins
  ⌊(sell_percentage * total_coins : ℚ)⌋ = 195 := by sorry

end coins_sold_proof_l327_32702


namespace rectangle_to_square_trapezoid_l327_32786

theorem rectangle_to_square_trapezoid (width height area_square : ℝ) (y : ℝ) : 
  width = 16 →
  height = 9 →
  area_square = width * height →
  y = (Real.sqrt area_square) / 2 →
  y = 6 := by sorry

end rectangle_to_square_trapezoid_l327_32786


namespace parents_can_catch_kolya_l327_32742

/-- Represents a point in the park --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a person in the park --/
structure Person :=
  (position : Point)
  (speed : ℝ)

/-- Represents the park with its alleys --/
structure Park :=
  (square_side : ℝ)
  (alley_length : ℝ)

/-- Checks if a point is on an alley --/
def is_on_alley (park : Park) (p : Point) : Prop :=
  (p.x = 0 ∨ p.x = park.square_side ∨ p.x = park.square_side / 2) ∨
  (p.y = 0 ∨ p.y = park.square_side ∨ p.y = park.square_side / 2)

/-- Represents the state of the chase --/
structure ChaseState :=
  (park : Park)
  (kolya : Person)
  (parent1 : Person)
  (parent2 : Person)

/-- Defines what it means for parents to catch Kolya --/
def parents_catch_kolya (state : ChaseState) : Prop :=
  ∃ (t : ℝ), t > 0 ∧ 
  ∃ (final_kolya final_parent1 final_parent2 : Point),
    is_on_alley state.park final_kolya ∧
    is_on_alley state.park final_parent1 ∧
    is_on_alley state.park final_parent2 ∧
    (final_kolya = final_parent1 ∨ final_kolya = final_parent2)

/-- The main theorem stating that parents can catch Kolya --/
theorem parents_can_catch_kolya (initial_state : ChaseState) :
  initial_state.kolya.speed = 3 * initial_state.parent1.speed ∧
  initial_state.kolya.speed = 3 * initial_state.parent2.speed ∧
  initial_state.park.square_side > 0 ∧
  initial_state.park.alley_length > 0 →
  parents_catch_kolya initial_state :=
sorry

end parents_can_catch_kolya_l327_32742


namespace triangle_inequalities_l327_32756

/-- Given a triangle with side lengths a, b, c, circumradius R, and inradius r,
    prove the inequalities abc ≥ (a+b-c)(a-b+c)(-a+b+c) and R ≥ 2r -/
theorem triangle_inequalities (a b c R r : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R > 0)
  (h_inradius : r > 0)
  (h_area : 4 * R * (r * (a + b + c) / 2) = a * b * c) :
  a * b * c ≥ (a + b - c) * (a - b + c) * (-a + b + c) ∧ R ≥ 2 * r := by
  sorry

end triangle_inequalities_l327_32756


namespace obstacle_course_time_l327_32775

/-- Represents the times for each segment of the obstacle course -/
structure ObstacleCourse :=
  (first_run : List Int)
  (door_opening : Int)
  (second_run : List Int)

/-- Calculates the total time to complete the obstacle course -/
def total_time (course : ObstacleCourse) : Int :=
  (course.first_run.sum + course.door_opening + course.second_run.sum)

/-- The theorem to prove -/
theorem obstacle_course_time :
  let course := ObstacleCourse.mk [225, 130, 88, 45, 120] 73 [175, 108, 75, 138]
  total_time course = 1177 := by
  sorry

end obstacle_course_time_l327_32775


namespace intersection_line_canonical_l327_32760

-- Define the planes
def plane1 (x y z : ℝ) : Prop := 3 * x + 4 * y + 3 * z + 1 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x - 4 * y - 2 * z + 4 = 0

-- Define the canonical form of the line
def canonical_line (x y z : ℝ) : Prop := (x + 1) / 4 = (y - 1/2) / 12 ∧ (y - 1/2) / 12 = z / (-20)

-- Theorem statement
theorem intersection_line_canonical : 
  ∀ x y z : ℝ, plane1 x y z ∧ plane2 x y z → canonical_line x y z :=
by
  sorry

end intersection_line_canonical_l327_32760


namespace propositions_3_and_4_are_true_l327_32712

theorem propositions_3_and_4_are_true :
  (∃ n : ℝ, ∀ m : ℝ, m * n = m) ∧
  (∀ n : ℝ, ∃ m : ℝ, m^2 < n) :=
by sorry

end propositions_3_and_4_are_true_l327_32712


namespace smallest_x_for_perfect_cube_l327_32762

theorem smallest_x_for_perfect_cube : ∃ (x : ℕ+), 
  (∀ (y : ℕ+), 2520 * y.val = (M : ℕ)^3 → x ≤ y) ∧ 
  (∃ (M : ℕ), 2520 * x.val = M^3) ∧
  x.val = 3675 := by
  sorry

end smallest_x_for_perfect_cube_l327_32762


namespace nick_pennsylvania_quarters_l327_32788

/-- Given a total number of quarters, calculate the number of Pennsylvania state quarters -/
def pennsylvania_quarters (total : ℕ) : ℕ :=
  let state_quarters := (2 * total) / 5
  (state_quarters / 2 : ℕ)

theorem nick_pennsylvania_quarters :
  pennsylvania_quarters 35 = 7 := by
  sorry

end nick_pennsylvania_quarters_l327_32788


namespace min_socks_for_different_colors_l327_32741

theorem min_socks_for_different_colors :
  let total_blue_socks : ℕ := 6
  let total_red_socks : ℕ := 6
  let min_socks : ℕ := 7
  ∀ (selected : ℕ), selected ≥ min_socks →
    ∃ (blue red : ℕ), blue + red = selected ∧
      blue ≤ total_blue_socks ∧
      red ≤ total_red_socks ∧
      (blue > 0 ∧ red > 0) :=
by sorry

end min_socks_for_different_colors_l327_32741


namespace certain_number_proof_l327_32713

theorem certain_number_proof (A B C X : ℝ) : 
  A / B = 5 / 6 →
  B / C = 6 / 8 →
  C = 42 →
  A + C = B + X →
  X = 36.75 := by
sorry

end certain_number_proof_l327_32713


namespace collinear_points_sum_l327_32737

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p2 = p1 + t • (p3 - p1) ∨ p1 = p2 + t • (p3 - p2)

theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 := by
  sorry

end collinear_points_sum_l327_32737


namespace function_composition_equality_l327_32765

theorem function_composition_equality (a b c d : ℝ) :
  let f := fun (x : ℝ) => a * x + b
  let g := fun (x : ℝ) => c * x + d
  (∀ x, f (g x) = g (f x)) ↔ (b = d ∨ a = c + 1) :=
sorry

end function_composition_equality_l327_32765


namespace triangle_on_parabola_bc_length_l327_32782

/-- Parabola function -/
def parabola (x : ℝ) : ℝ := x^2

/-- Triangle ABC -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Check if a point lies on the parabola -/
def onParabola (p : ℝ × ℝ) : Prop :=
  p.2 = parabola p.1

/-- Check if two points have the same y-coordinate (i.e., line is parallel to x-axis) -/
def parallelToXAxis (p q : ℝ × ℝ) : Prop :=
  p.2 = q.2

/-- Calculate the area of a triangle given its vertices -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Calculate the length of a line segment -/
noncomputable def segmentLength (p q : ℝ × ℝ) : ℝ :=
  sorry

/-- Main theorem -/
theorem triangle_on_parabola_bc_length (t : Triangle) :
  onParabola t.A ∧ onParabola t.B ∧ onParabola t.C ∧
  t.A = (1, 1) ∧
  parallelToXAxis t.B t.C ∧
  triangleArea t = 50 →
  ∃ ε > 0, |segmentLength t.B t.C - 5.8| < ε :=
sorry

end triangle_on_parabola_bc_length_l327_32782


namespace min_value_of_expression_l327_32740

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 * y * (4*x + 3*y) = 3) : 
  ∃ (min : ℝ), min = 2 * Real.sqrt 3 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 
  x'^2 * y' * (4*x' + 3*y') = 3 → 2*x' + 3*y' ≥ min := by
  sorry

end min_value_of_expression_l327_32740


namespace wire_length_proof_l327_32717

theorem wire_length_proof (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 30 ∧ 
  shorter_piece = (3/5) * longer_piece ∧
  total_length = shorter_piece + longer_piece →
  total_length = 80 := by
sorry

end wire_length_proof_l327_32717


namespace range_of_2alpha_minus_beta_l327_32753

theorem range_of_2alpha_minus_beta (α β : ℝ) 
  (h1 : π < α + β ∧ α + β < (4*π)/3)
  (h2 : -π < α - β ∧ α - β < -π/3) :
  ∀ x, (-π < x ∧ x < π/6) ↔ ∃ α' β', 
    (π < α' + β' ∧ α' + β' < (4*π)/3) ∧
    (-π < α' - β' ∧ α' - β' < -π/3) ∧
    x = 2*α' - β' :=
by sorry

end range_of_2alpha_minus_beta_l327_32753


namespace largest_common_number_l327_32783

/-- First sequence with initial term 5 and common difference 9 -/
def sequence1 (n : ℕ) : ℕ := 5 + 9 * n

/-- Second sequence with initial term 3 and common difference 8 -/
def sequence2 (m : ℕ) : ℕ := 3 + 8 * m

/-- Theorem stating that 167 is the largest common number in both sequences within the range 1 to 200 -/
theorem largest_common_number :
  ∃ (n m : ℕ),
    sequence1 n = sequence2 m ∧
    sequence1 n = 167 ∧
    sequence1 n ≤ 200 ∧
    ∀ (k l : ℕ), sequence1 k = sequence2 l → sequence1 k ≤ 200 → sequence1 k ≤ 167 :=
by sorry

end largest_common_number_l327_32783


namespace theater_attendance_l327_32703

/-- Proves the number of children attending a theater given total attendance and revenue --/
theorem theater_attendance (adults children : ℕ) 
  (total_attendance : adults + children = 280)
  (total_revenue : 60 * adults + 25 * children = 14000) :
  children = 80 := by sorry

end theater_attendance_l327_32703


namespace prize_problem_solution_l327_32767

/-- Represents the prices and quantities of notebooks and pens -/
structure PrizeInfo where
  notebook_price : ℕ
  pen_price : ℕ
  notebook_quantity : ℕ
  pen_quantity : ℕ

/-- Theorem stating the solution to the prize problem -/
theorem prize_problem_solution :
  ∃ (info : PrizeInfo),
    -- Each notebook costs 3 yuan more than each pen
    info.notebook_price = info.pen_price + 3 ∧
    -- The number of notebooks purchased for 390 yuan is the same as the number of pens purchased for 300 yuan
    390 / info.notebook_price = 300 / info.pen_price ∧
    -- The total cost of purchasing prizes for 50 students should not exceed 560 yuan
    info.notebook_quantity + info.pen_quantity = 50 ∧
    info.notebook_price * info.notebook_quantity + info.pen_price * info.pen_quantity ≤ 560 ∧
    -- The notebook price is 13 yuan
    info.notebook_price = 13 ∧
    -- The pen price is 10 yuan
    info.pen_price = 10 ∧
    -- The maximum number of notebooks that can be purchased is 20
    info.notebook_quantity = 20 ∧
    -- This is the maximum possible number of notebooks
    ∀ (other_info : PrizeInfo),
      other_info.notebook_price = other_info.pen_price + 3 →
      other_info.notebook_quantity + other_info.pen_quantity = 50 →
      other_info.notebook_price * other_info.notebook_quantity + other_info.pen_price * other_info.pen_quantity ≤ 560 →
      other_info.notebook_quantity ≤ info.notebook_quantity :=
by
  sorry


end prize_problem_solution_l327_32767


namespace expression_value_l327_32726

theorem expression_value (x y z : ℝ) (hx : x = 2) (hy : y = -3) (hz : z = 1) :
  x^2 + y^2 - z^2 + 2*x*y + 2*y*z = -6 := by
  sorry

end expression_value_l327_32726


namespace sum_squares_units_digit_3003_l327_32784

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def square (n : ℕ) : ℕ := n * n

def units_digit (n : ℕ) : ℕ := n % 10

theorem sum_squares_units_digit_3003 :
  units_digit (List.sum (List.map square (first_n_odd_integers 3003))) = 5 := by
  sorry

end sum_squares_units_digit_3003_l327_32784


namespace unique_k_square_sum_l327_32734

theorem unique_k_square_sum : ∃! (k : ℕ), k ≠ 1 ∧
  (∃ (n : ℕ), k = n^2 + (n+1)^2) ∧
  (∃ (m : ℕ), k^4 = m^2 + (m+1)^2) :=
by sorry

end unique_k_square_sum_l327_32734


namespace quadratic_inequality_solution_set_l327_32754

/-- Given that the solution set of ax^2 - bx - 1 > 0 is {x | -1/2 < x < -1/3},
    prove that the solution set of x^2 - bx - a ≥ 0 is {x | x ≥ 3 ∨ x ≤ 2} -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x : ℝ, ax^2 - b*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :
  ∀ x : ℝ, x^2 - b*x - a ≥ 0 ↔ x ≥ 3 ∨ x ≤ 2 :=
sorry

end quadratic_inequality_solution_set_l327_32754


namespace volume_of_specific_polyhedron_l327_32757

/-- Represents a cube in 3D space -/
structure Cube where
  edge_length : ℝ

/-- Represents a convex polyhedron formed by planes passing through midpoints of cube edges -/
structure ConvexPolyhedron where
  cube : Cube

/-- Calculate the volume of the convex polyhedron -/
def volume (p : ConvexPolyhedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific convex polyhedron -/
theorem volume_of_specific_polyhedron :
  ∀ (c : Cube) (p : ConvexPolyhedron),
    c.edge_length = 2 →
    p.cube = c →
    volume p = 32 / 3 :=
  sorry

end volume_of_specific_polyhedron_l327_32757


namespace angle_between_points_after_one_second_l327_32730

/-- Represents the angular velocity of a rotating point. -/
structure AngularVelocity where
  value : ℝ
  positive : value > 0

/-- Represents a rotating point on a circle. -/
structure RotatingPoint where
  velocity : AngularVelocity

/-- Calculates the angle between two rotating points after 1 second. -/
def angleBetweenPoints (p1 p2 : RotatingPoint) : ℝ := sorry

/-- Theorem stating the angle between two rotating points after 1 second. -/
theorem angle_between_points_after_one_second 
  (p1 p2 : RotatingPoint) 
  (h1 : p1.velocity.value - p2.velocity.value = 2 * Real.pi / 60)  -- Two more revolutions per minute
  (h2 : 1 / p1.velocity.value - 1 / p2.velocity.value = 5)  -- 5 seconds faster revolution
  : angleBetweenPoints p1 p2 = 12 * Real.pi / 180 ∨ 
    angleBetweenPoints p1 p2 = 60 * Real.pi / 180 := by
  sorry

end angle_between_points_after_one_second_l327_32730


namespace prime_power_sum_l327_32796

theorem prime_power_sum (a b c d e : ℕ) : 
  2^a * 3^b * 5^c * 7^d * 11^e = 6930 → 2*a + 3*b + 5*c + 7*d + 11*e = 31 := by
  sorry

end prime_power_sum_l327_32796


namespace quadratic_inequality_integer_solutions_l327_32778

theorem quadratic_inequality_integer_solutions :
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ 7 * x^2 + 25 * x + 24 ≤ 30) ∧ Finset.card S = 7 :=
sorry

end quadratic_inequality_integer_solutions_l327_32778


namespace problem_statement_l327_32723

theorem problem_statement (a b : ℝ) (h1 : 2*a + b = -3) (h2 : 2*a - b = 2) :
  4*a^2 - b^2 = -6 := by
  sorry

end problem_statement_l327_32723


namespace domain_of_composite_function_l327_32736

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-1) 2

-- State the theorem
theorem domain_of_composite_function :
  {x : ℝ | f (3 - 2*x) ∈ Set.range f} = Set.Icc (1/2) 2 :=
sorry

end domain_of_composite_function_l327_32736


namespace hyperbola_parameter_sum_l327_32794

/-- Theorem about the sum of parameters for a specific hyperbola -/
theorem hyperbola_parameter_sum :
  let center : ℝ × ℝ := (1, 3)
  let focus : ℝ × ℝ := (1, 9)
  let vertex : ℝ × ℝ := (1, 0)
  let h : ℝ := center.1
  let k : ℝ := center.2
  let a : ℝ := |k - vertex.2|
  let c : ℝ := |k - focus.2|
  let b : ℝ := Real.sqrt (c^2 - a^2)
  h + k + a + b = 7 + 3 * Real.sqrt 3 :=
by sorry

end hyperbola_parameter_sum_l327_32794


namespace sum_of_coefficients_l327_32745

def polynomial (x : ℝ) : ℝ :=
  -3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (x^4 + 3*x^2 - 2*x + 1) - 4 * (x^6 - 5*x + 7)

theorem sum_of_coefficients :
  polynomial 1 = 12 :=
sorry

end sum_of_coefficients_l327_32745


namespace min_wires_for_unit_cube_l327_32799

/-- Represents a piece of wire with a given length -/
structure Wire where
  length : ℕ

/-- Represents a cube -/
structure Cube where
  edgeLength : ℕ
  numEdges : ℕ := 12
  numVertices : ℕ := 8

def availableWires : List Wire := [
  { length := 1 },
  { length := 2 },
  { length := 3 },
  { length := 4 },
  { length := 5 },
  { length := 6 },
  { length := 7 }
]

def targetCube : Cube := { edgeLength := 1 }

/-- Returns the minimum number of wire pieces needed to form the cube -/
def minWiresForCube (wires : List Wire) (cube : Cube) : ℕ := sorry

theorem min_wires_for_unit_cube :
  minWiresForCube availableWires targetCube = 4 := by sorry

end min_wires_for_unit_cube_l327_32799


namespace units_digit_of_sum_l327_32793

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sequence_term (n : ℕ) : ℕ := factorial n + n

def sum_sequence (n : ℕ) : ℕ := 
  List.sum (List.map sequence_term (List.range n))

theorem units_digit_of_sum : 
  (sum_sequence 10) % 10 = 8 := by sorry

end units_digit_of_sum_l327_32793


namespace inequality_proof_l327_32719

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (2 * x^2) / (y + z) + (2 * y^2) / (z + x) + (2 * z^2) / (x + y) ≥ 1 := by
  sorry

end inequality_proof_l327_32719


namespace diamond_equation_solution_l327_32738

-- Define the binary operation ◇
noncomputable def diamond (a b : ℝ) : ℝ := a / b

-- State the theorem
theorem diamond_equation_solution :
  (∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 → diamond a (diamond b c) = diamond (diamond a b) c) →
  (∀ (a : ℝ), a ≠ 0 → diamond a a = 1) →
  diamond 504 (diamond 12 (25 / 21)) = 50 := by
  sorry

end diamond_equation_solution_l327_32738


namespace complex_number_equality_l327_32722

theorem complex_number_equality (z : ℂ) : z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3) → z = 1 - Complex.I * Real.sqrt 3 := by
  sorry

end complex_number_equality_l327_32722


namespace equation_transformation_correctness_l327_32744

theorem equation_transformation_correctness :
  -- Option A is incorrect
  (∀ x : ℝ, 3 + x = 7 → x ≠ 7 + 3) ∧
  -- Option B is incorrect
  (∀ x : ℝ, 5 * x = -4 → x ≠ -5/4) ∧
  -- Option C is incorrect
  (∀ x : ℝ, 7/4 * x = 3 → x ≠ 3 * 7/4) ∧
  -- Option D is correct
  (∀ x : ℝ, -(x - 2) / 4 = 1 → -(x - 2) = 4) :=
by sorry

end equation_transformation_correctness_l327_32744


namespace ternary_2101211_equals_octal_444_l327_32766

/-- Converts a ternary number represented as a list of digits to its decimal value. -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Converts a decimal number to its octal representation as a list of digits. -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Theorem stating that the ternary number 2101211 is equal to the octal number 444. -/
theorem ternary_2101211_equals_octal_444 :
  decimal_to_octal (ternary_to_decimal [1, 1, 2, 1, 0, 1, 2]) = [4, 4, 4] := by
  sorry

#eval ternary_to_decimal [1, 1, 2, 1, 0, 1, 2]
#eval decimal_to_octal (ternary_to_decimal [1, 1, 2, 1, 0, 1, 2])

end ternary_2101211_equals_octal_444_l327_32766


namespace arccos_cos_2x_solution_set_l327_32709

theorem arccos_cos_2x_solution_set :
  ∀ x : ℝ, (Real.arccos (Real.cos (2 * x)) = x) ↔ 
    (∃ k : ℤ, x = 2 * k * π ∨ x = 2 * π / 3 + 2 * k * π ∨ x = -(2 * π / 3) + 2 * k * π) :=
by sorry

end arccos_cos_2x_solution_set_l327_32709


namespace selene_total_cost_l327_32743

/-- Calculate the total cost of Selene's purchase --/
def calculate_total_cost (camera_price : ℚ) (camera_count : ℕ) (frame_price : ℚ) (frame_count : ℕ)
  (card_price : ℚ) (card_count : ℕ) (camera_discount : ℚ) (frame_discount : ℚ) (card_discount : ℚ)
  (camera_frame_tax : ℚ) (card_tax : ℚ) : ℚ :=
  let camera_total := camera_price * camera_count
  let frame_total := frame_price * frame_count
  let card_total := card_price * card_count
  let camera_discounted := camera_total * (1 - camera_discount)
  let frame_discounted := frame_total * (1 - frame_discount)
  let card_discounted := card_total * (1 - card_discount)
  let camera_frame_subtotal := camera_discounted + frame_discounted
  let camera_frame_taxed := camera_frame_subtotal * (1 + camera_frame_tax)
  let card_taxed := card_discounted * (1 + card_tax)
  camera_frame_taxed + card_taxed

/-- Theorem stating that Selene's total cost is $691.72 --/
theorem selene_total_cost :
  calculate_total_cost 110 2 120 3 30 4 (7/100) (5/100) (10/100) (6/100) (4/100) = 69172/100 := by
  sorry

end selene_total_cost_l327_32743


namespace magnitude_of_sum_l327_32795

/-- Given two vectors a and b in ℝ², prove that |2a + b| = 3 -/
theorem magnitude_of_sum (a b : ℝ × ℝ) :
  (‖a‖ = 1) →
  (b = (1, 2)) →
  (a • b = 0) →
  ‖2 • a + b‖ = 3 := by
  sorry

end magnitude_of_sum_l327_32795


namespace cubic_sum_theorem_l327_32746

theorem cubic_sum_theorem (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = 2) 
  (h3 : a * b * c = 3) : 
  a^3 + b^3 + c^3 = 9 := by sorry

end cubic_sum_theorem_l327_32746


namespace students_representing_x_percent_of_boys_l327_32747

theorem students_representing_x_percent_of_boys 
  (total_population : ℝ) 
  (boys_percentage : ℝ) 
  (x : ℝ) 
  (h1 : total_population = 113.38934190276818)
  (h2 : boys_percentage = 70) :
  (x / 100) * (boys_percentage / 100 * total_population) = 
  (x / 100) * 79.37253933173772 :=
by
  sorry

end students_representing_x_percent_of_boys_l327_32747


namespace sean_money_difference_l327_32764

theorem sean_money_difference (fritz_money : ℕ) (rick_sean_total : ℕ) : 
  fritz_money = 40 →
  rick_sean_total = 96 →
  ∃ (sean_money : ℕ),
    sean_money > fritz_money / 2 ∧
    3 * sean_money + sean_money = rick_sean_total ∧
    sean_money - fritz_money / 2 = 4 :=
by sorry

end sean_money_difference_l327_32764


namespace range_of_x_l327_32705

def p (x : ℝ) := Real.log (x^2 - 2*x - 2) ≥ 0

def q (x : ℝ) := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) (hp : p x) (hq : ¬q x) : x ≥ 4 ∨ x ≤ -1 := by
  sorry

end range_of_x_l327_32705


namespace product_of_sums_and_differences_l327_32773

theorem product_of_sums_and_differences (P Q R S : ℝ) : P * Q * R * S = 1 :=
  by
  have h1 : P = Real.sqrt 2011 + Real.sqrt 2010 := by sorry
  have h2 : Q = -Real.sqrt 2011 - Real.sqrt 2010 := by sorry
  have h3 : R = Real.sqrt 2011 - Real.sqrt 2010 := by sorry
  have h4 : S = Real.sqrt 2010 - Real.sqrt 2011 := by sorry
  sorry

#check product_of_sums_and_differences

end product_of_sums_and_differences_l327_32773


namespace disjoint_subsets_prime_products_l327_32781

/-- A function that constructs 100 disjoint subsets of positive integers -/
def construct_subsets : Fin 100 → Set ℕ := sorry

/-- Predicate to check if a number is a product of m distinct primes from a set -/
def is_product_of_m_primes (n : ℕ) (m : ℕ) (S : Set ℕ) : Prop := sorry

/-- Main theorem statement -/
theorem disjoint_subsets_prime_products :
  ∃ (A : Fin 100 → Set ℕ), 
    (∀ i j, i ≠ j → Disjoint (A i) (A j)) ∧
    (∀ (S : Set ℕ) (hS : Set.Infinite S) (h_prime : ∀ p ∈ S, Nat.Prime p),
      ∃ (m : ℕ) (a : Fin 100 → ℕ), 
        ∀ i, a i ∈ A i ∧ is_product_of_m_primes (a i) m S) :=
sorry

end disjoint_subsets_prime_products_l327_32781


namespace difference_of_squares_divided_l327_32716

theorem difference_of_squares_divided : (311^2 - 297^2) / 14 = 608 := by
  sorry

end difference_of_squares_divided_l327_32716


namespace problem_statement_l327_32748

theorem problem_statement (x y : ℚ) 
  (h1 : 3 * x + 4 * y = 0)
  (h2 : x = y + 3) :
  5 * y = -45 / 7 := by
sorry

end problem_statement_l327_32748


namespace labourer_savings_is_30_l327_32714

/-- Calculates the savings of a labourer after clearing debt -/
def labourerSavings (monthlyIncome : ℕ) (initialExpenditure : ℕ) (initialMonths : ℕ)
  (reducedExpenditure : ℕ) (reducedMonths : ℕ) : ℕ :=
  let initialDebt := initialMonths * initialExpenditure - initialMonths * monthlyIncome
  let availableAmount := reducedMonths * monthlyIncome - reducedMonths * reducedExpenditure
  availableAmount - initialDebt

/-- The labourer's savings after clearing debt is 30 -/
theorem labourer_savings_is_30 :
  labourerSavings 78 85 6 60 4 = 30 := by
  sorry

end labourer_savings_is_30_l327_32714


namespace rectangle_area_l327_32704

/-- The area of a rectangle with width 81/4 cm and height 148/9 cm is 333 cm². -/
theorem rectangle_area : 
  let width : ℚ := 81 / 4
  let height : ℚ := 148 / 9
  (width * height : ℚ) = 333 := by sorry

end rectangle_area_l327_32704


namespace book_probabilities_l327_32701

/-- Represents the book collection with given properties -/
structure BookCollection where
  total : ℕ
  liberal_arts : ℕ
  hardcover : ℕ
  softcover_science : ℕ
  total_eq : total = 100
  liberal_arts_eq : liberal_arts = 40
  hardcover_eq : hardcover = 70
  softcover_science_eq : softcover_science = 20

/-- Calculates the probability of selecting a liberal arts hardcover book -/
def prob_liberal_arts_hardcover (bc : BookCollection) : ℚ :=
  (bc.hardcover - bc.softcover_science : ℚ) / bc.total

/-- Calculates the probability of selecting a liberal arts book then a hardcover book -/
def prob_liberal_arts_then_hardcover (bc : BookCollection) : ℚ :=
  (bc.liberal_arts : ℚ) / bc.total * (bc.hardcover : ℚ) / bc.total

/-- Main theorem stating the probabilities -/
theorem book_probabilities (bc : BookCollection) :
    prob_liberal_arts_hardcover bc = 3/10 ∧
    prob_liberal_arts_then_hardcover bc = 28/100 := by
  sorry

end book_probabilities_l327_32701


namespace expression_simplification_l327_32751

theorem expression_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) :
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2) = 
  3 / (2 * (-b - c + b * c)) :=
by sorry

end expression_simplification_l327_32751


namespace parabola_intersection_length_l327_32798

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P = focus + t • (Q - focus) ∨ Q = focus + t • (P - focus)

-- Define the theorem
theorem parabola_intersection_length 
  (P Q : ℝ × ℝ) 
  (h_P : parabola P.1 P.2) 
  (h_Q : parabola Q.1 Q.2) 
  (h_line : line_through_focus P Q) 
  (h_sum : P.1 + Q.1 = 9) : 
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 11 :=
sorry

end parabola_intersection_length_l327_32798


namespace village_households_l327_32776

/-- The number of households in a village where:
    1. Each household requires 20 litres of water per month
    2. 2000 litres of water lasts for 10 months for all households -/
def number_of_households : ℕ := 10

/-- The amount of water required per household per month (in litres) -/
def water_per_household_per_month : ℕ := 20

/-- The total amount of water available (in litres) -/
def total_water : ℕ := 2000

/-- The number of months the water supply lasts -/
def months_supply : ℕ := 10

theorem village_households :
  number_of_households * water_per_household_per_month * months_supply = total_water :=
by sorry

end village_households_l327_32776


namespace exists_special_box_l327_32707

/-- A rectangular box with integer dimensions (a, b, c) where the volume is four times the surface area -/
def SpecialBox (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = 8 * (a * b + b * c + c * a)

/-- There exists at least one ordered triple (a, b, c) satisfying the SpecialBox conditions -/
theorem exists_special_box : ∃ (a b c : ℕ), SpecialBox a b c := by
  sorry

end exists_special_box_l327_32707


namespace circle_point_perpendicular_l327_32750

theorem circle_point_perpendicular (m : ℝ) : m > 0 →
  (∃ (P : ℝ × ℝ), P.1^2 + P.2^2 = 1 ∧ 
    ((P.1 + m) * (P.1 - m) + (P.2 - 2) * (P.2 - 2) = 0)) →
  (3 : ℝ) - 1 = 2 := by sorry

end circle_point_perpendicular_l327_32750


namespace square_diagonal_quadrilateral_l327_32791

/-- Given a square with side length a, this theorem proves the properties of a quadrilateral
    formed by the endpoints of a diagonal and the centers of inscribed circles of the two
    isosceles right triangles created by that diagonal. -/
theorem square_diagonal_quadrilateral (a : ℝ) (h : a > 0) :
  ∃ (perimeter area : ℝ),
    perimeter = 4 * a * Real.sqrt (2 - Real.sqrt 2) ∧
    area = a^2 * (Real.sqrt 2 - 1) :=
by sorry

end square_diagonal_quadrilateral_l327_32791


namespace pencils_per_box_l327_32789

theorem pencils_per_box (total_boxes : ℕ) (kept_pencils : ℕ) (num_friends : ℕ) (pencils_per_friend : ℕ) : 
  total_boxes = 10 →
  kept_pencils = 10 →
  num_friends = 5 →
  pencils_per_friend = 8 →
  (total_boxes * (kept_pencils + num_friends * pencils_per_friend)) / total_boxes = 5 :=
by sorry

end pencils_per_box_l327_32789


namespace one_cow_one_bag_days_l327_32718

/-- Given that 52 cows eat 104 bags of husk in 78 days, 
    prove that it takes 39 days for one cow to eat one bag of husk. -/
theorem one_cow_one_bag_days (cows : ℕ) (bags : ℕ) (days : ℕ) 
  (h1 : cows = 52) 
  (h2 : bags = 104) 
  (h3 : days = 78) : 
  (bags * days) / (cows * bags) = 39 := by
  sorry

end one_cow_one_bag_days_l327_32718


namespace largest_integer_with_remainder_l327_32771

theorem largest_integer_with_remainder : ∃ n : ℕ, 
  (n < 120) ∧ 
  (n % 8 = 7) ∧ 
  (∀ m : ℕ, m < 120 ∧ m % 8 = 7 → m ≤ n) ∧
  (n = 119) := by
sorry

end largest_integer_with_remainder_l327_32771


namespace fourteen_machines_four_minutes_l327_32735

/-- The number of bottles produced by a given number of machines in a given time -/
def bottles_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  let base_machines := 6
  let base_production := 270
  let production_per_machine_per_minute := base_production / base_machines
  machines * production_per_machine_per_minute * minutes

/-- Theorem stating that 14 machines produce 2520 bottles in 4 minutes -/
theorem fourteen_machines_four_minutes :
  bottles_produced 14 4 = 2520 := by
  sorry

end fourteen_machines_four_minutes_l327_32735


namespace soda_cost_l327_32785

/-- The cost of items in cents -/
structure Cost where
  burger : ℕ
  soda : ℕ

/-- The given conditions of the problem -/
def problem_conditions (c : Cost) : Prop :=
  2 * c.burger + c.soda = 210 ∧ c.burger + 2 * c.soda = 240

/-- The theorem stating that under the given conditions, a soda costs 90 cents -/
theorem soda_cost (c : Cost) : problem_conditions c → c.soda = 90 := by
  sorry

end soda_cost_l327_32785


namespace inequality_proof_l327_32755

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_cond : |Real.sqrt (a * b) - Real.sqrt (c * d)| ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := by
sorry

end inequality_proof_l327_32755


namespace rational_sum_l327_32727

theorem rational_sum (a b : ℚ) (h : |3 - a| + (b + 2)^2 = 0) : a + b = 1 := by
  sorry

end rational_sum_l327_32727


namespace solution_using_determinants_l327_32721

/-- Definition of 2x2 determinant -/
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- System of equations -/
def equation1 (x y : ℝ) : Prop := 2 * x - y = 1
def equation2 (x y : ℝ) : Prop := 3 * x + 2 * y = 11

/-- Determinants for the system -/
def D : ℝ := det2x2 2 (-1) 3 2
def D_x : ℝ := det2x2 1 (-1) 11 2
def D_y : ℝ := det2x2 2 1 3 11

/-- Theorem: Solution of the system using determinant method -/
theorem solution_using_determinants :
  ∃ (x y : ℝ), equation1 x y ∧ equation2 x y ∧ x = D_x / D ∧ y = D_y / D :=
sorry

end solution_using_determinants_l327_32721


namespace original_number_proof_l327_32706

/-- Given a number n formed by adding a digit h in the 10's place of 284,
    where n is divisible by 6 and h = 1, prove that the original number
    without the 10's digit is 284. -/
theorem original_number_proof (n : ℕ) (h : ℕ) :
  n = 2000 + h * 100 + 84 →
  h = 1 →
  n % 6 = 0 →
  2000 + 84 = 284 :=
by sorry

end original_number_proof_l327_32706


namespace g_composition_half_l327_32772

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

-- State the theorem
theorem g_composition_half : g (g (1/2)) = 1/2 := by
  sorry

end g_composition_half_l327_32772


namespace tuzik_meets_ivan_l327_32724

/-- The time when Tuzik reaches Ivan -/
def meeting_time : Real :=
  -- Define the meeting time as 47 minutes after 12:00
  47

/-- Proof that Tuzik reaches Ivan at the calculated meeting time -/
theorem tuzik_meets_ivan (total_distance : Real) (ivan_speed : Real) (tuzik_speed : Real) 
  (ivan_start_time : Real) (tuzik_start_time : Real) :
  total_distance = 12000 →  -- 12 km in meters
  ivan_speed = 1 →          -- 1 m/s
  tuzik_speed = 9 →         -- 9 m/s
  ivan_start_time = 0 →     -- 12:00 represented as 0 minutes
  tuzik_start_time = 30 →   -- 12:30 represented as 30 minutes
  meeting_time = 47 := by
  sorry

#check tuzik_meets_ivan

end tuzik_meets_ivan_l327_32724


namespace first_discount_percentage_l327_32761

theorem first_discount_percentage
  (list_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : list_price = 70)
  (h2 : final_price = 61.74)
  (h3 : second_discount = 0.01999999999999997)
  : ∃ (first_discount : ℝ),
    first_discount = 0.1 ∧
    final_price = list_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end first_discount_percentage_l327_32761


namespace completing_square_transformation_l327_32779

theorem completing_square_transformation (x : ℝ) :
  x^2 - 8*x - 11 = 0 ↔ (x - 4)^2 = 27 :=
by sorry

end completing_square_transformation_l327_32779


namespace trajectory_and_fixed_points_l327_32768

-- Define the points and lines
def F : ℝ × ℝ := (1, 0)
def H : ℝ × ℝ := (1, 2)
def l : Set (ℝ × ℝ) := {p | p.1 = -1}

-- Define the trajectory C
def C : Set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}

-- Define the function for the circle with MN as diameter
def circle_MN (m : ℝ) : Set (ℝ × ℝ) := 
  {p | p.1^2 + 2*p.1 - 3 + p.2^2 + (4/m)*p.2 = 0}

-- Theorem statement
theorem trajectory_and_fixed_points :
  -- Part 1: Trajectory C
  (∀ Q : ℝ × ℝ, (∃ P : ℝ × ℝ, P ∈ l ∧ 
    (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = (Q.1 - F.1)^2 + (Q.2 - F.2)^2) 
    → Q ∈ C) ∧
  -- Part 2: Fixed points on the circle
  (∀ m : ℝ, m ≠ 0 → (-3, 0) ∈ circle_MN m ∧ (1, 0) ∈ circle_MN m) :=
sorry

end trajectory_and_fixed_points_l327_32768


namespace expression_evaluation_l327_32749

theorem expression_evaluation : (75 / 1.5) * (500 / 25) - (300 / 0.03) + (125 * 4 / 0.1) = -4000 := by
  sorry

end expression_evaluation_l327_32749


namespace absolute_value_inequality_l327_32739

theorem absolute_value_inequality (a b : ℝ) (h : |a - b| > 2) :
  ∀ x : ℝ, |x - a| + |x - b| > 2 :=
by sorry

end absolute_value_inequality_l327_32739


namespace smallest_divisible_by_1_to_10_l327_32790

theorem smallest_divisible_by_1_to_10 : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end smallest_divisible_by_1_to_10_l327_32790


namespace range_of_a_l327_32700

/-- Given sets A and B, where A is [-2, 4) and B is {x | x^2 - ax - 4 ≤ 0},
    if B is a subset of A, then a is in the range [0, 3). -/
theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := {x | -2 ≤ x ∧ x < 4}
  let B : Set ℝ := {x | x^2 - a*x - 4 ≤ 0}
  B ⊆ A → 0 ≤ a ∧ a < 3 := by
  sorry

end range_of_a_l327_32700


namespace base_subtraction_equality_l327_32777

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their original bases
def num1 : List Nat := [5, 2, 3]  -- 325 in base 6 (reversed for easier conversion)
def num2 : List Nat := [1, 3, 2]  -- 231 in base 5 (reversed for easier conversion)

-- State the theorem
theorem base_subtraction_equality :
  to_base_10 num1 6 - to_base_10 num2 5 = 59 := by
  sorry

end base_subtraction_equality_l327_32777


namespace line_problem_l327_32732

/-- A line in the xy-plane defined by y = 2x + 4 -/
def line (x : ℝ) : ℝ := 2 * x + 4

/-- Point P on the x-axis -/
def P (p : ℝ) : ℝ × ℝ := (p, 0)

/-- Point R where the line intersects the y-axis -/
def R : ℝ × ℝ := (0, line 0)

/-- Point Q where the line intersects the vertical line through P -/
def Q (p : ℝ) : ℝ × ℝ := (p, line p)

/-- Area of the quadrilateral OPQR -/
def area_OPQR (p : ℝ) : ℝ := p * (p + 4)

theorem line_problem (p : ℝ) (h : p > 0) :
  R.2 = 4 ∧
  Q p = (p, 2 * p + 4) ∧
  area_OPQR p = p * (p + 4) ∧
  (p = 8 → area_OPQR p = 96) ∧
  (area_OPQR p = 77 → p = 7) := by
  sorry

end line_problem_l327_32732


namespace max_value_fraction_l327_32752

theorem max_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 3 ≤ y ∧ y ≤ 6) :
  (∀ a b, -3 ≤ a ∧ a ≤ -1 ∧ 3 ≤ b ∧ b ≤ 6 → (a + b) / a ≤ (x + y) / x) →
  (x + y) / x = -2 :=
sorry

end max_value_fraction_l327_32752


namespace max_value_cos_sin_l327_32731

theorem max_value_cos_sin (x : Real) : 3 * Real.cos x + Real.sin x ≤ Real.sqrt 10 := by
  sorry

end max_value_cos_sin_l327_32731


namespace dvd_pack_discounted_price_l327_32729

/-- The price of a DVD pack after discount -/
def price_after_discount (original_price discount : ℕ) : ℕ :=
  original_price - discount

/-- Theorem: The price of a DVD pack after a $25 discount is $51, given that the original price is $76 -/
theorem dvd_pack_discounted_price :
  price_after_discount 76 25 = 51 := by
  sorry

end dvd_pack_discounted_price_l327_32729


namespace polygon_sides_l327_32720

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 2 * 360 + 180 → n = 7 :=
by sorry

end polygon_sides_l327_32720


namespace cos_300_degrees_l327_32728

theorem cos_300_degrees : Real.cos (300 * π / 180) = Real.sqrt 3 / 2 := by sorry

end cos_300_degrees_l327_32728


namespace max_fall_time_bound_l327_32787

/-- Represents the movement rules and conditions for ants on an m × m checkerboard. -/
structure AntCheckerboard (m : ℕ) :=
  (m_pos : m > 0)
  (board_size : Fin m → Fin m → Bool)
  (ant_positions : Set (Fin m × Fin m))
  (ant_directions : (Fin m × Fin m) → (Int × Int))
  (collision_rules : (Fin m × Fin m) → (Int × Int) → (Int × Int))

/-- The maximum time for the last ant to fall off the board. -/
def max_fall_time (m : ℕ) (board : AntCheckerboard m) : ℚ :=
  3 * m / 2 - 1

/-- Theorem stating that the maximum time for the last ant to fall off is 3m/2 - 1. -/
theorem max_fall_time_bound (m : ℕ) (board : AntCheckerboard m) :
  ∀ (t : ℚ), (∃ (ant : Fin m × Fin m), ant ∈ board.ant_positions) →
  t ≤ max_fall_time m board :=
sorry

end max_fall_time_bound_l327_32787


namespace find_x_l327_32715

theorem find_x : ∃ x : ℝ, (5 * x) / (180 / 3) + 80 = 81 ∧ x = 12 := by
  sorry

end find_x_l327_32715


namespace smallest_number_with_conditions_l327_32770

theorem smallest_number_with_conditions : ∃ n : ℕ, 
  (n = 1801) ∧ 
  (∀ m : ℕ, m < n → 
    (11 ∣ n) ∧ 
    (n % 2 = 1) ∧ 
    (n % 3 = 1) ∧ 
    (n % 4 = 1) ∧ 
    (n % 5 = 1) ∧ 
    (n % 6 = 1) ∧ 
    (n % 8 = 1) → 
    ¬((11 ∣ m) ∧ 
      (m % 2 = 1) ∧ 
      (m % 3 = 1) ∧ 
      (m % 4 = 1) ∧ 
      (m % 5 = 1) ∧ 
      (m % 6 = 1) ∧ 
      (m % 8 = 1))) :=
by sorry

end smallest_number_with_conditions_l327_32770


namespace hyperbola_specific_equation_l327_32797

/-- Represents a hyperbola with center at the origin -/
structure Hyperbola where
  /-- The distance from the center to a focus -/
  c : ℝ
  /-- The slope of the asymptotes -/
  m : ℝ

/-- The equation of the hyperbola given its parameters -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / (h.c^2 / (1 + h.m^2)) - y^2 / (h.c^2 * h.m^2 / (1 + h.m^2)) = 1

theorem hyperbola_specific_equation :
  let h : Hyperbola := ⟨5, 3/4⟩
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2/16 - y^2/9 = 1 :=
sorry

end hyperbola_specific_equation_l327_32797


namespace a_25_mod_26_l327_32759

/-- Definition of a_n as the integer obtained by concatenating all integers from 1 to n -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem stating that a_25 mod 26 = 13 -/
theorem a_25_mod_26 : a 25 % 26 = 13 := by sorry

end a_25_mod_26_l327_32759


namespace thirtieth_triangular_number_l327_32733

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end thirtieth_triangular_number_l327_32733


namespace mrs_white_orchard_yield_l327_32758

/-- Represents the dimensions and crop yields of Mrs. White's orchard -/
structure Orchard where
  length_paces : ℕ
  width_paces : ℕ
  feet_per_pace : ℕ
  tomato_yield_per_sqft : ℚ
  cucumber_yield_per_sqft : ℚ

/-- Calculates the expected crop yield from the orchard -/
def expected_yield (o : Orchard) : ℚ :=
  let area_sqft := (o.length_paces * o.feet_per_pace) * (o.width_paces * o.feet_per_pace)
  let half_area_sqft := area_sqft / 2
  let tomato_yield := half_area_sqft * o.tomato_yield_per_sqft
  let cucumber_yield := half_area_sqft * o.cucumber_yield_per_sqft
  tomato_yield + cucumber_yield

/-- Mrs. White's orchard -/
def mrs_white_orchard : Orchard :=
  { length_paces := 10
  , width_paces := 30
  , feet_per_pace := 3
  , tomato_yield_per_sqft := 3/4
  , cucumber_yield_per_sqft := 2/5 }

theorem mrs_white_orchard_yield :
  expected_yield mrs_white_orchard = 1552.5 := by
  sorry

end mrs_white_orchard_yield_l327_32758


namespace derivative_of_f_at_1_l327_32763

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_of_f_at_1 : 
  deriv f 1 = 2 := by sorry

end derivative_of_f_at_1_l327_32763
