import Mathlib

namespace elvis_recording_time_l156_15633

theorem elvis_recording_time (total_songs : ℕ) (studio_hours : ℕ) (writing_time_per_song : ℕ) (total_editing_time : ℕ) :
  total_songs = 10 →
  studio_hours = 5 →
  writing_time_per_song = 15 →
  total_editing_time = 30 →
  (studio_hours * 60 - total_songs * writing_time_per_song - total_editing_time) / total_songs = 12 := by
  sorry

end elvis_recording_time_l156_15633


namespace aubriella_poured_gallons_l156_15625

/-- Proves that Aubriella has poured 18 gallons into the fish tank -/
theorem aubriella_poured_gallons
  (tank_capacity : ℕ)
  (remaining_gallons : ℕ)
  (seconds_per_gallon : ℕ)
  (pouring_time_minutes : ℕ)
  (h1 : tank_capacity = 50)
  (h2 : remaining_gallons = 32)
  (h3 : seconds_per_gallon = 20)
  (h4 : pouring_time_minutes = 6) :
  tank_capacity - remaining_gallons = 18 :=
by sorry

end aubriella_poured_gallons_l156_15625


namespace sum_of_squares_of_roots_sum_of_squares_of_specific_equation_l156_15667

theorem sum_of_squares_of_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a ≠ 0 → a*x^2 + b*x + c = 0 → r₁^2 + r₂^2 = (b^2 - 2*a*c) / a^2 :=
by
  sorry

theorem sum_of_squares_of_specific_equation :
  let r₁ := (-(-14) + Real.sqrt ((-14)^2 - 4*1*8)) / (2*1)
  let r₂ := (-(-14) - Real.sqrt ((-14)^2 - 4*1*8)) / (2*1)
  r₁^2 + r₂^2 = 180 :=
by
  sorry

end sum_of_squares_of_roots_sum_of_squares_of_specific_equation_l156_15667


namespace range_of_a_l156_15669

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2*a*x - 1 + 3*a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f a 0 < f a 1) →
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ f a x = 0) →
  (1/7 < a ∧ a < 1/5) :=
sorry

end range_of_a_l156_15669


namespace triangle_shape_l156_15658

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove that if 
    a/cos(A) = b/cos(B) = c/cos(C) and sin(A) = 2sin(B)cos(C), then A = B = C. -/
theorem triangle_shape (a b c A B C : ℝ) 
    (h1 : a / Real.cos A = b / Real.cos B) 
    (h2 : b / Real.cos B = c / Real.cos C)
    (h3 : Real.sin A = 2 * Real.sin B * Real.cos C)
    (h4 : 0 < A ∧ A < π)
    (h5 : 0 < B ∧ B < π)
    (h6 : 0 < C ∧ C < π)
    (h7 : A + B + C = π) : 
  A = B ∧ B = C := by
  sorry

end triangle_shape_l156_15658


namespace road_greening_costs_l156_15631

/-- Represents a road greening project with two plans. -/
structure RoadGreeningProject where
  total_length : ℝ
  plan_a_type_a : ℝ
  plan_a_type_b : ℝ
  plan_a_cost : ℝ
  plan_b_type_a : ℝ
  plan_b_type_b : ℝ
  plan_b_cost : ℝ

/-- Calculates the cost per stem of type A and B flowers. -/
def calculate_flower_costs (project : RoadGreeningProject) : ℝ × ℝ := sorry

/-- Calculates the minimum total cost of the project. -/
def calculate_min_cost (project : RoadGreeningProject) : ℝ := sorry

/-- Theorem stating the correct flower costs and minimum project cost. -/
theorem road_greening_costs (project : RoadGreeningProject) 
  (h1 : project.total_length = 1500)
  (h2 : project.plan_a_type_a = 2)
  (h3 : project.plan_a_type_b = 3)
  (h4 : project.plan_a_cost = 22)
  (h5 : project.plan_b_type_a = 1)
  (h6 : project.plan_b_type_b = 5)
  (h7 : project.plan_b_cost = 25) :
  let (cost_a, cost_b) := calculate_flower_costs project
  calculate_flower_costs project = (5, 4) ∧ 
  calculate_min_cost project = 36000 := by
  sorry

end road_greening_costs_l156_15631


namespace four_color_plane_exists_l156_15673

-- Define the color type
inductive Color
| Red | Blue | Green | Yellow | Purple

-- Define the space as a type of points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the condition that each color appears at least once
axiom all_colors_present : ∀ c : Color, ∃ p : Point, coloring p = c

-- Define a plane
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a function to check if a point is on a plane
def on_plane (plane : Plane) (point : Point) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

-- Define a function to count distinct colors on a plane
def count_colors_on_plane (plane : Plane) : ℕ := sorry

-- The main theorem
theorem four_color_plane_exists :
  ∃ plane : Plane, count_colors_on_plane plane ≥ 4 := sorry

end four_color_plane_exists_l156_15673


namespace robotics_club_max_participants_l156_15650

theorem robotics_club_max_participants 
  (physics : Finset ℕ)
  (math : Finset ℕ)
  (programming : Finset ℕ)
  (h1 : physics.card = 8)
  (h2 : math.card = 7)
  (h3 : programming.card = 11)
  (h4 : (physics ∩ math).card ≥ 2)
  (h5 : (math ∩ programming).card ≥ 3)
  (h6 : (physics ∩ programming).card ≥ 4) :
  (physics ∪ math ∪ programming).card ≤ 19 :=
sorry

end robotics_club_max_participants_l156_15650


namespace three_card_selection_count_l156_15653

/-- The number of ways to select 3 different cards in order from a set of 13 cards -/
def select_three_cards : ℕ := 13 * 12 * 11

/-- Theorem stating that selecting 3 different cards in order from 13 cards results in 1716 possibilities -/
theorem three_card_selection_count : select_three_cards = 1716 := by
  sorry

end three_card_selection_count_l156_15653


namespace fraction_product_power_l156_15637

theorem fraction_product_power : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end fraction_product_power_l156_15637


namespace last_two_digits_of_sum_of_factorials_15_l156_15634

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sumOfFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def lastTwoDigits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_of_sum_of_factorials_15 :
  lastTwoDigits (sumOfFactorials 15) = 13 := by
  sorry

end last_two_digits_of_sum_of_factorials_15_l156_15634


namespace isosceles_points_count_l156_15635

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the property of being acute
def isAcute (t : Triangle) : Prop := sorry

-- Define the ordering of side lengths
def sideOrdering (t : Triangle) : Prop := sorry

-- Define the property of a point P making isosceles triangles with AB and BC
def makesIsosceles (P : ℝ × ℝ) (t : Triangle) : Prop := sorry

-- The main theorem
theorem isosceles_points_count (t : Triangle) : 
  isAcute t → sideOrdering t → ∃! (points : Finset (ℝ × ℝ)), 
    Finset.card points = 15 ∧ 
    ∀ P ∈ points, makesIsosceles P t := by
  sorry

end isosceles_points_count_l156_15635


namespace grace_garden_medium_bed_rows_l156_15640

/-- Represents a raised bed garden with large and medium beds -/
structure RaisedBedGarden where
  large_beds : Nat
  medium_beds : Nat
  large_bed_rows : Nat
  large_bed_seeds_per_row : Nat
  medium_bed_seeds_per_row : Nat
  total_seeds : Nat

/-- Calculates the number of rows in medium beds -/
def medium_bed_rows (garden : RaisedBedGarden) : Nat :=
  let large_bed_seeds := garden.large_beds * garden.large_bed_rows * garden.large_bed_seeds_per_row
  let medium_bed_seeds := garden.total_seeds - large_bed_seeds
  medium_bed_seeds / garden.medium_bed_seeds_per_row

/-- Theorem stating that for the given garden configuration, medium beds have 6 rows -/
theorem grace_garden_medium_bed_rows :
  let garden : RaisedBedGarden := {
    large_beds := 2,
    medium_beds := 2,
    large_bed_rows := 4,
    large_bed_seeds_per_row := 25,
    medium_bed_seeds_per_row := 20,
    total_seeds := 320
  }
  medium_bed_rows garden = 6 := by
  sorry

end grace_garden_medium_bed_rows_l156_15640


namespace optimal_playground_max_area_l156_15672

/-- Represents a rectangular playground with given constraints -/
structure Playground where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 190
  length_constraint : length ≥ 100
  width_constraint : width ≥ 60

/-- The area of a playground -/
def area (p : Playground) : ℝ := p.length * p.width

/-- The optimal playground dimensions -/
def optimal_playground : Playground := {
  length := 100,
  width := 90,
  perimeter_constraint := by sorry,
  length_constraint := by sorry,
  width_constraint := by sorry
}

/-- Theorem stating that the optimal playground has the maximum area -/
theorem optimal_playground_max_area :
  ∀ p : Playground, area p ≤ area optimal_playground := by sorry

end optimal_playground_max_area_l156_15672


namespace profit_margin_increase_l156_15682

theorem profit_margin_increase (initial_margin : ℝ) (final_margin : ℝ) 
  (price_increase : ℝ) : 
  initial_margin = 0.25 →
  final_margin = 0.40 →
  price_increase = (1 + final_margin) / (1 + initial_margin) - 1 →
  price_increase = 0.12 := by
  sorry

#check profit_margin_increase

end profit_margin_increase_l156_15682


namespace calculation_result_l156_15670

theorem calculation_result : 2002 * 20032003 - 2003 * 20022002 = 0 := by
  sorry

end calculation_result_l156_15670


namespace rhombus_diagonal_l156_15604

theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) :
  area = 300 ∧ d2 = 20 ∧ area = (d1 * d2) / 2 → d1 = 30 := by
  sorry

end rhombus_diagonal_l156_15604


namespace red_balls_count_l156_15678

/-- Given a bag with 15 balls, prove that there are 6 red balls if the probability
    of drawing two red balls at random at the same time is 2/35. -/
theorem red_balls_count (total : ℕ) (prob : ℚ) (h1 : total = 15) (h2 : prob = 2/35) :
  ∃ r : ℕ, r = 6 ∧
  (r : ℚ) / total * ((r - 1) : ℚ) / (total - 1) = prob :=
sorry

end red_balls_count_l156_15678


namespace arithmetic_mean_of_numbers_l156_15690

def numbers : List ℝ := [18, 27, 45]

theorem arithmetic_mean_of_numbers : 
  (List.sum numbers) / (List.length numbers) = 30 := by
  sorry

end arithmetic_mean_of_numbers_l156_15690


namespace ln_lower_bound_l156_15630

theorem ln_lower_bound (n : ℕ) (k : ℕ) (h : k = (Nat.factorization n).support.card) :
  Real.log n ≥ k * Real.log 2 := by
  sorry

end ln_lower_bound_l156_15630


namespace sally_lost_balloons_l156_15623

theorem sally_lost_balloons (initial_orange : ℕ) (current_orange : ℕ) 
  (h1 : initial_orange = 9) 
  (h2 : current_orange = 7) : 
  initial_orange - current_orange = 2 := by
  sorry

end sally_lost_balloons_l156_15623


namespace simplify_fraction_product_l156_15695

theorem simplify_fraction_product : 8 * (15 / 4) * (-40 / 45) = -80 / 3 := by
  sorry

end simplify_fraction_product_l156_15695


namespace quadratic_expression_equality_l156_15614

theorem quadratic_expression_equality : ∃ (a b c : ℝ), 
  (∀ x, 2 * (x - 3)^2 - 12 = a * x^2 + b * x + c) ∧ 
  (10 * a - b - 4 * c = 8) := by
  sorry

end quadratic_expression_equality_l156_15614


namespace triangle_ABC_dot_product_l156_15620

def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (5, 6)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem triangle_ABC_dot_product :
  dot_product vector_AB vector_AC = 9 := by sorry

end triangle_ABC_dot_product_l156_15620


namespace square_circle_union_area_l156_15677

/-- The area of the union of a square with side length 8 and a circle with radius 12
    centered at the center of the square is equal to 144π. -/
theorem square_circle_union_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let union_area : ℝ := max square_area circle_area
  union_area = 144 * π := by
  sorry

end square_circle_union_area_l156_15677


namespace percentage_excess_l156_15610

/-- Given two positive real numbers A and B with a specific ratio and sum condition,
    this theorem proves the formula for the percentage by which B exceeds A. -/
theorem percentage_excess (x y A B : ℝ) : 
  x > 0 → y > 0 → A > 0 → B > 0 →
  A / B = (5 * y^2) / (6 * x) →
  2 * x + 3 * y = 42 →
  ((B - A) / A) * 100 = ((126 - 9*y - 5*y^2) / (5*y^2)) * 100 := by
  sorry

end percentage_excess_l156_15610


namespace unique_solution_iff_k_eq_neg_three_fourths_l156_15641

/-- The equation (x + 3) / (kx - 2) = x has exactly one solution if and only if k = -3/4 -/
theorem unique_solution_iff_k_eq_neg_three_fourths (k : ℝ) : 
  (∃! x : ℝ, (x + 3) / (k * x - 2) = x) ↔ k = -3/4 := by
sorry

end unique_solution_iff_k_eq_neg_three_fourths_l156_15641


namespace percentage_married_students_l156_15688

theorem percentage_married_students (T : ℝ) (T_pos : T > 0) : 
  let male_students := 0.7 * T
  let female_students := 0.3 * T
  let married_male_students := (2/7) * male_students
  let married_female_students := (1/3) * female_students
  let total_married_students := married_male_students + married_female_students
  (total_married_students / T) * 100 = 30 := by sorry

end percentage_married_students_l156_15688


namespace perpendicular_bisector_of_intersection_l156_15666

-- Define the two circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 13
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersection :
  ∃ (A B : ℝ × ℝ), 
    (circle1 A.1 A.2 ∧ circle2 A.1 A.2) ∧ 
    (circle1 B.1 B.2 ∧ circle2 B.1 B.2) ∧ 
    A ≠ B ∧
    (∀ (x y : ℝ), perp_bisector x y ↔ 
      (x - (A.1 + B.1)/2)^2 + (y - (A.2 + B.2)/2)^2 = 
      ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4) :=
sorry

end perpendicular_bisector_of_intersection_l156_15666


namespace ages_sum_l156_15644

/-- Represents the ages of three people A, B, and C --/
structure Ages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The conditions of the problem --/
def problem_conditions (ages : Ages) : Prop :=
  ages.b = 30 ∧ 
  ∃ x : ℕ, x > 0 ∧ 
    ages.a - 10 = x ∧
    ages.b - 10 = 2 * x ∧
    ages.c - 10 = 3 * x

/-- The theorem to prove --/
theorem ages_sum (ages : Ages) : 
  problem_conditions ages → ages.a + ages.b + ages.c = 90 :=
by
  sorry

end ages_sum_l156_15644


namespace parallel_lines_theorem_l156_15607

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a b c d e f : ℝ) : Prop :=
  a / b = d / e

/-- Given two lines l₁: ax + 3y - 3 = 0 and l₂: 4x + 6y - 1 = 0,
    if they are parallel, then a = 2 -/
theorem parallel_lines_theorem (a : ℝ) :
  parallel_lines a 3 (-3) 4 6 (-1) → a = 2 := by
  sorry

end parallel_lines_theorem_l156_15607


namespace binomial_10_choose_6_l156_15674

theorem binomial_10_choose_6 : Nat.choose 10 6 = 210 := by
  sorry

end binomial_10_choose_6_l156_15674


namespace intersection_point_l156_15608

-- Define the line
def line (x y : ℝ) : Prop := 5 * y - 2 * x = 10

-- Define a point on the x-axis
def on_x_axis (x y : ℝ) : Prop := y = 0

-- Theorem: The point (-5, 0) is on the line and the x-axis
theorem intersection_point : 
  line (-5) 0 ∧ on_x_axis (-5) 0 := by sorry

end intersection_point_l156_15608


namespace number_problem_l156_15619

theorem number_problem (x : ℝ) : 0.20 * x - 4 = 6 → x = 50 := by
  sorry

end number_problem_l156_15619


namespace no_three_digit_multiple_base_l156_15615

/-- Definition of a valid base for a number x -/
def valid_base (x : ℕ) (b : ℕ) : Prop :=
  b ≥ 2 ∧ b ≤ 10 ∧ (b - 1)^4 < x ∧ x < b^4

/-- Definition of a three-digit number in base b -/
def three_digit (x : ℕ) (b : ℕ) : Prop :=
  b^2 ≤ x ∧ x < b^3

/-- Main theorem: No three-digit number represents multiple values in different bases -/
theorem no_three_digit_multiple_base :
  ¬ ∃ (x : ℕ) (b1 b2 : ℕ), x < 10000 ∧ b1 < b2 ∧
    valid_base x b1 ∧ valid_base x b2 ∧
    three_digit x b1 ∧ three_digit x b2 :=
sorry

end no_three_digit_multiple_base_l156_15615


namespace line_plane_perpendicularity_l156_15656

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a b : Line) (α : Plane) :
  parallel a α → perpendicular b α → perpendicularLines a b :=
sorry

end line_plane_perpendicularity_l156_15656


namespace max_plates_on_table_l156_15676

/-- The radius of the table in meters -/
def table_radius : ℝ := 1

/-- The radius of each plate in meters -/
def plate_radius : ℝ := 0.15

/-- The maximum number of plates that can fit on the table -/
def max_plates : ℕ := 44

/-- Theorem stating that the maximum number of plates that can fit on the table is 44 -/
theorem max_plates_on_table :
  ∀ k : ℕ, 
    (k : ℝ) * π * plate_radius^2 ≤ π * table_radius^2 ↔ k ≤ max_plates :=
by sorry

end max_plates_on_table_l156_15676


namespace root_transformation_l156_15691

theorem root_transformation {b : ℝ} (a b c d : ℝ) :
  (a^4 - b*a - 3 = 0) ∧
  (b^4 - b*b - 3 = 0) ∧
  (c^4 - b*c - 3 = 0) ∧
  (d^4 - b*d - 3 = 0) →
  (3*(-1/a)^4 - b*(-1/a)^3 - 1 = 0) ∧
  (3*(-1/b)^4 - b*(-1/b)^3 - 1 = 0) ∧
  (3*(-1/c)^4 - b*(-1/c)^3 - 1 = 0) ∧
  (3*(-1/d)^4 - b*(-1/d)^3 - 1 = 0) :=
by sorry

end root_transformation_l156_15691


namespace complex_in_second_quadrant_l156_15692

theorem complex_in_second_quadrant : 
  let z : ℂ := Complex.mk (Real.cos 3) (Real.sin 3)
  Complex.re z < 0 ∧ Complex.im z > 0 := by
  sorry

end complex_in_second_quadrant_l156_15692


namespace contract_completion_hours_l156_15611

/-- Represents the contract completion problem -/
structure ContractProblem where
  total_days : ℕ
  initial_men : ℕ
  initial_hours : ℕ
  days_passed : ℕ
  work_completed : ℚ
  additional_men : ℕ

/-- Calculates the required daily work hours to complete the contract on time -/
def required_hours (p : ContractProblem) : ℚ :=
  let total_man_hours := p.initial_men * p.initial_hours * p.total_days
  let remaining_man_hours := (1 - p.work_completed) * total_man_hours
  let remaining_days := p.total_days - p.days_passed
  let total_men := p.initial_men + p.additional_men
  remaining_man_hours / (total_men * remaining_days)

/-- Theorem stating that the required work hours for the given problem is approximately 7.16 -/
theorem contract_completion_hours (p : ContractProblem) 
  (h1 : p.total_days = 46)
  (h2 : p.initial_men = 117)
  (h3 : p.initial_hours = 8)
  (h4 : p.days_passed = 33)
  (h5 : p.work_completed = 4/7)
  (h6 : p.additional_men = 81) :
  ∃ ε > 0, abs (required_hours p - 7.16) < ε :=
sorry

end contract_completion_hours_l156_15611


namespace inequality_system_solution_l156_15697

def satisfies_inequalities (x : ℤ) : Prop :=
  (2 * (x - 1) < x + 3) ∧ ((2 * x + 1) / 3 > x - 1)

def non_negative_integer_solutions : Set ℤ :=
  {x : ℤ | x ≥ 0 ∧ satisfies_inequalities x}

theorem inequality_system_solution :
  non_negative_integer_solutions = {0, 1, 2, 3} :=
sorry

end inequality_system_solution_l156_15697


namespace min_minutes_for_plan_b_cheaper_l156_15645

def plan_a_cost (minutes : ℕ) : ℚ := 15 + (12 / 100) * minutes
def plan_b_cost (minutes : ℕ) : ℚ := 30 + (6 / 100) * minutes

theorem min_minutes_for_plan_b_cheaper : 
  ∀ m : ℕ, m ≥ 251 → plan_b_cost m < plan_a_cost m ∧
  ∀ n : ℕ, n < 251 → plan_a_cost n ≤ plan_b_cost n :=
by sorry

end min_minutes_for_plan_b_cheaper_l156_15645


namespace base_eight_digits_of_512_l156_15609

theorem base_eight_digits_of_512 : ∃ n : ℕ, n > 0 ∧ 8^(n-1) ≤ 512 ∧ 512 < 8^n ∧ n = 4 := by
  sorry

end base_eight_digits_of_512_l156_15609


namespace cube_roots_of_unity_sum_l156_15629

theorem cube_roots_of_unity_sum (ω ω_conj : ℂ) : 
  ω = (-1 + Complex.I * Real.sqrt 3) / 2 →
  ω_conj = (-1 - Complex.I * Real.sqrt 3) / 2 →
  ω^3 = 1 →
  ω_conj^3 = 1 →
  ω^4 + ω_conj^4 - 2 = -3 := by
  sorry

end cube_roots_of_unity_sum_l156_15629


namespace particular_number_addition_l156_15616

theorem particular_number_addition : ∃ x : ℝ, 0.46 + x = 0.72 ∧ x = 0.26 := by
  sorry

end particular_number_addition_l156_15616


namespace midpoint_trajectory_l156_15626

/-- Given a circle and a moving chord, prove the trajectory of the chord's midpoint -/
theorem midpoint_trajectory (x y : ℝ) :
  (∀ a b : ℝ, a^2 + b^2 = 25 → (x - a)^2 + (y - b)^2 ≤ 9) →
  x^2 + y^2 = 16 :=
sorry

end midpoint_trajectory_l156_15626


namespace absolute_value_inequalities_l156_15681

theorem absolute_value_inequalities (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| > a → a < 3) ∧
  (∀ x : ℝ, |x - 1| - |x + 3| < a → a > 4) := by
  sorry

end absolute_value_inequalities_l156_15681


namespace multiples_count_multiples_count_equals_188_l156_15659

theorem multiples_count : ℕ :=
  let range_start := 1
  let range_end := 600
  let count_multiples_of (n : ℕ) := (range_end / n : ℕ)
  let multiples_of_5 := count_multiples_of 5
  let multiples_of_7 := count_multiples_of 7
  let multiples_of_35 := count_multiples_of 35
  multiples_of_5 + multiples_of_7 - multiples_of_35

theorem multiples_count_equals_188 : multiples_count = 188 := by
  sorry

end multiples_count_multiples_count_equals_188_l156_15659


namespace units_digit_of_fraction_l156_15655

def product_20_to_30 : ℕ := 20 * 21 * 22 * 23 * 24 * 25 * 26 * 27 * 28 * 29 * 30

theorem units_digit_of_fraction (h : product_20_to_30 % 8000 = 6) :
  (product_20_to_30 / 8000) % 10 = 6 := by
  sorry

end units_digit_of_fraction_l156_15655


namespace captain_age_is_27_l156_15683

/-- Represents the age of the cricket team captain -/
def captain_age : ℕ := sorry

/-- Represents the age of the wicket keeper -/
def wicket_keeper_age : ℕ := sorry

/-- The number of players in the cricket team -/
def team_size : ℕ := 11

/-- The average age of the whole team -/
def team_average_age : ℕ := 24

theorem captain_age_is_27 :
  captain_age = 27 ∧
  wicket_keeper_age = captain_age + 3 ∧
  team_size * team_average_age = captain_age + wicket_keeper_age + (team_size - 2) * (team_average_age - 1) :=
by sorry

end captain_age_is_27_l156_15683


namespace max_side_length_exists_max_side_length_l156_15606

/-- A triangle with integer side lengths and perimeter 24 -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  perimeter : a + b + c = 24

/-- The maximum side length of a triangle is 11 -/
theorem max_side_length (t : Triangle) : t.a ≤ 11 ∧ t.b ≤ 11 ∧ t.c ≤ 11 :=
sorry

/-- There exists a triangle with maximum side length 11 -/
theorem exists_max_side_length : ∃ (t : Triangle), t.a = 11 ∨ t.b = 11 ∨ t.c = 11 :=
sorry

end max_side_length_exists_max_side_length_l156_15606


namespace rental_problem_l156_15664

/-- Rental problem theorem -/
theorem rental_problem (first_hour_rate : ℕ) (additional_hour_rate : ℕ) (total_paid : ℕ) :
  first_hour_rate = 25 →
  additional_hour_rate = 10 →
  total_paid = 125 →
  ∃ (hours : ℕ), hours = 11 ∧ 
    total_paid = first_hour_rate + (hours - 1) * additional_hour_rate :=
by
  sorry

end rental_problem_l156_15664


namespace refrigerator_loss_percentage_l156_15648

/-- Represents the problem of calculating the loss percentage on a refrigerator. -/
def RefrigeratorLossProblem (refrigerator_cp mobile_cp : ℕ) (mobile_profit overall_profit : ℕ) : Prop :=
  let refrigerator_sp := refrigerator_cp + mobile_cp + overall_profit - (mobile_cp + mobile_cp * mobile_profit / 100)
  let loss_percentage := (refrigerator_cp - refrigerator_sp) * 100 / refrigerator_cp
  loss_percentage = 5

/-- The main theorem stating that given the problem conditions, the loss percentage on the refrigerator is 5%. -/
theorem refrigerator_loss_percentage :
  RefrigeratorLossProblem 15000 8000 10 50 := by
  sorry

end refrigerator_loss_percentage_l156_15648


namespace equation_solution_l156_15649

theorem equation_solution (y : ℝ) (h : (1 : ℝ) / 3 + 1 / y = 7 / 9) : y = 9 / 4 := by
  sorry

end equation_solution_l156_15649


namespace bert_spending_l156_15657

theorem bert_spending (n : ℝ) : 
  (3/4 * n - 9) / 2 = 12 → n = 44 := by sorry

end bert_spending_l156_15657


namespace intersection_when_a_is_one_subset_condition_equivalent_to_range_l156_15646

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem intersection_when_a_is_one :
  A 1 ∩ B = {x | 1 < x ∧ x < 2} := by sorry

theorem subset_condition_equivalent_to_range :
  ∀ a : ℝ, A a ⊆ A a ∩ B ↔ 2 ≤ a ∧ a ≤ 4 := by sorry

end intersection_when_a_is_one_subset_condition_equivalent_to_range_l156_15646


namespace endangered_animal_population_l156_15687

/-- The population of an endangered animal after n years, given an initial population and annual decrease rate. -/
def population (m : ℝ) (r : ℝ) (n : ℕ) : ℝ := m * (1 - r) ^ n

/-- Theorem stating that given specific conditions, the population after 3 years will be 5832. -/
theorem endangered_animal_population :
  let m : ℝ := 8000  -- Initial population
  let r : ℝ := 0.1   -- Annual decrease rate (10%)
  let n : ℕ := 3     -- Number of years
  population m r n = 5832 := by
  sorry

end endangered_animal_population_l156_15687


namespace project_completion_time_l156_15647

/-- The number of days it takes for person A to complete the project alone -/
def A_days : ℕ := 20

/-- The number of days it takes for both A and B to complete the project together,
    with A quitting 10 days before completion -/
def total_days : ℕ := 18

/-- The number of days A works before quitting -/
def A_work_days : ℕ := total_days - 10

/-- The rate at which person A completes the project per day -/
def A_rate : ℚ := 1 / A_days

theorem project_completion_time (B_days : ℕ) :
  (A_work_days : ℚ) * (A_rate + 1 / B_days) + (10 : ℚ) * (1 / B_days) = 1 →
  B_days = 30 := by sorry

end project_completion_time_l156_15647


namespace mn_positive_necessary_not_sufficient_l156_15632

-- Define the condition for a hyperbola with foci on the x-axis
def is_hyperbola_x_axis (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0

-- Define the equation
def equation (m n x y : ℝ) : Prop :=
  m * x^2 - n * y^2 = 1

-- Theorem statement
theorem mn_positive_necessary_not_sufficient :
  ∀ m n : ℝ,
    (is_hyperbola_x_axis m n → m * n > 0) ∧
    ¬(m * n > 0 → is_hyperbola_x_axis m n) :=
by sorry

end mn_positive_necessary_not_sufficient_l156_15632


namespace ella_coin_value_l156_15621

/-- Represents the number of coins Ella has -/
def total_coins : ℕ := 18

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the number of nickels Ella has -/
def nickels : ℕ := sorry

/-- Represents the number of dimes Ella has -/
def dimes : ℕ := sorry

/-- The total number of coins is the sum of nickels and dimes -/
axiom coin_sum : nickels + dimes = total_coins

/-- If Ella had two more dimes, she would have an equal number of nickels and dimes -/
axiom equal_with_two_more : nickels = dimes + 2

/-- The theorem to be proved -/
theorem ella_coin_value : 
  nickels * nickel_value + dimes * dime_value = 130 :=
sorry

end ella_coin_value_l156_15621


namespace derek_savings_l156_15699

theorem derek_savings (a₁ a₂ : ℕ) (sum : ℕ) : 
  a₁ = 2 → a₂ = 4 → sum = 4096 → 
  ∃ (r : ℚ), r > 0 ∧ 
    (∀ n : ℕ, n > 0 → n ≤ 12 → a₁ * r^(n-1) = a₂ * r^(n-2)) ∧
    (sum = a₁ * (1 - r^12) / (1 - r)) →
  a₁ * r^2 = 8 := by
sorry

end derek_savings_l156_15699


namespace least_value_quadratic_l156_15600

theorem least_value_quadratic (x : ℝ) :
  (4 * x^2 + 7 * x + 3 = 5) → x ≥ -2 :=
by sorry

end least_value_quadratic_l156_15600


namespace magnitude_of_z_l156_15622

theorem magnitude_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end magnitude_of_z_l156_15622


namespace average_of_sqrt_equation_l156_15601

theorem average_of_sqrt_equation (x : ℝ) :
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, Real.sqrt (3 * x^2 + 4 * x + 1) = Real.sqrt 37 ↔ x = x₁ ∨ x = x₂) ∧
  (x₁ + x₂) / 2 = -2/3) :=
by sorry

end average_of_sqrt_equation_l156_15601


namespace monic_polynomial_square_decomposition_l156_15618

theorem monic_polynomial_square_decomposition
  (P : Polynomial ℤ)
  (h_monic : P.Monic)
  (h_even_degree : Even P.degree)
  (h_infinite_squares : ∃ S : Set ℤ, Infinite S ∧ ∀ x ∈ S, ∃ y : ℤ, 0 < y ∧ P.eval x = y^2) :
  ∃ Q : Polynomial ℤ, P = Q^2 :=
sorry

end monic_polynomial_square_decomposition_l156_15618


namespace gcd_problem_l156_15665

theorem gcd_problem (b : ℤ) (h : ∃ (k : ℤ), b = 7 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 34 * b + 76) (b + 16) = 7 := by
  sorry

end gcd_problem_l156_15665


namespace derivative_f_at_2_l156_15680

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_f_at_2 : 
  deriv f 2 = 3 * Real.exp 2 := by
  sorry

end derivative_f_at_2_l156_15680


namespace sum_of_last_two_digits_of_9_pow_2023_l156_15602

theorem sum_of_last_two_digits_of_9_pow_2023 : ∃ (a b : ℕ), 
  (9^2023 : ℕ) % 100 = 10 * a + b ∧ a + b = 11 := by sorry

end sum_of_last_two_digits_of_9_pow_2023_l156_15602


namespace f_inequality_l156_15693

noncomputable def f (x : ℝ) : ℝ := Real.log (|x| + 1) / Real.log (1/2) + 1 / (x^2 + 1)

theorem f_inequality (x : ℝ) : f x > f (2*x - 1) ↔ x > 1 ∨ x < 1/3 := by
  sorry

end f_inequality_l156_15693


namespace square_of_binomial_b_value_l156_15661

theorem square_of_binomial_b_value (p b : ℝ) : 
  (∃ q : ℝ, ∀ x : ℝ, x^2 + p*x + b = (x + q)^2) → 
  p = -10 → 
  b = 25 := by
sorry

end square_of_binomial_b_value_l156_15661


namespace difference_greater_than_one_l156_15685

theorem difference_greater_than_one (x : ℕ+) :
  (x.val + 3 : ℚ) / 2 - (2 * x.val - 1 : ℚ) / 3 > 1 ↔ x.val < 5 := by
  sorry

end difference_greater_than_one_l156_15685


namespace jan1_2010_is_sunday_l156_15654

-- Define days of the week
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

-- Theorem statement
theorem jan1_2010_is_sunday :
  advanceDay DayOfWeek.Saturday 3653 = DayOfWeek.Sunday := by
  sorry


end jan1_2010_is_sunday_l156_15654


namespace contradiction_proof_l156_15651

theorem contradiction_proof (x a b : ℝ) : 
  x^2 - (a + b)*x - a*b ≠ 0 → x ≠ a ∧ x ≠ b := by
  sorry

end contradiction_proof_l156_15651


namespace trapezium_height_l156_15698

theorem trapezium_height (a b area : ℝ) (ha : a = 20) (hb : b = 18) (harea : area = 247) :
  (2 * area) / (a + b) = 13 := by
  sorry

end trapezium_height_l156_15698


namespace riverside_park_adjustment_plans_l156_15638

/-- Represents the number of riverside theme parks -/
def total_parks : ℕ := 7

/-- Represents the number of parks to be removed -/
def parks_to_remove : ℕ := 2

/-- Represents the number of parks that can be adjusted (excluding the ends) -/
def adjustable_parks : ℕ := total_parks - 2

/-- Represents the number of adjacent park pairs that cannot be removed together -/
def adjacent_pairs : ℕ := adjustable_parks - 1

theorem riverside_park_adjustment_plans :
  (adjustable_parks.choose parks_to_remove) - adjacent_pairs = 6 := by
  sorry

end riverside_park_adjustment_plans_l156_15638


namespace tourist_growth_rate_l156_15679

theorem tourist_growth_rate (x : ℝ) : 
  (1 - 0.4) * (1 - 0.5) * (1 + x) = 2 :=
by sorry

end tourist_growth_rate_l156_15679


namespace weight_of_A_l156_15612

theorem weight_of_A (a b c d : ℝ) : 
  (a + b + c) / 3 = 70 →
  (a + b + c + d) / 4 = 70 →
  ((b + c + d + (d + 3)) / 4 = 68) →
  a = 81 := by sorry

end weight_of_A_l156_15612


namespace steve_jellybeans_l156_15652

/-- Given the following conditions:
    - Matilda has half as many jellybeans as Matt
    - Matt has ten times as many jellybeans as Steve
    - Matilda has 420 jellybeans
    Prove that Steve has 84 jellybeans -/
theorem steve_jellybeans (steve matt matilda : ℕ) 
  (h1 : matilda = matt / 2)
  (h2 : matt = 10 * steve)
  (h3 : matilda = 420) :
  steve = 84 := by
  sorry

end steve_jellybeans_l156_15652


namespace jose_age_is_19_l156_15624

-- Define the ages of the individuals
def inez_age : ℕ := 18
def alice_age : ℕ := inez_age - 3
def zack_age : ℕ := inez_age + 5
def jose_age : ℕ := zack_age - 4

-- Theorem to prove Jose's age
theorem jose_age_is_19 : jose_age = 19 := by
  sorry

end jose_age_is_19_l156_15624


namespace line_equidistant_point_value_l156_15660

/-- A line passing through (4, 4) with slope 0.5, equidistant from (0, A) and (12, 8), implies A = 32 -/
theorem line_equidistant_point_value (A : ℝ) : 
  let line_point : ℝ × ℝ := (4, 4)
  let line_slope : ℝ := 0.5
  let point_P : ℝ × ℝ := (0, A)
  let point_Q : ℝ × ℝ := (12, 8)
  (∃ (line : ℝ → ℝ), 
    (line (line_point.1) = line_point.2) ∧ 
    ((line (line_point.1 + 1) - line (line_point.1)) / 1 = line_slope) ∧
    (∃ (midpoint : ℝ × ℝ), 
      (midpoint.1 = (point_P.1 + point_Q.1) / 2) ∧
      (midpoint.2 = (point_P.2 + point_Q.2) / 2) ∧
      (line midpoint.1 = midpoint.2))) →
  A = 32 := by
sorry

end line_equidistant_point_value_l156_15660


namespace exist_special_pair_l156_15671

theorem exist_special_pair : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  ((7^7 : ℕ) ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) ∧ 
  (((a.val = 18 ∧ b.val = 1) ∨ (a.val = 1 ∧ b.val = 18))) :=
by sorry

end exist_special_pair_l156_15671


namespace tan_theta_in_terms_of_x_l156_15663

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (acute_θ : 0 < θ ∧ θ < π/2) 
  (x_gt_1 : x > 1) 
  (h : Real.sin (θ/2) = Real.sqrt ((x + 1)/(2*x))) : 
  Real.tan θ = Real.sqrt (2*x - 1) / (x - 1) := by
  sorry

end tan_theta_in_terms_of_x_l156_15663


namespace mean_of_four_numbers_with_sum_half_l156_15694

theorem mean_of_four_numbers_with_sum_half (a b c d : ℚ) 
  (sum_condition : a + b + c + d = 1/2) : 
  (a + b + c + d) / 4 = 1/8 := by
sorry

end mean_of_four_numbers_with_sum_half_l156_15694


namespace jim_buicks_count_l156_15617

/-- The number of model cars Jim has for each brand -/
structure ModelCars where
  ford : ℕ
  buick : ℕ
  chevy : ℕ

/-- Jim's collection of model cars satisfying the given conditions -/
def jim_collection : ModelCars → Prop
  | ⟨f, b, c⟩ => f + b + c = 301 ∧ b = 4 * f ∧ f = 2 * c + 3

theorem jim_buicks_count (cars : ModelCars) (h : jim_collection cars) : cars.buick = 220 := by
  sorry

end jim_buicks_count_l156_15617


namespace square_cut_in_half_l156_15636

/-- A square with side length 8 is cut in half to create two congruent rectangles. -/
theorem square_cut_in_half (square_side : ℝ) (rect_width rect_height : ℝ) : 
  square_side = 8 →
  rect_width * rect_height = square_side * square_side / 2 →
  rect_width = square_side ∨ rect_height = square_side →
  (rect_width = 4 ∧ rect_height = 8) ∨ (rect_width = 8 ∧ rect_height = 4) := by
  sorry

end square_cut_in_half_l156_15636


namespace vertex_in_third_quadrant_l156_15642

/-- Definition of the parabola --/
def parabola (x : ℝ) : ℝ := -2 * (x + 3)^2 - 21

/-- Definition of the vertex of the parabola --/
def vertex : ℝ × ℝ := (-3, parabola (-3))

/-- Definition of the third quadrant --/
def in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

/-- Theorem: The vertex of the parabola is in the third quadrant --/
theorem vertex_in_third_quadrant : in_third_quadrant vertex := by
  sorry

end vertex_in_third_quadrant_l156_15642


namespace range_of_x_when_m_is_2_range_of_m_when_p_necessary_not_sufficient_l156_15675

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 6*x + 5 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Part 1
theorem range_of_x_when_m_is_2 :
  ∀ x : ℝ, (p x ∧ q x 2) → (1 ≤ x ∧ x ≤ 3) :=
sorry

-- Part 2
theorem range_of_m_when_p_necessary_not_sufficient :
  ∀ m : ℝ, (m > 0 ∧ (∀ x : ℝ, q x m → p x) ∧ (∃ x : ℝ, p x ∧ ¬(q x m))) → m ≥ 4 :=
sorry

end range_of_x_when_m_is_2_range_of_m_when_p_necessary_not_sufficient_l156_15675


namespace sale_markdown_l156_15628

theorem sale_markdown (regular_price sale_price : ℝ) 
  (h : sale_price * (1 + 0.25) = regular_price) :
  (regular_price - sale_price) / regular_price = 0.2 := by
sorry

end sale_markdown_l156_15628


namespace special_cubes_in_4x5x6_prism_l156_15662

/-- Represents a rectangular prism with integer dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of unit cubes in a rectangular prism that have
    either exactly one face on the surface or no faces on the surface -/
def count_special_cubes (prism : RectangularPrism) : ℕ :=
  let interior_cubes := (prism.length - 2) * (prism.width - 2) * (prism.height - 2)
  let one_face_cubes := 2 * ((prism.width - 2) * (prism.height - 2) +
                             (prism.length - 2) * (prism.height - 2) +
                             (prism.length - 2) * (prism.width - 2))
  interior_cubes + one_face_cubes

/-- The main theorem stating that a 4x5x6 prism has 76 special cubes -/
theorem special_cubes_in_4x5x6_prism :
  count_special_cubes ⟨4, 5, 6⟩ = 76 := by
  sorry

end special_cubes_in_4x5x6_prism_l156_15662


namespace common_chord_length_l156_15627

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_C A.1 A.2 ∧
  circle_O B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem common_chord_length (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 95 := by
  sorry

end common_chord_length_l156_15627


namespace geometric_sequence_property_l156_15613

def is_geometric_sequence (α : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, α (n + 1) = α n * r

theorem geometric_sequence_property (α : ℕ → ℝ) (h_geo : is_geometric_sequence α) 
  (h_prod : α 4 * α 5 * α 6 = 27) : α 5 = 3 := by
  sorry

end geometric_sequence_property_l156_15613


namespace hockey_games_played_l156_15643

theorem hockey_games_played (layla_goals : ℕ) (kristin_goals_difference : ℕ) (average_goals : ℕ) 
  (h1 : layla_goals = 104)
  (h2 : kristin_goals_difference = 24)
  (h3 : average_goals = 92)
  (h4 : layla_goals - kristin_goals_difference = average_goals * 2) :
  2 = (layla_goals + (layla_goals - kristin_goals_difference)) / average_goals := by
  sorry

end hockey_games_played_l156_15643


namespace car_wash_price_l156_15686

theorem car_wash_price (oil_change_price : ℕ) (repair_price : ℕ) (oil_changes : ℕ) (repairs : ℕ) (car_washes : ℕ) (total_earnings : ℕ) :
  oil_change_price = 20 →
  repair_price = 30 →
  oil_changes = 5 →
  repairs = 10 →
  car_washes = 15 →
  total_earnings = 475 →
  (oil_change_price * oil_changes + repair_price * repairs + car_washes * 5 = total_earnings) :=
by
  sorry


end car_wash_price_l156_15686


namespace cans_left_to_load_l156_15603

/-- Given a packing scenario for canned juice, prove the number of cans left to be loaded. -/
theorem cans_left_to_load 
  (cans_per_carton : ℕ) 
  (total_cartons : ℕ) 
  (loaded_cartons : ℕ) 
  (h1 : cans_per_carton = 20)
  (h2 : total_cartons = 50)
  (h3 : loaded_cartons = 40) :
  (total_cartons - loaded_cartons) * cans_per_carton = 200 :=
by sorry

end cans_left_to_load_l156_15603


namespace simplify_expression_1_simplify_expression_2_l156_15605

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 6 * (2 * x - 1) - 3 * (5 + 2 * x) = 6 * x - 21 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) : (4 * a^2 - 8 * a - 9) + 3 * (2 * a^2 - 2 * a - 5) = 10 * a^2 - 14 * a - 24 := by
  sorry

end simplify_expression_1_simplify_expression_2_l156_15605


namespace little_john_money_l156_15684

/-- Calculates the remaining money after spending on sweets and giving to friends -/
def remaining_money (initial : ℚ) (spent_on_sweets : ℚ) (given_to_each_friend : ℚ) (num_friends : ℕ) : ℚ :=
  initial - spent_on_sweets - (given_to_each_friend * num_friends)

/-- Theorem stating that given the specific amounts, the remaining money is $2.05 -/
theorem little_john_money : 
  remaining_money 5.10 1.05 1.00 2 = 2.05 := by
  sorry

#eval remaining_money 5.10 1.05 1.00 2

end little_john_money_l156_15684


namespace specific_tetrahedron_properties_l156_15689

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (A₁ A₂ A₃ A₄ : Point3D) : ℝ :=
  sorry

/-- Calculates the height of a tetrahedron from a vertex to the opposite face -/
def tetrahedronHeight (A₁ A₂ A₃ A₄ : Point3D) : ℝ :=
  sorry

/-- Theorem stating the volume and height of a specific tetrahedron -/
theorem specific_tetrahedron_properties :
  let A₁ : Point3D := ⟨-2, 0, -4⟩
  let A₂ : Point3D := ⟨-1, 7, 1⟩
  let A₃ : Point3D := ⟨4, -8, -4⟩
  let A₄ : Point3D := ⟨1, -4, 6⟩
  (tetrahedronVolume A₁ A₂ A₃ A₄ = 250 / 3) ∧
  (tetrahedronHeight A₁ A₂ A₃ A₄ = 5 * Real.sqrt 2) :=
by
  sorry

end specific_tetrahedron_properties_l156_15689


namespace vectors_not_coplanar_l156_15696

/-- Given vectors a, b, and c in ℝ³, prove they are not coplanar. -/
theorem vectors_not_coplanar :
  let a : ℝ × ℝ × ℝ := (3, 10, 5)
  let b : ℝ × ℝ × ℝ := (-2, -2, -3)
  let c : ℝ × ℝ × ℝ := (2, 4, 3)
  ¬(∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by sorry

end vectors_not_coplanar_l156_15696


namespace sandbox_width_l156_15668

theorem sandbox_width (perimeter : ℝ) (width : ℝ) (length : ℝ) : 
  perimeter = 30 →
  length = 2 * width →
  perimeter = 2 * width + 2 * length →
  width = 5 := by
  sorry

end sandbox_width_l156_15668


namespace simplify_expression_1_simplify_expression_2_l156_15639

-- First expression
theorem simplify_expression_1 : 
  (Real.sqrt 12 + Real.sqrt 20) - (3 - Real.sqrt 5) = 2 * Real.sqrt 3 + 3 * Real.sqrt 5 - 3 := by
  sorry

-- Second expression
theorem simplify_expression_2 : 
  Real.sqrt 8 * Real.sqrt 6 - 3 * Real.sqrt 6 + Real.sqrt 2 = 4 * Real.sqrt 3 - 3 * Real.sqrt 6 + Real.sqrt 2 := by
  sorry

end simplify_expression_1_simplify_expression_2_l156_15639
