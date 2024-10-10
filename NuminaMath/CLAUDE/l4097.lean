import Mathlib

namespace jacksons_grade_calculation_l4097_409756

/-- Calculates Jackson's grade based on his time allocation and point system. -/
def jacksons_grade (video_game_hours : ℝ) (study_ratio : ℝ) (kindness_ratio : ℝ) 
  (study_points_per_hour : ℝ) (kindness_points_per_hour : ℝ) : ℝ :=
  let study_hours := video_game_hours * study_ratio
  let kindness_hours := video_game_hours * kindness_ratio
  study_hours * study_points_per_hour + kindness_hours * kindness_points_per_hour

theorem jacksons_grade_calculation :
  jacksons_grade 12 (1/3) (1/4) 20 40 = 200 := by
  sorry

end jacksons_grade_calculation_l4097_409756


namespace athlete_distance_l4097_409728

/-- Proves that an athlete running at 28.8 km/h for 25 seconds covers a distance of 200 meters. -/
theorem athlete_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 28.8 → time = 25 → distance = speed * time * 1000 / 3600 → distance = 200 := by
  sorry

end athlete_distance_l4097_409728


namespace tickets_problem_l4097_409712

/-- The total number of tickets Tate and Peyton have together -/
def total_tickets (tate_initial : ℕ) (tate_additional : ℕ) : ℕ :=
  let tate_total := tate_initial + tate_additional
  let peyton_tickets := tate_total / 2
  tate_total + peyton_tickets

/-- Theorem stating that given the initial conditions, Tate and Peyton have 51 tickets together -/
theorem tickets_problem (tate_initial : ℕ) (tate_additional : ℕ) 
    (h1 : tate_initial = 32) 
    (h2 : tate_additional = 2) : 
  total_tickets tate_initial tate_additional = 51 := by
  sorry

end tickets_problem_l4097_409712


namespace positive_solution_quadratic_equation_l4097_409783

theorem positive_solution_quadratic_equation :
  ∃ x : ℝ, x > 0 ∧ 
  (1/3) * (4 * x^2 - 2) = (x^2 - 75*x - 15) * (x^2 + 50*x + 10) ∧
  x = (75 + Real.sqrt 5693) / 2 := by
  sorry

end positive_solution_quadratic_equation_l4097_409783


namespace triangle_inconsistency_l4097_409744

theorem triangle_inconsistency : ¬ ∃ (a b c : ℝ),
  (a = 40 ∧ b = 50 ∧ c = 2 * (a + b) ∧ a + b + c = 160) ∧
  (a + b > c ∧ b + c > a ∧ a + c > b) := by
  sorry

end triangle_inconsistency_l4097_409744


namespace sector_central_angle_l4097_409764

/-- Given a circular sector with arc length 4 and area 2, 
    prove that its central angle is 4 radians. -/
theorem sector_central_angle (arc_length : ℝ) (area : ℝ) (θ : ℝ) :
  arc_length = 4 →
  area = 2 →
  θ = arc_length / (2 * area / arc_length) →
  θ = 4 := by
sorry

end sector_central_angle_l4097_409764


namespace four_x_plus_t_is_odd_l4097_409772

theorem four_x_plus_t_is_odd (x t : ℤ) (h : 2 * x - t = 11) : Odd (4 * x + t) := by
  sorry

end four_x_plus_t_is_odd_l4097_409772


namespace no_positive_integer_sequence_exists_positive_irrational_sequence_l4097_409761

/-- Part 1: Non-existence of positive integer sequence --/
theorem no_positive_integer_sequence :
  ¬ ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, (f (n + 1))^2 ≥ 2 * (f n) * (f (n + 2)) :=
sorry

/-- Part 2: Existence of positive irrational sequence --/
theorem exists_positive_irrational_sequence :
  ∃ f : ℕ+ → ℝ, (∀ n : ℕ+, Irrational (f n)) ∧
    (∀ n : ℕ+, f n > 0) ∧
    (∀ n : ℕ+, (f (n + 1))^2 ≥ 2 * (f n) * (f (n + 2))) :=
sorry

end no_positive_integer_sequence_exists_positive_irrational_sequence_l4097_409761


namespace inequality_equivalence_l4097_409759

theorem inequality_equivalence (x : ℝ) :
  abs (2 * x - 1) + abs (x + 1) ≥ x + 2 ↔ x ≤ 0 ∨ x ≥ 1 := by
  sorry

end inequality_equivalence_l4097_409759


namespace right_pyramid_surface_area_l4097_409792

/-- Represents a right pyramid with a parallelogram base -/
structure RightPyramid where
  base_side1 : ℝ
  base_side2 : ℝ
  base_angle : ℝ
  height : ℝ

/-- Calculates the total surface area of a right pyramid -/
def total_surface_area (p : RightPyramid) : ℝ :=
  sorry

theorem right_pyramid_surface_area :
  let p := RightPyramid.mk 12 14 (π / 3) 15
  total_surface_area p = 168 * Real.sqrt 3 + 216 * Real.sqrt 29 := by
  sorry

end right_pyramid_surface_area_l4097_409792


namespace marble_division_l4097_409729

theorem marble_division (x : ℝ) : 
  (5*x + 2) + (2*x - 1) + (x + 4) = 35 → 
  ∃ (a b c : ℕ), a + b + c = 35 ∧ 
    (a : ℝ) = 5*x + 2 ∧ 
    (b : ℝ) = 2*x - 1 ∧ 
    (c : ℝ) = x + 4 := by
  sorry

end marble_division_l4097_409729


namespace parabola_line_intersection_area_l4097_409711

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  equation : (ℝ × ℝ) → Prop

/-- Line structure -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Triangle structure -/
structure Triangle where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- Function to calculate distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Function to calculate area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Main theorem -/
theorem parabola_line_intersection_area 
  (p : Parabola) 
  (l : Line) 
  (A B : ℝ × ℝ) 
  (h1 : p.equation = fun (x, y) ↦ y^2 = 4*x)
  (h2 : p.focus = (1, 0))
  (h3 : l.point1 = p.focus)
  (h4 : p.equation A ∧ p.equation B)
  (h5 : distance A p.focus = 3) :
  triangleArea { point1 := (0, 0), point2 := A, point3 := B } = 3 * Real.sqrt 2 / 2 :=
sorry

end parabola_line_intersection_area_l4097_409711


namespace min_h25_for_tenuous_min_sum_l4097_409718

/-- A function h : ℕ → ℤ is tenuous if h(x) + h(y) > 2 * y^2 for all positive integers x and y. -/
def Tenuous (h : ℕ → ℤ) : Prop :=
  ∀ x y : ℕ, x > 0 → y > 0 → h x + h y > 2 * y^2

/-- The sum of h(1) to h(30) for a function h : ℕ → ℤ. -/
def SumH30 (h : ℕ → ℤ) : ℤ :=
  (Finset.range 30).sum (λ i => h (i + 1))

theorem min_h25_for_tenuous_min_sum (h : ℕ → ℤ) :
  Tenuous h → (∀ g : ℕ → ℤ, Tenuous g → SumH30 h ≤ SumH30 g) → h 25 ≥ 1189 := by
  sorry

end min_h25_for_tenuous_min_sum_l4097_409718


namespace stationery_difference_is_fifty_l4097_409733

/-- The number of stationery pieces Georgia has -/
def georgia_stationery : ℕ := 25

/-- The number of stationery pieces Lorene has -/
def lorene_stationery : ℕ := 3 * georgia_stationery

/-- The difference in stationery pieces between Lorene and Georgia -/
def stationery_difference : ℕ := lorene_stationery - georgia_stationery

theorem stationery_difference_is_fifty : stationery_difference = 50 := by
  sorry

end stationery_difference_is_fifty_l4097_409733


namespace jake_jill_difference_l4097_409704

/-- The number of peaches each person has -/
structure Peaches where
  jill : ℕ
  steven : ℕ
  jake : ℕ

/-- Given conditions about peach quantities -/
def peach_conditions (p : Peaches) : Prop :=
  p.jill = 87 ∧
  p.steven = p.jill + 18 ∧
  p.jake = p.steven - 5

/-- Theorem stating the difference between Jake's and Jill's peaches -/
theorem jake_jill_difference (p : Peaches) :
  peach_conditions p → p.jake - p.jill = 13 := by
  sorry

end jake_jill_difference_l4097_409704


namespace police_emergency_number_has_large_prime_factor_l4097_409705

/-- A number is a police emergency number if it ends with 133 in decimal system -/
def is_police_emergency_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 1000 * k + 133

/-- Every police emergency number has a prime factor greater than 7 -/
theorem police_emergency_number_has_large_prime_factor (n : ℕ) 
  (h : is_police_emergency_number n) : 
  ∃ p : ℕ, p > 7 ∧ Nat.Prime p ∧ p ∣ n := by
  sorry

end police_emergency_number_has_large_prime_factor_l4097_409705


namespace red_tickets_for_yellow_l4097_409758

/-- The number of yellow tickets needed to win a Bible -/
def yellow_tickets_needed : ℕ := 10

/-- The number of blue tickets needed to obtain one red ticket -/
def blue_per_red : ℕ := 10

/-- Tom's current yellow tickets -/
def tom_yellow : ℕ := 8

/-- Tom's current red tickets -/
def tom_red : ℕ := 3

/-- Tom's current blue tickets -/
def tom_blue : ℕ := 7

/-- Additional blue tickets Tom needs -/
def additional_blue : ℕ := 163

/-- The number of red tickets required to obtain one yellow ticket -/
def red_per_yellow : ℕ := 7

theorem red_tickets_for_yellow : 
  (yellow_tickets_needed - tom_yellow) * red_per_yellow = 
  (additional_blue + tom_blue) / blue_per_red - tom_red := by
  sorry

end red_tickets_for_yellow_l4097_409758


namespace rectangles_in_4x4_grid_l4097_409727

/-- The number of rectangles in a 4x4 grid -/
def num_rectangles_4x4 : ℕ := 36

/-- The number of horizontal or vertical lines in a 4x4 grid -/
def grid_lines : ℕ := 4

/-- Theorem: The number of rectangles in a 4x4 grid is 36 -/
theorem rectangles_in_4x4_grid :
  (grid_lines.choose 2) * (grid_lines.choose 2) = num_rectangles_4x4 := by
  sorry

end rectangles_in_4x4_grid_l4097_409727


namespace yuna_and_friends_count_l4097_409790

/-- Given a line of people where Yuna is 4th from the front and 6th from the back,
    the total number of people in the line is 9. -/
theorem yuna_and_friends_count (people : ℕ) (yuna_position_front yuna_position_back : ℕ) :
  yuna_position_front = 4 →
  yuna_position_back = 6 →
  people = 9 :=
by sorry

end yuna_and_friends_count_l4097_409790


namespace dogs_with_tags_and_collars_l4097_409775

theorem dogs_with_tags_and_collars (total : ℕ) (tags : ℕ) (collars : ℕ) (neither : ℕ) 
  (h_total : total = 80)
  (h_tags : tags = 45)
  (h_collars : collars = 40)
  (h_neither : neither = 1) :
  total = tags + collars - (tags + collars - total + neither) := by
  sorry

#check dogs_with_tags_and_collars

end dogs_with_tags_and_collars_l4097_409775


namespace cloth_sale_profit_per_meter_l4097_409724

/-- Calculates the profit per meter of cloth given the total meters sold,
    total selling price, and cost price per meter. -/
def profit_per_meter (total_meters : ℕ) (total_selling_price : ℕ) (cost_price_per_meter : ℕ) : ℚ :=
  (total_selling_price - total_meters * cost_price_per_meter : ℚ) / total_meters

/-- Proves that for a specific cloth sale, the profit per meter is 7 -/
theorem cloth_sale_profit_per_meter :
  profit_per_meter 80 10000 118 = 7 := by
  sorry

end cloth_sale_profit_per_meter_l4097_409724


namespace number_problem_l4097_409701

theorem number_problem (x : ℝ) : 0.4 * x + 60 = x → x = 100 := by
  sorry

end number_problem_l4097_409701


namespace chess_tournament_schedules_l4097_409776

/-- Represents a chess tournament between two schools --/
structure ChessTournament where
  /-- Number of players in each school --/
  players_per_school : Nat
  /-- Number of games each player plays against each opponent from the other school --/
  games_per_opponent : Nat
  /-- Number of games played simultaneously in each round --/
  games_per_round : Nat

/-- Calculates the total number of games in the tournament --/
def totalGames (t : ChessTournament) : Nat :=
  t.players_per_school * t.players_per_school * t.games_per_opponent

/-- Calculates the number of rounds in the tournament --/
def numberOfRounds (t : ChessTournament) : Nat :=
  totalGames t / t.games_per_round

/-- Theorem stating the number of ways to schedule the tournament --/
theorem chess_tournament_schedules (t : ChessTournament) 
  (h1 : t.players_per_school = 4)
  (h2 : t.games_per_opponent = 2)
  (h3 : t.games_per_round = 4) :
  Nat.factorial (numberOfRounds t) = 40320 := by
  sorry


end chess_tournament_schedules_l4097_409776


namespace divisor_problem_l4097_409720

theorem divisor_problem (original : ℕ) (added : ℕ) (divisor : ℕ) : 
  original = 821562 →
  added = 6 →
  (original + added) % divisor = 0 →
  ∀ d : ℕ, d < added → (original + d) % divisor ≠ 0 →
  divisor = 6 :=
by sorry

end divisor_problem_l4097_409720


namespace cube_sum_zero_l4097_409749

theorem cube_sum_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = a^4 + b^4 + c^4) :
  a^3 + b^3 + c^3 = 0 := by
  sorry

end cube_sum_zero_l4097_409749


namespace average_hours_worked_l4097_409787

def april_hours : ℕ := 6
def june_hours : ℕ := 5
def september_hours : ℕ := 8
def days_per_month : ℕ := 30
def num_months : ℕ := 3

def total_hours : ℕ := april_hours * days_per_month + june_hours * days_per_month + september_hours * days_per_month

theorem average_hours_worked (h : total_hours = april_hours * days_per_month + june_hours * days_per_month + september_hours * days_per_month) : 
  total_hours / num_months = 190 := by
  sorry

end average_hours_worked_l4097_409787


namespace students_taking_neither_subject_l4097_409700

-- Define the total number of students in the drama club
def total_students : ℕ := 60

-- Define the number of students taking mathematics
def math_students : ℕ := 40

-- Define the number of students taking physics
def physics_students : ℕ := 35

-- Define the number of students taking both mathematics and physics
def both_subjects : ℕ := 25

-- Theorem to prove
theorem students_taking_neither_subject : 
  total_students - (math_students + physics_students - both_subjects) = 10 := by
  sorry

end students_taking_neither_subject_l4097_409700


namespace certain_number_problem_l4097_409780

theorem certain_number_problem (C : ℝ) : C - |(-10 + 6)| = 26 → C = 30 := by
  sorry

end certain_number_problem_l4097_409780


namespace angle_sum_in_special_pentagon_l4097_409769

theorem angle_sum_in_special_pentagon (x y : ℝ) 
  (h1 : 0 ≤ x ∧ x < 180) 
  (h2 : 0 ≤ y ∧ y < 180) : 
  x + y = 50 := by
  sorry

end angle_sum_in_special_pentagon_l4097_409769


namespace circle_and_line_equations_l4097_409747

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 1 ∨ y = -x + 1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y + 2 * Real.sqrt 2 = 0

theorem circle_and_line_equations :
  ∃ (a : ℝ),
    a ≤ 0 ∧
    circle_M 0 (-2) ∧
    (∃ (x y : ℝ), circle_M x y ∧ tangent_line x y) ∧
    line_l 0 1 ∧
    (∃ (A B : ℝ × ℝ),
      circle_M A.1 A.2 ∧
      circle_M B.1 B.2 ∧
      line_l A.1 A.2 ∧
      line_l B.1 B.2 ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = 14) :=
by
  sorry


end circle_and_line_equations_l4097_409747


namespace pyramid_surface_area_l4097_409770

/-- The total surface area of a pyramid formed from a cube --/
theorem pyramid_surface_area (a : ℝ) (h : a > 0) : 
  let cube_edge := a
  let base_side := a * Real.sqrt 2 / 2
  let slant_height := 3 * a * Real.sqrt 2 / 4
  let lateral_area := 4 * (base_side * slant_height / 2)
  let base_area := base_side ^ 2
  lateral_area + base_area = 2 * a ^ 2 := by
  sorry

#check pyramid_surface_area

end pyramid_surface_area_l4097_409770


namespace function_always_positive_l4097_409726

theorem function_always_positive (x : ℝ) : 
  (∀ a ∈ Set.Icc (-1 : ℝ) 1, x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ 
  (x < 1 ∨ x > 3) := by
sorry

end function_always_positive_l4097_409726


namespace tangent_circle_radius_l4097_409786

/-- A circle is tangent to the coordinate axes and the hypotenuse of a 45-45-90 triangle -/
structure TangentCircle where
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The radius of the circle -/
  radius : ℝ
  /-- The circle is tangent to the x-axis -/
  tangent_x : center.2 = radius
  /-- The circle is tangent to the y-axis -/
  tangent_y : center.1 = radius
  /-- The circle is tangent to the hypotenuse of the 45-45-90 triangle -/
  tangent_hypotenuse : center.1 + center.2 + radius = 2 * Real.sqrt 2

/-- The side length of the 45-45-90 triangle -/
def triangleSide : ℝ := 2

/-- The theorem stating that the radius of the tangent circle is √2 -/
theorem tangent_circle_radius :
  ∀ (c : TangentCircle), c.radius = Real.sqrt 2 := by
  sorry

end tangent_circle_radius_l4097_409786


namespace robot_wear_combinations_l4097_409778

/-- Represents the number of ways to wear items on one arm -/
def waysPerArm : ℕ := 1

/-- Represents the number of arms -/
def numArms : ℕ := 2

/-- Represents the number of ways to order items between arms -/
def waysBetweenArms : ℕ := 1

/-- Calculates the total number of ways to wear all items -/
def totalWays : ℕ := waysPerArm ^ numArms * waysBetweenArms

theorem robot_wear_combinations : totalWays = 4 := by
  sorry

end robot_wear_combinations_l4097_409778


namespace imaginary_part_of_z_l4097_409739

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I)^2 / z = 1 + Complex.I) : 
  Complex.im z = -1 := by
sorry

end imaginary_part_of_z_l4097_409739


namespace unique_solution_quadratic_l4097_409723

theorem unique_solution_quadratic :
  ∃! (q : ℝ), q ≠ 0 ∧ (∃! x, q * x^2 - 8 * x + 16 = 0) :=
by
  -- The proof goes here
  sorry

end unique_solution_quadratic_l4097_409723


namespace octagon_coloring_count_l4097_409719

/-- Represents a coloring of 8 disks arranged in an octagon. -/
structure OctagonColoring where
  blue : Finset (Fin 8)
  red : Finset (Fin 8)
  yellow : Finset (Fin 8)
  partition : Disjoint blue red ∧ Disjoint blue yellow ∧ Disjoint red yellow
  cover : blue ∪ red ∪ yellow = Finset.univ
  blue_count : blue.card = 4
  red_count : red.card = 3
  yellow_count : yellow.card = 1

/-- The group of symmetries of an octagon. -/
def OctagonSymmetry : Type := Unit -- Placeholder, actual implementation would be more complex

/-- Two colorings are equivalent if one can be obtained from the other by a symmetry. -/
def equivalent (c₁ c₂ : OctagonColoring) (sym : OctagonSymmetry) : Prop := sorry

/-- The number of distinct colorings under symmetry. -/
def distinctColorings : ℕ := sorry

/-- The main theorem: There are exactly 26 distinct colorings. -/
theorem octagon_coloring_count : distinctColorings = 26 := by sorry

end octagon_coloring_count_l4097_409719


namespace green_ball_count_l4097_409791

/-- Represents the number of balls of each color --/
structure BallCount where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The conditions of the problem --/
def validBallCount (bc : BallCount) : Prop :=
  bc.red + bc.blue + bc.green = 50 ∧
  ∀ (subset : ℕ), subset ≤ 50 → subset ≥ 34 → bc.red > 0 ∧
  ∀ (subset : ℕ), subset ≤ 50 → subset ≥ 35 → bc.blue > 0 ∧
  ∀ (subset : ℕ), subset ≤ 50 → subset ≥ 36 → bc.green > 0

/-- The theorem to be proved --/
theorem green_ball_count (bc : BallCount) (h : validBallCount bc) :
  15 ≤ bc.green ∧ bc.green ≤ 17 := by
  sorry

end green_ball_count_l4097_409791


namespace yard_trees_l4097_409743

/-- The number of trees in a yard with given length and tree spacing -/
def num_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  yard_length / tree_spacing + 1

/-- Theorem: In a 400-meter yard with trees spaced 16 meters apart, there are 26 trees -/
theorem yard_trees : num_trees 400 16 = 26 := by
  sorry

end yard_trees_l4097_409743


namespace cube_root_equation_solutions_l4097_409771

theorem cube_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (10 * x - 2) ^ (1/3) + (20 * x + 3) ^ (1/3) - 5 * x ^ (1/3)
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = -1/25 ∨ x = 1/375 := by
  sorry

end cube_root_equation_solutions_l4097_409771


namespace smallest_positive_integer_congruence_l4097_409773

theorem smallest_positive_integer_congruence :
  ∃! (x : ℕ), x > 0 ∧ (45 * x + 13) % 30 = 5 ∧ ∀ (y : ℕ), y > 0 ∧ (45 * y + 13) % 30 = 5 → x ≤ y :=
sorry

end smallest_positive_integer_congruence_l4097_409773


namespace range_of_f_inverse_l4097_409702

/-- The function f(x) = 2 - log₂(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 - Real.log x / Real.log 2

/-- The inverse function of f -/
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

theorem range_of_f_inverse :
  Set.range f = Set.Ioi 1 →
  Set.range f_inv = Set.Ioo 0 2 :=
by sorry

end range_of_f_inverse_l4097_409702


namespace divisibility_by_twelve_l4097_409794

theorem divisibility_by_twelve (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) :=
by sorry

end divisibility_by_twelve_l4097_409794


namespace arithmetic_sequence_12th_term_l4097_409715

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 1) :
  a 12 = 15 := by
sorry

end arithmetic_sequence_12th_term_l4097_409715


namespace min_value_fraction_l4097_409717

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 2) :
  (x + y + z) / (x * y * z) ≥ 27 / 4 :=
sorry

end min_value_fraction_l4097_409717


namespace field_width_l4097_409709

/-- Proves that a rectangular field of length 60 m with a 2.5 m wide path around it,
    having a path area of 1200 sq m, has a width of 175 m. -/
theorem field_width (field_length : ℝ) (path_width : ℝ) (path_area : ℝ) :
  field_length = 60 →
  path_width = 2.5 →
  path_area = 1200 →
  ∃ field_width : ℝ,
    (field_length + 2 * path_width) * (field_width + 2 * path_width) -
    field_length * field_width = path_area ∧
    field_width = 175 := by
  sorry


end field_width_l4097_409709


namespace decimal_division_proof_l4097_409796

theorem decimal_division_proof : (0.045 : ℝ) / (0.005 : ℝ) = 9 := by sorry

end decimal_division_proof_l4097_409796


namespace average_speed_two_hours_l4097_409768

/-- The average speed of a car given its distances traveled in two hours -/
theorem average_speed_two_hours (d1 d2 : ℝ) : d1 = 80 → d2 = 60 → (d1 + d2) / 2 = 70 := by
  sorry

end average_speed_two_hours_l4097_409768


namespace systematic_sampling_problem_l4097_409752

/-- Systematic sampling function -/
def systematicSample (start : ℕ) (interval : ℕ) (n : ℕ) : ℕ :=
  start + interval * (n - 1)

/-- Theorem for systematic sampling in the given problem -/
theorem systematic_sampling_problem :
  let totalStudents : ℕ := 500
  let selectedStudents : ℕ := 50
  let interval : ℕ := totalStudents / selectedStudents
  let start : ℕ := 6
  ∀ n : ℕ, 
    125 ≤ systematicSample start interval n ∧ 
    systematicSample start interval n ≤ 140 → 
    systematicSample start interval n = 126 ∨ 
    systematicSample start interval n = 136 :=
by
  sorry

#check systematic_sampling_problem

end systematic_sampling_problem_l4097_409752


namespace misread_signs_count_l4097_409785

def f (x : ℝ) : ℝ := 10*x^9 + 9*x^8 + 8*x^7 + 7*x^6 + 6*x^5 + 5*x^4 + 4*x^3 + 3*x^2 + 2*x + 1

theorem misread_signs_count :
  let correct_result := f (-1)
  let incorrect_result := 7
  let difference := incorrect_result - correct_result
  difference / 2 = 6 := by sorry

end misread_signs_count_l4097_409785


namespace line_passes_through_fixed_point_l4097_409781

/-- Trajectory of the center of the moving circle -/
def trajectory (x y : ℝ) : Prop := y^2 = 8*x

/-- Line l passing through a point (x, y) with slope t -/
def line_l (t m x y : ℝ) : Prop := x = t*y + m

/-- Angle bisector condition for ∠PBQ -/
def angle_bisector (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ / (x₁ + 3) + y₂ / (x₂ + 3) = 0

/-- Main theorem: Line l passes through (3, 0) given the conditions -/
theorem line_passes_through_fixed_point
  (t m x₁ y₁ x₂ y₂ : ℝ)
  (h_traj₁ : trajectory x₁ y₁)
  (h_traj₂ : trajectory x₂ y₂)
  (h_line₁ : line_l t m x₁ y₁)
  (h_line₂ : line_l t m x₂ y₂)
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂))
  (h_not_vertical : t ≠ 0)
  (h_bisector : angle_bisector x₁ y₁ x₂ y₂) :
  m = 3 :=
sorry

end line_passes_through_fixed_point_l4097_409781


namespace double_burger_cost_l4097_409730

/-- The cost of a double burger given the following conditions:
  - Total spent: $68.50
  - Total number of hamburgers: 50
  - Single burger cost: $1.00 each
  - Number of double burgers: 37
-/
theorem double_burger_cost :
  let total_spent : ℚ := 68.5
  let total_burgers : ℕ := 50
  let single_burger_cost : ℚ := 1
  let double_burgers : ℕ := 37
  let single_burgers : ℕ := total_burgers - double_burgers
  let double_burger_cost : ℚ := (total_spent - (single_burgers : ℚ) * single_burger_cost) / (double_burgers : ℚ)
  double_burger_cost = 1.5 := by
  sorry

end double_burger_cost_l4097_409730


namespace nested_expression_value_l4097_409799

theorem nested_expression_value : (3*(3*(3*(3*(3*(3*(3+2)+2)+2)+2)+2)+2)+2) = 4373 := by
  sorry

end nested_expression_value_l4097_409799


namespace not_A_necessary_not_sufficient_for_not_B_l4097_409748

-- Define propositions A and B
variable (A B : Prop)

-- Define what it means for A to be sufficient but not necessary for B
def sufficient_not_necessary (A B : Prop) : Prop :=
  (A → B) ∧ ¬(B → A)

-- Theorem statement
theorem not_A_necessary_not_sufficient_for_not_B
  (h : sufficient_not_necessary A B) :
  (¬B → ¬A) ∧ ¬(¬A → ¬B) := by
  sorry

end not_A_necessary_not_sufficient_for_not_B_l4097_409748


namespace combination_ratio_l4097_409760

def combination (n k : ℕ) : ℚ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem combination_ratio :
  (combination 5 2) / (combination 7 3) = 2 / 7 := by
  sorry

end combination_ratio_l4097_409760


namespace question_one_question_two_l4097_409788

-- Define the sets A, B, and M
def A (a : ℝ) : Set ℝ := {x | x^2 + (a - 1) * x - a > 0}
def B (a b : ℝ) : Set ℝ := {x | (x + a) * (x + b) > 0}
def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

-- Define the complement of B in ℝ
def C_I_B (a b : ℝ) : Set ℝ := {x | ¬((x + a) * (x + b) > 0)}

-- Theorem for question 1
theorem question_one (a b : ℝ) (h1 : a < b) (h2 : C_I_B a b = M) : 
  a = -1 ∧ b = 3 := by sorry

-- Theorem for question 2
theorem question_two (a b : ℝ) (h : a > b ∧ b > -1) : 
  A a ∩ B a b = {x | x < -a ∨ x > 1} := by sorry

end question_one_question_two_l4097_409788


namespace hyperbola_standard_equation_l4097_409714

/-- A hyperbola with asymptotes forming an acute angle of 60° and passing through (√2, √3) -/
structure Hyperbola where
  /-- The acute angle formed by the asymptotes -/
  angle : ℝ
  /-- The point through which the hyperbola passes -/
  point : ℝ × ℝ
  /-- The angle is 60° -/
  angle_is_60 : angle = 60 * π / 180
  /-- The point is (√2, √3) -/
  point_is_sqrt : point = (Real.sqrt 2, Real.sqrt 3)

/-- The standard equation of the hyperbola -/
def standard_equation (h : Hyperbola) : (ℝ → ℝ → Prop) → Prop :=
  λ eq ↦ (eq = λ x y ↦ x^2/1 - y^2/3 = 1) ∨ (eq = λ x y ↦ x^2/7 - y^2/(7/3) = 1)

/-- Theorem stating that the given hyperbola has one of the two standard equations -/
theorem hyperbola_standard_equation (h : Hyperbola) :
  ∃ eq : ℝ → ℝ → Prop, standard_equation h eq :=
sorry

end hyperbola_standard_equation_l4097_409714


namespace mask_production_rates_l4097_409777

/-- Represents the daily production rate of masks in millions before equipment change -/
def initial_rate : ℝ := 40

/-- Represents the daily production rate of masks in millions after equipment change -/
def final_rate : ℝ := 56

/-- Represents the number of masks left to produce in millions -/
def remaining_masks : ℝ := 280

/-- Represents the increase in production efficiency as a decimal -/
def efficiency_increase : ℝ := 0.4

/-- Represents the number of days saved due to equipment change -/
def days_saved : ℝ := 2

theorem mask_production_rates :
  (remaining_masks / initial_rate - remaining_masks / (initial_rate * (1 + efficiency_increase)) = days_saved) ∧
  (final_rate = initial_rate * (1 + efficiency_increase)) := by
  sorry

end mask_production_rates_l4097_409777


namespace ternary_121_equals_16_l4097_409731

/-- Converts a ternary (base-3) number to decimal --/
def ternary_to_decimal (t₂ t₁ t₀ : ℕ) : ℕ :=
  t₂ * 3^2 + t₁ * 3^1 + t₀ * 3^0

/-- The ternary number 121₃ is equal to the decimal number 16 --/
theorem ternary_121_equals_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end ternary_121_equals_16_l4097_409731


namespace square_side_length_l4097_409740

theorem square_side_length (x : ℝ) (triangle_side : ℝ) (square_side : ℝ) : 
  triangle_side = 2 * x →
  4 * square_side = 3 * triangle_side →
  x = 4 →
  square_side = 6 := by
sorry

end square_side_length_l4097_409740


namespace complex_equation_solution_l4097_409736

theorem complex_equation_solution (a : ℝ) : 
  (a * Complex.I) / (2 - Complex.I) = 1 - 2 * Complex.I → a = -5 := by
sorry

end complex_equation_solution_l4097_409736


namespace inequality_theorem_equality_condition_l4097_409741

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (h₁ : x₁ * y₁ - z₁^2 > 0) (h₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) :=
by sorry

theorem equality_condition (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (h₁ : x₁ * y₁ - z₁^2 > 0) (h₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔ 
  x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂ :=
by sorry

end inequality_theorem_equality_condition_l4097_409741


namespace f_zero_and_no_extreme_value_l4097_409735

noncomputable section

/-- The function f(x) = (x+2)lnx + ax^2 - 4x + 7a -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + 2) * Real.log x + a * x^2 - 4 * x + 7 * a

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.log x + (x + 2) / x + 2 * a * x - 4

theorem f_zero_and_no_extreme_value :
  (∀ x > 0, f (1/2) x = 0 ↔ x = 1) ∧
  (∀ a ≥ 1/2, ∀ x > 0, f_derivative a x ≥ 0) :=
sorry

end f_zero_and_no_extreme_value_l4097_409735


namespace imaginary_part_of_z_l4097_409753

/-- Given a complex number z satisfying (1 + z) / i = 1 - z, 
    the imaginary part of z is 1 -/
theorem imaginary_part_of_z (z : ℂ) (h : (1 + z) / Complex.I = 1 - z) : 
  Complex.im z = 1 := by
  sorry

end imaginary_part_of_z_l4097_409753


namespace difference_of_squares_factorization_l4097_409710

theorem difference_of_squares_factorization (x y : ℝ) : 
  x^2 - 4*y^2 = (x + 2*y) * (x - 2*y) := by sorry

end difference_of_squares_factorization_l4097_409710


namespace profit_share_difference_example_l4097_409779

/-- Represents the profit share calculation for a business partnership. -/
structure ProfitShare where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  b_profit : ℕ

/-- Calculates the difference between profit shares of A and C. -/
def profit_share_difference (ps : ProfitShare) : ℕ :=
  let total_ratio := ps.a_investment + ps.b_investment + ps.c_investment
  let unit_profit := ps.b_profit * total_ratio / ps.b_investment
  let a_profit := unit_profit * ps.a_investment / total_ratio
  let c_profit := unit_profit * ps.c_investment / total_ratio
  c_profit - a_profit

/-- Theorem stating the difference in profit shares for the given scenario. -/
theorem profit_share_difference_example :
  profit_share_difference ⟨8000, 10000, 12000, 2000⟩ = 800 := by
  sorry


end profit_share_difference_example_l4097_409779


namespace quadratic_inequalities_l4097_409765

def quadratic_inequality_A (x : ℝ) := x^2 - 12*x + 20 > 0

def quadratic_inequality_B (x : ℝ) := x^2 - 5*x + 6 < 0

def quadratic_inequality_C (x : ℝ) := 9*x^2 - 6*x + 1 > 0

def quadratic_inequality_D (x : ℝ) := -2*x^2 + 2*x - 3 > 0

theorem quadratic_inequalities :
  (∀ x, quadratic_inequality_A x ↔ (x < 2 ∨ x > 10)) ∧
  (∀ x, quadratic_inequality_B x ↔ (2 < x ∧ x < 3)) ∧
  (∃ x, ¬quadratic_inequality_C x) ∧
  (∀ x, ¬quadratic_inequality_D x) :=
sorry

end quadratic_inequalities_l4097_409765


namespace range_of_g_l4097_409766

def f (x : ℝ) : ℝ := 4 * x + 1

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → 85 ≤ g x ∧ g x ≤ 853 :=
by sorry

end range_of_g_l4097_409766


namespace no_representation_as_sum_of_squares_and_ninth_power_l4097_409798

theorem no_representation_as_sum_of_squares_and_ninth_power (p : ℕ) (m : ℤ) 
  (h_prime : Nat.Prime p) (h_form : p = 4 * m + 1) :
  ¬ ∃ (x y z : ℤ), 216 * (p : ℤ)^3 = x^2 + y^2 + z^9 := by
  sorry

end no_representation_as_sum_of_squares_and_ninth_power_l4097_409798


namespace orthocenter_of_triangle_l4097_409734

/-- The orthocenter of a triangle is the point where all three altitudes intersect. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, 4, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter A B C = (1/2, 8, 1/2) := by sorry

end orthocenter_of_triangle_l4097_409734


namespace cube_volume_problem_l4097_409706

theorem cube_volume_problem (reference_cube_volume : ℝ) 
  (unknown_cube_surface_area : ℝ) (reference_cube_surface_area : ℝ) :
  reference_cube_volume = 8 →
  unknown_cube_surface_area = 3 * reference_cube_surface_area →
  reference_cube_surface_area = 6 * (reference_cube_volume ^ (1/3)) ^ 2 →
  let unknown_cube_side_length := (unknown_cube_surface_area / 6) ^ (1/2)
  unknown_cube_side_length ^ 3 = 24 * Real.sqrt 3 := by
sorry

end cube_volume_problem_l4097_409706


namespace minimal_polynomial_with_roots_l4097_409751

/-- The polynomial we're proving is correct -/
def f (x : ℝ) : ℝ := x^4 - 8*x^3 + 14*x^2 + 8*x - 3

/-- A root of a polynomial -/
def is_root (p : ℝ → ℝ) (r : ℝ) : Prop := p r = 0

/-- A polynomial with rational coefficients -/
def has_rational_coeffs (p : ℝ → ℝ) : Prop := 
  ∃ (a b c d e : ℚ), ∀ x, p x = a*x^4 + b*x^3 + c*x^2 + d*x + e

theorem minimal_polynomial_with_roots : 
  (is_root f (2 + Real.sqrt 3)) ∧ 
  (is_root f (2 + Real.sqrt 5)) ∧ 
  (has_rational_coeffs f) ∧
  (∀ g : ℝ → ℝ, has_rational_coeffs g → is_root g (2 + Real.sqrt 3) → 
    is_root g (2 + Real.sqrt 5) → (∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = a * g x) → 
    (∃ n : ℕ, ∀ x, g x = (f x) * x^n)) := 
sorry

end minimal_polynomial_with_roots_l4097_409751


namespace min_value_quadratic_sum_l4097_409742

theorem min_value_quadratic_sum :
  ∀ x y : ℝ, (2*x - y + 3)^2 + (x + 2*y - 1)^2 ≥ 295/72 := by
  sorry

end min_value_quadratic_sum_l4097_409742


namespace marble_capacity_l4097_409795

/-- Given a container of volume 24 cm³ holding 75 marbles, 
    prove that a container of volume 72 cm³ will hold 225 marbles, 
    assuming a linear relationship between volume and marble capacity. -/
theorem marble_capacity 
  (volume_small : ℝ) 
  (marbles_small : ℕ) 
  (volume_large : ℝ) 
  (h1 : volume_small = 24) 
  (h2 : marbles_small = 75) 
  (h3 : volume_large = 72) : 
  (volume_large / volume_small) * marbles_small = 225 := by
sorry

end marble_capacity_l4097_409795


namespace skew_lines_iff_b_neq_two_sevenths_l4097_409721

def line1 (b t : ℝ) : ℝ × ℝ × ℝ := (2 + 3*t, 1 + 4*t, b + 2*t)
def line2 (u : ℝ) : ℝ × ℝ × ℝ := (5 + u, 3 - u, 2 + 2*u)

def are_skew (b : ℝ) : Prop :=
  ∀ t u : ℝ, line1 b t ≠ line2 u

theorem skew_lines_iff_b_neq_two_sevenths (b : ℝ) :
  are_skew b ↔ b ≠ 2/7 :=
sorry

end skew_lines_iff_b_neq_two_sevenths_l4097_409721


namespace high_heels_cost_high_heels_cost_proof_l4097_409782

/-- The cost of one pair of high heels given the following conditions:
  - Fern buys one pair of high heels and five pairs of ballet slippers
  - The price of five pairs of ballet slippers is 2/3 of the price of the high heels
  - The total cost is $260
-/
theorem high_heels_cost : ℝ → Prop :=
  fun high_heels_price =>
    let ballet_slippers_price := (2 / 3) * high_heels_price
    let total_cost := high_heels_price + 5 * ballet_slippers_price
    total_cost = 260 → high_heels_price = 60

/-- Proof of the high heels cost theorem -/
theorem high_heels_cost_proof : ∃ (price : ℝ), high_heels_cost price :=
  sorry

end high_heels_cost_high_heels_cost_proof_l4097_409782


namespace trapezoid_area_l4097_409755

/-- Given four identical trapezoids that form a square, prove the area of each trapezoid --/
theorem trapezoid_area (base_small : ℝ) (base_large : ℝ) (square_area : ℝ) :
  base_small = 30 →
  base_large = 50 →
  square_area = 2500 →
  (∃ (trapezoid_area : ℝ), 
    trapezoid_area = (square_area - base_small ^ 2) / 4 ∧ 
    trapezoid_area = 400) := by
  sorry

end trapezoid_area_l4097_409755


namespace parabola_directrix_l4097_409737

/-- A parabola is defined by its equation in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The directrix of a parabola is a line parallel to the x-axis -/
structure Directrix where
  y : ℝ

/-- Given a parabola y = (x^2 - 8x + 16) / 8, its directrix is y = -1/2 -/
theorem parabola_directrix (p : Parabola) (d : Directrix) :
  p.a = 1/8 ∧ p.b = -1 ∧ p.c = 2 → d.y = -1/2 := by
  sorry

#check parabola_directrix

end parabola_directrix_l4097_409737


namespace diving_class_capacity_is_270_l4097_409746

/-- The number of people who can take diving classes in 3 weeks -/
def diving_class_capacity : ℕ :=
  let weekday_classes_per_day : ℕ := 2
  let weekend_classes_per_day : ℕ := 4
  let weekdays_per_week : ℕ := 5
  let weekend_days_per_week : ℕ := 2
  let people_per_class : ℕ := 5
  let weeks : ℕ := 3

  let weekday_classes_per_week : ℕ := weekday_classes_per_day * weekdays_per_week
  let weekend_classes_per_week : ℕ := weekend_classes_per_day * weekend_days_per_week
  let total_classes_per_week : ℕ := weekday_classes_per_week + weekend_classes_per_week
  let people_per_week : ℕ := total_classes_per_week * people_per_class
  
  people_per_week * weeks

/-- Theorem stating that the diving class capacity for 3 weeks is 270 people -/
theorem diving_class_capacity_is_270 : diving_class_capacity = 270 := by
  sorry

end diving_class_capacity_is_270_l4097_409746


namespace number_division_l4097_409725

theorem number_division (x : ℤ) : (x - 39 = 54) → (x / 3 = 31) := by
  sorry

end number_division_l4097_409725


namespace stratified_sampling_second_grade_l4097_409793

/-- Represents the number of students in each grade -/
structure GradeDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of students -/
def total_students (g : GradeDistribution) : ℕ :=
  g.first + g.second + g.third

/-- Represents the sample size for each grade -/
structure SampleDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total sample size -/
def total_sample (s : SampleDistribution) : ℕ :=
  s.first + s.second + s.third

/-- Checks if the sample distribution is proportional to the grade distribution -/
def is_proportional_sample (g : GradeDistribution) (s : SampleDistribution) : Prop :=
  g.first * s.second = g.second * s.first ∧
  g.second * s.third = g.third * s.second

theorem stratified_sampling_second_grade
  (g : GradeDistribution)
  (s : SampleDistribution)
  (h1 : total_students g = 2000)
  (h2 : g.first = 5 * g.third)
  (h3 : g.second = 3 * g.third)
  (h4 : total_sample s = 20)
  (h5 : is_proportional_sample g s) :
  s.second = 6 := by
  sorry

#check stratified_sampling_second_grade

end stratified_sampling_second_grade_l4097_409793


namespace sequence_limit_zero_l4097_409707

/-- Given an infinite sequence {a_n} where the limit of (a_{n+1} - a_n/2) as n approaches infinity is 0,
    prove that the limit of a_n as n approaches infinity is 0. -/
theorem sequence_limit_zero
  (a : ℕ → ℝ)
  (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |a (n + 1) - a n / 2| < ε) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n| < ε :=
sorry

end sequence_limit_zero_l4097_409707


namespace cheerful_not_green_l4097_409713

structure Snake where
  isGreen : Bool
  isCheerful : Bool
  canMultiply : Bool
  canDivide : Bool

def TomCollection : Nat := 15

theorem cheerful_not_green (snakes : Finset Snake) 
  (h1 : snakes.card = TomCollection)
  (h2 : (snakes.filter (fun s => s.isGreen)).card = 5)
  (h3 : (snakes.filter (fun s => s.isCheerful)).card = 6)
  (h4 : ∀ s ∈ snakes, s.isCheerful → s.canMultiply)
  (h5 : ∀ s ∈ snakes, s.isGreen → ¬s.canDivide)
  (h6 : ∀ s ∈ snakes, ¬s.canDivide → ¬s.canMultiply) :
  ∀ s ∈ snakes, s.isCheerful → ¬s.isGreen :=
sorry

end cheerful_not_green_l4097_409713


namespace last_three_digits_factorial_sum_15_l4097_409732

def last_three_digits (n : ℕ) : ℕ := n % 1000

def factorial_sum (n : ℕ) : ℕ :=
  (List.range n).map Nat.factorial |> List.sum

theorem last_three_digits_factorial_sum_15 :
  last_three_digits (factorial_sum 15) = 193 := by
  sorry

end last_three_digits_factorial_sum_15_l4097_409732


namespace distribute_books_equal_distribute_books_scenario1_distribute_books_scenario2_l4097_409774

/-- The number of ways to distribute 7 different books among 3 people -/
def distribute_books (scenario : Nat) : Nat :=
  match scenario with
  | 1 => 630  -- One person gets 1 book, one gets 2 books, and one gets 4 books
  | 2 => 630  -- One person gets 3 books, and two people each get 2 books
  | _ => 0    -- Invalid scenario

/-- Proof that both distribution scenarios result in 630 ways -/
theorem distribute_books_equal : distribute_books 1 = distribute_books 2 := by
  sorry

/-- Proof that the number of ways to distribute books in scenario 1 is 630 -/
theorem distribute_books_scenario1 : distribute_books 1 = 630 := by
  sorry

/-- Proof that the number of ways to distribute books in scenario 2 is 630 -/
theorem distribute_books_scenario2 : distribute_books 2 = 630 := by
  sorry

end distribute_books_equal_distribute_books_scenario1_distribute_books_scenario2_l4097_409774


namespace cubic_equation_with_double_root_l4097_409738

/-- Given a cubic equation 2x^3 + 9x^2 - 117x + k = 0 where two roots are equal and k is positive,
    prove that k = 47050/216 -/
theorem cubic_equation_with_double_root (k : ℝ) : 
  (∃ x y : ℝ, (2 * x^3 + 9 * x^2 - 117 * x + k = 0) ∧ 
               (2 * y^3 + 9 * y^2 - 117 * y + k = 0) ∧
               (x ≠ y)) ∧
  (∃ z : ℝ, (2 * z^3 + 9 * z^2 - 117 * z + k = 0) ∧
            (∃ w : ℝ, w ≠ z ∧ 2 * w^3 + 9 * w^2 - 117 * w + k = 0)) ∧
  (k > 0) →
  k = 47050 / 216 := by
  sorry

end cubic_equation_with_double_root_l4097_409738


namespace frame_interior_edges_sum_l4097_409708

/-- Represents a rectangular picture frame -/
structure Frame where
  outerLength : ℝ
  outerWidth : ℝ
  frameWidth : ℝ

/-- Calculates the area of the frame -/
def frameArea (f : Frame) : ℝ :=
  f.outerLength * f.outerWidth - (f.outerLength - 2 * f.frameWidth) * (f.outerWidth - 2 * f.frameWidth)

/-- Calculates the sum of the lengths of all four interior edges of the frame -/
def interiorEdgesSum (f : Frame) : ℝ :=
  2 * (f.outerLength - 2 * f.frameWidth) + 2 * (f.outerWidth - 2 * f.frameWidth)

/-- Theorem stating that for a frame with given dimensions, the sum of interior edges is 7 -/
theorem frame_interior_edges_sum :
  ∃ (f : Frame),
    f.outerLength = 7 ∧
    f.frameWidth = 2 ∧
    frameArea f = 30 ∧
    interiorEdgesSum f = 7 := by
  sorry

end frame_interior_edges_sum_l4097_409708


namespace quadratic_transformation_l4097_409763

/-- Given that ax^2 + bx + c can be expressed as 4(x - 5)^2 + 16, prove that when 5ax^2 + 5bx + 5c 
    is expressed in the form n(x - h)^2 + k, the value of h is 5. -/
theorem quadratic_transformation (a b c : ℝ) 
    (h : ∀ x, a * x^2 + b * x + c = 4 * (x - 5)^2 + 16) :
    ∃ n k, ∀ x, 5 * a * x^2 + 5 * b * x + 5 * c = n * (x - 5)^2 + k := by
  sorry

end quadratic_transformation_l4097_409763


namespace probability_of_six_on_fifth_roll_l4097_409745

def fair_die_prob : ℚ := 1 / 6
def biased_die_6_prob : ℚ := 2 / 3
def biased_die_6_other_prob : ℚ := 1 / 15
def biased_die_3_prob : ℚ := 1 / 2
def biased_die_3_other_prob : ℚ := 1 / 10

def initial_pick_prob : ℚ := 1 / 3

def observed_rolls : ℕ := 4
def observed_sixes : ℕ := 3
def observed_threes : ℕ := 1

theorem probability_of_six_on_fifth_roll :
  let fair_prob := initial_pick_prob * (fair_die_prob ^ observed_sixes * fair_die_prob ^ observed_threes)
  let biased_6_prob := initial_pick_prob * (biased_die_6_prob ^ observed_sixes * biased_die_6_other_prob ^ observed_threes)
  let biased_3_prob := initial_pick_prob * (biased_die_3_other_prob ^ observed_sixes * biased_die_3_prob ^ observed_threes)
  let total_prob := fair_prob + biased_6_prob + biased_3_prob
  (biased_6_prob / total_prob) * biased_die_6_prob = 8 / 135 / (3457.65 / 3888) * (2 / 3) := by
  sorry

end probability_of_six_on_fifth_roll_l4097_409745


namespace sum_of_fifth_powers_divisible_by_30_l4097_409789

theorem sum_of_fifth_powers_divisible_by_30 (a b c : ℤ) (h : 30 ∣ (a + b + c)) :
  30 ∣ (a^5 + b^5 + c^5) := by sorry

end sum_of_fifth_powers_divisible_by_30_l4097_409789


namespace function_and_angle_theorem_l4097_409754

/-- Given a function f and an angle α, proves that f(x) = cos x and 
    (√2 f(2α - π/4) - 1) / (1 - tan α) = 2/5 under certain conditions -/
theorem function_and_angle_theorem (f : ℝ → ℝ) (ω φ α : ℝ) : 
  ω > 0 → 
  0 ≤ φ ∧ φ ≤ π → 
  (∀ x, f x = Real.sin (ω * x + φ)) →
  (∀ x, f x = f (-x)) →
  (∃ x₁ x₂, abs (x₁ - x₂) = Real.sqrt (4 + Real.pi^2) ∧ 
    f x₁ = 1 ∧ f x₂ = -1) →
  Real.tan α + 1 / Real.tan α = 5 →
  (∀ x, f x = Real.cos x) ∧ 
  (Real.sqrt 2 * f (2 * α - Real.pi / 4) - 1) / (1 - Real.tan α) = 2 / 5 := by
sorry

end function_and_angle_theorem_l4097_409754


namespace teddy_bear_production_solution_l4097_409767

/-- Represents the teddy bear production problem -/
structure TeddyBearProduction where
  /-- The number of days originally planned -/
  days : ℕ
  /-- The number of teddy bears ordered -/
  order : ℕ

/-- The conditions of the teddy bear production problem are satisfied -/
def satisfies_conditions (p : TeddyBearProduction) : Prop :=
  20 * p.days + 100 = p.order ∧ 23 * p.days - 20 = p.order

/-- The theorem stating the solution to the teddy bear production problem -/
theorem teddy_bear_production_solution :
  ∃ (p : TeddyBearProduction), satisfies_conditions p ∧ p.days = 40 ∧ p.order = 900 :=
sorry

end teddy_bear_production_solution_l4097_409767


namespace phd_time_ratio_l4097_409750

/-- Represents the time spent in years for each phase of John's PhD journey -/
structure PhDTime where
  total : ℝ
  acclimation : ℝ
  basics : ℝ
  research : ℝ
  dissertation : ℝ

/-- Theorem stating the ratio of dissertation writing time to acclimation time -/
theorem phd_time_ratio (t : PhDTime) : 
  t.total = 7 ∧ 
  t.acclimation = 1 ∧ 
  t.basics = 2 ∧ 
  t.research = t.basics * 1.75 ∧
  t.total = t.acclimation + t.basics + t.research + t.dissertation →
  t.dissertation / t.acclimation = 0.5 := by
  sorry


end phd_time_ratio_l4097_409750


namespace intersection_when_m_is_2_sufficient_not_necessary_condition_l4097_409757

def p (x : ℝ) := x^2 + 2*x - 8 < 0

def q (x m : ℝ) := (x - 1 + m)*(x - 1 - m) ≤ 0

def A := {x : ℝ | p x}

def B (m : ℝ) := {x : ℝ | q x m}

theorem intersection_when_m_is_2 :
  B 2 ∩ A = {x : ℝ | -1 ≤ x ∧ x < 2} :=
sorry

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, p x → q x m) ∧ (∃ x : ℝ, q x m ∧ ¬p x) ↔ m ≥ 5 :=
sorry

end intersection_when_m_is_2_sufficient_not_necessary_condition_l4097_409757


namespace perpendicular_bisector_of_intersection_l4097_409716

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the perpendicular bisector
def perpendicular_bisector (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersection :
  ∀ (x y : ℝ), perpendicular_bisector x y ↔
  (∃ (t : ℝ), (1 - t) • A.1 + t • B.1 = x ∧ (1 - t) • A.2 + t • B.2 = y) :=
sorry

end perpendicular_bisector_of_intersection_l4097_409716


namespace product_equivalence_l4097_409762

theorem product_equivalence : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
  (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 := by
  sorry

end product_equivalence_l4097_409762


namespace problem_statement_l4097_409703

theorem problem_statement : (2002 - 1999)^2 / 169 = 9 / 169 := by
  sorry

end problem_statement_l4097_409703


namespace fishing_line_sections_l4097_409797

theorem fishing_line_sections (num_reels : ℕ) (reel_length : ℕ) (section_length : ℕ) : 
  num_reels = 3 → reel_length = 100 → section_length = 10 → 
  (num_reels * reel_length) / section_length = 30 := by
  sorry

end fishing_line_sections_l4097_409797


namespace base_number_proof_l4097_409784

theorem base_number_proof (x : ℝ) : 9^7 = x^14 → x = 3 := by
  sorry

end base_number_proof_l4097_409784


namespace best_fit_highest_r_squared_model_1_best_fit_l4097_409722

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  name : String
  r_squared : Real

/-- Determines if a model has the best fit among a list of models -/
def has_best_fit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

/-- The model with the highest R² value has the best fit -/
theorem best_fit_highest_r_squared (models : List RegressionModel) (model : RegressionModel) 
    (h : model ∈ models) :
    has_best_fit model models ↔ ∀ m ∈ models, model.r_squared ≥ m.r_squared :=
  sorry

/-- Given four specific models, prove that Model ① has the best fit -/
theorem model_1_best_fit :
  let models : List RegressionModel := [
    ⟨"①", 0.976⟩,
    ⟨"②", 0.776⟩,
    ⟨"③", 0.076⟩,
    ⟨"④", 0.351⟩
  ]
  let model_1 : RegressionModel := ⟨"①", 0.976⟩
  has_best_fit model_1 models :=
  sorry

end best_fit_highest_r_squared_model_1_best_fit_l4097_409722
