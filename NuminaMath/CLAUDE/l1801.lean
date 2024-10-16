import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_geometric_progression_l1801_180138

theorem sum_of_squares_geometric_progression
  (a r : ℝ) 
  (h1 : -1 < r ∧ r < 1) 
  (h2 : ∃ (S : ℝ), S = a / (1 - r)) : 
  ∃ (T : ℝ), T = a^2 / (1 - r^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_geometric_progression_l1801_180138


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l1801_180172

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (t : ℝ), (x + 3)^2 + (y + 2)^2 = 25 ∧ t - y + 1 = 0

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  y = (5/12) * x + 43/12

theorem circle_and_line_properties :
  -- Circle C passes through (0,2) and (2,-2)
  circle_C 0 2 ∧ circle_C 2 (-2) ∧
  -- Line m passes through (1,4)
  line_m 1 4 ∧
  -- The chord length of the intersection between circle C and line m is 6
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_m x₁ y₁ ∧ line_m x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) :=
by
  sorry

#check circle_and_line_properties

end NUMINAMATH_CALUDE_circle_and_line_properties_l1801_180172


namespace NUMINAMATH_CALUDE_inequality_product_l1801_180163

theorem inequality_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_product_l1801_180163


namespace NUMINAMATH_CALUDE_mrs_hilt_carnival_tickets_cost_l1801_180149

/-- Represents the cost and quantity of carnival tickets --/
structure CarnivalTickets where
  kids_usual_cost : ℚ
  kids_usual_quantity : ℕ
  adults_usual_cost : ℚ
  adults_usual_quantity : ℕ
  kids_deal_cost : ℚ
  kids_deal_quantity : ℕ
  adults_deal_cost : ℚ
  adults_deal_quantity : ℕ
  kids_bought : ℕ
  adults_bought : ℕ

/-- Calculates the total cost of carnival tickets --/
def total_cost (tickets : CarnivalTickets) : ℚ :=
  let kids_deal_used := tickets.kids_bought / tickets.kids_deal_quantity
  let kids_usual_used := tickets.kids_bought % tickets.kids_deal_quantity / tickets.kids_usual_quantity
  let adults_deal_used := tickets.adults_bought / tickets.adults_deal_quantity
  let adults_usual_used := tickets.adults_bought % tickets.adults_deal_quantity / tickets.adults_usual_quantity
  kids_deal_used * tickets.kids_deal_cost +
  kids_usual_used * tickets.kids_usual_cost +
  adults_deal_used * tickets.adults_deal_cost +
  adults_usual_used * tickets.adults_usual_cost

/-- Theorem: The total cost of Mrs. Hilt's carnival tickets is $15 --/
theorem mrs_hilt_carnival_tickets_cost :
  let tickets : CarnivalTickets := {
    kids_usual_cost := 1/4,
    kids_usual_quantity := 4,
    adults_usual_cost := 2/3,
    adults_usual_quantity := 3,
    kids_deal_cost := 4,
    kids_deal_quantity := 20,
    adults_deal_cost := 8,
    adults_deal_quantity := 15,
    kids_bought := 24,
    adults_bought := 18
  }
  total_cost tickets = 15 := by sorry


end NUMINAMATH_CALUDE_mrs_hilt_carnival_tickets_cost_l1801_180149


namespace NUMINAMATH_CALUDE_math_expressions_evaluation_l1801_180131

theorem math_expressions_evaluation :
  (∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y) →
  (∀ (x : ℝ), x ≥ 0 → (Real.sqrt x) ^ 2 = x) →
  (∀ (x y : ℝ), y ≠ 0 → Real.sqrt (x / y) = Real.sqrt x / Real.sqrt y) →
  (Real.sqrt 5 * Real.sqrt 15 - Real.sqrt 12 = 3 * Real.sqrt 3) ∧
  ((Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) = 1) ∧
  ((Real.sqrt 20 + 5) / Real.sqrt 5 = 2 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_math_expressions_evaluation_l1801_180131


namespace NUMINAMATH_CALUDE_three_lines_intersection_l1801_180102

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The three lines in the problem -/
def line1 : Line := ⟨1, 1, 1⟩
def line2 : Line := ⟨2, -1, 8⟩
def line3 (a : ℝ) : Line := ⟨a, 3, -5⟩

/-- The theorem to be proved -/
theorem three_lines_intersection (a : ℝ) : 
  (parallel line1 (line3 a) ∨ parallel line2 (line3 a)) → 
  a = 3 ∨ a = -6 :=
sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l1801_180102


namespace NUMINAMATH_CALUDE_sales_volume_equation_l1801_180189

def daily_sales_volume (x : ℝ) : ℝ := -x + 38

theorem sales_volume_equation :
  (∀ x y : ℝ, y = daily_sales_volume x → (x = 13 → y = 25) ∧ (x = 18 → y = 20)) ∧
  (daily_sales_volume 13 = 25) ∧
  (daily_sales_volume 18 = 20) :=
by sorry

end NUMINAMATH_CALUDE_sales_volume_equation_l1801_180189


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l1801_180170

theorem min_value_x_plus_2y (x y : ℝ) (h : x^2 + 4*y^2 - 2*x + 8*y + 1 = 0) :
  ∃ (m : ℝ), m = -2*Real.sqrt 2 - 1 ∧ ∀ (a b : ℝ), a^2 + 4*b^2 - 2*a + 8*b + 1 = 0 → m ≤ a + 2*b :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l1801_180170


namespace NUMINAMATH_CALUDE_power_of_three_simplification_l1801_180164

theorem power_of_three_simplification :
  3^2012 - 6 * 3^2013 + 2 * 3^2014 = 3^2012 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_simplification_l1801_180164


namespace NUMINAMATH_CALUDE_function_properties_l1801_180123

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + Real.cos x + a

theorem function_properties (a : ℝ) :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f a (x + T) = f a x ∧ 
   ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f a (x + T') = f a x) → T ≤ T') ∧
  (∃ (M : ℝ), M = 3 ∧ (∀ (x : ℝ), f a x ≤ M) → a = 1) ∧
  (∀ (k : ℤ), ∀ (x y : ℝ), 
    2 * k * Real.pi - 2 * Real.pi / 3 ≤ x ∧ 
    x < y ∧ 
    y ≤ 2 * k * Real.pi + Real.pi / 3 → 
    f a x < f a y) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1801_180123


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1801_180100

theorem complex_equation_solution (z : ℂ) : 
  (1 + 2*I)*z = 4 + 3*I → z = 2 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1801_180100


namespace NUMINAMATH_CALUDE_impossible_to_tile_rectangle_with_all_tetrominoes_l1801_180132

/-- Represents the different types of tetrominoes -/
inductive Tetromino
  | I
  | Square
  | Z
  | T
  | L

/-- Represents a color in a checkerboard pattern -/
inductive Color
  | Black
  | White

/-- Represents the coverage of squares by a tetromino on a checkerboard -/
structure TetrominoCoverage where
  black : Nat
  white : Nat

/-- The number of squares covered by each tetromino -/
def tetromino_size : Nat := 4

/-- The coverage of squares by each type of tetromino on a checkerboard -/
def tetromino_coverage (t : Tetromino) : TetrominoCoverage :=
  match t with
  | Tetromino.I => ⟨2, 2⟩
  | Tetromino.Square => ⟨2, 2⟩
  | Tetromino.Z => ⟨2, 2⟩
  | Tetromino.L => ⟨2, 2⟩
  | Tetromino.T => ⟨3, 1⟩  -- or ⟨1, 3⟩, doesn't matter for the proof

/-- Theorem stating that it's impossible to tile a rectangle with one of each tetromino type -/
theorem impossible_to_tile_rectangle_with_all_tetrominoes :
  ¬ ∃ (w h : Nat), w * h = 5 * tetromino_size ∧
    (∃ (c : Color), 
      (List.sum (List.map (λ t => (tetromino_coverage t).black) [Tetromino.I, Tetromino.Square, Tetromino.Z, Tetromino.T, Tetromino.L]) = w * h / 2) ∨
      (List.sum (List.map (λ t => (tetromino_coverage t).white) [Tetromino.I, Tetromino.Square, Tetromino.Z, Tetromino.T, Tetromino.L]) = w * h / 2)) :=
by sorry


end NUMINAMATH_CALUDE_impossible_to_tile_rectangle_with_all_tetrominoes_l1801_180132


namespace NUMINAMATH_CALUDE_covered_digits_sum_l1801_180117

/-- Represents a five-digit number with some digits possibly covered -/
structure PartialNumber :=
  (d1 d2 d3 d4 d5 : Option Nat)

/-- The sum of the visible digits in a PartialNumber -/
def visibleSum (n : PartialNumber) : Nat :=
  (n.d1.getD 0) * 10000 + (n.d2.getD 0) * 1000 + (n.d3.getD 0) * 100 + (n.d4.getD 0) * 10 + (n.d5.getD 0)

/-- The number of covered digits in a PartialNumber -/
def coveredCount (n : PartialNumber) : Nat :=
  (if n.d1.isNone then 1 else 0) +
  (if n.d2.isNone then 1 else 0) +
  (if n.d3.isNone then 1 else 0) +
  (if n.d4.isNone then 1 else 0) +
  (if n.d5.isNone then 1 else 0)

theorem covered_digits_sum (n1 n2 n3 : PartialNumber) :
  visibleSum n1 + visibleSum n2 + visibleSum n3 = 57263 - 1000 - 200 - 9 ∧
  coveredCount n1 + coveredCount n2 + coveredCount n3 = 3 →
  ∃ (p1 p2 p3 : Nat), 
    (p1 = 1 ∧ p2 = 2 ∧ p3 = 9) ∧
    visibleSum n1 + visibleSum n2 + visibleSum n3 + p1 * 1000 + p2 * 100 + p3 = 57263 :=
by sorry


end NUMINAMATH_CALUDE_covered_digits_sum_l1801_180117


namespace NUMINAMATH_CALUDE_hot_dog_buns_per_student_l1801_180141

theorem hot_dog_buns_per_student (
  buns_per_package : ℕ)
  (packages_bought : ℕ)
  (num_classes : ℕ)
  (students_per_class : ℕ)
  (h1 : buns_per_package = 8)
  (h2 : packages_bought = 30)
  (h3 : num_classes = 4)
  (h4 : students_per_class = 30)
  : (buns_per_package * packages_bought) / (num_classes * students_per_class) = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_buns_per_student_l1801_180141


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1801_180197

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1801_180197


namespace NUMINAMATH_CALUDE_fourth_root_inequality_l1801_180156

theorem fourth_root_inequality (x : ℝ) :
  (x^(1/4) - 3 / (x^(1/4) + 4) ≥ 0) ↔ (0 ≤ x ∧ x ≤ 81) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_inequality_l1801_180156


namespace NUMINAMATH_CALUDE_masters_proportion_in_team_l1801_180188

/-- Represents a team of juniors and masters in a shooting tournament. -/
structure ShootingTeam where
  juniors : ℕ
  masters : ℕ

/-- Calculates the proportion of masters in the team. -/
def mastersProportion (team : ShootingTeam) : ℚ :=
  team.masters / (team.juniors + team.masters)

/-- The theorem stating the proportion of masters in the team under given conditions. -/
theorem masters_proportion_in_team (team : ShootingTeam) 
  (h1 : 22 * team.juniors + 47 * team.masters = 41 * (team.juniors + team.masters)) :
  mastersProportion team = 19 / 25 := by
  sorry

#eval (19 : ℚ) / 25  -- To verify that 19/25 is indeed equal to 0.76

end NUMINAMATH_CALUDE_masters_proportion_in_team_l1801_180188


namespace NUMINAMATH_CALUDE_two_digit_cube_sum_square_l1801_180185

theorem two_digit_cube_sum_square : 
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ (((n / 10) + (n % 10))^3 = n^2) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_cube_sum_square_l1801_180185


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1801_180193

theorem complex_magnitude_problem (z : ℂ) (h : (z + 1) / (z - 2) = 1 - 3*I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1801_180193


namespace NUMINAMATH_CALUDE_cylinder_volume_l1801_180158

/-- The volume of a cylinder whose lateral surface unfolds into a square with side length 4 -/
theorem cylinder_volume (h : Real) (r : Real) : 
  h = 4 ∧ 2 * Real.pi * r = 4 → Real.pi * r^2 * h = 16 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l1801_180158


namespace NUMINAMATH_CALUDE_airport_distance_l1801_180166

-- Define the problem parameters
def initial_speed : ℝ := 45
def speed_increase : ℝ := 20
def late_time : ℝ := 0.75  -- 45 minutes in hours
def early_time : ℝ := 0.25  -- 15 minutes in hours

-- Define the theorem
theorem airport_distance : ∃ (d : ℝ), d = 241.875 ∧ 
  ∃ (t : ℝ), 
    d = initial_speed * (t + late_time) ∧
    d - initial_speed = (initial_speed + speed_increase) * (t - (1 + early_time)) :=
by
  sorry


end NUMINAMATH_CALUDE_airport_distance_l1801_180166


namespace NUMINAMATH_CALUDE_prime_iff_factorial_congruence_l1801_180180

theorem prime_iff_factorial_congruence (p : ℕ) (hp : p > 1) : 
  Nat.Prime p ↔ (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_iff_factorial_congruence_l1801_180180


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1801_180127

/-- A line in 2D space defined by the equation x - y - 1 = 0 -/
def line (x y : ℝ) : Prop := x - y - 1 = 0

/-- The second quadrant of a 2D coordinate system -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Theorem stating that the line x - y - 1 = 0 does not pass through the second quadrant -/
theorem line_not_in_second_quadrant :
  ∀ x y : ℝ, line x y → ¬(second_quadrant x y) :=
sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1801_180127


namespace NUMINAMATH_CALUDE_candy_parade_total_l1801_180109

/-- The total number of candy pieces caught by Tabitha and her friends at the Christmas parade -/
theorem candy_parade_total (tabitha stan : ℕ) (julie carlos : ℕ) 
    (h1 : tabitha = 22)
    (h2 : stan = 13)
    (h3 : julie = tabitha / 2)
    (h4 : carlos = 2 * stan) :
  tabitha + stan + julie + carlos = 72 := by
  sorry

end NUMINAMATH_CALUDE_candy_parade_total_l1801_180109


namespace NUMINAMATH_CALUDE_polygon_sides_l1801_180168

theorem polygon_sides (sum_interior_angles : ℝ) (n : ℕ) : 
  sum_interior_angles = 1260 → (n - 2) * 180 = sum_interior_angles → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1801_180168


namespace NUMINAMATH_CALUDE_f_properties_l1801_180153

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem f_properties :
  (∀ x ∈ Set.Ioo (-2 : ℝ) 0, ∀ y ∈ Set.Ioo (-2 : ℝ) 0, x < y → f x > f y) ∧
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = x₁ - 2012 ∧ f x₂ = x₂ - 2012 ∧
    ∀ x, x ≠ x₁ ∧ x ≠ x₂ → f x ≠ x - 2012) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1801_180153


namespace NUMINAMATH_CALUDE_remainder_theorem_l1801_180110

theorem remainder_theorem : 7 * 10^20 + 1^20 ≡ 8 [ZMOD 9] := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1801_180110


namespace NUMINAMATH_CALUDE_salary_decrease_percentage_l1801_180101

def original_salary : ℝ := 4000.0000000000005
def initial_increase_rate : ℝ := 0.1
def final_salary : ℝ := 4180

theorem salary_decrease_percentage :
  ∃ (decrease_rate : ℝ),
    final_salary = original_salary * (1 + initial_increase_rate) * (1 - decrease_rate) ∧
    decrease_rate = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_salary_decrease_percentage_l1801_180101


namespace NUMINAMATH_CALUDE_chess_tournament_players_l1801_180174

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- number of players excluding the lowest 8
  total_players : ℕ := n + 8
  
  -- Each player played exactly one game against each other player
  total_games : ℕ := total_players.choose 2
  
  -- Point distribution condition
  point_distribution : 
    2 * n.choose 2 + 56 = (total_players * (total_players - 1)) / 2

/-- The theorem stating that the total number of players in the tournament is 21 -/
theorem chess_tournament_players : 
  ∀ t : ChessTournament, t.total_players = 21 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l1801_180174


namespace NUMINAMATH_CALUDE_simplify_expression_l1801_180150

theorem simplify_expression (c : ℝ) : ((3 * c + 5) - 3 * c) / 2 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1801_180150


namespace NUMINAMATH_CALUDE_chipped_marbles_count_l1801_180181

def marble_counts : List Nat := [17, 20, 22, 24, 26, 35, 37, 40]

def total_marbles : Nat := marble_counts.sum

theorem chipped_marbles_count (jane_count george_count : Nat) 
  (h1 : jane_count = 3 * george_count)
  (h2 : jane_count + george_count = total_marbles - (marble_counts.get! 0 + marble_counts.get! 7))
  (h3 : ∃ (i j : Fin 8), i ≠ j ∧ 
    marble_counts.get! i.val + marble_counts.get! j.val = total_marbles - (jane_count + george_count) ∧
    (marble_counts.get! i.val = 40 ∨ marble_counts.get! j.val = 40)) :
  40 ∈ marble_counts ∧ 
  ∃ (i j : Fin 8), i ≠ j ∧ 
    marble_counts.get! i.val + marble_counts.get! j.val = total_marbles - (jane_count + george_count) ∧
    (marble_counts.get! i.val = 40 ∨ marble_counts.get! j.val = 40) :=
by sorry

end NUMINAMATH_CALUDE_chipped_marbles_count_l1801_180181


namespace NUMINAMATH_CALUDE_triangle_problem_l1801_180129

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : (t.a + t.c) / (t.a + t.b) = (t.b - t.a) / t.c)
  (h2 : t.b = Real.sqrt 7)
  (h3 : Real.sin t.C = 2 * Real.sin t.A) :
  t.B = 2 * π / 3 ∧ min t.a (min t.b t.c) = 1 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1801_180129


namespace NUMINAMATH_CALUDE_intersection_M_N_l1801_180121

-- Define the sets M and N
def M : Set ℝ := {x | x > -1}
def N : Set ℝ := {x | x^2 - x - 6 < 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1801_180121


namespace NUMINAMATH_CALUDE_find_n_l1801_180178

theorem find_n : ∃ n : ℤ, (7 : ℝ) ^ (2 * n) = (1 / 49) ^ (n - 12) ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_find_n_l1801_180178


namespace NUMINAMATH_CALUDE_cannonball_max_height_l1801_180122

/-- The height function of the cannonball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- The maximum height reached by the cannonball -/
def max_height : ℝ := 161

/-- Theorem stating that the maximum height reached by the cannonball is 161 meters -/
theorem cannonball_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_cannonball_max_height_l1801_180122


namespace NUMINAMATH_CALUDE_spaceship_total_distance_l1801_180177

/-- The distance traveled by a spaceship between three locations -/
def spaceship_distance (earth_to_x : ℝ) (x_to_y : ℝ) (y_to_earth : ℝ) : ℝ :=
  earth_to_x + x_to_y + y_to_earth

/-- Theorem: The total distance traveled by the spaceship is 0.7 light-years -/
theorem spaceship_total_distance :
  spaceship_distance 0.5 0.1 0.1 = 0.7 := by
  sorry

#eval spaceship_distance 0.5 0.1 0.1

end NUMINAMATH_CALUDE_spaceship_total_distance_l1801_180177


namespace NUMINAMATH_CALUDE_cat_dog_positions_l1801_180175

/-- Represents the number of positions for the cat -/
def cat_positions : Nat := 4

/-- Represents the number of positions for the dog -/
def dog_positions : Nat := 6

/-- Represents the total number of moves -/
def total_moves : Nat := 317

/-- Calculates the final position of an animal given its number of positions and total moves -/
def final_position (positions : Nat) (moves : Nat) : Nat :=
  moves % positions

theorem cat_dog_positions :
  (final_position cat_positions total_moves = 0) ∧
  (final_position dog_positions total_moves = 5) := by
  sorry

end NUMINAMATH_CALUDE_cat_dog_positions_l1801_180175


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l1801_180148

/-- Proves that adding 1.8 litres of pure alcohol to a 6-litre solution
    that is 35% alcohol results in a 50% alcohol solution -/
theorem alcohol_solution_proof :
  let initial_volume : ℝ := 6
  let initial_percentage : ℝ := 0.35
  let target_percentage : ℝ := 0.5
  let added_alcohol : ℝ := 1.8
  let final_volume := initial_volume + added_alcohol
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  (final_alcohol / final_volume) = target_percentage := by sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l1801_180148


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_point_l1801_180108

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the left focus F
def left_focus : ℝ × ℝ := (-1, 0)

-- Define line l (implicitly through its properties)
def line_l (x y : ℝ) : Prop := ∃ k : ℝ, y = k * (x + 1)

-- Define points A and B
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the perpendicular point H
def point_H : ℝ × ℝ := sorry

-- State the theorem
theorem ellipse_perpendicular_point :
  ∀ (A B : ℝ × ℝ),
    ellipse A.1 A.2 →
    ellipse B.1 B.2 →
    line_l A.1 A.2 →
    line_l B.1 B.2 →
    (A.1 * B.1 + A.2 * B.2 = 0) →
    (point_H = (-2/3, Real.sqrt 2/3) ∨ point_H = (-2/3, -Real.sqrt 2/3)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_point_l1801_180108


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l1801_180179

theorem perpendicular_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, x + 2 * m * y - 1 = 0 ∧ (3 * m - 1) * x - m * y - 1 = 0 →
    ((-1 / (2 * m)) * ((3 * m - 1) / m) = -1)) →
  m = 1 ∨ m = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l1801_180179


namespace NUMINAMATH_CALUDE_square_difference_of_solutions_l1801_180176

theorem square_difference_of_solutions (α β : ℝ) : 
  α^2 = 2*α + 1 → β^2 = 2*β + 1 → α ≠ β → (α - β)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_solutions_l1801_180176


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l1801_180191

theorem product_from_lcm_gcd (x y : ℕ+) 
  (h_lcm : Nat.lcm x y = 72) 
  (h_gcd : Nat.gcd x y = 8) : 
  x * y = 576 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l1801_180191


namespace NUMINAMATH_CALUDE_lindas_furniture_spending_l1801_180184

/-- Given Linda's original savings and the cost of a TV, 
    prove the fraction of her savings spent on furniture. -/
theorem lindas_furniture_spending 
  (original_savings : ℚ) 
  (tv_cost : ℚ) 
  (h1 : original_savings = 600)
  (h2 : tv_cost = 150) : 
  (original_savings - tv_cost) / original_savings = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_lindas_furniture_spending_l1801_180184


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1801_180111

/-- Given a parabola with equation x^2 = 12y, the distance from its focus to its directrix is 6 -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), x^2 = 12*y →
  ∃ (focus_x focus_y directrix_y : ℝ),
    (focus_x = 0 ∧ focus_y = 3 ∧ directrix_y = -3) ∧
    (focus_y - directrix_y = 6) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1801_180111


namespace NUMINAMATH_CALUDE_special_operation_result_l1801_180107

/-- The "※" operation for integers -/
def star (a b : ℤ) : ℤ := a + b - 1

/-- The "#" operation for integers -/
def hash (a b : ℤ) : ℤ := a * b - 1

/-- Theorem stating that 4#[(6※8)※(3#5)] = 103 -/
theorem special_operation_result : hash 4 (star (star 6 8) (hash 3 5)) = 103 := by
  sorry

end NUMINAMATH_CALUDE_special_operation_result_l1801_180107


namespace NUMINAMATH_CALUDE_sin_shift_pi_half_even_sin_shift_pi_half_is_even_l1801_180192

/-- The function f(x) = 2sin(x + π/2) is an even function -/
theorem sin_shift_pi_half_even (x : ℝ) : 2 * Real.sin (x + π / 2) = 2 * Real.sin ((-x) + π / 2) := by
  sorry

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = 2sin(x + π/2) is an even function -/
theorem sin_shift_pi_half_is_even : IsEven (λ x ↦ 2 * Real.sin (x + π / 2)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_pi_half_even_sin_shift_pi_half_is_even_l1801_180192


namespace NUMINAMATH_CALUDE_b_current_age_l1801_180159

/-- Given two people A and B, where:
    1) In 10 years, A will be twice as old as B was 10 years ago.
    2) A is currently 8 years older than B.
    This theorem proves that B's current age is 38 years. -/
theorem b_current_age (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10))
  (h2 : a = b + 8) : 
  b = 38 := by
sorry

end NUMINAMATH_CALUDE_b_current_age_l1801_180159


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l1801_180139

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℝ := 3 * n + 4 * n^2

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℝ := 8 * r - 1

theorem arithmetic_progression_rth_term (r : ℕ) :
  a r = S r - S (r - 1) := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l1801_180139


namespace NUMINAMATH_CALUDE_jacket_price_calculation_l1801_180195

/-- Calculates the final price of a jacket after discount and tax. -/
def finalPrice (originalPrice discount tax : ℝ) : ℝ :=
  let discountedPrice := originalPrice * (1 - discount)
  discountedPrice * (1 + tax)

/-- Theorem stating that the final price of a $120 jacket with 30% discount and 10% tax is $92.4 -/
theorem jacket_price_calculation :
  finalPrice 120 0.3 0.1 = 92.4 := by
  sorry

#eval finalPrice 120 0.3 0.1

end NUMINAMATH_CALUDE_jacket_price_calculation_l1801_180195


namespace NUMINAMATH_CALUDE_rain_probability_l1801_180199

theorem rain_probability (p_monday p_tuesday p_no_rain : ℝ) 
  (h1 : p_monday = 0.62)
  (h2 : p_tuesday = 0.54)
  (h3 : p_no_rain = 0.28)
  : p_monday + p_tuesday - (1 - p_no_rain) = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l1801_180199


namespace NUMINAMATH_CALUDE_dinitrogen_trioxide_weight_l1801_180190

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Nitrogen atoms in Dinitrogen trioxide -/
def N_count : ℕ := 2

/-- The number of Oxygen atoms in Dinitrogen trioxide -/
def O_count : ℕ := 3

/-- The molecular weight of Dinitrogen trioxide in g/mol -/
def molecular_weight_N2O3 : ℝ := N_count * atomic_weight_N + O_count * atomic_weight_O

/-- Theorem stating that the molecular weight of Dinitrogen trioxide is 76.02 g/mol -/
theorem dinitrogen_trioxide_weight : molecular_weight_N2O3 = 76.02 := by
  sorry

end NUMINAMATH_CALUDE_dinitrogen_trioxide_weight_l1801_180190


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l1801_180126

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 7 = (y + 2)^2

-- Define the intersection points
def intersection_points : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
    (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
    (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
    (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄)

-- Theorem statement
theorem intersection_sum_zero :
  intersection_points →
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
    (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
    (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
    (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l1801_180126


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l1801_180130

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : 1 / x + 1 / y = 1) 
  (h4 : 1 / x - 1 / y = 5) : 
  x + y = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l1801_180130


namespace NUMINAMATH_CALUDE_ferry_crossings_parity_ferry_crossings_opposite_ferry_after_99_crossings_l1801_180194

/-- Represents the two banks of the river --/
inductive Bank : Type
| Left : Bank
| Right : Bank

/-- Returns the opposite bank --/
def opposite_bank (b : Bank) : Bank :=
  match b with
  | Bank.Left => Bank.Right
  | Bank.Right => Bank.Left

/-- Represents the state of the ferry after a number of crossings --/
def ferry_position (start : Bank) (crossings : Nat) : Bank :=
  if crossings % 2 = 0 then start else opposite_bank start

theorem ferry_crossings_parity (start : Bank) (crossings : Nat) :
  ferry_position start crossings = start ↔ crossings % 2 = 0 :=
sorry

theorem ferry_crossings_opposite (start : Bank) (crossings : Nat) :
  ferry_position start crossings = opposite_bank start ↔ crossings % 2 = 1 :=
sorry

theorem ferry_after_99_crossings (start : Bank) :
  ferry_position start 99 = opposite_bank start :=
sorry

end NUMINAMATH_CALUDE_ferry_crossings_parity_ferry_crossings_opposite_ferry_after_99_crossings_l1801_180194


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l1801_180134

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), a ≡ 61 [ZMOD 70] ∧ b ≡ 43 [ZMOD 70] ∧ (3 * a + 9 * b) ≡ 0 [ZMOD 70] := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l1801_180134


namespace NUMINAMATH_CALUDE_proportion_equality_l1801_180157

theorem proportion_equality (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l1801_180157


namespace NUMINAMATH_CALUDE_unique_pair_divisibility_l1801_180182

theorem unique_pair_divisibility : 
  ∃! (n m : ℕ), n > 2 ∧ m > 2 ∧ 
  (∃ (S : Set ℕ), Set.Infinite S ∧ 
    ∀ k ∈ S, (k^n + k^2 - 1) ∣ (k^m + k - 1)) ∧
  n = 3 ∧ m = 5 :=
sorry

end NUMINAMATH_CALUDE_unique_pair_divisibility_l1801_180182


namespace NUMINAMATH_CALUDE_complement_M_inter_N_eq_singleton_three_l1801_180183

def M : Set ℤ := {x | x < 3}
def N : Set ℤ := {x | x < 4}
def U : Set ℤ := Set.univ

theorem complement_M_inter_N_eq_singleton_three :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_M_inter_N_eq_singleton_three_l1801_180183


namespace NUMINAMATH_CALUDE_expression_evaluation_l1801_180165

theorem expression_evaluation : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) + 1 / 4 = 37 / 60 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1801_180165


namespace NUMINAMATH_CALUDE_quadrilateral_is_trapezoid_l1801_180119

/-- A quadrilateral with two parallel sides of different lengths is a trapezoid -/
def is_trapezoid (A B C D : ℝ × ℝ) : Prop :=
  ∃ (l₁ l₂ : ℝ), l₁ ≠ l₂ ∧ 
  (B.1 - A.1) / (B.2 - A.2) = (D.1 - C.1) / (D.2 - C.2) ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = l₁^2 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = l₂^2

/-- The quadratic equation whose roots are the lengths of AB and CD -/
def length_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 - 3*m*x + 2*m^2 + m - 2 = 0

theorem quadrilateral_is_trapezoid (A B C D : ℝ × ℝ) (m : ℝ) :
  (∃ (l₁ l₂ : ℝ), 
    length_equation m l₁ ∧ 
    length_equation m l₂ ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = l₁^2 ∧
    (D.1 - C.1)^2 + (D.2 - C.2)^2 = l₂^2 ∧
    (B.1 - A.1) / (B.2 - A.2) = (D.1 - C.1) / (D.2 - C.2)) →
  is_trapezoid A B C D :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_is_trapezoid_l1801_180119


namespace NUMINAMATH_CALUDE_ear_muffs_total_l1801_180171

/-- The number of ear muffs bought before December -/
def before_december : ℕ := 1346

/-- The number of ear muffs bought during December -/
def during_december : ℕ := 6444

/-- The total number of ear muffs bought -/
def total_ear_muffs : ℕ := before_december + during_december

theorem ear_muffs_total : total_ear_muffs = 7790 := by
  sorry

end NUMINAMATH_CALUDE_ear_muffs_total_l1801_180171


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1801_180162

theorem rectangle_perimeter (a b : ℝ) (h1 : a * b = 24) (h2 : a^2 + b^2 = 11^2) : 
  2 * (a + b) = 26 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1801_180162


namespace NUMINAMATH_CALUDE_license_plate_count_l1801_180113

/-- The number of vowels in the alphabet, considering Y as a vowel -/
def num_vowels : ℕ := 6

/-- The number of consonants in the alphabet -/
def num_consonants : ℕ := 20

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_consonants * num_vowels * num_vowels * num_consonants * num_vowels * num_digits

theorem license_plate_count : total_license_plates = 403200 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1801_180113


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_tangent_lines_correct_l1801_180114

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a line is tangent to a circle centered at the origin -/
def isTangentToCircle (l : Line) (r : ℝ) : Prop :=
  (l.a ^ 2 + l.b ^ 2) * r ^ 2 = l.c ^ 2

theorem tangent_lines_to_circle (p : Point) (r : ℝ) :
  p.x ^ 2 + p.y ^ 2 > r ^ 2 →
  (∃ l₁ l₂ : Line,
    (pointOnLine p l₁ ∧ isTangentToCircle l₁ r) ∧
    (pointOnLine p l₂ ∧ isTangentToCircle l₂ r) ∧
    ((l₁.a = 1 ∧ l₁.b = 0 ∧ l₁.c = -p.x) ∨
     (l₂.a = 5 ∧ l₂.b = -12 ∧ l₂.c = 26))) :=
by sorry

/-- The main theorem that proves the given tangent lines are correct -/
theorem tangent_lines_correct : 
  ∃ l₁ l₂ : Line,
    (pointOnLine ⟨2, 3⟩ l₁ ∧ isTangentToCircle l₁ 2) ∧
    (pointOnLine ⟨2, 3⟩ l₂ ∧ isTangentToCircle l₂ 2) ∧
    ((l₁.a = 1 ∧ l₁.b = 0 ∧ l₁.c = -2) ∨
     (l₂.a = 5 ∧ l₂.b = -12 ∧ l₂.c = 26)) :=
by
  apply tangent_lines_to_circle ⟨2, 3⟩ 2
  norm_num

end NUMINAMATH_CALUDE_tangent_lines_to_circle_tangent_lines_correct_l1801_180114


namespace NUMINAMATH_CALUDE_triangle_special_angle_l1801_180143

open Real

/-- In a triangle ABC, given that 2b cos A = 2c - √3a, prove that angle B is π/6 --/
theorem triangle_special_angle (a b c : ℝ) (A B C : ℝ) (h : 2 * b * cos A = 2 * c - Real.sqrt 3 * a) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π →
  B = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l1801_180143


namespace NUMINAMATH_CALUDE_water_level_rise_l1801_180116

/-- Calculates the rise in water level when a cube is fully immersed in a rectangular vessel. -/
theorem water_level_rise (cube_edge : ℝ) (vessel_length : ℝ) (vessel_width : ℝ) :
  cube_edge = 17 →
  vessel_length = 20 →
  vessel_width = 15 →
  ∃ (water_rise : ℝ), abs (water_rise - (cube_edge^3 / (vessel_length * vessel_width))) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_water_level_rise_l1801_180116


namespace NUMINAMATH_CALUDE_walking_distance_l1801_180105

/-- 
Given a person who walks at 10 km/hr, if increasing their speed to 16 km/hr 
would allow them to walk 36 km more in the same time, then the actual distance 
traveled is 60 km.
-/
theorem walking_distance (actual_speed : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) 
  (h1 : actual_speed = 10)
  (h2 : faster_speed = 16)
  (h3 : extra_distance = 36)
  (h4 : (actual_distance / actual_speed) = ((actual_distance + extra_distance) / faster_speed)) :
  actual_distance = 60 :=
by
  sorry

#check walking_distance

end NUMINAMATH_CALUDE_walking_distance_l1801_180105


namespace NUMINAMATH_CALUDE_initial_water_ratio_l1801_180128

/-- Proves that the ratio of initial water to tank capacity is 1:2 given the specified conditions --/
theorem initial_water_ratio (tank_capacity : ℝ) (inflow_rate : ℝ) (outflow_rate1 : ℝ) (outflow_rate2 : ℝ) (fill_time : ℝ) :
  tank_capacity = 6000 →
  inflow_rate = 500 →
  outflow_rate1 = 250 →
  outflow_rate2 = 1000 / 6 →
  fill_time = 36 →
  (tank_capacity - (inflow_rate - outflow_rate1 - outflow_rate2) * fill_time) / tank_capacity = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_ratio_l1801_180128


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l1801_180120

theorem square_root_of_sixteen (x : ℝ) : x^2 = 16 ↔ x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l1801_180120


namespace NUMINAMATH_CALUDE_terminal_zeros_125_360_l1801_180135

def number_of_terminal_zeros (a b : ℕ) : ℕ :=
  sorry

theorem terminal_zeros_125_360 : 
  let a := 125
  let b := 360
  number_of_terminal_zeros a b = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_terminal_zeros_125_360_l1801_180135


namespace NUMINAMATH_CALUDE_star_three_two_l1801_180198

/-- The star operation defined as a * b = a * b^3 - b^2 + 2 -/
def star (a b : ℝ) : ℝ := a * b^3 - b^2 + 2

/-- Theorem stating that 3 star 2 equals 22 -/
theorem star_three_two : star 3 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_star_three_two_l1801_180198


namespace NUMINAMATH_CALUDE_sum_of_first_10_terms_equals_560_l1801_180125

def arithmetic_sequence_1 (n : ℕ) : ℕ := 4 * n - 2
def arithmetic_sequence_2 (n : ℕ) : ℕ := 6 * n - 4

def common_sequence (n : ℕ) : ℕ := 12 * n - 10

def sum_of_first_n_terms (n : ℕ) : ℕ := n * (12 * n - 8) / 2

theorem sum_of_first_10_terms_equals_560 :
  sum_of_first_n_terms 10 = 560 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_10_terms_equals_560_l1801_180125


namespace NUMINAMATH_CALUDE_negative_number_identification_l1801_180187

theorem negative_number_identification :
  let a := -(-2)
  let b := abs (-2)
  let c := (-2)^2
  let d := (-2)^3
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d < 0) := by sorry

end NUMINAMATH_CALUDE_negative_number_identification_l1801_180187


namespace NUMINAMATH_CALUDE_slope_range_l1801_180137

/-- A line passing through (0,2) that intersects the circle (x-2)^2 + (y-2)^2 = 1 -/
structure IntersectingLine where
  slope : ℝ
  passes_through_origin : (0 : ℝ) = slope * 0 + 2
  intersects_circle : ∃ (x y : ℝ), y = slope * x + 2 ∧ (x - 2)^2 + (y - 2)^2 = 1

/-- The theorem stating the range of possible slopes for the intersecting line -/
theorem slope_range (l : IntersectingLine) : 
  l.slope ∈ Set.Icc (-Real.sqrt 3 / 3) (Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_slope_range_l1801_180137


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l1801_180140

/-- Definition of a quadratic equation in x -/
def is_quadratic_in_x (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x - 6

/-- Theorem: The given equation is a quadratic equation in x -/
theorem equation_is_quadratic : is_quadratic_in_x f := by
  sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l1801_180140


namespace NUMINAMATH_CALUDE_unique_sums_count_l1801_180155

def bag_C : Finset ℕ := {1, 3, 7, 9}
def bag_D : Finset ℕ := {4, 6, 8}

theorem unique_sums_count : 
  Finset.card ((bag_C.product bag_D).image (fun (p : ℕ × ℕ) => p.1 + p.2)) = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_sums_count_l1801_180155


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l1801_180115

theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  a * Real.cos B = b * Real.sin A →
  C = π / 6 →
  c = 2 →
  b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l1801_180115


namespace NUMINAMATH_CALUDE_stick_length_4_forms_triangle_stick_length_1_cannot_form_triangle_stick_length_2_cannot_form_triangle_stick_length_3_cannot_form_triangle_l1801_180147

/-- Triangle inequality check function -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: A stick of length 4 can form a triangle with sticks of lengths 3 and 6 -/
theorem stick_length_4_forms_triangle :
  triangle_inequality 3 6 4 :=
sorry

/-- Theorem: A stick of length 1 cannot form a triangle with sticks of lengths 3 and 6 -/
theorem stick_length_1_cannot_form_triangle :
  ¬ triangle_inequality 3 6 1 :=
sorry

/-- Theorem: A stick of length 2 cannot form a triangle with sticks of lengths 3 and 6 -/
theorem stick_length_2_cannot_form_triangle :
  ¬ triangle_inequality 3 6 2 :=
sorry

/-- Theorem: A stick of length 3 cannot form a triangle with sticks of lengths 3 and 6 -/
theorem stick_length_3_cannot_form_triangle :
  ¬ triangle_inequality 3 6 3 :=
sorry

end NUMINAMATH_CALUDE_stick_length_4_forms_triangle_stick_length_1_cannot_form_triangle_stick_length_2_cannot_form_triangle_stick_length_3_cannot_form_triangle_l1801_180147


namespace NUMINAMATH_CALUDE_additional_toothpicks_eq_351_l1801_180118

/-- The number of toothpicks needed for a 3-step staircase -/
def initial_toothpicks : ℕ := 18

/-- The ratio of the geometric progression for toothpick increase -/
def progression_ratio : ℕ := 3

/-- Calculate the total additional toothpicks needed to complete a 6-step staircase -/
def additional_toothpicks : ℕ :=
  let step4_increase := (initial_toothpicks / 2) * progression_ratio
  let step5_increase := step4_increase * progression_ratio
  let step6_increase := step5_increase * progression_ratio
  step4_increase + step5_increase + step6_increase

/-- Theorem stating that the additional toothpicks needed is 351 -/
theorem additional_toothpicks_eq_351 : additional_toothpicks = 351 := by
  sorry

end NUMINAMATH_CALUDE_additional_toothpicks_eq_351_l1801_180118


namespace NUMINAMATH_CALUDE_find_T_l1801_180144

theorem find_T : ∃ T : ℚ, (1/3 : ℚ) * (1/5 : ℚ) * T = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ T = 75 := by
  sorry

end NUMINAMATH_CALUDE_find_T_l1801_180144


namespace NUMINAMATH_CALUDE_rectangle_square_comparison_l1801_180142

/-- Proves that for a rectangle with a 3:1 length-to-width ratio and 75 cm² area,
    the difference between the side of a square with equal area and the rectangle's width
    is greater than 3 cm. -/
theorem rectangle_square_comparison : ∀ (length width : ℝ),
  length / width = 3 →
  length * width = 75 →
  ∃ (square_side : ℝ),
    square_side^2 = 75 ∧
    square_side - width > 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_square_comparison_l1801_180142


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l1801_180106

theorem max_product_sum_2000 :
  (∀ a b : ℤ, a + b = 2000 → a * b ≤ 1000000) ∧
  (∃ a b : ℤ, a + b = 2000 ∧ a * b = 1000000) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l1801_180106


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_l1801_180154

theorem sqrt_abs_sum_zero_implies_power (m n : ℝ) :
  Real.sqrt (m - 2) + |n + 3| = 0 → (m + n)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_l1801_180154


namespace NUMINAMATH_CALUDE_complement_of_union_l1801_180104

-- Define the universal set U
def U : Set ℕ := {x | x ≤ 9 ∧ x > 0}

-- Define sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

-- State the theorem
theorem complement_of_union :
  (U \ (A ∪ B)) = {7, 8, 9} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l1801_180104


namespace NUMINAMATH_CALUDE_production_days_l1801_180136

/-- Given that:
    1. The average daily production for the past n days was 50 units.
    2. Today's production is 110 units.
    3. The new average including today's production is 55 units.
    Prove that n = 11. -/
theorem production_days (n : ℕ) : (n * 50 + 110) / (n + 1) = 55 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l1801_180136


namespace NUMINAMATH_CALUDE_walking_speed_problem_l1801_180145

/-- The walking speeds of two people meeting on a path --/
theorem walking_speed_problem (total_distance : ℝ) (time_diff : ℝ) (meeting_time : ℝ) (speed_diff : ℝ) :
  total_distance = 1200 →
  time_diff = 6 →
  meeting_time = 12 →
  speed_diff = 20 →
  ∃ (v : ℝ),
    v > 0 ∧
    (meeting_time + time_diff) * v + meeting_time * (v + speed_diff) = total_distance ∧
    v = 32 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l1801_180145


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1801_180167

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - 3 * x)}
def N : Set ℝ := {x | x^2 - 1 < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 1/3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1801_180167


namespace NUMINAMATH_CALUDE_full_price_revenue_l1801_180152

/-- Represents the concert ticket sales problem. -/
structure ConcertTickets where
  total_tickets : ℕ
  total_revenue : ℕ
  full_price : ℕ
  discounted_price : ℕ
  full_price_tickets : ℕ
  discounted_tickets : ℕ

/-- The revenue generated from full-price tickets is $4500. -/
theorem full_price_revenue (ct : ConcertTickets)
  (h1 : ct.total_tickets = 200)
  (h2 : ct.total_revenue = 4500)
  (h3 : ct.discounted_price = ct.full_price / 3)
  (h4 : ct.total_tickets = ct.full_price_tickets + ct.discounted_tickets)
  (h5 : ct.total_revenue = ct.full_price * ct.full_price_tickets + ct.discounted_price * ct.discounted_tickets) :
  ct.full_price * ct.full_price_tickets = 4500 := by
  sorry

end NUMINAMATH_CALUDE_full_price_revenue_l1801_180152


namespace NUMINAMATH_CALUDE_two_part_trip_average_speed_l1801_180186

/-- Calculates the average speed of a two-part trip -/
theorem two_part_trip_average_speed
  (total_distance : ℝ)
  (first_part_distance : ℝ)
  (first_part_speed : ℝ)
  (second_part_speed : ℝ)
  (h1 : total_distance = 450)
  (h2 : first_part_distance = 300)
  (h3 : first_part_speed = 20)
  (h4 : second_part_speed = 15)
  (h5 : first_part_distance < total_distance) :
  (total_distance) / ((first_part_distance / first_part_speed) + ((total_distance - first_part_distance) / second_part_speed)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_two_part_trip_average_speed_l1801_180186


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_proportions_l1801_180173

/-- Represents the number of people in each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Calculates the total population -/
def totalPopulation (p : Population) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- Represents the sample sizes for each age group -/
structure Sample :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Calculates the total sample size -/
def sampleSize (s : Sample) : ℕ :=
  s.elderly + s.middleAged + s.young

/-- Checks if the sample is proportional to the population -/
def isProportionalSample (p : Population) (s : Sample) : Prop :=
  s.elderly * totalPopulation p = p.elderly * sampleSize s ∧
  s.middleAged * totalPopulation p = p.middleAged * sampleSize s ∧
  s.young * totalPopulation p = p.young * sampleSize s

theorem stratified_sampling_correct_proportions 
  (p : Population) 
  (s : Sample) :
  p.elderly = 28 → 
  p.middleAged = 56 → 
  p.young = 84 → 
  sampleSize s = 36 → 
  isProportionalSample p s → 
  s.elderly = 6 ∧ s.middleAged = 12 ∧ s.young = 18 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_proportions_l1801_180173


namespace NUMINAMATH_CALUDE_circle_and_lines_properties_l1801_180161

-- Define the circle C
def circle_C (a : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + y^2 = 4}

-- Define the tangent line
def tangent_line := {(x, y) : ℝ × ℝ | 3*x - 4*y + 4 = 0}

-- Define the intersecting line l
def line_l (k : ℝ) := {(x, y) : ℝ × ℝ | y = k*x - 3}

-- Main theorem
theorem circle_and_lines_properties :
  ∃ (a : ℝ), a > 0 ∧
  (∀ (p : ℝ × ℝ), p ∈ circle_C a → p ∉ tangent_line) ∧
  (∃ (q : ℝ × ℝ), q ∈ circle_C a ∧ q ∈ tangent_line) →
  (a = 2) ∧
  (∀ (k x₁ y₁ x₂ y₂ : ℝ),
    (x₁, y₁) ∈ circle_C 2 ∧ (x₁, y₁) ∈ line_l k ∧
    (x₂, y₂) ∈ circle_C 2 ∧ (x₂, y₂) ∈ line_l k ∧
    (x₁, y₁) ≠ (x₂, y₂) →
    (k = 3 → x₁ * x₂ + y₁ * y₂ = -9/5) ∧
    (x₁ * x₂ + y₁ * y₂ = 8 → k = (-3 + Real.sqrt 29) / 4)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_lines_properties_l1801_180161


namespace NUMINAMATH_CALUDE_tan_function_value_l1801_180112

theorem tan_function_value (f : ℝ → ℝ) :
  (∀ x, f x = Real.tan (2 * x + π / 3)) →
  f (25 * π / 6) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_function_value_l1801_180112


namespace NUMINAMATH_CALUDE_unique_integral_solution_l1801_180133

theorem unique_integral_solution (x y z n : ℤ) 
  (eq1 : x * y + y * z + z * x = 3 * n^2 - 1)
  (eq2 : x + y + z = 3 * n)
  (h1 : x ≥ y)
  (h2 : y ≥ z) :
  x = n + 1 ∧ y = n ∧ z = n - 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_integral_solution_l1801_180133


namespace NUMINAMATH_CALUDE_remaining_money_is_16000_l1801_180160

def salary : ℚ := 160000

def food_fraction : ℚ := 1/5
def rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5

def remaining_money : ℚ := salary - (food_fraction * salary + rent_fraction * salary + clothes_fraction * salary)

theorem remaining_money_is_16000 : remaining_money = 16000 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_is_16000_l1801_180160


namespace NUMINAMATH_CALUDE_king_will_be_checked_l1801_180151

/-- Represents a chess piece -/
inductive Piece
| King
| Rook

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents the state of the chessboard -/
structure ChessboardState :=
  (kingPos : Position)
  (rookPositions : List Position)

/-- Represents a move in the game -/
inductive Move
| KingMove (newPos : Position)
| RookMove (oldPos : Position) (newPos : Position)

/-- The game ends when the king is in check or reaches the top-right corner -/
def gameEnded (state : ChessboardState) : Prop :=
  (state.kingPos.x = 20 ∧ state.kingPos.y = 20) ∨
  state.rookPositions.any (λ rookPos => rookPos.x = state.kingPos.x ∨ rookPos.y = state.kingPos.y)

/-- A valid game sequence -/
def ValidGameSequence : List Move → Prop :=
  sorry

/-- The theorem to be proved -/
theorem king_will_be_checked
  (initialState : ChessboardState)
  (h1 : initialState.kingPos = ⟨1, 1⟩)
  (h2 : initialState.rookPositions.length = 10)
  (h3 : ∀ pos ∈ initialState.rookPositions, pos.x ≤ 20 ∧ pos.y ≤ 20) :
  ∀ (moves : List Move), ValidGameSequence moves →
    ∃ (n : Nat), let finalState := (moves.take n).foldl (λ s m => sorry) initialState
                 gameEnded finalState :=
sorry

end NUMINAMATH_CALUDE_king_will_be_checked_l1801_180151


namespace NUMINAMATH_CALUDE_negative_x_times_three_minus_x_l1801_180196

theorem negative_x_times_three_minus_x (x : ℝ) : -x * (3 - x) = -3*x + x^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_times_three_minus_x_l1801_180196


namespace NUMINAMATH_CALUDE_min_value_xyz_l1801_180124

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 18) :
  x + 3 * y + 6 * z ≥ 3 * (2 * Real.sqrt 6 + 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_xyz_l1801_180124


namespace NUMINAMATH_CALUDE_john_park_distance_l1801_180146

/-- John's journey to the park -/
theorem john_park_distance (speed : ℝ) (time_minutes : ℝ) (h1 : speed = 9) (h2 : time_minutes = 2) :
  speed * (time_minutes / 60) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_john_park_distance_l1801_180146


namespace NUMINAMATH_CALUDE_sin_A_range_l1801_180169

theorem sin_A_range (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- positive angles
  A + B + C = π ∧ -- sum of angles in a triangle
  C = π / 3 ∧ 
  a = 6 ∧ 
  1 ≤ b ∧ b ≤ 4 ∧
  a / (Real.sin A) = b / (Real.sin B) ∧ -- sine rule
  a / (Real.sin A) = c / (Real.sin C) ∧ -- sine rule
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) -- cosine rule
  →
  3 * Real.sqrt 93 / 31 ≤ Real.sin A ∧ Real.sin A ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_sin_A_range_l1801_180169


namespace NUMINAMATH_CALUDE_book_reading_time_l1801_180103

/-- Calculates the remaining reading time for a book -/
def remaining_reading_time (total_pages : ℕ) (pages_per_hour : ℕ) (monday_hours : ℕ) (tuesday_hours : ℚ) : ℚ :=
  let pages_read := (monday_hours : ℚ) * pages_per_hour + tuesday_hours * pages_per_hour
  let pages_left := (total_pages : ℚ) - pages_read
  pages_left / pages_per_hour

theorem book_reading_time :
  remaining_reading_time 248 16 3 (13/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l1801_180103
