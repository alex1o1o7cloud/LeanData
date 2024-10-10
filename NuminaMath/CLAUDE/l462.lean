import Mathlib

namespace cone_lateral_surface_area_l462_46235

theorem cone_lateral_surface_area 
  (r : ℝ) 
  (h : ℝ) 
  (lateral_surface_area : ℝ → ℝ → ℝ) :
  r = 2 →
  h = 4 * Real.sqrt 2 →
  lateral_surface_area r h = 12 * Real.pi :=
by sorry

end cone_lateral_surface_area_l462_46235


namespace inner_square_probability_10x10_l462_46299

/-- Represents a square checkerboard -/
structure Checkerboard where
  size : ℕ
  total_squares : ℕ
  perimeter_squares : ℕ
  inner_squares : ℕ

/-- Creates a 10x10 checkerboard -/
def create_10x10_board : Checkerboard :=
  { size := 10,
    total_squares := 100,
    perimeter_squares := 36,
    inner_squares := 64 }

/-- Calculates the probability of choosing an inner square -/
def inner_square_probability (board : Checkerboard) : ℚ :=
  board.inner_squares / board.total_squares

theorem inner_square_probability_10x10 :
  inner_square_probability create_10x10_board = 16 / 25 := by
  sorry

end inner_square_probability_10x10_l462_46299


namespace x_cubed_coefficient_l462_46240

/-- The coefficient of x^3 in the expansion of (3x^3 + 2x^2 + 3x + 4)(5x^2 + 7x + 6) is 47 -/
theorem x_cubed_coefficient : 
  let p₁ : Polynomial ℤ := 3 * X^3 + 2 * X^2 + 3 * X + 4
  let p₂ : Polynomial ℤ := 5 * X^2 + 7 * X + 6
  (p₁ * p₂).coeff 3 = 47 := by
sorry

end x_cubed_coefficient_l462_46240


namespace projectile_max_height_l462_46232

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 + 50 * t + 10

-- Theorem statement
theorem projectile_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 41.25 := by
  sorry

end projectile_max_height_l462_46232


namespace circle_equations_l462_46276

-- Define the circle N
def circle_N (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 10

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 5)^2 = 10

-- Define the trajectory of midpoint M
def trajectory_M (x y : ℝ) : Prop := (x - 5/2)^2 + (y - 2)^2 = 5/2

-- Define points A, B, and C
def point_A : ℝ × ℝ := (3, 1)
def point_B : ℝ × ℝ := (-1, 3)
def point_C : ℝ × ℝ := (3, 0)

-- Define the line that contains the center of circle N
def center_line (x y : ℝ) : Prop := 3*x - y - 2 = 0

-- Define a point D on circle N
def point_D (x y : ℝ) : Prop := circle_N x y

-- Define the midpoint M of segment CD
def midpoint_M (x y x_D y_D : ℝ) : Prop := x = (x_D + 3)/2 ∧ y = y_D/2

theorem circle_equations :
  (∀ x y, circle_N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) ∧
  (∀ x y, symmetric_circle x y ↔ (x - 1)^2 + (y - 5)^2 = 10) ∧
  (∀ x y, (∃ x_D y_D, point_D x_D y_D ∧ midpoint_M x y x_D y_D) → trajectory_M x y) :=
sorry

end circle_equations_l462_46276


namespace parabola_vertex_l462_46290

/-- The parabola defined by y = x^2 - 2 -/
def parabola (x : ℝ) : ℝ := x^2 - 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (0, -2)

/-- Theorem: The vertex of the parabola y = x^2 - 2 is at the point (0, -2) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 := by
sorry

end parabola_vertex_l462_46290


namespace tan_beta_plus_pi_fourth_l462_46217

theorem tan_beta_plus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3/5) 
  (h2 : Real.tan α = 1/3) : 
  Real.tan (β + π/4) = 11/7 := by sorry

end tan_beta_plus_pi_fourth_l462_46217


namespace compare_sqrt_expressions_l462_46215

theorem compare_sqrt_expressions : 7 * Real.sqrt 2 < 3 * Real.sqrt 11 := by
  sorry

end compare_sqrt_expressions_l462_46215


namespace fourth_grade_students_left_l462_46292

/-- The number of students who left during the year -/
def students_left (initial : ℕ) (new : ℕ) (final : ℕ) : ℕ :=
  initial + new - final

theorem fourth_grade_students_left : students_left 11 42 47 = 6 := by
  sorry

end fourth_grade_students_left_l462_46292


namespace vacuum_time_solution_l462_46213

def chores_problem (vacuum_time : ℝ) : Prop :=
  let other_chores_time := 3 * vacuum_time
  vacuum_time + other_chores_time = 12

theorem vacuum_time_solution :
  ∃ (t : ℝ), chores_problem t ∧ t = 3 :=
sorry

end vacuum_time_solution_l462_46213


namespace total_students_l462_46223

theorem total_students (N : ℕ) 
  (provincial_total : ℕ) (provincial_sample : ℕ)
  (experimental_sample : ℕ) (regular_sample : ℕ) (sino_canadian_sample : ℕ)
  (h1 : provincial_total = 96)
  (h2 : provincial_sample = 12)
  (h3 : experimental_sample = 21)
  (h4 : regular_sample = 25)
  (h5 : sino_canadian_sample = 43)
  (h6 : N * provincial_sample = provincial_total * (provincial_sample + experimental_sample + regular_sample + sino_canadian_sample)) :
  N = 808 := by
sorry

end total_students_l462_46223


namespace frog_border_probability_l462_46231

/-- Represents a position on the 4x4 grid -/
structure Position where
  x : Fin 4
  y : Fin 4

/-- Represents the possible directions of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines the grid and movement rules -/
def Grid :=
  { pos : Position // true }

/-- Determines if a position is on the border of the grid -/
def is_border (pos : Position) : Bool :=
  pos.x = 0 ∨ pos.x = 3 ∨ pos.y = 0 ∨ pos.y = 3

/-- Calculates the next position after a hop in a given direction -/
def next_position (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.Up => ⟨(pos.x + 1) % 4, pos.y⟩
  | Direction.Down => ⟨(pos.x + 3) % 4, pos.y⟩
  | Direction.Left => ⟨pos.x, (pos.y + 3) % 4⟩
  | Direction.Right => ⟨pos.x, (pos.y + 1) % 4⟩

/-- Calculates the probability of reaching the border within n hops -/
def border_probability (start : Position) (n : Nat) : Rat :=
  sorry

/-- The main theorem to be proved -/
theorem frog_border_probability :
  border_probability ⟨1, 1⟩ 3 = 39 / 64 :=
sorry

end frog_border_probability_l462_46231


namespace quadratic_equations_solutions_l462_46293

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 3 ∧ x₂ = 3 - Real.sqrt 3 ∧
    x₁^2 - 6*x₁ + 6 = 0 ∧ x₂^2 - 6*x₂ + 6 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 5 ∧
    (x₁ - 1) * (x₁ - 3) = 8 ∧ (x₂ - 1) * (x₂ - 3) = 8) :=
by sorry

end quadratic_equations_solutions_l462_46293


namespace emily_score_proof_l462_46272

def emily_scores : List ℝ := [88, 92, 85, 90, 97]
def target_mean : ℝ := 91
def sixth_score : ℝ := 94

theorem emily_score_proof :
  let all_scores := emily_scores ++ [sixth_score]
  (all_scores.sum / all_scores.length : ℝ) = target_mean := by
  sorry

end emily_score_proof_l462_46272


namespace min_sum_absolute_values_l462_46242

theorem min_sum_absolute_values :
  ∀ x : ℝ, |x + 3| + |x + 6| + |x + 7| ≥ 4 ∧
  ∃ x : ℝ, |x + 3| + |x + 6| + |x + 7| = 4 :=
sorry

end min_sum_absolute_values_l462_46242


namespace richards_average_touchdowns_l462_46230

def archie_record : ℕ := 89
def total_games : ℕ := 16
def richards_games : ℕ := 14
def remaining_games : ℕ := 2
def remaining_avg : ℕ := 3

theorem richards_average_touchdowns :
  let total_touchdowns := archie_record + 1
  let remaining_touchdowns := remaining_games * remaining_avg
  let richards_touchdowns := total_touchdowns - remaining_touchdowns
  (richards_touchdowns : ℚ) / richards_games = 6 := by sorry

end richards_average_touchdowns_l462_46230


namespace remaining_money_l462_46277

def base_8_to_10 (n : ℕ) : ℕ := 
  4 * 8^3 + 4 * 8^2 + 4 * 8^1 + 4 * 8^0

def savings : ℕ := base_8_to_10 4444

def ticket_cost : ℕ := 1000

theorem remaining_money : 
  savings - ticket_cost = 1340 := by sorry

end remaining_money_l462_46277


namespace novels_per_month_l462_46205

/-- Given that each novel has 200 pages and 9600 pages of novels are read in a year,
    prove that 4 novels are read in a month. -/
theorem novels_per_month :
  ∀ (pages_per_novel : ℕ) (pages_per_year : ℕ) (months_per_year : ℕ),
    pages_per_novel = 200 →
    pages_per_year = 9600 →
    months_per_year = 12 →
    (pages_per_year / pages_per_novel) / months_per_year = 4 :=
by
  sorry

end novels_per_month_l462_46205


namespace probability_of_red_ball_l462_46201

-- Define the number of red and black balls
def num_red_balls : ℕ := 7
def num_black_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_black_balls

-- Define the probability of drawing a red ball
def prob_red_ball : ℚ := num_red_balls / total_balls

-- Theorem statement
theorem probability_of_red_ball :
  prob_red_ball = 7 / 10 := by sorry

end probability_of_red_ball_l462_46201


namespace power_sum_difference_l462_46245

theorem power_sum_difference : 3^(2+3+4) - (3^2 + 3^3 + 3^4 + 3^5) = 19323 := by
  sorry

end power_sum_difference_l462_46245


namespace total_books_read_is_72sc_l462_46263

/-- Calculates the total number of books read by a school's student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  let books_per_month : ℕ := 6
  let months_per_year : ℕ := 12
  let books_per_student_per_year : ℕ := books_per_month * months_per_year
  let total_students : ℕ := c * s
  books_per_student_per_year * total_students

/-- Theorem stating that the total number of books read by the entire student body in one year is 72sc -/
theorem total_books_read_is_72sc (c s : ℕ) : total_books_read c s = 72 * c * s := by
  sorry

end total_books_read_is_72sc_l462_46263


namespace june_math_book_price_l462_46271

/-- The price of a math book that satisfies June's shopping constraints -/
def math_book_price : ℝ → Prop := λ x =>
  let total_budget : ℝ := 500
  let num_math_books : ℕ := 4
  let num_science_books : ℕ := num_math_books + 6
  let science_book_price : ℝ := 10
  let num_art_books : ℕ := 2 * num_math_books
  let art_book_price : ℝ := 20
  let music_books_cost : ℝ := 160
  (num_math_books : ℝ) * x + 
  (num_science_books : ℝ) * science_book_price + 
  (num_art_books : ℝ) * art_book_price + 
  music_books_cost = total_budget

theorem june_math_book_price : ∃ x : ℝ, math_book_price x ∧ x = 20 := by
  sorry

end june_math_book_price_l462_46271


namespace gum_given_by_steve_l462_46209

theorem gum_given_by_steve (initial_gum : ℕ) (final_gum : ℕ) 
  (h1 : initial_gum = 38) (h2 : final_gum = 54) :
  final_gum - initial_gum = 16 := by
  sorry

end gum_given_by_steve_l462_46209


namespace train_speed_fraction_l462_46212

/-- Given a train journey where:
  1. The train reached its destination in 8 hours at a certain fraction of its own speed.
  2. If the train had run at its full speed, it would have taken 4 hours less.
  This theorem proves that the fraction of the train's own speed at which it was running is 1/2. -/
theorem train_speed_fraction (full_speed : ℝ) (fraction : ℝ) 
  (h1 : fraction * full_speed * 8 = full_speed * 4) : fraction = 1 / 2 := by
  sorry

end train_speed_fraction_l462_46212


namespace onion_harvest_weight_l462_46226

theorem onion_harvest_weight (initial_bags : ℕ) (trips : ℕ) (bag_weight : ℕ) : 
  initial_bags = 10 → trips = 20 → bag_weight = 50 →
  (initial_bags * ((2 ^ trips) - 1)) * bag_weight = 524287500 := by
  sorry

end onion_harvest_weight_l462_46226


namespace divisible_by_eight_count_l462_46260

theorem divisible_by_eight_count : 
  (Finset.filter (fun n => n % 8 = 0) (Finset.Icc 200 400)).card = 26 := by
  sorry

end divisible_by_eight_count_l462_46260


namespace distance_between_stations_l462_46237

def passenger_train_speed : ℚ := 1/2
def express_train_speed : ℚ := 1
def catch_up_distance : ℚ := 244/9

theorem distance_between_stations (x : ℚ) : 
  (x/3 + 4*(x/3 - catch_up_distance)/3 = x - catch_up_distance) → x = 528 := by
  sorry

end distance_between_stations_l462_46237


namespace five_digit_cube_root_l462_46214

theorem five_digit_cube_root (n : ℕ) : 
  (10000 ≤ n ∧ n < 100000) →  -- n is a five-digit number
  (n % 10 = 3) →              -- n ends in 3
  (∃ k : ℕ, k^3 = n) →        -- n has an integer cube root
  (n = 19683 ∨ n = 50653) :=  -- n is either 19683 or 50653
by sorry

end five_digit_cube_root_l462_46214


namespace sequence_property_l462_46234

def sequence_a : ℕ → ℕ
  | 0 => 1
  | n + 1 => sequence_a n ^ 2 + n * sequence_a n

def S : Set ℕ := {p : ℕ | Nat.Prime p ∧ ∃ i, p ∣ sequence_a i}

theorem sequence_property :
  (Set.Infinite S) ∧ (S ≠ {p : ℕ | Nat.Prime p}) := by
  sorry

end sequence_property_l462_46234


namespace problem_1_problem_2_problem_3_problem_4_l462_46267

-- Problem 1
theorem problem_1 : (-6) + (-13) = -19 := by sorry

-- Problem 2
theorem problem_2 : (3 : ℚ) / 5 + (-3 / 4) = -3 / 20 := by sorry

-- Problem 3
theorem problem_3 : (4.7 : ℝ) + (-0.8) + 5.3 + (-8.2) = 1 := by sorry

-- Problem 4
theorem problem_4 : (-1 : ℚ) / 6 + 1 / 3 + (-1 / 12) = 1 / 12 := by sorry

end problem_1_problem_2_problem_3_problem_4_l462_46267


namespace factorization_a_squared_minus_ab_l462_46246

theorem factorization_a_squared_minus_ab (a b : ℝ) : a^2 - a*b = a*(a - b) := by
  sorry

end factorization_a_squared_minus_ab_l462_46246


namespace amber_bronze_selection_l462_46297

/-- Represents a cell in the grid -/
inductive Cell
| Amber
| Bronze

/-- Represents the grid -/
def Grid (a b : ℕ) := Fin (a + b + 1) → Fin (a + b + 1) → Cell

/-- Counts the number of amber cells in the grid -/
def countAmber (g : Grid a b) : ℕ := sorry

/-- Counts the number of bronze cells in the grid -/
def countBronze (g : Grid a b) : ℕ := sorry

/-- Represents a selection of cells -/
def Selection (a b : ℕ) := Fin (a + b) → Fin (a + b + 1) × Fin (a + b + 1)

/-- Checks if a selection is valid (no two cells in the same row or column) -/
def isValidSelection (s : Selection a b) : Prop := sorry

/-- Counts the number of amber cells in a selection -/
def countAmberInSelection (g : Grid a b) (s : Selection a b) : ℕ := sorry

/-- Counts the number of bronze cells in a selection -/
def countBronzeInSelection (g : Grid a b) (s : Selection a b) : ℕ := sorry

theorem amber_bronze_selection (a b : ℕ) (g : Grid a b) 
  (ha : a > 0) (hb : b > 0)
  (hamber : countAmber g ≥ a^2 + a*b - b)
  (hbronze : countBronze g ≥ b^2 + a*b - a) :
  ∃ (s : Selection a b), 
    isValidSelection s ∧ 
    countAmberInSelection g s = a ∧ 
    countBronzeInSelection g s = b := by
  sorry

end amber_bronze_selection_l462_46297


namespace roots_on_circle_l462_46204

open Complex

theorem roots_on_circle : ∃ (r : ℝ), r = 2 * Real.sqrt 3 / 3 ∧
  ∀ (z : ℂ), (z - 2) ^ 6 = 64 * z ^ 6 →
    ∃ (c : ℂ), abs (z - c) = r :=
by sorry

end roots_on_circle_l462_46204


namespace max_display_sum_l462_46288

def DigitalWatch := ℕ × ℕ

def valid_time (t : DigitalWatch) : Prop :=
  1 ≤ t.1 ∧ t.1 ≤ 12 ∧ t.2 < 60

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def display_sum (t : DigitalWatch) : ℕ :=
  digit_sum t.1 + digit_sum t.2

theorem max_display_sum :
  ∃ (t : DigitalWatch), valid_time t ∧
    ∀ (t' : DigitalWatch), valid_time t' →
      display_sum t' ≤ display_sum t ∧
      display_sum t = 23 := by
  sorry

end max_display_sum_l462_46288


namespace remainder_problem_l462_46281

theorem remainder_problem (m : ℤ) (h : m % 288 = 47) : m % 24 = 23 := by
  sorry

end remainder_problem_l462_46281


namespace student_grade_problem_l462_46285

theorem student_grade_problem (grade_history grade_third : ℝ) 
  (h1 : grade_history = 84)
  (h2 : grade_third = 69)
  (h3 : (grade_math + grade_history + grade_third) / 3 = 75) :
  grade_math = 72 := by
  sorry

end student_grade_problem_l462_46285


namespace polynomial_simplification_l462_46221

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  -3 + 23*x - x^2 + 2*x^3 := by
  sorry

end polynomial_simplification_l462_46221


namespace vector_proof_l462_46225

/-- Given two planar vectors a and b, with a parallel to b, and a linear combination
    of these vectors with a third vector c equal to the zero vector,
    prove that c is equal to (-7, 14). -/
theorem vector_proof (a b c : ℝ × ℝ) (m : ℝ) : 
  a = (1, -2) →
  b = (2, m) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  3 • a + 2 • b + c = (0, 0) →
  c = (-7, 14) := by
  sorry

end vector_proof_l462_46225


namespace tree_planting_probability_l462_46268

def num_cedar : ℕ := 4
def num_pine : ℕ := 3
def num_alder : ℕ := 6

def total_trees : ℕ := num_cedar + num_pine + num_alder

def probability_no_adjacent_alders : ℚ := 2 / 4290

theorem tree_planting_probability :
  let total_arrangements : ℕ := (Nat.factorial total_trees) / 
    (Nat.factorial num_cedar * Nat.factorial num_pine * Nat.factorial num_alder)
  let valid_arrangements : ℕ := Nat.choose (num_cedar + num_pine + 1) num_alder * 
    (Nat.factorial (num_cedar + num_pine) / (Nat.factorial num_cedar * Nat.factorial num_pine))
  (valid_arrangements : ℚ) / total_arrangements = probability_no_adjacent_alders :=
sorry

end tree_planting_probability_l462_46268


namespace jake_final_bitcoin_count_l462_46241

def bitcoin_transactions (initial : ℕ) (investment : ℕ) (donation1 : ℕ) (debt : ℕ) (donation2 : ℕ) : ℕ :=
  let after_investment := initial - investment + 2 * investment
  let after_donation1 := after_investment - donation1
  let after_sharing := after_donation1 - (after_donation1 / 2)
  let after_debt_collection := after_sharing + debt
  let after_quadrupling := 4 * after_debt_collection
  after_quadrupling - donation2

theorem jake_final_bitcoin_count :
  bitcoin_transactions 120 40 25 5 15 = 277 := by
  sorry

end jake_final_bitcoin_count_l462_46241


namespace opposite_of_2021_l462_46218

theorem opposite_of_2021 : ∃ x : ℝ, x + 2021 = 0 ∧ x = -2021 := by sorry

end opposite_of_2021_l462_46218


namespace largest_n_binomial_sum_l462_46282

theorem largest_n_binomial_sum : ∃ (n : ℕ), (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) ∧ 
  (∀ (m : ℕ), Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 m → m ≤ n) ∧ n = 7 :=
by sorry

end largest_n_binomial_sum_l462_46282


namespace complete_square_existence_l462_46243

theorem complete_square_existence :
  ∃ (k : ℤ) (a : ℝ), ∀ z : ℝ, z^2 - 6*z + 17 = (z + a)^2 + k := by
  sorry

end complete_square_existence_l462_46243


namespace set_intersection_example_l462_46251

theorem set_intersection_example : 
  let A : Set Int := {-1, 1}
  let B : Set Int := {-3, 0, 1}
  A ∩ B = {1} := by
sorry

end set_intersection_example_l462_46251


namespace cubic_equation_one_real_root_l462_46264

theorem cubic_equation_one_real_root (a : ℝ) : 
  (∃! x : ℝ, x^3 - a*x^2 - 2*a*x + a^2 - 1 = 0) → a < 3/4 :=
by sorry

end cubic_equation_one_real_root_l462_46264


namespace power_division_sum_product_difference_l462_46289

theorem power_division_sum_product_difference (a b c d e f g : ℤ) :
  a = -4 ∧ b = 4 ∧ c = 2 ∧ d = 3 ∧ e = 7 →
  a^6 / b^4 + c^5 * d - e^2 = 63 := by
  sorry

end power_division_sum_product_difference_l462_46289


namespace difference_of_squares_factorization_l462_46208

theorem difference_of_squares_factorization (m : ℤ) : m^2 - 1 = (m + 1) * (m - 1) := by
  sorry

end difference_of_squares_factorization_l462_46208


namespace min_value_a_l462_46206

theorem min_value_a (a : ℝ) : 
  (∀ x > a, 2 * x + 2 / (x - a) ≥ 7) ↔ a ≥ 3/2 := by
  sorry

end min_value_a_l462_46206


namespace uneven_picture_distribution_l462_46238

theorem uneven_picture_distribution (total_pictures : Nat) (num_albums : Nat) : 
  total_pictures = 101 → num_albums = 7 → ¬∃ (pics_per_album : Nat), total_pictures = num_albums * pics_per_album :=
by
  sorry

end uneven_picture_distribution_l462_46238


namespace sticker_distribution_l462_46219

/-- The number of ways to distribute n identical objects among k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 9 identical stickers among 3 distinct sheets of paper -/
theorem sticker_distribution : distribute 9 3 = 55 := by sorry

end sticker_distribution_l462_46219


namespace representative_distribution_l462_46210

/-- The number of ways to distribute n items into k groups with at least one item in each group -/
def distribute_with_minimum (n k : ℕ) : ℕ := sorry

/-- The number of classes from which representatives are selected -/
def num_classes : ℕ := 4

/-- The total number of student representatives to be selected -/
def total_representatives : ℕ := 6

/-- Theorem stating that the number of ways to distribute 6 representatives among 4 classes,
    with at least one representative in each class, is equal to 10 -/
theorem representative_distribution :
  distribute_with_minimum total_representatives num_classes = 10 := by sorry

end representative_distribution_l462_46210


namespace triangle_properties_l462_46295

theorem triangle_properties (a b c A B C : ℝ) (h1 : a * Real.sin A - c * Real.sin C = (a - b) * Real.sin B)
  (h2 : c = 2 * Real.sqrt 3) (h3 : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) :
  C = π / 3 ∧ a + b + c = 6 + 2 * Real.sqrt 3 := by
  sorry

end triangle_properties_l462_46295


namespace min_team_a_size_l462_46283

theorem min_team_a_size (a b : ℕ) : 
  (∃ c : ℕ, 2 * (a - 90) = b + 90 ∧ a + c = 6 * (b - c)) →
  a ≥ 153 :=
sorry

end min_team_a_size_l462_46283


namespace inequality_solution_l462_46250

noncomputable section

variables (a x : ℝ)

def inequality := (a * (x - 1)) / (x - 2) > 1

def solution : Prop :=
  (0 < a ∧ a < 1 → 2 < x ∧ x < (a - 2) / (a - 1)) ∧
  (a = 1 → x > 2) ∧
  (a > 1 → x > 2 ∨ x < (a - 2) / (a - 1))

theorem inequality_solution (h : a > 0) : inequality a x ↔ solution a x := by sorry

end

end inequality_solution_l462_46250


namespace seven_balls_four_boxes_l462_46294

/-- The number of ways to partition n into at most k parts, where the order doesn't matter -/
def partition_count (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 11 ways to partition 7 into at most 4 parts -/
theorem seven_balls_four_boxes : partition_count 7 4 = 11 := by sorry

end seven_balls_four_boxes_l462_46294


namespace solve_for_e_l462_46273

-- Define the functions p and q
def p (x : ℝ) : ℝ := 5 * x - 17
def q (e : ℝ) (x : ℝ) : ℝ := 4 * x - e

-- State the theorem
theorem solve_for_e : 
  ∀ e : ℝ, p (q e 3) = 23 → e = 4 := by
  sorry

end solve_for_e_l462_46273


namespace hyperbola_eccentricity_sqrt_5_l462_46222

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The left focus of the hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

theorem hyperbola_eccentricity_sqrt_5 
  {a b c : ℝ} (h : Hyperbola a b) 
  (focus_c : left_focus h = (-c, 0))
  (point_P : c^2/a^2 - 4 = 1) : 
  eccentricity h = Real.sqrt 5 := by sorry

end hyperbola_eccentricity_sqrt_5_l462_46222


namespace counterexample_ten_l462_46266

theorem counterexample_ten : 
  ¬(¬(Nat.Prime 10) → Nat.Prime (10 + 2)) :=
by sorry

end counterexample_ten_l462_46266


namespace triangle_perimeter_bound_l462_46228

/-- A convex polygon -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  convex : Convex ℝ vertices
  finite : Finite vertices

/-- The perimeter of a polygon -/
def perimeter (p : ConvexPolygon) : ℝ := sorry

/-- A triangle formed by three vertices of a polygon -/
def triangle_of_polygon (p : ConvexPolygon) : Set (ConvexPolygon) := sorry

theorem triangle_perimeter_bound (G : ConvexPolygon) :
  ∃ T ∈ triangle_of_polygon G, perimeter T ≥ 0.7 * perimeter G := by sorry

end triangle_perimeter_bound_l462_46228


namespace no_solution_for_four_l462_46259

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def three_digit_number (x y : ℕ) : ℕ :=
  100 * x + 30 + y

theorem no_solution_for_four :
  ∀ y : ℕ, y < 10 →
    ¬(is_divisible_by_11 (three_digit_number 4 y)) ∧
    (∀ x : ℕ, x < 10 → x ≠ 4 →
      ∃ y : ℕ, y < 10 ∧ is_divisible_by_11 (three_digit_number x y)) :=
by sorry

end no_solution_for_four_l462_46259


namespace exist_non_adjacent_non_sharing_l462_46265

/-- A simple graph with 17 vertices where each vertex has degree 4. -/
structure Graph17Deg4 where
  vertices : Finset (Fin 17)
  edges : Finset (Fin 17 × Fin 17)
  vertex_count : vertices.card = 17
  degree_4 : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 4

/-- Two vertices are adjacent if there's an edge between them. -/
def adjacent (G : Graph17Deg4) (u v : Fin 17) : Prop :=
  (u, v) ∈ G.edges ∨ (v, u) ∈ G.edges

/-- Two vertices share a common neighbor if there exists a vertex adjacent to both. -/
def share_neighbor (G : Graph17Deg4) (u v : Fin 17) : Prop :=
  ∃ w : Fin 17, w ∈ G.vertices ∧ adjacent G u w ∧ adjacent G v w

/-- There exist two vertices that are neither adjacent nor share a common neighbor. -/
theorem exist_non_adjacent_non_sharing (G : Graph17Deg4) :
  ∃ u v : Fin 17, u ∈ G.vertices ∧ v ∈ G.vertices ∧ u ≠ v ∧
    ¬(adjacent G u v) ∧ ¬(share_neighbor G u v) := by
  sorry

end exist_non_adjacent_non_sharing_l462_46265


namespace midnight_temperature_l462_46256

/-- Given the temperature changes throughout the day in a certain city, 
    prove that the temperature at midnight is 24°C. -/
theorem midnight_temperature 
  (morning_temp : ℝ)
  (afternoon_increase : ℝ)
  (midnight_decrease : ℝ)
  (h1 : morning_temp = 30)
  (h2 : afternoon_increase = 1)
  (h3 : midnight_decrease = 7) :
  morning_temp + afternoon_increase - midnight_decrease = 24 :=
by sorry

end midnight_temperature_l462_46256


namespace thomas_lost_pieces_l462_46200

theorem thomas_lost_pieces (total_start : Nat) (player_start : Nat) (audrey_lost : Nat) (total_end : Nat) :
  total_start = 32 →
  player_start = 16 →
  audrey_lost = 6 →
  total_end = 21 →
  player_start - (total_end - (player_start - audrey_lost)) = 5 := by
  sorry

end thomas_lost_pieces_l462_46200


namespace regular_polygon_108_degrees_has_5_sides_l462_46291

/-- A regular polygon with interior angles measuring 108 degrees has 5 sides. -/
theorem regular_polygon_108_degrees_has_5_sides :
  ∀ n : ℕ,
  n ≥ 3 →
  (180 * (n - 2) : ℝ) = (108 * n : ℝ) →
  n = 5 :=
by
  sorry

end regular_polygon_108_degrees_has_5_sides_l462_46291


namespace line_equation_through_points_l462_46253

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = x₁ + t * (x₂ - x₁) ∧ p.2 = y₁ + t * (y₂ - y₁)}

-- Define the general form of a line equation
def general_line_equation (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

theorem line_equation_through_points :
  line_through_points 2 5 0 3 = general_line_equation 1 (-1) 3 := by
  sorry

end line_equation_through_points_l462_46253


namespace inverse_proposition_geometric_sequence_l462_46269

theorem inverse_proposition_geometric_sequence (a b c : ℝ) :
  (∀ {a b c : ℝ}, (∃ r : ℝ, b = a * r ∧ c = b * r) → b^2 = a * c) →
  (b^2 = a * c → ∃ r : ℝ, b = a * r ∧ c = b * r) :=
by sorry

end inverse_proposition_geometric_sequence_l462_46269


namespace six_balls_three_boxes_l462_46286

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ballsInBoxes (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 82 ways to put 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : ballsInBoxes 6 3 = 82 := by
  sorry

end six_balls_three_boxes_l462_46286


namespace optimal_distribution_l462_46261

/-- Represents the production process for assembling products -/
structure ProductionProcess where
  totalWorkers : ℕ
  productsToAssemble : ℕ
  paintTime : ℕ
  dryTime : ℕ
  assemblyTime : ℕ

/-- Represents the distribution of workers -/
structure WorkerDistribution where
  painters : ℕ
  assemblers : ℕ

/-- Calculates the production time for a given worker distribution -/
def productionTime (process : ProductionProcess) (dist : WorkerDistribution) : ℕ :=
  sorry

/-- Checks if a worker distribution is valid for the given process -/
def isValidDistribution (process : ProductionProcess) (dist : WorkerDistribution) : Prop :=
  dist.painters + dist.assemblers ≤ process.totalWorkers

/-- Theorem stating the optimal worker distribution for the given process -/
theorem optimal_distribution (process : ProductionProcess) 
  (h1 : process.totalWorkers = 10)
  (h2 : process.productsToAssemble = 50)
  (h3 : process.paintTime = 10)
  (h4 : process.dryTime = 5)
  (h5 : process.assemblyTime = 20) :
  ∃ (optDist : WorkerDistribution), 
    optDist.painters = 3 ∧ 
    optDist.assemblers = 6 ∧
    isValidDistribution process optDist ∧
    ∀ (dist : WorkerDistribution), 
      isValidDistribution process dist → 
      productionTime process optDist ≤ productionTime process dist :=
  sorry

end optimal_distribution_l462_46261


namespace second_term_of_sequence_l462_46257

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : Prop :=
  ∃ n : ℕ, n > 1 ∧ aₙ = a₁ + (n - 1) * d

theorem second_term_of_sequence (a₁ a₂ aₙ d : ℕ) :
  a₁ = 34 → d = 11 → aₙ = 89 → arithmetic_sequence a₁ d aₙ → a₂ = 45 := by
  sorry

end second_term_of_sequence_l462_46257


namespace consecutive_pages_sum_l462_46278

theorem consecutive_pages_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 285 := by
  sorry

end consecutive_pages_sum_l462_46278


namespace problem_1_problem_2_l462_46249

-- Problem 1
theorem problem_1 : 
  |(-2 : ℝ)| + Real.sqrt 2 * Real.tan (45 * π / 180) - Real.sqrt 8 - (2023 - Real.pi) ^ (0 : ℝ) = 1 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : 
  ∀ x : ℝ, x ≠ 2 → ((2 * x - 3) / (x - 2) - 1 / (2 - x) = 1 ↔ x = 0) := by
  sorry

end problem_1_problem_2_l462_46249


namespace plot_length_is_60_l462_46229

/-- Proves that the length of a rectangular plot is 60 meters given the specified conditions -/
theorem plot_length_is_60 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 20 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  cost_per_meter * perimeter = total_cost →
  length = 60 := by
  sorry

end plot_length_is_60_l462_46229


namespace point_not_on_line_l462_46244

/-- Given a line y = mx + b where m and b are real numbers satisfying mb > 0,
    prove that the point (2023, 0) cannot lie on this line. -/
theorem point_not_on_line (m b : ℝ) (h : m * b > 0) :
  ¬(0 = 2023 * m + b) := by
  sorry

end point_not_on_line_l462_46244


namespace max_intersections_after_300_turns_l462_46255

/-- The number of intersections formed by n lines on a plane -/
def num_intersections (n : ℕ) : ℕ := n.choose 2

/-- The number of lines after 300 turns -/
def total_lines : ℕ := 3 + 300

theorem max_intersections_after_300_turns :
  num_intersections total_lines = 45853 := by
  sorry

end max_intersections_after_300_turns_l462_46255


namespace divisor_problem_l462_46227

theorem divisor_problem (d : ℕ+) (n : ℕ) (h1 : n % d = 3) (h2 : (2 * n) % d = 2) : d = 4 := by
  sorry

end divisor_problem_l462_46227


namespace simple_interest_time_calculation_l462_46280

/-- Simple interest calculation theorem -/
theorem simple_interest_time_calculation
  (P : ℝ) (R : ℝ) (SI : ℝ)
  (h_P : P = 800)
  (h_R : R = 6.25)
  (h_SI : SI = 200)
  (h_formula : SI = P * R * (SI * 100 / (P * R)) / 100) :
  SI * 100 / (P * R) = 4 := by
  sorry

end simple_interest_time_calculation_l462_46280


namespace sheridan_fish_count_l462_46203

/-- The number of fish Mrs. Sheridan initially had -/
def initial_fish : ℕ := 22

/-- The number of fish Mrs. Sheridan received from her sister -/
def received_fish : ℕ := 47

/-- The total number of fish Mrs. Sheridan has now -/
def total_fish : ℕ := initial_fish + received_fish

theorem sheridan_fish_count : total_fish = 69 := by
  sorry

end sheridan_fish_count_l462_46203


namespace radius_is_five_l462_46220

/-- A configuration of tangent lines to a circle -/
structure TangentConfiguration where
  -- The radius of the circle
  r : ℝ
  -- The length of tangent line AB
  ab : ℝ
  -- The length of tangent line CD
  cd : ℝ
  -- The length of tangent line EF
  ef : ℝ
  -- AB and CD are parallel
  parallel_ab_cd : True
  -- A, C, and D are points of tangency
  tangency_points : True
  -- EF intersects AB and CD
  ef_intersects : True
  -- Tangency point for EF falls between AB and CD
  ef_tangency_between : True
  -- Given lengths
  ab_length : ab = 7
  cd_length : cd = 12
  ef_length : ef = 25

/-- The theorem stating that the radius is 5 given the configuration -/
theorem radius_is_five (config : TangentConfiguration) : config.r = 5 := by
  sorry

end radius_is_five_l462_46220


namespace total_sibling_age_l462_46254

/-- Represents the ages of the siblings -/
structure SiblingAges where
  susan : ℝ
  arthur : ℝ
  tom : ℝ
  bob : ℝ
  emily : ℝ
  david : ℝ
  youngest : ℝ

/-- Theorem stating the total age of the siblings -/
theorem total_sibling_age (ages : SiblingAges) : 
  ages.arthur = ages.susan + 2 →
  ages.tom = ages.bob - 3 →
  ages.emily = ages.susan / 2 →
  ages.david = (ages.arthur + ages.tom + ages.emily) / 3 →
  ages.susan - ages.tom = 2 * (ages.emily - ages.david) →
  ages.bob = 11 →
  ages.susan = 15 →
  ages.emily = ages.youngest + 2.5 →
  ages.susan + ages.arthur + ages.tom + ages.bob + ages.emily + ages.david + ages.youngest = 74.5 := by
  sorry


end total_sibling_age_l462_46254


namespace imon_disentanglement_l462_46207

/-- Represents the set of imons and their entanglements -/
structure ImonConfiguration where
  imons : Set Nat
  entangled : Nat → Nat → Bool

/-- Operation (i): Remove an imon entangled with an odd number of other imons -/
def removeOddEntangled (config : ImonConfiguration) : ImonConfiguration :=
  sorry

/-- Operation (ii): Double the set of imons -/
def doubleImons (config : ImonConfiguration) : ImonConfiguration :=
  sorry

/-- Checks if there are any entangled imons in the configuration -/
def hasEntangledImons (config : ImonConfiguration) : Bool :=
  sorry

/-- Represents a sequence of operations -/
inductive Operation
  | Remove
  | Double

theorem imon_disentanglement 
  (initial : ImonConfiguration) : 
  ∃ (ops : List Operation), 
    let final := ops.foldl (λ config op => 
      match op with
      | Operation.Remove => removeOddEntangled config
      | Operation.Double => doubleImons config
    ) initial
    ¬ hasEntangledImons final :=
  sorry

end imon_disentanglement_l462_46207


namespace part1_part2_l462_46298

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |x - 2*a + 3|

-- Part 1: When a = 2
theorem part1 : {x : ℝ | f x 2 ≤ 9} = {x : ℝ | -2 ≤ x ∧ x ≤ 4} := by sorry

-- Part 2: When a ≠ 2
theorem part2 : ∀ a : ℝ, a ≠ 2 → 
  ((∀ x : ℝ, f x a ≥ 4) ↔ (a ≤ -2/3 ∨ a ≥ 14/3)) := by sorry

end part1_part2_l462_46298


namespace quadratic_decreasing_implies_m_range_l462_46252

/-- Given a quadratic function f(x) = x^2 + 4mx + n that is decreasing on the interval [2, 6],
    prove that m ≤ -3. -/
theorem quadratic_decreasing_implies_m_range
  (f : ℝ → ℝ)
  (m n : ℝ)
  (h_f : ∀ x, f x = x^2 + 4*m*x + n)
  (h_decreasing : ∀ x y, x ∈ Set.Icc 2 6 → y ∈ Set.Icc 2 6 → x < y → f x > f y) :
  m ≤ -3 := by
  sorry

end quadratic_decreasing_implies_m_range_l462_46252


namespace linear_function_passes_through_points_linear_function_unique_l462_46239

/-- A linear function passing through two points (2, 3) and (3, 2) -/
def linearFunction (x : ℝ) : ℝ := -x + 5

/-- The theorem stating that the linear function passes through the given points -/
theorem linear_function_passes_through_points :
  linearFunction 2 = 3 ∧ linearFunction 3 = 2 := by
  sorry

/-- The theorem stating that the linear function is unique -/
theorem linear_function_unique (f : ℝ → ℝ) :
  f 2 = 3 → f 3 = 2 → ∀ x, f x = linearFunction x := by
  sorry

end linear_function_passes_through_points_linear_function_unique_l462_46239


namespace f_6_equals_16_l462_46262

def f : ℕ → ℕ 
  | x => if x < 5 then 2^x else f (x-1)

theorem f_6_equals_16 : f 6 = 16 := by
  sorry

end f_6_equals_16_l462_46262


namespace intersection_condition_l462_46270

/-- 
Given two equations:
1) y = √(2x^2 + 2x - m)
2) y = x - 2
This theorem states that for these equations to have a real intersection, 
m must be greater than or equal to 12.
-/
theorem intersection_condition (x y m : ℝ) : 
  (y = Real.sqrt (2 * x^2 + 2 * x - m) ∧ y = x - 2) → m ≥ 12 := by
  sorry

end intersection_condition_l462_46270


namespace six_digit_divisible_by_396_l462_46258

theorem six_digit_divisible_by_396 (n : ℕ) :
  (∃ x y z : ℕ, 
    0 ≤ x ∧ x ≤ 9 ∧
    0 ≤ y ∧ y ≤ 9 ∧
    0 ≤ z ∧ z ≤ 9 ∧
    n = 100000 * x + 10000 * y + 3420 + z) →
  n % 396 = 0 →
  n = 453420 ∨ n = 413424 :=
by sorry

end six_digit_divisible_by_396_l462_46258


namespace max_value_of_sum_l462_46211

theorem max_value_of_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 5) :
  ∃ (M : ℝ), M = Real.sqrt 70 ∧ x + 2*y + 3*z ≤ M ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ + 2*y₀ + 3*z₀ = M :=
sorry

end max_value_of_sum_l462_46211


namespace inverse_g_87_l462_46274

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 6

-- State the theorem
theorem inverse_g_87 : g⁻¹ 87 = 3 := by sorry

end inverse_g_87_l462_46274


namespace max_segment_product_l462_46296

-- Define the segment AB of unit length
def unitSegment : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Define a function to calculate the product of segment lengths
def segmentProduct (a b c : ℝ) : ℝ :=
  a * (a + b) * 1 * b * (b + c) * c

-- Theorem statement
theorem max_segment_product :
  ∃ (max : ℝ), max = Real.sqrt 5 / 125 ∧
  ∀ (a b c : ℝ), a ∈ unitSegment → b ∈ unitSegment → c ∈ unitSegment →
  a + b + c = 1 → segmentProduct a b c ≤ max :=
sorry

end max_segment_product_l462_46296


namespace library_repacking_l462_46224

theorem library_repacking (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) : 
  initial_boxes = 1500 → 
  books_per_initial_box = 45 → 
  books_per_new_box = 47 → 
  (initial_boxes * books_per_initial_box) % books_per_new_box = 8 := by
  sorry

end library_repacking_l462_46224


namespace total_sand_needed_l462_46284

/-- The amount of sand in grams needed to fill one square inch -/
def sand_per_square_inch : ℕ := 3

/-- The length of the rectangular patch in inches -/
def rectangle_length : ℕ := 6

/-- The width of the rectangular patch in inches -/
def rectangle_width : ℕ := 7

/-- The side length of the square patch in inches -/
def square_side : ℕ := 5

/-- Calculates the area of a rectangle given its length and width -/
def rectangle_area (length width : ℕ) : ℕ := length * width

/-- Calculates the area of a square given its side length -/
def square_area (side : ℕ) : ℕ := side * side

/-- Calculates the amount of sand needed for a given area -/
def sand_needed (area : ℕ) : ℕ := area * sand_per_square_inch

/-- Theorem stating the total amount of sand needed for Jason's sand art -/
theorem total_sand_needed :
  sand_needed (rectangle_area rectangle_length rectangle_width) +
  sand_needed (square_area square_side) = 201 := by
  sorry


end total_sand_needed_l462_46284


namespace derivative_f_at_neg_one_l462_46236

def f (x : ℝ) : ℝ := x^2 * (x + 1)

theorem derivative_f_at_neg_one :
  (deriv f) (-1) = 1 := by sorry

end derivative_f_at_neg_one_l462_46236


namespace machine_value_theorem_l462_46233

/-- Calculates the machine's value after 2 years given the initial conditions -/
def machine_value_after_two_years (initial_value : ℝ) (depreciation_rate_year1 : ℝ) 
  (depreciation_rate_subsequent : ℝ) (inflation_rate_year1 : ℝ) (inflation_rate_year2 : ℝ) 
  (maintenance_cost_year1 : ℝ) (maintenance_cost_increase_rate : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the machine's value after 2 years is $754.58 -/
theorem machine_value_theorem : 
  machine_value_after_two_years 1000 0.12 0.08 0.02 0.035 50 0.05 = 754.58 := by
  sorry

end machine_value_theorem_l462_46233


namespace total_rainfall_three_years_l462_46216

def average_rainfall_2003 : ℝ := 50
def rainfall_increase_2004 : ℝ := 3
def rainfall_increase_2005 : ℝ := 5
def months_per_year : ℕ := 12

theorem total_rainfall_three_years : 
  (average_rainfall_2003 * months_per_year) +
  ((average_rainfall_2003 + rainfall_increase_2004) * months_per_year) +
  ((average_rainfall_2003 + rainfall_increase_2004 + rainfall_increase_2005) * months_per_year) = 1932 := by
  sorry

end total_rainfall_three_years_l462_46216


namespace point_same_side_l462_46248

def line (x y : ℝ) : ℝ := x + y - 1

def same_side (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (line x₁ y₁ > 0 ∧ line x₂ y₂ > 0) ∨ (line x₁ y₁ < 0 ∧ line x₂ y₂ < 0)

theorem point_same_side : same_side 1 2 (-1) 3 := by
  sorry

end point_same_side_l462_46248


namespace initial_amount_problem_l462_46279

theorem initial_amount_problem (initial_amount : ℝ) : 
  (initial_amount * (1 + 1/8) * (1 + 1/8) = 97200) → initial_amount = 76800 := by
  sorry

end initial_amount_problem_l462_46279


namespace range_of_a_l462_46275

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else Real.log x

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → -1 ≤ a ∧ a < 1/2 :=
by sorry

end range_of_a_l462_46275


namespace square_of_binomial_l462_46287

theorem square_of_binomial (a b : ℝ) : (2*a - 3*b)^2 = 4*a^2 - 12*a*b + 9*b^2 := by
  sorry

end square_of_binomial_l462_46287


namespace journey_speed_l462_46202

theorem journey_speed (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) :
  total_distance = 240 ∧ 
  total_time = 20 ∧ 
  first_half_speed = 10 →
  (total_distance / 2) / ((total_time - (total_distance / 2) / first_half_speed)) = 15 :=
by sorry

end journey_speed_l462_46202


namespace two_roots_of_f_l462_46247

/-- The function f(x) = 2^x - 3x has exactly two real roots -/
theorem two_roots_of_f : ∃! (n : ℕ), n = 2 ∧ (∃ (S : Set ℝ), S = {x : ℝ | 2^x - 3*x = 0} ∧ Finite S ∧ Nat.card S = n) := by
  sorry

end two_roots_of_f_l462_46247
