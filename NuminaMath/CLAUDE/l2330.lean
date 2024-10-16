import Mathlib

namespace NUMINAMATH_CALUDE_blue_contour_area_relation_l2330_233099

/-- Represents the area of a blue contour on a sphere. -/
def blueContourArea (sphereRadius : ℝ) (contourArea : ℝ) : Prop :=
  contourArea ≥ 0 ∧ contourArea ≤ 4 * Real.pi * sphereRadius^2

/-- Theorem stating the relationship between blue contour areas on two concentric spheres. -/
theorem blue_contour_area_relation
  (r₁ : ℝ) (r₂ : ℝ) (a₁ : ℝ) (a₂ : ℝ)
  (h_r₁ : r₁ = 4)
  (h_r₂ : r₂ = 6)
  (h_a₁ : a₁ = 27)
  (h_positive : r₁ > 0 ∧ r₂ > 0)
  (h_contour₁ : blueContourArea r₁ a₁)
  (h_contour₂ : blueContourArea r₂ a₂)
  (h_proportion : a₁ / a₂ = (r₁ / r₂)^2) :
  a₂ = 60.75 :=
sorry

end NUMINAMATH_CALUDE_blue_contour_area_relation_l2330_233099


namespace NUMINAMATH_CALUDE_mr_a_loss_l2330_233000

/-- Calculates the total loss for Mr. A in a house transaction --/
def calculate_loss (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : ℝ :=
  let first_sale_price := initial_value * (1 - loss_percent)
  let second_sale_price := first_sale_price * (1 + gain_percent)
  second_sale_price - initial_value

/-- Theorem stating that Mr. A loses $2040 in the house transaction --/
theorem mr_a_loss :
  calculate_loss 12000 0.15 0.20 = 2040 := by sorry

end NUMINAMATH_CALUDE_mr_a_loss_l2330_233000


namespace NUMINAMATH_CALUDE_trig_identity_l2330_233026

theorem trig_identity : 
  Real.cos (70 * π / 180) * Real.cos (335 * π / 180) + 
  Real.sin (110 * π / 180) * Real.sin (25 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2330_233026


namespace NUMINAMATH_CALUDE_g_at_negative_three_l2330_233044

def g (x : ℝ) : ℝ := 3 * x^5 - 5 * x^4 + 7 * x^3 - 10 * x^2 - 12 * x + 36

theorem g_at_negative_three : g (-3) = -1341 := by sorry

end NUMINAMATH_CALUDE_g_at_negative_three_l2330_233044


namespace NUMINAMATH_CALUDE_estimate_pi_l2330_233060

theorem estimate_pi (total_beans : ℕ) (beans_in_circle : ℕ) 
  (h1 : total_beans = 80) (h2 : beans_in_circle = 64) : 
  (4 * beans_in_circle : ℝ) / total_beans = 3.2 :=
sorry

end NUMINAMATH_CALUDE_estimate_pi_l2330_233060


namespace NUMINAMATH_CALUDE_differential_equation_solution_l2330_233097

open Real

theorem differential_equation_solution 
  (y : ℝ → ℝ) 
  (C₁ C₂ : ℝ) 
  (h : ∀ x, y x = (C₁ + C₂ * x) * exp (3 * x) + exp x - 8 * x^2 * exp (3 * x)) :
  ∀ x, (deriv^[2] y) x - 6 * (deriv y) x + 9 * y x = 4 * exp x - 16 * exp (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l2330_233097


namespace NUMINAMATH_CALUDE_count_valid_triangles_l2330_233012

/-- A triangle with integral side lengths --/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  sum_eq_12 : a + b + c = 12
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of all valid IntTriangles --/
def validTriangles : Finset IntTriangle := sorry

theorem count_valid_triangles : Finset.card validTriangles = 6 := by sorry

end NUMINAMATH_CALUDE_count_valid_triangles_l2330_233012


namespace NUMINAMATH_CALUDE_square_difference_l2330_233042

theorem square_difference (x y z w : ℝ) 
  (sum_xy : x + y = 10)
  (diff_xy : x - y = 8)
  (sum_yz : y + z = 15)
  (sum_zw : z + w = 20) :
  x^2 - w^2 = 45 := by sorry

end NUMINAMATH_CALUDE_square_difference_l2330_233042


namespace NUMINAMATH_CALUDE_hexadecagon_triangles_l2330_233076

/-- The number of sides in a regular hexadecagon -/
def n : ℕ := 16

/-- The number of vertices to choose for each triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular hexadecagon -/
def num_triangles : ℕ := Nat.choose n k

theorem hexadecagon_triangles : num_triangles = 560 := by
  sorry

end NUMINAMATH_CALUDE_hexadecagon_triangles_l2330_233076


namespace NUMINAMATH_CALUDE_train_length_l2330_233033

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 36 → ∃ (length_m : ℝ), 
  (abs (length_m - 600.12) < 0.01) ∧ (length_m = speed_kmh * (5/18) * time_s) := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2330_233033


namespace NUMINAMATH_CALUDE_max_pieces_on_board_l2330_233093

/-- Represents a piece on the grid -/
inductive Piece
| Red
| Blue

/-- Represents a cell on the grid -/
structure Cell :=
(row : Nat)
(col : Nat)
(piece : Option Piece)

/-- Represents the game board -/
structure Board :=
(cells : List Cell)
(rowCount : Nat)
(colCount : Nat)

/-- Checks if a cell contains a piece -/
def Cell.hasPiece (cell : Cell) : Bool :=
  cell.piece.isSome

/-- Counts the number of pieces on the board -/
def Board.pieceCount (board : Board) : Nat :=
  board.cells.filter Cell.hasPiece |>.length

/-- Checks if a piece sees exactly five pieces of the other color in its row and column -/
def Board.validPiecePlacement (board : Board) (cell : Cell) : Bool :=
  sorry

/-- Checks if all pieces on the board satisfy the placement rule -/
def Board.validBoard (board : Board) : Bool :=
  board.cells.all (Board.validPiecePlacement board)

theorem max_pieces_on_board (board : Board) :
  board.rowCount = 200 ∧ board.colCount = 200 ∧ board.validBoard →
  board.pieceCount ≤ 3800 :=
sorry

end NUMINAMATH_CALUDE_max_pieces_on_board_l2330_233093


namespace NUMINAMATH_CALUDE_center_is_nine_l2330_233094

def Grid := Fin 3 → Fin 3 → Nat

def is_valid_arrangement (g : Grid) : Prop :=
  (∀ n : Nat, n ∈ Finset.range 9 → ∃ i j, g i j = n + 1) ∧
  (∀ i j, g i j ∈ Finset.range 9 → g i j ≤ 9) ∧
  (∀ n : Nat, n ∈ Finset.range 8 → 
    ∃ i j k l, g i j = n + 1 ∧ g k l = n + 2 ∧ 
    ((i = k ∧ (j = l + 1 ∨ j + 1 = l)) ∨ 
     (j = l ∧ (i = k + 1 ∨ i + 1 = k))))

def top_edge_sum (g : Grid) : Nat :=
  g 0 0 + g 0 1 + g 0 2

theorem center_is_nine (g : Grid) 
  (h1 : is_valid_arrangement g) 
  (h2 : top_edge_sum g = 15) : 
  g 1 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_center_is_nine_l2330_233094


namespace NUMINAMATH_CALUDE_meaningful_sqrt_fraction_range_l2330_233052

theorem meaningful_sqrt_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = (Real.sqrt (4 - x)) / (Real.sqrt (x - 1))) ↔ (1 < x ∧ x ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_fraction_range_l2330_233052


namespace NUMINAMATH_CALUDE_f_properties_l2330_233078

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 * Real.exp x

theorem f_properties :
  (∃ x, f x = 0) ∧
  (∃ x₁ x₂, IsLocalMax f x₁ ∧ IsLocalMin f x₂) ∧
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = 1 ∧ f x₂ = 1 ∧ f x₃ = 1 ∧
    ∀ x, f x = 1 → x = x₁ ∨ x = x₂ ∨ x = x₃) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l2330_233078


namespace NUMINAMATH_CALUDE_candy_bar_distribution_l2330_233037

theorem candy_bar_distribution (total_bars : ℕ) (spare_bars : ℕ) (num_friends : ℕ) 
  (h1 : total_bars = 24)
  (h2 : spare_bars = 10)
  (h3 : num_friends = 7)
  : (total_bars - spare_bars) / num_friends = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_distribution_l2330_233037


namespace NUMINAMATH_CALUDE_book_price_increase_l2330_233021

theorem book_price_increase (initial_price : ℝ) : 
  let decrease_rate : ℝ := 0.20
  let net_change_rate : ℝ := 0.11999999999999986
  let price_after_decrease : ℝ := initial_price * (1 - decrease_rate)
  let final_price : ℝ := initial_price * (1 + net_change_rate)
  ∃ (increase_rate : ℝ), 
    price_after_decrease * (1 + increase_rate) = final_price ∧ 
    abs (increase_rate - 0.4) < 0.00000000000001 := by
  sorry

end NUMINAMATH_CALUDE_book_price_increase_l2330_233021


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l2330_233049

/-- The distance from a point on the line y = 2x + 1 to the x-axis -/
theorem distance_to_x_axis (k : ℝ) : 
  let M : ℝ × ℝ := (-2, k)
  let line_eq : ℝ → ℝ := λ x => 2 * x + 1
  k = line_eq (-2) →
  |k| = 3 := by
sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l2330_233049


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2330_233040

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 6}
def B : Set Nat := {2, 3, 4}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2330_233040


namespace NUMINAMATH_CALUDE_prop_q_indeterminate_l2330_233019

theorem prop_q_indeterminate (h1 : p ∨ q) (h2 : ¬(¬p)) : 
  (q ∨ ¬q) ∧ (∃ (v : Prop), v = q) :=
by sorry

end NUMINAMATH_CALUDE_prop_q_indeterminate_l2330_233019


namespace NUMINAMATH_CALUDE_inequality_proof_l2330_233066

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2330_233066


namespace NUMINAMATH_CALUDE_simplest_radical_form_among_options_l2330_233017

def is_simplest_radical_form (x : ℝ) : Prop :=
  ∃ n : ℕ, x = Real.sqrt n ∧ 
  (∀ m : ℕ, m ^ 2 ∣ n → m = 1) ∧
  (∀ a b : ℕ, n ≠ a / b)

theorem simplest_radical_form_among_options : 
  is_simplest_radical_form (Real.sqrt 10) ∧
  ¬is_simplest_radical_form (Real.sqrt 9) ∧
  ¬is_simplest_radical_form (Real.sqrt 20) ∧
  ¬is_simplest_radical_form (Real.sqrt (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_simplest_radical_form_among_options_l2330_233017


namespace NUMINAMATH_CALUDE_jill_cookie_sales_l2330_233031

def cookie_sales (goal : ℕ) (first second third fourth fifth : ℕ) : Prop :=
  let total_sold := first + second + third + fourth + fifth
  goal - total_sold = 75

theorem jill_cookie_sales :
  cookie_sales 150 5 20 10 30 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jill_cookie_sales_l2330_233031


namespace NUMINAMATH_CALUDE_safari_animal_count_l2330_233029

/-- The safari animal count problem -/
theorem safari_animal_count 
  (total : ℕ) 
  (antelopes : ℕ) 
  (rabbits : ℕ) 
  (hyenas : ℕ) 
  (wild_dogs : ℕ) 
  (leopards : ℕ) :
  total = 605 →
  antelopes = 80 →
  rabbits > antelopes →
  hyenas = antelopes + rabbits - 42 →
  wild_dogs = hyenas + 50 →
  leopards = rabbits / 2 →
  total = antelopes + rabbits + hyenas + wild_dogs + leopards →
  rabbits - antelopes = 70 := by
sorry

end NUMINAMATH_CALUDE_safari_animal_count_l2330_233029


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2330_233082

theorem simplify_sqrt_expression : 
  Real.sqrt 5 - Real.sqrt 20 + Real.sqrt 45 = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2330_233082


namespace NUMINAMATH_CALUDE_unique_x_intercept_l2330_233058

theorem unique_x_intercept (x : ℝ) : 
  ∃! x, (x - 4) * (x^2 + 4*x + 13) = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_intercept_l2330_233058


namespace NUMINAMATH_CALUDE_distinct_sums_count_l2330_233064

def bag_X : Finset ℕ := {2, 5, 7}
def bag_Y : Finset ℕ := {1, 4, 8}

theorem distinct_sums_count : 
  Finset.card ((bag_X.product bag_Y).image (fun p => p.1 + p.2)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sums_count_l2330_233064


namespace NUMINAMATH_CALUDE_multiple_of_24_multiple_of_3_and_8_six_hundred_is_multiple_of_24_l2330_233083

theorem multiple_of_24 : ∃ (n : ℕ), 600 = 24 * n := by
  sorry

theorem multiple_of_3_and_8 (x : ℕ) : x % 24 = 0 ↔ x % 3 = 0 ∧ x % 8 = 0 := by
  sorry

theorem six_hundred_is_multiple_of_24 : 600 % 24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_24_multiple_of_3_and_8_six_hundred_is_multiple_of_24_l2330_233083


namespace NUMINAMATH_CALUDE_rhombus_area_from_square_midpoints_l2330_233048

/-- The area of a rhombus formed by connecting the midpoints of a square with side length 4 is 8 -/
theorem rhombus_area_from_square_midpoints (s : ℝ) (h : s = 4) : 
  let rhombus_area := s^2 / 2
  rhombus_area = 8 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_from_square_midpoints_l2330_233048


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l2330_233007

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  total_members : Nat
  captain_age : Nat
  wicket_keeper_age : Nat
  team_average_age : Nat

/-- The difference between the average age of remaining players and the whole team -/
def age_difference (team : CricketTeam) : Rat :=
  let remaining_members := team.total_members - 2
  let total_age := team.team_average_age * team.total_members
  let remaining_age := total_age - team.captain_age - team.wicket_keeper_age
  let remaining_average := remaining_age / remaining_members
  team.team_average_age - remaining_average

/-- Theorem stating the age difference for a specific cricket team -/
theorem cricket_team_age_difference :
  ∃ (team : CricketTeam),
    team.total_members = 11 ∧
    team.captain_age = 26 ∧
    team.wicket_keeper_age = team.captain_age + 5 ∧
    team.team_average_age = 24 ∧
    age_difference team = 1 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l2330_233007


namespace NUMINAMATH_CALUDE_product_modulo_l2330_233028

theorem product_modulo : (2345 * 1554) % 700 = 630 := by
  sorry

end NUMINAMATH_CALUDE_product_modulo_l2330_233028


namespace NUMINAMATH_CALUDE_unique_solution_unique_solution_l2330_233089

-- Define the sets A and B
def A (k : ℕ) : Set ℕ := {1, 2, 3, k}
def B (a : ℕ) : Set ℕ := {4, 7, a^4, a^2 + 3*a}

-- Define the function f
def f (x : ℕ) : ℕ := 3*x + 1

-- Theorem statement
theorem unique_solution (a k : ℕ) :
  (∀ x ∈ A k, ∃ y ∈ B a, f x = y) ∧ 
  (∀ y ∈ B a, ∃ x ∈ A k, f x = y) →
  a = 2 ∧ k = 5 := by
  sorry

-- Alternative theorem statement if the above doesn't compile
theorem unique_solution' (a k : ℕ) :
  (∀ x, x ∈ A k → ∃ y ∈ B a, f x = y) ∧ 
  (∀ y, y ∈ B a → ∃ x ∈ A k, f x = y) →
  a = 2 ∧ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_unique_solution_l2330_233089


namespace NUMINAMATH_CALUDE_white_balls_count_white_balls_count_specific_l2330_233010

/-- The number of white balls in a bag, given the total number of balls,
    the number of balls of each color (except white), and the probability
    of choosing a ball that is neither red nor purple. -/
theorem white_balls_count (total green yellow red purple : ℕ)
                          (prob_not_red_purple : ℚ) : ℕ :=
  let total_balls : ℕ := total
  let green_balls : ℕ := green
  let yellow_balls : ℕ := yellow
  let red_balls : ℕ := red
  let purple_balls : ℕ := purple
  let prob_not_red_or_purple : ℚ := prob_not_red_purple
  24

/-- The number of white balls is 24 given the specific conditions. -/
theorem white_balls_count_specific : white_balls_count 60 18 2 15 3 (7/10) = 24 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_white_balls_count_specific_l2330_233010


namespace NUMINAMATH_CALUDE_solve_rent_problem_l2330_233090

def rent_problem (n : ℕ) : Prop :=
  let original_average : ℚ := 800
  let increased_rent : ℚ := 800 * (1 + 1/4)
  let new_average : ℚ := 850
  (n * original_average + (increased_rent - 800)) / n = new_average

theorem solve_rent_problem : 
  ∃ (n : ℕ), n > 0 ∧ rent_problem n ∧ n = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_rent_problem_l2330_233090


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2330_233095

theorem smallest_number_with_given_remainders :
  ∃ (x : ℕ), x > 0 ∧
    x % 11 = 9 ∧
    x % 13 = 11 ∧
    x % 15 = 13 ∧
    (∀ y : ℕ, y > 0 ∧ y % 11 = 9 ∧ y % 13 = 11 ∧ y % 15 = 13 → x ≤ y) ∧
    x = 2143 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2330_233095


namespace NUMINAMATH_CALUDE_machine_value_after_two_years_l2330_233098

def initial_value : ℝ := 8000
def depreciation_rate : ℝ := 0.15

def market_value_after_two_years (initial : ℝ) (rate : ℝ) : ℝ :=
  initial * (1 - rate) * (1 - rate)

theorem machine_value_after_two_years :
  market_value_after_two_years initial_value depreciation_rate = 5780 := by
  sorry

end NUMINAMATH_CALUDE_machine_value_after_two_years_l2330_233098


namespace NUMINAMATH_CALUDE_finns_purchase_theorem_l2330_233050

/-- The cost of Finn's purchase given the conditions of the problem -/
def finns_purchase_cost (paper_clip_cost index_card_cost : ℚ) : ℚ :=
  12 * paper_clip_cost + 10 * index_card_cost

/-- The theorem stating the cost of Finn's purchase -/
theorem finns_purchase_theorem :
  ∃ (index_card_cost : ℚ),
    15 * (1.85 : ℚ) + 7 * index_card_cost = 55.40 ∧
    finns_purchase_cost 1.85 index_card_cost = 61.70 := by
  sorry

#eval finns_purchase_cost (1.85 : ℚ) (3.95 : ℚ)

end NUMINAMATH_CALUDE_finns_purchase_theorem_l2330_233050


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l2330_233025

theorem football_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  : (total_players - throwers) / 3 * 2 + throwers = 59 := by
  sorry

end NUMINAMATH_CALUDE_football_team_right_handed_players_l2330_233025


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2330_233038

theorem unique_solution_quadratic (k : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2330_233038


namespace NUMINAMATH_CALUDE_no_double_application_function_l2330_233088

theorem no_double_application_function :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 2017 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l2330_233088


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l2330_233047

noncomputable section

-- Define the ellipse Γ
def Γ (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

-- Define the line l
def l (x y m : ℝ) : Prop := y = x + m

-- Define the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem ellipse_intersection_theorem :
  ∃ (xA yA xB yB m x₀ : ℝ),
    Γ xA yA ∧ Γ xB yB ∧  -- A and B are on the ellipse
    l xA yA m ∧ l xB yB m ∧  -- A and B are on the line l
    distance xA yA xB yB = 3 * Real.sqrt 2 ∧  -- |AB| = 3√2
    distance x₀ 2 xA yA = distance x₀ 2 xB yB ∧  -- |PA| = |PB|
    (x₀ = -3 ∨ x₀ = -1) :=
by sorry

end

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l2330_233047


namespace NUMINAMATH_CALUDE_shooter_probability_l2330_233039

theorem shooter_probability (p_10 p_9 p_8 : ℝ) 
  (h1 : p_10 = 0.24)
  (h2 : p_9 = 0.28)
  (h3 : p_8 = 0.19) :
  1 - p_10 - p_9 = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l2330_233039


namespace NUMINAMATH_CALUDE_sqrt_product_equals_21_l2330_233023

theorem sqrt_product_equals_21 (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (7 * x) * Real.sqrt (21 * x) = 21) : 
  x = 21 / 97 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_21_l2330_233023


namespace NUMINAMATH_CALUDE_reciprocal_sum_quarters_fifths_l2330_233013

theorem reciprocal_sum_quarters_fifths : (1 / (1 / 4 + 1 / 5) : ℚ) = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_quarters_fifths_l2330_233013


namespace NUMINAMATH_CALUDE_tips_fraction_l2330_233085

/-- Given a worker who works for 7 months, with one month's tips being twice
    the average of the other 6 months, the fraction of total tips from that
    one month is 1/4. -/
theorem tips_fraction (total_months : ℕ) (special_month_tips : ℝ) 
    (other_months_tips : ℝ) : 
    total_months = 7 →
    special_month_tips = 2 * (other_months_tips / 6) →
    special_month_tips / (special_month_tips + other_months_tips) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_tips_fraction_l2330_233085


namespace NUMINAMATH_CALUDE_cl_ab_ratio_l2330_233051

/-- A regular pentagon with specific points and angle conditions -/
structure RegularPentagonWithPoints where
  /-- The side length of the regular pentagon -/
  s : ℝ
  /-- Point K on side AE -/
  k : ℝ
  /-- Point L on side CD -/
  l : ℝ
  /-- The sum of angles LAE and KCD is 108° -/
  angle_sum : k + l = 108
  /-- The ratio of AK to KE is 3:7 -/
  length_ratio : k / (s - k) = 3 / 7
  /-- The side length is positive -/
  s_pos : s > 0
  /-- K is between A and E -/
  k_between : 0 < k ∧ k < s
  /-- L is between C and D -/
  l_between : 0 < l ∧ l < s

/-- The theorem stating the ratio of CL to AB in the given pentagon -/
theorem cl_ab_ratio (p : RegularPentagonWithPoints) : (p.s - p.l) / p.s = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_cl_ab_ratio_l2330_233051


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2330_233074

/-- Represents a rectangle with its diagonal divided into 12 equal segments -/
structure DividedRectangle where
  totalSegments : ℕ
  nonShadedArea : ℝ

/-- Calculates the area of shaded parts in a divided rectangle -/
def shadedArea (rect : DividedRectangle) : ℝ :=
  sorry

/-- Theorem stating the relationship between non-shaded and shaded areas -/
theorem shaded_area_calculation (rect : DividedRectangle) 
  (h1 : rect.totalSegments = 12)
  (h2 : rect.nonShadedArea = 10) :
  shadedArea rect = 14 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2330_233074


namespace NUMINAMATH_CALUDE_lorenzo_stamps_l2330_233070

def stamps_needed (current : ℕ) (row_size : ℕ) : ℕ :=
  (row_size - (current % row_size)) % row_size

theorem lorenzo_stamps : stamps_needed 37 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lorenzo_stamps_l2330_233070


namespace NUMINAMATH_CALUDE_f_lower_bound_g_min_max_l2330_233008

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := x^2 + Real.log x

def g (x : ℝ) : ℝ := x^2 - 2 * Real.log x

-- State the theorems
theorem f_lower_bound (x : ℝ) (hx : x > 0) : f x ≥ (x^3 + x - 1) / x := by sorry

theorem g_min_max :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, g x ≥ 1) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, g x = 1) ∧
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, g x ≤ 4 - 2 * Real.log 2) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, g x = 4 - 2 * Real.log 2) := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_g_min_max_l2330_233008


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2330_233092

theorem inequality_solution_set (a b c : ℝ) :
  (∀ x : ℝ, a * x + b > c ↔ x < 4) →
  (∀ x : ℝ, a * (x - 3) + b > c ↔ x < 7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2330_233092


namespace NUMINAMATH_CALUDE_interval_of_increase_l2330_233096

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 5*x + 6) / Real.log (1/4)

def domain (x : ℝ) : Prop := x < 2 ∨ x > 3

theorem interval_of_increase :
  ∀ x y, domain x → domain y → x < y → x < 2 → f x > f y := by sorry

end NUMINAMATH_CALUDE_interval_of_increase_l2330_233096


namespace NUMINAMATH_CALUDE_average_temperature_proof_l2330_233084

theorem average_temperature_proof (tuesday wednesday thursday friday : ℝ) : 
  (tuesday + wednesday + thursday) / 3 = 32 →
  friday = 44 →
  tuesday = 38 →
  (wednesday + thursday + friday) / 3 = 34 := by
sorry

end NUMINAMATH_CALUDE_average_temperature_proof_l2330_233084


namespace NUMINAMATH_CALUDE_unique_magnitude_quadratic_roots_l2330_233004

theorem unique_magnitude_quadratic_roots : ∃! m : ℝ, ∀ z : ℂ, z^2 - 6*z + 20 = 0 → Complex.abs z = m := by
  sorry

end NUMINAMATH_CALUDE_unique_magnitude_quadratic_roots_l2330_233004


namespace NUMINAMATH_CALUDE_quadratic_has_real_root_l2330_233075

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_real_root_l2330_233075


namespace NUMINAMATH_CALUDE_average_difference_l2330_233063

def num_students : ℕ := 120
def num_teachers : ℕ := 5
def class_sizes : List ℕ := [40, 30, 20, 15, 15]

def t : ℚ := (num_students : ℚ) / num_teachers

def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes) : ℚ) / num_students

theorem average_difference : t - s = -3.92 := by sorry

end NUMINAMATH_CALUDE_average_difference_l2330_233063


namespace NUMINAMATH_CALUDE_double_age_in_four_years_l2330_233069

/-- The number of years until Fouad's age is double Ahmed's age -/
def years_until_double_age (ahmed_age : ℕ) (fouad_age : ℕ) : ℕ :=
  fouad_age - ahmed_age

theorem double_age_in_four_years (ahmed_age : ℕ) (fouad_age : ℕ) 
  (h1 : ahmed_age = 11) (h2 : fouad_age = 26) : 
  years_until_double_age ahmed_age fouad_age = 4 := by
  sorry

#check double_age_in_four_years

end NUMINAMATH_CALUDE_double_age_in_four_years_l2330_233069


namespace NUMINAMATH_CALUDE_probability_square_or_circle_l2330_233014

theorem probability_square_or_circle (total : ℕ) (squares : ℕ) (circles : ℕ) 
  (h1 : total = 10) 
  (h2 : squares = 4) 
  (h3 : circles = 3) :
  (squares + circles : ℚ) / total = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_square_or_circle_l2330_233014


namespace NUMINAMATH_CALUDE_songs_in_playlists_l2330_233027

theorem songs_in_playlists (n : ℕ) :
  ∃ (k : ℕ), n = 12 + 9 * k ↔ ∃ (m : ℕ), n = 9 * m + 3 :=
by sorry

end NUMINAMATH_CALUDE_songs_in_playlists_l2330_233027


namespace NUMINAMATH_CALUDE_outfits_count_l2330_233053

/-- The number of different outfits that can be created given a specific number of shirts, pants, and ties. --/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) : ℕ :=
  shirts * pants * (ties + 1)

/-- Theorem stating that with 8 shirts, 5 pants, and 6 ties, the number of possible outfits is 280. --/
theorem outfits_count : number_of_outfits 8 5 6 = 280 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l2330_233053


namespace NUMINAMATH_CALUDE_robin_gum_count_l2330_233079

theorem robin_gum_count (initial : Real) (additional : Real) (total : Real) : 
  initial = 18.0 → additional = 44.0 → total = initial + additional → total = 62.0 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l2330_233079


namespace NUMINAMATH_CALUDE_women_no_traits_l2330_233016

/-- Represents the number of women in the population -/
def total_population : ℕ := 200

/-- Probability of having only one specific trait -/
def prob_one_trait : ℚ := 1/20

/-- Probability of having precisely two specific traits -/
def prob_two_traits : ℚ := 2/25

/-- Probability of having all three traits, given a woman has X and Y -/
def prob_all_given_xy : ℚ := 1/4

/-- Number of women with only one trait -/
def women_one_trait : ℕ := 10

/-- Number of women with exactly two traits -/
def women_two_traits : ℕ := 16

/-- Number of women with all three traits -/
def women_all_traits : ℕ := 5

/-- Theorem stating the number of women with none of the three traits -/
theorem women_no_traits : 
  total_population - 3 * women_one_trait - 3 * women_two_traits - women_all_traits = 117 := by
  sorry

end NUMINAMATH_CALUDE_women_no_traits_l2330_233016


namespace NUMINAMATH_CALUDE_binomial_not_perfect_power_l2330_233081

theorem binomial_not_perfect_power (n k l m : ℕ) : 
  l ≥ 2 → 4 ≤ k → k ≤ n - 4 → (n.choose k) ≠ m^l := by
  sorry

end NUMINAMATH_CALUDE_binomial_not_perfect_power_l2330_233081


namespace NUMINAMATH_CALUDE_intersection_line_equation_l2330_233067

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 20

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (x1 y1 x2 y2 : ℝ),
    circle1 x1 y1 ∧ circle2 x1 y1 ∧
    circle1 x2 y2 ∧ circle2 x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) →
    line x1 y1 ∧ line x2 y2 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l2330_233067


namespace NUMINAMATH_CALUDE_shirt_pricing_l2330_233055

theorem shirt_pricing (total_shirts : ℕ) (first_shirt_price second_shirt_price : ℚ) 
  (remaining_shirts : ℕ) (min_avg_remaining : ℚ) :
  total_shirts = 6 →
  first_shirt_price = 40 →
  second_shirt_price = 50 →
  remaining_shirts = 4 →
  min_avg_remaining = 52.5 →
  (first_shirt_price + second_shirt_price + remaining_shirts * min_avg_remaining) / total_shirts = 50 := by
  sorry

end NUMINAMATH_CALUDE_shirt_pricing_l2330_233055


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_i_l2330_233056

theorem complex_expression_equals_negative_i :
  let i : ℂ := Complex.I
  (1 + 2*i) * i^3 + 2*i^2 = -i :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_i_l2330_233056


namespace NUMINAMATH_CALUDE_dannys_english_marks_l2330_233020

theorem dannys_english_marks 
  (math_marks : ℕ) 
  (physics_marks : ℕ) 
  (chemistry_marks : ℕ) 
  (biology_marks : ℕ) 
  (average_marks : ℕ) 
  (total_subjects : ℕ) 
  (h1 : math_marks = 65) 
  (h2 : physics_marks = 82) 
  (h3 : chemistry_marks = 67) 
  (h4 : biology_marks = 75) 
  (h5 : average_marks = 73) 
  (h6 : total_subjects = 5) : 
  ∃ (english_marks : ℕ), english_marks = 76 ∧ 
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / total_subjects = average_marks :=
by
  sorry

end NUMINAMATH_CALUDE_dannys_english_marks_l2330_233020


namespace NUMINAMATH_CALUDE_notebook_payment_possible_l2330_233091

theorem notebook_payment_possible : ∃ (a b : ℕ), 27 * a - 16 * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_notebook_payment_possible_l2330_233091


namespace NUMINAMATH_CALUDE_property_holds_iff_one_or_two_l2330_233006

-- Define the property for a given k
def has_property (k : ℕ) : Prop :=
  k ≥ 1 ∧
  ∀ (coloring : ℤ → Fin k),
  ∃ (a : ℕ → ℤ),
    (∀ i < 2023, a i < a (i + 1)) ∧
    (∀ i < 2023, ∃ n : ℕ, a (i + 1) - a i = 2^n) ∧
    (∀ i < 2023, coloring (a i) = coloring (a 0))

-- State the theorem
theorem property_holds_iff_one_or_two :
  ∀ k : ℕ, has_property k ↔ k = 1 ∨ k = 2 := by sorry

end NUMINAMATH_CALUDE_property_holds_iff_one_or_two_l2330_233006


namespace NUMINAMATH_CALUDE_trivia_team_total_score_l2330_233062

def trivia_team_points : Prop :=
  let total_members : ℕ := 12
  let absent_members : ℕ := 4
  let present_members : ℕ := total_members - absent_members
  let scores : List ℕ := [8, 12, 9, 5, 10, 7, 14, 11]
  scores.length = present_members ∧ scores.sum = 76

theorem trivia_team_total_score : trivia_team_points := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_total_score_l2330_233062


namespace NUMINAMATH_CALUDE_desired_depth_calculation_desired_depth_is_50_l2330_233032

/-- Calculates the desired depth to be dug given the initial and changed conditions -/
theorem desired_depth_calculation (initial_men : ℕ) (initial_hours : ℕ) (initial_depth : ℕ) 
  (extra_men : ℕ) (new_hours : ℕ) : ℕ :=
  let total_men : ℕ := initial_men + extra_men
  let initial_man_hours : ℕ := initial_men * initial_hours
  let new_man_hours : ℕ := total_men * new_hours
  let desired_depth : ℕ := (new_man_hours * initial_depth) / initial_man_hours
  desired_depth

/-- Proves that the desired depth to be dug is 50 meters -/
theorem desired_depth_is_50 : 
  desired_depth_calculation 45 8 30 55 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_desired_depth_calculation_desired_depth_is_50_l2330_233032


namespace NUMINAMATH_CALUDE_gcd_459_357_l2330_233054

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2330_233054


namespace NUMINAMATH_CALUDE_sum_of_complex_numbers_l2330_233080

-- Define the complex numbers
def z1 (a b : ℂ) : ℂ := 2 * a + b * Complex.I
def z2 (c d : ℂ) : ℂ := c + 3 * d * Complex.I
def z3 (e f : ℂ) : ℂ := e + f * Complex.I

-- State the theorem
theorem sum_of_complex_numbers (a b c d e f : ℂ) :
  b = 4 →
  e = -2 * a - c →
  z1 a b + z2 c d + z3 e f = 6 * Complex.I →
  d + f = 2 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_complex_numbers_l2330_233080


namespace NUMINAMATH_CALUDE_fred_initial_sheets_l2330_233003

def initial_sheets : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun x received given final =>
    x + received - given = final

theorem fred_initial_sheets :
  ∃ x : ℕ, initial_sheets x 307 156 363 ∧ x = 212 :=
by
  sorry

end NUMINAMATH_CALUDE_fred_initial_sheets_l2330_233003


namespace NUMINAMATH_CALUDE_no_solution_for_four_l2330_233030

theorem no_solution_for_four : 
  ∀ X : ℕ, X < 10 →
  (∀ Y : ℕ, Y < 10 → ¬(100 * X + 30 + Y) % 11 = 0) ↔ X = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_four_l2330_233030


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l2330_233046

/-- Proves that the number of adult tickets sold is 40 given the conditions of the problem -/
theorem adult_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_receipts ∧ 
    adult_tickets = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_adult_tickets_sold_l2330_233046


namespace NUMINAMATH_CALUDE_highway_intersection_probability_l2330_233005

theorem highway_intersection_probability (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  let p_enter := 1 / n
  let p_exit := 1 / n
  (k - 1) * (n - k) * p_enter * p_exit +
  p_enter * (n - k) * p_exit +
  (k - 1) * p_enter * p_exit =
  (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2 := by
  sorry


end NUMINAMATH_CALUDE_highway_intersection_probability_l2330_233005


namespace NUMINAMATH_CALUDE_rook_placements_corners_removed_8x8_l2330_233022

/-- Represents a chessboard with corners removed -/
def CornersRemovedChessboard : Type := Unit

/-- The number of ways to place non-attacking rooks on a corners-removed chessboard -/
def num_rook_placements (board : CornersRemovedChessboard) : ℕ := 21600

/-- The theorem stating the number of ways to place eight non-attacking rooks
    on an 8x8 chessboard with its four corners removed -/
theorem rook_placements_corners_removed_8x8 (board : CornersRemovedChessboard) :
  num_rook_placements board = 21600 := by sorry

end NUMINAMATH_CALUDE_rook_placements_corners_removed_8x8_l2330_233022


namespace NUMINAMATH_CALUDE_swallow_flock_max_weight_l2330_233011

/-- Represents the weight capacity of different swallow types and their quantities in a flock -/
structure SwallowFlock where
  american_capacity : ℕ
  european_capacity : ℕ
  african_capacity : ℕ
  total_swallows : ℕ
  american_count : ℕ
  european_count : ℕ
  african_count : ℕ

/-- Calculates the maximum weight a flock of swallows can carry -/
def max_carry_weight (flock : SwallowFlock) : ℕ :=
  flock.american_count * flock.american_capacity +
  flock.european_count * flock.european_capacity +
  flock.african_count * flock.african_capacity

/-- Theorem stating the maximum weight the specific flock can carry -/
theorem swallow_flock_max_weight :
  ∃ (flock : SwallowFlock),
    flock.american_capacity = 5 ∧
    flock.european_capacity = 2 * flock.american_capacity ∧
    flock.african_capacity = 3 * flock.american_capacity ∧
    flock.total_swallows = 120 ∧
    flock.american_count = 2 * flock.european_count ∧
    flock.african_count = 3 * flock.american_count ∧
    flock.american_count + flock.european_count + flock.african_count = flock.total_swallows ∧
    max_carry_weight flock = 1415 :=
  sorry

end NUMINAMATH_CALUDE_swallow_flock_max_weight_l2330_233011


namespace NUMINAMATH_CALUDE_smallest_number_of_guesses_l2330_233077

def is_determinable (guesses : List Nat) : Prop :=
  ∀ N : Nat, 1 < N → N < 100 → 
    ∃! N', 1 < N' → N' < 100 → 
      ∀ g ∈ guesses, g % N = g % N'

theorem smallest_number_of_guesses :
  ∃ guesses : List Nat,
    guesses.length = 6 ∧
    is_determinable guesses ∧
    ∀ guesses' : List Nat, guesses'.length < 6 → ¬is_determinable guesses' :=
sorry

end NUMINAMATH_CALUDE_smallest_number_of_guesses_l2330_233077


namespace NUMINAMATH_CALUDE_election_vote_difference_l2330_233041

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 7500 → 
  candidate_percentage = 35/100 → 
  (total_votes : ℚ) * candidate_percentage - (total_votes : ℚ) * (1 - candidate_percentage) = -2250 := by
  sorry

end NUMINAMATH_CALUDE_election_vote_difference_l2330_233041


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2330_233065

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_incr : is_increasing_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_diff : a 4 - a 3 = 4) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = a n * q := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2330_233065


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2330_233072

/-- A line in the xy-plane can be represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Two lines are perpendicular if the product of their slopes is -1. -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem perpendicular_line_equation (l : Line) (h1 : l.y_intercept = 1) 
    (h2 : perpendicular l (Line.mk (1/2) 0)) : 
  l.slope = -2 ∧ ∀ x y : ℝ, y = l.slope * x + l.y_intercept ↔ y = -2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2330_233072


namespace NUMINAMATH_CALUDE_fa_f_product_zero_l2330_233018

/-- Given a point F, a line l, and a circle C, prove that |FA| · |F| = 0 --/
theorem fa_f_product_zero (F : ℝ × ℝ) (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : 
  F.1 = 0 →
  l = {(x, y) : ℝ × ℝ | -Real.sqrt 3 * y = 0} →
  C = {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 22} →
  ∃ (A : ℝ × ℝ), A ∈ l ∧ (‖A - F‖ * ‖F‖ = 0) := by
  sorry

#check fa_f_product_zero

end NUMINAMATH_CALUDE_fa_f_product_zero_l2330_233018


namespace NUMINAMATH_CALUDE_triangle_area_problem_l2330_233059

theorem triangle_area_problem (A B C : Real) (a b c : Real) :
  c = 2 →
  C = π / 3 →
  let m : Real × Real := (Real.sin C + Real.sin (B - A), 4)
  let n : Real × Real := (Real.sin (2 * A), 1)
  (∃ (k : Real), m.1 = k * n.1 ∧ m.2 = k * n.2) →
  let S := (1 / 2) * a * c * Real.sin B
  S = (4 * Real.sqrt 13) / 13 ∨ S = (2 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l2330_233059


namespace NUMINAMATH_CALUDE_largest_digit_sum_l2330_233002

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_digit_sum (a b c y : ℕ) (ha : is_digit a) (hb : is_digit b) (hc : is_digit c)
  (hy : 0 < y ∧ y ≤ 15) (h_frac : (a * 100 + b * 10 + c : ℚ) / 1000 = 1 / y) :
  a + b + c ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l2330_233002


namespace NUMINAMATH_CALUDE_parallelogram_distance_l2330_233036

/-- A parallelogram with given dimensions -/
structure Parallelogram where
  side1 : ℝ  -- Length of one pair of parallel sides
  side2 : ℝ  -- Length of the other pair of parallel sides
  height1 : ℝ  -- Height corresponding to side1
  height2 : ℝ  -- Height corresponding to side2 (to be proved)

/-- Theorem stating the relationship between the dimensions of the parallelogram -/
theorem parallelogram_distance (p : Parallelogram) 
  (h1 : p.side1 = 20) 
  (h2 : p.side2 = 75) 
  (h3 : p.height1 = 60) : 
  p.height2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_distance_l2330_233036


namespace NUMINAMATH_CALUDE_angles_on_y_equals_x_l2330_233045

/-- The set of angles whose terminal side lies on the line y = x -/
def anglesOnLine : Set ℝ :=
  {α | ∃ (k : ℤ), α = k * Real.pi + Real.pi / 4}

/-- The line y = x -/
def lineYEqualsX (x : ℝ) : ℝ := x

theorem angles_on_y_equals_x :
  {α : ℝ | ∃ (x : ℝ), Real.cos α = x ∧ Real.sin α = lineYEqualsX x} = anglesOnLine := by
  sorry

end NUMINAMATH_CALUDE_angles_on_y_equals_x_l2330_233045


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l2330_233087

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m ^ 3 = n

def prime_factors (n : ℕ) : List ℕ := sorry

def satisfies_conditions (n : ℕ) : Prop :=
  n > 0 ∧
  ¬ is_prime n ∧
  ¬ is_cube n ∧
  (prime_factors n).length % 2 = 0 ∧
  ∀ p ∈ prime_factors n, p > 60

theorem smallest_satisfying_number : 
  satisfies_conditions 3721 ∧ 
  ∀ m : ℕ, m < 3721 → ¬ satisfies_conditions m :=
sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l2330_233087


namespace NUMINAMATH_CALUDE_sophists_count_l2330_233086

/-- Represents the types of inhabitants on the Isle of Logic -/
inductive Inhabitant
  | Knight
  | Liar
  | Sophist

/-- The Isle of Logic and its inhabitants -/
structure IsleOfLogic where
  knights : Nat
  liars : Nat
  sophists : Nat

/-- Predicate to check if a statement is valid for a sophist -/
def isSophistStatement (isle : IsleOfLogic) (statementLiars : Nat) : Prop :=
  statementLiars ≠ isle.liars ∧ 
  ¬(statementLiars = isle.liars + 1 ∧ isle.sophists > isle.liars)

/-- Theorem: The number of sophists on the Isle of Logic -/
theorem sophists_count (isle : IsleOfLogic) : 
  isle.knights = 40 →
  isle.liars = 25 →
  isSophistStatement isle 26 →
  isle.sophists ≤ 26 →
  isle.sophists = 27 := by
  sorry

end NUMINAMATH_CALUDE_sophists_count_l2330_233086


namespace NUMINAMATH_CALUDE_sour_count_theorem_l2330_233068

/-- Represents the number of sours of each type -/
structure SourCounts where
  cherry : ℕ
  lemon : ℕ
  orange : ℕ
  grape : ℕ

/-- Calculates the total number of sours -/
def total_sours (counts : SourCounts) : ℕ :=
  counts.cherry + counts.lemon + counts.orange + counts.grape

/-- Represents the ratio between two quantities -/
structure Ratio where
  num : ℕ
  denom : ℕ

theorem sour_count_theorem (counts : SourCounts) 
  (cherry_lemon_ratio : Ratio) (lemon_grape_ratio : Ratio) :
  counts.cherry = 32 →
  cherry_lemon_ratio = Ratio.mk 4 5 →
  counts.cherry * cherry_lemon_ratio.denom = counts.lemon * cherry_lemon_ratio.num →
  4 * (counts.cherry + counts.lemon + counts.orange) = 3 * (counts.cherry + counts.lemon) →
  lemon_grape_ratio = Ratio.mk 3 2 →
  counts.lemon * lemon_grape_ratio.denom = counts.grape * lemon_grape_ratio.num →
  total_sours counts = 123 := by
  sorry

#check sour_count_theorem

end NUMINAMATH_CALUDE_sour_count_theorem_l2330_233068


namespace NUMINAMATH_CALUDE_line_equation_represents_line_l2330_233009

/-- A line in the 2D plane defined by the equation y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The set of points (x, y) satisfying a linear equation -/
def LinePoints (l : Line) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = l.m * p.1 + l.b}

theorem line_equation_represents_line :
  ∃ (l : Line), l.m = 2 ∧ l.b = 1 ∧
  LinePoints l = {p : ℝ × ℝ | p.2 = 2 * p.1 + 1} :=
by sorry

end NUMINAMATH_CALUDE_line_equation_represents_line_l2330_233009


namespace NUMINAMATH_CALUDE_parabola_equation_l2330_233043

/-- Theorem: For a parabola y² = 2px where p > 0, if a line passing through its focus
    intersects the parabola at two points P(x₁, y₁) and Q(x₂, y₂) such that x₁ + x₂ = 2
    and |PQ| = 4, then the equation of the parabola is y² = 4x. -/
theorem parabola_equation (p : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  p > 0 →
  y₁^2 = 2*p*x₁ →
  y₂^2 = 2*p*x₂ →
  x₁ + x₂ = 2 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16 →
  ∀ x y, y^2 = 2*p*x → y^2 = 4*x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2330_233043


namespace NUMINAMATH_CALUDE_sum_of_sixth_powers_of_roots_l2330_233001

theorem sum_of_sixth_powers_of_roots (p q : ℝ) : 
  p^2 - 3*p*Real.sqrt 3 + 3 = 0 → 
  q^2 - 3*q*Real.sqrt 3 + 3 = 0 → 
  p^6 + q^6 = 99171 := by
sorry

end NUMINAMATH_CALUDE_sum_of_sixth_powers_of_roots_l2330_233001


namespace NUMINAMATH_CALUDE_expression_simplification_l2330_233057

theorem expression_simplification (a b : ℝ) (h : a / b = 1 / 3) :
  1 - (a - b) / (a - 2 * b) / ((a^2 - b^2) / (a^2 - 4 * a * b + 4 * b^2)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2330_233057


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2330_233035

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 1 = 1 →
  a 5 = 16 →
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2330_233035


namespace NUMINAMATH_CALUDE_cos_360_degrees_l2330_233061

theorem cos_360_degrees : Real.cos (2 * Real.pi) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_360_degrees_l2330_233061


namespace NUMINAMATH_CALUDE_sin_315_degrees_l2330_233024

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l2330_233024


namespace NUMINAMATH_CALUDE_same_color_probability_value_l2330_233073

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 5

def same_color_probability : ℚ :=
  (Nat.choose white_balls drawn_balls + Nat.choose black_balls drawn_balls) /
  Nat.choose total_balls drawn_balls

theorem same_color_probability_value :
  same_color_probability = 77 / 3003 := by sorry

end NUMINAMATH_CALUDE_same_color_probability_value_l2330_233073


namespace NUMINAMATH_CALUDE_optimal_station_location_l2330_233015

/-- Represents the optimal station location problem for Factory A --/
theorem optimal_station_location :
  let num_buildings : ℕ := 5
  let building_distances : List ℝ := [0, 50, 100, 150, 200]
  let worker_counts : List ℕ := [1, 2, 3, 4, 5]
  let total_workers : ℕ := worker_counts.sum
  
  -- Function to calculate total walking distance for a given station location
  let total_distance (station_location : ℝ) : ℝ :=
    List.sum (List.zipWith (fun d w => w * |station_location - d|) building_distances worker_counts)
  
  -- The optimal location minimizes the total walking distance
  ∃ (optimal_location : ℝ),
    (∀ (x : ℝ), total_distance optimal_location ≤ total_distance x) ∧
    optimal_location = 150
  := by sorry

end NUMINAMATH_CALUDE_optimal_station_location_l2330_233015


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_zero_l2330_233034

theorem ceiling_floor_sum_zero : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_zero_l2330_233034


namespace NUMINAMATH_CALUDE_count_zeros_up_to_3017_l2330_233071

/-- A function that checks if a positive integer contains the digit 0 in its base-ten representation -/
def containsZero (n : ℕ+) : Bool :=
  sorry

/-- The count of positive integers less than or equal to 3017 that contain the digit 0 -/
def countZeros : ℕ :=
  sorry

/-- Theorem stating that the count of positive integers less than or equal to 3017
    containing the digit 0 is equal to 1011 -/
theorem count_zeros_up_to_3017 : countZeros = 1011 := by
  sorry

end NUMINAMATH_CALUDE_count_zeros_up_to_3017_l2330_233071
