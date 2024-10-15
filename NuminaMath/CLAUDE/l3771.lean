import Mathlib

namespace NUMINAMATH_CALUDE_diet_soda_bottles_l3771_377192

/-- Given a grocery store inventory, calculate the number of diet soda bottles -/
theorem diet_soda_bottles (total : ℕ) (regular : ℕ) (h1 : total = 17) (h2 : regular = 9) :
  total - regular = 8 := by
  sorry

end NUMINAMATH_CALUDE_diet_soda_bottles_l3771_377192


namespace NUMINAMATH_CALUDE_rulers_equation_initial_rulers_count_l3771_377134

/-- The number of rulers initially in the drawer -/
def initial_rulers : ℕ := sorry

/-- The number of rulers added to the drawer -/
def added_rulers : ℕ := 14

/-- The final number of rulers in the drawer -/
def final_rulers : ℕ := 25

/-- Theorem stating that the initial number of rulers plus the added rulers equals the final number of rulers -/
theorem rulers_equation : initial_rulers + added_rulers = final_rulers := by sorry

/-- Theorem proving that the initial number of rulers is 11 -/
theorem initial_rulers_count : initial_rulers = 11 := by sorry

end NUMINAMATH_CALUDE_rulers_equation_initial_rulers_count_l3771_377134


namespace NUMINAMATH_CALUDE_tangent_slope_of_circle_l3771_377116

/-- Given a circle with center (1,3) and a point (4,7) on the circle,
    the slope of the line tangent to the circle at (4,7) is -3/4 -/
theorem tangent_slope_of_circle (center : ℝ × ℝ) (point : ℝ × ℝ) :
  center = (1, 3) →
  point = (4, 7) →
  (let slope_tangent := -(((point.2 - center.2) / (point.1 - center.1))⁻¹)
   slope_tangent = -3/4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_of_circle_l3771_377116


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_l3771_377109

/-- A regular polygon with exterior angles each measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18 :
  ∀ n : ℕ, 
  n > 0 → 
  (360 : ℝ) / n = 18 → 
  n = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_l3771_377109


namespace NUMINAMATH_CALUDE_linear_regression_passes_through_mean_point_l3771_377124

/-- Linear regression equation passes through the mean point -/
theorem linear_regression_passes_through_mean_point 
  (b a x_bar y_bar : ℝ) : 
  y_bar = b * x_bar + a :=
sorry

end NUMINAMATH_CALUDE_linear_regression_passes_through_mean_point_l3771_377124


namespace NUMINAMATH_CALUDE_carpet_cost_is_576_l3771_377172

/-- The total cost of carpet squares needed to cover a rectangular floor -/
def total_carpet_cost (floor_length floor_width carpet_side_length carpet_cost : ℕ) : ℕ :=
  let floor_area := floor_length * floor_width
  let carpet_area := carpet_side_length * carpet_side_length
  let num_carpets := floor_area / carpet_area
  num_carpets * carpet_cost

/-- Proof that the total cost of carpet squares for the given floor is $576 -/
theorem carpet_cost_is_576 :
  total_carpet_cost 24 64 8 24 = 576 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cost_is_576_l3771_377172


namespace NUMINAMATH_CALUDE_otimes_four_eight_l3771_377153

-- Define the operation ⊗
def otimes (a b : ℚ) : ℚ := a / b + b / a

-- Theorem statement
theorem otimes_four_eight : otimes 4 8 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_otimes_four_eight_l3771_377153


namespace NUMINAMATH_CALUDE_dinosaur_book_cost_l3771_377163

/-- The cost of a dinosaur book, given the total cost of three books and the costs of two of them. -/
theorem dinosaur_book_cost (total_cost dictionary_cost cookbook_cost : ℕ) 
  (h_total : total_cost = 37)
  (h_dict : dictionary_cost = 11)
  (h_cook : cookbook_cost = 7) :
  total_cost - dictionary_cost - cookbook_cost = 19 := by
  sorry

end NUMINAMATH_CALUDE_dinosaur_book_cost_l3771_377163


namespace NUMINAMATH_CALUDE_hypotenuse_length_of_special_triangle_l3771_377133

/-- Given a right-angled triangle with side lengths a, b, and c (where c is the hypotenuse),
    if the sum of squares of all sides is 2000 and the perimeter is 60,
    then the hypotenuse length is 10√10. -/
theorem hypotenuse_length_of_special_triangle (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a^2 + b^2 + c^2 = 2000 →
  a + b + c = 60 →
  c = 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_of_special_triangle_l3771_377133


namespace NUMINAMATH_CALUDE_total_beignets_in_16_weeks_l3771_377145

/-- The number of beignets Sandra eats each morning -/
def daily_beignets : ℕ := 3

/-- The number of weeks we're considering -/
def weeks : ℕ := 16

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating the total number of beignets Sandra will eat in 16 weeks -/
theorem total_beignets_in_16_weeks : 
  daily_beignets * days_per_week * weeks = 336 := by
  sorry

end NUMINAMATH_CALUDE_total_beignets_in_16_weeks_l3771_377145


namespace NUMINAMATH_CALUDE_max_value_parabola_l3771_377183

theorem max_value_parabola :
  (∀ x : ℝ, -x^2 + 5 ≤ 5) ∧ (∃ x : ℝ, -x^2 + 5 = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_parabola_l3771_377183


namespace NUMINAMATH_CALUDE_double_root_condition_l3771_377185

/-- 
For a quadratic equation ax^2 + bx + c = 0, if one root is double the other, 
then 2b^2 = 9ac.
-/
theorem double_root_condition (a b c : ℝ) (x₁ x₂ : ℝ) : 
  a ≠ 0 → 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) → 
  x₂ = 2 * x₁ → 
  2 * b^2 = 9 * a * c := by
sorry

end NUMINAMATH_CALUDE_double_root_condition_l3771_377185


namespace NUMINAMATH_CALUDE_system_solution_l3771_377194

theorem system_solution : 
  ∃! (x y : ℝ), (2 * x + y = 6) ∧ (x - y = 3) ∧ (x = 3) ∧ (y = 0) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3771_377194


namespace NUMINAMATH_CALUDE_unique_value_sum_l3771_377169

/-- Given that {a, b, c} = {0, 1, 2} and exactly one of (a ≠ 2), (b = 2), (c ≠ 0) is true,
    prove that a + 2b + 5c = 7 -/
theorem unique_value_sum (a b c : ℤ) : 
  ({a, b, c} : Set ℤ) = {0, 1, 2} →
  ((a ≠ 2) ∨ (b = 2) ∨ (c ≠ 0)) ∧
  (¬((a ≠ 2) ∧ (b = 2)) ∧ ¬((a ≠ 2) ∧ (c ≠ 0)) ∧ ¬((b = 2) ∧ (c ≠ 0))) →
  a + 2*b + 5*c = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_value_sum_l3771_377169


namespace NUMINAMATH_CALUDE_chess_tournament_players_l3771_377130

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- Number of players not in the lowest 15
  -- Each player plays exactly one match against every other player
  total_games : ℕ := (n + 15).choose 2
  -- Points from games between n players not in the lowest 15
  points_among_n : ℕ := n.choose 2
  -- Points earned by n players against the lowest 15
  points_n_vs_15 : ℕ := n.choose 2
  -- Points earned by the lowest 15 players among themselves
  points_among_15 : ℕ := 105
  -- Total points in the tournament
  total_points : ℕ := 2 * points_among_n + 2 * points_among_15

/-- The theorem stating that the total number of players in the tournament is 50 -/
theorem chess_tournament_players (t : ChessTournament) : t.n + 15 = 50 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l3771_377130


namespace NUMINAMATH_CALUDE_simplify_expression_l3771_377114

theorem simplify_expression (x : ℝ) : (3*x)^5 + (4*x^2)*(3*x^2) = 243*x^5 + 12*x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3771_377114


namespace NUMINAMATH_CALUDE_impossible_to_divide_into_l_pieces_l3771_377139

/-- Represents a chessboard cell --/
inductive Cell
| Black
| White

/-- Represents an L-shaped piece --/
structure LPiece :=
(cells : Fin 4 → Cell)

/-- Represents a chessboard --/
def Chessboard := Fin 8 → Fin 8 → Cell

/-- Returns the color of a cell based on its coordinates --/
def cellColor (row col : Fin 8) : Cell :=
  if (row.val + col.val) % 2 = 0 then Cell.Black else Cell.White

/-- Checks if a cell is in the central 2x2 square --/
def isCentralSquare (row col : Fin 8) : Prop :=
  (row = 3 ∨ row = 4) ∧ (col = 3 ∨ col = 4)

/-- Represents the modified chessboard with central 2x2 square removed --/
def ModifiedChessboard : Type :=
  { cell : Fin 8 × Fin 8 // ¬isCentralSquare cell.1 cell.2 }

/-- The main theorem stating that it's impossible to divide the modified chessboard into L-shaped pieces --/
theorem impossible_to_divide_into_l_pieces :
  ¬∃ (pieces : List LPiece), 
    (pieces.length > 0) ∧ 
    (∀ (cell : ModifiedChessboard), ∃! (piece : LPiece) (i : Fin 4), 
      piece ∈ pieces ∧ piece.cells i = cellColor cell.val.1 cell.val.2) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_divide_into_l_pieces_l3771_377139


namespace NUMINAMATH_CALUDE_regression_line_at_25_l3771_377161

/-- The regression line equation is y = 0.5x - 0.81 -/
def regression_line (x : ℝ) : ℝ := 0.5 * x - 0.81

/-- Theorem: Given the regression line equation y = 0.5x - 0.81, when x = 25, y = 11.69 -/
theorem regression_line_at_25 : regression_line 25 = 11.69 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_at_25_l3771_377161


namespace NUMINAMATH_CALUDE_farmer_apples_l3771_377186

theorem farmer_apples (initial_apples given_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : given_apples = 88) :
  initial_apples - given_apples = 39 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l3771_377186


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l3771_377162

def a (n : ℕ+) : ℕ := 2 * n.val - 1

theorem sum_of_specific_terms : 
  a 4 + a 5 + a 6 + a 7 + a 8 = 55 := by sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l3771_377162


namespace NUMINAMATH_CALUDE_binomial_30_3_l3771_377171

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l3771_377171


namespace NUMINAMATH_CALUDE_inverse_of_B_squared_l3771_377144

def B_inv : Matrix (Fin 2) (Fin 2) ℤ := !![1, 4; -2, -7]

theorem inverse_of_B_squared :
  let B_squared_inv : Matrix (Fin 2) (Fin 2) ℤ := !![(-7), (-24); 12, 41]
  (B_inv * B_inv) * (B_inv⁻¹ * B_inv⁻¹) = 1 ∧ (B_inv⁻¹ * B_inv⁻¹) * (B_inv * B_inv) = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_squared_l3771_377144


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3771_377104

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 2) > (5/x) + (21/10)) ↔ (-2 < x ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3771_377104


namespace NUMINAMATH_CALUDE_cos_2017pi_over_3_l3771_377198

theorem cos_2017pi_over_3 : Real.cos (2017 * Real.pi / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2017pi_over_3_l3771_377198


namespace NUMINAMATH_CALUDE_prob_sum_three_is_half_l3771_377167

/-- A fair coin toss outcome -/
inductive CoinToss
  | heads
  | tails

/-- The numeric value associated with a coin toss -/
def coinValue (t : CoinToss) : ℕ :=
  match t with
  | CoinToss.heads => 1
  | CoinToss.tails => 2

/-- The sample space of two coin tosses -/
def sampleSpace : List (CoinToss × CoinToss) :=
  [(CoinToss.heads, CoinToss.heads),
   (CoinToss.heads, CoinToss.tails),
   (CoinToss.tails, CoinToss.heads),
   (CoinToss.tails, CoinToss.tails)]

/-- The event where the sum of two coin tosses is 3 -/
def sumThreeEvent (t : CoinToss × CoinToss) : Bool :=
  coinValue t.1 + coinValue t.2 = 3

/-- Theorem: The probability of obtaining a sum of 3 when tossing a fair coin twice is 1/2 -/
theorem prob_sum_three_is_half :
  (sampleSpace.filter sumThreeEvent).length / sampleSpace.length = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_three_is_half_l3771_377167


namespace NUMINAMATH_CALUDE_sum_of_digits_eight_to_hundred_l3771_377148

theorem sum_of_digits_eight_to_hundred (n : ℕ) (h : n = 8^100) : 
  (n % 100 / 10 + n % 10) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_eight_to_hundred_l3771_377148


namespace NUMINAMATH_CALUDE_simplify_expression_l3771_377190

theorem simplify_expression (x y : ℝ) : (3 * x + 22) + (150 * y + 22) = 3 * x + 150 * y + 44 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3771_377190


namespace NUMINAMATH_CALUDE_math_club_team_selection_l3771_377182

def math_club_selection (boys girls : ℕ) (team_size : ℕ) (team_boys team_girls : ℕ) : ℕ :=
  Nat.choose boys team_boys * Nat.choose girls team_girls

theorem math_club_team_selection :
  math_club_selection 7 9 6 4 2 = 1260 :=
by sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l3771_377182


namespace NUMINAMATH_CALUDE_patricia_books_count_l3771_377147

/-- Given the number of books read by Candice, calculate the number of books read by Patricia -/
def books_read_by_patricia (candice_books : ℕ) : ℕ :=
  let amanda_books := candice_books / 3
  let kara_books := amanda_books / 2
  7 * kara_books

/-- Theorem stating that if Candice read 18 books, Patricia read 21 books -/
theorem patricia_books_count (h : books_read_by_patricia 18 = 21) : 
  books_read_by_patricia 18 = 21 := by
  sorry

#eval books_read_by_patricia 18

end NUMINAMATH_CALUDE_patricia_books_count_l3771_377147


namespace NUMINAMATH_CALUDE_gnomes_in_fifth_house_l3771_377188

theorem gnomes_in_fifth_house 
  (total_houses : ℕ)
  (gnomes_per_house : ℕ)
  (houses_with_known_gnomes : ℕ)
  (total_gnomes : ℕ)
  (h1 : total_houses = 5)
  (h2 : gnomes_per_house = 3)
  (h3 : houses_with_known_gnomes = 4)
  (h4 : total_gnomes = 20) :
  total_gnomes - (houses_with_known_gnomes * gnomes_per_house) = 8 :=
by
  sorry

#check gnomes_in_fifth_house

end NUMINAMATH_CALUDE_gnomes_in_fifth_house_l3771_377188


namespace NUMINAMATH_CALUDE_white_bread_served_l3771_377151

/-- Given that a restaurant served 0.5 loaf of wheat bread and a total of 0.9 loaves,
    prove that 0.4 loaves of white bread were served. -/
theorem white_bread_served (wheat_bread : ℝ) (total_bread : ℝ) (white_bread : ℝ)
    (h1 : wheat_bread = 0.5)
    (h2 : total_bread = 0.9)
    (h3 : white_bread = total_bread - wheat_bread) :
    white_bread = 0.4 := by
  sorry

#check white_bread_served

end NUMINAMATH_CALUDE_white_bread_served_l3771_377151


namespace NUMINAMATH_CALUDE_ratio_equality_l3771_377155

theorem ratio_equality (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_eq1 : (x + z) / (2 * z - x) = x / y)
  (h_eq2 : (z + 2 * y) / (2 * x - z) = x / y) : 
  x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l3771_377155


namespace NUMINAMATH_CALUDE_equation_equivalence_l3771_377174

theorem equation_equivalence (x y : ℝ) 
  (hx : x ≠ 0 ∧ x ≠ 5) (hy : y ≠ 0 ∧ y ≠ 7) : 
  (3 / x + 2 / y = 1 / 3) ↔ (x = 9 * y / (y - 6)) :=
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3771_377174


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l3771_377159

theorem relationship_between_exponents 
  (a b c d : ℝ) (x y q z : ℝ) 
  (h1 : a^(x+1) = c^(q+2)) 
  (h2 : a^(x+1) = b) 
  (h3 : c^(y+3) = a^(z+4)) 
  (h4 : c^(y+3) = d) 
  (h5 : a ≠ 0) 
  (h6 : c ≠ 0) : 
  (q+2)*(z+4) = (y+3)*(x+1) := by
sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l3771_377159


namespace NUMINAMATH_CALUDE_yellow_ball_packs_l3771_377132

theorem yellow_ball_packs (red_packs green_packs balls_per_pack total_balls : ℕ) 
  (h1 : red_packs = 3)
  (h2 : green_packs = 8)
  (h3 : balls_per_pack = 19)
  (h4 : total_balls = 399) :
  ∃ yellow_packs : ℕ, 
    yellow_packs * balls_per_pack + red_packs * balls_per_pack + green_packs * balls_per_pack = total_balls ∧
    yellow_packs = 10 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_packs_l3771_377132


namespace NUMINAMATH_CALUDE_remainder_sum_l3771_377195

theorem remainder_sum (D : ℕ) (h1 : D > 0) (h2 : 242 % D = 4) (h3 : 698 % D = 8) :
  (242 + 698) % D = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3771_377195


namespace NUMINAMATH_CALUDE_banana_ratio_l3771_377164

/-- Theorem about the ratio of bananas in Raj's basket to bananas eaten -/
theorem banana_ratio (initial_bananas : ℕ) (bananas_left_on_tree : ℕ) (bananas_eaten : ℕ) :
  initial_bananas = 310 →
  bananas_left_on_tree = 100 →
  bananas_eaten = 70 →
  (initial_bananas - bananas_left_on_tree - bananas_eaten) / bananas_eaten = 2 := by
  sorry


end NUMINAMATH_CALUDE_banana_ratio_l3771_377164


namespace NUMINAMATH_CALUDE_cos_2x_at_min_y_l3771_377120

theorem cos_2x_at_min_y (x : ℝ) : 
  let y := 2 * (Real.sin x)^6 + (Real.cos x)^6
  (∀ z : ℝ, y ≤ 2 * (Real.sin z)^6 + (Real.cos z)^6) →
  Real.cos (2 * x) = 3 - 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cos_2x_at_min_y_l3771_377120


namespace NUMINAMATH_CALUDE_f_less_than_g_implies_a_bound_l3771_377113

open Real

/-- The function f parameterized by a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (2*a + 1) * x + 2 * log x

/-- The function g -/
def g (x : ℝ) : ℝ := x^2 - 2*x

/-- The theorem statement -/
theorem f_less_than_g_implies_a_bound 
  (h : ∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Ioo 0 2, f a x₁ < g x₂) : 
  a > log 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_less_than_g_implies_a_bound_l3771_377113


namespace NUMINAMATH_CALUDE_unique_multiplication_solution_l3771_377118

/-- Represents a three-digit number in the form abb --/
def three_digit (a b : Nat) : Nat := 100 * a + 10 * b + b

/-- Represents a four-digit number in the form bcb1 --/
def four_digit (b c : Nat) : Nat := 1000 * b + 100 * c + 10 * b + 1

theorem unique_multiplication_solution :
  ∃! (a b c : Nat),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    three_digit a b * c = four_digit b c ∧
    a = 5 ∧ b = 3 ∧ c = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_multiplication_solution_l3771_377118


namespace NUMINAMATH_CALUDE_triangle_inequality_l3771_377138

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3771_377138


namespace NUMINAMATH_CALUDE_symmetry_implies_exponent_l3771_377115

theorem symmetry_implies_exponent (a b : ℝ) : 
  (2 * a + 1 = 1 ∧ -3 * a = -(3 - b)) → b^a = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_exponent_l3771_377115


namespace NUMINAMATH_CALUDE_intersection_equals_N_l3771_377105

def M : Set ℝ := {x | x ≤ 1}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem intersection_equals_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_intersection_equals_N_l3771_377105


namespace NUMINAMATH_CALUDE_parabola_equation_l3771_377141

/-- A parabola with vertex at the origin, axis of symmetry along the x-axis,
    and passing through the point (-2, 2√2) has the equation y^2 = -4x. -/
theorem parabola_equation (p : ℝ × ℝ) 
    (vertex_origin : p.1 = 0 ∧ p.2 = 0)
    (axis_x : ∀ (x y : ℝ), y^2 = -4*x → y^2 = -4*(-x))
    (point_on_parabola : (-2)^2 + (2*Real.sqrt 2)^2 = -4*(-2)) :
  ∀ (x y : ℝ), y^2 = -4*x ↔ (x, y) ∈ {(a, b) | b^2 = -4*a} :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3771_377141


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l3771_377177

theorem mans_rowing_speed (river_speed : ℝ) (round_trip_time : ℝ) (total_distance : ℝ) (still_water_speed : ℝ) : 
  river_speed = 2 →
  round_trip_time = 1 →
  total_distance = 5.333333333333333 →
  still_water_speed = 7.333333333333333 →
  (total_distance / 2) / (round_trip_time / 2) = still_water_speed - river_speed ∧
  (total_distance / 2) / (round_trip_time / 2) = still_water_speed + river_speed :=
by sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_l3771_377177


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3771_377197

theorem quadratic_factorization (y a b : ℤ) :
  2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b) →
  a - b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3771_377197


namespace NUMINAMATH_CALUDE_event_attendance_l3771_377149

theorem event_attendance (total : ℕ) (movie picnic gaming : ℕ) 
  (movie_picnic movie_gaming picnic_gaming : ℕ) (all_three : ℕ) 
  (h1 : total = 200)
  (h2 : movie = 50)
  (h3 : picnic = 80)
  (h4 : gaming = 60)
  (h5 : movie_picnic = 35)
  (h6 : movie_gaming = 10)
  (h7 : picnic_gaming = 20)
  (h8 : all_three = 8) :
  movie + picnic + gaming - (movie_picnic + movie_gaming + picnic_gaming) + all_three = 133 := by
sorry

end NUMINAMATH_CALUDE_event_attendance_l3771_377149


namespace NUMINAMATH_CALUDE_yankees_to_mets_ratio_l3771_377128

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The total number of baseball fans in the town -/
def total_fans : ℕ := 330

/-- The given number of NY Mets fans -/
def mets_fans : ℕ := 88

/-- Theorem stating that the ratio of NY Yankees fans to NY Mets fans is 3:2 -/
theorem yankees_to_mets_ratio (fc : FanCounts) : 
  fc.yankees = fc.mets * 3 / 2 ∧ 
  fc.mets = mets_fans ∧ 
  fc.red_sox = fc.mets * 5 / 4 ∧ 
  fc.yankees + fc.mets + fc.red_sox = total_fans :=
by sorry

end NUMINAMATH_CALUDE_yankees_to_mets_ratio_l3771_377128


namespace NUMINAMATH_CALUDE_incorrect_inequality_l3771_377176

theorem incorrect_inequality (x y : ℝ) (h : x > y) : ¬(-3 * x > -3 * y) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l3771_377176


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_t_1_l3771_377165

-- Define the position function S(t)
def S (t : ℝ) : ℝ := 2 * t^2 + t

-- Define the velocity function as the derivative of S(t)
def v (t : ℝ) : ℝ := 4 * t + 1

-- Theorem statement
theorem instantaneous_velocity_at_t_1 : v 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_t_1_l3771_377165


namespace NUMINAMATH_CALUDE_area_between_line_and_curve_l3771_377175

/-- The area enclosed by the line y=4x and the curve y=x^3 is 8 -/
theorem area_between_line_and_curve : 
  let f (x : ℝ) := 4 * x
  let g (x : ℝ) := x^3
  ∫ x in (-2)..2, |f x - g x| = 8 := by sorry

end NUMINAMATH_CALUDE_area_between_line_and_curve_l3771_377175


namespace NUMINAMATH_CALUDE_correct_equation_l3771_377158

theorem correct_equation (y : ℝ) : -9 * y^2 + 16 * y^2 = 7 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l3771_377158


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l3771_377123

theorem inequality_holds_iff (a : ℝ) : 
  (∀ x : ℝ, (4:ℝ)^(x^2) + 2*(2*a+1) * (2:ℝ)^(x^2) + 4*a^2 - 3 > 0) ↔ 
  (a < -1 ∨ a ≥ Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l3771_377123


namespace NUMINAMATH_CALUDE_store_price_calculation_l3771_377102

/-- If an item's online price is 300 yuan and it's 20% less than the store price,
    then the store price is 375 yuan. -/
theorem store_price_calculation (online_price store_price : ℝ) : 
  online_price = 300 →
  online_price = store_price - 0.2 * store_price →
  store_price = 375 := by
  sorry

end NUMINAMATH_CALUDE_store_price_calculation_l3771_377102


namespace NUMINAMATH_CALUDE_sum_of_cuboid_vertices_l3771_377196

/-- Given that the sum of edges and faces of all cuboids is 216, 
    prove that the sum of vertices of all cuboids is 96. -/
theorem sum_of_cuboid_vertices (n : ℕ) : 
  n * (12 + 6) = 216 → n * 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cuboid_vertices_l3771_377196


namespace NUMINAMATH_CALUDE_marble_difference_l3771_377193

theorem marble_difference (jar1_blue jar1_red jar2_blue jar2_red : ℕ) :
  jar1_blue + jar1_red = jar2_blue + jar2_red →
  7 * jar1_red = 3 * jar1_blue →
  3 * jar2_red = 2 * jar2_blue →
  jar1_red + jar2_red = 80 →
  jar1_blue - jar2_blue = 80 / 7 := by
sorry

end NUMINAMATH_CALUDE_marble_difference_l3771_377193


namespace NUMINAMATH_CALUDE_total_height_increase_four_centuries_l3771_377121

/-- Represents the increase in height per decade for a specific plant species -/
def height_increase_per_decade : ℝ := 75

/-- Represents the number of decades in 4 centuries -/
def decades_in_four_centuries : ℕ := 40

/-- Theorem: The total increase in height over 4 centuries is 3000 meters -/
theorem total_height_increase_four_centuries : 
  height_increase_per_decade * (decades_in_four_centuries : ℝ) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_total_height_increase_four_centuries_l3771_377121


namespace NUMINAMATH_CALUDE_sunzi_deer_problem_l3771_377122

/-- The number of deer that enter the city -/
def total_deer : ℕ := 100

/-- The number of families in the city -/
def num_families : ℕ := 75

theorem sunzi_deer_problem :
  (num_families : ℚ) + (1 / 3 : ℚ) * num_families = total_deer :=
by sorry

end NUMINAMATH_CALUDE_sunzi_deer_problem_l3771_377122


namespace NUMINAMATH_CALUDE_quadratic_function_monotonicity_l3771_377166

-- Define the function
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem quadratic_function_monotonicity :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → f x₁ > f x₂) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_monotonicity_l3771_377166


namespace NUMINAMATH_CALUDE_circle_center_quadrant_l3771_377125

theorem circle_center_quadrant (α : Real) :
  (∃ x y : Real, x^2 * Real.cos α - y^2 * Real.sin α + 2 = 0) →  -- hyperbola condition
  let center := (- Real.cos α, Real.sin α)
  (center.1 < 0 ∧ center.2 > 0) ∨ (center.1 > 0 ∧ center.2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_quadrant_l3771_377125


namespace NUMINAMATH_CALUDE_road_trip_gas_cost_l3771_377146

/-- Calculates the total cost of filling a car's gas tank at multiple stations -/
def total_gas_cost (tank_capacity : ℝ) (gas_prices : List ℝ) : ℝ :=
  (gas_prices.map (· * tank_capacity)).sum

/-- Proves that the total cost of filling a 12-gallon tank at 4 stations with given prices is $180 -/
theorem road_trip_gas_cost : 
  let tank_capacity : ℝ := 12
  let gas_prices : List ℝ := [3, 3.5, 4, 4.5]
  total_gas_cost tank_capacity gas_prices = 180 := by
  sorry

#eval total_gas_cost 12 [3, 3.5, 4, 4.5]

end NUMINAMATH_CALUDE_road_trip_gas_cost_l3771_377146


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3771_377111

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2 * x₁ + 3 = 0 ∧ a * x₂^2 + 2 * x₂ + 3 = 0) →
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3771_377111


namespace NUMINAMATH_CALUDE_boats_first_meeting_distance_l3771_377168

/-- Two boats traveling across a river, meeting twice without stopping at shores -/
structure BoatMeeting where
  /-- Total distance between shore A and B in yards -/
  total_distance : ℝ
  /-- Distance from shore B to the second meeting point in yards -/
  second_meeting_distance : ℝ
  /-- Distance from shore A to the first meeting point in yards -/
  first_meeting_distance : ℝ

/-- Theorem stating that the boats first meet at 300 yards from shore A -/
theorem boats_first_meeting_distance (meeting : BoatMeeting)
    (h1 : meeting.total_distance = 1200)
    (h2 : meeting.second_meeting_distance = 300) :
    meeting.first_meeting_distance = 300 := by
  sorry


end NUMINAMATH_CALUDE_boats_first_meeting_distance_l3771_377168


namespace NUMINAMATH_CALUDE_solve_tank_problem_l3771_377101

def tank_problem (initial_capacity : ℝ) (leak_rate1 leak_rate2 fill_rate : ℝ)
  (leak_duration1 leak_duration2 fill_duration : ℝ) (missing_amount : ℝ) : Prop :=
  let total_loss := leak_rate1 * leak_duration1 + leak_rate2 * leak_duration2
  let remaining_after_loss := initial_capacity - total_loss
  let current_amount := initial_capacity - missing_amount
  let amount_added := current_amount - remaining_after_loss
  fill_rate = amount_added / fill_duration

theorem solve_tank_problem :
  tank_problem 350000 32000 10000 40000 5 10 3 140000 := by
  sorry

end NUMINAMATH_CALUDE_solve_tank_problem_l3771_377101


namespace NUMINAMATH_CALUDE_bags_difference_l3771_377103

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 8

/-- The number of bags Tiffany found the next day -/
def next_day_bags : ℕ := 7

/-- Theorem stating the difference in bags between Monday and the next day -/
theorem bags_difference : monday_bags - next_day_bags = 1 := by
  sorry

end NUMINAMATH_CALUDE_bags_difference_l3771_377103


namespace NUMINAMATH_CALUDE_min_value_theorem_l3771_377129

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 100) + (y + 1/x) * (y + 1/x - 100) ≥ -2500 ∧
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + 1/x₀ = 50 ∧
    (x₀ + 1/x₀) * (x₀ + 1/x₀ - 100) + (x₀ + 1/x₀) * (x₀ + 1/x₀ - 100) = -2500) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3771_377129


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l3771_377136

theorem sqrt_product_simplification (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (8 * x) = 60 * x * Real.sqrt (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l3771_377136


namespace NUMINAMATH_CALUDE_max_value_d_l3771_377108

theorem max_value_d (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (product_condition : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_d_l3771_377108


namespace NUMINAMATH_CALUDE_cabbage_area_is_one_sq_foot_l3771_377152

/-- Represents the cabbage garden problem --/
structure CabbageGarden where
  area_this_year : ℕ
  area_last_year : ℕ
  cabbages_this_year : ℕ
  cabbages_last_year : ℕ

/-- The area per cabbage is 1 square foot --/
theorem cabbage_area_is_one_sq_foot (garden : CabbageGarden)
  (h1 : garden.area_this_year = garden.cabbages_this_year)
  (h2 : garden.area_last_year = garden.cabbages_last_year)
  (h3 : garden.cabbages_this_year = 4096)
  (h4 : garden.cabbages_last_year = 3969)
  (h5 : ∃ n : ℕ, garden.area_this_year = n * n)
  (h6 : ∃ m : ℕ, garden.area_last_year = m * m) :
  garden.area_this_year / garden.cabbages_this_year = 1 := by
  sorry

#check cabbage_area_is_one_sq_foot

end NUMINAMATH_CALUDE_cabbage_area_is_one_sq_foot_l3771_377152


namespace NUMINAMATH_CALUDE_centers_on_line_l3771_377135

-- Define the family of circles
def circle_family (k : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + (4*k + 10)*y + 10*k + 20 = 0

-- Define the line equation
def center_line (x y : ℝ) : Prop :=
  2*x - y - 5 = 0

-- Theorem statement
theorem centers_on_line :
  ∀ k : ℝ, k ≠ -1 →
  ∃ x y : ℝ, circle_family k x y ∧ center_line x y :=
sorry

end NUMINAMATH_CALUDE_centers_on_line_l3771_377135


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3771_377180

theorem profit_percentage_calculation (cost_price selling_price : ℝ) :
  cost_price = 620 →
  selling_price = 775 →
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3771_377180


namespace NUMINAMATH_CALUDE_damaged_manuscript_multiplication_l3771_377143

theorem damaged_manuscript_multiplication : ∃ (x y : ℕ), 
  x > 0 ∧ y > 0 ∧
  10 ≤ x * 8 ∧ x * 8 < 100 ∧
  100 ≤ x * (y / 10) ∧ x * (y / 10) < 1000 ∧
  y % 10 = 8 ∧
  x * y = 1176 := by
sorry

end NUMINAMATH_CALUDE_damaged_manuscript_multiplication_l3771_377143


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_in_cones_l3771_377112

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- The maximum squared radius of a sphere that can fit within two intersecting cones -/
def maxSphereRadiusSquared (ic : IntersectingCones) : ℝ :=
  sorry

/-- Theorem stating the maximum squared radius of a sphere in the given configuration -/
theorem max_sphere_radius_squared_in_cones :
  let ic : IntersectingCones := {
    cone1 := { baseRadius := 4, height := 10 },
    cone2 := { baseRadius := 4, height := 10 },
    intersectionDistance := 4
  }
  maxSphereRadiusSquared ic = 144 / 29 := by
  sorry

end NUMINAMATH_CALUDE_max_sphere_radius_squared_in_cones_l3771_377112


namespace NUMINAMATH_CALUDE_candle_flower_groupings_l3771_377156

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem candle_flower_groupings :
  (choose 6 3) * (choose 15 12) = 9100 := by
sorry

end NUMINAMATH_CALUDE_candle_flower_groupings_l3771_377156


namespace NUMINAMATH_CALUDE_sequence_general_term_l3771_377173

/-- The sequence defined by a₁ = -1 and aₙ₊₁ = 3aₙ - 1 has the general term aₙ = -(3ⁿ - 1)/2 -/
theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = -1) 
    (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n - 1) : 
    ∀ n : ℕ, n ≥ 1 → a n = -(3^n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3771_377173


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_l3771_377184

theorem quadratic_equations_common_root (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_common_root1 : ∃! x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + b*x + c = 0)
  (h_common_root2 : ∃! x : ℝ, x^2 + b*x + c = 0 ∧ x^2 + c*x + a = 0)
  (h_common_root3 : ∃! x : ℝ, x^2 + c*x + a = 0 ∧ x^2 + a*x + b = 0) :
  a^2 + b^2 + c^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_l3771_377184


namespace NUMINAMATH_CALUDE_triangle_inequality_l3771_377110

theorem triangle_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  1 / (b + c - a) + 1 / (c + a - b) + 1 / (a + b - c) > 9 / (a + b + c) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3771_377110


namespace NUMINAMATH_CALUDE_workers_payment_schedule_l3771_377131

theorem workers_payment_schedule (total_days : ℕ) (pay_per_day_worked : ℤ) (pay_returned_per_day_not_worked : ℤ) 
  (h1 : total_days = 30)
  (h2 : pay_per_day_worked = 100)
  (h3 : pay_returned_per_day_not_worked = 25)
  (h4 : ∃ (days_worked days_not_worked : ℕ), 
    days_worked + days_not_worked = total_days ∧ 
    pay_per_day_worked * days_worked - pay_returned_per_day_not_worked * days_not_worked = 0) :
  ∃ (days_not_worked : ℕ), days_not_worked = 24 := by
sorry

end NUMINAMATH_CALUDE_workers_payment_schedule_l3771_377131


namespace NUMINAMATH_CALUDE_stating_failed_both_percentage_l3771_377189

/-- Represents the percentage of students in various categories -/
structure ExamResults where
  failed_hindi : ℝ
  failed_english : ℝ
  passed_both : ℝ

/-- 
Calculates the percentage of students who failed in both Hindi and English
given the exam results.
-/
def percentage_failed_both (results : ExamResults) : ℝ :=
  results.failed_hindi + results.failed_english - (100 - results.passed_both)

/-- 
Theorem stating that given the specific exam results, 
the percentage of students who failed in both subjects is 27%.
-/
theorem failed_both_percentage 
  (results : ExamResults)
  (h1 : results.failed_hindi = 25)
  (h2 : results.failed_english = 48)
  (h3 : results.passed_both = 54) :
  percentage_failed_both results = 27 := by
  sorry

#eval percentage_failed_both ⟨25, 48, 54⟩

end NUMINAMATH_CALUDE_stating_failed_both_percentage_l3771_377189


namespace NUMINAMATH_CALUDE_work_earnings_equality_l3771_377137

theorem work_earnings_equality (t : ℚ) : 
  (t + 2) * (4*t - 2) = (4*t - 7) * (t + 1) + 4 → t = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equality_l3771_377137


namespace NUMINAMATH_CALUDE_derivative_of_f_l3771_377179

noncomputable def f (x : ℝ) : ℝ := (2 * Real.pi * x)^2

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 8 * Real.pi^2 * x := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3771_377179


namespace NUMINAMATH_CALUDE_james_singing_lessons_l3771_377107

/-- Calculates the number of singing lessons James gets given the conditions --/
def number_of_lessons (lesson_cost : ℕ) (james_payment : ℕ) : ℕ :=
  let total_cost := james_payment * 2
  let initial_paid_lessons := 10
  let remaining_cost := total_cost - (initial_paid_lessons * lesson_cost)
  let additional_paid_lessons := remaining_cost / (lesson_cost * 2)
  1 + initial_paid_lessons + additional_paid_lessons

/-- Theorem stating that James gets 13 singing lessons --/
theorem james_singing_lessons :
  number_of_lessons 5 35 = 13 := by
  sorry


end NUMINAMATH_CALUDE_james_singing_lessons_l3771_377107


namespace NUMINAMATH_CALUDE_exists_increasing_perfect_squares_sequence_l3771_377100

theorem exists_increasing_perfect_squares_sequence : 
  ∃ (a : ℕ → ℕ), 
    (∀ k : ℕ, k > 0 → ∃ n : ℕ, a k = n ^ 2) ∧ 
    (∀ k : ℕ, k > 0 → a k < a (k + 1)) ∧
    (∀ k : ℕ, k > 0 → (13 ^ k) ∣ (a k + 1)) := by
  sorry

end NUMINAMATH_CALUDE_exists_increasing_perfect_squares_sequence_l3771_377100


namespace NUMINAMATH_CALUDE_gold_tetrahedron_volume_l3771_377126

/-- Represents a cube with alternately colored vertices -/
structure ColoredCube where
  sideLength : ℝ
  vertexColors : Fin 8 → Bool  -- True for gold, False for red

/-- Calculates the volume of a tetrahedron formed by selected vertices of a cube -/
def tetrahedronVolume (cube : ColoredCube) (selectVertex : Fin 8 → Bool) : ℝ :=
  sorry

/-- The main theorem stating the volume of the gold-colored tetrahedron -/
theorem gold_tetrahedron_volume (cube : ColoredCube) 
  (h1 : cube.sideLength = 8)
  (h2 : ∀ i : Fin 8, cube.vertexColors i = (i.val % 2 == 0))  -- Alternating colors
  : tetrahedronVolume cube cube.vertexColors = 170.67 := by
  sorry

end NUMINAMATH_CALUDE_gold_tetrahedron_volume_l3771_377126


namespace NUMINAMATH_CALUDE_b_months_is_nine_l3771_377160

/-- Represents the pasture rental scenario -/
structure PastureRental where
  total_cost : ℝ
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  c_horses : ℕ
  c_months : ℕ
  b_payment : ℝ

/-- Theorem stating that given the conditions, b put in horses for 9 months -/
theorem b_months_is_nine (pr : PastureRental)
  (h1 : pr.total_cost = 435)
  (h2 : pr.a_horses = 12)
  (h3 : pr.a_months = 8)
  (h4 : pr.b_horses = 16)
  (h5 : pr.c_horses = 18)
  (h6 : pr.c_months = 6)
  (h7 : pr.b_payment = 180) :
  ∃ x : ℝ, x = 9 ∧ 
    pr.b_payment = (pr.total_cost / (pr.a_horses * pr.a_months + pr.b_horses * x + pr.c_horses * pr.c_months)) * (pr.b_horses * x) :=
by sorry


end NUMINAMATH_CALUDE_b_months_is_nine_l3771_377160


namespace NUMINAMATH_CALUDE_game_packing_l3771_377157

theorem game_packing (initial_games : Nat) (sold_games : Nat) (games_per_box : Nat) :
  initial_games = 35 →
  sold_games = 19 →
  games_per_box = 8 →
  (initial_games - sold_games) / games_per_box = 2 := by
  sorry

end NUMINAMATH_CALUDE_game_packing_l3771_377157


namespace NUMINAMATH_CALUDE_dyck_path_correspondence_l3771_377142

/-- A Dyck path is a lattice path of upsteps and downsteps that starts at the origin and never dips below the x-axis. -/
def DyckPath (n : ℕ) : Type := sorry

/-- A return in a Dyck path is a maximal sequence of contiguous downsteps that terminates on the x-axis. -/
def Return (path : DyckPath n) : Type := sorry

/-- Predicate to check if a return has even length -/
def hasEvenLengthReturn (path : DyckPath n) : Prop := sorry

/-- The number of Dyck n-paths -/
def numDyckPaths (n : ℕ) : ℕ := sorry

/-- The number of Dyck n-paths with no return of even length -/
def numDyckPathsNoEvenReturn (n : ℕ) : ℕ := sorry

/-- Theorem: The number of Dyck n-paths with no return of even length is equal to the number of Dyck (n-1) paths -/
theorem dyck_path_correspondence (n : ℕ) (h : n ≥ 1) :
  numDyckPathsNoEvenReturn n = numDyckPaths (n - 1) := by sorry

end NUMINAMATH_CALUDE_dyck_path_correspondence_l3771_377142


namespace NUMINAMATH_CALUDE_decreasing_linear_function_l3771_377191

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f y < f x

theorem decreasing_linear_function (k : ℝ) :
  is_decreasing (λ x : ℝ => (k + 1) * x) → k < -1 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_l3771_377191


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_quadratic_coefficients_l3771_377178

theorem quadratic_equation_conversion (x : ℝ) : 
  (x^2 - 8*x = 10) ↔ (x^2 - 8*x - 10 = 0) :=
by sorry

theorem quadratic_coefficients :
  ∃ (a b c : ℝ), (∀ x, x^2 - 8*x - 10 = 0 ↔ a*x^2 + b*x + c = 0) ∧ 
  a = 1 ∧ b = -8 ∧ c = -10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_quadratic_coefficients_l3771_377178


namespace NUMINAMATH_CALUDE_prob_units_digit_8_is_3_16_l3771_377154

def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

def units_digit (n : ℕ) : ℕ := n % 10

def prob_units_digit_8 : ℚ :=
  (Finset.filter (fun (a, b) => units_digit (3^a + 7^b) = 8) (Finset.product (Finset.range 100) (Finset.range 100))).card /
  (Finset.product (Finset.range 100) (Finset.range 100)).card

theorem prob_units_digit_8_is_3_16 : prob_units_digit_8 = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_units_digit_8_is_3_16_l3771_377154


namespace NUMINAMATH_CALUDE_jeans_purchase_savings_l3771_377199

/-- Calculates the total savings on a purchase with multiple discounts and a rebate --/
theorem jeans_purchase_savings 
  (original_price : ℝ)
  (sale_discount_percent : ℝ)
  (coupon_discount : ℝ)
  (credit_card_discount_percent : ℝ)
  (voucher_discount_percent : ℝ)
  (rebate : ℝ)
  (sales_tax_percent : ℝ)
  (h1 : original_price = 200)
  (h2 : sale_discount_percent = 30)
  (h3 : coupon_discount = 15)
  (h4 : credit_card_discount_percent = 15)
  (h5 : voucher_discount_percent = 10)
  (h6 : rebate = 20)
  (h7 : sales_tax_percent = 8.25) :
  ∃ (savings : ℝ), abs (savings - 116.49) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_jeans_purchase_savings_l3771_377199


namespace NUMINAMATH_CALUDE_water_drainage_proof_l3771_377140

/-- Represents the fraction of water remaining after n steps of draining -/
def waterRemaining (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

/-- The number of steps after which one-seventh of the water remains -/
def stepsToOneSeventh : ℕ := 12

theorem water_drainage_proof :
  waterRemaining stepsToOneSeventh = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_water_drainage_proof_l3771_377140


namespace NUMINAMATH_CALUDE_problem_solution_l3771_377106

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.cos y = 3005)
  (h2 : x + 3005 * Real.sin y = 3004)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 3004 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3771_377106


namespace NUMINAMATH_CALUDE_small_cup_volume_l3771_377181

theorem small_cup_volume (small_cup : ℝ) (large_container : ℝ) : 
  (8 * small_cup + 5400 = large_container) →
  (12 * 530 = large_container) →
  small_cup = 120 := by
sorry

end NUMINAMATH_CALUDE_small_cup_volume_l3771_377181


namespace NUMINAMATH_CALUDE_units_digit_of_product_units_digit_of_27_times_34_l3771_377170

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of a product depends only on the units digits of its factors -/
theorem units_digit_of_product (a b : ℕ) : 
  unitsDigit (a * b) = unitsDigit (unitsDigit a * unitsDigit b) := by sorry

theorem units_digit_of_27_times_34 : unitsDigit (27 * 34) = 8 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_units_digit_of_27_times_34_l3771_377170


namespace NUMINAMATH_CALUDE_trig_function_amplitude_l3771_377187

theorem trig_function_amplitude 
  (y : ℝ → ℝ) 
  (a b c d : ℝ) 
  (h1 : ∀ x, y x = a * Real.cos (b * x + c) + d) 
  (h2 : ∃ x, y x = 4) 
  (h3 : ∃ x, y x = 0) 
  (h4 : ∀ x, y x ≤ 4) 
  (h5 : ∀ x, y x ≥ 0) : 
  a = 2 := by sorry

end NUMINAMATH_CALUDE_trig_function_amplitude_l3771_377187


namespace NUMINAMATH_CALUDE_distance_between_points_l3771_377117

def point1 : ℝ × ℝ := (-5, 3)
def point2 : ℝ × ℝ := (6, -9)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 265 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3771_377117


namespace NUMINAMATH_CALUDE_existence_of_complementary_sequences_l3771_377119

def s (x y : ℝ) : Set ℕ := {s | ∃ n : ℕ, s = ⌊n * x + y⌋}

theorem existence_of_complementary_sequences (r : ℚ) (hr : r > 1) :
  ∃ u v : ℝ, (s r 0 ∩ s u v = ∅) ∧ (s r 0 ∪ s u v = Set.univ) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_complementary_sequences_l3771_377119


namespace NUMINAMATH_CALUDE_quadratic_root_bound_l3771_377127

theorem quadratic_root_bound (a b c : ℤ) (h_distinct : ∃ (x y : ℝ), x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) (h_pos : 0 < a) : 5 ≤ a := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_bound_l3771_377127


namespace NUMINAMATH_CALUDE_license_plate_count_l3771_377150

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 5

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of possible positions for the letter block -/
def block_positions : ℕ := digits_count + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  block_positions * (num_digits ^ digits_count) * (num_letters ^ letters_count)

theorem license_plate_count : total_license_plates = 105456000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3771_377150
