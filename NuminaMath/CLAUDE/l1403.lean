import Mathlib

namespace equal_earnings_l1403_140391

theorem equal_earnings (t : ℝ) : 
  (t - 4) * (3 * t - 7) = (3 * t - 12) * (t - 3) → t = 4 := by
  sorry

end equal_earnings_l1403_140391


namespace functional_equation_bound_l1403_140348

/-- Given real-valued functions f and g defined on ℝ satisfying certain conditions,
    prove that |g(y)| ≤ 1 for all y. -/
theorem functional_equation_bound (f g : ℝ → ℝ)
  (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∀ x, f x ≠ 0)
  (h3 : ∀ x, |f x| ≤ 1) :
  ∀ y, |g y| ≤ 1 := by
  sorry

end functional_equation_bound_l1403_140348


namespace harriet_miles_l1403_140344

theorem harriet_miles (total_miles : ℕ) (katarina_miles : ℕ) 
  (h1 : total_miles = 195)
  (h2 : katarina_miles = 51)
  (h3 : ∃ x : ℕ, x * 3 + katarina_miles = total_miles) :
  ∃ harriet_miles : ℕ, harriet_miles = 48 ∧ 
    harriet_miles * 3 + katarina_miles = total_miles := by
  sorry

end harriet_miles_l1403_140344


namespace smallest_two_digit_number_one_more_than_multiple_l1403_140341

theorem smallest_two_digit_number_one_more_than_multiple (n : ℕ) : n = 71 ↔ 
  (n ≥ 10 ∧ n < 100) ∧ 
  (∃ k : ℕ, n = 2 * k + 1 ∧ n = 5 * k + 1 ∧ n = 7 * k + 1) ∧
  (∀ m : ℕ, m < n → ¬(m ≥ 10 ∧ m < 100 ∧ ∃ k : ℕ, m = 2 * k + 1 ∧ m = 5 * k + 1 ∧ m = 7 * k + 1)) :=
by sorry

end smallest_two_digit_number_one_more_than_multiple_l1403_140341


namespace unique_solution_xy_l1403_140336

theorem unique_solution_xy (x y : ℝ) 
  (h1 : x^2 + y^2 = 2)
  (h2 : x^2 / (2 - y) + y^2 / (2 - x) = 2) :
  x = 1 ∧ y = 1 :=
by sorry

end unique_solution_xy_l1403_140336


namespace smallest_mu_inequality_l1403_140316

theorem smallest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∀ μ : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + 4*b^2 + 4*c^2 + d^2 ≥ 2*a*b + μ*b*c + 2*c*d) → μ ≥ 6) ∧
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + 4*b^2 + 4*c^2 + d^2 ≥ 2*a*b + 6*b*c + 2*c*d) :=
by sorry

end smallest_mu_inequality_l1403_140316


namespace solve_equation_l1403_140319

theorem solve_equation (x : ℝ) (h : x - 2*x + 3*x = 100) : x = 50 := by
  sorry

end solve_equation_l1403_140319


namespace highway_traffic_l1403_140390

/-- The number of vehicles involved in accidents per 100 million vehicles -/
def accident_rate : ℕ := 96

/-- The total number of vehicles involved in accidents last year -/
def total_accidents : ℕ := 2880

/-- The number of vehicles (in billions) that traveled on the highway last year -/
def vehicles_traveled : ℕ := 3

theorem highway_traffic :
  vehicles_traveled * 1000000000 = (total_accidents * 100000000) / accident_rate := by
  sorry

end highway_traffic_l1403_140390


namespace donation_theorem_l1403_140328

def donation_problem (total_donation : ℚ) 
  (community_pantry_fraction : ℚ) 
  (crisis_fund_fraction : ℚ) 
  (contingency_amount : ℚ) : Prop :=
  let remaining := total_donation - (community_pantry_fraction * total_donation) - (crisis_fund_fraction * total_donation)
  let livelihood_amount := remaining - contingency_amount
  livelihood_amount / remaining = 1 / 4

theorem donation_theorem : 
  donation_problem 240 (1/3) (1/2) 30 := by
  sorry

end donation_theorem_l1403_140328


namespace garden_tilling_time_l1403_140347

/-- Represents a rectangular obstacle in the garden -/
structure Obstacle where
  length : ℝ
  width : ℝ

/-- Represents the garden plot and tilling parameters -/
structure GardenPlot where
  shortBase : ℝ
  longBase : ℝ
  height : ℝ
  tillerWidth : ℝ
  tillingRate : ℝ
  obstacles : List Obstacle
  extraTimePerObstacle : ℝ

/-- Calculates the time required to till the garden plot -/
def tillingTime (plot : GardenPlot) : ℝ :=
  sorry

/-- Theorem stating the tilling time for the given garden plot -/
theorem garden_tilling_time :
  let plot : GardenPlot := {
    shortBase := 135,
    longBase := 170,
    height := 90,
    tillerWidth := 2.5,
    tillingRate := 1.5 / 3,
    obstacles := [
      { length := 20, width := 10 },
      { length := 15, width := 30 },
      { length := 10, width := 15 }
    ],
    extraTimePerObstacle := 15
  }
  abs (tillingTime plot - 173.08) < 0.01 := by
  sorry

end garden_tilling_time_l1403_140347


namespace brand_y_pen_price_l1403_140302

theorem brand_y_pen_price 
  (price_x : ℝ) 
  (total_pens : ℕ) 
  (total_cost : ℝ) 
  (num_x_pens : ℕ) 
  (h1 : price_x = 4)
  (h2 : total_pens = 12)
  (h3 : total_cost = 42)
  (h4 : num_x_pens = 6) :
  (total_cost - price_x * num_x_pens) / (total_pens - num_x_pens) = 3 := by
  sorry

end brand_y_pen_price_l1403_140302


namespace cody_game_expense_l1403_140384

theorem cody_game_expense (initial_amount birthday_gift final_amount : ℕ) 
  (h1 : initial_amount = 45)
  (h2 : birthday_gift = 9)
  (h3 : final_amount = 35) :
  initial_amount + birthday_gift - final_amount = 19 :=
by sorry

end cody_game_expense_l1403_140384


namespace books_remaining_correct_l1403_140311

/-- Calculates the number of books remaining on the shelf by the evening. -/
def books_remaining (initial : ℕ) (borrowed_lunch : ℕ) (added : ℕ) (borrowed_evening : ℕ) : ℕ :=
  initial - borrowed_lunch + added - borrowed_evening

/-- Proves that the number of books remaining on the shelf by the evening is correct. -/
theorem books_remaining_correct (initial : ℕ) (borrowed_lunch : ℕ) (added : ℕ) (borrowed_evening : ℕ)
    (h1 : initial = 100)
    (h2 : borrowed_lunch = 50)
    (h3 : added = 40)
    (h4 : borrowed_evening = 30) :
    books_remaining initial borrowed_lunch added borrowed_evening = 60 := by
  sorry

end books_remaining_correct_l1403_140311


namespace algebraic_simplification_l1403_140358

theorem algebraic_simplification (m n : ℝ) : 3 * m^2 * n - 3 * m^2 * n = 0 := by
  sorry

end algebraic_simplification_l1403_140358


namespace walking_distance_l1403_140346

/-- Proves that given a walking speed where 1 mile is covered in 20 minutes, 
    the distance covered in 40 minutes is 2 miles. -/
theorem walking_distance (speed : ℝ) (time : ℝ) : 
  speed = 1 / 20 → time = 40 → speed * time = 2 := by
  sorry

end walking_distance_l1403_140346


namespace box_length_calculation_l1403_140305

/-- The length of a cubic box given total volume, cost per box, and total cost -/
theorem box_length_calculation (total_volume : ℝ) (cost_per_box : ℝ) (total_cost : ℝ) :
  total_volume = 1080000 ∧ cost_per_box = 0.8 ∧ total_cost = 480 →
  ∃ (length : ℝ), abs (length - (total_volume / (total_cost / cost_per_box))^(1/3)) < 0.1 := by
  sorry

end box_length_calculation_l1403_140305


namespace school_students_count_l1403_140359

theorem school_students_count : ∃! n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧ 
  n % 6 = 1 ∧ n % 8 = 2 ∧ n % 9 = 3 ∧ n = 265 := by
  sorry

end school_students_count_l1403_140359


namespace inequality_contradiction_l1403_140356

theorem inequality_contradiction (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) := by
  sorry

end inequality_contradiction_l1403_140356


namespace reflection_across_x_axis_l1403_140333

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflection_across_x_axis :
  let P : ℝ × ℝ := (-2, 1)
  reflect_x P = (-2, -1) := by sorry

end reflection_across_x_axis_l1403_140333


namespace number_decrease_proof_l1403_140362

theorem number_decrease_proof (x v : ℝ) : 
  x > 0 → x = 7 → x - v = 21 * (1/x) → v = 4 := by
  sorry

end number_decrease_proof_l1403_140362


namespace percent_decrease_proof_l1403_140379

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 60) : 
  (original_price - sale_price) / original_price * 100 = 40 := by
  sorry

end percent_decrease_proof_l1403_140379


namespace log_equation_solution_l1403_140367

noncomputable def LogEquation (a b x : ℝ) : Prop :=
  5 * (Real.log x / Real.log b) ^ 2 + 2 * (Real.log x / Real.log a) ^ 2 = 10 * (Real.log x) ^ 2 / (Real.log a * Real.log b)

theorem log_equation_solution (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  LogEquation a b x → (b = a ^ (1 + Real.sqrt 15 / 5) ∨ b = a ^ (1 - Real.sqrt 15 / 5)) :=
by sorry

end log_equation_solution_l1403_140367


namespace gcd_of_specific_numbers_l1403_140334

theorem gcd_of_specific_numbers : Nat.gcd 333333 9999999 = 3 := by sorry

end gcd_of_specific_numbers_l1403_140334


namespace cubic_symmetry_l1403_140326

/-- A cubic function of the form ax^3 + bx + 6 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 6

/-- Theorem: For a cubic function f(x) = ax^3 + bx + 6, if f(5) = 7, then f(-5) = 5 -/
theorem cubic_symmetry (a b : ℝ) : f a b 5 = 7 → f a b (-5) = 5 := by
  sorry

end cubic_symmetry_l1403_140326


namespace initial_kola_percentage_l1403_140350

/-- Proves that the initial percentage of concentrated kola in a solution is 6% -/
theorem initial_kola_percentage (
  initial_volume : ℝ)
  (initial_water_percentage : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_kola : ℝ)
  (final_sugar_percentage : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percentage = 80)
  (h3 : added_sugar = 3.2)
  (h4 : added_water = 10)
  (h5 : added_kola = 6.8)
  (h6 : final_sugar_percentage = 14.111111111111112)
  : ∃ (initial_kola_percentage : ℝ),
    initial_kola_percentage = 6 ∧
    (initial_volume - initial_water_percentage / 100 * initial_volume - initial_kola_percentage / 100 * initial_volume + added_sugar) /
    (initial_volume + added_sugar + added_water + added_kola) =
    final_sugar_percentage / 100 :=
by sorry

end initial_kola_percentage_l1403_140350


namespace perpendicular_lines_l1403_140394

/-- Two lines ax+y-1=0 and x-y+3=0 are perpendicular if and only if a = 1 -/
theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, (a*x + y = 1 ∧ x - y = -3) → 
   ((-a) * 1 = -1)) ↔ a = 1 :=
by sorry

end perpendicular_lines_l1403_140394


namespace min_additional_teddy_bears_l1403_140392

def teddy_bears : ℕ := 37
def row_size : ℕ := 8

theorem min_additional_teddy_bears :
  let next_multiple := ((teddy_bears + row_size - 1) / row_size) * row_size
  next_multiple - teddy_bears = 3 := by
sorry

end min_additional_teddy_bears_l1403_140392


namespace closest_integer_to_sqrt_40_l1403_140329

theorem closest_integer_to_sqrt_40 :
  ∃ (n : ℤ), n = 6 ∧ ∀ (m : ℤ), |m - Real.sqrt 40| ≥ |n - Real.sqrt 40| :=
by sorry

end closest_integer_to_sqrt_40_l1403_140329


namespace characterization_of_valid_n_l1403_140396

def floor_sqrt (n : ℕ) : ℕ := Nat.sqrt n

def is_valid (n : ℕ) : Prop :=
  (n > 0) ∧
  (∃ k₁ : ℕ, n - 4 = k₁ * (floor_sqrt n - 2)) ∧
  (∃ k₂ : ℕ, n + 4 = k₂ * (floor_sqrt n + 2))

def special_set : Set ℕ := {2, 4, 11, 20, 31, 36, 44}

def general_form (a : ℕ) : ℕ := a^2 + 2*a - 4

theorem characterization_of_valid_n :
  ∀ n : ℕ, is_valid n ↔ (n ∈ special_set ∨ ∃ a : ℕ, a > 2 ∧ n = general_form a) :=
sorry

end characterization_of_valid_n_l1403_140396


namespace reciprocal_of_opposite_negative_two_thirds_l1403_140369

theorem reciprocal_of_opposite_negative_two_thirds :
  (-(- (2 : ℚ) / 3))⁻¹ = 3 / 2 := by sorry

end reciprocal_of_opposite_negative_two_thirds_l1403_140369


namespace olivers_earnings_theorem_l1403_140352

/-- Calculates the earnings of Oliver's laundry shop over three days -/
def olivers_earnings (price_per_kilo : ℝ) (day1_kilos : ℝ) (day2_increase : ℝ) : ℝ :=
  let day2_kilos := day1_kilos + day2_increase
  let day3_kilos := 2 * day2_kilos
  let total_kilos := day1_kilos + day2_kilos + day3_kilos
  price_per_kilo * total_kilos

/-- Theorem stating that Oliver's earnings for three days equal $70 -/
theorem olivers_earnings_theorem :
  olivers_earnings 2 5 5 = 70 := by
  sorry

#eval olivers_earnings 2 5 5

end olivers_earnings_theorem_l1403_140352


namespace matthew_hotdogs_l1403_140365

/-- The number of hotdogs Ella wants -/
def ella_hotdogs : ℕ := 2

/-- The number of hotdogs Emma wants -/
def emma_hotdogs : ℕ := 2

/-- The number of hotdogs Luke wants -/
def luke_hotdogs : ℕ := 2 * (ella_hotdogs + emma_hotdogs)

/-- The number of hotdogs Hunter wants -/
def hunter_hotdogs : ℕ := (3 * (ella_hotdogs + emma_hotdogs)) / 2

/-- The total number of hotdogs Matthew needs to cook -/
def total_hotdogs : ℕ := ella_hotdogs + emma_hotdogs + luke_hotdogs + hunter_hotdogs

theorem matthew_hotdogs : total_hotdogs = 14 := by
  sorry

end matthew_hotdogs_l1403_140365


namespace product_diversity_l1403_140351

theorem product_diversity (n k : ℕ+) :
  ∃ (m : ℕ), m = n + k - 2 ∧
  ∀ (A : Finset ℝ) (B : Finset ℝ),
    A.card = k ∧ B.card = n →
    (A.product B).card ≥ m ∧
    ∀ (m' : ℕ), m' > m →
      ∃ (A' : Finset ℝ) (B' : Finset ℝ),
        A'.card = k ∧ B'.card = n ∧ (A'.product B').card < m' :=
by sorry

end product_diversity_l1403_140351


namespace min_value_inequality_l1403_140349

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 5) :
  (9 / x) + (25 / y) + (49 / z) ≥ 45 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 5 ∧ (9 / x) + (25 / y) + (49 / z) = 45 :=
by sorry

end min_value_inequality_l1403_140349


namespace expression_equality_l1403_140376

theorem expression_equality : -1^4 + (-2)^3 / 4 * (5 - (-3)^2) = 7 := by
  sorry

end expression_equality_l1403_140376


namespace range_of_trigonometric_function_l1403_140388

theorem range_of_trigonometric_function :
  ∀ x : ℝ, 0 ≤ Real.cos x ^ 4 + Real.cos x * Real.sin x + Real.sin x ^ 4 ∧
           Real.cos x ^ 4 + Real.cos x * Real.sin x + Real.sin x ^ 4 ≤ 5 / 4 := by
  sorry

end range_of_trigonometric_function_l1403_140388


namespace second_player_wins_l1403_140332

/-- Represents the state of the game -/
structure GameState where
  grid_size : Nat
  piece1_pos : Nat
  piece2_pos : Nat

/-- Defines a valid move in the game -/
inductive Move
  | One
  | Two

/-- Applies a move to a game state -/
def apply_move (state : GameState) (player : Nat) (move : Move) : GameState :=
  match player, move with
  | 1, Move.One => { state with piece1_pos := state.piece1_pos + 1 }
  | 1, Move.Two => { state with piece1_pos := state.piece1_pos + 2 }
  | 2, Move.One => { state with piece2_pos := state.piece2_pos - 1 }
  | 2, Move.Two => { state with piece2_pos := state.piece2_pos - 2 }
  | _, _ => state

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (player : Nat) (move : Move) : Prop :=
  match player, move with
  | 1, Move.One => state.piece1_pos + 1 < state.piece2_pos
  | 1, Move.Two => state.piece1_pos + 2 < state.piece2_pos
  | 2, Move.One => state.piece1_pos < state.piece2_pos - 1
  | 2, Move.Two => state.piece1_pos < state.piece2_pos - 2
  | _, _ => False

/-- Checks if the game is over -/
def is_game_over (state : GameState) : Prop :=
  state.piece2_pos - state.piece1_pos <= 1

/-- Checks if the number of empty squares between pieces is a multiple of 3 -/
def is_multiple_of_three (state : GameState) : Prop :=
  (state.piece2_pos - state.piece1_pos - 1) % 3 = 0

/-- Theorem: The second player has a winning strategy if and only if
    the number of empty squares between pieces is always a multiple of 3
    after the second player's move -/
theorem second_player_wins (initial_state : GameState)
  (h_initial : initial_state.grid_size = 20 ∧
               initial_state.piece1_pos = 1 ∧
               initial_state.piece2_pos = 20) :
  (∀ (game_state : GameState),
   ∀ (move1 : Move),
   is_valid_move game_state 1 move1 →
   ∃ (move2 : Move),
   is_valid_move (apply_move game_state 1 move1) 2 move2 ∧
   is_multiple_of_three (apply_move (apply_move game_state 1 move1) 2 move2)) ↔
  (∃ (strategy : GameState → Move),
   ∀ (game_state : GameState),
   ¬is_game_over game_state →
   is_valid_move game_state 2 (strategy game_state) ∧
   is_multiple_of_three (apply_move game_state 2 (strategy game_state))) :=
sorry

end second_player_wins_l1403_140332


namespace rectangle_perimeter_l1403_140398

theorem rectangle_perimeter (x y : ℝ) 
  (h1 : 6 * x + 2 * y = 56)  -- perimeter of figure A
  (h2 : 4 * x + 6 * y = 56)  -- perimeter of figure B
  : 2 * x + 6 * y = 40 :=    -- perimeter of figure C
by sorry

end rectangle_perimeter_l1403_140398


namespace cube_sum_minus_product_eq_2003_l1403_140301

theorem cube_sum_minus_product_eq_2003 : 
  {(x, y, z) : ℤ × ℤ × ℤ | x^3 + y^3 + z^3 - 3*x*y*z = 2003} = 
  {(668, 668, 667), (668, 667, 668), (667, 668, 668)} := by
sorry

end cube_sum_minus_product_eq_2003_l1403_140301


namespace families_with_items_l1403_140374

theorem families_with_items (total_telephone : ℕ) (total_tricycle : ℕ) (both : ℕ)
  (h1 : total_telephone = 35)
  (h2 : total_tricycle = 65)
  (h3 : both = 20) :
  total_telephone + total_tricycle - both = 80 := by
  sorry

end families_with_items_l1403_140374


namespace min_value_theorem_l1403_140386

theorem min_value_theorem (a b : ℝ) (h1 : a + b = 2) (h2 : b > 0) :
  (∃ (x : ℝ), x = 1 / (2 * abs a) + abs a / b ∧
    (∀ (y : ℝ), y = 1 / (2 * abs a) + abs a / b → x ≤ y)) →
  (∃ (min_val : ℝ), min_val = 3/4 ∧
    (∃ (x : ℝ), x = 1 / (2 * abs a) + abs a / b ∧ x = min_val) ∧
    a = -2) :=
by sorry

end min_value_theorem_l1403_140386


namespace remainder_sum_mod_nine_l1403_140331

theorem remainder_sum_mod_nine (a b c : ℕ) : 
  0 < a ∧ a < 10 ∧
  0 < b ∧ b < 10 ∧
  0 < c ∧ c < 10 ∧
  (a * b * c) % 9 = 1 ∧
  (4 * c) % 9 = 5 ∧
  (7 * b) % 9 = (4 + b) % 9 →
  (a + b + c) % 9 = 8 := by
sorry

end remainder_sum_mod_nine_l1403_140331


namespace parabola_f_value_l1403_140377

/-- A parabola with equation x = dy² + ey + f, vertex at (5, 3), and passing through (2, 6) -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ
  vertex_condition : 5 = d * 3^2 + e * 3 + f
  point_condition : 2 = d * 6^2 + e * 6 + f

/-- The value of f for the given parabola is 2 -/
theorem parabola_f_value (p : Parabola) : p.f = 2 := by
  sorry


end parabola_f_value_l1403_140377


namespace sum_of_coefficients_l1403_140380

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (k : ℚ), k * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
    (a * Real.sqrt 6 + b * Real.sqrt 8) / c) →
  (∀ (x y z : ℕ+), 
    (∃ (l : ℚ), l * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
      (x * Real.sqrt 6 + y * Real.sqrt 8) / z) → 
    c ≤ z) →
  a + b + c = 30 := by
sorry

end sum_of_coefficients_l1403_140380


namespace subset_implies_a_range_l1403_140353

-- Define the sets S and P
def S : Set ℝ := {x | x^2 - 3*x - 10 < 0}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2*a + 15}

-- Theorem statement
theorem subset_implies_a_range (a : ℝ) : S ⊆ P a → a ∈ Set.Icc (-5) (-3) := by
  sorry

end subset_implies_a_range_l1403_140353


namespace no_integer_solutions_l1403_140357

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := by
sorry

end no_integer_solutions_l1403_140357


namespace fraction_equality_sum_l1403_140395

theorem fraction_equality_sum (M N : ℚ) :
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + 2 * N = 330 := by
  sorry

end fraction_equality_sum_l1403_140395


namespace min_positive_period_sin_l1403_140339

/-- The minimum positive period of the function y = 3 * sin(2x + π/4) is π -/
theorem min_positive_period_sin (x : ℝ) : 
  let f := fun x => 3 * Real.sin (2 * x + π / 4)
  ∃ p : ℝ, p > 0 ∧ (∀ x, f (x + p) = f x) ∧ 
    ∀ q : ℝ, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q :=
by sorry

end min_positive_period_sin_l1403_140339


namespace midpoint_vector_equation_l1403_140389

/-- Given two points P₁ and P₂ in ℝ², prove that the point P satisfying 
    the vector equation P₁P - PP₂ = 0 has coordinates (1, 1) -/
theorem midpoint_vector_equation (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (-1, 2) → P₂ = (3, 0) → (P.1 - P₁.1, P.2 - P₁.2) = (P₂.1 - P.1, P₂.2 - P.2) → 
  P = (1, 1) := by
sorry

end midpoint_vector_equation_l1403_140389


namespace jade_transactions_l1403_140309

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + (mabel * 10 / 100) →
  cal = anthony * 2 / 3 →
  jade = cal + 15 →
  jade = 81 := by
sorry

end jade_transactions_l1403_140309


namespace mikes_music_store_spending_l1403_140340

/-- The amount Mike spent on the trumpet -/
def trumpet_cost : ℚ := 145.16

/-- The amount Mike spent on the song book -/
def song_book_cost : ℚ := 5.84

/-- The total amount Mike spent at the music store -/
def total_spent : ℚ := trumpet_cost + song_book_cost

/-- Theorem stating that the total amount Mike spent is $151.00 -/
theorem mikes_music_store_spending :
  total_spent = 151.00 := by sorry

end mikes_music_store_spending_l1403_140340


namespace brick_width_calculation_l1403_140335

/-- Calculates the width of a brick given the wall dimensions, brick dimensions, and number of bricks --/
theorem brick_width_calculation (wall_length wall_height wall_thickness : ℝ)
                                (brick_length brick_height : ℝ)
                                (num_bricks : ℕ) :
  wall_length = 800 →
  wall_height = 600 →
  wall_thickness = 22.5 →
  brick_length = 125 →
  brick_height = 6 →
  num_bricks = 1280 →
  ∃ (brick_width : ℝ),
    brick_width = 11.25 ∧
    wall_length * wall_height * wall_thickness =
    num_bricks * brick_length * brick_width * brick_height :=
by
  sorry

#check brick_width_calculation

end brick_width_calculation_l1403_140335


namespace power_calculation_l1403_140327

theorem power_calculation : 3^15 * 9^5 / 27^6 = 3^7 := by
  sorry

end power_calculation_l1403_140327


namespace inequality_proof_l1403_140364

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 2015) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤ (Real.sqrt a + Real.sqrt b + Real.sqrt c) / Real.sqrt 2015 :=
by sorry

end inequality_proof_l1403_140364


namespace harry_hours_worked_l1403_140370

/-- Represents the payment structure and hours worked for an employee -/
structure Employee where
  baseHours : ℕ  -- Number of hours paid at base rate
  baseRate : ℝ   -- Base hourly rate
  overtimeRate : ℝ  -- Overtime hourly rate
  hoursWorked : ℕ  -- Total hours worked

/-- Calculates the total pay for an employee -/
def totalPay (e : Employee) : ℝ :=
  let baseAmount := min e.hoursWorked e.baseHours * e.baseRate
  let overtimeHours := max (e.hoursWorked - e.baseHours) 0
  baseAmount + overtimeHours * e.overtimeRate

/-- The main theorem to prove -/
theorem harry_hours_worked 
  (x : ℝ) 
  (harry : Employee) 
  (james : Employee) :
  harry.baseHours = 12 ∧ 
  harry.baseRate = x ∧ 
  harry.overtimeRate = 1.5 * x ∧
  james.baseHours = 40 ∧ 
  james.baseRate = x ∧ 
  james.overtimeRate = 2 * x ∧
  james.hoursWorked = 41 ∧
  totalPay harry = totalPay james →
  harry.hoursWorked = 32 := by
  sorry


end harry_hours_worked_l1403_140370


namespace angle_measure_proof_l1403_140323

theorem angle_measure_proof (AOB BOC : Real) : 
  AOB + BOC = 180 →  -- adjacent supplementary angles
  AOB = BOC + 18 →   -- AOB is 18° larger than BOC
  AOB = 99 :=        -- prove that AOB is 99°
by sorry

end angle_measure_proof_l1403_140323


namespace patch_net_profit_l1403_140375

/-- Calculates the net profit from selling patches --/
theorem patch_net_profit (order_quantity : ℕ) (cost_per_patch : ℚ) (sell_price : ℚ) : 
  order_quantity = 100 ∧ cost_per_patch = 125/100 ∧ sell_price = 12 →
  (sell_price * order_quantity) - (cost_per_patch * order_quantity) = 1075 := by
  sorry

end patch_net_profit_l1403_140375


namespace system_solution_l1403_140373

theorem system_solution (x y z w : ℝ) : 
  (x - y + z - w = 2) ∧
  (x^2 - y^2 + z^2 - w^2 = 6) ∧
  (x^3 - y^3 + z^3 - w^3 = 20) ∧
  (x^4 - y^4 + z^4 - w^4 = 60) →
  ((x = 1 ∧ y = 2 ∧ z = 3 ∧ w = 0) ∨
   (x = 1 ∧ y = 0 ∧ z = 3 ∧ w = 2) ∨
   (x = 3 ∧ y = 2 ∧ z = 1 ∧ w = 0) ∨
   (x = 3 ∧ y = 0 ∧ z = 1 ∧ w = 2)) := by
  sorry

end system_solution_l1403_140373


namespace max_shaded_area_trapezoid_l1403_140300

/-- Given a trapezoid ABCD with bases of length a and b, and area 1,
    the maximum area of the shaded region formed by moving points on the bases is ab / (a+b)^2 -/
theorem max_shaded_area_trapezoid (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let trapezoid_area : ℝ := 1
  ∃ (max_area : ℝ), max_area = a * b / (a + b)^2 ∧
    ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b →
      (x * y / ((a + b) * (x + y)) + (a - x) * (b - y) / ((a + b) * (a + b - x - y))) / (a + b) ≤ max_area :=
sorry

end max_shaded_area_trapezoid_l1403_140300


namespace bike_clamps_theorem_l1403_140372

/-- The number of bike clamps given away with each bicycle sale -/
def clamps_per_bike : ℕ := 2

/-- The number of bikes sold in the morning -/
def morning_sales : ℕ := 19

/-- The number of bikes sold in the afternoon -/
def afternoon_sales : ℕ := 27

/-- The total number of bike clamps given away -/
def total_clamps : ℕ := clamps_per_bike * (morning_sales + afternoon_sales)

theorem bike_clamps_theorem : total_clamps = 92 := by
  sorry

end bike_clamps_theorem_l1403_140372


namespace at_least_one_nonnegative_l1403_140383

theorem at_least_one_nonnegative (x y z : ℝ) :
  max (x^2 + y + 1/4) (max (y^2 + z + 1/4) (z^2 + x + 1/4)) ≥ 0 := by
  sorry

end at_least_one_nonnegative_l1403_140383


namespace power_product_rule_l1403_140304

theorem power_product_rule (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end power_product_rule_l1403_140304


namespace f_odd_implies_b_one_f_monotone_increasing_l1403_140363

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := Real.log (Real.sqrt (4 * x^2 + b) + 2 * x) / Real.log 2

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem f_odd_implies_b_one (b : ℝ) :
  is_odd (f b) → b = 1 := by sorry

theorem f_monotone_increasing (b : ℝ) :
  ∀ x₁ x₂, x₁ < x₂ → f b x₁ < f b x₂ := by sorry

end f_odd_implies_b_one_f_monotone_increasing_l1403_140363


namespace square_area_with_rectangle_division_l1403_140317

theorem square_area_with_rectangle_division (x : ℝ) (h1 : x > 0) : 
  let rectangle_area := 14
  let square_side := 4 * x
  let rectangle_width := x
  let rectangle_length := 3 * x
  rectangle_area = rectangle_width * rectangle_length →
  (square_side)^2 = 224/3 := by
sorry

end square_area_with_rectangle_division_l1403_140317


namespace sum_of_squares_of_roots_l1403_140378

theorem sum_of_squares_of_roots (u v w : ℝ) : 
  (3 * u^3 - 7 * u^2 + 6 * u + 15 = 0) →
  (3 * v^3 - 7 * v^2 + 6 * v + 15 = 0) →
  (3 * w^3 - 7 * w^2 + 6 * w + 15 = 0) →
  u^2 + v^2 + w^2 = 13/9 := by
  sorry

end sum_of_squares_of_roots_l1403_140378


namespace stating_min_sides_for_rotation_l1403_140371

/-- The rotation angle in degrees -/
def rotation_angle : ℚ := 25 + 30 / 60

/-- The fraction of a full circle that the rotation represents -/
def rotation_fraction : ℚ := rotation_angle / 360

/-- The minimum number of sides for the polygons -/
def min_sides : ℕ := 240

/-- 
  Theorem stating that the minimum number of sides for two identical polygons
  that coincide when one is rotated by 25°30' is 240
-/
theorem min_sides_for_rotation :
  ∀ n : ℕ, 
    (n > 0 ∧ (rotation_fraction * n).den = 1) → 
    n ≥ min_sides :=
sorry

end stating_min_sides_for_rotation_l1403_140371


namespace sequence_count_mod_l1403_140314

def sequence_count (n : ℕ) (max : ℕ) : ℕ :=
  let m := Nat.choose (max - n + n) n
  m / 3

theorem sequence_count_mod (n : ℕ) (max : ℕ) : 
  sequence_count n max % 1000 = 662 :=
sorry

#check sequence_count_mod 10 2018

end sequence_count_mod_l1403_140314


namespace problem_solution_l1403_140318

theorem problem_solution : 
  (∀ x : ℝ, (Real.sqrt 24 - Real.sqrt 6) / Real.sqrt 3 - (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) = Real.sqrt 2 - 1) ∧ 
  (∀ x : ℝ, 2 * x^3 - 16 = 0 ↔ x = 2) := by
sorry

end problem_solution_l1403_140318


namespace unique_square_pattern_l1403_140330

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  n.hundreds * 100 + n.tens * 10 + n.ones

/-- Checks if a number satisfies the squaring pattern -/
def satisfiesSquarePattern (n : ThreeDigitNumber) : Prop :=
  let square := n.toNat * n.toNat
  -- Add conditions here that check if the square follows the pattern
  -- This is a placeholder and should be replaced with actual conditions
  true

/-- The main theorem stating that 748 is the only number satisfying the conditions -/
theorem unique_square_pattern : 
  ∃! n : ThreeDigitNumber, satisfiesSquarePattern n ∧ n.toNat = 748 := by
  sorry

#check unique_square_pattern

end unique_square_pattern_l1403_140330


namespace john_zoo_animals_l1403_140303

def zoo_animals (snakes : ℕ) : ℕ :=
  let monkeys := 2 * snakes
  let lions := monkeys - 5
  let pandas := lions + 8
  let dogs := pandas / 3
  snakes + monkeys + lions + pandas + dogs

theorem john_zoo_animals :
  zoo_animals 15 = 114 := by sorry

end john_zoo_animals_l1403_140303


namespace kelly_carrot_harvest_l1403_140343

/-- Represents the number of carrots harvested from each bed -/
structure CarrotHarvest where
  bed1 : ℕ
  bed2 : ℕ
  bed3 : ℕ

/-- Calculates the total weight of carrots in pounds -/
def totalWeight (harvest : CarrotHarvest) (carrotsPerPound : ℕ) : ℕ :=
  (harvest.bed1 + harvest.bed2 + harvest.bed3) / carrotsPerPound

/-- Theorem stating that Kelly's carrot harvest weighs 39 pounds -/
theorem kelly_carrot_harvest :
  let harvest := CarrotHarvest.mk 55 101 78
  let carrotsPerPound := 6
  totalWeight harvest carrotsPerPound = 39 := by
  sorry


end kelly_carrot_harvest_l1403_140343


namespace inverse_of_proposition_l1403_140337

theorem inverse_of_proposition :
  (∀ a b : ℝ, a = -2*b → a^2 = 4*b^2) →
  (∀ a b : ℝ, a^2 = 4*b^2 → a = -2*b) :=
by sorry

end inverse_of_proposition_l1403_140337


namespace shortest_time_to_camp_l1403_140399

/-- The shortest time to reach the camp across a river -/
theorem shortest_time_to_camp (river_width : ℝ) (camp_distance : ℝ) 
  (swim_speed : ℝ) (walk_speed : ℝ) (h1 : river_width = 1) 
  (h2 : camp_distance = 1) (h3 : swim_speed = 2) (h4 : walk_speed = 3) :
  ∃ (t : ℝ), t = (1 + Real.sqrt 13) / (3 * Real.sqrt 13) ∧ 
  (∀ (x : ℝ), x ≥ 0 ∧ x ≤ 1 → 
    t ≤ x / swim_speed + (camp_distance - Real.sqrt (river_width^2 - x^2)) / walk_speed) :=
by sorry

end shortest_time_to_camp_l1403_140399


namespace probability_two_white_balls_correct_l1403_140342

/-- The probability of having two white balls in an urn, given the conditions of the problem -/
def probability_two_white_balls (n : ℕ) : ℚ :=
  (4:ℚ)^n / ((2:ℚ) * (3:ℚ)^n + (4:ℚ)^n)

/-- The theorem stating the probability of having two white balls in the urn -/
theorem probability_two_white_balls_correct (n : ℕ) :
  let total_balls : ℕ := 4
  let draws : ℕ := 2 * n
  let white_draws : ℕ := n
  probability_two_white_balls n = (4:ℚ)^n / ((2:ℚ) * (3:ℚ)^n + (4:ℚ)^n) :=
by sorry

end probability_two_white_balls_correct_l1403_140342


namespace distinct_prime_factors_of_product_l1403_140338

theorem distinct_prime_factors_of_product : ∃ (s : Finset Nat), 
  (∀ p ∈ s, Nat.Prime p) ∧ 
  (∀ p : Nat, Nat.Prime p → (p ∣ (86 * 88 * 90 * 92) ↔ p ∈ s)) ∧ 
  Finset.card s = 6 := by
sorry

end distinct_prime_factors_of_product_l1403_140338


namespace congruence_problem_l1403_140385

theorem congruence_problem (x : ℤ) 
  (h1 : (4 + x) % (2^3) = 3^2 % (2^3))
  (h2 : (6 + x) % (3^3) = 2^2 % (3^3))
  (h3 : (8 + x) % (5^3) = 7^2 % (5^3)) :
  x % 30 = 1 := by sorry

end congruence_problem_l1403_140385


namespace isosceles_triangle_leg_length_l1403_140312

theorem isosceles_triangle_leg_length 
  (base : ℝ) 
  (leg : ℝ) 
  (h1 : base = 8) 
  (h2 : leg^2 - 9*leg + 20 = 0) 
  (h3 : leg > base/2) : 
  leg = 5 := by
sorry

end isosceles_triangle_leg_length_l1403_140312


namespace min_width_proof_l1403_140310

/-- The minimum width of a rectangular area with given constraints -/
def min_width : ℝ := 10

/-- The length of the rectangular area -/
def length (w : ℝ) : ℝ := w + 20

/-- The area of the rectangular area -/
def area (w : ℝ) : ℝ := w * length w

theorem min_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 150 → w ≥ min_width) ∧
  (area min_width ≥ 150) ∧
  (min_width > 0) :=
sorry

end min_width_proof_l1403_140310


namespace machine_N_output_fraction_l1403_140306

/-- Represents the production time of a machine relative to machine N -/
structure MachineTime where
  relative_to_N : ℚ

/-- Represents the production rate of a machine -/
def production_rate (m : MachineTime) : ℚ := 1 / m.relative_to_N

/-- The production time of machine T -/
def machine_T : MachineTime := ⟨3/4⟩

/-- The production time of machine N -/
def machine_N : MachineTime := ⟨1⟩

/-- The production time of machine O -/
def machine_O : MachineTime := ⟨3/2⟩

/-- The total production rate of all machines -/
def total_rate : ℚ :=
  production_rate machine_T + production_rate machine_N + production_rate machine_O

/-- The fraction of total output produced by machine N -/
def fraction_by_N : ℚ := production_rate machine_N / total_rate

theorem machine_N_output_fraction :
  fraction_by_N = 1/3 := by sorry

end machine_N_output_fraction_l1403_140306


namespace partial_fraction_decomposition_l1403_140313

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 →
    (6 * x - 3) / (x^2 - 8*x + 15) = C / (x - 3) + D / (x - 5)) →
  C = -15/2 ∧ D = 27/2 := by
sorry

end partial_fraction_decomposition_l1403_140313


namespace absolute_value_inequality_solution_set_l1403_140355

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x^2 - 2| < 2} = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end absolute_value_inequality_solution_set_l1403_140355


namespace dartboard_double_score_angle_l1403_140307

theorem dartboard_double_score_angle (num_regions : ℕ) (probability : ℚ) :
  num_regions = 6 →
  probability = 1 / 8 →
  (360 : ℚ) * probability = 45 :=
by
  sorry

end dartboard_double_score_angle_l1403_140307


namespace complex_equation_solution_l1403_140321

theorem complex_equation_solution (a : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a * i) / (2 - i) = (1 - 2*i) / 5) : a = -1 := by sorry

end complex_equation_solution_l1403_140321


namespace Q_has_negative_root_l1403_140387

/-- The polynomial Q(x) = x^7 - 4x^6 + 2x^5 - 9x^3 + 2x + 16 -/
def Q (x : ℝ) : ℝ := x^7 - 4*x^6 + 2*x^5 - 9*x^3 + 2*x + 16

/-- The polynomial Q(x) has at least one negative root -/
theorem Q_has_negative_root : ∃ x : ℝ, x < 0 ∧ Q x = 0 := by sorry

end Q_has_negative_root_l1403_140387


namespace evaluate_P_l1403_140397

/-- The polynomial P(x) = x^3 - 6x^2 - 5x + 4 -/
def P (x : ℝ) : ℝ := x^3 - 6*x^2 - 5*x + 4

/-- Theorem stating that under given conditions, P(y) = -22 -/
theorem evaluate_P (y z : ℝ) (h : ∀ n : ℝ, z * P y = P (y - n) + P (y + n)) : P y = -22 := by
  sorry

end evaluate_P_l1403_140397


namespace mikes_cards_l1403_140354

theorem mikes_cards (x : ℕ) : x + 18 = 82 → x = 64 := by
  sorry

end mikes_cards_l1403_140354


namespace min_value_reciprocal_sum_l1403_140308

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 ∧ (1/a + 1/b = 2 ↔ a = 1 ∧ b = 1) := by
  sorry

end min_value_reciprocal_sum_l1403_140308


namespace max_regions_theorem_l1403_140345

/-- Represents a convex polygon with a given number of sides -/
structure ConvexPolygon where
  sides : ℕ

/-- Represents two convex polygons on a plane -/
structure TwoPolygonsOnPlane where
  polygon1 : ConvexPolygon
  polygon2 : ConvexPolygon
  sides_condition : polygon1.sides > polygon2.sides

/-- The maximum number of regions into which two convex polygons can divide a plane -/
def max_regions (polygons : TwoPolygonsOnPlane) : ℕ :=
  2 * polygons.polygon2.sides + 2

/-- Theorem stating the maximum number of regions formed by two convex polygons on a plane -/
theorem max_regions_theorem (polygons : TwoPolygonsOnPlane) :
  max_regions polygons = 2 * polygons.polygon2.sides + 2 := by
  sorry

end max_regions_theorem_l1403_140345


namespace work_completion_time_l1403_140361

/-- Represents the number of days it takes for B to complete the entire work -/
def days_for_B (days_for_A days_A_worked days_B_remaining : ℕ) : ℚ :=
  (4 * days_for_A * days_B_remaining) / (3 * days_for_A - 3 * days_A_worked)

theorem work_completion_time :
  days_for_B 40 10 45 = 60 := by sorry

end work_completion_time_l1403_140361


namespace stationery_box_sheets_l1403_140393

/-- Represents a box of stationery -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- The scenario described in the problem -/
def stationery_scenario (box : StationeryBox) : Prop :=
  (box.sheets - box.envelopes = 30) ∧ 
  (2 * box.envelopes = box.sheets)

/-- The theorem to prove -/
theorem stationery_box_sheets : 
  ∀ (box : StationeryBox), stationery_scenario box → box.sheets = 60 := by
  sorry

end stationery_box_sheets_l1403_140393


namespace function_properties_and_triangle_l1403_140360

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2 + 3

theorem function_properties_and_triangle (A B C : ℝ) (a b c : ℝ) :
  (∀ x, f x ≤ 4) ∧ 
  (∀ ε > 0, ∃ T > 0, T ≤ π ∧ ∀ x, f (x + T) = f x) ∧
  c = Real.sqrt 3 →
  f C = 4 →
  Real.sin A = 2 * Real.sin B →
  a = 2 ∧ b = 1 := by
  sorry

end function_properties_and_triangle_l1403_140360


namespace negation_of_universal_quantification_l1403_140368

theorem negation_of_universal_quantification :
  (¬ ∀ x : ℝ, x + Real.log x > 0) ↔ (∃ x : ℝ, x + Real.log x ≤ 0) := by
  sorry

end negation_of_universal_quantification_l1403_140368


namespace two_roots_theorem_l1403_140315

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y → f x < f y

def has_exactly_two_roots (f : ℝ → ℝ) : Prop :=
  ∃ a b, a < b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b

theorem two_roots_theorem (f : ℝ → ℝ) 
  (h1 : even_function f)
  (h2 : monotone_increasing_nonneg f)
  (h3 : f 1 * f 2 < 0) :
  has_exactly_two_roots f :=
sorry

end two_roots_theorem_l1403_140315


namespace star_placement_impossible_l1403_140381

/-- Represents a grid of cells that may contain stars. -/
def Grid := Fin 10 → Fin 10 → Bool

/-- Checks if a 2x2 square starting at (i, j) contains exactly two stars. -/
def has_two_stars_2x2 (grid : Grid) (i j : Fin 10) : Prop :=
  (grid i j).toNat + (grid i (j+1)).toNat + (grid (i+1) j).toNat + (grid (i+1) (j+1)).toNat = 2

/-- Checks if a 3x1 rectangle starting at (i, j) contains exactly one star. -/
def has_one_star_3x1 (grid : Grid) (i j : Fin 10) : Prop :=
  (grid i j).toNat + (grid (i+1) j).toNat + (grid (i+2) j).toNat = 1

/-- The main theorem stating the impossibility of the star placement. -/
theorem star_placement_impossible : 
  ¬∃ (grid : Grid), 
    (∀ i j : Fin 9, has_two_stars_2x2 grid i j) ∧ 
    (∀ i : Fin 8, ∀ j : Fin 10, has_one_star_3x1 grid i j) :=
sorry

end star_placement_impossible_l1403_140381


namespace quadratic_inequality_solution_set_l1403_140320

theorem quadratic_inequality_solution_set (x : ℝ) :
  (∃ y ∈ Set.Icc (24 - 2 * Real.sqrt 19) (24 + 2 * Real.sqrt 19), x = y) ↔ 
  x^2 - 48*x + 500 ≤ 0 := by
  sorry

end quadratic_inequality_solution_set_l1403_140320


namespace divisibility_of_fifth_power_differences_l1403_140324

theorem divisibility_of_fifth_power_differences (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (x - y) * (y - z) * (z - x)) := by
  sorry

end divisibility_of_fifth_power_differences_l1403_140324


namespace parabola_properties_l1403_140322

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem parabola_properties (a b c : ℝ) 
  (h_a_nonzero : a ≠ 0)
  (h_c_gt_3 : c > 3)
  (h_passes_through : parabola a b c 5 = 0)
  (h_symmetry_axis : -b / (2 * a) = 2) :
  (a * b * c < 0) ∧ 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ parabola a b c x₁ = 2 ∧ parabola a b c x₂ = 2) ∧
  (a < -3/5) := by
  sorry

end parabola_properties_l1403_140322


namespace rest_worker_salary_l1403_140382

def workshop (total_workers : ℕ) (avg_salary : ℚ) (technicians : ℕ) (avg_technician_salary : ℚ) : Prop :=
  total_workers = 12 ∧
  avg_salary = 9000 ∧
  technicians = 6 ∧
  avg_technician_salary = 12000

theorem rest_worker_salary (total_workers : ℕ) (avg_salary : ℚ) (technicians : ℕ) (avg_technician_salary : ℚ) :
  workshop total_workers avg_salary technicians avg_technician_salary →
  (total_workers * avg_salary - technicians * avg_technician_salary) / (total_workers - technicians) = 6000 :=
by
  sorry

#check rest_worker_salary

end rest_worker_salary_l1403_140382


namespace largest_number_with_given_hcf_lcm_factors_l1403_140325

theorem largest_number_with_given_hcf_lcm_factors :
  ∀ a b c : ℕ+,
  (∃ (hcf : ℕ+) (lcm : ℕ+), 
    (Nat.gcd a b = hcf) ∧ 
    (Nat.gcd (Nat.gcd a b) c = hcf) ∧
    (Nat.lcm (Nat.lcm a b) c = lcm) ∧
    (hcf = 59) ∧
    (∃ (k : ℕ+), lcm = hcf * 13 * (2^4) * 23 * k)) →
  max a (max b c) = 282256 :=
by sorry

end largest_number_with_given_hcf_lcm_factors_l1403_140325


namespace ratio_equality_l1403_140366

theorem ratio_equality : (240 : ℚ) / 1547 / (2 / 13) = (5 : ℚ) / 34 / (7 / 48) := by
  sorry

end ratio_equality_l1403_140366
