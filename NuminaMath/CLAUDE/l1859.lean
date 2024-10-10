import Mathlib

namespace kishore_savings_l1859_185957

def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 3940
def savings_percentage : ℚ := 1 / 10

def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous

theorem kishore_savings :
  let monthly_salary := total_expenses / (1 - savings_percentage)
  (monthly_salary * savings_percentage).floor = 2160 := by
  sorry

end kishore_savings_l1859_185957


namespace seonhos_wallet_problem_l1859_185909

theorem seonhos_wallet_problem (initial_money : ℚ) : 
  (initial_money / 4) * (1 / 3) = 2500 → initial_money = 10000 := by sorry

end seonhos_wallet_problem_l1859_185909


namespace min_cuts_for_4x4x4_cube_l1859_185955

/-- Represents a cube with given dimensions -/
structure Cube where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cut operation on a cube -/
inductive Cut
  | X : Cut  -- Cut parallel to YZ plane
  | Y : Cut  -- Cut parallel to XZ plane
  | Z : Cut  -- Cut parallel to XY plane

/-- Function to calculate the minimum number of cuts required -/
def min_cuts_to_unit_cubes (c : Cube) : ℕ :=
  sorry

/-- Theorem stating the minimum number of cuts required for a 4x4x4 cube -/
theorem min_cuts_for_4x4x4_cube :
  let initial_cube : Cube := { length := 4, width := 4, height := 4 }
  min_cuts_to_unit_cubes initial_cube = 9 := by
  sorry

end min_cuts_for_4x4x4_cube_l1859_185955


namespace store_profit_theorem_l1859_185965

/-- Represents the store's inventory and pricing information -/
structure Store :=
  (total_items : ℕ)
  (purchase_price_A : ℝ)
  (selling_price_A : ℝ)
  (purchase_price_B : ℝ)
  (original_selling_price_B : ℝ)
  (original_daily_sales_B : ℝ)
  (sales_increase_per_yuan : ℝ)
  (target_daily_profit_B : ℝ)

/-- Calculates the total profit based on the number of type A items -/
def total_profit (s : Store) (x : ℝ) : ℝ :=
  (s.selling_price_A - s.purchase_price_A) * x +
  (s.original_selling_price_B - s.purchase_price_B) * (s.total_items - x)

/-- Calculates the daily profit for type B items based on the new selling price -/
def daily_profit_B (s : Store) (new_price : ℝ) : ℝ :=
  (new_price - s.purchase_price_B) *
  (s.original_daily_sales_B + s.sales_increase_per_yuan * (s.original_selling_price_B - new_price))

/-- The main theorem stating the properties of the store's profit calculations -/
theorem store_profit_theorem (s : Store) (x : ℝ) :
  (s.total_items = 80 ∧
   s.purchase_price_A = 40 ∧
   s.selling_price_A = 55 ∧
   s.purchase_price_B = 28 ∧
   s.original_selling_price_B = 40 ∧
   s.original_daily_sales_B = 4 ∧
   s.sales_increase_per_yuan = 2 ∧
   s.target_daily_profit_B = 96) →
  (total_profit s x = 3 * x + 960 ∧
   (daily_profit_B s 34 = 96 ∨ daily_profit_B s 36 = 96)) :=
by sorry


end store_profit_theorem_l1859_185965


namespace digit_sum_problem_l1859_185992

theorem digit_sum_problem :
  ∀ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
    a + c = 10 →
    b + c + 1 = 10 →
    a + d + 1 = 11 →
    1000 * a + 100 * b + 10 * c + d + 100 * c + 10 * a = 1100 →
    a + b + c + d = 18 := by
  sorry

end digit_sum_problem_l1859_185992


namespace division_theorem_l1859_185900

theorem division_theorem (dividend divisor remainder quotient : ℕ) :
  dividend = 176 →
  divisor = 14 →
  remainder = 8 →
  quotient = 12 →
  dividend = divisor * quotient + remainder :=
by sorry

end division_theorem_l1859_185900


namespace dice_sum_repetition_l1859_185977

theorem dice_sum_repetition (n : ℕ) (m : ℕ) (h1 : n = 21) (h2 : m = 22) :
  m > n → ∀ f : ℕ → ℕ, ∃ i j, i < j ∧ j < m ∧ f i = f j :=
by sorry

end dice_sum_repetition_l1859_185977


namespace spacing_change_at_20th_post_l1859_185969

/-- Represents the fence with its posts and spacings -/
structure Fence where
  initialSpacing : ℝ
  changedSpacing : ℝ
  changePost : ℕ

/-- The fence satisfies the given conditions -/
def satisfiesConditions (f : Fence) : Prop :=
  f.initialSpacing > f.changedSpacing ∧
  f.initialSpacing * 15 = 48 ∧
  f.changedSpacing * (28 - f.changePost) + f.initialSpacing * (f.changePost - 16) = 36 ∧
  f.changePost > 16 ∧ f.changePost ≤ 28

/-- The theorem stating that the 20th post is where the spacing changes -/
theorem spacing_change_at_20th_post (f : Fence) (h : satisfiesConditions f) : f.changePost = 20 := by
  sorry

end spacing_change_at_20th_post_l1859_185969


namespace mean_temperature_l1859_185990

def temperatures : List ℝ := [-8, -5, -3, -5, 2, 4, 3, -1]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : ℝ) = -1.5 := by
  sorry

end mean_temperature_l1859_185990


namespace game_lives_proof_l1859_185984

/-- Calculates the total number of lives for all players in a game --/
def totalLives (initialPlayers newPlayers livesPerPlayer : ℕ) : ℕ :=
  (initialPlayers + newPlayers) * livesPerPlayer

/-- Proves that the total number of lives is 24 given the specified conditions --/
theorem game_lives_proof :
  let initialPlayers : ℕ := 2
  let newPlayers : ℕ := 2
  let livesPerPlayer : ℕ := 6
  totalLives initialPlayers newPlayers livesPerPlayer = 24 := by
  sorry


end game_lives_proof_l1859_185984


namespace reciprocal_multiplier_l1859_185913

theorem reciprocal_multiplier (x m : ℝ) : 
  x > 0 → x = 7 → x - 4 = m * (1/x) → m = 21 := by
sorry

end reciprocal_multiplier_l1859_185913


namespace expansion_coefficient_l1859_185914

/-- The coefficient of x^n in the expansion of (x-1/x)^m -/
def coeff (m n : ℕ) : ℤ :=
  if (m - n) % 2 = 0 
  then (-1)^((m - n) / 2) * (m.choose ((m - n) / 2))
  else 0

/-- The coefficient of x^6 in the expansion of (x^2+a)(x-1/x)^10 -/
def coeff_x6 (a : ℤ) : ℤ := coeff 10 6 + a * coeff 10 4

theorem expansion_coefficient (a : ℤ) : 
  coeff_x6 a = -30 → a = 2 := by sorry

end expansion_coefficient_l1859_185914


namespace largest_multiple_of_8_under_100_l1859_185936

theorem largest_multiple_of_8_under_100 : 
  ∀ n : ℕ, n % 8 = 0 ∧ n < 100 → n ≤ 96 :=
by
  sorry

end largest_multiple_of_8_under_100_l1859_185936


namespace difference_between_squares_l1859_185953

theorem difference_between_squares : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by
  sorry

end difference_between_squares_l1859_185953


namespace simplify_expression_l1859_185987

theorem simplify_expression (x : ℝ) : 8*x + 15 - 3*x + 27 = 5*x + 42 := by
  sorry

end simplify_expression_l1859_185987


namespace range_of_t_l1859_185916

def M : Set ℝ := {x | -2 < x ∧ x < 5}

def N (t : ℝ) : Set ℝ := {x | 2 - t < x ∧ x < 2*t + 1}

theorem range_of_t : 
  (∀ t : ℝ, M ∩ N t = N t) ↔ (∀ t : ℝ, t ≤ 2) :=
sorry

end range_of_t_l1859_185916


namespace unique_integer_square_less_than_triple_l1859_185996

theorem unique_integer_square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 := by
  sorry

end unique_integer_square_less_than_triple_l1859_185996


namespace sarahs_age_l1859_185983

theorem sarahs_age (ana mark billy sarah : ℝ) 
  (h1 : sarah = 3 * mark - 4)
  (h2 : mark = billy + 4)
  (h3 : billy = ana / 2)
  (h4 : ∃ (years : ℝ), ana + years = 15) :
  sarah = 30.5 := by
sorry

end sarahs_age_l1859_185983


namespace percentage_problem_l1859_185980

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 100) : 1.2 * x = 600 := by
  sorry

end percentage_problem_l1859_185980


namespace no_valid_tiling_exists_l1859_185951

/-- Represents a chessboard square --/
inductive Square
| Black
| White

/-- Represents a 2x1 domino --/
structure Domino :=
(first : Square)
(second : Square)

/-- Represents the modified 8x8 chessboard with corners removed --/
def ModifiedChessboard := Fin 62 → Square

/-- A tiling of the modified chessboard using dominos --/
def Tiling := Fin 31 → Domino

/-- Checks if a tiling is valid for the modified chessboard --/
def is_valid_tiling (board : ModifiedChessboard) (tiling : Tiling) : Prop :=
  ∀ i j : Fin 62, i ≠ j → 
    ∃ k : Fin 31, (tiling k).first = board i ∧ (tiling k).second = board j

/-- The main theorem stating that no valid tiling exists --/
theorem no_valid_tiling_exists :
  ¬∃ (board : ModifiedChessboard) (tiling : Tiling), is_valid_tiling board tiling :=
sorry

end no_valid_tiling_exists_l1859_185951


namespace parallel_vectors_x_equals_one_l1859_185910

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_equals_one :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (2*x + 1, 3)
  let b : ℝ × ℝ := (2 - x, 1)
  parallel a b → x = 1 := by
sorry

end parallel_vectors_x_equals_one_l1859_185910


namespace solution_set_all_reals_solution_set_interval_l1859_185960

-- Part 1
theorem solution_set_all_reals (x : ℝ) : 8 * x - 1 ≤ 16 * x^2 := by sorry

-- Part 2
theorem solution_set_interval (a x : ℝ) (h : a < 0) :
  x^2 - 2*a*x - 3*a^2 < 0 ↔ 3*a < x ∧ x < -a := by sorry

end solution_set_all_reals_solution_set_interval_l1859_185960


namespace log_inequality_range_l1859_185981

theorem log_inequality_range (a : ℝ) : 
  (a > 0 ∧ ∀ x : ℝ, 0 < x ∧ x ≤ 1 → 4 * x < Real.log x / Real.log a) ↔ 
  (0 < a ∧ a < 1) := by
sorry

end log_inequality_range_l1859_185981


namespace typing_speed_equation_l1859_185904

theorem typing_speed_equation (x : ℝ) : x > 0 → x + 6 > 0 →
  (Xiao_Ming_speed : ℝ) →
  (Xiao_Zhang_speed : ℝ) →
  (Xiao_Ming_speed = x) →
  (Xiao_Zhang_speed = x + 6) →
  (120 / Xiao_Ming_speed = 180 / Xiao_Zhang_speed) →
  120 / x = 180 / (x + 6) := by
sorry

end typing_speed_equation_l1859_185904


namespace sum_divisibility_odd_sum_divisibility_l1859_185970

theorem sum_divisibility (n : ℕ) :
  (∃ k : ℕ, 2 * n ∣ (n * (n + 1) / 2)) ↔ (∃ k : ℕ, n = 4 * k - 1) :=
sorry

theorem odd_sum_divisibility (n : ℕ) :
  (∃ k : ℕ, (2 * n + 1) ∣ (n * (n + 1) / 2)) ↔
  ((2 * n + 1) % 4 = 1 ∨ (2 * n + 1) % 4 = 3) :=
sorry

end sum_divisibility_odd_sum_divisibility_l1859_185970


namespace unique_D_value_l1859_185932

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Definition of our addition problem -/
def AdditionProblem (A B C D : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  1000 * A.val + 100 * A.val + 10 * C.val + B.val +
  1000 * B.val + 100 * C.val + 10 * B.val + D.val =
  1000 * B.val + 100 * D.val + 10 * A.val + B.val

theorem unique_D_value (A B C D : Digit) :
  AdditionProblem A B C D → D.val = 0 ∧ ∀ E : Digit, AdditionProblem A B C E → E = D :=
by sorry

end unique_D_value_l1859_185932


namespace mulch_price_per_pound_l1859_185991

/-- Given the cost of mulch in tons, calculate the price per pound -/
theorem mulch_price_per_pound (cost : ℝ) (tons : ℝ) (pounds_per_ton : ℝ) : 
  cost = 15000 → tons = 3 → pounds_per_ton = 2000 →
  cost / (tons * pounds_per_ton) = 2.5 := by sorry

end mulch_price_per_pound_l1859_185991


namespace water_level_rise_rate_l1859_185945

/-- The water level function with respect to time -/
def water_level (t : ℝ) : ℝ := 0.3 * t + 3

/-- The time domain -/
def time_domain : Set ℝ := { t | 0 ≤ t ∧ t ≤ 5 }

/-- The rate of change of the water level -/
def water_level_rate : ℝ := 0.3

theorem water_level_rise_rate :
  ∀ t ∈ time_domain, 
    (water_level (t + 1) - water_level t) = water_level_rate := by
  sorry

end water_level_rise_rate_l1859_185945


namespace inequality_equivalence_l1859_185978

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / (2 - x) ≥ 0 ↔ Real.log (x - 2) ≤ 0 :=
by sorry

end inequality_equivalence_l1859_185978


namespace function_equation_solver_l1859_185911

theorem function_equation_solver (f : ℝ → ℝ) :
  (∀ x, f (x + 1) = x^2 + 4*x + 1) →
  (∀ x, f x = x^2 + 2*x - 2) :=
by sorry

end function_equation_solver_l1859_185911


namespace probability_not_greater_than_two_l1859_185937

def card_set : Finset ℕ := {1, 2, 3, 4}

theorem probability_not_greater_than_two :
  (card_set.filter (λ x => x ≤ 2)).card / card_set.card = (1 : ℚ) / 2 := by
  sorry

end probability_not_greater_than_two_l1859_185937


namespace smallest_multiple_in_sequence_l1859_185982

theorem smallest_multiple_in_sequence (a : ℕ) : 
  (∀ i ∈ Finset.range 16, ∃ k : ℕ, a + 3 * i = 3 * k) →
  (6 * a + 3 * (0 + 1 + 2 + 3 + 4 + 5) = 5 * a + 3 * (11 + 12 + 13 + 14 + 15)) →
  a = 150 := by
sorry

end smallest_multiple_in_sequence_l1859_185982


namespace increasing_function_implies_a_bound_l1859_185922

/-- A function f is increasing on an interval [a, +∞) if for any x₁, x₂ in the interval with x₁ < x₂, we have f(x₁) < f(x₂) -/
def IncreasingOnInterval (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂

/-- The main theorem stating that if f(x) = x^2 - 2ax + 2 is increasing on [3, +∞), then a ≤ 3 -/
theorem increasing_function_implies_a_bound (a : ℝ) :
  IncreasingOnInterval (fun x => x^2 - 2*a*x + 2) 3 → a ≤ 3 := by
  sorry

end increasing_function_implies_a_bound_l1859_185922


namespace point_A_coordinates_l1859_185946

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally and vertically -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem point_A_coordinates :
  ∀ (x y : ℝ),
  let A : Point := ⟨2*x + y, x - 2*y⟩
  let B : Point := translate A 1 (-4)
  B = ⟨x - y, y⟩ →
  A = ⟨1, 3⟩ :=
by
  sorry


end point_A_coordinates_l1859_185946


namespace polynomial_simplification_l1859_185934

theorem polynomial_simplification (q : ℝ) :
  (2 * q^3 - 7 * q^2 + 3 * q - 4) + (5 * q^2 - 4 * q + 8) = 2 * q^3 - 2 * q^2 - q + 4 := by
  sorry

end polynomial_simplification_l1859_185934


namespace intersection_A_B_l1859_185952

def A : Set ℝ := {x | x + 2 = 0}
def B : Set ℝ := {x | x^2 - 4 = 0}

theorem intersection_A_B : A ∩ B = {-2} := by
  sorry

end intersection_A_B_l1859_185952


namespace prob_red_or_black_is_three_fourths_prob_red_or_black_or_white_is_eleven_twelfths_l1859_185962

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | Black
  | White
  | Green

/-- Represents the box of balls -/
structure BallBox where
  total : ℕ
  red : ℕ
  black : ℕ
  white : ℕ
  green : ℕ
  sum_constraint : red + black + white + green = total

/-- Calculates the probability of drawing a ball of a specific color -/
def prob_color (box : BallBox) (color : BallColor) : ℚ :=
  match color with
  | BallColor.Red => box.red / box.total
  | BallColor.Black => box.black / box.total
  | BallColor.White => box.white / box.total
  | BallColor.Green => box.green / box.total

/-- The box described in the problem -/
def problem_box : BallBox :=
  { total := 12
    red := 5
    black := 4
    white := 2
    green := 1
    sum_constraint := by simp }

theorem prob_red_or_black_is_three_fourths :
    prob_color problem_box BallColor.Red + prob_color problem_box BallColor.Black = 3/4 := by
  sorry

theorem prob_red_or_black_or_white_is_eleven_twelfths :
    prob_color problem_box BallColor.Red + prob_color problem_box BallColor.Black +
    prob_color problem_box BallColor.White = 11/12 := by
  sorry

end prob_red_or_black_is_three_fourths_prob_red_or_black_or_white_is_eleven_twelfths_l1859_185962


namespace furniture_cost_price_l1859_185915

theorem furniture_cost_price (computer_table_price chair_price bookshelf_price : ℝ)
  (h1 : computer_table_price = 8091)
  (h2 : chair_price = 5346)
  (h3 : bookshelf_price = 11700)
  (computer_table_markup : ℝ)
  (h4 : computer_table_markup = 0.24)
  (chair_markup : ℝ)
  (h5 : chair_markup = 0.18)
  (chair_discount : ℝ)
  (h6 : chair_discount = 0.05)
  (bookshelf_markup : ℝ)
  (h7 : bookshelf_markup = 0.30)
  (sales_tax : ℝ)
  (h8 : sales_tax = 0.045) :
  ∃ (computer_table_cost chair_cost bookshelf_cost : ℝ),
    computer_table_cost = computer_table_price / (1 + computer_table_markup) ∧
    chair_cost = chair_price / ((1 + chair_markup) * (1 - chair_discount)) ∧
    bookshelf_cost = bookshelf_price / (1 + bookshelf_markup) ∧
    computer_table_cost + chair_cost + bookshelf_cost = 20295 :=
by sorry

end furniture_cost_price_l1859_185915


namespace log_25_between_consecutive_integers_l1859_185927

theorem log_25_between_consecutive_integers :
  ∃ c d : ℤ, c + 1 = d ∧ (c : ℝ) < Real.log 25 / Real.log 10 ∧ Real.log 25 / Real.log 10 < d ∧ c + d = 3 := by
  sorry

end log_25_between_consecutive_integers_l1859_185927


namespace parabola_line_intersection_property_l1859_185949

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (x y : ℝ) : Prop :=
  ∃ (k : ℝ), y = k * (x - 1) ∨ x = 1

-- Theorem statement
theorem parabola_line_intersection_property 
  (x₁ y₁ x₂ y₂ : ℝ) (h_distinct : (x₁, y₁) ≠ (x₂, y₂)) :
  (∀ x y, line_through_points x₁ y₁ x₂ y₂ x y → line_through_focus x y → 
    parabola x₁ y₁ → parabola x₂ y₂ → x₁ * x₂ = 1) ∧
  (∃ x₁' y₁' x₂' y₂', (x₁', y₁') ≠ (x₂', y₂') ∧
    parabola x₁' y₁' ∧ parabola x₂' y₂' ∧ x₁' * x₂' = 1 ∧
    ¬(∀ x y, line_through_points x₁' y₁' x₂' y₂' x y → line_through_focus x y)) :=
sorry

end parabola_line_intersection_property_l1859_185949


namespace three_digit_sum_divisibility_l1859_185929

theorem three_digit_sum_divisibility (a b : ℕ) : 
  (100 * 2 + 10 * a + 3) + 326 = (500 + 10 * b + 9) → 
  (500 + 10 * b + 9) % 9 = 0 →
  a + b = 6 := by sorry

end three_digit_sum_divisibility_l1859_185929


namespace seashells_count_l1859_185905

theorem seashells_count (mary_shells jessica_shells : ℕ) 
  (h1 : mary_shells = 18) 
  (h2 : jessica_shells = 41) : 
  mary_shells + jessica_shells = 59 := by
  sorry

end seashells_count_l1859_185905


namespace diagonals_of_25_sided_polygon_convex_polygon_25_sides_diagonals_l1859_185972

theorem diagonals_of_25_sided_polygon : ℕ → ℕ
  | n => (n * (n - 1)) / 2 - n

theorem convex_polygon_25_sides_diagonals :
  diagonals_of_25_sided_polygon 25 = 275 := by
  sorry

end diagonals_of_25_sided_polygon_convex_polygon_25_sides_diagonals_l1859_185972


namespace f_properties_l1859_185986

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 - 2

-- State the theorem
theorem f_properties :
  -- Function f is decreasing on (-∞, 0) and (2, +∞), and increasing on (0, 2)
  (∀ x y, x < y ∧ ((x < 0 ∧ y < 0) ∨ (x > 2 ∧ y > 2)) → f x > f y) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y) ∧
  -- Maximum value on [-2, 2] is 18
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≤ 18) ∧
  (∃ x, x ∈ Set.Icc (-2) 2 ∧ f x = 18) ∧
  -- Minimum value on [-2, 2] is -2
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≥ -2) ∧
  (∃ x, x ∈ Set.Icc (-2) 2 ∧ f x = -2) :=
by sorry


end f_properties_l1859_185986


namespace tulips_to_remaining_ratio_l1859_185940

def total_flowers : ℕ := 12
def daisies : ℕ := 2
def sunflowers : ℕ := 4

def tulips : ℕ := total_flowers - (daisies + sunflowers)
def remaining_flowers : ℕ := tulips + sunflowers

theorem tulips_to_remaining_ratio :
  (tulips : ℚ) / (remaining_flowers : ℚ) = 3 / 5 := by
  sorry

end tulips_to_remaining_ratio_l1859_185940


namespace largest_divisible_n_l1859_185947

theorem largest_divisible_n : 
  ∀ n : ℕ, n > 882 → ¬(n + 9 ∣ n^3 + 99) ∧ (882 + 9 ∣ 882^3 + 99) := by
  sorry

end largest_divisible_n_l1859_185947


namespace min_reciprocal_sum_l1859_185967

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 12) :
  (1 / a + 1 / b) ≥ 1 / 3 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 12 ∧ 1 / a₀ + 1 / b₀ = 1 / 3 :=
by sorry

end min_reciprocal_sum_l1859_185967


namespace line_intersects_circle_l1859_185948

theorem line_intersects_circle (a : ℝ) (h : a ≥ 0) :
  ∃ (x y : ℝ), (a * x - y + Real.sqrt 2 * a = 0) ∧ (x^2 + y^2 = 9) := by
  sorry

end line_intersects_circle_l1859_185948


namespace special_sequence_second_term_l1859_185964

/-- An arithmetic sequence with three terms -/
structure ArithmeticSequence3 where
  a : ℤ  -- First term
  b : ℤ  -- Second term
  c : ℤ  -- Third term
  is_arithmetic : b - a = c - b

/-- The second term of an arithmetic sequence with 3² as first term and 3⁴ as third term -/
def second_term_of_special_sequence : ℤ := 45

/-- Theorem stating that the second term of the special arithmetic sequence is 45 -/
theorem special_sequence_second_term :
  ∀ (seq : ArithmeticSequence3), 
  seq.a = 3^2 ∧ seq.c = 3^4 → seq.b = second_term_of_special_sequence :=
by sorry

end special_sequence_second_term_l1859_185964


namespace inequalities_proof_l1859_185974

theorem inequalities_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : Real.sqrt a ^ 3 + Real.sqrt b ^ 3 + Real.sqrt c ^ 3 = 1) :
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end inequalities_proof_l1859_185974


namespace monochromatic_triangle_exists_l1859_185908

/-- A complete graph with 6 vertices where each edge is colored either black or red -/
def ColoredGraph6 := Fin 6 → Fin 6 → Bool

/-- A triangle in the graph is represented by three distinct vertices -/
def Triangle (G : ColoredGraph6) (a b c : Fin 6) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- A monochromatic triangle has all edges of the same color -/
def MonochromaticTriangle (G : ColoredGraph6) (a b c : Fin 6) : Prop :=
  Triangle G a b c ∧
  ((G a b = G b c ∧ G b c = G a c) ∨
   (G a b ≠ G b c ∧ G b c ≠ G a c ∧ G a c ≠ G a b))

/-- The main theorem: every 2-coloring of K6 contains a monochromatic triangle -/
theorem monochromatic_triangle_exists (G : ColoredGraph6) :
  ∃ (a b c : Fin 6), MonochromaticTriangle G a b c := by
  sorry


end monochromatic_triangle_exists_l1859_185908


namespace money_distribution_l1859_185989

theorem money_distribution (a b c : ℤ) 
  (total : a + b + c = 500)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 360) :
  c = 60 := by
sorry

end money_distribution_l1859_185989


namespace total_cookies_baked_l1859_185901

/-- Calculates the total number of cookies baked by a baker -/
theorem total_cookies_baked 
  (chocolate_chip_batches : ℕ) 
  (cookies_per_batch : ℕ) 
  (oatmeal_cookies : ℕ) : 
  chocolate_chip_batches * cookies_per_batch + oatmeal_cookies = 10 :=
by
  sorry

#check total_cookies_baked 2 3 4

end total_cookies_baked_l1859_185901


namespace fair_attendance_l1859_185954

theorem fair_attendance (projected_increase : Real) (actual_decrease : Real) :
  projected_increase = 0.25 →
  actual_decrease = 0.20 →
  (1 - actual_decrease) / (1 + projected_increase) * 100 = 64 := by
sorry

end fair_attendance_l1859_185954


namespace freshmen_assignment_l1859_185976

/-- The number of ways to assign n freshmen to k classes with at least one freshman in each class -/
def assignFreshmen (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to arrange m groups into k classes -/
def arrangeGroups (m k : ℕ) : ℕ :=
  sorry

theorem freshmen_assignment :
  assignFreshmen 5 3 * arrangeGroups 3 3 = 150 :=
sorry

end freshmen_assignment_l1859_185976


namespace veronica_cherry_pie_l1859_185919

/-- Given that:
  - There are 80 cherries in one pound
  - It takes 10 minutes to pit 20 cherries
  - It takes Veronica 2 hours to pit all the cherries
  Prove that Veronica needs 3 pounds of cherries for her pie. -/
theorem veronica_cherry_pie (cherries_per_pound : ℕ) (pit_time : ℕ) (pit_amount : ℕ) (total_time : ℕ) :
  cherries_per_pound = 80 →
  pit_time = 10 →
  pit_amount = 20 →
  total_time = 120 →
  (total_time / pit_time) * pit_amount / cherries_per_pound = 3 :=
by sorry

end veronica_cherry_pie_l1859_185919


namespace probability_two_cards_sum_19_l1859_185917

/-- Represents a standard 52-card deck --/
def StandardDeck : ℕ := 52

/-- Number of cards that can be part of the pair (9 or 10) --/
def ValidFirstCards : ℕ := 8

/-- Number of complementary cards after drawing the first card --/
def ComplementaryCards : ℕ := 4

/-- Probability of drawing two number cards totaling 19 from a standard deck --/
theorem probability_two_cards_sum_19 :
  (ValidFirstCards : ℚ) / StandardDeck * ComplementaryCards / (StandardDeck - 1) = 8 / 663 := by
  sorry

end probability_two_cards_sum_19_l1859_185917


namespace solve_for_a_l1859_185903

theorem solve_for_a : ∃ a : ℝ, 
  (∃ x y : ℝ, x = 1 ∧ y = 2 ∧ a * x - y = 3) → a = 5 := by
  sorry

end solve_for_a_l1859_185903


namespace erased_line_length_l1859_185941

/-- Proves that erasing 33 cm from a 1 m line results in a 67 cm line -/
theorem erased_line_length : 
  let initial_length_m : ℝ := 1
  let initial_length_cm : ℝ := initial_length_m * 100
  let erased_length_cm : ℝ := 33
  let final_length_cm : ℝ := initial_length_cm - erased_length_cm
  final_length_cm = 67 := by sorry

end erased_line_length_l1859_185941


namespace water_fountain_problem_l1859_185921

/-- Represents the number of men needed to build a water fountain -/
def men_needed (length : ℕ) (days : ℕ) (men : ℕ) : Prop :=
  ∃ (k : ℚ), k * (men * days) = length

theorem water_fountain_problem :
  men_needed 56 42 60 ∧ men_needed 7 3 35 →
  (∀ l₁ d₁ m₁ l₂ d₂ m₂,
    men_needed l₁ d₁ m₁ → men_needed l₂ d₂ m₂ →
    (m₁ * d₁ : ℚ) / l₁ = (m₂ * d₂ : ℚ) / l₂) →
  60 = (35 * 3 * 56) / (7 * 42) :=
by sorry

end water_fountain_problem_l1859_185921


namespace janet_tickets_l1859_185902

/-- The number of tickets needed for Janet's amusement park rides -/
def total_tickets (roller_coaster_tickets_per_ride : ℕ) 
                  (giant_slide_tickets_per_ride : ℕ) 
                  (roller_coaster_rides : ℕ) 
                  (giant_slide_rides : ℕ) : ℕ :=
  roller_coaster_tickets_per_ride * roller_coaster_rides + 
  giant_slide_tickets_per_ride * giant_slide_rides

/-- Theorem: Janet needs 47 tickets for her amusement park rides -/
theorem janet_tickets : 
  total_tickets 5 3 7 4 = 47 := by
  sorry

end janet_tickets_l1859_185902


namespace one_fourth_of_6_8_l1859_185994

theorem one_fourth_of_6_8 : (6.8 : ℚ) / 4 = 17 / 10 := by
  sorry

end one_fourth_of_6_8_l1859_185994


namespace decimal_to_base_k_l1859_185939

/-- Given that the decimal number 26 is equal to the base-k number 32, prove that k = 8 -/
theorem decimal_to_base_k (k : ℕ) (h : 3 * k + 2 = 26) : k = 8 := by
  sorry

end decimal_to_base_k_l1859_185939


namespace equation_b_is_quadratic_l1859_185993

/-- A quadratic equation in one variable is an equation that can be written in the form ax² + bx + c = 0, where a ≠ 0 and x is a variable. --/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(y) = 5y² - 5y represents the equation 5y = 5y². --/
def f (y : ℝ) : ℝ := 5 * y^2 - 5 * y

/-- Theorem: The equation 5y = 5y² is a quadratic equation. --/
theorem equation_b_is_quadratic : is_quadratic_equation f := by
  sorry

end equation_b_is_quadratic_l1859_185993


namespace range_of_f_l1859_185942

-- Define the function f
def f (x : ℝ) : ℝ := (x^2 - 2)^2

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ 4} := by sorry

end range_of_f_l1859_185942


namespace f_six_equals_one_half_l1859_185926

-- Define the function f
noncomputable def f : ℝ → ℝ := λ u => (u^2 - 8*u + 20) / 16

-- State the theorem
theorem f_six_equals_one_half :
  (∀ x : ℝ, f (4*x + 2) = x^2 - x + 1) → f 6 = 1/2 := by
  sorry

end f_six_equals_one_half_l1859_185926


namespace chris_fishing_trips_l1859_185930

theorem chris_fishing_trips (brian_trips : ℕ) (chris_trips : ℕ) (brian_fish_per_trip : ℕ) (total_fish : ℕ) :
  brian_trips = 2 * chris_trips →
  brian_fish_per_trip = 400 →
  total_fish = 13600 →
  brian_fish_per_trip * brian_trips + (chris_trips * (brian_fish_per_trip * 7 / 5)) = total_fish →
  chris_trips = 10 := by
sorry

end chris_fishing_trips_l1859_185930


namespace multiply_and_add_l1859_185999

theorem multiply_and_add : 45 * 55 + 45 * 45 = 4500 := by
  sorry

end multiply_and_add_l1859_185999


namespace function_inequality_l1859_185928

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := (1/3)^x - x^2

-- State the theorem
theorem function_inequality (x₀ x₁ x₂ m : ℝ) 
  (h1 : f x₀ = m) 
  (h2 : x₁ ∈ Set.Ioo 0 x₀) 
  (h3 : x₂ ∈ Set.Ioi x₀) : 
  f x₁ > m ∧ f x₂ < m := by
  sorry

end

end function_inequality_l1859_185928


namespace triangle_angles_l1859_185943

theorem triangle_angles (x y z : ℝ) : 
  (y + 150 + 160 = 360) →
  (z + 150 + 160 = 360) →
  (x + y + z = 180) →
  (x = 80 ∧ y = 50 ∧ z = 50) :=
by sorry

end triangle_angles_l1859_185943


namespace complex_fraction_simplification_l1859_185931

theorem complex_fraction_simplification :
  let z : ℂ := (10 : ℂ) - 8 * Complex.I
  let w : ℂ := (3 : ℂ) + 4 * Complex.I
  z / w = -(2 : ℂ) / 25 - (64 : ℂ) / 25 * Complex.I := by
  sorry

end complex_fraction_simplification_l1859_185931


namespace abc_product_l1859_185924

theorem abc_product (a b c : ℕ) : 
  Nat.Prime a → 
  Nat.Prime b → 
  Nat.Prime c → 
  a * b * c < 10000 → 
  2 * a + 3 * b = c → 
  4 * a + c + 1 = 4 * b → 
  a * b * c = 1118 := by
  sorry

end abc_product_l1859_185924


namespace new_recipe_water_amount_l1859_185985

/-- Represents a recipe ratio --/
structure RecipeRatio :=
  (flour : ℕ)
  (water : ℕ)
  (sugar : ℕ)

/-- The original recipe ratio --/
def original_ratio : RecipeRatio :=
  ⟨8, 4, 3⟩

/-- The new recipe ratio --/
def new_ratio : RecipeRatio :=
  ⟨4, 1, 3⟩

/-- Amount of sugar in the new recipe (in cups) --/
def new_sugar_amount : ℕ := 6

/-- Calculates the amount of water in the new recipe --/
def calculate_water_amount (r : RecipeRatio) (sugar_amount : ℕ) : ℚ :=
  (r.water : ℚ) * sugar_amount / r.sugar

/-- Theorem stating that the new recipe calls for 2 cups of water --/
theorem new_recipe_water_amount :
  calculate_water_amount new_ratio new_sugar_amount = 2 := by
  sorry

end new_recipe_water_amount_l1859_185985


namespace sum_of_coefficients_l1859_185956

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + 
    a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + a₉*(x+2)^9 + 
    a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 :=
by
  sorry

end sum_of_coefficients_l1859_185956


namespace arithmetic_sequence_sum_ratio_l1859_185918

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  arithmetic : ∀ n, a (n + 1) = a n + d
  S : ℕ → ℝ  -- Sum function
  sum_def : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence, if S₁/S₄ = 1/10, then S₃/S₅ = 2/5 -/
theorem arithmetic_sequence_sum_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.S 1 / seq.S 4 = 1 / 10) : 
  seq.S 3 / seq.S 5 = 2 / 5 := by
  sorry

end arithmetic_sequence_sum_ratio_l1859_185918


namespace sum_of_composite_function_l1859_185959

def p (x : ℝ) : ℝ := x^2 - 3*x + 2

def q (x : ℝ) : ℝ := -x^2

def eval_points : List ℝ := [0, 1, 2, 3, 4]

theorem sum_of_composite_function :
  (eval_points.map (λ x => q (p x))).sum = -12 := by sorry

end sum_of_composite_function_l1859_185959


namespace catch_up_time_l1859_185995

-- Define the speeds of the girl, young man, and tram
def girl_speed : ℝ := 1
def young_man_speed : ℝ := 2 * girl_speed
def tram_speed : ℝ := 5 * young_man_speed

-- Define the time the young man waits before exiting the tram
def wait_time : ℝ := 8

-- Define the theorem
theorem catch_up_time : 
  ∀ (t : ℝ), 
  (girl_speed * wait_time + tram_speed * wait_time + girl_speed * t = young_man_speed * t) → 
  t = 88 := by
  sorry

end catch_up_time_l1859_185995


namespace compound_molecular_weight_l1859_185975

/-- Atomic weights of elements in g/mol -/
def atomic_weight : String → ℝ
  | "H" => 1
  | "N" => 14
  | "O" => 16
  | "S" => 32
  | "Fe" => 56
  | _ => 0

/-- Molecular weight of a compound given its chemical formula -/
def molecular_weight (formula : String) : ℝ := sorry

/-- The compound (NH4)2SO4·Fe2(SO4)3·6H2O -/
def compound : String := "(NH4)2SO4·Fe2(SO4)3·6H2O"

/-- Theorem stating that the molecular weight of (NH4)2SO4·Fe2(SO4)3·6H2O is 772 g/mol -/
theorem compound_molecular_weight :
  molecular_weight compound = 772 := by sorry

end compound_molecular_weight_l1859_185975


namespace rectangular_garden_area_l1859_185938

/-- A rectangular garden with length three times its width and width of 15 meters has an area of 675 square meters. -/
theorem rectangular_garden_area :
  ∀ (length width area : ℝ),
  length = 3 * width →
  width = 15 →
  area = length * width →
  area = 675 := by
  sorry

end rectangular_garden_area_l1859_185938


namespace negative_integer_sum_with_square_is_six_l1859_185998

theorem negative_integer_sum_with_square_is_six (N : ℤ) : 
  N < 0 → N^2 + N = 6 → N = -3 := by
  sorry

end negative_integer_sum_with_square_is_six_l1859_185998


namespace three_distinct_roots_l1859_185979

open Real

theorem three_distinct_roots (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    (abs (x₁^3 - a^3) = x₁ - a) ∧
    (abs (x₂^3 - a^3) = x₂ - a) ∧
    (abs (x₃^3 - a^3) = x₃ - a)) ↔ 
  (-2 / sqrt 3 < a ∧ a < -1 / sqrt 3) :=
sorry

end three_distinct_roots_l1859_185979


namespace cats_meowing_time_l1859_185925

/-- The number of minutes the cats were meowing -/
def minutes : ℚ := 5

/-- The number of meows per minute for the first cat -/
def first_cat_meows : ℚ := 3

/-- The number of meows per minute for the second cat -/
def second_cat_meows : ℚ := 2 * first_cat_meows

/-- The number of meows per minute for the third cat -/
def third_cat_meows : ℚ := (1/3) * second_cat_meows

/-- The total number of meows -/
def total_meows : ℚ := 55

theorem cats_meowing_time :
  minutes * (first_cat_meows + second_cat_meows + third_cat_meows) = total_meows :=
by sorry

end cats_meowing_time_l1859_185925


namespace max_n_inequality_l1859_185958

theorem max_n_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (∀ n : ℝ, 1 / (a - b) + 1 / (b - c) ≥ n / (a - c)) →
  (∃ n : ℝ, 1 / (a - b) + 1 / (b - c) = n / (a - c) ∧
            ∀ m : ℝ, 1 / (a - b) + 1 / (b - c) ≥ m / (a - c) → m ≤ n) →
  (∃ n : ℝ, n = 4 ∧
            1 / (a - b) + 1 / (b - c) = n / (a - c) ∧
            ∀ m : ℝ, 1 / (a - b) + 1 / (b - c) ≥ m / (a - c) → m ≤ n) :=
by sorry


end max_n_inequality_l1859_185958


namespace sum_of_roots_zero_l1859_185933

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_roots_zero
  (a b c : ℝ)
  (h : ∀ x : ℝ, QuadraticPolynomial a b c (x^3 - x) ≥ QuadraticPolynomial a b c (x^2 - 1)) :
  b / a = 0 :=
sorry

end sum_of_roots_zero_l1859_185933


namespace num_valid_configs_eq_30_l1859_185968

/-- A valid grid configuration -/
structure GridConfig where
  numbers : Fin 9 → Bool
  positions : Fin 6 → Fin 9
  left_greater_right : ∀ i : Fin 2, positions (2*i) > positions (2*i + 1)
  top_smaller_bottom : ∀ i : Fin 3, positions i < positions (i + 3)
  all_different : ∀ i j : Fin 6, i ≠ j → positions i ≠ positions j
  used_numbers : ∀ i : Fin 9, numbers i = (∃ j : Fin 6, positions j = i)

/-- The number of valid grid configurations -/
def num_valid_configs : ℕ := sorry

/-- The main theorem: there are exactly 30 valid grid configurations -/
theorem num_valid_configs_eq_30 : num_valid_configs = 30 := by sorry

end num_valid_configs_eq_30_l1859_185968


namespace men_science_majors_percentage_l1859_185944

/-- Represents the composition of a college class -/
structure ClassComposition where
  total_students : ℕ
  women_science_majors : ℕ
  non_science_majors : ℕ
  men : ℕ

/-- Calculates the percentage of men who are science majors -/
def percentage_men_science_majors (c : ClassComposition) : ℚ :=
  let total_science_majors := c.total_students - c.non_science_majors
  let men_science_majors := total_science_majors - c.women_science_majors
  (men_science_majors : ℚ) / (c.men : ℚ) * 100

/-- Theorem stating the percentage of men who are science majors -/
theorem men_science_majors_percentage (c : ClassComposition) 
  (h1 : c.women_science_majors = c.total_students * 30 / 100)
  (h2 : c.non_science_majors = c.total_students * 60 / 100)
  (h3 : c.men = c.total_students * 40 / 100) :
  percentage_men_science_majors c = 25 := by
  sorry

end men_science_majors_percentage_l1859_185944


namespace original_number_proof_l1859_185935

theorem original_number_proof (increased_number : ℝ) (increase_percentage : ℝ) :
  increased_number = 480 ∧ increase_percentage = 0.2 →
  (1 + increase_percentage) * (increased_number / (1 + increase_percentage)) = 400 :=
by
  sorry

end original_number_proof_l1859_185935


namespace square_construction_implies_parallel_l1859_185997

-- Define the triangle ABC
variable (A B C : Plane)

-- Define the squares constructed on the sides of triangle ABC
variable (A₂ B₁ B₂ C₁ : Plane)

-- Define the additional squares
variable (A₃ A₄ B₃ B₄ : Plane)

-- Define the property of being a square
def is_square (P Q R S : Plane) : Prop := sorry

-- Define the property of being external to a triangle
def is_external_to_triangle (S₁ S₂ S₃ S₄ P Q R : Plane) : Prop := sorry

-- Define the property of being parallel
def is_parallel (P₁ P₂ Q₁ Q₂ : Plane) : Prop := sorry

theorem square_construction_implies_parallel :
  is_square A B B₁ A₂ →
  is_square B C C₁ B₂ →
  is_square C A A₁ C₂ →
  is_external_to_triangle A B B₁ A₂ A B C →
  is_external_to_triangle B C C₁ B₂ B C A →
  is_external_to_triangle C A A₁ C₂ C A B →
  is_square A₁ A₂ A₃ A₄ →
  is_square B₁ B₂ B₃ B₄ →
  is_external_to_triangle A₁ A₂ A₃ A₄ A A₁ A₂ →
  is_external_to_triangle B₁ B₂ B₃ B₄ B B₁ B₂ →
  is_parallel A₃ B₄ A B := by sorry

end square_construction_implies_parallel_l1859_185997


namespace squares_sum_equality_l1859_185971

/-- Represents a 3-4-5 right triangle with squares on each side -/
structure Triangle345WithSquares where
  /-- Area of the square on the side of length 3 -/
  A : ℝ
  /-- Area of the square on the side of length 4 -/
  B : ℝ
  /-- Area of the square on the hypotenuse (side of length 5) -/
  C : ℝ
  /-- The area of the square on side 3 is 9 -/
  h_A : A = 9
  /-- The area of the square on side 4 is 16 -/
  h_B : B = 16
  /-- The area of the square on the hypotenuse is 25 -/
  h_C : C = 25

/-- 
For a 3-4-5 right triangle with squares constructed on each side, 
the sum of the areas of the squares on the two shorter sides 
equals the area of the square on the hypotenuse.
-/
theorem squares_sum_equality (t : Triangle345WithSquares) : t.A + t.B = t.C := by
  sorry

end squares_sum_equality_l1859_185971


namespace unique_positive_solution_l1859_185906

/-- The polynomial function we're analyzing -/
def g (x : ℝ) : ℝ := x^10 + 9*x^9 + 20*x^8 + 2000*x^7 - 1500*x^6

/-- Theorem stating that g(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! x : ℝ, x > 0 ∧ g x = 0 := by
  sorry

end unique_positive_solution_l1859_185906


namespace circle_symmetry_l1859_185963

-- Define the original circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y + 24 = 0

-- Define the line l
def l (x y : ℝ) : Prop := x - 3*y - 5 = 0

-- Define the symmetric circle S
def S (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), S x y ↔ ∃ (x' y' : ℝ), C x' y' ∧
  (∃ (m : ℝ), l m ((y + y')/2) ∧ m = (x + x')/2) ∧
  ((y - y')/(x - x') = -3 ∨ x = x') :=
sorry

end circle_symmetry_l1859_185963


namespace negation_existential_derivative_l1859_185950

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

theorem negation_existential_derivative :
  (¬ ∃ x : ℝ, f' x ≥ 0) ↔ (∀ x : ℝ, f' x < 0) :=
sorry

end negation_existential_derivative_l1859_185950


namespace quadratic_inequality_l1859_185966

/-- A quadratic function f(x) = 3x^2 + ax + b where f(x-1) is an even function -/
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ 3 * x^2 + a * x + b

/-- The property that f(x-1) is an even function -/
def f_even (a b : ℝ) : Prop := ∀ x, f a b (x - 1) = f a b (-x - 1)

theorem quadratic_inequality (a b : ℝ) (h : f_even a b) :
  f a b (-1) < f a b (-3/2) ∧ f a b (-3/2) < f a b (3/2) :=
sorry

end quadratic_inequality_l1859_185966


namespace perfect_square_trinomials_l1859_185907

-- Perfect square trinomial properties
theorem perfect_square_trinomials 
  (x a b : ℝ) : 
  (x^2 + 6*x + 9 = (x + 3)^2) ∧ 
  (x^2 + 8*x + 16 = (x + 4)^2) ∧ 
  (x^2 - 12*x + 36 = (x - 6)^2) ∧ 
  (a^2 + 2*a*b + b^2 = (a + b)^2) ∧ 
  (a^2 - 2*a*b + b^2 = (a - b)^2) := by
  sorry

end perfect_square_trinomials_l1859_185907


namespace jerry_first_table_trays_l1859_185961

/-- The number of trays Jerry can carry at a time -/
def trays_per_trip : ℕ := 8

/-- The number of trips Jerry made -/
def number_of_trips : ℕ := 2

/-- The number of trays Jerry picked up from the second table -/
def trays_from_second_table : ℕ := 7

/-- The number of trays Jerry picked up from the first table -/
def trays_from_first_table : ℕ := trays_per_trip * number_of_trips - trays_from_second_table

theorem jerry_first_table_trays :
  trays_from_first_table = 9 :=
by sorry

end jerry_first_table_trays_l1859_185961


namespace unpainted_cubes_4x4x4_l1859_185973

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_units : Nat
  painted_per_face : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  cube.total_units - (cube.painted_per_face * 6)

/-- Theorem: A 4x4x4 cube with 4 unit squares painted on each face has 40 unpainted unit cubes -/
theorem unpainted_cubes_4x4x4 :
  let cube : PaintedCube := {
    size := 4,
    total_units := 64,
    painted_per_face := 4
  }
  unpainted_cubes cube = 40 := by
  sorry


end unpainted_cubes_4x4x4_l1859_185973


namespace smaller_cuboid_length_l1859_185920

/-- Proves that the length of smaller cuboids is 5 meters, given the specified conditions --/
theorem smaller_cuboid_length : 
  ∀ (large_length large_width large_height : ℝ) 
    (small_width small_height : ℝ) 
    (num_small_cuboids : ℕ),
  large_length = 18 →
  large_width = 15 →
  large_height = 2 →
  small_width = 2 →
  small_height = 3 →
  num_small_cuboids = 18 →
  ∃ (small_length : ℝ),
    small_length = 5 ∧
    large_length * large_width * large_height = 
      num_small_cuboids * small_length * small_width * small_height :=
by sorry

end smaller_cuboid_length_l1859_185920


namespace age_ratio_problem_l1859_185912

theorem age_ratio_problem (cindy_age jan_age marcia_age greg_age : ℕ) : 
  cindy_age = 5 →
  jan_age = cindy_age + 2 →
  ∃ k : ℕ, marcia_age = k * jan_age →
  greg_age = marcia_age + 2 →
  greg_age = 16 →
  marcia_age / jan_age = 2 := by
sorry

end age_ratio_problem_l1859_185912


namespace shift_sin_left_specific_sin_shift_l1859_185988

/-- Shifting a sinusoidal function to the left --/
theorem shift_sin_left (A ω φ δ : ℝ) :
  let f (x : ℝ) := A * Real.sin (ω * x + φ)
  let g (x : ℝ) := A * Real.sin (ω * (x + δ) + φ)
  ∀ x, f (x - δ) = g x := by sorry

/-- The specific shift problem --/
theorem specific_sin_shift :
  let f (x : ℝ) := 3 * Real.sin (2 * x - π / 6)
  let g (x : ℝ) := 3 * Real.sin (2 * x + π / 3)
  ∀ x, f (x - π / 4) = g x := by sorry

end shift_sin_left_specific_sin_shift_l1859_185988


namespace jessica_birth_year_l1859_185923

theorem jessica_birth_year (first_amc8_year : ℕ) (jessica_age : ℕ) :
  first_amc8_year = 1985 →
  jessica_age = 15 →
  (first_amc8_year + 10 - 1) - jessica_age = 1979 :=
by
  sorry

end jessica_birth_year_l1859_185923
