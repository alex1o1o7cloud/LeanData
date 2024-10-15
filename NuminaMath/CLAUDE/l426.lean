import Mathlib

namespace NUMINAMATH_CALUDE_button_problem_l426_42678

/-- Proof of the button-making problem --/
theorem button_problem (mari_buttons sue_buttons : ℕ) 
  (h_mari : mari_buttons = 8)
  (h_sue : sue_buttons = 22)
  (h_sue_half_kendra : sue_buttons * 2 = mari_buttons * x + 4)
  : x = 5 := by
  sorry

#check button_problem

end NUMINAMATH_CALUDE_button_problem_l426_42678


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_150_choose_75_l426_42684

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_two_digit_prime_factor_of_150_choose_75 :
  ∃ (p : ℕ), p = 47 ∧ 
    Prime p ∧ 
    10 ≤ p ∧ p < 100 ∧
    p ∣ binomial 150 75 ∧
    ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ binomial 150 75 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_150_choose_75_l426_42684


namespace NUMINAMATH_CALUDE_candidate_votes_l426_42624

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) :
  total_votes = 560000 →
  invalid_percent = 15 / 100 →
  candidate_percent = 75 / 100 →
  ↑⌊(1 - invalid_percent) * candidate_percent * total_votes⌋ = 357000 := by
  sorry

end NUMINAMATH_CALUDE_candidate_votes_l426_42624


namespace NUMINAMATH_CALUDE_pizza_slice_cost_l426_42699

theorem pizza_slice_cost (num_pizzas : ℕ) (slices_per_pizza : ℕ) (total_cost : ℚ) :
  num_pizzas = 3 →
  slices_per_pizza = 12 →
  total_cost = 72 →
  (5 : ℚ) * (total_cost / (↑num_pizzas * ↑slices_per_pizza)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_cost_l426_42699


namespace NUMINAMATH_CALUDE_prime_quadruples_sum_882_l426_42681

theorem prime_quadruples_sum_882 :
  ∀ p₁ p₂ p₃ p₄ : ℕ,
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ →
    p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ →
    p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882 →
    ((p₁ = 2 ∧ p₂ = 5 ∧ p₃ = 19 ∧ p₄ = 37) ∨
     (p₁ = 2 ∧ p₂ = 11 ∧ p₃ = 19 ∧ p₄ = 31) ∨
     (p₁ = 2 ∧ p₂ = 13 ∧ p₃ = 19 ∧ p₄ = 29)) :=
by sorry

end NUMINAMATH_CALUDE_prime_quadruples_sum_882_l426_42681


namespace NUMINAMATH_CALUDE_abs_four_implies_plus_minus_four_l426_42669

theorem abs_four_implies_plus_minus_four (x : ℝ) : |x| = 4 → x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_abs_four_implies_plus_minus_four_l426_42669


namespace NUMINAMATH_CALUDE_solution_count_of_system_l426_42682

theorem solution_count_of_system (x y : ℂ) : 
  (y = (x + 1)^2 ∧ x * y + y = 1) → 
  (∃! (xr yr : ℝ), yr = (xr + 1)^2 ∧ xr * yr + yr = 1) ∧
  (∃ (xc1 yc1 xc2 yc2 : ℂ), 
    (xc1 ≠ xc2) ∧
    (yc1 = (xc1 + 1)^2 ∧ xc1 * yc1 + yc1 = 1) ∧
    (yc2 = (xc2 + 1)^2 ∧ xc2 * yc2 + yc2 = 1) ∧
    (xc1.im ≠ 0 ∧ xc2.im ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_solution_count_of_system_l426_42682


namespace NUMINAMATH_CALUDE_prob_is_one_fourth_l426_42676

/-- The number of cards -/
def n : ℕ := 72

/-- The set of card numbers -/
def S : Finset ℕ := Finset.range n

/-- The set of multiples of 6 in S -/
def A : Finset ℕ := S.filter (fun x => x % 6 = 0)

/-- The set of multiples of 8 in S -/
def B : Finset ℕ := S.filter (fun x => x % 8 = 0)

/-- The probability of selecting a card that is a multiple of 6 or 8 or both -/
def prob : ℚ := (A ∪ B).card / S.card

theorem prob_is_one_fourth : prob = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_is_one_fourth_l426_42676


namespace NUMINAMATH_CALUDE_result_2011th_operation_l426_42657

/-- Represents the sequence of operations starting with 25 -/
def operationSequence : ℕ → ℕ
| 0 => 25
| 1 => 133
| 2 => 55
| 3 => 250
| (n + 4) => operationSequence n

/-- The result of the nth operation in the sequence -/
def nthOperationResult (n : ℕ) : ℕ := operationSequence (n % 4)

theorem result_2011th_operation :
  nthOperationResult 2011 = 133 := by sorry

end NUMINAMATH_CALUDE_result_2011th_operation_l426_42657


namespace NUMINAMATH_CALUDE_angle_between_vectors_solution_l426_42604

def angle_between_vectors (problem : Unit) : Prop :=
  ∃ (a b : ℝ × ℝ),
    let dot_product := (a.1 * b.1 + a.2 * b.2)
    let magnitude := fun v : ℝ × ℝ => Real.sqrt (v.1^2 + v.2^2)
    let angle := Real.arccos (dot_product / (magnitude a * magnitude b))
    (a.1 * a.1 + a.2 * a.2 - 2 * (a.1 * b.1 + a.2 * b.2) = 3) ∧
    (magnitude a = 1) ∧
    (b = (1, 1)) ∧
    (angle = 3 * Real.pi / 4)

theorem angle_between_vectors_solution : angle_between_vectors () := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_solution_l426_42604


namespace NUMINAMATH_CALUDE_line_parametric_to_standard_l426_42687

/-- Given a line with parametric equations x = -2 - 2t and y = 3 + √2 t,
    prove that its standard form is x + √2 y + 2 - 3√2 = 0 -/
theorem line_parametric_to_standard :
  ∀ (t x y : ℝ),
  (x = -2 - 2*t ∧ y = 3 + Real.sqrt 2 * t) →
  x + Real.sqrt 2 * y + 2 - 3 * Real.sqrt 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_parametric_to_standard_l426_42687


namespace NUMINAMATH_CALUDE_brendans_dad_fish_count_l426_42639

theorem brendans_dad_fish_count :
  ∀ (morning afternoon thrown_back total dad_catch : ℕ),
    morning = 8 →
    afternoon = 5 →
    thrown_back = 3 →
    total = 23 →
    dad_catch = total - (morning + afternoon - thrown_back) →
    dad_catch = 13 := by
  sorry

end NUMINAMATH_CALUDE_brendans_dad_fish_count_l426_42639


namespace NUMINAMATH_CALUDE_marble_probability_l426_42642

theorem marble_probability (total : ℕ) (p_white p_green p_yellow p_orange : ℚ) :
  total = 500 →
  p_white = 1/4 →
  p_green = 1/5 →
  p_yellow = 1/6 →
  p_orange = 1/10 →
  let p_red_blue := 1 - (p_white + p_green + p_yellow + p_orange)
  p_red_blue = 71/250 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l426_42642


namespace NUMINAMATH_CALUDE_silver_coins_removed_l426_42662

theorem silver_coins_removed (total_coins : ℕ) (initial_gold_percent : ℚ) (final_gold_percent : ℚ) :
  total_coins = 200 →
  initial_gold_percent = 2 / 100 →
  final_gold_percent = 20 / 100 →
  (total_coins : ℚ) * initial_gold_percent = (total_coins - (total_coins : ℚ) * initial_gold_percent * (1 / final_gold_percent - 1)) * final_gold_percent →
  ⌊total_coins - (total_coins : ℚ) * initial_gold_percent * (1 / final_gold_percent)⌋ = 180 :=
by sorry

end NUMINAMATH_CALUDE_silver_coins_removed_l426_42662


namespace NUMINAMATH_CALUDE_park_perimeter_calculation_l426_42661

/-- The perimeter of a rectangular park with given length and breadth. -/
def park_perimeter (length breadth : ℝ) : ℝ :=
  2 * (length + breadth)

/-- Theorem stating that the perimeter of a rectangular park with length 300 m and breadth 200 m is 1000 m. -/
theorem park_perimeter_calculation :
  park_perimeter 300 200 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_park_perimeter_calculation_l426_42661


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l426_42631

/-- A rectangular solid with prime edge lengths and volume 231 has surface area 262 -/
theorem rectangular_solid_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 231 →
  2 * (a * b + b * c + a * c) = 262 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l426_42631


namespace NUMINAMATH_CALUDE_negative_values_iff_a_outside_interval_l426_42600

/-- A quadratic function f(x) = x^2 - ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

/-- The function f takes negative values -/
def takes_negative_values (a : ℝ) : Prop :=
  ∃ x, f a x < 0

/-- The main theorem: f takes negative values iff a > 2 or a < -2 -/
theorem negative_values_iff_a_outside_interval :
  ∀ a : ℝ, takes_negative_values a ↔ (a > 2 ∨ a < -2) :=
sorry

end NUMINAMATH_CALUDE_negative_values_iff_a_outside_interval_l426_42600


namespace NUMINAMATH_CALUDE_pen_notebook_ratio_l426_42667

theorem pen_notebook_ratio (num_notebooks : ℕ) : 
  num_notebooks = 40 → 
  (5 : ℚ) / 4 * num_notebooks = 50 := by
  sorry

end NUMINAMATH_CALUDE_pen_notebook_ratio_l426_42667


namespace NUMINAMATH_CALUDE_max_correct_answers_l426_42623

/-- Represents the scoring system and results of a math contest. -/
structure MathContest where
  total_questions : ℕ
  correct_points : ℤ
  blank_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Theorem stating the maximum number of correct answers for the given contest conditions. -/
theorem max_correct_answers (contest : MathContest)
  (h1 : contest.total_questions = 60)
  (h2 : contest.correct_points = 5)
  (h3 : contest.blank_points = 0)
  (h4 : contest.incorrect_points = -2)
  (h5 : contest.total_score = 139) :
  ∃ (max_correct : ℕ), max_correct = 37 ∧
  ∀ (correct : ℕ), correct ≤ contest.total_questions →
    (∃ (blank incorrect : ℕ),
      correct + blank + incorrect = contest.total_questions ∧
      contest.correct_points * correct + contest.blank_points * blank + contest.incorrect_points * incorrect = contest.total_score) →
    correct ≤ max_correct :=
sorry

end NUMINAMATH_CALUDE_max_correct_answers_l426_42623


namespace NUMINAMATH_CALUDE_minimum_children_for_shared_birthday_l426_42614

theorem minimum_children_for_shared_birthday (n : ℕ) : 
  (∀ f : Fin n → Fin 366, ∃ d : Fin 366, (∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ f i = f j ∧ f j = f k)) ↔ 
  n ≥ 733 :=
sorry

end NUMINAMATH_CALUDE_minimum_children_for_shared_birthday_l426_42614


namespace NUMINAMATH_CALUDE_system_solution_l426_42689

theorem system_solution : ∃! (x y : ℝ), 
  (2 * Real.sqrt (2 * x + 3 * y) + Real.sqrt (5 - x - y) = 7) ∧ 
  (3 * Real.sqrt (5 - x - y) - Real.sqrt (2 * x + y - 3) = 1) ∧ 
  x = 3 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l426_42689


namespace NUMINAMATH_CALUDE_weekend_to_weekday_ratio_is_two_to_one_l426_42647

/-- Represents the earnings of an Italian restaurant over a month. -/
structure RestaurantEarnings where
  weekday_earnings : ℕ  -- Daily earnings on weekdays
  total_earnings : ℕ    -- Total earnings for the month
  weeks_per_month : ℕ   -- Number of weeks in the month

/-- Calculates the ratio of weekend to weekday earnings. -/
def weekend_to_weekday_ratio (r : RestaurantEarnings) : ℚ :=
  let weekday_total := r.weekday_earnings * 5 * r.weeks_per_month
  let weekend_total := r.total_earnings - weekday_total
  let weekend_daily := weekend_total / (2 * r.weeks_per_month)
  weekend_daily / r.weekday_earnings

/-- Theorem stating that the ratio of weekend to weekday earnings is 2:1. -/
theorem weekend_to_weekday_ratio_is_two_to_one 
  (r : RestaurantEarnings) 
  (h1 : r.weekday_earnings = 600)
  (h2 : r.total_earnings = 21600)
  (h3 : r.weeks_per_month = 4) : 
  weekend_to_weekday_ratio r = 2 := by
  sorry

end NUMINAMATH_CALUDE_weekend_to_weekday_ratio_is_two_to_one_l426_42647


namespace NUMINAMATH_CALUDE_smallest_e_value_l426_42625

theorem smallest_e_value (a b c d e : ℤ) : 
  (∃ (x : ℝ), a * x^4 + b * x^3 + c * x^2 + d * x + e = 0) →
  (a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0) →
  (a * 4^4 + b * 4^3 + c * 4^2 + d * 4 + e = 0) →
  (a * 8^4 + b * 8^3 + c * 8^2 + d * 8 + e = 0) →
  (a * (-1/4)^4 + b * (-1/4)^3 + c * (-1/4)^2 + d * (-1/4) + e = 0) →
  e > 0 →
  e ≥ 96 := by
sorry

end NUMINAMATH_CALUDE_smallest_e_value_l426_42625


namespace NUMINAMATH_CALUDE_circle_equation_to_circle_params_l426_42606

/-- A circle in the 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in the form ax² + by² + cx + dy + e = 0. -/
def CircleEquation (a b c d e : ℝ) : (ℝ × ℝ) → Prop :=
  fun p => a * p.1^2 + b * p.2^2 + c * p.1 + d * p.2 + e = 0

theorem circle_equation_to_circle_params :
  ∃! (circle : Circle),
    (∀ p, CircleEquation 1 1 (-4) 2 0 p ↔ (p.1 - circle.center.1)^2 + (p.2 - circle.center.2)^2 = circle.radius^2) ∧
    circle.center = (2, -1) ∧
    circle.radius = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_to_circle_params_l426_42606


namespace NUMINAMATH_CALUDE_problem_solution_l426_42664

-- Define the proposition for the first statement
def converse_square_sum_zero (x y : ℝ) : Prop :=
  x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Define the proposition for the second statement
def intersection_subset (A B : Set α) : Prop :=
  A ∩ B = A → A ⊆ B

-- Theorem combining both propositions
theorem problem_solution :
  (∀ x y : ℝ, converse_square_sum_zero x y) ∧
  (∀ A B : Set α, intersection_subset A B) :=
by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l426_42664


namespace NUMINAMATH_CALUDE_glass_volume_l426_42679

theorem glass_volume (V : ℝ) 
  (h1 : 0.4 * V = V - 0.6 * V) -- pessimist's glass is 60% empty
  (h2 : 0.6 * V - 0.4 * V = 46) -- difference in water volume is 46 ml
  : V = 230 := by
  sorry

end NUMINAMATH_CALUDE_glass_volume_l426_42679


namespace NUMINAMATH_CALUDE_table_tennis_match_results_l426_42646

/-- Represents a "best-of-3" table tennis match -/
structure TableTennisMatch where
  prob_a_win : ℝ
  prob_b_win : ℝ

/-- The probability of player A winning a single game -/
def prob_a_win (m : TableTennisMatch) : ℝ := m.prob_a_win

/-- The probability of player B winning a single game -/
def prob_b_win (m : TableTennisMatch) : ℝ := m.prob_b_win

/-- The probability of player A winning the entire match -/
def prob_a_win_match (m : TableTennisMatch) : ℝ :=
  (m.prob_a_win)^2 + 2 * m.prob_b_win * (m.prob_a_win)^2

/-- The expected number of games won by player A -/
def expected_games_won_a (m : TableTennisMatch) : ℝ :=
  1 * (2 * m.prob_a_win * (m.prob_b_win)^2) + 2 * ((m.prob_a_win)^2 + 2 * m.prob_b_win * (m.prob_a_win)^2)

/-- The variance of the number of games won by player A -/
def variance_games_won_a (m : TableTennisMatch) : ℝ :=
  (m.prob_b_win)^2 * (0 - expected_games_won_a m)^2 +
  (2 * m.prob_a_win * (m.prob_b_win)^2) * (1 - expected_games_won_a m)^2 +
  ((m.prob_a_win)^2 + 2 * m.prob_b_win * (m.prob_a_win)^2) * (2 - expected_games_won_a m)^2

theorem table_tennis_match_results (m : TableTennisMatch) 
  (h1 : m.prob_a_win = 0.6) 
  (h2 : m.prob_b_win = 0.4) : 
  prob_a_win_match m = 0.648 ∧ 
  expected_games_won_a m = 1.5 ∧ 
  variance_games_won_a m = 0.57 := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_match_results_l426_42646


namespace NUMINAMATH_CALUDE_this_year_cabbage_production_l426_42670

/-- Represents a square garden where cabbages are grown -/
structure CabbageGarden where
  side : ℕ -- Side length of the square garden

/-- Calculates the number of cabbages in a square garden -/
def cabbageCount (garden : CabbageGarden) : ℕ := garden.side ^ 2

/-- Theorem stating the number of cabbages produced this year -/
theorem this_year_cabbage_production 
  (last_year : CabbageGarden) 
  (this_year : CabbageGarden) 
  (h1 : cabbageCount this_year - cabbageCount last_year = 211) :
  cabbageCount this_year = 11236 := by
  sorry


end NUMINAMATH_CALUDE_this_year_cabbage_production_l426_42670


namespace NUMINAMATH_CALUDE_board_division_theorem_l426_42648

/-- Represents a cell on the board -/
structure Cell :=
  (x : Nat) (y : Nat) (shaded : Bool)

/-- Represents the board -/
def Board := List Cell

/-- Represents a rectangle on the board -/
structure Rectangle :=
  (topLeft : Cell) (width : Nat) (height : Nat)

/-- The initial board configuration -/
def initialBoard : Board := sorry

/-- Check if a cell is within a rectangle -/
def isInRectangle (cell : Cell) (rect : Rectangle) : Bool := sorry

/-- Count shaded cells in a rectangle -/
def countShadedCells (board : Board) (rect : Rectangle) : Nat := sorry

/-- Check if two rectangles are identical -/
def areIdenticalRectangles (rect1 rect2 : Rectangle) : Bool := sorry

/-- Main theorem -/
theorem board_division_theorem (board : Board) :
  ∃ (rect1 rect2 rect3 rect4 : Rectangle),
    (rect1.width = 4 ∧ rect1.height = 2) ∧
    (rect2.width = 4 ∧ rect2.height = 2) ∧
    (rect3.width = 4 ∧ rect3.height = 2) ∧
    (rect4.width = 4 ∧ rect4.height = 2) ∧
    areIdenticalRectangles rect1 rect2 ∧
    areIdenticalRectangles rect1 rect3 ∧
    areIdenticalRectangles rect1 rect4 ∧
    countShadedCells board rect1 = 3 ∧
    countShadedCells board rect2 = 3 ∧
    countShadedCells board rect3 = 3 ∧
    countShadedCells board rect4 = 3 :=
  sorry

end NUMINAMATH_CALUDE_board_division_theorem_l426_42648


namespace NUMINAMATH_CALUDE_corrected_mean_l426_42656

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 40 ∧ original_mean = 36 ∧ incorrect_value = 20 ∧ correct_value = 34 →
  (n * original_mean + (correct_value - incorrect_value)) / n = 36.35 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l426_42656


namespace NUMINAMATH_CALUDE_sqrt_division_equality_l426_42636

theorem sqrt_division_equality : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_division_equality_l426_42636


namespace NUMINAMATH_CALUDE_circle_in_circle_theorem_l426_42634

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a point
def Point := ℝ × ℝ

-- Define what it means for a point to be inside a circle
def is_inside (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 < c.radius^2

-- Define what it means for a point to be on a circle
def is_on (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define what it means for one circle to be contained in another
def is_contained (c1 c2 : Circle) : Prop :=
  ∀ (p : Point), is_on p c1 → is_inside p c2

-- State the theorem
theorem circle_in_circle_theorem (ω : Circle) (A B : Point) 
  (h1 : is_inside A ω) (h2 : is_inside B ω) : 
  ∃ (ω' : Circle), is_on A ω' ∧ is_on B ω' ∧ is_contained ω' ω := by
  sorry

end NUMINAMATH_CALUDE_circle_in_circle_theorem_l426_42634


namespace NUMINAMATH_CALUDE_first_divisor_problem_l426_42649

theorem first_divisor_problem (x : ℕ) : x = 31 ↔ 
  x > 9 ∧ 
  x < 282 ∧
  282 % x = 3 ∧
  282 % 9 = 3 ∧
  279 % x = 0 ∧
  ∀ y : ℕ, y > 9 ∧ y < x → (282 % y ≠ 3 ∨ 279 % y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l426_42649


namespace NUMINAMATH_CALUDE_no_real_solutions_l426_42673

theorem no_real_solutions : ∀ s : ℝ, s ≠ 2 → (s^2 - 5*s - 10) / (s - 2) ≠ 3*s + 6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l426_42673


namespace NUMINAMATH_CALUDE_vector_addition_l426_42612

/-- Given two 2D vectors a and b, prove that 2b + 3a equals (6,1) -/
theorem vector_addition (a b : ℝ × ℝ) (ha : a = (2, 1)) (hb : b = (0, -1)) :
  2 • b + 3 • a = (6, 1) := by sorry

end NUMINAMATH_CALUDE_vector_addition_l426_42612


namespace NUMINAMATH_CALUDE_focus_of_standard_parabola_l426_42683

/-- The focus of the parabola y = x^2 is at the point (0, 1/4). -/
theorem focus_of_standard_parabola :
  let f : ℝ × ℝ := (0, 1/4)
  let parabola := {(x, y) : ℝ × ℝ | y = x^2}
  f ∈ parabola ∧ ∀ p ∈ parabola, dist p f = dist p (0, -1/4) :=
by sorry

end NUMINAMATH_CALUDE_focus_of_standard_parabola_l426_42683


namespace NUMINAMATH_CALUDE_intersection_range_l426_42674

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 + 1 ≥ Real.sqrt (2 * (p.1^2 + p.2^2))}
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1 - a| + |p.2 - 1| ≤ 1}

-- State the theorem
theorem intersection_range (a : ℝ) :
  (M ∩ N a).Nonempty ↔ a ∈ Set.Icc (1 - Real.sqrt 6) (3 + Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l426_42674


namespace NUMINAMATH_CALUDE_problem_statement_l426_42640

theorem problem_statement (a b : ℝ) (h : 5 * a - 3 * b + 2 = 0) : 
  10 * a - 6 * b - 3 = -7 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l426_42640


namespace NUMINAMATH_CALUDE_total_onions_grown_l426_42616

/-- The total number of onions grown by Sara, Sally, and Fred is 18. -/
theorem total_onions_grown (sara_onions : ℕ) (sally_onions : ℕ) (fred_onions : ℕ)
  (h1 : sara_onions = 4)
  (h2 : sally_onions = 5)
  (h3 : fred_onions = 9) :
  sara_onions + sally_onions + fred_onions = 18 :=
by sorry

end NUMINAMATH_CALUDE_total_onions_grown_l426_42616


namespace NUMINAMATH_CALUDE_cadastral_value_calculation_l426_42655

/-- Calculates the cadastral value of a land plot given the tax amount and tax rate -/
theorem cadastral_value_calculation (tax_amount : ℝ) (tax_rate : ℝ) :
  tax_amount = 4500 →
  tax_rate = 0.003 →
  tax_amount = tax_rate * 1500000 := by
  sorry

#check cadastral_value_calculation

end NUMINAMATH_CALUDE_cadastral_value_calculation_l426_42655


namespace NUMINAMATH_CALUDE_min_black_edges_on_border_l426_42629

/-- Represents a small square in the grid -/
structure SmallSquare where
  blackTriangles : Fin 4
  blackEdges : Fin 4

/-- Represents the 5x5 grid -/
def Grid := Matrix (Fin 5) (Fin 5) SmallSquare

/-- Checks if two adjacent small squares have consistent edge colors -/
def consistentEdges (s1 s2 : SmallSquare) : Prop :=
  s1.blackEdges = s2.blackEdges

/-- Counts the number of black edges on the border of the grid -/
def countBorderBlackEdges (g : Grid) : ℕ :=
  sorry

/-- The main theorem stating the minimum number of black edges on the border -/
theorem min_black_edges_on_border (g : Grid) 
  (h1 : ∀ (i j : Fin 5), (g i j).blackTriangles = 3)
  (h2 : ∀ (i j k l : Fin 5), (j = k + 1 ∨ i = l + 1) → consistentEdges (g i j) (g k l)) :
  countBorderBlackEdges g ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_min_black_edges_on_border_l426_42629


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l426_42643

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, 1 + x^5 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l426_42643


namespace NUMINAMATH_CALUDE_ellipse_k_range_l426_42613

-- Define the curve
def ellipse_equation (x y k : ℝ) : Prop :=
  x^2 / (1 - k) + y^2 / (1 + k) = 1

-- Define the conditions for an ellipse
def is_ellipse (k : ℝ) : Prop :=
  1 - k > 0 ∧ 1 + k > 0 ∧ 1 - k ≠ 1 + k

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ ((-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l426_42613


namespace NUMINAMATH_CALUDE_only_statement_one_true_l426_42694

variable (b x y : ℝ)

theorem only_statement_one_true :
  (∀ x y, b * (x + y) = b * x + b * y) ∧
  (∃ x y, b^(x + y) ≠ b^x + b^y) ∧
  (∃ x y, Real.log (x + y) ≠ Real.log x + Real.log y) ∧
  (∃ x y, Real.log x / Real.log y ≠ Real.log (x * y)) ∧
  (∃ x y, b * (x / y) ≠ (b * x) / (b * y)) :=
by sorry

end NUMINAMATH_CALUDE_only_statement_one_true_l426_42694


namespace NUMINAMATH_CALUDE_stratified_sample_male_count_l426_42638

/-- Represents a stratified sample from a population of students -/
structure StratifiedSample where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ
  sample_female : ℕ
  sample_male : ℕ

/-- Theorem stating that in a given stratified sample, the number of male students in the sample is 18 -/
theorem stratified_sample_male_count 
  (sample : StratifiedSample) 
  (h1 : sample.total_students = 680)
  (h2 : sample.male_students = 360)
  (h3 : sample.female_students = 320)
  (h4 : sample.sample_female = 16)
  (h5 : sample.female_students * sample.sample_male = sample.male_students * sample.sample_female) :
  sample.sample_male = 18 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_male_count_l426_42638


namespace NUMINAMATH_CALUDE_max_revenue_is_70_l426_42665

/-- Represents the advertising problem for a company --/
structure AdvertisingProblem where
  maxTime : ℝ
  maxCost : ℝ
  rateA : ℝ
  rateB : ℝ
  revenueA : ℝ
  revenueB : ℝ

/-- Calculates the maximum revenue for the given advertising problem --/
def maxRevenue (p : AdvertisingProblem) : ℝ :=
  let x := 100  -- Time for TV Station A
  let y := 200  -- Time for TV Station B
  p.revenueA * x + p.revenueB * y

/-- Theorem stating that the maximum revenue is 70 million yuan --/
theorem max_revenue_is_70 (p : AdvertisingProblem) 
    (h1 : p.maxTime = 300)
    (h2 : p.maxCost = 900000)
    (h3 : p.rateA = 500)
    (h4 : p.rateB = 200)
    (h5 : p.revenueA = 0.3)
    (h6 : p.revenueB = 0.2) :
  maxRevenue p = 70 := by
  sorry

#eval maxRevenue { maxTime := 300, maxCost := 900000, rateA := 500, rateB := 200, revenueA := 0.3, revenueB := 0.2 }

end NUMINAMATH_CALUDE_max_revenue_is_70_l426_42665


namespace NUMINAMATH_CALUDE_athlete_weights_problem_l426_42659

theorem athlete_weights_problem (a b c : ℝ) (k₁ k₂ k₃ : ℤ) : 
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  (a + c) / 2 = 44 →
  a + b = 5 * k₁ →
  b + c = 5 * k₂ →
  a + c = 5 * k₃ →
  b = 40 := by
  sorry

end NUMINAMATH_CALUDE_athlete_weights_problem_l426_42659


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l426_42677

theorem sin_2alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.tan (π / 4 - α) = 1 / 3) : 
  Real.sin (2 * α) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l426_42677


namespace NUMINAMATH_CALUDE_rectangle_area_l426_42601

theorem rectangle_area (square_area : ℝ) (h1 : square_area = 36) : ∃ (rect_width rect_length rect_area : ℝ),
  rect_width ^ 2 = square_area ∧
  rect_length = 3 * rect_width ∧
  rect_area = rect_width * rect_length ∧
  rect_area = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l426_42601


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l426_42611

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 5 * Complex.I) * (a + b * Complex.I) = y * Complex.I) : 
  a / b = -5/3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l426_42611


namespace NUMINAMATH_CALUDE_hemisphere_to_spheres_l426_42695

/-- The radius of a sphere when a hemisphere is divided into equal parts -/
theorem hemisphere_to_spheres (r : Real) (n : Nat) (r_small : Real) : 
  r = 2 → n = 18 → (2/3 * π * r^3) = (n * (4/3 * π * r_small^3)) → r_small = (2/3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_to_spheres_l426_42695


namespace NUMINAMATH_CALUDE_border_area_is_198_l426_42645

-- Define the dimensions of the photograph
def photo_height : ℕ := 12
def photo_width : ℕ := 15

-- Define the width of the border
def border_width : ℕ := 3

-- Define the area of the border
def border_area : ℕ := 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width

-- Theorem statement
theorem border_area_is_198 : border_area = 198 := by
  sorry

end NUMINAMATH_CALUDE_border_area_is_198_l426_42645


namespace NUMINAMATH_CALUDE_factorial_simplification_l426_42622

theorem factorial_simplification (N : ℕ) :
  (Nat.factorial (N + 1)) / ((Nat.factorial (N - 1)) * (N + 2)) = N * (N + 1) / (N + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l426_42622


namespace NUMINAMATH_CALUDE_highway_length_l426_42633

/-- The length of a highway where two cars meet --/
theorem highway_length (v1 v2 t : ℝ) (h1 : v1 = 25) (h2 : v2 = 45) (h3 : t = 2.5) :
  (v1 + v2) * t = 175 := by
  sorry

end NUMINAMATH_CALUDE_highway_length_l426_42633


namespace NUMINAMATH_CALUDE_gum_cost_l426_42672

/-- The cost of gum in cents -/
def cost_per_piece : ℕ := 2

/-- The number of pieces of gum -/
def num_pieces : ℕ := 500

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The cost of 500 pieces of gum is 1000 cents and 10 dollars -/
theorem gum_cost :
  (num_pieces * cost_per_piece = 1000) ∧
  (num_pieces * cost_per_piece / cents_per_dollar = 10) :=
by sorry

end NUMINAMATH_CALUDE_gum_cost_l426_42672


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l426_42603

theorem square_circle_area_ratio (s r : ℝ) (h : 4 * s = 2 * Real.pi * r) :
  s^2 / (Real.pi * r^2) = 4 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l426_42603


namespace NUMINAMATH_CALUDE_exists_arrangement_for_23_l426_42680

/-- Fibonacci-like sequence with a specific recurrence relation -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence satisfying the required property for P = 23 -/
theorem exists_arrangement_for_23 : ∃ (F : ℕ → ℤ), F 12 ≡ 0 [ZMOD 23] := by
  sorry

end NUMINAMATH_CALUDE_exists_arrangement_for_23_l426_42680


namespace NUMINAMATH_CALUDE_binary_to_decimal_1010101_l426_42641

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number -/
def binary_num : List Nat := [1, 0, 1, 0, 1, 0, 1]

theorem binary_to_decimal_1010101 :
  binary_to_decimal (binary_num.reverse) = 85 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_1010101_l426_42641


namespace NUMINAMATH_CALUDE_movie_theater_sections_l426_42652

theorem movie_theater_sections (total_seats : ℕ) (seats_per_section : ℕ) (h1 : total_seats = 270) (h2 : seats_per_section = 30) :
  total_seats / seats_per_section = 9 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_sections_l426_42652


namespace NUMINAMATH_CALUDE_max_value_theorem_l426_42620

theorem max_value_theorem (a b c : ℝ) (h : a * b * c + a + c - b = 0) :
  ∃ (max : ℝ), max = 5/4 ∧ 
  ∀ (x y z : ℝ), x * y * z + x + z - y = 0 →
  (1 / (1 + x^2) - 1 / (1 + y^2) + 1 / (1 + z^2)) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l426_42620


namespace NUMINAMATH_CALUDE_sqrt_product_quotient_l426_42654

theorem sqrt_product_quotient :
  3 * Real.sqrt 5 * (2 * Real.sqrt 15) / Real.sqrt 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_quotient_l426_42654


namespace NUMINAMATH_CALUDE_factorization_m_squared_minus_3m_l426_42693

theorem factorization_m_squared_minus_3m (m : ℝ) : m^2 - 3*m = m*(m - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_m_squared_minus_3m_l426_42693


namespace NUMINAMATH_CALUDE_exists_int_between_sqrt2_and_sqrt17_l426_42663

theorem exists_int_between_sqrt2_and_sqrt17 : ∃ n : ℤ, Real.sqrt 2 < n ∧ n < Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_exists_int_between_sqrt2_and_sqrt17_l426_42663


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l426_42608

theorem fraction_product_simplification :
  (3 : ℚ) / 4 * 4 / 5 * 5 / 6 * 6 / 7 * 7 / 9 = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l426_42608


namespace NUMINAMATH_CALUDE_spherical_rotation_l426_42637

/-- Given a point with rectangular coordinates (-3, 2, 5) and corresponding 
    spherical coordinates (r, θ, φ), the point with spherical coordinates 
    (r, θ, φ-π/2) has rectangular coordinates (-3, 2, -5). -/
theorem spherical_rotation (r θ φ : Real) : 
  r * Real.sin φ * Real.cos θ = -3 ∧ 
  r * Real.sin φ * Real.sin θ = 2 ∧ 
  r * Real.cos φ = 5 → 
  r * Real.sin (φ - π/2) * Real.cos θ = -3 ∧
  r * Real.sin (φ - π/2) * Real.sin θ = 2 ∧
  r * Real.cos (φ - π/2) = -5 := by
sorry

end NUMINAMATH_CALUDE_spherical_rotation_l426_42637


namespace NUMINAMATH_CALUDE_infinitely_many_rational_pairs_sum_equals_product_l426_42607

theorem infinitely_many_rational_pairs_sum_equals_product :
  ∃ f : ℚ → ℚ × ℚ, Function.Injective f ∧ ∀ z, (f z).1 + (f z).2 = (f z).1 * (f z).2 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_rational_pairs_sum_equals_product_l426_42607


namespace NUMINAMATH_CALUDE_license_plate_count_l426_42635

/-- The number of letters in the alphabet -/
def number_of_letters : ℕ := 26

/-- The number of digits (0-9) -/
def number_of_digits : ℕ := 10

/-- The number of even (or odd) digits -/
def number_of_even_digits : ℕ := 5

/-- The total number of license plates with 2 letters followed by 2 digits,
    where one digit is odd and the other is even -/
def total_license_plates : ℕ := number_of_letters^2 * number_of_digits * number_of_even_digits

theorem license_plate_count : total_license_plates = 33800 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l426_42635


namespace NUMINAMATH_CALUDE_total_movies_l426_42621

-- Define the number of movies Timothy watched in 2009
def timothy_2009 : ℕ := 24

-- Define the number of movies Timothy watched in 2010
def timothy_2010 : ℕ := timothy_2009 + 7

-- Define the number of movies Theresa watched in 2009
def theresa_2009 : ℕ := timothy_2009 / 2

-- Define the number of movies Theresa watched in 2010
def theresa_2010 : ℕ := timothy_2010 * 2

-- Theorem to prove
theorem total_movies : timothy_2009 + timothy_2010 + theresa_2009 + theresa_2010 = 129 := by
  sorry

end NUMINAMATH_CALUDE_total_movies_l426_42621


namespace NUMINAMATH_CALUDE_interior_triangle_area_l426_42628

theorem interior_triangle_area (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) :
  (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_interior_triangle_area_l426_42628


namespace NUMINAMATH_CALUDE_inequality_solution_positive_for_all_x_zeros_greater_than_5_2_l426_42666

-- Define the function f
def f (k x : ℝ) : ℝ := x^2 - k*x + (2*k - 3)

-- Statement 1
theorem inequality_solution (x : ℝ) :
  f (3/2) x > 0 ↔ x < 0 ∨ x > 3/2 := by sorry

-- Statement 2
theorem positive_for_all_x (k : ℝ) :
  (∀ x, f k x > 0) ↔ 2 < k ∧ k < 6 := by sorry

-- Statement 3
theorem zeros_greater_than_5_2 (k : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 ∧ x₁ > 5/2 ∧ x₂ > 5/2) ↔
  6 < k ∧ k < 13/2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_positive_for_all_x_zeros_greater_than_5_2_l426_42666


namespace NUMINAMATH_CALUDE_platform_length_specific_platform_length_l426_42686

/-- The length of a platform given train speed, crossing time, and train length -/
theorem platform_length 
  (train_speed_kmph : ℝ) 
  (crossing_time_s : ℝ) 
  (train_length_m : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * crossing_time_s
  total_distance - train_length_m

/-- Proof of the specific platform length problem -/
theorem specific_platform_length : 
  platform_length 72 26 260.0416 = 259.9584 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_specific_platform_length_l426_42686


namespace NUMINAMATH_CALUDE_expression_evaluation_l426_42692

theorem expression_evaluation (a b c : ℝ) : 
  let d := a + b + c
  2 * (a^2 * b^2 + a^2 * c^2 + a^2 * d^2 + b^2 * c^2 + b^2 * d^2 + c^2 * d^2) - 
  (a^4 + b^4 + c^4 + d^4) + 8 * a * b * c * d = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l426_42692


namespace NUMINAMATH_CALUDE_jessie_min_score_l426_42690

/-- Represents the test scores and conditions for Jessie's problem -/
structure TestScores where
  max_score : ℕ
  first_three : Fin 3 → ℕ
  total_tests : ℕ
  target_average : ℕ

/-- The minimum score needed on one of the remaining tests -/
def min_score (ts : TestScores) : ℕ :=
  let total_needed := ts.target_average * ts.total_tests
  let current_total := (ts.first_three 0) + (ts.first_three 1) + (ts.first_three 2)
  let remaining_total := total_needed - current_total
  remaining_total - 2 * ts.max_score

/-- Theorem stating the minimum score Jessie needs to achieve -/
theorem jessie_min_score :
  let ts : TestScores := {
    max_score := 120,
    first_three := ![88, 105, 96],
    total_tests := 6,
    target_average := 90
  }
  min_score ts = 11 := by sorry

end NUMINAMATH_CALUDE_jessie_min_score_l426_42690


namespace NUMINAMATH_CALUDE_total_different_books_l426_42650

/-- The number of different books read by three people given their individual book counts and shared book information. -/
def differentBooksRead (tonyBooks deanBooks breannaBooks tonyDeanShared allShared : ℕ) : ℕ :=
  tonyBooks + deanBooks + breannaBooks - tonyDeanShared - 2 * allShared

/-- Theorem stating that Tony, Dean, and Breanna read 47 different books in total. -/
theorem total_different_books : 
  differentBooksRead 23 12 17 3 1 = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_different_books_l426_42650


namespace NUMINAMATH_CALUDE_hyperbola_equation_l426_42697

/-- Represents a hyperbola with equation x^2/a^2 - y^2/b^2 = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive_a : a > 0
  h_positive_b : b > 0

/-- Theorem: Given a hyperbola with an asymptote through (2, √3) and a focus at (-√7, 0),
    prove that a = 2 and b = √3 --/
theorem hyperbola_equation (h : Hyperbola)
  (h_asymptote : 2 * h.b = Real.sqrt 3 * h.a)
  (h_focus : h.a ^ 2 - h.b ^ 2 = 7) :
  h.a = 2 ∧ h.b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l426_42697


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l426_42653

theorem imaginary_part_of_z_is_zero (z : ℂ) (h : z * (Complex.I + 1) = 2 / (Complex.I - 1)) :
  z.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l426_42653


namespace NUMINAMATH_CALUDE_tan_value_from_trig_ratio_l426_42618

theorem tan_value_from_trig_ratio (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 2) :
  Real.tan α = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_trig_ratio_l426_42618


namespace NUMINAMATH_CALUDE_davids_english_marks_l426_42696

def marks_math : ℕ := 65
def marks_physics : ℕ := 82
def marks_chemistry : ℕ := 67
def marks_biology : ℕ := 85
def average_marks : ℕ := 72
def num_subjects : ℕ := 5

theorem davids_english_marks :
  ∃ (marks_english : ℕ),
    (marks_english + marks_math + marks_physics + marks_chemistry + marks_biology) / num_subjects = average_marks ∧
    marks_english = 61 := by
  sorry

end NUMINAMATH_CALUDE_davids_english_marks_l426_42696


namespace NUMINAMATH_CALUDE_gcd_459_357_l426_42658

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l426_42658


namespace NUMINAMATH_CALUDE_min_value_of_function_l426_42660

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 - 3*x + 3) + Real.sqrt (y^2 - 3*y + 3) + Real.sqrt (x^2 - Real.sqrt 3 * x * y + y^2) ≥ Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l426_42660


namespace NUMINAMATH_CALUDE_band_members_count_l426_42610

theorem band_members_count (flute trumpet trombone drummer clarinet french_horn saxophone piano violin guitar : ℕ) : 
  flute = 5 →
  trumpet = 3 * flute →
  trombone = trumpet - 8 →
  drummer = trombone + 11 →
  clarinet = 2 * flute →
  french_horn = trombone + 3 →
  saxophone = (trumpet + trombone) / 2 →
  piano = drummer + 2 →
  violin = french_horn - clarinet →
  guitar = 3 * flute →
  flute + trumpet + trombone + drummer + clarinet + french_horn + saxophone + piano + violin + guitar = 111 := by
sorry

end NUMINAMATH_CALUDE_band_members_count_l426_42610


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l426_42685

/-- Given a square with side length 2y that is divided into a center square
    with side length y and four congruent rectangles, prove that the perimeter
    of one of these rectangles is 3y. -/
theorem rectangle_perimeter (y : ℝ) (y_pos : 0 < y) :
  let large_square_side := 2 * y
  let center_square_side := y
  let rectangle_width := (large_square_side - center_square_side) / 2
  let rectangle_length := center_square_side
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  rectangle_perimeter = 3 * y :=
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l426_42685


namespace NUMINAMATH_CALUDE_extremum_values_l426_42671

/-- The function f(x) = x^3 - ax^2 - bx + a^2 has an extremum of 10 at x = 1 -/
def has_extremum (a b : ℝ) : Prop :=
  let f := fun x => x^3 - a*x^2 - b*x + a^2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∨
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1)

/-- The main theorem -/
theorem extremum_values (a b : ℝ) :
  has_extremum a b ∧ (1^3 - a*1^2 - b*1 + a^2 = 10) →
  (a = 3 ∧ b = -3) ∨ (a = -4 ∧ b = 11) := by sorry


end NUMINAMATH_CALUDE_extremum_values_l426_42671


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l426_42605

/-- Given a quadrilateral ABCD with diagonals intersecting at O, this theorem proves
    that under specific conditions, the length of AD is 2√57. -/
theorem quadrilateral_diagonal_length
  (A B C D O : ℝ × ℝ) -- Points in 2D space
  (h_intersect : (A.1 - C.1) * (B.2 - D.2) = (A.2 - C.2) * (B.1 - D.1)) -- Diagonals intersect
  (h_BO : dist B O = 5)
  (h_OD : dist O D = 7)
  (h_AO : dist A O = 9)
  (h_OC : dist O C = 4)
  (h_AB : dist A B = 6)
  (h_BD : dist B D = 6) :
  dist A D = 2 * Real.sqrt 57 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l426_42605


namespace NUMINAMATH_CALUDE_consecutive_20_divisibility_l426_42644

theorem consecutive_20_divisibility (n : ℤ) : 
  (∃ k ∈ Finset.range 20, (n + k) % 9 = 0) ∧ 
  (∃ k ∈ Finset.range 20, (n + k) % 9 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_20_divisibility_l426_42644


namespace NUMINAMATH_CALUDE_triangle_side_length_l426_42619

/-- Given a triangle ABC with angle A = 60°, area = √3, and b + c = 6, prove that side length a = 2√6 -/
theorem triangle_side_length (b c : ℝ) (h1 : b + c = 6) (h2 : (1/2) * b * c * (Real.sqrt 3 / 2) = Real.sqrt 3) : 
  Real.sqrt (b^2 + c^2 - b * c) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l426_42619


namespace NUMINAMATH_CALUDE_van_rental_equation_l426_42691

theorem van_rental_equation (x : ℕ) (h : x > 0) :
  (180 : ℝ) / x - 180 / (x + 2) = 3 ↔
  (∃ (y : ℝ), y > 0 ∧ 180 / x = y ∧ 180 / (x + 2) = y - 3) :=
by sorry

end NUMINAMATH_CALUDE_van_rental_equation_l426_42691


namespace NUMINAMATH_CALUDE_symmetric_points_m_value_l426_42632

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry with respect to the y-axis
def symmetricAboutYAxis (p1 p2 : Point2D) : Prop :=
  p1.x = -p2.x ∧ p1.y = p2.y

-- Theorem statement
theorem symmetric_points_m_value :
  let A : Point2D := ⟨-3, 4⟩
  let B : Point2D := ⟨3, m⟩
  symmetricAboutYAxis A B → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_m_value_l426_42632


namespace NUMINAMATH_CALUDE_incorrect_accuracy_statement_l426_42630

def accurate_to_nearest_hundred (x : ℝ) : Prop :=
  ∃ n : ℤ, x = (n : ℝ) * 100 ∧ |x - (n : ℝ) * 100| ≤ 50

theorem incorrect_accuracy_statement :
  ¬(accurate_to_nearest_hundred 2130) :=
sorry

end NUMINAMATH_CALUDE_incorrect_accuracy_statement_l426_42630


namespace NUMINAMATH_CALUDE_smallest_possible_b_l426_42688

def is_valid_polynomial (Q : ℤ → ℤ) (b : ℕ) : Prop :=
  b > 0 ∧
  Q 0 = b ∧ Q 4 = b ∧ Q 6 = b ∧ Q 10 = b ∧
  Q 1 = -b ∧ Q 5 = -b ∧ Q 7 = -b ∧ Q 11 = -b

theorem smallest_possible_b :
  ∀ Q : ℤ → ℤ, ∀ b : ℕ,
  is_valid_polynomial Q b →
  (∀ b' : ℕ, is_valid_polynomial Q b' → b ≤ b') →
  b = 1350 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l426_42688


namespace NUMINAMATH_CALUDE_probability_same_number_four_dice_l426_42602

/-- The number of sides on a standard die -/
def standardDieSides : ℕ := 6

/-- The number of dice being rolled -/
def numberOfDice : ℕ := 4

/-- The probability of all dice showing the same number -/
def probabilitySameNumber : ℚ := 1 / (standardDieSides ^ (numberOfDice - 1))

theorem probability_same_number_four_dice :
  probabilitySameNumber = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_number_four_dice_l426_42602


namespace NUMINAMATH_CALUDE_treys_total_time_l426_42609

/-- The number of tasks to clean the house -/
def clean_house_tasks : ℕ := 7

/-- The number of tasks to take a shower -/
def shower_tasks : ℕ := 1

/-- The number of tasks to make dinner -/
def dinner_tasks : ℕ := 4

/-- The time in minutes to complete each task -/
def time_per_task : ℕ := 10

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem: Given the conditions, the total time to complete Trey's list is 2 hours -/
theorem treys_total_time : 
  (clean_house_tasks + shower_tasks + dinner_tasks) * time_per_task / minutes_per_hour = 2 := by
  sorry

end NUMINAMATH_CALUDE_treys_total_time_l426_42609


namespace NUMINAMATH_CALUDE_ratio_problem_l426_42651

theorem ratio_problem (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) :
  (2 * a + b) / (b + 2 * c) = 5 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l426_42651


namespace NUMINAMATH_CALUDE_sequence_properties_l426_42617

/-- Given two sequences a and b with no equal items, and S_n as the sum of the first n terms of a. -/
def Sequence (a b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) * b n = S n + 1

theorem sequence_properties
  (a b : ℕ → ℝ) (S : ℕ → ℝ)
  (h_seq : Sequence a b S)
  (h_a1 : a 1 = 1)
  (h_bn : ∀ n, b n = n / 2)
  (h_geometric : ∃ q ≠ 1, ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : ∃ d, ∀ n, b (n + 1) = b n + d)
  (h_nonzero : ∀ n, a n ≠ 0) :
  (∃ q ≠ 1, ∀ n, (b n + 1 / (1 - q)) = (b 1 + 1 / (1 - q)) * q^(n - 1)) ∧
  (∀ n ≥ 2, a (n + 1) - a n = a n - a (n - 1) ↔ d = 1 / 2) :=
sorry

#check sequence_properties

end NUMINAMATH_CALUDE_sequence_properties_l426_42617


namespace NUMINAMATH_CALUDE_change_in_expression_l426_42615

/-- The original function -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

/-- The change in f when x is replaced by x + b -/
def delta_plus (x b : ℝ) : ℝ := f (x + b) - f x

/-- The change in f when x is replaced by x - b -/
def delta_minus (x b : ℝ) : ℝ := f (x - b) - f x

theorem change_in_expression (x b : ℝ) (h : b > 0) :
  (delta_plus x b = 3*x^2*b + 3*x*b^2 + b^3 - 2*b) ∧
  (delta_minus x b = -3*x^2*b + 3*x*b^2 - b^3 + 2*b) := by
  sorry

end NUMINAMATH_CALUDE_change_in_expression_l426_42615


namespace NUMINAMATH_CALUDE_alphabet_value_proof_l426_42626

/-- Given the alphabet values where H = 8, prove that A = 25 when PACK = 50, PECK = 54, and CAKE = 40 -/
theorem alphabet_value_proof (P A C K E : ℤ) (h1 : P + A + C + K = 50) (h2 : P + E + C + K = 54) (h3 : C + A + K + E = 40) : A = 25 := by
  sorry

end NUMINAMATH_CALUDE_alphabet_value_proof_l426_42626


namespace NUMINAMATH_CALUDE_congruent_triangles_side_lengths_isosceles_triangle_side_lengths_l426_42668

-- Part 1
theorem congruent_triangles_side_lengths (m n : ℝ) :
  (6 :: 8 :: 10 :: []).toFinset = (6 :: (2*m-2) :: (n+1) :: []).toFinset →
  ((m = 5 ∧ n = 9) ∨ (m = 6 ∧ n = 7)) := by sorry

-- Part 2
theorem isosceles_triangle_side_lengths (a b : ℝ) :
  (a = b ∧ a + a + 5 = 16) →
  ((a = 5 ∧ b = 6) ∨ (a = 5.5 ∧ b = 5)) := by sorry

end NUMINAMATH_CALUDE_congruent_triangles_side_lengths_isosceles_triangle_side_lengths_l426_42668


namespace NUMINAMATH_CALUDE_sum_ratio_equality_l426_42675

theorem sum_ratio_equality (a b c x y z : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 25)
  (h2 : x^2 + y^2 + z^2 = 36)
  (h3 : a*x + b*y + c*z = 30) :
  (a + b + c) / (x + y + z) = 5/6 := by
sorry

end NUMINAMATH_CALUDE_sum_ratio_equality_l426_42675


namespace NUMINAMATH_CALUDE_fixed_point_of_f_l426_42627

/-- The logarithm function with base a, where a > 0 and a ≠ 1 -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- The function f(x) = log_a(x+1) - 2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1) - 2

/-- Theorem: For any a > 0 and a ≠ 1, f(x) passes through the point (0, -2) -/
theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_f_l426_42627


namespace NUMINAMATH_CALUDE_jessica_coins_value_l426_42698

/-- Represents the value of a coin in cents -/
def coin_value (is_dime : Bool) : ℕ :=
  if is_dime then 10 else 5

/-- Calculates the total value of coins in cents -/
def total_value (num_nickels num_dimes : ℕ) : ℕ :=
  coin_value false * num_nickels + coin_value true * num_dimes

theorem jessica_coins_value :
  ∀ (num_nickels num_dimes : ℕ),
    num_nickels + num_dimes = 30 →
    total_value num_dimes num_nickels - total_value num_nickels num_dimes = 120 →
    total_value num_nickels num_dimes = 165 := by
  sorry

end NUMINAMATH_CALUDE_jessica_coins_value_l426_42698
