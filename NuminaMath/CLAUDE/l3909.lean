import Mathlib

namespace baseball_games_played_l3909_390908

theorem baseball_games_played (runs_1 runs_4 runs_5 : ℕ) (avg_runs : ℚ) : 
  runs_1 = 1 → runs_4 = 2 → runs_5 = 3 → avg_runs = 4 → 
  (runs_1 * 1 + runs_4 * 4 + runs_5 * 5 : ℚ) / (runs_1 + runs_4 + runs_5) = avg_runs → 
  runs_1 + runs_4 + runs_5 = 6 := by
sorry

end baseball_games_played_l3909_390908


namespace three_oclock_angle_l3909_390930

/-- The angle between the hour hand and minute hand at a given time -/
def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  sorry

theorem three_oclock_angle :
  clock_angle 3 0 = π / 2 := by
  sorry

end three_oclock_angle_l3909_390930


namespace identical_rows_from_increasing_sums_l3909_390932

theorem identical_rows_from_increasing_sums 
  (n : ℕ) 
  (row1 row2 : Fin n → ℝ) 
  (distinct : ∀ i j, i ≠ j → row1 i ≠ row1 j) 
  (increasing_row1 : ∀ i j, i < j → row1 i < row1 j) 
  (same_elements : ∀ x, ∃ i, row1 i = x ↔ ∃ j, row2 j = x) 
  (increasing_sums : ∀ i j, i < j → row1 i + row2 i < row1 j + row2 j) : 
  ∀ i, row1 i = row2 i :=
sorry

end identical_rows_from_increasing_sums_l3909_390932


namespace quadratic_function_properties_l3909_390905

def f (x : ℝ) : ℝ := x^2 + 1

theorem quadratic_function_properties :
  (f 0 = 1) ∧ (∀ x : ℝ, deriv f x > 0) := by
  sorry

end quadratic_function_properties_l3909_390905


namespace sequence_sum_l3909_390944

def geometric_sequence (a : ℕ → ℚ) (r : ℚ) :=
  ∀ n, a (n + 1) = r * a n

theorem sequence_sum (a : ℕ → ℚ) (r : ℚ) :
  geometric_sequence a r →
  a 0 = 16384 →
  a 5 = 16 →
  r = 1/4 →
  a 3 + a 4 = 320 := by
  sorry

end sequence_sum_l3909_390944


namespace triangle_cinema_seats_l3909_390924

/-- Represents a triangular seating arrangement in a cinema --/
structure TriangularCinema where
  best_seat_number : ℕ
  total_seats : ℕ

/-- Checks if a given TriangularCinema configuration is valid --/
def is_valid_cinema (c : TriangularCinema) : Prop :=
  ∃ n : ℕ,
    -- The number of rows is 2n + 1
    (2 * n + 1) * ((2 * n + 1) + 1) / 2 = c.total_seats ∧
    -- The best seat is in the middle row
    (n + 1) * (n + 2) / 2 = c.best_seat_number

/-- Theorem stating the relationship between the best seat number and total seats --/
theorem triangle_cinema_seats (c : TriangularCinema) :
  c.best_seat_number = 265 → is_valid_cinema c → c.total_seats = 1035 := by
  sorry

#check triangle_cinema_seats

end triangle_cinema_seats_l3909_390924


namespace jack_bought_36_books_l3909_390935

/-- The number of books Jack bought each month -/
def books_per_month : ℕ := sorry

/-- The price of each book in dollars -/
def price_per_book : ℕ := 20

/-- The total sale price at the end of the year in dollars -/
def total_sale_price : ℕ := 500

/-- The total loss in dollars -/
def total_loss : ℕ := 220

/-- Theorem stating that Jack bought 36 books each month -/
theorem jack_bought_36_books : books_per_month = 36 := by
  sorry

end jack_bought_36_books_l3909_390935


namespace sugar_trader_profit_l3909_390903

/-- Represents the profit calculation for a sugar trader. -/
theorem sugar_trader_profit (Q : ℝ) (C : ℝ) : Q > 0 → C > 0 → 
  (Q - 1200) * (1.08 * C) + 1200 * (1.12 * C) = Q * C * 1.11 → Q = 1600 := by
  sorry

#check sugar_trader_profit

end sugar_trader_profit_l3909_390903


namespace manuscript_fee_calculation_l3909_390995

def tax_rate_1 : ℚ := 14 / 100
def tax_rate_2 : ℚ := 11 / 100
def tax_threshold_1 : ℕ := 800
def tax_threshold_2 : ℕ := 4000
def tax_paid : ℕ := 420

theorem manuscript_fee_calculation (fee : ℕ) : 
  (tax_threshold_1 < fee ∧ fee ≤ tax_threshold_2 ∧ 
   (fee - tax_threshold_1) * tax_rate_1 = tax_paid) → 
  fee = 3800 :=
by sorry

end manuscript_fee_calculation_l3909_390995


namespace theater_camp_talents_l3909_390920

theorem theater_camp_talents (total_students : ℕ) 
  (cannot_sing cannot_dance both_talents : ℕ) : 
  total_students = 120 →
  cannot_sing = 30 →
  cannot_dance = 50 →
  both_talents = 10 →
  (total_students - cannot_sing) + (total_students - cannot_dance) - both_talents = 130 :=
by sorry

end theater_camp_talents_l3909_390920


namespace valid_money_distribution_exists_l3909_390918

/-- Represents the distribution of money among 5 people --/
structure MoneyDistribution where
  total : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Checks if a MoneyDistribution satisfies the given conditions --/
def satisfiesConditions (dist : MoneyDistribution) : Prop :=
  dist.total = 5000 ∧
  dist.a / dist.b = 3 / 2 ∧
  dist.b / dist.c = 4 / 5 ∧
  dist.d = 0.6 * dist.c ∧
  dist.e = 0.6 * dist.c ∧
  dist.a + dist.b + dist.c + dist.d + dist.e = dist.total

/-- Theorem stating the existence of a valid money distribution --/
theorem valid_money_distribution_exists : ∃ (dist : MoneyDistribution), satisfiesConditions dist := by
  sorry

#check valid_money_distribution_exists

end valid_money_distribution_exists_l3909_390918


namespace intersection_M_N_l3909_390970

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2}
def N : Set ℝ := {x | ∃ y, (x^2/2) + y^2 = 1}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = Set.Icc 0 (Real.sqrt 2) := by sorry

end intersection_M_N_l3909_390970


namespace fifth_bank_coins_l3909_390947

def coins_in_bank (n : ℕ) : ℕ := 72 + 9 * (n - 1)

theorem fifth_bank_coins :
  coins_in_bank 5 = 108 :=
by sorry

end fifth_bank_coins_l3909_390947


namespace average_score_is_two_average_score_independent_of_class_size_l3909_390977

/-- Represents the score distribution for a test -/
structure ScoreDistribution where
  threePoints : Real
  twoPoints : Real
  onePoint : Real
  zeroPoints : Real

/-- Calculates the average score given a score distribution -/
def averageScore (dist : ScoreDistribution) : Real :=
  3 * dist.threePoints + 2 * dist.twoPoints + 1 * dist.onePoint + 0 * dist.zeroPoints

/-- Theorem: The average score is 2 for the given score distribution -/
theorem average_score_is_two (dist : ScoreDistribution) 
    (h1 : dist.threePoints = 0.3)
    (h2 : dist.twoPoints = 0.5)
    (h3 : dist.onePoint = 0.1)
    (h4 : dist.zeroPoints = 0.1)
    (h5 : dist.threePoints + dist.twoPoints + dist.onePoint + dist.zeroPoints = 1) :
    averageScore dist = 2 := by
  sorry

/-- Corollary: The average score is independent of the number of students -/
theorem average_score_independent_of_class_size (n : Nat) (dist : ScoreDistribution) 
    (h1 : dist.threePoints = 0.3)
    (h2 : dist.twoPoints = 0.5)
    (h3 : dist.onePoint = 0.1)
    (h4 : dist.zeroPoints = 0.1)
    (h5 : dist.threePoints + dist.twoPoints + dist.onePoint + dist.zeroPoints = 1) :
    averageScore dist = 2 := by
  sorry

end average_score_is_two_average_score_independent_of_class_size_l3909_390977


namespace quadratic_root_value_l3909_390906

theorem quadratic_root_value (b : ℝ) : 
  (∀ x : ℝ, x^2 + Real.sqrt (b - 1) * x + b^2 - 4 = 0 → x = 0) →
  (b - 1 ≥ 0) →
  b = 2 := by
sorry

end quadratic_root_value_l3909_390906


namespace dynaco_shares_sold_is_150_l3909_390955

/-- Represents the stock portfolio problem --/
structure StockProblem where
  microtron_price : ℕ
  dynaco_price : ℕ
  total_shares : ℕ
  average_price : ℕ

/-- Calculates the number of Dynaco shares sold --/
def dynaco_shares_sold (p : StockProblem) : ℕ :=
  (p.total_shares * p.average_price - p.microtron_price * p.total_shares) / (p.dynaco_price - p.microtron_price)

/-- Theorem stating that given the problem conditions, 150 Dynaco shares were sold --/
theorem dynaco_shares_sold_is_150 (p : StockProblem) 
  (h1 : p.microtron_price = 36)
  (h2 : p.dynaco_price = 44)
  (h3 : p.total_shares = 300)
  (h4 : p.average_price = 40) :
  dynaco_shares_sold p = 150 := by
  sorry

#eval dynaco_shares_sold { microtron_price := 36, dynaco_price := 44, total_shares := 300, average_price := 40 }

end dynaco_shares_sold_is_150_l3909_390955


namespace trig_expression_equality_l3909_390912

theorem trig_expression_equality : 
  (Real.cos (27 * π / 180) - Real.sqrt 2 * Real.sin (18 * π / 180)) / Real.cos (63 * π / 180) = 1 := by
  sorry

end trig_expression_equality_l3909_390912


namespace distance_PQ_is_25_l3909_390991

/-- The distance between point P and the intersection point Q of lines l₁ and l₂ is 25. -/
theorem distance_PQ_is_25 
  (P : ℝ × ℝ)
  (l₁ : Set (ℝ × ℝ))
  (l₂ : Set (ℝ × ℝ))
  (Q : ℝ × ℝ)
  (h₁ : P = (3, 2))
  (h₂ : ∀ (x y : ℝ), (x, y) ∈ l₁ ↔ ∃ t, x = 3 + 4/5 * t ∧ y = 2 + 3/5 * t)
  (h₃ : ∀ (x y : ℝ), (x, y) ∈ l₂ ↔ x - 2*y + 11 = 0)
  (h₄ : Q ∈ l₁ ∧ Q ∈ l₂) :
  dist P Q = 25 := by
  sorry

#check distance_PQ_is_25

end distance_PQ_is_25_l3909_390991


namespace inverse_functions_theorem_l3909_390972

-- Define the set of graph labels
inductive GraphLabel
| A | B | C | D | E

-- Define a property for a function to have an inverse based on the Horizontal Line Test
def has_inverse (g : GraphLabel) : Prop :=
  match g with
  | GraphLabel.B => True
  | GraphLabel.C => True
  | _ => False

-- Define a function that checks if a graph passes the Horizontal Line Test
def passes_horizontal_line_test (g : GraphLabel) : Prop :=
  has_inverse g

-- Theorem statement
theorem inverse_functions_theorem :
  (∀ g : GraphLabel, has_inverse g ↔ passes_horizontal_line_test g) ∧
  (has_inverse GraphLabel.B ∧ has_inverse GraphLabel.C) ∧
  (¬ has_inverse GraphLabel.A ∧ ¬ has_inverse GraphLabel.D ∧ ¬ has_inverse GraphLabel.E) :=
sorry

end inverse_functions_theorem_l3909_390972


namespace middleAgedInPerformance_l3909_390988

/-- Represents the number of employees in each age group -/
structure EmployeeGroups where
  elderly : ℕ
  middleAged : ℕ
  young : ℕ

/-- Calculates the number of middle-aged employees selected in a stratified sample -/
def middleAgedSelected (total : ℕ) (groups : EmployeeGroups) (sampleSize : ℕ) : ℕ :=
  (sampleSize * groups.middleAged) / (groups.elderly + groups.middleAged + groups.young)

/-- Theorem: The number of middle-aged employees selected in the performance is 15 -/
theorem middleAgedInPerformance (total : ℕ) (groups : EmployeeGroups) (sampleSize : ℕ) 
    (h1 : total = 1200)
    (h2 : groups.elderly = 100)
    (h3 : groups.middleAged = 500)
    (h4 : groups.young = 600)
    (h5 : sampleSize = 36) :
  middleAgedSelected total groups sampleSize = 15 := by
  sorry

#eval middleAgedSelected 1200 ⟨100, 500, 600⟩ 36

end middleAgedInPerformance_l3909_390988


namespace range_of_c_l3909_390921

def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y

def q (c : ℝ) : Prop := ∀ x : ℝ, 2*c*x^2 + 2*x + 1 > 0

theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : (p c ∨ q c) ∧ ¬(p c ∧ q c)) :
  c ∈ Set.Ioo 0 (1/2) ∪ Set.Ici 1 :=
sorry

end range_of_c_l3909_390921


namespace point_on_circle_l3909_390952

/-- The coordinates of a point on the unit circle after moving counterclockwise from (1, 0) by an arc length of 2π/3 -/
theorem point_on_circle (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  P = (1, 0) → 
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = (2 * Real.pi / 3)^2 →
  Q.1^2 + Q.2^2 = 1 →
  Q = (-1/2, Real.sqrt 3 / 2) :=
by sorry

end point_on_circle_l3909_390952


namespace equation_has_two_solutions_l3909_390984

-- Define the equation
def equation (x : ℝ) : Prop := Real.sqrt (9 - x) = x * Real.sqrt (9 - x)

-- Theorem statement
theorem equation_has_two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ equation a ∧ equation b ∧ ∀ (x : ℝ), equation x → (x = a ∨ x = b) := by
  sorry

end equation_has_two_solutions_l3909_390984


namespace sad_children_count_l3909_390938

theorem sad_children_count (total : ℕ) (happy : ℕ) (neither : ℕ) 
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ)
  (h1 : total = 60)
  (h2 : happy = 30)
  (h3 : neither = 20)
  (h4 : boys = 19)
  (h5 : girls = 41)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neither_boys = 7)
  (h9 : total = happy + neither + (total - happy - neither)) :
  total - happy - neither = 10 := by
sorry

end sad_children_count_l3909_390938


namespace x29x_divisible_by_18_l3909_390945

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def four_digit_number (x : ℕ) : ℕ := 1000 * x + 290 + x

theorem x29x_divisible_by_18 :
  ∃! x : ℕ, is_single_digit x ∧ (four_digit_number x) % 18 = 0 ∧ x = 2 :=
sorry

end x29x_divisible_by_18_l3909_390945


namespace probability_of_9_heads_in_12_flips_l3909_390925

def num_flips : ℕ := 12
def num_heads : ℕ := 9

theorem probability_of_9_heads_in_12_flips :
  (num_flips.choose num_heads : ℚ) / 2^num_flips = 55 / 1024 := by
  sorry

end probability_of_9_heads_in_12_flips_l3909_390925


namespace cab_driver_income_l3909_390980

theorem cab_driver_income (day2 day3 day4 day5 average : ℝ) 
  (h1 : day2 = 400)
  (h2 : day3 = 750)
  (h3 : day4 = 400)
  (h4 : day5 = 500)
  (h5 : average = 460)
  (h6 : average = (day1 + day2 + day3 + day4 + day5) / 5) :
  day1 = 250 := by
  sorry

#check cab_driver_income

end cab_driver_income_l3909_390980


namespace max_k_value_l3909_390901

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the line
def line (k x : ℝ) (y : ℝ) : Prop := y = k*x - 2

-- Define the condition for a point on the line to be the center of a circle with radius 1 that intersects C
def intersects_C (k x y : ℝ) : Prop :=
  line k x y ∧ ∃ (x' y' : ℝ), circle_C x' y' ∧ (x - x')^2 + (y - y')^2 = 1

-- Theorem statement
theorem max_k_value :
  ∃ (k_max : ℝ), k_max = 4/3 ∧
  (∀ k : ℝ, (∃ x y : ℝ, intersects_C k x y) → k ≤ k_max) ∧
  (∃ x y : ℝ, intersects_C k_max x y) :=
sorry

end max_k_value_l3909_390901


namespace raine_steps_l3909_390914

/-- The number of steps Raine takes to walk to school -/
def steps_to_school : ℕ := 150

/-- The number of days Raine walks to and from school -/
def days : ℕ := 5

/-- The total number of steps Raine takes in five days -/
def total_steps : ℕ := 2 * steps_to_school * days

theorem raine_steps : total_steps = 1500 := by
  sorry

end raine_steps_l3909_390914


namespace inequality_solution_l3909_390950

theorem inequality_solution (x : ℝ) : 
  (x * (x + 2)) / ((x - 5)^2) ≥ 15 ↔ 
  (x ≥ 5/2 ∧ x < 5) ∨ (x > 5 ∧ x ≤ 75/7) :=
by sorry

end inequality_solution_l3909_390950


namespace bart_notepad_spending_l3909_390926

/-- The amount of money Bart spent on notepads -/
def money_spent (cost_per_notepad : ℚ) (pages_per_notepad : ℕ) (total_pages : ℕ) : ℚ :=
  (total_pages / pages_per_notepad) * cost_per_notepad

/-- Theorem: Given the conditions, Bart spent $10 on notepads -/
theorem bart_notepad_spending :
  let cost_per_notepad : ℚ := 5/4  -- $1.25 represented as a rational number
  let pages_per_notepad : ℕ := 60
  let total_pages : ℕ := 480
  money_spent cost_per_notepad pages_per_notepad total_pages = 10 := by
  sorry


end bart_notepad_spending_l3909_390926


namespace min_value_expression_l3909_390922

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 5) / Real.sqrt (x - 4) ≥ 6 ∧ ∃ y : ℝ, y > 4 ∧ (y + 5) / Real.sqrt (y - 4) = 6 :=
sorry

end min_value_expression_l3909_390922


namespace max_daily_sales_l3909_390934

def salesVolume (t : ℕ) : ℝ := -2 * t + 200

def price (t : ℕ) : ℝ :=
  if t ≤ 30 then 12 * t + 30 else 45

def dailySales (t : ℕ) : ℝ :=
  salesVolume t * price t

theorem max_daily_sales :
  ∃ (t : ℕ), 1 ≤ t ∧ t ≤ 50 ∧ ∀ (s : ℕ), 1 ≤ s ∧ s ≤ 50 → dailySales s ≤ dailySales t ∧ dailySales t = 54600 := by
  sorry

end max_daily_sales_l3909_390934


namespace max_value_of_sqrt_sum_l3909_390939

theorem max_value_of_sqrt_sum (x : ℝ) (h : -9 ≤ x ∧ x ≤ 9) : 
  ∃ (max : ℝ), max = 6 ∧ 
  (∀ y : ℝ, -9 ≤ y ∧ y ≤ 9 → Real.sqrt (9 + y) + Real.sqrt (9 - y) ≤ max) ∧
  (∃ z : ℝ, -9 ≤ z ∧ z ≤ 9 ∧ Real.sqrt (9 + z) + Real.sqrt (9 - z) = max) :=
by sorry

end max_value_of_sqrt_sum_l3909_390939


namespace min_value_reciprocal_sum_l3909_390994

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 1) :
  (1/x + 1/y) ≥ 5 + 2*Real.sqrt 6 :=
by sorry

end min_value_reciprocal_sum_l3909_390994


namespace bird_speed_theorem_l3909_390966

theorem bird_speed_theorem (d t : ℝ) (h1 : d = 50 * (t + 1/12)) (h2 : d = 70 * (t - 1/12)) :
  let r := d / t
  ∃ ε > 0, abs (r - 58) < ε :=
sorry

end bird_speed_theorem_l3909_390966


namespace quadratic_form_ratio_l3909_390961

theorem quadratic_form_ratio (j : ℝ) : 
  ∃ (c p q : ℝ), 8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q ∧ q / p = -151 / 3 := by
  sorry

end quadratic_form_ratio_l3909_390961


namespace roots_transformation_l3909_390946

theorem roots_transformation (a b c d : ℝ) : 
  (a^4 - 16*a - 2 = 0) ∧ 
  (b^4 - 16*b - 2 = 0) ∧ 
  (c^4 - 16*c - 2 = 0) ∧ 
  (d^4 - 16*d - 2 = 0) →
  ((a+b)/c^2)^4 - 16*((a+b)/c^2)^3 - 1/2 = 0 ∧
  ((a+c)/b^2)^4 - 16*((a+c)/b^2)^3 - 1/2 = 0 ∧
  ((b+c)/a^2)^4 - 16*((b+c)/a^2)^3 - 1/2 = 0 ∧
  ((b+d)/d^2)^4 - 16*((b+d)/d^2)^3 - 1/2 = 0 :=
by sorry

end roots_transformation_l3909_390946


namespace tan_value_from_trig_equation_l3909_390959

theorem tan_value_from_trig_equation (α : Real) 
  (h : (Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 1/5) : 
  Real.tan α = 8/3 := by
  sorry

end tan_value_from_trig_equation_l3909_390959


namespace cube_root_equation_solution_difference_l3909_390911

theorem cube_root_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (x₁ ≠ x₂) ∧ 
  ((9 - x₁^2 / 4)^(1/3 : ℝ) = -3) ∧ 
  ((9 - x₂^2 / 4)^(1/3 : ℝ) = -3) ∧ 
  (abs (x₁ - x₂) = 24) := by sorry

end cube_root_equation_solution_difference_l3909_390911


namespace probability_four_old_balls_value_l3909_390990

def total_balls : ℕ := 12
def new_balls : ℕ := 9
def old_balls : ℕ := 3
def drawn_balls : ℕ := 3

def probability_four_old_balls : ℚ :=
  (Nat.choose old_balls 2 * Nat.choose new_balls 1) / Nat.choose total_balls drawn_balls

theorem probability_four_old_balls_value :
  probability_four_old_balls = 27 / 220 := by
  sorry

end probability_four_old_balls_value_l3909_390990


namespace union_A_B_equals_open_interval_l3909_390968

-- Define set A
def A : Set ℝ := {x | Real.log (x - 1) < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x - 1}

-- Theorem statement
theorem union_A_B_equals_open_interval :
  A ∪ B = Set.Ioo 1 3 := by sorry

end union_A_B_equals_open_interval_l3909_390968


namespace tea_sale_prices_l3909_390960

structure Tea where
  name : String
  quantity : ℕ
  costPrice : ℚ
  profitPercentage : ℚ

def calculateSalePrice (tea : Tea) : ℚ :=
  tea.costPrice + tea.costPrice * (tea.profitPercentage / 100)

def teaA : Tea := ⟨"A", 120, 25, 45⟩
def teaB : Tea := ⟨"B", 60, 30, 35⟩
def teaC : Tea := ⟨"C", 40, 50, 25⟩
def teaD : Tea := ⟨"D", 30, 70, 20⟩

theorem tea_sale_prices :
  calculateSalePrice teaA = 36.25 ∧
  calculateSalePrice teaB = 40.5 ∧
  calculateSalePrice teaC = 62.5 ∧
  calculateSalePrice teaD = 84 := by
  sorry

end tea_sale_prices_l3909_390960


namespace rectangle_dimensions_l3909_390989

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Condition that length is greater than width -/
def Rectangle.lengthGreaterThanWidth (r : Rectangle) : Prop :=
  r.length > r.width

/-- Perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- Area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- Theorem stating the dimensions of the rectangle -/
theorem rectangle_dimensions (r : Rectangle) 
  (h1 : r.lengthGreaterThanWidth)
  (h2 : r.perimeter = 18)
  (h3 : r.area = 18) :
  r.length = 6 ∧ r.width = 3 := by
  sorry


end rectangle_dimensions_l3909_390989


namespace ben_win_probability_l3909_390974

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 5/8) (h2 : ¬ ∃ tie_prob : ℚ, tie_prob ≠ 0) :
  1 - lose_prob = 3/8 :=
by sorry

end ben_win_probability_l3909_390974


namespace investment_interest_theorem_l3909_390933

/-- Calculates the total interest paid in an 18-month investment contract with specific interest rates and reinvestment -/
def total_interest (initial_investment : ℝ) : ℝ :=
  let interest_6m := initial_investment * 0.02
  let balance_10m := initial_investment + interest_6m
  let interest_10m := balance_10m * 0.03
  let balance_18m := balance_10m + interest_10m
  let interest_18m := balance_18m * 0.04
  interest_6m + interest_10m + interest_18m

/-- Theorem stating that the total interest paid in the given investment scenario is $926.24 -/
theorem investment_interest_theorem :
  total_interest 10000 = 926.24 := by
  sorry

end investment_interest_theorem_l3909_390933


namespace fraction_ordering_l3909_390900

theorem fraction_ordering : 
  (21 : ℚ) / 17 < (18 : ℚ) / 13 ∧ (18 : ℚ) / 13 < (14 : ℚ) / 9 := by
  sorry

end fraction_ordering_l3909_390900


namespace unique_solution_square_equation_l3909_390978

theorem unique_solution_square_equation :
  ∃! x : ℝ, (2010 + x)^2 = x^2 := by sorry

end unique_solution_square_equation_l3909_390978


namespace student_grade_problem_l3909_390910

/-- Given a student's grades in three subjects, prove that if the second subject is 70%,
    the third subject is 90%, and the overall average is 70%, then the first subject must be 50%. -/
theorem student_grade_problem (grade1 grade2 grade3 : ℝ) : 
  grade2 = 70 → grade3 = 90 → (grade1 + grade2 + grade3) / 3 = 70 → grade1 = 50 := by
  sorry

end student_grade_problem_l3909_390910


namespace athlete_team_division_l3909_390996

theorem athlete_team_division (n : ℕ) (k : ℕ) (total : ℕ) (specific : ℕ) :
  n = 10 →
  k = 5 →
  total = n →
  specific = 2 →
  (Nat.choose (n - specific) (k - 1)) = 70 := by
  sorry

end athlete_team_division_l3909_390996


namespace largest_equal_cost_number_l3909_390956

/-- Calculates the sum of digits in decimal representation -/
def sumDecimalDigits (n : Nat) : Nat :=
  sorry

/-- Calculates the sum of digits in binary representation -/
def sumBinaryDigits (n : Nat) : Nat :=
  sorry

/-- Checks if the costs are equal for both options -/
def equalCost (n : Nat) : Prop :=
  sumDecimalDigits n = sumBinaryDigits n

theorem largest_equal_cost_number :
  (∀ m : Nat, m < 500 → m > 404 → ¬(equalCost m)) ∧
  equalCost 404 :=
sorry

end largest_equal_cost_number_l3909_390956


namespace cs_candidates_count_l3909_390969

theorem cs_candidates_count (m : ℕ) (n : ℕ) : 
  m = 4 → 
  m * (n.choose 2) = 84 → 
  n = 7 :=
by sorry

end cs_candidates_count_l3909_390969


namespace unclaimed_fraction_is_correct_l3909_390929

/-- Represents a participant in the chocolate distribution --/
inductive Participant
  | Dave
  | Emma
  | Frank
  | George

/-- The ratio of chocolate distribution for each participant --/
def distribution_ratio (p : Participant) : Rat :=
  match p with
  | Participant.Dave => 4/10
  | Participant.Emma => 3/10
  | Participant.Frank => 2/10
  | Participant.George => 1/10

/-- The order in which participants claim their share --/
def claim_order : List Participant :=
  [Participant.Dave, Participant.Emma, Participant.Frank, Participant.George]

/-- Calculate the fraction of chocolates claimed by a participant --/
def claimed_fraction (p : Participant) (remaining : Rat) : Rat :=
  (distribution_ratio p) * remaining

/-- Calculate the fraction of chocolates that remains unclaimed --/
def unclaimed_fraction : Rat :=
  let initial_remaining : Rat := 1
  let final_remaining := claim_order.foldl
    (fun remaining p => remaining - claimed_fraction p remaining)
    initial_remaining
  final_remaining

/-- Theorem: The fraction of chocolates that remains unclaimed is 37.8/125 --/
theorem unclaimed_fraction_is_correct :
  unclaimed_fraction = 378/1250 := by
  sorry


end unclaimed_fraction_is_correct_l3909_390929


namespace estimate_red_balls_l3909_390923

/-- Represents the number of balls in the bag -/
def total_balls : ℕ := 10

/-- Represents the total number of draws -/
def total_draws : ℕ := 1000

/-- Represents the number of times a red ball was drawn -/
def red_draws : ℕ := 200

/-- The estimated number of red balls in the bag -/
def estimated_red_balls : ℚ := (red_draws : ℚ) / total_draws * total_balls

theorem estimate_red_balls :
  estimated_red_balls = 2 := by sorry

end estimate_red_balls_l3909_390923


namespace quadratic_inequality_l3909_390986

-- Define the function f
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 - b*x + c

-- State the theorem
theorem quadratic_inequality (b c : ℝ) :
  (∀ x, f b c (1 + x) = f b c (1 - x)) →
  f b c 0 = 3 →
  ∀ x, f b c (b^x) ≤ f b c (c^x) := by
  sorry

end quadratic_inequality_l3909_390986


namespace committee_arrangement_count_l3909_390907

def num_women : ℕ := 7
def num_men : ℕ := 3
def num_rocking_chairs : ℕ := 7
def num_stools : ℕ := 3
def num_unique_chair : ℕ := 1
def total_seats : ℕ := num_women + num_men + num_unique_chair

def arrangement_count : ℕ := total_seats * (Nat.choose (total_seats - 1) num_stools)

theorem committee_arrangement_count :
  arrangement_count = 1320 :=
sorry

end committee_arrangement_count_l3909_390907


namespace digit_move_equals_multiply_divide_l3909_390940

def N : ℕ := 2173913043478260869565

theorem digit_move_equals_multiply_divide :
  (N * 4) / 5 = (N % 10^22) * 10 + (N / 10^22) :=
by sorry

end digit_move_equals_multiply_divide_l3909_390940


namespace managers_salary_solve_manager_salary_problem_l3909_390916

/-- Calculates the manager's salary given the number of employees, their average salary,
    and the increase in average salary when the manager's salary is added. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) : ℚ :=
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + avg_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_salary

/-- Proves that the manager's salary is 3300 given the problem conditions. -/
theorem solve_manager_salary_problem :
  managers_salary 20 1200 100 = 3300 := by
  sorry

end managers_salary_solve_manager_salary_problem_l3909_390916


namespace comet_orbit_equation_l3909_390983

/-- Represents the equation of an ellipse -/
structure EllipseEquation where
  a : ℝ
  b : ℝ
  (positive_a : 0 < a)
  (positive_b : 0 < b)

/-- Represents the orbital parameters of a comet -/
structure CometOrbit where
  perihelion : ℝ
  aphelion : ℝ
  (positive_perihelion : 0 < perihelion)
  (positive_aphelion : 0 < aphelion)
  (perihelion_less_than_aphelion : perihelion < aphelion)

/-- 
Given a comet's orbit with perihelion 2 AU and aphelion 6 AU from the Sun,
prove that its orbit equation is x²/16 + y²/12 = 1
-/
theorem comet_orbit_equation (orbit : CometOrbit) 
  (h_perihelion : orbit.perihelion = 2)
  (h_aphelion : orbit.aphelion = 6) : 
  ∃ (eq : EllipseEquation), eq.a = 4 ∧ eq.b = 2 * Real.sqrt 3 := by
  sorry

end comet_orbit_equation_l3909_390983


namespace no_preimage_range_l3909_390985

/-- The function f: ℝ → ℝ defined by f(x) = -x² + 2x -/
def f (x : ℝ) : ℝ := -x^2 + 2*x

/-- The theorem stating that k > 1 is the range of values for which f(x) = k has no solution -/
theorem no_preimage_range (k : ℝ) : 
  (∀ x, f x ≠ k) ↔ k > 1 := by
  sorry

#check no_preimage_range

end no_preimage_range_l3909_390985


namespace f_properties_l3909_390965

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 4 + 2 * Real.sin x * Real.cos x - Real.sin x ^ 4

theorem f_properties :
  (∀ x : ℝ, f (-x) ≠ f x ∧ f (-x) ≠ -f x) ∧
  (∀ ε > 0, ∃ x : ℝ, x > 0 ∧ x < π + ε ∧ f (x + π) = f x) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (- 3 * Real.pi / 8 + k * Real.pi) (k * Real.pi + Real.pi / 8))) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = Real.sqrt 2 ∧ ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f x) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -1 ∧ ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≥ f x) :=
by sorry

end f_properties_l3909_390965


namespace triangle_existence_l3909_390928

theorem triangle_existence (x : ℤ) : 
  (5 + x > 0) ∧ (2*x + 1 > 0) ∧ (3*x > 0) ∧
  (5 + x + 2*x + 1 > 3*x) ∧ (5 + x + 3*x > 2*x + 1) ∧ (2*x + 1 + 3*x > 5 + x) ↔ 
  x ≥ 2 := by
sorry

end triangle_existence_l3909_390928


namespace fence_painting_earnings_l3909_390913

/-- Calculate the total earnings from painting fences -/
theorem fence_painting_earnings
  (rate : ℝ)
  (num_fences : ℕ)
  (fence_length : ℝ)
  (h1 : rate = 0.20)
  (h2 : num_fences = 50)
  (h3 : fence_length = 500) :
  rate * (↑num_fences * fence_length) = 5000 := by
  sorry

end fence_painting_earnings_l3909_390913


namespace milk_bottle_recycling_l3909_390993

theorem milk_bottle_recycling (marcus_bottles john_bottles : ℕ) 
  (h1 : marcus_bottles = 25) 
  (h2 : john_bottles = 20) : 
  marcus_bottles + john_bottles = 45 := by
  sorry

end milk_bottle_recycling_l3909_390993


namespace sum_of_powers_l3909_390975

theorem sum_of_powers (w : ℂ) (hw : w^2 - w + 1 = 0) :
  w^97 + w^98 + w^99 + w^100 + w^101 + w^102 = -2 + 2*w := by
  sorry

end sum_of_powers_l3909_390975


namespace number_of_digits_l3909_390987

theorem number_of_digits (N : ℕ) : N = 2^12 * 5^8 → (Nat.digits 10 N).length = 10 := by sorry

end number_of_digits_l3909_390987


namespace max_value_sum_sqrt_l3909_390973

theorem max_value_sum_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧ 
  (∀ a' b' c' : ℝ, 0 < a' → 0 < b' → 0 < c' → a' + b' + c' = 1 →
    Real.sqrt (3 * a' + 1) + Real.sqrt (3 * b' + 1) + Real.sqrt (3 * c' + 1) ≤ max) ∧
  Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) = max :=
sorry

end max_value_sum_sqrt_l3909_390973


namespace initial_number_proof_l3909_390941

theorem initial_number_proof (x : ℕ) : x - 109 = 109 + 68 → x = 286 := by
  sorry

end initial_number_proof_l3909_390941


namespace intersection_of_M_and_N_l3909_390902

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {y | ∃ x, y = Real.cos x}

theorem intersection_of_M_and_N : M ∩ N = M := by sorry

end intersection_of_M_and_N_l3909_390902


namespace not_divisible_by_5_and_7_count_count_less_than_1000_l3909_390951

theorem not_divisible_by_5_and_7_count : Nat → Nat
  | n => (n + 1) - (n / 5 + n / 7 - n / 35)

theorem count_less_than_1000 :
  not_divisible_by_5_and_7_count 999 = 686 := by
  sorry

end not_divisible_by_5_and_7_count_count_less_than_1000_l3909_390951


namespace total_cost_equation_l3909_390981

/-- Represents the total cost of tickets for a school trip to Green World -/
def totalCost (x : ℕ) : ℕ :=
  40 * x + 60

/-- Theorem stating the relationship between the number of students and the total cost -/
theorem total_cost_equation (x : ℕ) (y : ℕ) :
  y = totalCost x ↔ y = 40 * x + 60 := by sorry

end total_cost_equation_l3909_390981


namespace perpendicular_bisector_of_chord_l3909_390962

/-- The line that intersects the unit circle -/
def intersecting_line (x y : ℝ) : Prop := x - y + 1 = 0

/-- The unit circle -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The perpendicular bisector of the chord -/
def perpendicular_bisector (x y : ℝ) : Prop := x + y = 0

/-- Theorem: The perpendicular bisector of the chord formed by the intersection
    of the line x - y + 1 = 0 and the unit circle x^2 + y^2 = 1 
    has the equation x + y = 0 -/
theorem perpendicular_bisector_of_chord :
  ∀ (x y : ℝ), 
  intersecting_line x y → unit_circle x y →
  perpendicular_bisector x y :=
sorry

end perpendicular_bisector_of_chord_l3909_390962


namespace count_positive_integer_solutions_l3909_390943

/-- The number of positive integer solutions for the equation x + y + z + t = 15 -/
theorem count_positive_integer_solutions : 
  (Finset.filter (fun (x : ℕ × ℕ × ℕ × ℕ) => x.1 + x.2.1 + x.2.2.1 + x.2.2.2 = 15) 
    (Finset.product (Finset.range 15) (Finset.product (Finset.range 15) 
      (Finset.product (Finset.range 15) (Finset.range 15))))).card = 364 := by
  sorry

#check count_positive_integer_solutions

end count_positive_integer_solutions_l3909_390943


namespace min_value_xy_l3909_390957

theorem min_value_xy (x y : ℝ) (hx : x > 1) (hy : y > 1) (h_log : Real.log x * Real.log y = Real.log 3) :
  ∀ z, x * y ≥ z → z ≤ 9 :=
sorry

end min_value_xy_l3909_390957


namespace max_sum_of_squares_l3909_390967

theorem max_sum_of_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 ∧ 
  n ∈ Finset.range 1982 ∧ 
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end max_sum_of_squares_l3909_390967


namespace reflection_theorem_l3909_390948

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflects a point across the line y = x - 2 -/
def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p_translated := (p.1, p.2 - 2)
  let p_reflected := (p_translated.2, p_translated.1)
  (p_reflected.1, p_reflected.2 + 2)

/-- The triangle ABC -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {(3, 4), (6, 8), (5, 1)}

theorem reflection_theorem :
  let A : ℝ × ℝ := (3, 4)
  let A' := reflect_x A
  let A'' := reflect_line A'
  A'' = (-6, 5) :=
by
  sorry

end reflection_theorem_l3909_390948


namespace function_equation_solution_l3909_390979

/-- Given functions f and g satisfying the condition for all x and y, 
    prove that f and g have the specified forms. -/
theorem function_equation_solution 
  (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f y + g x - g y = Real.sin x + Real.cos y) :
  (∃ c : ℝ, (∀ x : ℝ, f x = (Real.sin x + Real.cos x) / 2 ∧ 
                       g x = (Real.sin x - Real.cos x) / 2 + c)) :=
by sorry

end function_equation_solution_l3909_390979


namespace system_has_three_solutions_l3909_390931

/-- The system of equations has exactly 3 distinct real solutions -/
theorem system_has_three_solutions :
  ∃! (S : Set (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (a b c d : ℝ), (a, b, c, d) ∈ S ↔ 
      (a = (b + c + d)^3 ∧
       b = (a + c + d)^3 ∧
       c = (a + b + d)^3 ∧
       d = (a + b + c)^3)) ∧
    S.ncard = 3 := by
  sorry

end system_has_three_solutions_l3909_390931


namespace particular_number_calculation_l3909_390992

theorem particular_number_calculation (x : ℝ) (h : 2.5 * x - 2.49 = 22.01) :
  (x / 2.5) + 2.49 + 22.01 = 28.42 := by
sorry

end particular_number_calculation_l3909_390992


namespace square_sequence_formulas_l3909_390963

/-- The number of squares in the nth figure of the sequence -/
def num_squares (n : ℕ) : ℕ := 2 * n^2 - 2 * n + 1

/-- The first formula: (2n-1)^2 - 4 * (n(n-1)/2) -/
def formula_a (n : ℕ) : ℕ := (2 * n - 1)^2 - 2 * n * (n - 1)

/-- The third formula: 1 + (1 + 2 + ... + (n-1)) * 4 -/
def formula_c (n : ℕ) : ℕ := 1 + 2 * n * (n - 1)

/-- The fourth formula: (n-1)^2 + n^2 -/
def formula_d (n : ℕ) : ℕ := (n - 1)^2 + n^2

theorem square_sequence_formulas (n : ℕ) : 
  n > 0 → num_squares n = formula_a n ∧ num_squares n = formula_c n ∧ num_squares n = formula_d n :=
by sorry

end square_sequence_formulas_l3909_390963


namespace average_sum_abs_diff_l3909_390909

/-- A permutation of integers from 1 to 12 -/
def Permutation := Fin 12 → Fin 12

/-- The sum of absolute differences for a given permutation -/
def sumAbsDiff (p : Permutation) : ℚ :=
  |p 0 - p 1| + |p 2 - p 3| + |p 4 - p 5| + |p 6 - p 7| + |p 8 - p 9| + |p 10 - p 11|

/-- The set of all permutations of integers from 1 to 12 -/
def allPermutations : Finset Permutation := sorry

/-- The average value of sumAbsDiff over all permutations -/
def averageValue : ℚ := (allPermutations.sum sumAbsDiff) / allPermutations.card

theorem average_sum_abs_diff : averageValue = 143 / 33 := by sorry

end average_sum_abs_diff_l3909_390909


namespace function_inequality_l3909_390936

open Real

theorem function_inequality (f : ℝ → ℝ) (h : ∀ x > 0, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x) :
  4 < f 2 / f 1 ∧ f 2 / f 1 < 8 := by
  sorry

end function_inequality_l3909_390936


namespace jesse_stamp_collection_l3909_390953

theorem jesse_stamp_collection (total : ℕ) (european : ℕ) (asian : ℕ) 
  (h1 : total = 444)
  (h2 : european = 3 * asian)
  (h3 : total = european + asian) :
  european = 333 := by
sorry

end jesse_stamp_collection_l3909_390953


namespace hyperbola_eccentricity_l3909_390942

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point on the hyperbola -/
def Point := ℝ × ℝ

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The angle between three points -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) (P : Point) 
  (h1 : distance P h.F₂ = 2 * distance P h.F₁)
  (h2 : angle P h.F₁ h.F₂ = Real.pi / 3) : 
  eccentricity h = (1 + Real.sqrt 13) / 2 := sorry

end hyperbola_eccentricity_l3909_390942


namespace last_digit_sum_powers_l3909_390999

theorem last_digit_sum_powers : (1993^2002 + 1995^2002) % 10 = 4 := by
  sorry

end last_digit_sum_powers_l3909_390999


namespace expression_simplification_l3909_390998

theorem expression_simplification (a b : ℝ) (ha : a > 0) :
  a^(1/3) * b^(1/2) * (-3 * a^(1/2) * b^(1/3)) / ((1/3) * a^(1/6) * b^(5/6)) = -9 * a^(2/3) :=
by sorry

end expression_simplification_l3909_390998


namespace ceiling_plus_one_l3909_390917

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

-- State the theorem
theorem ceiling_plus_one (x : ℝ) : ceiling (x + 1) = ceiling x + 1 := by
  sorry

end ceiling_plus_one_l3909_390917


namespace prime_divisibility_l3909_390976

theorem prime_divisibility (p q : ℕ) : 
  Prime p → Prime q → (p * q) ∣ (2^p + 2^q) → p = 2 ∧ q = 3 := by
  sorry

end prime_divisibility_l3909_390976


namespace range_of_a_l3909_390958

-- Define the inequality system
def inequality_system (a : ℝ) (x : ℝ) : Prop :=
  x > a ∧ x > 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | x > 1}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x, inequality_system a x ↔ x ∈ solution_set a) →
  a ≤ 1 :=
sorry

end range_of_a_l3909_390958


namespace magic_square_sum_l3909_390997

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e : ℕ)
  (sum : ℕ)
  (row1_sum : a + 22 + b = sum)
  (row2_sum : 20 + c + e = sum)
  (row3_sum : 28 + d + 19 = sum)
  (col1_sum : a + 20 + 28 = sum)
  (col2_sum : 22 + c + d = sum)
  (col3_sum : b + e + 19 = sum)
  (diag1_sum : a + c + 19 = sum)
  (diag2_sum : 28 + c + b = sum)

/-- Theorem: In the given magic square, d + e = 70 -/
theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 70 := by
  sorry

end magic_square_sum_l3909_390997


namespace keith_placed_scissors_l3909_390982

/-- The number of scissors Keith placed in the drawer -/
def scissors_placed (initial final : ℕ) : ℕ := final - initial

/-- Proof that Keith placed 22 scissors in the drawer -/
theorem keith_placed_scissors : scissors_placed 54 76 = 22 := by
  sorry

end keith_placed_scissors_l3909_390982


namespace ratio_problem_l3909_390954

theorem ratio_problem (second_part : ℝ) (percent : ℝ) (first_part : ℝ) : 
  second_part = 5 →
  percent = 120 →
  first_part / second_part = percent / 100 →
  first_part = 6 :=
by
  sorry

end ratio_problem_l3909_390954


namespace equal_sides_implies_rhombus_rhombus_equal_diagonals_implies_square_equal_angles_implies_rectangle_l3909_390927

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define properties of quadrilaterals
def Quadrilateral.is_rhombus (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.is_square (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.is_rectangle (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.has_equal_sides (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.has_equal_diagonals (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.has_equal_angles (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statements
theorem equal_sides_implies_rhombus (q : Quadrilateral) :
  q.has_equal_sides → q.is_rhombus :=
sorry

theorem rhombus_equal_diagonals_implies_square (q : Quadrilateral) :
  q.is_rhombus → q.has_equal_diagonals → q.is_square :=
sorry

theorem equal_angles_implies_rectangle (q : Quadrilateral) :
  q.has_equal_angles → q.is_rectangle :=
sorry

end equal_sides_implies_rhombus_rhombus_equal_diagonals_implies_square_equal_angles_implies_rectangle_l3909_390927


namespace prob_no_consecutive_heads_is_9_64_l3909_390964

/-- The number of ways to arrange k heads in n + 1 positions without consecutive heads -/
def arrange_heads (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of sequences without two consecutive heads in 10 coin tosses -/
def total_favorable_sequences : ℕ :=
  arrange_heads 10 0 + arrange_heads 9 1 + arrange_heads 8 2 +
  arrange_heads 7 3 + arrange_heads 6 4 + arrange_heads 5 5

/-- The total number of possible outcomes when tossing a coin 10 times -/
def total_outcomes : ℕ := 2^10

/-- The probability of no two consecutive heads in 10 coin tosses -/
def prob_no_consecutive_heads : ℚ := total_favorable_sequences / total_outcomes

theorem prob_no_consecutive_heads_is_9_64 :
  prob_no_consecutive_heads = 9/64 := by sorry

end prob_no_consecutive_heads_is_9_64_l3909_390964


namespace function_range_theorem_l3909_390919

open Real

theorem function_range_theorem (a : ℝ) (h₁ : a > 0) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 2, 
    a * x₁ + 2 = x₀^2 - 2*x₀) → 0 < a ∧ a ≤ 1/2 := by
  sorry

end function_range_theorem_l3909_390919


namespace triangle_circumcircle_diameter_l3909_390904

theorem triangle_circumcircle_diameter 
  (a : Real) 
  (B : Real) 
  (S : Real) : 
  a = 1 → 
  B = π / 4 → 
  S = 2 → 
  ∃ (b c d : Real), 
    c = 4 * Real.sqrt 2 ∧ 
    b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ 
    b = 5 ∧ 
    d = b / (Real.sin B) ∧ 
    d = 5 * Real.sqrt 2 := by
  sorry

end triangle_circumcircle_diameter_l3909_390904


namespace twentieth_term_of_specific_sequence_l3909_390949

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem twentieth_term_of_specific_sequence :
  arithmetic_sequence 2 4 20 = 78 := by
  sorry

end twentieth_term_of_specific_sequence_l3909_390949


namespace simplify_product_of_square_roots_l3909_390971

theorem simplify_product_of_square_roots (x : ℝ) (h : x > 0) :
  Real.sqrt (50 * x^3) * Real.sqrt (18 * x^2) * Real.sqrt (35 * x) = 30 * x^3 * Real.sqrt 35 := by
  sorry

end simplify_product_of_square_roots_l3909_390971


namespace parabola_point_order_l3909_390937

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x - 1)^2 - 2

theorem parabola_point_order (a b c d : ℝ) :
  parabola a 2 →
  parabola b 6 →
  parabola c d →
  d < 1 →
  a < 0 →
  b > 0 →
  a < c ∧ c < b := by
  sorry

end parabola_point_order_l3909_390937


namespace binary_101101110_equals_octal_556_l3909_390915

/-- Converts a binary number (represented as a list of bits) to a decimal number -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- Converts a decimal number to an octal number (represented as a list of digits) -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_101101110_equals_octal_556 :
  decimal_to_octal (binary_to_decimal [0, 1, 1, 1, 0, 1, 1, 0, 1]) = [5, 5, 6] := by
  sorry

end binary_101101110_equals_octal_556_l3909_390915
