import Mathlib

namespace tangent_line_p_values_l1784_178423

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 8 = 0

/-- The equation of the tangent line -/
def tangent_line (p x : ℝ) : Prop := x = -p/2

/-- The line is tangent to the circle -/
def is_tangent (p : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y ∧ tangent_line p x

/-- Theorem: If the line x = -p/2 is tangent to the circle x^2 + y^2 + 6x + 8 = 0, then p = 4 or p = 8 -/
theorem tangent_line_p_values (p : ℝ) : is_tangent p → p = 4 ∨ p = 8 := by
  sorry

end tangent_line_p_values_l1784_178423


namespace triangle_count_theorem_l1784_178407

/-- The number of trees planted in a triangular shape -/
def num_trees : ℕ := 21

/-- The number of ways to choose 3 trees from num_trees -/
def total_choices : ℕ := Nat.choose num_trees 3

/-- The number of ways to choose 3 collinear trees -/
def collinear_choices : ℕ := 114

/-- The number of ways to choose 3 trees to form a non-degenerate triangle -/
def non_degenerate_triangles : ℕ := total_choices - collinear_choices

theorem triangle_count_theorem : non_degenerate_triangles = 1216 := by
  sorry

end triangle_count_theorem_l1784_178407


namespace original_sales_tax_percentage_l1784_178427

/-- Proves that the original sales tax percentage was 0.5%, given the conditions of the problem -/
theorem original_sales_tax_percentage
  (new_tax_rate : ℚ)
  (market_price : ℚ)
  (tax_difference : ℚ)
  (h1 : new_tax_rate = 10 / 3)
  (h2 : market_price = 9000)
  (h3 : tax_difference = 15) :
  ∃ (original_tax_rate : ℚ),
    original_tax_rate = 1 / 2 ∧
    original_tax_rate * market_price - new_tax_rate * market_price = tax_difference :=
by
  sorry

end original_sales_tax_percentage_l1784_178427


namespace expression_evaluation_l1784_178440

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := 1
  (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 4 := by sorry

end expression_evaluation_l1784_178440


namespace sin_cos_inequality_l1784_178454

theorem sin_cos_inequality (t : ℝ) (h : t > 0) : 3 * Real.sin t < 2 * t + t * Real.cos t := by
  sorry

end sin_cos_inequality_l1784_178454


namespace inequality_system_solution_range_l1784_178460

theorem inequality_system_solution_range (m : ℝ) : 
  (∃ (s : Finset ℤ), s.card = 4 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x - m < 0 ∧ 7 - 2*x ≤ 1))) →
  (6 < m ∧ m ≤ 7) :=
sorry

end inequality_system_solution_range_l1784_178460


namespace film_review_analysis_l1784_178447

structure FilmReviewData where
  total_sample : ℕ
  male_count : ℕ
  female_count : ℕ
  male_negative : ℕ
  female_positive : ℕ
  significance_level : ℝ
  chi_square_critical : ℝ
  stratified_sample_size : ℕ
  coupon_recipients : ℕ

def chi_square_statistic (data : FilmReviewData) : ℝ := sorry

def is_associated (data : FilmReviewData) : Prop :=
  chi_square_statistic data > data.chi_square_critical

def probability_distribution (x : ℕ) : ℝ := sorry

def expected_value : ℝ := sorry

theorem film_review_analysis (data : FilmReviewData) 
  (h1 : data.total_sample = 220)
  (h2 : data.male_count = 110)
  (h3 : data.female_count = 110)
  (h4 : data.male_negative = 70)
  (h5 : data.female_positive = 60)
  (h6 : data.significance_level = 0.010)
  (h7 : data.chi_square_critical = 6.635)
  (h8 : data.stratified_sample_size = 10)
  (h9 : data.coupon_recipients = 3) :
  is_associated data ∧ 
  probability_distribution 0 = 1/30 ∧
  probability_distribution 1 = 3/10 ∧
  probability_distribution 2 = 1/2 ∧
  probability_distribution 3 = 1/6 ∧
  expected_value = 9/5 := by sorry

end film_review_analysis_l1784_178447


namespace fraction_to_decimal_l1784_178459

theorem fraction_to_decimal : (7 : ℚ) / 16 = (4375 : ℚ) / 10000 := by sorry

end fraction_to_decimal_l1784_178459


namespace crow_tree_problem_l1784_178496

theorem crow_tree_problem (x y : ℕ) : 
  (3 * y + 5 = x) → (5 * (y - 1) = x) → (x = 20 ∧ y = 5) := by
  sorry

end crow_tree_problem_l1784_178496


namespace shape_to_square_possible_l1784_178490

/-- Represents a shape on a graph paper -/
structure Shape where
  -- Add necessary fields to represent the shape
  -- For example, you might use a list of coordinates

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields to represent a triangle
  -- For example, you might use three points

/-- Represents a square -/
structure Square where
  -- Add necessary fields to represent a square
  -- For example, you might use side length and position

/-- Function to check if a list of triangles can form a square -/
def can_form_square (triangles : List Triangle) : Prop :=
  -- Define the logic to check if triangles can form a square
  sorry

/-- The main theorem stating that the shape can be divided into 5 triangles
    that can be rearranged to form a square -/
theorem shape_to_square_possible (s : Shape) : 
  ∃ (t1 t2 t3 t4 t5 : Triangle), 
    (can_form_square [t1, t2, t3, t4, t5]) ∧ 
    (-- Add condition that t1, t2, t3, t4, t5 are a valid division of s
     sorry) := by
  sorry

end shape_to_square_possible_l1784_178490


namespace prob_red_then_black_54_card_deck_l1784_178404

/-- A deck of cards with red and black cards, including jokers -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- The probability of drawing a red card first and a black card second -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) * d.black_cards / (d.total_cards * (d.total_cards - 1))

/-- The theorem stating the probability for the specific deck -/
theorem prob_red_then_black_54_card_deck :
  prob_red_then_black ⟨54, 26, 28⟩ = 364 / 1431 := by sorry

end prob_red_then_black_54_card_deck_l1784_178404


namespace sum_of_powers_eq_7290_l1784_178469

/-- The power of a triple of positive integers -/
def power (x y z : ℕ) : ℕ := max x (max y z) + min x (min y z)

/-- The sum of powers of all triples (x,y,z) where x,y,z ≤ 9 -/
def sum_of_powers : ℕ :=
  (Finset.range 9).sum (fun x =>
    (Finset.range 9).sum (fun y =>
      (Finset.range 9).sum (fun z =>
        power (x + 1) (y + 1) (z + 1))))

theorem sum_of_powers_eq_7290 : sum_of_powers = 7290 := by
  sorry

end sum_of_powers_eq_7290_l1784_178469


namespace min_cost_2009_l1784_178406

/-- Represents the available coin denominations in rubles -/
inductive Coin : Type
  | one : Coin
  | two : Coin
  | five : Coin
  | ten : Coin

/-- The value of a coin in rubles -/
def coinValue : Coin → Nat
  | Coin.one => 1
  | Coin.two => 2
  | Coin.five => 5
  | Coin.ten => 10

/-- An arithmetic expression using coins and operations -/
inductive Expr : Type
  | coin : Coin → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluates an expression to its numerical value -/
def eval : Expr → Int
  | Expr.coin c => coinValue c
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Calculates the cost of an expression in rubles -/
def cost : Expr → Nat
  | Expr.coin c => coinValue c
  | Expr.add e1 e2 => cost e1 + cost e2
  | Expr.sub e1 e2 => cost e1 + cost e2
  | Expr.mul e1 e2 => cost e1 + cost e2
  | Expr.div e1 e2 => cost e1 + cost e2

/-- Theorem: The minimum cost to represent 2009 is 23 rubles -/
theorem min_cost_2009 : 
  (∃ e : Expr, eval e = 2009 ∧ cost e = 23) ∧
  (∀ e : Expr, eval e = 2009 → cost e ≥ 23) := by sorry

end min_cost_2009_l1784_178406


namespace inequality_solution_set_abs_b_greater_than_two_l1784_178458

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- Theorem for part I
theorem inequality_solution_set (x : ℝ) :
  f x + f (x + 1) ≥ 5 ↔ x ≥ 4 ∨ x ≤ -1 :=
sorry

-- Theorem for part II
theorem abs_b_greater_than_two (a b : ℝ) :
  |a| > 1 → f (a * b) > |a| * f (b / a) → |b| > 2 :=
sorry

end inequality_solution_set_abs_b_greater_than_two_l1784_178458


namespace min_reciprocal_sum_l1784_178410

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 2) :
  (1/x + 1/y + 1/z) ≥ 4.5 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 ∧ 1/x + 1/y + 1/z = 4.5 :=
by sorry

end min_reciprocal_sum_l1784_178410


namespace unique_number_l1784_178478

theorem unique_number : ∃! n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧  -- three-digit number
  (n / 100 = 4) ∧  -- starts with 4
  ((n % 100) * 10 + 4 = (3 * n) / 4)  -- moving 4 to end results in 0.75 times original
  := by sorry

end unique_number_l1784_178478


namespace initial_men_count_l1784_178476

/-- Represents the initial number of men working on the project -/
def initial_men : ℕ := sorry

/-- Represents the number of days to complete the work with the initial group -/
def initial_days : ℕ := 40

/-- Represents the number of men who leave the project -/
def men_who_leave : ℕ := 20

/-- Represents the number of days worked before some men leave -/
def days_before_leaving : ℕ := 10

/-- Represents the number of days to complete the remaining work after some men leave -/
def remaining_days : ℕ := 40

/-- Work rate of one man per day -/
def work_rate : ℚ := 1 / (initial_men * initial_days)

/-- Fraction of work completed before some men leave -/
def work_completed_before_leaving : ℚ := work_rate * initial_men * days_before_leaving

/-- Fraction of work remaining after some men leave -/
def remaining_work : ℚ := 1 - work_completed_before_leaving

/-- The theorem states that given the conditions, the initial number of men is 80 -/
theorem initial_men_count : initial_men = 80 := by sorry

end initial_men_count_l1784_178476


namespace inequality_system_solution_l1784_178424

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x, (x ≥ b - 1 ∧ x < a / 2) ↔ (-3 ≤ x ∧ x < 3 / 2)) → 
  a * b = -6 := by
  sorry

end inequality_system_solution_l1784_178424


namespace quadratic_no_real_roots_l1784_178412

theorem quadratic_no_real_roots
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : c > 0)
  (h3 : |a - b| < c) :
  ∀ x : ℝ, a^2 * x^2 + (b^2 + a^2 - c^2) * x + b^2 ≠ 0 :=
by
  sorry

end quadratic_no_real_roots_l1784_178412


namespace original_number_exists_l1784_178448

theorem original_number_exists : ∃ x : ℝ, 4 * ((x^3 / 5)^2 + 15) = 224 := by
  sorry

end original_number_exists_l1784_178448


namespace birthday_pigeonhole_l1784_178449

theorem birthday_pigeonhole (n m : ℕ) (h1 : n = 39) (h2 : m = 12) :
  ∃ k : ℕ, k ≤ m ∧ 4 ≤ (n / m + (if n % m = 0 then 0 else 1)) := by
  sorry

end birthday_pigeonhole_l1784_178449


namespace extra_lives_in_first_level_l1784_178464

theorem extra_lives_in_first_level :
  let initial_lives : ℕ := 2
  let lives_gained_second_level : ℕ := 11
  let total_lives_after_second_level : ℕ := 19
  let extra_lives_first_level : ℕ := total_lives_after_second_level - lives_gained_second_level - initial_lives
  extra_lives_first_level = 6 :=
by sorry

end extra_lives_in_first_level_l1784_178464


namespace unique_prime_plus_10_14_prime_l1784_178456

theorem unique_prime_plus_10_14_prime :
  ∃! p : ℕ, Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) :=
by sorry

end unique_prime_plus_10_14_prime_l1784_178456


namespace classroom_seats_count_l1784_178428

/-- Represents a rectangular classroom with seats arranged in rows and columns. -/
structure Classroom where
  seats_left : ℕ  -- Number of seats to the left of the chosen seat
  seats_right : ℕ  -- Number of seats to the right of the chosen seat
  rows_front : ℕ  -- Number of rows in front of the chosen seat
  rows_back : ℕ  -- Number of rows behind the chosen seat

/-- Calculates the total number of seats in the classroom. -/
def total_seats (c : Classroom) : ℕ :=
  (c.seats_left + c.seats_right + 1) * (c.rows_front + c.rows_back + 1)

/-- Theorem stating that a classroom with the given properties has 399 seats. -/
theorem classroom_seats_count :
  ∀ (c : Classroom),
    c.seats_left = 6 →
    c.seats_right = 12 →
    c.rows_front = 7 →
    c.rows_back = 13 →
    total_seats c = 399 := by
  sorry

end classroom_seats_count_l1784_178428


namespace distance_to_rock_mist_mountains_value_l1784_178481

/-- The distance from the city to Rock Mist Mountains, including detours -/
def distance_to_rock_mist_mountains : ℝ :=
  let sky_falls_distance : ℝ := 8
  let rock_mist_multiplier : ℝ := 50
  let break_point_percentage : ℝ := 0.3
  let cloudy_heights_percentage : ℝ := 0.6
  let thunder_pass_detour : ℝ := 25
  
  sky_falls_distance * rock_mist_multiplier + thunder_pass_detour

/-- Theorem stating the distance to Rock Mist Mountains -/
theorem distance_to_rock_mist_mountains_value :
  distance_to_rock_mist_mountains = 425 := by
  sorry

end distance_to_rock_mist_mountains_value_l1784_178481


namespace daily_lottery_expected_profit_l1784_178485

/-- The expected profit from purchasing one "Daily Lottery" ticket -/
def expected_profit : ℝ := -0.9

/-- The price of one lottery ticket -/
def ticket_price : ℝ := 2

/-- The probability of winning the first prize -/
def first_prize_prob : ℝ := 0.001

/-- The probability of winning the second prize -/
def second_prize_prob : ℝ := 0.1

/-- The amount of the first prize -/
def first_prize_amount : ℝ := 100

/-- The amount of the second prize -/
def second_prize_amount : ℝ := 10

theorem daily_lottery_expected_profit :
  expected_profit = 
    first_prize_prob * first_prize_amount + 
    second_prize_prob * second_prize_amount - 
    ticket_price := by
  sorry

end daily_lottery_expected_profit_l1784_178485


namespace salary_fraction_on_food_l1784_178495

theorem salary_fraction_on_food 
  (salary : ℝ) 
  (rent_fraction : ℝ) 
  (clothes_fraction : ℝ) 
  (amount_left : ℝ) 
  (h1 : salary = 160000)
  (h2 : rent_fraction = 1/10)
  (h3 : clothes_fraction = 3/5)
  (h4 : amount_left = 16000)
  (h5 : salary * rent_fraction + salary * clothes_fraction + amount_left + salary * food_fraction = salary) :
  food_fraction = 1/5 := by
sorry

end salary_fraction_on_food_l1784_178495


namespace tile_side_length_proof_l1784_178414

/-- Represents a rectangular room with length and width in centimeters -/
structure Room where
  length : ℕ
  width : ℕ

/-- Represents a square tile with side length in centimeters -/
structure Tile where
  side_length : ℕ

/-- Calculates the area of a room in square centimeters -/
def room_area (r : Room) : ℕ := r.length * r.width

/-- Calculates the area of a tile in square centimeters -/
def tile_area (t : Tile) : ℕ := t.side_length * t.side_length

theorem tile_side_length_proof (r : Room) (num_tiles : ℕ) (h1 : r.length = 5000) (h2 : r.width = 1125) (h3 : num_tiles = 9000) :
  ∃ t : Tile, tile_area t * num_tiles = room_area r ∧ t.side_length = 25 := by
  sorry

end tile_side_length_proof_l1784_178414


namespace correct_number_of_selections_l1784_178475

/-- The number of volunteers who only speak Russian -/
def russian_only : ℕ := 3

/-- The number of volunteers who speak both Russian and English -/
def bilingual : ℕ := 4

/-- The total number of volunteers -/
def total_volunteers : ℕ := russian_only + bilingual

/-- The number of English translators to be selected -/
def english_translators : ℕ := 2

/-- The number of Russian translators to be selected -/
def russian_translators : ℕ := 2

/-- The total number of translators to be selected -/
def total_translators : ℕ := english_translators + russian_translators

/-- The function to calculate the number of ways to select translators -/
def num_ways_to_select_translators : ℕ := sorry

/-- Theorem stating that the number of ways to select translators is 60 -/
theorem correct_number_of_selections :
  num_ways_to_select_translators = 60 := by sorry

end correct_number_of_selections_l1784_178475


namespace existence_of_divisibility_l1784_178472

/-- The largest proper divisor of a positive integer -/
def largest_proper_divisor (n : ℕ) : ℕ := sorry

/-- The sequence u_n as defined in the problem -/
def u : ℕ → ℕ
  | 0 => sorry  -- This value is not specified in the problem
  | 1 => sorry  -- We only know u_1 > 0, but not its exact value
  | (n + 2) => u (n + 1) + largest_proper_divisor (u (n + 1))

theorem existence_of_divisibility :
  ∃ N : ℕ, ∀ n : ℕ, n > N → (3^2019 : ℕ) ∣ u n :=
sorry

end existence_of_divisibility_l1784_178472


namespace min_value_problem_l1784_178409

theorem min_value_problem (x y : ℝ) 
  (h1 : x - 1 ≥ 0)
  (h2 : x - y + 1 ≤ 0)
  (h3 : x + y - 4 ≤ 0) :
  ∃ (m : ℝ), m = 1/4 ∧ ∀ (a b : ℝ), 
    a - 1 ≥ 0 → a - b + 1 ≤ 0 → a + b - 4 ≤ 0 → 
    a / (b + 1) ≥ m :=
by sorry

end min_value_problem_l1784_178409


namespace xyz_value_l1784_178494

theorem xyz_value (x y z : ℝ) (h1 : x^2 * y * z^3 = 7^3) (h2 : x * y^2 = 7^9) : 
  x * y * z = 7^4 := by
sorry

end xyz_value_l1784_178494


namespace test_average_l1784_178426

theorem test_average (male_count : ℕ) (female_count : ℕ) (male_avg : ℝ) (female_avg : ℝ)
  (h1 : male_count = 8)
  (h2 : female_count = 24)
  (h3 : male_avg = 84)
  (h4 : female_avg = 92) :
  (male_count * male_avg + female_count * female_avg) / (male_count + female_count) = 90 := by
  sorry

end test_average_l1784_178426


namespace b_completion_time_l1784_178415

/-- The time it takes for person A to complete the work alone -/
def time_A : ℝ := 15

/-- The time A and B work together -/
def time_together : ℝ := 5

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.41666666666666663

/-- The time it takes for person B to complete the work alone -/
def time_B : ℝ := 20

/-- Theorem stating that given the conditions, B takes 20 days to complete the work alone -/
theorem b_completion_time :
  (time_together * (1 / time_A + 1 / time_B) = 1 - work_left) →
  time_B = 20 := by
  sorry

end b_completion_time_l1784_178415


namespace discount_problem_l1784_178451

theorem discount_problem (x y : ℝ) : 
  (100 - x / 100 * 100) * (1 - y / 100) = 55 →
  (100 - 55) / 100 * 100 = 45 := by
sorry

end discount_problem_l1784_178451


namespace ahmed_age_l1784_178442

theorem ahmed_age (fouad_age : ℕ) (ahmed_age : ℕ) (h : fouad_age = 26) :
  (fouad_age + 4 = 2 * ahmed_age) → ahmed_age = 15 := by
  sorry

end ahmed_age_l1784_178442


namespace basketball_wins_needed_l1784_178477

/-- Calculates the number of additional games a basketball team needs to win to achieve a target win percentage -/
theorem basketball_wins_needed
  (games_played : ℕ)
  (games_won : ℕ)
  (games_remaining : ℕ)
  (target_percentage : ℚ)
  (h1 : games_played = 50)
  (h2 : games_won = 35)
  (h3 : games_remaining = 25)
  (h4 : target_percentage = 64 / 100) :
  ⌈(target_percentage * ↑(games_played + games_remaining) - ↑games_won)⌉ = 13 :=
by sorry

end basketball_wins_needed_l1784_178477


namespace quadratic_inequality_solution_sets_l1784_178444

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set of the first inequality
def S₁ : Set ℝ := {x | x < -2 ∨ x > -1/2}

-- State the theorem
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : ∀ x, f a b c x < 0 ↔ x ∈ S₁) :
  ∀ x, f a (-b) c x > 0 ↔ 1/2 < x ∧ x < 2 :=
sorry

end quadratic_inequality_solution_sets_l1784_178444


namespace ceiling_fraction_evaluation_l1784_178497

theorem ceiling_fraction_evaluation : 
  (⌈⌈(23:ℝ)/9 - ⌈(35:ℝ)/21⌉⌉⌉ : ℝ) / ⌈⌈(36:ℝ)/9 + ⌈(9:ℝ)*23/36⌉⌉⌉ = 1/10 := by
  sorry

end ceiling_fraction_evaluation_l1784_178497


namespace work_completion_days_l1784_178425

/-- The number of days B takes to complete the work alone -/
def B : ℝ := 12

/-- The number of days A and B work together -/
def together_days : ℝ := 3

/-- The number of days B works alone after A leaves -/
def B_alone_days : ℝ := 3

/-- The number of days A takes to complete the work alone -/
def A : ℝ := 6

theorem work_completion_days : 
  together_days * (1 / A + 1 / B) + B_alone_days * (1 / B) = 1 :=
sorry

end work_completion_days_l1784_178425


namespace volume_of_specific_parallelepiped_l1784_178450

/-- A rectangular parallelepiped with vertices A B C D A₁ B₁ C₁ D₁ -/
structure RectangularParallelepiped where
  base_length : ℝ
  base_width : ℝ
  height : ℝ

/-- A plane passing through vertices A, C, and D₁ of the parallelepiped -/
structure DiagonalPlane where
  parallelepiped : RectangularParallelepiped
  dihedral_angle : ℝ

/-- The volume of a rectangular parallelepiped -/
def volume (p : RectangularParallelepiped) : ℝ :=
  p.base_length * p.base_width * p.height

/-- Theorem: Volume of the specific parallelepiped -/
theorem volume_of_specific_parallelepiped (p : RectangularParallelepiped) 
  (d : DiagonalPlane) (h1 : p.base_length = 4) (h2 : p.base_width = 3) 
  (h3 : d.parallelepiped = p) (h4 : d.dihedral_angle = π / 3) :
  volume p = (144 * Real.sqrt 3) / 5 := by
  sorry


end volume_of_specific_parallelepiped_l1784_178450


namespace sphere_in_cone_angle_l1784_178467

/-- Given a sphere inscribed in a cone, if the circle of tangency divides the surface
    of the sphere in the ratio of 1:4, then the angle between the generatrix of the cone
    and its base plane is arccos(3/5). -/
theorem sphere_in_cone_angle (R : ℝ) (α : ℝ) :
  R > 0 →  -- Radius is positive
  (2 * π * R^2 * (1 - Real.cos α)) / (4 * π * R^2) = 1/5 →  -- Surface area ratio condition
  α = Real.arccos (3/5) :=
by sorry

end sphere_in_cone_angle_l1784_178467


namespace jake_present_weight_l1784_178417

/-- Jake's present weight in pounds -/
def jake_weight : ℕ := 194

/-- Kendra's weight in pounds -/
def kendra_weight : ℕ := 287 - jake_weight

/-- The amount of weight Jake needs to lose to weigh twice as much as Kendra -/
def weight_loss : ℕ := jake_weight - 2 * kendra_weight

theorem jake_present_weight : jake_weight = 194 := by
  sorry

end jake_present_weight_l1784_178417


namespace direction_vector_c_value_l1784_178432

-- Define the two points on the line
def point1 : ℝ × ℝ := (-7, 3)
def point2 : ℝ × ℝ := (-3, -1)

-- Define the direction vector
def direction_vector (c : ℝ) : ℝ × ℝ := (4, c)

-- Theorem statement
theorem direction_vector_c_value :
  ∃ (c : ℝ), direction_vector c = (point2.1 - point1.1, point2.2 - point1.2) :=
by sorry

end direction_vector_c_value_l1784_178432


namespace horner_first_step_value_v₁_equals_30_l1784_178466

/-- Horner's Rule first step for polynomial evaluation -/
def horner_first_step (a₄ a₃ : ℝ) (x : ℝ) : ℝ :=
  a₄ * x + a₃

/-- The polynomial f(x) = 3x⁴ + 2x² + x + 4 -/
def f (x : ℝ) : ℝ :=
  3 * x^4 + 2 * x^2 + x + 4

theorem horner_first_step_value :
  horner_first_step 3 0 10 = 30 :=
by sorry

theorem v₁_equals_30 :
  horner_first_step 3 0 10 = 30 :=
by sorry

end horner_first_step_value_v₁_equals_30_l1784_178466


namespace exists_equivalent_expr_l1784_178461

/-- Represents the two possible binary operations in our system -/
inductive Op
| add
| sub

/-- Represents an expression in our system -/
inductive Expr
| var : String → Expr
| op : Op → Expr → Expr → Expr

/-- Evaluates an expression given an assignment of values to variables and a mapping of symbols to operations -/
def evaluate (e : Expr) (vars : String → ℝ) (sym_to_op : Op → Op) : ℝ :=
  match e with
  | Expr.var v => vars v
  | Expr.op o e1 e2 =>
    let v1 := evaluate e1 vars sym_to_op
    let v2 := evaluate e2 vars sym_to_op
    match sym_to_op o with
    | Op.add => v1 + v2
    | Op.sub => v1 - v2

/-- The theorem to be proved -/
theorem exists_equivalent_expr :
  ∃ (e : Expr),
    ∀ (vars : String → ℝ) (sym_to_op : Op → Op),
      evaluate e vars sym_to_op = 20 * vars "a" - 18 * vars "b" :=
sorry

end exists_equivalent_expr_l1784_178461


namespace alcohol_mixture_problem_l1784_178438

theorem alcohol_mixture_problem (x : ℝ) 
  (h1 : x > 0)  -- Ensure x is positive
  (h2 : 0.9 * x = 0.54 * (x + 16)) : x = 24 := by
  sorry

#check alcohol_mixture_problem

end alcohol_mixture_problem_l1784_178438


namespace charlie_running_steps_l1784_178418

/-- Given that Charlie makes 5350 steps on a 3-kilometer running field,
    prove that running 2 1/2 times around the field results in 13375 steps. -/
theorem charlie_running_steps (steps_per_field : ℕ) (field_length : ℝ) (laps : ℝ) :
  steps_per_field = 5350 →
  field_length = 3 →
  laps = 2.5 →
  (steps_per_field : ℝ) * laps = 13375 := by
  sorry

end charlie_running_steps_l1784_178418


namespace circle_properties_l1784_178487

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 4

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop := x + 2 * y = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define the theorem
theorem circle_properties :
  -- Part 1: Equation of circle O
  (∀ x y : ℝ, circle_O x y ↔ x^2 + y^2 = 4) ∧
  -- Part 2: Equation of line MN
  (∀ m n : ℝ × ℝ,
    circle_O m.1 m.2 ∧ circle_O n.1 n.2 ∧
    symmetry_line ((m.1 + n.1) / 2) ((m.2 + n.2) / 2) ∧
    (m.1 - n.1)^2 + (m.2 - n.2)^2 = 12 →
    ∃ b : ℝ, (2 * m.1 - m.2 + b = 0 ∧ 2 * n.1 - n.2 + b = 0) ∧ b^2 = 5) ∧
  -- Part 3: Range of PA · PB
  (∀ p : ℝ × ℝ,
    circle_O p.1 p.2 ∧
    ((p.1 + 2)^2 + p.2^2) * ((p.1 - 2)^2 + p.2^2) = (p.1^2 + p.2^2)^2 →
    -2 ≤ ((p.1 + 2) * (p.1 - 2) + p.2^2) ∧ ((p.1 + 2) * (p.1 - 2) + p.2^2) < 0) :=
by sorry

end circle_properties_l1784_178487


namespace anika_age_l1784_178419

theorem anika_age :
  ∀ (anika_age maddie_age : ℕ),
  anika_age = (4 * maddie_age) / 3 →
  (anika_age + 15 + maddie_age + 15) / 2 = 50 →
  anika_age = 40 :=
by
  sorry

end anika_age_l1784_178419


namespace arithmetic_progression_polynomial_p_l1784_178452

/-- A polynomial of the form x^4 + px^2 + qx - 144 with four distinct real roots in arithmetic progression -/
structure ArithmeticProgressionPolynomial where
  p : ℝ
  q : ℝ
  roots : Fin 4 → ℝ
  distinct_roots : ∀ i j, i ≠ j → roots i ≠ roots j
  arithmetic_progression : ∃ (a d : ℝ), ∀ i, roots i = a + i * d
  is_root : ∀ i, (roots i)^4 + p * (roots i)^2 + q * (roots i) - 144 = 0

/-- The value of p in an ArithmeticProgressionPolynomial is -40 -/
theorem arithmetic_progression_polynomial_p (poly : ArithmeticProgressionPolynomial) : poly.p = -40 := by
  sorry

end arithmetic_progression_polynomial_p_l1784_178452


namespace staircase_perimeter_l1784_178468

/-- A staircase-shaped region with specific properties -/
structure StaircaseRegion where
  congruentSides : ℕ
  sideLength : ℝ
  area : ℝ

/-- Calculate the perimeter of the staircase region -/
def perimeter (s : StaircaseRegion) : ℝ :=
  7 + 11 + 3 + 7 + s.congruentSides * s.sideLength

/-- Theorem: The perimeter of the specific staircase region is 39 feet -/
theorem staircase_perimeter :
  ∀ s : StaircaseRegion,
    s.congruentSides = 10 ∧
    s.sideLength = 1 ∧
    s.area = 74 →
    perimeter s = 39 := by
  sorry

end staircase_perimeter_l1784_178468


namespace probability_real_roots_l1784_178479

-- Define the interval [0,5]
def interval : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 5}

-- Define the condition for real roots
def has_real_roots (p : ℝ) : Prop := p^2 ≥ 4

-- Define the measure of the interval where the equation has real roots
def measure_real_roots : ℝ := 3

-- Define the total measure of the interval
def total_measure : ℝ := 5

-- State the theorem
theorem probability_real_roots : 
  (measure_real_roots / total_measure : ℝ) = 0.6 := by sorry

end probability_real_roots_l1784_178479


namespace rectangular_field_ratio_l1784_178434

theorem rectangular_field_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a < b) : 
  (a + b = 3 * a) → 
  (a + b - Real.sqrt (a^2 + b^2) = b / 3) → 
  a / b = 1 / 2 := by
sorry

end rectangular_field_ratio_l1784_178434


namespace ellipse_max_min_sum_l1784_178455

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the function we want to maximize/minimize
def f (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem ellipse_max_min_sum :
  (∃ x y : ℝ, ellipse x y ∧ f x y = Real.sqrt 5) ∧
  (∃ x y : ℝ, ellipse x y ∧ f x y = -Real.sqrt 5) ∧
  (∀ x y : ℝ, ellipse x y → f x y ≤ Real.sqrt 5) ∧
  (∀ x y : ℝ, ellipse x y → f x y ≥ -Real.sqrt 5) :=
sorry

end ellipse_max_min_sum_l1784_178455


namespace tetrahedron_edges_form_triangles_l1784_178405

/-- Represents a tetrahedron with edge lengths a, b, c, d, e, f -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  e_pos : 0 < e
  f_pos : 0 < f
  vertex_sum_equal : a + b + c = b + d + f ∧ a + b + c = c + d + e ∧ a + b + c = a + e + f

theorem tetrahedron_edges_form_triangles (t : Tetrahedron) :
  (t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b) ∧
  (t.b + t.d > t.f ∧ t.d + t.f > t.b ∧ t.f + t.b > t.d) ∧
  (t.c + t.d > t.e ∧ t.d + t.e > t.c ∧ t.e + t.c > t.d) ∧
  (t.a + t.e > t.f ∧ t.e + t.f > t.a ∧ t.f + t.a > t.e) := by
  sorry


end tetrahedron_edges_form_triangles_l1784_178405


namespace budget_allocation_l1784_178446

theorem budget_allocation (transportation research_development utilities supplies salaries equipment : ℝ) :
  transportation = 20 →
  research_development = 9 →
  utilities = 5 →
  supplies = 2 →
  salaries = 216 / 360 * 100 →
  transportation + research_development + utilities + supplies + salaries + equipment = 100 →
  equipment = 4 :=
by sorry

end budget_allocation_l1784_178446


namespace expression_simplification_and_evaluation_l1784_178492

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ -3 → x ≠ 3 →
  (1 - 1 / (x + 3)) / ((x^2 - 9) / (x^2 + 6*x + 9)) = (x + 2) / (x - 3) ∧
  (2 + 2) / (2 - 3) = -4 := by
  sorry

end expression_simplification_and_evaluation_l1784_178492


namespace test_probabilities_l1784_178491

/-- The probability that exactly two out of three students pass their tests. -/
def prob_two_pass (pA pB pC : ℚ) : ℚ :=
  pA * pB * (1 - pC) + pA * (1 - pB) * pC + (1 - pA) * pB * pC

/-- The probability that at least one out of three students fails their test. -/
def prob_at_least_one_fail (pA pB pC : ℚ) : ℚ :=
  1 - pA * pB * pC

theorem test_probabilities (pA pB pC : ℚ) 
  (hA : pA = 4/5) (hB : pB = 3/5) (hC : pC = 7/10) : 
  prob_two_pass pA pB pC = 113/250 ∧ 
  prob_at_least_one_fail pA pB pC = 83/125 := by
  sorry

#eval prob_two_pass (4/5) (3/5) (7/10)
#eval prob_at_least_one_fail (4/5) (3/5) (7/10)

end test_probabilities_l1784_178491


namespace cylinder_height_in_hemisphere_l1784_178488

theorem cylinder_height_in_hemisphere (r c h : ℝ) : 
  r > 0 → c > 0 → h > 0 →
  r = 8 → c = 3 →
  h^2 + c^2 = r^2 →
  h = Real.sqrt 55 := by
sorry

end cylinder_height_in_hemisphere_l1784_178488


namespace gcd_40304_30203_l1784_178457

theorem gcd_40304_30203 : Nat.gcd 40304 30203 = 1 := by
  sorry

end gcd_40304_30203_l1784_178457


namespace cases_in_2005_l1784_178489

/-- Calculates the number of cases in a given year assuming a linear decrease --/
def caseCount (initialYear initialCases finalYear finalCases targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let yearlyDecrease := totalDecrease / totalYears
  let targetYearsSinceInitial := targetYear - initialYear
  initialCases - (yearlyDecrease * targetYearsSinceInitial)

/-- Theorem stating that the number of cases in 2005 is 134,000 --/
theorem cases_in_2005 :
  caseCount 1980 800000 2010 800 2005 = 134000 := by
  sorry

end cases_in_2005_l1784_178489


namespace max_projection_area_is_one_l1784_178401

/-- A tetrahedron with specific properties -/
structure SpecialTetrahedron where
  /-- Two adjacent faces are isosceles right triangles -/
  isosceles_right_faces : Bool
  /-- The hypotenuse of the isosceles right triangles is 2 -/
  hypotenuse : ℝ
  /-- The dihedral angle between the two adjacent faces is 60 degrees -/
  dihedral_angle : ℝ
  /-- The tetrahedron rotates around the common edge of the two faces -/
  rotates_around_common_edge : Bool

/-- The maximum projection area of the rotating tetrahedron -/
def max_projection_area (t : SpecialTetrahedron) : ℝ := sorry

/-- Theorem stating that the maximum projection area is 1 -/
theorem max_projection_area_is_one (t : SpecialTetrahedron) 
  (h1 : t.isosceles_right_faces = true)
  (h2 : t.hypotenuse = 2)
  (h3 : t.dihedral_angle = Real.pi / 3)  -- 60 degrees in radians
  (h4 : t.rotates_around_common_edge = true) :
  max_projection_area t = 1 := by sorry

end max_projection_area_is_one_l1784_178401


namespace beautiful_equations_proof_l1784_178439

/-- Two linear equations are "beautiful equations" if the sum of their solutions is 1 -/
def beautiful_equations (eq1 eq2 : ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), eq1 x ∧ eq2 y ∧ x + y = 1

/-- The first pair of equations -/
def eq1 (x : ℝ) : Prop := 4 * x - (x + 5) = 1

/-- The second pair of equations -/
def eq2 (y : ℝ) : Prop := -2 * y - y = 3

/-- The third pair of equations -/
def eq3 (m : ℝ) (x : ℝ) : Prop := x / 2 + m = 0

/-- The fourth pair of equations -/
def eq4 (x : ℝ) : Prop := 3 * x = x + 4

theorem beautiful_equations_proof :
  (beautiful_equations eq1 eq2) ∧
  (∀ m : ℝ, beautiful_equations (eq3 m) eq4 → m = 1/2) := by sorry

end beautiful_equations_proof_l1784_178439


namespace equal_charges_at_60_minutes_l1784_178462

/-- United Telephone's base rate in dollars -/
def united_base : ℝ := 9

/-- United Telephone's per-minute rate in dollars -/
def united_per_minute : ℝ := 0.25

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℝ := 12

/-- Atlantic Call's per-minute rate in dollars -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which the charges are equal -/
def equal_minutes : ℝ := 60

theorem equal_charges_at_60_minutes :
  united_base + united_per_minute * equal_minutes =
  atlantic_base + atlantic_per_minute * equal_minutes :=
by sorry

end equal_charges_at_60_minutes_l1784_178462


namespace arithmetic_sequence_problem_l1784_178441

-- Define the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ := 3 - 2 * n

-- Define the sum of the first k terms
def sum_of_terms (k : ℕ) : ℤ := k * (arithmetic_sequence 1 + arithmetic_sequence k) / 2

-- Theorem statement
theorem arithmetic_sequence_problem :
  (arithmetic_sequence 1 = 1) ∧
  (arithmetic_sequence 3 = -3) ∧
  (∃ k : ℕ, sum_of_terms k = -35 ∧ k = 7) :=
sorry

end arithmetic_sequence_problem_l1784_178441


namespace car_selection_problem_l1784_178421

theorem car_selection_problem (cars : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) 
  (h1 : cars = 12)
  (h2 : selections_per_client = 4)
  (h3 : selections_per_car = 3) :
  (cars * selections_per_car) / selections_per_client = 9 := by
  sorry

end car_selection_problem_l1784_178421


namespace min_max_abs_expressions_l1784_178480

theorem min_max_abs_expressions (x y : ℝ) :
  ∃ (x₀ y₀ : ℝ), max (|2 * x₀ + y₀|) (max (|x₀ - y₀|) (|1 + y₀|)) = (1/2 : ℝ) ∧
  ∀ (x y : ℝ), (1/2 : ℝ) ≤ max (|2 * x + y|) (max (|x - y|) (|1 + y|)) :=
by sorry

end min_max_abs_expressions_l1784_178480


namespace cameron_typing_speed_l1784_178493

/-- The number of words Cameron could type per minute before breaking his arm -/
def words_before : ℕ := 10

/-- The difference in words typed in 5 minutes before and after breaking his arm -/
def word_difference : ℕ := 10

/-- The number of words Cameron could type per minute after breaking his arm -/
def words_after : ℕ := 8

/-- Proof that Cameron could type 8 words per minute after breaking his arm -/
theorem cameron_typing_speed :
  words_after = 8 ∧
  words_before * 5 - words_after * 5 = word_difference :=
by sorry

end cameron_typing_speed_l1784_178493


namespace dante_recipe_eggs_l1784_178473

theorem dante_recipe_eggs :
  ∀ (eggs flour : ℕ),
  flour = eggs / 2 →
  flour + eggs = 90 →
  eggs = 60 := by
sorry

end dante_recipe_eggs_l1784_178473


namespace calculate_expression_l1784_178486

theorem calculate_expression : -1^4 + 16 / (-2)^3 * |(-3) - 1| = -9 := by
  sorry

end calculate_expression_l1784_178486


namespace circle_transformation_l1784_178445

/-- Coordinate transformation φ -/
def φ (x y : ℝ) : ℝ × ℝ :=
  (4 * x, 2 * y)

/-- Original circle equation -/
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Transformed equation -/
def transformed_equation (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 4) = 1

theorem circle_transformation (x y : ℝ) :
  original_circle x y ↔ transformed_equation (φ x y).1 (φ x y).2 :=
by sorry

end circle_transformation_l1784_178445


namespace k_range_theorem_l1784_178463

theorem k_range_theorem (k : ℝ) : 
  (∀ m : ℝ, 0 < m ∧ m < 3/2 → (2/m) + (1/(3-2*m)) ≥ k^2 + 2*k) → 
  -3 ≤ k ∧ k ≤ 1 := by
sorry

end k_range_theorem_l1784_178463


namespace units_digit_of_M_M12_is_1_l1784_178499

/-- Modified Lucas sequence -/
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | (n + 2) => M (n + 1) + M n

/-- The 12th term of the Modified Lucas sequence -/
def M12 : ℕ := M 12

/-- Theorem stating that the units digit of M_{M₁₂} is 1 -/
theorem units_digit_of_M_M12_is_1 : M M12 % 10 = 1 := by
  sorry

end units_digit_of_M_M12_is_1_l1784_178499


namespace calculator_game_sum_l1784_178483

/-- Represents the state of the calculators -/
structure CalculatorState :=
  (calc1 : ℤ)
  (calc2 : ℤ)
  (calc3 : ℤ)

/-- The operation performed on the calculators in each turn -/
def squareOperation (state : CalculatorState) : CalculatorState :=
  { calc1 := state.calc1 ^ 2,
    calc2 := state.calc2 ^ 2,
    calc3 := state.calc3 ^ 2 }

/-- The initial state of the calculators -/
def initialState : CalculatorState :=
  { calc1 := 2,
    calc2 := -2,
    calc3 := 0 }

/-- The theorem to be proved -/
theorem calculator_game_sum (n : ℕ) (h : n ≥ 1) :
  (squareOperation^[n] initialState).calc1 +
  (squareOperation^[n] initialState).calc2 +
  (squareOperation^[n] initialState).calc3 = 8 :=
sorry

end calculator_game_sum_l1784_178483


namespace sheep_buying_equation_l1784_178422

/-- Represents the price of the sheep -/
def sheep_price (x : ℤ) : ℤ := 5 * x + 45

/-- Represents the total contribution when each person gives 7 coins -/
def contribution_7 (x : ℤ) : ℤ := 7 * x

theorem sheep_buying_equation (x : ℤ) : 
  sheep_price x = contribution_7 x - 3 := by sorry

end sheep_buying_equation_l1784_178422


namespace restaurant_bill_calculation_l1784_178474

def restaurant_bill (num_people_1 num_people_2 : ℕ) (cost_1 cost_2 service_charge : ℚ) 
  (discount_rate tip_rate : ℚ) : ℚ :=
  let meal_cost := num_people_1 * cost_1 + num_people_2 * cost_2
  let total_before_discount := meal_cost + service_charge
  let discount := discount_rate * meal_cost
  let total_after_discount := total_before_discount - discount
  let tip := tip_rate * total_before_discount
  total_after_discount + tip

theorem restaurant_bill_calculation :
  restaurant_bill 10 5 18 25 50 (5/100) (10/100) = 375.25 := by
  sorry

end restaurant_bill_calculation_l1784_178474


namespace solution_set_implies_b_range_l1784_178436

/-- The solution set of the inequality |3x-b| < 4 -/
def SolutionSet (b : ℝ) : Set ℝ :=
  {x : ℝ | |3*x - b| < 4}

/-- The set of integers 1, 2, and 3 -/
def IntegerSet : Set ℝ := {1, 2, 3}

/-- Theorem stating that if the solution set of |3x-b| < 4 is exactly {1, 2, 3}, then 5 < b < 7 -/
theorem solution_set_implies_b_range :
  ∀ b : ℝ, SolutionSet b = IntegerSet → 5 < b ∧ b < 7 := by
  sorry

end solution_set_implies_b_range_l1784_178436


namespace x_value_proof_l1784_178430

theorem x_value_proof (x : ℚ) 
  (h1 : 6 * x^2 + 5 * x - 1 = 0) 
  (h2 : 18 * x^2 + 17 * x - 1 = 0) : 
  x = 1/3 := by
sorry

end x_value_proof_l1784_178430


namespace cycle_price_problem_l1784_178484

theorem cycle_price_problem (selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : selling_price = 1170)
  (h2 : gain_percent = 30) :
  let original_price := selling_price / (1 + gain_percent / 100)
  original_price = 900 := by
sorry

end cycle_price_problem_l1784_178484


namespace cost_of_45_roses_l1784_178498

/-- The cost of a bouquet of roses with discount applied -/
def bouquet_cost (n : ℕ) : ℚ :=
  let base_cost := 30 * (n / 15 : ℚ)
  if n > 30 then base_cost * (1 - 1/10) else base_cost

/-- Theorem stating the cost of a bouquet with 45 roses -/
theorem cost_of_45_roses : bouquet_cost 45 = 81 := by
  sorry

end cost_of_45_roses_l1784_178498


namespace third_circle_radius_l1784_178470

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (c1.radius + c2.radius) ^ 2

/-- Checks if a circle is tangent to the x-axis -/
def is_tangent_to_x_axis (c : Circle) : Prop :=
  c.center.2 = c.radius

/-- The main theorem -/
theorem third_circle_radius 
  (circle_A circle_B circle_C : Circle)
  (h1 : circle_A.radius = 2)
  (h2 : circle_B.radius = 3)
  (h3 : are_externally_tangent circle_A circle_B)
  (h4 : circle_A.center.1 + 6 = circle_B.center.1)
  (h5 : circle_A.center.2 = circle_B.center.2)
  (h6 : are_externally_tangent circle_A circle_C)
  (h7 : are_externally_tangent circle_B circle_C)
  (h8 : is_tangent_to_x_axis circle_C) :
  circle_C.radius = 3 := by
  sorry

end third_circle_radius_l1784_178470


namespace power_function_with_specific_point_is_odd_l1784_178431

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem power_function_with_specific_point_is_odd
  (f : ℝ → ℝ)
  (h_power : isPowerFunction f)
  (h_point : f (Real.sqrt 3 / 3) = Real.sqrt 3) :
  isOddFunction f :=
sorry

end power_function_with_specific_point_is_odd_l1784_178431


namespace min_rooks_to_attack_all_white_cells_l1784_178435

/-- Represents a cell on the chessboard -/
structure Cell :=
  (row : Fin 9)
  (col : Fin 9)

/-- Determines if a cell is white based on its position -/
def isWhite (c : Cell) : Bool :=
  (c.row.val + c.col.val) % 2 = 0

/-- Represents a rook's position on the board -/
structure Rook :=
  (position : Cell)

/-- Determines if a cell is under attack by a rook -/
def isUnderAttack (c : Cell) (r : Rook) : Bool :=
  c.row = r.position.row ∨ c.col = r.position.col

/-- The main theorem stating the minimum number of rooks required -/
theorem min_rooks_to_attack_all_white_cells :
  ∃ (rooks : List Rook),
    rooks.length = 5 ∧
    (∀ c : Cell, isWhite c → ∃ r ∈ rooks, isUnderAttack c r) ∧
    (∀ (rooks' : List Rook),
      rooks'.length < 5 →
      ¬(∀ c : Cell, isWhite c → ∃ r ∈ rooks', isUnderAttack c r)) :=
by
  sorry

end min_rooks_to_attack_all_white_cells_l1784_178435


namespace yearly_subscription_cost_proof_l1784_178408

/-- The yearly subscription cost to professional magazines, given that a 50% reduction
    in the budget results in spending $470 less. -/
def yearly_subscription_cost : ℝ := 940

theorem yearly_subscription_cost_proof :
  yearly_subscription_cost - yearly_subscription_cost / 2 = 470 := by
  sorry

end yearly_subscription_cost_proof_l1784_178408


namespace arithmetic_geometric_mean_square_sum_l1784_178443

theorem arithmetic_geometric_mean_square_sum (a b : ℝ) :
  (a + b) / 2 = 20 → Real.sqrt (a * b) = Real.sqrt 135 → a^2 + b^2 = 1330 := by
  sorry

end arithmetic_geometric_mean_square_sum_l1784_178443


namespace xiaohuo_has_448_books_l1784_178437

/-- The number of books Xiaohuo, Xiaoyan, and Xiaoyi have collectively -/
def total_books : ℕ := 1248

/-- The number of books Xiaohuo has -/
def xiaohuo_books : ℕ := sorry

/-- The number of books Xiaoyan has -/
def xiaoyan_books : ℕ := sorry

/-- The number of books Xiaoyi has -/
def xiaoyi_books : ℕ := sorry

/-- Xiaohuo has 64 more books than Xiaoyan -/
axiom xiaohuo_more_than_xiaoyan : xiaohuo_books = xiaoyan_books + 64

/-- Xiaoyan has 32 fewer books than Xiaoyi -/
axiom xiaoyan_fewer_than_xiaoyi : xiaoyan_books = xiaoyi_books - 32

/-- The total number of books is the sum of books owned by each person -/
axiom total_books_sum : total_books = xiaohuo_books + xiaoyan_books + xiaoyi_books

/-- Theorem: Xiaohuo has 448 books -/
theorem xiaohuo_has_448_books : xiaohuo_books = 448 := by sorry

end xiaohuo_has_448_books_l1784_178437


namespace line_equation_l1784_178471

/-- A line in 2D space defined by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The problem statement -/
theorem line_equation : 
  ∃ (l : Line), 
    l.contains ⟨-1, 2⟩ ∧ 
    l.perpendicular ⟨2, -3, 0⟩ ∧
    l = ⟨3, 2, -1⟩ := by
  sorry

end line_equation_l1784_178471


namespace spinner_prime_sum_probability_l1784_178416

/-- Represents a spinner with numbered sectors -/
structure Spinner :=
  (sectors : List Nat)

/-- Checks if a number is prime -/
def isPrime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

/-- Calculates all possible sums from two spinners -/
def allSums (s1 s2 : Spinner) : List Nat :=
  List.join (s1.sectors.map (fun x => s2.sectors.map (fun y => x + y)))

/-- Counts the number of prime sums -/
def countPrimeSums (sums : List Nat) : Nat :=
  sums.filter isPrime |>.length

theorem spinner_prime_sum_probability :
  let spinner1 : Spinner := ⟨[1, 2, 3]⟩
  let spinner2 : Spinner := ⟨[3, 4, 5]⟩
  let allPossibleSums := allSums spinner1 spinner2
  let totalSums := allPossibleSums.length
  let primeSums := countPrimeSums allPossibleSums
  (primeSums : Rat) / totalSums = 4 / 9 := by
  sorry

end spinner_prime_sum_probability_l1784_178416


namespace range_of_m_l1784_178453

theorem range_of_m (f : ℝ → ℝ) (h : ∀ x ∈ Set.Icc 0 1, f x ≥ m) :
  Set.Iic (-3 : ℝ) = {m : ℝ | ∀ x ∈ Set.Icc 0 1, f x ≥ m} :=
by sorry

#check range_of_m (fun x ↦ x^2 - 4*x)

end range_of_m_l1784_178453


namespace smallest_number_satisfying_conditions_l1784_178420

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 2) % 12 = 0 ∧
  (n - 2) % 16 = 0 ∧
  (n - 2) % 18 = 0 ∧
  (n - 2) % 21 = 0 ∧
  (n - 2) % 28 = 0 ∧
  (n - 2) % 32 = 0 ∧
  (n - 2) % 45 = 0

def is_sum_of_consecutive_primes (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ Nat.Prime (p + 1) ∧ n = p + (p + 1)

theorem smallest_number_satisfying_conditions :
  (is_divisible_by_all 10090 ∧ is_sum_of_consecutive_primes 10090) ∧
  ∀ m : ℕ, m < 10090 → ¬(is_divisible_by_all m ∧ is_sum_of_consecutive_primes m) :=
by sorry

end smallest_number_satisfying_conditions_l1784_178420


namespace friend_savings_rate_l1784_178402

/-- Proves that given the initial amounts and saving rates, the friend's weekly savings
    that result in equal total savings after 25 weeks is 5 dollars. -/
theorem friend_savings_rate (your_initial : ℕ) (your_weekly : ℕ) (friend_initial : ℕ) (weeks : ℕ) :
  your_initial = 160 →
  your_weekly = 7 →
  friend_initial = 210 →
  weeks = 25 →
  ∃ (friend_weekly : ℕ),
    your_initial + your_weekly * weeks = friend_initial + friend_weekly * weeks ∧
    friend_weekly = 5 :=
by sorry

end friend_savings_rate_l1784_178402


namespace candice_arrival_time_l1784_178411

/-- Represents the driving scenario of Candice --/
structure DrivingScenario where
  initial_speed : ℕ
  final_speed : ℕ
  total_distance : ℚ
  drive_time : ℕ

/-- The conditions of Candice's drive --/
def candice_drive : DrivingScenario :=
  { initial_speed := 10,
    final_speed := 6,
    total_distance := 2/3,
    drive_time := 5 }

/-- Theorem stating that Candice arrives home at 5:05 PM --/
theorem candice_arrival_time (d : DrivingScenario) 
  (h1 : d.initial_speed > d.final_speed)
  (h2 : d.drive_time > 0)
  (h3 : d.total_distance = (d.initial_speed + d.final_speed + 1) * (d.initial_speed - d.final_speed) / 120) :
  d.drive_time = 5 ∧ d = candice_drive :=
sorry

end candice_arrival_time_l1784_178411


namespace trajectory_is_square_l1784_178433

-- Define the set of points (x, y) satisfying |x| + |y| = 1
def trajectory : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1| + |p.2| = 1}

-- Define a square in the plane
def isSquare (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), S = {p : ℝ × ℝ | max (|p.1 - a|) (|p.2 - b|) = 1/2}

-- Theorem statement
theorem trajectory_is_square : isSquare trajectory := by sorry

end trajectory_is_square_l1784_178433


namespace triangle_rotation_path_length_l1784_178413

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the path length of vertex C when rotating a triangle along a rectangle -/
def pathLengthC (t : Triangle) (r : Rectangle) : ℝ :=
  sorry

theorem triangle_rotation_path_length :
  let t : Triangle := { a := 2, b := 3, c := 4 }
  let r : Rectangle := { width := 8, height := 6 }
  pathLengthC t r = 12 * Real.pi :=
by sorry

end triangle_rotation_path_length_l1784_178413


namespace inscribed_square_area_l1784_178465

/-- The equation of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/8 = 1

/-- A point is on the square if its coordinates are equal in absolute value -/
def on_square (x y : ℝ) : Prop := |x| = |y|

/-- The square is inscribed in the ellipse -/
def inscribed_square (t : ℝ) : Prop := 
  ellipse t t ∧ on_square t t ∧ t > 0

/-- The area of the inscribed square -/
def square_area (t : ℝ) : ℝ := (2*t)^2

theorem inscribed_square_area : 
  ∃ t : ℝ, inscribed_square t ∧ square_area t = 32/3 := by sorry

end inscribed_square_area_l1784_178465


namespace exam_score_l1784_178403

/-- Calculates the total marks in an examination based on given parameters. -/
def totalMarks (totalQuestions : ℕ) (correctMarks : ℤ) (wrongMarks : ℤ) (correctAnswers : ℕ) : ℤ :=
  (correctAnswers : ℤ) * correctMarks + (totalQuestions - correctAnswers : ℤ) * wrongMarks

/-- Theorem stating that under the given conditions, the student secures 130 marks. -/
theorem exam_score :
  totalMarks 80 4 (-1) 42 = 130 := by
  sorry

end exam_score_l1784_178403


namespace logarithm_properties_l1784_178429

variable (a x y : ℝ)

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem logarithm_properties (h : a > 1) :
  (log a 1 = 0) ∧
  (log a a = 1) ∧
  (∀ x > 0, x < 1 → log a x < 0) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < x ∧ x < δ → log a x < -1/ε) :=
by sorry

end logarithm_properties_l1784_178429


namespace water_amount_in_sport_formulation_l1784_178482

/-- Represents the ratio of ingredients in a flavored drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

/-- The sport formulation of the drink -/
def sport_ratio (r : DrinkRatio) : DrinkRatio :=
  { flavoring := r.flavoring,
    corn_syrup := r.corn_syrup / 3,
    water := r.water * 2 }

theorem water_amount_in_sport_formulation :
  let sport := sport_ratio standard_ratio
  ∀ corn_syrup_oz : ℚ,
    corn_syrup_oz = 1 →
    (sport.water / sport.corn_syrup) * corn_syrup_oz = 15 := by
  sorry

end water_amount_in_sport_formulation_l1784_178482


namespace converse_not_always_true_l1784_178400

-- Define the types for points, lines, and planes in space
variable (Point Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)  -- plane contains line
variable (perp : Line → Plane → Prop)      -- line perpendicular to plane
variable (perp_planes : Plane → Plane → Prop)  -- plane perpendicular to plane

-- State the theorem
theorem converse_not_always_true 
  (b : Line) (α β : Plane) : 
  ¬(∀ b α β, (contains α b ∧ perp b β → perp_planes α β) → 
             (perp_planes α β → contains α b ∧ perp b β)) :=
sorry

end converse_not_always_true_l1784_178400
