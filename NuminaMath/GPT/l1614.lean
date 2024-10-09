import Mathlib

namespace find_const_functions_l1614_161452

theorem find_const_functions
  (f g : ℝ → ℝ)
  (hf : ∀ x y : ℝ, 0 < x → 0 < y → f (x^2 + y^2) = g (x * y)) :
  ∃ c : ℝ, (∀ x, 0 < x → f x = c) ∧ (∀ x, 0 < x → g x = c) :=
sorry

end find_const_functions_l1614_161452


namespace area_after_shortening_other_side_l1614_161405

-- Define initial dimensions of the index card
def initial_length := 5
def initial_width := 7
def initial_area := initial_length * initial_width

-- Define the area condition when one side is shortened by 2 inches
def shortened_side_length := initial_length - 2
def new_area_after_shortening_one_side := 21

-- Definition of the problem condition that results in 21 square inches area
def condition := 
  (shortened_side_length * initial_width = new_area_after_shortening_one_side)

-- Final statement
theorem area_after_shortening_other_side :
  condition →
  (initial_length * (initial_width - 2) = 25) :=
by
  intro h
  sorry

end area_after_shortening_other_side_l1614_161405


namespace arithmetic_mean_of_roots_l1614_161447

-- Definitions corresponding to the conditions
def quadratic_eqn (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- The term statement for the quadratic equation mean
theorem arithmetic_mean_of_roots : 
  ∀ (a b c : ℝ), a = 1 → b = 4 → c = 1 → (∃ (x1 x2 : ℝ), quadratic_eqn a b c x1 ∧ quadratic_eqn a b c x2 ∧ -4 / 2 = -2) :=
by
  -- skip the proof
  sorry

end arithmetic_mean_of_roots_l1614_161447


namespace remainder_modulus_9_l1614_161482

theorem remainder_modulus_9 : (9 * 7^18 + 2^18) % 9 = 1 := 
by sorry

end remainder_modulus_9_l1614_161482


namespace total_tickets_sold_l1614_161454

def price_adult_ticket : ℕ := 7
def price_child_ticket : ℕ := 4
def total_revenue : ℕ := 5100
def adult_tickets_sold : ℕ := 500

theorem total_tickets_sold : 
  ∃ (child_tickets_sold : ℕ), 
    price_adult_ticket * adult_tickets_sold + price_child_ticket * child_tickets_sold = total_revenue ∧
    adult_tickets_sold + child_tickets_sold = 900 :=
by
  sorry

end total_tickets_sold_l1614_161454


namespace usual_time_to_bus_stop_l1614_161495

theorem usual_time_to_bus_stop
  (T : ℕ) (S : ℕ)
  (h : S * T = (4/5 * S) * (T + 9)) :
  T = 36 :=
by
  sorry

end usual_time_to_bus_stop_l1614_161495


namespace algae_plants_in_milford_lake_l1614_161497

theorem algae_plants_in_milford_lake (original : ℕ) (increase : ℕ) : (original = 809) → (increase = 2454) → (original + increase = 3263) :=
by
  sorry

end algae_plants_in_milford_lake_l1614_161497


namespace triangle_third_side_count_l1614_161426

theorem triangle_third_side_count : 
  ∀ (x : ℕ), (3 < x ∧ x < 19) → ∃ (n : ℕ), n = 15 := 
by 
  sorry

end triangle_third_side_count_l1614_161426


namespace point_in_second_quadrant_coordinates_l1614_161448

variable (x y : ℝ)
variable (P : ℝ × ℝ)
variable (h1 : P.1 = x)
variable (h2 : P.2 = y)

def isInSecondQuadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def distanceToXAxis (P : ℝ × ℝ) : ℝ :=
  abs P.2

def distanceToYAxis (P : ℝ × ℝ) : ℝ :=
  abs P.1

theorem point_in_second_quadrant_coordinates (h1 : isInSecondQuadrant P)
    (h2 : distanceToXAxis P = 2)
    (h3 : distanceToYAxis P = 1) :
    P = (-1, 2) :=
by 
  sorry

end point_in_second_quadrant_coordinates_l1614_161448


namespace regression_estimate_l1614_161415

theorem regression_estimate :
  ∀ (x y : ℝ), (y = 0.50 * x - 0.81) → x = 25 → y = 11.69 :=
by
  intros x y h_eq h_x_val
  sorry

end regression_estimate_l1614_161415


namespace quadrilateral_condition_l1614_161401

variable (a b c d : ℝ)

theorem quadrilateral_condition (h1 : a + b + c + d = 2) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a < 1 ∧ b < 1 ∧ c < 1 ∧ d < 1 ∧ a + b + c > 1 :=
by
  sorry

end quadrilateral_condition_l1614_161401


namespace exists_c_gt_zero_l1614_161490

theorem exists_c_gt_zero (a b : ℕ) (h_a_square_free : ¬ ∃ (k : ℕ), k^2 ∣ a)
    (h_b_square_free : ¬ ∃ (k : ℕ), k^2 ∣ b) (h_a_b_distinct : a ≠ b) :
    ∃ c > 0, ∀ n : ℕ, n > 0 →
    |(n * Real.sqrt a % 1) - (n * Real.sqrt b % 1)| > c / n^3 := sorry

end exists_c_gt_zero_l1614_161490


namespace g_at_five_l1614_161480

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_five :
  (∀ x : ℝ, g (3 * x - 7) = 4 * x + 6) →
  g (5) = 22 :=
by
  intros h
  sorry

end g_at_five_l1614_161480


namespace value_of_a2022_l1614_161428

theorem value_of_a2022 (a : ℕ → ℤ) (h : ∀ (n k : ℕ), 1 ≤ n ∧ n ≤ 2022 ∧ 1 ≤ k ∧ k ≤ 2022 → a n - a k ≥ (n^3 : ℤ) - (k^3 : ℤ)) (ha1011 : a 1011 = 0) : 
  a 2022 = 7246031367 := 
by
  sorry

end value_of_a2022_l1614_161428


namespace range_of_a_l1614_161437

-- Define the propositions p and q
def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) : Prop :=
  a > 1

-- Translate the problem to a Lean 4 statement
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → a ∈ Set.Icc (-2 : ℝ) 1 ∪ Set.Ici 2 :=
by
  sorry

end range_of_a_l1614_161437


namespace town_population_l1614_161475

variable (P₀ P₁ P₂ : ℝ)

def population_two_years_ago (P₀ : ℝ) : Prop := P₀ = 800

def first_year_increase (P₀ P₁ : ℝ) : Prop := P₁ = P₀ * 1.25

def second_year_increase (P₁ P₂ : ℝ) : Prop := P₂ = P₁ * 1.15

theorem town_population 
  (h₀ : population_two_years_ago P₀)
  (h₁ : first_year_increase P₀ P₁)
  (h₂ : second_year_increase P₁ P₂) : 
  P₂ = 1150 := 
sorry

end town_population_l1614_161475


namespace only_A_forms_triangle_l1614_161449

def triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem only_A_forms_triangle :
  (triangle_inequality 5 6 10) ∧ ¬(triangle_inequality 5 2 9) ∧ ¬(triangle_inequality 5 7 12) ∧ ¬(triangle_inequality 3 4 8) :=
by
  sorry

end only_A_forms_triangle_l1614_161449


namespace team_A_minimum_workers_l1614_161417

-- Define the variables and conditions for the problem.
variables (A B c : ℕ)

-- Condition 1: If team A lends 90 workers to team B, Team B will have twice as many workers as Team A.
def condition1 : Prop :=
  2 * (A - 90) = B + 90

-- Condition 2: If team B lends c workers to team A, Team A will have six times as many workers as Team B.
def condition2 : Prop :=
  A + c = 6 * (B - c)

-- Define the proof goal.
theorem team_A_minimum_workers (h1 : condition1 A B) (h2 : condition2 A B c) : 
  153 ≤ A :=
sorry

end team_A_minimum_workers_l1614_161417


namespace cost_equivalence_min_sets_of_A_l1614_161400

noncomputable def cost_of_B := 120
noncomputable def cost_of_A := cost_of_B + 30

theorem cost_equivalence (x : ℕ) :
  (1200 / (x + 30) = 960 / x) → x = 120 :=
by
  sorry

theorem min_sets_of_A :
  ∀ m : ℕ, (150 * m + 120 * (20 - m) ≥ 2800) ↔ m ≥ 14 :=
by
  sorry

end cost_equivalence_min_sets_of_A_l1614_161400


namespace full_price_ticket_revenue_l1614_161473

theorem full_price_ticket_revenue (f t : ℕ) (p : ℝ) 
  (h1 : f + t = 160) 
  (h2 : f * p + t * (p / 3) = 2500) 
  (h3 : p = 30) :
  f * p = 1350 := 
by sorry

end full_price_ticket_revenue_l1614_161473


namespace jason_books_l1614_161446

theorem jason_books (books_per_shelf : ℕ) (num_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 45 → num_shelves = 7 → total_books = books_per_shelf * num_shelves → total_books = 315 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jason_books_l1614_161446


namespace combine_fraction_l1614_161406

variable (d : ℤ)

theorem combine_fraction : (3 + 4 * d) / 5 + 3 = (18 + 4 * d) / 5 := by
  sorry

end combine_fraction_l1614_161406


namespace find_x_l1614_161445

theorem find_x (x : ℕ) (h₁ : 3 * (Nat.factorial 8) / (Nat.factorial (8 - x)) = 4 * (Nat.factorial 9) / (Nat.factorial (9 - (x - 1)))) : x = 6 :=
sorry

end find_x_l1614_161445


namespace num_ordered_triples_l1614_161429

theorem num_ordered_triples : 
  {n : ℕ // ∃ (a b c : ℤ), 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = (2 * (a * b + b * c + c * a)) / 3 ∧ n = 3} :=
sorry

end num_ordered_triples_l1614_161429


namespace response_rate_percentage_50_l1614_161432

def questionnaire_response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ) : ℕ :=
  (responses_needed * 100) / questionnaires_mailed

theorem response_rate_percentage_50 
  (responses_needed : ℕ) 
  (questionnaires_mailed : ℕ) 
  (h1 : responses_needed = 300) 
  (h2 : questionnaires_mailed = 600) : 
  questionnaire_response_rate_percentage responses_needed questionnaires_mailed = 50 :=
by 
  rw [h1, h2]
  norm_num
  sorry

end response_rate_percentage_50_l1614_161432


namespace arccos_neg_one_eq_pi_l1614_161479

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l1614_161479


namespace find_cost_price_l1614_161498

noncomputable def original_cost_price (C S C_new S_new : ℝ) : Prop :=
  S = 1.25 * C ∧
  C_new = 0.80 * C ∧
  S_new = 1.25 * C - 10.50 ∧
  S_new = 1.04 * C

theorem find_cost_price (C S C_new S_new : ℝ) :
  original_cost_price C S C_new S_new → C = 50 :=
by
  sorry

end find_cost_price_l1614_161498


namespace triangle_inequality_sum_2_l1614_161462

theorem triangle_inequality_sum_2 (a b c : ℝ) (h_triangle : a + b + c = 2) (h_side_ineq : a + c > b ∧ a + b > c ∧ b + c > a):
  1 ≤ a * b + b * c + c * a - a * b * c ∧ a * b + b * c + c * a - a * b * c ≤ 1 + 1 / 27 :=
by
  sorry

end triangle_inequality_sum_2_l1614_161462


namespace compare_expression_l1614_161474

variable (m x : ℝ)

theorem compare_expression : x^2 - x + 1 > -2 * m^2 - 2 * m * x := 
sorry

end compare_expression_l1614_161474


namespace impossible_to_place_50_pieces_on_torus_grid_l1614_161466

theorem impossible_to_place_50_pieces_on_torus_grid :
  ¬ (∃ (a b c x y z : ℕ),
    a + b + c = 50 ∧
    2 * a ≤ x ∧ x ≤ 2 * b ∧
    2 * b ≤ y ∧ y ≤ 2 * c ∧
    2 * c ≤ z ∧ z ≤ 2 * a) :=
by
  sorry

end impossible_to_place_50_pieces_on_torus_grid_l1614_161466


namespace intersection_P_Q_l1614_161451

def P : Set ℝ := {x | |x| > 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {x | -2 ≤ x ∧ x < -1 ∨ 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_P_Q_l1614_161451


namespace katrina_cookies_left_l1614_161435

def initial_cookies : ℕ := 120
def morning_sales : ℕ := 3 * 12
def lunch_sales : ℕ := 57
def afternoon_sales : ℕ := 16
def total_sales : ℕ := morning_sales + lunch_sales + afternoon_sales
def cookies_left_to_take_home (initial: ℕ) (sold: ℕ) : ℕ := initial - sold

theorem katrina_cookies_left :
  cookies_left_to_take_home initial_cookies total_sales = 11 :=
by sorry

end katrina_cookies_left_l1614_161435


namespace cube_faces_edges_vertices_sum_l1614_161493

theorem cube_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  sorry

end cube_faces_edges_vertices_sum_l1614_161493


namespace arrangement_problem_l1614_161470
noncomputable def num_arrangements : ℕ := 144

theorem arrangement_problem (A B C D E F : ℕ) 
  (adjacent_easy : A = B) 
  (not_adjacent_difficult : E ≠ F) : num_arrangements = 144 :=
by sorry

end arrangement_problem_l1614_161470


namespace relative_frequency_defective_books_l1614_161464

theorem relative_frequency_defective_books 
  (N_defective : ℤ) (N_total : ℤ)
  (h_defective : N_defective = 5)
  (h_total : N_total = 100) :
  (N_defective : ℚ) / N_total = 0.05 := by
  sorry

end relative_frequency_defective_books_l1614_161464


namespace false_statements_count_is_3_l1614_161441

-- Define the statements
def statement1_false : Prop := ¬ (1 ≠ 1)     -- Not exactly one statement is false
def statement2_false : Prop := ¬ (2 ≠ 2)     -- Not exactly two statements are false
def statement3_false : Prop := ¬ (3 ≠ 3)     -- Not exactly three statements are false
def statement4_false : Prop := ¬ (4 ≠ 4)     -- Not exactly four statements are false
def statement5_false : Prop := ¬ (5 ≠ 5)     -- Not all statements are false

-- Prove that the number of false statements is 3
theorem false_statements_count_is_3 :
  (statement1_false → statement2_false →
  statement3_false → statement4_false →
  statement5_false → (3 = 3)) := by
  sorry

end false_statements_count_is_3_l1614_161441


namespace min_value_of_expression_l1614_161468

noncomputable def minExpression (x : ℝ) : ℝ := (15 - x) * (14 - x) * (15 + x) * (14 + x)

theorem min_value_of_expression : ∀ x : ℝ, ∃ m : ℝ, (m ≤ minExpression x) ∧ (m = -142.25) :=
by
  sorry

end min_value_of_expression_l1614_161468


namespace find_constants_l1614_161486

theorem find_constants (c d : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
     (r^3 + c*r^2 + 17*r + 10 = 0) ∧ (s^3 + c*s^2 + 17*s + 10 = 0) ∧
     (r^3 + d*r^2 + 22*r + 14 = 0) ∧ (s^3 + d*s^2 + 22*s + 14 = 0)) →
  (c = 8 ∧ d = 9) :=
by
  sorry

end find_constants_l1614_161486


namespace height_of_taller_tree_l1614_161487

theorem height_of_taller_tree 
  (h : ℝ) 
  (ratio_condition : (h - 20) / h = 2 / 3) : 
  h = 60 := 
by 
  sorry

end height_of_taller_tree_l1614_161487


namespace last_digit_11_power_11_last_digit_9_power_9_last_digit_9219_power_9219_last_digit_2014_power_2014_l1614_161444

-- Definition of function to calculate the last digit of a number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Proof statements
theorem last_digit_11_power_11 : last_digit (11 ^ 11) = 1 := sorry

theorem last_digit_9_power_9 : last_digit (9 ^ 9) = 9 := sorry

theorem last_digit_9219_power_9219 : last_digit (9219 ^ 9219) = 9 := sorry

theorem last_digit_2014_power_2014 : last_digit (2014 ^ 2014) = 6 := sorry

end last_digit_11_power_11_last_digit_9_power_9_last_digit_9219_power_9219_last_digit_2014_power_2014_l1614_161444


namespace daily_sales_volume_and_profit_profit_for_1200_yuan_profit_impossible_for_1800_yuan_l1614_161489

-- Part (1)
theorem daily_sales_volume_and_profit (x : ℝ) :
  let increase_in_sales := 2 * x
  let profit_per_piece := 40 - x
  increase_in_sales = 2 * x ∧ profit_per_piece = 40 - x :=
by
  sorry

-- Part (2)
theorem profit_for_1200_yuan (x : ℝ) (h1 : (40 - x) * (20 + 2 * x) = 1200) :
  x = 10 ∨ x = 20 :=
by
  sorry

-- Part (3)
theorem profit_impossible_for_1800_yuan :
  ¬ ∃ y : ℝ, (40 - y) * (20 + 2 * y) = 1800 :=
by
  sorry

end daily_sales_volume_and_profit_profit_for_1200_yuan_profit_impossible_for_1800_yuan_l1614_161489


namespace jogging_path_diameter_l1614_161491

theorem jogging_path_diameter 
  (d_pond : ℝ)
  (w_flowerbed : ℝ)
  (w_jogging_path : ℝ)
  (h_pond : d_pond = 20)
  (h_flowerbed : w_flowerbed = 10)
  (h_jogging_path : w_jogging_path = 12) :
  2 * (d_pond / 2 + w_flowerbed + w_jogging_path) = 64 :=
by
  sorry

end jogging_path_diameter_l1614_161491


namespace compare_abc_l1614_161431

noncomputable def a : ℝ := (1 / 2) * Real.cos (4 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (4 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (2 * 13 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (2 * 23 * Real.pi / 180)

theorem compare_abc : a < b ∧ b < c := by
  sorry

end compare_abc_l1614_161431


namespace hyperbola_eccentricity_l1614_161476

theorem hyperbola_eccentricity
    (a b e : ℝ)
    (ha : a > 0)
    (hb : b > 0)
    (h_hyperbola : ∀ x y, x ^ 2 / a^2 - y^2 / b^2 = 1)
    (h_circle : ∀ x y, (x - 2) ^ 2 + y ^ 2 = 4)
    (h_chord_length : ∀ x y, (x ^ 2 + y ^ 2)^(1/2) = 2) :
    e = 2 := 
sorry

end hyperbola_eccentricity_l1614_161476


namespace perpendicular_line_through_A_l1614_161469

variable (m : ℝ)

-- Conditions
def line1 (x y : ℝ) : Prop := x + (1 + m) * y + m - 2 = 0
def line2 (x y : ℝ) : Prop := m * x + 2 * y + 8 = 0
def pointA : ℝ × ℝ := (3, 2)

-- Question and proof
theorem perpendicular_line_through_A (h_parallel : ∃ x y, line1 m x y ∧ line2 m x y) :
  ∃ (t : ℝ), ∀ (x y : ℝ), (y = 2 * x + t) ↔ (2 * x - y - 4 = 0) :=
by
  sorry

end perpendicular_line_through_A_l1614_161469


namespace yura_finishes_problems_by_sept_12_l1614_161477

def total_problems := 91
def initial_date := 6 -- September 6
def problems_left_date := 8 -- September 8
def remaining_problems := 46
def decreasing_rate := 1

def problems_solved (z : ℕ) (day : ℕ) : ℕ :=
if day = 6 then z + 1 else if day = 7 then z else if day = 8 then z - 1 else z - (day - 7)

theorem yura_finishes_problems_by_sept_12 :
  ∃ z : ℕ, (problems_solved z 6 + problems_solved z 7 + problems_solved z 8 = total_problems - remaining_problems) ∧
           (problems_solved z 6 + problems_solved z 7 + problems_solved z 8 + problems_solved z 9 + problems_solved z 10 + problems_solved z 11 + problems_solved z 12 = total_problems) :=
sorry

end yura_finishes_problems_by_sept_12_l1614_161477


namespace profit_percentage_is_4_l1614_161457

-- Define the cost price and selling price
def cost_price : Nat := 600
def selling_price : Nat := 624

-- Calculate profit in dollars
def profit_dollars : Nat := selling_price - cost_price

-- Calculate profit percentage
def profit_percentage : Nat := (profit_dollars * 100) / cost_price

-- Prove that the profit percentage is 4%
theorem profit_percentage_is_4 : profit_percentage = 4 := by
  sorry

end profit_percentage_is_4_l1614_161457


namespace fewer_people_correct_l1614_161409

def pop_Springfield : ℕ := 482653
def pop_total : ℕ := 845640
def pop_new_city : ℕ := pop_total - pop_Springfield
def fewer_people : ℕ := pop_Springfield - pop_new_city

theorem fewer_people_correct : fewer_people = 119666 :=
by
  unfold fewer_people
  unfold pop_new_city
  unfold pop_total
  unfold pop_Springfield
  sorry

end fewer_people_correct_l1614_161409


namespace is_quadratic_function_l1614_161403

theorem is_quadratic_function (x : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = 2 * x + 3) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = 2 / x) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = (x - 1)^2 - x^2) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = 3 * x^2 - 1) ∧ (∃ a b c : ℝ, a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c))) :=
by
  sorry

end is_quadratic_function_l1614_161403


namespace geometric_sequence_term_302_l1614_161414

def geometric_sequence (a r : ℤ) (n : ℕ) : ℤ := a * r ^ (n - 1)

theorem geometric_sequence_term_302 :
  let a := 8
  let r := -2
  geometric_sequence a r 302 = -2^304 := by
  sorry

end geometric_sequence_term_302_l1614_161414


namespace triangle_area_difference_l1614_161481

-- Definitions per conditions
def right_angle (A B C : Type) (angle_EAB : Prop) : Prop := angle_EAB
def angle_ABC_eq_30 (A B C : Type) (angle_ABC : ℝ) : Prop := angle_ABC = 30
def length_AB_eq_5 (A B : Type) (AB : ℝ) : Prop := AB = 5
def length_BC_eq_7 (B C : Type) (BC : ℝ) : Prop := BC = 7
def length_AE_eq_10 (A E : Type) (AE : ℝ) : Prop := AE = 10
def lines_intersect_at_D (A B C E D : Type) (intersects : Prop) : Prop := intersects

-- Main theorem statement
theorem triangle_area_difference
  (A B C E D : Type)
  (angle_EAB : Prop)
  (right_EAB : right_angle A E B angle_EAB)
  (angle_ABC : ℝ)
  (angle_ABC_is_30 : angle_ABC_eq_30 A B C angle_ABC)
  (AB : ℝ)
  (AB_is_5 : length_AB_eq_5 A B AB)
  (BC : ℝ)
  (BC_is_7 : length_BC_eq_7 B C BC)
  (AE : ℝ)
  (AE_is_10 : length_AE_eq_10 A E AE)
  (intersects : Prop)
  (intersects_at_D : lines_intersect_at_D A B C E D intersects) :
  (area_ADE - area_BDC) = 16.25 := sorry

end triangle_area_difference_l1614_161481


namespace total_number_of_coins_l1614_161456

theorem total_number_of_coins (num_5c : Nat) (num_10c : Nat) (h1 : num_5c = 16) (h2 : num_10c = 16) : num_5c + num_10c = 32 := by
  sorry

end total_number_of_coins_l1614_161456


namespace squares_ratio_l1614_161483

noncomputable def inscribed_squares_ratio :=
  let x := 60 / 17
  let y := 780 / 169
  (x / y : ℚ)

theorem squares_ratio (x y : ℚ) (h₁ : x = 60 / 17) (h₂ : y = 780 / 169) :
  x / y = 169 / 220 := by
  rw [h₁, h₂]
  -- Here we would perform calculations to show equality, omitted for brevity.
  sorry

end squares_ratio_l1614_161483


namespace cost_of_each_shirt_l1614_161436

theorem cost_of_each_shirt
  (x : ℝ) 
  (h : 3 * x + 2 * 20 = 85) : x = 15 :=
sorry

end cost_of_each_shirt_l1614_161436


namespace students_taking_both_languages_l1614_161419

theorem students_taking_both_languages (F S B : ℕ) (hF : F = 21) (hS : S = 21) (h30 : 30 = F - B + (S - B)) : B = 6 :=
by
  rw [hF, hS] at h30
  sorry

end students_taking_both_languages_l1614_161419


namespace min_value_ge_9_l1614_161458

noncomputable def minValue (θ : ℝ) (h : θ ∈ Set.Ioo 0 (π / 2)) : ℝ :=
  1 / (Real.sin θ) ^ 2 + 4 / (Real.cos θ) ^ 2

theorem min_value_ge_9 (θ : ℝ) (h : θ ∈ Set.Ioo 0 (π / 2)) : minValue θ h ≥ 9 := 
  sorry

end min_value_ge_9_l1614_161458


namespace cos_theta_seven_l1614_161404

theorem cos_theta_seven {θ : ℝ} (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -8383 / 98304 :=
by
  sorry

end cos_theta_seven_l1614_161404


namespace pitcher_fill_four_glasses_l1614_161420

variable (P G : ℚ) -- P: Volume of pitcher, G: Volume of one glass
variable (h : P / 2 = 3 * G)

theorem pitcher_fill_four_glasses : (4 * G = 2 * P / 3) :=
by
  sorry

end pitcher_fill_four_glasses_l1614_161420


namespace quadratic_has_real_roots_find_value_of_m_l1614_161410

theorem quadratic_has_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2) ≠ 0 ∧ (x^2 - 4 * m * x + 3 * m^2 = 0) := 
by 
  sorry

theorem find_value_of_m (m : ℝ) (h1 : m > 0) (h2 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2 = 2)) :
  m = 1 := 
by 
  sorry

end quadratic_has_real_roots_find_value_of_m_l1614_161410


namespace secret_code_count_l1614_161427

noncomputable def number_of_secret_codes (colors slots : ℕ) : ℕ :=
  colors ^ slots

theorem secret_code_count : number_of_secret_codes 9 5 = 59049 := by
  sorry

end secret_code_count_l1614_161427


namespace relationship_in_size_l1614_161463

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 2.1
noncomputable def c : ℝ := Real.log (1.5) / Real.log (2)

theorem relationship_in_size : b > a ∧ a > c := by
  sorry

end relationship_in_size_l1614_161463


namespace inequality_holds_l1614_161484

theorem inequality_holds (c : ℝ) (X Y : ℝ) (h1 : X^2 - c * X - c = 0) (h2 : Y^2 - c * Y - c = 0) :
    X^3 + Y^3 + (X * Y)^3 ≥ 0 :=
sorry

end inequality_holds_l1614_161484


namespace sales_tax_difference_l1614_161433

/-- The difference in sales tax calculation given the changes in rate. -/
theorem sales_tax_difference 
  (market_price : ℝ := 9000) 
  (original_rate : ℝ := 0.035) 
  (new_rate : ℝ := 0.0333) 
  (difference : ℝ := 15.3) :
  market_price * original_rate - market_price * new_rate = difference :=
by
  /- The proof is omitted as per the instructions. -/
  sorry

end sales_tax_difference_l1614_161433


namespace find_expression_l1614_161439

-- Definitions based on the conditions provided
def prop_rel (y x : ℝ) (k : ℝ) : Prop :=
  y = k * (x - 2)

def prop_value_k (k : ℝ) : Prop :=
  k = -4

def prop_value_y (y x : ℝ) : Prop :=
  y = -4 * x + 8

theorem find_expression (y x k : ℝ) : 
  (prop_rel y x k) → 
  (x = 3) → 
  (y = -4) → 
  (prop_value_k k) → 
  (prop_value_y y x) :=
by
  intros h1 h2 h3 h4
  subst h4
  subst h3
  subst h2
  sorry

end find_expression_l1614_161439


namespace simplify_expression_l1614_161443

theorem simplify_expression (x : ℝ) 
  (h1 : x^2 - 4*x + 3 = (x-3)*(x-1))
  (h2 : x^2 - 6*x + 9 = (x-3)^2)
  (h3 : x^2 - 6*x + 8 = (x-2)*(x-4))
  (h4 : x^2 - 8*x + 15 = (x-3)*(x-5)) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = (x-1)*(x-5) / ((x-2)*(x-4)) :=
by
  sorry

end simplify_expression_l1614_161443


namespace compare_polynomials_l1614_161450

theorem compare_polynomials (x : ℝ) : 2 * x^2 - 2 * x + 1 > x^2 - 2 * x := 
by
  sorry

end compare_polynomials_l1614_161450


namespace polynomial_multiplication_correct_l1614_161453

noncomputable def polynomial_expansion : Polynomial ℤ :=
  (Polynomial.C (3 : ℤ) * Polynomial.X ^ 3 + Polynomial.C (4 : ℤ) * Polynomial.X ^ 2 - Polynomial.C (8 : ℤ) * Polynomial.X - Polynomial.C (5 : ℤ)) *
  (Polynomial.C (2 : ℤ) * Polynomial.X ^ 4 - Polynomial.C (3 : ℤ) * Polynomial.X ^ 2 + Polynomial.C (1 : ℤ))

theorem polynomial_multiplication_correct :
  polynomial_expansion = Polynomial.C (6 : ℤ) * Polynomial.X ^ 7 +
                         Polynomial.C (12 : ℤ) * Polynomial.X ^ 6 -
                         Polynomial.C (25 : ℤ) * Polynomial.X ^ 5 -
                         Polynomial.C (20 : ℤ) * Polynomial.X ^ 4 +
                         Polynomial.C (34 : ℤ) * Polynomial.X ^ 2 -
                         Polynomial.C (8 : ℤ) * Polynomial.X -
                         Polynomial.C (5 : ℤ) :=
by
  sorry

end polynomial_multiplication_correct_l1614_161453


namespace difference_of_cubes_not_divisible_by_19_l1614_161418

theorem difference_of_cubes_not_divisible_by_19 (a b : ℤ) : 
  ¬ (19 ∣ ((3 * a + 2) ^ 3 - (3 * b + 2) ^ 3)) := by
  sorry

end difference_of_cubes_not_divisible_by_19_l1614_161418


namespace morning_registration_count_l1614_161492

variable (M : ℕ) -- Number of students registered for the morning session
variable (MorningAbsentees : ℕ := 3) -- Absentees in the morning session
variable (AfternoonRegistered : ℕ := 24) -- Students registered for the afternoon session
variable (AfternoonAbsentees : ℕ := 4) -- Absentees in the afternoon session

theorem morning_registration_count :
  (M - MorningAbsentees) + (AfternoonRegistered - AfternoonAbsentees) = 42 → M = 25 :=
by
  sorry

end morning_registration_count_l1614_161492


namespace min_buses_l1614_161425

theorem min_buses (n : ℕ) : (47 * n >= 625) → (n = 14) :=
by {
  -- Proof is omitted since the problem only asks for the Lean statement, not the solution steps.
  sorry
}

end min_buses_l1614_161425


namespace twenty_percent_less_than_sixty_equals_one_third_more_than_what_number_l1614_161408

theorem twenty_percent_less_than_sixty_equals_one_third_more_than_what_number :
  (4 / 3) * n = 48 → n = 36 :=
by
  intro h
  sorry

end twenty_percent_less_than_sixty_equals_one_third_more_than_what_number_l1614_161408


namespace remainder_division_l1614_161430

theorem remainder_division (n : ℕ) :
  n = 2345678901 →
  n % 102 = 65 :=
by sorry

end remainder_division_l1614_161430


namespace enrollment_inversely_proportional_l1614_161440

theorem enrollment_inversely_proportional :
  ∃ k : ℝ, (40 * 2000 = k) → (s * 2500 = k) → s = 32 :=
by
  sorry

end enrollment_inversely_proportional_l1614_161440


namespace value_of_m_l1614_161422

theorem value_of_m (m : ℤ) (h : ∃ x : ℤ, x = 2 ∧ x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end value_of_m_l1614_161422


namespace time_to_cross_bridge_l1614_161488

-- Defining the given conditions
def length_of_train : ℕ := 110
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 140

-- Conversion factor from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℚ := (speed_kmh * 1000) / 3600

-- Calculating the speed in m/s
def speed_of_train_ms : ℚ := kmh_to_ms speed_of_train_kmh

-- Calculating total distance to be covered
def total_distance : ℕ := length_of_train + length_of_bridge

-- Expected time to cross the bridge
def expected_time : ℚ := total_distance / speed_of_train_ms

-- The proof statement
theorem time_to_cross_bridge :
  expected_time = 12.5 := by
  sorry

end time_to_cross_bridge_l1614_161488


namespace correct_calculation_l1614_161407

theorem correct_calculation (x : ℝ) :
  (x / 5 + 16 = 58) → (x / 15 + 74 = 88) :=
by
  sorry

end correct_calculation_l1614_161407


namespace find_lines_through_p_and_intersecting_circle_l1614_161471

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 25

noncomputable def passes_through (l : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  P.2 = l P.1

noncomputable def chord_length (c p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

theorem find_lines_through_p_and_intersecting_circle :
  ∃ l : ℝ → ℝ, (passes_through l (-2, 3)) ∧
  (∃ p1 p2 : ℝ × ℝ, trajectory_equation p1.1 p1.2 ∧ trajectory_equation p2.1 p2.2 ∧
  chord_length (1, 2) p1 p2 = 8^2) :=
by
  sorry

end find_lines_through_p_and_intersecting_circle_l1614_161471


namespace budget_per_friend_l1614_161412

-- Definitions for conditions
def total_budget : ℕ := 100
def parents_gift_cost : ℕ := 14
def number_of_parents : ℕ := 2
def number_of_friends : ℕ := 8

-- Statement to prove
theorem budget_per_friend :
  (total_budget - number_of_parents * parents_gift_cost) / number_of_friends = 9 :=
by
  sorry

end budget_per_friend_l1614_161412


namespace max_M_inequality_l1614_161461

theorem max_M_inequality :
  ∃ M : ℝ, (∀ x y : ℝ, x + y ≥ 0 → (x^2 + y^2)^3 ≥ M * (x^3 + y^3) * (x * y - x - y)) ∧ M = 32 :=
by {
  sorry
}

end max_M_inequality_l1614_161461


namespace sum_S6_l1614_161434

variable (a_n : ℕ → ℚ)
variable (d : ℚ)
variable (S : ℕ → ℚ)
variable (a1 : ℚ)

/-- Define arithmetic sequence with common difference -/
def arithmetic_seq (n : ℕ) := a1 + n * d

/-- Define the sum of the first n terms of the sequence -/
def sum_of_arith_seq (n : ℕ) := n * a1 + (n * (n - 1) / 2) * d

/-- The given conditions -/
axiom h1 : d = 5
axiom h2 : (a_n 1 = a1) ∧ (a_n 2 = a1 + d) ∧ (a_n 5 = a1 + 4 * d)
axiom geom_seq : (a1 + d)^2 = a1 * (a1 + 4 * d)

theorem sum_S6 : S 6 = 90 := by
  sorry

end sum_S6_l1614_161434


namespace total_people_going_to_museum_l1614_161478

def number_of_people_on_first_bus := 12
def number_of_people_on_second_bus := 2 * number_of_people_on_first_bus
def number_of_people_on_third_bus := number_of_people_on_second_bus - 6
def number_of_people_on_fourth_bus := number_of_people_on_first_bus + 9

theorem total_people_going_to_museum :
  number_of_people_on_first_bus + number_of_people_on_second_bus + number_of_people_on_third_bus + number_of_people_on_fourth_bus = 75 :=
by
  sorry

end total_people_going_to_museum_l1614_161478


namespace exterior_angle_BAC_l1614_161472

theorem exterior_angle_BAC (square_octagon_coplanar : Prop) (common_side_AD : Prop) : 
    angle_BAC = 135 :=
by
  sorry

end exterior_angle_BAC_l1614_161472


namespace ratio_of_down_payment_l1614_161411

theorem ratio_of_down_payment (C D : ℕ) (daily_min : ℕ) (days : ℕ) (balance : ℕ) (total_cost : ℕ) 
  (h1 : total_cost = 120)
  (h2 : daily_min = 6)
  (h3 : days = 10)
  (h4 : balance = daily_min * days) 
  (h5 : D + balance = total_cost) : 
  D / total_cost = 1 / 2 := 
  by
  sorry

end ratio_of_down_payment_l1614_161411


namespace chord_slope_range_l1614_161465

theorem chord_slope_range (x1 y1 x2 y2 x0 y0 : ℝ) (h1 : x1^2 + (y1^2)/4 = 1) (h2 : x2^2 + (y2^2)/4 = 1)
  (h3 : x0 = (x1 + x2) / 2) (h4 : y0 = (y1 + y2) / 2)
  (h5 : x0 = 1/2) (h6 : 1/2 ≤ y0 ∧ y0 ≤ 1) :
  -4 ≤ (-2 / y0) ∧ -2 ≤ (-2 / y0) :=
by
  sorry

end chord_slope_range_l1614_161465


namespace correct_calculation_l1614_161460

theorem correct_calculation (x : ℤ) (h : 7 * (x + 24) / 5 = 70) :
  (5 * x + 24) / 7 = 22 :=
sorry

end correct_calculation_l1614_161460


namespace solution_set_of_inequality_l1614_161455

variable {α : Type*} [LinearOrder α]

def is_decreasing (f : α → α) : Prop :=
  ∀ ⦃x y⦄, x < y → f y < f x

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_decreasing : is_decreasing f)
  (domain_cond : ∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → x ∈ Set.Ioo (-2 : ℝ) 2)
  : { x | x > 0 ∧ x < 1 } = { x | f x > f (2 - x) } :=
by {
  sorry
}

end solution_set_of_inequality_l1614_161455


namespace distances_inequality_l1614_161423

theorem distances_inequality (x y z : ℝ) (h : x + y + z = 1): x^2 + y^2 + z^2 ≥ x^3 + y^3 + z^3 + 6 * x * y * z :=
by
  sorry

end distances_inequality_l1614_161423


namespace exists_1998_distinct_natural_numbers_l1614_161467

noncomputable def exists_1998_distinct_numbers : Prop :=
  ∃ (s : Finset ℕ), s.card = 1998 ∧
    (∀ {x y : ℕ}, x ∈ s → y ∈ s → x ≠ y → (x * y) % ((x - y) ^ 2) = 0)

theorem exists_1998_distinct_natural_numbers : exists_1998_distinct_numbers :=
by
  sorry

end exists_1998_distinct_natural_numbers_l1614_161467


namespace largest_value_l1614_161459

theorem largest_value :
  let A := 1/2
  let B := 1/3 + 1/4
  let C := 1/4 + 1/5 + 1/6
  let D := 1/5 + 1/6 + 1/7 + 1/8
  let E := 1/6 + 1/7 + 1/8 + 1/9 + 1/10
  E > A ∧ E > B ∧ E > C ∧ E > D := by
sorry

end largest_value_l1614_161459


namespace find_AB_l1614_161402

theorem find_AB 
  (A B C Q N : Point)
  (h_AQ_QC : AQ / QC = 5 / 2)
  (h_CN_NB : CN / NB = 5 / 2)
  (h_QN : QN = 5 * Real.sqrt 2) : 
  AB = 7 * Real.sqrt 5 :=
sorry

end find_AB_l1614_161402


namespace fly_flies_more_than_10_meters_l1614_161494

theorem fly_flies_more_than_10_meters :
  ∃ (fly_path_length : ℝ), 
  (∃ (c : ℝ) (a b : ℝ), c = 5 ∧ a^2 + b^2 = c^2) →
  (fly_path_length > 10) := 
by
  sorry

end fly_flies_more_than_10_meters_l1614_161494


namespace andrew_donuts_l1614_161438

/--
Andrew originally asked for 3 donuts for each of his 2 friends, Brian and Samuel. 
Then invited 2 more friends and asked for the same amount of donuts for them. 
Andrew’s mother wants to buy one more donut for each of Andrew’s friends. 
Andrew's mother is also going to buy the same amount of donuts for Andrew as everybody else.
Given these conditions, the total number of donuts Andrew’s mother needs to buy is 20.
-/
theorem andrew_donuts : (3 * 2) + (3 * 2) + 4 + 4 = 20 :=
by
  -- Given:
  -- 1. Andrew asked for 3 donuts for each of his two friends, Brian and Samuel.
  -- 2. He later invited 2 more friends and asked for the same amount of donuts for them.
  -- 3. Andrew’s mother wants to buy one more donut for each of Andrew’s friends.
  -- 4. Andrew’s mother is going to buy the same amount of donuts for Andrew as everybody else.
  -- Prove: The total number of donuts Andrew’s mother needs to buy is 20.
  sorry

end andrew_donuts_l1614_161438


namespace x_plus_q_in_terms_of_q_l1614_161416

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x - 5| = q) (h2 : x > 5) : x + q = 2 * q + 5 :=
by
  sorry

end x_plus_q_in_terms_of_q_l1614_161416


namespace expression_equals_5_l1614_161485

theorem expression_equals_5 : (3^2 - 2^2) = 5 := by
  calc
    (3^2 - 2^2) = 5 := by sorry

end expression_equals_5_l1614_161485


namespace smallest_d_in_range_l1614_161413

theorem smallest_d_in_range (d : ℝ) : (∃ x : ℝ, x^2 + 5 * x + d = 5) ↔ d ≤ 45 / 4 := 
sorry

end smallest_d_in_range_l1614_161413


namespace tangent_perpendicular_point_l1614_161499

open Real

noncomputable def f (x : ℝ) : ℝ := exp x - (1 / 2) * x^2

theorem tangent_perpendicular_point :
  ∃ x0, (f x0 = 1) ∧ (x0 = 0) :=
sorry

end tangent_perpendicular_point_l1614_161499


namespace count_positive_numbers_is_three_l1614_161496

def negative_three := -3
def zero := 0
def negative_three_squared := (-3) ^ 2
def absolute_negative_nine := |(-9)|
def negative_one_raised_to_four := -1 ^ 4

def number_list : List Int := [ -negative_three, zero, negative_three_squared, absolute_negative_nine, negative_one_raised_to_four ]

def count_positive_numbers (lst: List Int) : Nat :=
  lst.foldl (λ acc x => if x > 0 then acc + 1 else acc) 0

theorem count_positive_numbers_is_three : count_positive_numbers number_list = 3 :=
by
  -- The proof will go here.
  sorry

end count_positive_numbers_is_three_l1614_161496


namespace arithmetic_mean_midpoint_l1614_161424

theorem arithmetic_mean_midpoint (a b : ℝ) : ∃ m : ℝ, m = (a + b) / 2 ∧ m = a + (b - a) / 2 :=
by
  sorry

end arithmetic_mean_midpoint_l1614_161424


namespace weighted_average_yield_l1614_161442

-- Define the conditions
def face_value_A : ℝ := 1000
def market_price_A : ℝ := 1200
def yield_A : ℝ := 0.18

def face_value_B : ℝ := 1000
def market_price_B : ℝ := 800
def yield_B : ℝ := 0.22

def face_value_C : ℝ := 1000
def market_price_C : ℝ := 1000
def yield_C : ℝ := 0.15

def investment_A : ℝ := 5000
def investment_B : ℝ := 3000
def investment_C : ℝ := 2000

-- Prove the weighted average yield
theorem weighted_average_yield :
  (investment_A + investment_B + investment_C) = 10000 →
  ((investment_A / (investment_A + investment_B + investment_C)) * yield_A +
   (investment_B / (investment_A + investment_B + investment_C)) * yield_B +
   (investment_C / (investment_A + investment_B + investment_C)) * yield_C) = 0.186 :=
by
  sorry

end weighted_average_yield_l1614_161442


namespace new_profit_is_122_03_l1614_161421

noncomputable def new_profit_percentage (P : ℝ) (tax_rate : ℝ) (profit_rate : ℝ) (market_increase_rate : ℝ) (months : ℕ) : ℝ :=
  let total_cost := P * (1 + tax_rate)
  let initial_selling_price := total_cost * (1 + profit_rate)
  let market_price_after_months := initial_selling_price * (1 + market_increase_rate) ^ months
  let final_selling_price := 2 * initial_selling_price
  let profit := final_selling_price - total_cost
  (profit / total_cost) * 100

theorem new_profit_is_122_03 :
  new_profit_percentage (P : ℝ) 0.18 0.40 0.05 3 = 122.03 := 
by
  sorry

end new_profit_is_122_03_l1614_161421
