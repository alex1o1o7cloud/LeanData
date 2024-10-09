import Mathlib

namespace num_girls_at_park_l2101_210102

theorem num_girls_at_park (G : ℕ) (h1 : 11 + 50 + G = 3 * 25) : G = 14 := by
  sorry

end num_girls_at_park_l2101_210102


namespace avg_student_headcount_l2101_210144

def student_headcount (yr1 yr2 yr3 yr4 : ℕ) : ℕ :=
  (yr1 + yr2 + yr3 + yr4) / 4

theorem avg_student_headcount :
  student_headcount 10600 10800 10500 10400 = 10825 :=
by
  sorry

end avg_student_headcount_l2101_210144


namespace average_of_measurements_l2101_210188

def measurements : List ℝ := [79.4, 80.6, 80.8, 79.1, 80.0, 79.6, 80.5]

theorem average_of_measurements : (measurements.sum / measurements.length) = 80 := by sorry

end average_of_measurements_l2101_210188


namespace no_descending_multiple_of_111_l2101_210174

theorem no_descending_multiple_of_111 :
  ¬ ∃ (n : ℕ), (∀ (i j : ℕ), (i < j ∧ (n / 10^i % 10) < (n / 10^j % 10)) ∨ (i = j)) ∧ 111 ∣ n :=
by
  sorry

end no_descending_multiple_of_111_l2101_210174


namespace sufficient_but_not_necessary_condition_l2101_210127

variable {x k : ℝ}

def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := (2 - x) / (x + 1) < 0

theorem sufficient_but_not_necessary_condition (h_suff : ∀ x, p x k → q x) (h_not_necessary : ∃ x, q x ∧ ¬p x k) : k > 2 :=
sorry

end sufficient_but_not_necessary_condition_l2101_210127


namespace Danai_can_buy_more_decorations_l2101_210187

theorem Danai_can_buy_more_decorations :
  let skulls := 12
  let broomsticks := 4
  let spiderwebs := 12
  let pumpkins := 24 -- 2 times the number of spiderwebs
  let cauldron := 1
  let planned_total := 83
  let budget_left := 10
  let current_decorations := skulls + broomsticks + spiderwebs + pumpkins + cauldron
  current_decorations = 53 → -- 12 + 4 + 12 + 24 + 1
  let additional_decorations_needed := planned_total - current_decorations
  additional_decorations_needed = 30 → -- 83 - 53
  (additional_decorations_needed - budget_left) = 20 → -- 30 - 10
  True := -- proving the statement
sorry

end Danai_can_buy_more_decorations_l2101_210187


namespace six_power_six_div_two_l2101_210140

theorem six_power_six_div_two : 6 ^ (6 / 2) = 216 := by
  sorry

end six_power_six_div_two_l2101_210140


namespace spadesuit_calculation_l2101_210184

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_calculation : spadesuit 5 (spadesuit 3 2) = 0 :=
by
  sorry

end spadesuit_calculation_l2101_210184


namespace martha_total_cost_l2101_210161

def weight_cheese : ℝ := 1.5
def weight_meat : ℝ := 0.55    -- converting grams to kg
def weight_pasta : ℝ := 0.28   -- converting grams to kg
def weight_tomatoes : ℝ := 2.2

def price_cheese_per_kg : ℝ := 6.30
def price_meat_per_kg : ℝ := 8.55
def price_pasta_per_kg : ℝ := 2.40
def price_tomatoes_per_kg : ℝ := 1.79

def tax_cheese : ℝ := 0.07
def tax_meat : ℝ := 0.06
def tax_pasta : ℝ := 0.08
def tax_tomatoes : ℝ := 0.05

def total_cost : ℝ :=
  let cost_cheese := weight_cheese * price_cheese_per_kg * (1 + tax_cheese)
  let cost_meat := weight_meat * price_meat_per_kg * (1 + tax_meat)
  let cost_pasta := weight_pasta * price_pasta_per_kg * (1 + tax_pasta)
  let cost_tomatoes := weight_tomatoes * price_tomatoes_per_kg * (1 + tax_tomatoes)
  cost_cheese + cost_meat + cost_pasta + cost_tomatoes

theorem martha_total_cost : total_cost = 19.9568 := by
  sorry

end martha_total_cost_l2101_210161


namespace total_games_for_18_players_l2101_210126

-- Define the number of players
def num_players : ℕ := 18

-- Define the function to calculate total number of games
def total_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- Theorem statement asserting the total number of games for 18 players
theorem total_games_for_18_players : total_games num_players = 612 :=
by
  -- proof goes here
  sorry

end total_games_for_18_players_l2101_210126


namespace probability_target_A_destroyed_probability_exactly_one_target_destroyed_l2101_210158

-- Definition of probabilities
def prob_A_hits_target_A := 1 / 2
def prob_A_hits_target_B := 1 / 2
def prob_B_hits_target_A := 1 / 3
def prob_B_hits_target_B := 2 / 5

-- The event of target A being destroyed
def prob_target_A_destroyed := prob_A_hits_target_A * prob_B_hits_target_A

-- The event of target B being destroyed
def prob_target_B_destroyed := prob_A_hits_target_B * prob_B_hits_target_B

-- Complementary events
def prob_target_A_not_destroyed := 1 - prob_target_A_destroyed
def prob_target_B_not_destroyed := 1 - prob_target_B_destroyed

-- Exactly one target being destroyed
def prob_exactly_one_target_destroyed := 
  (prob_target_A_destroyed * prob_target_B_not_destroyed) +
  (prob_target_B_destroyed * prob_target_A_not_destroyed)

theorem probability_target_A_destroyed : prob_target_A_destroyed = 1 / 6 := by
  -- Proof needed here
  sorry

theorem probability_exactly_one_target_destroyed : prob_exactly_one_target_destroyed = 3 / 10 := by
  -- Proof needed here
  sorry

end probability_target_A_destroyed_probability_exactly_one_target_destroyed_l2101_210158


namespace adam_played_rounds_l2101_210110

theorem adam_played_rounds (total_points points_per_round : ℕ) (h_total : total_points = 283) (h_per_round : points_per_round = 71) : total_points / points_per_round = 4 := by
  -- sorry is a placeholder for the actual proof
  sorry

end adam_played_rounds_l2101_210110


namespace range_of_m_l2101_210146

theorem range_of_m (m : ℝ) : 
  (∀ x, x^2 + 2 * x - m > 0 ↔ (x = 1 → x^2 + 2 * x - m ≤ 0) ∧ (x = 2 → x^2 + 2 * x - m > 0)) ↔ (3 ≤ m ∧ m < 8) := 
sorry

end range_of_m_l2101_210146


namespace most_reasonable_plan_l2101_210129

-- Defining the conditions as a type
inductive SurveyPlans
| A -- Surveying students in the second grade of School B
| C -- Randomly surveying 150 teachers
| B -- Surveying 600 students randomly selected from School C
| D -- Randomly surveying 150 students from each of the four schools

-- Define the main theorem asserting that the most reasonable plan is Option D
theorem most_reasonable_plan : SurveyPlans.D = SurveyPlans.D :=
by
  sorry

end most_reasonable_plan_l2101_210129


namespace consecutive_integer_sum_l2101_210117

noncomputable def sqrt17 : ℝ := Real.sqrt 17

theorem consecutive_integer_sum : ∃ (a b : ℤ), (b = a + 1) ∧ (a < sqrt17 ∧ sqrt17 < b) ∧ (a + b = 9) :=
by
  sorry

end consecutive_integer_sum_l2101_210117


namespace collinear_points_l2101_210109

theorem collinear_points (k : ℝ) :
  let p1 := (3, 1)
  let p2 := (6, 4)
  let p3 := (10, k + 9)
  let slope (a b : ℝ × ℝ) : ℝ := (b.snd - a.snd) / (b.fst - a.fst)
  slope p1 p2 = slope p1 p3 → k = -1 :=
by 
  let p1 := (3, 1)
  let p2 := (6, 4)
  let p3 := (10, k + 9)
  let slope (a b : ℝ × ℝ) : ℝ := (b.snd - a.snd) / (b.fst - a.fst)
  sorry

end collinear_points_l2101_210109


namespace number_of_consecutive_sum_sets_eq_18_l2101_210141

theorem number_of_consecutive_sum_sets_eq_18 :
  ∃! (S : ℕ → ℕ) (n a : ℕ), (n ≥ 2) ∧ (S n = (n * (2 * a + n - 1)) / 2) ∧ (S n = 18) :=
sorry

end number_of_consecutive_sum_sets_eq_18_l2101_210141


namespace smallest_multiple_of_36_with_digit_product_divisible_by_9_l2101_210180

theorem smallest_multiple_of_36_with_digit_product_divisible_by_9 :
  ∃ n : ℕ, n > 0 ∧ n % 36 = 0 ∧ (∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ (d1 * d2 * d3) % 9 = 0) ∧ n = 936 := 
by
  sorry

end smallest_multiple_of_36_with_digit_product_divisible_by_9_l2101_210180


namespace f_is_even_l2101_210143

-- Given an odd function g
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g (x)

-- Define the function f as given by the problem
def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  abs (g (x^2))

-- The theorem stating that f is an even function
theorem f_is_even (g : ℝ → ℝ) (h_odd : is_odd_function g) : ∀ x, f g x = f g (-x) :=
by
  sorry

end f_is_even_l2101_210143


namespace fish_kept_l2101_210183

theorem fish_kept (Leo_caught Agrey_more Sierra_more Leo_fish Returned : ℕ) 
                  (Agrey_caught : Agrey_more = 20) 
                  (Sierra_caught : Sierra_more = 15) 
                  (Leo_caught_cond : Leo_fish = 40) 
                  (Returned_cond : Returned = 30) : 
                  (Leo_fish + (Leo_fish + Agrey_more) + ((Leo_fish + Agrey_more) + Sierra_more) - Returned) = 145 :=
by
  sorry

end fish_kept_l2101_210183


namespace turtles_received_l2101_210171

theorem turtles_received (martha_turtles : ℕ) (marion_turtles : ℕ) (h1 : martha_turtles = 40) 
    (h2 : marion_turtles = martha_turtles + 20) : martha_turtles + marion_turtles = 100 := 
by {
    sorry
}

end turtles_received_l2101_210171


namespace required_run_rate_l2101_210130

theorem required_run_rate
  (run_rate_first_10_overs : ℝ)
  (target_runs : ℝ)
  (overs_first : ℕ)
  (overs_remaining : ℕ)
  (H_run_rate_10_overs : run_rate_first_10_overs = 3.2)
  (H_target_runs : target_runs = 222)
  (H_overs_first : overs_first = 10)
  (H_overs_remaining : overs_remaining = 40) :
  ((target_runs - run_rate_first_10_overs * overs_first) / overs_remaining) = 4.75 := 
by
  sorry

end required_run_rate_l2101_210130


namespace average_marks_in_6_subjects_l2101_210163

/-- The average marks Ashok secured in 6 subjects is 72
Given:
1. The average of marks in 5 subjects is 74.
2. Ashok secured 62 marks in the 6th subject.
-/
theorem average_marks_in_6_subjects (avg_5 : ℕ) (marks_6th : ℕ) (h_avg_5 : avg_5 = 74) (h_marks_6th : marks_6th = 62) : 
  ((avg_5 * 5 + marks_6th) / 6) = 72 :=
  by
  sorry

end average_marks_in_6_subjects_l2101_210163


namespace min_diff_between_y_and_x_l2101_210178

theorem min_diff_between_y_and_x (x y z : ℤ)
    (h1 : x < y)
    (h2 : y < z)
    (h3 : Even x)
    (h4 : Odd y)
    (h5 : Odd z)
    (h6 : z - x = 9) :
    y - x = 1 := 
  by sorry

end min_diff_between_y_and_x_l2101_210178


namespace parabola_line_intersection_distance_l2101_210153

theorem parabola_line_intersection_distance :
  ∀ (x y : ℝ), x^2 = -4 * y ∧ y = x - 1 ∧ x^2 + 4 * x + 4 = 0 →
  abs (y - -1 + (-1 - y)) = 8 :=
by
  sorry

end parabola_line_intersection_distance_l2101_210153


namespace wire_division_l2101_210107

theorem wire_division (L_wire_ft : Nat) (L_wire_inch : Nat) (L_part : Nat) (H1 : L_wire_ft = 5) (H2 : L_wire_inch = 4) (H3 : L_part = 16) :
  (L_wire_ft * 12 + L_wire_inch) / L_part = 4 :=
by 
  sorry

end wire_division_l2101_210107


namespace black_cards_taken_out_l2101_210192

theorem black_cards_taken_out (initial_black : ℕ) (remaining_black : ℕ) (total_cards : ℕ) (black_cards_per_deck : ℕ) :
  total_cards = 52 → black_cards_per_deck = 26 →
  initial_black = black_cards_per_deck → remaining_black = 22 →
  initial_black - remaining_black = 4 := by
  intros
  sorry

end black_cards_taken_out_l2101_210192


namespace ages_sum_l2101_210113

theorem ages_sum (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 :=
by sorry

end ages_sum_l2101_210113


namespace solve_a_b_c_d_l2101_210149

theorem solve_a_b_c_d (n a b c d : ℕ) (h0 : 0 ≤ a) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : 2^n = a^2 + b^2 + c^2 + d^2) : 
  (a, b, c, d) ∈ {p | p = (↑0, ↑0, ↑0, 2^n.div (↑4)) ∨
                  p = (↑0, ↑0, 2^n.div (↑4), 2^n.div (↑4)) ∨
                  p = (2^n.div (↑4), 2^n.div (↑4), 2^n.div (↑4), 2^n.div (↑4)) ∨
                  p = (2^n.div (↑4), 0, 0, 0) ∨
                  p = (0, 2^n.div (↑4), 0, 0) ∨
                  p = (0, 0, 2^n.div (↑4), 0) ∨
                  p = (0, 0, 0, 2^n.div (↑4))} :=
sorry

end solve_a_b_c_d_l2101_210149


namespace simplify_expression_l2101_210182

theorem simplify_expression :
  (Real.sin (Real.pi / 6) + (1 / 2) - 2007^0 + abs (-2) = 2) :=
by
  sorry

end simplify_expression_l2101_210182


namespace vincent_spent_224_l2101_210104

-- Defining the given conditions as constants
def num_books_animal : ℕ := 10
def num_books_outer_space : ℕ := 1
def num_books_trains : ℕ := 3
def cost_per_book : ℕ := 16

-- Summarizing the total number of books
def total_books : ℕ := num_books_animal + num_books_outer_space + num_books_trains
-- Calculating the total cost
def total_cost : ℕ := total_books * cost_per_book

-- Lean statement to prove that Vincent spent $224
theorem vincent_spent_224 : total_cost = 224 := by
  sorry

end vincent_spent_224_l2101_210104


namespace find_integer_n_l2101_210173

theorem find_integer_n : ∃ (n : ℤ), 0 ≤ n ∧ n < 23 ∧ 54126 % 23 = n :=
by
  use 13
  sorry

end find_integer_n_l2101_210173


namespace division_into_rectangles_l2101_210167

theorem division_into_rectangles (figure : Type) (valid_division : figure → Prop) : (∃ ways, ways = 8) :=
by {
  -- assume given conditions related to valid_division using "figure"
  sorry
}

end division_into_rectangles_l2101_210167


namespace value_of_x_l2101_210179

theorem value_of_x (x y : ℝ) (h₁ : x = y - 0.10 * y) (h₂ : y = 125 + 0.10 * 125) : x = 123.75 := 
by
  sorry

end value_of_x_l2101_210179


namespace price_per_eraser_l2101_210118

-- Definitions of the given conditions
def boxes_donated : ℕ := 48
def erasers_per_box : ℕ := 24
def total_money_made : ℝ := 864

-- The Lean statement to prove the price per eraser is $0.75
theorem price_per_eraser : (total_money_made / (boxes_donated * erasers_per_box) = 0.75) := by
  sorry

end price_per_eraser_l2101_210118


namespace general_term_formula_for_sequence_l2101_210191

theorem general_term_formula_for_sequence (a b : ℕ → ℝ) 
  (h1 : ∀ n, 2 * b n = a n + a (n + 1)) 
  (h2 : ∀ n, (a (n + 1))^2 = b n * b (n + 1)) 
  (h3 : a 1 = 1) 
  (h4 : a 2 = 3) :
  ∀ n, a n = (n^2 + n) / 2 :=
by
  sorry

end general_term_formula_for_sequence_l2101_210191


namespace complement_union_result_l2101_210101

open Set

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

theorem complement_union_result :
    U = { x | x < 6 } →
    A = {1, 2, 3} → 
    B = {2, 4, 5} → 
    (U \ A) ∪ (U \ B) = {0, 1, 3, 4, 5} :=
by
    intros hU hA hB
    sorry

end complement_union_result_l2101_210101


namespace cake_area_l2101_210162

theorem cake_area (n : ℕ) (a area_per_piece : ℕ) 
  (h1 : n = 25) 
  (h2 : a = 16) 
  (h3 : area_per_piece = 4 * 4) 
  (h4 : a = area_per_piece) : 
  n * a = 400 := 
by
  sorry

end cake_area_l2101_210162


namespace max_principals_and_assistant_principals_l2101_210181

theorem max_principals_and_assistant_principals : 
  ∀ (years term_principal term_assistant), (years = 10) ∧ (term_principal = 3) ∧ (term_assistant = 2) 
  → ∃ n, n = 9 :=
by
  sorry

end max_principals_and_assistant_principals_l2101_210181


namespace find_k_l2101_210119

theorem find_k (m n k : ℤ) (h1 : m = 2 * n + 5) (h2 : m + 2 = 2 * (n + k) + 5) : k = 0 := by
  sorry

end find_k_l2101_210119


namespace goldfish_initial_count_l2101_210147

theorem goldfish_initial_count (catsfish : ℕ) (fish_left : ℕ) (fish_disappeared : ℕ) (goldfish_initial : ℕ) :
  catsfish = 12 →
  fish_left = 15 →
  fish_disappeared = 4 →
  goldfish_initial = (fish_left + fish_disappeared) - catsfish →
  goldfish_initial = 7 :=
by
  intros h1 h2 h3 h4
  rw [h2, h3, h1] at h4
  exact h4

end goldfish_initial_count_l2101_210147


namespace sum_of_roots_l2101_210151

theorem sum_of_roots (a b c : ℝ) (h_eq : a = 1) (h_b : b = -5) (h_c : c = 6) :
  (-b / a) = 5 := by
sorry

end sum_of_roots_l2101_210151


namespace coins_fit_in_new_box_l2101_210194

-- Definitions
def diameters_bound (d : ℕ) : Prop :=
  d ≤ 10

def box_fits (length width : ℕ) (fits : Prop) : Prop :=
  fits

-- Conditions
axiom coins_diameter_bound : ∀ (d : ℕ), diameters_bound d
axiom original_box_fits : box_fits 30 70 True

-- Proof statement
theorem coins_fit_in_new_box : box_fits 40 60 True :=
sorry

end coins_fit_in_new_box_l2101_210194


namespace correct_choice_D_l2101_210189

variable (a b : Line) (α : Plane)

-- Definitions for the conditions
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry  -- Definition of perpendicular
def is_parallel_line (l1 l2 : Line) : Prop := sorry  -- Definition of parallel lines
def is_parallel_plane (l : Line) (p : Plane) : Prop := sorry  -- Definition of line parallel to plane
def is_subset (l : Line) (p : Plane) : Prop := sorry  -- Definition of line being in a plane

-- The statement of the problem
theorem correct_choice_D :
  (is_parallel_plane a α) ∧ (is_subset b α) → (is_parallel_plane a α) := 
by 
  sorry

end correct_choice_D_l2101_210189


namespace prime_digit_B_l2101_210168

-- Mathematical description
def six_digit_form (B : Nat) : Nat := 3 * 10^5 + 0 * 10^4 + 3 * 10^3 + 7 * 10^2 + 0 * 10^1 + B

-- Prime condition
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

theorem prime_digit_B (B : Nat) : is_prime (six_digit_form B) ↔ B = 3 :=
sorry

end prime_digit_B_l2101_210168


namespace tangent_line_equation_l2101_210134

theorem tangent_line_equation :
  ∀ (x : ℝ) (y : ℝ), y = 4 * x - x^3 → 
  (x = -1) → (y = -3) →
  (∀ (m : ℝ), m = 4 - 3 * (-1)^2) →
  ∃ (line_eq : ℝ → ℝ), (∀ x, line_eq x = x - 2) :=
by
  sorry

end tangent_line_equation_l2101_210134


namespace sum_of_consecutive_pairs_eq_pow_two_l2101_210106

theorem sum_of_consecutive_pairs_eq_pow_two (n m : ℕ) :
  ∃ n m : ℕ, (n * (n + 1) + m * (m + 1) = 2 ^ 2021) :=
sorry

end sum_of_consecutive_pairs_eq_pow_two_l2101_210106


namespace inequality_reciprocal_l2101_210175

theorem inequality_reciprocal (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) : (1 / a < 1 / b) :=
by
  sorry

end inequality_reciprocal_l2101_210175


namespace student_in_eighth_group_l2101_210123

-- Defining the problem: total students and their assignment into groups
def total_students : ℕ := 50
def students_assigned_numbers (n : ℕ) : Prop := n > 0 ∧ n ≤ total_students

-- Grouping students: Each group has 5 students
def grouped_students (group_num student_num : ℕ) : Prop := 
  student_num > (group_num - 1) * 5 ∧ student_num ≤ group_num * 5

-- Condition: Student 12 is selected from the third group
def condition : Prop := grouped_students 3 12

-- Goal: the number of the student selected from the eighth group is 37
theorem student_in_eighth_group : condition → grouped_students 8 37 :=
by
  sorry

end student_in_eighth_group_l2101_210123


namespace power_mod_l2101_210103

theorem power_mod (a : ℕ) : 5 ^ 2023 % 17 = 2 := by
  sorry

end power_mod_l2101_210103


namespace max_sector_area_l2101_210132

theorem max_sector_area (r l : ℝ) (hp : 2 * r + l = 40) : (1 / 2) * l * r ≤ 100 := 
by
  sorry

end max_sector_area_l2101_210132


namespace area_of_fourth_square_l2101_210159

open Real

theorem area_of_fourth_square
  (EF FG GH : ℝ)
  (hEF : EF = 5)
  (hFG : FG = 7)
  (hGH : GH = 8) :
  let EG := sqrt (EF^2 + FG^2)
  let EH := sqrt (EG^2 + GH^2)
  EH^2 = 138 :=
by
  sorry

end area_of_fourth_square_l2101_210159


namespace percentage_puppies_greater_profit_l2101_210185

/-- A dog breeder wants to know what percentage of puppies he can sell for a greater profit.
    Puppies with more than 4 spots sell for more money. The last litter had 10 puppies; 
    6 had 5 spots, 3 had 4 spots, and 1 had 2 spots.
    We need to prove that the percentage of puppies that can be sold for more profit is 60%. -/
theorem percentage_puppies_greater_profit
  (total_puppies : ℕ := 10)
  (puppies_with_5_spots : ℕ := 6)
  (puppies_with_4_spots : ℕ := 3)
  (puppies_with_2_spots : ℕ := 1)
  (puppies_with_more_than_4_spots := puppies_with_5_spots) :
  (puppies_with_more_than_4_spots : ℝ) / (total_puppies : ℝ) * 100 = 60 :=
by
  sorry

end percentage_puppies_greater_profit_l2101_210185


namespace derivative_at_2_l2101_210186

def f (x : ℝ) : ℝ := (x + 3) * (x + 2) * (x + 1) * x * (x - 1) * (x - 2) * (x - 3)

theorem derivative_at_2 : (deriv f 2) = -120 :=
by
  sorry

end derivative_at_2_l2101_210186


namespace expression_evaluation_l2101_210165

theorem expression_evaluation : (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end expression_evaluation_l2101_210165


namespace total_number_of_animals_l2101_210177

-- Prove that the total number of animals is 300 given the conditions described.
theorem total_number_of_animals (A : ℕ) (H₁ : 4 * (A / 3) = 400) : A = 300 :=
sorry

end total_number_of_animals_l2101_210177


namespace simplify_expression_l2101_210122

theorem simplify_expression (x y : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * y + 15 * y + 18 + 21 = 18 * x + 27 * y + 39 :=
by
  sorry

end simplify_expression_l2101_210122


namespace initial_position_is_minus_one_l2101_210170

def initial_position_of_A (A B C : ℤ) : Prop :=
  B = A - 3 ∧ C = B + 5 ∧ C = 1 ∧ A = -1

theorem initial_position_is_minus_one (A B C : ℤ) (h1 : B = A - 3) (h2 : C = B + 5) (h3 : C = 1) : A = -1 :=
  by sorry

end initial_position_is_minus_one_l2101_210170


namespace cone_volume_increase_l2101_210142

open Real

noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
noncomputable def new_height (h : ℝ) : ℝ := 2 * h
noncomputable def new_volume (r h : ℝ) : ℝ := cone_volume r (new_height h)

theorem cone_volume_increase (r h : ℝ) : new_volume r h = 2 * (cone_volume r h) :=
by
  sorry

end cone_volume_increase_l2101_210142


namespace units_digit_of_product_l2101_210198

/-
Problem: What is the units digit of the product of the first three even positive composite numbers?
Conditions: 
- The first three even positive composite numbers are 4, 6, and 8.
Proof: Prove that the units digit of their product is 2.
-/

def even_positive_composite_numbers := [4, 6, 8]
def product := List.foldl (· * ·) 1 even_positive_composite_numbers
def units_digit (n : Nat) := n % 10

theorem units_digit_of_product : units_digit product = 2 := by
  sorry

end units_digit_of_product_l2101_210198


namespace solution_set_of_inequality_l2101_210176

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_tangent : ∀ x₀ y₀, y₀ = f x₀ → (∀ x, f x = y₀ + (3*x₀^2 - 6*x₀)*(x - x₀)))
  (h_at_3 : f 3 = 0) :
  {x : ℝ | ((x - 1) / f x) ≥ 0} = {x : ℝ | x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 1} ∪ {x : ℝ | x > 3} :=
sorry

end solution_set_of_inequality_l2101_210176


namespace gcd_1987_2025_l2101_210193

theorem gcd_1987_2025 : Nat.gcd 1987 2025 = 1 := by
  sorry

end gcd_1987_2025_l2101_210193


namespace mary_money_left_l2101_210166

def initial_amount : Float := 150
def game_cost : Float := 60
def discount_percent : Float := 15 / 100
def remaining_percent_for_goggles : Float := 20 / 100
def tax_on_goggles : Float := 8 / 100

def money_left_after_shopping_trip (initial_amount : Float) (game_cost : Float) (discount_percent : Float) (remaining_percent_for_goggles : Float) (tax_on_goggles : Float) : Float :=
  let discount := game_cost * discount_percent
  let discounted_price := game_cost - discount
  let remainder_after_game := initial_amount - discounted_price
  let goggles_cost_before_tax := remainder_after_game * remaining_percent_for_goggles
  let tax := goggles_cost_before_tax * tax_on_goggles
  let final_goggles_cost := goggles_cost_before_tax + tax
  let remainder_after_goggles := remainder_after_game - final_goggles_cost
  remainder_after_goggles

#eval money_left_after_shopping_trip initial_amount game_cost discount_percent remaining_percent_for_goggles tax_on_goggles -- expected: 77.62

theorem mary_money_left (initial_amount game_cost discount_percent remaining_percent_for_goggles tax_on_goggles : Float) : 
  money_left_after_shopping_trip initial_amount game_cost discount_percent remaining_percent_for_goggles tax_on_goggles = 77.62 :=
by sorry

end mary_money_left_l2101_210166


namespace seventh_grade_caps_collection_l2101_210135

theorem seventh_grade_caps_collection (A B C : ℕ) 
  (h1 : A = (B + C) / 2) 
  (h2 : B = (A + C) / 3)
  (h3 : C = 150) : A + B + C = 360 := 
by 
  sorry

end seventh_grade_caps_collection_l2101_210135


namespace margaret_mean_score_l2101_210197

noncomputable def cyprian_scores : List ℕ := [82, 85, 89, 91, 95, 97]
noncomputable def cyprian_mean : ℕ := 88

theorem margaret_mean_score :
  let total_sum := List.sum cyprian_scores
  let cyprian_sum := cyprian_mean * 3
  let margaret_sum := total_sum - cyprian_sum
  let margaret_mean := (margaret_sum : ℚ) / 3
  margaret_mean = 91.66666666666667 := 
by 
  -- Definitions used in conditions, skipping steps.
  sorry

end margaret_mean_score_l2101_210197


namespace house_number_digits_cost_l2101_210120

/-
The constants represent:
- cost_1: the cost of 1 unit (1000 rubles)
- cost_12: the cost of 12 units (2000 rubles)
- cost_512: the cost of 512 units (3000 rubles)
- P: the cost per digit of a house number (1000 rubles)
- n: the number of digits in a house number
- The goal is to prove that the cost for 1, 12, and 512 units follows the pattern described
-/

theorem house_number_digits_cost :
  ∃ (P : ℕ),
    (P = 1000) ∧
    (∃ (cost_1 cost_12 cost_512 : ℕ),
      cost_1 = 1000 ∧
      cost_12 = 2000 ∧
      cost_512 = 3000 ∧
      (∃ n1 n2 n3 : ℕ,
        n1 = 1 ∧
        n2 = 2 ∧
        n3 = 3 ∧
        cost_1 = P * n1 ∧
        cost_12 = P * n2 ∧
        cost_512 = P * n3)) :=
by
  sorry

end house_number_digits_cost_l2101_210120


namespace bono_jelly_beans_l2101_210156

variable (t A B C : ℤ)

theorem bono_jelly_beans (h₁ : A + B = 6 * t + 3) 
                         (h₂ : A + C = 4 * t + 5) 
                         (h₃ : B + C = 6 * t) : 
                         B = 4 * t - 1 := by
  sorry

end bono_jelly_beans_l2101_210156


namespace students_passing_course_l2101_210137

theorem students_passing_course :
  let students_three_years_ago := 200
  let increase_factor := 1.5
  let students_two_years_ago := students_three_years_ago * increase_factor
  let students_last_year := students_two_years_ago * increase_factor
  let students_this_year := students_last_year * increase_factor
  students_this_year = 675 :=
by
  sorry

end students_passing_course_l2101_210137


namespace four_digit_positive_integers_count_l2101_210157

def first_two_digit_choices : Finset ℕ := {2, 3, 6}
def last_two_digit_choices : Finset ℕ := {3, 7, 9}

theorem four_digit_positive_integers_count :
  (first_two_digit_choices.card * first_two_digit_choices.card) *
  (last_two_digit_choices.card * (last_two_digit_choices.card - 1)) = 54 := by
sorry

end four_digit_positive_integers_count_l2101_210157


namespace sum_first_11_terms_l2101_210145

variable {a : ℕ → ℕ} -- a is the arithmetic sequence

-- Condition: a_4 + a_8 = 26
axiom condition : a 4 + a 8 = 26

-- Definition of arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

-- Definition of the sum of the first 11 terms
def S_11 (a : ℕ → ℕ) : ℕ := (11 * (a 1 + a 11)) / 2

-- The proof problem statement
theorem sum_first_11_terms (h : is_arithmetic_sequence a) : S_11 a = 143 := 
by 
  sorry

end sum_first_11_terms_l2101_210145


namespace find_angle_A_correct_l2101_210105

noncomputable def find_angle_A (BC AB angleC : ℝ) : ℝ :=
if BC = 3 ∧ AB = Real.sqrt 6 ∧ angleC = Real.pi / 4 then
  Real.pi / 3
else
  sorry

theorem find_angle_A_correct : find_angle_A 3 (Real.sqrt 6) (Real.pi / 4) = Real.pi / 3 :=
by
  -- proof goes here
  sorry

end find_angle_A_correct_l2101_210105


namespace value_of_first_equation_l2101_210133

theorem value_of_first_equation (x y a : ℝ) 
  (h₁ : 2 * x + y = a) 
  (h₂ : x + 2 * y = 10) 
  (h₃ : (x + y) / 3 = 4) : 
  a = 12 :=
by 
  sorry

end value_of_first_equation_l2101_210133


namespace max_fm_n_l2101_210100

noncomputable def ln (x : ℝ) : ℝ := Real.log x

def g (m n x : ℝ) : ℝ := (2 * m + 3) * x + n

def condition_f_g (m n : ℝ) : Prop := ∀ x > 0, ln x ≤ g m n x

def f (m : ℝ) : ℝ := 2 * m + 3

theorem max_fm_n (m n : ℝ) (h : condition_f_g m n) : (f m) * n ≤ 1 / Real.exp 2 := sorry

end max_fm_n_l2101_210100


namespace compute_star_l2101_210112

def star (x y : ℕ) := 4 * x + 6 * y

theorem compute_star : star 3 4 = 36 := 
by
  sorry

end compute_star_l2101_210112


namespace triangle_right_angle_l2101_210154

theorem triangle_right_angle (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : γ = α + β) : γ = 90 :=
by
  sorry

end triangle_right_angle_l2101_210154


namespace spade_5_7_8_l2101_210131

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_5_7_8 : spade 5 (spade 7 8) = -200 :=
by
  sorry

end spade_5_7_8_l2101_210131


namespace not_divisible_l2101_210152

theorem not_divisible (n k : ℕ) : ¬ (5 ^ n + 1) ∣ (5 ^ k - 1) :=
sorry

end not_divisible_l2101_210152


namespace problem_l2101_210136

open Real

noncomputable def f (x : ℝ) : ℝ := log x / log 2

theorem problem (f : ℝ → ℝ) (h : ∀ (x y : ℝ), f (x * y) = f x + f y) : 
  (∀ x : ℝ, f x = log x / log 2) :=
sorry

end problem_l2101_210136


namespace boat_speed_still_water_l2101_210155

theorem boat_speed_still_water (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 16) (h2 : upstream_speed = 9) : 
  (downstream_speed + upstream_speed) / 2 = 12.5 := 
by
  -- conditions explicitly stated above
  sorry

end boat_speed_still_water_l2101_210155


namespace bernie_postcards_l2101_210121

theorem bernie_postcards :
  let initial_postcards := 18
  let price_sell := 15
  let price_buy := 5
  let sold_postcards := initial_postcards / 2
  let earned_money := sold_postcards * price_sell
  let bought_postcards := earned_money / price_buy
  let remaining_postcards := initial_postcards - sold_postcards
  let final_postcards := remaining_postcards + bought_postcards
  final_postcards = 36 := by sorry

end bernie_postcards_l2101_210121


namespace greatest_product_of_two_integers_with_sum_300_l2101_210190

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l2101_210190


namespace arithmetic_sequence_a10_l2101_210169

variable {a : ℕ → ℝ}

-- Given the sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, (n < m) → (a (m + 1) - a m = a (n + 1) - a n)

-- Conditions
theorem arithmetic_sequence_a10 (h_arith : is_arithmetic_sequence a) 
                                (h1 : a 6 + a 8 = 16)
                                (h2 : a 4 = 1) :
  a 10 = 15 :=
sorry

end arithmetic_sequence_a10_l2101_210169


namespace exponential_comparison_l2101_210116

theorem exponential_comparison (a b c : ℝ) (h₁ : a = 0.5^((1:ℝ)/2))
                                          (h₂ : b = 0.5^((1:ℝ)/3))
                                          (h₃ : c = 0.5^((1:ℝ)/4)) : 
  a < b ∧ b < c := by
  sorry

end exponential_comparison_l2101_210116


namespace employee_n_salary_l2101_210195

theorem employee_n_salary (m n : ℝ) (h1 : m = 1.2 * n) (h2 : m + n = 594) :
  n = 270 :=
sorry

end employee_n_salary_l2101_210195


namespace sum_tens_ones_digit_of_7_pow_11_l2101_210196

/--
The sum of the tens digit and the ones digit of (3+4)^{11} is 7.
-/
theorem sum_tens_ones_digit_of_7_pow_11 : 
  let number := (3 + 4)^11
  let tens_digit := (number / 10) % 10
  let ones_digit := number % 10
  tens_digit + ones_digit = 7 :=
by
  sorry

end sum_tens_ones_digit_of_7_pow_11_l2101_210196


namespace solve_and_sum_solutions_l2101_210111

theorem solve_and_sum_solutions :
  (∀ x : ℝ, x^2 > 9 → 
  (∃ S : ℝ, (∀ x : ℝ, (x / 3 + x / Real.sqrt (x^2 - 9) = 35 / 12) → S = 8.75))) :=
sorry

end solve_and_sum_solutions_l2101_210111


namespace range_of_f_l2101_210199

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + 4 * Real.sin x + 6

theorem range_of_f :
  ∀ (x : ℝ), Real.sin x ≠ 2 → 
  (1 ≤ f x ∧ f x ≤ 11) :=
by 
  sorry

end range_of_f_l2101_210199


namespace voltage_relationship_l2101_210150

variables (x y z : ℝ) -- Coordinates representing positions on the lines
variables (I R U : ℝ) -- Representing current, resistance, and voltage respectively

-- Conditions translated into Lean
def I_def := I = 10^x
def R_def := R = 10^(-2 * y)
def U_def := U = 10^(-z)
def coord_relation := x + z = 2 * y

-- The final theorem to prove V = I * R under given conditions
theorem voltage_relationship : I = 10^x → R = 10^(-2 * y) → U = 10^(-z) → (x + z = 2 * y) → U = I * R :=
by 
  intros hI hR hU hXYZ
  sorry

end voltage_relationship_l2101_210150


namespace simple_interest_rate_l2101_210172

def principal : ℕ := 600
def amount : ℕ := 950
def time : ℕ := 5
def expected_rate : ℚ := 11.67

theorem simple_interest_rate (P A T : ℕ) (R : ℚ) :
  P = principal → A = amount → T = time → R = expected_rate →
  (A = P + P * R * T / 100) :=
by
  intros hP hA hT hR
  sorry

end simple_interest_rate_l2101_210172


namespace range_of_m_l2101_210108

theorem range_of_m (m x : ℝ) : (m-1 < x ∧ x < m+1) → (1/3 < x ∧ x < 1/2) → (-1/2 ≤ m ∧ m ≤ 4/3) :=
by
  intros h1 h2
  have h3 : 1/3 < m + 1 := by sorry
  have h4 : m - 1 < 1/2 := by sorry
  have h5 : -1/2 ≤ m := by sorry
  have h6 : m ≤ 4/3 := by sorry
  exact ⟨h5, h6⟩

end range_of_m_l2101_210108


namespace sum_of_odd_integers_21_to_51_l2101_210125

noncomputable def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

noncomputable def sum_arithmetic_seq (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_of_odd_integers_21_to_51 : sum_arithmetic_seq 21 2 51 = 576 := by
  sorry

end sum_of_odd_integers_21_to_51_l2101_210125


namespace carlos_payment_l2101_210114

theorem carlos_payment (A B C : ℝ) (hB_lt_A : B < A) (hB_lt_C : B < C) :
    B + (0.35 * (A + B + C) - B) = 0.35 * A - 0.65 * B + 0.35 * C :=
by sorry

end carlos_payment_l2101_210114


namespace billiard_expected_reflections_l2101_210139

noncomputable def expected_reflections : ℝ :=
  (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

theorem billiard_expected_reflections :
  expected_reflections = (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4)) :=
by
  sorry

end billiard_expected_reflections_l2101_210139


namespace initial_elephants_l2101_210148

theorem initial_elephants (E : ℕ) :
  (E + 35 + 135 + 125 = 315) → (5 * 35 / 7 = 25) → (5 * 25 = 125) → (135 = 125 + 10) →
  E = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_elephants_l2101_210148


namespace nancy_packs_of_crayons_l2101_210124

def total_crayons : ℕ := 615
def crayons_per_pack : ℕ := 15

theorem nancy_packs_of_crayons : total_crayons / crayons_per_pack = 41 :=
by
  sorry

end nancy_packs_of_crayons_l2101_210124


namespace product_gcd_lcm_l2101_210160

-- Define the numbers
def a : ℕ := 24
def b : ℕ := 60

-- Define the gcd and lcm
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b

-- Statement to prove: the product of gcd and lcm of 24 and 60 equals 1440
theorem product_gcd_lcm : gcd_ab * lcm_ab = 1440 := by
  -- gcd_ab = 12
  -- lcm_ab = 120
  -- Thus, 12 * 120 = 1440
  sorry

end product_gcd_lcm_l2101_210160


namespace seq_max_value_l2101_210164

theorem seq_max_value {a_n : ℕ → ℝ} (h : ∀ n, a_n n = (↑n + 2) * (3 / 4) ^ n) : 
  ∃ n, a_n n = max (a_n 1) (a_n 2) → (n = 1 ∨ n = 2) :=
by 
  sorry

end seq_max_value_l2101_210164


namespace exam_items_count_l2101_210138

theorem exam_items_count (x : ℝ) (hLiza : Liza_correct = 0.9 * x) (hRoseCorrect : Rose_correct = 0.9 * x + 2) (hRoseTotal : Rose_total = x) (hRoseIncorrect : Rose_incorrect = x - (0.9 * x + 2) ):
    Liza_correct + Rose_incorrect = Rose_total :=
by
    sorry

end exam_items_count_l2101_210138


namespace greg_sarah_apples_l2101_210115

-- Definitions and Conditions
variable {G : ℕ}
variable (H0 : 2 * G + 2 * G + (2 * G - 5) = 49)

-- Statement of the problem
theorem greg_sarah_apples : 
  2 * G = 18 :=
by
  sorry

end greg_sarah_apples_l2101_210115


namespace vacation_cost_l2101_210128

theorem vacation_cost (C : ℝ)
  (h1 : C / 5 - C / 8 = 60) :
  C = 800 :=
sorry

end vacation_cost_l2101_210128
