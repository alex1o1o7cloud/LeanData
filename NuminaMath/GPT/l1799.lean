import Mathlib

namespace circle_chords_integer_lengths_l1799_179984

theorem circle_chords_integer_lengths (P O : ℝ) (d r : ℝ) (n : ℕ) : 
  dist P O = d → r = 20 → d = 12 → n = 9 := by
  sorry

end circle_chords_integer_lengths_l1799_179984


namespace percent_of_decimal_l1799_179932

theorem percent_of_decimal : (3 / 8 / 100) * 240 = 0.9 :=
by
  sorry

end percent_of_decimal_l1799_179932


namespace supplement_of_angle_l1799_179981

theorem supplement_of_angle (A : ℝ) (h : 90 - A = A - 18) : 180 - A = 126 := by
    sorry

end supplement_of_angle_l1799_179981


namespace longest_side_of_triangle_l1799_179943

theorem longest_side_of_triangle (y : ℝ) 
  (side1 : ℝ := 8) (side2 : ℝ := y + 5) (side3 : ℝ := 3 * y + 2)
  (h_perimeter : side1 + side2 + side3 = 47) :
  max side1 (max side2 side3) = 26 :=
sorry

end longest_side_of_triangle_l1799_179943


namespace xy_in_A_l1799_179991

def A : Set ℤ :=
  {z | ∃ (a b k n : ℤ), z = a^2 + k * a * b + n * b^2}

theorem xy_in_A (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := sorry

end xy_in_A_l1799_179991


namespace compute_a1d1_a2d2_a3d3_l1799_179970

noncomputable def polynomial_equation (a1 a2 a3 d1 d2 d3: ℝ) : Prop :=
  ∀ x : ℝ, (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)

theorem compute_a1d1_a2d2_a3d3 (a1 a2 a3 d1 d2 d3 : ℝ) (h : polynomial_equation a1 a2 a3 d1 d2 d3) : 
  a1 * d1 + a2 * d2 + a3 * d3 = 1 :=
  sorry

end compute_a1d1_a2d2_a3d3_l1799_179970


namespace geometric_sequence_fourth_term_l1799_179928

theorem geometric_sequence_fourth_term (x : ℝ) (r : ℝ) (h1 : (3 * x + 3)^2 = x * (6 * x + 6)) 
(h2 : r = (3 * x + 3) / x) :
  (6 * x + 6) * r = -24 :=
by {
  -- Definitions of x, r and condition h1, h2 are given.
  -- Conclusion must follow that the fourth term is -24.
  sorry
}

end geometric_sequence_fourth_term_l1799_179928


namespace each_member_score_l1799_179925

def total_members : ℝ := 5.0
def members_didnt_show_up : ℝ := 2.0
def total_points_by_showed_up_members : ℝ := 6.0

theorem each_member_score
  (h1 : total_members - members_didnt_show_up = 3.0)
  (h2 : total_points_by_showed_up_members = 6.0) :
  total_points_by_showed_up_members / (total_members - members_didnt_show_up) = 2.0 :=
sorry

end each_member_score_l1799_179925


namespace total_amount_is_2500_l1799_179951

noncomputable def total_amount_divided (P1 : ℝ) (annual_income : ℝ) : ℝ :=
  let P2 := 2500 - P1
  let income_from_P1 := (5 / 100) * P1
  let income_from_P2 := (6 / 100) * P2
  income_from_P1 + income_from_P2

theorem total_amount_is_2500 : 
  (total_amount_divided 2000 130) = 130 :=
by
  sorry

end total_amount_is_2500_l1799_179951


namespace line_does_not_pass_third_quadrant_l1799_179914

theorem line_does_not_pass_third_quadrant (a b c x y : ℝ) (h_ac : a * c < 0) (h_bc : b * c < 0) :
  ¬(x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0) :=
sorry

end line_does_not_pass_third_quadrant_l1799_179914


namespace find_b_l1799_179902

theorem find_b (a b : ℝ) (h : ∀ x, 2 * x^2 - a * x + 4 < 0 ↔ 1 < x ∧ x < b) : b = 2 :=
sorry

end find_b_l1799_179902


namespace seating_arrangements_l1799_179994

theorem seating_arrangements (p : Fin 5 → Fin 5 → Prop) :
  (∃! i j : Fin 5, p i j ∧ i = j) →
  (∃! i j : Fin 5, p i j ∧ i ≠ j) →
  ∃ ways : ℕ,
  ways = 20 :=
by
  sorry

end seating_arrangements_l1799_179994


namespace convex_polygon_quadrilateral_division_l1799_179960

open Nat

theorem convex_polygon_quadrilateral_division (n : ℕ) : ℕ :=
  if h : n > 0 then
    1 / (2 * n - 1) * (Nat.choose (3 * n - 3) (n - 1))
  else
    0

end convex_polygon_quadrilateral_division_l1799_179960


namespace greatest_difference_l1799_179992

def difference_marbles : Nat :=
  let A_diff := 4 - 2
  let B_diff := 6 - 1
  let C_diff := 9 - 3
  max (max A_diff B_diff) C_diff

theorem greatest_difference :
  difference_marbles = 6 :=
by
  sorry

end greatest_difference_l1799_179992


namespace number_of_nonnegative_solutions_l1799_179933

theorem number_of_nonnegative_solutions : ∃ (count : ℕ), count = 1 ∧ ∀ x : ℝ, x^2 + 9 * x = 0 → x ≥ 0 → x = 0 := by
  sorry

end number_of_nonnegative_solutions_l1799_179933


namespace range_of_sum_of_two_l1799_179940

theorem range_of_sum_of_two (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a^2 + b^2 + c^2 = 1) : 
  0 ≤ a + b ∧ a + b ≤ 4 / 3 :=
by
  -- Proof goes here.
  sorry

end range_of_sum_of_two_l1799_179940


namespace investor_receives_7260_l1799_179987

-- Define the initial conditions
def principal : ℝ := 6000
def annual_rate : ℝ := 0.10
def compoundings_per_year : ℝ := 1
def years : ℝ := 2

-- Define the compound interest formula
noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- State the theorem: The investor will receive $7260 after two years
theorem investor_receives_7260 : compound_interest principal annual_rate compoundings_per_year years = 7260 := by
  sorry

end investor_receives_7260_l1799_179987


namespace minimum_value_of_3x_plus_4y_l1799_179952

theorem minimum_value_of_3x_plus_4y :
  ∀ (x y : ℝ), 0 < x → 0 < y → x + 3 * y = 5 * x * y → (3 * x + 4 * y) ≥ 24 / 5 :=
by
  sorry

end minimum_value_of_3x_plus_4y_l1799_179952


namespace probability_of_interval_l1799_179909

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l1799_179909


namespace cadence_worked_old_company_l1799_179963

theorem cadence_worked_old_company (y : ℕ) (h1 : (426000 : ℝ) = 
    5000 * 12 * y + 6000 * 12 * (y + 5 / 12)) :
    y = 3 := by
    sorry

end cadence_worked_old_company_l1799_179963


namespace at_least_one_non_zero_l1799_179985

theorem at_least_one_non_zero (a b : ℝ) : a^2 + b^2 > 0 ↔ (a ≠ 0 ∨ b ≠ 0) :=
by sorry

end at_least_one_non_zero_l1799_179985


namespace quadratic_inequality_solution_set_l1799_179976

theorem quadratic_inequality_solution_set (x : ℝ) :
  (x^2 - 3 * x - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 4) :=
sorry

end quadratic_inequality_solution_set_l1799_179976


namespace vacation_cost_division_l1799_179910

theorem vacation_cost_division (n : ℕ) (total_cost : ℕ) 
  (cost_difference : ℕ)
  (cost_per_person_5 : ℕ) :
  total_cost = 1000 → 
  cost_difference = 50 → 
  cost_per_person_5 = total_cost / 5 →
  (total_cost / n) = cost_per_person_5 + cost_difference → 
  n = 4 := 
by
  intros h1 h2 h3 h4
  sorry

end vacation_cost_division_l1799_179910


namespace part1_part2_l1799_179906

def is_sum_solution_equation (a b x : ℝ) : Prop :=
  x = b + a

def part1_statement := ¬ is_sum_solution_equation 3 4.5 (4.5 / 3)

def part2_statement (m : ℝ) : Prop :=
  is_sum_solution_equation 5 (m + 1) (m + 6) → m = (-29 / 4)

theorem part1 : part1_statement :=
by 
  -- Proof here
  sorry

theorem part2 (m : ℝ) : part2_statement m :=
by 
  -- Proof here
  sorry

end part1_part2_l1799_179906


namespace quadratic_no_real_roots_l1799_179946

theorem quadratic_no_real_roots (c : ℝ) (h : c > 1) : ∀ x : ℝ, x^2 + 2 * x + c ≠ 0 :=
by
  sorry

end quadratic_no_real_roots_l1799_179946


namespace range_of_a_l1799_179926

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 1) * x ^ 2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3 : ℝ) 1 :=
by
  sorry

end range_of_a_l1799_179926


namespace total_cost_l1799_179915

variables (p e n : ℕ) -- represent the costs of pencil, eraser, and notebook in cents

-- Given conditions
def conditions : Prop :=
  9 * p + 7 * e + 4 * n = 220 ∧
  p > n ∧ n > e ∧ e > 0

-- Prove the total cost
theorem total_cost (h : conditions p e n) : p + n + e = 26 :=
sorry

end total_cost_l1799_179915


namespace problem_statement_l1799_179904

-- Define the problem context
variables {a b c d : ℝ}

-- Define the conditions
def unit_square_condition (a b c d : ℝ) : Prop :=
  a^2 + b^2 + c^2 + d^2 ≥ 2 ∧ a^2 + b^2 + c^2 + d^2 ≤ 4 ∧ 
  a + b + c + d ≥ 2 * Real.sqrt 2 ∧ a + b + c + d ≤ 4

-- Provide the main theorem
theorem problem_statement (h : unit_square_condition a b c d) : 
  2 ≤ a^2 + b^2 + c^2 + d^2 ∧ a^2 + b^2 + c^2 + d^2 ≤ 4 ∧ 
  2 * Real.sqrt 2 ≤ a + b + c + d ∧ a + b + c + d ≤ 4 :=
  by 
  { sorry }  -- Proof to be completed

end problem_statement_l1799_179904


namespace tony_rope_length_l1799_179923

-- Define the lengths of the individual ropes.
def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]

-- Define the total number of ropes Tony has.
def num_ropes : ℕ := rope_lengths.length

-- Calculate the total length of ropes before tying them together.
def total_length_before_tying : ℝ := rope_lengths.sum

-- Define the length lost per knot.
def length_lost_per_knot : ℝ := 1.2

-- Calculate the total number of knots needed.
def num_knots : ℕ := num_ropes - 1

-- Calculate the total length lost due to knots.
def total_length_lost : ℝ := num_knots * length_lost_per_knot

-- Calculate the total length of the rope after tying them all together.
def total_length_after_tying : ℝ := total_length_before_tying - total_length_lost

-- The theorem we want to prove.
theorem tony_rope_length : total_length_after_tying = 35 :=
by sorry

end tony_rope_length_l1799_179923


namespace prob_not_green_is_six_over_eleven_l1799_179974

-- Define the odds for pulling a green marble
def odds_green : ℕ × ℕ := (5, 6)

-- Define the total number of events as the sum of both parts of the odds
def total_events : ℕ := odds_green.1 + odds_green.2

-- Define the probability of not pulling a green marble
def probability_not_green : ℚ := odds_green.2 / total_events

-- State the theorem
theorem prob_not_green_is_six_over_eleven : probability_not_green = 6 / 11 := by
  -- Proof goes here
  sorry

end prob_not_green_is_six_over_eleven_l1799_179974


namespace gum_pieces_bought_correct_l1799_179937

-- Define initial number of gum pieces
def initial_gum_pieces : ℕ := 10

-- Define number of friends Adrianna gave gum to
def friends_given_gum : ℕ := 11

-- Define the number of pieces Adrianna has left
def remaining_gum_pieces : ℕ := 2

-- Define a function to calculate the number of gum pieces Adrianna bought at the store
def gum_pieces_bought (initial_gum : ℕ) (given_gum : ℕ) (remaining_gum : ℕ) : ℕ :=
  (given_gum + remaining_gum) - initial_gum

-- Now state the theorem to prove the number of pieces bought is 3
theorem gum_pieces_bought_correct : 
  gum_pieces_bought initial_gum_pieces friends_given_gum remaining_gum_pieces = 3 :=
by
  sorry

end gum_pieces_bought_correct_l1799_179937


namespace pairwise_products_same_digit_l1799_179956

theorem pairwise_products_same_digit
  (a b c : ℕ)
  (h_ab : a % 10 ≠ b % 10)
  (h_ac : a % 10 ≠ c % 10)
  (h_bc : b % 10 ≠ c % 10)
  : (a * b % 10 = a * c % 10) ∧ (a * b % 10 = b * c % 10) :=
  sorry

end pairwise_products_same_digit_l1799_179956


namespace problems_per_page_l1799_179948

-- Define the initial conditions
def total_problems : ℕ := 101
def finished_problems : ℕ := 47
def remaining_pages : ℕ := 6

-- State the theorem
theorem problems_per_page : 54 / remaining_pages = 9 :=
by
  -- Sorry is used to ignore the proof step
  sorry

end problems_per_page_l1799_179948


namespace correct_options_l1799_179921

theorem correct_options :
  (1 + Real.tan 1) * (1 + Real.tan 44) = 2 ∧
  ¬((1 / Real.sin 10) - (Real.sqrt 3 / Real.cos 10) = 2) ∧
  (3 - Real.sin 70) / (2 - (Real.cos 10) ^ 2) = 2 ∧
  ¬(Real.tan 70 * Real.cos 10 * (Real.sqrt 3 * Real.tan 20 - 1) = 2) :=
sorry

end correct_options_l1799_179921


namespace train_speed_l1799_179977

theorem train_speed (d t s : ℝ) (h1 : d = 320) (h2 : t = 6) (h3 : s = 53.33) :
  s = d / t :=
by
  rw [h1, h2]
  sorry

end train_speed_l1799_179977


namespace toy_cost_l1799_179958

-- Conditions
def initial_amount : ℕ := 3
def allowance : ℕ := 7
def total_amount : ℕ := initial_amount + allowance
def number_of_toys : ℕ := 2

-- Question and Proof
theorem toy_cost :
  total_amount / number_of_toys = 5 :=
by
  sorry

end toy_cost_l1799_179958


namespace find_a_plus_c_l1799_179903

theorem find_a_plus_c (a b c d : ℝ)
  (h₁ : -(3 - a) ^ 2 + b = 6) (h₂ : (3 - c) ^ 2 + d = 6)
  (h₃ : -(7 - a) ^ 2 + b = 2) (h₄ : (7 - c) ^ 2 + d = 2) :
  a + c = 10 := sorry

end find_a_plus_c_l1799_179903


namespace magic_king_episodes_proof_l1799_179944

-- Let's state the condition in terms of the number of seasons and episodes:
def total_episodes (seasons: ℕ) (episodes_first_half: ℕ) (episodes_second_half: ℕ) : ℕ :=
  (seasons / 2) * episodes_first_half + (seasons / 2) * episodes_second_half

-- Define the conditions for the "Magic King" show
def magic_king_total_episodes : ℕ :=
  total_episodes 10 20 25

-- The statement of the problem - to prove that the total episodes is 225
theorem magic_king_episodes_proof : magic_king_total_episodes = 225 :=
by
  sorry

end magic_king_episodes_proof_l1799_179944


namespace compute_b_l1799_179968

theorem compute_b (x y b : ℚ) (h1 : 5 * x - 2 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (hy : y = 3) :
  b = 13 / 2 :=
sorry

end compute_b_l1799_179968


namespace interest_rate_per_annum_l1799_179973

def principal : ℝ := 8945
def simple_interest : ℝ := 4025.25
def time : ℕ := 5

theorem interest_rate_per_annum : (simple_interest * 100) / (principal * time) = 9 := by
  sorry

end interest_rate_per_annum_l1799_179973


namespace david_money_l1799_179917

theorem david_money (S : ℝ) (h_initial : 1500 - S = S - 500) : 1500 - S = 500 :=
by
  sorry

end david_money_l1799_179917


namespace value_of_f_g_3_l1799_179993

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3*x^2 - 2*x + 1

theorem value_of_f_g_3 : f (g 3) = 2134 :=
by 
  sorry

end value_of_f_g_3_l1799_179993


namespace percentage_deficit_l1799_179990

theorem percentage_deficit
  (L W : ℝ)
  (h1 : ∃(x : ℝ), 1.10 * L * (W * (1 - x / 100)) = L * W * 1.045) :
  ∃ (x : ℝ), x = 5 :=
by
  sorry

end percentage_deficit_l1799_179990


namespace common_ratio_of_geometric_sequence_l1799_179912

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n+1) = a n * q) → 
  (a 1 + a 5 = 17) → 
  (a 2 * a 4 = 16) → 
  (∀ i j, i < j → a i < a j) → 
  q = 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l1799_179912


namespace range_of_f_l1799_179975

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 - (Real.sin x) * (Real.cos x) + (Real.cos x) ^ 4

theorem range_of_f : Set.Icc 0 (9 / 8) = Set.range f := 
by
  sorry

end range_of_f_l1799_179975


namespace g_range_excludes_zero_l1799_179957

noncomputable def g (x : ℝ) : ℤ :=
if x > -1 then ⌈1 / (x + 1)⌉
else ⌊1 / (x + 1)⌋

theorem g_range_excludes_zero : ¬ ∃ x : ℝ, g x = 0 := 
by 
  sorry

end g_range_excludes_zero_l1799_179957


namespace range_of_a_l1799_179934

variable (x a : ℝ)

-- Definitions of conditions as hypotheses
def condition_p (x : ℝ) := |x + 1| ≤ 2
def condition_q (x a : ℝ) := x ≤ a
def sufficient_not_necessary (p q : Prop) := p → q ∧ ¬(q → p)

-- The theorem statement
theorem range_of_a : sufficient_not_necessary (condition_p x) (condition_q x a) → 1 ≤ a ∧ ∀ b, b < 1 → sufficient_not_necessary (condition_p x) (condition_q x b) → false :=
by
  intro h
  sorry

end range_of_a_l1799_179934


namespace sequence_n_l1799_179950

theorem sequence_n (a : ℕ → ℕ) (h : ∀ n : ℕ, 0 < n → (n^2 + 1) * a n = n * (a (n^2) + 1)) :
  ∀ n : ℕ, 0 < n → a n = n := 
by
  sorry

end sequence_n_l1799_179950


namespace hall_ratio_l1799_179907

theorem hall_ratio (w l : ℕ) (h1 : w * l = 450) (h2 : l - w = 15) : w / l = 1 / 2 :=
by sorry

end hall_ratio_l1799_179907


namespace luke_hotdogs_ratio_l1799_179972

-- Definitions
def hotdogs_per_sister : ℕ := 2
def total_sisters_hotdogs : ℕ := 2 * 2 -- Ella and Emma together
def hunter_hotdogs : ℕ := 6 -- 1.5 times the total of sisters' hotdogs
def total_hotdogs : ℕ := 14

-- Ratio proof problem statement
theorem luke_hotdogs_ratio :
  ∃ x : ℕ, total_hotdogs = total_sisters_hotdogs + 4 * x + hunter_hotdogs ∧ 
    (4 * x = 2 * 1 ∧ x = 1) := 
by 
  sorry

end luke_hotdogs_ratio_l1799_179972


namespace verify_euler_relation_for_transformed_cube_l1799_179935

def euler_relation_for_transformed_cube : Prop :=
  let V := 12
  let A := 24
  let F := 14
  V + F = A + 2

theorem verify_euler_relation_for_transformed_cube :
  euler_relation_for_transformed_cube :=
by
  sorry

end verify_euler_relation_for_transformed_cube_l1799_179935


namespace no_cubic_solution_l1799_179908

theorem no_cubic_solution (t : ℤ) : ¬ ∃ k : ℤ, (7 * t + 3 = k ^ 3) := by
  sorry

end no_cubic_solution_l1799_179908


namespace compound_interest_correct_l1799_179979
noncomputable def compound_interest_proof : Prop :=
  let si := 55
  let r := 5
  let t := 2
  let p := si * 100 / (r * t)
  let ci := p * ((1 + r / 100)^t - 1)
  ci = 56.375

theorem compound_interest_correct : compound_interest_proof :=
by {
  sorry
}

end compound_interest_correct_l1799_179979


namespace probability_palindrome_divisible_by_11_l1799_179924

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 = d5 ∧ d2 = d4

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def count_all_palindromes : ℕ :=
  9 * 10 * 10

def count_palindromes_div_by_11 : ℕ :=
  9 * 10

theorem probability_palindrome_divisible_by_11 :
  (count_palindromes_div_by_11 : ℚ) / count_all_palindromes = 1 / 10 :=
by sorry

end probability_palindrome_divisible_by_11_l1799_179924


namespace find_other_person_weight_l1799_179997

noncomputable def other_person_weight (n avg new_avg W1 : ℕ) : ℕ :=
  let total_initial := n * avg
  let new_n := n + 2
  let total_new := new_n * new_avg
  total_new - total_initial - W1

theorem find_other_person_weight:
  other_person_weight 23 48 51 78 = 93 := by
  sorry

end find_other_person_weight_l1799_179997


namespace ball_hits_ground_at_time_l1799_179930

-- Given definitions from the conditions
def y (t : ℝ) : ℝ := -4.9 * t^2 + 5 * t + 8

-- Statement of the problem: proving the time t when the ball hits the ground
theorem ball_hits_ground_at_time :
  ∃ t : ℝ, y t = 0 ∧ t = 1.887 := 
sorry

end ball_hits_ground_at_time_l1799_179930


namespace gcd_of_A_and_B_l1799_179919

theorem gcd_of_A_and_B (A B : ℕ) (h_lcm : Nat.lcm A B = 120) (h_ratio : A * 4 = B * 3) : Nat.gcd A B = 10 :=
sorry

end gcd_of_A_and_B_l1799_179919


namespace total_weight_of_10_moles_l1799_179971

theorem total_weight_of_10_moles
  (molecular_weight : ℕ)
  (moles : ℕ)
  (h_molecular_weight : molecular_weight = 2670)
  (h_moles : moles = 10) :
  moles * molecular_weight = 26700 := by
  -- By substituting the values from the hypotheses:
  -- We will get:
  -- 10 * 2670 = 26700
  sorry

end total_weight_of_10_moles_l1799_179971


namespace area_is_rational_l1799_179996

-- Definitions of the vertices of the triangle
def point1 : (ℤ × ℤ) := (2, 3)
def point2 : (ℤ × ℤ) := (5, 7)
def point3 : (ℤ × ℤ) := (3, 4)

-- Define a function to calculate the area of a triangle given vertices with integer coordinates
def triangle_area (A B C: (ℤ × ℤ)) : ℚ :=
  abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2

-- Define the area of our specific triangle
noncomputable def area_of_triangle_with_given_vertices := triangle_area point1 point2 point3

-- Proof statement
theorem area_is_rational : ∃ (Q : ℚ), Q = area_of_triangle_with_given_vertices := 
sorry

end area_is_rational_l1799_179996


namespace negation_of_exists_l1799_179978

variable (a : ℝ)

theorem negation_of_exists (h : ¬ ∃ x : ℝ, x^2 + a * x + 1 < 0) : ∀ x : ℝ, x^2 + a * x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l1799_179978


namespace product_of_five_numbers_is_256_l1799_179916

def possible_numbers : Set ℕ := {1, 2, 4}

theorem product_of_five_numbers_is_256 
  (x1 x2 x3 x4 x5 : ℕ) 
  (h1 : x1 ∈ possible_numbers) 
  (h2 : x2 ∈ possible_numbers) 
  (h3 : x3 ∈ possible_numbers) 
  (h4 : x4 ∈ possible_numbers) 
  (h5 : x5 ∈ possible_numbers) : 
  x1 * x2 * x3 * x4 * x5 = 256 :=
sorry

end product_of_five_numbers_is_256_l1799_179916


namespace find_number_l1799_179969

-- Define the conditions
def condition (x : ℝ) : Prop := 0.65 * x = (4/5) * x - 21

-- Prove that given the condition, x is 140.
theorem find_number (x : ℝ) (h : condition x) : x = 140 := by
  sorry

end find_number_l1799_179969


namespace exists_two_digit_number_l1799_179939

theorem exists_two_digit_number :
  ∃ x y : ℕ, (1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9) ∧ (10 * x + y = (x + y) * (x - y)) ∧ (10 * x + y = 48) :=
by
  sorry

end exists_two_digit_number_l1799_179939


namespace unique_intersection_point_l1799_179999

def line1 (x y : ℝ) : Prop := 3 * x + 2 * y = 9
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y = 10
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = 1
def line5 (x y : ℝ) : Prop := x + y = 4

theorem unique_intersection_point :
  ∃! (p : ℝ × ℝ), 
     line1 p.1 p.2 ∧ 
     line2 p.1 p.2 ∧ 
     line3 p.1 ∧ 
     line4 p.2 ∧ 
     line5 p.1 p.2 :=
sorry

end unique_intersection_point_l1799_179999


namespace remainder_when_divided_by_3x_minus_6_l1799_179965

def polynomial (x : ℝ) : ℝ := 5 * x^8 - 3 * x^7 + 2 * x^6 - 9 * x^4 + 3 * x^3 - 7

def evaluate_at (f : ℝ → ℝ) (a : ℝ) : ℝ := f a

theorem remainder_when_divided_by_3x_minus_6 :
  evaluate_at polynomial 2 = 897 :=
by
  -- Compute this value manually or use automated tools
  sorry

end remainder_when_divided_by_3x_minus_6_l1799_179965


namespace consistent_values_for_a_l1799_179927

def eq1 (x a : ℚ) : Prop := 10 * x^2 + x - a - 11 = 0
def eq2 (x a : ℚ) : Prop := 4 * x^2 + (a + 4) * x - 3 * a - 8 = 0

theorem consistent_values_for_a : ∃ x, (eq1 x 0 ∧ eq2 x 0) ∨ (eq1 x (-2) ∧ eq2 x (-2)) ∨ (eq1 x (54) ∧ eq2 x (54)) :=
by
  sorry

end consistent_values_for_a_l1799_179927


namespace average_coins_collected_per_day_l1799_179938

noncomputable def average_coins (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (a + (a + (n - 1) * d)) / 2

theorem average_coins_collected_per_day :
  average_coins 10 5 7 = 25 := by
  sorry

end average_coins_collected_per_day_l1799_179938


namespace lecture_hall_rows_l1799_179959

-- We define the total number of seats
def total_seats (n : ℕ) : ℕ := n * (n + 11)

-- We state the problem with the given conditions
theorem lecture_hall_rows : 
  (400 ≤ total_seats n) ∧ (total_seats n ≤ 440) → n = 16 :=
by
  sorry

end lecture_hall_rows_l1799_179959


namespace ellipse_eccentricity_l1799_179941

noncomputable def eccentricity_of_ellipse (a c : ℝ) : ℝ :=
  c / a

theorem ellipse_eccentricity (F1 A : ℝ) (v : ℝ) (a c : ℝ)
  (h1 : 4 * a = 10 * (a - c))
  (h2 : F1 = 0 ∧ A = 0 ∧ v ≠ 0) :
  eccentricity_of_ellipse a c = 3 / 5 := by
sorry

end ellipse_eccentricity_l1799_179941


namespace quadratic_sum_roots_l1799_179954

-- We define the quadratic function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- The function f passes through points (r, k) and (s, k)
variables (a b c r s k : ℝ)
variable (ha : a ≠ 0)
variable (hr : f a b c r = k)
variable (hs : f a b c s = k)

-- What we want to prove
theorem quadratic_sum_roots :
  f a b c (r + s) = c :=
sorry

end quadratic_sum_roots_l1799_179954


namespace largest_five_digit_integer_with_conditions_l1799_179945

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digits_product (n : ℕ) : ℕ :=
  (n % 10) * ((n / 10) % 10) * ((n / 100) % 10) * ((n / 1000) % 10) * ((n / 10000) % 10)

def digits_sum (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10) + ((n / 10000) % 10)

theorem largest_five_digit_integer_with_conditions :
  ∃ n : ℕ, is_five_digit n ∧ digits_product n = 40320 ∧ digits_sum n < 35 ∧
  ∀ m : ℕ, is_five_digit m ∧ digits_product m = 40320 ∧ digits_sum m < 35 → n ≥ m :=
sorry

end largest_five_digit_integer_with_conditions_l1799_179945


namespace fruit_days_l1799_179982

/-
  Henry and his brother believe in the famous phrase, "An apple a day, keeps the doctor away." 
  Henry's sister, however, believes that "A banana a day makes the trouble fade away" 
  and their father thinks that "An orange a day will keep the weaknesses at bay." 
  A box of apples contains 14 apples, a box of bananas has 20 bananas, and a box of oranges contains 12 oranges. 

  If Henry and his brother eat 1 apple each a day, their sister consumes 2 bananas per day, 
  and their father eats 3 oranges per day, how many days can the family of four continue eating fruits 
  if they have 3 boxes of apples, 4 boxes of bananas, and 5 boxes of oranges? 

  However, due to seasonal changes, oranges are only available for the first 20 days. 
  Moreover, Henry's sister has decided to only eat bananas on days when the day of the month is an odd number. 
  Considering these constraints, determine the total number of days the family of four can continue eating their preferred fruits.
-/

def apples_per_box := 14
def bananas_per_box := 20
def oranges_per_box := 12

def apples_boxes := 3
def bananas_boxes := 4
def oranges_boxes := 5

def daily_apple_consumption := 2
def daily_banana_consumption := 2
def daily_orange_consumption := 3

def orange_availability_days := 20

def odd_days_in_month := 16

def total_number_of_days : ℕ :=
  let total_apples := apples_boxes * apples_per_box
  let total_bananas := bananas_boxes * bananas_per_box
  let total_oranges := oranges_boxes * oranges_per_box
  
  let days_with_apples := total_apples / daily_apple_consumption
  let days_with_bananas := (total_bananas / (odd_days_in_month * daily_banana_consumption)) * 30
  let days_with_oranges := if total_oranges / daily_orange_consumption > orange_availability_days then orange_availability_days else total_oranges / daily_orange_consumption
  min (min days_with_apples days_with_oranges) (days_with_bananas / 30 * 30)

theorem fruit_days : total_number_of_days = 20 := 
  sorry

end fruit_days_l1799_179982


namespace total_tickets_l1799_179986

-- Define the initial number of tickets Tate has.
def tate_initial_tickets : ℕ := 32

-- Define the number of tickets Tate buys additionally.
def additional_tickets : ℕ := 2

-- Define the total number of tickets Tate has after buying more.
def tate_total_tickets : ℕ := tate_initial_tickets + additional_tickets

-- Define the total number of tickets Peyton has.
def peyton_tickets : ℕ := tate_total_tickets / 2

-- State the theorem to prove the total number of tickets Tate and Peyton have together.
theorem total_tickets : tate_total_tickets + peyton_tickets = 51 := by
  -- Placeholder for the proof
  sorry

end total_tickets_l1799_179986


namespace price_per_liter_l1799_179922

theorem price_per_liter (cost : ℕ) (bottles : ℕ) (liters_per_bottle : ℕ) (total_cost : ℕ) (total_liters : ℕ) :
  bottles = 6 → liters_per_bottle = 2 → total_cost = 12 → total_liters = 12 → cost = total_cost / total_liters → cost = 1 :=
by
  intros h_bottles h_liters_per_bottle h_total_cost h_total_liters h_cost_div
  sorry

end price_per_liter_l1799_179922


namespace Tanya_efficiency_higher_l1799_179905

variable (Sakshi_days Tanya_days : ℕ)
variable (Sakshi_efficiency Tanya_efficiency increase_in_efficiency percentage_increase : ℚ)

theorem Tanya_efficiency_higher (h1: Sakshi_days = 20) (h2: Tanya_days = 16) :
  Sakshi_efficiency = 1 / 20 ∧ Tanya_efficiency = 1 / 16 ∧ 
  increase_in_efficiency = Tanya_efficiency - Sakshi_efficiency ∧ 
  percentage_increase = (increase_in_efficiency / Sakshi_efficiency) * 100 ∧
  percentage_increase = 25 := by
  sorry

end Tanya_efficiency_higher_l1799_179905


namespace first_inequality_system_of_inequalities_l1799_179913

-- First inequality problem
theorem first_inequality (x : ℝ) : 
  1 - (x - 3) / 6 > x / 3 → x < 3 := 
sorry

-- System of inequalities problem
theorem system_of_inequalities (x : ℝ) : 
  (x + 1 ≥ 3 * (x - 3)) ∧ ((x + 2) / 3 - (x - 1) / 4 > 1) → (1 < x ∧ x ≤ 5) := 
sorry

end first_inequality_system_of_inequalities_l1799_179913


namespace infinite_integer_solutions_l1799_179931

variable (x : ℤ)

theorem infinite_integer_solutions (x : ℤ) : 
  ∃ (k : ℤ), ∀ n : ℤ, n > 2 → k = n :=
by {
  sorry
}

end infinite_integer_solutions_l1799_179931


namespace divide_24kg_into_parts_l1799_179920

theorem divide_24kg_into_parts (W : ℕ) (part1 part2 : ℕ) (h_sum : part1 + part2 = 24) :
  (part1 = 9 ∧ part2 = 15) ∨ (part1 = 15 ∧ part2 = 9) :=
by
  sorry

end divide_24kg_into_parts_l1799_179920


namespace problem_l1799_179918

noncomputable def f (x : ℝ) : ℝ := |x + 1| - |x|

theorem problem :
  (∀ x, f x ≤ 1) ∧
  (∃ x, f x = 1) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 → 
    ∃ x, (x = (a^2 / (b + 1) + b^2 / (a + 1)) ∧ x = 1 / 3)) :=
by {
  sorry
}

end problem_l1799_179918


namespace find_n_divisors_l1799_179911

theorem find_n_divisors (n : ℕ) (h1 : 2287 % n = 2028 % n)
                        (h2 : 2028 % n = 1806 % n) : n = 37 := 
by
  sorry

end find_n_divisors_l1799_179911


namespace find_Tom_favorite_numbers_l1799_179988

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_multiple_of (n k : ℕ) : Prop :=
  n % k = 0

def Tom_favorite_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 150 ∧
  is_multiple_of n 13 ∧
  ¬ is_multiple_of n 3 ∧
  is_multiple_of (sum_of_digits n) 4

theorem find_Tom_favorite_numbers :
  ∃ n : ℕ, Tom_favorite_number n ∧ (n = 130 ∨ n = 143) :=
by
  sorry

end find_Tom_favorite_numbers_l1799_179988


namespace math_proof_problem_l1799_179961

theorem math_proof_problem (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3 * x₁ * y₁^2 = 2008)
  (h₂ : y₁^3 - 3 * x₁^2 * y₁ = 2007)
  (h₃ : x₂^3 - 3 * x₂ * y₂^2 = 2008)
  (h₄ : y₂^3 - 3 * x₂^2 * y₂ = 2007)
  (h₅ : x₃^3 - 3 * x₃ * y₃^2 = 2008)
  (h₆ : y₃^3 - 3 * x₃^2 * y₃ = 2007) :
  (1 - x₁ / y₁) * (1 - x₂ / y₂) * (1 - x₃ / y₃) = 4015 / 2008 :=
by sorry

end math_proof_problem_l1799_179961


namespace quadratic_roots_l1799_179936

theorem quadratic_roots (b c : ℝ) (h : ∀ x : ℝ, x^2 + bx + c = 0 ↔ x^2 - 5 * x + 2 = 0):
  c / b = -4 / 21 :=
  sorry

end quadratic_roots_l1799_179936


namespace jackie_phil_probability_l1799_179962

noncomputable def probability_same_heads : ℚ :=
  let fair_coin := (1 + 1: ℚ)
  let p3_coin := (2 + 3: ℚ)
  let p2_coin := (1 + 2: ℚ)
  let generating_function := fair_coin * p3_coin * p2_coin
  let sum_of_coefficients := 30
  let sum_of_squares_of_coefficients := 290
  sum_of_squares_of_coefficients / (sum_of_coefficients ^ 2)

theorem jackie_phil_probability : probability_same_heads = 29 / 90 := by
  sorry

end jackie_phil_probability_l1799_179962


namespace ant_food_cost_l1799_179942

-- Definitions for the conditions
def number_of_ants : ℕ := 400
def food_per_ant : ℕ := 2
def job_charge : ℕ := 5
def leaf_charge : ℕ := 1 / 100 -- 1 penny is 1 cent which is 0.01 dollars
def leaves_raked : ℕ := 6000
def jobs_completed : ℕ := 4

-- Compute the total money earned from jobs
def money_from_jobs : ℕ := jobs_completed * job_charge

-- Compute the total money earned from raking leaves
def money_from_leaves : ℕ := leaves_raked * leaf_charge

-- Compute the total money earned
def total_money_earned : ℕ := money_from_jobs + money_from_leaves

-- Compute the total ounces of food needed
def total_food_needed : ℕ := number_of_ants * food_per_ant

-- Calculate the cost per ounce of food
def cost_per_ounce : ℕ := total_money_earned / total_food_needed

theorem ant_food_cost :
  cost_per_ounce = 1 / 10 := sorry

end ant_food_cost_l1799_179942


namespace product_expression_evaluation_l1799_179964

theorem product_expression_evaluation :
  (1 + 2 / 1) * (1 + 2 / 2) * (1 + 2 / 3) * (1 + 2 / 4) * (1 + 2 / 5) * (1 + 2 / 6) - 1 = 25 / 3 :=
by
  sorry

end product_expression_evaluation_l1799_179964


namespace divides_sum_if_divides_polynomial_l1799_179949

theorem divides_sum_if_divides_polynomial (x y : ℕ) : 
  x^2 ∣ x^2 + x * y + x + y → x^2 ∣ x + y :=
by
  sorry

end divides_sum_if_divides_polynomial_l1799_179949


namespace complement_intersection_l1799_179966

def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
def set_A : Set ℕ := {1, 3, 5}
def set_B : Set ℕ := {2, 3, 6}

theorem complement_intersection :
  ((universal_set \ set_A) ∩ set_B) = {2, 6} :=
by
  sorry

end complement_intersection_l1799_179966


namespace work_together_days_l1799_179967

theorem work_together_days (hA : ∃ d : ℝ, d > 0 ∧ d = 15)
                          (hB : ∃ d : ℝ, d > 0 ∧ d = 20)
                          (hfrac : ∃ f : ℝ, f = (23 / 30)) :
  ∃ d : ℝ, d = 2 := by
  sorry

end work_together_days_l1799_179967


namespace garden_area_l1799_179980

theorem garden_area (P : ℝ) (hP : P = 72) (l w : ℝ) (hL : l = 3 * w) (hPerimeter : 2 * l + 2 * w = P) : l * w = 243 := 
by
  sorry

end garden_area_l1799_179980


namespace a_and_b_together_complete_work_in_12_days_l1799_179947

-- Define the rate of work for b
def R_b : ℚ := 1 / 60

-- Define the rate of work for a based on the given condition that a is four times as fast as b
def R_a : ℚ := 4 * R_b

-- Define the combined rate of work for a and b working together
def R_a_plus_b : ℚ := R_a + R_b

-- Define the target time
def target_time : ℚ := 12

-- Proof statement
theorem a_and_b_together_complete_work_in_12_days :
  (R_a_plus_b * target_time) = 1 :=
by
  -- Proof omitted
  sorry

end a_and_b_together_complete_work_in_12_days_l1799_179947


namespace probability_of_selecting_two_girls_l1799_179998

def total_students : ℕ := 5
def boys : ℕ := 2
def girls : ℕ := 3
def selected_students : ℕ := 2

theorem probability_of_selecting_two_girls :
  (Nat.choose girls selected_students : ℝ) / (Nat.choose total_students selected_students : ℝ) = 0.3 := by
  sorry

end probability_of_selecting_two_girls_l1799_179998


namespace canoes_more_than_kayaks_l1799_179989

noncomputable def canoes_difference (C K : ℕ) : Prop :=
  15 * C + 18 * K = 405 ∧ 2 * C = 3 * K → C - K = 5

theorem canoes_more_than_kayaks (C K : ℕ) : canoes_difference C K :=
by
  sorry

end canoes_more_than_kayaks_l1799_179989


namespace probability_of_green_l1799_179953

-- Define the conditions
def P_R : ℝ := 0.15
def P_O : ℝ := 0.35
def P_B : ℝ := 0.2
def total_probability (P_Y P_G : ℝ) : Prop := P_R + P_O + P_B + P_Y + P_G = 1

-- State the theorem to be proven
theorem probability_of_green (P_Y : ℝ) (P_G : ℝ) (h : total_probability P_Y P_G) (P_Y_assumption : P_Y = 0.15) : P_G = 0.15 :=
by
  sorry

end probability_of_green_l1799_179953


namespace fivefold_composition_l1799_179901

def f (x : ℚ) : ℚ := -2 / x

theorem fivefold_composition :
  f (f (f (f (f (3))))) = -2 / 3 := 
by
  -- Proof goes here
  sorry

end fivefold_composition_l1799_179901


namespace Eric_rent_days_l1799_179929

-- Define the conditions given in the problem
def daily_rate := 50.00
def rate_14_days := 500.00
def total_cost := 800.00

-- State the problem as a theorem in Lean
theorem Eric_rent_days : ∀ (d : ℕ), (d : ℕ) = 20 :=
by
  sorry

end Eric_rent_days_l1799_179929


namespace find_abc_value_l1799_179955

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom h1 : a + 1 / b = 5
axiom h2 : b + 1 / c = 2
axiom h3 : c + 1 / a = 9 / 4

theorem find_abc_value : a * b * c = (7 + Real.sqrt 21) / 8 :=
by
  sorry

end find_abc_value_l1799_179955


namespace inequality_holds_iff_even_l1799_179983

theorem inequality_holds_iff_even (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∀ x y z : ℝ, (x - y) ^ a * (x - z) ^ b * (y - z) ^ c ≥ 0) ↔ (Even a ∧ Even b ∧ Even c) :=
by
  sorry

end inequality_holds_iff_even_l1799_179983


namespace percentage_shaded_l1799_179900

def area_rect (width height : ℝ) : ℝ := width * height

def overlap_area (side_length : ℝ) (width_rect : ℝ) (length_rect: ℝ) (length_total: ℝ) : ℝ :=
  (side_length - (length_total - length_rect)) * width_rect

theorem percentage_shaded (sqr_side length_rect width_rect total_length total_width : ℝ) (h1 : sqr_side = 12) (h2 : length_rect = 9) (h3 : width_rect = 12)
  (h4 : total_length = 18) (h5 : total_width = 12) :
  (overlap_area sqr_side width_rect length_rect total_length) / (area_rect total_width total_length) * 100 = 12.5 :=
by
  sorry

end percentage_shaded_l1799_179900


namespace find_D_l1799_179995

theorem find_D (P Q : ℕ) (h_pos : 0 < P ∧ 0 < Q) (h_eq : P + Q + P * Q = 90) : P + Q = 18 := by
  sorry

end find_D_l1799_179995
