import Mathlib

namespace sum_of_inserted_numbers_in_progressions_l1585_158519

theorem sum_of_inserted_numbers_in_progressions (x y : ℝ) (hx : 4 * (y / x) = x) (hy : 2 * y = x + 64) :
  x + y = 131 + 3 * Real.sqrt 129 :=
by
  sorry

end sum_of_inserted_numbers_in_progressions_l1585_158519


namespace find_k_l1585_158542

open Classical

theorem find_k 
    (z x y k : ℝ) 
    (k_pos_int : k > 0 ∧ ∃ n : ℕ, k = n)
    (prop1 : z - y = k * x)
    (prop2 : x - z = k * y)
    (cond : z = (5 / 3) * (x - y)) :
    k = 3 :=
by
  sorry

end find_k_l1585_158542


namespace chinese_mathematical_system_l1585_158534

noncomputable def problem_statement : Prop :=
  ∃ (x : ℕ) (y : ℕ),
    7 * x + 7 = y ∧ 
    9 * (x - 1) = y

theorem chinese_mathematical_system :
  problem_statement := by
  sorry

end chinese_mathematical_system_l1585_158534


namespace square_possible_length_l1585_158506

theorem square_possible_length (sticks : Finset ℕ) (H : sticks = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ∃ s, s = 9 ∧
  ∃ (a b c : ℕ), a ∈ sticks ∧ b ∈ sticks ∧ c ∈ sticks ∧ a + b + c = 9 :=
by
  sorry

end square_possible_length_l1585_158506


namespace cube_positive_integers_solution_l1585_158578

theorem cube_positive_integers_solution (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) :
  (∃ k : ℕ, 2^(Nat.factorial a) + 2^(Nat.factorial b) + 2^(Nat.factorial c) = k^3) ↔ 
    ( (a = 1 ∧ b = 1 ∧ c = 2) ∨ 
      (a = 1 ∧ b = 2 ∧ c = 1) ∨ 
      (a = 2 ∧ b = 1 ∧ c = 1) ) :=
by
  sorry

end cube_positive_integers_solution_l1585_158578


namespace total_children_estimate_l1585_158586

theorem total_children_estimate (k m n : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : k > 0) 
(h4 : n ≤ m) (h5 : n ≤ k) (h6 : m ≤ k) :
  (∃ (total : ℕ), total = k * m / n) :=
sorry

end total_children_estimate_l1585_158586


namespace circle_area_l1585_158572

theorem circle_area (r : ℝ) (h1 : 5 * (1 / (2 * Real.pi * r)) = 2 * r) : π * r^2 = 5 / 4 := by
  sorry

end circle_area_l1585_158572


namespace roses_picked_second_time_l1585_158517

-- Define the initial conditions
def initial_roses : ℝ := 37.0
def first_pick : ℝ := 16.0
def total_roses_after_second_picking : ℝ := 72.0

-- Define the calculation after the first picking
def roses_after_first_picking : ℝ := initial_roses + first_pick

-- The Lean statement to prove the number of roses picked the second time
theorem roses_picked_second_time : total_roses_after_second_picking - roses_after_first_picking = 19.0 := 
by
  -- Use the facts stated in the conditions
  sorry

end roses_picked_second_time_l1585_158517


namespace complex_fraction_l1585_158587

open Complex

theorem complex_fraction
  (a b : ℂ)
  (h : (a + b) / (a - b) - (a - b) / (a + b) = 2) :
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = ((a^2 + b^2) - 3) / 3 := 
by
  sorry

end complex_fraction_l1585_158587


namespace handshakes_at_meetup_l1585_158557

theorem handshakes_at_meetup :
  let gremlins := 25
  let imps := 20
  let sprites := 10
  ∃ (total_handshakes : ℕ), total_handshakes = 1095 :=
by
  sorry

end handshakes_at_meetup_l1585_158557


namespace Peggy_dolls_l1585_158525

theorem Peggy_dolls (initial_dolls granny_dolls birthday_dolls : ℕ) (h1 : initial_dolls = 6) (h2 : granny_dolls = 30) (h3 : birthday_dolls = granny_dolls / 2) : 
  initial_dolls + granny_dolls + birthday_dolls = 51 := by
  sorry

end Peggy_dolls_l1585_158525


namespace billy_unknown_lap_time_l1585_158592

theorem billy_unknown_lap_time :
  ∀ (time_first_5_laps time_next_3_laps time_last_lap time_margaret total_time_billy : ℝ) (lap_time_unknown : ℝ),
    time_first_5_laps = 2 ∧
    time_next_3_laps = 4 ∧
    time_last_lap = 2.5 ∧
    time_margaret = 10 ∧
    total_time_billy = time_margaret - 0.5 →
    (time_first_5_laps + time_next_3_laps + time_last_lap + lap_time_unknown = total_time_billy) →
    lap_time_unknown = 1 :=
by
  sorry

end billy_unknown_lap_time_l1585_158592


namespace marbles_leftover_l1585_158546

theorem marbles_leftover (g j : ℕ) (hg : g % 8 = 5) (hj : j % 8 = 6) :
  ((g + 5 + j) % 8) = 0 :=
by
  sorry

end marbles_leftover_l1585_158546


namespace marie_finishes_fourth_task_at_11_40_am_l1585_158596

-- Define the given conditions
def start_time : ℕ := 7 * 60 -- start time in minutes from midnight (7:00 AM)
def second_task_end_time : ℕ := 9 * 60 + 20 -- end time of second task in minutes from midnight (9:20 AM)
def num_tasks : ℕ := 4 -- four tasks
def task_duration : ℕ := (second_task_end_time - start_time) / 2 -- duration of one task

-- Define the goal to prove: the end time of the fourth task
def fourth_task_finish_time : ℕ := second_task_end_time + 2 * task_duration

theorem marie_finishes_fourth_task_at_11_40_am : fourth_task_finish_time = 11 * 60 + 40 := by
  sorry

end marie_finishes_fourth_task_at_11_40_am_l1585_158596


namespace range_of_m_l1585_158567

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x^4) / Real.log 3
noncomputable def g (x : ℝ) : ℝ := x * f x

theorem range_of_m (m : ℝ) : g (1 - m) < g (2 * m) → m > 1 / 3 :=
  by
  sorry

end range_of_m_l1585_158567


namespace expression_value_l1585_158527

theorem expression_value : 
  ∀ (x y z: ℤ), x = 2 ∧ y = -3 ∧ z = 1 → x^2 + y^2 - z^2 - 2*x*y = 24 := by
  sorry

end expression_value_l1585_158527


namespace find_D_l1585_158577

-- We define the points E and F
def E : ℝ × ℝ := (-3, -2)
def F : ℝ × ℝ := (5, 10)

-- Definition of point D with the given conditions
def D : ℝ × ℝ := (3, 7)

-- We state the main theorem to prove that D is such that ED = 2 * DF given E and F
theorem find_D (D : ℝ × ℝ) (ED_DF_relation : dist E D = 2 * dist D F) : D = (3, 7) :=
sorry

end find_D_l1585_158577


namespace log_product_eq_3_div_4_l1585_158537

theorem log_product_eq_3_div_4 : (Real.log 3 / Real.log 4) * (Real.log 8 / Real.log 9) = 3 / 4 :=
by
  sorry

end log_product_eq_3_div_4_l1585_158537


namespace polynomial_sum_evaluation_l1585_158563

noncomputable def q1 : Polynomial ℤ := Polynomial.X^3
noncomputable def q2 : Polynomial ℤ := Polynomial.X^2 + Polynomial.X + 1
noncomputable def q3 : Polynomial ℤ := Polynomial.X - 1
noncomputable def q4 : Polynomial ℤ := Polynomial.X^2 + 1

theorem polynomial_sum_evaluation :
  q1.eval 3 + q2.eval 3 + q3.eval 3 + q4.eval 3 = 52 :=
by
  sorry

end polynomial_sum_evaluation_l1585_158563


namespace average_eq_one_half_l1585_158549

variable (w x y : ℝ)

-- Conditions
variables (h1 : 2 / w + 2 / x = 2 / y)
variables (h2 : w * x = y)

theorem average_eq_one_half : (w + x) / 2 = 1 / 2 :=
by
  sorry

end average_eq_one_half_l1585_158549


namespace solution_set_of_inequality_l1585_158590

theorem solution_set_of_inequality (x : ℝ) : (x^2 + 4*x - 5 < 0) ↔ (-5 < x ∧ x < 1) :=
by sorry

end solution_set_of_inequality_l1585_158590


namespace pairs_of_natural_numbers_l1585_158562

theorem pairs_of_natural_numbers (a b : ℕ) (h₁ : b ∣ a + 1) (h₂ : a ∣ b + 1) :
    (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 1) ∨ (a = 3 ∧ b = 2) :=
by {
  sorry
}

end pairs_of_natural_numbers_l1585_158562


namespace sugar_spilled_l1585_158502

-- Define the initial amount of sugar and the amount left
def initial_sugar : ℝ := 9.8
def remaining_sugar : ℝ := 4.6

-- State the problem as a theorem
theorem sugar_spilled :
  initial_sugar - remaining_sugar = 5.2 := 
sorry

end sugar_spilled_l1585_158502


namespace proof_problem_l1585_158571

-- Condition for the first part: a quadratic inequality having a solution set
def quadratic_inequality (a : ℝ) :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → a * x^2 - 3 * x + 2 ≤ 0

-- Condition for the second part: the solution set of a rational inequality
def rational_inequality_solution (a : ℝ) (b : ℝ) :=
  ∀ x : ℝ, (x + 3) / (a * x - b) > 0 ↔ (x < -3 ∨ x > 2)

theorem proof_problem {a : ℝ} {b : ℝ} :
  (quadratic_inequality a → a = 1 ∧ b = 2) ∧ 
  (rational_inequality_solution 1 2) :=
by
  sorry

end proof_problem_l1585_158571


namespace constant_is_5_variables_are_n_and_S_l1585_158558

-- Define the conditions
def cost_per_box : ℕ := 5
def total_cost (n : ℕ) : ℕ := n * cost_per_box

-- Define the statement to be proved
-- constant is 5
theorem constant_is_5 : cost_per_box = 5 := 
by sorry

-- variables are n and S, where S is total_cost n
theorem variables_are_n_and_S (n : ℕ) : 
    ∃ S : ℕ, S = total_cost n :=
by sorry

end constant_is_5_variables_are_n_and_S_l1585_158558


namespace proof_problem_l1585_158535

noncomputable def f (x : ℝ) := 3 * Real.sin x + 2 * Real.cos x + 1

theorem proof_problem (a b c : ℝ) (h : ∀ x : ℝ, a * f x + b * f (x - c) = 1) :
  (b * Real.cos c / a) = -1 :=
sorry

end proof_problem_l1585_158535


namespace smallest_pencils_l1585_158521

theorem smallest_pencils (P : ℕ) :
  (P > 2) ∧
  (P % 5 = 2) ∧
  (P % 9 = 2) ∧
  (P % 11 = 2) →
  P = 497 := by
  sorry

end smallest_pencils_l1585_158521


namespace inequality_hold_l1585_158513

theorem inequality_hold (n : ℕ) (h1 : n > 1) : 1 + n * 2^((n - 1 : ℕ) / 2) < 2^n :=
by
  sorry

end inequality_hold_l1585_158513


namespace intersection_PQ_l1585_158538

def setP  := {x : ℝ | x * (x - 1) ≥ 0}
def setQ := {y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1}

theorem intersection_PQ : {x : ℝ | x > 1} = {z : ℝ | z ∈ setP ∧ z ∈ setQ} :=
by
  sorry

end intersection_PQ_l1585_158538


namespace men_in_first_group_l1585_158585

theorem men_in_first_group (x : ℕ) :
  (20 * 48 = x * 80) → x = 12 :=
by
  intro h_eq
  have : x = (20 * 48) / 80 := sorry
  exact this

end men_in_first_group_l1585_158585


namespace percentage_decrease_10_l1585_158522

def stocks_decrease (F J M : ℝ) (X : ℝ) : Prop :=
  J = F * (1 - X / 100) ∧
  J = M * 1.20 ∧
  M = F * 0.7500000000000007

theorem percentage_decrease_10 {F J M X : ℝ} (h : stocks_decrease F J M X) :
  X = 9.99999999999992 :=
by
  sorry

end percentage_decrease_10_l1585_158522


namespace prob_of_two_reds_is_7_over_20_expected_value_is_2_19_l1585_158554

section card_draws

/-- A card draw experiment with 10 cards: 5 red, 3 white, 2 blue. --/
inductive CardColor
| red
| white
| blue

def bag : List CardColor := List.replicate 5 CardColor.red ++ List.replicate 3 CardColor.white ++ List.replicate 2 CardColor.blue

/-- Probability of drawing exactly 2 red cards in up to 3 draws with the given conditions. --/
def prob_two_reds : ℚ :=
  (5 / 10) * (5 / 10) + 
  (5 / 10) * (2 / 10) * (5 / 10) + 
  (2 / 10) * (5 / 10) * (5 / 10)

theorem prob_of_two_reds_is_7_over_20 : prob_two_reds = 7 / 20 :=
  sorry

/-- Probability distribution of the number of draws necessary. --/
def prob_ξ_1 : ℚ := 3 / 10
def prob_ξ_2 : ℚ := 21 / 100
def prob_ξ_3 : ℚ := 49 / 100
def expected_value_ξ : ℚ :=
  1 * prob_ξ_1 + 2 * prob_ξ_2 + 3 * prob_ξ_3

theorem expected_value_is_2_19 : expected_value_ξ = 219 / 100 :=
  sorry

end card_draws

end prob_of_two_reds_is_7_over_20_expected_value_is_2_19_l1585_158554


namespace gcd_1729_1768_l1585_158591

theorem gcd_1729_1768 : Int.gcd 1729 1768 = 13 := by
  sorry

end gcd_1729_1768_l1585_158591


namespace lassis_from_mangoes_l1585_158588

def ratio (lassis mangoes : ℕ) : Prop := lassis = 11 * mangoes / 2

theorem lassis_from_mangoes (mangoes : ℕ) (h : mangoes = 10) : ratio 55 mangoes :=
by
  rw [h]
  unfold ratio
  sorry

end lassis_from_mangoes_l1585_158588


namespace minimum_value_2x_plus_y_l1585_158560

theorem minimum_value_2x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + y + 6 = x * y) : 
  2 * x + y ≥ 12 := 
sorry

end minimum_value_2x_plus_y_l1585_158560


namespace radius_of_circle_l1585_158532

/-- Given the equation of a circle x^2 + y^2 - 8 = 2x + 4y,
    we need to prove that the radius of the circle is sqrt 13. -/
theorem radius_of_circle : 
    ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 8 = 2*x + 4*y → r = Real.sqrt 13) :=
by
    sorry

end radius_of_circle_l1585_158532


namespace gideon_fraction_of_marbles_l1585_158531

variable (f : ℝ)

theorem gideon_fraction_of_marbles (marbles : ℝ) (age_now : ℝ) (age_future : ℝ) (remaining_marbles : ℝ) (future_age_with_remaining_marbles : Bool)
  (h1 : marbles = 100)
  (h2 : age_now = 45)
  (h3 : age_future = age_now + 5)
  (h4 : remaining_marbles = 2 * (1 - f) * marbles)
  (h5 : remaining_marbles = age_future)
  (h6 : future_age_with_remaining_marbles = (age_future = 50)) :
  f = 3 / 4 :=
by
  sorry

end gideon_fraction_of_marbles_l1585_158531


namespace find_inverse_mod_36_l1585_158526

-- Given condition
def inverse_mod_17 := (17 * 23) % 53 = 1

-- Definition for the problem statement
def inverse_mod_36 : Prop := (36 * 30) % 53 = 1

theorem find_inverse_mod_36 (h : inverse_mod_17) : inverse_mod_36 :=
sorry

end find_inverse_mod_36_l1585_158526


namespace sequence_inequality_l1585_158589

theorem sequence_inequality
  (a : ℕ → ℝ)
  (h₁ : a 1 = 0)
  (h₇ : a 7 = 0) :
  ∃ k : ℕ, k ≤ 5 ∧ a k + a (k + 2) ≤ a (k + 1) * Real.sqrt 3 := 
sorry

end sequence_inequality_l1585_158589


namespace bianca_total_books_l1585_158500

theorem bianca_total_books (shelves_mystery shelves_picture books_per_shelf : ℕ) 
  (h1 : shelves_mystery = 5) 
  (h2 : shelves_picture = 4) 
  (h3 : books_per_shelf = 8) : 
  (shelves_mystery * books_per_shelf + shelves_picture * books_per_shelf) = 72 := 
by 
  sorry

end bianca_total_books_l1585_158500


namespace car_speed_proof_l1585_158545

noncomputable def car_speed_second_hour 
  (speed_first_hour: ℕ) (average_speed: ℕ) (total_time: ℕ) 
  (speed_second_hour: ℕ) : Prop :=
  (speed_first_hour = 80) ∧ (average_speed = 70) ∧ (total_time = 2) → speed_second_hour = 60

theorem car_speed_proof : 
  car_speed_second_hour 80 70 2 60 := by
  sorry

end car_speed_proof_l1585_158545


namespace new_deck_card_count_l1585_158509

-- Define the conditions
def cards_per_time : ℕ := 30
def times_per_week : ℕ := 3
def weeks : ℕ := 11
def decks : ℕ := 18
def total_cards_tear_per_week : ℕ := cards_per_time * times_per_week
def total_cards_tear : ℕ := total_cards_tear_per_week * weeks
def total_cards_in_decks (cards_per_deck : ℕ) : ℕ := decks * cards_per_deck

-- Define the theorem we need to prove
theorem new_deck_card_count :
  ∃ (x : ℕ), total_cards_in_decks x = total_cards_tear ↔ x = 55 := by
  sorry

end new_deck_card_count_l1585_158509


namespace original_selling_price_l1585_158599

variable (P : ℝ)

def SP1 := 1.10 * P
def P_new := 0.90 * P
def SP2 := 1.17 * P
def price_diff := SP2 - SP1

theorem original_selling_price : price_diff = 49 → SP1 = 770 :=
by
  sorry

end original_selling_price_l1585_158599


namespace rabbit_clearing_10_square_yards_per_day_l1585_158508

noncomputable def area_cleared_by_one_rabbit_per_day (length width : ℕ) (rabbits : ℕ) (days : ℕ) : ℕ :=
  (length * width) / (3 * 3 * rabbits * days)

theorem rabbit_clearing_10_square_yards_per_day :
  area_cleared_by_one_rabbit_per_day 200 900 100 20 = 10 :=
by sorry

end rabbit_clearing_10_square_yards_per_day_l1585_158508


namespace line_equation_through_P_and_equidistant_from_A_B_l1585_158565

theorem line_equation_through_P_and_equidistant_from_A_B (P A B : ℝ × ℝ) (hP : P = (1, 2)) (hA : A = (2, 3)) (hB : B = (4, -5)) :
  (∃ l : ℝ × ℝ → Prop, ∀ x y, l (x, y) ↔ 4 * x + y - 6 = 0 ∨ 3 * x + 2 * y - 7 = 0) :=
sorry

end line_equation_through_P_and_equidistant_from_A_B_l1585_158565


namespace both_solve_prob_l1585_158573

variable (a b : ℝ) -- Define a and b as real numbers

-- Define the conditions
def not_solve_prob_A := (0 ≤ a) ∧ (a ≤ 1)
def not_solve_prob_B := (0 ≤ b) ∧ (b ≤ 1)
def independent := true -- independence is implicit by the question

-- Define the statement of the proof
theorem both_solve_prob (h1 : not_solve_prob_A a) (h2 : not_solve_prob_B b) :
  (1 - a) * (1 - b) = (1 - a) * (1 - b) :=
by sorry

end both_solve_prob_l1585_158573


namespace gold_bars_distribution_l1585_158555

theorem gold_bars_distribution 
  (initial_gold : ℕ) 
  (lost_gold : ℕ) 
  (num_friends : ℕ) 
  (remaining_gold : ℕ)
  (each_friend_gets : ℕ) :
  initial_gold = 100 →
  lost_gold = 20 →
  num_friends = 4 →
  remaining_gold = initial_gold - lost_gold →
  each_friend_gets = remaining_gold / num_friends →
  each_friend_gets = 20 :=
by
  intros
  sorry

end gold_bars_distribution_l1585_158555


namespace draw_9_cards_ensure_even_product_l1585_158564

theorem draw_9_cards_ensure_even_product :
  ∀ (cards : Finset ℕ), (∀ x ∈ cards, 1 ≤ x ∧ x ≤ 16) →
  (cards.card = 9) →
  (∃ (subset : Finset ℕ), subset ⊆ cards ∧ ∃ k ∈ subset, k % 2 = 0) :=
by
  sorry

end draw_9_cards_ensure_even_product_l1585_158564


namespace possible_values_a1_l1585_158550

theorem possible_values_a1 (m : ℕ) (h_m_pos : 0 < m)
    (a : ℕ → ℕ) (h_seq : ∀ n, a n.succ = if a n < 2^m then a n ^ 2 + 2^m else a n / 2)
    (h1 : ∀ n, a n > 0) :
    (∀ n, ∃ k : ℕ, a n = 2^k) ↔ (m = 2 ∧ ∃ ℓ : ℕ, a 0 = 2 ^ ℓ ∧ 0 < ℓ) :=
by sorry

end possible_values_a1_l1585_158550


namespace students_play_both_l1585_158556

def students_total : ℕ := 25
def hockey_players : ℕ := 15
def basketball_players : ℕ := 16
def neither_players : ℕ := 4

theorem students_play_both : 
  (students_total - neither_players) + (hockey_players + basketball_players - students_total + neither_players - students_total) = 10 :=
by 
  sorry

end students_play_both_l1585_158556


namespace cot_30_plus_cot_75_eq_2_l1585_158524

noncomputable def cot (θ : ℝ) : ℝ := 1 / Real.tan θ

theorem cot_30_plus_cot_75_eq_2 : cot 30 + cot 75 = 2 := by sorry

end cot_30_plus_cot_75_eq_2_l1585_158524


namespace number_is_square_plus_opposite_l1585_158569

theorem number_is_square_plus_opposite (x : ℝ) (hx : x = x^2 + -x) : x = 0 ∨ x = 2 :=
by sorry

end number_is_square_plus_opposite_l1585_158569


namespace base_conversion_equivalence_l1585_158520

theorem base_conversion_equivalence :
  ∃ (n : ℕ), (∃ (C B : ℕ), C < 9 ∧ B < 6 ∧ n = 9 * C + B) ∧
             (∃ (C B : ℕ), C < 9 ∧ B < 6 ∧ n = 6 * B + C) ∧
             n = 0 := 
by 
  sorry

end base_conversion_equivalence_l1585_158520


namespace maciek_total_cost_l1585_158568

-- Define the cost of pretzels without discount
def pretzel_price : ℝ := 4.0

-- Define the discounted price of pretzels when buying 3 or more packs
def pretzel_discount_price : ℝ := 3.5

-- Define the cost of chips without discount
def chips_price : ℝ := 7.0

-- Define the discounted price of chips when buying 2 or more packs
def chips_discount_price : ℝ := 6.0

-- Define the number of pretzels Maciek buys
def pretzels_bought : ℕ := 3

-- Define the number of chips Maciek buys
def chips_bought : ℕ := 4

-- Calculate the total cost of pretzels
def pretzel_cost : ℝ :=
  if pretzels_bought >= 3 then pretzels_bought * pretzel_discount_price else pretzels_bought * pretzel_price

-- Calculate the total cost of chips
def chips_cost : ℝ :=
  if chips_bought >= 2 then chips_bought * chips_discount_price else chips_bought * chips_price

-- Calculate the total amount Maciek needs to pay
def total_cost : ℝ :=
  pretzel_cost + chips_cost

theorem maciek_total_cost :
  total_cost = 34.5 :=
by 
  sorry

end maciek_total_cost_l1585_158568


namespace probability_all_same_room_probability_at_least_two_same_room_l1585_158505

/-- 
  Given that there are three people and each person is assigned to one of four rooms with equal probability,
  let P1 be the probability that all three people are assigned to the same room,
  and let P2 be the probability that at least two people are assigned to the same room.
  We need to prove:
  1. P1 = 1 / 16
  2. P2 = 5 / 8
-/
noncomputable def P1 : ℚ := sorry

noncomputable def P2 : ℚ := sorry

theorem probability_all_same_room :
  P1 = 1 / 16 :=
sorry

theorem probability_at_least_two_same_room :
  P2 = 5 / 8 :=
sorry

end probability_all_same_room_probability_at_least_two_same_room_l1585_158505


namespace ellipse_and_triangle_properties_l1585_158551

noncomputable def standard_equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  1/2 * a * b

theorem ellipse_and_triangle_properties :
  (∀ x y : ℝ, standard_equation_of_ellipse x y ↔ (x, y) = (1, 3/2) ∨ (x, y) = (1, -3/2)) ∧
  area_triangle 2 3 = 3 :=
by
  sorry

end ellipse_and_triangle_properties_l1585_158551


namespace dollar_function_twice_l1585_158595

noncomputable def f (N : ℝ) : ℝ := 0.4 * N + 2

theorem dollar_function_twice (N : ℝ) (h : N = 30) : (f ∘ f) N = 5 := 
by
  sorry

end dollar_function_twice_l1585_158595


namespace problem_to_prove_l1585_158548

theorem problem_to_prove
  (α : ℝ)
  (h : Real.sin (3 * Real.pi / 2 + α) = 1 / 3) :
  Real.cos (Real.pi - 2 * α) = -7 / 9 :=
by
  sorry -- proof required

end problem_to_prove_l1585_158548


namespace simplify_expression_l1585_158547

theorem simplify_expression (a : ℝ) (h₁ : a ≠ 2) (h₂ : a ≠ -2) : 
  (2 * a / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2)) :=
by
  -- proof to be added
  sorry

end simplify_expression_l1585_158547


namespace option_B_correct_l1585_158580

theorem option_B_correct (x y : ℝ) : 
  x * y^2 - y^2 * x = 0 :=
by sorry

end option_B_correct_l1585_158580


namespace number_of_numbers_in_last_group_l1585_158583

theorem number_of_numbers_in_last_group :
  ∃ n : ℕ, (60 * 13) = (57 * 6) + 50 + (61 * n) ∧ n = 6 :=
sorry

end number_of_numbers_in_last_group_l1585_158583


namespace exists_n_good_not_n_add_1_good_l1585_158543

-- Define the sum of digits function S
def S (k : ℕ) : ℕ := (k.digits 10).sum

-- Define what it means for a number to be n-good
def n_good (a n : ℕ) : Prop :=
  ∃ (a_seq : Fin (n + 1) → ℕ), (a_seq 0 = a) ∧ (∀ i : Fin n, a_seq i.succ = a_seq i - S (a_seq i))

-- Define the main theorem
theorem exists_n_good_not_n_add_1_good : ∀ n : ℕ, ∃ a : ℕ, n_good a n ∧ ¬n_good a (n + 1) :=
by
  sorry

end exists_n_good_not_n_add_1_good_l1585_158543


namespace find_f_2008_l1585_158501

variable (f : ℝ → ℝ) 
variable (g : ℝ → ℝ) -- g is the inverse of f

def satisfies_conditions (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (g x) = x) ∧ (∀ y : ℝ, g (f y) = y) ∧ 
  (f 9 = 18) ∧ (∀ x : ℝ, g (x + 1) = (f (x + 1)))

theorem find_f_2008 (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (h : satisfies_conditions f g) : f 2008 = -1981 :=
sorry

end find_f_2008_l1585_158501


namespace bacon_needed_l1585_158570

def eggs_per_plate : ℕ := 2
def bacon_per_plate : ℕ := 2 * eggs_per_plate
def customers : ℕ := 14
def bacon_total (eggs_per_plate bacon_per_plate customers : ℕ) : ℕ := customers * bacon_per_plate

theorem bacon_needed : bacon_total eggs_per_plate bacon_per_plate customers = 56 :=
by
  sorry

end bacon_needed_l1585_158570


namespace sequence_general_term_l1585_158598

theorem sequence_general_term (a : ℕ → ℚ) (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n+1) = (n * a n + 2 * (n+1)^2) / (n+2)) :
  ∀ n : ℕ, a n = (1 / 2 : ℚ) * n * (n + 1) := by
  sorry

end sequence_general_term_l1585_158598


namespace equivalent_annual_rate_l1585_158574

theorem equivalent_annual_rate :
  ∀ (annual_rate compounding_periods: ℝ), annual_rate = 0.08 → compounding_periods = 4 → 
  ((1 + (annual_rate / compounding_periods)) ^ compounding_periods - 1) * 100 = 8.24 :=
by
  intros annual_rate compounding_periods h_rate h_periods
  sorry

end equivalent_annual_rate_l1585_158574


namespace ratio_of_r_l1585_158511

theorem ratio_of_r
  (total : ℕ) (r_amount : ℕ) (pq_amount : ℕ)
  (h_total : total = 7000 )
  (h_r_amount : r_amount = 2800 )
  (h_pq_amount : pq_amount = total - r_amount) :
  (r_amount / Nat.gcd r_amount pq_amount, pq_amount / Nat.gcd r_amount pq_amount) = (2, 3) :=
by
  sorry

end ratio_of_r_l1585_158511


namespace two_digit_divisors_1995_l1585_158559

theorem two_digit_divisors_1995 :
  (∃ (n : Finset ℕ), (∀ x ∈ n, 10 ≤ x ∧ x < 100 ∧ 1995 % x = 0) ∧ n.card = 6 ∧ ∃ y ∈ n, y = 95) :=
by
  sorry

end two_digit_divisors_1995_l1585_158559


namespace teagan_total_cost_l1585_158540

theorem teagan_total_cost :
  let reduction_percentage := 20
  let original_price_shirt := 60
  let original_price_jacket := 90
  let reduced_price_shirt := original_price_shirt * (100 - reduction_percentage) / 100
  let reduced_price_jacket := original_price_jacket * (100 - reduction_percentage) / 100
  let cost_5_shirts := 5 * reduced_price_shirt
  let cost_10_jackets := 10 * reduced_price_jacket
  let total_cost := cost_5_shirts + cost_10_jackets
  total_cost = 960 := by
  sorry

end teagan_total_cost_l1585_158540


namespace trigonometric_expression_l1585_158507

theorem trigonometric_expression
  (α : ℝ)
  (h : (Real.tan α - 3) * (Real.sin α + Real.cos α + 3) = 0) :
  2 + (2 / 3) * Real.sin α ^ 2 + (1 / 4) * Real.cos α ^ 2 = 21 / 8 := 
by sorry

end trigonometric_expression_l1585_158507


namespace bobbo_minimum_speed_increase_l1585_158529

theorem bobbo_minimum_speed_increase
  (initial_speed: ℝ)
  (river_width : ℝ)
  (current_speed : ℝ)
  (waterfall_distance : ℝ)
  (midpoint_distance : ℝ)
  (required_increase: ℝ) :
  initial_speed = 2 ∧ river_width = 100 ∧ current_speed = 5 ∧ waterfall_distance = 175 ∧ midpoint_distance = 50 ∧ required_increase = 3 → 
  (required_increase = (50 / (50 / current_speed)) - initial_speed) := 
by
  sorry

end bobbo_minimum_speed_increase_l1585_158529


namespace negative_linear_correlation_l1585_158510

theorem negative_linear_correlation (x y : ℝ) (h : y = 3 - 2 * x) : 
  ∃ c : ℝ, c < 0 ∧ y = 3 + c * x := 
by  
  sorry

end negative_linear_correlation_l1585_158510


namespace max_airlines_l1585_158528

-- Definitions for the conditions
-- There are 200 cities
def num_cities : ℕ := 200

-- Calculate the total number of city pairs
def num_city_pairs (n : ℕ) : ℕ := (n * (n - 1)) / 2

def total_city_pairs : ℕ := num_city_pairs num_cities

-- Minimum spanning tree concept
def min_flights_per_airline (n : ℕ) : ℕ := n - 1

def total_flights_required : ℕ := num_cities * min_flights_per_airline num_cities

-- Claim: Maximum number of airlines
theorem max_airlines (n : ℕ) (h : n = 200) : ∃ m : ℕ, m = (total_city_pairs / (min_flights_per_airline n)) ∧ m = 100 :=
by sorry

end max_airlines_l1585_158528


namespace measure_angle_T_l1585_158597

theorem measure_angle_T (P Q R S T : ℝ) (h₀ : P = R) (h₁ : R = T) (h₂ : Q + S = 180)
  (h_sum : P + Q + R + T + S = 540) : T = 120 :=
by
  sorry

end measure_angle_T_l1585_158597


namespace min_jellybeans_l1585_158541

theorem min_jellybeans (n : ℕ) (h1 : n ≥ 150) (h2 : n % 17 = 15) : n = 151 :=
by { sorry }

end min_jellybeans_l1585_158541


namespace tiles_difference_ninth_eighth_rectangle_l1585_158579

theorem tiles_difference_ninth_eighth_rectangle : 
  let width (n : Nat) := 2 * n
  let height (n : Nat) := n
  let tiles (n : Nat) := width n * height n
  tiles 9 - tiles 8 = 34 :=
by
  intro width height tiles
  sorry

end tiles_difference_ninth_eighth_rectangle_l1585_158579


namespace winning_candidate_percentage_l1585_158581

theorem winning_candidate_percentage (v1 v2 v3 : ℕ) (h1 : v1 = 1136) (h2 : v2 = 7636) (h3 : v3 = 11628) :
  ((v3: ℝ) / (v1 + v2 + v3)) * 100 = 57 := by
  sorry

end winning_candidate_percentage_l1585_158581


namespace Ray_has_4_nickels_left_l1585_158516

def Ray_initial_cents := 95
def Ray_cents_to_Peter := 25
def Ray_cents_to_Randi := 2 * Ray_cents_to_Peter

-- There are 5 cents in each nickel
def cents_per_nickel := 5

-- Nickels Ray originally has
def Ray_initial_nickels := Ray_initial_cents / cents_per_nickel
-- Nickels given to Peter
def Ray_nickels_to_Peter := Ray_cents_to_Peter / cents_per_nickel
-- Nickels given to Randi
def Ray_nickels_to_Randi := Ray_cents_to_Randi / cents_per_nickel
-- Total nickels given away
def Ray_nickels_given_away := Ray_nickels_to_Peter + Ray_nickels_to_Randi
-- Nickels left with Ray
def Ray_nickels_left := Ray_initial_nickels - Ray_nickels_given_away

theorem Ray_has_4_nickels_left :
  Ray_nickels_left = 4 :=
by
  sorry

end Ray_has_4_nickels_left_l1585_158516


namespace line_parallel_eq_l1585_158584

theorem line_parallel_eq (x y : ℝ) (h1 : 3 * x - y = 6) (h2 : x = -2 ∧ y = 3) :
  ∃ m b, m = 3 ∧ b = 9 ∧ y = m * x + b :=
by
  sorry

end line_parallel_eq_l1585_158584


namespace path_count_l1585_158530

theorem path_count (f : ℕ → (ℤ × ℤ)) :
  (∀ n, (f (n + 1)).1 = (f n).1 + 1 ∨ (f (n + 1)).2 = (f n).2 + 1) ∧
  f 0 = (-6, -6) ∧ f 24 = (6, 6) ∧
  (∀ n, ¬(-3 ≤ (f n).1 ∧ (f n).1 ≤ 3 ∧ -3 ≤ (f n).2 ∧ (f n).2 ≤ 3)) →
  ∃ N, N = 2243554 :=
by {
  sorry
}

end path_count_l1585_158530


namespace remainder_of_9_pow_1995_mod_7_l1585_158575

theorem remainder_of_9_pow_1995_mod_7 : (9^1995) % 7 = 1 := 
by 
sorry

end remainder_of_9_pow_1995_mod_7_l1585_158575


namespace list_price_proof_l1585_158552

theorem list_price_proof (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  sorry

end list_price_proof_l1585_158552


namespace hired_year_l1585_158561

theorem hired_year (A W : ℕ) (Y : ℕ) (retire_year : ℕ) 
    (hA : A = 30) 
    (h_rule : A + W = 70) 
    (h_retire : retire_year = 2006) 
    (h_employment : retire_year - Y = W) 
    : Y = 1966 := 
by 
  -- proofs are skipped with 'sorry'
  sorry

end hired_year_l1585_158561


namespace minimum_value_l1585_158544

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / (x + 1) + 9 / y = 1) : 4 * x + y ≥ 21 :=
sorry

end minimum_value_l1585_158544


namespace negate_even_condition_l1585_158536

theorem negate_even_condition (a b c : ℤ) :
  (¬(∀ a b c : ℤ, ∃ x : ℚ, a * x^2 + b * x + c = 0 → Even a ∧ Even b ∧ Even c)) →
  (¬Even a ∨ ¬Even b ∨ ¬Even c) :=
by
  sorry

end negate_even_condition_l1585_158536


namespace determine_a_range_l1585_158582

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem determine_a_range (e : ℝ) (he : e = Real.exp 1) :
  ∃ a_range : Set ℝ, a_range = Set.Icc 1 (e + 1 / e) :=
by 
  sorry

end determine_a_range_l1585_158582


namespace intersection_points_l1585_158566

variables (a b c : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem intersection_points :
  ∃ p q : ℝ × ℝ, 
    (p = (0, c) ∨ p = (-1, a - b + c)) ∧ 
    (q = (0, c) ∨ q = (-1, a - b + c)) ∧
    p ≠ q ∧
    (∃ x : ℝ, (x, ax^2 + bx + c) = p) ∧
    (∃ x : ℝ, (x, -ax^3 + bx + c) = q) :=
by
  sorry

end intersection_points_l1585_158566


namespace ratio_of_a_to_b_l1585_158518

variable (a b c d : ℝ)

theorem ratio_of_a_to_b (h1 : c = 0.20 * a) (h2 : c = 0.10 * b) : a = (1 / 2) * b :=
by
  sorry

end ratio_of_a_to_b_l1585_158518


namespace family_ages_l1585_158504

theorem family_ages 
  (youngest : ℕ)
  (middle : ℕ := youngest + 2)
  (eldest : ℕ := youngest + 4)
  (mother : ℕ := 3 * youngest + 16)
  (father : ℕ := 4 * youngest + 18)
  (total_sum : youngest + middle + eldest + mother + father = 90) :
  youngest = 5 ∧ middle = 7 ∧ eldest = 9 ∧ mother = 31 ∧ father = 38 := 
by 
  sorry

end family_ages_l1585_158504


namespace chessboard_grains_difference_l1585_158514

open BigOperators

def grains_on_square (k : ℕ) : ℕ := 2^k

def sum_of_first_n_squares (n : ℕ) : ℕ := ∑ k in Finset.range (n + 1), grains_on_square k

theorem chessboard_grains_difference : 
  grains_on_square 12 - sum_of_first_n_squares 10 = 2050 := 
by 
  -- Proof of the statement goes here.
  sorry

end chessboard_grains_difference_l1585_158514


namespace common_chord_length_of_two_circles_l1585_158503

noncomputable def common_chord_length (r : ℝ) : ℝ :=
  if r = 10 then 10 * Real.sqrt 3 else sorry

theorem common_chord_length_of_two_circles (r : ℝ) (h : r = 10) :
  common_chord_length r = 10 * Real.sqrt 3 :=
by
  rw [h]
  sorry

end common_chord_length_of_two_circles_l1585_158503


namespace find_n_l1585_158593

-- Definitions for conditions given in the problem
def a₂ (a : ℕ → ℕ) : Prop := a 2 = 3
def consecutive_sum (S : ℕ → ℕ) (n : ℕ) : Prop := ∀ n > 3, S n - S (n - 3) = 51
def total_sum (S : ℕ → ℕ) (n : ℕ) : Prop := S n = 100

-- The main proof problem
theorem find_n (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h₁ : a₂ a) (h₂ : consecutive_sum S n) (h₃ : total_sum S n) : n = 10 :=
sorry

end find_n_l1585_158593


namespace simplify_and_evaluate_expression_l1585_158512

theorem simplify_and_evaluate_expression (x : ℚ) (h : x = 3) :
  ((2 * x^2 + 2 * x) / (x^2 - 1) - (x^2 - x) / (x^2 - 2 * x + 1)) / (x / (x + 1)) = 2 :=
by
  sorry

end simplify_and_evaluate_expression_l1585_158512


namespace gcf_7fact_8fact_l1585_158515

theorem gcf_7fact_8fact : 
  let f_7 := Nat.factorial 7
  let f_8 := Nat.factorial 8
  Nat.gcd f_7 f_8 = f_7 := 
by
  sorry

end gcf_7fact_8fact_l1585_158515


namespace worst_player_is_niece_l1585_158533

structure Player where
  name : String
  sex : String
  generation : Nat

def grandmother := Player.mk "Grandmother" "Female" 1
def niece := Player.mk "Niece" "Female" 2
def grandson := Player.mk "Grandson" "Male" 3
def son_in_law := Player.mk "Son-in-law" "Male" 2

def worst_player : Player := niece
def best_player : Player := grandmother

-- Conditions
def cousin_check : worst_player ≠ best_player ∧
                   worst_player.generation ≠ best_player.generation ∧ 
                   worst_player.sex ≠ best_player.sex := 
  by sorry

-- Prove that the worst player is the niece
theorem worst_player_is_niece : worst_player = niece :=
  by sorry

end worst_player_is_niece_l1585_158533


namespace part_one_part_two_l1585_158523

theorem part_one (g : ℝ → ℝ) (h : ∀ x, g x = |x - 1| + 2) : {x : ℝ | |g x| < 5} = {x : ℝ | -2 < x ∧ x < 4} :=
sorry

theorem part_two (f g : ℝ → ℝ) (h1 : ∀ x, f x = |2 * x - a| + |2 * x + 3|) (h2 : ∀ x, g x = |x - 1| + 2) 
(h3 : ∀ x1 : ℝ, ∃ x2 : ℝ, f x1 = g x2) : {a : ℝ | a ≥ -1 ∨ a ≤ -5} :=
sorry

end part_one_part_two_l1585_158523


namespace compute_expression_in_terms_of_k_l1585_158594

-- Define the main theorem to be proven, with all conditions directly translated to Lean statements.
theorem compute_expression_in_terms_of_k
  (x y : ℝ)
  (h : (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = k) :
    (x^8 + y^8) / (x^8 - y^8) - (x^8 - y^8) / (x^8 + y^8) = ((k - 2)^2 * (k + 2)^2) / (4 * k * (k^2 + 4)) :=
by
  sorry

end compute_expression_in_terms_of_k_l1585_158594


namespace max_M_range_a_l1585_158553

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a / x + x * Real.log x
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

theorem max_M (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x1 ≤ 2) (h3 : 0 ≤ x2) (h4 : x2 ≤ 2) : 
  4 ≤ g x1 - g x2 :=
sorry

theorem range_a (a : ℝ) (s t : ℝ) (h1 : 1 / 2 ≤ s) (h2 : s ≤ 2) (h3 : 1 / 2 ≤ t) (h4 : t ≤ 2) : 
  1 ≤ a ∧ f s a ≥ g t :=
sorry

end max_M_range_a_l1585_158553


namespace problem_a1_value_l1585_158539

theorem problem_a1_value (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) (h : ∀ x : ℝ, x^10 = a + a₁ * (x - 1) + a₂ * (x - 1)^2 + a₃ * (x - 1)^3 + a₄ * (x - 1)^4 + a₅ * (x - 1)^5 + a₆ * (x - 1)^6 + a₇ * (x - 1)^7 + a₈ * (x - 1)^8 + a₉ * (x - 1)^9 + a₁₀ * (x - 1)^10) :
  a₁ = 10 :=
sorry

end problem_a1_value_l1585_158539


namespace initial_markers_count_l1585_158576

   -- Let x be the initial number of markers Megan had.
   variable (x : ℕ)

   -- Conditions:
   def robert_gave_109_markers : Prop := true
   def total_markers_after_adding : ℕ := 326
   def markers_added_by_robert : ℕ := 109

   -- The total number of markers Megan has now is 326.
   def total_markers_eq (x : ℕ) : Prop := x + markers_added_by_robert = total_markers_after_adding

   -- Prove that initially Megan had 217 markers.
   theorem initial_markers_count : total_markers_eq 217 := by
     sorry
   
end initial_markers_count_l1585_158576
