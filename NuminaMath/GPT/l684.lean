import Mathlib

namespace Lacy_correct_percent_l684_68465

theorem Lacy_correct_percent (x : ℝ) (h1 : 7 * x > 0) : ((5 * 100) / 7) = 71.43 :=
by
  sorry

end Lacy_correct_percent_l684_68465


namespace probability_of_at_most_one_white_ball_l684_68460

open Nat

-- Definitions based on conditions in a)
def black_balls : ℕ := 10
def red_balls : ℕ := 12
def white_balls : ℕ := 3
def total_balls : ℕ := black_balls + red_balls + white_balls
def select_balls : ℕ := 3

-- The combinatorial function C(n, k) as defined in combinatorics
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Defining the expression and correct answer
def expr : ℚ := (C white_balls 1 * C (black_balls + red_balls) 2 + C (black_balls + red_balls) 3 : ℚ) / (C total_balls 3 : ℚ)
def correct_answer : ℚ := (C white_balls 0 * C (black_balls + red_balls) 3 + C white_balls 1 * C (black_balls + red_balls) 2 : ℚ) / (C total_balls 3 : ℚ)

-- Lean 4 theorem statement
theorem probability_of_at_most_one_white_ball :
  expr = correct_answer := sorry

end probability_of_at_most_one_white_ball_l684_68460


namespace frame_cover_100x100_l684_68422

theorem frame_cover_100x100 :
  ∃! (cover: (ℕ → ℕ → Prop)), (∀ (n : ℕ) (frame: ℕ → ℕ → Prop),
    (∃ (i j : ℕ), (cover (i + n) j ∧ frame (i + n) j ∧ cover (i - n) j ∧ frame (i - n) j) ∧
                   (∃ (k l : ℕ), (cover k (l + n) ∧ frame k (l + n) ∧ cover k (l - n) ∧ frame k (l - n)))) →
    (∃ (i' j' k' l' : ℕ), cover i' j' ∧ frame i' j' ∧ cover k' l' ∧ frame k' l')) :=
sorry

end frame_cover_100x100_l684_68422


namespace ratio_of_triangle_areas_l684_68410

theorem ratio_of_triangle_areas (a k : ℝ) (h_pos_a : 0 < a) (h_pos_k : 0 < k)
    (h_triangle_division : true) (h_square_area : ∃ s, s = a^2) (h_area_one_triangle : ∃ t, t = k * a^2) :
    ∃ r, r = (1 / (4 * k)) :=
by
  sorry

end ratio_of_triangle_areas_l684_68410


namespace isosceles_triangle_side_length_condition_l684_68476

theorem isosceles_triangle_side_length_condition (x y : ℕ) :
    y = x + 1 ∧ 2 * x + y = 16 → (y = 6 → x = 5) :=
by sorry

end isosceles_triangle_side_length_condition_l684_68476


namespace product_of_five_consecutive_integers_not_square_l684_68411

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (h : a > 0) :
  ¬ ∃ k : ℕ, k^2 = a * (a + 1) * (a + 2) * (a + 3) * (a + 4) :=
by
  sorry

end product_of_five_consecutive_integers_not_square_l684_68411


namespace min_even_integers_among_eight_l684_68442

theorem min_even_integers_among_eight :
  ∃ (x y z a b m n o : ℤ), 
    x + y + z = 30 ∧
    x + y + z + a + b = 49 ∧
    x + y + z + a + b + m + n + o = 78 ∧
    (∀ e : ℕ, (∀ x y z a b m n o : ℤ, x + y + z = 30 ∧ x + y + z + a + b = 49 ∧ x + y + z + a + b + m + n + o = 78 → 
    e = 2)) := sorry

end min_even_integers_among_eight_l684_68442


namespace min_value_of_quadratic_l684_68480

noncomputable def quadratic (x : ℝ) : ℝ := x^2 - 8*x + 18

theorem min_value_of_quadratic : ∃ x : ℝ, quadratic x = 2 ∧ (∀ y : ℝ, quadratic y ≥ 2) :=
by
  use 4
  sorry

end min_value_of_quadratic_l684_68480


namespace four_times_sum_of_squares_gt_sum_squared_l684_68475

open Real

theorem four_times_sum_of_squares_gt_sum_squared
  {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) :
  4 * (a^2 + b^2) > (a + b)^2 :=
sorry

end four_times_sum_of_squares_gt_sum_squared_l684_68475


namespace geometric_sequence_sum_l684_68473

variable {a : ℕ → ℝ}

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, (r > 0) ∧ (∀ n : ℕ, a (n + 1) = a n * r)

theorem geometric_sequence_sum
  (a_seq_geometric : is_geometric_sequence a)
  (a_pos : ∀ n : ℕ, a n > 0)
  (eqn : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100) :
  a 4 + a 6 = 10 :=
by
  sorry

end geometric_sequence_sum_l684_68473


namespace weekly_allowance_is_8_l684_68424

variable (A : ℝ)

def condition_1 (A : ℝ) : Prop := ∃ A : ℝ, A / 2 + 8 = 12

theorem weekly_allowance_is_8 (A : ℝ) (h : condition_1 A) : A = 8 :=
sorry

end weekly_allowance_is_8_l684_68424


namespace statement2_true_l684_68448

def digit : ℕ := sorry

def statement1 : Prop := digit = 2
def statement2 : Prop := digit ≠ 3
def statement3 : Prop := digit = 5
def statement4 : Prop := digit ≠ 6

def condition : Prop := (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (¬ statement1 ∨ ¬ statement2 ∨ ¬ statement3 ∨ ¬ statement4)

theorem statement2_true (h : condition) : statement2 :=
sorry

end statement2_true_l684_68448


namespace fraction_to_decimal_l684_68414

theorem fraction_to_decimal :
  (51 / 160 : ℝ) = 0.31875 := 
by
  sorry

end fraction_to_decimal_l684_68414


namespace solve_eq_l684_68484

theorem solve_eq {x : ℝ} (h : x * (x - 1) = x) : x = 0 ∨ x = 2 := 
by {
    sorry
}

end solve_eq_l684_68484


namespace number_of_linear_eqs_l684_68431

def is_linear_eq_in_one_var (eq : String) : Bool :=
  match eq with
  | "0.3x = 1" => true
  | "x/2 = 5x + 1" => true
  | "x = 6" => true
  | _ => false

theorem number_of_linear_eqs :
  let eqs := ["x - 2 = 2 / x", "0.3x = 1", "x/2 = 5x + 1", "x^2 - 4x = 3", "x = 6", "x + 2y = 0"]
  (eqs.filter is_linear_eq_in_one_var).length = 3 :=
by
  sorry

end number_of_linear_eqs_l684_68431


namespace part1_part2_l684_68440

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + (1 + a) * Real.exp (-x)

theorem part1 (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) ↔ a = 0 := by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, 0 < x → f x a ≥ a + 1) → a ≤ 3 := by
  sorry

end part1_part2_l684_68440


namespace no_distinct_roots_exist_l684_68481

theorem no_distinct_roots_exist :
  ¬ ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a^2 - 2 * b * a + c^2 = 0) ∧
  (b^2 - 2 * c * b + a^2 = 0) ∧ 
  (c^2 - 2 * a * c + b^2 = 0) := 
sorry

end no_distinct_roots_exist_l684_68481


namespace probability_of_red_card_l684_68443

theorem probability_of_red_card (successful_attempts not_successful_attempts : ℕ) (h : successful_attempts = 5) (h2 : not_successful_attempts = 8) : (successful_attempts / (successful_attempts + not_successful_attempts) : ℚ) = 5 / 13 := by
  sorry

end probability_of_red_card_l684_68443


namespace number_of_second_graders_l684_68468

-- Define the number of kindergartners, first graders, and total students
def k : ℕ := 14
def f : ℕ := 24
def t : ℕ := 42

-- Define the number of second graders
def s : ℕ := t - (k + f)

-- The theorem to prove
theorem number_of_second_graders : s = 4 := by
  -- We can use sorry here since we are not required to provide the proof
  sorry

end number_of_second_graders_l684_68468


namespace max_rubles_earned_l684_68496

theorem max_rubles_earned :
  ∀ (cards_with_1 cards_with_2 : ℕ), 
  cards_with_1 = 2013 ∧ cards_with_2 = 2013 →
  ∃ (max_moves : ℕ), max_moves = 5 :=
by
  intros cards_with_1 cards_with_2 h
  sorry

end max_rubles_earned_l684_68496


namespace value_of_m_l684_68487

-- Define the condition of the quadratic equation
def quadratic_equation (x m : ℝ) := x^2 - 2*x + m

-- State the equivalence to be proved
theorem value_of_m (m : ℝ) : (∃ x : ℝ, x = 1 ∧ quadratic_equation x m = 0) → m = 1 :=
by
  sorry

end value_of_m_l684_68487


namespace urn_gold_coins_percent_l684_68433

theorem urn_gold_coins_percent (perc_beads : ℝ) (perc_silver_coins : ℝ) (perc_gold_coins : ℝ) :
  perc_beads = 0.2 →
  perc_silver_coins = 0.4 →
  perc_gold_coins = 0.48 :=
by
  intros h1 h2
  sorry

end urn_gold_coins_percent_l684_68433


namespace mass_percentage_O_in_mixture_l684_68479

/-- Mass percentage of oxygen in a mixture of Acetone and Methanol -/
theorem mass_percentage_O_in_mixture 
  (mass_acetone: ℝ)
  (mass_methanol: ℝ)
  (mass_O_acetone: ℝ)
  (mass_O_methanol: ℝ) 
  (total_mass: ℝ) : 
  mass_acetone = 30 → 
  mass_methanol = 20 → 
  mass_O_acetone = (16 / 58.08) * 30 →
  mass_O_methanol = (16 / 32.04) * 20 →
  total_mass = mass_acetone + mass_methanol →
  ((mass_O_acetone + mass_O_methanol) / total_mass) * 100 = 36.52 :=
by
  sorry

end mass_percentage_O_in_mixture_l684_68479


namespace smallest_s_plus_d_l684_68439

theorem smallest_s_plus_d (s d : ℕ) (h_pos_s : s > 0) (h_pos_d : d > 0)
  (h_eq : 1 / s + 1 / (2 * s) + 1 / (3 * s) = 1 / (d^2 - 2 * d)) :
  s + d = 50 :=
sorry

end smallest_s_plus_d_l684_68439


namespace fraction_of_total_money_l684_68498

variable (Max Leevi Nolan Ollie : ℚ)

-- Condition: Each of Max, Leevi, and Nolan gave Ollie the same amount of money
variable (x : ℚ) (h1 : Max / 6 = x) (h2 : Leevi / 3 = x) (h3 : Nolan / 2 = x)

-- Proving that the fraction of the group's (Max, Leevi, Nolan, Ollie) total money possessed by Ollie is 3/11.
theorem fraction_of_total_money (h4 : Max + Leevi + Nolan + Ollie = Max + Leevi + Nolan + 3 * x) : 
  x / (Max + Leevi + Nolan + x) = 3 / 11 := 
by
  sorry

end fraction_of_total_money_l684_68498


namespace cook_weave_l684_68419

theorem cook_weave (Y C W OC CY CYW : ℕ) (hY : Y = 25) (hC : C = 15) (hW : W = 8) (hOC : OC = 2)
  (hCY : CY = 7) (hCYW : CYW = 3) : 
  ∃ (CW : ℕ), CW = 9 :=
by 
  have CW : ℕ := C - OC - (CY - CYW) 
  use CW
  sorry

end cook_weave_l684_68419


namespace line_equation_45_deg_through_point_l684_68402

theorem line_equation_45_deg_through_point :
  ∀ (x y : ℝ), 
  (∃ m k: ℝ, m = 1 ∧ k = 5 ∧ y = m * x + k) ∧ (∃ p q : ℝ, p = -2 ∧ q = 3 ∧ y = q ) :=  
  sorry

end line_equation_45_deg_through_point_l684_68402


namespace m_and_n_must_have_same_parity_l684_68477

-- Define the problem conditions
def square_has_four_colored_edges (square : Type) : Prop :=
  ∃ (colors : Fin 4 → square), true

def m_and_n_same_parity (m n : ℕ) : Prop :=
  (m % 2 = n % 2)

-- Formalize the proof statement based on the conditions
theorem m_and_n_must_have_same_parity (m n : ℕ) (square : Type)
  (H : square_has_four_colored_edges square) : 
  m_and_n_same_parity m n :=
by 
  sorry

end m_and_n_must_have_same_parity_l684_68477


namespace find_parallel_line_through_P_l684_68461

noncomputable def line_parallel_passing_through (p : (ℝ × ℝ)) (line : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, _) := line
  let (x, y) := p
  (a, b, - (a * x + b * y))

theorem find_parallel_line_through_P :
  line_parallel_passing_through (4, -1) (3, -4, 6) = (3, -4, -16) :=
by 
  sorry

end find_parallel_line_through_P_l684_68461


namespace volunteers_correct_l684_68485

-- Definitions of given conditions and the required result
def sheets_per_member : ℕ := 10
def cookies_per_sheet : ℕ := 16
def total_cookies : ℕ := 16000

-- Number of members who volunteered
def members : ℕ := total_cookies / (sheets_per_member * cookies_per_sheet)

-- Proof statement
theorem volunteers_correct :
  members = 100 :=
sorry

end volunteers_correct_l684_68485


namespace fewest_four_dollar_frisbees_l684_68420

theorem fewest_four_dollar_frisbees (x y: ℕ): 
    x + y = 64 ∧ 3 * x + 4 * y = 200 → y = 8 := by sorry

end fewest_four_dollar_frisbees_l684_68420


namespace minibuses_not_enough_l684_68472

def num_students : ℕ := 300
def minibus_capacity : ℕ := 23
def num_minibuses : ℕ := 13

theorem minibuses_not_enough :
  num_minibuses * minibus_capacity < num_students :=
by
  sorry

end minibuses_not_enough_l684_68472


namespace second_person_more_heads_probability_l684_68486

noncomputable def coin_flip_probability (n m : ℕ) : ℚ :=
  if n < m then 1 / 2 else 0

theorem second_person_more_heads_probability :
  coin_flip_probability 10 11 = 1 / 2 :=
by
  sorry

end second_person_more_heads_probability_l684_68486


namespace base7_to_base10_l684_68423

-- Define the base-7 number 521 in base-7
def base7_num : Nat := 5 * 7^2 + 2 * 7^1 + 1 * 7^0

-- State the theorem that needs to be proven
theorem base7_to_base10 : base7_num = 260 :=
by
  -- Proof steps will go here, but we'll skip and insert a sorry for now
  sorry

end base7_to_base10_l684_68423


namespace inradii_sum_l684_68408

theorem inradii_sum (ABCD : Type) (r_a r_b r_c r_d : ℝ) 
  (inscribed_quadrilateral : Prop) 
  (inradius_BCD : Prop) 
  (inradius_ACD : Prop) 
  (inradius_ABD : Prop) 
  (inradius_ABC : Prop) 
  (Tebo_theorem : Prop) :
  r_a + r_c = r_b + r_d := 
by
  sorry

end inradii_sum_l684_68408


namespace cyclic_quadrilateral_l684_68428

theorem cyclic_quadrilateral (T : ℕ) (S : ℕ) (AB BC CD DA : ℕ) (M N : ℝ × ℝ) (AC BD PQ MN : ℝ) (m n : ℕ) :
  T = 2378 → 
  S = 2 + 3 + 7 + 8 → 
  AB = S - 11 → 
  BC = 2 → 
  CD = 3 → 
  DA = 10 → 
  AC * BD = 47 → 
  PQ / MN = 1/2 → 
  m + n = 3 :=
by
  sorry

end cyclic_quadrilateral_l684_68428


namespace mean_of_squares_of_first_four_odd_numbers_l684_68497

theorem mean_of_squares_of_first_four_odd_numbers :
  (1^2 + 3^2 + 5^2 + 7^2) / 4 = 21 := 
by
  sorry

end mean_of_squares_of_first_four_odd_numbers_l684_68497


namespace lindsay_dolls_problem_l684_68467

theorem lindsay_dolls_problem :
  let blonde_dolls := 6
  let brown_dolls := 3 * blonde_dolls
  let black_dolls := brown_dolls / 2
  let red_dolls := 2 * black_dolls
  let combined_dolls := black_dolls + brown_dolls + red_dolls
  combined_dolls - blonde_dolls = 39 :=
by
  sorry

end lindsay_dolls_problem_l684_68467


namespace glycerin_solution_l684_68474

theorem glycerin_solution (x : ℝ) :
    let total_volume := 100
    let final_glycerin_percentage := 0.75
    let volume_first_solution := 75
    let volume_second_solution := 75
    let second_solution_percentage := 0.90
    let final_glycerin_volume := final_glycerin_percentage * total_volume
    let glycerin_second_solution := second_solution_percentage * volume_second_solution
    let glycerin_first_solution := x * volume_first_solution / 100
    glycerin_first_solution + glycerin_second_solution = final_glycerin_volume →
    x = 10 :=
by
    sorry

end glycerin_solution_l684_68474


namespace integer_roots_abs_sum_l684_68451

theorem integer_roots_abs_sum (p q r n : ℤ) :
  (∃ n : ℤ, (∀ x : ℤ, x^3 - 2023 * x + n = 0) ∧ p + q + r = 0 ∧ p * q + q * r + r * p = -2023) →
  |p| + |q| + |r| = 102 :=
by
  sorry

end integer_roots_abs_sum_l684_68451


namespace age_sum_l684_68447

variable {S R K : ℝ}

theorem age_sum 
  (h1 : S = R + 10)
  (h2 : S + 12 = 3 * (R - 5))
  (h3 : K = R / 2) :
  S + R + K = 56.25 := 
by 
  sorry

end age_sum_l684_68447


namespace log_4_135_eq_half_log_2_45_l684_68454

noncomputable def a : ℝ := Real.log 135 / Real.log 4
noncomputable def b : ℝ := Real.log 45 / Real.log 2

theorem log_4_135_eq_half_log_2_45 : a = b / 2 :=
by
  sorry

end log_4_135_eq_half_log_2_45_l684_68454


namespace inequality_neg_multiplication_l684_68456

theorem inequality_neg_multiplication (m n : ℝ) (h : m > n) : -2 * m < -2 * n :=
by {
  sorry
}

end inequality_neg_multiplication_l684_68456


namespace cost_price_is_92_percent_l684_68470

noncomputable def cost_price_percentage_of_selling_price (profit_percentage : ℝ) : ℝ :=
  let CP := (1 / ((profit_percentage / 100) + 1))
  CP * 100

theorem cost_price_is_92_percent (profit_percentage : ℝ) (h : profit_percentage = 8.695652173913043) :
  cost_price_percentage_of_selling_price profit_percentage = 92 :=
by
  rw [h]
  -- now we need to show that cost_price_percentage_of_selling_price 8.695652173913043 = 92
  -- by definition, cost_price_percentage_of_selling_price 8.695652173913043 is:
  -- let CP := 1 / (8.695652173913043 / 100 + 1)
  -- CP * 100 = (1 / (8.695652173913043 / 100 + 1)) * 100
  sorry

end cost_price_is_92_percent_l684_68470


namespace roger_allowance_spend_l684_68418

variable (A m s : ℝ)

-- Conditions from the problem
def condition1 : Prop := m = 0.25 * (A - 2 * s)
def condition2 : Prop := s = 0.10 * (A - 0.5 * m)
def goal : Prop := m + s = 0.59 * A

theorem roger_allowance_spend (h1 : condition1 A m s) (h2 : condition2 A m s) : goal A m s :=
  sorry

end roger_allowance_spend_l684_68418


namespace total_packing_peanuts_used_l684_68445

def large_order_weight : ℕ := 200
def small_order_weight : ℕ := 50
def large_orders_sent : ℕ := 3
def small_orders_sent : ℕ := 4

theorem total_packing_peanuts_used :
  (large_orders_sent * large_order_weight) + (small_orders_sent * small_order_weight) = 800 := 
by
  sorry

end total_packing_peanuts_used_l684_68445


namespace not_true_B_l684_68403

def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem not_true_B (x y : ℝ) : 2 * star x y ≠ star (2 * x) (2 * y) := by
  sorry

end not_true_B_l684_68403


namespace amount_received_is_500_l684_68459

-- Define the conditions
def books_per_month : ℕ := 3
def months_per_year : ℕ := 12
def price_per_book : ℕ := 20
def loss : ℕ := 220

-- Calculate number of books bought in a year
def books_per_year : ℕ := books_per_month * months_per_year

-- Calculate total amount spent on books in a year
def total_spent : ℕ := books_per_year * price_per_book

-- Calculate the amount Jack got from selling the books based on the given loss
def amount_received : ℕ := total_spent - loss

-- Proving the amount received is $500
theorem amount_received_is_500 : amount_received = 500 := by
  sorry

end amount_received_is_500_l684_68459


namespace jim_age_l684_68499

variable (J F S : ℕ)

theorem jim_age (h1 : J = 2 * F) (h2 : F = S + 9) (h3 : J - 6 = 5 * (S - 6)) : J = 46 := 
by
  sorry

end jim_age_l684_68499


namespace union_of_intervals_l684_68407

open Set

variable {α : Type*}

theorem union_of_intervals : 
  let A := Ioc (-1 : ℝ) 1
  let B := Ioo (0 : ℝ) 2
  A ∪ B = Ioo (-1 : ℝ) 2 := 
by
  let A := Ioc (-1 : ℝ) 1
  let B := Ioo (0 : ℝ) 2
  sorry

end union_of_intervals_l684_68407


namespace boys_girls_ratio_l684_68409

-- Definitions used as conditions
variable (B G : ℕ)

-- Conditions
def condition1 : Prop := B + G = 32
def condition2 : Prop := B = 2 * (G - 8)

-- Proof that the ratio of boys to girls initially is 1:1
theorem boys_girls_ratio (h1 : condition1 B G) (h2 : condition2 B G) : (B : ℚ) / G = 1 := by
  sorry

end boys_girls_ratio_l684_68409


namespace unique_solution_sin_tan_eq_l684_68471

noncomputable def S (x : ℝ) : ℝ := Real.tan (Real.sin x) - Real.sin x

theorem unique_solution_sin_tan_eq (h : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.arcsin (1/2) → S x < S y) :
  ∃! x, 0 ≤ x ∧ x ≤ Real.arcsin (1/2) ∧ Real.sin x = Real.tan (Real.sin x) := by
sorry

end unique_solution_sin_tan_eq_l684_68471


namespace parity_of_expression_l684_68466

theorem parity_of_expression (a b c : ℕ) (ha : a % 2 = 1) (hb : b % 2 = 0) :
  (3 ^ a + (b - 1) ^ 2 * (c + 1)) % 2 = if c % 2 = 0 then 1 else 0 :=
by
  sorry

end parity_of_expression_l684_68466


namespace factorization_implies_k_l684_68426

theorem factorization_implies_k (x y k : ℝ) (h : ∃ (a b c d e f : ℝ), 
                            x^3 + 3 * x^2 - 2 * x * y - k * x - 4 * y = (a * x + b * y + c) * (d * x^2 + e * xy + f)) :
  k = -2 :=
sorry

end factorization_implies_k_l684_68426


namespace find_angle_B_find_side_b_l684_68450

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {m n : ℝ × ℝ}
variable {dot_product_max : ℝ}

-- Conditions
def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.sin A + c * Real.sin C - b * Real.sin B = Real.sqrt 2 * a * Real.sin C

def vectors (m n : ℝ × ℝ) := 
  m = (Real.cos A, Real.cos (2 * A)) ∧ n = (12, -5)

def side_length_a (a : ℝ) := 
  a = 4

-- Questions and Proof Problems
theorem find_angle_B (A B C : ℝ) (a b c : ℝ) (h1 : triangle_condition a b c A B C) : 
  B = π / 4 :=
sorry

theorem find_side_b (A B C : ℝ) (a b c : ℝ) 
  (m n : ℝ × ℝ) (max_dot_product_condition : Real.cos A = 3 / 5) 
  (ha : side_length_a a) (hb : b = a * Real.sin B / Real.sin A) : 
  b = 5 * Real.sqrt 2 / 2 :=
sorry

end find_angle_B_find_side_b_l684_68450


namespace triangle_side_solution_l684_68438

/-- 
Given \( a \geq b \geq c > 0 \) and \( a < b + c \), a solution to the equation 
\( b \sqrt{x^{2} - c^{2}} + c \sqrt{x^{2} - b^{2}} = a x \) is provided by 
\( x = \frac{abc}{2 \sqrt{p(p-a)(p-b)(p-c)}} \) where \( p = \frac{1}{2}(a+b+c) \).
-/

theorem triangle_side_solution (a b c x : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a < b + c) :
  b * (Real.sqrt (x^2 - c^2)) + c * (Real.sqrt (x^2 - b^2)) = a * x → 
  x = (a * b * c) / (2 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :=
sorry

end triangle_side_solution_l684_68438


namespace steps_taken_l684_68401

noncomputable def andrewSpeed : ℝ := 1 -- Let Andrew's speed be represented by 1 feet per minute
noncomputable def benSpeed : ℝ := 3 * andrewSpeed -- Ben's speed is 3 times Andrew's speed
noncomputable def totalDistance : ℝ := 21120 -- Distance between the houses in feet
noncomputable def andrewStep : ℝ := 3 -- Each step of Andrew covers 3 feet

theorem steps_taken : (totalDistance / (andrewSpeed + benSpeed)) * andrewSpeed / andrewStep = 1760 := by
  sorry -- proof to be filled in later

end steps_taken_l684_68401


namespace max_men_with_all_items_l684_68449

theorem max_men_with_all_items (total_men married men_with_TV men_with_radio men_with_AC men_with_car men_with_smartphone : ℕ) 
  (H_married : married = 2300) 
  (H_TV : men_with_TV = 2100) 
  (H_radio : men_with_radio = 2600) 
  (H_AC : men_with_AC = 1800) 
  (H_car : men_with_car = 2500) 
  (H_smartphone : men_with_smartphone = 2200) : 
  ∃ m, m ≤ married ∧ m ≤ men_with_TV ∧ m ≤ men_with_radio ∧ m ≤ men_with_AC ∧ m ≤ men_with_car ∧ m ≤ men_with_smartphone ∧ m = 1800 := 
  sorry

end max_men_with_all_items_l684_68449


namespace find_value_of_y_l684_68441

theorem find_value_of_y (x y : ℚ) 
  (h1 : x = 51) 
  (h2 : x^3 * y - 2 * x^2 * y + x * y = 63000) : 
  y = 8 / 17 := 
by 
  sorry

end find_value_of_y_l684_68441


namespace sum_of_cubes_ratio_l684_68463

theorem sum_of_cubes_ratio (a b c d e f : ℝ) 
  (h1 : a + b + c = 0) (h2 : d + e + f = 0) :
  (a^3 + b^3 + c^3) / (d^3 + e^3 + f^3) = (a * b * c) / (d * e * f) := 
by 
  sorry

end sum_of_cubes_ratio_l684_68463


namespace blue_balls_count_l684_68464

theorem blue_balls_count (Y B : ℕ) (h_ratio : 4 * B = 3 * Y) (h_total : Y + B = 35) : B = 15 :=
sorry

end blue_balls_count_l684_68464


namespace quadratic_discriminant_constraint_l684_68469

theorem quadratic_discriminant_constraint (c : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 4*x1 + c = 0 ∧ x2^2 - 4*x2 + c = 0) ↔ c < 4 := 
by
  sorry

end quadratic_discriminant_constraint_l684_68469


namespace convex_pentagon_largest_angle_l684_68434

theorem convex_pentagon_largest_angle 
  (x : ℝ)
  (h1 : (x + 2) + (2 * x + 3) + (3 * x + 6) + (4 * x + 5) + (5 * x + 4) = 540) :
  5 * x + 4 = 532 / 3 :=
by
  sorry

end convex_pentagon_largest_angle_l684_68434


namespace maximum_marks_l684_68429

theorem maximum_marks (M : ℝ) (h : 0.5 * M = 50 + 10) : M = 120 :=
by
  sorry

end maximum_marks_l684_68429


namespace magnitude_of_power_l684_68462

-- Given conditions
def z : ℂ := 3 + 2 * Complex.I
def n : ℕ := 6

-- Mathematical statement to prove
theorem magnitude_of_power :
  Complex.abs (z ^ n) = 2197 :=
by
  sorry

end magnitude_of_power_l684_68462


namespace point_reflection_l684_68417

-- Define the original point and the reflection function
structure Point where
  x : ℝ
  y : ℝ

def reflect_y_axis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

-- Define the original point
def M : Point := ⟨-5, 2⟩

-- State the theorem to prove the reflection
theorem point_reflection : reflect_y_axis M = ⟨5, 2⟩ :=
  sorry

end point_reflection_l684_68417


namespace person_speed_approx_l684_68458

noncomputable def convertDistance (meters : ℝ) : ℝ := meters * 0.000621371
noncomputable def convertTime (minutes : ℝ) (seconds : ℝ) : ℝ := (minutes + (seconds / 60)) / 60
noncomputable def calculateSpeed (distance_miles : ℝ) (time_hours : ℝ) : ℝ := distance_miles / time_hours

theorem person_speed_approx (street_length_meters : ℝ) (time_min : ℝ) (time_sec : ℝ) :
  street_length_meters = 900 →
  time_min = 3 →
  time_sec = 20 →
  abs ((calculateSpeed (convertDistance street_length_meters) (convertTime time_min time_sec)) - 10.07) < 0.01 :=
by
  sorry

end person_speed_approx_l684_68458


namespace trigonometric_identity_l684_68432

theorem trigonometric_identity
  (x : ℝ) 
  (h_tan : Real.tan x = -1/2) :
  (3 * Real.sin x ^ 2 - 2) / (Real.sin x * Real.cos x) = 7 / 2 := 
by
  sorry

end trigonometric_identity_l684_68432


namespace total_number_of_crayons_l684_68455

def number_of_blue_crayons := 3
def number_of_red_crayons := 4 * number_of_blue_crayons
def number_of_green_crayons := 2 * number_of_red_crayons
def number_of_yellow_crayons := number_of_green_crayons / 2

theorem total_number_of_crayons :
  number_of_blue_crayons + number_of_red_crayons + number_of_green_crayons + number_of_yellow_crayons = 51 :=
by 
  -- Proof is not required
  sorry

end total_number_of_crayons_l684_68455


namespace shorter_leg_length_l684_68425

theorem shorter_leg_length (a b c : ℝ) (h1 : b = 10) (h2 : a^2 + b^2 = c^2) (h3 : c = 2 * a) : 
  a = 10 * Real.sqrt 3 / 3 :=
by
  sorry

end shorter_leg_length_l684_68425


namespace determine_a_l684_68494

theorem determine_a (a b c : ℤ)
  (vertex_condition : ∀ x : ℝ, x = 2 → ∀ y : ℝ, y = -3 → y = a * (x - 2) ^ 2 - 3)
  (point_condition : ∀ x : ℝ, x = 1 → ∀ y : ℝ, y = -2 → y = a * (x - 2) ^ 2 - 3) :
  a = 1 :=
by
  sorry

end determine_a_l684_68494


namespace train_speed_kmph_l684_68490

/-- Define the lengths of the train and bridge, as well as the time taken to cross the bridge. --/
def train_length : ℝ := 150
def bridge_length : ℝ := 150
def crossing_time_seconds : ℝ := 29.997600191984642

/-- Calculate the speed of the train in km/h. --/
theorem train_speed_kmph : 
  let total_distance := train_length + bridge_length
  let time_in_hours := crossing_time_seconds / 3600
  let speed_mph := total_distance / time_in_hours
  let speed_kmph := speed_mph / 1000
  speed_kmph = 36 := by
  /- Proof omitted -/
  sorry

end train_speed_kmph_l684_68490


namespace shares_difference_l684_68400

theorem shares_difference (x : ℝ) (h_ratio : 2.5 * x + 3.5 * x + 7.5 * x + 9.8 * x = (23.3 * x))
  (h_difference : 7.5 * x - 3.5 * x = 4500) : 9.8 * x - 2.5 * x = 8212.5 :=
by
  sorry

end shares_difference_l684_68400


namespace sarah_meets_vegetable_requirement_l684_68415

def daily_vegetable_requirement : ℝ := 2
def total_days : ℕ := 5
def weekly_requirement : ℝ := daily_vegetable_requirement * total_days

def sunday_consumption : ℝ := 3
def monday_consumption : ℝ := 1.5
def tuesday_consumption : ℝ := 1.5
def wednesday_consumption : ℝ := 1.5
def thursday_consumption : ℝ := 2.5

def total_consumption : ℝ := sunday_consumption + monday_consumption + tuesday_consumption + wednesday_consumption + thursday_consumption

theorem sarah_meets_vegetable_requirement : total_consumption = weekly_requirement :=
by
  sorry

end sarah_meets_vegetable_requirement_l684_68415


namespace hair_cut_length_l684_68478

-- Definitions corresponding to the conditions in the problem
def initial_length : ℕ := 18
def current_length : ℕ := 9

-- Statement to prove
theorem hair_cut_length : initial_length - current_length = 9 :=
by
  sorry

end hair_cut_length_l684_68478


namespace tetrahedron_through_hole_tetrahedron_cannot_through_hole_l684_68452

/--
A regular tetrahedron with edge length 1 can pass through a circular hole if and only if the radius \( R \) is at least 0.4478, given that the thickness of the hole can be neglected.
-/

theorem tetrahedron_through_hole (R : ℝ) (h1 : R = 0.45) : true :=
by sorry

theorem tetrahedron_cannot_through_hole (R : ℝ) (h1 : R = 0.44) : false :=
by sorry

end tetrahedron_through_hole_tetrahedron_cannot_through_hole_l684_68452


namespace solution_to_inequality_l684_68421

theorem solution_to_inequality (x : ℝ) :
  (∃ y : ℝ, y = x^(1/3) ∧ y + 3 / (y + 2) ≤ 0) ↔ x < -8 := 
sorry

end solution_to_inequality_l684_68421


namespace complex_square_l684_68491

theorem complex_square (a b : ℤ) (i : ℂ) (h1: a = 5) (h2: b = 3) (h3: i^2 = -1) :
  ((↑a) + (↑b) * i)^2 = 16 + 30 * i := by
  sorry

end complex_square_l684_68491


namespace count_false_propositions_l684_68406

theorem count_false_propositions 
  (P : Prop) 
  (inverse_P : Prop) 
  (negation_P : Prop) 
  (converse_P : Prop) 
  (h1 : ¬P) 
  (h2 : inverse_P) 
  (h3 : negation_P ↔ ¬P) 
  (h4 : converse_P ↔ P) : 
  ∃ n : ℕ, n = 2 ∧ 
  ¬P ∧ ¬converse_P ∧ 
  inverse_P ∧ negation_P := 
sorry

end count_false_propositions_l684_68406


namespace power_of_integer_is_two_l684_68444

-- Definitions based on conditions
def is_power_of_integer (n : ℕ) : Prop :=
  ∃ (k : ℕ) (m : ℕ), n = m^k

-- Given conditions translated to Lean definitions
def g : ℕ := 14
def n : ℕ := 3150 * g

-- The proof problem statement in Lean
theorem power_of_integer_is_two (h : g = 14) : is_power_of_integer n :=
sorry

end power_of_integer_is_two_l684_68444


namespace find_a_l684_68430

theorem find_a (a : ℝ) (h1 : 1 < a) (h2 : 1 + a = 3) : a = 2 :=
sorry

end find_a_l684_68430


namespace number_of_math_players_l684_68413

theorem number_of_math_players (total_players physics_players both_players : ℕ)
    (h1 : total_players = 25)
    (h2 : physics_players = 15)
    (h3 : both_players = 6)
    (h4 : total_players = physics_players + (total_players - physics_players - (total_players - physics_players - both_players)) + both_players ) :
  total_players - (physics_players - both_players) = 16 :=
sorry

end number_of_math_players_l684_68413


namespace polygon_sides_from_diagonals_l684_68435

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l684_68435


namespace ben_fewer_pints_than_kathryn_l684_68488

-- Define the conditions
def annie_picked := 8
def kathryn_picked := annie_picked + 2
def total_picked := 25

-- Add noncomputable because constants are involved
noncomputable def ben_picked : ℕ := total_picked - (annie_picked + kathryn_picked)

theorem ben_fewer_pints_than_kathryn : ben_picked = kathryn_picked - 3 := 
by 
  -- The problem statement does not require proof body
  sorry

end ben_fewer_pints_than_kathryn_l684_68488


namespace quadratic_to_completed_square_l684_68483

-- Define the given quadratic function.
def quadratic_function (x : ℝ) : ℝ := x^2 + 2 * x - 2

-- Define the completed square form of the function.
def completed_square_form (x : ℝ) : ℝ := (x + 1)^2 - 3

-- The theorem statement that needs to be proven.
theorem quadratic_to_completed_square :
  ∀ x : ℝ, quadratic_function x = completed_square_form x :=
by sorry

end quadratic_to_completed_square_l684_68483


namespace quadratic_roots_value_r_l684_68404

theorem quadratic_roots_value_r
  (a b m p r : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h_root1 : a^2 - m*a + 3 = 0)
  (h_root2 : b^2 - m*b + 3 = 0)
  (h_ab : a * b = 3)
  (h_root3 : (a + 1/b) * (b + 1/a) = r) :
  r = 16 / 3 :=
sorry

end quadratic_roots_value_r_l684_68404


namespace not_perfect_cube_of_N_l684_68405

-- Define a twelve-digit number
def N : ℕ := 100000000000

-- Define the condition that a number is a perfect cube
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℤ, n = k ^ 3

-- Problem statement: Prove that 100000000000 is not a perfect cube
theorem not_perfect_cube_of_N : ¬ is_perfect_cube N :=
by sorry

end not_perfect_cube_of_N_l684_68405


namespace inequality_relationship_l684_68446

variable (a b : ℝ)

theorem inequality_relationship
  (h1 : a < 0)
  (h2 : -1 < b ∧ b < 0) : a < a * b^2 ∧ a * b^2 < a * b :=
by
  sorry

end inequality_relationship_l684_68446


namespace range_of_x_l684_68427

-- Problem Statement
theorem range_of_x (x : ℝ) (h : 0 ≤ x - 8) : 8 ≤ x :=
by {
  sorry
}

end range_of_x_l684_68427


namespace ratio_of_socks_l684_68482

theorem ratio_of_socks (y p : ℝ) (h1 : 5 * p + y * 2 * p = 5 * p + 4 * y * p / 3) :
  (5 : ℝ) / y = 11 / 2 :=
by
  sorry

end ratio_of_socks_l684_68482


namespace inequality_solution_ab_l684_68492

theorem inequality_solution_ab (a b : ℝ) (h : ∀ x : ℝ, 2 < x ∧ x < 4 ↔ |x + a| < b) : a * b = -3 := 
by
  sorry

end inequality_solution_ab_l684_68492


namespace range_of_a_and_m_l684_68437

open Set

-- Definitions of the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 1 = 0}

-- Conditions as hypotheses
def condition1 : A ∪ B a = A := sorry
def condition2 : A ∩ C m = C m := sorry

-- Theorem to prove the correct range of a and m
theorem range_of_a_and_m : (a = 2 ∨ a = 3) ∧ (-2 < m ∧ m ≤ 2) :=
by
  -- Proof goes here
  sorry

end range_of_a_and_m_l684_68437


namespace min_value_ineq_solve_ineq_l684_68489

theorem min_value_ineq (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / a^3 + 1 / b^3 + 1 / c^3 + 3 * a * b * c) ≥ 6 :=
sorry

theorem solve_ineq (x : ℝ) (h : |x + 1| - 2 * x < 6) : x > -7/3 :=
sorry

end min_value_ineq_solve_ineq_l684_68489


namespace Lois_books_total_l684_68436

-- Definitions based on the conditions
def initial_books : ℕ := 150
def books_given_to_nephew : ℕ := initial_books / 4
def remaining_books : ℕ := initial_books - books_given_to_nephew
def non_fiction_books : ℕ := remaining_books * 60 / 100
def kept_non_fiction_books : ℕ := non_fiction_books / 2
def fiction_books : ℕ := remaining_books - non_fiction_books
def lent_fiction_books : ℕ := fiction_books / 3
def remaining_fiction_books : ℕ := fiction_books - lent_fiction_books
def newly_purchased_books : ℕ := 12

-- The total number of books Lois has now
def total_books_now : ℕ := kept_non_fiction_books + remaining_fiction_books + newly_purchased_books

-- Theorem statement
theorem Lois_books_total : total_books_now = 76 := by
  sorry

end Lois_books_total_l684_68436


namespace find_q_l684_68493

variable {a d q : ℝ}
variables (M N : Set ℝ)

theorem find_q (hM : M = {a, a + d, a + 2 * d}) 
              (hN : N = {a, a * q, a * q^2})
              (ha : a ≠ 0)
              (heq : M = N) :
  q = -1 / 2 :=
sorry

end find_q_l684_68493


namespace length_squared_t_graph_interval_l684_68495

noncomputable def p (x : ℝ) : ℝ := -x + 2
noncomputable def q (x : ℝ) : ℝ := x + 2
noncomputable def r (x : ℝ) : ℝ := 2
noncomputable def t (x : ℝ) : ℝ :=
  if x ≤ -2 then p x
  else if x ≤ 2 then r x
  else q x

theorem length_squared_t_graph_interval :
  let segment_length (f : ℝ → ℝ) (a b : ℝ) : ℝ := Real.sqrt ((f b - f a)^2 + (b - a)^2)
  segment_length t (-4) (-2) + segment_length t (-2) 2 + segment_length t 2 4 = 4 + 2 * Real.sqrt 32 →
  (4 + 2 * Real.sqrt 32)^2 = 80 :=
sorry

end length_squared_t_graph_interval_l684_68495


namespace phone_extension_permutations_l684_68457

theorem phone_extension_permutations : 
  (∃ (l : List ℕ), l = [5, 7, 8, 9, 0] ∧ Nat.factorial l.length = 120) :=
sorry

end phone_extension_permutations_l684_68457


namespace original_children_count_l684_68412

theorem original_children_count (x : ℕ) (h1 : 46800 / x + 1950 = 46800 / (x - 2))
    : x = 8 :=
sorry

end original_children_count_l684_68412


namespace value_of_x_l684_68453

theorem value_of_x (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 12) : x = 9 :=
by
  sorry

end value_of_x_l684_68453


namespace magnitude_of_b_l684_68416

open Real

noncomputable def a : ℝ × ℝ := (-sqrt 3, 1)

theorem magnitude_of_b (b : ℝ × ℝ)
    (h1 : (a.1 + 2 * b.1, a.2 + 2 * b.2) = (a.1, a.2))
    (h2 : (a.1 + b.1, a.2 + b.2) = (b.1, b.2)) :
    sqrt (b.1 ^ 2 + b.2 ^ 2) = sqrt 2 :=
sorry

end magnitude_of_b_l684_68416
