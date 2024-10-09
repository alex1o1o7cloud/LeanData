import Mathlib

namespace karlanna_marble_problem_l1395_139589

theorem karlanna_marble_problem : 
  ∃ (m_values : Finset ℕ), 
  (∀ m ∈ m_values, ∃ n : ℕ, m * n = 450 ∧ m > 1 ∧ n > 1) ∧ 
  m_values.card = 16 := 
by
  sorry

end karlanna_marble_problem_l1395_139589


namespace unique_k_value_l1395_139530
noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 2 ≤ m ∧ m ∣ n → m = n

theorem unique_k_value :
  (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 74 ∧ p * q = 213) ∧
  ∀ (p₁ q₁ k₁ p₂ q₂ k₂ : ℕ),
    is_prime p₁ ∧ is_prime q₁ ∧ p₁ + q₁ = 74 ∧ p₁ * q₁ = k₁ ∧
    is_prime p₂ ∧ is_prime q₂ ∧ p₂ + q₂ = 74 ∧ p₂ * q₂ = k₂ →
    k₁ = k₂ :=
by
  sorry

end unique_k_value_l1395_139530


namespace multiply_by_nine_l1395_139568

theorem multiply_by_nine (x : ℝ) (h : 9 * x = 36) : x = 4 :=
sorry

end multiply_by_nine_l1395_139568


namespace solve_first_equation_solve_second_equation_l1395_139523

theorem solve_first_equation (x : ℝ) : 3 * (x - 2)^2 - 27 = 0 ↔ x = 5 ∨ x = -1 :=
by {
  sorry
}

theorem solve_second_equation (x : ℝ) : 2 * (x + 1)^3 + 54 = 0 ↔ x = -4 :=
by {
  sorry
}

end solve_first_equation_solve_second_equation_l1395_139523


namespace candy_necklaces_per_pack_l1395_139516

theorem candy_necklaces_per_pack (packs_total packs_opened packs_left candies_left necklaces_per_pack : ℕ) 
  (h_total : packs_total = 9) 
  (h_opened : packs_opened = 4) 
  (h_left : packs_left = packs_total - packs_opened) 
  (h_candies_left : candies_left = 40) 
  (h_necklaces_per_pack : candies_left = packs_left * necklaces_per_pack) :
  necklaces_per_pack = 8 :=
by
  -- Proof goes here
  sorry

end candy_necklaces_per_pack_l1395_139516


namespace no_valid_coloring_l1395_139526

open Nat

-- Define the color type
inductive Color
| blue
| red
| green

-- Define the coloring function
def color : ℕ → Color := sorry

-- Define the properties of the coloring function
def valid_coloring (color : ℕ → Color) : Prop :=
  ∀ (m n : ℕ), m > 1 → n > 1 → color m ≠ color n → 
    color (m * n) ≠ color m ∧ color (m * n) ≠ color n

-- Theorem: It is not possible to color all natural numbers greater than 1 as described
theorem no_valid_coloring : ¬ ∃ (color : ℕ → Color), valid_coloring color :=
by
  sorry

end no_valid_coloring_l1395_139526


namespace abs_inequality_range_l1395_139561

theorem abs_inequality_range (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) ↔ a ≤ 4 := 
sorry

end abs_inequality_range_l1395_139561


namespace bill_left_with_22_l1395_139555

def bill_earnings (ounces : ℕ) (rate_per_ounce : ℕ) : ℕ :=
  ounces * rate_per_ounce

def bill_remaining_money (total_earnings : ℕ) (fine : ℕ) : ℕ :=
  total_earnings - fine

theorem bill_left_with_22 (ounces sold_rate fine total_remaining : ℕ)
  (h1 : ounces = 8)
  (h2 : sold_rate = 9)
  (h3 : fine = 50)
  (h4 : total_remaining = 22)
  : bill_remaining_money (bill_earnings ounces sold_rate) fine = total_remaining :=
by
  sorry

end bill_left_with_22_l1395_139555


namespace vasya_wins_l1395_139509

/-
  Petya and Vasya are playing a game where initially there are 2022 boxes, 
  each containing exactly one matchstick. In one move, a player can transfer 
  all matchsticks from one non-empty box to another non-empty box. They take turns, 
  with Petya starting first. The winner is the one who, after their move, has 
  at least half of all the matchsticks in one box for the first time. 

  We want to prove that Vasya will win the game with the optimal strategy.
-/

theorem vasya_wins : true :=
  sorry -- placeholder for the actual proof

end vasya_wins_l1395_139509


namespace hyperbola_asymptote_slopes_l1395_139596

theorem hyperbola_asymptote_slopes:
  (∀ (x y : ℝ), (x^2 / 144 - y^2 / 81 = 1) → (y = (3 / 4) * x ∨ y = -(3 / 4) * x)) :=
by
  sorry

end hyperbola_asymptote_slopes_l1395_139596


namespace operation_result_l1395_139546

def operation (a b : ℤ) : ℤ := a * (b + 2) + a * b

theorem operation_result : operation 3 (-1) = 0 :=
by
  sorry

end operation_result_l1395_139546


namespace cyc_inequality_l1395_139591

theorem cyc_inequality (x y z : ℝ) (hx : 0 < x ∧ x < 2) (hy : 0 < y ∧ y < 2) (hz : 0 < z ∧ z < 2) 
  (hxyz : x^2 + y^2 + z^2 = 3) : 
  3 / 2 < (1 + y^2) / (x + 2) + (1 + z^2) / (y + 2) + (1 + x^2) / (z + 2) ∧ 
  (1 + y^2) / (x + 2) + (1 + z^2) / (y + 2) + (1 + x^2) / (z + 2) < 3 := 
by
  sorry

end cyc_inequality_l1395_139591


namespace square_of_radius_l1395_139543

theorem square_of_radius 
  (AP PB CQ QD : ℝ) 
  (hAP : AP = 25)
  (hPB : PB = 35)
  (hCQ : CQ = 30)
  (hQD : QD = 40) 
  : ∃ r : ℝ, r^2 = 13325 := 
sorry

end square_of_radius_l1395_139543


namespace max_popsicles_is_13_l1395_139511

/-- Pablo's budgets and prices for buying popsicles. -/
structure PopsicleStore where
  single_popsicle_cost : ℕ
  three_popsicle_box_cost : ℕ
  five_popsicle_box_cost : ℕ
  starting_budget : ℕ

/-- The maximum number of popsicles Pablo can buy given the store's prices and his budget. -/
def maxPopsicles (store : PopsicleStore) : ℕ :=
  let num_five_popsicle_boxes := store.starting_budget / store.five_popsicle_box_cost
  let remaining_after_five_boxes := store.starting_budget % store.five_popsicle_box_cost
  let num_three_popsicle_boxes := remaining_after_five_boxes / store.three_popsicle_box_cost
  let remaining_after_three_boxes := remaining_after_five_boxes % store.three_popsicle_box_cost
  let num_single_popsicles := remaining_after_three_boxes / store.single_popsicle_cost
  num_five_popsicle_boxes * 5 + num_three_popsicle_boxes * 3 + num_single_popsicles

theorem max_popsicles_is_13 :
  maxPopsicles { single_popsicle_cost := 1, 
                 three_popsicle_box_cost := 2, 
                 five_popsicle_box_cost := 3, 
                 starting_budget := 8 } = 13 := by
  sorry

end max_popsicles_is_13_l1395_139511


namespace root_condition_l1395_139524

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x + m * x + m

theorem root_condition (m l : ℝ) (h : m < l) : 
  (∀ x : ℝ, f x m = 0 → x ≠ x) ∨ (∃ x : ℝ, f x m = 0) :=
sorry

end root_condition_l1395_139524


namespace sum_of_coefficients_l1395_139545

theorem sum_of_coefficients (a_5 a_4 a_3 a_2 a_1 a_0 : ℤ) :
  (x-2)^5 = a_5*x^5 + a_4*x^4 + a_3*x^3 + a_2*x^2 + a_1*x + a_0 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  sorry

end sum_of_coefficients_l1395_139545


namespace units_digit_of_x_l1395_139559

theorem units_digit_of_x (p x : ℕ): 
  (p * x = 32 ^ 10) → 
  (p % 10 = 6) → 
  (x % 4 = 0) → 
  (x % 10 = 1) :=
by
  sorry

end units_digit_of_x_l1395_139559


namespace train_overtakes_motorbike_time_l1395_139517

theorem train_overtakes_motorbike_time :
  let train_speed_kmph := 100
  let motorbike_speed_kmph := 64
  let train_length_m := 120.0096
  let relative_speed_kmph := train_speed_kmph - motorbike_speed_kmph
  let relative_speed_m_s := (relative_speed_kmph : ℝ) * (1 / 3.6)
  let time_seconds := train_length_m / relative_speed_m_s
  time_seconds = 12.00096 :=
sorry

end train_overtakes_motorbike_time_l1395_139517


namespace samantha_birth_year_l1395_139534

theorem samantha_birth_year
  (first_amc8_year : ℕ := 1985)
  (held_annually : ∀ (n : ℕ), n ≥ 0 → first_amc8_year + n = 1985 + n)
  (samantha_age_7th_amc8 : ℕ := 12) :
  ∃ (birth_year : ℤ), birth_year = 1979 :=
by
  sorry

end samantha_birth_year_l1395_139534


namespace compare_mixed_decimal_l1395_139590

def mixed_number_value : ℚ := -2 - 1 / 3  -- Representation of -2 1/3 as a rational number
def decimal_value : ℚ := -2.3             -- Representation of -2.3 as a rational number

theorem compare_mixed_decimal : mixed_number_value < decimal_value :=
sorry

end compare_mixed_decimal_l1395_139590


namespace average_weight_when_D_joins_is_53_l1395_139584

noncomputable def new_average_weight (A B C D E : ℕ) : ℕ :=
  (73 + B + C + D) / 4

theorem average_weight_when_D_joins_is_53 :
  (A + B + C) / 3 = 50 →
  A = 73 →
  (B + C + D + E) / 4 = 51 →
  E = D + 3 →
  73 + B + C + D = 212 →
  new_average_weight A B C D E = 53 :=
by
  sorry

end average_weight_when_D_joins_is_53_l1395_139584


namespace amount_per_person_is_correct_l1395_139505

-- Define the total amount and the number of people
def total_amount : ℕ := 2400
def number_of_people : ℕ := 9

-- State the main theorem to be proved
theorem amount_per_person_is_correct : total_amount / number_of_people = 266 := 
by sorry

end amount_per_person_is_correct_l1395_139505


namespace no_such_real_numbers_l1395_139547

noncomputable def have_integer_roots (a b c : ℝ) : Prop :=
  ∃ r s : ℤ, a * (r:ℝ)^2 + b * r + c = 0 ∧ a * (s:ℝ)^2 + b * s + c = 0

theorem no_such_real_numbers (a b c : ℝ) :
  have_integer_roots a b c → have_integer_roots (a + 1) (b + 1) (c + 1) → False :=
by
  -- proof will go here
  sorry

end no_such_real_numbers_l1395_139547


namespace ratio_of_sums_l1395_139538

noncomputable def sum_of_squares (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def square_of_sum (n : ℕ) : ℚ :=
  ((n * (n + 1)) / 2) ^ 2

theorem ratio_of_sums (n : ℕ) (h : n = 25) :
  sum_of_squares n / square_of_sum n = 1 / 19 :=
by
  have hn : n = 25 := h
  rw [hn]
  dsimp [sum_of_squares, square_of_sum]
  have : (25 * (25 + 1) * (2 * 25 + 1)) / 6 = 5525 := by norm_num
  have : ((25 * (25 + 1)) / 2) ^ 2 = 105625 := by norm_num
  norm_num
  sorry

end ratio_of_sums_l1395_139538


namespace find_c_value_l1395_139566

theorem find_c_value (x y n m c : ℕ) 
  (h1 : 10 * x + y = 8 * n) 
  (h2 : 10 + x + y = 9 * m) 
  (h3 : c = x + y) : 
  c = 8 := 
by
  sorry

end find_c_value_l1395_139566


namespace length_of_rectangular_plot_l1395_139542

variable (L : ℕ)

-- Given conditions
def width := 50
def poles := 14
def distance_between_poles := 20
def intervals := poles - 1
def perimeter := intervals * distance_between_poles

-- The perimeter of the rectangle in terms of length and width
def rectangle_perimeter := 2 * (L + width)

-- The main statement to be proven
theorem length_of_rectangular_plot :
  rectangle_perimeter L = perimeter → L = 80 :=
by
  sorry

end length_of_rectangular_plot_l1395_139542


namespace uncovered_side_length_l1395_139569

theorem uncovered_side_length (L W : ℝ) (h1 : L * W = 120) (h2 : L + 2 * W = 32) : L = 20 :=
sorry

end uncovered_side_length_l1395_139569


namespace books_sale_correct_l1395_139537

variable (books_original books_left : ℕ)

def books_sold (books_original books_left : ℕ) : ℕ :=
  books_original - books_left

theorem books_sale_correct : books_sold 108 66 = 42 := by
  -- Since there is no need for the solution steps, we can assert the proof
  sorry

end books_sale_correct_l1395_139537


namespace visual_range_increase_l1395_139535

def percent_increase (original new : ℕ) : ℕ :=
  ((new - original) * 100) / original

theorem visual_range_increase :
  percent_increase 50 150 = 200 := 
by
  -- the proof would go here
  sorry

end visual_range_increase_l1395_139535


namespace minimum_turns_to_exceed_1000000_l1395_139552

theorem minimum_turns_to_exceed_1000000 :
  let a : Fin 5 → ℕ := fun n => if n = 0 then 1 else 0
  (∀ n : ℕ, ∃ (b_2 b_3 b_4 b_5 : ℕ),
    a 4 + b_2 ≥ 0 ∧
    a 3 + b_3 ≥ 0 ∧
    a 2 + b_4 ≥ 0 ∧
    a 1 + b_5 ≥ 0 ∧
    b_2 * b_3 * b_4 * b_5 > 1000000 →
    b_2 + b_3 + b_4 + b_5 = n) → 
    ∃ n, n = 127 :=
by
  sorry

end minimum_turns_to_exceed_1000000_l1395_139552


namespace stream_speed_l1395_139581

theorem stream_speed (x : ℝ) (hb : ∀ t, t = 48 / (20 + x) → t = 24 / (20 - x)) : x = 20 / 3 :=
by
  have t := hb (48 / (20 + x)) rfl
  sorry

end stream_speed_l1395_139581


namespace ce_over_de_l1395_139522

theorem ce_over_de {A B C D E T : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ (A →ₗ[ℝ] B)]
  {AT DT BT ET CE DE : ℝ}
  (h1 : AT / DT = 2)
  (h2 : BT / ET = 3) :
  CE / DE = 1 / 2 := 
sorry

end ce_over_de_l1395_139522


namespace problem_solution_l1395_139513

def f (x : ℕ) : ℝ := sorry

axiom f_add_eq_mul (p q : ℕ) : f (p + q) = f p * f q
axiom f_one_eq_three : f 1 = 3

theorem problem_solution :
  (f 1 ^ 2 + f 2) / f 1 + 
  (f 2 ^ 2 + f 4) / f 3 + 
  (f 3 ^ 2 + f 6) / f 5 + 
  (f 4 ^ 2 + f 8) / f 7 = 24 := 
by
  sorry

end problem_solution_l1395_139513


namespace num_pairs_satisfying_equation_l1395_139518

theorem num_pairs_satisfying_equation :
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ x^2 - y^2 = 204 :=
by
  sorry

end num_pairs_satisfying_equation_l1395_139518


namespace sufficient_but_not_necessary_l1395_139558

theorem sufficient_but_not_necessary (a b : ℝ) (h1 : a > 2) (h2 : b > 1) : 
  (a + b > 3 ∧ a * b > 2) ∧ ∃ x y : ℝ, (x + y > 3 ∧ x * y > 2) ∧ (¬ (x > 2 ∧ y > 1)) :=
by 
  sorry

end sufficient_but_not_necessary_l1395_139558


namespace original_price_of_cycle_l1395_139594

theorem original_price_of_cycle 
    (selling_price : ℝ) 
    (loss_percentage : ℝ) 
    (h1 : selling_price = 1120)
    (h2 : loss_percentage = 0.20) : 
    ∃ P : ℝ, P = 1400 :=
by
  sorry

end original_price_of_cycle_l1395_139594


namespace fraction_identity_l1395_139580

noncomputable def calc_fractions (x y : ℝ) : ℝ :=
  (x + y) / (x - y)

theorem fraction_identity (x y : ℝ) (h : (1/x + 1/y) / (1/x - 1/y) = 1001) : calc_fractions x y = -1001 :=
by
  sorry

end fraction_identity_l1395_139580


namespace expression_equals_20_over_9_l1395_139531

noncomputable def complex_fraction_expression := 
  let a := 11 + 1 / 9
  let b := 3 + 2 / 5
  let c := 1 + 2 / 17
  let d := 8 + 2 / 5
  let e := 3.6
  let f := 2 + 6 / 25
  ((a - b * c) - d / e) / f

theorem expression_equals_20_over_9 : complex_fraction_expression = 20 / 9 :=
by
  sorry

end expression_equals_20_over_9_l1395_139531


namespace range_of_f_at_most_7_l1395_139520

theorem range_of_f_at_most_7 (f : ℤ × ℤ → ℝ)
  (H : ∀ (x y m n : ℤ), f (x + 3 * m - 2 * n, y - 4 * m + 5 * n) = f (x, y)) :
  ∃ (s : Finset ℝ), s.card ≤ 7 ∧ ∀ (a : ℤ × ℤ), f a ∈ s :=
sorry

end range_of_f_at_most_7_l1395_139520


namespace find_C_coordinates_l1395_139557

noncomputable def pointC_coordinates : Prop :=
  let A : (ℝ × ℝ) := (-2, 1)
  let B : (ℝ × ℝ) := (4, 9)
  ∃ C : (ℝ × ℝ), 
    (dist (A.1, A.2) (C.1, C.2) = 2 * dist (B.1, B.2) (C.1, C.2)) ∧ 
    C = (2, 19 / 3)

theorem find_C_coordinates : pointC_coordinates :=
  sorry

end find_C_coordinates_l1395_139557


namespace total_eggs_sold_l1395_139548

def initial_trays : Nat := 10
def dropped_trays : Nat := 2
def added_trays : Nat := 7
def eggs_per_tray : Nat := 36

theorem total_eggs_sold : initial_trays - dropped_trays + added_trays * eggs_per_tray = 540 := by
  sorry

end total_eggs_sold_l1395_139548


namespace girls_boys_difference_l1395_139507

variables (B G : ℕ) (x : ℕ)

-- Condition that relates boys and girls with a ratio
def ratio_condition : Prop := 3 * x = B ∧ 4 * x = G

-- Condition that the total number of students is 42
def total_students_condition : Prop := B + G = 42

-- We want to prove that the difference between the number of girls and boys is 6
theorem girls_boys_difference (h_ratio : ratio_condition B G x) (h_total : total_students_condition B G) : 
  G - B = 6 :=
sorry

end girls_boys_difference_l1395_139507


namespace solve_for_a_b_c_l1395_139560

-- Conditions and necessary context
def m_angle_A : ℝ := 60  -- In degrees
def BC_length : ℝ := 12  -- Length of BC in units
def angle_DBC_eq_three_times_angle_ECB (DBC ECB : ℝ) : Prop := DBC = 3 * ECB

-- Definitions for perpendicularity could be checked by defining angles
-- between lines, but we can assert these as properties.
axiom BD_perpendicular_AC : Prop
axiom CE_perpendicular_AB : Prop

-- The proof problem
theorem solve_for_a_b_c :
  ∃ (EC a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  b ≠ c ∧ 
  (∀ d, b ∣ d → d = b ∨ d = 1) ∧ 
  (∀ d, c ∣ d → d = c ∨ d = 1) ∧
  EC = a * (Real.sqrt b + Real.sqrt c) ∧ 
  a + b + c = 11 :=
by
  sorry

end solve_for_a_b_c_l1395_139560


namespace fifth_term_sequence_l1395_139500

theorem fifth_term_sequence 
  (a : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h2 : a 2 = 6)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 5 = -6 := 
by
  sorry

end fifth_term_sequence_l1395_139500


namespace no_third_quadrant_l1395_139503

theorem no_third_quadrant {a b : ℝ} (h1 : 0 < a) (h2 : a < 1) (h3 : -1 < b) : ∀ x y : ℝ, (y = a^x + b) → ¬ (x < 0 ∧ y < 0) :=
by
  intro x y h
  sorry

end no_third_quadrant_l1395_139503


namespace work_done_by_force_l1395_139586

def force : ℝ × ℝ := (-1, -2)
def displacement : ℝ × ℝ := (3, 4)

def work_done (F S : ℝ × ℝ) : ℝ :=
  F.1 * S.1 + F.2 * S.2

theorem work_done_by_force :
  work_done force displacement = -11 := 
by
  sorry

end work_done_by_force_l1395_139586


namespace angle_PQC_in_triangle_l1395_139541

theorem angle_PQC_in_triangle 
  (A B C P Q: ℝ)
  (h_in_triangle: A + B + C = 180)
  (angle_B_exterior_bisector: ∀ B_ext, B_ext = 180 - B →  angle_B = 90 - B / 2)
  (angle_C_exterior_bisector: ∀ C_ext, C_ext = 180 - C →  angle_C = 90 - C / 2)
  (h_PQ_BC_angle: ∀ PQ_angle BC_angle, PQ_angle = 30 → BC_angle = 30) :
  ∃ PQC_angle, PQC_angle = (180 - A) / 2 :=
by
  sorry

end angle_PQC_in_triangle_l1395_139541


namespace intervals_of_monotonicity_interval_max_min_l1395_139599

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x - 2

theorem intervals_of_monotonicity :
  (∀ (x : ℝ), x < -1 → deriv f x < 0) ∧ 
  (∀ (x : ℝ), -1 < x ∧ x < 3 → deriv f x > 0) ∧ 
  (∀ (x : ℝ), x > 3 → deriv f x < 0) := 
sorry

theorem interval_max_min :
  f 2 = 20 → f (-1) = -7 := 
sorry

end intervals_of_monotonicity_interval_max_min_l1395_139599


namespace intersection_first_quadrant_l1395_139595

theorem intersection_first_quadrant (a : ℝ) : 
  (∃ x y : ℝ, (ax + y = 4) ∧ (x - y = 2) ∧ (0 < x) ∧ (0 < y)) ↔ (-1 < a ∧ a < 2) :=
by
  sorry

end intersection_first_quadrant_l1395_139595


namespace constructible_triangle_l1395_139540

theorem constructible_triangle (k c delta : ℝ) (h1 : 2 * c < k) :
  ∃ (a b : ℝ), a + b + c = k ∧ a + b > c ∧ ∃ (α β : ℝ), α - β = delta :=
by
  sorry

end constructible_triangle_l1395_139540


namespace smallest_possible_positive_value_l1395_139588

theorem smallest_possible_positive_value (a b : ℤ) (h : a > b) :
  ∃ (x : ℚ), x = (a + b) / (a - b) + (a - b) / (a + b) ∧ x = 2 :=
sorry

end smallest_possible_positive_value_l1395_139588


namespace second_divisor_203_l1395_139563

theorem second_divisor_203 (x : ℕ) (h1 : 210 % 13 = 3) (h2 : 210 % x = 7) : x = 203 :=
by sorry

end second_divisor_203_l1395_139563


namespace days_A_worked_l1395_139521

theorem days_A_worked (W : ℝ) (x : ℝ) (hA : W / 15 * x = W - 6 * (W / 9))
  (hB : W = 6 * (W / 9)) : x = 5 :=
sorry

end days_A_worked_l1395_139521


namespace simplify_expression_correct_l1395_139501

variable (a b x y : ℝ) (i : ℂ)

noncomputable def simplify_expression (a b x y : ℝ) (i : ℂ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) ∧ (i^2 = -1) → (a * x + b * i * y) * (a * x - b * i * y) = a^2 * x^2 + b^2 * y^2

theorem simplify_expression_correct (a b x y : ℝ) (i : ℂ) :
  simplify_expression a b x y i := by
  sorry

end simplify_expression_correct_l1395_139501


namespace cube_axes_of_symmetry_regular_tetrahedron_axes_of_symmetry_l1395_139549

-- Definitions for geometric objects
def cube : Type := sorry
def regular_tetrahedron : Type := sorry

-- Definitions for axes of symmetry
def axes_of_symmetry (shape : Type) : Nat := sorry

-- Theorem statements
theorem cube_axes_of_symmetry : axes_of_symmetry cube = 13 := 
by 
  sorry

theorem regular_tetrahedron_axes_of_symmetry : axes_of_symmetry regular_tetrahedron = 7 :=
by 
  sorry

end cube_axes_of_symmetry_regular_tetrahedron_axes_of_symmetry_l1395_139549


namespace total_area_of_colored_paper_l1395_139506

-- Definitions
def num_pieces : ℝ := 3.2
def side_length : ℝ := 8.5

-- Theorem statement
theorem total_area_of_colored_paper : 
  let area_one_piece := side_length * side_length
  let total_area := area_one_piece * num_pieces
  total_area = 231.2 := by
  sorry

end total_area_of_colored_paper_l1395_139506


namespace internet_plan_cost_effective_l1395_139598

theorem internet_plan_cost_effective (d : ℕ) :
  (∀ (d : ℕ), d > 150 → 1500 + 10 * d < 20 * d) ↔ d = 151 :=
sorry

end internet_plan_cost_effective_l1395_139598


namespace tangent_line_to_circle_l1395_139544

noncomputable def r_tangent_to_circle : ℝ := 4

theorem tangent_line_to_circle
  (x y r : ℝ)
  (circle_eq : x^2 + y^2 = 2 * r)
  (line_eq : x - y = r) :
  r = r_tangent_to_circle :=
by
  sorry

end tangent_line_to_circle_l1395_139544


namespace g_at_minus_six_l1395_139556

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_at_minus_six : g (-6) = 43 / 16 := by
  sorry

end g_at_minus_six_l1395_139556


namespace hourly_rate_for_carriage_l1395_139510

theorem hourly_rate_for_carriage
  (d : ℕ) (s : ℕ) (f : ℕ) (c : ℕ)
  (h_d : d = 20)
  (h_s : s = 10)
  (h_f : f = 20)
  (h_c : c = 80) :
  (c - f) / (d / s) = 30 := by
  sorry

end hourly_rate_for_carriage_l1395_139510


namespace base_5_to_base_10_conversion_l1395_139565

/-- An alien creature communicated that it produced 263_5 units of a resource. 
    Convert this quantity to base 10. -/
theorem base_5_to_base_10_conversion : ∀ (n : ℕ), n = 2 * 5^2 + 6 * 5^1 + 3 * 5^0 → n = 83 :=
by
  intros n h
  rw [h]
  sorry

end base_5_to_base_10_conversion_l1395_139565


namespace volume_ratio_of_cubes_l1395_139583

theorem volume_ratio_of_cubes (e1 e2 : ℕ) (h1 : e1 = 9) (h2 : e2 = 36) :
  (e1^3 : ℚ) / (e2^3 : ℚ) = 1 / 64 := by
  sorry

end volume_ratio_of_cubes_l1395_139583


namespace largest_factor_of_form_l1395_139579

theorem largest_factor_of_form (n : ℕ) (h : n % 10 = 4) : 120 ∣ n * (n + 1) * (n + 2) :=
sorry

end largest_factor_of_form_l1395_139579


namespace solid_circles_2006_l1395_139585

noncomputable def circlePattern : Nat → Nat
| n => (2 + n * (n + 3)) / 2

theorem solid_circles_2006 :
  ∃ n, circlePattern n < 2006 ∧ circlePattern (n + 1) > 2006 ∧ n = 61 :=
by
  sorry

end solid_circles_2006_l1395_139585


namespace problem_solution_set_l1395_139562

theorem problem_solution_set (a b : ℝ) (h : ∀ x, (1 < x ∧ x < 2) ↔ ax^2 + x + b > 0) : a + b = -1 :=
sorry

end problem_solution_set_l1395_139562


namespace dancer_count_l1395_139553

theorem dancer_count (n : ℕ) : 
  ((n + 5) % 12 = 0) ∧ ((n + 5) % 10 = 0) ∧ (200 ≤ n) ∧ (n ≤ 300) → (n = 235 ∨ n = 295) := 
by
  sorry

end dancer_count_l1395_139553


namespace louisa_average_speed_l1395_139576

-- Problem statement
theorem louisa_average_speed :
  ∃ v : ℝ, (250 / v * v = 250 ∧ 350 / v * v = 350) ∧ ((350 / v) = (250 / v) + 3) ∧ v = 100 / 3 := by
  sorry

end louisa_average_speed_l1395_139576


namespace distance_between_A_and_B_l1395_139536

-- Given conditions as definitions

def total_time : ℝ := 4
def boat_speed : ℝ := 7.5
def stream_speed : ℝ := 2.5
def distance_AC : ℝ := 10

-- Define the possible solutions for the distance between A and B
def distance_AB (x : ℝ) := 
  (x / (boat_speed + stream_speed) + (x + distance_AC) / (boat_speed - stream_speed) = total_time) 
  ∨ 
  (x / (boat_speed + stream_speed) + (x - distance_AC) / (boat_speed - stream_speed) = total_time)

-- Problem statement
theorem distance_between_A_and_B :
  ∃ x : ℝ, (distance_AB x) ∧ (x = 20 ∨ x = 20 / 3) :=
sorry

end distance_between_A_and_B_l1395_139536


namespace max_value_of_b_over_a_squared_l1395_139582

variables {a b x y : ℝ}

def triangle_is_right (a b x y : ℝ) : Prop :=
  (a - x)^2 + (b - y)^2 = a^2 + b^2

theorem max_value_of_b_over_a_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b)
    (h4 : ∃ x y, a^2 + y^2 = b^2 + x^2 
                 ∧ b^2 + x^2 = (a - x)^2 + (b - y)^2
                 ∧ 0 ≤ x ∧ x < a 
                 ∧ 0 ≤ y ∧ y < b 
                 ∧ triangle_is_right a b x y) 
    : (b / a)^2 = 4 / 3 :=
sorry

end max_value_of_b_over_a_squared_l1395_139582


namespace work_together_days_l1395_139567

theorem work_together_days (a_days : ℕ) (b_days : ℕ) :
  a_days = 10 → b_days = 9 → (1 / ((1 / (a_days : ℝ)) + (1 / (b_days : ℝ)))) = 90 / 19 :=
by
  intros ha hb
  sorry

end work_together_days_l1395_139567


namespace intersection_of_A_and_B_l1395_139578

def A : Set ℝ := { x | 0 < x }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | 0 < x ∧ x ≤ 1 } := 
sorry

end intersection_of_A_and_B_l1395_139578


namespace fulfill_customer_order_in_nights_l1395_139575

structure JerkyCompany where
  batch_size : ℕ
  nightly_batches : ℕ

def customerOrder (ordered : ℕ) (current_stock : ℕ) : ℕ :=
  ordered - current_stock

def batchesNeeded (required : ℕ) (batch_size : ℕ) : ℕ :=
  required / batch_size

def daysNeeded (batches_needed : ℕ) (nightly_batches : ℕ) : ℕ :=
  batches_needed / nightly_batches

theorem fulfill_customer_order_in_nights :
  ∀ (ordered current_stock : ℕ) (jc : JerkyCompany),
    jc.batch_size = 10 →
    jc.nightly_batches = 1 →
    ordered = 60 →
    current_stock = 20 →
    daysNeeded (batchesNeeded (customerOrder ordered current_stock) jc.batch_size) jc.nightly_batches = 4 :=
by
  intros ordered current_stock jc h1 h2 h3 h4
  sorry

end fulfill_customer_order_in_nights_l1395_139575


namespace num_foxes_l1395_139533

structure Creature :=
  (is_squirrel : Bool)
  (is_fox : Bool)
  (is_salamander : Bool)

def Anna : Creature := sorry
def Bob : Creature := sorry
def Cara : Creature := sorry
def Daniel : Creature := sorry

def tells_truth (c : Creature) : Bool :=
  c.is_squirrel || (c.is_salamander && ¬c.is_fox)

def Anna_statement : Prop := Anna.is_fox ≠ Daniel.is_fox
def Bob_statement : Prop := tells_truth Bob ↔ Cara.is_salamander
def Cara_statement : Prop := tells_truth Cara ↔ Bob.is_fox
def Daniel_statement : Prop := tells_truth Daniel ↔ (Anna.is_squirrel ∧ Bob.is_squirrel ∧ Cara.is_squirrel ∨ Daniel.is_squirrel)

theorem num_foxes :
  (Anna.is_fox + Bob.is_fox + Cara.is_fox + Daniel.is_fox = 2) :=
  sorry

end num_foxes_l1395_139533


namespace ratio_blue_to_gold_l1395_139574

-- Define the number of brown stripes
def brown_stripes : Nat := 4

-- Given condition: There are three times as many gold stripes as brown stripes
def gold_stripes : Nat := 3 * brown_stripes

-- Given condition: There are 60 blue stripes
def blue_stripes : Nat := 60

-- The actual statement to prove
theorem ratio_blue_to_gold : blue_stripes / gold_stripes = 5 := by
  -- Proof would go here
  sorry

end ratio_blue_to_gold_l1395_139574


namespace sum_of_roots_quadratic_specific_sum_of_roots_l1395_139504

theorem sum_of_roots_quadratic:
  ∀ a b c : ℚ, a ≠ 0 → 
  ∀ x1 x2 : ℚ, (a * x1^2 + b * x1 + c = 0) ∧ 
               (a * x2^2 + b * x2 + c = 0) → 
               x1 + x2 = -b / a := 
by
  sorry

theorem specific_sum_of_roots:
  ∀ x1 x2 : ℚ, (12 * x1^2 + 19 * x1 - 21 = 0) ∧ 
               (12 * x2^2 + 19 * x2 - 21 = 0) → 
               x1 + x2 = -19 / 12 := 
by
  sorry

end sum_of_roots_quadratic_specific_sum_of_roots_l1395_139504


namespace base_10_to_base_7_conversion_l1395_139514

theorem base_10_to_base_7_conversion :
  ∃ (digits : ℕ → ℕ), 789 = digits 3 * 7^3 + digits 2 * 7^2 + digits 1 * 7^1 + digits 0 * 7^0 ∧
  digits 3 = 2 ∧ digits 2 = 2 ∧ digits 1 = 0 ∧ digits 0 = 5 :=
sorry

end base_10_to_base_7_conversion_l1395_139514


namespace problem_l1395_139515

theorem problem (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 6 = 976 :=
by
  sorry

end problem_l1395_139515


namespace bob_gave_terry_24_bushels_l1395_139571

def bushels_given_to_terry (total_bushels : ℕ) (ears_per_bushel : ℕ) (ears_left : ℕ) : ℕ :=
    (total_bushels * ears_per_bushel - ears_left) / ears_per_bushel

theorem bob_gave_terry_24_bushels : bushels_given_to_terry 50 14 357 = 24 := by
    sorry

end bob_gave_terry_24_bushels_l1395_139571


namespace neha_amount_removed_l1395_139550

theorem neha_amount_removed (N S M : ℝ) (x : ℝ) (total_amnt : ℝ) (M_val : ℝ) (ratio2 : ℝ) (ratio8 : ℝ) (ratio6 : ℝ) :
  total_amnt = 1100 →
  M_val = 102 →
  ratio2 = 2 →
  ratio8 = 8 →
  ratio6 = 6 →
  (M - 4 = ratio6 * x) →
  (S - 8 = ratio8 * x) →
  (N - (N - (ratio2 * x)) = ratio2 * x) →
  (N + S + M = total_amnt) →
  (N - 32.66 = N - (ratio2 * (total_amnt - M_val - 8 - 4) / (ratio2 + ratio8 + ratio6))) →
  N - (N - (ratio2 * ((total_amnt - M_val - 8 - 4) / (ratio2 + ratio8 + ratio6)))) = 826.70 :=
by
  intros
  sorry

end neha_amount_removed_l1395_139550


namespace sum_of_digits_l1395_139564

variables {a b c d : ℕ}

theorem sum_of_digits (h1 : ∀ (x y z w : ℕ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
                      (h2 : c + a = 10)
                      (h3 : b + c = 9)
                      (h4 : a + d = 10) :
  a + b + c + d = 18 :=
sorry

end sum_of_digits_l1395_139564


namespace new_profit_percentage_l1395_139573

def original_cost (c : ℝ) : ℝ := c
def original_selling_price (c : ℝ) : ℝ := 1.2 * c
def new_cost (c : ℝ) : ℝ := 0.9 * c
def new_selling_price (c : ℝ) : ℝ := 1.05 * 1.2 * c

theorem new_profit_percentage (c : ℝ) (hc : c > 0) :
  ((new_selling_price c - new_cost c) / new_cost c) * 100 = 40 :=
by
  sorry

end new_profit_percentage_l1395_139573


namespace total_apples_l1395_139539

def pinky_apples : ℕ := 36
def danny_apples : ℕ := 73

theorem total_apples :
  pinky_apples + danny_apples = 109 :=
by
  sorry

end total_apples_l1395_139539


namespace non_parallel_lines_a_l1395_139554

theorem non_parallel_lines_a (a : ℝ) :
  ¬ (a * -(1 / (a+2))) = a →
  ¬ (-1 / (a+2)) = 2 →
  a = 0 ∨ a = -3 :=
by
  sorry

end non_parallel_lines_a_l1395_139554


namespace triangle_right_l1395_139525

theorem triangle_right (a b c : ℝ) (h₀ : a ≠ c) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : ∃ x₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + 2 * a * x₀ + b^2 = 0 ∧ x₀^2 + 2 * c * x₀ - b^2 = 0) :
  a^2 = b^2 + c^2 := 
sorry

end triangle_right_l1395_139525


namespace right_triangle_hypotenuse_l1395_139508

theorem right_triangle_hypotenuse {a b c : ℝ} 
  (h1: a + b + c = 60) 
  (h2: a * b = 96) 
  (h3: a^2 + b^2 = c^2) : 
  c = 28.4 := 
sorry

end right_triangle_hypotenuse_l1395_139508


namespace incorrect_transformation_when_c_zero_l1395_139593

theorem incorrect_transformation_when_c_zero {a b c : ℝ} (h : a * c = b * c) (hc : c = 0) : a ≠ b :=
by
  sorry

end incorrect_transformation_when_c_zero_l1395_139593


namespace bakery_storage_l1395_139587

theorem bakery_storage (S F B : ℕ) 
  (h1 : S * 4 = F * 5) 
  (h2 : F = 10 * B) 
  (h3 : F * 1 = (B + 60) * 8) : S = 3000 :=
sorry

end bakery_storage_l1395_139587


namespace lost_revenue_is_correct_l1395_139551

-- Define the ticket prices
def general_admission_price : ℤ := 10
def children_price : ℤ := 6
def senior_price : ℤ := 8
def veteran_discount : ℤ := 2

-- Define the number of tickets sold
def general_tickets_sold : ℤ := 20
def children_tickets_sold : ℤ := 3
def senior_tickets_sold : ℤ := 4
def veteran_tickets_sold : ℤ := 2

-- Calculate the actual revenue from sold tickets
def actual_revenue := (general_tickets_sold * general_admission_price) + 
                      (children_tickets_sold * children_price) + 
                      (senior_tickets_sold * senior_price) + 
                      (veteran_tickets_sold * (general_admission_price - veteran_discount))

-- Define the maximum potential revenue assuming all tickets are sold at general admission price
def max_potential_revenue : ℤ := 50 * general_admission_price

-- Define the potential revenue lost
def potential_revenue_lost := max_potential_revenue - actual_revenue

-- The theorem to prove
theorem lost_revenue_is_correct : potential_revenue_lost = 234 := 
by
  -- Placeholder for proof
  sorry

end lost_revenue_is_correct_l1395_139551


namespace compute_expression_l1395_139577

noncomputable def given_cubic (x : ℝ) : Prop :=
  x ^ 3 - 7 * x ^ 2 + 12 * x = 18

theorem compute_expression (a b c : ℝ) (ha : given_cubic a) (hb : given_cubic b) (hc : given_cubic c) :
  (a + b + c = 7) → 
  (a * b + b * c + c * a = 12) → 
  (a * b * c = 18) → 
  (a * b / c + b * c / a + c * a / b = -6) :=
by 
  sorry

end compute_expression_l1395_139577


namespace intersection_point_of_lines_l1395_139502

theorem intersection_point_of_lines :
  ∃ x y : ℝ, 3 * x + 4 * y - 2 = 0 ∧ 2 * x + y + 2 = 0 := 
by
  sorry

end intersection_point_of_lines_l1395_139502


namespace total_bike_count_l1395_139519

def total_bikes (bikes_jungkook bikes_yoongi : Nat) : Nat :=
  bikes_jungkook + bikes_yoongi

theorem total_bike_count : total_bikes 3 4 = 7 := 
  by 
  sorry

end total_bike_count_l1395_139519


namespace ellipse_properties_l1395_139592

noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x * x) / (a * a) + (y * y) / (b * b) = 1

theorem ellipse_properties (a b c k : ℝ) (h_ab : a > b) (h_b : b > 1) (h_c : 2 * c = 2) 
  (h_area : (2 * Real.sqrt 3 / 3)^2 = 4 / 3) (h_slope : k ≠ 0)
  (h_PD : |(c - 4 * k^2 / (3 + 4 * k^2))^2 + (-3 * k / (3 + 4 * k^2))^2| = 3 * Real.sqrt 2 / 7) :
  (ellipse_equation 1 0 a b ∧
   (a = 2 ∧ b = Real.sqrt 3) ∧
   k = 1 ∨ k = -1) :=
by
  -- Prove the standard equation of the ellipse C and the value of k
  sorry

end ellipse_properties_l1395_139592


namespace sum_of_fifth_powers_l1395_139597

variable (a b c d : ℝ)

theorem sum_of_fifth_powers (h₁ : a + b = c + d) (h₂ : a^3 + b^3 = c^3 + d^3) :
  a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l1395_139597


namespace x0_equals_pm1_l1395_139572

-- Define the function f and its second derivative
def f (x : ℝ) : ℝ := x^3
def f'' (x : ℝ) : ℝ := 6 * x

-- Prove that if f''(x₀) = 6 then x₀ = ±1
theorem x0_equals_pm1 (x0 : ℝ) (h : f'' x0 = 6) : x0 = 1 ∨ x0 = -1 :=
by
  sorry

end x0_equals_pm1_l1395_139572


namespace star_4_3_l1395_139529

def star (a b : ℕ) : ℕ := a^2 + a * b - b^3

theorem star_4_3 : star 4 3 = 1 := 
by
  -- sorry is used to skip the proof
  sorry

end star_4_3_l1395_139529


namespace car_sales_total_l1395_139532

theorem car_sales_total (a b c : ℕ) (h1 : a = 14) (h2 : b = 16) (h3 : c = 27):
  a + b + c = 57 :=
by
  repeat {rwa [h1, h2, h3]}
  sorry

end car_sales_total_l1395_139532


namespace one_of_sum_of_others_l1395_139528

theorem one_of_sum_of_others (a b c : ℝ) 
  (cond1 : |a - b| ≥ |c|)
  (cond2 : |b - c| ≥ |a|)
  (cond3 : |c - a| ≥ |b|) :
  (a = b + c) ∨ (b = c + a) ∨ (c = a + b) :=
by
  sorry

end one_of_sum_of_others_l1395_139528


namespace rem_l1395_139527

def rem' (x y : ℚ) : ℚ := x - y * (⌊ x / (2 * y) ⌋)

theorem rem'_value : rem' (5 / 9 : ℚ) (-3 / 7) = 62 / 63 := by
  sorry

end rem_l1395_139527


namespace sum_of_digits_2_2010_mul_5_2012_mul_7_l1395_139512

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_2_2010_mul_5_2012_mul_7 : 
  sum_of_digits (2^2010 * 5^2012 * 7) = 13 :=
by {
  sorry
}

end sum_of_digits_2_2010_mul_5_2012_mul_7_l1395_139512


namespace remainder_of_power_of_five_modulo_500_l1395_139570

theorem remainder_of_power_of_five_modulo_500 :
  (5 ^ (5 ^ (5 ^ 2))) % 500 = 25 :=
by
  sorry

end remainder_of_power_of_five_modulo_500_l1395_139570
