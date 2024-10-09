import Mathlib

namespace min_value_problem_l1315_131598

theorem min_value_problem (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 57 * a + 88 * b + 125 * c ≥ 1148) :
  240 ≤ a^3 + b^3 + c^3 + 5 * a^2 + 5 * b^2 + 5 * c^2 :=
sorry

end min_value_problem_l1315_131598


namespace find_px_value_l1315_131567

noncomputable def p (a b c x : ℤ) := a * x^2 + b * x + c

theorem find_px_value {a b c : ℤ} 
  (h1 : p a b c 2 = 2) 
  (h2 : p a b c (-2) = -2) 
  (h3 : p a b c 9 = 3) 
  (h : a = -2 / 11) 
  (h4 : b = 1)
  (h5 : c = 8 / 11) :
  p a b c 14 = -230 / 11 :=
by
  sorry

end find_px_value_l1315_131567


namespace downstream_speed_l1315_131538

variable (Vu : ℝ) (Vs : ℝ)

theorem downstream_speed (h1 : Vu = 25) (h2 : Vs = 35) : (2 * Vs - Vu = 45) :=
by
  sorry

end downstream_speed_l1315_131538


namespace base_conversion_proof_l1315_131514

-- Definitions of the base-converted numbers
def b1463_7 := 3 * 7^0 + 6 * 7^1 + 4 * 7^2 + 1 * 7^3  -- 1463 in base 7
def b121_5 := 1 * 5^0 + 2 * 5^1 + 1 * 5^2  -- 121 in base 5
def b1754_6 := 4 * 6^0 + 5 * 6^1 + 7 * 6^2 + 1 * 6^3  -- 1754 in base 6
def b3456_7 := 6 * 7^0 + 5 * 7^1 + 4 * 7^2 + 3 * 7^3  -- 3456 in base 7

-- Formalizing the proof goal
theorem base_conversion_proof : (b1463_7 / b121_5 : ℤ) - b1754_6 * 2 + b3456_7 = 278 := by
  sorry  -- Proof is omitted

end base_conversion_proof_l1315_131514


namespace inequality_solution_l1315_131510

theorem inequality_solution (m : ℝ) (x : ℝ) (hm : 0 ≤ m ∧ m ≤ 1) (ineq : m * x^2 - 2 * x - m ≥ 2) : x ≤ -1 :=
sorry

end inequality_solution_l1315_131510


namespace halfway_miles_proof_l1315_131516

def groceries_miles : ℕ := 10
def haircut_miles : ℕ := 15
def doctor_miles : ℕ := 5

def total_miles : ℕ := groceries_miles + haircut_miles + doctor_miles

theorem halfway_miles_proof : total_miles / 2 = 15 := by
  -- calculation to follow
  sorry

end halfway_miles_proof_l1315_131516


namespace drive_time_from_city_B_to_city_A_l1315_131574

theorem drive_time_from_city_B_to_city_A
  (t : ℝ)
  (round_trip_distance : ℝ := 360)
  (saved_time_per_trip : ℝ := 0.5)
  (average_speed : ℝ := 80) :
  (80 * ((3 + t) - 2 * 0.5)) = 360 → t = 2.5 :=
by
  intro h
  sorry

end drive_time_from_city_B_to_city_A_l1315_131574


namespace find_period_l1315_131596

theorem find_period (A P R : ℕ) (I : ℕ) (T : ℚ) 
  (hA : A = 1120) 
  (hP : P = 896) 
  (hR : R = 5) 
  (hSI : I = A - P) 
  (hT : I = (P * R * T) / 100) :
  T = 5 := by 
  sorry

end find_period_l1315_131596


namespace system_of_linear_equations_m_l1315_131550

theorem system_of_linear_equations_m (x y m : ℝ) :
  (2 * x + y = 1 + 2 * m) →
  (x + 2 * y = 2 - m) →
  (x + y > 0) →
  ((2 * m + 1) * x - 2 * m < 1) →
  (x > 1) →
  (-3 < m ∧ m < -1/2) ∧ (m = -2 ∨ m = -1) :=
by
  intros h1 h2 h3 h4 h5
  -- Placeholder for proof steps
  sorry

end system_of_linear_equations_m_l1315_131550


namespace geom_seq_a3_a5_product_l1315_131544

-- Defining the conditions: a sequence and its sum formula
def geom_seq (a : ℕ → ℕ) := ∃ r : ℕ, ∀ n, a (n+1) = a n * r

def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = 2^(n-1) + a 1

-- The theorem statement
theorem geom_seq_a3_a5_product (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : geom_seq a) (h2 : sum_first_n_terms a S) : a 3 * a 5 = 16 := 
sorry

end geom_seq_a3_a5_product_l1315_131544


namespace ratio_sum_eq_seven_eight_l1315_131566

theorem ratio_sum_eq_seven_eight 
  (a b c x y z : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a * x + b * y + c * z = 56) :
  (a + b + c) / (x + y + z) = 7/8 :=
by
  sorry

end ratio_sum_eq_seven_eight_l1315_131566


namespace xy_sufficient_not_necessary_l1315_131563

theorem xy_sufficient_not_necessary (x y : ℝ) :
  (xy_lt_zero : x * y < 0) → abs (x - y) = abs x + abs y ∧ (abs (x - y) = abs x + abs y → x * y ≥ 0) := 
by
  sorry

end xy_sufficient_not_necessary_l1315_131563


namespace largest_nonrepresentable_by_17_11_l1315_131541

/--
In the USA, standard letter-size paper is 8.5 inches wide and 11 inches long. The largest integer that cannot be written as a sum of a whole number (possibly zero) of 17's and a whole number (possibly zero) of 11's is 159.
-/
theorem largest_nonrepresentable_by_17_11 : 
  ∀ (a b : ℕ), (∀ (n : ℕ), n = 17 * a + 11 * b -> n ≠ 159) ∧ 
               ¬ (∃ (a b : ℕ), 17 * a + 11 * b = 159) :=
by
  sorry

end largest_nonrepresentable_by_17_11_l1315_131541


namespace initial_mixture_volume_l1315_131536

/--
Given:
1. A mixture initially contains 20% water.
2. When 13.333333333333334 liters of water is added, water becomes 25% of the new mixture.

Prove that the initial volume of the mixture is 200 liters.
-/
theorem initial_mixture_volume (V : ℝ) (h1 : V > 0) (h2 : 0.20 * V + 13.333333333333334 = 0.25 * (V + 13.333333333333334)) : V = 200 :=
sorry

end initial_mixture_volume_l1315_131536


namespace part1_l1315_131560

def setA (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

def setB (x : ℝ) : Prop := x ≠ 0 ∧ x ≤ 5 ∧ 0 < x

def setC (a x : ℝ) : Prop := 3 * a ≤ x ∧ x ≤ 2 * a + 1

def setInter (x : ℝ) : Prop := setA x ∧ setB x

theorem part1 (a : ℝ) : (∀ x, setC a x → setInter x) ↔ (0 < a ∧ a ≤ 1 / 2 ∨ 1 < a) :=
sorry

end part1_l1315_131560


namespace regular_tickets_sold_l1315_131503

variables (S R : ℕ) (h1 : S + R = 65) (h2 : 10 * S + 15 * R = 855)

theorem regular_tickets_sold : R = 41 :=
sorry

end regular_tickets_sold_l1315_131503


namespace oldest_sibling_age_difference_l1315_131589

theorem oldest_sibling_age_difference 
  (D : ℝ) 
  (avg_age : ℝ) 
  (hD : D = 25.75) 
  (h_avg : avg_age = 30) :
  ∃ A : ℝ, (A - D ≥ 17) :=
by
  sorry

end oldest_sibling_age_difference_l1315_131589


namespace quadruplets_satisfy_l1315_131549

-- Define the condition in the problem
def equation (x y z w : ℝ) : Prop :=
  1 + (1 / x) + (2 * (x + 1) / (x * y)) + (3 * (x + 1) * (y + 2) / (x * y * z)) + (4 * (x + 1) * (y + 2) * (z + 3) / (x * y * z * w)) = 0

-- State the theorem
theorem quadruplets_satisfy (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  equation x y z w ↔ (x = -1 ∨ y = -2 ∨ z = -3 ∨ w = -4) :=
by
  sorry

end quadruplets_satisfy_l1315_131549


namespace range_of_a_l1315_131580

-- Definitions of sets A and B
def A (x : ℝ) : Prop := 1 < |x - 2| ∧ |x - 2| < 2
def B (x a : ℝ) : Prop := x^2 - (a + 1) * x + a < 0

-- The condition A ∩ B ≠ ∅
def nonempty_intersection (a : ℝ) : Prop := ∃ x : ℝ, A x ∧ B x a

-- Proving the required range of a
theorem range_of_a : {a : ℝ | nonempty_intersection a} = {a : ℝ | a < 1 ∨ a > 3} := by
  sorry

end range_of_a_l1315_131580


namespace canoes_vs_kayaks_l1315_131542

theorem canoes_vs_kayaks (C K : ℕ) (h1 : 9 * C + 12 * K = 432) (h2 : C = 4 * K / 3) : C - K = 6 :=
sorry

end canoes_vs_kayaks_l1315_131542


namespace math_proof_l1315_131532

theorem math_proof :
  ∀ (x y z : ℚ), (2 * x - 3 * y - 2 * z = 0) →
                  (x + 3 * y - 28 * z = 0) →
                  (z ≠ 0) →
                  (x^2 + 3 * x * y * z) / (y^2 + z^2) = 280 / 37 :=
by
  intros x y z h1 h2 h3
  sorry

end math_proof_l1315_131532


namespace jeffrey_fills_crossword_l1315_131524

noncomputable def prob_fill_crossword : ℚ :=
  let total_clues := 10
  let prob_knowing_all_clues := (1 / 2) ^ total_clues
  let prob_case_1 := (2 ^ 5) / (2 ^ total_clues)
  let prob_case_2 := (2 ^ 5) / (2 ^ total_clues)
  let prob_case_3 := 25 / (2 ^ total_clues)
  let overcounted_case := prob_knowing_all_clues
  (prob_case_1 + prob_case_2 + prob_case_3 - overcounted_case)

theorem jeffrey_fills_crossword : prob_fill_crossword = 11 / 128 := by
  sorry

end jeffrey_fills_crossword_l1315_131524


namespace complex_number_equality_l1315_131565

theorem complex_number_equality (i : ℂ) (h : i^2 = -1) : 1 + i + i^2 = i :=
by
  sorry

end complex_number_equality_l1315_131565


namespace remainder_of_polynomial_division_l1315_131520

theorem remainder_of_polynomial_division :
  ∀ (x : ℂ), ((x + 2) ^ 2023) % (x^2 + x + 1) = 1 :=
by
  sorry

end remainder_of_polynomial_division_l1315_131520


namespace problem_statement_l1315_131591

def T (m : ℕ) : ℕ := sorry
def H (m : ℕ) : ℕ := sorry

def p (m k : ℕ) : ℝ := 
  if k % 2 = 1 then 0 else sorry

theorem problem_statement (m : ℕ) : p m 0 ≥ p (m + 1) 0 := sorry

end problem_statement_l1315_131591


namespace solve_equation_l1315_131540

theorem solve_equation (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 → 
  x ≠ 3 → 
  x ≠ -2 → 
  x = -7 / 6 :=
by
  sorry

end solve_equation_l1315_131540


namespace circle_symmetric_line_l1315_131587

theorem circle_symmetric_line (m : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 + 2*x - 4*y = 0) → (3*x + y + m = 0)) →
  m = 1 :=
by
  intro h
  sorry

end circle_symmetric_line_l1315_131587


namespace lawsuit_win_probability_l1315_131513

theorem lawsuit_win_probability (P_L1 P_L2 P_W1 P_W2 : ℝ) (h1 : P_L2 = 0.5) 
  (h2 : P_L1 * P_L2 = P_W1 * P_W2 + 0.20 * P_W1 * P_W2)
  (h3 : P_W1 + P_L1 = 1)
  (h4 : P_W2 + P_L2 = 1) : 
  P_W1 = 1 / 2.20 :=
by
  sorry

end lawsuit_win_probability_l1315_131513


namespace probability_all_correct_l1315_131507

noncomputable def probability_mcq : ℚ := 1 / 3
noncomputable def probability_true_false : ℚ := 1 / 2

theorem probability_all_correct :
  (probability_mcq * probability_true_false * probability_true_false) = (1 / 12) :=
by
  sorry

end probability_all_correct_l1315_131507


namespace find_e_l1315_131583

theorem find_e 
  (a b c d e : ℕ) 
  (h1 : a = 16)
  (h2 : b = 2)
  (h3 : c = 3)
  (h4 : d = 12)
  (h5 : 32 / e = 288 / e) 
  : e = 9 := 
by
  sorry

end find_e_l1315_131583


namespace lives_per_player_l1315_131506

theorem lives_per_player (initial_players : ℕ) (additional_players : ℕ) (total_lives : ℕ) 
  (h1 : initial_players = 8) (h2 : additional_players = 2) (h3 : total_lives = 60) : 
  total_lives / (initial_players + additional_players) = 6 :=
by 
  sorry

end lives_per_player_l1315_131506


namespace grasshopper_twenty_five_jumps_l1315_131512

noncomputable def sum_natural (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem grasshopper_twenty_five_jumps :
  let total_distance := sum_natural 25
  total_distance % 2 = 1 -> 0 % 2 = 0 -> total_distance ≠ 0 :=
by
  intros total_distance_odd zero_even
  sorry

end grasshopper_twenty_five_jumps_l1315_131512


namespace dave_tray_problem_l1315_131539

theorem dave_tray_problem (n_trays_per_trip : ℕ) (n_trips : ℕ) (n_second_table : ℕ) : 
  (n_trays_per_trip = 9) → (n_trips = 8) → (n_second_table = 55) → 
  (n_trays_per_trip * n_trips - n_second_table = 17) :=
by
  sorry

end dave_tray_problem_l1315_131539


namespace geometric_sequence_first_term_l1315_131527

theorem geometric_sequence_first_term (a : ℕ) (r : ℕ)
    (h1 : a * r^2 = 27) 
    (h2 : a * r^3 = 81) : 
    a = 3 :=
by
  sorry

end geometric_sequence_first_term_l1315_131527


namespace solve_equation_l1315_131562

theorem solve_equation (x : ℝ) (h₁ : x ≠ -11) (h₂ : x ≠ -5) (h₃ : x ≠ -12) (h₄ : x ≠ -4) :
  (1 / (x + 11) + 1 / (x + 5) = 1 / (x + 12) + 1 / (x + 4)) ↔ x = -8 :=
by
  sorry

end solve_equation_l1315_131562


namespace barnyard_owl_hoots_per_minute_l1315_131578

theorem barnyard_owl_hoots_per_minute :
  (20 - 5) / 3 = 5 := 
by
  sorry

end barnyard_owl_hoots_per_minute_l1315_131578


namespace parabola_chord_length_l1315_131577

theorem parabola_chord_length (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) (h2 : y2^2 = 4 * x2) 
  (hx : x1 + x2 = 9) 
  (focus_line : ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b → y^2 = 4 * x) :
  |(x1 - 1, y1) - (x2 - 1, y2)| = 11 := 
sorry

end parabola_chord_length_l1315_131577


namespace find_b_minus_d_squared_l1315_131561

theorem find_b_minus_d_squared (a b c d : ℝ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 3) :
  (b - d) ^ 2 = 25 :=
sorry

end find_b_minus_d_squared_l1315_131561


namespace niko_percentage_profit_l1315_131528

theorem niko_percentage_profit
    (pairs_sold : ℕ)
    (cost_per_pair : ℕ)
    (profit_5_pairs : ℕ)
    (total_profit : ℕ)
    (num_pairs_remaining : ℕ)
    (cost_remaining_pairs : ℕ)
    (profit_remaining_pairs : ℕ)
    (percentage_profit : ℕ)
    (cost_5_pairs : ℕ):
    pairs_sold = 9 →
    cost_per_pair = 2 →
    profit_5_pairs = 1 →
    total_profit = 3 →
    num_pairs_remaining = 4 →
    cost_remaining_pairs = 8 →
    profit_remaining_pairs = 2 →
    percentage_profit = 25 →
    cost_5_pairs = 10 →
    (profit_remaining_pairs * 100 / cost_remaining_pairs) = percentage_profit :=
by
    intros
    sorry

end niko_percentage_profit_l1315_131528


namespace part1_part2_i_part2_ii_l1315_131535

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x * Real.log x - 1

theorem part1 (a : ℝ) (x : ℝ) : f x a + x^2 * f (1 / x) a = 0 :=
by sorry

theorem part2_i (a : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0) : 2 < a :=
by sorry

theorem part2_ii (a : ℝ) (x1 x2 x3 : ℝ) (h : x1 < x2 ∧ x2 < x3 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0) : x1 + x3 > 2 * a - 2 :=
by sorry

end part1_part2_i_part2_ii_l1315_131535


namespace correct_sample_in_survey_l1315_131554

-- Definitions based on conditions:
def total_population := 1500
def surveyed_population := 150
def sample_description := "the national security knowledge of the selected 150 teachers and students"

-- Hypotheses: conditions
variables (pop : ℕ) (surveyed : ℕ) (description : String)
  (h1 : pop = total_population)
  (h2 : surveyed = surveyed_population)
  (h3 : description = sample_description)

-- Theorem we want to prove
theorem correct_sample_in_survey : description = sample_description :=
  by sorry

end correct_sample_in_survey_l1315_131554


namespace total_distance_traveled_l1315_131504

-- Definitions of conditions
def bess_throw_distance : ℕ := 20
def bess_throws : ℕ := 4
def holly_throw_distance : ℕ := 8
def holly_throws : ℕ := 5
def bess_effective_throw_distance : ℕ := 2 * bess_throw_distance

-- Theorem statement
theorem total_distance_traveled :
  (bess_throws * bess_effective_throw_distance + holly_throws * holly_throw_distance) = 200 := 
  by sorry

end total_distance_traveled_l1315_131504


namespace power_fraction_example_l1315_131576

theorem power_fraction_example : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := 
by
  sorry

end power_fraction_example_l1315_131576


namespace find_y_common_solution_l1315_131517

theorem find_y_common_solution (y : ℝ) :
  (∃ x : ℝ, x^2 + y^2 = 11 ∧ x^2 = 4*y - 7) ↔ (7/4 ≤ y ∧ y ≤ Real.sqrt 11) :=
by
  sorry

end find_y_common_solution_l1315_131517


namespace solve_complex_problem_l1315_131579

-- Define the problem
def complex_sum_eq_two (a b : ℝ) (i : ℂ) : Prop :=
  a + b = 2

-- Define the conditions
def conditions (a b : ℝ) (i : ℂ) : Prop :=
  a + b * i = (1 - i) * (2 + i)

-- State the theorem
theorem solve_complex_problem (a b : ℝ) (i : ℂ) (h : conditions a b i) : complex_sum_eq_two a b i :=
by
  sorry -- Proof goes here

end solve_complex_problem_l1315_131579


namespace trigonometric_identity_l1315_131586

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) : 
  (2 * Real.sin θ - 4 * Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 / 2 := 
by
  sorry

end trigonometric_identity_l1315_131586


namespace daniel_stickers_l1315_131521

def stickers_data 
    (total_stickers : Nat)
    (fred_extra : Nat)
    (andrew_kept : Nat) : Prop :=
  total_stickers = 750 ∧ fred_extra = 120 ∧ andrew_kept = 130

theorem daniel_stickers (D : Nat) :
  stickers_data 750 120 130 → D + (D + 120) = 750 - 130 → D = 250 :=
by
  intros h_data h_eq
  sorry

end daniel_stickers_l1315_131521


namespace total_distance_l1315_131523

variable {D : ℝ}

theorem total_distance (h1 : D / 3 > 0)
                       (h2 : (2 / 3 * D) - (1 / 6 * D) > 0)
                       (h3 : (1 / 2 * D) - (1 / 10 * D) = 180) :
    D = 450 := 
sorry

end total_distance_l1315_131523


namespace obtuse_angles_in_second_quadrant_l1315_131571

theorem obtuse_angles_in_second_quadrant
  (θ : ℝ) 
  (is_obtuse : θ > 90 ∧ θ < 180) :
  90 < θ ∧ θ < 180 :=
by sorry

end obtuse_angles_in_second_quadrant_l1315_131571


namespace min_values_of_exprs_l1315_131548

theorem min_values_of_exprs (r s : ℝ) (hr : 0 < r) (hs : 0 < s) (h : (r + s - r * s) * (r + s + r * s) = r * s) :
  (r + s - r * s) = -3 + 2 * Real.sqrt 3 ∧ (r + s + r * s) = 3 + 2 * Real.sqrt 3 :=
by sorry

end min_values_of_exprs_l1315_131548


namespace multiply_negatives_l1315_131531

theorem multiply_negatives : (-2) * (-3) = 6 :=
  by 
  sorry

end multiply_negatives_l1315_131531


namespace range_of_x_l1315_131501

theorem range_of_x (x : ℝ) : (|x + 1| + |x - 1| = 2) → (-1 ≤ x ∧ x ≤ 1) :=
by
  intro h
  sorry

end range_of_x_l1315_131501


namespace kids_meals_sold_l1315_131581

theorem kids_meals_sold (x y : ℕ) (h1 : x / y = 2) (h2 : x + y = 12) : x = 8 :=
by
  sorry

end kids_meals_sold_l1315_131581


namespace min_sum_of_factors_l1315_131519

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 1806) (h2 : a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) : a + b + c ≥ 112 :=
sorry

end min_sum_of_factors_l1315_131519


namespace math_problem_l1315_131585

variable {f : ℝ → ℝ}

theorem math_problem (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
                     (h2 : ∀ x : ℝ, x > 0 → f x > 0)
                     (h3 : f 1 = 2) :
                     f 0 = 0 ∧
                     (∀ x : ℝ, f (-x) = -f x) ∧
                     (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) ∧
                     (∃ a : ℝ, f (2 - a) = 6 ∧ a = -1) := 
by
  sorry

end math_problem_l1315_131585


namespace find_denominators_l1315_131530

theorem find_denominators (f1 f2 f3 f4 f5 f6 f7 f8 f9 : ℚ)
  (h1 : f1 = 1/3) (h2 : f2 = 1/7) (h3 : f3 = 1/9) (h4 : f4 = 1/11) (h5 : f5 = 1/33)
  (h6 : ∃ (d₁ d₂ d₃ d₄ : ℕ), f6 = 1/d₁ ∧ f7 = 1/d₂ ∧ f8 = 1/d₃ ∧ f9 = 1/d₄ ∧
    (∀ d, d ∈ [d₁, d₂, d₃, d₄] → d % 10 = 5))
  (h7 : f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 = 1) :
  ∃ (d₁ d₂ d₃ d₄ : ℕ), (d₁ = 5) ∧ (d₂ = 15) ∧ (d₃ = 45) ∧ (d₄ = 385) :=
by
  sorry

end find_denominators_l1315_131530


namespace hearty_beads_count_l1315_131582

theorem hearty_beads_count :
  let blue_packages := 3
  let red_packages := 5
  let beads_per_package := 40
  let total_beads := blue_packages * beads_per_package + red_packages * beads_per_package
  total_beads = 320 :=
by
  let blue_packages := 3
  let red_packages := 5
  let beads_per_package := 40
  let total_beads := blue_packages * beads_per_package + red_packages * beads_per_package
  show total_beads = 320
  sorry

end hearty_beads_count_l1315_131582


namespace adrian_water_amount_l1315_131575

theorem adrian_water_amount
  (O S W : ℕ) 
  (h1 : S = 3 * O)
  (h2 : W = 5 * S)
  (h3 : O = 4) : W = 60 :=
by
  sorry

end adrian_water_amount_l1315_131575


namespace parabola_units_shift_l1315_131588

noncomputable def parabola_expression (A B : ℝ × ℝ) (x : ℝ) : ℝ :=
  let b := -5
  let c := 6
  x^2 + b * x + c

theorem parabola_units_shift (A B : ℝ × ℝ) (x : ℝ) (y : ℝ) :
  A = (2, 0) → B = (0, 6) → parabola_expression A B 4 = 2 →
  (y - 2 = 0) → true :=
by
  intro hA hB h4 hy
  sorry

end parabola_units_shift_l1315_131588


namespace unique_intersection_value_k_l1315_131505

theorem unique_intersection_value_k (k : ℝ) : (∀ x y: ℝ, (y = x^2) ∧ (y = 3*x + k) ↔ k = -9/4) :=
by
  sorry

end unique_intersection_value_k_l1315_131505


namespace arithmetic_sequence_common_difference_l1315_131555

theorem arithmetic_sequence_common_difference
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (a₁ d : ℤ)
  (h1 : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2)
  (h2 : ∀ n, a n = a₁ + (n - 1) * d)
  (h3 : S 5 = 5 * (a 4) - 10) :
  d = 2 := sorry

end arithmetic_sequence_common_difference_l1315_131555


namespace angle_mul_add_proof_solve_equation_proof_l1315_131502

-- For (1)
def angle_mul_add_example : Prop :=
  let a := 34 * 3600 + 25 * 60 + 20 -- 34°25'20'' to seconds
  let b := 35 * 60 + 42 * 60        -- 35°42' to total minutes
  let result := a * 3 + b * 60      -- Multiply a by 3 and convert b to seconds
  let final_result := result / 3600 -- Convert back to degrees
  final_result = 138 + (58 / 60)

-- For (2)
def solve_equation_example : Prop :=
  ∀ x : ℚ, (x + 1) / 2 - 1 = (2 - 3 * x) / 3 → x = 1 / 9

theorem angle_mul_add_proof : angle_mul_add_example := sorry

theorem solve_equation_proof : solve_equation_example := sorry

end angle_mul_add_proof_solve_equation_proof_l1315_131502


namespace max_abs_f_lower_bound_l1315_131559

theorem max_abs_f_lower_bound (a b M : ℝ) (hM : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → abs (x^2 + a*x + b) ≤ M) : 
  M ≥ 1/2 :=
sorry

end max_abs_f_lower_bound_l1315_131559


namespace undefined_values_l1315_131551

theorem undefined_values (a : ℝ) : a = -3 ∨ a = 3 ↔ (a^2 - 9 = 0) := sorry

end undefined_values_l1315_131551


namespace smallest_factor_to_end_with_four_zeros_l1315_131564

theorem smallest_factor_to_end_with_four_zeros :
  ∃ x : ℕ, (975 * 935 * 972 * x) % 10000 = 0 ∧
           (∀ y : ℕ, (975 * 935 * 972 * y) % 10000 = 0 → x ≤ y) ∧
           x = 20 := by
  -- The proof would go here.
  sorry

end smallest_factor_to_end_with_four_zeros_l1315_131564


namespace cost_of_one_dozen_pens_l1315_131529

theorem cost_of_one_dozen_pens (pen pencil : ℝ) (h_ratios : pen = 5 * pencil) (h_total : 3 * pen + 5 * pencil = 240) :
  12 * pen = 720 :=
by
  sorry

end cost_of_one_dozen_pens_l1315_131529


namespace simplify_radicals_l1315_131572

theorem simplify_radicals (q : ℝ) (hq : 0 < q) :
  (Real.sqrt (42 * q)) * (Real.sqrt (7 * q)) * (Real.sqrt (14 * q)) = 98 * q * Real.sqrt (3 * q) :=
by
  sorry

end simplify_radicals_l1315_131572


namespace centroid_triangle_PQR_l1315_131556

theorem centroid_triangle_PQR (P Q R S : ℝ × ℝ) 
  (P_coord : P = (2, 5)) 
  (Q_coord : Q = (9, 3)) 
  (R_coord : R = (4, -4))
  (S_is_centroid : S = (
    (P.1 + Q.1 + R.1) / 3,
    (P.2 + Q.2 + R.2) / 3)) :
  9 * S.1 + 4 * S.2 = 151 / 3 :=
by
  sorry

end centroid_triangle_PQR_l1315_131556


namespace sum_products_of_chords_l1315_131518

variable {r x y u v : ℝ}

theorem sum_products_of_chords (h1 : x * y = u * v) (h2 : 4 * r^2 = (x + y)^2 + (u + v)^2) :
  x * (x + y) + u * (u + v) = 4 * r^2 := by
sorry

end sum_products_of_chords_l1315_131518


namespace positive_real_solution_l1315_131547

def polynomial (x : ℝ) : ℝ := x^4 + 10*x^3 - 2*x^2 + 12*x - 9

theorem positive_real_solution (h : polynomial 1 = 0) : polynomial 1 > 0 := sorry

end positive_real_solution_l1315_131547


namespace balance_blue_balls_l1315_131592

noncomputable def weight_balance (G B Y W : ℝ) : ℝ :=
  3 * G + 3 * Y + 5 * W

theorem balance_blue_balls (G B Y W : ℝ)
  (hG : G = 2 * B)
  (hY : Y = 2 * B)
  (hW : W = (5 / 3) * B) :
  weight_balance G B Y W = (61 / 3) * B :=
by
  sorry

end balance_blue_balls_l1315_131592


namespace age_sum_proof_l1315_131522

theorem age_sum_proof (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 20) : a + b + c = 52 :=
by
  sorry

end age_sum_proof_l1315_131522


namespace ellipse_hyperbola_tangent_n_value_l1315_131590

theorem ellipse_hyperbola_tangent_n_value :
  (∃ n : ℝ, (∀ x y : ℝ, 4 * x^2 + y^2 = 4 ∧ x^2 - n * (y - 1)^2 = 1) ↔ n = 3 / 2) :=
by
  sorry

end ellipse_hyperbola_tangent_n_value_l1315_131590


namespace heights_proportional_l1315_131526

-- Define the problem conditions
def sides_ratio (a b c : ℕ) : Prop := a / b = 3 / 4 ∧ b / c = 4 / 5

-- Define the heights
def heights_ratio (h1 h2 h3 : ℕ) : Prop := h1 / h2 = 20 / 15 ∧ h2 / h3 = 15 / 12

-- Problem statement: Given the sides ratio, prove the heights ratio
theorem heights_proportional {a b c h1 h2 h3 : ℕ} (h : sides_ratio a b c) :
  heights_ratio h1 h2 h3 :=
sorry

end heights_proportional_l1315_131526


namespace ann_age_is_26_l1315_131525

theorem ann_age_is_26
  (a b : ℕ)
  (h1 : a + b = 50)
  (h2 : b = 2 * a / 3 + 2 * (a - b)) :
  a = 26 :=
by
  sorry

end ann_age_is_26_l1315_131525


namespace faster_by_airplane_l1315_131584

theorem faster_by_airplane : 
  let driving_time := 3 * 60 + 15 
  let airport_drive := 10
  let wait_to_board := 20
  let flight_duration := driving_time / 3
  let exit_plane := 10
  driving_time - (airport_drive + wait_to_board + flight_duration + exit_plane) = 90 := 
by
  let driving_time : ℕ := 3 * 60 + 15
  let airport_drive : ℕ := 10
  let wait_to_board : ℕ := 20
  let flight_duration : ℕ := driving_time / 3
  let exit_plane : ℕ := 10
  have h1 : driving_time = 195 := rfl
  have h2 : flight_duration = 65 := by norm_num [h1]
  have h3 : 195 - (10 + 20 + 65 + 10) = 195 - 105 := by norm_num
  have h4 : 195 - 105 = 90 := by norm_num
  exact h4

end faster_by_airplane_l1315_131584


namespace gumballs_in_packages_l1315_131500

theorem gumballs_in_packages (total_gumballs : ℕ) (gumballs_per_package : ℕ) (h1 : total_gumballs = 20) (h2 : gumballs_per_package = 5) :
  total_gumballs / gumballs_per_package = 4 :=
by {
  sorry
}

end gumballs_in_packages_l1315_131500


namespace max_value_abcd_l1315_131594

-- Define the digits and constraints on them
def distinct_digits (a b c d e : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

-- Encode the given problem as a Lean theorem
theorem max_value_abcd (a b c d e : ℕ) 
  (h₀ : distinct_digits a b c d e)
  (h₁ : 0 ≤ a ∧ a ≤ 9) 
  (h₂ : 0 ≤ b ∧ b ≤ 9) 
  (h₃ : 0 ≤ c ∧ c ≤ 9) 
  (h₄ : 0 ≤ d ∧ d ≤ 9)
  (h₅ : 0 ≤ e ∧ e ≤ 9)
  (h₆ : e ≠ 0)
  (h₇ : a * 1000 + b * 100 + c * 10 + d = (a * 100 + a * 10 + d) * e) :
  a * 1000 + b * 100 + c * 10 + d = 3015 :=
by {
  sorry
}

end max_value_abcd_l1315_131594


namespace sin_square_general_proposition_l1315_131552

-- Definitions for the given conditions
def sin_square_sum_30_90_150 : Prop :=
  (Real.sin (30 * Real.pi / 180))^2 + (Real.sin (90 * Real.pi / 180))^2 + (Real.sin (150 * Real.pi / 180))^2 = 3/2

def sin_square_sum_5_65_125 : Prop :=
  (Real.sin (5 * Real.pi / 180))^2 + (Real.sin (65 * Real.pi / 180))^2 + (Real.sin (125 * Real.pi / 180))^2 = 3/2

-- The general proposition we want to prove
theorem sin_square_general_proposition (α : ℝ) : 
  sin_square_sum_30_90_150 ∧ sin_square_sum_5_65_125 →
  (Real.sin (α * Real.pi / 180 - 60 * Real.pi / 180))^2 + 
  (Real.sin (α * Real.pi / 180))^2 + 
  (Real.sin (α * Real.pi / 180 + 60 * Real.pi / 180))^2 = 3/2 :=
by
  intro h
  -- Proof goes here
  sorry

end sin_square_general_proposition_l1315_131552


namespace unique_positive_real_solution_of_polynomial_l1315_131597

theorem unique_positive_real_solution_of_polynomial :
  ∃! x : ℝ, x > 0 ∧ (x^11 + 8 * x^10 + 15 * x^9 + 1000 * x^8 - 1200 * x^7 = 0) :=
by
  sorry

end unique_positive_real_solution_of_polynomial_l1315_131597


namespace total_balloons_l1315_131557

theorem total_balloons
  (g b y r : ℕ)  -- Number of green, blue, yellow, and red balloons respectively
  (equal_groups : g = b ∧ b = y ∧ y = r)
  (anya_took : y / 2 = 84) :
  g + b + y + r = 672 := by
sorry

end total_balloons_l1315_131557


namespace area_of_triangle_KDC_l1315_131568

open Real

noncomputable def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

theorem area_of_triangle_KDC
  (radius : ℝ) (chord_length : ℝ) (seg_KA : ℝ)
  (OX distance_DY : ℝ)
  (parallel : ∀ (PA PB : ℝ), PA = PB)
  (collinear : ∀ (PK PA PQ PB : ℝ), PK + PA + PQ + PB = PK + PQ + PA + PB)
  (hyp_radius : radius = 10)
  (hyp_chord_length : chord_length = 12)
  (hyp_seg_KA : seg_KA = 24)
  (hyp_OX : OX = 8)
  (hyp_distance_DY : distance_DY = 8) :
  triangle_area chord_length distance_DY = 48 :=
  by
  sorry

end area_of_triangle_KDC_l1315_131568


namespace all_Xanths_are_Yelps_and_Wicks_l1315_131545

-- Definitions for Zorbs, Yelps, Xanths, and Wicks
variable {U : Type} (Zorb Yelp Xanth Wick : U → Prop)

-- Conditions from the problem
axiom all_Zorbs_are_Yelps : ∀ u, Zorb u → Yelp u
axiom all_Xanths_are_Zorbs : ∀ u, Xanth u → Zorb u
axiom all_Xanths_are_Wicks : ∀ u, Xanth u → Wick u

-- The goal is to prove that all Xanths are Yelps and are Wicks
theorem all_Xanths_are_Yelps_and_Wicks : ∀ u, Xanth u → Yelp u ∧ Wick u := sorry

end all_Xanths_are_Yelps_and_Wicks_l1315_131545


namespace four_gt_sqrt_fourteen_l1315_131515

theorem four_gt_sqrt_fourteen : 4 > Real.sqrt 14 := 
  sorry

end four_gt_sqrt_fourteen_l1315_131515


namespace bus_dispatch_interval_l1315_131558

-- Variables representing the speeds of Xiao Nan and the bus
variable (V_1 V_2 : ℝ)
-- The interval between the dispatch of two buses
variable (interval : ℝ)

-- Stating the conditions in Lean

-- Xiao Nan notices a bus catches up with him every 10 minutes
def cond1 : Prop := ∃ s, s = 10 * (V_1 - V_2)

-- Xiao Yu notices he encounters a bus every 5 minutes
def cond2 : Prop := ∃ s, s = 5 * (V_1 + 3 * V_2)

-- Proof statement
theorem bus_dispatch_interval (h1 : cond1 V_1 V_2) (h2 : cond2 V_1 V_2) : interval = 8 := by
  -- Proof would be provided here
  sorry

end bus_dispatch_interval_l1315_131558


namespace cot_half_angle_product_geq_3sqrt3_l1315_131553

noncomputable def cot (x : ℝ) : ℝ := (Real.cos x) / (Real.sin x)

theorem cot_half_angle_product_geq_3sqrt3 {A B C : ℝ} (h : A + B + C = π) :
    cot (A / 2) * cot (B / 2) * cot (C / 2) ≥ 3 * Real.sqrt 3 := 
  sorry

end cot_half_angle_product_geq_3sqrt3_l1315_131553


namespace problem_statement_l1315_131593

theorem problem_statement (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : a + b + c + 2 = a * b * c) :
  (a+1) * (b+1) * (c+1) ≥ 27 ∧ ((a+1) * (b+1) * (c+1) = 27 → a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end problem_statement_l1315_131593


namespace product_lcm_gcd_eq_2160_l1315_131595

theorem product_lcm_gcd_eq_2160 :
  let a := 36
  let b := 60
  lcm a b * gcd a b = 2160 := by
  sorry

end product_lcm_gcd_eq_2160_l1315_131595


namespace smallest_x_for_equation_l1315_131533

theorem smallest_x_for_equation :
  ∃ x : ℝ, x = -15 ∧ (∀ y : ℝ, 3*y^2 + 39*y - 75 = y*(y + 16) → x ≤ y) ∧ 
  3*(-15)^2 + 39*(-15) - 75 = -15*(-15 + 16) :=
sorry

end smallest_x_for_equation_l1315_131533


namespace numbers_to_be_left_out_l1315_131570

axiom problem_conditions :
  let numbers := [2, 3, 4, 7, 10, 11, 12, 13, 15]
  let grid_numbers := [1, 9, 14, 5]
  numbers.sum + grid_numbers.sum = 106 ∧
  ∃ (left_out : ℕ) (remaining_numbers : List ℕ),
    numbers.erase left_out = remaining_numbers ∧
    (numbers.sum + grid_numbers.sum - left_out) = 96 ∧
    remaining_numbers.length = 8

theorem numbers_to_be_left_out :
  let numbers := [2, 3, 4, 7, 10, 11, 12, 13, 15]
  10 ∈ numbers ∧
  let grid_numbers := [1, 9, 14, 5]
  let total_sum := numbers.sum + grid_numbers.sum
  let grid_sum := total_sum - 10
  grid_sum % 12 = 0 ∧
  grid_sum = 96 :=
sorry

end numbers_to_be_left_out_l1315_131570


namespace quadratic_inequality_solution_set_l1315_131509

variable (a b : ℝ)

theorem quadratic_inequality_solution_set :
  (∀ x : ℝ, (a + b) * x + 2 * a - 3 * b < 0 ↔ x > -(3 / 4)) →
  (∀ x : ℝ, (a - 2 * b) * x ^ 2 + 2 * (a - b - 1) * x + (a - 2) > 0 ↔ -3 + 2 / b < x ∧ x < -1) :=
by
  sorry

end quadratic_inequality_solution_set_l1315_131509


namespace find_m_l1315_131573

theorem find_m (m : ℝ) : (m + 2) * (m - 2) + 3 * m * (m + 2) = 0 ↔ m = 1/2 ∨ m = -2 :=
by
  sorry

end find_m_l1315_131573


namespace sum_of_terms_in_fractional_array_l1315_131546

theorem sum_of_terms_in_fractional_array :
  (∑' (r : ℕ) (c : ℕ), (1 : ℝ) / ((3 * 4) ^ r) * (1 / (4 ^ c))) = (1 / 33) := sorry

end sum_of_terms_in_fractional_array_l1315_131546


namespace can_capacity_l1315_131508

-- Definition for the capacity of the can
theorem can_capacity 
  (milk_ratio water_ratio : ℕ) 
  (add_milk : ℕ) 
  (final_milk_ratio final_water_ratio : ℕ) 
  (capacity : ℕ) 
  (initial_milk initial_water : ℕ) 
  (h_initial_ratio : milk_ratio = 4 ∧ water_ratio = 3) 
  (h_additional_milk : add_milk = 8) 
  (h_final_ratio : final_milk_ratio = 2 ∧ final_water_ratio = 1) 
  (h_initial_amounts : initial_milk = 4 * (capacity - add_milk) / 7 ∧ initial_water = 3 * (capacity - add_milk) / 7) 
  (h_full_capacity : (initial_milk + add_milk) / initial_water = 2) 
  : capacity = 36 :=
sorry

end can_capacity_l1315_131508


namespace possible_values_of_a_l1315_131599

theorem possible_values_of_a (a : ℝ) : (2 < a ∧ a < 3 ∨ 3 < a ∧ a < 5) → (a = 5/2 ∨ a = 4) := 
by
  sorry

end possible_values_of_a_l1315_131599


namespace problem1_problem2_l1315_131569

def prop_p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

theorem problem1 (a : ℝ) (h_a : a = 1) (h_pq : ∃ x, prop_p x a ∧ prop_q x) :
  ∃ x, 2 < x ∧ x < 3 :=
by sorry

theorem problem2 (h_qp : ∀ x (a : ℝ), prop_q x → prop_p x a) :
  ∃ a, 1 < a ∧ a ≤ 2 :=
by sorry

end problem1_problem2_l1315_131569


namespace line_parallelism_theorem_l1315_131511

-- Definitions of the relevant geometric conditions
variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Conditions as hypotheses
axiom line_parallel_plane (m : Line) (α : Plane) : Prop
axiom line_in_plane (n : Line) (α : Plane) : Prop
axiom plane_intersection_line (α β : Plane) : Line
axiom line_parallel (m n : Line) : Prop

-- The problem statement in Lean 4
theorem line_parallelism_theorem 
  (h1 : line_parallel_plane m α) 
  (h2 : line_in_plane n β) 
  (h3 : plane_intersection_line α β = n) 
  (h4 : line_parallel_plane m β) : line_parallel m n :=
sorry

end line_parallelism_theorem_l1315_131511


namespace find_xy_l1315_131534

theorem find_xy (x y : ℝ) :
  0.75 * x - 0.40 * y = 0.20 * 422.50 →
  0.30 * x + 0.50 * y = 0.35 * 530 →
  x = 52.816 ∧ y = -112.222 :=
by
  intro h1 h2
  sorry

end find_xy_l1315_131534


namespace sum_of_squares_of_roots_l1315_131537

theorem sum_of_squares_of_roots :
  ∀ (r₁ r₂ : ℝ), (r₁ + r₂ = 15) → (r₁ * r₂ = 6) → (r₁^2 + r₂^2 = 213) :=
by
  intros r₁ r₂ h_sum h_prod
  -- Proof goes here, but skipping it for now
  sorry

end sum_of_squares_of_roots_l1315_131537


namespace alec_string_ways_l1315_131543

theorem alec_string_ways :
  let letters := ['A', 'C', 'G', 'N']
  let num_ways := 24 * 2 * 2
  num_ways = 96 := 
by
  sorry

end alec_string_ways_l1315_131543
