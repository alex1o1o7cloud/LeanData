import Mathlib

namespace maximum_students_per_dentist_l2193_219323

theorem maximum_students_per_dentist (dentists students : ℕ) (min_students : ℕ) (attended_students : ℕ)
  (h_dentists : dentists = 12)
  (h_students : students = 29)
  (h_min_students : min_students = 2)
  (h_total_students : attended_students = students) :
  ∃ max_students, 
    (∀ d, d < dentists → min_students ≤ attended_students / dentists) ∧
    (∀ d, d < dentists → attended_students = students - (dentists * min_students) + min_students) ∧
    max_students = 7 :=
by
  sorry

end maximum_students_per_dentist_l2193_219323


namespace rent_percentage_increase_l2193_219375

theorem rent_percentage_increase 
  (E : ℝ) 
  (h1 : ∀ (E : ℝ), rent_last_year = 0.25 * E)
  (h2 : ∀ (E : ℝ), earnings_this_year = 1.45 * E)
  (h3 : ∀ (E : ℝ), rent_this_year = 0.35 * earnings_this_year) :
  (rent_this_year / rent_last_year) * 100 = 203 := 
by 
  sorry

end rent_percentage_increase_l2193_219375


namespace mike_total_games_l2193_219315

theorem mike_total_games
  (non_working : ℕ)
  (price_per_game : ℕ)
  (total_earnings : ℕ)
  (h1 : non_working = 9)
  (h2 : price_per_game = 5)
  (h3 : total_earnings = 30) :
  non_working + (total_earnings / price_per_game) = 15 := 
by
  sorry

end mike_total_games_l2193_219315


namespace value_of_a_plus_b_l2193_219383

theorem value_of_a_plus_b (a b : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 3 ↔ ax^2 + bx + 3 < 0) :
  a + b = -3 :=
sorry

end value_of_a_plus_b_l2193_219383


namespace intersection_of_M_N_l2193_219357

-- Definitions of the sets M and N
def M : Set ℝ := { x | (x + 2) * (x - 1) < 0 }
def N : Set ℝ := { x | x + 1 < 0 }

-- Proposition stating that the intersection of M and N is { x | -2 < x < -1 }
theorem intersection_of_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < -1 } :=
  by
    sorry

end intersection_of_M_N_l2193_219357


namespace trigonometric_identity_l2193_219328

open Real 

theorem trigonometric_identity (x y : ℝ) (h₁ : P = x * cos y) (h₂ : Q = x * sin y) : 
  (P + Q) / (P - Q) + (P - Q) / (P + Q) = 2 * cos y / sin y := by 
  sorry

end trigonometric_identity_l2193_219328


namespace simplify_radical_1_simplify_radical_2_find_value_of_a_l2193_219364

-- Problem 1
theorem simplify_radical_1 : 7 + 2 * (Real.sqrt 10) = (Real.sqrt 2 + Real.sqrt 5) ^ 2 := 
by sorry

-- Problem 2
theorem simplify_radical_2 : (Real.sqrt (11 - 6 * (Real.sqrt 2))) = 3 - Real.sqrt 2 := 
by sorry

-- Problem 3
theorem find_value_of_a (a m n : ℕ) (h : a + 2 * Real.sqrt 21 = (Real.sqrt m + Real.sqrt n) ^ 2) : 
  a = 10 ∨ a = 22 := 
by sorry

end simplify_radical_1_simplify_radical_2_find_value_of_a_l2193_219364


namespace solve_trig_eq_l2193_219374

theorem solve_trig_eq (k : ℤ) :
  (8.410 * Real.sqrt 3 * Real.sin t - Real.sqrt (2 * (Real.sin t)^2 - Real.sin (2 * t) + 3 * Real.cos t^2) = 0) ↔
  (∃ k : ℤ, t = π / 4 + 2 * k * π ∨ t = -Real.arctan 3 + π * (2 * k + 1)) :=
sorry

end solve_trig_eq_l2193_219374


namespace percent_increase_stock_l2193_219399

theorem percent_increase_stock (P_open P_close: ℝ) (h1: P_open = 30) (h2: P_close = 45):
  (P_close - P_open) / P_open * 100 = 50 :=
by
  sorry

end percent_increase_stock_l2193_219399


namespace find_m_condition_l2193_219360

theorem find_m_condition (m : ℕ) (h : 9^4 = 3^(2*m)) : m = 4 := by
  sorry

end find_m_condition_l2193_219360


namespace minimum_product_xyz_l2193_219352

noncomputable def minimalProduct (x y z : ℝ) : ℝ :=
  3 * x^2 * (1 - 4 * x)

theorem minimum_product_xyz :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  z = 3 * x →
  x ≤ y ∧ y ≤ z →
  minimalProduct x y z = (9 / 343) :=
by
  intros x y z x_pos y_pos z_pos sum_eq1 z_eq3x inequalities
  sorry

end minimum_product_xyz_l2193_219352


namespace blue_ball_weight_l2193_219369

variable (b t x : ℝ)
variable (c1 : b = 3.12)
variable (c2 : t = 9.12)
variable (c3 : t = b + x)

theorem blue_ball_weight : x = 6 :=
by
  sorry

end blue_ball_weight_l2193_219369


namespace seashells_count_l2193_219348

theorem seashells_count (mary_seashells : ℕ) (keith_seashells : ℕ) (cracked_seashells : ℕ) 
  (h_mary : mary_seashells = 2) (h_keith : keith_seashells = 5) (h_cracked : cracked_seashells = 9) :
  (mary_seashells + keith_seashells = 7) ∧ (cracked_seashells > mary_seashells + keith_seashells) → false := 
by {
  sorry
}

end seashells_count_l2193_219348


namespace original_volume_l2193_219307

theorem original_volume (V : ℝ) (h1 : V > 0) 
    (h2 : (1/16) * V = 0.75) : V = 12 :=
by sorry

end original_volume_l2193_219307


namespace min_n_for_constant_term_l2193_219336

theorem min_n_for_constant_term :
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, 3 * n = 5 * r) → ∃ n : ℕ, n = 5 :=
by
  intros h
  sorry

end min_n_for_constant_term_l2193_219336


namespace intersection_A_B_range_of_m_l2193_219301

-- Step 1: Define sets A, B, and C
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

def B : Set ℝ := {x | -1 < x ∧ x < 3}

def C (m : ℝ) : Set ℝ := {x | m < x ∧ x < 2 * m - 1}

-- Step 2: Lean statements for the proof

-- (1) Prove A ∩ B = {x | 1 < x < 3}
theorem intersection_A_B : (A ∩ B) = {x | 1 < x ∧ x < 3} :=
by
  sorry

-- (2) Prove the range of m such that C ∪ B = B is (-∞, 2]
theorem range_of_m (m : ℝ) : (C m ∪ B = B) ↔ m ≤ 2 :=
by
  sorry

end intersection_A_B_range_of_m_l2193_219301


namespace g_of_neg2_l2193_219324

def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem g_of_neg2 : g (-2) = 7 / 3 := by
  sorry

end g_of_neg2_l2193_219324


namespace family_work_solution_l2193_219327

noncomputable def family_work_problem : Prop :=
  ∃ (M W : ℕ),
    M + W = 15 ∧
    (M * (9/120) + W * (6/180) = 1) ∧
    W = 3

theorem family_work_solution : family_work_problem :=
by
  sorry

end family_work_solution_l2193_219327


namespace series_result_l2193_219317

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l2193_219317


namespace number_b_smaller_than_number_a_l2193_219381

theorem number_b_smaller_than_number_a (A B : ℝ)
  (h : A = B + 1/4) : (B + 1/4 = A) ∧ (B < A) → B = (4 * A - A) / 5 := by
  sorry

end number_b_smaller_than_number_a_l2193_219381


namespace moles_of_NH4Cl_combined_l2193_219305

-- Define the chemical reaction equation
def reaction (NH4Cl H2O NH4OH HCl : ℕ) := 
  NH4Cl + H2O = NH4OH + HCl

-- Given conditions
def condition1 (H2O : ℕ) := H2O = 1
def condition2 (NH4OH : ℕ) := NH4OH = 1

-- Theorem statement: Prove that number of moles of NH4Cl combined is 1
theorem moles_of_NH4Cl_combined (H2O NH4OH NH4Cl HCl : ℕ) 
  (h1: condition1 H2O) (h2: condition2 NH4OH) (h3: reaction NH4Cl H2O NH4OH HCl) : 
  NH4Cl = 1 :=
sorry

end moles_of_NH4Cl_combined_l2193_219305


namespace basketball_game_l2193_219366

/-- Given the conditions of the basketball game:
  * a, ar, ar^2, ar^3 form the Dragons' scores
  * b, b + d, b + 2d, b + 3d form the Lions' scores
  * The game was tied at halftime: a + ar = b + (b + d)
  * The Dragons won by three points at the end: a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3
  * Neither team scored more than 100 points
Prove that the total number of points scored by the two teams in the first half is 30.
-/
theorem basketball_game (a r b d : ℕ) (h1 : a + a * r = b + (b + d))
  (h2 : a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3)
  (h3 : a * (1 + r + r^2 + r^3) < 100)
  (h4 : 4 * b + 6 * d < 100) :
  a + a * r + b + (b + d) = 30 :=
by
  sorry

end basketball_game_l2193_219366


namespace icosahedron_colorings_l2193_219388

theorem icosahedron_colorings :
  let n := 10
  let f := 9
  n! / 5 = 72576 :=
by
  sorry

end icosahedron_colorings_l2193_219388


namespace simplify_expression_l2193_219355

theorem simplify_expression (n : ℕ) : 
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+3)) = 29 / 24 :=
by sorry

end simplify_expression_l2193_219355


namespace sam_mary_total_balloons_l2193_219326

def Sam_initial_balloons : ℝ := 6.0
def Sam_gives : ℝ := 5.0
def Sam_remaining_balloons : ℝ := Sam_initial_balloons - Sam_gives

def Mary_balloons : ℝ := 7.0

def total_balloons : ℝ := Sam_remaining_balloons + Mary_balloons

theorem sam_mary_total_balloons : total_balloons = 8.0 :=
by
  sorry

end sam_mary_total_balloons_l2193_219326


namespace arithmetic_sequence_y_value_l2193_219397

theorem arithmetic_sequence_y_value :
  ∃ y : ℤ, (∃ a1 a3 : ℤ, a1 = 9 ∧ a3 = 81 ∧ y = (a1 + a3) / 2) → y = 45 :=
by
  sorry

end arithmetic_sequence_y_value_l2193_219397


namespace hyperbola_perimeter_l2193_219343

-- Lean 4 statement
theorem hyperbola_perimeter (a b m : ℝ) (h1 : a > 0) (h2 : b > 0)
  (F1 F2 : ℝ × ℝ) (A B : ℝ × ℝ)
  (hyperbola_eq : ∀ (x y : ℝ), (x,y) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1})
  (line_through_F1 : ∀ (x y : ℝ), x = F1.1)
  (A_B_on_hyperbola : (A.1^2/a^2 - A.2^2/b^2 = 1) ∧ (B.1^2/a^2 - B.2^2/b^2 = 1))
  (dist_AB : dist A B = m)
  (dist_relations : dist A F2 + dist B F2 - (dist A F1 + dist B F1) = 4 * a) : 
  dist A F2 + dist B F2 + dist A B = 4 * a + 2 * m :=
sorry

end hyperbola_perimeter_l2193_219343


namespace average_waiting_time_l2193_219376

theorem average_waiting_time 
  (bites_rod1 : ℕ) (bites_rod2 : ℕ) (total_time : ℕ)
  (avg_bites_rod1 : bites_rod1 = 3)
  (avg_bites_rod2 : bites_rod2 = 2)
  (total_bites : bites_rod1 + bites_rod2 = 5)
  (interval : total_time = 6) :
  (total_time : ℝ) / (bites_rod1 + bites_rod2 : ℝ) = 1.2 :=
by
  sorry

end average_waiting_time_l2193_219376


namespace proof_of_neg_p_or_neg_q_l2193_219380

variables (p q : Prop)

theorem proof_of_neg_p_or_neg_q (h₁ : ¬ (p ∧ q)) (h₂ : p ∨ q) : ¬ p ∨ ¬ q :=
  sorry

end proof_of_neg_p_or_neg_q_l2193_219380


namespace diane_honey_harvest_l2193_219393

theorem diane_honey_harvest (last_year : ℕ) (increase : ℕ) (this_year : ℕ) :
  last_year = 2479 → increase = 6085 → this_year = last_year + increase → this_year = 8564 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end diane_honey_harvest_l2193_219393


namespace completing_the_square_l2193_219347

theorem completing_the_square (x : ℝ) : (x^2 - 6*x + 7 = 0) → ((x - 3)^2 = 2) :=
by
  intro h
  sorry

end completing_the_square_l2193_219347


namespace min_neg_signs_to_zero_sum_l2193_219345

-- Definition of the set of numbers on the clock face
def clock_face_numbers : List ℤ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Sum of the clock face numbers
def sum_clock_face_numbers := clock_face_numbers.sum

-- Given condition that the sum of clock face numbers is 78
axiom sum_clock_face_numbers_is_78 : sum_clock_face_numbers = 78

-- Definition of the function to calculate the minimum number of negative signs needed
def min_neg_signs_needed (numbers : List ℤ) (target : ℤ) : ℕ :=
  sorry -- The implementation is omitted

-- Theorem stating the goal of our problem
theorem min_neg_signs_to_zero_sum : min_neg_signs_needed clock_face_numbers 39 = 4 :=
by
  -- Proof is omitted
  sorry

end min_neg_signs_to_zero_sum_l2193_219345


namespace sin_double_angle_l2193_219396

theorem sin_double_angle (x : ℝ) (h : Real.sin (Real.pi / 4 - x) = 3 / 5) : Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_double_angle_l2193_219396


namespace least_positive_divisible_by_smallest_primes_l2193_219325

def smallest_primes := [2, 3, 5, 7, 11]

noncomputable def product_of_smallest_primes :=
  List.foldl (· * ·) 1 smallest_primes

theorem least_positive_divisible_by_smallest_primes :
  product_of_smallest_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_smallest_primes_l2193_219325


namespace brass_total_l2193_219392

theorem brass_total (p_cu : ℕ) (p_zn : ℕ) (m_zn : ℕ) (B : ℕ) 
  (h_ratio : p_cu = 13) 
  (h_zn_ratio : p_zn = 7) 
  (h_zn_mass : m_zn = 35) : 
  (h_brass_total :  p_zn / (p_cu + p_zn) * B = m_zn) → B = 100 :=
sorry

end brass_total_l2193_219392


namespace study_time_difference_l2193_219303

def kwame_study_time : ℕ := 150
def connor_study_time : ℕ := 90
def lexia_study_time : ℕ := 97
def michael_study_time : ℕ := 225
def cassandra_study_time : ℕ := 165
def aria_study_time : ℕ := 720

theorem study_time_difference :
  (kwame_study_time + connor_study_time + michael_study_time + cassandra_study_time) + 187 = (lexia_study_time + aria_study_time) :=
by
  sorry

end study_time_difference_l2193_219303


namespace interval_monotonicity_minimum_value_range_of_a_l2193_219356

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a / x

theorem interval_monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x, 0 < x ∧ x < a → f x a > 0) ∧ (∀ x, x > a → f x a < 0) :=
sorry

theorem minimum_value (a : ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ Real.exp 1 → f x a ≥ 1) ∧ (∃ x, 1 ≤ x ∧ x ≤ Real.exp 1 ∧ f x a = 1) → a = 1 :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ x, x > 1 → f x a < 1 / 2 * x) → a < 1 / 2 :=
sorry

end interval_monotonicity_minimum_value_range_of_a_l2193_219356


namespace faucet_open_duration_l2193_219313

-- Initial definitions based on conditions in the problem
def init_water : ℕ := 120
def flow_rate : ℕ := 4
def rem_water : ℕ := 20

-- The equivalent Lean 4 statement to prove
theorem faucet_open_duration (t : ℕ) (H1: init_water - rem_water = flow_rate * t) : t = 25 :=
sorry

end faucet_open_duration_l2193_219313


namespace lives_per_each_player_l2193_219385

def num_initial_players := 8
def num_quit_players := 3
def total_remaining_lives := 15
def num_remaining_players := num_initial_players - num_quit_players
def lives_per_remaining_player := total_remaining_lives / num_remaining_players

theorem lives_per_each_player :
  lives_per_remaining_player = 3 := by
  sorry

end lives_per_each_player_l2193_219385


namespace combination_identity_l2193_219391

theorem combination_identity (C : ℕ → ℕ → ℕ)
  (comb_formula : ∀ n r, C r n = Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r)))
  (identity_1 : ∀ n r, C r n = C (n-r) n)
  (identity_2 : ∀ n r, C r (n+1) = C r n + C (r-1) n) :
  C 2 100 + C 97 100 = C 3 101 :=
by sorry

end combination_identity_l2193_219391


namespace arccos_one_eq_zero_l2193_219377

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l2193_219377


namespace exists_multiple_with_all_digits_l2193_219322

theorem exists_multiple_with_all_digits (n : ℕ) :
  ∃ m : ℕ, (m % n = 0) ∧ (∀ d : ℕ, d < 10 → d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9) := 
sorry

end exists_multiple_with_all_digits_l2193_219322


namespace find_A_l2193_219316

-- Definitions and conditions
def f (A B : ℝ) (x : ℝ) : ℝ := A * x - 3 * B^2 
def g (B C : ℝ) (x : ℝ) : ℝ := B * x + C

theorem find_A (A B C : ℝ) (hB : B ≠ 0) (hBC : B + C ≠ 0) :
  f A B (g B C 1) = 0 → A = (3 * B^2) / (B + C) :=
by
  -- Introduction of the hypotheses
  intro h
  sorry

end find_A_l2193_219316


namespace p_6_eq_163_l2193_219302

noncomputable def p (x : ℕ) : ℕ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) + x^2 + x + 1

theorem p_6_eq_163 : p 6 = 163 :=
by
  sorry

end p_6_eq_163_l2193_219302


namespace value_of_a_l2193_219314

theorem value_of_a (a b : ℤ) (h : (∀ x, x^2 - x - 1 = 0 → a * x^17 + b * x^16 + 1 = 0)) : a = 987 :=
by 
  sorry

end value_of_a_l2193_219314


namespace minimum_norm_of_v_l2193_219354

open Real 

-- Define the vector v and condition
noncomputable def v : ℝ × ℝ := sorry

-- Define the condition
axiom v_condition : ‖(v.1 + 4, v.2 + 2)‖ = 10

-- The statement that we need to prove
theorem minimum_norm_of_v : ‖v‖ = 10 - 2 * sqrt 5 :=
by
  sorry

end minimum_norm_of_v_l2193_219354


namespace horner_evaluation_at_two_l2193_219351

/-- Define the polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 * x^6 + 3 * x^5 + 5 * x^3 + 6 * x^2 + 7 * x + 8

/-- States that the value of f(2) using Horner's Rule equals 14. -/
theorem horner_evaluation_at_two : f 2 = 14 :=
sorry

end horner_evaluation_at_two_l2193_219351


namespace quadratic_inequality_solution_l2193_219368

theorem quadratic_inequality_solution (a b: ℝ) (h1: ∀ x: ℝ, 1 < x ∧ x < 2 → ax^2 + bx - 4 > 0) (h2: ∀ x: ℝ, x ≤ 1 ∨ x ≥ 2 → ax^2 + bx - 4 ≤ 0) : a + b = 4 :=
sorry

end quadratic_inequality_solution_l2193_219368


namespace number_of_daisies_is_two_l2193_219365

theorem number_of_daisies_is_two :
  ∀ (total_flowers daisies tulips sunflowers remaining_flowers : ℕ), 
    total_flowers = 12 →
    sunflowers = 4 →
    (3 / 5) * remaining_flowers = tulips →
    (2 / 5) * remaining_flowers = sunflowers →
    remaining_flowers = total_flowers - daisies - sunflowers →
    daisies = 2 :=
by
  intros total_flowers daisies tulips sunflowers remaining_flowers 
  sorry

end number_of_daisies_is_two_l2193_219365


namespace number_of_solutions_pi_equation_l2193_219398

theorem number_of_solutions_pi_equation : 
  ∃ (x0 x1 : ℝ), (x0 = 0 ∧ x1 = 1) ∧ ∀ x : ℝ, (π^(x-1) * x^2 + π^(x^2) * x - π^(x^2) = x^2 + x - 1 ↔ x = x0 ∨ x = x1)
:=
by sorry

end number_of_solutions_pi_equation_l2193_219398


namespace avg_marks_chem_math_l2193_219358

variable (P C M : ℝ)

theorem avg_marks_chem_math (h : P + C + M = P + 140) : (C + M) / 2 = 70 :=
by
  -- skip the proof, just provide the statement
  sorry

end avg_marks_chem_math_l2193_219358


namespace sum_of_sides_of_similar_triangle_l2193_219382

theorem sum_of_sides_of_similar_triangle (a b c : ℕ) (scale_factor : ℕ) (longest_side_sim : ℕ) (sum_of_other_sides_sim : ℕ) : 
  a * scale_factor = 21 → c = 7 → b = 5 → a = 3 → 
  sum_of_other_sides = a * scale_factor + b * scale_factor → 
sum_of_other_sides = 24 :=
by
  sorry

end sum_of_sides_of_similar_triangle_l2193_219382


namespace greatest_possible_fourth_term_l2193_219370

theorem greatest_possible_fourth_term {a d : ℕ} (h : 5 * a + 10 * d = 60) : a + 3 * (12 - a) ≤ 34 :=
by 
  sorry

end greatest_possible_fourth_term_l2193_219370


namespace largest_common_term_lt_300_l2193_219394

theorem largest_common_term_lt_300 :
  ∃ a : ℕ, a < 300 ∧ (∃ n : ℤ, a = 4 + 5 * n) ∧ (∃ m : ℤ, a = 3 + 7 * m) ∧ ∀ b : ℕ, b < 300 → (∃ n : ℤ, b = 4 + 5 * n) → (∃ m : ℤ, b = 3 + 7 * m) → b ≤ a :=
sorry

end largest_common_term_lt_300_l2193_219394


namespace trig_expression_value_l2193_219304

theorem trig_expression_value :
  (3 / (Real.sin (140 * Real.pi / 180))^2 - 1 / (Real.cos (140 * Real.pi / 180))^2) * (1 / (2 * Real.sin (10 * Real.pi / 180))) = 16 := 
by
  -- placeholder for proof
  sorry

end trig_expression_value_l2193_219304


namespace rationalize_sqrt_fraction_l2193_219361

theorem rationalize_sqrt_fraction {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) : 
  (Real.sqrt ((a : ℝ) / b)) = (Real.sqrt (a * (b / (b * b)))) → 
  (Real.sqrt (5 / 12)) = (Real.sqrt 15 / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l2193_219361


namespace calculate_geometric_sequence_sum_l2193_219359

def geometric_sequence (a₁ r : ℤ) (n : ℕ) : ℤ :=
  a₁ * r^n

theorem calculate_geometric_sequence_sum :
  let a₁ := 1
  let r := -2
  let a₂ := geometric_sequence a₁ r 1
  let a₃ := geometric_sequence a₁ r 2
  let a₄ := geometric_sequence a₁ r 3
  a₁ + |a₂| + a₃ + |a₄| = 15 :=
by
  sorry

end calculate_geometric_sequence_sum_l2193_219359


namespace carpet_dimensions_problem_l2193_219329

def carpet_dimensions (width1 width2 : ℕ) (l : ℕ) :=
  ∃ x y : ℕ, width1 = 38 ∧ width2 = 50 ∧ l = l ∧ x = 25 ∧ y = 50

theorem carpet_dimensions_problem (l : ℕ) :
  carpet_dimensions 38 50 l :=
by
  sorry

end carpet_dimensions_problem_l2193_219329


namespace combined_mean_is_254_over_15_l2193_219332

noncomputable def combined_mean_of_sets 
  (mean₁ : ℝ) (n₁ : ℕ) 
  (mean₂ : ℝ) (n₂ : ℕ) : ℝ :=
  (mean₁ * n₁ + mean₂ * n₂) / (n₁ + n₂)

theorem combined_mean_is_254_over_15 :
  combined_mean_of_sets 18 7 16 8 = (254 : ℝ) / 15 :=
by
  sorry

end combined_mean_is_254_over_15_l2193_219332


namespace remaining_volume_of_cube_l2193_219331

theorem remaining_volume_of_cube (s : ℝ) (r : ℝ) (h : ℝ) (π : ℝ) 
    (cube_volume : s = 5) 
    (cylinder_radius : r = 1.5) 
    (cylinder_height : h = 5) :
    s^3 - π * r^2 * h = 125 - 11.25 * π := by
  sorry

end remaining_volume_of_cube_l2193_219331


namespace find_a8_l2193_219386

variable {α : Type} [LinearOrderedField α]

/-- Given conditions of an arithmetic sequence -/
def arithmetic_sequence (a_n : ℕ → α) : Prop :=
  ∃ (a1 d : α), ∀ n : ℕ, a_n n = a1 + n * d

theorem find_a8 (a_n : ℕ → ℝ)
  (h_arith : arithmetic_sequence a_n)
  (h3 : a_n 3 = 5)
  (h5 : a_n 5 = 3) :
  a_n 8 = 0 :=
sorry

end find_a8_l2193_219386


namespace water_added_l2193_219373

theorem water_added (W x : ℕ) (h₁ : 2 * W = 5 * 10)
                    (h₂ : 2 * (W + x) = 7 * 10) :
  x = 10 :=
by
  sorry

end water_added_l2193_219373


namespace find_prime_pairs_l2193_219349

theorem find_prime_pairs (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : 0 < n) :
  p * (p + 1) + q * (q + 1) = n * (n + 1) ↔ (p = 3 ∧ q = 5 ∧ n = 6) ∨ (p = 5 ∧ q = 3 ∧ n = 6) ∨ (p = 2 ∧ q = 2 ∧ n = 3) :=
by
  sorry

end find_prime_pairs_l2193_219349


namespace farmer_land_area_l2193_219330

-- Variables representing the total land, and the percentages and areas.
variable {T : ℝ} (h_cleared : 0.85 * T =  V) (V_10_percent : 0.10 * V + 0.70 * V + 0.05 * V + 500 = V)
variable {total_acres : ℝ} (correct_total_acres : total_acres = 3921.57)

theorem farmer_land_area (h_cleared : 0.85 * T = V) (h_planted : 0.85 * V = 500) : T = 3921.57 :=
by
  sorry

end farmer_land_area_l2193_219330


namespace six_digit_quotient_l2193_219341

def six_digit_number (A B : ℕ) : ℕ := 100000 * A + 97860 + B

def divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

theorem six_digit_quotient (A B : ℕ) (hA : A = 5) (hB : B = 1)
  (h9786B : divisible_by_99 (six_digit_number A B)) : 
  six_digit_number A B / 99 = 6039 := by
  sorry

end six_digit_quotient_l2193_219341


namespace solve_for_five_minus_a_l2193_219389

theorem solve_for_five_minus_a (a b : ℤ) 
  (h1 : 5 + a = 6 - b)
  (h2 : 6 + b = 9 + a) : 
  5 - a = 6 := 
by 
  sorry

end solve_for_five_minus_a_l2193_219389


namespace problem_statement_l2193_219311

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem problem_statement : avg3 (avg3 (-1) 2 3) (avg2 2 3) 1 = 29 / 18 := 
by 
  sorry

end problem_statement_l2193_219311


namespace paused_time_l2193_219300

theorem paused_time (total_length remaining_length paused_at : ℕ) (h1 : total_length = 60) (h2 : remaining_length = 30) : paused_at = total_length - remaining_length :=
by
  sorry

end paused_time_l2193_219300


namespace ninth_term_arithmetic_sequence_l2193_219309

theorem ninth_term_arithmetic_sequence 
  (a1 a17 d a9 : ℚ) 
  (h1 : a1 = 2 / 3) 
  (h17 : a17 = 3 / 2) 
  (h_formula : a17 = a1 + 16 * d) 
  (h9_formula : a9 = a1 + 8 * d) :
  a9 = 13 / 12 := by
  sorry

end ninth_term_arithmetic_sequence_l2193_219309


namespace not_every_tv_owner_has_pass_l2193_219318

variable (Person : Type) (T P G : Person → Prop)

-- Condition 1: There exists a television owner who is not a painter.
axiom exists_tv_owner_not_painter : ∃ x, T x ∧ ¬ P x 

-- Condition 2: If someone has a pass to the Gellért Baths and is not a painter, they are not a television owner.
axiom pass_and_not_painter_imp_not_tv_owner : ∀ x, (G x ∧ ¬ P x) → ¬ T x

-- Prove: Not every television owner has a pass to the Gellért Baths.
theorem not_every_tv_owner_has_pass :
  ¬ ∀ x, T x → G x :=
by
  sorry -- Proof omitted

end not_every_tv_owner_has_pass_l2193_219318


namespace proof_problem_l2193_219339

noncomputable def problem_statement : Prop :=
  let p1 := ∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 - x + m = 0
  let p2 := ∀ x y : ℝ, x + y > 2 → x > 1 ∧ y > 1
  let p3 := ∃ x : ℝ, -2 < x ∧ x < 4 ∧ |x - 2| ≥ 3
  let p4 := ∀ a b c : ℝ, a ≠ 0 ∧ b^2 - 4 * a * c > 0 → ∃ x₁ x₂ : ℝ, x₁ * x₂ < 0
  p3 = true ∧ p1 = false ∧ p2 = false ∧ p4 = false

theorem proof_problem : problem_statement := 
sorry

end proof_problem_l2193_219339


namespace find_f5_l2193_219335

noncomputable def f : ℝ → ℝ := sorry

axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f4_value : f 4 = 5

theorem find_f5 : f 5 = 25 / 4 :=
by
  -- Proof goes here
  sorry

end find_f5_l2193_219335


namespace right_triangle_hypotenuse_l2193_219371

noncomputable def hypotenuse_length (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem right_triangle_hypotenuse :
  ∀ (a b : ℝ),
  (1/3) * Real.pi * b^2 * a = 675 * Real.pi →
  (1/3) * Real.pi * a^2 * b = 1215 * Real.pi →
  hypotenuse_length a b = 3 * Real.sqrt 106 :=
  by
  intros a b h1 h2
  sorry

end right_triangle_hypotenuse_l2193_219371


namespace intersection_sets_l2193_219321

theorem intersection_sets (M N : Set ℝ) :
  (M = {x | x * (x - 3) < 0}) → (N = {x | |x| < 2}) → (M ∩ N = {x | 0 < x ∧ x < 2}) :=
by
  intro hM hN
  rw [hM, hN]
  sorry

end intersection_sets_l2193_219321


namespace geometric_series_m_value_l2193_219372

theorem geometric_series_m_value (m : ℝ) : 
    let a : ℝ := 20
    let r₁ : ℝ := 1 / 2  -- Common ratio for the first series
    let S₁ : ℝ := a / (1 - r₁)  -- Sum of the first series
    let b : ℝ := 1 / 2 + m / 20  -- Common ratio for the second series
    let S₂ : ℝ := a / (1 - b)  -- Sum of the second series
    S₁ = 40 ∧ S₂ = 120 → m = 20 / 3 :=
sorry

end geometric_series_m_value_l2193_219372


namespace fraction_eq_zero_l2193_219367

theorem fraction_eq_zero {x : ℝ} (h : (6 * x) ≠ 0) : (x - 5) / (6 * x) = 0 ↔ x = 5 := 
by
  sorry

end fraction_eq_zero_l2193_219367


namespace christineTravelDistance_l2193_219312

-- Definition of Christine's speed and time
def christineSpeed : ℝ := 20
def christineTime : ℝ := 4

-- Theorem to prove the distance Christine traveled
theorem christineTravelDistance : christineSpeed * christineTime = 80 := by
  -- The proof is omitted
  sorry

end christineTravelDistance_l2193_219312


namespace arth_seq_val_a7_l2193_219319

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arth_seq_val_a7 {a : ℕ → ℝ} 
  (h_arith : arithmetic_sequence a)
  (h_positive : ∀ n : ℕ, 0 < a n)
  (h_eq : 2 * a 6 + 2 * a 8 = (a 7) ^ 2) :
  a 7 = 4 := 
by sorry

end arth_seq_val_a7_l2193_219319


namespace time_without_moving_walkway_l2193_219353

/--
Assume a person walks from one end to the other of a 90-meter long moving walkway at a constant rate in 30 seconds, assisted by the walkway. When this person reaches the end, they reverse direction and continue walking with the same speed, but this time it takes 120 seconds because the person is traveling against the direction of the moving walkway.

Prove that if the walkway were to stop moving, it would take this person 48 seconds to walk from one end of the walkway to the other.
-/
theorem time_without_moving_walkway : 
  ∀ (v_p v_w : ℝ),
  (v_p + v_w) * 30 = 90 →
  (v_p - v_w) * 120 = 90 →
  90 / v_p = 48 :=
by
  intros v_p v_w h1 h2
  have hpw := eq_of_sub_eq_zero (sub_eq_zero.mpr h1)
  have hmw := eq_of_sub_eq_zero (sub_eq_zero.mpr h2)
  sorry

end time_without_moving_walkway_l2193_219353


namespace find_m_for_parallel_vectors_l2193_219390

theorem find_m_for_parallel_vectors (m : ℝ) :
  let a := (1, m)
  let b := (2, -1)
  (2 * a.1 + b.1, 2 * a.2 + b.2) = (k * (a.1 - 2 * b.1), k * (a.2 - 2 * b.2)) → m = -1/2 :=
by
  sorry

end find_m_for_parallel_vectors_l2193_219390


namespace intersection_complement_A_B_l2193_219337

open Set

variable (x : ℝ)

def U := ℝ
def A := {x | -2 ≤ x ∧ x ≤ 3}
def B := {x | x < -1 ∨ x > 4}

theorem intersection_complement_A_B :
  {x | -2 ≤ x ∧ x ≤ 3} ∩ compl {x | x < -1 ∨ x > 4} = {x | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end intersection_complement_A_B_l2193_219337


namespace problem_part1_problem_part2_l2193_219378

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := abs (a - x)

def setA (a : ℝ) : Set ℝ := {x | f a (2 * x - 3 / 2) > 2 * f a (x + 2) + 2}

theorem problem_part1 {a : ℝ} (h : a = 3 / 2) : setA a = {x | x < 0} := by
  sorry

theorem problem_part2 {a : ℝ} (h : a = 3 / 2) (x0 : ℝ) (hx0 : x0 ∈ setA a) (x : ℝ) : 
    f a (x0 * x) ≥ x0 * f a x + f a (a * x0) := by
  sorry

end problem_part1_problem_part2_l2193_219378


namespace well_depth_l2193_219362

def daily_climb_up : ℕ := 4
def daily_slip_down : ℕ := 3
def total_days : ℕ := 27

theorem well_depth : (daily_climb_up * (total_days - 1) - daily_slip_down * (total_days - 1)) + daily_climb_up = 30 := by
  -- conditions
  let net_daily_progress := daily_climb_up - daily_slip_down
  let net_26_days_progress := net_daily_progress * (total_days - 1)

  -- proof to be completed
  sorry

end well_depth_l2193_219362


namespace Darcy_remaining_clothes_l2193_219346

/--
Darcy initially has 20 shirts and 8 pairs of shorts.
He folds 12 of the shirts and 5 of the pairs of shorts.
We want to prove that the total number of remaining pieces of clothing Darcy has to fold is 11.
-/
theorem Darcy_remaining_clothes
  (initial_shirts : Nat)
  (initial_shorts : Nat)
  (folded_shirts : Nat)
  (folded_shorts : Nat)
  (remaining_shirts : Nat)
  (remaining_shorts : Nat)
  (total_remaining : Nat) :
  initial_shirts = 20 → initial_shorts = 8 →
  folded_shirts = 12 → folded_shorts = 5 →
  remaining_shirts = initial_shirts - folded_shirts →
  remaining_shorts = initial_shorts - folded_shorts →
  total_remaining = remaining_shirts + remaining_shorts →
  total_remaining = 11 := by
  sorry

end Darcy_remaining_clothes_l2193_219346


namespace surface_area_is_33_l2193_219344

structure TShape where
  vertical_cubes : ℕ -- Number of cubes in the vertical line
  horizontal_cubes : ℕ -- Number of cubes in the horizontal line
  intersection_point : ℕ -- Intersection point in the vertical line
  
def surface_area (t : TShape) : ℕ :=
  let top_and_bottom := 9 + 9
  let side_vertical := (3 + 4) -- 3 for the top cube, 1 each for the other 4 cubes
  let side_horizontal := (4 - 1) * 2 -- each of 4 left and right minus intersection twice
  let intersection := 2
  top_and_bottom + side_vertical + side_horizontal + intersection

theorem surface_area_is_33 (t : TShape) (h1 : t.vertical_cubes = 5) (h2 : t.horizontal_cubes = 5) (h3 : t.intersection_point = 3) : 
  surface_area t = 33 := by
  sorry

end surface_area_is_33_l2193_219344


namespace binomial_10_3_eq_120_l2193_219308

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l2193_219308


namespace poly_divisibility_implies_C_D_l2193_219310

noncomputable def poly_condition : Prop :=
  ∃ (C D : ℤ), ∀ (α : ℂ), α^2 - α + 1 = 0 → α^103 + C * α^2 + D * α + 1 = 0

/- The translated proof problem -/
theorem poly_divisibility_implies_C_D (C D : ℤ) :
  (poly_condition) → (C = -1 ∧ D = 0) :=
by
  intro h
  sorry

end poly_divisibility_implies_C_D_l2193_219310


namespace kaleb_earnings_and_boxes_l2193_219342

-- Conditions
def initial_games : ℕ := 76
def games_sold : ℕ := 46
def price_15_dollar : ℕ := 20
def price_10_dollar : ℕ := 15
def price_8_dollar : ℕ := 11
def games_per_box : ℕ := 5

-- Definitions and proof problem
theorem kaleb_earnings_and_boxes (initial_games games_sold price_15_dollar price_10_dollar price_8_dollar games_per_box : ℕ) :
  let earnings := (price_15_dollar * 15) + (price_10_dollar * 10) + (price_8_dollar * 8)
  let remaining_games := initial_games - games_sold
  let boxes_needed := remaining_games / games_per_box
  earnings = 538 ∧ boxes_needed = 6 :=
by
  sorry

end kaleb_earnings_and_boxes_l2193_219342


namespace engineer_progress_l2193_219333

theorem engineer_progress (x : ℕ) : 
  ∀ (road_length_in_km : ℝ) 
    (total_days : ℕ) 
    (initial_men : ℕ) 
    (completed_work_in_km : ℝ) 
    (additional_men : ℕ) 
    (new_total_men : ℕ) 
    (remaining_work_in_km : ℝ) 
    (remaining_days : ℕ),
    road_length_in_km = 10 → 
    total_days = 300 → 
    initial_men = 30 → 
    completed_work_in_km = 2 → 
    additional_men = 30 → 
    new_total_men = 60 → 
    remaining_work_in_km = 8 → 
    remaining_days = total_days - x →
  (4 * (total_days - x) = 8 * x) →
  x = 100 :=
by
  intros road_length_in_km total_days initial_men completed_work_in_km additional_men new_total_men remaining_work_in_km remaining_days
  intros h1 h2 h3 h4 h5 h6 h7 h8 h_eqn
  -- Proof
  sorry

end engineer_progress_l2193_219333


namespace time_to_traverse_nth_mile_l2193_219384

theorem time_to_traverse_nth_mile (n : ℕ) (h : n ≥ 3) : ∃ t : ℕ, t = (n - 2)^2 :=
by
  -- Given:
  -- Speed varies inversely as the square of the number of miles already traveled.
  -- Speed is constant for each mile.
  -- The third mile is traversed in 4 hours.
  -- Show that:
  -- The time to traverse the nth mile is (n - 2)^2 hours.
  sorry

end time_to_traverse_nth_mile_l2193_219384


namespace cos_difference_of_angles_l2193_219320

theorem cos_difference_of_angles (α β : ℝ) 
    (h1 : Real.cos (α + β) = 1 / 5) 
    (h2 : Real.tan α * Real.tan β = 1 / 2) : 
    Real.cos (α - β) = 3 / 5 := 
sorry

end cos_difference_of_angles_l2193_219320


namespace horses_for_camels_l2193_219340

noncomputable def cost_of_one_elephant : ℕ := 11000
noncomputable def cost_of_one_ox : ℕ := 7333 -- approx.
noncomputable def cost_of_one_horse : ℕ := 1833 -- approx.
noncomputable def cost_of_one_camel : ℕ := 4400

theorem horses_for_camels (H : ℕ) :
  (H * cost_of_one_horse = cost_of_one_camel) → H = 2 :=
by
  -- skipping proof details
  sorry

end horses_for_camels_l2193_219340


namespace three_digit_number_digits_difference_l2193_219395

theorem three_digit_number_digits_difference (a b c : ℕ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : a < b) (h4 : b < c) :
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  reversed_number - original_number = 198 := by
  sorry

end three_digit_number_digits_difference_l2193_219395


namespace probability_of_x_in_interval_l2193_219350

noncomputable def interval_length (a b : ℝ) : ℝ := b - a

noncomputable def probability_in_interval : ℝ :=
  let length_total := interval_length (-2) 1
  let length_sub := interval_length 0 1
  length_sub / length_total

theorem probability_of_x_in_interval :
  probability_in_interval = 1 / 3 :=
by
  sorry

end probability_of_x_in_interval_l2193_219350


namespace count_integers_with_zero_l2193_219387

/-- There are 740 positive integers less than or equal to 3017 that contain the digit 0. -/
theorem count_integers_with_zero (n : ℕ) (h : n ≤ 3017) : 
  (∃ k : ℕ, k ≤ 3017 ∧ ∃ d : ℕ, d < 10 ∧ d ≠ 0 ∧ k / 10 ^ d % 10 = 0) ↔ n = 740 :=
by sorry

end count_integers_with_zero_l2193_219387


namespace change_is_five_l2193_219334

noncomputable def haircut_cost := 15
noncomputable def payment := 20
noncomputable def counterfeit := 20
noncomputable def exchanged_amount := (10 : ℤ) + 10
noncomputable def flower_shop_amount := 20

def change_given (payment haircut_cost: ℕ) : ℤ :=
payment - haircut_cost

theorem change_is_five : 
  change_given payment haircut_cost = 5 :=
by 
  sorry

end change_is_five_l2193_219334


namespace Linda_original_savings_l2193_219306

theorem Linda_original_savings (S : ℝ)
  (H1 : 3/4 * S + 1/4 * S = S)
  (H2 : 1/4 * S = 220) :
  S = 880 :=
sorry

end Linda_original_savings_l2193_219306


namespace handshakes_minimum_l2193_219379

/-- Given 30 people and each person shakes hands with exactly three others,
    the minimum possible number of handshakes is 45. -/
theorem handshakes_minimum (n k : ℕ) (h_n : n = 30) (h_k : k = 3) :
  (n * k) / 2 = 45 :=
by
  sorry

end handshakes_minimum_l2193_219379


namespace arithmetic_sequence_common_difference_divisible_by_p_l2193_219338

theorem arithmetic_sequence_common_difference_divisible_by_p 
  (n : ℕ) (a : ℕ → ℕ) (h1 : n ≥ 2021) (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j) 
  (h3 : a 1 > 2021) (h4 : ∀ i, 1 ≤ i → i ≤ n → Nat.Prime (a i)) : 
  ∀ p, Nat.Prime p → p < 2021 → ∃ d, (∀ m, 2 ≤ m → a m = a 1 + (m - 1) * d) ∧ p ∣ d := 
sorry

end arithmetic_sequence_common_difference_divisible_by_p_l2193_219338


namespace find_x2_plus_y2_l2193_219363

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
  sorry

end find_x2_plus_y2_l2193_219363
