import Mathlib

namespace sufficiency_but_not_necessary_l1933_193392

theorem sufficiency_but_not_necessary (x y : ℝ) : |x| + |y| ≤ 1 → x^2 + y^2 ≤ 1 ∧ ¬(x^2 + y^2 ≤ 1 → |x| + |y| ≤ 1) :=
by
  sorry

end sufficiency_but_not_necessary_l1933_193392


namespace probability_cond_satisfied_l1933_193357

-- Define the floor and log conditions
def cond1 (x : ℝ) : Prop := ⌊Real.log x / Real.log 2 + 1⌋ = ⌊Real.log x / Real.log 2⌋
def cond2 (x : ℝ) : Prop := ⌊Real.log (2 * x) / Real.log 2 + 1⌋ = ⌊Real.log (2 * x) / Real.log 2⌋
def valid_interval (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Main theorem stating the proof problem
theorem probability_cond_satisfied : 
  (∀ (x : ℝ), valid_interval x → cond1 x → cond2 x → x ∈ Set.Icc (0.25:ℝ) 0.5) → 
  (0.5 - 0.25) / 1 = 1 / 4 := 
by
  -- Proof omitted
  sorry

end probability_cond_satisfied_l1933_193357


namespace investment_amount_l1933_193359

theorem investment_amount (A_investment B_investment total_profit A_share : ℝ)
  (hA_investment : A_investment = 100)
  (hB_investment_months : B_investment > 0)
  (h_total_profit : total_profit = 100)
  (h_A_share : A_share = 50)
  (h_conditions : A_share / total_profit = (A_investment * 12) / ((A_investment * 12) + (B_investment * 6))) :
  B_investment = 200 :=
by {
  sorry
}

end investment_amount_l1933_193359


namespace no_solution_eqn_l1933_193365

theorem no_solution_eqn : ∀ x : ℝ, x ≠ -11 ∧ x ≠ -8 ∧ x ≠ -12 ∧ x ≠ -7 →
  ¬ (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) :=
by
  intros x h
  sorry

end no_solution_eqn_l1933_193365


namespace square_side_length_l1933_193354

theorem square_side_length (A : ℝ) (s : ℝ) (h : A = s^2) (hA : A = 144) : s = 12 :=
by 
  -- sorry is used to skip the proof
  sorry

end square_side_length_l1933_193354


namespace count_birds_l1933_193312

theorem count_birds (b m c : ℕ) (h1 : b + m + c = 300) (h2 : 2 * b + 4 * m + 3 * c = 708) : b = 192 := 
sorry

end count_birds_l1933_193312


namespace reciprocal_of_sum_is_correct_l1933_193368

theorem reciprocal_of_sum_is_correct : (1 / (1 / 4 + 1 / 6)) = 12 / 5 := by
  sorry

end reciprocal_of_sum_is_correct_l1933_193368


namespace find_sum_of_cubes_l1933_193328

noncomputable def roots_of_polynomial := 
  ∃ a b c : ℝ, 
    (6 * a^3 + 500 * a + 1001 = 0) ∧ 
    (6 * b^3 + 500 * b + 1001 = 0) ∧ 
    (6 * c^3 + 500 * c + 1001 = 0)

theorem find_sum_of_cubes (a b c : ℝ) 
  (h : roots_of_polynomial) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 500.5 := 
sorry

end find_sum_of_cubes_l1933_193328


namespace problem_l1933_193316

universe u

def U : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {x ∈ U | x ≠ 0} -- Placeholder, B itself is a generic subset of U
def A : Set ℕ := {x ∈ U | x = 3 ∨ x = 5 ∨ x = 9}

noncomputable def C_U (B : Set ℕ) : Set ℕ := {x ∈ U | ¬ (x ∈ B)}

axiom h1 : A ∩ B = {3, 5}
axiom h2 : A ∩ C_U B = {9}

theorem problem : A = {3, 5, 9} :=
by
  sorry

end problem_l1933_193316


namespace angle_B_lt_90_l1933_193307

theorem angle_B_lt_90 {a b c : ℝ} (h_arith : b = (a + c) / 2) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  ∃ (A B C : ℝ), B < 90 :=
sorry

end angle_B_lt_90_l1933_193307


namespace number_subtracted_l1933_193340

-- Define the variables x and y
variable (x y : ℝ)

-- Define the conditions
def condition1 := 6 * x - y = 102
def condition2 := x = 40

-- Define the theorem to prove
theorem number_subtracted (h1 : condition1 x y) (h2 : condition2 x) : y = 138 :=
sorry

end number_subtracted_l1933_193340


namespace complement_union_l1933_193364

open Set

def S : Set ℝ := { x | x > -2 }
def T : Set ℝ := { x | x^2 + 3*x - 4 ≤ 0 }

theorem complement_union :
  (compl S) ∪ T = { x : ℝ | x ≤ 1 } :=
sorry

end complement_union_l1933_193364


namespace ratio_men_to_women_l1933_193337

theorem ratio_men_to_women
  (W M : ℕ)      -- W is the number of women, M is the number of men
  (avg_height_all : ℕ) (avg_height_female : ℕ) (avg_height_male : ℕ)
  (h1 : avg_height_all = 180)
  (h2 : avg_height_female = 170)
  (h3 : avg_height_male = 182) 
  (h_avg : (170 * W + 182 * M) / (W + M) = 180) :
  M = 5 * W :=
by
  sorry

end ratio_men_to_women_l1933_193337


namespace anne_gave_sweettarts_to_three_friends_l1933_193397

theorem anne_gave_sweettarts_to_three_friends (sweettarts : ℕ) (eaten : ℕ) (friends : ℕ) 
  (h1 : sweettarts = 15) (h2 : eaten = 5) (h3 : sweettarts = friends * eaten) :
  friends = 3 := 
by 
  sorry

end anne_gave_sweettarts_to_three_friends_l1933_193397


namespace min_val_m_l1933_193375

theorem min_val_m (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h : 24 * m = n ^ 4) : m = 54 :=
sorry

end min_val_m_l1933_193375


namespace probability_of_finding_transmitter_l1933_193379

def total_license_plates : ℕ := 900
def inspected_vehicles : ℕ := 18

theorem probability_of_finding_transmitter : (inspected_vehicles : ℝ) / (total_license_plates : ℝ) = 0.02 :=
by
  sorry

end probability_of_finding_transmitter_l1933_193379


namespace suraj_average_after_9th_innings_l1933_193372

theorem suraj_average_after_9th_innings (A : ℕ) 
  (h1 : 8 * A + 90 = 9 * (A + 6)) : 
  (A + 6) = 42 :=
by
  sorry

end suraj_average_after_9th_innings_l1933_193372


namespace solve_x_l1933_193355

theorem solve_x (x : ℝ) (h : (x / 3) / 5 = 5 / (x / 3)) : x = 15 ∨ x = -15 :=
by sorry

end solve_x_l1933_193355


namespace a_5_eq_neg1_l1933_193378

-- Given conditions
def S (n : ℕ) : ℤ := n^2 - 10 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

-- The theorem to prove
theorem a_5_eq_neg1 : a 5 = -1 :=
by sorry

end a_5_eq_neg1_l1933_193378


namespace MarysTotalCandies_l1933_193380

-- Definitions for the conditions
def MegansCandies : Nat := 5
def MarysInitialCandies : Nat := 3 * MegansCandies
def MarysCandiesAfterAdding : Nat := MarysInitialCandies + 10

-- Theorem to prove that Mary has 25 pieces of candy in total
theorem MarysTotalCandies : MarysCandiesAfterAdding = 25 :=
by
  sorry

end MarysTotalCandies_l1933_193380


namespace B_value_l1933_193329

theorem B_value (A B : Nat) (hA : A < 10) (hB : B < 10) (h_div99 : (100000 * A + 10000 + 1000 * 5 + 100 * B + 90 + 4) % 99 = 0) :
  B = 3 :=
by
  -- skipping the proof
  sorry

end B_value_l1933_193329


namespace largest_stickers_per_page_l1933_193346

theorem largest_stickers_per_page :
  Nat.gcd (Nat.gcd 1050 1260) 945 = 105 := 
sorry

end largest_stickers_per_page_l1933_193346


namespace isabel_piggy_bank_l1933_193305

theorem isabel_piggy_bank:
  ∀ (initial_amount spent_on_toy spent_on_book remaining_amount : ℕ),
  initial_amount = 204 →
  spent_on_toy = initial_amount / 2 →
  remaining_amount = initial_amount - spent_on_toy →
  spent_on_book = remaining_amount / 2 →
  remaining_amount - spent_on_book = 51 :=
by
  sorry

end isabel_piggy_bank_l1933_193305


namespace jacket_initial_reduction_percent_l1933_193387

theorem jacket_initial_reduction_percent (P : ℝ) (x : ℝ) (h : P * (1 - x / 100) * 0.70 * 1.5873 = P) : x = 10 :=
sorry

end jacket_initial_reduction_percent_l1933_193387


namespace k_positive_first_third_quadrants_l1933_193322

theorem k_positive_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k*x > 0) ∧ (x < 0 → k*x < 0)) → k > 0 :=
by
  sorry

end k_positive_first_third_quadrants_l1933_193322


namespace baseball_card_difference_l1933_193323

theorem baseball_card_difference (marcus_cards carter_cards : ℕ) (h1 : marcus_cards = 210) (h2 : carter_cards = 152) : marcus_cards - carter_cards = 58 :=
by {
    --skip the proof
    sorry
}

end baseball_card_difference_l1933_193323


namespace parabola_hyperbola_intersection_l1933_193326

open Real

theorem parabola_hyperbola_intersection (p : ℝ) (hp : p > 0)
  (h_hyperbola : ∀ x y, (x^2 / 4 - y^2 = 1) → (y = 2*x ∨ y = -2*x))
  (h_parabola_directrix : ∀ y, (x^2 = 2 * p * y) → (x = -p/2)) 
  (h_area_triangle : (1/2) * (p/2) * (2 * p) = 1) :
  p = sqrt 2 := sorry

end parabola_hyperbola_intersection_l1933_193326


namespace sum_of_third_terms_arithmetic_progressions_l1933_193321

theorem sum_of_third_terms_arithmetic_progressions
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (d1 d2 : ℕ)
  (h1 : ∃ d1 : ℕ, ∀ n : ℕ, a (n + 1) = a 1 + n * d1)
  (h2 : ∃ d2 : ℕ, ∀ n : ℕ, b (n + 1) = b 1 + n * d2)
  (h3 : a 1 + b 1 = 7)
  (h4 : a 5 + b 5 = 35) :
  a 3 + b 3 = 21 :=
by
  sorry

end sum_of_third_terms_arithmetic_progressions_l1933_193321


namespace team_a_games_played_l1933_193314

theorem team_a_games_played (a b: ℕ) (hA_wins : 3 * a = 4 * wins_A)
(hB_wins : 2 * b = 3 * wins_B)
(hB_more_wins : wins_B = wins_A + 8)
(hB_more_loss : b - wins_B = a - wins_A + 8) :
  a = 192 := 
by
  sorry

end team_a_games_played_l1933_193314


namespace train_length_l1933_193356

theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 300)
  (h2 : time = 33) : (speed * 1000 / 3600) * time = 2750 := by
  sorry

end train_length_l1933_193356


namespace sixty_percent_of_N_l1933_193395

noncomputable def N : ℝ :=
  let x := (45 : ℝ)
  let frac := (3/4 : ℝ) * (1/3) * (2/5) * (1/2)
  20 * x / frac

theorem sixty_percent_of_N : (0.60 : ℝ) * N = 540 := by
  sorry

end sixty_percent_of_N_l1933_193395


namespace second_number_is_twenty_two_l1933_193334

theorem second_number_is_twenty_two (x y : ℕ) 
  (h1 : x + y = 33) 
  (h2 : y = 2 * x) : 
  y = 22 :=
by
  sorry

end second_number_is_twenty_two_l1933_193334


namespace milk_tea_sales_l1933_193386

-- Definitions
def relationship (x y : ℕ) : Prop := y = 10 * x + 2

-- Theorem statement
theorem milk_tea_sales (x y : ℕ) :
  relationship x y → (y = 822 → x = 82) :=
by
  intros h_rel h_y
  sorry

end milk_tea_sales_l1933_193386


namespace distance_to_workplace_l1933_193389

def driving_speed : ℕ := 40
def driving_time : ℕ := 3
def total_distance := driving_speed * driving_time
def one_way_distance := total_distance / 2

theorem distance_to_workplace : one_way_distance = 60 := by
  sorry

end distance_to_workplace_l1933_193389


namespace largest_cannot_be_sum_of_two_composites_l1933_193315

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l1933_193315


namespace pyramid_volume_eq_l1933_193371

noncomputable def volume_of_pyramid (base_length1 base_length2 height : ℝ) : ℝ :=
  (1 / 3) * base_length1 * base_length2 * height

theorem pyramid_volume_eq (base_length1 base_length2 height : ℝ) (h1 : base_length1 = 1) (h2 : base_length2 = 2) (h3 : height = 1) :
  volume_of_pyramid base_length1 base_length2 height = 2 / 3 := by
  sorry

end pyramid_volume_eq_l1933_193371


namespace number_of_lattice_points_l1933_193336

theorem number_of_lattice_points (A B : ℝ) (h : B - A = 10) :
  ∃ n, n = 10 ∨ n = 11 :=
sorry

end number_of_lattice_points_l1933_193336


namespace circles_tangent_area_l1933_193347

noncomputable def triangle_area (r1 r2 r3 : ℝ) := 
  let d1 := r1 + r2
  let d2 := r2 + r3
  let d3 := r1 + r3
  let s := (d1 + d2 + d3) / 2
  (s * (s - d1) * (s - d2) * (s - d3)).sqrt

theorem circles_tangent_area :
  let r1 := 5
  let r2 := 12
  let r3 := 13
  let area := triangle_area r1 r2 r3 / (4 * (r1 + r2 + r3)).sqrt
  area = 120 / 25 := 
by 
  sorry

end circles_tangent_area_l1933_193347


namespace work_completion_time_l1933_193384

-- Let's define the initial conditions
def total_days := 100
def initial_people := 10
def days1 := 20
def work_done1 := 1 / 4
def days2 (remaining_work_per_person: ℚ) := (3/4) / remaining_work_per_person
def remaining_people := initial_people - 2
def remaining_work_per_person_per_day := remaining_people * (work_done1 / (initial_people * days1))

-- Theorem stating that the total number of days to complete the work is 95
theorem work_completion_time : 
  days1 + days2 remaining_work_per_person_per_day = 95 := 
  by
    sorry -- Proof to be filled in

end work_completion_time_l1933_193384


namespace number_of_petri_dishes_l1933_193369

noncomputable def total_germs : ℝ := 0.036 * 10^5
noncomputable def germs_per_dish : ℝ := 99.99999999999999

theorem number_of_petri_dishes : 36 = total_germs / germs_per_dish :=
by sorry

end number_of_petri_dishes_l1933_193369


namespace total_spending_l1933_193310

-- Conditions used as definitions
def price_pants : ℝ := 110.00
def discount_pants : ℝ := 0.30
def number_of_pants : ℕ := 4

def price_socks : ℝ := 60.00
def discount_socks : ℝ := 0.30
def number_of_socks : ℕ := 2

-- Lean 4 statement to prove the total spending
theorem total_spending :
  (number_of_pants : ℝ) * (price_pants * (1 - discount_pants)) +
  (number_of_socks : ℝ) * (price_socks * (1 - discount_socks)) = 392.00 :=
by
  sorry

end total_spending_l1933_193310


namespace inequality_proof_l1933_193306

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 9 * y + 3 * z) * (x + 4 * y + 2 * z) * (2 * x + 12 * y + 9 * z) ≥ 1029 * x * y * z :=
by
  sorry

end inequality_proof_l1933_193306


namespace evaluate_expression_l1933_193313

theorem evaluate_expression : (2 * (-1) + 3) * (2 * (-1) - 3) - ((-1) - 1) * ((-1) + 5) = 3 := by
  sorry

end evaluate_expression_l1933_193313


namespace problem1_problem2_l1933_193390

open Nat

def binomial (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem problem1 : binomial 8 5 + binomial 100 98 * binomial 7 7 = 5006 := by
  sorry

theorem problem2 : binomial 5 0 + binomial 5 1 + binomial 5 2 + binomial 5 3 + binomial 5 4 + binomial 5 5 = 32 := by
  sorry

end problem1_problem2_l1933_193390


namespace alyssa_kittens_l1933_193345

theorem alyssa_kittens (original_kittens given_away: ℕ) (h1: original_kittens = 8) (h2: given_away = 4) :
  original_kittens - given_away = 4 :=
by
  sorry

end alyssa_kittens_l1933_193345


namespace min_f_abs_l1933_193382

def f (x y : ℤ) : ℤ := 5 * x^2 + 11 * x * y - 5 * y^2

theorem min_f_abs (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) : (∃ m, ∀ x y : ℤ, (x ≠ 0 ∨ y ≠ 0) → |f x y| ≥ m) ∧ 5 = 5 :=
by
  sorry -- proof goes here

end min_f_abs_l1933_193382


namespace inheritance_amount_l1933_193318

def federalTax (x : ℝ) : ℝ := 0.25 * x
def remainingAfterFederalTax (x : ℝ) : ℝ := x - federalTax x
def stateTax (x : ℝ) : ℝ := 0.15 * remainingAfterFederalTax x
def totalTaxes (x : ℝ) : ℝ := federalTax x + stateTax x

theorem inheritance_amount (x : ℝ) (h : totalTaxes x = 15000) : x = 41379 :=
by
  sorry

end inheritance_amount_l1933_193318


namespace find_B_age_l1933_193324

variable (a b c : ℕ)

def problem_conditions : Prop :=
  a = b + 2 ∧ b = 2 * c ∧ a + b + c = 22

theorem find_B_age (h : problem_conditions a b c) : b = 8 :=
by {
  sorry
}

end find_B_age_l1933_193324


namespace not_prime_257_1092_1092_l1933_193376

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem not_prime_257_1092_1092 :
  is_prime 1093 →
  ¬ is_prime (257 ^ 1092 + 1092) :=
by
  intro h_prime_1093
  -- Detailed steps are omitted, proof goes here
  sorry

end not_prime_257_1092_1092_l1933_193376


namespace complement_of_M_in_U_l1933_193308

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5, 6} := by
  sorry

end complement_of_M_in_U_l1933_193308


namespace common_value_of_7a_and_2b_l1933_193303

variable (a b : ℝ)

theorem common_value_of_7a_and_2b (h1 : 7 * a = 2 * b) (h2 : 42 * a * b = 674.9999999999999) :
  7 * a = 15 :=
by
  -- This place will contain the proof steps
  sorry

end common_value_of_7a_and_2b_l1933_193303


namespace problem_1_problem_2_l1933_193377

-- Problem 1 Lean statement
theorem problem_1 :
  (1 - 1^4 - (1/2) * (3 - (-3)^2)) = 2 :=
by sorry

-- Problem 2 Lean statement
theorem problem_2 :
  ((3/8 - 1/6 - 3/4) * 24) = -13 :=
by sorry

end problem_1_problem_2_l1933_193377


namespace hyperbola_equation_l1933_193330

theorem hyperbola_equation (h : ∃ (x y : ℝ), y = 1 / 2 * x) (p : (2, 2) ∈ {p : ℝ × ℝ | ((p.snd)^2 / 3) - ((p.fst)^2 / 12) = 1}) :
  ∀ (x y : ℝ), (y^2 / 3 - x^2 / 12 = 1) ↔ (∃ (a b : ℝ), y = a * x ∧ b * y = x ^ 2) :=
sorry

end hyperbola_equation_l1933_193330


namespace initial_floor_l1933_193394

theorem initial_floor (x y z : ℤ)
  (h1 : y = x - 7)
  (h2 : z = y + 3)
  (h3 : 13 = z + 8) :
  x = 9 :=
sorry

end initial_floor_l1933_193394


namespace number_neither_9_nice_nor_10_nice_500_l1933_193351

def is_k_nice (N k : ℕ) : Prop := ∃ a : ℕ, a > 0 ∧ (∃ m : ℕ, N = (k * m) + 1)

def count_k_nice (N k : ℕ) : ℕ :=
  (N - 1) / k + 1

def count_neither_9_nice_nor_10_nice (N : ℕ) : ℕ :=
  let count_9_nice := count_k_nice N 9
  let count_10_nice := count_k_nice N 10
  let lcm_9_10 := 90  -- lcm of 9 and 10
  let count_both := count_k_nice N lcm_9_10
  N - (count_9_nice + count_10_nice - count_both)

theorem number_neither_9_nice_nor_10_nice_500 : count_neither_9_nice_nor_10_nice 500 = 400 :=
  sorry

end number_neither_9_nice_nor_10_nice_500_l1933_193351


namespace sphere_cube_volume_ratio_l1933_193335

theorem sphere_cube_volume_ratio (d a : ℝ) (h_d : d = 12) (h_a : a = 6) :
  let r := d / 2
  let V_sphere := (4 / 3) * π * r^3
  let V_cube := a^3
  V_sphere / V_cube = (4 * π) / 3 :=
by
  sorry

end sphere_cube_volume_ratio_l1933_193335


namespace smallest_positive_solution_l1933_193333

theorem smallest_positive_solution :
  ∃ x : ℝ, x > 0 ∧ (x ^ 4 - 50 * x ^ 2 + 576 = 0) ∧ (∀ y : ℝ, y > 0 ∧ y ^ 4 - 50 * y ^ 2 + 576 = 0 → x ≤ y) ∧ x = 3 * Real.sqrt 2 :=
sorry

end smallest_positive_solution_l1933_193333


namespace breadth_of_rectangular_plot_l1933_193383

theorem breadth_of_rectangular_plot (b : ℝ) (A : ℝ) (l : ℝ)
  (h1 : A = 20 * b)
  (h2 : l = b + 10)
  (h3 : A = l * b) : b = 10 := by
  sorry

end breadth_of_rectangular_plot_l1933_193383


namespace miles_to_add_per_week_l1933_193319

theorem miles_to_add_per_week :
  ∀ (initial_miles_per_week : ℝ) 
    (percentage_increase : ℝ) 
    (total_days : ℕ) 
    (days_in_week : ℕ),
    initial_miles_per_week = 100 →
    percentage_increase = 0.2 →
    total_days = 280 →
    days_in_week = 7 →
    ((initial_miles_per_week * (1 + percentage_increase)) - initial_miles_per_week) / (total_days / days_in_week) = 3 :=
by
  intros initial_miles_per_week percentage_increase total_days days_in_week
  intros h1 h2 h3 h4
  sorry

end miles_to_add_per_week_l1933_193319


namespace problem_1_problem_2_l1933_193344
-- Import the entire Mathlib library.

-- Problem (1)
theorem problem_1 (x y : ℝ) (h1 : |x - 3 * y| < 1 / 2) (h2 : |x + 2 * y| < 1 / 6) : |x| < 3 / 10 :=
sorry

-- Problem (2)
theorem problem_2 (x y : ℝ) : x^4 + 16 * y^4 ≥ 2 * x^3 * y + 8 * x * y^3 :=
sorry

end problem_1_problem_2_l1933_193344


namespace prism_faces_l1933_193398

-- Define the number of edges in a prism and the function to calculate faces based on edges.
def number_of_faces (E : ℕ) : ℕ :=
  if E % 3 = 0 then (E / 3) + 2 else 0

-- The main statement to be proven: A prism with 18 edges has 8 faces.
theorem prism_faces (E : ℕ) (hE : E = 18) : number_of_faces E = 8 :=
by
  -- Just outline the argument here for clarity, the detailed steps will go in an actual proof.
  sorry

end prism_faces_l1933_193398


namespace yard_length_l1933_193399

theorem yard_length (n : ℕ) (d : ℕ) (k : ℕ) (h : k = n - 1) (hd : d = 5) (hn : n = 51) : (k * d) = 250 := 
by
  sorry

end yard_length_l1933_193399


namespace x_value_when_y_2000_l1933_193388

noncomputable def x_when_y_2000 (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (hxy_inv : ∀ x' y', x'^3 * y' = x^3 * y) (h_init : x = 2 ∧ y = 5) : ℝ :=
  if hy : y = 2000 then (1 / (50 : ℝ)^(1/3)) else x

-- Theorem statement
theorem x_value_when_y_2000 (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (hxy_inv : ∀ x' y', x'^3 * y' = x^3 * y) (h_init : x = 2 ∧ y = 5) :
  x_when_y_2000 x y hxy_pos hxy_inv h_init = 1 / (50 : ℝ)^(1/3) :=
sorry

end x_value_when_y_2000_l1933_193388


namespace calculate_dividend_l1933_193391

def faceValue : ℕ := 100
def premiumPercent : ℕ := 20
def dividendPercent : ℕ := 5
def investment : ℕ := 14400
def costPerShare : ℕ := faceValue + (premiumPercent * faceValue / 100)
def numberOfShares : ℕ := investment / costPerShare
def dividendPerShare : ℕ := faceValue * dividendPercent / 100
def totalDividend : ℕ := numberOfShares * dividendPerShare

theorem calculate_dividend :
  totalDividend = 600 := 
by
  sorry

end calculate_dividend_l1933_193391


namespace expression_of_y_l1933_193363

theorem expression_of_y (x y : ℝ) (h : x - y / 2 = 1) : y = 2 * x - 2 :=
sorry

end expression_of_y_l1933_193363


namespace both_complementary_angles_acute_is_certain_event_l1933_193302

def complementary_angles (A B : ℝ) : Prop :=
  A + B = 90

def acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

theorem both_complementary_angles_acute_is_certain_event (A B : ℝ) (h1 : complementary_angles A B) (h2 : acute_angle A) (h3 : acute_angle B) : (A < 90) ∧ (B < 90) :=
by
  sorry

end both_complementary_angles_acute_is_certain_event_l1933_193302


namespace problem_1_problem_2_l1933_193339

noncomputable def f (x a : ℝ) : ℝ := abs (x + a) + abs (x - 2)

-- (1) Prove that, given f(x) and a = -3, the solution set for f(x) ≥ 3 is (-∞, 1] ∪ [4, +∞)
theorem problem_1 (x : ℝ) : 
  (∃ (a : ℝ), a = -3 ∧ f x a ≥ 3) ↔ (x ≤ 1 ∨ x ≥ 4) :=
sorry

-- (2) Prove that for f(x) to be ≥ 3 for all x, the range of a is a ≥ 1 or a ≤ -5
theorem problem_2 : 
  (∀ (x : ℝ), f x a ≥ 3) ↔ (a ≥ 1 ∨ a ≤ -5) :=
sorry

end problem_1_problem_2_l1933_193339


namespace base_conversion_subtraction_l1933_193301

theorem base_conversion_subtraction :
  let n1 := 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 4 * 7^1 + 3 * 7^0
  let n2 := 1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0
  n1 - n2 = 7422 :=
by
  let n1 := 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 4 * 7^1 + 3 * 7^0
  let n2 := 1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0
  show n1 - n2 = 7422
  sorry

end base_conversion_subtraction_l1933_193301


namespace find_C_l1933_193348

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 300) 
  (h2 : A + C = 200) 
  (h3 : B + C = 350) : 
  C = 250 := 
  by sorry

end find_C_l1933_193348


namespace cos_240_degree_l1933_193327

theorem cos_240_degree : Real.cos (240 * Real.pi / 180) = -1/2 :=
by
  sorry

end cos_240_degree_l1933_193327


namespace rectangle_area_l1933_193362

variable (l w : ℕ)

-- Conditions
def length_eq_four_times_width (l w : ℕ) : Prop := l = 4 * w
def perimeter_eq_200 (l w : ℕ) : Prop := 2 * l + 2 * w = 200

-- Question: Prove the area is 1600 square centimeters
theorem rectangle_area (h1 : length_eq_four_times_width l w) (h2 : perimeter_eq_200 l w) :
  l * w = 1600 := by
  sorry

end rectangle_area_l1933_193362


namespace smallest_value_of_2a_plus_1_l1933_193385

theorem smallest_value_of_2a_plus_1 (a : ℝ) 
  (h : 6 * a^2 + 5 * a + 4 = 3) : 
  ∃ b : ℝ, b = 2 * a + 1 ∧ b = 0 := 
sorry

end smallest_value_of_2a_plus_1_l1933_193385


namespace minimum_value_l1933_193349

theorem minimum_value (x : ℝ) (h : x > -3) : 2 * x + (1 / (x + 3)) ≥ 2 * Real.sqrt 2 - 6 :=
sorry

end minimum_value_l1933_193349


namespace train_passing_time_correct_l1933_193366

noncomputable def train_passing_time (L1 L2 : ℕ) (S1 S2 : ℕ) : ℝ :=
  let S1_mps := S1 * (1000 / 3600)
  let S2_mps := S2 * (1000 / 3600)
  let relative_speed := S1_mps + S2_mps
  let total_length := L1 + L2
  total_length / relative_speed

theorem train_passing_time_correct :
  train_passing_time 105 140 45 36 = 10.89 := by
  sorry

end train_passing_time_correct_l1933_193366


namespace fraction_bad_teams_leq_l1933_193350

variable (teams total_teams : ℕ) (b : ℝ)

-- Given conditions
variable (cond₁ : total_teams = 18)
variable (cond₂ : teams = total_teams / 2)
variable (cond₃ : ∀ (rb_teams : ℕ), rb_teams ≠ 10 → rb_teams ≤ teams)

theorem fraction_bad_teams_leq (H : 18 * b ≤ teams) : b ≤ 1 / 2 :=
sorry

end fraction_bad_teams_leq_l1933_193350


namespace find_a_l1933_193352

theorem find_a (A B : Set ℝ) (a : ℝ)
  (hA : A = {1, 2})
  (hB : B = {a, a^2 + 1})
  (hUnion : A ∪ B = {0, 1, 2}) :
  a = 0 :=
sorry

end find_a_l1933_193352


namespace sum_zero_inv_sum_zero_a_plus_d_zero_l1933_193331

theorem sum_zero_inv_sum_zero_a_plus_d_zero 
  (a b c d : ℝ) (h1 : a ≤ b ∧ b ≤ c ∧ c ≤ d) 
  (h2 : a + b + c + d = 0) 
  (h3 : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 := 
  sorry

end sum_zero_inv_sum_zero_a_plus_d_zero_l1933_193331


namespace average_shifted_data_is_7_l1933_193311

variable (x1 x2 x3 : ℝ)

theorem average_shifted_data_is_7 (h : (x1 + x2 + x3) / 3 = 5) : 
  ((x1 + 2) + (x2 + 2) + (x3 + 2)) / 3 = 7 :=
by
  sorry

end average_shifted_data_is_7_l1933_193311


namespace problem1_problem2_l1933_193361

-- Definitions for Problem 1
def cond1 (x t : ℝ) : Prop := |2 * x + t| - t ≤ 8
def sol_set1 (x : ℝ) : Prop := -5 ≤ x ∧ x ≤ 4

theorem problem1 {t : ℝ} : (∀ x, cond1 x t → sol_set1 x) → t = 1 :=
sorry

-- Definitions for Problem 2
def cond2 (x y z : ℝ) : Prop := x^2 + (1 / 4) * y^2 + (1 / 9) * z^2 = 2

theorem problem2 {x y z : ℝ} : cond2 x y z → x + y + z ≤ 2 * Real.sqrt 7 :=
sorry

end problem1_problem2_l1933_193361


namespace not_divisible_1998_minus_1_by_1000_minus_1_l1933_193304

theorem not_divisible_1998_minus_1_by_1000_minus_1 (m : ℕ) : ¬ (1000^m - 1 ∣ 1998^m - 1) :=
sorry

end not_divisible_1998_minus_1_by_1000_minus_1_l1933_193304


namespace tom_profit_calculation_l1933_193309

theorem tom_profit_calculation :
  let flour_needed := 500
  let flour_per_bag := 50
  let flour_bag_cost := 20
  let salt_needed := 10
  let salt_cost_per_pound := 0.2
  let promotion_cost := 1000
  let tickets_sold := 500
  let ticket_price := 20

  let flour_bags := flour_needed / flour_per_bag
  let cost_flour := flour_bags * flour_bag_cost
  let cost_salt := salt_needed * salt_cost_per_pound
  let total_expenses := cost_flour + cost_salt + promotion_cost
  let total_revenue := tickets_sold * ticket_price

  let profit := total_revenue - total_expenses

  profit = 8798 := by
  sorry

end tom_profit_calculation_l1933_193309


namespace unique_solution_l1933_193353

theorem unique_solution (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (eq1 : x * y + y * z + z * x = 12) (eq2 : x * y * z = 2 + x + y + z) :
  x = 2 ∧ y = 2 ∧ z = 2 :=
by 
  sorry

end unique_solution_l1933_193353


namespace sum_of_square_roots_of_consecutive_odd_numbers_l1933_193360

theorem sum_of_square_roots_of_consecutive_odd_numbers :
  (Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9)) = 15 :=
by
  sorry

end sum_of_square_roots_of_consecutive_odd_numbers_l1933_193360


namespace eq_proof_l1933_193358

noncomputable def S_even : ℚ := 28
noncomputable def S_odd : ℚ := 24

theorem eq_proof : ( (S_even / S_odd - S_odd / S_even) * 2 ) = (13 / 21) :=
by
  sorry

end eq_proof_l1933_193358


namespace smallest_square_number_l1933_193393

theorem smallest_square_number (x y : ℕ) (hx : ∃ a, x = a ^ 2) (hy : ∃ b, y = b ^ 3) 
  (h_simp: ∃ c d, x / (y ^ 3) = c ^ 3 / d ^ 2 ∧ c > 1 ∧ d > 1): x = 64 := by
  sorry

end smallest_square_number_l1933_193393


namespace find_triples_l1933_193317

theorem find_triples (x p n : ℕ) (hp : Nat.Prime p) :
  2 * x * (x + 5) = p^n + 3 * (x - 1) →
  (x = 2 ∧ p = 5 ∧ n = 2) ∨ (x = 0 ∧ p = 3 ∧ n = 1) :=
by
  sorry

end find_triples_l1933_193317


namespace river_flow_volume_l1933_193338

theorem river_flow_volume (depth width : ℝ) (flow_rate_kmph : ℝ) :
  depth = 3 → width = 36 → flow_rate_kmph = 2 → 
  (depth * width) * (flow_rate_kmph * 1000 / 60) = 3599.64 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end river_flow_volume_l1933_193338


namespace find_fourth_number_l1933_193342

variables (A B C D E F : ℝ)

theorem find_fourth_number
  (h1 : A + B + C + D + E + F = 180)
  (h2 : A + B + C + D = 100)
  (h3 : D + E + F = 105) :
  D = 25 :=
by
  sorry

end find_fourth_number_l1933_193342


namespace john_chips_consumption_l1933_193370

/-- John starts the week with a routine. Every day, he eats one bag of chips for breakfast,
  two bags for lunch, and doubles the amount he had for lunch for dinner.
  Prove that by the end of the week, John consumed 49 bags of chips. --/
theorem john_chips_consumption : 
  ∀ (days_in_week : ℕ) (chips_breakfast : ℕ) (chips_lunch : ℕ) (chips_dinner : ℕ), 
    days_in_week = 7 ∧ chips_breakfast = 1 ∧ chips_lunch = 2 ∧ chips_dinner = 2 * chips_lunch →
    days_in_week * (chips_breakfast + chips_lunch + chips_dinner) = 49 :=
by
  intros days_in_week chips_breakfast chips_lunch chips_dinner
  sorry

end john_chips_consumption_l1933_193370


namespace domino_cover_grid_l1933_193367

-- Definitions representing the conditions:
def isPositive (n : ℕ) : Prop := n > 0
def divides (a b : ℕ) : Prop := ∃ k, b = k * a
def canCoverWithDominos (n k : ℕ) : Prop := ∀ i j, (i < n) → (j < n) → (∃ r, i = r * k ∨ j = r * k)

-- The hypothesis: n and k are positive integers
axiom n : ℕ
axiom k : ℕ
axiom n_positive : isPositive n
axiom k_positive : isPositive k

-- The main theorem
theorem domino_cover_grid (n k : ℕ) (n_positive : isPositive n) (k_positive : isPositive k) :
  canCoverWithDominos n k ↔ divides k n := by
  sorry

end domino_cover_grid_l1933_193367


namespace sum_difference_l1933_193343

def arithmetic_series_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def set_A_sum : ℕ :=
  arithmetic_series_sum 42 2 25

def set_B_sum : ℕ :=
  arithmetic_series_sum 62 2 25

theorem sum_difference :
  set_B_sum - set_A_sum = 500 :=
by
  sorry

end sum_difference_l1933_193343


namespace resized_height_l1933_193320

-- Define original dimensions
def original_width : ℝ := 4.5
def original_height : ℝ := 3

-- Define new width
def new_width : ℝ := 13.5

-- Define new height to be proven
def new_height : ℝ := 9

-- Theorem statement
theorem resized_height :
  (new_width / original_width) * original_height = new_height :=
by
  -- The statement that equates the new height calculated proportionately to 9
  sorry

end resized_height_l1933_193320


namespace books_read_in_common_l1933_193381

theorem books_read_in_common (T D B total X : ℕ) 
  (hT : T = 23) 
  (hD : D = 12) 
  (hB : B = 17) 
  (htotal : total = 47)
  (h_eq : (T - X) + (D - X) + B + 1 = total) : 
  X = 3 :=
by
  -- Here would go the proof details.
  sorry

end books_read_in_common_l1933_193381


namespace maximum_delta_value_l1933_193325

-- Definition of the sequence a 
def a (n : ℕ) : ℕ := 1 + n^3

-- Definition of δ_n as the gcd of consecutive terms in the sequence a
def delta (n : ℕ) : ℕ := Nat.gcd (a (n + 1)) (a n)

-- Main theorem statement
theorem maximum_delta_value : ∃ n, delta n = 7 :=
by
  -- Insert the proof later
  sorry

end maximum_delta_value_l1933_193325


namespace greatest_monthly_drop_l1933_193396

-- Definition of monthly price changes
def price_change_jan : ℝ := -1.00
def price_change_feb : ℝ := 2.50
def price_change_mar : ℝ := 0.00
def price_change_apr : ℝ := -3.00
def price_change_may : ℝ := -1.50
def price_change_jun : ℝ := 1.00

-- Proving the month with the greatest monthly drop in price
theorem greatest_monthly_drop :
  (price_change_apr < price_change_jan) ∧
  (price_change_apr < price_change_feb) ∧
  (price_change_apr < price_change_mar) ∧
  (price_change_apr < price_change_may) ∧
  (price_change_apr < price_change_jun) :=
by
  sorry

end greatest_monthly_drop_l1933_193396


namespace hyperbola_eccentricity_is_4_l1933_193332

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h_eq1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : ∀ y : ℝ, y^2 = 16 * (4 : ℝ))
  (h_focus : c = 4)
: ℝ := c / a

theorem hyperbola_eccentricity_is_4 (a b c : ℝ)
  (h_eq1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : ∀ y : ℝ, y^2 = 16 * (4 : ℝ))
  (h_focus : c = 4)
  (h_c2 : c^2 = a^2 + b^2)
  (h_bc : b^2 = a^2 * (c^2 / a^2 - 1))
: hyperbola_eccentricity a b c h_eq1 h_eq2 h_focus = 4 := by
  sorry

end hyperbola_eccentricity_is_4_l1933_193332


namespace bucket_P_turns_to_fill_the_drum_l1933_193373

-- Define the capacities of the buckets
def capacity_P := 3
def capacity_Q := 1

-- Define the total number of turns for both buckets together to fill the drum
def turns_together := 60

-- Define the total capacity of the drum that gets filled in the given scenario of the problem
def total_capacity := turns_together * (capacity_P + capacity_Q)

-- The question: How many turns does it take for bucket P alone to fill this total capacity?
def turns_P_alone : ℕ :=
  total_capacity / capacity_P

theorem bucket_P_turns_to_fill_the_drum :
  turns_P_alone = 80 :=
by
  sorry

end bucket_P_turns_to_fill_the_drum_l1933_193373


namespace cylindrical_to_rectangular_coords_l1933_193300

/--
Cylindrical coordinates (r, θ, z)
Rectangular coordinates (x, y, z)
-/
theorem cylindrical_to_rectangular_coords (r θ z : ℝ) (hx : x = r * Real.cos θ)
    (hy : y = r * Real.sin θ) (hz : z = z) :
    (r, θ, z) = (5, Real.pi / 4, 2) → (x, y, z) = (5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 2) :=
by
  sorry

end cylindrical_to_rectangular_coords_l1933_193300


namespace spent_on_video_game_l1933_193341

def saved_September : ℕ := 30
def saved_October : ℕ := 49
def saved_November : ℕ := 46
def money_left : ℕ := 67
def total_saved := saved_September + saved_October + saved_November

theorem spent_on_video_game : total_saved - money_left = 58 := by
  -- proof steps go here
  sorry

end spent_on_video_game_l1933_193341


namespace sequence_general_formula_l1933_193374

theorem sequence_general_formula (a : ℕ+ → ℝ) (h₀ : a 1 = 7 / 8)
  (h₁ : ∀ n : ℕ+, a (n + 1) = 1 / 2 * a n + 1 / 3) :
  ∀ n : ℕ+, a n = 5 / 24 * (1 / 2)^(n - 1 : ℕ) + 2 / 3 :=
by
  sorry

end sequence_general_formula_l1933_193374
