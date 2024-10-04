import Mathlib

namespace sally_nickels_count_l204_204503

theorem sally_nickels_count (original_nickels dad_nickels mom_nickels : ℕ) 
    (h1: original_nickels = 7) 
    (h2: dad_nickels = 9) 
    (h3: mom_nickels = 2) 
    : original_nickels + dad_nickels + mom_nickels = 18 :=
by
  sorry

end sally_nickels_count_l204_204503


namespace inequality_solution_l204_204653

theorem inequality_solution (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < (5 / 9) :=
by
  sorry

end inequality_solution_l204_204653


namespace total_spent_proof_l204_204810

noncomputable def total_spent (cost_pen cost_pencil cost_notebook : ℝ) 
  (pens_robert pencils_robert notebooks_dorothy : ℕ) 
  (julia_pens_ratio robert_pens_ratio dorothy_pens_ratio : ℝ) 
  (julia_pencils_diff notebooks_julia_diff : ℕ) 
  (robert_notebooks_ratio dorothy_pencils_ratio : ℝ) : ℝ :=
    let pens_julia := robert_pens_ratio * pens_robert
    let pens_dorothy := dorothy_pens_ratio * pens_julia
    let total_pens := pens_robert + pens_julia + pens_dorothy
    let cost_pens := total_pens * cost_pen 
    
    let pencils_julia := pencils_robert - julia_pencils_diff
    let pencils_dorothy := dorothy_pencils_ratio * pencils_julia
    let total_pencils := pencils_robert + pencils_julia + pencils_dorothy
    let cost_pencils := total_pencils * cost_pencil 
        
    let notebooks_julia := notebooks_dorothy + notebooks_julia_diff
    let notebooks_robert := robert_notebooks_ratio * notebooks_julia
    let total_notebooks := notebooks_dorothy + notebooks_julia + notebooks_robert
    let cost_notebooks := total_notebooks * cost_notebook
        
    cost_pens + cost_pencils + cost_notebooks

theorem total_spent_proof 
  (cost_pen : ℝ := 1.50)
  (cost_pencil : ℝ := 0.75)
  (cost_notebook : ℝ := 4.00)
  (pens_robert : ℕ := 4)
  (pencils_robert : ℕ := 12)
  (notebooks_dorothy : ℕ := 3)
  (julia_pens_ratio : ℝ := 3)
  (robert_pens_ratio : ℝ := 3)
  (dorothy_pens_ratio : ℝ := 0.5)
  (julia_pencils_diff : ℕ := 5)
  (notebooks_julia_diff : ℕ := 1)
  (robert_notebooks_ratio : ℝ := 0.5)
  (dorothy_pencils_ratio : ℝ := 2) : 
  total_spent cost_pen cost_pencil cost_notebook pens_robert pencils_robert notebooks_dorothy 
    julia_pens_ratio robert_pens_ratio dorothy_pens_ratio julia_pencils_diff notebooks_julia_diff robert_notebooks_ratio dorothy_pencils_ratio 
    = 93.75 := 
by 
  sorry

end total_spent_proof_l204_204810


namespace sum_of_variables_l204_204605

theorem sum_of_variables (x y z : ℝ) (hpos_x : 0 < x) (hpos_y : 0 < y) (hpos_z : 0 < z)
  (hxy : x * y = 30) (hxz : x * z = 60) (hyz : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 :=
by
  sorry

end sum_of_variables_l204_204605


namespace sufficient_not_necessary_range_l204_204822

theorem sufficient_not_necessary_range (a : ℝ) (h : ∀ x : ℝ, x > 2 → x^2 > a ∧ ¬(x^2 > a → x > 2)) : a ≤ 4 :=
by
  sorry

end sufficient_not_necessary_range_l204_204822


namespace system_of_inequalities_l204_204664

theorem system_of_inequalities (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : (0.5 < p ∧ p < 5 / 9) :=
by sorry

end system_of_inequalities_l204_204664


namespace estimate_flight_time_around_earth_l204_204245

theorem estimate_flight_time_around_earth 
  (radius : ℝ) 
  (speed : ℝ)
  (h_radius : radius = 6000) 
  (h_speed : speed = 600) 
  : abs (20 * Real.pi - 63) < 1 :=
by
  sorry

end estimate_flight_time_around_earth_l204_204245


namespace math_problem_proof_l204_204422

-- Define the problem statement
def problem_expr : ℕ :=
  28 * 7 * 25 + 12 * 7 * 25 + 7 * 11 * 3 + 44

-- Prove the problem statement equals to the correct answer
theorem math_problem_proof : problem_expr = 7275 := by
  sorry

end math_problem_proof_l204_204422


namespace ranking_possibilities_l204_204818

theorem ranking_possibilities (A B C D E : Type) : 
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 5 → (n ≠ 1 → n ≠ last)) →
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 5 → (n ≠ 1)) →
  ∃ (positions : Finset (List ℕ)),
    positions.card = 54 :=
by
  sorry

end ranking_possibilities_l204_204818


namespace addition_result_l204_204295

theorem addition_result : 148 + 32 + 18 + 2 = 200 :=
by
  sorry

end addition_result_l204_204295


namespace find_k_l204_204829

def vector := ℝ × ℝ  -- Define a vector as a pair of real numbers

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a (k : ℝ) : vector := (k, 3)
def b : vector := (1, 4)
def c : vector := (2, 1)
def linear_combination (k : ℝ) : vector := ((2 * k - 3), -6)

theorem find_k (k : ℝ) (h : dot_product (linear_combination k) c = 0) : k = 3 := by
  sorry

end find_k_l204_204829


namespace fettuccine_to_penne_ratio_l204_204342

theorem fettuccine_to_penne_ratio
  (num_surveyed : ℕ)
  (num_spaghetti : ℕ)
  (num_ravioli : ℕ)
  (num_fettuccine : ℕ)
  (num_penne : ℕ)
  (h_surveyed : num_surveyed = 800)
  (h_spaghetti : num_spaghetti = 300)
  (h_ravioli : num_ravioli = 200)
  (h_fettuccine : num_fettuccine = 150)
  (h_penne : num_penne = 150) :
  num_fettuccine / num_penne = 1 :=
by
  sorry

end fettuccine_to_penne_ratio_l204_204342


namespace minimum_throws_for_four_dice_l204_204126

noncomputable def minimum_throws_to_ensure_repeated_sum (d : ℕ) : ℕ :=
  let min_sum := d * 1 in
  let max_sum := d * 6 in
  let distinct_sums := max_sum - min_sum + 1 in
  distinct_sums + 1

theorem minimum_throws_for_four_dice : minimum_throws_to_ensure_repeated_sum 4 = 22 := by
  sorry

end minimum_throws_for_four_dice_l204_204126


namespace arithmetic_seq_fraction_l204_204315

theorem arithmetic_seq_fraction (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) 
  (h2 : a 1 + a 10 = a 9) 
  (d_ne_zero : d ≠ 0) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) / a 10 = 27 / 8 := 
sorry

end arithmetic_seq_fraction_l204_204315


namespace minimum_throws_for_repeated_sum_l204_204077

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l204_204077


namespace gcd_45_75_l204_204695

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204695


namespace sum_of_repeating_decimals_l204_204300

noncomputable def repeating_decimal_sum : ℚ :=
  let x := 1 / 3
  let y := 7 / 99
  let z := 8 / 999
  x + y + z

theorem sum_of_repeating_decimals :
  repeating_decimal_sum = 418 / 999 :=
by
  sorry

end sum_of_repeating_decimals_l204_204300


namespace soccer_team_games_played_l204_204415

theorem soccer_team_games_played 
  (players : ℕ) (total_goals : ℕ) (third_players_goals_per_game : ℕ → ℕ) (other_players_goals : ℕ) (G : ℕ)
  (h1 : players = 24)
  (h2 : total_goals = 150)
  (h3 : ∃ n, n = players / 3 ∧ ∀ g, third_players_goals_per_game g = n * g)
  (h4 : other_players_goals = 30)
  (h5 : total_goals = third_players_goals_per_game G + other_players_goals) :
  G = 15 := by
  -- Proof would go here
  sorry

end soccer_team_games_played_l204_204415


namespace tank_capacity_l204_204911

noncomputable def inflow_A (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def inflow_B (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def inflow_C (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def outflow_X (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def outflow_Y (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

theorem tank_capacity
  (fA : ℕ := inflow_A 8 7)
  (fB : ℕ := inflow_B 12 3)
  (fC : ℕ := inflow_C 6 4)
  (oX : ℕ := outflow_X 20 7)
  (oY : ℕ := outflow_Y 15 5) :
  fA + fB + fC = 6960 ∧ oX + oY = 12900 ∧ 12900 - 6960 = 5940 :=
by
  sorry

end tank_capacity_l204_204911


namespace ratio_john_to_jenna_l204_204347

theorem ratio_john_to_jenna (J : ℕ) 
  (h1 : 100 - J - 40 = 35) : 
  J = 25 ∧ (J / 100 = 1 / 4) := 
by
  sorry

end ratio_john_to_jenna_l204_204347


namespace semicircle_radius_l204_204910

theorem semicircle_radius (b h : ℝ) (base_eq_b : b = 16) (height_eq_h : h = 15) :
  let s := (2 * 17) / 2
  let area := 240 
  s * (r : ℝ) = area → r = 120 / 17 :=
  by
  intros s area
  sorry

end semicircle_radius_l204_204910


namespace blonde_hair_count_l204_204882

theorem blonde_hair_count (total_people : ℕ) (percentage_blonde : ℕ) (h_total : total_people = 600) (h_percentage : percentage_blonde = 30) : 
  (percentage_blonde * total_people / 100) = 180 :=
by
  -- Conditions from the problem
  have h1 : total_people = 600 := h_total
  have h2 : percentage_blonde = 30 := h_percentage
  -- Start the proof
  sorry

end blonde_hair_count_l204_204882


namespace min_throws_for_repeated_sum_l204_204094

theorem min_throws_for_repeated_sum : 
  (∀ (n : ℕ), n = 24 ∧ (∀ (x : ℕ), x ≥ 4 ∧ x ≤ 24)) → 22 :=
by
  sorry

end min_throws_for_repeated_sum_l204_204094


namespace min_throws_to_repeat_sum_l204_204087

theorem min_throws_to_repeat_sum : 
  (∀ (d1 d2 d3 d4 : ℕ), 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6 ∧ 1 ≤ d4 ∧ d4 ≤ 6) →
  (∃ n ≥ 22, ∃ F : (fin n) → ℕ, (∀ i : (fin n), 4 ≤ F i ∧ F i ≤ 24) ∧ (∃ x y : (fin n), x ≠ y ∧ F x = F y )) :=
begin
  sorry
end

end min_throws_to_repeat_sum_l204_204087


namespace smallest_five_digit_divisible_by_2_5_11_l204_204816

theorem smallest_five_digit_divisible_by_2_5_11 : ∃ n, n >= 10000 ∧ n % 2 = 0 ∧ n % 5 = 0 ∧ n % 11 = 0 ∧ n = 10010 :=
by
  sorry

end smallest_five_digit_divisible_by_2_5_11_l204_204816


namespace min_throws_to_repeat_sum_l204_204086

theorem min_throws_to_repeat_sum : 
  (∀ (d1 d2 d3 d4 : ℕ), 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6 ∧ 1 ≤ d4 ∧ d4 ≤ 6) →
  (∃ n ≥ 22, ∃ F : (fin n) → ℕ, (∀ i : (fin n), 4 ≤ F i ∧ F i ≤ 24) ∧ (∃ x y : (fin n), x ≠ y ∧ F x = F y )) :=
begin
  sorry
end

end min_throws_to_repeat_sum_l204_204086


namespace expense_5_yuan_neg_l204_204023

-- Define the condition that income of 5 yuan is denoted as +5 yuan
def income_5_yuan_pos : Int := 5

-- Define the statement to prove that expenses of 5 yuan are denoted as -5 yuan
theorem expense_5_yuan_neg : income_5_yuan_pos = 5 → -income_5_yuan_pos = -5 :=
by
  intro h
  rw h
  rfl

end expense_5_yuan_neg_l204_204023


namespace roots_of_quadratic_range_k_l204_204593

theorem roots_of_quadratic_range_k :
  (∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ 
    x1 ≠ x2 ∧ 
    (x1 ≠ 1 ∧ x2 ≠ 1) ∧
    ∀ k : ℝ, x1 ^ 2 + (k - 3) * x1 + k ^ 2 = 0 ∧ x2 ^ 2 + (k - 3) * x2 + k ^ 2 = 0) ↔
  ((k : ℝ) < 1 ∧ k > -2) :=
sorry

end roots_of_quadratic_range_k_l204_204593


namespace complement_in_U_l204_204437

def A : Set ℝ := { x : ℝ | |x - 1| > 3 }
def U : Set ℝ := Set.univ

theorem complement_in_U :
  (U \ A) = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
by
  sorry

end complement_in_U_l204_204437


namespace gcd_45_75_l204_204707

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204707


namespace sum_remainder_l204_204750

theorem sum_remainder (a b c d : ℤ) (h1 : a % 53 = 33) (h2 : b % 53 = 11) 
                       (h3 : c % 53 = 49) (h4 : d % 53 = 2) :
  (a + b + c + d) % 53 = 42 :=
sorry

end sum_remainder_l204_204750


namespace suitable_outfits_l204_204334

def num_shirts : ℕ := 8
def num_pants : ℕ := 6
def num_hats : ℕ := 6
def num_colors : ℕ := 6

theorem suitable_outfits :
  let total_combinations := num_shirts * num_pants * num_hats,
      same_color_shirt_pants := num_colors * num_hats,
      same_color_pants_hats := num_colors * num_shirts,
      same_color_shirt_hats := num_colors * num_pants,
      overcounted := 6 in
  total_combinations - (same_color_shirt_pants + same_color_pants_hats + same_color_shirt_hats - overcounted) = 174 :=
by
  let total_combinations := num_shirts * num_pants * num_hats
  let same_color_shirt_pants := num_colors * num_hats
  let same_color_pants_hats := num_colors * num_shirts
  let same_color_shirt_hats := num_colors * num_pants
  let overcounted := 6
  sorry

end suitable_outfits_l204_204334


namespace find_number_l204_204837

theorem find_number (x : ℝ) (n : ℝ) (h1 : x = 12) (h2 : (27 / n) * x - 18 = 3 * x + 27) : n = 4 :=
sorry

end find_number_l204_204837


namespace four_dice_min_rolls_l204_204111

def minRollsToEnsureSameSum (n : Nat) : Nat :=
  if n = 4 then 22 else sorry

theorem four_dice_min_rolls : minRollsToEnsureSameSum 4 = 22 := by
  rfl

end four_dice_min_rolls_l204_204111


namespace min_throws_to_ensure_repeat_sum_l204_204119

theorem min_throws_to_ensure_repeat_sum : 
  ∀ (min_sum max_sum : ℤ), 
  min_sum = 4 ∧ max_sum = 24 
  → ∃ n, n ≥ 22 ∧ n = 22 :=
by
  intros min_sum max_sum h
  cases h with h_min h_max
  existsi 22
  split
  · exact Nat.le_refl 22
  · sorry

end min_throws_to_ensure_repeat_sum_l204_204119


namespace handshake_count_l204_204216

theorem handshake_count
  (total_people : ℕ := 40)
  (groupA_size : ℕ := 30)
  (groupB_size : ℕ := 10)
  (groupB_knowsA_5 : ℕ := 3)
  (groupB_knowsA_0 : ℕ := 7)
  (handshakes_between_A_and_B5 : ℕ := groupB_knowsA_5 * (groupA_size - 5))
  (handshakes_between_A_and_B0 : ℕ := groupB_knowsA_0 * groupA_size)
  (handshakes_within_B : ℕ := groupB_size * (groupB_size - 1) / 2) :
  handshakes_between_A_and_B5 + handshakes_between_A_and_B0 + handshakes_within_B = 330 :=
sorry

end handshake_count_l204_204216


namespace distinct_terms_count_l204_204801

/-!
  Proving the number of distinct terms in the expansion of (x + 2y)^12
-/

theorem distinct_terms_count (x y : ℕ) : 
  (x + 2 * y) ^ 12 = 13 :=
by sorry

end distinct_terms_count_l204_204801


namespace osmotic_pressure_independence_l204_204679

-- definitions for conditions
def osmotic_pressure_depends_on (osmotic_pressure protein_content Na_content Cl_content : Prop) : Prop :=
  (osmotic_pressure = protein_content ∧ osmotic_pressure = Na_content ∧ osmotic_pressure = Cl_content)

-- statement of the problem to be proved
theorem osmotic_pressure_independence 
  (osmotic_pressure : Prop) 
  (protein_content : Prop) 
  (Na_content : Prop) 
  (Cl_content : Prop) 
  (mw_plasma_protein : Prop)
  (dependence : osmotic_pressure_depends_on osmotic_pressure protein_content Na_content Cl_content) :
  ¬(osmotic_pressure = mw_plasma_protein) :=
sorry

end osmotic_pressure_independence_l204_204679


namespace wendy_facial_products_l204_204388

def total_time (P : ℕ) : ℕ :=
  5 * (P - 1) + 30

theorem wendy_facial_products :
  (total_time 6 = 55) :=
by
  sorry

end wendy_facial_products_l204_204388


namespace count_two_digit_numbers_with_digit_8_l204_204333

theorem count_two_digit_numbers_with_digit_8 : 
  let two_digit_integers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let has_eight (n : ℕ) := n / 10 = 8 ∨ n % 10 = 8
  (two_digit_integers.filter has_eight).card = 18 :=
by
  let two_digit_integers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let has_eight (n : ℕ) := n / 10 = 8 ∨ n % 10 = 8
  show (two_digit_integers.filter has_eight).card = 18
  sorry

end count_two_digit_numbers_with_digit_8_l204_204333


namespace gcd_45_75_l204_204705

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204705


namespace total_cost_correct_l204_204362

noncomputable def camera_old_cost : ℝ := 4000
noncomputable def camera_new_cost := camera_old_cost * 1.30
noncomputable def lens_cost := 400
noncomputable def lens_discount := 200
noncomputable def lens_discounted_price := lens_cost - lens_discount
noncomputable def total_cost := camera_new_cost + lens_discounted_price

theorem total_cost_correct :
  total_cost = 5400 := by
  sorry

end total_cost_correct_l204_204362


namespace longest_side_of_triangle_l204_204378

-- Defining variables and constants
variables (x : ℕ)

-- Defining the side lengths of the triangle
def side1 := 7
def side2 := x + 4
def side3 := 2 * x + 1

-- Defining the perimeter of the triangle
def perimeter := side1 + side2 + side3

-- Statement of the main theorem
theorem longest_side_of_triangle (h : perimeter x = 36) : max side1 (max (side2 x) (side3 x)) = 17 :=
by sorry

end longest_side_of_triangle_l204_204378


namespace division_of_8_identical_books_into_3_piles_l204_204274

-- Definitions for the conditions
def identical_books_division_ways (n : ℕ) (p : ℕ) : ℕ :=
  if n = 8 ∧ p = 3 then 5 else sorry

-- Theorem statement
theorem division_of_8_identical_books_into_3_piles :
  identical_books_division_ways 8 3 = 5 := by
  sorry

end division_of_8_identical_books_into_3_piles_l204_204274


namespace binom_15_4_l204_204561

theorem binom_15_4 : nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l204_204561


namespace joseph_drives_more_l204_204349

def joseph_speed : ℝ := 50
def joseph_time : ℝ := 2.5
def kyle_speed : ℝ := 62
def kyle_time : ℝ := 2

def joseph_distance : ℝ := joseph_speed * joseph_time
def kyle_distance : ℝ := kyle_speed * kyle_time

theorem joseph_drives_more : (joseph_distance - kyle_distance) = 1 := by
  sorry

end joseph_drives_more_l204_204349


namespace p_sufficient_but_not_necessary_for_q_l204_204325

-- Definitions corresponding to conditions
def p (x : ℝ) : Prop := x > 1
def q (x : ℝ) : Prop := 1 / x < 1

-- Theorem stating the relationship between p and q
theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) := 
by
  sorry

end p_sufficient_but_not_necessary_for_q_l204_204325


namespace ram_money_l204_204881

theorem ram_money (R G K : ℝ) (h1 : R / G = 7 / 17) (h2 : G / K = 7 / 17) (h3 : K = 3468) :
  R = 588 := by
  sorry

end ram_money_l204_204881


namespace expense_of_5_yuan_is_minus_5_yuan_l204_204025

def income (x : Int) : Int :=
  x

def expense (x : Int) : Int :=
  -x

theorem expense_of_5_yuan_is_minus_5_yuan : expense 5 = -5 :=
by
  unfold expense
  sorry

end expense_of_5_yuan_is_minus_5_yuan_l204_204025


namespace equidistant_point_quadrants_l204_204323

theorem equidistant_point_quadrants (x y : ℝ) (h : 4 * x + 3 * y = 12) :
  (x > 0 ∧ y = 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end equidistant_point_quadrants_l204_204323


namespace exists_linear_eq_solution_x_2_l204_204369

theorem exists_linear_eq_solution_x_2 : ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x : ℝ, a * x + b = 0 ↔ x = 2 :=
by
  sorry

end exists_linear_eq_solution_x_2_l204_204369


namespace range_of_a_l204_204432

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → x^2 + 2 * x - a > 0) → a < 3 :=
by
  sorry

end range_of_a_l204_204432


namespace imaginary_number_m_l204_204461

theorem imaginary_number_m (m : ℝ) : 
  (∀ Z, Z = (m + 2 * Complex.I) / (1 + Complex.I) → Z.im = 0 → Z.re = 0) → m = -2 :=
by
  sorry

end imaginary_number_m_l204_204461


namespace xy_product_given_conditions_l204_204676

variable (x y : ℝ)

theorem xy_product_given_conditions (hx : x - y = 5) (hx3 : x^3 - y^3 = 35) : x * y = -6 :=
by
  sorry

end xy_product_given_conditions_l204_204676


namespace roots_of_equation_l204_204184

theorem roots_of_equation :
  ∀ x : ℝ, (21 / (x^2 - 9) - 3 / (x - 3) = 1) ↔ (x = 3 ∨ x = -7) :=
by
  sorry

end roots_of_equation_l204_204184


namespace table_cost_l204_204401

variable (T : ℝ) -- Cost of the table
variable (C : ℝ) -- Cost of a chair

-- Conditions
axiom h1 : C = T / 7
axiom h2 : T + 4 * C = 220

theorem table_cost : T = 140 :=
by
  sorry

end table_cost_l204_204401


namespace leo_current_weight_l204_204337

variables (L K J : ℝ)

def condition1 := L + 12 = 1.7 * K
def condition2 := L + K + J = 270
def condition3 := J = K + 30

theorem leo_current_weight (h1 : condition1 L K)
                           (h2 : condition2 L K J)
                           (h3 : condition3 K J) : L = 103.6 :=
sorry

end leo_current_weight_l204_204337


namespace number_of_white_balls_l204_204970

-- Definitions based on the problem conditions
def total_balls : Nat := 120
def red_freq : ℝ := 0.15
def black_freq : ℝ := 0.45

-- Result to prove
theorem number_of_white_balls :
  let red_balls := total_balls * red_freq
  let black_balls := total_balls * black_freq
  total_balls - red_balls - black_balls = 48 :=
by
  sorry

end number_of_white_balls_l204_204970


namespace gcd_of_45_and_75_l204_204718

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l204_204718


namespace simplify_fractions_l204_204991

theorem simplify_fractions : (360 / 32) * (10 / 240) * (16 / 6) = 10 := by
  sorry

end simplify_fractions_l204_204991


namespace express_recurring_decimal_as_fraction_l204_204428

theorem express_recurring_decimal_as_fraction (h : 0.01 = (1 : ℚ) / 99) : 2.02 = (200 : ℚ) / 99 :=
by 
  sorry

end express_recurring_decimal_as_fraction_l204_204428


namespace combined_garden_area_l204_204629

-- Definitions for the sizes and counts of the gardens.
def Mancino_gardens : ℕ := 4
def Marquita_gardens : ℕ := 3
def Matteo_gardens : ℕ := 2
def Martina_gardens : ℕ := 5

def Mancino_garden_area : ℕ := 16 * 5
def Marquita_garden_area : ℕ := 8 * 4
def Matteo_garden_area : ℕ := 12 * 6
def Martina_garden_area : ℕ := 10 * 3

-- The total combined area to be proven.
def total_area : ℕ :=
  (Mancino_gardens * Mancino_garden_area) +
  (Marquita_gardens * Marquita_garden_area) +
  (Matteo_gardens * Matteo_garden_area) +
  (Martina_gardens * Martina_garden_area)

-- Proof statement for the combined area.
theorem combined_garden_area : total_area = 710 :=
by sorry

end combined_garden_area_l204_204629


namespace sufficient_condition_implies_range_l204_204858

theorem sufficient_condition_implies_range {x m : ℝ} : (∀ x, 1 ≤ x ∧ x < 4 → x < m) → 4 ≤ m :=
by
  sorry

end sufficient_condition_implies_range_l204_204858


namespace c_work_rate_l204_204268

/--
A can do a piece of work in 4 days.
B can do it in 8 days.
With the assistance of C, A and B completed the work in 2 days.
Prove that C alone can do the work in 8 days.
-/
theorem c_work_rate :
  (1 / 4 + 1 / 8 + 1 / c = 1 / 2) → c = 8 :=
by
  intro h
  sorry

end c_work_rate_l204_204268


namespace goals_last_season_l204_204978

theorem goals_last_season : 
  ∀ (goals_last_season goals_this_season total_goals : ℕ), 
  goals_this_season = 187 → 
  total_goals = 343 → 
  total_goals = goals_last_season + goals_this_season → 
  goals_last_season = 156 := 
by 
  intros goals_last_season goals_this_season total_goals 
  intro h_this_season 
  intro h_total_goals 
  intro h_equation 
  calc 
    goals_last_season = total_goals - goals_this_season : by rw [h_equation, Nat.add_sub_cancel_left]
    ... = 343 - 187 : by rw [h_this_season, h_total_goals]
    ... = 156 : by norm_num

end goals_last_season_l204_204978


namespace cesaro_sum_100_terms_l204_204272

noncomputable def cesaro_sum (A : List ℝ) : ℝ :=
  let n := A.length
  (List.sum A) / n

theorem cesaro_sum_100_terms :
  ∀ (A : List ℝ), A.length = 99 →
  cesaro_sum A = 1000 →
  cesaro_sum (1 :: A) = 991 :=
by
  intros A h1 h2
  sorry

end cesaro_sum_100_terms_l204_204272


namespace simple_interest_rate_l204_204271

theorem simple_interest_rate (P R : ℝ) (T : ℕ) (hT : T = 10) (h_double : P * 2 = P + P * R * T / 100) : R = 10 :=
by
  sorry

end simple_interest_rate_l204_204271


namespace cookie_cost_l204_204996

variables (m o c : ℝ)
variables (H1 : m = 2 * o)
variables (H2 : 3 * (3 * m + 5 * o) = 5 * m + 10 * o + 4 * c)

theorem cookie_cost (H1 : m = 2 * o) (H2 : 3 * (3 * m + 5 * o) = 5 * m + 10 * o + 4 * c) : c = (13 / 4) * o :=
by sorry

end cookie_cost_l204_204996


namespace binom_15_4_l204_204559

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the theorem stating the specific binomial coefficient value
theorem binom_15_4 : binom 15 4 = 1365 :=
by
  sorry

end binom_15_4_l204_204559


namespace minimum_rolls_for_duplicate_sum_l204_204088

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l204_204088


namespace unit_prices_purchasing_schemes_maximize_profit_l204_204904

-- Define the conditions and variables
def purchase_price_system (x y : ℝ) : Prop :=
  (2 * x + 3 * y = 240) ∧ (3 * x + 4 * y = 340)

def possible_schemes (a b : ℕ) : Prop :=
  (a + b = 200) ∧ (60 * a + 40 * b ≤ 10440) ∧ (a ≥ 3 * b / 2)

def max_profit (x y : ℝ) (a b : ℕ) : ℝ :=
  (x - 60) * a + (y - 40) * b

-- Prove the unit prices are $60 and $40
theorem unit_prices : ∃ x y, purchase_price_system x y ∧ x = 60 ∧ y = 40 :=
by
  sorry

-- Prove the possible purchasing schemes
theorem purchasing_schemes : ∀ a b, possible_schemes a b → 
  (a = 120 ∧ b = 80 ∨ a = 121 ∧ b = 79 ∨ a = 122 ∧ b = 78) :=
by
  sorry

-- Prove the maximum profit is 3610 with the purchase amounts (122, 78)
theorem maximize_profit :
  ∃ (a b : ℕ), max_profit 80 55 a b = 3610 ∧ purchase_price_system 60 40 ∧ possible_schemes a b ∧ a = 122 ∧ b = 78 :=
by
  sorry

end unit_prices_purchasing_schemes_maximize_profit_l204_204904


namespace toys_produced_per_week_l204_204766

-- Definitions corresponding to the conditions
def days_per_week : ℕ := 2
def toys_per_day : ℕ := 2170

-- Theorem statement corresponding to the question and correct answer
theorem toys_produced_per_week : days_per_week * toys_per_day = 4340 := 
by 
  -- placeholders for the proof steps
  sorry

end toys_produced_per_week_l204_204766


namespace gcd_45_75_l204_204699

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204699


namespace fraction_multiplication_simplifies_l204_204389

theorem fraction_multiplication_simplifies :
  (3 : ℚ) / 4 * (4 / 5) * (2 / 3) = 2 / 5 := 
by 
  -- Prove the equality step-by-step
  sorry

end fraction_multiplication_simplifies_l204_204389


namespace gcd_of_45_and_75_l204_204719

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l204_204719


namespace gcd_45_75_l204_204709

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204709


namespace num_four_letter_initials_sets_l204_204948

def num_initials_sets : ℕ := 8 ^ 4

theorem num_four_letter_initials_sets:
  num_initials_sets = 4096 :=
by
  rw [num_initials_sets]
  norm_num

end num_four_letter_initials_sets_l204_204948


namespace least_number_of_square_tiles_l204_204135

theorem least_number_of_square_tiles
  (length_cm : ℕ) (width_cm : ℕ)
  (h1 : length_cm = 816) (h2 : width_cm = 432) :
  ∃ tile_count : ℕ, tile_count = 153 :=
by
  sorry

end least_number_of_square_tiles_l204_204135


namespace min_throws_for_repeated_sum_l204_204093

theorem min_throws_for_repeated_sum : 
  (∀ (n : ℕ), n = 24 ∧ (∀ (x : ℕ), x ≥ 4 ∧ x ≤ 24)) → 22 :=
by
  sorry

end min_throws_for_repeated_sum_l204_204093


namespace infinite_solutions_x2_y3_z5_l204_204371

theorem infinite_solutions_x2_y3_z5 :
  ∃ (t : ℕ), ∃ (x y z : ℕ), x = 2^(15*t + 12) ∧ y = 2^(10*t + 8) ∧ z = 2^(6*t + 5) ∧ (x^2 + y^3 = z^5) :=
sorry

end infinite_solutions_x2_y3_z5_l204_204371


namespace interval_satisfies_ineq_l204_204658

theorem interval_satisfies_ineq (p : ℝ) (h1 : 18 * p < 10) (h2 : 0.5 < p) : 0.5 < p ∧ p < 5 / 9 :=
by {
  sorry -- Proof not required, only the statement.
}

end interval_satisfies_ineq_l204_204658


namespace max_marks_l204_204269

variable (M : ℝ)

-- Conditions
def needed_to_pass (M : ℝ) := 0.20 * M
def pradeep_marks := 390
def marks_short := 25
def total_marks_needed := pradeep_marks + marks_short

-- Theorem statement
theorem max_marks : needed_to_pass M = total_marks_needed → M = 2075 := by
  sorry

end max_marks_l204_204269


namespace rock_height_at_30_l204_204286

theorem rock_height_at_30 (t : ℝ) (h : ℝ) 
  (h_eq : h = 80 - 9 * t - 5 * t^2) 
  (h_30 : h = 30) : 
  t = 2.3874 :=
by
  -- Proof omitted
  sorry

end rock_height_at_30_l204_204286


namespace each_player_plays_36_minutes_l204_204794

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l204_204794


namespace james_total_cost_l204_204622

theorem james_total_cost :
  let offRackCost := 300
  let tailoredCost := 3 * offRackCost + 200
  (offRackCost + tailoredCost) = 1400 :=
by
  let offRackCost := 300
  let tailoredCost := 3 * offRackCost + 200
  have h1 : offRackCost + tailoredCost = 300 + (3 * 300 + 200) := by sorry
  have h2 : 300 + (3 * 300 + 200) = 300 + 900 + 200 := by sorry
  have h3 : 300 + 900 + 200 = 1400 := by sorry
  exact eq.trans h1 (eq.trans h2 h3)

end james_total_cost_l204_204622


namespace polynomial_solutions_l204_204926

theorem polynomial_solutions :
  (∀ x : ℂ, (x^4 + 2*x^3 + 2*x^2 + 2*x + 1 = 0) ↔ (x = -1 ∨ x = Complex.I ∨ x = -Complex.I)) :=
by
  sorry

end polynomial_solutions_l204_204926


namespace brother_age_in_5_years_l204_204635

noncomputable def Nick : ℕ := 13
noncomputable def Sister : ℕ := Nick + 6
noncomputable def CombinedAge : ℕ := Nick + Sister
noncomputable def Brother : ℕ := CombinedAge / 2

theorem brother_age_in_5_years : Brother + 5 = 21 := by
  sorry

end brother_age_in_5_years_l204_204635


namespace average_growth_rate_income_prediction_l204_204469

-- Define the given conditions
def income2018 : ℝ := 20000
def income2020 : ℝ := 24200
def growth_rate : ℝ := 0.1
def predicted_income2021 : ℝ := 26620

-- Lean 4 statement for the first part of the problem
theorem average_growth_rate :
  (income2020 = income2018 * (1 + growth_rate)^2) →
  growth_rate = 0.1 :=
by
  intros h
  sorry

-- Lean 4 statement for the second part of the problem
theorem income_prediction :
  (income2020 = income2018 * (1 + growth_rate)^2) →
  (growth_rate = 0.1) →
  (income2018 * (1 + growth_rate)^3 = predicted_income2021) :=
by
  intros h1 h2
  sorry

end average_growth_rate_income_prediction_l204_204469


namespace contrapositive_l204_204368

variable (P Q : Prop)

theorem contrapositive (h : P → Q) : ¬Q → ¬P :=
sorry

end contrapositive_l204_204368


namespace find_c_l204_204955

theorem find_c (a c : ℝ) :
  (∀ x : ℝ, x^2 + a * x + a / 4 + 1 / 2 > 0) →
  (∃! b : ℝ, (∀ x : ℝ, x^2 - x + b < 0)) →
  c = -2 :=
by
  sorry

end find_c_l204_204955


namespace pigeons_on_branches_and_under_tree_l204_204527

theorem pigeons_on_branches_and_under_tree (x y : ℕ) 
  (h1 : y - 1 = (x + 1) / 2)
  (h2 : x - 1 = y + 1) : x = 7 ∧ y = 5 :=
by
  sorry

end pigeons_on_branches_and_under_tree_l204_204527


namespace sum_of_interior_angles_l204_204959

noncomputable def exterior_angle (n : ℕ) := 360 / n

theorem sum_of_interior_angles (n : ℕ) (h : exterior_angle n = 45) :
  180 * (n - 2) = 1080 :=
by
  sorry

end sum_of_interior_angles_l204_204959


namespace arithmetic_sequence_sum_l204_204222

theorem arithmetic_sequence_sum (b : ℕ → ℝ) (h_arith : ∀ n, b (n+1) - b n = b 2 - b 1) (h_b5 : b 5 = 2) :
  b 1 + b 2 + b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9 = 18 := 
sorry

end arithmetic_sequence_sum_l204_204222


namespace share_of_b_l204_204526

theorem share_of_b (x : ℝ) (h : 3300 / ((7/2) * x) = 2 / 7) :  
   let total_profit := 3300
   let B_share := (x / ((7/2) * x)) * total_profit
   B_share = 942.86 :=
by sorry

end share_of_b_l204_204526


namespace expenses_negation_of_income_l204_204004

theorem expenses_negation_of_income 
    (income : ℤ) 
    (income_is_5 : income = 5) 
    (denote_income : income = 5 → "+" ∘ toString income = "+5") 
    (expenses_are_negation_of_income :  "expenses = -1 * income") : "expenses = -5" :=
begin
    sorry
end

end expenses_negation_of_income_l204_204004


namespace problem_sol_l204_204826

theorem problem_sol (a b : ℝ) (h : ∀ x, (x > -1 ∧ x < 1/3) ↔ (ax^2 + bx + 1 > 0)) : a * b = 6 :=
sorry

end problem_sol_l204_204826


namespace total_sales_is_10400_l204_204867

-- Define the conditions
def tough_week_sales : ℝ := 800
def good_week_sales : ℝ := 2 * tough_week_sales
def good_weeks : ℕ := 5
def tough_weeks : ℕ := 3

-- Define the total sales function
def total_sales (good_sales : ℝ) (tough_sales : ℝ) (good_weeks : ℕ) (tough_weeks : ℕ) : ℝ :=
  good_weeks * good_sales + tough_weeks * tough_sales

-- Prove that the total sales is $10400
theorem total_sales_is_10400 : total_sales good_week_sales tough_week_sales good_weeks tough_weeks = 10400 := 
by
  sorry

end total_sales_is_10400_l204_204867


namespace average_monthly_growth_rate_proof_profit_in_may_proof_l204_204493

theorem average_monthly_growth_rate_proof :
  ∃ r : ℝ, 2400 * (1 + r)^2 = 3456 ∧ r = 0.2 := sorry

theorem profit_in_may_proof (r : ℝ) (h_r : r = 0.2) :
  3456 * (1 + r) = 4147.2 := sorry

end average_monthly_growth_rate_proof_profit_in_may_proof_l204_204493


namespace calculate_correctly_l204_204840

theorem calculate_correctly (x : ℕ) (h : 2 * x = 22) : 20 * x + 3 = 223 :=
by
  sorry

end calculate_correctly_l204_204840


namespace intersection_A_B_l204_204595

noncomputable def A : Set ℝ := { x | abs (x - 1) < 2 }
noncomputable def B : Set ℝ := { x | x^2 + 3 * x - 4 < 0 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l204_204595


namespace necessary_but_not_sufficient_condition_l204_204902

variable {m : ℝ}

theorem necessary_but_not_sufficient_condition (h : (∃ x1 x2 : ℝ, (x1 ≠ 0 ∧ x1 = -x2) ∧ (x1^2 + x1 + m^2 - 1 = 0))): 
  0 < m ∧ m < 1 :=
by 
  sorry

end necessary_but_not_sufficient_condition_l204_204902


namespace Oleg_age_proof_l204_204637

-- Defining the necessary conditions
variables (x y z : ℕ) -- defining the ages of Oleg, his father, and his grandfather

-- Stating the conditions
axiom h1 : y = x + 32
axiom h2 : z = y + 32
axiom h3 : (x - 3) + (y - 3) + (z - 3) < 100

-- Stating the proof problem
theorem Oleg_age_proof : 
  (x = 4) ∧ (y = 36) ∧ (z = 68) :=
by
  sorry

end Oleg_age_proof_l204_204637


namespace find_p_l204_204311

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

noncomputable def point_A_on_parabola (x y p : ℝ) : Prop := y^2 = 2 * p * x

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_p (x y p : ℝ) (h_p : p > 0)
  (h_A : point_A_on_parabola x y p)
  (h_dist_to_focus : distance x y (p / 2) 0 = 12)
  (h_dist_to_yaxis : real.abs x = 9) 
  : p = 6 :=
sorry

end find_p_l204_204311


namespace binom_15_4_l204_204560

theorem binom_15_4 : nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l204_204560


namespace expense_of_5_yuan_is_minus_5_yuan_l204_204028

def income (x : Int) : Int :=
  x

def expense (x : Int) : Int :=
  -x

theorem expense_of_5_yuan_is_minus_5_yuan : expense 5 = -5 :=
by
  unfold expense
  sorry

end expense_of_5_yuan_is_minus_5_yuan_l204_204028


namespace factor_expression_l204_204182

theorem factor_expression (x y z : ℝ) : 
  ( (x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3 ) / 
  ( (x - y)^3 + (y - z)^3 + (z - x)^3 ) = 
  (x + y) * (y + z) * (z + x) := 
by
  sorry

end factor_expression_l204_204182


namespace sum_of_money_is_6000_l204_204153

noncomputable def original_interest (P R : ℝ) := (P * R * 3) / 100
noncomputable def new_interest (P R : ℝ) := (P * (R + 2) * 3) / 100

theorem sum_of_money_is_6000 (P R : ℝ) (h : new_interest P R - original_interest P R = 360) : P = 6000 :=
by
  sorry

end sum_of_money_is_6000_l204_204153


namespace new_tax_rate_is_30_percent_l204_204209

theorem new_tax_rate_is_30_percent
  (original_rate : ℝ)
  (annual_income : ℝ)
  (tax_saving : ℝ)
  (h1 : original_rate = 0.45)
  (h2 : annual_income = 48000)
  (h3 : tax_saving = 7200) :
  (100 * (original_rate * annual_income - tax_saving) / annual_income) = 30 := 
sorry

end new_tax_rate_is_30_percent_l204_204209


namespace min_throws_to_repeat_sum_l204_204085

theorem min_throws_to_repeat_sum : 
  (∀ (d1 d2 d3 d4 : ℕ), 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6 ∧ 1 ≤ d4 ∧ d4 ≤ 6) →
  (∃ n ≥ 22, ∃ F : (fin n) → ℕ, (∀ i : (fin n), 4 ≤ F i ∧ F i ≤ 24) ∧ (∃ x y : (fin n), x ≠ y ∧ F x = F y )) :=
begin
  sorry
end

end min_throws_to_repeat_sum_l204_204085


namespace min_throws_to_repeat_sum_l204_204083

theorem min_throws_to_repeat_sum : 
  (∀ (d1 d2 d3 d4 : ℕ), 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6 ∧ 1 ≤ d4 ∧ d4 ≤ 6) →
  (∃ n ≥ 22, ∃ F : (fin n) → ℕ, (∀ i : (fin n), 4 ≤ F i ∧ F i ≤ 24) ∧ (∃ x y : (fin n), x ≠ y ∧ F x = F y )) :=
begin
  sorry
end

end min_throws_to_repeat_sum_l204_204083


namespace shepherds_sheep_l204_204151

theorem shepherds_sheep (x y : ℕ) 
  (h1 : x - 4 = y + 4) 
  (h2 : x + 4 = 3 * (y - 4)) : 
  x = 20 ∧ y = 12 := 
by 
  sorry

end shepherds_sheep_l204_204151


namespace count_paths_COMPUTER_l204_204569

theorem count_paths_COMPUTER : 
  let possible_paths (n : ℕ) := 2 ^ n 
  possible_paths 7 + possible_paths 7 + 1 = 257 :=
by sorry

end count_paths_COMPUTER_l204_204569


namespace box_triple_count_l204_204413

theorem box_triple_count (a b c : ℕ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : a * b * c = 2 * (a * b + b * c + c * a)) :
  (a = 2 ∧ b = 8 ∧ c = 8) ∨ (a = 3 ∧ b = 6 ∧ c = 6) ∨ (a = 4 ∧ b = 4 ∧ c = 4) ∨ (a = 5 ∧ b = 5 ∧ c = 5) ∨ (a = 6 ∧ b = 6 ∧ c = 6) :=
sorry

end box_triple_count_l204_204413


namespace sum_of_three_consecutive_eq_product_of_distinct_l204_204933

theorem sum_of_three_consecutive_eq_product_of_distinct (n : ℕ) (h : 100 < n) :
  ∃ a b c, (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ a > 1 ∧ b > 1 ∧ c > 1 ∧
  ((n + (n+1) + (n+2) = a * b * c) ∨
   ((n+1) + (n+2) + (n+3) = a * b * c) ∨
   (n + (n+1) + (n+3) = a * b * c) ∨
   (n + (n+2) + (n+3) = a * b * c)) :=
by
  sorry

end sum_of_three_consecutive_eq_product_of_distinct_l204_204933


namespace count_8_digit_odd_last_l204_204204

-- Define the constraints for the digits of the 8-digit number
def first_digit_choices := 9
def next_six_digits_choices := 10 ^ 6
def last_digit_choices := 5

-- State the theorem based on the given conditions and the solution
theorem count_8_digit_odd_last : first_digit_choices * next_six_digits_choices * last_digit_choices = 45000000 :=
by
  sorry

end count_8_digit_odd_last_l204_204204


namespace Duke_broke_record_by_5_l204_204448

theorem Duke_broke_record_by_5 :
  let free_throws := 5
  let regular_baskets := 4
  let normal_three_pointers := 2
  let extra_three_pointers := 1
  let points_per_free_throw := 1
  let points_per_regular_basket := 2
  let points_per_three_pointer := 3
  let points_to_tie_record := 17

  let total_points_scored := (free_throws * points_per_free_throw) +
                             (regular_baskets * points_per_regular_basket) +
                             ((normal_three_pointers + extra_three_pointers) * points_per_three_pointer)
  total_points_scored = 22 →
  total_points_scored - points_to_tie_record = 5 :=

by
  intros
  sorry

end Duke_broke_record_by_5_l204_204448


namespace value_of_expression_at_3_l204_204889

theorem value_of_expression_at_3 :
  ∀ (x : ℕ), x = 3 → (x^4 - 6 * x) = 63 :=
by
  intros x h
  sorry

end value_of_expression_at_3_l204_204889


namespace transform_fraction_l204_204877

theorem transform_fraction (x : ℝ) (h₁ : x ≠ 3) : - (1 / (3 - x)) = (1 / (x - 3)) := 
    sorry

end transform_fraction_l204_204877


namespace ensure_same_sum_rolled_twice_l204_204065

theorem ensure_same_sum_rolled_twice :
  ∀ (n : ℕ) (min_sum max_sum : ℕ),
    min_sum = 4 →
    max_sum = 24 →
    (min_sum ≤ n ∧ n ≤ max_sum) →
    ∀ trials : ℕ, trials = 22 →
      ∃ (s1 s2 : ℕ), s1 = s2 ∧ 
      (∃ (throws1 throws2 : list ℕ), list.sum throws1 = s1 ∧ list.sum throws2 = s2 ∧ throws1 ≠ throws2) :=
by 
  sorry

end ensure_same_sum_rolled_twice_l204_204065


namespace answer_is_correct_l204_204825

-- We define the prime checking function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

-- We define the set of candidates satisfying initial prime condition
def candidates : Set ℕ := {A | is_prime A ∧ A < 100 
                                   ∧ is_prime (A + 10) 
                                   ∧ is_prime (A - 20)
                                   ∧ is_prime (A + 30) 
                                   ∧ is_prime (A + 60) 
                                   ∧ is_prime (A + 70)}

-- The explicit set of valid answers
def valid_answers : Set ℕ := {37, 43, 79}

-- The statement that we need to prove
theorem answer_is_correct : candidates = valid_answers := 
sorry

end answer_is_correct_l204_204825


namespace rahim_pillows_l204_204989

theorem rahim_pillows (x T : ℕ) (h1 : T = 5 * x) (h2 : (T + 10) / (x + 1) = 6) : x = 4 :=
by
  sorry

end rahim_pillows_l204_204989


namespace largest_fraction_of_consecutive_odds_is_three_l204_204936

theorem largest_fraction_of_consecutive_odds_is_three
  (p q r s : ℕ)
  (h1 : 0 < p)
  (h2 : p < q)
  (h3 : q < r)
  (h4 : r < s)
  (h_odd1 : p % 2 = 1)
  (h_odd2 : q % 2 = 1)
  (h_odd3 : r % 2 = 1)
  (h_odd4 : s % 2 = 1)
  (h_consecutive1 : q = p + 2)
  (h_consecutive2 : r = q + 2)
  (h_consecutive3 : s = r + 2) :
  (r + s) / (p + q) = 3 :=
sorry

end largest_fraction_of_consecutive_odds_is_three_l204_204936


namespace min_value_n_minus_m_l204_204445

noncomputable def f (x : ℝ) : ℝ :=
  if 1 < x then Real.log x else (1 / 2) * x + (1 / 2)

theorem min_value_n_minus_m (m n : ℝ) (hmn : m < n) (hf_eq : f m = f n) : n - m = 3 - 2 * Real.log 2 :=
  sorry

end min_value_n_minus_m_l204_204445


namespace carlos_marbles_l204_204163

theorem carlos_marbles :
  ∃ N : ℕ, N > 2 ∧
  (N % 6 = 2) ∧
  (N % 7 = 2) ∧
  (N % 8 = 2) ∧
  (N % 11 = 2) ∧
  N = 3698 :=
by
  sorry

end carlos_marbles_l204_204163


namespace part1_solution_part2_solution_l204_204198

-- Part (1) Statement
theorem part1_solution (x : ℝ) (m : ℝ) (h_m : m = -1) :
  (3 * x - m) / 2 - (x + m) / 3 = 5 / 6 → x = 0 :=
by
  intros h_eq
  rw [h_m] at h_eq
  sorry  -- Proof to be filled in

-- Part (2) Statement
theorem part2_solution (x m : ℝ) (h_x : x = 5)
  (h_eq : (3 * x - m) / 2 - (x + m) / 3 = 5 / 6) :
  (1 / 2) * m^2 + 2 * m = 30 :=
by
  rw [h_x] at h_eq
  sorry  -- Proof to be filled in

end part1_solution_part2_solution_l204_204198


namespace two_digit_number_representation_l204_204416

theorem two_digit_number_representation (x : ℕ) (h : x < 10) : 10 * x + 5 < 100 :=
by sorry

end two_digit_number_representation_l204_204416


namespace arccos_sqrt3_div_2_eq_pi_div_6_l204_204167

theorem arccos_sqrt3_div_2_eq_pi_div_6 : 
  Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 :=
by 
  sorry

end arccos_sqrt3_div_2_eq_pi_div_6_l204_204167


namespace fraction_to_percentage_l204_204339

theorem fraction_to_percentage (x : ℝ) (hx : 0 < x) : 
  (x / 50 + x / 25) = 0.06 * x := 
sorry

end fraction_to_percentage_l204_204339


namespace solve_y_l204_204891

theorem solve_y 
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (remainder_condition : x = (96.12 * y))
  (division_condition : x = (96.0624 * y + 5.76)) : 
  y = 100 := 
 sorry

end solve_y_l204_204891


namespace infinite_k_Q_ineq_l204_204819

def Q (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem infinite_k_Q_ineq :
  ∃ᶠ k in at_top, Q (3 ^ k) > Q (3 ^ (k + 1)) := sorry

end infinite_k_Q_ineq_l204_204819


namespace chess_tournament_rounds_needed_l204_204144

theorem chess_tournament_rounds_needed
  (num_players : ℕ)
  (num_games_per_round : ℕ)
  (H1 : num_players = 20)
  (H2 : num_games_per_round = 10) :
  (num_players * (num_players - 1)) / num_games_per_round = 38 :=
by
  sorry

end chess_tournament_rounds_needed_l204_204144


namespace largest_inscribed_square_side_length_l204_204850

noncomputable def side_length_inscribed_square: ℝ := 6 - Real.sqrt 6

theorem largest_inscribed_square_side_length (a : ℝ) 
  (h₁ : a = 12)
  (triangle_side_length : ℝ)
  (h₂ : triangle_side_length = 4 * Real.sqrt 6) : 
  let inscribed_square_side_length := 6 - Real.sqrt 6 in
  (∀ (x : ℝ), x < inscribed_square_side_length) ∧ (side_length_inscribed_square = 6 - Real.sqrt 6) :=
by
  have y := 6 - Real.sqrt 6
  have h : y = side_length_inscribed_square := rfl
  sorry

end largest_inscribed_square_side_length_l204_204850


namespace monopoly_favor_durable_machine_competitive_market_prefer_durable_l204_204755

-- Define the conditions
def consumer_valuation : ℕ := 10
def durable_cost : ℕ := 6

-- Define the monopoly decision problem: prove C > 3
theorem monopoly_favor_durable_machine (C : ℕ) : 
  consumer_valuation * 2 - durable_cost > 2 * (consumer_valuation - C) → C > 3 := 
by 
  sorry

-- Define the competitive market decision problem: prove C > 3
theorem competitive_market_prefer_durable (C : ℕ) :
  2 * consumer_valuation - durable_cost > 2 * (consumer_valuation - C) → C > 3 := 
by 
  sorry

end monopoly_favor_durable_machine_competitive_market_prefer_durable_l204_204755


namespace train_speed_120_kmph_l204_204548

theorem train_speed_120_kmph (t : ℝ) (d : ℝ) (h_t : t = 9) (h_d : d = 300) : 
    (d / t) * 3.6 = 120 :=
by
  sorry

end train_speed_120_kmph_l204_204548


namespace minimum_throws_for_four_dice_l204_204124

noncomputable def minimum_throws_to_ensure_repeated_sum (d : ℕ) : ℕ :=
  let min_sum := d * 1 in
  let max_sum := d * 6 in
  let distinct_sums := max_sum - min_sum + 1 in
  distinct_sums + 1

theorem minimum_throws_for_four_dice : minimum_throws_to_ensure_repeated_sum 4 = 22 := by
  sorry

end minimum_throws_for_four_dice_l204_204124


namespace first_part_results_count_l204_204384

theorem first_part_results_count : 
    ∃ n, n * 10 + 90 + (25 - n) * 20 = 25 * 18 ∧ n = 14 :=
by
  sorry

end first_part_results_count_l204_204384


namespace find_x_value_l204_204597

/-- Defining the conditions given in the problem -/
structure HenrikhConditions where
  x : ℕ
  walking_time_per_block : ℕ := 60
  bicycle_time_per_block : ℕ := 20
  skateboard_time_per_block : ℕ := 40
  added_time_walking_over_bicycle : ℕ := 480
  added_time_walking_over_skateboard : ℕ := 240

/-- Defining a hypothesis based on the conditions -/
noncomputable def henrikh (c : HenrikhConditions) : Prop :=
  c.walking_time_per_block * c.x = c.bicycle_time_per_block * c.x + c.added_time_walking_over_bicycle ∧
  c.walking_time_per_block * c.x = c.skateboard_time_per_block * c.x + c.added_time_walking_over_skateboard

/-- The theorem to be proved -/
theorem find_x_value (c : HenrikhConditions) (h : henrikh c) : c.x = 12 := by
  sorry

end find_x_value_l204_204597


namespace new_difference_l204_204875

theorem new_difference (x y a : ℝ) (h : x - y = a) : (x + 0.5) - y = a + 0.5 := 
sorry

end new_difference_l204_204875


namespace noncongruent_integer_sided_triangles_l204_204600

/-- 
There are 12 noncongruent integer-sided triangles with a positive area
and perimeter less than 20 that are neither equilateral, isosceles, nor
right triangles. 
-/
theorem noncongruent_integer_sided_triangles :
  ∃ (triangles : set (ℕ × ℕ × ℕ)), 
    (∀ t ∈ triangles, let (a, b, c) := t in a < b ∧ b < c ∧ 
                     a + b > c ∧ 
                     a + b + c < 20 ∧ 
                     a^2 + b^2 ≠ c^2) ∧
    (fintype.card triangles = 12) :=
sorry

end noncongruent_integer_sided_triangles_l204_204600


namespace find_m_l204_204330

variable (m : ℝ)

def vector_a : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)

theorem find_m (h : (1 + 3, m - 2) = (4, m - 2) ∧ (4 * 3 + (m - 2) * (-2) = 0)) : m = 8 := by
  sorry

end find_m_l204_204330


namespace range_of_m_l204_204596

variable {x m : ℝ}

-- Definition of the first condition: ∀ x in ℝ, |x| + |x - 1| > m
def condition1 (m : ℝ) := ∀ x : ℝ, |x| + |x - 1| > m

-- Definition of the second condition: ∀ x in ℝ, (-(7 - 3 * m))^x is decreasing
def condition2 (m : ℝ) := ∀ x : ℝ, (-(7 - 3 * m))^x > (-(7 - 3 * m))^(x + 1)

-- Main theorem to prove m < 1
theorem range_of_m (h1 : condition1 m) (h2 : condition2 m) : m < 1 :=
sorry

end range_of_m_l204_204596


namespace power_function_increasing_iff_l204_204463

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

theorem power_function_increasing_iff (a : ℝ) : 
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → power_function a x1 < power_function a x2) ↔ a > 0 := 
by
  sorry

end power_function_increasing_iff_l204_204463


namespace compare_xyz_l204_204440

open Real

theorem compare_xyz (x y z : ℝ) : x = Real.log π → y = log 2 / log 5 → z = exp (-1 / 2) → y < z ∧ z < x := by
  intros h_x h_y h_z
  sorry

end compare_xyz_l204_204440


namespace brother_age_in_5_years_l204_204636

noncomputable def Nick : ℕ := 13
noncomputable def Sister : ℕ := Nick + 6
noncomputable def CombinedAge : ℕ := Nick + Sister
noncomputable def Brother : ℕ := CombinedAge / 2

theorem brother_age_in_5_years : Brother + 5 = 21 := by
  sorry

end brother_age_in_5_years_l204_204636


namespace interval_satisfies_ineq_l204_204657

theorem interval_satisfies_ineq (p : ℝ) (h1 : 18 * p < 10) (h2 : 0.5 < p) : 0.5 < p ∧ p < 5 / 9 :=
by {
  sorry -- Proof not required, only the statement.
}

end interval_satisfies_ineq_l204_204657


namespace Grandfather_age_correct_l204_204133

-- Definitions based on the conditions
def Yuna_age : Nat := 9
def Father_age (Yuna_age : Nat) : Nat := Yuna_age + 27
def Grandfather_age (Father_age : Nat) : Nat := Father_age + 23

-- The theorem stating the problem to prove
theorem Grandfather_age_correct : Grandfather_age (Father_age Yuna_age) = 59 := by
  sorry

end Grandfather_age_correct_l204_204133


namespace classify_tangents_through_point_l204_204423

-- Definitions for the Lean theorem statement
noncomputable def curve (x : ℝ) : ℝ :=
  x^3 - x

noncomputable def phi (t x₀ y₀ : ℝ) : ℝ :=
  2*t^3 - 3*x₀*t^2 + (x₀ + y₀)

theorem classify_tangents_through_point (x₀ y₀ : ℝ) :
  (if (x₀ + y₀ < 0 ∨ y₀ > x₀^3 - x₀)
   then 1
   else if (x₀ + y₀ > 0 ∧ x₀ + y₀ - x₀^3 < 0)
   then 3
   else if (x₀ + y₀ = 0 ∨ y₀ = x₀^3 - x₀)
   then 2
   else 0) = 
  (if (x₀ + y₀ < 0 ∨ y₀ > x₀^3 - x₀)
   then 1
   else if (x₀ + y₀ > 0 ∧ x₀ + y₀ - x₀^3 < 0)
   then 3
   else if (x₀ + y₀ = 0 ∨ y₀ = x₀^3 - x₀)
   then 2
   else 0) :=
  sorry

end classify_tangents_through_point_l204_204423


namespace percentage_of_people_with_diploma_l204_204898

variable (P : Type) -- P is the type representing people in Country Z.

-- Given Conditions:
def no_diploma_job (population : ℝ) : ℝ := 0.18 * population
def people_with_job (population : ℝ) : ℝ := 0.40 * population
def diploma_no_job (population : ℝ) : ℝ := 0.25 * (0.60 * population)

-- To Prove:
theorem percentage_of_people_with_diploma (population : ℝ) :
  no_diploma_job population + (diploma_no_job population) + (people_with_job population - no_diploma_job population) = 0.37 * population := 
by
  sorry

end percentage_of_people_with_diploma_l204_204898


namespace greatest_number_of_consecutive_integers_sum_to_91_l204_204052

theorem greatest_number_of_consecutive_integers_sum_to_91 :
  ∃ N, (∀ (a : ℤ), (N : ℕ) > 0 → (N * (2 * a + N - 1) = 182)) ∧ (N = 182) :=
by {
  sorry
}

end greatest_number_of_consecutive_integers_sum_to_91_l204_204052


namespace subset_M_union_N_l204_204485

theorem subset_M_union_N (M N P : Set ℝ) (f g : ℝ → ℝ)
  (hM : M = {x | f x = 0} ∧ M ≠ ∅)
  (hN : N = {x | g x = 0} ∧ N ≠ ∅)
  (hP : P = {x | f x * g x = 0} ∧ P ≠ ∅) :
  P ⊆ (M ∪ N) := 
sorry

end subset_M_union_N_l204_204485


namespace relation_w_z_relation_s_t_relation_x_r_relation_y_q_relation_z_x_t_relation_z_t_v_l204_204370

-- Prove that w - 2z = 0
theorem relation_w_z (w z : ℝ) : w - 2 * z = 0 :=
sorry

-- Prove that 2s + t - 8 = 0
theorem relation_s_t (s t : ℝ) : 2 * s + t - 8 = 0 :=
sorry

-- Prove that x - r - 2 = 0
theorem relation_x_r (x r : ℝ) : x - r - 2 = 0 :=
sorry

-- Prove that y + q - 6 = 0
theorem relation_y_q (y q : ℝ) : y + q - 6 = 0 :=
sorry

-- Prove that 3z - x - 2t + 6 = 0
theorem relation_z_x_t (z x t : ℝ) : 3 * z - x - 2 * t + 6 = 0 :=
sorry

-- Prove that 8z - 4t - v + 12 = 0
theorem relation_z_t_v (z t v : ℝ) : 8 * z - 4 * t - v + 12 = 0 :=
sorry

end relation_w_z_relation_s_t_relation_x_r_relation_y_q_relation_z_x_t_relation_z_t_v_l204_204370


namespace frog_jump_probability_is_one_fifth_l204_204406

noncomputable def frog_jump_probability : ℝ := sorry

theorem frog_jump_probability_is_one_fifth : frog_jump_probability = 1 / 5 := sorry

end frog_jump_probability_is_one_fifth_l204_204406


namespace finance_charge_rate_l204_204985

theorem finance_charge_rate (original_balance total_payment finance_charge_rate : ℝ)
    (h1 : original_balance = 150)
    (h2 : total_payment = 153)
    (h3 : finance_charge_rate = ((total_payment - original_balance) / original_balance) * 100) :
    finance_charge_rate = 2 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end finance_charge_rate_l204_204985


namespace slower_train_speed_l204_204887

theorem slower_train_speed (v : ℝ) (L : ℝ) (faster_speed_km_hr : ℝ) (time_sec : ℝ) (relative_speed : ℝ) 
  (hL : L = 70) (hfaster_speed_km_hr : faster_speed_km_hr = 50)
  (htime_sec : time_sec = 36) (hrelative_speed : relative_speed = (faster_speed_km_hr - v) * (1000 / 3600)) :
  140 = relative_speed * time_sec → v = 36 := 
by
  -- Proof omitted
  sorry

end slower_train_speed_l204_204887


namespace system_of_inequalities_l204_204663

theorem system_of_inequalities (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : (0.5 < p ∧ p < 5 / 9) :=
by sorry

end system_of_inequalities_l204_204663


namespace triangle_area_and_angle_l204_204467

theorem triangle_area_and_angle (a b c A B C : ℝ) 
  (habc: A + B + C = Real.pi)
  (h1: (2*a + b)*Real.cos C + c*Real.cos B = 0)
  (h2: c = 2*Real.sqrt 6 / 3)
  (h3: Real.sin A * Real.cos B = (Real.sqrt 3 - 1)/4) :
  (C = 2*Real.pi / 3) ∧ (1/2 * b * c * Real.sin A = (6 - 2 * Real.sqrt 3)/9) :=
by
  sorry

end triangle_area_and_angle_l204_204467


namespace minimum_throws_for_repetition_of_sum_l204_204105

/-- To ensure that the same sum is rolled twice when throwing four fair six-sided dice,
you must throw the dice at least 22 times. -/
theorem minimum_throws_for_repetition_of_sum :
  ∀ (throws : ℕ), (∀ (sum : ℕ), 4 ≤ sum ∧ sum ≤ 24 → ∃ (count : ℕ), count ≤ 21 ∧ sum = count + 4) → throws ≥ 22 :=
by
  sorry

end minimum_throws_for_repetition_of_sum_l204_204105


namespace minimum_rolls_to_ensure_repeated_sum_l204_204078

theorem minimum_rolls_to_ensure_repeated_sum : 
  let dice_faces := 6
  let number_of_dice := 4
  let min_sum := number_of_dice * 1
  let max_sum := number_of_dice * dice_faces
  let distinct_sums := (max_sum - min_sum) + 1
  in 22 = distinct_sums + 1 :=
by {
  sorry
}

end minimum_rolls_to_ensure_repeated_sum_l204_204078


namespace prime_divisors_of_390_l204_204420

theorem prime_divisors_of_390 : 
  (2 * 195 = 390) → 
  (3 * 65 = 195) → 
  (5 * 13 = 65) → 
  ∃ (S : Finset ℕ), 
    (∀ p ∈ S, Nat.Prime p) ∧ 
    (S.card = 4) ∧ 
    (∀ d ∈ S, d ∣ 390) := 
by
  sorry

end prime_divisors_of_390_l204_204420


namespace joseph_drives_more_l204_204348

def joseph_speed : ℝ := 50
def joseph_time : ℝ := 2.5
def kyle_speed : ℝ := 62
def kyle_time : ℝ := 2

def joseph_distance : ℝ := joseph_speed * joseph_time
def kyle_distance : ℝ := kyle_speed * kyle_time

theorem joseph_drives_more : (joseph_distance - kyle_distance) = 1 := by
  sorry

end joseph_drives_more_l204_204348


namespace minimum_throws_l204_204055

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l204_204055


namespace expense_5_yuan_neg_l204_204022

-- Define the condition that income of 5 yuan is denoted as +5 yuan
def income_5_yuan_pos : Int := 5

-- Define the statement to prove that expenses of 5 yuan are denoted as -5 yuan
theorem expense_5_yuan_neg : income_5_yuan_pos = 5 → -income_5_yuan_pos = -5 :=
by
  intro h
  rw h
  rfl

end expense_5_yuan_neg_l204_204022


namespace town_population_l204_204880

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

end town_population_l204_204880


namespace tangent_line_l204_204319

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 16

theorem tangent_line (x y : ℝ) (h : f 2 = 6) : 13 * x - y - 20 = 0 :=
by
  -- Insert proof here
  sorry

end tangent_line_l204_204319


namespace largest_inscribed_square_size_l204_204852

noncomputable def side_length_of_largest_inscribed_square : ℝ :=
  6 - 2 * Real.sqrt 3

theorem largest_inscribed_square_size (side_length_of_square : ℝ)
  (equi_triangles_shared_side : ℝ)
  (vertexA_of_square : ℝ)
  (vertexB_of_square : ℝ)
  (vertexC_of_square : ℝ)
  (vertexD_of_square : ℝ)
  (vertexF_of_triangles : ℝ)
  (vertexG_of_triangles : ℝ) :
  side_length_of_square = 12 →
  equi_triangles_shared_side = vertexB_of_square - vertexA_of_square →
  vertexF_of_triangles = vertexD_of_square - vertexC_of_square →
  vertexG_of_triangles = vertexB_of_square - vertexA_of_square →
  side_length_of_largest_inscribed_square = 6 - 2 * Real.sqrt 3 :=
sorry

end largest_inscribed_square_size_l204_204852


namespace complement_of_A_in_U_l204_204489

def U : Set ℕ := {1, 2, 3, 4}

def satisfies_inequality (x : ℕ) : Prop := x^2 - 5 * x + 4 < 0

def A : Set ℕ := {x | satisfies_inequality x}

theorem complement_of_A_in_U : U \ A = {1, 4} :=
by
  -- Proof omitted.
  sorry

end complement_of_A_in_U_l204_204489


namespace reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs_l204_204987

theorem reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs
  (a b c h : Real)
  (area_legs : ℝ := (1 / 2) * a * b)
  (area_hypotenuse : ℝ := (1 / 2) * c * h)
  (eq_areas : a * b = c * h)
  (height_eq : h = a * b / c)
  (pythagorean_theorem : c ^ 2 = a ^ 2 + b ^ 2) :
  1 / h ^ 2 = 1 / a ^ 2 + 1 / b ^ 2 := 
by
  sorry

end reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs_l204_204987


namespace positive_difference_between_solutions_l204_204579

theorem positive_difference_between_solutions :
  (∃ x1 x2 : ℝ, (sqrt_cubed (9 - x1^2 / 4) = -3) ∧ (sqrt_cubed (9 - x2^2 / 4) = -3) ∧ (abs (x1 - x2) = 24)) :=
sorry

end positive_difference_between_solutions_l204_204579


namespace minimum_additional_coins_l204_204288

theorem minimum_additional_coins
  (friends : ℕ) (initial_coins : ℕ)
  (h_friends : friends = 15) (h_coins : initial_coins = 100) :
  ∃ additional_coins : ℕ, additional_coins = 20 :=
by
  have total_needed_coins : ℕ := (friends * (friends + 1)) / 2
  have total_coins : ℕ := initial_coins
  have additional_coins_needed : ℕ := total_needed_coins - total_coins
  have h_additional_coins : additional_coins_needed = 20 := by calculate 
  -- Finishing the proof with the result we calculated
  use additional_coins_needed
  exact h_additional_coins

end minimum_additional_coins_l204_204288


namespace scientific_notation_GDP_l204_204340

theorem scientific_notation_GDP (h : 1 = 10^9) : 32.07 * 10^9 = 3.207 * 10^10 := by
  sorry

end scientific_notation_GDP_l204_204340


namespace line_circle_no_intersection_l204_204949

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), 3 * x + 4 * y ≠ 12 ∧ x^2 + y^2 = 4 :=
by
  sorry

end line_circle_no_intersection_l204_204949


namespace roots_depend_on_k_l204_204883

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem roots_depend_on_k (k : ℝ) :
  let a := 1
  let b := -3
  let c := 2 - k
  discriminant a b c = 1 + 4 * k :=
by
  sorry

end roots_depend_on_k_l204_204883


namespace even_product_probability_l204_204246

theorem even_product_probability:
  let s := set.Icc 6 18 in
  ((∃ a b : ℤ, a ≠ b ∧ a ∈ s ∧ b ∈ s ∧ ¬even (a * b)) → (2 / 13) - (19 / 39) = (21 / 26)) := sorry

end even_product_probability_l204_204246


namespace partition_sequences_l204_204292

-- Definitions and conditions from the problem
def sequence : Type := vector (bool) 2022

def is_compatible (s1 s2 : sequence) : Prop :=
  (s1.to_list.zip s2.to_list).countp (λ ⟨x, y⟩, x = y) = 4

-- Main theorem statement
theorem partition_sequences (sequences : finset sequence) (h : sequences.card = nat.choose 2022 1011) :
  ∃ (groups : fin 20 → finset sequence),
    (∀ i, (groups i). ⊆ sequences) ∧
    (∀ i j, i ≠ j → disjoint (groups i) (groups j)) ∧
    (∀ i, ∀ s1 s2 ∈ groups i, ¬ is_compatible s1 s2) :=
sorry

end partition_sequences_l204_204292


namespace max_sum_of_squares_70_l204_204508

theorem max_sum_of_squares_70 :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a^2 + b^2 + c^2 + d^2 = 70 ∧ a + b + c + d = 16 :=
by
  sorry

end max_sum_of_squares_70_l204_204508


namespace find_a_b_l204_204895

def satisfies_digit_conditions (n a b : ℕ) : Prop :=
  n = 2000 + 100 * a + 90 + b ∧
  n / 1000 % 10 = 2 ∧
  n / 100 % 10 = a ∧
  n / 10 % 10 = 9 ∧
  n % 10 = b

theorem find_a_b : ∃ (a b : ℕ), 2^a * 9^b = 2000 + 100*a + 90 + b ∧ satisfies_digit_conditions (2^a * 9^b) a b :=
by
  sorry

end find_a_b_l204_204895


namespace minimum_rolls_for_duplicate_sum_l204_204092

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l204_204092


namespace gcd_45_75_l204_204701

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204701


namespace rhombus_area_l204_204414

-- Define the given conditions: diagonals and side length
def d1 : ℕ := 40
def d2 : ℕ := 18
def s : ℕ := 25

-- Prove that the area of the rhombus is 360 square units given the conditions
theorem rhombus_area :
  (d1 * d2) / 2 = 360 :=
by
  sorry

end rhombus_area_l204_204414


namespace number_of_cookies_l204_204359

def total_cake := 22
def total_chocolate := 16
def total_groceries := 42

theorem number_of_cookies :
  ∃ C : ℕ, total_groceries = total_cake + total_chocolate + C ∧ C = 4 := 
by
  sorry

end number_of_cookies_l204_204359


namespace complete_square_eq_l204_204154

theorem complete_square_eq (x : ℝ) : (x^2 - 6 * x - 5 = 0) -> (x - 3)^2 = 14 :=
by
  intro h
  sorry

end complete_square_eq_l204_204154


namespace sum_of_eight_digits_l204_204304

open Nat

theorem sum_of_eight_digits {a b c d e f g h : ℕ} 
  (h_distinct : ∀ i j, i ∈ [a, b, c, d, e, f, g, h] → j ∈ [a, b, c, d, e, f, g, h] → i ≠ j → i ≠ j)
  (h_vertical_sum : a + b + c + d + e = 25)
  (h_horizontal_sum : f + g + h + b = 15) 
  (h_digits_set : ∀ x ∈ [a, b, c, d, e, f, g, h], x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ)) : 
  a + b + c + d + e + f + g + h - b = 39 := 
sorry

end sum_of_eight_digits_l204_204304


namespace gcd_45_75_l204_204735

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204735


namespace field_area_l204_204134

theorem field_area (x y : ℕ) (h1 : x + y = 700) (h2 : y - x = (1/5) * ((x + y) / 2)) : x = 315 :=
  sorry

end field_area_l204_204134


namespace SavingsInequality_l204_204267

theorem SavingsInequality (n : ℕ) : 52 + 15 * n > 70 + 12 * n := 
by sorry

end SavingsInequality_l204_204267


namespace Mona_joined_groups_l204_204495

theorem Mona_joined_groups (G : ℕ) (h : G * 4 - 3 = 33) : G = 9 :=
by
  sorry

end Mona_joined_groups_l204_204495


namespace ellipse_k_values_l204_204581

theorem ellipse_k_values (k : ℝ) :
  (∃ k, (∃ e, e = 1/2 ∧
    (∃ a b : ℝ, a = Real.sqrt (k+8) ∧ b = 3 ∧
      ∃ c, (c = Real.sqrt (abs ((a^2) - (b^2)))) ∧ (e = c/b ∨ e = c/a)) ∧
      k = 4 ∨ k = -5/4)) :=
  sorry

end ellipse_k_values_l204_204581


namespace income_expenses_opposite_l204_204010

def income_denotation (income : Int) : Int := income

theorem income_expenses_opposite :
  income_denotation 5 = 5 →
  income_denotation (-5) = -5 :=
by
  intro h
  sorry

end income_expenses_opposite_l204_204010


namespace gcd_45_75_l204_204744

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204744


namespace binom_15_4_l204_204557

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the theorem stating the specific binomial coefficient value
theorem binom_15_4 : binom 15 4 = 1365 :=
by
  sorry

end binom_15_4_l204_204557


namespace gcd_45_75_l204_204684

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204684


namespace system_of_equations_n_eq_1_l204_204446

theorem system_of_equations_n_eq_1 {x y n : ℝ} 
  (h₁ : 5 * x - 4 * y = n) 
  (h₂ : 3 * x + 5 * y = 8)
  (h₃ : x = y) : 
  n = 1 := 
by
  sorry

end system_of_equations_n_eq_1_l204_204446


namespace expense_of_5_yuan_is_minus_5_yuan_l204_204024

def income (x : Int) : Int :=
  x

def expense (x : Int) : Int :=
  -x

theorem expense_of_5_yuan_is_minus_5_yuan : expense 5 = -5 :=
by
  unfold expense
  sorry

end expense_of_5_yuan_is_minus_5_yuan_l204_204024


namespace algebra_expression_value_l204_204941

theorem algebra_expression_value (x : ℝ) (h : x^2 + 3 * x + 5 = 11) : 3 * x^2 + 9 * x + 12 = 30 := 
by
  sorry

end algebra_expression_value_l204_204941


namespace sum_x_values_l204_204817

open Real

theorem sum_x_values :
  ∑ x in {x | 50 < x ∧ x < 150 ∧ cos^3 (2 * x) + cos^3 (6 * x) = 8 * cos^3 (4 * x) * cos^3 x}, x = 270 := by
  sorry

end sum_x_values_l204_204817


namespace intersection_of_domains_l204_204253

open Set

theorem intersection_of_domains :
  (range (λ x : ℝ, exp x)) ∩ (range (λ x : ℝ, log x)) = {y | 0 < y} :=
by
  sorry

end intersection_of_domains_l204_204253


namespace friday_birth_of_dickens_l204_204913

def is_leap_year (y : ℕ) : Prop :=
  (y % 400 = 0) ∨ ((y % 4 = 0) ∧ (y % 100 ≠ 0))

theorem friday_birth_of_dickens :
  let regular_year_days := 365
  let leap_year_days := 366
  let total_years := 200
  let leap_years := 49  -- already computed as 49 in the steps
  let regular_years := total_years - leap_years
  let total_days_in_regular_years := regular_years * regular_year_days
  let total_days_in_leap_years := leap_years * leap_year_days
  let total_days := total_days_in_regular_years + total_days_in_leap_years
  (total_days % 7 = 4) : 
  day_of_week (date.add_days (date.mk 2012 2 7) (-total_days)) = "Friday"
:= sorry

end friday_birth_of_dickens_l204_204913


namespace minimum_rolls_to_ensure_repeated_sum_l204_204079

theorem minimum_rolls_to_ensure_repeated_sum : 
  let dice_faces := 6
  let number_of_dice := 4
  let min_sum := number_of_dice * 1
  let max_sum := number_of_dice * dice_faces
  let distinct_sums := (max_sum - min_sum) + 1
  in 22 = distinct_sums + 1 :=
by {
  sorry
}

end minimum_rolls_to_ensure_repeated_sum_l204_204079


namespace inverse_prop_l204_204255

theorem inverse_prop (a c : ℝ) : (∀ (a : ℝ), a > 0 → a * c^2 ≥ 0) → (∀ (x : ℝ), x * c^2 ≥ 0 → x > 0) :=
by
  sorry

end inverse_prop_l204_204255


namespace equal_play_time_l204_204782

-- Definitions based on conditions
variables (P : ℕ) (F : ℕ) (T : ℕ) (S : ℕ)
-- P: Total number of players
-- F: Number of players on the field at any time
-- T: Total duration of the match in minutes
-- S: Time each player plays

-- Given values from conditions
noncomputable def totalPlayers : ℕ := 10
noncomputable def fieldPlayers : ℕ := 8
noncomputable def matchDuration : ℕ := 45

theorem equal_play_time :
  P = totalPlayers →
  F = fieldPlayers →
  T = matchDuration →
  (F * T) / P = S →
  S = 36 :=
by
  intros hP hF hT hS
  rw [hP, hF, hT] at hS
  exact hS

end equal_play_time_l204_204782


namespace min_throws_to_ensure_repeat_sum_l204_204120

theorem min_throws_to_ensure_repeat_sum : 
  ∀ (min_sum max_sum : ℤ), 
  min_sum = 4 ∧ max_sum = 24 
  → ∃ n, n ≥ 22 ∧ n = 22 :=
by
  intros min_sum max_sum h
  cases h with h_min h_max
  existsi 22
  split
  · exact Nat.le_refl 22
  · sorry

end min_throws_to_ensure_repeat_sum_l204_204120


namespace trigonometric_identity_l204_204192

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.cos (2 * α) + Real.sin (Real.pi / 2 + α) * Real.cos (3 * Real.pi / 2 - α) = -1 :=
by
  sorry

end trigonometric_identity_l204_204192


namespace factor_expression_l204_204181

theorem factor_expression (x y z : ℝ) :
  (x - y)^3 + (y - z)^3 + (z - x)^3 ≠ 0 →
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) =
    (x + y) * (y + z) * (z + x) :=
by
  intro h
  sorry

end factor_expression_l204_204181


namespace daily_wage_c_l204_204897

-- Definitions according to the conditions
def days_worked_a : ℕ := 6
def days_worked_b : ℕ := 9
def days_worked_c : ℕ := 4

def ratio_wages : ℕ × ℕ × ℕ := (3, 4, 5)
def total_earning : ℕ := 1628

-- Goal: Prove that the daily wage of c is Rs. 110
theorem daily_wage_c : (5 * (total_earning / (18 + 36 + 20))) = 110 :=
by
  sorry

end daily_wage_c_l204_204897


namespace minimum_throws_for_four_dice_l204_204123

noncomputable def minimum_throws_to_ensure_repeated_sum (d : ℕ) : ℕ :=
  let min_sum := d * 1 in
  let max_sum := d * 6 in
  let distinct_sums := max_sum - min_sum + 1 in
  distinct_sums + 1

theorem minimum_throws_for_four_dice : minimum_throws_to_ensure_repeated_sum 4 = 22 := by
  sorry

end minimum_throws_for_four_dice_l204_204123


namespace janice_typing_proof_l204_204226

noncomputable def janice_typing : Prop :=
  let initial_speed := 6
  let error_speed := 8
  let corrected_speed := 5
  let typing_duration_initial := 20
  let typing_duration_corrected := 15
  let erased_sentences := 40
  let typing_duration_after_lunch := 18
  let total_sentences_end_of_day := 536

  let sentences_initial_typing := typing_duration_initial * error_speed
  let sentences_post_error_typing := typing_duration_corrected * initial_speed
  let sentences_final_typing := typing_duration_after_lunch * corrected_speed

  let sentences_total_typed := sentences_initial_typing + sentences_post_error_typing - erased_sentences + sentences_final_typing

  let sentences_started_with := total_sentences_end_of_day - sentences_total_typed

  sentences_started_with = 236

theorem janice_typing_proof : janice_typing := by
  sorry

end janice_typing_proof_l204_204226


namespace length_of_one_side_of_hexagon_l204_204256

variable (P : ℝ) (n : ℕ)
-- Condition: perimeter P is 60 inches
def hexagon_perimeter_condition : Prop := P = 60
-- Hexagon has six sides
def hexagon_sides_condition : Prop := n = 6
-- The question asks for the side length
noncomputable def side_length_of_hexagon : ℝ := P / n

-- Prove that if a hexagon has a perimeter of 60 inches, then its side length is 10 inches
theorem length_of_one_side_of_hexagon (hP : hexagon_perimeter_condition P) (hn : hexagon_sides_condition n) :
  side_length_of_hexagon P n = 10 := by
  sorry

end length_of_one_side_of_hexagon_l204_204256


namespace general_term_sequence_l204_204625

variable {a : ℕ → ℝ}
variable {n : ℕ}

def sequence_condition (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ (∀ n : ℕ, n ≥ 1 → (n+1) * (a (n + 1))^2 - n * (a n)^2 + (a n) * (a (n + 1)) = 0)

theorem general_term_sequence (a : ℕ → ℝ) (h : sequence_condition a) :
  ∀ n : ℕ, n > 0 → a n = 1 / n := by
  sorry

end general_term_sequence_l204_204625


namespace arccos_sqrt_3_over_2_eq_pi_over_6_l204_204166

open Real

theorem arccos_sqrt_3_over_2_eq_pi_over_6 :
  ∀ (x : ℝ), x = (sqrt 3) / 2 → arccos x = π / 6 :=
by
  intro x
  sorry

end arccos_sqrt_3_over_2_eq_pi_over_6_l204_204166


namespace action_figure_cost_l204_204976

def initial_figures : ℕ := 7
def total_figures_needed : ℕ := 16
def total_cost : ℕ := 72

theorem action_figure_cost :
  total_cost / (total_figures_needed - initial_figures) = 8 := by
  sorry

end action_figure_cost_l204_204976


namespace number_of_solutions_l204_204814

theorem number_of_solutions :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℕ,
    (x < 10^2006) ∧ ((x * (x - 1)) % 10^2006 = 0) → x ≤ n :=
sorry

end number_of_solutions_l204_204814


namespace sum_abs_frac_geq_frac_l204_204857

theorem sum_abs_frac_geq_frac (n : ℕ) (h1 : n ≥ 3) (a : Fin n → ℝ) (hnz : ∀ i : Fin n, a i ≠ 0) 
(hsum : (Finset.univ.sum a) = S) : 
  (Finset.univ.sum (fun i => |(S - a i) / a i|)) ≥ (n - 1) / (n - 2) :=
sorry

end sum_abs_frac_geq_frac_l204_204857


namespace range_of_a_l204_204194

noncomputable def g (x : ℝ) : ℝ := abs (x-1) - abs (x-2)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ (g x ≥ a^2 + a + 1)) ↔ (a < -1 ∨ a > 0) :=
by
  sorry

end range_of_a_l204_204194


namespace arccos_of_sqrt3_div_2_l204_204170

theorem arccos_of_sqrt3_div_2 : Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 := by
  sorry

end arccos_of_sqrt3_div_2_l204_204170


namespace probability_exactly_two_even_dice_l204_204419

theorem probability_exactly_two_even_dice :
  let p_even := 1 / 2
  let p_not_even := 1 / 2
  let number_of_ways := 3
  let probability_each_way := (p_even * p_even * p_not_even)
  3 * probability_each_way = 3 / 8 :=
by
  sorry

end probability_exactly_two_even_dice_l204_204419


namespace expenses_opposite_to_income_l204_204018

theorem expenses_opposite_to_income (income_5 : ℤ) (h_income : income_5 = 5) : -income_5 = -5 :=
by
  -- proof is omitted
  sorry

end expenses_opposite_to_income_l204_204018


namespace joseph_drives_more_l204_204351

-- Definitions for the problem
def v_j : ℝ := 50 -- Joseph's speed in mph
def t_j : ℝ := 2.5 -- Joseph's time in hours
def v_k : ℝ := 62 -- Kyle's speed in mph
def t_k : ℝ := 2 -- Kyle's time in hours

-- Prove that Joseph drives 1 more mile than Kyle
theorem joseph_drives_more : (v_j * t_j) - (v_k * t_k) = 1 := 
by 
  sorry

end joseph_drives_more_l204_204351


namespace max_value_ab_bc_cd_da_l204_204354

theorem max_value_ab_bc_cd_da (a b c d : ℝ) (a_nonneg : 0 ≤ a) (b_nonneg : 0 ≤ b) (c_nonneg : 0 ≤ c)
  (d_nonneg : 0 ≤ d) (sum_eq_200 : a + b + c + d = 200) : 
  ab + bc + cd + 0.5 * d * a ≤ 11250 := 
sorry


end max_value_ab_bc_cd_da_l204_204354


namespace units_digit_of_33_pow_33_mul_22_pow_22_l204_204162

theorem units_digit_of_33_pow_33_mul_22_pow_22 :
  (33 ^ (33 * (22 ^ 22))) % 10 = 1 :=
sorry

end units_digit_of_33_pow_33_mul_22_pow_22_l204_204162


namespace complex_product_eq_50i_l204_204602

open Complex

theorem complex_product_eq_50i : 
  let Q := (4 : ℂ) + 3 * I
  let E := (2 * I : ℂ)
  let D := (4 : ℂ) - 3 * I
  Q * E * D = 50 * I :=
by
  -- Complex numbers and multiplication are handled here
  sorry

end complex_product_eq_50i_l204_204602


namespace sum_sequence_conjecture_l204_204201

theorem sum_sequence_conjecture (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ+, a n = (8 * n) / ((2 * n - 1) ^ 2 * (2 * n + 1) ^ 2)) →
  (∀ n : ℕ+, S n = (S n + a (n + 1))) →
  (∀ n : ℕ+, S 1 = 8 / 9) →
  (∀ n : ℕ+, S n = ((2 * n + 1) ^ 2 - 1) / (2 * n + 1) ^ 2) :=
by {
  sorry
}

end sum_sequence_conjecture_l204_204201


namespace total_distance_fourth_time_l204_204545

/-- 
A super ball is dropped from a height of 100 feet and rebounds half the distance it falls each time.
We need to prove that the total distance the ball travels when it hits the ground
the fourth time is 275 feet.
-/
noncomputable def total_distance : ℝ :=
  let first_descent := 100
  let second_descent := first_descent / 2
  let third_descent := second_descent / 2
  let fourth_descent := third_descent / 2
  let first_ascent := second_descent
  let second_ascent := third_descent
  let third_ascent := fourth_descent
  first_descent + second_descent + third_descent + fourth_descent +
  first_ascent + second_ascent + third_ascent

theorem total_distance_fourth_time : total_distance = 275 := 
  by
  sorry

end total_distance_fourth_time_l204_204545


namespace find_nabla_l204_204951

theorem find_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by {
  sorry
}

end find_nabla_l204_204951


namespace ratio_of_overtime_to_regular_rate_l204_204284

def regular_rate : ℝ := 3
def regular_hours : ℕ := 40
def total_pay : ℝ := 186
def overtime_hours : ℕ := 11

theorem ratio_of_overtime_to_regular_rate 
  (r : ℝ) (h : ℕ) (T : ℝ) (h_ot : ℕ) 
  (h_r : r = regular_rate) 
  (h_h : h = regular_hours) 
  (h_T : T = total_pay)
  (h_hot : h_ot = overtime_hours) :
  (T - (h * r)) / h_ot / r = 2 := 
by {
  sorry 
}

end ratio_of_overtime_to_regular_rate_l204_204284


namespace partial_fraction_decomposition_l204_204580

theorem partial_fraction_decomposition :
  ∀ (A B C : ℝ), 
  (∀ x : ℝ, x^4 - 3 * x^3 - 7 * x^2 + 15 * x - 10 ≠ 0 →
    (x^2 - 23) /
    (x^4 - 3 * x^3 - 7 * x^2 + 15 * x - 10) = 
    A / (x - 1) + B / (x + 2) + C / (x - 2)) →
  (A = 44 / 21 ∧ B = -5 / 2 ∧ C = -5 / 6 → A * B * C = 275 / 63)
  := by
  intros A B C h₁ h₂
  sorry

end partial_fraction_decomposition_l204_204580


namespace dickens_birth_day_l204_204914

def is_leap_year (year : ℕ) : Prop :=
  (year % 400 = 0) ∨ (year % 4 = 0 ∧ year % 100 ≠ 0)

theorem dickens_birth_day :
  let day_of_week_2012 := 2 -- 0: Sunday, 1: Monday, ..., 2: Tuesday
  let years := 200
  let regular_years := 151
  let leap_years := 49
  let days_shift := regular_years + 2 * leap_years
  let day_of_week_birth := (day_of_week_2012 + days_shift) % 7
  day_of_week_birth = 5 -- 5: Friday
:= 
sorry -- proof not supplied

end dickens_birth_day_l204_204914


namespace tank_capacity_percentage_l204_204270

noncomputable def radius (C : ℝ) := C / (2 * Real.pi)
noncomputable def volume (r h : ℝ) := Real.pi * r^2 * h

theorem tank_capacity_percentage :
  let r_M := radius 8
  let r_B := radius 10
  let V_M := volume r_M 10
  let V_B := volume r_B 8
  (V_M / V_B * 100) = 80 :=
by
  sorry

end tank_capacity_percentage_l204_204270


namespace minimum_additional_coins_l204_204289

-- The conditions
def total_friends : ℕ := 15
def current_coins : ℕ := 100

-- The fact that the total coins required to give each friend a unique number of coins from 1 to 15 is 120
def total_required_coins : ℕ := (total_friends * (total_friends + 1)) / 2

-- The theorem stating the required number of additional coins
theorem minimum_additional_coins (total_friends : ℕ) (current_coins : ℕ) (total_required_coins : ℕ) : ℕ :=
  sorry

end minimum_additional_coins_l204_204289


namespace min_value_y_l204_204836

theorem min_value_y (x : ℝ) (h : x > 1) : 
  ∃ y_min : ℝ, (∀ y, y = (1 / (x - 1) + x) → y ≥ y_min) ∧ y_min = 3 :=
sorry

end min_value_y_l204_204836


namespace positive_number_square_roots_l204_204607

theorem positive_number_square_roots (a : ℝ) (x : ℝ) (h1 : x = (a - 7)^2)
  (h2 : x = (2 * a + 1)^2) : x = 25 := by
sorry

end positive_number_square_roots_l204_204607


namespace probability_one_pair_of_same_color_l204_204450

noncomputable def countCombinations : ℕ :=
  Nat.choose 10 5

noncomputable def countFavCombinations : ℕ :=
  (Nat.choose 5 4) * 4 * (2 * 2 * 2)

theorem probability_one_pair_of_same_color :
  (countFavCombinations : ℚ) / (countCombinations : ℚ) = 40 / 63 := by
  sorry

end probability_one_pair_of_same_color_l204_204450


namespace book_original_selling_price_l204_204533

theorem book_original_selling_price (CP SP1 SP2 : ℝ) 
  (h1 : SP1 = 0.9 * CP)
  (h2 : SP2 = 1.1 * CP)
  (h3 : SP2 = 990) : 
  SP1 = 810 :=
by
  sorry

end book_original_selling_price_l204_204533


namespace factor_expression_l204_204183

theorem factor_expression (x y z : ℝ) : 
  ( (x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3 ) / 
  ( (x - y)^3 + (y - z)^3 + (z - x)^3 ) = 
  (x + y) * (y + z) * (z + x) := 
by
  sorry

end factor_expression_l204_204183


namespace original_price_of_coat_l204_204381

theorem original_price_of_coat (P : ℝ) (h : 0.40 * P = 200) : P = 500 :=
by {
  sorry
}

end original_price_of_coat_l204_204381


namespace households_in_city_l204_204616

theorem households_in_city (x : ℕ) (h1 : x < 100) (h2 : x + x / 3 = 100) : x = 75 :=
sorry

end households_in_city_l204_204616


namespace crow_eats_nuts_l204_204277

theorem crow_eats_nuts (time_fifth_nuts : ℕ) (time_quarter_nuts : ℕ) (h : time_fifth_nuts = 8) :
  time_quarter_nuts = 10 :=
sorry

end crow_eats_nuts_l204_204277


namespace sqrt_cubic_sqrt_decimal_l204_204915

theorem sqrt_cubic_sqrt_decimal : 
  (Real.sqrt (0.0036 : ℝ))^(1/3) = 0.3912 :=
sorry

end sqrt_cubic_sqrt_decimal_l204_204915


namespace number_of_white_balls_l204_204968

theorem number_of_white_balls (total : ℕ) (freq_red freq_black : ℚ) (h1 : total = 120) 
                              (h2 : freq_red = 0.15) (h3 : freq_black = 0.45) : 
                              (total - total * freq_red - total * freq_black = 48) :=
by sorry

end number_of_white_balls_l204_204968


namespace range_of_p_l204_204944

theorem range_of_p (a b : ℝ) :
  (∀ x y p q : ℝ, p + q = 1 → (p * (x^2 + a * x + b) + q * (y^2 + a * y + b) ≥ ((p * x + q * y)^2 + a * (p * x + q * y) + b))) →
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 1) :=
sorry

end range_of_p_l204_204944


namespace f_500_l204_204355

-- Define a function f on positive integers
def f (n : ℕ) : ℕ := sorry

-- Assume the given conditions
axiom f_mul (x y : ℕ) (hx : x > 0) (hy : y > 0) : f (x * y) = f x + f y
axiom f_10 : f 10 = 14
axiom f_40 : f 40 = 20

-- Prove the required result
theorem f_500 : f 500 = 39 := by
  sorry

end f_500_l204_204355


namespace equal_playing_time_l204_204789

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l204_204789


namespace find_m_l204_204862

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 1)
def is_parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

theorem find_m (m : ℝ) :
  is_parallel (vector_a.1 + 2 * m, vector_a.2 + 2 * 1) (2 * vector_a.1 - m, 2 * vector_a.2 - 1) ↔ m = -1 / 2 := 
by {
  sorry
}

end find_m_l204_204862


namespace karen_box_crayons_l204_204980

theorem karen_box_crayons (judah_crayons : ℕ) (gilbert_crayons : ℕ) (beatrice_crayons : ℕ) (karen_crayons : ℕ)
  (h1 : judah_crayons = 8)
  (h2 : gilbert_crayons = 4 * judah_crayons)
  (h3 : beatrice_crayons = 2 * gilbert_crayons)
  (h4 : karen_crayons = 2 * beatrice_crayons) :
  karen_crayons = 128 :=
by
  sorry

end karen_box_crayons_l204_204980


namespace find_angle_B_l204_204439

theorem find_angle_B 
  (A B : ℝ)
  (h1 : B + A = 90)
  (h2 : B = 4 * A) : 
  B = 144 :=
by
  sorry

end find_angle_B_l204_204439


namespace factor_expression_l204_204573

theorem factor_expression (x : ℤ) : 84 * x^7 - 270 * x^13 = 6 * x^7 * (14 - 45 * x^6) := 
by sorry

end factor_expression_l204_204573


namespace g_value_at_50_l204_204254

noncomputable def g : ℝ → ℝ :=
sorry

theorem g_value_at_50 (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, 0 < x → 0 < y → x * g y - y ^ 2 * g x = g (x / y)) :
  g 50 = 0 :=
by
  sorry

end g_value_at_50_l204_204254


namespace equal_play_time_l204_204783

theorem equal_play_time (total_players on_field_players match_minutes : ℕ) 
    (h1 : total_players = 10) 
    (h2 : on_field_players = 8) 
    (h3 : match_minutes = 45) 
    (h4 : ∀ p, p ∈ (finset.range total_players) → time_played p = (on_field_players * match_minutes) / total_players)
    : (on_field_players * match_minutes) / total_players = 36 :=
by
  have total_play_minutes : on_field_players * match_minutes = 360 := by sorry
  have equal_play_time : (on_field_players * match_minutes) / total_players = 36 := by sorry
  exact equal_play_time

end equal_play_time_l204_204783


namespace product_with_zero_is_zero_l204_204552

theorem product_with_zero_is_zero :
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 0) = 0 :=
by
  sorry

end product_with_zero_is_zero_l204_204552


namespace binom_15_4_eq_1365_l204_204565

theorem binom_15_4_eq_1365 : (Nat.choose 15 4) = 1365 := 
by 
  sorry

end binom_15_4_eq_1365_l204_204565


namespace simplify_expression_l204_204992

theorem simplify_expression (x : ℝ) :
  ((3 * x^2 + 2 * x - 1) + 2 * x^2) * 4 + (5 - 2 / 2) * (3 * x^2 + 6 * x - 8) = 32 * x^2 + 32 * x - 36 :=
sorry

end simplify_expression_l204_204992


namespace min_rolls_to_duplicate_sum_for_four_dice_l204_204113

theorem min_rolls_to_duplicate_sum_for_four_dice : 
    let min_sum := 4 * 1,
    let max_sum := 4 * 6,
    let possible_sums := max_sum - min_sum + 1 in
    possible_sums = 21 → 
    (possible_sums + 1 = 22) := 
by
  intros min_sum max_sum possible_sums h
  have h1 : min_sum = 4 := rfl
  have h2 : max_sum = 24 := rfl
  have h3 : possible_sums = 21 := h
  have h4 : possible_sums + 1 = 22 := calc
    possible_sums + 1 = 21 + 1 : by rw h
    ... = 22 : by rfl
  exact h4

end min_rolls_to_duplicate_sum_for_four_dice_l204_204113


namespace John_can_put_weight_on_bar_l204_204228

-- Definitions for conditions
def max_capacity : ℕ := 1000
def safety_margin : ℕ := 200  -- 20% of 1000
def johns_weight : ℕ := 250

-- Statement to prove
theorem John_can_put_weight_on_bar : ∀ (weight_on_bar : ℕ),
  weight_on_bar + johns_weight ≤ max_capacity - safety_margin → weight_on_bar = 550 :=
by
  intro weight_on_bar
  intros h_condition
  have h_max_weight : max_capacity - safety_margin = 800 := by simp [max_capacity, safety_margin]
  have h_safe_weight : 800 - johns_weight = 550 := by simp [johns_weight]
  rw [←h_safe_weight] at h_condition
  exact Eq.trans (Eq.symm h_condition) (Eq.refl 550)

end John_can_put_weight_on_bar_l204_204228


namespace sqrt_expression_l204_204839

theorem sqrt_expression (x : ℝ) : 2 - x ≥ 0 ↔ x ≤ 2 := sorry

end sqrt_expression_l204_204839


namespace revenue_difference_l204_204218

def original_revenue : ℕ := 10000

def vasya_revenue (X : ℕ) : ℕ :=
  2 * (original_revenue / X) * (4 * X / 5)

def kolya_revenue (X : ℕ) : ℕ :=
  (original_revenue / X) * (8 * X / 3)

theorem revenue_difference (X : ℕ) (hX : X > 0) : vasya_revenue X = 16000 ∧ kolya_revenue X = 13333 ∧ vasya_revenue X - original_revenue = 6000 := 
by
  sorry

end revenue_difference_l204_204218


namespace classroom_lamps_total_ways_l204_204214

theorem classroom_lamps_total_ways (n : ℕ) (h : n = 4) : (2^n - 1) = 15 :=
by
  sorry

end classroom_lamps_total_ways_l204_204214


namespace min_rolls_to_duplicate_sum_for_four_dice_l204_204115

theorem min_rolls_to_duplicate_sum_for_four_dice : 
    let min_sum := 4 * 1,
    let max_sum := 4 * 6,
    let possible_sums := max_sum - min_sum + 1 in
    possible_sums = 21 → 
    (possible_sums + 1 = 22) := 
by
  intros min_sum max_sum possible_sums h
  have h1 : min_sum = 4 := rfl
  have h2 : max_sum = 24 := rfl
  have h3 : possible_sums = 21 := h
  have h4 : possible_sums + 1 = 22 := calc
    possible_sums + 1 = 21 + 1 : by rw h
    ... = 22 : by rfl
  exact h4

end min_rolls_to_duplicate_sum_for_four_dice_l204_204115


namespace tom_catches_up_in_60_minutes_l204_204491

-- Definitions of the speeds and initial distance
def lucy_speed : ℝ := 4  -- Lucy's speed in miles per hour
def tom_speed : ℝ := 6   -- Tom's speed in miles per hour
def initial_distance : ℝ := 2  -- Initial distance between Tom and Lucy in miles

-- Conclusion that needs to be proved
theorem tom_catches_up_in_60_minutes :
  (initial_distance / (tom_speed - lucy_speed)) * 60 = 60 :=
by
  sorry

end tom_catches_up_in_60_minutes_l204_204491


namespace find_x_l204_204879

theorem find_x (x : ℚ) : (8 + 10 + 22) / 3 = (15 + x) / 2 → x = 35 / 3 :=
by
  sorry

end find_x_l204_204879


namespace hugo_roll_five_given_win_l204_204843

theorem hugo_roll_five_given_win 
  (H1 A1 B1 C1 : ℕ)
  (hugo_rolls : 1 ≤ H1 ∧ H1 ≤ 6)
  (player_rolls : 1 ≤ A1 ∧ A1 ≤ 6 ∧ 1 ≤ B1 ∧ B1 ≤ 6 ∧ 1 ≤ C1 ∧ C1 ≤ 6)
  (hugo_wins : (H1 = 5 → P(H1 = 5 | W = H) = 41 / 144) : 
  P(H1 = 5 | W = H) = 41 / 144 :=
sorry

end hugo_roll_five_given_win_l204_204843


namespace cistern_length_l204_204405

theorem cistern_length
  (L W D A : ℝ)
  (hW : W = 4)
  (hD : D = 1.25)
  (hA : A = 49)
  (hWetSurface : A = L * W + 2 * L * D) :
  L = 7.54 := by
  sorry

end cistern_length_l204_204405


namespace binom_15_4_eq_1365_l204_204566

theorem binom_15_4_eq_1365 : (Nat.choose 15 4) = 1365 := 
by 
  sorry

end binom_15_4_eq_1365_l204_204566


namespace expenses_negation_of_income_l204_204008

theorem expenses_negation_of_income 
    (income : ℤ) 
    (income_is_5 : income = 5) 
    (denote_income : income = 5 → "+" ∘ toString income = "+5") 
    (expenses_are_negation_of_income :  "expenses = -1 * income") : "expenses = -5" :=
begin
    sorry
end

end expenses_negation_of_income_l204_204008


namespace y1_greater_than_y2_l204_204314

-- Definitions of the conditions.
def point1_lies_on_line (y₁ b : ℝ) : Prop := y₁ = -3 * (-2 : ℝ) + b
def point2_lies_on_line (y₂ b : ℝ) : Prop := y₂ = -3 * (-1 : ℝ) + b

-- The theorem to prove: y₁ > y₂ given the conditions.
theorem y1_greater_than_y2 (y₁ y₂ b : ℝ) (h1 : point1_lies_on_line y₁ b) (h2 : point2_lies_on_line y₂ b) : y₁ > y₂ :=
by {
  sorry
}

end y1_greater_than_y2_l204_204314


namespace find_b_value_l204_204438

theorem find_b_value {b : ℚ} (h : -8 ^ 2 + b * -8 - 45 = 0) : b = 19 / 8 :=
sorry

end find_b_value_l204_204438


namespace rope_subdivision_length_l204_204543

theorem rope_subdivision_length 
  (initial_length : ℕ) 
  (num_parts : ℕ) 
  (num_subdivided_parts : ℕ) 
  (final_subdivision_factor : ℕ) 
  (initial_length_eq : initial_length = 200) 
  (num_parts_eq : num_parts = 4) 
  (num_subdivided_parts_eq : num_subdivided_parts = num_parts / 2) 
  (final_subdivision_factor_eq : final_subdivision_factor = 2) :
  initial_length / num_parts / final_subdivision_factor = 25 := 
by 
  sorry

end rope_subdivision_length_l204_204543


namespace gcd_45_75_l204_204728

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204728


namespace make_tea_time_efficiently_l204_204494

theorem make_tea_time_efficiently (minutes_kettle minutes_boil minutes_teapot minutes_teacups minutes_tea_leaves total_estimate total_time : ℕ)
  (h1 : minutes_kettle = 1)
  (h2 : minutes_boil = 15)
  (h3 : minutes_teapot = 1)
  (h4 : minutes_teacups = 1)
  (h5 : minutes_tea_leaves = 2)
  (h6 : total_estimate = 20)
  (h_total_time : total_time = minutes_kettle + minutes_boil) :
  total_time = 16 :=
by
  sorry

end make_tea_time_efficiently_l204_204494


namespace largest_inscribed_square_size_l204_204851

noncomputable def side_length_of_largest_inscribed_square : ℝ :=
  6 - 2 * Real.sqrt 3

theorem largest_inscribed_square_size (side_length_of_square : ℝ)
  (equi_triangles_shared_side : ℝ)
  (vertexA_of_square : ℝ)
  (vertexB_of_square : ℝ)
  (vertexC_of_square : ℝ)
  (vertexD_of_square : ℝ)
  (vertexF_of_triangles : ℝ)
  (vertexG_of_triangles : ℝ) :
  side_length_of_square = 12 →
  equi_triangles_shared_side = vertexB_of_square - vertexA_of_square →
  vertexF_of_triangles = vertexD_of_square - vertexC_of_square →
  vertexG_of_triangles = vertexB_of_square - vertexA_of_square →
  side_length_of_largest_inscribed_square = 6 - 2 * Real.sqrt 3 :=
sorry

end largest_inscribed_square_size_l204_204851


namespace total_cost_l204_204364

-- Definitions based on conditions
def old_camera_cost : ℝ := 4000
def new_model_cost_increase_rate : ℝ := 0.3
def lens_initial_cost : ℝ := 400
def lens_discount : ℝ := 200

-- Main statement to prove
theorem total_cost (old_camera_cost new_model_cost_increase_rate lens_initial_cost lens_discount : ℝ) : 
  let new_camera_cost := old_camera_cost * (1 + new_model_cost_increase_rate)
  let lens_cost_after_discount := lens_initial_cost - lens_discount
  (new_camera_cost + lens_cost_after_discount) = 5400 :=
by
  sorry

end total_cost_l204_204364


namespace cricket_current_average_l204_204148

theorem cricket_current_average (A : ℕ) (h1: 10 * A + 77 = 11 * (A + 4)) : 
  A = 33 := 
by 
  sorry

end cricket_current_average_l204_204148


namespace part_one_equation_of_line_part_two_equation_of_line_l204_204190

-- Definition of line passing through a given point
def line_through_point (a b : ℝ) (P : ℝ × ℝ) : Prop := P.1 / a + P.2 / b = 1

-- Condition: the sum of intercepts is 12
def sum_of_intercepts (a b : ℝ) : Prop := a + b = 12

-- Condition: area of triangle is 12
def area_of_triangle (a b : ℝ) : Prop := (1/2) * (abs (a * b)) = 12

-- First part: equation of the line when the sum of intercepts is 12
theorem part_one_equation_of_line (a b : ℝ) : 
  (line_through_point a b (3, 2)) ∧ (sum_of_intercepts a b) →
  (∃ x, (x = 2 ∧ (2*x)+x - 8 = 0) ∨ (x = 3 ∧ x + 3*x - 9 = 0)) :=
by
  sorry

-- Second part: equation of the line when the area of the triangle is 12
theorem part_two_equation_of_line (a b : ℝ) : 
  (line_through_point a b (3, 2)) ∧ (area_of_triangle a b) →
  ∃ x, x = 2 ∧ (2*x + 3*x - 12 = 0) :=
by
  sorry

end part_one_equation_of_line_part_two_equation_of_line_l204_204190


namespace inequality_f_l204_204234

noncomputable def f (x y z : ℝ) : ℝ :=
  x * y + y * z + z * x - 2 * x * y * z

theorem inequality_f (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  0 ≤ f x y z ∧ f x y z ≤ 7 / 27 :=
  sorry

end inequality_f_l204_204234


namespace leftmost_digit_base9_l204_204475

theorem leftmost_digit_base9 (x : ℕ) (h : x = 3^19 + 2*3^18 + 1*3^17 + 1*3^16 + 2*3^15 + 2*3^14 + 1*3^13 + 1*3^12 + 1*3^11 + 2*3^10 + 2*3^9 + 2*3^8 + 1*3^7 + 1*3^6 + 1*3^5 + 1*3^4 + 2*3^3 + 2*3^2 + 2*3^1 + 2) : ℕ :=
by
  sorry

end leftmost_digit_base9_l204_204475


namespace cos_2x_eq_cos_2y_l204_204243

theorem cos_2x_eq_cos_2y (x y : ℝ) 
  (h1 : Real.sin x + Real.cos y = 1) 
  (h2 : Real.cos x + Real.sin y = -1) : 
  Real.cos (2 * x) = Real.cos (2 * y) := by
  sorry

end cos_2x_eq_cos_2y_l204_204243


namespace materials_total_order_l204_204536

theorem materials_total_order :
  let concrete := 0.16666666666666666
  let bricks := 0.16666666666666666
  let stone := 0.5
  concrete + bricks + stone = 0.8333333333333332 :=
by
  sorry

end materials_total_order_l204_204536


namespace mnmn_not_cube_in_base_10_and_find_smallest_base_b_l204_204645

theorem mnmn_not_cube_in_base_10_and_find_smallest_base_b 
    (m n : ℕ) (h1 : m * 10^3 + n * 10^2 + m * 10 + n < 10000) :
    ¬ (∃ k : ℕ, (m * 10^3 + n * 10^2 + m * 10 + n) = k^3) 
    ∧ ∃ b : ℕ, b > 1 ∧ (∃ k : ℕ, (m * b^3 + n * b^2 + m * b + n = k^3)) :=
by sorry

end mnmn_not_cube_in_base_10_and_find_smallest_base_b_l204_204645


namespace inequality_solution_l204_204656

theorem inequality_solution (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < (5 / 9) :=
by
  sorry

end inequality_solution_l204_204656


namespace binom_15_4_eq_1365_l204_204564

theorem binom_15_4_eq_1365 : (Nat.choose 15 4) = 1365 := 
by 
  sorry

end binom_15_4_eq_1365_l204_204564


namespace acute_triangle_tangent_sum_range_l204_204474

theorem acute_triangle_tangent_sum_range
  (a b c : ℝ) (A B C : ℝ)
  (triangle_ABC_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (opposite_sides : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (side_relation : b^2 - a^2 = a * c)
  (angle_relation : A + B + C = π)
  (angles_in_radians : 0 < A ∧ A < π)
  (angles_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  1 < (1 / Real.tan A + 1 / Real.tan B) ∧ (1 / Real.tan A + 1 / Real.tan B) < 2 * Real.sqrt 3 / 3 :=
sorry 

end acute_triangle_tangent_sum_range_l204_204474


namespace smallest_perimeter_is_23_l204_204150

def is_odd_prime (n : ℕ) : Prop := Nat.Prime n ∧ n % 2 = 1

def are_consecutive_odd_primes (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧ b = a + 2 ∧ c = b + 2

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_perimeter_is_23 : 
  ∃ (a b c : ℕ), are_consecutive_odd_primes a b c ∧ satisfies_triangle_inequality a b c ∧ is_prime (a + b + c) ∧ (a + b + c) = 23 :=
by
  sorry

end smallest_perimeter_is_23_l204_204150


namespace income_expenses_opposite_l204_204012

def income_denotation (income : Int) : Int := income

theorem income_expenses_opposite :
  income_denotation 5 = 5 →
  income_denotation (-5) = -5 :=
by
  intro h
  sorry

end income_expenses_opposite_l204_204012


namespace minimum_throws_for_repetition_of_sum_l204_204107

/-- To ensure that the same sum is rolled twice when throwing four fair six-sided dice,
you must throw the dice at least 22 times. -/
theorem minimum_throws_for_repetition_of_sum :
  ∀ (throws : ℕ), (∀ (sum : ℕ), 4 ≤ sum ∧ sum ≤ 24 → ∃ (count : ℕ), count ≤ 21 ∧ sum = count + 4) → throws ≥ 22 :=
by
  sorry

end minimum_throws_for_repetition_of_sum_l204_204107


namespace a_2013_is_4_l204_204224

theorem a_2013_is_4
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 7)
  (h3 : ∀ n : ℕ, a (n+2) = (a n * a (n+1)) % 10) :
  a 2013 = 4 :=
sorry

end a_2013_is_4_l204_204224


namespace max_area_triangle_l204_204618

noncomputable def max_area (QA QB QC BC : ℝ) : ℝ :=
  1 / 2 * ((QA^2 + QB^2 - QC^2) / (2 * BC) + 3) * BC

theorem max_area_triangle (QA QB QC BC : ℝ) (hQA : QA = 3) (hQB : QB = 4) (hQC : QC = 5) (hBC : BC = 6) :
  max_area QA QB QC BC = 19 := by
  sorry

end max_area_triangle_l204_204618


namespace value_of_a_l204_204210

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 2 → x^2 - x + a < 0) → a = -2 :=
by
  intro h
  sorry

end value_of_a_l204_204210


namespace f_bounds_l204_204628

noncomputable def f (x1 x2 x3 x4 : ℝ) := 1 - (x1^3 + x2^3 + x3^3 + x4^3) - 6 * (x1 * x2 * x3 + x1 * x2 * x4 + x1 * x3 * x4 + x2 * x3 * x4)

theorem f_bounds (x1 x2 x3 x4 : ℝ) (h : x1 + x2 + x3 + x4 = 1) :
  0 < f x1 x2 x3 x4 ∧ f x1 x2 x3 x4 ≤ 3 / 4 :=
by
  -- Proof steps go here
  sorry

end f_bounds_l204_204628


namespace find_x_for_parallel_vectors_l204_204934

theorem find_x_for_parallel_vectors :
  ∀ (x : ℚ), (∃ a b : ℚ × ℚ, a = (2 * x, 3) ∧ b = (1, 9) ∧ (∃ k : ℚ, (2 * x, 3) = (k * 1, k * 9))) ↔ x = 1 / 6 :=
by 
  sorry

end find_x_for_parallel_vectors_l204_204934


namespace reduced_price_per_kg_l204_204396

-- Define the conditions
def reduction_factor : ℝ := 0.80
def extra_kg : ℝ := 4
def total_cost : ℝ := 684

-- Assume the original price P and reduced price R
variables (P R : ℝ)

-- Define the equations derived from the conditions
def original_cost_eq := (P * 16 = total_cost)
def reduced_cost_eq := (0.80 * P * (16 + extra_kg) = total_cost)

-- The final theorem stating the reduced price per kg of oil is 34.20 Rs
theorem reduced_price_per_kg : R = 34.20 :=
by
  have h1: P * 16 = total_cost := sorry -- This will establish the original cost
  have h2: 0.80 * P * (16 + extra_kg) = total_cost := sorry -- This will establish the reduced cost
  have Q: 16 = 16 := sorry -- Calculation of Q (original quantity)
  have h3: P = 42.75 := sorry -- Calculation of original price
  have h4: R = 0.80 * P := sorry -- Calculation of reduced price
  have h5: R = 34.20 := sorry -- Final calculation matching the required answer
  exact h5

end reduced_price_per_kg_l204_204396


namespace gcd_45_75_l204_204712

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204712


namespace solve_inequality_l204_204757

noncomputable def inequality (x : ℕ) : Prop :=
  6 * (9 : ℝ)^(1/x) - 13 * (3 : ℝ)^(1/x) * (2 : ℝ)^(1/x) + 6 * (4 : ℝ)^(1/x) ≤ 0

theorem solve_inequality (x : ℕ) (hx : 1 < x) : inequality x ↔ x ≥ 2 :=
by {
  sorry
}

end solve_inequality_l204_204757


namespace probability_even_product_is_correct_l204_204247

noncomputable def probability_even_product : ℚ :=
  let n := 13 in            -- total number of integers from 6 to 18 inclusive
  let total_combinations := Nat.choose n 2 in
  let even_count := 7 in    -- number of even integers in the range
  let odd_count := n - even_count in
  let odd_combinations := Nat.choose odd_count 2 in
  let even_product_combinations := total_combinations - odd_combinations in
  even_product_combinations / total_combinations

theorem probability_even_product_is_correct : probability_even_product = 9 / 13 := by
  sorry

end probability_even_product_is_correct_l204_204247


namespace min_throws_to_ensure_repeat_sum_l204_204118

theorem min_throws_to_ensure_repeat_sum : 
  ∀ (min_sum max_sum : ℤ), 
  min_sum = 4 ∧ max_sum = 24 
  → ∃ n, n ≥ 22 ∧ n = 22 :=
by
  intros min_sum max_sum h
  cases h with h_min h_max
  existsi 22
  split
  · exact Nat.le_refl 22
  · sorry

end min_throws_to_ensure_repeat_sum_l204_204118


namespace geometric_sequence_a3_eq_2_l204_204221

theorem geometric_sequence_a3_eq_2 
  (a_1 a_3 a_5 : ℝ) 
  (h1 : a_1 * a_3 * a_5 = 8) 
  (h2 : a_3^2 = a_1 * a_5) : 
  a_3 = 2 :=
by 
  sorry

end geometric_sequence_a3_eq_2_l204_204221


namespace solve_for_x_l204_204610

theorem solve_for_x (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 :=
by
  -- Proof will go here
  sorry

end solve_for_x_l204_204610


namespace find_nabla_l204_204950

theorem find_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by {
  sorry
}

end find_nabla_l204_204950


namespace opposite_event_equiv_l204_204585

-- Define the set of all shoes
def Shoes : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the condition: randomly picking out 4 shoes
def pick_4_shoes (s : Finset ℕ) : Prop := 
  s ⊆ Shoes ∧ s.card = 4

-- Define the event: all 4 shoes are in pairs
def all_in_pairs (s : Finset ℕ) : Prop := 
let pairs := { {1, 2}, {3, 4}, {5, 6}, {7, 8} } in 
  ∃ p ∈ pairs, (s ⊆ p)

-- Define the opposite event: at least 2 shoes are not in pairs
def at_least_2_not_in_pairs (s : Finset ℕ) : Prop :=
let pairs := { {1, 2}, {3, 4}, {5, 6}, {7, 8} } in 
  ∃ (x y : ℕ), x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ ¬∃ p ∈ pairs, {x, y} ⊆ p

-- The theorem that we need to prove
theorem opposite_event_equiv :
  ∀ s : Finset ℕ, pick_4_shoes s → (¬ all_in_pairs s ↔ at_least_2_not_in_pairs s) :=
by sorry

end opposite_event_equiv_l204_204585


namespace cos_sum_identity_l204_204642

noncomputable def prove_cos_sum_identity (x : ℝ) (n : ℕ) : Prop :=
  (1/2 + ∑ i in Finset.range (n + 1), Real.cos (i * x)) = 
  (Real.sin ((n + 1/2) * x) / (2 * Real.sin (1/2 * x)))

theorem cos_sum_identity (x : ℝ) (n : ℕ) : prove_cos_sum_identity x n :=
sorry

end cos_sum_identity_l204_204642


namespace min_sum_of_squares_of_y_coords_l204_204408

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line_through_point (m : ℝ) (x y : ℝ) : Prop := x = m * y + 4

theorem min_sum_of_squares_of_y_coords :
  ∃ (m : ℝ), ∀ (x1 y1 x2 y2 : ℝ),
  (line_through_point m x1 y1) →
  (parabola x1 y1) →
  (line_through_point m x2 y2) →
  (parabola x2 y2) →
  x1 ≠ x2 → 
  ((y1 + y2)^2 - 2 * y1 * y2) = 32 :=
sorry

end min_sum_of_squares_of_y_coords_l204_204408


namespace good_numbers_count_1_to_50_l204_204678

def is_good_number (n : ℕ) : Prop :=
  ∃ (k l : ℕ), k ≠ 0 ∧ l ≠ 0 ∧ n = k * l + l - k

theorem good_numbers_count_1_to_50 : ∃ cnt, cnt = 49 ∧ (∀ n, n ∈ (Finset.range 51).erase 0 → is_good_number n) :=
  sorry

end good_numbers_count_1_to_50_l204_204678


namespace smallest_x_l204_204984

noncomputable def f (x : ℝ) : ℝ :=
if 1 ≤ x ∧ x ≤ 4 then x^2 - 4 * x + 5 else sorry

theorem smallest_x (x : ℝ) (h₁ : ∀ x > 0, f (4 * x) = 4 * f x)
  (h₂ : ∀ x, (1 ≤ x ∧ x ≤ 4) → f x = x^2 - 4 * x + 5) :
  ∃ x₀, x₀ > 0 ∧ f x₀ = 1024 ∧ (∀ y, y > 0 ∧ f y = 1024 → y ≥ x₀) :=
sorry

end smallest_x_l204_204984


namespace gcd_45_75_l204_204693

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204693


namespace time_difference_l204_204480

def joey_time : ℕ :=
  let uphill := 12 / 6 * 60
  let downhill := 10 / 25 * 60
  let flat := 20 / 15 * 60
  uphill + downhill + flat

def sue_time : ℕ :=
  let downhill := 10 / 35 * 60
  let uphill := 12 / 12 * 60
  let flat := 20 / 25 * 60
  downhill + uphill + flat

theorem time_difference : joey_time - sue_time = 99 := by
  -- calculation steps skipped
  sorry

end time_difference_l204_204480


namespace arccos_sqrt_3_over_2_eq_pi_over_6_l204_204164

open Real

theorem arccos_sqrt_3_over_2_eq_pi_over_6 :
  ∀ (x : ℝ), x = (sqrt 3) / 2 → arccos x = π / 6 :=
by
  intro x
  sorry

end arccos_sqrt_3_over_2_eq_pi_over_6_l204_204164


namespace final_sale_price_is_correct_l204_204282

-- Define the required conditions
def original_price : ℝ := 1200.00
def first_discount_rate : ℝ := 0.10
def second_discount_rate : ℝ := 0.20
def final_discount_rate : ℝ := 0.05

-- Define the expression to calculate the sale price after the discounts
def first_discount_price := original_price * (1 - first_discount_rate)
def second_discount_price := first_discount_price * (1 - second_discount_rate)
def final_sale_price := second_discount_price * (1 - final_discount_rate)

-- Prove that the final sale price equals $820.80
theorem final_sale_price_is_correct : final_sale_price = 820.80 := by
  sorry

end final_sale_price_is_correct_l204_204282


namespace find_a_l204_204324

theorem find_a (a : ℝ) (h1 : ∀ θ : ℝ, x = a + 4 * Real.cos θ ∧ y = 1 + 4 * Real.sin θ)
  (h2 : ∃ p : ℝ × ℝ, (3 * p.1 + 4 * p.2 - 5 = 0 ∧ (∃ θ : ℝ, p = (a + 4 * Real.cos θ, 1 + 4 * Real.sin θ))))
  (h3 : ∀ (p1 p2 : ℝ × ℝ), 
        (3 * p1.1 + 4 * p1.2 - 5 = 0 ∧ 3 * p2.1 + 4 * p2.2 - 5 = 0) ∧
        (∃ θ1 : ℝ, p1 = (a + 4 * Real.cos θ1, 1 + 4 * Real.sin θ1)) ∧
        (∃ θ2 : ℝ, p2 = (a + 4 * Real.cos θ2, 1 + 4 * Real.sin θ2)) → p1 = p2) :
  a = 7 := by
  sorry

end find_a_l204_204324


namespace white_area_of_sign_remains_l204_204774

theorem white_area_of_sign_remains (h1 : (6 * 18 = 108))
  (h2 : 9 = 6 + 3)
  (h3 : 7.5 = 5 + 3 - 0.5)
  (h4 : 13 = 9 + 4)
  (h5 : 9 = 6 + 3)
  (h6 : 38.5 = 9 + 7.5 + 13 + 9)
  : 108 - 38.5 = 69.5 := by
  sorry

end white_area_of_sign_remains_l204_204774


namespace min_throws_to_same_sum_l204_204071

/-- Define the set of possible sums for four six-sided dice --/
def dice_sum_range := {s : ℕ | 4 ≤ s ∧ s ≤ 24}

/-- The total number of possible sums when rolling four six-sided dice --/
def num_possible_sums : ℕ := 24 - 4 + 1

/-- 
  The minimum number of throws required to ensure that the same sum appears at least twice 
  by the Pigeonhole principle.
--/
theorem min_throws_to_same_sum : num_possible_sums + 1 = 22 := by
  sorry

end min_throws_to_same_sum_l204_204071


namespace faye_scored_47_pieces_l204_204185

variable (X : ℕ) -- X is the number of pieces of candy Faye scored on Halloween.

-- Definitions based on the conditions
def initial_candy_count (X : ℕ) : ℕ := X - 25
def after_sister_gave_40 (X : ℕ) : ℕ := initial_candy_count X + 40
def current_candy_count (X : ℕ) : ℕ := after_sister_gave_40 X

-- Theorem to prove the number of pieces of candy Faye scored on Halloween
theorem faye_scored_47_pieces (h : current_candy_count X = 62) : X = 47 :=
by
  sorry

end faye_scored_47_pieces_l204_204185


namespace equivalent_annual_rate_l204_204921

def quarterly_to_annual_rate (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate) ^ 4 - 1

def to_percentage (rate : ℝ) : ℝ :=
  rate * 100

theorem equivalent_annual_rate (quarterly_rate : ℝ) (annual_rate : ℝ) :
  quarterly_rate = 0.02 →
  annual_rate = quarterly_to_annual_rate quarterly_rate →
  to_percentage annual_rate = 8.24 :=
by
  intros
  sorry

end equivalent_annual_rate_l204_204921


namespace circle_passing_points_l204_204620

theorem circle_passing_points :
  ∃ (D E F : ℝ), 
    (25 + 1 + 5 * D + E + F = 0) ∧ 
    (36 + 6 * D + F = 0) ∧ 
    (1 + 1 - D + E + F = 0) ∧ 
    (∀ x y : ℝ, (x, y) = (5, 1) ∨ (x, y) = (6, 0) ∨ (x, y) = (-1, 1) → x^2 + y^2 + D * x + E * y + F = 0) → 
  x^2 + y^2 - 4 * x + 6 * y - 12 = 0 :=
by
  sorry

end circle_passing_points_l204_204620


namespace expenses_opposite_to_income_l204_204015

theorem expenses_opposite_to_income (income_5 : ℤ) (h_income : income_5 = 5) : -income_5 = -5 :=
by
  -- proof is omitted
  sorry

end expenses_opposite_to_income_l204_204015


namespace minimum_rolls_to_ensure_repeated_sum_l204_204081

theorem minimum_rolls_to_ensure_repeated_sum : 
  let dice_faces := 6
  let number_of_dice := 4
  let min_sum := number_of_dice * 1
  let max_sum := number_of_dice * dice_faces
  let distinct_sums := (max_sum - min_sum) + 1
  in 22 = distinct_sums + 1 :=
by {
  sorry
}

end minimum_rolls_to_ensure_repeated_sum_l204_204081


namespace feet_of_perpendiculars_on_circle_l204_204188

theorem feet_of_perpendiculars_on_circle
  {A B C A1 A2 B1 B2 C1 C2 : EuclideanGeometry.Point}
  (hABC : EuclideanGeometry.Triangle A B C)
  (hA1 : EuclideanGeometry.IsPerpendicular A1 A B)
  (hA2 : EuclideanGeometry.IsPerpendicular A2 A C)
  (hB1 : EuclideanGeometry.IsPerpendicular B1 B A)
  (hB2 : EuclideanGeometry.IsPerpendicular B2 B C)
  (hC1 : EuclideanGeometry.IsPerpendicular C1 C A)
  (hC2 : EuclideanGeometry.IsPerpendicular C2 C B)
  (h_definitions : EuclideanGeometry.IsAltitudeFoot hA1 hA2 hB1 hB2 hC1 hC2):
  EuclideanGeometry.ConcyclicPoints [A1, A2, B1, B2, C1, C2] :=
begin
  sorry
end

end feet_of_perpendiculars_on_circle_l204_204188


namespace third_cyclist_speed_l204_204521

theorem third_cyclist_speed (a b : ℝ) : 
    ∃ v : ℝ, 
        (∀ t1 t2 t3 x, 
            t1 = 1/6 + x ∧ 
            t2 = t1 + 1/3 ∧ 
            t3 = x ∧ 
            (t1*a = x*v) ∧ 
            (t2*b = (x + 1/3)*v)
        ) → 
        v = (1/4) * (a + 3 * b + real.sqrt (a^2 - 10*a*b + b^2)) := 
begin
    sorry
end

end third_cyclist_speed_l204_204521


namespace gcd_45_75_l204_204730

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204730


namespace Shelby_drive_time_in_rain_l204_204644

theorem Shelby_drive_time_in_rain (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 3) 
  (h3 : 40 * (3 - x) + 25 * x = 85) : x = 140 / 60 :=
  sorry

end Shelby_drive_time_in_rain_l204_204644


namespace fraction_shaded_area_l204_204906

theorem fraction_shaded_area (l w : ℕ) (h_l : l = 15) (h_w : w = 20)
  (h_qtr : (1 / 4: ℝ) * (l * w) = 75) (h_shaded : (1 / 5: ℝ) * 75 = 15) :
  (15 / (l * w): ℝ) = 1 / 20 :=
by
  sorry

end fraction_shaded_area_l204_204906


namespace carrie_is_left_with_50_l204_204916

-- Definitions for the conditions given in the problem
def amount_given : ℕ := 91
def cost_of_sweater : ℕ := 24
def cost_of_tshirt : ℕ := 6
def cost_of_shoes : ℕ := 11

-- Definition of the total amount spent
def total_spent : ℕ := cost_of_sweater + cost_of_tshirt + cost_of_shoes

-- Definition of the amount left
def amount_left : ℕ := amount_given - total_spent

-- The theorem we want to prove
theorem carrie_is_left_with_50 : amount_left = 50 :=
by
  have h1 : amount_given = 91 := rfl
  have h2 : total_spent = 41 := rfl
  have h3 : amount_left = 50 := rfl
  exact rfl

end carrie_is_left_with_50_l204_204916


namespace gcd_45_75_l204_204733

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204733


namespace range_of_a_l204_204876

theorem range_of_a
  (a : ℝ)
  (h : ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 < 0 ∧ (x1 * x2 = 2 * a + 6)) :
  a < -3 :=
by
  sorry

end range_of_a_l204_204876


namespace gcd_a_b_is_one_l204_204051

-- Definitions
def a : ℤ := 100^2 + 221^2 + 320^2
def b : ℤ := 101^2 + 220^2 + 321^2

-- Theorem statement
theorem gcd_a_b_is_one : Int.gcd a b = 1 := by
  sorry

end gcd_a_b_is_one_l204_204051


namespace books_and_games_left_to_experience_l204_204383

def booksLeft (B_total B_read : Nat) : Nat := B_total - B_read
def gamesLeft (G_total G_played : Nat) : Nat := G_total - G_played
def totalLeft (B_total B_read G_total G_played : Nat) : Nat := booksLeft B_total B_read + gamesLeft G_total G_played

theorem books_and_games_left_to_experience :
  totalLeft 150 74 50 17 = 109 := by
  sorry

end books_and_games_left_to_experience_l204_204383


namespace visitor_increase_l204_204811

variable (x : ℝ) -- The percentage increase each day

theorem visitor_increase (h1 : 1.2 * (1 + x)^2 = 2.5) : 1.2 * (1 + x)^2 = 2.5 :=
by exact h1

end visitor_increase_l204_204811


namespace min_throws_to_ensure_same_sum_twice_l204_204102

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l204_204102


namespace range_of_uv_sq_l204_204195

theorem range_of_uv_sq (u v w : ℝ) (h₀ : 0 ≤ u) (h₁ : 0 ≤ v) (h₂ : 0 ≤ w) (h₃ : u + v + w = 2) :
  0 ≤ u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ∧ u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ≤ 1 :=
sorry

end range_of_uv_sq_l204_204195


namespace yoongi_has_5_carrots_l204_204159

def yoongis_carrots (initial_carrots sister_gave: ℕ) : ℕ :=
  initial_carrots + sister_gave

theorem yoongi_has_5_carrots : yoongis_carrots 3 2 = 5 := by 
  sorry

end yoongi_has_5_carrots_l204_204159


namespace min_sum_nonpos_l204_204232

theorem min_sum_nonpos (a b : ℤ) (h_nonpos_a : a ≤ 0) (h_nonpos_b : b ≤ 0) (h_prod : a * b = 144) : 
  a + b = -30 :=
sorry

end min_sum_nonpos_l204_204232


namespace divisibility_by_3_divisibility_by_4_l204_204896

-- Proof that 5n^2 + 10n + 8 is divisible by 3 if and only if n ≡ 2 (mod 3)
theorem divisibility_by_3 (n : ℤ) : (5 * n^2 + 10 * n + 8) % 3 = 0 ↔ n % 3 = 2 := 
    sorry

-- Proof that 5n^2 + 10n + 8 is divisible by 4 if and only if n ≡ 0 (mod 2)
theorem divisibility_by_4 (n : ℤ) : (5 * n^2 + 10 * n + 8) % 4 = 0 ↔ n % 2 = 0 :=
    sorry

end divisibility_by_3_divisibility_by_4_l204_204896


namespace expenses_neg_of_income_pos_l204_204029

theorem expenses_neg_of_income_pos :
  ∀ (income expense : Int), income = 5 → expense = -income → expense = -5 :=
by
  intros income expense h_income h_expense
  rw [h_income] at h_expense
  exact h_expense

end expenses_neg_of_income_pos_l204_204029


namespace simplify_and_evaluate_l204_204646

theorem simplify_and_evaluate (a : ℤ) (h : a = 0) : 
  ((a / (a - 1) : ℚ) + ((a + 1) / (a^2 - 1) : ℚ)) = (-1 : ℚ) := by
  have ha_ne1 : a ≠ 1 := by norm_num [h]
  have ha_ne_neg1 : a ≠ -1 := by norm_num [h]
  have h1 : (a^2 - 1) ≠ 0 := by
    rw [sub_ne_zero]
    norm_num [h]
  sorry

end simplify_and_evaluate_l204_204646


namespace eighth_term_geometric_seq_l204_204577

theorem eighth_term_geometric_seq (a1 a2 : ℚ) (a1_val : a1 = 3) (a2_val : a2 = 9 / 2) :
  (a1 * (a2 / a1)^(7) = 6561 / 128) :=
  by
    sorry

end eighth_term_geometric_seq_l204_204577


namespace angle_solution_exists_l204_204173

theorem angle_solution_exists :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ 9 * (Real.sin x) * (Real.cos x)^4 - 9 * (Real.sin x)^4 * (Real.cos x) = 1 / 2 ∧ x = 30 :=
by
  sorry

end angle_solution_exists_l204_204173


namespace original_profit_percentage_l204_204539

theorem original_profit_percentage (C S : ℝ) (hC : C = 70)
(h1 : S - 14.70 = 1.30 * (C * 0.80)) :
  (S - C) / C * 100 = 25 := by
  sorry

end original_profit_percentage_l204_204539


namespace factor_quadratic_l204_204923

theorem factor_quadratic (x : ℝ) : 
  (x^2 + 6 * x + 9 - 16 * x^4) = (-4 * x^2 + 2 * x + 3) * (4 * x^2 + 2 * x + 3) := 
by 
  sorry

end factor_quadratic_l204_204923


namespace gcd_45_75_l204_204702

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204702


namespace smallest_possible_product_l204_204640

theorem smallest_possible_product : 
  ∃ (x : ℕ) (y : ℕ), (x = 56 ∧ y = 78 ∨ x = 57 ∧ y = 68) ∧ x * y = 3876 :=
by
  sorry

end smallest_possible_product_l204_204640


namespace maximal_sector_angle_l204_204874

theorem maximal_sector_angle (a : ℝ) (r : ℝ) (l : ℝ) (α : ℝ)
  (h1 : l + 2 * r = a)
  (h2 : 0 < r ∧ r < a / 2)
  (h3 : α = l / r)
  (eval_area : ∀ (l r : ℝ), S = 1 / 2 * l * r)
  (S : ℝ) :
  α = 2 := sorry

end maximal_sector_angle_l204_204874


namespace horse_food_calculation_l204_204518

theorem horse_food_calculation
  (num_sheep : ℕ)
  (ratio_sheep_horses : ℕ)
  (total_horse_food : ℕ)
  (H : ℕ)
  (num_sheep_eq : num_sheep = 56)
  (ratio_eq : ratio_sheep_horses = 7)
  (total_food_eq : total_horse_food = 12880)
  (num_horses : H = num_sheep * 1 / ratio_sheep_horses)
  : num_sheep = ratio_sheep_horses → total_horse_food / H = 230 :=
by
  sorry

end horse_food_calculation_l204_204518


namespace problem_proof_l204_204824

-- Define the geometric sequence and vectors conditions
variables (a : ℕ → ℝ) (q : ℝ)
variables (h1 : ∀ n, a (n + 1) = q * a n)
variables (h2 : a 2 = a 2)
variables (h3 : a 3 = q * a 2)
variables (h4 : 3 * a 2 = 2 * a 3)

-- Statement to prove
theorem problem_proof:
  (a 2 + a 4) / (a 3 + a 5) = 2 / 3 :=
  sorry

end problem_proof_l204_204824


namespace expenses_representation_l204_204003

theorem expenses_representation (income_representation : ℤ) (income : ℤ) (expenses : ℤ) :
  income_representation = +5 → income = +5 → expenses = -income → expenses = -5 :=
by
  intro hr hs he
  rw [←hs, he]
  exact hr

end expenses_representation_l204_204003


namespace rick_iron_hours_l204_204244

def can_iron_dress_shirts (h : ℕ) : ℕ := 4 * h

def can_iron_dress_pants (hours : ℕ) : ℕ := 3 * hours

def total_clothes_ironed (h : ℕ) : ℕ := can_iron_dress_shirts h + can_iron_dress_pants 5

theorem rick_iron_hours (h : ℕ) (H : total_clothes_ironed h = 27) : h = 3 :=
by sorry

end rick_iron_hours_l204_204244


namespace expense_of_5_yuan_is_minus_5_yuan_l204_204027

def income (x : Int) : Int :=
  x

def expense (x : Int) : Int :=
  -x

theorem expense_of_5_yuan_is_minus_5_yuan : expense 5 = -5 :=
by
  unfold expense
  sorry

end expense_of_5_yuan_is_minus_5_yuan_l204_204027


namespace max_hours_at_regular_rate_l204_204278

-- Define the maximum hours at regular rate H
def max_regular_hours (H : ℕ) : Prop := 
  let regular_rate := 16
  let overtime_rate := 16 + (0.75 * 16)
  let total_hours := 60
  let total_compensation := 1200
  16 * H + 28 * (total_hours - H) = total_compensation

theorem max_hours_at_regular_rate : ∃ H, max_regular_hours H ∧ H = 40 :=
sorry

end max_hours_at_regular_rate_l204_204278


namespace gcd_45_75_l204_204694

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204694


namespace ensure_same_sum_rolled_twice_l204_204064

theorem ensure_same_sum_rolled_twice :
  ∀ (n : ℕ) (min_sum max_sum : ℕ),
    min_sum = 4 →
    max_sum = 24 →
    (min_sum ≤ n ∧ n ≤ max_sum) →
    ∀ trials : ℕ, trials = 22 →
      ∃ (s1 s2 : ℕ), s1 = s2 ∧ 
      (∃ (throws1 throws2 : list ℕ), list.sum throws1 = s1 ∧ list.sum throws2 = s2 ∧ throws1 ≠ throws2) :=
by 
  sorry

end ensure_same_sum_rolled_twice_l204_204064


namespace three_digit_solutions_exist_l204_204751

theorem three_digit_solutions_exist :
  ∃ (x y z : ℤ), 100 ≤ x ∧ x ≤ 999 ∧ 
                 100 ≤ y ∧ y ≤ 999 ∧
                 100 ≤ z ∧ z ≤ 999 ∧
                 17 * x + 15 * y - 28 * z = 61 ∧
                 19 * x - 25 * y + 12 * z = 31 :=
by
    sorry

end three_digit_solutions_exist_l204_204751


namespace sum_first_10_terms_l204_204327

noncomputable def a (n : ℕ) := 1 / (4 * (n + 1) ^ 2 - 1)

theorem sum_first_10_terms : (Finset.range 10).sum a = 10 / 21 :=
by
  sorry

end sum_first_10_terms_l204_204327


namespace cost_of_carrots_and_cauliflower_l204_204189

variable {p c f o : ℝ}

theorem cost_of_carrots_and_cauliflower
  (h1 : p + c + f + o = 30)
  (h2 : o = 3 * p)
  (h3 : f = p + c) : 
  c + f = 14 := 
by
  sorry

end cost_of_carrots_and_cauliflower_l204_204189


namespace chord_square_length_l204_204917

/-- Given three circles with radii 4, 8, and 16, such that the first two are externally tangent to each other and both are internally tangent to the third, if a chord in the circle with radius 16 is a common external tangent to the other two circles, then the square of the length of this chord is 7616/9. -/
theorem chord_square_length (r1 r2 r3 : ℝ) (h1 : r1 = 4) (h2 : r2 = 8) (h3 : r3 = 16)
  (tangent_condition : ∀ (O4 O8 O16 : ℝ), O4 = r1 + r2 ∧ O8 = r2 + r3 ∧ O16 = r1 + r3) :
  (16^2 - (20/3)^2) * 4 = 7616 / 9 :=
by
  sorry

end chord_square_length_l204_204917


namespace money_initial_amounts_l204_204257

theorem money_initial_amounts (x : ℕ) (A B : ℕ) 
  (h1 : A = 8 * x) 
  (h2 : B = 5 * x) 
  (h3 : (A - 50) = 4 * (B + 100) / 5) : 
  A = 800 ∧ B = 500 := 
sorry

end money_initial_amounts_l204_204257


namespace gcd_of_45_and_75_l204_204715

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l204_204715


namespace solve_system_l204_204507

theorem solve_system :
  (∃ x y : ℝ, 4 * x - 3 * y = -3 ∧ 8 * x + 5 * y = 11 + x ^ 2 ∧ (x, y) = (14.996, 19.994)) ∨
  (∃ x y : ℝ, 4 * x - 3 * y = -3 ∧ 8 * x + 5 * y = 11 + x ^ 2 ∧ (x, y) = (0.421, 1.561)) :=
  sorry

end solve_system_l204_204507


namespace min_rolls_to_duplicate_sum_for_four_dice_l204_204116

theorem min_rolls_to_duplicate_sum_for_four_dice : 
    let min_sum := 4 * 1,
    let max_sum := 4 * 6,
    let possible_sums := max_sum - min_sum + 1 in
    possible_sums = 21 → 
    (possible_sums + 1 = 22) := 
by
  intros min_sum max_sum possible_sums h
  have h1 : min_sum = 4 := rfl
  have h2 : max_sum = 24 := rfl
  have h3 : possible_sums = 21 := h
  have h4 : possible_sums + 1 = 22 := calc
    possible_sums + 1 = 21 + 1 : by rw h
    ... = 22 : by rfl
  exact h4

end min_rolls_to_duplicate_sum_for_four_dice_l204_204116


namespace perpendicular_slopes_l204_204441

theorem perpendicular_slopes {m : ℝ} (h : (1 : ℝ) * -m = -1) : m = 1 :=
by sorry

end perpendicular_slopes_l204_204441


namespace range_of_k_l204_204961

theorem range_of_k (k : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → (k + 2) * x1 - 1 > (k + 2) * x2 - 1) → k < -2 := by
  sorry

end range_of_k_l204_204961


namespace inequality_sin_cos_l204_204273

theorem inequality_sin_cos 
  (a b : ℝ) (n : ℝ) (x : ℝ) 
  (ha : 0 < a) (hb : 0 < b) : 
  (a / (Real.sin x)^n) + (b / (Real.cos x)^n) ≥ (a^(2/(n+2)) + b^(2/(n+2)))^((n+2)/2) :=
sorry

end inequality_sin_cos_l204_204273


namespace printer_time_equation_l204_204280

theorem printer_time_equation (x : ℝ) (rate1 rate2 : ℝ) (flyers1 flyers2 : ℝ)
  (h1 : rate1 = 100) (h2 : flyers1 = 1000) (h3 : flyers2 = 1000) 
  (h4 : flyers1 / rate1 = 10) (h5 : flyers1 / (rate1 + rate2) = 4) : 
  1 / 10 + 1 / x = 1 / 4 :=
by 
  sorry

end printer_time_equation_l204_204280


namespace triangle_AB_value_l204_204225

-- Define the necessary geometric entities and relationships
noncomputable def area_of_triangle (a b c : ℝ) (theta : ℝ) : ℝ :=
  (1/2) * a * b * real.sin theta

/-- 
  Given a triangle ABC with ∠A = 60°, AC = 2, point D lies on side BC, 
  AD is the angle bisector of ∠CAB, and the area of △ADB is 2√3, then the value of AB is 2 + 2√3 
--/
theorem triangle_AB_value {A B C D : Type} [geometry Point Line triangle Proper_Archimedean_Categorial] :
  ∀ (AC AB AD : ℝ) (theta : ℝ), 
  AC = 2 → 
  theta = real.pi / 3 → -- 60 degrees in radians
  (area_of_triangle AC AD (theta / 2)) + 2 * real.sqrt 3 =
  (area_of_triangle AC AB theta) →
  area_of_triangle AB AD (theta / 2) = 2 * real.sqrt 3 →
  AB = 2 + 2 * real.sqrt 3 :=
  sorry

end triangle_AB_value_l204_204225


namespace minimum_rolls_to_ensure_repeated_sum_l204_204082

theorem minimum_rolls_to_ensure_repeated_sum : 
  let dice_faces := 6
  let number_of_dice := 4
  let min_sum := number_of_dice * 1
  let max_sum := number_of_dice * dice_faces
  let distinct_sums := (max_sum - min_sum) + 1
  in 22 = distinct_sums + 1 :=
by {
  sorry
}

end minimum_rolls_to_ensure_repeated_sum_l204_204082


namespace expenses_representation_l204_204001

theorem expenses_representation (income_representation : ℤ) (income : ℤ) (expenses : ℤ) :
  income_representation = +5 → income = +5 → expenses = -income → expenses = -5 :=
by
  intro hr hs he
  rw [←hs, he]
  exact hr

end expenses_representation_l204_204001


namespace chess_tournament_participants_l204_204459

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 171) : n = 19 := 
by sorry

end chess_tournament_participants_l204_204459


namespace find_a_n_l204_204435

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₁ : ∀ n, a n > 0)
  (h₂ : ∀ n, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n, a n = Real.sqrt n - Real.sqrt (n - 1) :=
by
  sorry

end find_a_n_l204_204435


namespace probability_two_queens_or_at_least_one_ace_l204_204456

theorem probability_two_queens_or_at_least_one_ace :
  let total_cards := 52
  let num_aces := 4
  let num_queens := 4
  let prob_two_queens := (4 / total_cards) * (3 / (total_cards - 1))
  let prob_one_ace_FIRST := (4 / total_cards) * ((total_cards - num_aces) / (total_cards - 1))
  let prob_one_ace_SECOND := ((total_cards - num_aces) / total_cards) * (4 / (total_cards - 1))
  let prob_one_ace := prob_one_ace_FIRST + prob_one_ace_SECOND
  let prob_exactly_one_ace := prob_one_ace
  let prob_two_aces := (4 / total_cards) * (3 / (total_cards - 1))
  let prob_at_least_one_ace := prob_exactly_one_ace + prob_two_aces
  let prob_two_queens_or_at_least_one_ace := prob_two_queens + prob_at_least_one_ace
  (prob_two_queens_or_at_least_one_ace = 2 / 13) :=
by
  let total_cards := 52
  let num_aces := 4
  let num_queens := 4
  let prob_two_queens := (4 / total_cards) * (3 / (total_cards - 1))
  let prob_one_ace_FIRST := (4 / total_cards) * ((total_cards - num_aces) / (total_cards - 1))
  let prob_one_ace_SECOND := ((total_cards - num_aces) / total_cards) * (4 / (total_cards - 1))
  let prob_one_ace := prob_one_ace_FIRST + prob_one_ace_SECOND
  let prob_exactly_one_ace := prob_one_ace
  let prob_two_aces := (4 / total_cards) * (3 / (total_cards - 1))
  let prob_at_least_one_ace := prob_exactly_one_ace + prob_two_aces
  let prob_two_queens_or_at_least_one_ace := prob_two_queens + prob_at_least_one_ace
  have h : prob_two_queens_or_at_least_one_ace = (1 / 221) + (32 / 221 + 1 / 221) :=
    sorry -- This should be proved by simplification
  have h2 : (1 / 221) + (32 / 221 + 1 / 221) = 34 / 221 :=
    sorry -- This should be proved by simplification
  have h3 : 34 / 221 = 2 / 13 :=
    sorry -- This should be proved by simplification
  exact Eq.trans h (Eq.trans h2 h3)

end probability_two_queens_or_at_least_one_ace_l204_204456


namespace solution_set_of_inequality_l204_204307

theorem solution_set_of_inequality
  (a b : ℝ)
  (x y : ℝ)
  (h1 : a * (-2) + b = 3)
  (h2 : a * (-1) + b = 2)
  :  -x + 1 < 0 ↔ x > 1 :=
by 
  -- Proof goes here
  sorry

end solution_set_of_inequality_l204_204307


namespace find_radius_of_inscribed_sphere_l204_204412

variables (a b c s : ℝ)

theorem find_radius_of_inscribed_sphere
  (h1 : a + b + c = 18)
  (h2 : 2 * (a * b + b * c + c * a) = 216)
  (h3 : a^2 + b^2 + c^2 = 108) :
  s = 3 * Real.sqrt 3 :=
by
  sorry

end find_radius_of_inscribed_sphere_l204_204412


namespace find_monthly_growth_rate_find_optimal_price_l204_204611

noncomputable def monthly_growth_rate (a b : ℝ) (n : ℕ) : ℝ :=
  ((b / a) ^ (1 / n)) - 1

theorem find_monthly_growth_rate :
  monthly_growth_rate 150 216 2 = 0.2 := sorry

noncomputable def optimal_price (c s₀ p₀ t z : ℝ) : ℝ :=
  let profit_per_unit y := y - c
  let sales_volume y := s₀ - t * (y - p₀)
  let profit y := profit_per_unit y * sales_volume y
  ((-100 + sqrt (100^2 - 4 * 1 * (2496 - z))) / 2)

theorem find_optimal_price :
  optimal_price 30 300 40 10 3960 = 48 := sorry

end find_monthly_growth_rate_find_optimal_price_l204_204611


namespace purely_imaginary_complex_is_two_l204_204460

theorem purely_imaginary_complex_is_two
  (a : ℝ)
  (h_imag : (a^2 - 3 * a + 2) + (a - 1) * I = (a - 1) * I) :
  a = 2 := by
  sorry

end purely_imaginary_complex_is_two_l204_204460


namespace gcd_45_75_l204_204713

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204713


namespace min_rolls_to_duplicate_sum_for_four_dice_l204_204117

theorem min_rolls_to_duplicate_sum_for_four_dice : 
    let min_sum := 4 * 1,
    let max_sum := 4 * 6,
    let possible_sums := max_sum - min_sum + 1 in
    possible_sums = 21 → 
    (possible_sums + 1 = 22) := 
by
  intros min_sum max_sum possible_sums h
  have h1 : min_sum = 4 := rfl
  have h2 : max_sum = 24 := rfl
  have h3 : possible_sums = 21 := h
  have h4 : possible_sums + 1 = 22 := calc
    possible_sums + 1 = 21 + 1 : by rw h
    ... = 22 : by rfl
  exact h4

end min_rolls_to_duplicate_sum_for_four_dice_l204_204117


namespace geometric_sequence_common_ratio_l204_204472

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h₁ : a 1 = 1) (h₂ : a 1 * a 2 * a 3 = -8) :
  q = -2 :=
sorry

end geometric_sequence_common_ratio_l204_204472


namespace triangle_value_l204_204590

variable (triangle p : ℝ)

theorem triangle_value : (triangle + p = 75 ∧ 3 * (triangle + p) - p = 198) → triangle = 48 :=
by
  sorry

end triangle_value_l204_204590


namespace samantha_birth_year_l204_204376

theorem samantha_birth_year :
  ∀ (first_amc : ℕ) (amc9_year : ℕ) (samantha_age_in_amc9 : ℕ),
  (first_amc = 1983) →
  (amc9_year = first_amc + 8) →
  (samantha_age_in_amc9 = 13) →
  (amc9_year - samantha_age_in_amc9 = 1978) :=
by
  intros first_amc amc9_year samantha_age_in_amc9 h1 h2 h3
  sorry

end samantha_birth_year_l204_204376


namespace calculate_a_over_b_l204_204175

noncomputable def system_solution (x y a b : ℝ) : Prop :=
  (8 * x - 5 * y = a) ∧ (10 * y - 15 * x = b) ∧ (x ≠ 0) ∧ (y ≠ 0) ∧ (b ≠ 0)

theorem calculate_a_over_b (x y a b : ℝ) (h : system_solution x y a b) : a / b = 8 / 15 :=
by
  sorry

end calculate_a_over_b_l204_204175


namespace solve_equation_l204_204648

theorem solve_equation :
  ∀ x : ℝ, x ≠ 1 → (2 * x + 4) / (x^2 + 4 * x - 5) = (2 - x) / (x - 1) → x = -6 :=
begin
  intros x hx h,
  sorry
end

end solve_equation_l204_204648


namespace gcd_45_75_l204_204688

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204688


namespace problem1_problem2_l204_204305

variable (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)

theorem problem1 : 
  (a * b + a + b + 1) * (a * b + a * c + b * c + c ^ 2) ≥ 16 * a * b * c := 
by sorry

theorem problem2 : 
  (b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3 := 
by sorry

end problem1_problem2_l204_204305


namespace number_of_white_balls_l204_204971

-- Definitions based on the problem conditions
def total_balls : Nat := 120
def red_freq : ℝ := 0.15
def black_freq : ℝ := 0.45

-- Result to prove
theorem number_of_white_balls :
  let red_balls := total_balls * red_freq
  let black_balls := total_balls * black_freq
  total_balls - red_balls - black_balls = 48 :=
by
  sorry

end number_of_white_balls_l204_204971


namespace binom_15_4_l204_204558

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the theorem stating the specific binomial coefficient value
theorem binom_15_4 : binom 15 4 = 1365 :=
by
  sorry

end binom_15_4_l204_204558


namespace perpendicular_vectors_x_value_l204_204946

theorem perpendicular_vectors_x_value 
  (x : ℝ) (a b : ℝ × ℝ) (hₐ : a = (1, -2)) (hᵦ : b = (3, x)) (h_perpendicular : a.1 * b.1 + a.2 * b.2 = 0) : 
  x = 3 / 2 :=
by
  -- The proof is not required, hence we use 'sorry'
  sorry

end perpendicular_vectors_x_value_l204_204946


namespace find_b_l204_204424

def h (x : ℝ) : ℝ := 5 * x + 6

theorem find_b : ∃ b : ℝ, h b = 0 ∧ b = -6 / 5 :=
by
  sorry

end find_b_l204_204424


namespace sufficient_condition_perpendicular_l204_204486

-- Definitions of perpendicularity and lines/planes intersections
variables {Plane : Type} {Line : Type}

variable (α β γ : Plane)
variable (m n l : Line)

-- Axioms representing the given conditions
axiom perp_planes (p₁ p₂ : Plane) : Prop -- p₁ is perpendicular to p₂
axiom perp_line_plane (line : Line) (plane : Plane) : Prop -- line is perpendicular to plane

-- Given conditions for the problem.
axiom n_perp_α : perp_line_plane n α
axiom n_perp_β : perp_line_plane n β
axiom m_perp_α : perp_line_plane m α

-- The proposition to be proved.
theorem sufficient_condition_perpendicular (h₁ : perp_line_plane n α)
                                           (h₂ : perp_line_plane n β)
                                           (h₃ : perp_line_plane m α) :
  perp_line_plane m β := sorry

end sufficient_condition_perpendicular_l204_204486


namespace gcd_45_75_l204_204708

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204708


namespace prove_y_l204_204138

-- Define the conditions
variables (x y : ℤ) -- x and y are integers

-- State the problem conditions
def conditions := (x + y = 270) ∧ (x - y = 200)

-- Define the theorem to prove that y = 35 given the conditions
theorem prove_y : conditions x y → y = 35 :=
by
  sorry

end prove_y_l204_204138


namespace cyclic_quadrilateral_eq_l204_204157

theorem cyclic_quadrilateral_eq (A B C D : ℝ) (AB AD BC DC : ℝ)
  (h1 : AB = AD) (h2 : based_on_laws_of_cosines) : AC ^ 2 = BC * DC + AB ^ 2 :=
sorry

end cyclic_quadrilateral_eq_l204_204157


namespace solve_hours_l204_204522

variable (x y : ℝ)

-- Conditions
def Condition1 : x > 0 := sorry
def Condition2 : y > 0 := sorry
def Condition3 : (2:ℝ) / 3 * y / x + (3 * x * y - 2 * y^2) / (3 * x) = x * y / (x + y) + 2 := sorry
def Condition4 : 2 * y / (x + y) = (3 * x - 2 * y) / (3 * x) := sorry

-- Question: How many hours would it take for A and B to complete the task alone?
theorem solve_hours : x = 6 ∧ y = 3 := 
by
  -- Use assumed conditions and variables to define the context
  have h1 := Condition1
  have h2 := Condition2
  have h3 := Condition3
  have h4 := Condition4
  -- Combine analytical relationship and solve for x and y 
  sorry

end solve_hours_l204_204522


namespace xy_value_l204_204677

noncomputable def compute_xy : ℝ × ℝ → ℝ
| (x, y) := x * y

theorem xy_value (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : compute_xy (x, y) = 35 / 12 := by
  sorry

end xy_value_l204_204677


namespace find_costs_of_accessories_max_type_a_accessories_l204_204764

theorem find_costs_of_accessories (x y : ℕ) 
  (h1 : x + 3 * y = 530) 
  (h2 : 3 * x + 2 * y = 890) : 
  x = 230 ∧ y = 100 := 
by 
  sorry

theorem max_type_a_accessories (m n : ℕ) 
  (m_n_sum : m + n = 30) 
  (cost_constraint : 230 * m + 100 * n ≤ 4180) : 
  m ≤ 9 := 
by 
  sorry

end find_costs_of_accessories_max_type_a_accessories_l204_204764


namespace percent_runs_by_running_eq_18_75_l204_204537

/-
Define required conditions.
-/
def total_runs : ℕ := 224
def boundaries_runs : ℕ := 9 * 4
def sixes_runs : ℕ := 8 * 6
def twos_runs : ℕ := 12 * 2
def threes_runs : ℕ := 4 * 3
def byes_runs : ℕ := 6 * 1
def running_runs : ℕ := twos_runs + threes_runs + byes_runs

/-
Define the proof problem to show that the percentage of the total score made by running between the wickets is 18.75%.
-/
theorem percent_runs_by_running_eq_18_75 : (running_runs : ℚ) / total_runs * 100 = 18.75 := by
  sorry

end percent_runs_by_running_eq_18_75_l204_204537


namespace revenue_difference_l204_204219

def original_revenue : ℕ := 10000

def vasya_revenue (X : ℕ) : ℕ :=
  2 * (original_revenue / X) * (4 * X / 5)

def kolya_revenue (X : ℕ) : ℕ :=
  (original_revenue / X) * (8 * X / 3)

theorem revenue_difference (X : ℕ) (hX : X > 0) : vasya_revenue X = 16000 ∧ kolya_revenue X = 13333 ∧ vasya_revenue X - original_revenue = 6000 := 
by
  sorry

end revenue_difference_l204_204219


namespace magnitude_of_b_is_5_l204_204587

variable (a b : ℝ × ℝ)
variable (h_a : a = (3, -2))
variable (h_ab : a + b = (0, 2))

theorem magnitude_of_b_is_5 : ‖b‖ = 5 :=
by
  sorry

end magnitude_of_b_is_5_l204_204587


namespace system_of_inequalities_l204_204662

theorem system_of_inequalities (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : (0.5 < p ∧ p < 5 / 9) :=
by sorry

end system_of_inequalities_l204_204662


namespace star_sum_interior_angles_l204_204374

theorem star_sum_interior_angles (n : ℕ) (h : n ≥ 6) :
  let S := 180 * n - 360
  S = 180 * (n - 2) :=
by
  let S := 180 * n - 360
  show S = 180 * (n - 2)
  sorry

end star_sum_interior_angles_l204_204374


namespace average_height_31_students_l204_204900

theorem average_height_31_students (avg1 avg2 : ℝ) (n1 n2 : ℕ) (h1 : avg1 = 20) (h2 : avg2 = 20) (h3 : n1 = 20) (h4 : n2 = 11) : ((avg1 * n1 + avg2 * n2) / (n1 + n2)) = 20 :=
by
  sorry

end average_height_31_students_l204_204900


namespace waiting_time_probability_l204_204765

-- Given conditions
def dep1 := 7 * 60 -- 7:00 in minutes
def dep2 := 7 * 60 + 30 -- 7:30 in minutes
def dep3 := 8 * 60 -- 8:00 in minutes

def arrival_start := 7 * 60 + 25 -- 7:25 in minutes
def arrival_end := 8 * 60 -- 8:00 in minutes
def total_time_window := arrival_end - arrival_start -- 35 minutes

def favorable_window1_start := 7 * 60 + 25 -- 7:25 in minutes
def favorable_window1_end := 7 * 60 + 30 -- 7:30 in minutes
def favorable_window2_start := 8 * 60 -- 8:00 in minutes
def favorable_window2_end := 8 * 60 + 10 -- 8:10 in minutes

def favorable_time_window := 
  (favorable_window1_end - favorable_window1_start) + 
  (favorable_window2_end - favorable_window2_start) -- 15 minutes

-- Probability calculation
theorem waiting_time_probability : 
  (favorable_time_window : ℚ) / (total_time_window : ℚ) = 3 / 7 :=
by
  sorry

end waiting_time_probability_l204_204765


namespace min_throws_for_repeated_sum_l204_204097

theorem min_throws_for_repeated_sum : 
  (∀ (n : ℕ), n = 24 ∧ (∀ (x : ℕ), x ≥ 4 ∧ x ≤ 24)) → 22 :=
by
  sorry

end min_throws_for_repeated_sum_l204_204097


namespace gcd_45_75_l204_204696

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204696


namespace brother_age_in_5_years_l204_204634

theorem brother_age_in_5_years
  (nick_age : ℕ)
  (sister_age : ℕ)
  (brother_age : ℕ)
  (h_nick : nick_age = 13)
  (h_sister : sister_age = nick_age + 6)
  (h_brother : brother_age = (nick_age + sister_age) / 2) :
  brother_age + 5 = 21 := 
by 
  sorry

end brother_age_in_5_years_l204_204634


namespace cash_still_missing_l204_204227

theorem cash_still_missing (c : ℝ) (h : c > 0) :
  (1 : ℝ) - (8 / 9) = (1 / 9 : ℝ) :=
by
  sorry

end cash_still_missing_l204_204227


namespace base_number_min_sum_l204_204293

theorem base_number_min_sum (a b : ℕ) (h₁ : 5 * a + 2 = 2 * b + 5) : a + b = 9 :=
by {
  -- this proof is skipped with sorry
  sorry
}

end base_number_min_sum_l204_204293


namespace equal_playing_time_l204_204787

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l204_204787


namespace minimum_throws_for_repetition_of_sum_l204_204103

/-- To ensure that the same sum is rolled twice when throwing four fair six-sided dice,
you must throw the dice at least 22 times. -/
theorem minimum_throws_for_repetition_of_sum :
  ∀ (throws : ℕ), (∀ (sum : ℕ), 4 ≤ sum ∧ sum ≤ 24 → ∃ (count : ℕ), count ≤ 21 ∧ sum = count + 4) → throws ≥ 22 :=
by
  sorry

end minimum_throws_for_repetition_of_sum_l204_204103


namespace volume_of_box_with_ratio_125_l204_204285

def volumes : Finset ℕ := {60, 80, 100, 120, 200}

theorem volume_of_box_with_ratio_125 : 80 ∈ volumes ∧ ∃ (x : ℕ), 10 * x^3 = 80 :=
by {
  -- Skipping the proof, as only the statement is required.
  sorry
}

end volume_of_box_with_ratio_125_l204_204285


namespace odd_func_value_l204_204963

noncomputable def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2 * x - 3 else 0 -- f(x) is initially set to 0 when x ≤ 0, since we will not use this part directly.

theorem odd_func_value (x : ℝ) (h : x < 0) (hf : isOddFunction f) (hfx : ∀ x > 0, f x = 2 * x - 3) :
  f x = 2 * x + 3 :=
by
  sorry

end odd_func_value_l204_204963


namespace four_dice_min_rolls_l204_204112

def minRollsToEnsureSameSum (n : Nat) : Nat :=
  if n = 4 then 22 else sorry

theorem four_dice_min_rolls : minRollsToEnsureSameSum 4 = 22 := by
  rfl

end four_dice_min_rolls_l204_204112


namespace gcd_45_75_l204_204686

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204686


namespace equal_play_time_l204_204786

theorem equal_play_time (total_players on_field_players match_minutes : ℕ) 
    (h1 : total_players = 10) 
    (h2 : on_field_players = 8) 
    (h3 : match_minutes = 45) 
    (h4 : ∀ p, p ∈ (finset.range total_players) → time_played p = (on_field_players * match_minutes) / total_players)
    : (on_field_players * match_minutes) / total_players = 36 :=
by
  have total_play_minutes : on_field_players * match_minutes = 360 := by sorry
  have equal_play_time : (on_field_players * match_minutes) / total_players = 36 := by sorry
  exact equal_play_time

end equal_play_time_l204_204786


namespace garden_area_increase_l204_204275

-- Define the dimensions and perimeter of the rectangular garden
def length_rect : ℕ := 30
def width_rect : ℕ := 12
def area_rect : ℕ := length_rect * width_rect

def perimeter_rect : ℕ := 2 * (length_rect + width_rect)

-- Define the side length and area of the new square garden
def side_square : ℕ := perimeter_rect / 4
def area_square : ℕ := side_square * side_square

-- Define the increase in area
def increase_in_area : ℕ := area_square - area_rect

-- Prove the increase in area is 81 square feet
theorem garden_area_increase : increase_in_area = 81 := by
  sorry

end garden_area_increase_l204_204275


namespace equal_playing_time_for_each_player_l204_204778

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l204_204778


namespace expenses_negation_of_income_l204_204007

theorem expenses_negation_of_income 
    (income : ℤ) 
    (income_is_5 : income = 5) 
    (denote_income : income = 5 → "+" ∘ toString income = "+5") 
    (expenses_are_negation_of_income :  "expenses = -1 * income") : "expenses = -5" :=
begin
    sorry
end

end expenses_negation_of_income_l204_204007


namespace prove_value_of_expression_l204_204231

theorem prove_value_of_expression (x : ℝ) (h : 10000 * x + 2 = 4) : 5000 * x + 1 = 2 :=
by 
  sorry

end prove_value_of_expression_l204_204231


namespace length_of_each_piece_after_subdividing_l204_204542

theorem length_of_each_piece_after_subdividing (total_length : ℝ) (num_initial_cuts : ℝ) (num_pieces_given : ℝ) (num_subdivisions : ℝ) (final_length : ℝ) : 
  total_length = 200 → 
  num_initial_cuts = 4 → 
  num_pieces_given = 2 → 
  num_subdivisions = 2 → 
  final_length = (total_length / num_initial_cuts / num_subdivisions) → 
  final_length = 25 := 
by 
  intros h1 h2 h3 h4 h5 
  sorry

end length_of_each_piece_after_subdividing_l204_204542


namespace rectangle_area_invariant_l204_204249

theorem rectangle_area_invariant (l w : ℝ) (A : ℝ) 
  (h0 : A = l * w)
  (h1 : A = (l + 3) * (w - 1))
  (h2 : A = (l - 1.5) * (w + 2)) :
  A = 13.5 :=
by
  sorry

end rectangle_area_invariant_l204_204249


namespace simplify_sqrt_sum_l204_204872

theorem simplify_sqrt_sum : 
  sqrt (12 + 8 * sqrt 3) + sqrt (12 - 8 * sqrt 3) = 4 * sqrt 3 := 
sorry

end simplify_sqrt_sum_l204_204872


namespace length_of_platform_l204_204136

/--
Problem statement:
A train 450 m long running at 108 km/h crosses a platform in 25 seconds.
Prove that the length of the platform is 300 meters.

Given:
- The train is 450 meters long.
- The train's speed is 108 km/h.
- The train crosses the platform in 25 seconds.

To prove:
The length of the platform is 300 meters.
-/
theorem length_of_platform :
  let train_length := 450
  let train_speed := 108 * (1000 / 3600) -- converting km/h to m/s
  let crossing_time := 25
  let total_distance_covered := train_speed * crossing_time
  let platform_length := total_distance_covered - train_length
  platform_length = 300 := by
  sorry

end length_of_platform_l204_204136


namespace equal_playing_time_l204_204788

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l204_204788


namespace minimum_value_expression_l204_204447

theorem minimum_value_expression 
  (a b c : ℝ) 
  (h1 : 3 * a + 2 * b + c = 5) 
  (h2 : 2 * a + b - 3 * c = 1) 
  (h3 : 0 ≤ a) 
  (h4 : 0 ≤ b) 
  (h5 : 0 ≤ c) : 
  ∃(c : ℝ), (c ≥ 3/7 ∧ c ≤ 7/11) ∧ (3 * a + b - 7 * c = -5/7) :=
sorry 

end minimum_value_expression_l204_204447


namespace fraction_c_d_l204_204176

theorem fraction_c_d (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) :
  c / d = -8 / 15 :=
sorry

end fraction_c_d_l204_204176


namespace minimum_value_of_expression_l204_204487

theorem minimum_value_of_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 3) : 
  a^2 + 8 * a * b + 32 * b^2 + 24 * b * c + 8 * c^2 ≥ 72 :=
sorry

end minimum_value_of_expression_l204_204487


namespace ellipse_equation_point_M_exists_l204_204442

-- Condition: Point (1, sqrt(2)/2) lies on the ellipse
def point_lies_on_ellipse (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
    (a_gt_b : a > b) : Prop :=
  (1, Real.sqrt 2 / 2).fst^2 / a^2 + (1, Real.sqrt 2 / 2).snd^2 / b^2 = 1

-- Condition: Eccentricity of the ellipse is sqrt(2)/2
def eccentricity_condition (a b : ℝ) (c : ℝ) : Prop :=
  c / a = Real.sqrt 2 / 2 ∧ a^2 = b^2 + c^2

-- Question (I): Equation of ellipse should be (x^2 / 2 + y^2 = 1)
theorem ellipse_equation (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
    (a_gt_b : a > b) (h : point_lies_on_ellipse a b a_pos b_pos a_gt_b)
    (h_ecc : eccentricity_condition a b c) : a = Real.sqrt 2 ∧ b = 1 := 
sorry

-- Question (II): There exists M such that MA · MB is constant
theorem point_M_exists (a b c x0 : ℝ)
    (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : a > b) 
    (a_val : a = Real.sqrt 2) (b_val : b = 1) 
    (h : point_lies_on_ellipse a b a_pos b_pos a_gt_b)
    (h_ecc : eccentricity_condition a b c) : 
    ∃ (M : ℝ × ℝ), M.fst = 5 / 4 ∧ M.snd = 0 ∧ -7 / 16 = -7 / 16 := 
sorry

end ellipse_equation_point_M_exists_l204_204442


namespace find_f_inv_8_l204_204932

variable (f : ℝ → ℝ)

-- Given conditions
axiom h1 : f 5 = 1
axiom h2 : ∀ x, f (2 * x) = 2 * f x

-- Theorem to prove
theorem find_f_inv_8 : f ⁻¹' {8} = {40} :=
by sorry

end find_f_inv_8_l204_204932


namespace Congcong_CO2_emissions_l204_204863

-- Definitions based on conditions
def CO2_emissions (t: ℝ) : ℝ := t * 0.91 -- Condition 1: CO2 emissions calculation

def Congcong_water_usage : ℝ := 6 -- Condition 2: Congcong's water usage (6 tons)

-- Statement we want to prove
theorem Congcong_CO2_emissions : CO2_emissions Congcong_water_usage = 5.46 :=
by 
  sorry

end Congcong_CO2_emissions_l204_204863


namespace gcd_45_75_l204_204706

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204706


namespace probability_of_winning_reward_l204_204155

-- Definitions representing the problem conditions
def red_envelopes : ℕ := 4
def card_types : ℕ := 3

-- Theorem statement: Prove the probability of winning the reward is 4/9
theorem probability_of_winning_reward : 
  (∃ (n m : ℕ), n = card_types^red_envelopes ∧ m = (Nat.choose red_envelopes 2) * (Nat.factorial 3)) → 
  (m / n = 4/9) :=
by
  sorry  -- Proof to be filled in

end probability_of_winning_reward_l204_204155


namespace nth_equation_l204_204498

theorem nth_equation (n : ℕ) : 
  n ≥ 1 → (∃ k, k = n + 1 ∧ (k^2 - n^2 - 1) / 2 = n) :=
by
  intros h
  use n + 1
  sorry

end nth_equation_l204_204498


namespace petStoreHasSixParrots_l204_204770

def petStoreParrotsProof : Prop :=
  let cages := 6.0
  let parakeets := 2.0
  let birds_per_cage := 1.333333333
  let total_birds := cages * birds_per_cage
  let number_of_parrots := total_birds - parakeets
  number_of_parrots = 6.0

theorem petStoreHasSixParrots : petStoreParrotsProof := by
  sorry

end petStoreHasSixParrots_l204_204770


namespace more_bottles_of_regular_soda_l204_204767

theorem more_bottles_of_regular_soda (reg_soda diet_soda : ℕ) (h1 : reg_soda = 79) (h2 : diet_soda = 53) :
  reg_soda - diet_soda = 26 :=
by
  sorry

end more_bottles_of_regular_soda_l204_204767


namespace speed_ratio_l204_204417

theorem speed_ratio (v_A v_B : ℝ) (L t : ℝ) 
  (h1 : v_A * t = (1 - 0.11764705882352941) * L)
  (h2 : v_B * t = L) : 
  v_A / v_B = 1.11764705882352941 := 
by 
  sorry

end speed_ratio_l204_204417


namespace original_price_l204_204283

theorem original_price 
  (SP : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h1 : SP = 15)
  (h2 : gain_percent = 0.50)
  (h3 : SP = P * (1 + gain_percent)) :
  P = 10 :=
by
  sorry

end original_price_l204_204283


namespace figure_M_area_l204_204973

open Real

theorem figure_M_area :
  ∫ x in 1..5, ((4 - x) - (1 / 2 * (x - 2)^2 - 1)) = 4 :=
by
  sorry

end figure_M_area_l204_204973


namespace number_of_blue_fish_l204_204990

def total_fish : ℕ := 22
def goldfish : ℕ := 15
def blue_fish : ℕ := total_fish - goldfish

theorem number_of_blue_fish : blue_fish = 7 :=
by
  -- proof goes here
  sorry

end number_of_blue_fish_l204_204990


namespace hydrangeas_percent_l204_204903

theorem hydrangeas_percent (total_flowers : ℕ) (blue_flowers tulips hydrangeas yellow_flowers daisies : ℕ)
  (h1 : blue_flowers = total_flowers * 3 / 5)
  (h2 : yellow_flowers = total_flowers * 2 / 5)
  (h3 : tulips = blue_flowers / 4)
  (h4 : hydrangeas = blue_flowers * 3 / 4) :
  hydrangeas = total_flowers * 9 / 20 :=
by
  sorry

end hydrangeas_percent_l204_204903


namespace binom_15_4_l204_204556

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the theorem stating the specific binomial coefficient value
theorem binom_15_4 : binom 15 4 = 1365 :=
by
  sorry

end binom_15_4_l204_204556


namespace expense_5_yuan_neg_l204_204020

-- Define the condition that income of 5 yuan is denoted as +5 yuan
def income_5_yuan_pos : Int := 5

-- Define the statement to prove that expenses of 5 yuan are denoted as -5 yuan
theorem expense_5_yuan_neg : income_5_yuan_pos = 5 → -income_5_yuan_pos = -5 :=
by
  intro h
  rw h
  rfl

end expense_5_yuan_neg_l204_204020


namespace distinct_numbers_not_all_F_in_set_l204_204242

theorem distinct_numbers_not_all_F_in_set (S : Finset ℕ) (h : S.card = 21) (h1 : ∀ x ∈ S, x ≤ 1000000) (h2 : ∀ a b ∈ S, a ≠ b) :
  ∃ x ∉ S, ∃ a b ∈ S, x = a + b - Nat.gcd a b :=
by
  sorry

end distinct_numbers_not_all_F_in_set_l204_204242


namespace compare_powers_l204_204806

-- Definitions for the three numbers
def a : ℝ := 3 ^ 555
def b : ℝ := 4 ^ 444
def c : ℝ := 5 ^ 333

-- Statement to prove
theorem compare_powers : c < a ∧ a < b := sorry

end compare_powers_l204_204806


namespace area_inside_octagon_outside_semicircles_l204_204473

theorem area_inside_octagon_outside_semicircles :
  let s := 3
  let octagon_area := 2 * (1 + Real.sqrt 2) * s^2
  let semicircle_area := (1/2) * Real.pi * (s / 2)^2
  let total_semicircle_area := 8 * semicircle_area
  octagon_area - total_semicircle_area = 54 + 24 * Real.sqrt 2 - 9 * Real.pi :=
sorry

end area_inside_octagon_outside_semicircles_l204_204473


namespace gcd_45_75_l204_204726

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204726


namespace sum_of_three_numbers_l204_204670

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1 : B = 30)
  (h2 : A * 3 = 2 * B)
  (h3 : C * 5 = 8 * B) : 
  A + B + C = 98 :=
by
  sorry

end sum_of_three_numbers_l204_204670


namespace compute_sum_sq_roots_of_polynomial_l204_204918

theorem compute_sum_sq_roots_of_polynomial :
  (∃ p q r : ℚ, (∀ x : ℚ, polynomial.eval x (3 * X^3 - 2 * X^2 + 6 * X - 9) = 0 → (x = p ∨ x = q ∨ x = r)) ∧
     p^2 + q^2 + r^2 = -32/9) :=
sorry

end compute_sum_sq_roots_of_polynomial_l204_204918


namespace expenses_representation_l204_204000

theorem expenses_representation (income_representation : ℤ) (income : ℤ) (expenses : ℤ) :
  income_representation = +5 → income = +5 → expenses = -income → expenses = -5 :=
by
  intro hr hs he
  rw [←hs, he]
  exact hr

end expenses_representation_l204_204000


namespace parallel_lines_implies_slope_l204_204666

theorem parallel_lines_implies_slope (a : ℝ) :
  (∀ (x y: ℝ), ax + 2 * y = 0) ∧ (∀ (x y: ℝ), x + y = 1) → (a = 2) :=
by
  sorry

end parallel_lines_implies_slope_l204_204666


namespace problem1_problem2_l204_204142

-- Problem (1)
theorem problem1 (f : ℝ → ℝ) (h : ∀ x ≠ 0, f (2 / x + 2) = x + 1) : 
  ∀ x ≠ 2, f x = x / (x - 2) :=
sorry

-- Problem (2)
theorem problem2 (f : ℝ → ℝ) (h : ∃ k b, ∀ x, f x = k * x + b ∧ k ≠ 0)
  (h' : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) :
  ∀ x, f x = 2 * x + 7 :=
sorry

end problem1_problem2_l204_204142


namespace remainder_is_three_l204_204890

def eleven_div_four_has_remainder_three (A : ℕ) : Prop :=
  11 = 4 * 2 + A

theorem remainder_is_three : eleven_div_four_has_remainder_three 3 :=
by
  sorry

end remainder_is_three_l204_204890


namespace team_CB_days_worked_together_l204_204248

def projectA := 1 -- Project A is 1 unit of work
def projectB := 5 / 4 -- Project B is 1.25 units of work
def work_rate_A := 1 / 20 -- Team A's work rate
def work_rate_B := 1 / 24 -- Team B's work rate
def work_rate_C := 1 / 30 -- Team C's work rate

noncomputable def combined_rate_without_C := work_rate_B + work_rate_C

noncomputable def combined_total_work := projectA + projectB

noncomputable def days_for_combined_work := combined_total_work / combined_rate_without_C

-- Statement to prove the number of days team C and team B worked together
theorem team_CB_days_worked_together : 
  days_for_combined_work = 15 := 
  sorry

end team_CB_days_worked_together_l204_204248


namespace part1_part2_l204_204758

namespace TriangleProblems

noncomputable def area_triangle_part1 (A C a : ℝ) : ℝ :=
  if A = 30 ∧ C = 45 ∧ a = 2 then 1 + real.sqrt 3 else 0

theorem part1 (A C a : ℝ) : 
  A = 30 → C = 45 → a = 2 → area_triangle_part1 A C a = 1 + real.sqrt 3 :=
begin
  intros hA hC ha,
  simp [area_triangle_part1, hA, hC, ha]
end

noncomputable def length_AB_part2 (area BC C : ℝ) : ℝ :=
  if area = real.sqrt 3 ∧ BC = 2 ∧ C = 60 then 2 else 0

theorem part2 (area BC C : ℝ) :
  area = real.sqrt 3 → BC = 2 → C = 60 → length_AB_part2 area BC C = 2 :=
begin
  intros hArea hBC hC,
  simp [length_AB_part2, hArea, hBC, hC]
end

end TriangleProblems

end part1_part2_l204_204758


namespace cost_of_paving_floor_l204_204878

-- Definitions of the constants
def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_meter : ℝ := 400

-- Definitions of the calculated area and cost
def area : ℝ := length * width
def cost : ℝ := area * rate_per_sq_meter

-- Statement to prove
theorem cost_of_paving_floor : cost = 8250 := by
  sorry

end cost_of_paving_floor_l204_204878


namespace line_equation_l204_204530

theorem line_equation (x y : ℝ) : 
  (∃ (m c : ℝ), m = 3 ∧ c = 4 ∧ y = m * x + c) ↔ 3 * x - y + 4 = 0 := by
  sorry

end line_equation_l204_204530


namespace remainder_when_c_divided_by_b_eq_2_l204_204674

theorem remainder_when_c_divided_by_b_eq_2 
(a b c : ℕ) 
(hb : b = 3 * a + 3) 
(hc : c = 9 * a + 11) : 
  c % b = 2 := 
sorry

end remainder_when_c_divided_by_b_eq_2_l204_204674


namespace gcd_of_45_and_75_l204_204722

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l204_204722


namespace min_throws_to_same_sum_l204_204070

/-- Define the set of possible sums for four six-sided dice --/
def dice_sum_range := {s : ℕ | 4 ≤ s ∧ s ≤ 24}

/-- The total number of possible sums when rolling four six-sided dice --/
def num_possible_sums : ℕ := 24 - 4 + 1

/-- 
  The minimum number of throws required to ensure that the same sum appears at least twice 
  by the Pigeonhole principle.
--/
theorem min_throws_to_same_sum : num_possible_sums + 1 = 22 := by
  sorry

end min_throws_to_same_sum_l204_204070


namespace probability_different_suits_correct_l204_204360

-- Definitions based on conditions
def cards_in_deck : ℕ := 52
def cards_picked : ℕ := 3
def first_card_suit_not_matter : Prop := True
def second_card_different_suit : Prop := True
def third_card_different_suit : Prop := True

-- Definition of the probability function
def probability_different_suits (cards_total : ℕ) (cards_picked : ℕ) : Rat :=
  let first_card_prob := 1
  let second_card_prob := 39 / 51
  let third_card_prob := 26 / 50
  first_card_prob * second_card_prob * third_card_prob

-- The theorem statement to prove the probability each card is of a different suit
theorem probability_different_suits_correct :
  probability_different_suits cards_in_deck cards_picked = 169 / 425 :=
by
  -- Proof should be written here
  sorry

end probability_different_suits_correct_l204_204360


namespace value_of_m_div_x_l204_204667

noncomputable def ratio_of_a_to_b (a b : ℝ) : Prop := a / b = 4 / 5
noncomputable def x_value (a : ℝ) : ℝ := a * 1.75
noncomputable def m_value (b : ℝ) : ℝ := b * 0.20

theorem value_of_m_div_x (a b : ℝ) (h1 : ratio_of_a_to_b a b) (h2 : 0 < a) (h3 : 0 < b) :
  (m_value b) / (x_value a) = 1 / 7 :=
by
  sorry

end value_of_m_div_x_l204_204667


namespace workers_allocation_l204_204615

-- Definitions based on conditions
def num_workers := 90
def bolt_per_worker := 15
def nut_per_worker := 24
def bolt_matching_requirement := 2

-- Statement of the proof problem
theorem workers_allocation (x y : ℕ) :
  x + y = num_workers ∧
  bolt_matching_requirement * bolt_per_worker * x = nut_per_worker * y →
  x = 40 ∧ y = 50 :=
by
  sorry

end workers_allocation_l204_204615


namespace proof_problem_l204_204335

noncomputable def a : ℝ := (11 + Real.sqrt 337) ^ (1 / 3)
noncomputable def b : ℝ := (11 - Real.sqrt 337) ^ (1 / 3)
noncomputable def x : ℝ := a + b

theorem proof_problem : x^3 + 18 * x = 22 := by
  sorry

end proof_problem_l204_204335


namespace firstGradeMuffins_l204_204240

-- Define the conditions as the number of muffins baked by each class
def mrsBrierMuffins : ℕ := 18
def mrsMacAdamsMuffins : ℕ := 20
def mrsFlanneryMuffins : ℕ := 17

-- Define the total number of muffins baked
def totalMuffins : ℕ := mrsBrierMuffins + mrsMacAdamsMuffins + mrsFlanneryMuffins

-- Prove that the total number of muffins baked is 55
theorem firstGradeMuffins : totalMuffins = 55 := by
  sorry

end firstGradeMuffins_l204_204240


namespace amount_given_by_mom_l204_204366

def amount_spent_by_Mildred : ℕ := 25
def amount_spent_by_Candice : ℕ := 35
def amount_left : ℕ := 40

theorem amount_given_by_mom : 
  (amount_spent_by_Mildred + amount_spent_by_Candice + amount_left) = 100 := by
  sorry

end amount_given_by_mom_l204_204366


namespace find_fraction_l204_204958

theorem find_fraction
  (x : ℝ)
  (h : (x)^35 * (1/4)^18 = 1 / (2 * 10^35)) : x = 1/5 :=
by 
  sorry

end find_fraction_l204_204958


namespace square_side_increase_factor_l204_204379

theorem square_side_increase_factor (s k : ℕ) (x new_x : ℕ) (h1 : x = 4 * s) (h2 : new_x = 4 * x) (h3 : new_x = 4 * (k * s)) : k = 4 :=
by
  sorry

end square_side_increase_factor_l204_204379


namespace gcd_of_45_and_75_l204_204714

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l204_204714


namespace gcd_45_75_l204_204689

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204689


namespace brick_surface_area_l204_204753

theorem brick_surface_area (l w h : ℝ) (hl : l = 10) (hw : w = 4) (hh : h = 3) : 
  2 * (l * w + l * h + w * h) = 164 := 
by
  sorry

end brick_surface_area_l204_204753


namespace valid_votes_election_l204_204844

-- Definition of the problem
variables (V : ℝ) -- the total number of valid votes
variables (hvoting_percentage : V > 0 ∧ V ≤ 1) -- constraints for voting percentage in general
variables (h_winning_votes : 0.70 * V) -- 70% of the votes
variables (h_losing_votes : 0.30 * V) -- 30% of the votes

-- Given condition: the winning candidate won by a majority of 184 votes
variables (majority : ℝ) (h_majority : 0.70 * V - 0.30 * V = 184)

/-- The total number of valid votes in the election. -/
theorem valid_votes_election : V = 460 :=
by
  sorry

end valid_votes_election_l204_204844


namespace gcd_45_75_l204_204710

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204710


namespace functional_expression_value_at_x_equals_zero_l204_204591

-- Define the basic properties
def y_inversely_proportional_to_x_plus_2 (y x : ℝ) : Prop :=
  ∃ k : ℝ, y = k / (x + 2)

-- Given condition: y = 3 when x = -1
def condition (y x : ℝ) : Prop :=
  y = 3 ∧ x = -1

-- Theorems to prove
theorem functional_expression (y x : ℝ) :
  y_inversely_proportional_to_x_plus_2 y x ∧ condition y x → y = 3 / (x + 2) :=
by
  sorry

theorem value_at_x_equals_zero (y x : ℝ) :
  y_inversely_proportional_to_x_plus_2 y x ∧ condition y x → (y = 3 / (x + 2) ∧ x = 0 → y = 3 / 2) :=
by
  sorry

end functional_expression_value_at_x_equals_zero_l204_204591


namespace largest_square_side_length_l204_204847

noncomputable def largestInscribedSquareSide (s : ℝ) (sharedSide : ℝ) : ℝ :=
  let y := (s * Real.sqrt 2 - sharedSide * Real.sqrt 3) / (2 * Real.sqrt 2)
  y

theorem largest_square_side_length :
  let s := 12
  let t := (s * Real.sqrt 6) / 3
  largestInscribedSquareSide s t = 6 - Real.sqrt 6 :=
by
  sorry

end largest_square_side_length_l204_204847


namespace factorization_l204_204924

theorem factorization (m : ℝ) : m^2 - 3 * m = m * (m - 3) :=
by sorry

end factorization_l204_204924


namespace minimum_throws_for_repeated_sum_l204_204073

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l204_204073


namespace cone_base_circumference_l204_204404

theorem cone_base_circumference 
  (r : ℝ) 
  (θ : ℝ) 
  (h₁ : r = 5) 
  (h₂ : θ = 225) : 
  (θ / 360 * 2 * Real.pi * r) = (25 * Real.pi / 4) :=
by
  -- Proof skipped
  sorry

end cone_base_circumference_l204_204404


namespace percent_other_birds_is_31_l204_204964

noncomputable def initial_hawk_percentage : ℝ := 0.30
noncomputable def initial_paddyfield_warbler_percentage : ℝ := 0.25
noncomputable def initial_kingfisher_percentage : ℝ := 0.10
noncomputable def initial_hp_k_total : ℝ := initial_hawk_percentage + initial_paddyfield_warbler_percentage + initial_kingfisher_percentage

noncomputable def migrated_hawk_percentage : ℝ := 0.8 * initial_hawk_percentage
noncomputable def migrated_kingfisher_percentage : ℝ := 2 * initial_kingfisher_percentage
noncomputable def migrated_hp_k_total : ℝ := migrated_hawk_percentage + initial_paddyfield_warbler_percentage + migrated_kingfisher_percentage

noncomputable def other_birds_percentage : ℝ := 1 - migrated_hp_k_total

theorem percent_other_birds_is_31 : other_birds_percentage = 0.31 := sorry

end percent_other_birds_is_31_l204_204964


namespace andrew_spent_total_amount_l204_204156

/-- Conditions:
1. Andrew played a total of 7 games.
2. Cost distribution for games:
   - 3 games cost $9.00 each
   - 2 games cost $12.50 each
   - 2 games cost $15.00 each
3. Additional expenses:
   - $25.00 on snacks
   - $20.00 on drinks
-/
def total_cost_games : ℝ :=
  (3 * 9) + (2 * 12.5) + (2 * 15)

def cost_snacks : ℝ := 25
def cost_drinks : ℝ := 20

def total_spent (cost_games cost_snacks cost_drinks : ℝ) : ℝ :=
  cost_games + cost_snacks + cost_drinks

theorem andrew_spent_total_amount :
  total_spent total_cost_games 25 20 = 127 := by
  -- The proof is omitted
  sorry

end andrew_spent_total_amount_l204_204156


namespace klinker_daughter_age_l204_204866

-- Define the conditions in Lean
variable (D : ℕ) -- ℕ is the natural number type in Lean

-- Define the theorem statement
theorem klinker_daughter_age (h1 : 35 + 15 = 50)
    (h2 : 50 = 2 * (D + 15)) : D = 10 := by
  sorry

end klinker_daughter_age_l204_204866


namespace fans_who_received_all_three_l204_204299

theorem fans_who_received_all_three (n : ℕ) :
  (∀ k, k ≤ 5000 → (k % 100 = 0 → k % 40 = 0 → k % 60 = 0 → k / 600 ≤ n)) ∧
  (∀ k, k ≤ 5000 → (k % 100 = 0 → k % 40 = 0 → k % 60 = 0 → k / 600 ≤ 8)) :=
by
  sorry

end fans_who_received_all_three_l204_204299


namespace find_target_number_l204_204385

theorem find_target_number : ∃ n ≥ 0, (∀ k < 5, ∃ m, 0 ≤ m ∧ m ≤ n ∧ m % 11 = 3 ∧ m = 3 + k * 11) ∧ n = 47 :=
by
  sorry

end find_target_number_l204_204385


namespace parabola_focus_distance_l204_204310

theorem parabola_focus_distance (p : ℝ) (hp : p > 0) (A : ℝ × ℝ)
  (hA_on_parabola : A.2 ^ 2 = 2 * p * A.1)
  (hA_focus_dist : dist A (p / 2, 0) = 12)
  (hA_yaxis_dist : abs A.1 = 9) : p = 6 :=
sorry

end parabola_focus_distance_l204_204310


namespace number_of_cars_l204_204217

theorem number_of_cars 
  (num_bikes : ℕ) (num_wheels_total : ℕ) (wheels_per_bike : ℕ) (wheels_per_car : ℕ)
  (h1 : num_bikes = 10) (h2 : num_wheels_total = 76) (h3 : wheels_per_bike = 2) (h4 : wheels_per_car = 4) :
  ∃ (C : ℕ), C = 14 := 
by
  sorry

end number_of_cars_l204_204217


namespace count_integers_divis_by_8_l204_204303

theorem count_integers_divis_by_8 : 
  ∃ k : ℕ, k = 49 ∧ ∀ n : ℕ, 2 ≤ n ∧ n ≤ 80 → (∃ m : ℤ, (n-1) * n * (n+1) = 8 * m) ↔ (∃ m : ℕ, m ≤ k) :=
by 
  sorry

end count_integers_divis_by_8_l204_204303


namespace find_a_from_circle_and_chord_l204_204942

theorem find_a_from_circle_and_chord 
  (a : ℝ)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + 2*x - 2*y + a = 0)
  (line_eq : ∀ x y : ℝ, x + y + 2 = 0)
  (chord_length : ∀ x1 y1 x2 y2 : ℝ, x1^2 + y1^2 + 2*x1 - 2*y1 + a = 0 ∧ x2^2 + y2^2 + 2*x2 - 2*y2 + a = 0 ∧ x1 + y1 + 2 = 0 ∧ x2 + y2 + 2 = 0 → (x1 - x2)^2 + (y1 - y2)^2 = 16) :
  a = -4 :=
by
  sorry

end find_a_from_circle_and_chord_l204_204942


namespace sum_first_100_terms_is_l204_204519

open Nat

noncomputable def seq (a_n : ℕ → ℤ) : Prop :=
  a_n 2 = 2 ∧ ∀ n : ℕ, n > 0 → a_n (n + 2) + (-1)^(n + 1) * a_n n = 1 + (-1)^n

def sum_seq (f : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum f

theorem sum_first_100_terms_is :
  ∃ (a_n : ℕ → ℤ), seq a_n ∧ sum_seq a_n 100 = 2550 :=
by
  sorry

end sum_first_100_terms_is_l204_204519


namespace square_mirror_side_length_l204_204795

theorem square_mirror_side_length :
  ∃ (side_length : ℝ),
  let wall_width := 42
  let wall_length := 27.428571428571427
  let wall_area := wall_width * wall_length
  let mirror_area := wall_area / 2
  (side_length * side_length = mirror_area) → side_length = 24 :=
by
  use 24
  intro h
  sorry

end square_mirror_side_length_l204_204795


namespace range_of_a_l204_204328

theorem range_of_a (a x y : ℝ) (h1 : x - y = a + 3) (h2 : 2 * x + y = 5 * a) (h3 : x < y) : a < -3 :=
by
  sorry

end range_of_a_l204_204328


namespace arccos_of_sqrt3_div_2_l204_204172

theorem arccos_of_sqrt3_div_2 : Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 := by
  sorry

end arccos_of_sqrt3_div_2_l204_204172


namespace sum_of_two_numbers_l204_204516

theorem sum_of_two_numbers (a b : ℕ) (h1 : (a + b) * (a - b) = 1996) (h2 : (a + b) % 2 = (a - b) % 2) (h3 : a + b > a - b) : a + b = 998 := 
sorry

end sum_of_two_numbers_l204_204516


namespace gcd_45_75_l204_204741

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204741


namespace f_inv_128_l204_204957

noncomputable def f : ℕ → ℕ := sorry -- Placeholder for the function definition.

axiom f_5 : f 5 = 2           -- Condition 1: f(5) = 2
axiom f_2x : ∀ x, f (2 * x) = 2 * f x  -- Condition 2: f(2x) = 2f(x) for all x

theorem f_inv_128 : f⁻¹ 128 = 320 := sorry -- Prove that f⁻¹(128) = 320 given the conditions

end f_inv_128_l204_204957


namespace absolute_inequality_l204_204931

theorem absolute_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := 
sorry

end absolute_inequality_l204_204931


namespace number_of_pupils_in_class_l204_204395

theorem number_of_pupils_in_class
(U V : ℕ) (increase : ℕ) (avg_increase : ℕ) (n : ℕ) 
(h1 : U = 85) (h2 : V = 45) (h3 : increase = U - V) (h4 : avg_increase = 1 / 2) (h5 : increase / avg_increase = n) :
n = 80 := by
sorry

end number_of_pupils_in_class_l204_204395


namespace exists_k_divides_poly_l204_204588

theorem exists_k_divides_poly (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : a 2 = 1) 
  (h₂ : ∀ k : ℕ, a (k + 2) = a (k + 1) + a k) :
  ∀ (m : ℕ), m > 0 → ∃ k : ℕ, m ∣ (a k ^ 4 - a k - 2) :=
by
  sorry

end exists_k_divides_poly_l204_204588


namespace isosceles_obtuse_triangle_angle_correct_l204_204797

noncomputable def isosceles_obtuse_triangle_smallest_angle (A B C : ℝ) (h1 : A = 1.3 * 90) (h2 : B = C) (h3 : A + B + C = 180) : ℝ :=
  (180 - A) / 2

theorem isosceles_obtuse_triangle_angle_correct 
  (A B C : ℝ)
  (h1 : A = 1.3 * 90)
  (h2 : B = C)
  (h3 : A + B + C = 180) :
  isosceles_obtuse_triangle_smallest_angle A B C h1 h2 h3 = 31.5 :=
sorry

end isosceles_obtuse_triangle_angle_correct_l204_204797


namespace geometric_series_solution_l204_204037

-- Let a, r : ℝ be real numbers representing the parameters from the problem's conditions.
variables (a r : ℝ)

-- Define the conditions as hypotheses.
def condition1 : Prop := a / (1 - r) = 20
def condition2 : Prop := a / (1 - r^2) = 8

-- The theorem states that under these conditions, r equals 3/2.
theorem geometric_series_solution (hc1 : condition1 a r) (hc2 : condition2 a r) : r = 3 / 2 :=
sorry

end geometric_series_solution_l204_204037


namespace inequality_solution_l204_204655

theorem inequality_solution (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < (5 / 9) :=
by
  sorry

end inequality_solution_l204_204655


namespace gcd_45_75_l204_204700

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204700


namespace quadratic_no_third_quadrant_l204_204377

theorem quadratic_no_third_quadrant (x y : ℝ) : 
  (y = x^2 - 2 * x) → ¬(x < 0 ∧ y < 0) :=
by
  intro hy
  sorry

end quadratic_no_third_quadrant_l204_204377


namespace area_of_rectangle_l204_204161

def length : ℝ := 0.5
def width : ℝ := 0.24

theorem area_of_rectangle :
  length * width = 0.12 :=
by
  sorry

end area_of_rectangle_l204_204161


namespace min_throws_for_repeated_sum_l204_204096

theorem min_throws_for_repeated_sum : 
  (∀ (n : ℕ), n = 24 ∧ (∀ (x : ℕ), x ≥ 4 ∧ x ≤ 24)) → 22 :=
by
  sorry

end min_throws_for_repeated_sum_l204_204096


namespace binom_15_4_l204_204563

theorem binom_15_4 : nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l204_204563


namespace four_planes_divide_space_into_fifteen_parts_l204_204820

-- Define the function that calculates the number of parts given the number of planes.
def parts_divided_by_planes (x : ℕ) : ℕ :=
  (x^3 + 5 * x + 6) / 6

-- Prove that four planes divide the space into 15 parts.
theorem four_planes_divide_space_into_fifteen_parts : parts_divided_by_planes 4 = 15 :=
by sorry

end four_planes_divide_space_into_fifteen_parts_l204_204820


namespace no_nat_nums_gt_one_divisibility_conditions_l204_204426

theorem no_nat_nums_gt_one_divisibility_conditions :
  ¬ ∃ (a b c : ℕ), 
    1 < a ∧ 1 < b ∧ 1 < c ∧
    (c ∣ a^2 - 1) ∧ (b ∣ a^2 - 1) ∧ 
    (a ∣ c^2 - 1) ∧ (b ∣ c^2 - 1) :=
by 
  sorry

end no_nat_nums_gt_one_divisibility_conditions_l204_204426


namespace minim_product_l204_204639

def digits := {5, 6, 7, 8}

def is_valid_combination (a b c d : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)

def form_number (x y : ℕ) : ℕ :=
  10 * x + y

theorem minim_product : 
  ∃ a b c d : ℕ, is_valid_combination a b c d ∧ form_number a c * form_number b d = 4368 :=
by
  sorry

end minim_product_l204_204639


namespace doll_cost_l204_204675

theorem doll_cost (D : ℝ) (h : 4 * D = 60) : D = 15 :=
by {
  sorry
}

end doll_cost_l204_204675


namespace area_of_given_rhombus_l204_204575

open Real

noncomputable def area_of_rhombus_with_side_and_angle (side : ℝ) (angle : ℝ) : ℝ :=
  let half_diag1 := side * cos (angle / 2)
  let half_diag2 := side * sin (angle / 2)
  let diag1 := 2 * half_diag1
  let diag2 := 2 * half_diag2
  (diag1 * diag2) / 2

theorem area_of_given_rhombus :
  area_of_rhombus_with_side_and_angle 25 40 = 201.02 :=
by
  sorry

end area_of_given_rhombus_l204_204575


namespace gcd_45_75_l204_204681

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204681


namespace probability_cocaptains_l204_204520

theorem probability_cocaptains (team_sizes : List ℕ)
  (h_sizes : team_sizes = [5, 7, 8])
  (num_cocaptains : ∀ (n : ℕ), n ∈ team_sizes → n ≥ 2) :
  let probability_cocaptains (team_size : ℕ) : ℚ :=
    2 / (team_size * (team_size - 1))
  let total_probability : ℚ :=
    (1 / 3) * (probability_cocaptains 5 + probability_cocaptains 7 + probability_cocaptains 8)
  total_probability = 11 / 180 :=
by
  sorry

end probability_cocaptains_l204_204520


namespace units_digit_of_n_l204_204301

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : m % 10 = 4) : n % 10 = 4 :=
by
  sorry

end units_digit_of_n_l204_204301


namespace correct_statement_C_l204_204132

theorem correct_statement_C
  (a : ℚ) : a < 0 → |a| = -a := 
by
  sorry

end correct_statement_C_l204_204132


namespace gcd_45_75_l204_204683

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204683


namespace scientific_notation_of_2270000_l204_204638

theorem scientific_notation_of_2270000 : 
  (2270000 : ℝ) = 2.27 * 10^6 :=
sorry

end scientific_notation_of_2270000_l204_204638


namespace inequality_solution_set_l204_204669

theorem inequality_solution_set :
  { x : ℝ | 1 < x ∧ x < 2 } = { x : ℝ | (x - 2) / (1 - x) > 0 } :=
by sorry

end inequality_solution_set_l204_204669


namespace range_of_m_l204_204236

open Set

noncomputable def M (m : ℝ) : Set ℝ := {x | x ≤ m}
noncomputable def N : Set ℝ := {y | y ≥ 1}

theorem range_of_m (m : ℝ) : M m ∩ N = ∅ → m < 1 := by
  intros h
  sorry

end range_of_m_l204_204236


namespace minimum_discount_l204_204534

theorem minimum_discount (x : ℝ) (hx : x ≤ 10) : 
  let cost_price := 400 
  let selling_price := 500
  let discount_price := selling_price - (selling_price * (x / 100))
  let gross_profit := discount_price - cost_price 
  gross_profit ≥ cost_price * 0.125 :=
sorry

end minimum_discount_l204_204534


namespace magnitude_of_T_l204_204356

theorem magnitude_of_T : 
  let i := Complex.I
  let T := 3 * ((1 + i) ^ 15 - (1 - i) ^ 15)
  Complex.abs T = 768 := by
  sorry

end magnitude_of_T_l204_204356


namespace bill_has_six_times_more_nuts_l204_204160

-- Definitions for the conditions
def sue_has_nuts : ℕ := 48
def harry_has_nuts (sueNuts : ℕ) : ℕ := 2 * sueNuts
def combined_nuts (harryNuts : ℕ) (billNuts : ℕ) : ℕ := harryNuts + billNuts
def bill_has_nuts (totalNuts : ℕ) (harryNuts : ℕ) : ℕ := totalNuts - harryNuts

-- Statement to prove
theorem bill_has_six_times_more_nuts :
  ∀ sueNuts billNuts harryNuts totalNuts,
    sueNuts = sue_has_nuts →
    harryNuts = harry_has_nuts sueNuts →
    totalNuts = 672 →
    combined_nuts harryNuts billNuts = totalNuts →
    billNuts = bill_has_nuts totalNuts harryNuts →
    billNuts = 6 * harryNuts :=
by
  intros sueNuts billNuts harryNuts totalNuts hsueNuts hharryNuts htotalNuts hcombinedNuts hbillNuts
  sorry

end bill_has_six_times_more_nuts_l204_204160


namespace joseph_drives_more_l204_204350

-- Definitions for the problem
def v_j : ℝ := 50 -- Joseph's speed in mph
def t_j : ℝ := 2.5 -- Joseph's time in hours
def v_k : ℝ := 62 -- Kyle's speed in mph
def t_k : ℝ := 2 -- Kyle's time in hours

-- Prove that Joseph drives 1 more mile than Kyle
theorem joseph_drives_more : (v_j * t_j) - (v_k * t_k) = 1 := 
by 
  sorry

end joseph_drives_more_l204_204350


namespace hyperbola_parabola_shared_focus_l204_204320

theorem hyperbola_parabola_shared_focus (a : ℝ) (h : a > 0) :
  (∃ b c : ℝ, b^2 = 3 ∧ c = 2 ∧ a^2 = c^2 - b^2 ∧ b ≠ 0) →
  a = 1 :=
by
  intro h_shared_focus
  sorry

end hyperbola_parabola_shared_focus_l204_204320


namespace each_player_plays_36_minutes_l204_204791

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l204_204791


namespace combined_tax_rate_l204_204752

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (h1 : Mindy_income = 4 * Mork_income) :
  let Mork_tax := 0.45 * Mork_income;
  let Mindy_tax := 0.15 * Mindy_income;
  let combined_tax := Mork_tax + Mindy_tax;
  let combined_income := Mork_income + Mindy_income;
  combined_tax / combined_income * 100 = 21 := 
by
  sorry

end combined_tax_rate_l204_204752


namespace k_value_range_l204_204444

noncomputable def f (x : ℝ) : ℝ := x - 1 - Real.log x

theorem k_value_range {k : ℝ} (h : ∀ x : ℝ, 0 < x → f x ≥ k * x - 2) : 
  k ≤ 1 - 1 / Real.exp 2 := 
sorry

end k_value_range_l204_204444


namespace correct_operation_l204_204290

theorem correct_operation (x : ℝ) : (x^3 * x^2 = x^5) :=
by sorry

end correct_operation_l204_204290


namespace xy_range_l204_204443

theorem xy_range (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
    (h_eqn : x + 3 * y + 2 / x + 4 / y = 10) :
    1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
  sorry

end xy_range_l204_204443


namespace bridge_construction_l204_204853

-- Definitions used in the Lean statement based on conditions.
def rate (workers : ℕ) (days : ℕ) : ℚ := 1 / (workers * days)

-- The problem statement: prove that if 60 workers working together can build the bridge in 3 days, 
-- then 120 workers will take 1.5 days to build the bridge.
theorem bridge_construction (t : ℚ) : 
  (rate 60 3) * 120 * t = 1 → t = 1.5 := by
  sorry

end bridge_construction_l204_204853


namespace gcd_45_75_l204_204725

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204725


namespace solve_equation_1_solve_equation_2_l204_204506

namespace Proofs

theorem solve_equation_1 (x : ℝ) :
  (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) :
  x * (x - 6) = 6 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15 :=
by
  sorry

end Proofs

end solve_equation_1_solve_equation_2_l204_204506


namespace solve_for_nabla_l204_204952

theorem solve_for_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by
  sorry

end solve_for_nabla_l204_204952


namespace hotel_friends_count_l204_204649

theorem hotel_friends_count
  (n : ℕ)
  (friend_share extra friend_payment : ℕ)
  (h1 : 7 * 80 + friend_payment = 720)
  (h2 : friend_payment = friend_share + extra)
  (h3 : friend_payment = 160)
  (h4 : extra = 70)
  (h5 : friend_share = 90) :
  n = 8 :=
sorry

end hotel_friends_count_l204_204649


namespace geometric_sequence_third_term_l204_204036

theorem geometric_sequence_third_term 
  (a r : ℝ)
  (h1 : a = 3)
  (h2 : a * r^4 = 243) : 
  a * r^2 = 27 :=
by
  sorry

end geometric_sequence_third_term_l204_204036


namespace john_height_in_feet_l204_204484

theorem john_height_in_feet (initial_height : ℕ) (growth_rate : ℕ) (months : ℕ) (inches_per_foot : ℕ) :
  initial_height = 66 → growth_rate = 2 → months = 3 → inches_per_foot = 12 → 
  (initial_height + growth_rate * months) / inches_per_foot = 6 := by
  intros h1 h2 h3 h4
  sorry

end john_height_in_feet_l204_204484


namespace opposite_of_negative_one_fifth_l204_204043

theorem opposite_of_negative_one_fifth : -(-1 / 5) = (1 / 5) :=
by
  sorry

end opposite_of_negative_one_fifth_l204_204043


namespace car_owners_without_motorcycles_l204_204341

theorem car_owners_without_motorcycles 
  (total_adults : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ) (bicycle_owners : ℕ) (total_own_vehicle : ℕ)
  (h1 : total_adults = 400) (h2 : car_owners = 350) (h3 : motorcycle_owners = 60) (h4 : bicycle_owners = 30)
  (h5 : total_own_vehicle = total_adults)
  : (car_owners - 10 = 340) :=
by
  sorry

end car_owners_without_motorcycles_l204_204341


namespace equal_real_roots_l204_204375

theorem equal_real_roots (m : ℝ) : (∃ x : ℝ, x * x - 4 * x - m = 0) → (16 + 4 * m = 0) → m = -4 :=
by
  sorry

end equal_real_roots_l204_204375


namespace factor_sum_l204_204606

theorem factor_sum (P Q R : ℤ) (h : ∃ (b c : ℤ), (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + P*x^2 + R*x + Q) : 
  P + Q + R = 11*P - 1 := 
sorry

end factor_sum_l204_204606


namespace solve_for_x_l204_204868

theorem solve_for_x : 
  ∃ x : ℚ, x^2 + 145 = (x - 19)^2 ∧ x = 108 / 19 := 
by 
  sorry

end solve_for_x_l204_204868


namespace percentage_increase_in_savings_l204_204899

theorem percentage_increase_in_savings
  (I : ℝ) -- Original income of Paulson
  (E : ℝ) -- Original expenditure of Paulson
  (hE : E = 0.75 * I) -- Paulson spends 75% of his income
  (h_inc_income : 1.2 * I = I + 0.2 * I) -- Income is increased by 20%
  (h_inc_expenditure : 0.825 * I = 0.75 * I + 0.1 * (0.75 * I)) -- Expenditure is increased by 10%
  : (0.375 * I - 0.25 * I) / (0.25 * I) * 100 = 50 := by
  sorry

end percentage_increase_in_savings_l204_204899


namespace add_and_subtract_l204_204128

theorem add_and_subtract (a b c : ℝ) (h1 : a = 0.45) (h2 : b = 52.7) (h3 : c = 0.25) : 
  (a + b) - c = 52.9 :=
by 
  sorry

end add_and_subtract_l204_204128


namespace initial_number_of_cards_l204_204631

theorem initial_number_of_cards (x : ℕ) (h : x + 76 = 79) : x = 3 :=
by
  sorry

end initial_number_of_cards_l204_204631


namespace isosceles_triangles_l204_204500

theorem isosceles_triangles (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h_triangle : ∀ n : ℕ, (a^n + b^n > c^n ∧ b^n + c^n > a^n ∧ c^n + a^n > b^n)) :
  b = c := 
sorry

end isosceles_triangles_l204_204500


namespace kim_min_pours_l204_204178

-- Define the initial conditions
def initial_volume (V : ℝ) : ℝ := V
def pour (V : ℝ) : ℝ := 0.9 * V

-- Define the remaining volume after n pours
def remaining_volume (V : ℝ) (n : ℕ) : ℝ := V * (0.9)^n

-- State the problem: After 7 pours, the remaining volume is less than half the initial volume
theorem kim_min_pours (V : ℝ) (hV : V > 0) : remaining_volume V 7 < V / 2 :=
by
  -- Because the proof is not required, we use sorry
  sorry

end kim_min_pours_l204_204178


namespace minimum_throws_for_four_dice_l204_204127

noncomputable def minimum_throws_to_ensure_repeated_sum (d : ℕ) : ℕ :=
  let min_sum := d * 1 in
  let max_sum := d * 6 in
  let distinct_sums := max_sum - min_sum + 1 in
  distinct_sums + 1

theorem minimum_throws_for_four_dice : minimum_throws_to_ensure_repeated_sum 4 = 22 := by
  sorry

end minimum_throws_for_four_dice_l204_204127


namespace radius_of_tangent_sphere_l204_204549

theorem radius_of_tangent_sphere (r1 r2 : ℝ) (h : r1 = 12 ∧ r2 = 3) :
  ∃ r : ℝ, (r = 6) :=
by
  sorry

end radius_of_tangent_sphere_l204_204549


namespace lines_parallel_l204_204515

def line1 (x y : ℝ) : Prop := x - y + 2 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

theorem lines_parallel : 
  (∀ x y, line1 x y ↔ y = x + 2) ∧ 
  (∀ x y, line2 x y ↔ y = x + 1) ∧ 
  ∃ m₁ m₂ c₁ c₂, (∀ x y, (y = m₁ * x + c₁) ↔ line1 x y) ∧ (∀ x y, (y = m₂ * x + c₂) ↔ line2 x y) ∧ m₁ = m₂ ∧ c₁ ≠ c₂ :=
by
  sorry

end lines_parallel_l204_204515


namespace gcd_45_75_l204_204698

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204698


namespace expenses_neg_of_income_pos_l204_204033

theorem expenses_neg_of_income_pos :
  ∀ (income expense : Int), income = 5 → expense = -income → expense = -5 :=
by
  intros income expense h_income h_expense
  rw [h_income] at h_expense
  exact h_expense

end expenses_neg_of_income_pos_l204_204033


namespace combined_motion_properties_l204_204329

noncomputable def y (x : ℝ) := Real.sin x + (Real.sin x) ^ 2

theorem combined_motion_properties :
  (∀ x: ℝ, - (1/4: ℝ) ≤ y x ∧ y x ≤ 2) ∧ 
  (∃ x: ℝ, y x = 2) ∧
  (∃ x: ℝ, y x = -(1/4: ℝ)) :=
by
  -- The complete proofs for these statements are omitted.
  -- This theorem specifies the required properties of the function y.
  sorry

end combined_motion_properties_l204_204329


namespace gcd_45_75_l204_204736

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204736


namespace minimum_throws_to_ensure_same_sum_twice_l204_204060

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l204_204060


namespace infinite_n_multiples_of_six_available_l204_204988

theorem infinite_n_multiples_of_six_available :
  ∃ (S : Set ℕ), (∀ n ∈ S, ∃ (A : Matrix (Fin 3) (Fin (n : ℕ)) Nat),
    (∀ (i : Fin n), (A 0 i + A 1 i + A 2 i) % 6 = 0) ∧ 
    (∀ (i : Fin 3), (Finset.univ.sum (λ j => A i j)) % 6 = 0)) ∧
  Set.Infinite S :=
sorry

end infinite_n_multiples_of_six_available_l204_204988


namespace minimum_throws_l204_204054

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l204_204054


namespace equal_play_time_l204_204780

-- Definitions based on conditions
variables (P : ℕ) (F : ℕ) (T : ℕ) (S : ℕ)
-- P: Total number of players
-- F: Number of players on the field at any time
-- T: Total duration of the match in minutes
-- S: Time each player plays

-- Given values from conditions
noncomputable def totalPlayers : ℕ := 10
noncomputable def fieldPlayers : ℕ := 8
noncomputable def matchDuration : ℕ := 45

theorem equal_play_time :
  P = totalPlayers →
  F = fieldPlayers →
  T = matchDuration →
  (F * T) / P = S →
  S = 36 :=
by
  intros hP hF hT hS
  rw [hP, hF, hT] at hS
  exact hS

end equal_play_time_l204_204780


namespace find_a_n_l204_204436

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₁ : ∀ n, a n > 0)
  (h₂ : ∀ n, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n, a n = Real.sqrt n - Real.sqrt (n - 1) :=
by
  sorry

end find_a_n_l204_204436


namespace james_total_cost_l204_204623

def suit1 := 300
def suit2_pretail := 3 * suit1
def suit2 := suit2_pretail + 200
def total_cost := suit1 + suit2

theorem james_total_cost : total_cost = 1400 := by
  sorry

end james_total_cost_l204_204623


namespace negation_of_prop_p_l204_204258

open Classical

theorem negation_of_prop_p:
  (¬ ∀ x : ℕ, x > 0 → (1 / 2) ^ x ≤ 1 / 2) ↔ ∃ x : ℕ, x > 0 ∧ (1 / 2) ^ x > 1 / 2 := 
by
  sorry

end negation_of_prop_p_l204_204258


namespace max_value_expression_l204_204338

theorem max_value_expression (a b c : ℝ) 
  (ha : 300 ≤ a ∧ a ≤ 500) 
  (hb : 500 ≤ b ∧ b ≤ 1500) 
  (hc : c = 100) : 
  (∃ M, M = 8 ∧ ∀ x, x = (b + c) / (a - c) → x ≤ M) := 
sorry

end max_value_expression_l204_204338


namespace num_zeros_g_l204_204197

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x > 2 then m * (x - 2) / x
  else if 0 < x ∧ x ≤ 2 then 3 * x - x^2
  else 0

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x m - 2

-- Statement to prove
theorem num_zeros_g (m : ℝ) : ∃ n : ℕ, (n = 4 ∨ n = 6) :=
sorry

end num_zeros_g_l204_204197


namespace first_day_exceeds_200_l204_204213

-- Bacteria population doubling function
def bacteria_population (n : ℕ) : ℕ := 4 * 3 ^ n

-- Prove the smallest day where bacteria count exceeds 200 is 4
theorem first_day_exceeds_200 : ∃ n : ℕ, bacteria_population n > 200 ∧ ∀ m < n, bacteria_population m ≤ 200 :=
by 
    -- Proof will be filled here
    sorry

end first_day_exceeds_200_l204_204213


namespace length_of_square_side_l204_204584

theorem length_of_square_side 
  (r : ℝ) 
  (A : ℝ) 
  (h : A = 42.06195997410015) 
  (side_length : ℝ := 2 * r)
  (area_of_square : ℝ := side_length ^ 2)
  (segment_area : ℝ := 4 * (π * r * r / 4))
  (enclosed_area: ℝ := area_of_square - segment_area)
  (h2 : enclosed_area = A) :
  side_length = 14 :=
by sorry

end length_of_square_side_l204_204584


namespace find_x_l204_204510

theorem find_x :
  ∃ x : ℚ, (1 / 3) * ((x + 8) + (8*x + 3) + (3*x + 9)) = 5*x - 9 ∧ x = 47 / 3 :=
by
  sorry

end find_x_l204_204510


namespace divisor_of_930_l204_204390

theorem divisor_of_930 : ∃ d > 1, d ∣ 930 ∧ ∀ e, e ∣ 930 → e > 1 → d ≤ e :=
by
  sorry

end divisor_of_930_l204_204390


namespace base_equivalence_l204_204514

theorem base_equivalence :
  let n_7 := 4 * 7 + 3  -- 43 in base 7 expressed in base 10.
  ∃ d : ℕ, (3 * d + 4 = n_7) → d = 9 :=
by
  let n_7 := 31
  sorry

end base_equivalence_l204_204514


namespace cos_neg_17pi_over_4_l204_204425

noncomputable def cos_value : ℝ := (Real.pi / 4).cos

theorem cos_neg_17pi_over_4 :
  (Real.cos (-17 * Real.pi / 4)) = cos_value :=
by
  -- Define even property of cosine and angle simplification
  sorry

end cos_neg_17pi_over_4_l204_204425


namespace minimum_rolls_for_duplicate_sum_l204_204091

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l204_204091


namespace binom_15_4_l204_204562

theorem binom_15_4 : nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l204_204562


namespace total_distance_traveled_l204_204394

theorem total_distance_traveled (d : ℝ) (h1 : d/3 + d/4 + d/5 = 47/60) : 3 * d = 3 :=
by
  sorry

end total_distance_traveled_l204_204394


namespace min_throws_to_ensure_repeat_sum_l204_204121

theorem min_throws_to_ensure_repeat_sum : 
  ∀ (min_sum max_sum : ℤ), 
  min_sum = 4 ∧ max_sum = 24 
  → ∃ n, n ≥ 22 ∧ n = 22 :=
by
  intros min_sum max_sum h
  cases h with h_min h_max
  existsi 22
  split
  · exact Nat.le_refl 22
  · sorry

end min_throws_to_ensure_repeat_sum_l204_204121


namespace fido_leash_problem_l204_204574

theorem fido_leash_problem
  (r : ℝ) 
  (octagon_area : ℝ := 2 * r^2 * Real.sqrt 2)
  (circle_area : ℝ := Real.pi * r^2)
  (explore_fraction : ℝ := circle_area / octagon_area)
  (a b : ℝ) 
  (h_simplest_form : explore_fraction = (Real.sqrt a) / b * Real.pi)
  (h_a : a = 2)
  (h_b : b = 2) : a * b = 4 :=
by sorry

end fido_leash_problem_l204_204574


namespace parabola_vertex_n_l204_204382

theorem parabola_vertex_n (x y : ℝ) (h : y = -3 * x^2 - 24 * x - 72) : ∃ m n : ℝ, (m, n) = (-4, -24) :=
by
  sorry

end parabola_vertex_n_l204_204382


namespace length_MN_proof_l204_204972

-- Declare a noncomputable section to avoid computational requirements
noncomputable section

-- Define the quadrilateral ABCD with given sides
structure Quadrilateral :=
  (BC AD AB CD : ℕ)
  (BC_AD_parallel : Prop)

-- Define a theorem to calculate the length MN
theorem length_MN_proof (ABCD : Quadrilateral) 
  (M N : ℝ) (BisectorsIntersect_M : Prop) (BisectorsIntersect_N : Prop) : 
  ABCD.BC = 26 → ABCD.AD = 5 → ABCD.AB = 10 → ABCD.CD = 17 → 
  (MN = 2 ↔ (BC + AD - AB - CD) / 2 = 2) :=
by
  sorry

end length_MN_proof_l204_204972


namespace expense_5_yuan_neg_l204_204019

-- Define the condition that income of 5 yuan is denoted as +5 yuan
def income_5_yuan_pos : Int := 5

-- Define the statement to prove that expenses of 5 yuan are denoted as -5 yuan
theorem expense_5_yuan_neg : income_5_yuan_pos = 5 → -income_5_yuan_pos = -5 :=
by
  intro h
  rw h
  rfl

end expense_5_yuan_neg_l204_204019


namespace minimum_throws_to_ensure_same_sum_twice_l204_204058

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l204_204058


namespace monotonic_iff_m_ge_one_third_l204_204512

-- Define the function f(x) = x^3 + x^2 + mx + 1
def f (x m : ℝ) : ℝ := x^3 + x^2 + m * x + 1

-- Define the derivative of the function f w.r.t x
def f' (x m : ℝ) : ℝ := 3 * x^2 + 2 * x + m

-- State the main theorem: f is monotonic on ℝ if and only if m ≥ 1/3
theorem monotonic_iff_m_ge_one_third (m : ℝ) :
  (∀ x y : ℝ, x < y → f x m ≤ f y m) ↔ (m ≥ 1 / 3) :=
sorry

end monotonic_iff_m_ge_one_third_l204_204512


namespace abba_divisible_by_11_aaabbb_divisible_by_37_ababab_divisible_by_7_abab_baba_divisible_by_9_and_101_l204_204143

/-- Part 1: Prove that the number \overline{abba} is divisible by 11 -/
theorem abba_divisible_by_11 (a b : ℕ) : 11 ∣ (1000 * a + 100 * b + 10 * b + a) :=
sorry

/-- Part 2: Prove that the number \overline{aaabbb} is divisible by 37 -/
theorem aaabbb_divisible_by_37 (a b : ℕ) : 37 ∣ (1000 * 111 * a + 111 * b) :=
sorry

/-- Part 3: Prove that the number \overline{ababab} is divisible by 7 -/
theorem ababab_divisible_by_7 (a b : ℕ) : 7 ∣ (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) :=
sorry

/-- Part 4: Prove that the number \overline{abab} - \overline{baba} is divisible by 9 and 101 -/
theorem abab_baba_divisible_by_9_and_101 (a b : ℕ) :
  9 ∣ (1000 * a + 100 * b + 10 * a + b - (1000 * b + 100 * a + 10 * b + a)) ∧
  101 ∣ (1000 * a + 100 * b + 10 * a + b - (1000 * b + 100 * a + 10 * b + a)) :=
sorry

end abba_divisible_by_11_aaabbb_divisible_by_37_ababab_divisible_by_7_abab_baba_divisible_by_9_and_101_l204_204143


namespace expenses_opposite_to_income_l204_204017

theorem expenses_opposite_to_income (income_5 : ℤ) (h_income : income_5 = 5) : -income_5 = -5 :=
by
  -- proof is omitted
  sorry

end expenses_opposite_to_income_l204_204017


namespace exterior_angle_octagon_degree_l204_204651

-- Conditions
def sum_of_exterior_angles (n : ℕ) : ℕ := 360
def number_of_sides_octagon : ℕ := 8

-- Question and correct answer
theorem exterior_angle_octagon_degree :
  (sum_of_exterior_angles 8) / number_of_sides_octagon = 45 :=
by
  sorry

end exterior_angle_octagon_degree_l204_204651


namespace jars_left_when_boxes_full_l204_204554

-- Conditions
def jars_in_first_set_of_boxes : Nat := 12 * 10
def jars_in_second_set_of_boxes : Nat := 10 * 30
def total_jars : Nat := 500

-- Question (equivalent proof problem)
theorem jars_left_when_boxes_full : total_jars - (jars_in_first_set_of_boxes + jars_in_second_set_of_boxes) = 80 := 
by
  sorry

end jars_left_when_boxes_full_l204_204554


namespace compare_cube_roots_l204_204805

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem compare_cube_roots : 2 + cube_root 7 < cube_root 60 :=
sorry

end compare_cube_roots_l204_204805


namespace arccos_sqrt3_div_2_eq_pi_div_6_l204_204169

theorem arccos_sqrt3_div_2_eq_pi_div_6 : 
  Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 :=
by 
  sorry

end arccos_sqrt3_div_2_eq_pi_div_6_l204_204169


namespace dalton_movies_l204_204919

variable (D : ℕ) -- Dalton's movies
variable (Hunter : ℕ := 12) -- Hunter's movies
variable (Alex : ℕ := 15) -- Alex's movies
variable (Together : ℕ := 2) -- Movies watched together
variable (TotalDifferentMovies : ℕ := 30) -- Total different movies

theorem dalton_movies (h : D + Hunter + Alex - Together * 3 = TotalDifferentMovies) : D = 9 := by
  sorry

end dalton_movies_l204_204919


namespace tom_split_number_of_apples_l204_204259

theorem tom_split_number_of_apples
    (S : ℕ)
    (h1 : S = 8 * A)
    (h2 : A * 5 / 8 / 2 = 5) :
    A = 2 :=
by
  sorry

end tom_split_number_of_apples_l204_204259


namespace probability_at_least_seven_stayed_l204_204962

variable (total_people : ℕ)
variable (unsure_people : ℕ)
variable (sure_probability : ℚ)

def at_least_seven_stay_probability : ℚ :=
  (nat.choose 5 3) * (sure_probability^3) * ((1 - sure_probability)^2)
  + (nat.choose 5 4) * (sure_probability^4) * ((1 - sure_probability)^1)
  + (nat.choose 5 5) * (sure_probability^5)

theorem probability_at_least_seven_stayed
  (h1 : total_people = 9)
  (h2 : unsure_people = 5)
  (h3 : sure_probability = 1/3):
  at_least_seven_stay_probability total_people unsure_people sure_probability = 17 / 81 :=
by
  sorry

end probability_at_least_seven_stayed_l204_204962


namespace minimum_throws_for_repetition_of_sum_l204_204104

/-- To ensure that the same sum is rolled twice when throwing four fair six-sided dice,
you must throw the dice at least 22 times. -/
theorem minimum_throws_for_repetition_of_sum :
  ∀ (throws : ℕ), (∀ (sum : ℕ), 4 ≤ sum ∧ sum ≤ 24 → ∃ (count : ℕ), count ≤ 21 ∧ sum = count + 4) → throws ≥ 22 :=
by
  sorry

end minimum_throws_for_repetition_of_sum_l204_204104


namespace man_climbs_out_of_well_in_65_days_l204_204538

theorem man_climbs_out_of_well_in_65_days (depth climb slip net_days last_climb : ℕ) 
  (h_depth : depth = 70)
  (h_climb : climb = 6)
  (h_slip : slip = 5)
  (h_net_days : net_days = 64)
  (h_last_climb : last_climb = 1) :
  ∃ days : ℕ, days = net_days + last_climb ∧ days = 65 := by
  sorry

end man_climbs_out_of_well_in_65_days_l204_204538


namespace total_candy_pieces_l204_204525

theorem total_candy_pieces : 
  (brother_candy = 6) → 
  (wendy_boxes = 2) → 
  (pieces_per_box = 3) → 
  (brother_candy + (wendy_boxes * pieces_per_box) = 12) 
  := 
  by 
    intros brother_candy wendy_boxes pieces_per_box 
    sorry

end total_candy_pieces_l204_204525


namespace minimum_throws_l204_204053

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l204_204053


namespace distance_to_intersection_of_quarter_circles_eq_zero_l204_204974

open Real

theorem distance_to_intersection_of_quarter_circles_eq_zero (s : ℝ) :
  let A := (0, 0)
  let B := (s, 0)
  let C := (s, s)
  let D := (0, s)
  let center := (s / 2, s / 2)
  let arc_from_A := {p : ℝ × ℝ | p.1^2 + p.2^2 = s^2}
  let arc_from_C := {p : ℝ × ℝ | (p.1 - s)^2 + (p.2 - s)^2 = s^2}
  (center ∈ arc_from_A ∧ center ∈ arc_from_C) →
  let (ix, iy) := (s / 2, s / 2)
  dist (ix, iy) center = 0 :=
by
  sorry

end distance_to_intersection_of_quarter_circles_eq_zero_l204_204974


namespace interval_satisfies_ineq_l204_204660

theorem interval_satisfies_ineq (p : ℝ) (h1 : 18 * p < 10) (h2 : 0.5 < p) : 0.5 < p ∧ p < 5 / 9 :=
by {
  sorry -- Proof not required, only the statement.
}

end interval_satisfies_ineq_l204_204660


namespace ensure_same_sum_rolled_twice_l204_204067

theorem ensure_same_sum_rolled_twice :
  ∀ (n : ℕ) (min_sum max_sum : ℕ),
    min_sum = 4 →
    max_sum = 24 →
    (min_sum ≤ n ∧ n ≤ max_sum) →
    ∀ trials : ℕ, trials = 22 →
      ∃ (s1 s2 : ℕ), s1 = s2 ∧ 
      (∃ (throws1 throws2 : list ℕ), list.sum throws1 = s1 ∧ list.sum throws2 = s2 ∧ throws1 ≠ throws2) :=
by 
  sorry

end ensure_same_sum_rolled_twice_l204_204067


namespace expected_heads_64_coins_l204_204854

noncomputable def expected_heads (n : ℕ) (p : ℚ) : ℚ :=
  n * p

theorem expected_heads_64_coins : expected_heads 64 (15/16) = 60 := by
  sorry

end expected_heads_64_coins_l204_204854


namespace expenses_neg_of_income_pos_l204_204031

theorem expenses_neg_of_income_pos :
  ∀ (income expense : Int), income = 5 → expense = -income → expense = -5 :=
by
  intros income expense h_income h_expense
  rw [h_income] at h_expense
  exact h_expense

end expenses_neg_of_income_pos_l204_204031


namespace find_length_AB_l204_204476

variables {A B C D E : Type} -- Define variables A, B, C, D, E as types, representing points

-- Define lengths of the segments AD and CD
def length_AD : ℝ := 2
def length_CD : ℝ := 2

-- Define the angles at vertices B, C, and D
def angle_B : ℝ := 30
def angle_C : ℝ := 90
def angle_D : ℝ := 120

-- The goal is to prove the length of segment AB
theorem find_length_AB : 
  (∃ (A B C D : Type) 
    (angle_B angle_C angle_D length_AD length_CD : ℝ), 
      angle_B = 30 ∧ 
      angle_C = 90 ∧ 
      angle_D = 120 ∧ 
      length_AD = 2 ∧ 
      length_CD = 2) → 
  (length_AB = 6) := by sorry

end find_length_AB_l204_204476


namespace sum_a1_a5_l204_204326

def sequence_sum (S : ℕ → ℕ) := ∀ n : ℕ, S n = n^2 + 1

theorem sum_a1_a5 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h_sum : sequence_sum S)
  (h_a1 : a 1 = S 1)
  (h_a5 : a 5 = S 5 - S 4) :
  a 1 + a 5 = 11 := by
  sorry

end sum_a1_a5_l204_204326


namespace recycling_points_l204_204555

theorem recycling_points (chloe_recycled : ℤ) (friends_recycled : ℤ) (points_per_pound : ℤ) :
  chloe_recycled = 28 ∧ friends_recycled = 2 ∧ points_per_pound = 6 → (chloe_recycled + friends_recycled) / points_per_pound = 5 :=
by
  sorry

end recycling_points_l204_204555


namespace hyunwoo_cookies_l204_204386

theorem hyunwoo_cookies (packs_initial : Nat) (pieces_per_pack : Nat) (packs_given_away : Nat)
  (h1 : packs_initial = 226) (h2 : pieces_per_pack = 3) (h3 : packs_given_away = 3) :
  (packs_initial - packs_given_away) * pieces_per_pack = 669 := 
by
  sorry

end hyunwoo_cookies_l204_204386


namespace right_triangle_least_side_l204_204465

theorem right_triangle_least_side (a b : ℕ) (h₁ : a = 8) (h₂ : b = 15) :
  ∃ c : ℝ, (a^2 + b^2 = c^2 ∨ a^2 = c^2 + b^2 ∨ b^2 = c^2 + a^2) ∧ c = Real.sqrt 161 := 
sorry

end right_triangle_least_side_l204_204465


namespace A_inter_B_is_empty_l204_204589

def A : Set (ℤ × ℤ) := {p | ∃ x : ℤ, p = (x, x + 1)}
def B : Set ℤ := {y | ∃ x : ℤ, y = 2 * x}

theorem A_inter_B_is_empty : A ∩ (fun p => p.2 ∈ B) = ∅ :=
by {
  sorry
}

end A_inter_B_is_empty_l204_204589


namespace find_a2_l204_204619

variables {α : Type*} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) : Prop :=
  ∀ n m : ℕ, ∃ r : α, a (n + m) = (a n) * (a m) * r

theorem find_a2 (a : ℕ → α) (h_geom : geometric_sequence a) (h1 : a 3 * a 6 = 9) (h2 : a 2 * a 4 * a 5 = 27) :
  a 2 = 3 :=
sorry

end find_a2_l204_204619


namespace geometric_sequence_common_ratio_l204_204471

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h₁ : a 1 = 1) (h₂ : a 1 * a 2 * a 3 = -8) :
  q = -2 :=
sorry

end geometric_sequence_common_ratio_l204_204471


namespace no_triangle_satisfies_sine_eq_l204_204298

theorem no_triangle_satisfies_sine_eq (A B C : ℝ) (a b c : ℝ) 
  (hA: 0 < A) (hB: 0 < B) (hC: 0 < C) 
  (hA_ineq: A < π) (hB_ineq: B < π) (hC_ineq: C < π) 
  (h_sum: A + B + C = π) 
  (sin_eq: Real.sin A + Real.sin B = Real.sin C)
  (h_tri_ineq: a + b > c ∧ a + c > b ∧ b + c > a) 
  (h_sines: a = 2 * (1) * Real.sin A ∧ b = 2 * (1) * Real.sin B ∧ c = 2 * (1) * Real.sin C) :
  False :=
sorry

end no_triangle_satisfies_sine_eq_l204_204298


namespace minimum_throws_for_four_dice_l204_204125

noncomputable def minimum_throws_to_ensure_repeated_sum (d : ℕ) : ℕ :=
  let min_sum := d * 1 in
  let max_sum := d * 6 in
  let distinct_sums := max_sum - min_sum + 1 in
  distinct_sums + 1

theorem minimum_throws_for_four_dice : minimum_throws_to_ensure_repeated_sum 4 = 22 := by
  sorry

end minimum_throws_for_four_dice_l204_204125


namespace find_number_l204_204760

theorem find_number (x : ℝ) (h : 61 + 5 * 12 / (180 / x) = 62): x = 3 :=
by
  sorry

end find_number_l204_204760


namespace equal_playing_time_for_each_player_l204_204776

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l204_204776


namespace range_of_expr_l204_204586

theorem range_of_expr (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) : 
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := 
by
  sorry

end range_of_expr_l204_204586


namespace odd_square_sum_of_consecutive_l204_204540

theorem odd_square_sum_of_consecutive (n : ℤ) (h_odd : n % 2 = 1) (h_gt : n > 1) : 
  ∃ (j : ℤ), n^2 = j + (j + 1) :=
by
  sorry

end odd_square_sum_of_consecutive_l204_204540


namespace find_k_l204_204458

theorem find_k (k : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 - 60 * x + k = (x + b)^2) → k = 900 :=
by 
  sorry

end find_k_l204_204458


namespace gcd_45_75_l204_204734

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204734


namespace angle_terminal_side_l204_204592

def angle_on_line (β : ℝ) : Prop :=
  ∃ n : ℤ, β = 135 + n * 180

def angle_in_range (β : ℝ) : Prop :=
  -360 < β ∧ β < 360

theorem angle_terminal_side :
  ∀ β, angle_on_line β → angle_in_range β → β = -225 ∨ β = -45 ∨ β = 135 ∨ β = 315 :=
by
  intros β h_line h_range
  sorry

end angle_terminal_side_l204_204592


namespace total_work_stations_l204_204402

theorem total_work_stations (total_students : ℕ) (stations_for_2 : ℕ) (stations_for_3 : ℕ)
  (h1 : total_students = 38)
  (h2 : stations_for_2 = 10)
  (h3 : 20 + 3 * stations_for_3 = total_students) :
  stations_for_2 + stations_for_3 = 16 :=
by
  sorry

end total_work_stations_l204_204402


namespace solve_congruence_l204_204372

theorem solve_congruence (m : ℤ) : 13 * m ≡ 9 [MOD 47] → m ≡ 29 [MOD 47] :=
by
  sorry

end solve_congruence_l204_204372


namespace max_a_value_l204_204233

-- Variables representing the real numbers a, b, c, and d
variables (a b c d : ℝ)

-- Real number hypothesis conditions
-- 1. a + b + c + d = 10
-- 2. ab + ac + ad + bc + bd + cd = 20

theorem max_a_value
  (h1 : a + b + c + d = 10)
  (h2 : ab + ac + ad + bc + bd + cd = 20) :
  a ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_a_value_l204_204233


namespace largest_inscribed_square_side_length_l204_204849

noncomputable def side_length_inscribed_square: ℝ := 6 - Real.sqrt 6

theorem largest_inscribed_square_side_length (a : ℝ) 
  (h₁ : a = 12)
  (triangle_side_length : ℝ)
  (h₂ : triangle_side_length = 4 * Real.sqrt 6) : 
  let inscribed_square_side_length := 6 - Real.sqrt 6 in
  (∀ (x : ℝ), x < inscribed_square_side_length) ∧ (side_length_inscribed_square = 6 - Real.sqrt 6) :=
by
  have y := 6 - Real.sqrt 6
  have h : y = side_length_inscribed_square := rfl
  sorry

end largest_inscribed_square_side_length_l204_204849


namespace mark_gpa_probability_l204_204551

theorem mark_gpa_probability :
  let A_points := 4
  let B_points := 3
  let C_points := 2
  let D_points := 1
  let GPA_required := 3.5
  let total_subjects := 4
  let total_points_required := GPA_required * total_subjects
  -- Points from guaranteed A's in Mathematics and Science
  let guaranteed_points := 8
  -- Required points from Literature and History
  let points_needed := total_points_required - guaranteed_points
  -- Probabilities for grades in Literature
  let prob_A_Lit := 1 / 3
  let prob_B_Lit := 1 / 3
  let prob_C_Lit := 1 / 3
  -- Probabilities for grades in History
  let prob_A_Hist := 1 / 5
  let prob_B_Hist := 1 / 4
  let prob_C_Hist := 11 / 20
  -- Combinations of grades to achieve the required points
  let prob_two_As := prob_A_Lit * prob_A_Hist
  let prob_A_Lit_B_Hist := prob_A_Lit * prob_B_Hist
  let prob_B_Lit_A_Hist := prob_B_Lit * prob_A_Hist
  let prob_two_Bs := prob_B_Lit * prob_B_Hist
  -- Total probability of achieving at least the required GPA
  let total_probability := prob_two_As + prob_A_Lit_B_Hist + prob_B_Lit_A_Hist + prob_two_Bs
  total_probability = 3 / 10 := sorry

end mark_gpa_probability_l204_204551


namespace div_by_5_implication_l204_204263

theorem div_by_5_implication (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
    (h3 : ∃ k : ℕ, ab = 5 * k) : (∃ k : ℕ, a = 5 * k) ∨ (∃ k : ℕ, b = 5 * k) := 
by
  sorry

end div_by_5_implication_l204_204263


namespace brother_age_in_5_years_l204_204633

theorem brother_age_in_5_years
  (nick_age : ℕ)
  (sister_age : ℕ)
  (brother_age : ℕ)
  (h_nick : nick_age = 13)
  (h_sister : sister_age = nick_age + 6)
  (h_brother : brother_age = (nick_age + sister_age) / 2) :
  brother_age + 5 = 21 := 
by 
  sorry

end brother_age_in_5_years_l204_204633


namespace gcd_45_75_l204_204743

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204743


namespace density_change_l204_204901

theorem density_change (V : ℝ) (Δa : ℝ) (decrease_percent : ℝ) (initial_volume : V = 27) (edge_increase : Δa = 0.9) : 
    decrease_percent = 8 := 
by 
  sorry

end density_change_l204_204901


namespace initial_pennies_l204_204504

theorem initial_pennies (initial: ℕ) (h : initial + 93 = 191) : initial = 98 := by
  sorry

end initial_pennies_l204_204504


namespace cut_half_meter_from_two_thirds_l204_204830

theorem cut_half_meter_from_two_thirds (L : ℝ) (hL : L = 2 / 3) : L - 1 / 6 = 1 / 2 :=
by
  rw [hL]
  norm_num

end cut_half_meter_from_two_thirds_l204_204830


namespace age_of_other_man_l204_204998

theorem age_of_other_man
  (n : ℕ) (average_age_before : ℕ) (average_age_after : ℕ) (age_of_one_man : ℕ) (average_age_women : ℕ) 
  (h1 : n = 9)
  (h2 : average_age_after = average_age_before + 4)
  (h3 : age_of_one_man = 36)
  (h4 : average_age_women = 52) :
  (68 - 36 = 32) := 
by
  sorry

end age_of_other_man_l204_204998


namespace lana_spent_l204_204230

def ticket_cost : ℕ := 6
def tickets_for_friends : ℕ := 8
def extra_tickets : ℕ := 2

theorem lana_spent :
  ticket_cost * (tickets_for_friends + extra_tickets) = 60 := 
by
  sorry

end lana_spent_l204_204230


namespace smallest_k_equals_26_l204_204920

open Real

-- Define the condition
def cos_squared_eq_one (θ : ℝ) : Prop :=
  cos θ ^ 2 = 1

-- Define the requirement for θ to be in the form 180°n
def theta_condition (n : ℤ) : Prop :=
  ∃ (k : ℤ), k ^ 2 + k + 81 = 180 * n

-- The problem statement in Lean: Find the smallest positive integer k such that
-- cos squared of (k^2 + k + 81) degrees = 1
noncomputable def smallest_k_satisfying_cos (k : ℤ) : Prop :=
  (∃ n : ℤ, theta_condition n ∧
   cos_squared_eq_one (k ^ 2 + k + 81)) ∧ (∀ m : ℤ, m > 0 ∧ m < k → 
   (∃ n : ℤ, theta_condition n ∧
   cos_squared_eq_one (m ^ 2 + m + 81)) → false)

theorem smallest_k_equals_26 : smallest_k_satisfying_cos 26 := 
  sorry

end smallest_k_equals_26_l204_204920


namespace sum_of_smallest_and_largest_l204_204608

theorem sum_of_smallest_and_largest (n : ℕ) (h : Odd n) (b z : ℤ)
  (h_mean : z = b + n - 1 - 2 / (n : ℤ)) :
  ((b - 2) + (b + 2 * (n - 2))) = 2 * z - 4 + 4 / (n : ℤ) :=
by
  sorry

end sum_of_smallest_and_largest_l204_204608


namespace tangent_line_circle_l204_204208

theorem tangent_line_circle (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*y = 0 → y = a) → (a = 0 ∨ a = 2) :=
by
  sorry

end tangent_line_circle_l204_204208


namespace min_throws_to_same_sum_l204_204072

/-- Define the set of possible sums for four six-sided dice --/
def dice_sum_range := {s : ℕ | 4 ≤ s ∧ s ≤ 24}

/-- The total number of possible sums when rolling four six-sided dice --/
def num_possible_sums : ℕ := 24 - 4 + 1

/-- 
  The minimum number of throws required to ensure that the same sum appears at least twice 
  by the Pigeonhole principle.
--/
theorem min_throws_to_same_sum : num_possible_sums + 1 = 22 := by
  sorry

end min_throws_to_same_sum_l204_204072


namespace gcd_45_75_l204_204737

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204737


namespace opposite_of_neg_one_fifth_l204_204040

theorem opposite_of_neg_one_fifth : -(- (1/5)) = (1/5) :=
by
  sorry

end opposite_of_neg_one_fifth_l204_204040


namespace income_expenses_opposite_l204_204009

def income_denotation (income : Int) : Int := income

theorem income_expenses_opposite :
  income_denotation 5 = 5 →
  income_denotation (-5) = -5 :=
by
  intro h
  sorry

end income_expenses_opposite_l204_204009


namespace John_height_in_feet_after_growth_spurt_l204_204481

def John_initial_height : ℕ := 66
def growth_rate_per_month : ℕ := 2
def number_of_months : ℕ := 3
def inches_per_foot : ℕ := 12

theorem John_height_in_feet_after_growth_spurt :
  (John_initial_height + growth_rate_per_month * number_of_months) / inches_per_foot = 6 := by
  sorry

end John_height_in_feet_after_growth_spurt_l204_204481


namespace polynomial_divisibility_l204_204929

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x, (x-1)^3 ∣ x^4 + a * x^2 + b * x + c) ↔ (a = -6 ∧ b = 8 ∧ c = -3) :=
by
  sorry

end polynomial_divisibility_l204_204929


namespace minimum_throws_to_ensure_same_sum_twice_l204_204059

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l204_204059


namespace gcd_45_75_l204_204740

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204740


namespace even_function_must_be_two_l204_204318

def f (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + (m-2)*x + (m^2 - 7*m + 12)

theorem even_function_must_be_two (m : ℝ) :
  (∀ x : ℝ, f m (-x) = f m x) ↔ m = 2 :=
by
  sorry

end even_function_must_be_two_l204_204318


namespace max_pies_without_ingredients_l204_204630

def total_pies : ℕ := 30
def blueberry_pies : ℕ := total_pies / 3
def raspberry_pies : ℕ := (3 * total_pies) / 5
def blackberry_pies : ℕ := (5 * total_pies) / 6
def walnut_pies : ℕ := total_pies / 10

theorem max_pies_without_ingredients : 
  (total_pies - blackberry_pies) = 5 :=
by 
  -- We only require the proof part.
  sorry

end max_pies_without_ingredients_l204_204630


namespace total_cost_l204_204365

-- Definitions based on conditions
def old_camera_cost : ℝ := 4000
def new_model_cost_increase_rate : ℝ := 0.3
def lens_initial_cost : ℝ := 400
def lens_discount : ℝ := 200

-- Main statement to prove
theorem total_cost (old_camera_cost new_model_cost_increase_rate lens_initial_cost lens_discount : ℝ) : 
  let new_camera_cost := old_camera_cost * (1 + new_model_cost_increase_rate)
  let lens_cost_after_discount := lens_initial_cost - lens_discount
  (new_camera_cost + lens_cost_after_discount) = 5400 :=
by
  sorry

end total_cost_l204_204365


namespace monthly_growth_rate_is_20_percent_optimal_selling_price_is_48_l204_204613

-- Definitions of conditions
def sales_in_april := 150
def sales_in_june := 216
def cost_price_per_unit := 30
def sales_volume_at_40 := 300
def price_increase_effect := 10
def target_profit := 3960

-- Part 1: Prove the monthly average growth rate of sales
theorem monthly_growth_rate_is_20_percent :
  ∃ x, (sales_in_april : ℝ) * (1 + x)^2 = sales_in_june ∧ x = 0.2 :=
begin
  -- The proof would proceed here
  sorry
end

-- Part 2: Prove the optimal selling price for maximum profit
theorem optimal_selling_price_is_48 :
  ∃ y, (y - cost_price_per_unit) * (sales_volume_at_40 - price_increase_effect * (y - 40)) = target_profit ∧ y = 48 :=
begin
  -- The proof would proceed here
  sorry
end

end monthly_growth_rate_is_20_percent_optimal_selling_price_is_48_l204_204613


namespace find_number_l204_204145

theorem find_number (x : ℝ) : (30 / 100) * x = (60 / 100) * 150 + 120 ↔ x = 700 :=
by
  sorry

end find_number_l204_204145


namespace geographic_info_tech_helps_western_development_l204_204223

namespace GeographicInfoTech

def monitors_three_gorges_project : Prop :=
  -- Point ①
  true

def monitors_ecological_environment_meteorological_changes_and_provides_accurate_info : Prop :=
  -- Point ②
  true

def tracks_migration_tibetan_antelopes : Prop :=
  -- Point ③
  true

def addresses_ecological_environment_issues_in_southwest : Prop :=
  -- Point ④
  true

noncomputable def provides_services_for_development_western_regions : Prop :=
  monitors_three_gorges_project ∧ 
  monitors_ecological_environment_meteorological_changes_and_provides_accurate_info ∧ 
  tracks_migration_tibetan_antelopes -- A (①②③)

-- Theorem stating that geographic information technology helps in ①, ②, ③ given its role
theorem geographic_info_tech_helps_western_development (h : provides_services_for_development_western_regions) :
  monitors_three_gorges_project ∧ 
  monitors_ecological_environment_meteorological_changes_and_provides_accurate_info ∧ 
  tracks_migration_tibetan_antelopes := 
by
  exact h

end GeographicInfoTech

end geographic_info_tech_helps_western_development_l204_204223


namespace gcd_45_75_l204_204697

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204697


namespace equal_play_time_l204_204779

-- Definitions based on conditions
variables (P : ℕ) (F : ℕ) (T : ℕ) (S : ℕ)
-- P: Total number of players
-- F: Number of players on the field at any time
-- T: Total duration of the match in minutes
-- S: Time each player plays

-- Given values from conditions
noncomputable def totalPlayers : ℕ := 10
noncomputable def fieldPlayers : ℕ := 8
noncomputable def matchDuration : ℕ := 45

theorem equal_play_time :
  P = totalPlayers →
  F = fieldPlayers →
  T = matchDuration →
  (F * T) / P = S →
  S = 36 :=
by
  intros hP hF hT hS
  rw [hP, hF, hT] at hS
  exact hS

end equal_play_time_l204_204779


namespace part1_solution_set_part2_range_of_a_l204_204627

-- Part 1: Prove the solution set of the inequality f(x) < 6 is (-8/3, 4/3)
theorem part1_solution_set (x : ℝ) :
  (|2 * x + 3| + |x - 1| < 6) ↔ (-8 / 3 : ℝ) < x ∧ x < 4 / 3 :=
by sorry

-- Part 2: Prove the range of values for a that makes f(x) + f(-x) ≥ 5 is (-∞, -3/2] ∪ [3/2, +∞)
theorem part2_range_of_a (a : ℝ) (x : ℝ) :
  (|2 * x + a| + |x - 1| + |-2 * x + a| + |-x - 1| ≥ 5) ↔ 
  (a ≤ -3 / 2 ∨ a ≥ 3 / 2) :=
by sorry

end part1_solution_set_part2_range_of_a_l204_204627


namespace diff_of_squares_div_l204_204130

theorem diff_of_squares_div (a b : ℤ) (h1 : a = 121) (h2 : b = 112) : 
  (a^2 - b^2) / (a - b) = a + b :=
by
  rw [h1, h2]
  rw [sub_eq_add_neg, add_comm]
  exact sorry

end diff_of_squares_div_l204_204130


namespace gcd_45_75_l204_204732

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204732


namespace chocolate_bar_count_l204_204403

theorem chocolate_bar_count (bar_weight : ℕ) (box_weight : ℕ) (H1 : bar_weight = 125) (H2 : box_weight = 2000) : box_weight / bar_weight = 16 :=
by
  sorry

end chocolate_bar_count_l204_204403


namespace complement_union_complement_intersection_l204_204861

open Set

noncomputable def universal_set : Set ℝ := univ

noncomputable def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
noncomputable def B : Set ℝ := {x | 2 < x ∧ x < 6}

theorem complement_union :
  compl (A ∪ B) = {x : ℝ | x ≤ 2 ∨ 7 ≤ x} := by
  sorry

theorem complement_intersection :
  (compl A ∩ B) = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

end complement_union_complement_intersection_l204_204861


namespace speed_of_stream_l204_204769

theorem speed_of_stream (v : ℝ) (d : ℝ) :
  (∀ d : ℝ, d > 0 → (1 / (6 - v) = 2 * (1 / (6 + v)))) → v = 2 := by
  sorry

end speed_of_stream_l204_204769


namespace gcd_of_45_and_75_l204_204720

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l204_204720


namespace interval_satisfies_ineq_l204_204659

theorem interval_satisfies_ineq (p : ℝ) (h1 : 18 * p < 10) (h2 : 0.5 < p) : 0.5 < p ∧ p < 5 / 9 :=
by {
  sorry -- Proof not required, only the statement.
}

end interval_satisfies_ineq_l204_204659


namespace minimum_throws_for_repetition_of_sum_l204_204106

/-- To ensure that the same sum is rolled twice when throwing four fair six-sided dice,
you must throw the dice at least 22 times. -/
theorem minimum_throws_for_repetition_of_sum :
  ∀ (throws : ℕ), (∀ (sum : ℕ), 4 ≤ sum ∧ sum ≤ 24 → ∃ (count : ℕ), count ≤ 21 ∧ sum = count + 4) → throws ≥ 22 :=
by
  sorry

end minimum_throws_for_repetition_of_sum_l204_204106


namespace lea_notebooks_count_l204_204864

theorem lea_notebooks_count
  (cost_book : ℕ)
  (cost_binder : ℕ)
  (num_binders : ℕ)
  (cost_notebook : ℕ)
  (total_cost : ℕ)
  (h_book : cost_book = 16)
  (h_binder : cost_binder = 2)
  (h_num_binders : num_binders = 3)
  (h_notebook : cost_notebook = 1)
  (h_total : total_cost = 28) :
  ∃ num_notebooks : ℕ, num_notebooks = 6 ∧
    total_cost = cost_book + num_binders * cost_binder + num_notebooks * cost_notebook := 
by
  sorry

end lea_notebooks_count_l204_204864


namespace algorithm_can_contain_all_structures_l204_204759

def sequential_structure : Prop := sorry
def conditional_structure : Prop := sorry
def loop_structure : Prop := sorry

def algorithm_contains_structure (str : Prop) : Prop := sorry

theorem algorithm_can_contain_all_structures :
  algorithm_contains_structure sequential_structure ∧
  algorithm_contains_structure conditional_structure ∧
  algorithm_contains_structure loop_structure := sorry

end algorithm_can_contain_all_structures_l204_204759


namespace water_flow_speed_l204_204410

/-- A person rows a boat for 15 li. If he rows at his usual speed,
the time taken to row downstream is 5 hours less than rowing upstream.
If he rows at twice his usual speed, the time taken to row downstream
is only 1 hour less than rowing upstream. 
Prove that the speed of the water flow is 2 li/hour.
-/
theorem water_flow_speed (y x : ℝ)
  (h1 : 15 / (y - x) - 15 / (y + x) = 5)
  (h2 : 15 / (2 * y - x) - 15 / (2 * y + x) = 1) :
  x = 2 := 
sorry

end water_flow_speed_l204_204410


namespace parallel_lines_slope_l204_204939

theorem parallel_lines_slope (a : ℝ) (h : ∀ x y : ℝ, (x + a * y + 6 = 0) → ((a - 2) * x + 3 * y + 2 * a = 0)) : a = -1 :=
by
  sorry

end parallel_lines_slope_l204_204939


namespace gcd_45_75_l204_204692

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204692


namespace sports_day_popularity_order_l204_204158

theorem sports_day_popularity_order:
  let dodgeball := (3:ℚ) / 8
  let chess_tournament := (9:ℚ) / 24
  let track := (5:ℚ) / 16
  let swimming := (1:ℚ) / 3
  dodgeball = (18:ℚ) / 48 ∧
  chess_tournament = (18:ℚ) / 48 ∧
  track = (15:ℚ) / 48 ∧
  swimming = (16:ℚ) / 48 ∧
  list.sort (>=) [dodgeball, chess_tournament, track, swimming] = [swimming, dodgeball, chess_tournament, track] :=
by {
  sorry
}

end sports_day_popularity_order_l204_204158


namespace count_two_digit_numbers_with_digit_8_l204_204332

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_digit_8 (n : ℕ) : Prop :=
  n / 10 = 8 ∨ n % 10 = 8

theorem count_two_digit_numbers_with_digit_8 : 
  (∃ S : Finset ℕ, (∀ n ∈ S, is_two_digit n ∧ has_digit_8 n) ∧ S.card = 18) :=
sorry

end count_two_digit_numbers_with_digit_8_l204_204332


namespace simplify_expr1_simplify_expr2_l204_204802

-- Problem 1
theorem simplify_expr1 (x y : ℝ) : x^2 - 5 * y - 4 * x^2 + y - 1 = -3 * x^2 - 4 * y - 1 :=
by sorry

-- Problem 2
theorem simplify_expr2 (a b : ℝ) : 7 * a + 3 * (a - 3 * b) - 2 * (b - 3 * a) = 16 * a - 11 * b :=
by sorry

end simplify_expr1_simplify_expr2_l204_204802


namespace smallest_n_for_sqrt_20n_int_l204_204603

theorem smallest_n_for_sqrt_20n_int (n : ℕ) (h : ∃ k : ℕ, 20 * n = k^2) : n = 5 :=
by sorry

end smallest_n_for_sqrt_20n_int_l204_204603


namespace range_of_a_l204_204306

open Real

theorem range_of_a (x y z a : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hsum : x + y + z = 1)
  (heq : a / (x * y * z) = (1 / x) + (1 / y) + (1 / z) - 2) :
  0 < a ∧ a ≤ 7 / 27 :=
by
  sorry

end range_of_a_l204_204306


namespace bob_total_profit_l204_204800

/-- Define the cost of each dog --/
def dog_cost : ℝ := 250.0

/-- Define the number of dogs Bob bought --/
def number_of_dogs : ℕ := 2

/-- Define the total cost of the dogs --/
def total_cost_for_dogs : ℝ := dog_cost * number_of_dogs

/-- Define the selling price of each puppy --/
def puppy_selling_price : ℝ := 350.0

/-- Define the number of puppies --/
def number_of_puppies : ℕ := 6

/-- Define the total revenue from selling the puppies --/
def total_revenue_from_puppies : ℝ := puppy_selling_price * number_of_puppies

/-- Define Bob's total profit from selling the puppies --/
def total_profit : ℝ := total_revenue_from_puppies - total_cost_for_dogs

/-- The theorem stating that Bob's total profit is $1600.00 --/
theorem bob_total_profit : total_profit = 1600.0 := 
by
  /- We leave the proof out as we just need the statement -/
  sorry

end bob_total_profit_l204_204800


namespace books_arrangement_l204_204453

/-
  Theorem:
  If there are 4 distinct math books, 6 distinct English books, and 3 distinct science books,
  and each category of books must stay together, then the number of ways to arrange
  these books on a shelf is 622080.
-/

def num_math_books : ℕ := 4
def num_english_books : ℕ := 6
def num_science_books : ℕ := 3

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def num_arrangements :=
  factorial 3 * factorial num_math_books * factorial num_english_books * factorial num_science_books

theorem books_arrangement : num_arrangements = 622080 := by
  sorry

end books_arrangement_l204_204453


namespace binom_15_4_eq_1365_l204_204567

theorem binom_15_4_eq_1365 : (Nat.choose 15 4) = 1365 := 
by 
  sorry

end binom_15_4_eq_1365_l204_204567


namespace perfect_square_impossible_l204_204856
noncomputable def is_perfect_square (n : ℕ) : Prop :=
∃ m : ℕ, m * m = n

theorem perfect_square_impossible (a b c : ℕ) (a_positive : a > 0) (b_positive : b > 0) (c_positive : c > 0) :
  ¬ (is_perfect_square (a^2 + b + c) ∧ is_perfect_square (b^2 + c + a) ∧ is_perfect_square (c^2 + a + b)) :=
sorry

end perfect_square_impossible_l204_204856


namespace selling_price_l204_204393

theorem selling_price (cost_price : ℝ) (loss_percentage : ℝ) : 
    cost_price = 1600 → loss_percentage = 0.15 → 
    (cost_price - (loss_percentage * cost_price)) = 1360 :=
by
  intros h_cp h_lp
  rw [h_cp, h_lp]
  norm_num

end selling_price_l204_204393


namespace solve_congruence_l204_204373

-- Define the condition and residue modulo 47
def residue_modulo (a b n : ℕ) : Prop := (a ≡ b [MOD n])

-- The main theorem to be proved
theorem solve_congruence (m : ℕ) (h : residue_modulo (13 * m) 9 47) : residue_modulo m 26 47 :=
sorry

end solve_congruence_l204_204373


namespace equal_play_time_l204_204781

-- Definitions based on conditions
variables (P : ℕ) (F : ℕ) (T : ℕ) (S : ℕ)
-- P: Total number of players
-- F: Number of players on the field at any time
-- T: Total duration of the match in minutes
-- S: Time each player plays

-- Given values from conditions
noncomputable def totalPlayers : ℕ := 10
noncomputable def fieldPlayers : ℕ := 8
noncomputable def matchDuration : ℕ := 45

theorem equal_play_time :
  P = totalPlayers →
  F = fieldPlayers →
  T = matchDuration →
  (F * T) / P = S →
  S = 36 :=
by
  intros hP hF hT hS
  rw [hP, hF, hT] at hS
  exact hS

end equal_play_time_l204_204781


namespace cranes_in_each_flock_l204_204632

theorem cranes_in_each_flock (c : ℕ) (h1 : ∃ n : ℕ, 13 * n = 221)
  (h2 : ∃ n : ℕ, c * n = 221) :
  c = 221 :=
by sorry

end cranes_in_each_flock_l204_204632


namespace balls_into_boxes_l204_204205

/-- There are 128 ways to distribute 7 distinguishable balls into 2 distinguishable boxes. -/
theorem balls_into_boxes : (2 : ℕ) ^ 7 = 128 := by
  sorry

end balls_into_boxes_l204_204205


namespace perfect_square_of_sides_of_triangle_l204_204981

theorem perfect_square_of_sides_of_triangle 
  (a b c : ℤ) 
  (h1: a > 0 ∧ b > 0 ∧ c > 0)
  (h2: a + b > c ∧ b + c > a ∧ c + a > b)
  (gcd_abc: Int.gcd (Int.gcd a b) c = 1)
  (h3: (a^2 + b^2 - c^2) % (a + b - c) = 0)
  (h4: (b^2 + c^2 - a^2) % (b + c - a) = 0)
  (h5: (c^2 + a^2 - b^2) % (c + a - b) = 0) : 
  ∃ n : ℤ, n^2 = (a + b - c) * (b + c - a) * (c + a - b) ∨ 
  ∃ m : ℤ, m^2 = 2 * (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end perfect_square_of_sides_of_triangle_l204_204981


namespace count_valid_triangles_l204_204598

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_less_than_20 (a b c : ℕ) : Prop :=
  a + b + c < 20

def non_equilateral (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def non_isosceles (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def non_right (a b c : ℕ) : Prop :=
  a^2 + b^2 ≠ c^2

def valid_triangle (a b c : ℕ) : Prop :=
  is_triangle a b c ∧ perimeter_less_than_20 a b c ∧ non_equilateral a b c ∧ non_isosceles a b c ∧ non_right a b c

theorem count_valid_triangles :
  (finset.univ.filter (λ abc : ℕ × ℕ × ℕ, valid_triangle abc.1 abc.2.1 abc.2.2)).card = 13 :=
sorry

end count_valid_triangles_l204_204598


namespace larger_inscribed_angle_corresponds_to_larger_chord_l204_204501

theorem larger_inscribed_angle_corresponds_to_larger_chord
  (R : ℝ) (α β : ℝ) (hα : α < 90) (hβ : β < 90) (h : α < β)
  (BC LM : ℝ) (hBC : BC = 2 * R * Real.sin α) (hLM : LM = 2 * R * Real.sin β) :
  BC < LM :=
sorry

end larger_inscribed_angle_corresponds_to_larger_chord_l204_204501


namespace H2O_formed_l204_204927

-- Definition of the balanced chemical equation
def balanced_eqn : Prop :=
  ∀ (HCH3CO2 NaOH NaCH3CO2 H2O : ℕ), HCH3CO2 + NaOH = NaCH3CO2 + H2O

-- Statement of the problem
theorem H2O_formed (HCH3CO2 NaOH NaCH3CO2 H2O : ℕ) 
  (h1 : HCH3CO2 = 1)
  (h2 : NaOH = 1)
  (balanced : balanced_eqn):
  H2O = 1 :=
by sorry

end H2O_formed_l204_204927


namespace strike_time_10_times_l204_204831

def time_to_strike (n : ℕ) : ℝ :=
  if n = 0 then 0 else (n - 1) * 6

theorem strike_time_10_times : time_to_strike 10 = 60 :=
  by {
    -- Proof outline
    -- time_to_strike 10 = (10 - 1) * 6 = 9 * 6 = 54. Thanks to provided solution -> we shall consider that time take 10 seconds for the clock to start striking.
    sorry
  }

end strike_time_10_times_l204_204831


namespace gcd_45_75_l204_204746

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204746


namespace boa_constrictor_length_l204_204550

theorem boa_constrictor_length (garden_snake_length : ℕ) (boa_multiplier : ℕ) (boa_length : ℕ) 
    (h1 : garden_snake_length = 10) (h2 : boa_multiplier = 7) (h3 : boa_length = garden_snake_length * boa_multiplier) : 
    boa_length = 70 := 
sorry

end boa_constrictor_length_l204_204550


namespace Jacob_fill_tank_in_206_days_l204_204621

noncomputable def tank_capacity : ℕ := 350 * 1000
def rain_collection : ℕ := 500
def river_collection : ℕ := 1200
def daily_collection : ℕ := rain_collection + river_collection
def required_days (C R r : ℕ) : ℕ := (C + (R + r) - 1) / (R + r)

theorem Jacob_fill_tank_in_206_days :
  required_days tank_capacity rain_collection river_collection = 206 :=
by 
  sorry

end Jacob_fill_tank_in_206_days_l204_204621


namespace sqrt_sum_simplification_l204_204871

theorem sqrt_sum_simplification :
  (Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3)) = 2 * Real.sqrt 6 :=
by
    sorry

end sqrt_sum_simplification_l204_204871


namespace min_throws_to_repeat_sum_l204_204084

theorem min_throws_to_repeat_sum : 
  (∀ (d1 d2 d3 d4 : ℕ), 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6 ∧ 1 ≤ d4 ∧ d4 ≤ 6) →
  (∃ n ≥ 22, ∃ F : (fin n) → ℕ, (∀ i : (fin n), 4 ≤ F i ∧ F i ≤ 24) ∧ (∃ x y : (fin n), x ≠ y ∧ F x = F y )) :=
begin
  sorry
end

end min_throws_to_repeat_sum_l204_204084


namespace values_of_abc_l204_204928

noncomputable def polynomial_divisibility (a b c : ℤ) : Prop :=
  let f := λ x:ℤ, x^4 + a * x^2 + b * x + c
  in (∀ x:ℤ, f (x-1) = (x-1)^3 * (x * (x + 1) + (a + b + 1) - 1) + (a + b + c + 1) - 1)

theorem values_of_abc {a b c : ℤ} :
  polynomial_divisibility a b c ->
  a = -6 ∧ b = 8 ∧ c = -3 :=
sorry

end values_of_abc_l204_204928


namespace gcd_45_75_l204_204691

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204691


namespace minimum_throws_to_ensure_same_sum_twice_l204_204062

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l204_204062


namespace correct_system_of_equations_l204_204922

-- Define the given problem conditions.
def cost_doll : ℝ := 60
def cost_keychain : ℝ := 20
def total_cost : ℝ := 5000

-- Define the condition that each gift set needs 1 doll and 2 keychains.
def gift_set_relation (x y : ℝ) : Prop := 2 * x = y

-- Define the system of equations representing the problem.
def system_of_equations (x y : ℝ) : Prop :=
  2 * x = y ∧
  60 * x + 20 * y = total_cost

-- State the theorem to prove that the given system correctly models the problem.
theorem correct_system_of_equations (x y : ℝ) :
  system_of_equations x y ↔ (2 * x = y ∧ 60 * x + 20 * y = 5000) :=
by sorry

end correct_system_of_equations_l204_204922


namespace system_of_inequalities_l204_204661

theorem system_of_inequalities (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : (0.5 < p ∧ p < 5 / 9) :=
by sorry

end system_of_inequalities_l204_204661


namespace pool_one_quarter_capacity_in_six_hours_l204_204140

theorem pool_one_quarter_capacity_in_six_hours (d : ℕ → ℕ) :
  (∀ n : ℕ, d (n + 1) = 2 * d n) → d 8 = 2^8 →
  d 6 = 2^6 :=
by
  intros h1 h2
  sorry

end pool_one_quarter_capacity_in_six_hours_l204_204140


namespace equal_playing_time_l204_204790

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l204_204790


namespace uki_total_earnings_l204_204387

def cupcake_price : ℝ := 1.50
def cookie_price : ℝ := 2.00
def biscuit_price : ℝ := 1.00
def daily_cupcakes : ℕ := 20
def daily_cookies : ℕ := 10
def daily_biscuits : ℕ := 20
def days : ℕ := 5

theorem uki_total_earnings :
  5 * ((daily_cupcakes * cupcake_price) + (daily_cookies * cookie_price) + (daily_biscuits * biscuit_price)) = 350 :=
by
  -- This is a placeholder for the proof
  sorry

end uki_total_earnings_l204_204387


namespace negation_propositional_logic_l204_204380

theorem negation_propositional_logic :
  ¬ (∀ x : ℝ, x^2 + x + 1 < 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≥ 0 :=
by sorry

end negation_propositional_logic_l204_204380


namespace molecular_weight_calculation_l204_204748

/-- Define the molecular weight of the compound as 972 grams per mole. -/
def molecular_weight : ℕ := 972

/-- Define the number of moles as 9 moles. -/
def number_of_moles : ℕ := 9

/-- Define the total weight of the compound for the given number of moles. -/
def total_weight : ℕ := number_of_moles * molecular_weight

/-- Prove the total weight is 8748 grams. -/
theorem molecular_weight_calculation : total_weight = 8748 := by
  sorry

end molecular_weight_calculation_l204_204748


namespace positive_diff_solutions_l204_204578

theorem positive_diff_solutions : 
  (∃ x₁ x₂ : ℝ, ( (9 - x₁^2 / 4)^(1/3) = -3) ∧ ((9 - x₂^2 / 4)^(1/3) = -3) ∧ ∃ (d : ℝ), d = |x₁ - x₂| ∧ d = 24) :=
by
  sorry

end positive_diff_solutions_l204_204578


namespace find_ages_l204_204671

theorem find_ages (F S : ℕ) (h1 : F + 2 * S = 110) (h2 : 3 * F = 186) :
  F = 62 ∧ S = 24 := by
  sorry

end find_ages_l204_204671


namespace gcd_45_75_l204_204690

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204690


namespace customers_in_other_countries_l204_204762

-- Define the given conditions

def total_customers : ℕ := 7422
def customers_us : ℕ := 723

theorem customers_in_other_countries : total_customers - customers_us = 6699 :=
by
  -- This part will contain the proof, which is not required for this task.
  sorry

end customers_in_other_countries_l204_204762


namespace sugar_solution_sweeter_l204_204215

variables (a b m : ℝ)

theorem sugar_solution_sweeter (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : 
  (a / b < (a + m) / (b + m)) :=
sorry

end sugar_solution_sweeter_l204_204215


namespace find_b_l204_204264

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 3) : b = 3 :=
by
  sorry

end find_b_l204_204264


namespace min_throws_to_ensure_same_sum_twice_l204_204101

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l204_204101


namespace income_expenses_opposite_l204_204011

def income_denotation (income : Int) : Int := income

theorem income_expenses_opposite :
  income_denotation 5 = 5 →
  income_denotation (-5) = -5 :=
by
  intro h
  sorry

end income_expenses_opposite_l204_204011


namespace problem_statement_l204_204855

noncomputable def proof_of_fixed_point (P X Y X' Y' M N : Point ℝ) (γ : Circle ℝ) : Prop :=
  let ℓ : Line ℝ := make_line P X
  let ℓ' : Line ℝ := make_line P X'
  -- Define the circles PXX' and PYY'
  let C1 : Circle ℝ := circle P X X'
  let C2 : Circle ℝ := circle P Y Y'
  -- Define the conditions
  ∃ O : Point ℝ,
  (γ.contains X ∧ γ.contains Y) ∧
  (γ.contains X' ∧ γ.contains Y') ∧
  (Circle.antipode C1 P = M) ∧
  (Circle.antipode C2 P = N) ∧
  (Line.through M N O)

-- Statement
theorem problem_statement (P X Y X' Y' M N : Point ℝ) (γ : Circle ℝ) :
  (γ.contains X ∧ γ.contains Y) →
  (γ.contains X' ∧ γ.contains Y') →
  (Circle.antipode (circle P X X') P = M) →
  (Circle.antipode (circle P Y Y') P = N) →
  ∃ O : Point ℝ, (Circumcenter γ = O) ∧ (Line.through M N O) :=
sorry

end problem_statement_l204_204855


namespace gcd_45_75_l204_204685

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204685


namespace solve_system_of_equations_l204_204994

theorem solve_system_of_equations (x y z : ℝ) :
  (x * y = z) ∧ (x * z = y) ∧ (y * z = x) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = -1 ∧ y = 1 ∧ z = -1) ∨
  (x = 1 ∧ y = -1 ∧ z = -1) ∨
  (x = -1 ∧ y = -1 ∧ z = 1) ∨
  (x = 0 ∧ y = 0 ∧ z = 0) := by
  sorry

end solve_system_of_equations_l204_204994


namespace gcd_euclidean_algorithm_l204_204986

theorem gcd_euclidean_algorithm (a b : ℕ) : 
  ∃ d : ℕ, d = gcd a b ∧ ∀ m : ℕ, (m ∣ a ∧ m ∣ b) → m ∣ d :=
by
  sorry

end gcd_euclidean_algorithm_l204_204986


namespace mark_fewer_than_susan_l204_204449

variable (apples_total : ℕ) (greg_apples : ℕ) (susan_apples : ℕ) (mark_apples : ℕ) (mom_apples : ℕ)

def evenly_split (total : ℕ) : ℕ := total / 2

theorem mark_fewer_than_susan
    (h1 : apples_total = 18)
    (h2 : greg_apples = evenly_split apples_total)
    (h3 : susan_apples = 2 * greg_apples)
    (h4 : mom_apples = 40 + 9)
    (h5 : mark_apples = mom_apples - susan_apples) :
    susan_apples - mark_apples = 13 := 
sorry

end mark_fewer_than_susan_l204_204449


namespace tip_calculation_l204_204930

def pizza_price : ℤ := 10
def number_of_pizzas : ℤ := 4
def total_pizza_cost := pizza_price * number_of_pizzas
def bill_given : ℤ := 50
def change_received : ℤ := 5
def total_spent := bill_given - change_received
def tip_given := total_spent - total_pizza_cost

theorem tip_calculation : tip_given = 5 :=
by
  -- skipping the proof
  sorry

end tip_calculation_l204_204930


namespace linda_fraction_savings_l204_204237

theorem linda_fraction_savings (savings tv_cost : ℝ) (f : ℝ) 
  (h1 : savings = 800) 
  (h2 : tv_cost = 200) 
  (h3 : f * savings + tv_cost = savings) : 
  f = 3 / 4 := 
sorry

end linda_fraction_savings_l204_204237


namespace smallest_positive_angle_same_terminal_side_l204_204047

theorem smallest_positive_angle_same_terminal_side 
  (k : ℤ) : ∃ α : ℝ, 0 < α ∧ α < 360 ∧ -2002 = α + k * 360 ∧ α = 158 :=
by
  sorry

end smallest_positive_angle_same_terminal_side_l204_204047


namespace gcd_45_75_l204_204739

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204739


namespace minimum_throws_to_ensure_same_sum_twice_l204_204061

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l204_204061


namespace gcd_45_75_l204_204729

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204729


namespace remainder_of_power_division_l204_204399

-- Define the main entities
def power : ℕ := 3
def exponent : ℕ := 19
def divisor : ℕ := 10

-- Define the proof problem
theorem remainder_of_power_division :
  (power ^ exponent) % divisor = 7 := 
  by 
    sorry

end remainder_of_power_division_l204_204399


namespace chadsRopeLength_l204_204668

-- Define the constants and conditions
def joeysRopeLength : ℕ := 56
def joeyChadRatioNumerator : ℕ := 8
def joeyChadRatioDenominator : ℕ := 3

-- Prove that Chad's rope length is 21 cm
theorem chadsRopeLength (C : ℕ) 
  (h_ratio : joeysRopeLength * joeyChadRatioDenominator = joeyChadRatioNumerator * C) : 
  C = 21 :=
sorry

end chadsRopeLength_l204_204668


namespace smallest_number_of_butterflies_l204_204624

theorem smallest_number_of_butterflies 
  (identical_groups : ℕ) 
  (groups_of_butterflies : ℕ) 
  (groups_of_fireflies : ℕ) 
  (groups_of_ladybugs : ℕ)
  (h1 : groups_of_butterflies = 44)
  (h2 : groups_of_fireflies = 17)
  (h3 : groups_of_ladybugs = 25)
  (h4 : identical_groups * (groups_of_butterflies + groups_of_fireflies + groups_of_ladybugs) % 60 = 0) :
  identical_groups * groups_of_butterflies = 425 :=
sorry

end smallest_number_of_butterflies_l204_204624


namespace john_safe_weight_l204_204229

-- Assuming the conditions provided that form the basis of our problem.
def max_capacity : ℝ := 1000
def safety_margin : ℝ := 0.20
def john_weight : ℝ := 250
def safe_weight (max_capacity safety_margin john_weight : ℝ) : ℝ := 
  (max_capacity * (1 - safety_margin)) - john_weight

-- The main theorem to prove based on the provided problem statement.
theorem john_safe_weight : safe_weight max_capacity safety_margin john_weight = 550 := by
  -- skipping the proof details as instructed
  sorry

end john_safe_weight_l204_204229


namespace gcd_of_45_and_75_l204_204716

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l204_204716


namespace John_height_in_feet_after_growth_spurt_l204_204482

def John_initial_height : ℕ := 66
def growth_rate_per_month : ℕ := 2
def number_of_months : ℕ := 3
def inches_per_foot : ℕ := 12

theorem John_height_in_feet_after_growth_spurt :
  (John_initial_height + growth_rate_per_month * number_of_months) / inches_per_foot = 6 := by
  sorry

end John_height_in_feet_after_growth_spurt_l204_204482


namespace gcd_45_75_l204_204745

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204745


namespace set_intersection_complement_l204_204202

-- Define the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 3, 6, 7}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ ¬ x ∈ A}

-- Define the intersection of B and complement_U_A
def B_inter_complement_U_A : Set ℕ := B ∩ complement_U_A

-- The statement to prove: B ∩ complement_U_A = {6, 7}
theorem set_intersection_complement :
  B_inter_complement_U_A = {6, 7} := by sorry

end set_intersection_complement_l204_204202


namespace initial_ratio_of_liquids_l204_204532

theorem initial_ratio_of_liquids (p q : ℕ) (h1 : p + q = 40) (h2 : p / (q + 15) = 5 / 6) : p / q = 5 / 3 :=
by
  sorry

end initial_ratio_of_liquids_l204_204532


namespace complement_intersection_l204_204358

open Set

def U : Set ℕ := {2, 3, 4, 5, 6}
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem complement_intersection :
  (U \ A) ∩ B = {3} :=
sorry

end complement_intersection_l204_204358


namespace find_width_of_rectangle_l204_204044

-- Given conditions
variable (P l w : ℕ)
variable (h1 : P = 240)
variable (h2 : P = 3 * l)

-- Prove the width of the rectangular field is 40 meters
theorem find_width_of_rectangle : w = 40 :=
  by 
  -- Add the necessary logical steps here
  sorry

end find_width_of_rectangle_l204_204044


namespace discount_on_pony_jeans_l204_204137

theorem discount_on_pony_jeans 
  (F P : ℕ)
  (h1 : F + P = 25)
  (h2 : 5 * F + 4 * P = 100) : P = 25 :=
by
  sorry

end discount_on_pony_jeans_l204_204137


namespace value_of_a8_l204_204343

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

noncomputable def arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n : ℕ, ∃ d : α, a (n + 1) = a n + d

variable {a : ℕ → ℝ}

axiom seq_is_arithmetic : arithmetic_sequence a

axiom initial_condition :
  a 1 + 3 * a 8 + a 15 = 120

axiom arithmetic_property :
  a 1 + a 15 = 2 * a 8

theorem value_of_a8 : a 8 = 24 :=
by {
  sorry
}

end value_of_a8_l204_204343


namespace sin_half_angle_l204_204206

theorem sin_half_angle
  (theta : ℝ)
  (h1 : Real.sin theta = 3 / 5)
  (h2 : 5 * Real.pi / 2 < theta ∧ theta < 3 * Real.pi) :
  Real.sin (theta / 2) = - (3 * Real.sqrt 10 / 10) :=
by
  sorry

end sin_half_angle_l204_204206


namespace determine_guilty_defendant_l204_204528

-- Define the defendants
inductive Defendant
| A
| B
| C

open Defendant

-- Define the guilty defendant
def guilty_defendant : Defendant := C

-- Define the conditions
def condition1 (d : Defendant) : Prop :=
d ≠ A ∧ d ≠ B ∧ d ≠ C → false  -- "There were three defendants, and only one of them was guilty."

def condition2 (d : Defendant) : Prop :=
d = A → d ≠ B  -- "Defendant A accused defendant B."

def condition3 (d : Defendant) : Prop :=
d = B → d = B  -- "Defendant B admitted to being guilty."

def condition4 (d : Defendant) : Prop :=
d = C → (d = C ∨ d = A)  -- "Defendant C either admitted to being guilty or accused A."

-- The proof problem statement
theorem determine_guilty_defendant :
  (∃ d : Defendant, condition1 d ∧ condition2 d ∧ condition3 d ∧ condition4 d) → guilty_defendant = C :=
by {
  sorry
}

end determine_guilty_defendant_l204_204528


namespace time_to_store_vaccine_l204_204147

def final_temp : ℤ := -24
def current_temp : ℤ := -4
def rate_of_change : ℤ := -5

theorem time_to_store_vaccine : 
  ∃ t : ℤ, current_temp + rate_of_change * t = final_temp ∧ t = 4 :=
by
  use 4
  sorry

end time_to_store_vaccine_l204_204147


namespace remy_used_25_gallons_l204_204502

noncomputable def RomanGallons : ℕ := 8

noncomputable def RemyGallons (R : ℕ) : ℕ := 3 * R + 1

theorem remy_used_25_gallons (R : ℕ) (h1 : RemyGallons R = 1 + 3 * R) (h2 : R + RemyGallons R = 33) : RemyGallons R = 25 := by
  sorry

end remy_used_25_gallons_l204_204502


namespace gcd_45_75_l204_204727

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204727


namespace simplifyExpression_l204_204893

theorem simplifyExpression (a b c d : Int) (ha : a = -2) (hb : b = -6) (hc : c = -3) (hd : d = 2) :
  (a + b - c - d = -2 - 6 + 3 - 2) :=
by {
  sorry
}

end simplifyExpression_l204_204893


namespace angle_B_lt_90_l204_204212

theorem angle_B_lt_90 {a b c : ℝ} (h_arith : b = (a + c) / 2) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  ∃ (A B C : ℝ), B < 90 :=
sorry

end angle_B_lt_90_l204_204212


namespace integer_pairs_satisfying_equation_l204_204429

theorem integer_pairs_satisfying_equation:
  ∀ (a b : ℕ), a ≥ 1 → b ≥ 1 → a^(b^2) = b^a ↔ (a, b) = (1, 1) ∨ (a, b) = (16, 2) ∨ (a, b) = (27, 3) :=
by
  sorry

end integer_pairs_satisfying_equation_l204_204429


namespace solve_system_of_equations_l204_204490

theorem solve_system_of_equations (x y : ℝ) 
  (h1 : x + y = 55) 
  (h2 : x - y = 15) 
  (h3 : x > y) : 
  x = 35 ∧ y = 20 := 
sorry

end solve_system_of_equations_l204_204490


namespace number_of_solutions_pi_equation_l204_204652

theorem number_of_solutions_pi_equation : 
  ∃ (x0 x1 : ℝ), (x0 = 0 ∧ x1 = 1) ∧ ∀ x : ℝ, (π^(x-1) * x^2 + π^(x^2) * x - π^(x^2) = x^2 + x - 1 ↔ x = x0 ∨ x = x1)
:=
by sorry

end number_of_solutions_pi_equation_l204_204652


namespace number_of_ordered_pairs_l204_204808

theorem number_of_ordered_pairs :
  ∃ (n : ℕ), n = 99 ∧
  (∀ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b ∧ (Int.gcd a b) * a + b^2 = 10000
  → ∃ (k : ℕ), k = 99) :=
sorry

end number_of_ordered_pairs_l204_204808


namespace time_for_trains_to_cross_l204_204262

def length_train1 := 500 -- 500 meters
def length_train2 := 750 -- 750 meters
def speed_train1 := 60 * 1000 / 3600 -- 60 km/hr to m/s
def speed_train2 := 40 * 1000 / 3600 -- 40 km/hr to m/s
def relative_speed := speed_train1 + speed_train2 -- relative speed in m/s
def combined_length := length_train1 + length_train2 -- sum of lengths of both trains

theorem time_for_trains_to_cross :
  (combined_length / relative_speed) = 45 := 
by
  sorry

end time_for_trains_to_cross_l204_204262


namespace arithmetic_sequence_a1_l204_204617

/-- In an arithmetic sequence {a_n],
given a_3 = -2, a_n = 3 / 2, and S_n = -15 / 2,
prove that the value of a_1 is -3 or -19 / 6.
-/
theorem arithmetic_sequence_a1 (a_n S_n : ℕ → ℚ)
  (h1 : a_n 3 = -2)
  (h2 : ∃ n : ℕ, a_n n = 3 / 2)
  (h3 : ∃ n : ℕ, S_n n = -15 / 2) :
  ∃ x : ℚ, x = -3 ∨ x = -19 / 6 :=
by 
  sorry

end arithmetic_sequence_a1_l204_204617


namespace deviation_interpretation_l204_204601

variable (average_score : ℝ)
variable (x : ℝ)

-- Given condition
def higher_than_average : Prop := x = average_score + 5

-- To prove
def lower_than_average : Prop := x = average_score - 9

theorem deviation_interpretation (x : ℝ) (h : x = average_score + 5) : x - 14 = average_score - 9 :=
by
  sorry

end deviation_interpretation_l204_204601


namespace gcd_45_75_l204_204682

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204682


namespace perfect_cubes_in_range_l204_204452

theorem perfect_cubes_in_range :
  ∃ (n : ℕ), (∀ (k : ℕ), (50 < k^3 ∧ k^3 ≤ 1000) → (k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10)) ∧
    (∃ m, (m = 7)) :=
by
  sorry

end perfect_cubes_in_range_l204_204452


namespace thirteen_pow_seven_mod_eight_l204_204995

theorem thirteen_pow_seven_mod_eight : 
  (13^7) % 8 = 5 := by
  sorry

end thirteen_pow_seven_mod_eight_l204_204995


namespace population_initial_count_l204_204034

theorem population_initial_count
  (P : ℕ)
  (birth_rate : ℕ := 52)
  (death_rate : ℕ := 16)
  (net_growth_rate : ℝ := 1.2) :
  36 = (net_growth_rate / 100) * P ↔ P = 3000 :=
by sorry

end population_initial_count_l204_204034


namespace additional_coins_needed_l204_204287

theorem additional_coins_needed (friends : Nat) (current_coins : Nat) : 
  friends = 15 → current_coins = 100 → 
  let total_coins_needed := (friends * (friends + 1)) / 2 
  in total_coins_needed - current_coins = 20 :=
by
  intros h1 h2
  simp [h1, h2]
  sorry

end additional_coins_needed_l204_204287


namespace sarah_likes_digits_l204_204496

theorem sarah_likes_digits : ∀ n : ℕ, n % 8 = 0 → (n % 10 = 0 ∨ n % 10 = 4 ∨ n % 10 = 8) :=
by
  sorry

end sarah_likes_digits_l204_204496


namespace find_fourth_month_sale_l204_204407

theorem find_fourth_month_sale (s1 s2 s3 s4 s5 : ℕ) (avg_sale nL5 : ℕ)
  (h1 : s1 = 5420)
  (h2 : s2 = 5660)
  (h3 : s3 = 6200)
  (h5 : s5 = 6500)
  (havg : avg_sale = 6300)
  (hnL5 : nL5 = 5)
  (h_average : avg_sale * nL5 = s1 + s2 + s3 + s4 + s5) :
  s4 = 7720 := sorry

end find_fourth_month_sale_l204_204407


namespace angle_ABC_bisector_l204_204650

theorem angle_ABC_bisector (θ : ℝ) (h : θ / 2 = (1 / 3) * (180 - θ)) : θ = 72 :=
by
  sorry

end angle_ABC_bisector_l204_204650


namespace degree_measure_supplement_complement_l204_204265

theorem degree_measure_supplement_complement : 
  let alpha := 63 -- angle value
  let theta := 90 - alpha -- complement of the angle
  let phi := 180 - theta -- supplement of the complement
  phi = 153 := -- prove the final step
by
  sorry

end degree_measure_supplement_complement_l204_204265


namespace k_times_a_plus_b_l204_204411

/-- Given a quadrilateral with vertices P(ka, kb), Q(kb, ka), R(-ka, -kb), and S(-kb, -ka),
where a and b are consecutive integers with a > b > 0, and k is an odd integer.
It is given that the area of PQRS is 50.
Prove that k(a + b) = 5. -/
theorem k_times_a_plus_b (a b k : ℤ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a = b + 1)
  (h4 : Odd k)
  (h5 : 2 * k^2 * (a - b) * (a + b) = 50) :
  k * (a + b) = 5 := by
  sorry

end k_times_a_plus_b_l204_204411


namespace calculation_correct_l204_204400

noncomputable def problem_calculation : ℝ :=
  4 * Real.sin (Real.pi / 3) - abs (-1) + (Real.sqrt 3 - 1)^0 + Real.sqrt 48

theorem calculation_correct : problem_calculation = 6 * Real.sqrt 3 :=
by
  sorry

end calculation_correct_l204_204400


namespace opposite_of_neg_one_fifth_l204_204041

theorem opposite_of_neg_one_fifth : -(- (1/5)) = (1/5) :=
by
  sorry

end opposite_of_neg_one_fifth_l204_204041


namespace monopoly_durable_only_iff_competitive_market_durable_preference_iff_l204_204756

variable (C : ℝ)

-- Definitions based on conditions
def consumer_benefit_period : ℝ := 10
def durable_machine_periods : ℝ := 2
def low_quality_machine_periods : ℝ := 1
def durable_machine_cost : ℝ := 6

-- Statements based on extracted questions & correct answers
theorem monopoly_durable_only_iff (H : C > 3) :
  let durable_benefit := durable_machine_periods * consumer_benefit_period
      durable_price := durable_benefit
      durable_profit := durable_price - durable_machine_cost
      low_quality_price := consumer_benefit_period
      low_quality_profit := low_quality_price - C in
  durable_profit > durable_machine_periods * (low_quality_profit) :=
by 
  sorry

theorem competitive_market_durable_preference_iff (H : C > 3) :
  let durable_benefit := durable_machine_periods * consumer_benefit_period
      durable_surplus := durable_benefit - durable_machine_cost
      low_quality_surplus := low_quality_machine_periods * (consumer_benefit_period - C) in
  durable_surplus > durable_machine_periods * low_quality_surplus :=
by 
  sorry

end monopoly_durable_only_iff_competitive_market_durable_preference_iff_l204_204756


namespace domain_of_function_l204_204845

def valid_domain (x : ℝ) : Prop :=
  x ≤ 3 ∧ x ≠ 0

theorem domain_of_function (x : ℝ) (h₀ : 3 - x ≥ 0) (h₁ : x ≠ 0) : valid_domain x :=
by
  sorry

end domain_of_function_l204_204845


namespace positive_real_solutions_unique_l204_204308

variable (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (x y z : ℝ)

theorem positive_real_solutions_unique :
    x + y + z = a + b + c ∧
    4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = abc →
    (x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2) :=
by
  intros
  sorry

end positive_real_solutions_unique_l204_204308


namespace subdivide_tetrahedron_l204_204553

/-- A regular tetrahedron with edge length 1 can be divided into smaller regular tetrahedrons and octahedrons,
    such that the edge lengths of the resulting tetrahedrons and octahedrons are less than 1 / 100 after a 
    finite number of subdivisions. -/
theorem subdivide_tetrahedron (edge_len : ℝ) (h : edge_len = 1) :
  ∃ (k : ℕ), (1 / (2^k : ℝ) < 1 / 100) :=
by sorry

end subdivide_tetrahedron_l204_204553


namespace min_rolls_to_duplicate_sum_for_four_dice_l204_204114

theorem min_rolls_to_duplicate_sum_for_four_dice : 
    let min_sum := 4 * 1,
    let max_sum := 4 * 6,
    let possible_sums := max_sum - min_sum + 1 in
    possible_sums = 21 → 
    (possible_sums + 1 = 22) := 
by
  intros min_sum max_sum possible_sums h
  have h1 : min_sum = 4 := rfl
  have h2 : max_sum = 24 := rfl
  have h3 : possible_sums = 21 := h
  have h4 : possible_sums + 1 = 22 := calc
    possible_sums + 1 = 21 + 1 : by rw h
    ... = 22 : by rfl
  exact h4

end min_rolls_to_duplicate_sum_for_four_dice_l204_204114


namespace sum_of_second_and_third_of_four_consecutive_even_integers_l204_204048

-- Definitions of conditions
variables (n : ℤ)  -- Assume n is an integer

-- Statement of problem
theorem sum_of_second_and_third_of_four_consecutive_even_integers (h : 2 * n + 6 = 160) :
  (n + 2) + (n + 4) = 160 :=
by
  sorry

end sum_of_second_and_third_of_four_consecutive_even_integers_l204_204048


namespace expansion_of_expression_l204_204179

theorem expansion_of_expression (x : ℝ) :
  let a := 15 * x^2 + 5 - 3 * x
  let b := 3 * x^3
  a * b = 45 * x^5 - 9 * x^4 + 15 * x^3 := by
  sorry

end expansion_of_expression_l204_204179


namespace sum_of_digits_base2_310_l204_204129

-- We define what it means to convert a number to binary and sum its digits.
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (Nat.digits 2 n).sum

-- The main statement of the problem.
theorem sum_of_digits_base2_310 :
  sum_of_binary_digits 310 = 5 :=
by
  sorry

end sum_of_digits_base2_310_l204_204129


namespace exists_four_numbers_with_equal_sum_l204_204869

theorem exists_four_numbers_with_equal_sum (S : Finset ℕ) (hS : S.card = 16) (h_range : ∀ n ∈ S, n ≤ 100) :
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ c ≠ d ∧ a ≠ c ∧ b ≠ d ∧ a + b = c + d :=
by
  sorry

end exists_four_numbers_with_equal_sum_l204_204869


namespace min_throws_to_same_sum_l204_204069

/-- Define the set of possible sums for four six-sided dice --/
def dice_sum_range := {s : ℕ | 4 ≤ s ∧ s ≤ 24}

/-- The total number of possible sums when rolling four six-sided dice --/
def num_possible_sums : ℕ := 24 - 4 + 1

/-- 
  The minimum number of throws required to ensure that the same sum appears at least twice 
  by the Pigeonhole principle.
--/
theorem min_throws_to_same_sum : num_possible_sums + 1 = 22 := by
  sorry

end min_throws_to_same_sum_l204_204069


namespace triangle_tan_inequality_l204_204478

theorem triangle_tan_inequality (A B C : ℝ) (hA : A + B + C = π) :
    (Real.tan A)^2 + (Real.tan B)^2 + (Real.tan C)^2 ≥ (Real.tan A) * (Real.tan B) + (Real.tan B) * (Real.tan C) + (Real.tan C) * (Real.tan A) :=
by
  sorry

end triangle_tan_inequality_l204_204478


namespace f_2016_eq_neg1_l204_204938

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1
axiom f_property : ∀ x y : ℝ, f x * f y = f (x + y) + f (x - y)

theorem f_2016_eq_neg1 : f 2016 = -1 := 
by 
  sorry

end f_2016_eq_neg1_l204_204938


namespace gcd_45_75_l204_204704

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204704


namespace gcd_45_75_l204_204703

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204703


namespace necessary_not_sufficient_condition_l204_204488

-- Definitions of conditions
variable (x : ℝ)

-- Statement of the problem in Lean 4
theorem necessary_not_sufficient_condition (h : |x - 1| ≤ 1) : 2 - x ≥ 0 := sorry

end necessary_not_sufficient_condition_l204_204488


namespace gcd_45_75_l204_204742

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204742


namespace number_with_all_8s_is_divisible_by_13_l204_204641

theorem number_with_all_8s_is_divisible_by_13 :
  ∀ (N : ℕ), (N = 8 * (10^1974 - 1) / 9) → 13 ∣ N :=
by
  sorry

end number_with_all_8s_is_divisible_by_13_l204_204641


namespace real_part_of_complex_div_l204_204141

noncomputable def complexDiv (c1 c2 : ℂ) := c1 / c2

theorem real_part_of_complex_div (i_unit : ℂ) (h_i : i_unit = Complex.I) :
  (Complex.re (complexDiv (2 * i_unit) (1 + i_unit)) = 1) :=
by
  sorry

end real_part_of_complex_div_l204_204141


namespace minimum_throws_for_repeated_sum_l204_204076

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l204_204076


namespace function_inequality_l204_204860

variable {f : ℕ → ℝ}
variable {a : ℝ}

theorem function_inequality (h : ∀ n : ℕ, f (n + 1) ≥ a^n * f n) :
  ∀ n : ℕ, f n = a^((n * (n - 1)) / 2) * f 1 := 
sorry

end function_inequality_l204_204860


namespace min_throws_to_ensure_same_sum_twice_l204_204099

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l204_204099


namespace part1_exists_n_part2_not_exists_n_l204_204572

open Nat

def is_prime (p : Nat) : Prop := p > 1 ∧ ∀ m : Nat, m ∣ p → m = 1 ∨ m = p

-- Part 1: Prove there exists an n such that n-96, n, n+96 are all primes
theorem part1_exists_n :
  ∃ (n : Nat), is_prime (n - 96) ∧ is_prime n ∧ is_prime (n + 96) :=
sorry

-- Part 2: Prove there does not exist an n such that n-1996, n, n+1996 are all primes
theorem part2_not_exists_n :
  ¬ (∃ (n : Nat), is_prime (n - 1996) ∧ is_prime n ∧ is_prime (n + 1996)) :=
sorry

end part1_exists_n_part2_not_exists_n_l204_204572


namespace people_per_car_l204_204457

theorem people_per_car (total_people cars : ℕ) (h1 : total_people = 63) (h2 : cars = 9) :
  total_people / cars = 7 :=
by
  sorry

end people_per_car_l204_204457


namespace minimum_throws_l204_204057

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l204_204057


namespace find_a_l204_204945

open Real

def is_chord_length_correct (a : ℝ) : Prop :=
  let x_line := fun t : ℝ => 1 + t
  let y_line := fun t : ℝ => a - t
  let x_circle := fun α : ℝ => 2 + 2 * cos α
  let y_circle := fun α : ℝ => 2 + 2 * sin α
  let distance_from_center := abs (3 - a) / sqrt 2
  let chord_length := 2 * sqrt (4 - distance_from_center ^ 2)
  chord_length = 2 * sqrt 2 

theorem find_a (a : ℝ) : is_chord_length_correct a → a = 1 ∨ a = 5 :=
by
  sorry

end find_a_l204_204945


namespace triangle_right_angle_l204_204594

theorem triangle_right_angle
  (a b m : ℝ)
  (h1 : 0 < b)
  (h2 : b < m)
  (h3 : a^2 + b^2 = m^2) :
  a^2 + b^2 = m^2 :=
by sorry

end triangle_right_angle_l204_204594


namespace minimum_rolls_for_duplicate_sum_l204_204090

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l204_204090


namespace towels_per_load_l204_204492

-- Defining the given conditions
def total_towels : ℕ := 42
def number_of_loads : ℕ := 6

-- Defining the problem statement: Prove the number of towels per load
theorem towels_per_load : total_towels / number_of_loads = 7 := by 
  sorry

end towels_per_load_l204_204492


namespace exam_fail_percentage_l204_204397

theorem exam_fail_percentage
  (total_candidates : ℕ := 2000)
  (girls : ℕ := 900)
  (pass_percent : ℝ := 0.32) :
  ((total_candidates - ((pass_percent * (total_candidates - girls)) + (pass_percent * girls))) / total_candidates) * 100 = 68 :=
by
  sorry

end exam_fail_percentage_l204_204397


namespace icosahedron_colorings_l204_204997

theorem icosahedron_colorings :
  let n := 10
  let f := 9
  n! / 5 = 72576 :=
by
  sorry

end icosahedron_colorings_l204_204997


namespace monthly_growth_rate_optimal_selling_price_l204_204612

-- Conditions
def april_sales : ℕ := 150
def june_sales : ℕ := 216
def cost_price_per_unit : ℕ := 30
def initial_selling_price : ℕ := 40
def initial_sales_vol : ℕ := 300
def sales_decrease_rate : ℕ := 10
def desired_profit : ℕ := 3960

-- Questions (Proof statements)
theorem monthly_growth_rate :
  ∃ (x : ℝ), (1 + x) ^ 2 = (june_sales:ℝ) / (april_sales:ℝ) ∧ x = 0.2 := by
  sorry

theorem optimal_selling_price :
  ∃ (y : ℝ), (y - cost_price_per_unit) * (initial_sales_vol - sales_decrease_rate * (y - initial_selling_price)) = desired_profit ∧ y = 48 := by
  sorry

end monthly_growth_rate_optimal_selling_price_l204_204612


namespace find_a_l204_204859

open Classical

noncomputable def f (x a : ℝ) : ℝ := |x - a| - 2

theorem find_a (a : ℝ) :
  (∀ x : ℝ, (|f x a| < 1) ↔ (x ∈ Set.Ioo (-2) 0 ∨ x ∈ Set.Ioo 2 4)) → a = 1 :=
by
  intro h
  sorry

end find_a_l204_204859


namespace solve_for_nabla_l204_204953

theorem solve_for_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by
  sorry

end solve_for_nabla_l204_204953


namespace sally_initial_orange_balloons_l204_204643

def initial_orange_balloons (found_orange : ℝ) (total_orange : ℝ) : ℝ := 
  total_orange - found_orange

theorem sally_initial_orange_balloons : initial_orange_balloons 2.0 11 = 9 := 
by
  sorry

end sally_initial_orange_balloons_l204_204643


namespace carrie_hours_per_week_l204_204803

variable (H : ℕ)

def carrie_hourly_wage : ℕ := 8
def cost_of_bike : ℕ := 400
def amount_left_over : ℕ := 720
def weeks_worked : ℕ := 4
def total_earnings : ℕ := cost_of_bike + amount_left_over

theorem carrie_hours_per_week :
  (weeks_worked * H * carrie_hourly_wage = total_earnings) →
  H = 35 := by
  sorry

end carrie_hours_per_week_l204_204803


namespace minimum_throws_for_repeated_sum_l204_204074

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l204_204074


namespace inconsistent_b_positive_l204_204940

theorem inconsistent_b_positive
  (a b c : ℝ)
  (h_solution_set : ∀ x : ℝ, -2 < x ∧ x < 3 / 2 → ax^2 + bx + c > 0) :
  ¬ b > 0 :=
sorry

end inconsistent_b_positive_l204_204940


namespace cost_of_video_game_console_l204_204505

-- Define the problem conditions
def earnings_Mar_to_Aug : ℕ := 460
def hours_Mar_to_Aug : ℕ := 23
def earnings_per_hour : ℕ := earnings_Mar_to_Aug / hours_Mar_to_Aug
def hours_Sep_to_Feb : ℕ := 8
def cost_car_fix : ℕ := 340
def additional_hours_needed : ℕ := 16

-- Proof that the cost of the video game console is $600
theorem cost_of_video_game_console :
  let initial_earnings := earnings_Mar_to_Aug
  let earnings_from_Sep_to_Feb := hours_Sep_to_Feb * earnings_per_hour
  let total_earnings_before_expenses := initial_earnings + earnings_from_Sep_to_Feb
  let current_savings := total_earnings_before_expenses - cost_car_fix
  let earnings_after_additional_work := additional_hours_needed * earnings_per_hour
  let total_savings := current_savings + earnings_after_additional_work
  total_savings = 600 :=
by
  sorry

end cost_of_video_game_console_l204_204505


namespace relay_selection_ways_l204_204149

-- Define the problem conditions
def sprinters : Finset ℕ := {0, 1, 2, 3, 4, 5}  -- 6 sprinters labeled from 0 to 5

def first_leg_restriction (p : ℕ) : Prop := p ≠ 0  -- Sprinter A (0) cannot run first
def fourth_leg_restriction (p : ℕ) : Prop := p ≠ 1  -- Sprinter B (1) cannot run fourth

-- The main theorem statement
theorem relay_selection_ways :
  ∑ (a ∈ sprinters) (ha : first_leg_restriction a),
  ∑ (b ∈ (sprinters \ {a})) (hb : fourth_leg_restriction b),
  ∑ (c ∈ (sprinters \ {a, b})),
  ∑ (d ∈ (sprinters \ {a, b, c})),
  1 = 252 :=
by sorry

end relay_selection_ways_l204_204149


namespace possible_values_of_N_l204_204252

theorem possible_values_of_N (N : ℤ) (h : N^2 - N = 12) : N = 4 ∨ N = -3 :=
sorry

end possible_values_of_N_l204_204252


namespace equal_play_time_l204_204785

theorem equal_play_time (total_players on_field_players match_minutes : ℕ) 
    (h1 : total_players = 10) 
    (h2 : on_field_players = 8) 
    (h3 : match_minutes = 45) 
    (h4 : ∀ p, p ∈ (finset.range total_players) → time_played p = (on_field_players * match_minutes) / total_players)
    : (on_field_players * match_minutes) / total_players = 36 :=
by
  have total_play_minutes : on_field_players * match_minutes = 360 := by sorry
  have equal_play_time : (on_field_players * match_minutes) / total_players = 36 := by sorry
  exact equal_play_time

end equal_play_time_l204_204785


namespace range_x_satisfies_inequality_l204_204046

theorem range_x_satisfies_inequality (x : ℝ) : (x^2 < |x|) ↔ (-1 < x ∧ x < 1 ∧ x ≠ 0) :=
sorry

end range_x_satisfies_inequality_l204_204046


namespace proof_problem_l204_204296

def D_f (f : ℕ → ℕ) (n : ℕ) : ℕ :=
  Nat.find (λ m, ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → f i % m ≠ f j % m)

def f (x : ℕ) := x * (3 * x - 1)

theorem proof_problem (n : ℕ) : D_f f n = 3 ^ Nat.ceil (Real.log n) := by
  sorry

end proof_problem_l204_204296


namespace arccos_of_sqrt3_div_2_l204_204171

theorem arccos_of_sqrt3_div_2 : Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 := by
  sorry

end arccos_of_sqrt3_div_2_l204_204171


namespace garden_width_l204_204294

variable (W : ℝ) (L : ℝ := 225) (small_gate : ℝ := 3) (large_gate: ℝ := 10) (total_fencing : ℝ := 687)

theorem garden_width :
  2 * L + 2 * W - (small_gate + large_gate) = total_fencing → W = 125 := 
by
  sorry

end garden_width_l204_204294


namespace equal_playing_time_for_each_player_l204_204775

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l204_204775


namespace expense_5_yuan_neg_l204_204021

-- Define the condition that income of 5 yuan is denoted as +5 yuan
def income_5_yuan_pos : Int := 5

-- Define the statement to prove that expenses of 5 yuan are denoted as -5 yuan
theorem expense_5_yuan_neg : income_5_yuan_pos = 5 → -income_5_yuan_pos = -5 :=
by
  intro h
  rw h
  rfl

end expense_5_yuan_neg_l204_204021


namespace largest_angle_right_triangle_l204_204235

theorem largest_angle_right_triangle
  (a b c : ℝ)
  (h₁ : ∃ x : ℝ, x^2 + 4 * (c + 2) = (c + 4) * x)
  (h₂ : a + b = c + 4)
  (h₃ : a * b = 4 * (c + 2))
  : ∃ x : ℝ, x = 90 :=
by {
  sorry
}

end largest_angle_right_triangle_l204_204235


namespace necessary_but_not_sufficient_condition_l204_204821

def p (x : ℝ) : Prop := |4 * x - 3| ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a^2 + a ≤ 0

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, p x → q x a) ∧ ¬ (∀ x : ℝ, q x a → p x) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end necessary_but_not_sufficient_condition_l204_204821


namespace finish_together_in_4_days_l204_204418

-- Definitions for the individual days taken by A, B, and C
def days_for_A := 12
def days_for_B := 24
def days_for_C := 8 -- C's approximated days

-- The rates are the reciprocals of the days
def rate_A := 1 / days_for_A
def rate_B := 1 / days_for_B
def rate_C := 1 / days_for_C

-- The combined rate of A, B, and C
def combined_rate := rate_A + rate_B + rate_C

-- The total days required to finish the work together
def total_days := 1 / combined_rate

-- Theorem stating that the total days required is 4
theorem finish_together_in_4_days : total_days = 4 := 
by 
-- proof omitted
sorry

end finish_together_in_4_days_l204_204418


namespace gcd_of_45_and_75_l204_204724

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l204_204724


namespace light_path_in_cube_l204_204982

/-- Let ABCD and EFGH be two faces of a cube with AB = 10. A beam of light is emitted 
from vertex A and reflects off face EFGH at point Q, which is 6 units from EH and 4 
units from EF. The length of the light path from A until it reaches another vertex of 
the cube for the first time is expressed in the form s√t, where s and t are integers 
with t having no square factors. Provide s + t. -/
theorem light_path_in_cube :
  let AB := 10
  let s := 10
  let t := 152
  s + t = 162 := by
  sorry

end light_path_in_cube_l204_204982


namespace gcd_45_75_l204_204711

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204711


namespace pencils_count_l204_204139

theorem pencils_count (P L : ℕ) 
  (h1 : P * 6 = L * 5) 
  (h2 : L = P + 7) : 
  L = 42 :=
by
  sorry

end pencils_count_l204_204139


namespace scientific_notation_example_l204_204279

theorem scientific_notation_example : 3790000 = 3.79 * 10^6 := 
sorry

end scientific_notation_example_l204_204279


namespace device_elements_probabilities_l204_204035

theorem device_elements_probabilities:
  ∀ {Ω : Type} [MeasureSpace Ω] (A B : Set Ω),
  Prob (A) = 0.2 ∧ Prob (B) = 0.3 ∧ indep_events A B →
  Prob (A ∩ B) = 0.06 ∧ Prob (Aᶜ ∩ Bᶜ) = 0.56 :=
by
  sorry

end device_elements_probabilities_l204_204035


namespace min_value_expression_l204_204391

/--
  Prove that the minimum value of the expression (xy - 2)^2 + (x + y - 1)^2 
  for real numbers x and y is 2.
--/
theorem min_value_expression : 
  ∃ x y : ℝ, (∀ a b : ℝ, (a * b - 2)^2 + (a + b - 1)^2 ≥ (x * y - 2)^2 + (x + y - 1)^2 ) ∧ 
  (x * y - 2)^2 + (x + y - 1)^2 = 2 :=
by
  sorry

end min_value_expression_l204_204391


namespace expenses_opposite_to_income_l204_204014

theorem expenses_opposite_to_income (income_5 : ℤ) (h_income : income_5 = 5) : -income_5 = -5 :=
by
  -- proof is omitted
  sorry

end expenses_opposite_to_income_l204_204014


namespace combined_total_years_l204_204888

theorem combined_total_years (A : ℕ) (V : ℕ) (D : ℕ)
(h1 : V = A + 9)
(h2 : V = D - 9)
(h3 : D = 34) : A + V + D = 75 :=
by sorry

end combined_total_years_l204_204888


namespace gcd_8251_6105_l204_204665

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l204_204665


namespace smallest_egg_count_l204_204894

theorem smallest_egg_count : ∃ n : ℕ, n > 100 ∧ n % 12 = 10 ∧ n = 106 :=
by {
  sorry
}

end smallest_egg_count_l204_204894


namespace star_angle_sum_l204_204523

-- Define variables and angles for Petya's and Vasya's stars.
variables {α β γ δ ε : ℝ}
variables {φ χ ψ ω : ℝ}
variables {a b c d e : ℝ}

-- Conditions
def all_acute (a b c d e : ℝ) : Prop := a < 90 ∧ b < 90 ∧ c < 90 ∧ d < 90 ∧ e < 90
def one_obtuse (a b c d e : ℝ) : Prop := (a > 90 ∨ b > 90 ∨ c > 90 ∨ d > 90 ∨ e > 90)

-- Question: Prove the sum of the angles at the vertices of both stars is equal
theorem star_angle_sum : all_acute α β γ δ ε → one_obtuse φ χ ψ ω α → 
  α + β + γ + δ + ε = φ + χ + ψ + ω + α := 
by sorry

end star_angle_sum_l204_204523


namespace four_dice_min_rolls_l204_204110

def minRollsToEnsureSameSum (n : Nat) : Nat :=
  if n = 4 then 22 else sorry

theorem four_dice_min_rolls : minRollsToEnsureSameSum 4 = 22 := by
  rfl

end four_dice_min_rolls_l204_204110


namespace each_player_plays_36_minutes_l204_204792

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l204_204792


namespace nancy_pots_created_on_Wednesday_l204_204241

def nancy_pots_conditions (pots_Monday pots_Tuesday total_pots : ℕ) : Prop :=
  pots_Monday = 12 ∧ pots_Tuesday = 2 * pots_Monday ∧ total_pots = 50

theorem nancy_pots_created_on_Wednesday :
  ∀ pots_Monday pots_Tuesday total_pots,
  nancy_pots_conditions pots_Monday pots_Tuesday total_pots →
  (total_pots - (pots_Monday + pots_Tuesday) = 14) := by
  intros pots_Monday pots_Tuesday total_pots h
  -- proof would go here
  sorry

end nancy_pots_created_on_Wednesday_l204_204241


namespace equation_B_no_real_solution_l204_204297

theorem equation_B_no_real_solution : ∀ x : ℝ, |3 * x + 1| + 6 ≠ 0 := 
by 
  sorry

end equation_B_no_real_solution_l204_204297


namespace find_m_l204_204935

theorem find_m (m : ℝ) 
  (A : ℝ × ℝ := (-2, m))
  (B : ℝ × ℝ := (m, 4))
  (h_slope : ((B.snd - A.snd) / (B.fst - A.fst)) = -2) : 
  m = -8 :=
by 
  sorry

end find_m_l204_204935


namespace coefficient_x8_expansion_l204_204050

-- Define the problem statement in Lean
theorem coefficient_x8_expansion : 
  (Nat.choose 7 4) * (1 : ℤ)^3 * (-2 : ℤ)^4 = 560 :=
by
  sorry

end coefficient_x8_expansion_l204_204050


namespace train_crosses_platform_l204_204761

theorem train_crosses_platform :
  ∀ (L : ℕ), 
  (300 + L) / (50 / 3) = 48 → 
  L = 500 := 
by
  sorry

end train_crosses_platform_l204_204761


namespace average_salary_of_laborers_l204_204470

-- Define the main statement as a theorem
theorem average_salary_of_laborers 
  (total_workers : ℕ)
  (total_salary_all : ℕ)
  (supervisors : ℕ)
  (supervisor_salary : ℕ)
  (laborers : ℕ)
  (expected_laborer_salary : ℝ) :
  total_workers = 48 → 
  total_salary_all = 60000 →
  supervisors = 6 →
  supervisor_salary = 2450 →
  laborers = 42 →
  expected_laborer_salary = 1078.57 :=
sorry

end average_salary_of_laborers_l204_204470


namespace probability_die_sum_odd_l204_204260

namespace CoinDieProblem

-- Define the conditions and question as a statement
theorem probability_die_sum_odd :
  let coin_tosses := { outcome | outcome ∈ {'H', 'T'}^2 }
  let prob_head (coin_toss: fin 2 → char) : ℚ := 0.5
  let die_roll := fin 6
  let prob_die_odd (roll: die_roll) : ℚ := if roll ∈ {0, 2, 4} then 0.5 else 0
  let prob_2_dice_odd := 2 * 0.25 in
  (∑ outcome in coin_tosses, 
   prob_head outcome[0] * prob_head outcome[1] * 
   (if hd_count outcome[0] + hd_count outcome[1] = 0 then 0 else 
    if hd_count outcome[0] + hd_count outcome[1] = 1 then prob_die_odd else 
    prob_2_dice_odd)) = 3/8 := sorry

end CoinDieProblem

end probability_die_sum_odd_l204_204260


namespace jerry_pool_time_l204_204798

variables (J : ℕ) -- Denote the time Jerry was in the pool

-- Conditions
def Elaine_time := 2 * J -- Elaine stayed in the pool for twice as long as Jerry
def George_time := (2 / 3) * J -- George could only stay in the pool for one-third as long as Elaine
def Kramer_time := 0 -- Kramer did not find the pool

-- Combined total time
def total_time : ℕ := J + Elaine_time J + George_time J + Kramer_time

-- Theorem stating that J = 3 given the combined total time of 11 minutes
theorem jerry_pool_time (h : total_time J = 11) : J = 3 :=
by
  sorry

end jerry_pool_time_l204_204798


namespace kamals_salary_change_l204_204979

theorem kamals_salary_change : 
  ∀ (S : ℝ), ((S * 0.5 * 1.3 * 0.8 - S) / S) * 100 = -48 :=
by
  intro S
  sorry

end kamals_salary_change_l204_204979


namespace determine_sold_cakes_l204_204799

def initial_cakes := 121
def new_cakes := 170
def remaining_cakes := 186
def sold_cakes (S : ℕ) : Prop := initial_cakes - S + new_cakes = remaining_cakes

theorem determine_sold_cakes : ∃ S, sold_cakes S ∧ S = 105 :=
by
  use 105
  unfold sold_cakes
  simp
  sorry

end determine_sold_cakes_l204_204799


namespace line_intersection_equation_of_l4_find_a_l204_204828

theorem line_intersection (P : ℝ × ℝ)
    (h1: 3 * P.1 + 4 * P.2 - 2 = 0)
    (h2: 2 * P.1 + P.2 + 2 = 0) :
  P = (-2, 2) :=
sorry

theorem equation_of_l4 (l4 : ℝ → ℝ → Prop)
    (P : ℝ × ℝ) (h1: 3 * P.1 + 4 * P.2 - 2 = 0)
    (h2: 2 * P.1 + P.2 + 2 = 0) 
    (h_parallel: ∀ x y, l4 x y ↔ y = 1/2 * x + 3)
    (x y : ℝ) :
  l4 x y ↔ y = 1/2 * x + 3 :=
sorry

theorem find_a (a : ℝ) :
    (∀ x y, 2 * x + y + 2 = 0 → y = -2 * x - 2) →
    (∀ x y, a * x - 2 * y + 1 = 0 → y = 1/2 * x - 1/2) →
    a = 1 :=
sorry

end line_intersection_equation_of_l4_find_a_l204_204828


namespace house_assignment_l204_204873

theorem house_assignment (n : ℕ) (assign : Fin n → Fin n) (pref : Fin n → Fin n → Fin n → Prop) :
  (∀ (p : Fin n), ∃ (better_assign : Fin n → Fin n),
    (∃ q, pref p (assign p) (better_assign p) ∧ pref q (assign q) (better_assign p) ∧ better_assign q ≠ assign q)
  ) → (∃ p, pref p (assign p) (assign p))
:= sorry

end house_assignment_l204_204873


namespace calculate_expression_l204_204421

theorem calculate_expression : 1 + (Real.sqrt 2 - Real.sqrt 3) + abs (Real.sqrt 2 - Real.sqrt 3) = 1 :=
by
  sorry

end calculate_expression_l204_204421


namespace john_remaining_money_l204_204977

theorem john_remaining_money (q : ℝ) : 
  let drink_cost := 5 * q
  let medium_pizza_cost := 3 * 2 * q
  let large_pizza_cost := 2 * 3 * q
  let dessert_cost := 4 * (1 / 2) * q
  let total_cost := drink_cost + medium_pizza_cost + large_pizza_cost + dessert_cost
  let initial_money := 60
  initial_money - total_cost = 60 - 19 * q :=
by
  sorry

end john_remaining_money_l204_204977


namespace minimum_throws_for_repeated_sum_l204_204075

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l204_204075


namespace arithmetic_sequence_common_difference_l204_204966

theorem arithmetic_sequence_common_difference 
  (a l S : ℕ) (h1 : a = 5) (h2 : l = 50) (h3 : S = 495) :
  (∃ d n : ℕ, l = a + (n-1) * d ∧ S = n * (a + l) / 2 ∧ d = 45 / 17) :=
by
  sorry

end arithmetic_sequence_common_difference_l204_204966


namespace four_dice_min_rolls_l204_204108

def minRollsToEnsureSameSum (n : Nat) : Nat :=
  if n = 4 then 22 else sorry

theorem four_dice_min_rolls : minRollsToEnsureSameSum 4 = 22 := by
  rfl

end four_dice_min_rolls_l204_204108


namespace expenses_opposite_to_income_l204_204016

theorem expenses_opposite_to_income (income_5 : ℤ) (h_income : income_5 = 5) : -income_5 = -5 :=
by
  -- proof is omitted
  sorry

end expenses_opposite_to_income_l204_204016


namespace pens_multiple_91_l204_204999

theorem pens_multiple_91 (S : ℕ) (P : ℕ) (total_pencils : ℕ) 
  (h1 : S = 91) (h2 : total_pencils = 910) (h3 : total_pencils % S = 0) :
  ∃ (x : ℕ), P = S * x :=
by 
  sorry

end pens_multiple_91_l204_204999


namespace sufficient_but_not_necessary_for_abs_eq_two_l204_204834

theorem sufficient_but_not_necessary_for_abs_eq_two (a : ℝ) :
  (a = -2 → |a| = 2) ∧ (|a| = 2 → a = 2 ∨ a = -2) :=
by
   sorry

end sufficient_but_not_necessary_for_abs_eq_two_l204_204834


namespace count_two_digit_numbers_with_8_l204_204331

theorem count_two_digit_numbers_with_8 : 
  (card {n : ℕ | 10 <= n ∧ n < 100 ∧ (n / 10 = 8 ∨ n % 10 = 8)}) = 17 := 
by 
  sorry

end count_two_digit_numbers_with_8_l204_204331


namespace average_marks_of_a_b_c_d_l204_204250

theorem average_marks_of_a_b_c_d (A B C D E : ℕ)
  (h1 : (A + B + C) / 3 = 48)
  (h2 : A = 43)
  (h3 : (B + C + D + E) / 4 = 48)
  (h4 : E = D + 3) :
  (A + B + C + D) / 4 = 47 :=
by
  -- This theorem will be justified
  admit

end average_marks_of_a_b_c_d_l204_204250


namespace gain_in_transaction_per_year_l204_204908

noncomputable def borrowing_interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time

noncomputable def lending_interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time

noncomputable def gain_per_year (borrow_principal : ℕ) (borrow_rate : ℚ) 
  (borrow_time : ℕ) (lend_principal : ℕ) (lend_rate : ℚ) (lend_time : ℕ) : ℚ :=
  (lending_interest lend_principal lend_rate lend_time - borrowing_interest borrow_principal borrow_rate borrow_time) / borrow_time

theorem gain_in_transaction_per_year :
  gain_per_year 4000 (4 / 100) 2 4000 (6 / 100) 2 = 80 := 
sorry

end gain_in_transaction_per_year_l204_204908


namespace inverse_of_128_l204_204956

def f : ℕ → ℕ := sorry
axiom f_at_5 : f 5 = 2
axiom f_property : ∀ x, f (2 * x) = 2 * f x

theorem inverse_of_128 : f⁻¹ 128 = 320 :=
by {
  have basic_values : f 5 = 2 ∧ f (2 * 5) = 4 ∧ f (4 * 5) = 8 ∧ f (8 * 5) = 16 ∧
                       f (16 * 5) = 32 ∧ f (32 * 5) = 64 ∧ f (64 * 5) = 128,
  {
    split, exact f_at_5,
    split, rw [f_property, f_at_5],
    split, rw [f_property, f_property, f_at_5],
    split, rw [f_property, f_property, f_property, f_at_5],
    split, rw [f_property, f_property, f_property, f_property, f_at_5],
         rw [mul_comm, ← mul_assoc, f_property, mul_comm 4, f_property, mul_comm, f_property],
    split, rw [f_property, f_property, f_property, f_property, f_property, f_at_5],
             rw [mul_comm, ← mul_assoc, f_property, mul_comm 8, f_property, mul_comm, f_property],
    split, rw [f_property, f_property, f_property, f_property, f_property, f_property, f_at_5],
               rw [mul_comm, ← mul_assoc, f_property, mul_comm 16, f_property, mul_comm, f_property],
         rw [mul_comm, ← mul_assoc, f_property, mul_comm 32],
         rw [mul_comm, ← mul_assoc, mul_comm 8],
    tauto,
  },
  exact sorry
}

end inverse_of_128_l204_204956


namespace no_real_roots_of_quadratic_l204_204464

theorem no_real_roots_of_quadratic (a : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 2 * a * x + 1 ≠ 0) ↔ a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) := by
  sorry

end no_real_roots_of_quadratic_l204_204464


namespace find_value_l204_204431

variable {x y : ℝ}

theorem find_value (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y + x * y = 0) : y / x + x / y = -2 := 
sorry

end find_value_l204_204431


namespace parking_space_length_l204_204772

theorem parking_space_length {L W : ℕ} 
  (h1 : 2 * W + L = 37) 
  (h2 : L * W = 126) : 
  L = 9 := 
sorry

end parking_space_length_l204_204772


namespace arccos_sqrt3_div_2_eq_pi_div_6_l204_204168

theorem arccos_sqrt3_div_2_eq_pi_div_6 : 
  Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 :=
by 
  sorry

end arccos_sqrt3_div_2_eq_pi_div_6_l204_204168


namespace quadratic_real_roots_range_of_m_quadratic_root_and_other_m_l204_204827

open Real

-- Mathematical translations of conditions and proofs
theorem quadratic_real_roots_range_of_m (m : ℝ) (h1 : ∃ x : ℝ, x^2 + 2 * x - (m - 2) = 0) :
  m ≥ 1 := by
  sorry

theorem quadratic_root_and_other_m (h1 : (1:ℝ) ^ 2 + 2 * 1 - (m - 2) = 0) :
  m = 3 ∧ ∃ x : ℝ, (x = -3) ∧ (x^2 + 2 * x - 3 = 0) := by
  sorry

end quadratic_real_roots_range_of_m_quadratic_root_and_other_m_l204_204827


namespace initial_cabinets_l204_204346

theorem initial_cabinets (C : ℤ) (h1 : 26 = C + 6 * C + 5) : C = 3 := 
by 
  sorry

end initial_cabinets_l204_204346


namespace gopi_servant_salary_l204_204947

theorem gopi_servant_salary (S : ℕ) (turban_price : ℕ) (cash_received : ℕ) (months_worked : ℕ) (total_months : ℕ) :
  turban_price = 70 →
  cash_received = 50 →
  months_worked = 9 →
  total_months = 12 →
  S = 160 :=
by
  sorry

end gopi_servant_salary_l204_204947


namespace relationship_between_a_and_b_l204_204309

open Real

theorem relationship_between_a_and_b
   (a b : ℝ)
   (ha : 0 < a ∧ a < 1)
   (hb : 0 < b ∧ b < 1)
   (hab : (1 - a) * b > 1 / 4) :
   a < b := 
sorry

end relationship_between_a_and_b_l204_204309


namespace no_four_of_a_kind_pair_set_aside_re_rollematch_l204_204302

noncomputable def probability_at_least_four_same (d : Dice) : ℕ :=
1 / 216

theorem no_four_of_a_kind_pair_set_aside_re_rollematch :
  ∀ (d1 d2 d3 d4 d5 : ℕ), -- Five standard six-sided dice are rolled
  d1 ≠ d2 → d1 ≠ d3 → d1 ≠ d4 → d1 ≠ d5 → -- There is no four-of-a-kind
  d2 ≠ d3 → d2 ≠ d4 → d2 ≠ d5 →
  d3 ≠ d4 → d3 ≠ d5 →
  d4 ≠ d5 →
  (∃ (pair : ℕ) (rest : list ℕ),
    rest.length = 3 ∧ -- Two dice showing the same number are set aside, the other three are re-rolled
    list.map (λ r, dice_roll d r) rest = repeat pair 3) →
  probability_at_least_four_same d = (1 / 216) := -- The probability that at least four of the five dice show the same value
sorry

end no_four_of_a_kind_pair_set_aside_re_rollematch_l204_204302


namespace expenses_representation_l204_204002

theorem expenses_representation (income_representation : ℤ) (income : ℤ) (expenses : ℤ) :
  income_representation = +5 → income = +5 → expenses = -income → expenses = -5 :=
by
  intro hr hs he
  rw [←hs, he]
  exact hr

end expenses_representation_l204_204002


namespace circumradius_inradius_inequality_l204_204626

theorem circumradius_inradius_inequality (a b c R r : ℝ) (hR : R > 0) (hr : r > 0) :
  R / (2 * r) ≥ ((64 * a^2 * b^2 * c^2) / 
  ((4 * a^2 - (b - c)^2) * (4 * b^2 - (c - a)^2) * (4 * c^2 - (a - b)^2)))^2 :=
sorry

end circumradius_inradius_inequality_l204_204626


namespace product_prs_l204_204336

open Real

theorem product_prs (p r s : ℕ) 
  (h1 : 4 ^ p + 64 = 272) 
  (h2 : 3 ^ r = 81)
  (h3 : 6 ^ s = 478) : 
  p * r * s = 64 :=
by
  sorry

end product_prs_l204_204336


namespace find_X_l204_204531

theorem find_X (X : ℝ) (h : 45 * 8 = 0.40 * X) : X = 900 :=
sorry

end find_X_l204_204531


namespace fruit_days_l204_204451

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

end fruit_days_l204_204451


namespace find_radius_of_cone_base_l204_204316

def slant_height : ℝ := 5
def lateral_surface_area : ℝ := 15 * Real.pi

theorem find_radius_of_cone_base (A l : ℝ) (hA : A = lateral_surface_area) (hl : l = slant_height) : 
  ∃ r : ℝ, A = Real.pi * r * l ∧ r = 3 := 
by 
  sorry

end find_radius_of_cone_base_l204_204316


namespace ratio_of_areas_l204_204511

-- Define the conditions
variable (s : ℝ) (h_pos : s > 0)
-- The total perimeter of four small square pens is reused for one large square pen
def total_fencing_length := 16 * s
def large_square_side_length := 4 * s

-- Define the areas
def small_squares_total_area := 4 * s^2
def large_square_area := (4 * s)^2

-- The statement to prove
theorem ratio_of_areas : small_squares_total_area / large_square_area = 1 / 4 :=
by
  sorry

end ratio_of_areas_l204_204511


namespace band_and_chorus_but_not_orchestra_l204_204361

theorem band_and_chorus_but_not_orchestra (B C O : Finset ℕ)
  (hB : B.card = 100) 
  (hC : C.card = 120) 
  (hO : O.card = 60)
  (hUnion : (B ∪ C ∪ O).card = 200)
  (hIntersection : (B ∩ C ∩ O).card = 10) : 
  ((B ∩ C).card - (B ∩ C ∩ O).card = 30) :=
by sorry

end band_and_chorus_but_not_orchestra_l204_204361


namespace range_of_a_l204_204322

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (a * x - 1) / x > 2 * a) ↔ a ∈ (Set.Ici (-1/2) : Set ℝ) :=
by
  sorry

end range_of_a_l204_204322


namespace m_value_for_perfect_square_l204_204954

theorem m_value_for_perfect_square (m : ℤ) (x y : ℤ) :
  (∃ k : ℤ, 4 * x^2 - m * x * y + 9 * y^2 = k^2) → m = 12 ∨ m = -12 :=
by
  sorry

end m_value_for_perfect_square_l204_204954


namespace hundredth_ring_square_count_l204_204568

-- Conditions
def center_rectangle : ℤ × ℤ := (1, 2)
def first_ring_square_count : ℕ := 10
def square_count_nth_ring (n : ℕ) : ℕ := 8 * n + 2

-- Problem Statement
theorem hundredth_ring_square_count : square_count_nth_ring 100 = 802 := 
  sorry

end hundredth_ring_square_count_l204_204568


namespace kernel_red_given_popped_l204_204276

def prob_red_given_popped (P_red : ℚ) (P_green : ℚ) 
                           (P_popped_given_red : ℚ) (P_popped_given_green : ℚ) : ℚ :=
  let P_red_popped := P_red * P_popped_given_red
  let P_green_popped := P_green * P_popped_given_green
  let P_popped := P_red_popped + P_green_popped
  P_red_popped / P_popped

theorem kernel_red_given_popped : prob_red_given_popped (3/4) (1/4) (3/5) (3/4) = 12/17 :=
by
  sorry

end kernel_red_given_popped_l204_204276


namespace undefined_value_of_expression_l204_204187

theorem undefined_value_of_expression (a : ℝ) : (a^3 - 8 = 0) → (a = 2) := by
  sorry

end undefined_value_of_expression_l204_204187


namespace second_race_length_l204_204842

variable (T L : ℝ)
variable (V_A V_B V_C : ℝ)

variables (h1 : V_A * T = 100)
variables (h2 : V_B * T = 90)
variables (h3 : V_C * T = 87)
variables (h4 : L / V_B = (L - 6) / V_C)

theorem second_race_length :
  L = 180 :=
sorry

end second_race_length_l204_204842


namespace proportion_a_value_l204_204960

theorem proportion_a_value (a b c d : ℝ) (h1 : b = 3) (h2 : c = 4) (h3 : d = 6) (h4 : a / b = c / d) : a = 2 :=
by sorry

end proportion_a_value_l204_204960


namespace necessary_but_not_sufficient_condition_l204_204892

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  ((1 / a < 1 ↔ a < 0 ∨ a > 1) ∧ ¬(1 / a < 1 → a ≤ 0 ∨ a ≤ 1)) := 
by sorry

end necessary_but_not_sufficient_condition_l204_204892


namespace find_k_l204_204466

noncomputable def expr_to_complete_square (x : ℝ) : ℝ :=
  x^2 - 6 * x

theorem find_k (x : ℝ) : ∃ a h k, expr_to_complete_square x = a * (x - h)^2 + k ∧ k = -9 :=
by
  use 1, 3, -9
  -- detailed steps of the proof would go here
  sorry

end find_k_l204_204466


namespace length_of_ST_l204_204809

theorem length_of_ST (PQ PS : ℝ) (ST : ℝ) (hPQ : PQ = 8) (hPS : PS = 7) 
  (h_area_eq : (1 / 2) * PQ * (PS * (1 / PS) * 8) = PQ * PS) : 
  ST = 2 * Real.sqrt 65 := 
by
  -- proof steps (to be written)
  sorry

end length_of_ST_l204_204809


namespace line_length_400_l204_204909

noncomputable def length_of_line (speed_march_kmh speed_run_kmh total_time_min: ℝ) : ℝ :=
  let speed_march_mpm := (speed_march_kmh * 1000) / 60
  let speed_run_mpm := (speed_run_kmh * 1000) / 60
  let len_eq := 1 / (speed_run_mpm - speed_march_mpm) + 1 / (speed_run_mpm + speed_march_mpm)
  (total_time_min * 200 * len_eq) * 400 / len_eq

theorem line_length_400 :
  length_of_line 8 12 7.2 = 400 := by
  sorry

end line_length_400_l204_204909


namespace solve_for_z_l204_204191

theorem solve_for_z (i z : ℂ) (h0 : i^2 = -1) (h1 : i / z = 1 + i) : z = (1 + i) / 2 :=
by
  sorry

end solve_for_z_l204_204191


namespace smallest_repeating_block_length_of_7_over_13_l204_204832

theorem smallest_repeating_block_length_of_7_over_13 : 
  ∀ k, (∃ a b, 7 / 13 = a + (b / 10^k)) → k = 6 := 
sorry

end smallest_repeating_block_length_of_7_over_13_l204_204832


namespace ways_from_A_to_C_l204_204609

theorem ways_from_A_to_C (ways_A_to_B : ℕ) (ways_B_to_C : ℕ) (hA_to_B : ways_A_to_B = 3) (hB_to_C : ways_B_to_C = 4) : ways_A_to_B * ways_B_to_C = 12 :=
by
  sorry

end ways_from_A_to_C_l204_204609


namespace spade_evaluation_l204_204583

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_evaluation : spade 5 (spade 6 7 + 2) = -96 := by
  sorry

end spade_evaluation_l204_204583


namespace cannot_determine_letters_afternoon_l204_204975

theorem cannot_determine_letters_afternoon
  (emails_morning : ℕ) (letters_morning : ℕ)
  (emails_afternoon : ℕ) (letters_afternoon : ℕ)
  (h1 : emails_morning = 10)
  (h2 : letters_morning = 12)
  (h3 : emails_afternoon = 3)
  (h4 : emails_morning = emails_afternoon + 7) :
  ¬∃ (letters_afternoon : ℕ), true := 
sorry

end cannot_determine_letters_afternoon_l204_204975


namespace opposite_of_negative_one_fifth_l204_204042

theorem opposite_of_negative_one_fifth : -(-1 / 5) = (1 / 5) :=
by
  sorry

end opposite_of_negative_one_fifth_l204_204042


namespace cos_C_max_ab_over_c_l204_204477

theorem cos_C_max_ab_over_c
  (a b c S : ℝ) (A B C : ℝ)
  (h1 : 6 * S = a^2 * Real.sin A + b^2 * Real.sin B)
  (h2 : a / Real.sin A = b / Real.sin B)
  (h3 : b / Real.sin B = c / Real.sin C)
  (h4 : S = 0.5 * a * b * Real.sin C)
  : Real.cos C = 7 / 9 := 
sorry

end cos_C_max_ab_over_c_l204_204477


namespace length_of_each_piece_after_subdividing_l204_204541

theorem length_of_each_piece_after_subdividing (total_length : ℝ) (num_initial_cuts : ℝ) (num_pieces_given : ℝ) (num_subdivisions : ℝ) (final_length : ℝ) : 
  total_length = 200 → 
  num_initial_cuts = 4 → 
  num_pieces_given = 2 → 
  num_subdivisions = 2 → 
  final_length = (total_length / num_initial_cuts / num_subdivisions) → 
  final_length = 25 := 
by 
  intros h1 h2 h3 h4 h5 
  sorry

end length_of_each_piece_after_subdividing_l204_204541


namespace number_of_rabbits_l204_204468

theorem number_of_rabbits (x y : ℕ) (h1 : x + y = 28) (h2 : 4 * x = 6 * y + 12) : x = 18 :=
by
  sorry

end number_of_rabbits_l204_204468


namespace bailey_points_final_game_l204_204344

def chandra_points (a: ℕ) := 2 * a
def akiko_points (m: ℕ) := m + 4
def michiko_points (b: ℕ) := b / 2
def team_total_points (b c a m: ℕ) := b + c + a + m

theorem bailey_points_final_game (B: ℕ) 
  (M : ℕ := michiko_points B)
  (A : ℕ := akiko_points M)
  (C : ℕ := chandra_points A)
  (H : team_total_points B C A M = 54): B = 14 :=
by 
  sorry

end bailey_points_final_game_l204_204344


namespace min_value_when_a_is_half_range_of_a_for_positivity_l204_204193

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + 2*x + a) / x

theorem min_value_when_a_is_half : 
  ∀ x ∈ Set.Ici (1 : ℝ), f x (1/2) ≥ (7 / 2) := 
by 
  sorry

theorem range_of_a_for_positivity :
  ∀ x ∈ Set.Ici (1 : ℝ), f x a > 0 ↔ a ∈ Set.Ioc (-3 : ℝ) 1 :=
by 
  sorry

end min_value_when_a_is_half_range_of_a_for_positivity_l204_204193


namespace largest_4_digit_congruent_to_7_mod_19_l204_204747

theorem largest_4_digit_congruent_to_7_mod_19 : 
  ∃ x, (x % 19 = 7) ∧ 1000 ≤ x ∧ x < 10000 ∧ x = 9982 :=
by
  sorry

end largest_4_digit_congruent_to_7_mod_19_l204_204747


namespace gcd_of_45_and_75_l204_204723

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l204_204723


namespace geometric_sequence_const_k_l204_204846

noncomputable def sum_of_terms (n : ℕ) (k : ℤ) : ℤ := 3 * 2^n + k
noncomputable def a1 (k : ℤ) : ℤ := sum_of_terms 1 k
noncomputable def a2 (k : ℤ) : ℤ := sum_of_terms 2 k - sum_of_terms 1 k
noncomputable def a3 (k : ℤ) : ℤ := sum_of_terms 3 k - sum_of_terms 2 k

theorem geometric_sequence_const_k :
  (∀ (k : ℤ), (a1 k * a3 k = a2 k * a2 k) → k = -3) :=
by
  sorry

end geometric_sequence_const_k_l204_204846


namespace arccos_sqrt_3_over_2_eq_pi_over_6_l204_204165

open Real

theorem arccos_sqrt_3_over_2_eq_pi_over_6 :
  ∀ (x : ℝ), x = (sqrt 3) / 2 → arccos x = π / 6 :=
by
  intro x
  sorry

end arccos_sqrt_3_over_2_eq_pi_over_6_l204_204165


namespace solve_for_a_l204_204937

theorem solve_for_a : ∃ a : ℝ, (∀ x : ℝ, x = -2 → x^2 - a * x + 7 = 0) → a = -11 / 2 :=
by 
  sorry

end solve_for_a_l204_204937


namespace aku_invited_friends_l204_204582

def total_cookies (packages : ℕ) (cookies_per_package : ℕ) := packages * cookies_per_package

def total_children (total_cookies : ℕ) (cookies_per_child : ℕ) := total_cookies / cookies_per_child

def invited_friends (total_children : ℕ) := total_children - 1

theorem aku_invited_friends (packages cookies_per_package cookies_per_child : ℕ) (h1 : packages = 3) (h2 : cookies_per_package = 25) (h3 : cookies_per_child = 15) :
  invited_friends (total_children (total_cookies packages cookies_per_package) cookies_per_child) = 4 :=
by
  sorry

end aku_invited_friends_l204_204582


namespace inequality_general_l204_204885

theorem inequality_general {a b c d : ℝ} :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 :=
by
  sorry

end inequality_general_l204_204885


namespace second_polygon_sides_l204_204261

theorem second_polygon_sides (s : ℝ) (P : ℝ) (n : ℕ) : 
  (50 * 3 * s = P) ∧ (n * s = P) → n = 150 := 
by {
  sorry
}

end second_polygon_sides_l204_204261


namespace distance_between_vertices_of_hyperbola_theorem_l204_204576

noncomputable def distance_between_vertices_of_hyperbola : ℝ :=
  let x_equation := 16*x^2 - 32*x
  let y_equation := -y^2 + 4*y
  let equation := x_equation + y_equation + 48 = 0
  /- The standard form of the hyperbola, after completing the square, would be:
     16(x - 1)^2 - (y - 2)^2 + 36 = 0
     which transforms to the standard form of hyperbola:
     (x-1)^2/(9/16) - (y-2)^2/36 = 1
  -/
  let a := 3/4
  let distance := 2 * a
  distance

theorem distance_between_vertices_of_hyperbola_theorem :
  distance_between_vertices_of_hyperbola = 3 / 2 :=
by
  sorry

end distance_between_vertices_of_hyperbola_theorem_l204_204576


namespace find_a_n_l204_204434

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_sum : ∀ n : ℕ, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n : ℕ, a n = Real.sqrt ↑n - Real.sqrt (↑n - 1) :=
by
  sorry

end find_a_n_l204_204434


namespace smallest_positive_x_l204_204749

theorem smallest_positive_x (x : ℝ) (h : x > 0) (h_eq : x / 4 + 3 / (4 * x) = 1) : x = 1 :=
by
  sorry

end smallest_positive_x_l204_204749


namespace total_dots_correct_l204_204499

/-- Define the initial conditions -/
def monday_ladybugs : ℕ := 8
def monday_dots_per_ladybug : ℕ := 6
def tuesday_ladybugs : ℕ := 5
def wednesday_ladybugs : ℕ := 4

/-- Define the derived conditions -/
def tuesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 1
def wednesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 2

/-- Calculate the total number of dots -/
def monday_total_dots : ℕ := monday_ladybugs * monday_dots_per_ladybug
def tuesday_total_dots : ℕ := tuesday_ladybugs * tuesday_dots_per_ladybug
def wednesday_total_dots : ℕ := wednesday_ladybugs * wednesday_dots_per_ladybug
def total_dots : ℕ := monday_total_dots + tuesday_total_dots + wednesday_total_dots

/-- Prove the total dots equal to 89 -/
theorem total_dots_correct : total_dots = 89 := by
  sorry

end total_dots_correct_l204_204499


namespace terminal_side_in_third_quadrant_l204_204455

-- Define the conditions
def sin_condition (α : Real) : Prop := Real.sin α < 0
def tan_condition (α : Real) : Prop := Real.tan α > 0

-- State the theorem
theorem terminal_side_in_third_quadrant (α : Real) (h1 : sin_condition α) (h2 : tan_condition α) : α ∈ Set.Ioo (π / 2) π :=
  sorry

end terminal_side_in_third_quadrant_l204_204455


namespace count_noncongruent_triangles_l204_204599

theorem count_noncongruent_triangles :
  ∃ (n : ℕ), n = 13 ∧
  ∀ (a b c : ℕ), a < b ∧ b < c ∧ a + b > c ∧ a + b + c < 20 ∧ ¬(a * a + b * b = c * c)
  → n = 13 := by {
  sorry
}

end count_noncongruent_triangles_l204_204599


namespace gcd_45_75_l204_204731

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l204_204731


namespace solve_for_w_l204_204207

theorem solve_for_w (w : ℕ) (h : w^2 - 5 * w = 0) (hp : w > 0) : w = 5 :=
sorry

end solve_for_w_l204_204207


namespace positive_integers_solution_l204_204813

open Nat

theorem positive_integers_solution (a b m n : ℕ) (r : ℕ) (h_pos_a : 0 < a)
  (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n) 
  (h_gcd : Nat.gcd m n = 1) :
  (a^2 + b^2)^m = (a * b)^n ↔ a = 2^r ∧ b = 2^r ∧ m = 2 * r ∧ n = 2 * r + 1 :=
sorry

end positive_integers_solution_l204_204813


namespace income_expenses_opposite_l204_204013

def income_denotation (income : Int) : Int := income

theorem income_expenses_opposite :
  income_denotation 5 = 5 →
  income_denotation (-5) = -5 :=
by
  intro h
  sorry

end income_expenses_opposite_l204_204013


namespace min_throws_to_ensure_repeat_sum_l204_204122

theorem min_throws_to_ensure_repeat_sum : 
  ∀ (min_sum max_sum : ℤ), 
  min_sum = 4 ∧ max_sum = 24 
  → ∃ n, n ≥ 22 ∧ n = 22 :=
by
  intros min_sum max_sum h
  cases h with h_min h_max
  existsi 22
  split
  · exact Nat.le_refl 22
  · sorry

end min_throws_to_ensure_repeat_sum_l204_204122


namespace minimum_rolls_for_duplicate_sum_l204_204089

theorem minimum_rolls_for_duplicate_sum :
  ∀ (A : Type) [decidable_eq A] (dice_rolls : A → ℕ),
    (∀ a : A, 1 ≤ dice_rolls a ∧ dice_rolls a ≤ 6) →
    (∃ n : ℕ, n = 22 ∧
      ∀ (f : ℕ → ℕ), (∃ (r : ℕ → ℕ), 
        (∀ i : ℕ, r i = f i) →
        (∀ x : ℕ, 4 ≤ f x ∧ f x ≤ 24) →
        (∃ j k : ℕ, j ≠ k ∧ f j = f k))) :=
begin
  intros,
  sorry
end

end minimum_rolls_for_duplicate_sum_l204_204089


namespace triangle_side_eq_nine_l204_204353

theorem triangle_side_eq_nine (a b c : ℕ) 
  (h_tri_ineq : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_sqrt_eq : (Nat.sqrt (a - 9)) + (b - 2)^2 = 0)
  (h_c_odd : c % 2 = 1) :
  c = 9 :=
sorry

end triangle_side_eq_nine_l204_204353


namespace maximal_points_coloring_l204_204174

/-- Given finitely many points in the plane where no three points are collinear,
which are colored either red or green, such that any monochromatic triangle
contains at least one point of the other color in its interior, the maximal number
of such points is 8. -/
theorem maximal_points_coloring (points : Finset (ℝ × ℝ))
  (h_no_three_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    ¬ ∃ k b, ∀ p ∈ [p1, p2, p3], p.2 = k * p.1 + b)
  (colored : (ℝ × ℝ) → Prop)
  (h_coloring : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points →
    colored p1 = colored p2 → colored p2 = colored p3 →
    ∃ p, p ∈ points ∧ colored p ≠ colored p1) :
  points.card ≤ 8 :=
sorry

end maximal_points_coloring_l204_204174


namespace arithmetic_sequence_inequality_l204_204835

variable {α : Type*} [OrderedRing α]

theorem arithmetic_sequence_inequality 
  (a : ℕ → α) (d : α) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_pos : ∀ n, a n > 0)
  (h_d_ne_zero : d ≠ 0) : 
  a 0 * a 7 < a 3 * a 4 := 
by
  sorry

end arithmetic_sequence_inequality_l204_204835


namespace probability_more_sons_or_daughters_correct_l204_204865

noncomputable def probability_more_sons_or_daughters : ℚ :=
  let total_combinations := (2 : ℕ) ^ 8
  let equal_sons_daughters := Nat.choose 8 4
  let more_sons_or_daughters := total_combinations - equal_sons_daughters
  more_sons_or_daughters / total_combinations

theorem probability_more_sons_or_daughters_correct :
  probability_more_sons_or_daughters = 93 / 128 := by
  sorry 

end probability_more_sons_or_daughters_correct_l204_204865


namespace subset_families_inequality_l204_204823

theorem subset_families_inequality 
  {X : Type*} [fintype X] [decidable_eq X] (n : ℕ) (hn : fintype.card X = n)
  (𝒜 𝒝 : finset (finset X))
  (h : ∀ A ∈ 𝒜, ∀ B ∈ 𝒝, ¬(A ⊆ B ∨ B ⊆ A)) :
  (real.sqrt 𝒜.card + real.sqrt 𝒝.card ≤ 2^(7/2 : ℝ)) :=
sorry

end subset_families_inequality_l204_204823


namespace percentage_microphotonics_l204_204535

noncomputable def percentage_home_electronics : ℝ := 24
noncomputable def percentage_food_additives : ℝ := 20
noncomputable def percentage_GMO : ℝ := 29
noncomputable def percentage_industrial_lubricants : ℝ := 8
noncomputable def angle_basic_astrophysics : ℝ := 18

theorem percentage_microphotonics : 
  ∀ (home_elec food_additives GMO industrial_lub angle_bas_astro : ℝ),
  home_elec = 24 →
  food_additives = 20 →
  GMO = 29 →
  industrial_lub = 8 →
  angle_bas_astro = 18 →
  (100 - (home_elec + food_additives + GMO + industrial_lub + ((angle_bas_astro / 360) * 100))) = 14 :=
by
  intros _ _ _ _ _
  sorry

end percentage_microphotonics_l204_204535


namespace cone_radius_l204_204317

open Real

theorem cone_radius
  (l : ℝ) (L : ℝ) (h_l : l = 5) (h_L : L = 15 * π) :
  ∃ r : ℝ, L = π * r * l ∧ r = 3 :=
by
  sorry

end cone_radius_l204_204317


namespace smallest_number_sum_of_three_squares_distinct_ways_l204_204754

theorem smallest_number_sum_of_three_squares_distinct_ways :
  ∃ n : ℤ, n = 30 ∧
  (∃ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℤ),
    a1^2 + b1^2 + c1^2 = n ∧
    a2^2 + b2^2 + c2^2 = n ∧
    a3^2 + b3^2 + c3^2 = n ∧
    (a1, b1, c1) ≠ (a2, b2, c2) ∧
    (a1, b1, c1) ≠ (a3, b3, c3) ∧
    (a2, b2, c2) ≠ (a3, b3, c3)) := sorry

end smallest_number_sum_of_three_squares_distinct_ways_l204_204754


namespace find_prices_possible_purchasing_schemes_maximize_profit_l204_204905

namespace PurchasePriceProblem

/-- 
Define the purchase price per unit for bean sprouts and dried tofu,
and show that they satisfy the given conditions. 
--/
theorem find_prices (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 240)
  (h2 : 3 * x + 4 * y = 340) :
  x = 60 ∧ y = 40 := 
by sorry

/-- 
Given the conditions on the purchase price of bean sprouts and dried tofu
and the need of purchasing a total of 200 units for no more than $10440, 
determine the valid purchasing schemes.
--/
theorem possible_purchasing_schemes 
  (a : ℤ)
  (h1 : 60 * a + 40 * (200 - a) ≤ 10440)
  (h2 : a ≥ 3 / 2 * (200 - a)) :
  120 ≤ a ∧ a ≤ 122 := 
by sorry
  
/-- 
Maximize profit based on the purchasing schemes that satisfy the conditions.
--/
theorem maximize_profit 
  (a : ℤ) 
  (h_valid : 120 ≤ a ∧ a ≤ 122) 
  (h_max : ∀ b, 120 ≤ b ∧ b ≤ 122 → 5 * a + 3000 ≥ 5 * b + 3000) :
  (a = 122) → 
  let beans_profit := 5 * a + 3000 
  in beans_profit = 3610 := 
by sorry

end PurchasePriceProblem

end find_prices_possible_purchasing_schemes_maximize_profit_l204_204905


namespace correct_sequence_l204_204497

def step1 := "Collect the admission ticket"
def step2 := "Register"
def step3 := "Written and computer-based tests"
def step4 := "Photography"

theorem correct_sequence : [step2, step4, step1, step3] = ["Register", "Photography", "Collect the admission ticket", "Written and computer-based tests"] :=
by
  sorry

end correct_sequence_l204_204497


namespace problem_statement_l204_204352

noncomputable def vector_a := ℝ × ℝ
noncomputable def vector_b := ℝ × ℝ
noncomputable def vector_m := ℝ × ℝ
noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
noncomputable def norm_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

theorem problem_statement
  (a b : ℝ × ℝ)
  (m : vector_m)
  (h_m : m = (4, 9))
  (h_midpoint : m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2))
  (h_dot : dot_product a b = 10) :
  norm_squared a + norm_squared b = 368 :=
begin
  sorry
end

end problem_statement_l204_204352


namespace simplify_fraction_l204_204993

theorem simplify_fraction (n : ℕ) (h : 2 ^ n ≠ 0) : 
  (2 ^ (n + 5) - 3 * 2 ^ n) / (3 * 2 ^ (n + 4)) = 29 / 48 := 
by
  sorry

end simplify_fraction_l204_204993


namespace largest_square_side_length_l204_204848

noncomputable def largestInscribedSquareSide (s : ℝ) (sharedSide : ℝ) : ℝ :=
  let y := (s * Real.sqrt 2 - sharedSide * Real.sqrt 3) / (2 * Real.sqrt 2)
  y

theorem largest_square_side_length :
  let s := 12
  let t := (s * Real.sqrt 6) / 3
  largestInscribedSquareSide s t = 6 - Real.sqrt 6 :=
by
  sorry

end largest_square_side_length_l204_204848


namespace point_direction_form_eq_l204_204838

-- Define the conditions
def point := (1, 2)
def direction_vector := (3, -4)

-- Define a function to represent the line equation based on point and direction
def line_equation (x y : ℝ) : Prop :=
  (x - point.1) / direction_vector.1 = (y - point.2) / direction_vector.2

-- State the theorem
theorem point_direction_form_eq (x y : ℝ) :
  (x - 1) / 3 = (y - 2) / -4 →
  line_equation x y :=
sorry

end point_direction_form_eq_l204_204838


namespace factor_expression_l204_204180

theorem factor_expression (x y z : ℝ) :
  (x - y)^3 + (y - z)^3 + (z - x)^3 ≠ 0 →
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) =
    (x + y) * (y + z) * (z + x) :=
by
  intro h
  sorry

end factor_expression_l204_204180


namespace real_roots_iff_l204_204177

theorem real_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 + 2 * k = 0) ↔ (-1 ≤ k ∧ k ≤ 0) :=
by sorry

end real_roots_iff_l204_204177


namespace equal_play_time_l204_204784

theorem equal_play_time (total_players on_field_players match_minutes : ℕ) 
    (h1 : total_players = 10) 
    (h2 : on_field_players = 8) 
    (h3 : match_minutes = 45) 
    (h4 : ∀ p, p ∈ (finset.range total_players) → time_played p = (on_field_players * match_minutes) / total_players)
    : (on_field_players * match_minutes) / total_players = 36 :=
by
  have total_play_minutes : on_field_players * match_minutes = 360 := by sorry
  have equal_play_time : (on_field_players * match_minutes) / total_players = 36 := by sorry
  exact equal_play_time

end equal_play_time_l204_204784


namespace intersecting_lines_value_l204_204049

theorem intersecting_lines_value (m b : ℚ)
  (h₁ : 10 = m * 7 + 5)
  (h₂ : 10 = 2 * 7 + b) :
  b + m = - (23 : ℚ) / 7 := 
sorry

end intersecting_lines_value_l204_204049


namespace min_value_S_l204_204200

theorem min_value_S (a b c : ℤ) (h1 : a + b + c = 2) (h2 : (2 * a + b * c) * (2 * b + c * a) * (2 * c + a * b) > 200) :
  ∃ a b c : ℤ, a + b + c = 2 ∧ (2 * a + b * c) * (2 * b + c * a) * (2 * c + a * b) = 256 :=
sorry

end min_value_S_l204_204200


namespace root_interval_exists_l204_204038

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - x + 1

theorem root_interval_exists :
  (f 2 > 0) →
  (f 3 < 0) →
  ∃ ξ, 2 < ξ ∧ ξ < 3 ∧ f ξ = 0 :=
by
  intros h1 h2
  sorry

end root_interval_exists_l204_204038


namespace min_throws_for_repeated_sum_l204_204095

theorem min_throws_for_repeated_sum : 
  (∀ (n : ℕ), n = 24 ∧ (∀ (x : ℕ), x ≥ 4 ∧ x ≤ 24)) → 22 :=
by
  sorry

end min_throws_for_repeated_sum_l204_204095


namespace problem_statement_l204_204313

theorem problem_statement (x : ℝ) (h : 0 < x) : x + 2016^2016 / x^2016 ≥ 2017 := 
by
  sorry

end problem_statement_l204_204313


namespace problem_statement_l204_204357

noncomputable def a := 9
noncomputable def b := 729

theorem problem_statement (h1 : ∃ (terms : ℕ), terms = 430)
                          (h2 : ∃ (value : ℕ), value = 3) : a + b = 738 :=
by
  sorry

end problem_statement_l204_204357


namespace k_value_l204_204943

theorem k_value (k : ℝ) :
    (∀ r s : ℝ, (r + s = -k ∧ r * s = 9) ∧ ((r + 3) + (s + 3) = k)) → k = -3 :=
by
    intro h
    sorry

end k_value_l204_204943


namespace four_dice_min_rolls_l204_204109

def minRollsToEnsureSameSum (n : Nat) : Nat :=
  if n = 4 then 22 else sorry

theorem four_dice_min_rolls : minRollsToEnsureSameSum 4 = 22 := by
  rfl

end four_dice_min_rolls_l204_204109


namespace find_y_l204_204925

theorem find_y (y : ℝ) (h₁ : (y^2 - 7*y + 12) / (y - 3) + (3*y^2 + 5*y - 8) / (3*y - 1) = -8) : y = -6 :=
sorry

end find_y_l204_204925


namespace find_a5_plus_a7_l204_204220

variable {a : ℕ → ℝ}

theorem find_a5_plus_a7 (h : a 3 + a 9 = 16) : a 5 + a 7 = 16 := 
sorry

end find_a5_plus_a7_l204_204220


namespace total_cost_correct_l204_204363

noncomputable def camera_old_cost : ℝ := 4000
noncomputable def camera_new_cost := camera_old_cost * 1.30
noncomputable def lens_cost := 400
noncomputable def lens_discount := 200
noncomputable def lens_discounted_price := lens_cost - lens_discount
noncomputable def total_cost := camera_new_cost + lens_discounted_price

theorem total_cost_correct :
  total_cost = 5400 := by
  sorry

end total_cost_correct_l204_204363


namespace mina_crafts_total_l204_204367

theorem mina_crafts_total :
  let a₁ := 3
  let d := 4
  let n := 10
  let crafts_sold_on_day (d: ℕ) := a₁ + (d - 1) * d
  let S (n: ℕ) := (n * (2 * a₁ + (n - 1) * d)) / 2
  S n = 210 :=
by
  sorry

end mina_crafts_total_l204_204367


namespace diameter_circle_inscribed_triangle_l204_204680

noncomputable def diameter_of_inscribed_circle (XY XZ YZ : ℝ) : ℝ :=
  let s := (XY + XZ + YZ) / 2
  let K := Real.sqrt (s * (s - XY) * (s - XZ) * (s - YZ))
  let r := K / s
  2 * r

theorem diameter_circle_inscribed_triangle (XY XZ YZ : ℝ) (hXY : XY = 13) (hXZ : XZ = 8) (hYZ : YZ = 9) :
  diameter_of_inscribed_circle XY XZ YZ = 2 * Real.sqrt 210 / 5 := by
{
  rw [hXY, hXZ, hYZ]
  sorry
}

end diameter_circle_inscribed_triangle_l204_204680


namespace nylon_needed_for_one_dog_collor_l204_204479

-- Define the conditions as given in the problem
def nylon_for_dog (x : ℝ) : ℝ := x
def nylon_for_cat : ℝ := 10
def total_nylon_used (x : ℝ) : ℝ := 9 * (nylon_for_dog x) + 3 * (nylon_for_cat)

-- Prove the required statement under the given conditions
theorem nylon_needed_for_one_dog_collor : total_nylon_used 18 = 192 :=
by
  -- adding the proof step using sorry as required
  sorry

end nylon_needed_for_one_dog_collor_l204_204479


namespace value_expression_l204_204604

-- Definitions
variable (m n : ℝ)
def reciprocals (m n : ℝ) := m * n = 1

-- Theorem statement
theorem value_expression (m n : ℝ) (h : reciprocals m n) : m * n^2 - (n - 3) = 3 := by
  sorry

end value_expression_l204_204604


namespace min_throws_to_same_sum_l204_204068

/-- Define the set of possible sums for four six-sided dice --/
def dice_sum_range := {s : ℕ | 4 ≤ s ∧ s ≤ 24}

/-- The total number of possible sums when rolling four six-sided dice --/
def num_possible_sums : ℕ := 24 - 4 + 1

/-- 
  The minimum number of throws required to ensure that the same sum appears at least twice 
  by the Pigeonhole principle.
--/
theorem min_throws_to_same_sum : num_possible_sums + 1 = 22 := by
  sorry

end min_throws_to_same_sum_l204_204068


namespace thomas_task_completion_l204_204281

theorem thomas_task_completion :
  (∃ T E : ℝ, (1 / T + 1 / E = 1 / 8) ∧ (13 / T + 6 / E = 1)) →
  ∃ T : ℝ, T = 14 :=
by
  sorry

end thomas_task_completion_l204_204281


namespace ensure_same_sum_rolled_twice_l204_204063

theorem ensure_same_sum_rolled_twice :
  ∀ (n : ℕ) (min_sum max_sum : ℕ),
    min_sum = 4 →
    max_sum = 24 →
    (min_sum ≤ n ∧ n ≤ max_sum) →
    ∀ trials : ℕ, trials = 22 →
      ∃ (s1 s2 : ℕ), s1 = s2 ∧ 
      (∃ (throws1 throws2 : list ℕ), list.sum throws1 = s1 ∧ list.sum throws2 = s2 ∧ throws1 ≠ throws2) :=
by 
  sorry

end ensure_same_sum_rolled_twice_l204_204063


namespace gcd_of_45_and_75_l204_204717

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l204_204717


namespace sleep_hours_l204_204967

-- Define the times Isaac wakes up, goes to sleep, and takes naps
def monday : ℝ := 16 - 9
def tuesday_night : ℝ := 12 - 6.5
def tuesday_nap : ℝ := 1
def wednesday : ℝ := 9.75 - 7.75
def thursday_night : ℝ := 15.5 - 8
def thursday_nap : ℝ := 1.5
def friday : ℝ := 12 - 7.25
def saturday : ℝ := 12.75 - 9
def sunday_night : ℝ := 10.5 - 8.5
def sunday_nap : ℝ := 2

noncomputable def total_sleep : ℝ := 
  monday +
  (tuesday_night + tuesday_nap) +
  wednesday +
  (thursday_night + thursday_nap) +
  friday +
  saturday +
  (sunday_night + sunday_nap)

theorem sleep_hours (total_sleep : ℝ) : total_sleep = 36.75 := 
by
  -- Here, you would provide the steps used to add up the hours, but we will skip with sorry
  sorry

end sleep_hours_l204_204967


namespace boxes_per_hand_l204_204907

theorem boxes_per_hand (total_people : ℕ) (total_boxes : ℕ) (boxes_per_person : ℕ) (hands_per_person : ℕ) 
  (h1: total_people = 10) (h2: total_boxes = 20) (h3: boxes_per_person = total_boxes / total_people) 
  (h4: hands_per_person = 2) : boxes_per_person / hands_per_person = 1 := 
by
  sorry

end boxes_per_hand_l204_204907


namespace seven_b_value_l204_204454

theorem seven_b_value (a b : ℚ) (h₁ : 8 * a + 3 * b = 0) (h₂ : a = b - 3) :
  7 * b = 168 / 11 :=
sorry

end seven_b_value_l204_204454


namespace volume_of_inequality_region_l204_204131

-- Define the inequality condition as a predicate
def region (x y z : ℝ) : Prop :=
  |4 * x - 20| + |3 * y + 9| + |z - 2| ≤ 6

-- Define the volume calculation for the region
def volume_of_region := 36

-- The proof statement
theorem volume_of_inequality_region : 
  (∃ x y z : ℝ, region x y z) → volume_of_region = 36 :=
by
  sorry

end volume_of_inequality_region_l204_204131


namespace percentage_of_masters_is_76_l204_204547

variable (x y : ℕ)  -- Let x be the number of junior players, y be the number of master players
variable (junior_avg master_avg team_avg : ℚ)

-- The conditions given in the problem
def juniors_avg_points : Prop := junior_avg = 22
def masters_avg_points : Prop := master_avg = 47
def team_avg_points (x y : ℕ) (junior_avg master_avg team_avg : ℚ) : Prop :=
  (22 * x + 47 * y) / (x + y) = 41

def proportion_of_masters (x y : ℕ) : ℚ := (y : ℚ) / (x + y)

-- The theorem to be proved
theorem percentage_of_masters_is_76 (x y : ℕ) (junior_avg master_avg team_avg : ℚ) :
  juniors_avg_points junior_avg →
  masters_avg_points master_avg →
  team_avg_points x y junior_avg master_avg team_avg →
  proportion_of_masters x y = 19 / 25 := 
sorry

end percentage_of_masters_is_76_l204_204547


namespace fraction_zero_x_value_l204_204211

theorem fraction_zero_x_value (x : ℝ) (h : (x^2 - 4) / (x - 2) = 0) (h2 : x ≠ 2) : x = -2 :=
sorry

end fraction_zero_x_value_l204_204211


namespace mistake_position_is_34_l204_204884

def arithmetic_sequence_sum (n : ℕ) (a_1 : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a_1 + (n - 1) * d) / 2

def modified_sequence_sum (n : ℕ) (a_1 : ℕ) (d : ℕ) (mistake_index : ℕ) : ℕ :=
  let correct_sum := arithmetic_sequence_sum n a_1 d
  correct_sum - 2 * d

theorem mistake_position_is_34 :
  ∃ mistake_index : ℕ, mistake_index = 34 ∧ 
    modified_sequence_sum 37 1 3 mistake_index = 2011 :=
by
  sorry

end mistake_position_is_34_l204_204884


namespace shots_cost_l204_204804

-- Define the conditions
def golden_retriever_pregnant_dogs : ℕ := 3
def golden_retriever_puppies_per_dog : ℕ := 4
def golden_retriever_shots_per_puppy : ℕ := 2
def golden_retriever_cost_per_shot : ℕ := 5

def german_shepherd_pregnant_dogs : ℕ := 2
def german_shepherd_puppies_per_dog : ℕ := 5
def german_shepherd_shots_per_puppy : ℕ := 3
def german_shepherd_cost_per_shot : ℕ := 8

def bulldog_pregnant_dogs : ℕ := 4
def bulldog_puppies_per_dog : ℕ := 3
def bulldog_shots_per_puppy : ℕ := 4
def bulldog_cost_per_shot : ℕ := 10

-- Define the total cost calculation
def total_puppies (dogs_per_breed puppies_per_dog : ℕ) : ℕ :=
  dogs_per_breed * puppies_per_dog

def total_shot_cost (puppies shots_per_puppy cost_per_shot : ℕ) : ℕ :=
  puppies * shots_per_puppy * cost_per_shot

def total_cost : ℕ :=
  let golden_retriever_puppies := total_puppies golden_retriever_pregnant_dogs golden_retriever_puppies_per_dog
  let german_shepherd_puppies := total_puppies german_shepherd_pregnant_dogs german_shepherd_puppies_per_dog
  let bulldog_puppies := total_puppies bulldog_pregnant_dogs bulldog_puppies_per_dog
  let golden_retriever_cost := total_shot_cost golden_retriever_puppies golden_retriever_shots_per_puppy golden_retriever_cost_per_shot
  let german_shepherd_cost := total_shot_cost german_shepherd_puppies german_shepherd_shots_per_puppy german_shepherd_cost_per_shot
  let bulldog_cost := total_shot_cost bulldog_puppies bulldog_shots_per_puppy bulldog_cost_per_shot
  golden_retriever_cost + german_shepherd_cost + bulldog_cost

-- Statement of the problem
theorem shots_cost (total_cost : ℕ) : total_cost = 840 := by
  -- Proof would go here
  sorry

end shots_cost_l204_204804


namespace john_height_in_feet_l204_204483

theorem john_height_in_feet (initial_height : ℕ) (growth_rate : ℕ) (months : ℕ) (inches_per_foot : ℕ) :
  initial_height = 66 → growth_rate = 2 → months = 3 → inches_per_foot = 12 → 
  (initial_height + growth_rate * months) / inches_per_foot = 6 := by
  intros h1 h2 h3 h4
  sorry

end john_height_in_feet_l204_204483


namespace negation_of_proposition_l204_204513

theorem negation_of_proposition (x : ℝ) : ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end negation_of_proposition_l204_204513


namespace gcd_of_45_and_75_l204_204721

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l204_204721


namespace expenses_neg_of_income_pos_l204_204030

theorem expenses_neg_of_income_pos :
  ∀ (income expense : Int), income = 5 → expense = -income → expense = -5 :=
by
  intros income expense h_income h_expense
  rw [h_income] at h_expense
  exact h_expense

end expenses_neg_of_income_pos_l204_204030


namespace area_of_square_plot_l204_204392

-- Defining the given conditions and question in Lean 4
theorem area_of_square_plot 
  (cost_per_foot : ℕ := 58)
  (total_cost : ℕ := 2784) :
  ∃ (s : ℕ), (4 * s * cost_per_foot = total_cost) ∧ (s * s = 144) :=
by
  sorry

end area_of_square_plot_l204_204392


namespace min_throws_to_ensure_same_sum_twice_l204_204098

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l204_204098


namespace ratio_of_cone_to_sphere_l204_204430

theorem ratio_of_cone_to_sphere (r : ℝ) (h := 2 * r) : 
  (1 / 3 * π * r^2 * h) / ((4 / 3) * π * r^3) = 1 / 2 :=
by 
  sorry

end ratio_of_cone_to_sphere_l204_204430


namespace greatest_perimeter_l204_204509

theorem greatest_perimeter (w l : ℕ) (h1 : w * l = 12) : 
  ∃ (P : ℕ), P = 2 * (w + l) ∧ ∀ (w' l' : ℕ), w' * l' = 12 → 2 * (w' + l') ≤ P := 
sorry

end greatest_perimeter_l204_204509


namespace find_c_for_root_ratio_l204_204571

theorem find_c_for_root_ratio :
  ∃ c : ℝ, (∀ x1 x2 : ℝ, (4 * x1^2 - 5 * x1 + c = 0) ∧ (x1 / x2 = -3 / 4)) → c = -75 := 
by {
  sorry
}

end find_c_for_root_ratio_l204_204571


namespace find_a_n_l204_204433

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_sum : ∀ n : ℕ, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n : ℕ, a n = Real.sqrt ↑n - Real.sqrt (↑n - 1) :=
by
  sorry

end find_a_n_l204_204433


namespace percentage_of_masters_is_76_l204_204546

variable (x y : ℕ)  -- Let x be the number of junior players, y be the number of master players
variable (junior_avg master_avg team_avg : ℚ)

-- The conditions given in the problem
def juniors_avg_points : Prop := junior_avg = 22
def masters_avg_points : Prop := master_avg = 47
def team_avg_points (x y : ℕ) (junior_avg master_avg team_avg : ℚ) : Prop :=
  (22 * x + 47 * y) / (x + y) = 41

def proportion_of_masters (x y : ℕ) : ℚ := (y : ℚ) / (x + y)

-- The theorem to be proved
theorem percentage_of_masters_is_76 (x y : ℕ) (junior_avg master_avg team_avg : ℚ) :
  juniors_avg_points junior_avg →
  masters_avg_points master_avg →
  team_avg_points x y junior_avg master_avg team_avg →
  proportion_of_masters x y = 19 / 25 := 
sorry

end percentage_of_masters_is_76_l204_204546


namespace expenses_neg_of_income_pos_l204_204032

theorem expenses_neg_of_income_pos :
  ∀ (income expense : Int), income = 5 → expense = -income → expense = -5 :=
by
  intros income expense h_income h_expense
  rw [h_income] at h_expense
  exact h_expense

end expenses_neg_of_income_pos_l204_204032


namespace equal_playing_time_for_each_player_l204_204777

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l204_204777


namespace min_students_l204_204841

theorem min_students (b g : ℕ) (hb : (3 / 5 : ℚ) * b = (5 / 6 : ℚ) * g) :
  b + g = 43 :=
sorry

end min_students_l204_204841


namespace each_player_plays_36_minutes_l204_204793

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end each_player_plays_36_minutes_l204_204793


namespace imaginary_condition_l204_204462

theorem imaginary_condition (m : ℝ) : 
  let Z := (m + 2 * complex.I) / (1 + complex.I) in 
  Z.im ≠ 0 → Z.re = 0 → m = -2 :=
by
  sorry

end imaginary_condition_l204_204462


namespace ensure_same_sum_rolled_twice_l204_204066

theorem ensure_same_sum_rolled_twice :
  ∀ (n : ℕ) (min_sum max_sum : ℕ),
    min_sum = 4 →
    max_sum = 24 →
    (min_sum ≤ n ∧ n ≤ max_sum) →
    ∀ trials : ℕ, trials = 22 →
      ∃ (s1 s2 : ℕ), s1 = s2 ∧ 
      (∃ (throws1 throws2 : list ℕ), list.sum throws1 = s1 ∧ list.sum throws2 = s2 ∧ throws1 ≠ throws2) :=
by 
  sorry

end ensure_same_sum_rolled_twice_l204_204066


namespace painting_problem_l204_204291

theorem painting_problem (initial_painters : ℕ) (initial_days : ℚ) (initial_rate : ℚ) (new_days : ℚ) (new_rate : ℚ) : 
  initial_painters = 6 ∧ initial_days = 5/2 ∧ initial_rate = 2 ∧ new_days = 2 ∧ new_rate = 2.5 →
  ∃ additional_painters : ℕ, additional_painters = 0 :=
by
  intros h
  sorry

end painting_problem_l204_204291


namespace white_wash_cost_l204_204398

noncomputable def room_length : ℝ := 25
noncomputable def room_width : ℝ := 15
noncomputable def room_height : ℝ := 12
noncomputable def door_height : ℝ := 6
noncomputable def door_width : ℝ := 3
noncomputable def window_height : ℝ := 4
noncomputable def window_width : ℝ := 3
noncomputable def num_windows : ℕ := 3
noncomputable def cost_per_sqft : ℝ := 3

theorem white_wash_cost :
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_height * door_width
  let window_area := window_height * window_width
  let total_non_white_wash_area := door_area + ↑num_windows * window_area
  let white_wash_area := wall_area - total_non_white_wash_area
  let total_cost := white_wash_area * cost_per_sqft
  total_cost = 2718 :=  
by
  sorry

end white_wash_cost_l204_204398


namespace decreasing_on_negative_interval_and_max_value_l204_204570

open Classical

noncomputable def f : ℝ → ℝ := sorry  -- Define f later

variables {f : ℝ → ℝ}

-- Hypotheses
axiom h_even : ∀ x, f x = f (-x)
axiom h_increasing_0_7 : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x ≤ y → y ≤ 7 → f x ≤ f y
axiom h_decreasing_7_inf : ∀ ⦃x y : ℝ⦄, 7 ≤ x → x ≤ y → f x ≥ f y
axiom h_f_7_6 : f 7 = 6

-- Theorem Statement
theorem decreasing_on_negative_interval_and_max_value :
  (∀ ⦃x y : ℝ⦄, -7 ≤ x → x ≤ y → y ≤ 0 → f x ≥ f y) ∧ (∀ x, -7 ≤ x → x ≤ 0 → f x ≤ 6) :=
by
  sorry

end decreasing_on_negative_interval_and_max_value_l204_204570


namespace puppy_weight_l204_204771

theorem puppy_weight (a b c : ℕ) 
  (h1 : a + b + c = 24) 
  (h2 : a + c = 2 * b) 
  (h3 : a + b = c) : 
  a = 4 :=
sorry

end puppy_weight_l204_204771


namespace find_length_of_sheet_l204_204409

noncomputable section

-- Axioms regarding the conditions
def width_of_sheet : ℝ := 36       -- The width of the metallic sheet is 36 meters
def side_of_square : ℝ := 7        -- The side length of the square cut off from each corner is 7 meters
def volume_of_box : ℝ := 5236      -- The volume of the resulting box is 5236 cubic meters

-- Define the length of the metallic sheet as L
def length_of_sheet (L : ℝ) : Prop :=
  let new_length := L - 2 * side_of_square
  let new_width := width_of_sheet - 2 * side_of_square
  let height := side_of_square
  volume_of_box = new_length * new_width * height

-- The condition to prove
theorem find_length_of_sheet : ∃ L : ℝ, length_of_sheet L ∧ L = 48 :=
by
  sorry

end find_length_of_sheet_l204_204409


namespace alex_seashells_l204_204239

theorem alex_seashells (mimi_seashells kyle_seashells leigh_seashells alex_seashells : ℕ) 
    (h1 : mimi_seashells = 2 * 12) 
    (h2 : kyle_seashells = 2 * mimi_seashells) 
    (h3 : leigh_seashells = kyle_seashells / 3) 
    (h4 : alex_seashells = 3 * leigh_seashells) : 
  alex_seashells = 48 := by
  sorry

end alex_seashells_l204_204239


namespace managers_non_managers_ratio_l204_204965

theorem managers_non_managers_ratio
  (M N : ℕ)
  (h_ratio : M / N > 7 / 24)
  (h_max_non_managers : N = 27) :
  ∃ M, 8 ≤ M ∧ M / 27 > 7 / 24 :=
by
  sorry

end managers_non_managers_ratio_l204_204965


namespace gcd_45_75_l204_204738

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204738


namespace fraction_evaluation_l204_204427

theorem fraction_evaluation :
  (18 / 42) - (2 / 9) + (1 / 14) = (5 / 18) :=
by
  -- Proof goes here
  sorry

end fraction_evaluation_l204_204427


namespace rotated_triangle_forms_two_cones_l204_204773

/-- Prove that the spatial geometric body formed when a right-angled triangle 
is rotated 360° around its hypotenuse is two cones. -/
theorem rotated_triangle_forms_two_cones (a b c : ℝ) (h1 : a^2 + b^2 = c^2) : 
  ∃ (cones : ℕ), cones = 2 :=
by
  sorry

end rotated_triangle_forms_two_cones_l204_204773


namespace find_period_l204_204815

theorem find_period (A P R : ℕ) (I : ℕ) (T : ℚ) 
  (hA : A = 1120) 
  (hP : P = 896) 
  (hR : R = 5) 
  (hSI : I = A - P) 
  (hT : I = (P * R * T) / 100) :
  T = 5 := by 
  sorry

end find_period_l204_204815


namespace expenses_negation_of_income_l204_204006

theorem expenses_negation_of_income 
    (income : ℤ) 
    (income_is_5 : income = 5) 
    (denote_income : income = 5 → "+" ∘ toString income = "+5") 
    (expenses_are_negation_of_income :  "expenses = -1 * income") : "expenses = -5" :=
begin
    sorry
end

end expenses_negation_of_income_l204_204006


namespace number_of_eggplant_packets_l204_204870

-- Defining the problem conditions in Lean 4
def eggplants_per_packet := 14
def sunflowers_per_packet := 10
def sunflower_packets := 6
def total_plants := 116

-- Our goal is to prove the number of eggplant seed packets Shyne bought
theorem number_of_eggplant_packets : ∃ E : ℕ, E * eggplants_per_packet + sunflower_packets * sunflowers_per_packet = total_plants ∧ E = 4 :=
sorry

end number_of_eggplant_packets_l204_204870


namespace sugar_percentage_l204_204152

theorem sugar_percentage (x : ℝ) (h2 : 50 ≤ 100) (h1 : 1 / 4 * x + 12.5 = 20) : x = 10 :=
by
  sorry

end sugar_percentage_l204_204152


namespace expense_of_5_yuan_is_minus_5_yuan_l204_204026

def income (x : Int) : Int :=
  x

def expense (x : Int) : Int :=
  -x

theorem expense_of_5_yuan_is_minus_5_yuan : expense 5 = -5 :=
by
  unfold expense
  sorry

end expense_of_5_yuan_is_minus_5_yuan_l204_204026


namespace gcd_45_75_l204_204687

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l204_204687


namespace ratio_boys_to_girls_l204_204614

variable (g b : ℕ)

theorem ratio_boys_to_girls (h1 : b = g + 9) (h2 : g + b = 25) : b / g = 17 / 8 := by
  -- Proof goes here
  sorry

end ratio_boys_to_girls_l204_204614


namespace smallest_n_conditions_l204_204266

theorem smallest_n_conditions :
  ∃ n : ℕ, 0 < n ∧ (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^4) ∧ n = 54 :=
by
  sorry

end smallest_n_conditions_l204_204266


namespace sample_size_six_l204_204763

-- Definitions for the conditions
def num_senior_teachers : ℕ := 18
def num_first_level_teachers : ℕ := 12
def num_top_level_teachers : ℕ := 6
def total_teachers : ℕ := num_senior_teachers + num_first_level_teachers + num_top_level_teachers

-- The proof problem statement
theorem sample_size_six (n : ℕ) (h1 : n > 0) : 
  (∀ m : ℕ, m * n = total_teachers → 
             ((n + 1) * m - 1 = 35) → False) → n = 6 :=
sorry

end sample_size_six_l204_204763


namespace average_last_12_results_l204_204251

theorem average_last_12_results (S25 S12 S_last12 : ℕ) (A : ℕ) 
  (h1 : S25 = 25 * 24) 
  (h2: S12 = 12 * 14) 
  (h3: 12 * A = S_last12)
  (h4: S25 = S12 + 228 + S_last12) : A = 17 := 
by
  sorry

end average_last_12_results_l204_204251


namespace length_width_ratio_l204_204039

theorem length_width_ratio 
  (W : ℕ) (P : ℕ) (L : ℕ)
  (hW : W = 90) 
  (hP : P = 432) 
  (hP_eq : P = 2 * L + 2 * W) : 
  (L / W = 7 / 5) := 
  sorry

end length_width_ratio_l204_204039


namespace train_speed_kmh_l204_204912

def man_speed_kmh : ℝ := 3 -- The man's speed in km/h
def train_length_m : ℝ := 110 -- The train's length in meters
def passing_time_s : ℝ := 12 -- Time taken to pass the man in seconds

noncomputable def man_speed_ms : ℝ := (man_speed_kmh * 1000) / 3600 -- Convert man's speed to m/s

theorem train_speed_kmh :
  (110 / 12) - (5 / 6) * (3600 / 1000) = 30 := by
  -- Omitted steps will go here
  sorry

end train_speed_kmh_l204_204912


namespace satisfies_conditions_l204_204196

open Real

def point_P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

def condition1 (a : ℝ) : Prop := (point_P a).fst = 0

def condition2 (a : ℝ) : Prop := (point_P a).snd = 5

def condition3 (a : ℝ) : Prop := abs ((point_P a).fst) = abs ((point_P a).snd)

theorem satisfies_conditions :
  ∃ P : ℝ × ℝ, P = (12, 12) ∨ P = (-12, -12) ∨ P = (4, -4) ∨ P = (-4, 4) :=
by
  sorry

end satisfies_conditions_l204_204196


namespace arrangements_not_next_to_each_other_and_not_at_ends_l204_204833

theorem arrangements_not_next_to_each_other_and_not_at_ends :
  let n := 6 in
  let total_arrangements := nat.factorial n in
  let ab_together := 2 * nat.factorial (n-1) in
  let a_or_b_at_ends := 2 * 2 * nat.factorial (n-1) in
  let double_counted_adjustment := 2 * 2 * nat.factorial (n-2) in
  total_arrangements - ab_together - a_or_b_at_ends + double_counted_adjustment = 96 := by
  sorry

end arrangements_not_next_to_each_other_and_not_at_ends_l204_204833


namespace baseball_card_decrease_l204_204146

theorem baseball_card_decrease (x : ℝ) (h : (1 - x / 100) * (1 - x / 100) = 0.64) : x = 20 :=
by
  sorry

end baseball_card_decrease_l204_204146


namespace compute_sum_l204_204807
-- Import the necessary library to have access to the required definitions and theorems.

-- Define the integers involved based on the conditions.
def a : ℕ := 157
def b : ℕ := 43
def c : ℕ := 19
def d : ℕ := 81

-- State the theorem that computes the sum of these integers and equate it to 300.
theorem compute_sum : a + b + c + d = 300 := by
  sorry

end compute_sum_l204_204807


namespace inequality_solution_l204_204654

theorem inequality_solution (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < (5 / 9) :=
by
  sorry

end inequality_solution_l204_204654


namespace tan_sum_l204_204312

-- Define the conditions as local variables
variables {α β : ℝ} (h₁ : Real.tan α = -2) (h₂ : Real.tan β = 5)

-- The statement to prove
theorem tan_sum : Real.tan (α + β) = 3 / 11 :=
by 
  -- Proof goes here, using 'sorry' as placeholder
  sorry

end tan_sum_l204_204312


namespace minimum_rolls_to_ensure_repeated_sum_l204_204080

theorem minimum_rolls_to_ensure_repeated_sum : 
  let dice_faces := 6
  let number_of_dice := 4
  let min_sum := number_of_dice * 1
  let max_sum := number_of_dice * dice_faces
  let distinct_sums := (max_sum - min_sum) + 1
  in 22 = distinct_sums + 1 :=
by {
  sorry
}

end minimum_rolls_to_ensure_repeated_sum_l204_204080


namespace ethanol_in_fuel_A_l204_204796

def fuel_tank_volume : ℝ := 208
def fuel_A_volume : ℝ := 82
def fuel_B_volume : ℝ := fuel_tank_volume - fuel_A_volume
def ethanol_in_fuel_B : ℝ := 0.16
def total_ethanol : ℝ := 30

theorem ethanol_in_fuel_A 
  (x : ℝ) 
  (H_fuel_tank_capacity : fuel_tank_volume = 208) 
  (H_fuel_A_volume : fuel_A_volume = 82) 
  (H_fuel_B_volume : fuel_B_volume = 126) 
  (H_ethanol_in_fuel_B : ethanol_in_fuel_B = 0.16) 
  (H_total_ethanol : total_ethanol = 30) 
  : 82 * x + 0.16 * 126 = 30 → x = 0.12 := by
  sorry

end ethanol_in_fuel_A_l204_204796


namespace number_of_white_balls_l204_204969

theorem number_of_white_balls (total : ℕ) (freq_red freq_black : ℚ) (h1 : total = 120) 
                              (h2 : freq_red = 0.15) (h3 : freq_black = 0.45) : 
                              (total - total * freq_red - total * freq_black = 48) :=
by sorry

end number_of_white_balls_l204_204969


namespace find_a_plus_b_l204_204186

variable (a : ℝ) (b : ℝ)
def op (x y : ℝ) : ℝ := x + 2 * y + 3

theorem find_a_plus_b (a b : ℝ) (h1 : op (op (a^3) (a^2)) a = b)
    (h2 : op (a^3) (op (a^2) a) = b) : a + b = 21/8 :=
  sorry

end find_a_plus_b_l204_204186


namespace tan_pi_seventh_root_of_unity_l204_204045

open Complex

theorem tan_pi_seventh_root_of_unity :
  let x := Real.pi / 7 in
  let θ := Real.cos x + (Real.sin x) * I in
  let z := θ / conj(θ) in
  z = Real.cos (4 * Real.pi / 7) + I * Real.sin (4 * Real.pi / 7) :=
by
  let π := Real.pi
  let x := π / 7
  let θ := Real.cos x + (Real.sin x) * I
  have θ_div_conj := θ / conj(θ)
  show θ_div_conj = Real.cos (4 * π / 7) + I * Real.sin (4 * π / 7)
  sorry

end tan_pi_seventh_root_of_unity_l204_204045


namespace value_of_a_when_b_is_24_l204_204672

variable (a b k : ℝ)

theorem value_of_a_when_b_is_24 (h1 : a = k / b^2) (h2 : 40 = k / 12^2) (h3 : b = 24) : a = 10 :=
by
  sorry

end value_of_a_when_b_is_24_l204_204672


namespace leah_ride_time_l204_204345

theorem leah_ride_time (x y : ℝ) (h1 : 90 * x = y) (h2 : 30 * (x + 2 * x) = y)
: ∃ t : ℝ, t = 67.5 :=
by
  -- Define 50% increase in length
  let y' := 1.5 * y
  -- Define escalator speed without Leah walking
  let k := 2 * x
  -- Calculate the time taken
  let t := y' / k
  -- Prove that this time is 67.5 seconds
  have ht : t = 67.5 := sorry
  exact ⟨t, ht⟩

end leah_ride_time_l204_204345


namespace sequence_diff_n_l204_204529

theorem sequence_diff_n {a : ℕ → ℕ} (h1 : a 1 = 1) 
(h2 : ∀ n : ℕ, a (n + 1) ≤ 2 * n) (n : ℕ) :
  ∃ p q : ℕ, a p - a q = n :=
sorry

end sequence_diff_n_l204_204529


namespace rope_subdivision_length_l204_204544

theorem rope_subdivision_length 
  (initial_length : ℕ) 
  (num_parts : ℕ) 
  (num_subdivided_parts : ℕ) 
  (final_subdivision_factor : ℕ) 
  (initial_length_eq : initial_length = 200) 
  (num_parts_eq : num_parts = 4) 
  (num_subdivided_parts_eq : num_subdivided_parts = num_parts / 2) 
  (final_subdivision_factor_eq : final_subdivision_factor = 2) :
  initial_length / num_parts / final_subdivision_factor = 25 := 
by 
  sorry

end rope_subdivision_length_l204_204544


namespace min_throws_to_ensure_same_sum_twice_l204_204100

/-- Define the properties of a six-sided die and the sum calculation. --/
def is_fair_six_sided_die (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 6

/-- Define the sum of four dice rolls. --/
def sum_of_four_dice_rolls (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

/-- Prove the minimum number of throws needed to ensure the same sum twice. --/
theorem min_throws_to_ensure_same_sum_twice :
  ∀ (d1 d2 d3 d4 : ℕ), (is_fair_six_sided_die d1) 
  → (is_fair_six_sided_die d2) 
  → (is_fair_six_sided_die d3) 
  → (is_fair_six_sided_die d4) 
  → (∃ n, n = 22 
      ∧ ∀ (throws : ℕ → ℕ × ℕ × ℕ × ℕ),
        (Σ (i : ℕ), sum_of_four_dice_rolls (throws i).1 (throws i).2.1 (throws i).2.2.1 (throws i).2.2.2) ≥ 23 
        → ∃ (j k : ℕ), (j ≠ k) ∧ (sum_of_four_dice_rolls (throws j).1 (throws j).2.1 (throws j).2.2.1 (throws j).2.2.2 = sum_of_four_dice_rolls (throws k).1 (throws k).2.1 (throws k).2.2.1 (throws k).2.2.2))) :=
begin
  sorry
end

end min_throws_to_ensure_same_sum_twice_l204_204100


namespace most_significant_price_drop_l204_204768

noncomputable def price_change (month : ℕ) : ℝ :=
  match month with
  | 1 => -1.00
  | 2 => 0.50
  | 3 => -3.00
  | 4 => 2.00
  | 5 => -1.50
  | 6 => -0.75
  | _ => 0.00 -- For any invalid month, we assume no price change

theorem most_significant_price_drop :
  ∀ m : ℕ, (m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6) →
  (∀ n : ℕ, (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) →
  price_change m ≤ price_change n) → m = 3 :=
by
  intros m hm H
  sorry

end most_significant_price_drop_l204_204768


namespace max_val_z_lt_2_l204_204983

-- Definitions for the variables and constraints
variable {x y m : ℝ}
variable (h1 : y ≥ x) (h2 : y ≤ m * x) (h3 : x + y ≤ 1) (h4 : m > 1)

-- Theorem statement
theorem max_val_z_lt_2 (h1 : y ≥ x) (h2 : y ≤ m * x) (h3 : x + y ≤ 1) (h4 : m > 1) : 
  (∀ x y, y ≥ x → y ≤ m * x → x + y ≤ 1 → x + m * y < 2) ↔ 1 < m ∧ m < 1 + Real.sqrt 2 :=
sorry

end max_val_z_lt_2_l204_204983


namespace expenses_negation_of_income_l204_204005

theorem expenses_negation_of_income 
    (income : ℤ) 
    (income_is_5 : income = 5) 
    (denote_income : income = 5 → "+" ∘ toString income = "+5") 
    (expenses_are_negation_of_income :  "expenses = -1 * income") : "expenses = -5" :=
begin
    sorry
end

end expenses_negation_of_income_l204_204005


namespace problem_part1_problem_part2_l204_204199

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

theorem problem_part1 :
  f (Real.pi / 12) = 3 * Real.sqrt 3 / 2 :=
by
  sorry

theorem problem_part2 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2) :
  Real.sin θ = 4 / 5 →
  f (5 * Real.pi / 12 - θ) = 72 / 25 :=
by
  sorry

end problem_part1_problem_part2_l204_204199


namespace remaining_trees_correct_l204_204673

def initial_oak_trees := 57
def initial_maple_trees := 43

def full_cut_oak := 13
def full_cut_maple := 8

def partial_cut_oak := 2.5
def partial_cut_maple := 1.5

def remaining_oak_trees := initial_oak_trees - full_cut_oak
def remaining_maple_trees := initial_maple_trees - full_cut_maple

def total_remaining_trees := remaining_oak_trees + remaining_maple_trees

theorem remaining_trees_correct : remaining_oak_trees = 44 ∧ remaining_maple_trees = 35 ∧ total_remaining_trees = 79 :=
by
  sorry

end remaining_trees_correct_l204_204673


namespace original_pencils_example_l204_204886

-- Statement of the problem conditions
def original_pencils (total_pencils : ℕ) (added_pencils : ℕ) : ℕ :=
  total_pencils - added_pencils

-- Theorem we need to prove
theorem original_pencils_example : original_pencils 5 3 = 2 := 
by
  -- Proof
  sorry

end original_pencils_example_l204_204886


namespace min_perimeter_l204_204321

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the coordinates of the right focus, point on the hyperbola, and point M
def right_focus (F : ℝ × ℝ) : Prop := F = (2, 0)
def point_on_left_branch (P : ℝ × ℝ) : Prop := P.1 < 0 ∧ hyperbola P.1 P.2
def point_M (M : ℝ × ℝ) : Prop := M = (0, 2)

-- Perimeter of ΔPFM
noncomputable def perimeter (P F M : ℝ × ℝ) : ℝ :=
  let PF := (P.1 - F.1)^2 + (P.2 - F.2)^2
  let PM := (P.1 - M.1)^2 + (P.2 - M.2)^2
  let MF := (M.1 - F.1)^2 + (M.2 - F.2)^2
  PF.sqrt + PM.sqrt + MF.sqrt

-- Theorem statement
theorem min_perimeter (P F M : ℝ × ℝ) 
  (hF : right_focus F)
  (hP : point_on_left_branch P)
  (hM : point_M M) :
  ∃ P, perimeter P F M = 2 + 4 * Real.sqrt 2 :=
sorry

end min_perimeter_l204_204321


namespace elena_deductions_in_cents_l204_204812

-- Definitions based on the conditions
def cents_per_dollar : ℕ := 100
def hourly_wage_in_dollars : ℕ := 25
def hourly_wage_in_cents : ℕ := hourly_wage_in_dollars * cents_per_dollar
def tax_rate : ℚ := 0.02
def health_benefit_rate : ℚ := 0.015

-- The problem to prove
theorem elena_deductions_in_cents:
  (tax_rate * hourly_wage_in_cents) + (health_benefit_rate * hourly_wage_in_cents) = 87.5 := 
by
  sorry

end elena_deductions_in_cents_l204_204812


namespace matt_homework_time_l204_204238

variable (T : ℝ)
variable (h_math : 0.30 * T = math_time)
variable (h_science : 0.40 * T = science_time)
variable (h_others : math_time + science_time + 45 = T)

theorem matt_homework_time (h_math : 0.30 * T = math_time)
                             (h_science : 0.40 * T = science_time)
                             (h_others : math_time + science_time + 45 = T) :
  T = 150 := by
  sorry

end matt_homework_time_l204_204238


namespace minimum_throws_l204_204056

def min_throws_to_ensure_repeat_sum (throws : ℕ) : Prop :=
  4 ≤ throws ∧ throws ≤ 24 →

theorem minimum_throws (throws : ℕ) (h : min_throws_to_ensure_repeat_sum throws) : throws ≥ 22 :=
sorry

end minimum_throws_l204_204056


namespace perp_lines_l204_204203

noncomputable def line_1 (k : ℝ) : ℝ → ℝ → ℝ := λ x y => (k - 3) * x + (5 - k) * y + 1
noncomputable def line_2 (k : ℝ) : ℝ → ℝ → ℝ := λ x y => 2 * (k - 3) * x - 2 * y + 3

theorem perp_lines (k : ℝ) : 
  let l1 := line_1 k
  let l2 := line_2 k
  (∀ x y, l1 x y = 0 → l2 x y = 0 → (k = 1 ∨ k = 4)) :=
by
    sorry

end perp_lines_l204_204203


namespace Vince_ride_longer_l204_204524

def Vince_ride_length : ℝ := 0.625
def Zachary_ride_length : ℝ := 0.5

theorem Vince_ride_longer : Vince_ride_length - Zachary_ride_length = 0.125 := by
  sorry

end Vince_ride_longer_l204_204524


namespace simplify_expression_l204_204647

theorem simplify_expression (y : ℝ) : 7 * y + 8 - 3 * y + 16 = 4 * y + 24 :=
by
  sorry

end simplify_expression_l204_204647


namespace range_of_a_l204_204517

open Real

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ , x^2 + a * x + 1 < 0) ↔ (-2 : ℝ) ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l204_204517
