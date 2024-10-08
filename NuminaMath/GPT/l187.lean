import Mathlib

namespace lark_lock_combination_count_l187_187079

-- Definitions for the conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

def lark_lock_combination (a b c : ℕ) : Prop := 
  is_odd a ∧ is_even b ∧ is_multiple_of_5 c ∧ 1 ≤ a ∧ a ≤ 30 ∧ 1 ≤ b ∧ b ≤ 30 ∧ 1 ≤ c ∧ c ≤ 30

-- The core theorem
theorem lark_lock_combination_count : 
  (∃ a b c : ℕ, lark_lock_combination a b c) ↔ (15 * 15 * 6 = 1350) :=
by
  sorry

end lark_lock_combination_count_l187_187079


namespace remaining_coins_denomination_l187_187110

def denomination_of_remaining_coins (total_coins : ℕ) (total_value : ℕ) (paise_20_count : ℕ) (paise_20_value : ℕ) : ℕ :=
  let remaining_coins := total_coins - paise_20_count
  let remaining_value := total_value - paise_20_count * paise_20_value
  remaining_value / remaining_coins

theorem remaining_coins_denomination :
  denomination_of_remaining_coins 334 7100 250 20 = 25 :=
by
  sorry

end remaining_coins_denomination_l187_187110


namespace jimmy_more_sheets_than_tommy_l187_187579

-- Definitions for the conditions
def initial_jimmy_sheets : ℕ := 58
def initial_tommy_sheets : ℕ := initial_jimmy_sheets + 25
def ashton_gives_jimmy : ℕ := 85
def jessica_gives_jimmy : ℕ := 47
def cousin_gives_tommy : ℕ := 30
def aunt_gives_tommy : ℕ := 19

-- Lean 4 statement for the proof problem
theorem jimmy_more_sheets_than_tommy :
  let final_jimmy_sheets := initial_jimmy_sheets + ashton_gives_jimmy + jessica_gives_jimmy;
  let final_tommy_sheets := initial_tommy_sheets + cousin_gives_tommy + aunt_gives_tommy;
  final_jimmy_sheets - final_tommy_sheets = 58 :=
by sorry

end jimmy_more_sheets_than_tommy_l187_187579


namespace imaginary_unit_cubic_l187_187204

def imaginary_unit_property (i : ℂ) : Prop :=
  i^2 = -1

theorem imaginary_unit_cubic (i : ℂ) (h : imaginary_unit_property i) : 1 + i^3 = 1 - i :=
  sorry

end imaginary_unit_cubic_l187_187204


namespace eq_of_divisibility_l187_187186

theorem eq_of_divisibility (a b : ℕ) (h : (a^2 + b^2) ∣ (a * b)) : a = b :=
  sorry

end eq_of_divisibility_l187_187186


namespace total_distance_both_l187_187716

-- Define conditions
def speed_onur : ℝ := 35  -- km/h
def speed_hanil : ℝ := 45  -- km/h
def daily_hours_onur : ℝ := 7
def additional_distance_hanil : ℝ := 40
def days_in_week : ℕ := 7

-- Define the daily biking distance for Onur and Hanil
def distance_onur_daily : ℝ := speed_onur * daily_hours_onur
def distance_hanil_daily : ℝ := distance_onur_daily + additional_distance_hanil

-- Define the number of days Onur and Hanil bike in a week
def working_days_onur : ℕ := 5
def working_days_hanil : ℕ := 6

-- Define the total distance covered by Onur and Hanil in a week
def total_distance_onur_week : ℝ := distance_onur_daily * working_days_onur
def total_distance_hanil_week : ℝ := distance_hanil_daily * working_days_hanil

-- Proof statement
theorem total_distance_both : total_distance_onur_week + total_distance_hanil_week = 2935 := by
  sorry

end total_distance_both_l187_187716


namespace mario_total_flowers_l187_187040

def hibiscus_flower_count (n : ℕ) : ℕ :=
  let h1 := 2 + 3 * n
  let h2 := (2 * 2) + 4 * n
  let h3 := (4 * (2 * 2)) + 5 * n
  h1 + h2 + h3

def rose_flower_count (n : ℕ) : ℕ :=
  let r1 := 3 + 2 * n
  let r2 := 5 + 3 * n
  r1 + r2

def sunflower_flower_count (n : ℕ) : ℕ :=
  6 * 2^n

def total_flower_count (n : ℕ) : ℕ :=
  hibiscus_flower_count n + rose_flower_count n + sunflower_flower_count n

theorem mario_total_flowers :
  total_flower_count 2 = 88 :=
by
  unfold total_flower_count hibiscus_flower_count rose_flower_count sunflower_flower_count
  norm_num

end mario_total_flowers_l187_187040


namespace volume_of_box_l187_187252

theorem volume_of_box (l w h : ℝ) (h1 : l * w = 24) (h2 : w * h = 16) (h3 : l * h = 6) :
  l * w * h = 48 :=
by
  sorry

end volume_of_box_l187_187252


namespace total_number_of_possible_outcomes_l187_187331

-- Define the conditions
def num_faces_per_die : ℕ := 6
def num_dice : ℕ := 2

-- Define the question as a hypothesis and the answer as the conclusion
theorem total_number_of_possible_outcomes :
  (num_faces_per_die * num_faces_per_die) = 36 := 
by
  -- Provide a proof outline, this is used to skip the actual proof
  sorry

end total_number_of_possible_outcomes_l187_187331


namespace find_tan_half_sum_of_angles_l187_187933

theorem find_tan_half_sum_of_angles (x y : ℝ) 
  (h₁ : Real.cos x + Real.cos y = 1)
  (h₂ : Real.sin x + Real.sin y = 1 / 2) : 
  Real.tan ((x + y) / 2) = 1 / 2 := 
by 
  sorry

end find_tan_half_sum_of_angles_l187_187933


namespace soccer_team_total_games_l187_187796

variable (total_games : ℕ)
variable (won_games : ℕ)

-- Given conditions
def team_won_percentage (p : ℝ) := p = 0.60
def team_won_games (w : ℕ) := w = 78

-- The proof goal
theorem soccer_team_total_games 
    (h1 : team_won_percentage 0.60)
    (h2 : team_won_games 78) :
    total_games = 130 :=
sorry

end soccer_team_total_games_l187_187796


namespace common_difference_d_l187_187322

open Real

-- Define the arithmetic sequence and relevant conditions
variable (a : ℕ → ℝ) -- Define the sequence as a function from natural numbers to real numbers
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific conditions from our problem
def problem_conditions (a : ℕ → ℝ) (d : ℝ) : Prop :=
  is_arithmetic_sequence a d ∧
  a 1 = 1 ∧
  (a 2) ^ 2 = a 1 * a 6

-- The goal is to prove that the common difference d is either 0 or 3
theorem common_difference_d (a : ℕ → ℝ) (d : ℝ) :
  problem_conditions a d → (d = 0 ∨ d = 3) := by
  sorry

end common_difference_d_l187_187322


namespace positive_difference_of_squares_l187_187690

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 8) : a^2 - b^2 = 320 :=
by
  sorry

end positive_difference_of_squares_l187_187690


namespace rearrangement_inequality_l187_187482

theorem rearrangement_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + c) + b / (a + c) + c / (a + b)) ≥ (3 / 2) ∧ (a = b ∧ b = c ∧ c = a ↔ (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2)) :=
by 
  -- Proof omitted
  sorry

end rearrangement_inequality_l187_187482


namespace andy_more_candies_than_caleb_l187_187167

theorem andy_more_candies_than_caleb :
  let billy_initial := 6
  let caleb_initial := 11
  let andy_initial := 9
  let father_packet := 36
  let billy_additional := 8
  let caleb_additional := 11
  let billy_total := billy_initial + billy_additional
  let caleb_total := caleb_initial + caleb_additional
  let total_given := billy_additional + caleb_additional
  let andy_additional := father_packet - total_given
  let andy_total := andy_initial + andy_additional
  andy_total - caleb_total = 4 :=
by {
  sorry
}

end andy_more_candies_than_caleb_l187_187167


namespace find_n_from_A_k_l187_187574

theorem find_n_from_A_k (n : ℕ) (A : ℕ → ℕ) (h1 : A 1 = Int.natAbs (n + 1))
  (h2 : ∀ k : ℕ, k > 0 → A k = Int.natAbs (n + (2 * k - 1)))
  (h3 : A 100 = 2005) : n = 1806 :=
sorry

end find_n_from_A_k_l187_187574


namespace f_neg_m_l187_187247

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

-- State the problem as a theorem
theorem f_neg_m (a b m : ℝ) (h : f a b m = 6) : f a b (-m) = -4 :=
by
  -- Proof is not required
  sorry

end f_neg_m_l187_187247


namespace sum_of_x_y_z_l187_187613

theorem sum_of_x_y_z (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 2 * y) : x + y + z = 10 * x := by
  sorry

end sum_of_x_y_z_l187_187613


namespace sequence_contains_perfect_square_l187_187304

noncomputable def f (n : ℕ) : ℕ := n + Nat.floor (Real.sqrt n)

theorem sequence_contains_perfect_square (m : ℕ) : ∃ k : ℕ, ∃ p : ℕ, f^[k] m = p * p := by
  sorry

end sequence_contains_perfect_square_l187_187304


namespace painters_complete_three_rooms_in_three_hours_l187_187900

theorem painters_complete_three_rooms_in_three_hours :
  ∃ P, (∀ (P : ℕ), (P * 3) = 3) ∧ (9 * 9 = 27) → P = 3 := by
  sorry

end painters_complete_three_rooms_in_three_hours_l187_187900


namespace friends_raise_funds_l187_187820

theorem friends_raise_funds (total_amount friends_count min_amount amount_per_person: ℕ)
  (h1 : total_amount = 3000)
  (h2 : friends_count = 10)
  (h3 : min_amount = 300)
  (h4 : amount_per_person = total_amount / friends_count) :
  amount_per_person = min_amount :=
by
  sorry

end friends_raise_funds_l187_187820


namespace collinear_points_d_value_l187_187030

theorem collinear_points_d_value (a b c d : ℚ)
  (h1 : b = a)
  (h2 : c = -(a+1)/2)
  (collinear : (4 * d * (4 * a + 5) + a + 1 = 0)) :
  d = 9/20 :=
by {
  sorry
}

end collinear_points_d_value_l187_187030


namespace greatest_integer_func_l187_187456

noncomputable def pi_approx : ℝ := 3.14159

theorem greatest_integer_func : (⌊2 * pi_approx - 6⌋ : ℝ) = 0 := 
by
  sorry

end greatest_integer_func_l187_187456


namespace quadratic_other_root_is_three_l187_187210

-- Steps for creating the Lean statement following the identified conditions
variable (b : ℝ)

theorem quadratic_other_root_is_three (h1 : ∀ x : ℝ, x^2 - 2 * x - b = 0 → (x = -1 ∨ x = 3)) : 
  ∀ x : ℝ, x^2 - 2 * x - b = 0 → x = -1 ∨ x = 3 :=
by
  -- The proof is omitted
  exact h1

end quadratic_other_root_is_three_l187_187210


namespace probability_green_or_blue_l187_187263

-- Define the properties of the 10-sided die
def total_faces : ℕ := 10
def red_faces : ℕ := 4
def yellow_faces : ℕ := 3
def green_faces : ℕ := 2
def blue_faces : ℕ := 1

-- Define the number of favorable outcomes
def favorable_outcomes : ℕ := green_faces + blue_faces

-- Define the probability function
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- The theorem to prove
theorem probability_green_or_blue :
  probability favorable_outcomes total_faces = 3 / 10 :=
by
  sorry

end probability_green_or_blue_l187_187263


namespace solve_system_of_equations_l187_187677

theorem solve_system_of_equations (x y z : ℝ) :
  x + y + z = 1 ∧ x^3 + y^3 + z^3 = 1 ∧ xyz = -16 ↔ 
  (x = 1 ∧ y = 4 ∧ z = -4) ∨ (x = 1 ∧ y = -4 ∧ z = 4) ∨ 
  (x = 4 ∧ y = 1 ∧ z = -4) ∨ (x = 4 ∧ y = -4 ∧ z = 1) ∨ 
  (x = -4 ∧ y = 1 ∧ z = 4) ∨ (x = -4 ∧ y = 4 ∧ z = 1) := 
by
  sorry

end solve_system_of_equations_l187_187677


namespace perfectSquareLastFourDigits_l187_187425

noncomputable def lastThreeDigitsForm (n : ℕ) : Prop :=
  ∃ a : ℕ, a ≤ 9 ∧ n % 1000 = a * 111

noncomputable def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem perfectSquareLastFourDigits (n : ℕ) :
  lastThreeDigitsForm n →
  isPerfectSquare n →
  (n % 10000 = 0 ∨ n % 10000 = 1444) :=
by {
  sorry
}

end perfectSquareLastFourDigits_l187_187425


namespace polar_eq_to_cartesian_l187_187577

-- Define the conditions
def polar_to_cartesian_eq (ρ : ℝ) : Prop :=
  ρ = 2 → (∃ x y : ℝ, x^2 + y^2 = ρ^2)

-- State the main theorem/proof problem
theorem polar_eq_to_cartesian : polar_to_cartesian_eq 2 :=
by
  -- Proof sketch:
  --   Given ρ = 2
  --   We have ρ^2 = 4
  --   By converting to Cartesian coordinates: x^2 + y^2 = ρ^2
  --   Result: x^2 + y^2 = 4
  sorry

end polar_eq_to_cartesian_l187_187577


namespace tic_tac_toe_winning_boards_l187_187397

-- Define the board as a 4x4 grid
def Board := Array (Array (Option Bool))

-- Define a function that returns all possible board states after 3 moves
noncomputable def numberOfWinningBoards : Nat := 140

theorem tic_tac_toe_winning_boards:
  numberOfWinningBoards = 140 :=
by
  sorry

end tic_tac_toe_winning_boards_l187_187397


namespace sequence_initial_value_l187_187038

theorem sequence_initial_value (a : ℕ → ℚ) 
  (h : ∀ n : ℕ, a (n + 1)^2 - a (n + 1) = a n) : a 1 = 0 ∨ a 1 = 2 :=
sorry

end sequence_initial_value_l187_187038


namespace age_twice_in_years_l187_187267

theorem age_twice_in_years (x : ℕ) : (40 + x = 2 * (12 + x)) → x = 16 :=
by {
  sorry
}

end age_twice_in_years_l187_187267


namespace negation_of_proposition_l187_187086

theorem negation_of_proposition
  (h : ∀ x : ℝ, x^2 - 2 * x + 2 > 0) :
  ∃ x : ℝ, x^2 - 2 * x + 2 ≤ 0 :=
sorry

end negation_of_proposition_l187_187086


namespace distance_from_O_is_450_l187_187127

noncomputable def find_distance_d (A B C P Q O : Type) 
  (side_length : ℝ) (PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ : ℝ) : ℝ :=
    if h : PA = PB ∧ PB = PC ∧ QA = QB ∧ QB = QC ∧ side_length = 600 ∧
           dist_OA = dist_OB ∧ dist_OB = dist_OC ∧ dist_OC = dist_OP ∧ dist_OP = dist_OQ ∧
           -- condition of 120 degree dihedral angle translates to specific geometric constraints
           true -- placeholder for the actual geometrical configuration that proves the problem
    then 450
    else 0 -- default or indication of inconsistency in conditions

-- Assuming all conditions hold true
theorem distance_from_O_is_450 (A B C P Q O : Type) 
  (side_length : ℝ) (PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ : ℝ)
  (h : PA = PB ∧ PB = PC ∧ QA = QB ∧ QB = QC ∧ side_length = 600 ∧
       dist_OA = dist_OB ∧ dist_OB = dist_OC ∧ dist_OC = dist_OP ∧ dist_OP = dist_OQ ∧
       -- adding condition of 120 degree dihedral angle
       true) -- true is a placeholder, the required proof to be filled in
  : find_distance_d A B C P Q O side_length PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ = 450 :=
by
  -- proof goes here
  sorry

end distance_from_O_is_450_l187_187127


namespace wine_remaining_percentage_l187_187173

theorem wine_remaining_percentage :
  let initial_wine := 250.0 -- initial wine in liters
  let daily_fraction := (249.0 / 250.0)
  let days := 50
  let remaining_wine := (daily_fraction ^ days) * initial_wine
  let percentage_remaining := (remaining_wine / initial_wine) * 100
  percentage_remaining = 81.846 :=
by
  sorry

end wine_remaining_percentage_l187_187173


namespace exam_room_selection_l187_187229

theorem exam_room_selection (rooms : List ℕ) (n : ℕ) 
    (fifth_room_selected : 5 ∈ rooms) (twentyfirst_room_selected : 21 ∈ rooms) :
    rooms = [5, 13, 21, 29, 37, 45, 53, 61] → 
    37 ∈ rooms ∧ 53 ∈ rooms :=
by
  sorry

end exam_room_selection_l187_187229


namespace regular_polygon_sides_l187_187746

-- Define the number of sides
def n : ℕ := sorry

-- The interior angle condition
def interior_angle_condition (n : ℕ) : Prop := 
  let interior_angle := (180 * (n - 2)) / n
  interior_angle = 160

-- Prove the statement
theorem regular_polygon_sides (h : interior_angle_condition n) : n = 18 := 
  sorry

end regular_polygon_sides_l187_187746


namespace divides_if_not_divisible_by_4_l187_187401

theorem divides_if_not_divisible_by_4 (n : ℕ) :
  (¬ (4 ∣ n)) → (5 ∣ (1^n + 2^n + 3^n + 4^n)) :=
by sorry

end divides_if_not_divisible_by_4_l187_187401


namespace rubber_duck_charity_fundraiser_l187_187400

noncomputable def charity_raised (price_small price_medium price_large : ℕ) 
(bulk_discount_threshold_small bulk_discount_threshold_medium bulk_discount_threshold_large : ℕ)
(bulk_discount_rate_small bulk_discount_rate_medium bulk_discount_rate_large : ℝ)
(tax_rate_small tax_rate_medium tax_rate_large : ℝ)
(sold_small sold_medium sold_large : ℕ) : ℝ :=
  let cost_small := price_small * sold_small
  let cost_medium := price_medium * sold_medium
  let cost_large := price_large * sold_large

  let discount_small := if sold_small >= bulk_discount_threshold_small then 
                          (bulk_discount_rate_small * cost_small) else 0
  let discount_medium := if sold_medium >= bulk_discount_threshold_medium then 
                          (bulk_discount_rate_medium * cost_medium) else 0
  let discount_large := if sold_large >= bulk_discount_threshold_large then 
                          (bulk_discount_rate_large * cost_large) else 0

  let after_discount_small := cost_small - discount_small
  let after_discount_medium := cost_medium - discount_medium
  let after_discount_large := cost_large - discount_large

  let tax_small := tax_rate_small * after_discount_small
  let tax_medium := tax_rate_medium * after_discount_medium
  let tax_large := tax_rate_large * after_discount_large

  let total_small := after_discount_small + tax_small
  let total_medium := after_discount_medium + tax_medium
  let total_large := after_discount_large + tax_large

  total_small + total_medium + total_large

theorem rubber_duck_charity_fundraiser :
  charity_raised 2 3 5 10 15 20 0.1 0.15 0.2
  0.05 0.07 0.09 150 221 185 = 1693.10 :=
by 
  -- implementation of math corresponding to problem's solution
  sorry

end rubber_duck_charity_fundraiser_l187_187400


namespace minimum_disks_needed_l187_187544

theorem minimum_disks_needed :
  ∀ (n_files : ℕ) (disk_space : ℝ) (mb_files_1 : ℕ) (size_file_1 : ℝ) (mb_files_2 : ℕ) (size_file_2 : ℝ) (remaining_files : ℕ) (size_remaining_files : ℝ),
    n_files = 30 →
    disk_space = 1.5 →
    mb_files_1 = 4 →
    size_file_1 = 1.0 →
    mb_files_2 = 10 →
    size_file_2 = 0.6 →
    remaining_files = 16 →
    size_remaining_files = 0.5 →
    ∃ (min_disks : ℕ), min_disks = 13 :=
by
  sorry

end minimum_disks_needed_l187_187544


namespace emma_age_when_sister_is_56_l187_187234

theorem emma_age_when_sister_is_56 (e s : ℕ) (he : e = 7) (hs : s = e + 9) : 
  (s + (56 - s) - 9 = 47) :=
by {
  sorry
}

end emma_age_when_sister_is_56_l187_187234


namespace certain_number_divisibility_l187_187614

theorem certain_number_divisibility {n : ℕ} (h : ∃ count : ℕ, count = 50 ∧ (count = (300 / (2 * n)))) : n = 3 :=
by
  sorry

end certain_number_divisibility_l187_187614


namespace prove_expression_value_l187_187557

-- Define the conditions
variables {a b c d m : ℤ}
variable (h1 : a + b = 0)
variable (h2 : |m| = 2)
variable (h3 : c * d = 1)

-- State the theorem
theorem prove_expression_value : (a + b) / (4 * m) + 2 * m ^ 2 - 3 * c * d = 5 :=
by
  -- Proof goes here
  sorry

end prove_expression_value_l187_187557


namespace price_of_other_frisbees_l187_187648

-- Lean 4 Statement
theorem price_of_other_frisbees (P : ℝ) (x : ℕ) (h1 : x ≥ 40) (h2 : P * x + 4 * (60 - x) = 200) :
  P = 3 := 
  sorry

end price_of_other_frisbees_l187_187648


namespace find_value_l187_187909

theorem find_value (N : ℝ) (h : 1.20 * N = 6000) : 0.20 * N = 1000 :=
sorry

end find_value_l187_187909


namespace ethanol_solution_exists_l187_187581

noncomputable def ethanol_problem : Prop :=
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 204 ∧ 0.12 * x + 0.16 * (204 - x) = 30

theorem ethanol_solution_exists : ethanol_problem :=
sorry

end ethanol_solution_exists_l187_187581


namespace part1_part2_l187_187915

variable (α : ℝ)

theorem part1 (h : Real.tan α = 2) : (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 := 
by
  sorry

theorem part2 (h : Real.tan α = 2) : Real.sin α ^ 2 + Real.sin (2 * α) = 8 / 5 :=
by
  sorry

end part1_part2_l187_187915


namespace op_assoc_l187_187832

open Real

def op (x y : ℝ) : ℝ := x + y - x * y

theorem op_assoc (x y z : ℝ) : op (op x y) z = op x (op y z) := by
  sorry

end op_assoc_l187_187832


namespace games_bought_from_friend_is_21_l187_187762

-- Definitions from the conditions
def games_bought_at_garage_sale : ℕ := 8
def non_working_games : ℕ := 23
def good_games : ℕ := 6

-- The total number of games John has is the sum of good and non-working games
def total_games : ℕ := good_games + non_working_games

-- The number of games John bought from his friend
def games_from_friend : ℕ := total_games - games_bought_at_garage_sale

-- Statement to prove
theorem games_bought_from_friend_is_21 : games_from_friend = 21 := by
  sorry

end games_bought_from_friend_is_21_l187_187762


namespace smallest_int_ending_in_9_divisible_by_11_l187_187651

theorem smallest_int_ending_in_9_divisible_by_11:
  ∃ x : ℕ, (∃ k : ℤ, x = 10 * k + 9) ∧ x % 11 = 0 ∧ x = 99 :=
by
  sorry

end smallest_int_ending_in_9_divisible_by_11_l187_187651


namespace blocks_used_l187_187438

theorem blocks_used (initial_blocks used_blocks : ℕ) (h_initial : initial_blocks = 78) (h_left : initial_blocks - used_blocks = 59) : used_blocks = 19 := by
  sorry

end blocks_used_l187_187438


namespace acquaintances_unique_l187_187413

theorem acquaintances_unique (N : ℕ) : ∃ acquaintances : ℕ → ℕ, 
  (∀ i j k : ℕ, i < N → j < N → k < N → i ≠ j → j ≠ k → i ≠ k → 
    acquaintances i ≠ acquaintances j ∨ acquaintances j ≠ acquaintances k ∨ acquaintances i ≠ acquaintances k) :=
sorry

end acquaintances_unique_l187_187413


namespace josh_total_payment_with_tax_and_discount_l187_187181

-- Definitions
def total_string_cheeses (pack1 : ℕ) (pack2 : ℕ) (pack3 : ℕ) : ℕ :=
  pack1 + pack2 + pack3

def total_cost_before_tax_and_discount (n : ℕ) (cost_per_cheese : ℚ) : ℚ :=
  n * cost_per_cheese

def discount_amount (cost : ℚ) (discount_rate : ℚ) : ℚ :=
  cost * discount_rate

def discounted_cost (cost : ℚ) (discount : ℚ) : ℚ :=
  cost - discount

def sales_tax_amount (cost : ℚ) (tax_rate : ℚ) : ℚ :=
  cost * tax_rate

def total_cost (cost : ℚ) (tax : ℚ) : ℚ :=
  cost + tax

-- The statement
theorem josh_total_payment_with_tax_and_discount :
  let cost_per_cheese := 0.10
  let discount_rate := 0.05
  let tax_rate := 0.12
  total_cost (discounted_cost (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese)
                              (discount_amount (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese) discount_rate))
             (sales_tax_amount (discounted_cost (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese)
                                               (discount_amount (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese) discount_rate)) tax_rate) = 6.81 := 
  sorry

end josh_total_payment_with_tax_and_discount_l187_187181


namespace investment_time_l187_187044

theorem investment_time (P R diff : ℝ) (T : ℕ) 
  (hP : P = 1500)
  (hR : R = 0.10)
  (hdiff : diff = 15)
  (h1 : P * ((1 + R) ^ T - 1) - (P * R * T) = diff) 
  : T = 2 := 
by
  -- proof steps here
  sorry

end investment_time_l187_187044


namespace Martha_knitting_grandchildren_l187_187249

theorem Martha_knitting_grandchildren (T_hat T_scarf T_mittens T_socks T_sweater T_total : ℕ)
  (h_hat : T_hat = 2) (h_scarf : T_scarf = 3) (h_mittens : T_mittens = 2)
  (h_socks : T_socks = 3) (h_sweater : T_sweater = 6) (h_total : T_total = 48) :
  (T_total / (T_hat + T_scarf + T_mittens + T_socks + T_sweater)) = 3 := by
  sorry

end Martha_knitting_grandchildren_l187_187249


namespace find_initial_candies_l187_187326

-- Define the initial number of candies as x
def initial_candies (x : ℕ) : ℕ :=
  let first_day := (3 * x) / 4 - 3
  let second_day := (3 * first_day) / 5 - 5
  let third_day := second_day - 7
  let final_candies := (5 * third_day) / 6
  final_candies

-- Formal statement of the theorem
theorem find_initial_candies (x : ℕ) (h : initial_candies x = 10) : x = 44 :=
  sorry

end find_initial_candies_l187_187326


namespace girls_attending_event_l187_187597

theorem girls_attending_event (g b : ℕ) 
  (h1 : g + b = 1500)
  (h2 : 3 / 4 * g + 2 / 5 * b = 900) :
  3 / 4 * g = 643 := 
by
  sorry

end girls_attending_event_l187_187597


namespace eel_cost_l187_187754

theorem eel_cost (J E : ℝ) (h1 : E = 9 * J) (h2 : J + E = 200) : E = 180 :=
by
  sorry

end eel_cost_l187_187754


namespace integer_values_count_l187_187643

theorem integer_values_count (x : ℤ) :
  ∃ k, (∀ n : ℤ, (3 ≤ Real.sqrt (3 * n + 1) ∧ Real.sqrt (3 * n + 1) < 5) ↔ ((n = 3) ∨ (n = 4) ∨ (n = 5) ∨ (n = 6) ∨ (n = 7)) ∧ k = 5) :=
by
  sorry

end integer_values_count_l187_187643


namespace sequence_general_term_l187_187802

theorem sequence_general_term (a : ℕ → ℤ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = (2 ^ n) - 1 := 
sorry

end sequence_general_term_l187_187802


namespace smallest_base10_integer_l187_187956

theorem smallest_base10_integer :
  ∃ (n : ℕ) (X : ℕ) (Y : ℕ), 
  (0 ≤ X ∧ X < 6) ∧ (0 ≤ Y ∧ Y < 8) ∧ 
  (n = 7 * X) ∧ (n = 9 * Y) ∧ n = 63 :=
by
  sorry

end smallest_base10_integer_l187_187956


namespace Jennifer_apples_l187_187457

-- Define the conditions
def initial_apples : ℕ := 7
def found_apples : ℕ := 74

-- The theorem to prove
theorem Jennifer_apples : initial_apples + found_apples = 81 :=
by
  -- proof goes here, but we use sorry to skip the proof step
  sorry

end Jennifer_apples_l187_187457


namespace sum_of_first_30_terms_l187_187494

variable (a : Nat → ℤ)
variable (d : ℤ)
variable (S_30 : ℤ)

-- Conditions from part a)
def condition1 := a 1 + a 2 + a 3 = 3
def condition2 := a 28 + a 29 + a 30 = 165

-- Question translated to Lean 4 statement
theorem sum_of_first_30_terms 
  (h1 : condition1 a)
  (h2 : condition2 a) :
  S_30 = 840 := 
sorry

end sum_of_first_30_terms_l187_187494


namespace correct_calculation_l187_187345

variable (a : ℝ)

theorem correct_calculation (a : ℝ) : (2 * a)^2 / (4 * a) = a := by
  sorry

end correct_calculation_l187_187345


namespace map_distance_l187_187747

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l187_187747


namespace total_animals_in_savanna_l187_187396

/-- Define the number of lions, snakes, and giraffes in Safari National Park. --/
def safari_lions : ℕ := 100
def safari_snakes : ℕ := safari_lions / 2
def safari_giraffes : ℕ := safari_snakes - 10

/-- Define the number of lions, snakes, and giraffes in Savanna National Park based on conditions. --/
def savanna_lions : ℕ := 2 * safari_lions
def savanna_snakes : ℕ := 3 * safari_snakes
def savanna_giraffes : ℕ := safari_giraffes + 20

/-- Calculate the total number of animals in Savanna National Park. --/
def total_savanna_animals : ℕ := savanna_lions + savanna_snakes + savanna_giraffes

/-- Proof statement that the total number of animals in Savanna National Park is 410.
My goal is to prove that total_savanna_animals is equal to 410. --/
theorem total_animals_in_savanna : total_savanna_animals = 410 :=
by
  sorry

end total_animals_in_savanna_l187_187396


namespace apples_difference_l187_187453

def jimin_apples : ℕ := 7
def grandpa_apples : ℕ := 13
def younger_brother_apples : ℕ := 8
def younger_sister_apples : ℕ := 5

theorem apples_difference :
  grandpa_apples - younger_sister_apples = 8 :=
by
  sorry

end apples_difference_l187_187453


namespace cost_to_paint_cube_l187_187594

theorem cost_to_paint_cube (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (side_length : ℝ) (total_cost : ℝ) :
  cost_per_kg = 36.50 →
  coverage_per_kg = 16 →
  side_length = 8 →
  total_cost = (6 * side_length^2 / coverage_per_kg) * cost_per_kg →
  total_cost = 876 :=
by
  intros h1 h2 h3 h4
  sorry

end cost_to_paint_cube_l187_187594


namespace compound_interest_l187_187251

noncomputable def final_amount (P : ℕ) (r : ℚ) (t : ℕ) :=
  P * ((1 : ℚ) + r) ^ t

theorem compound_interest : 
  final_amount 20000 0.20 10 = 123834.73 := 
by 
  sorry

end compound_interest_l187_187251


namespace area_proof_l187_187554

-- Define the problem conditions
variables (l w : ℕ)
def length_is_four_times_width : Prop := l = 4 * w
def perimeter_is_200 : Prop := 2 * l + 2 * w = 200

-- Define the target to prove
def area_of_rectangle : Prop := (l * w = 1600)


-- Lean 4 statement to prove the area given the conditions
theorem area_proof (h1 : length_is_four_times_width l w) (h2 : perimeter_is_200 l w) : area_of_rectangle l w := 
  sorry

end area_proof_l187_187554


namespace shelves_needed_number_of_shelves_l187_187632

-- Define the initial number of books
def initial_books : Float := 46.0

-- Define the number of additional books added by the librarian
def additional_books : Float := 10.0

-- Define the number of books each shelf can hold
def books_per_shelf : Float := 4.0

-- Define the total number of books
def total_books : Float := initial_books + additional_books

-- The mathematical proof statement for the number of shelves needed
theorem shelves_needed : Float := total_books / books_per_shelf

-- The required statement proving that the number of shelves needed is 14.0
theorem number_of_shelves : shelves_needed = 14.0 := by
  sorry

end shelves_needed_number_of_shelves_l187_187632


namespace running_speed_l187_187140

theorem running_speed
  (walking_speed : Float)
  (walking_time : Float)
  (running_time : Float)
  (distance : Float) :
  walking_speed = 8 → walking_time = 3 → running_time = 1.5 → distance = walking_speed * walking_time → 
  (distance / running_time) = 16 :=
by
  intros h_walking_speed h_walking_time h_running_time h_distance
  sorry

end running_speed_l187_187140


namespace min_value_sin6_cos6_l187_187208

open Real

theorem min_value_sin6_cos6 (x : ℝ) : sin x ^ 6 + 2 * cos x ^ 6 ≥ 2 / 3 :=
by
  sorry

end min_value_sin6_cos6_l187_187208


namespace parallelogram_area_l187_187565

theorem parallelogram_area (base height : ℝ) (h_base : base = 36) (h_height : height = 24) : 
    base * height = 864 :=
by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l187_187565


namespace simplify_expression_l187_187585

theorem simplify_expression : 
  3 * Real.sqrt 48 - 9 * Real.sqrt (1 / 3) - Real.sqrt 3 * (2 - Real.sqrt 27) = 7 * Real.sqrt 3 + 9 :=
by
  -- The proof is omitted as per the instructions
  sorry

end simplify_expression_l187_187585


namespace parallelogram_area_288_l187_187534

/-- A statement of the area of a given parallelogram -/
theorem parallelogram_area_288 
  (AB BC : ℝ)
  (hAB : AB = 24)
  (hBC : BC = 30)
  (height_from_A_to_DC : ℝ)
  (h_height : height_from_A_to_DC = 12)
  (is_parallelogram : true) :
  AB * height_from_A_to_DC = 288 :=
by
  -- We are focusing only on stating the theorem; the proof is not required.
  sorry

end parallelogram_area_288_l187_187534


namespace unique_y_for_diamond_l187_187801

def diamond (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y + 1

theorem unique_y_for_diamond :
  ∃! y : ℝ, diamond 4 y = 21 :=
by
  sorry

end unique_y_for_diamond_l187_187801


namespace overall_average_length_of_ropes_l187_187065

theorem overall_average_length_of_ropes :
  let ropes := 6
  let third_part := ropes / 3
  let average1 := 70
  let average2 := 85
  let length1 := third_part * average1
  let length2 := (ropes - third_part) * average2
  let total_length := length1 + length2
  let overall_average := total_length / ropes
  overall_average = 80 := by
sorry

end overall_average_length_of_ropes_l187_187065


namespace largest_t_value_maximum_t_value_l187_187085

noncomputable def largest_t : ℚ :=
  (5 : ℚ) / 2

theorem largest_t_value (t : ℚ) :
  (13 * t^2 - 34 * t + 12) / (3 * t - 2) + 5 * t = 6 * t - 1 →
  t ≤ (5 : ℚ) / 2 :=
sorry

theorem maximum_t_value (t : ℚ) :
  (13 * t^2 - 34 * t + 12) / (3 * t - 2) + 5 * t = 6 * t - 1 →
  (5 : ℚ) / 2 = largest_t :=
sorry

end largest_t_value_maximum_t_value_l187_187085


namespace unique_pair_natural_numbers_l187_187528

theorem unique_pair_natural_numbers (a b : ℕ) :
  (∀ n : ℕ, ∃ c : ℕ, a ^ n + b ^ n = c ^ (n + 1)) → (a = 2 ∧ b = 2) :=
by
  sorry

end unique_pair_natural_numbers_l187_187528


namespace smallest_next_divisor_l187_187200

theorem smallest_next_divisor (m : ℕ) (h_digit : 10000 ≤ m ∧ m < 100000) (h_odd : m % 2 = 1) (h_div : 437 ∣ m) :
  ∃ d : ℕ, 437 < d ∧ d ∣ m ∧ (∀ e : ℕ, 437 < e ∧ e < d → ¬ e ∣ m) ∧ d = 475 := 
sorry

end smallest_next_divisor_l187_187200


namespace rug_length_l187_187493

theorem rug_length (d : ℕ) (x y : ℕ) (h1 : x * x + y * y = d * d) (h2 : y / x = 2) (h3 : (x = 25 ∧ y = 50)) : 
  x = 25 := 
sorry

end rug_length_l187_187493


namespace solve_equation_l187_187274

theorem solve_equation : ∃ x : ℚ, (2*x + 1) / 4 - 1 = x - (10*x + 1) / 12 ∧ x = 5 / 2 :=
by
  sorry

end solve_equation_l187_187274


namespace no_solution_A_eq_B_l187_187618

theorem no_solution_A_eq_B (a : ℝ) (h1 : a = 2 * a) (h2 : a ≠ 2) : false := by
  sorry

end no_solution_A_eq_B_l187_187618


namespace range_of_a_for_false_proposition_l187_187337

theorem range_of_a_for_false_proposition :
  ∀ a : ℝ, (¬ ∃ x : ℝ, a * x ^ 2 + a * x + 1 ≤ 0) ↔ (0 ≤ a ∧ a < 4) :=
by sorry

end range_of_a_for_false_proposition_l187_187337


namespace least_possible_z_minus_x_l187_187930

theorem least_possible_z_minus_x (x y z : ℤ) (h₁ : x < y) (h₂ : y < z) (h₃ : y - x > 11) 
  (h₄ : Even x) (h₅ : Odd y) (h₆ : Odd z) : z - x = 15 :=
sorry

end least_possible_z_minus_x_l187_187930


namespace neighbor_packs_l187_187622

theorem neighbor_packs (n : ℕ) :
  let milly_balloons := 3 * 6 -- Milly and Floretta use 3 packs of their own
  let neighbor_balloons := n * 6 -- some packs of the neighbor's balloons, each contains 6 balloons
  let total_balloons := milly_balloons + neighbor_balloons -- total balloons
  -- They split balloons evenly; Milly takes 7 extra, then Floretta has 8 left
  total_balloons / 2 + 7 = total_balloons - 15
  → n = 2 := sorry

end neighbor_packs_l187_187622


namespace total_pencils_l187_187615

theorem total_pencils (initial_additional1 initial_additional2 : ℕ) (h₁ : initial_additional1 = 37) (h₂ : initial_additional2 = 17) : (initial_additional1 + initial_additional2) = 54 :=
by sorry

end total_pencils_l187_187615


namespace triple_nested_application_l187_187556

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2 * n + 3

theorem triple_nested_application : g (g (g 3)) = 49 := by
  sorry

end triple_nested_application_l187_187556


namespace value_of_d_l187_187012

theorem value_of_d (d y : ℤ) (h₁ : y = 2) (h₂ : 5 * y^2 - 8 * y + 55 = d) : d = 59 := by
  sorry

end value_of_d_l187_187012


namespace mangoes_combined_l187_187983

variable (Alexis Dilan Ashley : ℕ)

theorem mangoes_combined :
  (Alexis = 60) → (Alexis = 4 * (Dilan + Ashley)) → (Alexis + Dilan + Ashley = 75) := 
by
  intros h₁ h₂
  sorry

end mangoes_combined_l187_187983


namespace sugarCubeWeight_l187_187535

theorem sugarCubeWeight
  (ants1 : ℕ) (sugar_cubes1 : ℕ) (weight1 : ℕ) (hours1 : ℕ)
  (ants2 : ℕ) (sugar_cubes2 : ℕ) (hours2 : ℕ) :
  ants1 = 15 →
  sugar_cubes1 = 600 →
  weight1 = 10 →
  hours1 = 5 →
  ants2 = 20 →
  sugar_cubes2 = 960 →
  hours2 = 3 →
  ∃ weight2 : ℕ, weight2 = 5 := by
  sorry

end sugarCubeWeight_l187_187535


namespace perimeter_of_square_36_l187_187625

variable (a s P : ℕ)

def is_square_area : Prop := a = s * s
def is_square_perimeter : Prop := P = 4 * s
def condition : Prop := 5 * a = 10 * P + 45

theorem perimeter_of_square_36 (h1 : is_square_area a s) (h2 : is_square_perimeter P s) (h3 : condition a P) : P = 36 := 
by
  sorry

end perimeter_of_square_36_l187_187625


namespace tan_of_sine_plus_cosine_eq_neg_4_over_3_l187_187882

variable {A : ℝ}

theorem tan_of_sine_plus_cosine_eq_neg_4_over_3 
  (h : Real.sin A + Real.cos A = -4/3) : 
  Real.tan A = -4/3 :=
sorry

end tan_of_sine_plus_cosine_eq_neg_4_over_3_l187_187882


namespace odd_integers_equality_l187_187729

-- Definitions
def is_odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

def divides (d n : ℤ) := ∃ k : ℤ, n = d * k

-- Main statement
theorem odd_integers_equality (a b : ℤ) (ha_pos : 0 < a) (hb_pos : 0 < b)
 (ha_odd : is_odd a) (hb_odd : is_odd b)
 (h_div : divides (2 * a * b + 1) (a^2 + b^2 + 1))
 : a = b :=
by 
  sorry

end odd_integers_equality_l187_187729


namespace largest_possible_P10_l187_187665

noncomputable def P (x : ℤ) : ℤ := x^2 + 3*x + 3

theorem largest_possible_P10 : P 10 = 133 := by
  sorry

end largest_possible_P10_l187_187665


namespace blake_total_expenditure_l187_187197

noncomputable def total_cost (rooms : ℕ) (primer_cost : ℝ) (paint_cost : ℝ) (primer_discount : ℝ) : ℝ :=
  let primer_needed := rooms
  let paint_needed := rooms
  let discounted_primer_cost := primer_cost * (1 - primer_discount)
  let total_primer_cost := primer_needed * discounted_primer_cost
  let total_paint_cost := paint_needed * paint_cost
  total_primer_cost + total_paint_cost

theorem blake_total_expenditure :
  total_cost 5 30 25 0.20 = 245 := 
by
  sorry

end blake_total_expenditure_l187_187197


namespace overall_winning_percentage_is_fifty_l187_187141

def winning_percentage_of_first_games := (40 / 100) * 30
def total_games_played := 40
def remaining_games := total_games_played - 30
def winning_percentage_of_remaining_games := (80 / 100) * remaining_games
def total_games_won := winning_percentage_of_first_games + winning_percentage_of_remaining_games

theorem overall_winning_percentage_is_fifty : 
  (total_games_won / total_games_played) * 100 = 50 := 
by
  sorry

end overall_winning_percentage_is_fifty_l187_187141


namespace squirrel_acorns_left_l187_187184

noncomputable def acorns_per_winter_month (total_acorns : ℕ) (months : ℕ) (acorns_taken_total : ℕ) : ℕ :=
  let per_month := total_acorns / months
  let acorns_taken_per_month := acorns_taken_total / months
  per_month - acorns_taken_per_month

theorem squirrel_acorns_left (total_acorns : ℕ) (months : ℕ) (acorns_taken_total : ℕ) :
  total_acorns = 210 → months = 3 → acorns_taken_total = 30 → acorns_per_winter_month total_acorns months acorns_taken_total = 60 :=
by intros; sorry

end squirrel_acorns_left_l187_187184


namespace qiqi_initial_batteries_qiqi_jiajia_difference_after_transfer_l187_187033

variable (m : Int)

theorem qiqi_initial_batteries (m : Int) : 
  let Qiqi_initial := 2 * m - 2
  Qiqi_initial = 2 * m - 2 := sorry

theorem qiqi_jiajia_difference_after_transfer (m : Int) : 
  let Qiqi_after := 2 * m - 2 - 2
  let Jiajia_after := m + 2
  Qiqi_after - Jiajia_after = m - 6 := sorry

end qiqi_initial_batteries_qiqi_jiajia_difference_after_transfer_l187_187033


namespace find_x_l187_187078

theorem find_x (x n q r : ℕ) (h_n : n = 220080) (h_sum : n = (x + 445) * (2 * (x - 445)) + r) (h_r : r = 80) : 
  x = 555 :=
by
  have eq1 : n = 220080 := h_n
  have eq2 : n =  (x + 445) * (2 * (x - 445)) + r := h_sum
  have eq3 : r = 80 := h_r
  sorry

end find_x_l187_187078


namespace no_solutions_to_equation_l187_187609

theorem no_solutions_to_equation (a b c : ℤ) : a^2 + b^2 - 8 * c ≠ 6 := 
by 
-- sorry to skip the proof part
sorry

end no_solutions_to_equation_l187_187609


namespace simplified_value_of_f_l187_187126

variable (x : ℝ)

noncomputable def f : ℝ := 3 * x + 5 - 4 * x^2 + 2 * x - 7 + x^2 - 3 * x + 8

theorem simplified_value_of_f : f x = -3 * x^2 + 2 * x + 6 := by
  unfold f
  sorry

end simplified_value_of_f_l187_187126


namespace automotive_test_l187_187817

noncomputable def total_distance (D : ℝ) (t : ℝ) : ℝ := 3 * D

theorem automotive_test (D : ℝ) (h_time : (D / 4 + D / 5 + D / 6 = 37)) : total_distance D 37 = 180 :=
  by
    -- This skips the proof, only the statement is given
    sorry

end automotive_test_l187_187817


namespace find_number_l187_187959

-- Define the variables and the conditions as theorems to be proven in Lean.
theorem find_number (x : ℤ) 
  (h1 : (x - 16) % 37 = 0)
  (h2 : (x - 16) / 37 = 23) :
  x = 867 :=
sorry

end find_number_l187_187959


namespace find_x_l187_187722

def star (p q : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 - q.1, p.2 + q.2)

theorem find_x (x y : ℤ) :
  star (3, 3) (0, 0) = star (x, y) (3, 2) → x = 6 :=
by
  intro h
  sorry

end find_x_l187_187722


namespace market_value_correct_l187_187595

noncomputable def market_value : ℝ :=
  let dividend_income (M : ℝ) := 0.12 * M
  let fees (M : ℝ) := 0.01 * M
  let taxes (M : ℝ) := 0.15 * dividend_income M
  have yield_after_fees_and_taxes : ∀ M, 0.08 * M = dividend_income M - fees M - taxes M := 
    by sorry
  86.96

theorem market_value_correct :
  market_value = 86.96 := 
by
  sorry

end market_value_correct_l187_187595


namespace necessary_but_not_sufficient_l187_187993

def M := {x : ℝ | 0 < x ∧ x ≤ 3}
def N := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient (a : ℝ) (haM : a ∈ M) : (a ∈ N → a ∈ M) ∧ ¬(a ∈ M → a ∈ N) :=
by {
  sorry
}

end necessary_but_not_sufficient_l187_187993


namespace range_of_m_for_inequality_l187_187458

-- Define the condition
def condition (x : ℝ) := x ∈ Set.Iic (-1)

-- Define the inequality for proving the range of m
def inequality_holds (m x : ℝ) : Prop := (m - m^2) * 4^x + 2^x + 1 > 0

-- Prove the range of m for the given conditions such that the inequality holds
theorem range_of_m_for_inequality :
  (∀ (x : ℝ), condition x → inequality_holds m x) ↔ (-2 < m ∧ m < 3) :=
sorry

end range_of_m_for_inequality_l187_187458


namespace connected_graphs_bound_l187_187273

noncomputable def num_connected_graphs (n : ℕ) : ℕ := sorry
  
theorem connected_graphs_bound (n : ℕ) : 
  num_connected_graphs n ≥ (1/2) * 2^(n*(n-1)/2) := 
sorry

end connected_graphs_bound_l187_187273


namespace problem_statement_l187_187855

theorem problem_statement (a : ℝ) (h : (a + 1/a)^2 = 12) : a^3 + 1/a^3 = 18 * Real.sqrt 3 :=
by
  -- We'll skip the proof as per instruction
  sorry

end problem_statement_l187_187855


namespace first_part_length_l187_187944

def total_length : ℝ := 74.5
def part_two : ℝ := 21.5
def part_three : ℝ := 21.5
def part_four : ℝ := 16

theorem first_part_length :
  total_length - (part_two + part_three + part_four) = 15.5 :=
by
  sorry

end first_part_length_l187_187944


namespace pyramid_volume_in_unit_cube_l187_187443

noncomputable def base_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

noncomputable def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ := (1 / 3) * base_area * height

theorem pyramid_volume_in_unit_cube : 
  let s := Real.sqrt 2 / 2
  let height := 1
  pyramid_volume (base_area s) height = Real.sqrt 3 / 24 :=
by
  sorry

end pyramid_volume_in_unit_cube_l187_187443


namespace remainder_of_product_l187_187262

open Nat

theorem remainder_of_product (a b : ℕ) (ha : a % 5 = 4) (hb : b % 5 = 3) :
  (a * b) % 5 = 2 :=
by
  sorry

end remainder_of_product_l187_187262


namespace mixtilinear_incircle_radius_l187_187920
open Real

variable (AB BC AC : ℝ)
variable (r_A : ℝ)

def triangle_conditions : Prop :=
  AB = 65 ∧ BC = 33 ∧ AC = 56

theorem mixtilinear_incircle_radius 
  (h : triangle_conditions AB BC AC)
  : r_A = 12.89 := 
sorry

end mixtilinear_incircle_radius_l187_187920


namespace fraction_to_decimal_l187_187090

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l187_187090


namespace ratio_of_heights_l187_187786

theorem ratio_of_heights (a b : ℝ) (area_ratio_is_9_4 : a / b = 9 / 4) :
  ∃ h₁ h₂ : ℝ, h₁ / h₂ = 3 / 2 :=
by
  sorry

end ratio_of_heights_l187_187786


namespace children_of_exceptions_l187_187676

theorem children_of_exceptions (x y : ℕ) (h : 6 * x + 2 * y = 58) (hx : x = 8) : y = 5 :=
by
  sorry

end children_of_exceptions_l187_187676


namespace range_of_m_l187_187339

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0)
  (h_equation : (2 / x) + (1 / y) = 1 / 3)
  (h_inequality : x + 2 * y > m^2 - 2 * m) : 
  -4 < m ∧ m < 6 := 
sorry

end range_of_m_l187_187339


namespace volume_of_remaining_solid_after_removing_tetrahedra_l187_187314

theorem volume_of_remaining_solid_after_removing_tetrahedra :
  let cube_volume := 1
  let tetrahedron_volume := (1/3) * (1/2) * (1/2) * (1/2) * (1/2)
  cube_volume - 8 * tetrahedron_volume = 5 / 6 := by
  let cube_volume := 1
  let tetrahedron_volume := (1/3) * (1/2) * (1/2) * (1/2) * (1/2)
  have h : cube_volume - 8 * tetrahedron_volume = 5 / 6 := sorry
  exact h

end volume_of_remaining_solid_after_removing_tetrahedra_l187_187314


namespace minimize_sum_areas_l187_187610

theorem minimize_sum_areas (x : ℝ) (h_wire_length : 0 < x ∧ x < 1) :
    let side_length := x / 4
    let square_area := (side_length ^ 2)
    let circle_radius := (1 - x) / (2 * Real.pi)
    let circle_area := Real.pi * (circle_radius ^ 2)
    let total_area := square_area + circle_area
    total_area = (x^2 / 16 + (1 - x)^2 / (4 * Real.pi)) -> 
    x = Real.pi / (Real.pi + 4) :=
by
  sorry

end minimize_sum_areas_l187_187610


namespace mopping_time_is_30_l187_187761

def vacuuming_time := 45
def dusting_time := 60
def brushing_time_per_cat := 5
def number_of_cats := 3
def total_free_time := 180
def free_time_left := 30

def total_cleaning_time := total_free_time - free_time_left
def brushing_time := brushing_time_per_cat * number_of_cats
def time_other_tasks := vacuuming_time + dusting_time + brushing_time

theorem mopping_time_is_30 : total_cleaning_time - time_other_tasks = 30 := by
  -- Calculation proof would go here
  sorry

end mopping_time_is_30_l187_187761


namespace no_six_digit_number_meets_criteria_l187_187778

def valid_digit (n : ℕ) := 2 ≤ n ∧ n ≤ 8

theorem no_six_digit_number_meets_criteria :
  ¬ ∃ (digits : Finset ℕ), digits.card = 6 ∧ (∀ x ∈ digits, valid_digit x) ∧ (digits.sum id = 42) :=
by {
  sorry
}

end no_six_digit_number_meets_criteria_l187_187778


namespace min_marbles_to_draw_l187_187793

theorem min_marbles_to_draw (reds greens blues yellows oranges purples : ℕ)
  (h_reds : reds = 35)
  (h_greens : greens = 25)
  (h_blues : blues = 24)
  (h_yellows : yellows = 18)
  (h_oranges : oranges = 15)
  (h_purples : purples = 12)
  : ∃ n : ℕ, n = 103 ∧ (∀ r g b y o p : ℕ, 
       r ≤ reds ∧ g ≤ greens ∧ b ≤ blues ∧ y ≤ yellows ∧ o ≤ oranges ∧ p ≤ purples ∧ 
       r < 20 ∧ g < 20 ∧ b < 20 ∧ y < 20 ∧ o < 20 ∧ p < 20 → r + g + b + y + o + p < n) ∧
      (∀ r g b y o p : ℕ, 
       r ≤ reds ∧ g ≤ greens ∧ b ≤ blues ∧ y ≤ yellows ∧ o ≤ oranges ∧ p ≤ purples ∧ 
       r + g + b + y + o + p = n → r = 20 ∨ g = 20 ∨ b = 20 ∨ y = 20 ∨ o = 20 ∨ p = 20) :=
sorry

end min_marbles_to_draw_l187_187793


namespace find_first_number_l187_187002

variable (x y : ℕ)

theorem find_first_number (h1 : y = 11) (h2 : x + (y + 3) = 19) : x = 5 :=
by
  sorry

end find_first_number_l187_187002


namespace animal_market_problem_l187_187439

theorem animal_market_problem:
  ∃ (s c : ℕ), 0 < s ∧ 0 < c ∧ 28 * s + 27 * c = 1200 ∧ c > s :=
by
  sorry

end animal_market_problem_l187_187439


namespace cricket_runs_l187_187657

variable (A B C D E : ℕ)

theorem cricket_runs
  (h1 : (A + B + C + D + E) = 180)
  (h2 : D = E + 5)
  (h3 : A = E + 8)
  (h4 : B = D + E)
  (h5 : B + C = 107) :
  E = 20 := by
  sorry

end cricket_runs_l187_187657


namespace find_duration_l187_187733

noncomputable def machine_times (x : ℝ) : Prop :=
  let tP := x + 5
  let tQ := x + 3
  let tR := 2 * (x * (x + 3) / 3)
  (1 / tP + 1 / tQ + 1 / tR = 1 / x) ∧ (tP > 0) ∧ (tQ > 0) ∧ (tR > 0)

theorem find_duration {x : ℝ} (h : machine_times x) : x = 3 :=
sorry

end find_duration_l187_187733


namespace percentage_disliked_by_both_l187_187978

theorem percentage_disliked_by_both 
  (total_comic_books : ℕ) 
  (percentage_females_like : ℕ) 
  (comic_books_males_like : ℕ) :
  total_comic_books = 300 →
  percentage_females_like = 30 →
  comic_books_males_like = 120 →
  ((total_comic_books - (total_comic_books * percentage_females_like / 100) - comic_books_males_like) * 100 / total_comic_books) = 30 :=
by
  intros h1 h2 h3
  sorry

end percentage_disliked_by_both_l187_187978


namespace triangles_from_sticks_l187_187023

theorem triangles_from_sticks (a1 a2 a3 a4 a5 a6 : ℕ) (h_diff: a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a1 ≠ a6 
∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a2 ≠ a6 
∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a3 ≠ a6 
∧ a4 ≠ a5 ∧ a4 ≠ a6 
∧ a5 ≠ a6) (h_order: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6) : 
  (a1 + a3 > a5 ∧ a1 + a5 > a3 ∧ a3 + a5 > a1) ∧ 
  (a2 + a4 > a6 ∧ a2 + a6 > a4 ∧ a4 + a6 > a2) :=
by
  sorry

end triangles_from_sticks_l187_187023


namespace real_number_solution_pure_imaginary_solution_zero_solution_l187_187489

noncomputable def real_number_condition (m : ℝ) : Prop :=
  m^2 - 3 * m + 2 = 0

noncomputable def pure_imaginary_condition (m : ℝ) : Prop :=
  (2 * m^2 - 3 * m - 2 = 0) ∧ ¬(m^2 - 3 * m + 2 = 0)

noncomputable def zero_condition (m : ℝ) : Prop :=
  (2 * m^2 - 3 * m - 2 = 0) ∧ (m^2 - 3 * m + 2 = 0)

theorem real_number_solution (m : ℝ) : real_number_condition m ↔ (m = 1 ∨ m = 2) := 
sorry

theorem pure_imaginary_solution (m : ℝ) : pure_imaginary_condition m ↔ (m = -1 / 2) :=
sorry

theorem zero_solution (m : ℝ) : zero_condition m ↔ (m = 2) :=
sorry

end real_number_solution_pure_imaginary_solution_zero_solution_l187_187489


namespace value_of_x_l187_187185

theorem value_of_x (x : ℝ) (h : 0.5 * x = 0.25 * 1500 - 30) : x = 690 :=
by
  sorry

end value_of_x_l187_187185


namespace inequality_solution_l187_187107

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l187_187107


namespace total_square_footage_l187_187441

-- Definitions from the problem conditions
def price_per_square_foot : ℝ := 98
def total_property_value : ℝ := 333200

-- The mathematical statement to prove
theorem total_square_footage : (total_property_value / price_per_square_foot) = 3400 :=
by
  -- Proof goes here (skipped with sorry)
  sorry

end total_square_footage_l187_187441


namespace last_month_games_l187_187836

-- Definitions and conditions
def this_month := 9
def next_month := 7
def total_games := 24

-- Question to prove
theorem last_month_games : total_games - (this_month + next_month) = 8 := 
by 
  sorry

end last_month_games_l187_187836


namespace calculate_total_driving_time_l187_187467

/--
A rancher needs to transport 400 head of cattle to higher ground 60 miles away.
His truck holds 20 head of cattle and travels at 60 miles per hour.
Prove that the total driving time to transport all cattle is 40 hours.
-/
theorem calculate_total_driving_time
  (total_cattle : Nat)
  (cattle_per_trip : Nat)
  (distance_one_way : Nat)
  (speed : Nat)
  (round_trip_miles : Nat)
  (total_miles : Nat)
  (total_time_hours : Nat)
  (h1 : total_cattle = 400)
  (h2 : cattle_per_trip = 20)
  (h3 : distance_one_way = 60)
  (h4 : speed = 60)
  (h5 : round_trip_miles = 2 * distance_one_way)
  (h6 : total_miles = (total_cattle / cattle_per_trip) * round_trip_miles)
  (h7 : total_time_hours = total_miles / speed) :
  total_time_hours = 40 :=
by
  sorry

end calculate_total_driving_time_l187_187467


namespace rationalize_denominator_l187_187406

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l187_187406


namespace monotonically_increasing_iff_l187_187707

noncomputable def f (x : ℝ) (a : ℝ) := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotonically_increasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a ≤ f y a) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) := 
sorry

end monotonically_increasing_iff_l187_187707


namespace complement_intersection_l187_187818

open Set -- Open the Set namespace to simplify notation for set operations

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def M : Set ℤ := {-1, 0, 1, 3}
def N : Set ℤ := {-2, 0, 2, 3}

theorem complement_intersection : (U \ M) ∩ N = ({-2, 2} : Set ℤ) :=
by
  sorry

end complement_intersection_l187_187818


namespace latin_student_sophomore_probability_l187_187508

variable (F S J SE : ℕ) -- freshmen, sophomores, juniors, seniors total
variable (FL SL JL SEL : ℕ) -- freshmen, sophomores, juniors, seniors taking latin
variable (p : ℚ) -- probability fraction
variable (m n : ℕ) -- relatively prime integers

-- Let the total number of students be 100 for simplicity in percentage calculations
-- Let us encode the given conditions
def conditions := 
  F = 40 ∧ 
  S = 30 ∧ 
  J = 20 ∧ 
  SE = 10 ∧ 
  FL = 40 ∧ 
  SL = S * 80 / 100 ∧ 
  JL = J * 50 / 100 ∧ 
  SEL = SE * 20 / 100

-- The probability calculation
def probability_sophomore (SL : ℕ) (FL SL JL SEL : ℕ) : ℚ := SL / (FL + SL + JL + SEL)

-- Target probability as a rational number
def target_probability := (6 : ℚ) / 19

theorem latin_student_sophomore_probability : 
  conditions F S J SE FL SL JL SEL → 
  probability_sophomore SL FL SL JL SEL = target_probability ∧ 
  m + n = 25 := 
by 
  sorry

end latin_student_sophomore_probability_l187_187508


namespace distance_to_axes_l187_187638

def point (P : ℝ × ℝ) : Prop :=
  P = (3, 5)

theorem distance_to_axes (P : ℝ × ℝ) (hx : P = (3, 5)) : 
  abs P.2 = 5 ∧ abs P.1 = 3 :=
by 
  sorry

end distance_to_axes_l187_187638


namespace necessary_but_not_sufficient_condition_l187_187950

variable (p q : Prop)

theorem necessary_but_not_sufficient_condition (hpq : p ∨ q) (h : p ∧ q) : p ∧ q ↔ (p ∨ q) := by
  sorry

end necessary_but_not_sufficient_condition_l187_187950


namespace xiao_dong_not_both_understand_english_and_french_l187_187586

variables (P Q : Prop)

theorem xiao_dong_not_both_understand_english_and_french (h : ¬ (P ∧ Q)) : P → ¬ Q :=
sorry

end xiao_dong_not_both_understand_english_and_french_l187_187586


namespace james_carrot_sticks_l187_187777

theorem james_carrot_sticks (x : ℕ) (h : x + 15 = 37) : x = 22 :=
by {
  sorry
}

end james_carrot_sticks_l187_187777


namespace second_alloy_amount_l187_187076

theorem second_alloy_amount (x : ℝ) : 
  (0.10 * 15 + 0.08 * x = 0.086 * (15 + x)) → 
  x = 35 := by 
sorry

end second_alloy_amount_l187_187076


namespace arithmetic_sequence_ratio_l187_187352

def arithmetic_sum (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_ratio :
  let a1 := 3
  let d1 := 3
  let l1 := 99
  let a2 := 4
  let d2 := 4
  let l2 := 100
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let sum1 := arithmetic_sum a1 d1 n1
  let sum2 := arithmetic_sum a2 d2 n2
  sum1 / sum2 = 1683 / 1300 :=
by {
  let a1 := 3
  let d1 := 3
  let l1 := 99
  let a2 := 4
  let d2 := 4
  let l2 := 100
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let sum1 := arithmetic_sum a1 d1 n1
  let sum2 := arithmetic_sum a2 d2 n2
  sorry
}

end arithmetic_sequence_ratio_l187_187352


namespace james_total_points_l187_187283

def points_per_correct_answer : ℕ := 2
def bonus_points_per_round : ℕ := 4
def total_rounds : ℕ := 5
def questions_per_round : ℕ := 5
def total_questions : ℕ := total_rounds * questions_per_round
def questions_missed_by_james : ℕ := 1
def questions_answered_by_james : ℕ := total_questions - questions_missed_by_james
def points_for_correct_answers : ℕ := questions_answered_by_james * points_per_correct_answer
def complete_rounds_by_james : ℕ := total_rounds - 1  -- Since James missed one question, he has 4 complete rounds
def bonus_points_by_james : ℕ := complete_rounds_by_james * bonus_points_per_round
def total_points : ℕ := points_for_correct_answers + bonus_points_by_james

theorem james_total_points : total_points = 64 := by
  sorry

end james_total_points_l187_187283


namespace find_y_l187_187515

theorem find_y {x y : ℝ} (hx : (8 : ℝ) = (1/4 : ℝ) * x) (hy : (y : ℝ) = (1/4 : ℝ) * (20 : ℝ)) (hprod : x * y = 160) : y = 5 :=
by {
  sorry
}

end find_y_l187_187515


namespace option_A_equal_l187_187394

theorem option_A_equal : (-2: ℤ)^(3: ℕ) = ((-2: ℤ)^(3: ℕ)) :=
by
  sorry

end option_A_equal_l187_187394


namespace handshakes_total_count_l187_187112

/-
Statement:
There are 30 gremlins and 20 imps at a Regional Mischief Meet. Only half of the imps are willing to shake hands with each other.
All cooperative imps shake hands with each other. All imps shake hands with each gremlin. Gremlins shake hands with every
other gremlin as well as all the imps. Each pair of creatures shakes hands at most once. Prove that the total number of handshakes is 1080.
-/

theorem handshakes_total_count (gremlins imps cooperative_imps : ℕ)
  (H1 : gremlins = 30)
  (H2 : imps = 20)
  (H3 : cooperative_imps = imps / 2) :
  let handshakes_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_cooperative_imps := cooperative_imps * (cooperative_imps - 1) / 2
  let handshakes_imps_gremlins := imps * gremlins
  handshakes_gremlins + handshakes_cooperative_imps + handshakes_imps_gremlins = 1080 := 
by {
  sorry
}

end handshakes_total_count_l187_187112


namespace factor_expr_l187_187916

theorem factor_expr (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3 * x + x^2) := 
sorry

end factor_expr_l187_187916


namespace calculate_down_payment_l187_187499

theorem calculate_down_payment : 
  let monthly_fee := 12
  let years := 3
  let total_paid := 482
  let num_months := years * 12
  let total_monthly_payments := num_months * monthly_fee
  let down_payment := total_paid - total_monthly_payments
  down_payment = 50 :=
by
  sorry

end calculate_down_payment_l187_187499


namespace solve_x_l187_187976

theorem solve_x (x : ℝ) (h1 : 8 * x^2 + 7 * x - 1 = 0) (h2 : 24 * x^2 + 53 * x - 7 = 0) : x = 1 / 8 :=
by
  sorry

end solve_x_l187_187976


namespace similarity_coordinates_l187_187659

theorem similarity_coordinates {B B1 : ℝ × ℝ} 
  (h₁ : ∃ (k : ℝ), k = 2 ∧ 
         (∀ (x y : ℝ), B = (x, y) → ∀ (x₁ y₁ : ℝ), B1 = (x₁, y₁) → x₁ = x / k ∨ x₁ = x / -k) ∧ 
         (∀ (x y : ℝ), B = (x, y) → ∀ (x₁ y₁ : ℝ), B1 = (x₁, y₁) → y₁ = y / k ∨ y₁ = y / -k))
  (h₂ : B = (-4, -2)) :
  B1 = (-2, -1) ∨ B1 = (2, 1) :=
sorry

end similarity_coordinates_l187_187659


namespace smallest_integer_for_inequality_l187_187600

theorem smallest_integer_for_inequality :
  ∃ x : ℤ, x^2 < 2 * x + 1 ∧ ∀ y : ℤ, y^2 < 2 * y + 1 → x ≤ y := sorry

end smallest_integer_for_inequality_l187_187600


namespace five_a_squared_plus_one_divisible_by_three_l187_187169

theorem five_a_squared_plus_one_divisible_by_three (a : ℤ) (h : a % 3 ≠ 0) : (5 * a^2 + 1) % 3 = 0 :=
sorry

end five_a_squared_plus_one_divisible_by_three_l187_187169


namespace only_powers_of_2_satisfy_condition_l187_187081

theorem only_powers_of_2_satisfy_condition:
  ∀ (n : ℕ), n ≥ 2 →
  (∃ (x : ℕ → ℕ), 
    ∀ (i j : ℕ), 
      0 < i ∧ i < n → 0 < j ∧ j < n → i ≠ j ∧ (n ∣ (2 * i + j)) → x i < x j) ↔
      ∃ (s : ℕ), n = 2^s ∧ s ≥ 1 :=
by
  sorry

end only_powers_of_2_satisfy_condition_l187_187081


namespace triangle_inequality_check_triangle_sets_l187_187607

theorem triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem check_triangle_sets :
  ¬triangle_inequality 1 2 3 ∧
  triangle_inequality 2 2 2 ∧
  ¬triangle_inequality 2 2 4 ∧
  ¬triangle_inequality 1 3 5 :=
by
  sorry

end triangle_inequality_check_triangle_sets_l187_187607


namespace seq_geom_seq_of_geom_and_arith_l187_187591

theorem seq_geom_seq_of_geom_and_arith (a : ℕ → ℕ) (b : ℕ → ℕ) 
  (h1 : ∃ a₁ : ℕ, ∀ n : ℕ, a n = a₁ * 2^(n-1))
  (h2 : ∃ b₁ d : ℕ, d = 3 ∧ ∀ n : ℕ, b (n + 1) = b₁ + n * d ∧ b₁ > 0) :
  ∃ r : ℕ, r = 8 ∧ ∃ a₁ : ℕ, ∀ n : ℕ, a (b (n + 1)) = a₁ * r^n :=
by
  sorry

end seq_geom_seq_of_geom_and_arith_l187_187591


namespace jack_shoes_time_l187_187074

theorem jack_shoes_time (J : ℝ) (h : J + 2 * (J + 3) = 18) : J = 4 :=
by
  sorry

end jack_shoes_time_l187_187074


namespace intersection_two_elements_l187_187661

open Real Set

-- Definitions
def M (k : ℝ) : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ y = k * (x - 1) + 1}
def N : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ x^2 + y^2 - 2 * y = 0}

-- Statement of the problem
theorem intersection_two_elements (k : ℝ) (hk : k ≠ 0) :
  ∃ x1 y1 x2 y2 : ℝ,
    (x1, y1) ∈ M k ∧ (x1, y1) ∈ N ∧ 
    (x2, y2) ∈ M k ∧ (x2, y2) ∈ N ∧ 
    (x1, y1) ≠ (x2, y2) := sorry

end intersection_two_elements_l187_187661


namespace appropriate_selling_price_l187_187562

-- Define the given conditions
def cost_per_kg : ℝ := 40
def base_price : ℝ := 50
def base_sales_volume : ℝ := 500
def sales_decrease_per_yuan : ℝ := 10
def available_capital : ℝ := 10000
def desired_profit : ℝ := 8000

-- Define the sales volume function dependent on selling price x
def sales_volume (x : ℝ) : ℝ := base_sales_volume - (x - base_price) * sales_decrease_per_yuan

-- Define the profit function dependent on selling price x
def profit (x : ℝ) : ℝ := (x - cost_per_kg) * sales_volume x

-- Prove that the appropriate selling price is 80 yuan
theorem appropriate_selling_price : 
  ∃ x : ℝ, profit x = desired_profit ∧ x = 80 :=
by
  sorry

end appropriate_selling_price_l187_187562


namespace initial_population_l187_187617

theorem initial_population (P : ℝ) (h1 : 0.76 * P = 3553) : P = 4678 :=
by
  sorry

end initial_population_l187_187617


namespace slope_of_line_AF_parabola_l187_187721

theorem slope_of_line_AF_parabola (A : ℝ × ℝ)
  (hA_on_parabola : A.snd ^ 2 = 4 * A.fst)
  (h_dist_focus : Real.sqrt ((A.fst - 1) ^ 2 + A.snd ^ 2) = 4) :
  (A.snd / (A.fst - 1) = Real.sqrt 3 ∨ A.snd / (A.fst - 1) = -Real.sqrt 3) :=
sorry

end slope_of_line_AF_parabola_l187_187721


namespace f_seven_l187_187336

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (h : ℝ) : f (-h) = -f (h)
axiom periodic_function (h : ℝ) : f (h + 4) = f (h)
axiom f_one : f 1 = 2

theorem f_seven : f (7) = -2 :=
by
  sorry

end f_seven_l187_187336


namespace intersection_complement_eq_l187_187446

/-- Define the sets U, A, and B -/
def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {3, 7, 9}
def B : Set ℕ := {1, 9}

/-- Define the complement of B with respect to U -/
def complement_U_B : Set ℕ := U \ B

/-- Theorem stating the intersection of A and the complement of B with respect to U -/
theorem intersection_complement_eq : A ∩ complement_U_B = {3, 7} :=
by
  sorry

end intersection_complement_eq_l187_187446


namespace hundred_million_is_ten_times_ten_million_one_million_is_hundred_times_ten_thousand_l187_187719

-- Definitions for the given problem
def one_hundred_million : ℕ := 100000000
def ten_million : ℕ := 10000000
def one_million : ℕ := 1000000
def ten_thousand : ℕ := 10000

-- Proving the statements
theorem hundred_million_is_ten_times_ten_million :
  one_hundred_million = 10 * ten_million :=
by
  sorry

theorem one_million_is_hundred_times_ten_thousand :
  one_million = 100 * ten_thousand :=
by
  sorry

end hundred_million_is_ten_times_ten_million_one_million_is_hundred_times_ten_thousand_l187_187719


namespace negation_of_universal_statement_l187_187427

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, x^3 - 3 * x > 0) ↔ ∃ x : ℝ, x^3 - 3 * x ≤ 0 :=
by
  sorry

end negation_of_universal_statement_l187_187427


namespace sfl_entrances_l187_187289

theorem sfl_entrances (people_per_entrance total_people entrances : ℕ) 
  (h1: people_per_entrance = 283) 
  (h2: total_people = 1415) 
  (h3: total_people = people_per_entrance * entrances) 
  : entrances = 5 := 
  by 
  rw [h1, h2] at h3
  sorry

end sfl_entrances_l187_187289


namespace range_of_m_minimum_value_ab_l187_187805

-- Define the given condition as a predicate on the real numbers
def domain_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0

-- Define the first part of the proof problem: range of m
theorem range_of_m :
  (∀ m : ℝ, domain_condition m) → ∀ m : ℝ, m ≤ 6 :=
sorry

-- Define the second part of the proof problem: minimum value of 4a + 7b
theorem minimum_value_ab (n : ℝ) (a b : ℝ) (h : n = 6) :
  (∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (4 / (a + 5 * b) + 1 / (3 * a + 2 * b) = n)) → 
  ∃ (a b : ℝ), 4 * a + 7 * b = 3 / 2 :=
sorry

end range_of_m_minimum_value_ab_l187_187805


namespace average_gas_mileage_round_trip_l187_187827

theorem average_gas_mileage_round_trip :
  let distance_to_conference := 150
  let distance_return_trip := 150
  let mpg_sedan := 25
  let mpg_hybrid := 40
  let total_distance := distance_to_conference + distance_return_trip
  let gas_used_sedan := distance_to_conference / mpg_sedan
  let gas_used_hybrid := distance_return_trip / mpg_hybrid
  let total_gas_used := gas_used_sedan + gas_used_hybrid
  let average_gas_mileage := total_distance / total_gas_used
  average_gas_mileage = 31 := by
    sorry

end average_gas_mileage_round_trip_l187_187827


namespace last_three_digits_of_2_pow_15000_l187_187156

-- We need to define the given condition as a hypothesis and then state the goal.
theorem last_three_digits_of_2_pow_15000 :
  (2 ^ 500 ≡ 1 [MOD 1250]) → (2 ^ 15000 ≡ 1 [MOD 1000]) := by
  sorry

end last_three_digits_of_2_pow_15000_l187_187156


namespace incorrect_operation_l187_187196

noncomputable def a : ℤ := -2

def operation_A (a : ℤ) : ℤ := abs a
def operation_B (a : ℤ) : ℤ := abs (a - 2) + abs (a + 1)
def operation_C (a : ℤ) : ℤ := -a ^ 3 + a + (-a) ^ 2
def operation_D (a : ℤ) : ℤ := abs a ^ 2

theorem incorrect_operation :
  operation_D a ≠ abs 4 :=
by
  sorry

end incorrect_operation_l187_187196


namespace negation_of_proposition_l187_187087

theorem negation_of_proposition :
  (∀ x : ℝ, x > 0 → x + (1 / x) ≥ 2) →
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + (1 / x₀) < 2) :=
sorry

end negation_of_proposition_l187_187087


namespace triangle_third_side_range_l187_187750

theorem triangle_third_side_range {x : ℤ} : 
  (7 < x ∧ x < 17) → (4 ≤ x ∧ x ≤ 16) :=
by
  sorry

end triangle_third_side_range_l187_187750


namespace number_of_classes_l187_187399

theorem number_of_classes (total_basketballs classes_basketballs : ℕ) (h1 : total_basketballs = 54) (h2 : classes_basketballs = 7) : total_basketballs / classes_basketballs = 7 := by
  sorry

end number_of_classes_l187_187399


namespace selling_price_calculation_l187_187411

-- Given conditions
def cost_price : ℚ := 110
def gain_percent : ℚ := 13.636363636363626

-- Theorem Statement
theorem selling_price_calculation : 
  (cost_price * (1 + gain_percent / 100)) = 125 :=
by
  sorry

end selling_price_calculation_l187_187411


namespace solution_set_of_quadratic_inequality_2_l187_187255

-- Definitions
variables {a b c x : ℝ}
def quadratic_inequality_1 (a b c x : ℝ) := a * x^2 + b * x + c < 0
def quadratic_inequality_2 (a b c x : ℝ) := a * x^2 - b * x + c > 0

-- Conditions
axiom condition_1 : ∀ x, quadratic_inequality_1 a b c x ↔ (x < -2 ∨ x > -1/2)
axiom condition_2 : a < 0
axiom condition_3 : ∃ x, a * x^2 + b * x + c = 0 ∧ (x = -2 ∨ x = -1/2)
axiom condition_4 : b = 5 * a / 2
axiom condition_5 : c = a

-- Proof Problem
theorem solution_set_of_quadratic_inequality_2 : ∀ x, quadratic_inequality_2 a b c x ↔ (1/2 < x ∧ x < 2) :=
by
  -- Proof goes here
  sorry

end solution_set_of_quadratic_inequality_2_l187_187255


namespace roots_polynomial_value_l187_187969

theorem roots_polynomial_value (a b c : ℝ) 
  (h1 : a + b + c = 15)
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 12) :
  (2 + a) * (2 + b) * (2 + c) = 130 := 
by
  sorry

end roots_polynomial_value_l187_187969


namespace first_pump_half_time_l187_187092

theorem first_pump_half_time (t : ℝ) : 
  (∃ (t : ℝ), (1/(2*t) + 1/1.1111111111111112) * (1/2) = 1/2) -> 
  t = 5 :=
by
  sorry

end first_pump_half_time_l187_187092


namespace anna_score_below_90_no_A_l187_187912

def score_implies_grade (score : ℝ) : Prop :=
  score > 90 → true

theorem anna_score_below_90_no_A (score : ℝ) (A_grade : Prop) (h : score_implies_grade score) :
  score < 90 → ¬ A_grade :=
by sorry

end anna_score_below_90_no_A_l187_187912


namespace multiplication_example_l187_187555

theorem multiplication_example : 28 * (9 + 2 - 5) * 3 = 504 := by 
  sorry

end multiplication_example_l187_187555


namespace find_A_l187_187868

def divisible_by(a b : ℕ) := b % a = 0

def valid_digit_A (A : ℕ) : Prop := (A = 0 ∨ A = 2 ∨ A = 4 ∨ A = 6 ∨ A = 8) ∧ divisible_by A 75

theorem find_A : ∃! A : ℕ, valid_digit_A A :=
by {
  sorry
}

end find_A_l187_187868


namespace calculate_a_plus_b_l187_187980

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def g (x : ℝ) : ℝ := 3 * x - 7

theorem calculate_a_plus_b (a b : ℝ) (h : ∀ x : ℝ, g (f a b x) = 4 * x + 6) : a + b = 17 / 3 :=
by
  sorry

end calculate_a_plus_b_l187_187980


namespace div_add_example_l187_187069

theorem div_add_example : 150 / (10 / 2) + 5 = 35 := by
  sorry

end div_add_example_l187_187069


namespace cube_root_expression_l187_187421

theorem cube_root_expression (N : ℝ) (h : N > 1) : 
    (N^(1/3)^(1/3)^(1/3)^(1/3)) = N^(40/81) :=
sorry

end cube_root_expression_l187_187421


namespace round_to_nearest_whole_l187_187791

theorem round_to_nearest_whole (x : ℝ) (hx : x = 12345.49999) : round x = 12345 := by
  -- Proof omitted.
  sorry

end round_to_nearest_whole_l187_187791


namespace option_D_is_divisible_by_9_l187_187161

theorem option_D_is_divisible_by_9 (k : ℕ) (hk : k > 0) : 9 ∣ 3 * (2 + 7^k) := 
sorry

end option_D_is_divisible_by_9_l187_187161


namespace emily_jumping_game_l187_187258

def tiles_number (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 5 = 2

theorem emily_jumping_game : tiles_number 47 :=
by
  unfold tiles_number
  sorry

end emily_jumping_game_l187_187258


namespace sum_of_consecutive_even_numbers_l187_187190

theorem sum_of_consecutive_even_numbers (n : ℤ) 
  (h : n + 4 = 14) : n + (n + 2) + (n + 4) + (n + 6) = 52 :=
by
  sorry

end sum_of_consecutive_even_numbers_l187_187190


namespace solutions_to_equation_l187_187775

variable (x : ℝ)

def original_eq : Prop :=
  (3 * x - 9) / (x^2 - 6 * x + 8) = (x + 1) / (x - 2)

theorem solutions_to_equation : (original_eq 1 ∧ original_eq 5) :=
by
  sorry

end solutions_to_equation_l187_187775


namespace prove_n_eq_one_l187_187017

-- Definitions of the vectors a and b
def vector_a (n : ℝ) : ℝ × ℝ := (1, n)
def vector_b (n : ℝ) : ℝ × ℝ := (-1, n - 2)

-- Definition of collinearity between two vectors
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- Theorem to prove that if a and b are collinear, then n = 1
theorem prove_n_eq_one (n : ℝ) (h_collinear : collinear (vector_a n) (vector_b n)) : n = 1 :=
sorry

end prove_n_eq_one_l187_187017


namespace area_circle_minus_square_l187_187558

theorem area_circle_minus_square {r : ℝ} (h : r = 1/2) : 
  (π * r^2) - (1^2) = (π / 4) - 1 :=
by
  rw [h]
  sorry

end area_circle_minus_square_l187_187558


namespace hyeoncheol_initial_money_l187_187372

theorem hyeoncheol_initial_money
  (X : ℕ)
  (h1 : X / 2 / 2 = 1250) :
  X = 5000 :=
sorry

end hyeoncheol_initial_money_l187_187372


namespace quadratic_congruence_solution_l187_187444

theorem quadratic_congruence_solution (p : ℕ) (hp : Nat.Prime p) : 
  ∃ n : ℕ, 6 * n^2 + 5 * n + 1 ≡ 0 [MOD p] := 
sorry

end quadratic_congruence_solution_l187_187444


namespace arithmetic_sequence_a5_l187_187198

theorem arithmetic_sequence_a5 {a : ℕ → ℝ} (h₁ : a 2 + a 8 = 16) : a 5 = 8 :=
sorry

end arithmetic_sequence_a5_l187_187198


namespace statement_true_when_b_le_a_div_5_l187_187725

theorem statement_true_when_b_le_a_div_5
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h₀ : ∀ x : ℝ, f x = 5 * x + 3)
  (h₁ : ∀ x : ℝ, |f x + 7| < a ↔ |x + 2| < b)
  (h₂ : 0 < a)
  (h₃ : 0 < b) :
  b ≤ a / 5 :=
by
  sorry

end statement_true_when_b_le_a_div_5_l187_187725


namespace max_trading_cards_l187_187829

theorem max_trading_cards (h : 10 ≥ 1.25 * nat):
  nat ≤ 8 :=
sorry

end max_trading_cards_l187_187829


namespace find_real_number_l187_187656

theorem find_real_number (x : ℝ) (h1 : 0 < x) (h2 : ⌊x⌋ * x = 72) : x = 9 :=
sorry

end find_real_number_l187_187656


namespace room_height_l187_187147

-- Define the conditions
def total_curtain_length : ℕ := 101
def extra_material : ℕ := 5

-- Define the statement to be proven
theorem room_height : total_curtain_length - extra_material = 96 :=
by
  sorry

end room_height_l187_187147


namespace more_boys_than_girls_l187_187937

theorem more_boys_than_girls (total_people : ℕ) (num_girls : ℕ) (num_boys : ℕ) (more_boys : ℕ) : 
  total_people = 133 ∧ num_girls = 50 ∧ num_boys = total_people - num_girls ∧ more_boys = num_boys - num_girls → more_boys = 33 :=
by 
  sorry

end more_boys_than_girls_l187_187937


namespace gcd_lcm_product_75_90_l187_187180

theorem gcd_lcm_product_75_90 :
  let a := 75
  let b := 90
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  gcd_ab * lcm_ab = 6750 :=
by
  let a := 75
  let b := 90
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  sorry

end gcd_lcm_product_75_90_l187_187180


namespace color_plane_with_two_colors_l187_187384

/-- Given a finite set of circles that divides the plane into regions, we can color the plane such that no two adjacent regions have the same color. -/
theorem color_plane_with_two_colors (circles : Finset (Set ℝ)) :
  (∀ (r1 r2 : Set ℝ), (r1 ∩ r2).Nonempty → ∃ (coloring : Set ℝ → Bool), (coloring r1 ≠ coloring r2)) :=
  sorry

end color_plane_with_two_colors_l187_187384


namespace clock_angle_8_15_l187_187514

theorem clock_angle_8_15:
  ∃ angle : ℝ, time_on_clock = 8.25 → angle = 157.5 := sorry

end clock_angle_8_15_l187_187514


namespace find_y_value_l187_187639

/-- Given angles and conditions, find the value of y in the geometric figure. -/
theorem find_y_value
  (AB_parallel_DC : true) -- AB is parallel to DC
  (ACE_straight_line : true) -- ACE is a straight line
  (angle_ACF : ℝ := 130) -- ∠ACF = 130°
  (angle_CBA : ℝ := 60) -- ∠CBA = 60°
  (angle_ACB : ℝ := 100) -- ∠ACB = 100°
  (angle_ADC : ℝ := 125) -- ∠ADC = 125°
  : 35 = 35 := -- y = 35°
by
  sorry

end find_y_value_l187_187639


namespace maximize_Miraflores_win_l187_187929

-- Definitions based on given conditions
def voters_count (n : ℕ) : ℕ := 2 * n
def support_Miraflores (n : ℕ) : ℕ := n + 1
def support_opponent (n : ℕ) : ℕ := n - 1

-- Theorem statement
theorem maximize_Miraflores_win (n : ℕ) (hn : n > 0) : 
  ∃ (d1 d2 : ℕ), d1 = 1 ∧ d2 = 2 * n - 1 ∧ support_Miraflores n > support_opponent n := 
sorry

end maximize_Miraflores_win_l187_187929


namespace arithmetic_sequence_value_l187_187465

variable (a : ℕ → ℝ)
variable (a₁ d a₇ a₅ : ℝ)
variable (h_seq : ∀ n, a n = a₁ + (n - 1) * d)
variable (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120)

theorem arithmetic_sequence_value :
  a 7 - 1/3 * a 5 = 16 :=
sorry

end arithmetic_sequence_value_l187_187465


namespace christine_stickers_l187_187708

theorem christine_stickers (stickers_has stickers_needs : ℕ) (h_has : stickers_has = 11) (h_needs : stickers_needs = 19) : 
  stickers_has + stickers_needs = 30 :=
by 
  sorry

end christine_stickers_l187_187708


namespace not_periodic_fraction_l187_187739

theorem not_periodic_fraction :
  ¬ ∃ (n k : ℕ), ∀ m ≥ n + k, ∃ l, 10^m + l = 10^(m+n) + l ∧ ((0.1234567891011121314 : ℝ) = (0.1234567891011121314 + l / (10^(m+n)))) :=
sorry

end not_periodic_fraction_l187_187739


namespace q_zero_l187_187361

noncomputable def q (x : ℝ) : ℝ := sorry -- Definition of the polynomial q(x) is required here.

theorem q_zero : 
  (∀ n : ℕ, n ≤ 7 → q (3^n) = 1 / 3^n) →
  q 0 = 0 :=
by 
  sorry

end q_zero_l187_187361


namespace ahmed_goats_is_13_l187_187288

def adam_goats : ℕ := 7

def andrew_goats : ℕ := 2 * adam_goats + 5

def ahmed_goats : ℕ := andrew_goats - 6

theorem ahmed_goats_is_13 : ahmed_goats = 13 :=
by
  sorry

end ahmed_goats_is_13_l187_187288


namespace mean_steps_per_day_l187_187598

theorem mean_steps_per_day (total_steps : ℕ) (days_in_april : ℕ) (h_total : total_steps = 243000) (h_days : days_in_april = 30) :
  (total_steps / days_in_april) = 8100 :=
by
  sorry

end mean_steps_per_day_l187_187598


namespace unsuccessful_attempts_124_l187_187506

theorem unsuccessful_attempts_124 (num_digits: ℕ) (choices_per_digit: ℕ) (total_attempts: ℕ):
  num_digits = 3 → choices_per_digit = 5 → total_attempts = choices_per_digit ^ num_digits →
  total_attempts - 1 = 124 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact sorry

end unsuccessful_attempts_124_l187_187506


namespace problem_statement_l187_187862

theorem problem_statement (x : ℝ) (h : 7 * x = 3) : 150 * (1 / x) = 350 :=
by
  sorry

end problem_statement_l187_187862


namespace a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l187_187681

variables (a b c d : ℝ)

-- Given conditions
def first_condition : Prop := a + b = c + d
def second_condition : Prop := a^3 + b^3 = c^3 + d^3

-- Proof problem for part (a)
theorem a_b_fifth_power_equals_c_d_fifth_power 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : a^5 + b^5 = c^5 + d^5 := 
sorry

-- Proof problem for part (b)
theorem cannot_conclude_fourth_powers 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : ¬ (a^4 + b^4 = c^4 + d^4) :=
sorry

end a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l187_187681


namespace arithmetic_seq_properties_l187_187059

theorem arithmetic_seq_properties (a : ℕ → ℝ) (d a1 : ℝ) (S : ℕ → ℝ) :
  (a 1 + a 3 = 8) ∧ (a 4 ^ 2 = a 2 * a 9) →
  ((a1 = 4 ∧ d = 0 ∧ (∀ n, S n = 4 * n)) ∨
   (a1 = 1 ∧ d = 3 ∧ (∀ n, S n = (3 * n^2 - n) / 2))) := 
sorry

end arithmetic_seq_properties_l187_187059


namespace first_discount_percentage_l187_187492

theorem first_discount_percentage (original_price final_price : ℝ) (additional_discount : ℝ) (x : ℝ) 
  (h1 : original_price = 400) 
  (h2 : additional_discount = 0.05) 
  (h3 : final_price = 342) 
  (hx : (original_price * (100 - x) / 100) * (1 - additional_discount) = final_price) :
  x = 10 := 
sorry

end first_discount_percentage_l187_187492


namespace fish_population_estimate_l187_187541

theorem fish_population_estimate 
  (caught_first : ℕ) 
  (caught_first_marked : ℕ) 
  (caught_second : ℕ) 
  (caught_second_marked : ℕ) 
  (proportion_eq : (caught_second_marked : ℚ) / caught_second = (caught_first_marked : ℚ) / caught_first) 
  : caught_first * caught_second / caught_second_marked = 750 := 
by 
  sorry

-- Conditions used as definitions in Lean 4
def pond_fish_total (caught_first : ℕ) (caught_second : ℕ) (caught_second_marked : ℕ) : ℚ :=
  (caught_first : ℚ) * (caught_second : ℚ) / (caught_second_marked : ℚ)

-- Example usage of conditions
example : pond_fish_total 30 50 2 = 750 := 
by
  sorry

end fish_population_estimate_l187_187541


namespace domain_of_composed_function_l187_187878

theorem domain_of_composed_function {f : ℝ → ℝ} (h : ∀ x, -1 < x ∧ x < 1 → f x ∈ Set.Ioo (-1:ℝ) 1) :
  ∀ x, 0 < x ∧ x < 1 → f (2*x-1) ∈ Set.Ioo (-1:ℝ) 1 := by
  sorry

end domain_of_composed_function_l187_187878


namespace sin_double_angle_solution_l187_187066

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l187_187066


namespace lukas_averages_points_l187_187852

theorem lukas_averages_points (total_points : ℕ) (num_games : ℕ) (average_points : ℕ)
  (h_total: total_points = 60) (h_games: num_games = 5) : average_points = total_points / num_games :=
sorry

end lukas_averages_points_l187_187852


namespace log_relation_l187_187291

noncomputable def a := Real.log 3 / Real.log 4
noncomputable def b := Real.log 3 / Real.log 0.4
def c := (1 / 2) ^ 2

theorem log_relation (h1 : a = Real.log 3 / Real.log 4)
                     (h2 : b = Real.log 3 / Real.log 0.4)
                     (h3 : c = (1 / 2) ^ 2) : a > c ∧ c > b :=
by
  sorry

end log_relation_l187_187291


namespace last_place_is_Fedya_l187_187635

def position_is_valid (position : ℕ) := position >= 1 ∧ position <= 4

variable (Misha Anton Petya Fedya : ℕ)

axiom Misha_statement: position_is_valid Misha → Misha ≠ 1 ∧ Misha ≠ 4
axiom Anton_statement: position_is_valid Anton → Anton ≠ 4
axiom Petya_statement: position_is_valid Petya → Petya = 1
axiom Fedya_statement: position_is_valid Fedya → Fedya = 4

theorem last_place_is_Fedya : ∃ (x : ℕ), x = Fedya ∧ Fedya = 4 :=
by
  sorry

end last_place_is_Fedya_l187_187635


namespace decimal_equiv_of_fraction_l187_187563

theorem decimal_equiv_of_fraction : (1 / 5) ^ 2 = 0.04 := by
  sorry

end decimal_equiv_of_fraction_l187_187563


namespace total_grazing_area_l187_187892

-- Define the dimensions of the field
def field_width : ℝ := 46
def field_height : ℝ := 20

-- Define the length of the rope
def rope_length : ℝ := 17

-- Define the radius and position of the fenced area
def fenced_radius : ℝ := 5
def fenced_distance_x : ℝ := 25
def fenced_distance_y : ℝ := 10

-- Given the conditions, prove the total grazing area
theorem total_grazing_area (field_width field_height rope_length fenced_radius fenced_distance_x fenced_distance_y : ℝ) :
  (π * rope_length^2 / 4) = 227.07 :=
by
  sorry

end total_grazing_area_l187_187892


namespace radio_range_l187_187806

-- Define constants for speeds and time
def speed_team_1 : ℝ := 20
def speed_team_2 : ℝ := 30
def time : ℝ := 2.5

-- Define the distances each team travels
def distance_team_1 := speed_team_1 * time
def distance_team_2 := speed_team_2 * time

-- Define the total distance which is the range of the radios
def total_distance := distance_team_1 + distance_team_2

-- Prove that the total distance when they lose radio contact is 125 miles
theorem radio_range : total_distance = 125 := by
  sorry

end radio_range_l187_187806


namespace pump_rates_l187_187953

theorem pump_rates (x y z : ℝ)
(h1 : x + y + z = 14)
(h2 : z = x + 3)
(h3 : y = 11 - 2 * x)
(h4 : 9 / x = (28 - 2 * y) / z)
: x = 3 ∧ y = 5 ∧ z = 6 :=
by
  sorry

end pump_rates_l187_187953


namespace clock_angle_at_7_oclock_l187_187955

theorem clock_angle_at_7_oclock : 
  ∀ (hour_angle minute_angle : ℝ), 
    (12 : ℝ) * (30 : ℝ) = 360 →
    (7 : ℝ) * (30 : ℝ) = 210 →
    (210 : ℝ) > 180 →
    (360 : ℝ) - (210 : ℝ) = 150 →
    hour_angle = 7 * 30 →
    minute_angle = 0 →
    min (abs (hour_angle - minute_angle)) (abs ((360 - hour_angle) - minute_angle)) = 150 := by
  sorry

end clock_angle_at_7_oclock_l187_187955


namespace myrtle_hens_l187_187480

/-- Myrtle has some hens that lay 3 eggs a day. She was gone for 7 days and told her neighbor 
    to take as many as they would like. The neighbor took 12 eggs. Once home, Myrtle collected 
    the remaining eggs, dropping 5 on the way into her house. Myrtle has 46 eggs. Prove 
    that Myrtle has 3 hens. -/
theorem myrtle_hens (eggs_per_hen_per_day hens days neighbor_took dropped remaining_hens_eggs : ℕ) 
    (h1 : eggs_per_hen_per_day = 3) 
    (h2 : days = 7) 
    (h3 : neighbor_took = 12) 
    (h4 : dropped = 5) 
    (h5 : remaining_hens_eggs = 46) : 
    hens = 3 := 
by 
  sorry

end myrtle_hens_l187_187480


namespace tenth_number_in_row_1_sum_of_2023rd_numbers_l187_187815

noncomputable def a (n : ℕ) := (-2)^n
noncomputable def b (n : ℕ) := a n + (n + 1)

theorem tenth_number_in_row_1 : a 10 = (-2)^10 := 
sorry

theorem sum_of_2023rd_numbers : a 2023 + b 2023 = -(2^2024) + 2024 := 
sorry

end tenth_number_in_row_1_sum_of_2023rd_numbers_l187_187815


namespace graph_EQ_a_l187_187041

theorem graph_EQ_a (x y : ℝ) : (x - 2) * (y + 3) = 0 ↔ x = 2 ∨ y = -3 :=
by sorry

end graph_EQ_a_l187_187041


namespace find_m_of_odd_function_l187_187284

theorem find_m_of_odd_function (m : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = ((x + 3) * (x + m)) / x)
  (h₂ : ∀ x, f (-x) = -f x) : m = -3 :=
sorry

end find_m_of_odd_function_l187_187284


namespace lcm_factor_l187_187295

theorem lcm_factor (A B : ℕ) (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) 
  (hcf_eq : hcf = 15) (factor1_eq : factor1 = 11) (A_eq : A = 225) 
  (hcf_divides_A : hcf ∣ A) (lcm_eq : Nat.lcm A B = hcf * factor1 * factor2) : 
  factor2 = 15 :=
by
  sorry

end lcm_factor_l187_187295


namespace sequence_positive_and_divisible_l187_187275

theorem sequence_positive_and_divisible:
  ∃ (a : ℕ → ℕ), 
    (a 1 = 2) ∧ (a 2 = 500) ∧ (a 3 = 2000) ∧ 
    (∀ n ≥ 2, (a (n + 2) + a (n + 1)) * a (n - 1) = a (n + 1) * (a (n + 1) + a (n - 1))) ∧ 
    (∀ n, a n > 0) ∧ 
    (2 ^ 2000 ∣ a 2000) := 
sorry

end sequence_positive_and_divisible_l187_187275


namespace passengers_on_plane_l187_187202

variables (P : ℕ) (fuel_per_mile : ℕ := 20) (fuel_per_person : ℕ := 3) (fuel_per_bag : ℕ := 2)
variables (num_crew : ℕ := 5) (bags_per_person : ℕ := 2) (trip_distance : ℕ := 400)
variables (total_fuel : ℕ := 106000)

def total_people := P + num_crew
def total_bags := bags_per_person * total_people
def total_fuel_per_mile := fuel_per_mile + fuel_per_person * P + fuel_per_bag * total_bags
def total_trip_fuel := trip_distance * total_fuel_per_mile

theorem passengers_on_plane : total_trip_fuel = total_fuel → P = 33 := 
by
  sorry

end passengers_on_plane_l187_187202


namespace cost_price_percentage_l187_187611

theorem cost_price_percentage (MP CP SP : ℝ) (h1 : SP = 0.88 * MP) (h2 : SP = 1.375 * CP) :
  (CP / MP) * 100 = 64 :=
by
  sorry

end cost_price_percentage_l187_187611


namespace shopkeeper_net_loss_percent_l187_187108

theorem shopkeeper_net_loss_percent (cp : ℝ)
  (sp1 sp2 sp3 sp4 : ℝ)
  (h_cp : cp = 1000)
  (h_sp1 : sp1 = cp * 1.1)
  (h_sp2 : sp2 = cp * 0.9)
  (h_sp3 : sp3 = cp * 1.2)
  (h_sp4 : sp4 = cp * 0.75) :
  ((cp + cp + cp + cp) - (sp1 + sp2 + sp3 + sp4)) / (cp + cp + cp + cp) * 100 = 1.25 :=
by sorry

end shopkeeper_net_loss_percent_l187_187108


namespace base8_357_plus_base13_4CD_eq_1084_l187_187602

def C := 12
def D := 13

def base8_357 := 3 * (8^2) + 5 * (8^1) + 7 * (8^0)
def base13_4CD := 4 * (13^2) + C * (13^1) + D * (13^0)

theorem base8_357_plus_base13_4CD_eq_1084 :
  base8_357 + base13_4CD = 1084 :=
by
  sorry

end base8_357_plus_base13_4CD_eq_1084_l187_187602


namespace marys_number_l187_187923

theorem marys_number (j m : ℕ) (h₁ : j * m = 2002)
  (h₂ : ∃ k, k * m = 2002 ∧ k ≠ j)
  (h₃ : ∃ l, j * l = 2002 ∧ l ≠ m) :
  m = 1001 :=
sorry

end marys_number_l187_187923


namespace people_sharing_pizzas_l187_187415

-- Definitions based on conditions
def number_of_pizzas : ℝ := 21.0
def slices_per_person : ℝ := 2.625
def slices_per_pizza : ℝ := 8.0

-- Theorem to prove the number of people
theorem people_sharing_pizzas : (number_of_pizzas * slices_per_pizza) / slices_per_person = 64 :=
by
  sorry

end people_sharing_pizzas_l187_187415


namespace find_smallest_int_cube_ends_368_l187_187310

theorem find_smallest_int_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 500 = 368 ∧ n = 14 :=
by
  sorry

end find_smallest_int_cube_ends_368_l187_187310


namespace ott_fraction_of_total_money_l187_187517

-- Definitions for the conditions
def Moe_initial_money (x : ℕ) : ℕ := 3 * x
def Loki_initial_money (x : ℕ) : ℕ := 5 * x
def Nick_initial_money (x : ℕ) : ℕ := 4 * x
def Total_initial_money (x : ℕ) : ℕ := Moe_initial_money x + Loki_initial_money x + Nick_initial_money x
def Ott_received_money (x : ℕ) : ℕ := 3 * x

-- Making the statement we want to prove
theorem ott_fraction_of_total_money (x : ℕ) : 
  (Ott_received_money x) / (Total_initial_money x) = 1 / 4 := by
  sorry

end ott_fraction_of_total_money_l187_187517


namespace min_value_x2_plus_y2_l187_187164

theorem min_value_x2_plus_y2 :
  ∀ x y : ℝ, (x + 5)^2 + (y - 12)^2 = 196 → x^2 + y^2 ≥ 1 :=
by
  intros x y h
  sorry

end min_value_x2_plus_y2_l187_187164


namespace fourth_powers_sum_is_8432_l187_187629

def sum_fourth_powers (x y : ℝ) : ℝ := x^4 + y^4

theorem fourth_powers_sum_is_8432 (x y : ℝ) (h₁ : x + y = 10) (h₂ : x * y = 4) : 
  sum_fourth_powers x y = 8432 :=
by
  sorry

end fourth_powers_sum_is_8432_l187_187629


namespace total_spent_on_index_cards_l187_187931

-- Definitions for conditions
def index_cards_per_student : ℕ := 10
def periods_per_day : ℕ := 6
def students_per_class : ℕ := 30
def cost_per_pack : ℕ := 3
def cards_per_pack : ℕ := 50

-- Theorem to be proven
theorem total_spent_on_index_cards :
  let total_students := students_per_class * periods_per_day
  let total_cards := total_students * index_cards_per_student
  let packs_needed := total_cards / cards_per_pack
  let total_cost := packs_needed * cost_per_pack
  total_cost = 108 :=
by
  sorry

end total_spent_on_index_cards_l187_187931


namespace onions_left_l187_187340

def sallyOnions : ℕ := 5
def fredOnions : ℕ := 9
def onionsGivenToSara : ℕ := 4

theorem onions_left : (sallyOnions + fredOnions) - onionsGivenToSara = 10 := by
  sorry

end onions_left_l187_187340


namespace curve_equation_represents_line_l187_187654

noncomputable def curve_is_line (x y : ℝ) : Prop :=
(x^2 + y^2 - 2*x) * (x + y - 3)^(1/2) = 0

theorem curve_equation_represents_line (x y : ℝ) :
curve_is_line x y ↔ (x + y = 3) :=
by sorry

end curve_equation_represents_line_l187_187654


namespace box_dimensions_l187_187128

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by
  sorry

end box_dimensions_l187_187128


namespace average_of_P_Q_R_is_correct_l187_187828

theorem average_of_P_Q_R_is_correct (P Q R : ℝ) 
  (h1 : 1001 * R - 3003 * P = 6006) 
  (h2 : 2002 * Q + 4004 * P = 8008) : 
  (P + Q + R)/3 = (2 * (P + 5))/3 :=
sorry

end average_of_P_Q_R_is_correct_l187_187828


namespace repeating_decimal_to_fraction_l187_187179

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + (6 / 10) / 9) : x = 11 / 30 :=
by
  sorry

end repeating_decimal_to_fraction_l187_187179


namespace no_solutions_l187_187155

theorem no_solutions (x y : ℤ) (h : 8 * x + 3 * y^2 = 5) : False :=
by
  sorry

end no_solutions_l187_187155


namespace largest_of_consecutive_even_integers_l187_187731

theorem largest_of_consecutive_even_integers (x : ℤ) (h : 25 * (x + 24) = 10000) : x + 48 = 424 :=
sorry

end largest_of_consecutive_even_integers_l187_187731


namespace total_cost_price_l187_187230

variables (C_table C_chair C_shelf : ℝ)

axiom h1 : 1.24 * C_table = 8091
axiom h2 : 1.18 * C_chair = 5346
axiom h3 : 1.30 * C_shelf = 11700

theorem total_cost_price :
  C_table + C_chair + C_shelf = 20055.51 :=
sorry

end total_cost_price_l187_187230


namespace bruce_three_times_son_in_six_years_l187_187628

-- Define the current ages of Bruce and his son
def bruce_age : ℕ := 36
def son_age : ℕ := 8

-- Define the statement to be proved
theorem bruce_three_times_son_in_six_years :
  ∃ (x : ℕ), x = 6 ∧ ∀ t, (t = x) → (bruce_age + t = 3 * (son_age + t)) :=
by
  sorry

end bruce_three_times_son_in_six_years_l187_187628


namespace maximum_obtuse_dihedral_angles_l187_187432

-- condition: define what a tetrahedron is and its properties
structure Tetrahedron :=
  (edges : Fin 6 → ℝ)   -- represents the 6 edges
  (dihedral_angles : Fin 6 → ℝ) -- represents the 6 dihedral angles

-- Define obtuse angle in degrees
def is_obtuse (angle : ℝ) : Prop := angle > 90 ∧ angle < 180

-- Theorem statement
theorem maximum_obtuse_dihedral_angles (T : Tetrahedron) : 
  (∃ count : ℕ, count = 3 ∧ (∀ i, is_obtuse (T.dihedral_angles i) → count <= 3)) := sorry

end maximum_obtuse_dihedral_angles_l187_187432


namespace prism_surface_area_is_14_l187_187470

-- Definition of the rectangular prism dimensions
def prism_length : ℕ := 3
def prism_width : ℕ := 1
def prism_height : ℕ := 1

-- Definition of the surface area of the rectangular prism
def surface_area (l w h : ℕ) : ℕ :=
  2 * (l * w + w * h + h * l)

-- Theorem statement: The surface area of the resulting prism is 14
theorem prism_surface_area_is_14 : surface_area prism_length prism_width prism_height = 14 :=
  sorry

end prism_surface_area_is_14_l187_187470


namespace denominator_of_first_fraction_l187_187101

theorem denominator_of_first_fraction (y x : ℝ) (h : y > 0) (h_eq : (9 * y) / x + (3 * y) / 10 = 0.75 * y) : x = 20 :=
by
  sorry

end denominator_of_first_fraction_l187_187101


namespace find_interval_for_a_l187_187398

-- Define the system of equations as a predicate
def system_of_equations (a x y z : ℝ) : Prop := 
  x + y + z = 0 ∧ x * y + y * z + a * z * x = 0

-- Define the condition that (0, 0, 0) is the only solution
def unique_solution (a : ℝ) : Prop :=
  ∀ x y z : ℝ, system_of_equations a x y z → x = 0 ∧ y = 0 ∧ z = 0

-- Rewrite the proof problem as a Lean statement
theorem find_interval_for_a :
  ∀ a : ℝ, unique_solution a ↔ 0 < a ∧ a < 4 :=
by
  sorry

end find_interval_for_a_l187_187398


namespace exist_pair_lcm_gcd_l187_187007

theorem exist_pair_lcm_gcd (a b: ℤ) : 
  ∃ a b : ℤ, Int.lcm a b - Int.gcd a b = 19 := 
sorry

end exist_pair_lcm_gcd_l187_187007


namespace expand_product_l187_187685

theorem expand_product (y : ℝ) : (y + 3) * (y + 7) = y^2 + 10 * y + 21 := by
  sorry

end expand_product_l187_187685


namespace dorchester_puppies_washed_l187_187936

theorem dorchester_puppies_washed
  (total_earnings : ℝ)
  (daily_pay : ℝ)
  (earnings_per_puppy : ℝ)
  (p : ℝ)
  (h1 : total_earnings = 76)
  (h2 : daily_pay = 40)
  (h3 : earnings_per_puppy = 2.25)
  (hp : (total_earnings - daily_pay) / earnings_per_puppy = p) :
  p = 16 := sorry

end dorchester_puppies_washed_l187_187936


namespace find_y_l187_187431

theorem find_y (x y : ℝ) : x - y = 8 ∧ x + y = 14 → y = 3 := by
  sorry

end find_y_l187_187431


namespace factor_polynomial_l187_187313

theorem factor_polynomial {x : ℝ} : 4 * x^3 - 16 * x = 4 * x * (x + 2) * (x - 2) := 
sorry

end factor_polynomial_l187_187313


namespace bacteria_elimination_l187_187475

theorem bacteria_elimination (d N : ℕ) (hN : N = 50 - 6 * (d - 1)) (hCondition : N ≤ 0) : d = 10 :=
by
  -- We can straightforwardly combine the given conditions and derive the required theorem.
  sorry

end bacteria_elimination_l187_187475


namespace range_of_a_l187_187312

theorem range_of_a (f : ℝ → ℝ) (h1 : ∀ x, f (x - 3) = f (3 - (x - 3))) (h2 : ∀ x, 0 ≤ x → f x = x^2 + 2 * x) :
  {a : ℝ | f (2 - a^2) > f a} = {a | -2 < a ∧ a < 1} :=
by
  sorry

end range_of_a_l187_187312


namespace soybeans_to_oil_l187_187670

theorem soybeans_to_oil 
    (kg_soybeans_to_tofu : ℝ)
    (kg_soybeans_to_oil : ℝ)
    (price_soybeans : ℝ)
    (price_tofu : ℝ)
    (price_oil : ℝ)
    (purchase_amount : ℝ)
    (sales_amount : ℝ)
    (amount_to_oil : ℝ)
    (used_soybeans_for_oil : ℝ) :
    kg_soybeans_to_tofu = 3 →
    kg_soybeans_to_oil = 6 →
    price_soybeans = 2 →
    price_tofu = 3 →
    price_oil = 15 →
    purchase_amount = 920 →
    sales_amount = 1800 →
    used_soybeans_for_oil = 360 →
    (6 * amount_to_oil) = 360 →
    15 * amount_to_oil + 3 * (460 - 6 * amount_to_oil) = 1800 :=
by sorry

end soybeans_to_oil_l187_187670


namespace martin_discounted_tickets_l187_187183

-- Definitions of the problem conditions
def total_tickets (F D : ℕ) := F + D = 10
def total_cost (F D : ℕ) := 2 * F + (16/10) * D = 184/10

-- Statement of the proof
theorem martin_discounted_tickets (F D : ℕ) (h1 : total_tickets F D) (h2 : total_cost F D) :
  D = 4 :=
sorry

end martin_discounted_tickets_l187_187183


namespace part_I_part_II_l187_187995

open Real

def f (x m n : ℝ) := abs (x - m) + abs (x + n)

theorem part_I (m n M : ℝ) (h1 : m + n = 9) (h2 : ∀ x : ℝ, f x m n ≥ M) : M ≤ 9 := 
sorry

theorem part_II (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9) : (a + b) * (a^3 + b^3) ≥ 81 := 
sorry

end part_I_part_II_l187_187995


namespace sum_of_roots_l187_187412

theorem sum_of_roots (x₁ x₂ b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + b = 0 → x = x₁ ∨ x = x₂) :
  x₁ + x₂ = 2 :=
sorry

end sum_of_roots_l187_187412


namespace find_number_l187_187994

theorem find_number :
  ∃ x : ℝ, (x - 1.9) * 1.5 + 32 / 2.5 = 20 ∧ x = 13.9 :=
by
  sorry

end find_number_l187_187994


namespace expected_value_8_sided_die_l187_187788

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end expected_value_8_sided_die_l187_187788


namespace count_isosceles_triangles_perimeter_25_l187_187758

theorem count_isosceles_triangles_perimeter_25 : 
  ∃ n : ℕ, (
    n = 6 ∧ 
    (∀ x b : ℕ, 
      2 * x + b = 25 → 
      b < 2 * x → 
      b > 0 →
      ∃ m : ℕ, 
        m = (x - 7) / 5
    ) 
  ) := sorry

end count_isosceles_triangles_perimeter_25_l187_187758


namespace inequality_abc_l187_187583

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^3) / (a^2 + a * b + b^2) + (b^3) / (b^2 + b * c + c^2) + (c^3) / (c^2 + c * a + a^2) ≥ (a + b + c) / 3 := 
by
    sorry

end inequality_abc_l187_187583


namespace find_original_price_l187_187423

-- Define the original price P
variable (P : ℝ)

-- Define the conditions as per the given problem
def revenue_equation (P : ℝ) : Prop :=
  820 = (10 * 0.60 * P) + (20 * 0.85 * P) + (18 * P)

-- Prove that the revenue equation implies P = 20
theorem find_original_price (P : ℝ) (h : revenue_equation P) : P = 20 :=
  by sorry

end find_original_price_l187_187423


namespace attraction_ticket_cost_for_parents_l187_187395

noncomputable def total_cost (children parents adults: ℕ) (entrance_cost child_attraction_cost adult_attraction_cost: ℕ) : ℕ :=
  (children + parents + adults) * entrance_cost + children * child_attraction_cost + adults * (adult_attraction_cost)

theorem attraction_ticket_cost_for_parents
  (children parents adults: ℕ) 
  (entrance_cost child_attraction_cost total_cost_of_family: ℕ) 
  (h_children: children = 4)
  (h_parents: parents = 2)
  (h_adults: adults = 1)
  (h_entrance_cost: entrance_cost = 5)
  (h_child_attraction_cost: child_attraction_cost = 2)
  (h_total_cost_of_family: total_cost_of_family = 55)
  : (total_cost children parents adults entrance_cost child_attraction_cost 4 / 3) = total_cost_of_family - (children + parents + adults) * entrance_cost - children * child_attraction_cost := 
sorry

end attraction_ticket_cost_for_parents_l187_187395


namespace willy_days_worked_and_missed_l187_187476

theorem willy_days_worked_and_missed:
  ∃ (x : ℚ), 8 * x = 10 * (30 - x) ∧ x = 50/3 ∧ (30 - x) = 40/3 :=
by
  sorry

end willy_days_worked_and_missed_l187_187476


namespace value_of_x_squared_minus_y_squared_l187_187783

theorem value_of_x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x_squared_minus_y_squared_l187_187783


namespace value_of_x_l187_187961

theorem value_of_x :
  ∃ x : ℝ, x = 1.13 * 80 :=
sorry

end value_of_x_l187_187961


namespace kate_change_is_correct_l187_187616

-- Define prices of items
def gum_price : ℝ := 0.89
def chocolate_price : ℝ := 1.25
def chips_price : ℝ := 2.49

-- Define sales tax rate
def tax_rate : ℝ := 0.06

-- Define the total money Kate gave to the clerk
def payment : ℝ := 10.00

-- Define total cost of items before tax
def total_before_tax := gum_price + chocolate_price + chips_price

-- Define the sales tax
def sales_tax := tax_rate * total_before_tax

-- Define the correct answer for total cost
def total_cost := total_before_tax + sales_tax

-- Define the correct amount of change Kate should get back
def change := payment - total_cost

theorem kate_change_is_correct : abs (change - 5.09) < 0.01 :=
by
  sorry

end kate_change_is_correct_l187_187616


namespace evaluate_g_of_h_l187_187842

def g (x : ℝ) : ℝ := 3 * x^2 - 4

def h (x : ℝ) : ℝ := 5 * x^3 + 2

theorem evaluate_g_of_h : g (h (-2)) = 4328 := 
by
  sorry

end evaluate_g_of_h_l187_187842


namespace teams_B_and_C_worked_together_days_l187_187608

def workload_project_B := 5/4
def time_team_A_project_A := 20
def time_team_B_project_A := 24
def time_team_C_project_A := 30

def equation1 (x y : ℕ) : Prop := 
  3 * x + 5 * y = 60

def equation2 (x y : ℕ) : Prop := 
  9 * x + 5 * y = 150

theorem teams_B_and_C_worked_together_days (x : ℕ) (y : ℕ) :
  equation1 x y ∧ equation2 x y → x = 15 := 
by 
  sorry

end teams_B_and_C_worked_together_days_l187_187608


namespace eval_expression_l187_187371

theorem eval_expression : (500 * 500) - (499 * 501) = 1 := by
  sorry

end eval_expression_l187_187371


namespace part_I_part_II_l187_187049

theorem part_I : 
  (∀ x : ℝ, |x - (2 : ℝ)| ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) :=
  sorry

theorem part_II :
  (∀ a b c : ℝ, a - 2 * b + c = 2 → a^2 + b^2 + c^2 ≥ 2 / 3) :=
  sorry

end part_I_part_II_l187_187049


namespace eight_distinct_solutions_l187_187655

noncomputable def f (x : ℝ) : ℝ := x^2 - 2

theorem eight_distinct_solutions : 
  ∃ S : Finset ℝ, S.card = 8 ∧ ∀ x ∈ S, f (f (f x)) = x :=
sorry

end eight_distinct_solutions_l187_187655


namespace average_infection_rate_l187_187360

theorem average_infection_rate (x : ℕ) : 
  1 + x + x * (1 + x) = 81 :=
sorry

end average_infection_rate_l187_187360


namespace sum_mod_9_is_6_l187_187118

noncomputable def sum_modulo_9 : ℤ :=
  1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999

theorem sum_mod_9_is_6 : sum_modulo_9 % 9 = 6 := 
  by
    sorry

end sum_mod_9_is_6_l187_187118


namespace curve_transformation_l187_187488

theorem curve_transformation :
  (∀ (x y : ℝ), 2 * (5 * x)^2 + 8 * (3 * y)^2 = 1) → (∀ (x y : ℝ), 50 * x^2 + 72 * y^2 = 1) :=
by
  intros h x y
  have h1 : 2 * (5 * x)^2 + 8 * (3 * y)^2 = 1 := h x y
  sorry

end curve_transformation_l187_187488


namespace ellipse_C_properties_l187_187001

open Real

noncomputable def ellipse_eq (b : ℝ) : Prop :=
  (∀ (x y : ℝ), (x = 1 ∧ y = sqrt 3 / 2) → (x^2 / 4 + y^2 / b^2 = 1))

theorem ellipse_C_properties : 
  (∀ (C : ℝ → ℝ → Prop), 
    (C 0 0) ∧ 
    (∀ x y, C x y → (x = 0 ↔ y = 0)) ∧ 
    (∀ x, C x 0) ∧ 
    (∃ x y, C x y ∧ x = 1 ∧ y = sqrt 3 / 2) →
    (∃ b, b > 0 ∧ b^2 = 1 ∧ ellipse_eq b)) ∧
  (∀ P A B : ℝ × ℝ, 
    (P.1 = P.1 ∧ P.1 ≠ 0 ∧ P.2 = 0 ∧ -2 ≤ P.1 ∧ P.1 ≤ 2) →
    (A.2 = 1/2 * (A.1 - P.1) ∧ B.2 = 1/2 * (B.1 - P.1)) →
    ((P.1 - A.1)^2 + A.2^2 + (P.1 - B.1)^2 + B.2^2 = 5)) :=
by sorry

end ellipse_C_properties_l187_187001


namespace right_triangle_of_condition_l187_187075

theorem right_triangle_of_condition
  (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h_trig : Real.sin γ - Real.cos α = Real.cos β) :
  (α = 90) ∨ (β = 90) :=
sorry

end right_triangle_of_condition_l187_187075


namespace sum_first_2017_terms_l187_187526

-- Given sequence definition
def a : ℕ → ℕ
| 0       => 0 -- a_0 (dummy term for 1-based index convenience)
| 1       => 1
| (n + 2) => 3 * 2^(n) - a (n + 1)

-- Sum of the first n terms of the sequence {a_n}
def S : ℕ → ℕ
| 0       => 0
| (n + 1) => S n + a (n + 1)

-- Theorem to prove
theorem sum_first_2017_terms : S 2017 = 2^2017 - 1 :=
sorry

end sum_first_2017_terms_l187_187526


namespace number_div_addition_l187_187410

-- Define the given conditions
def original_number (q d r : ℕ) : ℕ := (q * d) + r

theorem number_div_addition (q d r a b : ℕ) (h1 : d = 6) (h2 : q = 124) (h3 : r = 4) (h4 : a = 24) (h5 : b = 8) :
  ((original_number q d r + a) / b : ℚ) = 96.5 :=
by 
  sorry

end number_div_addition_l187_187410


namespace growth_rate_correct_max_avg_visitors_correct_l187_187984

-- Define the conditions from part 1
def visitors_march : ℕ := 80000
def visitors_may : ℕ := 125000

-- Define the monthly average growth rate
def monthly_avg_growth_rate (x : ℝ) : Prop :=
(1 + x)^2 = (visitors_may / visitors_march : ℝ)

-- Define the condition for June
def visitors_june_1_10 : ℕ := 66250
def max_avg_visitors_per_day (y : ℝ) : Prop :=
6.625 + 20 * y ≤ 15.625

-- Prove the monthly growth rate
theorem growth_rate_correct : ∃ x : ℝ, monthly_avg_growth_rate x ∧ x = 0.25 := sorry

-- Prove the max average visitors per day in June
theorem max_avg_visitors_correct : ∃ y : ℝ, max_avg_visitors_per_day y ∧ y = 0.45 := sorry

end growth_rate_correct_max_avg_visitors_correct_l187_187984


namespace ratio_sum_of_square_lengths_equals_68_l187_187567

theorem ratio_sum_of_square_lengths_equals_68 (a b c : ℕ) 
  (h1 : (∃ (r : ℝ), r = 50 / 98) → a = 5 ∧ b = 14 ∧ c = 49) :
  a + b + c = 68 :=
by
  sorry -- Proof is not required

end ratio_sum_of_square_lengths_equals_68_l187_187567


namespace rectangle_difference_l187_187221

theorem rectangle_difference (A d x y : ℝ) (h1 : x * y = A) (h2 : x^2 + y^2 = d^2) :
  x - y = 2 * Real.sqrt A := 
sorry

end rectangle_difference_l187_187221


namespace sequence_relation_l187_187848

theorem sequence_relation
  (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 :=
sorry

end sequence_relation_l187_187848


namespace maximum_integer_value_of_a_l187_187254

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (2 - a) * x - a * Real.log x

theorem maximum_integer_value_of_a (a : ℝ) (h : ∀ x ≥ 1, f x a > 0) : a ≤ 2 :=
sorry

end maximum_integer_value_of_a_l187_187254


namespace simplify_polynomial_l187_187054

theorem simplify_polynomial : 
  (5 - 3 * x - 7 * x^2 + 3 + 12 * x - 9 * x^2 - 8 + 15 * x + 21 * x^2) = (5 * x^2 + 24 * x) :=
by 
  sorry

end simplify_polynomial_l187_187054


namespace fraction_option_C_l187_187512

def is_fraction (expr : String) : Prop := 
  expr = "fraction"

def option_C_fraction (x : ℝ) : Prop :=
  ∃ (numerator : ℝ), ∃ (denominator : ℝ), 
  numerator = 2 ∧ denominator = x + 3

theorem fraction_option_C (x : ℝ) (h : x ≠ -3) :
  is_fraction "fraction" ↔ option_C_fraction x :=
by 
  sorry

end fraction_option_C_l187_187512


namespace find_radius_l187_187669

theorem find_radius (AB EO : ℝ) (AE BE : ℝ) (h1 : AB = AE + BE) (h2 : AE = 2 * BE) (h3 : EO = 7) :
  ∃ R : ℝ, R = 11 := by
  sorry

end find_radius_l187_187669


namespace number_of_valid_n_l187_187072

-- The definition for determining the number of positive integers n ≤ 2000 that can be represented as
-- floor(x) + floor(4x) + floor(5x) = n for some real number x.

noncomputable def count_valid_n : ℕ :=
  (200 : ℕ) * 3 + (200 : ℕ) * 2 + 1 + 1

theorem number_of_valid_n : count_valid_n = 802 :=
  sorry

end number_of_valid_n_l187_187072


namespace roots_situation_depends_on_k_l187_187745

theorem roots_situation_depends_on_k (k : ℝ) : 
  let a := 1
  let b := -3
  let c := 2 - k
  let Δ := b^2 - 4 * a * c
  (Δ > 0) ∨ (Δ = 0) ∨ (Δ < 0) :=
by
  intros
  sorry

end roots_situation_depends_on_k_l187_187745


namespace reciprocal_power_l187_187649

theorem reciprocal_power (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 :=
by sorry

end reciprocal_power_l187_187649


namespace find_number_multiplied_l187_187448

theorem find_number_multiplied (m : ℕ) (h : 9999 * m = 325027405) : m = 32505 :=
by {
  sorry
}

end find_number_multiplied_l187_187448


namespace perfect_square_expression_l187_187973

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, m^2 = (2 * l - n - k) * (2 * l - n + k) / 2 :=
by 
  sorry

end perfect_square_expression_l187_187973


namespace zero_in_interval_l187_187055

open Real

noncomputable def f (x : ℝ) : ℝ := log x + x - 3

theorem zero_in_interval (a b : ℕ) (h1 : b - a = 1) (h2 : 1 ≤ a) (h3 : 1 ≤ b) 
  (h4 : f a < 0) (h5 : 0 < f b) : a + b = 5 :=
sorry

end zero_in_interval_l187_187055


namespace baker_earnings_l187_187279

-- Define the number of cakes and pies sold
def cakes_sold := 453
def pies_sold := 126

-- Define the prices per cake and pie
def price_per_cake := 12
def price_per_pie := 7

-- Calculate the total earnings
def total_earnings : ℕ := (cakes_sold * price_per_cake) + (pies_sold * price_per_pie)

-- Theorem stating the baker's earnings
theorem baker_earnings : total_earnings = 6318 := by
  unfold total_earnings cakes_sold pies_sold price_per_cake price_per_pie
  sorry

end baker_earnings_l187_187279


namespace mutual_acquainted_or_unacquainted_l187_187525

theorem mutual_acquainted_or_unacquainted :
  ∀ (G : SimpleGraph (Fin 6)), 
  ∃ (V : Finset (Fin 6)), V.card = 3 ∧ ((∀ (u v : Fin 6), u ∈ V → v ∈ V → G.Adj u v) ∨ (∀ (u v : Fin 6), u ∈ V → v ∈ V → ¬G.Adj u v)) :=
by
  sorry

end mutual_acquainted_or_unacquainted_l187_187525


namespace neutral_equilibrium_l187_187869

noncomputable def equilibrium_ratio (r h : ℝ) : ℝ := r / h

theorem neutral_equilibrium (r h : ℝ) (k : ℝ) : (equilibrium_ratio r h = k) → (k = Real.sqrt 2) :=
by
  intro h1
  have h1' : (r / h = k) := h1
  sorry

end neutral_equilibrium_l187_187869


namespace product_of_powers_l187_187246

theorem product_of_powers :
  ((-1 : Int)^3) * ((-2 : Int)^2) = -4 := by
  sorry

end product_of_powers_l187_187246


namespace difference_in_pennies_l187_187193

theorem difference_in_pennies (p : ℤ) : 
  let alice_nickels := 3 * p + 2
  let bob_nickels := 2 * p + 6
  let difference_nickels := alice_nickels - bob_nickels
  let difference_in_pennies := difference_nickels * 5
  difference_in_pennies = 5 * p - 20 :=
by
  sorry

end difference_in_pennies_l187_187193


namespace find_number_l187_187233

theorem find_number (x : ℝ) (h : x = 12) : ( ( 17.28 / x ) / ( 3.6 * 0.2 ) ) = 2 := 
by
  -- Proof will be here
  sorry

end find_number_l187_187233


namespace find_p_l187_187527

-- Define the coordinates of the points
structure Point where
  x : Real
  y : Real

def Q := Point.mk 0 15
def A := Point.mk 3 15
def B := Point.mk 15 0
def O := Point.mk 0 0
def C (p : Real) := Point.mk 0 p

-- Given the area of triangle ABC and the coordinates of Q, A, B, O, and C, prove that p = 12.75
theorem find_p (p : Real) (h_area_ABC : 36 = 36) (h_Q : Q = Point.mk 0 15)
                (h_A : A = Point.mk 3 15) (h_B : B = Point.mk 15 0) 
                (h_O : O = Point.mk 0 0) : p = 12.75 := 
sorry

end find_p_l187_187527


namespace simplify_fraction_l187_187003

theorem simplify_fraction : (75 : ℚ) / (100 : ℚ) = (3 : ℚ) / (4 : ℚ) :=
by
  sorry

end simplify_fraction_l187_187003


namespace hyperbola_range_m_l187_187187

theorem hyperbola_range_m (m : ℝ) : (m - 2) * (m - 6) < 0 ↔ 2 < m ∧ m < 6 :=
by sorry

end hyperbola_range_m_l187_187187


namespace club_boys_count_l187_187587

theorem club_boys_count (B G : ℕ) (h1 : B + G = 30) (h2 : (1 / 3 : ℝ) * G + B = 18) : B = 12 :=
by
  -- We would proceed with the steps here, but add 'sorry' to indicate incomplete proof
  sorry

end club_boys_count_l187_187587


namespace sequence_properties_l187_187010

theorem sequence_properties (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 1 = 1 →
  (∀ n : ℕ, a (n + 1) + a n = 4 * n) →
  (∀ n : ℕ, a n = 2 * n - 1) ∧ (a 2023 = 4045) :=
by
  sorry

end sequence_properties_l187_187010


namespace jennifer_fifth_score_l187_187473

theorem jennifer_fifth_score :
  ∀ (x : ℝ), (85 + 90 + 87 + 92 + x) / 5 = 89 → x = 91 :=
by
  sorry

end jennifer_fifth_score_l187_187473


namespace gain_percentage_l187_187985

theorem gain_percentage (SP1 SP2 CP: ℝ) (h1 : SP1 = 102) (h2 : SP2 = 144) (h3 : SP1 = CP - 0.15 * CP) :
  ((SP2 - CP) / CP) * 100 = 20 := by
sorry

end gain_percentage_l187_187985


namespace zacks_friends_l187_187812

theorem zacks_friends (initial_marbles : ℕ) (marbles_kept : ℕ) (marbles_per_friend : ℕ) 
  (h_initial : initial_marbles = 65) (h_kept : marbles_kept = 5) 
  (h_per_friend : marbles_per_friend = 20) : (initial_marbles - marbles_kept) / marbles_per_friend = 3 :=
by
  sorry

end zacks_friends_l187_187812


namespace jenna_practice_minutes_l187_187424

theorem jenna_practice_minutes :
  ∀ (practice_6_days practice_2_days target_total target_average: ℕ),
    practice_6_days = 6 * 80 →
    practice_2_days = 2 * 105 →
    target_average = 100 →
    target_total = 9 * target_average →
  ∃ practice_9th_day, (practice_6_days + practice_2_days + practice_9th_day = target_total) ∧ practice_9th_day = 210 :=
by sorry

end jenna_practice_minutes_l187_187424


namespace men_in_second_group_l187_187433

theorem men_in_second_group (M : ℕ) (h1 : 36 * 18 = M * 24) : M = 27 :=
by {
  sorry
}

end men_in_second_group_l187_187433


namespace probability_at_least_one_girl_l187_187753

theorem probability_at_least_one_girl (boys girls : ℕ) (total : ℕ) (choose_two : ℕ) : 
  boys = 3 → girls = 2 → total = boys + girls → choose_two = 2 → 
  1 - (Nat.choose boys choose_two) / (Nat.choose total choose_two) = 7 / 10 :=
by
  sorry

end probability_at_least_one_girl_l187_187753


namespace JacobProof_l187_187491

def JacobLadders : Prop :=
  let costPerRung : ℤ := 2
  let costPer50RungLadder : ℤ := 50 * costPerRung
  let num50RungLadders : ℤ := 10
  let totalPayment : ℤ := 3400
  let cost1 : ℤ := num50RungLadders * costPer50RungLadder
  let remainingAmount : ℤ := totalPayment - cost1
  let numRungs20Ladders : ℤ := remainingAmount / costPerRung
  numRungs20Ladders = 1200

theorem JacobProof : JacobLadders := by
  sorry

end JacobProof_l187_187491


namespace large_diagonal_proof_l187_187268

variable (a b : ℝ) (α : ℝ)
variable (h₁ : a < b)
variable (h₂ : 1 < a) -- arbitrary positive scalar to make obtuse properties hold

noncomputable def large_diagonal_length : ℝ :=
  Real.sqrt (a^2 + b^2 + 2 * b * (Real.cos α * Real.sqrt (a^2 - b^2 * Real.sin α^2) + b * Real.sin α^2))

theorem large_diagonal_proof
  (h₃ : 90 < α + Real.arcsin (b * Real.sin α / a)) :
  large_diagonal_length a b α = Real.sqrt (a^2 + b^2 + 2 * b * (Real.cos α * Real.sqrt (a^2 - b^2 * Real.sin α^2) + b * Real.sin α^2)) :=
sorry

end large_diagonal_proof_l187_187268


namespace total_people_present_l187_187218

/-- This definition encapsulates all the given conditions: 
    The number of parents, pupils, staff members, and performers. -/
def num_parents : ℕ := 105
def num_pupils : ℕ := 698
def num_staff : ℕ := 45
def num_performers : ℕ := 32

/-- Theorem stating that the total number of people present in the program is 880 
    given the stated conditions. -/
theorem total_people_present : num_parents + num_pupils + num_staff + num_performers = 880 :=
by 
  /- We can use Lean's capabilities to verify the arithmetics. -/
  sorry

end total_people_present_l187_187218


namespace range_estimate_of_expression_l187_187799

theorem range_estimate_of_expression : 
  6 < (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 ∧ 
       (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 < 7 :=
by
  sorry

end range_estimate_of_expression_l187_187799


namespace negation_of_exists_proposition_l187_187804

theorem negation_of_exists_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) → (∀ n : ℕ, n^2 ≤ 2^n) := 
by 
  sorry

end negation_of_exists_proposition_l187_187804


namespace inequalities_not_hold_range_a_l187_187391

theorem inequalities_not_hold_range_a (a : ℝ) :
  (¬ ∀ x : ℝ, x^2 - a * x + 1 ≤ 0) ∧ (¬ ∀ x : ℝ, a * x^2 + x - 1 > 0) ↔ (-2 < a ∧ a ≤ -1 / 4) :=
by
  sorry

end inequalities_not_hold_range_a_l187_187391


namespace dealer_cash_discount_percentage_l187_187872

-- Definitions of the given conditions
variable (C : ℝ) (n m : ℕ) (profit_p list_ratio : ℝ)
variable (h_n : n = 25) (h_m : m = 20) (h_profit : profit_p = 1.36) (h_list_ratio : list_ratio = 2)

-- The statement we need to prove
theorem dealer_cash_discount_percentage 
  (h_eff_selling_price : (m : ℝ) / n * C = profit_p * C)
  : ((list_ratio * C - (m / n * C)) / (list_ratio * C) * 100 = 60) :=
by
  sorry

end dealer_cash_discount_percentage_l187_187872


namespace proof_C_D_values_l187_187096

-- Given the conditions
def denominator_factorization (x : ℝ) : Prop :=
  3 * x ^ 2 - x - 14 = (3 * x + 7) * (x - 2)

def fraction_equality (x : ℝ) (C D : ℝ) : Prop :=
  (3 * x ^ 2 + 7 * x - 20) / (3 * x ^ 2 - x - 14) =
  C / (x - 2) + D / (3 * x + 7)

-- The values to be proven
def values_C_D : Prop :=
  ∃ C D : ℝ, C = -14 / 13 ∧ D = 81 / 13 ∧ ∀ x : ℝ, (denominator_factorization x → fraction_equality x C D)

theorem proof_C_D_values : values_C_D :=
sorry

end proof_C_D_values_l187_187096


namespace find_n_l187_187109

theorem find_n (n : ℕ) : (Nat.lcm n 10 = 36) ∧ (Nat.gcd n 10 = 5) → n = 18 :=
by
  -- The proof will be provided here
  sorry

end find_n_l187_187109


namespace ascending_order_l187_187720

theorem ascending_order (a b c d : ℝ) (h1 : a = -6) (h2 : b = 0) (h3 : c = Real.sqrt 5) (h4 : d = Real.pi) :
  a < b ∧ b < c ∧ c < d :=
by
  sorry

end ascending_order_l187_187720


namespace speed_of_man_in_still_water_l187_187437

theorem speed_of_man_in_still_water
  (v_m v_s : ℝ)
  (h1 : v_m + v_s = 4)
  (h2 : v_m - v_s = 2) :
  v_m = 3 := 
by sorry

end speed_of_man_in_still_water_l187_187437


namespace comparison_among_abc_l187_187162

noncomputable def a : ℝ := 2^(1/5)
noncomputable def b : ℝ := (1/5)^2
noncomputable def c : ℝ := Real.log (1/5) / Real.log 2

theorem comparison_among_abc : a > b ∧ b > c :=
by
  -- Assume the necessary conditions and the conclusion.
  sorry

end comparison_among_abc_l187_187162


namespace spring_stretch_150N_l187_187037

-- Definitions for the conditions
def spring_stretch (weight : ℕ) : ℕ :=
  if weight = 100 then 20 else sorry

-- The theorem to prove
theorem spring_stretch_150N : spring_stretch 150 = 30 := by
  sorry

end spring_stretch_150N_l187_187037


namespace three_digit_numbers_last_three_digits_of_square_l187_187366

theorem three_digit_numbers_last_three_digits_of_square (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (n^2 % 1000) = n ↔ n = 376 ∨ n = 625 := 
sorry

end three_digit_numbers_last_three_digits_of_square_l187_187366


namespace water_drain_rate_l187_187734

theorem water_drain_rate
  (total_volume : ℕ)
  (total_time : ℕ)
  (H1 : total_volume = 300)
  (H2 : total_time = 25) :
  total_volume / total_time = 12 := 
by
  sorry

end water_drain_rate_l187_187734


namespace flight_cost_l187_187870

theorem flight_cost (ground_school_cost flight_portion_addition total_cost flight_portion_cost: ℕ) 
  (h₁ : ground_school_cost = 325)
  (h₂ : flight_portion_addition = 625)
  (h₃ : flight_portion_cost = ground_school_cost + flight_portion_addition):
  flight_portion_cost = 950 :=
by
  -- placeholder for proofs
  sorry

end flight_cost_l187_187870


namespace ticket_price_l187_187571

theorem ticket_price (Olivia_money : ℕ) (Nigel_money : ℕ) (left_money : ℕ) (total_tickets : ℕ)
  (h1 : Olivia_money = 112)
  (h2 : Nigel_money = 139)
  (h3 : left_money = 83)
  (h4 : total_tickets = 6) :
  (Olivia_money + Nigel_money - left_money) / total_tickets = 28 :=
by
  sorry

end ticket_price_l187_187571


namespace rectangle_ratio_l187_187481

theorem rectangle_ratio (s : ℝ) (h : s > 0) :
    let large_square_side := 3 * s
    let rectangle_length := 3 * s
    let rectangle_width := 2 * s
    rectangle_length / rectangle_width = 3 / 2 := by
  sorry

end rectangle_ratio_l187_187481


namespace total_new_people_last_year_l187_187224

-- Define the number of new people born and the number of people immigrated
def new_people_born : ℕ := 90171
def people_immigrated : ℕ := 16320

-- Prove that the total number of new people is 106491
theorem total_new_people_last_year : new_people_born + people_immigrated = 106491 := by
  sorry

end total_new_people_last_year_l187_187224


namespace jamesons_sword_length_l187_187894

theorem jamesons_sword_length (c j j' : ℕ) (hC: c = 15) 
  (hJ: j = c + 23) (hJJ: j' = j - 5) : 
  j' = 2 * c + 3 := by 
  sorry

end jamesons_sword_length_l187_187894


namespace a_is_perfect_square_l187_187740

theorem a_is_perfect_square (a b : ℕ) (h : ab ∣ (a^2 + b^2 + a)) : (∃ k : ℕ, a = k^2) :=
sorry

end a_is_perfect_square_l187_187740


namespace geom_sequence_eq_l187_187474

theorem geom_sequence_eq :
  ∀ {a : ℕ → ℝ} {q : ℝ}, (∀ n, a (n + 1) = q * a n) →
  a 1 + a 2 + a 3 + a 4 + a 5 = 3 →
  a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 = 15 →
  a 1 - a 2 + a 3 - a 4 + a 5 = 5 :=
by
  intro a q hgeom hsum hsum_sq
  sorry

end geom_sequence_eq_l187_187474


namespace iced_coffee_cost_is_2_l187_187027

def weekly_latte_cost := 4 * 5
def annual_latte_cost := weekly_latte_cost * 52
def weekly_iced_coffee_cost (x : ℝ) := x * 3
def annual_iced_coffee_cost (x : ℝ) := weekly_iced_coffee_cost x * 52
def total_annual_coffee_cost (x : ℝ) := annual_latte_cost + annual_iced_coffee_cost x
def reduced_spending_goal (x : ℝ) := 0.75 * total_annual_coffee_cost x
def saved_amount := 338

theorem iced_coffee_cost_is_2 :
  ∃ x : ℝ, (total_annual_coffee_cost x - reduced_spending_goal x = saved_amount) → x = 2 :=
by
  sorry

end iced_coffee_cost_is_2_l187_187027


namespace meaningful_sqrt_domain_l187_187971

theorem meaningful_sqrt_domain (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) :=
by
  sorry

end meaningful_sqrt_domain_l187_187971


namespace powers_of_two_diff_div_by_1987_l187_187341

theorem powers_of_two_diff_div_by_1987 :
  ∃ a b : ℕ, a > b ∧ 1987 ∣ (2^a - 2^b) :=
by sorry

end powers_of_two_diff_div_by_1987_l187_187341


namespace fuel_a_added_l187_187281

theorem fuel_a_added (capacity : ℝ) (ethanolA : ℝ) (ethanolB : ℝ) (total_ethanol : ℝ) (x : ℝ) : 
  capacity = 200 ∧ ethanolA = 0.12 ∧ ethanolB = 0.16 ∧ total_ethanol = 28 →
  0.12 * x + 0.16 * (200 - x) = 28 → x = 100 :=
sorry

end fuel_a_added_l187_187281


namespace division_scaling_l187_187821

theorem division_scaling (h : 204 / 12.75 = 16) : 2.04 / 1.275 = 16 :=
sorry

end division_scaling_l187_187821


namespace circle_area_eq_pi_div_4_l187_187850

theorem circle_area_eq_pi_div_4 :
  ∀ (x y : ℝ), 3*x^2 + 3*y^2 - 9*x + 12*y + 27 = 0 -> (π * (1 / 2)^2 = π / 4) :=
by
  sorry

end circle_area_eq_pi_div_4_l187_187850


namespace sequence_formula_l187_187596

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (diff : ∀ n, a (n + 1) - a n = 3^n) :
  ∀ n, a n = (3^n - 1) / 2 :=
by
  sorry

end sequence_formula_l187_187596


namespace remaining_customers_l187_187569

theorem remaining_customers (initial: ℕ) (left: ℕ) (remaining: ℕ) 
  (h1: initial = 14) (h2: left = 11) : remaining = initial - left → remaining = 3 :=
by {
  sorry
}

end remaining_customers_l187_187569


namespace jenny_change_l187_187062

/-!
## Problem statement

Jenny is printing 7 copies of her 25-page essay. It costs $0.10 to print one page.
She also buys 7 pens, each costing $1.50. If she pays with $40, calculate the change she should get.
-/

def cost_per_page : ℝ := 0.10
def pages_per_copy : ℕ := 25
def num_copies : ℕ := 7
def cost_per_pen : ℝ := 1.50
def num_pens : ℕ := 7
def amount_paid : ℝ := 40.0

def total_pages : ℕ := num_copies * pages_per_copy

def cost_printing : ℝ := total_pages * cost_per_page
def cost_pens : ℝ := num_pens * cost_per_pen

def total_cost : ℝ := cost_printing + cost_pens

theorem jenny_change : amount_paid - total_cost = 12 := by
  -- proof here
  sorry

end jenny_change_l187_187062


namespace complex_purely_imaginary_l187_187566

theorem complex_purely_imaginary (a : ℂ) (h1 : a^2 - 3 * a + 2 = 0) (h2 : a - 1 ≠ 0) : a = 2 :=
sorry

end complex_purely_imaginary_l187_187566


namespace greatest_three_digit_multiple_of_17_l187_187724

theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ≤ 999 ∧ n ≥ 100 ∧ (∃ k : ℕ, n = 17 * k) ∧ 
  (∀ m : ℕ, m ≤ 999 → m ≥ 100 → (∃ k : ℕ, m = 17 * k) → m ≤ n) ∧ n = 986 := 
sorry

end greatest_three_digit_multiple_of_17_l187_187724


namespace instantaneous_velocity_at_t_2_l187_187794

theorem instantaneous_velocity_at_t_2 
  (t : ℝ) (x1 y1 x2 y2: ℝ) : 
  (t = 2) → 
  (x1 = 0) → (y1 = 4) → 
  (x2 = 12) → (y2 = -2) → 
  ((y2 - y1) / (x2 - x1) = -1 / 2) := 
by 
  intros ht hx1 hy1 hx2 hy2
  sorry

end instantaneous_velocity_at_t_2_l187_187794


namespace problem_statement_l187_187332

-- Define that the function f is even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define that the function f satisfies f(x) = f(2 - x)
def satisfies_symmetry (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

-- Define that the function f is decreasing on a given interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Define that the function f is increasing on a given interval
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Given hypotheses and the theorem to prove. We use two statements for clarity.
theorem problem_statement (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_symmetry : satisfies_symmetry f) 
  (h_decreasing_1_2 : is_decreasing_on f 1 2) : 
  is_increasing_on f (-2) (-1) ∧ is_decreasing_on f 3 4 := 
by 
  sorry

end problem_statement_l187_187332


namespace find_dividend_l187_187272

theorem find_dividend (divisor : ℕ) (partial_quotient : ℕ) (dividend : ℕ) 
                       (h_divisor : divisor = 12)
                       (h_partial_quotient : partial_quotient = 909809) 
                       (h_calculation : dividend = divisor * partial_quotient) : 
                       dividend = 10917708 :=
by
  rw [h_divisor, h_partial_quotient] at h_calculation
  exact h_calculation


end find_dividend_l187_187272


namespace find_difference_l187_187146

variable (k1 k2 t1 t2 : ℝ)

theorem find_difference (h1 : t1 = 5 / 9 * (k1 - 32))
                        (h2 : t2 = 5 / 9 * (k2 - 32))
                        (h3 : t1 = 105)
                        (h4 : t2 = 80) :
  k1 - k2 = 45 :=
by
  sorry

end find_difference_l187_187146


namespace sum_mod_17_l187_187257

/--
Given the sum of the numbers 82, 83, 84, 85, 86, 87, 88, and 89, and the divisor 17,
prove that the remainder when dividing this sum by 17 is 11.
-/
theorem sum_mod_17 : (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 11 :=
by
  sorry

end sum_mod_17_l187_187257


namespace real_part_fraction_l187_187311

theorem real_part_fraction {i : ℂ} (h : i^2 = -1) : (
  let numerator := 1 - i
  let denominator := (1 + i) ^ 2
  let fraction := numerator / denominator
  let real_part := (fraction.re)
  real_part
) = -1/2 := sorry

end real_part_fraction_l187_187311


namespace cole_drive_time_l187_187381

theorem cole_drive_time (D : ℝ) (T_work T_home : ℝ) 
  (h1 : T_work = D / 75) 
  (h2 : T_home = D / 105)
  (h3 : T_work + T_home = 4) : 
  T_work * 60 = 140 := 
by sorry

end cole_drive_time_l187_187381


namespace quadratic_inequality_solution_l187_187114

theorem quadratic_inequality_solution :
  {x : ℝ | -3 < x ∧ x < 1} = {x : ℝ | x * (x + 2) < 3} :=
by
  sorry

end quadratic_inequality_solution_l187_187114


namespace sequence_is_increasing_l187_187859

theorem sequence_is_increasing (a : ℕ → ℝ) (h : ∀ n : ℕ, a (n + 1) - a n = 2) : ∀ n : ℕ, a (n + 1) > a n :=
by
  intro n
  have h2 : a (n + 1) - a n = 2 := h n
  linarith

end sequence_is_increasing_l187_187859


namespace geometric_sequence_common_ratio_l187_187013

theorem geometric_sequence_common_ratio (a : ℕ → ℕ) (q : ℕ) (h2 : a 2 = 8) (h5 : a 5 = 64)
  (h_geom : ∀ n, a (n+1) = a n * q) : q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l187_187013


namespace minimum_value_f_range_of_m_l187_187098

noncomputable def f (x m : ℝ) := x - m * Real.log x - (m - 1) / x
noncomputable def g (x : ℝ) := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

theorem minimum_value_f (m : ℝ) (x : ℝ) (hx : 1 ≤ x ∧ x ≤ Real.exp 1) : 
  if m ≤ 2 then f x m = 2 - m 
  else if m ≥ Real.exp 1 + 1 then f x m = Real.exp 1 - m - (m - 1) / Real.exp 1 
  else f x m = m - 2 - m * Real.log (m - 1) :=
sorry

theorem range_of_m (m : ℝ) :
  (m ≤ 2 ∧ ∀ x2 ∈ [-2, 0], ∃ x1 ∈ [Real.exp 1, Real.exp 2], f x1 m ≤ g x2) ↔
  (m ∈ [ (Real.exp 2 - Real.exp 1 + 1) / (Real.exp 1 + 1), 2 ]) :=
sorry

end minimum_value_f_range_of_m_l187_187098


namespace triangle_third_side_length_l187_187913

theorem triangle_third_side_length (x: ℕ) (h1: x % 2 = 0) (h2: 2 + 14 > x) (h3: 14 - 2 < x) : x = 14 :=
by 
  sorry

end triangle_third_side_length_l187_187913


namespace gift_bag_combinations_l187_187242

theorem gift_bag_combinations (giftBags tissuePapers tags : ℕ) (h1 : giftBags = 10) (h2 : tissuePapers = 4) (h3 : tags = 5) : 
  giftBags * tissuePapers * tags = 200 := 
by 
  sorry

end gift_bag_combinations_l187_187242


namespace ab_value_l187_187449

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l187_187449


namespace diamond_evaluation_l187_187385

def diamond (X Y : ℚ) : ℚ := (2 * X + 3 * Y) / 5

theorem diamond_evaluation : diamond (diamond 3 15) 6 = 192 / 25 := 
by
  sorry

end diamond_evaluation_l187_187385


namespace at_least_one_negative_l187_187442

theorem at_least_one_negative (a : Fin 7 → ℤ) :
  (∀ i j : Fin 7, i ≠ j → a i ≠ a j) ∧
  (∀ l1 l2 l3 : Fin 7, 
    a l1 + a l2 + a l3 = a l1 + a l2 + a l3) ∧
  (∃ i : Fin 7, a i = 0) →
  (∃ i : Fin 7, a i < 0) :=
  by
  sorry

end at_least_one_negative_l187_187442


namespace total_football_games_l187_187560

theorem total_football_games (games_this_year : ℕ) (games_last_year : ℕ) (total_games : ℕ) : 
  games_this_year = 14 → games_last_year = 29 → total_games = games_this_year + games_last_year → total_games = 43 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_football_games_l187_187560


namespace systematic_sample_seat_number_l187_187083

theorem systematic_sample_seat_number (total_students sample_size interval : ℕ) (seat1 seat2 seat3 : ℕ) 
  (H_total_students : total_students = 56)
  (H_sample_size : sample_size = 4)
  (H_interval : interval = total_students / sample_size)
  (H_seat1 : seat1 = 3)
  (H_seat2 : seat2 = 31)
  (H_seat3 : seat3 = 45) :
  ∃ seat4 : ℕ, seat4 = 17 :=
by 
  sorry

end systematic_sample_seat_number_l187_187083


namespace option_C_correct_l187_187687

theorem option_C_correct : 5 + (-6) - (-7) = 5 - 6 + 7 := 
by
  sorry

end option_C_correct_l187_187687


namespace total_price_correct_l187_187285

-- Definitions of given conditions
def original_price : Float := 120
def discount_rate : Float := 0.30
def tax_rate : Float := 0.08

-- Definition of the final selling price
def sale_price : Float := original_price * (1 - discount_rate)
def total_selling_price : Float := sale_price * (1 + tax_rate)

-- Lean 4 statement to prove the total selling price is 90.72
theorem total_price_correct : total_selling_price = 90.72 := by
  sorry

end total_price_correct_l187_187285


namespace delta_comparison_eps_based_on_gamma_l187_187329

-- Definitions for the problem
variable {α β γ δ ε : ℝ}
variable {A B C : Type}
variable (s f m : Type)

-- Conditions from problem
variable (triangle_ABC : α ≠ β)
variable (median_s_from_C : s)
variable (angle_bisector_f : f)
variable (altitude_m : m)
variable (angle_between_f_m : δ = sorry)
variable (angle_between_f_s : ε = sorry)
variable (angle_at_vertex_C : γ = sorry)

-- Main statement to prove
theorem delta_comparison_eps_based_on_gamma (h1 : α ≠ β) (h2 : δ = sorry) (h3 : ε = sorry) (h4 : γ = sorry) :
  if γ < 90 then δ < ε else if γ = 90 then δ = ε else δ > ε :=
sorry

end delta_comparison_eps_based_on_gamma_l187_187329


namespace sufficient_but_not_necessary_l187_187917

variable (x : ℝ)

theorem sufficient_but_not_necessary : (x = 1) → (x^3 = x) ∧ (∀ y, y^3 = y → y = 1 → x ≠ y) :=
by
  sorry

end sufficient_but_not_necessary_l187_187917


namespace number_of_chicks_is_8_l187_187623

-- Define the number of total chickens
def total_chickens : ℕ := 15

-- Define the number of hens
def hens : ℕ := 3

-- Define the number of roosters
def roosters : ℕ := total_chickens - hens

-- Define the number of chicks
def chicks : ℕ := roosters - 4

-- State the main theorem
theorem number_of_chicks_is_8 : chicks = 8 := 
by
  -- the solution follows from the given definitions and conditions
  sorry

end number_of_chicks_is_8_l187_187623


namespace wall_width_l187_187732

theorem wall_width (area height : ℕ) (h1 : area = 16) (h2 : height = 4) : area / height = 4 :=
by
  sorry

end wall_width_l187_187732


namespace reflection_matrix_values_l187_187104

theorem reflection_matrix_values (a b : ℝ) (I : Matrix (Fin 2) (Fin 2) ℝ) :
  let R : Matrix (Fin 2) (Fin 2) ℝ := ![![a, 9/26], ![b, 17/26]]
  (R * R = I) → a = -17/26 ∧ b = 0 :=
by
  sorry

end reflection_matrix_values_l187_187104


namespace amount_saved_l187_187298

-- Initial conditions as definitions
def initial_amount : ℕ := 6000
def cost_ballpoint_pen : ℕ := 3200
def cost_eraser : ℕ := 1000
def cost_candy : ℕ := 500

-- Mathematical equivalent proof problem as a Lean theorem statement
theorem amount_saved : initial_amount - (cost_ballpoint_pen + cost_eraser + cost_candy) = 1300 := 
by 
  -- Proof is omitted
  sorry

end amount_saved_l187_187298


namespace expand_product_correct_l187_187547

noncomputable def expand_product (x : ℝ) : ℝ :=
  (3 / 7) * (7 / x^2 + 6 * x^3 - 2)

theorem expand_product_correct (x : ℝ) (h : x ≠ 0) :
  expand_product x = (3 / x^2) + (18 * x^3 / 7) - (6 / 7) := by
  unfold expand_product
  -- The proof will go here
  sorry

end expand_product_correct_l187_187547


namespace sum_of_cubes_eq_neg_27_l187_187691

theorem sum_of_cubes_eq_neg_27
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l187_187691


namespace min_value_343_l187_187974

noncomputable def min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ℝ :=
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a * b * c)

theorem min_value_343 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c ha hb hc = 343 :=
sorry

end min_value_343_l187_187974


namespace winter_sales_l187_187839

theorem winter_sales (T : ℕ) (spring_summer_sales : ℕ) (fall_sales : ℕ) (winter_sales : ℕ) 
  (h1 : T = 20) 
  (h2 : spring_summer_sales = 12) 
  (h3 : fall_sales = 4) 
  (h4 : T = spring_summer_sales + fall_sales + winter_sales) : 
     winter_sales = 4 := 
by 
  rw [h1, h2, h3] at h4
  linarith


end winter_sales_l187_187839


namespace infinite_power_tower_solution_l187_187736

theorem infinite_power_tower_solution : 
  ∃ x : ℝ, (∀ y, y = x ^ y → y = 4) → x = Real.sqrt 2 :=
by
  sorry

end infinite_power_tower_solution_l187_187736


namespace andrew_donates_160_to_homeless_shelter_l187_187524

/-- Andrew's bake sale earnings -/
def totalEarnings : ℕ := 400

/-- Amount kept by Andrew for ingredients -/
def ingredientsCost : ℕ := 100

/-- Amount Andrew donates from his own piggy bank -/
def piggyBankDonation : ℕ := 10

/-- The total amount Andrew donates to the homeless shelter -/
def totalDonationToHomelessShelter : ℕ :=
  let remaining := totalEarnings - ingredientsCost
  let halfDonation := remaining / 2
  halfDonation + piggyBankDonation

theorem andrew_donates_160_to_homeless_shelter : totalDonationToHomelessShelter = 160 := by
  sorry

end andrew_donates_160_to_homeless_shelter_l187_187524


namespace find_smallest_x_l187_187402

def smallest_x_divisible (y : ℕ) : ℕ :=
  if y = 11 then 257 else 0

theorem find_smallest_x : 
  smallest_x_divisible 11 = 257 ∧ 
  ∃ k : ℕ, 264 * k - 7 = 257 :=
by
  sorry

end find_smallest_x_l187_187402


namespace claudia_has_three_25_cent_coins_l187_187407

def number_of_coins (x y z : ℕ) := x + y + z = 15
def number_of_combinations (x y : ℕ) := 4 * x + 3 * y = 51

theorem claudia_has_three_25_cent_coins (x y z : ℕ) 
  (h1: number_of_coins x y z) 
  (h2: number_of_combinations x y): 
  z = 3 := 
by 
sorry

end claudia_has_three_25_cent_coins_l187_187407


namespace Eliza_first_more_than_300_paperclips_on_Thursday_l187_187744

theorem Eliza_first_more_than_300_paperclips_on_Thursday :
  ∃ k : ℕ, 5 * 3^k > 300 ∧ k = 4 := 
by
  sorry

end Eliza_first_more_than_300_paperclips_on_Thursday_l187_187744


namespace sum_of_roots_of_abs_quadratic_is_zero_l187_187663

theorem sum_of_roots_of_abs_quadratic_is_zero : 
  ∀ x : ℝ, (|x|^2 + |x| - 6 = 0) → (x = 2 ∨ x = -2) → (2 + (-2) = 0) :=
by
  intros x h h1
  sorry

end sum_of_roots_of_abs_quadratic_is_zero_l187_187663


namespace athlete_weight_l187_187053

theorem athlete_weight (a b c : ℤ) (k₁ k₂ k₃ : ℤ)
  (h1 : (a + b + c) / 3 = 42)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43)
  (h4 : a = 5 * k₁)
  (h5 : b = 5 * k₂)
  (h6 : c = 5 * k₃) :
  b = 40 :=
by
  sorry

end athlete_weight_l187_187053


namespace rectangle_breadth_l187_187217

/-- The breadth of the rectangle is 10 units given that
1. The length of the rectangle is two-fifths of the radius of a circle.
2. The radius of the circle is equal to the side of the square.
3. The area of the square is 1225 sq. units.
4. The area of the rectangle is 140 sq. units. -/
theorem rectangle_breadth (r l b : ℝ) (h_radius : r = 35) (h_length : l = (2 / 5) * r) (h_square : 35 * 35 = 1225) (h_area_rect : l * b = 140) : b = 10 :=
by
  sorry

end rectangle_breadth_l187_187217


namespace quadratic_roots_proof_l187_187851

theorem quadratic_roots_proof (b c : ℤ) :
  (∀ x : ℤ, x^2 + b * x + c = 0 ↔ (x = 1 ∨ x = -2)) → (b = 1 ∧ c = -2) :=
by
  sorry

end quadratic_roots_proof_l187_187851


namespace triangle_ABC_right_angled_l187_187015
open Real

theorem triangle_ABC_right_angled (A B C : ℝ) (a b c : ℝ)
  (h1 : cos (2 * A) - cos (2 * B) = 2 * sin C ^ 2)
  (h2 : a = sin A) (h3 : b = sin B) (h4 : c = sin C)
  : a^2 + c^2 = b^2 :=
by sorry

end triangle_ABC_right_angled_l187_187015


namespace coin_flip_probability_l187_187383

open Classical

noncomputable section

theorem coin_flip_probability :
  let total_outcomes := 2^10
  let exactly_five_heads_tails := Nat.choose 10 5 / total_outcomes
  let even_heads_probability := 1/2
  (even_heads_probability * (1 - exactly_five_heads_tails) / 2 = 193 / 512) :=
by
  sorry

end coin_flip_probability_l187_187383


namespace jessica_balloons_l187_187103

-- Given conditions
def joan_balloons : Nat := 9
def sally_balloons : Nat := 5
def total_balloons : Nat := 16

-- The theorem to prove the number of balloons Jessica has
theorem jessica_balloons : (total_balloons - (joan_balloons + sally_balloons) = 2) :=
by
  -- Proof goes here
  sorry

end jessica_balloons_l187_187103


namespace sum_of_3x3_matrix_arithmetic_eq_45_l187_187479

-- Statement: Prove that the sum of all nine elements of a 3x3 matrix, where each row and each column forms an arithmetic sequence and the middle element a_{22} = 5, is 45
theorem sum_of_3x3_matrix_arithmetic_eq_45 
  (matrix : ℤ → ℤ → ℤ)
  (arithmetic_row : ∀ i, matrix i 0 + matrix i 1 + matrix i 2 = 3 * matrix i 1)
  (arithmetic_col : ∀ j, matrix 0 j + matrix 1 j + matrix 2 j = 3 * matrix 1 j)
  (middle_elem : matrix 1 1 = 5) : 
  (matrix 0 0 + matrix 0 1 + matrix 0 2 + matrix 1 0 + matrix 1 1 + matrix 1 2 + matrix 2 0 + matrix 2 1 + matrix 2 2) = 45 :=
by
  sorry -- proof to be provided

end sum_of_3x3_matrix_arithmetic_eq_45_l187_187479


namespace neznaika_made_mistake_l187_187666

-- Define the total digits used from 1 to N pages
def totalDigits (N : ℕ) : ℕ :=
  let single_digit_pages := min N 9
  let double_digit_pages := if N > 9 then N - 9 else 0
  single_digit_pages * 1 + double_digit_pages * 2

-- The main statement we want to prove
theorem neznaika_made_mistake : ¬ ∃ N : ℕ, totalDigits N = 100 :=
by
  sorry

end neznaika_made_mistake_l187_187666


namespace Donny_change_l187_187576

/-- The change Donny will receive after filling up his truck. -/
theorem Donny_change
  (capacity : ℝ)
  (initial_fuel : ℝ)
  (cost_per_liter : ℝ)
  (money_available : ℝ)
  (change : ℝ) :
  capacity = 150 →
  initial_fuel = 38 →
  cost_per_liter = 3 →
  money_available = 350 →
  change = money_available - cost_per_liter * (capacity - initial_fuel) →
  change = 14 :=
by
  intros h_capacity h_initial_fuel h_cost_per_liter h_money_available h_change
  rw [h_capacity, h_initial_fuel, h_cost_per_liter, h_money_available] at h_change
  sorry

end Donny_change_l187_187576


namespace cubic_root_identity_l187_187537

theorem cubic_root_identity (x1 x2 x3 : ℝ) (h1 : x1^3 - 3*x1 - 1 = 0) (h2 : x2^3 - 3*x2 - 1 = 0) (h3 : x3^3 - 3*x3 - 1 = 0) (h4 : x1 < x2) (h5 : x2 < x3) :
  x3^2 - x2^2 = x3 - x1 :=
sorry

end cubic_root_identity_l187_187537


namespace complex_power_equality_l187_187813

namespace ComplexProof

open Complex

noncomputable def cos5 : ℂ := cos (5 * Real.pi / 180)

theorem complex_power_equality (w : ℂ) (h : w + 1 / w = 2 * cos5) : 
  w ^ 1000 + 1 / (w ^ 1000) = -((Real.sqrt 5 + 1) / 2) :=
sorry

end ComplexProof

end complex_power_equality_l187_187813


namespace intersection_M_N_l187_187113

noncomputable def M : Set ℕ := { x | 0 < x ∧ x < 8 }
def N : Set ℕ := { x | ∃ n : ℕ, x = 2 * n + 1 }
def K : Set ℕ := { 1, 3, 5, 7 }

theorem intersection_M_N : M ∩ N = K :=
by sorry

end intersection_M_N_l187_187113


namespace remainder_proof_l187_187689

def nums : List ℕ := [83, 84, 85, 86, 87, 88, 89, 90]
def mod : ℕ := 17

theorem remainder_proof : (nums.sum % mod) = 3 := by sorry

end remainder_proof_l187_187689


namespace smallest_positive_period_l187_187713

-- Define a predicate for a function to have a period
def is_periodic {α : Type*} [AddGroup α] (f : α → ℝ) (T : α) : Prop :=
  ∀ x, f (x) = f (x - T)

-- The actual problem statement
theorem smallest_positive_period {f : ℝ → ℝ} 
  (h : ∀ x : ℝ, f (3 * x) = f (3 * x - 3 / 2)) : 
  is_periodic f (1 / 2) ∧ 
  ¬ (∃ T : ℝ, 0 < T ∧ T < 1 / 2 ∧ is_periodic f T) :=
by
  sorry

end smallest_positive_period_l187_187713


namespace range_of_a_l187_187546

variable {a : ℝ}

def A := Set.Ioo (-1 : ℝ) 1
def B (a : ℝ) := Set.Ioo a (a + 1)

theorem range_of_a :
  B a ⊆ A ↔ (-1 : ℝ) ≤ a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l187_187546


namespace cowboy_cost_problem_l187_187539

/-- The cost of a sandwich, a cup of coffee, and a donut adds up to 0.40 dollars given the expenditure details of two cowboys. -/
theorem cowboy_cost_problem (S C D : ℝ) (h1 : 4 * S + C + 10 * D = 1.69) (h2 : 3 * S + C + 7 * D = 1.26) :
  S + C + D = 0.40 :=
by
  sorry

end cowboy_cost_problem_l187_187539


namespace better_offer_saves_800_l187_187603

theorem better_offer_saves_800 :
  let initial_order := 20000
  let discount1 (x : ℝ) := x * 0.70 * 0.90 - 800
  let discount2 (x : ℝ) := x * 0.75 * 0.80 - 1000
  discount1 initial_order - discount2 initial_order = 800 :=
by
  sorry

end better_offer_saves_800_l187_187603


namespace ratio_girls_to_boys_l187_187032

variable (g b : ℕ)

-- Conditions: total students are 30, six more girls than boys.
def total_students : Prop := g + b = 30
def six_more_girls : Prop := g = b + 6

-- Proof that the ratio of girls to boys is 3:2.
theorem ratio_girls_to_boys (ht : total_students g b) (hs : six_more_girls g b) : g / b = 3 / 2 :=
  sorry

end ratio_girls_to_boys_l187_187032


namespace candy_difference_l187_187800

theorem candy_difference 
  (total_candies : ℕ)
  (strawberry_candies : ℕ)
  (total_eq : total_candies = 821)
  (strawberry_eq : strawberry_candies = 267) : 
  (total_candies - strawberry_candies - strawberry_candies = 287) :=
by
  sorry

end candy_difference_l187_187800


namespace max_bananas_l187_187094

theorem max_bananas (a o b : ℕ) (h_a : a ≥ 1) (h_o : o ≥ 1) (h_b : b ≥ 1) (h_eq : 3 * a + 5 * o + 8 * b = 100) : b ≤ 11 :=
by {
  sorry
}

end max_bananas_l187_187094


namespace woman_waits_time_after_passing_l187_187877

-- Definitions based only on the conditions in a)
def man_speed : ℝ := 5 -- in miles per hour
def woman_speed : ℝ := 25 -- in miles per hour
def waiting_time_man_minutes : ℝ := 20 -- in minutes

-- Equivalent proof problem statement
theorem woman_waits_time_after_passing :
  let waiting_time_man_hours := waiting_time_man_minutes / 60
  let distance_man : ℝ := man_speed * waiting_time_man_hours
  let relative_speed : ℝ := woman_speed - man_speed
  let time_woman_covers_distance_hours := distance_man / relative_speed
  let time_woman_covers_distance_minutes := time_woman_covers_distance_hours * 60
  time_woman_covers_distance_minutes = 5 :=
by
  sorry

end woman_waits_time_after_passing_l187_187877


namespace eccentricity_of_hyperbola_l187_187568

noncomputable def hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (b * c) / (Real.sqrt (a^2 + b^2)) = (Real.sqrt 2 * c) / 3) : ℝ :=
  (3 * Real.sqrt 7) / 7

-- Ensure the function returns the correct eccentricity
theorem eccentricity_of_hyperbola (a b c : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (b * c) / (Real.sqrt (a^2 + b^2)) = (Real.sqrt 2 * c) / 3) : hyperbola_eccentricity a b c ha hb h = (3 * Real.sqrt 7) / 7 :=
sorry

end eccentricity_of_hyperbola_l187_187568


namespace polygon_angles_change_l187_187926

theorem polygon_angles_change (n : ℕ) :
  let initial_sum_interior := (n - 2) * 180
  let initial_sum_exterior := 360
  let new_sum_interior := (n + 2 - 2) * 180
  let new_sum_exterior := 360
  new_sum_exterior = initial_sum_exterior ∧ new_sum_interior - initial_sum_interior = 360 :=
by
  sorry

end polygon_angles_change_l187_187926


namespace relationship_between_M_n_and_N_n_plus_2_l187_187864

theorem relationship_between_M_n_and_N_n_plus_2 (n : ℕ) (h : 2 ≤ n) :
  let M_n := (n * (n + 1)) / 2 + 1
  let N_n_plus_2 := n + 3
  M_n < N_n_plus_2 :=
by
  sorry

end relationship_between_M_n_and_N_n_plus_2_l187_187864


namespace determine_a_range_l187_187710

noncomputable def single_element_intersection (a : ℝ) : Prop :=
  let A := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a * x + 1)}
  let B := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, |x|)}
  (∃ p : ℝ × ℝ, p ∈ A ∧ p ∈ B) ∧ 
  ∀ p₁ p₂ : ℝ × ℝ, p₁ ∈ A ∧ p₁ ∈ B → p₂ ∈ A ∧ p₂ ∈ B → p₁ = p₂

theorem determine_a_range : 
  ∀ a : ℝ, single_element_intersection a ↔ a ∈ Set.Iic (-1) ∨ a ∈ Set.Ici 1 :=
sorry

end determine_a_range_l187_187710


namespace suzanne_donation_l187_187684

theorem suzanne_donation :
  let base_donation := 10
  let total_distance := 5
  let total_donation := (List.range total_distance).foldl (fun acc km => acc + base_donation * 2 ^ km) 0
  total_donation = 310 :=
by
  let base_donation := 10
  let total_distance := 5
  let total_donation := (List.range total_distance).foldl (fun acc km => acc + base_donation * 2 ^ km) 0
  sorry

end suzanne_donation_l187_187684


namespace difference_of_x_y_l187_187874

theorem difference_of_x_y :
  ∀ (x y : ℤ), x + y = 10 → x = 14 → x - y = 18 :=
by
  intros x y h1 h2
  sorry

end difference_of_x_y_l187_187874


namespace arithmetic_sequence_n_value_l187_187451

theorem arithmetic_sequence_n_value (a : ℕ → ℤ) (a1 : a 1 = 1) (d : ℤ) (d_def : d = 3) (an : ∃ n, a n = 22) :
  ∃ n, n = 8 :=
by
  -- Assume the general term formula for the arithmetic sequence
  have general_term : ∀ n, a n = a 1 + (n-1) * d := sorry
  -- Use the given conditions
  have a_n_22 : ∃ n, a n = 22 := an
  -- Calculations to derive n = 8, skipped here
  sorry

end arithmetic_sequence_n_value_l187_187451


namespace silver_medals_count_l187_187468

def total_medals := 67
def gold_medals := 19
def bronze_medals := 16
def silver_medals := total_medals - gold_medals - bronze_medals

theorem silver_medals_count : silver_medals = 32 := by
  -- Proof goes here
  sorry

end silver_medals_count_l187_187468


namespace population_decrease_rate_l187_187120

theorem population_decrease_rate (r : ℕ) (h₀ : 6000 > 0) (h₁ : 4860 = 6000 * (1 - r / 100)^2) : r = 10 :=
by sorry

end population_decrease_rate_l187_187120


namespace negation_of_p_is_neg_p_l187_187999

-- Define the proposition p
def p : Prop := ∀ x : ℝ, x > 3 → x^3 - 27 > 0

-- Define the negation of proposition p
def neg_p : Prop := ∃ x : ℝ, x > 3 ∧ x^3 - 27 ≤ 0

-- The Lean statement that proves the problem
theorem negation_of_p_is_neg_p : ¬ p ↔ neg_p := by
  sorry

end negation_of_p_is_neg_p_l187_187999


namespace upper_limit_opinion_l187_187477

theorem upper_limit_opinion (w : ℝ) 
  (H1 : 61 < w ∧ w < 72) 
  (H2 : 60 < w ∧ w < 70) 
  (H3 : (61 + w) / 2 = 63) : w = 65 := 
by
  sorry

end upper_limit_opinion_l187_187477


namespace problem_1_problem_2_l187_187045

variable (x y : ℝ)
noncomputable def x_val : ℝ := 2 + Real.sqrt 3
noncomputable def y_val : ℝ := 2 - Real.sqrt 3

theorem problem_1 :
  3 * x_val^2 + 5 * x_val * y_val + 3 * y_val^2 = 47 := sorry

theorem problem_2 :
  Real.sqrt (x_val / y_val) + Real.sqrt (y_val / x_val) = 4 := sorry

end problem_1_problem_2_l187_187045


namespace events_A_B_mutually_exclusive_events_A_C_independent_l187_187278

-- Definitions for events A, B, and C
def event_A (x y : ℕ) : Prop := x + y = 7
def event_B (x y : ℕ) : Prop := (x * y) % 2 = 1
def event_C (x : ℕ) : Prop := x > 3

-- Proof problems to decide mutual exclusivity and independence
theorem events_A_B_mutually_exclusive :
  ∀ (x y : ℕ), event_A x y → ¬ event_B x y := 
by sorry

theorem events_A_C_independent :
  ∀ (x y : ℕ), (event_A x y) ↔ ∀ x y, event_C x ↔ event_A x y ∧ event_C x := 
by sorry

end events_A_B_mutually_exclusive_events_A_C_independent_l187_187278


namespace winston_initial_gas_l187_187430

theorem winston_initial_gas (max_gas : ℕ) (store_gas : ℕ) (doctor_gas : ℕ) :
  store_gas = 6 → doctor_gas = 2 → max_gas = 12 → max_gas - (store_gas + doctor_gas) = 4 → max_gas = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end winston_initial_gas_l187_187430


namespace max_dist_to_origin_from_curve_l187_187501

noncomputable def curve (θ : ℝ) : ℝ × ℝ :=
  let x := 3 + Real.sin θ
  let y := Real.cos θ
  (x, y)

theorem max_dist_to_origin_from_curve :
  ∃ M : ℝ × ℝ, (∃ θ : ℝ, M = curve θ) ∧ Real.sqrt (M.fst^2 + M.snd^2) ≤ 4 :=
by
  sorry

end max_dist_to_origin_from_curve_l187_187501


namespace min_value_of_f_solution_set_of_inequality_l187_187570

-- Define the given function f
def f (x : ℝ) : ℝ := abs (x - 1) + abs (2 * x + 4)

-- (1) Prove that the minimum value of y = f(x) is 3
theorem min_value_of_f : ∃ x : ℝ, f x = 3 := 
sorry

-- (2) Prove that the solution set of the inequality |f(x) - 6| ≤ 1 is [-10/3, -8/3] ∪ [0, 4/3]
theorem solution_set_of_inequality : 
  {x | |f x - 6| ≤ 1} = {x | -(10/3) ≤ x ∧ x ≤ -(8/3) ∨ 0 ≤ x ∧ x ≤ (4/3)} :=
sorry

end min_value_of_f_solution_set_of_inequality_l187_187570


namespace total_cost_of_crayons_l187_187940

theorem total_cost_of_crayons (crayons_per_half_dozen : ℕ)
    (number_of_half_dozens : ℕ)
    (cost_per_crayon : ℕ)
    (total_cost : ℕ) :
  crayons_per_half_dozen = 6 →
  number_of_half_dozens = 4 →
  cost_per_crayon = 2 →
  total_cost = crayons_per_half_dozen * number_of_half_dozens * cost_per_crayon →
  total_cost = 48 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end total_cost_of_crayons_l187_187940


namespace right_triangle_area_perimeter_ratio_l187_187088

theorem right_triangle_area_perimeter_ratio :
  let a := 4
  let b := 8
  let area := (1/2) * a * b
  let c := Real.sqrt (a^2 + b^2)
  let perimeter := a + b + c
  let ratio := area / perimeter
  ratio = 3 - Real.sqrt 5 :=
by
  sorry

end right_triangle_area_perimeter_ratio_l187_187088


namespace twin_primes_divisible_by_12_l187_187316

def isTwinPrime (p q : ℕ) : Prop :=
  p < q ∧ Nat.Prime p ∧ Nat.Prime q ∧ q - p = 2

theorem twin_primes_divisible_by_12 {p q r s : ℕ} 
  (h1 : isTwinPrime p q) 
  (h2 : p > 3) 
  (h3 : isTwinPrime r s) 
  (h4 : r > 3) :
  12 ∣ (p * r - q * s) := by
  sorry

end twin_primes_divisible_by_12_l187_187316


namespace find_side_b_l187_187533

-- Given the side and angle conditions in the triangle
variable (A B C : ℝ)
variable (a b c : ℝ)
variable (S : ℝ) 

-- Conditions provided in the problem
axiom side_a (h : a = 1) : True
axiom angle_B (h : B = Real.pi / 4) : True  -- 45 degrees in radians
axiom area_triangle (h : S = 2) : True

-- Final proof statement
theorem find_side_b (h₁ : a = 1) (h₂ : B = Real.pi / 4) (h₃ : S = 2) : 
  b = 5 := sorry

end find_side_b_l187_187533


namespace find_f_of_2_l187_187014

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f_of_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end find_f_of_2_l187_187014


namespace range_of_a_l187_187575

theorem range_of_a (a : ℝ) :
  (a > 0 ∧ (∃ x, x^2 - 4 * a * x + 3 * a^2 < 0)) →
  (∃ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2 * x - 8 > 0) →
  (2 < a ∧ a ≤ 2) := sorry

end range_of_a_l187_187575


namespace tan_alpha_eq_one_l187_187803

theorem tan_alpha_eq_one (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : Real.cos (α + β) = Real.sin (α - β)) : Real.tan α = 1 :=
sorry

end tan_alpha_eq_one_l187_187803


namespace peanut_total_correct_l187_187504

-- Definitions based on the problem conditions:

def jose_peanuts : ℕ := 85
def kenya_peanuts : ℕ := jose_peanuts + 48
def malachi_peanuts : ℕ := kenya_peanuts + 35
def total_peanuts : ℕ := jose_peanuts + kenya_peanuts + malachi_peanuts

-- Statement to be proven:
theorem peanut_total_correct : total_peanuts = 386 :=
by 
  -- The proof would be here, but we skip it according to the instruction
  sorry

end peanut_total_correct_l187_187504


namespace number_of_green_hats_l187_187511

variables (B G : ℕ)

-- Given conditions as definitions
def totalHats : Prop := B + G = 85
def totalCost : Prop := 6 * B + 7 * G = 530

-- The statement we need to prove
theorem number_of_green_hats (h1 : totalHats B G) (h2 : totalCost B G) : G = 20 :=
sorry

end number_of_green_hats_l187_187511


namespace no_three_real_numbers_satisfy_inequalities_l187_187093

theorem no_three_real_numbers_satisfy_inequalities (a b c : ℝ) :
  ¬ (|a| < |b - c| ∧ |b| < |c - a| ∧ |c| < |a - b| ) :=
by
  sorry

end no_three_real_numbers_satisfy_inequalities_l187_187093


namespace range_of_f_l187_187375

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem range_of_f :
  (∀ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → 0 ≤ f x ∧ f x ≤ 3) := sorry

end range_of_f_l187_187375


namespace total_hens_and_cows_l187_187845

theorem total_hens_and_cows (H C : ℕ) (hH : H = 28) (h_feet : 2 * H + 4 * C = 136) : H + C = 48 :=
by
  -- Proof goes here 
  sorry

end total_hens_and_cows_l187_187845


namespace solve_for_x_l187_187148

theorem solve_for_x : 
  (35 / (6 - (2 / 5)) = 25 / 4) := 
by
  sorry 

end solve_for_x_l187_187148


namespace find_q_sum_of_bn_l187_187679

-- Defining the sequences and conditions
def a (n : ℕ) (q : ℝ) : ℝ := q^(n-1)

def b (n : ℕ) (q : ℝ) : ℝ := a n q + n

-- Given that 2a_1, (1/2)a_3, a_2 form an arithmetic sequence
def condition_arithmetic_sequence (q : ℝ) : Prop :=
  2 * a 1 q + a 2 q = (1 / 2) * a 3 q + (1 / 2) * a 3 q

-- To be proved: Given conditions, prove q = 2
theorem find_q : ∃ q > 0, a 1 q = 1 ∧ a 2 q = q ∧ a 3 q = q^2 ∧ condition_arithmetic_sequence q ∧ q = 2 :=
by {
  sorry
}

-- Given b_n = a_n + n, prove T_n = (n(n+1))/2 + 2^n - 1
theorem sum_of_bn (n : ℕ) : 
  ∃ T_n : ℕ → ℝ, T_n n = (n * (n + 1)) / 2 + (2^n) - 1 :=
by {
  sorry
}

end find_q_sum_of_bn_l187_187679


namespace power_function_pass_through_point_l187_187755

theorem power_function_pass_through_point (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ a) (h_point : f 2 = 16) : a = 4 :=
sorry

end power_function_pass_through_point_l187_187755


namespace find_smaller_number_l187_187469

noncomputable def smaller_number (x y : ℝ) := y

theorem find_smaller_number 
  (x y : ℝ) 
  (h1 : x - y = 9) 
  (h2 : x + y = 46) :
  smaller_number x y = 18.5 :=
sorry

end find_smaller_number_l187_187469


namespace pyramid_volume_l187_187636

theorem pyramid_volume (a : ℝ) (h : a = 2)
  (b : ℝ) (hb : b = 18) :
  ∃ V, V = 2 * Real.sqrt 2 :=
by
  sorry

end pyramid_volume_l187_187636


namespace length_of_AE_l187_187020

theorem length_of_AE (AD AE EB EF: ℝ) (h_AD: AD = 80) (h_EB: EB = 40) (h_EF: EF = 30) 
  (h_eq_area: 2 * ((EB * EF) + (1 / 2) * (ED * (AD - EF))) = AD * (AD - AE)) : AE = 15 :=
  sorry

end length_of_AE_l187_187020


namespace shrub_height_at_end_of_2_years_l187_187436

theorem shrub_height_at_end_of_2_years (h₅ : ℕ) (h : ∀ n : ℕ, 0 < n → 243 = 3^5 * h₅) : ∃ h₂ : ℕ, h₂ = 9 :=
by sorry

end shrub_height_at_end_of_2_years_l187_187436


namespace fish_in_pond_l187_187727

noncomputable def number_of_fish (marked_first: ℕ) (marked_second: ℕ) (catch_first: ℕ) (catch_second: ℕ) : ℕ :=
  (marked_first * catch_second) / marked_second

theorem fish_in_pond (h1 : marked_first = 30) (h2 : marked_second = 2) (h3 : catch_first = 30) (h4 : catch_second = 40) :
  number_of_fish marked_first marked_second catch_first catch_second = 600 :=
by
  rw [h1, h2, h3, h4]
  sorry

end fish_in_pond_l187_187727


namespace plane_distance_l187_187505

theorem plane_distance :
  let distance_AB := 100
  let distance_BC := distance_AB + 50
  let distance_CD := 2 * distance_BC
  let total_distance_AD := distance_AB + distance_BC + distance_CD
  total_distance_AD = 550 :=
by
  intros
  let distance_AB := 100
  let distance_BC := distance_AB + 50
  let distance_CD := 2 * distance_BC
  let total_distance_AD := distance_AB + distance_BC + distance_CD
  sorry

end plane_distance_l187_187505


namespace probability_of_hitting_target_at_least_once_l187_187306

theorem probability_of_hitting_target_at_least_once :
  (∀ (p1 p2 : ℝ), p1 = 0.5 → p2 = 0.7 → (1 - (1 - p1) * (1 - p2)) = 0.85) :=
by
  intros p1 p2 h1 h2
  rw [h1, h2]
  -- This rw step simplifies (1 - (1 - 0.5) * (1 - 0.7)) to the desired result.
  sorry

end probability_of_hitting_target_at_least_once_l187_187306


namespace value_of_t_plus_k_l187_187986

noncomputable def f (x t : ℝ) : ℝ := x^3 + (t - 1) * x^2 - 1

theorem value_of_t_plus_k (k t : ℝ)
  (h1 : k ≠ 0)
  (h2 : ∀ x, f x t = 2 * x - 1)
  (h3 : ∃ x₁ x₂, f x₁ t = 2 * x₁ - 1 ∧ f x₂ t = 2 * x₂ - 1) :
  t + k = 7 :=
sorry

end value_of_t_plus_k_l187_187986


namespace symmetric_point_x_axis_l187_187046

theorem symmetric_point_x_axis (P : ℝ × ℝ) (hx : P = (2, 3)) : P.1 = 2 ∧ P.2 = -3 :=
by
  -- The proof is omitted
  sorry

end symmetric_point_x_axis_l187_187046


namespace brooke_social_studies_problems_l187_187816

theorem brooke_social_studies_problems :
  ∀ (math_problems science_problems total_minutes : Nat) 
    (math_time_per_problem science_time_per_problem soc_studies_time_per_problem : Nat)
    (soc_studies_problems : Nat),
  math_problems = 15 →
  science_problems = 10 →
  total_minutes = 48 →
  math_time_per_problem = 2 →
  science_time_per_problem = 3 / 2 → -- converting 1.5 minutes to a fraction
  soc_studies_time_per_problem = 1 / 2 → -- converting 30 seconds to a fraction
  math_problems * math_time_per_problem + science_problems * science_time_per_problem + soc_studies_problems * soc_studies_time_per_problem = 48 →
  soc_studies_problems = 6 :=
by
  intros math_problems science_problems total_minutes math_time_per_problem science_time_per_problem soc_studies_time_per_problem soc_studies_problems
  intros h_math_problems h_science_problems h_total_minutes h_math_time_per_problem h_science_time_per_problem h_soc_studies_time_per_problem h_eq
  sorry

end brooke_social_studies_problems_l187_187816


namespace general_formula_for_sequence_l187_187260

noncomputable def a_n (n : ℕ) : ℕ := sorry
noncomputable def S_n (n : ℕ) : ℕ := sorry

theorem general_formula_for_sequence {n : ℕ} (hn: n > 0)
  (h1: ∀ n, a_n n > 0)
  (h2: ∀ n, 4 * S_n n = (a_n n)^2 + 2 * (a_n n))
  : a_n n = 2 * n := sorry

end general_formula_for_sequence_l187_187260


namespace find_a2_l187_187461

theorem find_a2 
  (a1 a2 a3 : ℝ)
  (h1 : a1 * a2 * a3 = 15)
  (h2 : (3 / (a1 * 3 * a2)) + (15 / (3 * a2 * 5 * a3)) + (5 / (5 * a3 * a1)) = 3 / 5) :
  a2 = 3 :=
sorry

end find_a2_l187_187461


namespace work_completion_time_for_A_l187_187757

theorem work_completion_time_for_A 
  (B_work_rate : ℝ)
  (combined_work_rate : ℝ)
  (x : ℝ) 
  (B_work_rate_def : B_work_rate = 1 / 6)
  (combined_work_rate_def : combined_work_rate = 3 / 10) :
  (1 / x) + B_work_rate = combined_work_rate →
  x = 7.5 := 
by
  sorry

end work_completion_time_for_A_l187_187757


namespace bird_migration_difference_correct_l187_187296

def bird_migration_difference : ℕ := 54

/--
There are 250 bird families consisting of 3 different bird species, each with varying migration patterns.

Species A: 100 bird families; 35% fly to Africa, 65% fly to Asia
Species B: 120 bird families; 50% fly to Africa, 50% fly to Asia
Species C: 30 bird families; 10% fly to Africa, 90% fly to Asia

Prove that the difference in the number of bird families migrating to Asia and Africa is 54.
-/
theorem bird_migration_difference_correct (A_Africa_percent : ℕ := 35) (A_Asia_percent : ℕ := 65)
  (B_Africa_percent : ℕ := 50) (B_Asia_percent : ℕ := 50)
  (C_Africa_percent : ℕ := 10) (C_Asia_percent : ℕ := 90)
  (A_count : ℕ := 100) (B_count : ℕ := 120) (C_count : ℕ := 30) :
    bird_migration_difference = 
      (A_count * A_Asia_percent / 100 + B_count * B_Asia_percent / 100 + C_count * C_Asia_percent / 100) - 
      (A_count * A_Africa_percent / 100 + B_count * B_Africa_percent / 100 + C_count * C_Africa_percent / 100) :=
by sorry

end bird_migration_difference_correct_l187_187296


namespace parabola_line_intersection_l187_187154

theorem parabola_line_intersection :
  ∀ (x y : ℝ), 
  (y = 20 * x^2 + 19 * x) ∧ (y = 20 * x + 19) →
  y = 20 * x^3 + 19 * x^2 :=
by sorry

end parabola_line_intersection_l187_187154


namespace polygon_num_sides_l187_187292

theorem polygon_num_sides (s : ℕ) (h : 180 * (s - 2) > 2790) : s = 18 :=
sorry

end polygon_num_sides_l187_187292


namespace sequence_bound_l187_187540

theorem sequence_bound (a : ℕ → ℝ) (n : ℕ) 
  (h₁ : a 0 = 0) 
  (h₂ : a (n + 1) = 0)
  (h₃ : ∀ k, 1 ≤ k → k ≤ n → a (k - 1) - 2 * (a k) + (a (k + 1)) ≤ 1) 
  : ∀ k, 0 ≤ k → k ≤ n + 1 → a k ≤ (k * (n + 1 - k)) / 2 :=
sorry

end sequence_bound_l187_187540


namespace find_values_of_a_and_b_find_square_root_l187_187997

-- Define the conditions
def condition1 (a b : ℤ) : Prop := (2 * b - 2 * a)^3 = -8
def condition2 (a b : ℤ) : Prop := (4 * a + 3 * b)^2 = 9

-- State the problem to prove the values of a and b
theorem find_values_of_a_and_b (a b : ℤ) (h1 : condition1 a b) (h2 : condition2 a b) : 
  a = 3 ∧ b = -1 :=
sorry

-- State the problem to prove the square root of 5a - b
theorem find_square_root (a b : ℤ) (h1 : condition1 a b) (h2 : condition2 a b) (ha : a = 3) (hb : b = -1) :
  ∃ x : ℤ, x^2 = 5 * a - b ∧ (x = 4 ∨ x = -4) :=
sorry

end find_values_of_a_and_b_find_square_root_l187_187997


namespace percentage_salt_in_mixture_l187_187619

-- Conditions
def volume_pure_water : ℝ := 1
def volume_salt_solution : ℝ := 2
def salt_concentration : ℝ := 0.30
def total_volume : ℝ := volume_pure_water + volume_salt_solution
def amount_of_salt_in_solution : ℝ := salt_concentration * volume_salt_solution

-- Theorem
theorem percentage_salt_in_mixture :
  (amount_of_salt_in_solution / total_volume) * 100 = 20 :=
by
  sorry

end percentage_salt_in_mixture_l187_187619


namespace pencil_length_l187_187860

theorem pencil_length
  (R P L : ℕ)
  (h1 : P = R + 3)
  (h2 : P = L - 2)
  (h3 : R + P + L = 29) :
  L = 12 :=
by
  sorry

end pencil_length_l187_187860


namespace Jungkook_has_most_apples_l187_187590

-- Conditions
def Yoongi_apples : ℕ := 4
def Jungkook_apples_initial : ℕ := 6
def Jungkook_apples_additional : ℕ := 3
def Jungkook_total_apples : ℕ := Jungkook_apples_initial + Jungkook_apples_additional
def Yuna_apples : ℕ := 5

-- Statement (to prove)
theorem Jungkook_has_most_apples : Jungkook_total_apples > Yoongi_apples ∧ Jungkook_total_apples > Yuna_apples := by
  sorry

end Jungkook_has_most_apples_l187_187590


namespace abs_y_lt_inequality_sum_l187_187219

-- Problem (1)
theorem abs_y_lt {
  x y : ℝ
} (h1 : |x - y| < 1) (h2 : |2 * x + y| < 1) :
  |y| < 1 := by
  sorry

-- Problem (2)
theorem inequality_sum {
  a b c d : ℝ
} (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b) + 1 / (b - c) + 1 / (c - d)) ≥ 9 / (a - d) := by
  sorry

end abs_y_lt_inequality_sum_l187_187219


namespace no_intersection_of_curves_l187_187244

theorem no_intersection_of_curves :
  ∀ x y : ℝ, ¬ (3 * x^2 + 2 * y^2 = 4 ∧ 6 * x^2 + 3 * y^2 = 9) :=
by sorry

end no_intersection_of_curves_l187_187244


namespace sum_of_consecutive_naturals_l187_187856

theorem sum_of_consecutive_naturals (n : ℕ) : 
  ∃ S : ℕ, S = n * (n + 1) / 2 :=
by
  sorry

end sum_of_consecutive_naturals_l187_187856


namespace range_of_x_minus_2y_l187_187149

theorem range_of_x_minus_2y (x y : ℝ) (h₁ : -1 ≤ x) (h₂ : x < 2) (h₃ : 0 < y) (h₄ : y ≤ 1) :
  -3 ≤ x - 2 * y ∧ x - 2 * y < 2 :=
sorry

end range_of_x_minus_2y_l187_187149


namespace license_plate_count_l187_187879

theorem license_plate_count :
  let letters := 26
  let digits := 10
  let total_count := letters * (letters - 1) + letters
  total_count * digits = 6760 :=
by sorry

end license_plate_count_l187_187879


namespace proof_problem_l187_187073

noncomputable def question (a b c : ℝ) : ℝ := 
  (a ^ 2 * b ^ 2) / ((a ^ 2 + b * c) * (b ^ 2 + a * c)) +
  (a ^ 2 * c ^ 2) / ((a ^ 2 + b * c) * (c ^ 2 + a * b)) +
  (b ^ 2 * c ^ 2) / ((b ^ 2 + a * c) * (c ^ 2 + a * b))

theorem proof_problem (a b c : ℝ) (h : a ≠ 0) (h1 : b ≠ 0) (h2 : c ≠ 0) 
  (h3 : a ^ 2 + b ^ 2 + c ^ 2 = a * b + b * c + c * a ) : 
  question a b c = 1 := 
by 
  sorry

end proof_problem_l187_187073


namespace find_smallest_n_l187_187317

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l187_187317


namespace blueberry_picking_l187_187766

-- Define the amounts y1 and y2 as a function of x
variable (x : ℝ)
def y1 : ℝ := 60 + 18 * x
def y2 : ℝ := 150 + 15 * x

-- State the theorem about the relationships given the condition 
theorem blueberry_picking (hx : x > 10) : 
  y1 x = 60 + 18 * x ∧ y2 x = 150 + 15 * x :=
by
  sorry

end blueberry_picking_l187_187766


namespace football_field_area_l187_187507

theorem football_field_area (total_fertilizer : ℝ) (partial_fertilizer : ℝ) (partial_area : ℝ) (fertilizer_rate : ℝ) (total_area : ℝ) 
  (h1 : total_fertilizer = 800)
  (h2: partial_fertilizer = 300)
  (h3: partial_area = 3600)
  (h4: fertilizer_rate = partial_fertilizer / partial_area)
  (h5: total_area = total_fertilizer / fertilizer_rate) 
  : total_area = 9600 := 
sorry

end football_field_area_l187_187507


namespace number_of_valid_integers_l187_187673

theorem number_of_valid_integers (n : ℕ) (h1 : n ≤ 2021) (h2 : ∀ m : ℕ, m^2 ≤ n → n < (m + 1)^2 → ((m^2 + 1) ∣ (n^2 + 1))) : 
  ∃ k, k = 47 :=
by
  sorry

end number_of_valid_integers_l187_187673


namespace total_tape_length_l187_187008

-- Definitions based on the problem conditions
def first_side_songs : ℕ := 6
def second_side_songs : ℕ := 4
def song_length : ℕ := 4

-- Statement to prove the total tape length is 40 minutes
theorem total_tape_length : (first_side_songs + second_side_songs) * song_length = 40 := by
  sorry

end total_tape_length_l187_187008


namespace root_of_polynomial_l187_187057

theorem root_of_polynomial (a b : ℝ) (h₁ : a^4 + a^3 - 1 = 0) (h₂ : b^4 + b^3 - 1 = 0) : 
  (ab : ℝ) → ab * ab * ab * ab * ab * ab + ab * ab * ab * ab + ab * ab * ab - ab * ab - 1 = 0 :=
sorry

end root_of_polynomial_l187_187057


namespace trajectory_eq_l187_187790

theorem trajectory_eq {x y m : ℝ} (h : x^2 + y^2 - (4 * m + 2) * x - 2 * m * y + 4 * m^2 + 4 * m + 1 = 0) :
  x - 2 * y - 1 = 0 ∧ x ≠ 1 :=
sorry

end trajectory_eq_l187_187790


namespace solve_inequalities_l187_187165

theorem solve_inequalities :
  {x : ℝ | -3 < x ∧ x ≤ -2 ∨ 1 ≤ x ∧ x ≤ 2} =
  { x : ℝ | (5 / (x + 3) ≥ 1) ∧ (x^2 + x - 2 ≥ 0) } :=
sorry

end solve_inequalities_l187_187165


namespace sequence_arithmetic_difference_neg1_l187_187686

variable (a : ℕ → ℝ)

theorem sequence_arithmetic_difference_neg1 (h : ∀ n, a (n + 1) + 1 = a n) : ∀ n, a (n + 1) - a n = -1 :=
by
  intro n
  specialize h n
  linarith

-- Assuming natural numbers starting from 1 (ℕ^*), which is not directly available in Lean.
-- So we use assumptions accordingly.

end sequence_arithmetic_difference_neg1_l187_187686


namespace circles_point_distance_l187_187910

noncomputable section

-- Define the data for the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def CircleA (R : ℝ) (K : ℝ × ℝ) : Circle := 
  { center := K, radius := R }

def CircleB (R : ℝ) (K : ℝ × ℝ) : Circle := 
  { center := (K.1 + 2 * R, K.2), radius := R }

-- Define the condition that two circles touch each other at point K
def circles_touch (C1 C2 : Circle) (K : ℝ × ℝ) : Prop :=
  dist C1.center K = C1.radius ∧ dist C2.center K = C2.radius ∧ dist C1.center C2.center = C1.radius + C2.radius

-- Define the angle condition ∠AKB = 90°
def angle_AKB_is_right (A K B : ℝ × ℝ) : Prop :=
  -- Using the fact that a dot product being zero implies orthogonality
  let vec1 := (A.1 - K.1, A.2 - K.2)
  let vec2 := (B.1 - K.1, B.2 - K.2)
  vec1.1 * vec2.1 + vec1.2 * vec2.2 = 0

-- Define the points A and B being on their respective circles
def on_circle (A : ℝ × ℝ) (C : Circle) : Prop :=
  dist A C.center = C.radius

-- Define the theorem
theorem circles_point_distance 
  (R : ℝ) (K A B : ℝ × ℝ) 
  (C1 := CircleA R K) 
  (C2 := CircleB R K) 
  (h1 : circles_touch C1 C2 K) 
  (h2 : on_circle A C1) 
  (h3 : on_circle B C2) 
  (h4 : angle_AKB_is_right A K B) : 
  dist A B = 2 * R := 
sorry

end circles_point_distance_l187_187910


namespace ratio_Theresa_Timothy_2010_l187_187703

def Timothy_movies_2009 : Nat := 24
def Timothy_movies_2010 := Timothy_movies_2009 + 7
def Theresa_movies_2009 := Timothy_movies_2009 / 2
def total_movies := 129
def Timothy_total_movies := Timothy_movies_2009 + Timothy_movies_2010
def Theresa_total_movies := total_movies - Timothy_total_movies
def Theresa_movies_2010 := Theresa_total_movies - Theresa_movies_2009

theorem ratio_Theresa_Timothy_2010 :
  (Theresa_movies_2010 / Timothy_movies_2010) = 2 :=
by
  sorry

end ratio_Theresa_Timothy_2010_l187_187703


namespace sin_bound_l187_187231

theorem sin_bound (a : ℝ) (h : ¬ ∃ x : ℝ, Real.sin x > a) : a ≥ 1 := 
sorry

end sin_bound_l187_187231


namespace jordan_width_45_l187_187131

noncomputable def carolRectangleLength : ℕ := 15
noncomputable def carolRectangleWidth : ℕ := 24
noncomputable def jordanRectangleLength : ℕ := 8
noncomputable def carolRectangleArea : ℕ := carolRectangleLength * carolRectangleWidth
noncomputable def jordanRectangleWidth (area : ℕ) : ℕ := area / jordanRectangleLength

theorem jordan_width_45 : jordanRectangleWidth carolRectangleArea = 45 :=
by sorry

end jordan_width_45_l187_187131


namespace laptop_cost_l187_187240

theorem laptop_cost
  (C : ℝ) (down_payment := 0.2 * C + 20) (installments_paid := 65 * 4) (balance_after_4_months := 520)
  (h : C - (down_payment + installments_paid) = balance_after_4_months) :
  C = 1000 :=
by
  sorry

end laptop_cost_l187_187240


namespace union_of_A_and_B_l187_187536

def setA : Set ℝ := {x | 2 * x - 1 > 0}
def setB : Set ℝ := {x | abs x < 1}

theorem union_of_A_and_B : setA ∪ setB = {x : ℝ | x > -1} := 
by {
  sorry
}

end union_of_A_and_B_l187_187536


namespace inequality_transformation_l187_187522

theorem inequality_transformation (x y : ℝ) (h : 2 * x - 5 < 2 * y - 5) : x < y := 
by 
  sorry

end inequality_transformation_l187_187522


namespace problem1_problem2_l187_187203

def setA : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

def setB (m : ℝ) : Set ℝ := {x | (x - m + 2) * (x - m - 2) ≤ 0}

-- Problem 1: prove that if A ∩ B = {x | 0 ≤ x ≤ 3}, then m = 2
theorem problem1 (m : ℝ) : (setA ∩ setB m = {x | 0 ≤ x ∧ x ≤ 3}) → m = 2 :=
by
  sorry

-- Problem 2: prove that if A ⊆ complement of B, then m ∈ (-∞, -3) ∪ (5, +∞)
theorem problem2 (m : ℝ) : (setA ⊆ (fun x => x ∉ setB m)) → (m < -3 ∨ m > 5) :=
by
  sorry

end problem1_problem2_l187_187203


namespace minimum_cost_for_18_oranges_l187_187351

noncomputable def min_cost_oranges (x y : ℕ) : ℕ :=
  10 * x + 30 * y

theorem minimum_cost_for_18_oranges :
  (∃ x y : ℕ, 3 * x + 7 * y = 18 ∧ min_cost_oranges x y = 60) ∧ (60 / 18 = 10 / 3) :=
sorry

end minimum_cost_for_18_oranges_l187_187351


namespace negation_of_proposition_l187_187318

theorem negation_of_proposition :
  (∀ x y : ℝ, (x * y = 0 → x = 0 ∨ y = 0)) →
  (∃ x y : ℝ, x * y = 0 ∧ x ≠ 0 ∧ y ≠ 0) :=
sorry

end negation_of_proposition_l187_187318


namespace max_pencils_to_buy_l187_187134

-- Definition of costs and budget
def pin_cost : ℕ := 3
def pen_cost : ℕ := 4
def pencil_cost : ℕ := 9
def total_budget : ℕ := 72

-- Minimum purchase required: one pin and one pen
def min_purchase : ℕ := pin_cost + pen_cost

-- Remaining budget after minimum purchase
def remaining_budget : ℕ := total_budget - min_purchase

-- Maximum number of pencils can be bought with the remaining budget
def max_pencils := remaining_budget / pencil_cost

-- Theorem stating the maximum number of pencils Alice can purchase
theorem max_pencils_to_buy : max_pencils = 7 :=
by
  -- Proof would go here
  sorry

end max_pencils_to_buy_l187_187134


namespace batsman_average_46_innings_l187_187346

theorem batsman_average_46_innings {hs ls t_44 : ℕ} (h_diff: hs - ls = 180) (h_avg_44: t_44 = 58 * 44) (h_hiscore: hs = 194) : 
  (t_44 + hs + ls) / 46 = 60 := 
sorry

end batsman_average_46_innings_l187_187346


namespace alpha_eq_one_l187_187176

-- Definitions based on conditions from the problem statement.
variable (α : ℝ) 
variable (f : ℝ → ℝ)

-- The conditions defined as hypotheses
axiom functional_eq (x y : ℝ) : f (α * (x + y)) = f x + f y
axiom non_constant : ∃ x y : ℝ, f x ≠ 0

-- The statement to prove
theorem alpha_eq_one : (∃ f : ℝ → ℝ, (∀ x y : ℝ, f (α * (x + y)) = f x + f y) ∧ (∃ x y : ℝ, f x ≠ f y)) → α = 1 :=
by
  sorry

end alpha_eq_one_l187_187176


namespace smallest_possible_n_l187_187695

theorem smallest_possible_n (n : ℕ) (h1 : 0 < n) (h2 : 0 < 60) 
  (h3 : (Nat.lcm 60 n) / (Nat.gcd 60 n) = 24) : n = 20 :=
by sorry

end smallest_possible_n_l187_187695


namespace hyperbola_range_of_m_l187_187873

theorem hyperbola_range_of_m (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (1 + m) + (y^2) / (1 - m) = 1) → 
  (m < -1 ∨ m > 1) :=
by 
sorry

end hyperbola_range_of_m_l187_187873


namespace diameterOuterBoundary_l187_187496

-- Definitions based on the conditions in the problem
def widthWalkingPath : ℝ := 10
def widthGardenRing : ℝ := 12
def diameterPond : ℝ := 16

-- The main theorem that proves the diameter of the circle that forms the outer boundary of the walking path
theorem diameterOuterBoundary : 2 * ((diameterPond / 2) + widthGardenRing + widthWalkingPath) = 60 :=
by
  sorry

end diameterOuterBoundary_l187_187496


namespace add_ab_equals_four_l187_187765

theorem add_ab_equals_four (a b : ℝ) (h₁ : a * (a - 4) = 5) (h₂ : b * (b - 4) = 5) (h₃ : a ≠ b) : a + b = 4 :=
by
  sorry

end add_ab_equals_four_l187_187765


namespace sum_of_integers_l187_187151

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
by
  sorry

end sum_of_integers_l187_187151


namespace log_base_16_of_4_eq_half_l187_187519

noncomputable def logBase (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem log_base_16_of_4_eq_half :
  logBase 16 4 = 1 / 2 := by
sorry

end log_base_16_of_4_eq_half_l187_187519


namespace trapezium_side_length_l187_187294

theorem trapezium_side_length (a b h A x : ℝ) 
  (ha : a = 20) (hh : h = 15) (hA : A = 285) 
  (h_formula : A = 1 / 2 * (a + b) * h) : 
  b = 18 :=
by
  sorry

end trapezium_side_length_l187_187294


namespace phase_shift_of_cosine_l187_187631

theorem phase_shift_of_cosine (a b c : ℝ) (h : c = -π / 4 ∧ b = 3) :
  (-c / b) = π / 12 :=
by
  sorry

end phase_shift_of_cosine_l187_187631


namespace problem_part_1_problem_part_2_l187_187662
open Set Real

noncomputable def A (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a^2 - 2}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem problem_part_1 : A 3 ∪ B = {x | 1 < x ∧ x ≤ 7} := 
  by
  sorry

theorem problem_part_2 : (∀ a : ℝ, A a ∪ B = B → 2 < a ∧ a < sqrt 7) :=
  by 
  sorry

end problem_part_1_problem_part_2_l187_187662


namespace saly_needs_10_eggs_per_week_l187_187302

theorem saly_needs_10_eggs_per_week :
  let Saly_needs_per_week := S
  let Ben_needs_per_week := 14
  let Ked_needs_per_week := Ben_needs_per_week / 2
  let total_eggs_in_month := 124
  let weeks_per_month := 4
  let Ben_needs_per_month := Ben_needs_per_week * weeks_per_month
  let Ked_needs_per_month := Ked_needs_per_week * weeks_per_month
  let Saly_needs_per_month := total_eggs_in_month - (Ben_needs_per_month + Ked_needs_per_month)
  let S := Saly_needs_per_month / weeks_per_month
  Saly_needs_per_week = 10 :=
by
  sorry

end saly_needs_10_eggs_per_week_l187_187302


namespace domain_of_sqrt_1_minus_2_cos_l187_187653

theorem domain_of_sqrt_1_minus_2_cos (x : ℝ) (k : ℤ) :
  1 - 2 * Real.cos x ≥ 0 ↔ ∃ k : ℤ, (π / 3 + 2 * k * π ≤ x ∧ x ≤ 5 * π / 3 + 2 * k * π) :=
by
  sorry

end domain_of_sqrt_1_minus_2_cos_l187_187653


namespace bookstore_discount_l187_187987

noncomputable def discount_percentage (total_spent : ℝ) (over_22 : List ℝ) (under_20 : List ℝ) : ℝ :=
  let disc_over_22 := over_22.map (fun p => p * (1 - 0.30))
  let total_over_22 := disc_over_22.sum
  let total_with_under_20 := total_over_22 + 21
  let total_under_20 := under_20.sum
  let discount_received := total_spent - total_with_under_20
  let discount_percentage := (total_under_20 - discount_received) / total_under_20 * 100
  discount_percentage

theorem bookstore_discount :
  discount_percentage 95 [25.00, 35.00] [18.00, 12.00, 10.00] = 20 := by
  sorry

end bookstore_discount_l187_187987


namespace units_digit_of_fraction_example_l187_187500

def units_digit_of_fraction (n d : ℕ) : ℕ :=
  (n / d) % 10

theorem units_digit_of_fraction_example :
  units_digit_of_fraction (25 * 26 * 27 * 28 * 29 * 30) 1250 = 2 := by
  sorry

end units_digit_of_fraction_example_l187_187500


namespace inequality_proof_l187_187223

theorem inequality_proof
  (x y : ℝ) (h1 : x^2 + x * y + y^2 = (x + y)^2 - x * y) 
  (h2 : x + y ≥ 2 * Real.sqrt (x * y)) : 
  x + y + Real.sqrt (x * y) ≤ 3 * (x + y - Real.sqrt (x * y)) := 
by
  sorry

end inequality_proof_l187_187223


namespace leaves_problem_l187_187428

noncomputable def leaves_dropped_last_day (L : ℕ) (n : ℕ) : ℕ :=
  L - n * (L / 10)

theorem leaves_problem (L : ℕ) (n : ℕ) (h1 : L = 340) (h2 : leaves_dropped_last_day L n = 204) :
  n = 4 :=
by {
  sorry
}

end leaves_problem_l187_187428


namespace grasshopper_opposite_corner_moves_l187_187392

noncomputable def grasshopper_jump_count : ℕ :=
  Nat.factorial 27 / (Nat.factorial 9 * Nat.factorial 9 * Nat.factorial 9)

theorem grasshopper_opposite_corner_moves :
  grasshopper_jump_count = Nat.factorial 27 / (Nat.factorial 9 * Nat.factorial 9 * Nat.factorial 9) :=
by
  -- The detailed proof would go here.
  sorry

end grasshopper_opposite_corner_moves_l187_187392


namespace katy_books_l187_187992

theorem katy_books (june july aug : ℕ) (h1 : june = 8) (h2 : july = 2 * june) (h3 : june + july + aug = 37) :
  july - aug = 3 :=
by sorry

end katy_books_l187_187992


namespace circle_S_radius_properties_l187_187604

theorem circle_S_radius_properties :
  let DE := 120
  let DF := 120
  let EF := 68
  let R_radius := 20
  let S_radius := 52 - 6 * Real.sqrt 35
  let m := 52
  let n := 6
  let k := 35
  m + n * k = 262 := by
  sorry

end circle_S_radius_properties_l187_187604


namespace eq_expression_l187_187150

theorem eq_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
by
  sorry

end eq_expression_l187_187150


namespace chocolate_eggs_weeks_l187_187338

theorem chocolate_eggs_weeks (e: ℕ) (d: ℕ) (w: ℕ) (total: ℕ) (weeks: ℕ) 
    (initialEggs : e = 40)
    (dailyEggs : d = 2)
    (schoolDays : w = 5)
    (totalWeeks : weeks = total):
    total = e / (d * w) := by
sorry

end chocolate_eggs_weeks_l187_187338


namespace compound_interest_calculation_l187_187982

theorem compound_interest_calculation :
  let SI := (1833.33 * 16 * 6) / 100
  let CI := 2 * SI
  let principal_ci := 8000
  let rate_ci := 20
  let n := Real.log (1.4399995) / Real.log (1 + rate_ci / 100)
  n = 2 := by
  sorry

end compound_interest_calculation_l187_187982


namespace sequence_identity_l187_187887

noncomputable def a_n (n : ℕ) : ℝ := n + 1
noncomputable def b_n (n : ℕ) : ℝ := 2 * 3^(n-1)
noncomputable def S_n (n : ℕ) : ℝ := (n * (n+1)) / 2  -- Sum of first n terms of arithmetic sequence
noncomputable def T_n (n : ℕ) : ℝ := 2 * (3^n - 1) / 2  -- Sum of first n terms of geometric sequence
noncomputable def c_n (n : ℕ) : ℝ := 2 * a_n n / b_n n
noncomputable def C_n (n : ℕ) : ℝ := (15/2) - ((2 * n + 5) / (2 * 3^(n-1)))

theorem sequence_identity :
  a_n 1 = b_n 1 ∧
  2 * a_n 2 = b_n 2 ∧
  S_n 2 + T_n 2 = 13 ∧
  2 * S_n 3 = b_n 3 →
  (∀ n : ℕ, a_n n = n + 1 ∧ b_n n = 2 * 3^(n-1)) ∧
  (∀ n : ℕ, C_n n = (15/2) - ((2 * n + 5) / (2 * 3^(n-1)))) :=
sorry

end sequence_identity_l187_187887


namespace divide_payment_correctly_l187_187206

-- Define the number of logs contributed by each person
def logs_troikin : ℕ := 3
def logs_pyaterkin : ℕ := 5
def logs_bestoplivny : ℕ := 0

-- Define the total number of logs
def total_logs : ℕ := logs_troikin + logs_pyaterkin + logs_bestoplivny

-- Define the total number of logs used equally
def logs_per_person : ℚ := total_logs / 3

-- Define the total payment made by Bestoplivny 
def total_payment : ℕ := 80

-- Define the cost per log
def cost_per_log : ℚ := total_payment / logs_per_person

-- Define the contribution of each person to Bestoplivny
def bestoplivny_from_troikin : ℚ := logs_troikin - logs_per_person
def bestoplivny_from_pyaterkin : ℚ := logs_pyaterkin - (logs_per_person - bestoplivny_from_troikin)

-- Define the kopecks received by Troikina and Pyaterkin
def kopecks_troikin : ℚ := bestoplivny_from_troikin * cost_per_log
def kopecks_pyaterkin : ℚ := bestoplivny_from_pyaterkin * cost_per_log

-- Main theorem to prove the correct division of kopecks
theorem divide_payment_correctly : kopecks_troikin = 10 ∧ kopecks_pyaterkin = 70 :=
by
  -- ... Proof goes here
  sorry

end divide_payment_correctly_l187_187206


namespace evaluate_f_at_3_l187_187764

theorem evaluate_f_at_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 2 * x + 3) : f 3 = 7 :=
by
  -- proof goes here
  sorry

end evaluate_f_at_3_l187_187764


namespace percentage_markup_l187_187964

theorem percentage_markup (CP SP : ℕ) (hCP : CP = 800) (hSP : SP = 1000) :
  let Markup := SP - CP
  let PercentageMarkup := (Markup : ℚ) / CP * 100
  PercentageMarkup = 25 := by
  sorry

end percentage_markup_l187_187964


namespace convex_polyhedron_triangular_face_or_three_edges_vertex_l187_187559

theorem convex_polyhedron_triangular_face_or_three_edges_vertex
  (M N K : ℕ) 
  (euler_formula : N - M + K = 2) :
  ∃ (f : ℕ), (f ≤ N ∧ f = 3) ∨ ∃ (v : ℕ), (v ≤ K ∧ v = 3) := 
sorry

end convex_polyhedron_triangular_face_or_three_edges_vertex_l187_187559


namespace find_m_of_quad_roots_l187_187990

theorem find_m_of_quad_roots
  (a b : ℝ) (m : ℝ)
  (ha : a = 5)
  (hb : b = -4)
  (h_roots : ∀ x : ℂ, (x = (2 + Complex.I * Real.sqrt 143) / 5 ∨ x = (2 - Complex.I * Real.sqrt 143) / 5) →
                     (a * x^2 + b * x + m = 0)) :
  m = 7.95 :=
by
  -- Proof goes here
  sorry

end find_m_of_quad_roots_l187_187990


namespace probability_of_winning_l187_187911

-- Define the conditions
def total_tickets : ℕ := 10
def winning_tickets : ℕ := 3
def people : ℕ := 5
def losing_tickets : ℕ := total_tickets - winning_tickets

-- The probability calculation as per the conditions
def probability_at_least_one_wins : ℚ :=
  1 - ((Nat.choose losing_tickets people : ℚ) / (Nat.choose total_tickets people))

-- The statement to be proven
theorem probability_of_winning :
  probability_at_least_one_wins = 11 / 12 := 
sorry

end probability_of_winning_l187_187911


namespace hcf_two_numbers_l187_187981

theorem hcf_two_numbers (H a b : ℕ) (coprime_ab : Nat.gcd a b = 1) 
    (lcm_factors : a * b = 150) (larger_num : H * a = 450 ∨ H * b = 450) : H = 30 := 
by
  sorry

end hcf_two_numbers_l187_187981


namespace initial_distance_l187_187005

def relative_speed (v1 v2 : ℝ) : ℝ := v1 + v2

def total_distance (rel_speed time : ℝ) : ℝ := rel_speed * time

theorem initial_distance (v1 v2 time : ℝ) : (v1 = 1.6) → (v2 = 1.9) → 
                                            (time = 100) →
                                            total_distance (relative_speed v1 v2) time = 350 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp [relative_speed, total_distance]
  sorry

end initial_distance_l187_187005


namespace simplify_expr_l187_187972

theorem simplify_expr : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by 
  sorry

end simplify_expr_l187_187972


namespace c_sum_formula_l187_187773

noncomputable section

def arithmetic_sequence (a : Nat -> ℚ) : Prop :=
  a 3 = 2 ∧ (a 1 + 2 * ((a 2 - a 1) : ℚ)) = 2

def geometric_sequence (b : Nat -> ℚ) (a : Nat -> ℚ) : Prop :=
  b 1 = a 1 ∧ b 4 = a 15

def c_sequence (a : Nat -> ℚ) (b : Nat -> ℚ) (n : Nat) : ℚ :=
  a n + b n

def Tn (c : Nat -> ℚ) (n : Nat) : ℚ :=
  (Finset.range n).sum c

theorem c_sum_formula
  (a b c : Nat -> ℚ)
  (k : Nat) 
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b a)
  (hc : ∀ n, c n = c_sequence a b n) :
  Tn c k = k * (k + 3) / 4 + 2^k - 1 :=
by
  sorry

end c_sum_formula_l187_187773


namespace janine_read_pages_in_two_months_l187_187694

theorem janine_read_pages_in_two_months :
  (let books_last_month := 5
   let books_this_month := 2 * books_last_month
   let total_books := books_last_month + books_this_month
   let pages_per_book := 10
   total_books * pages_per_book = 150) := by
   sorry

end janine_read_pages_in_two_months_l187_187694


namespace arith_seq_sum_nine_l187_187042

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arith_seq := ∀ n : ℕ, a n = a 0 + (n - 1) * (a 1 - a 0)

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := 
  ∀ n : ℕ, S n = (n / 2) * (a 0 + a (n - 1))

theorem arith_seq_sum_nine (h_seq : arith_seq a) (h_sum : sum_first_n_terms a S) (h_S9 : S 9 = 18) : 
  a 2 + a 5 + a 8 = 6 :=
  sorry

end arith_seq_sum_nine_l187_187042


namespace value_of_k_l187_187122

theorem value_of_k (k : ℝ) (h1 : k ≠ 0) (h2 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → (k * x₁ - 100) < (k * x₂ - 100)) : k = 1 :=
by
  have h3 : k > 0 :=
    sorry -- We know that if y increases as x increases, then k > 0
  have h4 : k = 1 :=
    sorry -- For this specific problem, we can take k = 1 which satisfies the conditions
  exact h4

end value_of_k_l187_187122


namespace solve_for_x_l187_187530

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (-2, x)
def add_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def sub_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def is_parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

theorem solve_for_x : ∀ x : ℝ, is_parallel (add_vectors a (b x)) (sub_vectors a (b x)) → x = -4 :=
by
  intros x h_par
  sorry

end solve_for_x_l187_187530


namespace fraction_increase_each_year_l187_187895

variable (initial_value : ℝ := 57600)
variable (final_value : ℝ := 72900)
variable (years : ℕ := 2)

theorem fraction_increase_each_year :
  ∃ (f : ℝ), initial_value * (1 + f)^years = final_value ∧ f = 0.125 := by
  sorry

end fraction_increase_each_year_l187_187895


namespace eggs_given_by_Andrew_l187_187238

variable (total_eggs := 222)
variable (eggs_to_buy := 67)
variable (eggs_given : ℕ)

theorem eggs_given_by_Andrew :
  eggs_given = total_eggs - eggs_to_buy ↔ eggs_given = 155 := 
by 
  sorry

end eggs_given_by_Andrew_l187_187238


namespace remove_max_rooks_l187_187414

-- Defines the problem of removing the maximum number of rooks under given conditions
theorem remove_max_rooks (n : ℕ) (attacks_odd : (ℕ × ℕ) → ℕ) :
  (∀ p : ℕ × ℕ, (attacks_odd p) % 2 = 1 → true) →
  n = 8 →
  (∃ m, m = 59) :=
by
  intros _ _
  existsi 59
  sorry

end remove_max_rooks_l187_187414


namespace unique_solution_condition_l187_187520

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 4) ↔ d ≠ 4 :=
by
  sorry

end unique_solution_condition_l187_187520


namespace three_tenths_of_number_l187_187422

theorem three_tenths_of_number (x : ℝ) (h : (1/3) * (1/4) * x = 18) : (3/10) * x = 64.8 :=
sorry

end three_tenths_of_number_l187_187422


namespace train_crossing_time_l187_187551
-- Part a: Identifying the questions and conditions

-- Question: How long does it take for the train to cross the platform?
-- Conditions:
-- 1. Speed of the train: 72 km/hr
-- 2. Length of the goods train: 440 m
-- 3. Length of the platform: 80 m

-- Part b: Identifying the solution steps and the correct answers

-- The solution steps involve:
-- 1. Summing the lengths of the train and the platform to get the total distance the train needs to cover.
-- 2. Converting the speed of the train from km/hr to m/s.
-- 3. Using the formula Time = Distance / Speed to find the time.

-- Correct answer: 26 seconds

-- Part c: Translating the question, conditions, and correct answer to a mathematically equivalent proof problem

-- Goal: Prove that the time it takes for the train to cross the platform is 26 seconds given the provided conditions.

-- Part d: Writing the Lean 4 statement


-- Definitions based on the given conditions
def speed_kmh : ℕ := 72
def length_train : ℕ := 440
def length_platform : ℕ := 80

-- Definition based on the conversion step in the solution
def speed_ms : ℕ := (72 * 1000) / 3600 -- Converting speed from km/hr to m/s

-- Goal: Prove that the time it takes for the train to cross the platform is 26 seconds
theorem train_crossing_time : ((length_train + length_platform) : ℕ) / speed_ms = 26 := by
  sorry

end train_crossing_time_l187_187551


namespace find_primes_l187_187089

theorem find_primes (p : ℕ) (x y : ℕ) (hx : x > 0) (hy : y > 0) (hp : Nat.Prime p) : 
  (x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ (p = 2 ∨ p = 3 ∨ p = 7) := sorry

end find_primes_l187_187089


namespace XY_sym_diff_l187_187646

-- The sets X and Y
def X : Set ℤ := {1, 3, 5, 7}
def Y : Set ℤ := { x | x < 4 ∧ x ∈ Set.univ }

-- Definition of set operation (A - B)
def set_sub (A B : Set ℤ) : Set ℤ := { x | x ∈ A ∧ x ∉ B }

-- Definition of set operation (A * B)
def set_sym_diff (A B : Set ℤ) : Set ℤ := (set_sub A B) ∪ (set_sub B A)

-- Prove that X * Y = {-3, -2, -1, 0, 2, 5, 7}
theorem XY_sym_diff : set_sym_diff X Y = {-3, -2, -1, 0, 2, 5, 7} :=
by
  sorry

end XY_sym_diff_l187_187646


namespace net_population_change_l187_187380

theorem net_population_change (P : ℝ) : 
  let P1 := P * (6/5)
  let P2 := P1 * (7/10)
  let P3 := P2 * (6/5)
  let P4 := P3 * (7/10)
  (P4 / P - 1) * 100 = -29 := 
by
  sorry

end net_population_change_l187_187380


namespace second_eq_value_l187_187125

variable (x y z w : ℝ)

theorem second_eq_value (h1 : 4 * x * z + y * w = 3) (h2 : (2 * x + y) * (2 * z + w) = 15) : 
  x * w + y * z = 6 :=
by
  sorry

end second_eq_value_l187_187125


namespace system1_solution_system2_solution_l187_187143

-- For System (1)
theorem system1_solution (x y : ℝ) (h1 : y = 2 * x) (h2 : 3 * y + 2 * x = 8) : x = 1 ∧ y = 2 :=
by
  sorry

-- For System (2)
theorem system2_solution (s t : ℝ) (h1 : 2 * s - 3 * t = 2) (h2 : (s + 2 * t) / 3 = 3 / 2) : s = 5 / 2 ∧ t = 1 :=
by
  sorry

end system1_solution_system2_solution_l187_187143


namespace solution_set_quadratic_inequality_l187_187117

def quadraticInequalitySolutionSet 
  (x : ℝ) : Prop := 
  3 + 5 * x - 2 * x^2 > 0

theorem solution_set_quadratic_inequality :
  { x : ℝ | quadraticInequalitySolutionSet x } = 
  { x : ℝ | - (1:ℝ) / 2 < x ∧ x < 3 } :=
by {
  sorry
}

end solution_set_quadratic_inequality_l187_187117


namespace Taehyung_mother_age_l187_187795

theorem Taehyung_mother_age (Taehyung_young_brother_age : ℕ) (Taehyung_age_diff : ℕ) (Mother_age_diff : ℕ) (H1 : Taehyung_young_brother_age = 7) (H2 : Taehyung_age_diff = 5) (H3 : Mother_age_diff = 31) :
  ∃ (Mother_age : ℕ), Mother_age = 43 := 
by
  have Taehyung_age : ℕ := Taehyung_young_brother_age + Taehyung_age_diff
  have Mother_age := Taehyung_age + Mother_age_diff
  existsi (Mother_age)
  sorry

end Taehyung_mother_age_l187_187795


namespace simplify_and_evaluate_l187_187899

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (1 + 1 / a) / ((a^2 - 1) / a) = (Real.sqrt 2 / 2) :=
by
  sorry

end simplify_and_evaluate_l187_187899


namespace rectangular_prism_volume_l187_187177

theorem rectangular_prism_volume 
(l w h : ℝ) 
(h1 : l * w = 18) 
(h2 : w * h = 32) 
(h3 : l * h = 48) : 
l * w * h = 288 :=
sorry

end rectangular_prism_volume_l187_187177


namespace valbonne_middle_school_l187_187250

theorem valbonne_middle_school (students : Finset ℕ) (h : students.card = 367) :
  ∃ (date1 date2 : ℕ), date1 ≠ date2 ∧ date1 = date2 ∧ date1 ∈ students ∧ date2 ∈ students :=
by {
  sorry
}

end valbonne_middle_school_l187_187250


namespace total_black_dots_l187_187136

def num_butterflies : ℕ := 397
def black_dots_per_butterfly : ℕ := 12

theorem total_black_dots : num_butterflies * black_dots_per_butterfly = 4764 := by
  sorry

end total_black_dots_l187_187136


namespace hyperbola_asymptotes_angle_l187_187593

theorem hyperbola_asymptotes_angle {a b : ℝ} (h₁ : a > b) 
  (h₂ : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) 
  (h₃ : ∀ θ : ℝ, θ = Real.pi / 4) : a / b = Real.sqrt 2 :=
by
  sorry

end hyperbola_asymptotes_angle_l187_187593


namespace center_of_circle_l187_187484

theorem center_of_circle : 
  ∀ x y : ℝ, 4 * x^2 + 8 * x + 4 * y^2 - 12 * y + 29 = 0 → (x = -1 ∧ y = 3 / 2) :=
by
  sorry

end center_of_circle_l187_187484


namespace proof_problem_l187_187967

variable {a_n : ℕ → ℤ}
variable {b_n : ℕ → ℤ}
variable {c_n : ℕ → ℤ}
variable {T_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}

-- Conditions

-- 1. The common difference d of the arithmetic sequence {a_n} is greater than 0
def common_difference_positive (d : ℤ) : Prop :=
  d > 0

-- 2. a_2 and a_5 are the two roots of the equation x^2 - 12x + 27 = 0
def roots_of_quadratic (a2 a5 : ℤ) : Prop :=
  a2^2 - 12 * a2 + 27 = 0 ∧ a5^2 - 12 * a5 + 27 = 0

-- 3. The sum of the first n terms of the sequence {b_n} is S_n, and it is given that S_n = (3 / 2)(b_n - 1)
def sum_of_b_n (S_n b_n : ℕ → ℤ) : Prop :=
  ∀ n, S_n n = 3/2 * (b_n n - 1)

-- Define the sequences to display further characteristics

-- 1. Find the general formula for the sequences {a_n} and {b_n}
def general_formula_a (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = 2 * n - 1

def general_formula_b (b : ℕ → ℤ) : Prop :=
  ∀ n, b n = 3 ^ n

-- 2. Check if c_n = a_n * b_n and find the sum T_n
def c_n_equals_a_n_times_b_n (a b : ℕ → ℤ) (c : ℕ → ℤ) : Prop :=
  ∀ n, c n = a n * b n

def sum_T_n (T c : ℕ → ℤ) : Prop :=
  ∀ n, T n = 3 + (n - 1) * 3^(n + 1)

theorem proof_problem 
  (d : ℤ)
  (a2 a5 : ℤ)
  (S_n b_n : ℕ → ℤ)
  (a_n b_n c_n T_n : ℕ → ℤ) :
  common_difference_positive d ∧
  roots_of_quadratic a2 a5 ∧ 
  sum_of_b_n S_n b_n ∧ 
  general_formula_a a_n ∧ 
  general_formula_b b_n ∧ 
  c_n_equals_a_n_times_b_n a_n b_n c_n ∧ 
  sum_T_n T_n c_n :=
sorry

end proof_problem_l187_187967


namespace reggie_marbles_l187_187704

/-- Given that Reggie and his friend played 9 games in total,
    Reggie lost 1 game, and they bet 10 marbles per game.
    Prove that Reggie has 70 marbles after all games. -/
theorem reggie_marbles (total_games : ℕ) (lost_games : ℕ) (marbles_per_game : ℕ) (marbles_initial : ℕ) 
  (h_total_games : total_games = 9) (h_lost_games : lost_games = 1) (h_marbles_per_game : marbles_per_game = 10) 
  (h_marbles_initial : marbles_initial = 0) : 
  marbles_initial + (total_games - lost_games) * marbles_per_game - lost_games * marbles_per_game = 70 :=
by
  -- We proved this in the solution steps, but will skip the proof here with sorry.
  sorry

end reggie_marbles_l187_187704


namespace polynomial_equality_l187_187324

def P (x : ℝ) : ℝ := x ^ 3 - 3 * x ^ 2 - 3 * x - 1

noncomputable def x1 : ℝ := 1 - Real.sqrt 2
noncomputable def x2 : ℝ := 1 + Real.sqrt 2
noncomputable def x3 : ℝ := 1 - 2 * Real.sqrt 2
noncomputable def x4 : ℝ := 1 + 2 * Real.sqrt 2

theorem polynomial_equality :
  P x1 + P x2 = P x3 + P x4 :=
sorry

end polynomial_equality_l187_187324


namespace cats_not_liking_catnip_or_tuna_l187_187947

theorem cats_not_liking_catnip_or_tuna :
  ∀ (total_cats catnip_lovers tuna_lovers both_lovers : ℕ),
  total_cats = 80 →
  catnip_lovers = 15 →
  tuna_lovers = 60 →
  both_lovers = 10 →
  (total_cats - (catnip_lovers - both_lovers + both_lovers + tuna_lovers - both_lovers)) = 15 :=
by
  intros total_cats catnip_lovers tuna_lovers both_lovers ht hc ht hboth
  sorry

end cats_not_liking_catnip_or_tuna_l187_187947


namespace dynamic_load_L_value_l187_187545

theorem dynamic_load_L_value (T H : ℝ) (hT : T = 3) (hH : H = 6) : 
  (L : ℝ) = (50 * T^3) / (H^3) -> L = 6.25 := 
by 
  sorry 

end dynamic_load_L_value_l187_187545


namespace possible_values_of_m_l187_187264

open Set

variable (A B : Set ℤ)
variable (m : ℤ)

theorem possible_values_of_m (h₁ : A = {1, 2, m * m}) (h₂ : B = {1, m}) (h₃ : B ⊆ A) :
  m = 0 ∨ m = 2 :=
  sorry

end possible_values_of_m_l187_187264


namespace division_multiplication_l187_187216

theorem division_multiplication : (0.25 / 0.005) * 2 = 100 := 
by 
  sorry

end division_multiplication_l187_187216


namespace tim_total_points_l187_187601

-- Definitions based on the conditions
def points_single : ℕ := 1000
def points_tetris : ℕ := 8 * points_single
def singles_scored : ℕ := 6
def tetrises_scored : ℕ := 4

-- Theorem stating the total points scored by Tim
theorem tim_total_points : singles_scored * points_single + tetrises_scored * points_tetris = 38000 := by
  sorry

end tim_total_points_l187_187601


namespace sequence_periodicity_l187_187256

variable {a b : ℕ → ℤ}

theorem sequence_periodicity (h : ∀ n ≥ 3, 
    (a n - a (n - 1)) * (a n - a (n - 2)) + 
    (b n - b (n - 1)) * (b n - b (n - 2)) = 0) : 
    ∃ k > 0, a k + b k = a (k + 2018) + b (k + 2018) := 
    by
    sorry

end sequence_periodicity_l187_187256


namespace atlantic_call_charge_l187_187831

theorem atlantic_call_charge :
  let united_base := 6.00
  let united_per_min := 0.25
  let atlantic_base := 12.00
  let same_bill_minutes := 120
  let atlantic_total (charge_per_minute : ℝ) := atlantic_base + charge_per_minute * same_bill_minutes
  let united_total := united_base + united_per_min * same_bill_minutes
  united_total = atlantic_total 0.20 :=
by
  sorry

end atlantic_call_charge_l187_187831


namespace tiffany_found_bags_l187_187077

theorem tiffany_found_bags (initial_bags : ℕ) (total_bags : ℕ) (found_bags : ℕ) :
  initial_bags = 4 ∧ total_bags = 12 ∧ total_bags = initial_bags + found_bags → found_bags = 8 :=
by
  sorry

end tiffany_found_bags_l187_187077


namespace odd_function_has_zero_l187_187671

variable {R : Type} [LinearOrderedField R]

def is_odd_function (f : R → R) := ∀ x : R, f (-x) = -f x

theorem odd_function_has_zero {f : R → R} (h : is_odd_function f) : ∃ x : R, f x = 0 :=
sorry

end odd_function_has_zero_l187_187671


namespace length_OP_l187_187191

noncomputable def right_triangle_length_OP (XY XZ YZ : ℝ) (rO rP : ℝ) : ℝ :=
  let O := rO
  let P := rP
  -- Coordinates of point Y and Z can be O = (0, r), P = (OP, r)
  25 -- directly from the given correct answer

theorem length_OP (XY XZ YZ : ℝ) (rO rP : ℝ) (hXY : XY = 7) (hXZ : XZ = 24) (hYZ : YZ = 25) 
  (hO : rO = YZ - rO) (hP : rP = YZ - rP) : 
  right_triangle_length_OP XY XZ YZ rO rP = 25 :=
sorry

end length_OP_l187_187191


namespace solve_for_y_l187_187688

theorem solve_for_y (y : ℚ) : 
  y + 1 / 3 = 3 / 8 - 1 / 4 → y = -5 / 24 := 
by
  sorry

end solve_for_y_l187_187688


namespace pieces_info_at_most_two_identical_digits_l187_187692

def num_pieces_of_information_with_at_most_two_positions_as_0110 : Nat :=
  (Nat.choose 4 2 + Nat.choose 4 1 + Nat.choose 4 0)

theorem pieces_info_at_most_two_identical_digits :
  num_pieces_of_information_with_at_most_two_positions_as_0110 = 11 :=
by
  sorry

end pieces_info_at_most_two_identical_digits_l187_187692


namespace parallel_vectors_eq_l187_187287

theorem parallel_vectors_eq (t : ℝ) : ∀ (m n : ℝ × ℝ), m = (2, 8) → n = (-4, t) → (∃ k : ℝ, n = k • m) → t = -16 :=
by 
  intros m n hm hn h_parallel
  -- proof goes here
  sorry

end parallel_vectors_eq_l187_187287


namespace ratio_of_discounted_bricks_l187_187809

theorem ratio_of_discounted_bricks (total_bricks discounted_price full_price total_spending: ℝ) 
  (h1 : total_bricks = 1000) 
  (h2 : discounted_price = 0.25) 
  (h3 : full_price = 0.50) 
  (h4 : total_spending = 375) : 
  ∃ D : ℝ, (D / total_bricks = 1 / 2) ∧ (0.25 * D + 0.50 * (total_bricks - D) = total_spending) := 
  sorry

end ratio_of_discounted_bricks_l187_187809


namespace total_soldiers_correct_l187_187518

-- Definitions based on conditions
def num_generals := 8
def num_vanguards := 8^2
def num_flags := 8^3
def num_team_leaders := 8^4
def num_armored_soldiers := 8^5
def num_soldiers := 8 + 8^2 + 8^3 + 8^4 + 8^5 + 8^6

-- Prove total number of soldiers
theorem total_soldiers_correct : num_soldiers = (1 / 7 : ℝ) * (8^7 - 8) := by
  sorry

end total_soldiers_correct_l187_187518


namespace fraction_equality_l187_187466

theorem fraction_equality (x y : ℝ) : (-x + y) / (-x - y) = (x - y) / (x + y) :=
by sorry

end fraction_equality_l187_187466


namespace sequence_constant_l187_187374

theorem sequence_constant
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : ∀ n, Nat.Prime (Int.natAbs (a n)))
  (h2 : ∀ n, a (n + 2) = a (n + 1) + a n + d) :
  ∃ c : ℤ, ∀ n, a n = c :=
by
  sorry

end sequence_constant_l187_187374


namespace find_symmetric_point_l187_187928

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def line_equation (t : ℝ) : Point :=
  { x := -t, y := 1.5, z := 2 + t }

def M : Point := { x := -1, y := 0, z := -1 }

def is_midpoint (M M' M0 : Point) : Prop :=
  M0.x = (M.x + M'.x) / 2 ∧
  M0.y = (M.y + M'.y) / 2 ∧
  M0.z = (M.z + M'.z) / 2

theorem find_symmetric_point (M0 : Point) (h_line : ∃ t, M0 = line_equation t) :
  ∃ M' : Point, is_midpoint M M' M0 ∧ M' = { x := 3, y := 3, z := 3 } :=
sorry

end find_symmetric_point_l187_187928


namespace total_dollars_l187_187139

def mark_dollars : ℚ := 4 / 5
def carolyn_dollars : ℚ := 2 / 5
def jack_dollars : ℚ := 1 / 2

theorem total_dollars :
  mark_dollars + carolyn_dollars + jack_dollars = 1.7 := 
sorry

end total_dollars_l187_187139


namespace bus_speed_excluding_stoppages_l187_187418

theorem bus_speed_excluding_stoppages (S : ℝ) (h₀ : 0 < S) (h₁ : 36 = (2/3) * S) : S = 54 :=
by 
  sorry

end bus_speed_excluding_stoppages_l187_187418


namespace shorter_side_length_l187_187906

theorem shorter_side_length (a b : ℕ) (h1 : 2 * a + 2 * b = 42) (h2 : a * b = 108) : b = 9 :=
by
  sorry

end shorter_side_length_l187_187906


namespace max_pies_without_ingredients_l187_187988

theorem max_pies_without_ingredients
  (total_pies chocolate_pies berries_pies cinnamon_pies poppy_seeds_pies : ℕ)
  (h1 : total_pies = 60)
  (h2 : chocolate_pies = 1 / 3 * total_pies)
  (h3 : berries_pies = 3 / 5 * total_pies)
  (h4 : cinnamon_pies = 1 / 2 * total_pies)
  (h5 : poppy_seeds_pies = 1 / 5 * total_pies) : 
  total_pies - max chocolate_pies (max berries_pies (max cinnamon_pies poppy_seeds_pies)) = 24 := 
by
  sorry

end max_pies_without_ingredients_l187_187988


namespace min_value_m_n_l187_187004

theorem min_value_m_n (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_geom_mean : a * b = 4)
    (m n : ℝ) (h_m : m = b + 1 / a) (h_n : n = a + 1 / b) : m + n ≥ 5 :=
by
  sorry

end min_value_m_n_l187_187004


namespace alyssa_initial_puppies_l187_187699

theorem alyssa_initial_puppies (gave_away has_left : ℝ) (h1 : gave_away = 8.5) (h2 : has_left = 12.5) :
    (gave_away + has_left = 21) :=
by
    sorry

end alyssa_initial_puppies_l187_187699


namespace number_of_hardbacks_l187_187434

theorem number_of_hardbacks (H P : ℕ) (books total_books selections : ℕ) (comb : ℕ → ℕ → ℕ) :
  total_books = 8 →
  P = 2 →
  comb total_books 3 - comb H 3 = 36 →
  H = 6 :=
by sorry

end number_of_hardbacks_l187_187434


namespace square_side_measurement_error_l187_187021

theorem square_side_measurement_error (S S' : ℝ) (h1 : S' = S * Real.sqrt 1.0404) : 
  (S' - S) / S * 100 = 2 :=
by
  sorry

end square_side_measurement_error_l187_187021


namespace no_nonzero_integer_solution_l187_187297

theorem no_nonzero_integer_solution (x y z : ℤ) (h : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :
  x^2 + y^2 ≠ 3 * z^2 :=
by
  sorry

end no_nonzero_integer_solution_l187_187297


namespace problem_statement_l187_187307

theorem problem_statement (m n c d a : ℝ)
  (h1 : m = -n)
  (h2 : c * d = 1)
  (h3 : a = 2) :
  Real.sqrt (c * d) + 2 * (m + n) - a = -1 :=
by
  -- Proof steps are skipped with sorry 
  sorry

end problem_statement_l187_187307


namespace harriet_trip_time_to_B_l187_187841

variables (D : ℝ) (t1 t2 : ℝ)

-- Definitions based on the given problem
def speed_to_b_town := 100
def speed_to_a_ville := 150
def total_time := 5

-- The condition for the total time for the trip
def total_trip_time_eq := t1 / speed_to_b_town + t2 / speed_to_a_ville = total_time

-- Prove that the time Harriet took to drive from A-ville to B-town is 3 hours.
theorem harriet_trip_time_to_B (h : total_trip_time_eq D D) : t1 = 3 :=
sorry

end harriet_trip_time_to_B_l187_187841


namespace num_orders_javier_constraint_l187_187454

noncomputable def num_valid_orders : ℕ :=
  Nat.factorial 5 / 2

theorem num_orders_javier_constraint : num_valid_orders = 60 := 
by
  sorry

end num_orders_javier_constraint_l187_187454


namespace average_speed_trip_l187_187914

theorem average_speed_trip 
  (total_distance : ℕ)
  (first_distance : ℕ)
  (first_speed : ℕ)
  (second_distance : ℕ)
  (second_speed : ℕ)
  (h1 : total_distance = 60)
  (h2 : first_distance = 30)
  (h3 : first_speed = 60)
  (h4 : second_distance = 30)
  (h5 : second_speed = 30) :
  40 = total_distance / ((first_distance / first_speed) + (second_distance / second_speed)) :=
by sorry

end average_speed_trip_l187_187914


namespace largest_int_less_150_gcd_18_eq_6_l187_187633

theorem largest_int_less_150_gcd_18_eq_6 : ∃ (n : ℕ), n < 150 ∧ gcd n 18 = 6 ∧ ∀ (m : ℕ), m < 150 ∧ gcd m 18 = 6 → m ≤ n ∧ n = 138 := 
by
  sorry

end largest_int_less_150_gcd_18_eq_6_l187_187633


namespace only_n_equal_3_exists_pos_solution_l187_187680

theorem only_n_equal_3_exists_pos_solution :
  ∀ (n : ℕ), (∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2) ↔ n = 3 := 
by
  sorry

end only_n_equal_3_exists_pos_solution_l187_187680


namespace inequality_abc_l187_187553

variable {a b c : ℝ}

-- Assume a, b, c are positive real numbers
def positive_real_numbers (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Assume the sum of any two numbers is greater than the third
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Lean 4 statement for the proof problem
theorem inequality_abc (h1 : positive_real_numbers a b c) (h2 : triangle_inequality a b c) :
  abc ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
sorry

end inequality_abc_l187_187553


namespace cost_price_to_selling_price_ratio_l187_187286

variable (CP SP : ℝ)
variable (profit_percent : ℝ)

theorem cost_price_to_selling_price_ratio
  (h1 : profit_percent = 0.25)
  (h2 : SP = (1 + profit_percent) * CP) :
  (CP / SP) = 4 / 5 := by
  sorry

end cost_price_to_selling_price_ratio_l187_187286


namespace amount_after_3_years_l187_187237

theorem amount_after_3_years (P t A' : ℝ) (R : ℝ) :
  P = 800 → t = 3 → A' = 992 →
  (800 * ((R + 3) / 100) * 3 = 192) →
  (A = P * (1 + (R / 100) * t)) →
  A = 1160 := by
  intros hP ht hA' hR hA
  sorry

end amount_after_3_years_l187_187237


namespace range_of_m_l187_187572

def f (a b x : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

def f_prime (a b x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + b

def has_local_extremum_at (a b x : ℝ) : Prop :=
  f_prime a b x = 0 ∧ f a b x = 0

def h (a b m x : ℝ) : ℝ := f a b x - m + 1

theorem range_of_m (a b m : ℝ) :
  (has_local_extremum_at 2 9 (-1) ∧
   ∀ x, f 2 9 x = x^3 + 6 * x^2 + 9 * x + 4) →
  (∀ x, (x^3 + 6 * x^2 + 9 * x + 4 - m + 1 = 0) → 
  1 < m ∧ m < 5) := 
sorry

end range_of_m_l187_187572


namespace RebeccaHasTwentyMarbles_l187_187205

variable (groups : ℕ) (marbles_per_group : ℕ) (total_marbles : ℕ)

def totalMarbles (g m : ℕ) : ℕ :=
  g * m

theorem RebeccaHasTwentyMarbles
  (h1 : groups = 5)
  (h2 : marbles_per_group = 4)
  (h3 : total_marbles = totalMarbles groups marbles_per_group) :
  total_marbles = 20 :=
by {
  sorry
}

end RebeccaHasTwentyMarbles_l187_187205


namespace total_miles_traveled_l187_187497

-- Define the conditions
def travel_time_per_mile (n : ℕ) : ℕ :=
  match n with
  | 0 => 10
  | _ => 10 + 6 * n

def daily_miles (n : ℕ) : ℕ :=
  60 / travel_time_per_mile n

-- Statement of the problem
theorem total_miles_traveled : (daily_miles 0 + daily_miles 1 + daily_miles 2 + daily_miles 3 + daily_miles 4) = 20 := by
  sorry

end total_miles_traveled_l187_187497


namespace total_balloons_after_destruction_l187_187853

-- Define the initial numbers of balloons
def fredBalloons := 10.0
def samBalloons := 46.0
def destroyedBalloons := 16.0

-- Prove the total number of remaining balloons
theorem total_balloons_after_destruction : fredBalloons + samBalloons - destroyedBalloons = 40.0 :=
by
  sorry

end total_balloons_after_destruction_l187_187853


namespace piravena_total_round_trip_cost_l187_187901

noncomputable def piravena_round_trip_cost : ℝ :=
  let distance_AB := 4000
  let bus_cost_per_km := 0.20
  let flight_cost_per_km := 0.12
  let flight_booking_fee := 120
  let flight_cost := distance_AB * flight_cost_per_km + flight_booking_fee
  let bus_cost := distance_AB * bus_cost_per_km
  flight_cost + bus_cost

theorem piravena_total_round_trip_cost : piravena_round_trip_cost = 1400 := by
  -- Problem conditions for reference:
  -- distance_AC = 3000
  -- distance_AB = 4000
  -- bus_cost_per_km = 0.20
  -- flight_cost_per_km = 0.12
  -- flight_booking_fee = 120
  -- Piravena decides to fly from A to B but returns by bus
  sorry

end piravena_total_round_trip_cost_l187_187901


namespace Dave_guitar_strings_replacement_l187_187904

theorem Dave_guitar_strings_replacement :
  (2 * 6 * 12) = 144 := by
  sorry

end Dave_guitar_strings_replacement_l187_187904


namespace john_total_shirts_l187_187830

-- Define initial conditions
def initial_shirts : ℕ := 12
def additional_shirts : ℕ := 4

-- Statement of the problem
theorem john_total_shirts : initial_shirts + additional_shirts = 16 := by
  sorry

end john_total_shirts_l187_187830


namespace total_cost_of_bill_l187_187487

def original_price_curtis := 16.00
def original_price_rob := 18.00
def time_of_meal := 3

def is_early_bird_discount_applicable (time : ℕ) : Prop :=
  2 ≤ time ∧ time ≤ 4

theorem total_cost_of_bill :
  is_early_bird_discount_applicable time_of_meal →
  original_price_curtis / 2 + original_price_rob / 2 = 17.00 :=
by
  sorry

end total_cost_of_bill_l187_187487


namespace roots_of_quadratic_sum_of_sixth_powers_l187_187792

theorem roots_of_quadratic_sum_of_sixth_powers {u v : ℝ} 
  (h₀ : u^2 - 2*u*Real.sqrt 3 + 1 = 0)
  (h₁ : v^2 - 2*v*Real.sqrt 3 + 1 = 0)
  : u^6 + v^6 = 970 := 
by 
  sorry

end roots_of_quadratic_sum_of_sixth_powers_l187_187792


namespace area_of_equilateral_triangle_inscribed_in_square_l187_187485

variables {a : ℝ}

noncomputable def equilateral_triangle_area (a : ℝ) : ℝ :=
  a^2 * (2 * Real.sqrt 3 - 3)

theorem area_of_equilateral_triangle_inscribed_in_square (a : ℝ) :
  equilateral_triangle_area a = a^2 * (2 * Real.sqrt 3 - 3) :=
by sorry

end area_of_equilateral_triangle_inscribed_in_square_l187_187485


namespace find_value_of_a2_b2_c2_l187_187963

variable {a b c : ℝ}

theorem find_value_of_a2_b2_c2
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 := 
sorry

end find_value_of_a2_b2_c2_l187_187963


namespace find_principal_l187_187951

variable (P : ℝ) (r : ℝ) (t : ℕ) (CI : ℝ) (SI : ℝ)

-- Define simple and compound interest
def simple_interest (P r : ℝ) (t : ℕ) : ℝ := P * r * t
def compound_interest (P r : ℝ) (t : ℕ) : ℝ := P * (1 + r)^t - P

-- Given conditions
axiom H1 : r = 0.05
axiom H2 : t = 2
axiom H3 : compound_interest P r t - simple_interest P r t = 18

-- The principal sum is 7200
theorem find_principal : P = 7200 := 
by sorry

end find_principal_l187_187951


namespace students_no_A_l187_187543

theorem students_no_A
  (total_students : ℕ)
  (A_in_history : ℕ)
  (A_in_math : ℕ)
  (A_in_science : ℕ)
  (A_in_history_and_math : ℕ)
  (A_in_history_and_science : ℕ)
  (A_in_math_and_science : ℕ)
  (A_in_all_three : ℕ)
  (h_total_students : total_students = 40)
  (h_A_in_history : A_in_history = 10)
  (h_A_in_math : A_in_math = 15)
  (h_A_in_science : A_in_science = 8)
  (h_A_in_history_and_math : A_in_history_and_math = 5)
  (h_A_in_history_and_science : A_in_history_and_science = 3)
  (h_A_in_math_and_science : A_in_math_and_science = 4)
  (h_A_in_all_three : A_in_all_three = 2) :
  total_students - (A_in_history + A_in_math + A_in_science 
    - A_in_history_and_math - A_in_history_and_science - A_in_math_and_science 
    + A_in_all_three) = 17 := 
sorry

end students_no_A_l187_187543


namespace gasoline_price_increase_l187_187675

theorem gasoline_price_increase :
  ∀ (p_low p_high : ℝ), p_low = 14 → p_high = 23 → 
  ((p_high - p_low) / p_low) * 100 = 64.29 :=
by
  intro p_low p_high h_low h_high
  rw [h_low, h_high]
  sorry

end gasoline_price_increase_l187_187675


namespace employed_population_percentage_l187_187883

variable (P : ℝ) -- Total population
variable (percentage_employed_to_population : ℝ) -- Percentage of total population employed
variable (percentage_employed_males_to_population : ℝ := 0.42) -- 42% of population are employed males
variable (percentage_employed_females_to_employed : ℝ := 0.30) -- 30% of employed people are females

theorem employed_population_percentage :
  percentage_employed_to_population = 0.60 :=
sorry

end employed_population_percentage_l187_187883


namespace average_salary_proof_l187_187705

noncomputable def average_salary_of_all_workers (tech_workers : ℕ) (tech_avg_sal : ℕ) (total_workers : ℕ) (non_tech_avg_sal : ℕ) : ℕ :=
  let non_tech_workers := total_workers - tech_workers
  let total_tech_salary := tech_workers * tech_avg_sal
  let total_non_tech_salary := non_tech_workers * non_tech_avg_sal
  let total_salary := total_tech_salary + total_non_tech_salary
  total_salary / total_workers

theorem average_salary_proof : average_salary_of_all_workers 7 14000 28 6000 = 8000 := by
  sorry

end average_salary_proof_l187_187705


namespace factorization_1_factorization_2_factorization_3_factorization_4_l187_187743

-- Problem 1
theorem factorization_1 (a b : ℝ) : 
  4 * a^2 + 12 * a * b + 9 * b^2 = (2 * a + 3 * b)^2 :=
by sorry

-- Problem 2
theorem factorization_2 (a b : ℝ) : 
  16 * a^2 * (a - b) + 4 * b^2 * (b - a) = 4 * (a - b) * (2 * a - b) * (2 * a + b) :=
by sorry

-- Problem 3
theorem factorization_3 (m n : ℝ) : 
  25 * (m + n)^2 - 9 * (m - n)^2 = 4 * (4 * m + n) * (m + 4 * n) :=
by sorry

-- Problem 4
theorem factorization_4 (a b : ℝ) : 
  4 * a^2 - b^2 - 4 * a + 1 = (2 * a - 1 + b) * (2 * a - 1 - b) :=
by sorry

end factorization_1_factorization_2_factorization_3_factorization_4_l187_187743


namespace problem1_problem2_problem3_problem4_l187_187241

-- Problem 1
theorem problem1 : ∃ n : ℕ, n = 3^4 ∧ n = 81 :=
by
  sorry

-- Problem 2
theorem problem2 : ∃ n : ℕ, n = (Nat.choose 4 2) * 6 ∧ n = 36 :=
by
  sorry

-- Problem 3
theorem problem3 : ∃ n : ℕ, n = Nat.choose 4 2 ∧ n = 6 :=
by
  sorry

-- Problem 4
theorem problem4 : ∃ n : ℕ, n = 1 + (Nat.choose 4 1 + Nat.choose 4 2 / 2) + 6 ∧ n = 14 :=
by
  sorry

end problem1_problem2_problem3_problem4_l187_187241


namespace ad_eb_intersect_on_altitude_l187_187347

open EuclideanGeometry

variables {A B C D E F G K L C1 : Point}

-- Definitions for the problem
variables (triangleABC : Triangle A B C)
  (squareAEFC : Square A E F C)
  (squareBDGC : Square B D G C)
  (altitudeCC1 : Line C C1)
  (lineDA : Line A D)
  (lineEB : Line B E)

-- Definition of intersection
def intersects_on_altitude (pt : Point) : Prop :=
  pt ∈ lineDA ∧ pt ∈ lineEB ∧ pt ∈ altitudeCC1

-- The theorem to be proved
theorem ad_eb_intersect_on_altitude : 
  ∃ pt : Point, intersects_on_altitude lineDA lineEB altitudeCC1 pt := 
sorry

end ad_eb_intersect_on_altitude_l187_187347


namespace complement_intersection_eq_l187_187966

open Set

def P : Set ℝ := { x | x^2 - 2 * x ≥ 0 }
def Q : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

theorem complement_intersection_eq :
  (compl P) ∩ Q = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end complement_intersection_eq_l187_187966


namespace remainder_of_M_l187_187943

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l187_187943


namespace correct_answer_l187_187379

variables (A B : polynomial ℝ) (a : ℝ)

theorem correct_answer (hB : B = 3 * a^2 - 5 * a - 7) (hMistake : A - 2 * B = -2 * a^2 + 3 * a + 6) :
  A + 2 * B = 10 * a^2 - 17 * a - 22 :=
by
  sorry

end correct_answer_l187_187379


namespace length_of_second_train_l187_187486

/-
  Given:
  - l₁ : Length of the first train in meters
  - v₁ : Speed of the first train in km/h
  - v₂ : Speed of the second train in km/h
  - t : Time to cross the second train in seconds

  Prove:
  - l₂ : Length of the second train in meters = 299.9560035197185 meters
-/

variable (l₁ : ℝ) (v₁ : ℝ) (v₂ : ℝ) (t : ℝ) (l₂ : ℝ)

theorem length_of_second_train
  (h₁ : l₁ = 250)
  (h₂ : v₁ = 72)
  (h₃ : v₂ = 36)
  (h₄ : t = 54.995600351971845)
  (h_result : l₂ = 299.9560035197185) :
  (v₁ * 1000 / 3600 - v₂ * 1000 / 3600) * t - l₁ = l₂ := by
  sorry

end length_of_second_train_l187_187486


namespace amanda_final_quiz_score_l187_187440

theorem amanda_final_quiz_score
  (average_score_4quizzes : ℕ)
  (total_quizzes : ℕ)
  (average_a : ℕ)
  (current_score : ℕ)
  (required_total_score : ℕ)
  (required_score_final_quiz : ℕ) :
  average_score_4quizzes = 92 →
  total_quizzes = 5 →
  average_a = 93 →
  current_score = 4 * average_score_4quizzes →
  required_total_score = total_quizzes * average_a →
  required_score_final_quiz = required_total_score - current_score →
  required_score_final_quiz = 97 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end amanda_final_quiz_score_l187_187440


namespace total_dolphins_correct_l187_187207

-- Define the initial number of dolphins
def initialDolphins : Nat := 65

-- Define the multiplier for the dolphins joining from the river
def joiningMultiplier : Nat := 3

-- Define the total number of dolphins after joining
def totalDolphins : Nat := initialDolphins + (joiningMultiplier * initialDolphins)

-- Prove that the total number of dolphins is 260
theorem total_dolphins_correct : totalDolphins = 260 := by
  sorry

end total_dolphins_correct_l187_187207


namespace four_integers_sum_6_7_8_9_l187_187756

theorem four_integers_sum_6_7_8_9 (a b c d : ℕ)
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) :
  (a = 1) ∧ (b = 2) ∧ (c = 3) ∧ (d = 4) := 
by 
  sorry

end four_integers_sum_6_7_8_9_l187_187756


namespace net_moles_nh3_after_reactions_l187_187192

/-- Define the stoichiometry of the reactions and available amounts of reactants -/
def step1_reaction (nh4cl na2co3 : ℕ) : ℕ :=
  if nh4cl / 2 >= na2co3 then 
    2 * na2co3
  else 
    2 * (nh4cl / 2)

def step2_reaction (koh h3po4 : ℕ) : ℕ :=
  0  -- No NH3 produced in this step

theorem net_moles_nh3_after_reactions :
  let nh4cl := 3
  let na2co3 := 1
  let koh := 3
  let h3po4 := 1
  let nh3_after_step1 := step1_reaction nh4cl na2co3
  let nh3_after_step2 := step2_reaction koh h3po4
  nh3_after_step1 + nh3_after_step2 = 2 :=
by
  sorry

end net_moles_nh3_after_reactions_l187_187192


namespace find_a_l187_187034

theorem find_a (a : ℝ) : (∀ x : ℝ, (x^2 - 4 * x + a) + |x - 3| ≤ 5) → (∃ x : ℝ, x = 3) → a = 8 :=
by
  sorry

end find_a_l187_187034


namespace equation_of_parallel_line_l187_187271

noncomputable def is_parallel (m₁ m₂ : ℝ) := m₁ = m₂

theorem equation_of_parallel_line (m : ℝ) (b : ℝ) (x₀ y₀ : ℝ) (a b1 c : ℝ) :
  is_parallel m (1 / 2) → y₀ = -1 → x₀ = 0 → 
  (a = 1 ∧ b1 = -2 ∧ c = -2) →
  a * x₀ + b1 * y₀ + c = 0 :=
by
  intros h_parallel hy hx habc
  sorry

end equation_of_parallel_line_l187_187271


namespace total_area_of_farm_l187_187760

-- Define the number of sections and area of each section
def number_of_sections : ℕ := 5
def area_of_each_section : ℕ := 60

-- State the problem as proving the total area of the farm
theorem total_area_of_farm : number_of_sections * area_of_each_section = 300 :=
by sorry

end total_area_of_farm_l187_187760


namespace quadratic_has_one_solution_at_zero_l187_187158

theorem quadratic_has_one_solution_at_zero (k : ℝ) :
  ((k - 2) * (0 : ℝ)^2 + 3 * (0 : ℝ) + k^2 - 4 = 0) →
  (3^2 - 4 * (k - 2) * (k^2 - 4) = 0) → k = -2 :=
by
  intro h1 h2
  sorry

end quadratic_has_one_solution_at_zero_l187_187158


namespace solve_equation_l187_187683

theorem solve_equation (y : ℝ) (z : ℝ) (hz : z = y^(1/3)) :
  (6 * y^(1/3) - 3 * y^(4/3) = 12 + y^(1/3) + y) ↔ (3 * z^4 + z^3 - 5 * z + 12 = 0) :=
by sorry

end solve_equation_l187_187683


namespace partial_fraction_decomposition_l187_187701

theorem partial_fraction_decomposition (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (-x^2 + 5*x - 6) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1)) →
  A = 6 ∧ B = -7 ∧ C = 5 :=
by
  intro h
  sorry

end partial_fraction_decomposition_l187_187701


namespace find_x_l187_187031

variable (x : ℝ)

theorem find_x (h : 0.60 * x = (1/3) * x + 110) : x = 412.5 :=
sorry

end find_x_l187_187031


namespace meaningful_expression_range_l187_187142

theorem meaningful_expression_range {x : ℝ} : (∃ y : ℝ, y = 5 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end meaningful_expression_range_l187_187142


namespace find_abc_sum_l187_187807

-- Definitions and statements directly taken from conditions
def Q1 (x y : ℝ) : Prop := y = x^2 + 51/50
def Q2 (x y : ℝ) : Prop := x = y^2 + 23/2
def common_tangent_rational_slope (a b c : ℤ) : Prop :=
  ∃ (x y : ℝ), (a * x + b * y = c) ∧ (Q1 x y ∨ Q2 x y)

theorem find_abc_sum :
  ∃ (a b c : ℕ), 
    gcd (a) (gcd (b) (c)) = 1 ∧
    common_tangent_rational_slope (a) (b) (c) ∧
    a + b + c = 9 :=
  by sorry

end find_abc_sum_l187_187807


namespace child_haircut_cost_l187_187178

/-
Problem Statement:
- Women's haircuts cost $48.
- Tayzia and her two daughters get haircuts.
- Tayzia wants to give a 20% tip to the hair stylist, which amounts to $24.
Question: How much does a child's haircut cost?
-/

noncomputable def cost_of_child_haircut (C : ℝ) : Prop :=
  let women's_haircut := 48
  let tip := 24
  let total_cost_before_tip := women's_haircut + 2 * C
  total_cost_before_tip * 0.20 = tip ∧ total_cost_before_tip = 120 ∧ C = 36

theorem child_haircut_cost (C : ℝ) (h1 : cost_of_child_haircut C) : C = 36 :=
  by sorry

end child_haircut_cost_l187_187178


namespace quadratic_solution_range_l187_187386

theorem quadratic_solution_range (t : ℝ) :
  (∃ x : ℝ, x^2 - 2 * x - t = 0 ∧ -1 < x ∧ x < 4) ↔ (-1 ≤ t ∧ t < 8) := 
sorry

end quadratic_solution_range_l187_187386


namespace impossible_distinct_values_l187_187097

theorem impossible_distinct_values :
  ∀ a b c : ℝ, 
  (a * (a - 4) = 12) → 
  (b * (b - 4) = 12) → 
  (c * (c - 4) = 12) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) → 
  false := 
sorry

end impossible_distinct_values_l187_187097


namespace Jackie_hops_six_hops_distance_l187_187908

theorem Jackie_hops_six_hops_distance : 
  let a : ℝ := 1
  let r : ℝ := 1 / 2
  let S : ℝ := a * ((1 - r^6) / (1 - r))
  S = 63 / 32 :=
by 
  sorry

end Jackie_hops_six_hops_distance_l187_187908


namespace ratio_EG_FH_l187_187589

theorem ratio_EG_FH (EF FG EH : ℝ) (hEF : EF = 3) (hFG : FG = 7) (hEH : EH = 20) :
  (EF + FG) / (EH - EF) = 10 / 17 :=
by
  sorry

end ratio_EG_FH_l187_187589


namespace min_inv_sum_l187_187159

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ 1 = 2*a + b

theorem min_inv_sum (a b : ℝ) (h : minimum_value_condition a b) : 
  ∃ a b : ℝ, (1 / a + 1 / b = 3 + 2 * Real.sqrt 2) := 
by 
  have h1 : a > 0 := h.1;
  have h2 : b > 0 := h.2.1;
  have h3 : 1 = 2 * a + b := h.2.2;
  sorry

end min_inv_sum_l187_187159


namespace lcm_pairs_count_l187_187573

noncomputable def distinct_pairs_lcm_count : ℕ :=
  sorry

theorem lcm_pairs_count :
  distinct_pairs_lcm_count = 1502 :=
  sorry

end lcm_pairs_count_l187_187573


namespace average_new_data_set_is_5_l187_187737

variable {x1 x2 x3 x4 : ℝ}
variable (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0)
variable (var_sqr : ℝ) (h_var : var_sqr = (1 / 4) * (x1 ^ 2 + x2 ^ 2 + x3 ^ 2 + x4 ^ 2 - 16))

theorem average_new_data_set_is_5 (h_var : var_sqr = (1 / 4) * (x1 ^ 2 + x2 ^ 2 + x3 ^ 2 + x4 ^ 2 - 16)) : 
  (x1 + 3 + x2 + 3 + x3 + 3 + x4 + 3) / 4 = 5 := 
by 
  sorry

end average_new_data_set_is_5_l187_187737


namespace scientific_notation_conversion_l187_187798

theorem scientific_notation_conversion : 450000000 = 4.5 * 10^8 :=
by
  sorry

end scientific_notation_conversion_l187_187798


namespace monotonic_decreasing_interval_l187_187774

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

theorem monotonic_decreasing_interval :
  {x : ℝ | 0 < x ∧ x ≤ 1} = {x : ℝ | ∃ ε > 0, ∀ y, y < x → f y > f x ∧ y > 0} :=
sorry

end monotonic_decreasing_interval_l187_187774


namespace slope_AA_l187_187354

-- Define the points and conditions
variable (a b c d e f : ℝ)

-- Assumptions
#check (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
#check (a ≠ b ∧ c ≠ d ∧ e ≠ f)
#check (a+2 > 0 ∧ b > 0 ∧ c+2 > 0 ∧ d > 0 ∧ e+2 > 0 ∧ f > 0)

-- Main Statement
theorem slope_AA'_not_negative_one
    (H1: a > 0) (H2: b > 0) (H3: c > 0) (H4: d > 0)
    (H5: e > 0) (H6: f > 0) 
    (H7: a ≠ b) (H8: c ≠ d) (H9: e ≠ f)
    (H10: a + 2 > 0) (H11: c + 2 > 0) (H12: e + 2 > 0) : 
    (a ≠ b) → (c ≠ d) → (e ≠ f) → ¬( (a + 2 - b) / (b - a) = -1 ) :=
by
  sorry

end slope_AA_l187_187354


namespace circle_equation_of_diameter_l187_187344

theorem circle_equation_of_diameter (A B : ℝ × ℝ) (hA : A = (-4, -5)) (hB : B = (6, -1)) :
  ∃ h k r : ℝ, (x - h)^2 + (y - k)^2 = r ∧ h = 1 ∧ k = -3 ∧ r = 29 := 
by
  sorry

end circle_equation_of_diameter_l187_187344


namespace factorize_3m2_minus_12_l187_187763

theorem factorize_3m2_minus_12 (m : ℤ) : 
  3 * m^2 - 12 = 3 * (m - 2) * (m + 2) := 
sorry

end factorize_3m2_minus_12_l187_187763


namespace ceil_sqrt_product_l187_187612

noncomputable def ceil_sqrt_3 : ℕ := ⌈Real.sqrt 3⌉₊
noncomputable def ceil_sqrt_12 : ℕ := ⌈Real.sqrt 12⌉₊
noncomputable def ceil_sqrt_120 : ℕ := ⌈Real.sqrt 120⌉₊

theorem ceil_sqrt_product :
  ceil_sqrt_3 * ceil_sqrt_12 * ceil_sqrt_120 = 88 :=
by
  sorry

end ceil_sqrt_product_l187_187612


namespace camel_water_ratio_l187_187132

theorem camel_water_ratio (gallons_water : ℕ) (ounces_per_gallon : ℕ) (traveler_ounces : ℕ)
  (total_ounces : ℕ) (camel_ounces : ℕ) (ratio : ℕ) 
  (h1 : gallons_water = 2) 
  (h2 : ounces_per_gallon = 128) 
  (h3 : traveler_ounces = 32) 
  (h4 : total_ounces = gallons_water * ounces_per_gallon) 
  (h5 : camel_ounces = total_ounces - traveler_ounces)
  (h6 : ratio = camel_ounces / traveler_ounces) : 
  ratio = 7 := 
by
  sorry

end camel_water_ratio_l187_187132


namespace quarterly_to_annual_rate_l187_187715

theorem quarterly_to_annual_rate (annual_rate : ℝ) (quarterly_rate : ℝ) (n : ℕ) (effective_annual_rate : ℝ) : 
  annual_rate = 4.5 →
  quarterly_rate = annual_rate / 4 →
  n = 4 →
  effective_annual_rate = (1 + quarterly_rate / 100)^n →
  effective_annual_rate * 100 = 4.56 :=
by
  intros h1 h2 h3 h4
  sorry

end quarterly_to_annual_rate_l187_187715


namespace trip_cost_l187_187301

theorem trip_cost (original_price : ℕ) (discount : ℕ) (num_people : ℕ)
  (h1 : original_price = 147) (h2 : discount = 14) (h3 : num_people = 2) :
  num_people * (original_price - discount) = 266 :=
by
  sorry

end trip_cost_l187_187301


namespace Olga_paints_zero_boards_l187_187043

variable (t p q t' : ℝ)
variable (rv ro : ℝ)

-- Conditions
axiom Valera_solo_trip : 2 * t + p = 2
axiom Valera_and_Olga_painting_time : 2 * t' + q = 3
axiom Valera_painting_rate : rv = 11 / p
axiom Valera_Omega_painting_rate : rv * q + ro * q = 9
axiom Valera_walk_faster : t' > t

-- Question: How many boards will Olga be able to paint alone if she needs to return home 1 hour after leaving?
theorem Olga_paints_zero_boards :
  t' > 1 → 0 = 0 := 
by 
  sorry

end Olga_paints_zero_boards_l187_187043


namespace prime_p_impplies_p_eq_3_l187_187884

theorem prime_p_impplies_p_eq_3 (p : ℕ) (hp : Prime p) (hp2 : Prime (p^2 + 2)) : p = 3 :=
sorry

end prime_p_impplies_p_eq_3_l187_187884


namespace burn_all_bridges_mod_1000_l187_187903

theorem burn_all_bridges_mod_1000 :
  let m := 2013 * 2 ^ 2012
  let n := 3 ^ 2012
  (m + n) % 1000 = 937 :=
by
  sorry

end burn_all_bridges_mod_1000_l187_187903


namespace bicycle_weight_l187_187330

theorem bicycle_weight (b s : ℕ) (h1 : 10 * b = 5 * s) (h2 : 5 * s = 200) : b = 20 := 
by 
  sorry

end bicycle_weight_l187_187330


namespace multiple_of_9_is_multiple_of_3_l187_187070

theorem multiple_of_9_is_multiple_of_3 (n : ℤ) (h : ∃ k : ℤ, n = 9 * k) : ∃ m : ℤ, n = 3 * m :=
by
  sorry

end multiple_of_9_is_multiple_of_3_l187_187070


namespace fraction_multiplication_l187_187495

theorem fraction_multiplication :
  (3 / 4) ^ 5 * (4 / 3) ^ 2 = 8 / 19 :=
by
  sorry

end fraction_multiplication_l187_187495


namespace box_surface_area_l187_187194

variables (a b c : ℝ)

noncomputable def sum_edges : ℝ := 4 * (a + b + c)
noncomputable def diagonal_length : ℝ := Real.sqrt (a^2 + b^2 + c^2)
noncomputable def surface_area : ℝ := 2 * (a * b + b * c + c * a)

/- The problem states that the sum of the lengths of the edges and the diagonal length gives us these values. -/
theorem box_surface_area (h1 : sum_edges a b c = 168) (h2 : diagonal_length a b c = 25) : surface_area a b c = 1139 :=
sorry

end box_surface_area_l187_187194


namespace total_number_of_animals_l187_187502

-- Define the given conditions as hypotheses
def num_horses (T : ℕ) : Prop :=
  ∃ (H x z : ℕ), H + x + z = 75

def cows_vs_horses (T : ℕ) : Prop :=
  ∃ (w z : ℕ),  w = z + 10

-- Define the final conclusion we need to prove
def total_animals (T : ℕ) : Prop :=
  T = 170

-- The main theorem which states the conditions imply the conclusion
theorem total_number_of_animals (T : ℕ) (h1 : num_horses T) (h2 : cows_vs_horses T) : total_animals T :=
by
  -- Proof to be filled in later
  sorry

end total_number_of_animals_l187_187502


namespace continuity_at_2_l187_187377

theorem continuity_at_2 (f : ℝ → ℝ) (x0 : ℝ) (hf : ∀ x, f x = -4 * x ^ 2 - 8) :
  x0 = 2 → ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x0| < δ → |f x + 24| < ε := by
  sorry

end continuity_at_2_l187_187377


namespace debby_candy_problem_l187_187172

theorem debby_candy_problem (D : ℕ) (sister_candy : ℕ) (eaten : ℕ) (remaining : ℕ) 
  (h1 : sister_candy = 42) (h2 : eaten = 35) (h3 : remaining = 39) :
  D + sister_candy - eaten = remaining ↔ D = 32 :=
by
  sorry

end debby_candy_problem_l187_187172


namespace clark_discount_l187_187939

noncomputable def price_per_part : ℕ := 80
noncomputable def num_parts : ℕ := 7
noncomputable def total_paid : ℕ := 439

theorem clark_discount : (price_per_part * num_parts - total_paid) = 121 :=
by
  -- proof goes here
  sorry

end clark_discount_l187_187939


namespace john_total_spent_l187_187678

noncomputable def calculate_total_spent : ℝ :=
  let orig_price_A := 900.0
  let discount_A := 0.15 * orig_price_A
  let price_A := orig_price_A - discount_A
  let tax_A := 0.06 * price_A
  let total_A := price_A + tax_A
  let orig_price_B := 600.0
  let discount_B := 0.25 * orig_price_B
  let price_B := orig_price_B - discount_B
  let tax_B := 0.09 * price_B
  let total_B := price_B + tax_B
  let total_other_toys := total_A + total_B
  let price_lightsaber := 2 * total_other_toys
  let tax_lightsaber := 0.04 * price_lightsaber
  let total_lightsaber := price_lightsaber + tax_lightsaber
  total_other_toys + total_lightsaber

theorem john_total_spent : calculate_total_spent = 4008.312 := by
  sorry

end john_total_spent_l187_187678


namespace correct_equation_option_l187_187626

theorem correct_equation_option :
  (∀ (x : ℝ), (x = 4 → false) ∧ (x = -4 → false)) →
  (∀ (y : ℝ), (y = 12 → true) ∧ (y = -12 → false)) →
  (∀ (z : ℝ), (z = -7 → false) ∧ (z = 7 → true)) →
  (∀ (w : ℝ), (w = 2 → true)) →
  ∃ (option : ℕ), option = 4 := 
by
  sorry

end correct_equation_option_l187_187626


namespace expand_polynomial_l187_187561

theorem expand_polynomial :
  (3 * x ^ 2 - 4 * x + 3) * (-2 * x ^ 2 + 3 * x - 4) = -6 * x ^ 4 + 17 * x ^ 3 - 30 * x ^ 2 + 25 * x - 12 :=
by
  sorry

end expand_polynomial_l187_187561


namespace new_parabola_through_point_l187_187752

def original_parabola (x : ℝ) : ℝ := x ^ 2 + 2 * x - 1

theorem new_parabola_through_point : 
  (∃ b : ℝ, ∀ x : ℝ, (x ^ 2 + 2 * x - 1 + b) = (x ^ 2 + 2 * x + 3)) :=
by
  sorry

end new_parabola_through_point_l187_187752


namespace percentage_increase_l187_187378

theorem percentage_increase
  (initial_earnings new_earnings : ℝ)
  (h_initial : initial_earnings = 55)
  (h_new : new_earnings = 60) :
  ((new_earnings - initial_earnings) / initial_earnings * 100) = 9.09 :=
by
  sorry

end percentage_increase_l187_187378


namespace inequality_satisfied_l187_187849

open Real

theorem inequality_satisfied (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  a * sqrt b + b * sqrt c + c * sqrt a ≤ 1 / sqrt 3 :=
sorry

end inequality_satisfied_l187_187849


namespace outfit_combinations_l187_187048

theorem outfit_combinations (shirts ties hat_choices : ℕ) (h_shirts : shirts = 8) (h_ties : ties = 7) (h_hat_choices : hat_choices = 3) : shirts * ties * hat_choices = 168 := by
  sorry

end outfit_combinations_l187_187048


namespace smallest_w_l187_187404

theorem smallest_w (w : ℕ) (h1 : Nat.gcd 1452 w = 1) (h2 : 2 ∣ w ∧ 3 ∣ w ∧ 13 ∣ w) :
  (∃ (w : ℕ), 2^4 ∣ 1452 * w ∧ 3^3 ∣ 1452 * w ∧ 13^3 ∣ 1452 * w ∧ w > 0) ∧
  ∀ (w' : ℕ), (2^4 ∣ 1452 * w' ∧ 3^3 ∣ 1452 * w' ∧ 13^3 ∣ 1452 * w' ∧ w' > 0) → w ≤ w' :=
  sorry

end smallest_w_l187_187404


namespace find_circle_radius_l187_187056

-- Definitions of given distances and the parallel chord condition
def isChordParallelToDiameter (c d : ℝ × ℝ) (radius distance1 distance2 : ℝ) : Prop :=
  let p1 := distance1
  let p2 := distance2
  p1 = 5 ∧ p2 = 12 ∧ 
  -- Assuming distances from the end of the diameter to the ends of the chord
  true

-- The main theorem which states the radius of the circle given the conditions
theorem find_circle_radius
  (diameter chord : ℝ × ℝ)
  (R p1 p2 : ℝ)
  (h1 : isChordParallelToDiameter diameter chord R p1 p2) :
  R = 6.5 :=
  by
    sorry

end find_circle_radius_l187_187056


namespace smallest_determinant_and_min_ab_l187_187261

def determinant (a b : ℤ) : ℤ :=
  36 * b - 81 * a

theorem smallest_determinant_and_min_ab :
  (∃ (a b : ℤ), 0 < determinant a b ∧ determinant a b = 9 ∧ ∀ a' b', determinant a' b' = 9 → a' + b' ≥ a + b) ∧
  (∃ (a b : ℤ), a = 3 ∧ b = 7) :=
sorry

end smallest_determinant_and_min_ab_l187_187261


namespace gcd_5039_3427_l187_187696

def a : ℕ := 5039
def b : ℕ := 3427

theorem gcd_5039_3427 : Nat.gcd a b = 7 := by
  sorry

end gcd_5039_3427_l187_187696


namespace Isaiah_types_more_l187_187358

theorem Isaiah_types_more (Micah_rate Isaiah_rate : ℕ) (h_Micah : Micah_rate = 20) (h_Isaiah : Isaiah_rate = 40) :
  (Isaiah_rate * 60 - Micah_rate * 60) = 1200 :=
by
  -- Here we assume we need to prove this theorem
  sorry

end Isaiah_types_more_l187_187358


namespace problem_85_cube_plus_3_85_square_plus_3_85_plus_1_l187_187209

theorem problem_85_cube_plus_3_85_square_plus_3_85_plus_1 :
  85^3 + 3 * (85^2) + 3 * 85 + 1 = 636256 := 
sorry

end problem_85_cube_plus_3_85_square_plus_3_85_plus_1_l187_187209


namespace average_math_chemistry_l187_187738

variables (M P C : ℕ)

axiom h1 : M + P = 60
axiom h2 : C = P + 20

theorem average_math_chemistry : (M + C) / 2 = 40 :=
by
  sorry

end average_math_chemistry_l187_187738


namespace abs_c_five_l187_187991

theorem abs_c_five (a b c : ℤ) (h_coprime : Int.gcd a (Int.gcd b c) = 1) 
  (h1 : a = 2 * (b + c)) 
  (h2 : b = 3 * (a + c)) : 
  |c| = 5 :=
by
  sorry

end abs_c_five_l187_187991


namespace extra_food_needed_l187_187838

theorem extra_food_needed (f1 f2 : ℝ) (h1 : f1 = 0.5) (h2 : f2 = 0.9) :
  f2 - f1 = 0.4 :=
by sorry

end extra_food_needed_l187_187838


namespace find_x_l187_187824

theorem find_x (x : ℝ) (h : x > 0) (area : 1 / 2 * (2 * x) * x = 72) : x = 6 * Real.sqrt 2 :=
by
  sorry

end find_x_l187_187824


namespace pages_read_in_7_days_l187_187674

-- Definitions of the conditions
def total_hours : ℕ := 10
def days : ℕ := 5
def pages_per_hour : ℕ := 50
def reading_days : ℕ := 7

-- Compute intermediate steps
def hours_per_day : ℕ := total_hours / days
def pages_per_day : ℕ := pages_per_hour * hours_per_day

-- Lean statement to prove Tom reads 700 pages in 7 days
theorem pages_read_in_7_days :
  pages_per_day * reading_days = 700 :=
by
  -- We can add the intermediate steps here as sorry, as we will not do the proof
  sorry

end pages_read_in_7_days_l187_187674


namespace max_min_conditions_x_values_for_max_min_a2_x_values_for_max_min_aneg2_l187_187387

noncomputable def y (x : ℝ) (a b : ℝ) : ℝ := (Real.cos x)^2 - a * (Real.sin x) + b

theorem max_min_conditions (a b : ℝ) :
  (∃ x : ℝ, y x a b = 0 ∧ (∀ x' : ℝ, y x' a b ≤ 0)) ∧ 
  (∃ x : ℝ, y x a b = -4 ∧ (∀ x' : ℝ, y x' a b ≥ -4)) ↔ 
  (a = 2 ∧ b = -2) ∨ (a = -2 ∧ b = -2) := sorry

theorem x_values_for_max_min_a2 (k : ℤ) :
  (∀ x, y x 2 (-2) = 0 ↔ x = -Real.pi / 2 + 2 * Real.pi * k) ∧ 
  (∀ x, (y x 2 (-2)) = -4 ↔ x = Real.pi / 2 + 2 * Real.pi * k) := sorry

theorem x_values_for_max_min_aneg2 (k : ℤ) :
  (∀ x, y x (-2) (-2) = 0 ↔ x = Real.pi / 2 + 2 * Real.pi * k) ∧ 
  (∀ x, (y x (-2) (-2)) = -4 ↔ x = -Real.pi / 2 + 2 * Real.pi * k) := sorry

end max_min_conditions_x_values_for_max_min_a2_x_values_for_max_min_aneg2_l187_187387


namespace dhoni_leftover_percentage_l187_187771

variable (E : ℝ) (spent_on_rent : ℝ) (spent_on_dishwasher : ℝ)

def percent_spent_on_rent : ℝ := 0.40
def percent_spent_on_dishwasher : ℝ := 0.32

theorem dhoni_leftover_percentage (E : ℝ) :
  (1 - (percent_spent_on_rent + percent_spent_on_dishwasher)) * E / E = 0.28 :=
by
  sorry

end dhoni_leftover_percentage_l187_187771


namespace math_problem_l187_187582

theorem math_problem : (4 + 6 + 7) * 2 - 2 + (3 / 3) = 33 := 
by
  sorry

end math_problem_l187_187582


namespace symmetric_points_a_minus_b_l187_187429

theorem symmetric_points_a_minus_b (a b : ℝ) 
  (h1 : a = -5) 
  (h2 : b = -1) :
  a - b = -4 := 
sorry

end symmetric_points_a_minus_b_l187_187429


namespace proportion_condition_l187_187035

variable (a b c d a₁ b₁ c₁ d₁ : ℚ)

theorem proportion_condition
  (h₁ : a / b = c / d)
  (h₂ : a₁ / b₁ = c₁ / d₁) :
  (a + a₁) / (b + b₁) = (c + c₁) / (d + d₁) ↔ a * d₁ + a₁ * d = b₁ * c + b * c₁ := by
  sorry

end proportion_condition_l187_187035


namespace remainder_of_division_l187_187837

def num : ℤ := 1346584
def divisor : ℤ := 137
def remainder : ℤ := 5

theorem remainder_of_division 
  (h : 0 <= divisor) (h' : divisor ≠ 0) : 
  num % divisor = remainder := 
sorry

end remainder_of_division_l187_187837


namespace apple_box_weights_l187_187588

theorem apple_box_weights (a b c d : ℤ) 
  (h1 : a + b + c = 70)
  (h2 : a + b + d = 80)
  (h3 : a + c + d = 73)
  (h4 : b + c + d = 77) : 
  a = 23 ∧ b = 27 ∧ c = 20 ∧ d = 30 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end apple_box_weights_l187_187588


namespace seventh_graders_trip_count_l187_187922

theorem seventh_graders_trip_count (fifth_graders sixth_graders teachers_per_grade parents_per_grade grades buses seats_per_bus : ℕ) 
  (hf : fifth_graders = 109) 
  (hs : sixth_graders = 115)
  (ht : teachers_per_grade = 4) 
  (hp : parents_per_grade = 2) 
  (hg : grades = 3) 
  (hb : buses = 5)
  (hsb : seats_per_bus = 72) : 
  ∃ seventh_graders : ℕ, seventh_graders = 118 := 
by
  sorry

end seventh_graders_trip_count_l187_187922


namespace a_and_b_solution_l187_187672

noncomputable def solve_for_a_b (a b : ℕ) : Prop :=
  a > 0 ∧ (∀ b : ℤ, b > 0) ∧ (2 * a^b + 16 + 3 * a^b - 8) / 2 = 84 → a = 2 ∧ b = 5

theorem a_and_b_solution (a b : ℕ) (h : solve_for_a_b a b) : a = 2 ∧ b = 5 :=
sorry

end a_and_b_solution_l187_187672


namespace count_ab_bc_ca_l187_187881

noncomputable def count_ways : ℕ :=
  (Nat.choose 9 3)

theorem count_ab_bc_ca (a b c : ℕ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 1 ≤ b ∧ b ≤ 9) (hc : 1 ≤ c ∧ c ≤ 9) :
  (10 * a + b < 10 * b + c ∧ 10 * b + c < 10 * c + a) → count_ways = 84 :=
sorry

end count_ab_bc_ca_l187_187881


namespace polynomial_transformation_l187_187810

theorem polynomial_transformation (g : Polynomial ℝ) (x : ℝ)
  (h : g.eval (x^2 + 2) = x^4 + 6 * x^2 + 8 * x) : 
  g.eval (x^2 - 1) = x^4 - 1 := by
  sorry

end polynomial_transformation_l187_187810


namespace remainder_when_divided_by_39_l187_187259

theorem remainder_when_divided_by_39 (N : ℤ) (h1 : ∃ k : ℤ, N = 13 * k + 3) : N % 39 = 3 :=
sorry

end remainder_when_divided_by_39_l187_187259


namespace books_read_by_Megan_l187_187266

theorem books_read_by_Megan 
    (M : ℕ)
    (Kelcie : ℕ := M / 4)
    (Greg : ℕ := 2 * (M / 4) + 9)
    (total : M + Kelcie + Greg = 65) :
  M = 32 :=
by sorry

end books_read_by_Megan_l187_187266


namespace n_is_prime_l187_187370

variable {n : ℕ}

theorem n_is_prime (hn : n > 1) (hd : ∀ d : ℕ, d > 0 ∧ d ∣ n → d + 1 ∣ n + 1) :
  Prime n := 
sorry

end n_is_prime_l187_187370


namespace remainder_when_divided_by_x_add_1_l187_187605

def q (x : ℝ) : ℝ := 2 * x^4 - 3 * x^3 + 4 * x^2 - 5 * x + 6

theorem remainder_when_divided_by_x_add_1 :
  q 2 = 6 → q (-1) = 20 :=
by
  intro hq2
  sorry

end remainder_when_divided_by_x_add_1_l187_187605


namespace Jasper_height_in_10_minutes_l187_187124

noncomputable def OmarRate : ℕ := 240 / 12
noncomputable def JasperRate : ℕ := 3 * OmarRate
noncomputable def JasperHeight (time: ℕ) : ℕ := JasperRate * time

theorem Jasper_height_in_10_minutes :
  JasperHeight 10 = 600 :=
by
  sorry

end Jasper_height_in_10_minutes_l187_187124


namespace greatest_divisor_of_consecutive_product_l187_187867

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l187_187867


namespace arithmetic_series_first_term_l187_187889

theorem arithmetic_series_first_term 
  (a d : ℝ)
  (h1 : 50 * (2 * a + 99 * d) = 1800)
  (h2 : 50 * (2 * a + 199 * d) = 6300) :
  a = -26.55 :=
by
  sorry

end arithmetic_series_first_term_l187_187889


namespace sally_initial_cards_l187_187782

def initial_baseball_cards (t w s a : ℕ) : Prop :=
  a = w + s + t

theorem sally_initial_cards :
  ∃ (initial_cards : ℕ), initial_baseball_cards 9 24 15 initial_cards ∧ initial_cards = 48 :=
by
  use 48
  sorry

end sally_initial_cards_l187_187782


namespace g_45_l187_187138

noncomputable def g : ℝ → ℝ := sorry

axiom g_functional (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y
axiom g_30 : g 30 = 30

theorem g_45 : g 45 = 20 := by
  -- proof to be completed
  sorry

end g_45_l187_187138


namespace find_n_l187_187822

theorem find_n (n : ℕ) (h1 : 0 < n) : 
  ∃ n, n > 0 ∧ (Real.tan (Real.pi / (2 * n)) + Real.sin (Real.pi / (2 * n)) = n / 3) := 
sorry

end find_n_l187_187822


namespace find_y_l187_187996

theorem find_y (steps distance : ℕ) (total_steps : ℕ) (marking_step : ℕ)
  (h1 : total_steps = 8)
  (h2 : distance = 48)
  (h3 : marking_step = 6) :
  steps = distance / total_steps * marking_step → steps = 36 :=
by
  intros
  sorry

end find_y_l187_187996


namespace two_pow_add_three_perfect_square_two_pow_add_one_perfect_square_l187_187245

theorem two_pow_add_three_perfect_square (n : ℕ) :
  ∃ k, 2^n + 3 = k^2 ↔ n = 0 :=
by {
  sorry
}

theorem two_pow_add_one_perfect_square (n : ℕ) :
  ∃ k, 2^n + 1 = k^2 ↔ n = 3 :=
by {
  sorry
}

end two_pow_add_three_perfect_square_two_pow_add_one_perfect_square_l187_187245


namespace transformed_center_is_correct_l187_187163

-- Definition for transformations
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (dx : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2)

def translate_up (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Given conditions
def initial_center : ℝ × ℝ := (4, -3)
def reflection_center := reflect_x initial_center
def translated_right_center := translate_right reflection_center 5
def final_center := translate_up translated_right_center 3

-- The statement to be proved
theorem transformed_center_is_correct : final_center = (9, 6) :=
by
  sorry

end transformed_center_is_correct_l187_187163


namespace ball_bounces_below_2_feet_l187_187784

theorem ball_bounces_below_2_feet :
  ∃ k : ℕ, 500 * (2 / 3 : ℝ) ^ k < 2 ∧ ∀ n < k, 500 * (2 / 3 : ℝ) ^ n ≥ 2 :=
by
  sorry

end ball_bounces_below_2_feet_l187_187784


namespace chickens_on_farm_are_120_l187_187811

-- Given conditions
def Number_of_hens : ℕ := 52
def Difference_hens_roosters : ℕ := 16

-- Define the number of roosters based on the conditions
def Number_of_roosters : ℕ := Number_of_hens + Difference_hens_roosters

-- The total number of chickens is the sum of hens and roosters
def Total_number_of_chickens : ℕ := Number_of_hens + Number_of_roosters

-- Prove that the total number of chickens is 120
theorem chickens_on_farm_are_120 : Total_number_of_chickens = 120 := by
  -- leave this part unimplemented for proof.
  -- The steps would involve computing the values based on definitions
  sorry

end chickens_on_farm_are_120_l187_187811


namespace problem_inequality_l187_187641

variable {a b c : ℝ}

-- Assuming a, b, c are positive real numbers
variables (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Assuming abc = 1
variable (h_abc : a * b * c = 1)

theorem problem_inequality :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by sorry

end problem_inequality_l187_187641


namespace johnson_class_more_students_l187_187188

theorem johnson_class_more_students
  (finley_class_students : ℕ)
  (johnson_class_students : ℕ)
  (h_finley : finley_class_students = 24)
  (h_johnson : johnson_class_students = 22) :
  johnson_class_students - finley_class_students / 2 = 10 :=
  sorry

end johnson_class_more_students_l187_187188


namespace min_value_of_expression_l187_187225

noncomputable def quadratic_function_min_value (a b c : ℝ) : ℝ :=
  (3 * (a * 1^2 + b * 1 + c) + 6 * (a * 0^2 + b * 0 + c) - (a * (-1)^2 + b * (-1) + c)) /
  ((a * 0^2 + b * 0 + c) - (a * (-2)^2 + b * (-2) + c))

theorem min_value_of_expression (a b c : ℝ)
  (h1 : b > 2 * a)
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0)
  (h3 : a > 0) :
  quadratic_function_min_value a b c = 12 :=
sorry

end min_value_of_expression_l187_187225


namespace probability_door_opened_second_attempt_discarded_probability_door_opened_second_attempt_not_discarded_l187_187334

namespace ProbabilityKeys

-- Define the problem conditions and the probability computations
def keys : ℕ := 4
def successful_keys : ℕ := 2
def unsuccessful_keys : ℕ := 2

def probability_first_fail (k : ℕ) (s : ℕ) : ℚ := (s : ℚ) / (k : ℚ)
def probability_second_success_discarded (s : ℕ) (k : ℕ) : ℚ := (s : ℚ) / (s + 1 - 1: ℚ) 
def probability_second_success_not_discarded (s : ℕ) (k : ℕ) : ℚ := (s : ℚ) / (k : ℚ)

-- The statements to be proved
theorem probability_door_opened_second_attempt_discarded : 
  (probability_first_fail keys successful_keys) * (probability_second_success_discarded unsuccessful_keys keys) = (1 : ℚ) / (3 : ℚ) :=
by sorry

theorem probability_door_opened_second_attempt_not_discarded : 
  (probability_first_fail keys successful_keys) * (probability_second_success_not_discarded successful_keys keys) = (1 : ℚ) / (4 : ℚ) :=
by sorry

end ProbabilityKeys

end probability_door_opened_second_attempt_discarded_probability_door_opened_second_attempt_not_discarded_l187_187334


namespace value_of_5_S_3_l187_187463

def S (a b : ℕ) : ℕ := 4 * a + 6 * b + 1

theorem value_of_5_S_3 : S 5 3 = 39 := by
  sorry

end value_of_5_S_3_l187_187463


namespace eq_of_symmetric_translation_l187_187105

noncomputable def parabola (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 5

noncomputable def translate_left (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := f (x + k)

noncomputable def translate_up (g : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := g x + k

noncomputable def translate_parabola (x : ℝ) : ℝ := translate_up (translate_left parabola 3) 2 x

noncomputable def symmetric_parabola (h : ℝ → ℝ) (x : ℝ) : ℝ := h (-x)

theorem eq_of_symmetric_translation :
  symmetric_parabola translate_parabola x = 2 * x^2 - 8 * x + 3 :=
by
  sorry

end eq_of_symmetric_translation_l187_187105


namespace find_a_b_l187_187717

theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, (x + 2 > a ∧ x - 1 < b) ↔ (1 < x ∧ x < 3)) → a = 3 ∧ b = 2 :=
by
  intro h
  sorry

end find_a_b_l187_187717


namespace intersection_of_A_and_B_l187_187174

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def expected_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B : A ∩ B = expected_intersection :=
by
  sorry

end intersection_of_A_and_B_l187_187174


namespace total_watermelons_l187_187706

/-- Proof statement: Jason grew 37 watermelons and Sandy grew 11 watermelons. 
    Prove that they grew a total of 48 watermelons. -/
theorem total_watermelons (jason_watermelons : ℕ) (sandy_watermelons : ℕ) (total_watermelons : ℕ) 
                         (h1 : jason_watermelons = 37) (h2 : sandy_watermelons = 11) :
  total_watermelons = 48 :=
by
  sorry

end total_watermelons_l187_187706


namespace num_tables_l187_187309

/-- Given conditions related to tables, stools, and benches, we want to prove the number of tables -/
theorem num_tables 
  (t s b : ℕ) 
  (h1 : s = 8 * t)
  (h2 : b = 2 * t)
  (h3 : 3 * s + 6 * b + 4 * t = 816) : 
  t = 20 := 
sorry

end num_tables_l187_187309


namespace alpha_plus_beta_l187_187925

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem alpha_plus_beta (α β : ℝ) (hα : 0 ≤ α) (hαβ : α < Real.pi) (hβ : 0 ≤ β) (hββ : β < Real.pi)
  (hα_neq_β : α ≠ β) (hf_α : f α = 1 / 2) (hf_β : f β = 1 / 2) : α + β = (7 * Real.pi) / 6 :=
by
  sorry

end alpha_plus_beta_l187_187925


namespace cost_price_of_article_l187_187111

theorem cost_price_of_article (C MP : ℝ) (h1 : 0.90 * MP = 1.25 * C) (h2 : 1.25 * C = 65.97) : C = 52.776 :=
by
  sorry

end cost_price_of_article_l187_187111


namespace percentage_of_gold_coins_is_35_percent_l187_187647

-- Definitions of conditions
def percentage_of_objects_that_are_beads : ℝ := 0.30
def percentage_of_coins_that_are_silver : ℝ := 0.25
def percentage_of_coins_that_are_gold : ℝ := 0.50

-- Problem Statement
theorem percentage_of_gold_coins_is_35_percent 
  (h_beads : percentage_of_objects_that_are_beads = 0.30) 
  (h_silver_coins : percentage_of_coins_that_are_silver = 0.25) 
  (h_gold_coins : percentage_of_coins_that_are_gold = 0.50) :
  0.35 = 0.35 := 
sorry

end percentage_of_gold_coins_is_35_percent_l187_187647


namespace max_a_value_l187_187071

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

end max_a_value_l187_187071


namespace intersection_M_N_l187_187897

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N:
  M ∩ N = {1, 2} :=
sorry

end intersection_M_N_l187_187897


namespace find_integers_correct_l187_187886

noncomputable def find_integers (a b c d : ℤ) : Prop :=
  a + b + c = 6 ∧ a + b + d = 7 ∧ a + c + d = 8 ∧ b + c + d = 9

theorem find_integers_correct (a b c d : ℤ) (h : find_integers a b c d) : a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 :=
by
  sorry

end find_integers_correct_l187_187886


namespace m_plus_n_eq_123_l187_187450

/- Define the smallest prime number -/
def m : ℕ := 2

/- Define the largest integer less than 150 with exactly three positive divisors -/
def n : ℕ := 121

/- Prove that the sum of m and n is 123 -/
theorem m_plus_n_eq_123 : m + n = 123 := by
  -- By definition, m is 2 and n is 121
  -- So, their sum is 123
  rfl

end m_plus_n_eq_123_l187_187450


namespace multiplication_counts_l187_187356

open Polynomial

noncomputable def horner_multiplications (n : ℕ) : ℕ := n

noncomputable def direct_summation_multiplications (n : ℕ) : ℕ := n * (n + 1) / 2

theorem multiplication_counts (P : Polynomial ℝ) (x₀ : ℝ) (n : ℕ)
  (h_degree : P.degree = n) :
  horner_multiplications n = n ∧ direct_summation_multiplications n = (n * (n + 1)) / 2 :=
by
  sorry

end multiplication_counts_l187_187356


namespace diophantine_no_nonneg_solutions_l187_187102

theorem diophantine_no_nonneg_solutions {a b : ℕ} (ha : 0 < a) (hb : 0 < b) (h_gcd : Nat.gcd a b = 1) :
  ∃ (c : ℕ), (a * b - a - b + 1) / 2 = (a - 1) * (b - 1) / 2 := 
sorry

end diophantine_no_nonneg_solutions_l187_187102


namespace problem_statement_l187_187979

noncomputable def f : ℕ+ → ℝ := sorry

theorem problem_statement (x : ℕ+) :
  (f 1 = 1) →
  (∀ x, f (x + 1) = (2 * f x) / (f x + 2)) →
  f x = 2 / (x + 1) := 
sorry

end problem_statement_l187_187979


namespace krishan_nandan_investment_ratio_l187_187011

theorem krishan_nandan_investment_ratio
    (X t : ℝ) (k : ℝ)
    (h1 : X * t = 6000)
    (h2 : X * t + k * X * 2 * t = 78000) :
    k = 6 := by
  sorry

end krishan_nandan_investment_ratio_l187_187011


namespace weekly_earnings_proof_l187_187419

def minutes_in_hour : ℕ := 60
def hourly_rate : ℕ := 4

def monday_minutes : ℕ := 150
def tuesday_minutes : ℕ := 40
def wednesday_minutes : ℕ := 155
def thursday_minutes : ℕ := 45

def weekly_minutes : ℕ := monday_minutes + tuesday_minutes + wednesday_minutes + thursday_minutes
def weekly_hours : ℕ := weekly_minutes / minutes_in_hour

def sylvia_earnings : ℕ := weekly_hours * hourly_rate

theorem weekly_earnings_proof :
  sylvia_earnings = 26 := by
  sorry

end weekly_earnings_proof_l187_187419


namespace cubes_end_same_digits_l187_187521

theorem cubes_end_same_digits (a b : ℕ) (h : a % 1000 = b % 1000) : (a^3) % 1000 = (b^3) % 1000 := by
  sorry

end cubes_end_same_digits_l187_187521


namespace geometric_series_common_ratio_l187_187503

theorem geometric_series_common_ratio (a : ℝ) (r : ℝ) (S : ℝ) (h1 : S = a / (1 - r))
  (h2 : S = 16 * (r^2 * S)) : |r| = 1/4 :=
by
  sorry

end geometric_series_common_ratio_l187_187503


namespace total_population_l187_187063

theorem total_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 5 * t) : 
  b + g + t = 26 * t :=
by
  -- We state our theorem including assumptions and goal
  sorry -- placeholder for the proof

end total_population_l187_187063


namespace interest_earned_l187_187523

noncomputable def simple_interest (P R T : ℚ) : ℚ :=
  P * R * T

noncomputable def T_years : ℚ :=
  5 + (8 / 12) + (12 / 365)

def principal : ℚ := 30000
def rate : ℚ := 23.7 / 100

theorem interest_earned :
  simple_interest principal rate T_years = 40524 := by
  sorry

end interest_earned_l187_187523


namespace sam_found_seashells_l187_187998

def seashells_given : Nat := 18
def seashells_left : Nat := 17
def seashells_found : Nat := seashells_given + seashells_left

theorem sam_found_seashells : seashells_found = 35 := by
  sorry

end sam_found_seashells_l187_187998


namespace six_positive_integers_solution_count_l187_187353

theorem six_positive_integers_solution_count :
  ∃ (S : Finset (Finset ℕ)) (n : ℕ) (a b c x y z : ℕ), 
  a ≥ b → b ≥ c → x ≥ y → y ≥ z → 
  a + b + c = x * y * z → 
  x + y + z = a * b * c → 
  S.card = 7 := by
    sorry

end six_positive_integers_solution_count_l187_187353


namespace John_days_per_week_l187_187320

theorem John_days_per_week
    (patients_first : ℕ := 20)
    (patients_increase_rate : ℕ := 20)
    (patients_second : ℕ := (20 + (20 * 20 / 100)))
    (total_weeks_year : ℕ := 50)
    (total_patients_year : ℕ := 11000) :
    ∃ D : ℕ, (20 * D + (20 + (20 * 20 / 100)) * D) * total_weeks_year = total_patients_year ∧ D = 5 := by
  sorry

end John_days_per_week_l187_187320


namespace direction_vectors_of_line_l187_187416

theorem direction_vectors_of_line : 
  ∃ v : ℝ × ℝ, (3 * v.1 - 4 * v.2 = 0) ∧ (v = (1, 3/4) ∨ v = (4, 3)) :=
by
  sorry

end direction_vectors_of_line_l187_187416


namespace curve_y_all_real_l187_187018

theorem curve_y_all_real (y : ℝ) : ∃ (x : ℝ), 2 * x * |x| + y^2 = 1 :=
sorry

end curve_y_all_real_l187_187018


namespace find_k_l187_187276

open Real

-- Define the operation "※"
def star (a b : ℝ) : ℝ := a * b + a + b^2

-- Define the main theorem stating the problem
theorem find_k (k : ℝ) (hk : k > 0) (h : star 1 k = 3) : k = 1 := by
  sorry

end find_k_l187_187276


namespace negation_of_existence_l187_187888

theorem negation_of_existence (h: ¬ ∃ x : ℝ, x^2 + 1 < 0) : ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by
  sorry

end negation_of_existence_l187_187888


namespace range_of_a_function_greater_than_exp_neg_x_l187_187199

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a / x

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ f x a = 0) → (0 < a ∧ a ≤ 1 / Real.exp 1) :=
sorry

theorem function_greater_than_exp_neg_x (a : ℝ) (h : a ≥ 2 / Real.exp 1) (x : ℝ) (hx : 0 < x) : f x a > Real.exp (-x) :=
sorry

end range_of_a_function_greater_than_exp_neg_x_l187_187199


namespace actual_number_of_sides_l187_187116

theorem actual_number_of_sides (apparent_angle : ℝ) (distortion_factor : ℝ)
  (sum_exterior_angles : ℝ) (actual_sides : ℕ) :
  apparent_angle = 18 ∧ distortion_factor = 1.5 ∧ sum_exterior_angles = 360 ∧ 
  apparent_angle / distortion_factor = sum_exterior_angles / actual_sides →
  actual_sides = 30 :=
by
  sorry

end actual_number_of_sides_l187_187116


namespace domain_and_parity_range_of_a_l187_187052

noncomputable def f (a x : ℝ) := Real.log (1 + x) / Real.log a
noncomputable def g (a x : ℝ) := Real.log (1 - x) / Real.log a

theorem domain_and_parity (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x, f a x * g a x = f a (-x) * g a (-x)) ∧ (∀ x, -1 < x ∧ x < 1) :=
sorry

theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : f a 1 + g a (1/4) < 1) :
  (a ∈ (Set.Ioo 0 1 ∪ Set.Ioi (3/2))) :=
sorry

end domain_and_parity_range_of_a_l187_187052


namespace relationship_between_m_and_n_l187_187749

theorem relationship_between_m_and_n
  (a : ℝ) (b : ℝ) (ha : a > 2) (hb : b ≠ 0)
  (m : ℝ := a + 1 / (a - 2))
  (n : ℝ := 2^(2 - b^2)) :
  m > n :=
sorry

end relationship_between_m_and_n_l187_187749


namespace small_glass_cost_l187_187100

theorem small_glass_cost 
  (S : ℝ)
  (small_glass_cost : ℝ)
  (large_glass_cost : ℝ := 5)
  (initial_money : ℝ := 50)
  (num_small : ℝ := 8)
  (change : ℝ := 1)
  (num_large : ℝ := 5)
  (spent_money : ℝ := initial_money - change)
  (total_large_cost : ℝ := num_large * large_glass_cost)
  (total_cost : ℝ := num_small * S + total_large_cost)
  (total_cost_eq : total_cost = spent_money) :
  S = 3 :=
by
  sorry

end small_glass_cost_l187_187100


namespace parabola_difference_eq_l187_187039

variable (a b c : ℝ)

def original_parabola (x : ℝ) : ℝ := a * x^2 + b * x + c
def reflected_parabola (x : ℝ) : ℝ := -(a * x^2 + b * x + c)
def translated_original (x : ℝ) : ℝ := a * x^2 + b * x + c + 3
def translated_reflection (x : ℝ) : ℝ := -(a * x^2 + b * x + c) - 3

theorem parabola_difference_eq (x : ℝ) :
  (translated_original a b c x) - (translated_reflection a b c x) = 2 * a * x^2 + 2 * b * x + 2 * c + 6 :=
by 
  sorry

end parabola_difference_eq_l187_187039


namespace paving_path_DE_time_l187_187735

-- Define the conditions
variable (v : ℝ) -- Speed of Worker 1
variable (x : ℝ) -- Total distance for Worker 1
variable (d2 : ℝ) -- Total distance for Worker 2
variable (AD DE EF FC : ℝ) -- Distances in the path of Worker 2

-- Define the statement
theorem paving_path_DE_time :
  (AD + DE + EF + FC) = d2 ∧
  x = 9 * v ∧
  d2 = 10.8 * v ∧
  d2 = AD + DE + EF + FC ∧
  (∀ t, t = (DE / (1.2 * v)) * 60) ∧
  t = 45 :=
by
  sorry

end paving_path_DE_time_l187_187735


namespace percentage_difference_l187_187542

noncomputable def P : ℝ := 40
variables {w x y z : ℝ}
variables (H1 : w = x * (1 - P / 100))
variables (H2 : x = 0.6 * y)
variables (H3 : z = 0.54 * y)
variables (H4 : z = 1.5 * w)

-- Goal
theorem percentage_difference : P = 40 :=
by sorry -- Proof omitted

end percentage_difference_l187_187542


namespace haji_mother_tough_weeks_l187_187866

/-- Let's define all the conditions: -/
def tough_week_revenue : ℕ := 800
def good_week_revenue : ℕ := 2 * tough_week_revenue
def number_of_good_weeks : ℕ := 5
def total_revenue : ℕ := 10400

/-- Let's define the proofs for intermediate steps: -/
def good_weeks_revenue : ℕ := number_of_good_weeks * good_week_revenue
def tough_weeks_revenue : ℕ := total_revenue - good_weeks_revenue
def number_of_tough_weeks : ℕ := tough_weeks_revenue / tough_week_revenue

/-- Now the theorem which states that the number of tough weeks is 3. -/
theorem haji_mother_tough_weeks : number_of_tough_weeks = 3 := by
  sorry

end haji_mother_tough_weeks_l187_187866


namespace necessary_and_sufficient_l187_187548

theorem necessary_and_sufficient (a b : ℝ) : a > b ↔ a * |a| > b * |b| := sorry

end necessary_and_sufficient_l187_187548


namespace range_of_m_l187_187532

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3 * m) ↔ m ≤ -1 ∨ m ≥ 4 :=
by
  sorry

end range_of_m_l187_187532


namespace initial_blueberry_jelly_beans_l187_187957

-- Definitions for initial numbers of jelly beans and modified quantities after eating
variables (b c : ℕ)

-- Conditions stated as Lean hypothesis
axiom initial_relation : b = 2 * c
axiom new_relation : b - 5 = 4 * (c - 5)

-- Theorem statement to prove the initial number of blueberry jelly beans is 30
theorem initial_blueberry_jelly_beans : b = 30 :=
by
  sorry

end initial_blueberry_jelly_beans_l187_187957


namespace find_B_l187_187319

theorem find_B : 
  ∀ (A B : ℕ), A ≤ 9 → B ≤ 9 → (600 + 10 * A + 5) + (100 + B) = 748 → B = 3 :=
by
  intros A B hA hB hEq
  sorry

end find_B_l187_187319


namespace problem1_problem2_l187_187303

-- Definition for the first proof problem
theorem problem1 (a b : ℝ) (h : a ≠ b) :
  (a^2 / (a - b) - b^2 / (a - b)) = a + b :=
by
  sorry

-- Definition for the second proof problem
theorem problem2 (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  ((x^2 - 1) / ((x^2 + 2 * x + 1)) / (x^2 - x) / (x + 1)) = 1 / x :=
by
  sorry

end problem1_problem2_l187_187303


namespace first_term_correct_l187_187945

noncomputable def first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^3 / (1 - (r^3)) = 80) : ℝ :=
a

theorem first_term_correct (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^3 / (1 - (r^3)) = 80) :
  first_term a r h1 h2 = 3.42 :=
sorry

end first_term_correct_l187_187945


namespace proof_third_length_gcd_l187_187471

/-- Statement: The greatest possible length that can be used to measure the given lengths exactly is 1 cm, 
and the third length is an unspecified number of centimeters that is relatively prime to both 1234 cm and 898 cm. -/
def third_length_gcd (x : ℕ) : Prop := 
  Int.gcd 1234 898 = 1 ∧ Int.gcd (Int.gcd 1234 898) x = 1

noncomputable def greatest_possible_length : ℕ := 1

theorem proof_third_length_gcd (x : ℕ) (h : third_length_gcd x) : greatest_possible_length = 1 := by
  sorry

end proof_third_length_gcd_l187_187471


namespace greatest_product_sum_2000_l187_187714

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l187_187714


namespace photos_to_cover_poster_l187_187236

/-
We are given a poster of dimensions 3 feet by 5 feet, and photos of dimensions 3 inches by 5 inches.
We need to prove that the number of such photos required to cover the poster is 144.
-/

-- Convert feet to inches
def feet_to_inches(feet : ℕ) : ℕ := 12 * feet

-- Dimensions of the poster in inches
def poster_height_in_inches := feet_to_inches 3
def poster_width_in_inches := feet_to_inches 5

-- Area of the poster
def poster_area : ℕ := poster_height_in_inches * poster_width_in_inches

-- Dimensions and area of one photo in inches
def photo_height := 3
def photo_width := 5
def photo_area : ℕ := photo_height * photo_width

-- Number of photos required to cover the poster
def number_of_photos : ℕ := poster_area / photo_area

-- Theorem stating the required number of photos is 144
theorem photos_to_cover_poster : number_of_photos = 144 := by
  -- Proof is omitted
  sorry

end photos_to_cover_poster_l187_187236


namespace percentage_increase_sides_l187_187847

theorem percentage_increase_sides (P : ℝ) :
  (1 + P/100) ^ 2 = 1.3225 → P = 15 := 
by
  sorry

end percentage_increase_sides_l187_187847


namespace problem_1_system_solution_problem_2_system_solution_l187_187335

theorem problem_1_system_solution (x y : ℝ)
  (h1 : x - 2 * y = 1)
  (h2 : 4 * x + 3 * y = 26) :
  x = 5 ∧ y = 2 :=
sorry

theorem problem_2_system_solution (x y : ℝ)
  (h1 : 2 * x + 3 * y = 3)
  (h2 : 5 * x - 3 * y = 18) :
  x = 3 ∧ y = -1 :=
sorry

end problem_1_system_solution_problem_2_system_solution_l187_187335


namespace false_proposition_A_l187_187787

theorem false_proposition_A 
  (a b : ℝ)
  (root1_eq_1 : ∀ x, x^2 + a * x + b = 0 → x = 1)
  (root2_eq_3 : ∀ x, x^2 + a * x + b = 0 → x = 3)
  (sum_of_roots_eq_2 : -a = 2)
  (opposite_sign_roots : ∀ x1 x2, x1 * x2 < 0) :
  ∃ prop, prop = "A" :=
sorry

end false_proposition_A_l187_187787


namespace average_annual_percentage_decrease_l187_187365

theorem average_annual_percentage_decrease (P2018 P2020 : ℝ) (x : ℝ) 
  (h_initial : P2018 = 20000)
  (h_final : P2020 = 16200) :
  P2018 * (1 - x)^2 = P2020 :=
by
  sorry

end average_annual_percentage_decrease_l187_187365


namespace exist_abc_l187_187634

theorem exist_abc (n k : ℕ) (h1 : 20 < n) (h2 : 1 < k) (h3 : n % k^2 = 0) :
  ∃ a b c : ℕ, n = a * b + b * c + c * a :=
sorry

end exist_abc_l187_187634


namespace range_of_m_given_q_range_of_m_given_p_or_q_and_not_p_and_q_l187_187833

def quadratic_has_two_distinct_positive_roots (m : ℝ) : Prop :=
  4 * m^2 - 4 * (m + 2) > 0 ∧ -2 * m > 0 ∧ m + 2 > 0

def hyperbola_with_foci_on_y_axis (m : ℝ) : Prop :=
  m + 3 < 0 ∧ 1 - 2 * m > 0

theorem range_of_m_given_q (m : ℝ) :
  hyperbola_with_foci_on_y_axis m → m < -3 :=
by
  sorry

theorem range_of_m_given_p_or_q_and_not_p_and_q (m : ℝ) :
  (quadratic_has_two_distinct_positive_roots m ∨ hyperbola_with_foci_on_y_axis m) ∧
  ¬(quadratic_has_two_distinct_positive_roots m ∧ hyperbola_with_foci_on_y_axis m) →
  (-2 < m ∧ m < -1) ∨ m < -3 :=
by
  sorry

end range_of_m_given_q_range_of_m_given_p_or_q_and_not_p_and_q_l187_187833


namespace tangent_identity_l187_187490

theorem tangent_identity :
  Real.tan (55 * Real.pi / 180) * 
  Real.tan (65 * Real.pi / 180) * 
  Real.tan (75 * Real.pi / 180) = 
  Real.tan (85 * Real.pi / 180) :=
sorry

end tangent_identity_l187_187490


namespace polygon_interior_angle_eq_l187_187189

theorem polygon_interior_angle_eq (n : ℕ) (h : ∀ i, 1 ≤ i → i ≤ n → (interior_angle : ℝ) = 108) : n = 5 := 
sorry

end polygon_interior_angle_eq_l187_187189


namespace part_I_part_II_l187_187363

variable (a b c : ℝ)

theorem part_I (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : ∀ x : ℝ, |x + a| + |x - b| + c ≥ 4) : a + b + c = 4 :=
sorry

theorem part_II (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 4) : (1/4) * a^2 + (1/9) * b^2 + c^2 ≥ 8/7 :=
sorry

end part_I_part_II_l187_187363


namespace factor_expression_l187_187509

variable {R : Type*} [CommRing R]

theorem factor_expression (a b c : R) :
    a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) =
    (a - b) * (b - c) * (c - a) * ((a + b) * a^2 * b^2 + (b + c) * b^2 * c^2 + (a + c) * c^2 * a) :=
sorry

end factor_expression_l187_187509


namespace five_cubic_km_to_cubic_meters_l187_187024

theorem five_cubic_km_to_cubic_meters (km_to_m : 1 = 1000) : 
  5 * (1000 ^ 3) = 5000000000 := 
by
  sorry

end five_cubic_km_to_cubic_meters_l187_187024


namespace preimage_of_point_l187_187627

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

-- Define the statement of the problem
theorem preimage_of_point {x y : ℝ} (h1 : f x y = (3, 1)) : (x = 2 ∧ y = 1) :=
by
  sorry

end preimage_of_point_l187_187627


namespace shortest_paths_in_grid_l187_187814

-- Define a function that computes the binomial coefficient
def binom (m n : ℕ) : ℕ :=
  Nat.choose (m + n) n

-- Proof problem: Prove that the number of shortest paths in an m x n grid is binom(m, n)
theorem shortest_paths_in_grid (m n : ℕ) : binom m n = Nat.choose (m + n) n :=
by
  -- Intentionally left blank: proof is skipped
  sorry

end shortest_paths_in_grid_l187_187814


namespace max_value_is_one_l187_187228

noncomputable def max_expression (a b : ℝ) : ℝ :=
(a + b) ^ 2 / (a ^ 2 + 2 * a * b + b ^ 2)

theorem max_value_is_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  max_expression a b ≤ 1 :=
sorry

end max_value_is_one_l187_187228


namespace gerbil_weights_l187_187323

theorem gerbil_weights
  (puffy muffy scruffy fluffy tuffy : ℕ)
  (h1 : puffy = 2 * muffy)
  (h2 : muffy = scruffy - 3)
  (h3 : scruffy = 12)
  (h4 : fluffy = muffy + tuffy)
  (h5 : fluffy = puffy / 2)
  (h6 : tuffy = puffy / 2) :
  puffy + muffy + tuffy = 36 := by
  sorry

end gerbil_weights_l187_187323


namespace tournament_matches_divisible_by_7_l187_187452

-- Define the conditions of the chess tournament
def single_elimination_tournament_matches (players byes: ℕ) : ℕ :=
  players - 1

theorem tournament_matches_divisible_by_7 :
  single_elimination_tournament_matches 120 40 = 119 ∧ 119 % 7 = 0 :=
by
  sorry

end tournament_matches_divisible_by_7_l187_187452


namespace sin_law_of_sines_l187_187175

theorem sin_law_of_sines (a b : ℝ) (sin_A sin_B : ℝ)
  (h1 : a = 3)
  (h2 : b = 4)
  (h3 : sin_A = 3 / 5) :
  sin_B = 4 / 5 := 
sorry

end sin_law_of_sines_l187_187175


namespace Total_toys_l187_187768

-- Definitions from the conditions
def Mandy_toys : ℕ := 20
def Anna_toys : ℕ := 3 * Mandy_toys
def Amanda_toys : ℕ := Anna_toys + 2

-- The statement to be proven
theorem Total_toys : Mandy_toys + Anna_toys + Amanda_toys = 142 :=
by
  -- Add proof here
  sorry

end Total_toys_l187_187768


namespace man_speed_3_kmph_l187_187935

noncomputable def bullet_train_length : ℝ := 200 -- The length of the bullet train in meters
noncomputable def bullet_train_speed_kmph : ℝ := 69 -- The speed of the bullet train in km/h
noncomputable def time_to_pass_man : ℝ := 10 -- The time taken to pass the man in seconds
noncomputable def conversion_factor_kmph_to_mps : ℝ := 1000 / 3600 -- Conversion factor from km/h to m/s
noncomputable def bullet_train_speed_mps : ℝ := bullet_train_speed_kmph * conversion_factor_kmph_to_mps -- Speed of the bullet train in m/s
noncomputable def relative_speed : ℝ := bullet_train_length / time_to_pass_man -- Relative speed at which train passes the man
noncomputable def speed_of_man_mps : ℝ := relative_speed - bullet_train_speed_mps -- Speed of the man in m/s
noncomputable def conversion_factor_mps_to_kmph : ℝ := 3.6 -- Conversion factor from m/s to km/h
noncomputable def speed_of_man_kmph : ℝ := speed_of_man_mps * conversion_factor_mps_to_kmph -- Speed of the man in km/h

theorem man_speed_3_kmph :
  speed_of_man_kmph = 3 :=
by
  sorry

end man_speed_3_kmph_l187_187935


namespace triangle_area_integral_bound_l187_187898

def S := 200
def AC := 20
def dist_A_to_tangent := 25
def dist_C_to_tangent := 16
def largest_integer_not_exceeding (S : ℕ) (n : ℕ) : ℕ := n

theorem triangle_area_integral_bound (AC : ℕ) (dist_A_to_tangent : ℕ) (dist_C_to_tangent : ℕ) (S : ℕ) : 
  AC = 20 ∧ dist_A_to_tangent = 25 ∧ dist_C_to_tangent = 16 → largest_integer_not_exceeding S 20 = 10 :=
by
  sorry

end triangle_area_integral_bound_l187_187898


namespace polynomial_expansion_a6_l187_187630

theorem polynomial_expansion_a6 :
  let p := x^2 + x^7
  ∃ (a : ℕ → ℝ), p = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + a 3 * (x + 1)^3 + a 4 * (x + 1)^4 + a 5 * (x + 1)^5 + a 6 * (x + 1)^6 + a 7 * (x + 1)^7 ∧ a 6 = -7 := 
sorry

end polynomial_expansion_a6_l187_187630


namespace description_of_T_l187_187958

def T : Set (ℝ × ℝ) := { p | ∃ c, (4 = p.1 + 3 ∨ 4 = p.2 - 2 ∨ p.1 + 3 = p.2 - 2) 
                           ∧ (p.1 + 3 ≤ c ∨ p.2 - 2 ≤ c ∨ 4 ≤ c) }

theorem description_of_T : 
  (∀ p ∈ T, (∃ x y : ℝ, p = (x, y) ∧ ((x = 1 ∧ y ≤ 6) ∨ (y = 6 ∧ x ≤ 1) ∨ (y = x + 5 ∧ x ≥ 1 ∧ y ≥ 6)))) :=
sorry

end description_of_T_l187_187958


namespace vasya_expected_area_greater_l187_187016

/-- Vasya and Asya roll dice to cut out shapes and determine whose expected area is greater. -/
theorem vasya_expected_area_greater :
  let A : ℕ := 1
  let B : ℕ := 2
  (6 * 7 * (2 * 7 / 6) * 21 / 6) < (21 * 91 / 6) := 
by
  sorry

end vasya_expected_area_greater_l187_187016


namespace identity_holds_l187_187327

noncomputable def identity_proof : Prop :=
∀ (x : ℝ), (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x

theorem identity_holds : identity_proof :=
by
  sorry

end identity_holds_l187_187327


namespace different_tea_packets_or_miscalculation_l187_187875

theorem different_tea_packets_or_miscalculation : 
  ∀ (n_1 n_2 : ℕ), 3 ≤ t_1 ∧ t_1 ≤ 4 ∧ 3 ≤ t_2 ∧ t_2 ≤ 4 ∧
  (74 = t_1 * x ∧ 105 = t_2 * y → x ≠ y) ∨ 
  (∃ (e_1 e_2 : ℕ), (e_1 + e_2 = 74) ∧ (e_1 + e_2 = 105) → false) :=
by
  -- Construction based on the provided mathematical problem
  sorry

end different_tea_packets_or_miscalculation_l187_187875


namespace woman_work_rate_l187_187362

theorem woman_work_rate (W : ℝ) :
  (1 / 6) + W + (1 / 9) = (1 / 3) → W = (1 / 18) :=
by
  intro h
  sorry

end woman_work_rate_l187_187362


namespace totalProblemsSolved_l187_187168

-- Given conditions
def initialProblemsSolved : Nat := 45
def additionalProblemsSolved : Nat := 18

-- Statement to prove the total problems solved equals 63
theorem totalProblemsSolved : initialProblemsSolved + additionalProblemsSolved = 63 := 
by
  sorry

end totalProblemsSolved_l187_187168


namespace yellow_scores_l187_187269

theorem yellow_scores (W B : ℕ) 
  (h₁ : W / B = 7 / 6)
  (h₂ : (2 / 3 : ℚ) * (W - B) = 4) : 
  W + B = 78 :=
sorry

end yellow_scores_l187_187269


namespace problem_1_problem_2_l187_187825

def A (x : ℝ) : Prop := x^2 - 3*x - 10 ≤ 0
def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2*m - 1

theorem problem_1 (m : ℝ) : (∀ x, B m x → A x)  →  m ≤ 3 := 
sorry

theorem problem_2 (m : ℝ) : (¬ ∃ x, A x ∧ B m x) ↔ (m < 2 ∨ 4 < m) := 
sorry

end problem_1_problem_2_l187_187825


namespace number_of_squirrels_l187_187121

/-
Problem: Some squirrels collected 575 acorns. If each squirrel needs 130 acorns to get through the winter, each squirrel needs to collect 15 more acorns. 
Question: How many squirrels are there?
Conditions:
 1. Some squirrels collected 575 acorns.
 2. Each squirrel needs 130 acorns to get through the winter.
 3. Each squirrel needs to collect 15 more acorns.
Answer: 5 squirrels
-/

theorem number_of_squirrels (acorns_total : ℕ) (acorns_needed : ℕ) (acorns_short : ℕ) (S : ℕ)
  (h1 : acorns_total = 575)
  (h2 : acorns_needed = 130)
  (h3 : acorns_short = 15)
  (h4 : S * (acorns_needed - acorns_short) = acorns_total) :
  S = 5 :=
by
  sorry

end number_of_squirrels_l187_187121


namespace percentage_people_taking_bus_l187_187282

-- Definitions
def population := 80
def car_pollution := 10 -- pounds of carbon per car per year
def bus_pollution := 100 -- pounds of carbon per bus per year
def bus_capacity := 40 -- people per bus
def carbon_reduction := 100 -- pounds of carbon reduced per year after the bus is introduced

-- Problem statement in Lean 4
theorem percentage_people_taking_bus :
  (10 / 80 : ℝ) = 0.125 :=
by
  sorry

end percentage_people_taking_bus_l187_187282


namespace angle_C_is_110_degrees_l187_187637

def lines_are_parallel (l m : Type) : Prop := sorry
def angle_measure (A : Type) : ℝ := sorry
noncomputable def mangle (C : Type) : ℝ := sorry

theorem angle_C_is_110_degrees 
  (l m C D : Type) 
  (hlm : lines_are_parallel l m)
  (hCDl : lines_are_parallel C l)
  (hCDm : lines_are_parallel C m)
  (hA : angle_measure A = 100)
  (hB : angle_measure B = 150) :
  mangle C = 110 :=
by
  sorry

end angle_C_is_110_degrees_l187_187637


namespace mrs_hilt_bees_l187_187531

theorem mrs_hilt_bees (n : ℕ) (h : 3 * n = 432) : n = 144 := by
  sorry

end mrs_hilt_bees_l187_187531


namespace expected_number_of_draws_l187_187709

-- Given conditions
def redBalls : ℕ := 2
def blackBalls : ℕ := 5
def totalBalls : ℕ := redBalls + blackBalls

-- Definition of expected number of draws
noncomputable def expected_draws : ℚ :=
  (2 * (1/21) + 3 * (2/21) + 4 * (3/21) + 5 * (4/21) + 
   6 * (5/21) + 7 * (6/21))

-- The theorem statement to prove
theorem expected_number_of_draws :
  expected_draws = 16 / 3 := by
  sorry

end expected_number_of_draws_l187_187709


namespace seismic_activity_mismatch_percentage_l187_187152

theorem seismic_activity_mismatch_percentage
  (total_days : ℕ)
  (quiet_days_percentage : ℝ)
  (prediction_accuracy : ℝ)
  (predicted_quiet_days_percentage : ℝ)
  (quiet_prediction_correctness : ℝ)
  (active_days_percentage : ℝ)
  (incorrect_quiet_predictions : ℝ) :
  quiet_days_percentage = 0.8 →
  predicted_quiet_days_percentage = 0.64 →
  quiet_prediction_correctness = 0.7 →
  active_days_percentage = 0.2 →
  incorrect_quiet_predictions = predicted_quiet_days_percentage - (quiet_prediction_correctness * quiet_days_percentage) →
  (incorrect_quiet_predictions / active_days_percentage) * 100 = 40 := by
  sorry

end seismic_activity_mismatch_percentage_l187_187152


namespace calculate_expression_l187_187006

theorem calculate_expression : 
  (π - 3.14) ^ 0 - 8 ^ (2 / 3) + (1 / 5) ^ 2 * (Real.logb 2 32) + 5 ^ (Real.logb 5 3) = 1 / 5 :=
by
  sorry

end calculate_expression_l187_187006


namespace total_rooms_l187_187948

-- Definitions for the problem conditions
variables (x y : ℕ)

-- Given conditions
def condition1 : Prop := x = 8
def condition2 : Prop := 2 * x + 3 * y = 31

-- The theorem to prove
theorem total_rooms (h1 : condition1 x) (h2 : condition2 x y) : x + y = 13 :=
by sorry

end total_rooms_l187_187948


namespace repeating_decimal_sum_as_fraction_l187_187308

theorem repeating_decimal_sum_as_fraction :
  let d1 := 1 / 9    -- Representation of 0.\overline{1}
  let d2 := 1 / 99   -- Representation of 0.\overline{01}
  d1 + d2 = (4 : ℚ) / 33 := by
{
  sorry
}

end repeating_decimal_sum_as_fraction_l187_187308


namespace largest_number_divisible_by_48_is_9984_l187_187343

def largest_divisible_by_48 (n : ℕ) := ∀ m ≥ n, m % 48 = 0 → m ≤ 9999

theorem largest_number_divisible_by_48_is_9984 :
  largest_divisible_by_48 9984 ∧ 9999 / 10^3 = 9 ∧ 48 ∣ 9984 ∧ 9984 < 10000 :=
by
  sorry

end largest_number_divisible_by_48_is_9984_l187_187343


namespace highest_nitrogen_percentage_l187_187280

-- Define molar masses for each compound
def molar_mass_NH2OH : Float := 33.0
def molar_mass_NH4NO2 : Float := 64.1 
def molar_mass_N2O3 : Float := 76.0
def molar_mass_NH4NH2CO2 : Float := 78.1

-- Define mass of nitrogen atoms
def mass_of_nitrogen : Float := 14.0

-- Define the percentage calculations
def percentage_NH2OH : Float := (mass_of_nitrogen / molar_mass_NH2OH) * 100.0
def percentage_NH4NO2 : Float := (2 * mass_of_nitrogen / molar_mass_NH4NO2) * 100.0
def percentage_N2O3 : Float := (2 * mass_of_nitrogen / molar_mass_N2O3) * 100.0
def percentage_NH4NH2CO2 : Float := (2 * mass_of_nitrogen / molar_mass_NH4NH2CO2) * 100.0

-- Define the proof problem
theorem highest_nitrogen_percentage : percentage_NH4NO2 > percentage_NH2OH ∧
                                      percentage_NH4NO2 > percentage_N2O3 ∧
                                      percentage_NH4NO2 > percentage_NH4NH2CO2 :=
by 
  sorry

end highest_nitrogen_percentage_l187_187280


namespace team_A_games_42_l187_187420

noncomputable def team_games (a b : ℕ) : Prop :=
  (a * 2 / 3 + 7) = b * 5 / 8

theorem team_A_games_42 (a b : ℕ) (h1 : a * 2 / 3 = b * 5 / 8 - 7)
                                 (h2 : b = a + 14) :
  a = 42 :=
by
  sorry

end team_A_games_42_l187_187420


namespace art_piece_increase_is_correct_l187_187682

-- Define the conditions
def initial_price : ℝ := 4000
def future_multiplier : ℝ := 3
def future_price : ℝ := future_multiplier * initial_price

-- Define the goal
-- Proof that the increase in price is equal to $8000
theorem art_piece_increase_is_correct : future_price - initial_price = 8000 := 
by {
  -- We put sorry here to skip the actual proof
  sorry
}

end art_piece_increase_is_correct_l187_187682


namespace solution_set_of_inequality_l187_187843

theorem solution_set_of_inequality : { x : ℝ | 0 < x ∧ x < 2 } = { x : ℝ | 0 < x ∧ x < 2 } :=
by
  sorry

end solution_set_of_inequality_l187_187843


namespace calculate_opening_price_l187_187919

theorem calculate_opening_price (C : ℝ) (r : ℝ) (P : ℝ) 
  (h1 : C = 15)
  (h2 : r = 0.5)
  (h3 : C = P + r * P) :
  P = 10 :=
by sorry

end calculate_opening_price_l187_187919


namespace scientific_notation_periodicals_l187_187368

theorem scientific_notation_periodicals :
  (56000000 : ℝ) = 5.6 * 10^7 := by
sorry

end scientific_notation_periodicals_l187_187368


namespace hyperbola_condition_l187_187741

theorem hyperbola_condition (k : ℝ) : 
  (∃ (x y : ℝ), (x^2 / (4 + k) + y^2 / (1 - k) = 1)) ↔ (k < -4 ∨ k > 1) :=
by 
  sorry

end hyperbola_condition_l187_187741


namespace largest_number_l187_187051

theorem largest_number (a b c : ℕ) (h1: a ≤ b) (h2: b ≤ c) 
  (h3: (a + b + c) = 90) (h4: b = 32) (h5: b = a + 4) : c = 30 :=
sorry

end largest_number_l187_187051


namespace halfway_between_fractions_l187_187989

theorem halfway_between_fractions : 
  (2:ℚ) / 9 + (5 / 12) / 2 = 23 / 72 := 
sorry

end halfway_between_fractions_l187_187989


namespace john_drive_time_l187_187321

theorem john_drive_time
  (t : ℝ)
  (h1 : 60 * t + 90 * (15 / 4 - t) = 300)
  (h2 : 1 / 4 = 15 / 60)
  (h3 : 4 = 15 / 4 + t + 1 / 4)
  :
  t = 1.25 :=
by
  -- This introduces the hypothesis and begins the Lean proof.
  sorry

end john_drive_time_l187_187321


namespace neg_proposition_equiv_l187_187293

theorem neg_proposition_equiv (p : Prop) : (¬ (∃ n : ℕ, 2^n > 1000)) = (∀ n : ℕ, 2^n ≤ 1000) :=
by
  sorry

end neg_proposition_equiv_l187_187293


namespace mean_home_runs_correct_l187_187472

def mean_home_runs (players: List ℕ) (home_runs: List ℕ) : ℚ :=
  let total_runs := (List.zipWith (· * ·) players home_runs).sum
  let total_players := players.sum
  total_runs / total_players

theorem mean_home_runs_correct :
  mean_home_runs [6, 4, 3, 1, 1, 1] [6, 7, 8, 10, 11, 12] = 121 / 16 :=
by
  -- The proof should go here
  sorry

end mean_home_runs_correct_l187_187472


namespace wyatt_envelopes_fewer_l187_187578

-- Define assets for envelopes
variables (blue_envelopes yellow_envelopes : ℕ)

-- Conditions from the problem
def wyatt_conditions :=
  blue_envelopes = 10 ∧ yellow_envelopes < blue_envelopes ∧ blue_envelopes + yellow_envelopes = 16

-- Theorem: How many fewer yellow envelopes Wyatt has compared to blue envelopes?
theorem wyatt_envelopes_fewer (hb : blue_envelopes = 10) (ht : blue_envelopes + yellow_envelopes = 16) : 
  blue_envelopes - yellow_envelopes = 4 := 
by sorry

end wyatt_envelopes_fewer_l187_187578


namespace least_number_to_subtract_l187_187834

theorem least_number_to_subtract {x : ℕ} (h : x = 13604) : 
    ∃ n : ℕ, n = 32 ∧ (13604 - n) % 87 = 0 :=
by
  sorry

end least_number_to_subtract_l187_187834


namespace correct_answer_is_option_d_l187_187364

def is_quadratic (eq : String) : Prop :=
  eq = "a*x^2 + b*x + c = 0"

def OptionA : String := "1/x^2 + x - 1 = 0"
def OptionB : String := "3x + 1 = 5x + 4"
def OptionC : String := "x^2 + y = 0"
def OptionD : String := "x^2 - 2x + 1 = 0"

theorem correct_answer_is_option_d :
  is_quadratic OptionD :=
by
  sorry

end correct_answer_is_option_d_l187_187364


namespace bus_driver_hours_worked_l187_187779

-- Definitions based on the problem's conditions.
def regular_rate : ℕ := 20
def regular_hours : ℕ := 40
def overtime_rate : ℕ := regular_rate + (3 * (regular_rate / 4))  -- 75% higher
def total_compensation : ℕ := 1000

-- Theorem statement: The bus driver worked a total of 45 hours last week.
theorem bus_driver_hours_worked : 40 + ((total_compensation - (regular_rate * regular_hours)) / overtime_rate) = 45 := 
by 
  sorry

end bus_driver_hours_worked_l187_187779


namespace min_value_expression_l187_187759

variable (a b : ℝ)

theorem min_value_expression :
  0 < a →
  1 < b →
  a + b = 2 →
  (∃ x, x = 3 + 2 * Real.sqrt 2 ∧ (∀ y, y = (2 / a) + (1 / (b - 1)) → y ≥ x)) :=
by
  sorry

end min_value_expression_l187_187759


namespace find_x_of_equation_l187_187751

theorem find_x_of_equation (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end find_x_of_equation_l187_187751


namespace max_temp_difference_l187_187213

-- Define the highest and lowest temperatures
def highest_temp : ℤ := 3
def lowest_temp : ℤ := -3

-- State the theorem for maximum temperature difference
theorem max_temp_difference : highest_temp - lowest_temp = 6 := 
by 
  -- Provide the proof here
  sorry

end max_temp_difference_l187_187213


namespace symmetric_point_x_axis_l187_187460

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetricWithRespectToXAxis (p : Point3D) : Point3D :=
  {x := p.x, y := -p.y, z := -p.z}

theorem symmetric_point_x_axis :
  symmetricWithRespectToXAxis ⟨-1, -2, 3⟩ = ⟨-1, 2, -3⟩ :=
  by
    sorry

end symmetric_point_x_axis_l187_187460


namespace swim_team_girls_l187_187170

-- Definitions using the given conditions
variables (B G : ℕ)
theorem swim_team_girls (h1 : G = 5 * B) (h2 : G + B = 96) : G = 80 :=
sorry

end swim_team_girls_l187_187170


namespace total_emails_received_l187_187538

theorem total_emails_received :
  let e1 := 16
  let e2 := e1 / 2
  let e3 := e2 / 2
  let e4 := e3 / 2
  e1 + e2 + e3 + e4 = 30 :=
by
  sorry

end total_emails_received_l187_187538


namespace arithmetic_sequence_properties_l187_187464

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)
  (h2 : d ≠ 0)
  (h3 : ∀ n, S n ≤ S 8) :
  d < 0 ∧ S 17 ≤ 0 := 
sorry

end arithmetic_sequence_properties_l187_187464


namespace game_of_24_l187_187133

theorem game_of_24 : 
  let a := 3
  let b := -5
  let c := 6
  let d := -8
  ((b + c / a) * d = 24) :=
by
  let a := 3
  let b := -5
  let c := 6
  let d := -8
  show (b + c / a) * d = 24
  sorry

end game_of_24_l187_187133


namespace geometric_sequence_common_ratio_l187_187214

-- Define the geometric sequence with properties
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n < a (n + 1)

-- Main theorem
theorem geometric_sequence_common_ratio {a : ℕ → ℝ} {q : ℝ} (h_seq : increasing_geometric_sequence a q) (h_a1 : a 0 > 0) (h_eqn : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l187_187214


namespace sum_100_consecutive_from_neg49_l187_187095

noncomputable def sum_of_consecutive_integers (n : ℕ) (first_term : ℤ) : ℤ :=
  n * ( first_term + (first_term + n - 1) ) / 2

theorem sum_100_consecutive_from_neg49 : sum_of_consecutive_integers 100 (-49) = 50 :=
by sorry

end sum_100_consecutive_from_neg49_l187_187095


namespace f_zero_eq_one_positive_for_all_x_l187_187305

variables {R : Type*} [LinearOrderedField R] (f : R → R)

-- Conditions
axiom domain (x : R) : true -- This translates that f has domain (-∞, ∞)
axiom non_constant (x1 x2 : R) (h : x1 ≠ x2) : f x1 ≠ f x2
axiom functional_eq (x y : R) : f (x + y) = f x * f y

-- Questions
theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem positive_for_all_x (x : R) : f x > 0 :=
sorry

end f_zero_eq_one_positive_for_all_x_l187_187305


namespace count_perfect_cubes_between_10_and_2000_l187_187367

theorem count_perfect_cubes_between_10_and_2000 : 
  (∃ n_min n_max, n_min^3 ≥ 10 ∧ n_max^3 ≤ 2000 ∧ 
  (n_max - n_min + 1 = 10)) := 
sorry

end count_perfect_cubes_between_10_and_2000_l187_187367


namespace gcd_gx_x_eq_one_l187_187408

   variable (x : ℤ)
   variable (hx : ∃ k : ℤ, x = 34567 * k)

   def g (x : ℤ) : ℤ := (3 * x + 4) * (8 * x + 3) * (15 * x + 11) * (x + 15)

   theorem gcd_gx_x_eq_one : Int.gcd (g x) x = 1 :=
   by 
     sorry
   
end gcd_gx_x_eq_one_l187_187408


namespace remaining_inventory_l187_187781

def initial_inventory : Int := 4500
def bottles_sold_mon : Int := 2445
def bottles_sold_tue : Int := 906
def bottles_sold_wed : Int := 215
def bottles_sold_thu : Int := 457
def bottles_sold_fri : Int := 312
def bottles_sold_sat : Int := 239
def bottles_sold_sun : Int := 188

def bottles_received_tue : Int := 350
def bottles_received_thu : Int := 750
def bottles_received_sat : Int := 981

def total_bottles_sold : Int := bottles_sold_mon + bottles_sold_tue + bottles_sold_wed + bottles_sold_thu + bottles_sold_fri + bottles_sold_sat + bottles_sold_sun
def total_bottles_received : Int := bottles_received_tue + bottles_received_thu + bottles_received_sat

theorem remaining_inventory (initial_inventory bottles_sold_mon bottles_sold_tue bottles_sold_wed bottles_sold_thu bottles_sold_fri bottles_sold_sat bottles_sold_sun bottles_received_tue bottles_received_thu bottles_received_sat total_bottles_sold total_bottles_received : Int) :
  initial_inventory - total_bottles_sold + total_bottles_received = 819 :=
by
  sorry

end remaining_inventory_l187_187781


namespace events_mutually_exclusive_but_not_opposite_l187_187220

inductive Card
| black
| red
| white

inductive Person
| A
| B
| C

def event_A_gets_red (distribution : Person → Card) : Prop :=
  distribution Person.A = Card.red

def event_B_gets_red (distribution : Person → Card) : Prop :=
  distribution Person.B = Card.red

theorem events_mutually_exclusive_but_not_opposite (distribution : Person → Card) :
  event_A_gets_red distribution ∧ event_B_gets_red distribution → False :=
by sorry

end events_mutually_exclusive_but_not_opposite_l187_187220


namespace gas_cost_l187_187359

theorem gas_cost 
  (x : ℝ)
  (h1 : 5 * (x / 5) = x)
  (h2 : 8 * (x / 8) = x)
  (h3 : (x / 5) - 15.50 = (x / 8)) : 
  x = 206.67 :=
by
  sorry

end gas_cost_l187_187359


namespace find_k_value_l187_187357

variable {a : ℕ → ℕ} {S : ℕ → ℕ} 

axiom sum_of_first_n_terms (n : ℕ) (hn : n > 0) : S n = a n / n
axiom exists_Sk_inequality (k : ℕ) (hk : k > 0) : 1 < S k ∧ S k < 9

theorem find_k_value 
  (k : ℕ) (hk : k > 0) (hS : S k = a k / k) (hSk : 1 < S k ∧ S k < 9)
  (h_cond : ∀ n > 0, S n = n * S n ∧ S (n - 1) = S n * (n - 1)) : 
  k = 4 :=
sorry

end find_k_value_l187_187357


namespace H2O_production_l187_187405

theorem H2O_production (n : Nat) (m : Nat)
  (h1 : n = 3)
  (h2 : m = 3) :
  n = m → n = 3 := by
  sorry

end H2O_production_l187_187405


namespace ellipse_equation_from_hyperbola_l187_187693

theorem ellipse_equation_from_hyperbola :
  (∃ (a b : ℝ), ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1) →
  (x^2 / 16 + y^2 / 12 = 1)) :=
by
  sorry

end ellipse_equation_from_hyperbola_l187_187693


namespace cost_of_one_pack_l187_187426

-- Given condition
def total_cost (packs: ℕ) : ℕ := 110
def number_of_packs : ℕ := 10

-- Question: How much does one pack cost?
-- We need to prove that one pack costs 11 dollars
theorem cost_of_one_pack : (total_cost number_of_packs) / number_of_packs = 11 :=
by
  sorry

end cost_of_one_pack_l187_187426


namespace value_of_h_l187_187876

theorem value_of_h (h : ℤ) : (-1)^3 + h * (-1) - 20 = 0 → h = -21 :=
by
  intro h_cond
  sorry

end value_of_h_l187_187876


namespace original_total_price_l187_187299

theorem original_total_price (total_selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) 
  (selling_price_with_profit : total_selling_price/2 = original_price * (1 + profit_percent))
  (selling_price_with_loss : total_selling_price/2 = original_price * (1 - loss_percent)) :
  (original_price / (1 + profit_percent) + original_price / (1 - loss_percent) = 1333 + 1 / 3) := 
by
  sorry

end original_total_price_l187_187299


namespace probability_of_black_ball_l187_187355

theorem probability_of_black_ball 
  (p_red : ℝ)
  (p_white : ℝ)
  (h_red : p_red = 0.43)
  (h_white : p_white = 0.27)
  : (1 - p_red - p_white) = 0.3 :=
by 
  sorry

end probability_of_black_ball_l187_187355


namespace Glorys_favorite_number_l187_187025

variable (M G : ℝ)

theorem Glorys_favorite_number :
  (M = G / 3) →
  (M + G = 600) →
  (G = 450) :=
by
sorry

end Glorys_favorite_number_l187_187025


namespace square_side_length_l187_187698

/-- 
If a square is drawn by joining the midpoints of the sides of a given square and repeating this process continues indefinitely,
and the sum of the areas of all the squares is 32 cm²,
then the length of the side of the first square is 4 cm. 
-/
theorem square_side_length (s : ℝ) (h : ∑' n : ℕ, (s^2) * (1 / 2)^n = 32) : s = 4 := 
by 
  sorry

end square_side_length_l187_187698


namespace tournament_byes_and_games_l187_187599

/-- In a single-elimination tournament with 300 players initially registered,
- if the number of players in each subsequent round must be a power of 2,
- then 44 players must receive a bye in the first round, and 255 total games
- must be played to determine the champion. -/
theorem tournament_byes_and_games :
  let initial_players := 300
  let pow2_players := 256
  44 = initial_players - pow2_players ∧
  255 = pow2_players - 1 :=
by
  let initial_players := 300
  let pow2_players := 256
  have h_byes : 44 = initial_players - pow2_players := by sorry
  have h_games : 255 = pow2_players - 1 := by sorry
  exact ⟨h_byes, h_games⟩

end tournament_byes_and_games_l187_187599


namespace prime_divisors_of_1320_l187_187106

theorem prime_divisors_of_1320 : 
  ∃ (S : Finset ℕ), (S = {2, 3, 5, 11}) ∧ S.card = 4 := 
by
  sorry

end prime_divisors_of_1320_l187_187106


namespace find_common_difference_l187_187962

variable (a : ℕ → ℤ)  -- define the arithmetic sequence as a function from ℕ to ℤ
variable (d : ℤ)      -- define the common difference

-- Define the conditions
def conditions := (a 5 = 10) ∧ (a 12 = 31)

-- Define the formula for the nth term of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) (n : ℕ) := a 1 + d * (n - 1)

-- Prove that the common difference d is 3 given the conditions
theorem find_common_difference (h : conditions a) : d = 3 :=
sorry

end find_common_difference_l187_187962


namespace product_of_good_numbers_does_not_imply_sum_digits_property_l187_187060

-- Define what it means for a number to be "good".
def is_good (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

-- Define the sum of the digits of a number
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem statement
theorem product_of_good_numbers_does_not_imply_sum_digits_property :
  ∀ (A B : ℕ), is_good A → is_good B → is_good (A * B) →
  ¬ (sum_digits (A * B) = sum_digits A * sum_digits B) :=
by
  intros A B hA hB hAB
  -- The detailed proof is not provided here, hence we use sorry to skip it.
  sorry

end product_of_good_numbers_does_not_imply_sum_digits_property_l187_187060


namespace maria_towels_l187_187342

theorem maria_towels (green_towels white_towels given_towels : ℕ) (bought_green : green_towels = 40) 
(bought_white : white_towels = 44) (gave_mother : given_towels = 65) : 
  green_towels + white_towels - given_towels = 19 := by
sorry

end maria_towels_l187_187342


namespace total_obstacle_course_time_l187_187642

-- Definitions for the given conditions
def first_part_time : Nat := 7 * 60 + 23
def second_part_time : Nat := 73
def third_part_time : Nat := 5 * 60 + 58

-- State the main theorem
theorem total_obstacle_course_time :
  first_part_time + second_part_time + third_part_time = 874 :=
by
  sorry

end total_obstacle_course_time_l187_187642


namespace exists_integers_for_x_squared_minus_y_squared_eq_a_fifth_l187_187333

theorem exists_integers_for_x_squared_minus_y_squared_eq_a_fifth (a : ℤ) : 
  ∃ x y : ℤ, x^2 - y^2 = a^5 :=
sorry

end exists_integers_for_x_squared_minus_y_squared_eq_a_fifth_l187_187333


namespace find_number_of_children_l187_187949

theorem find_number_of_children (C B : ℕ) (H1 : B = 2 * C) (H2 : B = 4 * (C - 360)) : C = 720 := 
by
  sorry

end find_number_of_children_l187_187949


namespace find_y_l187_187047

theorem find_y (x y : ℤ) (h1 : x + y = 300) (h2 : x - y = 200) : y = 50 :=
sorry

end find_y_l187_187047


namespace cricket_team_right_handed_count_l187_187712

theorem cricket_team_right_handed_count 
  (total throwers non_throwers left_handed_non_throwers right_handed_non_throwers : ℕ) 
  (h_total : total = 70)
  (h_throwers : throwers = 37)
  (h_non_throwers : non_throwers = total - throwers)
  (h_left_handed_non_throwers : left_handed_non_throwers = non_throwers / 3)
  (h_right_handed_non_throwers : right_handed_non_throwers = non_throwers - left_handed_non_throwers)
  (h_all_throwers_right_handed : ∀ (t : ℕ), t = throwers → t = right_handed_non_throwers + (total - throwers) - (non_throwers / 3)) :
  right_handed_non_throwers + throwers = 59 := 
by 
  sorry

end cricket_team_right_handed_count_l187_187712


namespace simplify_expression_l187_187970

variable (a : ℤ)

theorem simplify_expression : (-2 * a) ^ 3 * a ^ 3 + (-3 * a ^ 3) ^ 2 = a ^ 6 :=
by sorry

end simplify_expression_l187_187970


namespace part1_part2_part3_l187_187664

variable {x y : ℚ}

def star (x y : ℚ) : ℚ := x * y + 1

theorem part1 : star 2 4 = 9 := by
  sorry

theorem part2 : star (star 1 4) (-2) = -9 := by
  sorry

theorem part3 (a b c : ℚ) : star a (b + c) + 1 = star a b + star a c := by
  sorry

end part1_part2_part3_l187_187664


namespace quadratic_function_correct_value_l187_187222

noncomputable def quadratic_function_value (a b x x1 x2 : ℝ) :=
  a * x^2 + b * x + 5

theorem quadratic_function_correct_value
  (a b x1 x2 : ℝ)
  (h_a : a ≠ 0)
  (h_A : quadratic_function_value a b x1 x1 x2 = 2002)
  (h_B : quadratic_function_value a b x2 x1 x2 = 2002) :
  quadratic_function_value a b (x1 + x2) x1 x2 = 5 :=
by
  sorry

end quadratic_function_correct_value_l187_187222


namespace simplify_expression_l187_187157

theorem simplify_expression (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ 2) (h3 : x ≠ 4) (h4 : x ≠ 5) :
  ( (x^2 - 4*x + 3) / (x^2 - 6*x + 9) ) / ( (x^2 - 6*x + 8) / (x^2 - 8*x + 15) ) =
  ( (x - 1) * (x - 5) ) / ( (x - 3) * (x - 4) * (x - 2) ) :=
by
  sorry

end simplify_expression_l187_187157


namespace eq_of_divisible_l187_187650

theorem eq_of_divisible (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : a + b ∣ 5 * a + 3 * b) : a = b :=
sorry

end eq_of_divisible_l187_187650


namespace find_missing_digit_l187_187462

theorem find_missing_digit (B : ℕ) : 
  (B = 2 ∨ B = 4 ∨ B = 7 ∨ B = 8 ∨ B = 9) → 
  (2 * 1000 + B * 100 + 4 * 10 + 0) % 15 = 0 → 
  B = 7 :=
by 
  intro h1 h2
  sorry

end find_missing_digit_l187_187462


namespace distinct_ordered_pairs_l187_187022

/-- There are 9 distinct ordered pairs of positive integers (m, n) such that the sum of the 
    reciprocals of m and n equals 1/6. -/
theorem distinct_ordered_pairs : 
  ∃ (s : Finset (ℕ × ℕ)), s.card = 9 ∧ 
  ∀ (p : ℕ × ℕ), p ∈ s → 
    (0 < p.1 ∧ 0 < p.2) ∧ 
    (1 / (p.1 : ℚ) + 1 / (p.2 : ℚ) = 1 / 6) :=
sorry

end distinct_ordered_pairs_l187_187022


namespace colored_pencils_count_l187_187516

-- Given conditions
def bundles := 7
def pencils_per_bundle := 10
def extra_colored_pencils := 3

-- Calculations based on conditions
def total_pencils : ℕ := bundles * pencils_per_bundle
def total_colored_pencils : ℕ := total_pencils + extra_colored_pencils

-- Statement to be proved
theorem colored_pencils_count : total_colored_pencils = 73 := by
  sorry

end colored_pencils_count_l187_187516


namespace part_one_part_two_l187_187195

noncomputable def f (a x : ℝ) := a * Real.log x - x + 1

theorem part_one (a : ℝ) (h : ∀ x > 0, f a x ≤ 0) : a = 1 := 
sorry

theorem part_two (h₁ : ∀ x > 0, f 1 x ≤ 0) (x : ℝ) (h₂ : 0 < x) (h₃ : x < Real.pi / 2) :
  Real.exp x * Real.sin x - x > f 1 x :=
sorry

end part_one_part_two_l187_187195


namespace initial_water_amount_l187_187277

variable (W : ℝ)
variable (evap_per_day : ℝ := 0.014)
variable (days : ℕ := 50)
variable (evap_percent : ℝ := 7.000000000000001)

theorem initial_water_amount :
  evap_per_day * (days : ℝ) = evap_percent / 100 * W → W = 10 :=
by
  sorry

end initial_water_amount_l187_187277


namespace maria_savings_percentage_is_33_l187_187844

noncomputable def regular_price : ℝ := 60
noncomputable def second_pair_price : ℝ := regular_price - (0.4 * regular_price)
noncomputable def third_pair_price : ℝ := regular_price - (0.6 * regular_price)
noncomputable def total_regular_price : ℝ := 3 * regular_price
noncomputable def total_discounted_price : ℝ := regular_price + second_pair_price + third_pair_price
noncomputable def savings : ℝ := total_regular_price - total_discounted_price
noncomputable def savings_percentage : ℝ := (savings / total_regular_price) * 100

theorem maria_savings_percentage_is_33 :
  savings_percentage = 33 :=
by
  sorry

end maria_savings_percentage_is_33_l187_187844


namespace eu_countries_2012_forms_set_l187_187067

def higher_level_skills_students := false -- Condition A can't form a set.
def tall_trees := false -- Condition B can't form a set.
def developed_cities := false -- Condition D can't form a set.
def eu_countries_2012 := true -- Condition C forms a set.

theorem eu_countries_2012_forms_set : 
  higher_level_skills_students = false ∧ tall_trees = false ∧ developed_cities = false ∧ eu_countries_2012 = true :=
by {
  sorry
}

end eu_countries_2012_forms_set_l187_187067


namespace angle_measure_l187_187036

theorem angle_measure (α : ℝ) (h1 : α - (90 - α) = 20) : α = 55 := by
  -- Proof to be provided here
  sorry

end angle_measure_l187_187036


namespace num_valid_n_l187_187769

theorem num_valid_n : ∃ k, k = 4 ∧ ∀ n : ℕ, (0 < n ∧ n < 50 ∧ ∃ m : ℕ, m > 0 ∧ n = m * (50 - n)) ↔ 
  (n = 25 ∨ n = 40 ∨ n = 45 ∨ n = 48) :=
by 
  sorry

end num_valid_n_l187_187769


namespace GuntherFreeTime_l187_187099

def GuntherCleaning : Nat := 45 + 60 + 30 + 15

def TotalFreeTime : Nat := 180

theorem GuntherFreeTime : TotalFreeTime - GuntherCleaning = 30 := by
  sorry

end GuntherFreeTime_l187_187099


namespace job_pay_per_pound_l187_187865

def p := 2
def M := 8 -- Monday
def T := 3 * M -- Tuesday
def W := 0 -- Wednesday
def R := 18 -- Thursday
def total_picked := M + T + W + R -- total berries picked
def money := 100 -- total money wanted

theorem job_pay_per_pound :
  total_picked = 50 → p = money / total_picked :=
by
  intro h
  rw [h]
  norm_num
  exact rfl

end job_pay_per_pound_l187_187865


namespace num_ordered_pairs_l187_187369

open Real 

-- Define the conditions
def eq_condition (x y : ℕ) : Prop :=
  x * (sqrt y) + y * (sqrt x) + (sqrt (2006 * x * y)) - (sqrt (2006 * x)) - (sqrt (2006 * y)) - 2006 = 0

-- Define the main problem statement
theorem num_ordered_pairs : ∃ (n : ℕ), n = 8 ∧ (∀ (x y : ℕ), eq_condition x y → x * y = 2006) :=
by
  sorry

end num_ordered_pairs_l187_187369


namespace eleven_pow_603_mod_500_eq_331_l187_187212

theorem eleven_pow_603_mod_500_eq_331 : 11^603 % 500 = 331 := by
  sorry

end eleven_pow_603_mod_500_eq_331_l187_187212


namespace bird_height_l187_187954

theorem bird_height (cat_height dog_height avg_height : ℕ) 
  (cat_height_eq : cat_height = 92)
  (dog_height_eq : dog_height = 94)
  (avg_height_eq : avg_height = 95) :
  let total_height := avg_height * 3 
  let bird_height := total_height - (cat_height + dog_height)
  bird_height = 99 := 
by
  sorry

end bird_height_l187_187954


namespace sum_midpoints_x_sum_midpoints_y_l187_187211

-- Defining the problem conditions
variables (a b c d e f : ℝ)
-- Sum of the x-coordinates of the triangle vertices is 15
def sum_x_coords (a b c : ℝ) : Prop := a + b + c = 15
-- Sum of the y-coordinates of the triangle vertices is 12
def sum_y_coords (d e f : ℝ) : Prop := d + e + f = 12

-- Proving the sum of x-coordinates of midpoints of sides is 15
theorem sum_midpoints_x (h1 : sum_x_coords a b c) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := 
by  
  sorry

-- Proving the sum of y-coordinates of midpoints of sides is 12
theorem sum_midpoints_y (h2 : sum_y_coords d e f) : 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 12 := 
by  
  sorry

end sum_midpoints_x_sum_midpoints_y_l187_187211


namespace single_cow_single_bag_l187_187858

-- Definitions given in the problem conditions
def cows : ℕ := 26
def bags : ℕ := 26
def days : ℕ := 26

-- Statement to be proved
theorem single_cow_single_bag : (1 : ℕ) = 26 := sorry

end single_cow_single_bag_l187_187858


namespace cost_difference_is_120_l187_187248

-- Define the monthly costs and duration
def rent_monthly_cost : ℕ := 20
def buy_monthly_cost : ℕ := 30
def months_in_a_year : ℕ := 12

-- Annual cost definitions
def annual_rent_cost : ℕ := rent_monthly_cost * months_in_a_year
def annual_buy_cost : ℕ := buy_monthly_cost * months_in_a_year

-- The main theorem to prove the difference in annual cost is $120
theorem cost_difference_is_120 : annual_buy_cost - annual_rent_cost = 120 := by
  sorry

end cost_difference_is_120_l187_187248


namespace smallest_multiple_of_6_and_15_l187_187885

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ b' : ℕ, b' > 0 ∧ b' % 6 = 0 ∧ b' % 15 = 0 → b ≤ b' :=
sorry

end smallest_multiple_of_6_and_15_l187_187885


namespace sin_theta_correct_l187_187373

noncomputable def sin_theta (a : ℝ) (h1 : a ≠ 0) (h2 : Real.tan θ = -a) : Real :=
  -Real.sqrt 2 / 2

theorem sin_theta_correct (a : ℝ) (h1 : a ≠ 0) (h2 : Real.tan (Real.arctan (-a)) = -a) : sin_theta a h1 h2 = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_theta_correct_l187_187373


namespace correct_result_l187_187243

theorem correct_result (x : ℤ) (h : x * 3 - 5 = 103) : (x / 3) - 5 = 7 :=
sorry

end correct_result_l187_187243


namespace arithmetic_sequence_sum_nine_l187_187153

variable {α : Type*} [LinearOrderedField α]

/-- An arithmetic sequence (a_n) is defined by a starting term a_1 and a common difference d. -/
def arithmetic_seq (a d n : α) : α := a + (n - 1) * d

/-- The sum of the first n terms of an arithmetic sequence. -/
def arithmetic_sum (a d n : α) : α := n / 2 * (2 * a + (n - 1) * d)

/-- Prove that for a given arithmetic sequence where a_2 + a_4 + a_9 = 24, the sum of the first 9 terms is 72. -/
theorem arithmetic_sequence_sum_nine 
  {a d : α}
  (h : arithmetic_seq a d 2 + arithmetic_seq a d 4 + arithmetic_seq a d 9 = 24) :
  arithmetic_sum a d 9 = 72 := 
by
  sorry

end arithmetic_sequence_sum_nine_l187_187153


namespace sum_of_altitudes_l187_187564

theorem sum_of_altitudes (x y : ℝ) (hline : 10 * x + 8 * y = 80):
  let A := 1 / 2 * 8 * 10
  let hypotenuse := Real.sqrt (8 ^ 2 + 10 ^ 2)
  let third_altitude := 80 / hypotenuse
  let sum_altitudes := 8 + 10 + third_altitude
  sum_altitudes = 18 + 40 / Real.sqrt 41 := by
  sorry

end sum_of_altitudes_l187_187564


namespace solve_equation_l187_187952

theorem solve_equation (x : ℝ) (h : x ≠ 1) (h_eq : x / (x - 1) = (x - 3) / (2 * x - 2)) : x = -3 :=
by
  sorry

end solve_equation_l187_187952


namespace kitty_cleaning_time_l187_187968

theorem kitty_cleaning_time
    (picking_up_toys : ℕ := 5)
    (vacuuming : ℕ := 20)
    (dusting_furniture : ℕ := 10)
    (total_time_4_weeks : ℕ := 200)
    (weeks : ℕ := 4)
    : (total_time_4_weeks - weeks * (picking_up_toys + vacuuming + dusting_furniture)) / weeks = 15 := by
    sorry

end kitty_cleaning_time_l187_187968


namespace eval_expression_l187_187227

noncomputable def T := (1 / (Real.sqrt 10 - Real.sqrt 8)) + (1 / (Real.sqrt 8 - Real.sqrt 6)) + (1 / (Real.sqrt 6 - Real.sqrt 4))

theorem eval_expression : T = (Real.sqrt 10 + 2 * Real.sqrt 8 + 2 * Real.sqrt 6 + 2) / 2 := 
by
  sorry

end eval_expression_l187_187227


namespace polynomial_divisibility_l187_187478

theorem polynomial_divisibility :
  ∃ (p : Polynomial ℤ), (Polynomial.X ^ 2 - Polynomial.X + 2) * p = Polynomial.X ^ 15 + Polynomial.X ^ 2 + 100 :=
by
  sorry

end polynomial_divisibility_l187_187478


namespace final_position_relative_total_fuel_needed_l187_187840

noncomputable def navigation_records : List ℤ := [-7, 11, -6, 10, -5]

noncomputable def fuel_consumption_rate : ℝ := 0.5

theorem final_position_relative (records : List ℤ) : 
  (records.sum = 3) := by 
  sorry

theorem total_fuel_needed (records : List ℤ) (rate : ℝ) : 
  (rate * (records.map Int.natAbs).sum = 19.5) := by 
  sorry

#check final_position_relative navigation_records
#check total_fuel_needed navigation_records fuel_consumption_rate

end final_position_relative_total_fuel_needed_l187_187840


namespace christopher_age_l187_187660

variable (C G F : ℕ)

theorem christopher_age (h1 : G = C + 8) (h2 : F = C - 2) (h3 : C + G + F = 60) : C = 18 := by
  sorry

end christopher_age_l187_187660


namespace Victor_more_scoops_l187_187772

def ground_almonds : ℝ := 1.56
def white_sugar : ℝ := 0.75

theorem Victor_more_scoops :
  ground_almonds - white_sugar = 0.81 :=
by
  sorry

end Victor_more_scoops_l187_187772


namespace petya_points_l187_187960

noncomputable def points_after_disqualification : ℕ :=
4

theorem petya_points (players: ℕ) (initial_points: ℕ) (disqualified: ℕ) (new_points: ℕ) : 
  players = 10 → 
  initial_points < (players * (players - 1) / 2) / players → 
  disqualified = 2 → 
  (players - disqualified) * (players - disqualified - 1) / 2 = new_points →
  new_points / (players - disqualified) < points_after_disqualification →
  points_after_disqualification > new_points / (players - disqualified) →
  points_after_disqualification = 4 :=
by 
  intros 
  exact sorry

end petya_points_l187_187960


namespace quadratic_two_roots_l187_187748

theorem quadratic_two_roots (b : ℝ) : 
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, (x = x₁ ∨ x = x₂) ↔ (x^2 + b*x - 3 = 0)) :=
by
  -- Indicate that a proof is required here
  sorry

end quadratic_two_roots_l187_187748


namespace sequence_value_l187_187652

theorem sequence_value (x : ℕ) : 
  (5 - 2 = 1 * 3) ∧ 
  (11 - 5 = 2 * 3) ∧ 
  (20 - 11 = 3 * 3) ∧ 
  (x - 20 = 4 * 3) ∧ 
  (47 - x = 5 * 3) → 
  x = 32 :=
by 
  intros h 
  sorry

end sequence_value_l187_187652


namespace proof_problem_l187_187026

def pos_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
∀ n, 4 * S n = (a n + 1) ^ 2

def sequence_condition (a : ℕ → ℝ) : Prop :=
a 0 = 1 ∧ ∀ n, a (n + 1) - a n = 2

def sum_sequence_T (a : ℕ → ℝ) (T : ℕ → ℝ) :=
∀ n, T n = (1 - 1 / (2 * n + 1))

def range_k (T : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n, T n ≥ k → k ≤ 2 / 3

theorem proof_problem (a : ℕ → ℝ) (S T : ℕ → ℝ) (k : ℝ) :
  pos_sequence a S → sequence_condition a → sum_sequence_T a T → range_k T k :=
by sorry

end proof_problem_l187_187026


namespace area_quotient_eq_correct_l187_187789

noncomputable def is_in_plane (x y z : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2

def supports (x y z a b c : ℝ) : Prop :=
  (x ≥ a ∧ y ≥ b ∧ z < c) ∨ (x ≥ a ∧ y < b ∧ z ≥ c) ∨ (x < a ∧ y ≥ b ∧ z ≥ c)

def in_S (x y z : ℝ) : Prop :=
  is_in_plane x y z ∧ supports x y z 1 (2/3) (1/3)

noncomputable def area_S : ℝ := 
  -- Placeholder for the computed area of S
  sorry

noncomputable def area_T : ℝ := 
  -- Placeholder for the computed area of T
  sorry

theorem area_quotient_eq_correct :
  (area_S / area_T) = (3 / (8 * Real.sqrt 3)) := 
  sorry

end area_quotient_eq_correct_l187_187789


namespace sarah_interview_combinations_l187_187700

theorem sarah_interview_combinations : 
  (1 * 2 * (2 + 3) * 5 * 1) = 50 := 
by
  sorry

end sarah_interview_combinations_l187_187700


namespace dad_caught_more_l187_187702

theorem dad_caught_more {trouts_caleb : ℕ} (h₁ : trouts_caleb = 2) 
    (h₂ : ∃ trouts_dad : ℕ, trouts_dad = 3 * trouts_caleb) : 
    ∃ more_trouts : ℕ, more_trouts = 4 := by
  sorry

end dad_caught_more_l187_187702


namespace consequent_in_ratio_4_6_l187_187232

theorem consequent_in_ratio_4_6 (h : 4 = 6 * (20 / x)) : x = 30 := 
by
  have h' : 4 * x = 6 * 20 := sorry -- cross-multiplication
  have h'' : x = 120 / 4 := sorry -- solving for x
  have hx : x = 30 := sorry -- simplifying 120 / 4

  exact hx

end consequent_in_ratio_4_6_l187_187232


namespace probability_A_B_l187_187082

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l187_187082


namespace point_A_is_closer_to_origin_l187_187529

theorem point_A_is_closer_to_origin (A B : ℤ) (hA : A = -2) (hB : B = 3) : abs A < abs B := by 
sorry

end point_A_is_closer_to_origin_l187_187529


namespace children_on_bus_l187_187871

theorem children_on_bus (initial_children additional_children total_children : ℕ) (h1 : initial_children = 26) (h2 : additional_children = 38) : total_children = initial_children + additional_children → total_children = 64 :=
by
  -- Proof goes here
  sorry

end children_on_bus_l187_187871


namespace percentage_more_than_cost_price_l187_187235

noncomputable def SP : ℝ := 7350
noncomputable def CP : ℝ := 6681.818181818181

theorem percentage_more_than_cost_price : 
  (SP - CP) / CP * 100 = 10 :=
by
  sorry

end percentage_more_than_cost_price_l187_187235


namespace tessellation_coloring_l187_187019

theorem tessellation_coloring :
  ∀ (T : Type) (colors : T → ℕ) (adjacent : T → T → Prop),
    (∀ t1 t2, adjacent t1 t2 → colors t1 ≠ colors t2) → 
    (∃ c1 c2 c3, ∀ t, colors t = c1 ∨ colors t = c2 ∨ colors t = c3) :=
sorry

end tessellation_coloring_l187_187019


namespace find_c_l187_187064

variable (y c : ℝ)

theorem find_c (h : y > 0) (h_expr : (7 * y / 20 + c * y / 10) = 0.6499999999999999 * y) : c = 3 := by
  sorry

end find_c_l187_187064


namespace price_of_large_slice_is_250_l187_187253

noncomputable def priceOfLargeSlice (totalSlices soldSmallSlices totalRevenue smallSlicePrice: ℕ) : ℕ :=
  let totalRevenueSmallSlices := soldSmallSlices * smallSlicePrice
  let totalRevenueLargeSlices := totalRevenue - totalRevenueSmallSlices
  let soldLargeSlices := totalSlices - soldSmallSlices
  totalRevenueLargeSlices / soldLargeSlices

theorem price_of_large_slice_is_250 :
  priceOfLargeSlice 5000 2000 1050000 150 = 250 :=
by
  sorry

end price_of_large_slice_is_250_l187_187253


namespace staplers_left_l187_187417

-- Definitions based on conditions
def initial_staplers : ℕ := 50
def dozen : ℕ := 12
def reports_stapled : ℕ := 3 * dozen

-- Statement of the theorem
theorem staplers_left (h : initial_staplers = 50) (d : dozen = 12) (r : reports_stapled = 3 * dozen) :
  (initial_staplers - reports_stapled) = 14 :=
sorry

end staplers_left_l187_187417


namespace relay_race_time_l187_187058

theorem relay_race_time (M S J T : ℕ) 
(hJ : J = 30)
(hS : S = J + 10)
(hM : M = 2 * S)
(hT : T = M - 7) : 
M + S + J + T = 223 :=
by sorry

end relay_race_time_l187_187058


namespace susie_total_earnings_l187_187028

def pizza_prices (type : String) (is_whole : Bool) : ℝ :=
  match type, is_whole with
  | "Margherita", false => 3
  | "Margherita", true  => 15
  | "Pepperoni", false  => 4
  | "Pepperoni", true   => 18
  | "Veggie Supreme", false => 5
  | "Veggie Supreme", true  => 22
  | "Meat Lovers", false => 6
  | "Meat Lovers", true  => 25
  | "Hawaiian", false   => 4.5
  | "Hawaiian", true    => 20
  | _, _                => 0

def topping_price (is_weekend : Bool) : ℝ :=
  if is_weekend then 1 else 2

def happy_hour_price : ℝ := 3

noncomputable def susie_earnings : ℝ :=
  let margherita_slices := 12 * happy_hour_price + 12 * pizza_prices "Margherita" false
  let pepperoni_slices := 8 * happy_hour_price + 8 * pizza_prices "Pepperoni" false + 6 * topping_price true
  let veggie_supreme_pizzas := 4 * pizza_prices "Veggie Supreme" true + 8 * topping_price true
  let margherita_whole_discounted := 3 * pizza_prices "Margherita" true - (3 * pizza_prices "Margherita" true) * 0.1
  let meat_lovers_slices := 10 * happy_hour_price + 10 * pizza_prices "Meat Lovers" false
  let hawaiian_slices := 12 * pizza_prices "Hawaiian" false + 4 * topping_price true
  let pepperoni_whole := pizza_prices "Pepperoni" true + 3 * topping_price true
  margherita_slices + pepperoni_slices + veggie_supreme_pizzas + margherita_whole_discounted + meat_lovers_slices + hawaiian_slices + pepperoni_whole

theorem susie_total_earnings : susie_earnings = 439.5 := by
  sorry

end susie_total_earnings_l187_187028


namespace number_of_roots_of_unity_l187_187621

theorem number_of_roots_of_unity (n : ℕ) (z : ℂ) (c d : ℤ) (h1 : n ≥ 3) (h2 : z^n = 1) (h3 : z^3 + (c : ℂ) * z + (d : ℂ) = 0) : 
  ∃ k : ℕ, k = 4 :=
by sorry

end number_of_roots_of_unity_l187_187621


namespace no_linear_term_l187_187896

theorem no_linear_term (m : ℤ) : (∀ (x : ℤ), (x^2 - x + m) * (x - 8) = x^3 - 9*x^2 - 8*m → (8 + m) = 0) → m = -8 :=
by
  sorry

end no_linear_term_l187_187896


namespace marble_draw_probability_l187_187513

def probability_first_white_second_red : ℚ :=
  let total_marbles := 10
  let white_marbles := 6
  let red_marbles := 4
  let first_white_probability := white_marbles / total_marbles
  let remaining_marbles_after_first_draw := total_marbles - 1
  let second_red_probability := red_marbles / remaining_marbles_after_first_draw
  first_white_probability * second_red_probability

theorem marble_draw_probability :
  probability_first_white_second_red = 4 / 15 := by
  sorry

end marble_draw_probability_l187_187513


namespace determine_sunday_l187_187455

def Brother := Prop -- A type to represent a brother

variable (A B : Brother)
variable (T D : Brother) -- T representing Tweedledum, D representing Tweedledee

-- Conditions translated into Lean
variable (H1 : (A = T) → (B = D))
variable (H2 : (B = D) → (A = T))

-- Define the day of the week as a proposition
def is_sunday := Prop

-- We want to state that given H1 and H2, it is Sunday
theorem determine_sunday (H1 : (A = T) → (B = D)) (H2 : (B = D) → (A = T)) : is_sunday := sorry

end determine_sunday_l187_187455


namespace cost_per_scarf_l187_187606

-- Define the cost of each earring
def cost_of_earring : ℕ := 6000

-- Define the number of earrings
def num_earrings : ℕ := 2

-- Define the cost of the iPhone
def cost_of_iphone : ℕ := 2000

-- Define the number of scarves
def num_scarves : ℕ := 4

-- Define the total value of the swag bag
def total_swag_bag_value : ℕ := 20000

-- Define the total value of diamond earrings and the iPhone
def total_value_of_earrings_and_iphone : ℕ := (num_earrings * cost_of_earring) + cost_of_iphone

-- Define the total value of the scarves
def total_value_of_scarves : ℕ := total_swag_bag_value - total_value_of_earrings_and_iphone

-- Define the cost of each designer scarf
def cost_of_each_scarf : ℕ := total_value_of_scarves / num_scarves

-- Prove that each designer scarf costs $1,500
theorem cost_per_scarf : cost_of_each_scarf = 1500 := by
  sorry

end cost_per_scarf_l187_187606


namespace verify_n_l187_187823

noncomputable def find_n (n : ℕ) : Prop :=
  let widget_rate1 := 3                             -- Widgets per worker-hour from the first condition
  let whoosit_rate1 := 2                            -- Whoosits per worker-hour from the first condition
  let widget_rate3 := 1                             -- Widgets per worker-hour from the third condition
  let minutes_per_widget := 1                       -- Arbitrary unit time for one widget
  let minutes_per_whoosit := 2                      -- 2 times unit time for one whoosit based on problem statement
  let whoosit_rate3 := 2 / 3                        -- Whoosits per worker-hour from the third condition
  let widget_rate2 := 540 / (90 * 3 : ℕ)            -- Widgets per hour in the second condition
  let whoosit_rate2 := n / (90 * 3 : ℕ)             -- Whoosits per hour in the second condition
  widget_rate2 = 2 ∧ whoosit_rate2 = 4 / 3 ∧
  (minutes_per_widget < minutes_per_whoosit) ∧
  (whoosit_rate2 = (4 / 3 : ℚ) ↔ n = 360)

theorem verify_n : find_n 360 :=
by sorry

end verify_n_l187_187823


namespace value_of_b_minus_a_l187_187846

theorem value_of_b_minus_a (a b : ℝ) (h1 : |a| = 1) (h2 : |b| = 3) (h3 : a < b) : b - a = 2 ∨ b - a = 4 :=
sorry

end value_of_b_minus_a_l187_187846


namespace max_cars_and_quotient_l187_187808

-- Definition of the problem parameters
def car_length : ℕ := 5
def speed_per_car_length : ℕ := 10
def hour_in_seconds : ℕ := 3600
def one_kilometer_in_meters : ℕ := 1000
def distance_in_meters_per_hour (n : ℕ) : ℕ := (10 * n) * one_kilometer_in_meters
def unit_distance (n : ℕ) : ℕ := car_length * (n + 1)

-- Hypotheses
axiom car_spacing : ∀ n : ℕ, unit_distance n = car_length * (n + 1)
axiom car_speed : ∀ n : ℕ, distance_in_meters_per_hour n = (10 * n) * one_kilometer_in_meters

-- Maximum whole number of cars M that can pass in one hour and the quotient when M is divided by 10
theorem max_cars_and_quotient : ∃ (M : ℕ), M = 3000 ∧ M / 10 = 300 := by
  sorry

end max_cars_and_quotient_l187_187808


namespace solve_quadratic_eq_solve_cubic_eq_l187_187445

-- Statement for the first equation
theorem solve_quadratic_eq (x : ℝ) : 9 * x^2 - 25 = 0 ↔ x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Statement for the second equation
theorem solve_cubic_eq (x : ℝ) : (x + 1)^3 - 27 = 0 ↔ x = 2 :=
by sorry

end solve_quadratic_eq_solve_cubic_eq_l187_187445


namespace geometric_sequence_divisibility_l187_187667

theorem geometric_sequence_divisibility 
  (a1 : ℚ) (h1 : a1 = 1 / 2) 
  (a2 : ℚ) (h2 : a2 = 10) 
  (n : ℕ) :
  ∃ (n : ℕ), a_n = (a1 * 20^(n - 1)) ∧ (n ≥ 4) ∧ (5000 ∣ a_n) :=
by
  sorry

end geometric_sequence_divisibility_l187_187667


namespace option_D_not_equal_l187_187905

def frac1 := (-15 : ℚ) / 12
def fracA := (-30 : ℚ) / 24
def fracB := -1 - (3 : ℚ) / 12
def fracC := -1 - (9 : ℚ) / 36
def fracD := -1 - (5 : ℚ) / 15
def fracE := -1 - (25 : ℚ) / 100

theorem option_D_not_equal :
  fracD ≠ frac1 := 
sorry

end option_D_not_equal_l187_187905


namespace prove_angle_BFD_l187_187350

def given_conditions (A : ℝ) (AFG AGF : ℝ) : Prop :=
  A = 40 ∧ AFG = AGF

theorem prove_angle_BFD (A AFG AGF BFD : ℝ) (h1 : given_conditions A AFG AGF) : BFD = 110 :=
  by
  -- Utilize the conditions h1 stating that A = 40 and AFG = AGF
  sorry

end prove_angle_BFD_l187_187350


namespace unique_solution_f_l187_187115

def f : ℕ → ℕ
  := sorry

namespace ProofProblem

theorem unique_solution_f (f : ℕ → ℕ)
  (h1 : ∀ (m n : ℕ), f m + f n - m * n ≠ 0)
  (h2 : ∀ (m n : ℕ), f m + f n - m * n ∣ m * f m + n * f n)
  : (∀ n : ℕ, f n = n^2) :=
sorry

end ProofProblem

end unique_solution_f_l187_187115


namespace find_p_l187_187718

variable (m n p : ℝ)

theorem find_p (h1 : m = n / 7 - 2 / 5)
               (h2 : m + p = (n + 21) / 7 - 2 / 5) : p = 3 := by
  sorry

end find_p_l187_187718


namespace ten_digit_number_l187_187119

open Nat

theorem ten_digit_number (a : Fin 10 → ℕ) (h1 : a 4 = 2)
  (h2 : a 8 = 3)
  (h3 : ∀ i, i < 8 → a i * a (i + 1) * a (i + 2) = 24) :
  a = ![4, 2, 3, 4, 2, 3, 4, 2, 3, 4] :=
sorry

end ten_digit_number_l187_187119


namespace factory_needs_to_produce_l187_187902

-- Define the given conditions
def weekly_production_target : ℕ := 6500
def production_mon_tue_wed : ℕ := 3 * 1200
def production_thu : ℕ := 800
def total_production_mon_thu := production_mon_tue_wed + production_thu
def required_production_fri := weekly_production_target - total_production_mon_thu

-- The theorem we need to prove
theorem factory_needs_to_produce : required_production_fri = 2100 :=
by
  -- The proof would go here
  sorry

end factory_needs_to_produce_l187_187902


namespace sin_2x_value_l187_187390

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x)

theorem sin_2x_value (x : ℝ) (h1 : f x = 5 / 3) (h2 : -Real.pi / 6 < x) (h3 : x < Real.pi / 6) :
  Real.sin (2 * x) = (Real.sqrt 3 - 2 * Real.sqrt 2) / 6 := 
sorry

end sin_2x_value_l187_187390


namespace valid_pairs_l187_187742

-- Define the target function and condition
def satisfies_condition (k l : ℤ) : Prop :=
  (7 * k - 5) * (4 * l - 3) = (5 * k - 3) * (6 * l - 1)

-- The theorem stating the exact pairs that satisfy the condition
theorem valid_pairs :
  ∀ (k l : ℤ), satisfies_condition k l ↔
    (k = 0 ∧ l = 6) ∨
    (k = 1 ∧ l = -1) ∨
    (k = 6 ∧ l = -6) ∨
    (k = 13 ∧ l = -7) ∨
    (k = -2 ∧ l = -22) ∨
    (k = -3 ∧ l = -15) ∨
    (k = -8 ∧ l = -10) ∨
    (k = -15 ∧ l = -9) :=
by
  sorry

end valid_pairs_l187_187742


namespace altitude_segments_of_acute_triangle_l187_187620

/-- If two altitudes of an acute triangle divide the sides into segments of lengths 5, 3, 2, and x units,
then x is equal to 10. -/
theorem altitude_segments_of_acute_triangle (a b c d e : ℝ) (h1 : a = 5) (h2 : b = 3) (h3 : c = 2) (h4 : d = x) :
  x = 10 :=
by
  sorry

end altitude_segments_of_acute_triangle_l187_187620


namespace total_marbles_in_bowls_l187_187644

theorem total_marbles_in_bowls :
  let second_bowl := 600
  let first_bowl := 3 / 4 * second_bowl
  let third_bowl := 1 / 2 * first_bowl
  let fourth_bowl := 1 / 3 * second_bowl
  first_bowl + second_bowl + third_bowl + fourth_bowl = 1475 :=
by
  sorry

end total_marbles_in_bowls_l187_187644


namespace probability_outside_circle_is_7_over_9_l187_187977

noncomputable def probability_point_outside_circle :
    ℚ :=
sorry

theorem probability_outside_circle_is_7_over_9 :
    probability_point_outside_circle = 7 / 9 :=
sorry

end probability_outside_circle_is_7_over_9_l187_187977


namespace simplify_expression_is_one_fourth_l187_187348

noncomputable def fourth_root (x : ℝ) : ℝ := x ^ (1 / 4)
noncomputable def square_root (x : ℝ) : ℝ := x ^ (1 / 2)
noncomputable def simplified_expression : ℝ := (fourth_root 81 - square_root 12.25) ^ 2

theorem simplify_expression_is_one_fourth : simplified_expression = 1 / 4 := 
by
  sorry

end simplify_expression_is_one_fourth_l187_187348


namespace evening_minivans_l187_187388

theorem evening_minivans (total_minivans afternoon_minivans : ℕ) (h_total : total_minivans = 5) 
(h_afternoon : afternoon_minivans = 4) : total_minivans - afternoon_minivans = 1 := 
by
  sorry

end evening_minivans_l187_187388


namespace expansion_number_of_terms_l187_187084

theorem expansion_number_of_terms (A B : Finset ℕ) (hA : A.card = 4) (hB : B.card = 5) : (A.card * B.card = 20) :=
by 
  sorry

end expansion_number_of_terms_l187_187084


namespace prime_pattern_l187_187863

theorem prime_pattern (n x : ℕ) (h1 : x = (10^n - 1) / 9) (h2 : Prime x) : Prime n :=
sorry

end prime_pattern_l187_187863


namespace packing_peanuts_per_large_order_l187_187265

/-- Definitions of conditions as stated -/
def large_orders : ℕ := 3
def small_orders : ℕ := 4
def total_peanuts_used : ℕ := 800
def peanuts_per_small : ℕ := 50

/-- The statement to prove, ensuring all conditions are utilized in the definitions -/
theorem packing_peanuts_per_large_order : 
  ∃ L, large_orders * L + small_orders * peanuts_per_small = total_peanuts_used ∧ L = 200 := 
by
  use 200
  -- Adding the necessary proof steps
  have h1 : large_orders = 3 := rfl
  have h2 : small_orders = 4 := rfl
  have h3 : peanuts_per_small = 50 := rfl
  have h4 : total_peanuts_used = 800 := rfl
  sorry

end packing_peanuts_per_large_order_l187_187265


namespace smallest_sum_BB_b_l187_187854

theorem smallest_sum_BB_b (B b : ℕ) (hB : 1 ≤ B ∧ B ≤ 4) (hb : b > 6) (h : 31 * B = 4 * b + 4) : B + b = 8 :=
sorry

end smallest_sum_BB_b_l187_187854


namespace inv_seq_not_arith_seq_l187_187000

theorem inv_seq_not_arith_seq (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_arith : ∃ d : ℝ, d ≠ 0 ∧ b = a + d ∧ c = a + 2 * d) :
  ¬ ∃ d' : ℝ, ∀ i j k : ℝ, i = 1 / a → j = 1 / b → k = 1 / c → j - i = d' ∧ k - j = d' :=
sorry

end inv_seq_not_arith_seq_l187_187000


namespace ratio_of_compositions_l187_187697

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

theorem ratio_of_compositions :
  f (g (f 2)) / g (f (g 2)) = 41 / 7 :=
by
  -- Proof will go here
  sorry

end ratio_of_compositions_l187_187697


namespace number_of_tables_large_meeting_l187_187776

-- Conditions
def table_length : ℕ := 2
def table_width : ℕ := 1
def side_length_large_meeting : ℕ := 7

-- To be proved: number of tables needed for a large meeting is 12.
theorem number_of_tables_large_meeting : 
  let tables_per_side := side_length_large_meeting / (table_length + table_width)
  ∃ total_tables, total_tables = 4 * tables_per_side ∧ total_tables = 12 :=
by
  sorry

end number_of_tables_large_meeting_l187_187776


namespace timber_volume_after_two_years_correct_l187_187137

-- Definitions based on the conditions in the problem
variables (a p b : ℝ) -- Assume a, p, and b are real numbers

-- Timber volume after one year
def timber_volume_one_year (a p b : ℝ) : ℝ := a * (1 + p) - b

-- Timber volume after two years
def timber_volume_two_years (a p b : ℝ) : ℝ := (timber_volume_one_year a p b) * (1 + p) - b

-- Prove that the timber volume after two years is equal to the given expression
theorem timber_volume_after_two_years_correct (a p b : ℝ) :
  timber_volume_two_years a p b = a * (1 + p)^2 - (2 + p) * b := sorry

end timber_volume_after_two_years_correct_l187_187137


namespace scientific_notation_l187_187483

theorem scientific_notation (n : ℝ) (h1 : n = 17600) : ∃ a b, (a = 1.76) ∧ (b = 4) ∧ n = a * 10^b :=
by {
  sorry
}

end scientific_notation_l187_187483


namespace inequality_C_false_l187_187239

theorem inequality_C_false (a b : ℝ) (ha : 1 < a) (hb : 1 < b) : (1 / a) ^ (1 / b) ≤ 1 := 
sorry

end inequality_C_false_l187_187239


namespace graph_of_equation_l187_187857

theorem graph_of_equation :
  ∀ x y : ℝ, (2 * x - 3 * y) ^ 2 = 4 * x ^ 2 + 9 * y ^ 2 → (x = 0 ∨ y = 0) :=
by
  intros x y h
  sorry

end graph_of_equation_l187_187857


namespace julie_net_monthly_income_is_l187_187730

section JulieIncome

def starting_pay : ℝ := 5.00
def additional_experience_pay_per_year : ℝ := 0.50
def years_of_experience : ℕ := 3
def work_hours_per_day : ℕ := 8
def work_days_per_week : ℕ := 6
def bi_weekly_bonus : ℝ := 50.00
def tax_rate : ℝ := 0.12
def insurance_premium_per_month : ℝ := 40.00
def missed_days : ℕ := 1

-- Calculate Julie's net monthly income
def net_monthly_income : ℝ :=
    let hourly_wage := starting_pay + additional_experience_pay_per_year * years_of_experience
    let daily_earnings := hourly_wage * work_hours_per_day
    let weekly_earnings := daily_earnings * (work_days_per_week - missed_days)
    let bi_weekly_earnings := weekly_earnings * 2
    let gross_monthly_income := bi_weekly_earnings * 2 + bi_weekly_bonus * 2
    let tax_deduction := gross_monthly_income * tax_rate
    let total_deductions := tax_deduction + insurance_premium_per_month
    gross_monthly_income - total_deductions

theorem julie_net_monthly_income_is : net_monthly_income = 963.20 :=
    sorry

end JulieIncome

end julie_net_monthly_income_is_l187_187730


namespace train_crossing_platform_time_l187_187770

theorem train_crossing_platform_time
  (length_train : ℝ)
  (length_platform : ℝ)
  (time_signal_pole : ℝ)
  (speed : ℝ)
  (time_platform_cross : ℝ)
  (v := length_train / time_signal_pole)
  (d := length_train + length_platform)
  (t := d / v) :
  length_train = 300 →
  length_platform = 250 →
  time_signal_pole = 18 →
  time_platform_cross = 33 →
  t = time_platform_cross := by
  sorry

end train_crossing_platform_time_l187_187770


namespace value_of_expression_l187_187918

theorem value_of_expression (a b : ℝ) (h : -3 * a - b = -1) : 3 - 6 * a - 2 * b = 1 :=
by
  sorry

end value_of_expression_l187_187918


namespace smallest_positive_multiple_of_6_and_15_gt_40_l187_187549

-- Define the LCM function to compute the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Define the statement of the proof problem
theorem smallest_positive_multiple_of_6_and_15_gt_40 : 
  ∃ a : ℕ, (a % 6 = 0) ∧ (a % 15 = 0) ∧ (a > 40) ∧ (∀ b : ℕ, (b % 6 = 0) ∧ (b % 15 = 0) ∧ (b > 40) → a ≤ b) :=
sorry

end smallest_positive_multiple_of_6_and_15_gt_40_l187_187549


namespace expansion_correct_l187_187123

variable (x y : ℝ)

theorem expansion_correct : 
  (3 * x - 15) * (4 * y + 20) = 12 * x * y + 60 * x - 60 * y - 300 :=
by
  sorry

end expansion_correct_l187_187123


namespace problem_proof_l187_187890

theorem problem_proof (x y : ℝ) (h_cond : (x + 3)^2 + |y - 2| = 0) : (x + y)^y = 1 :=
by
  sorry

end problem_proof_l187_187890


namespace find_point_coordinates_l187_187624

theorem find_point_coordinates (P : ℝ × ℝ)
  (h1 : P.1 < 0) -- Point P is in the second quadrant, so x < 0
  (h2 : P.2 > 0) -- Point P is in the second quadrant, so y > 0
  (h3 : abs P.2 = 4) -- distance from P to x-axis is 4
  (h4 : abs P.1 = 5) -- distance from P to y-axis is 5
  : P = (-5, 4) :=
by {
  -- point P is in the second quadrant, so x < 0 and y > 0
  -- |y| = 4 -> y = 4 
  -- |x| = 5 -> x = -5
  sorry
}

end find_point_coordinates_l187_187624


namespace Georgie_prank_l187_187726

theorem Georgie_prank (w : ℕ) (condition1 : w = 8) : 
  ∃ (ways : ℕ), ways = 336 := 
by
  sorry

end Georgie_prank_l187_187726


namespace three_scientists_same_topic_l187_187201

theorem three_scientists_same_topic
  (scientists : Finset ℕ)
  (h_size : scientists.card = 17)
  (topics : Finset ℕ)
  (h_topics : topics.card = 3)
  (communicates : ℕ → ℕ → ℕ)
  (h_communicate : ∀ a b : ℕ, a ≠ b → b ∈ scientists → communicates a b ∈ topics) :
  ∃ (a b c : ℕ), a ∈ scientists ∧ b ∈ scientists ∧ c ∈ scientists ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  communicates a b = communicates b c ∧ communicates b c = communicates a c := 
sorry

end three_scientists_same_topic_l187_187201


namespace ratio_of_oranges_to_limes_l187_187941

-- Constants and Definitions
def initial_fruits : ℕ := 150
def half_fruits : ℕ := 75
def oranges : ℕ := 50
def limes : ℕ := half_fruits - oranges
def ratio_oranges_limes : ℕ × ℕ := (oranges / Nat.gcd oranges limes, limes / Nat.gcd oranges limes)

-- Theorem Statement
theorem ratio_of_oranges_to_limes : ratio_oranges_limes = (2, 1) := by
  sorry

end ratio_of_oranges_to_limes_l187_187941


namespace find_fibonacci_x_l187_187819

def is_fibonacci (a b c : ℕ) : Prop :=
  c = a + b

theorem find_fibonacci_x (a b x : ℕ)
  (h₁ : a = 8)
  (h₂ : b = 13)
  (h₃ : is_fibonacci a b x) :
  x = 21 :=
by
  sorry

end find_fibonacci_x_l187_187819


namespace opposite_event_is_at_least_one_hit_l187_187315

def opposite_event_of_missing_both_times (hit1 hit2 : Prop) : Prop :=
  ¬(¬hit1 ∧ ¬hit2)

theorem opposite_event_is_at_least_one_hit (hit1 hit2 : Prop) :
  opposite_event_of_missing_both_times hit1 hit2 = (hit1 ∨ hit2) :=
by
  sorry

end opposite_event_is_at_least_one_hit_l187_187315


namespace find_a_and_theta_find_max_min_g_l187_187061

noncomputable def f (x a θ : ℝ) : ℝ := (a + 2 * (Real.cos x)^2) * Real.cos (2 * x + θ)

-- Provided conditions
variable (a : ℝ)
variable (θ : ℝ)
variable (is_odd : ∀ x, f x a θ = -f (-x) a θ)
variable (f_pi_over_4 : f ((Real.pi) / 4) a θ = 0)
variable (theta_in_range : 0 < θ ∧ θ < Real.pi)

-- To Prove
theorem find_a_and_theta :
  a = -1 ∧ θ = (Real.pi / 2) :=
sorry

-- Define g(x) and its domain
noncomputable def g (x : ℝ) : ℝ := f x (-1) (Real.pi / 2) + f (x + (Real.pi / 3)) (-1) (Real.pi / 2)

-- Provided domain condition
variable (x_in_domain : 0 ≤ x ∧ x ≤ (Real.pi / 4))

-- To Prove maximum and minimum value of g(x)
theorem find_max_min_g :
  (∀ x, x ∈ Set.Icc (0 : ℝ) (Real.pi / 4) → -((Real.sqrt 3) / 2) ≤ g x ∧ g x ≤ (Real.sqrt 3) / 2)
  ∧ ∃ x_min, g x_min = -((Real.sqrt 3) / 2) ∧ x_min = (Real.pi / 8)
  ∧ ∃ x_max, g x_max = ((Real.sqrt 3) / 2) ∧ x_max = (Real.pi / 4) :=
sorry

end find_a_and_theta_find_max_min_g_l187_187061


namespace sqrt_product_simplified_l187_187129

theorem sqrt_product_simplified (q : ℝ) (hq : 0 < q) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by
  sorry

end sqrt_product_simplified_l187_187129


namespace power_division_simplify_l187_187580

theorem power_division_simplify :
  ((9^9 / 9^8)^2 * 3^4) / 2^4 = 410 + 1/16 := by
  sorry

end power_division_simplify_l187_187580


namespace unsuitable_temperature_for_refrigerator_l187_187068

theorem unsuitable_temperature_for_refrigerator:
  let avg_temp := -18
  let variation := 2
  let min_temp := avg_temp - variation
  let max_temp := avg_temp + variation
  let temp_A := -17
  let temp_B := -18
  let temp_C := -19
  let temp_D := -22
  temp_D < min_temp ∨ temp_D > max_temp := by
  sorry

end unsuitable_temperature_for_refrigerator_l187_187068


namespace river_depth_l187_187942

theorem river_depth (width depth : ℝ) (flow_rate_kmph : ℝ) (volume_m3_per_min : ℝ) 
  (h1 : width = 75) 
  (h2 : flow_rate_kmph = 4) 
  (h3 : volume_m3_per_min = 35000) : 
  depth = 7 := 
by
  sorry

end river_depth_l187_187942


namespace blue_first_red_second_probability_l187_187270

-- Define the initial conditions
def initial_red_marbles : ℕ := 4
def initial_white_marbles : ℕ := 6
def initial_blue_marbles : ℕ := 2
def total_marbles : ℕ := initial_red_marbles + initial_white_marbles + initial_blue_marbles

-- Probability calculation under the given conditions
def probability_blue_first : ℚ := initial_blue_marbles / total_marbles
def remaining_marbles_after_blue : ℕ := total_marbles - 1
def remaining_red_marbles : ℕ := initial_red_marbles
def probability_red_second_given_blue_first : ℚ := remaining_red_marbles / remaining_marbles_after_blue

-- Combined probability
def combined_probability : ℚ := probability_blue_first * probability_red_second_given_blue_first

-- The statement to be proved
theorem blue_first_red_second_probability :
  combined_probability = 2 / 33 :=
sorry

end blue_first_red_second_probability_l187_187270


namespace fraction_to_decimal_l187_187135

theorem fraction_to_decimal :
  (3 / 8 : ℝ) = 0.375 :=
sorry

end fraction_to_decimal_l187_187135


namespace solveSystem_l187_187891

variable {r p q x y z : ℝ}

theorem solveSystem :
  
  -- The given system of equations
  (x + r * y - q * z = 1) ∧
  (-r * x + y + p * z = r) ∧ 
  (q * x - p * y + z = -q) →

  -- Solution equivalence using determined
  x = (1 - r ^ 2 + p ^ 2 - q ^ 2) / (1 + r ^ 2 + p ^ 2 + q ^ 2) :=
by sorry

end solveSystem_l187_187891


namespace trigonometric_identity1_trigonometric_identity2_l187_187182

theorem trigonometric_identity1 (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin (Real.pi - θ) + Real.cos (θ - Real.pi)) / (Real.sin (θ + Real.pi) + Real.cos (θ + Real.pi)) = -1/3 :=
by
  sorry

theorem trigonometric_identity2 (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin (2 * θ) = 4/5 :=
by
  sorry

end trigonometric_identity1_trigonometric_identity2_l187_187182


namespace power_of_5_in_8_factorial_l187_187797

theorem power_of_5_in_8_factorial :
  let x := Nat.factorial 8
  ∃ (i k m p : ℕ), 0 < i ∧ 0 < k ∧ 0 < m ∧ 0 < p ∧ x = 2^i * 3^k * 5^m * 7^p ∧ m = 1 :=
by
  sorry

end power_of_5_in_8_factorial_l187_187797


namespace fair_attendance_l187_187934

theorem fair_attendance (x y z : ℕ) 
    (h1 : y = 2 * x)
    (h2 : z = y - 200)
    (h3 : x + y + z = 2800) : x = 600 := by
  sorry

end fair_attendance_l187_187934


namespace perfect_square_condition_l187_187009

theorem perfect_square_condition (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
    (gcd_xyz : Nat.gcd (Nat.gcd x y) z = 1)
    (hx_dvd : x ∣ y * z * (x + y + z))
    (hy_dvd : y ∣ x * z * (x + y + z))
    (hz_dvd : z ∣ x * y * (x + y + z))
    (sum_dvd : x + y + z ∣ x * y * z) :
  ∃ m : ℕ, m * m = x * y * z * (x + y + z) := sorry

end perfect_square_condition_l187_187009


namespace machines_produce_12x_boxes_in_expected_time_l187_187658

-- Definitions corresponding to the conditions
def rate_A (x : ℕ) := x / 10
def rate_B (x : ℕ) := 2 * x / 5
def rate_C (x : ℕ) := 3 * x / 8
def rate_D (x : ℕ) := x / 4

-- Total combined rate when working together
def combined_rate (x : ℕ) := rate_A x + rate_B x + rate_C x + rate_D x

-- The time taken to produce 12x boxes given their combined rate
def time_to_produce (x : ℕ) : ℕ := 12 * x / combined_rate x

-- Goal: Time taken should be 32/3 minutes
theorem machines_produce_12x_boxes_in_expected_time (x : ℕ) : time_to_produce x = 32 / 3 :=
sorry

end machines_produce_12x_boxes_in_expected_time_l187_187658


namespace expression_value_l187_187668

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 3) :
  (a + b) / m - c * d + m = 2 ∨ (a + b) / m - c * d + m = -4 := 
by
  sorry

end expression_value_l187_187668


namespace no_integer_solution_l187_187389

theorem no_integer_solution (a b c d : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) :
  ¬ ∃ n : ℤ, n^4 - (a : ℤ)*n^3 - (b : ℤ)*n^2 - (c : ℤ)*n - (d : ℤ) = 0 :=
sorry

end no_integer_solution_l187_187389


namespace perfect_square_x4_x3_x2_x1_1_eq_x0_l187_187349

theorem perfect_square_x4_x3_x2_x1_1_eq_x0 :
  ∀ x : ℤ, ∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2 ↔ x = 0 :=
by sorry

end perfect_square_x4_x3_x2_x1_1_eq_x0_l187_187349


namespace lcm_14_21_35_l187_187711

-- Define the numbers
def a : ℕ := 14
def b : ℕ := 21
def c : ℕ := 35

-- Define the prime factorizations
def prime_factors_14 : List (ℕ × ℕ) := [(2, 1), (7, 1)]
def prime_factors_21 : List (ℕ × ℕ) := [(3, 1), (7, 1)]
def prime_factors_35 : List (ℕ × ℕ) := [(5, 1), (7, 1)]

-- Prove the least common multiple
theorem lcm_14_21_35 : Nat.lcm (Nat.lcm a b) c = 210 := by
  sorry

end lcm_14_21_35_l187_187711


namespace people_in_room_l187_187767

theorem people_in_room (people chairs : ℕ) (h1 : 5 / 8 * people = 4 / 5 * chairs)
  (h2 : chairs = 5 + 4 / 5 * chairs) : people = 32 :=
by
  sorry

end people_in_room_l187_187767


namespace correct_statement_C_l187_187290

-- Define the function
def linear_function (x : ℝ) : ℝ := -3 * x + 1

-- Define the condition for statement C
def statement_C (x : ℝ) : Prop := x > 1 / 3 → linear_function x < 0

-- The theorem to be proved
theorem correct_statement_C : ∀ x : ℝ, statement_C x := by
  sorry

end correct_statement_C_l187_187290


namespace time_spent_on_type_a_problems_l187_187932

theorem time_spent_on_type_a_problems 
  (total_problems : ℕ)
  (exam_time_minutes : ℕ)
  (type_a_problems : ℕ)
  (type_b_problem_time : ℕ)
  (total_time_type_a : ℕ)
  (h1 : total_problems = 200)
  (h2 : exam_time_minutes = 180)
  (h3 : type_a_problems = 50)
  (h4 : ∀ x : ℕ, type_b_problem_time = 2 * x)
  (h5 : ∀ x : ℕ, total_time_type_a = type_a_problems * type_b_problem_time)
  : total_time_type_a = 72 := 
by
  sorry

end time_spent_on_type_a_problems_l187_187932


namespace units_sold_to_customer_c_l187_187226

theorem units_sold_to_customer_c 
  (initial_units : ℕ)
  (defective_units : ℕ)
  (units_a : ℕ)
  (units_b : ℕ)
  (units_c : ℕ)
  (h_initial : initial_units = 20)
  (h_defective : defective_units = 5)
  (h_units_a : units_a = 3)
  (h_units_b : units_b = 5)
  (h_non_defective : initial_units - defective_units = 15)
  (h_sold_all : units_a + units_b + units_c = 15) :
  units_c = 7 := by
  -- use sorry to skip the proof
  sorry

end units_sold_to_customer_c_l187_187226


namespace payment_equation_1_payment_equation_2_cost_effective_40_combined_cost_effective_40_l187_187592

namespace ShoppingMall

def tea_set_price : ℕ := 200
def tea_bowl_price : ℕ := 20
def discount_option_1 (x : ℕ) : ℕ := 20 * x + 5400
def discount_option_2 (x : ℕ) : ℕ := 19 * x + 5700
def combined_option_40 : ℕ := 6000 + 190

theorem payment_equation_1 (x : ℕ) (hx : x > 30) : 
  discount_option_1 x = 20 * x + 5400 :=
by sorry

theorem payment_equation_2 (x : ℕ) (hx : x > 30) : 
  discount_option_2 x = 19 * x + 5700 :=
by sorry

theorem cost_effective_40 : discount_option_1 40 < discount_option_2 40 :=
by sorry

theorem combined_cost_effective_40 : combined_option_40 < discount_option_1 40 ∧ combined_option_40 < discount_option_2 40 :=
by sorry

end ShoppingMall

end payment_equation_1_payment_equation_2_cost_effective_40_combined_cost_effective_40_l187_187592


namespace percentage_reduction_l187_187640

variable (P R : ℝ)
variable (ReducedPrice : R = 15)
variable (AmountMore : 900 / 15 - 900 / P = 6)

theorem percentage_reduction (ReducedPrice : R = 15) (AmountMore : 900 / 15 - 900 / P = 6) :
  (P - R) / P * 100 = 10 :=
by
  sorry

end percentage_reduction_l187_187640


namespace value_of_y_l187_187965

theorem value_of_y (y : ℤ) (h : (2010 + y)^2 = y^2) : y = -1005 :=
sorry

end value_of_y_l187_187965


namespace marie_needs_8_days_to_pay_for_cash_register_l187_187300

-- Definitions of the conditions
def cost_of_cash_register : ℕ := 1040
def price_per_loaf : ℕ := 2
def loaves_per_day : ℕ := 40
def price_per_cake : ℕ := 12
def cakes_per_day : ℕ := 6
def daily_rent : ℕ := 20
def daily_electricity : ℕ := 2

-- Derive daily income and expenses
def daily_income : ℕ := (price_per_loaf * loaves_per_day) + (price_per_cake * cakes_per_day)
def daily_expenses : ℕ := daily_rent + daily_electricity
def daily_profit : ℕ := daily_income - daily_expenses

-- Define days needed to pay for the cash register
def days_needed : ℕ := cost_of_cash_register / daily_profit

-- Proof goal
theorem marie_needs_8_days_to_pay_for_cash_register : days_needed = 8 := by
  sorry

end marie_needs_8_days_to_pay_for_cash_register_l187_187300


namespace zoe_total_expenditure_is_correct_l187_187921

noncomputable def zoe_expenditure : ℝ :=
  let initial_app_cost : ℝ := 5
  let monthly_fee : ℝ := 8
  let first_two_months_fee : ℝ := 2 * monthly_fee
  let yearly_cost_without_discount : ℝ := 12 * monthly_fee
  let discount : ℝ := 0.15 * yearly_cost_without_discount
  let discounted_annual_plan : ℝ := yearly_cost_without_discount - discount
  let actual_annual_plan : ℝ := discounted_annual_plan - first_two_months_fee
  let in_game_items_cost : ℝ := 10
  let discounted_in_game_items_cost : ℝ := in_game_items_cost - (0.10 * in_game_items_cost)
  let upgraded_feature_cost : ℝ := 12
  let discounted_upgraded_feature_cost : ℝ := upgraded_feature_cost - (0.10 * upgraded_feature_cost)
  initial_app_cost + first_two_months_fee + actual_annual_plan + discounted_in_game_items_cost + discounted_upgraded_feature_cost

theorem zoe_total_expenditure_is_correct : zoe_expenditure = 122.4 :=
by
  sorry

end zoe_total_expenditure_is_correct_l187_187921


namespace picnic_students_count_l187_187171

theorem picnic_students_count (x : ℕ) (h1 : (x / 2) + (x / 3) + (x / 4) = 65) : x = 60 :=
by
  -- Proof goes here
  sorry

end picnic_students_count_l187_187171


namespace inequality_l187_187835

variable (a b m : ℝ)

theorem inequality (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < m) (h4 : a < b) :
  a / b < (a + m) / (b + m) :=
by
  sorry

end inequality_l187_187835


namespace min_sum_intercepts_of_line_l187_187091

theorem min_sum_intercepts_of_line (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 2 / b = 1) : a + b = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_sum_intercepts_of_line_l187_187091


namespace min_people_liking_both_l187_187325

theorem min_people_liking_both {A B U : Finset ℕ} (hU : U.card = 150) (hA : A.card = 130) (hB : B.card = 120) :
  (A ∩ B).card ≥ 100 :=
by
  -- Proof to be filled later
  sorry

end min_people_liking_both_l187_187325


namespace max_min_product_l187_187907

theorem max_min_product (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h1 : x + y + z = 15) (h2 : x * y + y * z + z * x = 45) :
    ∃ m : ℝ, m = min (x * y) (min (y * z) (z * x)) ∧ m ≤ 17.5 :=
by
  sorry

end max_min_product_l187_187907


namespace simplify_expression_l187_187393

variable (a b : ℤ)

theorem simplify_expression : 
  (15 * a + 45 * b) + (21 * a + 32 * b) - (12 * a + 40 * b) = 24 * a + 37 * b := 
    by sorry

end simplify_expression_l187_187393


namespace income_to_expenditure_ratio_l187_187780

-- Define the constants based on the conditions in step a)
def income : ℕ := 36000
def savings : ℕ := 4000

-- Define the expenditure as a function of income and savings
def expenditure (I S : ℕ) : ℕ := I - S

-- Define the ratio of two natural numbers
def ratio (a b : ℕ) : ℚ := a / b

-- Statement to be proved
theorem income_to_expenditure_ratio : 
  ratio income (expenditure income savings) = 9 / 8 :=
by
  sorry

end income_to_expenditure_ratio_l187_187780


namespace find_natural_numbers_l187_187728

theorem find_natural_numbers (n : ℕ) (p q : ℕ) (hp : p.Prime) (hq : q.Prime)
  (h : q = p + 2) (h1 : (2^n + p).Prime) (h2 : (2^n + q).Prime) :
    n = 1 ∨ n = 3 :=
by
  sorry

end find_natural_numbers_l187_187728


namespace second_discount_percentage_l187_187946

theorem second_discount_percentage 
    (original_price : ℝ) (final_price : ℝ) (first_discount : ℝ) (third_discount : ℝ) (second_discount : ℝ) :
      original_price = 9795.3216374269 →
      final_price = 6700 →
      first_discount = 0.20 →
      third_discount = 0.05 →
      (original_price * (1 - first_discount) * (1 - second_discount / 100) * (1 - third_discount) = final_price) →
      second_discount = 10 :=
by
  intros h_orig h_final h_first h_third h_eq
  sorry

end second_discount_percentage_l187_187946


namespace point_not_in_region_l187_187050

theorem point_not_in_region : ¬ (3 * 2 + 2 * 0 < 6) :=
by simp [lt_irrefl]

end point_not_in_region_l187_187050


namespace trigonometric_identity_application_l187_187409

theorem trigonometric_identity_application :
  (1 / 2) * (Real.sin (Real.pi / 12)) * (Real.cos (Real.pi / 12)) = (1 / 8) :=
by
  sorry

end trigonometric_identity_application_l187_187409


namespace range_of_a_l187_187584

variable (a : ℝ)

def proposition_p : Prop :=
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x^2 - a ≥ 0)

def proposition_q : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + (2 - a) = 0

theorem range_of_a (hp : proposition_p a) (hq : proposition_q a) : a ≤ -2 ∨ a = 1 :=
  sorry

end range_of_a_l187_187584


namespace sufficient_not_necessary_l187_187403

variable (x : ℝ)

theorem sufficient_not_necessary (h : |x| > 0) : (x > 0 ↔ true) :=
by 
  sorry

end sufficient_not_necessary_l187_187403


namespace units_digit_k_squared_plus_pow2_k_l187_187145

def n : ℕ := 4016
def k : ℕ := n^2 + 2^n

theorem units_digit_k_squared_plus_pow2_k :
  (k^2 + 2^k) % 10 = 7 := sorry

end units_digit_k_squared_plus_pow2_k_l187_187145


namespace remainder_1425_1427_1429_mod_12_l187_187924

theorem remainder_1425_1427_1429_mod_12 : 
  (1425 * 1427 * 1429) % 12 = 3 :=
by
  sorry

end remainder_1425_1427_1429_mod_12_l187_187924


namespace lcm_of_numbers_l187_187861

theorem lcm_of_numbers (a b : ℕ) (L : ℕ) 
  (h1 : a + b = 55) 
  (h2 : Nat.gcd a b = 5) 
  (h3 : (1 / (a : ℝ)) + (1 / (b : ℝ)) = 0.09166666666666666) : (Nat.lcm a b = 120) := 
sorry

end lcm_of_numbers_l187_187861


namespace rotation_phenomena_l187_187552

/-- 
The rotation of the hour hand fits the definition of rotation since it turns around 
the center of the clock, covering specific angles as time passes.
-/
def is_rotation_of_hour_hand : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The rotation of the Ferris wheel fits the definition of rotation since it turns around 
its central axis, making a complete circle.
-/
def is_rotation_of_ferris_wheel : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The annual decline of the groundwater level does not fit the definition of rotation 
since it is a vertical movement (translation).
-/
def is_not_rotation_of_groundwater_level : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The movement of the robots on the conveyor belt does not fit the definition of rotation 
since it is a linear/translational movement.
-/
def is_not_rotation_of_robots_on_conveyor : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
Proof that the phenomena which belong to rotation are exactly the rotation of the hour hand 
and the rotation of the Ferris wheel.
-/
theorem rotation_phenomena :
  is_rotation_of_hour_hand ∧ 
  is_rotation_of_ferris_wheel ∧ 
  is_not_rotation_of_groundwater_level ∧ 
  is_not_rotation_of_robots_on_conveyor →
  "①②" = "①②" :=
by
  intro h
  sorry

end rotation_phenomena_l187_187552


namespace find_range_of_m_l187_187785

-- Define properties of ellipses and hyperbolas
def isEllipseY (m : ℝ) : Prop := (8 - m > 2 * m - 1 ∧ 2 * m - 1 > 0)
def isHyperbola (m : ℝ) : Prop := (m + 1) * (m - 2) < 0

-- The range of 'm' such that (p ∨ q) is true and (p ∧ q) is false
def p_or_q_true_p_and_q_false (m : ℝ) : Prop := 
  (isEllipseY m ∨ isHyperbola m) ∧ ¬ (isEllipseY m ∧ isHyperbola m)

-- The range of the real number 'm'
def range_of_m (m : ℝ) : Prop := 
  (-1 < m ∧ m ≤ 1/2) ∨ (2 ≤ m ∧ m < 3)

-- Prove that the above conditions imply the correct range for m
theorem find_range_of_m (m : ℝ) : p_or_q_true_p_and_q_false m → range_of_m m :=
by
  sorry

end find_range_of_m_l187_187785


namespace third_vertex_coordinates_l187_187029

theorem third_vertex_coordinates (x : ℝ) (h : 6 * |x| = 96) : x = 16 ∨ x = -16 :=
by
  sorry

end third_vertex_coordinates_l187_187029


namespace intersecting_line_at_one_point_l187_187459

theorem intersecting_line_at_one_point (k : ℝ) :
  (∃ y : ℝ, k = -3 * y^2 - 4 * y + 7 ∧ 
           ∀ z : ℝ, k = -3 * z^2 - 4 * z + 7 → y = z) ↔ 
  k = 25 / 3 :=
by
  sorry

end intersecting_line_at_one_point_l187_187459


namespace min_sum_ab_max_product_ab_l187_187550

theorem min_sum_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) : a + b ≥ 2 :=
by
  sorry

theorem max_product_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : a * b ≤ 1 / 4 :=
by
  sorry

end min_sum_ab_max_product_ab_l187_187550


namespace total_oranges_in_stack_l187_187435

-- Definitions based on the given conditions
def base_layer_oranges : Nat := 5 * 8
def second_layer_oranges : Nat := 4 * 7
def third_layer_oranges : Nat := 3 * 6
def fourth_layer_oranges : Nat := 2 * 5
def fifth_layer_oranges : Nat := 1 * 4

-- Theorem statement equivalent to the math problem
theorem total_oranges_in_stack : base_layer_oranges + second_layer_oranges + third_layer_oranges + fourth_layer_oranges + fifth_layer_oranges = 100 :=
by
  sorry

end total_oranges_in_stack_l187_187435


namespace minimum_possible_value_of_Box_l187_187166

theorem minimum_possible_value_of_Box :
  ∃ a b : ℤ, a ≠ b ∧ a * b = 45 ∧ 
    (∀ c d : ℤ, c * d = 45 → c^2 + d^2 ≥ 106) ∧ a^2 + b^2 = 106 :=
by
  sorry

end minimum_possible_value_of_Box_l187_187166


namespace problem1_l187_187927

variable {x : ℝ} {b c : ℝ}

theorem problem1 (hb : b = 9) (hc : c = -11) :
  b + c = -2 := 
by
  simp [hb, hc]
  sorry

end problem1_l187_187927


namespace geometric_sequence_4th_term_is_2_5_l187_187893

variables (a r : ℝ) (n : ℕ)

def geometric_term (a r : ℝ) (n : ℕ) : ℝ := a * r^(n-1)

theorem geometric_sequence_4th_term_is_2_5 (a r : ℝ)
  (h1 : a = 125) 
  (h2 : geometric_term a r 8 = 72) :
  geometric_term a r 4 = 5 / 2 := 
sorry

end geometric_sequence_4th_term_is_2_5_l187_187893


namespace number_of_dodge_trucks_l187_187376

theorem number_of_dodge_trucks (V T F D : ℕ) (h1 : V = 5)
  (h2 : T = 2 * V) 
  (h3 : F = 2 * T)
  (h4 : F = D / 3) :
  D = 60 := 
by
  sorry

end number_of_dodge_trucks_l187_187376


namespace duration_of_each_class_is_3_l187_187160

theorem duration_of_each_class_is_3
    (weeks : ℕ) 
    (x : ℝ) 
    (weekly_additional_class_hours : ℝ) 
    (homework_hours_per_week : ℝ) 
    (total_hours : ℝ) 
    (h1 : weeks = 24)
    (h2 : weekly_additional_class_hours = 4)
    (h3 : homework_hours_per_week = 4)
    (h4 : total_hours = 336) :
    (2 * x + weekly_additional_class_hours + homework_hours_per_week) * weeks = total_hours → x = 3 := 
by 
  sorry

end duration_of_each_class_is_3_l187_187160


namespace line_symmetric_fixed_point_l187_187938

theorem line_symmetric_fixed_point (k : ℝ) :
  (∀ x, (∃ y, y = k * (x - 4))) →
  (∃ p : ℝ × ℝ, p = (2, 1) ∧ ∀ x, (∃ y, y = k * (x - 4))) →
  (∃ p : ℝ × ℝ, p = (2, 1)) →
  (∃ q : ℝ × ℝ, q = (0, 2)) →
  True := 
by sorry

end line_symmetric_fixed_point_l187_187938


namespace no_nontrivial_solutions_in_integers_l187_187328

theorem no_nontrivial_solutions_in_integers (a b c n : ℤ) 
  (h : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
  by
    sorry

end no_nontrivial_solutions_in_integers_l187_187328


namespace FGH_supermarkets_total_l187_187080

theorem FGH_supermarkets_total (US Canada : ℕ) 
  (h1 : US = 49) 
  (h2 : US = Canada + 14) : 
  US + Canada = 84 := 
by 
  sorry

end FGH_supermarkets_total_l187_187080


namespace jogging_friends_probability_l187_187382

theorem jogging_friends_probability
  (n p q r : ℝ)
  (h₀ : 1 > 0) -- Positive integers condition
  (h₁ : n = p - q * Real.sqrt r)
  (h₂ : ∀ prime, ¬ (r ∣ prime ^ 2)) -- r is not divisible by the square of any prime
  (h₃ : (60 - n)^2 = 1800) -- Derived from 50% meeting probability
  (h₄ : p = 60) -- Identified values from solution
  (h₅ : q = 30)
  (h₆ : r = 2) : 
  p + q + r = 92 :=
by
  sorry

end jogging_friends_probability_l187_187382


namespace max_non_overlapping_areas_l187_187144

theorem max_non_overlapping_areas (n : ℕ) (h : 0 < n) :
  ∃ k : ℕ, k = 4 * n + 1 :=
sorry

end max_non_overlapping_areas_l187_187144


namespace empty_with_three_pumps_in_12_minutes_l187_187498

-- Define the conditions
def conditions (a b x : ℝ) : Prop :=
  x = a + b ∧ 2 * x = 3 * a + b

-- Define the main theorem to prove
theorem empty_with_three_pumps_in_12_minutes (a b x : ℝ) (h : conditions a b x) : 
  (3 * (1 / 5) * x = a + (1 / 5) * b) ∧ ((1 / 5) * 60 = 12) := 
by
  -- Use the given conditions in the proof.
  sorry

end empty_with_three_pumps_in_12_minutes_l187_187498


namespace clara_cookies_l187_187723

theorem clara_cookies (x : ℕ) :
  50 * 12 + x * 20 + 70 * 16 = 3320 → x = 80 :=
by
  sorry

end clara_cookies_l187_187723


namespace count_valid_n_l187_187215

theorem count_valid_n:
  ( ∃ f: ℕ → ℕ, ∀ n, (0 < n ∧ n < 2012 → 7 ∣ (2^n - n^2) ↔ 7 ∣ (f n)) ∧ f 2012 = 576) → 
  ∃ valid_n_count: ℕ, valid_n_count = 576 := 
sorry

end count_valid_n_l187_187215


namespace a_congruent_b_mod_1008_l187_187880

theorem a_congruent_b_mod_1008 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b - b^a = 1008) : a ≡ b [MOD 1008] :=
by
  sorry

end a_congruent_b_mod_1008_l187_187880


namespace gcd_of_2475_and_7350_is_225_l187_187975

-- Definitions and conditions based on the factorization of the given numbers
def factor_2475 := (5^2 * 3^2 * 11)
def factor_7350 := (2 * 3^2 * 5^2 * 7)

-- Proof problem: showing the GCD of 2475 and 7350 is 225
theorem gcd_of_2475_and_7350_is_225 : Nat.gcd 2475 7350 = 225 :=
by
  -- Formal proof would go here
  sorry

end gcd_of_2475_and_7350_is_225_l187_187975


namespace power_of_product_l187_187510

variable (x y: ℝ)

theorem power_of_product :
  (-2 * x * y^3)^2 = 4 * x^2 * y^6 := 
by
  sorry

end power_of_product_l187_187510


namespace prob_diff_fruit_correct_l187_187130

noncomputable def prob_same_all_apple : ℝ := (0.4)^3
noncomputable def prob_same_all_orange : ℝ := (0.3)^3
noncomputable def prob_same_all_banana : ℝ := (0.2)^3
noncomputable def prob_same_all_grape : ℝ := (0.1)^3

noncomputable def prob_same_fruit_all_day : ℝ := 
  prob_same_all_apple + prob_same_all_orange + prob_same_all_banana + prob_same_all_grape

noncomputable def prob_diff_fruit (prob_same : ℝ) : ℝ := 1 - prob_same

theorem prob_diff_fruit_correct :
  prob_diff_fruit prob_same_fruit_all_day = 0.9 :=
by
  sorry

end prob_diff_fruit_correct_l187_187130


namespace vertex_of_parabola_is_correct_l187_187645

theorem vertex_of_parabola_is_correct :
  ∀ x y : ℝ, y = -5 * (x + 2) ^ 2 - 6 → (x = -2 ∧ y = -6) :=
by
  sorry

end vertex_of_parabola_is_correct_l187_187645


namespace factorize_difference_of_squares_l187_187447

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 81 = (a + 9) * (a - 9) :=
by
  sorry

end factorize_difference_of_squares_l187_187447


namespace non_athletic_parents_l187_187826

-- Define the conditions
variables (total_students athletic_dads athletic_moms both_athletic : ℕ)

-- Assume the given conditions
axiom h1 : total_students = 45
axiom h2 : athletic_dads = 17
axiom h3 : athletic_moms = 20
axiom h4 : both_athletic = 11

-- Statement to be proven
theorem non_athletic_parents : total_students - (athletic_dads - both_athletic + athletic_moms - both_athletic + both_athletic) = 19 :=
by {
  -- We intentionally skip the proof here
  sorry
}

end non_athletic_parents_l187_187826
