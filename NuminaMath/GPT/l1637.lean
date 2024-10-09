import Mathlib

namespace determine_a_l1637_163743

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem determine_a :
  {a : ℝ | 0 < a ∧ (f (a + 1) ≤ f (2 * a^2))} = {a : ℝ | 1 ≤ a ∧ a ≤ Real.sqrt 6 / 2 } :=
by
  sorry

end determine_a_l1637_163743


namespace find_principal_amount_l1637_163762

theorem find_principal_amount (A R T : ℝ) (P : ℝ) : 
  A = 1680 → R = 0.05 → T = 2.4 → 1.12 * P = 1680 → P = 1500 :=
by
  intros hA hR hT hEq
  sorry

end find_principal_amount_l1637_163762


namespace triangle_perimeter_l1637_163769

theorem triangle_perimeter (A r p : ℝ) (hA : A = 75) (hr : r = 2.5) :
  A = r * (p / 2) → p = 60 := by
  intros
  sorry

end triangle_perimeter_l1637_163769


namespace triangle_area_l1637_163746

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (3, -4)

-- State that the area of the triangle is 12.5 square units
theorem triangle_area :
  let base := 6 - 1
  let height := 1 - -4
  (1 / 2) * base * height = 12.5 := by
  sorry

end triangle_area_l1637_163746


namespace correct_answer_l1637_163758

theorem correct_answer (a b c : ℝ) : a - (b + c) = a - b - c :=
by sorry

end correct_answer_l1637_163758


namespace solve_for_x_l1637_163738

theorem solve_for_x (x : ℝ) (h : 3 + 5 * x = 28) : x = 5 :=
by {
  sorry
}

end solve_for_x_l1637_163738


namespace a_2020_equality_l1637_163780

variables (n : ℤ)

def cube (x : ℤ) : ℤ := x * x * x

lemma a_six_n (n : ℤ) :
  cube (n + 1) + cube (n - 1) + cube (-n) + cube (-n) = 6 * n :=
sorry

lemma a_six_n_plus_one (n : ℤ) :
  cube (n + 1) + cube (n - 1) + cube (-n) + cube (-n) + 1 = 6 * n + 1 :=
sorry

lemma a_six_n_minus_one (n : ℤ) :
  cube (n + 1) + cube (n - 1) + cube (-n) + cube (-n) - 1 = 6 * n - 1 :=
sorry

lemma a_six_n_plus_two (n : ℤ) :
  cube n + cube (n - 2) + cube (-n + 1) + cube (-n + 1) + 8 = 6 * n + 2 :=
sorry

lemma a_six_n_minus_two (n : ℤ) :
  cube (n + 2) + cube n + cube (-n - 1) + cube (-n - 1) + (-8) = 6 * n - 2 :=
sorry

lemma a_six_n_plus_three (n : ℤ) :
  cube (n - 3) + cube (n - 5) + cube (-n + 4) + cube (-n + 4) + 27 = 6 * n + 3 :=
sorry

theorem a_2020_equality :
  2020 = cube 339 + cube 337 + cube (-338) + cube (-338) + cube (-2) :=
sorry

end a_2020_equality_l1637_163780


namespace part_I_monotonicity_part_II_value_a_l1637_163787

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (x - 1)

def is_monotonic_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x < f y

def is_monotonic_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y < f x

theorem part_I_monotonicity :
  (is_monotonic_increasing f {x | 2 < x}) ∧
  ((is_monotonic_decreasing f {x | x < 1}) ∧ (is_monotonic_decreasing f {x | 1 < x ∧ x < 2})) :=
by
  sorry

theorem part_II_value_a (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → (Real.exp x * (x - 2)) / ((x - 1)^2) ≥ a * (Real.exp x / (x - 1))) → a ∈ Set.Iic 0 :=
by
  sorry

end part_I_monotonicity_part_II_value_a_l1637_163787


namespace union_A_B_l1637_163715

noncomputable def U := Set.univ ℝ

def A : Set ℝ := {x | x^2 - x - 2 = 0}

def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = x + 3}

theorem union_A_B : A ∪ B = { -1, 2, 5 } :=
by
  sorry

end union_A_B_l1637_163715


namespace simplify_pow_prod_eq_l1637_163721

noncomputable def simplify_pow_prod : ℝ :=
  (256:ℝ)^(1/4) * (625:ℝ)^(1/2)

theorem simplify_pow_prod_eq :
  simplify_pow_prod = 100 := by
  sorry

end simplify_pow_prod_eq_l1637_163721


namespace diminished_gcd_equals_100_l1637_163708

theorem diminished_gcd_equals_100 : Nat.gcd 7800 360 - 20 = 100 := by
  sorry

end diminished_gcd_equals_100_l1637_163708


namespace find_a_value_l1637_163710

noncomputable def a : ℝ := (384:ℝ)^(1/7)

variables (a b c : ℝ)
variables (h1 : a^2 / b = 2) (h2 : b^2 / c = 4) (h3 : c^2 / a = 6)

theorem find_a_value : a = 384^(1/7) :=
by
  sorry

end find_a_value_l1637_163710


namespace exists_indices_l1637_163719

-- Define the sequence condition
def is_sequence_of_all_positive_integers (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, ∃ m : ℕ, a m = n) ∧ (∀ n m1 m2 : ℕ, a m1 = n ∧ a m2 = n → m1 = m2)

-- Main theorem statement
theorem exists_indices 
  (a : ℕ → ℕ) 
  (h : is_sequence_of_all_positive_integers a) :
  ∃ (ℓ m : ℕ), 1 < ℓ ∧ ℓ < m ∧ (a 0 + a m = 2 * a ℓ) :=
by
  sorry

end exists_indices_l1637_163719


namespace average_age_l1637_163705

def proportion (x y z : ℕ) : Prop :=  y / x = 3 ∧ z / x = 4

theorem average_age (A B C : ℕ) 
    (h1 : proportion 2 6 8)
    (h2 : A = 15)
    (h3 : B = 45)
    (h4 : C = 60) :
    (A + B + C) / 3 = 40 := 
    by
    sorry

end average_age_l1637_163705


namespace highest_score_is_96_l1637_163709

theorem highest_score_is_96 :
  let standard_score := 85
  let deviations := [-9, -4, 11, -7, 0]
  let actual_scores := deviations.map (λ x => standard_score + x)
  actual_scores.maximum = 96 :=
by
  sorry

end highest_score_is_96_l1637_163709


namespace find_f_l1637_163782

theorem find_f {f : ℝ → ℝ} (h : ∀ x, f (1/x) = x / (1 - x)) : ∀ x, f x = 1 / (x - 1) :=
by
  sorry

end find_f_l1637_163782


namespace probability_of_three_heads_in_eight_tosses_l1637_163723

noncomputable def coin_toss_probability : ℚ :=
  let total_outcomes := 2^8
  let favorable_outcomes := Nat.choose 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_three_heads_in_eight_tosses : coin_toss_probability = 7 / 32 :=
  by
  sorry

end probability_of_three_heads_in_eight_tosses_l1637_163723


namespace find_number_thought_of_l1637_163773

theorem find_number_thought_of :
  ∃ x : ℝ, (6 * x^2 - 10) / 3 + 15 = 95 ∧ x = 5 * Real.sqrt 15 / 3 :=
by
  sorry

end find_number_thought_of_l1637_163773


namespace average_members_remaining_l1637_163752

theorem average_members_remaining :
  let initial_members := [7, 8, 10, 13, 6, 10, 12, 9]
  let members_leaving := [1, 2, 1, 2, 1, 2, 1, 2]
  let remaining_members := List.map (λ (x, y) => x - y) (List.zip initial_members members_leaving)
  let total_remaining := List.foldl Nat.add 0 remaining_members
  let num_families := initial_members.length
  total_remaining / num_families = 63 / 8 := by
    sorry

end average_members_remaining_l1637_163752


namespace distance_between_cities_l1637_163776

theorem distance_between_cities:
    ∃ (x y : ℝ),
    (x = 135) ∧
    (y = 175) ∧
    (7 / 9 * x = 105) ∧
    (x + 7 / 9 * x + y = 415) ∧
    (x = 27 / 35 * y) :=
by
  sorry

end distance_between_cities_l1637_163776


namespace union_of_A_and_B_l1637_163796

-- Definitions of sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 4}
def B : Set ℝ := {x | -3 ≤ x ∧ x < 1}

-- The theorem we aim to prove
theorem union_of_A_and_B : A ∪ B = { x | -3 ≤ x ∧ x ≤ 4 } :=
sorry

end union_of_A_and_B_l1637_163796


namespace remaining_amount_is_1520_l1637_163717

noncomputable def totalAmountToBePaid (deposit : ℝ) (depositRate : ℝ) (taxRate : ℝ) (processingFee : ℝ) : ℝ :=
  let fullPrice := deposit / depositRate
  let salesTax := taxRate * fullPrice
  let totalAdditionalExpenses := salesTax + processingFee
  (fullPrice - deposit) + totalAdditionalExpenses

theorem remaining_amount_is_1520 :
  totalAmountToBePaid 140 0.10 0.15 50 = 1520 := by
  sorry

end remaining_amount_is_1520_l1637_163717


namespace hyperbola_asymptotes_l1637_163786

theorem hyperbola_asymptotes (a b c : ℝ) (h : a > 0) (h_b_gt_0: b > 0) 
  (eqn1 : b = 2 * Real.sqrt 2 * a)
  (focal_distance : 2 * a = (2 * c)/3)
  (focal_length : c = 3 * a) : 
  (∀ x : ℝ, ∀ y : ℝ, (y = (2 * Real.sqrt 2) * x) ∨ (y = -(2 * Real.sqrt 2) * x)) := by
  sorry

end hyperbola_asymptotes_l1637_163786


namespace probability_Q_within_three_units_of_origin_l1637_163766

noncomputable def probability_within_three_units_of_origin :=
  let radius := 3
  let square_side := 10
  let circle_area := Real.pi * radius^2
  let square_area := square_side^2
  circle_area / square_area

theorem probability_Q_within_three_units_of_origin :
  probability_within_three_units_of_origin = 9 * Real.pi / 100 :=
by
  -- Since this proof is not required, we skip it with sorry.
  sorry

end probability_Q_within_three_units_of_origin_l1637_163766


namespace width_of_rectangle_l1637_163725

theorem width_of_rectangle (w l : ℝ) (h1 : l = 2 * w) (h2 : l * w = 1) : w = Real.sqrt 2 / 2 :=
sorry

end width_of_rectangle_l1637_163725


namespace find_n_l1637_163737

theorem find_n 
  (n : ℕ) 
  (b : ℕ → ℝ)
  (h₀ : b 0 = 28)
  (h₁ : b 1 = 81)
  (hn : b n = 0)
  (h_rec : ∀ j : ℕ, 1 ≤ j → j < n → b (j+1) = b (j-1) - 5 / b j)
  : n = 455 := 
sorry

end find_n_l1637_163737


namespace units_digit_of_large_powers_l1637_163724

theorem units_digit_of_large_powers : 
  (2^1007 * 6^1008 * 14^1009) % 10 = 2 := 
  sorry

end units_digit_of_large_powers_l1637_163724


namespace samantha_routes_l1637_163790

-- Definitions of the conditions
def blocks_west_to_sw_corner := 3
def blocks_south_to_sw_corner := 2
def blocks_east_to_school := 4
def blocks_north_to_school := 3
def ways_house_to_sw_corner : ℕ := Nat.choose (blocks_west_to_sw_corner + blocks_south_to_sw_corner) blocks_south_to_sw_corner
def ways_through_park : ℕ := 2
def ways_ne_corner_to_school : ℕ := Nat.choose (blocks_east_to_school + blocks_north_to_school) blocks_north_to_school

-- The proof statement
theorem samantha_routes : (ways_house_to_sw_corner * ways_through_park * ways_ne_corner_to_school) = 700 :=
by
  -- Using "sorry" as a placeholder for the actual proof
  sorry

end samantha_routes_l1637_163790


namespace probability_all_quitters_from_same_tribe_l1637_163748

noncomputable def total_ways_to_choose_quitters : ℕ := Nat.choose 18 3

noncomputable def ways_all_from_tribe (n : ℕ) : ℕ := Nat.choose n 3

noncomputable def combined_ways_same_tribe : ℕ :=
  ways_all_from_tribe 9 + ways_all_from_tribe 9

noncomputable def probability_same_tribe (total : ℕ) (same_tribe : ℕ) : ℚ :=
  same_tribe / total

theorem probability_all_quitters_from_same_tribe :
  probability_same_tribe total_ways_to_choose_quitters combined_ways_same_tribe = 7 / 34 :=
by
  sorry

end probability_all_quitters_from_same_tribe_l1637_163748


namespace reciprocal_relationship_l1637_163785

theorem reciprocal_relationship (a b : ℚ)
  (h1 : a = (-7 / 8) / (7 / 4 - 7 / 8 - 7 / 12))
  (h2 : b = (7 / 4 - 7 / 8 - 7 / 12) / (-7 / 8)) :
  a = - 1 / b :=
by sorry

end reciprocal_relationship_l1637_163785


namespace speed_of_stream_l1637_163765

-- Definitions based on conditions
def boat_speed_still_water : ℕ := 24
def travel_time : ℕ := 4
def downstream_distance : ℕ := 112

-- Theorem statement
theorem speed_of_stream : 
  ∀ (v : ℕ), downstream_distance = travel_time * (boat_speed_still_water + v) → v = 4 :=
by
  intros v h
  -- Proof omitted
  sorry

end speed_of_stream_l1637_163765


namespace points_earned_l1637_163735

-- Definitions from conditions
def points_per_enemy : ℕ := 8
def total_enemies : ℕ := 7
def enemies_not_destroyed : ℕ := 2

-- The proof statement
theorem points_earned :
  points_per_enemy * (total_enemies - enemies_not_destroyed) = 40 := 
by
  sorry

end points_earned_l1637_163735


namespace sum_of_digits_of_gcd_l1637_163764

def gcd_of_differences : ℕ := Int.gcd (Int.gcd 3360 2240) 5600

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem sum_of_digits_of_gcd :
  sum_of_digits gcd_of_differences = 4 :=
by
  sorry

end sum_of_digits_of_gcd_l1637_163764


namespace ones_digit_of_11_pow_46_l1637_163798

theorem ones_digit_of_11_pow_46 : (11 ^ 46) % 10 = 1 :=
by sorry

end ones_digit_of_11_pow_46_l1637_163798


namespace prob_a_wins_match_l1637_163733

-- Define the probability of A winning a single game
def prob_win_a_single_game : ℚ := 1 / 3

-- Define the probability of A winning two consecutive games
def prob_win_a_two_consec_games : ℚ := prob_win_a_single_game * prob_win_a_single_game

-- Define the probability of A winning two games with one loss in between
def prob_win_a_two_wins_one_loss_first : ℚ := prob_win_a_single_game * (1 - prob_win_a_single_game) * prob_win_a_single_game
def prob_win_a_two_wins_one_loss_second : ℚ := (1 - prob_win_a_single_game) * prob_win_a_single_game * prob_win_a_single_game

-- Define the total probability of A winning the match
def prob_a_winning_match : ℚ := prob_win_a_two_consec_games + prob_win_a_two_wins_one_loss_first + prob_win_a_two_wins_one_loss_second

-- The theorem to be proved
theorem prob_a_wins_match : prob_a_winning_match = 7 / 27 :=
by sorry

end prob_a_wins_match_l1637_163733


namespace original_number_of_men_l1637_163740

theorem original_number_of_men (x : ℕ) (h1 : x * 50 = (x - 10) * 60) : x = 60 :=
by
  sorry

end original_number_of_men_l1637_163740


namespace remainder_when_sum_divided_by_15_l1637_163731

theorem remainder_when_sum_divided_by_15 (a b c : ℕ) 
  (h1 : a % 15 = 11) 
  (h2 : b % 15 = 12) 
  (h3 : c % 15 = 13) : 
  (a + b + c) % 15 = 6 :=
  sorry

end remainder_when_sum_divided_by_15_l1637_163731


namespace total_artworks_created_l1637_163747

theorem total_artworks_created
  (students_group1 : ℕ := 24) (students_group2 : ℕ := 12)
  (kits_total : ℕ := 48)
  (kits_per_3_students : ℕ := 3) (kits_per_2_students : ℕ := 2)
  (artwork_types : ℕ := 3)
  (paintings_group1_1 : ℕ := 12 * 2) (drawings_group1_1 : ℕ := 12 * 4) (sculptures_group1_1 : ℕ := 12 * 1)
  (paintings_group1_2 : ℕ := 12 * 1) (drawings_group1_2 : ℕ := 12 * 5) (sculptures_group1_2 : ℕ := 12 * 3)
  (paintings_group2_1 : ℕ := 4 * 3) (drawings_group2_1 : ℕ := 4 * 6) (sculptures_group2_1 : ℕ := 4 * 3)
  (paintings_group2_2 : ℕ := 8 * 4) (drawings_group2_2 : ℕ := 8 * 7) (sculptures_group2_2 : ℕ := 8 * 1)
  : (paintings_group1_1 + paintings_group1_2 + paintings_group2_1 + paintings_group2_2) +
    (drawings_group1_1 + drawings_group1_2 + drawings_group2_1 + drawings_group2_2) +
    (sculptures_group1_1 + sculptures_group1_2 + sculptures_group2_1 + sculptures_group2_2) = 336 :=
by sorry

end total_artworks_created_l1637_163747


namespace expected_value_coins_heads_l1637_163793

noncomputable def expected_value_cents : ℝ :=
  let values := [1, 5, 10, 25, 50, 100]
  let probability_heads := 1 / 2
  probability_heads * (values.sum : ℝ)

theorem expected_value_coins_heads : expected_value_cents = 95.5 := by
  sorry

end expected_value_coins_heads_l1637_163793


namespace distribution_ways_l1637_163775

theorem distribution_ways (books students : ℕ) (h_books : books = 6) (h_students : students = 6) :
  ∃ ways : ℕ, ways = 6 * 5^6 ∧ ways = 93750 :=
by
  sorry

end distribution_ways_l1637_163775


namespace xiaoming_department_store_profit_l1637_163726

theorem xiaoming_department_store_profit:
  let P₁ := 40000   -- average monthly profit in Q1
  let L₂ := -15000  -- average monthly loss in Q2
  let L₃ := -18000  -- average monthly loss in Q3
  let P₄ := 32000   -- average monthly profit in Q4
  let P_total := (P₁ * 3 + L₂ * 3 + L₃ * 3 + P₄ * 3)
  P_total = 117000 := by
  sorry

end xiaoming_department_store_profit_l1637_163726


namespace amanda_tickets_l1637_163720

theorem amanda_tickets (F : ℕ) (h : 4 * F + 32 + 28 = 80) : F = 5 :=
by
  sorry

end amanda_tickets_l1637_163720


namespace solve_for_y_l1637_163734

theorem solve_for_y (y : ℤ) : (4 + y) / (6 + y) = (2 + y) / (3 + y) → y = 0 := by 
  sorry

end solve_for_y_l1637_163734


namespace honor_students_count_l1637_163716

noncomputable def G : ℕ := 13
noncomputable def B : ℕ := 11
def E_G : ℕ := 3
def E_B : ℕ := 4

theorem honor_students_count (h1 : G + B < 30) 
    (h2 : (E_G : ℚ) / G = 3 / 13) 
    (h3 : (E_B : ℚ) / B = 4 / 11) :
    E_G + E_B = 7 := 
sorry

end honor_students_count_l1637_163716


namespace function_increasing_intervals_l1637_163704

theorem function_increasing_intervals (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x < f (x + 1)) :
  (∃ x : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, abs (y - x) < δ → f y > f x) ∨ 
  (∀ x : ℝ, ∃ ε > 0, ∀ δ > 0, ∃ y : ℝ, abs (y - x) < δ ∧ f y < f x) :=
sorry

end function_increasing_intervals_l1637_163704


namespace max_min_values_of_x_l1637_163759

theorem max_min_values_of_x (x y z : ℝ) (h1 : x + y + z = 0) (h2 : (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ 2) :
  -2/3 ≤ x ∧ x ≤ 2/3 :=
sorry

end max_min_values_of_x_l1637_163759


namespace translation_vector_coords_l1637_163772

-- Definitions according to the given conditions
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 1
def translated_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- Statement that we need to prove
theorem translation_vector_coords :
  ∃ (a b : ℝ), 
  (∀ x y : ℝ, original_circle x y ↔ translated_circle (x - a) (y - b)) ∧
  (a, b) = (-1, 2) := 
sorry

end translation_vector_coords_l1637_163772


namespace range_of_a_l1637_163751

theorem range_of_a (A B C : Set ℝ) (a : ℝ) :
  A = { x | -1 < x ∧ x < 4 } →
  B = { x | -5 < x ∧ x < (3 / 2) } →
  C = { x | (1 - 2 * a) < x ∧ x < (2 * a) } →
  (C ⊆ (A ∩ B)) →
  a ≤ (3 / 4) :=
by
  intros hA hB hC hSubset
  sorry

end range_of_a_l1637_163751


namespace problem_solution_l1637_163779

-- Define the operation otimes
def otimes (x y : ℚ) : ℚ := (x * y) / (x + y / 3)

-- Define the specific values x and y
def x : ℚ := 4
def y : ℚ := 3/2 -- 1.5 in fraction form

-- Prove the mathematical statement
theorem problem_solution : (0.36 : ℚ) * (otimes x y) = 12 / 25 := by
  sorry

end problem_solution_l1637_163779


namespace gcd_linear_combination_l1637_163763

theorem gcd_linear_combination (a b : ℤ) : 
  Int.gcd (5 * a + 3 * b) (13 * a + 8 * b) = Int.gcd a b := 
sorry

end gcd_linear_combination_l1637_163763


namespace sequence_values_induction_proof_l1637_163732

def seq (a : ℕ → ℤ) := a 1 = 3 ∧ ∀ n : ℕ, a (n + 1) = a n ^ 2 - 2 * n * a n + 2

theorem sequence_values (a : ℕ → ℤ) (h : seq a) :
  a 2 = 5 ∧ a 3 = 7 ∧ a 4 = 9 :=
sorry

theorem induction_proof (a : ℕ → ℤ) (h : seq a) :
  ∀ n : ℕ, a n = 2 * n + 1 :=
sorry

end sequence_values_induction_proof_l1637_163732


namespace geometric_seq_prod_l1637_163784

-- Conditions: Geometric sequence and given value of a_1 * a_7 * a_13
variables {a : ℕ → ℝ}
variable (r : ℝ)

-- Definition of a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- The proof problem
theorem geometric_seq_prod (h_geo : geometric_sequence a r) (h_prod : a 1 * a 7 * a 13 = 8) :
  a 3 * a 11 = 4 :=
sorry

end geometric_seq_prod_l1637_163784


namespace cd_value_l1637_163744

theorem cd_value (a b c d : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (ab ac bd : ℝ) 
  (h_ab : ab = 2) (h_ac : ac = 5) (h_bd : bd = 6) :
  ∃ (cd : ℝ), cd = 3 :=
by sorry

end cd_value_l1637_163744


namespace science_book_pages_l1637_163707

theorem science_book_pages {history_pages novel_pages science_pages: ℕ} (h1: novel_pages = history_pages / 2) (h2: science_pages = 4 * novel_pages) (h3: history_pages = 300):
  science_pages = 600 :=
by
  sorry

end science_book_pages_l1637_163707


namespace minimum_value_of_x_plus_2y_l1637_163768

-- Definitions for the problem conditions
def isPositive (z : ℝ) : Prop := z > 0

def condition (x y : ℝ) : Prop := 
  isPositive x ∧ isPositive y ∧ (x + 2*y + 2*x*y = 8) 

-- Statement of the problem
theorem minimum_value_of_x_plus_2y (x y : ℝ) (h : condition x y) : x + 2 * y ≥ 4 :=
sorry

end minimum_value_of_x_plus_2y_l1637_163768


namespace manuscript_typing_cost_l1637_163741

theorem manuscript_typing_cost 
  (pages_total : ℕ) (pages_first_time : ℕ) (pages_revised_once : ℕ)
  (pages_revised_twice : ℕ) (rate_first_time : ℕ) (rate_revised : ℕ) 
  (cost_total : ℕ) :
  pages_total = 100 →
  pages_first_time = pages_total →
  pages_revised_once = 35 →
  pages_revised_twice = 15 →
  rate_first_time = 6 →
  rate_revised = 4 →
  cost_total = (pages_first_time * rate_first_time) +
              (pages_revised_once * rate_revised) +
              (pages_revised_twice * rate_revised * 2) →
  cost_total = 860 :=
by
  intros htot hfirst hrev1 hrev2 hr1 hr2 hcost
  sorry

end manuscript_typing_cost_l1637_163741


namespace min_distance_curves_l1637_163795

theorem min_distance_curves (P Q : ℝ × ℝ) (h1 : P.2 = (1/3) * Real.exp P.1) (h2 : Q.2 = Real.log (3 * Q.1)) :
  ∃ d : ℝ, d = Real.sqrt 2 * (Real.log 3 - 1) ∧ d = |P.1 - Q.1| := sorry

end min_distance_curves_l1637_163795


namespace general_formula_minimum_n_exists_l1637_163727

noncomputable def a_n (n : ℕ) : ℝ := 3 * (-2)^(n-1)
noncomputable def S_n (n : ℕ) : ℝ := 1 - (-2)^n

theorem general_formula (n : ℕ) : a_n n = 3 * (-2)^(n-1) :=
by sorry

theorem minimum_n_exists :
  (∃ n : ℕ, S_n n > 2016) ∧ (∀ m : ℕ, S_n m > 2016 → 11 ≤ m) :=
by sorry

end general_formula_minimum_n_exists_l1637_163727


namespace simplify_and_evaluate_l1637_163771

noncomputable def a := 3

theorem simplify_and_evaluate : (a^2 / (a + 1) - 1 / (a + 1)) = 2 := by
  sorry

end simplify_and_evaluate_l1637_163771


namespace first_discount_percentage_l1637_163789

noncomputable def saree_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initial_price * (1 - discount1 / 100) * (1 - discount2 / 100)

theorem first_discount_percentage (x : ℝ) : saree_price 400 x 20 = 240 → x = 25 :=
by sorry

end first_discount_percentage_l1637_163789


namespace schoolchildren_number_l1637_163754

theorem schoolchildren_number (n m S : ℕ) 
  (h1 : S = 22 * n + 3)
  (h2 : S = (n - 1) * m)
  (h3 : n ≤ 18)
  (h4 : m ≤ 36) : 
  S = 135 := 
sorry

end schoolchildren_number_l1637_163754


namespace art_collection_total_area_l1637_163701

-- Define the dimensions and quantities of the paintings
def square_painting_side := 6
def small_painting_width := 2
def small_painting_height := 3
def large_painting_width := 10
def large_painting_height := 15

def num_square_paintings := 3
def num_small_paintings := 4
def num_large_paintings := 1

-- Define areas of individual paintings
def square_painting_area := square_painting_side * square_painting_side
def small_painting_area := small_painting_width * small_painting_height
def large_painting_area := large_painting_width * large_painting_height

-- Define the total area calculation
def total_area :=
  num_square_paintings * square_painting_area +
  num_small_paintings * small_painting_area +
  num_large_paintings * large_painting_area

-- The theorem statement
theorem art_collection_total_area : total_area = 282 := by
  sorry

end art_collection_total_area_l1637_163701


namespace monotonicity_of_f_range_of_a_l1637_163799

open Real

noncomputable def f (x a : ℝ) : ℝ := a * exp x + 2 * exp (-x) + (a - 2) * x

noncomputable def f_prime (x a : ℝ) : ℝ := (a * exp (2 * x) + (a - 2) * exp x - 2) / exp x

theorem monotonicity_of_f (a : ℝ) : 
  (∀ x : ℝ, f_prime x a ≤ 0) ↔ (a ≤ 0) :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, f x a ≥ (a + 2) * cos x) ↔ (2 ≤ a) :=
sorry

end monotonicity_of_f_range_of_a_l1637_163799


namespace unique_a_b_l1637_163749

-- Define the properties of the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * a * x + b else 7 - 2 * x

-- The function satisfies f(f(x)) = x for all x in its domain
theorem unique_a_b (a b : ℝ) (h : ∀ x : ℝ, f a b (f a b x) = x) : a + b = 13 / 4 :=
sorry

end unique_a_b_l1637_163749


namespace evaluate_g_neg_1_l1637_163757

noncomputable def g (x : ℝ) : ℝ := -2 * x^2 + 5 * x - 7

theorem evaluate_g_neg_1 : g (-1) = -14 := 
by
  sorry

end evaluate_g_neg_1_l1637_163757


namespace persimmons_picked_l1637_163711

theorem persimmons_picked : 
  ∀ (J H : ℕ), (4 * J = H - 3) → (H = 35) → (J = 8) := 
by
  intros J H hJ hH
  sorry

end persimmons_picked_l1637_163711


namespace quad_func_minimum_l1637_163756

def quad_func (x : ℝ) : ℝ := x^2 - 8 * x + 5

theorem quad_func_minimum : ∀ x : ℝ, quad_func x ≥ -11 ∧ quad_func 4 = -11 :=
by
  sorry

end quad_func_minimum_l1637_163756


namespace line_intersects_circle_l1637_163797

variable {a x_0 y_0 : ℝ}

theorem line_intersects_circle (h1: x_0^2 + y_0^2 > a^2) (h2: a > 0) : 
  ∃ (p : ℝ × ℝ), (p.1 ^ 2 + p.2 ^ 2 = a ^ 2) ∧ (x_0 * p.1 + y_0 * p.2 = a ^ 2) :=
sorry

end line_intersects_circle_l1637_163797


namespace cody_candy_total_l1637_163703

theorem cody_candy_total
  (C_c : ℕ) (C_m : ℕ) (P_b : ℕ)
  (h1 : C_c = 7) (h2 : C_m = 3) (h3 : P_b = 8) :
  (C_c + C_m) * P_b = 80 :=
by
  sorry

end cody_candy_total_l1637_163703


namespace mark_deposit_is_88_l1637_163739

-- Definitions according to the conditions
def markDeposit := 88
def bryanDeposit (m : ℕ) := 5 * m - 40

-- The theorem we need to prove
theorem mark_deposit_is_88 : markDeposit = 88 := 
by 
  -- Since the condition states Mark deposited $88,
  -- this is trivially true.
  sorry

end mark_deposit_is_88_l1637_163739


namespace solution_set_of_inequality_l1637_163712

theorem solution_set_of_inequality :
  { x : ℝ | (x - 5) / (x + 1) ≤ 0 } = { x : ℝ | -1 < x ∧ x ≤ 5 } :=
sorry

end solution_set_of_inequality_l1637_163712


namespace find_x_value_l1637_163713

theorem find_x_value :
  ∀ (x : ℝ), 0.3 + 0.1 + 0.4 + x = 1 → x = 0.2 :=
by
  intros x h
  sorry

end find_x_value_l1637_163713


namespace value_of_a_l1637_163774

noncomputable def A : Set ℝ := {x | x^2 - x - 2 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 5}

theorem value_of_a (a : ℝ) (h : A ⊆ B a) : -3 ≤ a ∧ a ≤ -1 :=
by
  sorry

end value_of_a_l1637_163774


namespace dark_squares_more_than_light_l1637_163753

/--
A 9 × 9 board is composed of alternating dark and light squares, with the upper-left square being dark.
Prove that there is exactly 1 more dark square than light square.
-/
theorem dark_squares_more_than_light :
  let board_size := 9
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 4 * 5 + 5 * 4
  dark_squares - light_squares = 1 :=
by
  let board_size := 9
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 4 * 5 + 5 * 4
  show dark_squares - light_squares = 1
  sorry

end dark_squares_more_than_light_l1637_163753


namespace rational_solution_exists_l1637_163729

theorem rational_solution_exists (a b c : ℤ) (x₀ y₀ z₀ : ℤ) (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h₁ : a * x₀^2 + b * y₀^2 + c * z₀^2 = 0) (h₂ : x₀ ≠ 0 ∨ y₀ ≠ 0 ∨ z₀ ≠ 0) : 
  ∃ (x y z : ℚ), a * x^2 + b * y^2 + c * z^2 = 1 := 
sorry

end rational_solution_exists_l1637_163729


namespace person_next_to_Boris_arkady_galya_l1637_163770

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l1637_163770


namespace problem_part1_problem_part2_l1637_163777

variable (α : Real)
variable (h : Real.tan α = 1 / 2)

theorem problem_part1 : 
  (2 * Real.cos α - 3 * Real.sin α) / (3 * Real.cos α + 4 * Real.sin α) = 1 / 10 := sorry

theorem problem_part2 : 
  Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 11 / 5 := sorry

end problem_part1_problem_part2_l1637_163777


namespace sector_properties_l1637_163722

noncomputable def central_angle (l r : ℝ) : ℝ := l / r

noncomputable def sector_area (alpha r : ℝ) : ℝ := (1/2) * alpha * r^2

theorem sector_properties (l r : ℝ) (h_l : l = Real.pi) (h_r : r = 3) :
  central_angle l r = Real.pi / 3 ∧ sector_area (central_angle l r) r = 3 * Real.pi / 2 := 
  by
  sorry

end sector_properties_l1637_163722


namespace red_grapes_count_l1637_163760

-- Definitions of variables and conditions
variables (G R Ra B P : ℕ)
variables (cond1 : R = 3 * G + 7)
variables (cond2 : Ra = G - 5)
variables (cond3 : B = 4 * Ra)
variables (cond4 : P = (1 / 2) * B + 5)
variables (cond5 : G + R + Ra + B + P = 350)

-- Theorem statement
theorem red_grapes_count : R = 100 :=
by sorry

end red_grapes_count_l1637_163760


namespace maximum_fly_path_length_in_box_l1637_163718

theorem maximum_fly_path_length_in_box
  (length width height : ℝ)
  (h_length : length = 1)
  (h_width : width = 1)
  (h_height : height = 2) :
  ∃ l, l = (Real.sqrt 6 + 2 * Real.sqrt 5 + Real.sqrt 2 + 1) :=
by
  sorry

end maximum_fly_path_length_in_box_l1637_163718


namespace red_paint_intensity_l1637_163783

variable (I : ℝ) -- Intensity of the original paint
variable (P : ℝ) -- Volume of the original paint
variable (fraction_replaced : ℝ := 1) -- Fraction of original paint replaced
variable (new_intensity : ℝ := 20) -- New paint intensity
variable (replacement_intensity : ℝ := 20) -- Replacement paint intensity

theorem red_paint_intensity : new_intensity = replacement_intensity :=
by
  -- Placeholder for the actual proof
  sorry

end red_paint_intensity_l1637_163783


namespace max_sum_of_arithmetic_seq_l1637_163755

theorem max_sum_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h₁ : a 1 = 11) (h₂ : a 5 = -1) 
  (h₃ : ∀ n, a n = 14 - 3 * (n - 1)) 
  : ∀ n, (S n = (n * (a 1 + a n) / 2)) → max (S n) = 26 :=
sorry

end max_sum_of_arithmetic_seq_l1637_163755


namespace find_set_M_l1637_163702

variable (U M : Set ℕ)
variable [DecidableEq ℕ]

-- Universel set U is {1, 3, 5, 7}
def universal_set : Set ℕ := {1, 3, 5, 7}

-- define the complement C_U M
def complement (U M : Set ℕ) : Set ℕ := U \ M

-- M is the set to find such that complement of M in U is {5, 7}
theorem find_set_M (M : Set ℕ) (h : complement universal_set M = {5, 7}) : M = {1, 3} := by
  sorry

end find_set_M_l1637_163702


namespace total_letters_l1637_163706

theorem total_letters (brother_letters : ℕ) (greta_more_than_brother : ℕ) (mother_multiple : ℕ)
  (h_brother : brother_letters = 40)
  (h_greta : ∀ (brother_letters greta_letters : ℕ), greta_letters = brother_letters + greta_more_than_brother)
  (h_mother : ∀ (total_letters mother_letters : ℕ), mother_letters = mother_multiple * total_letters) :
  brother_letters + (brother_letters + greta_more_than_brother) + (mother_multiple * (brother_letters + (brother_letters + greta_more_than_brother))) = 270 :=
by
  sorry

end total_letters_l1637_163706


namespace score_difference_l1637_163794

theorem score_difference (chuck_score red_score : ℕ) (h1 : chuck_score = 95) (h2 : red_score = 76) : chuck_score - red_score = 19 := by
  sorry

end score_difference_l1637_163794


namespace dolphins_points_l1637_163736

variable (S D : ℕ)

theorem dolphins_points :
  (S + D = 36) ∧ (S = D + 12) → D = 12 :=
by
  sorry

end dolphins_points_l1637_163736


namespace compound_interest_is_correct_l1637_163700

noncomputable def compound_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

theorem compound_interest_is_correct :
  let P := 660 / (0.2 : ℝ)
  (compound_interest P 10 2) = 693 := 
by
  -- Definitions of simple_interest and compound_interest are used
  -- The problem conditions help us conclude
  let P := 660 / (0.2 : ℝ)
  have h1 : simple_interest P 10 2 = 660 := by sorry
  have h2 : compound_interest P 10 2 = 693 := by sorry
  exact h2

end compound_interest_is_correct_l1637_163700


namespace max_elephants_l1637_163778

def union_members : ℕ := 28
def non_union_members : ℕ := 37

/-- Given 28 union members and 37 non-union members, where elephants are distributed equally among
each group and each person initially receives at least one elephant, and considering 
the unique distribution constraint, the maximum number of elephants is 2072. -/
theorem max_elephants (n : ℕ) 
  (h1 : n % union_members = 0)
  (h2 : n % non_union_members = 0)
  (h3 : n ≥ union_members * non_union_members) :
  n = 2072 :=
by sorry

end max_elephants_l1637_163778


namespace find_x_l1637_163728

namespace IntegerProblem

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := 
by
  sorry

end IntegerProblem

end find_x_l1637_163728


namespace part1_part2_l1637_163761

variable {m n : ℤ}

theorem part1 (hm : |m| = 1) (hn : |n| = 4) (hprod : m * n < 0) : m + n = -3 ∨ m + n = 3 := sorry

theorem part2 (hm : |m| = 1) (hn : |n| = 4) : ∃ (k : ℤ), k = 5 ∧ ∀ x, x = m - n → x ≤ k := sorry

end part1_part2_l1637_163761


namespace fruit_seller_price_l1637_163788

theorem fruit_seller_price (C : ℝ) (h1 : 1.05 * C = 14.823529411764707) : 
  0.85 * C = 12 := 
sorry

end fruit_seller_price_l1637_163788


namespace total_cost_is_correct_l1637_163792

noncomputable def nights : ℕ := 3
noncomputable def cost_per_night : ℕ := 250
noncomputable def discount : ℕ := 100

theorem total_cost_is_correct :
  (nights * cost_per_night) - discount = 650 := by
sorry

end total_cost_is_correct_l1637_163792


namespace tetrahedron_cd_length_l1637_163791

theorem tetrahedron_cd_length (a b c d : Type) [MetricSpace a] [MetricSpace b] [MetricSpace c] [MetricSpace d] :
  let ab := 53
  let edge_lengths := [17, 23, 29, 39, 46, 53]
  ∃ cd, cd = 17 :=
by
  sorry

end tetrahedron_cd_length_l1637_163791


namespace sock_pairs_l1637_163781

theorem sock_pairs (n : ℕ) (h : ((2 * n) * (2 * n - 1)) / 2 = 90) : n = 10 :=
sorry

end sock_pairs_l1637_163781


namespace negation_of_universal_proposition_l1637_163730

def int_divisible_by_5 (n : ℤ) := ∃ k : ℤ, n = 5 * k
def int_odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℤ, int_divisible_by_5 n → int_odd n) ↔ (∃ n : ℤ, int_divisible_by_5 n ∧ ¬ int_odd n) :=
by
  sorry

end negation_of_universal_proposition_l1637_163730


namespace num_rectangles_in_5x5_grid_l1637_163745

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l1637_163745


namespace divides_n3_minus_7n_l1637_163714

theorem divides_n3_minus_7n (n : ℕ) : 6 ∣ n^3 - 7 * n := 
sorry

end divides_n3_minus_7n_l1637_163714


namespace count_valid_combinations_l1637_163742

-- Define the digits condition
def is_digit (d : ℕ) : Prop := d >= 0 ∧ d <= 9

-- Define the main proof statement
theorem count_valid_combinations (a b c: ℕ) (h1 : is_digit a)(h2 : is_digit b)(h3 : is_digit c) :
    (100 * a + 10 * b + c) + (100 * c + 10 * b + a) = 1069 → 
    ∃ (abc_combinations : ℕ), abc_combinations = 8 :=
by
  sorry

end count_valid_combinations_l1637_163742


namespace sum_of_remainders_l1637_163767

theorem sum_of_remainders 
  (a b c : ℕ) 
  (h1 : a % 53 = 37) 
  (h2 : b % 53 = 14) 
  (h3 : c % 53 = 7) : 
  (a + b + c) % 53 = 5 := 
by 
  sorry

end sum_of_remainders_l1637_163767


namespace tom_sleep_hours_l1637_163750

-- Define initial sleep hours and increase fraction
def initial_sleep_hours : ℕ := 6
def increase_fraction : ℚ := 1 / 3

-- Define the function to calculate increased sleep
def increased_sleep_hours (initial : ℕ) (fraction : ℚ) : ℚ :=
  initial * fraction

-- Define the function to calculate total sleep hours
def total_sleep_hours (initial : ℕ) (increased : ℚ) : ℚ :=
  initial + increased

-- Theorem stating Tom's total sleep hours per night after the increase
theorem tom_sleep_hours (initial : ℕ) (fraction : ℚ) (increased : ℚ) (total : ℚ) :
  initial = initial_sleep_hours →
  fraction = increase_fraction →
  increased = increased_sleep_hours initial fraction →
  total = total_sleep_hours initial increased →
  total = 8 :=
by
  intros h_init h_frac h_incr h_total
  rw [h_init, h_frac] at h_incr
  rw [h_init, h_incr] at h_total
  sorry

end tom_sleep_hours_l1637_163750
